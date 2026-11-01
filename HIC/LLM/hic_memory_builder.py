"""Build completed, historical HIC memories with resumable per-case checkpoints."""

import hashlib
import json
import math
import re
from pathlib import Path

from LLM import hic
from LLM.hic_qdrant_eval import (
    _atomic_write_json,
    _correction_text,
    _validated_forecast,
)


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_STAGES = ("summary", "forecast", "correction")
_M3_MODEL_ID = "BAAI/bge-m3"


def _finite_values(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric list")
    if length is not None and len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    try:
        normalized = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain only finite numbers") from None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must contain only finite numbers")
    return normalized


def _validate_cases(cases, dataset_name):
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("cases must be a non-empty list")
    validated = []
    seen_ids = set()
    expected_history_length = None
    expected_horizon = None
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise ValueError(f"case {index} must be a dictionary")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index} must have a non-empty case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if raw.get("dataset") not in (None, dataset_name):
            raise ValueError(f"case {case_id} belongs to a different dataset")
        series = _finite_values(raw.get("series"), f"series for {case_id}")
        truth = _finite_values(raw.get("truth"), f"truth for {case_id}")
        if expected_history_length is None:
            expected_history_length = len(series)
            expected_horizon = len(truth)
        if len(series) != expected_history_length or len(truth) != expected_horizon:
            raise ValueError("all memory cases must share history and horizon lengths")
        news = raw.get("news")
        if not isinstance(news, str):
            raise ValueError(f"news for {case_id} must be text")
        target_end = raw.get("target_end")
        if not isinstance(target_end, str) or not target_end.strip():
            raise ValueError(f"target_end for {case_id} must be present")
        covariates = raw.get("covariates") or {}
        if not isinstance(covariates, dict):
            raise ValueError(f"covariates for {case_id} must be a dictionary")
        normalized_covariates = {
            name: _finite_values(values, f"{name} history for {case_id}", length=len(series))
            for name, values in covariates.items()
        }
        normalized = {
            **raw,
            "series": series,
            "truth": truth,
            "news": news,
            "covariates": normalized_covariates,
        }
        validated.append(normalized)
    return validated


def _fingerprint(cases, *, summary_prompt_version="legacy"):
    stable_inputs = []
    for case in cases:
        stable_case = {
            "case_id": case["case_id"],
            "series": case["series"],
            "truth": case["truth"],
            "news": case["news"],
            "covariates": case["covariates"],
            "target_end": case["target_end"],
        }
        if "frequency" in case:
            stable_case["frequency"] = case["frequency"]
        if summary_prompt_version != "legacy":
            stable_case["target"] = case.get("target", "OT")
            stable_case["unit"] = case.get("unit", "unknown")
        if "group" in case:
            stable_case["group"] = case["group"]
        stable_inputs.append(stable_case)
    if summary_prompt_version != "legacy":
        fingerprint_payload = {
            "summary_prompt_version": summary_prompt_version,
            "cases": stable_inputs,
        }
    else:
        fingerprint_payload = stable_inputs
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_calls(state):
    return sum(
        stage not in state["cases"].get(case_id, {})
        for case_id in state["case_ids"]
        for stage in _STAGES
    )


def build_completed_memories(
    *,
    cases,
    template,
    store,
    embed_summary,
    call_json,
    dataset_name,
    dataset_description,
    run_id,
    checkpoint_path,
    max_calls,
    retry_pending=False,
    versioned_summary_prompt=False,
):
    """Generate and upsert each historical case only after its correction exists.

    Each case uses three DeepSeek logical calls: summary, initial forecast, then
    correction using the already-observed historical targets. BGE-M3 inference is
    injected and local; it is not counted as an API call.
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 letters, digits, '_' or '-' characters")
    if not isinstance(max_calls, int) or max_calls < 0:
        raise ValueError("max_calls must be a non-negative integer")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be non-empty")
    if not isinstance(dataset_description, str) or not dataset_description.strip():
        raise ValueError("dataset_description must be non-empty")
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty dictionary")
    if not callable(embed_summary) or not callable(call_json):
        raise ValueError("embed_summary and call_json must be callable")
    if getattr(store, "model_id", _M3_MODEL_ID) != _M3_MODEL_ID:
        raise ValueError("HIC training memory requires the BGE-M3 Qdrant collection")
    if not isinstance(versioned_summary_prompt, bool):
        raise ValueError("versioned_summary_prompt must be a boolean")

    normalized_cases = _validate_cases(cases, dataset_name)
    case_ids = [case["case_id"] for case in normalized_cases]
    history_length = len(normalized_cases[0]["series"])
    horizon = len(normalized_cases[0]["truth"])
    summary_prompt_version = hic.SUMMARY_PROMPT_VERSION if versioned_summary_prompt else "legacy"
    if versioned_summary_prompt:
        for case in normalized_cases:
            hic.build_summary_task_contract(
                target=case.get("target", "OT"),
                unit=case.get("unit", "unknown"),
                frequency=case.get("frequency"),
                horizon=horizon,
            )
    input_fingerprint = _fingerprint(
        normalized_cases, summary_prompt_version=summary_prompt_version,
    )
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is None:
        raise ValueError("checkpoint_path is required for resumable memory construction")

    if path.exists():
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != 1 or state.get("run_id") != run_id:
            raise ValueError("checkpoint schema or run_id does not match")
        if state.get("dataset") != dataset_name or state.get("case_ids") != case_ids:
            raise ValueError("checkpoint dataset or cases do not match current inputs")
        if state.get("summary_prompt_version", "legacy") != summary_prompt_version:
            raise ValueError("checkpoint summary prompt version does not match current run")
        if state.get("input_fingerprint") != input_fingerprint:
            raise ValueError("checkpoint inputs changed; use a new run_id")
        if state.get("history_length", history_length) != history_length or state.get("horizon", horizon) != horizon:
            raise ValueError("checkpoint history/horizon does not match current cases")
        state.setdefault("history_length", history_length)
        state.setdefault("horizon", horizon)
        if state.get("pending_call"):
            if not retry_pending:
                raise RuntimeError(
                    "checkpoint has an unresolved API call; pass retry_pending=True only "
                    "after reviewing whether that request completed"
                )
            state["pending_call"] = None
    else:
        if store.count() != 0:
            raise ValueError("a new memory run requires an empty isolated Qdrant collection")
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "dataset": dataset_name,
            "history_length": history_length,
            "horizon": horizon,
            "summary_prompt_version": summary_prompt_version,
            "case_ids": case_ids,
            "input_fingerprint": input_fingerprint,
            "logical_deepseek_calls": 0,
            "pending_call": None,
            "last_completed_call": None,
            "cases": {},
        }

    current_count = store.count()
    ingested_count = sum(
        bool(state["cases"].get(case_id, {}).get("ingested"))
        for case_id in case_ids
    )
    # A single extra point is possible if the process stopped after Qdrant upsert
    # but before atomically marking that upsert in the checkpoint.
    if current_count < ingested_count or current_count > ingested_count + 1:
        raise ValueError("Qdrant count does not match the resumable checkpoint")

    required = _required_calls(state)
    completed = state.get("last_completed_call")
    if completed:
        completed_case_id = completed.get("case_id")
        completed_stage = completed.get("stage")
        if completed_case_id not in case_ids or completed_stage not in _STAGES:
            raise ValueError("checkpoint contains an invalid completed API call")
        if completed_stage not in state["cases"].get(completed_case_id, {}):
            # The result is already persisted and will be replayed locally without
            # another provider call, so do not charge it again in preflight.
            required -= 1
    if state["logical_deepseek_calls"] + required > max_calls:
        raise ValueError(
            f"run requires {required} logical calls from the current checkpoint; "
            f"max_calls={max_calls} with {state['logical_deepseek_calls']} already used"
        )

    def save():
        _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt, commit_response):
        completed = state.get("last_completed_call")
        if completed:
            if completed.get("case_id") != case_id or completed.get("stage") != stage:
                raise RuntimeError("checkpoint contains an unapplied completed API result")
            response = completed["response"]
        else:
            if state.get("pending_call"):
                raise RuntimeError("checkpoint has an unresolved API call")
            if state["logical_deepseek_calls"] >= max_calls:
                raise RuntimeError(f"logical DeepSeek call budget ({max_calls}) exhausted")
            state["logical_deepseek_calls"] += 1
            state["pending_call"] = {"case_id": case_id, "stage": stage}
            save()
            response = call_json(prompt, temperature=0.0)
            if not isinstance(response, dict) or "_raw" in response:
                raise ValueError(f"{stage} response must be a valid JSON object")
            state["pending_call"] = None
            state["last_completed_call"] = {
                "case_id": case_id,
                "stage": stage,
                "response": response,
            }
            save()
        commit_response(response)
        state["last_completed_call"] = None
        save()

    for case in normalized_cases:
        case_id = case["case_id"]
        case_state = state["cases"].setdefault(case_id, {})

        if "summary" not in case_state:
            task_kwargs = {}
            if versioned_summary_prompt:
                task_kwargs = {
                    "target": case.get("target", "OT"),
                    "unit": case.get("unit", "unknown"),
                    "frequency": case.get("frequency"),
                    "horizon": horizon,
                }
            prompt = hic.build_summary_prompt(
                template,
                case["series"],
                case["news"],
                dataset_description=dataset_description,
                covariates=case["covariates"],
                **task_kwargs,
            )

            def commit_summary(response):
                if not response:
                    raise ValueError(f"summary response for {case_id} is empty")
                case_state["summary"] = response

            invoke(case_id, "summary", prompt, commit_summary)

        if "forecast" not in case_state:
            prompt = hic.build_reasoning_prompt(
                case_state["summary"],
                case["series"],
                covariates=case["covariates"],
                dataset_description=dataset_description,
                horizon=horizon,
                frequency=case.get("frequency"),
            )

            def commit_forecast(response):
                case_state["forecast"] = _validated_forecast(response, horizon)

            invoke(case_id, "forecast", prompt, commit_forecast)

        if "correction" not in case_state:
            prompt = hic.build_correction_prompt(
                case_state["summary"],
                case["series"],
                case["truth"],
                case_state["forecast"]["prediction"],
                case_state["forecast"]["reasoning"],
                covariates=case["covariates"],
                dataset_description=dataset_description,
            )

            def commit_correction(response):
                case_state["correction"] = _correction_text(response)

            invoke(case_id, "correction", prompt, commit_correction)

        if "embedding" not in case_state:
            vector = embed_summary(case_state["summary"], is_query=False)
            expected_dimension = getattr(store, "dimension", 1024)
            if not isinstance(vector, (list, tuple)) or len(vector) != expected_dimension:
                raise ValueError(
                    f"BGE-M3 embedding for {case_id} must have {expected_dimension} dimensions"
                )
            normalized_vector = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in normalized_vector):
                raise ValueError(f"BGE-M3 embedding for {case_id} contains non-finite values")
            if math.sqrt(sum(value * value for value in normalized_vector)) == 0:
                raise ValueError(f"BGE-M3 embedding for {case_id} must be non-zero")
            case_state["embedding"] = normalized_vector
            save()

        if not case_state.get("ingested"):
            store.upsert_completed_case(
                case_id=case_id,
                summary=case_state["summary"],
                correction=case_state["correction"],
                embedding=case_state["embedding"],
                observed_targets=case["truth"],
                prediction=case_state["forecast"]["prediction"],
                cutoff_date=case["target_end"],
                source=f"training:{dataset_name}:{run_id}",
            )
            case_state["ingested"] = True
            save()

    final_count = store.count()
    if final_count != len(normalized_cases):
        raise RuntimeError(
            f"Qdrant collection has {final_count} points; expected {len(normalized_cases)}"
        )
    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "history_length": history_length,
        "horizon": horizon,
        "embedding_model": _M3_MODEL_ID,
        "case_count": len(normalized_cases),
        "completed_case_ids": case_ids,
        "initial_collection_count": current_count,
        "checkpoint_ingested_count_at_start": ingested_count,
        "final_collection_count": final_count,
        "logical_deepseek_calls": state["logical_deepseek_calls"],
        "max_logical_deepseek_calls": max_calls,
        "input_fingerprint": input_fingerprint,
    }
