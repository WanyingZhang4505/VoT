"""Prequential HIC evaluation over chronological Economy holdout windows."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from . import hic
from .economy_pilot import (
    ECONOMY_DESCRIPTION,
    FORECAST_LENGTH,
    HISTORY_LENGTH,
    _correction_text,
    _finite_vector,
    _validated_embedding,
    _validated_forecast,
)


def _fingerprint_windows(windows):
    fingerprints = {}
    for window in windows:
        encoded = json.dumps(
            window,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fingerprints[window["case"]["case_id"]] = hashlib.sha256(encoded).hexdigest()
    return fingerprints


def _validate_windows(windows, seed_case_ids):
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("windows must be a non-empty chronological sequence")
    if not isinstance(seed_case_ids, (list, tuple)) or not seed_case_ids:
        raise ValueError("seed_case_ids must identify the existing completed memory")
    if any(not isinstance(case_id, str) or not case_id for case_id in seed_case_ids):
        raise ValueError("seed case IDs must be non-empty strings")
    if len(set(seed_case_ids)) != len(seed_case_ids):
        raise ValueError("seed case IDs must be unique")

    ids = set(seed_case_ids)
    previous_target_end = None
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or not isinstance(window.get("case"), dict):
            raise ValueError(f"window {index} must contain a case object")
        case = window["case"]
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("test case IDs must be unique and distinct from seed cases")
        ids.add(case_id)
        if "truth" in case or "target_values" in case:
            raise ValueError("held-out truth must remain separate from the case input")

        history_start = case.get("history_start")
        history_end = case.get("history_end")
        target_start = case.get("target_start")
        target_end = case.get("target_end")
        if not all(isinstance(value, str) and value for value in (
            history_start, history_end, target_start, target_end
        )):
            raise ValueError(f"window {case_id} must provide ISO date boundaries")
        if not history_start <= history_end < target_start <= target_end:
            raise ValueError(f"window {case_id} has invalid or overlapping history/target dates")
        if previous_target_end is not None and target_start <= previous_target_end:
            raise ValueError("test target windows must be chronological and non-overlapping")
        previous_target_end = target_end

        _finite_vector(case.get("series"), HISTORY_LENGTH, f"history for {case_id}")
        _finite_vector(window.get("truth"), FORECAST_LENGTH, f"truth for {case_id}")
        covariates = case.get("covariates")
        if not isinstance(covariates, dict) or not covariates:
            raise ValueError(f"case {case_id} must include its aligned covariate histories")
        for name, values in covariates.items():
            _finite_vector(values, HISTORY_LENGTH, f"{name} history for {case_id}")
        if not isinstance(case.get("news"), str) or not case["news"].strip():
            raise ValueError(f"case {case_id} must include cutoff-filtered news")


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _aggregate_metrics(test_states, prediction_key):
    predictions = []
    truth = []
    for state in test_states:
        predictions.extend(state[prediction_key]["prediction"])
        truth.extend(state["scoring"]["truth"])
    return hic.forecast_metrics(predictions, truth)


def run_economy_rolling_eval(
    *,
    windows,
    seed_case_ids,
    template,
    store,
    embed_summary,
    call_json,
    run_id,
    max_calls=30,
    checkpoint_path=None,
    resume=False,
    retry_pending=False,
):
    """Compare no-retrieval and top-1 HIC forecasts, then append each completed case.

    The supplied collection must contain only the declared seed cases and cases
    from earlier windows in this same chronological run. Held-out truth is first
    placed in the checkpoint after both forecasts have been persisted.
    """
    if not isinstance(run_id, str) or not run_id or len(run_id) > 64:
        raise ValueError("run_id must be a non-empty string of at most 64 characters")
    if not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty JSON object")
    _validate_windows(windows, seed_case_ids)
    fingerprints = _fingerprint_windows(windows)
    path = Path(checkpoint_path) if checkpoint_path else None
    if resume and (path is None or not path.is_file()):
        raise FileNotFoundError("resume requires an existing rolling-evaluation checkpoint")
    if not resume and path is not None and path.exists():
        raise FileExistsError("checkpoint already exists; pass resume=True or use a new run_id")

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "seed_case_ids": list(seed_case_ids),
        "test_fingerprints": fingerprints,
        "logical_deepseek_calls": 0,
        "pending_call": None,
        "responses": {},
        "tests": {},
    }
    if resume:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if (
            state.get("schema_version") != 1
            or state.get("run_id") != run_id
            or state.get("seed_case_ids") != list(seed_case_ids)
            or state.get("test_fingerprints") != fingerprints
        ):
            raise ValueError("checkpoint does not match this run, seed memory, and test windows")
        if state.get("pending_call"):
            if not retry_pending:
                raise RuntimeError(
                    "checkpoint has an unresolved API call; review it and pass retry_pending=True"
                )
            state["pending_call"] = None

    completed_ids = {
        case_id for case_id, test_state in state["tests"].items()
        if test_state.get("ingested")
    }
    current_ingest_pending = any(
        test_state.get("ingest_pending") and not test_state.get("ingested")
        for test_state in state["tests"].values()
    )
    expected_count = len(seed_case_ids) + len(completed_ids)
    actual_count = store.count()
    allowed_counts = {expected_count}
    if current_ingest_pending:
        allowed_counts.add(expected_count + 1)
    if actual_count not in allowed_counts:
        raise ValueError(
            f"vector collection count {actual_count} does not match the checkpoint; "
            f"expected one of {sorted(allowed_counts)}"
        )

    def save():
        if path is not None:
            _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt):
        response_key = f"{case_id}::{stage}"
        if response_key in state["responses"]:
            return state["responses"][response_key]
        if state.get("pending_call"):
            raise RuntimeError("checkpoint has an unresolved API call")
        if state["logical_deepseek_calls"] >= max_calls:
            raise RuntimeError(f"DeepSeek logical-call budget ({max_calls}) exhausted")
        state["logical_deepseek_calls"] += 1
        state["pending_call"] = {"case_id": case_id, "stage": stage}
        save()
        response = call_json(prompt, temperature=0.0)
        if not isinstance(response, dict):
            raise ValueError(f"{stage} response must be a JSON object")
        state["responses"][response_key] = response
        state["pending_call"] = None
        save()
        return response

    previous_description = hic.dataset_description
    hic.dataset_description = ECONOMY_DESCRIPTION
    eligible_case_ids = set(seed_case_ids)
    try:
        for window in windows:
            case = window["case"]
            case_id = case["case_id"]
            if case_id in completed_ids:
                eligible_case_ids.add(case_id)
                continue

            history = _finite_vector(case["series"], HISTORY_LENGTH, f"history for {case_id}")
            test_state = state["tests"].setdefault(case_id, {})
            if "summary" not in test_state:
                response = invoke(
                    case_id,
                    "summary",
                    hic.build_summary_prompt(
                        template,
                        {"OT": history, **case["covariates"]},
                        case["news"],
                    ),
                )
                if not response:
                    raise ValueError(f"summary response for {case_id} must not be empty")
                test_state["summary"] = response
                save()

            if "embeddings" not in test_state:
                test_state["embeddings"] = {
                    "query": _validated_embedding(
                        embed_summary(test_state["summary"], is_query=True),
                        f"query embedding for {case_id}",
                    ),
                    "passage": _validated_embedding(
                        embed_summary(test_state["summary"], is_query=False),
                        f"passage embedding for {case_id}",
                    ),
                }
                save()

            if "retrieved_case" not in test_state:
                retrieval = store.search(test_state["embeddings"]["query"], limit=1)
                if not retrieval or not retrieval.get("correction"):
                    raise RuntimeError(f"no completed correction was retrieved for {case_id}")
                if retrieval.get("case_id") not in eligible_case_ids:
                    raise RuntimeError(
                        f"retrieval returned a future or unknown case for {case_id}: "
                        f"{retrieval.get('case_id')}"
                    )
                test_state["retrieved_case"] = retrieval
                save()

            for arm, correction in (
                ("control", None),
                ("retrieval", test_state["retrieved_case"]["correction"]),
            ):
                if arm in test_state:
                    continue
                prompt = hic.build_reasoning_prompt(
                    test_state["summary"],
                    history,
                    correction,
                    covariates=case["covariates"],
                )
                test_state[arm] = _validated_forecast(
                    invoke(case_id, f"forecast_{arm}", prompt)
                )
                save()

            truth = _finite_vector(window["truth"], FORECAST_LENGTH, f"truth for {case_id}")
            if "scoring" not in test_state:
                test_state["scoring"] = {
                    "truth": truth,
                    "control_metrics": hic.forecast_metrics(
                        test_state["control"]["prediction"], truth
                    ),
                    "retrieval_metrics": hic.forecast_metrics(
                        test_state["retrieval"]["prediction"], truth
                    ),
                    "persistence_metrics": hic.forecast_metrics(
                        hic.persistence_forecast(history, FORECAST_LENGTH), truth
                    ),
                    "history_mean_metrics": hic.forecast_metrics(
                        [sum(history) / len(history)] * FORECAST_LENGTH, truth
                    ),
                }
                save()
            elif test_state["scoring"].get("truth") != truth:
                raise ValueError(f"truth changed for checkpointed case {case_id}")

            if "correction" not in test_state:
                retrieval_forecast = test_state["retrieval"]
                prompt = hic.build_correction_prompt(
                    test_state["summary"],
                    history,
                    truth,
                    retrieval_forecast["prediction"],
                    retrieval_forecast["reasoning"],
                )
                test_state["correction"] = _correction_text(
                    invoke(case_id, "correction_retrieval", prompt)
                )
                save()

            if not test_state.get("ingested"):
                test_state["ingest_pending"] = True
                save()
                store.upsert_completed_case(
                    case_id=case_id,
                    summary=test_state["summary"],
                    correction=test_state["correction"],
                    embedding=test_state["embeddings"]["passage"],
                    observed_targets=truth,
                    prediction=test_state["retrieval"]["prediction"],
                    cutoff_date=case["target_end"],
                    source=f"economy-rolling-eval:{run_id}",
                )
                test_state["ingested"] = True
                test_state["ingest_pending"] = False
                save()
                completed_ids.add(case_id)
            eligible_case_ids.add(case_id)

        result_states = [state["tests"][window["case"]["case_id"]] for window in windows]
        tests = []
        for window, test_state in zip(windows, result_states):
            case = window["case"]
            control_metrics = test_state["scoring"]["control_metrics"]
            retrieval_metrics = test_state["scoring"]["retrieval_metrics"]
            tests.append({
                "case_id": case["case_id"],
                "history_start": case["history_start"],
                "history_end": case["history_end"],
                "target_start": case["target_start"],
                "target_end": case["target_end"],
                "truth": test_state["scoring"]["truth"],
                "control_prediction": test_state["control"]["prediction"],
                "control_metrics": control_metrics,
                "retrieval_prediction": test_state["retrieval"]["prediction"],
                "retrieval_metrics": retrieval_metrics,
                "persistence_metrics": test_state["scoring"]["persistence_metrics"],
                "history_mean_metrics": test_state["scoring"]["history_mean_metrics"],
                "retrieved_case": test_state["retrieved_case"],
                "mae_improvement_vs_control": control_metrics["mae"] - retrieval_metrics["mae"],
                "rmse_improvement_vs_control": control_metrics["rmse"] - retrieval_metrics["rmse"],
            })
        final_count = store.count()
        expected_final_count = len(seed_case_ids) + len(windows)
        if final_count != expected_final_count:
            raise RuntimeError(
                f"expected {expected_final_count} seeded/completed cases, found {final_count}"
            )
        return {
            "run_id": run_id,
            "dataset": "Economy_sort",
            "embedding_model": "BAAI/bge-m3",
            "forecast_model": "deepseek-flash",
            "retrieval_k": 1,
            "initial_collection_count": len(seed_case_ids),
            "final_collection_count": final_count,
            "seed_case_ids": list(seed_case_ids),
            "test_case_ids": [window["case"]["case_id"] for window in windows],
            "logical_deepseek_calls": state["logical_deepseek_calls"],
            "max_logical_deepseek_calls": max_calls,
            "tests": tests,
            "aggregate": {
                "control_metrics": _aggregate_metrics(result_states, "control"),
                "retrieval_metrics": _aggregate_metrics(result_states, "retrieval"),
            },
        }
    finally:
        hic.dataset_description = previous_description
