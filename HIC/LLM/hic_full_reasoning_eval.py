"""Paired, fixed-memory evaluation of full HIC corrective reasoning."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from LLM import hic
from LLM.hic_graph_data import build_scored_case_record, write_scored_case_jsonl
from LLM.deepseek_client import API_URL, call_chat_completion
from LLM.hic_qdrant_eval import _validated_forecast
from LLM.native_hic_data import load_native_hic_splits
from LLM.native_hic_experiment import make_numeric_baselines
from LLM.native_hic_pilot import _atomic_json
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.run_economy_hic_pilot import make_http_attempt_budget
from LLM.hic_phase2_eval import (
    build_validation_uncertainty,
    fuse_forecasts,
    select_validation_fusion,
)


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"
SOURCE_RUN_ID = "native_hic_pilot_native_climate_h12_20261018_02"
DEFAULT_RUN_ID = "hic_full_reasoning_climate_h12_20261018_01"
ARMS = ("no_retrieval", "summary_retrieval", "full_reasoning")
MAX_LOGICAL_CALLS = 80
MAX_HTTP_ATTEMPTS = 160
_OUTPUT_MARKER = "\n    OUTPUT JSON exactly:"


def _json_copy(value, label):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite JSON-compatible data") from None


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _request_identity(prompt):
    """Fingerprint the non-secret identity of the logical DeepSeek request."""
    body = {
        "model": hic.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": hic.mdl_cfg.get("max_output_tokens", 2048),
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    return {
        **_request_settings(),
        "request_sha256": _sha256_json({"api_url": API_URL, "body": body}),
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
    }


def _request_settings():
    return {
        "api_url": API_URL,
        "model": hic.MODEL,
        "temperature": 0.0,
        "max_tokens": hic.mdl_cfg.get("max_output_tokens", 2048),
    }


def _require_request_identity(record, expected, *, call_id, artifact):
    if not isinstance(record, dict) or record.get("request_identity") != expected:
        raise RuntimeError(
            f"{artifact} for {call_id} has a missing or different request identity; "
            "do not reuse it for this prompt or model configuration; inspect the "
            "checkpoint and start a new run ID if the request changed"
        )


def normalize_full_correction(response, *, raw_response):
    """Retain the complete parsed correction and the exact provider text."""
    if not isinstance(response, dict):
        raise ValueError("correction response must be a JSON object")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("raw_response must be non-empty provider text")
    normalized = _json_copy(response, "correction response")
    for field in ("Improved Reasoning", "Key Insights"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"correction response must contain non-empty {field}")
    return {
        "schema_version": 1,
        "response": normalized,
        "raw_response": raw_response,
    }


def render_full_correction(record):
    """Serialize a checkpoint record or Qdrant correction for prompt context."""
    if isinstance(record, str):
        try:
            response = json.loads(record)
        except json.JSONDecodeError:
            raise ValueError("record must contain a parsed full correction response") from None
    elif isinstance(record, dict):
        response = record.get("response")
        if (
            record.get("schema_version") == 2
            and isinstance(record.get("verified_facts"), dict)
            and isinstance(record.get("reflection"), dict)
        ):
            return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    else:
        response = None
    if (
        isinstance(response, dict)
        and response.get("schema_version") == 2
        and isinstance(response.get("verified_facts"), dict)
        and isinstance(response.get("reflection"), dict)
    ):
        return json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    if not isinstance(response, dict):
        raise ValueError("record must contain a parsed full correction response")
    for field in ("Improved Reasoning", "Key Insights"):
        value = response.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"full correction is missing {field}")
    return json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False)


def build_arm_prompt(
    arm,
    *,
    summary,
    series,
    dataset_description,
    horizon,
    frequency,
    covariates=None,
    retrieval=None,
    canonical_json_inputs=False,
):
    """Build a shared-input forecast prompt with one of the three HIC conditions."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    base = hic.build_reasoning_prompt(
        summary,
        series,
        dataset_description=dataset_description,
        covariates=covariates,
        horizon=horizon,
        frequency=frequency,
        canonical_json_inputs=canonical_json_inputs,
    )
    if arm == "no_retrieval":
        return base
    if not isinstance(retrieval, dict):
        raise ValueError(f"{arm} requires one shared completed-memory retrieval")
    case_id = retrieval.get("case_id")
    score = retrieval.get("score")
    past_summary = retrieval.get("summary")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("retrieved memory must include its case_id")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("retrieved memory must include a finite cosine score")
    if not isinstance(past_summary, (dict, list, str)):
        raise ValueError("retrieved memory must include its historical summary")

    head, marker, tail = base.rpartition(_OUTPUT_MARKER)
    if not marker:
        raise RuntimeError("base HIC prompt no longer contains its output marker")
    context = [
        "RETRIEVED COMPLETED TRAINING CASE",
        f"case_id: {case_id}",
        f"cosine_similarity: {float(score):.8f}",
        "historical_summary:",
        json.dumps(past_summary, ensure_ascii=False, sort_keys=True, indent=2),
    ]
    if arm == "summary_retrieval":
        context.extend([
            "Use this past summary only to compare the present context. It contains no corrected reasoning.",
            "Do not treat the old case as current evidence or copy any old forecast.",
        ])
    else:
        context.append("full_outcome_corrected_reasoning:")
        context.append(render_full_correction(retrieval.get("correction")))
        context.extend([
            "Check whether the old error mechanism applies to the present summary and numeric history.",
            "Use a lesson only when current evidence supports it; otherwise ignore it.",
            "Do not copy any old numeric forecast or claim the old event is happening now.",
        ])
    return head + "\n\n" + "\n".join(context) + marker + tail


def aggregate_arm_metrics(windows):
    """Aggregate paired forecast errors by target point, not by window mean."""
    if not windows:
        raise ValueError("at least one scored window is required")
    names = None
    totals = {}
    for window in windows:
        scoring = window.get("scoring")
        if not isinstance(scoring, dict):
            raise ValueError("every window must have completed scoring")
        truth = scoring.get("truth")
        forecasts = scoring.get("forecasts")
        if not isinstance(forecasts, dict) or not forecasts:
            raise ValueError("every scored window must contain forecasts")
        current_names = set(forecasts)
        if names is None:
            names = current_names
            totals = {name: {"target_points": 0, "windows": 0, "absolute_error": 0.0, "squared_error": 0.0} for name in names}
        elif current_names != names:
            raise ValueError("all windows must contain the same forecast arms")
        for name, prediction in forecasts.items():
            metrics = hic.forecast_metrics(prediction, truth)
            total = totals[name]
            total["target_points"] += metrics["n"]
            total["windows"] += 1
            total["absolute_error"] += metrics["mae"] * metrics["n"]
            total["squared_error"] += metrics["rmse"] ** 2 * metrics["n"]
    return {
        name: {
            "target_points": row["target_points"],
            "windows": row["windows"],
            "mae": row["absolute_error"] / row["target_points"],
            "mse": row["squared_error"] / row["target_points"],
            "rmse": math.sqrt(row["squared_error"] / row["target_points"]),
        }
        for name, row in totals.items()
    }


def _paths(run_id):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    stem = EVALUATIONS / safe
    collection = collection_name_for("BAAI/bge-m3", f"hic_fullreason_{digest}")
    return {
        "checkpoint": Path(f"{stem}.checkpoint.json"),
        "http_ledger": Path(f"{stem}.http_attempts.json"),
        "validation": Path(f"{stem}.validation.json"),
        "validation_graph_jsonl": Path(f"{stem}.validation.graph_records.jsonl"),
        "freeze": Path(f"{stem}.freeze.json"),
        "final": Path(f"{stem}.final.json"),
        "final_graph_jsonl": Path(f"{stem}.outer_test.graph_records.jsonl"),
        "qdrant": Path(f"{stem}.qdrant"),
        "collection": collection,
    }


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON artifact {Path(path).name}: {type(error).__name__}") from None


def _load_source_material():
    report_path = EVALUATIONS / f"{SOURCE_RUN_ID}.json"
    training_path = EVALUATIONS / f"{SOURCE_RUN_ID}.training.checkpoint.json"
    validation_path = EVALUATIONS / f"{SOURCE_RUN_ID}.validation.checkpoint.json"
    template_path = EVALUATIONS / f"{SOURCE_RUN_ID}.template.json"
    source_report = _load_json(report_path)
    training_cache = _load_json(training_path)
    validation_cache = _load_json(validation_path)
    template = _load_json(template_path)
    if source_report.get("status") != "completed_validation_pilot":
        raise ValueError("source Climate H12 run is not marked complete")
    if source_report.get("plan", {}).get("dataset") != "Climate" or source_report.get("plan", {}).get("horizon") != 12:
        raise ValueError("source report is not the expected Climate H12 run")
    expected_template_hash = source_report.get("template", {}).get("sha256")
    actual_template_hash = _sha256_json(template)
    if expected_template_hash != actual_template_hash:
        raise ValueError("cached template does not match the completed source report")

    splits = load_native_hic_splits(
        "Climate", horizon=12, include_outer_test_targets=False
    )
    training_cases = {case["case_id"]: case for case in splits["fit_cases"]}
    validation_windows = {
        window["case"]["case_id"]: window for window in splits["validation_windows"]
    }
    fit_ids = source_report["plan"]["training_case_ids"]
    validation_ids = source_report["plan"]["validation_case_ids"]
    if set(fit_ids) != set(training_cache.get("case_ids", [])):
        raise ValueError("cached fit IDs do not match the source report")
    if set(fit_ids) != set(training_cases):
        raise ValueError("source Climate fit cases do not match the completed training cache")
    if set(validation_ids) != set(validation_cache.get("case_ids", [])):
        raise ValueError("cached validation IDs do not match the source report")
    if set(validation_ids) != set(validation_windows):
        raise ValueError("source Climate validation windows do not match the source report")

    for case_id in fit_ids:
        cached = training_cache.get("cases", {}).get(case_id)
        if not isinstance(cached, dict) or not all(key in cached for key in ("summary", "forecast", "embedding", "correction")):
            raise ValueError(f"cached fit case {case_id} is incomplete")
        if len(cached["embedding"]) != 1024:
            raise ValueError(f"cached fit embedding has wrong dimension for {case_id}")
        _validated_forecast(cached["forecast"], horizon=12)
    for case_id in validation_ids:
        cached = validation_cache.get("windows", {}).get(case_id)
        if not isinstance(cached, dict) or not isinstance(cached.get("summary"), dict):
            raise ValueError(f"cached validation summary is missing for {case_id}")
        vectors = cached.get("embeddings", {})
        if not isinstance(vectors.get("query"), list) or len(vectors["query"]) != 1024:
            raise ValueError(f"cached validation query embedding is missing for {case_id}")
        truth = cached.get("scoring", {}).get("truth")
        if truth != validation_windows[case_id]["truth"]:
            raise ValueError(f"cached validation truth differs from source for {case_id}")

    fit_rows = [training_cases[case_id] for case_id in fit_ids]
    validation_rows = [validation_windows[case_id] for case_id in validation_ids]
    dev_fingerprint = _sha256_json({
        "fit": [
            {key: case.get(key) for key in ("case_id", "series", "truth", "news", "covariates", "target_start", "target_end")}
            for case in fit_rows
        ],
        "validation": [
            {
                "case": {key: window["case"].get(key) for key in ("case_id", "series", "news", "covariates", "target_start", "target_end")},
                "truth": window["truth"],
            }
            for window in validation_rows
        ],
        "fit_cache_hash": _sha256_bytes(training_path.read_bytes()),
        "validation_cache_hash": _sha256_bytes(validation_path.read_bytes()),
        "template_hash": actual_template_hash,
    })
    return {
        "source_report": source_report,
        "training_cache": training_cache,
        "validation_cache": validation_cache,
        "training_cases": training_cases,
        "validation_windows": validation_windows,
        "fit_ids": fit_ids,
        "validation_ids": validation_ids,
        "template": template,
        "template_path": template_path,
        "template_hash": actual_template_hash,
        "dataset_description": source_report["validation"]["dataset_description"],
        "dev_fingerprint": dev_fingerprint,
        "source_hashes": {
            "source_report": _sha256_bytes(report_path.read_bytes()),
            "training_cache": _sha256_bytes(training_path.read_bytes()),
            "validation_cache": _sha256_bytes(validation_path.read_bytes()),
            "template": actual_template_hash,
        },
        "outer_test_case_ids": source_report["plan"].get("outer_test_case_ids") or [],
        "splits_metadata": {
            "fit_end": splits["boundaries"]["fit_end"],
            "validation_end": splits["boundaries"]["validation_end"],
            "outer_test_start": splits["boundaries"]["outer_test_start"],
            "cadence": splits["cadence"],
            "history_length": splits["history_length"],
            "horizon": splits["horizon"],
        },
    }


def _has_scored_output(value):
    if not isinstance(value, dict):
        return False
    return any(key in value for key in (
        "prediction", "predictions", "forecast", "forecasts", "scoring", "metrics",
        "scored_outcome", "revealed_after_forecast",
    ))


def _audit_holdout_exposure(test_case_ids):
    """Fail if a candidate test case already has a saved forecast or score."""
    candidates = set(test_case_ids)
    exposures = []
    for path in sorted(EVALUATIONS.rglob("*.json")) + sorted(EVALUATIONS.rglob("*.jsonl")):
        if path.name.startswith(DEFAULT_RUN_ID):
            continue
        try:
            if path.suffix == ".jsonl":
                documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                documents = [_load_json(path)]
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        def walk(node, location):
            if isinstance(node, dict):
                direct = node.get("case_id")
                case_obj = node.get("case")
                nested_id = case_obj.get("case_id") if isinstance(case_obj, dict) else None
                known = node.get("known_at_forecast")
                known_id = known.get("case_id") if isinstance(known, dict) else None
                candidate = next((item for item in (direct, nested_id, known_id) if item in candidates), None)
                if candidate is not None and _has_scored_output(node):
                    exposures.append({"file": str(path.relative_to(ROOT)), "case_id": candidate, "path": location})
                for key, child in node.items():
                    walk(child, f"{location}.{key}")
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    walk(child, f"{location}[{index}]")

        for index, document in enumerate(documents):
            walk(document, f"$[{index}]")
    return {"candidate_case_ids": sorted(candidates), "exposures": exposures, "clean": not exposures}


def _config_code_hashes():
    files = (
        Path(__file__),
        ROOT / "LLM" / "hic.py",
        ROOT / "config" / "model.yaml",
        ROOT / "LLM" / "qdrant_store.py",
        ROOT / "LLM" / "native_hic_data.py",
        ROOT / "LLM" / "deepseek_client.py",
        ROOT / "LLM" / "hic_graph_data.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): _sha256_bytes(path.read_bytes())
        for path in files
    }


def _load_checkpoint(paths, material):
    checkpoint_path = paths["checkpoint"]
    if checkpoint_path.exists():
        state = _load_json(checkpoint_path)
        if state.get("schema_version") != 1 or state.get("run_id") != DEFAULT_RUN_ID:
            raise ValueError("checkpoint schema or run ID does not match")
        if state.get("dev_fingerprint") != material["dev_fingerprint"]:
            raise ValueError("source inputs changed; use a new run ID")
        if state.get("request_settings") != _request_settings():
            raise ValueError("checkpoint request settings are missing or changed; use a new run ID")
        if not paths["http_ledger"].is_file():
            raise ValueError("checkpoint exists but its cumulative HTTP ledger is missing")
        return state, True
    if paths["http_ledger"].exists():
        raise ValueError("HTTP ledger exists without a checkpoint; inspect it before resuming")
    state = {
        "schema_version": 1,
        "run_id": DEFAULT_RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "dev_fingerprint": material["dev_fingerprint"],
        "template_sha256": material["template_hash"],
        "request_settings": _request_settings(),
        "logical_calls_used": 0,
        "pending_call": None,
        "last_raw_call": None,
        "responses": {},
        "fit_corrections": {},
        "validation_windows": {},
        "test_windows": {},
        "failed_call_retries": [],
    }
    return state, False


def _save_checkpoint(path, state):
    _atomic_json(path, state)


def _invoke_json(*, call_id, prompt, state, paths, budget, api_key, max_calls, retry_pending=False):
    request_identity = _request_identity(prompt)
    existing = state["responses"].get(call_id)
    if existing is not None:
        _require_request_identity(
            existing, request_identity, call_id=call_id, artifact="cached response"
        )
        return existing["response"]
    last_raw = state.get("last_raw_call")
    if last_raw and last_raw.get("call_id") == call_id:
        _require_request_identity(
            last_raw, request_identity, call_id=call_id, artifact="saved raw response"
        )
        parsed = hic.extract_json(last_raw["raw_response"])
        if isinstance(parsed, dict) and "_raw" not in parsed:
            state["responses"][call_id] = {
                "raw_response": last_raw["raw_response"],
                "response": parsed,
                "request_identity": request_identity,
                "prompt_sha256": request_identity["prompt_sha256"],
                "model": hic.MODEL,
                "temperature": 0.0,
                "max_tokens": request_identity["max_tokens"],
            }
            state["last_raw_call"] = None
            state["pending_call"] = None
            _save_checkpoint(paths["checkpoint"], state)
            return parsed
        if not retry_pending:
            raise RuntimeError(
                f"saved provider response for {call_id} is not valid JSON; inspect checkpoint and rerun with --retry-pending"
            )
        state["failed_call_retries"].append({
            "call_id": call_id,
            "raw_response": last_raw["raw_response"],
            "request_identity": request_identity,
        })
        state["last_raw_call"] = None
        state["pending_call"] = None
    pending = state.get("pending_call")
    if pending is not None:
        if isinstance(pending, str):
            raise RuntimeError(
                f"unresolved API call {pending} has no request identity; "
                "do not retry it automatically; inspect the checkpoint and start a new run ID"
            )
        if not isinstance(pending, dict):
            raise RuntimeError("checkpoint pending_call has an invalid request identity record")
        pending_id = pending.get("call_id")
        if pending_id != call_id:
            raise RuntimeError(f"unresolved API call {pending_id}; resume that stage first")
        if pending.get("request_identity") != request_identity:
            raise RuntimeError(
                f"unresolved API call {call_id} has a different request identity; "
                "do not retry it with changed inputs or model settings"
            )
    if pending is not None and not retry_pending:
        raise RuntimeError(
            f"API call {call_id} is unresolved; rerun with --retry-pending after reviewing the checkpoint"
        )
    if pending is not None and retry_pending:
        state["failed_call_retries"].append({
            "call_id": call_id,
            "request_identity": request_identity,
            "status": "request outcome unresolved",
        })
        state["pending_call"] = None
    if state["logical_calls_used"] >= max_calls:
        raise RuntimeError(f"logical API call budget ({max_calls}) is exhausted")
    state["logical_calls_used"] += 1
    state["pending_call"] = {
        "call_id": call_id,
        "request_identity": request_identity,
    }
    state["last_raw_call"] = None
    _save_checkpoint(paths["checkpoint"], state)
    raw = call_chat_completion(
        prompt,
        api_key=api_key,
        model=hic.MODEL,
        temperature=0.0,
        max_tokens=hic.mdl_cfg.get("max_output_tokens", 2048),
        attempt_budget=budget,
    )
    state["last_raw_call"] = {
        "call_id": call_id,
        "raw_response": raw,
        "request_identity": request_identity,
    }
    _save_checkpoint(paths["checkpoint"], state)
    parsed = hic.extract_json(raw)
    if not isinstance(parsed, dict) or "_raw" in parsed:
        raise ValueError(f"DeepSeek response for {call_id} was not a JSON object; raw response is checkpointed")
    state["responses"][call_id] = {
        "raw_response": raw,
        "response": parsed,
        "request_identity": request_identity,
        "prompt_sha256": request_identity["prompt_sha256"],
        "model": hic.MODEL,
        "temperature": 0.0,
        "max_tokens": request_identity["max_tokens"],
    }
    state["last_raw_call"] = None
    state["pending_call"] = None
    _save_checkpoint(paths["checkpoint"], state)
    return parsed


def _open_local_store(paths):
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(paths["qdrant"]))
    store = QdrantHicStore(
        model_id="BAAI/bge-m3",
        collection_name=paths["collection"],
        client=client,
    )
    store.ensure_collection()
    return client, store


def _upsert_fit_memory(store, material, state, case_id):
    full_correction = state["fit_corrections"][case_id]
    cached = material["training_cache"]["cases"][case_id]
    case = material["training_cases"][case_id]
    return store.upsert_completed_case(
        case_id=case_id,
        summary=cached["summary"],
        correction=render_full_correction(full_correction),
        embedding=cached["embedding"],
        observed_targets=case["truth"],
        prediction=cached["forecast"]["prediction"],
        cutoff_date=case["target_end"],
        source=f"full-correction:{DEFAULT_RUN_ID}:fit-only",
    )


def _make_call_prompt(arm, case, summary, retrieval, dataset_description, horizon):
    return build_arm_prompt(
        arm,
        summary=summary,
        series=case["series"],
        dataset_description=dataset_description,
        horizon=horizon,
        frequency=case.get("frequency", "monthly"),
        covariates=case.get("covariates"),
        retrieval=retrieval,
    )


def _score_window(window_state, window, *, horizon):
    forecasts = {
        arm: _validated_forecast(window_state["forecasts"][arm], horizon=horizon)["prediction"]
        for arm in ARMS
    }
    truth = [float(value) for value in window["truth"]]
    if len(truth) != horizon:
        raise ValueError("scoring truth has an unexpected horizon")
    baseline_forecasts = make_numeric_baselines(window["case"]["series"], horizon=horizon)
    all_forecasts = {**forecasts, **{f"baseline:{key}": value for key, value in baseline_forecasts.items()}}
    metrics = {name: hic.forecast_metrics(prediction, truth) for name, prediction in all_forecasts.items()}
    window_state["scoring"] = {
        "truth": truth,
        "target_timestamps": list(window["target_timestamps"]),
        "forecasts": all_forecasts,
        "metrics": metrics,
    }


def _score_report(windows):
    result = aggregate_arm_metrics(windows)
    baseline_names = sorted({
        name for window in windows for name in window["scoring"]["forecasts"]
        if name.startswith("baseline:")
    })
    for name in baseline_names:
        result[name] = aggregate_arm_metrics([
            {"scoring": {
                "truth": window["scoring"]["truth"],
                "forecasts": {name: window["scoring"]["forecasts"][name]},
            }}
            for window in windows
        ])[name]
    return result


def _add_validation_blend(windows, *, bootstrap_replicates=2000, seed=20261018):
    """Add a validation-selected HIC/numeric blend and paired uncertainty report."""
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("validation windows must be a non-empty sequence")
    selection_windows = []
    for window in windows:
        scoring = window.get("scoring")
        if not isinstance(scoring, dict):
            raise ValueError("every validation window must include scoring")
        forecasts = scoring.get("forecasts")
        if not isinstance(forecasts, dict):
            raise ValueError("every validation window must include forecasts")
        numeric = {
            name.removeprefix("baseline:"): prediction
            for name, prediction in forecasts.items()
            if name.startswith("baseline:")
        }
        selection_windows.append({
            "case_id": window.get("case_id"),
            "target_start": window.get("target_start"),
            "truth": scoring.get("truth"),
            "full_reasoning": forecasts.get("full_reasoning"),
            "numeric_forecasts": numeric,
        })

    selection = select_validation_fusion(selection_windows)
    output_windows = _json_copy(list(windows), "validation windows")
    selected_baseline = f"baseline:{selection['numeric_baseline']}"
    for window in output_windows:
        scoring = window["scoring"]
        prediction = fuse_forecasts(
            scoring["forecasts"]["full_reasoning"],
            scoring["forecasts"][selected_baseline],
            alpha=selection["alpha"],
        )
        metrics = hic.forecast_metrics(prediction, scoring["truth"])
        scoring["forecasts"]["validation_blend"] = prediction
        scoring.setdefault("metrics", {})["validation_blend"] = metrics
        window["derived_forecasts"] = {
            "validation_blend": {
                "prediction": prediction,
                "numeric_baseline": selection["numeric_baseline"],
                "alpha": selection["alpha"],
                "selected_on_validation": True,
            }
        }

    uncertainty = build_validation_uncertainty(
        output_windows,
        selected_numeric_baseline=selected_baseline,
        blend_arm="validation_blend",
        selection_case_ids=[window["case_id"] for window in output_windows],
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return {
        "windows": output_windows,
        "selection": selection,
        "uncertainty": uncertainty,
    }


def _write_graph_snapshot(destination, records):
    destination = Path(destination)
    if destination.exists():
        existing = [
            json.loads(line)
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if existing != records:
            raise FileExistsError(
                f"graph snapshot already exists with different records: {destination.name}"
            )
        return
    write_scored_case_jsonl(destination, records)


def _build_graph_records(windows, *, saved_by_id, source_by_id, summaries, embeddings, template_sha256):
    """Export scored HIC results using the existing leakage-aware GraphBundle contract."""
    records = []
    for window in windows:
        case_id = window["case_id"]
        source = source_by_id[case_id]
        saved = saved_by_id[case_id]
        retrieval = saved["retrieval"]
        scoring = saved["scoring"]
        forecasts = {
            "0": saved["forecasts"]["no_retrieval"]["prediction"],
            "summary_retrieval": saved["forecasts"]["summary_retrieval"]["prediction"],
            "full_reasoning": saved["forecasts"]["full_reasoning"]["prediction"],
        }
        corrections = {
            "0": "No historical retrieval context was supplied.",
            "summary_retrieval": (
                f"Retrieved completed fit case {retrieval['case_id']}; supplied its summary only."
            ),
            "full_reasoning": retrieval["correction"],
        }
        baseline_forecasts = {
            name.removeprefix("baseline:"): values
            for name, values in scoring["forecasts"].items()
            if name.startswith("baseline:")
        }
        records.append(build_scored_case_record(
            source["case"],
            target_timestamps=scoring["target_timestamps"],
            summary=summaries[case_id],
            embedding=embeddings[case_id],
            forecasts=forecasts,
            truth=scoring["truth"],
            corrections=corrections,
            baseline_forecasts=baseline_forecasts,
            provenance={
                "experiment": "fixed-memory-full-reasoning-ab",
                "run_id": DEFAULT_RUN_ID,
                "embedding_model": "BAAI/bge-m3",
                "retrieval_collection": _paths(DEFAULT_RUN_ID)["collection"],
                "retrieved_fit_case_id": retrieval["case_id"],
                "retrieval_cosine": float(retrieval["score"]),
                "template_sha256": template_sha256,
            },
        ))
    return records


def _build_validation_report(material, state, paths):
    windows = []
    for case_id in material["validation_ids"]:
        saved = state["validation_windows"][case_id]
        source = material["validation_windows"][case_id]
        windows.append({
            "case_id": case_id,
            "history_start": source["case"]["history_start"],
            "history_end": source["case"]["history_end"],
            "target_start": source["case"]["target_start"],
            "target_end": source["case"]["target_end"],
            "target_timestamps": saved["scoring"]["target_timestamps"],
            "retrieval": saved["retrieval"],
            "summary": saved["summary"],
            "forecasts": saved["forecasts"],
            "scoring": saved["scoring"],
        })
    validation_additions = _add_validation_blend(windows)
    windows = validation_additions["windows"]
    aggregates = _score_report(windows)
    best_hic = min(ARMS, key=lambda arm: (aggregates[arm]["mae"], aggregates[arm]["rmse"], arm))
    baseline_names = [name for name in aggregates if name.startswith("baseline:")]
    best_baseline = min(baseline_names, key=lambda name: (aggregates[name]["mae"], aggregates[name]["rmse"], name))
    per_window = []
    for window in windows:
        metrics = window["scoring"]["metrics"]
        row = {"case_id": window["case_id"], "target_start": window["target_start"], "target_end": window["target_end"]}
        row["mae"] = {name: metrics[name]["mae"] for name in metrics}
        row["full_reasoning_delta_vs_no_retrieval"] = metrics["full_reasoning"]["mae"] - metrics["no_retrieval"]["mae"]
        row["full_reasoning_delta_vs_summary"] = metrics["full_reasoning"]["mae"] - metrics["summary_retrieval"]["mae"]
        per_window.append(row)
    return {
        "schema_version": 1,
        "run_id": DEFAULT_RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "status": "validation_complete",
        "protocol": {
            "dataset": "Climate",
            "cadence": "monthly",
            "history_length": 8,
            "horizon": 12,
            "fit_memory_size": len(material["fit_ids"]),
            "validation_windows": len(windows),
            "memory_policy": "fixed training-only memory; no validation-window insertion",
            "retrieval": "one top-1 BAAI/bge-m3 cosine hit; identical hit supplied to both retrieval arms",
            "arms": list(ARMS),
            "temperature": 0.0,
            "test_policy": "outer test remains unscored until a freeze manifest exists",
        },
        "source_hashes": material["source_hashes"],
        "dev_fingerprint": material["dev_fingerprint"],
        "template_sha256": material["template_hash"],
        "logical_calls_used": state["logical_calls_used"],
        "windows": windows,
        "aggregate_metrics": aggregates,
        "validation_selection": {
            "best_hic_arm_by_mae": best_hic,
            "best_numeric_baseline_by_mae": best_baseline,
            "validation_fusion": validation_additions["selection"],
            "primary_hypothesis_for_test": "full_reasoning",
            "selection_rule": "all three fixed HIC arms are pre-registered for the one final paired test; choose the validation-best numeric baseline for the primary comparison; the validation-tuned blend is exploratory and must be frozen before any test scoring",
        },
        "paired_uncertainty": validation_additions["uncertainty"],
        "per_window_diagnostics": per_window,
        "artifacts": {
            "checkpoint": str(paths["checkpoint"].relative_to(ROOT)),
            "qdrant_path": str(paths["qdrant"].relative_to(ROOT)),
            "collection": paths["collection"],
            "graph_records": str(paths["validation_graph_jsonl"].relative_to(ROOT)),
        },
        "limitations": [
            "Six development windows provide descriptive paired evidence, not statistical significance.",
            "The local Climate CSV does not document the OT target's physical meaning or unit.",
            "The result is an event/HIC branch experiment, not the complete VoT ETA/AFF system.",
        ],
    }


def prepare_plan(run_id=DEFAULT_RUN_ID):
    material = _load_source_material()
    exposure = _audit_holdout_exposure(material["outer_test_case_ids"])
    paths = _paths(run_id)
    existing = [str(path) for key, path in paths.items() if key != "collection" and path.exists()]
    return {
        "run_id": run_id,
        "dataset": "Climate",
        "history_length": 8,
        "horizon": 12,
        "fit_cases_with_cached_summary_vector_forecast": len(material["fit_ids"]),
        "validation_cases_with_cached_summary_vector_truth": len(material["validation_ids"]),
        "fit_correction_calls_to_restore_full_response": len(material["fit_ids"]),
        "paired_validation_forecast_calls": len(material["validation_ids"]) * len(ARMS),
        "outer_test_case_ids_from_existing_split_manifest": material["outer_test_case_ids"],
        "previous_scored_holdout_exposures": exposure,
        "expected_new_calls": len(material["fit_ids"]) + len(material["validation_ids"]) * len(ARMS) + len(material["outer_test_case_ids"]) * (1 + len(ARMS)),
        "logical_call_ceiling": MAX_LOGICAL_CALLS,
        "http_attempt_ceiling": MAX_HTTP_ATTEMPTS,
        "estimated_qdrant_seed_count": len(material["fit_ids"]),
        "validation_insertions": 0,
        "existing_run_artifacts": existing,
        "development_fingerprint": material["dev_fingerprint"],
        "template_sha256": material["template_hash"],
        "split_boundaries": material["splits_metadata"],
    }


def run_validation(*, env_file, retry_pending=False):
    material = _load_source_material()
    paths = _paths(DEFAULT_RUN_ID)
    state, resume = _load_checkpoint(paths, material)
    if paths["freeze"].exists() or paths["final"].exists():
        raise FileExistsError("validation cannot be rerun after the configuration has been frozen")
    load_dotenv(Path(env_file), override=False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is unavailable; set it in the specified env file")
    if not resume:
        _save_checkpoint(paths["checkpoint"], state)
    budget = make_http_attempt_budget(
        max_attempts=MAX_HTTP_ATTEMPTS,
        ledger_path=paths["http_ledger"],
        resume=resume,
    )
    client, store = _open_local_store(paths)
    try:
        for case_id in material["fit_ids"]:
            case = material["training_cases"][case_id]
            cached = material["training_cache"]["cases"][case_id]
            prompt = hic.build_correction_prompt(
                cached["summary"],
                case["series"],
                case["truth"],
                cached["forecast"]["prediction"],
                cached["forecast"].get("reasoning", ""),
                covariates=case.get("covariates"),
                dataset_description=material["dataset_description"],
            )
            call_id = f"fit-correction:{case_id}"
            request_identity = _request_identity(prompt)
            if case_id not in state["fit_corrections"]:
                response = _invoke_json(
                    call_id=call_id, prompt=prompt, state=state, paths=paths,
                    budget=budget, api_key=api_key, max_calls=MAX_LOGICAL_CALLS,
                    retry_pending=retry_pending,
                )
                raw = state["responses"][call_id]["raw_response"]
                state["fit_corrections"][case_id] = {
                    **normalize_full_correction(response, raw_response=raw),
                    "request_identity": request_identity,
                }
                _save_checkpoint(paths["checkpoint"], state)
            else:
                _require_request_identity(
                    state["fit_corrections"][case_id], request_identity,
                    call_id=call_id, artifact="cached fit correction",
                )
            _upsert_fit_memory(store, material, state, case_id)

        if store.count() != len(material["fit_ids"]):
            raise RuntimeError("Qdrant must contain exactly the fixed completed fit memory before validation")
        fit_id_set = set(material["fit_ids"])
        for case_id in material["validation_ids"]:
            cached = material["validation_cache"]["windows"][case_id]
            source_window = material["validation_windows"][case_id]
            window_state = state["validation_windows"].setdefault(case_id, {"forecasts": {}})
            if "retrieval" not in window_state:
                hit = store.search(cached["embeddings"]["query"], limit=1)
                if not hit or hit.get("case_id") not in fit_id_set:
                    raise RuntimeError(f"top-1 retrieval for {case_id} did not come from the fixed fit bank")
                window_state["retrieval"] = hit
                window_state["summary"] = cached["summary"]
                window_state["query_embedding_sha256"] = _sha256_json(cached["embeddings"]["query"])
                _save_checkpoint(paths["checkpoint"], state)
            retrieval = window_state["retrieval"]
            for arm in ARMS:
                prompt = _make_call_prompt(
                    arm, source_window["case"], cached["summary"], retrieval,
                    material["dataset_description"], 12,
                )
                call_id = f"validation:{case_id}:{arm}"
                if arm in window_state["forecasts"]:
                    _require_request_identity(
                        {"request_identity": window_state["forecasts"][arm].get("request_identity")},
                        _request_identity(prompt), call_id=call_id,
                        artifact="cached validation forecast",
                    )
                    continue
                response = _invoke_json(
                    call_id=call_id, prompt=prompt, state=state, paths=paths,
                    budget=budget, api_key=api_key, max_calls=MAX_LOGICAL_CALLS,
                    retry_pending=retry_pending,
                )
                forecast = _validated_forecast(response, horizon=12)
                window_state["forecasts"][arm] = {
                    "prediction": forecast["prediction"],
                    "reasoning": forecast["reasoning"],
                    "provider_response": response,
                    "request_identity": state["responses"][call_id]["request_identity"],
                    "prompt_sha256": state["responses"][call_id]["prompt_sha256"],
                }
                _save_checkpoint(paths["checkpoint"], state)
            if "scoring" not in window_state:
                _score_window(window_state, source_window, horizon=12)
                _save_checkpoint(paths["checkpoint"], state)

        report = _build_validation_report(material, state, paths)
        graph_records = _build_graph_records(
            report["windows"],
            saved_by_id=state["validation_windows"],
            source_by_id=material["validation_windows"],
            summaries={
                case_id: material["validation_cache"]["windows"][case_id]["summary"]
                for case_id in material["validation_ids"]
            },
            embeddings={
                case_id: material["validation_cache"]["windows"][case_id]["embeddings"]["query"]
                for case_id in material["validation_ids"]
            },
            template_sha256=material["template_hash"],
        )
        _write_graph_snapshot(paths["validation_graph_jsonl"], graph_records)
        _atomic_json(paths["validation"], report)
        return report
    finally:
        client.close()


def freeze_configuration():
    material = _load_source_material()
    paths = _paths(DEFAULT_RUN_ID)
    if paths["freeze"].exists():
        raise FileExistsError("freeze manifest already exists; it is immutable")
    report = _load_json(paths["validation"])
    if report.get("status") != "validation_complete" or report.get("dev_fingerprint") != material["dev_fingerprint"]:
        raise ValueError("completed validation report does not match current development inputs")
    exposure = _audit_holdout_exposure(material["outer_test_case_ids"])
    if not exposure["clean"]:
        raise ValueError("candidate outer-test cases have prior scored exposures; cannot call this an independent test")
    aggregates = report["aggregate_metrics"]
    best_numeric = report["validation_selection"]["best_numeric_baseline_by_mae"]
    manifest = {
        "schema_version": 1,
        "run_id": DEFAULT_RUN_ID,
        "status": "frozen_before_outer_test",
        "source_run_id": SOURCE_RUN_ID,
        "source_hashes": material["source_hashes"],
        "dev_fingerprint": material["dev_fingerprint"],
        "template_sha256": material["template_hash"],
        "code_sha256": _config_code_hashes(),
        "validation_report_sha256": _sha256_bytes(paths["validation"].read_bytes()),
        "outer_test_exposure_audit": exposure,
        "outer_test_case_ids": material["outer_test_case_ids"],
        "frozen_settings": {
            "dataset": "Climate",
            "horizon": 12,
            "history_length": 8,
            "llm_request": _request_settings(),
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": 1024,
            "distance": "cosine",
            "top_k": 1,
            "memory_policy": "25 completed fit cases, fixed throughout validation and test",
            "arms": list(ARMS),
            "temperature": 0.0,
            "primary_test_arm": "full_reasoning",
            "selected_numeric_baseline": best_numeric,
            "selection_metric": "pooled target-point MAE; RMSE secondary",
            "test_analysis": "report all preregistered arms and numeric baselines once; do not tune based on test",
        },
        "validation_metrics": aggregates,
    }
    _atomic_json(paths["freeze"], manifest)
    return manifest


def _verify_freeze(paths, material):
    if not paths["freeze"].is_file():
        raise FileNotFoundError("outer test requires an explicit frozen configuration manifest")
    manifest = _load_json(paths["freeze"])
    if manifest.get("status") != "frozen_before_outer_test":
        raise ValueError("freeze manifest has an invalid status")
    if manifest.get("dev_fingerprint") != material["dev_fingerprint"]:
        raise ValueError("development inputs differ from frozen manifest")
    if manifest.get("code_sha256") != _config_code_hashes():
        raise ValueError("pipeline code changed after freeze; do not use the existing test manifest")
    if manifest.get("frozen_settings", {}).get("llm_request") != _request_settings():
        raise ValueError("DeepSeek request settings changed after freeze; do not run the test")
    if manifest.get("validation_report_sha256") != _sha256_bytes(paths["validation"].read_bytes()):
        raise ValueError("validation report changed after freeze")
    return manifest


def run_outer_test(*, env_file, retry_pending=False):
    material = _load_source_material()
    paths = _paths(DEFAULT_RUN_ID)
    manifest = _verify_freeze(paths, material)
    state, resume = _load_checkpoint(paths, material)
    if paths["final"].exists():
        raise FileExistsError("final report already exists; the outer test is single-use")
    splits = load_native_hic_splits("Climate", horizon=12, include_outer_test_targets=True)
    test_windows = splits["test_windows"]
    expected_ids = manifest["outer_test_case_ids"]
    actual_ids = [window["case"]["case_id"] for window in test_windows]
    if actual_ids != expected_ids:
        raise ValueError("loaded Climate outer-test cases differ from the frozen split manifest")
    if len(test_windows) != 7:
        raise ValueError("expected the seven frozen Climate H12 test windows")

    load_dotenv(Path(env_file), override=False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is unavailable; set it in the specified env file")
    budget = make_http_attempt_budget(
        max_attempts=MAX_HTTP_ATTEMPTS,
        ledger_path=paths["http_ledger"],
        resume=True,
    )
    client, store = _open_local_store(paths)
    embedder = None
    try:
        if store.count() != len(material["fit_ids"]):
            raise RuntimeError("frozen test requires the unchanged fit-only Qdrant collection")
        fit_id_set = set(material["fit_ids"])
        for window in test_windows:
            case = window["case"]
            case_id = case["case_id"]
            row = state["test_windows"].setdefault(case_id, {"forecasts": {}})
            summary_call = f"test:{case_id}:summary"
            summary_prompt = hic.build_summary_prompt(
                material["template"], case["series"], case["news"],
                dataset_description=material["dataset_description"],
                covariates=case.get("covariates"),
            )
            if "summary" not in row:
                raw_summary = _invoke_json(
                    call_id=summary_call,
                    prompt=summary_prompt,
                    state=state, paths=paths, budget=budget, api_key=api_key,
                    max_calls=MAX_LOGICAL_CALLS, retry_pending=retry_pending,
                )
                row["summary"] = raw_summary
                row["summary_request_identity"] = state["responses"][summary_call]["request_identity"]
                _save_checkpoint(paths["checkpoint"], state)
            else:
                _require_request_identity(
                    {"request_identity": row.get("summary_request_identity")},
                    _request_identity(summary_prompt), call_id=summary_call,
                    artifact="cached outer-test summary",
                )
            if "query_embedding" not in row:
                if embedder is None:
                    embedder = hic.get_embedder()
                text = hic.summary_to_embedding_text(row["summary"])
                row["query_embedding"] = embedder.encode([text], is_query=True)[0]
                if len(row["query_embedding"]) != 1024:
                    raise RuntimeError("BGE-M3 returned an invalid query vector")
                _save_checkpoint(paths["checkpoint"], state)
            if "retrieval" not in row:
                hit = store.search(row["query_embedding"], limit=1)
                if not hit or hit.get("case_id") not in fit_id_set:
                    raise RuntimeError(f"outer-test retrieval for {case_id} did not come from the frozen fit bank")
                row["retrieval"] = hit
                row["query_embedding_sha256"] = _sha256_json(row["query_embedding"])
                _save_checkpoint(paths["checkpoint"], state)
            for arm in ARMS:
                prompt = _make_call_prompt(
                    arm, case, row["summary"], row["retrieval"],
                    material["dataset_description"], 12,
                )
                call_id = f"test:{case_id}:{arm}"
                if arm in row["forecasts"]:
                    _require_request_identity(
                        {"request_identity": row["forecasts"][arm].get("request_identity")},
                        _request_identity(prompt), call_id=call_id,
                        artifact="cached outer-test forecast",
                    )
                    continue
                response = _invoke_json(
                    call_id=call_id, prompt=prompt, state=state, paths=paths,
                    budget=budget, api_key=api_key, max_calls=MAX_LOGICAL_CALLS,
                    retry_pending=retry_pending,
                )
                forecast = _validated_forecast(response, horizon=12)
                row["forecasts"][arm] = {
                    "prediction": forecast["prediction"],
                    "reasoning": forecast["reasoning"],
                    "provider_response": response,
                    "request_identity": state["responses"][call_id]["request_identity"],
                    "prompt_sha256": state["responses"][call_id]["prompt_sha256"],
                }
                _save_checkpoint(paths["checkpoint"], state)

        if any(set(state["test_windows"][case_id]["forecasts"]) != set(ARMS) for case_id in actual_ids):
            raise RuntimeError("not all frozen outer-test forecasts are saved")
        # Test truths are applied only after every prediction has been durably checkpointed.
        for window in test_windows:
            case_id = window["case"]["case_id"]
            row = state["test_windows"][case_id]
            if "scoring" not in row:
                _score_window(row, window, horizon=12)
                _save_checkpoint(paths["checkpoint"], state)

        output_windows = []
        for case_id in actual_ids:
            row = state["test_windows"][case_id]
            source = next(window for window in test_windows if window["case"]["case_id"] == case_id)
            output_windows.append({
                "case_id": case_id,
                "history_start": source["case"]["history_start"],
                "history_end": source["case"]["history_end"],
                "target_start": source["case"]["target_start"],
                "target_end": source["case"]["target_end"],
                "retrieval": row["retrieval"],
                "summary": row["summary"],
                "forecasts": row["forecasts"],
                "scoring": row["scoring"],
            })
        aggregates = _score_report(output_windows)
        selected_baseline = manifest["frozen_settings"]["selected_numeric_baseline"]
        helpful = []
        harmful = []
        for row in output_windows:
            errors = row["scoring"]["metrics"]
            delta_k0 = errors["full_reasoning"]["mae"] - errors["no_retrieval"]["mae"]
            delta_summary = errors["full_reasoning"]["mae"] - errors["summary_retrieval"]["mae"]
            item = {
                "case_id": row["case_id"],
                "retrieved_case_id": row["retrieval"]["case_id"],
                "cosine_similarity": row["retrieval"]["score"],
                "full_reasoning_delta_mae_vs_no_retrieval": delta_k0,
                "full_reasoning_delta_mae_vs_summary": delta_summary,
                "full_reasoning_mae": errors["full_reasoning"]["mae"],
                "no_retrieval_mae": errors["no_retrieval"]["mae"],
                "summary_retrieval_mae": errors["summary_retrieval"]["mae"],
                "selected_numeric_baseline_mae": errors[selected_baseline]["mae"],
            }
            if delta_k0 < 0 and delta_summary < 0:
                helpful.append(item)
            if delta_k0 > 0 and delta_summary > 0:
                harmful.append(item)
        final = {
            "schema_version": 1,
            "run_id": DEFAULT_RUN_ID,
            "status": "outer_test_complete",
            "freeze_manifest_sha256": _sha256_bytes(paths["freeze"].read_bytes()),
            "protocol": manifest["frozen_settings"],
            "logical_calls_used": state["logical_calls_used"],
            "windows": output_windows,
            "aggregate_metrics": aggregates,
            "primary_comparisons": {
                "full_reasoning_vs_no_retrieval_mae_delta": aggregates["full_reasoning"]["mae"] - aggregates["no_retrieval"]["mae"],
                "full_reasoning_vs_summary_mae_delta": aggregates["full_reasoning"]["mae"] - aggregates["summary_retrieval"]["mae"],
                "full_reasoning_vs_selected_numeric_mae_delta": aggregates["full_reasoning"]["mae"] - aggregates[selected_baseline]["mae"],
                "selected_numeric_baseline": selected_baseline,
            },
            "case_diagnostics": {"full_reasoning_helped_vs_both_retrieval_controls": helpful, "full_reasoning_hurt_vs_both_retrieval_controls": harmful},
            "artifacts": {
                "freeze_manifest": str(paths["freeze"].relative_to(ROOT)),
                "checkpoint": str(paths["checkpoint"].relative_to(ROOT)),
                "graph_records": str(paths["final_graph_jsonl"].relative_to(ROOT)),
                "qdrant_collection": paths["collection"],
            },
            "limitations": [
                "Seven temporal windows are a single held-out test block and do not establish broad generalization or statistical significance.",
                "The Climate OT target's physical meaning and unit are undocumented in the source metadata.",
                "This measures the HIC event branch; ETA/AFF fusion from full VoT is not included.",
            ],
        }
        graph_records = _build_graph_records(
            output_windows,
            saved_by_id=state["test_windows"],
            source_by_id={window["case"]["case_id"]: window for window in test_windows},
            summaries={case_id: state["test_windows"][case_id]["summary"] for case_id in actual_ids},
            embeddings={case_id: state["test_windows"][case_id]["query_embedding"] for case_id in actual_ids},
            template_sha256=material["template_hash"],
        )
        _write_graph_snapshot(paths["final_graph_jsonl"], graph_records)
        _atomic_json(paths["final"], final)
        return final
    finally:
        if embedder is not None:
            del embedder
        client.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--phase", choices=("plan", "validation", "freeze", "outer-test"), default="plan")
    parser.add_argument("--execute", action="store_true", help="allow bounded DeepSeek calls for the selected phase")
    parser.add_argument("--env-file", type=Path, default=Path(r"C:\Users\admin\Desktop\Research GNN\config\api.env"))
    parser.add_argument("--retry-pending", action="store_true", help="retry a saved unresolved/invalid call after reviewing its checkpoint")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.run_id != DEFAULT_RUN_ID:
        raise SystemExit(f"this verified cache is bound to run ID {DEFAULT_RUN_ID}")
    if args.phase == "plan":
        print(json.dumps(prepare_plan(args.run_id), ensure_ascii=False, indent=2))
        return 0
    if args.phase in {"validation", "outer-test"} and not args.execute:
        raise SystemExit(f"--phase {args.phase} requires --execute")
    if args.phase == "validation":
        result = run_validation(env_file=args.env_file, retry_pending=args.retry_pending)
    elif args.phase == "freeze":
        result = freeze_configuration()
    else:
        result = run_outer_test(env_file=args.env_file, retry_pending=args.retry_pending)
    print(json.dumps({
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "logical_calls_used": result.get("logical_calls_used"),
        "artifact": str((_paths(DEFAULT_RUN_ID)["freeze"] if args.phase == "freeze" else _paths(DEFAULT_RUN_ID)["final"] if args.phase == "outer-test" else _paths(DEFAULT_RUN_ID)["validation"]).relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
