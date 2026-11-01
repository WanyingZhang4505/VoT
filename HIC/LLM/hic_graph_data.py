"""Leakage-aware records for completed HIC forecast windows."""

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

from LLM.hic import forecast_metrics


SCHEMA_VERSION = 1
_POST_SCORE_KEYS = {
    "truth",
    "target_values",
    "metrics",
    "correction",
    "corrections",
    "gain_vs_k0",
}


def _finite_values(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric list")
    if length is not None and len(values) != length:
        raise ValueError(f"{label} must have the same non-zero length as its aligned values")
    try:
        normalized = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain only finite numbers") from None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must contain only finite numbers")
    return normalized


def _json_copy(value, label):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite JSON-compatible data") from None


def _timestamp_values(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty timestamp list")
    result = []
    parsed = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain ISO timestamp strings")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{label} must contain ISO timestamp strings") from None
        result.append(value)
        parsed.append(stamp)
    if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def _normalize_mapping_values(mapping, label, *, length=None):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{label} must be a non-empty mapping")
    normalized = {}
    for raw_name, values in mapping.items():
        name = str(raw_name)
        if not name.strip() or name in normalized:
            raise ValueError(f"{label} must have unique non-empty names")
        normalized[name] = _finite_values(values, f"{label}[{name}]", length=length)
    return normalized


def _contains_post_score_key(value):
    if isinstance(value, dict):
        return any(
            str(key).lower() in _POST_SCORE_KEYS or _contains_post_score_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_post_score_key(item) for item in value)
    return False


def build_scored_case_record(
    case,
    *,
    target_timestamps,
    summary,
    embedding,
    forecasts,
    truth,
    corrections,
    baseline_forecasts,
    provenance,
):
    """Create an immutable-style record separating as-of features from outcomes.

    ``case`` is expected to come from the cutoff-aware native HIC loader. The
    record builder deliberately ignores extra fields such as ``truth`` on that
    input, so they cannot accidentally become forecast-time graph features.
    """
    if not isinstance(case, dict):
        raise ValueError("case must be a dictionary")
    case_id = case.get("case_id")
    dataset = case.get("dataset")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case must have a non-empty case_id")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError(f"case {case_id} must have a non-empty dataset")

    history = _finite_values(case.get("series"), f"history for {case_id}")
    history_timestamps = _timestamp_values(
        case.get("source_dates"), f"history timestamps for {case_id}"
    )
    if len(history_timestamps) != len(history):
        raise ValueError(f"history timestamps and values must align for {case_id}")
    parsed_history = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in history_timestamps
    ]

    required_dates = ("history_start", "history_end", "target_start", "target_end")
    dates = {}
    for key in required_dates:
        value = case.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"case {case_id} must include {key}")
        try:
            dates[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"case {case_id} has an invalid {key}") from None
    if dates["history_start"] > dates["history_end"]:
        raise ValueError(f"case {case_id} history cutoffs are reversed")
    if dates["target_start"] > dates["target_end"]:
        raise ValueError(f"case {case_id} target cutoffs are reversed")
    if parsed_history[0] != dates["history_start"] or parsed_history[-1] > dates["history_end"]:
        raise ValueError(
            f"history timestamps must start at history_start and end no later than history_end for {case_id}"
        )
    if dates["history_end"] >= dates["target_start"]:
        raise ValueError(f"target period must begin after the forecast cutoff for {case_id}")

    target_steps = _timestamp_values(target_timestamps, f"target timestamps for {case_id}")
    observed = _finite_values(truth, f"truth for {case_id}")
    if len(target_steps) != len(observed):
        raise ValueError(f"target timestamps and truth must align for {case_id}")
    forecast_values = _normalize_mapping_values(forecasts, "forecasts", length=len(observed))
    if "0" not in forecast_values:
        raise ValueError("forecasts must include the K=0 control arm named '0'")

    correction_values = {}
    if not isinstance(corrections, dict):
        raise ValueError("corrections must be a mapping by forecast arm")
    for raw_name, text in corrections.items():
        name = str(raw_name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"correction for arm {name} must be non-empty text")
        correction_values[name] = text.strip()
    if set(correction_values) != set(forecast_values):
        raise ValueError("corrections must have exactly one entry for each forecast arm")

    baselines = _normalize_mapping_values(
        baseline_forecasts, "baseline_forecasts", length=len(observed)
    )
    covariates = case.get("covariates") or {}
    if not isinstance(covariates, dict):
        raise ValueError(f"covariates for {case_id} must be a mapping")
    normalized_covariates = {
        str(name): _finite_values(values, f"covariate {name} for {case_id}", length=len(history))
        for name, values in covariates.items()
    }
    if len(normalized_covariates) != len(covariates):
        raise ValueError(f"covariate names must be unique for {case_id}")

    facts = case.get("news", "")
    if not isinstance(facts, str):
        raise ValueError(f"news for {case_id} must be text")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("provenance must be a non-empty mapping")
    if not isinstance(provenance.get("embedding_model"), str) or not provenance["embedding_model"].strip():
        raise ValueError("provenance must identify embedding_model")
    vector = _finite_values(embedding, f"embedding for {case_id}")
    summary_copy = _json_copy(summary, "summary")
    provenance_copy = _json_copy(provenance, "provenance")
    group = case.get("group", [])
    if not isinstance(group, (list, tuple)) or not all(isinstance(item, (str, int, float)) for item in group):
        raise ValueError(f"group for {case_id} must be a list of scalar path components")

    metrics = {
        arm: forecast_metrics(prediction, observed)
        for arm, prediction in forecast_values.items()
    }
    baseline_metrics = {
        name: forecast_metrics(prediction, observed)
        for name, prediction in baselines.items()
    }
    control_mae = metrics["0"]["mae"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "known_at_forecast": {
            "case_id": case_id,
            "dataset": dataset,
            "series_key": {"dataset": dataset, "group": list(group)},
            "group_path": list(group),
            "frequency": case.get("frequency", "unspecified"),
            "forecast_cutoff": case["history_end"],
            "history_start": case["history_start"],
            "history_end": case["history_end"],
            "target_start": case["target_start"],
            "target_end": case["target_end"],
            "history_timestamps": history_timestamps,
            "history_values": history,
            "fact_text": facts,
            "numeric_covariates": normalized_covariates,
            "hic_summary": summary_copy,
            "embedding": {
                "model": provenance["embedding_model"],
                "dimension": len(vector),
                "vector": vector,
            },
            "forecasts": forecast_values,
            "baseline_forecasts": baselines,
        },
        "revealed_after_forecast": {
            "target_timestamps": target_steps,
            "truth": observed,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "corrections": correction_values,
            "gain_vs_k0": {
                arm: control_mae - row["mae"] for arm, row in metrics.items()
            },
        },
        "provenance": provenance_copy,
    }
    validate_scored_case_record(record)
    return record


def validate_scored_case_record(record):
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"record must use schema_version {SCHEMA_VERSION}")
    known = record.get("known_at_forecast")
    revealed = record.get("revealed_after_forecast")
    if not isinstance(known, dict) or not isinstance(revealed, dict):
        raise ValueError("record must contain known_at_forecast and revealed_after_forecast objects")
    if _contains_post_score_key(known):
        raise ValueError("known_at_forecast contains a post-score field")
    if not isinstance(known.get("case_id"), str) or not known["case_id"].strip():
        raise ValueError("known_at_forecast must have a non-empty case_id")
    for key in ("dataset", "frequency", "forecast_cutoff", "history_start", "history_end", "target_start", "target_end"):
        if not isinstance(known.get(key), str) or not known[key].strip():
            raise ValueError(f"known_at_forecast must include {key}")
    if known["forecast_cutoff"] != known["history_end"]:
        raise ValueError("forecast_cutoff must equal history_end")
    history = _finite_values(known.get("history_values"), "history_values")
    history_times = _timestamp_values(known.get("history_timestamps"), "history_timestamps")
    if len(history) != len(history_times):
        raise ValueError("history timestamps and values must align")
    parsed_history = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in history_times]
    parsed_start = datetime.fromisoformat(known["history_start"].replace("Z", "+00:00"))
    parsed_end = datetime.fromisoformat(known["history_end"].replace("Z", "+00:00"))
    parsed_target_start = datetime.fromisoformat(known["target_start"].replace("Z", "+00:00"))
    parsed_target_end = datetime.fromisoformat(known["target_end"].replace("Z", "+00:00"))
    if parsed_history[0] != parsed_start or parsed_history[-1] > parsed_end:
        raise ValueError("history timestamps must start at history_start and end no later than history_end")
    if parsed_end >= parsed_target_start or parsed_target_end < parsed_target_start:
        raise ValueError("target period must follow the forecast cutoff")
    if not isinstance(known.get("fact_text"), str):
        raise ValueError("fact_text must be text")
    covariates = known.get("numeric_covariates")
    if not isinstance(covariates, dict):
        raise ValueError("numeric_covariates must be a mapping")
    for name, values in covariates.items():
        _finite_values(values, f"numeric_covariates[{name}]", length=len(history))
    if not isinstance(known.get("hic_summary"), (dict, list, str, int, float, bool, type(None))):
        raise ValueError("hic_summary must be JSON-compatible data")
    _json_copy(known["hic_summary"], "hic_summary")
    embedding = known.get("embedding")
    if not isinstance(embedding, dict) or not isinstance(embedding.get("model"), str):
        raise ValueError("embedding metadata must include its model")
    vector = _finite_values(embedding.get("vector"), "embedding vector")
    if embedding.get("dimension") != len(vector):
        raise ValueError("embedding dimension does not match its vector")
    forecasts = _normalize_mapping_values(known.get("forecasts"), "forecasts")
    if "0" not in forecasts:
        raise ValueError("forecasts must include the K=0 control arm named '0'")
    baseline_forecasts = _normalize_mapping_values(known.get("baseline_forecasts"), "baseline_forecasts")
    truth = _finite_values(revealed.get("truth"), "revealed truth")
    target_times = _timestamp_values(revealed.get("target_timestamps"), "revealed target timestamps")
    if len(truth) != len(target_times):
        raise ValueError("target timestamps and truth must align")
    forecasts = _normalize_mapping_values(known["forecasts"], "forecasts", length=len(truth))
    baseline_forecasts = _normalize_mapping_values(
        known["baseline_forecasts"], "baseline_forecasts", length=len(truth)
    )
    corrections = revealed.get("corrections")
    if not isinstance(corrections, dict) or set(corrections) != set(forecasts):
        raise ValueError("revealed corrections must align with forecast arms")
    if not all(isinstance(text, str) and text.strip() for text in corrections.values()):
        raise ValueError("revealed corrections must contain non-empty text")
    metrics = revealed.get("metrics")
    gains = revealed.get("gain_vs_k0")
    baselines = revealed.get("baseline_metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(forecasts):
        raise ValueError("revealed metrics must align with forecast arms")
    if not isinstance(gains, dict) or set(gains) != set(forecasts):
        raise ValueError("revealed gains must align with forecast arms")
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError("revealed baseline_metrics must be a non-empty mapping")
    if set(baselines) != set(baseline_forecasts):
        raise ValueError("revealed baseline metrics must align with forecast-time baseline forecasts")
    expected_metrics = {
        arm: forecast_metrics(prediction, truth)
        for arm, prediction in forecasts.items()
    }
    for arm, expected in expected_metrics.items():
        actual = metrics[arm]
        if not isinstance(actual, dict) or any(
            not math.isclose(float(actual.get(metric, float("nan"))), value, rel_tol=1e-9, abs_tol=1e-12)
            for metric, value in expected.items()
        ):
            raise ValueError(f"metrics for arm {arm} do not match the recorded forecasts and truth")
    for name, expected in {
        name: forecast_metrics(prediction, truth)
        for name, prediction in baseline_forecasts.items()
    }.items():
        actual = baselines[name]
        if not isinstance(actual, dict) or any(
            not math.isclose(float(actual.get(metric, float("nan"))), value, rel_tol=1e-9, abs_tol=1e-12)
            for metric, value in expected.items()
        ):
            raise ValueError(f"baseline metrics for {name} do not match recorded forecasts and truth")
    control_mae = expected_metrics["0"]["mae"]
    for arm, expected in expected_metrics.items():
        try:
            gain = float(gains[arm])
        except (TypeError, ValueError):
            raise ValueError(f"gain for arm {arm} must be finite") from None
        if not math.isfinite(gain) or not math.isclose(
            gain, control_mae - expected["mae"], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"gain for arm {arm} does not match K=0 MAE difference")
    _json_copy(record.get("provenance"), "provenance")
    return None


def forecast_features(record):
    """Return a detached copy of fields legitimately available at forecast time."""
    validate_scored_case_record(record)
    return _json_copy(record["known_at_forecast"], "known_at_forecast")


def supervision_labels(record):
    """Return labels/outcomes for training only after a forecast is completed."""
    revealed = record.get("revealed_after_forecast") if isinstance(record, dict) else None
    required = ("target_timestamps", "truth", "metrics", "baseline_metrics", "corrections", "gain_vs_k0")
    if not isinstance(revealed, dict) or any(key not in revealed for key in required):
        raise ValueError("supervision requires a completed scored record")
    validate_scored_case_record(record)
    return _json_copy({key: revealed[key] for key in required}, "supervision labels")


def write_scored_case_jsonl(path, records, *, overwrite=False):
    """Atomically write a complete JSONL snapshot; refuse accidental replacement."""
    destination = Path(path)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a non-empty list")
    case_ids = []
    for record in records:
        validate_scored_case_record(record)
        case_ids.append(record["known_at_forecast"]["case_id"])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate case_id in scored-case snapshot")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
                handle.write("\n")
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            # Windows rename fails if the destination already exists, giving a
            # same-volume atomic publication without requiring hard-link rights.
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
