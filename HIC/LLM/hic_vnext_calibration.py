"""Leakage-safe, chronological availability plans for fit-only HIC calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

from LLM import hic
from LLM.hic_grid_runner import _atomic_json, _hash_json, _validate_embedding
from LLM.hic_qdrant_eval import _validated_forecast
from LLM.hic_verified_memory import (
    CORRECTION_PROMPT_VERSION_V3,
    CORRECTION_PROMPT_VERSION_V4,
    CORRECTION_SCHEMA_VERSION,
    CORRECTION_SCHEMA_VERSION_V3,
    CORRECTION_SCHEMA_VERSION_V4,
    FACTS_SCHEMA_VERSION,
    build_verified_correction_prompt,
    build_verified_correction_prompt_v3,
    build_verified_correction_prompt_v4,
    compute_error_facts,
    make_verified_correction_record,
    make_verified_correction_record_v3,
    make_verified_correction_record_v4,
    rank_applicable_memories,
    render_verified_correction,
    render_verified_correction_v3,
    render_verified_correction_v4,
)
from LLM.hic_vnext import (
    apply_bounded_residual,
    build_bounded_residual_prompt,
    build_forecast_behavior_profile,
    build_hic_in_context_prediction_prompt,
    build_paired_direct_forecast_prompt,
    rerank_by_text_curve_and_behavior,
    rerank_by_text_then_curve,
    validate_bounded_residual_response,
)
from LLM.qdrant_store import collection_name_for


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_FIT_CALIBRATION_PROTOCOL = "hic_vnext_prequential_fit_v2"
_FIT_CALIBRATION_PROTOCOL_V3 = "hic_vnext_prequential_fit_v3"
_FIT_CALIBRATION_PROTOCOL_V4 = "hic_vnext_prequential_fit_v4"
_DIRECT_HIC_PROTOCOL = "hic_codex_agent_prequential_direct_hic_v1"
_CALIBRATION_K = (1, 3)
_CALIBRATION_ALPHAS = (0.25, 0.5, 1.0)
_CALIBRATION_BOUNDS = (0.5, 1.0)
_CALIBRATION_DATASETS = ("Agriculture", "Climate", "Economy_sort", "Energy", "SocialGood")
_CALIBRATION_HORIZONS = {
    "Agriculture": (6, 8, 10, 12),
    "Climate": (6, 8, 10, 12),
    "Economy_sort": (6, 8, 10, 12),
    "Energy": (12, 24, 36, 48),
    "SocialGood": (6, 8, 10, 12),
}
_FORBIDDEN_CHECKPOINT_KEYS = {
    "truth", "observed_targets", "target_values", "target_timestamps",
    "test_targets", "outer_test_targets", "test_truth", "outer_test_truth",
}


def _date(value, field, case_id):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required for {case_id}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{field} must be an ISO date for {case_id}") from None


def build_prequential_fit_plan(
    fit_cases,
    *,
    dataset,
    horizon,
    frequency,
    target,
    unit,
):
    """List only completed, strictly earlier fit memories for each pseudo-query.

    The returned plan contains identities and cutoff dates, not series, truth,
    model outputs, or correction text. A query can use a memory only when that
    memory's target period ended before the query's history window began.
    """
    if not isinstance(fit_cases, (list, tuple)):
        raise ValueError("fit_cases must be a list")
    if not fit_cases:
        raise ValueError("fit_cases must not be empty")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be non-empty")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    for name, value in (("frequency", frequency), ("target", target), ("unit", unit)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")

    records = []
    seen_ids = set()
    for index, case in enumerate(fit_cases):
        if not isinstance(case, dict):
            raise ValueError(f"fit case {index} must be a mapping")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"fit case {index} must have a non-empty case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate fit case_id: {case_id}")
        seen_ids.add(case_id)
        if case.get("dataset") != dataset:
            raise ValueError(f"dataset mismatch for {case_id}")
        if case.get("frequency") != frequency:
            raise ValueError(f"frequency mismatch for {case_id}")

        history_start = _date(case.get("history_start"), "history_start", case_id)
        target_end = _date(case.get("target_end"), "target_end", case_id)
        if target_end < history_start:
            raise ValueError(f"target_end precedes history_start for {case_id}")
        has_news = (
            case.get("news_quality") == "valid"
            and isinstance(case.get("news"), str)
            and bool(case["news"].strip())
        )
        records.append({
            "case_id": case_id,
            "history_start": history_start,
            "target_end": target_end,
            "has_news": has_news,
        })

    eligible_records = [record for record in records if record["has_news"]]
    ordered = sorted(
        eligible_records,
        key=lambda record: (record["history_start"], record["case_id"]),
    )
    queries = []
    for query in ordered:
        prior_memories = sorted(
            (
                memory for memory in eligible_records
                if memory["case_id"] != query["case_id"]
                and memory["target_end"] < query["history_start"]
            ),
            key=lambda memory: (memory["target_end"], memory["case_id"]),
        )
        queries.append({
            "query_case_id": query["case_id"],
            "query_history_start": query["history_start"].isoformat(),
            "eligible_memory_case_ids": [item["case_id"] for item in prior_memories],
            "eligible_memory_count": len(prior_memories),
        })

    counts = [item["eligible_memory_count"] for item in queries]
    return {
        "schema_version": 1,
        "task_contract": {
            "dataset": dataset,
            "horizon": horizon,
            "frequency": frequency,
            "target": target,
            "unit": unit,
        },
        "fit_case_count": len(records),
        "eligible_query_count": len(queries),
        "fit_cases_excluded_for_missing_news": len(records) - len(eligible_records),
        "queries_with_at_least_one_prior_memory": sum(count >= 1 for count in counts),
        "queries_with_at_least_three_prior_memories": sum(count >= 3 for count in counts),
        "queries": queries,
    }


def _finite_vector(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite numeric values") from None
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numeric values")
    return result


def _json_copy(value, label):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite JSON-compatible data") from None


def _assert_no_holdout_payload(value, *, where="checkpoint"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_CHECKPOINT_KEYS:
                raise ValueError(f"{where} contains a forbidden holdout/outcome field: {key}")
            _assert_no_holdout_payload(nested, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_holdout_payload(nested, where=f"{where}[{index}]")


def _score_prediction(prediction, truth):
    errors = [float(prediction_value) - float(truth_value)
              for prediction_value, truth_value in zip(prediction, truth)]
    mse = sum(error * error for error in errors) / len(errors)
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def add_applicability_screening(prompt):
    """Add R1's explicit memory-transfer checks without changing output schema."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    return (
        prompt.rstrip()
        + "\n\nR1 HISTORICAL-EXPERIENCE APPLICABILITY CHECK:\n"
        "Before proposing any residual, assess each retrieved experience separately. "
        "Use it only when the target, frequency, and horizon match; the current event "
        "or series shape has a defensible connection to that case; and its documented "
        "error mechanism is relevant to the current baseline. State the supporting "
        "case ID and the specific evidence. Similar wording alone is insufficient. "
        "Do not infer causality or transfer a correction direction from an unrelated "
        "case. If any key applicability check is unclear or memories conflict, abstain "
        "and return zero residuals. Preserve the exact JSON output schema above."
    )


def _variant_key(memory_count, alpha, bound_scale):
    return f"k{memory_count}_a{alpha:g}_c{bound_scale:g}"


def _fit_query_view(case):
    # Never send target-window dates, labels, or outcomes to forecast-time calls.
    excluded = {
        "truth", "target", "label", "labels", "outcome", "outcomes", "y",
        "observed_targets", "target_values", "target_timestamps", "target_start", "target_end",
        "outer_test_targets", "test_targets", "test_truth", "outer_test_truth",
    }
    return _json_copy(
        {key: value for key, value in case.items() if key not in excluded},
        f"forecast-time case {case.get('case_id')}",
    )


def _forecast_temporal_context(case):
    """Expose known time ranges without including target values or truth."""
    fields = {
        "history_period": ("history_start", "history_end"),
        "forecast_period": ("target_start", "target_end"),
    }
    context = {}
    for output_key, input_keys in fields.items():
        values = {"start": case.get(input_keys[0]), "end": case.get(input_keys[1])}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError(f"{output_key} dates are required for {case.get('case_id')}")
        context[output_key] = values
    return context


def _aggregate_prequential_scores(query_records):
    if not query_records:
        return {}
    keys = sorted(query_records[0]["scores"]["variants"])
    output = {}
    for key in keys:
        maes = []
        mses = []
        rmses = []
        wins = ties = fallbacks = 0
        for record in query_records:
            baseline_mae = record["scores"]["baseline"]["mae"]
            variant = record["scores"]["variants"][key]
            maes.append(variant["mae"])
            mses.append(variant["mse"])
            rmses.append(variant["rmse"])
            if variant["mae"] < baseline_mae - 1e-12:
                wins += 1
            elif math.isclose(variant["mae"], baseline_mae, rel_tol=0.0, abs_tol=1e-12):
                ties += 1
            if variant["fallback_reason"] is not None:
                fallbacks += 1
        output[key] = {
            "query_count": len(query_records),
            "mean_case_mae": sum(maes) / len(maes),
            "mean_case_mse": sum(mses) / len(mses),
            "mean_case_rmse": sum(rmses) / len(rmses),
            "mae_wins": wins,
            "mae_ties": ties,
            "mae_losses": len(query_records) - wins - ties,
            "fallback_count": fallbacks,
        }
    return output


def _aggregate_prequential_baseline(query_records):
    if not query_records:
        return {"query_count": 0, "mean_case_mae": None, "mean_case_mse": None, "mean_case_rmse": None}
    count = len(query_records)
    return {
        "query_count": count,
        "mean_case_mae": sum(record["scores"]["baseline"]["mae"] for record in query_records) / count,
        "mean_case_mse": sum(record["scores"]["baseline"]["mse"] for record in query_records) / count,
        "mean_case_rmse": sum(record["scores"]["baseline"]["rmse"] for record in query_records) / count,
    }


def _calibration_variant_keys():
    return {
        _variant_key(memory_count, alpha, bound_scale)
        for memory_count in _CALIBRATION_K
        for alpha in _CALIBRATION_ALPHAS
        for bound_scale in _CALIBRATION_BOUNDS
    }


def select_prequential_calibration_candidate(run_results, *, expected_configurations=None):
    """Choose one global candidate from a frozen fit-only configuration scope.

    By default, require the original 20-configuration matrix. A scoped run may
    pass its frozen configuration allowlist explicitly. Each family is weighted
    equally; within a family, its selected configuration summaries are weighted
    equally. Acceptance requires positive macro-family
    MAE improvement, non-worsening macro-family MSE, and no family MAE regression
    greater than 2%. Among accepted candidates, family-equalized case win rate
    leads, followed by MAE improvement, fewer memories, and smaller alpha/bound.
    """
    if not isinstance(run_results, (list, tuple)) or not run_results:
        raise ValueError("fit calibration selection requires a non-empty result list")

    full_scope = {
        (dataset, horizon)
        for dataset, horizons in _CALIBRATION_HORIZONS.items()
        for horizon in horizons
    }
    if expected_configurations is None:
        if len(run_results) != len(full_scope):
            raise ValueError("fit calibration selection requires exactly 20 configuration results")
        expected_scope = full_scope
    else:
        if not isinstance(expected_configurations, (list, tuple)) or not expected_configurations:
            raise ValueError("expected_configurations must be a non-empty frozen scope")
        expected_scope = set()
        for config in expected_configurations:
            if isinstance(config, dict):
                dataset, horizon = config.get("dataset"), config.get("horizon")
            elif isinstance(config, (list, tuple)) and len(config) == 2:
                dataset, horizon = config
            else:
                raise ValueError("each expected configuration must identify dataset and horizon")
            identity = (dataset, horizon)
            if identity not in full_scope:
                raise ValueError(f"expected configuration is outside the frozen matrix: {identity}")
            if identity in expected_scope:
                raise ValueError(f"duplicate expected configuration: {identity}")
            expected_scope.add(identity)
        if {dataset for dataset, _ in expected_scope} != set(_CALIBRATION_DATASETS):
            raise ValueError("the frozen calibration scope must represent all five dataset families")

    expected_variants = _calibration_variant_keys()
    by_dataset = {dataset: [] for dataset in _CALIBRATION_DATASETS}
    seen_configs = set()
    for result in run_results:
        if not isinstance(result, dict):
            raise ValueError("each configuration result must be a mapping")
        dataset = result.get("dataset")
        horizon = result.get("horizon")
        if dataset not in by_dataset:
            raise ValueError(f"unexpected fit calibration dataset: {dataset}")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("each configuration result must include a positive integer horizon")
        config_id = (dataset, horizon)
        if config_id in seen_configs:
            raise ValueError(f"duplicate fit calibration configuration: {config_id}")
        seen_configs.add(config_id)

        baseline = result.get("baseline_aggregate")
        variants = result.get("variant_aggregates")
        if not isinstance(baseline, dict) or not isinstance(variants, dict):
            raise ValueError(f"configuration {config_id} is missing calibration aggregates")
        if set(variants) != expected_variants:
            raise ValueError(f"configuration {config_id} does not contain the complete 12-variant grid")
        query_count = baseline.get("query_count")
        if type(query_count) is not int or query_count <= 0:
            raise ValueError(f"configuration {config_id} has an invalid baseline query_count")
        row = {"baseline": {}, "variants": {}}
        for metric in ("mean_case_mae", "mean_case_mse"):
            value = baseline.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"configuration {config_id} has invalid baseline {metric}")
            row["baseline"][metric] = float(value)
        row["query_count"] = query_count

        for key, aggregate in variants.items():
            if not isinstance(aggregate, dict) or aggregate.get("query_count") != query_count:
                raise ValueError(f"configuration {config_id} has inconsistent query counts for {key}")
            metrics = {}
            for metric in ("mean_case_mae", "mean_case_mse"):
                value = aggregate.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                    raise ValueError(f"configuration {config_id} has invalid {metric} for {key}")
                metrics[metric] = float(value)
            wins = aggregate.get("mae_wins")
            ties = aggregate.get("mae_ties")
            losses = aggregate.get("mae_losses")
            if any(type(value) is not int or value < 0 for value in (wins, ties, losses)):
                raise ValueError(f"configuration {config_id} has invalid case outcomes for {key}")
            if wins + ties + losses != query_count:
                raise ValueError(f"configuration {config_id} has incomplete case outcomes for {key}")
            metrics["win_rate"] = wins / query_count
            row["variants"][key] = metrics
        by_dataset[dataset].append(row)

    if seen_configs != expected_scope:
        if expected_configurations is None:
            raise ValueError("fit calibration results must match the frozen 20-configuration horizon matrix")
        raise ValueError("fit calibration results do not match the frozen expected configuration scope")

    candidate_metrics = {}
    for key in sorted(expected_variants):
        family_metrics = {}
        family_mae_improvements = []
        family_mse_improvements = []
        family_win_rates = []
        rejection_reasons = []
        for dataset, rows in by_dataset.items():
            baseline_mae = sum(row["baseline"]["mean_case_mae"] for row in rows) / len(rows)
            baseline_mse = sum(row["baseline"]["mean_case_mse"] for row in rows) / len(rows)
            variant_mae = sum(row["variants"][key]["mean_case_mae"] for row in rows) / len(rows)
            variant_mse = sum(row["variants"][key]["mean_case_mse"] for row in rows) / len(rows)
            win_rate = sum(row["variants"][key]["win_rate"] for row in rows) / len(rows)
            mae_change = (variant_mae - baseline_mae) / baseline_mae
            mse_improvement = (baseline_mse - variant_mse) / baseline_mse
            family_mae_improvements.append(-mae_change)
            family_mse_improvements.append(mse_improvement)
            family_win_rates.append(win_rate)
            family_metrics[dataset] = {
                "configuration_count": len(rows),
                "baseline_mae": baseline_mae,
                "variant_mae": variant_mae,
                "mae_relative_change": mae_change,
                "baseline_mse": baseline_mse,
                "variant_mse": variant_mse,
                "mse_relative_improvement": mse_improvement,
                "case_win_rate": win_rate,
            }
            if mae_change > 0.02 + 1e-12:
                rejection_reasons.append(f"{dataset}:family_mae_regression_exceeds_2_percent")

        macro_mae_improvement = sum(family_mae_improvements) / len(family_mae_improvements)
        macro_mse_improvement = sum(family_mse_improvements) / len(family_mse_improvements)
        macro_win_rate = sum(family_win_rates) / len(family_win_rates)
        if macro_mae_improvement <= 1e-12:
            rejection_reasons.append("macro_family_mae_not_improved")
        if macro_mse_improvement < -1e-12:
            rejection_reasons.append("macro_family_mse_worsened")
        parameters = key.split("_")
        candidate_metrics[key] = {
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "macro_family_mae_improvement": macro_mae_improvement,
            "macro_family_mse_improvement": macro_mse_improvement,
            "macro_family_case_win_rate": macro_win_rate,
            "family_metrics": family_metrics,
            "memory_count": int(parameters[0][1:]),
            "alpha": float(parameters[1][1:]),
            "bound_scale": float(parameters[2][1:]),
        }

    eligible_variants = [key for key, metrics in candidate_metrics.items() if metrics["eligible"]]
    selected_variant = None
    if eligible_variants:
        selected_variant = min(
            eligible_variants,
            key=lambda key: (
                -candidate_metrics[key]["macro_family_case_win_rate"],
                -candidate_metrics[key]["macro_family_mae_improvement"],
                candidate_metrics[key]["memory_count"],
                candidate_metrics[key]["alpha"],
                candidate_metrics[key]["bound_scale"],
            ),
        )
    return {
        "protocol": "hic_vnext_fit_candidate_selection_v1",
        "configuration_count": len(seen_configs),
        "family_count": sum(bool(rows) for rows in by_dataset.values()),
        "dataset_families": [dataset for dataset in _CALIBRATION_DATASETS if by_dataset[dataset]],
        "expected_configurations": [
            {"dataset": dataset, "horizon": horizon}
            for dataset, horizon in sorted(expected_scope)
        ],
        "candidate_count": len(candidate_metrics),
        "eligible_variants": eligible_variants,
        "selected_variant": selected_variant,
        "candidate_metrics": candidate_metrics,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
    }


def run_prequential_fit_calibration(
    fit_cases,
    *,
    dataset,
    horizon,
    frequency,
    target,
    unit,
    source_sha256,
    run_id,
    template,
    dataset_description,
    request_cache,
    call_json,
    embed_summary,
    store,
    checkpoint_path,
    retry_unresolved=False,
    correction_schema_version=CORRECTION_SCHEMA_VERSION,
    collection_scope="default",
    collection_name_override=None,
    correction_mode="r0",
    task_id_namespace=None,
    inference_mode="bounded_residual",
    memory_version=None,
    initial_completed_records=None,
    initial_memory_records=None,
    initial_memory_source_sha256=None,
    prior_correction_schema_versions=None,
    execution_metadata=None,
    max_new_queries=None,
    checkpoint_manifest_migration=None,
):
    """Run a resumable, strictly prequential fit-only HIC calibration pass.

    Each forecast is persisted before the fit truth is read. That truth is then
    used only to score the fit pseudo-query and build its completed memory for
    later queries. API, embedding, and Qdrant dependencies are injected; this
    function makes no network call by itself.
    """
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be non-empty")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    for name, value in (("frequency", frequency), ("target", target), ("unit", unit),
                        ("run_id", run_id), ("dataset_description", dataset_description)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty JSON object")
    _json_copy(template, "template")
    if type(retry_unresolved) is not bool:
        raise ValueError("retry_unresolved must be a boolean")
    if max_new_queries is not None and (type(max_new_queries) is not int or max_new_queries <= 0):
        raise ValueError("max_new_queries must be a positive integer when provided")
    if correction_mode not in {"r0", "r1"}:
        raise ValueError("correction_mode must be 'r0' or 'r1'")
    if inference_mode not in {
        "bounded_residual",
        "direct_hic_context",
        "direct_hic_context_v2",
        "direct_hic_context_v3",
    }:
        raise ValueError("unsupported inference_mode")
    if task_id_namespace is not None and (
        not isinstance(task_id_namespace, str) or not task_id_namespace.strip()
    ):
        raise ValueError("task_id_namespace must be non-empty text when provided")
    if memory_version is not None and (
        not isinstance(memory_version, str) or not memory_version.strip()
    ):
        raise ValueError("memory_version must be non-empty text when provided")
    if type(correction_schema_version) is not int or correction_schema_version not in {
        CORRECTION_SCHEMA_VERSION, CORRECTION_SCHEMA_VERSION_V3,
        CORRECTION_SCHEMA_VERSION_V4,
    }:
        raise ValueError("correction_schema_version must be 2, 3, or 4")
    if prior_correction_schema_versions is None:
        prior_correction_schema_versions = [correction_schema_version]
    if (
        not isinstance(prior_correction_schema_versions, (list, tuple))
        or not prior_correction_schema_versions
        or any(
            type(version) is not int
            or version not in {
                CORRECTION_SCHEMA_VERSION, CORRECTION_SCHEMA_VERSION_V3,
                CORRECTION_SCHEMA_VERSION_V4,
            }
            for version in prior_correction_schema_versions
        )
        or len(set(prior_correction_schema_versions)) != len(prior_correction_schema_versions)
    ):
        raise ValueError("prior_correction_schema_versions must contain supported unique schema versions")
    prior_correction_schema_versions = list(prior_correction_schema_versions)
    if initial_memory_records is None:
        seed_memory_records = {}
    else:
        if not isinstance(initial_memory_records, dict):
            raise ValueError("initial_memory_records must be a dictionary")
        seed_memory_records = _json_copy(initial_memory_records, "initial prior memory records")
        if not seed_memory_records:
            raise ValueError("initial_memory_records must not be empty when provided")
        if not isinstance(initial_memory_source_sha256, str) or not _SHA256_RE.fullmatch(
            initial_memory_source_sha256
        ):
            raise ValueError("initial_memory_source_sha256 must be a lowercase SHA-256 digest")
        for seed_id, memory in seed_memory_records.items():
            metadata = memory.get("metadata") if isinstance(memory, dict) else None
            if (
                not isinstance(seed_id, str)
                or not seed_id.strip()
                or not isinstance(memory, dict)
                or memory.get("case_id") != seed_id
                or not isinstance(memory.get("summary"), dict)
                or not isinstance(memory.get("correction"), str)
                or not memory.get("correction", "").strip()
                or not isinstance(metadata, dict)
                or metadata.get("dataset") != dataset
                or metadata.get("horizon") != horizon
                or metadata.get("frequency") != frequency
                or metadata.get("target") != target
                or metadata.get("unit") != unit
                or metadata.get("source_sha256") != source_sha256
                or metadata.get("memory_version") != memory_version
                or metadata.get("seeded_source_memory") is not True
                or metadata.get("completed") is not True
                or metadata.get("split") != "fit"
                or metadata.get("correction_schema_version") not in prior_correction_schema_versions
            ):
                raise ValueError(f"invalid initial prior memory record: {seed_id}")
            _validate_embedding(memory.get("embedding"), 1024, seed_id)
    prior_prompt_versions_by_schema = {
        str(CORRECTION_SCHEMA_VERSION): [None],
        str(CORRECTION_SCHEMA_VERSION_V3): [CORRECTION_PROMPT_VERSION_V3],
        str(CORRECTION_SCHEMA_VERSION_V4): [CORRECTION_PROMPT_VERSION_V4],
    }
    if inference_mode == "direct_hic_context_v3" and correction_schema_version != CORRECTION_SCHEMA_VERSION_V4:
        raise ValueError("direct_hic_context_v3 requires correction_schema_version=4")
    if collection_scope not in {"default", "scope16", "codex_agent_scope16"}:
        raise ValueError(
            "collection_scope must be 'default', 'scope16', or 'codex_agent_scope16'"
        )
    protocol = (
        _DIRECT_HIC_PROTOCOL
        if inference_mode.startswith("direct_hic_context")
        else (
            _FIT_CALIBRATION_PROTOCOL
            if correction_schema_version == CORRECTION_SCHEMA_VERSION
            else (
                _FIT_CALIBRATION_PROTOCOL_V3
                if correction_schema_version == CORRECTION_SCHEMA_VERSION_V3
                else _FIT_CALIBRATION_PROTOCOL_V4
            )
        )
    )
    correction_prompt_version = (
        None
        if correction_schema_version == CORRECTION_SCHEMA_VERSION
        else (
            CORRECTION_PROMPT_VERSION_V3
            if correction_schema_version == CORRECTION_SCHEMA_VERSION_V3
            else CORRECTION_PROMPT_VERSION_V4
        )
    )
    if not hasattr(request_cache, "invoke") or not hasattr(request_cache, "mark_persisted"):
        raise ValueError("request_cache must provide invoke and mark_persisted")
    if not callable(call_json) or not callable(embed_summary):
        raise ValueError("call_json and embed_summary must be callable")
    if not all(hasattr(store, name) for name in ("ensure_collection", "count", "search_many", "upsert_completed_case")):
        raise ValueError("store does not implement the required Qdrant HIC interface")
    if collection_name_override is None:
        collection_generation = (
            "vnext"
            if correction_schema_version == CORRECTION_SCHEMA_VERSION
            else (
                "vnext_v3"
                if correction_schema_version == CORRECTION_SCHEMA_VERSION_V3
                else "vnext_v4"
            )
        )
        if collection_scope != "default":
            collection_generation = f"{collection_generation}_{collection_scope}"
        expected_collection = collection_name_for(
            "BAAI/bge-m3", suffix=f"{collection_generation}_{dataset.lower()}_h{horizon}"
        )
    else:
        if (
            not isinstance(collection_name_override, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,254}", collection_name_override)
        ):
            raise ValueError("collection_name_override must be a valid lowercase Qdrant name")
        expected_collection = collection_name_override
    if getattr(store, "model_id", "BAAI/bge-m3") != "BAAI/bge-m3":
        raise ValueError("prequential fit calibration requires BAAI/bge-m3 embeddings")
    actual_collection = getattr(store, "collection_name", expected_collection)
    if actual_collection != expected_collection:
        raise ValueError(
            f"isolated collection mismatch: expected {expected_collection}, got {actual_collection}"
        )

    plan = build_prequential_fit_plan(
        fit_cases,
        dataset=dataset,
        horizon=horizon,
        frequency=frequency,
        target=target,
        unit=unit,
    )
    case_by_id = {case["case_id"]: case for case in fit_cases}
    forecast_input_identity = []
    history_length = None
    for case_id in (item["query_case_id"] for item in plan["queries"]):
        case = case_by_id[case_id]
        series = _finite_vector(case.get("series"), f"history for {case_id}")
        if len(series) < 2:
            raise ValueError(f"history for {case_id} must contain at least two values")
        if history_length is None:
            history_length = len(series)
        elif len(series) != history_length:
            raise ValueError(
                "all fit query histories in one dataset/horizon run must have the same length"
            )
        forecast_input_identity.append({
            "case_id": case_id,
            "temporal_contract": {
                key: case.get(key)
                for key in ("history_start", "history_end", "target_start", "target_end")
            },
            "forecast_view": _fit_query_view(case),
        })
    if history_length is None:
        raise ValueError("no fit queries with valid news are available for prequential calibration")
    forecast_inputs_sha256 = _hash_json(
        _json_copy(forecast_input_identity, "fit forecast-input identity")
    )
    seed_records = {}
    if initial_completed_records is not None:
        if not isinstance(initial_completed_records, dict):
            raise ValueError("initial_completed_records must be a dictionary")
        seed_records = _json_copy(initial_completed_records, "completed seed records")
        ordered_ids = [item["query_case_id"] for item in plan["queries"]]
        if set(seed_records) != set(ordered_ids[:len(seed_records)]):
            raise ValueError("completed seed records must be a chronological query prefix")
        for seed_id, seeded in seed_records.items():
            metadata = (seeded.get("memory") or {}).get("metadata") or {}
            if (
                seeded.get("status") != "complete"
                or seeded.get("ingested") is not True
                or not {"forecast", "memory", "scores"}.issubset(seeded)
                or (seeded.get("memory") or {}).get("case_id") != seed_id
                or metadata.get("dataset") != dataset
                or metadata.get("horizon") != horizon
                or metadata.get("source_sha256") != source_sha256
            ):
                raise ValueError(f"invalid completed seed record: {seed_id}")
    if execution_metadata is not None:
        if not isinstance(execution_metadata, dict) or not execution_metadata:
            raise ValueError("execution_metadata must be a non-empty dictionary")
        execution_metadata = _json_copy(execution_metadata, "execution metadata")
    manifest = {
        "protocol": protocol,
        "run_id": run_id,
        "dataset": dataset,
        "horizon": horizon,
        "frequency": frequency,
        "target": target,
        "unit": unit,
        "history_length": history_length,
        "source_sha256": source_sha256,
        "forecast_inputs_sha256": forecast_inputs_sha256,
        "template_sha256": hashlib.sha256(
            json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "dataset_description_sha256": hashlib.sha256(dataset_description.encode("utf-8")).hexdigest(),
        "request_settings": getattr(request_cache, "request_settings", None),
        "embedding_model": "BAAI/bge-m3",
        "inference_mode": inference_mode,
        "collection_name": expected_collection,
        "prior_correction_schema_versions": prior_correction_schema_versions,
        "plan": plan,
    }
    if memory_version is not None:
        manifest["memory_version"] = memory_version
    if initial_completed_records is not None:
        manifest["seed_completed_records_sha256"] = _hash_json(seed_records)
    if initial_memory_records is not None:
        manifest["seed_memory_records_sha256"] = _hash_json(seed_memory_records)
        manifest["seed_memory_source_sha256"] = initial_memory_source_sha256
    if execution_metadata is not None:
        manifest["execution_metadata"] = execution_metadata
    if correction_mode != "r0":
        manifest["correction_mode"] = correction_mode
    if task_id_namespace is not None and task_id_namespace != run_id:
        manifest["task_id_namespace"] = task_id_namespace
    if correction_schema_version != CORRECTION_SCHEMA_VERSION:
        manifest["correction_schema_version"] = correction_schema_version
        manifest["correction_prompt_version"] = correction_prompt_version
    try:
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        raise ValueError("request cache settings must be finite JSON-compatible data") from None

    path = Path(checkpoint_path)
    query_ids = [item["query_case_id"] for item in plan["queries"]]
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != 1
            or state.get("protocol") != protocol
            or set(state.get("queries", {})) != set(query_ids)
            or (
                memory_version is not None
                and state.get("memory_version") != memory_version
            )
            or (
                correction_schema_version != CORRECTION_SCHEMA_VERSION
                and (
                    state.get("correction_schema_version") != correction_schema_version
                    or state.get("correction_prompt_version") != correction_prompt_version
                )
            )
        ):
            raise ValueError("prequential checkpoint manifest does not match the current fit plan")
        if state.get("manifest_sha256") != manifest_sha256:
            migration = checkpoint_manifest_migration
            prior_manifest = dict(manifest)
            collection_rename = (
                state.get("collection_name") == "agent_test_3_hic_vnext"
                and expected_collection == "agent_test_5"
                and manifest.get("collection_name") == "agent_test_5"
            )
            if collection_rename:
                prior_manifest["collection_name"] = "agent_test_3_hic_vnext"

            prior_description_hash = None
            if migration is not None:
                required_migration_fields = {
                    "migration_id", "dataset", "horizon", "case_id",
                    "previous_description", "reason", "evidence_path",
                }
                if (
                    not isinstance(migration, dict)
                    or set(migration) != required_migration_fields
                    or migration.get("dataset") != dataset
                    or migration.get("horizon") != horizon
                    or not all(
                        isinstance(migration.get(key), str) and migration[key].strip()
                        for key in ("migration_id", "case_id", "previous_description", "reason", "evidence_path")
                    )
                    or migration["case_id"] not in state["queries"]
                    or state["queries"][migration["case_id"]].get("status") == "complete"
                    or migration["previous_description"] == dataset_description
                ):
                    raise ValueError("prequential checkpoint manifest does not match the current fit plan")
                prior_description_hash = hashlib.sha256(
                    migration["previous_description"].encode("utf-8")
                ).hexdigest()
                prior_manifest["dataset_description_sha256"] = prior_description_hash

            if (
                not collection_rename and migration is None
            ) or (
                migration is not None
                and manifest["dataset_description_sha256"]
                != hashlib.sha256(dataset_description.encode("utf-8")).hexdigest()
            ) or _hash_json(prior_manifest) != state.get("manifest_sha256"):
                raise ValueError("checkpoint manifest migration does not match the saved manifest")
            migrations = state.get("manifest_migrations", [])
            if not isinstance(migrations, list):
                raise ValueError("checkpoint manifest migration history is invalid")
            if collection_rename:
                migrations.append({
                    "migration_id": "qdrant_agent_test_5_physical_rename_20261028",
                    "changed_field": "collection_name",
                    "previous_collection_name": "agent_test_3_hic_vnext",
                    "current_collection_name": "agent_test_5",
                    "previous_manifest_sha256": state["manifest_sha256"],
                    "current_manifest_sha256": manifest_sha256,
                    "reason": "Resume against the renamed physical collection while preserving its frozen point IDs.",
                    "evidence_path": "docs/history/hic_test4_test5_rename_recovery_20261028.md",
                    "migrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                })
            if migration is not None:
                migrations.append({
                    "migration_id": migration["migration_id"],
                    "dataset": dataset,
                    "horizon": horizon,
                    "case_id": migration["case_id"],
                    "changed_field": "dataset_description_sha256",
                    "previous_manifest_sha256": state["manifest_sha256"],
                    "current_manifest_sha256": manifest_sha256,
                    "previous_description_sha256": prior_description_hash,
                    "current_description_sha256": manifest["dataset_description_sha256"],
                    "reason": migration["reason"],
                    "evidence_path": migration["evidence_path"],
                    "migrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                })
            state["manifest_sha256"] = manifest_sha256
            state["collection_name"] = expected_collection
            state["manifest_migrations"] = migrations
    else:
        if checkpoint_manifest_migration is not None:
            raise ValueError("checkpoint manifest migration requires an existing checkpoint")
        state = {
            "schema_version": 1,
            "protocol": protocol,
            "manifest_sha256": manifest_sha256,
            "outer_test_targets_read": False,
            "validation_truth_read": False,
            "dataset": dataset,
            "horizon": horizon,
            "run_id": run_id,
            "inference_mode": inference_mode,
            "collection_name": expected_collection,
            "history_length": history_length,
            "queries": {
                case_id: seed_records.get(case_id, {"status": "planned"})
                for case_id in query_ids
            },
        }
        if execution_metadata is not None:
            state["execution_metadata"] = execution_metadata
        if correction_schema_version != CORRECTION_SCHEMA_VERSION:
            state["correction_schema_version"] = correction_schema_version
            state["correction_prompt_version"] = correction_prompt_version
        if memory_version is not None:
            state["memory_version"] = memory_version

    _assert_no_holdout_payload(state)
    store.ensure_collection()
    persisted_count = sum(
        record.get("ingested") is True for record in state["queries"].values()
    )
    expected_store_case_ids = set(seed_memory_records) | {
        case_id
        for case_id, record in state["queries"].items()
        if record.get("ingested") is True
    }
    if store.count() != len(expected_store_case_ids):
        raise ValueError("isolated Qdrant point count does not match the resumable calibration checkpoint")
    _atomic_json(path, state)

    def save():
        _assert_no_holdout_payload(state)
        _atomic_json(path, state)

    def invoke(case_id, stage, prompt, validator, commit):
        task_namespace = task_id_namespace or run_id
        result = request_cache.invoke(
            f"{task_namespace}:{case_id}:{stage}",
            prompt,
            call_json=call_json,
            validator=validator,
            retry_unresolved=retry_unresolved,
        )
        identity = result["request_identity"]
        commit(result["response"], identity)
        save()
        request_cache.mark_persisted(identity["sha256"])

    prior_memories = dict(seed_memory_records)
    newly_completed_queries = 0
    for item in plan["queries"]:
        case_id = item["query_case_id"]
        case = case_by_id[case_id]
        query_view = _fit_query_view(case)
        record = state["queries"][case_id]
        was_complete_before_call = record.get("status") == "complete" and record.get("ingested") is True
        if case_id in seed_records:
            # Frozen histories have already completed every semantic stage.
            # Rewriting the whole checkpoint twice per old record on each
            # resumed task would multiply disk I/O by the history size.
            if record != seed_records[case_id]:
                raise ValueError(f"completed seed record changed: {case_id}")
            known_truth = _finite_vector(case["truth"], f"seed fit truth for {case_id}", length=horizon)
            if _hash_json(known_truth) != record.get("fit_truth_sha256"):
                raise ValueError(f"fit truth changed since saved seed scores for {case_id}")
            prior_memories[case_id] = record["memory"]
            continue
        forecast_state = record.setdefault("forecast", {})
        allowed_ids = item["eligible_memory_case_ids"]
        missing_prior = [memory_id for memory_id in allowed_ids if memory_id not in prior_memories]
        if missing_prior:
            raise ValueError(
                f"prequential order reached {case_id} before earlier memories were completed: {missing_prior}"
            )

        if "summary" not in forecast_state:
            prompt = hic.build_summary_prompt(
                template,
                query_view["series"],
                query_view["news"],
                dataset_description=dataset_description,
                covariates=query_view.get("covariates"),
                news_quality_detail=query_view.get("news_quality_detail"),
                canonical_json_inputs=True,
                target=target,
                unit=unit,
                frequency=frequency,
                horizon=horizon,
            )

            def commit_summary(response, identity):
                if not isinstance(response, dict) or not response:
                    raise ValueError("summary response must be a non-empty JSON object")
                forecast_state["summary"] = _json_copy(response, "summary response")
                forecast_state["summary_request_sha256"] = identity["sha256"]
                forecast_state["summary_content_sha256"] = _hash_json(response)

            record["status"] = "forecast_running"
            save()
            invoke(
                case_id,
                "summary_vnext",
                prompt,
                lambda response: isinstance(response, dict) and bool(response),
                commit_summary,
            )

        if "baseline" not in forecast_state:
            if inference_mode == "direct_hic_context_v3":
                prompt = build_paired_direct_forecast_prompt(
                    forecast_state["summary"],
                    query_view["series"],
                    horizon=horizon,
                    frequency=frequency,
                    target=target,
                    unit=unit,
                    covariates=query_view.get("covariates"),
                    dataset_description=dataset_description,
                    temporal_context=_forecast_temporal_context(case),
                )
            else:
                prompt = hic.build_reasoning_prompt(
                    forecast_state["summary"],
                    query_view["series"],
                    covariates=query_view.get("covariates"),
                    dataset_description=dataset_description,
                    horizon=horizon,
                    frequency=frequency,
                    canonical_json_inputs=True,
                )

            def commit_baseline(response, identity):
                parsed = _validated_forecast(response, horizon)
                forecast_state["baseline"] = {
                    "prediction": parsed["prediction"],
                    "reasoning": parsed["reasoning"],
                    "request_sha256": identity["sha256"],
                    "prediction_sha256": _hash_json(parsed["prediction"]),
                }

            invoke(
                case_id,
                "baseline_no_retrieval_vnext",
                prompt,
                lambda response: _validated_forecast(response, horizon),
                commit_baseline,
            )

        if "query_embedding" not in forecast_state:
            forecast_state["query_embedding"] = _validate_embedding(
                embed_summary(forecast_state["summary"], is_query=True), 1024, case_id
            )
            save()

        if inference_mode == "direct_hic_context_v3" and "baseline_behavior_profile" not in forecast_state:
            forecast_state["baseline_behavior_profile"] = build_forecast_behavior_profile(
                case["series"], forecast_state["baseline"]["prediction"]
            )
            save()

        if "retrieval_trace" not in forecast_state:
            if not allowed_ids:
                ranked_memories = []
            else:
                all_candidates = store.search_many(
                    forecast_state["query_embedding"], limit=max(1, store.count())
                )
                if not isinstance(all_candidates, list):
                    raise ValueError("Qdrant search_many must return a list")
                candidate_ids = [candidate.get("case_id") for candidate in all_candidates]
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise ValueError("Qdrant returned duplicate memory IDs")
                if not set(allowed_ids).issubset(set(candidate_ids)):
                    raise ValueError("Qdrant is missing a strictly earlier completed fit memory")
                candidate_pool = [
                    candidate for candidate in all_candidates
                    if candidate.get("case_id") in set(allowed_ids)
                ]
                for candidate in candidate_pool:
                    metadata = candidate.get("metadata") or {}
                    if _date(metadata.get("target_end"), "memory.target_end", candidate["case_id"]) >= _date(
                        case["history_start"], "query.history_start", case_id
                    ):
                        raise ValueError("Qdrant returned a memory whose outcome was not complete before query history")
                    if metadata.get("summary_prompt_version") != hic.SUMMARY_PROMPT_VERSION:
                        raise ValueError("vNext collection contains a memory from another summary prompt version")
                    if (
                        memory_version is not None
                        and metadata.get("memory_version") != memory_version
                    ):
                        raise ValueError("Qdrant returned a memory from another memory version")
                    if metadata.get("correction_schema_version") not in prior_correction_schema_versions:
                        raise ValueError("vNext collection contains a memory from an unsupported correction schema")
                    candidate["history_series"] = metadata.get("history_series")
                    if inference_mode == "direct_hic_context_v3":
                        candidate["baseline_behavior_profile"] = metadata.get(
                            "baseline_behavior_profile"
                        )
                query_metadata = {
                    "case_id": case_id,
                    "dataset": dataset,
                    "source_sha256": source_sha256,
                    "target": target,
                    "unit": unit,
                    "frequency": frequency,
                    "horizon": horizon,
                    "history_end": case["history_end"],
                    "news_quality": "valid",
                    "required_facts_schema_version": FACTS_SCHEMA_VERSION,
                    "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
                    "required_correction_schema_versions": prior_correction_schema_versions,
                    "required_correction_prompt_versions_by_schema": {
                        str(version): prior_prompt_versions_by_schema[str(version)]
                        for version in prior_correction_schema_versions
                    },
                }
                if memory_version is not None:
                    query_metadata["required_memory_version"] = memory_version
                if correction_prompt_version is not None:
                    query_metadata["required_correction_prompt_version"] = correction_prompt_version
                filtered = rank_applicable_memories(
                    candidate_pool,
                    query_metadata,
                    require_horizon_text_consistency=True,
                )
                if inference_mode == "direct_hic_context_v2":
                    ranked_memories = rerank_by_text_then_curve(
                        filtered["eligible"], case["series"], text_top_k=5
                    )[:1]
                elif inference_mode == "direct_hic_context_v3":
                    ranked_memories = rerank_by_text_curve_and_behavior(
                        filtered["eligible"],
                        case["series"],
                        forecast_state["baseline"]["prediction"],
                        text_top_k=5,
                    )[:1]
                elif inference_mode == "direct_hic_context":
                    # VoT HIC retrieves one nearest summary embedding and passes
                    # its correction into the Reasoner prompt as context.
                    ranked_memories = sorted(
                        filtered["eligible"],
                        key=lambda memory: (-float(memory["score"]), memory["case_id"]),
                    )[:1]
                else:
                    ranked_memories = rerank_by_text_then_curve(
                        filtered["eligible"], case["series"], text_top_k=5
                    )[:3]
            forecast_state["retrieval_trace"] = []
            for memory in ranked_memories:
                trace = {
                    "case_id": memory["case_id"],
                    "cosine_score": memory["score"],
                    "text_similarity": memory.get("text_similarity", (float(memory["score"]) + 1.0) / 2.0),
                    "curve_similarity": memory.get("curve_similarity"),
                    "combined_score": memory.get("combined_score", float(memory["score"])),
                }
                if inference_mode == "direct_hic_context_v3":
                    trace.update({
                        "behavior_similarity": memory["behavior_similarity"],
                        "query_behavior_profile": memory["query_behavior_profile"],
                        "selection_reason": memory["selection_reason"],
                    })
                forecast_state["retrieval_trace"].append(trace)
            save()

        trace_by_id = {row["case_id"]: row for row in forecast_state["retrieval_trace"]}
        eligible_ranked = []
        for case_id_from_trace, trace in trace_by_id.items():
            if case_id_from_trace not in set(allowed_ids):
                raise ValueError("retrieval trace contains a memory outside the prequential plan")
            # Rehydrate prompt-safe memory fields from the already checkpointed record.
            memory = prior_memories[case_id_from_trace]
            eligible_ranked.append({
                "case_id": case_id_from_trace,
                "summary": memory["summary"],
                "correction": memory["correction"],
                "score": trace["cosine_score"],
                "text_similarity": trace["text_similarity"],
                "curve_similarity": trace["curve_similarity"],
                "combined_score": trace["combined_score"],
            })
            if inference_mode == "direct_hic_context_v3":
                eligible_ranked[-1]["baseline_behavior_profile"] = memory["metadata"].get(
                    "baseline_behavior_profile"
                )
                eligible_ranked[-1]["behavior_similarity"] = trace["behavior_similarity"]
                eligible_ranked[-1]["selection_reason"] = trace["selection_reason"]
                eligible_ranked[-1]["query_behavior_profile"] = trace[
                    "query_behavior_profile"
                ]
        eligible_ranked.sort(key=lambda row: (-row["combined_score"], row["case_id"]))

        if inference_mode.startswith("direct_hic_context"):
            if "hic" not in forecast_state:
                if not eligible_ranked:
                    forecast_state["hic"] = {
                        **forecast_state["baseline"],
                        "selected_case_ids": [],
                        "fallback_reason": "no_prior_memory",
                        "request_sha256": forecast_state["baseline"]["request_sha256"],
                    }
                else:
                    selected = eligible_ranked[0]
                    if inference_mode == "direct_hic_context_v3":
                        prompt = build_paired_direct_forecast_prompt(
                            forecast_state["summary"],
                            query_view["series"],
                            horizon=horizon,
                            frequency=frequency,
                            target=target,
                            unit=unit,
                            covariates=query_view.get("covariates"),
                            dataset_description=dataset_description,
                            temporal_context=_forecast_temporal_context(case),
                            retrieved_memory=selected,
                        )
                    else:
                        prompt = build_hic_in_context_prediction_prompt(
                            forecast_state["summary"],
                            case["series"],
                            selected,
                            horizon=horizon,
                            frequency=frequency,
                            covariates=query_view.get("covariates"),
                            dataset_description=dataset_description,
                        )

                    def commit_hic_forecast(response, identity):
                        parsed = _validated_forecast(response, horizon)
                        forecast_state["hic"] = {
                            "prediction": parsed["prediction"],
                            "reasoning": parsed["reasoning"],
                            "selected_case_ids": [selected["case_id"]],
                            "fallback_reason": None,
                            "request_sha256": identity["sha256"],
                        }

                    invoke(
                        case_id,
                        (
                            "hic_forecast_with_retrieved_context_v2"
                            if inference_mode == "direct_hic_context_v2"
                            else (
                                "hic_forecast_with_retrieved_context_v3"
                                if inference_mode == "direct_hic_context_v3"
                                else "hic_forecast_with_retrieved_context_v1"
                            )
                        ),
                        prompt,
                        lambda response: _validated_forecast(response, horizon),
                        commit_hic_forecast,
                    )
        else:
            residual_state = forecast_state.setdefault("residuals_by_k", {})
            for memory_count in _CALIBRATION_K:
                key = str(memory_count)
                if key in residual_state:
                    continue
                selected = eligible_ranked[:memory_count]
                if not selected:
                    residual_state[key] = {
                        "residual": [0.0] * horizon,
                        "reasoning": "No strictly earlier completed memory was available.",
                        "abstain": True,
                        "selected_case_ids": [],
                        "request_sha256": None,
                    }
                    save()
                    continue
                prompt = build_bounded_residual_prompt(
                    forecast_state["summary"],
                    case["series"],
                    forecast_state["baseline"]["prediction"],
                    selected,
                    horizon=horizon,
                    frequency=frequency,
                )
                if correction_mode == "r1":
                    prompt = add_applicability_screening(prompt)

                def commit_residual(response, identity, *, selected_ids=[row["case_id"] for row in selected], name=key):
                    parsed = validate_bounded_residual_response(response, horizon=horizon)
                    residual_state[name] = {
                        **parsed,
                        "selected_case_ids": selected_ids,
                        "request_sha256": identity["sha256"],
                    }

                invoke(
                    case_id,
                    (
                        f"bounded_residual_k{memory_count}_v1"
                        if correction_mode == "r0"
                        else f"bounded_residual_r1_k{memory_count}_v1"
                    ),
                    prompt,
                    lambda response: validate_bounded_residual_response(response, horizon=horizon),
                    commit_residual,
                )

        record["status"] = "forecast_saved"
        save()

        # The first truth access is deliberately after the full forecast is checkpointed.
        truth = _finite_vector(case["truth"], f"fit truth for {case_id}", length=horizon)
        truth_sha256 = _hash_json(truth)
        if "scores" in record:
            if record.get("fit_truth_sha256") != truth_sha256:
                raise ValueError(
                    f"fit truth changed since saved scores for {case_id}; refusing stale calibration state"
                )
        else:
            baseline = forecast_state["baseline"]["prediction"]
            score_record = {"baseline": _score_prediction(baseline, truth), "variants": {}}
            if inference_mode.startswith("direct_hic_context"):
                hic_forecast = forecast_state["hic"]
                score_record["variants"]["full_hic_direct_context"] = {
                    **_score_prediction(hic_forecast["prediction"], truth),
                    "prediction": hic_forecast["prediction"],
                    "fallback_reason": hic_forecast["fallback_reason"],
                    "selected_case_ids": hic_forecast["selected_case_ids"],
                }
            else:
                for memory_count in _CALIBRATION_K:
                    response = residual_state[str(memory_count)]
                    selected_ids = response["selected_case_ids"]
                    for alpha in _CALIBRATION_ALPHAS:
                        for bound_scale in _CALIBRATION_BOUNDS:
                            bounded = apply_bounded_residual(
                                baseline,
                                response["residual"],
                                case["series"],
                                alpha=alpha,
                                bound_scale=bound_scale,
                                eligible=bool(selected_ids),
                                abstain=response["abstain"],
                            )
                            score_record["variants"][_variant_key(memory_count, alpha, bound_scale)] = {
                                **_score_prediction(bounded["prediction"], truth),
                                "prediction": bounded["prediction"],
                                "fallback_reason": bounded["fallback_reason"],
                                "selected_case_ids": selected_ids,
                                "applied_delta": bounded["applied_delta"],
                            }
            record["scores"] = score_record
            record["fit_truth_sha256"] = truth_sha256
            record["status"] = "scored"
            save()

        if "memory" not in record:
            truth = _finite_vector(case["truth"], f"fit truth for {case_id}", length=horizon)
            facts = compute_error_facts(
                case["series"],
                forecast_state["baseline"]["prediction"],
                truth,
            )
            if correction_schema_version == CORRECTION_SCHEMA_VERSION:
                reflection_prompt = build_verified_correction_prompt(
                    forecast_state["summary"],
                    case["series"],
                    truth,
                    forecast_state["baseline"]["prediction"],
                    forecast_state["baseline"]["reasoning"],
                    facts,
                )
                reflection_stage = "fit_memory_reflection_v2"
                validate_reflection = lambda response: _valid_reflection(response, facts)
            elif correction_schema_version == CORRECTION_SCHEMA_VERSION_V3:
                reflection_prompt = build_verified_correction_prompt_v3(
                    forecast_state["summary"],
                    case["series"],
                    truth,
                    forecast_state["baseline"]["prediction"],
                    forecast_state["baseline"]["reasoning"],
                    facts,
                    task_context={
                        "dataset": dataset,
                        "target": target,
                        "unit": unit,
                        "frequency": frequency,
                        "horizon": horizon,
                    },
                )
                reflection_stage = "fit_memory_reflection_v3"
                validate_reflection = lambda response: _valid_reflection_v3(response, facts)
            else:
                reflection_prompt = build_verified_correction_prompt_v4(
                    forecast_state["summary"],
                    case["series"],
                    truth,
                    forecast_state["baseline"]["prediction"],
                    forecast_state["baseline"]["reasoning"],
                    facts,
                    task_context={
                        "dataset": dataset,
                        "target": target,
                        "unit": unit,
                        "frequency": frequency,
                        "horizon": horizon,
                    },
                )
                reflection_stage = "fit_memory_reflection_v4"
                validate_reflection = lambda response: _valid_reflection_v4(response, facts)

            def commit_reflection(response, identity):
                if correction_schema_version == CORRECTION_SCHEMA_VERSION:
                    correction_record = make_verified_correction_record(facts, response)
                    record["pending_correction"] = render_verified_correction(correction_record)
                elif correction_schema_version == CORRECTION_SCHEMA_VERSION_V3:
                    correction_record = make_verified_correction_record_v3(facts, response)
                    record["pending_correction"] = render_verified_correction_v3(correction_record)
                else:
                    correction_record = make_verified_correction_record_v4(facts, response)
                    record["pending_correction"] = render_verified_correction_v4(correction_record)
                record["correction_request_sha256"] = identity["sha256"]

            invoke(
                case_id,
                reflection_stage,
                reflection_prompt,
                validate_reflection,
                commit_reflection,
            )
            memory_embedding = _validate_embedding(
                embed_summary(forecast_state["summary"], is_query=False), 1024, case_id
            )
            summary = forecast_state["summary"]
            metadata = {
                "case_id": case_id,
                "dataset": dataset,
                "source_sha256": source_sha256,
                "target": target,
                "unit": unit,
                "frequency": frequency,
                "horizon": horizon,
                "target_end": case["target_end"],
                "split": "fit",
                "completed": True,
                "correction_schema_version": correction_schema_version,
                "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
                "summary_identity_status": "exact_request_response",
                "summary_request_sha256": forecast_state["summary_request_sha256"],
                "summary_content_sha256": forecast_state["summary_content_sha256"],
                "baseline_request_sha256": forecast_state["baseline"]["request_sha256"],
                "baseline_prediction_sha256": forecast_state["baseline"].get(
                    "prediction_sha256",
                    _hash_json(forecast_state["baseline"]["prediction"]),
                ),
                "correction_request_sha256": record["correction_request_sha256"],
                "news_quality": "valid",
                "history_series": [float(value) for value in case["series"]],
                "history_series_role": "pre_target_history_v1",
            }
            if execution_metadata is not None:
                metadata["execution_metadata"] = execution_metadata
            if correction_schema_version == CORRECTION_SCHEMA_VERSION_V4:
                metadata["baseline_behavior_profile"] = build_forecast_behavior_profile(
                    case["series"], forecast_state["baseline"]["prediction"]
                )
            if memory_version is not None:
                metadata["memory_version"] = memory_version
            if correction_prompt_version is not None:
                metadata["correction_prompt_version"] = correction_prompt_version
            record["memory"] = {
                "case_id": case_id,
                "summary": summary,
                "correction": record.pop("pending_correction"),
                "embedding": memory_embedding,
                "metadata": metadata,
            }
            record["status"] = "memory_prepared"
            save()

        if record.get("ingested") is not True:
            memory = record["memory"]
            upsert_args = dict(
                case_id=case_id,
                summary=memory["summary"],
                correction=memory["correction"],
                embedding=memory["embedding"],
                observed_targets=truth,
                prediction=forecast_state["baseline"]["prediction"],
                cutoff_date=case["target_end"],
                source=f"prequential_fit:{dataset}:{run_id}",
                metadata=memory["metadata"],
            )
            if initial_memory_records is not None:
                upsert_args["replace_seed_version"] = memory_version
            store.upsert_completed_case(**upsert_args)
            record["ingested"] = True
            record["status"] = "complete"
            save()
        elif record.get("status") != "complete":
            if "memory" not in record or "scores" not in record:
                raise ValueError(f"checkpoint marks {case_id} ingested without scores and memory")
            record["status"] = "complete"
            save()
        prior_memories[case_id] = record["memory"]
        if not was_complete_before_call:
            newly_completed_queries += 1
            if max_new_queries is not None and newly_completed_queries >= max_new_queries:
                break

    completed_records = [state["queries"][case_id] for case_id in query_ids]
    completed_count = sum(
        record.get("status") == "complete" and record.get("ingested") is True
        for record in completed_records
    )
    all_complete = completed_count == len(completed_records)
    if not all_complete and max_new_queries is None:
        raise RuntimeError("prequential fit calibration ended with incomplete query or memory stages")
    if not all_complete:
        state["completed_queries"] = completed_count
        state["last_batch_new_queries"] = newly_completed_queries
        state["last_batch_status"] = "partial"
        state["fit_cases_excluded_for_missing_news"] = plan["fit_cases_excluded_for_missing_news"]
        state.pop("baseline_aggregate", None)
        state.pop("variant_aggregates", None)
        save()
        return {
            "protocol": protocol,
            "status": "partial",
            "correction_mode": correction_mode,
            "inference_mode": inference_mode,
            "dataset": dataset,
            "horizon": horizon,
            "completed_queries": completed_count,
            "new_queries_completed": newly_completed_queries,
            "total_queries": len(completed_records),
            "query_ids": query_ids,
            "memory_count": store.count(),
            "correction_schema_version": correction_schema_version,
            "memory_version": memory_version,
            "outer_test_targets_read": False,
            "validation_truth_read": False,
        }
    state.pop("last_batch_status", None)
    state.pop("last_batch_new_queries", None)
    state["variant_aggregates"] = _aggregate_prequential_scores(completed_records)
    state["baseline_aggregate"] = _aggregate_prequential_baseline(completed_records)
    state["completed_queries"] = len(completed_records)
    state["fit_cases_excluded_for_missing_news"] = plan["fit_cases_excluded_for_missing_news"]
    save()
    expected_store_case_ids = set(seed_memory_records) | {
        record["memory"]["case_id"]
        for record in completed_records
        if record.get("ingested") is True
    }
    if store.count() != len(expected_store_case_ids):
        raise RuntimeError("Qdrant point count does not match completed prequential fit memories")
    return {
        "protocol": protocol,
        "status": "complete",
        "correction_mode": correction_mode,
        "inference_mode": inference_mode,
        "dataset": dataset,
        "horizon": horizon,
        "history_length": history_length,
        "completed_queries": len(completed_records),
        "new_queries_completed": newly_completed_queries,
        "query_ids": query_ids,
        "seed_memory_count": len(seed_memory_records),
        "seed_memory_source_sha256": initial_memory_source_sha256,
        "fit_cases_excluded_for_missing_news": plan["fit_cases_excluded_for_missing_news"],
        "memory_count": store.count(),
        "correction_schema_version": correction_schema_version,
        "correction_prompt_version": correction_prompt_version,
        "memory_version": memory_version,
        "baseline_aggregate": state["baseline_aggregate"],
        "variant_aggregates": state["variant_aggregates"],
        "provider_logical_calls": getattr(request_cache, "logical_calls", None),
        "outer_test_targets_read": False,
        "validation_truth_read": False,
    }


def _valid_reflection(response, facts):
    try:
        make_verified_correction_record(facts, response)
    except (TypeError, ValueError):
        return False
    return True


def _valid_reflection_v3(response, facts):
    try:
        make_verified_correction_record_v3(facts, response)
    except (TypeError, ValueError):
        return False
    return True


def _valid_reflection_v4(response, facts):
    try:
        make_verified_correction_record_v4(facts, response)
    except (TypeError, ValueError):
        return False
    return True
