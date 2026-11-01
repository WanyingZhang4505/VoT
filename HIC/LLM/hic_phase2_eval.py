"""Scalable, fixed-fit-memory HIC evaluation for the local VoT corpus."""

import hashlib
import csv
import json
import math
import argparse
import os
import re
import random
import tempfile
from pathlib import Path

from LLM.corpus_audit import DATASET_SPECS, DEFAULT_DATA_DIR
from LLM.native_hic_data import load_native_hic_splits
from LLM.native_hic_experiment import make_numeric_baselines


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"
TEMPLATE_DIR = ROOT / "LLM" / "template"
DEFAULT_RUN_PREFIX = "hic_phase2"
DUPLICATE_DATASETS = {"Economy": "Economy_sort"}
MEMORY_MODEL = "BAAI/bge-m3"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TEMPLATE_OVERRIDES = {"SocialGood": "socialgoods.json"}
VERIFIED_PIPELINE_VERSION = "hic_verified_v7"
TARGET_CONTRACTS_PATH = ROOT / "data" / "target_contracts.json"
EXPLORATORY_EVALUATION_SCOPE = "same_artifact_numeric_exploratory"
VERIFIED_FORECAST_ARMS = (
    "no_retrieval", "summary_retrieval", "full_reasoning", "low_similarity_control",
)
_FORECAST_INPUT_FIELDS = (
    "case_id", "dataset", "frequency", "group", "history_start", "history_end",
    "target_start", "target_end", "source_dates", "series", "news", "news_quality_detail", "covariates",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_target_contracts(contract_path=None):
    """Load the enforced target registry and return it indexed by dataset."""
    path = Path(contract_path) if contract_path is not None else TARGET_CONTRACTS_PATH
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"target contract registry cannot be read: {error}") from None
    if not isinstance(registry, dict) or type(registry.get("contract_version")) is not int:
        raise ValueError("target contract registry has an invalid contract_version")
    policy = registry.get("evaluation_policy")
    if not isinstance(policy, dict) or policy.get("registry_is_enforced_by_runner") is not True:
        raise ValueError("target contract registry has not enabled runner enforcement")
    if policy.get("same_artifact_numeric_evaluation") != "exploratory_only":
        raise ValueError("target contract registry must restrict numeric results to exploratory scope")
    if policy.get("event_semantic_claim") != "blocked_until_news_source_publication_time_and_target_relevance_are_audited":
        raise ValueError("target contract registry weakens the event-semantic evidence gate")
    if policy.get("cross_dataset_or_cross_target_memory") != "blocked_until_target_identity_unit_and_compatible_horizon_are_verified":
        raise ValueError("target contract registry weakens the cross-target memory gate")
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("target contract registry is missing its datasets list")
    by_dataset = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset"), str):
            raise ValueError("target contract registry contains an invalid dataset entry")
        dataset = entry["dataset"]
        if dataset in by_dataset:
            raise ValueError(f"target contract registry repeats dataset {dataset}")
        by_dataset[dataset] = entry
    return by_dataset, _sha256_file(path)


def _csv_row_count(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _validate_target_contract(
    dataset, contract, *, source_file, source_sha256, row_count, cadence,
):
    """Check one contract against the exact local numeric artifact and HIC target."""
    if not isinstance(contract, dict) or contract.get("dataset") != dataset:
        raise ValueError(f"target contract is missing or misidentified for {dataset}")
    if Path(str(contract.get("local_file", ""))).name != Path(source_file).name:
        raise ValueError(f"target contract local file does not match {dataset}")
    if contract.get("local_sha256") != source_sha256:
        raise ValueError(f"target contract source SHA-256 does not match {dataset}")
    if type(contract.get("local_rows")) is not int or contract["local_rows"] != row_count:
        raise ValueError(f"target contract row count does not match {dataset}")
    local_target = contract.get("local_target")
    if (
        not isinstance(local_target, dict)
        or local_target.get("field") != "OT"
        or type(local_target.get("dimensions_used_by_hic")) is not int
        or local_target.get("dimensions_used_by_hic") != 1
    ):
        raise ValueError(f"target contract does not describe the scalar OT target for {dataset}")
    cadence_names = {
        "month": "monthly", "week": "weekly", "day": "daily",
        "irregular": "irregular",
    }
    expected_frequency = cadence_names.get(cadence)
    if expected_frequency is None or local_target.get("frequency") != expected_frequency:
        raise ValueError(f"target contract cadence does not match {dataset}")
    mapping_status = contract.get("mapping_status")
    if not isinstance(mapping_status, str) or mapping_status not in {
        "verified", "partial", "conflict", "unknown",
    }:
        raise ValueError(f"target contract mapping status is invalid for {dataset}")
    reference = contract.get("time_mmd_reference")
    if not isinstance(reference, dict):
        reference = {}
    return {
        "contract_sha256": _sha256_json(contract),
        "source_sha256": source_sha256,
        "mapping_status": mapping_status,
        "local_target_field": "OT",
        "dimensions_used_by_hic": 1,
        "local_frequency": expected_frequency,
        "local_rows": row_count,
        "paper_target": reference.get("target"),
        "paper_dimensions": reference.get("dimensions"),
        "paper_frequency": reference.get("frequency"),
        "real_world_target_semantics_verified": mapping_status == "verified",
    }


def forecast_input(case):
    """Select fields available at forecast time; target values are never copied."""
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    result = {key: case[key] for key in _FORECAST_INPUT_FIELDS if key in case}
    if not isinstance(result.get("case_id"), str) or not result["case_id"]:
        raise ValueError("forecast input needs a case_id")
    if not isinstance(result.get("series"), list) or not result["series"]:
        raise ValueError("forecast input needs a non-empty observed history")
    if not isinstance(result.get("news"), str):
        raise ValueError("forecast input news must be text")
    return result


def fuse_forecasts(hic_prediction, numeric_prediction, *, alpha):
    """Blend a HIC forecast with a numeric forecast using validation-frozen alpha."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if not isinstance(hic_prediction, (list, tuple)) or not hic_prediction:
        raise ValueError("hic_prediction must be a non-empty sequence")
    if not isinstance(numeric_prediction, (list, tuple)) or len(numeric_prediction) != len(hic_prediction):
        raise ValueError("numeric_prediction must align with hic_prediction")
    try:
        hic_values = [float(value) for value in hic_prediction]
        numeric_values = [float(value) for value in numeric_prediction]
    except (TypeError, ValueError):
        raise ValueError("forecasts must contain only finite numeric values") from None
    if not all(math.isfinite(value) for value in hic_values + numeric_values):
        raise ValueError("forecasts must contain only finite numeric values")
    return [
        float(alpha) * hic_value + (1.0 - float(alpha)) * numeric_value
        for hic_value, numeric_value in zip(hic_values, numeric_values)
    ]


def _pooled_metrics(windows, forecasts):
    absolute_error = 0.0
    squared_error = 0.0
    target_count = 0
    for window, prediction in zip(windows, forecasts):
        truth = window["truth"]
        if len(prediction) != len(truth) or not truth:
            raise ValueError("each forecast must align with its non-empty truth")
        for predicted, observed in zip(prediction, truth):
            predicted = float(predicted)
            observed = float(observed)
            if not math.isfinite(predicted) or not math.isfinite(observed):
                raise ValueError("truth and forecasts must be finite")
            error = predicted - observed
            absolute_error += abs(error)
            squared_error += error * error
            target_count += 1
    if not target_count:
        raise ValueError("at least one validation target is required")
    return {
        "target_points": target_count,
        "windows": len(windows),
        "mae": absolute_error / target_count,
        "mse": squared_error / target_count,
        "rmse": math.sqrt(squared_error / target_count),
    }


def paired_moving_block_bootstrap(
    windows, *, arm, comparator, replicates=2000, seed=20261018
):
    """Estimate paired MAE/RMSE differences by circularly resampling time blocks.

    Windows must be in chronological order. Each sampled window keeps the two
    forecast arms paired, and each bootstrap replicate resamples contiguous
    blocks rather than treating overlapping windows as independent.
    """
    if not isinstance(windows, (list, tuple)) or len(windows) < 2:
        raise ValueError("at least two chronological windows are required")
    if not isinstance(arm, str) or not arm or not isinstance(comparator, str) or not comparator:
        raise ValueError("arm and comparator must be non-empty names")
    if type(replicates) is not int or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    prepared = []
    target_count = 0
    starts = [window.get("target_start") for window in windows]
    if all(start is not None for start in starts) and starts != sorted(starts):
        raise ValueError("windows with target_start must be in chronological order")
    for window in windows:
        scoring = window.get("scoring")
        if isinstance(scoring, dict):
            truth = scoring.get("truth")
            forecasts = scoring.get("forecasts")
        else:
            truth = window.get("truth")
            forecasts = window.get("forecasts")
        if not isinstance(truth, (list, tuple)) or not truth:
            raise ValueError("each window needs a non-empty truth sequence")
        if not isinstance(forecasts, dict) or arm not in forecasts or comparator not in forecasts:
            raise ValueError("each window must contain both paired forecast arms")
        pair = {}
        for name in (arm, comparator):
            prediction = forecasts[name]
            if not isinstance(prediction, (list, tuple)) or len(prediction) != len(truth):
                raise ValueError("paired forecasts must align with the truth sequence")
            try:
                values = [float(value) for value in prediction]
            except (TypeError, ValueError):
                raise ValueError("forecasts and truth must be finite numeric sequences") from None
            if not all(math.isfinite(value) for value in values):
                raise ValueError("forecasts and truth must be finite numeric sequences")
            pair[name] = values
        try:
            observed = [float(value) for value in truth]
        except (TypeError, ValueError):
            raise ValueError("forecasts and truth must be finite numeric sequences") from None
        if not all(math.isfinite(value) for value in observed):
            raise ValueError("forecasts and truth must be finite numeric sequences")
        prepared.append({"truth": observed, "forecasts": pair})
        target_count += len(observed)

    window_count = len(prepared)
    block_length = min(window_count, max(2, int(round(window_count ** (1 / 3)))))

    def metrics(indices, name):
        absolute_error = 0.0
        squared_error = 0.0
        count = 0
        for index in indices:
            row = prepared[index]
            for prediction, truth in zip(row["forecasts"][name], row["truth"]):
                error = prediction - truth
                absolute_error += abs(error)
                squared_error += error * error
                count += 1
        mse = squared_error / count
        return absolute_error / count, mse, math.sqrt(mse)

    observed_arm = metrics(range(window_count), arm)
    observed_comparator = metrics(range(window_count), comparator)
    point_delta = {
        "mae": observed_arm[0] - observed_comparator[0],
        "mse": observed_arm[1] - observed_comparator[1],
        "rmse": observed_arm[2] - observed_comparator[2],
    }
    rng = random.Random(seed)
    bootstrap = {"mae": [], "mse": [], "rmse": []}
    for _ in range(replicates):
        sample = []
        while len(sample) < window_count:
            start = rng.randrange(window_count)
            sample.extend((start + offset) % window_count for offset in range(block_length))
        sample = sample[:window_count]
        arm_metrics = metrics(sample, arm)
        comparator_metrics = metrics(sample, comparator)
        bootstrap["mae"].append(arm_metrics[0] - comparator_metrics[0])
        bootstrap["mse"].append(arm_metrics[1] - comparator_metrics[1])
        bootstrap["rmse"].append(arm_metrics[2] - comparator_metrics[2])

    def percentile(values, probability):
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    interpretation = "descriptive_only" if window_count < 5 else "exploratory_unstable"
    return {
        "method": "paired_circular_moving_block_bootstrap",
        "arm": arm,
        "comparator": comparator,
        "window_count": window_count,
        "target_point_count": target_count,
        "block_length_windows": block_length,
        "replicates": replicates,
        "seed": seed,
        "paired_delta": point_delta,
        "ci95": {
            metric_name: [
                percentile(bootstrap[metric_name], 0.025),
                percentile(bootstrap[metric_name], 0.975),
            ]
            for metric_name in ("mae", "mse", "rmse")
        },
        "negative_delta_favors_arm": True,
        "interpretation": interpretation,
        "caveat": (
            "One series and at most 40 validation windows; intervals are exploratory, "
            "not a significance claim."
        ),
    }


def macro_relative_improvement(configurations, *, arm, comparator, metric="mae"):
    """Average relative error reduction equally across dataset families.

    Multiple horizons within one dataset are first averaged within that family,
    so datasets with more configured horizons do not dominate the macro result.
    """
    if not isinstance(configurations, (list, tuple)) or not configurations:
        raise ValueError("configurations must be a non-empty sequence")
    if metric not in {"mae", "mse", "rmse"}:
        raise ValueError("metric must be mae, mse, or rmse")
    by_dataset = {}
    for configuration in configurations:
        dataset = configuration.get("dataset")
        metrics = configuration.get("arm_metrics")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("each configuration needs a dataset family")
        if not isinstance(metrics, dict) or arm not in metrics or comparator not in metrics:
            raise ValueError("each configuration must contain both compared arms")
        arm_value = float(metrics[arm][metric])
        comparator_value = float(metrics[comparator][metric])
        if not math.isfinite(arm_value) or not math.isfinite(comparator_value) or comparator_value <= 0:
            raise ValueError("error metrics must be finite and comparator error must be positive")
        by_dataset.setdefault(dataset, []).append(
            100.0 * (comparator_value - arm_value) / comparator_value
        )
    dataset_means = {
        dataset: sum(values) / len(values)
        for dataset, values in sorted(by_dataset.items())
    }
    return {
        "metric": metric,
        "arm": arm,
        "comparator": comparator,
        "per_dataset_relative_improvement_percent": dataset_means,
        "macro_relative_improvement_percent": sum(dataset_means.values()) / len(dataset_means),
        "dataset_family_count": len(dataset_means),
        "configuration_count": len(configurations),
        "aggregation": "equal weight per dataset family",
        "interpretation": "descriptive; configurations and horizons may be dependent",
    }


def build_validation_uncertainty(
    windows,
    *,
    selected_numeric_baseline=None,
    blend_arm=None,
    selection_case_ids=None,
    replicates=2000,
    seed=20261018,
):
    """Build paired intervals for available arms and disclose validation selection."""
    arm_sets = []
    for window in windows:
        scoring = window.get("scoring") if isinstance(window, dict) else None
        forecasts = scoring.get("forecasts") if isinstance(scoring, dict) else None
        if not isinstance(forecasts, dict) and isinstance(window, dict):
            forecasts = window.get("forecasts")
        arm_sets.append(set(forecasts) if isinstance(forecasts, dict) else set())
    if not arm_sets:
        raise ValueError("uncertainty windows are empty")
    if any(arms != arm_sets[0] for arms in arm_sets):
        varying_arms = set.union(*arm_sets) - set.intersection(*arm_sets)
        if varying_arms == {"low_similarity_control"}:
            raise ValueError(
                "low_similarity_control must be present in every window when available"
            )
        raise ValueError("forecast arms must be present consistently across uncertainty windows")
    available = arm_sets[0]
    comparisons = {}
    if {"summary_retrieval", "no_retrieval"}.issubset(available):
        comparisons["summary_retrieval_vs_no_retrieval"] = (
            "summary_retrieval", "no_retrieval"
        )
    if {"full_reasoning", "no_retrieval"}.issubset(available):
        comparisons["full_reasoning_vs_no_retrieval"] = (
            "full_reasoning", "no_retrieval"
        )
    if {"full_reasoning", "summary_retrieval"}.issubset(available):
        comparisons["full_reasoning_vs_summary_retrieval"] = (
            "full_reasoning", "summary_retrieval"
        )
    if not comparisons:
        raise ValueError("uncertainty requires at least one recognized paired comparison")
    low_similarity_presence = []
    for window in windows:
        scoring = window.get("scoring") if isinstance(window, dict) else None
        forecasts = scoring.get("forecasts") if isinstance(scoring, dict) else None
        if not isinstance(forecasts, dict) and isinstance(window, dict):
            forecasts = window.get("forecasts")
        low_similarity_presence.append(
            isinstance(forecasts, dict) and "low_similarity_control" in forecasts
        )
    if any(low_similarity_presence) and not all(low_similarity_presence):
        raise ValueError(
            "low_similarity_control must be present in every window when available"
        )
    comparisons_not_evaluated = {}
    if all(low_similarity_presence):
        comparisons["full_reasoning_vs_low_similarity_control"] = (
            "full_reasoning", "low_similarity_control"
        )
    else:
        comparisons_not_evaluated["full_reasoning_vs_low_similarity_control"] = (
            "low_similarity_control_not_provided"
        )
    if selected_numeric_baseline is not None:
        comparisons["full_reasoning_vs_selected_numeric_baseline"] = (
            "full_reasoning", selected_numeric_baseline
        )
    if blend_arm is not None and selected_numeric_baseline is not None:
        comparisons["validation_blend_vs_selected_numeric_baseline"] = (
            blend_arm, selected_numeric_baseline
        )

    window_case_ids = {
        window.get("case_id") for window in windows if window.get("case_id") is not None
    }
    selection_overlap = (
        None
        if selection_case_ids is None
        else bool(window_case_ids.intersection(selection_case_ids))
    )

    results = {}
    for index, (label, (arm, comparator)) in enumerate(comparisons.items()):
        result = paired_moving_block_bootstrap(
            windows,
            arm=arm,
            comparator=comparator,
            replicates=replicates,
            seed=seed + index,
        )
        selection_dependent = (
            (blend_arm is not None and arm == blend_arm)
            or (
                selected_numeric_baseline is not None
                and comparator == selected_numeric_baseline
            )
        )
        result["selected_on_same_validation_windows"] = (
            selection_overlap if selection_dependent else False
        )
        results[label] = result
    return {
        "method": "paired circular moving-block bootstrap over chronological windows",
        "comparisons": results,
        "comparisons_not_evaluated": comparisons_not_evaluated,
        "selected_numeric_baseline_uses_same_validation_windows": (
            selection_overlap if selected_numeric_baseline is not None else False
        ),
        "blend_uses_same_validation_windows": (
            selection_overlap if blend_arm is not None else False
        ),
        "selection_overlap_window_count": (
            len(window_case_ids.intersection(selection_case_ids))
            if selection_case_ids is not None else None
        ),
        "uncertainty_window_count": len(windows),
        "primary_comparison": (
            "full_reasoning_vs_no_retrieval"
            if set.intersection(*(set(row) for row in arm_sets)) == {
                "no_retrieval", "full_reasoning"
            }
            else None
        ),
        "selection_caveat": (
            (
                "The interval is tuning-biased because a selected baseline or blend overlaps "
                "the uncertainty windows. Use frozen data for confirmation."
                if selection_overlap is True else
                "Selection windows are disjoint from these uncertainty windows; the interval "
                "does not reuse those selection cases, but remains exploratory."
                if selection_overlap is False else
                "Selection-window IDs are unavailable, so overlap cannot be determined."
            )
            if selected_numeric_baseline is not None or blend_arm is not None else None
        ),
        "cross_dataset_summary": (
            "After all configurations finish, compute relative improvement per configuration, "
            "average horizons within each dataset family, then macro-average families equally."
        ),
    }


def select_validation_fusion(windows, *, alphas=ALPHAS):
    """Select a numeric baseline and HIC blend weight using development windows only."""
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("validation windows must be non-empty")
    alpha_values = tuple(alphas)
    if not alpha_values or any(
        not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 <= alpha <= 1
        for alpha in alpha_values
    ):
        raise ValueError("alphas must be non-empty values between zero and one")
    baseline_names = set(windows[0].get("numeric_forecasts", {}))
    if not baseline_names or any(
        set(window.get("numeric_forecasts", {})) != baseline_names
        for window in windows
    ):
        raise ValueError("every validation window must contain the same numeric baselines")
    for window in windows:
        if not window.get("case_id"):
            raise ValueError("every validation window must have a case_id")
        if "truth" not in window or "full_reasoning" not in window:
            raise ValueError("validation windows need truth and full_reasoning forecasts")

    baseline_metrics = {
        name: _pooled_metrics(
            windows, [window["numeric_forecasts"][name] for window in windows]
        )
        for name in sorted(baseline_names)
    }
    numeric_baseline = min(
        baseline_metrics, key=lambda name: (baseline_metrics[name]["mae"], name)
    )
    candidates = []
    for alpha in sorted(set(float(value) for value in alpha_values)):
        predictions = [
            fuse_forecasts(
                window["full_reasoning"],
                window["numeric_forecasts"][numeric_baseline],
                alpha=alpha,
            )
            for window in windows
        ]
        metrics = _pooled_metrics(windows, predictions)
        candidates.append((metrics["mae"], alpha, metrics))
    _, alpha, metrics = min(candidates, key=lambda row: (row[0], row[1]))
    return {
        "selection_data": "validation only",
        "numeric_baseline": numeric_baseline,
        "numeric_baseline_validation": baseline_metrics[numeric_baseline],
        "alpha": alpha,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "target_points": metrics["target_points"],
        "validation_windows": metrics["windows"],
        "alpha_candidates": [float(value) for value in sorted(set(alpha_values))],
        "candidate_validation_metrics": {
            str(alpha_value): candidate_metrics
            for _, alpha_value, candidate_metrics in sorted(candidates, key=lambda row: row[1])
        },
        "all_numeric_validation_metrics": baseline_metrics,
        "validation_score_is_tuning_biased": True,
        "validation_score_note": (
            "The blend weight and numeric baseline are selected on these same validation windows; "
            "the reported blend score is a tuning estimate, not an unbiased confirmation."
        ),
    }


def estimate_logical_calls(
    *, fit_cases, validation_windows, test_windows=0, template_calls=0,
    forecast_arm_count=3,
):
    """Estimate uncached calls: 3F + (1+A)V + (1+A)T + templates."""
    values = {
        "fit_cases": fit_cases,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "template_calls": template_calls,
        "forecast_arm_count": forecast_arm_count,
    }
    if any(type(value) is not int or value < 0 for value in values.values()) or not forecast_arm_count:
        raise ValueError("call counts must be non-negative integers")
    fit_calls = 3 * fit_cases
    validation_calls = (1 + forecast_arm_count) * validation_windows
    test_calls = (1 + forecast_arm_count) * test_windows
    return {
        "template": template_calls,
        "fit_memory": fit_calls,
        "validation": validation_calls,
        "outer_test": test_calls,
        "total": template_calls + fit_calls + validation_calls + test_calls,
    }


def _summary_cache_key(case, *, template_fingerprint, dataset_description):
    """Fingerprint only information available when the summary is generated."""
    stable = {
        "dataset": case.get("dataset"),
        "template_fingerprint": template_fingerprint,
        "dataset_description": dataset_description,
        "series": case.get("series"),
        "news": case.get("news"),
        "covariates": case.get("covariates", {}),
    }
    return _sha256_json(stable)


def estimate_cached_logical_calls(
    *, fit_cases, validation_windows, template_calls=0, outer_test_windows=0,
    template_fingerprint="template", forecast_arm_count=3,
):
    """Count calls when exact, forecast-time summaries are shared across horizons.

    Full-reasoning fit memories still require a horizon-specific initial forecast
    and outcome correction. Each validation window needs one forecast per arm.
    Outer-test inputs are deliberately excluded from the development summary cache.
    """
    if not isinstance(fit_cases, (list, tuple)) or not isinstance(validation_windows, (list, tuple)):
        raise ValueError("fit_cases and validation_windows must be sequences")
    if any(type(value) is not int or value < 0 for value in (
        template_calls, outer_test_windows, forecast_arm_count,
    )) or not forecast_arm_count:
        raise ValueError("template_calls and outer_test_windows must be non-negative integers")
    if not isinstance(template_fingerprint, str) or not template_fingerprint:
        raise ValueError("template_fingerprint must be non-empty text")
    summaries = set()
    for case in fit_cases:
        summaries.add(_summary_cache_key(
            case,
            template_fingerprint=template_fingerprint,
            dataset_description=case.get("dataset_description", ""),
        ))
    for window in validation_windows:
        case = window.get("case")
        if not isinstance(case, dict):
            raise ValueError("each validation window must contain a case object")
        summaries.add(_summary_cache_key(
            case,
            template_fingerprint=template_fingerprint,
            dataset_description=case.get("dataset_description", ""),
        ))
    fit_forecast_and_correction = 2 * len(fit_cases)
    validation_forecasts = forecast_arm_count * len(validation_windows)
    outer_calls = (1 + forecast_arm_count) * outer_test_windows
    return {
        "template": template_calls,
        "unique_summaries": len(summaries),
        "fit_forecast_and_correction": fit_forecast_and_correction,
        "validation_forecasts": validation_forecasts,
        "outer_test": outer_calls,
        "total": (
            template_calls + len(summaries) + fit_forecast_and_correction
            + validation_forecasts + outer_calls
        ),
    }
def _resolve_template(dataset, template_dir=None):
    root = Path(template_dir) if template_dir is not None else TEMPLATE_DIR
    filename = _TEMPLATE_OVERRIDES.get(dataset, f"{dataset.lower()}.json")
    filenames = [filename]
    slug_filename = f"{dataset.lower()}.json"
    if slug_filename not in filenames:
        filenames.append(slug_filename)
    for candidate_name in filenames:
        standard = root / candidate_name
        if not standard.is_file():
            continue
        try:
            value = json.loads(standard.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict) and value:
            return standard, value, "source_template"

    candidates = []
    for report_path in EVALUATIONS.glob("native_hic_pilot_*.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            report.get("status") != "completed_validation_pilot"
            or report.get("plan", {}).get("dataset") != dataset
        ):
            continue
        meta = report.get("template", {})
        candidate = Path(meta.get("path", ""))
        if not candidate.is_file():
            candidate = EVALUATIONS / f"{report_path.stem}.template.json"
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not value:
            continue
        expected_hash = meta.get("sha256")
        if expected_hash and _sha256_json(value) != expected_hash:
            continue
        candidates.append((report_path.stat().st_mtime_ns, candidate, value))
    if candidates:
        _, path, value = max(candidates, key=lambda row: row[0])
        return path, value, "verified_completed_run_template"
    return None, None, "template_generation_required"


def _dataset_path(dataset, data_dir=None):
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    spec = next(
        (item for item in DATASET_SPECS if Path(item.filename).stem == dataset), None
    )
    if spec is None:
        raise ValueError(f"unknown VoT dataset: {dataset}")
    return root / spec.filename


def build_phase2_grid_plan(*, data_dir=None, template_dir=None, run_tag="20261018"):
    """Create a development-only plan for every runnable dataset/horizon pair."""
    if not isinstance(run_tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", run_tag):
        raise ValueError("run_tag must contain 1-32 letters, digits, '_' or '-'")
    target_contracts, target_contract_registry_sha256 = _load_target_contracts()
    specs = [spec for spec in DATASET_SPECS if Path(spec.filename).stem not in DUPLICATE_DATASETS]
    plans = []
    summary_keys_by_dataset = {}
    template_fingerprints = {}
    for spec in specs:
        dataset = Path(spec.filename).stem
        path = _dataset_path(dataset, data_dir)
        if not path.is_file():
            plans.append({
                "dataset": dataset,
                "status": "unavailable",
                "source_file": spec.filename,
                "reason": "configured source CSV is absent",
            })
            continue
        source_sha256 = _sha256_file(path)
        target_contract = target_contracts.get(dataset)
        if target_contract is None:
            for horizon in spec.prediction_lengths:
                plans.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "status": "ineligible",
                    "source_file": spec.filename,
                    "source_sha256": source_sha256,
                    "reason": "no target contract is registered for this source artifact",
                })
            continue
        contract_metadata = _validate_target_contract(
            dataset,
            target_contract,
            source_file=spec.filename,
            source_sha256=source_sha256,
            row_count=_csv_row_count(path),
            cadence=spec.cadence,
        )
        template_path, _, template_status = _resolve_template(dataset, template_dir)
        if template_path is not None:
            template_fingerprint = _sha256_file(template_path)
        else:
            # The eventual template is one per dataset and therefore preserves
            # equality/inequality between its cases for this call estimate.
            template_fingerprint = f"pending-template:{dataset}"
        template_fingerprints[dataset] = template_fingerprint
        description = (
            f"Dataset: {dataset}, from the upstream VoT repository. The numeric target "
            "column is named OT; the local metadata does not document its unit or exact "
            "real-world interpretation, so treat it as an unnamed numeric target. Text "
            "facts are aligned to source periods; undated facts are not independently "
            "verified as publication-time news."
        )
        summary_keys_by_dataset.setdefault(dataset, set())
        template_calls = 0 if template_status != "template_generation_required" else 1
        for horizon in spec.prediction_lengths:
            try:
                split = load_native_hic_splits(
                    dataset,
                    horizon=horizon,
                    data_dir=data_dir,
                    include_outer_test_targets=False,
                    include_validation_truth=False,
                )
            except (OSError, ValueError) as error:
                plans.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "status": "ineligible",
                    "source_file": spec.filename,
                    "source_sha256": source_sha256,
                    "target_contract": contract_metadata,
                    "reason": f"{type(error).__name__}: {error}",
                })
                continue
            fit_count = len(split["fit_cases"])
            validation_count = len(split["validation_windows"])
            for case in split["fit_cases"]:
                summary_keys_by_dataset[dataset].add(_summary_cache_key(
                    case,
                    template_fingerprint=template_fingerprint,
                    dataset_description=description,
                ))
            for item in split["validation_windows"]:
                summary_keys_by_dataset[dataset].add(_summary_cache_key(
                    item["case"],
                    template_fingerprint=template_fingerprint,
                    dataset_description=description,
                ))
            if not fit_count or not validation_count:
                status = "ineligible"
                reason = "no usable fit cases or validation windows with safe history text"
            else:
                status = "development_ready"
                reason = None
            calls = estimate_logical_calls(
                fit_cases=fit_count,
                validation_windows=validation_count,
                template_calls=template_calls if horizon == spec.prediction_lengths[0] else 0,
                forecast_arm_count=len(VERIFIED_FORECAST_ARMS),
            )
            plans.append({
                "dataset": dataset,
                "horizon": horizon,
                "status": status,
                "reason": reason,
                "source_file": spec.filename,
                "source_sha256": source_sha256,
                "target_contract": contract_metadata,
                "cadence": split["cadence"],
                "history_length": split["history_length"],
                "fit_boundary": split["boundaries"]["fit_end"],
                "validation_boundary": split["boundaries"]["validation_end"],
                "fit_case_count": fit_count,
                "validation_window_count": validation_count,
                "fit_case_ids": [case["case_id"] for case in split["fit_cases"]],
                "validation_case_ids": [item["case"]["case_id"] for item in split["validation_windows"]],
                "quality": split["quality"],
                "template": {
                    "status": template_status,
                    "path": str(template_path) if template_path else None,
                    "calls_in_this_configuration": calls["template"],
                },
                "logical_call_estimate": calls,
                "outer_test": "deferred until configuration freeze; IDs and labels not materialized",
                "run_id": f"{DEFAULT_RUN_PREFIX}_{dataset.lower()}_h{horizon}_{run_tag}",
            })

    active = [row for row in plans if row.get("status") == "development_ready"]
    totals = {
        key: sum(row["logical_call_estimate"][key] for row in active)
        for key in ("template", "fit_memory", "validation", "total")
    }
    cached_template_calls = sum(
        1 for dataset in summary_keys_by_dataset
        if any(
            row.get("dataset") == dataset
            and row.get("template", {}).get("status") == "template_generation_required"
            for row in active
        )
    )
    cached_fit_calls = 2 * sum(row["fit_case_count"] for row in active)
    cached_validation_calls = len(VERIFIED_FORECAST_ARMS) * sum(
        row["validation_window_count"] for row in active
    )
    unique_summaries = sum(len(keys) for keys in summary_keys_by_dataset.values())
    cached_total = (
        cached_template_calls + cached_fit_calls + cached_validation_calls
        + unique_summaries
    )
    return {
        "schema_version": 1,
        "mode": "full_corpus_development_plan",
        "run_tag": run_tag,
        "pipeline_version": VERIFIED_PIPELINE_VERSION,
        "target_contract_registry_sha256": target_contract_registry_sha256,
        "target_contract_policy": "same_artifact_numeric_evaluation=exploratory_only",
        "evaluation_scope": EXPLORATORY_EVALUATION_SCOPE,
        "real_world_target_semantics_verified": False,
        "forecast_arms": list(VERIFIED_FORECAST_ARMS),
        "numeric_controls": ["persistence", "history_mean", "recent_mean_3", "linear_trend"],
        "fusion": {"method": "validation-selected convex blend of full_reasoning and best numeric baseline", "alpha_grid": list(ALPHAS)},
        "embedding": {"model": MEMORY_MODEL, "dimension": 1024, "similarity": "cosine", "max_length": 8192},
        "fit_split": 0.64,
        "validation_end_fraction": 0.8,
        "selection_confirmation_split": {
            "method": "chronological_earlier_half_selects_later_half_confirms",
            "same_windows_used_for_selection_and_confirmation": False,
        },
        "fit_window_total": sum(row["fit_case_count"] for row in active),
        "validation_window_total": sum(row["validation_window_count"] for row in active),
        "configuration_count": len(active),
        "logical_call_estimate_without_cache": totals,
        "summary_cache": {
            "scope": "exact dataset/template/as-of inputs; summaries are shared across horizons only",
            "unique_development_summaries": unique_summaries,
            "template_fingerprints": template_fingerprints,
        },
        "logical_call_estimate_with_summary_cache": {
            "template": cached_template_calls,
            "unique_summaries": unique_summaries,
            "fit_forecast_and_correction": cached_fit_calls,
            "validation_forecasts": cached_validation_calls,
            "outer_test": 0,
            "total": cached_total,
        },
        "provider_requests_made": 0,
        "embedding_inferences_made": 0,
        "qdrant_writes_made": 0,
        "outer_test_targets_read": False,
        "plans": plans,
    }


def _write_plan(path, plan):
    path = Path(path)
    payload = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"plan exists and differs; choose a new run tag: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="20261018")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = build_phase2_grid_plan(
        data_dir=args.data_dir, template_dir=args.template_dir, run_tag=args.run_tag
    )
    default_path = EVALUATIONS / f"hic_phase2_full_corpus_development_{args.run_tag}.plan.json"
    output = _write_plan(args.output or default_path, plan)
    print(json.dumps({
        "status": "development_plan_written",
        "artifact": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "configuration_count": plan["configuration_count"],
        "fit_windows": plan["fit_window_total"],
        "validation_windows": plan["validation_window_total"],
        "logical_calls_without_cache": plan["logical_call_estimate_without_cache"]["total"],
        "logical_calls_with_summary_cache": plan["logical_call_estimate_with_summary_cache"]["total"],
        "outer_test_targets_read": plan["outer_test_targets_read"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
