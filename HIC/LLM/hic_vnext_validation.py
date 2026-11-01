"""Holdout-safe validation scoring for paired HIC vNext forecasts."""

from __future__ import annotations

import math
from collections import defaultdict


_FORBIDDEN_TARGET_FIELDS = {"truth", "target_values", "observed_targets", "test_truth"}


def _contains_target_field(value, *, path="window"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_TARGET_FIELDS:
                return f"{path}.{key}"
            found = _contains_target_field(nested, path=f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _contains_target_field(nested, path=f"{path}[{index}]")
            if found:
                return found
    return None


def _finite_vector(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite numeric values") from None
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numeric values")
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return result


def validate_validation_window(window, *, expected_horizon):
    """Validate a forecast-cutoff window and reject any embedded target truth."""
    if type(expected_horizon) is not int or expected_horizon <= 0:
        raise ValueError("expected_horizon must be a positive integer")
    if not isinstance(window, dict):
        raise ValueError("validation window must be a mapping")
    leaked_field = _contains_target_field(window)
    if leaked_field:
        raise ValueError(f"validation input contains held-out truth at {leaked_field}")
    case = window.get("case")
    if not isinstance(case, dict):
        raise ValueError("validation window must contain a case mapping")
    case_id = case.get("case_id")
    dataset = case.get("dataset")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("validation case_id must be non-empty text")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("validation dataset must be non-empty text")
    history = _finite_vector(case.get("series"), f"history for {case_id}")
    if len(history) < 2:
        raise ValueError("validation history must contain at least two values")
    timestamps = window.get("target_timestamps")
    if timestamps is not None and (
        not isinstance(timestamps, (list, tuple)) or len(timestamps) != expected_horizon
    ):
        raise ValueError("target_timestamps length must equal expected_horizon")
    return {
        "case_id": case_id,
        "dataset": dataset,
        "horizon": expected_horizon,
        "history_length": len(history),
    }


def _metrics(values):
    if not values:
        raise ValueError("cannot score an empty prediction set")
    absolute = [abs(value) for value in values]
    squared = [value * value for value in values]
    mse = sum(squared) / len(squared)
    return {
        "mae": sum(absolute) / len(absolute),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "n": len(values),
    }


def _relative_gain_pct(baseline, full_hic):
    if baseline == 0:
        return 0.0 if full_hic == 0 else None
    return 100.0 * (baseline - full_hic) / baseline


def _direction(baseline, full_hic, *, tolerance=1e-12):
    difference = baseline - full_hic
    if difference > tolerance:
        return "improved"
    if difference < -tolerance:
        return "worsened"
    return "tie"


def score_paired_forecasts(forecast_rows, truth_rows):
    """Score paired forecast records and aggregate overlapping horizons by origin.

    Each origin contributes once to the unique-origin counts. If it appears at
    several horizons, its score is the unweighted mean of its per-task relative
    MAE gains; ties use a small absolute-error tolerance.
    """
    if not isinstance(forecast_rows, (list, tuple)) or not forecast_rows:
        raise ValueError("forecast_rows must be a non-empty sequence")
    if not isinstance(truth_rows, (list, tuple)) or not truth_rows:
        raise ValueError("truth_rows must be a non-empty sequence")

    def key_for(row, label):
        if not isinstance(row, dict):
            raise ValueError(f"each {label} row must be a mapping")
        dataset, case_id, horizon = row.get("dataset"), row.get("case_id"), row.get("horizon")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"{label} dataset must be non-empty text")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{label} case_id must be non-empty text")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError(f"{label} horizon must be a positive integer")
        return dataset, case_id, horizon

    forecast_by_key = {}
    for row in forecast_rows:
        key = key_for(row, "forecast")
        if key in forecast_by_key:
            raise ValueError(f"duplicate forecast row for {key}")
        baseline = _finite_vector(row.get("baseline_prediction"), "baseline_prediction", length=key[2])
        full_hic = _finite_vector(row.get("full_hic_prediction"), "full_hic_prediction", length=key[2])
        forecast_by_key[key] = (baseline, full_hic)

    truth_by_key = {}
    for row in truth_rows:
        key = key_for(row, "truth")
        if key in truth_by_key:
            raise ValueError(f"duplicate truth row for {key}")
        truth_by_key[key] = _finite_vector(row.get("truth"), "truth", length=key[2])
    if forecast_by_key.keys() != truth_by_key.keys():
        raise ValueError("forecast and truth case/horizon keys do not match")

    task_rows = []
    config_errors = defaultdict(lambda: {"baseline": [], "full_hic": []})
    origin_gains = defaultdict(list)
    origin_absolute_gains = defaultdict(list)
    for key in sorted(forecast_by_key):
        dataset, case_id, horizon = key
        baseline, full_hic = forecast_by_key[key]
        truth = truth_by_key[key]
        baseline_errors = [pred - actual for pred, actual in zip(baseline, truth)]
        full_hic_errors = [pred - actual for pred, actual in zip(full_hic, truth)]
        baseline_metrics = _metrics(baseline_errors)
        full_metrics = _metrics(full_hic_errors)
        config_key = f"{dataset}/H{horizon}"
        errors = config_errors[config_key]
        errors["baseline"].extend(baseline_errors)
        errors["full_hic"].extend(full_hic_errors)
        origin_key = (dataset, case_id)
        gain_pct = _relative_gain_pct(baseline_metrics["mae"], full_metrics["mae"])
        if gain_pct is not None:
            origin_gains[origin_key].append(gain_pct)
        origin_absolute_gains[origin_key].append(
            baseline_metrics["mae"] - full_metrics["mae"]
        )
        task_rows.append({
            "dataset": dataset,
            "case_id": case_id,
            "horizon": horizon,
            "baseline_metrics": baseline_metrics,
            "full_hic_metrics": full_metrics,
            "relative_mae_gain_pct": gain_pct,
            "mae_direction": _direction(baseline_metrics["mae"], full_metrics["mae"]),
        })

    task_counts = {name: 0 for name in ("improved", "worsened", "tie")}
    for row in task_rows:
        task_counts[row["mae_direction"]] += 1
    task_counts = {"total": len(task_rows), **task_counts}

    origin_rows = []
    for (dataset, case_id), changes in sorted(origin_absolute_gains.items()):
        gains = origin_gains[(dataset, case_id)]
        relative_gain = sum(gains) / len(gains) if gains else None
        absolute_gain = sum(changes) / len(changes)
        if len(gains) != len(changes):
            # A zero baseline MAE has no defined relative percentage. Keep that
            # horizon in the origin decision by falling back to absolute gain.
            direction = _direction(absolute_gain, 0.0)
        else:
            direction = _direction(relative_gain, 0.0)
        origin_rows.append({
            "dataset": dataset,
            "case_id": case_id,
            "horizon_count": len(changes),
            "mean_relative_mae_gain_pct": relative_gain,
            "mean_absolute_mae_gain": absolute_gain,
            "mae_direction": direction,
        })
    origin_counts = {name: 0 for name in ("improved", "worsened", "tie")}
    for row in origin_rows:
        origin_counts[row["mae_direction"]] += 1
    origin_counts = {"total": len(origin_rows), **origin_counts}

    configurations = {}
    family_gains = defaultdict(lambda: {"mae": [], "mse": [], "rmse": []})
    for config_key, errors in sorted(config_errors.items()):
        baseline_metrics = _metrics(errors["baseline"])
        full_metrics = _metrics(errors["full_hic"])
        gains = {
            metric: _relative_gain_pct(baseline_metrics[metric], full_metrics[metric])
            for metric in ("mae", "mse", "rmse")
        }
        dataset = config_key.rsplit("/H", 1)[0]
        for metric, gain in gains.items():
            if gain is not None:
                family_gains[dataset][metric].append(gain)
        task_subset = [row for row in task_rows if f"{row['dataset']}/H{row['horizon']}" == config_key]
        directions = {name: sum(row["mae_direction"] == name for row in task_subset)
                      for name in ("improved", "worsened", "tie")}
        configurations[config_key] = {
            "dataset": dataset,
            "horizon": int(config_key.rsplit("/H", 1)[1]),
            "validation_windows": len(task_subset),
            "target_points": len(errors["baseline"]),
            "baseline_metrics": baseline_metrics,
            "full_hic_metrics": full_metrics,
            "relative_gain_pct": gains,
            "task_mae_counts": {"total": len(task_subset), **directions},
        }
    per_family = {
        family: {
            f"{metric}_relative_gain_pct": (
                sum(values[metric]) / len(values[metric]) if values[metric] else None
            )
            for metric in ("mae", "mse", "rmse")
        }
        for family, values in sorted(family_gains.items())
    }
    family_equalized = {
        f"{metric}_relative_gain_pct": (
            sum(row[f"{metric}_relative_gain_pct"] for row in per_family.values()
                if row[f"{metric}_relative_gain_pct"] is not None)
            / sum(row[f"{metric}_relative_gain_pct"] is not None for row in per_family.values())
            if any(row[f"{metric}_relative_gain_pct"] is not None for row in per_family.values())
            else None
        )
        for metric in ("mae", "mse", "rmse")
    }
    return {
        "task_counts": task_counts,
        "origin_counts": origin_counts,
        "unique_origins": len(origin_rows),
        "configurations": configurations,
        "per_family": per_family,
        "family_equalized_gain_pct": family_equalized,
        "task_results": task_rows,
        "origin_results": origin_rows,
    }
