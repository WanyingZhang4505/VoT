"""Audit partial, fit-only HIC vNext calibration results without freezing a model."""

from __future__ import annotations

import math
import re


HORIZONS = {
    "Agriculture": (6, 8, 10, 12),
    "Climate": (6, 8, 10, 12),
    "Economy_sort": (6, 8, 10, 12),
    "Energy": (12, 24, 36, 48),
    "SocialGood": (6, 8, 10, 12),
}
VARIANTS = {
    f"k{k}_a{alpha:g}_c{bound:g}"
    for k in (1, 3)
    for alpha in (0.25, 0.5, 1.0)
    for bound in (0.5, 1.0)
}
_VARIANT_PATTERN = re.compile(r"^k([13])_a(0\.25|0\.5|1)_c(0\.5|1)$")


def _finite_metric(container, key, *, allow_zero):
    value = container.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid calibration metric: {key}")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"calibration metric out of range: {key}")
    return float(value)


def _family_metrics(records, variant):
    baseline_mae = sum(row["baseline_aggregate"]["mean_case_mae"] for row in records) / len(records)
    baseline_mse = sum(row["baseline_aggregate"]["mean_case_mse"] for row in records) / len(records)
    variant_mae = sum(row["variant_aggregates"][variant]["mean_case_mae"] for row in records) / len(records)
    variant_mse = sum(row["variant_aggregates"][variant]["mean_case_mse"] for row in records) / len(records)
    win_rate = sum(
        row["variant_aggregates"][variant]["mae_wins"] / row["completed_queries"]
        for row in records
    ) / len(records)
    return {
        "configuration_count": len(records),
        "fit_query_memberships": sum(row["completed_queries"] for row in records),
        "baseline_mae": baseline_mae,
        "variant_mae": variant_mae,
        "mae_relative_change": (variant_mae - baseline_mae) / baseline_mae,
        "baseline_mse": baseline_mse,
        "variant_mse": variant_mse,
        "mse_relative_improvement": (baseline_mse - variant_mse) / baseline_mse,
        "case_win_rate": win_rate,
    }


def summarize_partial_results(results):
    """Summarize available per-config fit results; never select a final candidate.

    A family contributes to candidate screening only when all four frozen
    horizons are present. Even with complete families, this report is partial
    development evidence and `final_candidate_frozen` is always false.
    """
    if not isinstance(results, (list, tuple)) or not results:
        raise ValueError("at least one configuration result is required")

    by_family = {name: [] for name in HORIZONS}
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("each result must be a mapping")
        dataset = result.get("dataset")
        horizon = result.get("horizon")
        if dataset not in HORIZONS or type(horizon) is not int or horizon not in HORIZONS[dataset]:
            raise ValueError("result does not belong to the frozen 20-configuration matrix")
        config = (dataset, horizon)
        if config in seen:
            raise ValueError(f"duplicate fit result: {dataset}/H{horizon}")
        seen.add(config)
        if result.get("protocol") != "hic_vnext_prequential_fit_v2":
            raise ValueError(f"unexpected fit result protocol: {dataset}/H{horizon}")
        if result.get("validation_truth_read") is not False or result.get("outer_test_targets_read") is not False:
            raise ValueError(f"holdout exposure is flagged in result: {dataset}/H{horizon}")

        query_count = result.get("completed_queries")
        baseline = result.get("baseline_aggregate")
        variants = result.get("variant_aggregates")
        if type(query_count) is not int or query_count <= 0 or not isinstance(baseline, dict):
            raise ValueError(f"missing fit query aggregates: {dataset}/H{horizon}")
        if baseline.get("query_count") != query_count:
            raise ValueError(f"baseline query count mismatch: {dataset}/H{horizon}")
        for metric in ("mean_case_mae", "mean_case_mse"):
            _finite_metric(baseline, metric, allow_zero=False)
        if not isinstance(variants, dict) or set(variants) != VARIANTS:
            raise ValueError(f"result must contain the complete 12-variant grid: {dataset}/H{horizon}")

        for key, aggregate in variants.items():
            match = _VARIANT_PATTERN.fullmatch(key)
            if not match or not isinstance(aggregate, dict):
                raise ValueError(f"invalid variant aggregate: {dataset}/H{horizon}/{key}")
            if aggregate.get("query_count") != query_count:
                raise ValueError(f"variant query count mismatch: {dataset}/H{horizon}/{key}")
            for metric in ("mean_case_mae", "mean_case_mse"):
                _finite_metric(aggregate, metric, allow_zero=True)
            outcomes = [aggregate.get(name) for name in ("mae_wins", "mae_ties", "mae_losses")]
            if any(type(value) is not int or value < 0 for value in outcomes):
                raise ValueError(f"invalid win/tie/loss counts: {dataset}/H{horizon}/{key}")
            if sum(outcomes) != query_count:
                raise ValueError(f"incomplete win/tie/loss counts: {dataset}/H{horizon}/{key}")
        by_family[dataset].append(result)

    complete_families = []
    incomplete_families = {}
    for dataset, records in by_family.items():
        horizons = sorted(item["horizon"] for item in records)
        if horizons == list(HORIZONS[dataset]):
            complete_families.append(dataset)
        elif horizons:
            incomplete_families[dataset] = {
                "available_horizons": horizons,
                "required_horizons": list(HORIZONS[dataset]),
            }

    family_results = {dataset: by_family[dataset] for dataset in complete_families}
    ranked = []
    if family_results:
        for variant in sorted(VARIANTS):
            family_metrics = {
                dataset: _family_metrics(records, variant)
                for dataset, records in family_results.items()
            }
            macro_mae_improvement = sum(
                -metric["mae_relative_change"] for metric in family_metrics.values()
            ) / len(family_metrics)
            macro_mse_improvement = sum(
                metric["mse_relative_improvement"] for metric in family_metrics.values()
            ) / len(family_metrics)
            macro_win_rate = sum(
                metric["case_win_rate"] for metric in family_metrics.values()
            ) / len(family_metrics)
            eligible = (
                macro_mae_improvement > 1e-12
                and macro_mse_improvement >= -1e-12
                and all(metric["mae_relative_change"] <= 0.02 + 1e-12
                        for metric in family_metrics.values())
            )
            match = _VARIANT_PATTERN.fullmatch(variant)
            ranked.append({
                "variant": variant,
                "memory_count": int(match.group(1)),
                "alpha": float(match.group(2)),
                "bound_scale": float(match.group(3)),
                "partial_guard_eligible": eligible,
                "macro_family_win_rate": macro_win_rate,
                "macro_family_mae_improvement": macro_mae_improvement,
                "macro_family_mse_improvement": macro_mse_improvement,
                "families": family_metrics,
            })
        ranked.sort(key=lambda row: (
            not row["partial_guard_eligible"],
            -row["macro_family_win_rate"],
            -row["macro_family_mae_improvement"],
            row["memory_count"],
            row["alpha"],
            row["bound_scale"],
        ))

    configs = [
        {"dataset": dataset, "horizon": horizon, "fit_queries": result["completed_queries"]}
        for (dataset, horizon), result in sorted(
            ((key, value) for family, records in by_family.items() for value in records
             for key in [(family, value["horizon"])]),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    return {
        "schema_version": 1,
        "status": "partial_fit_diagnostic_not_final_calibration",
        "configuration_count": len(seen),
        "configuration_total": 20,
        "configuration_coverage_fraction": len(seen) / 20,
        "coverage": {
            "configurations": f"{len(seen)}/20",
            "fraction": len(seen) / 20,
        },
        "fit_query_memberships": sum(result["completed_queries"] for records in by_family.values() for result in records),
        "fit_query_memberships_are_unique_across_configurations": False,
        "complete_families": complete_families,
        "incomplete_families": incomplete_families,
        "families_without_results": [name for name, records in by_family.items() if not records],
        "final_candidate_frozen": False,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "provider_calls": 0,
        "embedding_calls": 0,
        "qdrant_writes": 0,
        "configurations": configs,
        "ranked_candidates": ranked,
    }
