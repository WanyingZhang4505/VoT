"""Descriptive audit of completed fit-only HIC calibration checkpoints."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from LLM.hic_vnext_calibration import (
    _CALIBRATION_DATASETS,
    _CALIBRATION_HORIZONS,
    _calibration_variant_keys,
)


def _finite_metric(value, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{field} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


def _config_id(dataset: str, horizon: int) -> str:
    return f"{dataset}/H{horizon}"


def summarize_partial_fit_checkpoints(checkpoints):
    """Summarize completed query scores without selecting calibration parameters.

    Scores are aggregated as: query means within a configuration, configuration
    means within each dataset family, and then an equal mean across families
    that have completed fit data. An incomplete 20-configuration matrix is
    explicitly marked and must not be treated as the final selector result.
    """
    if not isinstance(checkpoints, (list, tuple)):
        raise ValueError("checkpoints must be a list or tuple")
    expected_variants = _calibration_variant_keys()
    expected_configs = {
        (dataset, horizon)
        for dataset, horizons in _CALIBRATION_HORIZONS.items()
        for horizon in horizons
    }
    seen_checkpoint_configs = set()
    complete_records = defaultdict(list)
    records_by_origin = defaultdict(list)
    records_by_config_origin = defaultdict(list)
    query_status_counts = Counter()
    unique_cases = set()

    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("each checkpoint must be a mapping")
        dataset = checkpoint.get("dataset")
        horizon = checkpoint.get("horizon")
        if dataset not in _CALIBRATION_DATASETS:
            raise ValueError(f"unexpected dataset: {dataset}")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("checkpoint horizon must be a positive integer")
        config = (dataset, horizon)
        if config not in expected_configs:
            raise ValueError(f"unexpected calibration configuration: {_config_id(*config)}")
        if config in seen_checkpoint_configs:
            raise ValueError(f"duplicate checkpoint configuration: {_config_id(*config)}")
        seen_checkpoint_configs.add(config)
        queries = checkpoint.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"checkpoint {_config_id(*config)} is missing query records")

        for query_id, query in queries.items():
            if not isinstance(query, dict):
                raise ValueError(f"query {query_id} must be a mapping")
            status = query.get("status")
            query_status_counts[str(status)] += 1
            if status != "complete":
                continue

            case_id = query.get("memory", {}).get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"complete query {query_id} is missing memory.case_id")
            unique_cases.add((dataset, case_id))
            scores = query.get("scores")
            if not isinstance(scores, dict) or not isinstance(scores.get("baseline"), dict):
                raise ValueError(f"complete query {query_id} is missing baseline scores")
            baseline = {
                metric: _finite_metric(
                    scores["baseline"].get(metric),
                    field=f"{query_id}.baseline.{metric}",
                )
                for metric in ("mae", "mse")
            }
            variants = scores.get("variants")
            if not isinstance(variants, dict) or set(variants) != expected_variants:
                raise ValueError(f"complete query {query_id} does not contain the complete 12-variant grid")
            normalized_variants = {}
            for key, variant in variants.items():
                if not isinstance(variant, dict):
                    raise ValueError(f"variant {key} for {query_id} must be a mapping")
                reason = variant.get("fallback_reason")
                if reason is not None and not isinstance(reason, str):
                    raise ValueError(f"variant {key} for {query_id} has an invalid fallback reason")
                normalized_variants[key] = {
                    "mae": _finite_metric(variant.get("mae"), field=f"{query_id}.{key}.mae", allow_zero=True),
                    "mse": _finite_metric(variant.get("mse"), field=f"{query_id}.{key}.mse", allow_zero=True),
                    "fallback_reason": reason,
                }
            record = {"baseline": baseline, "variants": normalized_variants}
            complete_records[config].append(record)
            records_by_origin[(dataset, case_id)].append(record)
            records_by_config_origin[(dataset, horizon, case_id)].append(record)

    configuration_rows = {}
    for config, records in complete_records.items():
        row = {
            "query_count": len(records),
            "baseline": {
                metric: sum(record["baseline"][metric] for record in records) / len(records)
                for metric in ("mae", "mse")
            },
            "variants": {},
        }
        for key in sorted(expected_variants):
            wins = losses = ties = 0
            fallbacks = Counter()
            metric_sums = Counter()
            for record in records:
                baseline_mae = record["baseline"]["mae"]
                variant = record["variants"][key]
                metric_sums["mae"] += variant["mae"]
                metric_sums["mse"] += variant["mse"]
                if variant["mae"] < baseline_mae - 1e-12:
                    wins += 1
                elif math.isclose(variant["mae"], baseline_mae, rel_tol=0.0, abs_tol=1e-12):
                    ties += 1
                else:
                    losses += 1
                fallbacks[variant["fallback_reason"] or "applied"] += 1
            row["variants"][key] = {
                "mae": metric_sums["mae"] / len(records),
                "mse": metric_sums["mse"] / len(records),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "fallback_counts": dict(sorted(fallbacks.items())),
            }
        configuration_rows[config] = row

    family_configs = defaultdict(list)
    for (dataset, horizon), row in configuration_rows.items():
        family_configs[dataset].append(((dataset, horizon), row))

    candidate_metrics = {}
    for key in sorted(expected_variants):
        family_metrics = {}
        for dataset in _CALIBRATION_DATASETS:
            rows = [row for _, row in family_configs.get(dataset, [])]
            if not rows:
                continue
            baseline_mae = sum(row["baseline"]["mae"] for row in rows) / len(rows)
            baseline_mse = sum(row["baseline"]["mse"] for row in rows) / len(rows)
            variant_mae = sum(row["variants"][key]["mae"] for row in rows) / len(rows)
            variant_mse = sum(row["variants"][key]["mse"] for row in rows) / len(rows)
            win_rate = sum(
                row["variants"][key]["wins"] / row["query_count"] for row in rows
            ) / len(rows)
            family_metrics[dataset] = {
                "configuration_count": len(rows),
                "fit_horizon_members": sum(row["query_count"] for row in rows),
                "mae_improvement_pct": (baseline_mae - variant_mae) / baseline_mae * 100,
                "mse_improvement_pct": (baseline_mse - variant_mse) / baseline_mse * 100,
                "mean_configuration_case_win_rate": win_rate,
            }

        if family_metrics:
            family_count = len(family_metrics)
            macro_mae = sum(row["mae_improvement_pct"] for row in family_metrics.values()) / family_count
            macro_mse = sum(row["mse_improvement_pct"] for row in family_metrics.values()) / family_count
            macro_win_rate = sum(
                row["mean_configuration_case_win_rate"] for row in family_metrics.values()
            ) / family_count
        else:
            macro_mae = macro_mse = macro_win_rate = 0.0

        outcomes = Counter()
        fallback_counts = Counter()
        for _, row in configuration_rows.items():
            variant = row["variants"][key]
            for outcome in ("wins", "losses", "ties"):
                outcomes[outcome] += variant[outcome]
            fallback_counts.update(variant["fallback_counts"])
        origin_improvements_by_family = defaultdict(list)
        origin_mse_improvements_by_family = defaultdict(list)
        origin_outcomes = Counter()
        origin_outcomes_by_family = defaultdict(Counter)
        for (dataset, _case_id), records in records_by_origin.items():
            member_improvements = [
                (record["baseline"]["mae"] - record["variants"][key]["mae"])
                / record["baseline"]["mae"]
                * 100
                for record in records
            ]
            origin_improvement = sum(member_improvements) / len(member_improvements)
            origin_improvements_by_family[dataset].append(origin_improvement)
            member_mse_improvements = [
                (record["baseline"]["mse"] - record["variants"][key]["mse"])
                / record["baseline"]["mse"]
                * 100
                for record in records
            ]
            origin_mse_improvements_by_family[dataset].append(
                sum(member_mse_improvements) / len(member_mse_improvements)
            )
            if origin_improvement > 1e-12:
                origin_outcomes["wins"] += 1
                origin_outcomes_by_family[dataset]["wins"] += 1
            elif math.isclose(origin_improvement, 0.0, rel_tol=0.0, abs_tol=1e-12):
                origin_outcomes["ties"] += 1
                origin_outcomes_by_family[dataset]["ties"] += 1
            else:
                origin_outcomes["losses"] += 1
                origin_outcomes_by_family[dataset]["losses"] += 1

        unique_origin_family_metrics = {}
        for dataset, improvements in origin_improvements_by_family.items():
            mse_improvements = origin_mse_improvements_by_family[dataset]
            counts = origin_outcomes_by_family[dataset]
            unique_origin_family_metrics[dataset] = {
                "unique_origin_count": len(improvements),
                "mae_improvement_pct": sum(improvements) / len(improvements),
                "mse_improvement_pct": sum(mse_improvements) / len(mse_improvements),
                "wins": counts["wins"],
                "losses": counts["losses"],
                "ties": counts["ties"],
                "win_rate": counts["wins"] / len(improvements),
            }

        config_origin_values = defaultdict(lambda: {"mae": [], "mse": [], "outcomes": Counter()})
        for (dataset, horizon, _case_id), records in records_by_config_origin.items():
            mae_changes = [
                (record["baseline"]["mae"] - record["variants"][key]["mae"])
                / record["baseline"]["mae"]
                * 100
                for record in records
            ]
            mse_changes = [
                (record["baseline"]["mse"] - record["variants"][key]["mse"])
                / record["baseline"]["mse"]
                * 100
                for record in records
            ]
            mae_change = sum(mae_changes) / len(mae_changes)
            mse_change = sum(mse_changes) / len(mse_changes)
            values = config_origin_values[(dataset, horizon)]
            values["mae"].append(mae_change)
            values["mse"].append(mse_change)
            if mae_change > 1e-12:
                values["outcomes"]["wins"] += 1
            elif math.isclose(mae_change, 0.0, rel_tol=0.0, abs_tol=1e-12):
                values["outcomes"]["ties"] += 1
            else:
                values["outcomes"]["losses"] += 1
        unique_origin_configuration_metrics = {}
        for (dataset, horizon), values in config_origin_values.items():
            origin_count = len(values["mae"])
            counts = values["outcomes"]
            unique_origin_configuration_metrics[_config_id(dataset, horizon)] = {
                "unique_origin_count": origin_count,
                "mae_improvement_pct": sum(values["mae"]) / origin_count,
                "mse_improvement_pct": sum(values["mse"]) / origin_count,
                "wins": counts["wins"],
                "losses": counts["losses"],
                "ties": counts["ties"],
                "win_rate": counts["wins"] / origin_count,
            }
        family_origin_win_rates = [
            sum(value > 1e-12 for value in improvements) / len(improvements)
            for improvements in origin_improvements_by_family.values()
        ]
        family_origin_mean_improvements = [
            sum(improvements) / len(improvements)
            for improvements in origin_improvements_by_family.values()
        ]
        family_origin_mean_mse_improvements = [
            sum(improvements) / len(improvements)
            for improvements in origin_mse_improvements_by_family.values()
        ]
        candidate_metrics[key] = {
            "available_family_macro_mae_improvement_pct": macro_mae,
            "available_family_macro_mse_improvement_pct": macro_mse,
            "available_family_macro_case_win_rate": macro_win_rate,
            "available_family_count": len(family_metrics),
            "family_metrics": family_metrics,
            "case_outcomes": {name: outcomes[name] for name in ("wins", "losses", "ties")},
            "unique_origin_count": len(records_by_origin),
            "unique_origin_outcomes": {
                name: origin_outcomes[name] for name in ("wins", "losses", "ties")
            },
            "unique_origin_macro_case_win_rate": (
                sum(family_origin_win_rates) / len(family_origin_win_rates)
                if family_origin_win_rates
                else 0.0
            ),
            "unique_origin_macro_mae_improvement_pct": (
                sum(family_origin_mean_improvements) / len(family_origin_mean_improvements)
                if family_origin_mean_improvements
                else 0.0
            ),
            "unique_origin_macro_mse_improvement_pct": (
                sum(family_origin_mean_mse_improvements)
                / len(family_origin_mean_mse_improvements)
                if family_origin_mean_mse_improvements
                else 0.0
            ),
            "unique_origin_family_metrics": unique_origin_family_metrics,
            "unique_origin_configuration_metrics": unique_origin_configuration_metrics,
            "fallback_counts": dict(sorted(fallback_counts.items())),
        }

    observed_complete_configs = set(configuration_rows)
    missing_configs = sorted(expected_configs - observed_complete_configs)
    checkpoint_only_configs = sorted(seen_checkpoint_configs - observed_complete_configs)
    return {
        "protocol": "hic_vnext_partial_fit_diagnostic_v1",
        "parameter_selection_performed": False,
        "coverage": {
            "expected_configuration_count": len(expected_configs),
            "checkpoint_configuration_count": len(seen_checkpoint_configs),
            "observed_configuration_count": len(observed_complete_configs),
            "complete_configuration_matrix": not missing_configs,
            "observed_configurations": [
                _config_id(*config) for config in sorted(observed_complete_configs)
            ],
            "missing_or_incomplete_configurations": [
                _config_id(*config) for config in missing_configs
            ],
            "checkpoint_configs_without_completed_queries": [
                _config_id(*config) for config in checkpoint_only_configs
            ],
            "families_with_completed_data": [
                dataset for dataset in _CALIBRATION_DATASETS if family_configs.get(dataset)
            ],
        },
        "observations": {
            "fit_horizon_members": sum(row["query_count"] for row in configuration_rows.values()),
            "unique_dataset_case_ids": len(unique_cases),
            "query_status_counts": dict(sorted(query_status_counts.items())),
            "aggregation": "query mean within config; equal config mean within family; equal mean across available families; unique-origin outcomes average per-query relative MAE changes across completed horizons within each (dataset, case_id), then weight origins equally within a dataset and datasets equally",
        },
        "candidate_metrics": candidate_metrics,
    }


def render_partial_fit_report(result):
    """Render a concise Markdown summary from a partial-fit audit result."""
    coverage = result["coverage"]
    observations = result["observations"]
    lines = [
        "# HIC vNext partial fit grid diagnostic",
        "",
        "This is a descriptive fit-only diagnostic from saved prequential checkpoints. It does not select parameters and is not a validation or outer-test result.",
        "",
        f"- Completed configurations: {coverage['observed_configuration_count']}/{coverage['expected_configuration_count']}",
        f"- Completed fit horizon members: {observations['fit_horizon_members']}",
        f"- Unique `(dataset, case_id)` pairs: {observations['unique_dataset_case_ids']}",
        f"- Completed-data families: {', '.join(coverage['families_with_completed_data']) or 'none'}",
        f"- Missing or incomplete configurations: {', '.join(coverage['missing_or_incomplete_configurations']) or 'none'}",
        f"- Full matrix complete: {coverage['complete_configuration_matrix']}",
        f"- Query statuses: `{observations['query_status_counts']}`",
        "",
        "MAE/MSE improvement is `(No Retrieval - variant) / No Retrieval`; positive values mean the bounded-correction variant is lower. The family macro weights available dataset families equally, and configurations equally within each family.",
        "",
        "| Variant | Horizon-member macro MAE improvement | Horizon-member macro MSE improvement | Horizon-member family-equal win rate | Unique-origin macro MAE improvement | Unique-origin macro MSE improvement | Unique-origin family-equal win rate | Unique-origin wins / losses / ties | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, metrics in sorted(result["candidate_metrics"].items()):
        outcomes = metrics["case_outcomes"]
        origin_outcomes = metrics["unique_origin_outcomes"]
        counts = metrics["fallback_counts"]
        lines.append(
            f"| `{key}` | {metrics['available_family_macro_mae_improvement_pct']:.3f}% | "
            f"{metrics['available_family_macro_mse_improvement_pct']:.3f}% | "
            f"{metrics['available_family_macro_case_win_rate']:.1%} | "
            f"{metrics['unique_origin_macro_mae_improvement_pct']:.3f}% | "
            f"{metrics['unique_origin_macro_mse_improvement_pct']:.3f}% | "
            f"{metrics['unique_origin_macro_case_win_rate']:.1%} | "
            f"{origin_outcomes['wins']} / {origin_outcomes['losses']} / {origin_outcomes['ties']} | `{counts}` |"
        )
    config_ids = sorted(
        {
            config_id
            for metrics in result["candidate_metrics"].values()
            for config_id in metrics["unique_origin_configuration_metrics"]
        }
    )
    lines.extend(
        [
            "",
            "## Family and horizon diagnostics across the 12-candidate grid",
            "",
            "Each configuration reports the range across all 12 candidates; this is descriptive and does not select a candidate.",
            "",
            "| Dataset / horizon | Unique fit origins | Candidates with lower origin-mean MAE | Candidates with lower MAE and no-worse MSE | Origin win-rate range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for config_id in config_ids:
        rows = [
            metrics["unique_origin_configuration_metrics"][config_id]
            for metrics in result["candidate_metrics"].values()
            if config_id in metrics["unique_origin_configuration_metrics"]
        ]
        origin_count = rows[0]["unique_origin_count"]
        mae_better = sum(row["mae_improvement_pct"] > 1e-12 for row in rows)
        both_better = sum(
            row["mae_improvement_pct"] > 1e-12 and row["mse_improvement_pct"] >= -1e-12
            for row in rows
        )
        win_rates = [row["win_rate"] for row in rows]
        lines.append(
            f"| `{config_id}` | {origin_count} | {mae_better}/12 | {both_better}/12 | "
            f"{min(win_rates):.1%}–{max(win_rates):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Horizon members can repeat the same forecast origin across different horizons; member-level wins are not independent case counts.",
            "- Unique-origin outcomes average each query's relative MAE improvement across completed horizons within `(dataset, case_id)`, then count each origin once; this is a descriptive de-duplication rule, not a substitute for the pre-registered final selector.",
            "- Missing configurations and families make these partial aggregates unsuitable for selecting or freezing a global parameter set.",
            "- These scores reuse saved fit truths only after each prequential forecast checkpoint; they do not establish validation or real-world accuracy.",
            "- The full 20-configuration fit matrix, followed by a frozen validation comparison, remains outstanding.",
            "",
        ]
    )
    return "\n".join(lines)


