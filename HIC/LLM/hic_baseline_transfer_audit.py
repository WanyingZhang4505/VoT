"""Offline, selection-only baseline choice with paired confirmation scoring."""

from __future__ import annotations

import math
from collections import Counter


NUMERIC_BASELINE_ORDER = (
    "history_mean",
    "linear_trend",
    "persistence",
    "recent_mean_3",
)


def _mae(truth, forecast, *, context: str) -> float:
    if not isinstance(truth, list) or not truth:
        raise ValueError(f"{context}: truth must be a non-empty list")
    if not isinstance(forecast, list) or len(forecast) != len(truth):
        raise ValueError(f"{context}: forecast length must match truth")
    pairs = []
    for actual, predicted in zip(truth, forecast):
        if isinstance(actual, bool) or isinstance(predicted, bool):
            raise ValueError(f"{context}: boolean values are not valid numeric targets")
        try:
            actual_value = float(actual)
            predicted_value = float(predicted)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: target and forecast must be numeric") from exc
        if not math.isfinite(actual_value) or not math.isfinite(predicted_value):
            raise ValueError(f"{context}: target and forecast values must be finite")
        pairs.append(abs(actual_value - predicted_value))
    return sum(pairs) / len(pairs)


def _rmse(truth, forecast, *, context: str) -> float:
    if not isinstance(truth, list) or not truth:
        raise ValueError(f"{context}: truth must be a non-empty list")
    if not isinstance(forecast, list) or len(forecast) != len(truth):
        raise ValueError(f"{context}: forecast length must match truth")
    squared_errors = []
    for actual, predicted in zip(truth, forecast):
        if isinstance(actual, bool) or isinstance(predicted, bool):
            raise ValueError(f"{context}: boolean values are not valid numeric targets")
        try:
            actual_value = float(actual)
            predicted_value = float(predicted)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: target and forecast must be numeric") from exc
        if not math.isfinite(actual_value) or not math.isfinite(predicted_value):
            raise ValueError(f"{context}: target and forecast values must be finite")
        squared_errors.append((actual_value - predicted_value) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def audit_reports(reports: list[dict]) -> dict:
    """Choose each config's numeric baseline on selection only; score it on confirmation."""
    if not reports:
        raise ValueError("at least one report is required")

    configurations = []
    all_confirmation = []
    seen_configs = set()

    for report in reports:
        if report.get("outer_test_targets_read") is not False:
            raise ValueError("report must explicitly assert outer-test targets were not read")
        dataset = report.get("dataset")
        horizon = report.get("horizon")
        config_key = (dataset, horizon)
        if not dataset or horizon is None or config_key in seen_configs:
            raise ValueError("each report must have a unique dataset and horizon")
        seen_configs.add(config_key)

        windows = report.get("validation_windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{dataset}/H{horizon}: validation windows are required")
        selection = [w for w in windows if w.get("development_role") == "selection"]
        confirmation = [w for w in windows if w.get("development_role") == "confirmation"]
        if not selection or not confirmation:
            raise ValueError(f"{dataset}/H{horizon}: both selection and confirmation windows are required")

        ids_by_role = {
            role: {window.get("case_id") for window in group}
            for role, group in (("selection", selection), ("confirmation", confirmation))
        }
        if None in ids_by_role["selection"] or None in ids_by_role["confirmation"]:
            raise ValueError(f"{dataset}/H{horizon}: each window needs a case_id")
        if ids_by_role["selection"] & ids_by_role["confirmation"]:
            raise ValueError(f"{dataset}/H{horizon}: selection and confirmation case IDs overlap")

        selection_scores = {}
        for arm in NUMERIC_BASELINE_ORDER:
            if all(arm in window.get("forecasts", {}) for window in selection):
                selection_scores[arm] = sum(
                    _mae(
                        window.get("truth"),
                        window["forecasts"][arm],
                        context=f"{dataset}/H{horizon}/{window['case_id']}/{arm}",
                    )
                    for window in selection
                ) / len(selection)
        if not selection_scores:
            raise ValueError(f"{dataset}/H{horizon}: no complete numeric baseline on selection")

        selected_baseline = min(
            NUMERIC_BASELINE_ORDER,
            key=lambda arm: selection_scores.get(arm, math.inf),
        )
        scored_windows = []
        for window in confirmation:
            forecasts = window.get("forecasts", {})
            required_arms = (
                "full_reasoning",
                "no_retrieval",
                "summary_retrieval",
                "low_similarity_control",
                selected_baseline,
            )
            missing = [arm for arm in required_arms if arm not in forecasts]
            if missing:
                raise ValueError(
                    f"{dataset}/H{horizon}/{window['case_id']}: missing confirmation arms {missing}"
                )
            scores = {
                arm: _mae(
                    window.get("truth"),
                    forecasts[arm],
                    context=f"{dataset}/H{horizon}/{window['case_id']}/{arm}",
                )
                for arm in required_arms
            }
            rmse_scores = {
                arm: _rmse(
                    window.get("truth"),
                    forecasts[arm],
                    context=f"{dataset}/H{horizon}/{window['case_id']}/{arm}",
                )
                for arm in required_arms
            }
            row = {
                "dataset": dataset,
                "horizon": horizon,
                "case_id": window["case_id"],
                "selected_baseline": selected_baseline,
                "hic_mae": scores["full_reasoning"],
                "hic_rmse": rmse_scores["full_reasoning"],
                "no_retrieval_mae": scores["no_retrieval"],
                "no_retrieval_rmse": rmse_scores["no_retrieval"],
                "summary_retrieval_mae": scores["summary_retrieval"],
                "summary_retrieval_rmse": rmse_scores["summary_retrieval"],
                "low_similarity_control_mae": scores["low_similarity_control"],
                "low_similarity_control_rmse": rmse_scores["low_similarity_control"],
                "selected_baseline_mae": scores[selected_baseline],
                "selected_baseline_rmse": rmse_scores[selected_baseline],
                "hic_minus_selected_baseline": scores["full_reasoning"] - scores[selected_baseline],
                "hic_minus_no_retrieval": scores["full_reasoning"] - scores["no_retrieval"],
                "hic_minus_summary_retrieval": scores["full_reasoning"] - scores["summary_retrieval"],
                "hic_minus_low_similarity_control": (
                    scores["full_reasoning"] - scores["low_similarity_control"]
                ),
                "hic_rmse_minus_selected_baseline": (
                    rmse_scores["full_reasoning"] - rmse_scores[selected_baseline]
                ),
                "hic_rmse_minus_no_retrieval": (
                    rmse_scores["full_reasoning"] - rmse_scores["no_retrieval"]
                ),
                "hic_rmse_minus_summary_retrieval": (
                    rmse_scores["full_reasoning"] - rmse_scores["summary_retrieval"]
                ),
                "hic_rmse_minus_low_similarity_control": (
                    rmse_scores["full_reasoning"] - rmse_scores["low_similarity_control"]
                ),
            }
            scored_windows.append(row)
            all_confirmation.append(row)

        selected_baseline_confirmation_mae = sum(
            row["selected_baseline_mae"] for row in scored_windows
        ) / len(scored_windows)
        hic_confirmation_mae = sum(row["hic_mae"] for row in scored_windows) / len(scored_windows)
        no_retrieval_confirmation_mae = sum(
            row["no_retrieval_mae"] for row in scored_windows
        ) / len(scored_windows)
        summary_confirmation_mae = sum(
            row["summary_retrieval_mae"] for row in scored_windows
        ) / len(scored_windows)
        low_similarity_confirmation_mae = sum(
            row["low_similarity_control_mae"] for row in scored_windows
        ) / len(scored_windows)
        selected_baseline_confirmation_rmse = sum(
            row["selected_baseline_rmse"] for row in scored_windows
        ) / len(scored_windows)
        hic_confirmation_rmse = sum(row["hic_rmse"] for row in scored_windows) / len(scored_windows)
        no_retrieval_confirmation_rmse = sum(
            row["no_retrieval_rmse"] for row in scored_windows
        ) / len(scored_windows)
        summary_confirmation_rmse = sum(
            row["summary_retrieval_rmse"] for row in scored_windows
        ) / len(scored_windows)
        low_similarity_confirmation_rmse = sum(
            row["low_similarity_control_rmse"] for row in scored_windows
        ) / len(scored_windows)
        configurations.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "selection_windows": len(selection),
                "confirmation_windows": len(confirmation),
                "selected_baseline": selected_baseline,
                "selection_mae_by_baseline": selection_scores,
                "confirmation_mae": {
                    "full_reasoning": hic_confirmation_mae,
                    "no_retrieval": no_retrieval_confirmation_mae,
                    "summary_retrieval": summary_confirmation_mae,
                    "low_similarity_control": low_similarity_confirmation_mae,
                    selected_baseline: selected_baseline_confirmation_mae,
                },
                "confirmation_rmse": {
                    "full_reasoning": hic_confirmation_rmse,
                    "no_retrieval": no_retrieval_confirmation_rmse,
                    "summary_retrieval": summary_confirmation_rmse,
                    "low_similarity_control": low_similarity_confirmation_rmse,
                    selected_baseline: selected_baseline_confirmation_rmse,
                },
                "relative_mae_improvement_vs_selected_baseline": _relative_improvement(
                    selected_baseline_confirmation_mae, hic_confirmation_mae
                ),
                "relative_mae_improvement_vs_no_retrieval": _relative_improvement(
                    no_retrieval_confirmation_mae, hic_confirmation_mae
                ),
                "relative_mae_improvement_vs_summary_retrieval": _relative_improvement(
                    summary_confirmation_mae, hic_confirmation_mae
                ),
                "relative_mae_improvement_vs_low_similarity_control": _relative_improvement(
                    low_similarity_confirmation_mae, hic_confirmation_mae
                ),
                "relative_rmse_improvement_vs_selected_baseline": _relative_improvement(
                    selected_baseline_confirmation_rmse, hic_confirmation_rmse
                ),
                "relative_rmse_improvement_vs_no_retrieval": _relative_improvement(
                    no_retrieval_confirmation_rmse, hic_confirmation_rmse
                ),
                "relative_rmse_improvement_vs_summary_retrieval": _relative_improvement(
                    summary_confirmation_rmse, hic_confirmation_rmse
                ),
                "relative_rmse_improvement_vs_low_similarity_control": _relative_improvement(
                    low_similarity_confirmation_rmse, hic_confirmation_rmse
                ),
                "hic_vs_selected_baseline": _win_tie_loss(
                    [row["hic_minus_selected_baseline"] for row in scored_windows]
                ),
                "hic_vs_no_retrieval": _win_tie_loss(
                    [row["hic_minus_no_retrieval"] for row in scored_windows]
                ),
                "hic_vs_summary_retrieval": _win_tie_loss(
                    [row["hic_minus_summary_retrieval"] for row in scored_windows]
                ),
                "hic_vs_low_similarity_control": _win_tie_loss(
                    [row["hic_minus_low_similarity_control"] for row in scored_windows]
                ),
                "hic_rmse_vs_selected_baseline": _win_tie_loss(
                    [row["hic_rmse_minus_selected_baseline"] for row in scored_windows]
                ),
                "hic_rmse_vs_no_retrieval": _win_tie_loss(
                    [row["hic_rmse_minus_no_retrieval"] for row in scored_windows]
                ),
                "hic_rmse_vs_summary_retrieval": _win_tie_loss(
                    [row["hic_rmse_minus_summary_retrieval"] for row in scored_windows]
                ),
                "hic_rmse_vs_low_similarity_control": _win_tie_loss(
                    [row["hic_rmse_minus_low_similarity_control"] for row in scored_windows]
                ),
            }
        )

    pooled_wins = _win_tie_loss([row["hic_minus_selected_baseline"] for row in all_confirmation])
    pooled_no_retrieval = _win_tie_loss([row["hic_minus_no_retrieval"] for row in all_confirmation])
    pooled_summary = _win_tie_loss([row["hic_minus_summary_retrieval"] for row in all_confirmation])
    pooled_low_similarity = _win_tie_loss(
        [row["hic_minus_low_similarity_control"] for row in all_confirmation]
    )
    pooled_rmse = {
        "vs_selected_baseline": _win_tie_loss(
            [row["hic_rmse_minus_selected_baseline"] for row in all_confirmation]
        ),
        "vs_no_retrieval": _win_tie_loss(
            [row["hic_rmse_minus_no_retrieval"] for row in all_confirmation]
        ),
        "vs_summary_retrieval": _win_tie_loss(
            [row["hic_rmse_minus_summary_retrieval"] for row in all_confirmation]
        ),
        "vs_low_similarity_control": _win_tie_loss(
            [row["hic_rmse_minus_low_similarity_control"] for row in all_confirmation]
        ),
    }
    family_relative = {}
    for dataset in sorted({config["dataset"] for config in configurations}):
        family_configs = [config for config in configurations if config["dataset"] == dataset]
        family_relative[dataset] = {
            "vs_selected_baseline": _mean_defined(
                config["relative_mae_improvement_vs_selected_baseline"]
                for config in family_configs
            ),
            "vs_no_retrieval": _mean_defined(
                config["relative_mae_improvement_vs_no_retrieval"]
                for config in family_configs
            ),
            "vs_summary_retrieval": _mean_defined(
                config["relative_mae_improvement_vs_summary_retrieval"]
                for config in family_configs
            ),
            "vs_low_similarity_control": _mean_defined(
                config["relative_mae_improvement_vs_low_similarity_control"]
                for config in family_configs
            ),
            "rmse_vs_selected_baseline": _mean_defined(
                config["relative_rmse_improvement_vs_selected_baseline"]
                for config in family_configs
            ),
            "rmse_vs_no_retrieval": _mean_defined(
                config["relative_rmse_improvement_vs_no_retrieval"]
                for config in family_configs
            ),
            "rmse_vs_summary_retrieval": _mean_defined(
                config["relative_rmse_improvement_vs_summary_retrieval"]
                for config in family_configs
            ),
            "rmse_vs_low_similarity_control": _mean_defined(
                config["relative_rmse_improvement_vs_low_similarity_control"]
                for config in family_configs
            ),
        }
    family_macro_relative = {
        key: _mean_defined(values[key] for values in family_relative.values())
        for key in (
            "vs_selected_baseline",
            "vs_no_retrieval",
            "vs_summary_retrieval",
            "vs_low_similarity_control",
        )
    }
    family_macro_relative_rmse = {
        key.removeprefix("rmse_"): value
        for key, value in {
            "rmse_vs_selected_baseline": _mean_defined(
                values["rmse_vs_selected_baseline"] for values in family_relative.values()
            ),
            "rmse_vs_no_retrieval": _mean_defined(
                values["rmse_vs_no_retrieval"] for values in family_relative.values()
            ),
            "rmse_vs_summary_retrieval": _mean_defined(
                values["rmse_vs_summary_retrieval"] for values in family_relative.values()
            ),
            "rmse_vs_low_similarity_control": _mean_defined(
                values["rmse_vs_low_similarity_control"] for values in family_relative.values()
            ),
        }.items()
    }

    return {
        "configurations": configurations,
        "summary": {
            "configurations": len(configurations),
            "confirmation_windows": len(all_confirmation),
            "hic_vs_selected_baseline": pooled_wins,
            "hic_vs_no_retrieval": pooled_no_retrieval,
            "hic_vs_summary_retrieval": pooled_summary,
            "hic_vs_low_similarity_control": pooled_low_similarity,
            "hic_rmse_vs": pooled_rmse,
            "family_macro_relative_mae_improvement": family_macro_relative,
            "family_macro_relative_rmse_improvement": family_macro_relative_rmse,
            "relative_mae_improvement_by_dataset_family": family_relative,
            "relative_rmse_improvement_by_dataset_family": {
                dataset: {
                    key.removeprefix("rmse_"): value
                    for key, value in values.items()
                    if key.startswith("rmse_")
                }
                for dataset, values in family_relative.items()
            },
        },
        "windows": all_confirmation,
        "limitations": [
            "Confirmation windows have already been inspected in prior development audits; this is not a pristine independent test.",
            "Windows can overlap across horizons and share underlying series, so pooled counts are descriptive, not independent-sample inference.",
            "The selected baseline is a static per-configuration choice based on a small selection split; it is a fallback benchmark, not a learned per-query routing rule.",
            "Target and news provenance remain incomplete; this analysis does not establish real-world target validity or causal benefit.",
        ],
    }


def _win_tie_loss(deltas: list[float]) -> dict:
    wins = sum(value < -1e-12 for value in deltas)
    ties = sum(abs(value) <= 1e-12 for value in deltas)
    losses = sum(value > 1e-12 for value in deltas)
    return {"wins": wins, "ties": ties, "losses": losses}


def _relative_improvement(baseline_mae: float, candidate_mae: float) -> float | None:
    if baseline_mae <= 1e-12:
        return None
    return (baseline_mae - candidate_mae) / baseline_mae


def _mean_defined(values) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def render_markdown(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# Selection-only numeric fallback transfer audit",
        "",
        "> Retrospective development diagnostic only. The confirmation windows have been inspected before; they are not an untouched test set. No API, embedding, Qdrant, or outer-test access is performed.",
        "",
        "## Summary",
        "",
        f"- Configurations: {summary['configurations']}; confirmation windows: {summary['confirmation_windows']}.",
        f"- HIC vs selection-chosen numeric baseline (wins/ties/losses): {summary['hic_vs_selected_baseline']['wins']}/{summary['hic_vs_selected_baseline']['ties']}/{summary['hic_vs_selected_baseline']['losses']}.",
        f"- HIC vs no-retrieval (wins/ties/losses): {summary['hic_vs_no_retrieval']['wins']}/{summary['hic_vs_no_retrieval']['ties']}/{summary['hic_vs_no_retrieval']['losses']}.",
        f"- HIC vs summary-only retrieval (wins/ties/losses): {summary['hic_vs_summary_retrieval']['wins']}/{summary['hic_vs_summary_retrieval']['ties']}/{summary['hic_vs_summary_retrieval']['losses']}.",
        f"- HIC vs low-similarity full-reasoning control (wins/ties/losses): {summary['hic_vs_low_similarity_control']['wins']}/{summary['hic_vs_low_similarity_control']['ties']}/{summary['hic_vs_low_similarity_control']['losses']}.",
        "- Relative MAE improvement (positive means HIC is better); first average configurations within each dataset family, then weight the five families equally:",
    ]
    relative = summary["family_macro_relative_mae_improvement"]
    lines.append(f"  - vs selection-chosen numeric baseline: {relative['vs_selected_baseline']:.2%}")
    lines.append(f"  - vs no-retrieval: {relative['vs_no_retrieval']:.2%}")
    lines.append(f"  - vs summary-only retrieval: {relative['vs_summary_retrieval']:.2%}")
    lines.append(f"  - vs low-similarity full-reasoning control: {relative['vs_low_similarity_control']:.2%}")
    lines.append("")
    lines.append("- Relative RMSE improvement (positive means HIC is better), using the same family-equal aggregation:")
    relative_rmse = summary["family_macro_relative_rmse_improvement"]
    lines.append(f"  - vs selection-chosen numeric baseline: {relative_rmse['vs_selected_baseline']:.2%}")
    lines.append(f"  - vs no-retrieval: {relative_rmse['vs_no_retrieval']:.2%}")
    lines.append(f"  - vs summary-only retrieval: {relative_rmse['vs_summary_retrieval']:.2%}")
    lines.append(f"  - vs low-similarity full-reasoning control: {relative_rmse['vs_low_similarity_control']:.2%}")
    lines.append("")
    lines.append("- Paired RMSE wins/ties/losses across the 54 confirmation windows (descriptive, not independent-sample inference):")
    for label, key in (
        ("selection-chosen numeric baseline", "vs_selected_baseline"),
        ("no-retrieval", "vs_no_retrieval"),
        ("summary-only retrieval", "vs_summary_retrieval"),
        ("low-similarity control", "vs_low_similarity_control"),
    ):
        lines.append(f"  - vs {label}: {_format_wtl(summary['hic_rmse_vs'][key])}")
    lines.append("")
    lines.append("| Dataset family | Configurations | vs selected baseline | vs no-retrieval | vs summary-only | vs low-similarity control |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset, values in summary["relative_mae_improvement_by_dataset_family"].items():
        family_count = sum(config["dataset"] == dataset for config in result["configurations"])
        baseline = "n/a" if values["vs_selected_baseline"] is None else f"{values['vs_selected_baseline']:.2%}"
        no_retrieval = "n/a" if values["vs_no_retrieval"] is None else f"{values['vs_no_retrieval']:.2%}"
        summary_only = "n/a" if values["vs_summary_retrieval"] is None else f"{values['vs_summary_retrieval']:.2%}"
        low_similarity = "n/a" if values["vs_low_similarity_control"] is None else f"{values['vs_low_similarity_control']:.2%}"
        lines.append(f"| {dataset} | {family_count} | {baseline} | {no_retrieval} | {summary_only} | {low_similarity} |")

    lines.extend(
        [
            "",
            "### Family-equal relative RMSE improvement",
            "",
            "| Dataset family | vs selected baseline | vs no-retrieval | vs summary-only | vs low-similarity control |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, values in summary["relative_rmse_improvement_by_dataset_family"].items():
        rendered = ["n/a" if values[key] is None else f"{values[key]:.2%}" for key in (
            "vs_selected_baseline", "vs_no_retrieval", "vs_summary_retrieval", "vs_low_similarity_control"
        )]
        lines.append(f"| {dataset} | " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## Per-configuration selection and confirmation",
            "",
            "| Dataset | Horizon | Selection n | Confirmation n | Baseline selected on selection | Baseline selection MAE | Confirmation MAE baseline | HIC MAE | HIC vs baseline W/T/L | HIC vs summary W/T/L | HIC vs low-sim W/T/L |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config in result["configurations"]:
        arm = config["selected_baseline"]
        counts = config["hic_vs_selected_baseline"]
        lines.append(
            f"| {config['dataset']} | {config['horizon']} | {config['selection_windows']} | {config['confirmation_windows']} | {arm} | {config['selection_mae_by_baseline'][arm]:.6f} | {config['confirmation_mae'][arm]:.6f} | {config['confirmation_mae']['full_reasoning']:.6f} | {counts['wins']}/{counts['ties']}/{counts['losses']} | {config['hic_vs_summary_retrieval']['wins']}/{config['hic_vs_summary_retrieval']['ties']}/{config['hic_vs_summary_retrieval']['losses']} | {config['hic_vs_low_similarity_control']['wins']}/{config['hic_vs_low_similarity_control']['ties']}/{config['hic_vs_low_similarity_control']['losses']} |"
        )

    lines.extend(
        [
            "",
            "### Per-configuration confirmation RMSE",
            "",
            "| Dataset | Horizon | Selected baseline | Baseline RMSE | HIC RMSE | HIC vs baseline W/T/L | HIC vs no-retrieval W/T/L | HIC vs summary-only W/T/L | HIC vs low-sim W/T/L |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config in result["configurations"]:
        arm = config["selected_baseline"]
        rmse = config["confirmation_rmse"]
        lines.append(
            f"| {config['dataset']} | {config['horizon']} | {arm} | {rmse[arm]:.6f} | {rmse['full_reasoning']:.6f} | "
            f"{_format_wtl(config['hic_rmse_vs_selected_baseline'])} | "
            f"{_format_wtl(config['hic_rmse_vs_no_retrieval'])} | "
            f"{_format_wtl(config['hic_rmse_vs_summary_retrieval'])} | "
            f"{_format_wtl(config['hic_rmse_vs_low_similarity_control'])} |"
        )

    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    return "\n".join(lines)


def _format_wtl(counts: dict) -> str:
    return f"{counts['wins']}/{counts['ties']}/{counts['losses']}"
