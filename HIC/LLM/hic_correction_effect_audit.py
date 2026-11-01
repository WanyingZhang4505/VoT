"""Offline, retrospective comparison of HIC summary-only and full-reasoning arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = ROOT / "LLM" / "evaluations" / "hic_v7_gatefix_full_20261020" / "jobs"
DEFAULT_OUTPUT = ROOT / "docs" / "hic_correction_effect_audit_20261020.md"
ARMS = ("summary_retrieval", "full_reasoning")
ROLES = ("selection", "confirmation")
TOLERANCE = 1e-9


def _finite_number(value, *, context):
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number")
    return number


def _forecast_pair(window, *, context):
    truth = window.get("truth")
    forecasts = window.get("forecasts")
    if not isinstance(truth, list) or not isinstance(forecasts, dict):
        raise ValueError(f"{context}: truth and forecasts must be present")
    if any(arm not in forecasts for arm in ARMS):
        raise ValueError(f"{context}: forecasts must include {', '.join(ARMS)}")
    summary = forecasts["summary_retrieval"]
    full = forecasts["full_reasoning"]
    if not isinstance(summary, list) or not isinstance(full, list):
        raise ValueError(f"{context}: forecast arms must be arrays")
    if not truth or len(truth) != len(summary) or len(truth) != len(full):
        raise ValueError(f"{context}: truth and forecast arrays must have the same non-zero length")
    return (
        [_finite_number(value, context=f"{context}/truth") for value in truth],
        [_finite_number(value, context=f"{context}/summary_retrieval") for value in summary],
        [_finite_number(value, context=f"{context}/full_reasoning") for value in full],
    )


def _prepare_report(report):
    if not isinstance(report, dict) or report.get("outer_test_targets_read") is not False:
        raise ValueError("every report must explicitly assert outer-test targets were not read")
    dataset = report.get("dataset")
    horizon = report.get("horizon")
    if not isinstance(dataset, str) or not dataset.strip() or type(horizon) is not int or horizon <= 0:
        raise ValueError("every report needs a dataset name and positive integer horizon")
    windows = report.get("validation_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"{dataset}/H{horizon}: validation windows are required")
    prepared = {"dataset": dataset, "horizon": horizon, "windows": {role: [] for role in ROLES}}
    seen_ids = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError(f"{dataset}/H{horizon}: every validation window must be an object")
        role = window.get("development_role")
        case_id = window.get("case_id")
        if role not in ROLES:
            raise ValueError(f"{dataset}/H{horizon}: unexpected or missing development role")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"{dataset}/H{horizon}: case IDs must be non-empty and unique")
        seen_ids.add(case_id)
        truth, summary, full = _forecast_pair(
            window, context=f"{dataset}/H{horizon}/{case_id}"
        )
        retrieval = window.get("retrieval")
        similarity = _finite_number(
            retrieval.get("score") if isinstance(retrieval, dict) else None,
            context=f"{dataset}/H{horizon}/{case_id}/retrieval_score",
        )
        if not 0.0 <= similarity <= 1.0:
            raise ValueError(f"{dataset}/H{horizon}/{case_id}: retrieval score must be in [0, 1]")
        point_closer = point_same = point_farther = changed = 0
        shift_total = summary_abs_error = full_abs_error = 0.0
        for actual, summary_prediction, full_prediction in zip(truth, summary, full):
            summary_error = abs(summary_prediction - actual)
            full_error = abs(full_prediction - actual)
            error_delta = full_error - summary_error
            difference = abs(full_prediction - summary_prediction)
            summary_abs_error += summary_error
            full_abs_error += full_error
            shift_total += difference
            changed += difference > TOLERANCE
            if error_delta < -TOLERANCE:
                point_closer += 1
            elif error_delta > TOLERANCE:
                point_farther += 1
            else:
                point_same += 1
        point_count = len(truth)
        summary_mae = summary_abs_error / point_count
        full_mae = full_abs_error / point_count
        window_row = {
            "case_id": case_id,
            "similarity": similarity,
            "point_count": point_count,
            "summary_mae": summary_mae,
            "full_reasoning_mae": full_mae,
            "mae_delta": full_mae - summary_mae,
            "points_moved_closer": point_closer,
            "points_same_error": point_same,
            "points_moved_farther": point_farther,
            "forecast_points_changed": changed,
            "forecast_points_unchanged": point_count - changed,
            "mean_absolute_forecast_shift": shift_total / point_count,
        }
        prepared["windows"][role].append(window_row)
    if any(not prepared["windows"][role] for role in ROLES):
        raise ValueError(f"{dataset}/H{horizon}: both selection and confirmation windows are required")
    return prepared


def _aggregate_windows(windows):
    if not windows:
        return {
            "window_count": 0,
            "target_points": 0,
            "summary_mae": None,
            "full_reasoning_mae": None,
            "full_reasoning_vs_summary_mae_delta": None,
            "points_moved_closer": 0,
            "points_same_error": 0,
            "points_moved_farther": 0,
            "forecast_points_changed": 0,
            "forecast_points_unchanged": 0,
            "mean_absolute_forecast_shift": None,
            "windows_wins_ties_losses": [0, 0, 0],
        }
    point_count = sum(row["point_count"] for row in windows)
    summary_error = sum(row["summary_mae"] * row["point_count"] for row in windows)
    full_error = sum(row["full_reasoning_mae"] * row["point_count"] for row in windows)
    shift = sum(row["mean_absolute_forecast_shift"] * row["point_count"] for row in windows)
    wins = ties = losses = 0
    for row in windows:
        if row["mae_delta"] < -TOLERANCE:
            wins += 1
        elif row["mae_delta"] > TOLERANCE:
            losses += 1
        else:
            ties += 1
    summary_mae = summary_error / point_count
    full_mae = full_error / point_count
    return {
        "window_count": len(windows),
        "target_points": point_count,
        "summary_mae": summary_mae,
        "full_reasoning_mae": full_mae,
        "full_reasoning_vs_summary_mae_delta": full_mae - summary_mae,
        "points_moved_closer": sum(row["points_moved_closer"] for row in windows),
        "points_same_error": sum(row["points_same_error"] for row in windows),
        "points_moved_farther": sum(row["points_moved_farther"] for row in windows),
        "forecast_points_changed": sum(row["forecast_points_changed"] for row in windows),
        "forecast_points_unchanged": sum(row["forecast_points_unchanged"] for row in windows),
        "mean_absolute_forecast_shift": shift / point_count,
        "windows_wins_ties_losses": [wins, ties, losses],
    }


def _family_macro(configurations, role):
    by_family = defaultdict(list)
    zero_baseline_configurations = 0
    for config in configurations:
        rows = config["windows"][role]
        summary_mae = statistics.mean(row["summary_mae"] for row in rows)
        full_mae = statistics.mean(row["full_reasoning_mae"] for row in rows)
        if summary_mae <= TOLERANCE:
            zero_baseline_configurations += 1
            continue
        by_family[config["dataset"]].append(100.0 * (summary_mae - full_mae) / summary_mae)
    family_scores = {
        family: statistics.mean(values) for family, values in sorted(by_family.items())
    }
    return {
        "relative_improvement_pct": statistics.mean(family_scores.values()) if family_scores else None,
        "family_count": len(family_scores),
        "families": family_scores,
        "zero_summary_error_configuration_count_excluded": zero_baseline_configurations,
    }


def audit_reports(reports):
    """Compare two paired arms in saved validation reports; never accesses source data."""
    if not reports:
        raise ValueError("at least one report is required")
    configurations = [_prepare_report(report) for report in reports]
    keys = [(item["dataset"], item["horizon"]) for item in configurations]
    if len(keys) != len(set(keys)):
        raise ValueError("each dataset/horizon configuration must be unique")

    roles = {}
    for role in (*ROLES, "all_development"):
        windows = [
            row
            for config in configurations
            for selected_role in (ROLES if role == "all_development" else (role,))
            for row in config["windows"][selected_role]
        ]
        roles[role] = {
            **_aggregate_windows(windows),
            "family_macro": _family_macro(configurations, role) if role != "all_development" else _family_macro_all(configurations),
        }
    similarity_pairwise = {
        role: _similarity_pairwise(configurations, role)
        for role in (*ROLES, "all_development")
    }

    rows = []
    for config in sorted(configurations, key=lambda item: (item["dataset"], item["horizon"])):
        row = {"dataset": config["dataset"], "horizon": config["horizon"]}
        for role in ROLES:
            row[role] = _aggregate_windows(config["windows"][role])
        rows.append(row)
    return {
        "configuration_count": len(configurations),
        "roles": roles,
        "similarity_pairwise": similarity_pairwise,
        "configurations": rows,
        "interpretation": (
            "Retrospective paired comparison. The full_reasoning prompt adds the retrieved full correction and explicit applicability instructions; "
            "therefore this measures the complete prompt-arm difference, not an isolated causal effect of correction text."
        ),
    }


def _family_macro_all(configurations):
    combined = []
    for config in configurations:
        combined.append({
            "dataset": config["dataset"],
            "windows": {
                "all": config["windows"]["selection"] + config["windows"]["confirmation"]
            },
        })
    return _family_macro(combined, "all")


def _similarity_pairwise(configurations, role):
    """Compare similarity ordering to correction benefit within each config only."""
    helpful = harmful = similarity_ties = effect_ties = comparable = 0
    configs_with_pairs = 0
    for config in configurations:
        roles = ROLES if role == "all_development" else (role,)
        windows = [row for selected_role in roles for row in config["windows"][selected_role]]
        config_comparable = 0
        for index, first in enumerate(windows):
            for second in windows[index + 1:]:
                score_delta = first["similarity"] - second["similarity"]
                if abs(score_delta) <= 1e-12:
                    similarity_ties += 1
                    continue
                effect_delta = first["mae_delta"] - second["mae_delta"]
                if abs(effect_delta) <= TOLERANCE:
                    effect_ties += 1
                    continue
                comparable += 1
                config_comparable += 1
                high_score_effect = first["mae_delta"] if score_delta > 0 else second["mae_delta"]
                low_score_effect = second["mae_delta"] if score_delta > 0 else first["mae_delta"]
                if high_score_effect < low_score_effect:
                    helpful += 1
                else:
                    harmful += 1
        configs_with_pairs += config_comparable > 0
    return {
        "comparable_pairs": comparable,
        "higher_similarity_more_helpful_pairs": helpful,
        "higher_similarity_more_harmful_pairs": harmful,
        "effect_tie_pairs_excluded": effect_ties,
        "similarity_tie_pairs_excluded": similarity_ties,
        "configurations_with_comparable_pairs": configs_with_pairs,
        "net_directional_association": (helpful - harmful) / comparable if comparable else None,
    }


def load_reports(report_dir):
    report_dir = Path(report_dir)
    paths = sorted(report_dir.glob("*.report.json"))
    if not paths:
        raise ValueError(f"no *.report.json files found in {report_dir}")
    reports = []
    source_hashes = {}
    for path in paths:
        raw = path.read_bytes()
        try:
            report = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}: invalid UTF-8 JSON report") from exc
        reports.append(report)
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    return reports, source_hashes


def _fmt_number(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(result, source_hashes):
    lines = [
        "# HIC correction-arm effect audit (retrospective)",
        "",
        "> Offline analysis of saved v7 development reports only. No API, embedding, Qdrant, or outer-test access occurred.",
        "",
        "## Comparison",
        "",
        "Each validation window has paired `summary_retrieval` and `full_reasoning` forecasts using the same current input and retrieved top-1 memory. The full-reasoning prompt additionally receives the saved correction and applicability instructions. This is a comparison of the full prompt arms, not an isolated causal estimate of the correction text.",
        "",
        "MAE is recomputed from saved forecast and truth arrays. Positive MAE improvement means full reasoning has lower error than summary-only. Point-level closer/farther counts compare each forecast point's absolute error. Windows are overlapping/dependent; confirmation was inspected in prior audits and is retrospective, not an independent test.",
        "",
        "## Aggregate results",
        "",
        "| Slice | Windows | Target points | Summary-only MAE | Full-reasoning MAE | Full − summary MAE | Windows W/T/L | Points closer / same / farther | Forecast points changed | Mean absolute forecast shift | Family-macro relative improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in (*ROLES, "all_development"):
        stats = result["roles"][role]
        wtl = "/".join(map(str, stats["windows_wins_ties_losses"]))
        moved = f"{stats['points_moved_closer']} / {stats['points_same_error']} / {stats['points_moved_farther']}"
        macro = stats["family_macro"]["relative_improvement_pct"]
        macro_text = f"{macro:+.3f}%" if macro is not None else "n/a"
        lines.append(
            f"| {role} | {stats['window_count']} | {stats['target_points']} | "
            f"{_fmt_number(stats['summary_mae'])} | {_fmt_number(stats['full_reasoning_mae'])} | "
            f"{_fmt_number(stats['full_reasoning_vs_summary_mae_delta'])} | {wtl} | {moved} | "
            f"{stats['forecast_points_changed']}/{stats['target_points']} | "
            f"{_fmt_number(stats['mean_absolute_forecast_shift'])} | {macro_text} |"
        )
    lines.extend([
        "",
        "Family-macro relative improvement gives equal weight to dataset families after averaging windows within each configuration and configurations within each family. Configurations with zero summary-only MAE are excluded from this percentage and counted below; absolute MAE rows still include them.",
        "",
        "## Per-configuration breakdown",
        "",
        "| Dataset | Horizon | Role | Windows | Summary MAE | Full MAE | Full − summary | W/T/L | Closer / same / farther points | Changed points |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for config in result["configurations"]:
        for role in ROLES:
            stats = config[role]
            wtl = "/".join(map(str, stats["windows_wins_ties_losses"]))
            moved = f"{stats['points_moved_closer']}/{stats['points_same_error']}/{stats['points_moved_farther']}"
            lines.append(
                f"| {config['dataset']} | {config['horizon']} | {role} | {stats['window_count']} | "
                f"{_fmt_number(stats['summary_mae'])} | {_fmt_number(stats['full_reasoning_mae'])} | "
                f"{_fmt_number(stats['full_reasoning_vs_summary_mae_delta'])} | {wtl} | {moved} | "
                f"{stats['forecast_points_changed']}/{stats['target_points']} |"
            )
    lines.extend([
        "",
        "## Does higher cosine similarity identify more useful corrections",
        "",
        "To reduce cross-dataset scale confounding, compare pairs only within the same dataset × horizon configuration. For each pair with unequal cosine and unequal correction effect, a pair is counted as helpful-aligned when the higher-cosine case has a lower `full_reasoning MAE − summary_retrieval MAE`; it is harmful-aligned when the higher-cosine case has a higher delta. Pairs reuse the same windows and are dependent; pair counts are not independent sample sizes. This is descriptive rank concordance, not a calibrated probability or causal test.",
        "",
        "| Slice | Comparable pairs | Higher similarity more helpful | Higher similarity more harmful | Similarity ties excluded | Effect ties excluded | Configurations represented | Net directional association |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for role in (*ROLES, "all_development"):
        stats = result["similarity_pairwise"][role]
        net = stats["net_directional_association"]
        net_text = "n/a" if net is None else f"{net:+.3f}"
        lines.append(
            f"| {role} | {stats['comparable_pairs']} | {stats['higher_similarity_more_helpful_pairs']} | "
            f"{stats['higher_similarity_more_harmful_pairs']} | {stats['similarity_tie_pairs_excluded']} | "
            f"{stats['effect_tie_pairs_excluded']} | {stats['configurations_with_comparable_pairs']} | {net_text} |"
        )
    lines.extend(["", "## Family-level relative improvement", ""])
    for role in ROLES:
        macro = result["roles"][role]["family_macro"]
        lines.append(f"### {role}")
        lines.append("")
        lines.append(f"Equal-family macro: {_fmt_number(macro['relative_improvement_pct'], 3)}%; families: {macro['family_count']}; zero-summary-error configurations excluded: {macro['zero_summary_error_configuration_count_excluded']}.")
        lines.append("")
        lines.append("| Family | Relative MAE improvement |")
        lines.append("|---|---:|")
        for family, value in macro["families"].items():
            lines.append(f"| {family} | {value:+.3f}% |")
        lines.append("")
    lines.extend([
        "## Interpretation limits",
        "",
        "- The comparison is retrospective and descriptive; development confirmation windows have already been inspected.",
        "- Temporal windows overlap and are not independent observations; no significance claim is made.",
        "- Similarity-pair comparisons reuse windows within each configuration; they are dependent and only describe rank direction.",
        "- A changed forecast moving closer to truth is evidence that the full prompt arm helped on that point, not proof that a retrieved correction was causally responsible.",
        "- These reports do not establish strict news provenance, forecast-cutoff availability, or correction applicability. Do not deploy a HIC route based on this audit alone.",
        "- The outer-test remains sealed; all input reports assert `outer_test_targets_read: false`.",
        "",
        "## Source report hashes",
        "",
    ])
    for name, digest in sorted(source_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    reports, source_hashes = load_reports(args.report_dir)
    result = audit_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result, source_hashes), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"configurations={result['configuration_count']}")
    for role in (*ROLES, "all_development"):
        stats = result["roles"][role]
        macro = stats["family_macro"]["relative_improvement_pct"]
        similarity = result["similarity_pairwise"][role]
        print(
            f"{role}: windows={stats['window_count']} points={stats['target_points']} "
            f"summary_mae={stats['summary_mae']:.8g} full_mae={stats['full_reasoning_mae']:.8g} "
            f"delta={stats['full_reasoning_vs_summary_mae_delta']:.8g} "
            f"window_wtl={stats['windows_wins_ties_losses']} "
            f"point_closer_same_farther={[stats['points_moved_closer'], stats['points_same_error'], stats['points_moved_farther']]} "
            f"family_macro_improvement={macro} "
            f"similarity_pairs={similarity['comparable_pairs']} "
            f"high_score_helpful_harmful={[similarity['higher_similarity_more_helpful_pairs'], similarity['higher_similarity_more_harmful_pairs']]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
