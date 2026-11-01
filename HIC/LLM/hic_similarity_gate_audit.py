"""Offline test of a single retrieval-similarity gate against numeric baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


NUMERIC_BASELINES = (
    "history_mean",
    "linear_trend",
    "persistence",
    "recent_mean_3",
)
NO_HIC_THRESHOLD = math.nextafter(1.0, math.inf)


def _finite_nonnegative(value, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: MAE must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: MAE must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context}: MAE must be a finite non-negative number")
    return number


def _mae(window: dict, arm: str, *, context: str) -> float:
    metrics = window.get("metrics")
    metric = metrics.get(arm) if isinstance(metrics, dict) else None
    if not isinstance(metric, dict) or "mae" not in metric:
        raise ValueError(f"{context}: missing per-window MAE for {arm}")
    return _finite_nonnegative(metric["mae"], context=f"{context}/{arm}")


def _similarity(window: dict, *, context: str) -> float:
    retrieval = window.get("retrieval")
    score = retrieval.get("score") if isinstance(retrieval, dict) else None
    if isinstance(score, bool):
        raise ValueError(f"{context}: retrieval similarity must be finite and in [0, 1]")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: retrieval similarity must be finite and in [0, 1]") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{context}: retrieval similarity must be finite and in [0, 1]")
    return value


def _prepare_config(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("outer_test_targets_read") is not False:
        raise ValueError("every report must explicitly assert outer-test targets were not read")
    dataset = report.get("dataset")
    horizon = report.get("horizon")
    if not isinstance(dataset, str) or not dataset.strip() or type(horizon) is not int or horizon <= 0:
        raise ValueError("each report needs a dataset name and positive integer horizon")
    windows = report.get("validation_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"{dataset}/H{horizon}: validation windows are required")

    selection = [window for window in windows if window.get("development_role") == "selection"]
    confirmation = [window for window in windows if window.get("development_role") == "confirmation"]
    if not selection or not confirmation:
        raise ValueError(f"{dataset}/H{horizon}: selection and confirmation windows are required")
    if len(selection) + len(confirmation) != len(windows):
        raise ValueError(f"{dataset}/H{horizon}: unexpected development role in validation windows")

    ids_by_role = {}
    for role, group in (("selection", selection), ("confirmation", confirmation)):
        ids = [window.get("case_id") for window in group]
        if any(not isinstance(case_id, str) or not case_id for case_id in ids):
            raise ValueError(f"{dataset}/H{horizon}: every {role} window needs a case ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset}/H{horizon}: duplicate {role} case IDs")
        ids_by_role[role] = set(ids)
    if ids_by_role["selection"] & ids_by_role["confirmation"]:
        raise ValueError(f"{dataset}/H{horizon}: selection and confirmation case IDs overlap")

    selection_mae = {}
    for arm in NUMERIC_BASELINES:
        try:
            values = [
                _mae(window, arm, context=f"{dataset}/H{horizon}/{window['case_id']}")
                for window in selection
            ]
        except ValueError:
            continue
        selection_mae[arm] = statistics.mean(values)
    if not selection_mae:
        raise ValueError(f"{dataset}/H{horizon}: no complete numeric baseline on selection")
    selected_baseline = min(
        NUMERIC_BASELINES,
        key=lambda arm: selection_mae.get(arm, math.inf),
    )

    prepared = {"dataset": dataset, "horizon": horizon, "selected_numeric_baseline": selected_baseline}
    for role, group in (("selection", selection), ("confirmation", confirmation)):
        prepared[role] = []
        for window in group:
            context = f"{dataset}/H{horizon}/{window['case_id']}"
            prepared[role].append(
                {
                    "case_id": window["case_id"],
                    "similarity": _similarity(window, context=context),
                    "baseline_mae": _mae(window, selected_baseline, context=context),
                    "hic_mae": _mae(window, "full_reasoning", context=context),
                }
            )
    if statistics.mean(row["baseline_mae"] for row in prepared["selection"]) <= 0:
        raise ValueError(f"{dataset}/H{horizon}: selection baseline MAE must be positive")
    return prepared


def _candidate_thresholds(configs: list[dict]) -> list[float]:
    scores = sorted({row["similarity"] for config in configs for row in config["selection"]})
    if not scores:
        raise ValueError("selection windows contain no retrieval scores")
    midpoints = [(left + right) / 2.0 for left, right in zip(scores, scores[1:])]
    return sorted({0.0, *midpoints, NO_HIC_THRESHOLD})


def _macro_metrics(configs: list[dict], role: str, threshold: float) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    routed_count = total_count = 0
    for config in configs:
        rows = config[role]
        total_count += len(rows)
        routed_count += sum(row["similarity"] >= threshold for row in rows)
        baseline_mae = statistics.mean(row["baseline_mae"] for row in rows)
        routed_mae = statistics.mean(
            row["hic_mae"] if row["similarity"] >= threshold else row["baseline_mae"]
            for row in rows
        )
        if baseline_mae <= 0:
            raise ValueError(
                f"{config['dataset']}/H{config['horizon']}/{role}: baseline MAE must be positive"
            )
        by_family[config["dataset"]].append(
            {
                "horizon": config["horizon"],
                "relative_improvement_pct": 100.0 * (baseline_mae - routed_mae) / baseline_mae,
                "baseline_mae": baseline_mae,
                "routed_mae": routed_mae,
                "window_count": len(rows),
            }
        )

    family_results = {}
    for dataset, config_rows in sorted(by_family.items()):
        family_results[dataset] = {
            "relative_improvement_pct": statistics.mean(
                item["relative_improvement_pct"] for item in config_rows
            ),
            "configuration_count": len(config_rows),
        }
    macro_improvement = statistics.mean(
        family["relative_improvement_pct"] for family in family_results.values()
    )
    return {
        "macro_relative_mae_improvement_pct": macro_improvement,
        "family_results": family_results,
        "routed_hic_count": routed_count,
        "window_count": total_count,
    }


def _win_tie_loss(configs: list[dict], threshold: float) -> tuple[list[int], list[int]]:
    hic_wtl = [0, 0, 0]
    policy_wtl = [0, 0, 0]
    for config in configs:
        for row in config["confirmation"]:
            difference = row["hic_mae"] - row["baseline_mae"]
            is_routed = row["similarity"] >= threshold
            if is_routed and abs(difference) <= 1e-12:
                hic_wtl[1] += 1
            elif is_routed and difference < 0:
                hic_wtl[0] += 1
            elif is_routed:
                hic_wtl[2] += 1
            if not is_routed:
                policy_wtl[1] += 1
            elif abs(difference) <= 1e-12:
                policy_wtl[1] += 1
            elif difference < 0:
                policy_wtl[0] += 1
            else:
                policy_wtl[2] += 1
    return hic_wtl, policy_wtl


def audit_reports(reports: list[dict]) -> dict:
    """Choose a global similarity cutoff on selection windows and freeze it for confirmation."""
    if not reports:
        raise ValueError("at least one report is required")
    configs = [_prepare_config(report) for report in reports]
    config_keys = [(config["dataset"], config["horizon"]) for config in configs]
    if len(config_keys) != len(set(config_keys)):
        raise ValueError("each dataset/horizon configuration must be unique")

    thresholds = _candidate_thresholds(configs)
    candidate_results = [
        (_macro_metrics(configs, "selection", threshold)["macro_relative_mae_improvement_pct"], threshold)
        for threshold in thresholds
    ]
    # On ties, prefer the stricter threshold, which routes fewer cases to HIC.
    selected_gain, selected_threshold = max(candidate_results, key=lambda item: (item[0], item[1]))

    selection = _macro_metrics(configs, "selection", selected_threshold)
    confirmation = _macro_metrics(configs, "confirmation", selected_threshold)
    all_hic = _macro_metrics(configs, "confirmation", 0.0)
    all_baseline = _macro_metrics(configs, "confirmation", NO_HIC_THRESHOLD)
    hic_wtl, policy_wtl = _win_tie_loss(configs, selected_threshold)
    configuration_results = []
    for config in sorted(configs, key=lambda item: (item["dataset"], item["horizon"])):
        configuration_results.append(
            {
                "dataset": config["dataset"],
                "horizon": config["horizon"],
                "selected_numeric_baseline": config["selected_numeric_baseline"],
                "selection_windows": len(config["selection"]),
                "confirmation_windows": len(config["confirmation"]),
                "selection_routed_hic_count": sum(
                    row["similarity"] >= selected_threshold for row in config["selection"]
                ),
                "confirmation_routed_hic_count": sum(
                    row["similarity"] >= selected_threshold for row in config["confirmation"]
                ),
            }
        )

    return {
        "configuration_count": len(configs),
        "selection_window_count": selection["window_count"],
        "confirmation_window_count": confirmation["window_count"],
        "selection_score_range": [
            min(row["similarity"] for config in configs for row in config["selection"]),
            max(row["similarity"] for config in configs for row in config["selection"]),
        ],
        "candidate_threshold_count": len(thresholds),
        "selected_threshold": selected_threshold,
        "selection_macro_relative_mae_improvement_pct": selected_gain,
        "selection_routed_hic_count": selection["routed_hic_count"],
        "confirmation_macro_relative_mae_improvement_pct": confirmation[
            "macro_relative_mae_improvement_pct"
        ],
        "confirmation_family_results": confirmation["family_results"],
        "confirmation_routed_hic_count": confirmation["routed_hic_count"],
        "confirmation_hic_routed_wins_ties_losses_vs_baseline": hic_wtl,
        "confirmation_policy_wins_ties_losses_vs_baseline": policy_wtl,
        "all_hic_confirmation_macro_relative_mae_improvement_pct": all_hic[
            "macro_relative_mae_improvement_pct"
        ],
        "all_baseline_confirmation_macro_relative_mae_improvement_pct": all_baseline[
            "macro_relative_mae_improvement_pct"
        ],
        "configurations": configuration_results,
        "outer_test_targets_read": False,
        "uses_api_embedding_or_qdrant": False,
        "reads_raw_truth_or_news": False,
        "interpretation": (
            "Retrospective diagnostic only. The saved confirmation split has been inspected before; "
            "it is not an independent final test. The score-only threshold is not a validated routing policy."
        ),
    }


def audit_report_paths(report_dir: Path) -> tuple[dict, list[dict]]:
    paths = sorted(Path(report_dir).glob("*.report.json"))
    if not paths:
        raise FileNotFoundError(f"no report JSON files found under {report_dir}")
    reports = []
    provenance = []
    for path in paths:
        raw = path.read_bytes()
        reports.append(json.loads(raw.decode("utf-8")))
        provenance.append({"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    result = audit_reports(reports)
    result["source_reports"] = provenance
    return result, paths


def render_markdown(result: dict, *, command: str) -> str:
    selection = result["selection_macro_relative_mae_improvement_pct"]
    confirmation = result["confirmation_macro_relative_mae_improvement_pct"]
    lines = [
        "# Similarity Threshold Transfer Audit",
        "",
        "Date: 2026-10-20 (Toronto)",
        "",
        "> Retrospective development diagnostic. The saved confirmation windows have already been inspected in prior audits, so this is not an untouched final test. The audit uses the recorded per-window MAE and cosine score only; it does not read raw news or truth arrays, call DeepSeek, run embeddings, access Qdrant, or read outer-test targets.",
        "",
        "## Question and frozen protocol",
        "",
        "Can one global cosine-similarity cutoff, selected only on the 61 selection windows, route HIC on helpful-looking cases and otherwise fall back to the numeric baseline?",
        "",
        "For each dataset × horizon configuration, the numeric baseline is selected by lowest mean selection-window MAE from history mean, linear trend, persistence, and recent mean 3. A global threshold is then selected to maximize the five-family-equal relative MAE improvement on selection windows. If candidate thresholds tie, the higher cutoff is chosen. The baseline and cutoff are frozen before scoring confirmation windows. The HIC arm is used when cosine similarity is greater than or equal to the cutoff; otherwise the selected numeric baseline is used.",
        "",
        "Relative MAE is computed by averaging window MAEs within each configuration, comparing the gated MAE with that configuration's selected-baseline MAE, averaging configurations within each family, and then weighting dataset families equally. Positive values mean lower error than the numeric baseline.",
        "",
        "## Results",
        "",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Configurations | {result['configuration_count']} |",
        f"| Selection windows | {result['selection_window_count']} |",
        f"| Confirmation windows | {result['confirmation_window_count']} |",
        f"| Selection cosine range | {result['selection_score_range'][0]:.6f}–{result['selection_score_range'][1]:.6f} |",
        f"| Candidate cutoffs tested | {result['candidate_threshold_count']} |",
        f"| Cutoff selected on selection | {result['selected_threshold']:.9f} |",
        f"| Selection family-macro relative MAE improvement | {selection:+.3f}% |",
        f"| Confirmation HIC-routed windows | {result['confirmation_routed_hic_count']}/{result['confirmation_window_count']} |",
        f"| Confirmation family-macro relative MAE improvement | {confirmation:+.3f}% |",
        f"| Always-HIC confirmation improvement | {result['all_hic_confirmation_macro_relative_mae_improvement_pct']:+.3f}% |",
        f"| Always-numeric-baseline confirmation improvement | {result['all_baseline_confirmation_macro_relative_mae_improvement_pct']:+.3f}% |",
        f"| Routed HIC windows vs numeric baseline, wins/ties/losses | {'/'.join(map(str, result['confirmation_hic_routed_wins_ties_losses_vs_baseline']))} |",
        f"| Whole routing policy vs numeric baseline, wins/ties/losses | {'/'.join(map(str, result['confirmation_policy_wins_ties_losses_vs_baseline']))} |",
        "",
        "### Confirmation results by family",
        "",
        "| Dataset family | Relative MAE improvement | Configurations |",
        "|---|---:|---:|",
    ]
    for dataset, values in sorted(result["confirmation_family_results"].items()):
        lines.append(
            f"| {dataset} | {values['relative_improvement_pct']:+.3f}% | {values['configuration_count']} |"
        )
    lines.extend(
        [
            "",
            "### Selected numeric fallback by configuration",
            "",
            "| Dataset | Horizon | Baseline | Selection n | Confirmation n | HIC routes in selection | HIC routes in confirmation |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for config in result["configurations"]:
        lines.append(
            f"| {config['dataset']} | {config['horizon']} | {config['selected_numeric_baseline']} | "
            f"{config['selection_windows']} | {config['confirmation_windows']} | "
            f"{config['selection_routed_hic_count']} | {config['confirmation_routed_hic_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The cutoff produces a small selection-only gain ({selection:+.3f}%), but after freezing it, the already-inspected confirmation windows show {confirmation:+.3f}% relative MAE change. It routes HIC on only {result['confirmation_routed_hic_count']} of {result['confirmation_window_count']} confirmation windows; the routed HIC cases have the win/tie/loss record shown above. This does not support cosine similarity as a reliable standalone applicability gate.",
            "",
            "The practical fallback remains evidence-based: require a verified task/target/frequency/horizon match, forecast-cutoff and news provenance/relevance, plus an independently specified correction-applicability rule. If those checks fail, return the numeric baseline. Current strict-evidence readiness remains 0/385 fit and 0/115 validation cases, so this threshold experiment is a diagnostic, not deployable routing evidence.",
            "",
            "All calculations use stored development reports. Confirmation outcomes are not independent and the target/news contracts remain incomplete. Do not tune or claim generalization from these windows; preserve the sealed outer-test for one final evaluation after evidence and policy are frozen.",
            "",
            "## Reproduction",
            "",
            "From `_research_gnn_working`:",
            "",
            "```powershell",
            command,
            "```",
            "",
            "Source report SHA-256 values:",
            "",
        ]
    )
    for item in result.get("source_reports", []):
        lines.append(f"- `{item['name']}`: `{item['sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("LLM/evaluations/hic_v7_gatefix_full_20261020/jobs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hic_similarity_threshold_transfer_audit_20261020.md"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result, _ = audit_report_paths(args.report_dir)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "python -m LLM.hic_similarity_gate_audit "
        f"--report-dir {args.report_dir.as_posix()} "
        f"--output {args.output.as_posix()} --overwrite"
    )
    args.output.write_text(render_markdown(result, command=command), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_threshold": result["selected_threshold"],
                "selection_macro_relative_mae_improvement_pct": result[
                    "selection_macro_relative_mae_improvement_pct"
                ],
                "confirmation_macro_relative_mae_improvement_pct": result[
                    "confirmation_macro_relative_mae_improvement_pct"
                ],
                "confirmation_routed_hic_count": result["confirmation_routed_hic_count"],
                "confirmation_window_count": result["confirmation_window_count"],
                "outer_test_targets_read": False,
                "api_or_qdrant_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
