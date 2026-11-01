"""Derived per-round Agent Test 6 metrics and progress reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from statistics import mean

from LLM.hic_agent_validation import canonical_hash


METRICS = ("mae", "mse", "rmse")
COMPLETE_STATUS = "memory_appended"
TECHNICAL_STATUSES = {"technical_failure", "failed", "error", "technical_review_required"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label} in a saved Test 6 score")
    return value


def _load_round_rows(round_dir):
    root = Path(round_dir).resolve()
    manifest = _read(root / "run_manifest.json")
    jobs = _read(root / "input_snapshots.json")
    expected_count = manifest.get("job_count", len(jobs))
    if (not isinstance(expected_count, int) or len(jobs) != expected_count
            or canonical_hash(jobs) != manifest.get("jobs_sha256")):
        raise ValueError("Test 6 target-blind job list differs from its frozen manifest")
    rows = []
    for job in jobs:
        state_path = root / "cases" / f"{job['job_id']}.json"
        state = _read(state_path) if state_path.is_file() else {}
        score_path = root / "scores" / f"{job['job_id']}.json"
        score = _read(score_path) if score_path.is_file() else None
        lock_path = root / "forecast_locks" / f"{job['job_id']}.json"
        lock = _read(lock_path) if lock_path.is_file() else None
        if score is not None:
            if score.get("case_id") != job["case_id"] or score.get("input_sha256") != job["input_sha256"]:
                raise ValueError(f"score identity mismatch for {job['case_id']}")
            if not lock or score.get("forecast_lock_sha256") != lock.get("forecast_lock_sha256"):
                raise ValueError(f"score is not bound to the frozen A/B lock for {job['case_id']}")
            if state.get("forecast_lock_sha256") != score.get("forecast_lock_sha256"):
                raise ValueError(f"case state and score lock disagree for {job['case_id']}")
            score_metrics = score.get("metrics", {})
            for arm in ("no_retrieval", "full_hic"):
                for metric in METRICS:
                    _finite(score_metrics[arm][metric], f"{arm}.{metric}")
        rows.append({"job": job, "state": state, "score": score})
    return root, manifest, rows


def _relative_change(a, b):
    return None if a == 0 else (a - b) / abs(a)


def _direction(a, b, tolerance=1e-12):
    if b < a - tolerance:
        return "improved"
    if b > a + tolerance:
        return "worsened"
    return "tie"


def _metrics_summary(rows):
    scored = [row for row in rows if row["score"] is not None]
    result = {"scored_case_count": len(scored), "metrics": {}, "directions": {}}
    for metric in METRICS:
        a_values = [row["score"]["metrics"]["no_retrieval"][metric] for row in scored]
        b_values = [row["score"]["metrics"]["full_hic"][metric] for row in scored]
        changes = [b - a for a, b in zip(a_values, b_values)]
        rel = [_relative_change(a, b) for a, b in zip(a_values, b_values)]
        paired = [(a, b) for a, b in zip(a_values, b_values)]
        directions = [_direction(a, b) for a, b in paired]
        result["metrics"][metric] = {
            "no_retrieval_mean_per_case": mean(a_values) if a_values else None,
            "full_hic_mean_per_case": mean(b_values) if b_values else None,
            "mean_absolute_delta_full_minus_no_retrieval": mean(changes) if changes else None,
            "mean_relative_improvement_per_case": mean([x for x in rel if x is not None])
            if any(x is not None for x in rel) else None,
            "relative_change_of_mean_error": (
                mean(b_values) / mean(a_values) - 1
                if a_values and mean(a_values) != 0 else None
            ),
        }
        result["directions"][metric] = {
            name: directions.count(name) for name in ("improved", "tie", "worsened")
        }
    mae_rel = [
        _relative_change(row["score"]["metrics"]["no_retrieval"]["mae"],
                         row["score"]["metrics"]["full_hic"]["mae"])
        for row in scored
    ]
    material = sum(value is not None and value > 0.01 for value in mae_rel)
    degradations = []
    for row, relative in zip(scored, mae_rel):
        a = row["score"]["metrics"]["no_retrieval"]["mae"]
        b = row["score"]["metrics"]["full_hic"]["mae"]
        if b > a + 1e-12:
            degradations.append((b - a, None if relative is None else -relative,
                                 row["job"]["case_id"]))
    max_abs = max(degradations, default=None, key=lambda item: item[0])
    valid_relative = [item for item in degradations if item[1] is not None]
    max_relative = max(valid_relative, default=None, key=lambda item: item[1])
    result["mae_strict_improvements"] = result["directions"]["mae"]["improved"]
    result["mae_ties"] = result["directions"]["mae"]["tie"]
    result["mae_degradations"] = result["directions"]["mae"]["worsened"]
    result["mae_improved_or_tied"] = result["mae_strict_improvements"] + result["mae_ties"]
    result["mae_improved_or_tied_rate"] = (
        result["mae_improved_or_tied"] / len(scored) if scored else None
    )
    result["mae_material_improvement_over_1pct_count"] = material
    result["max_absolute_mae_degradation"] = (
        {"delta": max_abs[0], "case_id": max_abs[2]} if max_abs else None
    )
    result["max_relative_mae_degradation"] = (
        {"relative_change": max_relative[1], "case_id": max_relative[2]}
        if max_relative else None
    )
    result["no_memory_fallback_count"] = sum(
        row["state"].get("full_hic", {}).get("fallback_reason") == "no_eligible_memory"
        for row in rows
    )
    result["technical_failure_or_review_count"] = sum(
        row["state"].get("status") in TECHNICAL_STATUSES
        or bool(row["state"].get("technical_anomalies"))
        for row in rows
    )
    return result


def build_round_report(round_dir, *, upto_cases=None):
    root, manifest, rows = _load_round_rows(round_dir)
    planned_case_count = len(rows)
    if upto_cases is not None:
        if not isinstance(upto_cases, int) or not 0 <= upto_cases <= len(rows):
            raise ValueError("upto_cases must be an integer within the frozen case list")
        rows = rows[:upto_cases]
    complete = [row for row in rows if row["state"].get("status") == COMPLETE_STATUS]
    terminal = [row for row in rows if row["state"].get("status") in {
        COMPLETE_STATUS, "technical_failure",
    }]
    scored = [row for row in rows if row["score"] is not None]
    by_config = {}
    for name, config in manifest["configurations"].items():
        group = [row for row in rows if row["job"]["configuration"] == name]
        summary = _metrics_summary(group)
        summary.update({
            "dataset": config["dataset"], "horizon": config["horizon"],
            "case_count": len(group),
            "memory_appended_count": sum(row["state"].get("status") == COMPLETE_STATUS for row in group),
            "retrieval_miss_fallback_count": sum(
                row["state"].get("full_hic", {}).get("fallback_reason") == "no_eligible_memory"
                for row in group
            ),
        })
        by_config[name] = summary

    by_family = {}
    family_relative = {}
    for dataset in sorted({config["dataset"] for config in manifest["configurations"].values()}):
        family_configs = {
            name: summary for name, summary in by_config.items() if summary["dataset"] == dataset
        }
        family_rows = [row for row in rows if row["job"]["dataset"] == dataset]
        family = _metrics_summary(family_rows)
        family["configs"] = family_configs
        by_family[dataset] = family
        for metric in METRICS:
            family_relative.setdefault(metric, {})[dataset] = family["metrics"][metric]["relative_change_of_mean_error"]

    family_names = sorted(by_family)
    equal_family_relative = {}
    for metric in METRICS:
        values = list(family_relative.get(metric, {}).values())
        equal_family_relative[metric] = mean(values) if values and all(v is not None for v in values) else None

    per_case = []
    for row in rows:
        job, state, score = row["job"], row["state"], row["score"]
        entry = {
            "dataset": job["dataset"], "horizon": job["horizon"],
            "case_id": job["case_id"], "status": state.get("status", "not_started"),
            "memory_appended": state.get("status") == COMPLETE_STATUS,
            "retrieval_candidate_count": state.get("retrieval_trace", {}).get("candidate_count"),
            "retrieval_selected_count": len(state.get("retrieval_trace", {}).get("selected", [])),
            "fallback_reason": state.get("full_hic", {}).get("fallback_reason"),
            "technical_anomalies": state.get("technical_anomalies", []),
        }
        if score is not None:
            entry["metrics"] = score["metrics"]
            entry["truth_sha256"] = score.get("truth_sha256")
            entry["forecast_lock_sha256"] = score.get("forecast_lock_sha256")
        per_case.append(entry)

    status_counts = {}
    for row in rows:
        status = row["state"].get("status", "not_started")
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "protocol": "agent_test6_round_report_v2_ratio_of_family_means",
        "round_id": manifest["round_id"], "collection_name": manifest["collection_name"],
        "case_order": [row["job"]["case_id"] for row in rows],
        "total_case_count": len(rows), "scored_case_count": len(scored),
        "planned_case_count": planned_case_count, "terminal_case_count": len(terminal),
        "technical_failure_case_count": sum(
            row["state"].get("status") == "technical_failure" for row in rows
        ),
        "complete_case_count": len(complete), "status_counts": status_counts,
        "overall_scored": _metrics_summary(rows),
        "by_configuration": by_config, "by_family": by_family,
        "family_relative_change_of_mean_error": family_relative,
        "family_equal_weight_relative_change": equal_family_relative,
        "family_weighting_definition": (
            "For each metric, average A and B errors over scored cases equally within each family, "
            "then compute mean(B)/mean(A)-1. Give each represented family equal weight; "
            "the combined value is undefined until every family ratio exists. Negative means improvement."
        ),
        "family_names": family_names,
        "per_case": per_case,
        "validation_truth_read": bool(scored),
        "outer_test_targets_read": False,
        "report_source_manifest_sha256": canonical_hash(manifest),
    }
    return report


def render_markdown(report):
    collection_name = report.get("collection_name", "agent_test_6")
    test_label = collection_name.replace("agent_test_", "Agent Test ")
    lines = [
        f"# {test_label} {report['round_id']} 进度报告", "",
        f"已终结：{report.get('terminal_case_count', report['complete_case_count'])}/{report['planned_case_count']}；"
        f"已完成并入库：{report['complete_case_count']}；已评分：{report['scored_case_count']}；"
        f"技术失败：{report.get('technical_failure_case_count', 0)}。", "",
        "所有case是顺序累积的开发评估。家族内先对case等权计算A/B平均误差，再计算B均值/A均值−1；"
        "各数据家族等权，负值表示改善。全部家族比率未齐全时不报告完整总体值。", "",
        "## 配置指标", "",
        "| 配置 | 已评分 | MAE A→B | MSE A→B | RMSE A→B | 改善/持平/退化 | >1%改善 | 无经验回退 | 技术异常/复核 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["by_configuration"].items():
        pairs = []
        for metric in METRICS:
            values = summary["metrics"][metric]
            a, b = values["no_retrieval_mean_per_case"], values["full_hic_mean_per_case"]
            pairs.append("—" if a is None else f"{a:.6g} → {b:.6g}")
        lines.append(
            f"| {name} | {summary['scored_case_count']} | {pairs[0]} | {pairs[1]} | {pairs[2]} | "
            f"{summary['mae_strict_improvements']}/{summary['mae_ties']}/{summary['mae_degradations']} | "
            f"{summary['mae_material_improvement_over_1pct_count']} | "
            f"{summary['no_memory_fallback_count']} | {summary['technical_failure_or_review_count']} |"
        )
    overall = report["overall_scored"]
    lines += ["", "## 汇总", "",
              f"MAE改善/持平/退化：{overall['mae_strict_improvements']}/"
              f"{overall['mae_ties']}/{overall['mae_degradations']}；"
              f"改善或持平比例：{overall['mae_improved_or_tied_rate']}；"
              f">1%实质改善：{overall['mae_material_improvement_over_1pct_count']}。",
              f"最大绝对MAE退化：{overall['max_absolute_mae_degradation']}；"
              f"最大相对MAE退化：{overall['max_relative_mae_degradation']}。", "",
              "家族等权相对误差变化（负值表示改善）：", "",
              "| 指标 | " + " | ".join(report.get("family_names", [])) + " | 家族等权平均 |",
              "|---|" + "---:|" * (len(report.get("family_names", [])) + 1)]
    for metric in METRICS:
        fam = report["family_equal_weight_relative_change"]
        val = fam.get(metric)
        fmt = lambda x: "—" if x is None else f"{x:.3%}"
        family_cells = [fmt(report["family_relative_change_of_mean_error"][metric].get(name))
                        for name in report.get("family_names", [])]
        lines.append(f"| {metric.upper()} | " + " | ".join(family_cells + [fmt(val)]) + " |")
    lines += ["", "详见同目录JSON中的逐case指标、锁定hash、检索/回退状态与异常清单。", ""]
    return "\n".join(lines)


def write_checkpoint_report(round_dir, *, label, upto_cases=None):
    if not re.fullmatch(r"(?:checkpoint_\d{2,3}|config_end_\d{2,3}|final_\d{2,3})", label):
        raise ValueError("report label must be checkpoint_NN[N], config_end_NN[N], or final_NN[N]")
    root = Path(round_dir).resolve()
    _, manifest, _ = _load_round_rows(root)
    expected = int(label.rsplit("_", 1)[1])
    report = build_round_report(root, upto_cases=upto_cases or expected)
    if report.get("terminal_case_count", report["complete_case_count"]) != expected:
        raise ValueError(f"report label {label} expects {expected} terminal cases, "
                         f"found {report.get('terminal_case_count', report['complete_case_count'])}")
    if label.startswith("final_") and expected != manifest.get("job_count", 33):
        raise ValueError("final report count must equal the frozen manifest job count")
    report_path = root / "reports" / f"{label}.json"
    _write_new(report_path, report)
    markdown_path = root / "reports" / f"{label}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(report))
    return report_path, markdown_path, report
