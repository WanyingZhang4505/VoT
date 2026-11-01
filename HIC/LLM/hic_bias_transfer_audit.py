"""Post-hoc analysis of whether retrieved bias corrections helped validation cases."""

from __future__ import annotations

import json
import math


def _finite_sequence(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{field} must contain only finite numbers")
        result.append(float(item))
    return result


def _mean(values):
    return sum(values) / len(values)


def _parse_correction(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("retrieved correction is not valid JSON") from None
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("retrieved correction must use schema_version 2")
    facts = value.get("verified_facts")
    if not isinstance(facts, dict) or facts.get("schema_version") != 1:
        raise ValueError("retrieved correction must contain verified_facts schema_version 1")
    bias = facts.get("mean_signed_error_prediction_minus_truth")
    if type(bias) not in (int, float) or not math.isfinite(float(bias)):
        raise ValueError("retrieved correction is missing a finite historical signed error")
    return float(bias)


def diagnose_window(window):
    """Compare signed-bias movement and MAE using one already-scored validation window.

    This is retrospective only: it uses validation truth after forecasts were saved.
    It must never be used to choose or apply a correction at prediction time.
    """
    if not isinstance(window, dict):
        raise ValueError("window must be a JSON object")
    case_id = window.get("case_id")
    role = window.get("development_role")
    retrieval = window.get("retrieval")
    forecasts = window.get("forecasts")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("window case_id is missing")
    if role not in {"selection", "confirmation"}:
        raise ValueError("window development_role must be selection or confirmation")
    if not isinstance(retrieval, dict):
        raise ValueError("window retrieval is missing")
    memory_id = retrieval.get("case_id")
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("retrieved memory case_id is missing")
    score = retrieval.get("score")
    if type(score) not in (int, float) or not math.isfinite(float(score)):
        raise ValueError("retrieval score must be finite")
    memory_bias = _parse_correction(retrieval.get("correction"))
    truth = _finite_sequence(window.get("truth"), "validation truth")
    if not isinstance(forecasts, dict):
        raise ValueError("window forecasts are missing")
    baseline = _finite_sequence(forecasts.get("no_retrieval"), "no_retrieval forecast")
    hic = _finite_sequence(forecasts.get("full_reasoning"), "full_reasoning forecast")
    if len(truth) != len(baseline) or len(truth) != len(hic):
        raise ValueError("truth and paired forecasts must have the same length")

    baseline_bias = _mean([prediction - actual for prediction, actual in zip(baseline, truth)])
    hic_bias = _mean([prediction - actual for prediction, actual in zip(hic, truth)])
    mean_shift = _mean([corrected - original for corrected, original in zip(hic, baseline)])
    baseline_mae = _mean([abs(prediction - actual) for prediction, actual in zip(baseline, truth)])
    hic_mae = _mean([abs(prediction - actual) for prediction, actual in zip(hic, truth)])
    delta_mae = hic_mae - baseline_mae

    stored_metrics = window.get("metrics")
    if stored_metrics is not None:
        if not isinstance(stored_metrics, dict):
            raise ValueError("stored metrics must be an object")
        for arm, recomputed in (("no_retrieval", baseline_mae), ("full_reasoning", hic_mae)):
            arm_metrics = stored_metrics.get(arm)
            stored = arm_metrics.get("mae") if isinstance(arm_metrics, dict) else None
            if type(stored) not in (int, float) or not math.isfinite(float(stored)):
                raise ValueError(f"stored {arm} MAE is missing or invalid")
            if not math.isclose(float(stored), recomputed, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"recomputed {arm} MAE differs from the saved report")

    return {
        "case_id": case_id,
        "development_role": role,
        "retrieved_memory_id": memory_id,
        "retrieval_score": float(score),
        "memory_mean_signed_error": memory_bias,
        "baseline_mean_error": baseline_bias,
        "mean_forecast_shift": mean_shift,
        "hic_mean_error": hic_bias,
        "baseline_mae": baseline_mae,
        "hic_mae": hic_mae,
        "delta_mae_hic_minus_baseline": delta_mae,
        "shift_opposes_memory_bias": memory_bias * mean_shift < 0,
        "bias_magnitude_reduced": abs(hic_bias) < abs(baseline_bias),
        "hic_improves_mae": delta_mae < 0,
    }


def audit_report(report):
    """Aggregate one completed development report; refuse outer-test artifacts."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if report.get("outer_test_targets_read") is not False:
        raise ValueError("report must explicitly confirm outer-test targets were not read")
    dataset = report.get("dataset")
    horizon = report.get("horizon")
    fit_ids = report.get("fit_case_ids")
    windows = report.get("validation_windows")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("report dataset is missing")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("report horizon must be a positive integer")
    if not isinstance(fit_ids, list) or not windows:
        raise ValueError("report fit_case_ids and validation_windows are required")
    fit_id_set = set(fit_ids)
    rows = [diagnose_window(window) for window in windows]
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("validation case IDs must be unique")
    if any(row["retrieved_memory_id"] not in fit_id_set for row in rows):
        raise ValueError("retrieved memories must belong to the fixed fit set")

    wins = sum(row["hic_improves_mae"] for row in rows)
    ties = sum(row["delta_mae_hic_minus_baseline"] == 0 for row in rows)
    bias_reduced = [row for row in rows if row["bias_magnitude_reduced"]]
    bias_not_reduced = [row for row in rows if not row["bias_magnitude_reduced"]]
    summary = {
        "windows": len(rows),
        "hic_mae_wins": wins,
        "hic_mae_ties": ties,
        "hic_mae_losses": len(rows) - wins - ties,
        "memory_bias_opposed_by_forecast_shift": sum(
            row["shift_opposes_memory_bias"] for row in rows
        ),
        "bias_magnitude_reduced_windows": len(bias_reduced),
        "bias_reduced_and_mae_wins": sum(row["hic_improves_mae"] for row in bias_reduced),
        "bias_reduced_but_mae_not_win": sum(not row["hic_improves_mae"] for row in bias_reduced),
        "bias_not_reduced_and_mae_losses": sum(not row["hic_improves_mae"] for row in bias_not_reduced),
        "bias_not_reduced_but_mae_win": sum(row["hic_improves_mae"] for row in bias_not_reduced),
    }
    return {"dataset": dataset, "horizon": horizon, "windows": rows, "summary": summary}


def render_markdown(audits):
    if not isinstance(audits, list) or not audits:
        raise ValueError("audits must be a non-empty list")
    lines = [
        "# HIC historical-bias transfer diagnostic (retrospective development analysis)",
        "",
        "> This audit uses already-scored validation truth after paired forecasts were saved. It is exploratory development analysis, not an online applicability rule, a clean confirmation test, or an outer-test result. The report emits derived metrics only, not raw truth arrays or news text.",
        "",
    ]
    for audit in audits:
        summary = audit["summary"]
        lines.extend([
            f"## {audit['dataset']} / H{audit['horizon']}",
            "",
            f"- Windows: {summary['windows']}; HIC MAE wins/ties/losses vs no-retrieval: {summary['hic_mae_wins']}/{summary['hic_mae_ties']}/{summary['hic_mae_losses']}.",
            f"- The full-HIC forecast shift opposed the retrieved memory's historical signed-error direction in {summary['memory_bias_opposed_by_forecast_shift']}/{summary['windows']} windows.",
            f"- Query-level mean signed-bias magnitude decreased in {summary['bias_magnitude_reduced_windows']}/{summary['windows']} windows; among those, {summary['bias_reduced_and_mae_wins']} also had lower MAE and {summary['bias_reduced_but_mae_not_win']} did not. When signed-bias magnitude did not decrease, MAE worsened in {summary['bias_not_reduced_and_mae_losses']}/{summary['windows']} windows.",
            "",
            "| Role | Query | Retrieved memory | Cosine | Memory signed error | Baseline signed error | HIC mean shift | HIC signed error | MAE delta (HIC minus no retrieval) | Bias magnitude reduced? | MAE improved? |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for row in audit["windows"]:
            lines.append(
                f"| {row['development_role']} | `{row['case_id']}` | `{row['retrieved_memory_id']}` | {row['retrieval_score']:.4f} | {row['memory_mean_signed_error']:.4f} | {row['baseline_mean_error']:.4f} | {row['mean_forecast_shift']:.4f} | {row['hic_mean_error']:.4f} | {row['delta_mae_hic_minus_baseline']:.4f} | {row['bias_magnitude_reduced']} | {row['hic_improves_mae']} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation boundary",
        "",
        "A correction can move the forecast opposite to the historical memory's signed error and still worsen the new case. The validation truth makes the direction of the query's baseline error observable retrospectively, but that direction is unavailable at prediction time. Therefore these counts diagnose transfer mismatch; they do not authorize a truth-dependent routing rule. A deployable gate must use only pre-cutoff information, be selected on development data, and then be frozen before a separate eligible test.",
        "",
    ])
    return "\n".join(lines)
