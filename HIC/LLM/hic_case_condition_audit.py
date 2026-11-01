"""Read-only, retrospective audit of retrieved HIC correction applicability."""

from __future__ import annotations

import json
import math

from LLM.hic_correction_consistency import find_horizon_claim_mismatches


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def profile_history(series):
    """Describe observed history changes without assigning an applicability label."""
    if not isinstance(series, (list, tuple)) or not series:
        raise ValueError("series must be a non-empty numeric sequence")
    values = [_finite(value, "series value") for value in series]
    deltas = [right - left for left, right in zip(values, values[1:])]
    signs = "".join("+" if value > 0 else "-" if value < 0 else "0" for value in deltas)
    trailing_declines = 0
    for value in reversed(deltas):
        if value >= 0:
            break
        trailing_declines += 1
    return {
        "history_length": len(values),
        "delta_signs": signs,
        "trailing_decline_steps": trailing_declines,
        "last_two_deltas": deltas[-2:],
    }


def parse_verified_correction(value):
    """Return the verified facts and model-written hypotheses from a v2 record."""
    if isinstance(value, str):
        try:
            record = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("correction must be valid JSON") from None
    else:
        record = value
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        raise ValueError("correction must use schema_version 2")
    facts = record.get("verified_facts")
    reflection = record.get("reflection")
    if not isinstance(facts, dict) or facts.get("schema_version") != 1:
        raise ValueError("correction must contain verified_facts schema_version 1")
    if not isinstance(reflection, dict):
        raise ValueError("correction must contain a reflection object")
    parsed = {}
    for field in ("applicability_conditions", "exclusions"):
        entries = reflection.get(field)
        if not isinstance(entries, list) or not entries or any(
            not isinstance(item, str) or not item.strip() for item in entries
        ):
            raise ValueError(f"correction reflection must contain non-empty {field}")
        parsed[field] = [" ".join(item.split()) for item in entries]
    parsed_facts = {"n": facts.get("n")}
    if type(parsed_facts["n"]) is not int or parsed_facts["n"] <= 0:
        raise ValueError("verified_facts.n must be a positive integer")
    for field in (
        "mae",
        "rmse",
        "mean_signed_error_prediction_minus_truth",
        "first_prediction_error",
        "first_actual_change_from_history",
        "first_prediction_change_from_history",
    ):
        if field in facts:
            parsed_facts[field] = _finite(facts[field], f"verified_facts.{field}")
    parsed_horizon = parsed_facts["n"]
    return {
        "correction_schema_version": 2,
        "verified_facts": parsed_facts,
        "horizon_claim_mismatches": find_horizon_claim_mismatches(
            parsed["applicability_conditions"], expected_horizon=parsed_horizon
        ),
        **parsed,
    }


def audit_report(report):
    """Build case-level diagnostics from an existing report, without reading truth."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    dataset = report.get("dataset")
    horizon = report.get("horizon")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("report dataset is missing")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("report horizon must be a positive integer")
    fit_ids = report.get("fit_case_ids")
    windows = report.get("validation_windows")
    if not isinstance(fit_ids, list) or not isinstance(windows, list) or not windows:
        raise ValueError("report must contain fit_case_ids and validation_windows")
    fit_id_set = set(fit_ids)
    if len(fit_id_set) != len(fit_ids):
        raise ValueError("fit_case_ids contains duplicates")

    rows = []
    memories = {}
    seen_windows = set()
    required_arms = ("full_reasoning", "no_retrieval", "low_similarity_control", "history_mean")
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("validation window must be an object")
        case_id = window.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_windows:
            raise ValueError("validation case IDs must be non-empty and unique")
        seen_windows.add(case_id)
        case = window.get("case")
        retrieval = window.get("retrieval")
        if not isinstance(case, dict) or not isinstance(retrieval, dict):
            raise ValueError("validation window is missing case or retrieval data")
        memory_id = retrieval.get("case_id")
        if memory_id not in fit_id_set:
            raise ValueError(f"retrieved memory {memory_id!r} is not a recorded fit case")
        correction = parse_verified_correction(retrieval.get("correction"))
        if memory_id in memories:
            if memories[memory_id]["correction"] != correction:
                raise ValueError("one retrieved memory has inconsistent correction records")
            memories[memory_id]["retrieval_count"] += 1
        else:
            memories[memory_id] = {"correction": correction, "retrieval_count": 1}

        history = profile_history(case.get("series"))
        score = _finite(retrieval.get("score"), "retrieval score")
        audit = window.get("retrieval_audit")
        negative = window.get("negative_control_retrieval")
        metrics = window.get("metrics")
        if not isinstance(audit, dict) or not isinstance(negative, dict) or not isinstance(metrics, dict):
            raise ValueError("validation window is missing retrieval or metric audit data")
        count = audit.get("eligible_candidate_count")
        if type(count) is not int or count < 0:
            raise ValueError("eligible_candidate_count must be a non-negative integer")
        for arm in required_arms:
            if not isinstance(metrics.get(arm), dict):
                raise ValueError(f"validation window is missing metrics for {arm}")
        arm_mae = {
            arm: _finite(metrics[arm].get("mae"), f"metrics.{arm}.mae")
            for arm in required_arms
        }
        arm_rmse = {
            arm: _finite(metrics[arm].get("rmse"), f"metrics.{arm}.rmse")
            for arm in required_arms
        }
        quality = case.get("news_quality_detail")
        if not isinstance(quality, dict):
            quality = {}
        low_score = negative.get("score")
        rows.append({
            "case_id": case_id,
            "development_role": window.get("development_role", "unknown"),
            "retrieved_memory_id": memory_id,
            "retrieved_memory_is_fit": True,
            "retrieval_score": score,
            "eligible_candidate_count": count,
            "negative_control_memory_id": negative.get("case_id"),
            "negative_control_score": (
                _finite(low_score, "negative-control score") if low_score is not None else None
            ),
            "history": history,
            "news_quality": case.get("news_quality", "unknown"),
            "news_source_verification": quality.get("source_verification", "unknown"),
            "news_target_relevance": quality.get("target_relevance", "unknown"),
            "news_fragment_count": quality.get("fragment_count"),
            "news_nonmention_caveat_count": quality.get("nonmention_caveat_count"),
            "mae": arm_mae,
            "rmse": arm_rmse,
            "delta_mae_full_minus_no_retrieval": arm_mae["full_reasoning"] - arm_mae["no_retrieval"],
            "delta_mae_full_minus_low_similarity": arm_mae["full_reasoning"] - arm_mae["low_similarity_control"],
            "delta_mae_full_minus_history_mean": arm_mae["full_reasoning"] - arm_mae["history_mean"],
            "delta_rmse_full_minus_no_retrieval": arm_rmse["full_reasoning"] - arm_rmse["no_retrieval"],
            "delta_rmse_full_minus_low_similarity": arm_rmse["full_reasoning"] - arm_rmse["low_similarity_control"],
            "delta_rmse_full_minus_history_mean": arm_rmse["full_reasoning"] - arm_rmse["history_mean"],
        })

    return {
        "dataset": dataset,
        "horizon": horizon,
        "real_world_target_semantics_verified": report.get("real_world_target_semantics_verified"),
        "target_contract": report.get("target_contract"),
        "windows": rows,
        "memories": memories,
    }


def _fmt(value):
    return "—" if value is None else f"{float(value):.4f}"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _role_win_counts(rows, delta_field):
    result = {}
    for role in sorted({row["development_role"] for row in rows}):
        subset = [row for row in rows if row["development_role"] == role]
        deltas = [row[delta_field] for row in subset]
        result[role] = {
            "windows": len(subset),
            "wins": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value > 0 for value in deltas),
        }
    return result


def render_markdown(audits):
    """Render retrospective evidence with explicit non-blind/tuning caveats."""
    if not isinstance(audits, list) or not audits:
        raise ValueError("audits must be a non-empty list")
    lines = [
        "# HIC H8/H10 retrieved-correction applicability audit (retrospective)",
        "",
        "> This is a read-only diagnostic over already-inspected development reports. Confirmation windows are retrospective and are not an untouched test set. No thresholds or routing rules are selected here. Target semantics remain unverified. The audit reads stored per-window metrics and pre-forecast history only; it does not inspect truth arrays, raw news text, outer-test data, or call an API/Qdrant.",
        "",
        "The reported correction text contains model-written applicability hypotheses and exclusions. They are shown as hypotheses, not verified causal claims. The HIC forecast prompt asks the model to consider whether a lesson applies, but the saved window schema does not contain a separate applicability accept/reject verdict. A new opt-in strict retrieval guard rejects only explicit lexical horizon contradictions against code-computed `verified_facts.n`; it does not alter the frozen v7/default retrieval path and does not prove semantic applicability.",
        "",
    ]
    all_rows = []
    for audit in audits:
        rows = audit["windows"]
        all_rows.extend(rows)
        lines.extend([
            f"## {audit['dataset']} / H{audit['horizon']}",
            "",
            f"- Windows: {len(rows)}; selection: {sum(r['development_role'] == 'selection' for r in rows)}; confirmation: {sum(r['development_role'] == 'confirmation' for r in rows)}.",
            f"- Target semantics verified: `{audit['real_world_target_semantics_verified']}`.",
            "- History sign strings are literal signs of consecutive observed changes (for example `+++--`); they are descriptive only and do not define a ‘strong trend’ threshold.",
            "",
        ])
        mismatched_memories = [
            item for item in audit["memories"].values()
            if item["correction"]["horizon_claim_mismatches"]
        ]
        mismatch_retrievals = sum(item["retrieval_count"] for item in mismatched_memories)
        lines.extend([
            f"- Strict horizon-text consistency scan: {len(mismatched_memories)}/{len(audit['memories'])} unique retrieved memories contain an explicit claim inconsistent with their verified horizon; these memories appeared in {mismatch_retrievals}/{len(rows)} retrieval windows. The scan is lexical and conservative, not semantic validation.",
            "",
            "### Per-window retrieval and outcomes",
            "",
            "| Role | Query case | Retrieved fit memory (cosine) | Candidate count | Low-sim memory (cosine) | 8-point history signs | Trailing declines | News quality / source / relevance | MAE: HIC / no-retrieval / low-sim / history-mean | ΔMAE HIC−no / HIC−low / HIC−mean |",
            "|---|---|---:|---:|---:|---|---:|---|---:|---:|",
        ])
        for row in rows:
            source = row["news_source_verification"]
            relevance = row["news_target_relevance"]
            quality = f"{row['news_quality']} / {source} / {relevance}"
            cosine = f"{row['retrieval_score']:.4f}"
            low = (
                f"{row['negative_control_memory_id']} ({row['negative_control_score']:.4f})"
                if row["negative_control_score"] is not None else "—"
            )
            maes = " / ".join(_fmt(row["mae"][arm]) for arm in (
                "full_reasoning", "no_retrieval", "low_similarity_control", "history_mean"
            ))
            deltas = " / ".join(_fmt(row[field]) for field in (
                "delta_mae_full_minus_no_retrieval",
                "delta_mae_full_minus_low_similarity",
                "delta_mae_full_minus_history_mean",
            ))
            lines.append(
                f"| {row['development_role']} | `{row['case_id']}` | `{row['retrieved_memory_id']}` ({cosine}) | {row['eligible_candidate_count']} | `{_escape(low)}` | `{row['history']['delta_signs']}` | {row['history']['trailing_decline_steps']} | `{_escape(quality)}` | {maes} | {deltas} |"
            )
        lines.append("")
        for label, field in (
            ("no-retrieval", "delta_mae_full_minus_no_retrieval"),
            ("low-similarity control", "delta_mae_full_minus_low_similarity"),
            ("history-mean", "delta_mae_full_minus_history_mean"),
        ):
            counts = _role_win_counts(rows, field)
            rendered = "; ".join(
                f"{role}: {counts[role]['wins']} wins / {counts[role]['ties']} ties / {counts[role]['losses']} losses (n={counts[role]['windows']})"
                for role in sorted(counts)
            )
            lines.append(f"- Descriptive HIC MAE comparisons vs {label}: {rendered}.")
        lines.append("")
        lines.extend(["### Retrieved memory hypotheses", ""])
        for memory_id, item in sorted(audit["memories"].items()):
            correction = item["correction"]
            facts = correction["verified_facts"]
            lines.append(
                f"#### `{memory_id}` (retrieved {item['retrieval_count']} time(s); correction schema v{correction['correction_schema_version']})"
            )
            lines.append("")
            lines.append(
                "Code-computed historical facts: "
                f"horizon n={facts['n']}; MAE={_fmt(facts.get('mae'))}; "
                f"RMSE={_fmt(facts.get('rmse'))}; "
                f"mean signed error (prediction−truth)={_fmt(facts.get('mean_signed_error_prediction_minus_truth'))}; "
                f"first prediction error={_fmt(facts.get('first_prediction_error'))}."
            )
            mismatches = correction["horizon_claim_mismatches"]
            if mismatches:
                claims = "; ".join(
                    f"condition {item['condition_index'] + 1} says horizon {item['claimed_horizon']} "
                    f"but verified facts say {item['expected_horizon']}"
                    for item in mismatches
                )
                lines.append(f"- Potential explicit horizon contradiction (lexical check, not exhaustive): {claims}.")
            else:
                lines.append("- Potential explicit horizon contradiction: none found by the narrow lexical scan (not proof of semantic consistency).")
            lines.append("Model-written applicability hypotheses (not machine-verified against the query):")
            lines.extend(f"- {condition}" for condition in correction["applicability_conditions"])
            lines.append("Model-written exclusions:")
            lines.extend(f"- {exclusion}" for exclusion in correction["exclusions"])
            lines.append("")
    lines.extend([
        "## What this audit establishes",
        "",
        "- It binds each top-1 retrieved correction to a recorded fit case and exposes the exact history-shape, text-quality metadata, candidate count, similarity scores, and per-window errors needed to review its stated conditions.",
        "- It does not establish that an LLM-written condition is true for the query, that a particular correction caused a forecast change, or that low-similarity controls are random/irrelevant. A negative delta means lower HIC error for that already-scored window; it is not an independent estimate.",
        "- A next valid gate must turn only prediction-time-verifiable conditions into explicit, logged checks (or abstain). Conditions referring to a query's unknown future error cannot be checked before prediction and must not be treated as query applicability evidence.",
        "- Do not tune thresholds from these confirmation rows. For an independent final claim, first close source/target/event-known-at provenance and obtain a genuinely unexposed evaluation interval after freezing the rule and baseline.",
        "",
    ])
    return "\n".join(lines)
