"""Deterministic offline integrity checks for saved HIC vNext fit memories."""

from __future__ import annotations

import json
import math
import re

from LLM import hic
from LLM.hic_correction_consistency import find_horizon_claim_mismatches
from LLM.hic_grid_runner import _hash_json
from LLM.hic_phase20_memory_audit import _summary_horizon_claims
from LLM.hic_verified_memory import (
    CORRECTION_REFLECTION_SCHEMA_VERSION_V3,
    CORRECTION_SCHEMA_VERSION,
    CORRECTION_SCHEMA_VERSION_V3,
    compute_error_facts,
    numeric_free_reflection_is_valid,
)
from LLM.hic_vnext_narrative_numeric_audit import audit_reflection_numeric_claims
from LLM.hic_vnext import apply_bounded_residual
from LLM.hic_vnext_calibration import _score_prediction, _variant_key


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_K_VALUES = (1, 3)
_ALPHAS = (0.25, 0.5, 1.0)
_BOUNDS = (0.5, 1.0)


def _decode_correction(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _finite_vector(value, expected_length=None):
    if not isinstance(value, list) or not value:
        return False
    if expected_length is not None and len(value) != expected_length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _metrics_match(saved, prediction, truth):
    if not isinstance(saved, dict):
        return False
    computed = _score_prediction(prediction, truth)
    return all(
        isinstance(saved.get(metric), (int, float))
        and math.isfinite(float(saved[metric]))
        and math.isclose(float(saved[metric]), value, rel_tol=1e-9, abs_tol=1e-9)
        for metric, value in computed.items()
    )


def audit_vnext_memory_state(
    source_case,
    state,
    *,
    dataset,
    horizon,
    target,
    unit,
    source_sha256,
    eligible_memory_ids,
    embedding_dimension=1024,
    expected_correction_schema_version=CORRECTION_SCHEMA_VERSION,
    expected_correction_prompt_version=None,
):
    """Audit one completed checkpoint record against its original fit case.

    The caller must pass only a fit case and the strictly-earlier case IDs from
    the prequential fit plan. This function does not load or score holdout data.
    """
    if not isinstance(source_case, dict) or not isinstance(state, dict):
        raise ValueError("source_case and state must be mappings")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if type(expected_correction_schema_version) is not int or expected_correction_schema_version not in {
        CORRECTION_SCHEMA_VERSION, CORRECTION_SCHEMA_VERSION_V3
    }:
        raise ValueError("expected_correction_schema_version must be a supported version")
    if expected_correction_schema_version == CORRECTION_SCHEMA_VERSION_V3:
        if not isinstance(expected_correction_prompt_version, str) or not expected_correction_prompt_version.strip():
            raise ValueError("expected_correction_prompt_version is required for correction schema v3")
    elif expected_correction_prompt_version is not None:
        raise ValueError("expected_correction_prompt_version is only supported for correction schema v3")
    if not isinstance(eligible_memory_ids, (set, frozenset)):
        raise ValueError("eligible_memory_ids must be a set of strictly earlier fit IDs")

    issues = []

    def check(ok, code):
        if not ok:
            issues.append(code)
        return bool(ok)

    complete = state.get("status") == "complete"
    check(complete, "checkpoint_not_complete")
    if not complete:
        return {
            "case_id": source_case.get("case_id"),
            "is_complete_memory": False,
            "issue_codes": issues,
        }

    memory = state.get("memory")
    forecast = state.get("forecast")
    scores = state.get("scores")
    if not isinstance(memory, dict) or not isinstance(forecast, dict) or not isinstance(scores, dict):
        check(False, "required_checkpoint_sections_missing")
        return {
            "case_id": source_case.get("case_id"),
            "is_complete_memory": True,
            "issue_codes": issues,
        }

    metadata = memory.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    case_id = source_case.get("case_id")
    check(memory.get("case_id") == case_id, "memory_case_id_mismatch")
    check(metadata.get("case_id") == case_id, "metadata_case_id_mismatch")
    check(metadata.get("dataset") == dataset == source_case.get("dataset"), "dataset_mismatch")
    check(metadata.get("source_sha256") == source_sha256, "source_hash_mismatch")
    check(metadata.get("split") == "fit" and metadata.get("completed") is True, "not_completed_fit_memory")
    check(metadata.get("frequency") == source_case.get("frequency"), "frequency_mismatch")
    check(metadata.get("horizon") == horizon, "horizon_mismatch")
    check(metadata.get("target") == target and metadata.get("unit") == unit, "target_or_unit_mismatch")
    check(metadata.get("target_end") == source_case.get("target_end"), "target_end_mismatch")
    check(metadata.get("news_quality") == "valid" == source_case.get("news_quality"), "news_quality_mismatch")
    check(
        metadata.get("correction_schema_version") == expected_correction_schema_version,
        "correction_schema_version_mismatch",
    )
    if expected_correction_schema_version == CORRECTION_SCHEMA_VERSION_V3:
        check(
            metadata.get("correction_prompt_version") == expected_correction_prompt_version,
            "correction_prompt_version_mismatch",
        )
    check(metadata.get("summary_prompt_version") == hic.SUMMARY_PROMPT_VERSION, "summary_prompt_version_mismatch")
    check(metadata.get("summary_identity_status") == "exact_request_response", "summary_identity_not_exact")
    check(metadata.get("history_series") == source_case.get("series"), "history_series_mismatch")

    summary = forecast.get("summary")
    summary_hash = _hash_json(summary) if isinstance(summary, (dict, list)) else None
    check(memory.get("summary") == summary, "memory_summary_mismatch")
    check(summary_hash is not None and forecast.get("summary_content_sha256") == summary_hash,
          "forecast_summary_content_hash_mismatch")
    check(metadata.get("summary_content_sha256") == summary_hash, "memory_summary_content_hash_mismatch")
    summary_request = forecast.get("summary_request_sha256")
    check(isinstance(summary_request, str) and _SHA256_RE.fullmatch(summary_request) is not None,
          "summary_request_hash_invalid")
    check(metadata.get("summary_request_sha256") == summary_request, "summary_request_identity_mismatch")

    truth = source_case.get("truth")
    baseline = forecast.get("baseline")
    baseline = baseline if isinstance(baseline, dict) else {}
    prediction = baseline.get("prediction")
    pred_ok = _finite_vector(prediction, horizon)
    check(pred_ok, "baseline_prediction_shape_or_value_invalid")
    if _finite_vector(truth, horizon):
        check(state.get("fit_truth_sha256") == _hash_json(truth), "fit_truth_hash_mismatch")
    else:
        check(False, "fit_truth_shape_or_value_invalid")

    correction = _decode_correction(memory.get("correction"))
    check(correction is not None, "correction_json_invalid")
    facts = correction.get("verified_facts") if correction else None
    reflection = correction.get("reflection") if correction else None
    check(
        isinstance(correction, dict)
        and correction.get("schema_version") == expected_correction_schema_version,
        "correction_schema_invalid",
    )
    if expected_correction_schema_version == CORRECTION_SCHEMA_VERSION_V3:
        check(
            isinstance(correction, dict)
            and correction.get("reflection_schema_version") == CORRECTION_REFLECTION_SCHEMA_VERSION_V3,
            "reflection_schema_invalid",
        )
        check(
            isinstance(reflection, dict) and numeric_free_reflection_is_valid(reflection),
            "reflection_numeric_free_check_failed",
        )
    check(isinstance(reflection, dict) and all(
        isinstance(reflection.get(key), str) and reflection[key].strip()
        for key in ("improved_reasoning_hypothesis", "key_insights_hypothesis")
    ) and all(
        isinstance(reflection.get(key), list)
        and all(isinstance(item, str) and item.strip() for item in reflection[key])
        for key in ("applicability_conditions", "exclusions")
    ), "reflection_fields_incomplete")
    check(state.get("correction_request_sha256") == metadata.get("correction_request_sha256")
          and isinstance(state.get("correction_request_sha256"), str)
          and _SHA256_RE.fullmatch(state.get("correction_request_sha256", "")) is not None,
          "correction_request_identity_mismatch")

    facts_match = False
    if pred_ok and _finite_vector(truth, horizon) and _finite_vector(source_case.get("series")):
        computed_facts = compute_error_facts(source_case["series"], prediction, truth)
        facts_match = facts == computed_facts
    check(facts_match, "verified_facts_mismatch")

    baseline_metrics_match = pred_ok and _finite_vector(truth, horizon) and _metrics_match(
        scores.get("baseline"), prediction, truth
    )
    check(baseline_metrics_match, "baseline_metrics_mismatch")

    trace = forecast.get("retrieval_trace")
    trace = trace if isinstance(trace, list) else None
    trace_ids = [item.get("case_id") for item in trace] if trace is not None and all(
        isinstance(item, dict) for item in trace
    ) else []
    trace_valid = trace is not None and len(trace_ids) == len(set(trace_ids))
    trace_valid = trace_valid and all(isinstance(case_id, str) and case_id in eligible_memory_ids
                                     for case_id in trace_ids)
    check(trace_valid, "retrieval_contains_ineligible_case")

    residuals = forecast.get("residuals_by_k")
    residuals = residuals if isinstance(residuals, dict) else {}
    variants = scores.get("variants")
    variants = variants if isinstance(variants, dict) else {}
    expected_variant_keys = {
        _variant_key(k, alpha, bound)
        for k in _K_VALUES for alpha in _ALPHAS for bound in _BOUNDS
    }
    variants_match = set(variants) == expected_variant_keys
    for k in _K_VALUES:
        response = residuals.get(str(k))
        response = response if isinstance(response, dict) else {}
        residual = response.get("residual")
        selected_ids = response.get("selected_case_ids")
        selected_ids = selected_ids if isinstance(selected_ids, list) else []
        check(_finite_vector(residual, horizon), f"residual_k{k}_shape_or_value_invalid")
        if trace is not None:
            check(selected_ids == trace_ids[:k], f"residual_k{k}_selection_mismatch")
        else:
            check(False, f"residual_k{k}_selection_unverifiable")
        if response.get("request_sha256") is not None:
            check(isinstance(response.get("request_sha256"), str)
                  and _SHA256_RE.fullmatch(response["request_sha256"]) is not None,
                  f"residual_k{k}_request_hash_invalid")
        for alpha in _ALPHAS:
            for bound in _BOUNDS:
                key = _variant_key(k, alpha, bound)
                entry = variants.get(key)
                if not isinstance(entry, dict) or not pred_ok:
                    variants_match = False
                    continue
                try:
                    expected = apply_bounded_residual(
                        prediction,
                        residual,
                        source_case["series"],
                        alpha=alpha,
                        bound_scale=bound,
                        eligible=bool(selected_ids),
                        abstain=bool(response.get("abstain")),
                    )
                except (TypeError, ValueError):
                    variants_match = False
                    continue
                if (
                    entry.get("prediction") != expected["prediction"]
                    or entry.get("applied_delta") != expected["applied_delta"]
                    or entry.get("fallback_reason") != expected["fallback_reason"]
                    or entry.get("selected_case_ids") != selected_ids
                    or not _metrics_match(entry, expected["prediction"], truth)
                ):
                    variants_match = False
    check(variants_match, "bounded_variant_scores_or_predictions_mismatch")

    embedding = memory.get("embedding")
    embedding_ok = _finite_vector(embedding, embedding_dimension)
    check(embedding_ok, "embedding_dimension_or_values_invalid")
    check(state.get("ingested") is True, "checkpoint_ingested_flag_false")

    summary_claims = _summary_horizon_claims(summary, horizon) if summary is not None else []
    correction_claims = (
        find_horizon_claim_mismatches(
            reflection.get("applicability_conditions", []), expected_horizon=horizon
        ) if isinstance(reflection, dict) else []
    )
    narrative_numeric_claim_audit = audit_reflection_numeric_claims(
        reflection,
        facts,
        history=source_case.get("series"),
        prediction=prediction,
        truth=truth,
    )
    return {
        "case_id": case_id,
        "is_complete_memory": True,
        "verified_facts_match_recomputed": facts_match,
        "score_metrics_match_recomputed": bool(baseline_metrics_match and variants_match),
        "retrieval_ids_unique_and_strictly_eligible": bool(trace_valid),
        "embedding_dimension": len(embedding) if isinstance(embedding, list) else None,
        "checkpoint_ingested_flag": state.get("ingested") is True,
        "summary_horizon_claim_mismatches": summary_claims,
        "correction_horizon_claim_mismatches": correction_claims,
        "narrative_numeric_claim_audit": narrative_numeric_claim_audit,
        "reflection_contains_digit_text": bool(
            isinstance(reflection, dict)
            and any(any(char.isdigit() for char in text)
                    for key in ("improved_reasoning_hypothesis", "key_insights_hypothesis",
                                "applicability_conditions", "exclusions")
                    for text in ([reflection.get(key)] if isinstance(reflection.get(key), str)
                                 else reflection.get(key, [])))
        ),
        "issue_codes": sorted(set(issues)),
    }
