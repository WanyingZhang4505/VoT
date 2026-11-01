"""Offline structural and numeric checks for saved Phase20 fit memories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from LLM.hic_verified_memory import canonical_json, compute_error_facts


_HORIZON_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "eighteen": 18,
    "twenty": 20,
    "twenty-four": 24,
}
_DURATION_CLAIM = re.compile(
    r"\b(?:next|future|upcoming|following)\s+"
    r"(?P<count>\d+|" + "|".join(sorted(_HORIZON_WORDS, key=len, reverse=True)) + r")"
    r"\s+(?:monthly\s+)?(?:time\s+)?"
    r"(?:steps?|months?|weeks?|days?|periods?|target\s+values?)\b",
    re.IGNORECASE,
)
_V2_REQUIRED_FACTS = ("mse", "history_length", "prediction_length", "truth_length")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def _parse_count(value: str) -> int | None:
    normalized = value.casefold()
    if normalized.isdigit():
        return int(normalized)
    return _HORIZON_WORDS.get(normalized)


def _summary_horizon_claims(summary: Any, expected_horizon: int) -> list[int]:
    claims = set()
    for text in _walk_strings(summary):
        for match in _DURATION_CLAIM.finditer(text):
            count = _parse_count(match.group("count"))
            if count is not None and count != expected_horizon:
                claims.add(count)
    return sorted(claims)


def _facts_match(stored: Any, computed: dict[str, Any]) -> bool | None:
    if not isinstance(stored, dict) or not stored:
        return None
    for key, stored_value in stored.items():
        if key == "schema_version":
            continue
        if key not in computed:
            return False
        computed_value = computed[key]
        if isinstance(stored_value, bool) or isinstance(computed_value, bool):
            if type(stored_value) is not type(computed_value) or stored_value != computed_value:
                return False
        elif isinstance(stored_value, (int, float)) and isinstance(computed_value, (int, float)):
            if not math.isfinite(float(stored_value)) or not math.isclose(
                float(stored_value), float(computed_value), rel_tol=1e-9, abs_tol=1e-9
            ):
                return False
        elif stored_value != computed_value:
            return False
    return True


def _decode_correction(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def audit_fit_state(source_case: dict[str, Any], state: dict[str, Any], *, horizon: int) -> dict[str, Any]:
    """Compare one saved memory state with its source fit window, without providers."""
    if not isinstance(source_case, dict) or not isinstance(state, dict):
        raise ValueError("source_case and state must be dictionaries")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    forecast = state.get("forecast")
    prediction = forecast.get("prediction") if isinstance(forecast, dict) else None
    history = source_case.get("series")
    truth = source_case.get("truth")
    prediction_length_matches = isinstance(prediction, list) and len(prediction) == horizon
    history_length_matches = isinstance(history, list) and len(history) > 0
    source_truth_length_matches = isinstance(truth, list) and len(truth) == horizon

    computed_facts = None
    if history_length_matches and prediction_length_matches and source_truth_length_matches:
        try:
            computed_facts = compute_error_facts(history, prediction, truth)
        except (TypeError, ValueError):
            computed_facts = None

    correction = _decode_correction(state.get("correction"))
    correction_facts = correction.get("verified_facts") if correction else None
    stored_facts = state.get("verified_facts")
    if not isinstance(stored_facts, dict):
        stored_facts = correction_facts
    facts_match = _facts_match(stored_facts, computed_facts) if computed_facts else None
    correction_facts_match = (
        _facts_match(correction_facts, computed_facts) if computed_facts else None
    )

    summary = state.get("summary")
    identity = state.get("summary_identity")
    identity = identity if isinstance(identity, dict) else {}
    content_hash = identity.get("content_sha256") or identity.get("summary_content_sha256")
    summary_hash_matches = None
    if isinstance(content_hash, str) and _SHA256_RE.fullmatch(content_hash):
        try:
            summary_hash_matches = hashlib.sha256(canonical_json(summary).encode("utf-8")).hexdigest() == content_hash
        except (TypeError, ValueError):
            summary_hash_matches = False
    request_hash = identity.get("request_sha256") or identity.get("summary_request_sha256")
    request_hash_present = isinstance(request_hash, str) and bool(_SHA256_RE.fullmatch(request_hash))

    reflection = correction.get("reflection") if correction else None
    reflection = reflection if isinstance(reflection, dict) else {}
    reflection_complete = all(
        isinstance(reflection.get(field), str) and reflection[field].strip()
        for field in ("improved_reasoning_hypothesis", "key_insights_hypothesis")
    ) and all(
        isinstance(reflection.get(field), list)
        and all(isinstance(item, str) and item.strip() for item in reflection[field])
        for field in ("applicability_conditions", "exclusions")
    )

    mismatch_claims = _summary_horizon_claims(summary, horizon)
    facts_schema = stored_facts.get("schema_version") if isinstance(stored_facts, dict) else None
    missing_v2 = [field for field in _V2_REQUIRED_FACTS if not isinstance(stored_facts, dict) or field not in stored_facts]
    embedding = state.get("embedding")
    embedding_dimension = len(embedding) if isinstance(embedding, list) else None

    return {
        "case_id": source_case.get("case_id"),
        "prediction_length_matches_horizon": prediction_length_matches,
        "history_length_matches_source": history_length_matches,
        "source_truth_length_matches_horizon": source_truth_length_matches,
        "computed_facts_match_stored": facts_match,
        "correction_facts_match_computed": correction_facts_match,
        "stored_facts_schema_version": facts_schema,
        "missing_v2_fact_fields": missing_v2,
        "summary_content_hash_matches": summary_hash_matches,
        "summary_request_hash_present": request_hash_present,
        "summary_horizon_claim_mismatches": mismatch_claims,
        "correction_json_valid": correction is not None,
        "correction_schema_version": correction.get("schema_version") if correction else None,
        "reflection_complete": reflection_complete,
        "ingested": state.get("ingested") is True,
        "embedding_dimension": embedding_dimension,
    }
