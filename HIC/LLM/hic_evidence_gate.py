"""Fail-closed checks for forecast-time task and news evidence provenance.

These checks validate that an auditable evidence record is structurally complete;
they do not independently verify that a publisher or article is truthful.
"""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlparse


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TASK_CONTRACT_FIELDS = (
    "dataset",
    "target",
    "target_id",
    "definition",
    "unit",
    "frequency",
    "horizon",
    "entity_scope",
)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _parse_cutoff(value):
    if not _nonempty(value):
        raise ValueError("missing_forecast_cutoff")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid_forecast_cutoff") from None
    if parsed.tzinfo is None:
        raise ValueError("forecast_cutoff_missing_timezone")
    return parsed.astimezone(timezone.utc)


def _parse_evidence_time(value):
    if not _nonempty(value):
        raise ValueError("missing_evidence_time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid_evidence_time") from None
    if parsed.tzinfo is None:
        raise ValueError("evidence_time_missing_timezone")
    return parsed.astimezone(timezone.utc)


def _validate_task_contract(contract):
    reasons = []
    if not isinstance(contract, dict):
        return ["missing_task_contract"]
    if contract.get("status") != "verified":
        reasons.append("target_contract_not_verified")
    for field in _TASK_CONTRACT_FIELDS:
        value = contract.get(field)
        if field == "horizon":
            if type(value) is not int or value <= 0:
                reasons.append("invalid_task_contract_horizon")
        elif not _nonempty(value):
            reasons.append(f"missing_task_contract_{field}")
    if not _nonempty(contract.get("verification_basis")):
        reasons.append("missing_task_contract_verification_basis")
    return reasons


def _validate_news_item(item, cutoff):
    reasons = []
    if not isinstance(item, dict):
        return ["invalid_news_evidence_record"]

    for field in ("title", "publisher", "verification_note", "relevance_basis", "text"):
        if not _nonempty(item.get(field)):
            reasons.append(f"missing_{field}")

    source_url = item.get("source_url")
    parsed_url = urlparse(source_url) if _nonempty(source_url) else None
    if (
        parsed_url is None
        or parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        reasons.append("missing_source_url" if not _nonempty(source_url) else "invalid_source_url")
    if item.get("source_verification") != "independently_verified":
        reasons.append("source_not_independently_verified")
    if item.get("target_relevance") != "relevant":
        reasons.append("news_relevance_not_verified")

    published_at = known_at = None
    for field in ("published_at", "known_at"):
        try:
            parsed = _parse_evidence_time(item.get(field))
        except ValueError as error:
            reasons.append(f"{field}_{error}")
            continue
        if field == "published_at":
            published_at = parsed
        else:
            known_at = parsed
    if cutoff is not None and published_at is not None and published_at > cutoff:
        reasons.append("evidence_published_after_cutoff")
    if cutoff is not None and known_at is not None and known_at > cutoff:
        reasons.append("evidence_known_after_cutoff")
    if published_at is not None and known_at is not None and known_at < published_at:
        reasons.append("evidence_known_before_publication")

    text = item.get("text")
    text_sha256 = item.get("text_sha256")
    if not isinstance(text_sha256, str) or not _SHA256_RE.fullmatch(text_sha256):
        reasons.append("invalid_news_text_sha256")
    elif _nonempty(text) and hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha256:
        reasons.append("news_text_hash_mismatch")
    return reasons


def case_evidence_eligibility(case):
    """Return (eligible, reasons) for one case's target/news evidence contract.

    News text must be exactly the concatenation of the independently documented
    snippets so unverified text cannot silently remain in the model prompt.
    """
    if not isinstance(case, dict):
        return False, ["invalid_case"]
    reasons = []
    cutoff = None
    try:
        cutoff = _parse_cutoff(case.get("forecast_cutoff_at"))
    except ValueError as error:
        reasons.append(str(error))

    reasons.extend(_validate_task_contract(case.get("task_contract")))
    contract = case.get("task_contract")
    if isinstance(contract, dict):
        for field in ("dataset", "target", "unit", "frequency", "horizon"):
            if field in case and contract.get(field) != case.get(field):
                reasons.append(f"task_contract_{field}_mismatch")
    evidence = case.get("news_evidence")
    if not isinstance(evidence, list) or not evidence:
        reasons.append("missing_news_evidence")
        return False, reasons
    for index, item in enumerate(evidence):
        reasons.extend(
            f"news_{index}:{reason}" for reason in _validate_news_item(item, cutoff)
        )
    expected_news = "\n".join(
        item.get("text", "") for item in evidence if isinstance(item, dict)
    )
    news = case.get("news")
    if isinstance(news, str) and news.strip():
        if news != expected_news:
            reasons.append("news_evidence_text_mismatch")
    else:
        expected_hash = case.get("news_text_sha256")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            reasons.append("missing_news_text")
        elif hashlib.sha256(expected_news.encode("utf-8")).hexdigest() != expected_hash:
            reasons.append("news_evidence_text_hash_mismatch")
    return not reasons, reasons


def paired_case_evidence_eligibility(query, memory):
    """Require complete evidence on both sides and an identical task contract."""
    query_ok, query_reasons = case_evidence_eligibility(query)
    memory_ok, memory_reasons = case_evidence_eligibility(memory)
    reasons = [f"query_evidence:{reason}" for reason in query_reasons]
    reasons.extend(f"memory_evidence:{reason}" for reason in memory_reasons)
    if query_ok and memory_ok:
        query_contract = query["task_contract"]
        memory_contract = memory["task_contract"]
        for field in _TASK_CONTRACT_FIELDS:
            if query_contract[field] != memory_contract[field]:
                reasons.append(f"task_contract_{field}_mismatch")
    return not reasons, reasons


def _task_contract_sha256(contract):
    encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_baseline(prediction, horizon):
    if not isinstance(prediction, (list, tuple)) or not prediction:
        raise ValueError("numeric baseline must be a non-empty forecast sequence")
    if horizon is not None and len(prediction) != horizon:
        raise ValueError("numeric baseline must contain exactly one value per forecast step")
    if any(type(value) is bool for value in prediction):
        raise ValueError("numeric baseline must contain finite numeric values")
    try:
        values = [float(value) for value in prediction]
    except (TypeError, ValueError):
        raise ValueError("numeric baseline must contain finite numeric values") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("numeric baseline must contain finite numeric values")
    return values


def _applicability_decision_reasons(decision, query, candidate):
    if not isinstance(decision, dict):
        return ["applicability_not_verified"]
    if decision.get("status") != "applicable":
        return ["correction_not_applicable"]
    if (
        decision.get("query_case_id") != query.get("case_id")
        or decision.get("memory_case_id") != candidate.get("case_id")
        or decision.get("task_contract_sha256")
        != _task_contract_sha256(query["task_contract"])
    ):
        return ["applicability_decision_binding_mismatch"]
    if not _nonempty(decision.get("rule_id")):
        return ["applicability_rule_missing"]
    checks = decision.get("checks")
    if not isinstance(checks, list) or not checks:
        return ["applicability_checks_missing"]
    reasons = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or not _nonempty(check.get("name")):
            reasons.append(f"applicability_check_{index}_invalid")
        elif check.get("passed") is not True:
            reasons.append(f"applicability_check_{check['name']}_failed")
        elif not _nonempty(check.get("evidence_ref")):
            reasons.append(f"applicability_check_{check['name']}_evidence_missing")
    return reasons


def choose_forecast_route(
    query,
    ranked_candidates,
    numeric_baseline,
    *,
    applicability_decisions=None,
):
    """Choose a verified correction memory or fail closed to a numeric baseline.

    ``ranked_candidates`` should be the full result object from
    ``rank_applicable_memories(..., require_verified_evidence=True)`` so strict
    retrieval exclusions remain visible in fallback reasons. A legacy list of
    ranked candidates is still accepted. Applicability decisions are supplied
    by a separately frozen, auditable rule; this function validates their
    binding and evidence shape but does not invent or infer applicability. The
    caller must supply the numeric baseline already selected and frozen for the
    task contract; this helper validates forecast shape, not baseline provenance.
    """
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    contract = query.get("task_contract")
    horizon = contract.get("horizon") if isinstance(contract, dict) else None
    if type(horizon) is not int or horizon <= 0:
        horizon = None
    baseline = _validated_baseline(numeric_baseline, horizon)
    retrieval_exclusions = {}
    if isinstance(ranked_candidates, dict):
        retrieval_exclusions = ranked_candidates.get("excluded_counts", {})
        if not isinstance(retrieval_exclusions, dict) or any(
            not isinstance(reason, str)
            or type(count) is not int
            or count < 0
            for reason, count in retrieval_exclusions.items()
        ):
            raise ValueError("ranked candidate exclusions must be reason/count pairs")
        ranked_candidates = ranked_candidates.get("eligible")
    if not isinstance(ranked_candidates, list):
        raise ValueError("ranked_candidates must be a list or strict ranking result")
    if applicability_decisions is None:
        applicability_decisions = {}
    if not isinstance(applicability_decisions, dict):
        raise ValueError("applicability_decisions must be a mapping by memory case ID")

    query_ok, query_reasons = case_evidence_eligibility(query)
    if not query_ok:
        return {
            "route": "numeric_baseline",
            "prediction": baseline,
            "memory_case_id": None,
            "fallback_reasons": list(dict.fromkeys(query_reasons)),
        }

    rejected = []
    for candidate in ranked_candidates:
        if not isinstance(candidate, dict):
            rejected.append("invalid_memory_record")
            continue
        case_id = candidate.get("case_id")
        metadata = candidate.get("metadata")
        if not _nonempty(case_id) or not isinstance(metadata, dict):
            rejected.append("invalid_memory_metadata")
            continue
        if case_id == query.get("case_id"):
            rejected.append("self_match_excluded")
            continue
        if metadata.get("case_id") != case_id or metadata.get("split") != "fit":
            rejected.append("memory_not_bound_to_fit_case")
            continue
        if metadata.get("completed") is not True:
            rejected.append("memory_not_completed")
            continue
        if (
            not isinstance(candidate.get("correction"), str)
            or not candidate["correction"].strip()
            or not isinstance(candidate.get("summary"), (dict, list, str))
        ):
            rejected.append("memory_content_incomplete")
            continue
        score = candidate.get("score")
        if type(score) not in (int, float) or not math.isfinite(float(score)):
            rejected.append("invalid_similarity")
            continue
        evidence_ok, evidence_reasons = paired_case_evidence_eligibility(query, metadata)
        if not evidence_ok:
            rejected.extend(evidence_reasons)
            continue
        decision_reasons = _applicability_decision_reasons(
            applicability_decisions.get(case_id), query, candidate
        )
        if decision_reasons:
            rejected.extend(decision_reasons)
            continue
        return {
            "route": "hic_correction",
            "prediction": None,
            "memory_case_id": case_id,
            "candidate": candidate,
            "fallback_reasons": [],
        }

    if not rejected and retrieval_exclusions:
        rejected.extend(sorted(retrieval_exclusions))
    if not rejected:
        rejected.append("no_eligible_memory")
    return {
        "route": "numeric_baseline",
        "prediction": baseline,
        "memory_case_id": None,
        "fallback_reasons": list(dict.fromkeys(rejected)),
    }


def execute_verified_forecast_route(
    query,
    ranked_candidates,
    numeric_baseline,
    forecast_from_memory,
    *,
    applicability_decisions=None,
):
    """Run HIC generation only after strict evidence and applicability pass.

    The caller supplies a frozen numeric baseline and a forecast callback that
    consumes the selected query/memory pair. When the route selector rejects
    every candidate, the callback is not invoked and the baseline is returned.
    Provider errors are deliberately propagated rather than silently hidden.
    """
    if not callable(forecast_from_memory):
        raise ValueError("forecast_from_memory must be callable")

    route = choose_forecast_route(
        query,
        ranked_candidates,
        numeric_baseline,
        applicability_decisions=applicability_decisions,
    )
    if route["route"] == "numeric_baseline":
        return route

    horizon = query["task_contract"]["horizon"]
    prediction = forecast_from_memory(query, route["candidate"])
    route["prediction"] = _validated_baseline(prediction, horizon)
    return route


def execute_verified_forecast_route_with_retriever(
    query,
    retrieve_ranked_candidates,
    numeric_baseline,
    forecast_from_memory,
    *,
    applicability_decisions=None,
):
    """Run the query-side evidence gate before invoking a retrieval callback.

    For an ineligible query, return the validated numeric baseline without
    invoking retrieval or HIC generation. For an eligible query, the callback
    must return the strict ranked-candidate result expected by
    ``execute_verified_forecast_route``. This helper does not connect to a
    vector store or establish that a caller has wired it into an application.
    """
    if not callable(retrieve_ranked_candidates):
        raise ValueError("retrieve_ranked_candidates must be callable")
    if not callable(forecast_from_memory):
        raise ValueError("forecast_from_memory must be callable")

    query_eligible, _ = case_evidence_eligibility(query)
    if not query_eligible:
        ranked_candidates = {"eligible": [], "excluded_counts": {}}
    else:
        ranked_candidates = retrieve_ranked_candidates(query)

    return execute_verified_forecast_route(
        query,
        ranked_candidates,
        numeric_baseline,
        forecast_from_memory,
        applicability_decisions=applicability_decisions,
    )
