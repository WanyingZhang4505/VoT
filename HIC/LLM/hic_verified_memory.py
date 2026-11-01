"""Versioned HIC correction records with separately computed error evidence."""

import json
import hashlib
import math
import re
from datetime import date, datetime

from LLM.hic_evidence_gate import case_evidence_eligibility, paired_case_evidence_eligibility
from LLM.hic_correction_consistency import find_horizon_claim_mismatches


FACTS_SCHEMA_VERSION = 2
CORRECTION_SCHEMA_VERSION = 2
CORRECTION_SCHEMA_VERSION_V3 = 3
CORRECTION_REFLECTION_SCHEMA_VERSION_V3 = 1
CORRECTION_PROMPT_VERSION_V3 = "hic_verified_correction_v3_task_bound_numeric_free"
CORRECTION_SCHEMA_VERSION_V4 = 4
CORRECTION_REFLECTION_SCHEMA_VERSION_V4 = 1
CORRECTION_PROMPT_VERSION_V4 = "hic_verified_correction_v4_transfer_conditions"
CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND = 5
CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND = 1
TEST6_ONLINE_MEMORY_PROTOCOLS = {
    "agent_test6_cross_horizon_v1",
    "agent_test6_sequential_online_v1",
    "agent_test6_semantic_cross_case_v1",
}
TEST6_SEMANTIC_TRANSFER_PROTOCOL = "agent_test6_semantic_cross_case_v1"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_NUMERIC_DIGIT_RE = re.compile(r"[0-9]")
_ERROR_DIRECTION_RE = re.compile(
    r"(?i)\b(over[- ]?estim\w*|under[- ]?estim\w*|"
    r"over[- ]?predict\w*|under[- ]?predict\w*|"
    r"error direction|error pattern|sign reversal|signed error|bias|"
    r"overshoot|undershoot)\b"
)
_HISTORICAL_OUTCOME_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:forecast|prediction|predicted|forecasted)\b[^.!?]{0,120}"
    r"\b(?:below|above|higher|lower|greater|less|closer|farther|under|over)\b"
    r"[^.!?]{0,100}\b(?:realized|actual|observed|true)\b"
    r"|\b(?:realized|actual|observed|true)\b[^.!?]{0,120}"
    r"\b(?:forecast|prediction|predicted|forecasted)\b[^.!?]{0,100}"
    r"\b(?:below|above|higher|lower|greater|less|closer|farther|under|over)\b"
    r"|\b(?:absolute\s+)?errors?\b[^.!?]{0,80}"
    r"\b(?:size|magnitude|amount|var(?:y|ied|ies)|change[ds]?|increase[ds]?|"
    r"decrease[ds]?|widen(?:ed|ing)?|narrow(?:ed|ing)?)\b"
    r"|\b(?:realized|actual)\s+(?:path|outcome)\b[^.!?]{0,100}"
    r"\b(?:later|eventually|then)\b[^.!?]{0,60}"
    r"\b(?:turn\w*|rose|fell|declin\w*|increas\w*|drop\w*|rebound\w*)\b"
    r")"
)
TEST6_TRANSFER_PROJECTION_VERSION = "agent_test6_outcome_blind_semantic_transfer_projection_v7"


def canonical_json(value):
    """Serialize JSON deterministically so key order cannot change request identity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def make_test6_outcome_blind_transfer_record(summary):
    """Build reusable guidance only from prediction-time summary metadata."""
    if not isinstance(summary, dict):
        raise ValueError("Test 6 transfer memory requires a prediction-time summary")
    notes = [
        "Compare a candidate forecast with the pre-target history and do not assume continuation or reversal beyond what the available evidence establishes.",
        "Use text or covariates only when their relation to the target, units, and time alignment is clear.",
    ]
    dataset_info = summary.get("dataset_info")
    if isinstance(dataset_info, dict) and (
        dataset_info.get("target_variable") in (None, "", "unknown")
        or dataset_info.get("target_unit") in (None, "", "unknown")
    ):
        notes.append(
            "When target identity or units are unclear, do not infer target-specific effects from domain text without an established link."
        )
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
        "transfer_schema_version": CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
        "reflection": {
            "applicability_conditions": notes,
            "exclusions": [
                "This transfer record contains no realized target values, forecast values, or outcome-specific error pattern."
            ],
        },
    }


def build_test6_transfer_projection(
    correction, *, source_horizon, test6_transfer_revision=None,
    outcome_blind=False, summary=None,
):
    """Expose only numeric-free, direction-free qualitative transfer notes to Test 6."""
    if type(source_horizon) is not int or source_horizon <= 0:
        raise ValueError("source_horizon must be a positive integer")
    if type(outcome_blind) is not bool:
        raise ValueError("outcome_blind must be a boolean")
    if outcome_blind:
        try:
            record = make_test6_outcome_blind_transfer_record(summary)
        except ValueError:
            return None
    else:
        try:
            record = json.loads(correction)
        except (TypeError, json.JSONDecodeError):
            return None
    reflection = record.get("reflection") if isinstance(record, dict) else None
    if not isinstance(reflection, dict):
        return None

    fields = []
    if record.get("schema_version") == CORRECTION_SCHEMA_VERSION_V4:
        fields.extend(("explanation_hypotheses", "applicability_conditions", "exclusions"))
    elif record.get("schema_version") == CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND:
        if record.get("transfer_schema_version") != CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND:
            return None
        fields.extend(("applicability_conditions", "exclusions"))
    elif record.get("schema_version") in {
        CORRECTION_SCHEMA_VERSION, CORRECTION_SCHEMA_VERSION_V3,
    }:
        fields.extend(("key_insights_hypothesis", "applicability_conditions", "exclusions"))
    else:
        return None

    safe = []
    for field in fields:
        values = reflection.get(field)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            if (
                isinstance(value, str) and value.strip()
                and not _NUMERIC_DIGIT_RE.search(value)
                and not _ERROR_DIRECTION_RE.search(value)
                and not _HISTORICAL_OUTCOME_RE.search(value)
            ):
                safe.append({"kind": field, "text": " ".join(value.split())})
    revision_notes = []
    if test6_transfer_revision is not None:
        if not isinstance(test6_transfer_revision, dict):
            return None
        notes = test6_transfer_revision.get("qualitative_transfer_notes")
        if not isinstance(notes, list) or not notes:
            return None
        for note in notes:
            if (
                isinstance(note, str) and note.strip()
                and not _NUMERIC_DIGIT_RE.search(note)
                and not _ERROR_DIRECTION_RE.search(note)
                and not _HISTORICAL_OUTCOME_RE.search(note)
            ):
                revision_notes.append(" ".join(note.split()))
        if not revision_notes:
            return None
    if not any(row["kind"] in {"applicability_conditions", "key_insights_hypothesis"}
               for row in safe) and not revision_notes:
        return None
    projection = {
        "historical_source_horizon": source_horizon,
        "qualitative_transfer_notes": safe,
        "limitations": (
            "This is qualitative experience from the recorded source horizon. "
            "It does not establish performance at a longer horizon."
        ),
    }
    if revision_notes:
        projection["test6_revision_notes"] = revision_notes
    return canonical_json(projection)


def test6_transfer_scope_eligibility(projection, *, source_metadata, query):
    """Fail closed when a memory's own scope excludes a migrated query."""
    if isinstance(projection, str):
        try:
            projection = json.loads(projection)
        except json.JSONDecodeError:
            return False, "invalid_transfer_projection"
    if not isinstance(projection, dict):
        return False, "invalid_transfer_projection"
    if not isinstance(source_metadata, dict) or not isinstance(query, dict):
        return False, "invalid_transfer_scope_context"
    source_horizon = projection.get("historical_source_horizon")
    query_horizon = query.get("horizon")
    if type(source_horizon) is not int or type(query_horizon) is not int:
        return False, "invalid_transfer_scope_horizon"
    notes = projection.get("qualitative_transfer_notes")
    if not isinstance(notes, list):
        return False, "invalid_transfer_projection"
    texts = [
        row.get("text", "").casefold()
        for row in notes
        if isinstance(row, dict) and isinstance(row.get("text"), str)
    ]

    def unresolved(value):
        return (
            not isinstance(value, str)
            or value.strip().casefold() in {
                "", "unknown", "unspecified", "undocumented", "none",
                "null", "n/a", "na",
            }
        )

    for field in ("target", "unit"):
        scope_mentions_field = any(
            re.search(rf"\b{field}\b", text) for text in texts
        )
        requires_verified_match = any(
            scope_mentions_field
            and re.search(r"\b(verified|confirmed|match|matching|same)\b", text)
            for text in texts
        )
        explicitly_excludes_unknown = any(
            scope_mentions_field
            and re.search(
                r"\b(?:do not transfer|do not apply|don't transfer|exclude|"
                r"not applicable|avoid)\b.{0,90}\b(?:unidentified|unverified|"
                r"unconfirmed|unknown)\b",
                text,
            )
            for text in texts
        )
        if (
            (requires_verified_match or explicitly_excludes_unknown)
            and (
                unresolved(source_metadata.get(field))
                or unresolved(query.get(field))
                or source_metadata.get(field) != query.get(field)
            )
        ):
            return False, "memory_requires_verified_target_or_unit"

    if source_horizon != query_horizon:
        number_words = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
            19: "nineteen", 20: "twenty", 24: "twenty-four", 36: "thirty-six",
            48: "forty-eight",
        }
        source_word = number_words.get(source_horizon)
        source_duration_claim = source_word and re.compile(
            rf"\b{re.escape(source_word)}[ -](?:month|months|week|weeks|step|steps)\b"
        )
        for text in texts:
            specific_source_duration = bool(
                source_duration_claim and source_duration_claim.search(text)
            )
            explicit_same_horizon = bool(
                re.search(r"\b(?:same|only|exact|specific|required)\b.{0,45}\bhorizon\b", text)
                or re.search(r"\bhorizon\b.{0,45}\b(?:same|only|exact|required)\b", text)
            )
            explicit_horizon_exclusion = bool(
                re.search(r"\b(?:do not transfer|do not apply|don't transfer|not applicable|exclude)\b.{0,90}\b(?:different|another|other|any|longer|shorter)\b.{0,45}\bhorizon\b", text)
            )
            if specific_source_duration or explicit_same_horizon or explicit_horizon_exclusion:
                return False, "memory_horizon_scope_mismatch"
    return True, "eligible"


def _finite_vector(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain only finite numbers") from None
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite numbers")
    return result


def _largest_step_change(values):
    if len(values) < 2:
        return {"signed_change": None, "absolute_change": None, "step_number": None}
    changes = [right - left for left, right in zip(values, values[1:])]
    index = max(range(len(changes)), key=lambda i: abs(changes[i]))
    return {
        "signed_change": changes[index],
        "absolute_change": abs(changes[index]),
        "step_number": index + 1,
    }


def _error_sign_runs(signed_errors):
    """Compress stepwise prediction errors into contiguous direction segments."""
    labels = [
        "overestimate" if error > 0 else "underestimate" if error < 0 else "exact"
        for error in signed_errors
    ]
    runs = []
    for index, direction in enumerate(labels):
        if runs and runs[-1]["direction"] == direction:
            runs[-1]["end_step"] = index + 1
        else:
            runs.append({
                "start_step": index + 1,
                "end_step": index + 1,
                "direction": direction,
            })
    return runs


def compute_error_facts(history, prediction, truth):
    """Calculate immutable numeric error facts for one completed fit forecast."""
    observed_history = _finite_vector(history, "history")
    predicted = _finite_vector(prediction, "prediction")
    observed = _finite_vector(truth, "truth")
    if len(predicted) != len(observed):
        raise ValueError("prediction and truth must have the same non-zero length")

    signed_errors = [guess - actual for guess, actual in zip(predicted, observed)]
    squared_error = sum(error * error for error in signed_errors) / len(signed_errors)
    actual_steps = _largest_step_change(observed)
    predicted_steps = _largest_step_change(predicted)
    return {
        "schema_version": FACTS_SCHEMA_VERSION,
        "n": len(observed),
        "history_length": len(observed_history),
        "prediction_length": len(predicted),
        "truth_length": len(observed),
        "mae": sum(abs(error) for error in signed_errors) / len(signed_errors),
        "mse": squared_error,
        "rmse": math.sqrt(squared_error),
        "mean_signed_error_prediction_minus_truth": sum(signed_errors) / len(signed_errors),
        "signed_error_by_step": signed_errors,
        "error_sign_runs": _error_sign_runs(signed_errors),
        "first_prediction_error": signed_errors[0],
        "first_actual_change_from_history": observed[0] - observed_history[-1],
        "first_prediction_change_from_history": predicted[0] - observed_history[-1],
        "max_abs_actual_step_change": actual_steps["absolute_change"],
        "max_actual_step_signed_change": actual_steps["signed_change"],
        "max_actual_step_number": actual_steps["step_number"],
        "max_abs_prediction_step_change": predicted_steps["absolute_change"],
        "max_prediction_step_signed_change": predicted_steps["signed_change"],
        "max_prediction_step_number": predicted_steps["step_number"],
    }


def build_verified_correction_prompt(
    summary, history, truth, prediction, original_reasoning, verified_facts,
):
    """Ask the LLM only for a reflection; code-owned facts remain authoritative."""
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    input_data = {
        "summary": summary,
        "history": history,
        "truth": truth,
        "prediction": prediction,
        "original_reasoning": original_reasoning,
        "verified_error_facts_computed_by_code": verified_facts,
    }
    return f"""You are reviewing one completed training forecast for a time-series task.

CASE AND OUTCOME DATA (training case only):
{canonical_json(input_data)}

The verified_error_facts_computed_by_code object was computed by code and is
authoritative arithmetic.
Do not replace, recalculate, contradict, or add numeric claims to those facts.
Explain possible reasons for the error only as hypotheses, not verified causes.
Use only evidence present in this case. State when the case does not support an
explanation. Keep applicability conditions specific to matching target, unit,
frequency, horizon, and a similar error pattern. Include clear exclusions when
the lesson should not transfer. Analyze the code-computed stepwise error signs
and describe changing error direction by segment; do not reduce a mixed pattern
to one overall correction direction. Do not repeat old numeric predictions as advice.

Return only a JSON object with exactly these fields:
{{"Improved Reasoning":"...","Key Insights":"...",
"applicability_conditions":["..."],"exclusions":["..."]}}
"""


def build_verified_correction_prompt_v3(
    summary, history, truth, prediction, original_reasoning, verified_facts, *, task_context,
):
    """Build a task-bound reflection prompt without changing the frozen v2 path.

    V3 keeps exact quantities in code-computed facts and asks the free-text
    reflection to describe patterns qualitatively. It is an opt-in helper; live
    runs must pin this prompt explicitly in a new run manifest before use.
    """
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    required_context = ("dataset", "target", "unit", "frequency", "horizon")
    if not isinstance(task_context, dict):
        raise ValueError("task_context must be an object")
    context = {}
    for key in required_context:
        value = task_context.get(key)
        if key == "horizon":
            if type(value) is not int or value <= 0:
                raise ValueError("task_context horizon must be a positive integer")
            context[key] = value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task_context {key} must be a non-empty string")
            context[key] = " ".join(value.split())

    input_data = {
        "task_context": context,
        "summary": summary,
        "history": history,
        "truth": truth,
        "prediction": prediction,
        "original_reasoning": original_reasoning,
        "verified_error_facts_computed_by_code": verified_facts,
    }
    return f"""You are reviewing one completed training forecast for a time-series task.

TASK CONTEXT:
{canonical_json(context)}

CASE AND OUTCOME DATA (training case only):
{canonical_json(input_data)}

The verified_error_facts_computed_by_code object is authoritative arithmetic.
Do not replace, recalculate, contradict, or add numeric claims to those facts.
Do not put numeric literals in the free-text reflection fields (including
digits, decimal values, percentages, dates, counts, or numeric ranges). Keep
exact quantities only in the code-computed facts; describe patterns
qualitatively in prose. Explain possible reasons as hypotheses, not verified
causes. Use only evidence present in this case and state when the case
does not support an explanation. Match applicability to the task context,
especially target, unit, frequency, horizon, and error pattern. Include clear
exclusions. Do not turn a hypothesis into a causal claim or repeat a forecast
value as advice. Please describe changing error direction using the code-computed
error-sign segments; preserve horizon stages instead of stating one global
over- or under-estimation lesson.

Return only a JSON object with exactly these fields:
{{"Improved Reasoning":"...","Key Insights":"...",
"applicability_conditions":["..."],"exclusions":["..."]}}
"""


def build_verified_correction_prompt_v4(
    summary, history, truth, prediction, original_reasoning, verified_facts, *, task_context,
):
    """Build a structured reflection that separates observed errors from transfer hypotheses."""
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    if not isinstance(task_context, dict):
        raise ValueError("task_context must be an object")
    context = {}
    for key in ("dataset", "target", "unit", "frequency", "horizon"):
        value = task_context.get(key)
        if key == "horizon":
            if type(value) is not int or value <= 0:
                raise ValueError("task_context horizon must be a positive integer")
            context[key] = value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task_context {key} must be a non-empty string")
            context[key] = " ".join(value.split())

    input_data = {
        "task_context": context,
        "summary": summary,
        "history": history,
        "truth": truth,
        "prediction": prediction,
        "original_reasoning": original_reasoning,
        "verified_error_facts_computed_by_code": verified_facts,
    }
    return f"""You are writing a reusable reflection for one completed training forecast.

TASK CONTEXT:
{canonical_json(context)}

CASE AND OUTCOME DATA (training case only):
{canonical_json(input_data)}

The code-computed verified_error_facts are the sole authority for numeric facts.
Do not alter them or put numeric literals, dates, counts, percentages, or ranges
in any free-text output field. Keep observed error patterns separate from
explanation hypotheses: describe the stepwise error-sign segments without
compressing a change in direction into one global correction rule. Explanations
are unverified hypotheses, not causes. Applicability and exclusion conditions
must be checkable using prediction-time evidence only: task context, visible
history, the current forecast's level/trend/volatility, and text quality or
relevance. Do not refer to the current case's truth or current error when
describing a condition. State what must not transfer, including old error signs,
absolute error magnitudes, and unsupported domain explanations. A single case is
not proof that a lesson generalizes.

Return only a JSON object with exactly these fields:
{{"observed_error_pattern":"...","explanation_hypotheses":["..."],
"applicability_conditions":["..."],"exclusions":["..."],
"non_transferable_content":["..."]}}
"""


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _text_list(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return [" ".join(item.split()) for item in value]


def make_verified_correction_record(verified_facts, model_response):
    """Keep code-derived facts separate; never accept model-supplied metric values."""
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    if not isinstance(model_response, dict):
        raise ValueError("model_response must be a JSON object")
    applicability = _text_list(
        model_response.get("applicability_conditions", []), "applicability_conditions"
    )
    exclusions = _text_list(model_response.get("exclusions", []), "exclusions")
    key_insights = model_response.get("Key Insights")
    if isinstance(key_insights, list):
        key_insights = "; ".join(_text_list(key_insights, "Key Insights"))
    if not applicability:
        raise ValueError("applicability_conditions must not be empty")
    if not exclusions:
        raise ValueError("exclusions must not be empty")
    reflection = {
        "improved_reasoning_hypothesis": _text(
            model_response.get("Improved Reasoning"), "Improved Reasoning"
        ),
        "key_insights_hypothesis": _text(key_insights, "Key Insights"),
        "applicability_conditions": applicability,
        "exclusions": exclusions,
    }
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "verified_facts": json.loads(canonical_json(verified_facts)),
        "reflection": reflection,
    }


def _numeric_free_text(value, label):
    text = _text(value, label)
    if _NUMERIC_DIGIT_RE.search(text):
        raise ValueError(f"{label} must not contain a numeric literal")
    return text


def _validate_numeric_free_reflection(reflection):
    if not isinstance(reflection, dict):
        raise ValueError("reflection must be an object")
    expected_fields = {
        "improved_reasoning_hypothesis", "key_insights_hypothesis",
        "applicability_conditions", "exclusions",
    }
    if set(reflection) != expected_fields:
        raise ValueError("v3 reflection must contain exactly the supported fields")
    for field in ("improved_reasoning_hypothesis", "key_insights_hypothesis"):
        _numeric_free_text(reflection.get(field), field)
    for field in ("applicability_conditions", "exclusions"):
        values = _text_list(reflection.get(field), field)
        if not values:
            raise ValueError(f"{field} must not be empty")
        for index, value in enumerate(values):
            _numeric_free_text(value, f"{field}[{index}]")
    return True


def numeric_free_reflection_is_valid(reflection):
    """Return whether a record has exactly the v3 fields and no digit text."""
    try:
        _validate_numeric_free_reflection(reflection)
    except ValueError:
        return False
    return True


def make_verified_correction_record_v3(verified_facts, model_response):
    """Create an opt-in v3 record with code-owned numeric facts.

    The frozen v2 runner is unchanged. A v3 run must pin both its prompt and
    record schemas in a new manifest before generating memories.
    """
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    if not isinstance(model_response, dict):
        raise ValueError("model_response must be a JSON object")
    expected_fields = {
        "Improved Reasoning", "Key Insights", "applicability_conditions", "exclusions"
    }
    if set(model_response) != expected_fields:
        raise ValueError("v3 model_response must contain exactly the supported reflection fields")
    key_insights = model_response.get("Key Insights")
    if isinstance(key_insights, list):
        key_insights = "; ".join(_text_list(key_insights, "Key Insights"))
    reflection = {
        "improved_reasoning_hypothesis": _numeric_free_text(
            model_response.get("Improved Reasoning"), "Improved Reasoning"
        ),
        "key_insights_hypothesis": _numeric_free_text(key_insights, "Key Insights"),
        "applicability_conditions": _text_list(
            model_response.get("applicability_conditions"), "applicability_conditions"
        ),
        "exclusions": _text_list(model_response.get("exclusions"), "exclusions"),
    }
    _validate_numeric_free_reflection(reflection)
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION_V3,
        "reflection_schema_version": CORRECTION_REFLECTION_SCHEMA_VERSION_V3,
        "verified_facts": json.loads(canonical_json(verified_facts)),
        "reflection": reflection,
    }


def _validate_numeric_free_reflection_v4(reflection):
    if not isinstance(reflection, dict):
        raise ValueError("reflection must be an object")
    expected_fields = {
        "observed_error_pattern", "explanation_hypotheses",
        "applicability_conditions", "exclusions", "non_transferable_content",
    }
    if set(reflection) != expected_fields:
        raise ValueError("v4 reflection must contain exactly the supported fields")
    _numeric_free_text(reflection.get("observed_error_pattern"), "observed_error_pattern")
    for field in (
        "explanation_hypotheses", "applicability_conditions", "exclusions",
        "non_transferable_content",
    ):
        values = _text_list(reflection.get(field), field)
        if not values:
            raise ValueError(f"{field} must not be empty")
        for index, value in enumerate(values):
            _numeric_free_text(value, f"{field}[{index}]")
    return True


def make_verified_correction_record_v4(verified_facts, model_response):
    """Create a task-bound v4 correction memory with explicit transfer limits."""
    if not isinstance(verified_facts, dict) or verified_facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("verified_facts must use the supported code-computed schema")
    if not isinstance(model_response, dict):
        raise ValueError("model_response must be a JSON object")
    expected_fields = {
        "observed_error_pattern", "explanation_hypotheses",
        "applicability_conditions", "exclusions", "non_transferable_content",
    }
    if set(model_response) != expected_fields:
        raise ValueError("v4 model_response must contain exactly the supported reflection fields")
    reflection = {
        "observed_error_pattern": _numeric_free_text(
            model_response.get("observed_error_pattern"), "observed_error_pattern"
        ),
    }
    for field in (
        "explanation_hypotheses", "applicability_conditions", "exclusions",
        "non_transferable_content",
    ):
        reflection[field] = _text_list(model_response.get(field), field)
    _validate_numeric_free_reflection_v4(reflection)
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION_V4,
        "reflection_schema_version": CORRECTION_REFLECTION_SCHEMA_VERSION_V4,
        "verified_facts": json.loads(canonical_json(verified_facts)),
        "reflection": reflection,
    }


def render_verified_correction(record):
    if not isinstance(record, dict) or record.get("schema_version") != CORRECTION_SCHEMA_VERSION:
        raise ValueError("record must use the supported verified-correction schema")
    if not isinstance(record.get("verified_facts"), dict) or not isinstance(record.get("reflection"), dict):
        raise ValueError("record must contain verified_facts and reflection objects")
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def render_verified_correction_v3(record):
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != CORRECTION_SCHEMA_VERSION_V3
        or record.get("reflection_schema_version") != CORRECTION_REFLECTION_SCHEMA_VERSION_V3
    ):
        raise ValueError("record must use the supported v3 verified-correction schemas")
    facts = record.get("verified_facts")
    if not isinstance(facts, dict) or facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("v3 record must contain the supported code-computed facts")
    _validate_numeric_free_reflection(record.get("reflection"))
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def render_verified_correction_v4(record):
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != CORRECTION_SCHEMA_VERSION_V4
        or record.get("reflection_schema_version") != CORRECTION_REFLECTION_SCHEMA_VERSION_V4
    ):
        raise ValueError("record must use the supported v4 verified-correction schemas")
    facts = record.get("verified_facts")
    if not isinstance(facts, dict) or facts.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise ValueError("v4 record must contain the supported code-computed facts")
    _validate_numeric_free_reflection_v4(record.get("reflection"))
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _iso_date(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO date or timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{label} must be an ISO date or timestamp") from None


def _forecast_origin_identity(case_id):
    """Normalize HIC fit/validation case IDs to dataset series origin identity."""
    if not isinstance(case_id, str):
        return None
    match = re.fullmatch(r"(.+?)_(?:fit|validation|test)_(.+)_(\d{8}T\d{6})", case_id)
    return (match.group(1), match.group(2), match.group(3)) if match else None


def _forecast_entity_identity(record):
    if not isinstance(record, dict):
        return None
    entity = record.get("entity")
    if isinstance(entity, list) and entity and all(
        isinstance(value, str) and value.strip() for value in entity
    ):
        dataset = record.get("dataset")
        if isinstance(dataset, str) and dataset.strip():
            return (dataset.casefold(), *entity)
        return None
    origin = _forecast_origin_identity(record.get("case_id"))
    return (origin[0].casefold(), origin[1]) if origin is not None else None


def memory_eligibility(memory, query):
    """Check provenance, completion, transfer scope, and temporal availability."""
    if not isinstance(memory, dict) or not isinstance(query, dict):
        return False, "invalid_record"
    if memory.get("case_id") == query.get("case_id"):
        return False, "same_case"
    if memory.get("completed") is not True:
        return False, "memory_incomplete"
    migration_protocol = query.get("horizon_migration_protocol")
    semantic_transfer = migration_protocol == TEST6_SEMANTIC_TRANSFER_PROTOCOL
    if memory.get("split") != "fit":
        online_round_id = query.get("test6_online_round_id")
        completed_online = (
            migration_protocol in TEST6_ONLINE_MEMORY_PROTOCOLS
            and memory.get("split") == "validation"
            and isinstance(memory.get("test6_round_id"), str)
            and bool(memory.get("test6_round_id").strip())
            and memory.get("test6_scoring_status") == "complete"
        )
        if semantic_transfer:
            completed_online = completed_online and (
                isinstance(online_round_id, str) and bool(online_round_id.strip())
            )
            if completed_online and memory.get("test6_round_id") == online_round_id:
                prior_case_ids = query.get("test6_prior_memory_case_ids")
                if (not isinstance(prior_case_ids, list)
                        or any(not isinstance(value, str) for value in prior_case_ids)):
                    return False, "invalid_test6_prior_memory_scope"
                if memory.get("case_id") not in prior_case_ids:
                    return False, "test6_memory_not_previously_completed"
        else:
            completed_online = completed_online and (
                isinstance(online_round_id, str)
                and bool(online_round_id.strip())
                and memory.get("test6_round_id") == online_round_id
            )
        if not completed_online:
            return False, "not_fit_or_completed_test6_memory"
    supported_correction_versions = {
        CORRECTION_SCHEMA_VERSION,
        CORRECTION_SCHEMA_VERSION_V3,
        CORRECTION_SCHEMA_VERSION_V4,
        CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
    }
    accepted_correction_versions = query.get("required_correction_schema_versions")
    if accepted_correction_versions is None:
        required_correction_version = query.get(
            "required_correction_schema_version", CORRECTION_SCHEMA_VERSION
        )
        if type(required_correction_version) is not int or required_correction_version not in supported_correction_versions:
            return False, "unsupported_required_correction_schema"
        accepted_correction_versions = [required_correction_version]
    elif (
        not isinstance(accepted_correction_versions, list)
        or not accepted_correction_versions
        or any(
            type(version) is not int or version not in supported_correction_versions
            for version in accepted_correction_versions
        )
        or len(set(accepted_correction_versions)) != len(accepted_correction_versions)
    ):
        return False, "unsupported_required_correction_schema"
    memory_correction_version = memory.get("correction_schema_version")
    if memory_correction_version not in accepted_correction_versions:
        return False, "correction_not_verified_schema"

    prompt_versions_by_schema = query.get("required_correction_prompt_versions_by_schema")
    if prompt_versions_by_schema is not None:
        if not isinstance(prompt_versions_by_schema, dict):
            return False, "unsupported_required_correction_prompt"
        allowed_prompt_versions = prompt_versions_by_schema.get(str(memory_correction_version))
        if (
            not isinstance(allowed_prompt_versions, list)
            or not allowed_prompt_versions
            or any(
                version is not None
                and (not isinstance(version, str) or not version.strip())
                for version in allowed_prompt_versions
            )
        ):
            return False, "unsupported_required_correction_prompt"
        if memory.get("correction_prompt_version") not in allowed_prompt_versions:
            return False, "correction_prompt_version_mismatch"
    else:
        required_prompt_version = query.get("required_correction_prompt_version")
        if required_prompt_version is not None:
            if (
                memory_correction_version not in {
                    CORRECTION_SCHEMA_VERSION_V3, CORRECTION_SCHEMA_VERSION_V4,
                }
                or not isinstance(required_prompt_version, str)
                or not required_prompt_version.strip()
            ):
                return False, "unsupported_required_correction_prompt"
            if memory.get("correction_prompt_version") != required_prompt_version:
                return False, "correction_prompt_version_mismatch"
    if memory.get("news_quality") != "valid":
        return False, "memory_has_no_valid_news"
    if query.get("news_quality") != "valid":
        return False, "query_has_no_valid_news"
    for record in (memory, query):
        source_sha256 = record.get("source_sha256")
        if not isinstance(source_sha256, str) or not source_sha256:
            return False, "missing_source_sha256"
        if not _SHA256_RE.fullmatch(source_sha256):
            return False, "invalid_source_sha256"
    if not semantic_transfer and memory["source_sha256"] != query["source_sha256"]:
        return False, "source_sha256_mismatch"
    if migration_protocol in {
        "agent_test6_cross_horizon_v1",
        "agent_test6_sequential_online_v1",
    }:
        memory_entity = _forecast_entity_identity(memory)
        query_entity = _forecast_entity_identity(query)
        if memory_entity is None or query_entity is None:
            return False, "entity_identity_unknown"
        if memory_entity != query_entity:
            return False, "entity_mismatch"
    allowed_source_horizons = query.get("allowed_source_horizons")
    if migration_protocol is not None and not semantic_transfer:
        if (migration_protocol not in TEST6_ONLINE_MEMORY_PROTOCOLS
                or not isinstance(allowed_source_horizons, list)
                or not allowed_source_horizons
                or any(type(value) is not int or value <= 0 for value in allowed_source_horizons)):
            return False, "invalid_horizon_migration_protocol"
    for field in ("dataset", "target", "unit", "frequency"):
        if field not in memory or field not in query or memory[field] in (None, "") or query[field] in (None, ""):
            return False, f"missing_{field}"
        if not semantic_transfer and memory[field] != query[field]:
            return False, f"{field}_mismatch"
    if "horizon" not in memory or type(memory["horizon"]) is not int:
        return False, "missing_horizon"
    if "horizon" not in query or type(query["horizon"]) is not int:
        return False, "missing_horizon"
    if memory["horizon"] != query["horizon"]:
        if migration_protocol not in {
            "agent_test6_cross_horizon_v1",
            TEST6_SEMANTIC_TRANSFER_PROTOCOL,
        }:
            return False, "horizon_mismatch"
        is_completed_test6_case = (
            memory.get("split") == "validation"
            and isinstance(query.get("test6_online_round_id"), str)
            and memory.get("test6_round_id") == query.get("test6_online_round_id")
            and memory.get("test6_scoring_status") == "complete"
        )
        if migration_protocol == "agent_test6_cross_horizon_v1" and is_completed_test6_case:
            return False, "source_horizon_not_allowed"
        if (migration_protocol in {
                "agent_test6_cross_horizon_v1",
            }
                and memory["horizon"] not in allowed_source_horizons):
            return False, "source_horizon_not_allowed"
    if migration_protocol in TEST6_ONLINE_MEMORY_PROTOCOLS:
        memory_origin = _forecast_origin_identity(memory.get("case_id"))
        query_origin = _forecast_origin_identity(query.get("case_id"))
        if memory_origin is not None and query_origin is not None and memory_origin == query_origin:
            return False, "same_forecast_origin"
    require_strict_history_start = query.get("require_strict_history_start", False)
    if type(require_strict_history_start) is not bool:
        return False, "invalid_temporal_metadata"
    try:
        outcome_available = _iso_date(memory.get("target_end"), "memory.target_end")
        if require_strict_history_start:
            query_cutoff = _iso_date(query.get("history_start"), "query.history_start")
        else:
            query_cutoff = _iso_date(query.get("history_end"), "query.history_end")
    except ValueError:
        return False, "invalid_temporal_metadata"
    if require_strict_history_start:
        if outcome_available >= query_cutoff:
            return False, "outcome_not_strictly_before_history_start"
    elif outcome_available > query_cutoff:
        return False, "outcome_not_available_by_query_cutoff"
    return True, "eligible"


def _summary_content_hash(summary):
    return hashlib.sha256(canonical_json(summary).encode("utf-8")).hexdigest()


def _candidate_content_eligibility(
    candidate, query, *, require_horizon_text_consistency=False
):
    """Require auditable summary lineage and a complete verified correction."""
    metadata = candidate.get("metadata")
    if metadata.get("case_id") != candidate.get("case_id"):
        return False, "candidate_metadata_id_mismatch"
    unresolved_units = {"", "unknown", "unspecified", "undocumented", "none", "null", "n/a", "na"}
    memory_unit = metadata.get("unit")
    query_unit = query.get("unit")
    memory_unit_unresolved = (
        not isinstance(memory_unit, str)
        or memory_unit.strip().casefold() in unresolved_units
    )
    query_unit_unresolved = (
        not isinstance(query_unit, str)
        or query_unit.strip().casefold() in unresolved_units
    )
    semantic_transfer = (
        query.get("horizon_migration_protocol")
        == TEST6_SEMANTIC_TRANSFER_PROTOCOL
    )
    if semantic_transfer:
        unit_compatibility_scope = "qualitative_semantic_transfer_unit_not_assumed"
    elif memory_unit_unresolved or query_unit_unresolved:
        if not (memory_unit_unresolved and query_unit_unresolved):
            return False, "unit_unverified"
        if metadata.get("source_sha256") != query.get("source_sha256"):
            return False, "unit_unverified"
        unit_compatibility_scope = "same_source_artifact_unknown_unit"
    else:
        unit_compatibility_scope = "verified_unit"

    summary = candidate.get("summary")
    if not isinstance(summary, (dict, list, str)) or not summary:
        return False, "invalid_summary"
    if isinstance(summary, str) and not summary.strip():
        return False, "invalid_summary"

    if metadata.get("summary_identity_status") != "exact_request_response":
        return False, "summary_identity_unresolved"
    request_hash = metadata.get("summary_request_sha256")
    content_hash = metadata.get("summary_content_sha256")
    if not isinstance(request_hash, str) or not _SHA256_RE.fullmatch(request_hash):
        return False, "summary_identity_invalid"
    if not isinstance(content_hash, str) or not _SHA256_RE.fullmatch(content_hash):
        return False, "summary_identity_invalid"
    try:
        if _summary_content_hash(summary) != content_hash:
            return False, "summary_content_hash_mismatch"
    except (TypeError, ValueError):
        return False, "invalid_summary"

    correction = candidate.get("correction")
    if not isinstance(correction, str) or not correction.strip():
        return False, "invalid_correction_record"
    try:
        record = json.loads(correction)
    except (TypeError, json.JSONDecodeError):
        return False, "invalid_correction_record"
    if not isinstance(record, dict):
        return False, "invalid_correction_record"
    correction_version = record.get("schema_version")
    if (
        correction_version not in {
            CORRECTION_SCHEMA_VERSION, CORRECTION_SCHEMA_VERSION_V3,
            CORRECTION_SCHEMA_VERSION_V4,
            CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
        }
        or metadata.get("correction_schema_version") != correction_version
    ):
        return False, "invalid_correction_record"
    if correction_version == CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND:
        completed_test6_memory = (
            metadata.get("split") == "validation"
            and metadata.get("test6_scoring_status") == "complete"
            and isinstance(metadata.get("test6_round_id"), str)
            and bool(metadata.get("test6_round_id").strip())
        )
        if (
            metadata.get("test6_transfer_correction_version")
            != CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND
            or not completed_test6_memory
            or (not semantic_transfer
                and metadata.get("test6_round_id") != query.get("test6_online_round_id"))
        ):
            return False, "outcome_blind_transfer_not_completed_test6_memory"
        try:
            expected_record = make_test6_outcome_blind_transfer_record(candidate.get("summary"))
        except ValueError:
            return False, "invalid_correction_record"
        if record != expected_record:
            return False, "invalid_correction_record"
        for field in (
            "test6_input_sha256", "test6_forecast_lock_sha256",
            "test6_truth_sha256", "test6_score_sha256",
            "test6_audit_reflection_sha256", "test6_reflection_request_sha256",
            "baseline_request_sha256", "baseline_prediction_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                return False, "test6_transfer_provenance_missing"
        return True, unit_compatibility_scope
    facts = record.get("verified_facts")
    reflection = record.get("reflection")
    if not isinstance(facts, dict) or not isinstance(reflection, dict):
        return False, "invalid_correction_record"
    if correction_version == CORRECTION_SCHEMA_VERSION_V3:
        for field in ("baseline_request_sha256", "baseline_prediction_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                return False, "baseline_provenance_missing"
        if record.get("reflection_schema_version") != CORRECTION_REFLECTION_SCHEMA_VERSION_V3:
            return False, "invalid_correction_record"
        try:
            _validate_numeric_free_reflection(reflection)
        except ValueError:
            return False, "invalid_correction_record"
    elif correction_version == CORRECTION_SCHEMA_VERSION_V4:
        for field in ("baseline_request_sha256", "baseline_prediction_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                return False, "baseline_provenance_missing"
        if record.get("reflection_schema_version") != CORRECTION_REFLECTION_SCHEMA_VERSION_V4:
            return False, "invalid_correction_record"
        try:
            _validate_numeric_free_reflection_v4(reflection)
        except ValueError:
            return False, "invalid_correction_record"
    facts_version = facts.get("schema_version")
    required_facts_version = query.get("required_facts_schema_version")
    if required_facts_version is not None:
        if facts_version != required_facts_version:
            return False, "invalid_correction_record"
    elif facts_version not in {1, FACTS_SCHEMA_VERSION}:
        return False, "invalid_correction_record"
    # Verified facts describe the historical forecast's own horizon. During an
    # explicitly authorized transfer, never relabel those facts as query horizon.
    horizon = metadata.get("horizon")
    if type(horizon) is not int or type(facts.get("n")) is not int or facts["n"] != horizon:
        return False, "correction_horizon_mismatch"

    numeric_fields = (
        "mae", "rmse", "mean_signed_error_prediction_minus_truth",
        "first_prediction_error", "first_actual_change_from_history",
        "first_prediction_change_from_history", "max_abs_actual_step_change",
        "max_actual_step_signed_change", "max_abs_prediction_step_change",
        "max_prediction_step_signed_change",
    )
    for field in numeric_fields:
        if field not in facts:
            return False, "invalid_correction_record"
        value = facts.get(field)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            return False, "invalid_correction_record"
    if facts_version == FACTS_SCHEMA_VERSION:
        if "mse" not in facts or type(facts.get("mse")) not in (int, float):
            return False, "invalid_correction_record"
        if not math.isfinite(float(facts["mse"])) or facts["mse"] < 0:
            return False, "invalid_correction_record"
        for field in ("history_length", "prediction_length", "truth_length"):
            if type(facts.get(field)) is not int or facts[field] <= 0:
                return False, "invalid_correction_record"
        if facts["prediction_length"] != facts["n"] or facts["truth_length"] != facts["n"]:
            return False, "invalid_correction_record"
    for field in ("max_actual_step_number", "max_prediction_step_number"):
        if field not in facts:
            return False, "invalid_correction_record"
        value = facts.get(field)
        if value is None and facts["n"] > 1:
            return False, "invalid_correction_record"
        if value is not None and (type(value) is not int or not 1 <= value < facts["n"]):
            return False, "invalid_correction_record"

    if correction_version == CORRECTION_SCHEMA_VERSION_V4:
        # The strict v4 validator above checks all five structured fields.
        applicability_conditions = reflection["applicability_conditions"]
    else:
        for field in ("improved_reasoning_hypothesis", "key_insights_hypothesis"):
            if not isinstance(reflection.get(field), str) or not reflection[field].strip():
                return False, "invalid_correction_record"
        for field in ("applicability_conditions", "exclusions"):
            values = reflection.get(field)
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                return False, "invalid_correction_record"
        applicability_conditions = reflection["applicability_conditions"]
    if require_horizon_text_consistency:
        # This narrow lexical guard rejects explicit contradictions only; it
        # does not establish semantic applicability of a correction.
        mismatches = find_horizon_claim_mismatches(
            applicability_conditions, expected_horizon=facts["n"]
        )
        if mismatches:
            return False, "correction_horizon_text_mismatch"
    return True, unit_compatibility_scope


def rank_applicable_memories(
    candidates,
    query,
    *,
    require_verified_evidence=False,
    require_horizon_text_consistency=False,
):
    """Rank completed fit memories; optionally require strict task/news evidence."""
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    if type(require_verified_evidence) is not bool or type(require_horizon_text_consistency) is not bool:
        raise ValueError("memory evidence options must be booleans")
    ranked = []
    excluded = {}
    if require_verified_evidence:
        query_eligible, query_reasons = case_evidence_eligibility(query)
        if not query_eligible:
            return {
                "eligible": [],
                "excluded_counts": {
                    f"query_evidence:{reason}": len(candidates)
                    for reason in query_reasons
                },
                "nearest": None,
                "lowest_similarity_control": None,
            }
    for candidate in candidates:
        if not isinstance(candidate, dict):
            excluded["invalid_record"] = excluded.get("invalid_record", 0) + 1
            continue
        metadata = candidate.get("metadata")
        if require_verified_evidence:
            evidence_ok, evidence_reasons = paired_case_evidence_eligibility(query, metadata)
            if not evidence_ok:
                for reason in evidence_reasons:
                    key = reason
                    excluded[key] = excluded.get(key, 0) + 1
                continue
        eligible, reason = memory_eligibility(metadata, query)
        if not eligible:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        eligible, reason = _candidate_content_eligibility(
            candidate,
            query,
            require_horizon_text_consistency=(
                require_verified_evidence or require_horizon_text_consistency
            ),
        )
        if not eligible:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            excluded["invalid_similarity"] = excluded.get("invalid_similarity", 0) + 1
            continue
        ranked.append({
            **candidate,
            "score": float(score),
            "unit_compatibility_scope": reason,
        })
    ranked.sort(key=lambda item: (-item["score"], item.get("case_id", "")))
    return {
        "eligible": ranked,
        "excluded_counts": dict(sorted(excluded.items())),
        "nearest": ranked[0] if ranked else None,
        "lowest_similarity_control": ranked[-1] if len(ranked) > 1 else None,
    }
