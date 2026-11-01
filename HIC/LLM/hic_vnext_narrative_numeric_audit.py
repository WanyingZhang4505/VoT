"""Conservative deterministic checks for numeric claims in HIC reflections.

Only explicitly named, unambiguous facts are compared with code-computed values.
All other numbers are retained as manual-review items rather than called wrong.
"""

from __future__ import annotations

import math
import re


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?P<percent>%)?(?![\w])"
)
_FIELD_NAMES = (
    "improved_reasoning_hypothesis",
    "key_insights_hypothesis",
    "applicability_conditions",
    "exclusions",
)
_SIGNED_FACTS = {
    "mean_signed_error_prediction_minus_truth",
    "first_prediction_error",
    "first_actual_change_from_history",
    "first_prediction_change_from_history",
    "max_actual_step_signed_change",
    "max_prediction_step_signed_change",
}
_FACT_ANCHORS = (
    ("mae", "mae"),
    ("mse", "mse"),
    ("rmse", "rmse"),
    ("mean_signed_error_prediction_minus_truth", "mean\\s+signed\\s+error"),
    ("first_prediction_error", "first(?:[- ]step)?\\s+prediction\\s+error|first[- ]step\\s+error"),
    ("first_actual_change_from_history", "first\\s+actual(?:\\s+step)?\\s+change(?:\\s+from\\s+history)?"),
    ("first_prediction_change_from_history", "(?:first\\s+(?:predicted|prediction)(?:\\s+step)?|predicted\\s+first(?:\\s+step)?)\\s+change(?:\\s+from\\s+history)?"),
    ("max_abs_actual_step_change", "(?:largest|maximum|max)\\s+(?:(?:absolute|abs)\\s+)?(?:actual|truth|observed)\\s+(?:step(?:-to-step)?\\s+)?(?:change|move)|(?:actual|truth|observed)[^.;!?,]{0,32}(?:largest|maximum|max)[^.;!?,]{0,20}step(?:\\s+(?:change|move))?"),
    ("max_abs_prediction_step_change", "(?:largest|maximum|max)\\s+(?:(?:absolute|abs)\\s+)?(?:predicted|prediction)\\s+(?:step(?:-to-step)?\\s+)?(?:change|move)"),
    ("max_actual_step_signed_change", "(?:largest|maximum|max)\\s+(?:signed\\s+)?(?:actual|truth|observed)\\s+(?:step(?:-to-step)?\\s+)?(?:change|move)?|(?:actual|truth|observed)[^.;!?,]{0,32}(?:largest|maximum|max)[^.;!?,]{0,20}step(?:\\s+(?:change|move))?"),
    ("max_prediction_step_signed_change", "(?:largest|maximum|max)\\s+(?:signed\\s+)?(?:predicted|prediction)\\s+(?:step(?:-to-step)?\\s+)?(?:change|move)?|(?:predicted|prediction)\\s+(?:step(?:-to-step)?\\s+(?:change|move)?)(?=[^.;!?]{0,16}\\bmax\\b)"),
)


def _reflection_texts(reflection):
    if not isinstance(reflection, dict):
        return []
    texts = []
    for field in _FIELD_NAMES:
        value = reflection.get(field)
        if isinstance(value, str):
            texts.append((field, value))
        elif isinstance(value, list):
            texts.extend((field, item) for item in value if isinstance(item, str))
    return texts


def _is_list_enumerator(text, start, end, raw):
    following = text[end:end + 1]
    if following == ")" and raw.isdigit() and len(raw) <= 2 and (
        end + 1 >= len(text) or text[end + 1].isspace()
    ):
        return True
    if following != ".":
        return False
    # A short integer followed by a period and whitespace is an enumerator only
    # at a paragraph/list boundary; years and decimal facts remain auditable.
    if not raw.isdigit() or len(raw) > 2 or end + 1 >= len(text) or not text[end + 1].isspace():
        return False
    prefix = text[max(0, text.rfind("\n", 0, start) + 1, text.rfind(";", 0, start) + 1):start]
    return not prefix.strip() or prefix.strip().lower() in {"insight", "key insights", "exclusion"}


def _shape_claim(text, number_match, facts):
    """Return a direct history/horizon claim for tightly specified wording."""
    start, end = number_match.span("number")
    left = text[max(0, start - 55):start]
    right = text[end:min(len(text), end + 55)]
    raw = number_match.group("number")
    if number_match.group("percent"):
        return None
    patterns = (
        ("history_length", rf"(?i)(?:{re.escape(raw)}\s*[- ](?:point|observation)s?\s+(?:history|window)|(?:history|lookback)[^.;!?]{{0,28}}{re.escape(raw)}\s*(?:points?|observations?))"),
        ("prediction_length", rf"(?i)(?:{re.escape(raw)}\s*[- ](?:step|month)(?:-ahead)?\s+(?:forecast|horizon)|(?:forecast|horizon)[^.;!?]{{0,28}}{re.escape(raw)}\s*(?:steps?|months?|periods?)|(?:over|across|for)\s+{re.escape(raw)}\s+steps?)"),
    )
    context = left + raw + right
    for key, pattern in patterns:
        if key in facts and re.search(pattern, context):
            return key
    return None


def _is_numeric_range(text, number_match):
    start, end = number_match.span("number")
    for match in re.finditer(
        r"(?<![\w.])\d+(?:\.\d+)?\s*(?:[-–—]|\bto\b)\s*\d+(?:\.\d+)?(?![\w.])",
        text,
        re.I,
    ):
        if match.start() <= start and end <= match.end():
            return True
    return False


def _is_abbreviated_magnitude(text, number_match):
    start, end = number_match.span("number")
    for match in re.finditer(
        r"(?i)(?<![\w.])\d+(?:\.\d+)?\s*(?:[kmb]|thousand|million|billion)(?!\w)",
        text,
    ):
        if match.start() <= start and end <= match.end():
            return True
    return False


def _step_index_claim(text, number_match, facts):
    raw = number_match.group("number")
    if number_match.group("percent") or not raw.isdigit():
        return None
    start, end = number_match.span("number")
    marker = next(
        (item for item in re.finditer(r"(?i)at\s+step\s+(?P<index>\d+)(?![\w])", text)
         if item.span("index") == (start, end)),
        None,
    )
    if marker is None:
        return None
    boundaries = list(re.finditer(r"[!?;,]|\.(?!\d)|\n|\b(?:while|whereas|versus|vs\.?)\b", text, re.I))
    left_boundary = max((item.end() for item in boundaries if item.end() <= start), default=0)
    right_boundary = min((item.start() for item in boundaries if item.start() >= end), default=len(text))
    sentence = text[left_boundary:right_boundary]
    has_max_step = bool(re.search(r"(?i)\b(?:largest|maximum|max)\b[^.;!?]*\bstep\b", sentence))
    has_actual = bool(re.search(r"(?i)\b(?:actual|truth|observed)\b", sentence))
    has_prediction = bool(re.search(r"(?i)\b(?:predicted|prediction)\b", sentence))
    if not has_max_step or has_actual == has_prediction:
        return None
    key = "max_actual_step_number" if has_actual else "max_prediction_step_number"
    return key if key in facts else None


def _explicit_fact_key_claim(text, number_match, facts):
    """Bind a number immediately following an exact structured fact-key label."""
    start = number_match.start("number")
    prefix = text[:start]
    for key in sorted(facts, key=len, reverse=True):
        if not isinstance(key, str):
            continue
        pattern = rf"(?i)(?<![\w]){re.escape(key)}(?![\w])\s*(?::|=)?\s*$"
        if re.search(pattern, prefix[-len(key) - 8:]):
            return key
    return None


def _nearest_fact_anchor(text, number_match, facts):
    if number_match.group("percent"):
        return None, "percentage_not_mapped_to_a_verified_fact"
    number_start, number_end = number_match.span("number")
    candidates = []
    for key, phrase in _FACT_ANCHORS:
        if key not in facts:
            continue
        for match in re.finditer(rf"(?i)(?:{phrase})", text):
            # A label after a number is too easy to attach to an unrelated
            # level/value earlier in the sentence. Only bind label -> value.
            if match.end() > number_start:
                continue
            distance = number_start - match.end()
            between = text[match.end():number_start]
            if re.search(r";|,|\n|[!?]|\.(?!\d)|\b(?:and|but|while|whereas|than|versus|vs\.?)\b", between, re.I):
                continue
            if distance <= 48:
                candidates.append((distance, key, match.start(), match.end()))
    if not candidates:
        return None, "no_unambiguous_verified_fact_label_near_number"
    candidates.sort()
    best_distance = candidates[0][0]
    best_keys = {item[1] for item in candidates if item[0] == best_distance}
    if len(best_keys) != 1:
        return None, "multiple_verified_fact_labels_are_equally_close"
    key = next(iter(best_keys))
    # If different fact labels are almost equally close, the text-to-fact
    # assignment is too ambiguous to call a contradiction.
    if len(candidates) > 1 and candidates[1][1] != key and candidates[1][0] - best_distance <= 3:
        return None, "multiple_verified_fact_labels_are_too_close"
    if key in {"max_actual_step_signed_change", "max_prediction_step_signed_change"}:
        raw = number_match.group("number")
        if float(raw.replace(",", "")) > 0 and float(facts[key]) < 0:
            return None, "positive_magnitude_or_signed_change_is_ambiguous"
    return key, None


def _precision_tolerance(raw, expected=None, text="", start=0):
    mantissa = re.split(r"[eE]", raw, maxsplit=1)[0]
    decimals = len(mantissa.partition(".")[2]) if "." in mantissa else 0
    tolerance = 0.5 * (10 ** -decimals) + 1e-9
    approximation_context = text[max(0, start - 18):start].lower()
    if re.search(r"(?:\babout\b|\baround\b|\broughly\b|\bapproximately\b|~|≈)\s*$", approximation_context):
        tolerance = max(tolerance, abs(float(expected or 0.0)) * 0.05)
    return tolerance


def _compare_claim(number_match, expected, key, text):
    raw = number_match.group("number")
    observed = float(raw.replace(",", ""))
    if not math.isfinite(observed) or not isinstance(expected, (int, float)) or not math.isfinite(float(expected)):
        return "unmapped", None, "non_finite_or_non_numeric_comparison"
    if key.startswith("max_abs_"):
        observed = abs(observed)
    if key in {"history_length", "prediction_length", "max_actual_step_number", "max_prediction_step_number"}:
        tolerance = 0.0
    else:
        tolerance = _precision_tolerance(raw, expected, text, number_match.start("number"))
        prefix = text[max(0, number_match.start("number") - 14):number_match.start("number")].lower()
        signed_context = text[max(0, number_match.start("number") - 48):number_match.start("number")].lower()
        if key in _SIGNED_FACTS and float(expected) < 0 and observed >= 0:
            if re.search(r"\b(?:negative|declin\w*|fell|fall\w*|drop\w*|decreas\w*|downward|down)\b", signed_context):
                observed = -abs(observed)
            elif raw.startswith("+") or re.search(r"\b(?:positive|increas\w*|ris\w*|rose|upward|up)\b", signed_context):
                observed = abs(observed)
            else:
                return "unmapped", None, "unsigned_number_does_not_state_negative_sign"
        elif observed >= 0 and re.search(r"\bnegative\s*$", prefix):
            observed = -observed
        elif observed >= 0 and re.search(r"\bpositive\s*$", prefix):
            observed = abs(observed)
    matched = abs(observed - float(expected)) <= tolerance
    return ("verified" if matched else "mismatch"), tolerance, None


def audit_reflection_numeric_claims(reflection, verified_facts, *, history=None, prediction=None, truth=None):
    """Check explicitly labeled numeric claims without treating ambiguity as error.

    `history`, `prediction`, and `truth` are accepted for caller traceability and
    future extensions; this first pass only compares against immutable facts and
    explicit history/horizon lengths. The function is deterministic and offline.
    """
    del history, prediction, truth
    facts = verified_facts if isinstance(verified_facts, dict) else {}
    claims = []
    ignored_enumerators = 0
    for field, text in _reflection_texts(reflection):
        for match in _NUMBER_RE.finditer(text):
            raw = match.group("number")
            if _is_list_enumerator(text, match.start("number"), match.end("number"), raw):
                ignored_enumerators += 1
                continue
            if _is_numeric_range(text, match):
                claims.append({
                    "field": field,
                    "text": text,
                    "raw_number": match.group(0),
                    "context": text[max(0, match.start() - 72):min(len(text), match.end() + 72)].strip(),
                    "fact_key": None,
                    "status": "unmapped",
                    "reason": "numeric_range_requires_manual_interpretation",
                })
                continue
            if _is_abbreviated_magnitude(text, match):
                claims.append({
                    "field": field,
                    "text": text,
                    "raw_number": match.group(0),
                    "context": text[max(0, match.start() - 72):min(len(text), match.end() + 72)].strip(),
                    "fact_key": None,
                    "status": "unmapped",
                    "reason": "abbreviated_magnitude_requires_unit_expansion",
                })
                continue
            fact_key = _explicit_fact_key_claim(text, match, facts)
            if fact_key is None:
                fact_key = _step_index_claim(text, match, facts)
            if fact_key is None:
                fact_key = _shape_claim(text, match, facts)
            reason = None
            if fact_key is None:
                fact_key, reason = _nearest_fact_anchor(text, match, facts)
            context = text[max(0, match.start() - 72):min(len(text), match.end() + 72)].strip()
            claim = {
                "field": field,
                "text": text,
                "raw_number": match.group(0),
                "context": context,
                "fact_key": fact_key,
            }
            if fact_key is None:
                claim.update({"status": "unmapped", "reason": reason})
            else:
                status, tolerance, compare_reason = _compare_claim(
                    match, facts.get(fact_key), fact_key, text
                )
                claim.update({
                    "status": status,
                    "expected": facts.get(fact_key),
                    "tolerance": tolerance,
                    "reason": compare_reason,
                })
            claims.append(claim)

    counts = {"verified": 0, "mismatch": 0, "unmapped": 0, "ignored_enumerators": ignored_enumerators}
    for claim in claims:
        counts[claim["status"]] += 1
    return {"counts": counts, "claims": claims}
