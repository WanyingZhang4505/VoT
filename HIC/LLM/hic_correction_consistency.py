"""Narrow consistency checks for model-written HIC correction metadata."""

import re


_SINGLE_STEP_CLAIM = re.compile(
    r"\b(?:one|single|1)[\s-]+(?:(?:next)[\s-]+)?"
    r"step(?:[\s-]+ahead)?[\s-]+(?:forecast[\s-]+)?horizon\b",
    re.IGNORECASE,
)
_NUMERIC_STEP_CLAIM = re.compile(
    r"\b(?P<horizon>\d+)[\s-]+step"
    r"(?:[\s-]+ahead)?[\s-]+(?:forecast[\s-]+)?horizon\b",
    re.IGNORECASE,
)


def find_horizon_claim_mismatches(conditions, *, expected_horizon):
    """Flag explicit English horizon claims conflicting with verified facts.

    This lexical check is intentionally narrow and is not a semantic verifier.
    It scans applicability conditions, not exclusions, because exclusions may
    legitimately describe a different horizon.
    """
    if type(expected_horizon) is not int or expected_horizon <= 0:
        raise ValueError("expected_horizon must be a positive integer")
    if not isinstance(conditions, list) or any(not isinstance(item, str) for item in conditions):
        raise ValueError("conditions must be a list of strings")
    mismatches = []
    for index, condition in enumerate(conditions):
        claims = []
        if expected_horizon != 1:
            claims.extend((match, 1, "single-step claim") for match in _SINGLE_STEP_CLAIM.finditer(condition))
        claims.extend(
            (match, int(match.group("horizon")), "explicit numeric horizon claim")
            for match in _NUMERIC_STEP_CLAIM.finditer(condition)
        )
        for match, claimed_horizon, rule in claims:
            if claimed_horizon != expected_horizon:
                mismatches.append({
                    "condition_index": index,
                    "condition": condition,
                    "claimed_horizon": claimed_horizon,
                    "expected_horizon": expected_horizon,
                    "rule": rule,
                })
    return mismatches
