"""Offline retrieval-ranking and bounded-correction primitives for HIC vNext."""

from __future__ import annotations

import math
import json
from statistics import fmean, pstdev


def _finite_vector(values, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite numbers") from None
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def curve_shape_similarity(left, right) -> float:
    """Compare standardized trajectory shape, ignoring level and positive scale.

    Returns a score in [0, 1]. Constant curves match only other constant curves;
    their absolute level is intentionally ignored, just as it is for nonconstant
    curves after standardization.
    """
    first = _finite_vector(left, "left history_series")
    second = _finite_vector(right, "right history_series")
    if len(first) != len(second):
        raise ValueError("history_series must have the same length")

    first_sd = pstdev(first)
    second_sd = pstdev(second)
    if first_sd == 0 or second_sd == 0:
        return 1.0 if first_sd == second_sd else 0.0

    first_mean = fmean(first)
    second_mean = fmean(second)
    squared_error = sum(
        (((a - first_mean) / first_sd) - ((b - second_mean) / second_sd)) ** 2
        for a, b in zip(first, second)
    ) / len(first)
    standardized_rmse = math.sqrt(squared_error)
    return max(0.0, min(1.0, 1.0 - standardized_rmse / 2.0))


def _resample_curve(values, target_length):
    """Linearly align trajectory shapes with different observed-history lengths."""
    curve = _finite_vector(values, "history_series")
    if type(target_length) is not int or target_length < 2:
        raise ValueError("target_length must be an integer of at least two")
    if len(curve) == 1:
        return [curve[0]] * target_length
    last_source_index = len(curve) - 1
    result = []
    for target_index in range(target_length):
        source_position = target_index * last_source_index / (target_length - 1)
        left_index = int(source_position)
        right_index = min(left_index + 1, last_source_index)
        fraction = source_position - left_index
        result.append(
            curve[left_index] * (1.0 - fraction) + curve[right_index] * fraction
        )
    return result


def rerank_by_text_then_curve(candidates, query_series, *, text_top_k=5):
    """Take the text-similar shortlist, then rank it by equal text/curve scores.

    Qdrant cosine scores are mapped from [-1, 1] to [0, 1] before combining.
    Each candidate must carry its pre-outcome history in ``history_series``;
    this function never reads labels, predictions, or correction outcomes.
    """
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    if type(text_top_k) is not int or text_top_k < 1:
        raise ValueError("text_top_k must be a positive integer")
    query = _finite_vector(query_series, "query_series")

    validated = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be a mapping")
        case_id = candidate.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each candidate must include a non-empty case_id")
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("each candidate must include a finite cosine score")
        history = _finite_vector(candidate.get("history_series"), "history_series")
        if len(history) != len(query):
            raise ValueError("query and candidate history_series must have the same length")
        if not -1.000001 <= float(score) <= 1.000001:
            raise ValueError("cosine score must be in [-1, 1]")
        validated.append({**candidate, "score": float(score), "history_series": history})

    text_shortlist = sorted(validated, key=lambda item: (-item["score"], item["case_id"]))[:text_top_k]
    ranked = []
    for candidate in text_shortlist:
        text_similarity = max(0.0, min(1.0, (candidate["score"] + 1.0) / 2.0))
        curve_similarity = curve_shape_similarity(query, candidate["history_series"])
        ranked.append({
            **candidate,
            "text_similarity": text_similarity,
            "curve_similarity": curve_similarity,
            "combined_score": 0.5 * text_similarity + 0.5 * curve_similarity,
        })
    return sorted(ranked, key=lambda item: (-item["combined_score"], item["case_id"]))


_FORECAST_BEHAVIOR_FIELDS = (
    "history_trend",
    "baseline_trend",
    "baseline_level_offset",
    "baseline_initial_jump",
    "baseline_volatility",
)


def _linear_slope(values):
    center = (len(values) - 1) / 2.0
    mean_value = fmean(values)
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return sum(
        (index - center) * (value - mean_value)
        for index, value in enumerate(values)
    ) / denominator


def build_forecast_behavior_profile(history, baseline_prediction):
    """Create a truth-free, scale-normalized profile of history and baseline shape."""
    observed = _finite_vector(history, "history")
    baseline = _finite_vector(baseline_prediction, "baseline_prediction")
    if len(observed) < 2:
        raise ValueError("history must contain at least two observations")

    history_scale = pstdev(observed)
    if history_scale <= 1e-12:
        history_scale = max(abs(fmean(observed)), 1.0)
    baseline_steps = [right - left for left, right in zip(baseline, baseline[1:])]
    baseline_volatility = pstdev(baseline_steps) if baseline_steps else 0.0
    return {
        "history_trend": _linear_slope(observed) / history_scale,
        "baseline_trend": _linear_slope(baseline) / history_scale if len(baseline) > 1 else 0.0,
        "baseline_level_offset": (fmean(baseline) - observed[-1]) / history_scale,
        "baseline_initial_jump": (baseline[0] - observed[-1]) / history_scale,
        "baseline_volatility": baseline_volatility / history_scale,
    }


def forecast_behavior_similarity(left, right) -> float:
    """Compare pre-outcome history/baseline profiles; no truth or score is accepted."""
    profiles = []
    for profile, label in ((left, "left baseline_behavior_profile"), (right, "right baseline_behavior_profile")):
        if not isinstance(profile, dict) or set(profile) != set(_FORECAST_BEHAVIOR_FIELDS):
            raise ValueError(f"{label} must contain the supported profile fields")
        values = []
        for field in _FORECAST_BEHAVIOR_FIELDS:
            value = profile.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label}.{field} must be finite")
            values.append(float(value))
        profiles.append(values)
    mean_absolute_distance = sum(
        abs(a - b) for a, b in zip(profiles[0], profiles[1])
    ) / len(_FORECAST_BEHAVIOR_FIELDS)
    return math.exp(-mean_absolute_distance)


def rerank_by_text_curve_and_behavior(
    candidates, query_history, query_baseline, *, text_top_k=5,
    allow_different_history_lengths=False,
):
    """Text shortlist then rank by history shape and truth-free baseline behavior."""
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    if type(text_top_k) is not int or text_top_k < 1:
        raise ValueError("text_top_k must be a positive integer")
    if type(allow_different_history_lengths) is not bool:
        raise ValueError("allow_different_history_lengths must be a boolean")
    history = _finite_vector(query_history, "query_history")
    query_profile = build_forecast_behavior_profile(history, query_baseline)
    validated = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be a mapping")
        case_id = candidate.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each candidate must include a non-empty case_id")
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("each candidate must include a finite cosine score")
        if not -1.000001 <= float(score) <= 1.000001:
            raise ValueError("cosine score must be in [-1, 1]")
        candidate_history = _finite_vector(candidate.get("history_series"), "history_series")
        if len(candidate_history) != len(history) and not allow_different_history_lengths:
            raise ValueError("query and candidate history_series must have the same length")
        profile = candidate.get("baseline_behavior_profile")
        if profile is not None:
            if not isinstance(profile, dict):
                raise ValueError("baseline_behavior_profile must be a mapping when present")
            # Validate the persisted profile before it can affect ranking.
            forecast_behavior_similarity(query_profile, profile)
        validated_candidate = {
            **candidate,
            "score": float(score),
            "history_series": candidate_history,
        }
        if profile is not None:
            validated_candidate["baseline_behavior_profile"] = profile
        validated.append(validated_candidate)

    shortlist = sorted(validated, key=lambda item: (-item["score"], item["case_id"]))[:text_top_k]
    ranked = []
    for candidate in shortlist:
        text_similarity = max(0.0, min(1.0, (candidate["score"] + 1.0) / 2.0))
        if len(history) == len(candidate["history_series"]):
            history_similarity = curve_shape_similarity(
                history, candidate["history_series"]
            )
        else:
            aligned_length = max(32, len(history), len(candidate["history_series"]))
            history_similarity = curve_shape_similarity(
                _resample_curve(history, aligned_length),
                _resample_curve(candidate["history_series"], aligned_length),
            )
        behavior_profile = candidate.get("baseline_behavior_profile")
        behavior_similarity = (
            forecast_behavior_similarity(query_profile, behavior_profile)
            if behavior_profile is not None else None
        )
        if behavior_similarity is None:
            # Test 3's frozen 575-point payloads predate this derived profile.
            # Rank on the two stored, truth-free signals and renormalize their
            # weights; never synthesize a historical forecast from a hash.
            combined_score = (
                0.4 * text_similarity + 0.3 * history_similarity
            ) / 0.7
            selection_reason = (
                "text_shortlist_then_curve_only_missing_historical_baseline_profile"
            )
        else:
            combined_score = (
                0.4 * text_similarity
                + 0.3 * history_similarity
                + 0.3 * behavior_similarity
            )
            selection_reason = "text_shortlist_then_curve_and_forecast_behavior"
        ranked.append({
            **candidate,
            "text_similarity": text_similarity,
            "curve_similarity": history_similarity,
            "behavior_similarity": behavior_similarity,
            "query_behavior_profile": query_profile,
            "combined_score": combined_score,
            "selection_reason": selection_reason,
        })
    return sorted(ranked, key=lambda item: (-item["combined_score"], item["case_id"]))


def apply_bounded_residual(
    baseline,
    residual,
    history,
    *,
    alpha,
    bound_scale,
    eligible=True,
    abstain=False,
):
    """Apply a scaled residual, clipped by recent-history population volatility.

    This is a deterministic post-processor: the residual can come from a model,
    but the final bound and exact fallback are enforced in local code.
    """
    base = _finite_vector(baseline, "baseline")
    delta = _finite_vector(residual, "residual")
    observed = _finite_vector(history, "history")
    if len(base) != len(delta):
        raise ValueError("baseline and residual must have the same length")
    if len(observed) < 2:
        raise ValueError("history must contain at least two values")
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    if (
        not isinstance(bound_scale, (int, float))
        or not math.isfinite(float(bound_scale))
        or bound_scale < 0
    ):
        raise ValueError("bound_scale must be a finite non-negative number")
    if type(eligible) is not bool or type(abstain) is not bool:
        raise ValueError("eligible and abstain must be booleans")

    fallback_reason = None
    if not eligible:
        fallback_reason = "no_eligible_memory"
    elif abstain:
        fallback_reason = "model_abstained"
    else:
        cap = float(bound_scale) * pstdev(observed)
        if cap == 0:
            fallback_reason = "zero_history_volatility"

    if fallback_reason is not None:
        applied_delta = [0.0] * len(base)
        prediction = base[:]
    else:
        cap = float(bound_scale) * pstdev(observed)
        applied_delta = [
            max(-cap, min(cap, float(alpha) * proposed)) for proposed in delta
        ]
        prediction = [value + adjustment for value, adjustment in zip(base, applied_delta)]

    return {
        "prediction": prediction,
        "applied_delta": applied_delta,
        "bound": 0.0 if fallback_reason else float(bound_scale) * pstdev(observed),
        "fallback_reason": fallback_reason,
    }


def build_bounded_residual_prompt(
    summary,
    history,
    baseline_prediction,
    retrieved_memory,
    *,
    horizon,
    frequency,
):
    """Ask the model for a residual only, using forecast-time information."""
    observed = _finite_vector(history, "history")
    baseline = _finite_vector(baseline_prediction, "baseline_prediction")
    if type(horizon) is not int or horizon <= 0 or len(baseline) != horizon:
        raise ValueError("baseline_prediction length must equal the positive horizon")
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError("frequency must be a non-empty string")
    if isinstance(retrieved_memory, dict):
        memories = [retrieved_memory]
    elif isinstance(retrieved_memory, list) and retrieved_memory:
        memories = retrieved_memory
    else:
        raise ValueError("retrieved_memory must be a memory mapping or a non-empty list")
    memory_view = []
    for memory in memories:
        if not isinstance(memory, dict):
            raise ValueError("each retrieved memory must be a mapping")
        memory_view.append({
            key: memory[key]
            for key in ("case_id", "summary", "correction", "score", "text_similarity", "curve_similarity")
            if key in memory
        })
    inputs = {
        "frequency": frequency,
        "horizon": horizon,
        "current_pre_target_summary": summary,
        "current_pre_target_history": observed,
        "no_retrieval_baseline_prediction": baseline,
        "retrieved_historical_experience": memory_view,
    }
    encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"""HIC_VNEXT_RESIDUAL_OUTPUT
You are a conservative residual-correction component. The No Retrieval forecast is
the baseline and must remain the prediction unless the retrieved experience gives
a concrete, applicable reason to adjust it. Do not generate a new absolute forecast.

The input contains only information available at the forecast cutoff. The historical
experience may be irrelevant; if so, set abstain to true and return zero residuals.
Return exactly {horizon} finite numeric residuals, one per future step, where each
residual is an additive change to the corresponding baseline value. Do not use or
invent future observations or ground truth. Keep the reasoning brief and state why
the retrieved experience does or does not apply.

FORECAST-TIME INPUT:
{encoded}

Return only this JSON schema:
{{"residual":[{', '.join('0' for _ in range(horizon))}],"reasoning":"...","abstain":false}}
"""


def build_paired_direct_forecast_prompt(
    summary,
    history,
    *,
    horizon,
    frequency,
    target,
    unit,
    covariates=None,
    dataset_description=None,
    temporal_context=None,
    retrieved_memory=None,
):
    """Build A/B prompts with an identical forecast-time input object.

    Arm A omits historical experience. Arm B adds exactly one retrieved
    experience; neither arm receives future truth or a pre-existing forecast.
    """
    observed = _finite_vector(history, "history")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    for name, value in (("frequency", frequency), ("target", target), ("unit", unit)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    if not isinstance(summary, (dict, list, str)):
        raise ValueError("summary must be JSON-compatible content")

    inputs = {
        "task_contract": {
            "target": " ".join(target.split()),
            "unit": " ".join(unit.split()),
            "frequency": " ".join(frequency.split()).lower(),
            "horizon": horizon,
        },
        "current_pre_target_summary": summary,
        "current_pre_target_history": observed,
    }
    if covariates is not None:
        inputs["current_pre_target_covariates"] = covariates
    if dataset_description is not None:
        if not isinstance(dataset_description, str):
            raise ValueError("dataset_description must be text when provided")
        inputs["dataset_context"] = dataset_description
    if temporal_context is not None:
        if not isinstance(temporal_context, dict):
            raise ValueError("temporal_context must be a mapping when provided")
        required_periods = {"history_period", "forecast_period"}
        if set(temporal_context) != required_periods:
            raise ValueError("temporal_context must contain history_period and forecast_period")
        normalized_periods = {}
        for period_name, period in temporal_context.items():
            if not isinstance(period, dict) or set(period) != {"start", "end"}:
                raise ValueError(f"{period_name} must contain exactly start and end")
            if any(not isinstance(value, str) or not value.strip() for value in period.values()):
                raise ValueError(f"{period_name} start and end must be non-empty text")
            normalized_periods[period_name] = {
                key: " ".join(value.split()) for key, value in period.items()
            }
        inputs["temporal_context"] = normalized_periods

    if retrieved_memory is not None:
        if not isinstance(retrieved_memory, dict):
            raise ValueError("retrieved_memory must be one memory mapping")
        case_id = retrieved_memory.get("case_id")
        correction = retrieved_memory.get("correction")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("retrieved_memory must include a non-empty case_id")
        if not isinstance(correction, str) or not correction.strip():
            raise ValueError("retrieved_memory must include non-empty correction guidance")
        memory_view = {"case_id": case_id, "correction": correction.strip()}
        source_horizon = retrieved_memory.get("source_horizon")
        if source_horizon is not None:
            if type(source_horizon) is not int or source_horizon <= 0:
                raise ValueError("retrieved_memory source_horizon must be a positive integer")
            memory_view["source_horizon"] = source_horizon
        if retrieved_memory.get("summary") is not None:
            memory_view["summary"] = retrieved_memory["summary"]
        source_context = retrieved_memory.get("source_context")
        if source_context is not None:
            if not isinstance(source_context, dict):
                raise ValueError("retrieved_memory source_context must be an object")
            memory_view["source_context"] = source_context
        inputs["retrieved_historical_experience"] = memory_view

    encoded = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    output_schema = json.dumps(
        {"prediction": [0] * horizon, "reasoning": "..."},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if retrieved_memory is None:
        protocol = "HIC_PAIRED_NO_RETRIEVAL_FORECAST"
        instructions = (
            "No historical experience is provided. Forecast from this case's visible "
            "pre-target inputs only."
        )
    else:
        protocol = "HIC_PAIRED_DIRECT_HIC_FORECAST"
        instructions = (
            "The retrieved historical experience must remain in the prompt as context. "
            "Its source_context and source_horizon preserve the original dataset, target, "
            "unit, frequency, and forecast horizon; these may differ from the current task. "
            "Treat its correction as a qualitative analogy. Compare its stated applicability "
            "and exclusions with the current visible inputs, and adapt only a relevant lesson. "
            "Do not assume cross-domain measurement compatibility or copy historical numeric "
            "forecasts, error magnitudes, or error directions. Do not claim causality. Do not "
            "decide whether to retrieve or omit it, and do not produce an abstain decision. "
            "Generate a fresh forecast from the current case."
        )
    return f"""{protocol}
You are the forecasting Reasoner in a historical in-context learning pipeline.
{instructions}

Use only information available before the target period. Do not infer, request, or
invent future observations or ground truth. Return exactly {horizon} finite numeric
predictions and concise reasoning in this JSON schema:
{output_schema}

FORECAST-TIME INPUT:
{encoded}
"""


def build_hic_in_context_prediction_prompt(
    summary,
    history,
    retrieved_memory,
    *,
    horizon,
    frequency,
    covariates=None,
    dataset_description=None,
):
    """Build the paper-style HIC prompt: retrieved correction is forecast context.

    Unlike the residual-correction experiment, this prompt asks the reasoner to
    produce a fresh forecast with the retrieved corrected example in context.
    There is no separate use/abstain gate and no additive residual contract.
    """
    observed = _finite_vector(history, "history")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError("frequency must be a non-empty string")
    if not isinstance(summary, (dict, list, str)):
        raise ValueError("summary must be JSON-compatible content")
    if not isinstance(retrieved_memory, dict):
        raise ValueError("retrieved_memory must be one historical memory mapping")
    case_id = retrieved_memory.get("case_id")
    correction = retrieved_memory.get("correction")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("retrieved_memory must include a case_id")
    if not isinstance(correction, str) or not correction.strip():
        raise ValueError("retrieved_memory must include correction guidance")
    memory_view = {
        "case_id": case_id,
        "summary": retrieved_memory.get("summary"),
        "correction": correction,
        "similarity": retrieved_memory.get("score"),
    }
    inputs = {
        "frequency": frequency,
        "prediction_horizon": horizon,
        "current_pre_target_summary": summary,
        "current_pre_target_history": observed,
        "retrieved_historical_corrected_example": memory_view,
    }
    if covariates:
        inputs["current_pre_target_covariates"] = covariates
    if dataset_description:
        inputs["dataset_context"] = dataset_description
    encoded = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    output_schema = json.dumps(
        {"prediction": [0] * horizon, "reasoning": "..."},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""HIC_RETRIEVAL_GUIDED_FORECAST
You are the forecasting Reasoner in a Historical In-Context Learning pipeline.
Generate a fresh forecast for the current case. The retrieved historical corrected
example is already selected and MUST be used as in-context guidance; do not decide
whether to retrieve it or whether to include it. Use its lesson as an example, adapt
it to the current evidence, and do not copy its old numeric forecast or claim that
it proves causality. The retrieved correction is prompt context, not a numeric
residual to add to another forecast. Do not produce an abstain decision.

Only use information available before the current target period. Do not infer,
request, or invent future observations or ground truth. Return exactly {horizon}
finite numeric predictions and concise reasoning in this JSON schema:
{output_schema}

FORECAST-TIME INPUT:
{encoded}
"""


def validate_bounded_residual_response(response, *, horizon):
    """Validate the model's additive residual contract before local clipping."""
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(response, dict) or set(response) != {"residual", "reasoning", "abstain"}:
        raise ValueError("residual response must contain exactly residual, reasoning, and abstain")
    residual = _finite_vector(response["residual"], "residual")
    if len(residual) != horizon:
        raise ValueError("residual length must equal horizon")
    reasoning = response["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be a non-empty string")
    if type(response["abstain"]) is not bool:
        raise ValueError("abstain must be a boolean")
    return {
        "residual": residual,
        "reasoning": " ".join(reasoning.split()),
        "abstain": response["abstain"],
    }
