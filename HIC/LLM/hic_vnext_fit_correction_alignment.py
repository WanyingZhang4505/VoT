"""Recheck fit-only HIC correction effects against source fit truths."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from LLM.hic_grid_runner import _hash_json


_EPSILON = 1e-10


def _finite_vector(values, *, field: str, length: int) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{field} must contain only numeric values")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{field} must contain finite values")
    return vector


def _scores(prediction: list[float], truth: list[float]) -> dict[str, float]:
    errors = [pred - actual for pred, actual in zip(prediction, truth, strict=True)]
    mse = sum(error * error for error in errors) / len(errors)
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": mse,
    }


def _check_saved_score(saved: dict, recomputed: dict, *, field: str) -> None:
    if not isinstance(saved, dict):
        raise ValueError(f"{field} is missing")
    for metric, value in recomputed.items():
        stored = saved.get(metric)
        if isinstance(stored, bool) or not isinstance(stored, (int, float)):
            raise ValueError(f"{field}.{metric} must be numeric")
        if not math.isfinite(float(stored)) or not math.isclose(
            float(stored), value, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise ValueError(f"{field}.{metric} does not match recomputed fit score")


def summarize_fit_correction_alignment(
    checkpoints,
    fit_cases_by_config,
    source_sha256_by_config,
    *,
    variant_key: str,
):
    """Describe how a saved bounded correction affected each fit forecast.

    Only source cases explicitly marked ``split == 'fit'`` are accepted. The
    checkpoint must explicitly state that validation truth and outer-test
    targets were not read. Source fit truths are hash-checked against the
    post-forecast fit-truth commitment before any score is recomputed.
    """
    if not isinstance(checkpoints, (list, tuple)):
        raise ValueError("checkpoints must be a list or tuple")
    if not isinstance(fit_cases_by_config, dict):
        raise ValueError("fit_cases_by_config must be a mapping")
    if not isinstance(source_sha256_by_config, dict):
        raise ValueError("source_sha256_by_config must be a mapping")
    if not isinstance(variant_key, str) or not variant_key.strip():
        raise ValueError("variant_key must be a non-empty string")

    configurations = {}
    completed = 0
    truth_hashes_verified = 0
    source_hashes_verified = 0
    scores_recomputed = 0
    seen_configs = set()

    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("each checkpoint must be a mapping")
        if checkpoint.get("validation_truth_read") is not False:
            raise ValueError("refusing checkpoint without validation_truth_read=false")
        if checkpoint.get("outer_test_targets_read") is not False:
            raise ValueError("refusing checkpoint without outer_test_targets_read=false")
        dataset = checkpoint.get("dataset")
        horizon = checkpoint.get("horizon")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("checkpoint dataset must be a non-empty string")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("checkpoint horizon must be a positive integer")
        config_key = (dataset, horizon)
        if config_key in seen_configs:
            raise ValueError(f"duplicate checkpoint configuration: {dataset}/H{horizon}")
        seen_configs.add(config_key)

        fit_cases = fit_cases_by_config.get(config_key)
        if not isinstance(fit_cases, dict):
            raise ValueError(f"missing fit source cases for {dataset}/H{horizon}")
        source_sha256 = source_sha256_by_config.get(config_key)
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError(f"missing source SHA-256 for {dataset}/H{horizon}")
        queries = checkpoint.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"checkpoint {dataset}/H{horizon} has no query mapping")

        row = {
            "query_count": 0,
            "query_outcomes": Counter(),
            "fallback_counts": Counter(),
            "pointwise_effects": Counter(),
        }
        for query_id, query in queries.items():
            if not isinstance(query, dict):
                raise ValueError(f"query {query_id} must be a mapping")
            if query.get("status") != "complete":
                continue
            case_id = query.get("memory", {}).get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"complete query {query_id} has no memory.case_id")
            case = fit_cases.get(case_id)
            if not isinstance(case, dict) or case.get("split") != "fit":
                raise ValueError(f"source case {case_id} is missing or is not fit-only")
            metadata = query.get("memory", {}).get("metadata")
            if (
                not isinstance(metadata, dict)
                or metadata.get("source_sha256") != source_sha256
            ):
                raise ValueError(f"fit source hash mismatch for {dataset}/H{horizon}/{case_id}")
            source_hashes_verified += 1
            truth = _finite_vector(case.get("truth"), field=f"{case_id}.fit_truth", length=horizon)
            if query.get("fit_truth_sha256") != _hash_json(truth):
                raise ValueError(f"fit truth hash mismatch for {dataset}/H{horizon}/{case_id}")
            truth_hashes_verified += 1

            forecast = query.get("forecast")
            scores = query.get("scores")
            if not isinstance(forecast, dict) or not isinstance(scores, dict):
                raise ValueError(f"complete query {query_id} is missing forecast or scores")
            baseline_record = forecast.get("baseline")
            baseline_scores = scores.get("baseline")
            variants = scores.get("variants")
            if not isinstance(baseline_record, dict) or not isinstance(variants, dict):
                raise ValueError(f"complete query {query_id} is missing baseline or variants")
            baseline = _finite_vector(
                baseline_record.get("prediction"), field=f"{query_id}.baseline", length=horizon
            )
            variant = variants.get(variant_key)
            if not isinstance(variant, dict):
                raise ValueError(f"complete query {query_id} is missing variant {variant_key}")
            prediction = _finite_vector(
                variant.get("prediction"), field=f"{query_id}.{variant_key}.prediction", length=horizon
            )
            delta = _finite_vector(
                variant.get("applied_delta"), field=f"{query_id}.{variant_key}.applied_delta", length=horizon
            )
            for step, (base_value, predicted_value, step_delta) in enumerate(
                zip(baseline, prediction, delta, strict=True), start=1
            ):
                if not math.isclose(
                    predicted_value, base_value + step_delta, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"{query_id}.{variant_key} prediction is not baseline plus applied_delta at step {step}"
                    )

            recomputed_baseline = _scores(baseline, truth)
            recomputed_variant = _scores(prediction, truth)
            _check_saved_score(baseline_scores, recomputed_baseline, field=f"{query_id}.baseline")
            _check_saved_score(variant, recomputed_variant, field=f"{query_id}.{variant_key}")
            scores_recomputed += 1
            completed += 1
            row["query_count"] += 1

            baseline_mae = recomputed_baseline["mae"]
            candidate_mae = recomputed_variant["mae"]
            if candidate_mae < baseline_mae - _EPSILON:
                row["query_outcomes"]["wins"] += 1
            elif candidate_mae > baseline_mae + _EPSILON:
                row["query_outcomes"]["losses"] += 1
            else:
                row["query_outcomes"]["ties"] += 1

            fallback = variant.get("fallback_reason")
            if fallback is not None and not isinstance(fallback, str):
                raise ValueError(f"{query_id}.{variant_key}.fallback_reason is invalid")
            row["fallback_counts"][fallback or "applied"] += 1

            for base_value, actual, step_delta, predicted in zip(
                baseline, truth, delta, prediction, strict=True
            ):
                before_error = base_value - actual
                after_error = predicted - actual
                before_abs = abs(before_error)
                after_abs = abs(after_error)
                if after_abs < before_abs - _EPSILON:
                    row["pointwise_effects"]["helped"] += 1
                elif after_abs > before_abs + _EPSILON:
                    row["pointwise_effects"]["worsened"] += 1
                else:
                    row["pointwise_effects"]["ties"] += 1

                if abs(step_delta) <= _EPSILON:
                    continue
                if abs(before_error) <= _EPSILON:
                    row["pointwise_effects"]["baseline_error_zero"] += 1
                elif before_error * step_delta < -_EPSILON:
                    row["pointwise_effects"]["correction_toward_truth"] += 1
                    if after_abs > before_abs + _EPSILON:
                        row["pointwise_effects"]["overshoots"] += 1
                elif before_error * step_delta > _EPSILON:
                    row["pointwise_effects"]["correction_away_from_truth"] += 1

        name = f"{dataset}/H{horizon}"
        configurations[name] = {
            "query_count": row["query_count"],
            "query_outcomes": {
                outcome: row["query_outcomes"][outcome]
                for outcome in ("wins", "losses", "ties")
            },
            "fallback_counts": dict(sorted(row["fallback_counts"].items())),
            "pointwise_effects": {
                effect: row["pointwise_effects"][effect]
                for effect in (
                    "helped", "worsened", "ties", "correction_toward_truth",
                    "correction_away_from_truth", "baseline_error_zero", "overshoots",
                )
            },
        }

    return {
        "protocol": "hic_vnext_fit_correction_alignment_v1",
        "split": "fit_only",
        "variant_key": variant_key,
        "parameter_selection_performed": False,
        "observations": {
            "completed_fit_queries": completed,
            "truth_hashes_verified": truth_hashes_verified,
            "source_hashes_verified": source_hashes_verified,
            "baseline_and_candidate_scores_recomputed": scores_recomputed,
        },
        "configurations": dict(sorted(configurations.items())),
    }
