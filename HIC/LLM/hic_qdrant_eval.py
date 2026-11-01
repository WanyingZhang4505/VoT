"""Leakage-safe prequential A/B evaluation for Qdrant-backed HIC memory."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from LLM import hic


ARM_SIZES = (0, 1, 10, 100)
MAX_LOGICAL_CALLS = 104
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def aggregate_arm_metrics(results_by_arm):
    """Aggregate equal/unequal target windows and count paired MAE wins vs K=0."""
    if 0 not in results_by_arm:
        raise ValueError("results_by_arm must include the K=0 control")
    if any(type(arm) is not int or arm < 0 for arm in results_by_arm):
        raise ValueError("arm keys must be non-negative integers")
    arms = sorted(results_by_arm)
    if len(arms) < 2:
        raise ValueError("results_by_arm must include at least one retrieval arm")
    control = results_by_arm[0]
    report = {}
    for arm in arms:
        rows = results_by_arm.get(arm)
        if not rows or len(rows) != len(control):
            raise ValueError(f"arm K={arm} must contain one result per control window")
        total_n = sum(int(row["metrics"].get("n", 1)) for row in rows)
        if total_n <= 0:
            raise ValueError("metric rows must contain at least one target point")
        mae = sum(
            float(row["metrics"]["mae"]) * int(row["metrics"].get("n", 1))
            for row in rows
        ) / total_n
        rmse = math.sqrt(
            sum(
                float(row["metrics"]["rmse"]) ** 2
                * int(row["metrics"].get("n", 1))
                for row in rows
            )
            / total_n
        )
        wins = sum(
            float(candidate["metrics"]["mae"])
            < float(baseline["metrics"]["mae"])
            for candidate, baseline in zip(rows, control)
        )
        ties = sum(
            math.isclose(
                float(candidate["metrics"]["mae"]),
                float(baseline["metrics"]["mae"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for candidate, baseline in zip(rows, control)
        )
        report[arm] = {
            "target_points": total_n,
            "aggregate_mae": mae,
            "aggregate_rmse": rmse,
            "windows_better_mae_than_k0": wins,
            "windows_tied_mae_with_k0": ties,
            "windows_total": len(rows),
        }
    return report


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _finite_values(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric list")
    if length is not None and len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} numeric values")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must contain only finite numeric values")
    return normalized


def _finite_twelve_values(values, label):
    """Backward-compatible helper for existing 8→12 HIC cases."""
    return _finite_values(values, label, length=12)


def _validated_forecast(response, horizon=12):
    if not isinstance(response, dict) or "prediction" not in response:
        raise ValueError("DeepSeek forecast response must contain a prediction array")
    prediction = _finite_values(response["prediction"], "prediction", length=horizon)
    reasoning = response.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("forecast reasoning must be a string")
    return {**response, "prediction": prediction, "reasoning": reasoning}


def _correction_text(response):
    if not isinstance(response, dict):
        raise ValueError("DeepSeek correction response must be a JSON object")
    improved_reasoning = response.get("Improved Reasoning")
    key_insights = response.get("Key Insights")
    if (
        isinstance(improved_reasoning, str) and improved_reasoning.strip()
        and isinstance(key_insights, str) and key_insights.strip()
    ):
        try:
            return json.dumps(
                response, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError):
            raise ValueError("DeepSeek correction response must be finite JSON data") from None

    correction = (
        key_insights
        or improved_reasoning
        or response.get("correction")
    )
    if not isinstance(correction, str) or not correction.strip():
        raise ValueError("DeepSeek correction response contains no usable lesson")
    return correction.strip()


def _validate_window_spec(windows):
    if not windows:
        raise ValueError("at least one evaluation window is required")
    identifiers = []
    seen_targets = {}
    observed_history_length = None
    observed_horizon = None
    for window in windows:
        case = window.get("case", {})
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every evaluation window needs a case.case_id")
        if case_id in identifiers:
            raise ValueError(f"duplicate evaluation case ID: {case_id}")
        identifiers.append(case_id)
        truth = _finite_values(window.get("truth"), f"truth for {case_id}")
        series = _finite_values(case.get("series"), f"history for {case_id}")
        if observed_horizon is None:
            observed_horizon = len(truth)
            observed_history_length = len(series)
        if len(truth) != observed_horizon or len(series) != observed_history_length:
            raise ValueError("all evaluation windows must share history and horizon lengths")
        target_steps = window.get("target_timestamps") or window.get("target_months", [])
        if target_steps:
            if len(target_steps) != observed_horizon or len(set(target_steps)) != observed_horizon:
                raise ValueError(f"{case_id} must have {observed_horizon} distinct target steps")
            group_key = (case.get("dataset"), tuple(case.get("group", ())))
            prior_targets = seen_targets.setdefault(group_key, set())
            overlap = prior_targets.intersection(target_steps)
            if overlap:
                raise ValueError(
                    f"evaluation target windows overlap at {sorted(overlap)[0]} for group {group_key}"
                )
            prior_targets.update(target_steps)
        if "truth" in case or "target_values" in case:
            raise ValueError(f"case input {case_id} must not contain held-out truth")
    return observed_history_length, observed_horizon, identifiers


def _validate_windows(windows):
    """Backward-compatible case-ID validator used by prompt A/B experiments."""
    return _validate_window_spec(windows)[2]


def run_m3_warmup(
    *,
    windows,
    template,
    stores,
    embed_summary,
    call_json,
    dataset_name="SocialGood",
    dataset_description=None,
    run_id="offline-validation",
    checkpoint_path=None,
    max_calls=15,
    retry_pending=False,
):
    """Run the chronological warmup cases with the sole active BGE-M3 model."""
    model_ids = ("BAAI/bge-m3",)
    if set(stores) != set(model_ids):
        raise ValueError("M3 warmup needs exactly one BGE-M3 store")
    history_length, horizon, case_ids = _validate_window_spec(windows)
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "case_ids": case_ids,
        "history_length": history_length,
        "horizon": horizon,
        "logical_deepseek_calls": 0,
        "pending_call": None,
        "last_completed_call": None,
        "windows": {},
    }
    path = Path(checkpoint_path) if checkpoint_path else None
    if path and path.exists():
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != 1 or state.get("run_id") != run_id:
            raise ValueError("validation checkpoint schema/run ID does not match")
        if state.get("case_ids") != case_ids:
            raise ValueError("validation checkpoint windows do not match current inputs")
        if state.get("history_length", history_length) != history_length or state.get("horizon", horizon) != horizon:
            raise ValueError("validation checkpoint history/horizon does not match current inputs")
        state.setdefault("history_length", history_length)
        state.setdefault("horizon", horizon)
        if state.get("pending_call"):
            if not retry_pending:
                raise RuntimeError(
                    "checkpoint has an unresolved API call; pass retry_pending=True only "
                    "after reviewing whether that request completed"
                )
            state["pending_call"] = None

    def save():
        if path:
            _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt, commit_response):
        completed = state.get("last_completed_call")
        if completed:
            if completed.get("case_id") != case_id or completed.get("stage") != stage:
                raise RuntimeError("checkpoint contains an unapplied validation API result")
            response = completed["response"]
        else:
            if state.get("pending_call"):
                raise RuntimeError(
                    "checkpoint has an unresolved API call; pass retry_pending=True only "
                    "after reviewing whether that request completed"
                )
            if state.get("logical_deepseek_calls", 0) >= max_calls:
                raise RuntimeError(f"validation API call budget ({max_calls}) exhausted")
            state["logical_deepseek_calls"] += 1
            state["pending_call"] = {"case_id": case_id, "stage": stage}
            save()
            response = call_json(prompt, temperature=0.0)
            if not isinstance(response, dict):
                raise ValueError(f"validation {stage} response must be a JSON object")
            state["pending_call"] = None
            state["last_completed_call"] = {
                "case_id": case_id,
                "stage": stage,
                "response": response,
            }
            save()
        commit_response(response)
        state["last_completed_call"] = None
        save()

    result_windows = []
    for window in windows:
        case = window["case"]
        case_id = case["case_id"]
        window_state = state["windows"].setdefault(
            case_id,
            {"forecasts": {}, "corrections": {}, "ingested_models": []},
        )
        if "summary" not in window_state:
            prompt = hic.build_summary_prompt(
                template,
                case["series"],
                case["news"],
                dataset_description=dataset_description,
                covariates=case.get("covariates"),
            )
            invoke(
                case_id,
                "summary",
                prompt,
                lambda response: window_state.__setitem__("summary", response),
            )
        summary = window_state["summary"]

        for model_id in model_ids:
            if model_id not in window_state.get("embeddings", {}):
                query_vector = embed_summary(model_id, summary, is_query=True)
                passage_vector = embed_summary(model_id, summary, is_query=False)
                if not query_vector or len(query_vector) != len(passage_vector):
                    raise ValueError(f"{model_id} returned invalid query/passage embeddings")
                window_state.setdefault("embeddings", {})[model_id] = {
                    "query": [float(value) for value in query_vector],
                    "passage": [float(value) for value in passage_vector],
                }
                save()

            if model_id in window_state["forecasts"]:
                continue
            query_vector = window_state["embeddings"][model_id]["query"]
            retrieval = stores[model_id].search(query_vector, limit=1)
            if not retrieval or not retrieval.get("correction"):
                raise RuntimeError(f"{model_id} validation collection returned no lesson")
            prompt = hic.build_reasoning_prompt(
                summary,
                case["series"],
                retrieval["correction"],
                covariates=case.get("covariates"),
                dataset_description=dataset_description,
                horizon=horizon,
                frequency=case.get("frequency"),
            )

            def commit_forecast(response):
                forecast = _validated_forecast(response, horizon)
                window_state["forecasts"][model_id] = {
                    "prediction": forecast["prediction"],
                    "reasoning": forecast["reasoning"],
                    "retrieval": retrieval,
                }

            invoke(case_id, f"forecast_{model_id}", prompt, commit_forecast)

        if set(window_state["forecasts"]) != set(model_ids):
            raise RuntimeError(f"BGE-M3 did not forecast {case_id}")

        # Keep this access after the two validation predictions for the window.
        truth = _finite_values(window["truth"], f"truth for {case_id}", length=horizon)
        if "truth" in case or "target_values" in case:
            raise ValueError(f"validation input {case_id} contains held-out targets")
        metrics = {
            model_id: hic.forecast_metrics(
                window_state["forecasts"][model_id]["prediction"], truth
            )
            for model_id in model_ids
        }
        window_state["scoring"] = {"truth": truth, "metrics": metrics}
        save()

        for model_id in model_ids:
            if model_id not in window_state["corrections"]:
                forecast = window_state["forecasts"][model_id]
                prompt = hic.build_correction_prompt(
                    summary,
                    case["series"],
                    truth,
                    forecast["prediction"],
                    forecast["reasoning"],
                    covariates=case.get("covariates"),
                    dataset_description=dataset_description,
                )
                invoke(
                    case_id,
                    f"correction_{model_id}",
                    prompt,
                    lambda response, selected_model=model_id: window_state[
                        "corrections"
                    ].__setitem__(selected_model, _correction_text(response)),
                )
            if model_id not in window_state["ingested_models"]:
                stores[model_id].upsert_completed_case(
                    case_id=case_id,
                    summary=summary,
                    correction=window_state["corrections"][model_id],
                    embedding=window_state["embeddings"][model_id]["passage"],
                    observed_targets=truth,
                    prediction=window_state["forecasts"][model_id]["prediction"],
                    cutoff_date=case["target_end"],
                    source=f"validation:{run_id}",
                )
                window_state["ingested_models"].append(model_id)
                save()

        result_windows.append(
            {
                "case_id": case_id,
                "target_start": case.get("target_start"),
                "target_end": case.get("target_end"),
                "target_months": list(window.get("target_months", [])),
                "target_timestamps": list(window.get("target_timestamps", [])),
                "summary": summary,
                "truth": truth,
                "models": {
                    model_id: {
                        **window_state["forecasts"][model_id],
                        "metrics": metrics[model_id],
                        "correction": window_state["corrections"][model_id],
                        "passage_embedding": window_state["embeddings"][model_id]["passage"],
                    }
                    for model_id in model_ids
                },
            }
        )

    model_metrics = {}
    for model_id in model_ids:
        rows = [window["models"][model_id]["metrics"] for window in result_windows]
        point_count = sum(row["n"] for row in rows)
        model_metrics[model_id] = {
            "target_points": point_count,
            "aggregate_mae": sum(row["mae"] * row["n"] for row in rows) / point_count,
            "aggregate_rmse": math.sqrt(
                sum(row["rmse"] ** 2 * row["n"] for row in rows) / point_count
            ),
        }
    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "history_length": history_length,
        "horizon": horizon,
        "validation_windows": result_windows,
        "model_metrics": model_metrics,
        "embedding_model": model_ids[0],
        "logical_deepseek_calls": state["logical_deepseek_calls"],
    }


def run_prequential_evaluation(
    *,
    windows,
    template,
    stores,
    embed_summary,
    call_json,
    embedding_model="BAAI/bge-m3",
    dataset_name="SocialGood",
    dataset_description=None,
    run_id="offline-test",
    checkpoint_path=None,
    max_calls=MAX_LOGICAL_CALLS,
    retry_pending=False,
):
    """Compare no retrieval with HIC memory, revealing each truth after forecasts.

    `stores` maps each positive retrieval K to a distinct collection seeded with
    that many completed training cases. The K=0 control is deliberately not
    connected to or updated in any store.
    External calls and embeddings are injected so ordering can be tested offline.
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 letters, digits, '_' or '-' characters")
    if not isinstance(max_calls, int) or max_calls < 0:
        raise ValueError("max_calls must be a non-negative integer")
    if not isinstance(stores, dict) or not stores:
        raise ValueError("stores must map at least one positive K to a Qdrant collection")
    memory_sizes = tuple(sorted(stores))
    if any(type(size) is not int or size <= 0 for size in memory_sizes):
        raise ValueError("store keys must be positive integer retrieval sizes")
    arm_sizes = (0, *memory_sizes)
    history_length, horizon, case_ids = _validate_window_spec(windows)

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "case_ids": case_ids,
        "history_length": history_length,
        "horizon": horizon,
        "memory_sizes": list(memory_sizes),
        "logical_deepseek_calls": 0,
        "pending_call": None,
        "last_completed_call": None,
        "windows": {},
    }
    path = Path(checkpoint_path) if checkpoint_path else None
    if path and path.exists():
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != 1 or state.get("run_id") != run_id:
            raise ValueError("checkpoint schema/run ID does not match this evaluation")
        if state.get("case_ids") != case_ids:
            raise ValueError("checkpoint evaluation windows do not match current inputs")
        if state.get("memory_sizes", list(memory_sizes)) != list(memory_sizes):
            raise ValueError("evaluation checkpoint retrieval sizes do not match current stores")
        state.setdefault("memory_sizes", list(memory_sizes))
        if state.get("history_length", history_length) != history_length or state.get("horizon", horizon) != horizon:
            raise ValueError("evaluation checkpoint history/horizon does not match current inputs")
        state.setdefault("history_length", history_length)
        state.setdefault("horizon", horizon)
        if state.get("pending_call"):
            if not retry_pending:
                raise RuntimeError(
                    "checkpoint has an unresolved API call; pass retry_pending=True only "
                    "after reviewing whether that request completed"
                )
            state["pending_call"] = None

    def save():
        if path:
            _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt, commit_response):
        completed = state.get("last_completed_call")
        if completed:
            if completed.get("case_id") != case_id or completed.get("stage") != stage:
                raise RuntimeError("checkpoint contains an unapplied completed API result")
            response = completed["response"]
        else:
            if state.get("pending_call"):
                raise RuntimeError(
                    "checkpoint has an unresolved API call; pass retry_pending=True only "
                    "after reviewing whether that request completed"
                )
            if state.get("logical_deepseek_calls", 0) >= max_calls:
                raise RuntimeError(f"logical DeepSeek call budget ({max_calls}) exhausted")
            state["logical_deepseek_calls"] = state.get("logical_deepseek_calls", 0) + 1
            state["pending_call"] = {"case_id": case_id, "stage": stage}
            save()
            response = call_json(prompt, temperature=0.0)
            if not isinstance(response, dict):
                raise ValueError(f"{stage} response must be a JSON object")
            state["pending_call"] = None
            state["last_completed_call"] = {
                "case_id": case_id,
                "stage": stage,
                "response": response,
            }
            save()
        commit_response(response)
        state["last_completed_call"] = None
        save()
        return response

    results = []
    for window in windows:
        case = window["case"]
        case_id = case["case_id"]
        window_state = state["windows"].setdefault(
            case_id,
            {"forecasts": {}, "corrections": {}, "ingested_arms": []},
        )

        if "summary" not in window_state:
            summary_prompt = hic.build_summary_prompt(
                template,
                case["series"],
                case["news"],
                dataset_description=dataset_description,
                covariates=case.get("covariates"),
            )
            invoke(
                case_id,
                "summary",
                summary_prompt,
                lambda response: window_state.__setitem__("summary", response),
            )
        summary = window_state["summary"]

        if "embeddings" not in window_state:
            query_vector = embed_summary(summary, is_query=True)
            passage_vector = embed_summary(summary, is_query=False)
            if (
                not isinstance(query_vector, (list, tuple))
                or not isinstance(passage_vector, (list, tuple))
                or not query_vector
                or len(query_vector) != len(passage_vector)
            ):
                raise ValueError(f"embedding for {case_id} must return aligned query/passage vectors")
            query_vector = [float(value) for value in query_vector]
            passage_vector = [float(value) for value in passage_vector]
            if not all(math.isfinite(value) for value in query_vector + passage_vector):
                raise ValueError(f"embedding for {case_id} contains non-finite values")
            window_state["embeddings"] = {
                "query": query_vector,
                "passage": passage_vector,
            }
            save()
        query_vector = window_state["embeddings"]["query"]
        passage_vector = window_state["embeddings"]["passage"]

        for arm in arm_sizes:
            arm_key = str(arm)
            if arm_key in window_state["forecasts"]:
                continue
            retrieval = None
            correction = None
            if arm:
                retrieval = stores[arm].search(query_vector, limit=1)
                if not retrieval or not retrieval.get("correction"):
                    raise RuntimeError(f"K={arm} collection returned no HIC correction")
                correction = retrieval["correction"]
            prompt = hic.build_reasoning_prompt(
                summary,
                case["series"],
                correction,
                covariates=case.get("covariates"),
                dataset_description=dataset_description,
                horizon=horizon,
                frequency=case.get("frequency"),
            )
            def commit_forecast(response):
                forecast = _validated_forecast(response, horizon)
                window_state["forecasts"][arm_key] = {
                    "prediction": forecast["prediction"],
                    "reasoning": forecast["reasoning"],
                    "retrieval": retrieval,
                }

            invoke(case_id, f"forecast_k{arm}", prompt, commit_forecast)

        if set(window_state["forecasts"]) != {str(arm) for arm in arm_sizes}:
            raise RuntimeError(f"not all forecasts completed for {case_id}")

        # Held-out truth is first read here, after all four forecast results are saved.
        truth = _finite_values(window["truth"], f"truth for {case_id}", length=horizon)
        if "scoring" not in window_state:
            metrics = {
                str(arm): hic.forecast_metrics(
                    window_state["forecasts"][str(arm)]["prediction"], truth
                )
                for arm in arm_sizes
            }
            persistence = hic.forecast_metrics(
                hic.persistence_forecast(case["series"], len(truth)), truth
            )
            window_state["scoring"] = {
                "truth": truth,
                "arms": metrics,
                "persistence": persistence,
            }
            save()
        elif window_state["scoring"].get("truth") != truth:
            raise ValueError(f"checkpoint truth changed for evaluation case {case_id}")

        for arm in memory_sizes:
            arm_key = str(arm)
            forecast = window_state["forecasts"][arm_key]
            if arm_key not in window_state["corrections"]:
                correction_prompt = hic.build_correction_prompt(
                    summary,
                    case["series"],
                    truth,
                    forecast["prediction"],
                    forecast["reasoning"],
                    covariates=case.get("covariates"),
                    dataset_description=dataset_description,
                )
                invoke(
                    case_id,
                    f"correction_k{arm}",
                    correction_prompt,
                    lambda response: window_state["corrections"].__setitem__(
                        arm_key, _correction_text(response)
                    ),
                )
            if arm not in window_state["ingested_arms"]:
                stores[arm].upsert_completed_case(
                    case_id=case_id,
                    summary=summary,
                    correction=window_state["corrections"][arm_key],
                    embedding=passage_vector,
                    observed_targets=truth,
                    prediction=forecast["prediction"],
                    cutoff_date=case["target_end"],
                    source=f"prequential:{run_id}",
                )
                window_state["ingested_arms"].append(arm)
                save()

        result_window = {
            "case_id": case_id,
            "history_start": case.get("history_start"),
            "history_end": case.get("history_end"),
            "target_start": case.get("target_start"),
            "target_end": case.get("target_end"),
            "target_months": list(window.get("target_months", [])),
            "target_timestamps": list(window.get("target_timestamps", [])),
            "truth": truth,
            "arms": {},
            "persistence_metrics": window_state["scoring"]["persistence"],
        }
        for arm in arm_sizes:
            forecast = window_state["forecasts"][str(arm)]
            result_window["arms"][str(arm)] = {
                **forecast,
                "metrics": window_state["scoring"]["arms"][str(arm)],
            }
        results.append(result_window)

    metrics_by_arm = {
        arm: [window["arms"][str(arm)] for window in results]
        for arm in arm_sizes
    }
    persistence_rows = [
        {"metrics": window["persistence_metrics"]} for window in results
    ]
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": dataset_name,
        "history_length": history_length,
        "horizon": horizon,
        "dataset_description": dataset_description,
        "embedding_model": embedding_model,
        "forecast_model": "deepseek-flash",
        "retrieval_memory_sizes": list(memory_sizes),
        "evaluation_protocol": (
            f"Prequential paired evaluation. Each window's {horizon} outcomes are revealed "
            f"only after forecasts for K={','.join(map(str, arm_sizes))} finish; "
            "retrieval arms then append their own correction before the next window."
        ),
        "metric_note": "Forecasting is evaluated with MAE/RMSE, not classification accuracy.",
        "logical_deepseek_calls": state["logical_deepseek_calls"],
        "max_logical_deepseek_calls": max_calls,
        "windows": results,
        "arms": aggregate_arm_metrics(metrics_by_arm),
        "persistence_baseline": aggregate_arm_metrics(
            {
                arm: persistence_rows for arm in arm_sizes
            }
        )[0],
        "caveat": (
            f"Evaluation is specific to {dataset_name}. Initial training windows may "
            "overlap and are not independent examples; report this dataset separately "
            "from datasets with different target scales or cadences."
        ),
    }
    return report
