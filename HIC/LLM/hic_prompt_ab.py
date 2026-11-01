"""Leakage-safe A/B evaluation of retrieved-correction prompt policies."""

import json
import math
from datetime import date
from pathlib import Path

from LLM import hic
from LLM.hic_qdrant_eval import (
    _atomic_write_json,
    _correction_text,
    _finite_twelve_values,
    _validated_forecast,
    _validate_windows,
)


POLICIES = ("current", "relevance_gated")
ARMS = ("no_retrieval", *POLICIES)
FORECAST_CALLS_PER_WINDOW = len(ARMS)
CORRECTION_CALLS_PER_WINDOW = len(POLICIES)


def _validate_training_cases(seed_cases, records_by_id, windows):
    if len(seed_cases) != 10:
        raise ValueError("prompt A/B requires exactly ten completed training cases")
    identifiers = [case.get("case_id") for case in seed_cases]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every seed case needs a non-empty case_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("seed case IDs must be unique")

    first_history_start = date.fromisoformat(windows[0]["case"]["history_start"])
    for case in seed_cases:
        _finite_twelve_values(case.get("truth"), f"training truth for {case['case_id']}")
        target_end = date.fromisoformat(case["target_end"])
        if target_end >= first_history_start:
            raise ValueError(
                f"training case {case['case_id']} is not complete before the first test history"
            )
        record = records_by_id.get(case["case_id"])
        if not isinstance(record, dict):
            raise ValueError(f"missing saved M3 record for {case['case_id']}")
        for field in ("summary", "correction", "embedding"):
            if field not in record:
                raise ValueError(f"saved M3 record {case['case_id']} is missing {field}")
    return identifiers


def _aggregate(rows):
    count = sum(row["metrics"]["n"] for row in rows)
    if count <= 0:
        raise ValueError("cannot aggregate an empty set of forecast metrics")
    return {
        "target_points": count,
        "aggregate_mae": sum(
            row["metrics"]["mae"] * row["metrics"]["n"] for row in rows
        ) / count,
        "aggregate_rmse": math.sqrt(
            sum(
                row["metrics"]["rmse"] ** 2 * row["metrics"]["n"]
                for row in rows
            ) / count
        ),
    }


def run_prompt_ab_evaluation(
    *,
    windows,
    summaries,
    stores,
    seed_cases,
    records_by_id,
    embed_summary,
    call_json,
    run_id,
    checkpoint_path=None,
    max_calls=12,
):
    """Compare K=0, current retrieval, and relevance-gated retrieval policies.

    Summaries are supplied by the caller so all arms use the same saved summary
    and this run isolates forecast prompting. Each Qdrant collection is seeded
    with the same ten completed historical cases. Test outcomes are consumed
    only after all three forecasts are committed for that case.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if set(stores) != set(POLICIES) or stores["current"] is stores["relevance_gated"]:
        raise ValueError("provide two distinct stores: current and relevance_gated")
    case_ids = _validate_windows(windows)
    if len(case_ids) != 2:
        raise ValueError("this prompt replay requires exactly the first two test windows")
    training_ids = _validate_training_cases(seed_cases, records_by_id, windows)
    if set(case_ids).intersection(training_ids):
        raise ValueError("test windows must not also appear in the training memory")
    if not isinstance(summaries, dict) or any(
        not isinstance(summaries.get(case_id), dict) for case_id in case_ids
    ):
        raise ValueError("a saved JSON summary is required for each evaluation case")

    expected_calls = len(windows) * (
        FORECAST_CALLS_PER_WINDOW + CORRECTION_CALLS_PER_WINDOW
    )
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls < expected_calls:
        raise ValueError(
            f"prompt A/B requires {expected_calls} logical calls; max_calls is too small"
        )

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "case_ids": case_ids,
        "training_case_ids": training_ids,
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
            raise ValueError("prompt A/B checkpoint schema/run ID does not match")
        if state.get("case_ids") != case_ids or state.get("training_case_ids") != training_ids:
            raise ValueError("prompt A/B checkpoint samples do not match current inputs")
        if state.get("pending_call"):
            raise RuntimeError(
                "checkpoint contains an unresolved API call; review it before retrying"
            )

    def save():
        if path:
            _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt, commit_response):
        completed = state.get("last_completed_call")
        if completed:
            if completed.get("case_id") != case_id or completed.get("stage") != stage:
                raise RuntimeError("checkpoint has an unapplied prompt A/B API result")
            response = completed["response"]
        else:
            if state.get("pending_call"):
                raise RuntimeError(
                    "checkpoint contains an unresolved API call; review it before retrying"
                )
            if state.get("logical_deepseek_calls", 0) >= max_calls:
                raise RuntimeError(f"logical DeepSeek call budget ({max_calls}) exhausted")
            state["logical_deepseek_calls"] += 1
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

    for store in stores.values():
        store.ensure_collection()
        for case in seed_cases:
            record = records_by_id[case["case_id"]]
            store.upsert_completed_case(
                case_id=case["case_id"],
                summary=record["summary"],
                correction=record["correction"],
                embedding=record["embedding"],
                observed_targets=case["truth"],
                cutoff_date=record.get("cutoff_date", case["target_end"]),
                source=record.get("source", "seeded-k10"),
            )
        if store.count() < len(seed_cases):
            raise RuntimeError("Qdrant seed collection is missing completed training cases")

    for window in windows:
        case = window["case"]
        case_id = case["case_id"]
        summary = summaries[case_id]
        window_state = state["windows"].setdefault(
            case_id,
            {"forecasts": {}, "corrections": {}, "ingested_policies": []},
        )
        query_vector = embed_summary(summary, is_query=True)
        passage_vector = embed_summary(summary, is_query=False)
        if (
            not isinstance(query_vector, (list, tuple))
            or not isinstance(passage_vector, (list, tuple))
            or not query_vector
            or len(query_vector) != len(passage_vector)
        ):
            raise ValueError(f"M3 returned invalid embeddings for {case_id}")

        retrievals = {
            policy: stores[policy].search(query_vector, limit=1)
            for policy in POLICIES
        }
        if any(
            not retrieval or not retrieval.get("correction")
            for retrieval in retrievals.values()
        ):
            raise RuntimeError(f"a retrieval arm returned no correction for {case_id}")

        prompts = {
            "no_retrieval": hic.build_reasoning_prompt(summary, case["series"]),
            "current": hic.build_reasoning_prompt(
                summary,
                case["series"],
                retrievals["current"]["correction"],
                retrieval_policy="current",
            ),
            "relevance_gated": hic.build_reasoning_prompt(
                summary,
                case["series"],
                retrievals["relevance_gated"]["correction"],
                retrieval_policy="relevance_gated",
            ),
        }
        for arm in ARMS:
            if arm in window_state["forecasts"]:
                continue

            def commit_forecast(response, selected_arm=arm):
                forecast = _validated_forecast(response)
                window_state["forecasts"][selected_arm] = {
                    "prediction": forecast["prediction"],
                    "reasoning": forecast["reasoning"],
                    "retrieval": retrievals.get(selected_arm),
                }

            invoke(case_id, f"forecast_{arm}", prompts[arm], commit_forecast)

        if set(window_state["forecasts"]) != set(ARMS):
            raise RuntimeError(f"not all prompt A/B forecasts completed for {case_id}")

        # Only now may the held-out target sequence enter scoring/correction state.
        truth = _finite_twelve_values(window["truth"], f"truth for {case_id}")
        if "truth" in case or "target_values" in case:
            raise ValueError(f"evaluation input {case_id} unexpectedly contains target truth")
        metrics = {
            arm: hic.forecast_metrics(
                window_state["forecasts"][arm]["prediction"], truth
            )
            for arm in ARMS
        }
        persistence = hic.forecast_metrics(
            hic.persistence_forecast(case["series"], len(truth)), truth
        )
        window_state["scoring"] = {"truth": truth, "metrics": metrics, "persistence": persistence}
        save()

        for policy in POLICIES:
            forecast = window_state["forecasts"][policy]
            if policy not in window_state["corrections"]:
                correction_prompt = hic.build_correction_prompt(
                    summary,
                    case["series"],
                    truth,
                    forecast["prediction"],
                    forecast["reasoning"],
                )
                invoke(
                    case_id,
                    f"correction_{policy}",
                    correction_prompt,
                    lambda response, selected_policy=policy: window_state[
                        "corrections"
                    ].__setitem__(selected_policy, _correction_text(response)),
                )
            if policy not in window_state["ingested_policies"]:
                stores[policy].upsert_completed_case(
                    case_id=case_id,
                    summary=summary,
                    correction=window_state["corrections"][policy],
                    embedding=passage_vector,
                    observed_targets=truth,
                    prediction=forecast["prediction"],
                    cutoff_date=case["target_end"],
                    source=f"prompt-ab:{run_id}:{policy}",
                )
                window_state["ingested_policies"].append(policy)
                save()

    result_windows = []
    for window in windows:
        case_id = window["case"]["case_id"]
        window_state = state["windows"][case_id]
        result = {
            "case_id": case_id,
            "history_start": window["case"].get("history_start"),
            "history_end": window["case"].get("history_end"),
            "target_start": window["case"].get("target_start"),
            "target_end": window["case"].get("target_end"),
            "target_months": list(window.get("target_months", [])),
            "summary": summaries[case_id],
            "truth": window_state["scoring"]["truth"],
            "persistence_metrics": window_state["scoring"]["persistence"],
            "arms": {},
        }
        for arm in ARMS:
            forecast = window_state["forecasts"][arm]
            result["arms"][arm] = {
                **forecast,
                "metrics": window_state["scoring"]["metrics"][arm],
            }
            if arm in window_state["corrections"]:
                result["arms"][arm]["generated_correction"] = window_state[
                    "corrections"
                ][arm]
        result_windows.append(result)

    rows_by_arm = {
        arm: [{"metrics": row["arms"][arm]["metrics"]} for row in result_windows]
        for arm in ARMS
    }
    aggregate = {arm: _aggregate(rows) for arm, rows in rows_by_arm.items()}
    baseline_mae = aggregate["no_retrieval"]["aggregate_mae"]
    for policy in POLICIES:
        aggregate[policy]["windows_better_mae_than_no_retrieval"] = sum(
            row["arms"][policy]["metrics"]["mae"]
            < row["arms"]["no_retrieval"]["metrics"]["mae"]
            for row in result_windows
        )
        aggregate[policy]["mae_relative_change_vs_no_retrieval"] = (
            aggregate[policy]["aggregate_mae"] - baseline_mae
        ) / baseline_mae if baseline_mae else None

    final_counts = {policy: store.count() for policy, store in stores.items()}
    expected_count = len(seed_cases) + len(windows)
    if any(count != expected_count for count in final_counts.values()):
        raise RuntimeError(
            f"final prompt A/B collection counts {final_counts}; expected {expected_count} each"
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "training_case_ids": training_ids,
        "summary_source": "reused from the prior M3 warmup report to isolate forecast prompting",
        "embedding_model": hic.BGE_MODEL_NAME,
        "forecast_model": "deepseek-flash",
        "protocol": (
            "Two-window prequential replay. K=0, current-prompt top-1, and "
            "relevance-gated top-1 forecasts share each saved summary/history. "
            "Truth is read only after all forecasts; each retrieval arm appends "
            "its own completed correction for the next window."
        ),
        "logical_deepseek_calls": state["logical_deepseek_calls"],
        "max_logical_deepseek_calls": max_calls,
        "final_collection_counts": final_counts,
        "arms": aggregate,
        "windows": result_windows,
        "limitation": (
            "The two windows were already used in the prior SocialGood experiment, "
            "so this is a retrospective development replay, not a blind holdout. "
            "It tests forecast-prompt behavior with fixed saved summaries; it does "
            "not retest raw-news summarization or establish generalization."
        ),
    }
