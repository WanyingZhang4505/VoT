"""Leakage-aware rolling windows for the monthly Economy HIC pilot."""

import csv
import json
import math
import os
import random
import re
import tempfile
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path

try:
    from . import hic
except ImportError:
    from LLM import hic

try:
    from .sample_data import filter_news_by_cutoff
except ImportError:
    from sample_data import filter_news_by_cutoff


HISTORY_LENGTH = 8
FORECAST_LENGTH = 12
DEFAULT_SEED = 20261018
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "Economy_sort.csv"
ECONOMY_DESCRIPTION = (
    "Monthly Economy_sort time series. OT is the numeric target; Exports and "
    "Imports are additional monthly covariates. The unit is not specified in "
    "the source file. Each supplied fact is a dated text fragment; use only "
    "facts dated on or before the end of the input history. The task is to "
    "forecast the next 12 monthly OT values, not to establish causality."
)


def _month_end(value):
    return value.replace(day=monthrange(value.year, value.month)[1])


def _load_rows(csv_path):
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Month", "date", "OT", "Exports", "Imports", "fact"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Economy CSV must contain columns {sorted(required)}")

    parsed = []
    for row in rows:
        try:
            month_date = datetime.strptime(row["Month"].strip(), "%B %Y").date()
            row_date = date.fromisoformat(row["date"].strip())
            value = float(row["OT"])
            exports = float(row["Exports"])
            imports = float(row["Imports"])
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Economy CSV contains an invalid Month, date, or OT value") from None
        if month_date != row_date.replace(day=1):
            raise ValueError(f"Month and date columns disagree at {row['Month']!r}")
        if not all(math.isfinite(number) for number in (value, exports, imports)):
            raise ValueError(f"Economy numeric values must be finite at {row['date']!r}")
        row = dict(row)
        row["date"] = row_date.isoformat()
        row["OT"] = value
        row["Exports"] = exports
        row["Imports"] = imports
        parsed.append((row_date, row))

    parsed.sort(key=lambda pair: pair[0])
    dates = [row_date for row_date, _ in parsed]
    if len(set(dates)) != len(dates):
        raise ValueError("Economy CSV contains duplicate monthly dates")
    for previous, current in zip(dates, dates[1:]):
        expected = (
            date(previous.year + 1, 1, 1)
            if previous.month == 12
            else date(previous.year, previous.month + 1, 1)
        )
        if current != expected:
            raise ValueError(
                f"Economy CSV must be monthly and contiguous; expected {expected}, found {current}"
            )
    return [row for _, row in parsed]


def _build_window(rows, history_start):
    history_end = history_start + HISTORY_LENGTH
    target_end = history_end + FORECAST_LENGTH
    history_rows = rows[history_start:history_end]
    target_rows = rows[history_end:target_end]
    if len(history_rows) != HISTORY_LENGTH or len(target_rows) != FORECAST_LENGTH:
        raise ValueError("Economy window does not contain 8 history and 12 target months")

    cutoff = _month_end(date.fromisoformat(history_rows[-1]["date"]))
    news = filter_news_by_cutoff(
        "; ".join(row.get("fact", "") or "" for row in history_rows),
        cutoff,
    )
    if not news:
        raise ValueError(
            f"No timestamped, cutoff-safe Economy facts for window "
            f"{history_rows[0]['date'][:7]}"
        )

    target_months = [row["date"][:7] for row in target_rows]
    case = {
        "case_id": f"economy_{history_rows[0]['date'][:7]}",
        "dataset": "Economy_sort",
        "history_start": history_rows[0]["date"],
        "history_end": history_rows[-1]["date"],
        "target_start": target_rows[0]["date"],
        "target_end": target_rows[-1]["date"],
        "source_start_index": history_start,
        "series": [row["OT"] for row in history_rows],
        "covariates": {
            "Exports": [row["Exports"] for row in history_rows],
            "Imports": [row["Imports"] for row in history_rows],
        },
        "news": news,
    }
    truth = [row["OT"] for row in target_rows]
    return case, truth, target_months


def load_economy_pilot_cases(
    csv_path=None,
    *,
    train_count=10,
    seed=DEFAULT_SEED,
):
    """Return random non-overlapping training cases and post-split test windows.

    The chronological split is made before candidate selection. Training targets
    end within the first 80% of observations. Test truth is returned separately
    from each test case so it cannot accidentally enter a forecast prompt.
    """
    if not isinstance(train_count, int) or train_count < 1:
        raise ValueError("train_count must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rows = _load_rows(csv_path)
    split_index = int(len(rows) * 0.8)
    if split_index < HISTORY_LENGTH + FORECAST_LENGTH:
        raise ValueError("Economy CSV is too short for an 80/20 training split")
    if len(rows) - split_index < FORECAST_LENGTH:
        raise ValueError("Economy CSV has no complete 12-month test horizon after the split")

    eligible_starts = []
    for history_start in range(split_index - HISTORY_LENGTH - FORECAST_LENGTH + 1):
        try:
            _build_window(rows, history_start)
        except ValueError as error:
            if str(error).startswith("No timestamped, cutoff-safe Economy facts"):
                continue
            raise
        eligible_starts.append(history_start)

    rng = random.Random(seed)
    shuffled_starts = list(eligible_starts)
    rng.shuffle(shuffled_starts)
    selected = []
    selected_target_spans = []
    for history_start in shuffled_starts:
        target_start = history_start + HISTORY_LENGTH
        target_stop = target_start + FORECAST_LENGTH
        overlaps = any(
            target_start < other_stop and other_start < target_stop
            for other_start, other_stop in selected_target_spans
        )
        if overlaps:
            continue
        selected.append(history_start)
        selected_target_spans.append((target_start, target_stop))
        if len(selected) == train_count:
            break
    if len(selected) < train_count:
        raise ValueError(
            f"Requested {train_count} non-overlapping training cases, but only "
            f"{len(selected)} fit before the chronological split."
        )

    train_cases = []
    for history_start in sorted(selected):
        case, truth, _ = _build_window(rows, history_start)
        case["case_id"] = f"economy_train_{case['history_start'][:7]}"
        case["truth"] = truth
        train_cases.append(case)

    test_windows = []
    last_target_start = len(rows) - FORECAST_LENGTH
    for target_start in range(split_index, last_target_start + 1, FORECAST_LENGTH):
        case, truth, target_months = _build_window(rows, target_start - HISTORY_LENGTH)
        case["case_id"] = f"economy_test_{case['target_start'][:7]}"
        test_windows.append(
            {
                "case": case,
                "truth": truth,
                "target_months": target_months,
            }
        )
    if not test_windows:
        raise ValueError("No complete, cutoff-safe Economy test windows are available")
    return train_cases, test_windows


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


def _finite_vector(values, expected_length, label):
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite numbers")
    return result


def _validated_forecast(response):
    if not isinstance(response, dict) or "prediction" not in response:
        raise ValueError("DeepSeek forecast response must contain a prediction array")
    prediction = _finite_vector(response["prediction"], FORECAST_LENGTH, "prediction")
    reasoning = response.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("forecast reasoning must be a string")
    return {**response, "prediction": prediction, "reasoning": reasoning}


def _correction_text(response):
    if not isinstance(response, dict):
        raise ValueError("DeepSeek correction response must be a JSON object")
    correction = (
        response.get("Key Insights")
        or response.get("Improved Reasoning")
        or response.get("correction")
    )
    if not isinstance(correction, str) or not correction.strip():
        raise ValueError("DeepSeek correction response contains no usable lesson")
    return correction.strip()


def _validated_embedding(vector, label):
    if not isinstance(vector, (list, tuple)) or len(vector) != 1024:
        raise ValueError(f"{label} must contain exactly 1,024 BGE-M3 dimensions")
    result = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    if not any(value != 0 for value in result):
        raise ValueError(f"{label} must be non-zero")
    return result


def _run_economy_hic_pilot(
    *,
    train_cases,
    test_window,
    template,
    store,
    embed_summary,
    call_json,
    run_id,
    max_calls=35,
    checkpoint_path=None,
    retry_pending=False,
):
    """Build a ten-case HIC memory, compare K=0/K=10 on case 11, then learn it."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, '_' or '-' characters")
    if not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    if len(train_cases) != 10:
        raise ValueError("the first Economy pilot requires exactly 10 training cases")

    case_ids = [case.get("case_id") for case in train_cases]
    if len(set(case_ids)) != 10 or any(not isinstance(value, str) or not value for value in case_ids):
        raise ValueError("training case IDs must be ten distinct non-empty strings")
    case = test_window.get("case", {})
    test_id = case.get("case_id")
    if not isinstance(test_id, str) or not test_id or test_id in case_ids:
        raise ValueError("test case must have an ID distinct from the training cases")
    if "truth" in case or "target_values" in case:
        raise ValueError("held-out test truth must not be stored in the test case input")
    _finite_vector(case.get("series"), HISTORY_LENGTH, "test history")
    _finite_vector(test_window.get("truth"), FORECAST_LENGTH, "test truth")

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "train_case_ids": case_ids,
        "test_case_id": test_id,
        "logical_deepseek_calls": 0,
        "pending_call": None,
        "responses": {},
        "training": {},
        "test": {},
    }
    path = Path(checkpoint_path) if checkpoint_path else None
    resuming = bool(path and path.exists())
    if resuming:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if (
            state.get("schema_version") != 1
            or state.get("run_id") != run_id
            or state.get("train_case_ids") != case_ids
            or state.get("test_case_id") != test_id
        ):
            raise ValueError("Economy pilot checkpoint does not match this run and input")
        if state.get("pending_call"):
            if not retry_pending:
                raise RuntimeError(
                    "checkpoint has an unresolved API call; review it and pass "
                    "retry_pending=True to retry"
                )
            state["pending_call"] = None
    elif store.count() != 0:
        raise ValueError("new Economy pilot collection must be empty")

    def save():
        if path:
            _atomic_write_json(path, state)

    def invoke(case_id, stage, prompt):
        response_key = f"{case_id}::{stage}"
        if response_key in state["responses"]:
            return state["responses"][response_key]
        if state.get("pending_call"):
            raise RuntimeError("checkpoint has an unresolved API call")
        if state["logical_deepseek_calls"] >= max_calls:
            raise RuntimeError(f"DeepSeek logical-call budget ({max_calls}) exhausted")
        state["logical_deepseek_calls"] += 1
        state["pending_call"] = {"case_id": case_id, "stage": stage}
        save()
        response = call_json(prompt, temperature=0.0)
        if not isinstance(response, dict):
            raise ValueError(f"{stage} response must be a JSON object")
        state["responses"][response_key] = response
        state["pending_call"] = None
        save()
        return response

    if template is None:
        template = invoke("economy_pilot", "template", hic.build_template_prompt())
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty JSON object")

    for train_case in train_cases:
        train_id = train_case["case_id"]
        entry = state["training"].setdefault(train_id, {})
        truth = _finite_vector(train_case.get("truth"), FORECAST_LENGTH, f"truth for {train_id}")
        history = _finite_vector(train_case.get("series"), HISTORY_LENGTH, f"history for {train_id}")
        if "summary" not in entry:
            entry["summary"] = invoke(
                train_id,
                "summary",
                hic.build_summary_prompt(
                    template,
                    {"OT": history, **train_case["covariates"]},
                    train_case["news"],
                ),
            )
            save()
        if "forecast" not in entry:
            prompt = hic.build_reasoning_prompt(
                entry["summary"],
                history,
                covariates=train_case["covariates"],
            )
            entry["forecast"] = _validated_forecast(invoke(train_id, "forecast", prompt))
            save()
        if "correction" not in entry:
            forecast = entry["forecast"]
            prompt = hic.build_correction_prompt(
                entry["summary"],
                history,
                truth,
                forecast["prediction"],
                forecast["reasoning"],
            )
            entry["correction"] = _correction_text(invoke(train_id, "correction", prompt))
            save()
        if "embedding" not in entry:
            entry["embedding"] = _validated_embedding(
                embed_summary(entry["summary"], is_query=False),
                f"embedding for {train_id}",
            )
            save()
        if not entry.get("ingested"):
            store.upsert_completed_case(
                case_id=train_id,
                summary=entry["summary"],
                correction=entry["correction"],
                embedding=entry["embedding"],
                observed_targets=truth,
                prediction=entry["forecast"]["prediction"],
                cutoff_date=train_case["target_end"],
                source=f"economy-pilot:{run_id}",
            )
            entry["ingested"] = True
            save()

    collection_count_after_train = store.count()
    if not all(state["training"].get(train_id, {}).get("ingested") for train_id in case_ids):
        raise RuntimeError("not all ten Economy training cases were marked ingested")
    test_checkpoint = state["test"]
    test_point_may_exist = bool(
        test_checkpoint.get("scored") and test_checkpoint.get("correction")
    )
    if collection_count_after_train not in (10, 11) or (
        collection_count_after_train == 11 and not test_point_may_exist
    ):
        raise RuntimeError(
            "expected ten training points, or a recoverable eleventh test point; "
            f"found {collection_count_after_train}"
        )
    if test_checkpoint.get("ingested") and collection_count_after_train != 11:
        raise RuntimeError("checkpoint says the test case is ingested, but Qdrant has no eleventh point")

    test_state = state["test"]
    history = _finite_vector(case["series"], HISTORY_LENGTH, "test history")
    if "summary" not in test_state:
        test_state["summary"] = invoke(
            test_id,
            "summary",
            hic.build_summary_prompt(
                template,
                {"OT": history, **case["covariates"]},
                case["news"],
            ),
        )
        save()
    if "embeddings" not in test_state:
        test_state["embeddings"] = {
            "query": _validated_embedding(
                embed_summary(test_state["summary"], is_query=True),
                f"query embedding for {test_id}",
            ),
            "passage": _validated_embedding(
                embed_summary(test_state["summary"], is_query=False),
                f"passage embedding for {test_id}",
            ),
        }
        save()
    if "retrieved_case" not in test_state:
        retrieval = store.search(test_state["embeddings"]["query"], limit=1)
        if not retrieval or not retrieval.get("correction"):
            raise RuntimeError("K=10 Qdrant collection returned no correction")
        test_state["retrieved_case"] = retrieval
        save()

    for arm, correction in (
        ("control", None),
        ("retrieval", test_state["retrieved_case"]["correction"]),
    ):
        if arm not in test_state:
            prompt = hic.build_reasoning_prompt(
                test_state["summary"],
                history,
                correction,
                covariates=case["covariates"],
            )
            test_state[arm] = _validated_forecast(
                invoke(test_id, f"forecast_{arm}", prompt)
            )
            save()

    # The held-out target is first used after both forecasts are checkpointed.
    truth = _finite_vector(test_window["truth"], FORECAST_LENGTH, f"truth for {test_id}")
    if "scoring" not in test_state:
        test_state["scoring"] = {
            "truth": truth,
            "control_metrics": hic.forecast_metrics(
                test_state["control"]["prediction"], truth
            ),
            "retrieval_metrics": hic.forecast_metrics(
                test_state["retrieval"]["prediction"], truth
            ),
            "persistence_metrics": hic.forecast_metrics(
                hic.persistence_forecast(history, FORECAST_LENGTH),
                truth,
            ),
            "history_mean_metrics": hic.forecast_metrics(
                [sum(history) / len(history)] * FORECAST_LENGTH,
                truth,
            ),
        }
        test_state["scored"] = True
        save()
    elif test_state["scoring"].get("truth") != truth:
        raise ValueError("test truth differs from the checkpointed scoring truth")

    if "correction" not in test_state:
        forecast = test_state["retrieval"]
        prompt = hic.build_correction_prompt(
            test_state["summary"],
            history,
            truth,
            forecast["prediction"],
            forecast["reasoning"],
        )
        test_state["correction"] = _correction_text(
            invoke(test_id, "correction_retrieval", prompt)
        )
        save()
    if not test_state.get("ingested"):
        store.upsert_completed_case(
            case_id=test_id,
            summary=test_state["summary"],
            correction=test_state["correction"],
            embedding=test_state["embeddings"]["passage"],
            observed_targets=truth,
            prediction=test_state["retrieval"]["prediction"],
            cutoff_date=case["target_end"],
            source=f"economy-pilot:{run_id}",
        )
        test_state["ingested"] = True
        save()

    collection_count_after_test = store.count()
    if collection_count_after_test != 11:
        raise RuntimeError(
            f"expected 11 Qdrant cases after scoring case 11, found {collection_count_after_test}"
        )
    control_mae = test_state["scoring"]["control_metrics"]["mae"]
    retrieval_mae = test_state["scoring"]["retrieval_metrics"]["mae"]
    return {
        "run_id": run_id,
        "dataset": "Economy_sort",
        "embedding_model": "BAAI/bge-m3",
        "forecast_model": "deepseek-flash",
        "template": template,
        "training_case_ids": case_ids,
        "collection_count_after_train": collection_count_after_train,
        "collection_count_after_test": collection_count_after_test,
        "logical_deepseek_calls": state["logical_deepseek_calls"],
        "max_logical_deepseek_calls": max_calls,
        "test": {
            "case_id": test_id,
            "history_start": case.get("history_start"),
            "history_end": case.get("history_end"),
            "target_start": case.get("target_start"),
            "target_end": case.get("target_end"),
            "truth": truth,
            "control_prediction": test_state["control"]["prediction"],
            "control_metrics": test_state["scoring"]["control_metrics"],
            "retrieval_prediction": test_state["retrieval"]["prediction"],
            "retrieval_metrics": test_state["scoring"]["retrieval_metrics"],
            "persistence_metrics": test_state["scoring"]["persistence_metrics"],
            "history_mean_metrics": test_state["scoring"]["history_mean_metrics"],
            "retrieval": test_state["retrieved_case"],
            "mae_improvement_vs_control": control_mae - retrieval_mae,
        },
    }


def run_economy_hic_pilot(
    *,
    train_cases,
    test_window,
    template,
    store,
    embed_summary,
    call_json,
    run_id,
    max_calls=35,
    checkpoint_path=None,
    retry_pending=False,
):
    """Run the pilot with Economy-specific prompts, restoring shared HIC state."""
    previous_description = hic.dataset_description
    hic.dataset_description = ECONOMY_DESCRIPTION
    try:
        return _run_economy_hic_pilot(
            train_cases=train_cases,
            test_window=test_window,
            template=template,
            store=store,
            embed_summary=embed_summary,
            call_json=call_json,
            run_id=run_id,
            max_calls=max_calls,
            checkpoint_path=checkpoint_path,
            retry_pending=retry_pending,
        )
    finally:
        hic.dataset_description = previous_description
