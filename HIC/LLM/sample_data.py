"""Deterministic, time-safe SocialGood samples for the VoT HIC demo."""

import csv
import random
import re
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path

HISTORY_LENGTH = 8
FORECAST_LENGTH = 12
MAX_TRAIN_CASES = 100
TRAIN_START_INDEX = 619
EXTRA_TRAIN_START_INDICES = tuple(range(360, 540, 20))
TEST_HISTORY_START_INDEX = 719

_LEADING_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s*(.*)$", re.DOTALL)
_EXPLICIT_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_EXISTING_TRAIN_NEWS = (
    "Following the September 11 attacks, economists warn of prolonged "
    "labor-market disruption; layoffs announced across sectors."
)
_EXISTING_TEST_NEWS = (
    "Amid the 2009-2010 financial crisis aftermath, reports point to a "
    "fragile labor market with slow, uneven recovery."
)


def _month_end(value):
    return value.replace(day=monthrange(value.year, value.month)[1])


def filter_news_by_cutoff(news, cutoff):
    """Keep timestamped fact fragments containing no explicit future date/year."""
    if not isinstance(cutoff, date):
        raise TypeError("cutoff must be a datetime.date")

    kept = []
    seen = set()
    for raw_fragment in str(news or "").split(";"):
        fragment = " ".join(raw_fragment.split())
        match = _LEADING_DATE.match(fragment)
        if match is None:
            continue
        try:
            leading_date = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if leading_date > cutoff:
            continue

        try:
            dates = [date.fromisoformat(value) for value in _EXPLICIT_DATE.findall(fragment)]
        except ValueError:
            continue
        if any(value > cutoff for value in dates):
            continue
        years = [int(value) for value in _YEAR.findall(fragment)]
        if any(year > cutoff.year for year in years):
            continue

        normalized = f"{match.group(1)}: {match.group(2).strip()}"
        key = normalized.casefold()
        if normalized.endswith(":") or key in seen:
            continue
        seen.add(key)
        kept.append(normalized)
    return "\n".join(kept)


def _load_rows(csv_path):
    path = Path(csv_path) if csv_path is not None else Path(__file__).resolve().parent.parent / "data" / "SocialGood.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"date", "OT", "fact"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"SocialGood CSV must contain columns {sorted(required)}")
    rows.sort(key=lambda row: row["date"])
    for row in rows:
        datetime.strptime(row["date"], "%Y-%m-%d")
        float(row["OT"])
    parsed_dates = [date.fromisoformat(row["date"]) for row in rows]
    if len(set(parsed_dates)) != len(parsed_dates):
        raise ValueError("SocialGood CSV contains duplicate monthly dates")
    for previous, current in zip(parsed_dates, parsed_dates[1:]):
        expected = (
            date(previous.year + 1, 1, 1)
            if previous.month == 12
            else date(previous.year, previous.month + 1, 1)
        )
        if current != expected:
            raise ValueError(
                f"SocialGood CSV must be monthly and contiguous; expected {expected}, found {current}"
            )
    return rows


def _window(rows, start_index, case_type):
    history_start = start_index
    history_end = history_start + HISTORY_LENGTH
    target_end = history_end + FORECAST_LENGTH
    if target_end > len(rows):
        raise ValueError(f"Not enough SocialGood rows for window starting at {start_index}")

    history_rows = rows[history_start:history_end]
    target_rows = rows[history_end:target_end]
    history_end_date = datetime.strptime(history_rows[-1]["date"], "%Y-%m-%d").date()
    cutoff = _month_end(history_end_date)
    if case_type == "existing_train":
        news = _EXISTING_TRAIN_NEWS
    elif case_type == "existing_test":
        news = _EXISTING_TEST_NEWS
    else:
        news = filter_news_by_cutoff("; ".join(row.get("fact", "") for row in history_rows), cutoff)
        if not news:
            raise ValueError(f"No timestamped, cutoff-safe fact fragments for window {start_index}")

    return {
        "case_id": f"socialgood_{case_type}_{history_rows[0]['date'][:7]}",
        "source_start_index": start_index,
        "history_start": history_rows[0]["date"],
        "history_end": history_rows[-1]["date"],
        "target_start": target_rows[0]["date"],
        "target_end": target_rows[-1]["date"],
        "series": [float(row["OT"]) for row in history_rows],
        "news": news,
        "truth": [float(row["OT"]) for row in target_rows],
    }


def _select_training_start_indices(rows, train_count):
    """Keep the original ten cases, then add time-spread valid historical windows."""
    seed_starts = [TRAIN_START_INDEX, *EXTRA_TRAIN_START_INDICES]
    if train_count <= len(seed_starts):
        return seed_starts[:train_count]

    last_safe_start = min(
        len(rows) - HISTORY_LENGTH - FORECAST_LENGTH,
        TEST_HISTORY_START_INDEX - HISTORY_LENGTH - FORECAST_LENGTH,
    )
    eligible = []
    for start_index in range(last_safe_start + 1):
        history_rows = rows[start_index:start_index + HISTORY_LENGTH]
        history_end = datetime.strptime(history_rows[-1]["date"], "%Y-%m-%d").date()
        cutoff = _month_end(history_end)
        news = filter_news_by_cutoff(
            "; ".join(row.get("fact", "") for row in history_rows), cutoff
        )
        if news:
            eligible.append(start_index)

    if any(start_index not in eligible for start_index in seed_starts):
        raise ValueError("A preserved training seed is not a valid historical window.")
    candidates = set(eligible) - set(seed_starts)
    if train_count > len(seed_starts) + len(candidates):
        raise ValueError(
            f"Requested {train_count} training cases, but only "
            f"{len(seed_starts) + len(candidates)} time-safe cases have usable text."
        )

    selected = list(seed_starts)
    while len(selected) < train_count:
        next_index = max(
            sorted(candidates),
            key=lambda candidate: (
                min(abs(candidate - selected_index) for selected_index in selected),
                -candidate,
            ),
        )
        selected.append(next_index)
        candidates.remove(next_index)
    return selected


def load_socialgood_cases(train_count=10, csv_path=None):
    """Return (training cases, test input, test truth) for a fixed temporal split.

    Test truth is deliberately returned separately so callers can score only after
    prediction and never accidentally include it in the HIC prompt.
    """
    if not isinstance(train_count, int) or not 1 <= train_count <= MAX_TRAIN_CASES:
        raise ValueError(f"train_count must be between 1 and {MAX_TRAIN_CASES}")

    rows = _load_rows(csv_path)
    starts = _select_training_start_indices(rows, train_count)
    train_cases = []
    for position, start_index in enumerate(starts):
        case_type = "existing_train" if position == 0 else "historical_train"
        case = _window(rows, start_index, case_type)
        case_end = datetime.strptime(case["target_end"], "%Y-%m-%d").date()
        test_start = datetime.strptime(rows[TEST_HISTORY_START_INDEX]["date"], "%Y-%m-%d").date()
        if case_end >= test_start:
            raise ValueError(f"Training target overlaps the fixed test history: {case['case_id']}")
        train_cases.append(case)

    test_case = _window(rows, TEST_HISTORY_START_INDEX, "existing_test")
    test_truth = test_case.pop("truth")
    return train_cases, test_case, test_truth


def load_socialgood_eval_windows(csv_path=None, max_windows=13):
    """Return rolling-origin inputs with each 12-month truth kept separately.

    Window histories advance by one forecast horizon. Thus target months do not
    overlap, while a later window may use earlier, already-observed outcomes in
    its history. The first input matches the repository's fixed 2009-12 test case.
    """
    if not isinstance(max_windows, int) or not 1 <= max_windows <= 13:
        raise ValueError("max_windows must be an integer between 1 and 13")

    rows = _load_rows(csv_path)
    last_start = len(rows) - HISTORY_LENGTH - FORECAST_LENGTH
    starts = list(
        range(TEST_HISTORY_START_INDEX, last_start + 1, FORECAST_LENGTH)
    )[:max_windows]
    if len(starts) < max_windows:
        raise ValueError(
            f"Requested {max_windows} evaluation windows, but only {len(starts)} "
            "fit in the SocialGood series."
        )

    windows = []
    seen_target_months = set()
    for start_index in starts:
        full_case = _window(rows, start_index, "rolling_eval")
        truth = full_case.pop("truth")
        target_months = [value[:7] for value in _date_months(full_case["target_start"], truth)]
        if len(set(target_months)) != FORECAST_LENGTH:
            raise ValueError(f"Evaluation case {full_case['case_id']} has duplicate target months")
        overlap = seen_target_months.intersection(target_months)
        if overlap:
            raise ValueError(
                f"Evaluation target windows overlap at {sorted(overlap)[0]}"
            )
        seen_target_months.update(target_months)
        windows.append(
            {
                "case": full_case,
                "truth": truth,
                "target_months": target_months,
            }
        )
    return windows


def select_seeded_cases(cases, count, seed):
    """Select a reproducible random subset without changing case objects/order."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    candidates = list(cases)
    if count > len(candidates):
        raise ValueError("count cannot exceed the number of available cases")
    identifiers = [case.get("case_id") for case in candidates]
    if any(not isinstance(case_id, str) or not case_id for case_id in identifiers):
        raise ValueError("every candidate case must have a non-empty case_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate case IDs must be unique")
    return random.Random(seed).sample(candidates, count)


def _date_months(target_start, truth):
    """Derive consecutive YYYY-MM dates for a validated 12-value target window."""
    if len(truth) != FORECAST_LENGTH:
        raise ValueError(f"A SocialGood target window must contain {FORECAST_LENGTH} values")
    year, month = (int(part) for part in target_start[:7].split("-"))
    months = []
    for _ in truth:
        months.append(f"{year:04d}-{month:02d}-01")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months
