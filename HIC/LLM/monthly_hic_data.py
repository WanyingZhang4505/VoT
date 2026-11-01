"""Build leakage-aware HIC windows for the five monthly VoT datasets."""

import csv
import math
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path

from LLM.sample_data import filter_news_by_cutoff


HISTORY_LENGTH = 8
FORECAST_LENGTH = 12
FIT_FRACTION = 0.64
VALIDATION_FRACTION = 0.8
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MONTHLY_DATASETS = {
    "Agriculture": {"filename": "Agriculture.csv", "date_column": "date"},
    "Climate": {"filename": "Climate.csv", "date_column": "Date"},
    "Economy_sort": {"filename": "Economy_sort.csv", "date_column": "date"},
    "Security": {"filename": "Security.csv", "date_column": "date"},
    "SocialGood": {"filename": "SocialGood.csv", "date_column": "date"},
}

_NON_FEATURE_COLUMNS = {
    "date", "Date", "Month", "fact", "preds", "start_date", "end_date",
    "prior_history_avg", "prior_history_std", "OT",
}


def _month_end(value):
    return value.replace(day=monthrange(value.year, value.month)[1])


def _parse_monthly_date(value):
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return datetime.strptime(text, "%Y%m").date().replace(day=1)
        except ValueError:
            raise ValueError(f"unsupported monthly date: {text!r}") from None


def _count_monthly_rows(path):
    """Count CSV records without converting or retaining any target values."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{Path(path).name} has no CSV header")
        return sum(1 for _ in reader)


def _load_monthly_rows(path, date_column, *, row_limit=None):
    if row_limit is not None and (
        not isinstance(row_limit, int) or isinstance(row_limit, bool) or row_limit < 0
    ):
        raise ValueError("row_limit must be a non-negative integer or None")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {date_column, "OT", "fact"}
        missing = sorted(required - set(fields))
        if not fields or missing:
            raise ValueError(f"{Path(path).name} is missing required columns: {missing}")
        rows = []
        for row_number, raw in enumerate(reader, start=2):
            if row_limit is not None and len(rows) >= row_limit:
                break
            try:
                stamp = _parse_monthly_date(raw[date_column])
                target = float(raw["OT"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{Path(path).name} has an invalid {date_column}/OT at row {row_number}"
                ) from None
            if not math.isfinite(target):
                raise ValueError(f"{Path(path).name} has a non-finite OT at row {row_number}")
            row = dict(raw)
            row[date_column] = stamp.isoformat()
            row["OT"] = target
            rows.append((stamp, row))

    if not rows:
        raise ValueError(f"{Path(path).name} contains no data rows")
    rows.sort(key=lambda pair: pair[0])
    stamps = [stamp for stamp, _ in rows]
    if len(stamps) != len(set(stamps)):
        raise ValueError(f"{Path(path).name} contains duplicate monthly dates")
    for previous, current in zip(stamps, stamps[1:]):
        expected = (
            date(previous.year + 1, 1, previous.day)
            if previous.month == 12
            else date(previous.year, previous.month + 1, previous.day)
        )
        if current != expected:
            raise ValueError(
                f"{Path(path).name} must be monthly and contiguous; "
                f"expected {expected}, found {current}"
            )
    return fields, [row for _, row in rows]


def _numeric_covariates(fields, rows, date_column):
    excluded = _NON_FEATURE_COLUMNS | {date_column}
    candidates = []
    for field in fields:
        if field in excluded:
            continue
        saw_numeric_value = False
        for row in rows:
            raw = row[field]
            if raw is None or not str(raw).strip():
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                break
            if not math.isfinite(value):
                break
            saw_numeric_value = True
        else:
            if saw_numeric_value:
                candidates.append(field)
    return candidates


def _build_window(rows, start, *, dataset, date_column, covariate_columns, prefix):
    history_end_index = start + HISTORY_LENGTH
    target_end_index = history_end_index + FORECAST_LENGTH
    history_rows = rows[start:history_end_index]
    target_rows = rows[history_end_index:target_end_index]
    if len(history_rows) != HISTORY_LENGTH or len(target_rows) != FORECAST_LENGTH:
        raise ValueError("window must contain exactly 8 history and 12 target months")

    history_end = date.fromisoformat(history_rows[-1][date_column])
    cutoff = _month_end(history_end)
    news = filter_news_by_cutoff(
        "; ".join(str(row.get("fact", "") or "") for row in history_rows),
        cutoff,
    )
    if not news:
        return None

    target_months = [row[date_column][:7] for row in target_rows]
    case = {
        "case_id": f"{dataset.lower()}_{prefix}_{history_rows[0][date_column][:7]}",
        "dataset": dataset,
        "source_start_index": start,
        "history_start": history_rows[0][date_column],
        "history_end": history_rows[-1][date_column],
        "target_start": target_rows[0][date_column],
        "target_end": target_rows[-1][date_column],
        "series": [row["OT"] for row in history_rows],
        "news": news,
    }
    covariates = {}
    for field in covariate_columns:
        values = []
        for row in history_rows:
            raw = row.get(field)
            if raw is None or not str(raw).strip():
                break
            try:
                value = float(raw)
            except (TypeError, ValueError):
                break
            if not math.isfinite(value):
                break
            values.append(value)
        if len(values) == HISTORY_LENGTH:
            covariates[field] = values
    if covariates:
        case["covariates"] = covariates
    truth = [row["OT"] for row in target_rows]
    return case, truth, target_months


def _evaluation_windows(rows, *, dataset, date_column, covariate_columns,
                        fit_boundary, validation_boundary):
    windows = {"validation": [], "test": []}
    last_start = len(rows) - HISTORY_LENGTH - FORECAST_LENGTH
    for start in range(0, last_start + 1, FORECAST_LENGTH):
        target_start = start + HISTORY_LENGTH
        target_stop = target_start + FORECAST_LENGTH
        if target_start >= fit_boundary and target_stop <= validation_boundary:
            split = "validation"
        elif target_start >= validation_boundary:
            split = "test"
        else:
            continue
        built = _build_window(
            rows,
            start,
            dataset=dataset,
            date_column=date_column,
            covariate_columns=covariate_columns,
            prefix=split,
        )
        if built is None:
            continue
        case, truth, target_months = built
        windows[split].append({
            "case": case,
            "truth": truth,
            "target_months": target_months,
        })
    return windows


def load_monthly_hic_splits(
    dataset,
    *,
    data_dir=None,
    history_length=HISTORY_LENGTH,
    horizon=FORECAST_LENGTH,
    fit_fraction=FIT_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    include_outer_test_targets=True,
):
    """Return eligible fit memories and non-overlapping evaluation windows.

    Fit memories slide one month at a time and include their historical target
    values. Validation and outer-test windows advance by one forecast horizon;
    their truths are stored outside ``case`` and can only be exposed after scoring.
    The first 64% is fit memory, the next 16% validation, and the last 20% is
    an untouched outer test (80% development in total). Set
    ``include_outer_test_targets=False`` for preflight work; this counts CSV
    records to establish split boundaries, then parses only the development
    prefix and defers outer-test windows and targets.
    """
    if dataset not in MONTHLY_DATASETS:
        raise ValueError(f"unsupported monthly HIC dataset: {dataset}")
    if history_length != HISTORY_LENGTH or horizon != FORECAST_LENGTH:
        raise ValueError("this HIC adapter currently supports only 8-history/12-target windows")
    if not 0 < fit_fraction < validation_fraction < 1:
        raise ValueError("split fractions must satisfy 0 < fit < validation < 1")
    if not isinstance(include_outer_test_targets, bool):
        raise ValueError("include_outer_test_targets must be a boolean")

    spec = MONTHLY_DATASETS[dataset]
    path = Path(data_dir) / spec["filename"] if data_dir else DEFAULT_DATA_DIR / spec["filename"]
    if include_outer_test_targets:
        fields, rows = _load_monthly_rows(path, spec["date_column"])
        row_count = len(rows)
    else:
        row_count = _count_monthly_rows(path)
        development_end = int(row_count * validation_fraction)
        fields, rows = _load_monthly_rows(
            path, spec["date_column"], row_limit=development_end
        )
        if len(rows) != development_end:
            raise ValueError("CSV row count changed during development-only loading")
    date_column = spec["date_column"]
    covariate_columns = _numeric_covariates(fields, rows, date_column)
    fit_boundary = int(row_count * fit_fraction)
    validation_boundary = int(row_count * validation_fraction)
    final_start = row_count - HISTORY_LENGTH - FORECAST_LENGTH

    fit_cases = []
    for start in range(0, final_start + 1):
        if start + HISTORY_LENGTH + FORECAST_LENGTH > fit_boundary:
            break
        built = _build_window(
            rows,
            start,
            dataset=dataset,
            date_column=date_column,
            covariate_columns=covariate_columns,
            prefix="fit",
        )
        if built is None:
            continue
        case, truth, target_months = built
        fit_cases.append({**case, "truth": truth, "target_months": target_months})

    windows = _evaluation_windows(
        rows,
        dataset=dataset,
        date_column=date_column,
        covariate_columns=covariate_columns,
        fit_boundary=fit_boundary,
        validation_boundary=validation_boundary,
    )
    return {
        "dataset": dataset,
        "history_length": history_length,
        "horizon": horizon,
        "fit_fraction": fit_fraction,
        "validation_fraction": validation_fraction,
        "row_count": row_count,
        "covariate_columns": covariate_columns,
        "boundaries": {
            "fit_end": rows[fit_boundary - 1][date_column],
            "validation_end": rows[validation_boundary - 1][date_column],
            "outer_test_start": (
                rows[validation_boundary][date_column]
                if include_outer_test_targets and validation_boundary < len(rows)
                else None
            ),
        },
        "outer_test_deferred": not include_outer_test_targets,
        "fit_cases": fit_cases,
        "validation_windows": windows["validation"],
        "test_windows": windows["test"],
    }
