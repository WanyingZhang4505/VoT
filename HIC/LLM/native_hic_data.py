"""Build leakage-aware HIC windows from the datasets configured by upstream VoT."""

import csv
import math
import re
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path

from LLM.corpus_audit import (
    DATASET_SPECS,
    DEFAULT_DATA_DIR,
    _is_consecutive,
    _numeric_target,
    _read_grouped_rows,
    _parse_timestamp,
)
from LLM.sample_data import filter_news_by_cutoff


FIT_FRACTION = 0.64
VALIDATION_FRACTION = 0.8
_EXCLUDED_NUMERIC_FIELDS = {
    "OT", "fact", "preds", "prior_history_avg", "prior_history_std",
    "start_date", "end_date", "date", "Date", "Month",
}
_LEADING_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s*(.*)$", re.DOTALL)
_EXPLICIT_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_RELEVANCE_CAVEAT = re.compile(
    r"\b(?:not\s+(?:(?:directly|explicitly|specifically)\s+)?mentioned|"
    r"no\s+direct\s+mention)\b",
    re.IGNORECASE,
)
_MISSING_INFORMATION_SUFFIX = re.compile(
    r"(?:\s+|^)(?:All other parts are (?:n/?a|not available),\s*"
    r"as there is no relevant information available in the search results\.|n/?a)"
    r"[.!?]*\s*$",
    re.IGNORECASE,
)
_CLIMATE_TARGET_METADATA = {
    "target": "contiguous U.S. monthly precipitation total",
    "unit": "inches",
    "target_metadata_source": (
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
        "national/time-series/110/pcp/12/12/1895-2025.csv"
    ),
    "target_metadata_evidence": (
        "The local target description identifies contiguous U.S. precipitation. "
        "Fit-only monthly OT sums for 1983-1990 match the NOAA-linked annual "
        "series to 0.1 inch; the original downloaded file hash is not retained."
    ),
}


def target_metadata_for_dataset(dataset):
    """Return the reviewed target mapping for a dataset, if one is available."""
    if dataset == "Climate":
        return dict(_CLIMATE_TARGET_METADATA)
    return None


def _dataset_spec(dataset):
    for spec in DATASET_SPECS:
        if Path(spec.filename).stem == dataset:
            return spec
    supported = ", ".join(Path(spec.filename).stem for spec in DATASET_SPECS)
    raise ValueError(f"unsupported dataset {dataset!r}; supported datasets: {supported}")


def _segments(rows, spec):
    segments = []
    current = []
    previous = None
    for timestamp, row in rows:
        if previous is not None and not _is_consecutive(previous, timestamp, spec):
            if current:
                segments.append(current)
            current = []
        current.append((timestamp, row))
        previous = timestamp
    if current:
        segments.append(current)
    return segments


def _period_end(timestamp, row, cadence):
    raw = str(row.get("end_date", "") or "").strip()
    parsed = _parse_timestamp(raw) if raw else None
    if parsed is not None:
        return parsed
    if cadence == "month":
        return timestamp.replace(day=monthrange(timestamp.year, timestamp.month)[1])
    return timestamp


def _safe_fact_fragments(row, timestamp, cadence):
    fact = str(row.get("fact", "") or "").strip()
    if not fact or fact.casefold() in {"na", "n/a", "none", "null"}:
        return []

    available = _period_end(timestamp, row, cadence)
    if cadence == "month":
        cutoff = date(available.year, available.month, monthrange(available.year, available.month)[1])
    else:
        cutoff = available.date()

    fragments = []
    for fragment in fact.split(";"):
        fragment = " ".join(fragment.split())
        fragment = _MISSING_INFORMATION_SUFFIX.sub("", fragment).strip()
        if not fragment or fragment.casefold() in {"na", "n/a", "none", "null"}:
            continue
        # Keep source dates only when present. For undated facts, the containing
        # row is an availability boundary, not a fabricated event/publication date.
        dated_match = _LEADING_DATE.match(fragment)
        if dated_match:
            if (
                filter_news_by_cutoff(fragment, cutoff)
                and not _is_missing_fact_body(dated_match.group(2))
            ):
                fragments.append(fragment)
            continue
        if _is_missing_fact_body(fragment):
            continue
        try:
            if any(date.fromisoformat(value) > cutoff for value in _EXPLICIT_DATE.findall(fragment)):
                continue
        except ValueError:
            continue
        if any(int(year) > cutoff.year for year in _YEAR.findall(fragment)):
            continue
        fragments.append(fragment)
    return fragments


def _is_missing_fact_body(value):
    """Recognize explicit empty-news markers without discarding terms like NAFTA."""
    normalized = " ".join(str(value or "").split()).casefold().strip()
    if normalized in {"na", "n/a", "none", "null"}:
        return True
    if normalized.startswith(("no information available", "no relevant information")):
        return True
    wrapped_na = re.fullmatch(r"n/?a\s*\(([^)]*)\)", normalized)
    if wrapped_na and wrapped_na.group(1).startswith((
        "no information available",
        "no information found",
        "no relevant information",
        "no relevant objective facts",
        "this source is not relevant",
    )):
        return True
    return bool(re.fullmatch(
        r"n/?a\s*(?:\((?:no information available|no relevant information)[^)]*\)|"
        r"[-:]\s*(?:no information available|no relevant information).*)",
        normalized,
    ))


def _history_fact_fragments(history_rows, spec):
    fragments = []
    for timestamp, row in history_rows:
        fragments.extend(_safe_fact_fragments(row, timestamp, spec.cadence))
    return fragments


def _history_news(history_rows, spec):
    return "\n".join(_history_fact_fragments(history_rows, spec))


def _news_quality_detail(fragments):
    caveat_count = sum(bool(_RELEVANCE_CAVEAT.search(fragment)) for fragment in fragments)
    return {
        "source_verification": (
            "not_independently_verified" if fragments else "not_available"
        ),
        "target_relevance": (
            "explicit_nonmention_caveat_present" if caveat_count else "not_assessed"
        ),
        "fragment_count": len(fragments),
        "nonmention_caveat_count": caveat_count,
    }


def _numeric_covariates(history_rows, fieldnames, spec):
    excluded = _EXCLUDED_NUMERIC_FIELDS | set(spec.group_columns) | {spec.date_column}
    result = {}
    for field in fieldnames:
        if field in excluded:
            continue
        values = []
        for _, row in history_rows:
            raw = str(row.get(field, "") or "").strip()
            if not raw:
                break
            try:
                value = float(raw)
            except ValueError:
                break
            if not math.isfinite(value):
                break
            values.append(value)
        if len(values) == len(history_rows):
            result[field] = values
    return result


def _numeric_covariate_names(groups, fieldnames, spec):
    excluded = _EXCLUDED_NUMERIC_FIELDS | set(spec.group_columns) | {spec.date_column}
    names = []
    for field in fieldnames:
        if field in excluded:
            continue
        raw_values = [
            str(row.get(field, "") or "").strip()
            for rows in groups.values()
            for _, row in rows
            if str(row.get(field, "") or "").strip()
        ]
        try:
            numeric_values = [float(raw) for raw in raw_values]
        except ValueError:
            continue
        if raw_values and all(math.isfinite(value) for value in numeric_values):
            names.append(field)
    return names


def _make_window(
    segment, start, *, dataset, spec, fieldnames, group_key, split, horizon,
    include_empty_news=False, include_target_truth=True,
):
    history_length = spec.sequence_length
    history = segment[start:start + history_length]
    target = segment[start + history_length:start + history_length + horizon]
    if len(history) != history_length or len(target) != horizon:
        return None

    news_fragments = _history_fact_fragments(history, spec)
    news = "\n".join(news_fragments)
    if not news and not include_empty_news:
        return None

    history_start, _ = history[0]
    history_end = _period_end(*history[-1], spec.cadence)
    target_start = target[0][0]
    target_end = _period_end(*target[-1], spec.cadence)
    group_label = "|".join(group_key) if group_key else "series"
    case_id = (
        f"{dataset.lower()}_{split}_{group_label}_"
        f"{history_start.strftime('%Y%m%dT%H%M%S')}"
    )
    case = {
        "case_id": case_id,
        "dataset": dataset,
        "frequency": {
            "month": "monthly",
            "week": "weekly",
            "day": "daily",
        }.get(spec.cadence, spec.cadence),
        "group": list(group_key),
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "source_dates": [stamp.isoformat() for stamp, _ in history],
        "series": [row["OT"] for _, row in history],
        "news": news,
        # Legacy `news_quality` only records whether cleaned text exists.
        "news_quality": "valid" if news else "no_valid_text",
        "news_quality_detail": _news_quality_detail(news_fragments),
    }
    if dataset == "Climate":
        case.update(_CLIMATE_TARGET_METADATA)
    covariates = _numeric_covariates(history, fieldnames, spec)
    if covariates:
        case["covariates"] = covariates
    truth = [row["OT"] for _, row in target] if include_target_truth else None
    target_timestamps = [stamp.isoformat() for stamp, _ in target]
    return case, truth, target_timestamps


def _scan_period_timestamps(csv_path, spec):
    """Read only dates needed to define chronological splits, never target values."""
    timestamps = set()
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if spec.date_column not in (reader.fieldnames or []):
            raise ValueError(f"{spec.filename} is missing required date column {spec.date_column!r}")
        for row in reader:
            timestamp = _parse_timestamp(row.get(spec.date_column))
            if timestamp is not None:
                timestamps.add(_period_end(timestamp, row, spec.cadence))
    return sorted(timestamps)


def _read_grouped_rows_without_targets(csv_path, spec, *, max_period_end):
    """Read row metadata/facts without parsing any target values."""
    grouped = {}
    quality = {
        "raw_rows": 0,
        "rows_after_cutoff": 0,
        "invalid_timestamps": 0,
        "rows_missing_group_key": 0,
        "blank_fact_rows": 0,
        "duplicate_timestamps": 0,
        "deduplicated_rows": 0,
    }
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {spec.date_column, "OT", "fact", *spec.group_columns}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"{spec.filename} is missing required columns: {missing}")
        for row in reader:
            quality["raw_rows"] += 1
            timestamp = _parse_timestamp(row.get(spec.date_column))
            if timestamp is None:
                quality["invalid_timestamps"] += 1
                continue
            if _period_end(timestamp, row, spec.cadence) > max_period_end:
                quality["rows_after_cutoff"] += 1
                continue
            group_key = tuple(str(row.get(column, "") or "").strip() for column in spec.group_columns)
            if any(not value for value in group_key):
                quality["rows_missing_group_key"] += 1
                continue
            raw_row = dict(row)
            raw_row["fact"] = str(row.get("fact", "") or "").strip()
            if not raw_row["fact"]:
                quality["blank_fact_rows"] += 1
            grouped.setdefault(group_key, {}).setdefault(timestamp, []).append(raw_row)

    result = {}
    for group_key, by_timestamp in grouped.items():
        rows = []
        for timestamp, duplicates in sorted(by_timestamp.items()):
            if len(duplicates) > 1:
                quality["duplicate_timestamps"] += 1
                quality["deduplicated_rows"] += len(duplicates) - 1
            merged = dict(duplicates[0])
            facts = list(dict.fromkeys(row["fact"] for row in duplicates if row["fact"]))
            merged["fact"] = "\n".join(facts)
            merged["_raw_target_rows"] = duplicates
            rows.append((timestamp, merged))
        result[group_key] = rows
    return result, quality, fieldnames


def _target_blind_history_row(timestamp, row):
    """Resolve a value only when its timestamp is already in a query's history."""
    raw_rows = row.get("_raw_target_rows") or [row]
    targets = [_numeric_target(raw) for raw in raw_rows]
    if any(value is None for value in targets) or len(set(targets)) != 1:
        return None
    clean = {key: value for key, value in row.items() if key != "_raw_target_rows"}
    clean["OT"] = targets[0]
    return timestamp, clean


def _build_target_blind_validation_windows(
    groups, fieldnames, spec, *, dataset, horizon, fit_end, validation_end,
    include_empty_news,
):
    windows = []
    quality = {
        "cadence_breaks": 0,
        "contiguous_segments": 0,
        "invalid_or_ambiguous_history_targets": 0,
        "windows_skipped_without_safe_history_text": 0,
        "validation_target_values_parsed": 0,
    }
    for group_key, rows in sorted(groups.items()):
        quality["cadence_breaks"] += sum(
            not _is_consecutive(left[0], right[0], spec)
            for left, right in zip(rows, rows[1:])
        )
        for segment in _segments(rows, spec):
            quality["contiguous_segments"] += 1
            final_start = len(segment) - spec.sequence_length - horizon
            for start in range(0, final_start + 1, horizon):
                history = [
                    _target_blind_history_row(timestamp, row)
                    for timestamp, row in segment[start:start + spec.sequence_length]
                ]
                if any(row is None for row in history):
                    quality["invalid_or_ambiguous_history_targets"] += 1
                    continue
                target = segment[
                    start + spec.sequence_length:
                    start + spec.sequence_length + horizon
                ]
                target_first = target[0][0]
                target_last = _period_end(target[-1][0], target[-1][1], spec.cadence)
                if target_first <= fit_end or target_last > validation_end:
                    continue

                case_segment = list(segment)
                case_segment[start:start + spec.sequence_length] = history
                built = _make_window(
                    case_segment,
                    start,
                    dataset=dataset,
                    spec=spec,
                    fieldnames=fieldnames,
                    group_key=group_key,
                    split="validation",
                    horizon=horizon,
                    include_empty_news=include_empty_news,
                    include_target_truth=False,
                )
                if built is None:
                    quality["windows_skipped_without_safe_history_text"] += 1
                    continue
                case, _, target_timestamps = built
                windows.append({"case": case, "target_timestamps": target_timestamps})
    windows.sort(key=lambda item: (item["case"]["history_start"], item["case"]["case_id"]))
    return windows, quality


def _build_fit_cases(groups, fieldnames, spec, *, dataset, horizon, fit_end, include_empty_news):
    cases = []
    for group_key, rows in sorted(groups.items()):
        for segment in _segments(rows, spec):
            final_start = len(segment) - spec.sequence_length - horizon
            for start in range(0, final_start + 1, horizon):
                target = segment[
                    start + spec.sequence_length:
                    start + spec.sequence_length + horizon
                ]
                if _period_end(target[-1][0], target[-1][1], spec.cadence) > fit_end:
                    continue
                built = _make_window(
                    segment,
                    start,
                    dataset=dataset,
                    spec=spec,
                    fieldnames=fieldnames,
                    group_key=group_key,
                    split="fit",
                    horizon=horizon,
                    include_empty_news=include_empty_news,
                )
                if built is None:
                    continue
                case, truth, target_timestamps = built
                cases.append({**case, "truth": truth, "target_timestamps": target_timestamps})
    cases.sort(key=lambda item: (item["history_start"], item["case_id"]))
    return cases


def load_native_hic_splits(
    dataset,
    *,
    horizon,
    data_dir=None,
    fit_fraction=FIT_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    include_outer_test_targets=True,
    include_empty_news=False,
    include_validation_truth=True,
):
    """Return time-ordered fit memories, validation windows, and a held-out test.

    The dataset's sequence length and legal horizons come from the original
    ``scripts/vot.sh`` configuration as mirrored in ``corpus_audit.py``. Cases
    advance by one full horizon, so target periods do not overlap. Undated facts
    remain undated; their source row is used only as an availability boundary.
    The loader never fabricates an event or publication timestamp. Set
    ``include_outer_test_targets=False`` during planning/execution preparation
    to derive boundaries from date metadata and parse targets only through the
    validation cutoff; outer-test windows are then deferred.
    """
    spec = _dataset_spec(dataset)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon not in spec.prediction_lengths:
        raise ValueError(
            f"supported horizons for {dataset} are {list(spec.prediction_lengths)}; got {horizon}"
        )
    if not 0 < fit_fraction < validation_fraction < 1:
        raise ValueError("split fractions must satisfy 0 < fit < validation < 1")

    if not isinstance(include_outer_test_targets, bool):
        raise ValueError("include_outer_test_targets must be a boolean")
    if not isinstance(include_empty_news, bool):
        raise ValueError("include_empty_news must be a boolean")
    if not isinstance(include_validation_truth, bool):
        raise ValueError("include_validation_truth must be a boolean")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    csv_path = root / spec.filename
    if include_outer_test_targets:
        groups, quality, fieldnames = _read_grouped_rows(csv_path, spec)
        all_timestamps = sorted({
            _period_end(stamp, row, spec.cadence)
            for rows in groups.values()
            for stamp, row in rows
        })
    else:
        # Split boundaries are derived from date metadata only. Then the normal
        # parser skips every row after validation_end before converting OT.
        all_timestamps = _scan_period_timestamps(csv_path, spec)
    if len(all_timestamps) < 3:
        raise ValueError(f"{dataset} has too few distinct timestamps for temporal splits")
    fit_boundary_index = max(1, min(len(all_timestamps) - 2, int(len(all_timestamps) * fit_fraction)))
    validation_boundary_index = max(
        fit_boundary_index + 1,
        min(len(all_timestamps) - 1, int(len(all_timestamps) * validation_fraction)),
    )
    fit_end = all_timestamps[fit_boundary_index - 1]
    validation_end = all_timestamps[validation_boundary_index - 1]

    if not include_outer_test_targets and not include_validation_truth:
        fit_groups, fit_quality, fit_fieldnames = _read_grouped_rows(
            csv_path, spec, max_period_end=fit_end
        )
        validation_groups, validation_quality, _ = _read_grouped_rows_without_targets(
            csv_path, spec, max_period_end=validation_end
        )
        numeric_fields = _numeric_covariate_names(fit_groups, fit_fieldnames, spec)
        fit_cases = _build_fit_cases(
            fit_groups,
            numeric_fields,
            spec,
            dataset=dataset,
            horizon=horizon,
            fit_end=fit_end,
            include_empty_news=include_empty_news,
        )
        validation_windows, window_quality = _build_target_blind_validation_windows(
            validation_groups,
            numeric_fields,
            spec,
            dataset=dataset,
            horizon=horizon,
            fit_end=fit_end,
            validation_end=validation_end,
            include_empty_news=include_empty_news,
        )
        quality = dict(validation_quality)
        quality.update(window_quality)
        quality.update({
            # This compatibility field now describes only the fit prefix.
            "invalid_target_rows": fit_quality["invalid_target_rows"],
            "fit_invalid_target_rows": fit_quality["invalid_target_rows"],
            "raw_rows": None,
            "rows_after_cutoff": None,
            "target_parse_scope": "fit prefix and each query's observed history only",
        })
        return {
            "dataset": dataset,
            "cadence": spec.cadence,
            "history_length": spec.sequence_length,
            "horizon": horizon,
            "fit_fraction": fit_fraction,
            "validation_fraction": validation_fraction,
            "row_count": None,
            "covariate_columns": numeric_fields,
            "boundaries": {
                "fit_end": fit_end.isoformat(),
                "validation_end": validation_end.isoformat(),
                "outer_test_start": None,
            },
            "quality": quality,
            "fit_cases": fit_cases,
            "validation_windows": validation_windows,
            "test_windows": [],
            "outer_test_deferred": True,
            "validation_truth_read": False,
        }

    if not include_outer_test_targets:
        groups, quality, fieldnames = _read_grouped_rows(
            csv_path, spec, max_period_end=validation_end
        )
        # Do not expose corpus-wide or post-cutoff row counts from the dev-only
        # view. The outer split is intentionally absent from its returned data.
        quality["raw_rows"] = None
        quality["rows_after_cutoff"] = None

    fit_cases, validation_windows, test_windows = [], [], []
    segment_count = 0
    cadence_breaks = 0
    skipped_no_text = {"fit": 0, "validation": 0, "test": 0}
    windows_without_valid_news = {"fit": 0, "validation": 0, "test": 0}
    windows_with_relevance_caveats = {"fit": 0, "validation": 0, "test": 0}
    relevance_caveat_fragments = {"fit": 0, "validation": 0, "test": 0}
    for group_key, rows in sorted(groups.items()):
        cadence_breaks += sum(
            not _is_consecutive(left[0], right[0], spec)
            for left, right in zip(rows, rows[1:])
        )
        for segment in _segments(rows, spec):
            segment_count += 1
            final_start = len(segment) - spec.sequence_length - horizon
            for start in range(0, final_start + 1, horizon):
                target = segment[start + spec.sequence_length:start + spec.sequence_length + horizon]
                target_first = target[0][0]
                target_last = _period_end(target[-1][0], target[-1][1], spec.cadence)
                if target_last <= fit_end:
                    split = "fit"
                elif target_first > fit_end and target_last <= validation_end:
                    split = "validation"
                elif target_first > validation_end:
                    split = "test"
                else:
                    continue

                built = _make_window(
                    segment, start, dataset=dataset, spec=spec,
                    fieldnames=fieldnames, group_key=group_key, split=split,
                    horizon=horizon, include_empty_news=include_empty_news,
                    include_target_truth=(split != "validation" or include_validation_truth),
                )
                if built is None:
                    skipped_no_text[split] += 1
                    continue
                case, truth, target_timestamps = built
                if case["news_quality"] == "no_valid_text":
                    windows_without_valid_news[split] += 1
                caveat_count = case["news_quality_detail"]["nonmention_caveat_count"]
                if caveat_count:
                    windows_with_relevance_caveats[split] += 1
                    relevance_caveat_fragments[split] += caveat_count
                if split == "fit":
                    fit_cases.append({**case, "truth": truth, "target_timestamps": target_timestamps})
                else:
                    item = {"case": case, "target_timestamps": target_timestamps}
                    if split != "validation" or include_validation_truth:
                        item["truth"] = truth
                    (validation_windows if split == "validation" else test_windows).append(item)

    fit_cases.sort(key=lambda item: (item["history_start"], item["case_id"]))
    validation_windows.sort(key=lambda item: (item["case"]["history_start"], item["case"]["case_id"]))
    test_windows.sort(key=lambda item: (item["case"]["history_start"], item["case"]["case_id"]))
    quality = dict(quality)
    quality.update({
        "cadence_breaks": cadence_breaks,
        "contiguous_segments": segment_count,
        "windows_skipped_without_safe_history_text": skipped_no_text,
        "windows_without_valid_news": windows_without_valid_news,
        "windows_with_explicit_nonmention_caveats": windows_with_relevance_caveats,
        "nonmention_caveat_fragments_in_windows": relevance_caveat_fragments,
        "fact_timestamp_policy": "source dates retained only when present; undated facts remain undated and row-aligned; explicit future dates filtered",
    })
    return {
        "dataset": dataset,
        "cadence": spec.cadence,
        "history_length": spec.sequence_length,
        "horizon": horizon,
        "fit_fraction": fit_fraction,
        "validation_fraction": validation_fraction,
        "row_count": quality["raw_rows"],
        "covariate_columns": _numeric_covariate_names(groups, fieldnames, spec),
        "boundaries": {
            "fit_end": fit_end.isoformat(),
            "validation_end": validation_end.isoformat(),
            "outer_test_start": (
                all_timestamps[validation_boundary_index].isoformat()
                if include_outer_test_targets else None
            ),
        },
        "quality": quality,
        "fit_cases": fit_cases,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "outer_test_deferred": not include_outer_test_targets,
    }


def load_native_hic_validation_truths(
    dataset,
    *,
    horizon,
    case_ids,
    data_dir=None,
    include_empty_news=False,
):
    """Load validation truth for an exact frozen input set after forecasting.

    Case selection is rebuilt through the target-blind input loader. Only after
    that selection is fixed are validation OT values parsed for scoring.
    Outer-test targets are never loaded by this helper.
    """
    if (
        not isinstance(case_ids, (list, tuple))
        or not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("case_ids must be a non-empty sequence of unique case IDs")

    splits = load_native_hic_splits(
        dataset,
        horizon=horizon,
        data_dir=data_dir,
        include_outer_test_targets=False,
        include_empty_news=include_empty_news,
        include_validation_truth=False,
    )
    validation_by_id = {
        item["case"]["case_id"]: item
        for item in splits["validation_windows"]
    }
    requested = set(case_ids)
    if set(validation_by_id) != requested:
        raise ValueError(
            "requested validation case IDs must exactly match the development split"
        )

    spec = _dataset_spec(dataset)
    csv_path = (Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR) / spec.filename
    validation_end = _parse_timestamp(splits["boundaries"]["validation_end"])
    truth_groups, _, _ = _read_grouped_rows(
        csv_path, spec, max_period_end=validation_end
    )
    truth_by_group_date = {
        group_key: {timestamp: row["OT"] for timestamp, row in rows}
        for group_key, rows in truth_groups.items()
    }

    result = {}
    for case_id in case_ids:
        item = validation_by_id[case_id]
        case = item["case"]
        group_key = tuple(case["group"])
        timestamps = item.get("target_timestamps")
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != horizon
            or not all(isinstance(value, str) and value for value in timestamps)
        ):
            raise ValueError(f"validation target timestamps for {case_id} are invalid")
        date_values = truth_by_group_date.get(group_key, {})
        truth = []
        for value in timestamps:
            timestamp = _parse_timestamp(value)
            if timestamp not in date_values:
                raise ValueError(
                    f"validation truth for {case_id} is missing or ambiguous at {value}"
                )
            truth.append(date_values[timestamp])
        if not all(math.isfinite(value) for value in truth):
            raise ValueError(f"validation truth for {case_id} must contain finite values")
        result[case_id] = {
            "truth": [float(value) for value in truth],
            "target_timestamps": timestamps,
        }
    return result


def load_native_hic_validation_truth_for_case(
    dataset, *, horizon, case, target_timestamps, data_dir=None,
    return_resolution=False,
):
    """Reveal only one already-forecast validation case's target values.

    The CSV date and group columns are scanned, but ``OT`` is parsed only for
    the exact requested timestamps and entity. Outer-test target values and
    every other validation target remain unread.
    """
    spec = _dataset_spec(dataset)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(case, dict) or case.get("dataset") != dataset:
        raise ValueError("case dataset does not match the requested validation dataset")
    if case.get("split", "validation") != "validation":
        raise ValueError("only validation case truth may be revealed")
    if not isinstance(target_timestamps, list) or len(target_timestamps) != horizon:
        raise ValueError("target_timestamps must contain exactly one horizon")
    parsed_timestamps = [_parse_timestamp(value) for value in target_timestamps]
    if any(value is None for value in parsed_timestamps) or len(set(parsed_timestamps)) != horizon:
        raise ValueError("target_timestamps must be unique valid timestamps")
    group = case.get("group")
    if not isinstance(group, list) or len(group) != len(spec.group_columns):
        raise ValueError("validation case must contain its exact entity/group key")
    group_key = tuple(str(value) for value in group)
    requested = set(parsed_timestamps)
    csv_path = (Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR) / spec.filename
    values_by_timestamp = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {spec.date_column, "OT", *spec.group_columns}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{spec.filename} is missing required columns: {missing}")
        for row in reader:
            timestamp = _parse_timestamp(row.get(spec.date_column))
            if timestamp not in requested:
                continue
            row_group = tuple(str(row.get(column, "") or "").strip()
                              for column in spec.group_columns)
            if row_group != group_key:
                continue
            value = _numeric_target(row)
            if value is None or not math.isfinite(value):
                raise ValueError(
                    f"validation target for {case['case_id']} is invalid at {timestamp.isoformat()}"
                )
            values_by_timestamp.setdefault(timestamp, []).append(float(value))
    truth = []
    duplicate_resolution = []
    for timestamp in parsed_timestamps:
        values = values_by_timestamp.get(timestamp, [])
        if not values:
            raise ValueError(
                f"validation target for {case['case_id']} is missing at {timestamp.isoformat()}"
            )
        if len(values) > 1:
            duplicate_resolution.append({
                "timestamp": timestamp.isoformat(),
                "candidate_values": values,
                "selected_row_ordinal": 1,
                "selection_policy": "first_raw_source_row",
            })
        truth.append(values[0])
    resolution = {
        "policy": "first_raw_source_row_per_timestamp",
        "duplicate_timestamps": duplicate_resolution,
    }
    if return_resolution:
        return truth, resolution
    return truth


def load_native_hic_prefix_splits(
    dataset,
    *,
    horizon,
    fit_end,
    validation_end,
    data_dir=None,
    include_empty_news=False,
):
    """Load fit and validation windows without parsing post-validation targets.

    Split boundaries must come from a saved experiment manifest. Rows whose
    observation period ends after ``validation_end`` are discarded before the
    target column is parsed. The result intentionally has no test split.
    """
    spec = _dataset_spec(dataset)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon not in spec.prediction_lengths:
        raise ValueError(
            f"supported horizons for {dataset} are {list(spec.prediction_lengths)}; got {horizon}"
        )
    fit_boundary = _parse_timestamp(fit_end)
    validation_boundary = _parse_timestamp(validation_end)
    if fit_boundary is None or validation_boundary is None:
        raise ValueError("fit_end and validation_end must be valid ISO timestamps")
    if validation_boundary <= fit_boundary:
        raise ValueError("validation_end must be after fit_end")
    if not isinstance(include_empty_news, bool):
        raise ValueError("include_empty_news must be a boolean")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    csv_path = root / spec.filename
    groups, quality, fieldnames = _read_grouped_rows(
        csv_path, spec, max_period_end=validation_boundary
    )
    fit_cases, validation_windows = [], []
    segment_count = 0
    cadence_breaks = 0
    skipped_no_text = {"fit": 0, "validation": 0}
    windows_without_valid_news = {"fit": 0, "validation": 0}
    windows_with_relevance_caveats = {"fit": 0, "validation": 0}
    relevance_caveat_fragments = {"fit": 0, "validation": 0}

    for group_key, rows in sorted(groups.items()):
        cadence_breaks += sum(
            not _is_consecutive(left[0], right[0], spec)
            for left, right in zip(rows, rows[1:])
        )
        for segment in _segments(rows, spec):
            segment_count += 1
            final_start = len(segment) - spec.sequence_length - horizon
            for start in range(0, final_start + 1, horizon):
                target = segment[
                    start + spec.sequence_length:
                    start + spec.sequence_length + horizon
                ]
                target_first = target[0][0]
                target_last = _period_end(target[-1][0], target[-1][1], spec.cadence)
                if target_last <= fit_boundary:
                    split = "fit"
                elif target_first > fit_boundary and target_last <= validation_boundary:
                    split = "validation"
                else:
                    continue

                built = _make_window(
                    segment,
                    start,
                    dataset=dataset,
                    spec=spec,
                    fieldnames=fieldnames,
                    group_key=group_key,
                    split=split,
                    horizon=horizon,
                    include_empty_news=include_empty_news,
                )
                if built is None:
                    skipped_no_text[split] += 1
                    continue
                case, truth, target_timestamps = built
                if case["news_quality"] == "no_valid_text":
                    windows_without_valid_news[split] += 1
                caveat_count = case["news_quality_detail"]["nonmention_caveat_count"]
                if caveat_count:
                    windows_with_relevance_caveats[split] += 1
                    relevance_caveat_fragments[split] += caveat_count
                if split == "fit":
                    fit_cases.append({
                        **case,
                        "truth": truth,
                        "target_timestamps": target_timestamps,
                    })
                else:
                    validation_windows.append({
                        "case": case,
                        "truth": truth,
                        "target_timestamps": target_timestamps,
                    })

    fit_cases.sort(key=lambda item: (item["history_start"], item["case_id"]))
    validation_windows.sort(
        key=lambda item: (item["case"]["history_start"], item["case"]["case_id"])
    )
    quality = dict(quality)
    quality.update({
        "cadence_breaks": cadence_breaks,
        "contiguous_segments": segment_count,
        "windows_skipped_without_safe_history_text": skipped_no_text,
        "windows_without_valid_news": windows_without_valid_news,
        "windows_with_explicit_nonmention_caveats": windows_with_relevance_caveats,
        "nonmention_caveat_fragments_in_windows": relevance_caveat_fragments,
        "fact_timestamp_policy": (
            "source dates retained only when present; undated facts remain undated "
            "and row-aligned; explicit future dates filtered"
        ),
    })
    return {
        "dataset": dataset,
        "cadence": spec.cadence,
        "history_length": spec.sequence_length,
        "horizon": horizon,
        "fit_end": fit_boundary.isoformat(),
        "validation_end": validation_boundary.isoformat(),
        "quality": quality,
        "fit_cases": fit_cases,
        "validation_windows": validation_windows,
    }
