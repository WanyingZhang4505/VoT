"""Offline inventory of time-safe HIC candidate windows in upstream VoT data."""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    date_column: str
    cadence: str
    sequence_length: int
    prediction_lengths: tuple
    group_columns: tuple = ()
    max_gap_hours: float = 24.0
    vot_sh_active: bool = False
    data_add_active: bool = False


# Sequence lengths and horizons mirror scripts/vot.sh. Only SocialGood is
# uncommented in that script's dataset list; the other entries are available
# configurations, not evidence that those experiments were run.
DATASET_SPECS = (
    DatasetSpec("Agriculture.csv", "date", "month", 8, (6, 8, 10, 12)),
    DatasetSpec("Climate.csv", "Date", "month", 8, (6, 8, 10, 12)),
    DatasetSpec("Economy_sort.csv", "date", "month", 8, (6, 8, 10, 12)),
    DatasetSpec("Economy.csv", "date", "month", 8, (6, 8, 10, 12)),
    DatasetSpec("Energy.csv", "date", "week", 36, (12, 24, 36, 48), data_add_active=True),
    DatasetSpec(
        "Environment.csv", "date", "day", 96, (48, 96, 192, 336),
        ("CBSA Code", "Category", "Defining Parameter", "Defining Site"),
    ),
    DatasetSpec("Health.csv", "date", "week", 36, (12, 24, 36, 48), ("REGION TYPE", "REGION")),
    DatasetSpec("Security.csv", "date", "month", 8, (6, 8, 10, 12)),
    DatasetSpec("SocialGood.csv", "date", "month", 8, (6, 8, 10, 12), vot_sh_active=True, data_add_active=True),
    DatasetSpec(
        "new_metr.csv", "date", "irregular", 96, (48, 96, 192, 336),
        max_gap_hours=24.0,
    ),
    DatasetSpec("Traffic.csv", "date", "irregular", 8, (6, 8, 10, 12)),
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_REPORT_PATH = ROOT / "LLM" / "evaluations" / "vot_hic_corpus_audit.json"


def _parse_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for pattern in ("%Y%m", "%Y-%m"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _is_consecutive(previous, current, spec):
    delta = current - previous
    if spec.cadence == "month":
        next_year, next_month = (
            (previous.year + 1, 1) if previous.month == 12
            else (previous.year, previous.month + 1)
        )
        return (
            current.year == next_year
            and current.month == next_month
            and current.day == previous.day
        )
    if spec.cadence == "week":
        return delta == timedelta(days=7)
    if spec.cadence == "day":
        return delta == timedelta(days=1)
    if spec.cadence == "irregular":
        return timedelta(0) < delta <= timedelta(hours=spec.max_gap_hours)
    raise ValueError(f"Unsupported cadence: {spec.cadence}")


def _row_period_end(row, timestamp, spec):
    raw_end = str(row.get("end_date", "") or "").strip()
    parsed_end = _parse_timestamp(raw_end) if raw_end else None
    if parsed_end is not None:
        return parsed_end
    if spec.cadence == "month":
        return timestamp.replace(day=monthrange(timestamp.year, timestamp.month)[1])
    return timestamp


def _numeric_target(row):
    try:
        number = float(row.get("OT", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_grouped_rows(csv_path, spec, *, max_period_end=None):
    if max_period_end is not None and not isinstance(max_period_end, datetime):
        raise ValueError("max_period_end must be a datetime or None")
    grouped = {}
    quality = {
        "raw_rows": 0,
        "rows_after_cutoff": 0,
        "invalid_timestamps": 0,
        "invalid_target_rows": 0,
        "rows_missing_group_key": 0,
        "blank_fact_rows": 0,
        "duplicate_timestamps": 0,
        "conflicting_duplicate_timestamps": 0,
        "deduplicated_rows": 0,
        "cadence_breaks": 0,
        "contiguous_segments": 0,
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
            if max_period_end is not None and _row_period_end(row, timestamp, spec) > max_period_end:
                quality["rows_after_cutoff"] += 1
                continue
            target = _numeric_target(row)
            if target is None:
                quality["invalid_target_rows"] += 1
                continue
            if not str(row.get("fact", "") or "").strip():
                quality["blank_fact_rows"] += 1
            group_key = tuple(str(row.get(column, "") or "").strip() for column in spec.group_columns)
            if any(not value for value in group_key):
                quality["rows_missing_group_key"] += 1
                continue
            row = dict(row)
            row["OT"] = target
            row["fact"] = str(row.get("fact", "") or "").strip()
            grouped.setdefault(group_key, {}).setdefault(timestamp, []).append(row)

    cleaned_groups = {}
    for group_key, by_timestamp in grouped.items():
        clean_rows = []
        for timestamp, duplicate_rows in sorted(by_timestamp.items()):
            if len(duplicate_rows) > 1:
                quality["duplicate_timestamps"] += 1
                quality["deduplicated_rows"] += len(duplicate_rows) - 1
                targets = {row["OT"] for row in duplicate_rows}
                if len(targets) != 1:
                    quality["conflicting_duplicate_timestamps"] += 1
                    # An ambiguous target is excluded instead of arbitrarily choosing
                    # one value. The resulting time gap will split the series.
                    continue
            row = dict(duplicate_rows[0])
            if len(duplicate_rows) > 1:
                unique_facts = list(dict.fromkeys(
                    row["fact"] for row in duplicate_rows if row["fact"]
                ))
                row["fact"] = "\n".join(unique_facts)
            clean_rows.append((timestamp, row))
        cleaned_groups[group_key] = clean_rows
    return cleaned_groups, quality, fieldnames


def _split_name(target_start, target_stop, fit_boundary, validation_boundary):
    if target_stop <= fit_boundary:
        return "fit"
    if target_start >= fit_boundary and target_stop <= validation_boundary:
        return "validation"
    if target_start >= validation_boundary:
        return "test"
    return "crossing_split"


def _empty_window_counts():
    return {
        "total": 0,
        "fit": 0,
        "validation": 0,
        "test": 0,
        "crossing_split": 0,
        "with_history_text": 0,
        "fit_with_history_text": 0,
        "validation_with_history_text": 0,
        "test_with_history_text": 0,
        "crossing_split_with_history_text": 0,
    }


def _count_segment_windows(segment, *, sequence_length, horizon, stride, fit_boundary, validation_boundary):
    counts = _empty_window_counts()
    final_start = len(segment) - sequence_length - horizon
    for local_start in range(0, final_start + 1, stride):
        target_start_position = segment[local_start + sequence_length][0]
        target_stop_position = segment[local_start + sequence_length + horizon - 1][0] + 1
        split = _split_name(
            target_start_position,
            target_stop_position,
            fit_boundary,
            validation_boundary,
        )
        counts["total"] += 1
        counts[split] += 1
        history = segment[local_start:local_start + sequence_length]
        if any(row.get("fact", "").strip() for _, _, row in history):
            counts["with_history_text"] += 1
            counts[f"{split}_with_history_text"] += 1
    return counts


def _add_counts(target, addition):
    for key, value in addition.items():
        target[key] += value


def _source_row_order_metadata(csv_path, spec):
    timestamps = []
    raw_rows = 0
    invalid_timestamps = 0
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_rows += 1
            timestamp = _parse_timestamp(row.get(spec.date_column))
            if timestamp is None:
                invalid_timestamps += 1
            else:
                timestamps.append(timestamp)
    inversions = sum(left > right for left, right in zip(timestamps, timestamps[1:]))
    duplicate_steps = sum(left == right for left, right in zip(timestamps, timestamps[1:]))
    return {
        "raw_row_count": raw_rows,
        "parsed_timestamp_count": len(timestamps),
        "invalid_timestamps": invalid_timestamps,
        "adjacent_timestamp_inversions_in_file_order": inversions,
        "adjacent_duplicate_timestamps_in_file_order": duplicate_steps,
        "first_timestamp_in_file_order": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp_in_file_order": timestamps[-1].isoformat() if timestamps else None,
        "source_loader_structural_windows": {
            str(horizon): max(0, raw_rows - spec.sequence_length - horizon + 1)
            for horizon in spec.prediction_lengths
        },
        "warning": (
            "Structural window counts follow Dataset_Custom's flat CSV row indexing. "
            "They do not prove chronological order, unique time steps, or valid entity semantics."
        ),
    }


def audit_dataset(csv_path, spec, *, fit_fraction=0.64, validation_fraction=0.8):
    """Count complete, cadence-consistent windows without calling an API."""
    if not 0 < fit_fraction < validation_fraction < 1:
        raise ValueError("split fractions must satisfy 0 < fit < validation < 1")
    path = Path(csv_path)
    grouped, quality, fieldnames = _read_grouped_rows(path, spec)
    source_order = _source_row_order_metadata(path, spec)
    group_sizes = [len(rows) for rows in grouped.values()]
    prepared_groups = []
    for rows in grouped.values():
        fit_boundary = int(len(rows) * fit_fraction)
        validation_boundary = int(len(rows) * validation_fraction)
        segments = []
        segment = []
        for position, pair in enumerate(rows):
            if segment and not _is_consecutive(segment[-1][1], pair[0], spec):
                quality["cadence_breaks"] += 1
                segments.append(segment)
                segment = []
            segment.append((position, pair[0], pair[1]))
        if segment:
            segments.append(segment)
        quality["contiguous_segments"] += len(segments)
        prepared_groups.append((segments, fit_boundary, validation_boundary))

    horizon_reports = []
    for horizon in spec.prediction_lengths:
        sliding = _empty_window_counts()
        non_overlapping = _empty_window_counts()
        for segments, fit_boundary, validation_boundary in prepared_groups:
            for segment in segments:
                _add_counts(sliding, _count_segment_windows(
                    segment,
                    sequence_length=spec.sequence_length,
                    horizon=horizon,
                    stride=1,
                    fit_boundary=fit_boundary,
                    validation_boundary=validation_boundary,
                ))
                _add_counts(non_overlapping, _count_segment_windows(
                    segment,
                    sequence_length=spec.sequence_length,
                    horizon=horizon,
                    stride=horizon,
                    fit_boundary=fit_boundary,
                    validation_boundary=validation_boundary,
                ))
        horizon_reports.append({
            "prediction_length": horizon,
            "upstream_loader_structural_windows": source_order[
                "source_loader_structural_windows"
            ][str(horizon)],
            "matches_original_hic_layout": (
                spec.filename == "SocialGood.csv"
                and spec.sequence_length == 8
                and horizon == 12
                and spec.cadence == "month"
                and not spec.group_columns
            ),
            "sliding_windows": sliding,
            "non_overlapping_windows": non_overlapping,
        })
    numeric_columns = []
    for column in fieldnames:
        if column in {spec.date_column, "fact", "preds", *spec.group_columns}:
            continue
        numeric_columns.append(column)
    return {
        "dataset": path.stem,
        "filename": path.name,
        "source_script_config": {
            "sequence_length": spec.sequence_length,
            "prediction_lengths": list(spec.prediction_lengths),
            "cadence_assumption": spec.cadence,
            "group_columns": list(spec.group_columns),
            "vot_sh_active": spec.vot_sh_active,
            "data_add_active": spec.data_add_active,
        },
        "quality": {
            **quality,
            "parsed_groups": len(grouped),
            "valid_numeric_rows_after_duplicate_resolution": sum(group_sizes),
            "group_size_min": min(group_sizes) if group_sizes else 0,
            "group_size_median": sorted(group_sizes)[len(group_sizes) // 2] if group_sizes else 0,
            "group_size_max": max(group_sizes) if group_sizes else 0,
            "fit_fraction": fit_fraction,
            "validation_fraction_of_full_series": validation_fraction,
        },
        "upstream_dataset_custom_row_view": source_order,
        "available_non_target_columns_not_used_by_this_inventory": numeric_columns,
        "horizons": horizon_reports,
    }


def _canonical_digest(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        canonical_rows = sorted(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in reader
        )
    return hashlib.sha256("\n".join(canonical_rows).encode("utf-8")).hexdigest()


def audit_corpus(data_dir=DEFAULT_DATA_DIR, *, fit_fraction=0.64, validation_fraction=0.8):
    """Audit all configured source CSVs and identify exact duplicate files."""
    data_dir = Path(data_dir)
    reports = []
    missing_files = []
    seen_digests = {}
    aliases = []
    for spec in DATASET_SPECS:
        path = data_dir / spec.filename
        if not path.exists():
            missing_files.append(spec.filename)
            continue
        digest = _canonical_digest(path)
        if digest in seen_digests:
            aliases.append({"filename": spec.filename, "duplicate_of": seen_digests[digest]})
            continue
        seen_digests[digest] = spec.filename
        reports.append(audit_dataset(
            path,
            spec,
            fit_fraction=fit_fraction,
            validation_fraction=validation_fraction,
        ))
    unexpected = sorted(
        path.name for path in data_dir.glob("*.csv")
        if path.name not in {spec.filename for spec in DATASET_SPECS}
    )
    return {
        "schema_version": 1,
        "method": {
            "split": "per series: first 64% fit, next 16% validation, final 20% test",
            "window_policy": "only complete numeric windows inside one cadence-contiguous segment; sliding and horizon-stride counts both reported",
            "text_policy": "report whether at least one non-empty fact occurs in the history; do not claim the text itself is leakage-free",
            "prediction_target": "OT",
            "api_calls": 0,
            "qdrant_writes": 0,
        },
        "datasets": reports,
        "missing_configured_files": missing_files,
        "exact_duplicate_files_skipped": aliases,
        "unconfigured_csv_files": unexpected,
        "interpretation": (
            "Strict candidate counts group by the declared entity columns and require "
            "cadence-contiguous observations. The separate upstream Dataset_Custom "
            "counts follow its flat CSV row indexing and are structural only; the loader "
            "does not group entities or sort timestamps. Neither count alone proves a "
            "complete, leakage-safe HIC case. Only SocialGood is enabled in scripts/vot.sh; "
            "other sequence/horizon settings are declared there but commented out. A full "
            "HIC run still needs dataset-specific prompt, aggregation, cadence, feature, "
            "and horizon support."
        ),
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--fit-fraction", type=float, default=0.64)
    parser.add_argument("--validation-fraction", type=float, default=0.8)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = audit_corpus(
        args.data_dir,
        fit_fraction=args.fit_fraction,
        validation_fraction=args.validation_fraction,
    )
    if not args.no_write:
        _write_json(args.output, result)
        result["report_path"] = str(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
