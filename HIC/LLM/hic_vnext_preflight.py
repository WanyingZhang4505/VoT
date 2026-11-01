"""Offline preflight audit helpers for the HIC vNext development workflow."""

from __future__ import annotations


def _ids(items, *, case_getter):
    result = [case_getter(item) for item in items]
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("case IDs must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("duplicate case IDs in loaded split")
    return sorted(result)


def audit_loaded_split(config, source_report, splits):
    """Validate a single configuration's fit/query identity and exposure boundary."""
    if not isinstance(config, dict) or not isinstance(source_report, dict) or not isinstance(splits, dict):
        raise ValueError("config, source_report, and splits must be mappings")
    if config.get("dataset") != source_report.get("dataset", config.get("dataset")):
        raise ValueError("source report dataset does not match configuration")
    if config.get("horizon") != source_report.get("horizon", config.get("horizon")):
        raise ValueError("source report horizon does not match configuration")

    fit_cases = splits.get("fit_cases")
    validation_windows = splits.get("validation_windows")
    test_windows = splits.get("test_windows")
    if not isinstance(fit_cases, list) or not isinstance(validation_windows, list):
        raise ValueError("fit_cases and validation_windows must be lists")
    if not isinstance(test_windows, list):
        raise ValueError("test_windows must be a list")
    if splits.get("outer_test_deferred") is not True or test_windows:
        raise ValueError("outer-test windows must remain deferred")
    if any("truth" in item for item in validation_windows):
        raise ValueError("validation truth must not be returned during preflight")

    eligible_fit = [
        case for case in fit_cases
        if case.get("news_quality") == "valid"
        and isinstance(case.get("news"), str)
        and case["news"].strip()
    ]
    fit_ids = _ids(eligible_fit, case_getter=lambda case: case.get("case_id"))
    expected_fit_ids = sorted(source_report.get("fit_case_ids", []))
    if fit_ids != expected_fit_ids:
        raise ValueError("fit case IDs differ from the frozen memory manifest")
    expected_fit_count = source_report.get("fit_memory_count")
    if type(expected_fit_count) is not int or len(eligible_fit) != expected_fit_count:
        raise ValueError("fit eligible count differs from the frozen memory manifest")
    if config.get("fit_memory_count") != expected_fit_count:
        raise ValueError("configuration fit count differs from its frozen report")

    validation_cases = [item.get("case") for item in validation_windows]
    if any(not isinstance(case, dict) for case in validation_cases):
        raise ValueError("each validation window must contain a case mapping")
    validation_ids = _ids(
        validation_cases, case_getter=lambda case: case.get("case_id")
    )
    expected_validation_ids = sorted(source_report.get("validation_case_ids", []))
    if validation_ids != expected_validation_ids:
        raise ValueError("validation case IDs differ from the frozen query manifest")

    audited_cases = eligible_fit + validation_cases
    cadence = splits.get("cadence")
    if not isinstance(cadence, str) or not cadence:
        raise ValueError("split cadence is missing")
    return {
        "dataset": config["dataset"],
        "horizon": config["horizon"],
        "cadence": cadence,
        "fit_eligible": len(eligible_fit),
        "fit_no_news_excluded": len(fit_cases) - len(eligible_fit),
        "validation_windows": len(validation_windows),
        "fit_case_ids": fit_ids,
        "validation_cases": [
            {
                "case_id": case["case_id"],
                "history_start": case.get("history_start"),
            }
            for case in sorted(validation_cases, key=lambda item: item["case_id"])
        ],
        "validation_truth_returned": 0,
        "outer_test_windows_returned": 0,
        "missing_news_evidence_fields": sum(
            not case.get("news_evidence") for case in audited_cases
        ),
        "missing_explicit_cutoff_fields": sum(
            not case.get("forecast_cutoff_at") for case in audited_cases
        ),
        "missing_embedded_contract_fields": sum(
            not case.get("task_contract") for case in audited_cases
        ),
        "target_mapping_status": config.get("target_mapping_status", "unknown"),
        "real_world_target_semantics_verified": bool(
            config.get("real_world_target_semantics_verified", False)
        ),
        "fit_end": splits.get("boundaries", {}).get("fit_end"),
        "validation_end": splits.get("boundaries", {}).get("validation_end"),
    }


def aggregate_preflight_rows(rows, *, source_manifest_sha256):
    """Summarize audited configurations and build fit/query identity indexes."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    keys = [(row["dataset"], row["horizon"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dataset/horizon configurations")
    fit_members = [
        (row["dataset"], case_id)
        for row in rows
        for case_id in row["fit_case_ids"]
    ]
    query_members = [
        (row["dataset"], case["case_id"], case["history_start"])
        for row in rows
        for case in row["validation_cases"]
    ]
    if any(not start for _dataset, _case_id, start in query_members):
        raise ValueError("validation history_start is required for origin deduplication")
    return {
        "schema_version": 1,
        "source_manifest_sha256": source_manifest_sha256,
        "configuration_count": len(rows),
        "fit_memory_memberships": len(fit_members),
        "unique_fit_cases": len(set(fit_members)),
        "fit_no_news_exclusions": sum(row["fit_no_news_excluded"] for row in rows),
        "validation_case_horizon_memberships": len(query_members),
        "unique_validation_origins": len({
            (dataset, history_start)
            for dataset, _case_id, history_start in query_members
        }),
        "validation_truth_returned": sum(row["validation_truth_returned"] for row in rows),
        "outer_test_windows_returned": sum(row["outer_test_windows_returned"] for row in rows),
        "missing_news_evidence_fields": sum(row["missing_news_evidence_fields"] for row in rows),
        "missing_explicit_cutoff_fields": sum(row["missing_explicit_cutoff_fields"] for row in rows),
        "missing_embedded_contract_fields": sum(row["missing_embedded_contract_fields"] for row in rows),
        "target_mapping_status_counts": _counts(row["target_mapping_status"] for row in rows),
        "real_world_target_semantics_verified_configurations": sum(
            row["real_world_target_semantics_verified"] for row in rows
        ),
        "outer_test_deferred": all(row["outer_test_windows_returned"] == 0 for row in rows),
        "validation_truth_not_returned": all(row["validation_truth_returned"] == 0 for row in rows),
        "configurations": sorted(rows, key=lambda row: (row["dataset"], row["horizon"])),
    }


def _counts(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


