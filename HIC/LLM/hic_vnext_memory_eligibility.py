"""Build a source-bound, fit-only admission manifest from memory audit rows."""

from __future__ import annotations

import hashlib
import json
import re


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_NUMERIC_COUNT_FIELDS = (
    "mismatch",
    "unmapped",
    "verified",
    "ignored_enumerators",
)


def _required_sha256(value, label):
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _memory_identity(row):
    if not isinstance(row, dict):
        raise ValueError("each memory audit row must be an object")
    dataset = row.get("dataset")
    case_id = row.get("case_id")
    horizon = row.get("horizon")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("memory audit row must include a dataset")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("memory audit row must include a case_id")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError(f"memory audit row {case_id!r} must include a positive horizon")
    return dataset, horizon, case_id


def _numeric_counts(row):
    audit = row.get("narrative_numeric_claim_audit")
    counts = audit.get("counts") if isinstance(audit, dict) else None
    if not isinstance(counts, dict):
        raise ValueError("memory audit row is missing narrative numeric-claim counts")
    normalized = {}
    for field in _NUMERIC_COUNT_FIELDS:
        value = counts.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"narrative numeric-claim count {field!r} must be a non-negative integer")
        normalized[field] = value
    return normalized


def _classify_row(row):
    dataset, horizon, case_id = _memory_identity(row)
    issues = row.get("issue_codes")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError(f"memory audit row {case_id!r} has invalid issue_codes")
    complete = row.get("is_complete_memory")
    if type(complete) is not bool:
        raise ValueError(f"memory audit row {case_id!r} has invalid completion status")
    counts = _numeric_counts(row)

    if not complete:
        status = "excluded_incomplete_memory"
    elif issues:
        status = "excluded_integrity_issue"
    elif counts["mismatch"]:
        status = "excluded_numeric_mismatch"
    elif counts["unmapped"]:
        status = "manual_review_unmapped"
    else:
        status = "eligible_numeric_claims_mapped"

    # The strict policy admits only fully checked memories. The alternate flag
    # preserves the larger candidate pool for a later, explicitly labeled
    # policy comparison; it never admits a known contradiction or integrity error.
    eligible = status == "eligible_numeric_claims_mapped"
    candidate_if_only_known_errors_blocked = (
        complete and not issues and counts["mismatch"] == 0
    )
    return {
        "dataset": dataset,
        "horizon": horizon,
        "case_id": case_id,
        "identity": f"{dataset}/H{horizon}/{case_id}",
        "status": status,
        "eligible_for_retrieval": eligible,
        "candidate_if_only_known_errors_blocked": candidate_if_only_known_errors_blocked,
        "manual_review_required": status == "manual_review_unmapped",
        "integrity_issue_codes": sorted(set(issues)),
        "numeric_claim_counts": counts,
    }


def _configuration_coverage(audit_report, records):
    source_configs = audit_report.get("configurations")
    if not isinstance(source_configs, list):
        raise ValueError("audit report must include configuration coverage metadata")

    config_flags = {}
    for config in source_configs:
        if not isinstance(config, dict):
            raise ValueError("each configuration coverage entry must be an object")
        dataset = config.get("dataset")
        horizon = config.get("horizon")
        result_present = config.get("result_present")
        checkpoint_present = config.get("checkpoint_present")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("configuration coverage entry must include a dataset")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("configuration coverage entry must include a positive horizon")
        if type(result_present) is not bool or type(checkpoint_present) is not bool:
            raise ValueError("configuration coverage flags must be booleans")
        if result_present and not checkpoint_present:
            raise ValueError("a result configuration must have a checkpoint")
        identity = (dataset, horizon)
        if identity in config_flags:
            raise ValueError("duplicate configuration coverage identity")
        config_flags[identity] = (result_present, checkpoint_present)

    reported = {}
    for field in (
        "result_configuration_count",
        "checkpoint_configuration_count",
        "expected_configuration_count",
    ):
        value = audit_report.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"audit report {field} must be a non-negative integer")
        reported[field] = value

    actual_results = sum(result for result, _ in config_flags.values())
    actual_checkpoints = sum(checkpoint for _, checkpoint in config_flags.values())
    if reported["expected_configuration_count"] != len(config_flags):
        raise ValueError("expected configuration count does not match configuration metadata")
    if reported["result_configuration_count"] != actual_results:
        raise ValueError("result configuration count does not match configuration metadata")
    if reported["checkpoint_configuration_count"] != actual_checkpoints:
        raise ValueError("checkpoint configuration count does not match configuration metadata")

    audited_configs = {
        (row["dataset"], row["horizon"])
        for row in records
    }
    if not audited_configs.issubset(config_flags):
        raise ValueError("memory audit rows refer to an unknown configuration")
    if any(not config_flags[identity][1] for identity in audited_configs):
        raise ValueError("memory audit rows exist for a configuration without a checkpoint")
    checkpoint_only_with_rows = sorted(
        identity
        for identity in audited_configs
        if config_flags[identity] == (False, True)
    )

    return {
        "audited_configuration_count": len(audited_configs),
        "result_configuration_count": actual_results,
        "checkpoint_configuration_count": actual_checkpoints,
        "expected_configuration_count": reported["expected_configuration_count"],
        "checkpoint_only_configurations_with_audited_rows": [
            f"{dataset}/H{horizon}" for dataset, horizon in checkpoint_only_with_rows
        ],
    }


def build_memory_eligibility_manifest(audit_report, *, audit_report_sha256):
    """Classify audited memories without modifying source records or holdouts.

    Strict admission requires a complete memory with no structural audit issue,
    no explicit numeric contradiction, and no unmapped numeric claim. A second
    candidate flag exposes the less strict policy that blocks only known errors;
    it is not the default retrieval-eligibility decision.
    """
    if not isinstance(audit_report, dict):
        raise ValueError("audit_report must be an object")
    _required_sha256(audit_report_sha256, "audit_report_sha256")
    if audit_report.get("schema_version") != 1:
        raise ValueError("unsupported memory audit report schema")
    if audit_report.get("fit_only") is not True:
        raise ValueError("memory eligibility requires a fit-only audit report")
    if audit_report.get("validation_truth_read") is not False:
        raise ValueError("audit report indicates validation truth was read")
    if audit_report.get("outer_test_targets_read") is not False:
        raise ValueError("audit report indicates outer-test targets were read")
    run_id = audit_report.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("audit report must include a run_id")
    manifest_sha256 = _required_sha256(
        audit_report.get("run_manifest_sha256"), "run_manifest_sha256"
    )
    source_rows = audit_report.get("case_audits")
    if not isinstance(source_rows, list):
        raise ValueError("audit report must include case_audits")
    audited_count = audit_report.get("completed_fit_memories_audited")
    if type(audited_count) is not int or audited_count != len(source_rows):
        raise ValueError("audited memory count does not match case_audits")

    records = [_classify_row(row) for row in source_rows]
    identities = [row["identity"] for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate fit memory identity in audit report")
    records.sort(key=lambda row: (row["dataset"], row["horizon"], row["case_id"]))
    configuration_coverage = _configuration_coverage(audit_report, records)

    status_counts = {
        "audited": len(records),
        "eligible_for_retrieval": sum(row["eligible_for_retrieval"] for row in records),
        "candidate_if_only_known_errors_blocked": sum(
            row["candidate_if_only_known_errors_blocked"] for row in records
        ),
        "excluded_incomplete_memory": sum(
            row["status"] == "excluded_incomplete_memory" for row in records
        ),
        "excluded_integrity_issue": sum(
            row["status"] == "excluded_integrity_issue" for row in records
        ),
        "excluded_numeric_mismatch": sum(
            row["status"] == "excluded_numeric_mismatch" for row in records
        ),
        "manual_review_unmapped": sum(
            row["status"] == "manual_review_unmapped" for row in records
        ),
    }
    manifest = {
        "schema_version": 1,
        "protocol": "hic_vnext_memory_eligibility_v1",
        "run_id": run_id,
        "run_manifest_sha256": manifest_sha256,
        "audit_report_sha256": audit_report_sha256,
        "audit_scope": "completed fit memories only",
        "configuration_coverage": configuration_coverage,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "admission_policy": {
            "strict": "admit only complete memories with no integrity issue, explicit numeric mismatch, or unmapped numeric claim",
            "alternate_candidate": "block only incomplete memories, integrity issues, and explicit numeric mismatches; unmapped numeric claims remain flagged for manual review",
            "source_records_modified": False,
        },
        "counts": status_counts,
        "records": records,
    }
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest
