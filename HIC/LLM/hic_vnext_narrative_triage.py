"""Deterministic profile of unresolved numeric claims in fit-memory reflections."""

from __future__ import annotations

import re


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_STATUSES = ("verified", "unmapped", "mismatch")
_COUNT_FIELDS = (*_STATUSES, "ignored_enumerators")


def _counter(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _non_empty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sorted_counts(counter):
    return {key: counter[key] for key in sorted(counter)}


def build_narrative_triage_report(audit_report, *, audit_report_sha256):
    """Aggregate unresolved claim reasons without treating them as errors.

    The source must be the fit-only narrative audit. This function does not
    inspect holdout targets, alter memories, or make retrieval decisions.
    """
    if not isinstance(audit_report, dict):
        raise ValueError("audit_report must be an object")
    if not isinstance(audit_report_sha256, str) or not _SHA256_RE.fullmatch(audit_report_sha256):
        raise ValueError("audit_report_sha256 must be a lowercase SHA-256 digest")
    if audit_report.get("schema_version") != 1:
        raise ValueError("unsupported narrative numeric audit schema")
    if audit_report.get("fit_only") is not True:
        raise ValueError("triage requires a fit-only source audit")
    if audit_report.get("validation_truth_read") is not False:
        raise ValueError("source audit indicates validation truth was read")
    if audit_report.get("outer_test_targets_read") is not False:
        raise ValueError("source audit indicates outer-test targets were read")

    run_id = _non_empty(audit_report.get("run_id"), "run_id")
    manifest_hash = audit_report.get("run_manifest_sha256")
    if not isinstance(manifest_hash, str) or not _SHA256_RE.fullmatch(manifest_hash):
        raise ValueError("run_manifest_sha256 must be a lowercase SHA-256 digest")
    rows = audit_report.get("case_audits")
    if not isinstance(rows, list):
        raise ValueError("audit_report must include case_audits")
    if _counter(audit_report.get("completed_fit_memories_audited"), "completed_fit_memories_audited") != len(rows):
        raise ValueError("completed memory count does not match case_audits")

    aggregate = {key: 0 for key in _COUNT_FIELDS}
    by_status = {key: 0 for key in _STATUSES}
    by_reason = {}
    unmapped_by_field = {}
    mismatch_by_field = {}
    config_rows = {}
    rows_with_unmapped = 0
    rows_with_mismatch = 0
    rows_with_both = 0
    identities = set()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each case audit must be an object")
        dataset = _non_empty(row.get("dataset"), "case dataset")
        horizon = row.get("horizon")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("case horizon must be a positive integer")
        case_id = _non_empty(row.get("case_id"), "case_id")
        identity = (dataset, horizon, case_id)
        if identity in identities:
            raise ValueError("duplicate case identity in narrative audit")
        identities.add(identity)

        nested = row.get("narrative_numeric_claim_audit")
        if not isinstance(nested, dict):
            raise ValueError(f"case {case_id!r} is missing narrative_numeric_claim_audit")
        claims = nested.get("claims")
        counts = nested.get("counts")
        if not isinstance(claims, list) or not isinstance(counts, dict):
            raise ValueError(f"case {case_id!r} has invalid claims or counts")
        row_counts = {key: 0 for key in _COUNT_FIELDS}
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(f"case {case_id!r} contains a malformed numeric claim")
            status = claim.get("status")
            if status not in _STATUSES:
                raise ValueError(f"case {case_id!r} has an unsupported claim status")
            field = _non_empty(claim.get("field"), "claim field")
            row_counts[status] += 1
            by_status[status] += 1
            if status == "unmapped":
                reason = _non_empty(claim.get("reason"), "unmapped claim reason")
                by_reason[reason] = by_reason.get(reason, 0) + 1
                unmapped_by_field[field] = unmapped_by_field.get(field, 0) + 1
            elif status == "mismatch":
                mismatch_by_field[field] = mismatch_by_field.get(field, 0) + 1
        for key in _STATUSES:
            reported = _counter(counts.get(key), f"case {case_id!r} count {key}")
            if reported != row_counts[key]:
                raise ValueError(f"case {case_id!r} claim counts do not match claim records")
            aggregate[key] += reported
        # Enumerators are intentionally excluded from claim records, so their
        # separately reported count cannot be reconstructed from `claims`.
        row_counts["ignored_enumerators"] = _counter(
            counts.get("ignored_enumerators"), f"case {case_id!r} count ignored_enumerators"
        )
        aggregate["ignored_enumerators"] += row_counts["ignored_enumerators"]

        has_unmapped = row_counts["unmapped"] > 0
        has_mismatch = row_counts["mismatch"] > 0
        rows_with_unmapped += has_unmapped
        rows_with_mismatch += has_mismatch
        rows_with_both += has_unmapped and has_mismatch
        key = (dataset, horizon)
        if key not in config_rows:
            config_rows[key] = {"memories": 0, **{field: 0 for field in _COUNT_FIELDS}}
        config_rows[key]["memories"] += 1
        for field in _COUNT_FIELDS:
            config_rows[key][field] += row_counts[field]

    envelope = audit_report.get("narrative_numeric_claim_counts")
    if not isinstance(envelope, dict):
        raise ValueError("source audit is missing narrative_numeric_claim_counts")
    for key in _COUNT_FIELDS:
        if aggregate[key] != _counter(envelope.get(key), f"source total {key}"):
            raise ValueError("row-level and source-envelope claim totals do not match")

    configurations = audit_report.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError("source audit is missing configuration metadata")
    expected_config_count = _counter(
        audit_report.get("expected_configuration_count"), "expected_configuration_count"
    )
    if len(configurations) != expected_config_count:
        raise ValueError("expected configuration count does not match metadata")
    config_manifest = []
    seen_configs = set()
    for config in configurations:
        if not isinstance(config, dict):
            raise ValueError("configuration metadata entry must be an object")
        dataset = _non_empty(config.get("dataset"), "configuration dataset")
        horizon = config.get("horizon")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("configuration horizon must be a positive integer")
        key = (dataset, horizon)
        if key in seen_configs:
            raise ValueError("duplicate configuration metadata")
        seen_configs.add(key)
        measured = config_rows.get(key, {"memories": 0, **{field: 0 for field in _COUNT_FIELDS}})
        config_manifest.append({
            "dataset": dataset,
            "horizon": horizon,
            "result_present": config.get("result_present") is True,
            "checkpoint_present": config.get("checkpoint_present") is True,
            **measured,
        })
    unknown_configs = set(config_rows) - seen_configs
    if unknown_configs:
        raise ValueError("case audits refer to configuration metadata not in the source envelope")
    config_manifest.sort(key=lambda item: (item["dataset"], item["horizon"]))

    return {
        "schema_version": 1,
        "protocol": "hic_vnext_narrative_claim_triage_v1",
        "run_id": run_id,
        "run_manifest_sha256": manifest_hash,
        "audit_report_sha256": audit_report_sha256,
        "scope": "completed fit-memory narrative claims only",
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "counts": {
            "memories_audited": len(rows),
            "numeric_claims_total": sum(by_status.values()),
            **by_status,
            "ignored_enumerators": aggregate["ignored_enumerators"],
            "rows_with_unmapped": rows_with_unmapped,
            "rows_with_mismatch": rows_with_mismatch,
            "rows_with_both_unmapped_and_mismatch": rows_with_both,
        },
        "unmapped_by_reason": _sorted_counts(by_reason),
        "unmapped_by_field": _sorted_counts(unmapped_by_field),
        "mismatch_by_field": _sorted_counts(mismatch_by_field),
        "configuration_coverage": config_manifest,
        "interpretation": {
            "unmapped": "unresolved for automatic matching; not proven false",
            "verified": "matches a code-computed fact under the source audit's matching rules; not validation of the causal explanation",
            "mismatch": "explicit disagreement with a code-computed fact; requires exclusion or correction",
            "use_limit": "diagnostic profile only; does not change memory eligibility, retrieval, predictions, or performance metrics",
        },
    }


def render_narrative_triage_markdown(report):
    """Render the deterministic triage report as a compact, human-readable note."""
    counts = report["counts"]
    lines = [
        "# HIC vNext Narrative Numeric Claim Triage",
        "",
        f"Run: `{report['run_id']}`",
        f"Source audit SHA-256: `{report['audit_report_sha256']}`",
        f"Scope: {report['scope']}",
        "",
        "## Summary",
        "",
        f"- Completed fit memories audited: **{counts['memories_audited']}**.",
        f"- Numeric claims: **{counts['numeric_claims_total']}** total; **{counts['verified']}** verified, **{counts['unmapped']}** unresolved, and **{counts['mismatch']}** explicit mismatches.",
        f"- Rows with unresolved claims: **{counts['rows_with_unmapped']}**; rows with explicit mismatches: **{counts['rows_with_mismatch']}**; rows with both: **{counts['rows_with_both_unmapped_and_mismatch']}**.",
        f"- Ignored list enumerators: **{counts['ignored_enumerators']}**.",
        "",
        "Unmapped claims are unresolved, not proven false. Verified numeric matches do not establish that a free-text causal explanation is correct. This report is diagnostic only and does not modify memory eligibility or establish a prediction improvement.",
        "",
        "## Unmapped claims by reason",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in report["unmapped_by_reason"].items())
    lines.extend([
        "",
        "## Unmapped claims by narrative field",
        "",
        "| Field | Count |",
        "|---|---:|",
    ])
    lines.extend(f"| `{field}` | {count} |" for field, count in report["unmapped_by_field"].items())
    lines.extend([
        "",
        "## Explicit mismatches by narrative field",
        "",
        "| Field | Count |",
        "|---|---:|",
    ])
    lines.extend(f"| `{field}` | {count} |" for field, count in report["mismatch_by_field"].items())
    lines.extend([
        "",
        "## Configuration coverage",
        "",
        "| Dataset | Horizon | Result | Checkpoint | Memories | Verified | Unmapped | Mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for config in report["configuration_coverage"]:
        lines.append(
            f"| {config['dataset']} | H{config['horizon']} | "
            f"{'yes' if config['result_present'] else 'no'} | "
            f"{'yes' if config['checkpoint_present'] else 'no'} | {config['memories']} | "
            f"{config['verified']} | {config['unmapped']} | {config['mismatch']} |"
        )
    lines.extend([
        "",
        "## Scope limits",
        "",
        "- The source audit is fit-only; validation truth and outer-test targets were not read.",
        "- Rows with unresolved claims remain subject to manual review. A lack of an automatic fact match is not evidence that the statement is false.",
        "- No source memory, embedding, Qdrant point, prompt, or prediction was changed by this triage.",
        "- Rebuild from the source audit with `scripts/build_hic_vnext_narrative_triage_20261021.py`.",
        "",
    ])
    return "\n".join(lines)
