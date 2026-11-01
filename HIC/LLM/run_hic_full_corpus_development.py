"""Execute or preflight the approved 12-configuration VoT/HIC phase.

The default mode is a side-effect-free preflight. Live execution requires
``--execute`` and uses one exact-request cache (optionally backed by read-only
legacy caches), one cumulative HTTP-attempt ledger, one usage ledger, and an
isolated BGE-M3 Qdrant collection per config.
"""

import argparse
import copy
from functools import wraps
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from LLM import hic
from LLM.deepseek_client import API_URL
from LLM.deepseek_usage import (
    DeepSeekCNYSpendBudget,
    DeepSeekUsageLedger,
    _CNY_BUDGET_PRICING_VERSION,
    _peak_cost_micros,
    exclusive_file_lock,
    summarize_usage_ledger,
)
from LLM.hic_grid_runner import (
    VALIDATION_TRUTH_POLICY,
    VERIFIED_PIPELINE_VERSION,
    _atomic_json,
    _hash_json,
    development_jobs,
    run_configuration,
    run_development_grid,
)
from LLM.hic_phase2_eval import (
    VERIFIED_FORECAST_ARMS,
    VERIFIED_PIPELINE_VERSION,
    _resolve_template,
    build_phase2_grid_plan,
    macro_relative_improvement,
)
from LLM.hic_request_cache import RequestCache
from LLM.native_hic_data import load_native_hic_validation_truths
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.run_economy_hic_pilot import make_http_attempt_budget


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"
PHASE_GUARD_DIR = EVALUATIONS / "hic_v7_phase12_guard"
DEFAULT_PLAN = EVALUATIONS / "hic_phase2_full_corpus_development_20261019_verified_v7_target_contract.plan.json"
MAX_AUTHORIZED_LOGICAL_CALLS = 4691
MAX_AUTHORIZED_HTTP_ATTEMPTS = 6000
MAX_PHASE_CONFIGURATIONS = 12
MAX_PHASE_SPEND_CNY = 5.50
PHASE_CONFIGURATIONS = (
    ("Agriculture", 6),
    ("Agriculture", 8),
    ("Agriculture", 10),
    ("Agriculture", 12),
    ("Climate", 6),
    ("Climate", 8),
    ("Economy_sort", 12),
    ("SocialGood", 12),
    ("Energy", 48),
    ("Economy_sort", 8),
    ("Energy", 24),
    ("SocialGood", 6),
)
PHASE20_NEW_CONFIGURATIONS = (
    ("Climate", 10),
    ("Climate", 12),
    ("Economy_sort", 6),
    ("Economy_sort", 10),
    ("SocialGood", 8),
    ("SocialGood", 10),
    ("Energy", 36),
    ("Energy", 12),
)
PHASE20_CUMULATIVE_SPEND_CNY = 15.0
PHASE20_INHERITED_SPEND_CNY = "8.098836"
PHASE20_INHERITED_USAGE_SHA256 = "06c1c8f50004f0cf45df104879c32d71af37c20ecba24ae136d71553c8f99397"
PHASE20_INHERITED_HTTP_200_EVENTS = 1485
PHASE20_RUN_ID = "hic_phase20_ab_20261020"
LEGACY_COMPLETED_RUN_DIR = EVALUATIONS / "hic_v7_gatefix_full_20261020"
QDRANT_TIMEOUT_SECONDS = 60
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate_unmatched_http_attempt_reserve(
    *, usage_events, attempts_used, request_cache, max_output_tokens,
):
    """Bound failed retries using the completed retry's prompt and frozen output cap."""
    if not isinstance(usage_events, list) or type(attempts_used) is not int or attempts_used < 0:
        raise ValueError("cannot conservatively bound unmatched HTTP attempts")
    if not isinstance(request_cache, dict) or not isinstance(request_cache.get("records"), dict):
        raise ValueError("cannot conservatively bound unmatched HTTP attempts without request records")
    if request_cache.get("logical_calls") != attempts_used:
        raise ValueError("cannot conservatively bound unmatched HTTP attempts: call count mismatch")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("cannot conservatively bound unmatched HTTP attempts: output limit missing")

    attempts_from_cache = 0
    successful_calls_from_cache = 0
    retries_from_cache = 0
    for record in request_cache["records"].values():
        if not isinstance(record, dict):
            raise ValueError("cannot conservatively bound unmatched HTTP attempts: malformed cache record")
        attempts = record.get("attempts")
        if type(attempts) is not int or attempts < 0:
            raise ValueError("cannot conservatively bound unmatched HTTP attempts: invalid attempt count")
        attempts_from_cache += attempts
        if attempts:
            if record.get("status") != "persisted" or "response" not in record:
                raise ValueError("cannot conservatively bound unmatched HTTP attempts without a saved retry response")
            successful_calls_from_cache += 1
            retries_from_cache += attempts - 1

    unmatched = attempts_used - len(usage_events)
    if (
        attempts_from_cache != attempts_used
        or unmatched < 0
        or successful_calls_from_cache != len(usage_events)
        or retries_from_cache != unmatched
    ):
        raise ValueError("cannot conservatively bound unmatched HTTP attempts from retry evidence")

    if not unmatched:
        return {
            "unmatched_attempts": 0,
            "prompt_tokens_upper_bound": 0,
            "output_tokens_upper_bound": max_output_tokens,
            "reserve_micros_per_attempt": 0,
            "reserve_micros": 0,
        }

    prompt_token_counts = []
    for event in usage_events:
        usage = event.get("usage") if isinstance(event, dict) else None
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("cannot conservatively bound unmatched HTTP attempts: prompt usage missing")
        prompt_token_counts.append(prompt_tokens)
    prompt_upper_bound = max(prompt_token_counts)
    reserve_per_attempt, _normalized = _peak_cost_micros({
        "prompt_tokens": prompt_upper_bound,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": prompt_upper_bound,
        "completion_tokens": max_output_tokens,
        "total_tokens": prompt_upper_bound + max_output_tokens,
    })
    return {
        "unmatched_attempts": unmatched,
        "prompt_tokens_upper_bound": prompt_upper_bound,
        "output_tokens_upper_bound": max_output_tokens,
        "reserve_micros_per_attempt": reserve_per_attempt,
        "reserve_micros": reserve_per_attempt * unmatched,
    }


def validate_authorized_budgets(max_logical_calls, max_http_attempts):
    if type(max_logical_calls) is not int or not 1 <= max_logical_calls <= MAX_AUTHORIZED_LOGICAL_CALLS:
        raise ValueError(
            f"logical-call authorization is capped at {MAX_AUTHORIZED_LOGICAL_CALLS}"
        )
    if type(max_http_attempts) is not int or not 1 <= max_http_attempts <= MAX_AUTHORIZED_HTTP_ATTEMPTS:
        raise ValueError(
            f"HTTP-attempt authorization is capped at {MAX_AUTHORIZED_HTTP_ATTEMPTS}"
        )
    return max_logical_calls, max_http_attempts


def validate_plan_call_budget(plan, max_logical_calls):
    """Fail before provider work when the summary-reuse call estimate exceeds authorization."""
    estimate = plan.get("logical_call_estimate_with_summary_cache", {}).get("total")
    if type(estimate) is not int or estimate < 0:
        raise ValueError("development plan is missing a summary-reuse call estimate")
    if estimate > max_logical_calls:
        raise ValueError(
            f"estimated {plan.get('pipeline_version', 'HIC')} calls under summary reuse "
            f"({estimate}) exceed the authorized logical-call cap "
            f"({max_logical_calls})"
        )
    return estimate


def validate_plan_protocol(plan):
    if (
        plan.get("pipeline_version") != VERIFIED_PIPELINE_VERSION
        or plan.get("forecast_arms") != list(VERIFIED_FORECAST_ARMS)
    ):
        raise ValueError("development plan does not match the current HIC runner protocol")
    registry_sha256 = plan.get("target_contract_registry_sha256")
    if (
        not isinstance(registry_sha256, str)
        or not re.fullmatch(r"[a-f0-9]{64}", registry_sha256)
        or plan.get("evaluation_scope") != "same_artifact_numeric_exploratory"
        or plan.get("target_contract_policy") != "same_artifact_numeric_evaluation=exploratory_only"
        or plan.get("real_world_target_semantics_verified") is not False
    ):
        raise ValueError("development plan is missing the required target-contract binding")
    for row in plan.get("plans", []):
        if not isinstance(row, dict) or row.get("status") != "development_ready":
            continue
        contract = row.get("target_contract")
        if (
            not isinstance(contract, dict)
            or not isinstance(contract.get("contract_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", contract["contract_sha256"])
            or contract.get("source_sha256") != row.get("source_sha256")
            or contract.get("local_target_field") != "OT"
            or type(contract.get("dimensions_used_by_hic")) is not int
            or contract.get("dimensions_used_by_hic") != 1
            or not isinstance(contract.get("mapping_status"), str)
            or contract.get("mapping_status") not in {"verified", "partial", "conflict", "unknown"}
            or contract.get("real_world_target_semantics_verified")
            is not (contract.get("mapping_status") == "verified")
        ):
            raise ValueError(
                f"development plan is missing a valid target contract for {row.get('dataset')}"
            )
    return True


def _safe_slug(value):
    slug = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_-")
    if not slug:
        raise ValueError("dataset name cannot form a safe run identifier")
    return slug


def build_live_job_manifest(plan, *, run_id):
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-40 ASCII letters, digits, '_' or '-'")
    base_jobs = development_jobs(plan)
    output = []
    seen_collections = set()
    for job in base_jobs:
        dataset = job["dataset"]
        horizon = job["horizon"]
        job_key = f"{dataset}:{horizon}"
        digest = hashlib.sha256(
            f"{run_id}|{job_key}|{job['source_sha256']}".encode("utf-8")
        ).hexdigest()
        slug = _safe_slug(dataset)
        safe_job_id = f"{slug[:18]}_h{horizon}_{digest[:8]}"
        collection_suffix = f"{slug[:15]}_h{horizon}_{digest[:10]}"
        collection_name_for("BAAI/bge-m3", collection_suffix)
        if collection_suffix in seen_collections:
            raise ValueError("live manifest produced duplicate Qdrant collection suffixes")
        seen_collections.add(collection_suffix)
        output.append({
            **job,
            "job_id": safe_job_id,
            "run_id": f"{run_id}_{digest[:10]}",
            "collection_suffix": collection_suffix,
        })
    if len(output) != len(base_jobs):
        raise RuntimeError("live manifest did not preserve every eligible configuration")
    return output


def select_development_plan(plan, *, dataset=None, horizon=None):
    """Return a copy containing only matching rows from a verified source plan."""
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        raise ValueError("dataset filter must be a non-empty string")
    if horizon is not None and (type(horizon) is not int or horizon <= 0):
        raise ValueError("horizon filter must be a positive integer")
    rows = plan.get("plans") if isinstance(plan, dict) else None
    if not isinstance(rows, list):
        raise ValueError("development plan rows are missing")
    ready_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "development_ready"]
    selected_rows = [
        row for row in ready_rows
        if (dataset is None or row.get("dataset") == dataset)
        and (horizon is None or row.get("horizon") == horizon)
    ]
    if not selected_rows:
        raise ValueError("no development-ready configurations match the requested filters")
    selected = copy.deepcopy(plan)
    selected["plans"] = selected_rows
    selected["selected_scope"] = {"dataset": dataset, "horizon": horizon}
    selected["source_configuration_count"] = len(ready_rows)
    return selected


def select_phase20_ab_plan(source_plan):
    """Freeze the eight unfinished rows for the cumulative-20 two-arm phase."""
    if (
        not isinstance(source_plan, dict)
        or source_plan.get("pipeline_version") != VERIFIED_PIPELINE_VERSION
        or source_plan.get("outer_test_targets_read") is not False
        or source_plan.get("forecast_arms") != list(VERIFIED_FORECAST_ARMS)
    ):
        raise ValueError("phase-20 profile requires the verified v7 source plan")
    rows = source_plan.get("plans")
    if not isinstance(rows, list):
        raise ValueError("phase-20 source plan rows are missing")
    ready = [row for row in rows if isinstance(row, dict) and row.get("status") == "development_ready"]
    by_key = {(row.get("dataset"), row.get("horizon")): row for row in ready}
    expected_all = set(PHASE_CONFIGURATIONS) | set(PHASE20_NEW_CONFIGURATIONS)
    if len(by_key) != len(ready) or not expected_all.issubset(by_key):
        missing = sorted(expected_all - set(by_key))
        raise ValueError(f"phase-20 allowlist is incomplete in the verified source plan: {missing}")

    selected = copy.deepcopy(source_plan)
    selected["plans"] = [copy.deepcopy(by_key[key]) for key in PHASE20_NEW_CONFIGURATIONS]
    selected["forecast_arms"] = ["no_retrieval", "full_reasoning"]
    selected["execution_profile"] = "phase20_no_retrieval_vs_full_hic_v1"
    selected["configuration_count"] = len(PHASE20_NEW_CONFIGURATIONS)
    selected["source_configuration_count"] = len(ready)
    selected["phase20_completed_configuration_count"] = len(PHASE_CONFIGURATIONS)
    selected["phase20_new_configuration_count"] = len(PHASE20_NEW_CONFIGURATIONS)
    selected["phase20_total_configuration_count"] = len(expected_all)
    fit_count = sum(row["fit_case_count"] for row in selected["plans"])
    validation_count = sum(row["validation_window_count"] for row in selected["plans"])
    uncached_bound = 3 * fit_count + 3 * validation_count
    selected["phase20_uncached_logical_call_bound"] = uncached_bound
    selected["logical_call_estimate_with_summary_cache"] = {
        "scope": "conservative uncached bound; frozen templates and exact request cache may reduce HTTP calls",
        "fit_cases": fit_count,
        "validation_windows": validation_count,
        "total": uncached_bound,
    }
    selected["run_tag"] = "phase20-ab-20261020"
    return selected


def build_dry_run_report(plan, *, run_id):
    jobs = build_live_job_manifest(plan, run_id=run_id)
    selected_keys = {(job["dataset"], job["horizon"]) for job in jobs}
    phase_keys = [key for key in PHASE_CONFIGURATIONS if key in selected_keys]
    rows_by_key = {
        (row.get("dataset"), row.get("horizon")): row
        for row in plan.get("plans", [])
        if isinstance(row, dict) and row.get("status") == "development_ready"
    }
    missing_templates = sorted({
        job["dataset"]
        for job in jobs
        if rows_by_key[(job["dataset"], job["horizon"])].get("template", {}).get("status")
        == "template_generation_required"
    })
    source_summary_reuse_estimate = plan.get("logical_call_estimate_with_summary_cache", {}).get("total")
    source_no_reuse_estimate = plan.get("logical_call_estimate_without_cache", {}).get("total")
    selected_scope = plan.get("selected_scope") or {}
    is_filtered_plan = any(selected_scope.get(key) is not None for key in ("dataset", "horizon"))
    summary_reuse_estimate = None if is_filtered_plan else source_summary_reuse_estimate
    no_reuse_estimate = None if is_filtered_plan else source_no_reuse_estimate
    return {
        "status": "dry_run",
        "run_id": run_id,
        "pipeline_version": plan.get("pipeline_version"),
        "target_contract_registry_sha256": plan.get("target_contract_registry_sha256"),
        "evaluation_scope": plan.get("evaluation_scope"),
        "real_world_target_semantics_verified": plan.get(
            "real_world_target_semantics_verified"
        ),
        "target_contract_status_by_dataset": {
            row["dataset"]: row["target_contract"]["mapping_status"]
            for row in plan.get("plans", [])
            if isinstance(row, dict)
            and row.get("status") == "development_ready"
            and isinstance(row.get("target_contract"), dict)
        },
        "validation_truth_policy": VALIDATION_TRUTH_POLICY,
        "forecast_arms": plan.get("forecast_arms"),
        "configuration_count": len(jobs),
        "approved_phase_configuration_count": len(phase_keys),
        "approved_phase_configurations": [
            {"dataset": dataset, "horizon": horizon}
            for dataset, horizon in phase_keys
        ],
        "approved_phase_max_new_spend_cny": MAX_PHASE_SPEND_CNY,
        "source_configuration_count": plan.get("source_configuration_count", len(jobs)),
        "selected_scope": plan.get("selected_scope"),
        "planned_fit_cases": sum(len(job["fit_case_ids"]) for job in jobs),
        "planned_validation_windows": sum(len(job["validation_case_ids"]) for job in jobs),
        "estimated_logical_calls_with_summary_reuse": summary_reuse_estimate,
        "estimated_logical_calls_without_summary_reuse": no_reuse_estimate,
        "source_full_plan_estimated_logical_calls_with_summary_reuse": source_summary_reuse_estimate,
        "source_full_plan_estimated_logical_calls_without_summary_reuse": source_no_reuse_estimate,
        "call_estimate_scope": (
            "full_source_plan_only_filtered_scope_not_reestimated"
            if is_filtered_plan else "selected_plan"
        ),
        "call_estimate_basis": (
            "Counts unique summary inputs per dataset/template/as-of with cross-horizon reuse; "
            "does not measure pre-existing durable request-cache hits."
        ),
        "template_generation_required": missing_templates,
        "max_authorized_logical_calls": MAX_AUTHORIZED_LOGICAL_CALLS,
        "max_authorized_http_attempts": MAX_AUTHORIZED_HTTP_ATTEMPTS,
        "summary_reuse_estimate_within_authorized_call_cap": (
            None if is_filtered_plan else (
                type(summary_reuse_estimate) is int
                and summary_reuse_estimate <= MAX_AUTHORIZED_LOGICAL_CALLS
            )
        ),
        "provider_requests_made": 0,
        "embedding_inferences_made": 0,
        "qdrant_writes_made": 0,
        "outer_test_targets_read": False,
        "job_ids": [job["job_id"] for job in jobs],
    }


def _dataset_description(dataset):
    return (
        f"Dataset: {dataset}, from the upstream VoT repository. The numeric target "
        "column is named OT; the local metadata does not document its unit or exact "
        "real-world interpretation, so treat it as an unnamed numeric target. Text "
        "facts are aligned to source periods; undated facts are not independently "
        "verified as publication-time news."
    )


def _run_verified_configuration(**kwargs):
    """Run one job with auditable corrections and deferred validation labels."""
    return run_configuration(
        **kwargs,
        validation_truth_loader=load_native_hic_validation_truths,
        verified_corrections=True,
    )


def _plan_signature(plan):
    jobs = development_jobs(plan)
    return [
        {
            "dataset": job["dataset"],
            "horizon": job["horizon"],
            "source_sha256": job["source_sha256"],
            "target_contract": next(
                row.get("target_contract")
                for row in plan.get("plans", [])
                if row.get("status") == "development_ready"
                and row.get("dataset") == job["dataset"]
                and row.get("horizon") == job["horizon"]
            ),
            "fit_case_ids": job["fit_case_ids"],
            "validation_case_ids": job["validation_case_ids"],
        }
        for job in jobs
    ]


def verify_plan_is_current(saved_plan, current_plan):
    if saved_plan.get("outer_test_targets_read") is not False:
        raise ValueError("saved plan does not affirm that outer-test targets are unread")
    if current_plan.get("outer_test_targets_read") is not False:
        raise ValueError("fresh plan unexpectedly materialized outer-test targets")
    try:
        validate_plan_protocol(current_plan)
    except ValueError:
        raise ValueError("fresh plan does not match the current HIC runner protocol") from None
    if _plan_signature(saved_plan) != _plan_signature(current_plan):
        raise ValueError(
            "saved plan no longer matches current source hashes or development windows; "
            "write and review a fresh plan before execution"
        )
    protocol_fields = (
        "pipeline_version",
        "target_contract_registry_sha256",
        "target_contract_policy",
        "evaluation_scope",
        "real_world_target_semantics_verified",
        "forecast_arms",
        "numeric_controls",
        "fusion",
        "embedding",
        "fit_split",
        "validation_end_fraction",
        "selection_confirmation_split",
    )
    if any(saved_plan.get(field) != current_plan.get(field) for field in protocol_fields):
        raise ValueError(
            "saved plan no longer matches the frozen evaluation protocol or source template fingerprints"
        )
    saved_templates = saved_plan.get("summary_cache", {}).get("template_fingerprints")
    current_templates = current_plan.get("summary_cache", {}).get("template_fingerprints")
    if saved_templates != current_templates:
        raise ValueError("saved plan no longer matches the protocol or source template fingerprints")
    return True


def verify_phase20_source_plan(saved_plan, current_plan):
    """Verify source windows and the eight new contracts across metadata revisions.

    The historical 12-config plan predates later registry/template metadata
    refreshes. This narrow check allows those global metadata changes while
    retaining strict equality for all sampled windows and the eight rows that
    will actually be executed. The old generic verifier remains unchanged.
    """
    if saved_plan.get("outer_test_targets_read") is not False:
        raise ValueError("saved plan does not affirm that outer-test targets are unread")
    if current_plan.get("outer_test_targets_read") is not False:
        raise ValueError("fresh plan unexpectedly materialized outer-test targets")
    try:
        validate_plan_protocol(saved_plan)
        validate_plan_protocol(current_plan)
    except ValueError:
        raise ValueError("phase-20 source plan does not match the verified HIC protocol") from None

    protocol_fields = (
        "pipeline_version",
        "target_contract_policy",
        "evaluation_scope",
        "real_world_target_semantics_verified",
        "forecast_arms",
        "numeric_controls",
        "fusion",
        "embedding",
        "fit_split",
        "validation_end_fraction",
        "selection_confirmation_split",
    )
    if any(saved_plan.get(field) != current_plan.get(field) for field in protocol_fields):
        raise ValueError("phase-20 source plan differs from the frozen HIC protocol")

    def keyed_rows(plan):
        rows = plan.get("plans")
        if not isinstance(rows, list):
            raise ValueError("phase-20 source plan rows are missing")
        ready = [row for row in rows if isinstance(row, dict) and row.get("status") == "development_ready"]
        keyed = {(row.get("dataset"), row.get("horizon")): row for row in ready}
        if len(keyed) != len(ready):
            raise ValueError("phase-20 source plan contains duplicate configuration rows")
        return keyed

    saved_rows = keyed_rows(saved_plan)
    current_rows = keyed_rows(current_plan)
    identity_fields = (
        "source_file",
        "source_sha256",
        "cadence",
        "history_length",
        "fit_boundary",
        "validation_boundary",
        "fit_case_count",
        "validation_window_count",
        "fit_case_ids",
        "validation_case_ids",
    )
    if set(saved_rows) != set(current_rows) or any(
        any(saved_rows[key].get(field) != current_rows[key].get(field) for field in identity_fields)
        for key in saved_rows.keys() & current_rows.keys()
    ):
        raise ValueError(
            "saved plan no longer matches current source hashes or development windows; "
            "write and review a fresh plan before execution"
        )

    missing_new = set(PHASE20_NEW_CONFIGURATIONS) - set(saved_rows)
    if missing_new:
        raise ValueError(f"phase-20 source plan is missing new configurations: {sorted(missing_new)}")
    if any(
        saved_rows[key].get("target_contract") != current_rows[key].get("target_contract")
        for key in PHASE20_NEW_CONFIGURATIONS
    ):
        raise ValueError("phase-20 target contract changed for an execution configuration")
    return True


def _request_settings():
    return {
        "provider": "deepseek",
        "endpoint": API_URL,
        "model": hic.MODEL,
        "temperature": 0.0,
        # RequestCache treats any field containing "token" as credential-like.
        "output_limit": hic.mdl_cfg.get("max_output_tokens", 2048),
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "embedding_model": hic.BGE_MODEL_NAME,
        "embedding_dimension": hic.BGE_EMBEDDING_DIMENSION,
        "embedding_max_length": hic.BGE_MAX_LENGTH,
    }


def _empty_usage_summary():
    return {
        "http_200_events": 0,
        "usage_known_events": 0,
        "usage_missing_events": 0,
        "usage_invalid_events": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": {"all_off_peak": 0.0, "all_peak": 0.0},
        "by_model": {},
    }


def _serialize_phase_execution(function):
    """Allow only one live runner to mutate this shared HIC phase at a time."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with exclusive_file_lock(PHASE_GUARD_DIR / "live_execution.lock"):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                run_dir = kwargs.get("run_dir")
                run_id = kwargs.get("run_id")
                if run_dir is not None and run_id is not None:
                    try:
                        _record_preflight_failure(
                            _run_paths(
                                run_dir, phase20_ab=bool(kwargs.get("phase20_ab", False))
                            ),
                            run_id=run_id, error=error
                        )
                    except Exception:
                        pass
                raise
    return wrapped


def _run_paths(run_dir, *, phase20_ab=False):
    run_dir = Path(run_dir)
    phase_guard_dir = run_dir if phase20_ab else PHASE_GUARD_DIR
    return {
        "root": run_dir,
        "manifest": run_dir / "run_manifest.json",
        "plan": run_dir / "plan.json",
        "request_cache": run_dir / "request_cache.json",
        "http_attempts": run_dir / "http_attempts.json",
        "usage": run_dir / "usage.jsonl",
        "cny_spend": phase_guard_dir / ("cny_spend.json" if phase20_ab else "cny_spend.json"),
        "phase_policy": phase_guard_dir / ("phase20_policy.json" if phase20_ab else "phase12_policy.json"),
        "grid_checkpoint": run_dir / "grid_checkpoint.json",
        "summary": run_dir / "summary.json",
        "templates": run_dir / "templates",
        "jobs": run_dir / "jobs",
    }


def _run_has_prior_provider_activity(checkpoint_path):
    """Fail closed when a run lacks spend history but shows prior provider traffic."""
    run_dir = Path(checkpoint_path).parent
    attempts_path = run_dir / "http_attempts.json"
    if attempts_path.is_file():
        attempts_state = _load_json(attempts_path)
        attempts_used = attempts_state.get("attempts_used") if isinstance(attempts_state, dict) else None
        if type(attempts_used) is not int or attempts_used < 0:
            raise RuntimeError(
                "HTTP-attempt history is invalid; cannot safely initialize a cumulative CNY ledger"
            )
        if attempts_used:
            return True

    usage_path = run_dir / "usage.jsonl"
    if usage_path.is_file():
        with usage_path.open("r", encoding="utf-8") as handle:
            if any(line.strip() for line in handle):
                return True
    return False


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(path, manifest):
    _atomic_json(path, manifest)


def _prepare_phase12_policy(
    *, path, spend_path, jobs, run_id, checkpoint_path, resume,
    configuration_allowlist=None, max_completed_configurations=None,
    max_cumulative_spend_cny=None, inherited_cost_cny="0",
    inherited_cost_source_sha256=None, execution_profile="phase12_v7",
):
    """Freeze a phase allowlist while preserving the legacy phase-12 contract."""
    phase20 = configuration_allowlist is not None
    configurations = PHASE_CONFIGURATIONS if not phase20 else tuple(configuration_allowlist)
    max_completed = (
        MAX_PHASE_CONFIGURATIONS
        if max_completed_configurations is None else max_completed_configurations
    )
    max_spend = (
        MAX_PHASE_SPEND_CNY
        if max_cumulative_spend_cny is None else max_cumulative_spend_cny
    )
    if type(max_completed) is not int or max_completed < 1:
        raise ValueError("phase configuration limit must be a positive integer")
    if not isinstance(configurations, tuple) or not configurations:
        raise ValueError("phase configuration allowlist must be a non-empty tuple")
    by_key = {(job["dataset"], job["horizon"]): job for job in jobs}
    if len(by_key) != len(jobs):
        raise ValueError("live job manifest contains duplicate dataset/horizon configurations")
    missing = [key for key in configurations if key not in by_key]
    if missing:
        raise ValueError(f"phase allowlist configurations are missing from the verified plan: {missing}")

    expected = {
        "schema_version": 1,
        "run_id": run_id,
        "max_completed_configurations": max_completed,
        "ordered_configurations": [
            {
                "dataset": dataset,
                "horizon": horizon,
                "grid_job_id": by_key[(dataset, horizon)]["run_id"],
            }
            for dataset, horizon in configurations
        ],
    }
    if phase20:
        expected.update({
            "execution_profile": execution_profile,
            "max_cumulative_spend_cny": max_spend,
            "inherited_cost_cny": str(inherited_cost_cny),
            "inherited_cost_source_sha256": inherited_cost_source_sha256,
        })
    else:
        expected["max_new_spend_cny"] = max_spend

    checkpoint = (
        _load_json(checkpoint_path)
        if Path(checkpoint_path).is_file()
        else {"jobs": {}}
    )
    grid_records = checkpoint.get("jobs", {})
    if not isinstance(grid_records, dict):
        raise ValueError("grid checkpoint has invalid job records")
    allowed_ids = {item["grid_job_id"] for item in expected["ordered_configurations"]}
    completed_ids = {
        job_id for job_id, record in grid_records.items()
        if isinstance(record, dict) and record.get("status") == "completed"
    }
    completed_outside = sorted(completed_ids - allowed_ids)
    if completed_outside:
        raise RuntimeError(
            "completed checkpoint configurations exceed the approved phase-20 scope"
            if phase20 else "completed checkpoint configurations exceed the approved 12-config scope"
        )
    if len(completed_ids) > max_completed:
        raise RuntimeError("completed configuration count already exceeds the approved phase limit")

    path = Path(path)
    spend_path = Path(spend_path)
    if not spend_path.is_file() and (
        completed_ids or _run_has_prior_provider_activity(checkpoint_path)
    ):
        raise RuntimeError(
            "cannot initialize an empty cumulative CNY ledger after prior jobs or provider activity; "
            "restore or reconcile the original spend ledger before resuming"
        )
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        if path.exists():
            if not resume:
                raise FileExistsError(
                    "phase-20 policy already exists; resume the same authorized phase"
                    if phase20 else "phase-12 policy already exists; resume the same authorized phase"
                )
            saved = _load_json(path)
            if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError("saved phase-12 policy differs from the authorized allowlist or spend cap")
            if type(saved.get("budget_ledger_initialized")) is not bool:
                raise ValueError("saved phase-12 policy is missing its budget-ledger state")
            if saved["budget_ledger_initialized"] and not spend_path.is_file():
                raise RuntimeError("cumulative CNY ledger is missing after initialization; refusing to reset it")
            return saved

        if spend_path.exists():
            raise RuntimeError("cumulative CNY ledger exists but the phase policy is missing; refusing reinitialization")
        if resume and not Path(checkpoint_path).is_file():
            raise FileNotFoundError("cannot establish the phase-12 allowlist without the existing grid checkpoint")
        policy = {
            **expected,
            "initialized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completed_job_ids_at_initialization": sorted(completed_ids),
            "budget_ledger_initialized": False,
        }
        _atomic_json(path, policy)
        return policy


def _phase20_inherited_usage_basis():
    """Recompute known cost and conservatively reserve the unmatched historical retry."""
    usage_path = LEGACY_COMPLETED_RUN_DIR / "usage.jsonl"
    if not usage_path.is_file():
        raise FileNotFoundError("completed 12-config usage ledger is missing")
    source_sha256 = _sha256_file(usage_path)
    if source_sha256 != PHASE20_INHERITED_USAGE_SHA256:
        raise ValueError("completed 12-config usage ledger changed after the phase-20 plan was frozen")
    attempts_path = LEGACY_COMPLETED_RUN_DIR / "http_attempts.json"
    request_cache_path = LEGACY_COMPLETED_RUN_DIR / "request_cache.json"
    manifest_path = LEGACY_COMPLETED_RUN_DIR / "run_manifest.json"
    attempts_ledger = _load_json(attempts_path)
    request_cache = _load_json(request_cache_path)
    source_manifest = _load_json(manifest_path)
    request_settings = _request_settings()
    request_settings_sha256 = _hash_json(request_settings)
    if source_manifest.get("request_settings_sha256") != request_settings_sha256:
        raise ValueError("historical retry reserve cannot use changed request settings")

    total_micros = 0
    event_ids = set()
    count = 0
    usage_events = []
    with usage_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"inherited usage ledger has invalid JSON at line {line_number}") from None
            if event.get("http_status") != 200 or event.get("model") != "deepseek-flash":
                raise ValueError("inherited usage ledger contains an unsupported HTTP event")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise ValueError("inherited usage ledger event IDs are missing or duplicated")
            event_ids.add(event_id)
            actual_micros, _normalized = _peak_cost_micros(event.get("usage"))
            total_micros += actual_micros
            count += 1
            usage_events.append(event)
    if count != PHASE20_INHERITED_HTTP_200_EVENTS or total_micros != 8_098_836:
        raise ValueError("inherited usage count or peak-rate cost differs from the frozen phase-20 basis")
    attempts_used = attempts_ledger.get("attempts_used")
    reserve = _estimate_unmatched_http_attempt_reserve(
        usage_events=usage_events,
        attempts_used=attempts_used,
        request_cache=request_cache,
        max_output_tokens=request_settings["output_limit"],
    )
    inherited_micros = total_micros + reserve["reserve_micros"]
    basis_sha256 = _hash_json({
        "usage_sha256": source_sha256,
        "http_attempts_sha256": _sha256_file(attempts_path),
        "request_cache_sha256": _sha256_file(request_cache_path),
        "request_settings_sha256": request_settings_sha256,
        "known_usage_cost_micros": total_micros,
        "unmatched_attempt_reserve": reserve,
        "pricing_version": _CNY_BUDGET_PRICING_VERSION,
    })

    def cny(micros):
        return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"

    return {
        "cost_cny": cny(inherited_micros),
        "known_usage_cost_cny": PHASE20_INHERITED_SPEND_CNY,
        "unknown_attempt_reserve_cny": cny(reserve["reserve_micros"]),
        "unmatched_attempts": reserve["unmatched_attempts"],
        "max_observed_prompt_tokens": reserve["prompt_tokens_upper_bound"],
        "max_output_tokens": reserve["output_tokens_upper_bound"],
        "unknown_attempt_reserve_micros_per_attempt": reserve["reserve_micros_per_attempt"],
        "http_attempts": attempts_used,
        "http_200_events": count,
        "usage_sha256": source_sha256,
        "basis_sha256": basis_sha256,
        "http_attempts_sha256": _sha256_file(attempts_path),
        "request_cache_sha256": _sha256_file(request_cache_path),
        "request_settings_sha256": request_settings_sha256,
        "pricing_version": _CNY_BUDGET_PRICING_VERSION,
    }


def build_phase20_ab_dry_run_report(source_plan):
    """Validate the frozen 20-config scope without provider, embedding, or Qdrant work."""
    plan = select_phase20_ab_plan(source_plan)
    basis = _phase20_inherited_usage_basis()
    source_manifest = _load_json(LEGACY_COMPLETED_RUN_DIR / "run_manifest.json")
    template_hashes = {}
    frozen_template_dir = LEGACY_COMPLETED_RUN_DIR / "templates"
    for dataset_name in sorted({row["dataset"] for row in plan["plans"]}):
        template_path, template_value, template_status = _resolve_template(
            dataset_name, frozen_template_dir
        )
        if (
            template_status != "source_template"
            or template_path is None
            or not template_path.resolve().is_relative_to(frozen_template_dir.resolve())
        ):
            raise ValueError(f"phase-20 requires a frozen source template for {dataset_name}")
        expected_file_hash = plan.get("summary_cache", {}).get("template_fingerprints", {}).get(dataset_name)
        if expected_file_hash != _sha256_file(template_path):
            raise ValueError(f"phase-20 plan template fingerprint mismatch for {dataset_name}")
        template_hash = _hash_json(template_value)
        if source_manifest.get("template_hashes", {}).get(dataset_name) != template_hash:
            raise ValueError(f"frozen template hash mismatch for {dataset_name}")
        template_hashes[dataset_name] = template_hash
    cached_responses = _load_json(LEGACY_COMPLETED_RUN_DIR / "request_cache.json")
    cache_records = cached_responses.get("records")
    if not isinstance(cache_records, dict):
        raise ValueError("completed request cache has no validated record map")
    fit_count = sum(row["fit_case_count"] for row in plan["plans"])
    validation_count = sum(row["validation_window_count"] for row in plan["plans"])
    return {
        "status": "dry_run",
        "execution_profile": plan["execution_profile"],
        "forecast_arms": plan["forecast_arms"],
        "completed_configurations_reused": len(PHASE_CONFIGURATIONS),
        "new_configurations": [
            {"dataset": dataset, "horizon": horizon}
            for dataset, horizon in PHASE20_NEW_CONFIGURATIONS
        ],
        "total_configuration_target": plan["phase20_total_configuration_count"],
        "new_fit_memberships": fit_count,
        "new_validation_windows": validation_count,
        "uncached_logical_call_upper_bound": plan["phase20_uncached_logical_call_bound"],
        "frozen_template_sha256": template_hashes,
        "read_only_request_cache_path": str(LEGACY_COMPLETED_RUN_DIR / "request_cache.json"),
        "read_only_request_cache_record_count": len(cache_records),
        "inherited_usage_basis": basis,
        "cumulative_budget_cap_cny": PHASE20_CUMULATIVE_SPEND_CNY,
        "remaining_budget_before_new_requests_cny": round(
            PHASE20_CUMULATIVE_SPEND_CNY - float(basis["cost_cny"]), 6
        ),
        "qdrant_collections_to_create": len(PHASE20_NEW_CONFIGURATIONS),
        "provider_requests_made": 0,
        "embedding_inferences_made": 0,
        "qdrant_writes_made": 0,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "known_limitations": [
            "Existing 12-config measurements are reused; the remaining 8 configurations are a development extension.",
            "Dataset target semantics and news provenance remain partially unverified.",
            "The 15 CNY limit is a peak-tariff estimate enforced per physical HTTP attempt; it is not the provider invoice.",
        ],
    }


def _template_for_job(
    *, dataset, row, run_dir, manifest, request_cache, call_json,
    retry_pending, template_dir=None,
):
    templates_dir = Path(run_dir) / "templates"
    template_path = templates_dir / f"{_safe_slug(dataset)}.json"
    source_path, source_template, source_status = _resolve_template(dataset, template_dir)
    prompt = hic.build_template_prompt(_dataset_description(dataset))

    if template_path.exists():
        template = _load_json(template_path)
        if not isinstance(template, dict) or not template:
            raise ValueError(f"saved template for {dataset} is not a non-empty JSON object")
        actual_hash = _hash_json(template)
        expected_hash = manifest.get("template_hashes", {}).get(dataset)
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(f"saved template for {dataset} changed after the run was frozen")
        if expected_hash is None and source_template is not None and template != source_template:
            raise ValueError(f"unrecorded template for {dataset} does not match its verified source")
        if expected_hash is None and source_template is None:
            cached = request_cache.invoke(
                f"{manifest['run_id']}:template:{dataset}", prompt,
                call_json=call_json,
                validator=lambda value: isinstance(value, dict) and bool(value),
                retry_unresolved=retry_pending,
            )
            if cached["response"] != template:
                raise ValueError(f"saved template for {dataset} differs from its exact cached response")
            request_cache.mark_persisted(cached["request_identity"]["sha256"])
        manifest.setdefault("template_hashes", {})[dataset] = actual_hash
        _write_manifest(_run_paths(run_dir)["manifest"], manifest)
        return template, {
            "path": str(template_path),
            "sha256": actual_hash,
            "source": "run_snapshot",
            "source_status": source_status,
        }

    if source_template is None:
        result = request_cache.invoke(
            f"{manifest['run_id']}:template:{dataset}", prompt,
            call_json=call_json,
            validator=lambda value: isinstance(value, dict) and bool(value),
            retry_unresolved=retry_pending,
        )
        template = result["response"]
    else:
        template = source_template

    if not isinstance(template, dict) or not template:
        raise ValueError(f"template for {dataset} must be a non-empty JSON object")
    _atomic_json(template_path, template)
    template_hash = _hash_json(template)
    manifest.setdefault("template_hashes", {})[dataset] = template_hash
    _write_manifest(_run_paths(run_dir)["manifest"], manifest)
    if source_template is None:
        request_cache.mark_persisted(result["request_identity"]["sha256"])
    return template, {
        "path": str(template_path),
        "sha256": template_hash,
        "source": str(source_path) if source_path else "generated_by_deepseek",
        "source_status": source_status,
    }


def _aggregate_grid(grid, jobs, usage_summary, *, request_cache, attempt_budget):
    configurations = []
    selection_configurations = []
    confirmation_configurations = []
    detailed = []
    completed_jobs = [
        job for job in jobs
        if grid.get("jobs", {}).get(job["run_id"], {}).get("status") == "completed"
    ]
    for job in completed_jobs:
        output = grid["jobs"][job["run_id"]]["output"]
        metrics = output["validation_metrics"]
        configurations.append({"dataset": job["dataset"], "arm_metrics": metrics})
        if output.get("selection_metrics"):
            selection_configurations.append({
                "dataset": job["dataset"],
                "arm_metrics": output["selection_metrics"],
            })
        if output.get("confirmation_metrics"):
            confirmation_configurations.append({
                "dataset": job["dataset"],
                "arm_metrics": output["confirmation_metrics"],
            })
        detailed.append({
            "dataset": job["dataset"],
            "horizon": job["horizon"],
            "run_id": job["run_id"],
            "fit_memory_count": output["fit_memory_count"],
            "validation_window_count": len(output["validation_windows"]),
            "validation_metrics": metrics,
            "selection_metrics": output.get("selection_metrics"),
            "confirmation_metrics": output.get("confirmation_metrics"),
            "confirmation_metrics_excluding_recorded_exposures": output.get(
                "confirmation_metrics_excluding_recorded_exposures"
            ),
            "selection_confirmation_split": output.get("selection_confirmation_split"),
            "validation_fusion": output["validation_fusion"],
            "validation_uncertainty": output["validation_uncertainty"],
            "report_path": output.get("report_path"),
            "qdrant_collection": output.get("qdrant_collection"),
            "logical_calls_this_configuration": output["logical_calls_this_configuration"],
        })

    available_arms = set(completed_jobs[0]["run_id"] and
        grid["jobs"][completed_jobs[0]["run_id"]]["output"].get("arms", [])) if completed_jobs else set()
    primary_ab_only = available_arms == {"no_retrieval", "full_reasoning"}
    arms = (
        ("full_reasoning",) if primary_ab_only
        else ("summary_retrieval", "full_reasoning", "low_similarity_control", "validation_blend")
    )
    comparators = (
        ("no_retrieval",) if primary_ab_only
        else (
            "no_retrieval", "low_similarity_control", "persistence", "history_mean",
            "recent_mean_3", "linear_trend",
        )
    )
    report_metrics = tuple(
        metric for metric in ("mae", "mse", "rmse")
        if all(
            metric in configuration["arm_metrics"].get(arm, {})
            for configuration in configurations
            for arm in configuration["arm_metrics"]
        )
    )

    def compare(config_rows):
        comparisons = {}
        if not config_rows:
            return comparisons
        for arm in arms:
            for comparator in comparators:
                if arm == comparator:
                    continue
                if all(
                    configuration["arm_metrics"].get(arm, {}).get(metric, 0) > 0
                    and configuration["arm_metrics"].get(comparator, {}).get(metric, 0) > 0
                    for configuration in config_rows
                    for metric in report_metrics
                ):
                    comparisons[f"{arm}_vs_{comparator}"] = {
                        metric: macro_relative_improvement(
                            config_rows, arm=arm, comparator=comparator, metric=metric
                        )
                        for metric in report_metrics
                    }
        return comparisons

    comparisons = compare(configurations)
    selection_comparisons = compare(selection_configurations)
    confirmation_comparisons = compare(confirmation_configurations)

    return {
        "schema_version": 1,
        "status": "phase_complete",
        "configuration_count": grid["job_count"],
        "completed_configuration_count": grid["completed_count"],
        "allowlisted_configuration_count": len(grid.get("execution_job_ids", [])),
        "completed_configuration_reports_included": len(completed_jobs),
        "whole_corpus_complete": grid["completed_count"] == grid["job_count"],
        "forecast_arms": sorted(available_arms),
        "outer_test_targets_read": False,
        "aggregation_note": (
            "Macro relative improvement gives equal weight to dataset families and averages "
            "horizons within each family first. It is descriptive; overlapping windows and "
            "single-series families are not independent evidence. Per-config paired block "
            "bootstrap intervals remain in the detailed results."
        ),
        "macro_relative_improvement": comparisons,
        "selection_macro_relative_improvement": selection_comparisons,
        "confirmation_macro_relative_improvement": confirmation_comparisons,
        "configurations": detailed,
        "usage": usage_summary,
        "request_cache_reuse": request_cache.reuse_summary(),
        "logical_calls_used": request_cache.logical_calls,
        "max_logical_calls": request_cache.max_logical_calls,
        "http_attempts_used": attempt_budget.attempts_used,
        "max_http_attempts": attempt_budget.max_attempts,
    }


def _prepare_manifest(
    *, paths, plan, run_id, settings, max_logical_calls, max_http_attempts,
    resume, request_cache_reuse_sources=(),
):
    plan_sha = _hash_json(plan)
    settings_sha = _hash_json(settings)
    request_cache_reuse_sources = copy.deepcopy(list(request_cache_reuse_sources or ()))
    if resume:
        required = ("manifest", "plan", "grid_checkpoint", "http_attempts")
        missing = [name for name in required if not paths[name].is_file()]
        if missing:
            raise FileNotFoundError(f"resume is missing required run artifacts: {', '.join(missing)}")
        saved_plan = _load_json(paths["plan"])
        if _hash_json(saved_plan) != plan_sha:
            raise ValueError("resume plan differs from the original frozen development plan")
        manifest = _load_json(paths["manifest"])
        expected = {
            "schema_version": 1,
            "pipeline_version": VERIFIED_PIPELINE_VERSION,
            "validation_truth_policy": VALIDATION_TRUTH_POLICY,
            "target_contract_registry_sha256": plan.get("target_contract_registry_sha256"),
            "evaluation_scope": plan.get("evaluation_scope"),
            "real_world_target_semantics_verified": plan.get(
                "real_world_target_semantics_verified"
            ),
            "run_id": run_id,
            "plan_sha256": plan_sha,
            "request_settings_sha256": settings_sha,
            "max_logical_calls": max_logical_calls,
            "max_http_attempts": max_http_attempts,
            "request_cache_reuse_sources": request_cache_reuse_sources,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"resume manifest does not match {key}")
        grid = _load_json(paths["grid_checkpoint"])
        job_records = grid.get("jobs", {})
        started_jobs = any(
            isinstance(record, dict) and record.get("status") != "planned"
            for record in job_records.values()
        )
        if (
            started_jobs or manifest.get("template_hashes")
        ) and not paths["request_cache"].is_file():
            raise RuntimeError(
                "cannot resume safely: request cache is missing after run progress"
            )
        return manifest

    if paths["root"].exists():
        raise FileExistsError("run directory already exists; use --resume or choose a new --run-id")
    paths["root"].mkdir(parents=True)
    paths["templates"].mkdir()
    paths["jobs"].mkdir()
    _atomic_json(paths["plan"], plan)
    manifest = {
        "schema_version": 1,
        "pipeline_version": VERIFIED_PIPELINE_VERSION,
        "validation_truth_policy": VALIDATION_TRUTH_POLICY,
        "target_contract_registry_sha256": plan.get("target_contract_registry_sha256"),
        "evaluation_scope": plan.get("evaluation_scope"),
        "real_world_target_semantics_verified": plan.get(
            "real_world_target_semantics_verified"
        ),
        "run_id": run_id,
        "plan_sha256": plan_sha,
        "request_settings_sha256": settings_sha,
        "max_logical_calls": max_logical_calls,
        "max_http_attempts": max_http_attempts,
        "request_cache_reuse_sources": request_cache_reuse_sources,
        "template_hashes": {},
        "outer_test_targets_read": False,
    }
    _write_manifest(paths["manifest"], manifest)
    return manifest


def _record_preflight_failure(paths, *, run_id, error):
    """Persist a safe failure note when local setup fails before grid execution."""
    if not paths["manifest"].is_file() or paths["summary"].exists():
        return
    logical_calls = 0
    if paths["request_cache"].is_file():
        logical_calls = _load_json(paths["request_cache"]).get("logical_calls", 0)
    http_attempts = 0
    if paths["http_attempts"].is_file():
        http_attempts = _load_json(paths["http_attempts"]).get("attempts_used", 0)
    usage = summarize_usage_ledger(paths["usage"]) if paths["usage"].is_file() else _empty_usage_summary()
    _atomic_json(paths["summary"], {
        "schema_version": 1,
        "status": "preflight_failed",
        "run_id": run_id,
        "run_directory": str(paths["root"]),
        "error_type": type(error).__name__,
        "error": str(error)[:500],
        "logical_calls_used": logical_calls,
        "http_attempts_used": http_attempts,
        "usage": usage,
        "outer_test_targets_read": False,
    })


def _build_live_plan(plan, jobs):
    live_plan = copy.deepcopy(plan)
    run_ids = {(job["dataset"], job["horizon"]): job["run_id"] for job in jobs}
    for row in live_plan["plans"]:
        if row.get("status") == "development_ready":
            row["run_id"] = run_ids[(row["dataset"], row["horizon"])]
    live_plan["run_tag"] = f"live-{jobs[0]['run_id'][:24]}"
    return live_plan


@_serialize_phase_execution
def execute_full_corpus_development(
    *, plan, run_id, run_dir, env_file, qdrant_url="http://127.0.0.1:6333",
    max_logical_calls=MAX_AUTHORIZED_LOGICAL_CALLS,
    max_http_attempts=MAX_AUTHORIZED_HTTP_ATTEMPTS,
    data_dir=None, template_dir=None, resume=False, retry_pending=False,
    reuse_request_cache_from=(), dataset=None, horizon=None, phase20_ab=False,
):
    """Run the legacy 12-config phase or the approved 20-config A/B extension."""
    if not isinstance(phase20_ab, bool):
        raise ValueError("phase20_ab must be a boolean")
    validate_authorized_budgets(max_logical_calls, max_http_attempts)
    validate_plan_call_budget(plan, max_logical_calls)
    validate_plan_protocol(plan)
    if phase20_ab:
        template_dir = LEGACY_COMPLETED_RUN_DIR / "templates"
    run_tag = plan.get("run_tag")
    if not isinstance(run_tag, str):
        raise ValueError("development plan has no run_tag")
    current_plan = build_phase2_grid_plan(
        data_dir=data_dir, template_dir=template_dir, run_tag=run_tag,
    )
    if phase20_ab:
        expected_template_dir = LEGACY_COMPLETED_RUN_DIR / "templates"
        if template_dir is not None and Path(template_dir).resolve() != expected_template_dir.resolve():
            raise ValueError("phase-20 execution must use the frozen completed-run templates")
        verify_phase20_source_plan(plan, current_plan)
        build_phase20_ab_dry_run_report(current_plan)
        plan = current_plan
    else:
        verify_plan_is_current(plan, current_plan)
    if dataset is not None or horizon is not None:
        raise ValueError("dataset/horizon filters are supported only for offline dry-runs")
    if phase20_ab:
        plan = select_phase20_ab_plan(plan)
        validate_plan_call_budget(plan, max_logical_calls)
        inherited_basis = _phase20_inherited_usage_basis()
        plan["inherited_budget_basis"] = inherited_basis
    else:
        inherited_basis = None
        plan = select_development_plan(plan, dataset=None, horizon=None)
    if retry_pending and not resume:
        raise ValueError("--retry-pending is allowed only together with --resume")
    if hic.MODEL != "deepseek-flash":
        raise ValueError("the approved run is restricted to the configured deepseek-flash model")
    if plan.get("outer_test_targets_read") is not False:
        raise ValueError("the development plan must affirm outer-test targets were not read")

    env_file = Path(env_file)
    if not env_file.is_file():
        raise FileNotFoundError(
            "DeepSeek env file was not found; pass --env-file pointing to the local api.env file."
        )
    load_dotenv(env_file, override=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        keys = sorted(dotenv_values(env_file).keys())
        raise RuntimeError(
            "DEEPSEEK_API_KEY is absent from the selected env file "
            f"(variable names found: {keys}); no secret value was displayed."
        )
    hic.DEEPSEEK_API_KEY = api_key

    settings = _request_settings()
    paths = _run_paths(run_dir, phase20_ab=phase20_ab)
    jobs = build_live_job_manifest(plan, run_id=run_id)
    live_plan = _build_live_plan(plan, jobs)
    row_by_key = {
        (row["dataset"], row["horizon"]): row
        for row in plan["plans"]
        if row.get("status") == "development_ready"
    }
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise RuntimeError("qdrant-client is required; install the project-pinned dependency first") from None
    client = QdrantClient(url=qdrant_url, timeout=QDRANT_TIMEOUT_SECONDS)
    existing = {item.name for item in client.get_collections().collections}
    collection_names = {
        job["job_id"]: collection_name_for("BAAI/bge-m3", job["collection_suffix"])
        for job in jobs
    }
    if not resume:
        collisions = sorted(existing.intersection(collection_names.values()))
        if collisions:
            raise FileExistsError(
                "isolated Qdrant collection name already exists; choose a fresh run ID: "
                + ", ".join(collisions)
            )
    else:
        # A missing collection can be recreated only when its checkpoint has no ingested fit case.
        for job in jobs:
            name = collection_names[job["job_id"]]
            checkpoint = paths["jobs"] / f"{job['job_id']}.checkpoint.json"
            if name not in existing and checkpoint.is_file():
                state = _load_json(checkpoint)
                if any(value.get("ingested") for value in state.get("fit", {}).values()):
                    raise RuntimeError(f"resume Qdrant collection {name} is missing fit memories")

    # Load only already-cached weights. A missing BGE-M3 snapshot fails before any provider call.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        embedder = hic.get_embedder()
    except Exception:
        client.close()
        raise
    if hic.BGE_EMBEDDING_DIMENSION != 1024 or embedder.max_length < 2048:
        client.close()
        raise RuntimeError("local BAAI/bge-m3 configuration does not match the frozen 1024/8192 setup")
    for module_name in ("transformers", "torch"):
        if module_name not in sys.modules:
            client.close()
            raise RuntimeError(f"local embedding dependency {module_name} did not load")

    # Validate request-cache settings before creating any fresh run artifacts.
    if phase20_ab and not reuse_request_cache_from:
        legacy_cache = LEGACY_COMPLETED_RUN_DIR / "request_cache.json"
        if not legacy_cache.is_file():
            raise FileNotFoundError("completed 12-config request cache is missing")
        reuse_request_cache_from = (legacy_cache,)
    request_cache = RequestCache(
        paths["request_cache"],
        request_settings=settings,
        max_logical_calls=max_logical_calls,
        reuse_cache_paths=reuse_request_cache_from,
    )

    # Create durable run artifacts only after local Qdrant and offline model preflight pass.
    manifest = _prepare_manifest(
        paths=paths, plan=plan, run_id=run_id, settings=settings,
        max_logical_calls=max_logical_calls,
        max_http_attempts=max_http_attempts,
        resume=resume,
        request_cache_reuse_sources=request_cache.reuse_sources,
    )
    if phase20_ab:
        source_manifest = _load_json(LEGACY_COMPLETED_RUN_DIR / "run_manifest.json")
        frozen_hashes = source_manifest.get("template_hashes", {})
        for dataset_name in sorted({job["dataset"] for job in jobs}):
            template_path = template_dir / f"{_safe_slug(dataset_name)}.json"
            template_value = _load_json(template_path)
            template_hash = _hash_json(template_value)
            if frozen_hashes.get(dataset_name) != template_hash:
                raise ValueError(f"frozen template hash mismatch for {dataset_name}")
            saved_hash = manifest.setdefault("template_hashes", {}).get(dataset_name)
            if saved_hash not in (None, template_hash):
                raise ValueError(f"phase-20 template snapshot changed for {dataset_name}")
            manifest["template_hashes"][dataset_name] = template_hash
        _write_manifest(paths["manifest"], manifest)
    phase_policy = _prepare_phase12_policy(
        path=paths["phase_policy"],
        spend_path=paths["cny_spend"],
        jobs=jobs,
        run_id=run_id,
        checkpoint_path=paths["grid_checkpoint"],
        resume=resume,
        **({
            "configuration_allowlist": PHASE20_NEW_CONFIGURATIONS,
            "max_completed_configurations": len(PHASE20_NEW_CONFIGURATIONS),
            "max_cumulative_spend_cny": PHASE20_CUMULATIVE_SPEND_CNY,
            "inherited_cost_cny": inherited_basis["cost_cny"],
            "inherited_cost_source_sha256": inherited_basis["basis_sha256"],
            "execution_profile": plan["execution_profile"],
        } if phase20_ab else {}),
    )
    execution_job_ids = [
        item["grid_job_id"] for item in phase_policy["ordered_configurations"]
    ]

    usage_ledger = DeepSeekUsageLedger(paths["usage"])
    attempt_budget = make_http_attempt_budget(
        max_attempts=max_http_attempts,
        ledger_path=paths["http_attempts"],
        resume=resume,
    )
    cny_ledger_exists = paths["cny_spend"].is_file()
    if phase_policy["budget_ledger_initialized"] and not cny_ledger_exists:
        raise RuntimeError("cumulative CNY ledger disappeared; refusing to reset the authorized phase budget")
    spend_budget = DeepSeekCNYSpendBudget(
        paths["cny_spend"],
        max_spend_cny=(
            PHASE20_CUMULATIVE_SPEND_CNY if phase20_ab else MAX_PHASE_SPEND_CNY
        ),
        resume=cny_ledger_exists,
        inherited_cost_cny=(inherited_basis["cost_cny"] if phase20_ab else "0"),
        inherited_cost_source_sha256=(
            inherited_basis["basis_sha256"] if phase20_ab else None
        ),
        allow_initialize_on_resume=(
            resume and not phase_policy["budget_ledger_initialized"]
        ),
        allow_reconcile_unresolved=bool(resume and retry_pending),
    )
    if retry_pending:
        spend_budget.conservatively_charge_unknown_reservations()
    if not phase_policy["budget_ledger_initialized"]:
        policy_lock = paths["phase_policy"].with_name(paths["phase_policy"].name + ".lock")
        with exclusive_file_lock(policy_lock):
            saved_policy = _load_json(paths["phase_policy"])
            if saved_policy.get("run_id") != run_id:
                raise RuntimeError("shared phase policy changed while initializing the CNY ledger")
            saved_policy["budget_ledger_initialized"] = True
            saved_policy["budget_ledger_initialized_at_utc"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            _atomic_json(paths["phase_policy"], saved_policy)
            phase_policy = saved_policy

    def call_json(prompt, temperature=0.0):
        return hic.call_json(
            prompt,
            temperature=temperature,
            attempt_budget=attempt_budget,
            spend_budget=spend_budget,
            usage_callback=usage_ledger.record_http_200,
        )

    def embed_summary(summary, *, is_query):
        text = hic.summary_to_embedding_text(summary)
        vectors = embedder.encode([text], is_query=is_query)
        if len(vectors) != 1:
            raise RuntimeError("BAAI/bge-m3 did not return exactly one vector")
        return vectors[0]

    run_job_by_key = {
        (job["dataset"], job["horizon"]): job
        for job in jobs
    }

    def run_job(job):
        key = (job["dataset"], job["horizon"])
        live_job = run_job_by_key[key]
        source_row = row_by_key[key]
        template, template_info = _template_for_job(
            dataset=job["dataset"], row=source_row, run_dir=paths["root"],
            manifest=manifest, request_cache=request_cache, call_json=call_json,
            retry_pending=retry_pending, template_dir=template_dir,
        )
        collection_name = collection_names[live_job["job_id"]]
        store = QdrantHicStore(
            model_id="BAAI/bge-m3", collection_name=collection_name,
            client=client,
        )
        store.ensure_collection()
        config_checkpoint = paths["jobs"] / f"{live_job['job_id']}.checkpoint.json"
        result = _run_verified_configuration(
            dataset=job["dataset"],
            horizon=job["horizon"],
            run_id=job["run_id"],
            template=template,
            dataset_description=_dataset_description(job["dataset"]),
            request_cache=request_cache,
            call_json=call_json,
            embed_summary=embed_summary,
            store=store,
            checkpoint_path=config_checkpoint,
            data_dir=data_dir,
            source_sha256=job["source_sha256"],
            retry_pending=retry_pending,
            forecast_arms=(plan["forecast_arms"] if phase20_ab else None),
        )
        report_path = paths["jobs"] / f"{live_job['job_id']}.report.json"
        result.update({
            "source_sha256": job["source_sha256"],
            "target_contract_registry_sha256": plan["target_contract_registry_sha256"],
            "evaluation_scope": plan["evaluation_scope"],
            "real_world_target_semantics_verified": source_row["target_contract"][
                "real_world_target_semantics_verified"
            ],
            "target_contract": source_row["target_contract"],
            "template": template_info,
            "qdrant_collection": collection_name,
            "qdrant_count_verified": store.count(),
            "report_path": str(report_path),
        })
        _atomic_json(report_path, result)
        return result

    try:
        grid = run_development_grid(
            live_plan, run_job=run_job,
            checkpoint_path=paths["grid_checkpoint"],
            protocol_version=VALIDATION_TRUTH_POLICY,
            execution_job_ids=execution_job_ids,
            max_completed_jobs=(
                len(PHASE20_NEW_CONFIGURATIONS) if phase20_ab
                else MAX_PHASE_CONFIGURATIONS
            ),
        )
        usage_summary = summarize_usage_ledger(paths["usage"]) if paths["usage"].is_file() else _empty_usage_summary()
        summary = _aggregate_grid(
            grid, jobs, usage_summary,
            request_cache=request_cache, attempt_budget=attempt_budget,
        )
        summary.update({
            "run_id": run_id,
            "selected_scope": plan.get("selected_scope"),
            "source_configuration_count": plan.get("source_configuration_count"),
            "target_contract_registry_sha256": plan["target_contract_registry_sha256"],
            "evaluation_scope": plan["evaluation_scope"],
            "real_world_target_semantics_verified": plan[
                "real_world_target_semantics_verified"
            ],
            "target_contract_status_by_dataset": {
                row["dataset"]: row["target_contract"]["mapping_status"]
                for row in plan["plans"]
                if row.get("status") == "development_ready"
            },
            "plan_sha256": _hash_json(plan),
            "request_settings_sha256": _hash_json(settings),
            "run_directory": str(paths["root"]),
            "plan_estimated_logical_calls_with_summary_cache": plan.get(
                "logical_call_estimate_with_summary_cache", {}
            ).get("total"),
            "phase_policy": phase_policy,
            "cny_spend": spend_budget.summary(),
        })
        _atomic_json(paths["summary"], summary)
        client.close()
        return summary
    except Exception as error:
        usage_summary = summarize_usage_ledger(paths["usage"]) if paths["usage"].is_file() else _empty_usage_summary()
        checkpoint = _load_json(paths["grid_checkpoint"]) if paths["grid_checkpoint"].is_file() else {"jobs": {}}
        status = {
            "schema_version": 1,
            "status": "stopped_with_checkpoint",
            "run_id": run_id,
            "run_directory": str(paths["root"]),
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "completed_configuration_count": sum(
                record.get("status") == "completed"
                for record in checkpoint.get("jobs", {}).values()
            ),
            "configuration_count": len(jobs),
            "allowlisted_configuration_count": len(execution_job_ids),
            "logical_calls_used": request_cache.logical_calls,
            "max_logical_calls": request_cache.max_logical_calls,
            "http_attempts_used": attempt_budget.attempts_used,
            "max_http_attempts": attempt_budget.max_attempts,
            "usage": usage_summary,
            "request_cache_reuse": request_cache.reuse_summary(),
            "phase_policy": phase_policy,
            "cny_spend": spend_budget.summary(),
            "outer_test_targets_read": False,
        }
        _atomic_json(paths["summary"], status)
        client.close()
        raise


def _load_and_verify_plan(plan_path, *, data_dir=None, template_dir=None):
    saved = _load_json(plan_path)
    run_tag = saved.get("run_tag")
    if not isinstance(run_tag, str):
        raise ValueError("saved plan has no run_tag")
    current = build_phase2_grid_plan(
        data_dir=data_dir,
        template_dir=template_dir,
        run_tag=run_tag,
    )
    verify_plan_is_current(saved, current)
    return current


def _load_phase20_source_plan(plan_path, *, data_dir=None, template_dir=None):
    saved = _load_json(plan_path)
    run_tag = saved.get("run_tag")
    if not isinstance(run_tag, str):
        raise ValueError("saved plan has no run_tag")
    expected_template_dir = LEGACY_COMPLETED_RUN_DIR / "templates"
    if template_dir is not None and Path(template_dir).resolve() != expected_template_dir.resolve():
        raise ValueError("phase-20 preflight must use the frozen completed-run templates")
    current = build_phase2_grid_plan(
        data_dir=data_dir,
        template_dir=expected_template_dir,
        run_tag=run_tag,
    )
    verify_phase20_source_plan(saved, current)
    return current


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-id", default="hic_full_dev_20261019_01")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / "config" / "api.env")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--dataset", help="filter an offline dry-run by dataset")
    parser.add_argument("--horizon", type=int, help="filter an offline dry-run by horizon")
    parser.add_argument(
        "--reuse-request-cache-from", action="append", type=Path, default=[],
        help="read exact validated responses from this cache without modifying it; repeatable",
    )
    parser.add_argument("--max-logical-calls", type=int, default=MAX_AUTHORIZED_LOGICAL_CALLS)
    parser.add_argument("--max-http-attempts", type=int, default=MAX_AUTHORIZED_HTTP_ATTEMPTS)
    parser.add_argument("--execute", action="store_true", help="run live provider, local embedder, and Qdrant work")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-pending", action="store_true", help="explicitly retry unresolved requests during --resume")
    parser.add_argument(
        "--phase20-ab", action="store_true",
        help="use the fixed 8-config, no-retrieval vs full-HIC extension under the cumulative CNY 15 cap",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_id = (
        PHASE20_RUN_ID
        if args.phase20_ab and args.run_id == "hic_full_dev_20261019_01"
        else args.run_id
    )
    run_dir = args.run_dir or (EVALUATIONS / run_id)
    try:
        validate_authorized_budgets(args.max_logical_calls, args.max_http_attempts)
        plan_path = args.plan
        template_dir = args.template_dir
        if args.phase20_ab and args.plan == DEFAULT_PLAN:
            plan_path = LEGACY_COMPLETED_RUN_DIR / "plan.json"
        if args.phase20_ab and template_dir is None:
            template_dir = LEGACY_COMPLETED_RUN_DIR / "templates"
        plan = (
            _load_phase20_source_plan(
                plan_path, data_dir=args.data_dir, template_dir=template_dir
            )
            if args.phase20_ab else
            _load_and_verify_plan(
                plan_path, data_dir=args.data_dir, template_dir=template_dir
            )
        )
        if not args.execute:
            if args.phase20_ab:
                print(json.dumps(
                    build_phase20_ab_dry_run_report(plan),
                    ensure_ascii=False,
                    indent=2,
                ))
                return 0
            selected_plan = select_development_plan(
                plan, dataset=args.dataset, horizon=args.horizon
            )
            print(json.dumps(build_dry_run_report(selected_plan, run_id=run_id), ensure_ascii=False, indent=2))
            return 0
        result = execute_full_corpus_development(
            plan=plan,
            run_id=run_id,
            run_dir=run_dir,
            env_file=args.env_file,
            qdrant_url=args.qdrant_url,
            max_logical_calls=args.max_logical_calls,
            max_http_attempts=args.max_http_attempts,
            data_dir=args.data_dir,
            template_dir=args.template_dir,
            resume=args.resume,
            retry_pending=args.retry_pending,
            reuse_request_cache_from=args.reuse_request_cache_from,
            dataset=args.dataset,
            horizon=args.horizon,
            phase20_ab=args.phase20_ab,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        diagnostic = {
            "status": "stopped",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }
        if args.execute:
            diagnostic["run_directory"] = str(run_dir)
        print(json.dumps({
            **diagnostic,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
