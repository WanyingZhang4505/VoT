"""Serial, truth-sealed Agent Test 6 runner with append-after-score HIC memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM.corpus_audit import DEFAULT_DATA_DIR
from LLM.hic import SUMMARY_PROMPT_VERSION, get_embedder, summary_to_embedding_text
from LLM.hic_agent_validation import (
    PROTOCOL, _summary_response, advance_agent_case,
    advance_test6_postscore_memory, canonical_hash,
)
from LLM.hic_qdrant_eval import _validated_forecast
from LLM.hic_codex_agent import (
    CodexTaskPending, CodexTaskQueue, CodexTaskRequestCache, _atomic_json,
)
from LLM.hic_verified_memory import (
    CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
    CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
    build_test6_transfer_projection, canonical_json,
    make_test6_outcome_blind_transfer_record,
)
from LLM.hic_agent_test6_reporting import write_checkpoint_report
from LLM.hic_vnext_batch import normalize_task_template
from LLM.hic_vnext_validation import validate_validation_window
from LLM.native_hic_data import (
    load_native_hic_splits, load_native_hic_validation_truth_for_case,
    target_metadata_for_dataset,
)
from LLM.native_hic_pilot import _dataset_description
from LLM.qdrant_store import QdrantHicStore


EXPERIMENT_ROOT = Path(os.environ.get(
    "HIC_AGENT_TEST_ROOT", str(ROOT / "LLM/evaluations/Agent Test 6")
)).resolve()
PREFLIGHT_ROOT = EXPERIMENT_ROOT / "preflight_20261029_0405"
SOURCE_EXPORT = PREFLIGHT_ROOT / "agent_test_3_original_export.json"
SOURCE_INDEX = ROOT / "LLM/evaluations/Agent Test 3/hic_agent_test3_frozen_batch6_20261025/combined_records.json"
SCOPE = (("SocialGood", 10, 13), ("SocialGood", 12, 11),
         ("Energy", 36, 5), ("Energy", 48, 4))
SOURCE_HORIZONS = {"SocialGood": (6, 8), "Energy": (12, 24)}
ROUND_COLLECTIONS = {round_id: "agent_test_6" for round_id in ("R0", "R1", "R2", "R3")}
MAX_TARGETED_REEVALUATIONS = 5
RUNNER_PROTOCOL = "agent_test6_serial_online_accumulation_v1"
SCALE_REVIEW_RATIO = 1000.0
TERMINAL_CASE_STATUSES = {"memory_appended", "technical_failure"}
TARGETBLIND_95_SCOPE = {
    ("Health", 12): 11, ("Health", 24): 4, ("Health", 48): 1,
    ("Security", 6): 7, ("Security", 8): 5, ("Security", 12): 3,
    ("new_metr", 48): 40, ("new_metr", 96): 19, ("new_metr", 336): 5,
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _source_hash(dataset):
    return hashlib.sha256((DEFAULT_DATA_DIR / f"{dataset}.csv").read_bytes()).hexdigest()


def _template(dataset, horizon, frequency):
    return normalize_task_template({
        "dataset_info": {
            "dataset_name": dataset,
            "target_variable": "OT (numeric target; exact interpretation unknown)",
            "target_unit": "unknown",
            "frequency": frequency,
            "data_modalities": "numeric time series, covariates when supplied, and dated text",
        },
        "predictive_factors": {
            "temporal_patterns": "Describe only visible direction, level, turning behavior, and volatility.",
            "covariates": "Use supplied covariates only; do not infer unavailable values.",
            "text_evidence": "Separate dated text from direct target observations and note relevance limits.",
            "uncertainty": "State unresolved evidence and forecast uncertainty without future claims.",
        },
        "forecast_outlook": {
            "expected_direction": "Concise expectation based only on pre-target evidence.",
            "key_signal": "Most relevant visible signal, or state evidence is insufficient.",
            "confidence": "Low, medium, or high with a short evidence-based reason.",
            "uncertainty": "Main ambiguity in the supplied history and text.",
            "horizon": f"The next {horizon} {frequency} target values.",
        },
    })


def prepare_round(round_id, *, output_dir=None):
    if round_id not in ROUND_COLLECTIONS:
        raise ValueError("round_id must be R0, R1, R2, or R3")
    output = Path(output_dir).resolve() if output_dir else EXPERIMENT_ROOT / round_id
    if not output.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError("round output must remain under Agent Test 6")
    if output.exists():
        raise FileExistsError(f"round output already exists; inspect and resume it: {output}")
    jobs, configurations = [], {}
    for dataset, horizon, expected_count in SCOPE:
        split = load_native_hic_splits(
            dataset, horizon=horizon, include_outer_test_targets=False,
            include_validation_truth=False,
        )
        if (split["validation_truth_read"] or split["test_windows"]
                or not split["outer_test_deferred"]):
            raise ValueError("target-blind scope guard failed")
        windows = split["validation_windows"]
        if len(windows) != expected_count:
            raise ValueError(f"expected {expected_count} cases for {dataset} H{horizon}")
        frequency = windows[0]["case"]["frequency"]
        contract = {
            "dataset": dataset, "horizon": horizon, "frequency": frequency,
            "target": "unknown", "unit": "unknown", "source_sha256": _source_hash(dataset),
            "test6_round_id": round_id,
        }
        configurations[f"{dataset}/H{horizon}"] = {
            "dataset": dataset, "horizon": horizon, "frequency": frequency,
            "contract": contract, "template": _template(dataset, horizon, frequency),
            "description": _dataset_description(dataset),
            "boundaries": split["boundaries"],
            "count": len(windows),
        }
        for window in windows:
            case = window["case"]
            validate_validation_window({"case": case}, expected_horizon=horizon)
            if "truth" in case or any(key in window for key in ("truth", "validation_truth")):
                raise ValueError("target-blind case unexpectedly contains validation truth")
            identity = {"round_id": round_id, "dataset": dataset,
                        "horizon": horizon, "case_id": case["case_id"]}
            jobs.append({
                **identity,
                "job_id": canonical_hash(identity),
                "configuration": f"{dataset}/H{horizon}",
                "case": case,
                "target_timestamps": window["target_timestamps"],
                "input_sha256": canonical_hash(case),
            })
    if len(jobs) != 33:
        raise ValueError(f"expected 33 tasks, got {len(jobs)}")
    manifest = {
        "run_id": f"agent_test_6_{round_id.lower()}",
        "round_id": round_id,
        "protocol": RUNNER_PROTOCOL,
        "paired_prompt_protocol": PROTOCOL,
        "collection_name": ROUND_COLLECTIONS[round_id],
        "embedding_model": "BAAI/bge-m3", "embedding_dimension": 1024,
        "source_index_sha256": hashlib.sha256(SOURCE_INDEX.read_bytes()).hexdigest(),
        "source_export_sha256": hashlib.sha256(SOURCE_EXPORT.read_bytes()).hexdigest(),
        "source_horizons": SOURCE_HORIZONS,
        "technical_review_thresholds": {
            "maximum_forecast_to_history_abs_ratio": SCALE_REVIEW_RATIO,
            "exact_cross_family_prediction_and_reasoning_match_requires_review": True,
        },
        "configurations": configurations,
        "jobs_sha256": canonical_hash(jobs),
        "job_count": len(jobs),
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "exposure_audit_complete": False,
        "raw_worker_smoke_complete": False,
        "r0_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "run_manifest.json", manifest)
    _write_new(output / "input_snapshots.json", jobs)
    _write_new(output / "progress.json", {
        "round_id": round_id, "status": "prepared_not_ready",
        "completed_forecasts": 0, "scored": 0, "memories_appended": 0,
        "total": 33, "next_case_id": jobs[0]["case_id"],
        "validation_truth_read": False, "outer_test_targets_read": False,
    })
    return output, manifest, jobs


def prepare_targetblind_95(
    target_manifest_path, *, output_dir, run_id=None, round_id=None,
    target_collection="agent_test_6", retrieval_mode="applicable",
):
    """Bind the existing immutable 95-case target-blind manifest to the resumable runner."""
    source_path = Path(target_manifest_path).resolve()
    output = Path(output_dir).resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError("95-case output must remain under Agent Test 6")
    if not output.is_dir():
        raise FileNotFoundError(f"pre-existing 95-case evidence directory is required: {output}")
    if not isinstance(target_collection, str) or not target_collection.strip():
        raise ValueError("target_collection must be non-empty text")
    if retrieval_mode not in {"applicable", "direct_nearest"}:
        raise ValueError("retrieval_mode must be applicable or direct_nearest")
    if not source_path.is_file():
        raise FileNotFoundError(f"target-blind manifest is missing: {source_path}")
    if (output / "run_manifest.json").exists() or (output / "input_snapshots.json").exists():
        raise FileExistsError("runner snapshot already exists; verify and resume instead of replacing it")
    source = _read(source_path)
    if (source.get("validation_truth_read") is not False
            or source.get("outer_test_targets_read") is not False
            or source.get("collection") != "agent_test_6"):
        raise ValueError("source manifest is not a sealed, target-blind Agent Test 6 manifest")
    source_jobs = source.get("jobs")
    if not isinstance(source_jobs, list) or len(source_jobs) != 95:
        raise ValueError("target-blind source manifest must contain exactly 95 jobs")
    observed = {}
    configurations = {}
    source_configurations = source.get("configurations")
    if not isinstance(source_configurations, list):
        raise ValueError("target-blind configuration inventory must be a list")
    config_by_key = {}
    for item in source_configurations:
        key = (item.get("dataset"), item.get("horizon"))
        if key in config_by_key:
            raise ValueError(f"duplicate target-blind configuration: {key}")
        config_by_key[key] = item
    if set(config_by_key) != set(TARGETBLIND_95_SCOPE):
        raise ValueError("target-blind configuration scope differs from the approved nine configurations")
    jobs = []
    source_run_id = source.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        raise ValueError("target-blind run_id is missing")
    run_id = run_id or source_run_id
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("runner run_id must be a non-empty string")
    round_id = round_id or "ONLINE95_20261030"
    if not isinstance(round_id, str) or not round_id.strip():
        raise ValueError("runner round_id must be a non-empty string")
    for dataset, horizon in TARGETBLIND_95_SCOPE:
        item = config_by_key[(dataset, horizon)]
        windows = [row for row in source_jobs
                   if row.get("dataset") == dataset and row.get("horizon") == horizon]
        if len(windows) != TARGETBLIND_95_SCOPE[(dataset, horizon)]:
            raise ValueError(f"unexpected case count for {dataset} H{horizon}")
        observed[(dataset, horizon)] = len(windows)
        frequency_values = {row.get("input", {}).get("frequency") for row in windows}
        if len(frequency_values) != 1 or None in frequency_values:
            raise ValueError(f"frequency is missing or inconsistent for {dataset} H{horizon}")
        frequency = next(iter(frequency_values))
        target_metadata = target_metadata_for_dataset(dataset)
        target = target_metadata.get("target", "unknown") if target_metadata else "unknown"
        unit = target_metadata.get("unit", "unknown") if target_metadata else "unknown"
        contract = {
            "dataset": dataset, "horizon": horizon, "frequency": frequency,
            "target": target, "unit": unit, "source_sha256": _source_hash(dataset),
            "test6_round_id": round_id,
        }
        configurations[f"{dataset}/H{horizon}"] = {
            "dataset": dataset, "horizon": horizon, "frequency": frequency,
            "contract": contract, "template": _template(dataset, horizon, frequency),
            "description": _dataset_description(dataset),
            "boundaries": item.get("boundaries", {}), "count": len(windows),
        }
        for row in windows:
            case = row.get("input")
            if not isinstance(case, dict) or row.get("truth_revealed") is not False:
                raise ValueError(f"case {row.get('case_id')} is not target-blind")
            if case.get("case_id") != row.get("case_id") or case.get("dataset") != dataset:
                raise ValueError("case identity does not match the target-blind manifest")
            if case.get("horizon") not in (None, horizon):
                raise ValueError("case input has conflicting horizon")
            snapshot_case = dict(case)
            snapshot_case["horizon"] = horizon
            validate_validation_window({"case": snapshot_case,
                                        "target_timestamps": row.get("target_timestamps")},
                                       expected_horizon=horizon)
            identity = {"round_id": round_id, "dataset": dataset,
                        "horizon": horizon, "case_id": row["case_id"]}
            jobs.append({
                **identity, "job_id": canonical_hash(identity),
                "configuration": f"{dataset}/H{horizon}",
                "case": snapshot_case,
                "target_timestamps": row["target_timestamps"],
                "input_sha256": canonical_hash(snapshot_case),
                "source_manifest_input_sha256": row["input_sha256"],
            })
    if len(jobs) != 95 or sum(observed.values()) != 95:
        raise ValueError("converted runner snapshot does not contain 95 approved cases")
    if source_path != (output / "target_blind_manifest.json").resolve():
        if hashlib.sha256(source_path.read_bytes()).digest() != hashlib.sha256(
                (output / "target_blind_manifest.json").read_bytes()).digest():
            raise ValueError("source manifest differs from the immutable run-directory manifest")
    manifest = {
        "run_id": run_id, "round_id": round_id,
        "protocol": ("agent_test7_direct_nearest_online_v1"
                     if target_collection == "agent_test_7" else RUNNER_PROTOCOL),
        "paired_prompt_protocol": PROTOCOL,
        "collection_name": target_collection, "retrieval_mode": retrieval_mode,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_path": str(source_path),
        "source_manifest_protocol": source.get("protocol"),
        "configurations": configurations, "jobs_sha256": canonical_hash(jobs),
        "job_count": len(jobs),
        # Test 7's user-facing cap is three B forecasts total: the initial
        # forecast plus at most two correction-driven revisions.
        "max_targeted_reevaluations": (
            2 if target_collection == "agent_test_7" else 3
        ),
        "max_full_hic_rounds_per_case": (
            3 if target_collection == "agent_test_7" else None
        ),
        "report_interval": 3, "final_report_label": "final_095",
        "legacy_33_case_report_boundaries": False,
        "validation_truth_read": False, "outer_test_targets_read": False,
        "exposure_audit_complete": False, "raw_worker_smoke_complete": False,
        "r0_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_new(output / "run_manifest.json", manifest)
    _write_new(output / "input_snapshots.json", jobs)
    _write_new(output / "runner_progress.json", {
        "run_id": run_id, "total": len(jobs), "completed_forecasts": 0,
        "scored": 0, "memories_appended": 0, "technical_failures": 0,
        "status": "prepared_not_ready", "next_case_id": jobs[0]["case_id"],
        "validation_truth_read": False, "outer_test_targets_read": False,
    })
    return manifest


def _load_round(output):
    output = Path(output).resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError("round output must remain under Agent Test 6")
    manifest = _read(output / "run_manifest.json")
    jobs = _read(output / "input_snapshots.json")
    expected_count = manifest.get("job_count", len(jobs))
    if (canonical_hash(jobs) != manifest.get("jobs_sha256")
            or not isinstance(expected_count, int) or len(jobs) != expected_count):
        raise ValueError("saved target-blind case snapshot differs from its manifest")
    return output, manifest, jobs


def _state_path(output, job):
    return output / "cases" / f"{job['job_id']}.json"


def _load_state(output, job):
    path = _state_path(output, job)
    return _read(path) if path.exists() else {}


def _save_state(output, job, state):
    _atomic_json(_state_path(output, job), state)


def _sync_progress(output, manifest, jobs):
    states = [_load_state(output, job) for job in jobs]
    terminal = sum(state.get("status") in TERMINAL_CASE_STATUSES for state in states)
    progress_path = output / "progress.json"
    progress = _read(progress_path) if progress_path.is_file() else {}
    next_index = _next_case_index(states)
    progress.update({
        "run_id": manifest.get("run_id", manifest.get("round_id", "unknown")),
        "total": len(jobs),
        "completed": terminal,
        "scored": sum(bool(state.get("score_sha256")) for state in states),
        "forecast_attempts": sum(bool(state.get("forecast_lock_sha256")) for state in states),
        "qdrant_mutations": sum(state.get("status") == "memory_appended" for state in states),
        "technical_failures": sum(state.get("status") == "technical_failure" for state in states),
        "status": (
            "complete" if terminal == len(jobs) else
            "prepared_not_started"
            if not manifest.get("r0_ready", False)
            and not any(state.get("forecast_lock_sha256") for state in states)
            else "running"
        ),
        "next_case_id": jobs[next_index]["case_id"] if next_index is not None else None,
        "validation_truth_read": any(state.get("score_sha256") or
                                      state.get("status") == "technical_failure"
                                      for state in states),
        "outer_test_targets_read": False,
        "input_snapshot_sha256": canonical_hash(_read(output / "input_snapshots.json")),
        "r0_ready": manifest.get("r0_ready", False),
    })
    _atomic_json(progress_path, progress)
    _atomic_json(output / "runner_progress.json", progress)
    return progress


def _forecast_anomalies(job, state, jobs, states):
    flags = []
    history_scale = max((abs(float(value)) for value in job["case"]["series"]), default=0.0)
    denominator = max(history_scale, 1e-12)
    for arm in ("baseline", "full_hic"):
        prediction = state[arm]["prediction"]
        ratio = max(abs(float(value)) for value in prediction) / denominator
        if ratio > SCALE_REVIEW_RATIO:
            flags.append({
                "type": "forecast_scale_anomaly", "arm": arm,
                "max_abs_prediction_to_history_ratio": ratio,
                "threshold": SCALE_REVIEW_RATIO,
                "diagnostic_status": "unresolved_before_scoring",
            })
    for other_job, other_state in zip(jobs, states):
        if other_job["dataset"] == job["dataset"] or other_state.get("status") not in {
            "forecast_complete", "technical_review_required",
        }:
            continue
        for arm, field in (("baseline", "baseline"), ("full_hic", "full_hic")):
            left, right = state[arm], other_state.get(field) or {}
            if (left.get("prediction") == right.get("prediction")
                    and isinstance(left.get("reasoning"), str)
                    and left.get("reasoning", "").strip()
                    and " ".join(left["reasoning"].split()).casefold()
                    == " ".join(right.get("reasoning", "").split()).casefold()):
                flags.append({
                    "type": "cross_family_exact_prediction_and_reasoning_match",
                    "arm": arm, "other_case_id": other_job["case_id"],
                    "other_dataset": other_job["dataset"],
                    "diagnostic_status": "unresolved_before_scoring",
                })
    return flags


def _optimization_state(output, job, state):
    direction = state.get("score_metrics", {}).get("mae_direction")
    if direction not in {"worsened", "tie"}:
        return {"status": "not_required", "attempts": 0}
    selected = state.get("retrieval_trace", {}).get("selected", [])
    replays = []
    attempts = []
    for path in (Path(output).parent / "targeted_reruns").glob("*/score.json"):
        targeted_manifest_path = path.parent / "targeted_run_manifest.json"
        if not targeted_manifest_path.is_file():
            continue
        targeted_manifest = _read(targeted_manifest_path)
        if targeted_manifest.get("job_id") != job["job_id"]:
            continue
        score = _read(path)
        if score.get("case_id") != job["case_id"]:
            continue
        revision_id = score.get("revision_id")
        lock = _read(path.parent / "forecast_lock.json")
        if lock.get("source_r0_forecast_lock_sha256") != state.get("forecast_lock_sha256"):
            continue
        if lock.get("baseline") != state.get("baseline"):
            raise ValueError("targeted reevaluation changed the locked No Retrieval A")
        if score.get("forecast_lock_sha256") != lock.get("forecast_lock_sha256"):
            raise ValueError("targeted reevaluation score and forecast lock differ")
        if isinstance(revision_id, str) and "current_memory_replay" in revision_id:
            replays.append(score)
        elif isinstance(revision_id, str) and "memory_revision_" in revision_id:
            attempts.append(score)
    replays.sort(key=lambda score: score["scored_utc"])
    attempts.sort(key=lambda score: score["scored_utc"])
    count = len(attempts)
    output = Path(output).resolve()
    manifest = _read(output / "run_manifest.json")
    limit = manifest.get("max_targeted_reevaluations", MAX_TARGETED_REEVALUATIONS)
    amendment_name = manifest.get("cap_amendment")
    if isinstance(amendment_name, str) and amendment_name:
        amendment_path = (output / amendment_name).resolve()
        if not amendment_path.is_relative_to(output):
            raise ValueError("cap amendment path must remain inside the Test 7 run directory")
        if amendment_path.is_file():
            amendment = _read(amendment_path)
            affected_jobs = amendment.get("affected_job_ids", [])
            if job.get("job_id") in affected_jobs:
                amended_limit = amendment.get("amended_max_targeted_reevaluations_per_case")
                if not isinstance(amended_limit, int) or amended_limit < limit:
                    raise ValueError("case-scoped cap amendment is invalid")
                limit = amended_limit
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("max_targeted_reevaluations must be a non-negative integer")
    if count > limit:
        raise ValueError(f"case {job['case_id']} exceeds {limit} targeted reevaluations")
    latest = attempts[-1] if attempts else None
    replay = replays[-1] if replays else None
    if latest is not None and latest.get("mae_direction") == "improved":
        status = "resolved"
    elif count == limit:
        status = "exhausted"
    elif count or selected or replay is not None:
        status = "pending"
    else:
        status = "no_eligible_memory"
    if not attempts and replay is not None and replay.get("mae_direction") == "improved":
        status = "resolved"
    latest_revision = (
        latest.get("revision_id") if latest else replay.get("revision_id") if replay else None
    )
    return {"status": status,
            "attempts": count, "limit": limit,
            "latest_revision": latest_revision}


def _latest_test7_targeted_attempt(output, job):
    """Return the latest completed Test 7 revision attempt without changing R0."""
    output = Path(output)
    attempts = []
    for manifest_path in (output.parent / "targeted_reruns").glob("*/targeted_run_manifest.json"):
        attempt_manifest = _read(manifest_path)
        if attempt_manifest.get("job_id") != job["job_id"]:
            continue
        score_path = manifest_path.parent / "score.json"
        state_path = manifest_path.parent / "case_state.json"
        lock_path = manifest_path.parent / "forecast_lock.json"
        if not (score_path.is_file() and state_path.is_file() and lock_path.is_file()):
            continue
        attempts.append((int(attempt_manifest.get("revision_number") or 0),
                         manifest_path.parent, attempt_manifest,
                         _read(score_path), _read(state_path), _read(lock_path)))
    attempts.sort(key=lambda item: (item[0], item[1].name))
    return attempts[-1] if attempts else None


def _completed_case_gate(output, jobs, current_index):
    for previous in jobs[:current_index]:
        state = _load_state(output, previous)
        if state.get("status") not in TERMINAL_CASE_STATUSES:
            raise RuntimeError(
                f"serial gate: prior case {previous['case_id']} must reach a scored/appended "
                "or documented technical-failure terminal state"
            )
        if state.get("status") == "technical_failure":
            continue
        optimization = _optimization_state(output, previous, state)
        if optimization["status"] == "pending":
            raise RuntimeError(
                f"optimization gate: prior regressed case {previous['case_id']} has "
                f"{optimization['attempts']}/{optimization['limit']} reevaluations; "
                "finish its targeted optimization first"
            )
    _ensure_due_reports(output, jobs)


def _ensure_due_reports(output, jobs):
    manifest = _read(Path(output) / "run_manifest.json")
    terminal = sum(_load_state(output, job).get("status") in TERMINAL_CASE_STATUSES for job in jobs)
    created = []
    if manifest.get("legacy_33_case_report_boundaries", len(jobs) == 33):
        for count in (13, 24, 29):
            label = f"config_end_{count:02d}"
            report_path = output / "reports" / f"{label}.json"
            if terminal >= count and not report_path.exists():
                _, markdown_path, _ = write_checkpoint_report(
                    output, label=label, upto_cases=count
                )
                created.extend([str(report_path), str(markdown_path)])
    interval = manifest.get("report_interval", 6)
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("report_interval must be a positive integer")
    for count in range(interval, terminal + 1, interval):
        label = f"checkpoint_{count:03d}"
        report_path = output / "reports" / f"{label}.json"
        if not report_path.exists():
            _, markdown_path, _ = write_checkpoint_report(
                output, label=label, upto_cases=count
            )
            created.extend([str(report_path), str(markdown_path)])
    total = len(jobs)
    final_label = manifest.get(
        "final_report_label", "final_33" if total == 33 else f"final_{total:03d}"
    )
    if terminal == total and not (output / "reports" / f"{final_label}.json").exists():
        report_path, markdown_path, _ = write_checkpoint_report(
            output, label=final_label, upto_cases=total
        )
        created.extend([str(report_path), str(markdown_path)])
    return created


def _queue_task_id_for_case(queue, manifest, job, state):
    """Resolve an in-flight inference task for a forecast or post-score stage."""
    if state.get("status") == "awaiting_agent":
        return state.get("pending_task_id")
    if state.get("status") == "scored":
        pending_task_id = state.get("pending_task_id")
        if pending_task_id and queue._path(pending_task_id).is_file():
            return pending_task_id
        task_id = (
            f"{manifest['run_id']}:{job['case_id']}:H{job['horizon']}:test6_postscore_reflection_v4"
        )
        if queue._path(task_id).is_file():
            return task_id
    return None


def _cache_for(queue, manifest, job, configuration):
    contract = configuration["contract"]
    return CodexTaskRequestCache(
        queue, run_id=manifest["run_id"], dataset=contract["dataset"],
        horizon=contract["horizon"], frequency=contract["frequency"],
        target=contract["target"], unit=contract["unit"],
        prompt_version=f"{manifest['collection_name']}_paired_v1_{job['round_id']}",
        memory_version=f"{manifest['collection_name']}_{job['round_id']}_memory_v1",
        require_raw_response_provenance=True,
    )


def _qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url="http://127.0.0.1:6333", timeout=30)


def _next_case_index(states):
    return next((index for index, state in enumerate(states)
                 if state.get("status") not in TERMINAL_CASE_STATUSES), None)


def _store(client, manifest, *, point_id_namespace=None):
    store = QdrantHicStore(
        model_id="BAAI/bge-m3", client=client,
        collection_name=manifest["collection_name"],
        point_id_namespace=(point_id_namespace or manifest["round_id"]),
    )
    if not client.collection_exists(manifest["collection_name"]):
        raise RuntimeError(f"Test 6 Qdrant round view {manifest['collection_name']} is missing")
    return store


def _embedder():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    embedder = get_embedder()
    if embedder.model_name != "BAAI/bge-m3" or embedder.max_length < 8192:
        raise RuntimeError("the frozen local BGE-M3 model is unavailable")
    return embedder


def advance_one(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    if not manifest.get("r0_ready"):
        raise RuntimeError("R0 gates are not complete; predictions are not started")
    states = [_load_state(output, job) for job in jobs]
    index = _next_case_index(states)
    if index is None:
        return {"status": "all_cases_terminal", "terminal": len(jobs),
                "memories_appended": sum(state.get("status") == "memory_appended" for state in states),
                "technical_failures": sum(state.get("status") == "technical_failure" for state in states)}
    job = jobs[index]
    _completed_case_gate(output, jobs, index)
    state = states[index]
    if state.get("status") in {"scored", "memory_ready", "memory_appended"}:
        raise RuntimeError(f"case is at {state['status']}; use its score/reflect/append command")
    configuration = manifest["configurations"][job["configuration"]]
    queue = CodexTaskQueue(output / "tasks", max_repairs=1)
    cache = _cache_for(queue, manifest, job, configuration)
    client = _qdrant_client()
    try:
        store = _store(client, manifest)
        embedder = _embedder()

        def embed_summary(summary):
            return embedder.encode([summary_to_embedding_text(summary)], is_query=True)[0]

        try:
            advance_agent_case(
                job["case"], state, contract=configuration["contract"],
                template=configuration["template"],
                dataset_description=configuration["description"],
                cache=cache, store=store, embed_summary=embed_summary,
                prior_online_case_ids=[
                    prior_job["case_id"]
                    for prior_job, prior_state in zip(jobs[:index], states[:index])
                    if prior_state.get("status") == "memory_appended"
                    and prior_state.get("score_sha256")
                ],
                retrieval_mode=manifest.get("retrieval_mode", "applicable"),
                persist=lambda: _save_state(output, job, state),
            )
        except CodexTaskPending as pending:
            state["pending_task_id"] = pending.task_id
            state["status"] = "awaiting_agent"
            _save_state(output, job, state)
            task = queue.get(pending.task_id)
            return {"status": "awaiting_agent", "completed_case_index": index + 1,
                    "total": len(jobs), "case_id": job["case_id"], "task_id": pending.task_id,
                    "task_type": task["task_type"], "queue_root": str(queue.root),
                    "task_identity_sha256": task["task_identity_sha256"],
                    "request_sha256": task["request_sha256"],
                    "worker_must_be_fresh": True}
        state["technical_anomalies"] = _forecast_anomalies(job, state, jobs, states)
        state["status"] = (
            "technical_review_required" if state["technical_anomalies"]
            else "forecast_complete"
        )
        state["forecast_lock_sha256"] = canonical_hash({
            "round_id": job["round_id"], "case_id": job["case_id"],
            "input_sha256": job["input_sha256"],
            "summary": state["summary"],
            "baseline": state["baseline"], "full_hic": state["full_hic"],
            "retrieval_trace": state["retrieval_trace"],
            "requests": state["requests"],
        })
        _save_state(output, job, state)
        lock_path = output / "forecast_locks" / f"{job['job_id']}.json"
        _write_new(lock_path, {
            "round_id": job["round_id"], "case_id": job["case_id"],
            "dataset": job["dataset"], "horizon": job["horizon"],
            "input_sha256": job["input_sha256"],
            "forecast_lock_sha256": state["forecast_lock_sha256"],
            "baseline": state["baseline"], "full_hic": state["full_hic"],
            "retrieval_trace": state["retrieval_trace"],
            "requests": state["requests"],
            "locked_utc": datetime.now(timezone.utc).isoformat(),
            "validation_truth_read": False, "outer_test_targets_read": False,
        })
        return {"status": state["status"], "technical_anomalies": state["technical_anomalies"],
                "completed_case_index": index + 1,
                "total": len(jobs), "case_id": job["case_id"],
                "forecast_lock_sha256": state["forecast_lock_sha256"]}
    finally:
        client.close()


def score_one(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    index = next((i for i, job in enumerate(jobs)
                  if _score_ready(_load_state(output, job))), None)
    if index is None:
        raise RuntimeError("there is no forecast-complete case awaiting scoring")
    _completed_case_gate(output, jobs, index)
    job = jobs[index]
    state = _load_state(output, job)
    lock = _read(output / "forecast_locks" / f"{job['job_id']}.json")
    if lock.get("forecast_lock_sha256") != state.get("forecast_lock_sha256"):
        raise ValueError("forecast lock does not match the current case checkpoint")
    case = job["case"]
    try:
        truth, truth_resolution = load_native_hic_validation_truth_for_case(
            job["dataset"], horizon=job["horizon"], case=case,
            target_timestamps=job["target_timestamps"],
            return_resolution=True,
        )
    except ValueError as exc:
        message = str(exc)
        if "validation target for " not in message or "missing or ambiguous" not in message:
            raise
        failure = {
            "run_id": manifest["run_id"], "case_id": job["case_id"],
            "dataset": job["dataset"], "horizon": job["horizon"],
            "input_sha256": job["input_sha256"],
            "forecast_lock_sha256": state["forecast_lock_sha256"],
            "failure_type": "ambiguous_or_missing_validation_target",
            "error": message, "scored": False, "mae_direction": None,
            "validation_truth_read": True, "outer_test_targets_read": False,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }
        failure_path = output / "technical_failures" / f"{job['job_id']}.json"
        if failure_path.exists():
            attempt = 2
            while (output / "technical_failures" /
                   f"{job['job_id']}_scoring_attempt_{attempt}.json").exists():
                attempt += 1
            failure_path = (output / "technical_failures" /
                            f"{job['job_id']}_scoring_attempt_{attempt}.json")
        _write_new(failure_path, failure)
        state["status"] = "technical_failure"
        state["technical_failure_sha256"] = canonical_hash(failure)
        state["technical_failure_type"] = failure["failure_type"]
        _save_state(output, job, state)
        _sync_progress(output, manifest, jobs)
        reports_created = _ensure_due_reports(output, jobs)
        return {"status": "technical_failure", "case_id": job["case_id"],
                "failure_type": failure["failure_type"], "scored": False,
                "reports_created": reports_created}
    errors_a = [a - y for a, y in zip(state["baseline"]["prediction"], truth)]
    errors_b = [b - y for b, y in zip(state["full_hic"]["prediction"], truth)]
    metrics = {}
    for name, errors in (("no_retrieval", errors_a), ("full_hic", errors_b)):
        mse = sum(error * error for error in errors) / len(errors)
        metrics[name] = {"mae": sum(abs(error) for error in errors) / len(errors),
                         "mse": mse, "rmse": math.sqrt(mse)}
    metrics["mae_direction"] = (
        "improved" if metrics["full_hic"]["mae"] < metrics["no_retrieval"]["mae"] - 1e-12
        else "worsened" if metrics["full_hic"]["mae"] > metrics["no_retrieval"]["mae"] + 1e-12
        else "tie"
    )
    score_record = {
        "round_id": job["round_id"], "case_id": job["case_id"],
        "dataset": job["dataset"], "horizon": job["horizon"],
        "input_sha256": job["input_sha256"],
        "forecast_lock_sha256": state["forecast_lock_sha256"],
        "truth": truth, "target_timestamps": job["target_timestamps"],
        "metrics": metrics,
        "truth_sha256": canonical_hash({"target_timestamps": job["target_timestamps"],
                                        "truth": truth}),
        "truth_resolution": truth_resolution,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "validation_truth_read": True, "outer_test_targets_read": False,
    }
    _write_new(output / "scores" / f"{job['job_id']}.json", score_record)
    state["status"] = "scored"
    state["score_sha256"] = canonical_hash(score_record)
    state["score_metrics"] = metrics
    _save_state(output, job, state)
    _sync_progress(output, manifest, jobs)
    return {"status": "scored", "case_id": job["case_id"], "metrics": metrics,
            "truth_sha256": score_record["truth_sha256"]}


def _score_ready(state):
    return (
        state.get("status") in {"forecast_complete", "technical_failure"}
        and bool(state.get("forecast_lock_sha256"))
        and isinstance(state.get("baseline"), dict)
        and isinstance(state.get("full_hic"), dict)
        and not state.get("score_sha256")
    )


def dispatch_one(output_dir, worker_context_id):
    output, manifest, jobs = _load_round(output_dir)
    if not isinstance(worker_context_id, str) or not worker_context_id.strip():
        raise ValueError("worker_context_id must be non-empty")
    queue = CodexTaskQueue(output / "tasks", max_repairs=1)
    candidate = None
    for index, job in enumerate(jobs):
        state = _load_state(output, job)
        task_id = _queue_task_id_for_case(queue, manifest, job, state)
        if task_id is None:
            continue
        task = queue.get(task_id)
        if task["status"] in {"pending", "repair_required"}:
            candidate = (index, job, state, task_id, task)
            break
    if candidate is None:
        raise RuntimeError("there is no queued Test 6 worker task")
    index, job, state, task_id, task = candidate
    _completed_case_gate(output, jobs, index)
    task_key = task["task_id_sha256"]
    dispatch_number = task["dispatch_count"] + 1
    response_path = output / "worker_responses" / f"{task_key}.{dispatch_number}.json"
    handoff_path = output / "handoffs" / f"{task_key}.{dispatch_number}.json"
    if response_path.exists() or handoff_path.exists():
        raise FileExistsError("task handoff artifacts already exist; inspect before reuse")
    # Build the prompt while the task is still pending. handoff_message rejects
    # dispatched tasks, so mutating the queue first would strand the task before
    # either its handoff artifact or worker response path is created.
    prompt = queue.handoff_message(task_id)
    dispatched = queue.mark_dispatched(
        task_id, worker_context_id=worker_context_id,
        context_isolation_mode="fresh_subagent_no_fork",
    )
    handoff = {
        "round_id": job["round_id"], "job_id": job["job_id"],
        "case_id": job["case_id"], "dataset": job["dataset"],
        "horizon": job["horizon"], "stage": task["metadata"]["stage"],
        "task_id": task_id, "task_id_sha256": task_key,
        "task_identity_sha256": task["task_identity_sha256"],
        "request_sha256": task["request_sha256"],
        "worker_context_id": worker_context_id,
        "dispatch_history": dispatched["dispatch_history"],
        "context_isolation_mode": "fresh_subagent_no_fork",
        "prompt": prompt,
        "raw_response_file": str(response_path.resolve()),
    }
    _write_new(handoff_path, handoff)
    return {"status": "dispatched", "case_id": job["case_id"],
            "task_id": task_id, "task_identity_sha256": task["task_identity_sha256"],
            "worker_context_id": worker_context_id,
            "handoff_file": str(handoff_path), "raw_response_file": str(response_path)}


def _task_response_validator(task):
    """Validate worker output before the queue marks a task complete."""
    schema = task.get("output_schema")
    if schema == "summary-json-v1":
        return _summary_response
    if schema == "baseline-json-v1":
        horizon = (task.get("metadata") or {}).get("horizon")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("forecast task is missing a valid horizon")

        def validate_forecast(response):
            _validated_forecast(response, horizon)
            return True

        return validate_forecast

    def validate_object(response):
        return isinstance(response, dict) and bool(response) and "error" not in response

    return validate_object


def collect_one(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    queue = CodexTaskQueue(output / "tasks", max_repairs=1)
    candidate = None
    for job in jobs:
        state = _load_state(output, job)
        task_id = _queue_task_id_for_case(queue, manifest, job, state)
        if task_id is None or not queue._path(task_id).is_file():
            continue
        task = queue.get(task_id)
        if task["status"] == "dispatched":
            candidate = (job, state, task_id, task)
            break
    if candidate is None:
        raise RuntimeError("there is no dispatched Test 6 task awaiting a worker response")
    job, state, task_id, task = candidate
    response_path = output / "worker_responses" / (
        f"{task['task_id_sha256']}.{task['dispatch_count']}.json"
    )
    if task["status"] != "dispatched":
        raise RuntimeError("only the currently dispatched task response can be collected")
    if not response_path.is_file():
        raise FileNotFoundError(f"worker response file is missing: {response_path}")
    raw_text = response_path.read_bytes().decode("utf-8")
    response = json.loads(raw_text)
    worker_id = task.get("worker_context_id")
    if (task.get("metadata") or {}).get("require_raw_response_provenance") is True:
        if not isinstance(response, dict):
            raise ValueError("provenance-bound worker response must be a JSON object")
        provenance = {
            "task_id": response.get("task_id"),
            "request_sha256": response.get("request_sha256"),
            "worker_context_id": response.get("worker_context_id"),
        }
        expected = {
            "task_id": task_id,
            "request_sha256": task.get("request_sha256"),
            "worker_context_id": worker_id,
        }
        if provenance != expected:
            raise ValueError("worker response provenance does not match the dispatched task")
        response = {
            key: value for key, value in response.items()
            if key not in {"task_id", "request_sha256", "worker_context_id"}
        }
    completed = queue.record_response(
        task_id, response, validator=_task_response_validator(task),
        worker_context_id=worker_id,
        raw_response_text=raw_text, response_source=str(response_path.resolve()),
    )
    if completed["status"] not in {"completed", "repair_required"}:
        raise RuntimeError(f"worker response was not accepted: {completed['status']}")
    return {"status": completed["status"], "case_id": job["case_id"],
            "task_id": task_id, "worker_context_id": worker_id,
            "raw_response_sha256": completed.get("raw_response_sha256"),
            "response_sha256": completed.get("response_sha256"),
            "response_source": completed.get("response_source")}


def reflect_one(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    index = next((i for i, job in enumerate(jobs)
                  if _load_state(output, job).get("status") == "scored"), None)
    if index is None:
        raise RuntimeError("there is no scored case awaiting correction reflection")
    _completed_case_gate(output, jobs, index)
    job = jobs[index]
    state = _load_state(output, job)
    score = _read(output / "scores" / f"{job['job_id']}.json")
    if score.get("forecast_lock_sha256") != state.get("forecast_lock_sha256"):
        raise ValueError("score is not bound to this locked A/B pair")
    reflection_state = state
    final_attempt = None
    if manifest.get("collection_name") == "agent_test_7":
        optimization = _optimization_state(output, job, state)
        if optimization["status"] == "pending":
            raise RuntimeError("finish the permitted Test 7 correction revisions before reflection")
        final_attempt = _latest_test7_targeted_attempt(output, job)
        if final_attempt is not None:
            _, attempt_dir, attempt_manifest, attempt_score, attempt_state, attempt_lock = final_attempt
            if (attempt_state.get("status") != "targeted_scored"
                    or attempt_score.get("forecast_lock_sha256") != attempt_lock.get("forecast_lock_sha256")
                    or attempt_score.get("case_id") != job["case_id"]):
                raise ValueError("latest Test 7 attempt is not a valid scored lock")
            reflection_state = dict(state)
            reflection_state.update({
                "status": "scored",
                "full_hic": attempt_state["full_hic"],
                "retrieval_trace": attempt_state["retrieval_trace"],
                "forecast_lock_sha256": attempt_lock["forecast_lock_sha256"],
                "requests": attempt_state.get("requests", {}),
            })
            score = dict(score)
            score["test7_final_attempt_metrics"] = attempt_score["metrics"]
            state["test7_final_attempt"] = {
                "revision_id": attempt_manifest.get("memory_revision_id"),
                "revision_number": attempt_manifest.get("revision_number"),
                "score_sha256": hashlib.sha256(
                    (attempt_dir / "score.json").read_bytes()
                ).hexdigest(),
                "forecast_lock_sha256": attempt_lock["forecast_lock_sha256"],
                "metrics": attempt_score["metrics"],
                "selected_point_id": attempt_state["retrieval_trace"]["selected"][0]["point_id"],
                "selected_revision_id": attempt_state["retrieval_trace"]["selected"][0].get(
                    "active_revision_id"
                ),
                "mae_direction": attempt_score["mae_direction"],
            }
    queue = CodexTaskQueue(output / "tasks", max_repairs=1)
    configuration = manifest["configurations"][job["configuration"]]
    cache = _cache_for(queue, manifest, job, configuration)
    try:
        memory = advance_test6_postscore_memory(
            job["case"], reflection_state, _read(output / "scores" / f"{job['job_id']}.json")["truth"],
            contract=configuration["contract"], cache=cache,
            persist=lambda: None,
        )
    except CodexTaskPending as pending:
        task = queue.get(pending.task_id)
        state["pending_task_id"] = pending.task_id
        _save_state(output, job, state)
        return {"status": "awaiting_agent", "case_id": job["case_id"],
                "task_id": pending.task_id, "task_type": task["task_type"],
                "queue_root": str(queue.root),
                "task_identity_sha256": task["task_identity_sha256"],
                "request_sha256": task["request_sha256"], "worker_must_be_fresh": True}
    state.pop("pending_task_id", None)
    state["online_memory"] = memory
    state["status"] = "memory_ready"
    _save_state(output, job, state)
    _write_new(output / "memory_revisions" / f"{job['job_id']}.json", {
        "round_id": job["round_id"], "case_id": job["case_id"],
        "source_horizon": memory["source_horizon"],
        "query_horizon": memory["query_horizon"],
        "basis_prediction": memory["basis_prediction"],
        "facts_sha256": memory["facts_sha256"],
        "correction_sha256": memory["correction_sha256"],
        "reflection_request_sha256": memory["reflection_request_sha256"],
        "correction": memory["correction"],
        "revision": "case_memory_v1",
        "diagnostic_sha256": state.get("diagnostic_sha256"),
    })
    return {"status": "memory_ready", "case_id": job["case_id"],
            "correction_sha256": memory["correction_sha256"]}


def append_one(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    index = next((i for i, job in enumerate(jobs)
                  if _load_state(output, job).get("status") == "memory_ready"), None)
    if index is None:
        raise RuntimeError("there is no completed correction awaiting Qdrant append")
    _completed_case_gate(output, jobs, index)
    job = jobs[index]
    state = _load_state(output, job)
    score = _read(output / "scores" / f"{job['job_id']}.json")
    revision = _read(output / "memory_revisions" / f"{job['job_id']}.json")
    correction = state["online_memory"]["correction"]
    if canonical_hash(json.loads(correction)) != revision["correction_sha256"]:
        raise ValueError("online memory correction differs from its frozen revision")
    transfer_record = make_test6_outcome_blind_transfer_record(state["summary"])
    transfer_correction = canonical_json(transfer_record)
    transfer_correction_sha256 = canonical_hash(transfer_record)
    audit_correction = json.loads(revision["correction"])
    audit_reflection = audit_correction.get("reflection")
    if not isinstance(audit_reflection, dict):
        raise ValueError("completed case memory revision has no qualitative reflection")
    revision_notes = []
    for field in ("explanation_hypotheses", "applicability_conditions", "exclusions"):
        values = audit_reflection.get(field, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            revision_notes.extend(value for value in values if isinstance(value, str))
    candidate_revision = {"qualitative_transfer_notes": revision_notes}
    projected = build_test6_transfer_projection(
        transfer_correction, source_horizon=job["horizon"],
        test6_transfer_revision=candidate_revision,
        outcome_blind=True, summary=state["summary"],
    )
    if projected is None:
        raise ValueError("completed case has no safe qualitative transfer revision")
    safe_revision_notes = json.loads(projected).get("test6_revision_notes", [])
    if not safe_revision_notes:
        raise ValueError("completed case reflection contains no safe transferable notes")
    test6_transfer_revision = {"qualitative_transfer_notes": safe_revision_notes}
    test6_transfer_revision_sha256 = canonical_hash(test6_transfer_revision)
    embedder = _embedder()
    embedding = embedder.encode(
        [summary_to_embedding_text(state["summary"])], is_query=False
    )[0]
    if len(embedding) != 1024 or not all(math.isfinite(float(value)) for value in embedding):
        raise ValueError("Test 6 memory embedding must be a finite BGE-M3 vector")
    configuration = manifest["configurations"][job["configuration"]]
    contract = configuration["contract"]
    point_id_namespace = f"{job['round_id']}:H{job['horizon']}"
    metadata = {
        "case_id": job["case_id"], "dataset": job["dataset"],
        "target": contract["target"], "unit": contract["unit"],
        "frequency": contract["frequency"], "horizon": job["horizon"],
        "source_horizon": job["horizon"], "query_horizon": job["horizon"],
        "entity": list(job["case"].get("group", [])),
        "source_sha256": contract["source_sha256"],
        "history_series": job["case"]["series"],
        "history_series_role": "pre_target_history_v1",
        "target_end": job["case"]["target_end"], "history_start": job["case"]["history_start"],
        "split": "validation", "completed": True,
        "news_quality": job["case"]["news_quality"],
        "test6_round_id": job["round_id"], "test6_scoring_status": "complete",
        "agent_test_collection": manifest["collection_name"],
        "test6_input_sha256": job["input_sha256"],
        "test6_forecast_lock_sha256": state["forecast_lock_sha256"],
        "test6_truth_sha256": score["truth_sha256"],
        "test6_score_sha256": state["score_sha256"],
        "test6_correction_sha256": transfer_correction_sha256,
        "test6_audit_reflection_sha256": revision["correction_sha256"],
        "test6_reflection_request_sha256": revision["reflection_request_sha256"],
        "summary_identity_status": "exact_request_response",
        "summary_request_sha256": state["requests"]["summary_direct_v2"],
        "summary_content_sha256": hashlib.sha256(
            canonical_json(state["summary"]).encode("utf-8")
        ).hexdigest(),
        "summary_prompt_version": SUMMARY_PROMPT_VERSION,
        "correction_schema_version": CORRECTION_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
        "test6_transfer_correction_version": CORRECTION_TRANSFER_SCHEMA_VERSION_TEST6_OUTCOME_BLIND,
        "test6_transfer_revision_id": "test6_outcome_blind_transfer_v1+case_reflection_v1",
        "test6_transfer_revision": test6_transfer_revision,
        "test6_transfer_revision_sha256": test6_transfer_revision_sha256,
        "baseline_request_sha256": state["requests"]["baseline_no_retrieval_paired_v1"],
        "baseline_prediction_sha256": canonical_hash(state["baseline"]["prediction"]),
        "memory_version": (
            f"{manifest['collection_name']}_{job['round_id']}_case_memory_v1"
        ),
        "point_identity_namespace": point_id_namespace,
    }
    if manifest.get("collection_name") == "agent_test_7":
        final_attempt = state.get("test7_final_attempt")
        if final_attempt is not None:
            metadata.update({
                "test7_final_attempt_revision_id": final_attempt["revision_id"],
                "test7_final_attempt_number": final_attempt["revision_number"],
                "test7_final_attempt_score_sha256": final_attempt["score_sha256"],
                "test7_final_attempt_lock_sha256": final_attempt["forecast_lock_sha256"],
                "test7_final_mae_direction": final_attempt["mae_direction"],
                "test7_retrieved_point_id": final_attempt["selected_point_id"],
                "test7_retrieved_revision_id": final_attempt["selected_revision_id"],
            })
    client = _qdrant_client()
    try:
        store = _store(
            client, manifest, point_id_namespace=point_id_namespace,
        )
        result = store.upsert_completed_case(
            case_id=job["case_id"], summary=state["summary"],
            correction=transfer_correction, embedding=embedding,
            observed_targets=score["truth"],
            prediction=state["full_hic"]["prediction"],
            cutoff_date=job["case"]["target_end"],
            source=f"{manifest['collection_name']}:{job['round_id']}:scored_validation",
            metadata=metadata,
        )
        point_id = store._point_id(
            store.collection_name, store.model_id, job["case_id"],
            namespace=point_id_namespace,
        )
        points = client.retrieve(
            collection_name=store.collection_name, ids=[point_id],
            with_payload=True, with_vectors=True,
        )
        if len(points) != 1 or points[0].payload.get("metadata") != metadata:
            raise ValueError("Qdrant append verification failed after upsert")
        state["status"] = "memory_appended"
        state["qdrant_point_id"] = point_id
        state["qdrant_append_result"] = result
        state["qdrant_count_after_append"] = store.count()
        state["qdrant_payload_sha256"] = canonical_hash(points[0].payload)
        state["qdrant_vector_dimension"] = len(points[0].vector)
        _save_state(output, job, state)
    finally:
        client.close()
    reports_created = _ensure_due_reports(output, jobs)
    _sync_progress(output, manifest, jobs)
    return {"status": "memory_appended", "case_id": job["case_id"],
            "point_id": state["qdrant_point_id"],
            "qdrant_count": state["qdrant_count_after_append"],
            "reports_created": reports_created}


def status(output_dir):
    output, manifest, jobs = _load_round(output_dir)
    states = [_load_state(output, job) for job in jobs]
    counts = {
        "forecast_complete": sum(bool(state.get("forecast_lock_sha256")) for state in states),
        "scored": sum(bool(state.get("score_sha256")) for state in states),
        "memory_ready": sum(state.get("status") == "memory_ready" for state in states),
        "memory_appended": sum(state.get("status") == "memory_appended" for state in states),
        "technical_failure": sum(state.get("status") == "technical_failure" for state in states),
    }
    reports_created = _ensure_due_reports(output, jobs)
    next_index = _next_case_index(states)
    pending_optimization = [
        {"case_id": job["case_id"], **_optimization_state(output, job, state)}
        for job, state in zip(jobs, states)
        if state.get("status") == "memory_appended"
        and _optimization_state(output, job, state)["status"] == "pending"
    ]
    terminal_count = sum(state.get("status") in TERMINAL_CASE_STATUSES for state in states)
    _sync_progress(output, manifest, jobs)
    return {"round_id": manifest["round_id"], "r0_ready": manifest["r0_ready"],
            "counts": counts, "total": len(jobs), "terminal_count": terminal_count,
            "next_case": jobs[next_index]["case_id"] if next_index is not None else None,
            "pending_optimization": pending_optimization,
            "next_case_blocked_by_optimization": bool(pending_optimization),
            "reports_created": reports_created,
            "validation_truth_read": any(
                state.get("score_sha256")
                or state.get("status") == "technical_failure"
                for state in states
            ),
            "outer_test_targets_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="create a new target-blind round manifest")
    prepare.add_argument("--round", dest="round_id", choices=ROUND_COLLECTIONS, required=True)
    prepare.add_argument("--output-dir", type=Path)
    prepare95 = sub.add_parser(
        "prepare-targetblind-95",
        help="bind the frozen 95-case target-blind manifest to the resumable runner",
    )
    prepare95.add_argument("--manifest", type=Path, required=True)
    prepare95.add_argument("--output-dir", type=Path, required=True)
    prepare95.add_argument("--run-id")
    prepare95.add_argument("--round-id")
    prepare95.add_argument("--collection-name", default="agent_test_6")
    prepare95.add_argument("--retrieval-mode", choices=("applicable", "direct_nearest"),
                           default="applicable")
    for command in ("advance-one", "score-one", "reflect-one", "append-one", "status"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
    dispatch = sub.add_parser("dispatch-one")
    dispatch.add_argument("--output-dir", type=Path, required=True)
    dispatch.add_argument("--worker-context-id", required=True)
    collect = sub.add_parser("collect-one")
    collect.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        output, manifest, jobs = prepare_round(args.round_id, output_dir=args.output_dir)
        result = {"output_dir": str(output), "round_id": manifest["round_id"],
                  "job_count": len(jobs), "validation_truth_read": False,
                  "outer_test_targets_read": False, "r0_ready": False}
    elif args.command == "prepare-targetblind-95":
        manifest = prepare_targetblind_95(
            args.manifest, output_dir=args.output_dir,
            run_id=args.run_id, round_id=args.round_id,
            target_collection=args.collection_name,
            retrieval_mode=args.retrieval_mode,
        )
        result = {"output_dir": str(args.output_dir.resolve()),
                  "round_id": manifest["round_id"], "job_count": manifest["job_count"],
                  "max_targeted_reevaluations": manifest["max_targeted_reevaluations"],
                  "report_interval": manifest["report_interval"],
                  "validation_truth_read": False, "outer_test_targets_read": False,
                  "r0_ready": False}
    elif args.command == "advance-one":
        result = advance_one(args.output_dir)
    elif args.command == "score-one":
        result = score_one(args.output_dir)
    elif args.command == "reflect-one":
        result = reflect_one(args.output_dir)
    elif args.command == "append-one":
        result = append_one(args.output_dir)
    elif args.command == "dispatch-one":
        result = dispatch_one(args.output_dir, args.worker_context_id)
    elif args.command == "collect-one":
        result = collect_one(args.output_dir)
    else:
        result = status(args.output_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
