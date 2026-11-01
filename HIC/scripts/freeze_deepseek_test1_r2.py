"""Freeze the short-path recovery run for DeepSeek Test 1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import freeze_agent_test8_deepseek as freezer
from scripts import run_agent_test8_deepseek as core


RUN_ID = "deepseek_test1_onepass_15_20261101_r2"
HIC_ROOT = ROOT.parents[2]
SOURCE_ARTIFACT_ROOT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101"
ARTIFACT_ROOT = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2"
RUN_DIR = ARTIFACT_ROOT / "onepass_15"
COLLECTION = "DeepSeek Test 1"
SNAPSHOT = SOURCE_ARTIFACT_ROOT / "preflight/qdrant_snapshot/agent_test_8_filtered_703.jsonl"
SOURCE_AUDIT = SOURCE_ARTIFACT_ROOT / "preflight/qdrant_source_audit.json"
RESTORE_RECEIPT = SOURCE_ARTIFACT_ROOT / "preflight/qdrant_restore_receipt.json"
PRIOR_FAILED_RUN = SOURCE_ARTIFACT_ROOT / "onepass_15"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite frozen run directory: {RUN_DIR}")
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    receipt = json.loads(RESTORE_RECEIPT.read_text(encoding="utf-8"))
    failure_files = sorted((PRIOR_FAILED_RUN / "technical_failures").glob("*.json"))
    state_files = sorted((PRIOR_FAILED_RUN / "cases").glob("*.json"))
    if len(failure_files) != 1 or len(state_files) != 1:
        raise ValueError("expected exactly one preserved short-path setup failure")
    prior_failure = json.loads(failure_files[0].read_text(encoding="utf-8"))
    prior_case = json.loads(state_files[0].read_text(encoding="utf-8"))
    if (prior_failure.get("type") != "FileNotFoundError"
            or prior_failure.get("validation_truth_read") is not False
            or prior_case.get("status") != "recovery_pending"
            or list((PRIOR_FAILED_RUN / "provider_calls").glob("*.json"))):
        raise ValueError("preserved failure does not prove API request was not submitted")
    if (audit.get("source_collection") != "agent_test_8"
            or audit.get("source_before", {}).get("points_count") != 718
            or audit.get("filtered_snapshot", {}).get("point_count") != 703
            or audit.get("baseline_vector_mismatches") != 0
            or audit.get("baseline_payload_mismatches") != 0
            or audit.get("excluded_count") != 15
            or receipt.get("collection") != COLLECTION
            or receipt.get("points_count") != 703
            or receipt.get("dimension") != 1024
            or receipt.get("distance") != "Cosine"):
        raise ValueError("Test 1 memory source audit or restore receipt is invalid")

    core.RUN_DIR = RUN_DIR
    core.COLLECTION = COLLECTION
    freezer.RUN_ID = RUN_ID
    freezer.RUN_DIR = RUN_DIR
    freezer.COLLECTION = COLLECTION
    freezer.SNAPSHOT = SNAPSHOT
    result = freezer.build()

    prior_attempt = {
        "classification": "local_request_record_creation_failed_before_api_submission",
        "run_id": "deepseek_test1_onepass_15_20261101",
        "case_id": prior_case.get("case_id"),
        "failure_type": prior_failure["type"],
        "failure_record_path": str(failure_files[0].relative_to(ROOT).as_posix()),
        "failure_record_sha256": _sha_file(failure_files[0]),
        "case_state_path": str(state_files[0].relative_to(ROOT).as_posix()),
        "case_state_sha256": _sha_file(state_files[0]),
        "provider_api_request_submitted": False,
        "validation_truth_read": False,
        "recovery_configuration": "same model, prompt, generation settings, and frozen input; shorter local output path",
    }
    recovery_path = ARTIFACT_ROOT / "prior_attempt_recovery_audit.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    if recovery_path.exists():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {recovery_path}")
    _write(recovery_path, prior_attempt)

    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "source_collection": "agent_test_8",
        "source_collection_live_point_count": 718,
        "source_collection_filtered_point_count": 703,
        "excluded_validation_feedback_count": 15,
        "excluded_validation_feedback_case_ids": sorted(
            row["case_id"] for row in audit["excluded_validation_feedback_points"]
        ),
        "memory_bank_scope": (
            "agent_test_8's 703 pre-existing points; excludes the 15 points "
            "derived from these same validation cases"
        ),
        "result_scope_warning": (
            "One-pass replay using Test 7 development memory retained in Agent Test 8; "
            "not an independent external benchmark"
        ),
        "prior_technical_failure_audit": str(recovery_path.relative_to(HIC_ROOT).as_posix()),
    })
    _write(manifest_path, manifest)

    preflight_path = RUN_DIR / "preflight/deepseek_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight.update({
        "retrieval_collection_source": "agent_test_8",
        "retrieval_collection_live_points_before_filter": 718,
        "retrieval_collection_points_after_excluding_current_cases": 703,
        "excluded_validation_feedback_count": 15,
        "qdrant_restore_receipt_sha256": _sha_file(RESTORE_RECEIPT),
        "qdrant_source_audit_sha256": _sha_file(SOURCE_AUDIT),
        "prior_technical_failure_audit_sha256": _sha_file(recovery_path),
    })
    _write(preflight_path, preflight)

    frozen_path = RUN_DIR / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen.update({
        "retrieval": f"{COLLECTION} direct_nearest Top-1; no eligibility filters",
        "source_collection": "agent_test_8",
        "source_collection_live_point_count": 718,
        "excluded_validation_feedback_count": 15,
        "excluded_validation_feedback_case_ids": manifest[
            "excluded_validation_feedback_case_ids"
        ],
        "target_collection": COLLECTION,
        "target_collection_frozen_point_count": 703,
        "run_manifest_sha256": _sha_file(manifest_path),
        "memory_source_audit_sha256": _sha_file(SOURCE_AUDIT),
        "qdrant_restore_receipt_sha256": _sha_file(RESTORE_RECEIPT),
        "prior_technical_failure_audit_sha256": _sha_file(recovery_path),
        "run_wrapper_sha256": _sha_file(ROOT / "scripts/run_deepseek_test1_r2.py"),
        "freeze_wrapper_sha256": _sha_file(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write(frozen_path, frozen)
    result.update({
        "status": "frozen_ready_pending_preflight_and_predictions",
        "run_dir": str(RUN_DIR.resolve()),
        "collection": COLLECTION,
        "source_collection": "agent_test_8",
        "source_points": 718,
        "excluded_validation_feedback_points": 15,
        "retrieval_points": 703,
        "prior_technical_failure": prior_attempt,
        "run_manifest_sha256": _sha_file(manifest_path),
        "snapshot_sha256": audit["filtered_snapshot"]["sha256"],
        "frozen_protocol_sha256": _sha_file(frozen_path),
        "api_key_configured": freezer.hic.DEEPSEEK_API_KEY != "",
        "api_calls": 0,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
