"""Freeze a one-pass replay using Agent Test 8's baseline minus its 15 new cases."""

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


RUN_ID = "deepseek_test1_onepass_15_20261101"
ARTIFACT_ROOT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101"
RUN_DIR = ARTIFACT_ROOT / "onepass_15"
COLLECTION = "DeepSeek Test 1"
SNAPSHOT = ARTIFACT_ROOT / "preflight/qdrant_snapshot/agent_test_8_filtered_703.jsonl"
SOURCE_AUDIT = ARTIFACT_ROOT / "preflight/qdrant_source_audit.json"
RESTORE_RECEIPT = ARTIFACT_ROOT / "preflight/qdrant_restore_receipt.json"


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
        "run_wrapper_sha256": _sha_file(ROOT / "scripts/run_deepseek_test1.py"),
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
