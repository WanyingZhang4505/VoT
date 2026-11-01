"""Run DeepSeek Test 1's frozen cases with short local artifact paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM.qdrant_store import QdrantHicStore
from scripts import run_agent_test8_deepseek as core


RUN_ID = "deepseek_test1_onepass_15_20261101_r2"
HIC_ROOT = ROOT.parents[2]
ARTIFACT_ROOT = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2"
PROJECT_ARTIFACT_ROOT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101"
RUN_DIR = ARTIFACT_ROOT / "onepass_15"
COLLECTION = "DeepSeek Test 1"
PROTOCOL_ID = "agent_test8_deepseek_one_pass_direct_nearest_v1"
FORECAST_PROTOCOL = "agent_test8_deepseek_paired_forecast_v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure() -> None:
    core.RUN_DIR = RUN_DIR
    core.COLLECTION = COLLECTION
    core.PROTOCOL_ID = PROTOCOL_ID
    core.FORECAST_PROTOCOL = FORECAST_PROTOCOL

    def verify_collection(store: QdrantHicStore) -> dict:
        if not store.client.collection_exists(COLLECTION):
            raise RuntimeError(f"{COLLECTION} is missing; restore its frozen snapshot first")
        info = store.client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        frozen = json.loads((RUN_DIR / "frozen_protocol.json").read_text(encoding="utf-8"))
        expected = frozen.get("target_collection_frozen_point_count")
        if (expected != 703 or str(info.status).lower() != "green"
                or info.points_count != expected or vectors.size != 1024
                or str(vectors.distance).lower() != "cosine"):
            raise RuntimeError(f"{COLLECTION} must be green with 703 1024-D Cosine points")
        return {"status": str(info.status), "points_count": info.points_count,
                "dimension": vectors.size, "distance": str(vectors.distance)}

    core._verify_collection = verify_collection
    core._configure_runtime()


def _verify_freeze() -> None:
    frozen = json.loads((RUN_DIR / "frozen_protocol.json").read_text(encoding="utf-8"))
    audit_path = PROJECT_ARTIFACT_ROOT / "preflight/qdrant_source_audit.json"
    receipt_path = PROJECT_ARTIFACT_ROOT / "preflight/qdrant_restore_receipt.json"
    recovery_path = ARTIFACT_ROOT / "prior_attempt_recovery_audit.json"
    if (frozen.get("target_collection") != COLLECTION
            or frozen.get("target_collection_frozen_point_count") != 703
            or frozen.get("source_collection") != "agent_test_8"
            or frozen.get("source_collection_live_point_count") != 718
            or frozen.get("excluded_validation_feedback_count") != 15
            or frozen.get("run_wrapper_sha256") != _sha_file(Path(__file__))
            or frozen.get("memory_source_audit_sha256") != _sha_file(audit_path)
            or frozen.get("qdrant_restore_receipt_sha256") != _sha_file(receipt_path)
            or frozen.get("prior_technical_failure_audit_sha256") != _sha_file(recovery_path)):
        raise ValueError("DeepSeek Test 1 frozen collection, provenance, or runner hash changed")


def main() -> None:
    _configure()
    _verify_freeze()
    core._load()
    core.main()


if __name__ == "__main__":
    main()
