"""Append the scored Agent Test 8 validation experiences to agent_test_8.

This is a post-evaluation ingestion step. It preserves the original 703 points,
uses a dedicated point-id namespace, and stores qualitative corrections only;
the Qdrant store API validates but does not persist raw target/prediction arrays.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic
from LLM.qdrant_store import QdrantHicStore
from scripts.run_agent_test6 import _embedder

COLLECTION = "agent_test_8"
NAMESPACE = "agent_test_8_postscore_validation_experience_v1"
EXPECTED_INITIAL_COUNT = 703
EXPECTED_ADDED_COUNT = 15
RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
POSTHOC_DIR = ROOT / "LLM/evaluations/Agent Test 8/posthoc_correction_driven_degraded5_iter5_20261101"
INGEST_DIR = ROOT / "LLM/evaluations/Agent Test 8/memory_ingest_after_eval_20261101"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    plan = read_json(INGEST_DIR / "audit/ingest_plan.json")
    if plan.get("target_collection") != COLLECTION or plan.get("case_count") != EXPECTED_ADDED_COUNT:
        raise RuntimeError("ingest plan scope/count does not match the guarded Test 8 write")
    if len(plan.get("cases", [])) != EXPECTED_ADDED_COUNT:
        raise RuntimeError("ingest plan case list is incomplete")

    store = QdrantHicStore(
        model_id="BAAI/bge-m3",
        collection_name=COLLECTION,
        point_id_namespace=NAMESPACE,
    )
    try:
        if not store.client.collection_exists(COLLECTION):
            raise RuntimeError("agent_test_8 collection does not exist")
        before_count = store.count()
        if before_count != EXPECTED_INITIAL_COUNT:
            raise RuntimeError(
                f"refusing append: expected {EXPECTED_INITIAL_COUNT} initial points, found {before_count}"
            )
        info = store.client.get_collection(COLLECTION)
        vector_config = info.config.params.vectors
        if isinstance(vector_config, dict):
            if len(vector_config) != 1:
                raise RuntimeError("agent_test_8 has unexpected named vectors")
            vector_config = next(iter(vector_config.values()))
        if getattr(vector_config, "size", None) != 1024 or str(getattr(vector_config, "distance", "")).lower().split(".")[-1] != "cosine":
            raise RuntimeError("agent_test_8 vector configuration is not 1024-dimensional Cosine")

        embedder = _embedder()
        items = []
        point_ids = set()
        for row in plan["cases"]:
            ordinal = int(row["ordinal"])
            state_files = list((RUN_DIR / "cases").glob(f"{ordinal:02d}_*.json"))
            if len(state_files) != 1:
                raise RuntimeError(f"expected one frozen case state for ordinal {ordinal}")
            state = read_json(state_files[0])
            truth_audit = read_json(RUN_DIR / "truth_audit" / f"{ordinal:02d}.json")
            if state["case_id"] != row["case_id"] or truth_audit["case_id"] != row["case_id"]:
                raise RuntimeError(f"case identity mismatch at ordinal {ordinal}")

            if row["metadata"]["posthoc_correction_feedback_used"]:
                response_path = POSTHOC_DIR / "tasks/responses" / f"{ordinal:02d}_revision1.json"
            else:
                response_path = RUN_DIR / "tasks/responses" / f"case{ordinal:02d}_b_attempt1.json"
            response = read_json(response_path)
            prediction = response.get("prediction")
            targets = truth_audit.get("truth_values")
            if not isinstance(prediction, list) or not isinstance(targets, list):
                raise RuntimeError(f"missing paired prediction/truth arrays for ordinal {ordinal}")
            if len(prediction) != int(row["horizon"]) or len(targets) != int(row["horizon"]):
                raise RuntimeError(f"paired horizon mismatch for ordinal {ordinal}")

            summary = row["summary"]
            vector = embedder.encode(
                [hic.summary_to_embedding_text(summary)], is_query=True
            )[0]
            point_id = store._point_id(COLLECTION, "BAAI/bge-m3", row["case_id"], namespace=NAMESPACE)
            if point_id in point_ids:
                raise RuntimeError("duplicate point identity in ingest plan")
            point_ids.add(point_id)
            existing = store.client.retrieve(
                collection_name=COLLECTION, ids=[point_id],
                with_payload=True, with_vectors=False,
            )
            if existing:
                raise RuntimeError(f"namespaced point already exists for ordinal {ordinal}")
            items.append({
                "row": row,
                "targets": targets,
                "prediction": prediction,
                "vector": vector,
                "response_sha256": sha256(response_path.read_bytes()),
                "truth_sha256": truth_audit["truth_sha256"],
                "forecast_lock_sha256": state["forecast_lock_sha256"],
                "point_id": point_id,
            })

        # Snapshot the live 703-point collection before the append.
        snapshot = store.client.create_snapshot(COLLECTION, wait=True)
        if snapshot is None or not getattr(snapshot, "name", None):
            raise RuntimeError("Qdrant did not return a collection snapshot name")
        snapshot_name = snapshot.name
        snapshot_url = (
            "http://127.0.0.1:6333/collections/"
            + quote(COLLECTION, safe="")
            + "/snapshots/"
            + quote(snapshot_name, safe="")
        )
        response = httpx.get(snapshot_url, timeout=300.0)
        response.raise_for_status()
        snapshot_path = INGEST_DIR / "snapshot" / snapshot_name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(response.content)
        if not snapshot_path.stat().st_size:
            raise RuntimeError("downloaded pre-write snapshot is empty")
        snapshot_receipt = {
            "collection": COLLECTION,
            "snapshot_name": snapshot_name,
            "snapshot_path": str(snapshot_path.resolve()),
            "snapshot_size_bytes": snapshot_path.stat().st_size,
            "snapshot_sha256": sha256(snapshot_path.read_bytes()),
            "point_count_before": before_count,
            "vector_size": getattr(vector_config, "size", None),
            "distance": str(getattr(vector_config, "distance", "")),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (INGEST_DIR / "audit/prewrite_snapshot.json").write_text(
            json.dumps(snapshot_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        outcomes = []
        for item in items:
            row = item["row"]
            metadata = dict(row["metadata"])
            metadata.update({
                "memory_namespace": NAMESPACE,
                "ingest_run_id": plan["run_id"],
                "stored_correction_sha256": row["correction_sha256"],
                "source_response_sha256": item["response_sha256"],
                "source_truth_sha256": item["truth_sha256"],
                "source_forecast_lock_sha256": item["forecast_lock_sha256"],
                "raw_targets_or_predictions_stored": False,
            })
            result = store.upsert_completed_case(
                case_id=row["case_id"],
                summary=row["summary"],
                correction=row["correction"],
                embedding=item["vector"],
                observed_targets=item["targets"],
                prediction=item["prediction"],
                source=row["source"],
                metadata=metadata,
            )
            outcomes.append({
                "ordinal": row["ordinal"],
                "case_id": row["case_id"],
                "point_id": item["point_id"],
                "result": result,
                "stored_correction_sha256": row["correction_sha256"],
                "response_sha256": item["response_sha256"],
                "truth_sha256": item["truth_sha256"],
                "direction": row["expected_direction"],
            })

        after_count = store.count()
        expected_final = EXPECTED_INITIAL_COUNT + EXPECTED_ADDED_COUNT
        if after_count != expected_final:
            raise RuntimeError(f"post-write point count {after_count}, expected {expected_final}")
        receipt = {
            "run_id": plan["run_id"],
            "collection": COLLECTION,
            "namespace": NAMESPACE,
            "point_count_before": before_count,
            "point_count_added": len(outcomes),
            "point_count_after": after_count,
            "snapshot": snapshot_receipt,
            "outcomes": outcomes,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (INGEST_DIR / "audit/write_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "complete",
            "collection": COLLECTION,
            "before": before_count,
            "added": len(outcomes),
            "after": after_count,
            "snapshot_sha256": snapshot_receipt["snapshot_sha256"],
        }, ensure_ascii=False))
        return 0
    finally:
        store.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
