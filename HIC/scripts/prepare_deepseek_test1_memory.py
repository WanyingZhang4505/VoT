"""Export Agent Test 8's 703 pre-existing points, excluding its 15 new cases."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COLLECTION = "agent_test_8"
TARGET_COLLECTION = "DeepSeek Test 1"
BASELINE_SNAPSHOT = ROOT / (
    "LLM/evaluations/Agent Test 8/remaining3families_15_20261101/"
    "preflight/qdrant_snapshot/agent_test_7_full_points.jsonl"
)
SOURCE_RUN = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
ARTIFACT_ROOT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101"
SNAPSHOT = ARTIFACT_ROOT / "preflight/qdrant_snapshot/agent_test_8_filtered_703.jsonl"
AUDIT = ARTIFACT_ROOT / "preflight/qdrant_source_audit.json"
URL = "http://127.0.0.1:6333"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    if SNAPSHOT.exists() or AUDIT.exists():
        raise FileExistsError("Test 1 source snapshot/audit already exists; refusing overwrite")
    baseline_rows = [json.loads(line) for line in
                     BASELINE_SNAPSHOT.read_text(encoding="utf-8").splitlines()]
    baseline = {str(row["id"]): row for row in baseline_rows}
    if len(baseline) != 703 or len(baseline_rows) != 703:
        raise ValueError("frozen baseline does not contain 703 unique points")
    expected_jobs = json.loads((SOURCE_RUN / "input_snapshots.json").read_text(encoding="utf-8"))
    expected_case_ids = {job["case_id"] for job in expected_jobs}
    if len(expected_jobs) != 15 or len(expected_case_ids) != 15:
        raise ValueError("frozen Agent Test 8 source does not contain 15 unique cases")

    client = QdrantClient(url=URL, timeout=60)
    try:
        if not client.collection_exists(SOURCE_COLLECTION):
            raise RuntimeError("live agent_test_8 collection is missing")
        if client.collection_exists(TARGET_COLLECTION):
            raise FileExistsError(f"{TARGET_COLLECTION} already exists; refusing overwrite")
        before = client.get_collection(SOURCE_COLLECTION)
        vectors = before.config.params.vectors
        if (str(before.status).lower() != "green" or before.points_count != 718
                or vectors.size != 1024 or str(vectors.distance).lower() != "cosine"):
            raise RuntimeError("agent_test_8 must be green with 718 1024-D Cosine points")

        live = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=SOURCE_COLLECTION, offset=offset, limit=128,
                with_payload=True, with_vectors=True,
            )
            for point in points:
                point_id = str(point.id)
                if point_id in live:
                    raise ValueError(f"duplicate live point ID: {point_id}")
                vector = point.vector
                if not isinstance(vector, list) or len(vector) != 1024 or any(
                    not isinstance(value, (float, int)) or not math.isfinite(float(value))
                    for value in vector
                ):
                    raise ValueError(f"invalid vector in source point {point_id}")
                live[point_id] = {"id": point_id, "vector": vector,
                                  "payload": point.payload or {}}
            if offset is None:
                break

        after = client.get_collection(SOURCE_COLLECTION)
        if (after.points_count != before.points_count
                or str(after.status).lower() != str(before.status).lower()):
            raise RuntimeError("source collection changed while its snapshot was exported")
        live_ids = set(live)
        baseline_ids = set(baseline)
        missing = baseline_ids - live_ids
        excluded = live_ids - baseline_ids
        if len(live) != 718 or missing or len(excluded) != 15:
            raise ValueError(
                f"expected baseline 703 plus 15 additions; missing={len(missing)}, "
                f"extra={len(excluded)}, live={len(live)}"
            )

        vector_mismatches, payload_mismatches = [], []
        for point_id in sorted(baseline_ids):
            if live[point_id]["vector"] != baseline[point_id]["vector"]:
                vector_mismatches.append(point_id)
            if live[point_id]["payload"] != baseline[point_id].get("payload", {}):
                payload_mismatches.append(point_id)
        if vector_mismatches or payload_mismatches:
            raise ValueError(
                f"the 703 baseline changed in live Test 8: vector mismatches="
                f"{len(vector_mismatches)}, payload mismatches={len(payload_mismatches)}"
            )

        excluded_rows = []
        for point_id in sorted(excluded):
            payload = live[point_id]["payload"]
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            case_id = metadata.get("case_id") or payload.get("case_id")
            dataset = metadata.get("dataset") or payload.get("dataset")
            horizon = metadata.get("horizon") or payload.get("horizon")
            excluded_rows.append({"point_id": point_id, "case_id": case_id,
                                  "dataset": dataset, "horizon": horizon})
        if {row["case_id"] for row in excluded_rows} != expected_case_ids:
            raise ValueError("the 15 non-baseline points do not match the frozen Test 8 cases")

        snapshot_bytes = b"".join(
            _canonical(live[point_id]) + b"\n" for point_id in sorted(baseline_ids)
        )
        snapshot_hash = _sha(snapshot_bytes)
        _atomic_bytes(SNAPSHOT, snapshot_bytes)
        audit = {
            "protocol": "deepseek_test1_agent_test8_source_filter_v1",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_collection": SOURCE_COLLECTION,
            "target_collection": TARGET_COLLECTION,
            "source_before": {"status": str(before.status),
                              "points_count": before.points_count,
                              "dimension": vectors.size, "distance": "Cosine"},
            "source_after": {"status": str(after.status),
                             "points_count": after.points_count},
            "baseline_source_snapshot": {
                "path": BASELINE_SNAPSHOT.relative_to(ROOT).as_posix(),
                "sha256": _sha(BASELINE_SNAPSHOT.read_bytes()),
                "point_count": len(baseline),
            },
            "filtered_snapshot": {"path": SNAPSHOT.relative_to(ROOT).as_posix(),
                                  "sha256": snapshot_hash,
                                  "bytes": len(snapshot_bytes),
                                  "point_count": 703, "dimension": 1024,
                                  "distance": "Cosine"},
            "baseline_points_missing": 0,
            "baseline_vector_mismatches": 0,
            "baseline_payload_mismatches": 0,
            "excluded_validation_feedback_points": excluded_rows,
            "excluded_count": len(excluded_rows),
            "excluded_case_ids_match_frozen_test8_cases": True,
            "validation_truth_read": False,
            "api_calls": 0,
            "qdrant_writes": 0,
        }
        _atomic_bytes(AUDIT, (_canonical(audit).decode("utf-8") + "\n").encode("utf-8"))
        print(json.dumps({"status": "filtered_snapshot_ready", **audit},
                         ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
