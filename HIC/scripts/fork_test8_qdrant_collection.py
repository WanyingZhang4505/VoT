"""Losslessly copy the live Test 7 Qdrant collection into Test 8."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
SNAPSHOT_DIR = RUN_DIR / "preflight" / "qdrant_snapshot"
SOURCE = "agent_test_7"
TARGET = "agent_test_8"
URL = "http://127.0.0.1:6333"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(point) -> list[float]:
    value = point.vector
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"point {point.id} has unexpected named vectors")
        value = next(iter(value.values()))
    if not isinstance(value, (list, tuple)) or len(value) != 1024:
        raise ValueError(f"point {point.id} does not have one 1024-dimensional vector")
    return [float(item) for item in value]


def _scroll(client: QdrantClient, collection: str) -> list[dict]:
    rows = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=128, offset=offset,
            with_payload=True, with_vectors=True,
        )
        rows.extend({"id": str(point.id), "vector": _vector(point),
                     "payload": point.payload or {}} for point in points)
        if offset is None:
            break
    rows.sort(key=lambda row: row["id"])
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            line = (_canonical(row) + "\n").encode("utf-8")
            digest.update(line)
            handle.write(line)
    return {"path": str(path.resolve()), "count": len(rows),
            "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(f"frozen Test 8 run directory is missing: {RUN_DIR}")
    if (RUN_DIR / "preflight" / "qdrant_copy_manifest.json").exists():
        raise FileExistsError("refusing to overwrite the Qdrant copy manifest")
    client = QdrantClient(url=URL, timeout=60)
    try:
        if not client.collection_exists(SOURCE):
            raise RuntimeError(f"source collection {SOURCE} is missing")
        if client.collection_exists(TARGET):
            target = client.get_collection(TARGET)
            raise FileExistsError(
                f"{TARGET} already exists (status={target.status}, points={target.points_count}); "
                "left untouched for read-only review"
            )
        source_info = client.get_collection(SOURCE)
        if str(source_info.status).lower() != "green":
            raise RuntimeError(f"source collection is not green: {source_info.status}")
        vectors = source_info.config.params.vectors
        if isinstance(vectors, dict):
            if len(vectors) != 1:
                raise ValueError("source must use one unnamed vector configuration")
            vectors = next(iter(vectors.values()))
        if vectors.size != 1024 or str(vectors.distance).lower() != "cosine":
            raise ValueError("source vector configuration differs from 1024-dimensional Cosine")

        source_rows = _scroll(client, SOURCE)
        source_before = {"points_count": source_info.points_count,
                         "exact_scanned_points": len(source_rows),
                         "status": str(source_info.status),
                         "vector_size": vectors.size,
                         "distance": str(vectors.distance)}
        if len(source_rows) != source_before["points_count"]:
            raise ValueError("source live count changed during full export")
        snapshot_file = SNAPSHOT_DIR / "agent_test_7_full_points.jsonl"
        snapshot = _write_jsonl(snapshot_file, source_rows)

        client.create_collection(
            collection_name=TARGET,
            vectors_config=models.VectorParams(size=vectors.size,
                                                distance=vectors.distance),
        )
        for start in range(0, len(source_rows), 64):
            batch = source_rows[start:start + 64]
            client.upsert(
                collection_name=TARGET,
                points=[models.PointStruct(id=row["id"], vector=row["vector"],
                                           payload=row["payload"]) for row in batch],
                wait=True,
            )
        target_rows = _scroll(client, TARGET)
        target_info = client.get_collection(TARGET)
        if len(target_rows) != len(source_rows):
            raise ValueError("restored Test 8 point count differs from frozen source export")
        source_by_id = {row["id"]: row for row in source_rows}
        target_by_id = {row["id"]: row for row in target_rows}
        missing = sorted(set(source_by_id) - set(target_by_id))
        extra = sorted(set(target_by_id) - set(source_by_id))
        differing = sorted(
            point_id for point_id in set(source_by_id) & set(target_by_id)
            if _canonical(source_by_id[point_id]) != _canonical(target_by_id[point_id])
        )
        if missing or extra or differing:
            raise ValueError(
                f"restore verification failed: missing={len(missing)}, "
                f"extra={len(extra)}, differing={len(differing)}"
            )
        result = {
            "protocol": "agent_test8_qdrant_full_point_export_restore_v1",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_collection": SOURCE, "target_collection": TARGET,
            "source_before": source_before,
            "target_after": {"points_count": target_info.points_count,
                             "exact_scanned_points": len(target_rows),
                             "status": str(target_info.status),
                             "vector_size": target_info.config.params.vectors.size,
                             "distance": str(target_info.config.params.vectors.distance)},
            "snapshot_file": snapshot,
            "restore_method": "canonical full point export JSONL and Qdrant upsert into new collection",
            "restore_status": "ok_exact_ids_payloads_vectors_match",
            "verification": {"missing_ids": len(missing), "extra_ids": len(extra),
                             "differing_points": len(differing),
                             "all_ids_payloads_vectors_identical": True},
            "source_modified": False,
        }
        out = RUN_DIR / "preflight" / "qdrant_copy_manifest.json"
        out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
