"""Restore the frozen 703-point Agent Test 8 baseline without overwriting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / (
    "LLM/evaluations/Agent Test 8/remaining3families_15_20261101/"
    "preflight/qdrant_snapshot/agent_test_7_full_points.jsonl"
)
DEFAULT_COLLECTION = "agent_test_8"
DEFAULT_URL = "http://127.0.0.1:6333"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--receipt-path", type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    rows = []
    point_ids = set()
    for line_number, line in enumerate(snapshot.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        vector = row.get("vector")
        point_id = row.get("id")
        if point_id in point_ids:
            raise ValueError(f"duplicate point ID at snapshot line {line_number}")
        point_ids.add(point_id)
        if (not isinstance(vector, list) or len(vector) != 1024
                or any(not math.isfinite(float(value)) for value in vector)):
            raise ValueError(f"invalid 1024-D vector at snapshot line {line_number}")
        rows.append(row)
    snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if len(rows) != 703:
        raise ValueError(f"expected 703 baseline points; found {len(rows)}")

    client = QdrantClient(url=args.url, timeout=60)
    try:
        if client.collection_exists(args.collection_name):
            raise FileExistsError(
                f"{args.collection_name} already exists; refusing to overwrite"
            )
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE,
            ),
        )
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            client.upsert(
                collection_name=args.collection_name,
                wait=True,
                points=[models.PointStruct(id=row["id"], vector=row["vector"],
                                           payload=row["payload"])
                        for row in batch],
            )
        info = client.get_collection(args.collection_name)
        actual_ids = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=args.collection_name, offset=offset, limit=256,
                with_payload=False, with_vectors=False,
            )
            actual_ids.update(point.id for point in points)
            if offset is None:
                break
        vectors = info.config.params.vectors
        if (info.points_count != 703 or len(actual_ids) != 703
                or actual_ids != point_ids or str(info.status).lower() != "green"
                or vectors.size != 1024
                or str(vectors.distance).lower() != "cosine"):
            raise RuntimeError(
                f"restore verification failed: status={info.status}, "
                f"count={info.points_count}, id_count={len(actual_ids)}"
            )
        receipt_path = args.receipt_path or (
            ROOT / ("restore_receipt.json" if args.collection_name == DEFAULT_COLLECTION
                    else f"restore_receipt_{args.collection_name}.json")
        )
        receipt_path = receipt_path.resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "source_snapshot": str(snapshot.relative_to(ROOT).as_posix()),
            "snapshot_sha256": snapshot_sha256,
            "snapshot_bytes": snapshot.stat().st_size,
            "snapshot_lines": len(rows),
            "collection": args.collection_name,
            "qdrant_url": args.url,
            "points_count": info.points_count,
            "unique_ids_verified": len(actual_ids),
            "status": str(info.status),
            "dimension": vectors.size,
            "distance": "Cosine",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
        }
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
