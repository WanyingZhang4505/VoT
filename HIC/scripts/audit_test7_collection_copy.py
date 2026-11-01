"""Export and compare Test 6/Test 7 Qdrant points without changing either collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "LLM/evaluations/Agent Test 7/preflight_20261030"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def export(client, name):
    rows = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=128, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                if len(vector) != 1:
                    raise ValueError(f"{name} point {point.id} has unexpected named vectors")
                vector = next(iter(vector.values()))
            rows.append({"id": str(point.id), "vector": list(vector),
                         "payload": point.payload or {}})
        if offset is None:
            break
    rows.sort(key=lambda row: row["id"])
    path = OUT / f"{name}_raw_points.jsonl"
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = (canonical(row) + "\n").encode("utf-8")
            digest.update(line)
            handle.write(line)
    return rows, {"path": str(path.resolve()), "count": len(rows),
                  "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url="http://127.0.0.1:6333")
    try:
        source, source_file = export(client, "agent_test_6")
        target, target_file = export(client, "agent_test_7")
        source_by_id = {row["id"]: row for row in source}
        target_by_id = {row["id"]: row for row in target}
        missing_target = sorted(set(source_by_id) - set(target_by_id))
        extra_target = sorted(set(target_by_id) - set(source_by_id))
        differing = sorted(
            point_id for point_id in set(source_by_id) & set(target_by_id)
            if canonical(source_by_id[point_id]) != canonical(target_by_id[point_id])
        )
        result = {
            "source_collection": "agent_test_6", "target_collection": "agent_test_7",
            "source": source_file, "target": target_file,
            "missing_target_ids": missing_target, "extra_target_ids": extra_target,
            "different_point_ids": differing,
            "all_ids_payloads_vectors_identical": not missing_target and not extra_target and not differing,
            "read_only": True,
        }
        (OUT / "collection_copy_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"missing_target_ids", "extra_target_ids", "different_point_ids"}},
                         ensure_ascii=False, indent=2))
        print(json.dumps({"missing_ids": len(missing_target),
                          "extra_ids": len(extra_target),
                          "different_points": len(differing)}, ensure_ascii=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
