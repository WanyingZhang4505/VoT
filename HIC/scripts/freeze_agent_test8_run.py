"""Run the final target-blind preflight and freeze Test 8 implementation hashes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic
from LLM.hic_agent_validation import PARAMETERS, PROTOCOL
from qdrant_client import QdrantClient
from scripts.run_agent_test6 import _embedder
from scripts.run_agent_test8 import PROTOCOL_ID


RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
FILES = (
    "scripts/prepare_agent_test8_manifest.py",
    "scripts/fork_test8_qdrant_collection.py",
    "scripts/run_agent_test8.py",
    "LLM/hic_agent_validation.py",
    "LLM/hic_vnext.py",
    "LLM/hic.py",
    "LLM/native_hic_data.py",
    "LLM/qdrant_store.py",
    "LLM/hic_codex_agent.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = RUN_DIR / "run_manifest.json"
    jobs_path = RUN_DIR / "input_snapshots.json"
    copy_path = RUN_DIR / "preflight/qdrant_copy_manifest.json"
    freeze_path = RUN_DIR / "frozen_protocol.json"
    preflight_path = RUN_DIR / "preflight/preflight_evidence.json"
    if freeze_path.exists() or preflight_path.exists():
        raise FileExistsError("Test 8 preflight/freeze evidence already exists; do not replace it")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    copy = json.loads(copy_path.read_text(encoding="utf-8"))
    if len(jobs) != 15 or manifest.get("job_count") != 15:
        raise ValueError("frozen Test 8 manifest is not 15 cases")
    if manifest.get("validation_truth_read") is not False or manifest.get("outer_test_targets_read") is not False:
        raise ValueError("truth seal is not intact")
    if copy.get("restore_status") != "ok_exact_ids_payloads_vectors_match":
        raise ValueError("Test 8 Qdrant restore is not exact")
    if copy["target_after"]["points_count"] != 703:
        raise ValueError("Test 8 point count differs from copied Test 7 source")
    for cfg in manifest["configurations"].values():
        if sha(Path(cfg["source_path"])) != cfg["source_sha256"]:
            raise ValueError(f"source data hash changed for {cfg['dataset']}/H{cfg['horizon']}")
    for job in jobs:
        if any(key in job["case"] for key in ("truth", "validation_truth", "target_values")):
            raise ValueError(f"truth-bearing target-blind input: {job['case_id']}")

    ready_status = urllib.request.urlopen("http://127.0.0.1:6333/readyz", timeout=5).status
    if ready_status != 200:
        raise RuntimeError(f"Qdrant /readyz returned HTTP {ready_status}")
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
    try:
        collection_names = {item.name for item in client.get_collections().collections}
        if not {"agent_test_7", "agent_test_8"}.issubset(collection_names):
            raise RuntimeError("Test 7 or Test 8 Qdrant collection is missing")
        infos = {name: client.get_collection(name) for name in ("agent_test_7", "agent_test_8")}
        for name, info in infos.items():
            vector = info.config.params.vectors
            if (str(info.status).lower() != "green" or info.points_count != 703
                    or vector.size != 1024 or str(vector.distance).lower() != "cosine"):
                raise RuntimeError(f"live {name} state differs from the frozen copy")
    finally:
        client.close()

    embedder = _embedder()
    vector = embedder.encode(["Agent Test 8 frozen offline embedding preflight"],
                             is_query=True)[0]
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if (embedder.model_name != "BAAI/bge-m3" or len(vector) != 1024
            or not all(math.isfinite(float(value)) for value in vector)
            or not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)):
        raise RuntimeError("offline BGE-M3 embedding check failed")
    preflight = {
        "protocol": "agent_test8_preflight_v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "docker_container": "researchgnn-qdrant-1",
        "qdrant_version": "1.19.2",
        "qdrant_rest_url": "http://127.0.0.1:6333",
        "qdrant_readyz_http_status": ready_status,
        "qdrant_collections": {
            name: {"status": str(info.status), "points_count": info.points_count,
                   "vector_size": info.config.params.vectors.size,
                   "distance": str(info.config.params.vectors.distance)}
            for name, info in infos.items()},
        "embedding": {"model": embedder.model_name, "dimension": len(vector),
                      "max_length": embedder.max_length, "finite": True,
                      "norm": norm, "smoke_vector_sha256": hashlib.sha256(
                          json.dumps([float(x) for x in vector], separators=(",", ":")).encode()
                      ).hexdigest()},
        "scope": {"job_count": len(jobs), "group_counts": manifest["group_counts"],
                  "case_ids": [job["case_id"] for job in jobs],
                  "cases_with_nonempty_text": sum(bool(job["case"].get("news")) for job in jobs),
                  "validation_truth_read": False, "outer_test_targets_read": False},
        "target_data_sha256": {key: value["source_sha256"]
                               for key, value in manifest["configurations"].items()},
        "qdrant_copy_manifest_sha256": sha(copy_path),
        "target_collection_writes_from_test8": 0,
    }
    preflight_path.write_text(json.dumps(preflight, ensure_ascii=False,
                                         sort_keys=True, indent=2) + "\n", encoding="utf-8")
    frozen = {
        "protocol": PROTOCOL_ID,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest_sha256": sha(manifest_path), "input_snapshots_sha256": sha(jobs_path),
        "jobs_sha256": manifest["jobs_sha256"],
        "implementation_files": {name: sha(ROOT / name) for name in FILES},
        "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
        "paired_forecast_protocol": PROTOCOL, "paired_forecast_parameters": PARAMETERS,
        "retrieval": "agent_test_8 direct_nearest Top-1; same Test 7 search semantics; no eligibility filtering",
        "embedding": {key: preflight["embedding"][key]
                      for key in ("model", "dimension", "max_length")},
        "score_protocol": "one A and one B; exact per-case truth reveal only after forecast lock; no revision",
        "qdrant_copy_manifest_sha256": sha(copy_path),
        "preflight_evidence_sha256": sha(preflight_path),
        "source_collection_immutable": True,
        "target_collection_frozen_point_count": 703,
        "truth_read_before_first_prediction": False,
        "outer_test_targets_read": False,
    }
    freeze_path.write_text(json.dumps(frozen, ensure_ascii=False,
                                     sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen_ready_before_predictions", "preflight": str(preflight_path),
                      "preflight_sha256": sha(preflight_path),
                      "frozen_protocol": str(freeze_path),
                      "frozen_protocol_sha256": sha(freeze_path),
                      "jobs_sha256": manifest["jobs_sha256"],
                      "implementation_file_sha256": frozen["implementation_files"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
