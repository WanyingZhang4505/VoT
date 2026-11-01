"""Create a new target-blind Agent Test 8 run bound to the DeepSeek API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic, hic_agent_validation
from LLM.deepseek_test8_provider import (
    PROMPT_VERSION, PROVIDER_PROTOCOL, preflight_configuration,
)
from scripts import run_agent_test8 as codex_runtime
from scripts.run_agent_test8_deepseek import (
    COLLECTION, FORECAST_PROTOCOL, PROTOCOL_ID,
)


SOURCE_RUN = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
RUN_ID = "agent_test8_deepseek_onepass_15_20261101_r2"
RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/deepseek_onepass_15_20261101_r2"
SNAPSHOT = SOURCE_RUN / "preflight/qdrant_snapshot/agent_test_7_full_points.jsonl"
IMPLEMENTATION_FILES = (
    "scripts/run_agent_test8_deepseek.py",
    "scripts/run_agent_test8.py",
    "scripts/run_agent_test6.py",
    "scripts/freeze_agent_test8_deepseek.py",
    "scripts/restore_agent_test8_snapshot.py",
    "LLM/deepseek_test8_provider.py",
    "LLM/deepseek_client.py",
    "LLM/hic_agent_validation.py",
    "LLM/hic.py",
    "LLM/hic_vnext.py",
    "LLM/native_hic_data.py",
    "LLM/qdrant_store.py",
    "config/model.yaml",
    "config/requirements.txt",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(encoded)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")


def _verify_snapshot() -> dict:
    rows = []
    seen_ids = set()
    with SNAPSHOT.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            point_id = row.get("id")
            vector = row.get("vector")
            if not isinstance(point_id, (str, int)) or point_id in seen_ids:
                raise ValueError(f"invalid or duplicate snapshot point ID at line {line_number}")
            seen_ids.add(point_id)
            if not isinstance(vector, list) or len(vector) != 1024:
                raise ValueError(f"snapshot vector at line {line_number} is not 1024-D")
            if any(not isinstance(value, (float, int))
                   or not math.isfinite(float(value)) for value in vector):
                raise ValueError(f"snapshot vector at line {line_number} is not numeric")
            rows.append(row)
    if len(rows) != 703:
        raise ValueError(f"expected 703 frozen baseline points; got {len(rows)}")
    return {"path": str(SNAPSHOT.relative_to(ROOT).as_posix()),
            "sha256": _sha_file(SNAPSHOT), "bytes": SNAPSHOT.stat().st_size,
            "point_count": len(rows), "unique_ids": len(seen_ids),
            "dimension": 1024, "distance": "Cosine"}


def build() -> dict:
    resume_partial_freeze = RUN_DIR.exists()
    if resume_partial_freeze:
        if (RUN_DIR / "frozen_protocol.json").exists():
            raise FileExistsError(f"DeepSeek Test 8 is already frozen: {RUN_DIR}")
        allowed = {
            "run_manifest.json", "input_snapshots.json",
            "preflight/deepseek_preflight.json",
        }
        present = {path.relative_to(RUN_DIR).as_posix()
                   for path in RUN_DIR.rglob("*") if path.is_file()}
        if not present.issubset(allowed):
            raise FileExistsError(
                f"refusing to resume a run directory containing other artifacts: {RUN_DIR}"
            )
    old_manifest = json.loads((SOURCE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    old_jobs = json.loads((SOURCE_RUN / "input_snapshots.json").read_text(encoding="utf-8"))
    old_freeze = json.loads((SOURCE_RUN / "frozen_protocol.json").read_text(encoding="utf-8"))
    source_implementation_matches = all(
        _sha_file(ROOT / name) == digest
        for name, digest in old_freeze.get("implementation_files", {}).items()
    )
    if (not source_implementation_matches
            or _sha_file(SOURCE_RUN / "run_manifest.json") != old_freeze["run_manifest_sha256"]
            or _sha_file(SOURCE_RUN / "input_snapshots.json") != old_freeze["input_snapshots_sha256"]
            or _canonical_hash(old_jobs) != old_manifest.get("jobs_sha256")
            or old_freeze.get("protocol") != codex_runtime.PROTOCOL_ID):
        raise ValueError("original frozen Agent Test 8 inputs failed integrity checks")
    if (len(old_jobs) != 15 or old_manifest.get("validation_truth_read") is not False
            or old_manifest.get("outer_test_targets_read") is not False):
        raise ValueError("the source Test 8 run is not a target-blind 15-case manifest")
    if any(any(key in job.get("case", {}) for key in (
            "truth", "validation_truth", "target_values")) for job in old_jobs):
        raise ValueError("source case snapshot includes validation truth")

    jobs = []
    for source_job in old_jobs:
        job = dict(source_job)
        job["run_id"] = RUN_ID
        identity = {"run_id": RUN_ID, "dataset": job["dataset"],
                    "horizon": job["horizon"], "case_id": job["case_id"]}
        job["job_id"] = _canonical_hash(identity)
        if _canonical_hash(job["case"]) != job["input_sha256"]:
            raise ValueError(f"source input hash changed for {job['case_id']}")
        jobs.append(job)

    counts = Counter(f"{job['dataset']}/H{job['horizon']}" for job in jobs)
    expected = {"Health/H36": 3, "Security/H10": 3, "new_metr/H192": 9}
    if len(jobs) != 15 or dict(sorted(counts.items())) != dict(sorted(expected.items())):
        raise ValueError(f"unexpected frozen scope: {dict(counts)}")

    configurations = json.loads(json.dumps(old_manifest["configurations"]))
    for value in configurations.values():
        relative_source = Path("data") / f"{value['dataset']}.csv"
        source_path = ROOT / relative_source
        if _sha_file(source_path) != value["source_sha256"]:
            raise ValueError(f"pinned source CSV hash changed for {value['dataset']}")
        value["source_path"] = relative_source.as_posix()

    snapshot = _verify_snapshot()
    manifest = {
        **old_manifest,
        "run_id": RUN_ID,
        "protocol": PROTOCOL_ID,
        "experiment_classification": "one_pass_frozen_evaluation_using_test7_memory_bank_deepseek",
        "collection_name": COLLECTION,
        "source_collection": "agent_test_7_snapshot_703",
        "configurations": configurations,
        "jobs_sha256": _canonical_hash(jobs),
        "llm_backend": "deepseek_api",
        "llm_model": str(hic.MODEL),
        "llm_prompt_version": PROMPT_VERSION,
        "group_counts": expected,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "new_cases_written_to_retrieval_collection": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "target_blind_deepseek_manifest_frozen_before_predictions",
    }
    manifest.pop("run_id_original", None)
    preflight = {
        "protocol": "agent_test8_deepseek_preflight_config_v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"case_count": len(jobs), "group_counts": expected,
                  "case_ids": [job["case_id"] for job in jobs]},
        "sources": {key: value["source_sha256"]
                    for key, value in configurations.items()},
        "retrieval_snapshot": snapshot,
        "retrieval_collection": COLLECTION,
        "restore_status": "pending; run the non-overwriting restore command",
        "provider": preflight_configuration(),
        "api_calls": 0,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
    }
    if resume_partial_freeze:
        existing_manifest = json.loads(
            (RUN_DIR / "run_manifest.json").read_text(encoding="utf-8")
        )
        existing_jobs = json.loads(
            (RUN_DIR / "input_snapshots.json").read_text(encoding="utf-8")
        )
        existing_preflight = json.loads(
            (RUN_DIR / "preflight/deepseek_preflight.json").read_text(encoding="utf-8")
        )
        if (existing_manifest.get("run_id") != RUN_ID
                or existing_manifest.get("protocol") != PROTOCOL_ID
                or existing_manifest.get("jobs_sha256") != manifest["jobs_sha256"]
                or existing_manifest.get("collection_name") != COLLECTION
                or existing_manifest.get("validation_truth_read") is not False
                or existing_manifest.get("outer_test_targets_read") is not False
                or existing_jobs != jobs
                or existing_preflight.get("retrieval_snapshot") != snapshot
                or existing_preflight.get("api_calls") != 0
                or existing_preflight.get("validation_truth_read") is not False):
            raise ValueError("partial DeepSeek run directory does not match frozen inputs")
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        _write_new(RUN_DIR / "run_manifest.json", manifest)
        _write_new(RUN_DIR / "input_snapshots.json", jobs)
        _write_new(RUN_DIR / "preflight/deepseek_preflight.json", preflight)
    frozen = {
        "protocol": PROTOCOL_ID,
        "provider_protocol": PROVIDER_PROTOCOL,
        "forecast_protocol": FORECAST_PROTOCOL,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest_sha256": _sha_file(RUN_DIR / "run_manifest.json"),
        "input_snapshots_sha256": _sha_file(RUN_DIR / "input_snapshots.json"),
        "jobs_sha256": manifest["jobs_sha256"],
        "implementation_files": {name: _sha_file(ROOT / name)
                                  for name in IMPLEMENTATION_FILES},
        "provider_settings": preflight["provider"],
        "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
        "paired_forecast_parameters": hic_agent_validation.PARAMETERS,
        "retrieval": "agent_test_8_deepseek direct_nearest Top-1; no eligibility filters",
        "retrieval_snapshot": snapshot,
        "score_protocol": "one A and one B per case; unlock only the corresponding validation truth after forecast lock",
        "source_collection_immutable": True,
        "target_collection": COLLECTION,
        "target_collection_frozen_point_count": 703,
        "qdrant_writes_during_evaluation": 0,
        "truth_read_before_first_prediction": False,
        "outer_test_targets_read": False,
    }
    _write_new(RUN_DIR / "frozen_protocol.json", frozen)
    return {"status": "frozen_ready_pending_api_key_and_qdrant_restore",
            "run_dir": str(RUN_DIR.resolve()), "case_count": len(jobs),
            "group_counts": expected, "jobs_sha256": manifest["jobs_sha256"],
            "snapshot_sha256": snapshot["sha256"],
            "frozen_protocol_sha256": _sha_file(RUN_DIR / "frozen_protocol.json"),
            "api_key_configured": bool(hic.DEEPSEEK_API_KEY),
            "api_calls": 0, "validation_truth_read": False,
            "outer_test_targets_read": False}


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
