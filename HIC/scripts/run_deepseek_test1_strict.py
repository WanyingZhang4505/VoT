"""Probe strict DeepSeek forecast calls, then run a frozen 15-case replicate."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic, hic_agent_validation
from LLM.deepseek_strict_tool_provider import (
    BACKEND, PROMPT_VERSION, PROVIDER_PROTOCOL, DeepSeekStrictRequestCache,
    build_forecast_tool, preflight_configuration,
)
from LLM.qdrant_store import QdrantHicStore
from scripts import freeze_agent_test8_deepseek as freezer
from scripts import run_agent_test8_deepseek as core
from scripts.run_agent_test6 import _embedder


HIC_ROOT = ROOT.parents[2]
COLLECTION = "DeepSeek Test 1"
PROTOCOL_ID = "agent_test8_deepseek_strict_function_call_one_pass_v1"
FORECAST_PROTOCOL = "agent_test8_deepseek_strict_fixed_fields_forecast_v1"
RUN_ID = "deepseek_test1_strict_full15_20261101_v2"
PILOT_RUN_ID = "deepseek_test1_strict_pilot4_20261101_v2"
ARTIFACT_ROOT = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_strict_function_call_v2"
RUN_DIR = ARTIFACT_ROOT / "full15"
PILOT_DIR = ARTIFACT_ROOT / "pilot4"
PRIOR_STRICT_PILOT_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_strict_function_call/pilot4"
PILOT_ORDINALS = (7, 9, 13, 14)
PREVIOUS_RUN_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2/onepass_15"
SNAPSHOT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_snapshot/agent_test_8_filtered_703.jsonl"
SOURCE_AUDIT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_source_audit.json"
RESTORE_RECEIPT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_restore_receipt.json"
_ORIGINAL_LOCK = core._lock_forecast


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _configure_freezer() -> None:
    freezer.RUN_ID = RUN_ID
    freezer.RUN_DIR = RUN_DIR
    freezer.COLLECTION = COLLECTION
    freezer.PROTOCOL_ID = PROTOCOL_ID
    freezer.FORECAST_PROTOCOL = FORECAST_PROTOCOL
    freezer.PROVIDER_PROTOCOL = PROVIDER_PROTOCOL
    freezer.PROMPT_VERSION = PROMPT_VERSION
    freezer.SNAPSHOT = SNAPSHOT
    freezer.preflight_configuration = preflight_configuration
    freezer.IMPLEMENTATION_FILES = tuple(dict.fromkeys((
        *freezer.IMPLEMENTATION_FILES,
        "LLM/deepseek_strict_tool_provider.py",
        "scripts/run_deepseek_test1_strict.py",
    )))


def _configure_runtime(run_dir: Path = RUN_DIR) -> None:
    core.RUN_DIR = run_dir
    core.COLLECTION = COLLECTION
    core.PROTOCOL_ID = PROTOCOL_ID
    core.FORECAST_PROTOCOL = FORECAST_PROTOCOL
    core.PROVIDER_PROTOCOL = PROVIDER_PROTOCOL
    core.PROMPT_VERSION = PROMPT_VERSION
    core.DeepSeekRequestCache = DeepSeekStrictRequestCache
    core.preflight_configuration = preflight_configuration
    core._configure_runtime()
    hic_agent_validation.PROTOCOL = FORECAST_PROTOCOL
    def strict_lock(manifest: dict, job: dict, state: dict) -> dict:
        cfg = manifest["configurations"][job["configuration"]]
        cache = DeepSeekStrictRequestCache(
            run_dir / "provider_calls", run_id=manifest["run_id"],
            dataset=job["dataset"], horizon=job["horizon"],
            frequency=cfg["frequency"], target="unknown", unit="unknown",
        )
        state["llm_backend"] = BACKEND
        state["generation_config"] = cache.generation_config()
        return _ORIGINAL_LOCK(manifest, job, state)

    core._lock_forecast = strict_lock


def _freeze_full_run() -> dict:
    if RUN_DIR.exists() or PILOT_DIR.exists():
        raise FileExistsError("strict function-call run or pilot directory already exists; refusing overwrite")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key is not configured")
    old_progress = json.loads((PREVIOUS_RUN_DIR / "progress.json").read_text(encoding="utf-8"))
    old_jobs = json.loads((PREVIOUS_RUN_DIR / "input_snapshots.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((PREVIOUS_RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if (old_progress.get("status") != "complete" or len(old_jobs) != 15
            or old_manifest.get("validation_truth_read") is not False
            or old_progress.get("outer_test_targets_read") is not False):
        raise ValueError("source DeepSeek run is not the expected target-blind 15-case run")
    prior_pilot_gate_path = PRIOR_STRICT_PILOT_DIR / "pilot_gate.json"
    prior_pilot_gate = json.loads(prior_pilot_gate_path.read_text(encoding="utf-8"))
    if (prior_pilot_gate.get("pass") is not False
            or prior_pilot_gate.get("validation_truth_read") is not False
            or prior_pilot_gate.get("outer_test_targets_read") is not False):
        raise ValueError("prior strict-call pilot is not the preserved pre-forecast failure")
    failures = []
    for ordinal in PILOT_ORDINALS:
        job = old_jobs[ordinal - 1]
        state_path = next((PREVIOUS_RUN_DIR / "cases").glob(f"{ordinal:02d}_*.json"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("status") != "technical_failure"
                or state.get("truth_read_audit") is not None):
            raise ValueError(f"case {ordinal} is not an unrevealed H192 length-failure case")
        failures.append({"ordinal": ordinal, "case_id": job["case_id"],
                         "state_sha256": _sha_file(state_path)})

    _configure_freezer()
    freeze_result = freezer.build()
    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "protocol": PROTOCOL_ID,
        "forecast_protocol": FORECAST_PROTOCOL,
        "llm_prompt_version": PROMPT_VERSION,
        "forecast_output_contract": "strict tool call with one required numeric field per horizon step",
        "qdrant_collection": COLLECTION,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "pilot_gate_ordinals": list(PILOT_ORDINALS),
        "pilot_failure_source_run": str(PREVIOUS_RUN_DIR.relative_to(HIC_ROOT).as_posix()),
    })
    _write(manifest_path, manifest)
    preflight_path = RUN_DIR / "preflight/deepseek_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight.update({"strict_provider": preflight_configuration(),
                      "strict_schema_sha256_by_horizon": {
                          str(h): _canonical_hash(build_forecast_tool(h))
                          for h in sorted({job["horizon"] for job in old_jobs})
                      },
                      "api_calls": 0, "validation_truth_read": False,
                      "outer_test_targets_read": False})
    _write(preflight_path, preflight)
    frozen_path = RUN_DIR / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen.update({
        "protocol": PROTOCOL_ID,
        "provider_protocol": PROVIDER_PROTOCOL,
        "forecast_protocol": FORECAST_PROTOCOL,
        "run_manifest_sha256": _sha_file(manifest_path),
        "provider_settings": preflight_configuration(),
        "strict_tool_call": {
            "endpoint": preflight_configuration()["forecast_endpoint"],
            "tool_name": "submit_forecast",
            "schema_sha256_by_horizon": preflight["strict_schema_sha256_by_horizon"],
            "required_field_pattern": "step_001..step_{horizon:03d}",
            "required_fields_only": True,
            "max_tokens": preflight_configuration()["max_tokens"],
        },
        "pilot_gate": {
            "ordinals": list(PILOT_ORDINALS),
            "required": "all four cases lock A and B with exact horizon lengths; truth remains unread",
            "full15_starts_only_after_pass": True,
        },
        "source_failed_cases": failures,
        "prior_strict_pilot_failure": {
            "path": str(PRIOR_STRICT_PILOT_DIR.relative_to(HIC_ROOT).as_posix()),
            "gate_sha256": _sha_file(prior_pilot_gate_path),
            "strict_forecast_calls_submitted": 0,
            "summary_calls_submitted": 4,
            "validation_truth_read": False,
        },
        "source_failed_run_progress_sha256": _sha_file(PREVIOUS_RUN_DIR / "progress.json"),
        "source_memory_audit_sha256": _sha_file(SOURCE_AUDIT),
        "qdrant_restore_receipt_sha256": _sha_file(RESTORE_RECEIPT),
        "frozen_utc": _utc(),
    })
    _write(frozen_path, frozen)

    _freeze_pilot(manifest, json.loads((RUN_DIR / "input_snapshots.json").read_text(encoding="utf-8")), frozen)
    return {"status": "strict_full15_and_pilot4_frozen_before_inference",
            "full_run_id": RUN_ID, "full_run_dir": str(RUN_DIR),
            "pilot_run_id": PILOT_RUN_ID, "pilot_dir": str(PILOT_DIR),
            "pilot_ordinals": list(PILOT_ORDINALS), "pilot_cases": failures,
            "source_collection": COLLECTION, "source_points": 703,
            "max_tokens": preflight_configuration()["max_tokens"],
            "api_calls": 0, "validation_truth_read": False,
            "outer_test_targets_read": False, "freeze_result": freeze_result}


def _freeze_pilot(full_manifest: dict, full_jobs: list[dict], full_frozen: dict) -> None:
    selected = []
    for ordinal in PILOT_ORDINALS:
        source = full_jobs[ordinal - 1]
        job = dict(source)
        job["run_id"] = PILOT_RUN_ID
        job["job_id"] = _canonical_hash({"run_id": PILOT_RUN_ID,
                                         "ordinal": ordinal,
                                         "case_id": job["case_id"]})
        selected.append(job)
    configs = full_manifest["configurations"]
    counts = Counter(f"{job['dataset']}/H{job['horizon']}" for job in selected)
    pilot_manifest = {
        **full_manifest,
        "run_id": PILOT_RUN_ID,
        "protocol": PROTOCOL_ID,
        "collection_name": COLLECTION,
        "group_counts": dict(sorted(counts.items())),
        "jobs_sha256": _canonical_hash(selected),
        "case_count": len(selected),
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "new_cases_written_to_retrieval_collection": False,
        "status": "strict_tool_pilot_frozen_target_blind",
        "source_full_run_id": RUN_ID,
    }
    PILOT_DIR.mkdir(parents=True, exist_ok=False)
    _write_new(PILOT_DIR / "run_manifest.json", pilot_manifest)
    _write_new(PILOT_DIR / "input_snapshots.json", selected)
    pilot_protocol = {
        "protocol": PROTOCOL_ID,
        "provider_protocol": PROVIDER_PROTOCOL,
        "forecast_protocol": FORECAST_PROTOCOL,
        "created_utc": _utc(),
        "full_run_id": RUN_ID,
        "full_frozen_protocol_sha256": _sha_file(RUN_DIR / "frozen_protocol.json"),
        "pilot_run_manifest_sha256": _sha_file(PILOT_DIR / "run_manifest.json"),
        "pilot_jobs_sha256": pilot_manifest["jobs_sha256"],
        "implementation_sha256": {
            "provider": _sha_file(ROOT / "LLM/deepseek_strict_tool_provider.py"),
            "runner": _sha_file(Path(__file__)),
        },
        "memory_collection": COLLECTION,
        "memory_point_count": 703,
        "cases": [{"ordinal": job["ordinal"], "case_id": job["case_id"],
                   "input_sha256": job["input_sha256"], "dataset": job["dataset"],
                   "horizon": job["horizon"]} for job in selected],
        "call_plan": "one summary call, one strict A call, one strict B call per pilot case",
        "stop_condition": "continue only if all 8 A/B forecast tool calls validate and lock at exact horizon",
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "status": "frozen_before_inference",
    }
    _write_new(PILOT_DIR / "pilot_protocol.json", pilot_protocol)
    _write_new(PILOT_DIR / "progress.json", {
        "run_id": PILOT_RUN_ID, "total": len(selected), "forecast_locked": 0,
        "scored": 0, "technical_failures": 0, "api_calls": 0,
        "validation_truth_read": False, "outer_test_targets_read": False,
        "qdrant_writes": 0, "status": "pilot_frozen_pending_preflight",
    })


def _verify_collection(store: QdrantHicStore) -> dict:
    if not store.client.collection_exists(COLLECTION):
        raise RuntimeError(f"{COLLECTION} is unavailable")
    info = store.client.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    if (str(info.status).lower() != "green" or info.points_count != 703
            or vectors.size != 1024 or str(vectors.distance).lower() != "cosine"):
        raise RuntimeError("strict-call run requires the green 703-point 1024-D Cosine collection")
    return {"status": str(info.status), "points_count": info.points_count,
            "dimension": vectors.size, "distance": str(vectors.distance)}


def _preflight() -> dict:
    _configure_runtime()
    manifest, jobs = core._load()
    provider = preflight_configuration()
    issues = []
    if not provider["api_key_configured"]:
        issues.append("DeepSeek API key missing")
    try:
        store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
        try:
            collection = _verify_collection(store)
        finally:
            store.client.close()
    except Exception as error:
        collection = {"status": "unavailable"}
        issues.append(f"Qdrant {type(error).__name__}: {str(error)[:180]}")
    try:
        embedder = _embedder()
        vector = [float(value) for value in embedder.encode(
            [hic.summary_to_embedding_text({"strict_call_preflight": True})],
            is_query=True,
        )[0]]
        norm = sum(value * value for value in vector) ** 0.5
        if (len(vector) != 1024 or not all(map(lambda x: x == x and abs(x) != float("inf"), vector))
                or abs(norm - 1.0) > 1e-6
                or embedder.model_name != "BAAI/bge-m3"):
            raise ValueError("offline BGE-M3 query vector failed 1024-D unit check")
        embedding = {"status": "ok", "model": embedder.model_name,
                     "dimension": len(vector), "norm": norm}
    except Exception as error:
        embedding = {"status": "unavailable"}
        issues.append(f"BGE-M3 {type(error).__name__}: {str(error)[:180]}")
    result = {"protocol": PROTOCOL_ID, "checked_utc": _utc(),
              "ready": not issues, "issues": issues,
              "full_run_id": manifest["run_id"], "case_count": len(jobs),
              "provider": provider, "qdrant": collection, "embedding": embedding,
              "validation_truth_read": False, "outer_test_targets_read": False,
              "qdrant_writes": 0}
    _write(RUN_DIR / "preflight" / "strict_live_preflight.json", result)
    return result


def _pilot_case(job: dict, manifest: dict) -> dict:
    cfg = manifest["configurations"][job["configuration"]]
    state_path = PILOT_DIR / "cases" / f"{job['ordinal']:02d}_{job['job_id']}.json"
    state = {"run_id": PILOT_RUN_ID, "ordinal": job["ordinal"],
             "case_id": job["case_id"], "status": "pending",
             "frozen_case_input_sha256": job["input_sha256"]}
    _write_new(state_path, state)
    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
    try:
        collection = _verify_collection(store)
        embedder = _embedder()
        cache = DeepSeekStrictRequestCache(
            PILOT_DIR / "provider_calls", run_id=PILOT_RUN_ID,
            dataset=job["dataset"], horizon=job["horizon"],
            frequency=cfg["frequency"], target="unknown", unit="unknown",
        )
        state["llm_backend"] = BACKEND
        state["deepseek_model"] = cache.model
        state["generation_config"] = cache.generation_config()
        state["qdrant_initial_collection"] = collection
        result = hic_agent_validation.advance_agent_case(
            job["case"], state, contract=cfg["contract"],
            template=cfg["template"], dataset_description=cfg["description"],
            cache=cache, store=store,
            embed_summary=lambda summary: embedder.encode(
                [hic.summary_to_embedding_text(summary)], is_query=True)[0],
            persist=lambda: _write(state_path, state),
            retrieval_mode="direct_nearest",
        )
        selected = result.get("retrieval_trace", {}).get("selected", [])
        if len(selected) != 1:
            raise ValueError("pilot did not retrieve exactly one direct-nearest point")
        points = store.client.retrieve(collection_name=COLLECTION,
                                       ids=[selected[0]["point_id"]],
                                       with_payload=True, with_vectors=False)
        if len(points) != 1:
            raise ValueError("retrieved memory point disappeared")
        selected[0]["source_metadata"] = (points[0].payload or {}).get("metadata", {})
        for arm in ("baseline", "full_hic"):
            prediction = state[arm].get("prediction")
            if not isinstance(prediction, list) or len(prediction) != job["horizon"]:
                raise ValueError(f"{arm} did not produce exactly {job['horizon']} values")
        state["locked_provider_responses"] = {}
        for stage, call_id in state.get("request_task_ids", {}).items():
            record = cache.get_record(call_id, state["requests"][stage])
            attempt = record["attempts"][-1]
            state["locked_provider_responses"][stage] = {
                "call_id": call_id,
                "request_sha256": record["request_identity"]["sha256"],
                "prompt_sha256": record["request_identity"]["prompt_sha256"],
                "request_body_sha256": record["request_body_sha256"],
                "raw_response_sha256": attempt.get("raw_assistant_content_sha256"),
                "raw_response_text": attempt.get("raw_assistant_content"),
                "response_source": BACKEND,
                "model": attempt.get("returned_model", cache.model),
                "usage": attempt.get("usage"),
                "finish_reason": attempt.get("finish_reason"),
                "tool_call_id": attempt.get("tool_call_id"),
            }
        state["status"] = "forecast_complete"
        state["validation_truth_read"] = False
        _write(state_path, state)
        _configure_runtime(PILOT_DIR)
        lock = core._lock_forecast(manifest, job, state)
        return {"status": "forecast_locked", "ordinal": job["ordinal"],
                "case_id": job["case_id"],
                "A_count": len(state["baseline"]["prediction"]),
                "B_count": len(state["full_hic"]["prediction"]),
                "forecast_lock_sha256": lock["forecast_lock_sha256"],
                "retrieved_point_id": selected[0]["point_id"],
                "retrieved_source_case_id": selected[0]["case_id"],
                "validation_truth_read": False}
    except Exception as error:
        failure = {"ordinal": job["ordinal"], "case_id": job["case_id"],
                   "type": type(error).__name__, "message": str(error)[:500],
                   "recorded_utc": _utc(), "validation_truth_read": False,
                   "outer_test_targets_read": False, "qdrant_writes": 0}
        state["status"] = "technical_failure"
        state["technical_failure"] = failure
        _write(state_path, state)
        _write(PILOT_DIR / "technical_failures" /
               f"{job['ordinal']:02d}.json", failure)
        return {"status": "technical_failure", "ordinal": job["ordinal"],
                "case_id": job["case_id"], "failure": failure}
    finally:
        store.client.close()


def _run_pilot() -> dict:
    _configure_runtime(RUN_DIR)
    full_manifest, full_jobs = core._load()
    pilot_manifest = json.loads((PILOT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    pilot_jobs = json.loads((PILOT_DIR / "input_snapshots.json").read_text(encoding="utf-8"))
    protocol = json.loads((PILOT_DIR / "pilot_protocol.json").read_text(encoding="utf-8"))
    if (protocol.get("status") != "frozen_before_inference"
            or protocol.get("full_frozen_protocol_sha256") != _sha_file(RUN_DIR / "frozen_protocol.json")
            or pilot_manifest.get("jobs_sha256") != _canonical_hash(pilot_jobs)
            or len(pilot_jobs) != 4 or protocol.get("validation_truth_read") is not False):
        raise ValueError("pilot protocol or selected case hashes failed freeze validation")
    preflight = json.loads((RUN_DIR / "preflight/strict_live_preflight.json").read_text(encoding="utf-8"))
    if (preflight.get("ready") is not True or preflight.get("validation_truth_read") is not False
            or preflight.get("outer_test_targets_read") is not False):
        raise RuntimeError("strict-call live preflight is not ready and truth-blind")
    if not preflight_configuration()["api_key_configured"]:
        raise RuntimeError("DeepSeek API key is not configured")
    results = []
    for job in pilot_jobs:
        results.append(_pilot_case(job, pilot_manifest))
    passed = len(results) == 4 and all(
        row.get("status") == "forecast_locked"
        and row.get("A_count") == row.get("B_count") ==
        next(job["horizon"] for job in pilot_jobs if job["ordinal"] == row["ordinal"])
        for row in results
    )
    gate = {"protocol": PROVIDER_PROTOCOL, "pilot_run_id": PILOT_RUN_ID,
            "completed_utc": _utc(), "cases": results,
            "pilot_case_count": 4, "strict_forecasts_valid": sum(
                row.get("status") == "forecast_locked" for row in results),
            "pass": passed, "full15_authorized_by_gate": passed,
            "validation_truth_read": False, "outer_test_targets_read": False,
            "qdrant_writes": 0}
    _write(PILOT_DIR / "pilot_gate.json", gate)
    _write(PILOT_DIR / "progress.json", {
        "run_id": PILOT_RUN_ID, "total": 4,
        "forecast_locked": gate["strict_forecasts_valid"], "scored": 0,
        "technical_failures": 4 - gate["strict_forecasts_valid"],
        "api_calls": sum(3 if row.get("status") == "forecast_locked" else 0
                         for row in results),
        "validation_truth_read": False, "outer_test_targets_read": False,
        "qdrant_writes": 0, "status": "pilot_pass" if passed else "pilot_no_go",
    })
    return gate


def _require_pilot_pass() -> dict:
    gate_path = PILOT_DIR / "pilot_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("four-case strict tool pilot has not finished")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("pass") is not True or gate.get("full15_authorized_by_gate") is not True
            or gate.get("validation_truth_read") is not False
            or gate.get("outer_test_targets_read") is not False):
        raise RuntimeError("pilot did not pass; the full 15-case run is blocked")
    return gate


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "freeze":
        result = _freeze_full_run()
    elif command == "preflight":
        _configure_runtime()
        result = _preflight()
    elif command == "pilot-four":
        result = _run_pilot()
    elif command == "pilot-status":
        path = PILOT_DIR / "pilot_gate.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(
            (PILOT_DIR / "progress.json").read_text(encoding="utf-8"))
    elif command == "run-all":
        gate = _require_pilot_pass()
        _configure_runtime()
        result = {"pilot_gate_sha256": _sha_file(PILOT_DIR / "pilot_gate.json"),
                  "pilot_pass": gate["pass"], "run": core.run_all()}
    elif command == "advance-one":
        _require_pilot_pass()
        _configure_runtime()
        result = core.advance_one()
    elif command == "score-one":
        _require_pilot_pass()
        _configure_runtime()
        result = core.frozen_runtime.score_one()
    elif command == "status":
        _configure_runtime()
        result = core.frozen_runtime.status()
    elif command == "finalize":
        _configure_runtime()
        result = core.frozen_runtime.finalize()
    else:
        raise SystemExit("usage: run_deepseek_test1_strict.py {freeze|preflight|pilot-four|pilot-status|run-all|advance-one|score-one|status|finalize}")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
