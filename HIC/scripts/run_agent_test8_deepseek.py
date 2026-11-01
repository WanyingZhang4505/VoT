"""Run the frozen 15-case Agent Test 8 protocol using DeepSeek API calls."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic, hic_agent_validation
from LLM.deepseek_test8_provider import (
    DeepSeekRequestCache, PROVIDER_PROTOCOL, preflight_configuration,
)
from LLM.native_hic_data import load_native_hic_validation_truth_for_case
from LLM.qdrant_store import QdrantHicStore
from scripts.run_agent_test6 import _embedder
from scripts import run_agent_test8 as frozen_runtime


RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/deepseek_onepass_15_20261101_r2"
COLLECTION = "agent_test_8_deepseek"
PROTOCOL_ID = "agent_test8_deepseek_one_pass_direct_nearest_v1"
FORECAST_PROTOCOL = "agent_test8_deepseek_paired_forecast_v1"
TERMINAL = {"scored", "technical_failure"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_runtime() -> None:
    frozen_runtime.RUN_DIR = RUN_DIR
    frozen_runtime.COLLECTION = COLLECTION
    frozen_runtime.PROTOCOL_ID = PROTOCOL_ID
    hic_agent_validation.PROTOCOL = FORECAST_PROTOCOL


def _load():
    _configure_runtime()
    manifest, jobs = frozen_runtime._load()
    frozen = json.loads((RUN_DIR / "frozen_protocol.json").read_text(encoding="utf-8"))
    snapshot = ROOT / frozen["retrieval_snapshot"]["path"]
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != frozen[
            "retrieval_snapshot"]["sha256"]:
        raise ValueError("frozen DeepSeek retrieval snapshot hash changed")
    for cfg in manifest["configurations"].values():
        source = (ROOT / cfg["source_path"]).resolve(strict=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != cfg["source_sha256"]:
            raise ValueError(f"source data hash changed for {cfg['dataset']}/H{cfg['horizon']}")
    if frozen.get("provider_protocol") != PROVIDER_PROTOCOL:
        raise ValueError("DeepSeek provider protocol differs from the frozen run")
    return manifest, jobs


def _verify_collection(client: QdrantHicStore) -> dict:
    if not client.client.collection_exists(COLLECTION):
        raise RuntimeError(
            f"{COLLECTION} is missing; restore the 703-point baseline first"
        )
    info = client.client.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    if (str(info.status).lower() != "green" or info.points_count != 703
            or vectors.size != 1024 or str(vectors.distance).lower() != "cosine"):
        raise RuntimeError(
            "DeepSeek Test 8 retrieval collection must be green, 703 points, "
            "1024-dimensional Cosine"
        )
    return {"status": str(info.status), "points_count": info.points_count,
            "dimension": vectors.size, "distance": str(vectors.distance)}


def preflight() -> dict:
    manifest, jobs = _load()
    provider = preflight_configuration()
    issues = []
    if not provider["api_key_configured"]:
        issues.append("DEEPSEEK_API_KEY is not configured in config/api.env")
    qdrant = {"status": "unavailable"}
    try:
        store = QdrantHicStore(
            model_id="BAAI/bge-m3", collection_name=COLLECTION,
        )
        try:
            qdrant = _verify_collection(store)
        finally:
            store.client.close()
    except Exception as error:
        issues.append(f"Qdrant: {type(error).__name__}: {str(error)[:240]}")
    embedding = {"status": "unavailable"}
    try:
        embedder = _embedder()
        vector = [float(value) for value in embedder.encode(
            [hic.summary_to_embedding_text({"preflight": "Agent Test 8 DeepSeek"})],
            is_query=True,
        )[0]]
        norm = math.sqrt(sum(value * value for value in vector))
        if (embedder.model_name != "BAAI/bge-m3" or len(vector) != 1024
                or not all(math.isfinite(value) for value in vector)
                or not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)):
            raise RuntimeError("BGE-M3 output is not a finite unit 1024-vector")
        embedding = {"status": "ok", "model": embedder.model_name,
                     "dimension": len(vector), "norm": norm}
    except Exception as error:
        issues.append(f"Embedding: {type(error).__name__}: {str(error)[:240]}")
    result = {
        "protocol": PROTOCOL_ID, "checked_utc": _utc(),
        "ready": not issues, "issues": issues, "run_id": manifest["run_id"],
        "case_count": len(jobs), "group_counts": manifest["group_counts"],
        "provider": provider, "qdrant": qdrant, "embedding": embedding,
        "validation_truth_read": False, "outer_test_targets_read": False,
        "qdrant_writes": 0,
    }
    output_path = RUN_DIR / "preflight" / "live_preflight_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                      indent=2) + "\n", encoding="utf-8")
    return result


def advance_one() -> dict:
    _configure_runtime()
    manifest, jobs = frozen_runtime._load()
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before inference")
    states = frozen_runtime._read_states(jobs)
    index = frozen_runtime._current(jobs, states)
    if index is None:
        return {"status": "all_cases_terminal",
                "progress": frozen_runtime._progress(manifest, jobs, states)}
    if any(state.get("status") not in TERMINAL for state in states[:index]):
        raise RuntimeError("serial case gate blocked by an earlier unscored case")
    job, state = jobs[index], dict(states[index])
    state.setdefault("frozen_case_input_sha256", job["input_sha256"])
    state.setdefault("ordinal", job["ordinal"])
    state.setdefault("case_id", job["case_id"])
    cfg = manifest["configurations"][job["configuration"]]
    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
    try:
        collection_state = _verify_collection(store)
        embedder = _embedder()
        provider = DeepSeekRequestCache(
            RUN_DIR / "provider_calls", run_id=manifest["run_id"],
            dataset=job["dataset"], horizon=job["horizon"],
            frequency=cfg["frequency"], target="unknown", unit="unknown",
        )
        state["llm_backend"] = "deepseek_api"
        state["deepseek_model"] = provider.model
        state["generation_config"] = {
            "temperature": provider.temperature, "max_tokens": provider.max_tokens,
            "thinking": "disabled", "response_format": "json_object",
        }
        state["qdrant_initial_collection"] = collection_state
        result = hic_agent_validation.advance_agent_case(
            job["case"], state, contract=cfg["contract"],
            template=cfg["template"], dataset_description=cfg["description"],
            cache=provider, store=store,
            embed_summary=lambda summary: embedder.encode(
                [hic.summary_to_embedding_text(summary)], is_query=True)[0],
            persist=lambda: frozen_runtime._atomic_json(
                frozen_runtime._state_path(job), state),
            retrieval_mode="direct_nearest",
        )
        selected = result.get("retrieval_trace", {}).get("selected", [])
        if len(selected) != 1:
            raise ValueError("Test 8 direct-nearest retrieval did not produce one hit")
        source_points = store.client.retrieve(
            collection_name=COLLECTION, ids=[selected[0]["point_id"]],
            with_payload=True, with_vectors=False,
        )
        if len(source_points) != 1:
            raise ValueError("retrieved source point disappeared before forecast lock")
        source_metadata = (source_points[0].payload or {}).get("metadata", {})
        selected[0]["source_metadata"] = source_metadata
        state["locked_provider_responses"] = {}
        for stage, call_id in state.get("request_task_ids", {}).items():
            record = provider.get_record(call_id, state["requests"][stage])
            attempt = record["attempts"][-1]
            state["locked_provider_responses"][stage] = {
                "call_id": call_id,
                "request_sha256": record["request_identity"]["sha256"],
                "prompt_sha256": record["request_identity"]["prompt_sha256"],
                "request_body_sha256": record["request_body_sha256"],
                "raw_response_sha256": attempt.get("raw_assistant_content_sha256"),
                "raw_response_text": attempt.get("raw_assistant_content"),
                "response_source": "deepseek_api",
                "model": attempt.get("returned_model", provider.model),
                "usage": attempt.get("usage"),
            }
    except Exception as error:
        failure = {
            "type": type(error).__name__, "message": str(error)[:500],
            "recorded_utc": _utc(), "validation_truth_read": False,
            "qdrant_writes": 0, "provider": "deepseek_api",
            "request_recovery_allowed": True,
        }
        state.setdefault("technical_attempts", []).append(failure)
        attempt_count = len(state["technical_attempts"])
        state["effective_retry_count"] = max(0, attempt_count - 1)
        state["status"] = "recovery_pending" if attempt_count == 1 else "technical_failure"
        frozen_runtime._atomic_json(frozen_runtime._state_path(job), state)
        frozen_runtime._atomic_json(
            RUN_DIR / "technical_failures" /
            f"{job['ordinal']:02d}_attempt_{attempt_count}.json", failure,
        )
        states[index] = state
        return {"status": state["status"], "ordinal": job["ordinal"],
                "case_id": job["case_id"], "failure": failure,
                "progress": frozen_runtime._progress(manifest, jobs, states)}
    finally:
        store.client.close()

    state["status"] = "forecast_complete"
    state["frozen_case_input_sha256"] = job["input_sha256"]
    state["effective_retry_count"] = len(state.get("technical_attempts", []))
    frozen_runtime._atomic_json(frozen_runtime._state_path(job), state)
    lock = _lock_forecast(manifest, job, state)
    states[index] = state
    return {
        "status": "forecast_locked", "ordinal": job["ordinal"],
        "case_id": job["case_id"], "forecast_lock_sha256": lock["forecast_lock_sha256"],
        "retrieved_point_id": selected[0].get("point_id"),
        "retrieved_source_case_id": selected[0].get("case_id"),
        "progress": frozen_runtime._progress(manifest, jobs, states),
    }


def _lock_forecast(manifest: dict, job: dict, state: dict) -> dict:
    if state.get("status") != "forecast_complete":
        raise ValueError("only a complete A/B pair can be forecast-locked")
    selected = state.get("retrieval_trace", {}).get("selected", [])
    if len(selected) != 1:
        raise ValueError("Test 8 requires one direct-nearest Top-1 retrieval hit")
    hit = selected[0]
    correction = hit.get("correction")
    if not isinstance(correction, str) or not correction:
        raise ValueError("retrieved correction is missing")
    metadata = hit.pop("source_metadata", None) or {}
    source_target_end = metadata.get("target_end") or metadata.get("cutoff_date")
    source_history_start = metadata.get("history_start")
    query_history_start = job["case"].get("history_start")
    time_eligible = (source_target_end < query_history_start
                     if isinstance(source_target_end, str)
                     and isinstance(query_history_start, str) else None)
    hit_audit = {
        **hit, "correction_sha256": frozen_runtime._sha_text(correction),
        "source_case_id": hit.get("case_id"),
        "source_history_start": source_history_start,
        "source_target_end": source_target_end,
        "query_history_start": query_history_start,
        "same_source_dataset": hit.get("source_dataset") == job["dataset"],
        "same_source_case": hit.get("case_id") == job["case_id"],
        "same_history_origin": (isinstance(source_history_start, str)
                                and source_history_start == query_history_start),
        "same_data_source": metadata.get("source_sha256") == manifest[
            "configurations"][job["configuration"]]["source_sha256"],
        "time_eligible_strictly_prior": time_eligible,
        "time_eligibility_policy": "source target_end < query history_start; audit only, never filters retrieval",
    }
    trace = {**state["retrieval_trace"], "selected": [hit_audit],
             "eligibility_filter_applied": False}
    lock = {
        "protocol": PROTOCOL_ID, "run_id": manifest["run_id"],
        "ordinal": job["ordinal"], "case_id": job["case_id"],
        "dataset": job["dataset"], "horizon": job["horizon"],
        "backend": "deepseek_api", "model": state.get("deepseek_model"),
        "generation_config": state.get("generation_config"),
        "frozen_case_input_sha256": job["input_sha256"],
        "semantic_input_sha256": state["input_sha256"],
        "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
        "forecast_protocol_version": FORECAST_PROTOCOL,
        "paired_parameters": frozen_runtime.PARAMETERS,
        "query_embedding_sha256": frozen_runtime.canonical_hash(state["query_embedding"]),
        "baseline": state["baseline"], "full_hic": state["full_hic"],
        "retrieval_trace": trace,
        "request_sha256": state.get("requests", {}),
        "provider_responses": state.get("locked_provider_responses", {}),
        "locked_utc": _utc(), "validation_truth_read": False,
        "outer_test_targets_read": False,
    }
    lock["forecast_lock_sha256"] = frozen_runtime.canonical_hash(lock)
    lock_path = RUN_DIR / "forecast_locks" / f"{job['ordinal']:02d}_{job['job_id']}.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
    state["retrieval_trace"] = trace
    state["forecast_lock_sha256"] = lock["forecast_lock_sha256"]
    state["status"] = "forecast_locked"
    state["forecast_locked_utc"] = lock["locked_utc"]
    frozen_runtime._atomic_json(frozen_runtime._state_path(job), state)
    return lock


def run_all() -> dict:
    """Advance cases serially; stop on the first recoverable technical failure."""
    manifest, jobs = _load()
    for _ in jobs:
        states = frozen_runtime._read_states(jobs)
        index = frozen_runtime._current(jobs, states)
        if index is None:
            break
        if states[index].get("status") == "forecast_locked":
            frozen_runtime.score_one()
            continue
        if states[index].get("status") in {"recovery_pending", "awaiting_worker"}:
            return {"status": "recovery_required", "case_id": jobs[index]["case_id"],
                    "progress": frozen_runtime.status()}
        result = advance_one()
        if result.get("status") == "forecast_locked":
            frozen_runtime.score_one()
        elif result.get("status") == "technical_failure":
            continue
        elif result.get("status") != "all_cases_terminal":
            return result
    states = frozen_runtime._read_states(jobs)
    if all(state.get("status") in TERMINAL for state in states):
        return frozen_runtime.finalize()
    return frozen_runtime.status()


def main() -> None:
    _configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("advance-one")
    sub.add_parser("score-one")
    sub.add_parser("run-all")
    sub.add_parser("status")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight()
    elif args.command == "advance-one":
        result = advance_one()
    elif args.command == "score-one":
        result = frozen_runtime.score_one()
    elif args.command == "run-all":
        result = run_all()
    elif args.command == "status":
        result = frozen_runtime.status()
    else:
        result = frozen_runtime.finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
