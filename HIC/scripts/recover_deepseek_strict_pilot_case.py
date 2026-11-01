"""Retry one malformed strict forecast once with identical frozen request data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic, hic_agent_validation
from LLM.deepseek_strict_tool_provider import (
    DeepSeekStrictRequestCache, make_strict_forecast_prompt,
)
from LLM.hic_qdrant_eval import _validated_forecast
from LLM.qdrant_store import QdrantHicStore
from scripts import run_agent_test8_deepseek as core
from scripts import run_deepseek_test1_strict as strict_runner


HIC_ROOT = ROOT.parents[2]
PILOT_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_strict_function_call_v2/pilot4"
RUN_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_strict_function_call_v2/full15"
COLLECTION = "DeepSeek Test 1"
ORDINAL = 7


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _main() -> None:
    protocol_path = PILOT_DIR / "recovery_07_protocol.json"
    receipt_path = PILOT_DIR / "recovery_07_receipt.json"
    if protocol_path.exists() or receipt_path.exists():
        raise FileExistsError("case 7 recovery is already recorded; refusing another retry")
    strict_runner._configure_runtime(RUN_DIR)
    full_manifest, full_jobs = core._load()
    pilot_manifest = json.loads((PILOT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    pilot_jobs = json.loads((PILOT_DIR / "input_snapshots.json").read_text(encoding="utf-8"))
    job = next(row for row in pilot_jobs if row["ordinal"] == ORDINAL)
    state_path = next((PILOT_DIR / "cases").glob(f"{ORDINAL:02d}_*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    initial_gate_path = PILOT_DIR / "pilot_gate.json"
    initial_gate = json.loads(initial_gate_path.read_text(encoding="utf-8"))
    if (initial_gate.get("pass") is not False
            or state.get("status") != "technical_failure"
            or state.get("truth_read_audit") is not None
            or "full_hic" in state
            or len(state.get("baseline", {}).get("prediction", [])) != 192):
        raise ValueError("case 7 is not the expected unrevealed strict-call B failure")
    failure_path = PILOT_DIR / "technical_failures" / "07.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("validation_truth_read") is not False:
        raise ValueError("failed case 7 unexpectedly accessed truth")

    cfg = pilot_manifest["configurations"][job["configuration"]]
    case = job["case"]
    selected = state.get("retrieval_trace", {}).get("selected", [])
    if len(selected) != 1:
        raise ValueError("case 7 must retain its frozen Top-1 retrieval hit")
    temporal_context = {
        "history_period": {"start": case.get("history_start"),
                           "end": case.get("history_end")},
        "forecast_period": {"start": case.get("target_start"),
                            "end": case.get("target_end")},
    }
    prompt = hic_agent_validation.build_paired_direct_forecast_prompt(
        summary=state["summary"], history=case["series"], horizon=job["horizon"],
        frequency=cfg["contract"]["frequency"], target=cfg["contract"]["target"],
        unit=cfg["contract"]["unit"], covariates=case.get("covariates"),
        dataset_description=cfg["description"], temporal_context=temporal_context,
        retrieved_memory=selected[0],
    )
    call_id = (f"{PILOT_DIR.joinpath('run_manifest.json').exists() and pilot_manifest['run_id']}:")
    call_id += f"{job['case_id']}:H{job['horizon']}:full_hic_forecast_paired_v2"
    provider = DeepSeekStrictRequestCache(
        PILOT_DIR / "provider_calls", run_id=pilot_manifest["run_id"],
        dataset=job["dataset"], horizon=job["horizon"],
        frequency=cfg["frequency"], target="unknown", unit="unknown",
    )
    transformed_prompt = make_strict_forecast_prompt(prompt, job["horizon"])
    identity = provider._identity(call_id, transformed_prompt, strict=True)
    failed_record = provider.get_record(call_id, identity["sha256"])
    failed_attempt = failed_record["attempts"][-1]
    if (failed_record["request_identity"]["sha256"] != identity["sha256"]
            or failed_attempt.get("status") != "validation_error"
            or "step_192" not in str(failed_attempt.get("validation_error"))
            or len(failed_record["request_body"]["tools"][0]["function"]["parameters"]["required"]) != 192
            or len(failed_record["request_body"]["tools"][0]["function"]["parameters"]["properties"]) != 192):
        raise ValueError("reconstructed retry differs from the original frozen strict request")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key is not configured")

    strict_runner._configure_runtime(PILOT_DIR)
    recovery_protocol = {
        "protocol": "deepseek_strict_pilot_identical_request_recovery_v1",
        "created_utc": _utc(),
        "pilot_run_id": pilot_manifest["run_id"],
        "ordinal": ORDINAL, "case_id": job["case_id"],
        "case_state_sha256_before_retry": _sha(state_path),
        "technical_failure_sha256": _sha(failure_path),
        "provider_record_sha256_before_retry": _sha(
            provider.output_dir / f"{__import__('hashlib').sha256((call_id + ':' + identity['sha256']).encode()).hexdigest()}.json"
        ),
        "request_identity_sha256": identity["sha256"],
        "prompt_sha256": identity["prompt_sha256"],
        "same_prompt_model_temperature_schema_endpoint_and_inputs": True,
        "retry_limit_for_this_failure": 1,
        "retry_reason": "strict tool call omitted required step_192; no case truth was read",
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "recovery_script_sha256": _sha(Path(__file__)),
    }
    _write_new(protocol_path, recovery_protocol)

    result = provider.invoke(
        call_id, prompt, call_json=None,
        validator=lambda response: _validated_forecast(response, job["horizon"]),
    )
    forecast = result["response"]
    if len(forecast["prediction"]) != job["horizon"]:
        raise ValueError("same-request strict retry did not return the exact horizon")
    state["full_hic"] = forecast
    state.setdefault("requests", {})["full_hic_forecast_paired_v2"] = result[
        "request_identity"]["sha256"]
    state.setdefault("request_task_ids", {})["full_hic_forecast_paired_v2"] = call_id
    state.pop("technical_failure", None)
    state["technical_recovery"] = {
        "protocol": recovery_protocol["protocol"],
        "original_failure_sha256": _sha(failure_path),
        "same_request_retry": True,
        "validation_truth_read": False,
    }
    for stage, task_id in state["request_task_ids"].items():
        record = provider.get_record(task_id, state["requests"][stage])
        attempt = record["attempts"][-1]
        state.setdefault("locked_provider_responses", {})[stage] = {
            "call_id": task_id,
            "request_sha256": record["request_identity"]["sha256"],
            "prompt_sha256": record["request_identity"]["prompt_sha256"],
            "request_body_sha256": record["request_body_sha256"],
            "raw_response_sha256": attempt.get("raw_assistant_content_sha256"),
            "raw_response_text": attempt.get("raw_assistant_content"),
            "response_source": "deepseek_api_strict_function_call",
            "model": attempt.get("returned_model", provider.model),
            "usage": attempt.get("usage"), "finish_reason": attempt.get("finish_reason"),
            "tool_call_id": attempt.get("tool_call_id"),
        }

    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
    try:
        info = store.client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        if (str(info.status).lower() != "green" or info.points_count != 703
                or vectors.size != 1024 or str(vectors.distance).lower() != "cosine"):
            raise RuntimeError("DeepSeek Test 1 collection changed during pilot recovery")
        hit = selected[0]
        point = store.client.retrieve(collection_name=COLLECTION,
                                      ids=[hit["point_id"]], with_payload=True,
                                      with_vectors=False)
        if len(point) != 1:
            raise ValueError("frozen pilot Top-1 point is missing")
        hit["source_metadata"] = (point[0].payload or {}).get("metadata", {})
    finally:
        store.client.close()

    state["generation_config"] = provider.generation_config()
    state["llm_backend"] = "deepseek_api_strict_function_call"
    state["status"] = "forecast_complete"
    state["frozen_case_input_sha256"] = job["input_sha256"]
    _write(state_path, state)
    lock = core._lock_forecast(pilot_manifest, job, state)
    recovery_receipt = {
        "protocol": recovery_protocol["protocol"],
        "ordinal": ORDINAL, "case_id": job["case_id"],
        "same_request_identity_sha256": identity["sha256"],
        "attempt_count": len(provider.get_record(call_id, identity["sha256"])["attempts"]),
        "original_attempt_raw_response_sha256": failed_attempt.get("raw_assistant_content_sha256"),
        "retry_raw_response_sha256": result["request_identity"]["sha256"],
        "A_count": len(state["baseline"]["prediction"]),
        "B_count": len(state["full_hic"]["prediction"]),
        "forecast_lock_sha256": lock["forecast_lock_sha256"],
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "completed_utc": _utc(),
    }
    _write_new(receipt_path, recovery_receipt)
    _write_new(PILOT_DIR / "pilot_gate_initial.json", initial_gate)

    rows = []
    for pilot_job in json.loads((PILOT_DIR / "input_snapshots.json").read_text(encoding="utf-8")):
        pilot_state_path = next((PILOT_DIR / "cases").glob(
            f"{pilot_job['ordinal']:02d}_*.json"))
        pilot_state = json.loads(pilot_state_path.read_text(encoding="utf-8"))
        rows.append({"ordinal": pilot_job["ordinal"], "case_id": pilot_job["case_id"],
                     "status": pilot_state.get("status"),
                     "A_count": len(pilot_state.get("baseline", {}).get("prediction", [])),
                     "B_count": len(pilot_state.get("full_hic", {}).get("prediction", [])),
                     "forecast_lock_sha256": pilot_state.get("forecast_lock_sha256")})
    passed = all(row["status"] == "forecast_locked"
                 and row["A_count"] == row["B_count"] == 192 for row in rows)
    gate = {
        "protocol": "deepseek_test1_strict_tool_forecast_v1",
        "pilot_run_id": pilot_manifest["run_id"],
        "completed_utc": _utc(), "cases": rows,
        "pilot_case_count": 4,
        "strict_forecasts_valid": sum(row["status"] == "forecast_locked" for row in rows),
        "same_request_technical_retries": 1,
        "initial_gate_sha256": _sha(initial_gate_path),
        "recovery_receipt_sha256": _sha(receipt_path),
        "pass": passed, "full15_authorized_by_gate": passed,
        "validation_truth_read": False, "outer_test_targets_read": False,
        "qdrant_writes": 0,
    }
    _write(PILOT_DIR / "pilot_gate.json", gate)
    _write(PILOT_DIR / "progress.json", {
        "run_id": pilot_manifest["run_id"], "total": 4,
        "forecast_locked": gate["strict_forecasts_valid"], "scored": 0,
        "technical_failures": 4 - gate["strict_forecasts_valid"],
        "same_request_technical_retries": 1,
        "validation_truth_read": False, "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "status": "pilot_pass_after_technical_recovery" if passed else "pilot_no_go",
    })
    print(json.dumps({"retry": recovery_receipt, "pilot_gate": gate},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
