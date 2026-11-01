"""Recover malformed H192 forecasts in a separate, auditable supplement.

The primary one-pass run remains immutable. Invalid-length JSON responses are
normalized without truth: keep the first H values when overlong; when short,
carry the final generated value forward to H. Failed raw responses are retained.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic, hic_agent_validation
from LLM.deepseek_test8_provider import DeepSeekRequestCache
from LLM.native_hic_data import load_native_hic_validation_truth_for_case
from LLM.qdrant_store import QdrantHicStore
from scripts import run_agent_test8_deepseek as core
from scripts.run_agent_test6 import _embedder


HIC_ROOT = ROOT.parents[2]
BASE_RUN_ID = "deepseek_test1_onepass_15_20261101_r2"
BASE_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2/onepass_15"
REPAIR_DIR = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2/h192_shape_repair"
SOURCE_AUDIT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_source_audit.json"
RESTORE_RECEIPT = ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_restore_receipt.json"
COLLECTION = "DeepSeek Test 1"
FORECAST_STAGES = {"baseline_no_retrieval_paired_v1", "full_hic_forecast_paired_v2"}
CASE_ORDINALS = (7, 9)
HORIZON = 192
REPAIR_PROTOCOL = "deepseek_test1_h192_shape_recovery_v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.json")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(response: dict, horizon: int) -> tuple[dict, dict]:
    if not isinstance(response, dict):
        raise ValueError("forecast response must be a JSON object")
    prediction = response.get("prediction")
    if (not isinstance(prediction, list) or not prediction
            or any(type(value) not in (int, float) or not math.isfinite(float(value))
                   for value in prediction)):
        raise ValueError("forecast output-shape repair requires a non-empty finite numeric array")
    original_count = len(prediction)
    if original_count > horizon:
        repaired = prediction[:horizon]
        rule = "truncate_tail_to_frozen_horizon"
    elif original_count < horizon:
        repaired = [*prediction, *([prediction[-1]] * (horizon - original_count))]
        rule = "carry_last_generated_value_forward_to_frozen_horizon"
    else:
        repaired = list(prediction)
        rule = "no_shape_change"
    result = dict(response)
    result["prediction"] = repaired
    return result, {"original_count": original_count, "repaired_count": len(repaired),
                    "horizon": horizon, "rule": rule}


def _event_path(call_id: str) -> Path:
    return REPAIR_DIR / "shape_repair_records" / f"{hashlib.sha256(call_id.encode()).hexdigest()}.json"


class ShapeRepairCache:
    def __init__(self, *, manifest: dict, dataset: str, horizon: int,
                 frequency: str, unit: str):
        self.run_id = BASE_RUN_ID
        self.dataset = dataset
        self.horizon = horizon
        self.source = DeepSeekRequestCache(
            BASE_DIR / "provider_calls", run_id=BASE_RUN_ID, dataset=dataset,
            horizon=horizon, frequency=frequency, target="unknown", unit=unit,
        )
        self.fresh = DeepSeekRequestCache(
            REPAIR_DIR / "provider_calls", run_id=BASE_RUN_ID, dataset=dataset,
            horizon=horizon, frequency=frequency, target="unknown", unit=unit,
        )
        self.manifest = manifest
        self._source_projection_ids: set[str] = set()

    def _source_record(self, call_id: str, prompt: str) -> tuple[dict | None, Path | None]:
        identity = self.source._identity(call_id, prompt)
        path = self.source._path(call_id, identity["sha256"])
        if not path.is_file():
            return None, None
        record = _load_json(path)
        if record.get("request_identity", {}).get("sha256") != identity["sha256"]:
            raise ValueError(f"source request identity changed for {call_id}")
        return record, path

    def _save_event(self, call_id: str, source: dict, source_path: Path,
                    attempt: dict, response: dict, shape: dict, *, source_only: bool) -> None:
        try:
            source_path_label = source_path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            source_path_label = source_path.relative_to(REPAIR_DIR).as_posix()
        event = {
            "protocol": REPAIR_PROTOCOL,
            "call_id": call_id,
            "request_identity_sha256": source["request_identity"]["sha256"],
            "request_body_sha256": source.get("request_body_sha256"),
            "source_record_path": source_path_label,
            "source_record_sha256": _sha_file(source_path),
            "source_attempt": attempt.get("attempt"),
            "source_attempt_status": attempt.get("status"),
            "raw_assistant_content_sha256": attempt.get("raw_assistant_content_sha256"),
            "source_only_no_additional_api_call": source_only,
            "shape_repair": shape,
            "normalized_response_sha256": _sha_bytes(_canonical(response)),
            "recorded_utc": _utc(),
            "validation_truth_read": False,
        }
        _atomic_json(_event_path(call_id), event)

    def invoke(self, call_id, prompt, *, call_json, validator=None,
               retry_unresolved=False):
        del retry_unresolved
        stage = call_id.rsplit(":", 1)[-1]
        if stage in FORECAST_STAGES:
            source, source_path = self._source_record(call_id, prompt)
            if source is not None and source.get("attempts"):
                attempt = source["attempts"][-1]
                if attempt.get("status") == "validation_error":
                    if "prediction must contain exactly" not in str(attempt.get("validation_error")):
                        raise ValueError("only exact forecast-length failures are repairable here")
                    raw = attempt.get("raw_assistant_content")
                    parsed = json.loads(raw) if isinstance(raw, str) else None
                    normalized, shape = _normalize(parsed, self.horizon)
                    if validator is not None and validator(normalized) is False:
                        raise ValueError("shape-normalized source response failed forecast validation")
                    self._save_event(call_id, source, source_path, attempt, normalized,
                                     shape, source_only=True)
                    self._source_projection_ids.add(self.source._identity(
                        call_id, prompt)["sha256"])
                    return {"response": normalized,
                            "request_identity": self.source._identity(call_id, prompt)}

            def normalize_then_validate(response):
                normalized, shape = _normalize(response, self.horizon)
                if validator is not None and validator(normalized) is False:
                    return False
                return True

            result = self.fresh.invoke(
                call_id, prompt, call_json=call_json,
                validator=normalize_then_validate,
            )
            record = self.fresh.get_record(
                call_id, result["request_identity"]["sha256"]
            )
            attempt = record["attempts"][-1]
            raw_response = attempt.get("parsed_response")
            normalized, shape = _normalize(raw_response, self.horizon)
            source_path = self.fresh._path(call_id, result["request_identity"]["sha256"])
            self._save_event(call_id, record, source_path, attempt, normalized,
                             shape, source_only=False)
            result["response"] = normalized
            return result
        return self.fresh.invoke(call_id, prompt, call_json=call_json,
                                 validator=validator)

    def mark_persisted(self, request_sha256):
        if request_sha256 in self._source_projection_ids:
            return True
        return self.fresh.mark_persisted(request_sha256)

    def get_record_for(self, stage: str, call_id: str, request_sha256: str) -> dict:
        try:
            return self.source.get_record(call_id, request_sha256)
        except (FileNotFoundError, OSError):
            return self.fresh.get_record(call_id, request_sha256)


def _read_main_run() -> tuple[dict, list[dict], dict]:
    manifest = _load_json(BASE_DIR / "run_manifest.json")
    jobs = _load_json(BASE_DIR / "input_snapshots.json")
    frozen = _load_json(BASE_DIR / "frozen_protocol.json")
    if (manifest.get("run_id") != BASE_RUN_ID
            or manifest.get("collection_name") != COLLECTION
            or frozen.get("target_collection_frozen_point_count") != 703
            or manifest.get("validation_truth_read") is not False):
        raise ValueError("base run does not match the frozen Test 1 memory/configuration")
    if len(jobs) != 15:
        raise ValueError("base run does not contain 15 frozen cases")
    return manifest, jobs, frozen


def _write_protocol(manifest: dict, jobs: list[dict], frozen: dict) -> None:
    if REPAIR_DIR.exists():
        raise FileExistsError(f"refusing to overwrite H192 repair lane: {REPAIR_DIR}")
    failures = []
    for ordinal in CASE_ORDINALS:
        job = jobs[ordinal - 1]
        state_path = BASE_DIR / "cases" / f"{ordinal:02d}_{job['job_id']}.json"
        state = _load_json(state_path)
        if (state.get("case_id") != job["case_id"]
                or state.get("status") != "technical_failure"
                or state.get("truth_read_audit") is not None
                or job["horizon"] != HORIZON):
            raise ValueError(f"case {ordinal} is not an unrevealed H192 technical failure")
        failures.append({"ordinal": ordinal, "case_id": job["case_id"],
                         "state_path": state_path.relative_to(HIC_ROOT).as_posix(),
                         "state_sha256": _sha_file(state_path),
                         "technical_failure_sha256": _sha_file(
                             BASE_DIR / "technical_failures" / f"{ordinal:02d}_attempt_2.json"
                         )})
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    REPAIR_DIR.mkdir(parents=True, exist_ok=False)
    shape_policy = {
        "overlong": "keep first 192 numeric values; discard only positions beyond horizon",
        "short": "carry the final generated numeric value forward until length 192",
        "exact": "leave unchanged",
        "truth_independent": True,
        "raw_api_response_preserved": True,
        "applies_only_in_supplemental_repair_lane": True,
    }
    protocol = {
        "protocol": REPAIR_PROTOCOL,
        "created_utc": _utc(),
        "base_run_id": BASE_RUN_ID,
        "base_run_manifest_sha256": _sha_file(BASE_DIR / "run_manifest.json"),
        "base_frozen_protocol_sha256": _sha_file(BASE_DIR / "frozen_protocol.json"),
        "source_collection": "agent_test_8",
        "collection": COLLECTION,
        "collection_points": 703,
        "excluded_feedback_points": 15,
        "memory_source_audit_sha256": _sha_file(SOURCE_AUDIT),
        "restore_receipt_sha256": _sha_file(RESTORE_RECEIPT),
        "cases": failures,
        "shape_policy": shape_policy,
        "model": str(hic.MODEL),
        "temperature": float(hic.mdl_cfg.get("temperature", 0.7)),
        "max_tokens": int(hic.mdl_cfg.get("max_output_tokens", 2048)),
        "thinking": "disabled",
        "response_format": "json_object",
        "api_key_configured": True,
        "truth_read_before_locks": False,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "classification": (
            "posthoc technical-shape recovery supplement; these scores are separate "
            "from the primary frozen one-pass score and must not be pooled with it"
        ),
    }
    _atomic_json(REPAIR_DIR / "repair_protocol.json", protocol)
    _atomic_json(REPAIR_DIR / "manifest.json", {
        "protocol": REPAIR_PROTOCOL,
        "repair_protocol_sha256": _sha_file(REPAIR_DIR / "repair_protocol.json"),
        "base_run_id": BASE_RUN_ID,
        "cases": failures,
        "status": "repair_frozen_before_revealing_failed_case_truths",
        "created_utc": _utc(),
    })


def _validate_resume(manifest: dict, jobs: list[dict]) -> None:
    """Resume only the preflight-frozen lane after a dependency failure."""
    protocol_path = REPAIR_DIR / "repair_protocol.json"
    manifest_path = REPAIR_DIR / "manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("existing repair directory lacks its frozen protocol files")
    protocol = _load_json(protocol_path)
    lane = _load_json(manifest_path)
    if (protocol.get("protocol") != REPAIR_PROTOCOL
            or protocol.get("base_run_id") != BASE_RUN_ID
            or protocol.get("collection") != COLLECTION
            or protocol.get("collection_points") != 703
            or protocol.get("base_run_manifest_sha256") != _sha_file(BASE_DIR / "run_manifest.json")
            or lane.get("repair_protocol_sha256") != _sha_file(protocol_path)
            or lane.get("status") != "repair_frozen_before_revealing_failed_case_truths"):
        raise RuntimeError("existing repair lane does not match its original frozen protocol")
    for ordinal in CASE_ORDINALS:
        job = jobs[ordinal - 1]
        path = REPAIR_DIR / "cases" / f"{ordinal:02d}_{job['job_id']}.json"
        if path.exists():
            state = _load_json(path)
            if (state.get("case_id") != job["case_id"]
                    or state.get("status") != "technical_recovery_pending"
                    or state.get("repair_source_state_sha256") != _sha_file(
                        BASE_DIR / "cases" / f"{ordinal:02d}_{job['job_id']}.json")
                    or state.get("truth_read_audit") is not None):
                raise RuntimeError(f"existing repair case {ordinal} has progressed; refusing resume")
    for folder in ("provider_calls", "forecast_locks", "truth_audit", "scores"):
        if any((REPAIR_DIR / folder).glob("*.json")):
            raise RuntimeError(f"repair lane already contains {folder}; refusing to repeat work")


def _metrics(prediction: list, truth: list) -> dict:
    errors = [float(value) - float(actual) for value, actual in zip(prediction, truth)]
    mse = sum(error * error for error in errors) / len(errors)
    return {"mae": sum(abs(error) for error in errors) / len(errors),
            "mse": mse, "rmse": math.sqrt(mse)}


def _repair_case(manifest: dict, job: dict) -> dict:
    ordinal = job["ordinal"]
    base_state_path = BASE_DIR / "cases" / f"{ordinal:02d}_{job['job_id']}.json"
    state = _load_json(base_state_path)
    state.pop("truth_read_audit", None)
    state["status"] = "technical_recovery_pending"
    state["repair_source_state_sha256"] = _sha_file(base_state_path)
    case_state_path = REPAIR_DIR / "cases" / f"{ordinal:02d}_{job['job_id']}.json"
    _atomic_json(case_state_path, state)

    cfg = manifest["configurations"][job["configuration"]]
    cache = ShapeRepairCache(
        manifest=manifest, dataset=job["dataset"], horizon=job["horizon"],
        frequency=cfg["frequency"], unit="unknown",
    )
    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
    try:
        info = store.client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        if (str(info.status).lower() != "green" or info.points_count != 703
                or vectors.size != 1024 or str(vectors.distance).lower() != "cosine"):
            raise RuntimeError("repair collection changed from the frozen 703-point copy")
        core.RUN_DIR = REPAIR_DIR
        core.COLLECTION = COLLECTION
        core.PROTOCOL_ID = "agent_test8_deepseek_one_pass_direct_nearest_v1"
        core.FORECAST_PROTOCOL = "agent_test8_deepseek_paired_forecast_v1"
        core._configure_runtime()
        hic_agent_validation.PROTOCOL = core.FORECAST_PROTOCOL
        result = hic_agent_validation.advance_agent_case(
            job["case"], state, contract=cfg["contract"],
            template=cfg["template"], dataset_description=cfg["description"],
            cache=cache, store=store,
            embed_summary=lambda summary: _embedder().encode(
                [hic.summary_to_embedding_text(summary)], is_query=True)[0],
            persist=lambda: _atomic_json(case_state_path, state),
            retrieval_mode="direct_nearest",
        )
        selected = result.get("retrieval_trace", {}).get("selected", [])
        if len(selected) != 1:
            raise ValueError("repair lane failed to preserve the frozen direct-nearest Top-1 hit")
        points = store.client.retrieve(
            collection_name=COLLECTION, ids=[selected[0]["point_id"]],
            with_payload=True, with_vectors=False,
        )
        if len(points) != 1:
            raise ValueError("Top-1 source point disappeared before repair forecast lock")
        selected[0]["source_metadata"] = (points[0].payload or {}).get("metadata", {})

        state["locked_provider_responses"] = {}
        for stage, call_id in state.get("request_task_ids", {}).items():
            request_sha = state["requests"][stage]
            record = cache.get_record_for(stage, call_id, request_sha)
            attempt = record["attempts"][-1]
            state["locked_provider_responses"][stage] = {
                "call_id": call_id,
                "request_sha256": request_sha,
                "prompt_sha256": record["request_identity"]["prompt_sha256"],
                "request_body_sha256": record["request_body_sha256"],
                "raw_response_sha256": attempt.get("raw_assistant_content_sha256"),
                "raw_response_text": attempt.get("raw_assistant_content"),
                "response_source": "deepseek_api_raw_with_supplemental_shape_repair",
                "model": attempt.get("returned_model", hic.MODEL),
                "usage": attempt.get("usage"),
            }
        state["status"] = "forecast_complete"
        state["shape_repair_protocol"] = REPAIR_PROTOCOL
        _atomic_json(case_state_path, state)
        lock = core._lock_forecast(manifest, job, state)
        events = []
        for call_id in state.get("request_task_ids", {}).values():
            p = _event_path(call_id)
            if p.is_file():
                events.append({"path": p.relative_to(REPAIR_DIR).as_posix(),
                               "sha256": _sha_file(p)})
        lock_path = REPAIR_DIR / "forecast_locks" / f"{ordinal:02d}_{job['job_id']}.json"
        lock.pop("forecast_lock_sha256", None)
        lock["shape_repair_protocol"] = REPAIR_PROTOCOL
        lock["shape_repair_records"] = events
        lock["primary_onepass_case_state_sha256"] = _sha_file(base_state_path)
        lock["validation_truth_read"] = False
        lock_sha = core.frozen_runtime.canonical_hash(lock)
        lock["forecast_lock_sha256"] = lock_sha
        _atomic_json(lock_path, lock)
        state["forecast_lock_sha256"] = lock_sha
        state["status"] = "forecast_locked"
        _atomic_json(case_state_path, state)

        unlock = {"case_id": job["case_id"], "ordinal": ordinal,
                  "forecast_lock_sha256": lock_sha, "truth_read_started_utc": _utc(),
                  "outer_test_targets_read": False}
        _atomic_json(REPAIR_DIR / "truth_audit" / f"{ordinal:02d}_unlock.json", unlock)
        truth, resolution = load_native_hic_validation_truth_for_case(
            job["dataset"], horizon=job["horizon"], case=job["case"],
            target_timestamps=job["target_timestamps"], return_resolution=True,
        )
        a = _metrics(state["baseline"]["prediction"], truth)
        b = _metrics(state["full_hic"]["prediction"], truth)
        direction = "improved" if b["mae"] < a["mae"] else "worsened" if b["mae"] > a["mae"] else "tied"
        truth_sha = core.frozen_runtime.canonical_hash({
            "target_timestamps": job["target_timestamps"], "truth": truth,
        })
        score = {
            "repair_protocol": REPAIR_PROTOCOL,
            "classification": "supplemental technical-shape recovery; not primary run score",
            "run_id": BASE_RUN_ID, "case_id": job["case_id"],
            "ordinal": ordinal, "dataset": job["dataset"], "horizon": job["horizon"],
            "base_input_sha256": job["input_sha256"],
            "forecast_lock_sha256": lock_sha, "truth_sha256": truth_sha,
            "A": a, "B": b, "mae_direction": direction,
            "truth": truth, "target_timestamps": job["target_timestamps"],
            "truth_resolution": resolution, "scored_utc": _utc(),
            "validation_truth_read": True, "outer_test_targets_read": False,
            "qdrant_writes": 0, "shape_repair_records": events,
        }
        _atomic_json(REPAIR_DIR / "truth_audit" / f"{ordinal:02d}.json", {
            "case_id": job["case_id"], "ordinal": ordinal,
            "forecast_lock_sha256": lock_sha,
            "truth_source": "load_native_hic_validation_truth_for_case",
            "truth_resolution": resolution, "truth_sha256": truth_sha,
            "truth_values": truth, "truth_read_utc": score["scored_utc"],
            "validation_truth_read": True, "outer_test_targets_read": False,
        })
        _atomic_json(REPAIR_DIR / "scores" / f"{ordinal:02d}.json", score)
        state["status"] = "scored"
        state["score_metrics"] = {"A": a, "B": b, "mae_direction": direction}
        state["truth_read_audit"] = {
            "truth_sha256": truth_sha, "validation_truth_read": True,
            "outer_test_targets_read": False, "truth_read_utc": score["scored_utc"],
        }
        state.pop("query_embedding", None)
        _atomic_json(case_state_path, state)
        return {"status": "scored", "ordinal": ordinal,
                "case_id": job["case_id"], "A": a, "B": b,
                "mae_direction": direction, "forecast_lock_sha256": lock_sha,
                "shape_repair_records": events}
    finally:
        store.client.close()


def main() -> None:
    manifest, jobs, frozen = _read_main_run()
    if REPAIR_DIR.exists():
        _validate_resume(manifest, jobs)
    else:
        _write_protocol(manifest, jobs, frozen)
    results = []
    for ordinal in CASE_ORDINALS:
        results.append(_repair_case(manifest, jobs[ordinal - 1]))
    summary = {
        "repair_protocol": REPAIR_PROTOCOL,
        "classification": "supplemental technical-shape recovery; separate from primary one-pass evaluation",
        "cases": results,
        "scored": sum(row.get("status") == "scored" for row in results),
        "technical_failures": sum(row.get("status") != "scored" for row in results),
        "truth_read_after_each_repair_lock": True,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "completed_utc": _utc(),
    }
    _atomic_json(REPAIR_DIR / "repair_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
