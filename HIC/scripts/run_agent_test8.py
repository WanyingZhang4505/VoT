"""Run Agent Test 8 as a serial, truth-sealed, one-pass evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from LLM import hic
from LLM.hic_agent_validation import (
    PARAMETERS, PROTOCOL, advance_agent_case, canonical_hash,
)
from LLM.hic_codex_agent import (
    CodexTaskPending, CodexTaskQueue, CodexTaskRequestCache, _atomic_json,
)
from LLM.native_hic_data import load_native_hic_validation_truth_for_case
from LLM.qdrant_store import QdrantHicStore
from scripts.run_agent_test6 import (
    _dataset_description, _embedder, _read, _template,
)


RUN_DIR = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
COLLECTION = "agent_test_8"
PROTOCOL_ID = "agent_test8_frozen_one_pass_direct_nearest_v1"
TERMINAL = {"scored", "technical_failure"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_path(job: dict) -> Path:
    return RUN_DIR / "cases" / f"{job['ordinal']:02d}_{job['job_id']}.json"


def _load() -> tuple[dict, list[dict]]:
    freeze_path = RUN_DIR / "frozen_protocol.json"
    if not freeze_path.is_file():
        raise RuntimeError("Test 8 protocol must be frozen before any runner operation")
    manifest = _read(RUN_DIR / "run_manifest.json")
    jobs = _read(RUN_DIR / "input_snapshots.json")
    frozen = _read(freeze_path)
    files_match = all(
        _sha_file(ROOT / name) == digest
        for name, digest in frozen.get("implementation_files", {}).items()
    )
    if (frozen.get("protocol") != PROTOCOL_ID
            or not files_match
            or frozen.get("run_manifest_sha256") != _sha_file(RUN_DIR / "run_manifest.json")
            or frozen.get("input_snapshots_sha256") != _sha_file(RUN_DIR / "input_snapshots.json")
            or frozen.get("jobs_sha256") != manifest.get("jobs_sha256")
            or frozen.get("implementation_files", {}).get("scripts/run_agent_test8.py")
            != _sha_file(Path(__file__))):
        raise ValueError("Test 8 frozen protocol or runner hash changed")
    if manifest.get("protocol") != PROTOCOL_ID or manifest.get("collection_name") != COLLECTION:
        raise ValueError("Test 8 manifest protocol or collection binding changed")
    if len(jobs) != 15 or canonical_hash(jobs) != manifest.get("jobs_sha256"):
        raise ValueError("frozen Test 8 case snapshot failed integrity validation")
    if manifest.get("validation_truth_read") is not False or manifest.get("outer_test_targets_read") is not False:
        raise ValueError("Test 8 manifest already records truth access")
    for job in jobs:
        state = _read(_state_path(job)) if _state_path(job).is_file() else {}
        if (state.get("frozen_case_input_sha256", job["input_sha256"])
                != job["input_sha256"]):
            raise ValueError(f"frozen input hash mismatch for {job['case_id']}")
        if "truth" in job["case"] or state.get("truth_read_audit"):
            if state.get("truth_read_audit") is None:
                raise ValueError("truth appeared outside a per-case post-lock audit")
    return manifest, jobs


def _read_states(jobs: list[dict]) -> list[dict]:
    return [_read(_state_path(job)) if _state_path(job).is_file()
            else {"status": "pending"} for job in jobs]


def _current(jobs: list[dict], states: list[dict]) -> int | None:
    return next((i for i, state in enumerate(states)
                 if state.get("status") not in TERMINAL), None)


def _cache(queue: CodexTaskQueue, manifest: dict, job: dict) -> CodexTaskRequestCache:
    cfg = manifest["configurations"][job["configuration"]]
    return CodexTaskRequestCache(
        queue, run_id=manifest["run_id"], dataset=job["dataset"],
        horizon=job["horizon"], frequency=cfg["frequency"],
        target="unknown", unit="unknown",
        prompt_version="agent_test8_paired_no_revision_v1",
        memory_version="agent_test8_frozen_memory_v1",
        require_raw_response_provenance=True,
    )


def _progress(manifest: dict, jobs: list[dict], states: list[dict]) -> dict:
    scored = [state for state in states if state.get("status") == "scored"]
    failed = [state for state in states if state.get("status") == "technical_failure"]
    value = {
        "run_id": manifest["run_id"], "total": len(jobs),
        "forecast_locked": sum(bool(s.get("forecast_lock_sha256")) for s in states),
        "scored": len(scored), "technical_failures": len(failed),
        "effective_retries": sum(int(s.get("effective_retry_count", 0)) for s in states),
        "qdrant_writes": 0,
        "validation_truth_read": any(s.get("truth_read_audit") for s in states),
        "outer_test_targets_read": False,
        "next_case_id": (jobs[_current(jobs, states)]["case_id"]
                         if _current(jobs, states) is not None else None),
        "status": "complete" if len(scored) + len(failed) == len(jobs) else "running",
        "updated_utc": _utc(),
    }
    _atomic_json(RUN_DIR / "progress.json", value)
    terminal_count = len(scored) + len(failed)
    if terminal_count and terminal_count % 3 == 0:
        _write_checkpoint(manifest, jobs, states, terminal_count)
    return value


def _write_checkpoint(manifest: dict, jobs: list[dict], states: list[dict], count: int) -> None:
    checkpoint = {
        "run_id": manifest["run_id"], "terminal_cases": count,
        "completed_case_ids": [job["case_id"] for job, state in zip(jobs, states)
                               if state.get("status") in TERMINAL],
        "scored": sum(s.get("status") == "scored" for s in states),
        "technical_failures": sum(s.get("status") == "technical_failure" for s in states),
        "qdrant_writes": 0, "validation_truth_read": any(
            s.get("truth_read_audit") for s in states),
        "outer_test_targets_read": False, "saved_utc": _utc(),
    }
    _atomic_json(RUN_DIR / "checkpoints" / f"checkpoint_{count:02d}.json", checkpoint)
    report = _metrics_report(manifest, jobs, states, partial=True)
    _atomic_json(RUN_DIR / "reports" / f"checkpoint_{count:02d}.json", report)


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
    time_eligible = None
    if isinstance(source_target_end, str) and isinstance(query_history_start, str):
        time_eligible = source_target_end < query_history_start
    same_origin = (isinstance(source_history_start, str)
                   and source_history_start == query_history_start)
    hit_audit = {
        **hit, "correction_sha256": _sha_text(correction),
        "source_case_id": hit.get("case_id"),
        "source_history_start": source_history_start,
        "source_target_end": source_target_end,
        "query_history_start": query_history_start,
        "same_source_dataset": hit.get("source_dataset") == job["dataset"],
        "same_source_case": hit.get("case_id") == job["case_id"],
        "same_history_origin": same_origin,
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
        "frozen_case_input_sha256": job["input_sha256"],
        "semantic_input_sha256": state["input_sha256"],
        "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
        "forecast_protocol_version": PROTOCOL,
        "paired_parameters": PARAMETERS,
        "query_embedding_sha256": canonical_hash(state["query_embedding"]),
        "baseline": state["baseline"], "full_hic": state["full_hic"],
        "retrieval_trace": trace,
        "request_sha256": state.get("requests", {}),
        "worker_responses": state.get("locked_worker_responses", {}),
        "locked_utc": _utc(), "validation_truth_read": False,
        "outer_test_targets_read": False,
    }
    lock["forecast_lock_sha256"] = canonical_hash(lock)
    lock_path = RUN_DIR / "forecast_locks" / f"{job['ordinal']:02d}_{job['job_id']}.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    state["retrieval_trace"] = trace
    state["forecast_lock_sha256"] = lock["forecast_lock_sha256"]
    state["status"] = "forecast_locked"
    state["forecast_locked_utc"] = lock["locked_utc"]
    _atomic_json(_state_path(job), state)
    return lock


def advance_one() -> dict:
    manifest, jobs = _load()
    states = _read_states(jobs)
    index = _current(jobs, states)
    if index is None:
        return {"status": "all_cases_terminal", "progress": _progress(manifest, jobs, states)}
    if any(state.get("status") not in TERMINAL for state in states[:index]):
        raise RuntimeError("serial case gate blocked by an earlier unscored case")
    job, state = jobs[index], states[index]
    state = dict(state)
    state.setdefault("frozen_case_input_sha256", job["input_sha256"])
    state.setdefault("ordinal", job["ordinal"])
    state.setdefault("case_id", job["case_id"])
    cfg = manifest["configurations"][job["configuration"]]
    queue = CodexTaskQueue(RUN_DIR / "tasks", max_repairs=1)
    cache = _cache(queue, manifest, job)
    client = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=COLLECTION)
    source_metadata = None
    try:
        if not client.client.collection_exists(COLLECTION):
            raise RuntimeError("Test 8 Qdrant collection is missing")
        current_count = client.count()
        if current_count != 703:
            raise RuntimeError(f"Test 8 frozen retrieval bank count changed: {current_count}")
        embedder = _embedder()
        result = advance_agent_case(
            job["case"], state, contract=cfg["contract"],
            template=cfg["template"], dataset_description=cfg["description"],
            cache=cache, store=client,
            embed_summary=lambda summary: embedder.encode(
                [hic.summary_to_embedding_text(summary)], is_query=True)[0],
            persist=lambda: _atomic_json(_state_path(job), state),
            retrieval_mode="direct_nearest",
        )
        selected = result.get("retrieval_trace", {}).get("selected", [])
        if selected:
            source_points = client.client.retrieve(
                collection_name=COLLECTION, ids=[selected[0]["point_id"]],
                with_payload=True, with_vectors=False,
            )
            if len(source_points) != 1:
                raise ValueError("direct-nearest source point disappeared before audit lock")
            source_metadata = (source_points[0].payload or {}).get("metadata", {})
    except CodexTaskPending as pending:
        state["pending_task_id"] = pending.task_id
        state["status"] = "awaiting_worker"
        _atomic_json(_state_path(job), state)
        task = queue.get(pending.task_id)
        return {"status": "awaiting_worker", "ordinal": job["ordinal"],
                "case_id": job["case_id"], "task_id": pending.task_id,
                "task_id_sha256": task["task_id_sha256"],
                "request_sha256": task.get("request_sha256"),
                "prompt_sha256": task["prompt_sha256"],
                "task_path": str(queue._path(pending.task_id).resolve()),
                "task_type": task["task_type"], "stage": task["metadata"].get("stage"),
                "status": task["status"]}
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "recorded_utc": _utc(), "validation_truth_read": False}
        state.setdefault("technical_attempts", []).append(failure)
        attempts = len(state["technical_attempts"])
        state["effective_retry_count"] = max(0, attempts - 1)
        state["status"] = "recovery_pending" if attempts == 1 else "technical_failure"
        _atomic_json(_state_path(job), state)
        _atomic_json(RUN_DIR / "technical_failures" / f"{job['ordinal']:02d}_attempt_{attempts}.json",
                     failure)
        states[index] = state
        return {"status": state["status"], "ordinal": job["ordinal"],
                "case_id": job["case_id"], "failure": failure,
                "progress": _progress(manifest, jobs, states)}
    finally:
        client.client.close()

    # No code path below reads validation truth or mutates the collection.
    selected = result.get("retrieval_trace", {}).get("selected", [])
    if len(selected) != 1:
        raise ValueError("Test 8 direct-nearest retrieval did not produce exactly one hit")
    selected[0]["source_metadata"] = source_metadata or {}
    state["status"] = "forecast_complete"
    state["frozen_case_input_sha256"] = job["input_sha256"]
    state["locked_worker_responses"] = {}
    for stage, task_id in state.get("request_task_ids", {}).items():
        task = queue.get(task_id)
        state["locked_worker_responses"][stage] = {
            "task_id": task_id, "prompt_version": task["prompt_version"],
            "prompt_sha256": task["prompt_sha256"],
            "request_sha256": task.get("request_sha256"),
            "raw_response_text": task.get("raw_response_text"),
            "raw_response_sha256": task.get("raw_response_sha256"),
            "response_source": task.get("response_source"),
            "worker_context_id": task.get("worker_context_id"),
        }
    lock = _lock_forecast(manifest, job, state)
    states[index] = state
    return {"status": "forecast_locked", "ordinal": job["ordinal"],
            "case_id": job["case_id"], "forecast_lock_sha256": lock["forecast_lock_sha256"],
            "retrieved_point_id": selected[0].get("point_id"),
            "retrieved_source_case_id": selected[0].get("case_id"),
            "progress": _progress(manifest, jobs, states)}


def dispatch_task(task_id: str, worker_context_id: str) -> dict:
    _load()
    queue = CodexTaskQueue(RUN_DIR / "tasks", max_repairs=1)
    task = queue.mark_dispatched(
        task_id, worker_context_id=worker_context_id,
        context_isolation_mode="fresh_subagent_no_fork",
    )
    return {"task_id": task_id, "task_identity_sha256": task["task_identity_sha256"],
            "request_sha256": task.get("request_sha256"),
            "worker_context_id": task["worker_context_id"],
            "context_isolation_mode": task["context_isolation_mode"],
            "dispatched_utc": task["dispatch_history"][-1]["dispatched_utc"]}


def record_task(task_id: str, worker_context_id: str, response_path: Path,
                response_source: str) -> dict:
    _load()
    queue = CodexTaskQueue(RUN_DIR / "tasks", max_repairs=1)
    raw = response_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("worker response must be one JSON object")
    task = queue.record_response(
        task_id, parsed, worker_context_id=worker_context_id,
        raw_response_text=raw, response_source=response_source,
    )
    return {"task_id": task_id, "status": task["status"],
            "response_sha256": task.get("response_sha256"),
            "raw_response_sha256": task.get("raw_response_sha256"),
            "validation_error_type": task.get("validation_error_type")}


def score_one() -> dict:
    manifest, jobs = _load()
    states = _read_states(jobs)
    index = next((i for i, s in enumerate(states)
                  if s.get("status") == "forecast_locked"), None)
    if index is None:
        raise RuntimeError("no forecast-locked case is awaiting truth reveal")
    if any(state.get("status") not in TERMINAL for state in states[:index]):
        raise RuntimeError("serial scoring gate blocked by an earlier case")
    job, state = jobs[index], states[index]
    lock_path = RUN_DIR / "forecast_locks" / f"{job['ordinal']:02d}_{job['job_id']}.json"
    lock = _read(lock_path)
    lock_sha = lock.pop("forecast_lock_sha256")
    if canonical_hash(lock) != lock_sha or state.get("forecast_lock_sha256") != lock_sha:
        raise ValueError("forecast lock hash does not match saved A/B outputs")
    if lock.get("validation_truth_read") is not False:
        raise ValueError("forecast lock was not truth sealed")
    truth, truth_resolution = load_native_hic_validation_truth_for_case(
        job["dataset"], horizon=job["horizon"], case=job["case"],
        target_timestamps=job["target_timestamps"], return_resolution=True,
    )
    prediction_a = state["baseline"]["prediction"]
    prediction_b = state["full_hic"]["prediction"]
    if not (len(truth) == len(prediction_a) == len(prediction_b) == job["horizon"]):
        raise ValueError("truth and locked A/B forecasts have different horizon lengths")
    errors_a = [float(a) - float(y) for a, y in zip(prediction_a, truth)]
    errors_b = [float(b) - float(y) for b, y in zip(prediction_b, truth)]

    def metrics(errors: list[float]) -> dict:
        mse = sum(error * error for error in errors) / len(errors)
        return {"mae": sum(abs(error) for error in errors) / len(errors),
                "mse": mse, "rmse": math.sqrt(mse)}

    a, b = metrics(errors_a), metrics(errors_b)
    direction = "improved" if b["mae"] < a["mae"] else "worsened" if b["mae"] > a["mae"] else "tied"
    audit = {
        "case_id": job["case_id"], "ordinal": job["ordinal"],
        "forecast_lock_sha256": lock_sha,
        "truth_source": "load_native_hic_validation_truth_for_case",
        "truth_resolution": truth_resolution, "truth_sha256": canonical_hash({
            "target_timestamps": job["target_timestamps"], "truth": truth}),
        "truth_values": truth, "truth_read_utc": _utc(),
        "validation_truth_read": True, "outer_test_targets_read": False,
    }
    score = {
        "run_id": manifest["run_id"], "case_id": job["case_id"],
        "ordinal": job["ordinal"], "dataset": job["dataset"],
        "horizon": job["horizon"], "input_sha256": job["input_sha256"],
        "forecast_lock_sha256": lock_sha, "truth_sha256": audit["truth_sha256"],
        "A": a, "B": b, "mae_direction": direction,
        "truth": truth, "target_timestamps": job["target_timestamps"],
        "truth_resolution": truth_resolution, "scored_utc": audit["truth_read_utc"],
        "validation_truth_read": True, "outer_test_targets_read": False,
    }
    _atomic_json(RUN_DIR / "truth_audit" / f"{job['ordinal']:02d}.json", audit)
    _atomic_json(RUN_DIR / "scores" / f"{job['ordinal']:02d}.json", score)
    state["status"] = "scored"
    state["score_sha256"] = canonical_hash(score)
    state["score_metrics"] = {"A": a, "B": b, "mae_direction": direction}
    state["truth_read_audit"] = audit
    state.pop("query_embedding", None)
    _atomic_json(_state_path(job), state)
    states[index] = state
    progress = _progress(manifest, jobs, states)
    return {"status": "scored", "ordinal": job["ordinal"],
            "case_id": job["case_id"], "A": a, "B": b,
            "mae_direction": direction, "truth_sha256": audit["truth_sha256"],
            "progress": progress}


def _metrics_report(manifest: dict, jobs: list[dict], states: list[dict],
                    *, partial: bool) -> dict:
    scored_rows = []
    for job, state in zip(jobs, states):
        if state.get("status") != "scored":
            continue
        scored_rows.append({"job": job, "state": state,
                            "score": _read(RUN_DIR / "scores" / f"{job['ordinal']:02d}.json")})

    def aggregate(rows: list[dict]) -> dict:
        result = {arm: {} for arm in ("A", "B")}
        for arm in ("A", "B"):
            for metric in ("mae", "mse", "rmse"):
                values = [row["score"][arm][metric] for row in rows]
                result[arm][metric] = sum(values) / len(values) if values else None
        directions = Counter(row["score"]["mae_direction"] for row in rows)
        result["mae_direction_counts"] = {
            key: directions.get(key, 0) for key in ("improved", "tied", "worsened")}
        result["n"] = len(rows)
        result["mae_direction_proportions"] = {
            key: directions.get(key, 0) / len(rows) if rows else None
            for key in ("improved", "tied", "worsened")}
        return result

    families = defaultdict(list)
    configs = defaultdict(list)
    for row in scored_rows:
        job = row["job"]
        families[job["dataset"]].append(row)
        configs[f"{job['dataset']}/H{job['horizon']}"].append(row)
    return {
        "run_id": manifest["run_id"],
        "classification": "基于 Test 7 记忆库的一次性冻结评估",
        "independent_external_benchmark_claim": False,
        "partial": partial, "status": "partial" if partial else "complete",
        "case_count": len(jobs), "scored": len(scored_rows),
        "technical_failures": sum(s.get("status") == "technical_failure" for s in states),
        "effective_retry_count": sum(int(s.get("effective_retry_count", 0)) for s in states),
        "overall": aggregate(scored_rows),
        "families": {key: aggregate(value) for key, value in sorted(families.items())},
        "configurations": {key: aggregate(value) for key, value in sorted(configs.items())},
        "case_results": [{"ordinal": row["job"]["ordinal"],
                           "case_id": row["job"]["case_id"],
                           "dataset": row["job"]["dataset"],
                           "horizon": row["job"]["horizon"],
                           "A": row["score"]["A"], "B": row["score"]["B"],
                           "mae_direction": row["score"]["mae_direction"],
                           "forecast_lock_sha256": row["score"]["forecast_lock_sha256"],
                           "retrieval": row["state"].get("retrieval_trace", {}).get("selected", [])}
                          for row in scored_rows],
        "qdrant_writes": 0, "validation_truth_read": bool(scored_rows),
        "outer_test_targets_read": False,
        "created_utc": _utc(),
    }


def status() -> dict:
    manifest, jobs = _load()
    return _progress(manifest, jobs, _read_states(jobs))


def finalize() -> dict:
    manifest, jobs = _load()
    states = _read_states(jobs)
    if any(state.get("status") not in TERMINAL for state in states):
        raise RuntimeError("cannot finalize before all cases are scored or documented technical failures")
    report = _metrics_report(manifest, jobs, states, partial=False)
    report["status"] = "complete_15_cases_terminal"
    _atomic_json(RUN_DIR / "reports" / "final_report.json", report)
    (RUN_DIR / "reports" / "final_report.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Agent Test 8 — 一次性冻结评估",
        "",
        f"状态：{report['status']}；评分 {report['scored']}/{report['case_count']}；技术失败 {report['technical_failures']}；有效重试 {report['effective_retry_count']}。",
        "",
        "本结果是基于 Test 7 开发记忆库的一次性冻结评估，不是未接触开发经验的外部 benchmark。Health/Security/new_metr 的 target/unit 保持未知口径，按原始数值计算。",
        "",
        "## 分组汇总",
        "",
        "| 分组 | n | A MAE / MSE / RMSE | B MAE / MSE / RMSE | 改善 / 持平 / 退化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in [("总体", report["overall"]), *report["families"].items(),
                         *report["configurations"].items()]:
        a, b = values["A"], values["B"]
        counts = values["mae_direction_counts"]
        fmt = lambda m: " / ".join("—" if m[k] is None else f"{m[k]:.8g}"
                                   for k in ("mae", "mse", "rmse"))
        lines.append(f"| {name} | {values['n']} | {fmt(a)} | {fmt(b)} | {counts['improved']} / {counts['tied']} / {counts['worsened']} |")
    lines += ["", "## 逐例结果", "", "| # | Case | A MAE/MSE/RMSE | B MAE/MSE/RMSE | MAE方向 | Top-1 point / source case | 时间资格 |",
              "|---:|---|---:|---:|---|---|---|"]
    for row in report["case_results"]:
        hit = row["retrieval"][0] if row["retrieval"] else {}
        fmt = lambda m: "/".join(f"{m[k]:.7g}" for k in ("mae", "mse", "rmse"))
        trace = row["retrieval"]
        lines.append(f"| {row['ordinal']} | `{row['case_id']}` | {fmt(row['A'])} | {fmt(row['B'])} | {row['mae_direction']} | `{hit.get('point_id')}` / `{hit.get('case_id')}` | {hit.get('time_eligible_strictly_prior', 'unresolved')} |")
    lines += ["", "Truth 解封逐例记录见 `truth_audit/`；outer-test targets 未读取。评估期间 Qdrant 写入数为 0。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("advance-one")
    dispatch = sub.add_parser("dispatch-task")
    dispatch.add_argument("--task-id", required=True)
    dispatch.add_argument("--worker-context-id", required=True)
    record = sub.add_parser("record-task")
    record.add_argument("--task-id", required=True)
    record.add_argument("--worker-context-id", required=True)
    record.add_argument("--response-file", type=Path, required=True)
    record.add_argument("--response-source", required=True)
    sub.add_parser("score-one")
    sub.add_parser("status")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "advance-one":
        result = advance_one()
    elif args.command == "dispatch-task":
        result = dispatch_task(args.task_id, args.worker_context_id)
    elif args.command == "record-task":
        result = record_task(args.task_id, args.worker_context_id,
                             args.response_file, args.response_source)
    elif args.command == "score-one":
        result = score_one()
    elif args.command == "status":
        result = status()
    else:
        result = finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
