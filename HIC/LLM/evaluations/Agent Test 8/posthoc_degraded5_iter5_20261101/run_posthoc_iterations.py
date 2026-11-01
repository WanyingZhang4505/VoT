"""Isolated, truth-aware post-hoc iteration for five Agent Test 8 cases.

This runner never writes to Qdrant. It starts from the completed Test 8
forecast locks and validation truths, permits per-case feedback, and keeps all
iterations in a separate experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SOURCE_RUN = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"
RUN_DIR = Path(__file__).resolve().parent
RUN_ID = "agent_test8_posthoc_degraded5_iter5_20261101"
MAX_ITERATIONS = 5
ORDINALS = (4, 5, 6, 9, 11)
EPSILON = 1e-12


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(data)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _case_dir(ordinal: int) -> Path:
    return RUN_DIR / "cases" / f"{ordinal:02d}"


def _case_state_path(ordinal: int) -> Path:
    return _case_dir(ordinal) / "case_state.json"


def _load_case(ordinal: int) -> dict:
    return _read(_case_state_path(ordinal))


def _metrics(prediction: list[float], truth: list[float]) -> dict:
    if len(prediction) != len(truth) or not prediction:
        raise ValueError("prediction and truth must have the same non-zero length")
    errors = [p - y for p, y in zip(prediction, truth)]
    mae = sum(abs(x) for x in errors) / len(errors)
    mse = sum(x * x for x in errors) / len(errors)
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse)}


def _direction(mae: float, baseline_mae: float) -> str:
    if mae < baseline_mae - EPSILON:
        return "improved"
    if mae > baseline_mae + EPSILON:
        return "worsened"
    return "tied"


def initialize() -> dict:
    if (RUN_DIR / "run_manifest.json").exists():
        raise FileExistsError("post-hoc experiment is already initialized")
    report = _read(SOURCE_RUN / "reports/final_report.json")
    frozen = _read(SOURCE_RUN / "frozen_protocol.json")
    run_manifest = {
        "run_id": RUN_ID,
        "source_run_id": report["run_id"],
        "classification": "post_hoc_truth_aware_optimization_not_benchmark",
        "created_utc": _utc(),
        "case_ordinals": list(ORDINALS),
        "max_revision_iterations_per_case": MAX_ITERATIONS,
        "initial_B_is_original_test8_locked_prediction": True,
        "comparison_baseline": "original Test 8 A, fixed per case",
        "continue_condition": "continue only while latest B MAE is worse than original A MAE",
        "early_stop_condition": "stop at first improved or tied B relative to original A",
        "truth_feedback_allowed": True,
        "outer_test_targets_read": False,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
        "source_frozen_protocol_sha256": _sha_json(frozen),
        "cases": [],
    }
    for ordinal in ORDINALS:
        lock_path = next((SOURCE_RUN / "forecast_locks").glob(f"{ordinal:02d}_*.json"))
        lock = _read(lock_path)
        truth_record = _read(SOURCE_RUN / "truth_audit" / f"{ordinal:02d}.json")
        base_b_task_path = None
        base_b_task = None
        for task_path in (SOURCE_RUN / "tasks").glob("*.json"):
            task = _read(task_path)
            if (task.get("metadata", {}).get("case_id") == lock["case_id"]
                    and task.get("metadata", {}).get("stage") == "full_hic_forecast_paired_v2"):
                base_b_task_path, base_b_task = task_path, task
                break
        if base_b_task is None:
            raise FileNotFoundError(f"original B task missing for ordinal {ordinal}")
        responses = lock["worker_responses"]
        a_raw = responses["baseline_no_retrieval_paired_v1"]["raw_response_text"]
        b_raw = responses["full_hic_forecast_paired_v2"]["raw_response_text"]
        a = json.loads(a_raw)["prediction"]
        b = json.loads(b_raw)["prediction"]
        truth = truth_record["truth_values"]
        if len(a) != len(truth) or len(b) != len(truth):
            raise ValueError(f"original forecast/truth length mismatch for ordinal {ordinal}")
        original_row = next(r for r in report["case_results"] if r["ordinal"] == ordinal)
        if original_row["mae_direction"] != "worsened":
            raise ValueError(f"ordinal {ordinal} is not an originally worsened case")
        state = {
            "ordinal": ordinal,
            "case_id": lock["case_id"],
            "dataset": lock["dataset"],
            "horizon": lock["horizon"],
            "initial_input_prompt": base_b_task["prompt"],
            "initial_B_task_sha256": _sha_bytes(base_b_task_path.read_bytes()),
            "initial_B_request_sha256": base_b_task["request_sha256"],
            "initial_retrieval": lock["retrieval_trace"]["selected"][0],
            "initial_correction": lock["retrieval_trace"]["selected"][0]["correction"],
            "original_A": a,
            "original_B0": b,
            "truth": truth,
            "truth_sha256": truth_record["truth_sha256"],
            "original_A_metrics": original_row["A"],
            "original_B0_metrics": original_row["B"],
            "original_B0_direction_vs_A": original_row["mae_direction"],
            "forecast_lock_sha256": lock["forecast_lock_sha256"],
            "outer_test_targets_read": False,
            "iterations": [],
            "status": "ready",
        }
        _write(_case_state_path(ordinal), state)
        run_manifest["cases"].append({
            "ordinal": ordinal,
            "case_id": lock["case_id"],
            "dataset": lock["dataset"],
            "horizon": lock["horizon"],
            "initial_A_mae": original_row["A"]["mae"],
            "initial_B_mae": original_row["B"]["mae"],
            "truth_sha256": truth_record["truth_sha256"],
            "initial_top1_point_id": lock["retrieval_trace"]["selected"][0]["point_id"],
            "initial_top1_source_case_id": lock["retrieval_trace"]["selected"][0]["case_id"],
            "initial_top1_time_eligible": lock["retrieval_trace"]["selected"][0]["time_eligible_strictly_prior"],
        })
    _write(RUN_DIR / "run_manifest.json", run_manifest)
    return {"status": "initialized", "case_count": len(ORDINALS), "run_id": RUN_ID}


def prepare_next(ordinal: int) -> dict:
    state = _load_case(ordinal)
    completed = state["iterations"]
    if completed:
        last = completed[-1]
        if last["direction_vs_original_A"] != "worsened":
            state["status"] = "stopped_at_first_non_worse"
            _write(_case_state_path(ordinal), state)
            return {"status": state["status"], "ordinal": ordinal,
                    "case_id": state["case_id"], "iterations": len(completed)}
    if len(completed) >= MAX_ITERATIONS:
        state["status"] = "max_iterations_reached"
        _write(_case_state_path(ordinal), state)
        return {"status": state["status"], "ordinal": ordinal,
                "case_id": state["case_id"], "iterations": len(completed)}
    iteration = len(completed) + 1
    previous_prediction = completed[-1]["prediction"] if completed else state["original_B0"]
    previous_correction = completed[-1]["updated_correction"] if completed else state["initial_correction"]
    previous_metrics = completed[-1]["metrics"] if completed else state["original_B0_metrics"]
    history = [{
        "iteration": row["iteration"],
        "mae": row["metrics"]["mae"],
        "mse": row["metrics"]["mse"],
        "rmse": row["metrics"]["rmse"],
        "direction_vs_original_A": row["direction_vs_original_A"],
        "updated_correction": row["updated_correction"],
        "prediction_sha256": row["prediction_sha256"],
    } for row in completed]
    task_id = f"{RUN_ID}:{ordinal:02d}:{state['case_id']}:iter{iteration}"
    schema_len = len(state["truth"])
    feedback = {
        "case_id": state["case_id"],
        "dataset": state["dataset"],
        "horizon": state["horizon"],
        "previous_prediction": previous_prediction,
        "previous_metrics_against_validation_truth": previous_metrics,
        "original_A_baseline_prediction": state["original_A"],
        "original_A_metrics": state["original_A_metrics"],
        "original_B0_metrics": state["original_B0_metrics"],
        "validation_truth_for_this_case": state["truth"],
        "truth_sha256": state["truth_sha256"],
        "previous_correction": previous_correction,
        "completed_revision_history": history,
        "outer_test_targets_read": False,
    }
    prompt = (
        state["initial_input_prompt"]
        + "\n\nPOST-HOC ITERATION OVERRIDE (this is not part of the frozen Test 8 benchmark):\n"
        + "This is a truth-aware development exercise explicitly requested by the user. "
        + "The validation truth below is already unsealed and may be used to diagnose "
        + "and revise this same case only. Do not use information from other cases or "
        + "outer-test data. Do not claim this is a blind evaluation. Keep the source "
        + "correction as context, but revise it only from the feedback supplied here. "
        + "Return one new forecast of exactly the specified length and a concise updated "
        + "correction that explains the empirical change; do not return scores.\n"
        + f"This is post-hoc revision {iteration} of at most {MAX_ITERATIONS}. "
        + f"The required prediction length is {schema_len}.\n"
        + "Return exactly this JSON shape: {\"prediction\":[finite numbers],"
        + "\"updated_correction\":\"...\",\"reasoning\":\"...\"}.\n"
        + "POST-HOC FEEDBACK JSON:\n"
        + json.dumps(feedback, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    )
    task = {
        "task_id": task_id,
        "ordinal": ordinal,
        "case_id": state["case_id"],
        "iteration": iteration,
        "horizon": state["horizon"],
        "request_sha256": _sha_bytes(prompt.encode("utf-8")),
        "prompt_version": "agent_test8_posthoc_truth_aware_revision_v1",
        "prompt": prompt,
        "output_schema": {"prediction_length": schema_len,
                          "required": ["prediction", "updated_correction", "reasoning"]},
        "response_file": str(RUN_DIR / "tasks/responses" / f"{ordinal:02d}_iter{iteration}.json"),
        "created_utc": _utc(),
    }
    task_path = RUN_DIR / "tasks" / f"{ordinal:02d}_iter{iteration}.json"
    _write(task_path, task)
    state["status"] = f"iter{iteration}_pending"
    state["current_task_id"] = task_id
    state["current_task_path"] = str(task_path)
    _write(_case_state_path(ordinal), state)
    return {"status": "pending", "ordinal": ordinal, "case_id": state["case_id"],
            "iteration": iteration, "task_id": task_id, "request_sha256": task["request_sha256"],
            "task_path": str(task_path), "response_file": task["response_file"]}


def record_score(ordinal: int, iteration: int, response_file: Path,
                 worker_context_id: str) -> dict:
    state = _load_case(ordinal)
    if state.get("status") != f"iter{iteration}_pending":
        raise ValueError("case is not waiting for this iteration")
    task = _read(RUN_DIR / "tasks" / f"{ordinal:02d}_iter{iteration}.json")
    response_bytes = response_file.read_bytes()
    response = json.loads(response_bytes.decode("utf-8"))
    if set(response) != {"prediction", "updated_correction", "reasoning"}:
        raise ValueError("response must contain exactly prediction, updated_correction, reasoning")
    prediction = response["prediction"]
    if (not isinstance(prediction, list) or len(prediction) != len(state["truth"])
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(float(v)) for v in prediction)):
        raise ValueError("prediction must contain exactly the required number of finite numeric values")
    if not isinstance(response["updated_correction"], str) or not response["updated_correction"].strip():
        raise ValueError("updated_correction must be non-empty text")
    if not isinstance(response["reasoning"], str) or not response["reasoning"].strip():
        raise ValueError("reasoning must be non-empty text")
    prediction = [float(x) for x in prediction]
    metrics = _metrics(prediction, state["truth"])
    previous = state["iterations"][-1]["prediction"] if state["iterations"] else state["original_B0"]
    previous_metrics = _metrics(previous, state["truth"])
    row = {
        "iteration": iteration,
        "task_id": task["task_id"],
        "request_sha256": task["request_sha256"],
        "response_sha256": _sha_bytes(response_bytes),
        "worker_context_id": worker_context_id,
        "response_source_path": str(response_file.resolve()),
        "prediction_sha256": _sha_json(prediction),
        "prediction": prediction,
        "updated_correction": response["updated_correction"],
        "correction_sha256": _sha_bytes(response["updated_correction"].encode("utf-8")),
        "reasoning": response["reasoning"],
        "metrics": metrics,
        "direction_vs_original_A": _direction(metrics["mae"], state["original_A_metrics"]["mae"]),
        "direction_vs_previous_B": _direction(metrics["mae"], previous_metrics["mae"]),
        "previous_metrics": previous_metrics,
        "truth_sha256": state["truth_sha256"],
        "scored_utc": _utc(),
        "outer_test_targets_read": False,
    }
    saved_response = RUN_DIR / "tasks" / "responses" / f"{ordinal:02d}_iter{iteration}.json"
    saved_response.parent.mkdir(parents=True, exist_ok=True)
    if response_file.resolve() != saved_response.resolve():
        saved_response.write_bytes(response_bytes)
        row["response_source_path"] = str(saved_response.resolve())
    _write(RUN_DIR / "scores" / f"{ordinal:02d}_iter{iteration}.json", row)
    state["iterations"].append(row)
    if row["direction_vs_original_A"] != "worsened":
        state["status"] = "stopped_at_first_non_worse"
    elif iteration >= MAX_ITERATIONS:
        state["status"] = "max_iterations_reached"
    else:
        state["status"] = "ready_for_next_iteration"
    state.pop("current_task_id", None)
    state.pop("current_task_path", None)
    _write(_case_state_path(ordinal), state)
    return {"status": state["status"], "ordinal": ordinal, "case_id": state["case_id"],
            "iteration": iteration, "A_mae": state["original_A_metrics"]["mae"],
            "B_mae": metrics["mae"], "direction_vs_A": row["direction_vs_original_A"],
            "direction_vs_previous_B": row["direction_vs_previous_B"],
            "prediction_sha256": row["prediction_sha256"]}


def status() -> dict:
    rows = []
    for ordinal in ORDINALS:
        state = _load_case(ordinal)
        rows.append({"ordinal": ordinal, "case_id": state["case_id"],
                     "status": state["status"], "iterations": len(state["iterations"]),
                     "latest_direction": state["iterations"][-1]["direction_vs_original_A"]
                     if state["iterations"] else state["original_B0_direction_vs_A"]})
    return {"run_id": RUN_ID, "cases": rows,
            "iterations_total": sum(row["iterations"] for row in rows),
            "outer_test_targets_read": False, "qdrant_writes": 0}


def finalize() -> dict:
    rows = []
    for ordinal in ORDINALS:
        state = _load_case(ordinal)
        if state["status"] in {"ready", "ready_for_next_iteration"} or state["status"].endswith("_pending"):
            raise RuntimeError(f"case {ordinal} still has iterations to run")
        last = state["iterations"][-1] if state["iterations"] else None
        rows.append({
            "ordinal": ordinal,
            "case_id": state["case_id"],
            "horizon": state["horizon"],
            "original_A": state["original_A_metrics"],
            "original_B0": state["original_B0_metrics"],
            "original_direction": state["original_B0_direction_vs_A"],
            "revision_count": len(state["iterations"]),
            "iterations": [{k: row[k] for k in (
                "iteration", "metrics", "direction_vs_original_A",
                "direction_vs_previous_B", "correction_sha256", "prediction_sha256",
                "response_sha256", "request_sha256", "worker_context_id",
            )} for row in state["iterations"]],
            "final_direction": last["direction_vs_original_A"] if last else state["original_B0_direction_vs_A"],
            "stopped_reason": state["status"],
            "initial_top1_point_id": state["initial_retrieval"]["point_id"],
            "initial_top1_source_case_id": state["initial_retrieval"]["case_id"],
            "initial_top1_time_eligible": state["initial_retrieval"]["time_eligible_strictly_prior"],
        })
    counts = {key: 0 for key in ("improved", "tied", "worsened")}
    for row in rows:
        counts[row["final_direction"]] += 1
    result = {
        "run_id": RUN_ID,
        "source_run_id": "agent_test8_remaining3families_15_20261101",
        "classification": "post-hoc validation-truth-aware tuning; not a benchmark result",
        "status": "complete",
        "case_count": len(rows),
        "iterations_total": sum(row["revision_count"] for row in rows),
        "max_iterations_per_case": MAX_ITERATIONS,
        "final_direction_counts_vs_original_A": counts,
        "case_results": rows,
        "validation_truth_feedback_used": True,
        "outer_test_targets_read": False,
        "qdrant_writes": 0,
        "created_utc": _utc(),
    }
    _write(RUN_DIR / "reports" / "final_report.json", result)
    lines = [
        "# Agent Test 8 — 五个 MAE 退化例的事后迭代",
        "",
        "本报告是已解封 validation truth 后的定向调优，不是盲测或独立 benchmark；原始 Test 8 15 例报告不变。每例以原始 A 固定比较，最多五轮；首次达到 MAE 改善或持平即停止。",
        "",
        f"最终方向（相对原始 A）：改善 {counts['improved']}，持平 {counts['tied']}，退化 {counts['worsened']}；总修订轮次 {result['iterations_total']}。",
        "",
        "| Case | Horizon | 原始 A MAE | 原始 B0 MAE | 修订轮数 | 最终方向 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['case_id']}` | {row['horizon']} | {row['original_A']['mae']:.8g} | {row['original_B0']['mae']:.8g} | {row['revision_count']} | {row['final_direction']} |")
    lines += ["", "每轮 MAE/MSE/RMSE、correction/forecast 哈希和 worker 记录见同目录 `final_report.json`、`scores/` 与 `cases/`。原始 Qdrant collections 均未写入；outer-test 未读取。", ""]
    (RUN_DIR / "reports" / "final_report.md").parent.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "reports" / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("initialize")
    prep = sub.add_parser("prepare-next")
    prep.add_argument("--ordinal", type=int, choices=ORDINALS, required=True)
    record = sub.add_parser("record-score")
    record.add_argument("--ordinal", type=int, choices=ORDINALS, required=True)
    record.add_argument("--iteration", type=int, choices=range(1, MAX_ITERATIONS + 1), required=True)
    record.add_argument("--response-file", type=Path, required=True)
    record.add_argument("--worker-context-id", required=True)
    sub.add_parser("status")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize()
    elif args.command == "prepare-next":
        result = prepare_next(args.ordinal)
    elif args.command == "record-score":
        result = record_score(args.ordinal, args.iteration, args.response_file,
                              args.worker_context_id)
    elif args.command == "status":
        result = status()
    else:
        result = finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
