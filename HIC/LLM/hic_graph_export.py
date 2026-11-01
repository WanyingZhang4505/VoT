"""Export a completed saved HIC run into leakage-aware graph records offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from LLM.hic_graph_data import build_scored_case_record, write_scored_case_jsonl
from LLM.hic_memory_builder import _fingerprint
from LLM.native_hic_data import load_native_hic_prefix_splits
from LLM.native_hic_experiment import make_numeric_baselines


def _load_json(path, label):
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value, source


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_numbers(left, right, label):
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        raise ValueError(f"{label} must be aligned numeric lists")
    if len(left) != len(right):
        raise ValueError(f"{label} lengths do not match")
    try:
        pairs = [(float(a), float(b)) for a, b in zip(left, right)]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain numeric values") from None
    if not all(
        math.isfinite(a) and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)
        for a, b in pairs
    ):
        raise ValueError(f"{label} values do not match")


def _checkpoint_embedding_config(report):
    config = report.get("embedding")
    if not isinstance(config, dict):
        raise ValueError("saved report is missing embedding configuration")
    model = config.get("model")
    dimensions = config.get("dimensions")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("saved report embedding model is missing")
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError("saved report embedding dimension is invalid")
    return config


def _saved_forecasts(value, case_id):
    if not isinstance(value, dict) or "0" not in value:
        raise ValueError(f"validation checkpoint forecasts are incomplete: {case_id}")
    result = {}
    for raw_arm, saved in value.items():
        arm = str(raw_arm)
        prediction = saved.get("prediction") if isinstance(saved, dict) else saved
        if not isinstance(prediction, list):
            raise ValueError(f"validation forecast arm {arm} is malformed: {case_id}")
        result[arm] = prediction
    return result


def build_checkpoint_scored_case_records(
    *,
    report_path,
    training_checkpoint_path,
    validation_checkpoint_path,
    data_dir=None,
):
    """Join only fit/validation source rows to saved HIC outputs.

    The loader stops before parsing target values after the report's validation
    boundary. The report/checkpoints contain validation labels already scored
    by the earlier run; outer-test values are neither needed nor loaded.
    """
    report, report_file = _load_json(report_path, "saved run report")
    training, training_file = _load_json(training_checkpoint_path, "training checkpoint")
    validation, validation_file = _load_json(validation_checkpoint_path, "validation checkpoint")

    if report.get("schema_version") != 1 or report.get("status") != "completed_validation_pilot":
        raise ValueError("saved report must be a completed schema-v1 validation pilot")
    run_id = report.get("run_id")
    plan = report.get("plan")
    boundaries = plan.get("boundaries") if isinstance(plan, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("saved report run_id is missing")
    if not isinstance(boundaries, dict):
        raise ValueError("saved report split boundaries are missing")
    dataset = plan.get("dataset")
    horizon = plan.get("horizon")
    training_ids = plan.get("training_case_ids")
    validation_ids = plan.get("validation_case_ids")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("saved report dataset is missing")
    if not isinstance(training_ids, list) or not training_ids or not all(isinstance(x, str) for x in training_ids):
        raise ValueError("saved report training_case_ids are missing")
    if not isinstance(validation_ids, list) or not validation_ids or not all(isinstance(x, str) for x in validation_ids):
        raise ValueError("saved report validation_case_ids are missing")
    if len(set(training_ids)) != len(training_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("saved report split IDs must be unique")
    if set(training_ids) & set(validation_ids):
        raise ValueError("training and validation case IDs overlap")
    if training.get("run_id") != f"{run_id}_train" or training.get("dataset") != dataset:
        raise ValueError("training checkpoint run or dataset does not match the saved report")
    if validation.get("run_id") != run_id:
        raise ValueError("validation checkpoint run does not match the saved report")
    if training.get("case_ids") != training_ids:
        raise ValueError("training checkpoint IDs do not match the saved report")
    if validation.get("case_ids") != validation_ids:
        raise ValueError("validation checkpoint IDs do not match the saved report")
    for name, checkpoint in (("training", training), ("validation", validation)):
        if checkpoint.get("pending_call"):
            raise ValueError(f"{name} checkpoint contains an unresolved API call")
        if checkpoint.get("history_length") != plan.get("history_length"):
            raise ValueError(f"{name} checkpoint history length does not match the report")
        if checkpoint.get("horizon") != horizon:
            raise ValueError(f"{name} checkpoint horizon does not match the report")

    embedding_config = _checkpoint_embedding_config(report)
    prefix = load_native_hic_prefix_splits(
        dataset,
        horizon=horizon,
        fit_end=boundaries.get("fit_end"),
        validation_end=boundaries.get("validation_end"),
        data_dir=data_dir,
    )
    fit_by_id = {case["case_id"]: case for case in prefix["fit_cases"]}
    validation_by_id = {
        item["case"]["case_id"]: item for item in prefix["validation_windows"]
    }
    if not set(training_ids).issubset(fit_by_id):
        missing = sorted(set(training_ids) - set(fit_by_id))
        raise ValueError(f"saved training cases are absent from the cutoff-safe source prefix: {missing}")
    if set(validation_ids) != set(validation_by_id):
        raise ValueError("saved validation case IDs do not match the cutoff-safe source prefix")
    if plan.get("training_count") != len(training_ids) or plan.get("validation_count") != len(validation_ids):
        raise ValueError("saved report split counts do not match its case ID lists")

    ordered_training_cases = [fit_by_id[case_id] for case_id in training_ids]
    if training.get("input_fingerprint") != _fingerprint(ordered_training_cases):
        raise ValueError("training checkpoint fingerprint does not match source fit cases")

    report_windows = report.get("validation", {}).get("windows")
    if not isinstance(report_windows, list):
        raise ValueError("saved report validation windows are missing")
    report_validation_by_id = {row.get("case_id"): row for row in report_windows if isinstance(row, dict)}
    if set(report_validation_by_id) != set(validation_ids):
        raise ValueError("saved report validation windows do not match its validation IDs")

    provenance_base = {
        "run_id": run_id,
        "source_report": report_file.name,
        "source_report_sha256": _sha256(report_file),
        "training_checkpoint": training_file.name,
        "training_checkpoint_sha256": _sha256(training_file),
        "validation_checkpoint": validation_file.name,
        "validation_checkpoint_sha256": _sha256(validation_file),
        "source_csv": f"{dataset}.csv",
        "source_prefix_validation_end": boundaries["validation_end"],
        "template_sha256": report.get("template", {}).get("sha256"),
        "embedding_model": embedding_config["model"],
        "embedding_configuration": embedding_config,
        "outer_test_targets_loaded": False,
    }

    records = []
    for case_id in training_ids:
        case = fit_by_id[case_id]
        state = training.get("cases", {}).get(case_id)
        if not isinstance(state, dict) or not all(
            state.get(key) for key in ("summary", "forecast", "correction", "embedding", "ingested")
        ):
            raise ValueError(f"training checkpoint record is incomplete: {case_id}")
        vector = state["embedding"]
        if len(vector) != embedding_config["dimensions"]:
            raise ValueError(f"training embedding dimension is invalid: {case_id}")
        forecasts = {"0": state["forecast"].get("prediction")}
        baseline_forecasts = make_numeric_baselines(case["series"], horizon=horizon)
        provenance = {**provenance_base, "split": "fit", "embedding_role": "stored passage"}
        records.append(build_scored_case_record(
            case,
            target_timestamps=case["target_timestamps"],
            summary=state["summary"],
            embedding=vector,
            forecasts=forecasts,
            truth=case["truth"],
            corrections={"0": state["correction"]},
            baseline_forecasts=baseline_forecasts,
            provenance=provenance,
        ))

    for case_id in validation_ids:
        item = validation_by_id[case_id]
        case = item["case"]
        source_truth = item["truth"]
        source_target_timestamps = item["target_timestamps"]
        report_window = report_validation_by_id[case_id]
        state = validation.get("windows", {}).get(case_id)
        if not isinstance(state, dict):
            raise ValueError(f"validation checkpoint record is missing: {case_id}")
        scoring = state.get("scoring")
        embeddings = state.get("embeddings")
        saved_forecasts = state.get("forecasts")
        corrections = state.get("corrections")
        if not isinstance(scoring, dict) or not isinstance(embeddings, dict):
            raise ValueError(f"validation checkpoint record is incomplete: {case_id}")
        forecasts = _saved_forecasts(saved_forecasts, case_id)
        if not isinstance(corrections, dict) or not isinstance(state.get("summary"), dict):
            raise ValueError(f"validation checkpoint summary/corrections are incomplete: {case_id}")
        query_embedding = embeddings.get("query")
        if not isinstance(query_embedding, list) or len(query_embedding) != embedding_config["dimensions"]:
            raise ValueError(f"validation query embedding dimension is invalid: {case_id}")

        _same_numbers(scoring.get("truth"), source_truth, f"validation checkpoint truth for {case_id}")
        _same_numbers(report_window.get("truth"), source_truth, f"validation report truth for {case_id}")
        if scoring.get("truth") != report_window.get("truth"):
            _same_numbers(scoring["truth"], report_window["truth"], f"saved validation truth for {case_id}")
        if scoring.get("target_timestamps", source_target_timestamps) != source_target_timestamps:
            raise ValueError(f"validation checkpoint target timestamps do not match source for {case_id}")
        if report_window.get("target_timestamps") != source_target_timestamps:
            raise ValueError(f"validation report target timestamps do not match source for {case_id}")

        report_arms = report_window.get("arms")
        if not isinstance(report_arms, dict) or set(report_arms) != set(forecasts):
            raise ValueError(f"saved validation report forecast arms do not match checkpoint for {case_id}")
        for arm, prediction in forecasts.items():
            _same_numbers(
                report_arms[arm].get("prediction"), prediction,
                f"validation checkpoint prediction arm {arm} for {case_id}",
            )
        correction_values = {
            str(arm): text for arm, text in corrections.items()
        }
        if "0" not in correction_values:
            correction_values["0"] = "No retrieved correction was generated for the K=0 control."
        if set(correction_values) != set(forecasts):
            raise ValueError(f"validation checkpoint corrections do not align with forecast arms: {case_id}")

        provenance = {**provenance_base, "split": "validation", "embedding_role": "stored query"}
        records.append(build_scored_case_record(
            case,
            target_timestamps=source_target_timestamps,
            summary=state["summary"],
            embedding=query_embedding,
            forecasts=forecasts,
            truth=source_truth,
            corrections=correction_values,
            baseline_forecasts=make_numeric_baselines(case["series"], horizon=horizon),
            provenance=provenance,
        ))

    records.sort(key=lambda record: (
        record["known_at_forecast"]["target_start"],
        record["known_at_forecast"]["case_id"],
    ))
    outer_ids = set(plan.get("outer_test_case_ids") or [])
    output_ids = {record["known_at_forecast"]["case_id"] for record in records}
    if output_ids & outer_ids:
        raise ValueError("export unexpectedly contains outer-test case IDs")
    max_record_target_end = max(
        record["known_at_forecast"]["target_end"] for record in records
    )
    if max_record_target_end > boundaries["validation_end"]:
        raise ValueError("export contains a target period beyond the validation cutoff")

    audit = {
        "run_id": run_id,
        "dataset": dataset,
        "horizon": horizon,
        "fit_cases": len(training_ids),
        "validation_cases": len(validation_ids),
        "total_records": len(records),
        "rows_after_validation_cutoff": prefix["quality"]["rows_after_cutoff"],
        "validation_cutoff": boundaries["validation_end"],
        "max_exported_target_end": max_record_target_end,
        "outer_test_targets_loaded": False,
        "outer_test_records_exported": len(output_ids & outer_ids),
    }
    return records, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records, audit = build_checkpoint_scored_case_records(
        report_path=args.report,
        training_checkpoint_path=args.training_checkpoint,
        validation_checkpoint_path=args.validation_checkpoint,
        data_dir=args.data_dir,
    )
    write_scored_case_jsonl(args.output, records)
    audit["output"] = str(args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
