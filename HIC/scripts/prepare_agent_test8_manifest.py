"""Build the frozen target-blind input manifest for Agent Test 8."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM.corpus_audit import DEFAULT_DATA_DIR
from LLM.native_hic_data import load_native_hic_splits
from LLM.hic_vnext_validation import validate_validation_window
from scripts.run_agent_test6 import _template, _dataset_description


SCOPE = (("Health", 36, 3), ("Security", 10, 3), ("new_metr", 192, 9))
RUN_ID = "agent_test8_remaining3families_15_20261101"
OUTPUT = ROOT / "LLM/evaluations/Agent Test 8/remaining3families_15_20261101"


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")


def build_manifest() -> tuple[dict, list[dict]]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Test 8 run directory: {OUTPUT}")
    jobs: list[dict] = []
    configurations: dict[str, dict] = {}
    for dataset, horizon, expected_count in SCOPE:
        split = load_native_hic_splits(
            dataset, horizon=horizon, include_outer_test_targets=False,
            include_validation_truth=False, include_empty_news=False,
        )
        windows = split["validation_windows"]
        if (len(windows) != expected_count or split.get("validation_truth_read") is not False
                or split["test_windows"] or not split["outer_test_deferred"]):
            raise ValueError(f"target-blind scope gate failed for {dataset}/H{horizon}")
        frequencies = {window["case"]["frequency"] for window in windows}
        if len(frequencies) != 1:
            raise ValueError(f"frequency is inconsistent for {dataset}/H{horizon}")
        frequency = next(iter(frequencies))
        source_path = DEFAULT_DATA_DIR / f"{dataset}.csv"
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        key = f"{dataset}/H{horizon}"
        configurations[key] = {
            "dataset": dataset, "horizon": horizon, "frequency": frequency,
            "contract": {
                "dataset": dataset, "horizon": horizon, "frequency": frequency,
                "target": "unknown", "unit": "unknown",
                "source_sha256": source_sha256,
            },
            "template": _template(dataset, horizon, frequency),
            "description": _dataset_description(dataset),
            "boundaries": split["boundaries"], "count": len(windows),
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha256,
        }
        for ordinal, window in enumerate(windows, start=1):
            case = window["case"]
            validate_validation_window({"case": case,
                                        "target_timestamps": window["target_timestamps"]},
                                       expected_horizon=horizon)
            if "truth" in case or "truth" in window or "validation_truth" in window:
                raise ValueError(f"truth-bearing input in target-blind case {case['case_id']}")
            identity = {"run_id": RUN_ID, "dataset": dataset,
                        "horizon": horizon, "case_id": case["case_id"]}
            jobs.append({
                **identity, "ordinal": len(jobs) + 1,
                "job_id": _canonical_hash(identity), "configuration": key,
                "case": case, "target_timestamps": window["target_timestamps"],
                "input_sha256": _canonical_hash(case),
            })
    counts = Counter((job["dataset"], job["horizon"]) for job in jobs)
    if len(jobs) != 15 or counts != Counter({(d, h): n for d, h, n in SCOPE}):
        raise ValueError(f"Test 8 scope mismatch: count={len(jobs)}, groups={dict(counts)}")
    manifest = {
        "run_id": RUN_ID,
        "protocol": "agent_test8_frozen_one_pass_direct_nearest_v1",
        "experiment_classification": "one_pass_frozen_evaluation_using_test7_memory_bank",
        "collection_name": "agent_test_8", "source_collection": "agent_test_7",
        "retrieval_mode": "direct_nearest_top1_no_eligibility_filters",
        "embedding_model": "BAAI/bge-m3", "embedding_dimension": 1024,
        "case_order": [f"{d}/H{h}" for d, h, _ in SCOPE],
        "configurations": configurations,
        "job_count": len(jobs), "group_counts": {f"{d}/H{h}": n for d, h, n in SCOPE},
        "jobs_sha256": _canonical_hash(jobs),
        "validation_truth_read": False, "outer_test_targets_read": False,
        "new_cases_written_to_retrieval_collection": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "target_blind_manifest_frozen_preflight_pending",
    }
    return manifest, jobs


def main() -> None:
    manifest, jobs = build_manifest()
    _write_new(OUTPUT / "run_manifest.json", manifest)
    _write_new(OUTPUT / "input_snapshots.json", jobs)
    (OUTPUT / "preflight").mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "run_id": manifest["run_id"], "manifest_sha256": hashlib.sha256(
            (OUTPUT / "run_manifest.json").read_bytes()).hexdigest(),
        "jobs_sha256": manifest["jobs_sha256"], "job_count": len(jobs),
        "groups": manifest["group_counts"], "case_ids": [job["case_id"] for job in jobs],
        "input_sha256": {job["case_id"]: job["input_sha256"] for job in jobs},
        "validation_truth_read": False, "outer_test_targets_read": False,
        "output": str(OUTPUT.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
