"""Run the remaining Economy holdout windows with an incremental M3 memory."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from LLM import hic
from LLM.economy_pilot import DEFAULT_SEED, DEFAULT_CSV_PATH, load_economy_pilot_cases
from LLM.economy_rolling_eval import _atomic_write_json, run_economy_rolling_eval
from LLM.qdrant_store import QdrantHicStore, collection_name_for


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"
DEFAULT_SEED_RUN_ID = "economy_20261018_covars01"


def _seed_paths(run_id):
    return (
        EVALUATIONS / f"economy_hic_{run_id}.json",
        EVALUATIONS / f"economy_hic_{run_id}.checkpoint.json",
    )


def _rolling_paths(run_id):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    collection_suffix = f"economy_roll_{safe_id.lower()[:16]}_{digest}"
    return (
        EVALUATIONS / f"economy_rolling_hic_{safe_id}.json",
        EVALUATIONS / f"economy_rolling_hic_{safe_id}.checkpoint.json",
        collection_name_for("BAAI/bge-m3", collection_suffix),
    )


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def completed_seed_records(
    *, training_cases, first_test_window, seed_report, seed_checkpoint, source_run_id
):
    """Rebuild the exact 10+1 completed experience records from a verified pilot."""
    expected_train_ids = [case["case_id"] for case in training_cases]
    test_case_id = first_test_window["case"]["case_id"]
    if seed_report.get("run_id") != source_run_id or seed_checkpoint.get("run_id") != source_run_id:
        raise ValueError("seed report/checkpoint do not match the requested source run")
    if (
        seed_report.get("training_case_ids") != expected_train_ids
        or seed_checkpoint.get("train_case_ids") != expected_train_ids
    ):
        raise ValueError("seed training cases differ from the original deterministic selection")
    if (
        seed_report.get("test", {}).get("case_id") != test_case_id
        or seed_checkpoint.get("test_case_id") != test_case_id
    ):
        raise ValueError("seed case 11 differs from the first chronological test window")
    if (
        seed_report.get("collection_count_after_test") != 11
        or seed_report.get("qdrant_count_verified") != 11
    ):
        raise ValueError("the source pilot report does not verify exactly 11 completed vectors")

    training_state = seed_checkpoint.get("training")
    if not isinstance(training_state, dict) or set(training_state) != set(expected_train_ids):
        raise ValueError("source checkpoint does not contain exactly the ten training cases")
    test_state = seed_checkpoint.get("test", {})
    if not (test_state.get("scored") and test_state.get("ingested")):
        raise ValueError("case 11 must be scored and ingested before it seeds a later window")
    if test_state.get("scoring", {}).get("truth") != first_test_window.get("truth"):
        raise ValueError("case 11 checkpoint truth does not match the source dataset window")

    records = []
    for case in training_cases:
        case_id = case["case_id"]
        entry = training_state[case_id]
        if not entry.get("ingested"):
            raise ValueError(f"training case {case_id} is not marked ingested")
        records.append({
            "case_id": case_id,
            "summary": entry.get("summary"),
            "correction": entry.get("correction"),
            "embedding": entry.get("embedding"),
            "observed_targets": case.get("truth"),
            "prediction": entry.get("forecast", {}).get("prediction"),
            "cutoff_date": case.get("target_end"),
            "source": f"economy-rolling-seed:{source_run_id}",
        })

    records.append({
        "case_id": test_case_id,
        "summary": test_state.get("summary"),
        "correction": test_state.get("correction"),
        "embedding": test_state.get("embeddings", {}).get("passage"),
        "observed_targets": first_test_window.get("truth"),
        "prediction": test_state.get("retrieval", {}).get("prediction"),
        "cutoff_date": first_test_window["case"].get("target_end"),
        "source": f"economy-rolling-seed:{source_run_id}",
    })
    for record in records:
        if not isinstance(record["summary"], dict) or not record["summary"]:
            raise ValueError(f"seed summary is missing for {record['case_id']}")
        if not isinstance(record["correction"], str) or not record["correction"].strip():
            raise ValueError(f"seed correction is missing for {record['case_id']}")
        if not isinstance(record["embedding"], list) or len(record["embedding"]) != 1024:
            raise ValueError(f"seed BGE-M3 embedding is missing or malformed for {record['case_id']}")
        if not isinstance(record["observed_targets"], list) or len(record["observed_targets"]) != 12:
            raise ValueError(f"seed targets are missing for {record['case_id']}")
        if not isinstance(record["prediction"], list) or len(record["prediction"]) != 12:
            raise ValueError(f"seed forecast is missing for {record['case_id']}")
        if not record["cutoff_date"]:
            raise ValueError(f"seed cutoff date is missing for {record['case_id']}")
    return records


def seed_completed_memory(
    *, store, training_cases, first_test_window, seed_report, seed_checkpoint, source_run_id
):
    records = completed_seed_records(
        training_cases=training_cases,
        first_test_window=first_test_window,
        seed_report=seed_report,
        seed_checkpoint=seed_checkpoint,
        source_run_id=source_run_id,
    )
    if store.count() != 0:
        raise ValueError("new rolling-evaluation collection must be empty before seeding")
    for record in records:
        store.upsert_completed_case(**record)
    if store.count() != len(records):
        raise RuntimeError(f"expected {len(records)} seeded cases, found {store.count()}")
    return [record["case_id"] for record in records]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="economy_rolling_20261018_01")
    parser.add_argument("--seed-run-id", default=DEFAULT_SEED_RUN_ID)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-pending", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate seed artifacts and list remaining windows without API or Qdrant writes",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, '_' or '-' characters")
    if args.max_calls < 1 or not 1 <= args.max_retries <= 6:
        raise ValueError("max_calls must be positive and max_retries must be between 1 and 6")

    train_cases, all_test_windows = load_economy_pilot_cases(
        args.csv, train_count=10, seed=DEFAULT_SEED
    )
    if len(all_test_windows) < 2:
        raise ValueError("the Economy data must contain case 11 and at least one later test window")
    first_test_window = all_test_windows[0]
    windows = all_test_windows[1:]
    seed_report_path, seed_checkpoint_path = _seed_paths(args.seed_run_id)
    if not seed_report_path.is_file() or not seed_checkpoint_path.is_file():
        raise FileNotFoundError("the verified case-11 seed report/checkpoint is missing")
    seed_report = _read_json(seed_report_path)
    seed_checkpoint = _read_json(seed_checkpoint_path)
    records = completed_seed_records(
        training_cases=train_cases,
        first_test_window=first_test_window,
        seed_report=seed_report,
        seed_checkpoint=seed_checkpoint,
        source_run_id=args.seed_run_id,
    )
    report_path, checkpoint_path, collection_name = _rolling_paths(args.run_id)
    required_calls = len(windows) * 4
    if not args.resume and args.max_calls < required_calls:
        raise ValueError(
            f"this run requires at least {required_calls} logical calls; "
            f"increase --max-calls above {args.max_calls}"
        )
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError("--resume requires the rolling-evaluation checkpoint")
    if not args.resume and (report_path.exists() or checkpoint_path.exists()):
        raise FileExistsError("report/checkpoint already exists; use --resume or another run ID")

    offline = {
        "run_id": args.run_id,
        "seed_run_id": args.seed_run_id,
        "dataset": "Economy_sort",
        "protocol": (
            "The 2017-03 case 11 was already used in development. Freeze its current "
            "BGE-M3/top-1 HIC configuration, then prospectively evaluate the six later "
            "chronological windows; reveal each target only after K=0 and K=1 forecasts, "
            "then append its completed correction for subsequent windows."
        ),
        "split": "Original chronological 80/20; first holdout window previously used for development, remaining windows reserved for this frozen run.",
        "seed_case_ids": [record["case_id"] for record in records],
        "test_case_ids": [window["case"]["case_id"] for window in windows],
        "expected_logical_deepseek_calls": required_calls,
        "max_logical_deepseek_calls": args.max_calls,
        "max_http_attempts_per_logical_call": args.max_retries,
        "provider_usage_metadata": "Not exposed by the current DeepSeek client wrapper.",
        "source_collection": seed_report.get("qdrant_collection"),
        "qdrant_collection": collection_name,
        "report_path": str(report_path),
        "checkpoint_path": str(checkpoint_path),
        "source_report_path": str(seed_report_path),
        "source_checkpoint_path": str(seed_checkpoint_path),
    }
    if args.dry_run:
        print(json.dumps(offline, ensure_ascii=False, indent=2))
        return 0

    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; secret value was not displayed")
    if not args.csv.is_file():
        raise FileNotFoundError(f"Economy source CSV not found: {args.csv}")

    store = QdrantHicStore(
        model_id="BAAI/bge-m3",
        collection_name=collection_name,
        url=args.qdrant_url,
    )
    exists = store.client.collection_exists(collection_name)
    if args.resume:
        if not exists:
            raise RuntimeError("rolling checkpoint exists but its Qdrant collection is missing")
    else:
        if exists:
            raise FileExistsError(
                f"refusing to reuse existing collection {collection_name}; choose a new run ID"
            )
        store.ensure_collection()
        seed_case_ids = seed_completed_memory(
            store=store,
            training_cases=train_cases,
            first_test_window=first_test_window,
            seed_report=seed_report,
            seed_checkpoint=seed_checkpoint,
            source_run_id=args.seed_run_id,
        )
        offline["seed_case_ids"] = seed_case_ids

    embedder = hic.get_embedder()

    def embed_summary(summary, is_query=False):
        text = hic.summary_to_embedding_text(summary)
        vectors = embedder.encode([text], is_query=is_query)
        if len(vectors) != 1:
            raise RuntimeError("BGE-M3 did not return exactly one embedding")
        return vectors[0]

    def call_json(prompt, temperature=0.0):
        raw = hic.call_llm(prompt, temperature=temperature, max_retries=args.max_retries)
        parsed = hic.extract_json(raw)
        if not isinstance(parsed, dict) or "_raw" in parsed:
            raise ValueError("DeepSeek response was not a valid JSON object")
        return parsed

    hic.LLM_CALL_COUNT = 0
    result = run_economy_rolling_eval(
        windows=windows,
        seed_case_ids=offline["seed_case_ids"],
        template=seed_report["template"],
        store=store,
        embed_summary=embed_summary,
        call_json=call_json,
        run_id=args.run_id,
        max_calls=args.max_calls,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        retry_pending=args.retry_pending,
    )
    result.update(offline)
    result["template"] = seed_report["template"]
    result["qdrant_count_verified"] = store.count()
    result["hic_logical_call_invocations"] = hic.LLM_CALL_COUNT
    _atomic_write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
