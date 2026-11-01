"""Run one bounded, leakage-aware Economy HIC/Qdrant pilot."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from LLM import hic
from LLM.economy_pilot import (
    DEFAULT_CSV_PATH,
    DEFAULT_SEED,
    load_economy_pilot_cases,
    run_economy_hic_pilot,
)
from LLM.deepseek_client import HttpAttemptBudget
from LLM.qdrant_store import QdrantHicStore, collection_name_for


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    preserve_temporary = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if attempt == 4:
                    preserve_temporary = True
                    raise PermissionError(
                        f"could not replace HTTP-attempt ledger after 5 attempts; "
                        f"complete recovery file retained at {temporary}"
                    ) from error
                time.sleep(0.1 * (attempt + 1))
    finally:
        if temporary is not None and temporary.exists() and not preserve_temporary:
            temporary.unlink()


def _paths(run_id):
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_id)
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    return (
        EVALUATIONS / f"economy_hic_{safe_id}.json",
        EVALUATIONS / f"economy_hic_{safe_id}.checkpoint.json",
        collection_name_for("BAAI/bge-m3", f"economy_pilot_{digest}"),
    )


def _baseline(history, truth):
    persistence = hic.persistence_forecast(history, len(truth))
    mean = [sum(history) / len(history)] * len(truth)
    return {
        "persistence": hic.forecast_metrics(persistence, truth),
        "history_mean": hic.forecast_metrics(mean, truth),
    }


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def make_http_attempt_budget(*, max_attempts, ledger_path, resume):
    """Create a cumulative physical-attempt cap persisted across resume."""
    ledger_path = Path(ledger_path)
    if resume:
        if not ledger_path.is_file():
            raise RuntimeError(
                "cannot resume safely without the matching HTTP attempt ledger"
            )
        try:
            saved = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("HTTP attempt ledger is unreadable") from None
        if not isinstance(saved, dict):
            raise RuntimeError("HTTP attempt ledger must contain a JSON object")
        if (
            saved.get("schema_version") != 1
            or saved.get("max_http_attempts") != max_attempts
        ):
            raise ValueError("HTTP attempt ledger does not match the requested cap")
        attempts_used = saved.get("attempts_used")
        if type(attempts_used) is not int or not 0 <= attempts_used <= max_attempts:
            raise ValueError("HTTP attempt ledger contains an invalid attempt count")
    else:
        if ledger_path.exists():
            raise FileExistsError("HTTP attempt ledger already exists; use --resume or a new run ID")
        attempts_used = 0

    def persist_attempt_count(used):
        _atomic_json(ledger_path, {
            "schema_version": 1,
            "max_http_attempts": max_attempts,
            "attempts_used": used,
        })

    budget = HttpAttemptBudget(
        max_attempts=max_attempts,
        attempts_used=attempts_used,
        on_consume=persist_attempt_count,
    )
    if not resume:
        persist_attempt_count(attempts_used)
    return budget


def make_call_json(*, attempt_budget):
    """Build the pilot's JSON caller with one shared physical-attempt budget."""
    def call_json(prompt, temperature=0.0):
        raw = hic.call_llm(
            prompt, temperature=temperature, attempt_budget=attempt_budget
        )
        parsed = hic.extract_json(raw)
        if not isinstance(parsed, dict) or "_raw" in parsed:
            raise ValueError("DeepSeek response was not a valid JSON object")
        return parsed

    return call_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="economy_20261018_01")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--max-calls", type=int, default=35)
    parser.add_argument(
        "--max-http-attempts", type=_positive_int,
        help="required for live runs; cumulative physical HTTP-attempt ceiling, including retries",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate selection and baselines without API or Qdrant writes")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.max_http_attempts is None:
        parser.error("--max-http-attempts is required for a live run")
    train_cases, test_windows = load_economy_pilot_cases(
        args.csv, train_count=10, seed=DEFAULT_SEED
    )
    test_window = test_windows[0]
    report_path, checkpoint_path, collection_name = _paths(args.run_id)
    offline = {
        "run_id": args.run_id,
        "dataset": "Economy_sort",
        "split": "chronological 80/20; random seeded selection within training partition",
        "seed": DEFAULT_SEED,
        "training_case_ids": [case["case_id"] for case in train_cases],
        "training_count": len(train_cases),
        "available_nonoverlapping_test_windows": len(test_windows),
        "first_test_case_id": test_window["case"]["case_id"],
        "first_test_history": [test_window["case"]["history_start"], test_window["case"]["history_end"]],
        "first_test_target": [test_window["case"]["target_start"], test_window["case"]["target_end"]],
        "first_test_baselines": _baseline(test_window["case"]["series"], test_window["truth"]),
        "qdrant_collection": collection_name,
        "report_path": str(report_path),
        "checkpoint_path": str(checkpoint_path),
        "max_logical_deepseek_calls": args.max_calls,
        "max_http_attempts": args.max_http_attempts,
        "http_attempt_ledger_path": str(
            checkpoint_path.with_name(f"{checkpoint_path.stem}.http_attempts.json")
        ),
        "source_note": "Economy_sort OT unit is unspecified; results are a pilot, not evidence of real-world forecast accuracy.",
    }
    if args.dry_run:
        print(json.dumps(offline, ensure_ascii=False, indent=2))
        return 0

    if args.max_calls < 35:
        raise ValueError("the first complete Economy pilot needs a logical-call budget of at least 35")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; secret value was not displayed")
    if not args.csv.is_file():
        raise FileNotFoundError(f"Economy source CSV not found: {args.csv}")
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError("--resume requires the matching Economy checkpoint")
    if not args.resume and checkpoint_path.exists():
        raise FileExistsError("checkpoint already exists; use --resume or choose a new --run-id")

    store = QdrantHicStore(
        model_id="BAAI/bge-m3",
        collection_name=collection_name,
        url=args.qdrant_url,
    )
    exists = store.client.collection_exists(collection_name)
    if args.resume:
        if not exists:
            raise RuntimeError("resume checkpoint exists but its Qdrant collection is missing")
    elif exists and store.count() != 0:
        raise FileExistsError(
            f"refusing to reuse non-empty Qdrant collection {collection_name}; choose a new run ID"
        )
    store.ensure_collection()

    embedder = hic.get_embedder()
    attempt_ledger_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.http_attempts.json"
    )
    http_attempt_budget = make_http_attempt_budget(
        max_attempts=args.max_http_attempts,
        ledger_path=attempt_ledger_path,
        resume=args.resume,
    )

    def embed_summary(summary, is_query=False):
        text = hic.summary_to_embedding_text(summary)
        vectors = embedder.encode([text], is_query=is_query)
        if len(vectors) != 1:
            raise RuntimeError("BGE-M3 did not return exactly one embedding")
        return vectors[0]

    call_json = make_call_json(attempt_budget=http_attempt_budget)

    result = run_economy_hic_pilot(
        train_cases=train_cases,
        test_window=test_window,
        template=None,
        store=store,
        embed_summary=embed_summary,
        call_json=call_json,
        run_id=args.run_id,
        max_calls=args.max_calls,
        checkpoint_path=checkpoint_path,
        retry_pending=args.retry_pending,
    )
    result.update(offline)
    result["qdrant_count_verified"] = store.count()
    result["hic_call_invocations"] = hic.LLM_CALL_COUNT
    result["http_attempts_used"] = http_attempt_budget.attempts_used
    _atomic_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
