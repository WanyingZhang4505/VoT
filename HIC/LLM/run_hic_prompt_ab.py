"""Run a fixed-summary, BGE-M3 HIC retrieval-prompt A/B replay."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from LLM import hic
from LLM.compare_bge_models import LocalDenseEncoder, M3_DIMENSION, M3_MODEL_ID
from LLM.hic_prompt_ab import run_prompt_ab_evaluation
from LLM.hic_qdrant_eval import _atomic_write_json
from LLM.qdrant_import import M3_MODEL, _m3_snapshot, build_seeded_training_sets
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.sample_data import load_socialgood_cases, load_socialgood_eval_windows


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "LLM" / "evaluations"
BASE_REPORT = OUTPUT_DIR / "vot_hic_qdrant_hic_m3only_20261017.json"
RANDOM_SEED = 20261017
MAX_CALLS = 12
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _default_run_id():
    return datetime.now(timezone.utc).strftime("promptab_%Y%m%dT%H%M%SZ")


def load_saved_summaries(report_path, expected_case_ids):
    """Load exact summary JSON from the previous M3 warmup by stable case ID."""
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"saved M3 summary report not found: {path}")
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("active_embedding_model") != hic.BGE_MODEL_NAME:
        raise ValueError("saved report does not use the currently active BGE-M3 model")
    windows = report.get("warmup", {}).get("validation_windows", [])
    summaries = {}
    for window in windows:
        case_id = window.get("case_id")
        summary = window.get("summary")
        if not isinstance(case_id, str) or not isinstance(summary, dict):
            continue
        if case_id in summaries:
            raise ValueError(f"saved M3 report contains duplicate summary for {case_id}")
        summaries[case_id] = summary
    missing = [case_id for case_id in expected_case_ids if case_id not in summaries]
    if missing:
        raise ValueError(f"saved summary is missing for: {', '.join(missing)}")
    return {case_id: summaries[case_id] for case_id in expected_case_ids}


def _collection_suffix(run_id, policy):
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"promptab_{digest}_{policy}"


def run_experiment(
    *,
    run_id,
    url="http://127.0.0.1:6333",
    max_calls=MAX_CALLS,
    summary_report=BASE_REPORT,
):
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain 1-64 ASCII letters, digits, '_' or '-'")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before running the experiment.")

    output_path = OUTPUT_DIR / f"vot_hic_prompt_ab_{run_id}.json"
    checkpoint_path = OUTPUT_DIR / f"vot_hic_prompt_ab_{run_id}.checkpoint.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
    if max_calls < 10 or max_calls > MAX_CALLS:
        raise ValueError(f"max_calls must be between 10 and {MAX_CALLS}")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise RuntimeError("Install qdrant-client==1.19.1 before running the experiment.") from None
    client = QdrantClient(url=url, timeout=20)

    all_training_cases, _, _ = load_socialgood_cases(train_count=100)
    _, seed_cases = build_seeded_training_sets(all_training_cases, seed=RANDOM_SEED)
    records_by_id = {record["case_id"]: record for record in _m3_snapshot(100)}
    windows = load_socialgood_eval_windows(max_windows=2)
    case_ids = [window["case"]["case_id"] for window in windows]
    summaries = load_saved_summaries(summary_report, case_ids)

    collection_names = {
        policy: collection_name_for(
            M3_MODEL, _collection_suffix(run_id, policy)
        )
        for policy in ("current", "relevance_gated")
    }
    has_checkpoint = checkpoint_path.is_file()
    for name in collection_names.values():
        if client.collection_exists(name) and not has_checkpoint:
            raise FileExistsError(
                f"Qdrant collection {name} already exists; choose a new run ID"
            )
    stores = {
        policy: QdrantHicStore(
            client=client,
            model_id=M3_MODEL,
            collection_name=name,
        )
        for policy, name in collection_names.items()
    }

    encoder = LocalDenseEncoder(M3_MODEL_ID, expected_dimension=M3_DIMENSION)
    token_audit = {"inputs": 0, "max_tokens": 0, "truncated_inputs": 0}

    def embed_summary(summary, *, is_query):
        text = hic.summary_to_embedding_text(summary)
        vectors, token_counts = encoder.encode(
            [text], is_query=is_query, batch_size=1
        )
        if len(vectors) != 1 or len(vectors[0]) != M3_DIMENSION:
            raise RuntimeError("BGE-M3 did not produce one correctly sized vector")
        token_audit["inputs"] += 1
        token_audit["max_tokens"] = max(token_audit["max_tokens"], token_counts[0])
        token_audit["truncated_inputs"] += token_counts[0] > encoder.max_length
        return vectors[0]

    hic.LLM_CALL_COUNT = 0

    def call_json(prompt, *, temperature):
        response = hic.extract_json(hic.call_llm(prompt, temperature=temperature))
        if not isinstance(response, dict) or "_raw" in response:
            raise ValueError("DeepSeek returned content that is not a valid JSON object")
        return response

    result = run_prompt_ab_evaluation(
        windows=windows,
        summaries=summaries,
        stores=stores,
        seed_cases=seed_cases,
        records_by_id=records_by_id,
        embed_summary=embed_summary,
        call_json=call_json,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        max_calls=max_calls,
    )
    result.update(
        {
            "dataset": "SocialGood OT monthly series",
            "random_seed": RANDOM_SEED,
            "initial_training_case_count": len(seed_cases),
            "evaluation_case_ids": case_ids,
            "summary_source_path": str(Path(summary_report)),
            "qdrant_collections": collection_names,
            "vector_dimension": M3_DIMENSION,
            "embedding_max_length": encoder.max_length,
            "embedding_token_audit": token_audit,
            "prompt_variants": {
                "no_retrieval": "Current reasoning prompt with no correction context.",
                "current": "Existing top-1 retrieval correction prompt.",
                "relevance_gated": (
                    "Top-1 retrieval prompt that requires checking whether the "
                    "prior error pattern applies before using the correction."
                ),
            },
            "logical_deepseek_calls_reported_by_client": hic.LLM_CALL_COUNT,
            "call_count_note": (
                "Counts logical chat calls; lower-level HTTP retries and token usage "
                "are not captured by the client."
            ),
        }
    )
    _atomic_write_json(output_path, result)
    return output_path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--summary-report", type=Path, default=BASE_REPORT)
    args = parser.parse_args(argv)
    run_id = args.run_id or _default_run_id()
    path, report = run_experiment(
        run_id=run_id,
        url=args.url,
        max_calls=args.max_calls,
        summary_report=args.summary_report,
    )
    print(
        json.dumps(
            {
                "report": str(path),
                "run_id": report["run_id"],
                "logical_deepseek_calls": report["logical_deepseek_calls"],
                "arms": report["arms"],
                "final_collection_counts": report["final_collection_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
