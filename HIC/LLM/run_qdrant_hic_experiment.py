"""Run the BGE-M3-only random-10 HIC warmup and final Qdrant evaluation."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from LLM import hic
from LLM.compare_bge_models import (
    LocalDenseEncoder,
    M3_DIMENSION,
    M3_MODEL_ID,
)
from LLM.hic_qdrant_eval import (
    MAX_LOGICAL_CALLS,
    _atomic_write_json,
    run_m3_warmup,
    run_prequential_evaluation,
)
from LLM.qdrant_import import (
    M3_MODEL,
    _m3_snapshot,
    build_seeded_training_sets,
    import_snapshots,
)
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.sample_data import load_socialgood_cases, load_socialgood_eval_windows


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "LLM" / "evaluations"
RANDOM_SEED = 20261017
WARMUP_COUNT = 3
FINAL_TEST_COUNT = 10
DEFAULT_MAX_CALLS = min(95, MAX_LOGICAL_CALLS)


def _run_id():
    return datetime.now(timezone.utc).strftime("qdrant_%Y%m%dT%H%M%SZ")


def _load_records():
    training_cases, _, _ = load_socialgood_cases(train_count=100)
    m3_records = _m3_snapshot(100)
    m3_by_id = {record["case_id"]: record for record in m3_records}
    case_ids = [case["case_id"] for case in training_cases]
    if list(m3_by_id) != case_ids:
        raise ValueError("BGE-M3 snapshot does not align with canonical training cases")
    return training_cases, m3_by_id


def _seed_store(store, cases, model_records):
    store.ensure_collection()
    for case in cases:
        record = model_records[case["case_id"]]
        store.upsert_completed_case(
            case_id=case["case_id"],
            summary=record["summary"],
            correction=record["correction"],
            embedding=record["embedding"],
            observed_targets=case["truth"],
            cutoff_date=case["target_end"],
            source=record["source"],
        )
    if store.count() != len(cases):
        raise RuntimeError(
            f"collection {store.collection_name} has {store.count()} points; "
            f"expected {len(cases)} after seeding"
        )


def _build_encoder(model_id):
    if model_id != M3_MODEL:
        raise ValueError("The active HIC workflow supports only BGE-M3")
    return LocalDenseEncoder(model_id, expected_dimension=M3_DIMENSION)


def _make_embedding_function(encoders, token_audit):
    def embed_summary(model_id, summary, is_query):
        text = hic.summary_to_embedding_text(summary)
        encoder = encoders[model_id]
        vectors, token_counts = encoder.encode([text], is_query=is_query, batch_size=1)
        if len(vectors) != 1:
            raise RuntimeError(f"{model_id} did not produce exactly one embedding")
        token_audit[model_id]["inputs"] += 1
        token_audit[model_id]["max_tokens"] = max(
            token_audit[model_id]["max_tokens"], token_counts[0]
        )
        token_audit[model_id]["truncated_inputs"] += token_counts[0] > encoder.max_length
        return vectors[0]

    return embed_summary


def _seed_suffix(run_id):
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]


def run_experiment(*, run_id, url="http://127.0.0.1:6333", max_calls=DEFAULT_MAX_CALLS):
    if not isinstance(max_calls, int) or max_calls < 0:
        raise ValueError("max_calls must be a non-negative integer")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before running the experiment.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"vot_hic_qdrant_{run_id}.json"
    validation_checkpoint = OUTPUT_DIR / f"{run_id}_validation.checkpoint.json"
    final_checkpoint = OUTPUT_DIR / f"{run_id}_final.checkpoint.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise RuntimeError("Install qdrant-client==1.19.1 before running the experiment.") from None

    client = QdrantClient(url=url, timeout=20)
    training_cases, m3_by_id = _load_records()
    arms, sampled_cases = build_seeded_training_sets(training_cases, seed=RANDOM_SEED)
    all_windows = load_socialgood_eval_windows(max_windows=13)
    warmup_windows = all_windows[:WARMUP_COUNT]
    final_windows = all_windows[WARMUP_COUNT:]
    if len(final_windows) != FINAL_TEST_COUNT:
        raise ValueError(f"expected {FINAL_TEST_COUNT} final windows, got {len(final_windows)}")

    # Preserve only the active model's stable 1/10/100 snapshots.
    snapshot_imports = import_snapshots(client, (1, 10, 100), (M3_MODEL,))
    namespace = _seed_suffix(run_id)
    warmup_stores = {
        M3_MODEL: QdrantHicStore(
            client=client,
            model_id=M3_MODEL,
            collection_name=collection_name_for(M3_MODEL, f"warmup_{namespace}"),
        ),
    }
    _seed_store(warmup_stores[M3_MODEL], sampled_cases, m3_by_id)

    encoders = {M3_MODEL: _build_encoder(M3_MODEL)}
    token_audit = {
        model_id: {"inputs": 0, "truncated_inputs": 0, "max_tokens": 0}
        for model_id in encoders
    }
    embed_summary = _make_embedding_function(encoders, token_audit)

    hic.LLM_CALL_COUNT = 0

    def call_json(prompt, *, temperature):
        response = hic.extract_json(
            hic.call_llm(prompt, temperature=temperature)
        )
        if not isinstance(response, dict) or "_raw" in response:
            raise ValueError("DeepSeek returned content that is not a valid JSON object")
        return response

    warmup = run_m3_warmup(
        windows=warmup_windows,
        template=hic.get_template(hic.industry),
        stores=warmup_stores,
        embed_summary=embed_summary,
        call_json=call_json,
        run_id=run_id,
        checkpoint_path=validation_checkpoint,
        max_calls=min(9, max_calls),
        retry_pending=False,
    )
    active_model = warmup["embedding_model"]

    final_stores = {
        arm: QdrantHicStore(
            client=client,
            model_id=active_model,
            collection_name=collection_name_for(
                active_model, f"eval_{namespace}_k{arm}"
            ),
        )
        for arm in (1, 10, 100)
    }
    active_model_records = m3_by_id
    for arm, cases in arms.items():
        _seed_store(final_stores[arm], cases, active_model_records)

    # Copy completed M3 warmup experiences into each final arm.
    validation_case_by_id = {
        window["case"]["case_id"]: window["case"] for window in warmup_windows
    }
    for window_result in warmup["validation_windows"]:
        model_result = window_result["models"][active_model]
        case = validation_case_by_id[window_result["case_id"]]
        for store in final_stores.values():
            store.upsert_completed_case(
                case_id=window_result["case_id"],
                summary=window_result["summary"],
                correction=model_result["correction"],
                embedding=model_result["passage_embedding"],
                observed_targets=window_result["truth"],
                prediction=model_result["prediction"],
                cutoff_date=case["target_end"],
                source=f"warmup:{run_id}:{active_model}",
            )

    final_call_budget = max_calls - warmup["logical_deepseek_calls"]
    if final_call_budget < 0:
        raise RuntimeError("validation exceeded the configured total logical call budget")
    final_evaluation = run_prequential_evaluation(
        windows=final_windows,
        template=hic.get_template(hic.industry),
        stores=final_stores,
        embed_summary=lambda summary, is_query: embed_summary(
            active_model, summary, is_query
        ),
        call_json=call_json,
        embedding_model=active_model,
        run_id=run_id,
        checkpoint_path=final_checkpoint,
        max_calls=final_call_budget,
    )

    final_counts = {str(arm): store.count() for arm, store in final_stores.items()}
    if final_counts != {"1": 14, "10": 23, "100": 113}:
        raise RuntimeError(f"unexpected final arm collection counts: {final_counts}")
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "SocialGood",
        "target": "OT monthly numeric series",
        "random_seed": RANDOM_SEED,
        "random_k10_case_ids": [case["case_id"] for case in sampled_cases],
        "random_k1_case_ids": [case["case_id"] for case in arms[1]],
        "k100_case_count": len(arms[100]),
        "warmup_description": (
            "The first three chronological windows are processed with BGE-M3 only. "
            "Their completed corrections are added to each HIC arm before the final "
            "ten-window test; they are excluded from final metrics."
        ),
        "active_embedding_model": active_model,
        "warmup": warmup,
        "embedding_token_audit": token_audit,
        "snapshot_imports": snapshot_imports,
        "final_evaluation": final_evaluation,
        "final_collection_counts": final_counts,
        "logical_deepseek_calls_total": (
            warmup["logical_deepseek_calls"]
            + final_evaluation["logical_deepseek_calls"]
        ),
        "planned_logical_call_budget": max_calls,
        "limitation": (
            "The 100 training cases overlap within one SocialGood monthly series. "
            "BGE-M3 is the only embedding model in this run. Three chronological "
            "warmup windows precede the final ten-window prequential test; all windows "
            "come from one SocialGood series and are not independent datasets."
        ),
    }
    _atomic_write_json(report_path, report)
    return report_path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    args = parser.parse_args(argv)
    run_id = args.run_id or _run_id()
    path, report = run_experiment(
        run_id=run_id,
        url=args.url,
        max_calls=args.max_calls,
    )
    print(
        json.dumps(
            {
                "report": str(path),
                "embedding_model": report["active_embedding_model"],
                "warmup_windows": len(report["warmup"]["validation_windows"]),
                "logical_deepseek_calls_total": report["logical_deepseek_calls_total"],
                "final_arms": report["final_evaluation"]["arms"],
                "final_collection_counts": report["final_collection_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
