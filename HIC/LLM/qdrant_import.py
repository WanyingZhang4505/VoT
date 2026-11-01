"""Import the repository's saved HIC/BGE-M3 snapshots into local Qdrant."""

import argparse
import json
from pathlib import Path

from LLM import hic
from LLM.bge_comparison import validate_case_alignment
from LLM.compare_bge_models import M3_DIMENSION, M3_KB_PATH
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.sample_data import load_socialgood_cases, select_seeded_cases


REPO_ROOT = Path(__file__).resolve().parent.parent
COUNTS = (1, 10, 100)
M3_MODEL = hic.BGE_MODEL_NAME


def build_seeded_training_sets(cases, seed=20261017):
    """Build nested K=1/K=10 random arms and the complete K=100 arm."""
    candidates = list(cases)
    if len(candidates) < 100:
        raise ValueError("the K=100 arm requires exactly 100 historical cases")
    if len(candidates) != 100:
        raise ValueError("training input must contain exactly 100 historical cases")
    sampled = select_seeded_cases(candidates, count=10, seed=seed)
    return {1: sampled[:1], 10: sampled, 100: candidates}, sampled


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def prepare_import_records(
    *, cases, case_ids, summaries, corrections, embeddings, source
):
    """Align persisted vectors/lessons to complete time-safe sample records."""
    expected_ids = [case.get("case_id") for case in cases]
    if expected_ids != list(case_ids):
        raise ValueError("case ID alignment does not match the canonical sample order")
    lengths = {len(cases), len(summaries), len(corrections), len(embeddings)}
    if len(lengths) != 1 or len(case_ids) != len(cases):
        raise ValueError("case IDs, samples, summaries, corrections, and vectors must align")

    prepared = []
    for case, summary, correction, vector in zip(
        cases, summaries, corrections, embeddings
    ):
        targets = case.get("truth")
        if not isinstance(targets, list) or len(targets) != 12:
            raise ValueError(
                f"historical case {case['case_id']} must have 12 observed targets"
            )
        if not isinstance(correction, str) or not correction.strip():
            raise ValueError(f"historical case {case['case_id']} has an empty correction")
        prepared.append(
            {
                "case_id": case["case_id"],
                "summary": summary,
                "correction": correction,
                "embedding": vector,
                "observed_targets": targets,
                "cutoff_date": case["target_end"],
                "source": source,
            }
        )
    return prepared


def _m3_snapshot(count):
    if not M3_KB_PATH.is_file():
        raise FileNotFoundError("Saved 100-case BGE-M3 vector cache is required.")
    cache = _read_json(M3_KB_PATH)
    if cache.get("model_id") != M3_MODEL:
        raise ValueError("BGE-M3 cache has an incompatible model ID.")
    vectors = cache.get("embeddings", [])
    ids = cache.get("case_ids", [])
    summary_texts = cache.get("summary_texts", [])
    if not (len(vectors) == len(ids) == len(summary_texts) == 100):
        raise ValueError("BGE-M3 cache must contain 100 aligned IDs, summaries, and vectors.")
    if any(len(vector) != M3_DIMENSION for vector in vectors):
        raise ValueError("BGE-M3 cache contains a vector with the wrong dimension.")

    legacy_records = _read_json(
        REPO_ROOT / "LLM" / "knowledge_base" / "kb_socialgoods_deepseek_100train.json"
    )
    output = _read_json(REPO_ROOT / "LLM" / "vot_hic_output_100train.json")
    cases, _, _ = load_socialgood_cases(train_count=100)
    canonical_ids = validate_case_alignment(cases, output.get("train_case_ids", []))
    if ids != canonical_ids or len(legacy_records) != 100:
        raise ValueError("BGE-M3 cache and canonical 100-case HIC records are misaligned.")
    for index, (record, summary_text) in enumerate(zip(legacy_records, summary_texts)):
        expected = hic.summary_to_embedding_text(record["summary"])
        if summary_text != expected:
            raise ValueError(f"BGE-M3 summary text differs from HIC case {ids[index]}.")

    cases_by_id = {case["case_id"]: case for case in cases}
    records_by_id = {case_id: record for case_id, record in zip(canonical_ids, legacy_records)}
    cache_index = {case_id: index for index, case_id in enumerate(ids)}
    selected_ids = ids[:count]
    selected_cases = [cases_by_id[case_id] for case_id in selected_ids]
    return prepare_import_records(
        cases=selected_cases,
        case_ids=selected_ids,
        summaries=[summary_texts[cache_index[case_id]] for case_id in selected_ids],
        corrections=[records_by_id[case_id]["correction"] for case_id in selected_ids],
        embeddings=[vectors[cache_index[case_id]] for case_id in selected_ids],
        source=M3_KB_PATH.name,
    )


def import_snapshots(client, counts=COUNTS, models=(M3_MODEL,)):
    selected_counts = tuple(counts)
    if not selected_counts or any(count not in COUNTS for count in selected_counts):
        raise ValueError("counts must be selected from 1, 10, and 100")
    selected_models = tuple(models)
    if not selected_models or any(model_id != M3_MODEL for model_id in selected_models):
        raise ValueError("only BGE-M3 is supported by the active HIC importer")
    reports = []
    for model_id in selected_models:
        for count in selected_counts:
            name = collection_name_for(model_id, f"train_{count}")
            store = QdrantHicStore(
                client=client,
                model_id=model_id,
                collection_name=name,
            )
            store.ensure_collection()
            records = _m3_snapshot(count)
            inserted = 0
            existing = 0
            for record in records:
                status = store.upsert_completed_case(**record)
                inserted += status == "inserted"
                existing += status == "existing"
            reports.append(
                {
                    "model_id": model_id,
                    "collection": name,
                    "dimension": store.dimension,
                    "requested_count": count,
                    "stored_count": store.count(),
                    "inserted": inserted,
                    "already_present": existing,
                }
            )
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--model", choices=(M3_MODEL,), default=M3_MODEL)
    parser.add_argument("--counts", nargs="+", type=int, choices=COUNTS, default=COUNTS)
    args = parser.parse_args(argv)
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise SystemExit("Install qdrant-client==1.19.1 first.") from None
    client = QdrantClient(url=args.url)
    reports = import_snapshots(client, args.counts, (args.model,))
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
