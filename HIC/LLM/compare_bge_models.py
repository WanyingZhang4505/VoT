"""Controlled BGE-small vs BGE-M3 experiment for the 100-case SocialGood HIC KB."""

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from LLM import hic
from LLM.bge_comparison import (
    prepare_query_text,
    retrieve_prefix,
    token_count_summary,
    validate_case_alignment,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "LLM" / "knowledge_base" / "kb_socialgoods_deepseek_100train.json"
OUTPUT_100_PATH = REPO_ROOT / "LLM" / "vot_hic_output_100train.json"
M3_KB_PATH = REPO_ROOT / "LLM" / "knowledge_base" / "kb_socialgood_bge_m3_100train.json"
PREFIX_SIZES = (1, 10, 100)
SMALL_MODEL_ID = "BAAI/bge-small-en-v1.5"
SMALL_DIMENSION = 384
SMALL_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)
M3_MODEL_ID = "BAAI/bge-m3"
M3_DIMENSION = 1024


class LocalDenseEncoder:
    """CLS-pooled, L2-normalized dense encoder with explicit token accounting."""

    def __init__(self, model_id, expected_dimension, query_instruction=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.query_instruction = query_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=True
        ).to(self.device)
        self.model.eval()

        configured_limits = [
            getattr(self.tokenizer, "model_max_length", None),
            getattr(self.model.config, "max_position_embeddings", None),
        ]
        usable_limits = [
            int(limit)
            for limit in configured_limits
            if isinstance(limit, (int, float)) and 0 < limit < 10_000_000
        ]
        if not usable_limits:
            raise RuntimeError(f"Could not determine the context limit for {model_id}.")
        self.max_length = min(usable_limits)

        model_dimension = getattr(self.model.config, "hidden_size", None)
        if model_dimension != expected_dimension:
            raise RuntimeError(
                f"{model_id} hidden size is {model_dimension}, expected {expected_dimension}."
            )

    def prepare(self, texts, is_query=False):
        instruction = self.query_instruction if is_query else None
        prepared = [prepare_query_text(text, instruction) for text in texts]
        encoded_full = self.tokenizer(
            prepared, add_special_tokens=True, truncation=False
        )["input_ids"]
        return prepared, [len(input_ids) for input_ids in encoded_full]

    def encode(self, texts, is_query=False, batch_size=1):
        if not texts:
            return [], []
        prepared, token_counts = self.prepare(texts, is_query=is_query)
        vectors = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state[:, 0]
                batch_vectors = self.torch.nn.functional.normalize(
                    hidden, p=2, dim=1
                )
            if batch_vectors.ndim != 2 or batch_vectors.shape[1] != self.expected_dimension:
                raise RuntimeError(
                    f"{self.model_id} returned vector shape {tuple(batch_vectors.shape)}."
                )
            if not self.torch.isfinite(batch_vectors).all():
                raise RuntimeError(f"{self.model_id} returned non-finite vectors.")
            vectors.extend(batch_vectors.cpu().tolist())
            completed = min(start + batch_size, len(prepared))
            if completed % 10 == 0 or completed == len(prepared):
                print(f"  {self.model_id}: encoded {completed}/{len(prepared)}")
        return vectors, token_counts


def _atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _new_report_path(output_dir):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"vot_hic_bge_small_vs_m3_controlled_{timestamp}"
    path = output_dir / f"{base}.json"
    suffix = 1
    while path.exists():
        path = output_dir / f"{base}_{suffix}.json"
        suffix += 1
    return path


def _load_inputs():
    if not KB_PATH.is_file() or not OUTPUT_100_PATH.is_file():
        raise FileNotFoundError(
            "The saved 100-case HIC KB and its 100-case run metadata are required."
        )
    with KB_PATH.open(encoding="utf-8") as handle:
        knowledge_base = json.load(handle)
    with OUTPUT_100_PATH.open(encoding="utf-8") as handle:
        old_output = json.load(handle)

    train_cases, test_case, test_truth = hic.load_socialgood_cases(train_count=100)
    case_ids = validate_case_alignment(train_cases, old_output.get("train_case_ids", []))
    if len(knowledge_base) != len(case_ids):
        raise ValueError("The saved 100-case KB count does not match its training IDs.")
    for index, record in enumerate(knowledge_base):
        if not isinstance(record, dict) or not all(
            key in record for key in ("summary", "correction", "embedding", "embedding_model")
        ):
            raise ValueError(f"Saved KB entry {index} is missing required fields.")
        if record["embedding_model"] != SMALL_MODEL_ID:
            raise ValueError(f"Saved KB entry {index} is not embedded with BGE-small.")
        if len(record["embedding"]) != SMALL_DIMENSION:
            raise ValueError(f"Saved BGE-small vector {index} has the wrong dimension.")
        if not isinstance(record["correction"], str) or not record["correction"].strip():
            raise ValueError(f"Saved KB correction {index} is empty or invalid.")
    return knowledge_base, case_ids, test_case, test_truth


def _load_or_build_m3_vectors(knowledge_base, case_ids, summary_texts, encoder):
    if M3_KB_PATH.exists():
        with M3_KB_PATH.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("model_id") != M3_MODEL_ID or cached.get("case_ids") != case_ids:
            raise ValueError("Existing BGE-M3 cache does not match this model/case order.")
        if cached.get("summary_texts") != summary_texts:
            raise ValueError("Existing BGE-M3 cache was built from different summaries.")
        vectors = cached.get("embeddings", [])
        if len(vectors) != len(case_ids) or any(
            len(vector) != M3_DIMENSION for vector in vectors
        ):
            raise ValueError("Existing BGE-M3 cache has incomplete or malformed vectors.")
        _, token_counts = encoder.prepare(summary_texts, is_query=False)
        if cached.get("training_token_counts") != token_counts:
            raise ValueError("Existing BGE-M3 cache token counts do not match this tokenizer.")
        return vectors, token_counts, "reused"

    vectors, token_counts = encoder.encode(
        summary_texts, is_query=False, batch_size=1
    )
    if len(vectors) != len(case_ids):
        raise RuntimeError("BGE-M3 did not return one vector per training summary.")
    cache = {
        "dataset": "SocialGood",
        "model_id": M3_MODEL_ID,
        "pooling": "CLS token from last_hidden_state, then L2 normalization",
        "max_length": encoder.max_length,
        "case_ids": case_ids,
        "summary_texts": summary_texts,
        "training_token_counts": token_counts,
        "embeddings": vectors,
    }
    _atomic_write_json(M3_KB_PATH, cache)
    return vectors, token_counts, "created"


def run_comparison(output_path=None):
    knowledge_base, case_ids, test_case, test_truth = _load_inputs()
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before running the comparison.")

    output_dir = Path(__file__).resolve().parent
    report_path = Path(output_path) if output_path else _new_report_path(output_dir)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {report_path}")

    summary_texts = [hic.summary_to_embedding_text(record["summary"]) for record in knowledge_base]
    small_vectors = [record["embedding"] for record in knowledge_base]
    small_ids = case_ids
    if any(not all(math.isfinite(float(value)) for value in vector) for vector in small_vectors):
        raise ValueError("Saved BGE-small KB contains non-finite embedding values.")

    template_path = hic.TEMPLATE_DIR / f"{hic.industry}.json"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"A saved HIC template is required to keep this run within the planned API-call budget: {template_path}"
        )
    with template_path.open(encoding="utf-8") as handle:
        template = json.load(handle)
    if not isinstance(template, dict) or not template:
        raise ValueError(f"Saved HIC template is empty or invalid: {template_path}")
    hic.LLM_CALL_COUNT = 0
    test_summary = hic.summarize(template, test_case["series"], test_case["news"])
    if not isinstance(test_summary, dict) or "_raw" in test_summary:
        raise ValueError("DeepSeek did not return a structured test summary.")

    report = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_summary": test_summary,
        "logical_deepseek_calls": hic.LLM_CALL_COUNT,
    }
    _atomic_write_json(report_path, report)

    try:
        small_encoder = LocalDenseEncoder(
            SMALL_MODEL_ID,
            SMALL_DIMENSION,
            query_instruction=SMALL_QUERY_INSTRUCTION,
        )
        small_query, small_query_counts = small_encoder.encode(
            [hic.summary_to_embedding_text(test_summary)], is_query=True
        )
        _, small_train_counts = small_encoder.prepare(summary_texts, is_query=False)
        small_max_length = small_encoder.max_length
        del small_encoder

        m3_encoder = LocalDenseEncoder(M3_MODEL_ID, M3_DIMENSION)
        m3_vectors, m3_token_counts, m3_cache_state = _load_or_build_m3_vectors(
            knowledge_base, case_ids, summary_texts, m3_encoder
        )
        m3_query, m3_query_counts = m3_encoder.encode(
            [hic.summary_to_embedding_text(test_summary)], is_query=True
        )
        m3_max_length = m3_encoder.max_length
        del m3_encoder
    except Exception as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        report["logical_deepseek_calls"] = hic.LLM_CALL_COUNT
        _atomic_write_json(report_path, report)
        raise
    baseline = hic.forecast_metrics(
        hic.persistence_forecast(test_case["series"], len(test_truth)), test_truth
    )
    report = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SocialGood",
        "protocol": {
            "training_corpus": str(KB_PATH),
            "prefix_sizes": list(PREFIX_SIZES),
            "same_test_summary_for_all_conditions": True,
            "same_test_history_for_all_conditions": True,
            "same_saved_100_case_summaries_and_corrections": True,
            "forecast_temperature": 0.0,
            "summary_temperature": 0.3,
            "deepseek_calls_planned": 7,
            "truth_excluded_from_all_prompts": True,
            "notes": [
                "The old 1/10/100 HIC artifacts were generated from different summaries; this run uses nested prefixes of the saved 100-case KB.",
                "Retrieval cosine and rank are descriptive; no judged relevant-case labels exist, so retrieval accuracy is not measured.",
                "Forecast metrics use one fixed 12-month holdout and are exploratory, not evidence of broad generalization.",
            ],
        },
        "training_case_ids": case_ids,
        "test_case": {
            key: test_case[key]
            for key in ("case_id", "history_start", "history_end", "series")
        },
        "test_summary": test_summary,
        "embedding_models": {
            SMALL_MODEL_ID: {
                "dimension": SMALL_DIMENSION,
                "pooling": "CLS token from last_hidden_state, then L2 normalization",
                "query_instruction": SMALL_QUERY_INSTRUCTION,
                "max_length": small_max_length,
                "training_token_counts": token_count_summary(small_train_counts),
                "training_inputs_over_limit": sum(
                    count > small_max_length for count in small_train_counts
                ),
                "query_untruncated_tokens": small_query_counts[0],
                "query_truncated": small_query_counts[0] > small_max_length,
                "training_vectors_source": "existing saved 100-case BGE-small KB",
            },
            M3_MODEL_ID: {
                "dimension": M3_DIMENSION,
                "pooling": "CLS token from last_hidden_state, then L2 normalization",
                "query_instruction": None,
                "max_length": m3_max_length,
                "training_token_counts": token_count_summary(m3_token_counts),
                "training_inputs_over_limit": sum(
                    count > m3_max_length for count in m3_token_counts
                ),
                "query_untruncated_tokens": m3_query_counts[0],
                "query_truncated": m3_query_counts[0] > m3_max_length,
                "training_vectors_cache": str(M3_KB_PATH),
                "training_vectors_cache_state": m3_cache_state,
            },
        },
        "software": {
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        },
        "persistence_baseline": baseline,
        "results": [],
        "test_ground_truth_posthoc_only": test_truth,
        "logical_deepseek_calls": 1,
        "logical_call_note": "Counts call_llm invocations; transport retries may make actual HTTP attempts higher.",
    }
    _atomic_write_json(report_path, report)

    model_candidates = (
        (SMALL_MODEL_ID, small_query[0], small_vectors, small_ids),
        (M3_MODEL_ID, m3_query[0], m3_vectors, case_ids),
    )
    try:
        for model_id, query_vector, train_vectors, ids in model_candidates:
            for prefix_size in PREFIX_SIZES:
                retrieval = retrieve_prefix(
                    query_vector, train_vectors, ids, prefix_size
                )
                selected = knowledge_base[retrieval["selected_index"]]
                prompt = hic.build_reasoning_prompt(
                    test_summary,
                    test_case["series"],
                    selected.get("correction"),
                )
                prediction = hic.extract_json(
                    hic.call_llm(prompt, temperature=0.0)
                )
                if not isinstance(prediction, dict) or "prediction" not in prediction:
                    raise ValueError(
                        f"DeepSeek returned no forecast for {model_id} with prefix {prefix_size}."
                    )
                metrics = hic.forecast_metrics(prediction["prediction"], test_truth)
                mae_relative_improvement = (
                    (baseline["mae"] - metrics["mae"]) / baseline["mae"]
                    if baseline["mae"]
                    else None
                )
                record = {
                    "model_id": model_id,
                    "prefix_size": prefix_size,
                    "retrieval": retrieval,
                    "selected_correction": selected["correction"],
                    "prediction": prediction,
                    "metrics": metrics,
                    "mae_relative_improvement_vs_persistence": mae_relative_improvement,
                }
                report["results"].append(record)
                report["logical_deepseek_calls"] = hic.LLM_CALL_COUNT
                _atomic_write_json(report_path, report)
                print(
                    f"{model_id} / prefix={prefix_size}: "
                    f"case={retrieval['selected_case_id']} "
                    f"cosine={retrieval['similarity']:.4f} "
                    f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}"
                )
    except Exception as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        report["logical_deepseek_calls"] = hic.LLM_CALL_COUNT
        _atomic_write_json(report_path, report)
        raise

    report["status"] = "complete"
    report["logical_deepseek_calls"] = hic.LLM_CALL_COUNT
    _atomic_write_json(report_path, report)
    print(f"Saved controlled comparison -> {report_path}")
    print(f"Logical DeepSeek calls: {hic.LLM_CALL_COUNT}; network retries may increase HTTP attempts.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BGE-small and BGE-M3 on the same nested 100-case HIC corpus."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new JSON report path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    run_comparison(args.output)
