"""Download/load local BGE once and smoke-test HIC summary embeddings."""

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM import hic


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def main():
    kb_path = ROOT / "LLM" / "knowledge_base" / "kb_socialgoods_deepseek.json"
    with open(kb_path, encoding="utf-8") as file:
        knowledge_base = json.load(file)
    if not knowledge_base:
        raise RuntimeError(f"No historical HIC cases found in {kb_path}.")

    historical_summary = knowledge_base[0]["summary"]
    query_summary = {
        "predictive_factors": {
            "Temporal Dynamics for prediction": (
                f"Recent unemployment observations: {hic.test_sample['series']}."
            ),
            "External Drivers for prediction": hic.test_sample["news"],
        },
        "forecast_outlook": {"key_driver": hic.test_sample["news"]},
    }

    embedder = hic.get_embedder()
    passage_text = hic.summary_to_embedding_text(historical_summary)
    query_text = hic.summary_to_embedding_text(query_summary)
    passage_vector = embedder.encode([passage_text], is_query=False)[0]
    query_vector = embedder.encode([query_text], is_query=True)[0]
    similarity = sum(left * right for left, right in zip(passage_vector, query_vector))
    selected, retrieval = hic.select_similar_case(query_summary, knowledge_base)

    result = {
        "model": hic.BGE_MODEL_NAME,
        "device": str(embedder.device),
        "dimension": len(passage_vector),
        "historical_summary_tokens": len(embedder.tokenizer.encode(passage_text)),
        "query_summary_tokens_with_instruction": len(
            embedder.tokenizer.encode(hic.BGE_QUERY_INSTRUCTION + query_text)
        ),
        "historical_vector_l2_norm": vector_norm(passage_vector),
        "query_vector_l2_norm": vector_norm(query_vector),
        "cosine_similarity": similarity,
        "retrieval_method": retrieval["method"],
        "selected_index": retrieval["selected_index"],
        "selected_record_matches_only_candidate": selected is knowledge_base[0],
    }
    if result["dimension"] != hic.BGE_EMBEDDING_DIMENSION:
        raise AssertionError(f"Unexpected embedding dimension: {result['dimension']}")
    if not all(math.isfinite(value) for value in (
        result["historical_vector_l2_norm"],
        result["query_vector_l2_norm"],
        result["cosine_similarity"],
    )):
        raise AssertionError("BGE produced a non-finite norm or cosine score.")
    if not math.isclose(result["historical_vector_l2_norm"], 1.0, abs_tol=1e-4):
        raise AssertionError("Historical embedding is not L2-normalized.")
    if not math.isclose(result["query_vector_l2_norm"], 1.0, abs_tol=1e-4):
        raise AssertionError("Query embedding is not L2-normalized.")
    if retrieval["method"] != "bge_small_cosine" or retrieval["selected_index"] != 0:
        raise AssertionError(f"Unexpected local HIC retrieval result: {retrieval}")
    print(json.dumps(result, indent=2))
    print("Note: the repository currently has only one historical case; this score is a smoke test, not retrieval-quality evidence.")


if __name__ == "__main__":
    main()
