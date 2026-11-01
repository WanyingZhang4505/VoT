"""Shared, testable helpers for controlled HIC embedding comparisons."""

import math


def prepare_query_text(text, instruction=None):
    """Apply a model-specific retrieval instruction only when one is configured."""
    if not isinstance(text, str):
        raise ValueError("query text must be a string")
    return f"{instruction}{text}" if instruction else text


def token_count_summary(counts):
    """Summarize token lengths with deterministic nearest-rank percentiles."""
    if not counts:
        raise ValueError("counts must contain at least one token length")
    values = sorted(int(value) for value in counts)
    if any(value < 0 for value in values):
        raise ValueError("token lengths cannot be negative")

    def nearest_rank(percentile):
        index = max(0, math.ceil(percentile * len(values)) - 1)
        return values[index]

    middle = len(values) // 2
    median = (
        float(values[middle])
        if len(values) % 2
        else (values[middle - 1] + values[middle]) / 2
    )
    return {
        "count": len(values),
        "min": values[0],
        "median": median,
        "p90_nearest_rank": nearest_rank(0.90),
        "p95_nearest_rank": nearest_rank(0.95),
        "max": values[-1],
    }


def validate_case_alignment(training_cases, stored_case_ids):
    """Ensure a saved KB's row order matches the deterministic source-case order."""
    expected = [case.get("case_id") for case in training_cases]
    if len(expected) != len(stored_case_ids):
        raise ValueError(
            f"Training case count ({len(expected)}) does not match saved IDs "
            f"({len(stored_case_ids)})."
        )
    if expected != list(stored_case_ids):
        raise ValueError("Saved KB case order does not match the canonical training order.")
    return expected


def _unit_vector(values, label):
    if not values:
        raise ValueError(f"{label} must not be empty")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"{label} must have a non-zero finite norm")
    return [value / norm for value in vector]


def retrieve_prefix(query, vectors, case_ids, prefix_size, top_k=5):
    """Retrieve by cosine within exactly the requested leading corpus prefix."""
    if not isinstance(prefix_size, int) or isinstance(prefix_size, bool):
        raise ValueError("prefix_size must be an integer")
    if prefix_size < 1 or prefix_size > len(vectors):
        raise ValueError("prefix_size must fit within the available candidate vectors")
    if len(vectors) != len(case_ids):
        raise ValueError("vectors and case_ids must have equal lengths")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    query_unit = _unit_vector(query, "query")
    scores = []
    for index, vector in enumerate(vectors[:prefix_size]):
        candidate_unit = _unit_vector(vector, f"candidate {index}")
        if len(candidate_unit) != len(query_unit):
            raise ValueError("query and candidate vectors must have equal dimensions")
        score = sum(left * right for left, right in zip(query_unit, candidate_unit))
        scores.append((index, score))

    ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
    best_index, best_score = ranked[0]
    return {
        "method": "dense_cosine_similarity",
        "candidate_prefix_size": prefix_size,
        "selected_index": best_index,
        "selected_case_id": case_ids[best_index],
        "similarity": best_score,
        "top_matches": [
            {"index": index, "case_id": case_ids[index], "similarity": score}
            for index, score in ranked[:top_k]
        ],
    }
