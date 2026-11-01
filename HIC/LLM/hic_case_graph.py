"""Build a deterministic, cutoff-safe case-memory graph from scored HIC records."""

import math
from datetime import datetime

from LLM.hic_graph_data import validate_scored_case_record


_EDGE_TYPES = ("temporal_predecessor", "same_series", "semantic_similarity")
_POST_SCORE_KEYS = {
    "truth", "target_values", "metrics", "correction", "corrections", "gain_vs_k0"
}
_FEATURE_NAMES = ("history_last", "history_mean", "history_std", "history_min", "history_max", "history_slope")


def _parse_time(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} must be an ISO timestamp") from None


def _contains_post_score_key(value):
    if isinstance(value, dict):
        return any(
            str(key).lower() in _POST_SCORE_KEYS or _contains_post_score_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_post_score_key(item) for item in value)
    return False


def _known_view(record, *, require_scored):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("records must use scored HIC graph schema version 1")
    known = record.get("known_at_forecast")
    if not isinstance(known, dict) or _contains_post_score_key(known):
        raise ValueError("known_at_forecast is missing or contains a post-score field")
    for name in ("case_id", "dataset", "forecast_cutoff", "history_start", "history_end", "target_start", "target_end"):
        if not isinstance(known.get(name), str) or not known[name].strip():
            raise ValueError(f"known_at_forecast must include {name}")
    if not isinstance(known.get("group_path"), list):
        raise ValueError("known_at_forecast.group_path must be a list")
    values = known.get("history_values")
    timestamps = known.get("history_timestamps")
    if not isinstance(values, list) or not values or len(values) != len(timestamps or []):
        raise ValueError("known history values and timestamps must be non-empty and aligned")
    try:
        normalized = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError("history values must be finite numeric data") from None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("history values must be finite numeric data")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("history timestamps must be a non-empty list")
    parsed_history = [_parse_time(value, "history timestamp") for value in timestamps]
    if parsed_history != sorted(parsed_history) or len(set(parsed_history)) != len(parsed_history):
        raise ValueError("history timestamps must be strictly increasing")
    history_start = _parse_time(known["history_start"], "history_start")
    history_end = _parse_time(known["history_end"], "history_end")
    if parsed_history[0] != history_start or parsed_history[-1] > history_end:
        raise ValueError("history timestamps must start at history_start and end no later than history_end")
    start = _parse_time(known["target_start"], "target_start")
    end = _parse_time(known["target_end"], "target_end")
    if end < start or history_end >= start:
        raise ValueError("case cutoffs are reversed or target begins before history ends")
    if known.get("forecast_cutoff") != known.get("history_end"):
        raise ValueError("forecast_cutoff must equal history_end")
    embedding = known.get("embedding")
    if not isinstance(embedding, dict) or not isinstance(embedding.get("vector"), list) or not embedding["vector"]:
        raise ValueError("embedding vector is required")
    try:
        vector = [float(value) for value in embedding["vector"]]
    except (TypeError, ValueError):
        raise ValueError("embedding vector must contain finite numbers") from None
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite numbers")
    if embedding.get("dimension") != len(vector):
        raise ValueError("embedding dimension does not match vector")
    scored = record.get("revealed_after_forecast")
    if require_scored:
        validate_scored_case_record(record)
        if not isinstance(scored, dict):
            raise ValueError("memory nodes must be completed scored records")
    return known, normalized, vector, start, end


def _series_key(known):
    return known["dataset"], tuple(str(value) for value in known["group_path"])


def _cosine(left, right):
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding vectors must have non-zero norm")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _scalar_features(history):
    mean = sum(history) / len(history)
    variance = sum((value - mean) ** 2 for value in history) / len(history)
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return [history[-1], mean, math.sqrt(variance), min(history), max(history), slope]


def build_case_graph(records, query_record, *, semantic_top_k, same_series_only=True):
    """Return an in-memory graph with edges only from earlier completed cases.

    Embedding vectors are returned in a separate named feature block. Similarity
    edges represent retrieval proximity, never causal relationships.
    """
    if not isinstance(records, (list, tuple)):
        raise ValueError("records must be a list of completed HIC records")
    if not isinstance(semantic_top_k, int) or isinstance(semantic_top_k, bool) or semantic_top_k < 0:
        raise ValueError("semantic_top_k must be a non-negative integer")
    if not isinstance(same_series_only, bool):
        raise ValueError("same_series_only must be a boolean")

    query_known, query_history, query_vector, query_start, _ = _known_view(
        query_record, require_scored=False
    )
    query_id = query_known["case_id"]
    query_series = _series_key(query_known)
    memories = []
    seen = {query_id}
    for record in records:
        known, history, vector, start, end = _known_view(record, require_scored=True)
        case_id = known["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id in graph: {case_id}")
        seen.add(case_id)
        # Strict inequality deliberately excludes equal-boundary targets.
        if end >= query_start:
            continue
        if same_series_only and _series_key(known) != query_series:
            continue
        memories.append({
            "id": case_id,
            "known": known,
            "history": history,
            "vector": vector,
            "start": start,
            "end": end,
            "truth": record["revealed_after_forecast"]["truth"],
        })
    memories.sort(key=lambda item: (item["end"], item["id"]))

    nodes = memories + [{
        "id": query_id,
        "known": query_known,
        "history": query_history,
        "vector": query_vector,
        "start": query_start,
        "end": _parse_time(query_known["target_end"], "query target_end"),
        "truth": None,
    }]
    dimensions = {len(node["vector"]) for node in nodes}
    if len(dimensions) != 1:
        raise ValueError("all graph-node embeddings must have the same dimension")
    node_ids = [node["id"] for node in nodes]
    query_index = len(nodes) - 1
    edges = []

    def add_edge(source, kind, weight):
        gap = query_start - memories[source]["end"]
        if gap.total_seconds() <= 0:
            raise ValueError("an edge to the query must originate before its forecast target")
        edges.append((source, query_index, kind, float(weight), gap.total_seconds() / 86400.0))

    matching_series = [
        index for index, node in enumerate(memories)
        if _series_key(node["known"]) == query_series
    ]
    for index in matching_series:
        add_edge(index, "same_series", 1.0)
    if matching_series:
        predecessor = max(matching_series, key=lambda index: (memories[index]["end"], memories[index]["id"]))
        add_edge(predecessor, "temporal_predecessor", 1.0)

    if semantic_top_k:
        ranked = [
            (_cosine(query_vector, node["vector"]), node["id"], index)
            for index, node in enumerate(memories)
        ]
        ranked.sort(key=lambda row: (-row[0], row[1]))
        for score, _, index in ranked[:semantic_top_k]:
            add_edge(index, "semantic_similarity", score)

    node_labels = [node["truth"] for node in memories] + [None]
    return {
        "node_ids": node_ids,
        "x": [_scalar_features(node["history"]) for node in nodes],
        "feature_names": list(_FEATURE_NAMES),
        "feature_blocks": {"embedding": [node["vector"] for node in nodes]},
        "edge_index": [
            [edge[0] for edge in edges],
            [edge[1] for edge in edges],
        ],
        "edge_type": [edge[2] for edge in edges],
        "edge_weight": [edge[3] for edge in edges],
        "edge_age": [edge[4] for edge in edges],
        "query_index": query_index,
        "eligible_memory_mask": [True] * len(memories) + [False],
        "labels": node_labels,
        "label_mask": [True] * len(memories) + [False],
        "edge_types": list(_EDGE_TYPES),
    }
