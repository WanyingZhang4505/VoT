"""Explicit, cutoff-aligned hierarchy snapshots for HIC time-series cases."""

import math
from datetime import datetime


def _timestamp(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} must be an ISO timestamp") from None


def _finite_list(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric list")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite numbers") from None
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _history_features(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return [values[-1], mean, math.sqrt(variance), min(values), max(values)]


def _check_acyclic(parent_by_entity):
    for entity in parent_by_entity:
        visited = set()
        current = entity
        while current is not None:
            if current in visited:
                raise ValueError("hierarchy metadata contains a cycle")
            visited.add(current)
            current = parent_by_entity.get(current)


def build_hierarchy_snapshot(cases_at_cutoff, hierarchy_spec, *, cutoff):
    """Build one graph snapshot from observed entity series and declared parents.

    Each input is ``{"entity_id": ..., "case": ..., "truth": [...]?}``.
    Parent-child edges are created only from ``hierarchy_spec.parent_by_entity``;
    group columns are descriptive and are never interpreted as hierarchy.
    """
    if not isinstance(cases_at_cutoff, (list, tuple)) or not cases_at_cutoff:
        raise ValueError("cases_at_cutoff must be a non-empty list")
    cutoff_time = _timestamp(cutoff, "cutoff")
    if not isinstance(hierarchy_spec, dict) or not isinstance(hierarchy_spec.get("parent_by_entity"), dict):
        raise ValueError("hierarchy_spec must provide an explicit parent_by_entity mapping")
    parent_by_entity = hierarchy_spec["parent_by_entity"]
    if not parent_by_entity:
        raise ValueError("hierarchy_spec.parent_by_entity must not be empty")
    normalized_parents = {}
    for entity, parent in parent_by_entity.items():
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("hierarchy entity IDs must be non-empty strings")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise ValueError(f"parent ID for {entity} must be a non-empty string or null")
        normalized_parents[entity] = parent

    nodes = []
    seen = set()
    expected_dataset = None
    expected_target_start = None
    expected_target_end = None
    expected_frequency = None
    horizon = None
    for row in cases_at_cutoff:
        if not isinstance(row, dict) or not isinstance(row.get("case"), dict):
            raise ValueError("each hierarchy row must contain a case object")
        entity_id = row.get("entity_id")
        case = row["case"]
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("each hierarchy row must have an explicit entity_id")
        series_id = (entity_id, case.get("history_end"))
        if series_id in seen:
            raise ValueError(f"duplicate entity/cutoff pair: {entity_id}/{case.get('history_end')}")
        seen.add(series_id)
        if entity_id not in normalized_parents:
            raise ValueError(f"entity {entity_id} is absent from explicit parent metadata")
        if case.get("history_end") != cutoff:
            raise ValueError("all cases must have the requested shared cutoff")
        target_start_value = case.get("target_start")
        target_end_value = case.get("target_end")
        target_start = _timestamp(target_start_value, "target_start")
        target_end = _timestamp(target_end_value, "target_end")
        if target_end < target_start or cutoff_time >= target_start:
            raise ValueError("case target must follow the common history cutoff")
        dataset = case.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("case dataset must be a non-empty string")
        frequency = case.get("frequency", "unspecified")
        if expected_dataset is None:
            expected_dataset = dataset
            expected_target_start = target_start_value
            expected_target_end = target_end_value
            expected_frequency = frequency
        elif (
            dataset != expected_dataset
            or target_start_value != expected_target_start
            or target_end_value != expected_target_end
            or frequency != expected_frequency
        ):
            raise ValueError("all nodes must share dataset, frequency, and forecast origin")

        history = _finite_list(case.get("series"), f"history for {entity_id}")
        history_dates = case.get("source_dates")
        if not isinstance(history_dates, (list, tuple)) or len(history_dates) != len(history):
            raise ValueError(f"history timestamps and values must align for {entity_id}")
        parsed_dates = [_timestamp(value, "history timestamp") for value in history_dates]
        if parsed_dates != sorted(parsed_dates) or len(set(parsed_dates)) != len(parsed_dates):
            raise ValueError(f"history timestamps must increase for {entity_id}")
        if _timestamp(case.get("history_start"), "history_start") != parsed_dates[0]:
            raise ValueError(f"history_start does not align for {entity_id}")
        if parsed_dates[-1] != cutoff_time:
            raise ValueError(f"history_end does not align with final history timestamp for {entity_id}")

        truth = row.get("truth")
        if truth is not None:
            truth = _finite_list(truth, f"truth for {entity_id}")
            if horizon is None:
                horizon = len(truth)
            elif len(truth) != horizon:
                raise ValueError("all available truth labels must have the same horizon")
        covariates = case.get("covariates") or {}
        if not isinstance(covariates, dict):
            raise ValueError(f"covariates for {entity_id} must be a mapping")
        normalized_covariates = {
            str(name): _finite_list(values, f"covariate {name} for {entity_id}")
            for name, values in covariates.items()
        }
        if any(len(values) != len(history) for values in normalized_covariates.values()):
            raise ValueError(f"covariates must align with history for {entity_id}")
        if any(not math.isfinite(value) for value in _history_features(history)):
            raise ValueError(f"history feature calculation failed for {entity_id}")
        nodes.append({
            "entity_id": entity_id,
            "case": case,
            "history": history,
            "covariates": normalized_covariates,
            "truth": truth,
        })

    entity_ids = {node["entity_id"] for node in nodes}
    for entity, parent in normalized_parents.items():
        if entity not in entity_ids:
            continue
        if parent is not None and parent not in entity_ids:
            raise ValueError(f"parent {parent} for entity {entity} is missing from this snapshot")
    _check_acyclic(normalized_parents)
    nodes.sort(key=lambda node: node["entity_id"])
    indices = {node["entity_id"]: index for index, node in enumerate(nodes)}

    covariate_names = sorted({
        name for node in nodes for name in node["covariates"]
    })
    feature_names = ["history_last", "history_mean", "history_std", "history_min", "history_max"]
    for name in covariate_names:
        feature_names.extend((f"{name}_last", f"{name}_mean", f"{name}_missing_indicator"))
    features = []
    for node in nodes:
        row_features = _history_features(node["history"])
        for name in covariate_names:
            values = node["covariates"].get(name)
            if values is None:
                row_features.extend((0.0, 0.0, 1.0))
            else:
                row_features.extend((values[-1], sum(values) / len(values), 0.0))
        features.append(row_features)

    parent_edges = sorted(
        (indices[parent], indices[entity])
        for entity, parent in normalized_parents.items()
        if parent is not None and entity in indices
    )
    labels = [node["truth"] for node in nodes]
    label_mask = [label is not None for label in labels]
    optional_embeddings = [node["case"].get("embedding") for node in nodes]
    feature_blocks = {}
    if any(value is not None for value in optional_embeddings):
        if not all(isinstance(value, (list, tuple)) and value for value in optional_embeddings):
            raise ValueError("if embeddings are supplied, every node must have one")
        dimensions = {len(value) for value in optional_embeddings}
        if len(dimensions) != 1:
            raise ValueError("hierarchy node embeddings must have the same dimension")
        normalized_embeddings = [_finite_list(value, "node embedding") for value in optional_embeddings]
        feature_blocks["embedding"] = normalized_embeddings

    return {
        "node_ids": [node["entity_id"] for node in nodes],
        "x": features,
        "feature_names": feature_names,
        "feature_blocks": feature_blocks,
        "edge_index": [
            [source for source, _ in parent_edges],
            [target for _, target in parent_edges],
        ],
        "edge_type": ["parent_to_child"] * len(parent_edges),
        "edge_weight": [1.0] * len(parent_edges),
        "cutoff": cutoff,
        "target_start": expected_target_start,
        "target_end": expected_target_end,
        "labels": labels,
        "label_mask": label_mask,
        "aggregation_targets_synthesized": False,
    }
