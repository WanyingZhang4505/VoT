"""Framework-neutral graph data contract for later GNN/model adapters.

This module defines structural validation and ID-to-index conversion only. It
does not prescribe what a node or relation means, how features are encoded, or
which graph model consumes the data.
"""

import copy
import json
from datetime import datetime, timezone


SCHEMA_VERSION = 1


def _timestamp(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} must be an ISO timestamp") from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _non_empty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _json_value(value, label):
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite JSON-compatible values") from None


def _validate_feature_blocks(blocks, label, *, cutoff):
    if not isinstance(blocks, dict):
        raise ValueError(f"{label} must be a mapping")
    for name, block in blocks.items():
        _non_empty_string(name, f"{label} name")
        if not isinstance(block, dict) or "values" not in block or "available_at" not in block:
            raise ValueError(f"{label}[{name}] must contain values and available_at")
        available_at = _timestamp(block["available_at"], f"{label}[{name}].available_at")
        if available_at > cutoff:
            raise ValueError(f"{label}[{name}] is available after graph as_of")
        _json_value(block["values"], f"{label}[{name}].values")
        _json_value(block, f"{label}[{name}]")


def validate_graph_bundle(bundle):
    """Validate a versioned graph snapshot without imposing domain vocabularies.

    Node and relation names are intentionally open-ended. Each node feature,
    edge, and edge feature carries an availability timestamp so the snapshot's
    ``as_of`` boundary can be audited. Supervision targets are kept separate;
    their availability may be later than the graph snapshot.
    """
    if (
        not isinstance(bundle, dict)
        or type(bundle.get("schema_version")) is not int
        or bundle.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"bundle must use schema_version {SCHEMA_VERSION}")
    graph_id = _non_empty_string(bundle.get("graph_id"), "graph_id")
    cutoff = _timestamp(bundle.get("as_of"), "as_of")
    nodes = bundle.get("nodes")
    edges = bundle.get("edges")
    targets = bundle.get("targets", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    if not isinstance(edges, list):
        raise ValueError("edges must be a list")
    if not isinstance(targets, list):
        raise ValueError("targets must be a list")

    node_ids = set()
    for index, node in enumerate(nodes):
        label = f"nodes[{index}]"
        if not isinstance(node, dict):
            raise ValueError(f"{label} must be an object")
        node_id = _non_empty_string(node.get("id"), f"{label}.id")
        if node_id in node_ids:
            raise ValueError(f"node IDs must be unique: {node_id}")
        node_ids.add(node_id)
        if "type" in node:
            _non_empty_string(node["type"], f"{label}.type")
        _validate_feature_blocks(node.get("features", {}), f"{label}.features", cutoff=cutoff)

    edge_ids = set()
    for index, edge in enumerate(edges):
        label = f"edges[{index}]"
        if not isinstance(edge, dict):
            raise ValueError(f"{label} must be an object")
        if "id" in edge:
            edge_id = _non_empty_string(edge["id"], f"{label}.id")
            if edge_id in edge_ids:
                raise ValueError(f"edge IDs must be unique: {edge_id}")
            edge_ids.add(edge_id)
        source = _non_empty_string(edge.get("source"), f"{label}.source")
        target = _non_empty_string(edge.get("target"), f"{label}.target")
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"{label} references an unknown node")
        _non_empty_string(edge.get("relation"), f"{label}.relation")
        available_at = _timestamp(edge.get("available_at"), f"{label}.available_at")
        if available_at > cutoff:
            raise ValueError(f"{label} is available after graph as_of")
        _validate_feature_blocks(edge.get("features", {}), f"{label}.features", cutoff=cutoff)
        _json_value(edge, label)

    target_keys = set()
    for index, target in enumerate(targets):
        label = f"targets[{index}]"
        if not isinstance(target, dict):
            raise ValueError(f"{label} must be an object")
        target_type = _non_empty_string(target.get("target_type"), f"{label}.target_type")
        target_id = _non_empty_string(target.get("target_id"), f"{label}.target_id")
        task = _non_empty_string(target.get("task"), f"{label}.task")
        if "value" not in target:
            raise ValueError(f"{label} must contain value")
        available_at = _timestamp(target.get("available_at"), f"{label}.available_at")
        if target_type == "node" and target_id not in node_ids:
            raise ValueError(f"{label} references an unknown node")
        if target_type == "edge" and target_id not in edge_ids:
            raise ValueError(f"{label} references an unknown edge")
        if target_type == "graph" and target_id != graph_id:
            raise ValueError(f"{label} must reference graph_id")
        key = (target_type, target_id, task, available_at)
        if key in target_keys:
            raise ValueError(f"duplicate supervision target: {target_type}/{target_id}/{task}")
        target_keys.add(key)
        _json_value(target, label)

    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    _json_value(bundle, "bundle")
    return None


def graph_bundle_from_indexed_components(
    *,
    graph_id,
    as_of,
    node_ids,
    edge_index,
    edge_relations,
    node_types=None,
    node_features=None,
    edge_features=None,
    edge_available_at=None,
    edge_ids=None,
    targets=None,
    metadata=None,
):
    """Adapt an indexed graph payload to the generic ID-based graph contract.

    Feature blocks are passed through unchanged as aligned ``{values,
    available_at}`` records. This function only restores stable endpoint IDs;
    it does not assign domain meanings, encode features, or select a model.
    """
    graph_id = _non_empty_string(graph_id, "graph_id")
    _timestamp(as_of, "as_of")
    if not isinstance(node_ids, (list, tuple)):
        raise ValueError("node_ids must be a list")
    normalized_node_ids = [
        _non_empty_string(node_id, f"node_ids[{index}]")
        for index, node_id in enumerate(node_ids)
    ]
    if len(set(normalized_node_ids)) != len(normalized_node_ids):
        raise ValueError("node_ids must be unique")
    if hasattr(edge_index, "tolist"):
        edge_index = edge_index.tolist()
    if not isinstance(edge_index, (list, tuple)) or len(edge_index) != 2:
        raise ValueError("edge_index must have source and target rows")
    source_indices, target_indices = edge_index
    if not isinstance(source_indices, (list, tuple)) or not isinstance(target_indices, (list, tuple)):
        raise ValueError("edge_index rows must be lists")
    edge_count = len(source_indices)
    if len(target_indices) != edge_count:
        raise ValueError("edge_index source and target rows must align")
    if not isinstance(edge_relations, (list, tuple)) or len(edge_relations) != edge_count:
        raise ValueError("edge_relations must contain one relation per edge")

    if node_types is not None and (not isinstance(node_types, (list, tuple)) or len(node_types) != len(normalized_node_ids)):
        raise ValueError("node_types must contain one value per node")
    if node_features is None:
        node_features = {}
    if not isinstance(node_features, dict):
        raise ValueError("node_features must be a mapping of aligned feature blocks")
    normalized_node_features = {}
    for name, blocks in node_features.items():
        _non_empty_string(name, "node feature block name")
        if not isinstance(blocks, (list, tuple)) or len(blocks) != len(normalized_node_ids):
            raise ValueError(f"node feature block {name} must contain one block per node")
        normalized_node_features[name] = [copy.deepcopy(block) for block in blocks]

    if edge_features is None:
        edge_features = [{} for _ in range(edge_count)]
    if not isinstance(edge_features, (list, tuple)) or len(edge_features) != edge_count:
        raise ValueError("edge_features must contain one mapping per edge")
    if edge_available_at is None:
        if edge_count:
            raise ValueError("edge_available_at must contain one timestamp per edge")
        edge_available_at = []
    if not isinstance(edge_available_at, (list, tuple)) or len(edge_available_at) != edge_count:
        raise ValueError("edge_available_at must contain one timestamp per edge")
    if edge_ids is not None and (not isinstance(edge_ids, (list, tuple)) or len(edge_ids) != edge_count):
        raise ValueError("edge_ids must contain one value per edge")

    nodes = []
    for index, node_id in enumerate(normalized_node_ids):
        node = {"id": node_id, "features": {
            name: blocks[index] for name, blocks in normalized_node_features.items()
        }}
        if node_types is not None and node_types[index] is not None:
            node["type"] = node_types[index]
        nodes.append(node)

    edges = []
    for index, (source, target) in enumerate(zip(source_indices, target_indices)):
        if (
            not isinstance(source, int) or isinstance(source, bool)
            or not isinstance(target, int) or isinstance(target, bool)
        ):
            raise ValueError("edge_index must contain integer node positions")
        if not 0 <= source < len(normalized_node_ids) or not 0 <= target < len(normalized_node_ids):
            raise ValueError("edge_index contains an index outside node_ids")
        edge = {
            "source": normalized_node_ids[source],
            "target": normalized_node_ids[target],
            "relation": edge_relations[index],
            "available_at": edge_available_at[index],
            "features": copy.deepcopy(edge_features[index]),
        }
        if edge_ids is not None and edge_ids[index] is not None:
            edge["id"] = edge_ids[index]
        edges.append(edge)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "as_of": as_of,
        "nodes": nodes,
        "edges": edges,
        "targets": [] if targets is None else copy.deepcopy(targets),
        "metadata": {} if metadata is None else copy.deepcopy(metadata),
    }
    validate_graph_bundle(bundle)
    return bundle


def indexed_graph_view(bundle):
    """Return a stable, framework-neutral sparse-index view of a graph bundle.

    Node rows keep their declared order. Type/relation IDs are assigned from
    sorted names for reproducibility. Raw feature blocks remain separate and
    are not flattened or normalized here.
    """
    validate_graph_bundle(bundle)
    nodes = bundle["nodes"]
    edges = bundle["edges"]
    node_ids = [node["id"] for node in nodes]
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_type_names = sorted({node["type"] for node in nodes if "type" in node})
    node_type_index = {name: index for index, name in enumerate(node_type_names)}
    edge_type_names = sorted({edge["relation"] for edge in edges})
    edge_type_index = {name: index for index, name in enumerate(edge_type_names)}
    feature_names = sorted({
        name
        for node in nodes
        for name in node.get("features", {})
    })

    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": bundle["graph_id"],
        "as_of": bundle["as_of"],
        "node_ids": node_ids,
        "node_index": node_index,
        "node_type_names": node_type_names,
        "node_type_ids": [
            node_type_index.get(node.get("type")) for node in nodes
        ],
        "feature_blocks": {
            name: [copy.deepcopy(node.get("features", {}).get(name)) for node in nodes]
            for name in feature_names
        },
        "edge_ids": [edge.get("id") for edge in edges],
        "edge_index": [
            [node_index[edge["source"]] for edge in edges],
            [node_index[edge["target"]] for edge in edges],
        ],
        "edge_type_names": edge_type_names,
        "edge_type_ids": [edge_type_index[edge["relation"]] for edge in edges],
        "edge_feature_blocks": [
            copy.deepcopy(edge.get("features", {})) for edge in edges
        ],
        "targets": copy.deepcopy(bundle.get("targets", [])),
        "metadata": copy.deepcopy(bundle.get("metadata", {})),
    }


def supervision_available_by(bundle, cutoff):
    """Return only labels whose outcome was observable by ``cutoff``."""
    validate_graph_bundle(bundle)
    threshold = _timestamp(cutoff, "supervision cutoff")
    return [
        copy.deepcopy(target)
        for target in bundle.get("targets", [])
        if _timestamp(target["available_at"], "target.available_at") <= threshold
    ]
