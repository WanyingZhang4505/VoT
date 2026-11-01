"""Build truth-free forecast-behavior metadata for an isolated HIC collection."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from LLM.hic_grid_runner import _hash_json
from LLM.hic_vnext import (
    build_forecast_behavior_profile,
    forecast_behavior_similarity,
)


PROFILE_VERSION = "forecast_behavior_profile_v1"


def profile_key(case_id, dataset, horizon):
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be non-empty text")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be non-empty text")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return (case_id, dataset.strip().casefold(), horizon)


def _record_at_pointer(checkpoint, pointer, *, case_id):
    value = checkpoint
    if not isinstance(pointer, list) or not pointer:
        raise ValueError(f"{case_id}: record_pointer must be a non-empty list")
    for key in pointer:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{case_id}: saved record pointer is invalid at {key!r}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"{case_id}: saved record is not an object")
    return value


def build_profile_map(combined_records_path, phase_directory):
    """Rebuild profiles from saved pre-target history and baseline forecasts.

    No truth, correction text, or evaluation score enters the profile.
    """
    index_path = Path(combined_records_path).resolve()
    phase_root = Path(phase_directory).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    configurations = index.get("configurations")
    if not isinstance(configurations, dict) or not configurations:
        raise ValueError("combined record index has no configurations")

    file_cache = {}
    profiles = {}
    for config_key, config in configurations.items():
        dataset = config.get("dataset")
        horizon = config.get("horizon")
        records = config.get("records")
        if not isinstance(records, dict):
            raise ValueError(f"{config_key}: records must be an object")
        for case_id, reference in records.items():
            if not isinstance(reference, dict):
                raise ValueError(f"{config_key}/{case_id}: record reference must be an object")
            raw_path = Path(reference.get("path", ""))
            record_path = raw_path if raw_path.is_absolute() else phase_root / raw_path
            record_path = record_path.resolve()
            if not record_path.is_relative_to(phase_root):
                raise ValueError(f"{config_key}/{case_id}: record path escapes phase directory")
            if record_path not in file_cache:
                file_cache[record_path] = json.loads(record_path.read_text(encoding="utf-8"))
            record = _record_at_pointer(
                file_cache[record_path], reference.get("record_pointer"), case_id=case_id
            )
            if record.get("status") != "complete" or record.get("ingested") is not True:
                raise ValueError(f"{config_key}/{case_id}: source memory is not complete and ingested")
            memory = record.get("memory")
            metadata = memory.get("metadata") if isinstance(memory, dict) else None
            history = metadata.get("history_series") if isinstance(metadata, dict) else None
            forecast = record.get("forecast", {}).get("baseline", {})
            baseline = forecast.get("prediction") if isinstance(forecast, dict) else None
            if not isinstance(history, list) or not isinstance(baseline, list):
                raise ValueError(f"{config_key}/{case_id}: saved history or baseline is missing")
            expected_hash = metadata.get("baseline_prediction_sha256")
            if expected_hash and expected_hash != _hash_json(baseline):
                raise ValueError(f"{config_key}/{case_id}: baseline prediction hash mismatch")

            key = profile_key(case_id, dataset, horizon)
            if key in profiles:
                raise ValueError(f"duplicate profile key: {key}")
            profiles[key] = build_forecast_behavior_profile(history, baseline)

    return profiles


def add_behavior_profile_to_payload(payload, profiles, *, profile_version=PROFILE_VERSION):
    """Return a copied payload with its matching truth-free profile attached."""
    if not isinstance(payload, dict):
        raise ValueError("Qdrant payload must be an object")
    if not isinstance(profile_version, str) or not profile_version.strip():
        raise ValueError("profile_version must be non-empty text")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Qdrant payload metadata must be an object")
    key = profile_key(payload.get("case_id"), metadata.get("dataset"), metadata.get("horizon"))
    if key not in profiles:
        raise ValueError(f"no saved profile for Qdrant memory {key}")
    if "baseline_behavior_profile" in metadata or "behavior_profile_version" in metadata:
        raise ValueError(f"Qdrant memory already has behavior profile metadata: {key}")
    profile = profiles[key]
    forecast_behavior_similarity(profile, profile)
    enriched = copy.deepcopy(payload)
    enriched["metadata"]["baseline_behavior_profile"] = dict(profile)
    enriched["metadata"]["behavior_profile_version"] = profile_version
    return enriched


def profiled_payload_matches(actual, expected, *, tolerance=1e-12):
    """Compare copied payloads while allowing JSON float round-trip noise in profiles."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    actual_metadata = actual.get("metadata")
    expected_metadata = expected.get("metadata")
    if not isinstance(actual_metadata, dict) or not isinstance(expected_metadata, dict):
        return False
    actual_profile = actual_metadata.get("baseline_behavior_profile")
    expected_profile = expected_metadata.get("baseline_behavior_profile")
    if not isinstance(actual_profile, dict) or not isinstance(expected_profile, dict):
        return False
    if actual_metadata.get("behavior_profile_version") != expected_metadata.get(
        "behavior_profile_version"
    ):
        return False
    if set(actual_profile) != set(expected_profile):
        return False

    actual_base = copy.deepcopy(actual)
    expected_base = copy.deepcopy(expected)
    for payload in (actual_base, expected_base):
        metadata = payload["metadata"]
        metadata.pop("baseline_behavior_profile", None)
        metadata.pop("behavior_profile_version", None)
    if actual_base != expected_base:
        return False

    for name, expected_value in expected_profile.items():
        actual_value = actual_profile[name]
        if (
            isinstance(actual_value, bool)
            or isinstance(expected_value, bool)
            or not isinstance(actual_value, (int, float))
            or not isinstance(expected_value, (int, float))
            or not math.isfinite(float(actual_value))
            or not math.isfinite(float(expected_value))
            or not math.isclose(
                float(actual_value), float(expected_value), rel_tol=tolerance, abs_tol=tolerance
            )
        ):
            return False
    return True


def _scroll_all(client, collection_name, *, batch_size):
    offset = None
    points = []
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(page)
        if offset is None:
            return points


def _vector_config(client, collection_name):
    info = client.get_collection(collection_name)
    config = info.config.params.vectors
    if isinstance(config, dict):
        if len(config) != 1:
            raise ValueError(f"{collection_name}: expected one vector configuration")
        config = next(iter(config.values()))
    size = getattr(config, "size", None)
    distance = getattr(config, "distance", None)
    if not isinstance(size, int) or size < 1 or distance is None:
        raise ValueError(f"{collection_name}: unsupported vector configuration")
    return size, distance


def clone_profiled_collection(
    client,
    *,
    source_collection,
    target_collection,
    profiles,
    profile_version=PROFILE_VERSION,
    batch_size=64,
):
    """Clone one collection and add profiles without mutating the source.

    A partially created target can be resumed only if every existing point is
    already an exact, profile-enriched copy of its source point.
    """
    from qdrant_client import models

    if not isinstance(source_collection, str) or not source_collection.strip():
        raise ValueError("source_collection must be non-empty text")
    if not isinstance(target_collection, str) or not target_collection.strip():
        raise ValueError("target_collection must be non-empty text")
    if source_collection == target_collection:
        raise ValueError("source and target collections must differ")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profiles must be a non-empty mapping")
    if not client.collection_exists(source_collection):
        raise ValueError(f"source collection does not exist: {source_collection}")

    source_count_before = int(client.count(collection_name=source_collection, exact=True).count)
    if source_count_before != len(profiles):
        raise ValueError(
            f"profile count ({len(profiles)}) does not match source points ({source_count_before})"
        )
    source_points = _scroll_all(client, source_collection, batch_size=batch_size)
    if len(source_points) != source_count_before:
        raise ValueError("source scroll count does not match exact Qdrant count")
    source_ids = [str(point.id) for point in source_points]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source collection contains duplicate point IDs")

    enriched_by_id = {}
    source_payload_by_id = {}
    source_keys = set()
    for point in source_points:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        key = profile_key(payload.get("case_id"), metadata.get("dataset"), metadata.get("horizon"))
        if key in source_keys:
            raise ValueError(f"source collection has duplicate memory key: {key}")
        source_keys.add(key)
        enriched = add_behavior_profile_to_payload(
            payload, profiles, profile_version=profile_version
        )
        vector = point.vector
        if isinstance(vector, dict) or not isinstance(vector, list):
            raise ValueError("source collection must use one unnamed dense vector")
        enriched_by_id[str(point.id)] = (vector, enriched)
        source_payload_by_id[str(point.id)] = copy.deepcopy(payload)
    if source_keys != set(profiles):
        raise ValueError("Qdrant memory keys do not exactly match saved profile keys")

    target_existed = client.collection_exists(target_collection)
    ids_to_upsert = set(enriched_by_id)
    vector_size, distance = _vector_config(client, source_collection)
    if not target_existed:
        client.create_collection(
            collection_name=target_collection,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
    else:
        if _vector_config(client, target_collection) != (vector_size, distance):
            raise ValueError("existing target vector configuration differs from source")
        existing = _scroll_all(client, target_collection, batch_size=batch_size)
        profiled_ids = set()
        for point in existing:
            point_id = str(point.id)
            expected = enriched_by_id.get(point_id)
            if expected is None:
                raise ValueError("existing target contains a point absent from source")
            vector, payload = expected
            if point.vector != vector:
                raise ValueError("existing target contains a mismatched point; refusing resume")
            if profiled_payload_matches(point.payload, payload):
                profiled_ids.add(point_id)
                continue
            if point.payload == source_payload_by_id[point_id]:
                continue
            raise ValueError("existing target contains a mismatched point; refusing resume")
        ids_to_upsert.difference_update(profiled_ids)
    target_already_complete = target_existed and not ids_to_upsert

    if ids_to_upsert:
        for start in range(0, len(source_points), batch_size):
            points = []
            for source_point in source_points[start:start + batch_size]:
                point_id = str(source_point.id)
                if point_id not in ids_to_upsert:
                    continue
                vector, payload = enriched_by_id[point_id]
                points.append(models.PointStruct(id=source_point.id, vector=vector, payload=payload))
            if points:
                client.upsert(collection_name=target_collection, points=points, wait=True)

    target_points = _scroll_all(client, target_collection, batch_size=batch_size)
    if len(target_points) != source_count_before:
        raise ValueError("target point count does not match source after copy")
    for point in target_points:
        expected = enriched_by_id.get(str(point.id))
        if (
            expected is None
            or point.vector != expected[0]
            or not profiled_payload_matches(point.payload, expected[1])
        ):
            raise ValueError("target verification failed for vector or payload")
    source_after = _scroll_all(client, source_collection, batch_size=batch_size)
    if len(source_after) != source_count_before:
        raise ValueError("source point count changed during migration")
    source_before_by_id = {str(point.id): point for point in source_points}
    for point in source_after:
        before = source_before_by_id.get(str(point.id))
        if before is None or point.vector != before.vector or point.payload != before.payload:
            raise ValueError("source collection changed during migration")

    return {
        "source_collection": source_collection,
        "target_collection": target_collection,
        "source_points_before": source_count_before,
        "target_points_after": len(target_points),
        "points_copied": len(target_points),
        "profiles_attached": len(target_points),
        "points_upserted": len(ids_to_upsert),
        "source_unchanged": True,
        "vectors_and_payloads_verified": True,
        "resumed_existing_target": target_existed,
        "target_already_complete": target_already_complete,
        "upserts_skipped": target_already_complete,
        "profile_version": profile_version,
    }
