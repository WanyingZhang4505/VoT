"""Load hash-verified completed fit memories for an isolated HIC run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pointer_value(root, pointer, *, case_id):
    value = root
    if not isinstance(pointer, list) or not pointer:
        raise ValueError(f"{case_id}: invalid record pointer")
    for key in pointer:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{case_id}: record pointer is missing {key!r}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"{case_id}: pointed record is not an object")
    return value


def load_frozen_seed_memories(
    index_path,
    config_key,
    *,
    dataset,
    horizon,
    fit_cases,
    memory_version,
    expected_source_sha256,
    expected_total=575,
    expected_configuration_count=16,
):
    """Return completed source memories for one config, bound to a frozen index.

    The index and each referenced checkpoint record are verified before use.
    Fit truth is used only to validate the saved seed and its target length; it
    is not copied into the returned memory or its Qdrant metadata.
    """
    if not isinstance(memory_version, str) or not memory_version.strip():
        raise ValueError("memory_version must be non-empty text")
    index_path = Path(index_path).resolve(strict=True)
    raw_index = index_path.read_bytes()
    index = json.loads(raw_index.decode("utf-8-sig"))
    configurations = index.get("configurations")
    if (
        index.get("total") != expected_total
        or index.get("completed") != expected_total
        or not isinstance(configurations, dict)
        or len(configurations) != expected_configuration_count
        or sum(len((row.get("records") or {})) for row in configurations.values()) != expected_total
    ):
        raise ValueError("frozen source index does not match its expected completed scope")
    config = configurations.get(config_key)
    if (
        not isinstance(config, dict)
        or config.get("dataset") != dataset
        or config.get("horizon") != horizon
        or not isinstance(config.get("records"), dict)
    ):
        raise ValueError(f"frozen source index has no matching config: {config_key}")
    all_cases = [row.get("case_id") for row in fit_cases]
    if len(all_cases) != len(set(all_cases)):
        raise ValueError("fit case list contains duplicate IDs")
    cases = {
        row.get("case_id"): row
        for row in fit_cases
        if row.get("news_quality") == "valid"
    }
    references = config["records"]
    if set(cases) != set(references):
        raise ValueError(f"eligible fit query IDs differ from frozen source records for {config_key}")

    phase_root = index_path.parent.resolve()
    file_cache = {}
    source_sha256 = hashlib.sha256(raw_index).hexdigest()
    memories = {}
    for case_id, reference in references.items():
        if not isinstance(reference, dict):
            raise ValueError(f"{case_id}: source reference is not an object")
        raw_path = Path(reference.get("path", ""))
        record_path = raw_path if raw_path.is_absolute() else phase_root / raw_path
        record_path = record_path.resolve(strict=True)
        if not record_path.is_relative_to(phase_root):
            raise ValueError(f"{case_id}: source reference escapes the frozen phase")
        if record_path not in file_cache:
            file_cache[record_path] = json.loads(record_path.read_text(encoding="utf-8"))
        record = _pointer_value(
            file_cache[record_path], reference.get("record_pointer"), case_id=case_id
        )
        if _canonical_sha256(record) != reference.get("record_sha256"):
            raise ValueError(f"{case_id}: frozen source record hash mismatch")
        if record.get("status") != "complete" or record.get("ingested") is not True:
            raise ValueError(f"{case_id}: frozen source memory is incomplete")
        source_memory = record.get("memory")
        metadata = source_memory.get("metadata") if isinstance(source_memory, dict) else None
        case = cases[case_id]
        baseline = (record.get("forecast") or {}).get("baseline") or {}
        prediction = baseline.get("prediction")
        truth = case.get("truth")
        if (
            not isinstance(source_memory, dict)
            or not isinstance(source_memory.get("summary"), dict)
            or not isinstance(source_memory.get("correction"), str)
            or not source_memory["correction"].strip()
            or not isinstance(metadata, dict)
            or source_memory.get("case_id") != case_id
            or metadata.get("case_id") != case_id
            or metadata.get("dataset") != dataset
            or metadata.get("horizon") != horizon
            or metadata.get("split") != "fit"
            or metadata.get("source_sha256") != expected_source_sha256
            or metadata.get("completed") is not True
            or metadata.get("news_quality") != "valid"
            or metadata.get("correction_schema_version") not in {2, 3, 4}
            or not isinstance(truth, list)
            or len(truth) != horizon
            or not isinstance(prediction, list)
            or len(prediction) != horizon
            or metadata.get("target_end") != case.get("target_end")
            or not isinstance(source_memory.get("embedding"), list)
            or len(source_memory["embedding"]) != 1024
        ):
            raise ValueError(f"{case_id}: frozen memory does not match its fit case contract")
        memory = copy.deepcopy(source_memory)
        memory_metadata = memory["metadata"]
        memory_metadata["memory_version"] = memory_version
        memory_metadata["seeded_source_memory"] = True
        memory_metadata["seed_source_memory_sha256"] = source_sha256
        if "baseline_behavior_profile" not in memory_metadata:
            from LLM.hic_vnext import build_forecast_behavior_profile

            history = memory_metadata.get("history_series")
            if not isinstance(history, list):
                raise ValueError(f"{case_id}: behavior profile requires saved history values")
            memory_metadata["baseline_behavior_profile"] = build_forecast_behavior_profile(
                history, prediction
            )
            memory_metadata["behavior_profile_version"] = "forecast_behavior_profile_v1"
        memories[case_id] = memory
    return memories, source_sha256

