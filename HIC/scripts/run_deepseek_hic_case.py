"""Run one truth-blind DeepSeek HIC forecast against a read-only Qdrant library."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_COLLECTION = "deepseek_test_1_imported_20261101"
DEFAULT_EXPECTED_POINTS = 718
PROTOCOL_ID = "deepseek_hic_new_case_direct_nearest_top1_v1"
FORBIDDEN_TRUTH_FIELDS = {
    "truth", "target_values", "observed_targets", "test_truth",
    "future_values", "future_targets", "forecast", "forecasts",
    "prediction", "predictions", "mae", "mse", "rmse", "score", "scores",
    "answer", "answers",
}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    if create_only and path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary.replace(path)


def _truth_field(value: object, *, path: str = "input") -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_TRUTH_FIELDS:
                return f"{path}.{key}"
            found = _truth_field(nested, path=f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _truth_field(nested, path=f"{path}[{index}]")
            if found:
                return found
    return None


def _parse_iso_date(value: object, label: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO date")
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"{label} must start with an ISO date YYYY-MM-DD") from None


def _default_template(dataset: str, target: str, unit: str,
                      frequency: str, horizon: int) -> dict:
    return {
        "dataset_info": {
            "dataset_name": dataset,
            "target_variable": target,
            "target_unit": unit,
            "frequency": frequency,
            "data_modalities": "observed numeric history and user-supplied dated text",
        },
        "predictive_factors": {
            "temporal_patterns": "Describe only patterns visible in the supplied history.",
            "covariates": "Use only supplied covariates known at the forecast cutoff.",
            "text_evidence": "Use only dated text available at the forecast cutoff.",
            "uncertainty": "State unresolved evidence and forecast uncertainty.",
        },
        "forecast_outlook": {
            "expected_direction": "Evidence-based expectation without future claims.",
            "key_signal": "The most relevant visible signal, or state that evidence is insufficient.",
            "confidence": "Low, medium, or high with a short evidence-based reason.",
            "uncertainty": "The main ambiguity in the supplied history and text.",
            "horizon": f"The next {horizon} {frequency} target values.",
        },
    }


def load_case_input(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"case input must be valid UTF-8 JSON: {error}") from None
    if not isinstance(document, dict):
        raise ValueError("case input root must be a JSON object")
    forbidden = _truth_field(document)
    if forbidden:
        raise ValueError(f"truth field is forbidden in prediction input: {forbidden}")
    allowed_root = {"case", "contract", "dataset_description", "template"}
    extra = sorted(set(document) - allowed_root)
    if extra:
        raise ValueError(f"unsupported top-level input fields: {extra[:5]}")

    case = document.get("case")
    contract = document.get("contract")
    description = document.get("dataset_description")
    if not isinstance(case, dict) or not isinstance(contract, dict):
        raise ValueError("case and contract must be JSON objects")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("dataset_description must be non-empty text")
    for key in ("case_id", "dataset"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"case.{key} must be non-empty text")
    for key in ("target", "unit", "frequency"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"contract.{key} must be non-empty text; use 'unknown' when not known")
    horizon = contract.get("horizon")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("contract.horizon must be a positive integer")

    series = case.get("series")
    if not isinstance(series, list) or len(series) < 2:
        raise ValueError("case.series must contain at least two observed history values")
    try:
        normalized_series = [float(value) for value in series]
    except (TypeError, ValueError):
        raise ValueError("case.series must contain finite numeric values") from None
    if not all(math.isfinite(value) for value in normalized_series):
        raise ValueError("case.series must contain finite numeric values")

    history_start = _parse_iso_date(case.get("history_start"), "case.history_start")
    history_end = _parse_iso_date(case.get("history_end"), "case.history_end")
    target_start = _parse_iso_date(case.get("target_start"), "case.target_start")
    target_end = _parse_iso_date(case.get("target_end"), "case.target_end")
    if not history_start <= history_end < target_start <= target_end:
        raise ValueError("case dates must satisfy history_start <= history_end < target_start <= target_end")
    timestamps = case.get("target_timestamps")
    if not isinstance(timestamps, list) or len(timestamps) != horizon:
        raise ValueError("case.target_timestamps must contain exactly contract.horizon dates")
    parsed_timestamps = [_parse_iso_date(value, "case.target_timestamps item")
                         for value in timestamps]
    if (parsed_timestamps[0] < target_start or parsed_timestamps[-1] > target_end
            or any(left >= right for left, right in zip(parsed_timestamps, parsed_timestamps[1:]))):
        raise ValueError("target_timestamps must be increasing dates inside the target period")
    if parsed_timestamps[0] != target_start or parsed_timestamps[-1] != target_end:
        raise ValueError("target_timestamps must begin/end at target_start/target_end")

    normalized_case = dict(case)
    normalized_case["series"] = normalized_series
    normalized_case["frequency"] = contract["frequency"]
    normalized_case.setdefault("news", "")
    normalized_contract = dict(contract)
    normalized_contract["dataset"] = case["dataset"]
    normalized_contract["source_sha256"] = hashlib.sha256(raw).hexdigest()
    template = document.get("template")
    if template is None:
        template = _default_template(case["dataset"], contract["target"],
                                     contract["unit"], contract["frequency"], horizon)
    if not isinstance(template, dict) or not template:
        raise ValueError("template, when supplied, must be a non-empty JSON object")
    return {
        "case": normalized_case,
        "contract": normalized_contract,
        "dataset_description": description.strip(),
        "template": template,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_collection(store, *, expected_points: int) -> dict:
    collection = store.collection_name
    if not store.client.collection_exists(collection):
        raise RuntimeError(f"Qdrant collection {collection!r} does not exist")
    info = store.client.get_collection(collection)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            raise RuntimeError("collection must contain exactly one unnamed vector")
        vectors = next(iter(vectors.values()))
    status = str(info.status).lower()
    count = int(info.points_count)
    dimension = int(vectors.size)
    distance = str(vectors.distance).lower()
    if status != "green" or count != expected_points or dimension != 1024 or distance != "cosine":
        raise RuntimeError(
            f"collection must be green with {expected_points} points and 1024-D Cosine vectors; "
            f"got {status}/{count}/{dimension}/{distance}"
        )
    return {"collection": collection, "status": status, "points_count": count,
            "dimension": dimension, "distance": "Cosine"}


def _preflight(*, collection: str, url: str, expected_points: int) -> dict:
    from LLM.deepseek_strict_tool_provider import preflight_configuration
    from LLM.qdrant_store import QdrantHicStore
    from scripts.run_agent_test6 import _embedder

    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=collection, url=url)
    try:
        qdrant = _validate_collection(store, expected_points=expected_points)
    finally:
        store.client.close()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    embedder = _embedder()
    vector = embedder.encode(["DeepSeek HIC one-case local preflight"], is_query=True)[0]
    if len(vector) != 1024 or not all(math.isfinite(float(value)) for value in vector):
        raise RuntimeError("local BGE-M3 must produce a finite 1024-dimensional vector")
    provider = preflight_configuration()
    provider["forecast_schema"] = "strict function call; exact horizon is defined by case input"
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "ready": bool(provider.get("api_key_configured")),
            "protocol": PROTOCOL_ID, "qdrant": qdrant,
            "embedding": {"model": embedder.model_name, "dimension": len(vector),
                          "offline": True},
            "provider": provider,
            "truth_read": False, "qdrant_writes": 0}


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("run-id may contain only letters, numbers, '_' and '-' (1-80 chars)")
    return value


def new_case_state(run_id: str, case_id: str) -> dict:
    return {"run_id": run_id, "case_id": case_id, "status": "pending",
            "truth_read": False, "qdrant_writes": 0}


def predict(args) -> dict:
    from LLM import hic_agent_validation
    from LLM.deepseek_strict_tool_provider import (
        BACKEND, PROMPT_VERSION, PROVIDER_PROTOCOL, DeepSeekStrictRequestCache,
    )
    from LLM.hic import summary_to_embedding_text
    from LLM.qdrant_store import QdrantHicStore
    from scripts.run_agent_test6 import _embedder

    loaded = load_case_input(args.input)
    run_id = _safe_run_id(args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ))
    run_dir = ROOT / "LLM" / "evaluations" / "DeepSeek HIC New Cases" / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory exists; use its exact inputs with --resume: {run_dir}")
    if args.resume and not run_dir.is_dir():
        raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")

    if args.resume:
        manifest_path = run_dir / "run_manifest.json"
        state_path = run_dir / "state.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != loaded["input_sha256"]:
            raise ValueError("resume input hash differs from the frozen run input")
        if manifest.get("collection") != args.collection or manifest.get("qdrant_url") != args.qdrant_url:
            raise ValueError("resume collection or Qdrant URL differs from the frozen run")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "forecast_complete" and (run_dir / "result.json").is_file():
            return state
    else:
        preflight = _preflight(collection=args.collection, url=args.qdrant_url,
                               expected_points=args.expected_points)
        if not preflight["ready"]:
            raise RuntimeError("preflight failed; configure the API key before prediction")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "provider_calls").mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1, "run_id": run_id, "protocol": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": loaded["case"]["case_id"],
            "input_sha256": loaded["input_sha256"],
            "input_truth_fields_rejected": True,
            "collection": args.collection, "expected_points": args.expected_points,
            "qdrant_url": args.qdrant_url, "retrieval_mode": "direct_nearest_top1",
            "retrieval_collection_write": False,
            "provider": BACKEND, "provider_protocol": PROVIDER_PROTOCOL,
            "prompt_version": PROMPT_VERSION,
            "generation_config": DeepSeekStrictRequestCache(
                run_dir / "provider_calls", run_id=run_id,
                dataset=loaded["contract"]["dataset"],
                horizon=loaded["contract"]["horizon"],
                frequency=loaded["contract"]["frequency"],
                target=loaded["contract"]["target"],
                unit=loaded["contract"]["unit"],
            ).generation_config(),
            "preflight": preflight, "truth_read": False,
        }
        _write_json(run_dir / "run_manifest.json", manifest, create_only=True)
        _write_json(run_dir / "case_input.json", {
            "case": loaded["case"], "contract": loaded["contract"],
            "dataset_description": loaded["dataset_description"],
            "template": loaded["template"],
        }, create_only=True)
        state = new_case_state(run_id, loaded["case"]["case_id"])
        _write_json(run_dir / "state.json", state, create_only=True)

    hic_agent_validation.PROTOCOL = PROTOCOL_ID
    (run_dir / "provider_calls").mkdir(parents=True, exist_ok=True)
    store = QdrantHicStore(model_id="BAAI/bge-m3", collection_name=args.collection,
                           url=args.qdrant_url)
    try:
        collection_info = _validate_collection(store, expected_points=args.expected_points)
        embedder = _embedder()
        contract = loaded["contract"]
        cache = DeepSeekStrictRequestCache(
            run_dir / "provider_calls", run_id=run_id,
            dataset=contract["dataset"], horizon=contract["horizon"],
            frequency=contract["frequency"], target=contract["target"], unit=contract["unit"],
        )

        def persist() -> None:
            _write_json(run_dir / "state.json", state)

        result = hic_agent_validation.advance_agent_case(
            loaded["case"], state, contract=contract,
            template=loaded["template"],
            dataset_description=loaded["dataset_description"],
            cache=cache, store=store,
            embed_summary=lambda summary: embedder.encode(
                [summary_to_embedding_text(summary)], is_query=True
            )[0],
            persist=persist, retrieval_mode="direct_nearest",
        )
        selected = result.get("retrieval_trace", {}).get("selected", [])
        if len(selected) != 1:
            raise RuntimeError("expected exactly one Top-1 correction; library returned none")
        source = store.client.retrieve(
            collection_name=args.collection, ids=[selected[0]["point_id"]],
            with_payload=True, with_vectors=False,
        )
        if len(source) != 1:
            raise RuntimeError("Top-1 Qdrant point could not be retrieved for audit")
        selected[0]["source_metadata"] = (source[0].payload or {}).get("metadata", {})
        for arm in ("baseline", "full_hic"):
            prediction = result[arm].get("prediction")
            if not isinstance(prediction, list) or len(prediction) != contract["horizon"]:
                raise RuntimeError(f"{arm} did not return exactly {contract['horizon']} values")
            if not all(math.isfinite(float(value)) for value in prediction):
                raise RuntimeError(f"{arm} returned non-finite forecast values")
        result["status"] = "forecast_complete"
        result["qdrant_initial_collection"] = collection_info
        result["truth_read"] = False
        result["qdrant_writes"] = 0
        result["provider_protocol"] = PROVIDER_PROTOCOL
        result["prompt_version"] = PROMPT_VERSION
        _write_json(run_dir / "state.json", result)
        _write_json(run_dir / "result.json", result, create_only=True)
        return result
    except Exception as error:
        state["status"] = "technical_failure"
        state["technical_failure"] = {
            "type": type(error).__name__, "message": str(error)[:500],
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "truth_read": False, "qdrant_writes": 0,
        }
        _write_json(run_dir / "state.json", state)
        raise
    finally:
        store.client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="check local Qdrant, BGE-M3, and API-key presence")
    preflight.add_argument("--collection", default=DEFAULT_COLLECTION)
    preflight.add_argument("--expected-points", type=int, default=DEFAULT_EXPECTED_POINTS)
    preflight.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    preflight.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("predict", help="predict one new truth-blind case with A/B outputs")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--collection", default=DEFAULT_COLLECTION)
    run.add_argument("--expected-points", type=int, default=DEFAULT_EXPECTED_POINTS)
    run.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        result = _preflight(collection=args.collection, url=args.qdrant_url,
                            expected_points=args.expected_points)
        _write_json(args.output, result, create_only=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if not result["ready"]:
            raise SystemExit(2)
    else:
        result = predict(args)
        print(json.dumps({
            "run_id": result["run_id"], "case_id": result["case_id"],
            "status": result["status"],
            "baseline_prediction": result["baseline"].get("prediction"),
            "full_hic_prediction": result["full_hic"].get("prediction"),
            "retrieval_trace": result["retrieval_trace"],
            "truth_read": result["truth_read"], "qdrant_writes": result["qdrant_writes"],
        }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
