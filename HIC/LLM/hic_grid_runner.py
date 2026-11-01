"""Resumable, fixed-fit-memory HIC runs for the native-cadence dev grid.

The module is deliberately provider-agnostic: callers inject their API and
embedding functions. Importing it or executing tests cannot make network calls.
Outer-test data are never requested by the loader used here.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

from LLM import hic
from LLM.hic_full_reasoning_eval import ARMS, build_arm_prompt, aggregate_arm_metrics
from LLM.hic_phase2_eval import (
    VERIFIED_FORECAST_ARMS,
    VERIFIED_PIPELINE_VERSION,
    build_validation_uncertainty,
    fuse_forecasts,
    select_validation_fusion,
)
from LLM.hic_qdrant_eval import _correction_text, _validated_forecast
from LLM.hic_request_cache import _canonical
from LLM.hic_verified_memory import (
    FACTS_SCHEMA_VERSION,
    build_verified_correction_prompt,
    compute_error_facts,
    make_verified_correction_record,
    rank_applicable_memories,
    render_verified_correction,
)
from LLM.hic_vnext import (
    apply_bounded_residual,
    build_bounded_residual_prompt,
    rerank_by_text_then_curve,
    validate_bounded_residual_response,
)
from LLM.hic_evidence_gate import case_evidence_eligibility
from LLM.native_hic_data import load_native_hic_splits
from LLM.native_hic_experiment import make_numeric_baselines


_FORBIDDEN_HOLDOUT_KEYS = {
    "test_case_ids", "outer_test_case_ids", "test_targets", "outer_test_targets",
    "test_windows", "outer_test_windows", "test_truth", "outer_test_truth",
}
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
VERIFIED_ARMS = VERIFIED_FORECAST_ARMS
VALIDATION_TRUTH_POLICY = "deferred_after_forecasts_v1"
STRICT_EVIDENCE_PIPELINE_VERSION = "hic_verified_v8_evidence_gate"
_STRICT_EVIDENCE_FIELDS = (
    "forecast_cutoff_at", "task_contract", "news_evidence", "news_text_sha256", "news",
)


def _hash_json(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    preserve_temporary = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError as error:
                if attempt == 4:
                    preserve_temporary = True
                    raise PermissionError(
                        f"could not replace grid checkpoint after 5 attempts; "
                        f"complete recovery file retained at {temporary}"
                    ) from error
                time.sleep(0.1 * (attempt + 1))
    finally:
        if temporary is not None and temporary.exists() and not preserve_temporary:
            temporary.unlink()


def _finite(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    try:
        output = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain finite numbers") from None
    if length is not None and len(output) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    if not all(math.isfinite(value) for value in output):
        raise ValueError(f"{label} must contain finite numbers")
    return output


def _validate_splits(splits, dataset, horizon, *, defer_validation_truth=False):
    if not isinstance(splits, dict) or splits.get("dataset") != dataset or splits.get("horizon") != horizon:
        raise ValueError("development split dataset/horizon does not match the requested configuration")
    if splits.get("outer_test_deferred") is not True or splits.get("test_windows"):
        raise ValueError("outer-test data must remain deferred and absent from the development split")
    fit_cases = splits.get("fit_cases")
    validation_windows = splits.get("validation_windows")
    if not isinstance(fit_cases, list) or not fit_cases or not isinstance(validation_windows, list) or not validation_windows:
        raise ValueError("configuration requires non-empty fit and validation windows")
    fit_ids = [case.get("case_id") for case in fit_cases if isinstance(case, dict)]
    validation_ids = [item.get("case", {}).get("case_id") for item in validation_windows if isinstance(item, dict)]
    if len(fit_ids) != len(fit_cases) or len(set(fit_ids)) != len(fit_ids) or any(not isinstance(v, str) or not v for v in fit_ids):
        raise ValueError("fit cases need unique non-empty case IDs")
    if len(validation_ids) != len(validation_windows) or len(set(validation_ids)) != len(validation_ids) or any(not isinstance(v, str) or not v for v in validation_ids):
        raise ValueError("validation windows need unique non-empty case IDs")
    if set(fit_ids) & set(validation_ids):
        raise ValueError("fit and validation case IDs must be disjoint")
    for item in validation_windows:
        case = item.get("case")
        if not isinstance(case, dict) or "truth" in case or "target_values" in case:
            raise ValueError("validation truth must be held outside the forecast input case")
        _finite(case.get("series"), f"history for {case['case_id']}")
        if not isinstance(case.get("news"), str):
            raise ValueError(f"news for {case['case_id']} must be text")
        if defer_validation_truth:
            if "truth" in item:
                raise ValueError(
                    f"validation truth for {case['case_id']} must be deferred from forecast inputs"
                )
            timestamps = item.get("target_timestamps")
            if (
                not isinstance(timestamps, list)
                or len(timestamps) != horizon
                or not all(isinstance(value, str) and value for value in timestamps)
            ):
                raise ValueError(
                    f"target timestamps for {case['case_id']} must contain {horizon} values"
                )
        elif "truth" not in item:
            raise ValueError(f"validation truth is missing for {case['case_id']}")
    for case in fit_cases:
        if not isinstance(case.get("news"), str):
            raise ValueError(f"news for {case.get('case_id')} must be text")
        _finite(case.get("series"), f"history for {case['case_id']}")
        _finite(case.get("truth"), f"fit target for {case['case_id']}", length=horizon)
    return fit_cases, validation_windows, fit_ids, validation_ids


def _case_fingerprint(case, *, include_truth):
    fields = (
        "case_id", "dataset", "frequency", "history_start", "history_end",
        "target_start", "target_end", "series", "news", "news_quality", "news_quality_detail", "covariates",
    )
    output = {key: case[key] for key in fields if key in case}
    if include_truth:
        output["truth"] = case["truth"]
    return output


def _config_fingerprint(*, dataset, horizon, run_id, template, description, source_sha256, fit_cases, validation_windows, boundaries, pipeline_version="hic_v1", exposure_ledger_sha256=None, forecast_arms=None):
    payload = {
        "pipeline_version": pipeline_version,
        "dataset": dataset,
        "horizon": horizon,
        "run_id": run_id,
        "template": template,
        "description": description,
        "source_sha256": source_sha256,
        "boundaries": boundaries,
        "fit_cases": [_case_fingerprint(case, include_truth=True) for case in fit_cases],
        "validation_inputs": [_case_fingerprint(item["case"], include_truth=False) for item in validation_windows],
    }
    if exposure_ledger_sha256 is not None:
        payload["exposure_ledger_sha256"] = exposure_ledger_sha256
    if forecast_arms is not None:
        payload["forecast_arms"] = list(forecast_arms)
    return _hash_json(payload)


def _load_external_exposure_ledger(ledger_path=None):
    """Load the project's known external target exposures and bind them by content hash."""
    path = Path(ledger_path) if ledger_path is not None else (
        Path(__file__).resolve().parents[1]
        / "data"
        / "external_reference_exposure_ledger_20261019.json"
    )
    raw = path.read_bytes()
    try:
        ledger = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"external exposure ledger is not valid UTF-8 JSON: {path}") from error
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise ValueError("external exposure ledger must use schema_version 1")
    incident_id = ledger.get("incident_id")
    rows = ledger.get("overlapping_planned_validation_windows")
    if not isinstance(incident_id, str) or not incident_id.strip() or not isinstance(rows, list):
        raise ValueError("external exposure ledger is missing its incident ID or window list")

    exposures = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("external exposure ledger entries must be objects")
        dataset, horizon, case_id = row.get("dataset"), row.get("horizon"), row.get("case_id")
        months = row.get("overlap_months")
        if (
            not isinstance(dataset, str) or not dataset.strip()
            or type(horizon) is not int or horizon <= 0
            or not isinstance(case_id, str) or not case_id.strip()
            or not isinstance(months, list) or not months
            or any(not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month) for month in months)
        ):
            raise ValueError("external exposure ledger contains an invalid window record")
        key = (dataset, horizon, case_id)
        current = exposures.setdefault(key, {"incident_ids": [], "overlap_months": []})
        if incident_id not in current["incident_ids"]:
            current["incident_ids"].append(incident_id)
        current["overlap_months"] = sorted(set(current["overlap_months"]) | set(months))
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "exposures": exposures,
    }


def _match_recorded_exposures(validation_windows, *, dataset, horizon, ledger):
    matched = {}
    for item in validation_windows:
        case = item["case"]
        case_id = case["case_id"]
        record = ledger["exposures"].get((dataset, horizon, case_id))
        if record is None:
            continue
        timestamps = item.get("target_timestamps")
        if not isinstance(timestamps, list) or not timestamps or not all(
            isinstance(value, str) and re.match(r"^\d{4}-\d{2}", value) for value in timestamps
        ):
            raise ValueError(f"exposed validation window {case_id} has no auditable target timestamps")
        target_months = {value[:7] for value in timestamps}
        if not set(record["overlap_months"]).issubset(target_months):
            raise ValueError(f"exposure months in ledger do not match target timestamps for {case_id}")
        matched[case_id] = {
            "incident_ids": list(record["incident_ids"]),
            "overlap_months": list(record["overlap_months"]),
        }
    return matched


def _validation_partition(validation_windows, *, recorded_exposures=None, exposure_ledger_sha256=None):
    """Split chronological dev windows once: earlier half selects, later half confirms."""
    recorded_exposures = recorded_exposures or {}
    ordered = []
    for item in validation_windows:
        case = item["case"]
        case_id = case["case_id"]
        raw_start = case.get("target_start")
        if not isinstance(raw_start, str) or not raw_start.strip():
            raise ValueError("verified validation requires an ISO target_start for temporal selection")
        try:
            parsed = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.combine(date.fromisoformat(raw_start), datetime.min.time())
            except ValueError:
                raise ValueError(f"target_start for {case_id} must be an ISO date or datetime") from None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        ordered.append((parsed, case_id))
    ordered.sort(key=lambda row: (row[0], row[1]))
    timestamps = [row[0] for row in ordered]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("verified validation cannot split windows sharing a target_start")
    case_ids = [case_id for _timestamp, case_id in ordered]
    selection_count = (len(case_ids) + 1) // 2
    selection_ids = case_ids[:selection_count]
    confirmation_ids = case_ids[selection_count:]
    recorded_exposure_ids = [case_id for case_id in case_ids if case_id in recorded_exposures]
    recorded_exposure_confirmation_ids = [
        case_id for case_id in confirmation_ids if case_id in recorded_exposures
    ]
    confirmation_ids_after_exposure_exclusion = [
        case_id for case_id in confirmation_ids if case_id not in recorded_exposures
    ]
    return {
        "selection_case_ids": selection_ids,
        "confirmation_case_ids": confirmation_ids,
        "recorded_exposure_case_ids": recorded_exposure_ids,
        "recorded_exposure_confirmation_case_ids": recorded_exposure_confirmation_ids,
        "confirmation_case_ids_after_recorded_exposure_exclusion": confirmation_ids_after_exposure_exclusion,
        "confirmation_after_recorded_exposure_status": (
            "available_after_known_exclusions"
            if confirmation_ids_after_exposure_exclusion else "none_remaining_after_known_exclusions"
        ),
        "exposure_ledger_sha256": exposure_ledger_sha256,
        "confirmation_status": "available" if selection_count < len(case_ids) else "unavailable_single_window",
    }


def _validate_embedding(vector, dimension, case_id):
    values = _finite(vector, f"embedding for {case_id}", length=dimension)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError(f"embedding for {case_id} must be non-zero")
    return values


def _commit_call(
    cache, *, run_id, case_id, stage, prompt, call_json, validator, commit, save,
    retry_unresolved=False, identity_commit=None,
):
    result = cache.invoke(
        f"{run_id}:{case_id}:{stage}", prompt,
        call_json=call_json,
        validator=validator,
        retry_unresolved=retry_unresolved,
    )
    commit(result["response"])
    if identity_commit is not None:
        identity_commit(result["request_identity"], result["response"])
    save()
    cache.mark_persisted(result["request_identity"]["sha256"])


def _summary_identity_metadata(case_state, request_cache):
    """Expose a summary as traceable only when its exact cache response matches."""
    identity = case_state.get("summary_identity")
    if not isinstance(identity, dict):
        return {"summary_identity_status": "unresolved"}
    request_sha256 = identity.get("request_sha256")
    content_sha256 = identity.get("content_sha256")
    records = getattr(request_cache, "records", None)
    record = records.get(request_sha256) if isinstance(records, dict) else None
    cached_identity = record.get("identity") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("status") not in {"validated", "persisted"}
        or not isinstance(cached_identity, dict)
        or cached_identity.get("sha256") != request_sha256
        or record.get("response") != case_state.get("summary")
        or not isinstance(content_sha256, str)
        or _hash_json(case_state.get("summary")) != content_sha256
    ):
        return {"summary_identity_status": "unresolved"}
    return {
        "summary_identity_status": "exact_request_response",
        "summary_request_sha256": request_sha256,
        "summary_content_sha256": content_sha256,
    }


def _strict_evidence_metadata(case):
    """Copy only the evidence fields understood by the strict retrieval gate."""
    return {key: case[key] for key in _STRICT_EVIDENCE_FIELDS if key in case}


def _summary_task_kwargs(case, horizon, *, enabled):
    if not enabled:
        return {}
    contract = hic.build_summary_task_contract(
        target=case.get("target", "OT"),
        unit=case.get("unit", "unknown"),
        frequency=case.get("frequency"),
        horizon=horizon,
    )
    return {
        "target": contract["target"],
        "unit": contract["unit"],
        "frequency": contract["frequency"],
        "horizon": contract["horizon"],
    }


def run_configuration(
    *,
    dataset,
    horizon,
    run_id,
    template,
    dataset_description,
    request_cache,
    call_json,
    embed_summary,
    store,
    checkpoint_path,
    data_dir=None,
    source_sha256,
    split_loader=None,
    validation_truth_loader=None,
    retry_pending=False,
    verified_corrections=False,
    strict_memory_evidence=False,
    exposure_ledger_path=None,
    forecast_arms=None,
    versioned_summary_prompt=False,
    vnext_retrieval=False,
    vnext_bounded_correction=False,
    vnext_alpha=0.5,
    vnext_bound_scale=0.5,
):
    """Run fit-memory construction and paired HIC validation for one grid row.

    Validation targets are not placed in prompts or scored until every paired
    forecast for every validation case has been saved. Validation cases are
    never added to the fit-only vector store.
    """
    if not all(isinstance(value, str) and value.strip() for value in (dataset, run_id, dataset_description)):
        raise ValueError("dataset, run_id, and dataset_description must be non-empty strings")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty JSON object")
    if not isinstance(verified_corrections, bool):
        raise ValueError("verified_corrections must be a boolean")
    if not isinstance(strict_memory_evidence, bool):
        raise ValueError("strict_memory_evidence must be a boolean")
    if not isinstance(versioned_summary_prompt, bool):
        raise ValueError("versioned_summary_prompt must be a boolean")
    if not isinstance(vnext_retrieval, bool):
        raise ValueError("vnext_retrieval must be a boolean")
    if not isinstance(vnext_bounded_correction, bool):
        raise ValueError("vnext_bounded_correction must be a boolean")
    if vnext_bounded_correction and not vnext_retrieval:
        raise ValueError("vNext bounded correction requires vNext history-curve retrieval")
    if vnext_retrieval and not (verified_corrections and versioned_summary_prompt):
        raise ValueError(
            "vNext retrieval requires verified corrections and the versioned summary prompt"
        )
    if vnext_bounded_correction:
        if forecast_arms != ("no_retrieval", "full_reasoning"):
            raise ValueError(
                "vNext bounded correction requires the paired no_retrieval/full_reasoning arms"
            )
        for name, value in (("vnext_alpha", vnext_alpha), ("vnext_bound_scale", vnext_bound_scale)):
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
    if forecast_arms is not None:
        if (
            not verified_corrections
            or not isinstance(forecast_arms, (list, tuple))
            or tuple(forecast_arms) != ("no_retrieval", "full_reasoning")
        ):
            raise ValueError(
                "custom verified forecast profile must be exactly no_retrieval then full_reasoning"
            )
    if strict_memory_evidence and not verified_corrections:
        raise ValueError("strict memory evidence requires verified corrections")
    if validation_truth_loader is not None and not callable(validation_truth_loader):
        raise ValueError("validation_truth_loader must be callable when provided")
    if verified_corrections and validation_truth_loader is None:
        raise ValueError(
            "verified HIC requires deferred validation-truth loading; "
            "pass validation_truth_loader"
        )
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be the 64-character hash recorded by the development plan")
    if not callable(call_json) or not callable(embed_summary):
        raise ValueError("call_json and embed_summary must be callable")
    if not hasattr(request_cache, "invoke") or not hasattr(store, "upsert_completed_case"):
        raise ValueError("request_cache and store do not implement the required HIC interfaces")
    if getattr(store, "model_id", None) != "BAAI/bge-m3":
        raise ValueError("fit memory must use a BAAI/bge-m3 collection")

    loader = split_loader or load_native_hic_splits
    defer_validation_truth = validation_truth_loader is not None
    loader_args = {
        "horizon": horizon,
        "data_dir": data_dir,
        "include_outer_test_targets": False,
    }
    if defer_validation_truth:
        loader_args["include_validation_truth"] = False
    if verified_corrections:
        loader_args["include_empty_news"] = True
    splits = loader(dataset, **loader_args)
    input_fit_cases, validation_windows, _input_fit_ids, validation_ids = _validate_splits(
        splits, dataset, horizon, defer_validation_truth=defer_validation_truth
    )
    if versioned_summary_prompt:
        for case in input_fit_cases + [item["case"] for item in validation_windows]:
            _summary_task_kwargs(case, horizon, enabled=True)
    if strict_memory_evidence:
        ineligible_validation = []
        for item in validation_windows:
            case = item["case"]
            eligible, reasons = case_evidence_eligibility(case)
            if not eligible:
                ineligible_validation.append((case["case_id"], reasons))
        if ineligible_validation:
            reason_counts = {}
            for _case_id, reasons in ineligible_validation:
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reason_counts.items())
            )
            raise ValueError(
                "strict evidence preflight rejected validation cases before provider calls "
                f"({len(ineligible_validation)}/{len(validation_windows)} ineligible)"
                + (f": {reason_summary}" if reason_summary else "")
            )
    fit_exclusions = []
    if verified_corrections:
        fit_cases = []
        for case in input_fit_cases:
            news_quality = case.get(
                "news_quality", "valid" if case["news"].strip() else "no_valid_text"
            )
            if news_quality != "valid" or not case["news"].strip():
                fit_exclusions.append({
                    "case_id": case["case_id"],
                    "reason": "no_valid_news_text",
                })
            else:
                fit_cases.append(case)
        if strict_memory_evidence:
            evidence_eligible_fit_cases = []
            for case in fit_cases:
                eligible, reasons = case_evidence_eligibility(case)
                if eligible:
                    evidence_eligible_fit_cases.append(case)
                else:
                    fit_exclusions.append({
                        "case_id": case["case_id"],
                        "reason": "strict_news_evidence_ineligible",
                        "details": reasons,
                    })
            fit_cases = evidence_eligible_fit_cases
        if not fit_cases:
            if strict_memory_evidence:
                reason_counts = {}
                for exclusion in fit_exclusions:
                    for reason in exclusion.get("details", [exclusion["reason"]]):
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                reason_summary = ", ".join(
                    f"{reason}={count}" for reason, count in sorted(reason_counts.items())
                )
                raise ValueError(
                    "strict evidence preflight rejected all fit cases before provider calls"
                    + (f": {reason_summary}" if reason_summary else "")
                )
            raise ValueError("verified HIC requires at least one fit case with valid news text")
    else:
        fit_cases = input_fit_cases
    fit_ids = [case["case_id"] for case in fit_cases]
    pipeline_version = (
        STRICT_EVIDENCE_PIPELINE_VERSION
        if strict_memory_evidence
        else VERIFIED_PIPELINE_VERSION if verified_corrections else "hic_v1"
    )
    if versioned_summary_prompt:
        pipeline_version = f"{pipeline_version}_{hic.SUMMARY_PROMPT_VERSION}"
    if vnext_retrieval:
        pipeline_version = f"{pipeline_version}_history_curve_rerank_v1"
    if vnext_bounded_correction:
        alpha_token = format(float(vnext_alpha), ".8g").replace(".", "p")
        bound_token = format(float(vnext_bound_scale), ".8g").replace(".", "p")
        pipeline_version = f"{pipeline_version}_bounded_residual_v1_a{alpha_token}_c{bound_token}"
    active_arms = (
        tuple(forecast_arms)
        if forecast_arms is not None
        else VERIFIED_ARMS if verified_corrections else ARMS
    )
    exposure_ledger = _load_external_exposure_ledger(exposure_ledger_path) if verified_corrections else None
    recorded_exposures = (
        _match_recorded_exposures(
            validation_windows, dataset=dataset, horizon=horizon, ledger=exposure_ledger
        )
        if verified_corrections else {}
    )
    validation_partition = (
        _validation_partition(
            validation_windows,
            recorded_exposures=recorded_exposures,
            exposure_ledger_sha256=exposure_ledger["sha256"],
        )
        if verified_corrections else None
    )
    fingerprint = _config_fingerprint(
        dataset=dataset, horizon=horizon, run_id=run_id, template=template,
        description=dataset_description, source_sha256=source_sha256,
        fit_cases=input_fit_cases, validation_windows=validation_windows,
        boundaries=splits.get("boundaries", {}),
        pipeline_version=(
            f"{pipeline_version}_deferred_truth_v1"
            if defer_validation_truth else pipeline_version
        ),
        exposure_ledger_sha256=exposure_ledger["sha256"] if exposure_ledger else None,
        forecast_arms=forecast_arms,
    )
    path = Path(checkpoint_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != 1 or state.get("config_fingerprint") != fingerprint:
            raise ValueError("configuration checkpoint does not match the current source, template, or split")
        if verified_corrections and state.get("validation_partition") != validation_partition:
            raise ValueError("verified checkpoint validation partition does not match the frozen temporal split")
        if verified_corrections and state.get("fit_exclusions") != fit_exclusions:
            raise ValueError("verified checkpoint fit-memory eligibility does not match the source split")
        if forecast_arms is not None and state.get("forecast_arms") != list(active_arms):
            raise ValueError("configuration checkpoint does not match the current forecast-arm profile")
        expected_truth_policy = (
            VALIDATION_TRUTH_POLICY
            if defer_validation_truth else "inline_compatibility"
        )
        saved_truth_policy = state.get(
            "validation_truth_policy", "inline_compatibility"
        )
        if saved_truth_policy != expected_truth_policy:
            raise ValueError(
                "configuration checkpoint uses a different validation-truth policy; "
                "start with a fresh checkpoint for this protocol"
            )
        if defer_validation_truth and any(
            "scoring" in state.get("validation", {}).get(case_id, {})
            for case_id in validation_ids
        ):
            incomplete = [
                case_id for case_id in validation_ids
                if set(state.get("validation", {}).get(case_id, {}).get("forecasts", {}))
                != set(active_arms)
            ]
            if incomplete:
                raise ValueError(
                    "checkpoint contains scored validation truth before all forecast arms "
                    f"were complete: {incomplete}"
                )
    else:
        if store.count() != 0:
            raise ValueError("a new HIC configuration requires an empty isolated fit-memory collection")
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "dataset": dataset,
            "horizon": horizon,
            "config_fingerprint": fingerprint,
            "fit_case_ids": fit_ids,
            "validation_case_ids": validation_ids,
            "fit": {},
            "validation": {},
            "validation_truth_policy": (
                VALIDATION_TRUTH_POLICY
                if defer_validation_truth else "inline_compatibility"
            ),
        }
        if verified_corrections:
            state["validation_partition"] = validation_partition
            state["fit_exclusions"] = fit_exclusions
        if forecast_arms is not None:
            state["forecast_arms"] = list(active_arms)

    def save():
        _atomic_json(path, state)

    expected_ingested = sum(bool(state["fit"].get(case_id, {}).get("ingested")) for case_id in fit_ids)
    current_count = store.count()
    if current_count < expected_ingested or current_count > expected_ingested + 1:
        raise ValueError("fit-memory collection count does not match the resumable checkpoint")

    calls_before = request_cache.logical_calls
    dimension = getattr(store, "dimension", 1024)
    for case in fit_cases:
        case_id = case["case_id"]
        case_state = state["fit"].setdefault(case_id, {})
        if "summary" not in case_state:
            prompt = hic.build_summary_prompt(
                template, case["series"], case["news"],
                dataset_description=dataset_description, covariates=case.get("covariates"),
                news_quality_detail=case.get("news_quality_detail"),
                canonical_json_inputs=verified_corrections,
                **_summary_task_kwargs(case, horizon, enabled=versioned_summary_prompt),
            )
            _commit_call(
                request_cache, run_id=run_id, case_id=case_id, stage="summary",
                prompt=prompt, call_json=call_json,
                validator=lambda value: isinstance(value, dict) and bool(value),
                commit=lambda value, target=case_state: target.__setitem__("summary", value), save=save,
                retry_unresolved=retry_pending,
                identity_commit=lambda identity, response, target=case_state: target.__setitem__(
                    "summary_identity", {
                        "request_sha256": identity["sha256"],
                        "content_sha256": _hash_json(response),
                    }
                ),
            )
        if "forecast" not in case_state:
            prompt = hic.build_reasoning_prompt(
                case_state["summary"], case["series"], covariates=case.get("covariates"),
                dataset_description=dataset_description, horizon=horizon,
                frequency=case.get("frequency"),
                canonical_json_inputs=verified_corrections,
            )
            _commit_call(
                request_cache, run_id=run_id, case_id=case_id, stage="forecast",
                prompt=prompt, call_json=call_json,
                validator=lambda value: _validated_forecast(value, horizon),
                commit=lambda value, target=case_state: target.__setitem__("forecast", _validated_forecast(value, horizon)),
                save=save,
                retry_unresolved=retry_pending,
            )
        if "correction" not in case_state:
            if verified_corrections:
                facts = compute_error_facts(
                    case["series"], case_state["forecast"]["prediction"], case["truth"]
                )
                case_state["verified_facts"] = facts
                prompt = build_verified_correction_prompt(
                    case_state["summary"], case["series"], case["truth"],
                    case_state["forecast"]["prediction"],
                    case_state["forecast"]["reasoning"], facts,
                )

                def correction_validator(value, expected_facts=facts):
                    try:
                        make_verified_correction_record(expected_facts, value)
                    except ValueError:
                        return False
                    return True

                def correction_commit(value, target=case_state, expected_facts=facts):
                    target["correction"] = render_verified_correction(
                        make_verified_correction_record(expected_facts, value)
                    )

                validator, commit = correction_validator, correction_commit
            else:
                prompt = hic.build_correction_prompt(
                    case_state["summary"], case["series"], case["truth"],
                    case_state["forecast"]["prediction"], case_state["forecast"]["reasoning"],
                    covariates=case.get("covariates"), dataset_description=dataset_description,
                )
                validator = lambda value: _correction_text(value) is not None
                commit = lambda value, target=case_state: target.__setitem__("correction", _correction_text(value))
            _commit_call(
                request_cache, run_id=run_id, case_id=case_id, stage="correction",
                prompt=prompt, call_json=call_json, validator=validator, commit=commit,
                save=save,
                retry_unresolved=retry_pending,
            )
        if "embedding" not in case_state:
            vector = _validate_embedding(
                embed_summary(case_state["summary"], is_query=False), dimension, case_id
            )
            case_state["embedding"] = vector
            save()
        if not case_state.get("ingested"):
            memory_metadata = None
            if verified_corrections:
                memory_metadata = {
                    "case_id": case_id,
                    "dataset": dataset,
                    "source_sha256": source_sha256,
                    "target": case.get("target", "OT"),
                    "unit": case.get("unit", "unknown"),
                    "frequency": case.get("frequency"),
                    "horizon": horizon,
                    "target_end": case["target_end"],
                    "split": "fit",
                    "completed": True,
                    "correction_schema_version": 2,
                    "summary_prompt_version": (
                        hic.SUMMARY_PROMPT_VERSION
                        if versioned_summary_prompt else "legacy"
                    ),
                    "news_quality": case.get(
                        "news_quality", "valid" if case["news"].strip() else "no_valid_text"
                    ),
                    "news_quality_detail": case.get("news_quality_detail"),
                    **{
                        key: case[key]
                        for key in (
                            "target_metadata_source",
                            "target_metadata_evidence",
                        )
                        if case.get(key)
                    },
                    **({
                        "history_series": [float(value) for value in case["series"]],
                        "history_series_role": "pre_target_history_v1",
                    } if vnext_retrieval else {}),
                    **(_strict_evidence_metadata(case) if strict_memory_evidence else {}),
                    **_summary_identity_metadata(case_state, request_cache),
                }
            store.upsert_completed_case(
                case_id=case_id,
                summary=case_state["summary"],
                correction=case_state["correction"],
                embedding=case_state["embedding"],
                observed_targets=case["truth"],
                prediction=case_state["forecast"]["prediction"],
                cutoff_date=case["target_end"],
                source=f"training:{dataset}:{run_id}",
                **({"metadata": memory_metadata} if memory_metadata is not None else {}),
            )
            case_state["ingested"] = True
            save()

    if store.count() != len(fit_ids):
        raise RuntimeError("fit-memory collection count differs from the completed fit-case count")

    for item in validation_windows:
        case = item["case"]
        case_id = case["case_id"]
        case_state = state["validation"].setdefault(case_id, {"forecasts": {}})
        if "summary" not in case_state:
            prompt = hic.build_summary_prompt(
                template, case["series"], case["news"],
                dataset_description=dataset_description, covariates=case.get("covariates"),
                news_quality_detail=case.get("news_quality_detail"),
                canonical_json_inputs=verified_corrections,
                **_summary_task_kwargs(case, horizon, enabled=versioned_summary_prompt),
            )
            _commit_call(
                request_cache, run_id=run_id, case_id=case_id, stage="summary",
                prompt=prompt, call_json=call_json,
                validator=lambda value: isinstance(value, dict) and bool(value),
                commit=lambda value, target=case_state: target.__setitem__("summary", value), save=save,
                retry_unresolved=retry_pending,
                identity_commit=lambda identity, response, target=case_state: target.__setitem__(
                    "summary_identity", {
                        "request_sha256": identity["sha256"],
                        "content_sha256": _hash_json(response),
                    }
                ),
            )
        if "retrieval" not in case_state:
            vector = _validate_embedding(
                embed_summary(case_state["summary"], is_query=True), dimension, case_id
            )
            if verified_corrections:
                candidates = store.search_many(vector, limit=len(fit_ids))
                if not isinstance(candidates, list):
                    raise ValueError("search_many must return a ranked list of fit memories")
                query_metadata = {
                    "case_id": case_id,
                    "dataset": dataset,
                    "source_sha256": source_sha256,
                    "target": case.get("target", "OT"),
                    "unit": case.get("unit", "unknown"),
                    "frequency": case.get("frequency"),
                    "horizon": horizon,
                    "history_end": case["history_end"],
                    **({
                        "required_facts_schema_version": FACTS_SCHEMA_VERSION,
                        "summary_prompt_version": hic.SUMMARY_PROMPT_VERSION,
                    } if versioned_summary_prompt else {}),
                    "news_quality": case.get(
                        "news_quality", "valid" if case["news"].strip() else "no_valid_text"
                    ),
                    **{
                        key: case[key]
                        for key in (
                            "target_metadata_source",
                            "target_metadata_evidence",
                        )
                        if case.get(key)
                    },
                    **(_strict_evidence_metadata(case) if strict_memory_evidence else {}),
                }
                if strict_memory_evidence:
                    ranked = rank_applicable_memories(
                        candidates,
                        query_metadata,
                        require_verified_evidence=True,
                        require_horizon_text_consistency=vnext_retrieval,
                    )
                elif vnext_retrieval:
                    ranked = rank_applicable_memories(
                        candidates,
                        query_metadata,
                        require_horizon_text_consistency=True,
                    )
                else:
                    ranked = rank_applicable_memories(candidates, query_metadata)
                for candidate in ranked["eligible"]:
                    if candidate.get("case_id") not in set(fit_ids):
                        raise ValueError("search_many returned a candidate outside fixed fit memory")
                    if not isinstance(candidate.get("correction"), str) or not candidate["correction"].strip():
                        raise ValueError("search_many returned a fit memory without a completed correction")
                    if not isinstance(candidate.get("summary"), (dict, list, str)):
                        raise ValueError("search_many returned a fit memory with an invalid summary")
                if vnext_retrieval and ranked["eligible"]:
                    candidates_with_history = []
                    for candidate in ranked["eligible"]:
                        metadata = candidate.get("metadata")
                        if not isinstance(metadata, dict) or metadata.get("history_series_role") != "pre_target_history_v1":
                            raise ValueError(
                                "vNext retrieval requires a versioned pre-target history payload"
                            )
                        candidates_with_history.append({
                            **candidate,
                            "history_series": metadata.get("history_series"),
                        })
                    reranked = rerank_by_text_then_curve(
                        candidates_with_history,
                        case["series"],
                        text_top_k=5,
                    )
                    ranked["nearest"] = reranked[0] if reranked else None
                    if len(reranked) > 1:
                        ranked["lowest_similarity_control"] = min(
                            reranked,
                            key=lambda candidate: (
                                candidate["combined_score"], candidate.get("case_id", "")
                            ),
                        )
                    case_state["history_curve_rerank"] = {
                        "status": "applied",
                        "text_shortlist_size": min(5, len(ranked["eligible"])),
                        "weights": {"text": 0.5, "curve": 0.5},
                        "selected_case_id": ranked["nearest"]["case_id"],
                        "selected_text_similarity": ranked["nearest"]["text_similarity"],
                        "selected_curve_similarity": ranked["nearest"]["curve_similarity"],
                        "selected_combined_score": ranked["nearest"]["combined_score"],
                    }
                elif vnext_retrieval:
                    case_state["history_curve_rerank"] = {
                        "status": "no_eligible_memory",
                        "text_shortlist_size": 0,
                        "weights": {"text": 0.5, "curve": 0.5},
                    }
                unit_scope_counts = {}
                for candidate in ranked["eligible"]:
                    scope = candidate["unit_compatibility_scope"]
                    unit_scope_counts[scope] = unit_scope_counts.get(scope, 0) + 1
                case_state["retrieval"] = ranked["nearest"]
                case_state["negative_control_retrieval"] = ranked["lowest_similarity_control"]
                case_state["retrieval_audit"] = {
                    "eligible_candidate_count": len(ranked["eligible"]),
                    "excluded_counts": ranked["excluded_counts"],
                    "unit_compatibility_scope_counts": dict(sorted(unit_scope_counts.items())),
                    "fallback_reason": (
                        None if ranked["nearest"] else "no_applicable_completed_fit_memory"
                    ),
                    **({"history_curve_rerank": case_state["history_curve_rerank"]} if vnext_retrieval else {}),
                }
            else:
                retrieval = store.search(vector, limit=1)
                if not isinstance(retrieval, dict) or retrieval.get("case_id") not in set(fit_ids):
                    raise ValueError(f"retrieval for {case_id} did not return a case from fixed fit memory")
                if not isinstance(retrieval.get("correction"), str) or not retrieval["correction"].strip():
                    raise ValueError(f"retrieval for {case_id} returned no completed correction")
                if not isinstance(retrieval.get("score"), (int, float)) or not math.isfinite(float(retrieval["score"])):
                    raise ValueError(f"retrieval for {case_id} returned an invalid similarity score")
                if not isinstance(retrieval.get("summary"), (dict, list, str)):
                    raise ValueError(f"retrieval for {case_id} returned an invalid summary")
                case_state["retrieval"] = retrieval
            save()
        for arm in active_arms:
            if arm in case_state["forecasts"]:
                continue
            if verified_corrections and arm != "no_retrieval":
                selected_retrieval = (
                    case_state["negative_control_retrieval"]
                    if arm == "low_similarity_control"
                    else case_state["retrieval"]
                )
                if selected_retrieval is None:
                    if "no_retrieval" not in case_state["forecasts"]:
                        raise RuntimeError("no-retrieval forecast must be saved before fallback")
                    case_state["forecasts"][arm] = json.loads(
                        json.dumps(case_state["forecasts"]["no_retrieval"])
                    )
                    case_state.setdefault("arm_fallbacks", {})[arm] = "no_applicable_memory"
                    save()
                    continue
            else:
                selected_retrieval = None if arm == "no_retrieval" else case_state["retrieval"]
            if vnext_bounded_correction and arm == "full_reasoning":
                baseline = case_state["forecasts"].get("no_retrieval")
                if not isinstance(baseline, dict):
                    raise RuntimeError("bounded HIC requires the saved No Retrieval forecast first")
                residual_output = case_state.get("residual_outputs", {}).get("full_reasoning")
                if residual_output is None:
                    prompt = build_bounded_residual_prompt(
                        case_state["summary"],
                        case["series"],
                        baseline["prediction"],
                        selected_retrieval,
                        horizon=horizon,
                        frequency=case.get("frequency") or "unknown",
                    )

                    def commit_residual(value, target=case_state):
                        parsed = validate_bounded_residual_response(value, horizon=horizon)
                        target.setdefault("residual_outputs", {})["full_reasoning"] = parsed

                    _commit_call(
                        request_cache,
                        run_id=run_id,
                        case_id=case_id,
                        stage="forecast_full_reasoning_residual_v1",
                        prompt=prompt,
                        call_json=call_json,
                        validator=lambda value: validate_bounded_residual_response(
                            value, horizon=horizon
                        ),
                        commit=commit_residual,
                        save=save,
                        retry_unresolved=retry_pending,
                    )
                    residual_output = case_state["residual_outputs"]["full_reasoning"]
                adjustment = apply_bounded_residual(
                    baseline["prediction"],
                    residual_output["residual"],
                    case["series"],
                    alpha=vnext_alpha,
                    bound_scale=vnext_bound_scale,
                    eligible=selected_retrieval is not None,
                    abstain=residual_output["abstain"],
                )
                case_state["forecasts"][arm] = _validated_forecast({
                    "prediction": adjustment["prediction"],
                    "reasoning": residual_output["reasoning"],
                }, horizon)
                case_state["bounded_correction"] = {
                    "alpha": float(vnext_alpha),
                    "bound_scale": float(vnext_bound_scale),
                    "max_adjustment": max(
                        (abs(value) for value in adjustment["applied_delta"]), default=0.0
                    ),
                    "bound": adjustment["bound"],
                    "fallback_reason": adjustment["fallback_reason"],
                    "raw_residual": residual_output["residual"],
                    "applied_delta": adjustment["applied_delta"],
                    "abstain": residual_output["abstain"],
                }
                save()
                continue
            prompt_arm = "full_reasoning" if arm == "low_similarity_control" else arm
            prompt = build_arm_prompt(
                prompt_arm, summary=case_state["summary"], series=case["series"],
                dataset_description=dataset_description, horizon=horizon,
                frequency=case.get("frequency"), covariates=case.get("covariates"),
                retrieval=selected_retrieval,
                canonical_json_inputs=verified_corrections,
            )
            _commit_call(
                request_cache, run_id=run_id, case_id=case_id, stage=f"forecast_{arm}",
                prompt=prompt, call_json=call_json,
                validator=lambda value: _validated_forecast(value, horizon),
                commit=lambda value, target=case_state, name=arm: target["forecasts"].__setitem__(
                    name, _validated_forecast(value, horizon)
                ),
                save=save,
                retry_unresolved=retry_pending,
            )

        if set(case_state["forecasts"]) != set(active_arms):
            raise RuntimeError(f"paired forecasts are incomplete for {case_id}")

    # Finish every validation forecast before reading any validation target.
    for item in validation_windows:
        case = item["case"]
        case_id = case["case_id"]
        case_state = state["validation"][case_id]
        if set(case_state["forecasts"]) != set(active_arms):
            raise RuntimeError(f"paired forecasts are incomplete for {case_id}")

    if defer_validation_truth:
        truth_records = validation_truth_loader(
            dataset,
            horizon=horizon,
            case_ids=list(validation_ids),
            data_dir=data_dir,
            include_empty_news=verified_corrections,
        )
        if not isinstance(truth_records, dict) or set(truth_records) != set(validation_ids):
            raise ValueError(
                "deferred validation truth IDs must exactly match the frozen validation cases"
            )
        truth_by_id = {}
        input_by_id = {item["case"]["case_id"]: item for item in validation_windows}
        for case_id in validation_ids:
            record = truth_records[case_id]
            if not isinstance(record, dict):
                raise ValueError(f"deferred validation truth for {case_id} must be an object")
            input_timestamps = input_by_id[case_id]["target_timestamps"]
            if record.get("target_timestamps") != input_timestamps:
                raise ValueError(f"validation target timestamps do not match for {case_id}")
            truth_by_id[case_id] = _finite(
                record.get("truth"), f"truth for {case_id}", length=horizon
            )
    else:
        truth_by_id = {
            item["case"]["case_id"]: _finite(
                item["truth"], f"truth for {item['case']['case_id']}", length=horizon
            )
            for item in validation_windows
        }

    # Score only after all forecast arms are checkpointed and deferred labels
    # have passed the exact ID/timestamp checks above.
    for item in validation_windows:
        case = item["case"]
        case_id = case["case_id"]
        case_state = state["validation"][case_id]
        truth = truth_by_id[case_id]
        if "scoring" in case_state:
            if case_state["scoring"].get("truth") != truth:
                raise ValueError(f"validation truth changed since checkpoint for {case_id}")
            continue
        predictions = {arm: case_state["forecasts"][arm]["prediction"] for arm in active_arms}
        numeric = make_numeric_baselines(case["series"], horizon=horizon)
        predictions.update(numeric)
        metrics = {name: hic.forecast_metrics(prediction, truth) for name, prediction in predictions.items()}
        case_state["scoring"] = {"truth": truth, "forecasts": predictions, "metrics": metrics}
        save()

    if store.count() != len(fit_ids):
        raise RuntimeError("validation unexpectedly changed the fixed fit-memory collection")
    scored_windows = []
    for item in validation_windows:
        case = item["case"]
        case_state = state["validation"][case["case_id"]]
        scored = case_state["scoring"]
        retrieval = case_state["retrieval"]
        negative_control_retrieval = case_state.get("negative_control_retrieval")
        scored_windows.append({
            "case_id": case["case_id"],
            **({
                "external_reference_exposure": (
                    {
                        "status": "recorded_exposure",
                        **recorded_exposures[case["case_id"]],
                    }
                    if case["case_id"] in recorded_exposures
                    else {"status": "not_listed_in_current_ledger"}
                )
            } if verified_corrections else {}),
            **({
                "development_role": (
                    "selection"
                    if case["case_id"] in set(validation_partition["selection_case_ids"])
                    else "confirmation"
                )
            } if verified_corrections else {}),
            "case": _case_fingerprint(case, include_truth=False),
            "target_start": case.get("target_start"),
            "retrieval": (
                {
                    "case_id": retrieval["case_id"],
                    "score": float(retrieval["score"]),
                    "summary": retrieval["summary"],
                    "correction": retrieval["correction"],
                } if retrieval else None
            ),
            **({
                "negative_control_retrieval": (
                    {
                        "case_id": negative_control_retrieval["case_id"],
                        "score": float(negative_control_retrieval["score"]),
                    } if negative_control_retrieval else None
                ),
                "retrieval_audit": case_state.get("retrieval_audit"),
                "arm_fallbacks": case_state.get("arm_fallbacks", {}),
            } if verified_corrections else {}),
            **({
                "bounded_correction": case_state.get("bounded_correction"),
                "residual_output": case_state.get("residual_outputs", {}).get("full_reasoning"),
            } if vnext_bounded_correction else {}),
            "truth": scored["truth"],
            "forecasts": scored["forecasts"],
            "metrics": scored["metrics"],
            "scoring": {"truth": scored["truth"], "forecasts": scored["forecasts"]},
        })
    selection_ids = set(validation_partition["selection_case_ids"]) if verified_corrections else None
    selection_rows = (
        [row for row in scored_windows if row["case_id"] in selection_ids]
        if verified_corrections else scored_windows
    )
    confirmation_rows = (
        [row for row in scored_windows if row["case_id"] in set(validation_partition["confirmation_case_ids"])]
        if verified_corrections else []
    )
    confirmation_rows_without_recorded_exposures = (
        [
            row for row in scored_windows
            if row["case_id"] in set(
                validation_partition["confirmation_case_ids_after_recorded_exposure_exclusion"]
            )
        ]
        if verified_corrections else []
    )
    fusion_inputs = [
        {
            "case_id": row["case_id"],
            "truth": row["truth"],
            "full_reasoning": row["forecasts"]["full_reasoning"],
            "numeric_forecasts": {
                name: row["forecasts"][name]
                for name in ("persistence", "history_mean", "recent_mean_3", "linear_trend")
            },
        }
        for row in selection_rows
    ]
    fusion = select_validation_fusion(fusion_inputs)
    if verified_corrections:
        fusion["selection_window_ids"] = [row["case_id"] for row in selection_rows]
    for row in scored_windows:
        row["forecasts"]["validation_blend"] = fuse_forecasts(
            row["forecasts"]["full_reasoning"],
            row["forecasts"][fusion["numeric_baseline"]],
            alpha=fusion["alpha"],
        )
        row["scoring"]["forecasts"]["validation_blend"] = row["forecasts"]["validation_blend"]
    arm_metrics = aggregate_arm_metrics(scored_windows)
    selection_metrics = aggregate_arm_metrics(selection_rows) if verified_corrections else None
    confirmation_metrics = (
        aggregate_arm_metrics(confirmation_rows)
        if verified_corrections and confirmation_rows else None
    )
    confirmation_metrics_excluding_recorded_exposures = (
        aggregate_arm_metrics(confirmation_rows_without_recorded_exposures)
        if verified_corrections and confirmation_rows_without_recorded_exposures else None
    )
    uncertainty = None
    uncertainty_windows = (
        confirmation_rows_without_recorded_exposures
        if verified_corrections
        else scored_windows
    )
    if len(uncertainty_windows) >= 2:
        uncertainty = build_validation_uncertainty(
            uncertainty_windows,
            selected_numeric_baseline=(
                fusion["numeric_baseline"] if forecast_arms is None else None
            ),
            blend_arm="validation_blend" if forecast_arms is None else None,
            selection_case_ids=(
                validation_partition["selection_case_ids"]
                if verified_corrections else None
            ),
        )
    return {
        "schema_version": 1,
        "pipeline_version": pipeline_version,
        "run_id": run_id,
        "dataset": dataset,
        "horizon": horizon,
        "fit_case_ids": fit_ids,
        **({
            "fit_input_case_ids": [case["case_id"] for case in input_fit_cases],
            "fit_input_case_count": len(input_fit_cases),
            "fit_exclusions": fit_exclusions,
        } if verified_corrections else {}),
        "validation_case_ids": validation_ids,
        "fit_memory_count": store.count(),
        "fit_memory_fixed_during_validation": True,
        "outer_test_targets_read": False,
        "validation_truth_deferred": defer_validation_truth,
        **({"strict_memory_evidence": True} if strict_memory_evidence else {}),
        "logical_calls_this_configuration": request_cache.logical_calls - calls_before,
        "arms": list(active_arms),
        **({
            "vnext_bounded_correction": {
                "alpha": float(vnext_alpha),
                "bound_scale": float(vnext_bound_scale),
                "residual_cache_reusable_across_parameters": True,
                "news_provenance_mode": (
                    "strictly_verified" if strict_memory_evidence
                    else "dataset_text_not_independently_source_verified"
                ),
            }
        } if vnext_bounded_correction else ({
            "vnext_retrieval": {
                "news_provenance_mode": (
                    "strictly_verified" if strict_memory_evidence
                    else "dataset_text_not_independently_source_verified"
                ),
                "horizon_text_consistency_required": True,
            }
        } if vnext_retrieval else {})),
        "validation_windows": scored_windows,
        "validation_metrics": arm_metrics,
        **({
            "selection_confirmation_split": validation_partition,
            "selection_metrics": selection_metrics,
            "confirmation_metrics": confirmation_metrics,
            "confirmation_metrics_excluding_recorded_exposures": confirmation_metrics_excluding_recorded_exposures,
            "external_reference_exposure_audit": {
                "ledger_file": exposure_ledger["path"].name,
                "ledger_sha256": exposure_ledger["sha256"],
                "matched_case_ids": [
                    case_id for case_id in validation_ids if case_id in recorded_exposures
                ],
                "absence_from_ledger_is_not_proof_of_no_exposure": True,
            },
        } if verified_corrections else {}),
        "validation_fusion": fusion,
        "validation_uncertainty": uncertainty,
        "validation_uncertainty_sample": {
            "scope": (
                "confirmation_excluding_recorded_exposures"
                if verified_corrections else "all_validation_windows"
            ),
            "window_count": len(uncertainty_windows),
            "case_ids": [row["case_id"] for row in uncertainty_windows],
            "recorded_exposure_windows_excluded": (
                len(confirmation_rows) - len(confirmation_rows_without_recorded_exposures)
                if verified_corrections else 0
            ),
        },
        "uncertainty_note": (
            "Bootstrap intervals with fewer than five windows are descriptive only; all current "
            "single-series intervals are exploratory and do not establish significance."
        ),
    }


def _scan_for_forbidden_fields(value, *, where="plan"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_HOLDOUT_KEYS:
                raise ValueError(f"development plan contains materialized outer-test data at {where}.{key}")
            _scan_for_forbidden_fields(nested, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_for_forbidden_fields(nested, where=f"{where}[{index}]")


def development_jobs(plan):
    """Extract every eligible dev config and reject materialized test targets."""
    if not isinstance(plan, dict) or plan.get("mode") != "full_corpus_development_plan":
        raise ValueError("plan must be a full_corpus_development_plan")
    if plan.get("outer_test_targets_read") is not False:
        raise ValueError("plan does not certify that outer-test targets remain unread")
    rows = plan.get("plans")
    if not isinstance(rows, list):
        raise ValueError("plan rows are missing")
    jobs = []
    seen = set()
    seen_run_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("plan rows must be objects")
        _scan_for_forbidden_fields(row)
        if row.get("status") != "development_ready":
            continue
        dataset, horizon, run_id = row.get("dataset"), row.get("horizon"), row.get("run_id")
        fit_ids, validation_ids = row.get("fit_case_ids"), row.get("validation_case_ids")
        if not isinstance(dataset, str) or not dataset or type(horizon) is not int or horizon <= 0:
            raise ValueError("eligible plan row has an invalid dataset/horizon")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("eligible plan row is missing run_id")
        source_sha256 = row.get("source_sha256")
        if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
            raise ValueError(f"eligible plan row {run_id} is missing a valid source SHA-256")
        if not isinstance(fit_ids, list) or not fit_ids or not isinstance(validation_ids, list) or not validation_ids:
            raise ValueError(f"eligible plan row {run_id} must list fit and validation IDs")
        if any(not isinstance(value, str) or not value for value in fit_ids + validation_ids):
            raise ValueError(f"eligible plan row {run_id} has invalid case IDs")
        if len(set(fit_ids)) != len(fit_ids) or len(set(validation_ids)) != len(validation_ids) or set(fit_ids) & set(validation_ids):
            raise ValueError(f"eligible plan row {run_id} has duplicate/overlapping case IDs")
        key = (dataset, horizon)
        if key in seen:
            raise ValueError(f"duplicate development configuration {dataset}/H{horizon}")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate development run_id {run_id}")
        seen.add(key)
        seen_run_ids.add(run_id)
        jobs.append({
            "job_id": run_id,
            "run_id": run_id,
            "dataset": dataset,
            "horizon": horizon,
            "source_sha256": source_sha256,
            "fit_case_ids": list(fit_ids),
            "validation_case_ids": list(validation_ids),
        })
    if not jobs:
        raise ValueError("plan contains no development-ready configurations")
    return jobs


def run_development_grid(
    plan, *, run_job, checkpoint_path, protocol_version="inline_compatibility",
    execution_job_ids=None, max_completed_jobs=None,
):
    """Run eligible configurations sequentially with an atomic resumable ledger."""
    if not callable(run_job):
        raise ValueError("run_job must be callable")
    if not isinstance(protocol_version, str) or not protocol_version.strip():
        raise ValueError("protocol_version must be a non-empty string")
    jobs = development_jobs(plan)
    all_job_ids = {job["job_id"] for job in jobs}
    if execution_job_ids is None:
        selected_job_id_order = [job["job_id"] for job in jobs]
    else:
        if (
            not isinstance(execution_job_ids, (list, tuple, set, frozenset))
            or any(not isinstance(job_id, str) or not job_id for job_id in execution_job_ids)
            or len(set(execution_job_ids)) != len(execution_job_ids)
        ):
            raise ValueError("execution_job_ids must be a unique sequence of non-empty job IDs")
        selected_job_id_order = list(execution_job_ids)
        selected_job_ids = set(selected_job_id_order)
        if not selected_job_ids or not selected_job_ids.issubset(all_job_ids):
            raise ValueError("execution_job_ids must be a non-empty subset of the frozen plan")
    if max_completed_jobs is not None and (
        type(max_completed_jobs) is not int or max_completed_jobs < 1
    ):
        raise ValueError("max_completed_jobs must be a positive integer when provided")
    manifest_payload = (
        jobs if protocol_version == "inline_compatibility"
        else {"jobs": jobs, "protocol_version": protocol_version}
    )
    manifest_hash = _hash_json(manifest_payload)
    path = Path(checkpoint_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        saved_protocol = state.get("protocol_version", "inline_compatibility")
        if (
            state.get("schema_version") != 1
            or state.get("manifest_sha256") != manifest_hash
            or saved_protocol != protocol_version
        ):
            raise ValueError("grid checkpoint manifest does not match the current development plan")
    else:
        state = {
            "schema_version": 1,
            "manifest_sha256": manifest_hash,
            "protocol_version": protocol_version,
            "outer_test_targets_read": False,
            "jobs": {job["job_id"]: {"status": "planned"} for job in jobs},
        }

    jobs_by_id = {job["job_id"]: job for job in jobs}
    for job_id in selected_job_id_order:
        job = jobs_by_id[job_id]
        record = state["jobs"][job_id]
        if record.get("status") == "completed":
            continue
        completed_count = sum(
            value.get("status") == "completed" for value in state["jobs"].values()
        )
        if max_completed_jobs is not None and completed_count >= max_completed_jobs:
            raise RuntimeError("phase configuration limit reached before starting another job")
        record["status"] = "running"
        record.pop("error_type", None)
        _atomic_json(path, state)
        try:
            output = run_job(job)
            _scan_for_forbidden_fields(output, where=f"output[{job['job_id']}]")
            _canonical(output)
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            _atomic_json(path, state)
            raise
        record["status"] = "completed"
        record["output"] = output
        _atomic_json(path, state)
    return {
        "manifest_sha256": manifest_hash,
        "completed_count": sum(value.get("status") == "completed" for value in state["jobs"].values()),
        "job_count": len(jobs),
        "execution_job_ids": selected_job_id_order,
        "outer_test_targets_read": False,
        "jobs": state["jobs"],
    }
