"""Build and execute a dry-run-first, fit-only HIC vNext batch."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from LLM.qdrant_store import collection_name_for


EXPECTED_HORIZONS = {
    "Agriculture": (6, 8, 10, 12),
    "Climate": (6, 8, 10, 12),
    "Economy_sort": (6, 8, 10, 12),
    "Energy": (12, 24, 36, 48),
    "SocialGood": (6, 8, 10, 12),
}
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SCOPE_PROTOCOL_V1 = "hic_phase20_support_based_16_config_scope_v1"
_SCOPE_PROTOCOL_V2 = "hic_phase20_completed12_plus_support4_scope_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inherited_spend_basis(path: Path) -> dict[str, str]:
    """Read a settled CNY ledger without modifying it and return its cumulative cost basis."""
    ledger_path = Path(path).resolve(strict=True)
    state = _read_json(ledger_path, "inherited CNY spend ledger")
    if type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        raise ValueError("inherited CNY spend ledger has an unsupported schema")
    cap_micros = state.get("max_spend_micros")
    inherited_micros = state.get("inherited_cost_micros")
    attempts = state.get("attempts")
    if (
        type(cap_micros) is not int or cap_micros <= 0
        or type(inherited_micros) is not int or inherited_micros < 0
        or not isinstance(attempts, list)
    ):
        raise ValueError("inherited CNY spend ledger has invalid totals")

    try:
        cap_decimal = Decimal(str(state.get("max_spend_cny")))
        inherited_decimal = Decimal(str(state.get("inherited_cost_cny", "0")))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("inherited CNY spend ledger has invalid decimal totals") from None
    if (
        not cap_decimal.is_finite()
        or cap_decimal * Decimal(1_000_000) != cap_micros
        or not inherited_decimal.is_finite()
        or inherited_decimal < 0
        or inherited_decimal * Decimal(1_000_000) != inherited_micros
    ):
        raise ValueError("inherited CNY spend ledger has inconsistent inherited cost")

    committed_micros = inherited_micros
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("inherited CNY spend ledger contains an invalid attempt")
        status = attempt.get("status")
        if not isinstance(status, str):
            raise ValueError("inherited CNY spend ledger contains an invalid attempt status")
        if status in {"reserved", "held", "unknown"}:
            raise ValueError("inherited CNY spend ledger contains unresolved attempts")
        if status in {"reconciled", "over_reserved"}:
            amount = attempt.get("actual_cost_micros")
        elif status == "conservatively_charged":
            amount = attempt.get("conservative_cost_micros")
        elif status == "cancelled_before_http":
            if attempt.get("reserved_cost_micros", 0) != 0:
                raise ValueError("cancelled pre-HTTP attempt retains a nonzero reservation")
            amount = 0
        else:
            raise ValueError("inherited CNY spend ledger contains an unknown attempt status")
        if type(amount) is not int or amount < 0:
            raise ValueError("inherited CNY spend ledger contains an invalid settled cost")
        committed_micros += amount

    if committed_micros > cap_micros:
        raise ValueError("inherited CNY spend ledger committed cost exceeds its cap")
    committed_cny = Decimal(committed_micros) / Decimal(1_000_000)
    return {
        "inherited_cost_cny": format(committed_cny, ".6f"),
        "inherited_cost_source_sha256": sha256_file(ledger_path),
    }


def _read_json(path: Path, label: str):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {Path(path)} ({type(error).__name__})") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def summarize_completed_result_for_reuse(config, result_path):
    """Validate a completed frozen result before counting it as reused scope work."""
    result = _read_json(result_path, "saved completed configuration result")
    expected = {
        "dataset": config.get("dataset"),
        "horizon": config.get("horizon"),
        "source_sha256": config.get("source_sha256"),
        "template_sha256": config.get("template_sha256"),
        "collection_name": config.get("collection_name"),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("saved completed result provenance mismatch")
    if result.get("validation_truth_read") is not False or result.get("outer_test_targets_read") is not False:
        raise ValueError("saved completed result has an invalid holdout boundary")
    completed_queries = result.get("completed_queries")
    memory_count = result.get("memory_count")
    if (
        type(completed_queries) is not int
        or completed_queries != config.get("eligible_fit_queries")
        or type(memory_count) is not int
        or memory_count < completed_queries
        or not isinstance(result.get("baseline_aggregate"), dict)
        or not isinstance(result.get("variant_aggregates"), dict)
    ):
        raise ValueError("saved completed result is incomplete or malformed")
    return {
        "dataset": config["dataset"],
        "horizon": config["horizon"],
        "completed_queries": completed_queries,
        "memory_count": memory_count,
        "baseline_aggregate": result["baseline_aggregate"],
        "variant_aggregates": result["variant_aggregates"],
        "result_reused": True,
    }


def normalize_task_template(template):
    """Return a copy whose horizon field defers to the explicit task contract."""
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty JSON object")
    try:
        normalized = json.loads(
            json.dumps(template, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError):
        raise ValueError("template must contain finite JSON-compatible values") from None
    outlook = normalized.get("forecast_outlook")
    if isinstance(outlook, dict) and "horizon" in outlook:
        outlook["horizon"] = (
            "State the horizon and frequency specified by the forecast-task contract."
        )
    return normalized


def validate_execution_limits(
    *, max_spend_cny, max_logical_calls, max_http_attempts, planned_call_upper_bound
):
    """Validate live-run budgets before the runner creates any persistent state."""
    if max_spend_cny is None:
        raise ValueError("--max-spend-cny is required with --execute")
    try:
        spend_cap = Decimal(str(max_spend_cny))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("--max-spend-cny must be a finite non-negative decimal") from None
    if not spend_cap.is_finite() or spend_cap < 0:
        raise ValueError("--max-spend-cny must be a finite non-negative decimal")
    if max_logical_calls is None or max_logical_calls < planned_call_upper_bound:
        raise ValueError(
            "--max-logical-calls must cover the no-cache preflight upper bound "
            f"({planned_call_upper_bound})"
        )
    if max_http_attempts is None or max_http_attempts < max_logical_calls:
        raise ValueError("--max-http-attempts is required and must be at least the logical-call cap")
    if spend_cap == 0:
        return "0"
    return format(spend_cap.normalize(), "f")


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_configuration_scope(scope_manifest_path):
    """Validate a post-hoc scope manifest and return its frozen config allowlist."""
    path = Path(scope_manifest_path).resolve(strict=True)
    manifest = _read_json(path, "configuration scope manifest")
    protocol = manifest.get("protocol")
    if protocol not in {_SCOPE_PROTOCOL_V1, _SCOPE_PROTOCOL_V2}:
        raise ValueError("configuration scope manifest has an unsupported protocol")
    selection = manifest.get("selection_rule")
    coverage = manifest.get("coverage")
    if (
        not isinstance(selection, dict)
        or selection.get("target_configuration_count") != 16
        or selection.get("performance_metrics_used_for_selection") is not False
        or not isinstance(coverage, dict)
        or coverage.get("source_configuration_count") != 20
        or coverage.get("selected_configuration_count") != 16
    ):
        raise ValueError("configuration scope manifest does not describe the frozen 16-of-20 scope")

    all_configs = {
        (dataset, horizon)
        for dataset, horizons in EXPECTED_HORIZONS.items()
        for horizon in horizons
    }

    def parse_rows(rows, label, expected_count):
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"16-config scope must declare exactly {expected_count} {label}")
        parsed = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError(f"{label} must be dataset/horizon objects")
            dataset = item.get("dataset")
            horizon = item.get("horizon")
            key = (dataset, horizon)
            if type(horizon) is not int or key not in all_configs or key in parsed:
                raise ValueError(f"{label} contain an unknown or duplicate configuration")
            parsed.append(key)
        return set(parsed)

    if protocol == _SCOPE_PROTOCOL_V2:
        if (
            selection.get("completed_configuration_count") != 12
            or selection.get("added_configuration_count") != 4
        ):
            raise ValueError("completed-plus-additions scope must contain twelve completed and four added configs")
        completed = parse_rows(selection.get("completed_configurations"), "completed configs", 12)
        added = parse_rows(selection.get("added_configurations"), "added configs", 4)
        selected = parse_rows(selection.get("selected_configurations"), "selected configs", 16)
        if completed & added or completed | added != selected:
            raise ValueError("completed and added configs must form the declared selected scope")
        if {dataset for dataset, _horizon in completed} != {"Agriculture", "Climate", "Economy_sort"}:
            raise ValueError("completed configs must be the twelve frozen Agriculture/Climate/Economy results")
        if {dataset for dataset, _horizon in added} != {"Energy", "SocialGood"}:
            raise ValueError("the four additions must cover Energy and SocialGood")
        added_by_family = {
            family: sum(dataset == family for dataset, _horizon in added)
            for family in ("Energy", "SocialGood")
        }
        if added_by_family != {"Energy": 2, "SocialGood": 2}:
            raise ValueError("additions must contain two Energy and two SocialGood configs")
        declared_selected = selection.get("selected_configurations")
        if not isinstance(declared_selected, list):
            raise ValueError("selected configs must be listed explicitly")
    else:
        dropped = parse_rows(selection.get("dropped"), "exclusions", 4)
        selected = all_configs - dropped
        completed = set()
        added = selected

    if len(selected) != 16:
        raise ValueError("configuration scope must select exactly 16 configurations")
    if {dataset for dataset, _horizon in selected} != set(EXPECTED_HORIZONS):
        raise ValueError("16-config scope must retain all five dataset families")
    declared_families = coverage.get("dataset_families")
    if not isinstance(declared_families, list) or set(declared_families) != set(EXPECTED_HORIZONS):
        raise ValueError("scope manifest must declare all five dataset families")
    dropped_rows = selection.get("dropped")
    dropped = all_configs - selected
    if not isinstance(dropped_rows, list) or len(dropped_rows) != 4:
        raise ValueError("16-config scope must exclude exactly four configurations")
    declared_dropped = parse_rows(dropped_rows, "exclusions", 4)
    if declared_dropped != dropped:
        raise ValueError("scope exclusions must be the complement of the selected configurations")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol": protocol,
        "selected": selected,
        "excluded": sorted(dropped),
        "completed": completed,
        "added": added,
    }


def build_fit_batch_plan(
    preflight_path,
    *,
    data_dir,
    template_paths,
    run_root,
    run_id,
    correction_schema_version=2,
    scope_manifest_path=None,
):
    """Validate frozen fit inputs and return a serializable, no-side-effect plan."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, '_' or '-' characters")
    if type(correction_schema_version) is not int or correction_schema_version not in (2, 3):
        raise ValueError("correction_schema_version must be 2 or 3")
    if not isinstance(template_paths, dict) or set(template_paths) != set(EXPECTED_HORIZONS):
        raise ValueError("template paths must cover exactly the five frozen datasets")

    preflight_path = Path(preflight_path).resolve(strict=True)
    data_dir = Path(data_dir).resolve(strict=True)
    run_root = Path(run_root).resolve()
    preflight = _read_json(preflight_path, "fit preflight")
    if preflight.get("protocol") != "fit_only_prequential_runner_preflight_v2":
        raise ValueError("fit preflight protocol is not the frozen v2 protocol")
    for field in (
        "validation_truth_returned",
        "outer_test_windows_returned",
        "provider_calls",
        "embedding_calls",
        "qdrant_writes",
    ):
        if preflight.get(field) != 0:
            raise ValueError(f"holdout boundary or side-effect preflight check failed: {field}")

    rows = preflight.get("configurations")
    expected = {
        (dataset, horizon)
        for dataset, horizons in EXPECTED_HORIZONS.items()
        for horizon in horizons
    }
    if preflight.get("configuration_count") != 20 or not isinstance(rows, list):
        raise ValueError("fit preflight must contain exactly 20 frozen configurations")
    keys = [(row.get("dataset"), row.get("horizon")) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("fit preflight configuration matrix differs from the frozen 20-config matrix")
    scope = None if scope_manifest_path is None else load_configuration_scope(scope_manifest_path)
    continue_completed_scope = scope is not None and scope["protocol"] == _SCOPE_PROTOCOL_V2
    selected_configurations = (
        expected
        if scope is None or continue_completed_scope
        else scope["selected"]
    )

    templates = {}
    for dataset, raw_path in template_paths.items():
        path = Path(raw_path).resolve(strict=True)
        template = _read_json(path, f"{dataset} template")
        if not template:
            raise ValueError(f"{dataset} template must be a non-empty JSON object")
        templates[dataset] = {
            "path": str(path),
            "source_sha256": sha256_file(path),
            "sha256": _canonical_json_sha256(normalize_task_template(template)),
        }

    planned_configs = []
    planned_calls = 0
    planned_embeddings = 0
    planned_searches = 0
    planned_upserts = 0
    for row in sorted(
        (item for item in rows if (item["dataset"], item["horizon"]) in selected_configurations),
        key=lambda item: (item["dataset"], item["horizon"]),
    ):
        dataset = row["dataset"]
        horizon = row["horizon"]
        source_path = data_dir / f"{dataset}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hash = row.get("source_data_sha256")
        if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
            raise ValueError(f"invalid frozen source hash for {dataset}/H{horizon}")
        if sha256_file(source_path) != source_hash:
            raise ValueError(f"source CSV hash changed for {dataset}/H{horizon}")

        query_plan = (row.get("plan") or {}).get("queries")
        if not isinstance(query_plan, list) or len(query_plan) != row.get("eligible_fit_queries"):
            raise ValueError(f"fit query plan/count mismatch for {dataset}/H{horizon}")
        if any(not isinstance(query, dict) or not query.get("query_case_id") for query in query_plan):
            raise ValueError(f"fit query IDs are invalid for {dataset}/H{horizon}")

        eligible = row["eligible_fit_queries"]
        with_prior = row.get("queries_with_prior_memory")
        if type(eligible) is not int or eligible < 0 or type(with_prior) is not int or not 0 <= with_prior <= eligible:
            raise ValueError(f"fit query counts are invalid for {dataset}/H{horizon}")
        calls = 3 * eligible + 2 * with_prior
        embeddings = 2 * eligible
        collection_generation = "vnext" if correction_schema_version == 2 else "vnext_v3"
        if scope is not None and not continue_completed_scope:
            collection_generation = f"{collection_generation}_scope16"
        collection = collection_name_for(
            "BAAI/bge-m3", suffix=f"{collection_generation}_{dataset.lower()}_h{horizon}"
        )
        planned_configs.append({
            "dataset": dataset,
            "horizon": horizon,
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "template_path": templates[dataset]["path"],
            "template_source_sha256": templates[dataset]["source_sha256"],
            "template_sha256": templates[dataset]["sha256"],
            "eligible_fit_queries": eligible,
            "fit_cases_excluded_for_missing_news": row.get("fit_cases_excluded_for_missing_news", 0),
            "queries_with_prior_memory": with_prior,
            "planned_uncached_provider_calls": calls,
            "planned_embedding_operations": embeddings,
            "planned_qdrant_searches": with_prior,
            "planned_memory_upserts": eligible,
            "collection_name": collection,
            "checkpoint_path": str(run_root / run_id / "checkpoints" / f"{dataset.lower()}_h{horizon}.json"),
            "result_path": str(run_root / run_id / "results" / f"{dataset.lower()}_h{horizon}.json"),
        })
        planned_calls += calls
        planned_embeddings += embeddings
        planned_searches += with_prior
        planned_upserts += eligible

    plan = {
        "schema_version": 1,
        "protocol": (
            "hic_vnext_prequential_fit_batch_v1"
            if correction_schema_version == 2
            else "hic_vnext_prequential_fit_batch_v3"
        ),
        "mode": "dry-run",
        "run_id": run_id,
        "run_directory": str(run_root / run_id),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "configuration_count": len(planned_configs),
        "eligible_fit_queries": sum(item["eligible_fit_queries"] for item in planned_configs),
        "planned_uncached_provider_calls": planned_calls,
        "planned_embedding_operations": planned_embeddings,
        "planned_qdrant_searches": planned_searches,
        "planned_memory_upserts": planned_upserts,
        "templates": templates,
        "collections": {
            f"{item['dataset']}/H{item['horizon']}": item["collection_name"]
            for item in planned_configs
        },
        "configurations": planned_configs,
        "validation_truth_read": False,
        "outer_test_targets_read": False,
        "provider_calls": 0,
        "embedding_calls": 0,
        "qdrant_writes": 0,
    }
    if scope is not None and not continue_completed_scope:
        plan["configuration_scope"] = "frozen_16_of_20_posthoc_development_subset"
        plan["source_configuration_count"] = len(rows)
        plan["scope_manifest_path"] = scope["path"]
        plan["scope_manifest_sha256"] = scope["sha256"]
        plan["excluded_configurations"] = [
            {"dataset": dataset, "horizon": horizon}
            for dataset, horizon in scope["excluded"]
        ]
    elif continue_completed_scope:
        by_key = {
            (item["dataset"], item["horizon"]): item
            for item in planned_configs
        }
        scheduled = [by_key[key] for key in sorted(scope["added"])]
        reused_results = [
            summarize_completed_result_for_reuse(by_key[key], by_key[key]["result_path"])
            for key in sorted(scope["completed"])
        ]
        checkpoint_progress = []
        remaining_fit_queries = 0
        for item in scheduled:
            checkpoint_path = Path(item["checkpoint_path"])
            expected_query_ids = {
                query["query_case_id"]
                for query in next(
                    row["plan"]["queries"]
                    for row in rows
                    if row["dataset"] == item["dataset"] and row["horizon"] == item["horizon"]
                )
            }
            status_counts = {"complete": 0, "forecast_running": 0, "planned": 0}
            if checkpoint_path.is_file():
                checkpoint = _read_json(checkpoint_path, "resumable fit checkpoint")
                expected_run_id = f"{run_id}_{item['dataset'].lower()}_h{item['horizon']}"
                if (
                    checkpoint.get("dataset") != item["dataset"]
                    or checkpoint.get("horizon") != item["horizon"]
                    or checkpoint.get("run_id") != expected_run_id
                    or checkpoint.get("collection_name") != item["collection_name"]
                    or checkpoint.get("protocol") != "hic_vnext_prequential_fit_v2"
                ):
                    raise ValueError(
                        f"existing checkpoint is not compatible with the frozen continuation: {checkpoint_path}"
                    )
                queries = checkpoint.get("queries")
                if not isinstance(queries, dict) or set(queries) != expected_query_ids:
                    raise ValueError(f"checkpoint query membership mismatch: {checkpoint_path}")
                unknown_statuses = set()
                for query_state in queries.values():
                    if not isinstance(query_state, dict):
                        unknown_statuses.add("invalid query state")
                    elif query_state.get("status") not in status_counts:
                        unknown_statuses.add(query_state.get("status"))
                if unknown_statuses:
                    raise ValueError(
                        f"checkpoint has unsupported query states {sorted(map(str, unknown_statuses))}: {checkpoint_path}"
                    )
                for row in queries.values():
                    status_counts[row["status"]] += 1
            else:
                status_counts["planned"] = len(expected_query_ids)
            remaining = len(expected_query_ids) - status_counts["complete"]
            remaining_fit_queries += remaining
            checkpoint_progress.append({
                "dataset": item["dataset"],
                "horizon": item["horizon"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.is_file(),
                "status_counts": status_counts,
                "remaining_fit_queries": remaining,
            })
        plan["execution_subset"] = {
            "protocol": scope["protocol"],
            "mode": "resume_only_four_additional_configs",
            "active_scope_configuration_count": len(scope["selected"]),
            "completed_configuration_count": len(scope["completed"]),
            "scheduled_configuration_count": len(scheduled),
            "completed_configurations": [
                {"dataset": dataset, "horizon": horizon}
                for dataset, horizon in sorted(scope["completed"])
            ],
            "scheduled_configurations": [
                {"dataset": item["dataset"], "horizon": item["horizon"]}
                for item in scheduled
            ],
            "excluded_configurations": [
                {"dataset": dataset, "horizon": horizon}
                for dataset, horizon in scope["excluded"]
            ],
            "scope_manifest_path": scope["path"],
            "scope_manifest_sha256": scope["sha256"],
            "completed_result_count": len(reused_results),
            "completed_fit_queries_reused": sum(
                row["completed_queries"] for row in reused_results
            ),
            "scheduled_fit_query_count": sum(item["eligible_fit_queries"] for item in scheduled),
            "remaining_fit_queries": remaining_fit_queries,
            "checkpoint_progress": checkpoint_progress,
            "planned_uncached_provider_calls": 5 * remaining_fit_queries,
            "planned_embedding_operations": 2 * remaining_fit_queries,
            "planned_qdrant_searches": remaining_fit_queries,
            "planned_memory_upserts": remaining_fit_queries,
        }
    if correction_schema_version == 3:
        plan["correction_schema_version"] = 3
    return plan


def dispatch_fit_batch(plan, *, execute=False, live_runner=None):
    """Keep dry-run as the default and require an explicit live-runner handoff."""
    if type(execute) is not bool:
        raise ValueError("execute must be a boolean")
    if not execute:
        return {"mode": "dry-run", "plan": plan}
    if not callable(live_runner):
        raise RuntimeError("live execution is not configured")
    return live_runner(plan)
