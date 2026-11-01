"""Pure accounting helpers for a resumable Codex-Agent HIC batch."""

from __future__ import annotations


_COUNT_FIELDS = (
    "fit_queries",
    "fit_queries_with_prior_memory",
    "validation_windows",
)


def _count(value, field, config_key):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer for {config_key}")
    return value


def build_agent_stage_inventory(
    configurations,
    *,
    fit_correction_calls_per_query_with_prior=2,
    validation_correction_calls_per_window=1,
):
    """Count semantic worker tasks without reading validation targets.

    Every fit pseudo-query has summary, baseline, and post-outcome reflection
    tasks. A correction task is needed only when an earlier memory exists; fit
    calibration currently evaluates K=1 and K=3, so it makes two such calls.
    Validation has summary, baseline, and one frozen full-HIC correction, but
    never a post-outcome reflection task.
    """
    if not isinstance(configurations, (list, tuple)) or not configurations:
        raise ValueError("configurations must be a non-empty list or tuple")
    fit_correction_calls_per_query_with_prior = _count(
        fit_correction_calls_per_query_with_prior,
        "fit_correction_calls_per_query_with_prior",
        "inventory",
    )
    validation_correction_calls_per_window = _count(
        validation_correction_calls_per_window,
        "validation_correction_calls_per_window",
        "inventory",
    )

    seen = set()
    fit_queries = 0
    fit_with_prior = 0
    validation_windows = 0
    rows = []
    for row in configurations:
        if not isinstance(row, dict):
            raise ValueError("each configuration must be a mapping")
        dataset = row.get("dataset")
        horizon = row.get("horizon")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be non-empty text")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer for {dataset}")
        key = (dataset, horizon)
        if key in seen:
            raise ValueError(f"duplicate configuration: {dataset}/H{horizon}")
        seen.add(key)

        counts = {
            field: _count(row.get(field), field, f"{dataset}/H{horizon}")
            for field in _COUNT_FIELDS
        }
        if counts["fit_queries_with_prior_memory"] > counts["fit_queries"]:
            raise ValueError(
                f"fit queries with prior memory exceed fit queries for {dataset}/H{horizon}"
            )
        fit_queries += counts["fit_queries"]
        fit_with_prior += counts["fit_queries_with_prior_memory"]
        validation_windows += counts["validation_windows"]
        rows.append({"dataset": dataset, "horizon": horizon, **counts})

    fit_without_prior = fit_queries - fit_with_prior
    fit_stages = {
        "summary": fit_queries,
        "baseline": fit_queries,
        "correction": fit_with_prior * fit_correction_calls_per_query_with_prior,
        "fit_reflection": fit_queries,
    }
    validation_stages = {
        "summary": validation_windows,
        "baseline": validation_windows,
        "correction": validation_windows * validation_correction_calls_per_window,
        "fit_reflection": 0,
    }
    stages = {
        stage: fit_stages[stage] + validation_stages[stage]
        for stage in fit_stages
    }
    return {
        "schema_version": 1,
        "configurations": rows,
        "configuration_count": len(rows),
        "fit_query_count": fit_queries,
        "fit_queries_with_prior_memory": fit_with_prior,
        "fit_queries_without_prior_memory": fit_without_prior,
        "validation_window_count": validation_windows,
        "fit_semantic_tasks_by_stage": fit_stages,
        "validation_semantic_tasks_by_stage": validation_stages,
        "semantic_tasks_by_stage": stages,
        "semantic_task_count": sum(stages.values()),
        "validation_reflection_tasks": 0,
        "validation_truth_in_agent_inputs": False,
        "outer_test_targets_read": False,
    }


def select_first_prequential_pair(fit_plan):
    """Return the earliest fit memory/query pair that can exercise retrieval."""
    if not isinstance(fit_plan, dict) or not isinstance(fit_plan.get("queries"), list):
        raise ValueError("fit_plan must contain an ordered queries list")
    queries = fit_plan["queries"]
    query_ids = []
    for row in queries:
        if not isinstance(row, dict):
            raise ValueError("fit-plan query rows must be mappings")
        case_id = row.get("query_case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in query_ids:
            raise ValueError("fit-plan query IDs must be unique non-empty strings")
        query_ids.append(case_id)
    position = {case_id: index for index, case_id in enumerate(query_ids)}
    for row in queries:
        case_id = row["query_case_id"]
        prior_ids = row.get("eligible_memory_case_ids")
        if not isinstance(prior_ids, list):
            raise ValueError("eligible_memory_case_ids must be a list")
        for memory_id in prior_ids:
            if memory_id not in position:
                raise ValueError("eligible memory ID is missing from the fit query plan")
            if position[memory_id] < position[case_id]:
                return [memory_id, case_id]
    raise ValueError("fit plan has no chronological pair with an earlier completed memory")


def select_prequential_prefix(fit_plan, *, count):
    """Return an exact chronological prefix of eligible prequential fit queries."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(fit_plan, dict) or not isinstance(fit_plan.get("queries"), list):
        raise ValueError("fit_plan must contain an ordered queries list")
    queries = fit_plan["queries"]
    ids = []
    for row in queries:
        if not isinstance(row, dict):
            raise ValueError("fit-plan query rows must be mappings")
        case_id = row.get("query_case_id")
        prior_ids = row.get("eligible_memory_case_ids")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("fit-plan query IDs must be unique non-empty strings")
        if not isinstance(prior_ids, list):
            raise ValueError("eligible_memory_case_ids must be a list")
        ids.append(case_id)
    if len(ids) < count:
        raise ValueError(f"only {len(ids)} prequential queries are available; {count} required")
    return ids[:count]
