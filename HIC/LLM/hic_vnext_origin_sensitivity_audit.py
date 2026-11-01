"""Compare partial HIC calibration by horizon membership and unique origin."""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from LLM.hic_vnext_partial_calibration_audit import HORIZONS, VARIANTS


def _metric(value, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name} metric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"invalid {name} metric")
    return number


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_holdout_exposure(record: dict, *, label: str) -> None:
    if record.get("validation_truth_read") is not False or record.get("outer_test_targets_read") is not False:
        raise ValueError(f"holdout exposure is flagged in {label}")


def load_fit_observations(run_root: Path) -> tuple[list[dict], dict]:
    """Load paired per-query metrics from saved fit results and checkpoints only."""
    base = Path(run_root).resolve(strict=True)
    manifest_path = base / "run_manifest.json"
    manifest = _read_json(manifest_path)
    _assert_no_holdout_exposure(manifest, label="run manifest")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run manifest is missing run_id")

    result_paths = sorted((base / "results").glob("*.json"))
    if not result_paths:
        raise ValueError("no fit result files found")
    observations = []
    hashes = {"run_manifest.json": _sha256(manifest_path)}
    configurations = []
    seen_configs = set()

    for result_path in result_paths:
        result = _read_json(result_path)
        dataset = result.get("dataset")
        horizon = result.get("horizon")
        if dataset not in HORIZONS or type(horizon) is not int or horizon not in HORIZONS[dataset]:
            raise ValueError(f"result is outside the frozen configuration matrix: {result_path.name}")
        config = (dataset, horizon)
        if config in seen_configs:
            raise ValueError(f"duplicate fit result configuration: {dataset}/H{horizon}")
        seen_configs.add(config)
        if result.get("protocol") != "hic_vnext_prequential_fit_v2":
            raise ValueError(f"unexpected result protocol: {result_path.name}")
        _assert_no_holdout_exposure(result, label=result_path.name)
        if result.get("run_id") not in (None, run_id):
            raise ValueError(f"result run_id mismatch: {result_path.name}")

        query_ids = result.get("query_ids")
        completed = result.get("completed_queries")
        if (
            not isinstance(query_ids, list)
            or type(completed) is not int
            or completed <= 0
            or len(query_ids) != completed
            or len(set(query_ids)) != len(query_ids)
        ):
            raise ValueError(f"result query index/count mismatch: {result_path.name}")

        checkpoint_text = result.get("checkpoint_path")
        if not isinstance(checkpoint_text, str) or not checkpoint_text:
            raise ValueError(f"result has no checkpoint path: {result_path.name}")
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base / checkpoint_path
        checkpoint_path = checkpoint_path.resolve(strict=True)
        try:
            checkpoint_relative = checkpoint_path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"checkpoint escapes the HIC fit run root: {result_path.name}") from exc
        checkpoint = _read_json(checkpoint_path)
        expected_checkpoint_run_id = f"{run_id}_{dataset.lower()}_h{horizon}"
        if (
            checkpoint.get("run_id") != expected_checkpoint_run_id
            or checkpoint.get("dataset") != dataset
            or checkpoint.get("horizon") != horizon
            or checkpoint.get("protocol") != "hic_vnext_prequential_fit_v2"
        ):
            raise ValueError(f"result/checkpoint identity mismatch: {result_path.name}")
        _assert_no_holdout_exposure(checkpoint, label=str(checkpoint_relative))
        queries = checkpoint.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"checkpoint has no query mapping: {checkpoint_relative}")

        for query_id in query_ids:
            if not isinstance(query_id, str) or query_id not in queries:
                raise ValueError(f"result query is missing from checkpoint: {result_path.name}/{query_id}")
            query = queries[query_id]
            if not isinstance(query, dict) or query.get("status") != "complete":
                raise ValueError(f"result includes a non-complete fit query: {result_path.name}/{query_id}")
            memory = query.get("memory")
            scores = query.get("scores")
            if not isinstance(memory, dict) or not isinstance(scores, dict):
                raise ValueError(f"fit query is missing memory or scores: {result_path.name}/{query_id}")
            case_id = memory.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"fit query has no memory case_id: {result_path.name}/{query_id}")
            baseline = scores.get("baseline")
            variants = scores.get("variants")
            if not isinstance(baseline, dict) or not isinstance(variants, dict) or set(variants) != VARIANTS:
                raise ValueError(f"fit query has incomplete paired metrics: {result_path.name}/{query_id}")
            observations.append({
                "dataset": dataset,
                "case_id": case_id,
                "horizon": horizon,
                "baseline": {
                    metric: _metric(baseline.get(metric), name=f"baseline {metric}", positive=True)
                    for metric in ("mae", "mse")
                },
                "variants": {
                    variant: {
                        metric: _metric(variants[variant].get(metric), name=f"{variant} {metric}", positive=False)
                        for metric in ("mae", "mse")
                    }
                    for variant in sorted(VARIANTS)
                },
            })

        hashes[result_path.relative_to(base).as_posix()] = _sha256(result_path)
        hashes[checkpoint_relative.as_posix()] = _sha256(checkpoint_path)
        configurations.append({
            "dataset": dataset,
            "horizon": horizon,
            "fit_query_memberships": completed,
            "result_file": result_path.name,
            "checkpoint_file": checkpoint_relative.as_posix(),
            "source_sha256": result.get("source_sha256"),
            "template_sha256": result.get("template_sha256"),
        })

    # Validate cross-horizon grain and metric values after loading all paired rows.
    summarize_origin_sensitivity(observations)
    return observations, {
        "run_id": run_id,
        "result_config_count": len(configurations),
        "configurations": sorted(configurations, key=lambda item: (item["dataset"], item["horizon"])),
        "input_sha256": dict(sorted(hashes.items())),
        "validation_truth_read": False,
        "outer_test_targets_read": False,
    }


def summarize_origin_sensitivity(rows, *, tolerance: float = 1e-12) -> dict:
    """Aggregate horizon-specific relative improvements once per dataset origin.

    Each origin first averages its available horizon-level relative changes.
    Origin-level results are then averaged within dataset family, and families
    receive equal weight in the macro summary. This is a sensitivity estimand;
    it does not choose or freeze a final calibration candidate.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("at least one fit observation is required")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    seen = set()
    grouped = defaultdict(lambda: defaultdict(lambda: {"mae": [], "mse": []}))
    memberships_by_origin = Counter()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each observation must be a mapping")
        dataset = row.get("dataset")
        case_id = row.get("case_id")
        horizon = row.get("horizon")
        if dataset not in HORIZONS or type(horizon) is not int or horizon not in HORIZONS[dataset]:
            raise ValueError("observation is outside the frozen fit configuration matrix")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        origin = (dataset, case_id)
        membership = (*origin, horizon)
        if membership in seen:
            raise ValueError(f"duplicate origin-horizon observation: {dataset}/{case_id}/H{horizon}")
        seen.add(membership)
        memberships_by_origin[origin] += 1

        baseline = row.get("baseline")
        variants = row.get("variants")
        if not isinstance(baseline, dict) or not isinstance(variants, dict) or set(variants) != VARIANTS:
            raise ValueError("each observation must have baseline metrics and the complete 12-variant grid")
        base_metrics = {
            metric: _metric(baseline.get(metric), name=f"baseline {metric}", positive=True)
            for metric in ("mae", "mse")
        }
        for variant in sorted(VARIANTS):
            values = variants[variant]
            if not isinstance(values, dict):
                raise ValueError(f"invalid variant metrics: {variant}")
            for metric in ("mae", "mse"):
                candidate = _metric(values.get(metric), name=f"{variant} {metric}", positive=False)
                grouped[origin][variant][metric].append(
                    (base_metrics[metric] - candidate) / base_metrics[metric]
                )

    families = sorted({dataset for dataset, _ in memberships_by_origin})
    horizons_by_family = defaultdict(set)
    for dataset, _case_id, horizon in seen:
        horizons_by_family[dataset].add(horizon)
    complete_families = [
        family for family in families
        if horizons_by_family[family] == set(HORIZONS[family])
    ]
    incomplete_families = {
        family: {
            "available_horizons": sorted(horizons_by_family[family]),
            "required_horizons": list(HORIZONS[family]),
        }
        for family in families
        if family not in complete_families
    }
    origin_metrics = defaultdict(lambda: defaultdict(dict))
    for origin, variants in grouped.items():
        for variant, metrics in variants.items():
            origin_metrics[origin[0]][variant][origin[1]] = {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }

    ranked = []
    variants_to_rank = sorted(VARIANTS) if complete_families else []
    for variant in variants_to_rank:
        by_family = {}
        for family in complete_families:
            origins = origin_metrics[family][variant]
            origin_count = len(origins)
            mae_values = [values["mae"] for values in origins.values()]
            mse_values = [values["mse"] for values in origins.values()]
            wins = sum(value > tolerance for value in mae_values)
            ties = sum(abs(value) <= tolerance for value in mae_values)
            losses = origin_count - wins - ties
            by_family[family] = {
                "unique_origin_count": origin_count,
                "horizon_membership_count": sum(memberships_by_origin[(family, case_id)] for case_id in origins),
                "origin_wins": wins,
                "origin_ties": ties,
                "origin_losses": losses,
                "origin_win_rate": wins / origin_count,
                "mean_origin_mae_improvement": sum(mae_values) / origin_count,
                "mean_origin_mse_improvement": sum(mse_values) / origin_count,
                "origins_with_over_1pct_mae_improvement": sum(value > 0.01 for value in mae_values),
            }

        ranked.append({
            "variant": variant,
            "family_equal_origin_win_rate": sum(item["origin_win_rate"] for item in by_family.values()) / len(by_family),
            "family_equal_origin_mae_improvement": sum(item["mean_origin_mae_improvement"] for item in by_family.values()) / len(by_family),
            "family_equal_origin_mse_improvement": sum(item["mean_origin_mse_improvement"] for item in by_family.values()) / len(by_family),
            "families": by_family,
        })
    ranked.sort(key=lambda item: (
        -item["family_equal_origin_win_rate"],
        -item["family_equal_origin_mae_improvement"],
        -item["family_equal_origin_mse_improvement"],
        item["variant"],
    ))

    return {
        "schema_version": 1,
        "status": "partial_unique_origin_sensitivity_not_final_selection",
        "aggregation": {
            "within_origin": "arithmetic mean of relative improvement across available horizons",
            "within_family": "equal-weight mean across unique (dataset, case_id) origins",
            "across_families": "equal-weight mean across observed families with all frozen horizons; incomplete families are excluded",
            "relative_improvement": "(baseline_error - variant_error) / baseline_error",
            "tie_tolerance": tolerance,
        },
        "coverage": {
            "fit_query_memberships": len(rows),
            "unique_origins": len(memberships_by_origin),
            "origin_membership_distribution": {
                str(count): number
                for count, number in sorted(Counter(memberships_by_origin.values()).items())
            },
            "families": families,
            "family_count": len(families),
            "complete_families": complete_families,
            "complete_family_count": len(complete_families),
            "incomplete_families": incomplete_families,
        },
        "final_candidate_frozen": False,
        "ranked_candidates": ranked,
    }
