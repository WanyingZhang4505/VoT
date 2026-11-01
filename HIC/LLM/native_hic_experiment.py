"""Offline native-cadence HIC inventory and numeric validation baselines.

This module does not call a language-model provider, load an embedding model, or
write to Qdrant. Its scores are validation-only numeric controls, not HIC results.
"""

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
from pathlib import Path

from LLM.corpus_audit import DATASET_SPECS, DEFAULT_DATA_DIR
from LLM.hic import forecast_metrics
from LLM.native_hic_data import load_native_hic_splits
from LLM.qdrant_store import collection_name_for


DEFAULT_SEED = 20261018
DEFAULT_RUN_ID = "native_hic_offline_20261018_01"
MEMORY_SIZES = (1, 10, 100)
EVALUATION_LOGICAL_CALLS_PER_WINDOW = 8
TEMPLATE_OVERRIDES = {"SocialGood": "socialgoods.json"}
DUPLICATE_DATASETS = {"Economy": "Economy_sort"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


def _validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id
    ):
        raise ValueError("run_id must contain 1-64 letters, digits, '_' or '-'")


def _finite_values(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must contain only finite values")
    return normalized


def make_numeric_baselines(history, *, horizon):
    """Build transparent forecasting controls from history, never from truth."""
    values = _finite_values(history, "history")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    recent = values[-3:]
    history_mean = sum(values) / len(values)
    recent_mean = sum(recent) / len(recent)

    x_mean = (len(values) - 1) / 2
    y_mean = history_mean
    denominator = sum((x - x_mean) ** 2 for x in range(len(values)))
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in enumerate(values))
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    trend = [intercept + slope * step for step in range(len(values), len(values) + horizon)]

    return {
        "persistence": [values[-1]] * horizon,
        "history_mean": [history_mean] * horizon,
        "recent_mean_3": [recent_mean] * horizon,
        "linear_trend": trend,
    }


def evaluate_numeric_baselines(windows):
    """Score fixed numeric controls on validation windows and retain paired detail."""
    if not windows:
        raise ValueError("at least one validation window is required")

    per_method = {}
    results = []
    seen_case_ids = set()
    for window in windows:
        case = window.get("case", {})
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError("validation windows need unique non-empty case IDs")
        seen_case_ids.add(case_id)
        if "truth" in case or "target_values" in case:
            raise ValueError(f"validation input {case_id} must not contain held-out truth")

        history = _finite_values(case.get("series"), f"history for {case_id}")
        truth = _finite_values(window.get("truth"), f"truth for {case_id}")
        forecasts = make_numeric_baselines(history, horizon=len(truth))
        metrics = {
            name: forecast_metrics(prediction, truth)
            for name, prediction in forecasts.items()
        }
        for name, row in metrics.items():
            per_method.setdefault(name, []).append(row)
        results.append({
            "case_id": case_id,
            "target_start": window.get("target_start"),
            "target_end": window.get("target_end"),
            "target_timestamps": list(window.get("target_timestamps", [])),
            "predictions": forecasts,
            "metrics": metrics,
        })

    aggregate = {}
    for name, rows in per_method.items():
        point_count = sum(row["n"] for row in rows)
        aggregate[name] = {
            "target_points": point_count,
            "windows": len(rows),
            "mae": sum(row["mae"] * row["n"] for row in rows) / point_count,
            "rmse": math.sqrt(
                sum(row["rmse"] ** 2 * row["n"] for row in rows) / point_count
            ),
            "windows_better_than_persistence_mae": sum(
                row["mae"] < baseline["mae"]
                for row, baseline in zip(rows, per_method["persistence"])
            ) if name != "persistence" else 0,
        }

    winners = {}
    for metric in ("mae", "rmse"):
        best_value = min(row[metric] for row in aggregate.values())
        winners[metric] = {
            "value": best_value,
            "methods": sorted(
                name for name, row in aggregate.items()
                if math.isclose(row[metric], best_value, rel_tol=1e-12, abs_tol=1e-12)
            ),
        }
    return {
        "purpose": "validation-only numeric reference; not an HIC/LLM result",
        "target_points": sum(row["n"] for row in per_method["persistence"]),
        "windows": results,
        "aggregate_metrics": aggregate,
        "winners": winners,
    }


def _template_status(dataset, template_dir):
    filename = TEMPLATE_OVERRIDES.get(dataset, f"{dataset.lower()}.json")
    path = Path(template_dir) / filename
    status = {"path": str(path), "available": path.is_file(), "valid_json": False}
    if not status["available"]:
        return status
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        status["error"] = f"{type(error).__name__}: {error}"
        return status
    status["valid_json"] = isinstance(value, dict) and bool(value)
    if not status["valid_json"]:
        status["error"] = "template must be a non-empty JSON object"
    return status


def _sample_nested_memory_ids(fit_cases, *, seed, dataset, horizon):
    stable_seed = int(hashlib.sha256(
        f"{seed}:{dataset}:{horizon}".encode("utf-8")
    ).hexdigest()[:8], 16)
    shuffled = list(fit_cases)
    random.Random(stable_seed).shuffle(shuffled)
    sampled = shuffled[: min(100, len(shuffled))]
    return {
        str(size): [case["case_id"] for case in sampled[:size]]
        for size in MEMORY_SIZES
    }


def _collection_names(run_id, dataset, horizon):
    slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")[:10]
    result = {}
    for size in MEMORY_SIZES:
        digest = hashlib.sha256(
            f"{run_id}:{dataset}:{horizon}:{size}".encode("utf-8")
        ).hexdigest()[:8]
        suffix = f"native_{slug}_h{horizon}_k{size}_{digest}"
        result[str(size)] = collection_name_for("BAAI/bge-m3", suffix)
    return result


def build_native_grid_manifest(
    *,
    run_id=DEFAULT_RUN_ID,
    datasets=None,
    horizons=None,
    seed=DEFAULT_SEED,
    data_dir=None,
    template_dir=None,
):
    """Inventory all configured native grids and score numeric validation controls.

    No provider request, embedding inference/model download, Qdrant operation, or
    outer-test metric calculation is performed.
    """
    _validate_run_id(run_id)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    specs = {Path(spec.filename).stem: spec for spec in DATASET_SPECS}
    selected = list(datasets) if datasets is not None else list(specs)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("datasets must be a non-empty list without duplicates")
    unsupported = sorted(set(selected) - set(specs))
    if unsupported:
        raise ValueError(f"unsupported dataset(s): {', '.join(unsupported)}")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    templates = Path(template_dir) if template_dir is not None else DEFAULT_TEMPLATE_DIR
    dataset_templates = {dataset: _template_status(dataset, templates) for dataset in selected}
    plans = []
    for dataset in selected:
        spec = specs[dataset]
        csv_path = root / spec.filename
        legal_horizons = tuple(spec.prediction_lengths)
        chosen_horizons = (
            list(horizons[dataset]) if horizons and dataset in horizons
            else list(legal_horizons)
        )
        if not chosen_horizons or len(chosen_horizons) != len(set(chosen_horizons)):
            raise ValueError(f"horizons for {dataset} must be non-empty and unique")
        illegal = sorted(set(chosen_horizons) - set(legal_horizons))
        if illegal:
            raise ValueError(f"unsupported {dataset} horizon(s): {illegal}")

        if dataset in DUPLICATE_DATASETS:
            plans.append({
                "dataset": dataset,
                "source_file": spec.filename,
                "status": "excluded_duplicate",
                "duplicate_of": DUPLICATE_DATASETS[dataset],
                "reason": "Exact duplicate source data are not counted twice.",
                "horizons": chosen_horizons,
            })
            continue
        if not csv_path.is_file():
            plans.append({
                "dataset": dataset,
                "source_file": spec.filename,
                "status": "unavailable",
                "reason": "Configured CSV is not present in the local corpus.",
                "horizons": chosen_horizons,
            })
            continue

        for horizon in chosen_horizons:
            split = load_native_hic_splits(
                dataset,
                horizon=horizon,
                data_dir=root,
                include_outer_test_targets=False,
            )
            validation_metrics = (
                evaluate_numeric_baselines(split["validation_windows"])
                if split["validation_windows"] else None
            )
            template = dataset_templates[dataset]
            fit_calls = 3 * len(split["fit_cases"])
            validation_calls = EVALUATION_LOGICAL_CALLS_PER_WINDOW * len(split["validation_windows"])
            outer_calls = None
            plans.append({
                "dataset": dataset,
                "source_file": spec.filename,
                "status": "audited",
                "cadence": split["cadence"],
                "history_length": split["history_length"],
                "horizon": horizon,
                "split_fractions": {
                    "fit": split["fit_fraction"],
                    "fit_plus_validation": split["validation_fraction"],
                },
                "boundaries": split["boundaries"],
                "quality": split["quality"],
                "covariate_columns": split["covariate_columns"],
                "fit_memory_count": len(split["fit_cases"]),
                "validation_window_count": len(split["validation_windows"]),
                "outer_test_deferred": True,
                "outer_test_window_count": None,
                "case_ids": {
                    "fit": [case["case_id"] for case in split["fit_cases"]],
                    "validation": [item["case"]["case_id"] for item in split["validation_windows"]],
                    "outer_test": None,
                },
                "memory_candidate_ids": _sample_nested_memory_ids(
                    split["fit_cases"], seed=seed, dataset=dataset, horizon=horizon
                ),
                "collections": _collection_names(run_id, dataset, horizon),
                "template": template,
                "offline_numeric_validation": validation_metrics,
                "call_budget_estimate": {
                    "template_generation": 0,
                    "fit_memory_build": fit_calls,
                    "validation_candidates_k0_k1_k10_k100": validation_calls,
                    "outer_test_after_freeze": outer_calls,
                    "before_outer_test": fit_calls + validation_calls,
                    "through_outer_test": None,
                },
                "experiment_ledger_fields": [
                    "template/schema version and exact diff",
                    "input/cutoff/covariate policy",
                    "embedding model, max sequence length, query/passage instructions, normalization",
                    "Qdrant collection, payload filter, top-k, score gate, retrieved correction IDs",
                    "forecast model, prompt version, temperature, and horizon/frequency",
                    "split/window IDs, logical calls/retries, per-window and aggregate metrics",
                ],
            })

        # A missing dataset template is generated once from fit data, not once per
        # horizon. Do not estimate generation from validation or outer-test text.
        dataset_plans = [
            plan for plan in plans
            if plan.get("dataset") == dataset and plan.get("status") == "audited"
        ]
        if (
            dataset_plans
            and not dataset_templates[dataset]["valid_json"]
            and any(plan["fit_memory_count"] for plan in dataset_plans)
        ):
            first_fit_plan = next(
                plan for plan in dataset_plans if plan["fit_memory_count"]
            )
            budget = first_fit_plan["call_budget_estimate"]
            budget["template_generation"] = 1
            budget["before_outer_test"] += 1

    active = [plan for plan in plans if plan["status"] == "audited"]
    deferred_total_keys = {"outer_test_after_freeze", "through_outer_test"}
    totals = {
        key: (
            None
            if key in deferred_total_keys
            else sum(plan["call_budget_estimate"][key] for plan in active)
        )
        for key in (
            "template_generation",
            "fit_memory_build",
            "validation_candidates_k0_k1_k10_k100",
            "outer_test_after_freeze",
            "before_outer_test",
            "through_outer_test",
        )
    }
    return {
        "mode": "offline-native-grid",
        "run_id": run_id,
        "seed": seed,
        "provider_requests_made": 0,
        "embedding_inferences_made": 0,
        "qdrant_writes_made": 0,
        "outer_test_metrics_calculated": False,
        "embedding_config": {
            "model": "BAAI/bge-m3",
            "dimension": 1024,
            "configured_max_length": 8192,
            "query_instruction": "",
            "pooling": "CLS token",
            "normalization": "L2",
            "similarity": "cosine",
            "effective_max_length": "min(configured_max_length, tokenizer/model limit at runtime)",
        },
        "logical_call_estimate_if_executed": totals,
        "numeric_control_methods": [
            "persistence",
            "history_mean",
            "recent_mean_3",
            "linear_trend",
        ],
        "selection_rule": (
            "Numeric controls are summarized per dataset and horizon; MAE and RMSE "
            "have separate validation winners. They are references, not HIC winners. "
            "No cross-dataset raw-error pooling is used."
        ),
        "plans": plans,
    }


def _write_new_json(path, value):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {destination}")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=tuple(
        Path(spec.filename).stem for spec in DATASET_SPECS
    ))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_native_grid_manifest(
        run_id=args.run_id,
        datasets=args.dataset,
        seed=args.seed,
        data_dir=args.data_dir,
        template_dir=args.template_dir,
    )
    if args.output:
        _write_new_json(args.output, manifest)
        print(json.dumps({
            "mode": manifest["mode"],
            "report": str(args.output),
            "plans": len(manifest["plans"]),
            "provider_requests_made": 0,
            "qdrant_writes_made": 0,
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
