"""Produce a side-effect-free manifest for the monthly HIC experiment."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from LLM.monthly_hic_data import MONTHLY_DATASETS, load_monthly_hic_splits
from LLM.qdrant_store import collection_name_for


DEFAULT_RUN_ID = "monthly_20261018_01"
DEFAULT_SEED = 20261018
EVALUATION_LOGICAL_CALLS_PER_WINDOW = 8
MEMORY_SIZES = (1, 10, 100)
TEMPLATE_FILES = {
    "Agriculture": "Agriculture.json",
    "Climate": "Climate.json",
    "Economy_sort": "Economy_sort.json",
    "Security": "Security.json",
    "SocialGood": "socialgoods.json",
}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


def _validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id
    ):
        raise ValueError("run_id must contain 1-64 letters, digits, '_' or '-'")


def _collection_suffix(run_id, dataset, memory_size):
    slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")[:10]
    digest = hashlib.sha256(
        f"{run_id}:{dataset}:{memory_size}".encode("utf-8")
    ).hexdigest()[:8]
    return f"m_{slug}_k{memory_size}_{digest}"


def _template_status(dataset, template_dir):
    path = Path(template_dir) / TEMPLATE_FILES[dataset]
    result = {
        "path": str(path),
        "available": False,
        "valid_json": False,
        "generation_calls_if_missing": 1,
    }
    if not path.is_file() or path.stat().st_size == 0:
        return result

    result["available"] = True
    try:
        with path.open(encoding="utf-8") as handle:
            template = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    result["valid_json"] = isinstance(template, dict) and bool(template)
    if not result["valid_json"]:
        result["error"] = "template must be a non-empty JSON object"
    return result


def _dataset_manifest(dataset, *, run_id, seed, data_dir, template_dir):
    split = load_monthly_hic_splits(
        dataset,
        data_dir=data_dir,
        include_outer_test_targets=False,
    )
    fit_cases = split["fit_cases"]
    validation = split["validation_windows"]
    if len(fit_cases) < 100:
        raise ValueError(f"{dataset} has fewer than 100 eligible fit memories")

    sampled_cases = random.Random(seed).sample(fit_cases, k=100)
    candidate_ids = {
        "1": [case["case_id"] for case in sampled_cases[:1]],
        "10": [case["case_id"] for case in sampled_cases[:10]],
        "100": [case["case_id"] for case in sampled_cases],
    }
    template = _template_status(dataset, template_dir)
    collections = {
        str(size): collection_name_for(
            "BAAI/bge-m3", _collection_suffix(run_id, dataset, size)
        )
        for size in MEMORY_SIZES
    }
    fit_build_calls = 3 * len(fit_cases)
    validation_calls = EVALUATION_LOGICAL_CALLS_PER_WINDOW * len(validation)
    template_calls = template["generation_calls_if_missing"] if not template["valid_json"] else 0

    return {
        "dataset": dataset,
        "row_count": split["row_count"],
        "history_length": split["history_length"],
        "horizon": split["horizon"],
        "split_fractions": {
            "fit": split["fit_fraction"],
            "fit_plus_validation": split["validation_fraction"],
        },
        "boundaries": split["boundaries"],
        "covariate_columns": split["covariate_columns"],
        "fit_memory_count": len(fit_cases),
        "validation_window_count": len(validation),
        "outer_test_deferred": True,
        "outer_test_window_count": None,
        "case_ids": {
            "fit": [case["case_id"] for case in fit_cases],
            "validation": [window["case"]["case_id"] for window in validation],
            "outer_test": None,
        },
        "memory_candidate_ids": candidate_ids,
        "template": template,
        "collections": collections,
        "call_budget": {
            "template_generation": template_calls,
            "fit_memory_build": fit_build_calls,
            "validation_candidates": validation_calls,
            "outer_test_after_freeze": None,
            "before_outer_test": template_calls + fit_build_calls + validation_calls,
            "through_outer_test": None,
        },
        "evaluation_protocol": (
            "Compare K=0/1/10/100 on the fixed chronological validation windows; "
            "each window uses one summary, four forecasts, and three post-score "
            "correction calls. This preflight parses only fit/validation rows and "
            "does not build or expose outer-test cases or targets."
        ),
    }


def build_dry_run_manifest(
    *,
    run_id=DEFAULT_RUN_ID,
    datasets=None,
    seed=DEFAULT_SEED,
    data_dir=None,
    template_dir=None,
):
    """Inventory the eligible monthly windows and estimate logical API calls.

    This function performs no provider requests, model downloads, or Qdrant I/O.
    """
    _validate_run_id(run_id)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    selected = list(datasets) if datasets is not None else list(MONTHLY_DATASETS)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("datasets must be a non-empty list without duplicates")
    unsupported = sorted(set(selected) - set(MONTHLY_DATASETS))
    if unsupported:
        raise ValueError(f"unsupported monthly HIC dataset(s): {', '.join(unsupported)}")

    effective_data_dir = Path(data_dir) if data_dir else None
    effective_template_dir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    plans = [
        _dataset_manifest(
            dataset,
            run_id=run_id,
            seed=seed,
            data_dir=effective_data_dir,
            template_dir=effective_template_dir,
        )
        for dataset in selected
    ]
    totals = {
        key: sum(plan["call_budget"][key] for plan in plans)
        for key in (
            "template_generation",
            "fit_memory_build",
            "validation_candidates",
            "before_outer_test",
        )
    }
    totals["outer_test_after_freeze"] = None
    totals["through_outer_test"] = None
    return {
        "mode": "dry-run-only",
        "run_id": run_id,
        "seed": seed,
        "embedding_model": "BAAI/bge-m3",
        "vector_dimension": 1024,
        "similarity": "cosine",
        "memory_sizes": list(MEMORY_SIZES),
        "api_requests_made": 0,
        "qdrant_writes_made": 0,
        "logical_call_estimate": totals,
        "datasets": plans,
        "limitations": [
            "This manifest only covers the five monthly datasets supported by the 8-history/12-target adapter.",
            "Templates absent or invalid are reported; dry-run never generates or edits them.",
            "Outer-test targets and case IDs are deliberately deferred; this is not a final-test plan.",
            "The manifest is not an HIC run and contains no forecast-accuracy result.",
        ],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=tuple(MONTHLY_DATASETS))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_dry_run_manifest(
        run_id=args.run_id,
        datasets=args.dataset,
        seed=args.seed,
        data_dir=args.data_dir,
        template_dir=args.template_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
