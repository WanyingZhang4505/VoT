"""Run a small, resumable native-cadence HIC validation pilot."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from LLM import hic
from LLM.hic_memory_builder import (
    _fingerprint,
    _validate_cases,
    build_completed_memories,
)
from LLM.hic_qdrant_eval import run_prequential_evaluation
from LLM.native_hic_data import load_native_hic_splits, target_metadata_for_dataset
from LLM.native_hic_experiment import make_numeric_baselines
from LLM.qdrant_store import QdrantHicStore, collection_name_for
from LLM.run_economy_hic_pilot import (
    make_call_json,
    make_http_attempt_budget,
)
from LLM.sample_data import select_seeded_cases


ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ROOT / "LLM" / "evaluations"
TEMPLATE_DIR = ROOT / "LLM" / "template"
DEFAULT_SEED = 20261018
DEFAULT_DATASET = "Agriculture"
DEFAULT_HORIZON = 12
DEFAULT_TRAIN_COUNT = 10
DEFAULT_VALIDATION_COUNT = 1
DEFAULT_MEMORY_SIZES = (1, 10)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TEMPLATE_OVERRIDES = {"SocialGood": "socialgoods.json"}


def _template_filename(dataset):
    return _TEMPLATE_OVERRIDES.get(dataset, f"{dataset.lower()}.json")


def _dataset_description(dataset):
    target_metadata = target_metadata_for_dataset(dataset)
    if target_metadata is not None:
        return (
            f"Dataset: {dataset}, from the upstream VoT repository. The numeric OT "
            f"target is mapped to {target_metadata['target']} "
            f"({target_metadata['unit']}). The mapping evidence is: "
            f"{target_metadata['target_metadata_evidence']} Source: "
            f"{target_metadata['target_metadata_source']} Treat this as a supported "
            "dataset-level mapping, not as proof that text facts cause target changes. "
            "Text facts are aligned to source periods; undated facts are not independently "
            "verified as publication-time news."
        )
    return (
        f"Dataset: {dataset}, from the upstream VoT repository. The numeric target "
        "column is named OT; the local metadata does not document its unit or exact "
        "real-world interpretation, so treat it as an unnamed numeric target. Text "
        "facts are aligned to source periods; undated facts are not independently "
        "verified as publication-time news."
    )


def _logical_calls_per_window(memory_sizes):
    # One summary, one forecast for K=0 plus one per retrieval size, and one
    # post-outcome correction for each retrieval arm.
    return 1 + (1 + len(memory_sizes)) + len(memory_sizes)


def prepare_pilot_inputs(
    *,
    dataset=DEFAULT_DATASET,
    horizon=DEFAULT_HORIZON,
    train_count=DEFAULT_TRAIN_COUNT,
    validation_count=DEFAULT_VALIDATION_COUNT,
    validation_start_index=0,
    seed=DEFAULT_SEED,
    data_dir=None,
    template_available=False,
    memory_sizes=DEFAULT_MEMORY_SIZES,
):
    """Prepare seeded fit memories and chronological validation inputs only."""
    if not isinstance(memory_sizes, (list, tuple)) or not memory_sizes:
        raise ValueError("memory_sizes must be a non-empty sequence")
    normalized_memory_sizes = tuple(sorted(memory_sizes))
    if any(type(size) is not int or size < 1 for size in normalized_memory_sizes):
        raise ValueError("memory sizes must be positive integers")
    if len(set(normalized_memory_sizes)) != len(normalized_memory_sizes):
        raise ValueError("memory sizes must be unique")
    if not isinstance(train_count, int) or isinstance(train_count, bool):
        raise ValueError("train_count must be an integer")
    if train_count < max(normalized_memory_sizes):
        raise ValueError(
            f"train_count must be at least {max(normalized_memory_sizes)} "
            "to seed every retrieval arm"
        )
    if not isinstance(validation_count, int) or isinstance(validation_count, bool) or validation_count < 1:
        raise ValueError("validation_count must be a positive integer")
    if (
        not isinstance(validation_start_index, int)
        or isinstance(validation_start_index, bool)
        or validation_start_index < 0
    ):
        raise ValueError("validation_start_index must be a non-negative integer")

    splits = load_native_hic_splits(
        dataset,
        horizon=horizon,
        data_dir=data_dir,
        include_outer_test_targets=False,
    )
    if train_count > len(splits["fit_cases"]):
        raise ValueError(
            f"requested {train_count} fit cases, but only {len(splits['fit_cases'])} are available"
        )
    validation_end_index = validation_start_index + validation_count
    if validation_end_index > len(splits["validation_windows"]):
        raise ValueError(
            f"requested validation indices {validation_start_index}..{validation_end_index - 1}, "
            f"but only {len(splits['validation_windows'])} windows are available"
        )

    training_cases = select_seeded_cases(
        splits["fit_cases"], count=train_count, seed=seed
    )
    validation_windows = splits["validation_windows"][
        validation_start_index:validation_end_index
    ]
    training_ids = [case["case_id"] for case in training_cases]
    validation_ids = [window["case"]["case_id"] for window in validation_windows]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("sampled training case IDs are not unique")
    if set(training_ids) & set(validation_ids):
        raise ValueError("training cases overlap validation cases")
    for window in validation_windows:
        if "truth" in window["case"] or "target_values" in window["case"]:
            raise ValueError(
                f"validation input {window['case']['case_id']} contains held-out truth"
            )

    template_calls = 0 if template_available else 1
    evaluation_calls = validation_count * _logical_calls_per_window(
        normalized_memory_sizes
    )
    plan = {
        "dataset": dataset,
        "cadence": splits["cadence"],
        "history_length": splits["history_length"],
        "horizon": horizon,
        "seed": seed,
        "split_fractions": {
            "fit": splits["fit_fraction"],
            "fit_plus_validation": splits["validation_fraction"],
        },
        "boundaries": splits["boundaries"],
        "training_count": len(training_cases),
        "training_case_ids": training_ids,
        "available_validation_windows": len(splits["validation_windows"]),
        "validation_count": len(validation_windows),
        "validation_start_index": validation_start_index,
        "validation_case_ids": validation_ids,
        "validation_truth_is_separate": all(
            "truth" not in window["case"] for window in validation_windows
        ),
        "outer_test_deferred": True,
        "outer_test_count": None,
        "outer_test_case_ids": None,
        "retrieval_memory_sizes": list(normalized_memory_sizes),
        "top_k_per_query": 1,
        "template_available": bool(template_available),
        "template_generation_logical_calls": template_calls,
        "training_logical_calls": 3 * len(training_cases),
        "validation_logical_calls": evaluation_calls,
        "estimated_logical_calls": (
            template_calls + 3 * len(training_cases) + evaluation_calls
        ),
        "selection_scope": "validation only; outer-test cases and targets are untouched",
    }
    return plan, training_cases, validation_windows, splits


def build_pilot_plan(**kwargs):
    """Return a JSON-serializable plan without making external calls or writes."""
    plan, _, _, _ = prepare_pilot_inputs(**kwargs)
    return plan


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    preserve_temporary = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if attempt == 4:
                    preserve_temporary = True
                    raise PermissionError(
                        f"could not replace checkpoint after 5 attempts; "
                        f"complete recovery file retained at {temporary}"
                    ) from error
                time.sleep(0.1 * (attempt + 1))
    finally:
        if temporary is not None and temporary.exists() and not preserve_temporary:
            temporary.unlink()


def _read_template(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"template is not valid JSON: {type(error).__name__}") from None
    if not isinstance(value, dict) or not value:
        raise ValueError("template must be a non-empty JSON object")
    return value


def _collection_names(run_id, dataset, horizon, memory_sizes):
    slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")[:10]
    result = {}
    for size in memory_sizes:
        digest = hashlib.sha256(
            f"{run_id}:{dataset}:{horizon}:{size}".encode("utf-8")
        ).hexdigest()[:8]
        suffix = f"native_{slug}_h{horizon}_k{size}_{digest}"
        result[size] = collection_name_for("BAAI/bge-m3", suffix)
    return result


def _collection_case_ids(client, collection_name):
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return {
        point.payload.get("case_id")
        for point in points
        if isinstance(point.payload, dict) and point.payload.get("case_id")
    }


def _make_qdrant_client(*, url=None, path=None):
    from qdrant_client import QdrantClient

    if url:
        return QdrantClient(url=url)
    if path:
        return QdrantClient(path=str(path))
    raise ValueError("configure either qdrant_url or qdrant_path")


def _seed_nested_memories_from_checkpoint(
    *, stores, training_cases, checkpoint_path, run_id, dataset,
    allow_additional_points=False,
):
    """Build nested K stores from one completed maximum-K training checkpoint."""
    if not isinstance(stores, dict) or not stores:
        raise ValueError("stores must contain at least one positive retrieval size")
    sizes = tuple(sorted(stores))
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("store keys must be positive integers")
    if len(training_cases) < sizes[-1]:
        raise ValueError("training cases are fewer than the largest retrieval size")
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    states = checkpoint.get("cases", {})
    for size in sizes:
        store = stores[size]
        store.ensure_collection()
        if size == sizes[-1]:
            if store.count() < size:
                raise RuntimeError("maximum-K collection is missing completed training memories")
            continue
        for case in training_cases[:size]:
            case_id = case["case_id"]
            state = states.get(case_id, {})
            required = ("summary", "forecast", "correction", "embedding", "ingested")
            if not all(state.get(key) for key in required):
                raise RuntimeError(
                    f"cannot seed K={size}; training memory {case_id} is incomplete"
                )
            store.upsert_completed_case(
                case_id=case_id,
                summary=state["summary"],
                correction=state["correction"],
                embedding=state["embedding"],
                observed_targets=case["truth"],
                prediction=state["forecast"]["prediction"],
                cutoff_date=case["target_end"],
                source=f"training:{dataset}:{run_id}",
            )
        if store.count() != size and not (
            allow_additional_points and store.count() >= size
        ):
            raise RuntimeError(f"K={size} collection does not contain exactly {size} memories")


def _training_report_from_checkpoint(*, checkpoint_path, cases, dataset, run_id, max_calls):
    """Validate and summarize a fully completed training checkpoint on resume."""
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    normalized_cases = _validate_cases(cases, dataset)
    case_ids = [case["case_id"] for case in normalized_cases]
    expected_run_id = f"{run_id}_train"
    if checkpoint.get("schema_version") != 1 or checkpoint.get("run_id") != expected_run_id:
        raise ValueError("training checkpoint schema or run ID does not match")
    if checkpoint.get("pending_call") or checkpoint.get("last_completed_call"):
        raise ValueError("training checkpoint contains an unresolved API result")
    if checkpoint.get("dataset") != dataset or checkpoint.get("case_ids") != case_ids:
        raise ValueError("training checkpoint dataset or cases do not match")
    if checkpoint.get("input_fingerprint") != _fingerprint(normalized_cases):
        raise ValueError("training checkpoint input fingerprint does not match")
    if checkpoint.get("history_length") != len(normalized_cases[0]["series"]):
        raise ValueError("training checkpoint history length does not match")
    if checkpoint.get("horizon") != len(normalized_cases[0]["truth"]):
        raise ValueError("training checkpoint horizon does not match")

    states = checkpoint.get("cases", {})
    required = ("summary", "forecast", "correction", "embedding", "ingested")
    for case_id in case_ids:
        state = states.get(case_id, {})
        if not all(state.get(key) for key in required):
            raise ValueError(f"training checkpoint is incomplete for {case_id}")
        if len(state["embedding"]) != 1024:
            raise ValueError(f"training checkpoint embedding dimension is invalid for {case_id}")

    logical_calls = checkpoint.get("logical_deepseek_calls")
    if logical_calls != 3 * len(case_ids) or logical_calls > max_calls:
        raise ValueError("training checkpoint logical-call count is inconsistent")
    return {
        "run_id": expected_run_id,
        "dataset": dataset,
        "history_length": len(normalized_cases[0]["series"]),
        "horizon": len(normalized_cases[0]["truth"]),
        "embedding_model": "BAAI/bge-m3",
        "case_count": len(case_ids),
        "completed_case_ids": case_ids,
        "initial_collection_count": 0,
        "checkpoint_ingested_count_at_start": 0,
        "final_collection_count": len(case_ids),
        "training_memory_case_count": len(case_ids),
        "logical_deepseek_calls": logical_calls,
        "max_logical_deepseek_calls": max_calls,
        "input_fingerprint": checkpoint["input_fingerprint"],
        "resumed_from_completed_checkpoint": True,
    }


def _validate_resume_memory_ids(
    *, actual_ids_by_size, training_case_ids, evaluation_checkpoint,
    validation_case_ids, memory_sizes,
):
    """Match Qdrant IDs to fit prefixes and checkpointed prequential writes."""
    validation_case_ids = list(validation_case_ids)
    if evaluation_checkpoint.get("case_ids") != validation_case_ids:
        raise RuntimeError("evaluation checkpoint case IDs do not match this run")
    if evaluation_checkpoint.get("memory_sizes") != list(memory_sizes):
        raise RuntimeError("evaluation checkpoint memory sizes do not match this run")
    windows = evaluation_checkpoint.get("windows", {})
    if not set(windows).issubset(validation_case_ids):
        raise RuntimeError("evaluation checkpoint contains an unknown validation case")

    for size in memory_sizes:
        expected = set(training_case_ids[:size])
        for case_id, state in windows.items():
            if size in state.get("ingested_arms", []):
                expected.add(case_id)

        actual = set(actual_ids_by_size[size])
        missing = expected - actual
        if missing:
            raise RuntimeError(f"K={size} collection is missing checkpointed IDs: {sorted(missing)}")
        untracked = actual - expected
        if len(untracked) > 1:
            raise RuntimeError(f"K={size} collection has untracked IDs: {sorted(untracked)}")
        for case_id in untracked:
            state = windows.get(case_id)
            arm = str(size)
            replayable_upsert = (
                case_id in validation_case_ids
                and state is not None
                and state.get("scoring") is not None
                and arm in state.get("forecasts", {})
                and arm in state.get("corrections", {})
                and size not in state.get("ingested_arms", [])
            )
            if not replayable_upsert:
                raise RuntimeError(f"K={size} collection has untracked ID: {case_id}")


def _embed_summary(embedder, summary, *, is_query):
    text = hic.summary_to_embedding_text(summary)
    vectors = embedder.encode([text], is_query=is_query)
    if len(vectors) != 1:
        raise RuntimeError("BGE-M3 did not return exactly one embedding")
    return vectors[0]


def _baseline_metrics(window):
    history = window["case"]["series"]
    truth = window["truth"]
    forecasts = make_numeric_baselines(history, horizon=len(truth))
    return {
        name: hic.forecast_metrics(prediction, truth)
        for name, prediction in forecasts.items()
    }


def _candidate_winners(validation_report, baselines_by_window):
    if not validation_report.get("windows"):
        raise ValueError("validation report must contain at least one window")
    first_arms = validation_report["windows"][0].get("arms", {})
    if "0" not in first_arms:
        raise ValueError("validation report must include the K=0 control")
    arm_keys = sorted(first_arms, key=int)
    candidates = {f"K={arm}": [] for arm in arm_keys}
    for window in validation_report["windows"]:
        if set(window.get("arms", {})) != set(arm_keys):
            raise ValueError("validation windows contain inconsistent retrieval arms")
        for arm in arm_keys:
            candidates[f"K={arm}"].append(window["arms"][arm]["metrics"])
    for method in ("persistence", "history_mean", "recent_mean_3", "linear_trend"):
        candidates[method] = [row[method] for row in baselines_by_window]

    aggregate = {}
    for method, rows in candidates.items():
        count = sum(row["n"] for row in rows)
        aggregate[method] = {
            "target_points": count,
            "mae": sum(row["mae"] * row["n"] for row in rows) / count,
            "rmse": (
                sum(row["rmse"] ** 2 * row["n"] for row in rows) / count
            ) ** 0.5,
        }
    winners = {
        metric: min(aggregate, key=lambda method: aggregate[method][metric])
        for metric in ("mae", "rmse")
    }
    return {"aggregate": aggregate, "observed_winner_by_metric": winners}


def aggregate_validation_windows(windows):
    """Pool paired validation metrics across windows by forecast target count."""
    if not windows:
        raise ValueError("at least one validation window is required")
    arm_keys = set(windows[0].get("arms", {}))
    if "0" not in arm_keys or len(arm_keys) < 2:
        raise ValueError("each validation window must include K=0 and a retrieval arm")
    results_by_arm = {int(arm): [] for arm in arm_keys}
    for window in windows:
        if set(window.get("arms", {})) != arm_keys:
            raise ValueError("validation windows must contain the same retrieval arms")
        for arm_key in sorted(arm_keys, key=int):
            results_by_arm[int(arm_key)].append(
                {"metrics": window["arms"][arm_key]["metrics"]}
            )
    from LLM.hic_qdrant_eval import aggregate_arm_metrics

    return {
        "window_count": len(windows),
        "arms": aggregate_arm_metrics(results_by_arm),
    }


def _validate_continuation_plan(source_plan, continuation_plan):
    """Require the same fit set and the immediately following validation windows."""
    for key in ("dataset", "horizon", "seed"):
        if source_plan.get(key) != continuation_plan.get(key):
            raise ValueError(f"continuation {key} differs from the source pilot")
    if source_plan.get("training_case_ids") != continuation_plan.get("training_case_ids"):
        raise ValueError("continuation training cases differ from the source pilot")
    expected_start = (
        source_plan.get("validation_start_index", 0)
        + source_plan.get("validation_count", len(source_plan.get("validation_case_ids", [])))
    )
    if continuation_plan.get("validation_start_index") != expected_start:
        raise ValueError("continuation windows must immediately follow the source validation")
    source_ids = set(source_plan.get("validation_case_ids", []))
    next_ids = set(continuation_plan.get("validation_case_ids", []))
    if source_ids & next_ids:
        raise ValueError("continuation validation windows overlap the source pilot")


def _report_template_hash(template):
    canonical = json.dumps(
        template, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _paths(run_id):
    return {
        "report": EVALUATIONS / f"native_hic_pilot_{run_id}.json",
        "training_checkpoint": EVALUATIONS / f"native_hic_pilot_{run_id}.training.checkpoint.json",
        "evaluation_checkpoint": EVALUATIONS / f"native_hic_pilot_{run_id}.validation.checkpoint.json",
        "generated_template": EVALUATIONS / f"native_hic_pilot_{run_id}.template.json",
        "http_attempt_ledger": EVALUATIONS / f"native_hic_pilot_{run_id}.http_attempts.json",
    }


def _load_or_generate_template(
    *, dataset, dataset_description, source_path, generated_path, call_json, resume
):
    if source_path.is_file():
        template = _read_template(source_path)
        return template, source_path, "existing", 0
    if generated_path.is_file():
        if not resume:
            raise FileExistsError(
                "a generated template for this run already exists; use --resume or a new run ID"
            )
        template = _read_template(generated_path)
        return template, generated_path, "generated_resume", 1
    template = call_json(hic.build_template_prompt(dataset_description), temperature=0.0)
    if not isinstance(template, dict) or not template:
        raise ValueError("DeepSeek did not generate a non-empty JSON template")
    _atomic_json(generated_path, template)
    return template, generated_path, "generated", 1


def run_native_hic_pilot(args):
    if not _RUN_ID.fullmatch(args.run_id):
        raise ValueError("run ID must be 1-64 letters, digits, '_' or '-' characters")
    if args.continue_from_run_id:
        if tuple(args.memory_sizes) != DEFAULT_MEMORY_SIZES:
            raise ValueError("continuation uses the memory sizes saved in its source pilot")
        return continue_native_hic_validation(args)
    memory_sizes = tuple(args.memory_sizes)
    template_source = args.template_path or (TEMPLATE_DIR / _template_filename(args.dataset))
    source_template_valid = False
    if template_source.is_file():
        _read_template(template_source)
        source_template_valid = True
    plan, training_cases, validation_windows, _ = prepare_pilot_inputs(
        dataset=args.dataset,
        horizon=args.horizon,
        train_count=args.train_count,
        validation_count=args.validation_count,
        validation_start_index=args.validation_start_index,
        seed=args.seed,
        data_dir=args.data_dir,
        template_available=source_template_valid,
        memory_sizes=memory_sizes,
    )
    paths = _paths(args.run_id)
    collection_names = _collection_names(
        args.run_id, args.dataset, args.horizon, memory_sizes
    )
    plan = {
        **plan,
        "run_id": args.run_id,
        "template_source_path": str(template_source),
        "generated_template_path": str(paths["generated_template"]),
        "qdrant_url": args.qdrant_url,
        "qdrant_path": str(args.qdrant_path) if args.qdrant_path else None,
        "collections": {str(key): value for key, value in collection_names.items()},
        "max_logical_calls": args.max_calls,
        "max_http_attempts": args.max_http_attempts,
        "report_path": str(paths["report"]),
        "training_checkpoint_path": str(paths["training_checkpoint"]),
        "evaluation_checkpoint_path": str(paths["evaluation_checkpoint"]),
        "http_attempt_ledger_path": str(paths["http_attempt_ledger"]),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if args.max_http_attempts is None:
        raise ValueError("--max-http-attempts is required for live API calls")
    if args.max_calls < plan["estimated_logical_calls"]:
        raise ValueError(
            f"this run needs at least {plan['estimated_logical_calls']} logical calls; "
            f"configured max_calls={args.max_calls}"
        )
    if args.api_env:
        load_dotenv(args.api_env, override=False)
    hic.DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DeepSeek key is unavailable; provide --api-env or set DEEPSEEK_API_KEY. "
            "The key value was not displayed."
        )
    probe_client = _make_qdrant_client(url=args.qdrant_url, path=args.qdrant_path)
    try:
        probe_client.get_collections()
    finally:
        probe_client.close()
    evaluation_checkpoint_available = paths["evaluation_checkpoint"].is_file()
    if args.resume:
        if not paths["training_checkpoint"].is_file():
            raise FileNotFoundError("--resume requires the matching training checkpoint")
        if not paths["http_attempt_ledger"].is_file():
            raise FileNotFoundError("--resume requires the cumulative HTTP-attempt ledger")
        if paths["evaluation_checkpoint"].exists() and not evaluation_checkpoint_available:
            raise FileNotFoundError("evaluation checkpoint path exists but is not a file")
    elif any(path.exists() for path in paths.values()):
        raise FileExistsError("run artifacts already exist; use --resume or choose a new run ID")

    attempt_budget = make_http_attempt_budget(
        max_attempts=args.max_http_attempts,
        ledger_path=paths["http_attempt_ledger"],
        resume=args.resume,
    )
    call_json = make_call_json(attempt_budget=attempt_budget)
    description = _dataset_description(args.dataset)
    template, template_path, template_origin, template_calls = _load_or_generate_template(
        dataset=args.dataset,
        dataset_description=description,
        source_path=template_source,
        generated_path=paths["generated_template"],
        call_json=call_json,
        resume=args.resume,
    )
    template_hash = hashlib.sha256(
        json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    client = _make_qdrant_client(url=args.qdrant_url, path=args.qdrant_path)
    stores = {
        size: QdrantHicStore(
            model_id="BAAI/bge-m3",
            collection_name=collection_names[size],
            client=client,
        )
        for size in memory_sizes
    }
    try:
        for size, collection_name in collection_names.items():
            exists = client.collection_exists(collection_name)
            if args.resume and not exists:
                raise RuntimeError(f"resume collection K={size} is missing from Qdrant")
            if not args.resume and exists:
                raise FileExistsError(
                    f"Qdrant collection {collection_name} already exists; choose a new run ID"
                )
        for store in stores.values():
            store.ensure_collection()
        largest_memory_size = memory_sizes[-1]
        training_checkpoint = None
        if args.resume:
            training_checkpoint = json.loads(
                paths["training_checkpoint"].read_text(encoding="utf-8")
            )
        training_states = (training_checkpoint or {}).get("cases", {})
        training_is_complete = bool(training_checkpoint) and all(
            all(
                training_states.get(case["case_id"], {}).get(key)
                for key in ("summary", "forecast", "correction", "embedding", "ingested")
            )
            for case in training_cases
        )
        if args.resume and training_is_complete:
            training_report = _training_report_from_checkpoint(
                checkpoint_path=paths["training_checkpoint"],
                cases=training_cases,
                dataset=args.dataset,
                run_id=args.run_id,
                max_calls=args.max_calls,
            )
        else:
            training_report = build_completed_memories(
                cases=training_cases,
                template=template,
                store=stores[largest_memory_size],
                embed_summary=lambda summary, *, is_query: _embed_summary(
                    hic.get_embedder(), summary, is_query=is_query
                ),
                call_json=call_json,
                dataset_name=args.dataset,
                dataset_description=description,
                run_id=f"{args.run_id}_train",
                checkpoint_path=paths["training_checkpoint"],
                max_calls=args.max_calls,
                retry_pending=args.retry_pending,
            )
        _seed_nested_memories_from_checkpoint(
            stores=stores,
            training_cases=training_cases,
            checkpoint_path=paths["training_checkpoint"],
            run_id=args.run_id,
            dataset=args.dataset,
            allow_additional_points=args.resume,
        )
        remaining_eval_call_budget = (
            args.max_calls
            - template_calls
            - training_report["logical_deepseek_calls"]
        )
        if remaining_eval_call_budget < 0:
            raise RuntimeError("training phase exceeded the overall logical-call budget")
        before_ids = {
            size: _collection_case_ids(client, collection_names[size])
            for size in memory_sizes
        }
        if args.resume and evaluation_checkpoint_available:
            evaluation_checkpoint = json.loads(
                paths["evaluation_checkpoint"].read_text(encoding="utf-8")
            )
            _validate_resume_memory_ids(
                actual_ids_by_size=before_ids,
                training_case_ids=plan["training_case_ids"],
                evaluation_checkpoint=evaluation_checkpoint,
                validation_case_ids=plan["validation_case_ids"],
                memory_sizes=memory_sizes,
            )
        else:
            for size in memory_sizes:
                expected = set(plan["training_case_ids"][:size])
                if before_ids[size] != expected:
                    raise RuntimeError(f"K={size} collection does not match its nested training prefix")

        embedder = hic.get_embedder()
        validation_report = run_prequential_evaluation(
            windows=validation_windows,
            template=template,
            stores=stores,
            embed_summary=lambda summary, is_query: _embed_summary(
                embedder, summary, is_query=is_query
            ),
            call_json=call_json,
            embedding_model="BAAI/bge-m3",
            dataset_name=args.dataset,
            dataset_description=description,
            run_id=args.run_id,
            checkpoint_path=paths["evaluation_checkpoint"],
            max_calls=remaining_eval_call_budget,
            retry_pending=args.retry_pending,
        )
        baselines_by_window = [_baseline_metrics(window) for window in validation_windows]
        final_ids = {
            size: _collection_case_ids(client, collection_names[size])
            for size in memory_sizes
        }
        validation_ids = set(plan["validation_case_ids"])
        if any(not validation_ids.issubset(final_ids[size]) for size in memory_sizes):
            raise RuntimeError("completed validation case was not appended to every Qdrant arm")
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "completed_validation_pilot",
            "plan": plan,
            "template": {
                "origin": template_origin,
                "path": str(template_path),
                "sha256": template_hash,
                "top_level_keys": list(template),
            },
            "embedding": {
                "model": "BAAI/bge-m3",
                "dimensions": 1024,
                "max_length": getattr(embedder, "max_length", hic.BGE_MAX_LENGTH),
                "pooling": "CLS token",
                "normalization": "L2",
                "query_instruction": hic.BGE_QUERY_INSTRUCTION,
                "truncation": "right truncation to the configured effective max length",
            },
            "qdrant": {
                "url": args.qdrant_url,
                "path": str(args.qdrant_path) if args.qdrant_path else None,
                "storage_mode": "http_service" if args.qdrant_url else "embedded_local",
                "distance": "cosine",
                "top_k_per_query": 1,
                "collections": {str(key): value for key, value in collection_names.items()},
                "counts_before_validation": {
                    str(size): len(before_ids[size]) for size in memory_sizes
                },
                "counts_after_validation": {
                    str(size): len(final_ids[size]) for size in memory_sizes
                },
                "training_case_ids_before_validation": {
                    str(size): sorted(before_ids[size]) for size in memory_sizes
                },
            },
            "training": training_report,
            "validation": validation_report,
            "numeric_baselines_by_window": baselines_by_window,
            "validation_candidate_ranking": _candidate_winners(
                validation_report, baselines_by_window
            ),
            "call_accounting": {
                "template_generation_logical_calls": template_calls,
                "training_logical_calls": training_report["logical_deepseek_calls"],
                "validation_logical_calls": validation_report["logical_deepseek_calls"],
                "total_logical_calls": (
                    template_calls
                    + training_report["logical_deepseek_calls"]
                    + validation_report["logical_deepseek_calls"]
                ),
                "physical_http_attempts": attempt_budget.attempts_used,
                "max_physical_http_attempts": attempt_budget.max_attempts,
                "logical_call_count_is_not_physical_retry_count": True,
            },
            "selection_caveat": (
                f"This pilot uses {len(validation_windows)} chronological validation windows. "
                "Its lowest observed MAE/RMSE is a validation result, not a globally "
                "optimal workflow. Outer-test windows and targets were not evaluated."
            ),
        }
        _atomic_json(paths["report"], report)
        print(json.dumps({
            "status": report["status"],
            "run_id": args.run_id,
            "training_case_ids": plan["training_case_ids"],
            "validation_case_ids": plan["validation_case_ids"],
            "retrieval_memory_sizes": plan["retrieval_memory_sizes"],
            "validation_candidate_ranking": report["validation_candidate_ranking"],
            "qdrant_counts_after_validation": report["qdrant"]["counts_after_validation"],
            "logical_calls": report["call_accounting"]["total_logical_calls"],
            "physical_attempts": report["call_accounting"]["physical_http_attempts"],
            "report_path": str(paths["report"]),
        }, ensure_ascii=False, indent=2))
        return report
    finally:
        client.close()


def continue_native_hic_validation(args):
    """Score later validation windows using the completed pilot's exact memory."""
    if args.continue_from_run_id == args.run_id:
        raise ValueError("continuation run ID must differ from the source pilot run ID")
    source_paths = _paths(args.continue_from_run_id)
    if not source_paths["report"].is_file():
        raise FileNotFoundError("source pilot report does not exist")
    source_report = json.loads(source_paths["report"].read_text(encoding="utf-8"))
    if source_report.get("status") != "completed_validation_pilot":
        raise ValueError("source run must be a completed validation pilot")
    source_plan = source_report.get("plan", {})
    if source_plan.get("dataset") != args.dataset or source_plan.get("horizon") != args.horizon:
        raise ValueError("continuation dataset/horizon must match the source pilot")
    if source_plan.get("seed") != args.seed or source_plan.get("training_count") != args.train_count:
        raise ValueError("continuation seed/training count must match the source pilot")

    template_meta = source_report.get("template", {})
    template_path = Path(template_meta.get("path", ""))
    template = _read_template(template_path)
    template_hash = _report_template_hash(template)
    if template_hash != template_meta.get("sha256"):
        raise ValueError("source pilot template hash does not match its report")

    plan, training_cases, validation_windows, _ = prepare_pilot_inputs(
        dataset=args.dataset,
        horizon=args.horizon,
        train_count=args.train_count,
        validation_count=args.validation_count,
        validation_start_index=args.validation_start_index,
        seed=args.seed,
        data_dir=args.data_dir,
        template_available=True,
    )
    _validate_continuation_plan(source_plan, plan)
    source_training = [case["case_id"] for case in training_cases]
    if source_training != source_plan.get("training_case_ids"):
        raise ValueError("reconstructed source training cases do not match the saved pilot")

    source_qdrant = source_report.get("qdrant", {})
    if source_qdrant.get("url") != args.qdrant_url:
        raise ValueError("continuation must use the source pilot's Qdrant URL")
    source_path = source_qdrant.get("path")
    current_path = str(Path(args.qdrant_path).resolve()) if args.qdrant_path else None
    saved_path = str(Path(source_path).resolve()) if source_path else None
    if saved_path != current_path:
        raise ValueError("continuation must use the source pilot's Qdrant storage path")
    collection_names = {
        int(key): value for key, value in source_qdrant.get("collections", {}).items()
    }
    if set(collection_names) != set(DEFAULT_MEMORY_SIZES):
        raise ValueError("source pilot does not contain the expected K=1 and K=10 collections")

    paths = _paths(args.run_id)
    validation_calls = args.validation_count * _logical_calls_per_window(DEFAULT_MEMORY_SIZES)
    plan = {
        **plan,
        "run_id": args.run_id,
        "continuation_of_run_id": args.continue_from_run_id,
        "reuses_source_template": True,
        "source_template_path": str(template_path),
        "source_template_sha256": template_hash,
        "reuses_training_cases": True,
        "reuses_qdrant_collections": {str(k): v for k, v in collection_names.items()},
        "template_generation_logical_calls": 0,
        "training_logical_calls": 0,
        "validation_logical_calls": validation_calls,
        "estimated_logical_calls": validation_calls,
        "max_logical_calls": args.max_calls,
        "max_http_attempts": args.max_http_attempts,
        "report_path": str(paths["report"]),
        "evaluation_checkpoint_path": str(paths["evaluation_checkpoint"]),
        "http_attempt_ledger_path": str(paths["http_attempt_ledger"]),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if args.max_http_attempts is None:
        raise ValueError("--max-http-attempts is required for live API calls")
    if args.max_calls < validation_calls:
        raise ValueError(
            f"continuation needs {validation_calls} logical calls; configured max_calls={args.max_calls}"
        )
    if args.api_env:
        load_dotenv(args.api_env, override=False)
    hic.DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DeepSeek key is unavailable; provide --api-env or set DEEPSEEK_API_KEY. "
            "The key value was not displayed."
        )

    if paths["report"].exists():
        raise FileExistsError("continuation report already exists; choose a new run ID")
    if args.resume:
        if not paths["evaluation_checkpoint"].is_file() or not paths["http_attempt_ledger"].is_file():
            raise FileNotFoundError("--resume requires matching evaluation and HTTP-attempt checkpoints")
    elif paths["evaluation_checkpoint"].exists() or paths["http_attempt_ledger"].exists():
        raise FileExistsError("continuation artifacts already exist; use --resume or a new run ID")

    attempt_budget = make_http_attempt_budget(
        max_attempts=args.max_http_attempts,
        ledger_path=paths["http_attempt_ledger"],
        resume=args.resume,
    )
    call_json = make_call_json(attempt_budget=attempt_budget)
    client = _make_qdrant_client(url=args.qdrant_url, path=args.qdrant_path)
    stores = {
        size: QdrantHicStore(
            model_id="BAAI/bge-m3",
            collection_name=collection_names[size],
            client=client,
        )
        for size in DEFAULT_MEMORY_SIZES
    }
    try:
        if any(not client.collection_exists(name) for name in collection_names.values()):
            raise RuntimeError("one or more source pilot Qdrant collections are missing")
        prior_validation_ids = set(source_plan.get("validation_case_ids", []))
        source_training_list = source_plan.get("training_case_ids", [])
        source_training_ids = set(source_training_list)
        before_ids = {
            size: _collection_case_ids(client, collection_names[size])
            for size in DEFAULT_MEMORY_SIZES
        }
        if not source_training_ids.issubset(before_ids[10]):
            raise RuntimeError("source K=10 collection is missing fitted training cases")
        if not source_training_list or source_training_list[0] not in before_ids[1]:
            raise RuntimeError("source K=1 collection is missing its seeded training case")
        if not prior_validation_ids.issubset(before_ids[1]) or not prior_validation_ids.issubset(before_ids[10]):
            raise RuntimeError("source validation case was not ingested into both retrieval arms")
        next_ids = set(plan["validation_case_ids"])
        already_present = next_ids & (before_ids[1] | before_ids[10])
        if already_present and not args.resume:
            raise RuntimeError("new validation windows already exist in Qdrant; refusing duplicate scoring")

        embedder = hic.get_embedder()
        continuation_report = run_prequential_evaluation(
            windows=validation_windows,
            template=template,
            stores=stores,
            embed_summary=lambda summary, is_query: _embed_summary(
                embedder, summary, is_query=is_query
            ),
            call_json=call_json,
            embedding_model="BAAI/bge-m3",
            dataset_name=args.dataset,
            dataset_description=_dataset_description(args.dataset),
            run_id=args.run_id,
            checkpoint_path=paths["evaluation_checkpoint"],
            max_calls=args.max_calls,
            retry_pending=args.retry_pending,
        )
        new_baselines = [_baseline_metrics(window) for window in validation_windows]
        final_ids = {
            size: _collection_case_ids(client, collection_names[size])
            for size in DEFAULT_MEMORY_SIZES
        }
        if not next_ids.issubset(final_ids[1]) or not next_ids.issubset(final_ids[10]):
            raise RuntimeError("completed continuation case was not appended to both Qdrant arms")

        all_windows = [
            *source_report["validation"]["windows"],
            *continuation_report["windows"],
        ]
        all_baselines = [
            *source_report["numeric_baselines_by_window"],
            *new_baselines,
        ]
        cumulative = aggregate_validation_windows(all_windows)
        ranking = _candidate_winners({"windows": all_windows}, all_baselines)
        source_call_accounting = source_report.get("call_accounting", {})
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "completed_validation_continuation",
            "continuation_of_run_id": args.continue_from_run_id,
            "plan": plan,
            "template": template_meta,
            "embedding": source_report.get("embedding"),
            "qdrant": {
                **source_qdrant,
                "counts_before_continuation": {
                    str(size): len(before_ids[size]) for size in DEFAULT_MEMORY_SIZES
                },
                "counts_after_continuation": {
                    str(size): len(final_ids[size]) for size in DEFAULT_MEMORY_SIZES
                },
            },
            "continuation_validation": continuation_report,
            "numeric_baselines_new_windows": new_baselines,
            "cumulative_validation": {
                "windows": all_windows,
                "case_ids": [window["case_id"] for window in all_windows],
                "metrics": cumulative,
                "numeric_baselines_by_window": all_baselines,
                "candidate_ranking": ranking,
            },
            "call_accounting": {
                "continuation_logical_calls": continuation_report["logical_deepseek_calls"],
                "continuation_physical_http_attempts": attempt_budget.attempts_used,
                "continuation_max_physical_http_attempts": attempt_budget.max_attempts,
                "source_pilot_logical_calls": source_call_accounting.get("total_logical_calls"),
                "source_pilot_physical_http_attempts": source_call_accounting.get("physical_http_attempts"),
                "combined_source_and_continuation_logical_calls": (
                    (source_call_accounting.get("total_logical_calls") or 0)
                    + continuation_report["logical_deepseek_calls"]
                ),
            },
            "selection_caveat": (
                "This continuation adds chronological validation windows without touching "
                "outer-test targets. Rankings are exploratory and specific to this dataset; "
                "three windows are still too few to claim a robust or globally optimal method."
            ),
        }
        _atomic_json(paths["report"], report)
        print(json.dumps({
            "status": report["status"],
            "run_id": args.run_id,
            "continuation_of_run_id": args.continue_from_run_id,
            "new_validation_case_ids": plan["validation_case_ids"],
            "cumulative_validation_case_ids": report["cumulative_validation"]["case_ids"],
            "cumulative_candidate_ranking": ranking,
            "qdrant_counts_after_continuation": report["qdrant"]["counts_after_continuation"],
            "logical_calls": continuation_report["logical_deepseek_calls"],
            "physical_attempts": attempt_budget.attempts_used,
            "report_path": str(paths["report"]),
        }, ensure_ascii=False, indent=2))
        return report
    finally:
        client.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="native_agri_h12_20261018_01")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument(
        "--memory-sizes", type=_memory_size_list, default=DEFAULT_MEMORY_SIZES,
        help="comma-separated nested Qdrant memory sizes (default: 1,10)",
    )
    parser.add_argument("--validation-count", type=int, default=DEFAULT_VALIDATION_COUNT)
    parser.add_argument("--validation-start-index", type=int, default=0)
    parser.add_argument("--continue-from-run-id", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--template-path", type=Path, default=None)
    parser.add_argument("--api-env", type=Path, default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument(
        "--qdrant-path", type=Path,
        default=ROOT / "qdrant_native_hic_pilot",
        help="persistent embedded Qdrant directory; ignored when --qdrant-url is supplied",
    )
    parser.add_argument("--max-calls", type=_positive_int, default=None)
    parser.add_argument("--max-http-attempts", type=_positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform provider requests and Qdrant writes (requires explicit call caps)",
    )
    return parser


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _memory_size_list(value):
    try:
        parts = value.split(",")
        if any(not item.strip() for item in parts):
            raise ValueError
        sizes = tuple(int(item.strip()) for item in parts)
    except (AttributeError, ValueError):
        raise argparse.ArgumentTypeError(
            "use comma-separated positive integers, e.g. 1,10,24"
        ) from None
    if not sizes or any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("memory sizes must be unique positive integers")
    return tuple(sorted(sizes))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("--execute and --dry-run cannot be used together")
    if not args.execute:
        args.dry_run = True
    else:
        if args.max_calls is None:
            raise ValueError("--max-calls is required with --execute")
        if args.max_http_attempts is None:
            raise ValueError("--max-http-attempts is required with --execute")
    run_native_hic_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
