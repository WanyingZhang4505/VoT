"""Recompute the frozen Phase20 HIC baseline on a support-based 16-config scope."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("mae", "mse", "rmse")
ARMS = {"baseline": "no_retrieval", "hic": "full_reasoning"}
COMPLETED_PLUS_SUPPORT_PROTOCOL = "hic_phase20_completed12_plus_support4_scope_v2"


def _direction(value: float, tolerance: float = 1e-9) -> str:
    if value > tolerance:
        return "improved"
    if value < -tolerance:
        return "worsened"
    return "tie"


def select_configurations(configurations, *, target_count=16):
    """Keep the most-supported configs while preserving every dataset family.

    Drop order is outcome-blind: fewest validation windows, then fewest fit
    memories, then dataset and horizon for deterministic tie-breaking.
    """
    configurations = list(configurations)
    if not isinstance(target_count, int) or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    if target_count > len(configurations):
        raise ValueError("target_count cannot exceed the available config count")

    keys = [(row["dataset"], int(row["horizon"])) for row in configurations]
    if len(set(keys)) != len(keys):
        raise ValueError("configuration dataset/horizon keys must be unique")
    family_counts = defaultdict(int)
    for row in configurations:
        family_counts[row["dataset"]] += 1
    if target_count < len(family_counts):
        raise ValueError("target_count must retain at least one config per family")

    candidates = sorted(
        configurations,
        key=lambda row: (
            int(row["scopes"]["all"]["windows"]),
            int(row["fit_memory_count"]),
            row["dataset"],
            int(row["horizon"]),
        ),
    )
    excluded = set()
    for row in candidates:
        if len(configurations) - len(excluded) <= target_count:
            break
        family = row["dataset"]
        if family_counts[family] <= 1:
            continue
        excluded.add((family, int(row["horizon"])))
        family_counts[family] -= 1

    if len(configurations) - len(excluded) != target_count:
        raise ValueError("could not reach target count without dropping a family")
    return [row for row in configurations if (row["dataset"], int(row["horizon"])) not in excluded]


def select_completed_plus_supported_configurations(
    configurations, *, completed_keys, target_count=16
):
    """Retain completed configs, then add equal numbers from uncovered families.

    Additions are chosen without looking at HIC outcomes: within each uncovered
    family, rank by validation windows, fit-memory members, then horizon.
    """
    configurations = list(configurations)
    keys = [(row["dataset"], int(row["horizon"])) for row in configurations]
    if len(keys) != len(set(keys)):
        raise ValueError("configuration dataset/horizon keys must be unique")
    if not isinstance(target_count, int) or target_count < 1 or target_count > len(configurations):
        raise ValueError("target_count must be a positive integer within the available configs")

    completed_keys = set(completed_keys)
    config_by_key = dict(zip(keys, configurations))
    if len(completed_keys) != target_count - 4:
        raise ValueError("this scope requires exactly twelve completed configs and four additions")
    unknown = completed_keys - set(keys)
    if unknown:
        raise ValueError(f"completed configurations are not in the frozen source: {sorted(unknown)}")

    families = sorted({dataset for dataset, _horizon in keys})
    completed_families = {dataset for dataset, _horizon in completed_keys}
    uncovered_families = [dataset for dataset in families if dataset not in completed_families]
    addition_slots = target_count - len(completed_keys)
    if not uncovered_families or addition_slots % len(uncovered_families) != 0:
        raise ValueError("four additions must be allocated evenly across uncovered dataset families")
    per_family = addition_slots // len(uncovered_families)

    added_keys = []
    for family in uncovered_families:
        candidates = [
            row for row in configurations
            if row["dataset"] == family and (family, int(row["horizon"])) not in completed_keys
        ]
        candidates.sort(
            key=lambda row: (
                -int(row["scopes"]["all"]["windows"]),
                -int(row["fit_memory_count"]),
                int(row["horizon"]),
            )
        )
        if len(candidates) < per_family:
            raise ValueError(f"not enough candidate configs to cover {family}")
        added_keys.extend((family, int(row["horizon"])) for row in candidates[:per_family])

    selected_keys = completed_keys | set(added_keys)
    if len(selected_keys) != target_count:
        raise ValueError("completed plus added configurations do not reach the target count")
    selected = [row for row in configurations if (row["dataset"], int(row["horizon"])) in selected_keys]
    added = [config_by_key[key] for key in added_keys]
    return selected, added


def aggregate_origin_outcomes(rows):
    """Average horizon-relative gains within each (dataset, case_id) origin."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["case_id"])].append(row)

    origins = []
    for (dataset, case_id), members in sorted(grouped.items()):
        mae_gain = sum(float(row["mae_gain_pct"]) for row in members) / len(members)
        mse_gain = sum(float(row["mse_gain_pct"]) for row in members) / len(members)
        origins.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "horizons": sorted(int(row["horizon"]) for row in members),
                "mae_gain_pct": mae_gain,
                "mae_direction": _direction(mae_gain),
                "mse_gain_pct": mse_gain,
                "mse_direction": _direction(mse_gain),
            }
        )

    def counts(metric):
        result = {name: 0 for name in ("improved", "worsened", "tie")}
        for origin in origins:
            result[origin[f"{metric}_direction"]] += 1
        return result

    return {
        "origin_count": len(origins),
        "mae_directions": counts("mae"),
        "mse_directions": counts("mse"),
        "origins": origins,
    }


def _metrics(values, truth):
    if len(values) != len(truth) or not values:
        raise ValueError("forecast and truth must be nonempty and have equal lengths")
    errors = [float(pred) - float(actual) for pred, actual in zip(values, truth)]
    if not all(math.isfinite(value) for value in errors):
        raise ValueError("forecast and truth values must be finite")
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error * error for error in errors) / len(errors)
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "n": len(errors)}


def _gain_pct(baseline, hic):
    if baseline == 0:
        if hic == 0:
            return 0.0
        raise ValueError("relative gain is undefined when baseline metric is zero")
    return 100.0 * (baseline - hic) / baseline


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scope_audit(
    final_results_path,
    *,
    project_root,
    target_count=16,
    completed_configuration_keys=None,
):
    final_results_path = Path(final_results_path)
    project_root = Path(project_root)
    source_doc = json.loads(final_results_path.read_text(encoding="utf-8"))
    all_configs = source_doc["configurations"]
    if len(all_configs) != 20:
        raise ValueError(f"expected frozen 20-config source, found {len(all_configs)}")
    if completed_configuration_keys is None:
        selected = select_configurations(all_configs, target_count=target_count)
        added = []
        protocol = "hic_phase20_support_based_16_config_scope_v1"
    else:
        selected, added = select_completed_plus_supported_configurations(
            all_configs,
            completed_keys=completed_configuration_keys,
            target_count=target_count,
        )
        protocol = COMPLETED_PLUS_SUPPORT_PROTOCOL
    selected_keys = {(row["dataset"], int(row["horizon"])) for row in selected}
    completed_keys = set(completed_configuration_keys or ())
    added_order = [(row["dataset"], int(row["horizon"])) for row in added]
    added_keys = set(added_order)
    dropped = [row for row in all_configs if (row["dataset"], int(row["horizon"])) not in selected_keys]

    per_config = []
    task_rows = []
    family_values = defaultdict(lambda: defaultdict(list))
    total_target_points = 0
    total_fit_members = 0
    total_validation_windows = 0
    for config in selected:
        report_rel = Path(str(config["source_report"]).replace("\\", "/"))
        report_path = project_root / report_rel
        if not report_path.is_file():
            raise FileNotFoundError(f"missing frozen configuration report: {report_path}")
        report_hash = _sha256(report_path)
        if report_hash != config["source_sha256"]:
            raise ValueError(f"source hash mismatch for {config['dataset']}/H{config['horizon']}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("outer_test_targets_read") is not False:
            raise ValueError(f"outer-test boundary is not clean for {config['dataset']}/H{config['horizon']}")
        windows = report.get("validation_windows")
        if not isinstance(windows, list) or len(windows) != config["scopes"]["all"]["windows"]:
            raise ValueError(f"validation window count mismatch for {config['dataset']}/H{config['horizon']}")

        arrays = {arm: [] for arm in ARMS.values()}
        truths = []
        for window in windows:
            truth = window.get("truth")
            forecasts = window.get("forecasts", {})
            if window.get("case_id") != window.get("case", {}).get("case_id"):
                raise ValueError("case_id mismatch between validation row and input case")
            if window.get("case", {}).get("dataset") != config["dataset"]:
                raise ValueError("dataset mismatch in validation row")
            for arm in ARMS.values():
                if arm not in forecasts:
                    raise ValueError(f"missing {arm} forecast")
                arrays[arm].extend(forecasts[arm])
            truths.extend(truth)
            window_metrics = {
                "baseline": _metrics(forecasts[ARMS["baseline"]], truth),
                "hic": _metrics(forecasts[ARMS["hic"]], truth),
            }
            task_rows.append(
                {
                    "dataset": config["dataset"],
                    "horizon": int(config["horizon"]),
                    "case_id": window["case_id"],
                    "development_role": window.get("development_role"),
                    "target_points": len(truth),
                    "mae_gain_pct": _gain_pct(window_metrics["baseline"]["mae"], window_metrics["hic"]["mae"]),
                    "mse_gain_pct": _gain_pct(window_metrics["baseline"]["mse"], window_metrics["hic"]["mse"]),
                }
            )

        aggregate = {
            "baseline": _metrics(arrays[ARMS["baseline"]], truths),
            "hic": _metrics(arrays[ARMS["hic"]], truths),
        }
        gains = {
            metric: _gain_pct(aggregate["baseline"][metric], aggregate["hic"][metric])
            for metric in METRICS
        }
        for metric, value in gains.items():
            family_values[config["dataset"]][metric].append(value)
        per_config.append(
            {
                "dataset": config["dataset"],
                "horizon": int(config["horizon"]),
                "validation_windows": len(windows),
                "target_points": len(truths),
                "fit_memory_members": int(config["fit_memory_count"]),
                "baseline_metrics": aggregate["baseline"],
                "full_hic_metrics": aggregate["hic"],
                "relative_gain_pct": gains,
                "mae_direction": _direction(gains["mae"]),
                "source_report": report_rel.as_posix(),
                "source_sha256": report_hash,
            }
        )
        total_target_points += len(truths)
        total_fit_members += int(config["fit_memory_count"])
        total_validation_windows += len(windows)

    families = {}
    for dataset, metrics in sorted(family_values.items()):
        families[dataset] = {
            f"{metric}_relative_gain_pct": sum(values) / len(values)
            for metric, values in sorted(metrics.items())
        }
        families[dataset]["configuration_count"] = sum(
            row["dataset"] == dataset for row in selected
        )
    family_macro = {
        f"{metric}_relative_gain_pct": sum(
            entry[f"{metric}_relative_gain_pct"] for entry in families.values()
        ) / len(families)
        for metric in METRICS
    }
    origin_audit = aggregate_origin_outcomes(task_rows)
    dropped_rows = []
    for config in dropped:
        dropped_rows.append(
            {
                "dataset": config["dataset"],
                "horizon": int(config["horizon"]),
                "validation_windows": int(config["scopes"]["all"]["windows"]),
                "fit_memory_members": int(config["fit_memory_count"]),
                "reason": (
                    "outside the completed-12 plus four-addition active scope"
                    if protocol == COMPLETED_PLUS_SUPPORT_PROTOCOL
                    else "lowest validation-window support; tie-break by lower fit-memory count"
                ),
            }
        )

    config_ref = lambda key: {"dataset": key[0], "horizon": key[1]}
    if protocol == COMPLETED_PLUS_SUPPORT_PROTOCOL:
        selection_rule = {
            "target_configuration_count": target_count,
            "completed_configuration_count": len(completed_keys),
            "added_configuration_count": len(added_keys),
            "performance_metrics_used_for_selection": False,
            "selection_timing": "The completed 12 are frozen outputs; the 4 additions were selected after the 20-config benchmark had been reviewed, so all resulting comparisons are post-hoc development evidence.",
            "rule": "Retain all 12 completed Agriculture, Climate, and Economy_sort configurations. Add the two highest-support Energy configurations and the two highest-support SocialGood configurations; ties use fit-memory count then ascending horizon.",
            "completed_configurations": [config_ref(key) for key in sorted(completed_keys)],
            "added_configurations": [config_ref(key) for key in added_order],
            "selected_configurations": [
                config_ref((row["dataset"], int(row["horizon"]))) for row in selected
            ],
            "dropped": dropped_rows,
        }
    else:
        selection_rule = {
            "target_configuration_count": target_count,
            "performance_metrics_used_for_selection": False,
            "selection_timing": "Post-hoc scope amendment after the 20-config Phase20 results were already available; development evidence only, not preregistered confirmation.",
            "rule": "Drop configs in ascending validation-window count, then ascending fit-memory count, then dataset and horizon; never drop the last config of a dataset family.",
            "dropped": dropped_rows,
        }

    return {
        "protocol": protocol,
        "selection_rule": selection_rule,
        "coverage": {
            "source_configuration_count": len(all_configs),
            "selected_configuration_count": len(selected),
            "validation_case_horizon_memberships": total_validation_windows,
            "validation_target_point_memberships": total_target_points,
            "fit_memory_memberships": total_fit_members,
            "unique_validation_origins": origin_audit["origin_count"],
            "dataset_families": sorted(families),
            "family_configuration_counts": {name: value["configuration_count"] for name, value in families.items()},
        },
        "baseline_metrics": {
            "config_mae_directions": {
                name: sum(row["mae_direction"] == name for row in per_config)
                for name in ("improved", "worsened", "tie")
            },
            "equal_family_macro_relative_gain_pct": family_macro,
            "per_family": families,
            "unique_origin_mae_directions": origin_audit["mae_directions"],
            "unique_origin_mse_directions": origin_audit["mse_directions"],
        },
        "origin_results": origin_audit["origins"],
        "config_results": per_config,
        "provenance": {
            "source_results_file": final_results_path.name,
            "source_results_sha256": _sha256(final_results_path),
            "validation_truth_used": True,
            "outer_test_targets_read": False,
            "interpretation": "Reaggregation of already-exposed Phase20 validation results; development evidence only, not an independent confirmation set.",
        },
    }


def render_markdown(audit):
    coverage = audit["coverage"]
    baseline = audit["baseline_metrics"]
    selection = audit["selection_rule"]
    lines = [
        "# HIC 16-config scope: frozen Phase20 baseline reaggregation",
        "",
        "This report reaggregates saved Phase20 predictions and validation truth; it does not call a model or change the frozen 20-config source.",
        "The rule uses support counts rather than model performance and retains all five dataset families. However, the scope change is post-hoc: the original 20-config results had already been reviewed, so this remains development evidence, not independent confirmation.",
        "",
        "## Scope",
        "",
        f"- Selected configs: {coverage['selected_configuration_count']} / {coverage['source_configuration_count']}",
        f"- Validation case-horizon memberships: {coverage['validation_case_horizon_memberships']}",
        f"- Target-point memberships: {coverage['validation_target_point_memberships']}",
        f"- Fit-memory memberships: {coverage['fit_memory_memberships']}",
        f"- Unique dataset-case origins: {coverage['unique_validation_origins']}",
        f"- Dataset families: {', '.join(coverage['dataset_families'])}",
        f"- Config counts by family: {coverage['family_configuration_counts']}",
        "",
        "### Selection rule",
        "",
        selection["rule"],
        "",
        "| Excluded config | Validation windows | Fit-memory members |",
        "|---|---:|---:|",
    ]
    for row in selection["dropped"]:
        lines.append(f"| {row['dataset']} H{row['horizon']} | {row['validation_windows']} | {row['fit_memory_members']} |")
    lines.extend(
        [
            "",
            "## Frozen baseline on the selected 16 configs",
            "",
            f"- Config-level MAE directions: {baseline['config_mae_directions']}",
            f"- Unique-origin MAE directions: {baseline['unique_origin_mae_directions']}",
            f"- Equal-family macro relative gains (%): {baseline['equal_family_macro_relative_gain_pct']}",
            "  - Positive means Full HIC has lower error than No Retrieval.",
            "",
            "| Dataset | Configs | MAE gain | MSE gain | RMSE gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, result in baseline["per_family"].items():
        lines.append(
            f"| {dataset} | {result['configuration_count']} | "
            f"{result['mae_relative_gain_pct']:.3f}% | {result['mse_relative_gain_pct']:.3f}% | "
            f"{result['rmse_relative_gain_pct']:.3f}% |"
        )
    if selection.get("completed_configurations"):
        lines.extend(
            [
                "",
                "### Twelve completed configurations retained",
                "",
            ]
        )
        lines.extend(
            f"- {row['dataset']} H{row['horizon']}"
            for row in selection["completed_configurations"]
        )
        lines.extend(["", "### Four additional configurations", ""])
        lines.extend(
            f"- {row['dataset']} H{row['horizon']}"
            for row in selection["added_configurations"]
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- The 16-config scope retains the five families but has unequal config counts; family-level macro results give each family equal weight and average configs equally within family.",
            "- The selection rule did not rank configs by HIC performance, but was applied after the original 20-config outcomes were visible; treat the reduced-scope metrics as post-hoc development evidence.",
            "- Task memberships are overlapping horizons and are not independent samples. Unique-origin direction first averages selected horizon-relative gains within each `(dataset, case_id)`.",
            "- These are previously exposed Phase20 validation results, repurposed as development evidence. They cannot be called independent confirmation.",
            "- The original 20-config result files remain unchanged and can be reaggregated later.",
            "- This report does not claim the new vNext HIC has improved validation performance; it only redefines the scope and restates the frozen baseline.",
            "",
            f"Source SHA-256: `{audit['provenance']['source_results_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_audit(audit, *, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")

