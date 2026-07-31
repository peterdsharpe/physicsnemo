#!/usr/bin/env python3
"""Reduce the preregistered SHIFT-SUV validation-only adaptation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ARCHITECTURES = ("mesh_transformer", "geotransolver")
INITIALIZATIONS = ("scratch", "pretrained")
FAMILIES = ("estate", "fastback")
SAMPLE_COUNTS = (64, 128)
LEARNING_RATES = ((3e-4, "3e-4"), (1e-3, "1e-3"), (3e-3, "3e-3"))
REGISTERED_EPOCHS = (0, 10, 20, 30, 40, 50, 60, 69)
EVALUATION_SEEDS = (20_260_730, 20_260_731)
N_VALIDATION = 99
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_730


def _run_id(
    architecture: str,
    initialization: str,
    family: str,
    sample_count: int,
    learning_rate_name: str,
) -> str:
    return (
        f"pilot_{architecture}_{initialization}_{family}_n{sample_count}"
        f"_lr{learning_rate_name}_seed42"
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):
                    # A time-limited segment may be killed during its final
                    # append. Only an unterminated tail record is discardable.
                    break
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield record


def _load_run(
    run_root: Path,
    terminal_event: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics_path = run_root / "metrics.jsonl"
    train_log = run_root / "train.log"
    if not metrics_path.exists():
        eligible_epochs = (
            {
                int(epoch)
                for epoch in terminal_event["eligible_registered_epoch_indices"]
            }
            if terminal_event is not None
            else set(REGISTERED_EPOCHS)
        )
        return {
            "run_id": run_root.name,
            "completed": False,
            "non_finite": terminal_event is not None,
            "completed_in_latest_segment": False,
            "non_finite_in_latest_segment": False,
            "terminal_event": terminal_event,
            "eligible_registered_epochs": eligible_epochs,
            "summaries": {},
            "steps": {},
            "missing": "metrics.jsonl",
        }

    summaries: dict[int, dict[str, Any]] = {}
    steps: dict[int, dict[int, dict[str, Any]]] = {}
    pending_steps: dict[int, dict[int, dict[str, Any]]] = {}
    for record in _read_jsonl(metrics_path):
        phase = record.get("phase")
        epoch = int(record.get("epoch", -1))
        if phase == "val_summary" and epoch in REGISTERED_EPOCHS:
            summaries[epoch] = record
            steps[epoch] = pending_steps.get(epoch, {}).copy()
        elif phase == "val_step":
            if epoch in REGISTERED_EPOCHS:
                val_step = int(record["val_step"])
                if val_step == 0:
                    pending_steps[epoch] = {}
                pending_steps.setdefault(epoch, {})[val_step] = record

    train_text = train_log.read_text(errors="replace") if train_log.exists() else ""
    completed_in_latest_segment = "Training completed!" in train_text
    non_finite_in_latest_segment = "loss guard triggered" in train_text
    if terminal_event is not None:
        eligible_epochs = {
            int(epoch) for epoch in terminal_event["eligible_registered_epoch_indices"]
        }
        if not eligible_epochs <= set(REGISTERED_EPOCHS):
            raise ValueError(
                f"{run_root.name}: terminal ledger contains unregistered epochs "
                f"{sorted(eligible_epochs - set(REGISTERED_EPOCHS))}"
            )
        summaries = {
            epoch: summary
            for epoch, summary in summaries.items()
            if epoch in eligible_epochs
        }
        steps = {
            epoch: epoch_steps
            for epoch, epoch_steps in steps.items()
            if epoch in eligible_epochs
        }
        completed = False
        non_finite = True
    else:
        eligible_epochs = set(REGISTERED_EPOCHS)
        completed = completed_in_latest_segment
        non_finite = non_finite_in_latest_segment
    return {
        "run_id": run_root.name,
        "completed": completed,
        "non_finite": non_finite,
        "completed_in_latest_segment": completed_in_latest_segment,
        "non_finite_in_latest_segment": non_finite_in_latest_segment,
        "terminal_event": terminal_event,
        "eligible_registered_epochs": eligible_epochs,
        "summaries": summaries,
        "steps": steps,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_terminal_events(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported terminal-event schema")
    events: dict[str, dict[str, Any]] = {}
    for event in payload.get("events", []):
        run_id = event["run_id"]
        if run_id in events:
            raise ValueError(f"{path}: duplicate terminal event for {run_id}")
        events[run_id] = event
    return events


def _load_fixed_pass(
    fixed_root: Path,
    run_id: str,
    epoch: int,
    seed: int,
) -> dict[str, Any] | None:
    """Load one atomic fixed-validation pass, or its invalid marker."""
    completed_epoch = epoch + 1
    run_root = fixed_root / run_id
    stem = f"epoch_{completed_epoch}_seed_{seed}"
    invalid_path = run_root / f"{stem}.invalid.txt"
    if invalid_path.exists():
        return {
            "invalid": True,
            "reason": invalid_path.read_text(errors="replace").strip(),
            "path": str(invalid_path),
        }

    path = run_root / f"{stem}.jsonl"
    if not path.exists():
        return None

    config: dict[str, Any] | None = None
    complete = False
    summary: dict[str, Any] | None = None
    steps: dict[int, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        phase = record.get("phase")
        if phase == "fixed_eval_config":
            config = record
        elif phase == "val_step":
            if int(record.get("epoch", -1)) != epoch:
                raise ValueError(f"{path}: unexpected epoch in validation step")
            steps[int(record["val_step"])] = record
        elif phase == "val_summary":
            if int(record.get("epoch", -1)) != epoch:
                raise ValueError(f"{path}: unexpected epoch in validation summary")
            summary = record
        elif phase == "fixed_eval_complete":
            complete = True

    if config is None or summary is None or not complete:
        return None
    if config.get("run_id") != run_id:
        raise ValueError(f"{path}: run_id does not match its directory")
    if int(config.get("completed_epoch", -1)) != completed_epoch:
        raise ValueError(f"{path}: completed epoch does not match its filename")
    if int(config.get("evaluation_seed", -1)) != seed:
        raise ValueError(f"{path}: evaluation seed does not match its filename")
    if config.get("training_loader_iterated") is not False:
        raise ValueError(f"{path}: training loader was not declared untouched")
    if int(config.get("validation_cases", -1)) != N_VALIDATION:
        raise ValueError(f"{path}: expected {N_VALIDATION} validation cases")

    expected = set(range(N_VALIDATION))
    if set(steps) != expected:
        missing = sorted(expected - set(steps))
        extra = sorted(set(steps) - expected)
        raise ValueError(
            f"{path}: invalid validation steps; missing={missing}, extra={extra}"
        )
    for val_step, record in steps.items():
        for metric in ("pressure_l2", "wss_l2"):
            value = float(record[metric])
            if not np.isfinite(value):
                raise ValueError(
                    f"{path}: non-finite {metric} at validation step {val_step}"
                )
    return {
        "invalid": False,
        "summary": summary,
        "steps": steps,
        "path": str(path),
    }


def _load_fixed_runs(
    fixed_root: Path,
    runs: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
    fixed_runs: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    invalid_candidates: list[dict[str, Any]] = []

    for run_id, run in runs.items():
        summaries: dict[int, dict[str, float]] = {}
        steps: dict[int, dict[int, dict[str, float]]] = {}
        for epoch in sorted(run["eligible_registered_epochs"]):
            passes = [
                _load_fixed_pass(fixed_root, run_id, epoch, seed)
                for seed in EVALUATION_SEEDS
            ]
            invalid_passes = [
                (seed, result)
                for seed, result in zip(EVALUATION_SEEDS, passes)
                if result is not None and result["invalid"]
            ]
            if invalid_passes:
                invalid_candidates.append(
                    {
                        "run_id": run_id,
                        "completed_epoch": epoch + 1,
                        "invalid_passes": [
                            {
                                "evaluation_seed": seed,
                                "reason": result["reason"],
                                "path": result["path"],
                            }
                            for seed, result in invalid_passes
                        ],
                    }
                )
                continue
            if any(result is None for result in passes):
                missing_seeds = [
                    seed
                    for seed, result in zip(EVALUATION_SEEDS, passes)
                    if result is None
                ]
                incomplete.append(
                    f"{run_id}: fixed validation missing completed epoch "
                    f"{epoch + 1}, seeds={missing_seeds}"
                )
                continue

            valid_passes = [result for result in passes if result is not None]
            epoch_steps: dict[int, dict[str, float]] = {}
            for val_step in range(N_VALIDATION):
                epoch_steps[val_step] = {
                    metric: float(
                        np.mean(
                            [
                                result["steps"][val_step][metric]
                                for result in valid_passes
                            ]
                        )
                    )
                    for metric in ("pressure_l2", "wss_l2")
                }
            steps[epoch] = epoch_steps
            summaries[epoch] = {
                metric: float(
                    np.mean(
                        [
                            epoch_steps[val_step][metric]
                            for val_step in range(N_VALIDATION)
                        ]
                    )
                )
                for metric in ("pressure_l2", "wss_l2")
            }
        fixed_runs[run_id] = {
            "run_id": run_id,
            "summaries": summaries,
            "steps": steps,
        }

    return fixed_runs, sorted(set(incomplete)), invalid_candidates


def _quantiles(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def _case_vector(run: dict[str, Any], epoch: int, metric: str) -> np.ndarray:
    records = run["steps"].get(epoch, {})
    expected = set(range(N_VALIDATION))
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(
            f"{run['run_id']} epoch {epoch}: invalid validation steps; "
            f"missing={missing}, extra={extra}"
        )
    return np.asarray([records[index][metric] for index in range(N_VALIDATION)])


def _ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    point = float(numerator.mean() / denominator.mean())
    samples = numerator[bootstrap_indices].mean(axis=1) / denominator[
        bootstrap_indices
    ].mean(axis=1)
    return {
        "point": point,
        "paired_case_bootstrap_95_ci": _quantiles(samples),
    }


def _select_runs(
    runs: dict[str, dict[str, Any]],
    fixed_runs: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], list[str]]:
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    incomplete: list[str] = []
    for architecture, initialization, family, sample_count in product(
        ARCHITECTURES, INITIALIZATIONS, FAMILIES, SAMPLE_COUNTS
    ):
        candidates = []
        group_incomplete = False
        for learning_rate, learning_rate_name in LEARNING_RATES:
            run_id = _run_id(
                architecture,
                initialization,
                family,
                sample_count,
                learning_rate_name,
            )
            run = runs[run_id]
            fixed_run = fixed_runs[run_id]
            missing_epochs = [
                epoch
                for epoch in run["eligible_registered_epochs"]
                if epoch not in run["summaries"]
            ]
            if not run["completed"] and not run["non_finite"]:
                incomplete.append(
                    f"{run_id}: completed={run['completed']}, "
                    f"missing_completed_epochs="
                    f"{[epoch + 1 for epoch in missing_epochs]}"
                )
                group_incomplete = True
                continue
            if missing_epochs:
                incomplete.append(
                    f"{run_id}: missing online integrity summaries for "
                    f"completed epochs="
                    f"{[epoch + 1 for epoch in missing_epochs]}"
                )
                group_incomplete = True
                continue
            for epoch, summary in sorted(fixed_run["summaries"].items()):
                candidates.append(
                    (
                        float(summary["pressure_l2"]),
                        epoch,
                        learning_rate,
                        learning_rate_name,
                        run_id,
                        summary,
                        "completed" if run["completed"] else "non_finite",
                    )
                )
        if group_incomplete or not candidates:
            if not group_incomplete and not candidates:
                incomplete.append(
                    f"{architecture}/{initialization}/{family}/n{sample_count}: "
                    "no finite fixed-validation candidate"
                )
            continue
        pressure_l2, epoch, learning_rate, lr_name, run_id, summary, status = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        selected[(architecture, initialization, family, sample_count)] = {
            "run_id": run_id,
            "learning_rate": learning_rate,
            "learning_rate_name": lr_name,
            "completed_epoch": epoch + 1,
            "epoch_index": epoch,
            "pressure_l2": pressure_l2,
            "wss_l2": float(summary["wss_l2"]),
            "run_terminal_status": status,
        }
    return selected, sorted(set(incomplete))


def _reduce_complete(
    fixed_runs: dict[str, dict[str, Any]],
    selected: dict[tuple[str, str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    selected_vectors: dict[tuple[str, str, str, int, str], np.ndarray] = {}
    selected_records = []
    for key, choice in selected.items():
        architecture, initialization, family, sample_count = key
        run = fixed_runs[choice["run_id"]]
        for metric in ("pressure_l2", "wss_l2"):
            selected_vectors[(*key, metric)] = _case_vector(
                run, choice["epoch_index"], metric
            )
        selected_records.append(
            {
                "architecture": architecture,
                "initialization": initialization,
                "family": family,
                "sample_count": sample_count,
                **{
                    name: value
                    for name, value in choice.items()
                    if name != "epoch_index"
                },
            }
        )

    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, N_VALIDATION, size=(N_BOOTSTRAP, N_VALIDATION)
    )
    same_budget = []
    half_label = []
    architecture_interaction = []
    family_gain_ordering = []

    for architecture, family, sample_count in product(
        ARCHITECTURES, FAMILIES, SAMPLE_COUNTS
    ):
        scratch = selected_vectors[
            architecture, "scratch", family, sample_count, "pressure_l2"
        ]
        pretrained = selected_vectors[
            architecture, "pretrained", family, sample_count, "pressure_l2"
        ]
        gain = _ratio_summary(scratch, pretrained, bootstrap_indices)
        same_budget.append(
            {
                "architecture": architecture,
                "family": family,
                "sample_count": sample_count,
                "scratch_over_pretrained": gain,
                "material_benefit": gain["point"] >= 1.10,
            }
        )

    for architecture, family in product(ARCHITECTURES, FAMILIES):
        pretrained_choice = selected[architecture, "pretrained", family, 64]
        scratch_choice = selected[architecture, "scratch", family, 128]
        pretrained_pressure = selected_vectors[
            architecture, "pretrained", family, 64, "pressure_l2"
        ]
        scratch_pressure = selected_vectors[
            architecture, "scratch", family, 128, "pressure_l2"
        ]
        pretrained_wss = selected_vectors[
            architecture, "pretrained", family, 64, "wss_l2"
        ]
        scratch_wss = selected_vectors[architecture, "scratch", family, 128, "wss_l2"]
        pressure_ratio = _ratio_summary(
            pretrained_pressure, scratch_pressure, bootstrap_indices
        )
        wss_ratio = _ratio_summary(pretrained_wss, scratch_wss, bootstrap_indices)
        horizon_sufficient = all(
            choice["completed_epoch"] != REGISTERED_EPOCHS[-1] + 1
            for choice in (pretrained_choice, scratch_choice)
        )
        optimizer_bracketed = all(
            choice["learning_rate"] == LEARNING_RATES[1][0]
            for choice in (pretrained_choice, scratch_choice)
        )
        half_label.append(
            {
                "architecture": architecture,
                "family": family,
                "selected_completed_epochs": {
                    "pretrained_64": pretrained_choice["completed_epoch"],
                    "scratch_128": scratch_choice["completed_epoch"],
                },
                "selected_learning_rates": {
                    "pretrained_64": pretrained_choice["learning_rate"],
                    "scratch_128": scratch_choice["learning_rate"],
                },
                "pretrained_64_over_scratch_128_pressure": pressure_ratio,
                "pretrained_64_over_scratch_128_wss": wss_ratio,
                "material_half_label_success": pressure_ratio["point"] <= 0.95,
                "field_safety": wss_ratio["point"] <= 1.10,
                "optimization_horizon_sufficient": horizon_sufficient,
                "optimization_grid_bracketed": optimizer_bracketed,
                "strong_validation_signal": (
                    pressure_ratio["paired_case_bootstrap_95_ci"][1] < 1.0
                ),
            }
        )

    for family in FAMILIES:
        bootstrap_gains: dict[str, np.ndarray] = {}
        point_gains: dict[str, float] = {}
        for architecture in ARCHITECTURES:
            ratios = []
            ratio_points = []
            for sample_count in SAMPLE_COUNTS:
                scratch = selected_vectors[
                    architecture, "scratch", family, sample_count, "pressure_l2"
                ]
                pretrained = selected_vectors[
                    architecture, "pretrained", family, sample_count, "pressure_l2"
                ]
                ratios.append(
                    scratch[bootstrap_indices].mean(axis=1)
                    / pretrained[bootstrap_indices].mean(axis=1)
                )
                ratio_points.append(float(scratch.mean() / pretrained.mean()))
            bootstrap_gains[architecture] = np.sqrt(ratios[0] * ratios[1])
            point_gains[architecture] = float(
                np.sqrt(ratio_points[0] * ratio_points[1])
            )
        interaction_samples = (
            bootstrap_gains["mesh_transformer"] / bootstrap_gains["geotransolver"]
        )
        point = point_gains["mesh_transformer"] / point_gains["geotransolver"]
        architecture_interaction.append(
            {
                "family": family,
                "mesh_transformer_gain": point_gains["mesh_transformer"],
                "geotransolver_gain": point_gains["geotransolver"],
                "mesh_transformer_over_geotransolver": {
                    "point": point,
                    "paired_case_bootstrap_95_ci": _quantiles(interaction_samples),
                },
                "boundary_specific_increment": point >= 1.10,
            }
        )

    # The geometry-only study prospectively established fastback as the
    # morphology-nearer family. Its joint prediction is therefore a
    # fastback/estate gain ratio of at least 1.10 in each architecture.
    # Families contain different cases, so resample them independently while
    # preserving all within-family pairings across budgets and initializations.
    family_bootstrap_indices = {
        family: np.random.default_rng(BOOTSTRAP_SEED + offset).integers(
            0, N_VALIDATION, size=(N_BOOTSTRAP, N_VALIDATION)
        )
        for offset, family in enumerate(FAMILIES, start=1)
    }
    for architecture in ARCHITECTURES:
        point_gains: dict[str, float] = {}
        bootstrap_gains: dict[str, np.ndarray] = {}
        for family in FAMILIES:
            indices = family_bootstrap_indices[family]
            ratio_points = []
            ratio_samples = []
            for sample_count in SAMPLE_COUNTS:
                scratch = selected_vectors[
                    architecture, "scratch", family, sample_count, "pressure_l2"
                ]
                pretrained = selected_vectors[
                    architecture, "pretrained", family, sample_count, "pressure_l2"
                ]
                ratio_points.append(float(scratch.mean() / pretrained.mean()))
                ratio_samples.append(
                    scratch[indices].mean(axis=1)
                    / pretrained[indices].mean(axis=1)
                )
            point_gains[family] = float(np.sqrt(ratio_points[0] * ratio_points[1]))
            bootstrap_gains[family] = np.sqrt(
                ratio_samples[0] * ratio_samples[1]
            )

        point = point_gains["fastback"] / point_gains["estate"]
        samples = bootstrap_gains["fastback"] / bootstrap_gains["estate"]
        family_gain_ordering.append(
            {
                "architecture": architecture,
                "estate_gain": point_gains["estate"],
                "fastback_gain": point_gains["fastback"],
                "fastback_over_estate": {
                    "point": point,
                    "independent_family_bootstrap_95_ci": _quantiles(samples),
                },
                "geometry_scaffold_material_ordering": point >= 1.10,
                "directionally_consistent_with_morphology": point >= 1.0,
                "opposite_to_morphology_ordering": point < 1.0,
            }
        )

    advance = {
        architecture: all(
            record["material_half_label_success"]
            and record["field_safety"]
            and record["optimization_horizon_sufficient"]
            for record in half_label
            if record["architecture"] == architecture
        )
        for architecture in ARCHITECTURES
    }
    lr_refinement_required = {
        architecture: any(
            not record["optimization_grid_bracketed"]
            for record in half_label
            if record["architecture"] == architecture
        )
        for architecture in ARCHITECTURES
    }
    return {
        "selected_validation_points": sorted(
            selected_records,
            key=lambda item: (
                item["architecture"],
                item["initialization"],
                item["family"],
                item["sample_count"],
            ),
        ),
        "same_budget_gain": same_budget,
        "half_label_test": half_label,
        "architecture_interaction": architecture_interaction,
        "family_gain_ordering": family_gain_ordering,
        "decision": {
            "advance_to_confirmation": advance,
            "confirmation_learning_rate_refinement_required": (
                lr_refinement_required
            ),
            "boundary_specific_increment_on_both_families": all(
                record["boundary_specific_increment"]
                for record in architecture_interaction
            ),
            "geometry_scaffold_material_ordering_on_both_architectures": all(
                record["geometry_scaffold_material_ordering"]
                for record in family_gain_ordering
            ),
            "morphology_proximity_sufficient_falsified": any(
                record["opposite_to_morphology_ordering"]
                for record in family_gain_ordering
            ),
            "test_splits_remain_sealed": True,
        },
        "bootstrap": {
            "method": "paired validation-case resampling",
            "replicates": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "cross_family_method": (
                "independent family resampling with within-family pairing"
            ),
            "cross_family_seeds": {
                family: BOOTSTRAP_SEED + offset
                for offset, family in enumerate(FAMILIES, start=1)
            },
            "limitation": (
                "Descriptive only: the same validation cases selected learning "
                "rate and epoch."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--terminal-events", type=Path, required=True)
    parser.add_argument("--fixed-validation-root", type=Path, required=True)
    parser.add_argument("--fixed-validation-manifest", type=Path, required=True)
    parser.add_argument("--fixed-evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    terminal_events = _load_terminal_events(args.terminal_events)
    runs: dict[str, dict[str, Any]] = {}
    for architecture, initialization, family, sample_count, (_, lr_name) in product(
        ARCHITECTURES,
        INITIALIZATIONS,
        FAMILIES,
        SAMPLE_COUNTS,
        LEARNING_RATES,
    ):
        run_id = _run_id(architecture, initialization, family, sample_count, lr_name)
        runs[run_id] = _load_run(
            args.pilot_root / run_id,
            terminal_events.get(run_id),
        )

    unknown_terminal_runs = sorted(set(terminal_events) - set(runs))
    if unknown_terminal_runs:
        raise ValueError(
            "terminal-event ledger names runs outside the registered design: "
            f"{unknown_terminal_runs}"
        )

    fixed_runs, fixed_incomplete, invalid_fixed_candidates = _load_fixed_runs(
        args.fixed_validation_root,
        runs,
    )
    selected, selection_incomplete = _select_runs(runs, fixed_runs)
    incomplete = sorted(set(fixed_incomplete + selection_incomplete))
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if not incomplete else "incomplete",
        "analysis_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": _sha256(args.preregistration),
        "terminal_events_sha256": _sha256(args.terminal_events),
        "fixed_validation_manifest_sha256": _sha256(args.fixed_validation_manifest),
        "fixed_evaluator_sha256": _sha256(args.fixed_evaluator),
        "expected_training_runs": len(runs),
        "completed_training_runs": sum(run["completed"] for run in runs.values()),
        "non_finite_training_runs": sorted(
            run_id for run_id, run in runs.items() if run["non_finite"]
        ),
        "invalid_fixed_validation_candidates": invalid_fixed_candidates,
        "registered_completed_epochs": [epoch + 1 for epoch in REGISTERED_EPOCHS],
        "fixed_validation_seeds": list(EVALUATION_SEEDS),
        "selection_metric": (
            "mean per-case validation pressure relative L2 after averaging "
            "the two fixed point-subsample seeds within case"
        ),
        "incomplete_runs": incomplete,
    }
    if not incomplete:
        result.update(_reduce_complete(fixed_runs, selected))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
