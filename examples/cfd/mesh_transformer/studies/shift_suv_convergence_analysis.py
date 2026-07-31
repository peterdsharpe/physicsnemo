#!/usr/bin/env python3
"""Reduce the preregistered matched-update SHIFT-SUV validation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ARCHITECTURES = ("mesh_transformer", "geotransolver")
FAMILIES = ("estate", "fastback")
ROLES = ("pretrained_64", "scratch_128")
LEARNING_RATES = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
EVALUATION_SEEDS = (20_260_730, 20_260_731)
N_VALIDATION = 99
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_730
N_SELECTION_FOLDS = 5
SELECTION_CROSSFIT_SEED = 20_260_732


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):
                    break
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield record


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def _load_fixed_pass(
    fixed_root: Path,
    run_id: str,
    completed_epoch: int,
    seed: int,
) -> dict[str, Any] | None:
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

    expected_epoch_index = completed_epoch - 1
    config: dict[str, Any] | None = None
    complete = False
    steps: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] | None = None
    for record in _read_jsonl(path):
        phase = record.get("phase")
        if phase == "fixed_eval_config":
            config = record
        elif phase == "val_step":
            if int(record.get("epoch", -1)) != expected_epoch_index:
                raise ValueError(f"{path}: unexpected epoch in validation step")
            steps[int(record["val_step"])] = record
        elif phase == "val_summary":
            if int(record.get("epoch", -1)) != expected_epoch_index:
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
    if set(steps) != set(range(N_VALIDATION)):
        missing = sorted(set(range(N_VALIDATION)) - set(steps))
        extra = sorted(set(steps) - set(range(N_VALIDATION)))
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
        "path": str(path),
        "steps": steps,
    }


def _average_fixed_passes(
    fixed_root: Path,
    run_id: str,
    completed_epoch: int,
) -> tuple[dict[str, np.ndarray] | None, list[str], list[dict[str, Any]]]:
    passes = [
        _load_fixed_pass(fixed_root, run_id, completed_epoch, seed)
        for seed in EVALUATION_SEEDS
    ]
    incomplete = [
        f"{run_id}: missing epoch {completed_epoch}, seed {seed}"
        for seed, result in zip(EVALUATION_SEEDS, passes)
        if result is None
    ]
    invalid = [
        {
            "run_id": run_id,
            "completed_epoch": completed_epoch,
            "evaluation_seed": seed,
            "reason": result["reason"],
            "path": result["path"],
        }
        for seed, result in zip(EVALUATION_SEEDS, passes)
        if result is not None and result["invalid"]
    ]
    if incomplete or invalid:
        return None, incomplete, invalid

    valid_passes = [result for result in passes if result is not None]
    vectors = {
        metric: np.asarray(
            [
                np.mean(
                    [
                        float(result["steps"][val_step][metric])
                        for result in valid_passes
                    ]
                )
                for val_step in range(N_VALIDATION)
            ],
            dtype=np.float64,
        )
        for metric in ("pressure_l2", "wss_l2")
    }
    return vectors, [], []


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
        "paired_case_bootstrap_95_ci": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))
        ],
    }


def _selection_crossfit(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Select on four case folds and evaluate on the held-out fold."""
    permutation = np.random.default_rng(SELECTION_CROSSFIT_SEED).permutation(
        N_VALIDATION
    )
    folds = np.array_split(permutation, N_SELECTION_FOLDS)
    outputs = {
        metric: np.empty(N_VALIDATION, dtype=np.float64)
        for metric in ("pressure_l2", "wss_l2")
    }
    selections = []
    all_indices = np.arange(N_VALIDATION)
    for fold_index, held_out in enumerate(folds):
        train = np.setdiff1d(all_indices, held_out, assume_unique=True)
        selected = min(
            candidates,
            key=lambda candidate: (
                float(candidate["vectors"]["pressure_l2"][train].mean()),
                (
                    float(candidate["learning_rate"])
                    if candidate["learning_rate"] is not None
                    else -1.0
                ),
            ),
        )
        for metric in outputs:
            outputs[metric][held_out] = selected["vectors"][metric][held_out]
        selections.append(
            {
                "fold": fold_index,
                "held_out_case_indices": sorted(int(index) for index in held_out),
                "source": selected["source"],
                "run_id": selected["run_id"],
                "continuation_learning_rate": selected["learning_rate"],
                "rate_grid_edge": selected["rate_grid_edge"],
            }
        )
    return outputs, selections


def _parent_scratch_choices(
    parent_summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    choices: dict[tuple[str, str], dict[str, Any]] = {}
    for record in parent_summary["selected_validation_points"]:
        if record["initialization"] != "scratch" or int(record["sample_count"]) != 128:
            continue
        key = (str(record["architecture"]), str(record["family"]))
        if key in choices:
            raise ValueError(f"duplicate parent scratch-128 choice: {key}")
        choices[key] = record
    expected = set(product(ARCHITECTURES, FAMILIES))
    if set(choices) != expected:
        raise ValueError(
            f"parent scratch-128 choices differ from design: {set(choices)}"
        )
    return choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--parent-fixed-root", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--analysis-clarification", type=Path, required=True)
    parser.add_argument("--prospective-forecast", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--fixed-validation-manifest", type=Path, required=True)
    parser.add_argument("--fixed-evaluator", type=Path, required=True)
    parser.add_argument("--terminal-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _read_json(args.preregistration)
    clarification = _read_json(args.analysis_clarification)
    forecast = _read_json(args.prospective_forecast)
    parent_summary = _read_json(args.parent_summary)
    terminal_events = _read_json(args.terminal_events)
    preregistration_sha256 = _sha256(args.preregistration)
    if (
        clarification.get("frozen_preregistration_sha256")
        != preregistration_sha256
    ):
        raise ValueError("analysis clarification names a different preregistration")
    if (
        forecast.get("parent_result_sha256")
        != preregistration["parent_result"]["sha256"]
    ):
        raise ValueError("prospective forecast names a different parent result")
    if _sha256(args.parent_summary) != preregistration["parent_result"]["sha256"]:
        raise ValueError("parent result hash differs from preregistration")
    if parent_summary.get("status") != "complete":
        raise ValueError("parent result is not complete")
    if not parent_summary["decision"]["test_splits_remain_sealed"]:
        raise ValueError("parent result does not certify sealed test splits")
    if terminal_events.get("test_splits_accessed") is not False:
        raise ValueError("terminal ledger does not certify sealed test splits")

    training_rows = _read_manifest(args.training_manifest)
    training_by_run = {row["run_id"]: row for row in training_rows}
    if len(training_by_run) != len(training_rows):
        raise ValueError("duplicate run in training manifest")
    if len(training_rows) != int(preregistration["design"]["total_training_runs"]):
        raise ValueError("training manifest does not cover the registered design")
    if (
        terminal_events.get("training_manifest_sha256")
        != _sha256(args.training_manifest)
    ):
        raise ValueError("terminal ledger names a different training manifest")

    fixed_summary_path = args.fixed_validation_manifest.with_suffix(
        args.fixed_validation_manifest.suffix + ".summary.json"
    )
    fixed_summary = _read_json(fixed_summary_path)
    if (
        fixed_summary.get("manifest_sha256")
        != _sha256(args.fixed_validation_manifest)
    ):
        raise ValueError("fixed-validation manifest hash differs from its summary")
    if fixed_summary.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("fixed-validation manifest used a different protocol")
    if (
        fixed_summary.get("analysis_clarification_sha256")
        != _sha256(args.analysis_clarification)
    ):
        raise ValueError("fixed-validation manifest used a different clarification")
    if (
        fixed_summary.get("terminal_events_sha256")
        != _sha256(args.terminal_events)
    ):
        raise ValueError("fixed-validation manifest used a different terminal ledger")
    if fixed_summary.get("test_splits_accessed") is not False:
        raise ValueError("fixed-validation manifest does not certify sealed tests")

    fixed_rows = _read_manifest(args.fixed_validation_manifest)
    registered_candidates: dict[
        tuple[str, str, str, float, int], dict[str, str]
    ] = {}
    seeds_by_candidate: dict[tuple[str, str, str, float, int], set[int]] = {}
    for row in fixed_rows:
        key = (
            row["architecture"],
            row["family"],
            row["role"],
            float(row["continuation_learning_rate"]),
            int(row["completed_epoch"]),
        )
        registered_candidates[key] = row
        seeds_by_candidate.setdefault(key, set()).add(int(row["evaluation_seed"]))
    for key, seeds in seeds_by_candidate.items():
        if seeds != set(EVALUATION_SEEDS):
            raise ValueError(f"{key}: fixed manifest seeds are {sorted(seeds)}")

    fixed_root = args.task_root / "fixed_validation"
    vectors_by_candidate: dict[
        tuple[str, str, str, float, int], dict[str, np.ndarray]
    ] = {}
    incomplete: list[str] = []
    invalid_candidates: list[dict[str, Any]] = []
    for key, row in registered_candidates.items():
        vectors, candidate_incomplete, candidate_invalid = _average_fixed_passes(
            fixed_root,
            row["run_id"],
            int(row["completed_epoch"]),
        )
        incomplete.extend(candidate_incomplete)
        invalid_candidates.extend(candidate_invalid)
        if vectors is not None:
            vectors_by_candidate[key] = vectors

    parent_choices = _parent_scratch_choices(parent_summary)
    parent_vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, choice in parent_choices.items():
        vectors, candidate_incomplete, candidate_invalid = _average_fixed_passes(
            args.parent_fixed_root,
            choice["run_id"],
            int(choice["completed_epoch"]),
        )
        incomplete.extend(candidate_incomplete)
        invalid_candidates.extend(candidate_invalid)
        if vectors is not None:
            parent_vectors[key] = vectors

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete" if incomplete else "complete",
        "analysis_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": preregistration_sha256,
        "analysis_clarification_sha256": _sha256(args.analysis_clarification),
        "prospective_forecast_sha256": _sha256(args.prospective_forecast),
        "parent_summary_sha256": _sha256(args.parent_summary),
        "training_manifest_sha256": _sha256(args.training_manifest),
        "fixed_validation_manifest_sha256": _sha256(
            args.fixed_validation_manifest
        ),
        "fixed_evaluator_sha256": _sha256(args.fixed_evaluator),
        "terminal_events_sha256": _sha256(args.terminal_events),
        "fixed_validation_seeds": list(EVALUATION_SEEDS),
        "invalid_fixed_validation_candidates": invalid_candidates,
        "incomplete": sorted(set(incomplete)),
        "test_splits_remain_sealed": True,
    }
    if incomplete:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        raise SystemExit(2)

    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, N_VALIDATION, size=(N_BOOTSTRAP, N_VALIDATION)
    )
    stage_records: list[dict[str, Any]] = []
    stage_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    comparison_stages = preregistration["design"]["comparison_stages"]
    for architecture, family, stage in product(
        ARCHITECTURES,
        FAMILIES,
        comparison_stages,
    ):
        stage_name = str(stage["name"])
        role_choices: dict[str, dict[str, Any]] = {}
        role_vectors: dict[str, dict[str, np.ndarray]] = {}
        role_candidates: dict[str, list[dict[str, Any]]] = {}
        for role in ROLES:
            if stage_name == "early" and role == "scratch_128":
                choice = parent_choices[architecture, family]
                candidate = {
                    "source": "parent",
                    "run_id": choice["run_id"],
                    "learning_rate": None,
                    "rate_grid_edge": False,
                    "vectors": parent_vectors[architecture, family],
                }
                role_candidates[role] = [candidate]
                role_choices[role] = {
                    "source": "parent",
                    "run_id": choice["run_id"],
                    "completed_epoch": int(choice["completed_epoch"]),
                    "continuation_learning_rate": None,
                    "parent_learning_rate": float(choice["learning_rate"]),
                    "pressure_l2": float(
                        parent_vectors[architecture, family]["pressure_l2"].mean()
                    ),
                    "wss_l2": float(
                        parent_vectors[architecture, family]["wss_l2"].mean()
                    ),
                    "rate_grid_edge": False,
                }
                role_vectors[role] = parent_vectors[architecture, family]
                continue

            epoch_key = f"{role}_continuation_epoch"
            completed_epoch = int(stage[epoch_key])
            candidates = []
            for learning_rate in LEARNING_RATES:
                key = (
                    architecture,
                    family,
                    role,
                    learning_rate,
                    completed_epoch,
                )
                vectors = vectors_by_candidate.get(key)
                if vectors is None:
                    continue
                row = registered_candidates[key]
                candidates.append(
                    {
                        "source": "continuation",
                        "run_id": row["run_id"],
                        "learning_rate": learning_rate,
                        "rate_grid_edge": learning_rate in (
                            LEARNING_RATES[0],
                            LEARNING_RATES[-1],
                        ),
                        "vectors": vectors,
                    }
                )
            if not candidates:
                raise ValueError(
                    f"{architecture}/{family}/{stage_name}/{role}: "
                    "no finite registered candidate"
                )
            selected_candidate = min(
                candidates,
                key=lambda candidate: (
                    float(candidate["vectors"]["pressure_l2"].mean()),
                    float(candidate["learning_rate"]),
                ),
            )
            role_candidates[role] = candidates
            vectors = selected_candidate["vectors"]
            learning_rate = float(selected_candidate["learning_rate"])
            role_choices[role] = {
                "source": "continuation",
                "run_id": selected_candidate["run_id"],
                "completed_epoch": completed_epoch,
                "continuation_learning_rate": learning_rate,
                "pressure_l2": float(vectors["pressure_l2"].mean()),
                "wss_l2": float(vectors["wss_l2"].mean()),
                "rate_grid_edge": selected_candidate["rate_grid_edge"],
            }
            role_vectors[role] = vectors

        pressure_ratio = _ratio_summary(
            role_vectors["pretrained_64"]["pressure_l2"],
            role_vectors["scratch_128"]["pressure_l2"],
            bootstrap_indices,
        )
        wss_ratio = _ratio_summary(
            role_vectors["pretrained_64"]["wss_l2"],
            role_vectors["scratch_128"]["wss_l2"],
            bootstrap_indices,
        )
        crossfit_vectors = {}
        crossfit_selections = {}
        for role in ROLES:
            vectors, selections = _selection_crossfit(role_candidates[role])
            crossfit_vectors[role] = vectors
            crossfit_selections[role] = selections
        crossfit_pressure_ratio = _ratio_summary(
            crossfit_vectors["pretrained_64"]["pressure_l2"],
            crossfit_vectors["scratch_128"]["pressure_l2"],
            bootstrap_indices,
        )
        crossfit_wss_ratio = _ratio_summary(
            crossfit_vectors["pretrained_64"]["wss_l2"],
            crossfit_vectors["scratch_128"]["wss_l2"],
            bootstrap_indices,
        )
        crossfit_rate_edge = any(
            selection["rate_grid_edge"]
            for selections in crossfit_selections.values()
            for selection in selections
        )
        record = {
            "architecture": architecture,
            "family": family,
            "stage": stage_name,
            "total_updates": {
                "pretrained_64": int(stage["pretrained_64_total_updates"]),
                "scratch_128": int(stage["scratch_128_total_updates"]),
            },
            "absolute_update_mismatch": abs(
                int(stage["pretrained_64_total_updates"])
                - int(stage["scratch_128_total_updates"])
            ),
            "selected": role_choices,
            "pretrained_64_over_scratch_128_pressure": pressure_ratio,
            "pretrained_64_over_scratch_128_wss": wss_ratio,
            "selection_crossfit": {
                "folds": N_SELECTION_FOLDS,
                "permutation_seed": SELECTION_CROSSFIT_SEED,
                "selected_by_role": crossfit_selections,
                "pretrained_64_over_scratch_128_pressure": (
                    crossfit_pressure_ratio
                ),
                "pretrained_64_over_scratch_128_wss": crossfit_wss_ratio,
                "rate_grid_edge": crossfit_rate_edge,
            },
        }
        stage_records.append(record)
        stage_lookup[architecture, family, stage_name] = record

    family_decisions = []
    for architecture, family in product(ARCHITECTURES, FAMILIES):
        middle = stage_lookup[architecture, family, "middle"]
        final = stage_lookup[architecture, family, "final"]
        role_relative_changes = {
            role: abs(
                final["selected"][role]["pressure_l2"]
                / middle["selected"][role]["pressure_l2"]
                - 1.0
            )
            for role in ROLES
        }
        ratio_relative_change = abs(
            final["pretrained_64_over_scratch_128_pressure"]["point"]
            / middle["pretrained_64_over_scratch_128_pressure"]["point"]
            - 1.0
        )
        pressure_ratio = final[
            "pretrained_64_over_scratch_128_pressure"
        ]
        wss_ratio = final["pretrained_64_over_scratch_128_wss"]
        crossfit_pressure_ratio = final["selection_crossfit"][
            "pretrained_64_over_scratch_128_pressure"
        ]
        crossfit_wss_ratio = final["selection_crossfit"][
            "pretrained_64_over_scratch_128_wss"
        ]
        primary_rate_edge = any(
            final["selected"][role]["rate_grid_edge"] for role in ROLES
        )
        crossfit_rate_edge = bool(final["selection_crossfit"]["rate_grid_edge"])
        primary_material_success = pressure_ratio["point"] <= 0.95
        crossfit_material_success = crossfit_pressure_ratio["point"] <= 0.95
        primary_strong_signal = (
            pressure_ratio["paired_case_bootstrap_95_ci"][1] < 1.0
        )
        crossfit_strong_signal = (
            crossfit_pressure_ratio["paired_case_bootstrap_95_ci"][1] < 1.0
        )
        primary_field_safety = wss_ratio["point"] <= 1.10
        crossfit_field_safety = crossfit_wss_ratio["point"] <= 1.10
        plateau = (
            all(change < 0.02 for change in role_relative_changes.values())
            and ratio_relative_change < 0.02
        )
        family_decisions.append(
            {
                "architecture": architecture,
                "family": family,
                "final_pressure_ratio": pressure_ratio,
                "final_wss_ratio": wss_ratio,
                "selection_crossfit_pressure_ratio": crossfit_pressure_ratio,
                "selection_crossfit_wss_ratio": crossfit_wss_ratio,
                "primary_material_half_label_success": primary_material_success,
                "selection_crossfit_material_half_label_success": (
                    crossfit_material_success
                ),
                "material_half_label_success": (
                    primary_material_success and crossfit_material_success
                ),
                "primary_strong_validation_signal": primary_strong_signal,
                "selection_crossfit_strong_validation_signal": (
                    crossfit_strong_signal
                ),
                "strong_validation_signal": (
                    primary_strong_signal and crossfit_strong_signal
                ),
                "primary_field_safety": primary_field_safety,
                "selection_crossfit_field_safety": crossfit_field_safety,
                "field_safety": primary_field_safety and crossfit_field_safety,
                "middle_to_final_absolute_relative_change": {
                    **role_relative_changes,
                    "pressure_ratio": ratio_relative_change,
                },
                "plateau_gate": plateau,
                "primary_final_rate_grid_edge": primary_rate_edge,
                "selection_crossfit_final_rate_grid_edge": crossfit_rate_edge,
                "final_rate_grid_edge": primary_rate_edge or crossfit_rate_edge,
            }
        )

    architecture_decisions = []
    for architecture in ARCHITECTURES:
        records = [
            record
            for record in family_decisions
            if record["architecture"] == architecture
        ]
        plateau = all(record["plateau_gate"] for record in records)
        rate_edge = any(record["final_rate_grid_edge"] for record in records)
        advance = (
            not rate_edge
            and plateau
            and all(
                record["material_half_label_success"]
                and record["strong_validation_signal"]
                and record["field_safety"]
                for record in records
            )
        )
        close_as_optimization_only = (
            not rate_edge
            and plateau
            and any(
                record["final_pressure_ratio"][
                    "paired_case_bootstrap_95_ci"
                ][0]
                > 0.95
                and record["selection_crossfit_pressure_ratio"][
                    "paired_case_bootstrap_95_ci"
                ][0]
                > 0.95
                for record in records
            )
        )
        if rate_edge and not plateau:
            next_action = "refine_rate_grid_and_extend_same_checkpoints"
        elif rate_edge:
            next_action = "refine_continuation_rate_grid"
        elif not plateau:
            next_action = "extend_same_checkpoints_without_interpretation"
        elif advance:
            next_action = "advance_to_multiseed_multisource_confirmation"
        elif close_as_optimization_only:
            next_action = "close_broad_half_label_branch_as_optimization_only"
        else:
            next_action = "hold_as_inconclusive_validation_pilot"
        architecture_decisions.append(
            {
                "architecture": architecture,
                "plateau_gate_on_both_families": plateau,
                "unresolved_final_rate_grid_edge": rate_edge,
                "advance_to_multiseed_confirmation": advance,
                "close_as_optimization_only": close_as_optimization_only,
                "next_action": next_action,
            }
        )

    curve_predictions = {
        (record["architecture"], record["family"]): record
        for record in forecast["amendments"][0]["predictions"]
    }
    forecast_evaluation = []
    for architecture, family in product(ARCHITECTURES, FAMILIES):
        prediction = curve_predictions[architecture, family]
        actual_early = stage_lookup[architecture, family, "early"][
            "pretrained_64_over_scratch_128_pressure"
        ]["point"]
        actual_final = stage_lookup[architecture, family, "final"][
            "pretrained_64_over_scratch_128_pressure"
        ]["point"]
        forecast_evaluation.append(
            {
                "architecture": architecture,
                "family": family,
                "predicted_H_early": float(prediction["predicted_H_early"]),
                "actual_H_early": actual_early,
                "predicted_H_final": float(prediction["predicted_H_final"]),
                "actual_H_final": actual_final,
                "actual_over_predicted": {
                    "early": actual_early / float(prediction["predicted_H_early"]),
                    "final": actual_final / float(prediction["predicted_H_final"]),
                },
                "scratch_catchup_predicted": (
                    float(prediction["predicted_H_final"])
                    > float(prediction["predicted_H_early"])
                ),
                "scratch_catchup_observed": actual_final > actual_early,
            }
        )

    result.update(
        {
            "stage_results": stage_records,
            "family_decisions": family_decisions,
            "architecture_decisions": architecture_decisions,
            "prospective_curve_forecast_evaluation": forecast_evaluation,
            "bootstrap": {
                "primary_method": "paired validation-case resampling",
                "replicates": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "primary_limitation": (
                    "Descriptive: the same validation cases select the "
                    "continuation learning rate and quantify uncertainty."
                ),
                "selection_crossfit": {
                    "method": (
                        "five-fold case cross-fitting; select pressure-minimizing "
                        "rate on four folds and evaluate pressure and WSS on the "
                        "held-out fold"
                    ),
                    "folds": N_SELECTION_FOLDS,
                    "permutation_seed": SELECTION_CROSSFIT_SEED,
                    "decision_role": (
                        "conservative sensitivity gate; it may block but not "
                        "create advancement or closure"
                    ),
                },
            },
            "decision_rules_applied": {
                "material_half_label_threshold": 0.95,
                "strong_signal_upper_bound": 1.0,
                "field_safety_threshold": 1.10,
                "plateau_absolute_relative_change_threshold": 0.02,
                "test_splits_remain_sealed": True,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
