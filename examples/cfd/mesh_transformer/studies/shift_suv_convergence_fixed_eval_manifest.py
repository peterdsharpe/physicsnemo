#!/usr/bin/env python3
"""Freeze fixed-validation evaluations for the matched-update study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

EVALUATION_SEEDS = (20_260_730, 20_260_731)
N_LANES = 8
ESTIMATED_SECONDS = {
    "mesh_transformer": 42.0,
    "geotransolver": 13.0,
}
MODEL_NAMES = {
    "mesh_transformer": "MeshTransformer",
    "geotransolver": "GeoTransolver",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _read_training_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty training manifest")
    run_ids = [row["run_id"] for row in rows]
    unit_ids = [row["unit_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"{path}: duplicate run_id")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError(f"{path}: duplicate unit_id")
    return rows


def _checkpoint_paths(
    run_root: Path,
    architecture: str,
    completed_epoch: int,
) -> tuple[Path, Path]:
    checkpoint_root = run_root / "checkpoints"
    model_name = MODEL_NAMES[architecture]
    return (
        checkpoint_root / f"{model_name}.0.{completed_epoch}.mdlus",
        checkpoint_root / f"checkpoint.0.{completed_epoch}.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--analysis-clarification", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-events-output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _read_json(args.preregistration)
    clarification = _read_json(args.analysis_clarification)
    preregistration_sha256 = _sha256(args.preregistration)
    if (
        clarification.get("frozen_preregistration_sha256")
        != preregistration_sha256
    ):
        raise ValueError("analysis clarification names a different preregistration")
    if preregistration["scope"]["test_policy"] != (
        "No test case may be read, trained on, or evaluated."
    ):
        raise ValueError("unexpected test policy")

    training_summary_path = args.training_manifest.with_suffix(
        args.training_manifest.suffix + ".summary.json"
    )
    training_summary = _read_json(training_summary_path)
    if training_summary.get("manifest_sha256") != _sha256(args.training_manifest):
        raise ValueError("training manifest hash differs from its summary")
    if training_summary.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("training manifest used a different preregistration")
    if training_summary.get("test_splits_accessed") is not False:
        raise ValueError("training manifest does not certify sealed test splits")

    rows = _read_training_manifest(args.training_manifest)
    expected_runs = int(preregistration["design"]["total_training_runs"])
    if len(rows) != expected_runs:
        raise ValueError(f"found {len(rows)} training runs, expected {expected_runs}")

    known_terminal_markers = {
        args.task_root / f"TERMINAL_CANDIDATE_UNIT_{row['unit_id']}" for row in rows
    }
    unknown_terminal_markers = sorted(
        set(args.task_root.glob("TERMINAL_CANDIDATE_UNIT_*"))
        - known_terminal_markers
    )
    if unknown_terminal_markers:
        raise ValueError(
            "terminal markers outside the training manifest: "
            f"{unknown_terminal_markers}"
        )

    blocked = sorted(args.task_root.glob("BLOCKED_CONVERGENCE*"))
    if blocked:
        raise ValueError(f"unresolved convergence blockers: {blocked}")

    evaluations: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    for row in rows:
        architecture = row["architecture"]
        role = row["role"]
        if architecture not in MODEL_NAMES:
            raise ValueError(f"unknown architecture: {architecture}")
        role_design = preregistration["design"]["half_label_roles"].get(role)
        if role_design is None:
            raise ValueError(f"unknown role: {role}")

        registered_epochs = tuple(
            int(epoch)
            for epoch in role_design["registered_continuation_epochs"]
        )
        declared_final_epoch = int(row["continuation_epochs"])
        if registered_epochs[-1] != declared_final_epoch:
            raise ValueError(
                f"{row['run_id']}: final registered epoch differs from training"
            )

        run_root = args.task_root / "convergence_runs" / row["run_id"]
        train_log = run_root / "train.log"
        train_text = (
            train_log.read_text(errors="replace") if train_log.exists() else ""
        )
        terminal_marker = (
            args.task_root / f"TERMINAL_CANDIDATE_UNIT_{row['unit_id']}"
        )
        final_model, final_training = _checkpoint_paths(
            run_root,
            architecture,
            declared_final_epoch,
        )
        completed = (
            "Training completed!" in train_text
            and final_model.is_file()
            and final_training.is_file()
        )
        terminal = terminal_marker.is_file()
        if completed and terminal:
            raise ValueError(f"{row['run_id']}: both completed and terminal")
        if terminal and "loss guard triggered" not in train_text:
            raise ValueError(
                f"{row['run_id']}: terminal marker without non-finite-loss guard"
            )
        if not completed and not terminal:
            raise ValueError(f"{row['run_id']}: training is incomplete")

        eligible_epochs = []
        for completed_epoch in registered_epochs:
            model_checkpoint, training_checkpoint = _checkpoint_paths(
                run_root,
                architecture,
                completed_epoch,
            )
            exists = model_checkpoint.is_file() and training_checkpoint.is_file()
            if completed and not exists:
                raise FileNotFoundError(
                    f"{row['run_id']}: missing registered checkpoint "
                    f"{completed_epoch}"
                )
            if exists:
                eligible_epochs.append(completed_epoch)
                evaluations.extend(
                    {
                        "training_unit_id": int(row["unit_id"]),
                        "architecture": architecture,
                        "family": row["family"],
                        "role": role,
                        "sample_count": int(row["sample_count"]),
                        "continuation_learning_rate_name": row[
                            "continuation_learning_rate_name"
                        ],
                        "continuation_learning_rate": float(
                            row["continuation_learning_rate"]
                        ),
                        "run_id": row["run_id"],
                        "completed_epoch": completed_epoch,
                        "evaluation_seed": seed,
                        "run_terminal_status": (
                            "non_finite" if terminal else "completed"
                        ),
                    }
                    for seed in EVALUATION_SEEDS
                )

        if terminal:
            terminal_events.append(
                {
                    "training_unit_id": int(row["unit_id"]),
                    "run_id": row["run_id"],
                    "architecture": architecture,
                    "family": row["family"],
                    "role": role,
                    "continuation_learning_rate": float(
                        row["continuation_learning_rate"]
                    ),
                    "reason": "non_finite_loss_guard",
                    "eligible_registered_completed_epochs": eligible_epochs,
                    "terminal_marker_sha256": _sha256(terminal_marker),
                    "train_log_sha256": _sha256(train_log),
                }
            )

    lane_loads = [0.0] * N_LANES
    lane_counts = [0] * N_LANES
    for record in sorted(
        evaluations,
        key=lambda item: (
            -ESTIMATED_SECONDS[item["architecture"]],
            item["run_id"],
            item["completed_epoch"],
            item["evaluation_seed"],
        ),
    ):
        lane = min(range(N_LANES), key=lambda index: (lane_loads[index], index))
        record["lane"] = lane
        lane_loads[lane] += ESTIMATED_SECONDS[record["architecture"]]
        lane_counts[lane] += 1

    fieldnames = (
        "unit_id",
        "lane",
        "training_unit_id",
        "architecture",
        "family",
        "role",
        "sample_count",
        "continuation_learning_rate_name",
        "continuation_learning_rate",
        "run_id",
        "completed_epoch",
        "evaluation_seed",
        "run_terminal_status",
    )
    output_rows = sorted(
        evaluations,
        key=lambda item: (
            item["lane"],
            item["architecture"],
            item["run_id"],
            item["completed_epoch"],
            item["evaluation_seed"],
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for unit_id, record in enumerate(output_rows):
            writer.writerow(
                {
                    "unit_id": unit_id,
                    **{
                        field: record[field]
                        for field in fieldnames
                        if field != "unit_id"
                    },
                }
            )

    terminal_payload = {
        "schema_version": 1,
        "training_manifest_sha256": _sha256(args.training_manifest),
        "events": sorted(terminal_events, key=lambda item: item["training_unit_id"]),
        "test_splits_accessed": False,
    }
    args.terminal_events_output.parent.mkdir(parents=True, exist_ok=True)
    args.terminal_events_output.write_text(
        json.dumps(terminal_payload, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "schema_version": 1,
        "manifest_sha256": _sha256(args.output),
        "training_manifest_sha256": _sha256(args.training_manifest),
        "preregistration_sha256": preregistration_sha256,
        "analysis_clarification_sha256": _sha256(args.analysis_clarification),
        "terminal_events_sha256": _sha256(args.terminal_events_output),
        "training_runs": len(rows),
        "terminal_training_runs": len(terminal_events),
        "evaluation_units": len(output_rows),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "lane_counts": lane_counts,
        "estimated_lane_seconds": lane_loads,
        "test_splits_accessed": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
