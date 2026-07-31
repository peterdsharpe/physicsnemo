#!/usr/bin/env python3
"""Freeze the eligible fixed-validation checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

ARCHITECTURES = ("mesh_transformer", "geotransolver")
INITIALIZATIONS = ("scratch", "pretrained")
FAMILIES = ("estate", "fastback")
SAMPLE_COUNTS = (64, 128)
LEARNING_RATE_NAMES = ("3e-4", "1e-3", "3e-3")
REGISTERED_COMPLETED_EPOCHS = (1, 11, 21, 31, 41, 51, 61, 70)
EVALUATION_SEEDS = (20_260_730, 20_260_731)
N_LANES = 8


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_name(architecture: str, completed_epoch: int) -> str:
    model_name = (
        "MeshTransformer" if architecture == "mesh_transformer" else "GeoTransolver"
    )
    return f"{model_name}.0.{completed_epoch}.mdlus"


def _load_terminal_events(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    events = {event["run_id"]: event for event in payload["events"]}
    if len(events) != len(payload["events"]):
        raise ValueError(f"{path}: duplicate run in terminal-event ledger")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--terminal-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    terminal_events = _load_terminal_events(args.terminal_events)
    known_runs: set[str] = set()
    evaluations: list[dict[str, Any]] = []

    for (
        architecture,
        initialization,
        family,
        sample_count,
        learning_rate_name,
    ) in product(
        ARCHITECTURES,
        INITIALIZATIONS,
        FAMILIES,
        SAMPLE_COUNTS,
        LEARNING_RATE_NAMES,
    ):
        run_id = _run_id(
            architecture,
            initialization,
            family,
            sample_count,
            learning_rate_name,
        )
        known_runs.add(run_id)
        run_root = args.pilot_root / run_id
        event = terminal_events.get(run_id)
        if event is None:
            train_log = run_root / "train.log"
            train_text = (
                train_log.read_text(errors="replace") if train_log.exists() else ""
            )
            if "Training completed!" not in train_text:
                raise ValueError(f"{run_id}: training is not complete")
            completed_epochs = REGISTERED_COMPLETED_EPOCHS
        else:
            completed_epochs = tuple(
                int(epoch) + 1 for epoch in event["eligible_registered_epoch_indices"]
            )

        for completed_epoch in completed_epochs:
            checkpoint = (
                run_root
                / "checkpoints"
                / _checkpoint_name(architecture, completed_epoch)
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            evaluations.extend(
                {
                    "architecture": architecture,
                    "initialization": initialization,
                    "family": family,
                    "sample_count": sample_count,
                    "learning_rate_name": learning_rate_name,
                    "run_id": run_id,
                    "completed_epoch": completed_epoch,
                    "evaluation_seed": evaluation_seed,
                    "estimated_seconds": (
                        42.0 if architecture == "mesh_transformer" else 13.0
                    ),
                }
                for evaluation_seed in EVALUATION_SEEDS
            )

    unknown_terminal_runs = sorted(set(terminal_events) - known_runs)
    if unknown_terminal_runs:
        raise ValueError(
            "terminal-event ledger contains runs outside the registered design: "
            f"{unknown_terminal_runs}"
        )

    lane_loads = [0.0] * N_LANES
    lane_counts = [0] * N_LANES
    for record in sorted(
        evaluations,
        key=lambda item: (
            -item["estimated_seconds"],
            item["run_id"],
            item["completed_epoch"],
            item["evaluation_seed"],
        ),
    ):
        lane = min(range(N_LANES), key=lambda index: (lane_loads[index], index))
        record["lane"] = lane
        lane_loads[lane] += record["estimated_seconds"]
        lane_counts[lane] += 1

    fieldnames = (
        "unit_id",
        "lane",
        "architecture",
        "initialization",
        "family",
        "sample_count",
        "learning_rate_name",
        "run_id",
        "completed_epoch",
        "evaluation_seed",
    )
    rows = sorted(
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
        for unit_id, record in enumerate(rows):
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

    summary = {
        "schema_version": 1,
        "manifest_sha256": _sha256(args.output),
        "terminal_events_sha256": _sha256(args.terminal_events),
        "training_runs": len(known_runs),
        "terminal_training_runs": len(terminal_events),
        "evaluation_units": len(rows),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "lane_counts": lane_counts,
        "estimated_lane_seconds": lane_loads,
        "test_splits_accessed": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
