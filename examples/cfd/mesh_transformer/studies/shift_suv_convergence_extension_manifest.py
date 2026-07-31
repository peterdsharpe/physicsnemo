#!/usr/bin/env python3
"""Freeze the matched-update SHIFT-SUV convergence-extension runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

ARCHITECTURES = ("mesh_transformer", "geotransolver")
FAMILIES = ("estate", "fastback")
ROLES = {
    "pretrained_64": {
        "initialization": "pretrained",
        "sample_count": 64,
    },
    "scratch_128": {
        "initialization": "scratch",
        "sample_count": 128,
    },
}
LEARNING_RATES = (
    ("1e-4", 1.0e-4),
    ("3e-4", 3.0e-4),
    ("1e-3", 1.0e-3),
    ("3e-3", 3.0e-3),
)
N_LANES = 8
ESTIMATED_SECONDS_PER_CASE = {
    "mesh_transformer": 1.05,
    "geotransolver": 0.30,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_name(architecture: str) -> str:
    return (
        "MeshTransformer" if architecture == "mesh_transformer" else "GeoTransolver"
    )


def _selected_points(
    summary: dict[str, Any],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for record in summary["selected_validation_points"]:
        key = (
            str(record["architecture"]),
            str(record["initialization"]),
            str(record["family"]),
            int(record["sample_count"]),
        )
        if key in selected:
            raise ValueError(f"duplicate selected validation point: {key}")
        selected[key] = record
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent = _read_json(args.parent_summary)
    preregistration = _read_json(args.preregistration)
    expected_parent_sha = preregistration["parent_result"]["sha256"]
    actual_parent_sha = _sha256(args.parent_summary)
    if actual_parent_sha != expected_parent_sha:
        raise ValueError(
            "parent result hash differs from preregistration: "
            f"{actual_parent_sha} != {expected_parent_sha}"
        )
    if parent.get("status") != "complete":
        raise ValueError("parent adaptation result is not complete")
    if not parent["decision"]["test_splits_remain_sealed"]:
        raise ValueError("parent result does not certify sealed test splits")

    design = preregistration["design"]
    if tuple(design["architectures"]) != ARCHITECTURES:
        raise ValueError("preregistered architectures differ from the implementation")
    if tuple(design["families"]) != FAMILIES:
        raise ValueError("preregistered families differ from the implementation")
    if tuple(design["continuation_learning_rates"]) != tuple(
        value for _name, value in LEARNING_RATES
    ):
        raise ValueError(
            "preregistered continuation rates differ from the implementation"
        )

    selected = _selected_points(parent)
    records: list[dict[str, Any]] = []
    checkpoint_hashes: dict[Path, str] = {}
    for architecture, family, role, (rate_name, learning_rate) in product(
        ARCHITECTURES,
        FAMILIES,
        ROLES,
        LEARNING_RATES,
    ):
        role_design = design["half_label_roles"][role]
        role_contract = ROLES[role]
        if role_design["initialization"] != role_contract["initialization"]:
            raise ValueError(f"{role}: initialization differs from preregistration")
        if int(role_design["sample_count"]) != role_contract["sample_count"]:
            raise ValueError(f"{role}: sample count differs from preregistration")

        key = (
            architecture,
            role_contract["initialization"],
            family,
            role_contract["sample_count"],
        )
        point = selected.get(key)
        if point is None:
            raise ValueError(f"missing selected parent point: {key}")
        if int(point["completed_epoch"]) != 70:
            raise ValueError(f"{key}: selected parent checkpoint is not epoch 70")
        if point["run_terminal_status"] != "completed":
            raise ValueError(f"{key}: selected parent run did not complete")

        model_name = _model_name(architecture)
        parent_checkpoint = (
            args.pilot_root
            / point["run_id"]
            / "checkpoints"
            / f"{model_name}.0.70.mdlus"
        )
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
        if parent_checkpoint not in checkpoint_hashes:
            checkpoint_hashes[parent_checkpoint] = _sha256(parent_checkpoint)
        parent_checkpoint_sha256 = checkpoint_hashes[parent_checkpoint]

        continuation_epochs = int(role_design["continuation_epochs"])
        scheduler_step_size = int(
            design["continuation_scheduler"]["step_size_epochs"][role]
        )
        run_id = (
            f"convergence_{architecture}_{role}_{family}"
            f"_lr{rate_name}_seed42"
        )
        records.append(
            {
                "architecture": architecture,
                "family": family,
                "role": role,
                "initialization": role_contract["initialization"],
                "sample_count": role_contract["sample_count"],
                "parent_run_id": point["run_id"],
                "parent_completed_epoch": 70,
                "parent_checkpoint": str(parent_checkpoint),
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "continuation_learning_rate_name": rate_name,
                "continuation_learning_rate": learning_rate,
                "continuation_epochs": continuation_epochs,
                "scheduler_step_size": scheduler_step_size,
                "run_id": run_id,
                "estimated_seconds": (
                    role_contract["sample_count"]
                    * continuation_epochs
                    * ESTIMATED_SECONDS_PER_CASE[architecture]
                ),
            }
        )

    expected_runs = int(design["total_training_runs"])
    if len(records) != expected_runs:
        raise AssertionError(f"built {len(records)} runs, expected {expected_runs}")

    lane_loads = [0.0] * N_LANES
    lane_counts = [0] * N_LANES
    for record in sorted(
        records,
        key=lambda item: (
            -item["estimated_seconds"],
            item["architecture"],
            item["family"],
            item["role"],
            item["continuation_learning_rate"],
        ),
    ):
        lane = min(range(N_LANES), key=lambda index: (lane_loads[index], index))
        record["lane"] = lane
        lane_loads[lane] += float(record["estimated_seconds"])
        lane_counts[lane] += 1

    fieldnames = (
        "unit_id",
        "lane",
        "architecture",
        "family",
        "role",
        "initialization",
        "sample_count",
        "parent_run_id",
        "parent_completed_epoch",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "continuation_learning_rate_name",
        "continuation_learning_rate",
        "continuation_epochs",
        "scheduler_step_size",
        "run_id",
    )
    rows = sorted(
        records,
        key=lambda item: (
            item["lane"],
            item["architecture"],
            item["family"],
            item["role"],
            item["continuation_learning_rate"],
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
        "parent_summary_sha256": actual_parent_sha,
        "preregistration_sha256": _sha256(args.preregistration),
        "training_runs": len(rows),
        "lane_counts": lane_counts,
        "estimated_lane_seconds": lane_loads,
        "test_splits_accessed": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
