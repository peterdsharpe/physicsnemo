# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test an exact uncoupled carrier plus a learned mode-conversion residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import nonseparable_rank_census as spectral
import nonseparable_response_relation as relation
import nonseparable_spectral_prior as reduction
import nonseparable_target_saturation as base
import torch


class ResidualKernel(base.KernelField):
    """The independent coordinate kernel with an exactly zero readout."""

    def __init__(self, query_x: torch.Tensor, output_modes: int) -> None:
        super().__init__(query_x, output_modes)
        output_layer = self.network[-1]
        if not isinstance(output_layer, torch.nn.Linear):
            raise TypeError("KernelField must end in a linear readout")
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)


def exact_target_and_carrier(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    family = relation.ResponseFamily(
        relation.RESOLUTIONS[-1],
        device=device,
        dtype=torch.float64,
    )
    target = family.response(relation.TARGET_LENGTH, relation.TARGET_SCREENING)
    profiles = (
        family.laplacian + relation.POTENTIAL_SCALE * family.reaction
    )
    diagonal_profiles = torch.diag_embed(
        torch.diagonal(profiles, dim1=-2, dim2=-1)
    )
    carrier = spectral.boundary_kernel(
        diagonal_profiles[None, ...],
        family.query_x,
    )[0, ..., : relation.INPUT_CHANNELS]
    variances = 1.0 / relation.column_energies(target)
    variances = variances / variances.mean()
    return (
        family.query_x.float(),
        target.float(),
        carrier.float(),
        variances.float(),
    )


def run_arm(
    *,
    arm: str,
    budget: int | None,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    query_x, target, carrier, drive_variances = exact_target_and_carrier(device)
    model = ResidualKernel(query_x, target.shape[-2]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )

    drives: torch.Tensor | None = None
    fields: torch.Tensor | None = None
    data_diagnostics: dict[str, Any] | None = None
    if arm == "target":
        if budget not in reduction.BUDGETS:
            raise ValueError(f"target budget must be one of {reduction.BUDGETS}")
        drives, fields, data_diagnostics = base.sampled_fields(
            target,
            drive_variances,
            budget=budget,
            seed=seed,
            design="orthogonal",
        )
    elif arm != "capacity":
        raise ValueError("arm must be 'capacity' or 'target'")

    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    initial_prediction = carrier + model()
    initial_error = float(
        base.weighted_operator_error(
            initial_prediction,
            target,
            drive_variances,
        )
        .detach()
        .cpu()
    )
    for step in range(1, base.STEPS + 1):
        kernel = carrier + model()
        if arm == "capacity":
            loss = base.weighted_operator_error(
                kernel,
                target,
                drive_variances,
            ).square()
        else:
            if drives is None or fields is None:
                raise RuntimeError("target arms require sampled fields")
            prediction = torch.einsum("qoi,bi->bqo", kernel, drives)
            loss = torch.sum((prediction - fields).square()) / torch.sum(
                fields.square()
            ).clamp_min(1.0e-30)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 1_000 == 0 or step == base.STEPS:
            record = {
                "step": step,
                "training_relative_rms": math.sqrt(float(loss.detach().cpu())),
                "held_out_operator_relative_rms": float(
                    base.weighted_operator_error(
                        carrier + model(),
                        target,
                        drive_variances,
                    )
                    .detach()
                    .cpu()
                ),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    final_prediction = (carrier + model()).detach()
    return {
        "schema_version": 1,
        "study": "nonseparable_diagonal_carrier_v1",
        "preregistration": (
            "results/nonseparable_diagonal_carrier_preregistration_2026-07-29.json"
        ),
        "arm": arm,
        "budget": budget,
        "seed": seed,
        "device": str(device),
        "steps": base.STEPS,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "carrier_relative_rms": float(
            base.weighted_operator_error(
                carrier,
                target,
                drive_variances,
            ).cpu()
        ),
        "initial_operator_relative_rms": initial_error,
        "final_operator_relative_rms": float(
            base.weighted_operator_error(
                final_prediction,
                target,
                drive_variances,
            ).cpu()
        ),
        "data_diagnostics": data_diagnostics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
    }


def reduce_reports(input_dir: Path) -> dict[str, Any]:
    report = reduction.reduce_reports(input_dir)
    report["study"] = "nonseparable_diagonal_carrier_v1"
    report["verdict"] = (
        "advance_to_source_transfer"
        if all(report["gates"].values())
        else "reject_diagonal_carrier_instrument"
    )
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--arm", choices=("capacity", "target"), required=True)
    run_parser.add_argument("--budget", type=int)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--output", type=Path, required=True)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--input-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        report = run_arm(
            arm=args.arm,
            budget=args.budget,
            seed=args.seed,
            device=torch.device(args.device),
        )
    else:
        report = reduce_reports(args.input_dir)
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": report.get("verdict"),
                "final_operator_relative_rms": report.get(
                    "final_operator_relative_rms"
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
