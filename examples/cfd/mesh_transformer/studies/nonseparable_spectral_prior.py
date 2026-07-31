# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test a spectrally local response-kernel prior on the nonseparable task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import nonseparable_response_relation as relation
import nonseparable_target_saturation as base
import torch
from torch import nn

SEEDS = (17, 29, 43)
BUDGETS = (1, 2, 4, 8)


class SpectralLocalKernel(nn.Module):
    """One shared scalar kernel over position and both mode indices."""

    def __init__(self, query_x: torch.Tensor, output_modes: int) -> None:
        super().__init__()
        x = query_x.reshape(-1)
        x_features = [x]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * math.pi * frequency * x
            x_features.extend((torch.sin(phase), torch.cos(phase)))
        x_features_tensor = torch.stack(x_features, dim=-1)

        output_index = torch.arange(
            output_modes,
            device=x.device,
            dtype=x.dtype,
        )
        input_index = torch.arange(
            relation.INPUT_CHANNELS,
            device=x.device,
            dtype=x.dtype,
        )
        output_normalized = output_index / max(output_modes - 1, 1)
        input_normalized = input_index / max(relation.INPUT_CHANNELS - 1, 1)
        difference = (
            output_index[:, None] - input_index[None, :]
        ) / max(output_modes - 1, 1)
        same_mode = (
            output_index[:, None] == input_index[None, :]
        ).to(dtype=x.dtype)
        mode_features = torch.stack(
            (
                output_normalized[:, None].expand(-1, relation.INPUT_CHANNELS),
                input_normalized[None, :].expand(output_modes, -1),
                difference,
                torch.abs(difference),
                same_mode,
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                x_features_tensor[:, None, None, :].expand(
                    -1,
                    output_modes,
                    relation.INPUT_CHANNELS,
                    -1,
                ),
                mode_features[None, :, :, :].expand(x.shape[0], -1, -1, -1),
            ),
            dim=-1,
        )
        self.register_buffer("features", features)
        self.network = nn.Sequential(
            nn.Linear(features.shape[-1], base.HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(base.HIDDEN_WIDTH, base.HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(base.HIDDEN_WIDTH, base.HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(base.HIDDEN_WIDTH, 1),
        )

    def forward(self) -> torch.Tensor:
        return self.network(self.features).squeeze(-1)


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
    query_x, target, drive_variances = base.exact_target(device)
    model = SpectralLocalKernel(query_x, target.shape[-2]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )

    drives: torch.Tensor | None = None
    fields: torch.Tensor | None = None
    data_diagnostics: dict[str, Any] | None = None
    if arm == "target":
        if budget not in BUDGETS:
            raise ValueError(f"target budget must be one of {BUDGETS}")
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
    initial_error = float(
        base.weighted_operator_error(model(), target, drive_variances)
        .detach()
        .cpu()
    )
    for step in range(1, base.STEPS + 1):
        kernel = model()
        if arm == "capacity":
            loss = base.weighted_operator_error(
                kernel,
                target,
                drive_variances,
            ).square()
        else:
            assert drives is not None and fields is not None
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
                        model(),
                        target,
                        drive_variances,
                    )
                    .detach()
                    .cpu()
                ),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    final_kernel = model().detach()
    return {
        "schema_version": 1,
        "study": "nonseparable_spectral_prior_v1",
        "preregistration": (
            "results/nonseparable_spectral_prior_preregistration_2026-07-29.json"
        ),
        "arm": arm,
        "budget": budget,
        "seed": seed,
        "device": str(device),
        "steps": base.STEPS,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_operator_relative_rms": initial_error,
        "final_operator_relative_rms": float(
            base.weighted_operator_error(
                final_kernel,
                target,
                drive_variances,
            ).cpu()
        ),
        "data_diagnostics": data_diagnostics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def reduce_reports(input_dir: Path) -> dict[str, Any]:
    capacity_values = []
    for seed in SEEDS:
        path = input_dir / f"capacity_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        capacity_values.append(
            float(json.loads(path.read_text())["final_operator_relative_rms"])
        )
    target: dict[int, list[float]] = {}
    for budget in BUDGETS:
        target[budget] = []
        for seed in SEEDS:
            path = input_dir / f"target_n{budget}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            target[budget].append(
                float(json.loads(path.read_text())["final_operator_relative_rms"])
            )

    capacity_geomean = geometric_mean(capacity_values)
    aggregate = {
        budget: {
            "geometric_mean": geometric_mean(values),
            "values": values,
        }
        for budget, values in target.items()
    }
    gates = {
        "capacity": (
            capacity_geomean <= 0.005 and max(capacity_values) <= 0.0075
        ),
        "target_threshold": aggregate[8]["geometric_mean"] <= 0.015,
        "one_field_not_saturated": aggregate[1]["geometric_mean"] >= 0.025,
        "material_improvement": (
            aggregate[4]["geometric_mean"]
            <= 0.8 * aggregate[1]["geometric_mean"]
            and aggregate[8]["geometric_mean"]
            <= 0.5 * aggregate[1]["geometric_mean"]
        ),
        "monotonicity": all(
            aggregate[right]["geometric_mean"]
            < aggregate[left]["geometric_mean"]
            for left, right in zip(BUDGETS[:-1], BUDGETS[1:])
        ),
    }
    return {
        "schema_version": 1,
        "study": "nonseparable_spectral_prior_v1",
        "capacity": {
            "geometric_mean": capacity_geomean,
            "values": capacity_values,
        },
        "target_only": aggregate,
        "gates": gates,
        "verdict": (
            "advance_to_source_transfer"
            if all(gates.values())
            else "reject_spectral_local_instrument"
        ),
    }


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
