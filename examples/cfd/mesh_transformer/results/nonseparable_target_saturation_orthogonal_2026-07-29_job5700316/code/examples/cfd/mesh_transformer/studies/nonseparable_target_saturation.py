# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Measure target-field saturation on the admitted nonseparable operator."""

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
import torch
from torch import nn

STEPS = 6_000
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
HIDDEN_WIDTH = 128
HIDDEN_LAYERS = 3
BUDGETS = (1, 2, 4, 8)
SEEDS = (17, 29, 43)


class KernelField(nn.Module):
    """Coordinate network returning eight boundary-response coefficients."""

    def __init__(self, query_x: torch.Tensor, output_modes: int) -> None:
        super().__init__()
        x = query_x.reshape(-1)
        coordinate_features = [x]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * math.pi * frequency * x
            coordinate_features.extend((torch.sin(phase), torch.cos(phase)))
        x_features = torch.stack(coordinate_features, dim=-1)
        one_hot = torch.eye(
            output_modes,
            device=query_x.device,
            dtype=query_x.dtype,
        )
        features = torch.cat(
            (
                x_features[:, None, :].expand(-1, output_modes, -1),
                one_hot[None, :, :].expand(x.shape[0], -1, -1),
            ),
            dim=-1,
        )
        self.register_buffer("features", features)

        layers: list[nn.Module] = [
            nn.Linear(features.shape[-1], HIDDEN_WIDTH),
            nn.SiLU(),
        ]
        for _ in range(HIDDEN_LAYERS - 1):
            layers.extend((nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH), nn.SiLU()))
        layers.append(nn.Linear(HIDDEN_WIDTH, relation.INPUT_CHANNELS))
        self.network = nn.Sequential(*layers)

    def forward(self) -> torch.Tensor:
        return self.network(self.features)


def weighted_operator_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    drive_variances: torch.Tensor,
) -> torch.Tensor:
    error = torch.mean(
        torch.sum(
            (prediction - target).square()
            * drive_variances[None, None, :],
            dim=(-2, -1),
        )
    )
    scale = torch.mean(
        torch.sum(
            target.square() * drive_variances[None, None, :],
            dim=(-2, -1),
        )
    )
    return torch.sqrt(error / scale)


def exact_target(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    family = relation.ResponseFamily(
        relation.RESOLUTIONS[-1],
        device=device,
        dtype=torch.float64,
    )
    target = family.response(relation.TARGET_LENGTH, relation.TARGET_SCREENING)
    variances = 1.0 / relation.column_energies(target)
    variances = variances / variances.mean()
    return family.query_x.float(), target.float(), variances.float()


def sampled_fields(
    target: torch.Tensor,
    drive_variances: torch.Tensor,
    *,
    budget: int,
    seed: int,
    design: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if design == "gaussian":
        generator = torch.Generator(device=target.device).manual_seed(
            1_000_003 * seed + 97 * budget
        )
        whitened = torch.randn(
            budget,
            relation.INPUT_CHANNELS,
            generator=generator,
            device=target.device,
            dtype=target.dtype,
        )
    elif design == "orthogonal":
        generator = torch.Generator(device=target.device).manual_seed(
            1_000_003 * seed + 193
        )
        matrix = torch.randn(
            relation.INPUT_CHANNELS,
            relation.INPUT_CHANNELS,
            generator=generator,
            device=target.device,
            dtype=target.dtype,
        )
        orthogonal, triangular = torch.linalg.qr(matrix)
        signs = torch.sign(torch.diagonal(triangular))
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        orthogonal = orthogonal * signs[None, :]
        whitened = math.sqrt(relation.INPUT_CHANNELS) * orthogonal[:budget]
    else:
        raise ValueError("design must be 'gaussian' or 'orthogonal'")
    drives = whitened * torch.sqrt(drive_variances)[None, :]
    fields = torch.einsum("qoi,bi->bqo", target, drives)
    singular_values = torch.linalg.svdvals(whitened)
    diagnostics = {
        "design": design,
        "whitened_drive_rank": int(torch.linalg.matrix_rank(whitened).cpu()),
        "whitened_drive_singular_values": [
            float(value.cpu()) for value in singular_values
        ],
        "whitened_drive_condition_number": (
            float((singular_values[0] / singular_values[-1]).cpu())
            if budget >= relation.INPUT_CHANNELS
            else None
        ),
    }
    return drives, fields, diagnostics


def run_arm(
    *,
    arm: str,
    budget: int | None,
    seed: int,
    device: torch.device,
    design: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    query_x, target, drive_variances = exact_target(device)
    model = KernelField(query_x, target.shape[-2]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    drives: torch.Tensor | None = None
    fields: torch.Tensor | None = None
    data_diagnostics: dict[str, Any] | None = None
    if arm == "target":
        if budget not in BUDGETS:
            raise ValueError(f"target budget must be one of {BUDGETS}")
        drives, fields, data_diagnostics = sampled_fields(
            target,
            drive_variances,
            budget=budget,
            seed=seed,
            design=design,
        )
    elif arm != "capacity":
        raise ValueError("arm must be 'capacity' or 'target'")

    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    initial_error = float(
        weighted_operator_error(model(), target, drive_variances).detach().cpu()
    )
    for step in range(1, STEPS + 1):
        kernel = model()
        if arm == "capacity":
            loss = weighted_operator_error(kernel, target, drive_variances).square()
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
        if step == 1 or step % 1_000 == 0 or step == STEPS:
            record = {
                "step": step,
                "training_relative_rms": math.sqrt(float(loss.detach().cpu())),
                "held_out_operator_relative_rms": float(
                    weighted_operator_error(
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
    per_input_errors = []
    for index in range(relation.INPUT_CHANNELS):
        numerator = torch.mean(
            torch.sum(
                (final_kernel[..., index] - target[..., index]).square(),
                dim=-1,
            )
        )
        denominator = torch.mean(
            torch.sum(target[..., index].square(), dim=-1)
        )
        per_input_errors.append(float(torch.sqrt(numerator / denominator).cpu()))
    return {
        "schema_version": 1,
        "study": "nonseparable_target_saturation_v1",
        "preregistration": (
            "results/nonseparable_target_saturation_orthogonal_amendment_2026-07-29.json"
            if arm == "target" and design == "orthogonal"
            else "results/nonseparable_target_saturation_preregistration_2026-07-29.json"
        ),
        "arm": arm,
        "target_design": design if arm == "target" else None,
        "budget": budget,
        "seed": seed,
        "device": str(device),
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_operator_relative_rms": initial_error,
        "final_operator_relative_rms": float(
            weighted_operator_error(final_kernel, target, drive_variances).cpu()
        ),
        "per_input_mode_relative_rms": per_input_errors,
        "data_diagnostics": data_diagnostics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def reduce_reports(input_dir: Path, *, design: str) -> dict[str, Any]:
    capacity_path = input_dir / "capacity_seed17.json"
    if not capacity_path.is_file():
        raise FileNotFoundError(capacity_path)
    capacity = json.loads(capacity_path.read_text())
    target_reports: dict[int, list[dict[str, Any]]] = {}
    for budget in BUDGETS:
        reports = []
        for seed in SEEDS:
            prefix = "target" if design == "gaussian" else f"target_{design}"
            path = input_dir / f"{prefix}_n{budget}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            reports.append(json.loads(path.read_text()))
        target_reports[budget] = reports

    errors = {
        budget: [
            float(report["final_operator_relative_rms"])
            for report in target_reports[budget]
        ]
        for budget in BUDGETS
    }
    aggregate = {
        budget: {
            "geometric_mean": geometric_mean(values),
            "values": values,
        }
        for budget, values in errors.items()
    }
    gates = {
        "capacity": float(capacity["final_operator_relative_rms"]) <= 0.005,
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
        "study": "nonseparable_target_saturation_v1",
        "target_design": design,
        "capacity_control_relative_rms": float(
            capacity["final_operator_relative_rms"]
        ),
        "target_only": aggregate,
        "gates": gates,
        "verdict": (
            "advance_to_source_transfer"
            if all(gates.values())
            else "reject_learning_instrument"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--arm", choices=("capacity", "target"), required=True)
    run_parser.add_argument("--budget", type=int)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument(
        "--design",
        choices=("gaussian", "orthogonal"),
        default="gaussian",
    )
    run_parser.add_argument("--output", type=Path, required=True)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--input-dir", type=Path, required=True)
    reduce_parser.add_argument(
        "--design",
        choices=("gaussian", "orthogonal"),
        required=True,
    )
    reduce_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        report = run_arm(
            arm=args.arm,
            budget=args.budget,
            seed=args.seed,
            device=torch.device(args.device),
            design=args.design,
        )
    else:
        report = reduce_reports(args.input_dir, design=args.design)
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
