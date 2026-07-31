# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pointwise and solution-aligned supervision of one learned kernel."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import drifted_screened_principal_part as base
import numpy as np
import torch
from provenance import runtime_environment, source_provenance

STUDY = "drifted_screened_supervision_v1"
ARMS = ("pointwise", "solution", "hybrid")
SEEDS = base.SEEDS
TRAINING_SEED = 149_000_081
BOUNDARY_SPECTRUM_SEED = 151_000_087
TRAIN_STEPS = 4_000
TRAIN_BOUNDARY_POINTS = 48
TRAIN_QUERY_POINTS = 64
TRAIN_QUADRATURE_ORDER = 8
LEARNING_RATE = 1.0e-3
TRAIN_SOLUTION_MODES = (0, 1, 2, 3)
HELD_OUT_SOLUTION_MODES = (5, 6, 7, 8)

EVALUATION_CASES = base.EVALUATION_CASES
EVALUATION_BOUNDARY_POINTS = base.EVALUATION_BOUNDARY_POINTS
EVALUATION_QUERY_POINTS = base.EVALUATION_QUERY_POINTS
RESOLUTION_CASES = base.RESOLUTION_CASES
RESOLUTIONS = base.RESOLUTIONS
QUADRATURE_ORDER = base.QUADRATURE_ORDER
CHECK_QUADRATURE_ORDER = base.CHECK_QUADRATURE_ORDER
KERNEL_EVALUATION_PAIRS = base.KERNEL_EVALUATION_PAIRS


@dataclass(frozen=True)
class LayerPairs:
    """Pair invariants and quadrature weights for one layer evaluation."""

    radius: torch.Tensor
    normal_dot_direction: torch.Tensor
    drift_dot_direction: torch.Tensor
    integration_weights: torch.Tensor


def layer_pairs(
    query_points: torch.Tensor,
    boundary: base.Mesh,
    *,
    drift: torch.Tensor,
    quadrature_order: int,
) -> LayerPairs:
    """Construct the exact panel pairs shared by every supervision arm."""

    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
    nodes_np = np.concatenate(((nodes_np - 1.0) / 2.0, (nodes_np + 1.0) / 2.0))
    weights_np = np.concatenate((weights_np / 2.0, weights_np / 2.0))
    nodes = torch.as_tensor(
        nodes_np, device=query_points.device, dtype=query_points.dtype
    )
    weights = torch.as_tensor(
        weights_np, device=query_points.device, dtype=query_points.dtype
    )
    vertices = boundary.points[boundary.cells]
    midpoint = vertices.mean(dim=1)
    half_edge = 0.5 * (vertices[:, 1] - vertices[:, 0])
    lengths = 2.0 * torch.linalg.vector_norm(half_edge, dim=-1)
    source_points = midpoint[:, None, :] + nodes[None, :, None] * half_edge[:, None, :]
    displacement = query_points[:, None, None, :] - source_points[None, :, :, :]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / radius[..., None]
    normal_dot_direction = torch.einsum(
        "qsgd,sd->qsg", direction, boundary.cell_normals
    )
    drift_dot_direction = torch.einsum("qsgd,d->qsg", direction, drift)
    integration_weights = weights[None, None, :] * (0.5 * lengths)[None, :, None]
    return LayerPairs(
        radius=radius,
        normal_dot_direction=normal_dot_direction,
        drift_dot_direction=drift_dot_direction,
        integration_weights=integration_weights,
    )


def scaled_kernel(
    model: base.ScaledKernelModel | None,
    pairs: LayerPairs,
    *,
    kappa: float,
    drift_magnitude: float,
) -> torch.Tensor:
    """Evaluate a learned or exact scaled kernel without detaching gradients."""

    radius = pairs.radius
    kappa_values = torch.full_like(radius, kappa)
    drift_values = torch.full_like(radius, drift_magnitude)
    if model is None:
        return base.exact_scaled_double_kernel(
            radius,
            pairs.normal_dot_direction,
            pairs.drift_dot_direction,
            kappa_values,
            drift_values,
        )
    shape = radius.shape
    return model(
        radius.reshape(-1),
        pairs.normal_dot_direction.reshape(-1),
        pairs.drift_dot_direction.reshape(-1),
        kappa_values.reshape(-1),
        drift_values.reshape(-1),
    ).reshape(shape)


def integrate_scaled_kernel(
    values: torch.Tensor,
    pairs: LayerPairs,
) -> torch.Tensor:
    """Integrate scaled-kernel values into a query-by-panel influence matrix."""

    return ((values / pairs.radius) * pairs.integration_weights).sum(dim=-1)


def training_losses(
    model: base.ScaledKernelModel,
    sample: base.DriftedScreenedSample,
    *,
    quadrature_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized pointwise-kernel and end-to-end solution losses."""

    boundary_pairs = layer_pairs(
        sample.boundary.cell_centroids,
        sample.boundary,
        drift=sample.drift,
        quadrature_order=quadrature_order,
    )
    field_pairs = layer_pairs(
        sample.query_points,
        sample.boundary,
        drift=sample.drift,
        quadrature_order=quadrature_order,
    )
    drift_magnitude = float(torch.linalg.vector_norm(sample.drift))
    learned_boundary = scaled_kernel(
        model,
        boundary_pairs,
        kappa=sample.kappa,
        drift_magnitude=drift_magnitude,
    )
    learned_field = scaled_kernel(
        model,
        field_pairs,
        kappa=sample.kappa,
        drift_magnitude=drift_magnitude,
    )
    exact_boundary = scaled_kernel(
        None,
        boundary_pairs,
        kappa=sample.kappa,
        drift_magnitude=drift_magnitude,
    )
    exact_field = scaled_kernel(
        None,
        field_pairs,
        kappa=sample.kappa,
        drift_magnitude=drift_magnitude,
    )
    scale = base.DOUBLE_COEFFICIENT**2
    kernel_loss = (
        0.5
        * (
            (learned_boundary - exact_boundary).square().mean()
            + (learned_field - exact_field).square().mean()
        )
        / scale
    )

    learned_trace = integrate_scaled_kernel(learned_boundary, boundary_pairs)
    learned_trace = learned_trace + 0.5 * torch.eye(
        sample.boundary.n_cells,
        device=learned_trace.device,
        dtype=learned_trace.dtype,
    )
    learned_field_matrix = integrate_scaled_kernel(learned_field, field_pairs)
    learned_density = torch.linalg.solve(learned_trace, sample.boundary_values)
    prediction = learned_field_matrix @ learned_density
    denominator = (
        sample.target.square().sum().clamp_min(torch.finfo(sample.target.dtype).tiny)
    )
    solution_loss = (prediction - sample.target).square().sum() / denominator
    return kernel_loss, solution_loss


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    n_boundary: int,
    n_query: int,
    quadrature_order: int,
    feature_system: str = "raw",
) -> tuple[base.ScaledKernelModel, list[dict[str, float]]]:
    """Train identical free-principal models under one of three losses."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = base.ScaledKernelModel("free_principal", feature_system=feature_system).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-6
    )
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        sample = base.build_pde_sample(
            TRAINING_SEED + step,
            split="in_distribution",
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=torch.float32,
            solution_modes=TRAIN_SOLUTION_MODES,
        )
        kernel_loss, solution_loss = training_losses(
            model,
            sample,
            quadrature_order=quadrature_order,
        )
        if arm == "pointwise":
            loss = kernel_loss
        elif arm == "solution":
            loss = solution_loss
        else:
            loss = kernel_loss + solution_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            record = {
                "step": step,
                "optimized_loss": float(loss.detach()),
                "pointwise_kernel_loss": float(kernel_loss.detach()),
                "solution_loss": float(solution_loss.detach()),
            }
            history.append(record)
            print(
                f"HEARTBEAT phase=train completed_units={step} "
                f"arm={arm} seed={seed} loss={record['optimized_loss']:.6e}",
                flush=True,
            )
    return model, history


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "maximum": float(tensor.max()),
    }


def evaluate_boundary_spectrum(
    model: base.ScaledKernelModel,
    *,
    device: torch.device,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    quadrature_order: int,
    check_quadrature_order: int,
) -> dict[str, Any]:
    """Evaluate held-out boundary modes on in-distribution operators."""

    cases = [
        base.evaluate_pde_case(
            model,
            seed=BOUNDARY_SPECTRUM_SEED + index,
            split="in_distribution",
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
            solution_modes=HELD_OUT_SOLUTION_MODES,
        )
        for index in range(n_cases)
    ]
    return {
        "solution_modes": list(HELD_OUT_SOLUTION_MODES),
        "metrics": {
            key: _summary([case[key] for case in cases]) for key in base.PDE_METRICS
        },
        "cases": cases,
    }


def run_arm(
    *,
    arm: str,
    seed: int,
    device: torch.device,
    train_steps: int = TRAIN_STEPS,
    train_boundary_points: int = TRAIN_BOUNDARY_POINTS,
    train_query_points: int = TRAIN_QUERY_POINTS,
    train_quadrature_order: int = TRAIN_QUADRATURE_ORDER,
    evaluation_cases: int = EVALUATION_CASES,
    evaluation_boundary_points: int = EVALUATION_BOUNDARY_POINTS,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
    resolution_cases: int = RESOLUTION_CASES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
    kernel_evaluation_pairs: int = KERNEL_EVALUATION_PAIRS,
) -> dict[str, Any]:
    """Train and evaluate one registered arm/seed pair."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    started = time.time()
    model, history = train_model(
        arm,
        seed=seed,
        device=device,
        steps=train_steps,
        n_boundary=train_boundary_points,
        n_query=train_query_points,
        quadrature_order=train_quadrature_order,
    )
    model = model.to(device=device, dtype=torch.float64).eval()
    return {
        "study": STUDY,
        "arm": arm,
        "seed": seed,
        "protocol": {
            "train_steps": train_steps,
            "train_boundary_points": train_boundary_points,
            "train_query_points": train_query_points,
            "train_quadrature_order_per_half_panel": train_quadrature_order,
            "train_solution_modes": list(TRAIN_SOLUTION_MODES),
            "held_out_solution_modes": list(HELD_OUT_SOLUTION_MODES),
            "learning_rate": LEARNING_RATE,
            "evaluation_cases_per_split": evaluation_cases,
            "evaluation_boundary_points": evaluation_boundary_points,
            "evaluation_query_points": evaluation_query_points,
            "resolution_cases_per_split": resolution_cases,
            "resolutions": list(resolutions),
            "quadrature_order_per_half_panel": quadrature_order,
            "check_quadrature_order_per_half_panel": check_quadrature_order,
            "kernel_evaluation_pairs_per_split": kernel_evaluation_pairs,
            "dtype_training": "float32",
            "dtype_evaluation": "float64",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": history,
        "learned_singular_coefficient": float(model.singular_coefficient.detach()),
        "kernel_evaluation": base.evaluate_kernel_splits(
            model,
            device=device,
            n_pairs=kernel_evaluation_pairs,
        ),
        "pde_evaluation": base.evaluate_pde_splits(
            model,
            device=device,
            n_cases=evaluation_cases,
            n_boundary=evaluation_boundary_points,
            n_query=evaluation_query_points,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        ),
        "boundary_spectrum_evaluation": evaluate_boundary_spectrum(
            model,
            device=device,
            n_cases=evaluation_cases,
            n_boundary=evaluation_boundary_points,
            n_query=evaluation_query_points,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        ),
        "resolution_evaluation": base.evaluate_resolution_splits(
            model,
            device=device,
            n_cases=resolution_cases,
            resolutions=resolutions,
            n_query=evaluation_query_points,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        ),
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "elapsed_seconds": time.time() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument(
        "--train-boundary-points", type=int, default=TRAIN_BOUNDARY_POINTS
    )
    parser.add_argument("--train-query-points", type=int, default=TRAIN_QUERY_POINTS)
    parser.add_argument(
        "--train-quadrature-order", type=int, default=TRAIN_QUADRATURE_ORDER
    )
    parser.add_argument("--evaluation-cases", type=int, default=EVALUATION_CASES)
    parser.add_argument(
        "--evaluation-boundary-points", type=int, default=EVALUATION_BOUNDARY_POINTS
    )
    parser.add_argument(
        "--evaluation-query-points", type=int, default=EVALUATION_QUERY_POINTS
    )
    parser.add_argument("--resolution-cases", type=int, default=RESOLUTION_CASES)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(RESOLUTIONS),
    )
    parser.add_argument("--quadrature-order", type=int, default=QUADRATURE_ORDER)
    parser.add_argument(
        "--check-quadrature-order", type=int, default=CHECK_QUADRATURE_ORDER
    )
    parser.add_argument(
        "--kernel-evaluation-pairs", type=int, default=KERNEL_EVALUATION_PAIRS
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_arm(
        arm=args.arm,
        seed=args.seed,
        device=torch.device(args.device),
        train_steps=args.train_steps,
        train_boundary_points=args.train_boundary_points,
        train_query_points=args.train_query_points,
        train_quadrature_order=args.train_quadrature_order,
        evaluation_cases=args.evaluation_cases,
        evaluation_boundary_points=args.evaluation_boundary_points,
        evaluation_query_points=args.evaluation_query_points,
        resolution_cases=args.resolution_cases,
        resolutions=tuple(args.resolutions),
        quadrature_order=args.quadrature_order,
        check_quadrature_order=args.check_quadrature_order,
        kernel_evaluation_pairs=args.kernel_evaluation_pairs,
    )
    base.atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "seed": report["seed"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
