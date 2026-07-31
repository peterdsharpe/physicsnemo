# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test raw operator inputs against physically sufficient similarity variables."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import drifted_screened_principal_part as base
import drifted_screened_supervision as supervision
import torch
from provenance import runtime_environment, source_provenance

STUDY = "drifted_screened_similarity_v1"
ARM_SPECS = {
    "raw_pointwise": ("raw", "pointwise"),
    "similarity_pointwise": ("similarity", "pointwise"),
    "raw_hybrid": ("raw", "hybrid"),
    "similarity_hybrid": ("similarity", "hybrid"),
}
ARMS = tuple(ARM_SPECS)
SEEDS = supervision.SEEDS

TRAIN_STEPS = supervision.TRAIN_STEPS
TRAIN_BOUNDARY_POINTS = supervision.TRAIN_BOUNDARY_POINTS
TRAIN_QUERY_POINTS = supervision.TRAIN_QUERY_POINTS
TRAIN_QUADRATURE_ORDER = supervision.TRAIN_QUADRATURE_ORDER
LEARNING_RATE = supervision.LEARNING_RATE
TRAIN_SOLUTION_MODES = supervision.TRAIN_SOLUTION_MODES
HELD_OUT_SOLUTION_MODES = supervision.HELD_OUT_SOLUTION_MODES

EVALUATION_CASES = supervision.EVALUATION_CASES
EVALUATION_BOUNDARY_POINTS = supervision.EVALUATION_BOUNDARY_POINTS
EVALUATION_QUERY_POINTS = supervision.EVALUATION_QUERY_POINTS
RESOLUTION_CASES = supervision.RESOLUTION_CASES
RESOLUTIONS = supervision.RESOLUTIONS
QUADRATURE_ORDER = supervision.QUADRATURE_ORDER
CHECK_QUADRATURE_ORDER = supervision.CHECK_QUADRATURE_ORDER
KERNEL_EVALUATION_PAIRS = supervision.KERNEL_EVALUATION_PAIRS


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
    """Train and evaluate one registered coordinate-by-loss arm."""

    if arm not in ARM_SPECS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    feature_system, loss = ARM_SPECS[arm]
    started = time.time()
    model, history = supervision.train_model(
        loss,
        seed=seed,
        device=device,
        steps=train_steps,
        n_boundary=train_boundary_points,
        n_query=train_query_points,
        quadrature_order=train_quadrature_order,
        feature_system=feature_system,
    )
    model = model.to(device=device, dtype=torch.float64).eval()
    return {
        "study": STUDY,
        "arm": arm,
        "seed": seed,
        "feature_system": feature_system,
        "loss": loss,
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
        "boundary_spectrum_evaluation": supervision.evaluate_boundary_spectrum(
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
