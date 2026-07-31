# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test a fixed canonical double-layer processor across screened Laplace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from laplace_readout_factorial import atomic_write_json
from provenance import runtime_environment, source_provenance
from screened_laplace import SPLITS, ScreenedSample, build_screened_sample
from screened_trace_mechanism import (
    DOUBLE_COEFFICIENT,
    field_matrix,
    relative_l2,
    richardson,
    trace_matrix,
)

from physicsnemo.mesh import Mesh

STUDY = "screened_canonical_double_layer_v1"
EVALUATION_SEED = 97_000_037
RESOLUTION_SEED = 101_000_037
SPLIT_ORDER = tuple(sorted(SPLITS))
EVALUATION_CASES = 64
EVALUATION_BOUNDARY_POINTS = 64
EVALUATION_QUERY_POINTS = 512
RESOLUTION_CASES = 8
RESOLUTIONS = (64, 128, 256)
QUADRATURE_ORDER = 64
CHECK_QUADRATURE_ORDER = 32
RICHARDSON_STEPS = 8


def _normalized_geometry(sample: ScreenedSample) -> tuple[Mesh, torch.Tensor]:
    boundary = sample.domain.boundaries["dirichlet"]
    length = sample.domain.global_data["reference_length"].reshape(())
    weights = boundary.cell_areas / length
    center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
    center = center / weights.sum()
    normalized = Mesh(
        points=(boundary.points - center) / length,
        cells=boundary.cells,
    )
    queries = (sample.domain.interior.points - center) / length
    if torch.any(
        torch.sum(normalized.cell_normals * normalized.cell_centroids, dim=-1) >= 0
    ):
        raise RuntimeError("the registered control requires inward panel normals")
    return normalized, queries


def _normalized_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
) -> float:
    denominator = torch.linalg.vector_norm(target).clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return float(torch.linalg.vector_norm(first - second) / denominator)


def evaluate_case(
    *,
    seed: int,
    split: str,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
) -> dict[str, float]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    sample = build_screened_sample(
        seed,
        **SPLITS[split],
        n_boundary=n_boundary,
        n_query=n_query,
        device=device,
        dtype=torch.float64,
    )
    boundary, queries = _normalized_geometry(sample)
    kappa = sample.kappa_tilde
    matrix = trace_matrix(
        boundary,
        layer="double",
        kappa=kappa,
        quadrature_order=quadrature_order,
        zero_diagonal=False,
    )
    check_matrix = trace_matrix(
        boundary,
        layer="double",
        kappa=kappa,
        quadrature_order=check_quadrature_order,
        zero_diagonal=False,
    )
    propagation = field_matrix(
        queries,
        boundary,
        layer="double",
        kappa=kappa,
        quadrature_order=quadrature_order,
    )
    values = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    dense_density = torch.linalg.solve(matrix, values)
    iterative_density = richardson(matrix, values, steps=RICHARDSON_STEPS)
    dense_field = propagation @ dense_density
    iterative_field = propagation @ iterative_density
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return {
        "kappa_tilde": kappa,
        "dense_field_relative_l2": relative_l2(dense_field, sample.target),
        "iterative_field_relative_l2": relative_l2(iterative_field, sample.target),
        "iterative_minus_dense_relative_target_l2": _normalized_difference(
            iterative_field, dense_field, sample.target
        ),
        "dense_trace_relative_l2": relative_l2(matrix @ dense_density, values),
        "iterative_trace_relative_l2": relative_l2(matrix @ iterative_density, values),
        "density_relative_l2": relative_l2(iterative_density, dense_density),
        "condition_number": float(torch.linalg.cond(matrix)),
        "unit_richardson_spectral_radius": float(
            torch.linalg.eigvals(identity - matrix).abs().max()
        ),
        "quadrature_32_to_64_relative_frobenius": relative_l2(check_matrix, matrix),
    }


def _summary(cases: list[dict[str, float]], key: str) -> dict[str, float]:
    values = torch.tensor([case[key] for case in cases], dtype=torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "maximum": float(values.max()),
    }


CASE_METRICS = (
    "dense_field_relative_l2",
    "iterative_field_relative_l2",
    "iterative_minus_dense_relative_target_l2",
    "dense_trace_relative_l2",
    "iterative_trace_relative_l2",
    "density_relative_l2",
    "condition_number",
    "unit_richardson_spectral_radius",
    "quadrature_32_to_64_relative_frobenius",
)


def evaluate_split_bank(
    *,
    eval_seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    quadrature_order: int,
    check_quadrature_order: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        cases = [
            evaluate_case(
                seed=eval_seed + 7_919 * case + 1_000_003 * split_index,
                split=split,
                n_boundary=n_boundary,
                n_query=n_query,
                device=device,
                quadrature_order=quadrature_order,
                check_quadrature_order=check_quadrature_order,
            )
            for case in range(n_cases)
        ]
        result[split] = {
            "metrics": {key: _summary(cases, key) for key in CASE_METRICS},
            "cases": cases,
        }
        print(
            f"HEARTBEAT phase=evaluation completed_units={split_index + 1} "
            f"split={split}",
            flush=True,
        )
    return result


def evaluate_resolution_bank(
    *,
    resolution_seed: int,
    n_cases: int,
    resolutions: tuple[int, ...],
    n_query: int,
    device: torch.device,
    quadrature_order: int,
    check_quadrature_order: int,
) -> dict[str, Any]:
    if tuple(sorted(set(resolutions))) != resolutions:
        raise ValueError("resolutions must be unique and increasing")
    result: dict[str, Any] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        per_resolution: dict[int, list[dict[str, float]]] = {
            resolution: [] for resolution in resolutions
        }
        for case in range(n_cases):
            seed = resolution_seed + 7_919 * case + 1_000_003 * split_index
            targets: list[torch.Tensor] = []
            queries: list[torch.Tensor] = []
            for resolution in resolutions:
                sample = build_screened_sample(
                    seed,
                    **SPLITS[split],
                    n_boundary=resolution,
                    n_query=n_query,
                    device=device,
                    dtype=torch.float64,
                )
                targets.append(sample.target)
                queries.append(sample.domain.interior.points)
                per_resolution[resolution].append(
                    evaluate_case(
                        seed=seed,
                        split=split,
                        n_boundary=resolution,
                        n_query=n_query,
                        device=device,
                        quadrature_order=quadrature_order,
                        check_quadrature_order=check_quadrature_order,
                    )
                )
            if not all(
                torch.equal(targets[0], target) and torch.equal(queries[0], query)
                for target, query in zip(targets[1:], queries[1:])
            ):
                raise RuntimeError(
                    "resolution ladder did not preserve the continuum problem"
                )
        means = {
            str(resolution): {
                "dense_field_relative_l2_mean": _summary(
                    per_resolution[resolution], "dense_field_relative_l2"
                )["mean"],
                "iterative_field_relative_l2_mean": _summary(
                    per_resolution[resolution], "iterative_field_relative_l2"
                )["mean"],
                "iterative_minus_dense_relative_target_l2_mean": _summary(
                    per_resolution[resolution],
                    "iterative_minus_dense_relative_target_l2",
                )["mean"],
            }
            for resolution in resolutions
        }
        iterative_errors = [
            means[str(resolution)]["iterative_field_relative_l2_mean"]
            for resolution in resolutions
        ]
        dense_errors = [
            means[str(resolution)]["dense_field_relative_l2_mean"]
            for resolution in resolutions
        ]
        result[split] = {
            "resolutions": means,
            "iterative_monotone": all(
                later <= earlier
                for earlier, later in zip(iterative_errors, iterative_errors[1:])
            ),
            "dense_monotone": all(
                later <= earlier
                for earlier, later in zip(dense_errors, dense_errors[1:])
            ),
        }
        print(
            f"HEARTBEAT phase=resolution "
            f"completed_units={len(SPLIT_ORDER) + split_index + 1} split={split}",
            flush=True,
        )
    return result


def apply_registered_decision(
    evaluation: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    split_checks: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        metrics = evaluation[split]["metrics"]
        split_checks[split] = {
            "mean_dense_discrepancy_at_most_0_005": (
                metrics["iterative_minus_dense_relative_target_l2"]["mean"] <= 0.005
            ),
            "max_dense_discrepancy_at_most_0_02": (
                metrics["iterative_minus_dense_relative_target_l2"]["maximum"] <= 0.02
            ),
            "mean_trace_residual_at_most_0_005": (
                metrics["iterative_trace_relative_l2"]["mean"] <= 0.005
            ),
            "max_trace_residual_at_most_0_02": (
                metrics["iterative_trace_relative_l2"]["maximum"] <= 0.02
            ),
            "mean_field_error_at_most_0_10": (
                metrics["iterative_field_relative_l2"]["mean"] <= 0.10
            ),
            "resolution_monotone": resolution[split]["iterative_monotone"],
        }
    dense_trace_sanity = all(
        evaluation[split]["metrics"]["dense_trace_relative_l2"]["mean"] <= 1.0e-10
        for split in SPLIT_ORDER
    )
    quadrature_resolved = (
        max(
            evaluation[split]["metrics"]["quadrature_32_to_64_relative_frobenius"][
                "maximum"
            ]
            for split in SPLIT_ORDER
        )
        <= 1.0e-3
    )
    fixed_processor_sufficient = (
        dense_trace_sanity
        and quadrature_resolved
        and all(all(checks.values()) for checks in split_checks.values())
    )
    dense_field_stable = all(
        evaluation[split]["metrics"]["dense_field_relative_l2"]["mean"] <= 0.10
        for split in SPLIT_ORDER
    )
    if not quadrature_resolved or not dense_trace_sanity:
        verdict = "numerically_unresolved"
    elif fixed_processor_sufficient:
        verdict = "fixed_double_layer_processor_sufficient"
    elif dense_field_stable:
        verdict = "parameter_conditioned_preconditioner_earned"
    else:
        verdict = "boundary_representation_or_kernel_bottleneck"
    return {
        "verdict": verdict,
        "fixed_processor_sufficient": fixed_processor_sufficient,
        "dense_trace_sanity": dense_trace_sanity,
        "quadrature_resolved": quadrature_resolved,
        "dense_field_stable": dense_field_stable,
        "split_checks": split_checks,
    }


def run_study(
    *,
    device: torch.device,
    evaluation_cases: int = EVALUATION_CASES,
    evaluation_boundary_points: int = EVALUATION_BOUNDARY_POINTS,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
    resolution_cases: int = RESOLUTION_CASES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
) -> dict[str, Any]:
    evaluation = evaluate_split_bank(
        eval_seed=EVALUATION_SEED,
        n_cases=evaluation_cases,
        n_boundary=evaluation_boundary_points,
        n_query=evaluation_query_points,
        device=device,
        quadrature_order=quadrature_order,
        check_quadrature_order=check_quadrature_order,
    )
    resolution = evaluate_resolution_bank(
        resolution_seed=RESOLUTION_SEED,
        n_cases=resolution_cases,
        resolutions=resolutions,
        n_query=evaluation_query_points,
        device=device,
        quadrature_order=quadrature_order,
        check_quadrature_order=check_quadrature_order,
    )
    report: dict[str, Any] = {
        "study": STUDY,
        "protocol": {
            "evaluation_seed": EVALUATION_SEED,
            "resolution_seed": RESOLUTION_SEED,
            "splits": list(SPLIT_ORDER),
            "evaluation_cases_per_split": evaluation_cases,
            "evaluation_boundary_points": evaluation_boundary_points,
            "evaluation_query_points": evaluation_query_points,
            "resolution_cases_per_split": resolution_cases,
            "resolutions": list(resolutions),
            "quadrature_order": quadrature_order,
            "check_quadrature_order": check_quadrature_order,
            "richardson_steps": RICHARDSON_STEPS,
            "double_coefficient": DOUBLE_COEFFICIENT,
            "jump": 0.5,
            "normal_orientation": "inward",
            "dtype": "float64",
        },
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "evaluation": evaluation,
        "resolution": resolution,
    }
    if (
        evaluation_cases == EVALUATION_CASES
        and evaluation_boundary_points == EVALUATION_BOUNDARY_POINTS
        and evaluation_query_points == EVALUATION_QUERY_POINTS
        and resolution_cases == RESOLUTION_CASES
        and resolutions == RESOLUTIONS
        and quadrature_order == QUADRATURE_ORDER
        and check_quadrature_order == CHECK_QUADRATURE_ORDER
    ):
        report["decision"] = apply_registered_decision(evaluation, resolution)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_study(device=torch.device(args.device))
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "study": STUDY,
                "output": str(args.output),
                "decision": report["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
