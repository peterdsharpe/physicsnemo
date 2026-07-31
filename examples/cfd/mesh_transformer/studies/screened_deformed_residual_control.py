# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test residual-controlled screened double layers on deformed geometries."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from laplace_readout_factorial import atomic_write_json
from provenance import runtime_environment, source_provenance
from screened_laplace import modified_bessel_i
from screened_trace_mechanism import field_matrix, relative_l2, trace_matrix

from physicsnemo.mesh import Mesh

STUDY = "screened_deformed_residual_control_v1"
EVALUATION_SEED = 109_000_037
RESOLUTION_SEED = 113_000_039
EVALUATION_CASES = 32
EVALUATION_BOUNDARY_POINTS = 128
EVALUATION_QUERY_POINTS = 512
RESOLUTION_CASES = 8
RESOLUTIONS = (64, 128, 256)
QUADRATURE_ORDER = 64
CHECK_QUADRATURE_ORDER = 32
TRACE_TOLERANCE = 1.0e-6
MAX_ITERATIONS = 32

SPLITS: dict[str, dict[str, Any]] = {
    "in_distribution": {
        "kappa_range": (0.5, 2.0),
        "deformation_range": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "solution_modes": (0, 1, 2, 3),
    },
    "stronger_deformation": {
        "kappa_range": (0.5, 2.0),
        "deformation_range": (0.2, 0.3),
        "geometry_modes": (2, 3, 4),
        "solution_modes": (0, 1, 2, 3),
    },
    "unseen_geometry_modes": {
        "kappa_range": (0.5, 2.0),
        "deformation_range": (0.05, 0.15),
        "geometry_modes": (5, 6, 7),
        "solution_modes": (0, 1, 2, 3),
    },
    "ood_low_screening": {
        "kappa_range": (0.05, 0.3),
        "deformation_range": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "solution_modes": (0, 1, 2, 3),
    },
    "ood_high_screening": {
        "kappa_range": (3.0, 5.0),
        "deformation_range": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "solution_modes": (0, 1, 2, 3),
    },
}
SPLIT_ORDER = tuple(SPLITS)


@dataclass(frozen=True)
class DeformedScreenedSample:
    boundary: Mesh
    query_points: torch.Tensor
    boundary_values: torch.Tensor
    target: torch.Tensor
    center: torch.Tensor
    reference_length: torch.Tensor
    kappa_tilde: float
    deformation: float
    solution_modes: tuple[int, ...]
    solution_phases: torch.Tensor


def regular_screened_field(
    points: torch.Tensor,
    *,
    kappa_tilde: float,
    modes: tuple[int, ...],
    phases: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a globally regular exact solution of screened Laplace."""

    radius = points.norm(dim=-1)
    angle = torch.atan2(points[:, 1], points[:, 0])
    result = torch.zeros_like(radius)
    scale = 1.0 / math.sqrt(len(modes))
    kappa = torch.as_tensor(kappa_tilde, device=points.device, dtype=points.dtype)
    for index, mode in enumerate(modes):
        radial = modified_bessel_i(mode, kappa * radius) / modified_bessel_i(
            mode, kappa
        )
        result = result + scale * radial * torch.cos(mode * angle + phases[index])
    return result


def _radial_boundary(
    angles: torch.Tensor,
    *,
    deformation: float,
    modes: tuple[int, ...],
    phases: torch.Tensor,
) -> torch.Tensor:
    result = torch.ones_like(angles)
    for index, mode in enumerate(modes):
        result = result + (deformation / len(modes)) * torch.cos(
            mode * angles + phases[index]
        )
    return result


def build_deformed_sample(
    seed: int,
    *,
    split: str,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> DeformedScreenedSample:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    spec = SPLITS[split]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(
            torch.empty((), dtype=torch.float64).uniform_(
                low, high, generator=generator
            )
        )

    center = torch.tensor((uniform(-2.0, 2.0), uniform(-2.0, 2.0)), dtype=torch.float64)
    reference_length = uniform(0.5, 2.0)
    rotation = uniform(0.0, 2.0 * math.pi)
    kappa_tilde = uniform(*spec["kappa_range"])
    deformation = uniform(*spec["deformation_range"])
    geometry_phases = torch.tensor(
        [uniform(0.0, 2.0 * math.pi) for _ in spec["geometry_modes"]],
        dtype=torch.float64,
    )
    solution_phases = torch.tensor(
        [uniform(0.0, 2.0 * math.pi) for _ in spec["solution_modes"]],
        dtype=torch.float64,
    )

    vertex_angles = (
        rotation
        + 2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    )
    vertex_radius = _radial_boundary(
        vertex_angles,
        deformation=deformation,
        modes=spec["geometry_modes"],
        phases=geometry_phases,
    )
    normalized_vertices = vertex_radius[:, None] * torch.stack(
        (vertex_angles.cos(), vertex_angles.sin()), dim=-1
    )
    points = center + reference_length * normalized_vertices
    index = torch.arange(n_boundary)
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    boundary = Mesh(
        points=points.to(device=device, dtype=dtype),
        cells=cells.to(device=device),
    )
    center_device = center.to(device=device, dtype=dtype)
    length_device = torch.tensor(reference_length, device=device, dtype=dtype)
    normalized_centroids = (boundary.cell_centroids - center_device) / length_device
    solution_phases_device = solution_phases.to(device=device, dtype=dtype)
    boundary_values = regular_screened_field(
        normalized_centroids,
        kappa_tilde=kappa_tilde,
        modes=spec["solution_modes"],
        phases=solution_phases_device,
    )

    query_angles = (
        2.0 * math.pi * torch.rand(n_query, dtype=torch.float64, generator=generator)
    )
    query_fraction = 0.9 * torch.sqrt(
        torch.rand(n_query, dtype=torch.float64, generator=generator)
    )
    query_radius = query_fraction * _radial_boundary(
        query_angles,
        deformation=deformation,
        modes=spec["geometry_modes"],
        phases=geometry_phases,
    )
    normalized_queries = query_radius[:, None] * torch.stack(
        (query_angles.cos(), query_angles.sin()), dim=-1
    )
    target = regular_screened_field(
        normalized_queries.to(device=device, dtype=dtype),
        kappa_tilde=kappa_tilde,
        modes=spec["solution_modes"],
        phases=solution_phases_device,
    )
    query_points = center + reference_length * normalized_queries

    if torch.any(
        torch.sum(
            boundary.cell_normals * (boundary.cell_centroids - center_device), dim=-1
        )
        >= 0
    ):
        raise RuntimeError("deformed boundary is not stored with inward normals")
    return DeformedScreenedSample(
        boundary=boundary,
        query_points=query_points.to(device=device, dtype=dtype),
        boundary_values=boundary_values,
        target=target,
        center=center_device,
        reference_length=length_device,
        kappa_tilde=kappa_tilde,
        deformation=deformation,
        solution_modes=spec["solution_modes"],
        solution_phases=solution_phases_device,
    )


def _normalized_geometry(
    sample: DeformedScreenedSample,
) -> tuple[Mesh, torch.Tensor]:
    return (
        Mesh(
            points=(sample.boundary.points - sample.center) / sample.reference_length,
            cells=sample.boundary.cells,
        ),
        (sample.query_points - sample.center) / sample.reference_length,
    )


def residual_controlled_richardson(
    matrix: torch.Tensor,
    values: torch.Tensor,
    *,
    tolerance: float = TRACE_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[torch.Tensor, int, float, bool]:
    density = values.clone()
    for step in range(max_iterations + 1):
        residual = values - matrix @ density
        relative_residual = float(
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(values).clamp_min(torch.finfo(values.dtype).tiny)
        )
        if relative_residual <= tolerance:
            return density, step, relative_residual, True
        if step < max_iterations:
            density = density + residual
    return density, max_iterations, relative_residual, False


def _normalized_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(
        torch.linalg.vector_norm(first - second)
        / torch.linalg.vector_norm(target).clamp_min(torch.finfo(target.dtype).tiny)
    )


def evaluate_case(
    *,
    seed: int,
    split: str,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
) -> dict[str, Any]:
    sample = build_deformed_sample(
        seed,
        split=split,
        n_boundary=n_boundary,
        n_query=n_query,
        device=device,
    )
    boundary, queries = _normalized_geometry(sample)
    matrix = trace_matrix(
        boundary,
        layer="double",
        kappa=sample.kappa_tilde,
        quadrature_order=quadrature_order,
        zero_diagonal=False,
    )
    check_matrix = trace_matrix(
        boundary,
        layer="double",
        kappa=sample.kappa_tilde,
        quadrature_order=check_quadrature_order,
        zero_diagonal=False,
    )
    propagation = field_matrix(
        queries,
        boundary,
        layer="double",
        kappa=sample.kappa_tilde,
        quadrature_order=quadrature_order,
    )
    dense_density = torch.linalg.solve(matrix, sample.boundary_values)
    iterative_density, steps, final_residual, converged = (
        residual_controlled_richardson(matrix, sample.boundary_values)
    )
    dense_field = propagation @ dense_density
    iterative_field = propagation @ iterative_density
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return {
        "kappa_tilde": sample.kappa_tilde,
        "deformation": sample.deformation,
        "converged": converged,
        "iterations": steps,
        "stopping_trace_relative_l2": final_residual,
        "dense_field_relative_l2": relative_l2(dense_field, sample.target),
        "iterative_field_relative_l2": relative_l2(iterative_field, sample.target),
        "iterative_minus_dense_relative_target_l2": _normalized_difference(
            iterative_field, dense_field, sample.target
        ),
        "dense_trace_relative_l2": relative_l2(
            matrix @ dense_density, sample.boundary_values
        ),
        "condition_number": float(torch.linalg.cond(matrix)),
        "unit_richardson_spectral_radius": float(
            torch.linalg.eigvals(identity - matrix).abs().max()
        ),
        "quadrature_32_to_64_relative_frobenius": relative_l2(check_matrix, matrix),
    }


NUMERIC_METRICS = (
    "iterations",
    "stopping_trace_relative_l2",
    "dense_field_relative_l2",
    "iterative_field_relative_l2",
    "iterative_minus_dense_relative_target_l2",
    "dense_trace_relative_l2",
    "condition_number",
    "unit_richardson_spectral_radius",
    "quadrature_32_to_64_relative_frobenius",
)


def _summary(cases: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = torch.tensor([float(case[key]) for case in cases], dtype=torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "maximum": float(values.max()),
    }


def evaluate_split_bank(
    *,
    seed: int,
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
                seed=seed + 7_919 * case + 1_000_003 * split_index,
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
            "all_converged": all(case["converged"] for case in cases),
            "metrics": {key: _summary(cases, key) for key in NUMERIC_METRICS},
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
    seed: int,
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
        per_resolution: dict[int, list[dict[str, Any]]] = {
            resolution: [] for resolution in resolutions
        }
        for case in range(n_cases):
            case_seed = seed + 7_919 * case + 1_000_003 * split_index
            samples = [
                build_deformed_sample(
                    case_seed,
                    split=split,
                    n_boundary=resolution,
                    n_query=n_query,
                    device=device,
                )
                for resolution in resolutions
            ]
            if not all(
                torch.equal(samples[0].query_points, sample.query_points)
                and torch.equal(samples[0].target, sample.target)
                for sample in samples[1:]
            ):
                raise RuntimeError(
                    "resolution ladder did not preserve the continuum problem"
                )
            for resolution in resolutions:
                per_resolution[resolution].append(
                    evaluate_case(
                        seed=case_seed,
                        split=split,
                        n_boundary=resolution,
                        n_query=n_query,
                        device=device,
                        quadrature_order=quadrature_order,
                        check_quadrature_order=check_quadrature_order,
                    )
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
                "iterations_mean": _summary(per_resolution[resolution], "iterations")[
                    "mean"
                ],
                "iterations_maximum": _summary(
                    per_resolution[resolution], "iterations"
                )["maximum"],
                "all_converged": all(
                    case["converged"] for case in per_resolution[resolution]
                ),
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
            "all_cases_converged": evaluation[split]["all_converged"],
            "maximum_iterations_at_most_32": (
                metrics["iterations"]["maximum"] <= MAX_ITERATIONS
            ),
            "mean_dense_discrepancy_at_most_0_001": (
                metrics["iterative_minus_dense_relative_target_l2"]["mean"] <= 0.001
            ),
            "max_dense_discrepancy_at_most_0_005": (
                metrics["iterative_minus_dense_relative_target_l2"]["maximum"] <= 0.005
            ),
            "dense_mean_field_error_at_most_0_02": (
                metrics["dense_field_relative_l2"]["mean"] <= 0.02
            ),
            "iterative_resolution_monotone": resolution[split]["iterative_monotone"],
        }
    dense_trace_sanity = all(
        evaluation[split]["metrics"]["dense_trace_relative_l2"]["maximum"] <= 1.0e-10
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
    shared_processor_sufficient = (
        dense_trace_sanity
        and quadrature_resolved
        and all(all(checks.values()) for checks in split_checks.values())
    )
    dense_field_stable = all(
        evaluation[split]["metrics"]["dense_field_relative_l2"]["mean"] <= 0.02
        for split in SPLIT_ORDER
    )
    if not dense_trace_sanity or not quadrature_resolved:
        verdict = "numerically_unresolved"
    elif shared_processor_sufficient:
        verdict = "shared_residual_processor_sufficient"
    elif dense_field_stable:
        verdict = "conditioned_preconditioner_earned"
    else:
        verdict = "boundary_representation_bottleneck"
    return {
        "verdict": verdict,
        "shared_processor_sufficient": shared_processor_sufficient,
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
        seed=EVALUATION_SEED,
        n_cases=evaluation_cases,
        n_boundary=evaluation_boundary_points,
        n_query=evaluation_query_points,
        device=device,
        quadrature_order=quadrature_order,
        check_quadrature_order=check_quadrature_order,
    )
    resolution = evaluate_resolution_bank(
        seed=RESOLUTION_SEED,
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
            "splits": SPLITS,
            "evaluation_cases_per_split": evaluation_cases,
            "evaluation_boundary_points": evaluation_boundary_points,
            "evaluation_query_points": evaluation_query_points,
            "resolution_cases_per_split": resolution_cases,
            "resolutions": list(resolutions),
            "quadrature_order": quadrature_order,
            "check_quadrature_order": check_quadrature_order,
            "trace_tolerance": TRACE_TOLERANCE,
            "max_iterations": MAX_ITERATIONS,
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
