# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Separate screened-layer trace errors from density-iteration instability."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import numpy as np
import torch
from laplace_readout_factorial import atomic_write_json
from provenance import runtime_environment, source_provenance

from physicsnemo.mesh import Mesh

STUDY = "screened_trace_mechanism_v1"
KAPPAS = (0.3, 0.1, 0.05, 0.02, 0.01, 0.001)
DENSITY_NAMES = ("charged", "zero_charge")
LAYER_NAMES = ("single", "double")
METHOD_NAMES = (
    "dense_full",
    "dense_zero_diagonal",
    "richardson_full",
    "richardson_zero_diagonal",
)
N_PANELS = 64
QUADRATURE_ORDER = 256
CHECK_QUADRATURE_ORDER = 128
RICHARDSON_STEPS = 8
SINGLE_COEFFICIENT = 1.0 / (2.0 * math.pi)
DOUBLE_COEFFICIENT = 1.0 / (2.0 * math.pi)


@lru_cache(maxsize=None)
def _gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2 or order % 2:
        raise ValueError("quadrature order must be an even integer at least two")
    return np.polynomial.legendre.leggauss(order)


def unit_polygon(
    n_panels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Mesh:
    """Return the stored counter-clockwise polygon with inward mesh normals."""

    if n_panels < 8:
        raise ValueError("n_panels must be at least eight")
    angles = (
        2.0 * math.pi * torch.arange(n_panels, device=device, dtype=dtype) / n_panels
    )
    points = torch.stack((angles.cos(), angles.sin()), dim=-1)
    index = torch.arange(n_panels, device=device)
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    polygon = Mesh(points=points, cells=cells)
    if torch.any(torch.sum(polygon.cell_normals * polygon.cell_centroids, dim=-1) >= 0):
        raise RuntimeError("the registered screened control requires inward normals")
    return polygon


def panel_midpoint_angles(boundary: Mesh) -> torch.Tensor:
    return torch.atan2(boundary.cell_centroids[:, 1], boundary.cell_centroids[:, 0])


def manufactured_densities(boundary: Mesh) -> dict[str, torch.Tensor]:
    angles = panel_midpoint_angles(boundary)
    return {
        "charged": 1.0 + 0.25 * angles.cos(),
        "zero_charge": angles.cos(),
    }


def interior_queries(
    *,
    device: torch.device,
    dtype: torch.dtype,
    angles_per_ring: int = 128,
) -> torch.Tensor:
    radii = torch.tensor((0.25, 0.6, 0.9), device=device, dtype=dtype)
    angles = (
        2.0
        * math.pi
        * torch.arange(angles_per_ring, device=device, dtype=dtype)
        / angles_per_ring
    )
    return torch.cat(
        [
            radius * torch.stack((angles.cos(), angles.sin()), dim=-1)
            for radius in radii
        ],
        dim=0,
    )


def yukawa_panel_influence(
    query_points: torch.Tensor,
    boundary: Mesh,
    *,
    kappa: float,
    single_coefficient: float,
    double_coefficient: float,
    quadrature_order: int,
) -> torch.Tensor:
    """Integrate the canonical screened single/double kernels on each panel."""

    nodes_np, weights_np = _gauss_rule(quadrature_order)
    nodes = torch.as_tensor(
        nodes_np, device=query_points.device, dtype=query_points.dtype
    )
    weights = torch.as_tensor(
        weights_np, device=query_points.device, dtype=query_points.dtype
    )
    vertices = boundary.points[boundary.cells]
    panel_start, panel_end = vertices[:, 0], vertices[:, 1]
    midpoint = 0.5 * (panel_start + panel_end)
    half_edge = 0.5 * (panel_end - panel_start)
    lengths = 2.0 * half_edge.norm(dim=-1)
    points = midpoint[:, None, :] + nodes[None, :, None] * half_edge[:, None, :]
    displacement = query_points[:, None, None, :] - points[None, :, :, :]
    radius = displacement.norm(dim=-1)
    scaled = float(kappa) * radius
    k0 = torch.special.modified_bessel_k0(scaled)
    k1 = torch.special.modified_bessel_k1(scaled)
    normal_dot = torch.einsum("qsgd,sd->qsg", displacement, boundary.cell_normals)
    kernel = single_coefficient * k0 + double_coefficient * (
        float(kappa) * k1 * normal_dot / radius
    )
    return (kernel * weights[None, None, :]).sum(dim=-1) * (0.5 * lengths)[None, :]


def trace_matrix(
    boundary: Mesh,
    *,
    layer: str,
    kappa: float,
    quadrature_order: int,
    zero_diagonal: bool,
) -> torch.Tensor:
    """Return the interior trace matrix in the implemented inward-normal frame."""

    if layer not in LAYER_NAMES:
        raise ValueError(f"layer must be one of {LAYER_NAMES}")
    influence = yukawa_panel_influence(
        boundary.cell_centroids,
        boundary,
        kappa=kappa,
        single_coefficient=(SINGLE_COEFFICIENT if layer == "single" else 0.0),
        double_coefficient=(DOUBLE_COEFFICIENT if layer == "double" else 0.0),
        quadrature_order=quadrature_order,
    ).clone()
    if layer == "double":
        influence.diagonal().zero_()
    if zero_diagonal:
        influence.diagonal().zero_()
    if layer == "double":
        influence = influence + 0.5 * torch.eye(
            boundary.n_cells,
            device=boundary.points.device,
            dtype=boundary.points.dtype,
        )
    return influence


def field_matrix(
    query_points: torch.Tensor,
    boundary: Mesh,
    *,
    layer: str,
    kappa: float,
    quadrature_order: int,
) -> torch.Tensor:
    if layer not in LAYER_NAMES:
        raise ValueError(f"layer must be one of {LAYER_NAMES}")
    return yukawa_panel_influence(
        query_points,
        boundary,
        kappa=kappa,
        single_coefficient=(SINGLE_COEFFICIENT if layer == "single" else 0.0),
        double_coefficient=(DOUBLE_COEFFICIENT if layer == "double" else 0.0),
        quadrature_order=quadrature_order,
    )


def richardson(
    matrix: torch.Tensor,
    values: torch.Tensor,
    *,
    steps: int = RICHARDSON_STEPS,
) -> torch.Tensor:
    density = values.clone()
    for _ in range(steps):
        density = density + values - matrix @ density
    return density


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float(torch.linalg.vector_norm(actual - expected) / denominator)


def _method_metrics(
    density: torch.Tensor,
    *,
    true_density: torch.Tensor,
    full_trace: torch.Tensor,
    boundary_values: torch.Tensor,
    field: torch.Tensor,
    true_field: torch.Tensor,
) -> dict[str, float]:
    return {
        "density_relative_l2": relative_l2(density, true_density),
        "true_trace_relative_l2": relative_l2(full_trace @ density, boundary_values),
        "field_relative_l2": relative_l2(field @ density, true_field),
    }


def _operator_metrics(matrix: torch.Tensor) -> dict[str, float]:
    iteration = (
        torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype) - matrix
    )
    return {
        "condition_number": float(torch.linalg.cond(matrix)),
        "unit_richardson_spectral_radius": float(
            torch.linalg.eigvals(iteration).abs().max()
        ),
    }


def evaluate_cell(
    boundary: Mesh,
    queries: torch.Tensor,
    *,
    layer: str,
    density_name: str,
    true_density: torch.Tensor,
    kappa: float,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
) -> dict[str, Any]:
    full_trace = trace_matrix(
        boundary,
        layer=layer,
        kappa=kappa,
        quadrature_order=quadrature_order,
        zero_diagonal=False,
    )
    zero_trace = trace_matrix(
        boundary,
        layer=layer,
        kappa=kappa,
        quadrature_order=quadrature_order,
        zero_diagonal=True,
    )
    check_trace = trace_matrix(
        boundary,
        layer=layer,
        kappa=kappa,
        quadrature_order=check_quadrature_order,
        zero_diagonal=False,
    )
    field = field_matrix(
        queries,
        boundary,
        layer=layer,
        kappa=kappa,
        quadrature_order=quadrature_order,
    )
    boundary_values = full_trace @ true_density
    true_field = field @ true_density
    densities = {
        "dense_full": torch.linalg.solve(full_trace, boundary_values),
        "dense_zero_diagonal": torch.linalg.solve(zero_trace, boundary_values),
        "richardson_full": richardson(full_trace, boundary_values),
        "richardson_zero_diagonal": richardson(zero_trace, boundary_values),
    }
    return {
        "layer": layer,
        "density": density_name,
        "kappa_tilde": kappa,
        "full_operator": _operator_metrics(full_trace),
        "zero_diagonal_operator": _operator_metrics(zero_trace),
        "quadrature_128_to_256_relative_frobenius": relative_l2(
            check_trace, full_trace
        ),
        "omitted_diagonal_relative_frobenius": relative_l2(zero_trace, full_trace),
        "methods": {
            name: _method_metrics(
                density,
                true_density=true_density,
                full_trace=full_trace,
                boundary_values=boundary_values,
                field=field,
                true_field=true_field,
            )
            for name, density in densities.items()
        },
    }


def apply_registered_decisions(cells: list[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (cell["layer"], cell["density"], float(cell["kappa_tilde"])): cell
        for cell in cells
    }

    def cell(layer: str, density: str, kappa: float) -> dict[str, Any]:
        return index[(layer, density, kappa)]

    dense_sanity_max = max(
        entry["methods"]["dense_full"]["density_relative_l2"] for entry in cells
    )
    charged_03 = cell("single", "charged", 0.3)
    charged_001 = cell("single", "charged", 0.01)
    charge_growth = (
        charged_001["methods"]["richardson_full"]["density_relative_l2"]
        / charged_03["methods"]["richardson_full"]["density_relative_l2"]
    )
    charge_instability = (
        dense_sanity_max <= 1.0e-10
        and any(
            cell("single", "charged", kappa)["full_operator"][
                "unit_richardson_spectral_radius"
            ]
            > 1.0
            for kappa in KAPPAS
            if kappa <= 0.05
        )
        and charge_growth >= 100.0
    )

    zero_005 = cell("single", "zero_charge", 0.05)
    zero_03 = cell("single", "zero_charge", 0.3)
    zero_0001 = cell("single", "zero_charge", 0.001)

    def self_error(entry: dict[str, Any]) -> float:
        metrics = entry["methods"]["dense_zero_diagonal"]
        return max(metrics["density_relative_l2"], metrics["field_relative_l2"])

    self_growth = self_error(zero_0001) / self_error(zero_03)
    self_omission = (
        dense_sanity_max <= 1.0e-10
        and self_error(zero_005) >= 0.05
        and self_growth >= 1.25
    )

    double_cells = [entry for entry in cells if entry["layer"] == "double"]
    double_stable = (
        dense_sanity_max <= 1.0e-10
        and all(
            entry["full_operator"]["unit_richardson_spectral_radius"] < 0.75
            for entry in double_cells
        )
        and all(
            entry["methods"]["richardson_full"]["density_relative_l2"] < 0.01
            for entry in double_cells
        )
    )
    return {
        "dense_manufacture_sanity": {
            "passed": dense_sanity_max <= 1.0e-10,
            "maximum_density_relative_l2": dense_sanity_max,
            "ceiling": 1.0e-10,
        },
        "charge_mode_iteration_instability": {
            "passed": charge_instability,
            "richardson_error_growth_0_3_to_0_01": charge_growth,
            "growth_floor": 100.0,
        },
        "self_panel_omission": {
            "passed": self_omission,
            "zero_charge_error_at_0_05": self_error(zero_005),
            "zero_charge_error_growth_0_3_to_0_001": self_growth,
            "error_floor": 0.05,
            "growth_floor": 1.25,
        },
        "canonical_double_layer_stability": {
            "passed": double_stable,
            "spectral_radius_ceiling": 0.75,
            "richardson_density_error_ceiling": 0.01,
        },
    }


def run_study(
    *,
    device: torch.device,
    n_panels: int = N_PANELS,
    kappas: tuple[float, ...] = KAPPAS,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
) -> dict[str, Any]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float64
    boundary = unit_polygon(n_panels, device=device, dtype=dtype)
    queries = interior_queries(device=device, dtype=dtype)
    densities = manufactured_densities(boundary)
    cells = [
        evaluate_cell(
            boundary,
            queries,
            layer=layer,
            density_name=density_name,
            true_density=densities[density_name],
            kappa=kappa,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        )
        for layer in LAYER_NAMES
        for density_name in DENSITY_NAMES
        for kappa in kappas
    ]
    report: dict[str, Any] = {
        "study": STUDY,
        "protocol": {
            "n_panels": n_panels,
            "kappas": list(kappas),
            "densities": list(DENSITY_NAMES),
            "layers": list(LAYER_NAMES),
            "methods": list(METHOD_NAMES),
            "quadrature_order": quadrature_order,
            "check_quadrature_order": check_quadrature_order,
            "richardson_steps": RICHARDSON_STEPS,
            "single_coefficient": SINGLE_COEFFICIENT,
            "double_coefficient": DOUBLE_COEFFICIENT,
            "normal_orientation": "inward",
            "dtype": "float64",
        },
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "cells": cells,
    }
    if (
        n_panels == N_PANELS
        and kappas == KAPPAS
        and quadrature_order == QUADRATURE_ORDER
        and check_quadrature_order == CHECK_QUADRATURE_ORDER
    ):
        report["decisions"] = apply_registered_decisions(cells)
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
                "decisions": report["decisions"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
