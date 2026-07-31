# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 falsifiers for a common reference-surface evaluation operator.

The experiment separates three claims that are easy to conflate:

1. the restriction/prolongation algebra is exactly mass-adjoint;
2. a full-cover remesh can pass explicit geometric coverage/orientation gates;
3. a sparse panel subset cannot pass those gates and needs a separately
   labeled reconstruction prior.

No model is trained.  The dated JSON artifact is the deciding record for the
pre-registration in ``book/18-notebook.qmd``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from common_surface_transfer import (  # noqa: E402
    ReferenceSurfaceMap,
    build_reference_surface_map,
    build_voronoi_reconstruction,
)
from provenance import runtime_environment, source_provenance  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import cell_measures  # noqa: E402
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral  # noqa: E402
from physicsnemo.mesh.remeshing import remesh  # noqa: E402


def _sphere(subdivisions: int) -> Mesh:
    raw = sphere_icosahedral.load(radius=1.0, subdivisions=subdivisions)
    return Mesh(points=raw.points.double(), cells=raw.cells)


def _scalar_field(points: torch.Tensor) -> torch.Tensor:
    return (
        1.0
        + 0.35 * points[:, 0]
        - 0.22 * points[:, 1]
        + 0.17 * points[:, 2].square()
        + 0.09 * points[:, 0] * points[:, 2]
    )


def _tangent_field(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    raw = torch.stack(
        (
            0.4 + points[:, 1] - 0.2 * points[:, 2],
            -0.3 + 0.5 * points[:, 2] + 0.1 * points[:, 0],
            0.2 - 0.7 * points[:, 0] + 0.3 * points[:, 1],
        ),
        dim=-1,
    )
    return raw - (raw * normals).sum(dim=-1, keepdim=True) * normals


def _expanded(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return weights.reshape(weights.shape + (1,) * (values.ndim - 1))


def _squared_norm(values: torch.Tensor, measures: torch.Tensor) -> torch.Tensor:
    return (_expanded(measures, values) * values.square()).sum()


def _relative_l2(
    actual: torch.Tensor,
    expected: torch.Tensor,
    measures: torch.Tensor,
) -> float:
    numerator = _squared_norm(actual - expected, measures)
    denominator = _squared_norm(expected, measures).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float(torch.sqrt(numerator / denominator))


def _relative_vector_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float((actual - expected).norm() / denominator)


def _scatter_first_moment(
    transfer: ReferenceSurfaceMap,
    centroids: torch.Tensor,
) -> torch.Tensor:
    weighted = transfer.reference_measures[:, None] * centroids
    result = weighted.new_zeros((transfer.n_representation_cells, 3))
    result.index_add_(0, transfer.reference_to_representation, weighted)
    return result


def _algebra_diagnostics(transfer: ReferenceSurfaceMap) -> dict[str, float]:
    generator = torch.Generator().manual_seed(20_260_727)
    reference = torch.randn(
        transfer.n_reference_cells,
        2,
        3,
        dtype=transfer.reference_measures.dtype,
        generator=generator,
    )
    representation = torch.randn(
        transfer.n_representation_cells,
        2,
        3,
        dtype=transfer.reference_measures.dtype,
        generator=generator,
    )
    restricted = transfer.restrict_reference(reference)
    prolonged = transfer.prolong_to_reference(representation)

    lhs = (
        _expanded(transfer.reference_measures, prolonged) * prolonged * reference
    ).sum()
    rhs = (
        _expanded(transfer.representation_measures, representation)
        * representation
        * restricted
    ).sum()
    adjoint_relative_error = float(
        (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1.0)
    )

    roundtrip = transfer.restrict_reference(prolonged)
    roundtrip_max_abs = float((roundtrip - representation).abs().max())

    restricted_constant = transfer.restrict_reference(
        torch.ones(
            transfer.n_reference_cells,
            dtype=transfer.reference_measures.dtype,
        )
    )
    prolonged_constant = transfer.prolong_to_reference(
        torch.ones(
            transfer.n_representation_cells,
            dtype=transfer.reference_measures.dtype,
        )
    )
    constant_max_abs = max(
        float((restricted_constant - 1).abs().max()),
        float((prolonged_constant - 1).abs().max()),
    )

    reference_integral = (
        _expanded(transfer.reference_measures, reference) * reference
    ).sum(dim=0)
    representation_integral = (
        _expanded(transfer.representation_measures, restricted) * restricted
    ).sum(dim=0)
    integral_relative_error = float(
        (representation_integral - reference_integral).norm()
        / reference_integral.norm().clamp_min(
            torch.finfo(reference_integral.dtype).tiny
        )
    )

    prediction = torch.randn(
        transfer.n_representation_cells,
        2,
        3,
        dtype=transfer.reference_measures.dtype,
        generator=generator,
    )
    total = _squared_norm(
        transfer.prolong_to_reference(prediction) - reference,
        transfer.reference_measures,
    )
    represented = _squared_norm(
        prediction - restricted,
        transfer.representation_measures,
    )
    floor = _squared_norm(
        transfer.project_reference(reference) - reference,
        transfer.reference_measures,
    )
    pythagorean_relative_error = float(
        (total - represented - floor).abs() / total.clamp_min(1.0)
    )
    return {
        "constant_max_abs_error": constant_max_abs,
        "integral_relative_error": integral_relative_error,
        "adjoint_relative_error": adjoint_relative_error,
        "representation_roundtrip_max_abs_error": roundtrip_max_abs,
        "pythagorean_relative_error": pythagorean_relative_error,
    }


def _field_diagnostics(
    reference: Mesh,
    representation: Mesh,
    transfer: ReferenceSurfaceMap,
) -> dict[str, Any]:
    measures = transfer.reference_measures
    centroids = reference.cell_centroids
    normals = reference.cell_normals
    pressure = _scalar_field(centroids)
    wss = _tangent_field(centroids, normals)

    pressure_projected = transfer.project_reference(pressure)
    wss_raw_projected = transfer.project_reference(wss)
    wss_projected = (
        wss_raw_projected
        - (wss_raw_projected * normals).sum(dim=-1, keepdim=True) * normals
    )

    traction = pressure[:, None] * normals + wss
    projected_traction = pressure_projected[:, None] * normals + wss_projected
    force = (measures[:, None] * traction).sum(dim=0)
    projected_force = (measures[:, None] * projected_traction).sum(dim=0)
    moment = (measures[:, None] * torch.cross(centroids, traction, dim=-1)).sum(dim=0)
    projected_moment = (
        measures[:, None] * torch.cross(centroids, projected_traction, dim=-1)
    ).sum(dim=0)

    represented_traction = transfer.restrict_reference(traction)
    raw_projected_traction = transfer.prolong_to_reference(represented_traction)
    raw_projected_moment = (
        measures[:, None] * torch.cross(centroids, raw_projected_traction, dim=-1)
    ).sum(dim=0)
    first_moments = _scatter_first_moment(transfer, centroids)
    common_refinement_moment = torch.cross(
        first_moments,
        represented_traction,
        dim=-1,
    ).sum(dim=0)
    native_centroid_moment = (
        transfer.representation_measures[:, None]
        * torch.cross(
            representation.cell_centroids,
            represented_traction,
            dim=-1,
        )
    ).sum(dim=0)

    return {
        "pressure_projection_floor_relative_l2": _relative_l2(
            pressure_projected,
            pressure,
            measures,
        ),
        "wss_raw_p0_projection_floor_relative_l2": _relative_l2(
            wss_raw_projected,
            wss,
            measures,
        ),
        "wss_tangent_projection_floor_relative_l2": _relative_l2(
            wss_projected,
            wss,
            measures,
        ),
        "truth_wss_tangency_max_abs": float((wss * normals).sum(dim=-1).abs().max()),
        "projected_wss_tangency_max_abs": float(
            (wss_projected * normals).sum(dim=-1).abs().max()
        ),
        "traction_force_projection_relative_error": _relative_vector_error(
            projected_force,
            force,
        ),
        "traction_moment_projection_relative_error": _relative_vector_error(
            projected_moment,
            moment,
        ),
        "common_refinement_first_moment_vs_direct_relative_error": (
            _relative_vector_error(common_refinement_moment, raw_projected_moment)
        ),
        "naive_representation_centroid_moment_relative_error": (
            _relative_vector_error(native_centroid_moment, raw_projected_moment)
        ),
    }


def _triangle(center_x: float, z: float, *, upward: bool) -> torch.Tensor:
    points = torch.tensor(
        [
            [center_x - 0.01, -0.01, z],
            [center_x + 0.01, -0.01, z],
            [center_x, 0.01, z],
        ],
        dtype=torch.float64,
    )
    return points if upward else points[[0, 2, 1]]


def _disjoint_triangles(specification: list[tuple[float, float, bool]]) -> Mesh:
    points = torch.cat(
        [_triangle(x, z, upward=upward) for x, z, upward in specification],
        dim=0,
    )
    cells = torch.arange(len(points), dtype=torch.long).reshape(-1, 3)
    return Mesh(points=points, cells=cells)


def _thin_sheet_diagnostic() -> dict[str, Any]:
    reference = _disjoint_triangles(
        [
            (0.0, 0.01, True),
            (0.2, 0.01, True),
            (0.0, 0.00, False),
        ]
    )
    representation = _disjoint_triangles(
        [
            (0.2, 0.01, True),
            (0.0, 0.00, False),
        ]
    )
    ambient, ambient_geometry = build_voronoi_reconstruction(
        reference,
        representation,
    )
    normal_aware, normal_geometry = build_voronoi_reconstruction(
        reference,
        representation,
        normal_weight=0.2,
    )
    return {
        "ambient_negative_normal_fraction": float(
            (ambient_geometry.reference_normal_alignment < 0).double().mean()
        ),
        "normal_aware_negative_normal_fraction": float(
            (normal_geometry.reference_normal_alignment < 0).double().mean()
        ),
        "ambient_map_sha256": ambient.sha256(),
        "normal_aware_map_sha256": normal_aware.sha256(),
        "ambient_algebra": _algebra_diagnostics(ambient),
        "normal_aware_algebra": _algebra_diagnostics(normal_aware),
    }


def _sparse_coverage_rejection(reference: Mesh, n_cells: int = 40) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        reference.n_cells,
        generator=generator,
    )[:n_cells]
    sparse = reference.slice_cells(indices)
    try:
        build_reference_surface_map(
            reference,
            sparse,
            max_distance=0.1,
            min_normal_alignment=0.8,
        )
    except ValueError as error:
        return {
            "rejected": True,
            "error": str(error),
            "n_sparse_cells": sparse.n_cells,
        }
    return {
        "rejected": False,
        "error": None,
        "n_sparse_cells": sparse.n_cells,
    }


def _evaluate_reference(
    reference: Mesh,
    representation: Mesh,
) -> tuple[ReferenceSurfaceMap, dict[str, Any]]:
    transfer, geometry = build_reference_surface_map(
        reference,
        representation,
        max_distance=0.1,
        min_normal_alignment=0.9,
    )
    return transfer, {
        "n_reference_cells": reference.n_cells,
        "n_representation_cells": representation.n_cells,
        "reference_area": float(cell_measures(reference).sum()),
        "representation_native_area": float(cell_measures(representation).sum()),
        "represented_area": float(transfer.representation_measures.sum()),
        "map_sha256": transfer.sha256(),
        "geometry": geometry.summary(),
        "algebra": _algebra_diagnostics(transfer),
        "fields": _field_diagnostics(reference, representation, transfer),
    }


def _relative_measure_l1(
    coarse: ReferenceSurfaceMap,
    fine: ReferenceSurfaceMap,
) -> float:
    return float(
        (fine.representation_measures - coarse.representation_measures).abs().sum()
        / fine.representation_measures.sum()
    )


def _normalized_measure_shape_l1(
    coarse: ReferenceSurfaceMap,
    fine: ReferenceSurfaceMap,
) -> float:
    coarse_shape = coarse.representation_measures / coarse.representation_measures.sum()
    fine_shape = fine.representation_measures / fine.representation_measures.sum()
    return float((fine_shape - coarse_shape).abs().sum())


def _relative_total_measure_change(
    coarse: ReferenceSurfaceMap,
    fine: ReferenceSurfaceMap,
) -> float:
    coarse_total = coarse.representation_measures.sum()
    fine_total = fine.representation_measures.sum()
    return float((fine_total - coarse_total).abs() / fine_total)


def _field_change(
    coarse: dict[str, Any],
    fine: dict[str, Any],
) -> dict[str, float]:
    return {
        key: float(fine["fields"][key] - coarse["fields"][key])
        for key in (
            "pressure_projection_floor_relative_l2",
            "wss_tangent_projection_floor_relative_l2",
            "traction_force_projection_relative_error",
            "traction_moment_projection_relative_error",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(EXAMPLE_ROOT / "results" / "phase1_common_surface_2026-07-27.json"),
    )
    args = parser.parse_args()

    reference_3 = _sphere(3)
    representation = remesh(reference_3, n_clusters=80)
    transfer_3, refinement_3 = _evaluate_reference(reference_3, representation)
    reference_4 = _sphere(4)
    transfer_4, refinement_4 = _evaluate_reference(reference_4, representation)
    reference_5 = _sphere(5)
    transfer_5, refinement_5 = _evaluate_reference(reference_5, representation)

    measure_discrepancy_3_to_4 = _relative_measure_l1(transfer_3, transfer_4)
    measure_discrepancy_4_to_5 = _relative_measure_l1(transfer_4, transfer_5)
    thin_sheet = _thin_sheet_diagnostic()
    sparse_rejection = _sparse_coverage_rejection(reference_3)

    algebra_values = tuple(refinement_5["algebra"].values())
    gates = {
        "h1_float64_algebra_below_1e-12": max(algebra_values) < 1.0e-12,
        "h2_ambient_thin_sheet_has_crossing": (
            thin_sheet["ambient_negative_normal_fraction"] > 0
        ),
        "h2_normal_aware_thin_sheet_has_no_crossing": (
            thin_sheet["normal_aware_negative_normal_fraction"] == 0
        ),
        "full_cover_symmetric_distance_below_0p1": max(
            refinement_5["geometry"]["reference_to_representation_distance"]["max"],
            refinement_5["geometry"]["representation_to_reference_distance"]["max"],
        )
        < 0.1,
        "full_cover_minimum_normal_dot_above_0p9": min(
            refinement_5["geometry"]["reference_to_representation_normal_dot"]["min"],
            refinement_5["geometry"]["representation_to_reference_normal_dot"]["min"],
        )
        > 0.9,
        "sparse_panel_set_rejected_by_coverage_gate": sparse_rejection["rejected"],
        "common_refinement_first_moment_below_1e-12": refinement_5["fields"][
            "common_refinement_first_moment_vs_direct_relative_error"
        ]
        < 1.0e-12,
        "projected_wss_tangent_below_1e-12": refinement_5["fields"][
            "projected_wss_tangency_max_abs"
        ]
        < 1.0e-12,
    }
    result = {
        "status": "phase1_synthetic_common_surface_falsifiers",
        "date": "2026-07-27",
        "scope": (
            "Mass-adjoint P0 algebra and discrete common-refinement geometry "
            "on an independently remeshed sphere; sparse Voronoi maps remain "
            "explicit reconstruction sensitivities, not physical overlap."
        ),
        "remesher": {
            "name": "pyacvd",
            "requested_vertex_clusters": 80,
            "output_faces": representation.n_cells,
        },
        "reference_subdivision_3": refinement_3,
        "reference_subdivision_4": refinement_4,
        "reference_subdivision_5": refinement_5,
        "exploratory_reference_refinement_convergence": {
            "interpretation": (
                "No threshold was preregistered for this diagnostic. It tests "
                "whether the discrete closest-face partition and reported "
                "projection floors stabilize as reference quadrature is refined."
            ),
            "represented_measure_relative_l1": {
                "subdivision_3_to_4": measure_discrepancy_3_to_4,
                "subdivision_4_to_5": measure_discrepancy_4_to_5,
            },
            "represented_measure_normalized_shape_l1": {
                "subdivision_3_to_4": _normalized_measure_shape_l1(
                    transfer_3,
                    transfer_4,
                ),
                "subdivision_4_to_5": _normalized_measure_shape_l1(
                    transfer_4,
                    transfer_5,
                ),
            },
            "reference_total_measure_relative_change": {
                "subdivision_3_to_4": _relative_total_measure_change(
                    transfer_3,
                    transfer_4,
                ),
                "subdivision_4_to_5": _relative_total_measure_change(
                    transfer_4,
                    transfer_5,
                ),
            },
            "field_metric_absolute_change": {
                "subdivision_3_to_4": _field_change(refinement_3, refinement_4),
                "subdivision_4_to_5": _field_change(refinement_4, refinement_5),
            },
        },
        "thin_sheet": thin_sheet,
        "sparse_full_cover_gate": sparse_rejection,
        "declared_algebra_geometry_gates": gates,
        "all_declared_algebra_geometry_gates_pass": all(gates.values()),
        "partition_convergence_scope": (
            "These gates do not test convergence of individual represented "
            "measures under reference refinement. See the separately gated "
            "fixed-carrier convergence artifact before treating this discrete "
            "map as a quantitatively converged polygon-overlay surrogate."
        ),
        "runtime": runtime_environment(torch.device("cpu")),
        "source_provenance": source_provenance(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
