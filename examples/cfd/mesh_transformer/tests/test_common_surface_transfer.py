# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for the Phase-1 reference-surface operators."""

import numpy as np
import pytest
import torch
from common_surface_transfer import (
    ReferenceSurfaceMap,
    build_reference_surface_map,
    build_voronoi_reconstruction,
)
from phase1_fixed_carrier_convergence import _mesh_sha256, _representation

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, compose_measure_weights
from physicsnemo.mesh.primitives.surfaces import plane


def _weighted_inner(
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    expanded = weights.reshape(weights.shape + (1,) * (left.ndim - 1))
    return (expanded * left * right).sum()


def _weighted_squared_norm(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = weights.reshape(weights.shape + (1,) * (values.ndim - 1))
    return (expanded * values.square()).sum()


def test_reference_surface_map_exact_algebra_for_trailing_dimensions():
    assignment = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    reference_measures = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    transfer = ReferenceSurfaceMap.from_assignment(
        assignment,
        reference_measures,
        n_representation_cells=3,
    )

    generator = torch.Generator().manual_seed(17)
    reference = torch.randn(5, 2, 3, dtype=torch.float64, generator=generator)
    representation = torch.randn(3, 2, 3, dtype=torch.float64, generator=generator)

    restricted = transfer.restrict_reference(reference)
    prolonged = transfer.prolong_to_reference(representation)
    torch.testing.assert_close(
        transfer.restrict_reference(prolonged),
        representation,
        atol=1.0e-15,
        rtol=1.0e-15,
    )

    lhs = _weighted_inner(
        prolonged,
        reference,
        transfer.reference_measures,
    )
    rhs = _weighted_inner(
        representation,
        restricted,
        transfer.representation_measures,
    )
    torch.testing.assert_close(lhs, rhs, atol=1.0e-13, rtol=1.0e-13)

    constant_reference = torch.ones(5, dtype=torch.float64)
    constant_representation = torch.ones(3, dtype=torch.float64)
    torch.testing.assert_close(
        transfer.restrict_reference(constant_reference),
        constant_representation,
    )
    torch.testing.assert_close(
        transfer.prolong_to_reference(constant_representation),
        constant_reference,
    )

    reference_integral = (transfer.reference_measures[:, None, None] * reference).sum(
        dim=0
    )
    representation_integral = (
        transfer.representation_measures[:, None, None] * restricted
    ).sum(dim=0)
    torch.testing.assert_close(
        reference_integral,
        representation_integral,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_reference_projection_has_exact_pythagorean_error_decomposition():
    transfer = ReferenceSurfaceMap.from_assignment(
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        torch.tensor([1.0, 0.5, 2.0, 1.0, 1.5, 3.0], dtype=torch.float64),
        n_representation_cells=3,
    )
    generator = torch.Generator().manual_seed(137)
    truth = torch.randn(6, 4, dtype=torch.float64, generator=generator)
    prediction = torch.randn(3, 4, dtype=torch.float64, generator=generator)
    restricted_truth = transfer.restrict_reference(truth)

    total = _weighted_squared_norm(
        transfer.prolong_to_reference(prediction) - truth,
        transfer.reference_measures,
    )
    represented = _weighted_squared_norm(
        prediction - restricted_truth,
        transfer.representation_measures,
    )
    floor = _weighted_squared_norm(
        transfer.project_reference(truth) - truth,
        transfer.reference_measures,
    )
    torch.testing.assert_close(
        total,
        represented + floor,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_reference_surface_map_rejects_empty_representation_cells():
    with pytest.raises(ValueError, match="receive no positive reference measure"):
        ReferenceSurfaceMap.from_assignment(
            torch.tensor([0, 0, 2], dtype=torch.long),
            torch.ones(3, dtype=torch.float64),
            n_representation_cells=4,
        )


def test_frozen_representation_npz_preserves_exact_mesh(tmp_path):
    reference = _disjoint_triangles([(0.0, 0.0, True)])
    expected = _disjoint_triangles(
        [
            (0.0, 0.0, True),
            (0.2, 0.0, True),
        ]
    )
    frozen_path = tmp_path / "representation.npz"
    np.savez(
        frozen_path,
        points=expected.points.numpy(),
        cells=expected.cells.numpy(),
    )

    actual, source = _representation(
        reference,
        frozen_path=frozen_path,
        export_path=None,
    )

    assert _mesh_sha256(actual) == _mesh_sha256(expected)
    assert source["kind"] == "frozen_npz"
    assert source["sha256"]


def test_full_cover_plane_map_uses_effective_reference_measure():
    reference_raw = plane.load(subdivisions=8)
    representation_raw = plane.load(subdivisions=2)
    reference = Mesh(
        points=reference_raw.points.double(),
        cells=reference_raw.cells,
    )
    representation = Mesh(
        points=representation_raw.points.double(),
        cells=representation_raw.cells,
    )
    factors = torch.linspace(
        0.5,
        1.5,
        reference.n_cells,
        dtype=reference.points.dtype,
    )
    compose_measure_weights(reference, factors)

    transfer, diagnostics = build_reference_surface_map(
        reference,
        representation,
        max_distance=1.0e-6,
        min_normal_alignment=0.999,
    )

    torch.testing.assert_close(
        transfer.representation_measures.sum(),
        cell_measures(reference).sum(),
    )
    assert not torch.equal(cell_measures(reference), reference.cell_areas)
    assert float(diagnostics.reference_distance.max()) < 1.0e-6
    assert float(diagnostics.reference_normal_alignment.min()) > 0.999


def test_full_cover_geometry_gate_rejects_holes_and_reversed_orientation():
    reference_raw = plane.load(subdivisions=8)
    coarse_raw = plane.load(subdivisions=2)
    reference = Mesh(
        points=reference_raw.points.double(),
        cells=reference_raw.cells,
    )
    coarse = Mesh(points=coarse_raw.points.double(), cells=coarse_raw.cells)

    keep = coarse.cell_centroids[:, 0] < coarse.cell_centroids[:, 0].median()
    incomplete = coarse.slice_cells(torch.nonzero(keep, as_tuple=False).flatten())
    with pytest.raises(ValueError, match="coverage gate failed"):
        build_reference_surface_map(
            reference,
            incomplete,
            max_distance=0.05,
            min_normal_alignment=0.9,
        )

    reversed_mesh = Mesh(
        points=coarse.points,
        cells=coarse.cells.flip(-1),
    )
    with pytest.raises(ValueError, match="orientation gate failed"):
        build_reference_surface_map(
            reference,
            reversed_mesh,
            max_distance=1.0e-6,
            min_normal_alignment=0.9,
        )


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


def test_normal_aware_voronoi_reconstruction_removes_thin_sheet_crossing():
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

    ambient, ambient_diagnostics = build_voronoi_reconstruction(
        reference,
        representation,
    )
    normal_aware, normal_diagnostics = build_voronoi_reconstruction(
        reference,
        representation,
        normal_weight=0.2,
    )

    assert float(
        (ambient_diagnostics.reference_normal_alignment < 0).double().mean()
    ) == pytest.approx(1.0 / 3.0)
    assert (
        float((normal_diagnostics.reference_normal_alignment < 0).double().mean())
        == 0.0
    )

    for transfer in (ambient, normal_aware):
        probe = torch.tensor([1.2, -0.7], dtype=torch.float64)
        torch.testing.assert_close(
            transfer.restrict_reference(transfer.prolong_to_reference(probe)),
            probe,
            atol=1.0e-15,
            rtol=1.0e-15,
        )
