# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mathematical contracts for the analytic conformal Laplace generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    ConformalGeometry,
    SimilarityTransform,
    apply_similarity,
    boundary_outward_normals,
    build_domain_sample,
    conformal_derivative,
    conformal_map,
    evaluate_potential,
    identity_similarity,
    physical_area_jacobian,
    sample_disk_preimages,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
)

from physicsnemo.mesh import DomainMesh  # noqa: E402


@pytest.fixture
def geometry() -> ConformalGeometry:
    """Provide one deterministic, nontrivially deformed test geometry."""

    return sample_geometry(
        1203,
        modes=(2, 3, 5),
        deformation_range=(0.42, 0.42),
        dtype=torch.float64,
    )


def test_sampled_geometry_has_certified_non_degeneracy(
    geometry: ConformalGeometry,
) -> None:
    """The coefficient constraint must imply the stated derivative bounds."""
    torch.testing.assert_close(
        geometry.deformation_bound,
        torch.tensor(0.42, dtype=torch.float64),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    preimages = sample_disk_preimages(81, 4096, dtype=torch.float64)
    derivative = conformal_derivative(geometry, preimages)
    bound = geometry.deformation_bound
    assert torch.all(derivative.abs() >= 1.0 - bound - 2.0e-15)
    assert torch.all(derivative.abs() <= 1.0 + bound + 2.0e-15)
    assert torch.all(derivative.real >= 1.0 - bound - 2.0e-15)

    # The integrated derivative estimate gives the same bi-Lipschitz bound for
    # arbitrary pairs, independently checking that the map cannot self-cross.
    first = sample_disk_preimages(83, 2048, dtype=torch.float64)
    second = sample_disk_preimages(89, 2048, dtype=torch.float64)
    input_distance = (first - second).abs()
    output_distance = (
        conformal_map(geometry, first) - conformal_map(geometry, second)
    ).abs()
    ratio = output_distance / input_distance
    assert torch.all(ratio >= 1.0 - bound - 2.0e-14)
    assert torch.all(ratio <= 1.0 + bound + 2.0e-14)


def test_invalid_coefficient_bound_is_rejected() -> None:
    """Reject conformal coefficients that cannot certify injectivity."""

    with pytest.raises(ValueError, match="strictly less than one"):
        ConformalGeometry(
            modes=(2,),
            coefficients=torch.tensor([0.5 + 0.0j], dtype=torch.complex128),
        )


@pytest.mark.parametrize("reflection", [False, True])
def test_boundary_cells_produce_outward_normals(
    geometry: ConformalGeometry, reflection: bool
) -> None:
    """Cell winding must follow the physical orientation, including parity."""
    drive = sample_drive(19, dtype=torch.float64)
    similarity = sample_similarity(
        23,
        scale_range=(1.7, 1.7),
        translation_extent=3.0,
        reflection=reflection,
        dtype=torch.float64,
    )
    sample = build_domain_sample(
        geometry,
        drive,
        n_boundary=512,
        n_query=8,
        query_seed=29,
        similarity=similarity,
    )
    boundary = sample.domain.boundaries["dirichlet"]
    midpoint_angles = torch.angle(sample.boundary_midpoint_preimages)
    exact = boundary_outward_normals(geometry, midpoint_angles, similarity)
    agreement = torch.sum(boundary.cell_normals * exact, dim=-1)
    assert agreement.min().item() > 0.99999


def test_boundary_values_are_exact_trace_samples(geometry: ConformalGeometry) -> None:
    """Store the analytic Dirichlet trace exactly at panel midpoints."""

    drive = sample_drive(
        311,
        modes=(1, 2, 4, 7),
        regularity=1.5,
        boundary_rms=2.3,
        dtype=torch.float64,
    )
    sample = build_domain_sample(
        geometry,
        drive,
        n_boundary=97,
        n_query=16,
        query_seed=313,
    )
    expected = evaluate_potential(drive, sample.boundary_midpoint_preimages)
    actual = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        drive.boundary_rms,
        torch.tensor(2.3, dtype=torch.float64),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_targets_are_harmonic_by_autodiff() -> None:
    """The real part of the sampled holomorphic polynomial has zero Laplacian."""
    drive = sample_drive(
        401,
        modes=tuple(range(1, 13)),
        regularity=0.75,
        dtype=torch.float64,
    )
    coordinates = torch.tensor(
        [[0.13, -0.27], [-0.51, 0.22], [0.04, 0.71], [0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    preimages = torch.complex(coordinates[:, 0], coordinates[:, 1])
    potential = evaluate_potential(drive, preimages)
    gradient = torch.autograd.grad(potential.sum(), coordinates, create_graph=True)[0]
    laplacian = torch.zeros(coordinates.shape[0], dtype=torch.float64)
    for axis in range(2):
        second = torch.autograd.grad(
            gradient[:, axis].sum(),
            coordinates,
            create_graph=True,
            retain_graph=True,
        )[0][:, axis]
        laplacian = laplacian + second
    torch.testing.assert_close(
        laplacian,
        torch.zeros_like(laplacian),
        rtol=0.0,
        atol=2.0e-13,
    )


def test_sampling_is_deterministic_and_seeds_are_factorized() -> None:
    """Keep geometry, drive, and query randomness deterministic and separate."""

    first_geometry = sample_geometry(501, dtype=torch.float32)
    second_geometry = sample_geometry(501, dtype=torch.float32)
    other_geometry = sample_geometry(502, dtype=torch.float32)
    torch.testing.assert_close(
        first_geometry.coefficients, second_geometry.coefficients, rtol=0.0, atol=0.0
    )
    assert not torch.equal(first_geometry.coefficients, other_geometry.coefficients)

    first_drive = sample_drive(601, dtype=torch.float32)
    second_drive = sample_drive(601, dtype=torch.float32)
    torch.testing.assert_close(first_drive.constant, second_drive.constant)
    torch.testing.assert_close(first_drive.coefficients, second_drive.coefficients)

    first = build_domain_sample(
        first_geometry,
        first_drive,
        n_boundary=48,
        n_query=64,
        query_seed=701,
    )
    second = build_domain_sample(
        second_geometry,
        second_drive,
        n_boundary=48,
        n_query=64,
        query_seed=701,
    )
    torch.testing.assert_close(
        first.domain.interior.points, second.domain.interior.points
    )
    torch.testing.assert_close(first.target, second.target)
    torch.testing.assert_close(first.area_jacobian, second.area_jacobian)
    torch.testing.assert_close(
        first.domain.boundaries["dirichlet"].points,
        second.domain.boundaries["dirichlet"].points,
    )

    changed_queries = build_domain_sample(
        second_geometry,
        second_drive,
        n_boundary=48,
        n_query=64,
        query_seed=702,
    )
    # Changing only the query seed must not change geometry or boundary data.
    torch.testing.assert_close(
        first.domain.boundaries["dirichlet"].points,
        changed_queries.domain.boundaries["dirichlet"].points,
    )
    torch.testing.assert_close(
        first.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        changed_queries.domain.boundaries["dirichlet"].cell_data["boundary_value"],
    )
    assert not torch.equal(
        first.domain.interior.points, changed_queries.domain.interior.points
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_similarity_covariance_and_physical_jacobian(
    geometry: ConformalGeometry, reflection: bool
) -> None:
    """Transform geometry covariantly and area weights by squared scale."""

    drive = sample_drive(811, dtype=torch.float64)
    base = build_domain_sample(
        geometry,
        drive,
        n_boundary=128,
        n_query=96,
        query_seed=813,
    )
    similarity = sample_similarity(
        821,
        scale_range=(3.25, 3.25),
        translation_extent=4.0,
        reflection=reflection,
        dtype=torch.float64,
    )
    transformed = transform_sample(base, similarity)

    torch.testing.assert_close(
        transformed.domain.interior.points,
        apply_similarity(base.domain.interior.points, similarity),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        transformed.domain.boundaries["dirichlet"].points,
        apply_similarity(base.domain.boundaries["dirichlet"].points, similarity),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(transformed.target, base.target, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        transformed.domain.interior.point_data["preimage_radius"],
        base.domain.interior.point_data["preimage_radius"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        transformed.area_jacobian,
        similarity.scale.square() * base.area_jacobian,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        transformed.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        base.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        transformed.domain.global_data["reference_length"], similarity.scale
    )

    expected_normals = torch.einsum(
        "nd,ed->ne",
        base.domain.boundaries["dirichlet"].cell_normals,
        similarity.rotation,
    )
    torch.testing.assert_close(
        transformed.domain.boundaries["dirichlet"].cell_normals,
        expected_normals,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_domain_contract_and_exact_metadata(geometry: ConformalGeometry) -> None:
    """Expose the declared mesh fields and exact analytic metadata."""

    drive = sample_drive(901, dtype=torch.float64)
    query_preimages = torch.tensor(
        [0.0 + 0.0j, 0.25 - 0.1j, -0.4 + 0.5j], dtype=torch.complex128
    )
    similarity = SimilarityTransform(
        scale=torch.tensor(2.0, dtype=torch.float64),
        rotation=torch.eye(2, dtype=torch.float64),
        translation=torch.tensor([1.2, -0.7], dtype=torch.float64),
    )
    sample = build_domain_sample(
        geometry,
        drive,
        n_boundary=32,
        query_preimages=query_preimages,
        similarity=similarity,
    )

    assert isinstance(sample.domain, DomainMesh)
    assert set(sample.domain.boundaries.keys()) == {"dirichlet"}
    assert sample.domain.boundaries["dirichlet"].n_cells == 32
    assert set(sample.domain.interior.point_data.keys()) == {
        "potential",
        "area_jacobian",
        "preimage_radius",
    }
    torch.testing.assert_close(
        sample.target, evaluate_potential(drive, query_preimages)
    )
    torch.testing.assert_close(
        sample.area_jacobian,
        physical_area_jacobian(geometry, query_preimages, similarity),
    )
    torch.testing.assert_close(
        sample.domain.interior.point_data["preimage_radius"], query_preimages.abs()
    )


def test_identity_similarity_is_exact() -> None:
    """Leave coordinates bitwise unchanged under the identity similarity."""

    identity = identity_similarity(dtype=torch.float64)
    points = torch.tensor([[0.2, -0.4], [1.1, 3.2]], dtype=torch.float64)
    torch.testing.assert_close(
        apply_similarity(points, identity), points, rtol=0.0, atol=0.0
    )
