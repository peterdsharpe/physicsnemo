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

"""Theory-level contracts for the benchmark-local STF multipole model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    ConformalGeometry,
    HarmonicDrive,
    build_domain_sample,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
    unit_circle,
)
from stf_multipole import (  # noqa: E402
    STFMultipolePotential,
    planar_stf_coordinates,
)

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _unit_disk_geometry() -> ConformalGeometry:
    return ConformalGeometry(
        modes=(), coefficients=torch.empty(0, dtype=torch.complex128)
    )


def _pure_mode_drive(mode: int, amplitude: complex = 0.6 + 0.8j) -> HarmonicDrive:
    return HarmonicDrive(
        constant=torch.zeros((), dtype=torch.float64),
        modes=(mode,),
        coefficients=torch.tensor([amplitude], dtype=torch.complex128),
    )


def _ring(radius: float, count: int = 96) -> torch.Tensor:
    angles = 2.0 * math.pi * torch.arange(count, dtype=torch.float64) / count
    return radius * unit_circle(angles)


def _replace_domain_data(
    domain: DomainMesh,
    *,
    boundary_values: torch.Tensor,
    boundary_points: torch.Tensor | None = None,
    query_points: torch.Tensor | None = None,
) -> DomainMesh:
    original_boundary = domain.boundaries["dirichlet"]
    boundary = (
        original_boundary.with_data(cell_data={"boundary_value": boundary_values})
        if boundary_points is None
        else Mesh(
            points=boundary_points,
            cells=original_boundary.cells,
            cell_data={"boundary_value": boundary_values},
        )
    )
    interior = (
        domain.interior
        if query_points is None
        else Mesh(points=query_points, point_data={})
    )
    return DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data=domain.global_data,
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("reflection", [False, True])
def test_stf_contraction_is_o2_invariant(order: int, reflection: bool) -> None:
    """Matching STF contractions must survive rotations and reflections."""

    first = torch.tensor([[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]], dtype=torch.float64)
    second = torch.tensor([[-0.2, 0.8], [0.5, -1.3], [0.7, 0.6]], dtype=torch.float64)
    transform = sample_similarity(
        711,
        scale_range=(1.0, 1.0),
        translation_extent=0.0,
        reflection=reflection,
        dtype=torch.float64,
    ).rotation
    transformed_first = torch.einsum("nd,ed->ne", first, transform)
    transformed_second = torch.einsum("nd,ed->ne", second, transform)
    contraction = torch.sum(
        planar_stf_coordinates(first, order) * planar_stf_coordinates(second, order),
        dim=-1,
    )
    transformed_contraction = torch.sum(
        planar_stf_coordinates(transformed_first, order)
        * planar_stf_coordinates(transformed_second, order),
        dim=-1,
    )
    torch.testing.assert_close(
        transformed_contraction, contraction, rtol=2.0e-14, atol=2.0e-14
    )


@pytest.mark.parametrize("lmax", [1, 2, 4])
@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
def test_disk_mode_is_present_exactly_when_its_order_exists(
    lmax: int, mode: int
) -> None:
    """The truncated decoder has a sharp, width-independent angular ceiling."""

    sample = build_domain_sample(
        _unit_disk_geometry(),
        _pure_mode_drive(mode),
        n_boundary=256,
        query_preimages=_ring(0.71),
    )
    model = STFMultipolePotential(lmax=lmax).double()
    prediction = model(sample.domain).point_data["potential"]
    target = sample.target
    if mode <= lmax:
        relative_error = torch.linalg.vector_norm(prediction - target)
        relative_error = relative_error / torch.linalg.vector_norm(target)
        assert relative_error.item() < 5.0e-4
    else:
        torch.testing.assert_close(
            prediction, torch.zeros_like(prediction), rtol=0.0, atol=2.0e-14
        )


def test_constant_lift_and_drive_superposition_are_exact() -> None:
    """All learned paths must remain homogeneous linear maps of boundary data."""

    geometry = sample_geometry(
        811, modes=(2, 3), deformation_range=(0.31, 0.31), dtype=torch.float64
    )
    sample = build_domain_sample(
        geometry,
        sample_drive(813, dtype=torch.float64),
        n_boundary=91,
        n_query=73,
        query_seed=817,
    )
    boundary = sample.domain.boundaries["dirichlet"]
    first = boundary.cell_data["boundary_value"]
    second = sample_drive(821, dtype=torch.float64)
    second_sample = build_domain_sample(
        geometry,
        second,
        n_boundary=91,
        query_preimages=sample.query_preimages,
    )
    second_values = second_sample.domain.boundaries["dirichlet"].cell_data[
        "boundary_value"
    ]
    model = STFMultipolePotential(lmax=4, channels_per_order=3).double()

    # Exercise nonconstant learned geometry gates while preserving their
    # operator-only inputs.
    generator = torch.Generator().manual_seed(823)
    with torch.no_grad():
        for module in (*model.source_gates, *model.query_gates):
            final = module.network[-1]
            final.weight.copy_(
                0.03
                * torch.randn(
                    final.weight.shape,
                    generator=generator,
                    dtype=final.weight.dtype,
                )
            )
            final.bias.copy_(
                0.03
                * torch.randn(
                    final.bias.shape,
                    generator=generator,
                    dtype=final.bias.dtype,
                )
            )

    alpha, beta = 1.7, -0.4
    first_prediction = model(
        _replace_domain_data(sample.domain, boundary_values=first)
    ).point_data["potential"]
    second_prediction = model(
        _replace_domain_data(sample.domain, boundary_values=second_values)
    ).point_data["potential"]
    combined_prediction = model(
        _replace_domain_data(
            sample.domain, boundary_values=alpha * first + beta * second_values
        )
    ).point_data["potential"]
    torch.testing.assert_close(
        combined_prediction,
        alpha * first_prediction + beta * second_prediction,
        rtol=3.0e-13,
        atol=3.0e-13,
    )

    constant = first.new_full(first.shape, 2.75)
    constant_prediction = model(
        _replace_domain_data(sample.domain, boundary_values=constant)
    ).point_data["potential"]
    torch.testing.assert_close(
        constant_prediction,
        constant_prediction.new_full(constant_prediction.shape, 2.75),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_complete_model_is_similarity_invariant(reflection: bool) -> None:
    """A scalar potential must be unchanged by joint physical similarities."""

    geometry = sample_geometry(
        907, modes=(2, 4), deformation_range=(0.37, 0.37), dtype=torch.float64
    )
    sample = build_domain_sample(
        geometry,
        sample_drive(911, modes=(1, 2, 3, 4), dtype=torch.float64),
        n_boundary=127,
        n_query=83,
        query_seed=919,
    )
    transformed = transform_sample(
        sample,
        sample_similarity(
            929,
            scale_range=(2.4, 2.4),
            translation_extent=3.0,
            reflection=reflection,
            dtype=torch.float64,
        ),
    )
    model = STFMultipolePotential(lmax=4, channels_per_order=3).double()
    base_prediction = model(sample.domain).point_data["potential"]
    transformed_prediction = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(
        transformed_prediction,
        base_prediction,
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_boundary_encoding_is_reusable_and_query_separable() -> None:
    """Chunked evaluation must reuse one source encoding without changing values."""

    sample = build_domain_sample(
        sample_geometry(1009, dtype=torch.float64),
        sample_drive(1013, dtype=torch.float64),
        n_boundary=67,
        n_query=89,
        query_seed=1019,
    )
    model = STFMultipolePotential(lmax=4).double()
    encoding = model.encode_boundary(
        sample.domain.boundaries["dirichlet"],
        sample.domain.global_data["reference_length"],
    )
    points = sample.domain.interior.points
    complete = model.decode_points(points, encoding)
    chunked = torch.cat(
        (
            model.decode_points(points[:31], encoding),
            model.decode_points(points[31:], encoding),
        )
    )
    torch.testing.assert_close(chunked, complete, rtol=0.0, atol=0.0)


def test_disk_quadrature_converges_at_second_order() -> None:
    """Panel refinement must converge to the represented mode without retuning."""

    model = STFMultipolePotential(lmax=4).double()
    errors: list[float] = []
    for n_boundary in (12, 24, 48, 96):
        sample = build_domain_sample(
            _unit_disk_geometry(),
            _pure_mode_drive(4),
            n_boundary=n_boundary,
            query_preimages=_ring(0.78),
        )
        prediction = model(sample.domain).point_data["potential"]
        relative_error = torch.linalg.vector_norm(prediction - sample.target)
        relative_error = relative_error / torch.linalg.vector_norm(sample.target)
        errors.append(relative_error.item())
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert all(coarse / fine > 3.5 for coarse, fine in zip(errors[:-1], errors[1:]))


def test_drive_query_and_parameter_gradients_are_finite() -> None:
    """Moment construction and evaluation must retain end-to-end autograd."""

    sample = build_domain_sample(
        sample_geometry(1103, dtype=torch.float64),
        sample_drive(1109, dtype=torch.float64),
        n_boundary=37,
        n_query=29,
        query_seed=1117,
    )
    values = (
        sample.domain.boundaries["dirichlet"]
        .cell_data["boundary_value"]
        .detach()
        .clone()
        .requires_grad_()
    )
    boundary_points = (
        sample.domain.boundaries["dirichlet"].points.detach().clone().requires_grad_()
    )
    points = sample.domain.interior.points.detach().clone().requires_grad_()
    domain = _replace_domain_data(
        sample.domain,
        boundary_values=values,
        boundary_points=boundary_points,
        query_points=points,
    )
    model = STFMultipolePotential(lmax=4).double()
    prediction = model(domain).point_data["potential"]
    prediction.square().mean().backward()

    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert (
        boundary_points.grad is not None and torch.isfinite(boundary_points.grad).all()
    )
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert values.grad.abs().max().item() > 0.0
    assert boundary_points.grad.abs().max().item() > 0.0
    assert points.grad.abs().max().item() > 0.0
    assert model.coefficients.grad is not None
    assert torch.isfinite(model.coefficients.grad).all()
    assert model.coefficients.grad.abs().max().item() > 0.0


def test_lmax_is_a_physical_order_not_an_arbitrary_integer() -> None:
    """Only the benchmark's prespecified physical truncations are accepted."""

    with pytest.raises(ValueError, match="lmax must be one of"):
        STFMultipolePotential(lmax=3)
