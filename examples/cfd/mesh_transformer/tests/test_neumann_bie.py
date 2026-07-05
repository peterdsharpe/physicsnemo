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

"""Contracts for the Neumann generator extension and Neumann panel BIE."""

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
    boundary_outward_normals,
    build_domain_sample,
    build_neumann_domain_sample,
    evaluate_flux,
    evaluate_potential,
    map_to_physical,
    sample_drive,
    sample_geometry,
    sample_similarity,
)
from self_consistent_kernel import NeumannHarmonicPanelBIE  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _case(seed_offset: int = 0):
    geometry = sample_geometry(
        401 + seed_offset, deformation_range=(0.25, 0.25), dtype=torch.float64
    )
    drive = sample_drive(
        402 + seed_offset, modes=(1, 2, 3), regularity=0.0, dtype=torch.float64
    )
    return geometry, drive


def _neumann_sample(n_query: int = 19, *, similarity=None):
    geometry, drive = _case()
    return build_neumann_domain_sample(
        geometry,
        drive,
        n_boundary=24,
        n_query=n_query,
        query_seed=403,
        similarity=similarity,
    )


def _tiny_model(**overrides) -> NeumannHarmonicPanelBIE:
    settings = dict(regular_orders=2, n_iterations=4)
    settings.update(overrides)
    return NeumannHarmonicPanelBIE(**settings).to(dtype=torch.float64)


def _with_boundary_flux(domain: DomainMesh, flux: torch.Tensor) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"]
    return DomainMesh(
        interior=domain.interior,
        boundaries={"dirichlet": boundary.with_data(cell_data={"boundary_flux": flux})},
        global_data=domain.global_data,
    )


def _two_test_panels():
    panel_start = torch.tensor([[0.4, -0.9], [-1.2, 0.3]], dtype=torch.float64)
    panel_end = torch.tensor([[0.9, -0.5], [-0.8, 0.9]], dtype=torch.float64)
    tangent = panel_end - panel_start
    lengths = tangent.norm(dim=-1)
    tangent = tangent / lengths[:, None]
    normals = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=-1)
    return panel_start, panel_end, lengths, tangent, normals


def test_evaluate_flux_matches_autograd_through_the_map_chain() -> None:
    """Verify the closed-form flux against exact autograd of u(x(z))."""

    geometry, drive = _case()
    similarity = sample_similarity(
        404,
        scale_range=(1.7, 1.7),
        translation_extent=1.0,
        reflection=True,
        dtype=torch.float64,
    )
    sample = build_neumann_domain_sample(
        geometry, drive, n_boundary=16, n_query=5, query_seed=405, similarity=similarity
    )
    preimages = sample.boundary_midpoint_preimages
    normals = boundary_outward_normals(geometry, torch.angle(preimages), similarity)
    flux = evaluate_flux(geometry, drive, similarity, preimages, normals)

    # grad_x u = J^{-T} grad_z u with J = dx/dz through the full map chain;
    # this is exact differentiation, tighter than directional finite
    # differences of the potential.
    for index in range(preimages.shape[0]):
        z_components = (
            torch.stack((preimages[index].real, preimages[index].imag))
            .clone()
            .requires_grad_(True)
        )

        def physical_point(z_components: torch.Tensor) -> torch.Tensor:
            z = torch.complex(z_components[0], z_components[1])
            return map_to_physical(geometry, z[None], similarity)[0]

        jacobian = torch.autograd.functional.jacobian(physical_point, z_components)
        potential = evaluate_potential(
            drive, torch.complex(z_components[0], z_components[1])[None]
        )[0]
        (gradient_z,) = torch.autograd.grad(potential, z_components)
        gradient_x = torch.linalg.solve(jacobian.T, gradient_z)
        torch.testing.assert_close(
            flux[index],
            normals[index] @ gradient_x,
            rtol=1.0e-11,
            atol=1.0e-12,
        )


def test_evaluate_flux_similarity_scaling_and_constant_drive() -> None:
    """Scale flux by 1/s under similarities; constant drives have zero flux."""

    geometry, drive = _case()
    identity_sample = build_neumann_domain_sample(
        geometry, drive, n_boundary=16, n_query=5, query_seed=406
    )
    preimages = identity_sample.boundary_midpoint_preimages
    base_normals = boundary_outward_normals(
        geometry, torch.angle(preimages), identity_sample.similarity
    )
    base_flux = evaluate_flux(
        geometry, drive, identity_sample.similarity, preimages, base_normals
    )

    similarity = sample_similarity(
        407,
        scale_range=(3.0, 3.0),
        translation_extent=2.0,
        reflection=True,
        dtype=torch.float64,
    )
    transformed_normals = boundary_outward_normals(
        geometry, torch.angle(preimages), similarity
    )
    transformed_flux = evaluate_flux(
        geometry, drive, similarity, preimages, transformed_normals
    )
    torch.testing.assert_close(
        transformed_flux,
        base_flux / similarity.scale,
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    constant_drive = HarmonicDrive(
        constant=torch.tensor(0.9, dtype=torch.float64),
        modes=(),
        coefficients=torch.empty(0, dtype=torch.complex128),
    )
    zero_flux = evaluate_flux(
        geometry, constant_drive, identity_sample.similarity, preimages, base_normals
    )
    torch.testing.assert_close(
        zero_flux, torch.zeros_like(zero_flux), rtol=0.0, atol=0.0
    )


def test_neumann_sample_is_compatible_and_gauge_fixed() -> None:
    """Zero the discrete flux integral and gauge the targets exactly."""

    geometry, drive = _case()
    similarity = sample_similarity(
        408,
        scale_range=(2.2, 2.2),
        translation_extent=1.5,
        reflection=False,
        dtype=torch.float64,
    )
    neumann = build_neumann_domain_sample(
        geometry,
        drive,
        n_boundary=32,
        n_query=11,
        query_seed=409,
        similarity=similarity,
    )
    dirichlet = build_domain_sample(
        geometry,
        drive,
        n_boundary=32,
        query_preimages=neumann.query_preimages,
        similarity=similarity,
    )
    boundary = neumann.domain.boundaries["dirichlet"]
    weights = boundary.cell_areas

    compatibility = torch.sum(weights * boundary.cell_data["boundary_flux"])
    torch.testing.assert_close(
        compatibility, torch.zeros_like(compatibility), rtol=0.0, atol=1.0e-13
    )

    raw_values = dirichlet.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    gauge = torch.sum(weights * raw_values) / weights.sum()
    torch.testing.assert_close(
        boundary.cell_data["boundary_value"],
        raw_values - gauge,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        neumann.target,
        dirichlet.target - gauge,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(neumann.area_jacobian, dirichlet.area_jacobian)


def test_panel_influence_matches_brute_force_on_both_sides() -> None:
    """Verify the log/singular/entire closed forms against dense quadrature.

    Queries sit on both sides of each panel, close to the panel plane, and on
    the panel's line extension -- the configurations where a principal-branch
    jump of ``log`` would corrupt a naive antiderivative evaluation.
    """

    torch.manual_seed(410)
    model = _tiny_model(regular_orders=2, n_iterations=0)
    with torch.no_grad():
        model.log_coefficient.normal_(std=1.0)
        model.singular_coefficient.normal_(std=1.0)
        model.regular_coefficients.normal_(std=1.0)

    panel_start, panel_end, lengths, tangent, normals = _two_test_panels()
    midpoints = 0.5 * (panel_start + panel_end)
    queries = torch.cat(
        (
            midpoints + 0.3 * normals,
            midpoints - 0.3 * normals,
            midpoints + 0.01 * normals,
            midpoints - 0.01 * normals,
            panel_start - 0.7 * lengths[:, None] * tangent,
        )
    )
    exact = model._influence(queries, panel_start, panel_end, normals)

    n_sub = 200_000
    t = (torch.arange(n_sub, dtype=torch.float64) + 0.5) / n_sub
    brute = torch.zeros_like(exact)
    c0 = model.log_coefficient
    c1 = model.singular_coefficient
    d = model.regular_coefficients
    for s in range(panel_start.shape[0]):
        points = panel_start[s][None, :] + t[:, None] * (panel_end[s] - panel_start[s])
        for q in range(queries.shape[0]):
            r = queries[q][None, :] - points
            parallel = r @ normals[s]
            perpendicular = r[:, 0] * normals[s][1] - r[:, 1] * normals[s][0]
            zeta = torch.complex(parallel, perpendicular)
            kernel = c0 * torch.log(zeta.abs()) + c1 * zeta.reciprocal().real
            power = torch.ones_like(zeta)
            for k in range(model.regular_orders + 1):
                kernel = kernel + d[k] * power.real
                power = power * zeta
            brute[q, s] = kernel.mean() * lengths[s]
    torch.testing.assert_close(exact, brute, rtol=1.0e-6, atol=1.0e-8)


def test_self_panel_log_integral_matches_closed_form() -> None:
    """Evaluate the finite self-panel log integral L(log(L/2) - 1) exactly."""

    model = _tiny_model(regular_orders=0, n_iterations=0)
    with torch.no_grad():
        model.log_coefficient.fill_(1.0)
        model.singular_coefficient.zero_()
        model.regular_coefficients.zero_()
    panel_start, panel_end, lengths, _, normals = _two_test_panels()
    midpoints = 0.5 * (panel_start + panel_end)
    diagonal = model._influence(
        midpoints, panel_start, panel_end, normals, zero_singular_diagonal=True
    ).diagonal()
    torch.testing.assert_close(
        diagonal,
        lengths * (torch.log(lengths / 2.0) - 1.0),
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_flux_trace_matches_normal_finite_differences() -> None:
    """Check the autograd Neumann trace against centered finite differences."""

    torch.manual_seed(411)
    model = _tiny_model(regular_orders=2, n_iterations=0)
    with torch.no_grad():
        model.log_coefficient.normal_(std=0.5)
        model.singular_coefficient.normal_(std=0.5)
        model.regular_coefficients.normal_(std=0.5)
    panel_start, panel_end, _, _, normals = _two_test_panels()
    midpoints = 0.5 * (panel_start + panel_end)

    matrix = model._flux_trace_matrix(midpoints, panel_start, panel_end, normals)
    step = 1.0e-6
    for row in range(midpoints.shape[0]):
        forward_row = model._influence(
            midpoints[row : row + 1] + step * normals[row : row + 1],
            panel_start,
            panel_end,
            normals,
        )[0]
        backward_row = model._influence(
            midpoints[row : row + 1] - step * normals[row : row + 1],
            panel_start,
            panel_end,
            normals,
        )[0]
        difference = (forward_row - backward_row) / (2.0 * step)
        difference[row] = 0.0  # the model's zeroed flux-trace diagonal
        torch.testing.assert_close(matrix[row], difference, rtol=1.0e-5, atol=1.0e-7)

    density = torch.randn(midpoints.shape[0], dtype=torch.float64)
    applied = model._apply_flux_trace(
        midpoints, panel_start, panel_end, normals, density
    )
    torch.testing.assert_close(applied, matrix @ density, rtol=1.0e-13, atol=1.0e-14)


def test_neumann_bie_is_linear_and_gauge_consistent() -> None:
    """Preserve flux superposition and report a zero-boundary-mean potential."""

    torch.manual_seed(412)
    model = _tiny_model()
    with torch.no_grad():
        model.log_coefficient.normal_(std=0.1)
        model.singular_coefficient.normal_(std=0.1)
        model.regular_coefficients.normal_(std=0.1)
    sample = _neumann_sample()
    boundary = sample.domain.boundaries["dirichlet"]
    first = torch.randn(boundary.n_cells, dtype=torch.float64)
    second = torch.randn(boundary.n_cells, dtype=torch.float64)

    def predict(flux: torch.Tensor) -> torch.Tensor:
        return model(_with_boundary_flux(sample.domain, flux)).point_data["potential"]

    torch.testing.assert_close(
        predict(1.7 * first - 0.4 * second),
        1.7 * predict(first) - 0.4 * predict(second),
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    # Neumann data determine the potential only up to a constant; the model's
    # defensive compatibility correction removes constant flux shifts exactly.
    torch.testing.assert_close(
        predict(first + 3.7),
        predict(first),
        rtol=0.0,
        atol=1.0e-11,
    )

    # The model's own gauge convention: the boundary mean of its reported
    # potential, evaluated through its exact panel-integrated trace, is zero.
    (
        _,
        weights,
        _,
        _,
        panel_start,
        panel_end,
        midpoints,
        _,
        density,
    ) = model._boundary_state(sample.domain)
    trace_influence = model._influence(
        midpoints,
        panel_start,
        panel_end,
        boundary.cell_normals,
        zero_singular_diagonal=True,
    )
    gauge = torch.sum(weights * (trace_influence @ density)) / weights.sum()
    reported_trace = trace_influence @ density - gauge
    boundary_mean = torch.sum(weights * reported_trace) / weights.sum()
    torch.testing.assert_close(
        boundary_mean, torch.zeros_like(boundary_mean), rtol=0.0, atol=1.0e-14
    )


def test_neumann_bie_is_o2_similarity_invariant() -> None:
    """Keep scalar predictions invariant under similarities and reflections."""

    torch.manual_seed(413)
    model = _tiny_model()
    with torch.no_grad():
        model.log_coefficient.normal_(std=0.1)
        model.singular_coefficient.normal_(std=0.1)
        model.regular_coefficients.normal_(std=0.1)
    base = _neumann_sample()
    transformed = build_neumann_domain_sample(
        base.geometry,
        base.drive,
        n_boundary=24,
        query_preimages=base.query_preimages,
        similarity=sample_similarity(
            414,
            scale_range=(3.5, 3.5),
            translation_extent=2.0,
            reflection=True,
            dtype=torch.float64,
        ),
    )
    expected = model(base.domain).point_data["potential"]
    actual = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=5.0e-10, atol=5.0e-11)


def test_neumann_bie_query_set_independence_is_exact() -> None:
    """Predict identically at a point regardless of other requested points."""

    torch.manual_seed(415)
    model = _tiny_model()
    sample = _neumann_sample(n_query=7)
    domain = sample.domain

    full = model(domain).point_data["potential"]
    single_domain = DomainMesh(
        interior=Mesh(points=domain.interior.points[3:4]),
        boundaries=dict(domain.boundaries.items()),
        global_data=domain.global_data,
    )
    single = model(single_domain).point_data["potential"]
    torch.testing.assert_close(single, full[3:4], rtol=0.0, atol=0.0)


def test_neumann_bie_influence_is_harmonic_in_the_query() -> None:
    """Check the ambient autograd Laplacian vanishes for random coefficients."""

    torch.manual_seed(416)
    model = _tiny_model(n_iterations=0)
    with torch.no_grad():
        model.log_coefficient.normal_(std=1.0)
        model.singular_coefficient.normal_(std=1.0)
        model.regular_coefficients.normal_(std=1.0)
    sample = _neumann_sample(n_query=9)
    boundary = sample.domain.boundaries["dirichlet"]
    vertices = boundary.points[boundary.cells]

    x = sample.domain.interior.points[5:6].clone().requires_grad_(True)
    influence = model._influence(
        x, vertices[:, 0], vertices[:, 1], boundary.cell_normals
    )
    (gradient,) = torch.autograd.grad(influence.sum(), x, create_graph=True)
    laplacian = torch.zeros((), dtype=torch.float64)
    for component in range(2):
        (second_derivative,) = torch.autograd.grad(
            gradient[0, component], x, create_graph=True
        )
        laplacian = laplacian + second_derivative[0, component]
    torch.testing.assert_close(
        laplacian, torch.zeros_like(laplacian), rtol=0.0, atol=1.0e-9
    )


def test_every_parameter_receives_finite_gradient() -> None:
    """Reach every coefficient and relaxation step from the interior loss."""

    torch.manual_seed(417)
    model = _tiny_model()
    output = model(_neumann_sample().domain).point_data["potential"]
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.relaxation.grad.abs().sum() > 0.0
    assert model.log_coefficient.grad.abs() > 0.0


def test_neumann_model_requires_flux_data() -> None:
    """Fail loudly on Dirichlet-only samples instead of guessing."""

    geometry, drive = _case()
    dirichlet = build_domain_sample(geometry, drive, n_boundary=16, n_query=4)
    model = _tiny_model()
    with pytest.raises(ValueError, match="boundary_flux"):
        model(dirichlet.domain)


def test_oracle_single_layer_dense_solve_recovers_the_disk_solution() -> None:
    """Pin the sign/gauge convention against the analytic disk problem.

    With the documented ``-1/2`` interior jump, the analytic single-layer
    member of the family is ``c0 = +1/(2 pi)`` (the model's density is the
    negative of the classical one; flipping both signs is the same operator).
    A dense solve replaces Richardson so only the kernel-and-jump convention
    is under test.  For ``u = Re(z)`` on the unit disk the exact density is
    smooth, so 128 panels must reach well under 5% relative error; the wrong
    sign fails at the 200% level.
    """

    disk = ConformalGeometry(
        modes=(), coefficients=torch.empty(0, dtype=torch.complex128)
    )
    drive = HarmonicDrive(
        constant=torch.zeros((), dtype=torch.float64),
        modes=(1,),
        coefficients=torch.tensor([1.0 + 0.0j], dtype=torch.complex128),
    )
    sample = build_neumann_domain_sample(
        disk, drive, n_boundary=128, n_query=256, query_seed=418
    )
    model = _tiny_model(regular_orders=0, n_iterations=0)
    with torch.no_grad():
        model.log_coefficient.fill_(-1.0 / (2.0 * math.pi))
        model.singular_coefficient.zero_()
        model.regular_coefficients.zero_()

    (
        boundary,
        weights,
        center,
        length,
        panel_start,
        panel_end,
        midpoints,
        flux,
        _,
    ) = model._boundary_state(sample.domain)
    normals = boundary.cell_normals
    trace_matrix = model._flux_trace_matrix(midpoints, panel_start, panel_end, normals)
    operator = (
        0.5 * torch.eye(trace_matrix.shape[0], dtype=torch.float64) + trace_matrix
    )
    density = torch.linalg.solve(operator, flux)

    trace_influence = model._influence(
        midpoints, panel_start, panel_end, normals, zero_singular_diagonal=True
    )
    gauge = torch.sum(weights * (trace_influence @ density)) / weights.sum()
    queries = (sample.domain.interior.points - center) / length
    prediction = (
        model._influence(queries, panel_start, panel_end, normals) @ density - gauge
    )
    relative_error = torch.sqrt(
        torch.sum((prediction - sample.target).square())
        / torch.sum(sample.target.square())
    ).detach()
    assert float(relative_error) < 0.05
