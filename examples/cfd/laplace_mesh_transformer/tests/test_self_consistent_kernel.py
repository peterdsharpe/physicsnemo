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

"""Contracts for the learned self-consistent boundary-integral kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    build_domain_sample,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
)
from models import InvariantPairKernel  # noqa: E402
from self_consistent_kernel import SelfConsistentPairKernel  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _sample(n_query: int = 19):
    geometry = sample_geometry(301, deformation_range=(0.25, 0.25), dtype=torch.float64)
    drive = sample_drive(302, modes=(1, 2, 3), regularity=0.0, dtype=torch.float64)
    return build_domain_sample(
        geometry, drive, n_boundary=24, n_query=n_query, query_seed=303
    )


def _tiny_model(**overrides) -> SelfConsistentPairKernel:
    settings = dict(hidden_dim=12, hidden_layers=2, n_iterations=4)
    settings.update(overrides)
    return SelfConsistentPairKernel(**settings).to(dtype=torch.float64)


def _with_boundary_values(domain: DomainMesh, values: torch.Tensor) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"]
    return DomainMesh(
        interior=domain.interior,
        boundaries={
            "dirichlet": boundary.with_data(cell_data={"boundary_value": values})
        },
        global_data=domain.global_data,
    )


def test_self_consistent_kernel_is_linear_in_boundary_drive() -> None:
    """Preserve exact drive superposition through solve and propagation."""

    torch.manual_seed(304)
    model = _tiny_model()
    sample = _sample()
    boundary = sample.domain.boundaries["dirichlet"]
    first = torch.randn(boundary.n_cells, dtype=torch.float64)
    second = torch.randn(boundary.n_cells, dtype=torch.float64)

    def predict(values: torch.Tensor) -> torch.Tensor:
        return model(_with_boundary_values(sample.domain, values)).point_data[
            "potential"
        ]

    torch.testing.assert_close(
        predict(1.7 * first - 0.4 * second),
        1.7 * predict(first) - 0.4 * predict(second),
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_constant_drive_reproduces_the_constant_exactly() -> None:
    """Lift constants analytically: zero residual density, zero trace error."""

    torch.manual_seed(305)
    model = _tiny_model()
    sample = _sample()
    boundary = sample.domain.boundaries["dirichlet"]
    constant = torch.full((boundary.n_cells,), 0.731, dtype=torch.float64)
    domain = _with_boundary_values(sample.domain, constant)

    prediction = model(domain).point_data["potential"]
    torch.testing.assert_close(
        prediction,
        torch.full_like(prediction, 0.731),
        rtol=0.0,
        atol=1.0e-14,
    )
    residual = model.collocation_residual(domain)
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-14
    )


def test_self_consistent_kernel_is_o2_similarity_invariant() -> None:
    """Keep scalar predictions invariant under similarities and reflections."""

    torch.manual_seed(306)
    model = _tiny_model()
    sample = _sample()
    transformed = transform_sample(
        sample,
        sample_similarity(
            307,
            scale_range=(3.5, 3.5),
            translation_extent=2.0,
            reflection=True,
            dtype=torch.float64,
        ),
    )

    expected = model(sample.domain).point_data["potential"]
    actual = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=5.0e-10, atol=5.0e-11)


def test_query_set_independence_is_exact() -> None:
    """Predict identically at a point regardless of other requested points."""

    torch.manual_seed(308)
    model = _tiny_model()
    sample = _sample(n_query=7)
    domain = sample.domain

    full = model(domain).point_data["potential"]
    single_interior = Mesh(points=domain.interior.points[3:4])
    single_domain = DomainMesh(
        interior=single_interior,
        boundaries=dict(domain.boundaries.items()),
        global_data=domain.global_data,
    )
    single = model(single_domain).point_data["potential"]
    torch.testing.assert_close(single, full[3:4], rtol=0.0, atol=0.0)


def test_zero_iterations_reduces_to_invariant_pair_kernel() -> None:
    """Make ``n_iterations=0`` the exact InvariantPairKernel control."""

    torch.manual_seed(309)
    model = _tiny_model(n_iterations=0)
    control = InvariantPairKernel(hidden_dim=12, hidden_layers=2).to(
        dtype=torch.float64
    )
    control.kernel.load_state_dict(model.kernel.mlp.state_dict())
    sample = _sample()

    torch.testing.assert_close(
        model(sample.domain).point_data["potential"],
        control(sample.domain).point_data["potential"],
        rtol=0.0,
        atol=0.0,
    )


def test_every_parameter_receives_finite_gradient() -> None:
    """Reach the kernel and every relaxation step from the interior loss."""

    torch.manual_seed(310)
    model = _tiny_model()
    output = model(_sample().domain).point_data["potential"]
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.relaxation.grad.abs().sum() > 0.0


def test_untied_variant_learns_an_independent_solve_kernel() -> None:
    """Register separate solve parameters only in the untied ablation."""

    tied = _tiny_model()
    untied = _tiny_model(tied=False)
    assert tied.solve_kernel is tied.kernel
    assert untied.solve_kernel is not untied.kernel
    tied_count = sum(p.numel() for p in tied.parameters())
    untied_count = sum(p.numel() for p in untied.parameters())
    kernel_count = sum(p.numel() for p in tied.kernel.parameters())
    assert untied_count == tied_count + kernel_count


def test_invariant_laplacian_matches_ambient_autograd() -> None:
    """Verify the closed-form chain rule against a brute-force x-Laplacian."""

    torch.manual_seed(311)
    model = _tiny_model(kernel_pde_loss=True)
    source = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    x = torch.tensor([[1.1, 0.4]], dtype=torch.float64, requires_grad=True)

    def kappa_of_x(point: torch.Tensor) -> torch.Tensor:
        displacement = point[:, None, :] - source[None, :, :]
        features = torch.stack(
            (
                displacement.square().sum(dim=-1),
                torch.einsum("qsd,sd->qs", displacement, normal),
            ),
            dim=-1,
        )
        return model.kernel.mlp(features).squeeze()

    kappa = kappa_of_x(x)
    (gradient,) = torch.autograd.grad(kappa, x, create_graph=True)
    laplacian = torch.zeros((), dtype=torch.float64)
    for component in range(2):
        (second,) = torch.autograd.grad(gradient[0, component], x, create_graph=True)
        laplacian = laplacian + second[0, component]

    displacement = (x.detach() - source).reshape(1, 2)
    features = torch.stack(
        (
            displacement.square().sum(dim=-1),
            torch.einsum("pd,pd->p", displacement, normal),
        ),
        dim=-1,
    ).requires_grad_(True)
    value = model.kernel.mlp(features).squeeze(-1)
    (first,) = torch.autograd.grad(value.sum(), features, create_graph=True)
    (second_a,) = torch.autograd.grad(first[..., 0].sum(), features, create_graph=True)
    (second_b,) = torch.autograd.grad(first[..., 1].sum(), features, create_graph=True)
    a = features[..., 0].detach()
    b = features[..., 1].detach()
    closed_form = (
        4.0 * a * second_a[..., 0]
        + 4.0 * b * second_a[..., 1]
        + second_b[..., 1]
        + 4.0 * first[..., 0]
    )
    torch.testing.assert_close(
        closed_form.squeeze(), laplacian, rtol=1.0e-10, atol=1.0e-12
    )


def test_auxiliary_loss_is_finite_and_reaches_the_kernel() -> None:
    """Backpropagate both auxiliary terms into the shared kernel."""

    torch.manual_seed(312)
    model = _tiny_model(trace_loss=True, kernel_pde_loss=True, pde_pair_samples=64)
    loss = model.auxiliary_loss(_sample().domain)
    assert torch.isfinite(loss)
    assert loss >= 0.0
    loss.backward()
    kernel_gradients = [p.grad for p in model.kernel.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in kernel_gradients)


def test_trace_auxiliary_is_exactly_zero_for_constant_drive() -> None:
    """Constant data has zero residual drive, hence zero trace penalty."""

    torch.manual_seed(313)
    model = _tiny_model(trace_loss=True)
    sample = _sample()
    boundary = sample.domain.boundaries["dirichlet"]
    constant = torch.full((boundary.n_cells,), -1.4, dtype=torch.float64)
    domain = _with_boundary_values(sample.domain, constant)
    loss = model.auxiliary_loss(domain)
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0.0, atol=1.0e-25)


def test_harmonic_kernel_is_exactly_harmonic_in_the_query() -> None:
    """Check the ambient autograd Laplacian vanishes for random coefficients."""

    torch.manual_seed(314)
    from self_consistent_kernel import _HarmonicPairKernel

    kernel = _HarmonicPairKernel(singular_orders=2, regular_orders=2).to(
        dtype=torch.float64
    )
    with torch.no_grad():
        kernel.coefficients.normal_(std=1.0)
    source = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    x = torch.tensor([[1.1, 0.4]], dtype=torch.float64, requires_grad=True)

    kappa = kernel(x[:, None, :] - source[None, :, :], normal).squeeze()
    (gradient,) = torch.autograd.grad(kappa, x, create_graph=True)
    laplacian = torch.zeros((), dtype=torch.float64)
    for component in range(2):
        (second,) = torch.autograd.grad(gradient[0, component], x, create_graph=True)
        laplacian = laplacian + second[0, component]
    torch.testing.assert_close(
        laplacian, torch.zeros_like(laplacian), rtol=0.0, atol=1.0e-10
    )


def test_harmonic_kernel_is_reflection_invariant_and_coincident_safe() -> None:
    """Flip the frame handedness and hit the coincident point without NaN."""

    torch.manual_seed(315)
    from self_consistent_kernel import _HarmonicPairKernel

    kernel = _HarmonicPairKernel().to(dtype=torch.float64)
    with torch.no_grad():
        kernel.coefficients.normal_(std=1.0)
    displacement = torch.tensor([[[0.7, -0.4]]], dtype=torch.float64)
    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    reflect = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

    value = kernel(displacement, normal)
    mirrored = kernel(displacement * reflect[None, :, :], normal * reflect)
    torch.testing.assert_close(mirrored, value, rtol=1.0e-14, atol=1.0e-14)

    coincident = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    out = kernel(coincident, normal)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0.0, atol=0.0)
    loss = kernel(
        torch.cat((coincident, displacement), dim=1),
        torch.cat((normal, normal), dim=0),
    ).sum()
    loss.backward()
    assert torch.isfinite(kernel.coefficients.grad).all()


def test_harmonic_family_preserves_all_model_contracts() -> None:
    """Run superposition, similarity, and query independence for harmonic BIE."""

    torch.manual_seed(316)
    model = SelfConsistentPairKernel(
        kernel_family="harmonic", n_iterations=4, trace_loss=True
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.kernel.coefficients.normal_(std=0.3)
    sample = _sample(n_query=9)
    boundary = sample.domain.boundaries["dirichlet"]
    first = torch.randn(boundary.n_cells, dtype=torch.float64)
    second = torch.randn(boundary.n_cells, dtype=torch.float64)

    def predict(domain):
        return model(domain).point_data["potential"]

    torch.testing.assert_close(
        predict(_with_boundary_values(sample.domain, 0.9 * first + 2.1 * second)),
        0.9 * predict(_with_boundary_values(sample.domain, first))
        + 2.1 * predict(_with_boundary_values(sample.domain, second)),
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    transformed = transform_sample(
        sample,
        sample_similarity(
            317,
            scale_range=(2.5, 2.5),
            translation_extent=1.5,
            reflection=True,
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        predict(transformed.domain),
        predict(sample.domain),
        rtol=5.0e-10,
        atol=5.0e-11,
    )

    full = predict(sample.domain)
    single_domain = DomainMesh(
        interior=Mesh(points=sample.domain.interior.points[4:5]),
        boundaries=dict(sample.domain.boundaries.items()),
        global_data=sample.domain.global_data,
    )
    torch.testing.assert_close(predict(single_domain), full[4:5], rtol=0.0, atol=0.0)

    aux = model.auxiliary_loss(sample.domain)
    assert torch.isfinite(aux)


def test_panel_influence_matches_brute_force_quadrature() -> None:
    """Verify exact panel integrals against dense midpoint quadrature."""

    torch.manual_seed(318)
    from self_consistent_kernel import HarmonicPanelBIE

    model = HarmonicPanelBIE(regular_orders=3, n_iterations=0).to(dtype=torch.float64)
    with torch.no_grad():
        model.singular_coefficient.normal_(std=1.0)
        model.regular_coefficients.normal_(std=1.0)

    panel_start = torch.tensor([[0.4, -0.9], [-1.2, 0.3]], dtype=torch.float64)
    panel_end = torch.tensor([[0.9, -0.5], [-0.8, 0.9]], dtype=torch.float64)
    tangent = panel_end - panel_start
    lengths = tangent.norm(dim=-1)
    tangent = tangent / lengths[:, None]
    normals = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=-1)
    queries = torch.tensor([[0.1, 0.2], [-0.3, -0.4]], dtype=torch.float64)

    exact = model._influence(queries, panel_start, panel_end, normals)

    n_sub = 200_000
    t = (torch.arange(n_sub, dtype=torch.float64) + 0.5) / n_sub
    brute = torch.zeros_like(exact)
    c1 = model.singular_coefficient
    d = model.regular_coefficients
    for s in range(panel_start.shape[0]):
        points = panel_start[s][None, :] + t[:, None] * (panel_end[s] - panel_start[s])
        for q in range(queries.shape[0]):
            r = queries[q][None, :] - points
            parallel = r @ normals[s]
            perpendicular = r[:, 0] * normals[s][1] - r[:, 1] * normals[s][0]
            zeta = torch.complex(parallel, perpendicular)
            kernel = c1 * (zeta.reciprocal()).real
            power = torch.ones_like(zeta)
            for k in range(model.regular_orders + 1):
                kernel = kernel + d[k] * power.real
                power = power * zeta
            brute[q, s] = kernel.mean() * lengths[s]
    torch.testing.assert_close(exact, brute, rtol=1.0e-8, atol=1.0e-10)


def test_panel_bie_reduces_to_analytic_double_layer_at_oracle_point() -> None:
    """Match the analytic influence exactly at c1 = -1/(2 pi), d = 0."""

    import math as _math

    from layer_potential import double_layer_influence
    from self_consistent_kernel import HarmonicPanelBIE

    model = HarmonicPanelBIE(regular_orders=2, n_iterations=0).to(dtype=torch.float64)
    with torch.no_grad():
        model.singular_coefficient.fill_(-1.0 / (2.0 * _math.pi))
        model.regular_coefficients.zero_()

    sample = _sample(n_query=11)
    boundary = sample.domain.boundaries["dirichlet"]
    vertices = boundary.points[boundary.cells]
    queries = sample.domain.interior.points

    mine = model._influence(
        queries, vertices[:, 0], vertices[:, 1], boundary.cell_normals
    )
    analytic = double_layer_influence(boundary, queries)
    torch.testing.assert_close(mine, analytic, rtol=1.0e-12, atol=1.0e-13)


def test_panel_bie_preserves_contracts_and_is_harmonic() -> None:
    """Check superposition, similarity, query independence, and harmonicity."""

    torch.manual_seed(319)
    from self_consistent_kernel import HarmonicPanelBIE

    model = HarmonicPanelBIE(n_iterations=4).to(dtype=torch.float64)
    with torch.no_grad():
        model.singular_coefficient.normal_(std=0.1)
        model.regular_coefficients.normal_(std=0.1)
    sample = _sample(n_query=9)
    boundary = sample.domain.boundaries["dirichlet"]
    first = torch.randn(boundary.n_cells, dtype=torch.float64)
    second = torch.randn(boundary.n_cells, dtype=torch.float64)

    def predict(domain):
        return model(domain).point_data["potential"]

    torch.testing.assert_close(
        predict(_with_boundary_values(sample.domain, 1.3 * first - 0.7 * second)),
        1.3 * predict(_with_boundary_values(sample.domain, first))
        - 0.7 * predict(_with_boundary_values(sample.domain, second)),
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    transformed = transform_sample(
        sample,
        sample_similarity(
            320,
            scale_range=(2.0, 2.0),
            translation_extent=1.0,
            reflection=True,
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        predict(transformed.domain),
        predict(sample.domain),
        rtol=5.0e-10,
        atol=5.0e-11,
    )

    full = predict(sample.domain)
    single_domain = DomainMesh(
        interior=Mesh(points=sample.domain.interior.points[2:3]),
        boundaries=dict(sample.domain.boundaries.items()),
        global_data=sample.domain.global_data,
    )
    torch.testing.assert_close(predict(single_domain), full[2:3], rtol=0.0, atol=0.0)

    x = sample.domain.interior.points[5:6].clone().requires_grad_(True)
    vertices = boundary.points[boundary.cells]
    influence = model._influence(
        x, vertices[:, 0], vertices[:, 1], boundary.cell_normals
    )
    value = influence.sum()
    (gradient,) = torch.autograd.grad(value, x, create_graph=True)
    laplacian = torch.zeros((), dtype=torch.float64)
    for component in range(2):
        (second_derivative,) = torch.autograd.grad(
            gradient[0, component], x, create_graph=True
        )
        laplacian = laplacian + second_derivative[0, component]
    torch.testing.assert_close(
        laplacian, torch.zeros_like(laplacian), rtol=0.0, atol=1.0e-9
    )
