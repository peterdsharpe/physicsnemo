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

"""PDE and symmetry contracts for the Laplace layer-potential controls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    build_domain_sample,
    sample_disk_preimages,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
)
from layer_potential import (  # noqa: E402
    DirectDoubleLayerPotential,
    EncodedDoubleLayerPotential,
    LearnedDensityDoubleLayerPotential,
    SolvedDoubleLayerPotential,
    double_layer_collocation_matrix,
    double_layer_influence,
    evaluate_double_layer,
    solve_double_layer_density,
)
from models import MeshTransformerConfig  # noqa: E402

from physicsnemo.mesh import DomainMesh  # noqa: E402


def _sample(*, n_boundary: int = 64, n_query: int = 48):
    geometry = sample_geometry(
        6101,
        modes=(2, 3),
        deformation_range=(0.3, 0.3),
        dtype=torch.float64,
    )
    drive = sample_drive(
        6103,
        modes=(1, 2, 3, 4, 5, 6),
        regularity=0.0,
        dtype=torch.float64,
    )
    preimages = 0.85 * sample_disk_preimages(6107, n_query, dtype=torch.float64)
    return build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        query_preimages=preimages,
    )


def _replace_boundary_values(domain: DomainMesh, values: torch.Tensor) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"].with_data(
        cell_data={"boundary_value": values}
    )
    return DomainMesh(
        interior=domain.interior,
        boundaries={"dirichlet": boundary},
        global_data=domain.global_data,
    )


def _prediction(model, domain: DomainMesh) -> torch.Tensor:
    return model(domain).point_data["potential"]


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((prediction - target).square()) / target.square().sum())


def _tiny_encoder_config() -> MeshTransformerConfig:
    return MeshTransformerConfig(
        operator_scalar_dim=8,
        operator_vector_dim=2,
        drive_scalar_dim=8,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=1,  # Constructed transiently, then omitted by the hybrid.
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=11,
        attention_chunk_size=17,
    )


def test_closed_panel_measure_integrates_constant_density_exactly() -> None:
    """Exact panel measure must recover the closed-curve winding identity."""

    sample = _sample(n_boundary=37, n_query=31)
    boundary = sample.domain.boundaries["dirichlet"]
    ones = torch.ones(boundary.n_cells, dtype=torch.float64)

    potential = evaluate_double_layer(
        boundary, ones, sample.domain.interior.points, query_chunk_size=7
    )
    torch.testing.assert_close(
        potential, torch.ones_like(potential), rtol=2.0e-14, atol=2.0e-14
    )

    # The half jump plus principal-value panel integrals has the same winding
    # identity on the boundary.  This also tests that self-panels do not pass
    # through the pointwise singular kernel.
    system = double_layer_collocation_matrix(boundary)
    torch.testing.assert_close(system @ ones, ones, rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.parametrize(
    "model",
    [
        DirectDoubleLayerPotential(),
        SolvedDoubleLayerPotential(),
        LearnedDensityDoubleLayerPotential(n_iterations=2),
    ],
)
def test_all_layer_controls_lift_constants_outside_chordal_polygon(model) -> None:
    """The exact constant solution must not depend on polygon winding side."""

    n_boundary = 16
    geometry = sample_geometry(
        6113,
        modes=(2,),
        deformation_range=(0.0, 0.0),
        dtype=torch.float64,
    )
    drive = sample_drive(6115, modes=(1,), dtype=torch.float64)
    angles = (
        2.0
        * torch.pi
        * (torch.arange(n_boundary, dtype=torch.float64) + 0.5)
        / n_boundary
    )
    # Radius .99 lies inside the smooth unit disk but outside the 16-gon near
    # panel mid-angles (its apothem is cos(pi/16) ~= .9808).
    queries = torch.polar(torch.full_like(angles, 0.99), angles)
    sample = build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        query_preimages=queries,
    )
    constant = torch.full((n_boundary,), 1.7, dtype=torch.float64)
    domain = _replace_boundary_values(sample.domain, constant)

    torch.testing.assert_close(
        _prediction(model.double(), domain),
        torch.full((queries.numel(),), 1.7, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_pointwise_evaluation_rejects_boundary_singularity() -> None:
    """A boundary query must use a jump relation, not point evaluation."""

    sample = _sample(n_boundary=24, n_query=4)
    boundary = sample.domain.boundaries["dirichlet"]
    with pytest.raises(ValueError, match="singular on a panel"):
        double_layer_influence(boundary, boundary.cell_centroids[:1])


def test_solved_density_enforces_trace_and_resolves_high_modes() -> None:
    """The solved density must fit the trace and propagate higher modes."""

    sample = _sample(n_boundary=96, n_query=256)
    model = SolvedDoubleLayerPotential(query_chunk_size=29)

    prediction = _prediction(model, sample.domain)
    assert _relative_l2(prediction, sample.target).item() < 3.0e-3
    assert model.collocation_residual(sample.domain).abs().max().item() < 1.0e-12
    assert model.collocation_loss(sample.domain).item() < 1.0e-24


def test_panel_refinement_converges_to_continuous_solution() -> None:
    """Constant-panel solutions must converge under boundary refinement."""

    coarse = _sample(n_boundary=24, n_query=128)
    fine = build_domain_sample(
        coarse.geometry,
        coarse.drive,
        n_boundary=96,
        query_preimages=coarse.query_preimages,
    )
    model = SolvedDoubleLayerPotential()
    coarse_error = _relative_l2(_prediction(model, coarse.domain), coarse.target)
    fine_error = _relative_l2(_prediction(model, fine.domain), fine.target)

    assert fine_error.item() < 0.01
    assert fine_error.item() < 0.15 * coarse_error.item()


@pytest.mark.parametrize(
    "model",
    [
        DirectDoubleLayerPotential(query_chunk_size=11),
        SolvedDoubleLayerPotential(query_chunk_size=11),
        LearnedDensityDoubleLayerPotential(
            n_iterations=5, query_chunk_size=11
        ).double(),
    ],
    ids=("direct", "solved", "learned-density"),
)
def test_controls_are_exactly_linear_in_boundary_drive(model) -> None:
    """Every double-layer control must preserve boundary-data superposition."""

    sample = _sample(n_boundary=31, n_query=23)
    boundary_values = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    first = torch.sin(torch.arange(31, dtype=torch.float64))
    second = boundary_values - 0.3 * first
    first_domain = _replace_boundary_values(sample.domain, first)
    second_domain = _replace_boundary_values(sample.domain, second)
    combined_domain = _replace_boundary_values(
        sample.domain, 1.7 * first - 0.4 * second
    )

    expected = 1.7 * _prediction(model, first_domain) - 0.4 * _prediction(
        model, second_domain
    )
    actual = _prediction(model, combined_domain)
    torch.testing.assert_close(actual, expected, rtol=3.0e-13, atol=3.0e-13)


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize(
    "model",
    [
        DirectDoubleLayerPotential(query_chunk_size=13),
        SolvedDoubleLayerPotential(query_chunk_size=13),
        LearnedDensityDoubleLayerPotential(
            n_iterations=4, query_chunk_size=13
        ).double(),
    ],
    ids=("direct", "solved", "learned-density"),
)
def test_controls_are_similarity_invariant(model, reflection: bool) -> None:
    """Layer controls must commute with all physical similarities."""

    sample = _sample(n_boundary=41, n_query=29)
    similarity = sample_similarity(
        6113,
        scale_range=(3.7, 3.7),
        translation_extent=11.0,
        reflection=reflection,
        dtype=torch.float64,
    )
    transformed = transform_sample(sample, similarity)

    torch.testing.assert_close(
        _prediction(model, transformed.domain),
        _prediction(model, sample.domain),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    "model",
    [
        DirectDoubleLayerPotential,
        SolvedDoubleLayerPotential,
        LearnedDensityDoubleLayerPotential,
    ],
)
def test_query_chunking_is_exact(model) -> None:
    """Query chunk size must affect memory only, up to roundoff."""

    sample = _sample(n_boundary=29, n_query=37)
    kwargs = {"n_iterations": 3} if model is LearnedDensityDoubleLayerPotential else {}
    small = model(query_chunk_size=3, **kwargs).double()
    large = model(query_chunk_size=1000, **kwargs).double()
    if model is LearnedDensityDoubleLayerPotential:
        large.load_state_dict(small.state_dict())

    torch.testing.assert_close(
        _prediction(small, sample.domain),
        _prediction(large, sample.domain),
        rtol=3.0e-15,
        atol=3.0e-15,
    )


def test_solved_double_layer_is_harmonic_away_from_boundary() -> None:
    """The analytic panel potential must have zero interior Laplacian."""

    sample = _sample(n_boundary=48, n_query=3)
    boundary = sample.domain.boundaries["dirichlet"]
    values = boundary.cell_data["boundary_value"]
    density = solve_double_layer_density(boundary, values).detach()
    queries = sample.domain.interior.points.detach().requires_grad_(True)
    potential = evaluate_double_layer(boundary, density, queries, query_chunk_size=2)

    laplacians: list[torch.Tensor] = []
    for query_index in range(queries.shape[0]):
        gradient = torch.autograd.grad(
            potential[query_index],
            queries,
            create_graph=True,
            retain_graph=True,
        )[0][query_index]
        laplacian = queries.new_zeros(())
        for axis in range(2):
            hessian_row = torch.autograd.grad(
                gradient[axis],
                queries,
                create_graph=True,
                retain_graph=True,
            )[0]
            laplacian = laplacian + hessian_row[query_index, axis]
        laplacians.append(laplacian)

    torch.testing.assert_close(
        torch.stack(laplacians),
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_learned_density_processor_has_trainable_collocation_objective() -> None:
    """Learned Richardson steps must reduce and differentiate trace loss."""

    sample = _sample(n_boundary=43, n_query=7)
    one_step = LearnedDensityDoubleLayerPotential(n_iterations=1).double()
    eight_steps = LearnedDensityDoubleLayerPotential(n_iterations=8).double()

    one_step_loss = one_step.collocation_loss(sample.domain)
    eight_step_loss = eight_steps.collocation_loss(sample.domain)
    assert eight_step_loss.item() < 1.0e-3 * one_step_loss.item()

    eight_step_loss.backward()
    assert eight_steps.relaxation.grad is not None
    assert torch.all(torch.isfinite(eight_steps.relaxation.grad))
    assert torch.all(eight_steps.relaxation.grad != 0.0)


def test_density_solver_accepts_multiple_right_hand_sides() -> None:
    """The dense solve must support batched boundary drives."""

    sample = _sample(n_boundary=27, n_query=5)
    boundary = sample.domain.boundaries["dirichlet"]
    values = boundary.cell_data["boundary_value"]
    right_hand_sides = torch.stack((values, 2.0 * values + 1.0), dim=-1)
    density = solve_double_layer_density(boundary, right_hand_sides)

    residual = double_layer_collocation_matrix(boundary) @ density - right_hand_sides
    assert residual.abs().max().item() < 1.0e-12


def test_encoded_control_omits_query_decoder_and_lifts_constants_exactly() -> None:
    """The encoded hybrid must omit dead parameters and preserve constants."""

    torch.manual_seed(6203)
    sample = _sample(n_boundary=29, n_query=17)
    constant = torch.full((29,), 1.7, dtype=torch.float64)
    domain = _replace_boundary_values(sample.domain, constant)
    model = EncodedDoubleLayerPotential(_tiny_encoder_config()).double()

    assert len(model.encoder.query_blocks) == 0
    assert model.encoder.output_projection is None
    assert not any("query_blocks" in name for name, _ in model.named_parameters())
    assert any(
        name.startswith("density_projection.") for name, _ in model.named_parameters()
    )
    assert all(parameter.requires_grad for parameter in model.parameters())
    torch.testing.assert_close(
        model.panel_density(domain), constant, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        _prediction(model, domain),
        torch.full((17,), 1.7, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.collocation_loss(domain),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_encoded_control_is_exactly_linear_in_boundary_drive() -> None:
    """The encoded density and potential must both obey superposition."""

    torch.manual_seed(6211)
    sample = _sample(n_boundary=27, n_query=19)
    model = EncodedDoubleLayerPotential(_tiny_encoder_config()).double()
    first = torch.cos(torch.arange(27, dtype=torch.float64))
    second = torch.sin(0.7 * torch.arange(27, dtype=torch.float64))
    first_domain = _replace_boundary_values(sample.domain, first)
    second_domain = _replace_boundary_values(sample.domain, second)
    combined_domain = _replace_boundary_values(
        sample.domain, -0.8 * first + 1.3 * second
    )

    expected = -0.8 * _prediction(model, first_domain) + 1.3 * _prediction(
        model, second_domain
    )
    actual = _prediction(model, combined_domain)
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)

    expected_density = -0.8 * model.panel_density(first_domain) + 1.3 * (
        model.panel_density(second_domain)
    )
    torch.testing.assert_close(
        model.panel_density(combined_domain),
        expected_density,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_encoded_control_is_similarity_invariant(reflection: bool) -> None:
    """Encoding, density projection, and decoding must commute with O(2)."""

    torch.manual_seed(6217)
    sample = _sample(n_boundary=31, n_query=21)
    transformed = transform_sample(
        sample,
        sample_similarity(
            6221,
            scale_range=(4.1, 4.1),
            translation_extent=9.0,
            reflection=reflection,
            dtype=torch.float64,
        ),
    )
    model = EncodedDoubleLayerPotential(_tiny_encoder_config()).double()

    torch.testing.assert_close(
        _prediction(model, transformed.domain),
        _prediction(model, sample.domain),
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    torch.testing.assert_close(
        model.collocation_residual(transformed.domain),
        model.collocation_residual(sample.domain),
        rtol=5.0e-12,
        atol=5.0e-12,
    )


@pytest.mark.parametrize("loss_kind", ["interior", "collocation"])
def test_encoded_control_losses_use_every_trainable_parameter(loss_kind: str) -> None:
    """Each training objective must reach every registered parameter."""

    torch.manual_seed(6229)
    sample = _sample(n_boundary=25, n_query=13)
    model = EncodedDoubleLayerPotential(_tiny_encoder_config()).double()

    if loss_kind == "interior":
        loss = (_prediction(model, sample.domain) - sample.target).square().mean()
    else:
        loss = model.collocation_loss(sample.domain)
    loss.backward()

    parameters = dict(model.named_parameters())
    assert parameters
    assert not any("query_blocks" in name for name in parameters)
    for name, parameter in parameters.items():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.all(torch.isfinite(parameter.grad)), name
        assert torch.any(parameter.grad != 0.0), name
