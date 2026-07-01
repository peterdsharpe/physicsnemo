# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Algebraic contracts for the typed mesh-transformer blocks."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import ScalarVectorState
from physicsnemo.experimental.nn.mesh_attention.block import (
    GeometryConditionedLinear,
    LinearMeshFieldBlock,
    MeshOperatorBlock,
    NonlinearZeroMeshFieldBlock,
)
from physicsnemo.mesh import Mesh


def _mesh(
    device: torch.device | str,
    *,
    transform: torch.Tensor | None = None,
) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, -0.2],
            [0.2, 1.1, 0.1],
            [-0.1, 0.3, 1.0],
            [1.0, 0.8, 0.7],
        ],
        device=device,
        dtype=torch.float64,
    )
    if transform is not None:
        points = torch.einsum("nd,ed->ne", points, transform)
    cells = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 3, 4]],
        device=device,
        dtype=torch.long,
    )
    return Mesh(points=points, cells=cells)


def _state(
    n: int,
    scalar_dim: int,
    vector_dim: int,
    device: torch.device | str,
    *,
    seed: int,
) -> ScalarVectorState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return ScalarVectorState(
        torch.randn(n, scalar_dim, generator=generator, dtype=torch.float64).to(device),
        torch.randn(n, vector_dim, 3, generator=generator, dtype=torch.float64).to(
            device
        ),
    )


def _combine(
    first: ScalarVectorState,
    second: ScalarVectorState,
    alpha: float,
    beta: float,
) -> ScalarVectorState:
    return ScalarVectorState(
        alpha * first.scalars + beta * second.scalars,
        alpha * first.vectors + beta * second.vectors,
    )


def _rotate(state: ScalarVectorState, transform: torch.Tensor) -> ScalarVectorState:
    return ScalarVectorState(
        state.scalars,
        torch.einsum("ncd,ed->nce", state.vectors, transform),
    )


def _orthogonal(device: torch.device | str, *, reflection: bool) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(341)
    transform, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform.to(device)


def _assert_close(
    actual: ScalarVectorState,
    expected: ScalarVectorState,
    *,
    exact: bool = False,
) -> None:
    tolerance = (
        {"rtol": 0.0, "atol": 0.0}
        if exact
        else {
            "rtol": 4.0e-11,
            "atol": 4.0e-11,
        }
    )
    torch.testing.assert_close(actual.scalars, expected.scalars, **tolerance)
    torch.testing.assert_close(actual.vectors, expected.vectors, **tolerance)


def _zeros_like(state: ScalarVectorState) -> ScalarVectorState:
    return ScalarVectorState(
        torch.zeros_like(state.scalars), torch.zeros_like(state.vectors)
    )


def _use_float64_moments(layer: torch.nn.Module) -> torch.nn.Module:
    """Keep property tests from silently accumulating double inputs in FP32."""
    layer.attention.accumulation_dtype = torch.float64
    return layer


def _randomize_all_parameters(layer: torch.nn.Module) -> torch.nn.Module:
    """Exercise contracts away from identity/zero-gate initializations."""
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.numel():
                parameter.uniform_(-0.35, 0.35)
    return layer


def test_geometry_conditioned_map_is_linear_in_field(device):
    layer = _randomize_all_parameters(
        GeometryConditionedLinear(3, 2, 2, 2, 4, 3).to(
            device=device, dtype=torch.float64
        )
    )
    geometry = _state(7, 3, 2, device, seed=1)
    first = _state(7, 2, 2, device, seed=2)
    second = _state(7, 2, 2, device, seed=3)
    alpha, beta = -1.3, 0.47

    actual = layer(geometry, _combine(first, second, alpha, beta))
    expected = _combine(layer(geometry, first), layer(geometry, second), alpha, beta)
    _assert_close(actual, expected)
    _assert_close(layer(geometry, _zeros_like(first)), _zeros_like(actual), exact=True)


def test_geometry_conditioned_map_supports_vector_only_fields(device):
    layer = GeometryConditionedLinear(3, 2, 0, 2, 3, 2).to(
        device=device, dtype=torch.float64
    )
    geometry = _state(7, 3, 2, device, seed=20)
    field = _state(7, 0, 2, device, seed=21)

    output = layer(geometry, field)

    assert output.scalars.shape == (7, 3)
    assert output.vectors.shape == (7, 2, 3)
    _assert_close(layer(geometry, _zeros_like(field)), _zeros_like(output), exact=True)


@pytest.mark.parametrize("reflection", [False, True])
def test_geometry_conditioned_map_is_o3_equivariant(device, reflection):
    layer = _randomize_all_parameters(
        GeometryConditionedLinear(3, 2, 2, 2, 4, 3).to(
            device=device, dtype=torch.float64
        )
    )
    geometry = _state(7, 3, 2, device, seed=4)
    field = _state(7, 2, 2, device, seed=5)
    transform = _orthogonal(device, reflection=reflection)

    output = layer(geometry, field)
    transformed = layer(_rotate(geometry, transform), _rotate(field, transform))
    _assert_close(transformed, _rotate(output, transform))


def test_linear_field_block_obeys_joint_cross_superposition(device):
    """Both source and receiver fields are linear arguments at fixed geometry."""
    layer = _randomize_all_parameters(
        _use_float64_moments(
            LinearMeshFieldBlock(3, 2, 2, 2, heads=2, scalar_rank=3, vector_rank=2).to(
                device=device, dtype=torch.float64
            )
        )
    )
    mesh = _mesh(device)
    source_geometry = _state(mesh.n_cells, 3, 2, device, seed=6)
    query_geometry = _state(6, 3, 2, device, seed=7)
    source_first = _state(mesh.n_cells, 2, 2, device, seed=8)
    source_second = _state(mesh.n_cells, 2, 2, device, seed=9)
    query_first = _state(6, 2, 2, device, seed=10)
    query_second = _state(6, 2, 2, device, seed=11)
    alpha, beta = 1.7, -0.23

    def apply(source: ScalarVectorState, query: ScalarVectorState) -> ScalarVectorState:
        return layer.cross(mesh, query_geometry, source_geometry, source, query)

    actual = apply(
        _combine(source_first, source_second, alpha, beta),
        _combine(query_first, query_second, alpha, beta),
    )
    expected = _combine(
        apply(source_first, query_first),
        apply(source_second, query_second),
        alpha,
        beta,
    )
    _assert_close(actual, expected)

    source_zero = _zeros_like(source_first)
    query_zero = _zeros_like(query_first)
    zero_output = apply(source_zero, query_zero)
    _assert_close(zero_output, _zeros_like(zero_output), exact=True)


def test_nonlinear_field_block_is_exactly_zero_preserving(device):
    layer = _randomize_all_parameters(
        _use_float64_moments(
            NonlinearZeroMeshFieldBlock(
                3, 2, 2, 2, heads=2, scalar_rank=3, vector_rank=2
            ).to(device=device, dtype=torch.float64)
        )
    )
    mesh = _mesh(device)
    source_geometry = _state(mesh.n_cells, 3, 2, device, seed=12)
    query_geometry = _state(6, 3, 2, device, seed=13)
    source_zero = ScalarVectorState.zeros(
        mesh.n_cells,
        2,
        2,
        3,
        device=device,
        dtype=torch.float64,
    )
    query_zero = ScalarVectorState.zeros(6, 2, 2, 3, device=device, dtype=torch.float64)

    cross_without_seed = layer.cross(mesh, query_geometry, source_geometry, source_zero)
    cross_with_zero_seed = layer.cross(
        mesh, query_geometry, source_geometry, source_zero, query_zero
    )
    self_output = layer(mesh, source_geometry, source_zero)

    _assert_close(cross_without_seed, _zeros_like(cross_without_seed), exact=True)
    _assert_close(cross_with_zero_seed, _zeros_like(cross_with_zero_seed), exact=True)
    _assert_close(self_output, _zeros_like(self_output), exact=True)


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("block_kind", ["operator", "linear", "nonlinear"])
def test_mesh_blocks_are_o3_equivariant(device, reflection, block_kind):
    mesh = _mesh(device)
    geometry = _state(mesh.n_cells, 3, 2, device, seed=14)
    field = _state(mesh.n_cells, 2, 2, device, seed=15)
    if block_kind == "operator":
        layer = _randomize_all_parameters(
            _use_float64_moments(
                MeshOperatorBlock(3, 2, heads=2, scalar_rank=3, vector_rank=2).to(
                    device=device, dtype=torch.float64
                )
            )
        )
        state = geometry
    elif block_kind == "linear":
        layer = _randomize_all_parameters(
            _use_float64_moments(
                LinearMeshFieldBlock(
                    3, 2, 2, 2, heads=2, scalar_rank=3, vector_rank=2
                ).to(device=device, dtype=torch.float64)
            )
        )
        state = field
    else:
        layer = _randomize_all_parameters(
            _use_float64_moments(
                NonlinearZeroMeshFieldBlock(
                    3, 2, 2, 2, heads=2, scalar_rank=3, vector_rank=2
                ).to(device=device, dtype=torch.float64)
            )
        )
        state = field

    if block_kind == "operator":
        output = layer(mesh, state)
    else:
        output = layer(mesh, geometry, state)

    transform = _orthogonal(device, reflection=reflection)
    transformed_mesh = _mesh(device, transform=transform)
    if block_kind == "operator":
        transformed = layer(transformed_mesh, _rotate(state, transform))
    else:
        transformed = layer(
            transformed_mesh,
            _rotate(geometry, transform),
            _rotate(state, transform),
        )
    _assert_close(transformed, _rotate(output, transform))


@pytest.mark.parametrize(
    "block_type",
    [LinearMeshFieldBlock, NonlinearZeroMeshFieldBlock],
)
def test_field_blocks_support_scalar_only_fields(device, block_type):
    """An empty vector tensor is a documented, first-class state shape."""
    mesh = _mesh(device)
    geometry = _state(mesh.n_cells, 3, 2, device, seed=16)
    field = _state(mesh.n_cells, 2, 0, device, seed=17)
    layer = _use_float64_moments(
        block_type(3, 2, 2, 0, heads=2, scalar_rank=3, vector_rank=2).to(
            device=device, dtype=torch.float64
        )
    )

    output = layer(mesh, geometry, field)

    assert output.scalars.shape == (mesh.n_cells, 2)
    assert output.vectors.shape == (mesh.n_cells, 0, 3)


@pytest.mark.parametrize(
    "block_type",
    [LinearMeshFieldBlock, NonlinearZeroMeshFieldBlock],
)
def test_field_blocks_support_fully_scalar_states(device, block_type):
    """Configured vector key ranks should vanish when no vector basis exists."""
    mesh = _mesh(device)
    geometry = _state(mesh.n_cells, 3, 0, device, seed=18)
    field = _state(mesh.n_cells, 2, 0, device, seed=19)
    layer = _use_float64_moments(
        block_type(3, 0, 2, 0, heads=2, scalar_rank=3, vector_rank=2).to(
            device=device, dtype=torch.float64
        )
    )

    output = layer(mesh, geometry, field)

    assert output.scalars.shape == (mesh.n_cells, 2)
    assert output.vectors.shape == (mesh.n_cells, 0, 3)
