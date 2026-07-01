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

"""Mathematical contracts for global, typed Galerkin mesh attention."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import (
    MeshAttention,
    ScalarVectorState,
)
from physicsnemo.mesh import Mesh


def _source_mesh(
    device: torch.device | str,
    dtype: torch.dtype,
    *,
    cell_permutation: torch.Tensor | None = None,
    orthogonal_transform: torch.Tensor | None = None,
) -> Mesh:
    """Return a small, nonuniform triangular source mesh embedded in 3D."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.1],
            [0.2, 1.1, -0.1],
            [0.1, 0.2, 0.9],
            [1.2, 0.7, 0.4],
        ],
        device=device,
        dtype=dtype,
    )
    cells = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ],
        device=device,
        dtype=torch.long,
    )
    if orthogonal_transform is not None:
        points = torch.einsum("nd,ed->ne", points, orthogonal_transform)
    if cell_permutation is not None:
        cells = cells[cell_permutation]
    return Mesh(points=points, cells=cells)


def _state(
    n_entities: int,
    scalar_channels: int,
    vector_channels: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int,
    requires_grad: bool = False,
) -> ScalarVectorState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scalars = torch.randn(
        n_entities, scalar_channels, generator=generator, dtype=dtype
    ).to(device)
    vectors = torch.randn(
        n_entities, vector_channels, 3, generator=generator, dtype=dtype
    ).to(device)
    if requires_grad:
        scalars.requires_grad_()
        vectors.requires_grad_()
    return ScalarVectorState(scalars, vectors)


def _layer(device: torch.device | str) -> MeshAttention:
    return MeshAttention(
        query_scalar_dim=3,
        query_vector_dim=2,
        key_scalar_dim=4,
        key_vector_dim=3,
        value_scalar_dim=2,
        value_vector_dim=2,
        out_scalar_dim=3,
        out_vector_dim=2,
        heads=2,
        scalar_rank=3,
        vector_rank=2,
        scalar_value_dim=2,
        vector_value_dim=2,
        accumulation_dtype=torch.float64,
    ).to(device=device, dtype=torch.float64)


def _rotate_state(
    state: ScalarVectorState, transform: torch.Tensor
) -> ScalarVectorState:
    return ScalarVectorState(
        state.scalars,
        torch.einsum("ncd,ed->nce", state.vectors, transform),
    )


def _orthogonal(device: torch.device | str, *, reflection: bool) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(123)
    transform, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform.to(device)


def test_factorized_matches_dense_forward_and_gradients(device):
    """Associative source moments must preserve both values and autograd."""
    torch.manual_seed(7)
    layer = _layer(device)
    mesh = _source_mesh(device, torch.float64)
    mesh.points.requires_grad_()
    query = _state(
        5, 3, 2, device=device, dtype=torch.float64, seed=1, requires_grad=True
    )
    key = _state(
        6, 4, 3, device=device, dtype=torch.float64, seed=2, requires_grad=True
    )
    value = _state(
        6, 2, 2, device=device, dtype=torch.float64, seed=3, requires_grad=True
    )
    leaves = (
        query.scalars,
        query.vectors,
        key.scalars,
        key.vectors,
        value.scalars,
        value.vectors,
        mesh.points,
        *tuple(layer.parameters()),
    )

    scalar_cotangent = torch.randn(5, 3, device=device, dtype=torch.float64)
    vector_cotangent = torch.randn(5, 2, 3, device=device, dtype=torch.float64)

    factorized = layer(mesh, query, key, value)
    factorized_loss = (factorized.scalars * scalar_cotangent).sum() + (
        factorized.vectors * vector_cotangent
    ).sum()
    # ``Mesh`` memoizes its differentiable cell measures, so retain their graph
    # while evaluating the second (dense) contraction against the same points.
    factorized_gradients = torch.autograd.grad(
        factorized_loss, leaves, retain_graph=True
    )

    dense = layer.forward_reference(mesh, query, key, value)
    dense_loss = (dense.scalars * scalar_cotangent).sum() + (
        dense.vectors * vector_cotangent
    ).sum()
    dense_gradients = torch.autograd.grad(dense_loss, leaves)

    torch.testing.assert_close(
        factorized.scalars, dense.scalars, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        factorized.vectors, dense.vectors, rtol=1e-12, atol=1e-12
    )
    for factorized_gradient, dense_gradient in zip(
        factorized_gradients, dense_gradients, strict=True
    ):
        torch.testing.assert_close(
            factorized_gradient, dense_gradient, rtol=2e-11, atol=2e-11
        )


@pytest.mark.parametrize("reflection", [False, True])
def test_o3_equivariance(device, reflection):
    """Scalars are invariant and polar-vector outputs co-transform under O(3)."""
    torch.manual_seed(8)
    layer = _layer(device)
    mesh = _source_mesh(device, torch.float64)
    query = _state(5, 3, 2, device=device, dtype=torch.float64, seed=4)
    key = _state(6, 4, 3, device=device, dtype=torch.float64, seed=5)
    value = _state(6, 2, 2, device=device, dtype=torch.float64, seed=6)
    output = layer(mesh, query, key, value)

    transform = _orthogonal(device, reflection=reflection)
    transformed_output = layer(
        _source_mesh(device, torch.float64, orthogonal_transform=transform),
        _rotate_state(query, transform),
        _rotate_state(key, transform),
        _rotate_state(value, transform),
    )

    torch.testing.assert_close(
        transformed_output.scalars, output.scalars, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        transformed_output.vectors,
        torch.einsum("ncd,ed->nce", output.vectors, transform),
        rtol=2e-11,
        atol=2e-11,
    )


def test_source_permutation_invariance_and_query_permutation_equivariance(device):
    """Cell ordering cannot affect the integral; receiver ordering only reorders it."""
    torch.manual_seed(9)
    layer = _layer(device)
    mesh = _source_mesh(device, torch.float64)
    query = _state(5, 3, 2, device=device, dtype=torch.float64, seed=7)
    key = _state(6, 4, 3, device=device, dtype=torch.float64, seed=8)
    value = _state(6, 2, 2, device=device, dtype=torch.float64, seed=9)
    output = layer(mesh, query, key, value)

    source_permutation = torch.tensor([4, 1, 5, 0, 3, 2], device=device)
    source_permuted = layer(
        _source_mesh(
            device,
            torch.float64,
            cell_permutation=source_permutation,
        ),
        query,
        key.slice(source_permutation),
        value.slice(source_permutation),
    )
    torch.testing.assert_close(
        source_permuted.scalars, output.scalars, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        source_permuted.vectors, output.vectors, rtol=1e-12, atol=1e-12
    )

    query_permutation = torch.tensor([2, 4, 0, 3, 1], device=device)
    query_permuted = layer(mesh, query.slice(query_permutation), key, value)
    torch.testing.assert_close(
        query_permuted.scalars,
        output.scalars[query_permutation],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        query_permuted.vectors,
        output.vectors[query_permutation],
        rtol=1e-12,
        atol=1e-12,
    )


def test_projection_chunking_preserves_the_operator_and_gradients(device):
    torch.manual_seed(25)
    layer = _layer(device)
    mesh = _source_mesh(device, torch.float64)
    mesh.points.requires_grad_()
    query = _state(
        5, 3, 2, device=device, dtype=torch.float64, seed=25, requires_grad=True
    )
    key = _state(
        6, 4, 3, device=device, dtype=torch.float64, seed=26, requires_grad=True
    )
    value = _state(
        6, 2, 2, device=device, dtype=torch.float64, seed=27, requires_grad=True
    )
    leaves = (
        query.scalars,
        query.vectors,
        key.scalars,
        key.vectors,
        value.scalars,
        value.vectors,
        mesh.points,
        *tuple(layer.parameters()),
    )
    scalar_cotangent = torch.randn(5, 3, device=device, dtype=torch.float64)
    vector_cotangent = torch.randn(5, 2, 3, device=device, dtype=torch.float64)

    layer.entity_chunk_size = None
    unchunked = layer(mesh, query, key, value)
    unchunked_loss = (unchunked.scalars * scalar_cotangent).sum() + (
        unchunked.vectors * vector_cotangent
    ).sum()
    layer.entity_chunk_size = 2
    chunked = layer(mesh, query, key, value)
    chunked_loss = (chunked.scalars * scalar_cotangent).sum() + (
        chunked.vectors * vector_cotangent
    ).sum()

    # Mesh memoizes differentiable cell measures, so retain their graph while
    # differentiating the second execution against the same point coordinates.
    unchunked_gradients = torch.autograd.grad(unchunked_loss, leaves, retain_graph=True)
    chunked_gradients = torch.autograd.grad(chunked_loss, leaves)

    torch.testing.assert_close(
        chunked.scalars, unchunked.scalars, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        chunked.vectors, unchunked.vectors, rtol=2e-12, atol=2e-12
    )
    for chunked_gradient, unchunked_gradient in zip(
        chunked_gradients, unchunked_gradients, strict=True
    ):
        torch.testing.assert_close(
            chunked_gradient, unchunked_gradient, rtol=2e-11, atol=2e-11
        )


def test_bias_free_value_path_obeys_superposition(device):
    """The declared linear mode is exactly linear in values with fixed Q/K."""
    torch.manual_seed(10)
    layer = MeshAttention(
        query_scalar_dim=3,
        query_vector_dim=2,
        key_scalar_dim=4,
        key_vector_dim=3,
        value_scalar_dim=2,
        value_vector_dim=2,
        out_scalar_dim=3,
        out_vector_dim=2,
        heads=2,
        scalar_rank=3,
        vector_rank=2,
        scalar_value_dim=2,
        vector_value_dim=2,
        value_scalar_bias=False,
        value_include_vector_invariants=False,
        output_scalar_bias=False,
        accumulation_dtype=torch.float64,
    ).to(device=device, dtype=torch.float64)
    mesh = _source_mesh(device, torch.float64)
    query = _state(5, 3, 2, device=device, dtype=torch.float64, seed=10)
    key = _state(6, 4, 3, device=device, dtype=torch.float64, seed=11)
    first = _state(6, 2, 2, device=device, dtype=torch.float64, seed=12)
    second = _state(6, 2, 2, device=device, dtype=torch.float64, seed=13)
    alpha, beta = -1.7, 0.35
    combined = ScalarVectorState(
        alpha * first.scalars + beta * second.scalars,
        alpha * first.vectors + beta * second.vectors,
    )

    first_output = layer(mesh, query, key, first)
    second_output = layer(mesh, query, key, second)
    combined_output = layer(mesh, query, key, combined)
    zero_output = layer(
        mesh,
        query,
        key,
        ScalarVectorState(
            torch.zeros_like(first.scalars), torch.zeros_like(first.vectors)
        ),
    )

    torch.testing.assert_close(
        combined_output.scalars,
        alpha * first_output.scalars + beta * second_output.scalars,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        combined_output.vectors,
        alpha * first_output.vectors + beta * second_output.vectors,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        zero_output.scalars, torch.zeros_like(zero_output.scalars), atol=0, rtol=0
    )
    torch.testing.assert_close(
        zero_output.vectors, torch.zeros_like(zero_output.vectors), atol=0, rtol=0
    )


def test_moment_reductions_keep_fp32_floor_under_autocast(device):
    """Ambient AMP must not silently lower the quadrature accumulation dtype."""
    layer = MeshAttention(
        query_scalar_dim=2,
        query_vector_dim=1,
        key_scalar_dim=2,
        key_vector_dim=1,
        value_scalar_dim=2,
        value_vector_dim=1,
        out_scalar_dim=2,
        out_vector_dim=1,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
        scalar_value_dim=2,
        vector_value_dim=1,
    ).to(device=device, dtype=torch.float32)
    mesh = _source_mesh(device, torch.float32)
    key = _state(6, 2, 1, device=device, dtype=torch.float32, seed=20)
    value = _state(6, 2, 1, device=device, dtype=torch.float32, seed=21)
    autocast_dtype = (
        torch.float16 if torch.device(device).type == "cuda" else torch.bfloat16
    )

    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=autocast_dtype,
        enabled=True,
    ):
        moments = layer.build_moments(mesh, key, value)

    for tensor in moments.__dict__.values():
        assert tensor.dtype == torch.float32

    # Cached moments may legitimately be reused after receivers move to a
    # higher working precision; evaluation promotes both operands explicitly.
    layer = layer.to(dtype=torch.float64)
    query = _state(4, 2, 1, device=device, dtype=torch.float64, seed=28)
    promoted = layer.evaluate_moments(query, moments)
    assert promoted.scalars.dtype == torch.float64
    assert promoted.vectors.dtype == torch.float64


def test_vector_only_attention_configuration(device):
    """Zero scalar ranks/channels are valid and do not create dummy features."""
    layer = MeshAttention(
        query_scalar_dim=0,
        query_vector_dim=1,
        key_scalar_dim=0,
        key_vector_dim=1,
        value_scalar_dim=0,
        value_vector_dim=1,
        out_scalar_dim=0,
        out_vector_dim=2,
        heads=1,
        scalar_rank=0,
        vector_rank=2,
        scalar_value_dim=0,
        vector_value_dim=2,
        accumulation_dtype=torch.float64,
    ).to(device=device, dtype=torch.float64)
    mesh = _source_mesh(device, torch.float64)
    query = _state(5, 0, 1, device=device, dtype=torch.float64, seed=22)
    key = _state(6, 0, 1, device=device, dtype=torch.float64, seed=23)
    value = _state(6, 0, 1, device=device, dtype=torch.float64, seed=24)

    factorized = layer(mesh, query, key, value)
    dense = layer.forward_reference(mesh, query, key, value)

    assert factorized.scalars.shape == (5, 0)
    assert factorized.vectors.shape == (5, 2, 3)
    torch.testing.assert_close(factorized.vectors, dense.vectors)


def _scalar_integral_layer(device: torch.device | str) -> MeshAttention:
    layer = MeshAttention(
        query_scalar_dim=1,
        query_vector_dim=0,
        key_scalar_dim=1,
        key_vector_dim=0,
        value_scalar_dim=1,
        value_vector_dim=0,
        out_scalar_dim=1,
        out_vector_dim=0,
        heads=1,
        scalar_rank=1,
        vector_rank=0,
        scalar_value_dim=1,
        vector_value_dim=0,
        qk_scalar_bias=False,
        value_scalar_bias=False,
        value_include_vector_invariants=False,
        output_scalar_bias=False,
        accumulation_dtype=torch.float64,
        entity_chunk_size=None,
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        layer.query_projection.scalar.weight.fill_(1.0)
        layer.key_projection.scalar.weight.fill_(1.0)
        layer.value_projection.scalar.weight.fill_(1.0)
        layer.scalar_output.weight.fill_(1.0)
    return layer


def _scalar_state(values: torch.Tensor, spatial_dim: int) -> ScalarVectorState:
    return ScalarVectorState(
        values[:, None],
        values.new_empty(values.shape[0], 0, spatial_dim),
    )


def test_controlled_kernel_has_dense_source_to_query_jacobian(device):
    r"""Every source atom has an explicit path to every receiver.

    With one scalar rank and every projection set to the identity, the layer is

    .. math::

        y_i = q_i \sum_j w_j k_j v_j,

    so its value Jacobian is the dense matrix
    :math:`\partial y_i/\partial v_j = q_i k_j w_j`.  This guards the global
    information-flow contract independently of the factorized and dense code
    paths, which could otherwise acquire the same accidental mask.
    """
    layer = _scalar_integral_layer(device)
    mesh = _source_mesh(device, torch.float64)
    queries = torch.tensor([1.25, -0.4, 2.1], device=device, dtype=torch.float64)
    keys = torch.tensor(
        [0.7, -1.1, 0.3, 1.4, -0.8, 0.55],
        device=device,
        dtype=torch.float64,
    )
    values = torch.tensor(
        [-0.2, 0.9, 1.7, -0.6, 0.1, 1.2],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    output = layer(
        mesh,
        _scalar_state(queries, 3),
        _scalar_state(keys, 3),
        _scalar_state(values, 3),
    ).scalars[:, 0]
    jacobian = torch.stack(
        [
            torch.autograd.grad(output[index], values, retain_graph=True)[0]
            for index in range(output.shape[0])
        ]
    )
    expected = queries[:, None] * keys[None, :] * mesh.cell_areas[None, :]

    torch.testing.assert_close(jacobian, expected, rtol=2.0e-14, atol=2.0e-14)
    assert torch.count_nonzero(jacobian).item() == jacobian.numel()


def test_conserved_measure_splitting_is_exact(device):
    """Identical quadrature atoms may be split without changing the operator."""
    layer = _scalar_integral_layer(device)
    dtype = torch.float64
    original = Mesh(
        points=torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=device, dtype=dtype),
        cells=torch.tensor([[0, 1]], device=device),
    )
    split = Mesh(
        points=torch.tensor(
            [[-0.5, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.5, 0.0]],
            device=device,
            dtype=dtype,
        ),
        cells=torch.tensor([[0, 1], [2, 3]], device=device),
    )
    query = _scalar_state(torch.tensor([1.0, -0.4], device=device, dtype=dtype), 2)
    original_state = _scalar_state(torch.ones(1, device=device, dtype=dtype), 2)
    split_state = _scalar_state(torch.ones(2, device=device, dtype=dtype), 2)

    original_output = layer(original, query, original_state, original_state).scalars
    split_output = layer(split, query, split_state, split_state).scalars

    torch.testing.assert_close(split_output, original_output, rtol=0, atol=0)


def test_smooth_boundary_quadrature_converges_under_refinement(device):
    """A fixed smooth kernel/value pair converges on genuine polygon remeshes."""
    layer = _scalar_integral_layer(device)
    dtype = torch.float64
    query = _scalar_state(torch.ones(1, device=device, dtype=dtype), 2)
    errors: list[float] = []

    for n_cells in (16, 32, 64):
        angles = torch.arange(n_cells, device=device, dtype=dtype)
        angles = angles * (2.0 * torch.pi / n_cells)
        points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        indices = torch.arange(n_cells, device=device)
        cells = torch.stack((indices, torch.roll(indices, -1)), dim=-1)
        mesh = Mesh(points=points, cells=cells)
        keys = _scalar_state(torch.ones(n_cells, device=device, dtype=dtype), 2)
        values = _scalar_state(mesh.cell_centroids[:, 0].square(), 2)

        result = layer(mesh, query, keys, values).scalars.squeeze()
        errors.append(abs(result.item() - torch.pi))

    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.01
