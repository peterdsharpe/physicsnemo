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

"""Contracts for the 2D pseudoscalar (``0o``) type-system extension.

Covers, mirroring the ablation-knob discipline of the earlier polynomial and
scalar-only knobs: reflection equivariance of every new typed product (wedge,
rotation, pseudo pair, scalar-pseudo), the pseudo sectors of TypedProjection /
GeometryConditionedLinear / the field blocks / MeshAttention, the model-level
declaration surface (the ``"0o"`` rank token and its validation), an
end-to-end model reflection test, forward/backward/row-stability with pseudo
channels, and the mandatory bitwise-default regression: with no pseudoscalar
declarations and ``drive_pseudo_dim=0`` nothing may change.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import (
    MeshAttention,
    ScalarVectorState,
    TypedProjection,
    _pair_wedges,
    _pseudo_pair_invariants,
    _vector_perp,
    _wedge_invariants,
)
from physicsnemo.experimental.nn.mesh_attention.block import (
    GeometryConditionedLinear,
    LinearMeshFieldBlock,
    NonlinearZeroMeshFieldBlock,
)
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _orthogonal_2d(
    seed: int,
    *,
    reflection: bool,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    transform, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=dtype))
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform


def _state(
    n: int,
    scalar_dim: int,
    vector_dim: int,
    pseudo_dim: int,
    device: torch.device | str,
    *,
    seed: int,
) -> ScalarVectorState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return ScalarVectorState(
        torch.randn(n, scalar_dim, generator=generator, dtype=torch.float64).to(device),
        torch.randn(n, vector_dim, 2, generator=generator, dtype=torch.float64).to(
            device
        ),
        torch.randn(n, pseudo_dim, generator=generator, dtype=torch.float64).to(device),
    )


def _transform_state(
    state: ScalarVectorState, transform: torch.Tensor
) -> ScalarVectorState:
    """Apply an O(2) map: scalars fixed, vectors rotated, pseudos times det."""
    det = torch.linalg.det(transform)
    return ScalarVectorState(
        state.scalars,
        torch.einsum("ncd,ed->nce", state.vectors, transform),
        state.pseudos * det,
    )


def _assert_state_close(
    actual: ScalarVectorState,
    expected: ScalarVectorState,
    *,
    exact: bool = False,
) -> None:
    tolerance = (
        {"rtol": 0.0, "atol": 0.0} if exact else {"rtol": 4.0e-11, "atol": 4.0e-11}
    )
    torch.testing.assert_close(actual.scalars, expected.scalars, **tolerance)
    torch.testing.assert_close(actual.vectors, expected.vectors, **tolerance)
    torch.testing.assert_close(actual.pseudos, expected.pseudos, **tolerance)


def _zeros_like(state: ScalarVectorState) -> ScalarVectorState:
    return ScalarVectorState(
        torch.zeros_like(state.scalars),
        torch.zeros_like(state.vectors),
        torch.zeros_like(state.pseudos),
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
        alpha * first.pseudos + beta * second.pseudos,
    )


def _randomize_all_parameters(layer: torch.nn.Module) -> torch.nn.Module:
    """Exercise contracts away from identity/zero-gate initializations."""
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.numel():
                parameter.uniform_(-0.35, 0.35)
    return layer


def _segment_mesh(
    device: torch.device | str,
    *,
    transform: torch.Tensor | None = None,
) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.1, 0.2],
            [0.9, 1.3],
            [-0.2, 1.0],
            [0.4, 0.5],
        ],
        device=device,
        dtype=torch.float64,
    )
    if transform is not None:
        points = torch.einsum("nd,ed->ne", points, transform.to(device))
    cells = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4]], device=device, dtype=torch.long
    )
    return Mesh(points=points, cells=cells)


def _circle_boundary(
    n_cells: int,
    device: torch.device | str,
    *,
    transform: torch.Tensor | None = None,
    reverse_orientation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-circle points and clockwise cells, hence outward normals."""
    angles = 2.0 * torch.pi * torch.arange(n_cells, device=device, dtype=torch.float64)
    angles = angles / n_cells
    points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    if transform is not None:
        points = torch.einsum("nd,ed->ne", points, transform.to(device))
    indices = torch.arange(n_cells, device=device)
    cells = torch.stack((torch.roll(indices, -1), indices), dim=-1)
    if reverse_orientation:
        # An orientation-reversing coordinate change must also reverse
        # segment winding for oriented cell normals to represent the same
        # polar normal field.
        cells = cells[:, [1, 0]]
    return points, cells


def _pseudo_domain(
    device: torch.device | str,
    *,
    n_boundary: int = 16,
    transform: torch.Tensor | None = None,
    reflection: bool = False,
) -> DomainMesh:
    """A 2D domain with scalar+pseudoscalar boundary drives and global data.

    Under an orthogonal ``transform`` with determinant ``det``: the true
    scalar ``forcing`` is unchanged, the pseudoscalar ``sheet_strength`` and
    ``circulation`` are multiplied by ``det``, and the polar ``freestream``
    rotates with the frame.
    """
    det = -1.0 if reflection else 1.0
    points, cells = _circle_boundary(
        n_boundary, device, transform=transform, reverse_orientation=reflection
    )
    generator = torch.Generator(device="cpu").manual_seed(97)
    forcing = torch.randn(n_boundary, generator=generator, dtype=torch.float64)
    sheet_strength = torch.randn(n_boundary, generator=generator, dtype=torch.float64)
    boundary = Mesh(
        points=points,
        cells=cells,
        cell_data={
            "forcing": forcing.to(device),
            "sheet_strength": det * sheet_strength.to(device),
        },
    )
    query_points = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4], [-0.4, 0.2]],
        device=device,
        dtype=torch.float64,
    )
    freestream = torch.tensor([0.8, -0.5], device=device, dtype=torch.float64)
    if transform is not None:
        query_points = torch.einsum("nd,ed->ne", query_points, transform.to(device))
        freestream = torch.einsum("d,ed->e", freestream, transform.to(device))
    return DomainMesh(
        interior=Mesh(points=query_points),
        boundaries={"wall": boundary},
        global_data={
            "circulation": torch.tensor(det * 0.7, device=device, dtype=torch.float64),
            "freestream": freestream,
        },
    )


def _pseudo_model(
    device: torch.device | str,
    *,
    seed: int = 331,
    **overrides,
) -> MeshTransformer:
    """A small 2D model exercising pseudo drives, outputs, and channels."""
    torch.manual_seed(seed)
    kwargs = dict(
        n_spatial_dims=2,
        output_field_ranks={"pressure": 0, "velocity": 1, "swirl": "0o"},
        boundary_field_ranks={
            "wall": {"drive": {"forcing": 0, "sheet_strength": "0o"}}
        },
        global_field_ranks={
            "drive": {"circulation": "0o", "freestream": 1},
        },
        field_mode="linear",
        query_decoder="kernel",
        operator_scalar_dim=6,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        drive_pseudo_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=1,
        heads=2,
        scalar_rank=3,
        vector_rank=2,
    )
    kwargs.update(overrides)
    model = MeshTransformer(**kwargs).to(device=device, dtype=torch.float64).eval()
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    return model


# ---------------------------------------------------------------------------
# The four new typed products, under random rotations and reflections.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,tolerance",
    [
        (torch.float32, {"rtol": 1.0e-5, "atol": 1.0e-6}),
        (torch.float64, {"rtol": 1.0e-12, "atol": 1.0e-12}),
    ],
    ids=["fp32", "fp64"],
)
@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_planar_products_transform_correctly(
    device, dtype, tolerance, reflection, seed
):
    r"""The closed 2D product set over {0e, 0o, 1o} transforms correctly.

    Under an orthogonal map ``R`` with ``det R = sigma``: ``v -> R v``,
    ``s -> s``, ``p -> sigma p``.  Then the wedge ``v ^ w`` is 0o
    (``-> sigma (v ^ w)``), the pseudo pair ``p q`` is 0e (invariant),
    ``s p`` is 0o, and the rotation product ``p v_perp`` is a polar vector
    (``-> R (p v_perp)``) because ``v_perp`` itself is axial
    (``-> sigma R v_perp``).  ``sigma`` is the numerically computed
    determinant of the QR-sampled map (``+-1`` up to roundoff), hence the
    relative tolerance component.
    """
    transform = _orthogonal_2d(seed, reflection=reflection, dtype=dtype).to(device)
    sigma = torch.linalg.det(transform)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    v = torch.randn(7, 3, 2, generator=generator, dtype=dtype).to(device)
    w = torch.randn(7, 4, 2, generator=generator, dtype=dtype).to(device)
    p = torch.randn(7, 3, generator=generator, dtype=dtype).to(device)
    s = torch.randn(7, 2, generator=generator, dtype=dtype).to(device)

    def rotate(vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncd,ed->nce", vectors, transform)

    # Wedge: 1o x 1o -> 0o.
    torch.testing.assert_close(
        _pair_wedges(rotate(v), rotate(w)),
        sigma * _pair_wedges(v, w),
        **tolerance,
    )
    torch.testing.assert_close(
        _wedge_invariants(rotate(v)),
        sigma * _wedge_invariants(v),
        **tolerance,
    )
    # Pseudo pairs: 0o x 0o -> 0e (invariant: sigma^2 = 1).
    torch.testing.assert_close(
        _pseudo_pair_invariants(sigma * p),
        _pseudo_pair_invariants(p),
        **tolerance,
    )
    # 0e x 0o -> 0o (part of the closed set; exact up to sigma roundoff).
    torch.testing.assert_close(
        s[:, :, None] * (sigma * p)[:, None, :],
        sigma * (s[:, :, None] * p[:, None, :]),
        **tolerance,
    )
    # The perpendicular is axial: perp(R v) = sigma R perp(v).
    torch.testing.assert_close(
        _vector_perp(rotate(v)),
        sigma * rotate(_vector_perp(v)),
        **tolerance,
    )
    # Rotation product: 0o x 1o -> 1o (polar).
    product = p[:, :, None, None] * _vector_perp(v)[:, None, :, :]
    transformed_product = (sigma * p)[:, :, None, None] * _vector_perp(rotate(v))[
        :, None, :, :
    ]
    torch.testing.assert_close(
        transformed_product,
        torch.einsum("nfcd,ed->nfce", product, transform),
        **tolerance,
    )


def test_planar_products_reject_non_planar_inputs():
    """The 0o sector is 2D only; 3D inputs fail loudly, not silently."""
    vectors = torch.randn(4, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="requires 2D"):
        _wedge_invariants(vectors)
    with pytest.raises(ValueError, match="axial vector"):
        _vector_perp(vectors)
    with pytest.raises(ValueError, match="requires 2D"):
        _pair_wedges(vectors, vectors)


# ---------------------------------------------------------------------------
# TypedProjection pseudo sector.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reflection", [False, True])
def test_typed_projection_pseudo_is_o2_equivariant(device, reflection):
    layer = _randomize_all_parameters(
        TypedProjection(3, 3, 4, 2, scalar_bias=True, pseudo_in=2, pseudo_out=3).to(
            device=device, dtype=torch.float64
        )
    )
    state = _state(7, 3, 3, 2, device, seed=21)
    transform = _orthogonal_2d(341, reflection=reflection).to(device)

    output = layer(state)
    transformed = layer(_transform_state(state, transform))
    _assert_state_close(transformed, _transform_state(output, transform))


def test_typed_projection_pseudo_validation():
    """Pseudo outputs need a basis; widths are validated; no silent drops."""
    with pytest.raises(ValueError, match="pseudo or vector-pair"):
        TypedProjection(2, 0, 2, 0, scalar_bias=True, pseudo_in=0, pseudo_out=1)
    with pytest.raises(ValueError, match="pseudo or vector-pair"):
        TypedProjection(
            2,
            3,
            2,
            0,
            scalar_bias=True,
            include_vector_invariants=False,
            pseudo_in=0,
            pseudo_out=1,
        )
    # A single vector channel has no wedge pair either.
    with pytest.raises(ValueError, match="pseudo or vector-pair"):
        TypedProjection(2, 1, 2, 0, scalar_bias=True, pseudo_in=0, pseudo_out=1)
    layer = TypedProjection(2, 2, 2, 1, scalar_bias=True, pseudo_in=2, pseudo_out=1)
    with pytest.raises(ValueError, match="pseudoscalar channels"):
        layer(_state(3, 2, 2, 1, "cpu", seed=5))


def test_typed_projection_without_invariants_is_linear_in_pseudos(device):
    """The quadratic lifts are all gated by ``include_vector_invariants``."""
    layer = _randomize_all_parameters(
        TypedProjection(
            2,
            2,
            3,
            2,
            scalar_bias=False,
            include_vector_invariants=False,
            pseudo_in=2,
            pseudo_out=2,
        ).to(device=device, dtype=torch.float64)
    )
    first = _state(6, 2, 2, 2, device, seed=31)
    second = _state(6, 2, 2, 2, device, seed=32)
    alpha, beta = 1.7, -0.31
    actual = layer(_combine(first, second, alpha, beta))
    expected = _combine(layer(first), layer(second), alpha, beta)
    _assert_state_close(actual, expected)
    _assert_state_close(layer(_zeros_like(first)), _zeros_like(actual), exact=True)


# ---------------------------------------------------------------------------
# GeometryConditionedLinear pseudo branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reflection", [False, True])
def test_geometry_conditioned_pseudo_is_o2_equivariant(device, reflection):
    layer = _randomize_all_parameters(
        GeometryConditionedLinear(
            3, 3, 2, 2, 4, 3, field_pseudo_dim=2, out_pseudo_dim=2
        ).to(device=device, dtype=torch.float64)
    )
    geometry = _state(7, 3, 3, 0, device, seed=41)
    field = _state(7, 2, 2, 2, device, seed=42)
    transform = _orthogonal_2d(342, reflection=reflection).to(device)

    output = layer(geometry, field)
    transformed = layer(
        _transform_state(geometry, transform), _transform_state(field, transform)
    )
    _assert_state_close(transformed, _transform_state(output, transform))


def test_geometry_conditioned_pseudo_is_linear_in_field(device):
    layer = _randomize_all_parameters(
        GeometryConditionedLinear(
            3, 3, 2, 2, 4, 3, field_pseudo_dim=2, out_pseudo_dim=2
        ).to(device=device, dtype=torch.float64)
    )
    geometry = _state(7, 3, 3, 0, device, seed=43)
    first = _state(7, 2, 2, 2, device, seed=44)
    second = _state(7, 2, 2, 2, device, seed=45)
    alpha, beta = -1.3, 0.47

    actual = layer(geometry, _combine(first, second, alpha, beta))
    expected = _combine(layer(geometry, first), layer(geometry, second), alpha, beta)
    _assert_state_close(actual, expected)
    _assert_state_close(
        layer(geometry, _zeros_like(first)), _zeros_like(actual), exact=True
    )


def test_geometry_conditioned_pseudo_validation(device):
    """The geometry stream is parity-even; pseudo outputs need a basis."""
    layer = GeometryConditionedLinear(
        3, 3, 2, 2, 4, 3, field_pseudo_dim=2, out_pseudo_dim=2
    ).to(device=device, dtype=torch.float64)
    geometry_with_pseudos = _state(7, 3, 3, 1, device, seed=46)
    field = _state(7, 2, 2, 2, device, seed=47)
    with pytest.raises(ValueError, match="no pseudoscalar sector"):
        layer(geometry_with_pseudos, field)
    # No pseudo basis at all: no field pseudos, no geometry vectors.
    with pytest.raises(ValueError, match="pseudoscalar output requires"):
        GeometryConditionedLinear(3, 0, 2, 0, 4, 0, out_pseudo_dim=2)
    # The rotation product makes a vector reachable from pseudos alone.
    GeometryConditionedLinear(3, 3, 0, 0, 0, 3, field_pseudo_dim=2)


# ---------------------------------------------------------------------------
# Field blocks with pseudo channels.
# ---------------------------------------------------------------------------


def _use_float64_moments(layer: torch.nn.Module) -> torch.nn.Module:
    layer.attention.accumulation_dtype = torch.float64
    return layer


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize(
    "block_type", [LinearMeshFieldBlock, NonlinearZeroMeshFieldBlock]
)
def test_field_blocks_with_pseudo_are_o2_equivariant(device, reflection, block_type):
    layer = _randomize_all_parameters(
        _use_float64_moments(
            block_type(
                3,
                2,
                2,
                2,
                heads=2,
                scalar_rank=3,
                vector_rank=2,
                field_pseudo_dim=2,
            ).to(device=device, dtype=torch.float64)
        )
    )
    mesh = _segment_mesh(device)
    geometry = _state(mesh.n_cells, 3, 2, 0, device, seed=51)
    field = _state(mesh.n_cells, 2, 2, 2, device, seed=52)

    output = layer(mesh, geometry, field)

    transform = _orthogonal_2d(343, reflection=reflection).to(device)
    transformed = layer(
        _segment_mesh(device, transform=transform),
        _transform_state(geometry, transform),
        _transform_state(field, transform),
    )
    _assert_state_close(transformed, _transform_state(output, transform))


def test_linear_block_with_pseudo_obeys_superposition(device):
    layer = _randomize_all_parameters(
        _use_float64_moments(
            LinearMeshFieldBlock(
                3,
                2,
                2,
                2,
                heads=2,
                scalar_rank=3,
                vector_rank=2,
                field_pseudo_dim=2,
            ).to(device=device, dtype=torch.float64)
        )
    )
    mesh = _segment_mesh(device)
    geometry = _state(mesh.n_cells, 3, 2, 0, device, seed=53)
    first = _state(mesh.n_cells, 2, 2, 2, device, seed=54)
    second = _state(mesh.n_cells, 2, 2, 2, device, seed=55)
    alpha, beta = 1.7, -0.23

    actual = layer(mesh, geometry, _combine(first, second, alpha, beta))
    expected = _combine(
        layer(mesh, geometry, first), layer(mesh, geometry, second), alpha, beta
    )
    _assert_state_close(actual, expected)


def test_nonlinear_block_with_pseudo_is_exactly_zero_preserving(device):
    layer = _randomize_all_parameters(
        _use_float64_moments(
            NonlinearZeroMeshFieldBlock(
                3,
                2,
                2,
                2,
                heads=2,
                scalar_rank=3,
                vector_rank=2,
                field_pseudo_dim=2,
            ).to(device=device, dtype=torch.float64)
        )
    )
    mesh = _segment_mesh(device)
    geometry = _state(mesh.n_cells, 3, 2, 0, device, seed=56)
    zero_field = ScalarVectorState.zeros(
        mesh.n_cells, 2, 2, 2, pseudo_channels=2, device=device, dtype=torch.float64
    )
    output = layer(mesh, geometry, zero_field)
    _assert_state_close(output, _zeros_like(output), exact=True)


# ---------------------------------------------------------------------------
# MeshAttention pseudo values.
# ---------------------------------------------------------------------------


def _pseudo_attention(device: torch.device | str) -> MeshAttention:
    torch.manual_seed(701)
    return (
        MeshAttention(
            query_scalar_dim=3,
            query_vector_dim=2,
            key_scalar_dim=3,
            key_vector_dim=2,
            value_scalar_dim=2,
            value_vector_dim=2,
            out_scalar_dim=2,
            out_vector_dim=2,
            heads=2,
            scalar_rank=3,
            vector_rank=2,
            scalar_value_dim=2,
            vector_value_dim=2,
            query_pseudo_dim=2,
            key_pseudo_dim=2,
            value_pseudo_dim=2,
            out_pseudo_dim=2,
            pseudo_value_dim=2,
            accumulation_dtype=torch.float64,
            entity_chunk_size=3,
        )
        .to(device=device, dtype=torch.float64)
        .eval()
    )


def test_pseudo_attention_factorized_matches_dense(device):
    """The joint invariant-value moment machinery matches the dense oracle."""
    attention = _pseudo_attention(device)
    mesh = _segment_mesh(device)
    query = _state(6, 3, 2, 2, device, seed=61)
    key = _state(mesh.n_cells, 3, 2, 2, device, seed=62)
    value = _state(mesh.n_cells, 2, 2, 2, device, seed=63)

    factorized = attention(mesh, query, key, value)
    dense = attention.forward_reference(mesh, query, key, value)
    _assert_state_close(factorized, dense)
    assert factorized.pseudos.shape == (6, 2)
    assert factorized.pseudos.abs().sum() > 0


@pytest.mark.parametrize("reflection", [False, True])
def test_pseudo_attention_is_o2_equivariant(device, reflection):
    attention = _pseudo_attention(device)
    mesh = _segment_mesh(device)
    query = _state(6, 3, 2, 2, device, seed=64)
    key = _state(mesh.n_cells, 3, 2, 2, device, seed=65)
    value = _state(mesh.n_cells, 2, 2, 2, device, seed=66)

    output = attention(mesh, query, key, value)
    transform = _orthogonal_2d(344, reflection=reflection).to(device)
    transformed = attention(
        _segment_mesh(device, transform=transform),
        _transform_state(query, transform),
        _transform_state(key, transform),
        _transform_state(value, transform),
    )
    _assert_state_close(transformed, _transform_state(output, transform))


# ---------------------------------------------------------------------------
# Model declaration surface: the "0o" rank token and its validation.
# ---------------------------------------------------------------------------


def test_pseudo_declarations_are_validated():
    base = dict(
        output_field_ranks={"pressure": 0},
        boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
        query_decoder="moment",
    )
    # 2D only: in 3D the analogous object is the axial vector (out of scope).
    with pytest.raises(ValueError, match="axial vector"):
        MeshTransformer(
            n_spatial_dims=3,
            **{
                **base,
                "output_field_ranks": {"pressure": 0, "swirl": "0o"},
                "drive_pseudo_dim": 2,
            },
        )
    with pytest.raises(ValueError, match="axial vector"):
        MeshTransformer(n_spatial_dims=3, **base, drive_pseudo_dim=2)
    # The operator stream is parity-even: no pseudoscalar operator fields.
    with pytest.raises(ValueError, match="parity-even"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={
                "wall": {"operator": {"chirality": "0o"}, "drive": {"forcing": 0}}
            },
            drive_pseudo_dim=2,
        )
    with pytest.raises(ValueError, match="parity-even"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
            global_field_ranks={"operator": {"chirality": "0o"}},
            drive_pseudo_dim=2,
        )
    # Declared pseudo fields require the channel knob to be on.
    with pytest.raises(ValueError, match="drive_pseudo_dim > 0"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0, "swirl": "0o"},
            boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
        )
    with pytest.raises(ValueError, match="drive_pseudo_dim > 0"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
            global_field_ranks={"drive": {"circulation": "0o"}},
        )
    # Scalar-only mode has no wedge/rotation read-out paths.
    with pytest.raises(ValueError, match="scalar-only"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
            operator_vector_dim=0,
            drive_vector_dim=0,
            vector_rank=0,
            drive_pseudo_dim=2,
        )
    # Unknown string tokens name the full token menu.
    with pytest.raises(ValueError, match="0, 1, or the pseudoscalar token"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={"wall": {"drive": {"forcing": "1e"}}},
        )
    # A legal pseudo-everything configuration constructs.
    MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"swirl": "0o"},
        boundary_field_ranks={"wall": {"drive": {"sheet_strength": "0o"}}},
        global_field_ranks={"drive": {"circulation": "0o"}},
        drive_pseudo_dim=2,
    )


# ---------------------------------------------------------------------------
# Bitwise-default regression: the knob off must change nothing.
# ---------------------------------------------------------------------------


def test_pseudo_knob_default_is_bitwise_noop(device):
    """Explicitly passing ``drive_pseudo_dim=0`` must not change anything.

    Mirrors ``test_polynomial_member_knob_default_is_bitwise_noop`` and
    ``test_scalar_only_knob_default_is_bitwise_noop``: same seed gives the
    same parameter tensors in the same order and bitwise identical outputs.
    Additionally, the default state dict must contain no pseudo-sector
    parameters at all -- the structural half of the guarantee that existing
    models are bitwise identical to the pre-extension code (no new
    parameters, no extra RNG draws, no changed operations at width zero).
    """

    def build(**overrides) -> MeshTransformer:
        torch.manual_seed(83)
        model = (
            MeshTransformer(
                n_spatial_dims=2,
                output_field_ranks={"potential": 0},
                boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
                field_mode="linear",
                query_decoder="kernel",
                operator_scalar_dim=7,
                operator_vector_dim=3,
                drive_scalar_dim=9,
                drive_vector_dim=3,
                operator_layers=1,
                drive_layers=1,
                heads=2,
                scalar_rank=4,
                vector_rank=2,
                **overrides,
            )
            .to(device=device, dtype=torch.float64)
            .eval()
        )
        for module in model.modules():
            if hasattr(module, "accumulation_dtype"):
                module.accumulation_dtype = torch.float64
        return model

    reference = build()
    explicit = build(drive_pseudo_dim=0)

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    assert not any("pseudo" in name for name in reference_state)
    for name, expected in reference_state.items():
        torch.testing.assert_close(explicit_state[name], expected, rtol=0.0, atol=0.0)

    points, cells = _circle_boundary(16, device)
    generator = torch.Generator(device="cpu").manual_seed(84)
    forcing = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = DomainMesh(
        interior=Mesh(
            points=torch.tensor(
                [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]],
                device=device,
                dtype=torch.float64,
            )
        ),
        boundaries={
            "wall": Mesh(points=points, cells=cells, cell_data={"forcing": forcing})
        },
    )
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# End-to-end model contracts with pseudo channels.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
@pytest.mark.parametrize("reflection", [False, True])
def test_pseudo_model_is_o2_equivariant_with_consistent_orientation(
    device, reflection, query_decoder
):
    """Scalar outputs are invariant, vectors rotate, pseudos flip.

    The mirrored problem carries mirrored geometry (with reversed segment
    winding so normals stay consistent), a rotated freestream, and
    sign-flipped pseudoscalar drives (circulation and sheet strength); the
    pseudoscalar output must flip sign with them.
    """
    model = _pseudo_model(device, query_decoder=query_decoder)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.2, 0.2)
    transform = _orthogonal_2d(345, reflection=reflection).to(device)
    det = -1.0 if reflection else 1.0

    with torch.no_grad():
        original = model(_pseudo_domain(device))
        transformed = model(
            _pseudo_domain(device, transform=transform, reflection=reflection)
        )

    tolerance = {"rtol": 3.0e-10, "atol": 3.0e-11}
    torch.testing.assert_close(
        transformed.point_data["pressure"],
        original.point_data["pressure"],
        **tolerance,
    )
    torch.testing.assert_close(
        transformed.point_data["velocity"],
        torch.einsum("nd,ed->ne", original.point_data["velocity"], transform),
        **tolerance,
    )
    torch.testing.assert_close(
        transformed.point_data["swirl"],
        det * original.point_data["swirl"],
        **tolerance,
    )
    assert original.point_data["swirl"].abs().sum() > 0


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_pseudo_model_forward_backward_and_drive_liveness(device, query_decoder):
    """Every parameter trains and the pseudoscalar drive is live.

    The gradient of the loss with respect to the circulation input must be
    nonzero: the whole point of the sector is that a pseudoscalar drive has
    equivariant read-out paths instead of being provably dead.
    """
    model = _pseudo_model(device, query_decoder=query_decoder)
    domain = _pseudo_domain(device)
    domain.global_data["circulation"].requires_grad_()

    output = model(domain)
    loss = (
        output.point_data["pressure"].square().sum()
        + output.point_data["velocity"].square().sum()
        + output.point_data["swirl"].square().sum()
    )
    loss.backward()

    assert domain.global_data["circulation"].grad is not None
    assert torch.count_nonzero(domain.global_data["circulation"].grad)
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
    pseudo_parameters = [
        name for name, _ in model.named_parameters() if "pseudo" in name
    ]
    assert pseudo_parameters, "pseudo sector created no parameters"


def test_pseudo_model_kernel_row_stability_is_bitwise(device):
    """Query-subset independence stays bitwise with pseudo channels."""
    model = _pseudo_model(device)
    domain = _pseudo_domain(device)
    queries = domain.interior.points
    subset = torch.tensor([2, 0, 3], device=device)

    with torch.no_grad():
        encoded = model.encode(domain)
        message_full = model.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model.kernel_decoder(queries[subset], encoded.kernel_cache)
        full = model.decode(encoded)
        partial = model.decode(encoded, Mesh(points=queries[subset]))

    assert encoded.kernel_cache.value_pseudos is not None
    assert encoded.kernel_cache.value_pseudos.shape[-1] > 0
    for sector in ("scalars", "vectors", "pseudos"):
        torch.testing.assert_close(
            getattr(message_subset, sector),
            getattr(message_full, sector)[subset],
            rtol=0.0,
            atol=0.0,
        )
    for key in ("pressure", "velocity", "swirl"):
        torch.testing.assert_close(
            partial.point_data[key],
            full.point_data[key][subset],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_pseudo_nonlinear_mode_is_exactly_zero_preserving(device, query_decoder):
    """Zero drive (including zero pseudos) produces exactly zero output."""
    model = _pseudo_model(
        device, query_decoder=query_decoder, field_mode="zero_preserving_nonlinear"
    )
    domain = _pseudo_domain(device)
    zero_domain = DomainMesh(
        interior=domain.interior,
        boundaries={
            "wall": domain.boundaries["wall"].with_data(
                cell_data={
                    "forcing": torch.zeros_like(
                        domain.boundaries["wall"].cell_data["forcing"]
                    ),
                    "sheet_strength": torch.zeros_like(
                        domain.boundaries["wall"].cell_data["sheet_strength"]
                    ),
                }
            )
        },
        global_data={
            "circulation": torch.zeros_like(domain.global_data["circulation"]),
            "freestream": torch.zeros_like(domain.global_data["freestream"]),
        },
    )
    with torch.no_grad():
        output = model(zero_domain)
    for key in ("pressure", "velocity", "swirl"):
        prediction = output.point_data[key]
        torch.testing.assert_close(
            prediction, torch.zeros_like(prediction), rtol=0.0, atol=0.0
        )


def test_pseudo_model_linear_mode_superposition_includes_pseudos(device):
    """The fixed-geometry superposition law extends to pseudoscalar drives."""
    model = _pseudo_model(device)
    alpha, beta = 1.3, -0.6

    def scaled_domain(scale: float) -> DomainMesh:
        base = _pseudo_domain(device)
        wall = base.boundaries["wall"]
        return DomainMesh(
            interior=base.interior,
            boundaries={
                "wall": wall.with_data(
                    cell_data={
                        key: scale * value for key, value in wall.cell_data.items()
                    }
                )
            },
            global_data={key: scale * value for key, value in base.global_data.items()},
        )

    with torch.no_grad():
        one = model(scaled_domain(1.0))
        combined = model(scaled_domain(alpha + beta))
    for key in ("pressure", "velocity", "swirl"):
        torch.testing.assert_close(
            combined.point_data[key],
            (alpha + beta) * one.point_data[key],
            rtol=4.0e-11,
            atol=4.0e-11,
        )
