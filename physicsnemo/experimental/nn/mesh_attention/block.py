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

r"""Equivariant residual blocks for global boundary-to-boundary operators.

The geometry and physical-field streams are intentionally separate.  Geometry
blocks are nonlinear.  Linear field blocks are exactly linear in their field
argument at fixed geometry; nonlinear field blocks are structurally
zero-preserving.  Keeping these as different Python classes makes it difficult
for a future normalization, activation, or bias to silently invalidate the
linear-mode contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from physicsnemo.mesh import Mesh

from .attention import (
    AttentionMoments,
    MeshAttention,
    ScalarVectorState,
    TypedProjection,
    _gram_invariants,
)


def _add(left: ScalarVectorState, right: ScalarVectorState) -> ScalarVectorState:
    """Add two identically typed states with an informative shape failure."""
    if left.scalars.shape != right.scalars.shape:
        raise ValueError(
            "Scalar residual shapes differ: "
            f"{tuple(left.scalars.shape)} != {tuple(right.scalars.shape)}"
        )
    if left.vectors.shape != right.vectors.shape:
        raise ValueError(
            "Vector residual shapes differ: "
            f"{tuple(left.vectors.shape)} != {tuple(right.vectors.shape)}"
        )
    return ScalarVectorState(left.scalars + right.scalars, left.vectors + right.vectors)


class TypedRMSNorm(nn.Module):
    r"""O(D)-equivariant normalization for a nonlinear geometry state.

    Scalars receive ordinary RMS normalization.  All vector channels at one
    entity share an invariant RMS denominator, so Cartesian components are
    never mixed and rotations/reflections commute with this operation.

    This class is only used on the geometry stream.  It must not be inserted
    in :class:`LinearMeshFieldBlock`, where normalization by field amplitude
    would violate superposition.
    """

    def __init__(self, scalar_dim: int, vector_dim: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scalar_weight = nn.Parameter(torch.ones(scalar_dim))
        self.vector_weight = nn.Parameter(torch.ones(vector_dim))

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        if state.scalars.shape[-1]:
            scalar_rms = state.scalars.square().mean(dim=-1, keepdim=True)
            scalars = state.scalars * torch.rsqrt(scalar_rms + self.eps)
            scalars = scalars * self.scalar_weight
        else:
            scalars = state.scalars

        if state.vectors.shape[1]:
            vector_rms = state.vectors.square().mean(dim=(1, 2), keepdim=True)
            vectors = state.vectors * torch.rsqrt(vector_rms + self.eps)
            vectors = vectors * self.vector_weight[None, :, None]
        else:
            vectors = state.vectors
        return ScalarVectorState(scalars, vectors.to(dtype=scalars.dtype))


class StateLayerScale(nn.Module):
    """Learned per-channel residual scales that preserve tensor type."""

    def __init__(self, scalar_dim: int, vector_dim: int, init: float = 1.0e-2) -> None:
        super().__init__()
        self.scalar_scale = nn.Parameter(torch.full((scalar_dim,), init))
        self.vector_scale = nn.Parameter(torch.full((vector_dim,), init))

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        return ScalarVectorState(
            state.scalars * self.scalar_scale,
            state.vectors * self.vector_scale[None, :, None],
        )


class GeometryFeedForward(nn.Module):
    """A small nonlinear O(D)-equivariant pointwise geometry network."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        hidden_scalar = max(hidden_ratio * scalar_dim, 1)
        hidden_vector = max(hidden_ratio * vector_dim, vector_dim)
        self.input = TypedProjection(
            scalar_dim,
            vector_dim,
            hidden_scalar,
            hidden_vector,
            scalar_bias=True,
        )
        self.vector_gate = (
            nn.Linear(hidden_scalar, hidden_vector) if hidden_vector else None
        )
        self.output = TypedProjection(
            hidden_scalar,
            hidden_vector,
            scalar_dim,
            vector_dim,
            scalar_bias=True,
        )

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        hidden = self.input(state)
        scalars = torch.nn.functional.silu(hidden.scalars)
        if hidden.vectors.shape[1]:
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for non-empty vector state")
            gates = torch.sigmoid(self.vector_gate(scalars))
            vectors = hidden.vectors * gates[:, :, None]
        else:
            vectors = hidden.vectors
        return self.output(ScalarVectorState(scalars, vectors))


class GeometryConditionedLinear(nn.Module):
    r"""An O(D)-equivariant map linear in ``field`` at fixed ``geometry``.

    Direct scalar and vector channel mixing is augmented by the two elementary
    geometry-mediated type changes: vector--geometry dot products create
    scalars, and scalar coefficients multiplying geometry vectors create
    vectors.  Invariant geometry gates modulate the result but never read the
    field, so the complete map remains exactly linear in ``field``.

    In Clebsch-Gordan terms the five branches realize ``0e -> 0e`` and
    ``1o -> 1o`` Schur channel mixing, ``1o x 1o -> 0e``
    (``scalar_from_vector_dots``), ``0e x 1o -> 1o`` (``vector_from_scalar``),
    and ``(1o x 1o -> 0e) x 1o -> 1o`` (``vector_from_vector_dots``).  For
    ``l <= 1`` inputs and outputs under O(D) this set is complete: the
    remaining formal paths are parity-forbidden (Levi-Civita) or reduce to
    these by isotropic-tensor identities.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        out_scalar_dim: int,
        out_vector_dim: int,
    ) -> None:
        super().__init__()
        if out_vector_dim and not (
            field_vector_dim or (field_scalar_dim and geometry_vector_dim)
        ):
            raise ValueError(
                "A vector output requires a field-vector or geometry-vector basis"
            )
        self.geometry_vector_dim = geometry_vector_dim
        self.geometry_scalar_dim = geometry_scalar_dim
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim

        geometry_invariants = (
            geometry_scalar_dim + geometry_vector_dim * (geometry_vector_dim + 1) // 2
        )
        self.scalar_gate = (
            nn.Linear(geometry_invariants, out_scalar_dim) if out_scalar_dim else None
        )
        self.vector_gate = (
            nn.Linear(geometry_invariants, out_vector_dim) if out_vector_dim else None
        )
        if self.scalar_gate is not None:
            nn.init.zeros_(self.scalar_gate.weight)
            nn.init.zeros_(self.scalar_gate.bias)
        if self.vector_gate is not None:
            nn.init.zeros_(self.vector_gate.weight)
            nn.init.zeros_(self.vector_gate.bias)

        self.scalar_from_scalar = (
            nn.Linear(field_scalar_dim, out_scalar_dim, bias=False)
            if field_scalar_dim and out_scalar_dim
            else None
        )
        if out_scalar_dim and field_vector_dim and geometry_vector_dim:
            self.scalar_from_vector_dots = nn.Linear(
                field_vector_dim * geometry_vector_dim,
                out_scalar_dim,
                bias=False,
            )
        else:
            self.scalar_from_vector_dots = None

        if out_vector_dim and field_vector_dim:
            self.vector_from_vector = nn.Parameter(
                torch.randn(out_vector_dim, field_vector_dim)
                / math.sqrt(field_vector_dim)
            )
        else:
            self.register_parameter("vector_from_vector", None)

        if out_vector_dim and field_vector_dim and geometry_vector_dim:
            self.vector_from_vector_dots = nn.Linear(
                field_vector_dim * geometry_vector_dim,
                out_vector_dim * geometry_vector_dim,
                bias=False,
            )
        else:
            self.vector_from_vector_dots = None

        if out_vector_dim and field_scalar_dim and geometry_vector_dim:
            self.vector_from_scalar = nn.Linear(
                field_scalar_dim,
                out_vector_dim * geometry_vector_dim,
                bias=False,
            )
        else:
            self.vector_from_scalar = None

    @staticmethod
    def _geometry_invariants(geometry: ScalarVectorState) -> torch.Tensor:
        return torch.cat((geometry.scalars, _gram_invariants(geometry.vectors)), dim=-1)

    def forward(
        self,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        if geometry.n_entities != field.n_entities:
            raise ValueError("geometry and field entity counts must match")
        if geometry.n_spatial_dims != field.n_spatial_dims:
            raise ValueError("geometry and field spatial dimensions must match")
        if geometry.scalars.shape[1] != self.geometry_scalar_dim:
            raise ValueError("geometry has the wrong number of scalar channels")
        if geometry.vectors.shape[1] != self.geometry_vector_dim:
            raise ValueError("geometry has the wrong number of vector channels")
        if field.scalars.shape[1] != self.field_scalar_dim:
            raise ValueError("field has the wrong number of scalar channels")
        if field.vectors.shape[1] != self.field_vector_dim:
            raise ValueError("field has the wrong number of vector channels")
        invariants = self._geometry_invariants(geometry)

        scalar_terms: list[torch.Tensor] = []
        if self.scalar_from_scalar is not None:
            scalar_terms.append(self.scalar_from_scalar(field.scalars))
        if self.scalar_from_vector_dots is not None:
            dots = torch.einsum(
                "nfd,ngd->nfg", field.vectors, geometry.vectors
            ).flatten(1)
            scalar_terms.append(self.scalar_from_vector_dots(dots))
        if self.out_scalar_dim:
            if self.scalar_gate is None:
                raise RuntimeError("scalar gate missing for scalar output")
            scalar_gates = 2.0 * torch.sigmoid(self.scalar_gate(invariants))
            if scalar_terms:
                scalars = scalar_terms[0]
                for term in scalar_terms[1:]:
                    scalars = scalars + term
            else:
                scalars = torch.zeros_like(scalar_gates)
            # 2 sigmoid(0) = 1: conditioning begins as a neutral
            # multiplicative gate while retaining an unrestricted,
            # geometry-dependent derivative.
            scalars = scalars * scalar_gates
        else:
            scalars = None

        if self.out_vector_dim:
            vector_terms: list[torch.Tensor] = []
            if self.vector_from_vector is not None:
                vector_terms.append(
                    torch.einsum("of,nfd->nod", self.vector_from_vector, field.vectors)
                )
            if self.vector_from_vector_dots is not None:
                dots = torch.einsum(
                    "nfd,ngd->nfg", field.vectors, geometry.vectors
                ).flatten(1)
                coefficients = self.vector_from_vector_dots(dots).reshape(
                    field.n_entities,
                    self.out_vector_dim,
                    self.geometry_vector_dim,
                )
                vector_terms.append(
                    torch.einsum("nog,ngd->nod", coefficients, geometry.vectors)
                )
            if self.vector_from_scalar is not None:
                coefficients = self.vector_from_scalar(field.scalars).reshape(
                    field.n_entities,
                    self.out_vector_dim,
                    self.geometry_vector_dim,
                )
                vector_terms.append(
                    torch.einsum("nog,ngd->nod", coefficients, geometry.vectors)
                )
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for vector output")
            vector_gates = 2.0 * torch.sigmoid(self.vector_gate(invariants))
            if vector_terms:
                vectors = vector_terms[0]
                for term in vector_terms[1:]:
                    vectors = vectors + term
            else:
                vectors = vector_gates.new_zeros(
                    field.n_entities,
                    self.out_vector_dim,
                    field.n_spatial_dims,
                )
            vectors = vectors * vector_gates[:, :, None]
        else:
            vectors = field.vectors.new_empty(field.n_entities, 0, field.n_spatial_dims)
        if scalars is None:
            scalars = vectors.new_empty(field.n_entities, 0)
        else:
            vectors = vectors.to(dtype=scalars.dtype)
        return ScalarVectorState(scalars, vectors)


class ZeroPreservingFeedForward(nn.Module):
    """Nonlinear equivariant update whose output is exactly zero at zero field."""

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
    ) -> None:
        super().__init__()
        self.lift = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
        )
        invariant_dim = (
            geometry_scalar_dim
            + geometry_vector_dim * (geometry_vector_dim + 1) // 2
            + field_scalar_dim
            + field_vector_dim * (field_vector_dim + 1) // 2
        )
        self.scalar_gate = nn.Linear(invariant_dim, field_scalar_dim)
        self.vector_gate = (
            nn.Linear(invariant_dim, field_vector_dim) if field_vector_dim else None
        )
        self.project = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
        )

    def forward(
        self,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        hidden = self.lift(geometry, field)
        invariants = torch.cat(
            (
                geometry.scalars,
                _gram_invariants(geometry.vectors),
                hidden.scalars,
                _gram_invariants(hidden.vectors),
            ),
            dim=-1,
        )
        scalars = hidden.scalars * torch.sigmoid(self.scalar_gate(invariants))
        vectors = hidden.vectors
        if vectors.shape[1]:
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for non-empty vector state")
            vectors = vectors * torch.sigmoid(self.vector_gate(invariants))[:, :, None]
        return self.project(geometry, ScalarVectorState(scalars, vectors))


class MeshOperatorBlock(nn.Module):
    """Nonlinear global self-interaction block for operator geometry."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        hidden_ratio: int = 2,
        layer_scale: float = 1.0e-2,
        entity_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        self.attention_norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.attention = MeshAttention(
            query_scalar_dim=scalar_dim,
            query_vector_dim=vector_dim,
            key_scalar_dim=scalar_dim,
            key_vector_dim=vector_dim,
            value_scalar_dim=scalar_dim,
            value_vector_dim=vector_dim,
            out_scalar_dim=scalar_dim,
            out_vector_dim=vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank if vector_dim else 0,
            scalar_value_dim=max(scalar_dim // heads, 1),
            vector_value_dim=max(vector_dim // heads, 1) if vector_dim else 0,
            value_scalar_bias=True,
            output_scalar_bias=True,
            entity_chunk_size=entity_chunk_size,
        )
        self.attention_scale = StateLayerScale(scalar_dim, vector_dim, init=layer_scale)
        self.feed_forward_norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.feed_forward = GeometryFeedForward(
            scalar_dim, vector_dim, hidden_ratio=hidden_ratio
        )
        self.feed_forward_scale = StateLayerScale(
            scalar_dim, vector_dim, init=layer_scale
        )

    def forward(self, source_mesh: Mesh, state: ScalarVectorState) -> ScalarVectorState:
        normalized = self.attention_norm(state)
        state = _add(
            state,
            self.attention_scale(
                self.attention(source_mesh, normalized, normalized, normalized)
            ),
        )
        return _add(
            state,
            self.feed_forward_scale(self.feed_forward(self.feed_forward_norm(state))),
        )


class PointwiseGeometryBlock(nn.Module):
    """Nonlinear typed feature map with no interaction or spatial cutoff."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        hidden_ratio: int = 2,
        layer_scale: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.feed_forward = GeometryFeedForward(
            scalar_dim, vector_dim, hidden_ratio=hidden_ratio
        )
        self.scale = StateLayerScale(scalar_dim, vector_dim, init=layer_scale)

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        return _add(
            state,
            self.scale(self.feed_forward(self.norm(state))),
        )


class LinearMeshFieldBlock(nn.Module):
    r"""Global field block with an exact fixed-geometry superposition law."""

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        layer_scale: float = 1.0e-2,
        message_layer_scale: float | None = None,
        entity_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.attention = MeshAttention(
            query_scalar_dim=geometry_scalar_dim,
            query_vector_dim=geometry_vector_dim,
            key_scalar_dim=geometry_scalar_dim,
            key_vector_dim=geometry_vector_dim,
            value_scalar_dim=field_scalar_dim,
            value_vector_dim=field_vector_dim,
            out_scalar_dim=field_scalar_dim,
            out_vector_dim=field_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank if geometry_vector_dim else 0,
            scalar_value_dim=max(field_scalar_dim // heads, 1),
            vector_value_dim=(
                max(field_vector_dim // heads, 1) if field_vector_dim else 0
            ),
            value_scalar_bias=False,
            value_include_vector_invariants=False,
            output_scalar_bias=False,
            entity_chunk_size=entity_chunk_size,
        )
        self.pointwise = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
        )
        self.message_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=(layer_scale if message_layer_scale is None else message_layer_scale),
        )
        self.pointwise_scale = StateLayerScale(
            field_scalar_dim, field_vector_dim, init=layer_scale
        )

    def build_source_moments(
        self,
        source_mesh: Mesh,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
    ) -> AttentionMoments:
        """Compress one global source integral for reuse by many queries."""
        return self.attention.build_moments(source_mesh, source_geometry, source_field)

    def evaluate_cross(
        self,
        query_geometry: ScalarVectorState,
        moments: AttentionMoments,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        """Evaluate precomputed source moments at independent receivers."""
        message = self.message_scale(
            self.attention.evaluate_moments(query_geometry, moments)
        )
        state = message if query_field is None else _add(query_field, message)
        return _add(
            state,
            self.pointwise_scale(self.pointwise(query_geometry, state)),
        )

    def cross(
        self,
        source_mesh: Mesh,
        query_geometry: ScalarVectorState,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        return self.evaluate_cross(
            query_geometry,
            self.build_source_moments(source_mesh, source_geometry, source_field),
            query_field,
        )

    def forward(
        self,
        source_mesh: Mesh,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        return self.cross(source_mesh, geometry, geometry, field, field)


class NonlinearZeroMeshFieldBlock(nn.Module):
    r"""Global content-dependent field block with exact zero preservation."""

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        layer_scale: float = 1.0e-2,
        message_layer_scale: float | None = None,
        entity_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        combined_scalar = geometry_scalar_dim + field_scalar_dim
        combined_vector = geometry_vector_dim + field_vector_dim
        self.attention = MeshAttention(
            query_scalar_dim=combined_scalar,
            query_vector_dim=combined_vector,
            key_scalar_dim=combined_scalar,
            key_vector_dim=combined_vector,
            value_scalar_dim=field_scalar_dim,
            value_vector_dim=field_vector_dim,
            out_scalar_dim=field_scalar_dim,
            out_vector_dim=field_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=(vector_rank if geometry_vector_dim + field_vector_dim else 0),
            scalar_value_dim=max(field_scalar_dim // heads, 1),
            vector_value_dim=(
                max(field_vector_dim // heads, 1) if field_vector_dim else 0
            ),
            value_scalar_bias=False,
            value_include_vector_invariants=True,
            output_scalar_bias=False,
            entity_chunk_size=entity_chunk_size,
        )
        self.pointwise = ZeroPreservingFeedForward(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
        )
        self.message_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=(layer_scale if message_layer_scale is None else message_layer_scale),
        )
        self.pointwise_scale = StateLayerScale(
            field_scalar_dim, field_vector_dim, init=layer_scale
        )

    def build_source_moments(
        self,
        source_mesh: Mesh,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
    ) -> AttentionMoments:
        """Compress content-dependent source keys and values for query reuse."""
        return self.attention.build_moments(
            source_mesh,
            source_geometry.cat(source_field),
            source_field,
        )

    def evaluate_cross(
        self,
        query_geometry: ScalarVectorState,
        moments: AttentionMoments,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        """Evaluate cached source moments while preserving the zero solution."""
        if query_field is None:
            query_field = ScalarVectorState.zeros(
                query_geometry.n_entities,
                self.field_scalar_dim,
                self.field_vector_dim,
                query_geometry.n_spatial_dims,
                device=query_geometry.scalars.device,
                dtype=query_geometry.scalars.dtype,
            )
            residual = False
        else:
            residual = True
        message = self.message_scale(
            self.attention.evaluate_moments(query_geometry.cat(query_field), moments)
        )
        state = _add(query_field, message) if residual else message
        return _add(
            state,
            self.pointwise_scale(self.pointwise(query_geometry, state)),
        )

    def cross(
        self,
        source_mesh: Mesh,
        query_geometry: ScalarVectorState,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        return self.evaluate_cross(
            query_geometry,
            self.build_source_moments(source_mesh, source_geometry, source_field),
            query_field,
        )

    def forward(
        self,
        source_mesh: Mesh,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        return self.cross(source_mesh, geometry, geometry, field, field)


__all__ = [
    "GeometryConditionedLinear",
    "LinearMeshFieldBlock",
    "MeshOperatorBlock",
    "NonlinearZeroMeshFieldBlock",
    "PointwiseGeometryBlock",
    "TypedRMSNorm",
    "ZeroPreservingFeedForward",
]
