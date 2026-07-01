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

r"""Global, rank-typed Galerkin attention on a boundary mesh.

The mathematical operator in this module is deliberately small.  A source
``Mesh`` supplies the quadrature measure and scalar/vector key and value
features supply a finite-rank kernel.  Source moments are formed once and may
then be evaluated at any number of receivers without coupling the receivers to
one another.

No spatial neighbourhood, radial cutoff, softmax, or tree is part of the
operator.  Hierarchical acceleration for a future non-separable kernel belongs
behind a separate numerical backend and must converge to a dense oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.integration import (
    _integrate_weighted_moment,
    integrate_moment,
)


@dataclass(frozen=True)
class ScalarVectorState:
    r"""A packed collection of invariant scalars and polar vectors.

    ``scalars`` has shape ``(N, C_s)`` and ``vectors`` has shape
    ``(N, C_v, D)``.  Zero vector channels are represented by an empty tensor,
    not ``None``; this keeps compiled call signatures stable.
    """

    scalars: Float[torch.Tensor, "n scalar_channels"]
    vectors: Float[torch.Tensor, "n vector_channels spatial_dims"]

    @property
    def n_entities(self) -> int:
        return self.scalars.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        return self.vectors.shape[-1]

    def validate(self, *, label: str = "state") -> None:
        if self.scalars.ndim != 2:
            raise ValueError(
                f"{label}.scalars must have shape (N, C), got "
                f"{tuple(self.scalars.shape)}"
            )
        if self.vectors.ndim != 3:
            raise ValueError(
                f"{label}.vectors must have shape (N, C, D), got "
                f"{tuple(self.vectors.shape)}"
            )
        if self.scalars.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"{label} scalar/vector entity counts differ: "
                f"{self.scalars.shape[0]} != {self.vectors.shape[0]}"
            )
        if self.scalars.device != self.vectors.device:
            raise ValueError(f"{label} scalar/vector devices differ")
        if self.scalars.dtype != self.vectors.dtype:
            raise ValueError(f"{label} scalar/vector dtypes differ")

    @classmethod
    def zeros(
        cls,
        n_entities: int,
        scalar_channels: int,
        vector_channels: int,
        n_spatial_dims: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "ScalarVectorState":
        return cls(
            scalars=torch.zeros(
                n_entities, scalar_channels, device=device, dtype=dtype
            ),
            vectors=torch.zeros(
                n_entities,
                vector_channels,
                n_spatial_dims,
                device=device,
                dtype=dtype,
            ),
        )

    def cat(self, other: "ScalarVectorState") -> "ScalarVectorState":
        if self.n_entities != other.n_entities:
            raise ValueError("Cannot concatenate states with different entity counts")
        if self.n_spatial_dims != other.n_spatial_dims:
            raise ValueError("Cannot concatenate states with different spatial dims")
        return ScalarVectorState(
            torch.cat((self.scalars, other.scalars), dim=-1),
            torch.cat((self.vectors, other.vectors), dim=1),
        )

    def slice(self, item: slice | torch.Tensor) -> "ScalarVectorState":
        return ScalarVectorState(self.scalars[item], self.vectors[item])


@dataclass(frozen=True)
class TypedQK:
    scalars: torch.Tensor  # (N, H, R_s)
    vectors: torch.Tensor  # (N, H, R_v, D)


@dataclass(frozen=True)
class TypedValues:
    scalars: torch.Tensor  # (N, H, F_s)
    vectors: torch.Tensor  # (N, H, F_v, D)


@dataclass(frozen=True)
class AttentionMoments:
    r"""Quadrature-integrated source moments for typed attention."""

    scalar_key_scalar_value: torch.Tensor  # (H, R_s, F_s)
    vector_key_scalar_value: torch.Tensor  # (H, R_v, D, F_s)
    scalar_key_vector_value: torch.Tensor  # (H, R_s, F_v, D)
    vector_key_vector_value: torch.Tensor  # (H, R_v, D, F_v, D)


def _gram_invariants(vectors: torch.Tensor) -> torch.Tensor:
    """Return the upper triangle of each per-entity vector Gram matrix."""
    n, channels, _ = vectors.shape
    if channels == 0:
        return vectors.new_empty(n, 0)
    gram = torch.einsum("ncd,ned->nce", vectors, vectors)
    rows, cols = torch.triu_indices(channels, channels, device=vectors.device)
    return gram[:, rows, cols]


class TypedProjection(nn.Module):
    r"""Project scalar/vector state without mixing Cartesian components."""

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        *,
        scalar_bias: bool,
        include_vector_invariants: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_in = scalar_in
        self.vector_in = vector_in
        self.scalar_out = scalar_out
        self.vector_out = vector_out
        self.include_vector_invariants = include_vector_invariants
        n_invariants = (
            vector_in * (vector_in + 1) // 2 if include_vector_invariants else 0
        )
        self.scalar = (
            nn.Linear(scalar_in + n_invariants, scalar_out, bias=scalar_bias)
            if scalar_out
            else None
        )
        if vector_out and not vector_in:
            raise ValueError(
                "TypedProjection cannot create a vector without an input vector basis"
            )
        if vector_out:
            self.vector_weight = nn.Parameter(
                torch.randn(vector_out, vector_in) / math.sqrt(max(vector_in, 1))
            )
        else:
            self.register_parameter("vector_weight", None)

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        if self.scalar is not None:
            scalar_input = state.scalars
            if self.include_vector_invariants:
                scalar_input = torch.cat(
                    (scalar_input, _gram_invariants(state.vectors)), dim=-1
                )
            scalars = self.scalar(scalar_input)
        else:
            scalars = None
        if self.vector_out:
            vectors = torch.einsum("oc,ncd->nod", self.vector_weight, state.vectors)
        else:
            vectors = state.vectors.new_empty(state.n_entities, 0, state.n_spatial_dims)
        if scalars is None:
            scalars = vectors.new_empty(state.n_entities, 0)
        else:
            vectors = vectors.to(dtype=scalars.dtype)
        return ScalarVectorState(scalars, vectors)


class MeshAttention(nn.Module):
    r"""Exact global Galerkin attention for scalar and polar-vector fields.

    Queries and keys may contain rank-0 and rank-1 channels.  The invariant
    pair coefficient is

    .. math::

        a_{ijh}=q^0_{ih}\cdot k^0_{jh}
        +\sum_r q^1_{ihr}\cdot k^1_{jhr}.

    Values retain their scalar/vector type.  Associativity evaluates the dense
    quadrature sum without constructing an ``N_target x N_source`` matrix.
    ``entity_chunk_size`` bounds live projection workspace in inference.  With
    autograd enabled, PyTorch retains each chunk's saved activations for the
    backward pass, so total saved activation memory remains linear in entity
    count rather than being bounded by one chunk.
    """

    def __init__(
        self,
        *,
        query_scalar_dim: int,
        query_vector_dim: int,
        key_scalar_dim: int,
        key_vector_dim: int,
        value_scalar_dim: int,
        value_vector_dim: int,
        out_scalar_dim: int,
        out_vector_dim: int,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        scalar_value_dim: int = 8,
        vector_value_dim: int = 4,
        qk_scalar_bias: bool = True,
        value_scalar_bias: bool = False,
        value_include_vector_invariants: bool = True,
        output_scalar_bias: bool = False,
        accumulation_dtype: torch.dtype | None = torch.float32,
        entity_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if scalar_rank < 0 or vector_rank < 0:
            raise ValueError("attention ranks must be non-negative")
        if scalar_rank + vector_rank == 0:
            raise ValueError("at least one scalar or vector key rank is required")
        if scalar_value_dim < 0 or vector_value_dim < 0:
            raise ValueError("value dimensions must be non-negative")
        if entity_chunk_size is not None and (
            isinstance(entity_chunk_size, bool)
            or not isinstance(entity_chunk_size, int)
            or entity_chunk_size < 1
        ):
            raise ValueError("entity_chunk_size must be a positive integer or None")

        self.heads = heads
        self.query_scalar_dim = query_scalar_dim
        self.query_vector_dim = query_vector_dim
        self.key_scalar_dim = key_scalar_dim
        self.key_vector_dim = key_vector_dim
        self.value_scalar_dim = value_scalar_dim
        self.value_vector_dim = value_vector_dim
        self.scalar_rank = scalar_rank
        self.vector_rank = vector_rank
        self.scalar_value_dim = scalar_value_dim
        self.vector_value_dim = vector_value_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        self.accumulation_dtype = accumulation_dtype
        self.entity_chunk_size = entity_chunk_size

        self.query_projection = TypedProjection(
            query_scalar_dim,
            query_vector_dim,
            heads * scalar_rank,
            heads * vector_rank,
            scalar_bias=qk_scalar_bias,
        )
        self.key_projection = TypedProjection(
            key_scalar_dim,
            key_vector_dim,
            heads * scalar_rank,
            heads * vector_rank,
            scalar_bias=qk_scalar_bias,
        )
        self.value_projection = TypedProjection(
            value_scalar_dim,
            value_vector_dim,
            heads * scalar_value_dim,
            heads * vector_value_dim,
            scalar_bias=value_scalar_bias,
            include_vector_invariants=value_include_vector_invariants,
        )

        if out_scalar_dim and not scalar_value_dim:
            raise ValueError("Scalar output requires at least one scalar value channel")
        self.scalar_output = (
            nn.Linear(
                heads * scalar_value_dim,
                out_scalar_dim,
                bias=output_scalar_bias,
            )
            if out_scalar_dim
            else None
        )
        if out_vector_dim and not vector_value_dim:
            raise ValueError("Vector output requires at least one vector value channel")
        if out_vector_dim:
            self.vector_output_weight = nn.Parameter(
                torch.randn(out_vector_dim, heads, vector_value_dim)
                / math.sqrt(max(heads * vector_value_dim, 1))
            )
        else:
            self.register_parameter("vector_output_weight", None)

    def _accumulation_type(self, *tensors: torch.Tensor) -> torch.dtype:
        """Promote inputs with a precision floor, never downcast FP64."""
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        if self.accumulation_dtype is not None:
            dtype = torch.promote_types(dtype, self.accumulation_dtype)
        return dtype

    @staticmethod
    def _validate_projection_state(
        state: ScalarVectorState,
        *,
        scalar_dim: int,
        vector_dim: int,
        label: str,
    ) -> None:
        state.validate(label=label)
        if state.scalars.shape[1] != scalar_dim:
            raise ValueError(
                f"{label}.scalars has {state.scalars.shape[1]} channels; "
                f"expected {scalar_dim}"
            )
        if state.vectors.shape[1] != vector_dim:
            raise ValueError(
                f"{label}.vectors has {state.vectors.shape[1]} channels; "
                f"expected {vector_dim}"
            )

    def project_queries(self, state: ScalarVectorState) -> TypedQK:
        self._validate_projection_state(
            state,
            scalar_dim=self.query_scalar_dim,
            vector_dim=self.query_vector_dim,
            label="query_state",
        )
        projected = self.query_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        # The vector ranks contain D Cartesian components, so the invariant
        # signed dot product has R_s + D R_v independently varying terms.
        score_scale = 1.0 / math.sqrt(max(self.scalar_rank + d * self.vector_rank, 1))
        return TypedQK(
            projected.scalars.reshape(n, self.heads, self.scalar_rank) * score_scale,
            projected.vectors.reshape(n, self.heads, self.vector_rank, d) * score_scale,
        )

    def project_keys(self, state: ScalarVectorState) -> TypedQK:
        self._validate_projection_state(
            state,
            scalar_dim=self.key_scalar_dim,
            vector_dim=self.key_vector_dim,
            label="key_state",
        )
        projected = self.key_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        return TypedQK(
            projected.scalars.reshape(n, self.heads, self.scalar_rank),
            projected.vectors.reshape(n, self.heads, self.vector_rank, d),
        )

    def project_values(self, state: ScalarVectorState) -> TypedValues:
        self._validate_projection_state(
            state,
            scalar_dim=self.value_scalar_dim,
            vector_dim=self.value_vector_dim,
            label="value_state",
        )
        projected = self.value_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        return TypedValues(
            projected.scalars.reshape(n, self.heads, self.scalar_value_dim),
            projected.vectors.reshape(n, self.heads, self.vector_value_dim, d),
        )

    def build_moments(
        self,
        source_mesh: Mesh,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
    ) -> AttentionMoments:
        r"""Project and quadrature-integrate keys and values once."""
        if source_mesh.n_cells != key_state.n_entities:
            raise ValueError("source Mesh cell count must match key state entity count")
        if key_state.n_entities != value_state.n_entities:
            raise ValueError("key and value entity counts must match")
        if source_mesh.n_spatial_dims != key_state.n_spatial_dims:
            raise ValueError("source Mesh and key state spatial dims differ")
        if source_mesh.n_spatial_dims != value_state.n_spatial_dims:
            raise ValueError("source Mesh and value state spatial dims differ")

        # Attention heads are aligned groups, not axes to outer-product with
        # one another. The Mesh owns quadrature measure; the shared weighted
        # primitive owns NaN and accumulation policy.
        def _moments_from_projected(
            keys: TypedQK,
            values: TypedValues,
            weights: torch.Tensor | None,
        ) -> AttentionMoments:
            # Cartesian components are independently varying finite-rank
            # features in the signed kernel. Flatten them next to the scalar
            # features so all four typed key/value moments share one weighted
            # matrix multiplication. Slicing the joint moment back into typed
            # blocks preserves the public representation and evaluation math.
            key_features = torch.cat(
                (keys.scalars, keys.vectors.flatten(start_dim=2)), dim=-1
            )
            value_features = torch.cat(
                (values.scalars, values.vectors.flatten(start_dim=2)), dim=-1
            )
            if weights is None:
                joint_moment = integrate_moment(
                    source_mesh,
                    key_features,
                    value_features,
                    aligned_dims=1,
                    accumulation_dtype=self.accumulation_dtype,
                    nan_policy="propagate",
                )
            else:
                joint_moment = _integrate_weighted_moment(
                    key_features,
                    value_features,
                    weights,
                    aligned_dims=1,
                    accumulation_dtype=self.accumulation_dtype,
                    nan_policy="propagate",
                )

            scalar_rank = self.scalar_rank
            scalar_value_dim = self.scalar_value_dim
            spatial_dim = keys.vectors.shape[-1]
            return AttentionMoments(
                scalar_key_scalar_value=joint_moment[
                    :, :scalar_rank, :scalar_value_dim
                ],
                vector_key_scalar_value=joint_moment[
                    :, scalar_rank:, :scalar_value_dim
                ].reshape(
                    self.heads,
                    self.vector_rank,
                    spatial_dim,
                    self.scalar_value_dim,
                ),
                scalar_key_vector_value=joint_moment[
                    :, :scalar_rank, scalar_value_dim:
                ].reshape(
                    self.heads,
                    self.scalar_rank,
                    self.vector_value_dim,
                    spatial_dim,
                ),
                vector_key_vector_value=joint_moment[
                    :, scalar_rank:, scalar_value_dim:
                ].reshape(
                    self.heads,
                    self.vector_rank,
                    spatial_dim,
                    self.vector_value_dim,
                    spatial_dim,
                ),
            )

        chunk_size = self.entity_chunk_size
        if chunk_size is None or key_state.n_entities <= chunk_size:
            return _moments_from_projected(
                self.project_keys(key_state),
                self.project_values(value_state),
                None,
            )

        accumulated: AttentionMoments | None = None
        weights = source_mesh.cell_areas
        for start in range(0, key_state.n_entities, chunk_size):
            item = slice(start, min(start + chunk_size, key_state.n_entities))
            chunk_moments = _moments_from_projected(
                self.project_keys(key_state.slice(item)),
                self.project_values(value_state.slice(item)),
                weights[item],
            )
            if accumulated is None:
                accumulated = chunk_moments
            else:
                accumulated = AttentionMoments(
                    accumulated.scalar_key_scalar_value
                    + chunk_moments.scalar_key_scalar_value,
                    accumulated.vector_key_scalar_value
                    + chunk_moments.vector_key_scalar_value,
                    accumulated.scalar_key_vector_value
                    + chunk_moments.scalar_key_vector_value,
                    accumulated.vector_key_vector_value
                    + chunk_moments.vector_key_vector_value,
                )
        if accumulated is None:
            raise RuntimeError("Cannot build attention moments from an empty source")
        return accumulated

    def evaluate_moments(
        self,
        query_state: ScalarVectorState,
        moments: AttentionMoments,
    ) -> ScalarVectorState:
        r"""Evaluate cached source moments independently at each receiver."""
        d = query_state.n_spatial_dims
        expected_shapes = (
            (self.heads, self.scalar_rank, self.scalar_value_dim),
            (self.heads, self.vector_rank, d, self.scalar_value_dim),
            (self.heads, self.scalar_rank, self.vector_value_dim, d),
            (self.heads, self.vector_rank, d, self.vector_value_dim, d),
        )
        actual_shapes = (
            tuple(moments.scalar_key_scalar_value.shape),
            tuple(moments.vector_key_scalar_value.shape),
            tuple(moments.scalar_key_vector_value.shape),
            tuple(moments.vector_key_vector_value.shape),
        )
        if actual_shapes != expected_shapes:
            raise ValueError(
                "AttentionMoments are incompatible with this layer/query; "
                f"expected {expected_shapes}, got {actual_shapes}"
            )
        moment_tensors = (
            moments.scalar_key_scalar_value,
            moments.vector_key_scalar_value,
            moments.scalar_key_vector_value,
            moments.vector_key_vector_value,
        )
        if any(
            tensor.device != query_state.scalars.device for tensor in moment_tensors
        ):
            raise ValueError("AttentionMoments and query_state must share a device")
        chunk_size = self.entity_chunk_size
        if chunk_size is not None and query_state.n_entities > chunk_size:
            outputs = [
                self.evaluate_moments(
                    query_state.slice(
                        slice(
                            start,
                            min(start + chunk_size, query_state.n_entities),
                        )
                    ),
                    moments,
                )
                for start in range(0, query_state.n_entities, chunk_size)
            ]
            return ScalarVectorState(
                torch.cat([output.scalars for output in outputs], dim=0),
                torch.cat([output.vectors for output in outputs], dim=0),
            )
        queries = self.project_queries(query_state)
        output_dtype = query_state.scalars.dtype
        dtype = self._accumulation_type(
            queries.scalars,
            queries.vectors,
            moments.scalar_key_scalar_value,
            moments.vector_key_scalar_value,
            moments.scalar_key_vector_value,
            moments.vector_key_vector_value,
        )
        qs = queries.scalars.to(dtype)
        qv = queries.vectors.to(dtype)
        scalar_key_scalar_value = moments.scalar_key_scalar_value.to(dtype)
        vector_key_scalar_value = moments.vector_key_scalar_value.to(dtype)
        scalar_key_vector_value = moments.scalar_key_vector_value.to(dtype)
        vector_key_vector_value = moments.vector_key_vector_value.to(dtype)

        with torch.autocast(device_type=qs.device.type, enabled=False):
            scalar_heads = torch.einsum(
                "nhr,hrf->nhf", qs, scalar_key_scalar_value
            ) + torch.einsum("nhrd,hrdf->nhf", qv, vector_key_scalar_value)
            vector_heads = torch.einsum(
                "nhr,hrfd->nhfd", qs, scalar_key_vector_value
            ) + torch.einsum("nhrd,hrdfe->nhfe", qv, vector_key_vector_value)

        scalars = (
            self.scalar_output(
                scalar_heads.to(output_dtype).reshape(
                    query_state.n_entities,
                    self.heads * self.scalar_value_dim,
                )
            )
            if self.scalar_output is not None
            else query_state.scalars.new_empty(query_state.n_entities, 0)
        )
        if self.out_vector_dim:
            vectors = torch.einsum(
                "ohf,nhfd->nod",
                self.vector_output_weight,
                vector_heads.to(output_dtype),
            )
        else:
            vectors = query_state.vectors.new_empty(
                query_state.n_entities, 0, query_state.n_spatial_dims
            )
        return ScalarVectorState(scalars, vectors.to(dtype=scalars.dtype))

    def forward(
        self,
        source_mesh: Mesh,
        query_state: ScalarVectorState,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
    ) -> ScalarVectorState:
        return self.evaluate_moments(
            query_state,
            self.build_moments(source_mesh, key_state, value_state),
        )

    def forward_reference(
        self,
        source_mesh: Mesh,
        query_state: ScalarVectorState,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
    ) -> ScalarVectorState:
        r"""Dense all-pairs oracle for values and gradient tests."""
        if source_mesh.n_cells != key_state.n_entities:
            raise ValueError("source Mesh cell count must match key state entity count")
        if key_state.n_entities != value_state.n_entities:
            raise ValueError("key and value entity counts must match")
        if (
            source_mesh.n_spatial_dims != key_state.n_spatial_dims
            or source_mesh.n_spatial_dims != value_state.n_spatial_dims
            or source_mesh.n_spatial_dims != query_state.n_spatial_dims
        ):
            raise ValueError(
                "source Mesh, query, key, and value spatial dimensions must match"
            )
        q = self.project_queries(query_state)
        k = self.project_keys(key_state)
        v = self.project_values(value_state)
        dtype = self._accumulation_type(
            q.scalars,
            q.vectors,
            k.scalars,
            k.vectors,
            v.scalars,
            v.vectors,
            source_mesh.cell_areas,
        )
        with torch.autocast(device_type=q.scalars.device.type, enabled=False):
            score = torch.einsum(
                "mhr,nhr->mnh", q.scalars.to(dtype), k.scalars.to(dtype)
            ) + torch.einsum("mhrd,nhrd->mnh", q.vectors.to(dtype), k.vectors.to(dtype))
            weighted_score = score * source_mesh.cell_areas.to(dtype)[None, :, None]
            scalar_heads = torch.einsum(
                "mnh,nhf->mhf", weighted_score, v.scalars.to(dtype)
            )
            vector_heads = torch.einsum(
                "mnh,nhfd->mhfd", weighted_score, v.vectors.to(dtype)
            )
        output_dtype = query_state.scalars.dtype
        scalars = (
            self.scalar_output(
                scalar_heads.to(output_dtype).reshape(
                    query_state.n_entities,
                    self.heads * self.scalar_value_dim,
                )
            )
            if self.scalar_output is not None
            else query_state.scalars.new_empty(query_state.n_entities, 0)
        )
        if self.out_vector_dim:
            vectors = torch.einsum(
                "ohf,nhfd->nod",
                self.vector_output_weight,
                vector_heads.to(output_dtype),
            )
        else:
            vectors = query_state.vectors.new_empty(
                query_state.n_entities, 0, query_state.n_spatial_dims
            )
        return ScalarVectorState(scalars, vectors.to(dtype=scalars.dtype))


__all__ = [
    "AttentionMoments",
    "MeshAttention",
    "ScalarVectorState",
    "TypedProjection",
]
