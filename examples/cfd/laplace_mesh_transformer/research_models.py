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

r"""Benchmark-local models that isolate MeshTransformer design hypotheses.

Nothing in this module is a proposed public API.  In particular,
``EncodedInvariantPairKernel`` is the dense oracle for the factorial ablation:
it retains the current boundary encoder and replaces only the separable
boundary-to-query moment decoder with an explicit relative pair kernel.
"""

from __future__ import annotations

import math

import torch
from models import (
    MeshTransformerConfig,
    _benchmark_boundary,
    _constant_exact_boundary_mean,
    _prediction_mesh,
    build_mesh_transformer,
)
from torch import nn

from physicsnemo.experimental.nn.mesh_attention.attention import _gram_invariants
from physicsnemo.experimental.nn.mesh_attention.block import GeometryConditionedLinear
from physicsnemo.mesh import DomainMesh


class EncodedInvariantPairKernel(nn.Module):
    r"""Current boundary encoder followed by a dense invariant pair decoder.

    The kernel is conditioned only on the operator/geometry stream.  A separate
    equivariant linear map converts the encoded drive to scalar source-density
    channels.  Consequently the complete model remains exactly linear in the
    Dirichlet drive at fixed geometry even though the pair kernel is nonlinear.

    For normalized displacement :math:`r=(x-y)/L`, source normal :math:`n`,
    source operator scalars :math:`s`, and operator vectors :math:`v_a`, the
    kernel sees only the joint invariants

    .. math::

       |r|^2,\quad n\cdot r,\quad s,\quad
       v_a\cdot r,\quad v_a\cdot n,\quad v_a\cdot v_b.

    It contains no absolute position, axis-dependent Fourier feature, softmax,
    fitted radius, or interaction cutoff.  Dense execution is intentional: it
    is the exact oracle against which a future hierarchical backend would be
    checked.
    """

    def __init__(
        self,
        config: MeshTransformerConfig | None = None,
        *,
        density_channels: int = 16,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        query_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        config = MeshTransformerConfig() if config is None else config
        for name, value in (
            ("density_channels", density_channels),
            ("hidden_dim", hidden_dim),
            ("hidden_layers", hidden_layers),
            ("query_chunk_size", query_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.backbone = build_mesh_transformer(config)
        # ``MeshTransformer.encode`` normally also prepares stock query
        # moments. This ablation deliberately removes that decoder so neither
        # its parameters nor its otherwise-unused moment construction enter
        # the capacity/runtime comparison. The eventual winning design should
        # expose a dedicated boundary encoder rather than retain this
        # benchmark-local surgery.
        self.backbone.query_blocks = nn.ModuleList()
        self.backbone.output_projection = nn.Identity()
        self.density_channels = density_channels
        self.query_chunk_size = query_chunk_size
        self.density_projection = GeometryConditionedLinear(
            config.operator_scalar_dim,
            config.operator_vector_dim,
            config.drive_scalar_dim,
            config.drive_vector_dim,
            density_channels,
            0,
        )

        vector_invariants = (
            config.operator_vector_dim * (config.operator_vector_dim + 1) // 2
        )
        kernel_features = (
            2
            + config.operator_scalar_dim
            + 2 * config.operator_vector_dim
            + vector_invariants
        )
        layers: list[nn.Module] = [nn.Linear(kernel_features, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        output = nn.Linear(hidden_dim, density_channels, bias=False)
        nn.init.normal_(output.weight, std=1.0e-2 / math.sqrt(hidden_dim))
        layers.append(output)
        self.kernel = nn.Sequential(*layers)

    @staticmethod
    def _residual_domain(domain: DomainMesh) -> tuple[DomainMesh, torch.Tensor]:
        boundary = _benchmark_boundary(domain)
        weights = boundary.cell_areas
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual_boundary = boundary.with_data(
            cell_data={"boundary_value": values - mean}
        )
        return (
            DomainMesh(
                interior=domain.interior,
                boundaries={"dirichlet": residual_boundary},
                global_data=domain.global_data,
            ),
            mean,
        )

    @staticmethod
    def _pair_features(
        displacement: torch.Tensor,
        normals: torch.Tensor,
        operator_scalars: torch.Tensor,
        operator_vectors: torch.Tensor,
        operator_grams: torch.Tensor,
    ) -> torch.Tensor:
        n_query, n_source, _ = displacement.shape
        normal_dot_displacement = torch.einsum("qsd,sd->qs", displacement, normals)
        vector_dot_displacement = torch.einsum(
            "qsd,svd->qsv", displacement, operator_vectors
        )
        vector_dot_normal = torch.einsum("svd,sd->sv", operator_vectors, normals)
        source_features = torch.cat(
            (operator_scalars, vector_dot_normal, operator_grams), dim=-1
        )
        return torch.cat(
            (
                displacement.square().sum(dim=-1, keepdim=True),
                normal_dot_displacement.unsqueeze(-1),
                source_features.unsqueeze(0).expand(n_query, n_source, -1),
                vector_dot_displacement,
            ),
            dim=-1,
        )

    def forward(self, domain: DomainMesh):
        residual_domain, mean = self._residual_domain(domain)
        encoded = self.backbone.encode(residual_domain)
        density = self.density_projection(
            encoded.operator_state, encoded.drive_state
        ).scalars

        source_points = encoded.source_mesh.cell_centroids
        normals = encoded.source_mesh.cell_normals
        weights = encoded.source_mesh.cell_areas
        operator_scalars = encoded.operator_state.scalars
        operator_vectors = encoded.operator_state.vectors
        operator_grams = _gram_invariants(operator_vectors)
        query_points = (
            domain.interior.points - encoded.center
        ) / encoded.reference_length

        predictions: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            query = query_points[start : start + self.query_chunk_size]
            displacement = query[:, None, :] - source_points[None, :, :]
            features = self._pair_features(
                displacement,
                normals,
                operator_scalars,
                operator_vectors,
                operator_grams,
            )
            coefficients = self.kernel(features)
            predictions.append(
                mean + torch.einsum("qsc,sc,s->q", coefficients, density, weights)
            )

        potential = (
            torch.cat(predictions)
            if predictions
            else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)


__all__ = ["EncodedInvariantPairKernel"]
