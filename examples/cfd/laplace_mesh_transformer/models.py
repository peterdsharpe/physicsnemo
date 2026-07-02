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

"""Models and physically controlled baselines for the Laplace benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh


@dataclass(frozen=True)
class MeshTransformerConfig:
    """Finite-capacity settings used by the reference benchmark."""

    operator_scalar_dim: int = 32
    operator_vector_dim: int = 8
    drive_scalar_dim: int = 48
    drive_vector_dim: int = 12
    operator_layers: int = 2
    drive_layers: int = 1
    query_layers: int = 1
    heads: int = 4
    scalar_rank: int = 12
    vector_rank: int = 4
    query_chunk_size: int = 65536
    attention_chunk_size: int | None = 65536


def build_mesh_transformer(config: MeshTransformerConfig) -> MeshTransformer:
    """Construct the scalar, linear Dirichlet-to-interior benchmark model."""

    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={
            "dirichlet": {
                "operator": {},
                "drive": {"boundary_value": 0},
            }
        },
        global_field_ranks={"operator": {}, "drive": {}},
        reference_length_key="reference_length",
        field_mode="linear",
        **asdict(config),
    )


class MeanLiftedDirichletModel(nn.Module):
    r"""Enforce the exact constant solution and learn only the residual map.

    If :math:`\bar g` is the boundary-measure mean, this wrapper evaluates

    .. math::

        u_\theta[g] = \bar g + N_\theta[g-\bar g].

    The construction is linear in the Dirichlet data, exactly reproduces every
    constant boundary condition, and introduces no coordinate frame, length
    scale, or locality heuristic.  It is Laplace-specific physics structure,
    so it remains an example wrapper rather than a generic MeshTransformer
    contract.
    """

    def __init__(self, residual_model: MeshTransformer) -> None:
        super().__init__()
        self.residual_model = residual_model

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        weights = boundary.cell_areas
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual_boundary = boundary.with_data(
            cell_data={"boundary_value": values - mean}
        )
        residual_domain = DomainMesh(
            interior=domain.interior,
            boundaries={"dirichlet": residual_boundary},
            global_data=domain.global_data,
        )
        residual = self.residual_model(residual_domain)
        return residual.with_data(
            point_data={"potential": residual.point_data["potential"] + mean},
            cell_data={},
            global_data=domain.global_data,
        )


def build_lifted_mesh_transformer(
    config: MeshTransformerConfig,
) -> MeanLiftedDirichletModel:
    """Construct the constant-consistent Laplace specialization."""

    return MeanLiftedDirichletModel(build_mesh_transformer(config))


def _benchmark_boundary(domain: DomainMesh) -> Mesh:
    if set(domain.boundaries.keys()) != {"dirichlet"}:
        raise ValueError("benchmark domains must contain only a 'dirichlet' boundary")
    boundary = domain.boundaries["dirichlet"]
    if "boundary_value" not in boundary.cell_data:
        raise ValueError("dirichlet.cell_data must contain 'boundary_value'")
    return boundary


def _reference_length(domain: DomainMesh) -> torch.Tensor:
    try:
        length = domain.global_data["reference_length"].reshape(())
    except KeyError:
        raise ValueError("domain.global_data must contain 'reference_length'") from None
    if not torch.compiler.is_compiling() and (
        not torch.isfinite(length).item() or length.item() <= 0.0
    ):
        raise ValueError("reference_length must be finite and positive")
    return length


def _constant_exact_boundary_mean(
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Return a linear quadrature mean that preserves constants bit-for-bit."""

    anchor = values[0]
    return anchor + torch.sum(weights * (values - anchor)) / weights.sum()


def _prediction_mesh(domain: DomainMesh, potential: torch.Tensor) -> Mesh:
    return domain.interior.with_data(
        point_data={"potential": potential},
        cell_data={},
        global_data=domain.global_data,
    )


class BoundaryMean(nn.Module):
    r"""Parameter-free, quadrature-weighted constant baseline.

    This baseline exactly reproduces constant Dirichlet data.  It deliberately
    has no spatial capacity and therefore quantifies how much a learned model
    improves over predicting one domain-wide value.
    """

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        weights = boundary.cell_areas
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        return _prediction_mesh(domain, mean.expand(domain.interior.n_points))


class InvariantPairKernel(nn.Module):
    r"""Dense linear pair-kernel baseline with the same physical symmetries.

    For normalized query position :math:`x`, source centroid :math:`y`, and
    outward normal :math:`n`, the learned kernel sees only

    .. math::

        (\lVert x-y\rVert^2,\; n\cdot(x-y)).

    These are joint O(2) invariants and contain no absolute position or fitted
    interaction radius.  The output is linear in the boundary data and uses
    the boundary measure.  Unlike ``MeshTransformer``, it materializes dense
    query-source pairs and has no global geometry encoder.  It is therefore a
    useful control for the expressiveness/cost tradeoff, not a proposed
    production architecture.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or hidden_layers < 1 or query_chunk_size < 1:
            raise ValueError(
                "hidden_dim, hidden_layers, and chunk size must be positive"
            )

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        final = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(final.weight, std=1.0e-2 / hidden_dim**0.5)
        layers.append(final)
        self.kernel = nn.Sequential(*layers)
        self.query_chunk_size = query_chunk_size

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        length = _reference_length(domain)
        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            total_measure = weights.sum()
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / total_measure
            source_points = (boundary.cell_centroids - center) / length
            query_points = (domain.interior.points - center) / length
            normals = boundary.cell_normals

        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual = values - mean

        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            query = query_points[start : start + self.query_chunk_size]
            displacement = query[:, None, :] - source_points[None, :, :]
            features = torch.stack(
                (
                    displacement.square().sum(dim=-1),
                    torch.einsum("qsd,sd->qs", displacement, normals),
                ),
                dim=-1,
            )
            pair_kernel = self.kernel(features).squeeze(-1)
            chunks.append(
                mean + torch.einsum("qs,s,s->q", pair_kernel, weights, residual)
            )

        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


__all__ = [
    "BoundaryMean",
    "InvariantPairKernel",
    "MeanLiftedDirichletModel",
    "MeshTransformerConfig",
    "build_lifted_mesh_transformer",
    "build_mesh_transformer",
    "parameter_count",
]
