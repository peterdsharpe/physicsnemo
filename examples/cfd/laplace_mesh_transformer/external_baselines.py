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

r"""Common-protocol GLOBE and GeoTransolver controls for the Laplace study.

These adapters retain each model's native processing while making every
benchmark adaptation explicit rather than silently granting MeshTransformer's
contracts. GLOBE receives a dimensionless boundary mesh and explicit
relative-kernel scale. GeoTransolver receives query tokens and a separate
boundary-context cloud with its multiscale fixed-radius path disabled. The
latter is therefore a radius-free GALE/context ablation, not a reproduction of
the published GeoTransolver. Neither adapter claims exact drive superposition;
the Cartesian GeoTransolver features also do not provide constructive O(2)
equivariance, so that property is tested empirically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from models import _benchmark_boundary, _prediction_mesh, _reference_length
from torch import nn

from physicsnemo.experimental.models.geotransolver import GeoTransolver
from physicsnemo.experimental.models.globe import GLOBE
from physicsnemo.mesh import DomainMesh, Mesh


@dataclass(frozen=True)
class NormalizedBoundaryQuery:
    """A shared dimensionless frame used only by external adapters."""

    boundary: Mesh
    query_points: torch.Tensor


def normalize_domain(domain: DomainMesh) -> NormalizedBoundaryQuery:
    """Center and nondimensionalize without fitting a data-dependent scale."""

    boundary = _benchmark_boundary(domain)
    length = _reference_length(domain)
    with torch.autocast(device_type=boundary.points.device.type, enabled=False):
        weights = boundary.cell_areas
        center = torch.einsum("s,sd->d", weights, boundary.cell_centroids)
        center = center / weights.sum()
        normalized_boundary = Mesh(
            points=(boundary.points - center) / length,
            cells=boundary.cells,
            cell_data=boundary.cell_data,
        )
        query_points = (domain.interior.points - center) / length
    return NormalizedBoundaryQuery(normalized_boundary, query_points)


class GlobeLaplaceAdapter(nn.Module):
    """Run stock GLOBE on the exact conformal-Laplace protocol."""

    def __init__(
        self,
        *,
        communication_layers: int = 0,
        theta: float = 0.0,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        latent_scalars: int = 12,
        latent_vectors: int = 6,
        n_spherical_harmonics: int = 4,
        leaf_size: int = 1,
        network_type: Literal["pade", "mlp"] = "pade",
    ) -> None:
        super().__init__()
        if communication_layers < 0:
            raise ValueError("communication_layers must be nonnegative")
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("hidden dimensions/layers must be positive")
        self.model = GLOBE(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_source_data_ranks={"dirichlet": {"boundary_value": 0}},
            reference_length_names=("domain",),
            reference_area=1.0,
            global_data_ranks={},
            n_communication_hyperlayers=communication_layers,
            n_latent_scalars=latent_scalars,
            n_latent_vectors=latent_vectors,
            hidden_layer_sizes=[hidden_dim] * hidden_layers,
            n_spherical_harmonics=n_spherical_harmonics,
            theta=theta,
            leaf_size=leaf_size,
            network_type=network_type,
            use_gradient_checkpointing=False,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        normalized = normalize_domain(domain)
        parameter = next(self.model.parameters())
        boundary = normalized.boundary.to(
            device=parameter.device, dtype=parameter.dtype
        )
        query_points = normalized.query_points.to(
            device=parameter.device, dtype=parameter.dtype
        )
        one = query_points.new_ones(())
        result = self.model(
            prediction_points=query_points,
            boundary_meshes={"dirichlet": boundary},
            reference_lengths={"domain": one},
            prediction_chunk_size=None,
        )
        return _prediction_mesh(domain, result.point_data["potential"])


class GeoTransolverLaplaceAdapter(nn.Module):
    """Use query tokens with a separate normalized boundary-context cloud."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        slices: int = 16,
    ) -> None:
        super().__init__()
        self.model = GeoTransolver(
            functional_dim=3,
            out_dim=1,
            geometry_dim=7,
            global_dim=None,
            n_layers=layers,
            n_hidden=hidden_dim,
            n_head=heads,
            slice_num=slices,
            use_te=False,
            include_local_features=False,
            attention_type="GALE",
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        normalized = normalize_domain(domain)
        parameter = next(self.model.parameters())
        boundary = normalized.boundary.to(
            device=parameter.device, dtype=parameter.dtype
        )
        query = normalized.query_points.to(
            device=parameter.device, dtype=parameter.dtype
        )
        query_features = torch.cat((query, query.new_ones(query.shape[0], 1)), dim=-1)
        areas = boundary.cell_areas[:, None]
        values = boundary.cell_data["boundary_value"][:, None]
        boundary_features = torch.cat(
            (
                boundary.cell_centroids,
                boundary.cell_normals,
                areas,
                values,
                values.new_ones(values.shape),
            ),
            dim=-1,
        )
        output = self.model(
            query_features.unsqueeze(0),
            geometry=boundary_features.unsqueeze(0),
        )
        return _prediction_mesh(domain, output[0, :, 0])


__all__ = [
    "GeoTransolverLaplaceAdapter",
    "GlobeLaplaceAdapter",
    "NormalizedBoundaryQuery",
    "normalize_domain",
]
