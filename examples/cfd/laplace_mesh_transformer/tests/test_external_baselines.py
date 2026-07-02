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

"""Smoke and data-contract checks for common-protocol external controls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    build_domain_sample,
    sample_drive,
    sample_geometry,
)
from external_baselines import (  # noqa: E402
    GeoTransolverLaplaceAdapter,
    GlobeLaplaceAdapter,
    normalize_domain,
)
from models import parameter_count  # noqa: E402
from train import make_model  # noqa: E402


def _sample():
    geometry = sample_geometry(301, deformation_range=(0.2, 0.2))
    drive = sample_drive(302, modes=(1, 2, 3), regularity=0.0)
    return build_domain_sample(
        geometry, drive, n_boundary=12, n_query=9, query_seed=303
    )


def test_external_normalization_uses_declared_reference_length() -> None:
    """Normalize external controls with the mesh-declared physical scale."""

    sample = _sample()
    normalized = normalize_domain(sample.domain)
    weights = normalized.boundary.cell_areas
    center = (
        torch.einsum("s,sd->d", weights, normalized.boundary.cell_centroids)
        / weights.sum()
    )
    torch.testing.assert_close(center, torch.zeros_like(center), atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize(
    "model",
    [
        GlobeLaplaceAdapter(hidden_dim=8, hidden_layers=1, network_type="mlp"),
        GeoTransolverLaplaceAdapter(hidden_dim=16, layers=1, heads=2, slices=4),
    ],
)
def test_external_adapter_forward_and_gradients(model: torch.nn.Module) -> None:
    """Require each external adapter to predict and backpropagate end to end."""

    torch.manual_seed(304)
    sample = _sample()
    output = model(sample.domain).point_data["potential"]
    assert output.shape == sample.target.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_geotransolver_control_has_no_ball_query_radii() -> None:
    """Keep the GeoTransolver control free of hand-chosen spatial radii."""

    model = GeoTransolverLaplaceAdapter(hidden_dim=16, layers=1, heads=2, slices=4)
    assert model.model.include_local_features is False
    assert model.model.radii == []


def test_external_registry_preserves_backend_and_capacity_labels() -> None:
    """Label external backend and capacity variants without ambiguity."""

    exact = make_model("globe_exact", "reference")
    hierarchical = make_model("globe_hierarchical", "reference")
    assert exact.model.theta == 0.0
    assert hierarchical.model.theta == 1.0

    matched = make_model("geotransolver_matched", "reference")
    published_scale = make_model("geotransolver_published_scale", "reference")
    assert parameter_count(matched) == 133_083
    assert parameter_count(published_scale) == 29_144_481
