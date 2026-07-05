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

"""Smoke, faithfulness, and capacity checks for the Transolver baseline."""

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
from models import parameter_count  # noqa: E402
from train import make_model  # noqa: E402
from transolver import (  # noqa: E402
    PhysicsAttention,
    TransolverLaplaceAdapter,
    build_transolver,
)


def _sample():
    geometry = sample_geometry(401, deformation_range=(0.2, 0.2))
    drive = sample_drive(402, modes=(1, 2, 3), regularity=0.0)
    return build_domain_sample(
        geometry, drive, n_boundary=12, n_query=9, query_seed=403
    )


def _tiny_adapter() -> TransolverLaplaceAdapter:
    return TransolverLaplaceAdapter(
        hidden_dim=16, layers=2, heads=2, slice_num=4, mlp_ratio=1
    )


def test_transolver_forward_and_gradients() -> None:
    """Require the adapter to predict and backpropagate end to end."""

    torch.manual_seed(404)
    model = _tiny_adapter()
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


def test_physics_attention_layer_contract() -> None:
    """Slice weights must be a per-token softmax and the layer shape-stable."""

    torch.manual_seed(405)
    layer = PhysicsAttention(dim=16, heads=2, dim_head=8, slice_num=4)
    x = torch.randn(2, 11, 16)
    out = layer(x)
    assert out.shape == x.shape
    x_mid = layer.in_project_x(x).reshape(2, 11, 2, 8).permute(0, 2, 1, 3).contiguous()
    slice_weights = layer.softmax(layer.in_project_slice(x_mid) / layer.temperature)
    torch.testing.assert_close(
        slice_weights.sum(dim=-1), torch.ones(2, 2, 11), atol=1e-6, rtol=0.0
    )


def test_transolver_temperature_is_learnable_per_head() -> None:
    """Preserve the official learnable per-head slice temperature at 0.5."""

    model = build_transolver("matched")
    for block in model.model.blocks:
        temperature = block.attn.temperature
        assert temperature.requires_grad
        assert temperature.shape == (1, 4, 1, 1)
        torch.testing.assert_close(
            temperature, torch.full_like(temperature, 0.5), atol=0.0, rtol=0.0
        )


def test_transolver_registry_parameter_counts() -> None:
    """Pin both preset capacities and the matched-arm parameter budget."""

    matched = make_model("transolver_matched", "reference")
    native = make_model("transolver_native", "reference")
    assert parameter_count(matched) == 103_053
    assert parameter_count(native) == 2_809_409
    reference = 104_261  # mesh_transformer_kernel_singonly
    assert abs(parameter_count(matched) - reference) / reference < 0.10


def test_build_transolver_rejects_unknown_preset() -> None:
    """Fail loudly on capacity labels that were never declared."""

    with pytest.raises(ValueError, match="unknown Transolver preset"):
        build_transolver("published")
