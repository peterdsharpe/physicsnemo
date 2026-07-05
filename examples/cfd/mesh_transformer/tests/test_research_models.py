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

"""Contracts for benchmark-local research models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    build_domain_sample,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
)
from models import MeshTransformerConfig  # noqa: E402
from research_models import EncodedInvariantPairKernel  # noqa: E402

from physicsnemo.mesh import DomainMesh  # noqa: E402


def _sample():
    geometry = sample_geometry(101, deformation_range=(0.25, 0.25), dtype=torch.float64)
    drive = sample_drive(102, modes=(1, 2, 3), regularity=0.0, dtype=torch.float64)
    return build_domain_sample(
        geometry, drive, n_boundary=24, n_query=19, query_seed=103
    )


def _tiny_model() -> EncodedInvariantPairKernel:
    config = MeshTransformerConfig(
        operator_scalar_dim=8,
        operator_vector_dim=3,
        drive_scalar_dim=8,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=1,
        heads=1,
        scalar_rank=3,
        vector_rank=2,
    )
    return EncodedInvariantPairKernel(
        config, density_channels=4, hidden_dim=12, hidden_layers=2
    ).to(dtype=torch.float64)


def test_encoded_pair_kernel_is_linear_in_boundary_drive() -> None:
    """Preserve exact drive superposition through the encoded pair model."""

    torch.manual_seed(104)
    model = _tiny_model()
    sample = _sample()
    boundary = sample.domain.boundaries["dirichlet"]
    first = torch.randn(boundary.n_cells, dtype=torch.float64)
    second = torch.randn(boundary.n_cells, dtype=torch.float64)

    def predict(values: torch.Tensor) -> torch.Tensor:
        domain = DomainMesh(
            interior=sample.domain.interior,
            boundaries={
                "dirichlet": boundary.with_data(cell_data={"boundary_value": values})
            },
            global_data=sample.domain.global_data,
        )
        return model(domain).point_data["potential"]

    torch.testing.assert_close(
        predict(1.7 * first - 0.4 * second),
        1.7 * predict(first) - 0.4 * predict(second),
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_encoded_pair_kernel_is_o2_similarity_invariant() -> None:
    """Keep scalar predictions invariant under similarities and reflections."""

    torch.manual_seed(105)
    model = _tiny_model()
    sample = _sample()
    transformed = transform_sample(
        sample,
        sample_similarity(
            106,
            scale_range=(3.5, 3.5),
            translation_extent=2.0,
            reflection=True,
            dtype=torch.float64,
        ),
    )

    expected = model(sample.domain).point_data["potential"]
    actual = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=5.0e-10, atol=5.0e-11)


def test_encoded_pair_kernel_has_end_to_end_gradients() -> None:
    """Use every encoded pair-kernel parameter in differentiable prediction."""

    torch.manual_seed(107)
    model = _tiny_model()
    output = model(_sample().domain).point_data["potential"]
    output.square().mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
