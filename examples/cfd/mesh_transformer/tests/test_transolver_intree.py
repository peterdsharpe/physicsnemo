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

"""Smoke and capacity checks for the in-tree Transolver baseline arms."""

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
from transolver import TRANSOLVER_PRESETS  # noqa: E402
from transolver_intree import (  # noqa: E402
    INTREE_TRANSOLVER_PRESETS,
    InTreeTransolverLaplaceAdapter,
    build_transolver_intree,
)


def _sample():
    geometry = sample_geometry(501, deformation_range=(0.2, 0.2))
    drive = sample_drive(502, modes=(1, 2, 3), regularity=0.0)
    return build_domain_sample(
        geometry, drive, n_boundary=12, n_query=9, query_seed=503
    )


def test_intree_transolver_forward_and_gradients() -> None:
    """Require the in-tree adapter to predict and backpropagate end to end."""

    torch.manual_seed(504)
    model = InTreeTransolverLaplaceAdapter(
        hidden_dim=16, layers=2, heads=2, slice_num=4, mlp_ratio=1
    )
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


def test_intree_presets_mirror_ported_presets() -> None:
    """Pin the in-tree arms to the exact ported hyperparameter settings."""

    assert INTREE_TRANSOLVER_PRESETS == {
        "intree_matched": TRANSOLVER_PRESETS["matched"],
        "intree_native": TRANSOLVER_PRESETS["native"],
    }


def test_intree_registry_parameter_counts() -> None:
    """Pin both in-tree capacities against the ported parameter budgets.

    Each in-tree preset is exactly ``hidden_dim`` parameters smaller than the
    corresponding ported preset: the official (ported) model allocates an
    inert ``placeholder`` parameter that the in-tree model omits.
    """

    matched = make_model("transolver_intree_matched", "reference")
    native = make_model("transolver_intree_native", "reference")
    assert parameter_count(matched) == 103_053 - 64 == 102_989
    assert parameter_count(native) == 2_809_409 - 256 == 2_809_153
    reference = 104_261  # mesh_transformer_kernel_singonly
    assert abs(parameter_count(matched) - reference) / reference < 0.10


def test_intree_temperature_matches_port_at_init_but_is_clamped() -> None:
    """Document the sole training-relevant divergence from the ported arm."""

    model = build_transolver_intree("intree_matched")
    for block in model.model.blocks:
        temperature = block.Attn.temperature
        assert temperature.requires_grad
        # In-tree layout is (1, 1, heads, 1) vs the port's (1, heads, 1, 1);
        # both are one learnable temperature per head initialized at 0.5.
        assert temperature.shape == (1, 1, 4, 1)
        torch.testing.assert_close(
            temperature, torch.full_like(temperature, 0.5), atol=0.0, rtol=0.0
        )


def test_build_transolver_intree_rejects_unknown_preset() -> None:
    """Fail loudly on capacity labels that were never declared."""

    with pytest.raises(ValueError, match="unknown in-tree Transolver preset"):
        build_transolver_intree("matched")


def test_intree_screened_arm_forward_capacity_and_kappa_dependence() -> None:
    """Exercise the screened-Laplace registry arm end to end.

    The screening scalar enters as one extra constant function channel on
    every token, so (a) the preset gains exactly ``2 * hidden_dim``
    parameters over the 2D bank (the widened first preprocess linear), and
    (b) the prediction must actually depend on ``global_data["screening"]``
    with the geometry and boundary data held fixed.
    """

    import screened_laplace

    from physicsnemo.mesh import DomainMesh

    torch.manual_seed(505)
    model = screened_laplace._build_model("transolver_intree_matched")
    assert parameter_count(model) == 102_989 + 2 * 64 == 103_117

    sample = screened_laplace.build_screened_sample(
        506, kappa_range=(0.5, 2.0), modes=(0, 1, 2, 3), n_boundary=16, n_query=8
    )
    output = model(sample.domain).point_data["potential"]
    assert output.shape == sample.target.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    rescreened = dict(sample.domain.global_data.items())
    rescreened["screening"] = 2.0 * rescreened["screening"]
    shifted = model(
        DomainMesh(
            interior=sample.domain.interior,
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=rescreened,
        )
    ).point_data["potential"]
    assert not torch.allclose(output, shifted)


def test_intree_3d_arm_forward_capacity_and_shell_merge() -> None:
    """Exercise the 3D-Laplace registry arm end to end.

    3D lifts the token contract to ``space_dim=3`` (3D coordinates, 6
    function channels), adding ``2 * (2 * hidden_dim)`` parameters over the
    2D bank; the shell tier must run through the same multi-boundary merge
    as the MeshTransformer arms.
    """

    import laplace3d_study
    from laplace3d import build_laplace3d_sample

    torch.manual_seed(507)
    model = laplace3d_study._build_model("transolver_intree_matched")
    assert parameter_count(model) == 102_989 + 4 * 64 == 103_245

    sphere = build_laplace3d_sample(508, tier="sphere", subdivisions=1, n_query=16)
    output = model(sphere.domain).point_data["potential"]
    assert output.shape == sphere.target.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    shell = build_laplace3d_sample(509, tier="shell", subdivisions=1, n_query=16)
    shell_output = model(shell.domain).point_data["potential"]
    assert shell_output.shape == shell.target.shape
    assert torch.isfinite(shell_output).all()
