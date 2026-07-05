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

"""Contracts for the 3D all-Dirichlet study driver's report shape.

The model/quadrature mathematics is covered by the studies that consume this
driver; these tests pin the *reporting* contract -- most recently the
additive operator-fidelity block (per-split strong-form residual and sampled
maximum-principle violation)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from laplace3d_study import (  # noqa: E402
    SPLITS_3D,
    _build_model,
    fidelity_metrics,
    run_experiment,
)


def test_driver_smoke_reports_per_split_fidelity(tmp_path: Path) -> None:
    """A zero-step CPU run writes finite splits plus the fidelity block.

    ``SolidAngleBIE`` is harmonic by construction (a learned multiple of the
    exact solid-angle double layer), so its per-split strong-form residual
    pins the diagnostic's zero even untrained, while its untrained
    coefficients need not respect the sampled boundary range -- the
    maximum-principle entries are only required finite and non-negative.
    """

    report = run_experiment(
        model_name="solid_angle_bie",
        steps=0,
        seed=3,
        output_dir=str(tmp_path),
        device="cpu",
        eval_cases=1,
    )
    on_disk = json.loads((tmp_path / "solid_angle_bie_seed3.json").read_text())
    for payload in (report, on_disk):
        assert set(payload["splits"]) == set(SPLITS_3D)
        for value in payload["splits"].values():
            assert math.isfinite(value)
        fidelity = payload["fidelity"]
        assert set(fidelity["pde_residual"]) == set(SPLITS_3D)
        assert all(abs(v) < 1.0e-9 for v in fidelity["pde_residual"].values())
        assert set(fidelity["max_principle_violation"]) == set(SPLITS_3D)
        for value in fidelity["max_principle_violation"].values():
            assert math.isfinite(value)
            assert value >= 0.0


def test_fidelity_block_covers_learned_transformer_arm() -> None:
    """The autograd residual runs through the full 3D kernel-decoder arm.

    Coarse single-split spec keeps the double-backward cheap; at random
    initialization the residual must simply be finite (harmonicity is
    diagnosed, not enforced).
    """

    import torch

    torch.manual_seed(0)
    model = _build_model("mesh_transformer_kernel_singpair")
    coarse = {
        "sphere": {
            "tier": "sphere",
            "bc_regime": "dirichlet",
            "subdivisions": 1,
            "n_query": 16,
        }
    }
    block = fidelity_metrics(model, splits=coarse, seed=83_000_019, device="cpu")
    assert math.isfinite(block["pde_residual"]["sphere"])
    assert math.isfinite(block["max_principle_violation"]["sphere"])
    assert block["max_principle_violation"]["sphere"] >= 0.0
