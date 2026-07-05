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

"""Tests for the MeshTransformer scale-ladder study."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import scale_study  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cost harness measures CUDA wall/memory"
)


def test_power_law_fit_recovers_planted_exponents() -> None:
    """The log-space fit must recover exact exponents on synthetic data."""

    points = [
        (q, s, 2.5 * q**1.0 * s**1.0)
        for q in (1.0e3, 1.0e4, 1.0e5)
        for s in (320.0, 1280.0, 5120.0)
    ]
    fit = scale_study.fit_power_law(points)
    assert fit is not None
    assert math.isclose(fit["query_exponent"], 1.0, abs_tol=1.0e-9)
    assert math.isclose(fit["source_exponent"], 1.0, abs_tol=1.0e-9)
    assert math.isclose(fit["coefficient"], 2.5, rel_tol=1.0e-9)
    assert fit["r_squared"] > 1.0 - 1.0e-12

    single = scale_study.fit_power_law_sources([(s, 0.7 * s**2.0) for s in (2, 4, 8)])
    assert single is not None
    assert math.isclose(single["source_exponent"], 2.0, abs_tol=1.0e-9)


def test_power_law_fit_refuses_degenerate_grids() -> None:
    """One-point (smoke) grids must yield no fit, never fake exponents."""

    assert scale_study.fit_power_law([(64.0, 80.0, 1.0)]) is None
    # Variation in only one axis is still unidentifiable in the other.
    assert scale_study.fit_power_law([(64.0, 80.0, 1.0), (128.0, 80.0, 2.0)]) is None
    assert scale_study.fit_power_law_sources([(80.0, 1.0)]) is None


def test_scale_model_is_singpair_with_one_encoder_layer() -> None:
    """The measured architecture: 2 exact members, 1 operator layer."""

    model = scale_study.build_scale_model()
    inner = model.model
    assert len(inner.operator_blocks) == 1
    decoder = inner.kernel_decoder
    assert decoder.include_single_layer_member
    assert not decoder.include_polynomial_members
    assert decoder.mlp_members == 0
    assert decoder.n_members == 2


@requires_cuda
def test_cost_harness_single_point_writes_report(tmp_path: Path) -> None:
    """One tiny grid point runs on CUDA and the report JSON is written."""

    report = scale_study.run_study(
        mode="cost",
        output_dir=str(tmp_path),
        device="cuda",
        seed=5,
        subdivisions=(1,),  # 80 triangles
        queries=(64,),
        warmup=0,
        repeats=1,
    )
    path = tmp_path / "scale_study_cost_seed5.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["model"] == "mesh_transformer_kernel_singpair_enc1"
    (point,) = data["cost"]["grid"]
    assert point["status"] == "ok"
    assert point["n_boundary_cells"] == 80
    assert point["n_query"] == 64
    for key in ("forward_ms", "encode_ms", "decode_ms", "forward_backward_ms"):
        assert point[key]["median_ms"] > 0.0
    assert point["forward_peak_bytes"] > 0
    assert point["train_peak_bytes"] >= point["forward_peak_bytes"]
    # A single point cannot support exponent fits; the report says so.
    assert data["cost"]["scaling_fits"]["decode_ms"] is None
    assert report["cost"]["extrapolation"]["what_breaks_first"]
