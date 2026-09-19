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

"""Tests for ComposeQuadratureMeasure (HT-corrected boundary measure for the total-measure scale, 2026-09-13)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain_transforms import ComposeQuadratureMeasure  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import scale_measures  # noqa: E402


def _surface(n_tri=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(3 * n_tri, 3, generator=g)
    cells = torch.arange(3 * n_tri).reshape(n_tri, 3)
    return Mesh(points=pts, cells=cells)


def test_sum_recovers_full_area_after_subsample_factor():
    """Retain the complete surface area after the sampling correction."""
    mesh = _surface()
    full_area = float(mesh.cell_areas.sum())
    keep = torch.arange(0, mesh.n_cells, 4)  # 1-in-4 subsample
    sub = mesh.slice_cells(keep)
    scale_measures(sub, mesh.n_cells / keep.numel())
    out = ComposeQuadratureMeasure()(sub)
    q = out.cell_data["_effective_measure"]
    assert q.shape == (keep.numel(),)
    assert torch.allclose(q, sub.cell_areas * (mesh.n_cells / keep.numel()))
    # unbiased estimate of the full area: within the sampling noise of a 1-in-4 draw
    assert abs(float(q.sum()) - full_area) / full_area < 0.25
    assert (
        float(sub.cell_areas.sum()) / full_area < 0.35
    )  # the raw areas are a quarter of the body


def test_identity_without_subsampling_and_point_cloud_passthrough():
    """Materialize geometric cell measures without inventing point measures."""
    mesh = _surface()
    out = ComposeQuadratureMeasure()(mesh)
    assert torch.allclose(out.cell_data["_effective_measure"], mesh.cell_areas)
    cloud = Mesh(points=torch.rand(50, 3))
    assert ComposeQuadratureMeasure()(cloud) is cloud
