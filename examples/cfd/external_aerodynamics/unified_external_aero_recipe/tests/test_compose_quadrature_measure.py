# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ComposeQuadratureMeasure (HT-corrected boundary measure for the total-measure scale, 2026-09-13)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain_transforms import ComposeQuadratureMeasure  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import compose_measure_weights  # noqa: E402


def _surface(n_tri=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(3 * n_tri, 3, generator=g)
    cells = torch.arange(3 * n_tri).reshape(n_tri, 3)
    return Mesh(points=pts, cells=cells)


def test_sum_recovers_full_area_after_subsample_factor():
    mesh = _surface()
    full_area = float(mesh.cell_areas.sum())
    keep = torch.arange(0, mesh.n_cells, 4)  # 1-in-4 subsample
    sub = mesh.slice_cells(keep)
    compose_measure_weights(sub, mesh.n_cells / keep.numel())
    out = ComposeQuadratureMeasure()(sub)
    q = out.cell_data["quadrature_measure"]
    assert q.shape == (keep.numel(),)
    assert torch.allclose(q, sub.cell_areas * (mesh.n_cells / keep.numel()))
    # unbiased estimate of the full area: within the sampling noise of a 1-in-4 draw
    assert abs(float(q.sum()) - full_area) / full_area < 0.25
    assert float(sub.cell_areas.sum()) / full_area < 0.35  # the raw areas are a quarter of the body


def test_identity_without_subsampling_and_point_cloud_passthrough():
    mesh = _surface()
    out = ComposeQuadratureMeasure()(mesh)
    assert torch.allclose(out.cell_data["quadrature_measure"], mesh.cell_areas)
    cloud = Mesh(points=torch.rand(50, 3))
    assert ComposeQuadratureMeasure()(cloud) is cloud
