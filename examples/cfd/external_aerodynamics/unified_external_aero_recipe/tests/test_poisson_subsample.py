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

"""Tests for ``PoissonBiasedSubsampleMesh`` (front/back bias and area modes).

The consistency benchmark (2026-09-10) adds ``weight_mode="area"``:
inclusion probability proportional to cell area, with exact
Horvitz--Thompson inverse-inclusion weights folded into the mesh measure.
These tests pin (a) the proportionality, (b) the HT measure fold, and
(c) that the default front/back mode is bit-identical to the pre-existing
implementation.
"""

from __future__ import annotations

import pytest
import torch
from domain_transforms import PoissonBiasedSubsampleMesh

from physicsnemo.datapipes.transforms.mesh.transforms import _compact_points
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import (
    cell_measures,
    scale_measures,
)


def _graded_plate(n: int = 50, ratio: float = 10.0) -> Mesh:
    """Right-triangle mesh of a plate with geometrically graded x-spacing.

    Cell areas vary by a factor ``ratio`` across the plate, so an
    area-proportional sampler and a uniform-over-cells sampler differ.
    """
    t = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float32)
    xs = (ratio**t - 1.0) / (ratio - 1.0)
    ys = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float32)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack(
        (gx.reshape(-1), gy.reshape(-1), torch.zeros_like(gx).reshape(-1)), dim=-1
    )
    idx = torch.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    v00 = idx[:-1, :-1].reshape(-1)
    v10 = idx[1:, :-1].reshape(-1)
    v01 = idx[:-1, 1:].reshape(-1)
    v11 = idx[1:, 1:].reshape(-1)
    cells = torch.cat(
        (
            torch.stack((v00, v10, v11), dim=-1),
            torch.stack((v00, v11, v01), dim=-1),
        ),
        dim=0,
    )
    return Mesh(points=points, cells=cells)


def _reference_front_back(mesh: Mesh, n_expected: int, bias: float, seed: int) -> Mesh:
    """The pre-``weight_mode`` implementation, verbatim, for bit-identity."""
    n = mesh.n_cells
    x = mesh.cell_centroids[:, 0]
    w = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    c = n_expected / w.sum()
    pi = (c * w).clamp(max=1.0)
    deficit = n_expected - pi.sum()
    if deficit > 0:
        free = pi < 1.0
        pi[free] = (pi[free] * (1 + deficit / pi[free].sum())).clamp(max=1.0)
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(n, generator=g) < pi
    indices = keep.nonzero(as_tuple=True)[0]
    out = _compact_points(mesh.slice_cells(indices))
    scale_measures(out, 1.0 / pi[indices])
    return out


def test_area_mode_inclusion_proportional_to_area():
    """Kept-cell HT weight times area is constant: pi_i = c * area_i exactly."""
    mesh = _graded_plate()
    n_expected = 500
    # No cell may clamp: c * max(area) < 1 with c = n_expected / total area.
    areas = mesh.cell_areas
    assert n_expected * areas.max() / areas.sum() < 1.0
    tf = PoissonBiasedSubsampleMesh(n_cells_expected=n_expected, weight_mode="area")
    tf.set_generator(torch.Generator().manual_seed(0))
    sub = tf(mesh)
    assert 0 < sub.n_cells < mesh.n_cells
    # weight_i = 1/pi_i = total_area / (n_expected * area_i)
    prod = cell_measures(sub)
    expected = areas.sum() / n_expected
    torch.testing.assert_close(
        prod, torch.full_like(prod, expected.item()), rtol=1e-4, atol=0.0
    )


def test_area_mode_ht_fold_preserves_measure_in_expectation():
    """Mean HT-estimated total area over draws is within 3% of the true area."""
    mesh = _graded_plate()
    true_total = mesh.cell_areas.sum().item()
    tf = PoissonBiasedSubsampleMesh(n_cells_expected=1000, weight_mode="area")
    totals = []
    counts = []
    for seed in range(100):
        tf.set_generator(torch.Generator().manual_seed(seed))
        sub = tf(mesh)
        totals.append(cell_measures(sub).sum().item())
        counts.append(sub.n_cells)
    mean_total = sum(totals) / len(totals)
    assert abs(mean_total / true_total - 1.0) < 0.03
    # Expected count is honored (Poisson count has relative sd ~ 1/sqrt(n)).
    mean_count = sum(counts) / len(counts)
    assert abs(mean_count / 1000 - 1.0) < 0.05


def test_area_mode_differs_from_uniform_over_cells():
    """Area weighting favours large cells: the kept fraction rises with area."""
    mesh = _graded_plate()
    tf = PoissonBiasedSubsampleMesh(n_cells_expected=1000, weight_mode="area")
    tf.set_generator(torch.Generator().manual_seed(1))
    sub = tf(mesh)
    areas = mesh.cell_areas
    small = areas < areas.median()
    # Fraction of kept cells that are "small" must be well below 1/2.
    kept_small = (sub.cell_areas < areas.median()).float().mean().item()
    assert kept_small < 0.4
    assert small.float().mean().item() == pytest.approx(0.5, abs=0.01)


def test_default_front_back_mode_is_bit_identical():
    """The default path reproduces the pre-``weight_mode`` implementation exactly."""
    mesh = _graded_plate()
    for seed in (0, 7):
        tf = PoissonBiasedSubsampleMesh(n_cells_expected=800, bias=10.0)
        tf.set_generator(torch.Generator().manual_seed(seed))
        got = tf(mesh)
        ref = _reference_front_back(mesh, 800, 10.0, seed)
        assert got.n_cells == ref.n_cells
        assert torch.equal(got.points, ref.points)
        assert torch.equal(got.cells, ref.cells)
        assert torch.equal(cell_measures(got), cell_measures(ref))


def test_invalid_weight_mode_rejected():
    """Reject unsupported sampling-weight modes."""
    with pytest.raises(ValueError, match="weight_mode"):
        PoissonBiasedSubsampleMesh(n_cells_expected=10, weight_mode="volume")
