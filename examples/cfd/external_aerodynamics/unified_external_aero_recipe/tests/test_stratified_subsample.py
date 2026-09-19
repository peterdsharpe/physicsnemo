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

"""StratifiedSubsampleMesh: the uniform-over-cells inclusion law of SubsampleMesh with a systematic
(Morton-ordered) draw. Exact count, constant weights N/n, no duplicates, inclusion frequency n/N for
every cell, and a lower variance than the independent draw for a smooth area-weighted integral."""

import sys
from pathlib import Path

import torch

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from domain_transforms import StratifiedSubsampleMesh  # noqa: E402

from physicsnemo.datapipes.transforms.mesh import SubsampleMesh  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import cell_measures  # noqa: E402


def _surface(n_cells: int, seed: int = 0) -> Mesh:
    """A random triangulated patch: n_cells small triangles scattered over the unit square in x-y."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(n_cells, 3, generator=g)
    centers[:, 2] = 0.0
    offsets = 0.01 * torch.randn(n_cells, 3, 3, generator=g)
    points = (centers[:, None, :] + offsets).reshape(-1, 3)
    return Mesh(points=points, cells=torch.arange(3 * n_cells).reshape(n_cells, 3))


def _draw(transform, mesh, seed):
    transform._generator = torch.Generator().manual_seed(seed)
    return transform(mesh)


def test_count_weights_and_no_duplicates():
    """A fixed-size draw represents each retained cell with its corrected measure."""
    mesh = _surface(4000)
    sub = _draw(StratifiedSubsampleMesh(n_cells=400), mesh, 1)
    assert sub.n_cells == 400
    w = cell_measures(sub) / sub.cell_areas
    assert torch.allclose(w, torch.full_like(w, 4000 / 400))


def test_inclusion_frequency_is_n_over_N():
    """Every cell has the requested inclusion frequency across repeated draws."""
    mesh = _surface(200)
    t = StratifiedSubsampleMesh(n_cells=20, compact=False)
    counts = torch.zeros(200)
    draws = 600
    for s in range(draws):
        t._generator = torch.Generator().manual_seed(s)
        order = t._morton_order(mesh.cell_centroids)
        step = 200 / 20
        offset = torch.rand((), generator=t._generator) * step
        pos = torch.floor(offset + step * torch.arange(20, dtype=torch.float64)).long()
        counts[order[pos]] += 1
    expected = draws * 20 / 200
    assert counts.sum() == draws * 20
    assert (counts - expected).abs().max() <= 0.3 * expected


def test_systematic_draw_has_lower_variance_than_independent_for_a_smooth_integral():
    """Corrected stratified estimates reduce variance for a smooth integrand."""
    mesh = _surface(5000)
    f = torch.sin(2 * torch.pi * mesh.cell_centroids[:, 0]) * torch.cos(
        2 * torch.pi * mesh.cell_centroids[:, 1]
    )
    areas = mesh.cell_areas
    truth = float((f * areas).sum())

    def estimate(transform, seed):
        sub = _draw(transform, mesh, seed)
        fs = torch.sin(2 * torch.pi * sub.cell_centroids[:, 0]) * torch.cos(
            2 * torch.pi * sub.cell_centroids[:, 1]
        )
        return float((fs * cell_measures(sub)).sum())

    strat = torch.tensor(
        [estimate(StratifiedSubsampleMesh(n_cells=250), s) for s in range(40)]
    )
    iid = torch.tensor([estimate(SubsampleMesh(n_cells=250), s) for s in range(40)])
    assert abs(float(strat.mean()) - truth) < 3 * float(strat.std()) / 40**0.5 + 1e-3
    assert float(strat.var()) < 0.5 * float(iid.var())
