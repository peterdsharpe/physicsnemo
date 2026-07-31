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

"""RandomResolutionSubsampleMesh: per-sample draws with unbiased measure."""

import pytest
import torch

from domain_transforms import RandomResolutionSubsampleMesh
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures


def _grid_mesh(n_side: int = 40) -> Mesh:
    """A flat triangulated grid with n_side^2 * 2 cells of equal area."""
    xs = torch.linspace(0.0, 1.0, n_side + 1)
    px, py = torch.meshgrid(xs, xs, indexing="ij")
    points = torch.stack(
        (px.flatten(), py.flatten(), torch.zeros_like(px.flatten())), dim=-1
    )
    cells = []
    for i in range(n_side):
        for j in range(n_side):
            a = i * (n_side + 1) + j
            b = a + 1
            c = a + n_side + 1
            d = c + 1
            cells.append([a, b, c])
            cells.append([b, d, c])
    return Mesh(points=points, cells=torch.tensor(cells, dtype=torch.long))


class TestRandomResolutionSubsampleMesh:
    def test_draws_only_declared_choices_and_covers_them(self):
        transform = RandomResolutionSubsampleMesh(n_cells_choices=[100, 200, 400])
        seen = set()
        for _ in range(40):
            out = transform(_grid_mesh(20))  # 800 cells
            assert out.n_cells in (100, 200, 400)
            seen.add(out.n_cells)
        assert seen == {100, 200, 400}

    def test_measure_total_is_unbiased_at_every_draw(self):
        """HT weights must make total measure match the full mesh exactly."""
        full = _grid_mesh(20)
        total = cell_measures(full).sum()
        transform = RandomResolutionSubsampleMesh(n_cells_choices=[100, 400])
        for _ in range(10):
            out = transform(_grid_mesh(20))
            torch.testing.assert_close(
                cell_measures(out).sum(), total, rtol=1e-5, atol=0.0
            )

    def test_rejects_degenerate_choice_lists(self):
        with pytest.raises(ValueError, match="at least two positive"):
            RandomResolutionSubsampleMesh(n_cells_choices=[100])
        with pytest.raises(ValueError, match="at least two positive"):
            RandomResolutionSubsampleMesh(n_cells_choices=[100, -5])
