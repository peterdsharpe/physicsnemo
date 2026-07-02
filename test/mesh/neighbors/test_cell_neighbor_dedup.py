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

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_tetrahedra


def test_cell_to_cells_deduplicates_pairs_from_multiple_shared_edges() -> None:
    mesh = two_tetrahedra.load()

    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=2)

    assert adjacency.to_list() == [[1], [0]]


def test_cell_to_cells_handles_three_cells_sharing_one_facet(device) -> None:
    """Every retained shared-facet group may be used without a second filter."""
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.5, 1.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ],
        device=device,
    )
    cells = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [5, 6, 7],
        ],
        device=device,
    )
    mesh = Mesh(points=points, cells=cells)

    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)

    assert torch.equal(adjacency.offsets, cells.new_tensor([0, 2, 4, 6, 6]))
    assert torch.equal(adjacency.indices, cells.new_tensor([1, 2, 0, 2, 0, 1]))
