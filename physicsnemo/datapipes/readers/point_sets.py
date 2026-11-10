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

"""Independent point-set selections for domain mesh readers."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures


def sample_point_sets(
    domain: DomainMesh,
    specifications: dict[str, dict],
    *,
    seed: int,
    epoch: int,
    index: int,
) -> DomainMesh:
    """Gather configured point sets without materializing complete memmap fields.

    See ``DomainMeshReader.point_sets`` for the configuration contract. Sources
    refer to original meshes unless a preceding output has that name. Shared
    indices require the identical source mesh and association.
    """
    outputs = {}
    selections = {}
    for name, spec in specifications.items():
        source = spec["source"]
        mesh = outputs.get(source)
        if mesh is None:
            mesh = (
                domain.interior if source == "interior" else domain.boundaries[source]
            )
        association = spec.get("association", "points")
        if association not in ("points", "cell_centroids"):
            raise ValueError(
                "point-set association must be 'points' or 'cell_centroids'"
            )
        count = spec.get("count")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise ValueError("point-set count must be a positive integer or None")
        stream = spec["stream"]
        if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
            raise ValueError("point-set stream must be a nonnegative integer")
        total = mesh.n_points if association == "points" else mesh.n_cells
        if total == 0:
            raise ValueError(f"Point-set source {source!r} has no {association}")
        shared = spec.get("indices_from")
        if shared is not None:
            previous_mesh, previous_association, ids = selections[shared]
            if mesh is not previous_mesh or association != previous_association:
                raise ValueError(
                    "indices_from must reference the same source and association"
                )
        elif count is None or count >= total:
            ids = torch.arange(total)
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, index, stream])
            )
            ids = torch.from_numpy(np.sort(rng.choice(total, count, replace=False)))
        selections[name] = mesh, association, ids
        if association == "points":
            points = mesh.points[ids].clone()
            data = mesh.point_data
        else:
            points = mesh.points[mesh.cells[ids].long()].mean(dim=1)
            data = mesh.cell_data
        fields = TensorDict({}, batch_size=[len(ids)])
        for src, dst in spec.get("fields", {}).items():
            fields.set(as_nested_key(dst), data[as_nested_key(src)][ids].clone())
        outputs[name] = Mesh(points=points, point_data=fields)
        retained = spec.get("retain_cells_as")
        if retained is not None:
            if association != "cell_centroids":
                raise ValueError("retain_cells_as requires cell_centroids")
            if (
                retained == "interior"
                or retained in outputs
                or retained in specifications
            ):
                raise ValueError("retain_cells_as must name a distinct output boundary")
            vertices, cells = torch.unique(
                mesh.cells[ids].long(), sorted=True, return_inverse=True
            )
            selected = Mesh(
                points=mesh.points[vertices].clone(), cells=cells.reshape(len(ids), -1)
            )
            measures = (
                mesh.cell_data["_effective_measure"][ids].clone()
                if "_effective_measure" in mesh.cell_data
                else cell_measures(selected)
            )
            set_cell_measures(selected, measures * (total / len(ids)))
            outputs[retained] = selected
    interior = outputs.pop("interior", None)
    if interior is None:
        interior = Mesh(
            points=next(iter(outputs.values())).points.new_empty(
                (0, domain.interior.n_spatial_dims)
            )
        )
    return DomainMesh(
        interior=interior, boundaries=outputs, global_data=domain.global_data.clone()
    )
