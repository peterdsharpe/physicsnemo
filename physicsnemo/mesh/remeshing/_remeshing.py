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

"""Main remeshing entry point.

This module wires together all components of the remeshing pipeline.
"""

import math
from typing import TYPE_CHECKING

import torch

from physicsnemo.core.version_check import OptionalImport, require_version_spec

### Optional dependency. ``pyacvd`` is a lazy proxy: construction does not
### import the package; the friendly ``ImportError`` (with the
### ``[mesh-extras]`` install hint) fires only on first attribute access. The
### ``@require_version_spec("pyacvd")`` decorator on ``remesh`` raises that
### same error proactively before any function-body work happens.
if TYPE_CHECKING:
    import pyacvd

    from physicsnemo.mesh.mesh import Mesh
else:
    pyacvd = OptionalImport("pyacvd")


@require_version_spec("pyacvd")
def remesh(
    mesh: "Mesh",
    n_clusters: int,
) -> "Mesh":
    """Uniform remeshing of a 2D triangle surface via clustering.

    Creates a simplified mesh with approximately ``n_clusters`` vertices
    uniformly distributed across the geometry. Uses the ACVD (Approximate
    Centroidal Voronoi Diagram) clustering algorithm.

    The algorithm:
    1. Weights vertices by their dual volumes (Voronoi areas)
    2. Initializes clusters via area-based region growing
    3. Minimizes energy by iteratively reassigning vertices
    4. Reconstructs a simplified mesh from cluster adjacency

    This is restricted to 2D triangle surfaces embedded in 2D or 3D space --
    the cases the underlying ``pyacvd`` ACVD clustering supports.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to remesh
    n_clusters : int
        Target number of output vertices. The actual number may vary
        slightly depending on mesh topology.

    Returns
    -------
    Mesh
        Remeshed mesh with approximately ``n_clusters`` vertices. The vertices are
        cluster centroids, and cells connect adjacent clusters.

    Raises
    ------
    NotImplementedError
        If the mesh is not a 2D triangle surface embedded in 2D or 3D.
    ValueError
        If ``n_clusters`` is less than three, the mesh is empty, or its geometry
        has zero or non-finite extent.
    ImportError
        If the optional ``pyacvd`` dependency is not installed.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> from physicsnemo.mesh.remeshing import remesh
    >>> mesh = sphere_icosahedral.load(subdivisions=3)
    >>> # Remesh a triangle mesh to approximately 100 cluster centroids
    >>> simplified = remesh(mesh, n_clusters=100)
    >>> assert simplified.n_cells > 0

    Notes
    -----
    - Restricted to 2D triangle surfaces embedded in 2D or 3D
    - Preserves mesh topology qualitatively but not quantitatively
    - Point and cell data are not transferred (topology changes fundamentally)
    - Global data is preserved because it is independent of mesh topology
    - Output cell orientation may differ from input
    """
    from physicsnemo.mesh.io.io_pyvista import to_pyvista
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.repair import repair_mesh

    # pyacvd ACVD clustering is a triangle-surface algorithm: it only handles a
    # PolyData of triangles. PyVista represents planar 2D inputs by padding a
    # zero third coordinate, so both 2D and 3D embedding spaces are valid.
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims not in (2, 3):
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 2D or 3D "
            "(the pyacvd ACVD clustering is surface-only). Got "
            f"n_manifold_dims={mesh.n_manifold_dims}, "
            f"n_spatial_dims={mesh.n_spatial_dims}."
        )
    if n_clusters < 3:
        raise ValueError(f"n_clusters must be at least 3, got {n_clusters=}")
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("Cannot remesh an empty mesh.")

    # PyVista exports reduced precision and pyacvd runs on CPU. Centering and
    # scaling before that boundary prevents float32 from collapsing translated
    # float64 points or under-resolving micro-scale geometry. Denormalization is
    # performed back in the input precision below.
    working_dtype = (
        torch.float64 if mesh.points.dtype == torch.float64 else torch.float32
    )
    working_points = mesh.points.detach().to(working_dtype)
    lower = working_points.amin(dim=0)
    upper = working_points.amax(dim=0)
    extent = upper - lower
    center = lower + extent / 2
    scale = extent.amax()
    scale_value = scale.item()
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("Cannot remesh geometry with zero or non-finite extent.")

    normalized = Mesh(
        points=(working_points - center) / scale,
        cells=mesh.cells,
    )
    clustering = pyacvd.Clustering(to_pyvista(normalized))
    clustering.cluster(n_clusters)
    clustered = clustering.create_mesh()

    normalized_points = torch.from_numpy(clustered.points.copy())[
        :, : mesh.n_spatial_dims
    ].to(dtype=working_dtype)
    cells = torch.from_numpy(clustered.regular_faces.copy()).long()

    # Keep the existing cleanup guarantees. Since this geometry has unit
    # extent, repair_mesh's absolute defaults now act as relative tolerances
    # instead of accidentally depending on the input's physical units.
    cleaned, _stats = repair_mesh(Mesh(points=normalized_points, cells=cells))
    points = (
        cleaned.points.to(device=mesh.points.device, dtype=working_dtype) * scale
        + center
    ).to(mesh.points.dtype)

    return Mesh(
        points=points,
        cells=cleaned.cells.to(mesh.cells.device),
        global_data=mesh.global_data,
    )
