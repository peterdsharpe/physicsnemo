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

r"""Three-dimensional Laplace benchmark: varied geometry, topology, and BCs.

Exact labels on *arbitrary* domains come from exterior point charges: for
source points :math:`y_j` strictly outside :math:`\Omega`,

.. math::

   u(x) = \sum_j \frac{c_j}{4\pi\,\lVert x - y_j\rVert}

is exactly harmonic inside :math:`\Omega`, and every boundary-condition type
(Dirichlet trace, Neumann flux :math:`n\cdot\nabla u`, Robin
:math:`u + \beta L\,\partial_n u` with dimensionless :math:`\beta`) is
analytic.  This replaces the 2D conformal trick and works for deformed and
multiply connected geometries alike.

Geometry tiers (shape *and* connectivity variation):

- ``sphere``: random radius, center, and mesh orientation.
- ``star``: star-shaped radial perturbations
  :math:`r(\hat u) = R\,(1 + \sum \varepsilon\, Y_{lm}(\hat u))`,
  homeomorphic to the sphere but with varied curvature.
- ``shell``: spherical shells (outer boundary + inner boundary), multiply
  connected; the two boundaries may carry different BC types.

Boundary-condition regimes (always well-posed: at least one boundary carries
Dirichlet or Robin data; all-Neumann is excluded because it is nonunique up
to constants):

- ``dirichlet``: Dirichlet everywhere.
- ``mixed``: Dirichlet on one boundary, Neumann on the other; single-boundary
  tiers fall back to all-Dirichlet so the regime stays well-posed (a per-cell
  hemisphere split is future work).
- ``robin``: Robin everywhere with per-boundary dimensionless coefficient
  :math:`\beta \in [0.2, 2]` (positive: well-posed).

Meshes are subdivided icosahedra built directly as PhysicsNeMo ``Mesh``
objects (triangles embedded in 3D); ``cell_areas``, ``cell_normals``, and
``cell_centroids`` come from the Mesh module -- this file is also a test of
its dimensional genericity.  This is a benchmark-local research asset, not a
proposed public API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from physicsnemo.mesh import DomainMesh, Mesh

_FOUR_PI = 4.0 * math.pi


def icosphere(subdivisions: int = 2, dtype: torch.dtype = torch.float64):
    """Return unit-sphere vertices and outward-oriented triangle indices."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, phi, 0),
        (1, phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1, phi),
        (0, 1, phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    points = torch.tensor(vertices, dtype=dtype)
    points = points / points.norm(dim=-1, keepdim=True)
    cells = torch.tensor(faces, dtype=torch.long)
    for _ in range(subdivisions):
        edge_midpoint: dict[tuple[int, int], int] = {}
        new_points = [points]
        new_cells = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in edge_midpoint:
                mid = points[a] + points[b]
                mid = mid / mid.norm()
                new_points.append(mid[None, :])
                edge_midpoint[key] = sum(p.shape[0] for p in new_points) - 1
            return edge_midpoint[key]

        for a, b, c in cells.tolist():
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_cells.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        points = torch.cat(new_points, dim=0)
        cells = torch.tensor(new_cells, dtype=torch.long)
    return points, cells


def _real_spherical_harmonic(degree: int, order: int, unit: torch.Tensor):
    """Small set of low-degree real spherical harmonics (unnormalized)."""

    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    table = {
        (2, 0): 3.0 * z * z - 1.0,
        (2, 1): x * z,
        (2, 2): x * x - y * y,
        (3, 0): z * (5.0 * z * z - 3.0),
        (3, 2): z * (x * x - y * y),
        (4, 0): 35.0 * z**4 - 30.0 * z * z + 3.0,
        (4, 3): x * z * (x * x - 3.0 * y * y),
    }
    return table[(degree, order)]


_STAR_MODES = ((2, 0), (2, 1), (2, 2), (3, 0), (3, 2))
_STAR_MODES_OOD = ((4, 0), (4, 3))


@dataclass(frozen=True)
class Laplace3DSample:
    """One exact 3D Laplace problem with named boundaries and BC roles."""

    domain: DomainMesh
    target: torch.Tensor
    bc_types: dict[str, str]  # boundary name -> {dirichlet, neumann, robin}
    tier: str


def _potential_and_gradient(
    points: torch.Tensor, charges: torch.Tensor, positions: torch.Tensor
):
    displacement = points[:, None, :] - positions[None, :, :]
    distance = displacement.norm(dim=-1)
    potential = (charges[None, :] / (_FOUR_PI * distance)).sum(dim=-1)
    gradient = -(
        charges[None, :, None] * displacement / (_FOUR_PI * distance.pow(3)[..., None])
    ).sum(dim=1)
    return potential, gradient


def _surface_mesh(
    base_radius: float,
    center: torch.Tensor,
    rotation: torch.Tensor,
    star: list[tuple[tuple[int, int], float]],
    *,
    subdivisions: int,
    inward: bool,
    dtype: torch.dtype,
) -> Mesh:
    unit, cells = icosphere(subdivisions, dtype=torch.float64)
    radial = torch.full((unit.shape[0],), float(base_radius), dtype=torch.float64)
    for (degree, order), amplitude in star:
        radial = radial + base_radius * amplitude * _real_spherical_harmonic(
            degree, order, unit
        )
    points = (radial[:, None] * unit) @ rotation.T + center
    if inward:
        cells = cells[:, [0, 2, 1]]  # flip orientation: normals point inward
    return Mesh(points=points.to(dtype), cells=cells)


def build_laplace3d_sample(
    seed: int,
    *,
    tier: str = "sphere",
    bc_regime: str = "dirichlet",
    subdivisions: int = 2,
    n_query: int = 200,
    star_modes: tuple = _STAR_MODES,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Laplace3DSample:
    """Sample one exact problem from a (tier, BC-regime) cell of the suite."""

    if tier not in ("sphere", "star", "shell"):
        raise ValueError("tier must be sphere, star, or shell")
    if bc_regime not in ("dirichlet", "mixed", "robin"):
        raise ValueError("bc_regime must be dirichlet, mixed, or robin")
    generator = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float, shape=()):
        return torch.empty(shape, dtype=torch.float64).uniform_(
            low, high, generator=generator
        )

    radius = float(uniform(0.7, 1.5))
    center = uniform(-1.0, 1.0, (3,))
    axis = torch.nn.functional.normalize(uniform(-1.0, 1.0, (3,)), dim=0)
    angle = float(uniform(0.0, 2.0 * math.pi))
    K = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    rotation = (
        torch.eye(3, dtype=torch.float64)
        + math.sin(angle) * K
        + (1.0 - math.cos(angle)) * (K @ K)
    )

    star: list[tuple[tuple[int, int], float]] = []
    if tier == "star":
        for mode in star_modes:
            amplitude = float(uniform(-0.06, 0.06))
            star.append((mode, amplitude))

    boundaries: dict[str, Mesh] = {}
    boundaries["outer"] = _surface_mesh(
        radius,
        center,
        rotation,
        star,
        subdivisions=subdivisions,
        inward=False,
        dtype=torch.float64,
    )
    # Empirical radial extent of the actual surface (exact for the mesh,
    # rather than an analytic harmonic bound): governs both the manufactured
    # source stand-off and the interior-query acceptance window.
    vertex_radii = (boundaries["outer"].points - center).norm(dim=-1)
    max_radius = float(vertex_radii.max())
    min_radius = float(vertex_radii.min())
    inner_radius = 0.0
    if tier == "shell":
        inner_radius = radius * float(uniform(0.35, 0.55))
        boundaries["inner"] = _surface_mesh(
            inner_radius,
            center,
            rotation,
            [],
            subdivisions=max(subdivisions - 1, 1),
            inward=True,
            dtype=torch.float64,
        )

    # Exterior manufactured sources: outside the outer surface (with margin
    # covering star excursions), plus -- for shells -- inside the cavity, so
    # the solution has genuinely multiply-connected structure.
    n_out = 6
    directions = torch.nn.functional.normalize(uniform(-1.0, 1.0, (n_out, 3)), dim=-1)
    source_radius = 1.5 * max_radius
    positions = center + source_radius * directions
    charges = uniform(-1.0, 1.0, (n_out,))
    if tier == "shell":
        cavity = center + 0.4 * inner_radius * torch.nn.functional.normalize(
            uniform(-1.0, 1.0, (2, 3)), dim=-1
        )
        positions = torch.cat((positions, cavity), dim=0)
        charges = torch.cat((charges, uniform(-1.0, 1.0, (2,))), dim=0)
    # Normalize amplitude so targets are O(1) across radii.
    charges = charges * (_FOUR_PI * radius)

    # Interior queries by rejection against the empirical radial extent of
    # the sampled surface (rotation preserves norms, so local-frame radii are
    # directly comparable to the world-frame vertex radii about the center).
    lo = inner_radius * 1.15 if tier == "shell" else 0.0
    hi = 0.92 * min_radius
    queries = []
    while sum(q.shape[0] for q in queries) < n_query:
        candidate = uniform(-1.0, 1.0, (4 * n_query, 3)) * hi
        r = candidate.norm(dim=-1)
        keep = (r < hi) & (r > (lo if lo else -1.0))
        queries.append(candidate[keep])
    local = torch.cat(queries, dim=0)[:n_query]
    query_points = local @ rotation.T + center

    target, _ = _potential_and_gradient(query_points, charges, positions)

    bc_types: dict[str, str] = {}
    names = sorted(boundaries)
    for index, name in enumerate(names):
        if bc_regime == "dirichlet":
            bc_types[name] = "dirichlet"
        elif bc_regime == "robin":
            bc_types[name] = "robin"
        else:
            # Mixed: exactly one boundary keeps Dirichlet for well-posedness.
            bc_types[name] = "dirichlet" if index == 0 else "neumann"
    if bc_regime == "mixed" and len(names) == 1:
        bc_types[names[0]] = "dirichlet"  # single boundary: stay well-posed

    out_boundaries: dict[str, Mesh] = {}
    for name, mesh in boundaries.items():
        centroids = mesh.cell_centroids
        normals = mesh.cell_normals
        value, gradient = _potential_and_gradient(centroids, charges, positions)
        flux = torch.einsum("nd,nd->n", normals, gradient)
        cell_data: dict[str, torch.Tensor] = {}
        if bc_types[name] == "dirichlet":
            cell_data["boundary_value"] = value
        elif bc_types[name] == "neumann":
            cell_data["boundary_flux"] = flux * radius  # dimensionless L * du/dn
        else:
            beta = float(uniform(0.2, 2.0))
            cell_data["robin_value"] = value + beta * radius * flux
            cell_data["robin_beta"] = torch.full_like(value, beta)
        out_boundaries[name] = mesh.with_data(
            cell_data={
                k: v.to(device=device, dtype=dtype) for k, v in cell_data.items()
            }
        )

    domain = DomainMesh(
        interior=Mesh(points=query_points.to(device=device, dtype=dtype)),
        boundaries={
            k: Mesh(
                points=v.points.to(device=device, dtype=dtype),
                cells=v.cells.to(device),
                cell_data=dict(v.cell_data.items()),
            )
            for k, v in out_boundaries.items()
        },
        global_data={
            "reference_length": torch.tensor(radius, device=device, dtype=dtype)
        },
    )
    return Laplace3DSample(
        domain=domain,
        target=target.to(device=device, dtype=dtype),
        bc_types=bc_types,
        tier=tier,
    )


def solid_angle_influence(
    query_points: torch.Tensor,
    triangle_vertices: torch.Tensor,
) -> torch.Tensor:
    r"""Exact double-layer triangle integrals via van Oosterom--Strackee.

    Entry ``(i, j)`` is :math:`\int_{T_j} n\cdot(y - x_i)/(4\pi\lVert y -
    x_i\rVert^3)\,dS_y = -\Omega_{ij}/(4\pi)` with :math:`\Omega` the signed
    solid angle subtended by triangle ``j`` at ``x_i`` (outward normals).
    """

    a = triangle_vertices[None, :, 0, :] - query_points[:, None, :]
    b = triangle_vertices[None, :, 1, :] - query_points[:, None, :]
    c = triangle_vertices[None, :, 2, :] - query_points[:, None, :]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    numerator = torch.einsum("qsd,qsd->qs", a, torch.cross(b, c, dim=-1))
    denominator = (
        la * lb * lc
        + torch.einsum("qsd,qsd->qs", a, b) * lc
        + torch.einsum("qsd,qsd->qs", b, c) * la
        + torch.einsum("qsd,qsd->qs", c, a) * lb
    )
    return -2.0 * torch.atan2(numerator, denominator) / _FOUR_PI


__all__ = [
    "Laplace3DSample",
    "build_laplace3d_sample",
    "icosphere",
    "solid_angle_influence",
]
