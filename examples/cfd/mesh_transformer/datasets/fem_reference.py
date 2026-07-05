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

r"""Traditional-numerical-solver reference for the boundary-to-interior bank.

Every previous ground truth in this example is a *manufactured* solution
(harmonic polynomials pulled through certified conformal maps, Liouville
metric potentials, ...), so the reachable set of boundary traces is exactly
the reachable set of the construction.  This module removes that coupling: a
plain 2D finite-element solver produces interior values for **arbitrary**
Dirichlet traces on **arbitrary** polygonal (possibly multiply-connected)
domains, so benchmark datasets can carry solver-verified targets whose
boundary conditions are *not* induced by any harmonic construction.

Scope (deliberately narrow, this is a reference generator, not a solver
library):

- Equations: Laplace :math:`-\Delta u = 0` and the screened variant
  :math:`-\Delta u + \kappa^2 u = 0` (both homogeneous, Dirichlet only).
- Geometry: one outer closed polyline loop plus zero or more inner hole
  loops.  Meshing uses the ``triangle`` package (Shewchuk's Triangle):
  constrained Delaunay with a global area constraint and a 30-degree
  minimum-angle quality bound (flags ``pq30a<area>``); holes are marked via
  interior seed points obtained by triangulating each hole loop alone.
- Elements: straight P2 (quadratic Lagrange) triangles.  Stiffness and mass
  matrices are assembled with a 6-point order-4 Dunavant rule (exact for the
  quartic mass integrand; the quadratic stiffness integrand needs only
  order 2), Dirichlet conditions are imposed by elimination, and the reduced
  system is solved with ``scipy.sparse.linalg.spsolve``.
- Query evaluation: point location with
  ``matplotlib.tri.TrapezoidMapTriFinder`` on the P1 sub-triangulation,
  followed by exact P2 shape-function evaluation at the barycentric
  coordinates.  Queries that fall marginally outside the polygonal domain
  (inscribed polygons of smooth curves leave a sliver of the true domain
  uncovered) are snapped to the closest point of the nearest incident
  triangle; the snap count and worst snap distance are reported as
  diagnostics.

Accuracy and cost at production settings
----------------------------------------

For P2 elements the interior L2 error scales as :math:`O(h^3)`; the second
error source is the polygonal approximation of smooth benchmark boundaries
(sagitta :math:`\approx h_b^2 \kappa_{\text{curv}} / 8` for boundary spacing
:math:`h_b`).  Production settings used by the dataset generator are
``target_h = 0.02`` with 2048 boundary vertices on unit-scale domains: the
verification suite measures relative interior L2 errors of ``1.1e-8``
(Laplace) and ``1.1e-7`` (screened) against exterior-charge exact solutions
on a strongly deformed star at these settings — three-plus orders of
magnitude inside the ``1e-4`` acceptance target, leaving ample headroom for
the rougher random traces of the generated datasets (whose per-case
self-consistency checks land near ``1e-6``).  One production solve meshes
about 3.6e4 triangles (7.4e4 P2 nodes) and takes about 8 seconds of CPU
time, dominated by the sparse direct solve; see ``FEMDiagnostics`` timings.

This is a benchmark-local research utility, not a proposed public API.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial

__all__ = [
    "FEMDiagnostics",
    "FEMSolution",
    "k0_charge_potential",
    "log_charge_potential",
    "solve_dirichlet",
]

_EQUATIONS = ("laplace", "screened")

# Order-4 Dunavant quadrature on the reference triangle, barycentric
# coordinates with weights summing to one (integral = |T| * sum(w * f)).
_QUAD_BARYCENTRIC, _QUAD_WEIGHTS = (
    lambda a=0.445948490915965, b=0.091576213509771: (
        np.array(
            [
                [1.0 - 2.0 * a, a, a],
                [a, 1.0 - 2.0 * a, a],
                [a, a, 1.0 - 2.0 * a],
                [1.0 - 2.0 * b, b, b],
                [b, 1.0 - 2.0 * b, b],
                [b, b, 1.0 - 2.0 * b],
            ]
        ),
        np.array([0.223381589678011] * 3 + [0.109951743655322] * 3),
    )
)()


def _p2_shape_values(barycentric: np.ndarray) -> np.ndarray:
    """Evaluate the six P2 shape functions at barycentric points.

    Local numbering: nodes 0..2 are the vertices, node 3 is the midpoint of
    edge (1, 2), node 4 of edge (2, 0), and node 5 of edge (0, 1).
    """

    l0, l1, l2 = barycentric[..., 0], barycentric[..., 1], barycentric[..., 2]
    return np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l0,
            4.0 * l0 * l1,
        ),
        axis=-1,
    )


def _p2_shape_gradients(barycentric: np.ndarray) -> np.ndarray:
    """P2 shape gradients with respect to reference coordinates (l1, l2).

    ``l0 = 1 - l1 - l2`` is eliminated, so the result has shape
    ``(*barycentric.shape[:-1], 6, 2)``.
    """

    l0, l1, l2 = barycentric[..., 0], barycentric[..., 1], barycentric[..., 2]
    zeros = np.zeros_like(l0)
    d_dl1 = np.stack(
        (
            1.0 - 4.0 * l0,
            4.0 * l1 - 1.0,
            zeros,
            4.0 * l2,
            -4.0 * l2,
            4.0 * (l0 - l1),
        ),
        axis=-1,
    )
    d_dl2 = np.stack(
        (
            1.0 - 4.0 * l0,
            zeros,
            4.0 * l2 - 1.0,
            4.0 * l1,
            4.0 * (l0 - l2),
            -4.0 * l1,
        ),
        axis=-1,
    )
    return np.stack((d_dl1, d_dl2), axis=-1)


@dataclass(frozen=True)
class FEMDiagnostics:
    """Mesh, solve, and query-evaluation diagnostics for one Dirichlet solve."""

    equation: str
    kappa: float
    target_h: float
    n_vertices: int
    n_triangles: int
    n_nodes: int
    n_dirichlet_nodes: int
    max_edge_length: float
    trace_min: float
    trace_max: float
    solution_min: float
    solution_max: float
    linear_residual: float
    n_queries: int
    n_queries_snapped: int
    max_snap_distance: float
    mesh_seconds: float
    assemble_seconds: float
    solve_seconds: float
    evaluate_seconds: float

    def as_dict(self) -> dict:
        """Return a JSON-serializable copy (plain Python scalars only)."""

        result: dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                result[key] = value
            elif isinstance(value, int):
                result[key] = int(value)
            else:
                result[key] = float(value)
        return result


@dataclass(frozen=True)
class FEMSolution:
    """Interior values at the query points plus solver diagnostics.

    ``node_points``/``node_values``/``triangles`` describe the full P2
    solution (vertex nodes first, then edge-midpoint nodes) and are retained
    only when ``keep_mesh=True`` was requested; they are ``None`` otherwise.
    """

    u_query: np.ndarray
    diagnostics: FEMDiagnostics
    node_points: np.ndarray | None = None
    node_values: np.ndarray | None = None
    triangles: np.ndarray | None = None


def _validate_loops(boundary_loops: Sequence[np.ndarray]) -> list[np.ndarray]:
    loops = [np.asarray(loop, dtype=np.float64) for loop in boundary_loops]
    if not loops:
        raise ValueError("boundary_loops must contain at least the outer loop")
    for index, loop in enumerate(loops):
        if loop.ndim != 2 or loop.shape[1] != 2 or loop.shape[0] < 3:
            raise ValueError(
                f"loop {index} must have shape (n >= 3, 2), got {loop.shape}"
            )
        if not np.isfinite(loop).all():
            raise ValueError(f"loop {index} contains non-finite coordinates")
        if np.any(np.linalg.norm(np.diff(loop, axis=0), axis=1) == 0.0) or (
            np.linalg.norm(loop[0] - loop[-1]) == 0.0
        ):
            raise ValueError(f"loop {index} contains duplicate consecutive points")
    return loops


def _loop_segments(loops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate loop vertices and build closed per-loop segment lists."""

    vertices = np.concatenate(loops, axis=0)
    segments = []
    offset = 0
    for loop in loops:
        n = loop.shape[0]
        first = np.arange(n, dtype=np.int32) + offset
        second = np.roll(first, -1)
        segments.append(np.stack((first, second), axis=1))
        offset += n
    return vertices, np.concatenate(segments, axis=0)


def _polygon_interior_point(loop: np.ndarray) -> np.ndarray:
    """Return a point strictly inside a simple closed polygon.

    Robust for arbitrary simple polygons: constrained-Delaunay triangulate
    the polygon alone (cheap, no refinement) and take the centroid of any
    resulting triangle.
    """

    import triangle as _triangle

    n = loop.shape[0]
    ring = np.stack(
        (np.arange(n, dtype=np.int32), np.roll(np.arange(n, dtype=np.int32), -1)),
        axis=1,
    )
    result = _triangle.triangulate({"vertices": loop, "segments": ring}, "pQ")
    first = result["triangles"][0]
    return result["vertices"][first].mean(axis=0)


def _triangulate(
    loops: Sequence[np.ndarray], target_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quality-mesh the (possibly multiply-connected) polygonal domain.

    Returns ``(vertices, triangles, vertex_markers, boundary_segments)``.
    The area constraint is the area of an equilateral triangle with edge
    ``target_h``; together with the ``q30`` quality flag this bounds edge
    lengths near ``target_h`` away from the (finer) boundary polyline.
    """

    import triangle as _triangle

    vertices, segments = _loop_segments(loops)
    pslg: dict = {"vertices": vertices, "segments": segments}
    if len(loops) > 1:
        pslg["holes"] = np.stack(
            [_polygon_interior_point(loop) for loop in loops[1:]], axis=0
        )
    max_area = math.sqrt(3.0) / 4.0 * target_h * target_h
    flags = f"pq30a{max_area:.15f}Q"
    result = _triangle.triangulate(pslg, flags)
    return (
        np.asarray(result["vertices"], dtype=np.float64),
        np.asarray(result["triangles"], dtype=np.int64),
        np.asarray(result["vertex_markers"], dtype=np.int64).reshape(-1),
        np.asarray(result["segments"], dtype=np.int64),
    )


def _build_p2_connectivity(
    n_vertices: int, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign edge-midpoint node ids and build (n_tri, 6) P2 connectivity.

    Returns ``(connectivity, unique_edges)`` where ``unique_edges`` has shape
    ``(n_edges, 2)`` with sorted vertex pairs; midpoint node ``n_vertices + e``
    sits on ``unique_edges[e]``.
    """

    local_edges = triangles[:, [[1, 2], [2, 0], [0, 1]]].reshape(-1, 2)
    local_edges = np.sort(local_edges, axis=1)
    unique_edges, inverse = np.unique(local_edges, axis=0, return_inverse=True)
    midpoint_nodes = n_vertices + inverse.reshape(-1, 3)
    connectivity = np.concatenate((triangles, midpoint_nodes), axis=1)
    return connectivity, unique_edges


def _assemble(
    vertices: np.ndarray,
    triangles: np.ndarray,
    connectivity: np.ndarray,
    n_nodes: int,
    *,
    with_mass: bool,
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix | None]:
    """Vectorized P2 stiffness (and optional mass) assembly."""

    corner = vertices[triangles]  # (n_tri, 3, 2)
    jacobian = np.stack(
        (corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=2
    )  # (n_tri, 2, 2), columns are the edge vectors
    determinant = (
        jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    )
    if np.any(determinant <= 0.0):
        raise RuntimeError("triangulation produced a non-positive Jacobian")
    inverse = (
        np.stack(
            (
                np.stack((jacobian[:, 1, 1], -jacobian[:, 0, 1]), axis=1),
                np.stack((-jacobian[:, 1, 0], jacobian[:, 0, 0]), axis=1),
            ),
            axis=1,
        )
        / determinant[:, None, None]
    )
    areas = 0.5 * determinant

    n_tri = triangles.shape[0]
    stiffness_local = np.zeros((n_tri, 6, 6))
    mass_local = np.zeros((n_tri, 6, 6)) if with_mass else None
    shape_values = _p2_shape_values(_QUAD_BARYCENTRIC)  # (n_q, 6)
    shape_gradients = _p2_shape_gradients(_QUAD_BARYCENTRIC)  # (n_q, 6, 2)
    for q in range(_QUAD_BARYCENTRIC.shape[0]):
        weight = _QUAD_WEIGHTS[q]
        physical = np.einsum("ab,nbc->nac", shape_gradients[q], inverse)
        stiffness_local += (weight * areas)[:, None, None] * np.einsum(
            "nab,ncb->nac", physical, physical
        )
        if with_mass:
            outer = np.outer(shape_values[q], shape_values[q])
            mass_local += (weight * areas)[:, None, None] * outer[None]

    rows = np.repeat(connectivity, 6, axis=1).reshape(-1)
    cols = np.tile(connectivity, (1, 6)).reshape(-1)
    stiffness = scipy.sparse.coo_matrix(
        (stiffness_local.reshape(-1), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    mass = (
        scipy.sparse.coo_matrix(
            (mass_local.reshape(-1), (rows, cols)), shape=(n_nodes, n_nodes)
        ).tocsr()
        if with_mass
        else None
    )
    return stiffness, mass


def _dirichlet_nodes(
    n_vertices: int,
    vertex_markers: np.ndarray,
    boundary_segments: np.ndarray,
    unique_edges: np.ndarray,
) -> np.ndarray:
    """Node ids (vertices and edge midpoints) lying on the domain boundary."""

    boundary_vertices = np.nonzero(vertex_markers != 0)[0]
    keys = unique_edges[:, 0].astype(np.int64) * (n_vertices + 1) + unique_edges[:, 1]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    seg = np.sort(boundary_segments, axis=1)
    seg_keys = seg[:, 0].astype(np.int64) * (n_vertices + 1) + seg[:, 1]
    positions = np.searchsorted(sorted_keys, seg_keys)
    if np.any(positions >= sorted_keys.shape[0]) or np.any(
        sorted_keys[np.minimum(positions, sorted_keys.shape[0] - 1)] != seg_keys
    ):
        raise RuntimeError("a boundary segment is not an edge of the triangulation")
    midpoint_nodes = n_vertices + order[positions]
    return np.concatenate((boundary_vertices, midpoint_nodes))


def _interpolate_vertex_trace(
    loops: Sequence[np.ndarray],
    per_vertex_values: Sequence[np.ndarray],
    points: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate per-loop-vertex traces at points on the loops.

    Boundary nodes produced by meshing (original vertices, segment Steiner
    points, edge midpoints) all lie on some input segment up to roundoff;
    each point is projected onto the nearest candidate input segment found
    through a KD-tree over segment midpoints, and the trace is interpolated
    linearly along that segment — exactly the trace of the polygonal problem
    being solved.
    """

    starts, ends, value_starts, value_ends = [], [], [], []
    for loop, values in zip(loops, per_vertex_values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (loop.shape[0],):
            raise ValueError(
                "per-vertex boundary values must match their loop vertex count"
            )
        starts.append(loop)
        ends.append(np.roll(loop, -1, axis=0))
        value_starts.append(values)
        value_ends.append(np.roll(values, -1))
    starts = np.concatenate(starts, axis=0)
    ends = np.concatenate(ends, axis=0)
    value_starts = np.concatenate(value_starts)
    value_ends = np.concatenate(value_ends)

    midpoints = 0.5 * (starts + ends)
    tree = scipy.spatial.cKDTree(midpoints)
    k = min(8, midpoints.shape[0])
    _, candidates = tree.query(points, k=k)
    candidates = candidates.reshape(points.shape[0], -1)

    seg_start = starts[candidates]  # (n, k, 2)
    seg_vector = ends[candidates] - seg_start
    seg_length2 = np.maximum(np.sum(seg_vector**2, axis=-1), 1.0e-300)
    t = np.clip(
        np.sum((points[:, None, :] - seg_start) * seg_vector, axis=-1) / seg_length2,
        0.0,
        1.0,
    )
    projections = seg_start + t[..., None] * seg_vector
    distances2 = np.sum((points[:, None, :] - projections) ** 2, axis=-1)
    best = np.argmin(distances2, axis=1)
    rows = np.arange(points.shape[0])
    chosen = candidates[rows, best]
    t_best = t[rows, best]
    return (1.0 - t_best) * value_starts[chosen] + t_best * value_ends[chosen]


def _evaluate_queries(
    vertices: np.ndarray,
    triangles: np.ndarray,
    connectivity: np.ndarray,
    node_values: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Locate queries and evaluate the P2 field; snap marginal outsiders."""

    import matplotlib.tri as mtri

    triangulation = mtri.Triangulation(
        vertices[:, 0], vertices[:, 1], triangles=triangles
    )
    finder = triangulation.get_trifinder()  # TrapezoidMapTriFinder
    located = np.asarray(
        finder(
            np.ascontiguousarray(query_points[:, 0]),
            np.ascontiguousarray(query_points[:, 1]),
        ),
        dtype=np.int64,
    )

    missing = np.nonzero(located < 0)[0]
    n_snapped = int(missing.shape[0])
    max_snap = 0.0
    if n_snapped:
        # Snap each stray query to a triangle incident to its nearest vertex;
        # barycentric clipping below then evaluates at the closest point of
        # that triangle.  Strays arise only from the sliver between an
        # inscribed polygon and the smooth curve it samples.
        vertex_to_triangle = np.full(vertices.shape[0], -1, dtype=np.int64)
        flat = triangles.reshape(-1)
        vertex_to_triangle[flat[::-1]] = np.repeat(
            np.arange(triangles.shape[0])[::-1], 3
        )
        tree = scipy.spatial.cKDTree(vertices)
        _, nearest_vertex = tree.query(query_points[missing])
        located[missing] = vertex_to_triangle[nearest_vertex]
        if np.any(located < 0):
            raise RuntimeError("query point location failed")

    corner = vertices[triangles[located]]  # (n_q, 3, 2)
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    rhs = query_points - corner[:, 0]
    determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    l1 = (rhs[:, 0] * edge2[:, 1] - rhs[:, 1] * edge2[:, 0]) / determinant
    l2 = (edge1[:, 0] * rhs[:, 1] - edge1[:, 1] * rhs[:, 0]) / determinant
    barycentric = np.stack((1.0 - l1 - l2, l1, l2), axis=1)
    if n_snapped:
        clipped = np.clip(barycentric[missing], 0.0, None)
        clipped /= clipped.sum(axis=1, keepdims=True)
        barycentric[missing] = clipped
        snapped_points = np.einsum("nc,ncd->nd", clipped, corner[missing])
        max_snap = float(
            np.linalg.norm(snapped_points - query_points[missing], axis=1).max()
        )

    values = np.einsum(
        "na,na->n", _p2_shape_values(barycentric), node_values[connectivity[located]]
    )
    return values, n_snapped, max_snap


def solve_dirichlet(
    boundary_loops: Sequence[np.ndarray],
    boundary_values: Callable[[np.ndarray], np.ndarray] | Sequence[np.ndarray],
    query_points: np.ndarray,
    *,
    equation: str = "laplace",
    kappa: float = 0.0,
    target_h: float = 0.02,
    degree: int = 2,
    keep_mesh: bool = False,
) -> FEMSolution:
    r"""Solve a homogeneous Dirichlet problem and evaluate interior queries.

    Parameters
    ----------
    boundary_loops
        Sequence of ``(n_i, 2)`` float arrays.  The first loop is the outer
        boundary; every further loop is an inner hole.  Loops are closed
        implicitly (do not repeat the first vertex) and must be simple and
        mutually disjoint.
    boundary_values
        Either a callable mapping ``(n, 2)`` physical points to ``(n,)``
        trace values (evaluated at every boundary node the mesher produces),
        or a sequence of per-vertex trace arrays — one array per loop, one
        value per loop vertex — interpolated linearly along the polyline.
    query_points
        ``(n_q, 2)`` interior evaluation points.
    equation
        ``"laplace"`` for :math:`-\Delta u = 0` or ``"screened"`` for
        :math:`-\Delta u + \kappa^2 u = 0`.
    kappa
        Screening constant; must be positive iff ``equation == "screened"``.
    target_h
        Interior edge-length target; the mesher receives the corresponding
        equilateral-triangle area constraint plus a 30-degree quality bound.
    degree
        Finite-element degree; only P2 (``degree=2``) is implemented.
    keep_mesh
        When true, retain P2 node coordinates, node values, and the P1
        triangulation on the returned solution for plotting or debugging.

    Returns
    -------
    FEMSolution
        Query values plus :class:`FEMDiagnostics`.
    """

    if degree != 2:
        raise NotImplementedError("only degree=2 (P2 triangles) is implemented")
    if equation not in _EQUATIONS:
        raise ValueError(f"equation must be one of {_EQUATIONS}, got {equation!r}")
    if equation == "screened":
        if not (math.isfinite(kappa) and kappa > 0.0):
            raise ValueError("screened equation requires finite kappa > 0")
    elif kappa != 0.0:
        raise ValueError("kappa must be 0.0 for the laplace equation")
    if not (math.isfinite(target_h) and target_h > 0.0):
        raise ValueError("target_h must be finite and positive")
    loops = _validate_loops(boundary_loops)
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 2:
        raise ValueError("query_points must have shape (n_q, 2)")
    if not np.isfinite(query_points).all():
        raise ValueError("query_points must be finite")
    values_callable = callable(boundary_values)
    if not values_callable and len(boundary_values) != len(loops):
        raise ValueError("per-vertex boundary_values must provide one array per loop")

    start = time.perf_counter()
    vertices, triangles, vertex_markers, boundary_segments = _triangulate(
        loops, target_h
    )
    mesh_seconds = time.perf_counter() - start

    start = time.perf_counter()
    n_vertices = vertices.shape[0]
    connectivity, unique_edges = _build_p2_connectivity(n_vertices, triangles)
    n_nodes = n_vertices + unique_edges.shape[0]
    node_points = np.concatenate(
        (vertices, 0.5 * (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]])),
        axis=0,
    )
    stiffness, mass = _assemble(
        vertices,
        triangles,
        connectivity,
        n_nodes,
        with_mass=(equation == "screened"),
    )
    system = stiffness if mass is None else stiffness + (kappa * kappa) * mass

    dirichlet = _dirichlet_nodes(
        n_vertices, vertex_markers, boundary_segments, unique_edges
    )
    dirichlet_points = node_points[dirichlet]
    if values_callable:
        trace = np.asarray(boundary_values(dirichlet_points), dtype=np.float64)
        if trace.shape != (dirichlet.shape[0],):
            raise ValueError(
                "boundary_values callable must return one value per input point"
            )
    else:
        trace = _interpolate_vertex_trace(loops, boundary_values, dirichlet_points)
    if not np.isfinite(trace).all():
        raise ValueError("boundary trace evaluated to non-finite values")
    assemble_seconds = time.perf_counter() - start

    start = time.perf_counter()
    is_dirichlet = np.zeros(n_nodes, dtype=bool)
    is_dirichlet[dirichlet] = True
    free = np.nonzero(~is_dirichlet)[0]
    node_values = np.zeros(n_nodes)
    node_values[dirichlet] = trace
    reduced = system[free][:, free].tocsc()
    rhs = -system[free][:, dirichlet] @ trace
    node_values[free] = scipy.sparse.linalg.spsolve(reduced, rhs)
    residual = float(
        np.linalg.norm(reduced @ node_values[free] - rhs)
        / max(np.linalg.norm(rhs), 1.0e-300)
    )
    solve_seconds = time.perf_counter() - start

    start = time.perf_counter()
    u_query, n_snapped, max_snap = _evaluate_queries(
        vertices, triangles, connectivity, node_values, query_points
    )
    evaluate_seconds = time.perf_counter() - start

    edge_vectors = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
    diagnostics = FEMDiagnostics(
        equation=equation,
        kappa=float(kappa),
        target_h=float(target_h),
        n_vertices=int(n_vertices),
        n_triangles=int(triangles.shape[0]),
        n_nodes=int(n_nodes),
        n_dirichlet_nodes=int(dirichlet.shape[0]),
        max_edge_length=float(np.linalg.norm(edge_vectors, axis=1).max()),
        trace_min=float(trace.min()),
        trace_max=float(trace.max()),
        solution_min=float(node_values.min()),
        solution_max=float(node_values.max()),
        linear_residual=residual,
        n_queries=int(query_points.shape[0]),
        n_queries_snapped=n_snapped,
        max_snap_distance=max_snap,
        mesh_seconds=mesh_seconds,
        assemble_seconds=assemble_seconds,
        solve_seconds=solve_seconds,
        evaluate_seconds=evaluate_seconds,
    )
    return FEMSolution(
        u_query=u_query,
        diagnostics=diagnostics,
        node_points=node_points if keep_mesh else None,
        node_values=node_values if keep_mesh else None,
        triangles=triangles if keep_mesh else None,
    )


def log_charge_potential(
    points: np.ndarray, centers: np.ndarray, charges: np.ndarray
) -> np.ndarray:
    r"""Exact harmonic field :math:`u = \sum_j q_j \log |x - c_j|`.

    Harmonic wherever no ``c_j`` lies; place the charges outside the domain
    to manufacture exact Dirichlet problems with non-polynomial traces.
    """

    points = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(
        points[:, None, :] - np.asarray(centers, dtype=np.float64)[None], axis=-1
    )
    return np.log(distances) @ np.asarray(charges, dtype=np.float64)


def k0_charge_potential(
    points: np.ndarray,
    centers: np.ndarray,
    charges: np.ndarray,
    kappa: float,
) -> np.ndarray:
    r"""Exact screened field :math:`u = \sum_j q_j K_0(\kappa |x - c_j|)`.

    Each Bessel-``K0`` source satisfies :math:`-\Delta u + \kappa^2 u = 0`
    away from its center (modified Helmholtz fundamental solution).
    """

    from scipy.special import k0

    points = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(
        points[:, None, :] - np.asarray(centers, dtype=np.float64)[None], axis=-1
    )
    return k0(kappa * distances) @ np.asarray(charges, dtype=np.float64)
