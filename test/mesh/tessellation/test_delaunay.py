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

"""Tests for physicsnemo.mesh.tessellation.delaunay.

Covers the algorithmic guarantees layer by layer: the empty-circumcircle
property of the plain Bowyer-Watson triangulation, conformity of constrained
segment recovery, exterior/hole removal, Ruppert refinement quality bounds,
bitwise determinism, and a production-shaped geometry (square cavity with a
star hole, the ns_cavity_star benchmark family's shape) at realistic
resolution.
"""

import math

import numpy as np
import pytest
import torch

from physicsnemo.mesh.tessellation import delaunay_mesh, polygon_interior_point
from physicsnemo.mesh.tessellation.delaunay import _delaunay_triangulation

### Geometry fixtures ---------------------------------------------------------


def _square_loop(n_per_edge: int, half: float = 1.0) -> np.ndarray:
    """Axis-aligned square [-half, half]^2 with n_per_edge points per side."""
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    points = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        points.extend(a + t * (b - a) for t in np.arange(n_per_edge) / n_per_edge)
    return np.array(points)


def _star_loop(
    n: int, *, radius: float = 1.0, amplitude: float = 0.3, lobes: int = 5
) -> np.ndarray:
    """Star-deformed circle r(theta) = radius * (1 + amplitude cos(lobes theta))."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + amplitude * np.cos(lobes * theta))
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)


def _loop_segments(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated loop vertices and the closed per-loop segment index pairs."""
    vertices = np.concatenate(loops, axis=0)
    segments = []
    offset = 0
    for loop in loops:
        n = loop.shape[0]
        first = np.arange(n) + offset
        segments.append(np.stack((first, np.roll(first, -1)), axis=1))
        offset += n
    return vertices, np.concatenate(segments, axis=0)


def _points_in_polygon(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Even-odd ray-crossing point-in-polygon test (vectorized, half-open)."""
    x, y = points[:, 0:1], points[:, 1:2]
    ax, ay = loop[:, 0][None], loop[:, 1][None]
    bx = np.roll(loop[:, 0], -1)[None]
    by = np.roll(loop[:, 1], -1)[None]
    straddles = (ay <= y) != (by <= y)
    crosses = x < ax + (y - ay) * (bx - ax) / (by - ay + (ay == by))
    return (straddles & crosses).sum(axis=1) % 2 == 1


def _triangle_geometry(
    points: torch.Tensor, triangles: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signed areas, minimum angles in degrees, centroids) per triangle."""
    p = points.numpy()
    tri = triangles.numpy()
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )

    def angles(u, v):
        cosine = (u * v).sum(1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    minimum_angle = np.minimum(
        np.minimum(angles(b - a, c - a), angles(a - b, c - b)),
        angles(a - c, b - c),
    )
    return 0.5 * doubled, minimum_angle, (a + b + c) / 3.0


def _point_segment_distances(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Distance from each point to its nearest segment among (starts, ends)."""
    vector = ends - starts  # (s, 2)
    length2 = np.maximum((vector**2).sum(axis=1), 1.0e-300)
    t = np.clip(
        ((points[:, None, :] - starts[None]) * vector[None]).sum(-1) / length2[None],
        0.0,
        1.0,
    )
    projections = starts[None] + t[..., None] * vector[None]
    return np.linalg.norm(points[:, None, :] - projections, axis=-1).min(axis=1)


### Plain Delaunay: empty-circumcircle property -------------------------------


@pytest.mark.parametrize("seed,n", [(0, 200), (1, 500), (2, 1000)])
def test_delaunay_property_on_pseudo_random_points(seed, n):
    """No point lies strictly inside any circumcircle (tol 1e-12, unit box)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, (n, 2))
    triangles = _delaunay_triangulation(points)

    # A triangulation of the convex hull: positive orientation, correct count
    # (Euler: 2n - hull - 2 triangles), and total area equal to the hull's.
    a, b, c = (points[triangles[:, k]] for k in range(3))
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    assert (doubled > 0.0).all()
    import scipy.spatial

    hull = scipy.spatial.ConvexHull(points)
    assert triangles.shape[0] == 2 * n - hull.vertices.shape[0] - 2
    assert 0.5 * doubled.sum() == pytest.approx(hull.volume, rel=1e-12)

    # Empty circumcircle, brute force against every point.
    denominator = 2.0 * doubled
    bl = ((b - a) ** 2).sum(1)
    cl = ((c - a) ** 2).sum(1)
    ux = a[:, 0] + ((c[:, 1] - a[:, 1]) * bl - (b[:, 1] - a[:, 1]) * cl) / denominator
    uy = a[:, 1] + ((b[:, 0] - a[:, 0]) * cl - (c[:, 0] - a[:, 0]) * bl) / denominator
    radius = np.sqrt((ux - a[:, 0]) ** 2 + (uy - a[:, 1]) ** 2)
    distances = np.sqrt(
        (points[:, 0][None] - ux[:, None]) ** 2
        + (points[:, 1][None] - uy[:, None]) ** 2
    )
    violation = (radius[:, None] - distances).max()
    assert violation <= 1.0e-12, f"circumcircle violated by {violation:.3e}"


### Constrained Delaunay: conformity ------------------------------------------


def _assert_segments_conforming(
    loops: list[np.ndarray],
    points: torch.Tensor,
    markers: torch.Tensor,
    boundary_segments: torch.Tensor,
    tolerance: float = 1.0e-9,
):
    """Every input segment is exactly tiled by output subsegments; markers hold."""
    vertices, input_segments = _loop_segments(loops)
    p = points.numpy()
    marks = markers.numpy()
    segs = boundary_segments.numpy()

    # Input vertices come first, bit-identically, and are boundary-marked.
    np.testing.assert_array_equal(p[: vertices.shape[0]], vertices)
    assert (marks[: vertices.shape[0]] == 1).all()

    # Subsegment endpoints are boundary-marked; nothing else is.
    assert (marks[segs.reshape(-1)] == 1).all()
    on_boundary = np.zeros(p.shape[0], dtype=bool)
    on_boundary[segs.reshape(-1)] = True
    assert (marks == on_boundary.astype(np.int64)).all()

    # Assign each subsegment to the input segment its endpoints lie on, and
    # verify the parameter intervals tile [0, 1] for every input segment.
    starts = vertices[input_segments[:, 0]]
    ends = vertices[input_segments[:, 1]]
    scale = np.abs(vertices).max()
    intervals: dict[int, list[tuple[float, float]]] = {
        i: [] for i in range(input_segments.shape[0])
    }
    for u, v in segs:
        midpoint = 0.5 * (p[u] + p[v])
        distances = _point_segment_distances(midpoint[None], starts, ends)
        parent = None
        for candidate in np.argsort(
            np.linalg.norm(0.5 * (starts + ends) - midpoint[None], axis=1)
        )[:8]:
            s, e = starts[candidate], ends[candidate]
            length2 = ((e - s) ** 2).sum()
            tu = ((p[u] - s) @ (e - s)) / length2
            tv = ((p[v] - s) @ (e - s)) / length2
            du = np.linalg.norm(p[u] - (s + tu * (e - s)))
            dv = np.linalg.norm(p[v] - (s + tv * (e - s)))
            if (
                max(du, dv) <= tolerance * scale
                and -1e-9 <= min(tu, tv)
                and max(tu, tv) <= 1.0 + 1e-9
            ):
                parent = int(candidate)
                intervals[parent].append((min(tu, tv), max(tu, tv)))
                break
        assert parent is not None, f"subsegment ({u}, {v}) lies on no input segment"
        assert distances[0] <= tolerance * scale
    for index, spans in intervals.items():
        assert spans, f"input segment {index} has no subsegments"
        spans.sort()
        assert spans[0][0] == pytest.approx(0.0, abs=1e-9)
        assert spans[-1][1] == pytest.approx(1.0, abs=1e-9)
        for (_, hi), (lo, _) in zip(spans[:-1], spans[1:]):
            assert lo == pytest.approx(hi, abs=1e-9), f"gap in segment {index}"


def test_cdt_conformity_without_refinement():
    """Pure CDT (no refinement): the input segments ARE the output segments."""
    loops = [_star_loop(48, amplitude=0.35), _star_loop(24, radius=0.3, lobes=3)]
    points, triangles, markers, segments = delaunay_mesh(
        loops, max_area=None, min_angle_degrees=0.0
    )
    vertices, input_segments = _loop_segments(loops)
    assert points.shape[0] == vertices.shape[0]  # no Steiner points at all
    assert segments.shape[0] == input_segments.shape[0]
    directed = {(int(u), int(v)) for u, v in segments.numpy()}
    for a, b in input_segments:
        assert (a, b) in directed or (b, a) in directed
    _assert_segments_conforming(loops, points, markers, segments)


def test_cdt_conformity_with_refinement():
    """With refinement, every input segment is a chain of output subsegments."""
    loops = [_square_loop(6), _star_loop(16, radius=0.35)]
    points, triangles, markers, segments = delaunay_mesh(
        loops, max_area=0.01, min_angle_degrees=30.0
    )
    assert points.shape[0] > sum(loop.shape[0] for loop in loops)
    assert segments.shape[0] > 6 * 4 + 16  # boundary refinement did split
    _assert_segments_conforming(loops, points, markers, segments)


### Hole and exterior removal --------------------------------------------------


def test_hole_and_exterior_removal():
    """No triangle centroid falls inside a hole or outside the outer loop."""
    outer = _star_loop(96, amplitude=0.4)
    hole_a = _star_loop(32, radius=0.25, lobes=3) + np.array([0.45, 0.0])
    hole_b = _star_loop(16, radius=0.12, amplitude=0.0) - np.array([0.45, 0.0])
    points, triangles, markers, segments = delaunay_mesh(
        [outer, hole_a, hole_b], max_area=0.02
    )
    areas, _, centroids = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert _points_in_polygon(centroids, outer).all()
    assert not _points_in_polygon(centroids, hole_a).any()
    assert not _points_in_polygon(centroids, hole_b).any()
    # The mesh area equals the outer area minus the holes (polygon shoelace).

    def shoelace(loop):
        return 0.5 * abs(
            np.sum(loop[:, 0] * np.roll(loop[:, 1], -1))
            - np.sum(loop[:, 1] * np.roll(loop[:, 0], -1))
        )

    expected = shoelace(outer) - shoelace(hole_a) - shoelace(hole_b)
    assert areas.sum() == pytest.approx(expected, rel=1e-12)


### Ruppert refinement bounds ---------------------------------------------------


@pytest.mark.parametrize("min_angle", [20.0, 30.0, 33.0])
def test_refinement_quality_bounds(min_angle):
    """All angles >= bound - 0.5 deg; all areas <= max_area * (1 + 1e-9)."""
    max_area = math.sqrt(3.0) / 4.0 * 0.08**2
    loops = [_star_loop(128, amplitude=0.25), _star_loop(32, radius=0.3, lobes=3)]
    points, triangles, markers, segments = delaunay_mesh(
        loops, max_area=max_area, min_angle_degrees=min_angle
    )
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert areas.max() <= max_area * (1.0 + 1.0e-9)
    assert minimum_angles.min() >= min_angle - 0.5

    # Every boundary subsegment (and thus every marker-1 vertex) lies on an
    # input segment, including midpoints inserted by encroachment splits.
    vertices, input_segments = _loop_segments(loops)
    boundary_vertices = points.numpy()[markers.numpy() == 1]
    distances = _point_segment_distances(
        boundary_vertices,
        vertices[input_segments[:, 0]],
        vertices[input_segments[:, 1]],
    )
    assert distances.max() <= 1.0e-9
    _assert_segments_conforming(loops, points, markers, segments)


def test_angle_only_refinement_without_area_bound():
    """max_area=None still refines away skinny CDT triangles."""
    loops = [_star_loop(64)]
    points, triangles, markers, segments = delaunay_mesh(loops, max_area=None)
    _, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert minimum_angles.min() >= 30.0 - 0.5


### Determinism -----------------------------------------------------------------


def test_bitwise_determinism():
    """Two identical calls return bitwise-identical tensors."""
    loops = [_star_loop(64, amplitude=0.3), _star_loop(24, radius=0.3)]
    first = delaunay_mesh(loops, max_area=0.005, min_angle_degrees=30.0)
    second = delaunay_mesh(
        [loop.copy() for loop in loops], max_area=0.005, min_angle_degrees=30.0
    )
    for tensor_a, tensor_b in zip(first, second):
        assert tensor_a.dtype == tensor_b.dtype
        assert torch.equal(tensor_a, tensor_b)


### Production-shaped geometry ----------------------------------------------------


def test_square_cavity_with_star_hole_at_production_resolution():
    """The ns_cavity_star shape (square cavity + star hole) meshes at h=0.05."""
    h = 0.05
    max_area = math.sqrt(3.0) / 4.0 * h * h
    outer = _square_loop(40)  # spacing 0.05 on a side of length 2
    hole = _star_loop(128, radius=0.35, amplitude=0.3, lobes=5)
    points, triangles, markers, segments = delaunay_mesh(
        [outer, hole], max_area=max_area, min_angle_degrees=30.0
    )
    areas, minimum_angles, centroids = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert areas.max() <= max_area * (1.0 + 1.0e-9)
    assert minimum_angles.min() >= 29.5
    assert _points_in_polygon(centroids, outer).all()
    assert not _points_in_polygon(centroids, hole).any()

    # Sane counts: the domain area over the mean quality-triangle area brackets
    # the triangle count within loose constant factors.
    domain_area = 4.0 - np.pi * 0.35**2 * (1.0 + 0.3**2 / 2.0)
    assert (
        1.0 * domain_area / max_area
        <= triangles.shape[0]
        <= 4.0 * domain_area / max_area
    )
    assert torch.long == triangles.dtype == markers.dtype == segments.dtype
    assert points.dtype == torch.float64


### polygon_interior_point ---------------------------------------------------------


@pytest.mark.parametrize("winding", [1, -1])
def test_polygon_interior_point_star_and_square(winding):
    """The returned point is strictly inside, for both windings."""
    for loop in (
        _square_loop(1)[::winding].copy(),
        _star_loop(48, amplitude=0.45)[::winding].copy(),
        np.array([[0.0, 0.0], [4.0, 2.0], [0.0, 4.0], [1.0, 2.0]])[::winding].copy(),
    ):
        inside = polygon_interior_point(loop)
        assert inside.shape == (2,)
        assert inside.dtype == torch.float64
        assert _points_in_polygon(inside.numpy()[None], loop).all()
        distance = _point_segment_distances(
            inside.numpy()[None], loop, np.roll(loop, -1, axis=0)
        )
        assert distance[0] > 0.0


def test_polygon_interior_point_accepts_torch_input():
    loop = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    inside = polygon_interior_point(loop)
    assert 0.0 < float(inside[0]) < 2.0
    assert 0.0 < float(inside[1]) < 1.0


### Validation errors ---------------------------------------------------------------


def test_rejects_bad_arguments():
    square = _square_loop(1)
    with pytest.raises(ValueError, match="at least the outer boundary"):
        delaunay_mesh([])
    with pytest.raises(ValueError, match=r"shape \(n >= 3, 2\)"):
        delaunay_mesh([np.zeros((2, 2))])
    with pytest.raises(ValueError, match="non-finite"):
        delaunay_mesh([np.array([[0.0, 0.0], [1.0, np.nan], [1.0, 1.0]])])
    with pytest.raises(ValueError, match="duplicate consecutive"):
        delaunay_mesh([np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])])
    with pytest.raises(ValueError, match="duplicate vertex"):
        delaunay_mesh([square, square])
    with pytest.raises(ValueError, match="max_area"):
        delaunay_mesh([square], max_area=0.0)
    with pytest.raises(ValueError, match=r"min_angle_degrees must lie in \[0, 33\]"):
        delaunay_mesh([square], min_angle_degrees=34.0)
    with pytest.raises(ValueError, match="cross"):
        delaunay_mesh(
            [
                square,
                np.array([[0.0, 0.0], [3.0, 0.5], [0.0, 1.0]]),  # crosses outer
            ]
        )
