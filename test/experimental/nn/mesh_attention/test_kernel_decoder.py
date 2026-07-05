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

"""Contracts specific to the dense kernel-basis query decoder.

The generic model contracts (superposition, zero drive, O(D) and similarity
covariance, permutations, chunking, checkpointing) are parametrized over both
query decoders in ``test_model_contracts.py``.  This module tests what is new
in kernel mode: the exact singular quadrature member, bitwise query-set
independence, the encode-time source cache, and the removal of the separable
decoder's angular-order ceiling.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    exact_double_layer_member,
    exact_single_layer_member,
)
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh


def _circle_boundary(
    n_cells: int,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
) -> Mesh:
    """Unit-circle boundary with clockwise cells, hence outward normals."""
    angles = 2.0 * torch.pi * torch.arange(n_cells, device=device, dtype=dtype)
    angles = angles / n_cells
    points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    indices = torch.arange(n_cells, device=device)
    cells = torch.stack((torch.roll(indices, -1), indices), dim=-1)
    return Mesh(points=points, cells=cells)


def _disk_domain(
    n_boundary: int,
    query_points: torch.Tensor,
    boundary_values: torch.Tensor,
) -> DomainMesh:
    boundary = _circle_boundary(
        n_boundary, query_points.device, dtype=query_points.dtype
    )
    return DomainMesh(
        interior=Mesh(points=query_points),
        boundaries={
            "disk": boundary.with_data(cell_data={"boundary_value": boundary_values})
        },
    )


def _disk_model(
    query_decoder: str,
    device: torch.device | str,
    *,
    seed: int = 2111,
    **overrides,
) -> MeshTransformer:
    """A small linear scalar 2D model matching the ceiling-test assumptions."""
    torch.manual_seed(seed)
    kwargs = dict(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
        field_mode="linear",
        query_decoder=query_decoder,
        operator_scalar_dim=7,
        operator_vector_dim=3,
        drive_scalar_dim=9,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=2,
        scalar_rank=4,
        vector_rank=2,
    )
    kwargs.update(overrides)
    model = MeshTransformer(**kwargs).to(device=device, dtype=torch.float64).eval()
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    return model


def _tetrahedron_mesh(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
) -> Mesh:
    """Closed tetrahedral surface with consistently outward-oriented faces."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.1],
            [0.1, 1.1, -0.1],
            [-0.1, 0.2, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    cells = torch.tensor(
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        device=device,
        dtype=torch.long,
    )
    return Mesh(points=points, cells=cells)


# ---------------------------------------------------------------------------
# Exact singular member: quadrature identity, orientation, and Gauss identity.
# ---------------------------------------------------------------------------


def test_exact_member_matches_brute_force_quadrature_2d(device):
    """The segment member equals the numerically integrated double layer."""
    generator = torch.Generator().manual_seed(41)
    start = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    end = start + torch.randn(5, 2, generator=generator, dtype=torch.float64)
    tangent = end - start
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    # Arbitrary per-panel normal orientation: the sigma factor must absorb it.
    flip = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    normals = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1) * flip[:, None]
    queries = 3.0 * torch.randn(4, 2, generator=generator, dtype=torch.float64)
    panel_vertices = torch.stack((start, end), dim=1).to(device)
    queries, normals = queries.to(device), normals.to(device)

    exact = exact_double_layer_member(queries, panel_vertices, normals)

    n_nodes = 20000
    t = (torch.arange(n_nodes, dtype=torch.float64, device=device) + 0.5) / n_nodes
    nodes = (
        panel_vertices[:, None, 0, :]
        + t[None, :, None]
        * (panel_vertices[:, 1, :] - panel_vertices[:, 0, :])[:, None, :]
    )
    ds = (panel_vertices[:, 1, :] - panel_vertices[:, 0, :]).norm(dim=-1) / n_nodes
    r = queries[:, None, None, :] - nodes[None, :, :, :]
    b = torch.einsum("qsnd,sd->qsn", r, normals)
    a = r.square().sum(dim=-1)
    brute = (b / (2.0 * math.pi * a)).sum(dim=-1) * ds[None, :]

    torch.testing.assert_close(exact, brute, rtol=0.0, atol=1.0e-5)


def test_exact_member_matches_brute_force_quadrature_3d(device):
    """The triangle member equals the numerically integrated double layer."""
    generator = torch.Generator().manual_seed(43)
    vertices = torch.randn(4, 3, 3, generator=generator, dtype=torch.float64)
    edge_one = vertices[:, 1, :] - vertices[:, 0, :]
    edge_two = vertices[:, 2, :] - vertices[:, 0, :]
    winding = torch.cross(edge_one, edge_two, dim=-1)
    flip = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
    normals = winding / winding.norm(dim=-1, keepdim=True) * flip[:, None]
    queries = 4.0 * torch.randn(3, 3, generator=generator, dtype=torch.float64)
    vertices, normals, queries = (
        vertices.to(device),
        normals.to(device),
        queries.to(device),
    )

    exact = exact_double_layer_member(queries, vertices, normals)

    resolution = 200
    steps = (
        torch.arange(resolution, dtype=torch.float64, device=device) + 0.5
    ) / resolution
    u, v = torch.meshgrid(steps, steps, indexing="ij")
    inside = (u + v) < 1.0
    u, v = u[inside], v[inside]
    nodes = (
        vertices[:, None, 0, :]
        + u[None, :, None] * edge_one.to(device)[:, None, :]
        + v[None, :, None] * edge_two.to(device)[:, None, :]
    )
    area = winding.to(device).norm(dim=-1) / 2.0
    node_area = 2.0 * area / (resolution * resolution)
    r = queries[:, None, None, :] - nodes[None, :, :, :]
    b = torch.einsum("qskd,sd->qsk", r, normals)
    distance = r.norm(dim=-1)
    brute = (b / (4.0 * math.pi * distance.pow(3))).sum(dim=-1) * node_area[None, :]

    torch.testing.assert_close(exact, brute, rtol=0.0, atol=1.0e-3)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_exact_member_is_odd_in_the_normal(device, n_dims):
    """Flipping the supplied normals must exactly negate the member."""
    generator = torch.Generator().manual_seed(47)
    if n_dims == 2:
        mesh = _circle_boundary(12, "cpu")
        queries = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    else:
        mesh = _tetrahedron_mesh("cpu")
        queries = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    vertices = mesh.points[mesh.cells].to(device)
    normals = mesh.cell_normals.to(device)
    queries = queries.to(device)

    member = exact_double_layer_member(queries, vertices, normals)
    flipped = exact_double_layer_member(queries, vertices, -normals)

    torch.testing.assert_close(flipped, -member, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_exact_member_interior_gauss_identity(device, n_dims):
    """Rows sum to exactly -1 at interior points of a closed outward surface."""
    if n_dims == 2:
        mesh = _circle_boundary(48, "cpu")
        queries = torch.tensor(
            [[0.0, 0.0], [0.31, -0.22], [0.7, 0.12]], dtype=torch.float64
        )
    else:
        mesh = _tetrahedron_mesh("cpu")
        queries = torch.tensor(
            [[0.18, 0.21, 0.17], [0.42, 0.11, 0.23], [0.25, 0.22, 0.51]],
            dtype=torch.float64,
        )
    member = exact_double_layer_member(
        queries.to(device),
        mesh.points[mesh.cells].to(device),
        mesh.cell_normals.to(device),
    )
    sums = member.sum(dim=1)
    torch.testing.assert_close(
        sums, torch.full_like(sums, -1.0), rtol=0.0, atol=1.0e-12
    )


# ---------------------------------------------------------------------------
# Exact single-layer member: quadrature identity, orientation independence,
# and classical constant-density potentials.
# ---------------------------------------------------------------------------


def _icosphere(subdivisions: int, dtype: torch.dtype = torch.float64):
    """Unit icosphere (ported from examples/.../laplace3d.py ``icosphere``)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
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


#: Perpendicular query offsets as multiples of panel size; the last is the
#: mandated very-near-field case.  Per the singular-quadrature contract, the
#: closed form must track adaptive quadrature to <= 1e-6 relative error far
#: from the panel and <= 1e-4 at ~1e-3 panel sizes (in practice it is exact
#: to roundoff at every distance).
_QUERY_OFFSET_SCALES = (3.0, 0.7, 1.0e-2, 1.0e-3)


def test_exact_single_layer_member_matches_adaptive_quadrature_2d(device):
    """The segment member equals adaptively integrated -log|x-y|/(2 pi)."""
    integrate = pytest.importorskip("scipy.integrate")
    generator = torch.Generator().manual_seed(151)
    start = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    end = start + torch.randn(5, 2, generator=generator, dtype=torch.float64)
    panel_vertices = torch.stack((start, end), dim=1)
    edge = end - start
    length = edge.norm(dim=-1)
    perpendicular = torch.stack((-edge[:, 1], edge[:, 0]), dim=-1) / length[:, None]
    queries = []
    for panel in range(panel_vertices.shape[0]):
        for scale, along, side in zip(
            _QUERY_OFFSET_SCALES, (0.35, -0.2, 0.8, 0.5), (1.0, -1.0, 1.0, -1.0)
        ):
            foot = start[panel] + along * edge[panel]
            queries.append(foot + side * scale * length[panel] * perpendicular[panel])
    queries = torch.stack(queries)

    exact = exact_single_layer_member(
        queries.to(device), panel_vertices.to(device)
    ).cpu()

    reference = torch.zeros_like(exact)
    for i, query in enumerate(queries.numpy()):
        for j in range(panel_vertices.shape[0]):
            p1, p2 = start[j].numpy(), end[j].numpy()

            def integrand(t):
                point = p1 + t * (p2 - p1)
                distance = ((query - point) ** 2).sum() ** 0.5
                return -math.log(distance) / (2.0 * math.pi) * length[j].item()

            value, _ = integrate.quad(
                integrand, 0.0, 1.0, epsabs=1.0e-13, epsrel=1.0e-13, limit=400
            )
            reference[i, j] = value

    near = torch.tensor(
        [scale <= 1.0e-2 for scale in _QUERY_OFFSET_SCALES] * panel_vertices.shape[0]
    )
    torch.testing.assert_close(
        exact[~near], reference[~near], rtol=1.0e-6, atol=1.0e-12
    )
    torch.testing.assert_close(exact[near], reference[near], rtol=1.0e-4, atol=1.0e-12)


def test_exact_single_layer_member_matches_adaptive_quadrature_3d(device):
    """The triangle member equals adaptively integrated 1/(4 pi |x-y|)."""
    integrate = pytest.importorskip("scipy.integrate")
    generator = torch.Generator().manual_seed(157)
    vertices = torch.randn(3, 3, 3, generator=generator, dtype=torch.float64)
    edge_one = vertices[:, 1, :] - vertices[:, 0, :]
    edge_two = vertices[:, 2, :] - vertices[:, 0, :]
    winding = torch.cross(edge_one, edge_two, dim=-1)
    double_area = winding.norm(dim=-1)
    normal = winding / double_area[:, None]
    size = double_area.sqrt()
    queries = []
    for panel in range(vertices.shape[0]):
        for scale, (u, v), side in zip(
            _QUERY_OFFSET_SCALES,
            ((0.9, 0.7), (-0.2, 0.4), (0.25, 0.3), (0.4, 0.35)),
            (1.0, -1.0, -1.0, 1.0),
        ):
            foot = vertices[panel, 0] + u * edge_one[panel] + v * edge_two[panel]
            queries.append(foot + side * scale * size[panel] * normal[panel])
    queries = torch.stack(queries)

    exact = exact_single_layer_member(queries.to(device), vertices.to(device)).cpu()

    reference = torch.zeros_like(exact)
    for i, query in enumerate(queries.numpy()):
        for j in range(vertices.shape[0]):
            origin = vertices[j, 0].numpy()
            e1 = edge_one[j].numpy()
            e2 = edge_two[j].numpy()
            jacobian = double_area[j].item()

            def integrand(v, u):
                point = origin + u * e1 + v * e2
                distance = ((query - point) ** 2).sum() ** 0.5
                return jacobian / (4.0 * math.pi * distance)

            value, _ = integrate.dblquad(
                integrand,
                0.0,
                1.0,
                0.0,
                lambda u: 1.0 - u,
                epsabs=1.0e-11,
                epsrel=1.0e-11,
            )
            reference[i, j] = value

    near = torch.tensor(
        [scale <= 1.0e-2 for scale in _QUERY_OFFSET_SCALES] * vertices.shape[0]
    )
    torch.testing.assert_close(
        exact[~near], reference[~near], rtol=1.0e-6, atol=1.0e-12
    )
    torch.testing.assert_close(exact[near], reference[near], rtol=1.0e-4, atol=1.0e-12)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_single_layer_member_is_orientation_independent(device, n_dims):
    """Reversing vertex order must leave the single layer unchanged.

    This is the deliberate contrast with the double-layer member, which is
    odd in the supplied normal (orientation sign bugs were a recurring
    failure mode); the single layer has no orientation to get wrong and its
    signature takes no normals at all.
    """
    generator = torch.Generator().manual_seed(163)
    if n_dims == 2:
        mesh = _circle_boundary(12, "cpu")
        reversed_order = torch.tensor([1, 0])
    else:
        mesh = _tetrahedron_mesh("cpu")
        reversed_order = torch.tensor([0, 2, 1])
    vertices = mesh.points[mesh.cells].to(device)
    queries = torch.randn(6, n_dims, generator=generator, dtype=torch.float64)
    queries = queries.to(device)

    member = exact_single_layer_member(queries, vertices)
    flipped = exact_single_layer_member(queries, vertices[:, reversed_order, :])

    torch.testing.assert_close(flipped, member, rtol=0.0, atol=1.0e-13)


def test_single_layer_member_is_finite_on_the_panel(device):
    """On-panel and on-vertex queries hit the finite integrable limits.

    2D check is analytic: at a segment midpoint the integral is
    ``l*log(l/2) - l``, so the member is ``-(l*log(l/2) - l)/(2 pi)``.
    """
    segments = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0]], [[1.0, -1.0], [1.5, 0.3]]], dtype=torch.float64
    ).to(device)
    midpoints = segments.mean(dim=1)
    vertices = segments[:, 0, :]
    member = exact_single_layer_member(
        torch.cat((midpoints, vertices), dim=0), segments
    )
    assert torch.isfinite(member).all()
    lengths = (segments[:, 1, :] - segments[:, 0, :]).norm(dim=-1)
    analytic_midpoint = -(lengths * torch.log(lengths / 2.0) - lengths) / (
        2.0 * math.pi
    )
    torch.testing.assert_close(
        member[:2].diagonal(), analytic_midpoint, rtol=1.0e-12, atol=1.0e-12
    )
    analytic_vertex = -(lengths * torch.log(lengths) - lengths) / (2.0 * math.pi)
    torch.testing.assert_close(
        member[2:].diagonal(), analytic_vertex, rtol=1.0e-12, atol=1.0e-12
    )

    tetrahedron = _tetrahedron_mesh(device)
    triangles = tetrahedron.points[tetrahedron.cells]
    centroids = triangles.mean(dim=1)
    member3 = exact_single_layer_member(
        torch.cat((centroids, triangles[:, 0, :]), dim=0), triangles
    )
    assert torch.isfinite(member3).all()
    assert (member3 > 0.0).all()


def test_single_layer_constant_density_on_circle_matches_analytic(device):
    """Constant density on the unit circle: zero inside, -log|x| outside.

    The 2D free-space Green's function makes a unit single layer on the
    circle of radius R produce ``-R log(max(|x|, R))``; on the unit circle
    (the decoder's normalized frame fixes the log gauge at ``|x-y|/L_ref``)
    that is 0 inside and ``-log|x|`` outside.  This is precisely the
    topological (winding) field ``a + b log r`` that has zero double-layer
    representation -- the completeness gap this member closes.
    """
    mesh = _circle_boundary(512, device)
    queries = torch.tensor(
        [
            [0.0, 0.0],
            [0.31, -0.22],
            [-0.5, 0.4],
            [1.5, 0.0],
            [-1.4, 1.4],
            [0.0, -3.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    potential = exact_single_layer_member(queries, mesh.points[mesh.cells]).sum(dim=1)
    radius = queries.norm(dim=-1)
    expected = -torch.log(radius.clamp_min(1.0))
    torch.testing.assert_close(potential, expected, rtol=0.0, atol=5.0e-4)


def test_single_layer_constant_density_on_sphere_matches_analytic(device):
    """Constant density on the unit sphere: R inside, R^2/|x| outside.

    ``int_S dA / (4 pi |x-y|)`` for the unit sphere is 1 at interior points
    and ``1/|x|`` outside; the icosphere triangulation (subdivisions=3,
    1280 faces) reproduces both to the facet discretization error.
    """
    points, cells = _icosphere(3)
    vertices = points[cells].to(device)
    queries = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.35, -0.2, 0.1],
            [-0.1, 0.45, 0.3],
            [1.5, 0.0, 0.0],
            [0.0, -2.0, 1.0],
            [2.0, 2.0, -1.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    potential = exact_single_layer_member(queries, vertices).sum(dim=1)
    expected = 1.0 / queries.norm(dim=-1).clamp_min(1.0)
    torch.testing.assert_close(potential, expected, rtol=5.0e-3, atol=0.0)


def test_exact_single_layer_member_rejects_unsupported_inputs():
    queries4 = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="2D segments and\n?.*3D triangles"):
        exact_single_layer_member(queries4, torch.zeros(3, 2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"shape \(S, 3, 3\)"):
        exact_single_layer_member(
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(3, 2, 3, dtype=torch.float64),
        )


def test_exact_member_rejects_unsupported_inputs():
    queries4 = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="2D segments and\n?.*3D triangles"):
        exact_double_layer_member(
            queries4,
            torch.zeros(3, 2, 4, dtype=torch.float64),
            torch.zeros(3, 4, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"shape \(S, 3, 3\)"):
        exact_double_layer_member(
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(3, 2, 3, dtype=torch.float64),
            torch.zeros(3, 3, dtype=torch.float64),
        )


# ---------------------------------------------------------------------------
# Constructor and cache schema errors.
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_query_decoder_configurations():
    with pytest.raises(ValueError, match="query_decoder must be"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            query_decoder="separable",
        )
    with pytest.raises(ValueError, match="requires n_spatial_dims 2 or 3"):
        MeshTransformer(
            n_spatial_dims=4,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            query_decoder="kernel",
        )
    with pytest.raises(ValueError, match="include_polynomial_members must be a bool"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            query_decoder="kernel",
            kernel_include_polynomial_members=1,
        )
    with pytest.raises(ValueError, match="include_single_layer_member must be a bool"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            query_decoder="kernel",
            kernel_include_single_layer_member=1,
        )


def test_kernel_decoder_requires_simplex_boundary_cells(device):
    """A 3D kernel decoder must reject non-triangle boundary cells clearly."""
    from physicsnemo.experimental.nn.mesh_attention.attention import (
        ScalarVectorState,
    )
    from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
        LinearKernelBasisCrossDecoder,
    )

    torch.manual_seed(53)
    decoder = LinearKernelBasisCrossDecoder(
        n_spatial_dims=3,
        operator_scalar_dim=2,
        operator_vector_dim=1,
        drive_scalar_dim=2,
        drive_vector_dim=1,
        heads=1,
    ).to(device=device, dtype=torch.float64)
    # Segment cells embedded in 3D: wrong arity for the triangle member.
    points = torch.randn(4, 3, device=device, dtype=torch.float64)
    segments = Mesh(points=points, cells=torch.tensor([[0, 1], [2, 3]], device=device))
    state = ScalarVectorState(
        torch.zeros(2, 2, device=device, dtype=torch.float64),
        torch.zeros(2, 1, 3, device=device, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="3 vertices"):
        decoder.build_source_cache(segments, state, state)


def test_decoder_mode_and_cache_mismatches_are_rejected(device):
    queries = torch.tensor([[0.2, 0.1], [0.1, -0.3]], dtype=torch.float64).to(device)
    generator = torch.Generator().manual_seed(59)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    kernel_model = _disk_model("kernel", device)
    encoded = kernel_model.encode(domain)
    stripped = dataclasses.replace(encoded, kernel_cache=None)
    with pytest.raises(ValueError, match="no kernel-decoder cache"):
        kernel_model.decode(stripped)

    moment_model = _disk_model("moment", device)
    with pytest.raises(ValueError, match="query moments"):
        moment_model.decode(encoded)


# ---------------------------------------------------------------------------
# Query-set independence, cache reuse, and initialization.
# ---------------------------------------------------------------------------


def test_kernel_query_set_independence_is_bitwise(device):
    """Decoded values must not depend on which other queries are requested.

    Every query row is contracted with batch-shape-independent reductions,
    so in float64 this holds bitwise at both the decoder-message and full
    decode level, not merely to a tolerance.  ``query_chunk_size=1`` makes
    the full-decode claim principled rather than lucky: the learned query
    blocks use ordinary GEMMs, whose per-row rounding may change with the
    batch shape, so decode is compared per-row (the same discipline as the
    chunk-1/2 models in ``test_model_contracts``).
    """
    model = _disk_model("kernel", device, seed=61, query_chunk_size=1)
    angles = 0.173 + 2.0 * torch.pi * torch.arange(64, dtype=torch.float64) / 64.0
    queries = 0.61 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    queries = queries.to(device)
    generator = torch.Generator().manual_seed(62)
    values = torch.randn(24, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(24, queries, values)
    subset = torch.tensor([13, 2, 40, 5], device=device)

    with torch.no_grad():
        encoded = model.encode(domain)
        message_full = model.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model.kernel_decoder(queries[subset], encoded.kernel_cache)
        full = model.decode(encoded).point_data["potential"]
        partial = model.decode(encoded, Mesh(points=queries[subset]))

    torch.testing.assert_close(
        message_subset.scalars, message_full.scalars[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        message_subset.vectors, message_full.vectors[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
    )


def test_decode_reuses_cached_kernel_source_state(device, monkeypatch):
    """Repeated decodes must reuse the encode-time kernel cache."""
    model = _disk_model("kernel", device)
    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(67)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    encoded = model.encode(_disk_domain(16, queries, values))
    expected = model.decode(encoded)
    assert encoded.query_moments == ()
    assert encoded.kernel_cache is not None

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("decode rebuilt the kernel source cache")

    monkeypatch.setattr(model.kernel_decoder, "build_source_cache", fail_if_rebuilt)
    actual = model.decode(encoded)
    torch.testing.assert_close(
        actual.point_data["potential"],
        expected.point_data["potential"],
        rtol=0.0,
        atol=0.0,
    )


def test_kernel_message_scale_initializes_as_read_in():
    """The kernel message seeds the query state: scale starts at one."""
    model = _disk_model("kernel", "cpu")
    scale = model.kernel_decoder.message_scale
    torch.testing.assert_close(scale.scalar_scale, torch.ones_like(scale.scalar_scale))
    torch.testing.assert_close(scale.vector_scale, torch.ones_like(scale.vector_scale))
    assert len(model.query_blocks) == 0


def test_kernel_mode_supports_vector_outputs_in_two_dimensions(device):
    """2D construction smoke: typed vector predictions flow through kernel mode."""
    torch.manual_seed(71)
    model = (
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0, "flux": 1},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            field_mode="linear",
            query_decoder="kernel",
            operator_scalar_dim=4,
            operator_vector_dim=2,
            drive_scalar_dim=4,
            drive_vector_dim=2,
            operator_layers=1,
            drive_layers=1,
            heads=1,
            scalar_rank=2,
            vector_rank=1,
        )
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(72)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    output = model(_disk_domain(16, queries, values))
    assert output.point_data["potential"].shape == (3,)
    assert output.point_data["flux"].shape == (3, 2)
    assert torch.isfinite(output.point_data["potential"]).all()
    assert torch.isfinite(output.point_data["flux"]).all()


# ---------------------------------------------------------------------------
# Polynomial-member ablation knob.
# ---------------------------------------------------------------------------


def test_polynomial_member_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything.

    Same seed, same parameter tensors in the same order, and bitwise
    identical outputs: the knob at its default leaves the member set --
    exact singular, polynomial {1, b, a}, learned MLP -- untouched.
    """
    reference = _disk_model("kernel", device, seed=79)
    explicit = _disk_model(
        "kernel", device, seed=79, kernel_include_polynomial_members=True
    )
    decoder = reference.kernel_decoder
    assert decoder.include_polynomial_members is True
    assert decoder.n_members == 1 + 3 + decoder.mlp_members

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name, expected in reference_state.items():
        torch.testing.assert_close(explicit_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(80)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("mlp_members", [8, 0])
def test_polynomial_off_ablation_constructs_trains_and_stays_row_stable(
    device, mlp_members
):
    """Polynomial-off arms must construct, run forward/backward, and keep
    bitwise query-set independence.

    ``mlp_members=8`` is the "singular + MLP members only" arm and
    ``mlp_members=0`` the degenerate "singular-only" science arm, whose
    dictionary is the exact double-layer member alone; both are legitimate
    configurations, not errors.
    """
    model = _disk_model(
        "kernel",
        device,
        seed=73,
        kernel_include_polynomial_members=False,
        kernel_mlp_members=mlp_members,
    )
    decoder = model.kernel_decoder
    assert decoder.include_polynomial_members is False
    assert decoder.mlp_members == mlp_members
    assert decoder.n_members == 1 + mlp_members
    if mlp_members == 0:
        assert decoder.member_mlp is None

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4], [-0.4, 0.2]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(74)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    potential = model(domain).point_data["potential"]
    assert potential.shape == (4,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    subset = torch.tensor([2, 0], device=device)
    with torch.no_grad():
        encoded = model.encode(domain)
        message_full = model.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model.kernel_decoder(queries[subset], encoded.kernel_cache)
    torch.testing.assert_close(
        message_subset.scalars, message_full.scalars[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        message_subset.vectors, message_full.vectors[subset], rtol=0.0, atol=0.0
    )


# ---------------------------------------------------------------------------
# Single-layer member knob and the two-member "singpair" science arm.
# ---------------------------------------------------------------------------


def test_single_layer_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything.

    Mirrors ``test_polynomial_member_knob_default_is_bitwise_noop``: with
    ``kernel_include_single_layer_member=False`` spelled out, same seed gives
    the same parameter tensors in the same order and bitwise identical
    outputs, preserving the pruned two-family dictionary exactly.
    """
    reference = _disk_model("kernel", device, seed=107)
    explicit = _disk_model(
        "kernel", device, seed=107, kernel_include_single_layer_member=False
    )
    decoder = reference.kernel_decoder
    assert decoder.include_single_layer_member is False
    assert decoder.n_members == 1 + 3 + decoder.mlp_members

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name, expected in reference_state.items():
        torch.testing.assert_close(explicit_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(108)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_singpair_arm_constructs_trains_and_stays_row_stable_2d(device):
    """The 2D singular-pair arm: exactly two exact members, trainable, stable.

    ``include_single_layer_member=True`` with polynomial and MLP members off
    is the minimal Green-representation-complete dictionary (double layer
    plus single layer); it must construct, run forward/backward with finite
    gradients, and keep the bitwise query-subset independence contract.
    """
    model = _disk_model(
        "kernel",
        device,
        seed=109,
        kernel_include_polynomial_members=False,
        kernel_mlp_members=0,
        # Per-row decode keeps the full-decode bitwise claim independent of
        # batch-shape-dependent GEMM rounding in the learned query blocks.
        query_chunk_size=1,
        kernel_include_single_layer_member=True,
    )
    decoder = model.kernel_decoder
    assert decoder.include_single_layer_member is True
    assert decoder.include_polynomial_members is False
    assert decoder.member_mlp is None
    assert decoder.n_members == 2

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4], [-0.4, 0.2]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(110)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    potential = model(domain).point_data["potential"]
    assert potential.shape == (4,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    subset = torch.tensor([3, 1], device=device)
    with torch.no_grad():
        encoded = model.encode(domain)
        message_full = model.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model.kernel_decoder(queries[subset], encoded.kernel_cache)
        full = model.decode(encoded).point_data["potential"]
        partial = model.decode(encoded, Mesh(points=queries[subset]))
    torch.testing.assert_close(
        message_subset.scalars, message_full.scalars[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        message_subset.vectors, message_full.vectors[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
    )


def test_singpair_arm_constructs_trains_and_stays_row_stable_3d(device):
    """The 3D singular-pair arm mirrors the 2D contract on a triangle mesh."""
    torch.manual_seed(113)
    model = (
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"dirichlet": {"drive": {"boundary_value": 0}}},
            field_mode="linear",
            query_decoder="kernel",
            kernel_include_polynomial_members=False,
            kernel_mlp_members=0,
            kernel_include_single_layer_member=True,
            operator_scalar_dim=7,
            operator_vector_dim=3,
            drive_scalar_dim=9,
            drive_vector_dim=3,
            operator_layers=1,
            drive_layers=1,
            heads=2,
            scalar_rank=4,
            vector_rank=2,
            # Per-row decode keeps the full-decode bitwise claim independent
            # of batch-shape-dependent GEMM rounding in the learned query
            # blocks.
            query_chunk_size=1,
        )
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    assert model.kernel_decoder.n_members == 2

    boundary = _tetrahedron_mesh(device)
    generator = torch.Generator().manual_seed(114)
    values = torch.randn(4, generator=generator, dtype=torch.float64).to(device)
    queries = torch.tensor(
        [
            [0.18, 0.21, 0.17],
            [0.42, 0.11, 0.23],
            [0.25, 0.22, 0.51],
            [0.1, 0.55, 0.2],
        ],
        dtype=torch.float64,
        device=device,
    )
    domain = DomainMesh(
        interior=Mesh(points=queries),
        boundaries={
            "dirichlet": boundary.with_data(cell_data={"boundary_value": values})
        },
    )

    potential = model(domain).point_data["potential"]
    assert potential.shape == (4,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    subset = torch.tensor([2, 0], device=device)
    with torch.no_grad():
        encoded = model.encode(domain)
        normalized = (queries - encoded.center) / encoded.reference_length
        message_full = model.kernel_decoder(normalized, encoded.kernel_cache)
        message_subset = model.kernel_decoder(normalized[subset], encoded.kernel_cache)
        full = model.decode(encoded).point_data["potential"]
        partial = model.decode(encoded, Mesh(points=queries[subset]))
    torch.testing.assert_close(
        message_subset.scalars, message_full.scalars[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
    )


# ---------------------------------------------------------------------------
# Scalar-only (vector-channel-off) encoder ablation.
# ---------------------------------------------------------------------------


def test_scalar_only_knob_default_is_bitwise_noop(device):
    """Explicitly passing the vector-ful defaults must not change anything.

    Mirrors ``test_polynomial_member_knob_default_is_bitwise_noop`` for the
    relaxed vector-dimension validation: constructing with the signature
    defaults spelled out must give the same parameter tensors in the same
    order and bitwise identical outputs as the unstated defaults.
    """

    def build(**overrides) -> MeshTransformer:
        torch.manual_seed(83)
        model = (
            MeshTransformer(
                n_spatial_dims=2,
                output_field_ranks={"potential": 0},
                boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
                field_mode="linear",
                query_decoder="kernel",
                **overrides,
            )
            .to(device=device, dtype=torch.float64)
            .eval()
        )
        for module in model.modules():
            if hasattr(module, "accumulation_dtype"):
                module.accumulation_dtype = torch.float64
        return model

    reference = build()
    explicit = build(operator_vector_dim=8, drive_vector_dim=16, vector_rank=4)
    assert reference.kernel_decoder.vector_value_dim > 0

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name, expected in reference_state.items():
        torch.testing.assert_close(explicit_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(84)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_scalar_only_mode_rejects_incoherent_configurations():
    """Vector widths must be both zero or both positive, with rank-1 I/O gone.

    The scalar-only coherence rule: a lone zero vector width would leave one
    stream's vectors without an equivariant read-out path, a positive
    ``vector_rank`` would be silently ignored, and rank-1 outputs or drive
    inputs have no representation basis without encoder vector channels.
    """
    base = dict(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
        query_decoder="kernel",
    )
    with pytest.raises(ValueError, match="both zero"):
        MeshTransformer(**base, operator_vector_dim=0)
    with pytest.raises(ValueError, match="both zero"):
        MeshTransformer(**base, drive_vector_dim=0)
    with pytest.raises(ValueError, match="requires vector_rank=0"):
        MeshTransformer(**base, operator_vector_dim=0, drive_vector_dim=0)
    scalar_only = dict(operator_vector_dim=0, drive_vector_dim=0, vector_rank=0)
    with pytest.raises(ValueError, match="rank-1 output"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0, "flux": 1},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            query_decoder="kernel",
            **scalar_only,
        )
    with pytest.raises(ValueError, match="rank-1 drive"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={
                "disk": {"drive": {"boundary_value": 0, "traction": 1}}
            },
            query_decoder="kernel",
            **scalar_only,
        )
    with pytest.raises(ValueError, match="rank-1 global drive"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
            global_field_ranks={"drive": {"freestream": 1}},
            query_decoder="kernel",
            **scalar_only,
        )
    # Rank-1 *operator* inputs stay legal: they reach the scalar stream as
    # Gram invariants against the position and normal channels.
    MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={
            "disk": {"operator": {"anisotropy": 1}, "drive": {"boundary_value": 0}}
        },
        query_decoder="kernel",
        **scalar_only,
    )


@pytest.mark.parametrize("include_polynomial", [False, True])
def test_scalar_only_kernel_arm_constructs_trains_and_stays_row_stable(
    device, include_polynomial
):
    """Scalar-only kernel arms must construct, train, and stay row stable.

    ``operator_vector_dim=drive_vector_dim=vector_rank=0`` removes the
    encoder's oriented (rank-1) state entirely; boundary normals still reach
    the decoder through its pair invariants and exact double-layer member.
    Every remaining (nonzero-width) parameter must receive a finite gradient,
    and the bitwise query-subset independence contract must survive with
    zero-width vector streams.
    """
    model = _disk_model(
        "kernel",
        device,
        seed=89,
        operator_vector_dim=0,
        drive_vector_dim=0,
        vector_rank=0,
        kernel_mlp_members=0,
        kernel_include_polynomial_members=include_polynomial,
        # Per-row decode keeps the full-decode bitwise claim independent of
        # batch-shape-dependent GEMM rounding in the learned query blocks.
        query_chunk_size=1,
    )
    decoder = model.kernel_decoder
    assert decoder.operator_vector_dim == 0
    assert decoder.drive_vector_dim == 0
    assert decoder.vector_value_dim == 0
    assert decoder.member_mlp is None
    assert decoder.n_members == 1 + (3 if include_polynomial else 0)
    assert decoder.vector_output_weight.numel() == 0

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4], [-0.4, 0.2]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(90)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    potential = model(domain).point_data["potential"]
    assert potential.shape == (4,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"

    subset = torch.tensor([2, 0, 3], device=device)
    with torch.no_grad():
        encoded = model.encode(domain)
        message_full = model.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model.kernel_decoder(queries[subset], encoded.kernel_cache)
        full = model.decode(encoded).point_data["potential"]
        partial = model.decode(encoded, Mesh(points=queries[subset]))
    assert encoded.operator_state.vectors.shape[1] == 0
    assert encoded.drive_state.vectors.shape[1] == 0
    assert encoded.kernel_cache.pair_vectors.shape[1] == 0
    assert encoded.kernel_cache.value_vectors.shape[2] == 0
    assert message_full.vectors.shape == (4, 0, 2)
    torch.testing.assert_close(
        message_subset.scalars, message_full.scalars[subset], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
    )


def test_scalar_only_moment_decoder_constructs_and_trains(device):
    """The relaxed vector widths must also hold for the separable decoder."""
    model = _disk_model(
        "moment",
        device,
        seed=97,
        operator_vector_dim=0,
        drive_vector_dim=0,
        vector_rank=0,
    )
    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(98)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    potential = model(_disk_domain(16, queries, values)).point_data["potential"]
    assert potential.shape == (3,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# Bounded output-gate-invariant knob (the far-field gate-collapse fix).
# ---------------------------------------------------------------------------


def test_bounded_output_gate_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything.

    Mirrors ``test_polynomial_member_knob_default_is_bitwise_noop``: same
    seed gives the same parameter tensors in the same order and bitwise
    identical outputs.  Because the knob adds no parameters (it only
    compactifies the invariants feeding the output projection's sigmoid
    gates), the knob-ON state dict is bitwise identical too -- the change
    lives entirely in the gate's input map.
    """
    reference = _disk_model("kernel", device, seed=101)
    explicit = _disk_model(
        "kernel", device, seed=101, bounded_output_gate_invariants=False
    )
    bounded = _disk_model(
        "kernel", device, seed=101, bounded_output_gate_invariants=True
    )
    assert reference.output_projection.bounded_gate_invariants is False
    assert explicit.output_projection.bounded_gate_invariants is False
    assert bounded.output_projection.bounded_gate_invariants is True

    reference_state = reference.state_dict()
    for other in (explicit, bounded):
        other_state = other.state_dict()
        assert list(reference_state) == list(other_state)
        for name, expected in reference_state.items():
            torch.testing.assert_close(
                other_state[name], expected, rtol=0.0, atol=0.0
            )

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(102)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_bounded_output_gate_arm_constructs_trains_and_stays_row_stable(device):
    """The knob-on arm must run forward/backward -- including at query radii
    far beyond the boundary scale, the regime whose raw-gate saturation
    motivated the knob -- and keep bitwise query-subset independence.
    """
    model = _disk_model(
        "kernel",
        device,
        seed=103,
        bounded_output_gate_invariants=True,
        query_chunk_size=1,
    )
    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [12.0, 9.0], [-40.0, 25.0]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(104)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    potential = model(domain).point_data["potential"]
    assert potential.shape == (4,)
    assert torch.isfinite(potential).all()
    potential.square().sum().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    subset = torch.tensor([3, 0], device=device)
    with torch.no_grad():
        encoded = model.encode(domain)
        full = model.decode(encoded).point_data["potential"]
        partial = model.decode(encoded, Mesh(points=queries[subset]))
    torch.testing.assert_close(
        partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
    )


# ---------------------------------------------------------------------------
# Bounded (compactified) query-position injection: the source-side completion
# of the far-field fix.  The gate knob above bounded only the gate INPUTS and
# fired its falsifier (it unmasked polynomially growing direct-drive
# branches); this knob compactifies the injected query position itself, so
# every learned query-radius dependence is bounded at once while the kernel
# dictionary's exact members keep the raw coordinates.
# ---------------------------------------------------------------------------


def test_bounded_query_geometry_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything.

    Mirrors ``test_bounded_output_gate_knob_default_is_bitwise_noop``: the
    knob adds no parameters (it only compactifies the query position fed to
    the operator-state injection), so the knob-ON state dict is bitwise
    identical too -- the change lives entirely in the injection map.
    """
    reference = _disk_model("kernel", device, seed=107)
    explicit = _disk_model("kernel", device, seed=107, bounded_query_geometry=False)
    bounded = _disk_model("kernel", device, seed=107, bounded_query_geometry=True)
    assert reference.bounded_query_geometry is False
    assert explicit.bounded_query_geometry is False
    assert bounded.bounded_query_geometry is True

    reference_state = reference.state_dict()
    for other in (explicit, bounded):
        other_state = other.state_dict()
        assert list(reference_state) == list(other_state)
        for name, expected in reference_state.items():
            torch.testing.assert_close(other_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(108)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_bounded_query_geometry_changes_only_the_learned_coefficient_path(device):
    """The knob must compactify the injection and NOTHING else.

    The kernel decoder's pair invariants and exact singular members read the
    raw normalized query points through a separate ``decode`` argument --
    they are the physics and must keep their exact, unbounded radial
    dependence.  With identical weights, the knob-on model must therefore
    produce a bitwise identical source encoding and bitwise identical kernel
    messages at arbitrary radii, while the injected position channel becomes
    the compactified x/sqrt(1+|x|^2) (direction preserved, magnitude in
    [0, 1)) and the decoded output -- whose gates and direct-drive geometry
    vectors read the injection -- changes.
    """
    reference = _disk_model("kernel", device, seed=109)
    bounded = _disk_model("kernel", device, seed=110, bounded_query_geometry=True)
    bounded.load_state_dict(reference.state_dict())  # knob adds no parameters

    queries = torch.tensor(
        [[0.2, 0.1], [12.0, 9.0], [-300.0, 400.0]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(111)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    with torch.no_grad():
        encoded_reference = reference.encode(domain)
        encoded_bounded = bounded.encode(domain)
        # Source side untouched: the boundary is compact, so nothing on the
        # encode path reads a query position.
        cache_reference = encoded_reference.kernel_cache
        cache_bounded = encoded_bounded.kernel_cache
        for name in ("coefficients", "value_scalars", "value_vectors", "weights"):
            torch.testing.assert_close(
                getattr(cache_bounded, name),
                getattr(cache_reference, name),
                rtol=0.0,
                atol=0.0,
            )

        normalized = (
            queries - encoded_reference.center
        ) / encoded_reference.reference_length
        # Exact-member path untouched: bitwise identical kernel messages far
        # beyond any training radius.
        message_reference = reference.kernel_decoder(normalized, cache_reference)
        message_bounded = bounded.kernel_decoder(normalized, cache_bounded)
        torch.testing.assert_close(
            message_bounded.scalars, message_reference.scalars, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            message_bounded.vectors, message_reference.vectors, rtol=0.0, atol=0.0
        )

        # The injection itself: raw |x| for the reference, x/sqrt(1+|x|^2)
        # for the bounded model (same direction, magnitude in [0, 1)).  The
        # disk schema has no boundary/global operator vectors, so channel 0
        # is the injected position.
        injected_reference = reference._query_operator_input(
            normalized, encoded_reference.global_operator_state
        ).vectors[:, 0, :]
        injected_bounded = bounded._query_operator_input(
            normalized, encoded_bounded.global_operator_state
        ).vectors[:, 0, :]
        radii = normalized.norm(dim=-1)
        torch.testing.assert_close(injected_reference, normalized, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            injected_bounded,
            normalized * (1.0 + radii.square()).rsqrt()[:, None],
        )
        assert (injected_bounded.norm(dim=-1) < 1.0).all()

        # The learned-coefficient path DID change: the decoded output reads
        # the injection through the operator lift, gates, and direct-drive
        # geometry vectors.
        full_reference = reference.decode(encoded_reference).point_data["potential"]
        full_bounded = bounded.decode(encoded_bounded).point_data["potential"]
        assert torch.isfinite(full_bounded).all()
        assert not torch.allclose(full_bounded, full_reference)

        # Bitwise query-subset independence holds with the knob on.
        subset = torch.tensor([2, 0], device=device)
        partial = bounded.decode(encoded_bounded, Mesh(points=queries[subset]))
    torch.testing.assert_close(
        partial.point_data["potential"], full_bounded[subset], rtol=0.0, atol=0.0
    )

    # Forward/backward stays finite at query radii far beyond the boundary
    # scale -- the regime the knob exists for.
    potential = bounded(domain).point_data["potential"]
    potential.square().sum().backward()
    gradients = [
        parameter.grad for parameter in bounded.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


# ---------------------------------------------------------------------------
# Iteration 30's far-field DECAY STRUCTURE (fourth rung of the ladder).
# History: iteration 27 diagnosed the exterior far-field failure as query-side
# coefficient extrapolation (gate collapse); 28 bounded the gate inputs only
# and fired its falsifier (the collapse had been load-bearing over growing
# direct-drive branches); 29 bounded the query geometry at the source --
# coefficient extrapolation excluded by construction -- but plateaued because
# BOUNDED IS NOT DECAYING: the direct drive converged to a direction-dependent
# constant while the exact field decays like r^-2, and the single-layer
# member's log-r tail was fit by near-cancellation.  The two knobs below add
# the missing structure: an analytic 1/(1+|x|^2) envelope on the query-side
# direct drive, and a zero-net-charge deflation of the single-layer member.
# ---------------------------------------------------------------------------


def _freestream_disk_model(
    device: torch.device | str,
    *,
    seed: int,
    **overrides,
) -> MeshTransformer:
    """A disk model with a global rank-1 drive, so the query-side
    direct-drive path (the lifted global drive at the query) is live."""
    return _disk_model(
        "kernel",
        device,
        seed=seed,
        global_field_ranks={"drive": {"freestream": 1}},
        **overrides,
    )


def _freestream_disk_domain(
    n_boundary: int,
    query_points: torch.Tensor,
    boundary_values: torch.Tensor,
    freestream: torch.Tensor,
) -> DomainMesh:
    boundary = _circle_boundary(
        n_boundary, query_points.device, dtype=query_points.dtype
    )
    return DomainMesh(
        interior=Mesh(points=query_points),
        boundaries={
            "disk": boundary.with_data(cell_data={"boundary_value": boundary_values})
        },
        global_data={"freestream": freestream},
    )


def test_decaying_direct_drive_requires_the_kernel_decoder(device):
    """The knob's premise (exact members carry the radial physics, the direct
    drive is a bounded local term) only exists in kernel mode; the moment
    decoder has no direct/member split, so the flag must be rejected."""
    with pytest.raises(ValueError, match="decaying_direct_drive requires"):
        _disk_model("moment", device, seed=113, decaying_direct_drive=True)


def test_decaying_direct_drive_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything, and the
    knob adds no parameters (state dicts interchangeable; the flag alone
    selects the parameterization)."""
    reference = _freestream_disk_model(device, seed=115)
    explicit = _freestream_disk_model(device, seed=115, decaying_direct_drive=False)
    decaying = _freestream_disk_model(device, seed=115, decaying_direct_drive=True)
    assert reference.decaying_direct_drive is False
    assert explicit.decaying_direct_drive is False
    assert decaying.decaying_direct_drive is True

    reference_state = reference.state_dict()
    for other in (explicit, decaying):
        other_state = other.state_dict()
        assert list(reference_state) == list(other_state)
        for name, expected in reference_state.items():
            torch.testing.assert_close(other_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(116)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    freestream = torch.tensor([0.8, -0.5], dtype=torch.float64, device=device)
    domain = _freestream_disk_domain(16, queries, values, freestream)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_decaying_direct_drive_scales_only_the_direct_contribution(device):
    """The envelope must multiply exactly the direct (non-member-mediated)
    part by 1/(1+|x|^2) of the RAW normalized radius, and nothing else.

    Contracts checked with shared weights: (i) the source encoding and the
    kernel messages are bitwise untouched at arbitrary radii (the exact
    members keep reading raw positions); (ii) a zero global drive makes the
    knob a bitwise no-op (the member-mediated path never sees the envelope);
    (iii) the decoded output obeys prediction_on = message_part +
    envelope * direct_part, with the envelope of the RAW radius even when
    ``bounded_query_geometry`` compactifies the learned injection -- the two
    knobs compose without the envelope reading the compactified radius.
    """
    reference = _freestream_disk_model(device, seed=117)
    decaying = _freestream_disk_model(device, seed=118, decaying_direct_drive=True)
    decaying.load_state_dict(reference.state_dict())  # knob adds no parameters
    composed = _freestream_disk_model(
        device,
        seed=119,
        decaying_direct_drive=True,
        bounded_query_geometry=True,
    )
    composed.load_state_dict(reference.state_dict())

    queries = torch.tensor(
        [[0.2, 0.1], [1.5, -0.7], [12.0, 9.0], [-300.0, 400.0]],
        dtype=torch.float64,
    ).to(device)
    generator = torch.Generator().manual_seed(120)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    freestream = torch.tensor([0.8, -0.5], dtype=torch.float64, device=device)
    domain = _freestream_disk_domain(16, queries, values, freestream)

    with torch.no_grad():
        encoded_reference = reference.encode(domain)
        encoded_decaying = decaying.encode(domain)
        # (i) Source side and kernel messages bitwise untouched.
        for name in ("coefficients", "value_scalars", "value_vectors", "weights"):
            torch.testing.assert_close(
                getattr(encoded_decaying.kernel_cache, name),
                getattr(encoded_reference.kernel_cache, name),
                rtol=0.0,
                atol=0.0,
            )
        normalized = (
            queries - encoded_reference.center
        ) / encoded_reference.reference_length
        message = reference.kernel_decoder(normalized, encoded_reference.kernel_cache)
        message_decaying = decaying.kernel_decoder(
            normalized, encoded_decaying.kernel_cache
        )
        torch.testing.assert_close(
            message_decaying.scalars, message.scalars, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            message_decaying.vectors, message.vectors, rtol=0.0, atol=0.0
        )

        # (ii) Zero global drive: the direct path is exactly zero, so the
        # knob is a bitwise no-op and the member-mediated path is untouched.
        quiescent = _freestream_disk_domain(
            16, queries, values, torch.zeros_like(freestream)
        )
        torch.testing.assert_close(
            decaying(quiescent).point_data["potential"],
            reference(quiescent).point_data["potential"],
            rtol=0.0,
            atol=0.0,
        )

        # (iii) The exact envelope identity.  The output projection is
        # linear in its field state, so the message part of the output is
        # computable directly and the direct part is the remainder.
        envelope = 1.0 / (1.0 + normalized.square().sum(dim=-1))
        prediction_reference = reference.decode(encoded_reference).point_data[
            "potential"
        ]
        prediction_decaying = decaying.decode(encoded_decaying).point_data["potential"]
        assert not torch.allclose(prediction_decaying, prediction_reference)
        for model, encoded in ((decaying, encoded_decaying), (composed, None)):
            encoded = model.encode(domain) if encoded is None else encoded
            query_operator = model.operator_input_block(
                model.operator_lift(
                    model._query_operator_input(
                        normalized, encoded.global_operator_state
                    )
                )
            )
            message_part = model.output_projection(query_operator, message).scalars[
                :, 0
            ]
            # The knob-off direct part under this model's query operator:
            # rebuild it from a knob-off twin sharing the same weights.
            twin = _freestream_disk_model(
                device,
                seed=121,
                bounded_query_geometry=model.bounded_query_geometry,
            )
            twin.load_state_dict(reference.state_dict())
            prediction_off = twin.decode(twin.encode(domain)).point_data["potential"]
            direct_part = prediction_off - message_part
            torch.testing.assert_close(
                model.decode(encoded).point_data["potential"],
                message_part + envelope * direct_part,
                rtol=1.0e-9,
                atol=1.0e-12,
            )

        # Bitwise query-subset independence holds with the knob on.
        subset = torch.tensor([3, 0], device=device)
        partial = decaying.decode(encoded_decaying, Mesh(points=queries[subset]))
        torch.testing.assert_close(
            partial.point_data["potential"],
            prediction_decaying[subset],
            rtol=0.0,
            atol=0.0,
        )

    # Forward/backward stays finite at far radii with the knob on.
    potential = decaying(domain).point_data["potential"]
    potential.square().sum().backward()
    gradients = [
        parameter.grad
        for parameter in decaying.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


# ---------------------------------------------------------------------------
# Monopole-free single layer: the second decay-structure knob.
# ---------------------------------------------------------------------------


def _singpair_model(device, *, seed: int, **overrides) -> MeshTransformer:
    return _disk_model(
        "kernel",
        device,
        seed=seed,
        kernel_include_polynomial_members=False,
        kernel_mlp_members=0,
        kernel_include_single_layer_member=True,
        **overrides,
    )


def test_monopole_free_single_layer_requires_the_single_layer_member(device):
    """Without the single-layer member there is no monopole to control."""
    with pytest.raises(ValueError, match="monopole_free_single_layer"):
        _disk_model("kernel", device, seed=123, kernel_monopole_free_single_layer=True)


def test_monopole_free_single_layer_knob_default_is_bitwise_noop(device):
    """Explicitly passing the default knob must not change anything, and the
    knob adds no parameters (the deflation is a fixed rank-one projection of
    the member column, so state dicts are interchangeable)."""
    reference = _singpair_model(device, seed=125)
    explicit = _singpair_model(
        device, seed=125, kernel_monopole_free_single_layer=False
    )
    deflated = _singpair_model(device, seed=125, kernel_monopole_free_single_layer=True)
    assert reference.kernel_decoder.monopole_free_single_layer is False
    assert explicit.kernel_decoder.monopole_free_single_layer is False
    assert deflated.kernel_decoder.monopole_free_single_layer is True
    assert deflated.kernel_decoder.n_members == 2

    reference_state = reference.state_dict()
    for other in (explicit, deflated):
        other_state = other.state_dict()
        assert list(reference_state) == list(other_state)
        for name, expected in reference_state.items():
            torch.testing.assert_close(other_state[name], expected, rtol=0.0, atol=0.0)

    queries = torch.tensor(
        [[0.2, 0.1], [0.1, -0.3], [0.5, 0.4]], dtype=torch.float64
    ).to(device)
    generator = torch.Generator().manual_seed(126)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)
    with torch.no_grad():
        expected = reference(domain).point_data["potential"]
        actual = explicit(domain).point_data["potential"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_monopole_free_single_layer_kills_the_monopole_tail_structurally(device):
    """The deflation is exactly the measure-weighted rank-one projection, and
    it kills the log-r tail for ANY conditioned coefficients.

    Contracts with shared weights: (i) the source cache is bitwise untouched
    (the projection acts on the member column at decode time, never on the
    conditioned coefficient/value tensors); (ii) the knob-on minus knob-off
    message difference is proportional, per output channel, to the
    uniform-density boundary potential M(x) = sum_s member_SL(x, s) -- the
    exact rank-one fingerprint, confirming the double layer is untouched;
    (iii) far out, the undeflated message GROWS (the generic net monopole's
    log-r tail) while the deflated message DECAYS like 1/r (dipole-and-up
    survivors) -- structural absence, not fitted cancellation.
    """
    reference = _singpair_model(device, seed=127)
    deflated = _singpair_model(device, seed=128, kernel_monopole_free_single_layer=True)
    deflated.load_state_dict(reference.state_dict())  # knob adds no parameters

    queries = torch.tensor(
        [[1.7, 0.4], [-2.3, 1.1], [5.0, -3.0], [0.4, 9.0], [-20.0, 12.0]],
        dtype=torch.float64,
    ).to(device)
    generator = torch.Generator().manual_seed(129)
    values = torch.randn(16, generator=generator, dtype=torch.float64).to(device)
    domain = _disk_domain(16, queries, values)

    with torch.no_grad():
        encoded_reference = reference.encode(domain)
        encoded_deflated = deflated.encode(domain)
        # (i) Conditioned coefficients and values bitwise untouched.
        for name in ("coefficients", "value_scalars", "value_vectors", "weights"):
            torch.testing.assert_close(
                getattr(encoded_deflated.kernel_cache, name),
                getattr(encoded_reference.kernel_cache, name),
                rtol=0.0,
                atol=0.0,
            )

        normalized = (
            queries - encoded_reference.center
        ) / encoded_reference.reference_length
        message_reference = reference.kernel_decoder(
            normalized, encoded_reference.kernel_cache
        )
        message_deflated = deflated.kernel_decoder(
            normalized, encoded_deflated.kernel_cache
        )
        # (ii) Rank-one fingerprint: diff(x, channel) = M(x) * k_channel.
        uniform_potential = exact_single_layer_member(
            normalized, encoded_reference.kernel_cache.panel_vertices
        ).sum(dim=-1)
        for diff in (
            (message_deflated.scalars - message_reference.scalars).flatten(1),
            (message_deflated.vectors - message_reference.vectors).flatten(1),
        ):
            per_channel = diff / uniform_potential[:, None]
            spread = per_channel.amax(dim=0) - per_channel.amin(dim=0)
            scale = per_channel.abs().amax(dim=0).clamp_min(1.0e-12)
            assert float((spread / scale).max()) < 1.0e-9

        # (iii) Structural tail: undeflated grows (log r), deflated decays
        # (~1/r) between r = 1e3 and r = 1e6.
        def message_norm(model, cache, radius: float) -> float:
            point = torch.tensor(
                [[0.6 * radius, 0.8 * radius]],
                dtype=torch.float64,
                device=device,
            )
            state = model.kernel_decoder(point, cache)
            return float(
                torch.cat((state.scalars.flatten(), state.vectors.flatten())).norm()
            )

        near_off = message_norm(reference, encoded_reference.kernel_cache, 1.0e3)
        far_off = message_norm(reference, encoded_reference.kernel_cache, 1.0e6)
        near_on = message_norm(deflated, encoded_deflated.kernel_cache, 1.0e3)
        far_on = message_norm(deflated, encoded_deflated.kernel_cache, 1.0e6)
        assert far_off > 1.2 * near_off, "undeflated log tail should grow"
        assert far_on < near_on / 100.0, "deflated message should decay like 1/r"

        # Bitwise query-subset independence holds with the knob on.
        subset = torch.tensor([4, 1], device=device)
        full = deflated.decode(encoded_deflated).point_data["potential"]
        partial = deflated.decode(encoded_deflated, Mesh(points=queries[subset]))
        torch.testing.assert_close(
            partial.point_data["potential"], full[subset], rtol=0.0, atol=0.0
        )

    # Forward/backward stays finite with the knob on.
    potential = deflated(domain).point_data["potential"]
    potential.square().sum().backward()
    gradients = [
        parameter.grad
        for parameter in deflated.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


# ---------------------------------------------------------------------------
# The angular-order capability that motivates the kernel decoder.
# ---------------------------------------------------------------------------


def _ring_fourier_high_order_ratio(
    model: MeshTransformer,
    device: torch.device | str,
    *,
    seed: int,
) -> float:
    """Largest relative Fourier coefficient of order >= 3 on a centered ring.

    This ports the random-weight ring scan of the benchmark's spectral
    diagnostics: on a centered disk with a scalar linear decoder, the O(2)
    equivariant map is diagonal in angular order, so ring content of order
    three or higher is exactly the capability the separable moment decoder
    lacks.
    """
    n_boundary, n_ring = 24, 64
    angles = 0.173 + 2.0 * torch.pi * torch.arange(n_ring, dtype=torch.float64) / (
        n_ring
    )
    queries = 0.61 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    queries = queries.to(device)
    generator = torch.Generator().manual_seed(seed + 10_000)
    values = torch.randn(n_boundary, generator=generator, dtype=torch.float64)
    domain = _disk_domain(n_boundary, queries, values.to(device))

    with torch.no_grad():
        ring = model(domain).point_data["potential"]

    modes = torch.arange(0, n_ring // 2 + 1, dtype=torch.float64, device=device)
    analysis = torch.exp(
        -1j * angles.to(device)[:, None].to(torch.complex128) * modes[None, :]
    )
    coefficients = torch.einsum("am,a->m", analysis, ring.to(torch.complex128)) / n_ring
    high = coefficients[3:].abs().max().item()
    scale = max(ring.abs().max().item(), coefficients[:3].abs().max().item())
    return high / (1.0 + scale)


@pytest.mark.parametrize("seed", [2111, 2113])
def test_random_weight_kernel_decoder_produces_high_angular_orders(device, seed):
    """The kernel decoder must emit angular orders the moment decoder cannot.

    These two assertions document the fix together: the separable moment
    decoder has an exact ``m <= 2`` ring ceiling (README section 6.2; the
    benchmark's spectral tests assert it across widths and depths), while the
    dense kernel decoder's pair members -- the subtended-angle singular
    member alone -- carry every angular order.
    """
    moment_ratio = _ring_fourier_high_order_ratio(
        _disk_model("moment", device, seed=seed), device, seed=seed
    )
    kernel_ratio = _ring_fourier_high_order_ratio(
        _disk_model("kernel", device, seed=seed), device, seed=seed
    )
    assert moment_ratio <= 2.0e-11
    assert kernel_ratio >= 1.0e-6
    assert kernel_ratio > 1.0e3 * moment_ratio
