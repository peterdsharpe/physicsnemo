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

r"""Encoder-stress testbed: 2D Laplace on geometries that force all-to-all
boundary information flow.

Every existing 2D benchmark in this example lives on a (deformed) disk, where
the boundary-to-interior map is well approximated by kernels of *pairwise*
euclidean invariants: for convex domains, two boundary points that are close
in :math:`\mathbb{R}^2` are also close along the boundary, so a pair kernel
never has to "see around" geometry.  The architectural claim behind the
attention encoder -- that boundary cells must exchange information globally
before the query decode -- is therefore untested.  This file adds two exact
Laplace benchmark families whose Poisson kernels are *not* functions of pair
invariants:

**Family A -- multi-body** (``family="multi_body"``): the domain is the
interior of a circle minus two disjoint disks (multiply connected, three
Dirichlet boundaries).  Exact solutions come from exterior point charges,

.. math::

   u(x) = \sum_j q_j \log \lVert x - c_j \rVert ,

with the :math:`c_j` placed strictly outside the domain closure: inside the
two excluded disks and outside the outer circle.  Some in-disk charges are
deliberately biased toward the gap-facing surface of their disk, so the
narrow-gap regime couples the two bodies strongly: the Dirichlet trace on the
near face of one disk is dominated by a charge hidden inside the *other*
disk.  The difficulty dial is the surface gap over the sum of disk radii.

**Family B -- deep cavity** (``family="deep_cavity"``): a simply connected
domain whose boundary is a disk with a deep, narrow, smoothly filleted slot
(a "C-shape").  The curve is piecewise arc/line/fillet with tangent
continuity, resampled at equal arclength; simplicity (non-self-intersection)
is verified numerically for every sample.  Charges are placed *inside the
slot* -- outside the domain but nestled deep in the cavity -- so two interior
points on opposite cavity walls are euclidean-close while the field between
them is mediated by the whole cavity geometry.  The difficulty dial is the
slot depth over its width.

Both families follow the linear-benchmark conventions exactly: boundaries are
2D line-segment cells with exact ``boundary_value`` Dirichlet data at true
curve parameter midpoints, all boundaries are merged into one ``"dirichlet"``
:class:`~physicsnemo.mesh.Mesh` (identity conveyed geometrically), targets are
the exact potential at rejection-sampled interior queries, and
``reference_length`` sits in ``global_data``.  Charges are renormalized so the
boundary trace has unit RMS; over-peaked draws are rejected deterministically.

Every generated sample is *certified* at build time: cell-orientation winding
numbers place all queries inside and all charges outside the domain, charge
standoff from the discretized boundary is checked numerically, the cavity
curve passes an exact all-pairs segment-intersection test, and the multi-body
boundaries are verified disjoint.

This is a benchmark-local research prototype, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import _paths  # noqa: F401
import torch
from models import (
    BoundaryMean,
    InvariantPairKernel,
    MeshTransformerConfig,
    build_mesh_transformer,
)
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

_TWO_PI = 2.0 * math.pi


def _substream(seed: int, stream: int) -> int:
    """Derive independent deterministic seeds without mutable RNG state."""

    return seed + 15_485_863 * stream


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(
        torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator)
    )


# ---------------------------------------------------------------------------
# Exact solution: exterior logarithmic point charges
# ---------------------------------------------------------------------------


def harmonic_potential(
    points: torch.Tensor,
    charge_positions: torch.Tensor,
    charge_strengths: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`u(x) = \sum_j q_j \log\lVert x - c_j\rVert`.

    Exactly harmonic wherever :math:`x \neq c_j`; the generators guarantee all
    charges keep a positive standoff from the domain closure.
    """

    displacement = points[:, None, :] - charge_positions[None, :, :]
    distance = displacement.norm(dim=-1)
    return (charge_strengths[None, :] * torch.log(distance)).sum(dim=-1)


def harmonic_gradient(
    points: torch.Tensor,
    charge_positions: torch.Tensor,
    charge_strengths: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`\nabla u = \sum_j q_j (x - c_j)/\lVert x - c_j\rVert^2`."""

    displacement = points[:, None, :] - charge_positions[None, :, :]
    distance_sq = displacement.square().sum(dim=-1)
    return (
        charge_strengths[None, :, None] * displacement / distance_sq[..., None]
    ).sum(dim=1)


# ---------------------------------------------------------------------------
# Geometry predicates (winding, simplicity, distances)
# ---------------------------------------------------------------------------


def winding_number(loop: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Integer winding of a closed vertex loop around each point.

    ``loop`` is an ordered ``(n, 2)`` vertex polygon (implicitly closed);
    counterclockwise loops give ``+1`` for enclosed points.  Points on the
    polygon itself are undefined; callers keep a positive margin.
    """

    start = loop[None, :, :] - points[:, None, :]
    end = torch.roll(loop, -1, dims=0)[None, :, :] - points[:, None, :]
    cross = start[..., 0] * end[..., 1] - start[..., 1] * end[..., 0]
    dot = (start * end).sum(dim=-1)
    total = torch.atan2(cross, dot).sum(dim=-1)
    return torch.round(total / _TWO_PI).long()


def mesh_cell_winding(boundary: Mesh, points: torch.Tensor) -> torch.Tensor:
    """Integer winding of a segment-cell boundary mesh around each point.

    Sums the signed subtended angle of every cell in its *cell* orientation,
    so this validates panel winding (the benchmark's outward-normal
    convention makes closed boundaries wind ``-1`` around interior points).
    """

    vertices = boundary.points[boundary.cells].to(points.dtype)
    start = vertices[None, :, 0, :] - points[:, None, :]
    end = vertices[None, :, 1, :] - points[:, None, :]
    cross = start[..., 0] * end[..., 1] - start[..., 1] * end[..., 0]
    dot = (start * end).sum(dim=-1)
    total = torch.atan2(cross, dot).sum(dim=-1)
    return torch.round(total / _TWO_PI).long()


def polyline_is_simple(loop: torch.Tensor) -> bool:
    """Exact all-pairs test that a closed polygon does not self-intersect.

    Non-adjacent segment pairs must not intersect (proper crossings,
    touchings, and collinear overlaps all count as intersections); adjacent
    segments share exactly one endpoint by construction.
    """

    n = loop.shape[0]
    if n < 3:
        return False
    p = loop
    q = torch.roll(loop, -1, dims=0)

    def orient(a, b, c):  # sign of cross(b - a, c - a)
        return torch.sign(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    a, b = p[:, None, :], q[:, None, :]  # segment i
    c, d = p[None, :, :], q[None, :, :]  # segment j
    intersects = (
        (
            (orient(a, b, c) * orient(a, b, d) < 0)
            & (orient(c, d, a) * orient(c, d, b) < 0)
        )
        | (
            # Conservative: any degenerate/collinear contact counts as a failure.
            (orient(a, b, c) == 0) & _on_segment(a, b, c)
        )
        | ((orient(a, b, d) == 0) & _on_segment(a, b, d))
    )
    index = torch.arange(n)
    gap = (index[:, None] - index[None, :]) % n
    non_adjacent = (gap != 0) & (gap != 1) & (gap != n - 1)
    return bool((~(intersects & non_adjacent)).all())


def _on_segment(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return ((c >= lo) & (c <= hi)).all(dim=-1)


def _min_distance_to_loop(points: torch.Tensor, loop: torch.Tensor) -> torch.Tensor:
    """Minimum euclidean distance from each point to a closed vertex loop."""

    start = loop
    edge = torch.roll(loop, -1, dims=0) - loop
    edge_sq = edge.square().sum(dim=-1).clamp_min(1.0e-30)
    offset = points[:, None, :] - start[None, :, :]
    t = ((offset * edge[None, :, :]).sum(dim=-1) / edge_sq[None, :]).clamp(0.0, 1.0)
    nearest = start[None, :, :] + t[..., None] * edge[None, :, :]
    return (points[:, None, :] - nearest).norm(dim=-1).min(dim=-1).values


def _signed_area(loop: torch.Tensor) -> float:
    """Shoelace area: positive for counterclockwise vertex order."""

    rolled = torch.roll(loop, -1, dims=0)
    return 0.5 * float((loop[:, 0] * rolled[:, 1] - loop[:, 1] * rolled[:, 0]).sum())


def _in_domain(loops: tuple[torch.Tensor, ...], points: torch.Tensor) -> torch.Tensor:
    """Domain membership from CCW loops: inside loop 0, outside every hole."""

    inside = winding_number(loops[0], points) == 1
    for hole in loops[1:]:
        inside = inside & (winding_number(hole, points) == 0)
    return inside


# ---------------------------------------------------------------------------
# Family A: multi-body (outer circle minus one or two disks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiBodyGeometry:
    """Outer circle plus one or two excluded disks (float64 tensors)."""

    outer_center: torch.Tensor  # (2,)
    outer_radius: float
    disk_centers: torch.Tensor  # (n_bodies, 2)
    disk_radii: torch.Tensor  # (n_bodies,)
    gap_ratio: float  # surface gap / (r1 + r2); nan for single body


def sample_multi_body_geometry(
    seed: int,
    *,
    gap_ratio_range: tuple[float, float],
    n_bodies: int,
) -> MultiBodyGeometry:
    """Sample disk radii, the inter-body gap, and the enclosing circle.

    Disk radii are U(0.25, 0.45).  For two bodies the surface gap is
    ``gap_ratio * (r1 + r2)`` along a random direction; the outer radius is
    the configuration extent divided by a fill fraction U(0.55, 0.8), which
    guarantees at least 20% clearance between disks and the outer circle.
    """

    if n_bodies not in (1, 2):
        raise ValueError("n_bodies must be 1 or 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    radii = [_uniform(generator, 0.25, 0.45) for _ in range(n_bodies)]
    axis_angle = _uniform(generator, 0.0, _TWO_PI)
    direction = torch.tensor(
        [math.cos(axis_angle), math.sin(axis_angle)], dtype=torch.float64
    )
    if n_bodies == 2:
        gap_ratio = _uniform(generator, *gap_ratio_range)
        gap = gap_ratio * (radii[0] + radii[1])
        centers = torch.stack(
            (
                -(0.5 * gap + radii[0]) * direction,
                (0.5 * gap + radii[1]) * direction,
            )
        )
        extent = 0.5 * gap + 2.0 * max(radii)
        fill = _uniform(generator, 0.55, 0.8)
    else:
        gap_ratio = math.nan
        offset = _uniform(generator, 0.0, 1.0) * radii[0]
        centers = (offset * direction)[None, :]
        extent = offset + radii[0]
        fill = _uniform(generator, 0.4, 0.7)
    outer_radius = extent / fill
    outer_center = torch.tensor(
        [_uniform(generator, -0.5, 0.5), _uniform(generator, -0.5, 0.5)],
        dtype=torch.float64,
    )
    return MultiBodyGeometry(
        outer_center=outer_center,
        outer_radius=outer_radius,
        disk_centers=centers + outer_center,
        disk_radii=torch.tensor(radii, dtype=torch.float64),
        gap_ratio=gap_ratio,
    )


def _circle_loop(
    center: torch.Tensor, radius: float, n_panels: int, phase: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """CCW vertex loop and true-circle parameter midpoints."""

    angles = phase + _TWO_PI * torch.arange(n_panels, dtype=torch.float64) / n_panels
    vertices = center + radius * torch.stack((angles.cos(), angles.sin()), dim=-1)
    mid = angles + math.pi / n_panels
    midpoints = center + radius * torch.stack((mid.cos(), mid.sin()), dim=-1)
    return vertices, midpoints


def multi_body_boundary_loops(
    geometry: MultiBodyGeometry,
    *,
    n_outer: int,
    n_disk: int,
    phase_seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Return CCW vertex loops and matching midpoints: outer first, then disks."""

    generator = torch.Generator(device="cpu").manual_seed(phase_seed)
    loops, midpoints = [], []
    outer_v, outer_m = _circle_loop(
        geometry.outer_center,
        geometry.outer_radius,
        n_outer,
        _uniform(generator, 0.0, _TWO_PI),
    )
    loops.append(outer_v)
    midpoints.append(outer_m)
    for center, radius in zip(geometry.disk_centers, geometry.disk_radii):
        v, m = _circle_loop(
            center, float(radius), n_disk, _uniform(generator, 0.0, _TWO_PI)
        )
        loops.append(v)
        midpoints.append(m)
    return tuple(loops), tuple(midpoints)


def _sample_multi_body_charges(
    seed: int,
    geometry: MultiBodyGeometry,
    *,
    n_charge_range: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charges inside the excluded disks and outside the outer circle.

    With two bodies, one charge per disk is always biased toward the
    gap-facing surface (radius U(0.55, 0.85) of the disk radius, direction
    toward the other disk with +-0.5 rad jitter) -- the placements that make
    the target field depend on inter-body boundary coupling.  One charge is
    always exterior; the remainder mix all placement roles uniformly.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_bodies = geometry.disk_centers.shape[0]
    count = int(
        torch.randint(
            n_charge_range[0],
            n_charge_range[1] + 1,
            (),
            generator=generator,
            dtype=torch.int64,
        )
    )
    if n_bodies == 2:
        base_roles = ["disk0_gap", "disk1_gap", "exterior"]
        extra_roles = (
            "disk0_gap",
            "disk1_gap",
            "disk0_uniform",
            "disk1_uniform",
            "exterior",
        )
    else:
        base_roles = ["disk0_uniform", "exterior"]
        extra_roles = ("disk0_uniform", "exterior")
    roles = list(base_roles)
    while len(roles) < count:
        pick = int(
            torch.randint(
                0, len(extra_roles), (), generator=generator, dtype=torch.int64
            )
        )
        roles.append(extra_roles[pick])
    roles = roles[:count]

    positions = []
    for role in roles:
        if role == "exterior":
            radius = geometry.outer_radius * _uniform(generator, 1.25, 2.2)
            angle = _uniform(generator, 0.0, _TWO_PI)
            positions.append(
                geometry.outer_center
                + radius
                * torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
            )
            continue
        disk = int(role[4])
        center = geometry.disk_centers[disk]
        disk_radius = float(geometry.disk_radii[disk])
        if role.endswith("_gap"):
            other = geometry.disk_centers[1 - disk]
            toward = other - center
            toward = toward / toward.norm()
            jitter = _uniform(generator, -0.5, 0.5)
            cos_j, sin_j = math.cos(jitter), math.sin(jitter)
            direction = torch.stack(
                (
                    cos_j * toward[0] - sin_j * toward[1],
                    sin_j * toward[0] + cos_j * toward[1],
                )
            )
            rho = disk_radius * _uniform(generator, 0.55, 0.85)
        else:
            angle = _uniform(generator, 0.0, _TWO_PI)
            direction = torch.tensor(
                [math.cos(angle), math.sin(angle)], dtype=torch.float64
            )
            rho = disk_radius * 0.85 * math.sqrt(_uniform(generator, 0.0, 1.0))
        positions.append(center + rho * direction)

    strengths = []
    for _ in roles:
        sign = 1.0 if _uniform(generator, 0.0, 1.0) < 0.5 else -1.0
        strengths.append(sign * _uniform(generator, 0.25, 1.0))
    return torch.stack(positions), torch.tensor(strengths, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Family B: deep cavity (disk with a smoothly filleted U-slot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CavityGeometry:
    """Disk of radius R with a slot of half-width a and depth D along a ray.

    The slot bottom is a semicircular cap of radius ``slot_half_width``
    centered at distance ``radius - slot_depth + slot_half_width`` from the
    center; the mouth corners are tangent-continuous fillets of radius
    ``fillet_radius``.  ``slot_depth == 0`` degenerates to the plain circle
    (convex control).
    """

    center: torch.Tensor  # (2,)
    radius: float
    slot_depth: float  # D, measured from the circle to the slot bottom point
    slot_half_width: float  # a
    fillet_radius: float
    slot_angle: float

    def __post_init__(self) -> None:
        if self.slot_depth > 0.0:
            if self.slot_half_width <= 0.0 or self.fillet_radius <= 0.0:
                raise ValueError("slotted cavities need positive width and fillet")
            if self.radius - self.slot_depth < 0.1 * self.radius:
                raise ValueError("slot may not reach the domain center")
            reach = self.slot_half_width + self.fillet_radius
            if (self.radius - self.fillet_radius) ** 2 <= reach**2:
                raise ValueError("mouth fillet does not fit inside the circle")
            # The straight wall spans [b, x_f]; require it to be non-degenerate
            # so the bottom cap and the mouth fillets never collide.
            x_f = math.sqrt((self.radius - self.fillet_radius) ** 2 - reach**2)
            b = self.radius - self.slot_depth + self.slot_half_width
            if x_f - b < 0.05 * self.slot_half_width:
                raise ValueError("slot too shallow for its width and fillet")


def _arc_points(
    center: torch.Tensor, radius: float, start: float, end: float, n: int
) -> torch.Tensor:
    """Half-open arc sampling [start, end): endpoint owned by the next piece."""

    t = start + (end - start) * torch.arange(n, dtype=torch.float64) / n
    return center + radius * torch.stack((t.cos(), t.sin()), dim=-1)


def _line_points(p0: torch.Tensor, p1: torch.Tensor, n: int) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64)[:, None] / n
    return p0[None, :] + t * (p1 - p0)[None, :]


def _cavity_dense_curve(geometry: CavityGeometry) -> torch.Tensor:
    """Dense CCW polyline of the cavity curve in world coordinates."""

    R = geometry.radius
    if geometry.slot_depth <= 0.0:
        local = _arc_points(torch.zeros(2, dtype=torch.float64), R, 0.0, _TWO_PI, 8192)
    else:
        a = geometry.slot_half_width
        rho = geometry.fillet_radius
        b = R - geometry.slot_depth + a  # bottom-cap center distance
        x_f = math.sqrt((R - rho) ** 2 - (a + rho) ** 2)
        beta = math.atan2(a + rho, x_f)
        origin = torch.zeros(2, dtype=torch.float64)
        upper_fillet_center = torch.tensor([x_f, a + rho], dtype=torch.float64)
        lower_fillet_center = torch.tensor([x_f, -(a + rho)], dtype=torch.float64)
        cap_center = torch.tensor([b, 0.0], dtype=torch.float64)
        pieces = [
            _arc_points(origin, R, beta, _TWO_PI - beta, 4096),
            _arc_points(lower_fillet_center, rho, -beta, 0.5 * math.pi, 512),
            _line_points(
                torch.tensor([x_f, -a], dtype=torch.float64),
                torch.tensor([b, -a], dtype=torch.float64),
                1024,
            ),
            _arc_points(cap_center, a, -0.5 * math.pi, -1.5 * math.pi, 1024),
            _line_points(
                torch.tensor([b, a], dtype=torch.float64),
                torch.tensor([x_f, a], dtype=torch.float64),
                1024,
            ),
            _arc_points(upper_fillet_center, rho, -0.5 * math.pi, beta, 512),
        ]
        local = torch.cat(pieces, dim=0)
    cos_t, sin_t = math.cos(geometry.slot_angle), math.sin(geometry.slot_angle)
    rotation = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
    return geometry.center + local @ rotation.T


def _resample_closed(
    dense: torch.Tensor, n_panels: int, phase: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-arclength panel vertices and midpoints on a dense closed polyline."""

    edge = torch.roll(dense, -1, dims=0) - dense
    lengths = edge.norm(dim=-1)
    cumulative = torch.cat(
        (torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, dim=0))
    )
    perimeter = float(cumulative[-1])

    def interpolate(targets: torch.Tensor) -> torch.Tensor:
        index = torch.searchsorted(cumulative, targets, right=True) - 1
        index = index.clamp(0, dense.shape[0] - 1)
        fraction = (targets - cumulative[index]) / lengths[index].clamp_min(1.0e-30)
        return dense[index] + fraction[:, None] * edge[index]

    steps = torch.arange(n_panels, dtype=torch.float64)
    vertex_t = (phase * perimeter + steps * perimeter / n_panels) % perimeter
    midpoint_t = (phase * perimeter + (steps + 0.5) * perimeter / n_panels) % perimeter
    return interpolate(vertex_t), interpolate(midpoint_t)


def cavity_boundary_loop(
    geometry: CavityGeometry,
    *,
    n_panels: int,
    phase: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CCW panel vertices, curve parameter midpoints, and the dense curve."""

    dense = _cavity_dense_curve(geometry)
    vertices, midpoints = _resample_closed(dense, n_panels, phase)
    return vertices, midpoints, dense


def sample_cavity_geometry(
    seed: int,
    *,
    depth_range: tuple[float, float],
    half_width_range: tuple[float, float],
) -> CavityGeometry:
    """Sample radius, slot direction, and the depth/width difficulty dial.

    ``depth_range`` and ``half_width_range`` are fractions of the disk radius;
    the mouth fillet radius is fixed at 0.6x the slot half-width (comparable
    to the bottom cap, so both are resolvable at the panel scale).
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    radius = _uniform(generator, 0.8, 1.4)
    depth_fraction = _uniform(generator, *depth_range)
    half_width_fraction = _uniform(generator, *half_width_range)
    slot_angle = _uniform(generator, 0.0, _TWO_PI)
    center = torch.tensor(
        [_uniform(generator, -0.5, 0.5), _uniform(generator, -0.5, 0.5)],
        dtype=torch.float64,
    )
    if depth_fraction <= 0.0:
        return CavityGeometry(
            center=center,
            radius=radius,
            slot_depth=0.0,
            slot_half_width=0.0,
            fillet_radius=0.0,
            slot_angle=slot_angle,
        )
    return CavityGeometry(
        center=center,
        radius=radius,
        slot_depth=depth_fraction * radius,
        slot_half_width=half_width_fraction * radius,
        fillet_radius=0.6 * half_width_fraction * radius,
        slot_angle=slot_angle,
    )


def _sample_cavity_charges(
    seed: int,
    geometry: CavityGeometry,
    *,
    n_charge_range: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charges nestled inside the slot plus exterior charges.

    Slot charges live in the strip ``|y| <= 0.5 a`` between the bottom cap
    and the mouth (biased toward the deep end), so their standoff from every
    boundary piece is at least ``0.4 a`` by construction; exterior charges
    sit at radius U(1.3, 2.4) R.  Convex controls place all charges outside.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = int(
        torch.randint(
            n_charge_range[0],
            n_charge_range[1] + 1,
            (),
            generator=generator,
            dtype=torch.int64,
        )
    )
    n_slot = max(2, round(0.4 * count)) if geometry.slot_depth > 0.0 else 0

    cos_t, sin_t = math.cos(geometry.slot_angle), math.sin(geometry.slot_angle)
    rotation = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
    positions = []
    for index in range(count):
        if index < n_slot:
            R = geometry.radius
            a = geometry.slot_half_width
            rho = geometry.fillet_radius
            b = R - geometry.slot_depth + a
            x_f = math.sqrt((R - rho) ** 2 - (a + rho) ** 2)
            x_lo, x_hi = b - 0.3 * a, x_f - 0.2 * a
            depth_bias = _uniform(generator, 0.0, 1.0) ** 2
            x = x_lo + (x_hi - x_lo) * depth_bias
            y = 0.5 * a * _uniform(generator, -1.0, 1.0)
            local = torch.tensor([x, y], dtype=torch.float64)
            positions.append(geometry.center + rotation @ local)
        else:
            radius = geometry.radius * _uniform(generator, 1.3, 2.4)
            angle = _uniform(generator, 0.0, _TWO_PI)
            positions.append(
                geometry.center
                + radius
                * torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
            )
    strengths = []
    for _ in range(count):
        sign = 1.0 if _uniform(generator, 0.0, 1.0) < 0.5 else -1.0
        strengths.append(sign * _uniform(generator, 0.25, 1.0))
    return torch.stack(positions), torch.tensor(strengths, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Shared sample assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderStressSample:
    """One exact Laplace Dirichlet problem plus its certification payload."""

    domain: DomainMesh
    target: torch.Tensor
    family: str
    geometry: object
    charge_positions: torch.Tensor  # (K, 2) float64, cpu
    charge_strengths: torch.Tensor  # (K,) float64, cpu (post-normalization)
    boundary_loops: tuple[torch.Tensor, ...]  # CCW float64 vertex loops
    boundary_midpoints: torch.Tensor  # (n_cells, 2) float64, merged order


def _loop_mesh(
    vertices: torch.Tensor, values: torch.Tensor, *, clockwise_cells: bool
) -> Mesh:
    """Segment-cell mesh over a CCW vertex loop with chosen cell winding.

    Clockwise cells give outward normals under the mesh module's 90-degree-CCW
    edge-normal convention; counterclockwise cells give inward normals (used
    for excluded disks, whose domain-outward direction points *into* the
    disk).  Cell ``k`` always spans vertices ``k`` and ``k + 1`` so the
    supplied midpoint values stay aligned.
    """

    n = vertices.shape[0]
    index = torch.arange(n)
    successor = torch.roll(index, -1)
    if clockwise_cells:
        cells = torch.stack((successor, index), dim=-1)
    else:
        cells = torch.stack((index, successor), dim=-1)
    return Mesh(
        points=vertices,
        cells=cells,
        cell_data={"boundary_value": values},
    )


def _rejection_sample_queries(
    seed: int,
    loops: tuple[torch.Tensor, ...],
    *,
    n_query: int,
    reference_length: float,
    center: torch.Tensor,
    half_extent: float,
    focus_box: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    focus_fraction: float = 0.5,
) -> torch.Tensor:
    """Uniform rejection sampling inside the domain with a boundary margin.

    ``focus_box`` optionally supplies (rotation, low, high) of a local-frame
    box; that fraction of the queries is rejection-sampled from the box
    instead (used to concentrate cavity queries near the slot walls).
    Membership is decided by exact winding numbers against the CCW loops;
    every accepted point keeps ``0.01 * reference_length`` clearance from all
    boundary polygons.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    margin = 0.01 * reference_length

    def accept(candidates: torch.Tensor) -> torch.Tensor:
        keep = _in_domain(loops, candidates)
        for loop in loops:
            keep = keep & (_min_distance_to_loop(candidates, loop) > margin)
        return candidates[keep]

    def collect(n_target: int, propose) -> torch.Tensor:
        found: list[torch.Tensor] = []
        total = 0
        for _ in range(200):
            batch = accept(propose(4 * n_target))
            found.append(batch)
            total += batch.shape[0]
            if total >= n_target:
                break
        else:
            raise RuntimeError("query rejection sampling failed to converge")
        return torch.cat(found)[:n_target]

    def propose_uniform(n: int) -> torch.Tensor:
        offsets = (
            torch.rand(n, 2, dtype=torch.float64, generator=generator) * 2.0 - 1.0
        ) * half_extent
        return center + offsets

    n_focus = 0
    focused = torch.empty(0, 2, dtype=torch.float64)
    if focus_box is not None:
        rotation, low, high = focus_box
        n_focus = int(round(focus_fraction * n_query))

        def propose_focus(n: int) -> torch.Tensor:
            unit = torch.rand(n, 2, dtype=torch.float64, generator=generator)
            local = low[None, :] + unit * (high - low)[None, :]
            return center + local @ rotation.T

        focused = collect(n_focus, propose_focus)
    uniform = collect(n_query - n_focus, propose_uniform)
    return torch.cat((focused, uniform))


_PEAK_CAP = 8.0
_MIN_TARGET_RMS = 0.05


def _normalized_charge_fields(
    seed: int,
    *,
    sample_charges,
    midpoints: torch.Tensor,
    queries: torch.Tensor,
    max_attempts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw charges until the RMS-normalized fields pass the amplitude gates.

    Charges are rescaled so the boundary trace has unit RMS; draws whose peak
    |u| exceeds ``8`` anywhere, or whose interior-target RMS collapses below
    ``0.05``, are rejected and redrawn from a deterministic substream.
    """

    for attempt in range(max_attempts):
        positions, strengths = sample_charges(_substream(seed, 3 + attempt))
        boundary_values = harmonic_potential(midpoints, positions, strengths)
        scale = float(boundary_values.square().mean().sqrt())
        if not math.isfinite(scale) or scale < 1.0e-9:
            continue
        strengths = strengths / scale
        boundary_values = boundary_values / scale
        target = harmonic_potential(queries, positions, strengths)
        peak = max(float(boundary_values.abs().max()), float(target.abs().max()))
        target_rms = float(target.square().mean().sqrt())
        if peak <= _PEAK_CAP and target_rms >= _MIN_TARGET_RMS:
            return positions, strengths, boundary_values, target
    raise RuntimeError(
        f"no admissible charge draw in {max_attempts} attempts for seed {seed}"
    )


def _certify_sample(
    boundary: Mesh,
    loops: tuple[torch.Tensor, ...],
    queries: torch.Tensor,
    charges: torch.Tensor,
    *,
    charge_standoff: float,
) -> None:
    """Winding, standoff, and simplicity certification for one sample."""

    query_winding = mesh_cell_winding(boundary, queries)
    if not bool((query_winding == -1).all()):
        raise RuntimeError("a query point is not interior to the merged boundary")
    charge_winding = mesh_cell_winding(boundary, charges)
    if not bool((charge_winding == 0).all()):
        raise RuntimeError("a charge is not exterior to the domain")
    for loop in loops:
        if not polyline_is_simple(loop):
            raise RuntimeError("a boundary loop self-intersects")
        if _signed_area(loop) <= 0.0:
            raise RuntimeError("a boundary loop is not counterclockwise")
        if float(_min_distance_to_loop(charges, loop).min()) < charge_standoff:
            raise RuntimeError("a charge is too close to a boundary loop")
    for first in range(len(loops)):
        for second in range(first + 1, len(loops)):
            gap = _min_distance_to_loop(loops[first], loops[second])
            if float(gap.min()) <= 0.0:
                raise RuntimeError("boundary loops are not disjoint")


def _finalize_sample(
    *,
    family: str,
    geometry: object,
    parts: list[Mesh],
    loops: tuple[torch.Tensor, ...],
    midpoints: torch.Tensor,
    queries: torch.Tensor,
    target: torch.Tensor,
    positions: torch.Tensor,
    strengths: torch.Tensor,
    reference_length: float,
    charge_standoff: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> EncoderStressSample:
    merged = Mesh.merge(parts) if len(parts) > 1 else parts[0]
    _certify_sample(merged, loops, queries, positions, charge_standoff=charge_standoff)
    boundary = Mesh(
        points=merged.points.to(device=device, dtype=dtype),
        cells=merged.cells.to(device),
        cell_data={
            "boundary_value": merged.cell_data["boundary_value"].to(
                device=device, dtype=dtype
            )
        },
    )
    interior = Mesh(points=queries.to(device=device, dtype=dtype))
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": torch.tensor(
                reference_length, device=device, dtype=dtype
            )
        },
    )
    return EncoderStressSample(
        domain=domain,
        target=target.to(device=device, dtype=dtype),
        family=family,
        geometry=geometry,
        charge_positions=positions,
        charge_strengths=strengths,
        boundary_loops=loops,
        boundary_midpoints=midpoints,
    )


def build_multi_body_sample(
    seed: int,
    *,
    gap_ratio_range: tuple[float, float] = (0.5, 1.5),
    n_charge_range: tuple[int, int] = (4, 8),
    n_bodies: int = 2,
    n_outer: int = 96,
    n_disk: int = 48,
    n_query: int = 256,
    max_attempts: int = 32,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EncoderStressSample:
    """Build one exact multi-body Laplace problem (Family A).

    All geometry and exact values are computed in float64 on the CPU and cast
    at the end; a seed identifies one sample independent of execution device.
    The three (or two) boundaries are merged into a single ``"dirichlet"``
    boundary mesh; body identity is conveyed purely geometrically.
    """

    geometry = sample_multi_body_geometry(
        _substream(seed, 0), gap_ratio_range=gap_ratio_range, n_bodies=n_bodies
    )
    loops, midpoint_parts = multi_body_boundary_loops(
        geometry, n_outer=n_outer, n_disk=n_disk, phase_seed=_substream(seed, 2)
    )
    midpoints = torch.cat(midpoint_parts)
    queries = _rejection_sample_queries(
        _substream(seed, 1),
        loops,
        n_query=n_query,
        reference_length=geometry.outer_radius,
        center=geometry.outer_center,
        half_extent=geometry.outer_radius,
    )
    positions, strengths, boundary_values, target = _normalized_charge_fields(
        seed,
        sample_charges=lambda s: _sample_multi_body_charges(
            s, geometry, n_charge_range=n_charge_range
        ),
        midpoints=midpoints,
        queries=queries,
        max_attempts=max_attempts,
    )
    parts = [_loop_mesh(loops[0], boundary_values[:n_outer], clockwise_cells=True)]
    for index in range(1, len(loops)):
        offset = n_outer + (index - 1) * n_disk
        parts.append(
            _loop_mesh(
                loops[index],
                boundary_values[offset : offset + n_disk],
                clockwise_cells=False,
            )
        )
    charge_standoff = 0.1 * float(geometry.disk_radii.min())
    return _finalize_sample(
        family="multi_body",
        geometry=geometry,
        parts=parts,
        loops=loops,
        midpoints=midpoints,
        queries=queries,
        target=target,
        positions=positions,
        strengths=strengths,
        reference_length=geometry.outer_radius,
        charge_standoff=charge_standoff,
        device=device,
        dtype=dtype,
    )


def build_deep_cavity_sample(
    seed: int,
    *,
    depth_range: tuple[float, float] = (0.35, 0.55),
    half_width_range: tuple[float, float] = (0.10, 0.16),
    n_charge_range: tuple[int, int] = (4, 8),
    n_boundary: int = 224,
    n_query: int = 256,
    max_attempts: int = 32,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EncoderStressSample:
    """Build one exact deep-cavity Laplace problem (Family B).

    Panels are placed at equal arclength on the smooth slot curve (so the
    walls, bottom cap, and mouth fillets are all resolved at the panel
    scale).  Half the queries are drawn from a box around the slot so the
    benchmark probes the cavity interior; convex controls (zero depth) use
    uniform queries only.
    """

    geometry = sample_cavity_geometry(
        _substream(seed, 0),
        depth_range=depth_range,
        half_width_range=half_width_range,
    )
    phase_generator = torch.Generator(device="cpu").manual_seed(_substream(seed, 2))
    phase = _uniform(phase_generator, 0.0, 1.0)
    vertices, midpoints, _ = cavity_boundary_loop(
        geometry, n_panels=n_boundary, phase=phase
    )
    loops = (vertices,)

    focus_box = None
    if geometry.slot_depth > 0.0:
        R, a = geometry.radius, geometry.slot_half_width
        b = R - geometry.slot_depth + a
        cos_t, sin_t = math.cos(geometry.slot_angle), math.sin(geometry.slot_angle)
        rotation = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64)
        low = torch.tensor([b - 3.0 * a, -5.0 * a], dtype=torch.float64)
        high = torch.tensor([min(R, b + geometry.slot_depth), 5.0 * a])
        focus_box = (rotation, low, high.to(torch.float64))
    queries = _rejection_sample_queries(
        _substream(seed, 1),
        loops,
        n_query=n_query,
        reference_length=geometry.radius,
        center=geometry.center,
        half_extent=geometry.radius,
        focus_box=focus_box,
        focus_fraction=0.5,
    )
    positions, strengths, boundary_values, target = _normalized_charge_fields(
        seed,
        sample_charges=lambda s: _sample_cavity_charges(
            s, geometry, n_charge_range=n_charge_range
        ),
        midpoints=midpoints,
        queries=queries,
        max_attempts=max_attempts,
    )
    parts = [_loop_mesh(vertices, boundary_values, clockwise_cells=True)]
    charge_standoff = (
        0.3 * geometry.slot_half_width
        if geometry.slot_depth > 0.0
        else 0.2 * geometry.radius
    )
    return _finalize_sample(
        family="deep_cavity",
        geometry=geometry,
        parts=parts,
        loops=loops,
        midpoints=midpoints,
        queries=queries,
        target=target,
        positions=positions,
        strengths=strengths,
        reference_length=geometry.radius,
        charge_standoff=charge_standoff,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Splits, evaluation, and driver
# ---------------------------------------------------------------------------

SPLITS: dict[str, dict[str, dict]] = {
    "multi_body": {
        "in_distribution": {
            "gap_ratio_range": (0.5, 1.5),
            "n_charge_range": (4, 8),
            "n_bodies": 2,
        },
        "narrow_gap": {
            "gap_ratio_range": (0.05, 0.2),
            "n_charge_range": (4, 8),
            "n_bodies": 2,
        },
        "unseen_charge_count": {
            "gap_ratio_range": (0.5, 1.5),
            "n_charge_range": (12, 16),
            "n_bodies": 2,
        },
        "single_body": {
            "gap_ratio_range": (0.5, 1.5),
            "n_charge_range": (4, 8),
            "n_bodies": 1,
        },
    },
    "deep_cavity": {
        "in_distribution": {
            "depth_range": (0.35, 0.55),
            "half_width_range": (0.10, 0.16),
            "n_charge_range": (4, 8),
        },
        "deep_slot": {
            "depth_range": (0.65, 0.80),
            "half_width_range": (0.055, 0.09),
            "n_charge_range": (4, 8),
        },
        "convex_control": {
            "depth_range": (0.0, 0.0),
            "half_width_range": (0.10, 0.16),
            "n_charge_range": (4, 8),
        },
    },
}

SAMPLE_BUILDERS = {
    "multi_body": build_multi_body_sample,
    "deep_cavity": build_deep_cavity_sample,
}


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _build_model(model_name: str) -> nn.Module:
    """Instantiate one arm of the encoder-coupling comparison.

    ``pair_kernel`` is the load-bearing control: it sees only euclidean pair
    invariants of (query, source), so if the families genuinely require
    all-to-all boundary information flow it must degrade on ``narrow_gap``
    and ``deep_slot`` while remaining competitive on convex/wide-gap tiers.
    """

    if model_name == "boundary_mean":
        return BoundaryMean()
    if model_name == "pair_kernel":
        return InvariantPairKernel()
    if model_name == "mesh_transformer_kernel_singonly":
        return build_mesh_transformer(
            MeshTransformerConfig(),
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=False,
        )
    if model_name in (
        "mesh_transformer_kernel_singpair_enc0",
        "mesh_transformer_kernel_singpair_enc1",
    ):
        # Encoder-depth eval: with 0 operator layers there is NO attention
        # among boundary cells (pure lift + conditioned kernel decode), so
        # this arm measures directly whether all-to-all boundary coupling
        # earns its place on the stress families.
        from dataclasses import replace

        return build_mesh_transformer(
            replace(
                MeshTransformerConfig(),
                operator_layers=int(model_name[-1]),
            ),
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
        )
    if model_name == "mesh_transformer_kernel_singpair":
        # Two exact singular members (double + single layer). Both families
        # are multiply connected (multi_body always; deep_cavity is simply
        # connected, serving as the no-effect control), so this arm tests
        # the single layer's net-flux completeness fix on 2D holes.
        return build_mesh_transformer(
            MeshTransformerConfig(),
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
        )
    if model_name == "mesh_transformer_kernel_nomlp":
        return build_mesh_transformer(
            MeshTransformerConfig(),
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=True,
        )
    raise ValueError(f"unknown model {model_name!r}")


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    *,
    family: str,
    eval_seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Mean relative L2 per split on frozen, deterministic evaluation banks."""

    build = SAMPLE_BUILDERS[family]
    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS[family].items())):
        errors = []
        for case in range(n_cases):
            sample = build(
                eval_seed + 7919 * case + 1_000_003 * split_index,
                device=device,
                dtype=dtype,
                **spec,
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        report[name] = sum(errors) / len(errors)
    return report


def pde_residual(
    model: nn.Module,
    *,
    family: str,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    r"""Return ``||lap u_pred|| * L^2 / ||u_pred||`` via autograd (float64).

    Computed at 32 interior points on two cases of the requested split
    (default in-distribution, the historical convention).  The exact solution
    scores float-noise zero; so does any harmonic prediction
    (``BoundaryMean`` and the exact singular kernel member are harmonic by
    construction), so this diagnoses *harmonicity*, not accuracy.
    """

    build = SAMPLE_BUILDERS[family]
    spec = SPLITS[family][split]
    model.eval()
    residuals = []
    for case in range(2):
        sample = build(
            seed + case, n_query=32, device=device, dtype=torch.float64, **spec
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        u = model_fp64(domain).point_data["potential"]
        laplacian = torch.zeros_like(u)
        if u.grad_fn is not None:
            (gradient,) = torch.autograd.grad(
                u.sum(), points, create_graph=True, allow_unused=True
            )
            if gradient is not None:
                for component in range(2):
                    (second,) = torch.autograd.grad(
                        gradient[:, component].sum(),
                        points,
                        create_graph=True,
                        allow_unused=True,
                    )
                    if second is not None:
                        laplacian = laplacian + second[:, component]
        length = sample.domain.global_data["reference_length"].reshape(())
        residual = laplacian.detach() * length**2
        residuals.append(
            float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


@torch.no_grad()
def max_principle_violation(
    model: nn.Module,
    *,
    family: str,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    """Sampled boundary-range violation of the harmonic maximum principle.

    Uses the shared :func:`metrics.sampled_boundary_range_violation`
    convention: the prediction's excursion beyond the sampled Dirichlet range
    over the merged boundary, normalized by sampled boundary RMS.  Both
    families are interior Dirichlet Laplace problems, so the principle is
    licensed (on the multiply connected multi-body family the bound is the
    range over *all* merged boundaries, which is exactly what the sampled
    trace provides).  This is a discretization-aware proxy -- the exact
    continuous trace range is not enclosed, unlike the conformal bank's
    certified diagnostic -- so conservative sampling can hide a small
    violation but cannot create one on the sampled trace itself.  Two cases
    per split, float64, the split's full query set.
    """

    from metrics import sampled_boundary_range_violation

    build = SAMPLE_BUILDERS[family]
    spec = SPLITS[family][split]
    model_fp64 = model.double()
    model_fp64.eval()
    violations = []
    for case in range(2):
        sample = build(seed + case, device=device, dtype=torch.float64, **spec)
        prediction = model_fp64(sample.domain).point_data["potential"]
        boundary_values = sample.domain.boundaries["dirichlet"].cell_data[
            "boundary_value"
        ]
        violations.append(
            float(sampled_boundary_range_violation(prediction, boundary_values))
        )
    return sum(violations) / len(violations)


def fidelity_metrics(
    model: nn.Module,
    *,
    family: str,
    seed: int,
    device: torch.device,
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split, the strong-form residual under the driver's existing
    convention (:func:`pde_residual`: float64 autograd, two cases, 32
    interior queries) plus the sampled maximum-principle violation
    (:func:`max_principle_violation`) -- both deliberately subsampled so the
    block stays cheap relative to training.
    """

    return {
        "pde_residual": {
            name: pde_residual(
                model, family=family, seed=seed, device=device, split=name
            )
            for name in sorted(SPLITS[family])
        },
        "pde_residual_note": (
            "||lap u|| L^2 / ||u|| via float64 autograd at 32 interior "
            "points on two cases per split (the top-level 'pde_residual' is "
            "the in_distribution entry); harmonicity of the prediction, not "
            "accuracy -- the exact solution and any harmonic model score ~0"
        ),
        "max_principle_violation": {
            name: max_principle_violation(
                model, family=family, seed=seed, device=device, split=name
            )
            for name in sorted(SPLITS[family])
        },
        "max_principle_note": (
            "sampled boundary-range violation normalized by boundary RMS "
            "(metrics.sampled_boundary_range_violation), two cases per "
            "split; a discretization-aware proxy for the harmonic maximum "
            "principle, not the conformal bank's certified continuous "
            "enclosure"
        ),
    }


def run_experiment(
    *,
    model_name: str,
    family: str,
    steps: int,
    seed: int,
    device: str,
    output_dir: str,
    eval_cases: int = 16,
) -> dict:
    """Train one arm on the family's in-distribution split and report."""

    if family not in SPLITS:
        raise ValueError(f"unknown family {family!r}")
    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    build = SAMPLE_BUILDERS[family]
    train_spec = SPLITS[family]["in_distribution"]
    model = _build_model(model_name).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]

    best_state, best_val, history = None, float("inf"), []
    start_time = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-6)
        for step in range(1, steps + 1):
            model.train()
            sample = build(
                seed + 104_729 * step, device=device_t, dtype=dtype, **train_spec
            )
            prediction = model(sample.domain).point_data["potential"]
            loss = torch.sum((prediction - sample.target).square()) / torch.sum(
                sample.target.square()
            ).clamp_min(1.0e-30)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 250 == 0 or step == steps:
                validation = evaluate_splits(
                    model,
                    family=family,
                    eval_seed=71_000_011,
                    n_cases=4,
                    device=device_t,
                    dtype=dtype,
                )["in_distribution"]
                history.append({"step": step, "validation_relative_l2": validation})
                if validation < best_val:
                    best_val = validation
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_val = evaluate_splits(
            model,
            family=family,
            eval_seed=71_000_011,
            n_cases=4,
            device=device_t,
            dtype=dtype,
        )["in_distribution"]

    report = {
        "model": model_name,
        "family": family,
        "equation": "laplace: lap u = 0 (encoder-stress geometries)",
        "seed": seed,
        "steps": steps,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_validation_relative_l2": best_val,
        "splits": evaluate_splits(
            model,
            family=family,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            device=device_t,
            dtype=dtype,
        ),
        "pde_residual": pde_residual(
            model, family=family, seed=83_000_019, device=device_t
        ),
        "pde_residual_scale_note": (
            "||lap u|| L^2 / ||u||: harmonicity of the prediction, not "
            "accuracy; the exact solution and any harmonic model score ~0"
        ),
        "fidelity": fidelity_metrics(
            model, family=family, seed=83_000_019, device=device_t
        ),
        "state": {
            k: v.tolist() for k, v in model.state_dict().items() if v.numel() <= 16
        },
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{family}_{model_name}_seed{seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "boundary_mean",
            "pair_kernel",
            "mesh_transformer_kernel_singonly",
            "mesh_transformer_kernel_singpair",
            "mesh_transformer_kernel_singpair_enc0",
            "mesh_transformer_kernel_singpair_enc1",
            "mesh_transformer_kernel_nomlp",
        ),
    )
    parser.add_argument(
        "--family", required=True, choices=("multi_body", "deep_cavity")
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-cases", type=int, default=16)
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        family=arguments.family,
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
        eval_cases=arguments.eval_cases,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("model", "family", "splits", "pde_residual")}
        )
    )
