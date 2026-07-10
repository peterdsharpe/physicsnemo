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

"""Contracts for the declared boundary-trace mode (``trace_of``).

The mode is the constructive consequence of the GeoTransolver-gap verdict
(book/18-notebook.qmd): on boundary-to-boundary tasks every query sits ON a
source panel, exactly at the double layer's jump discontinuity, and the
closed form serves an accidental signed-zero branch (measured: the interior
limit, while the physical surface data are the exterior trace).  The
declared mode (a) replaces the own-panel double-layer entries with the
exact exterior one-sided limit ``+1/2`` and (b) gives each query typed
read-outs of its OWN cell's post-attention encoded states.  The contracts
pinned here:

- the jump relation itself, numerically, on the pre-registered analytic
  case: constant density on a closed circle (2D) and sphere (3D) -- the
  corrected trace must equal the analytic exterior limit (0) to roundoff,
  and must equal the uncorrected evaluation at ``+epsilon`` outside;
- the single-layer member's VALUE needs no correction: it is continuous
  across the boundary (only its normal derivative jumps) and its on-panel
  closed form matches the analytic on-surface potential;
- the default-off knob is bitwise the historical model (state dict and
  outputs), and misdeclarations fail loudly (unknown boundary, moment
  decoder, query count != cell count, stale trace-free encodings);
- the own-cell read-outs are live, trainable parameters;
- similarity/parity equivariance, drive-linearity (linear mode), and zero
  preservation (nonlinear mode) survive with the mode on;
- query chunking stays a pure memory control (the identity map is
  declared per absolute index, not per chunk), and the mode composes
  bitwise with checkpointed decode chunks;
- trace models survive the Module save / from_checkpoint round trip.
"""

from __future__ import annotations

import math

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    exact_double_layer_member,
    exact_single_layer_member,
    exterior_trace_self_entries,
)
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

_BOUNDARY_RANKS = {
    "vehicle": {"operator": {}, "drive": {}},
    "far": {"operator": {}, "drive": {"forcing": 0}},
}
_GLOBAL_RANKS = {"operator": {}, "drive": {"flow_direction": 1}}
_OUTPUT_RANKS = {"pressure": 0, "wss": 1}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _circle_boundary(
    n_cells: int,
    device: torch.device | str,
    *,
    radius: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> Mesh:
    """Circle boundary with clockwise cells, hence outward normals."""
    angles = 2.0 * torch.pi * torch.arange(n_cells, device=device, dtype=dtype)
    angles = angles / n_cells
    points = radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    indices = torch.arange(n_cells, device=device)
    cells = torch.stack((torch.roll(indices, -1), indices), dim=-1)
    return Mesh(points=points, cells=cells)


def _icosphere(subdivisions: int, dtype: torch.dtype = torch.float64):
    """Unit icosphere with outward-oriented faces (ported from the suite)."""
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


def _patch(
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    n: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.tensor(origin, device=device, dtype=dtype)
    uu = torch.tensor(u, device=device, dtype=dtype)
    vv = torch.tensor(v, device=device, dtype=dtype)
    s = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    points = (
        o[None, None, :]
        + s[:, None, None] * uu[None, None, :]
        + s[None, :, None] * vv[None, None, :]
    ).reshape(-1, 3)
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            cells.append([a, b, a + 1])
            cells.append([b, b + 1, a + 1])
    return points, torch.tensor(cells, device=device, dtype=torch.long)


def _orthogonal(
    device: torch.device | str,
    *,
    reflection: bool,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(219)
    transform, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform.to(device)


# ---------------------------------------------------------------------------
# Model and domain helpers
# ---------------------------------------------------------------------------


def _model(
    device: torch.device | str,
    *,
    trace_of: str | None = "vehicle",
    field_mode: str = "zero_preserving_nonlinear",
    dtype: torch.dtype = torch.float64,
    seed: int = 947,
    **overrides,
) -> MeshTransformer:
    torch.manual_seed(seed)
    kwargs = dict(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        field_mode=field_mode,
        query_decoder="kernel",
        trace_of=trace_of,
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        # Small enough that the 18-query trace boundary decodes in several
        # chunks, exercising the chunk-local declared-identity offsets.
        query_chunk_size=5,
    )
    kwargs.update(overrides)
    model = MeshTransformer(**kwargs).to(device=device, dtype=dtype)
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = dtype
    model.eval()
    return model


def _domain(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
    drive_scale: float = 1.0,
    transform: torch.Tensor | None = None,
    scale: float = 1.0,
    translation: torch.Tensor | None = None,
    reverse_boundary_orientation: bool = False,
    drop_queries: int = 0,
) -> DomainMesh:
    """Two-boundary domain whose interior IS the vehicle's cell centroids.

    ``vehicle`` (the declared trace boundary; 18 cells) sorts AFTER ``far``
    (8 cells) in the model's canonical boundary order, so the declared
    cell range starts at a nonzero merged-source offset -- the alignment
    bookkeeping the mode must get right.
    """
    vehicle_points, vehicle_cells = _patch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.1),
        (0.0, 1.0, -0.1),
        4,
        device=device,
        dtype=dtype,
    )
    far_points, far_cells = _patch(
        (-2.0, -2.0, 1.5),
        (5.0, 0.0, 0.2),
        (0.0, 5.0, 0.3),
        3,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(88)
    forcing = drive_scale * (
        torch.randn(far_cells.shape[0], generator=generator, dtype=dtype).to(device)
    )
    flow_direction = drive_scale * torch.tensor(
        [0.9, 0.3, -0.2], device=device, dtype=dtype
    )

    if transform is not None:
        vehicle_points = torch.einsum("nd,ed->ne", vehicle_points, transform)
        far_points = torch.einsum("nd,ed->ne", far_points, transform)
        flow_direction = torch.einsum("d,ed->e", flow_direction, transform)
    if reverse_boundary_orientation:
        vehicle_cells = vehicle_cells[:, [0, 2, 1]]
        far_cells = far_cells[:, [0, 2, 1]]
    if translation is None:
        translation = vehicle_points.new_zeros(3)
    else:
        translation = translation.to(device=device, dtype=dtype)
    vehicle_points = scale * vehicle_points + translation
    far_points = scale * far_points + translation

    vehicle = Mesh(points=vehicle_points, cells=vehicle_cells)
    interior_points = vehicle.cell_centroids.clone()
    if drop_queries:
        interior_points = interior_points[:-drop_queries]
    return DomainMesh(
        interior=Mesh(points=interior_points),
        boundaries={
            "vehicle": vehicle,
            "far": Mesh(points=far_points, cells=far_cells).with_data(
                cell_data={"forcing": forcing}
            ),
        },
        global_data={"flow_direction": flow_direction},
    )


# ---------------------------------------------------------------------------
# The jump relation, pinned numerically (the pre-registered analytic case).
# ---------------------------------------------------------------------------


def test_exterior_trace_matches_analytic_exterior_limit_2d(device):
    """Constant density on a circle: corrected trace == exterior limit (0).

    Closed outward polygon Gauss bookkeeping: rows sum to exactly -1
    inside, 0 outside, and the smooth-point principal value is -1/2 with
    the own-panel PV exactly zero -- so the exterior-corrected trace rows
    must sum to exactly 0.  The uncorrected on-panel closed form lands on
    an accidental signed-zero +-1/2 branch, never the PV.
    """
    mesh = _circle_boundary(96, device, radius=2.0)
    vertices = mesh.points[mesh.cells]
    normals = mesh.cell_normals
    centroids = mesh.cell_centroids
    indices = torch.arange(mesh.n_cells, device=device)
    # Outward orientation is a precondition of the trace-side convention.
    assert bool(((normals * centroids).sum(-1) > 0).all())

    member = exact_double_layer_member(centroids, vertices, normals)
    # The accidental branch: |own entry| == 1/2 (a one-sided limit, never
    # the principal value 0).
    torch.testing.assert_close(
        member[indices, indices].abs(),
        torch.full((mesh.n_cells,), 0.5, device=device, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-9,
    )

    corrected = exterior_trace_self_entries(member, indices)
    assert torch.equal(
        corrected[indices, indices],
        torch.full((mesh.n_cells,), 0.5, device=device, dtype=torch.float64),
    )
    sums = corrected.sum(dim=1)
    torch.testing.assert_close(sums, torch.zeros_like(sums), rtol=0.0, atol=1.0e-12)

    # The corrected trace is the one-sided limit from the declared side:
    # continuous match to the uncorrected evaluation at +epsilon outside,
    # while -epsilon (interior) sits a full jump away at -1.
    eps = 1.0e-6
    outside = exact_double_layer_member(
        centroids + eps * normals, vertices, normals
    ).sum(dim=1)
    inside = exact_double_layer_member(
        centroids - eps * normals, vertices, normals
    ).sum(dim=1)
    torch.testing.assert_close(sums, outside, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(
        inside, torch.full_like(inside, -1.0), rtol=0.0, atol=1.0e-4
    )


def test_exterior_trace_matches_analytic_exterior_limit_3d(device):
    """Constant density on a sphere: corrected trace == exterior limit (0)."""
    points, cells = _icosphere(2)
    mesh = Mesh(points=points.to(device), cells=cells.to(device))
    vertices = mesh.points[mesh.cells]
    normals = mesh.cell_normals
    centroids = mesh.cell_centroids
    indices = torch.arange(mesh.n_cells, device=device)
    assert bool(((normals * centroids).sum(-1) > 0).all())
    # Sanity: the closed triangulation satisfies the interior Gauss identity.
    origin = torch.zeros(1, 3, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        exact_double_layer_member(origin, vertices, normals).sum(),
        torch.tensor(-1.0, device=device, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )

    member = exact_double_layer_member(centroids, vertices, normals)
    torch.testing.assert_close(
        member[indices, indices].abs(),
        torch.full((mesh.n_cells,), 0.5, device=device, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-9,
    )

    corrected = exterior_trace_self_entries(member, indices)
    assert torch.equal(
        corrected[indices, indices],
        torch.full((mesh.n_cells,), 0.5, device=device, dtype=torch.float64),
    )
    sums = corrected.sum(dim=1)
    torch.testing.assert_close(sums, torch.zeros_like(sums), rtol=0.0, atol=1.0e-12)

    eps = 1.0e-6
    outside = exact_double_layer_member(
        centroids + eps * normals, vertices, normals
    ).sum(dim=1)
    inside = exact_double_layer_member(
        centroids - eps * normals, vertices, normals
    ).sum(dim=1)
    torch.testing.assert_close(sums, outside, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(
        inside, torch.full_like(inside, -1.0), rtol=0.0, atol=1.0e-4
    )


@pytest.mark.parametrize("n_dims", [2, 3])
def test_single_layer_value_is_continuous_and_needs_no_correction(device, n_dims):
    """The single layer's on-panel VALUE is the two-sided limit already.

    The single-layer potential is continuous across its own layer (only
    its normal derivative jumps, by minus the density), so trace mode
    corrects nothing: the exact closed form on the panel must agree with
    both one-sided evaluations at +-epsilon and with the classical
    constant-density on-surface potential (-R ln R on a circle of radius
    R; R on a sphere of radius R), up to panelization error.
    """
    if n_dims == 2:
        radius = 2.0
        mesh = _circle_boundary(512, device, radius=radius)
        analytic = -radius * math.log(radius)
        panelization_atol = 1.0e-2
    else:
        radius = 1.0
        points, cells = _icosphere(2)
        mesh = Mesh(points=points.to(device), cells=cells.to(device))
        analytic = radius
        panelization_atol = 5.0e-2
    vertices = mesh.points[mesh.cells]
    normals = mesh.cell_normals
    centroids = mesh.cell_centroids

    eps = 1.0e-6
    on_surface = exact_single_layer_member(centroids, vertices).sum(dim=1)
    outside = exact_single_layer_member(centroids + eps * normals, vertices).sum(dim=1)
    inside = exact_single_layer_member(centroids - eps * normals, vertices).sum(dim=1)

    assert torch.isfinite(on_surface).all()
    torch.testing.assert_close(on_surface, outside, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(on_surface, inside, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(
        on_surface,
        torch.full_like(on_surface, analytic),
        rtol=0.0,
        atol=panelization_atol,
    )


def test_exterior_trace_self_entries_validation(device):
    """Malformed declarations fail loudly at the member level."""
    member = torch.zeros(3, 5, device=device, dtype=torch.float64)
    good = torch.arange(3, device=device)
    with pytest.raises(ValueError, match="shape \\(Q, S\\)"):
        exterior_trace_self_entries(member[0], good)
    with pytest.raises(ValueError, match="matching the query count"):
        exterior_trace_self_entries(member, good[:2])
    with pytest.raises(ValueError, match="torch.long"):
        exterior_trace_self_entries(member, good.to(torch.int32))


# ---------------------------------------------------------------------------
# Default-off and validation contracts
# ---------------------------------------------------------------------------


def test_trace_off_is_bitwise_default(device):
    """Knob off == knob absent: identical parameters and identical outputs."""
    baseline = _model(device, trace_of=None)
    torch.manual_seed(947)
    implicit = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        field_mode="zero_preserving_nonlinear",
        query_decoder="kernel",
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=5,
    ).to(device=device, dtype=torch.float64)
    for module in implicit.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    implicit.eval()

    baseline_state = baseline.state_dict()
    implicit_state = implicit.state_dict()
    assert set(baseline_state) == set(implicit_state)
    assert not any("trace_" in key for key in baseline_state)
    for key in baseline_state:
        assert torch.equal(baseline_state[key], implicit_state[key]), key

    domain = _domain(device)
    with torch.no_grad():
        out_baseline = baseline(domain)
        out_implicit = implicit(domain)
    for name in _OUTPUT_RANKS:
        assert torch.equal(
            out_baseline.point_data[name], out_implicit.point_data[name]
        ), name


def test_trace_mode_constructor_validation():
    """Misdeclarations are rejected at construction, loudly."""
    with pytest.raises(ValueError, match="declared boundar"):
        _model("cpu", trace_of="wing")
    with pytest.raises(ValueError, match="query_decoder='kernel'"):
        _model("cpu", trace_of="vehicle", query_decoder="moment")
    with pytest.raises(TypeError, match="non-empty string"):
        _model("cpu", trace_of="")


def test_trace_alignment_mismatch_is_rejected(device):
    """Query count != declared cell count is a loud declaration error."""
    model = _model(device)
    with pytest.raises(ValueError, match="cell centroids"):
        model(_domain(device, drop_queries=1))

    # An encoding produced WITHOUT the declared mode carries no trace
    # alignment and must be rejected rather than silently mis-decoded.
    plain = _model(device, trace_of=None)
    encoded = plain.encode(_domain(device))
    with pytest.raises(ValueError, match="no declared trace alignment"):
        model.decode(encoded)


# ---------------------------------------------------------------------------
# The own-cell read-outs are live; contracts survive with the mode on
# ---------------------------------------------------------------------------


def test_trace_read_outs_are_live_and_trainable(device):
    """Both typed read-outs receive finite nonzero gradients."""
    model = _model(device)
    domain = _domain(device)

    output = model(domain)
    for name in _OUTPUT_RANKS:
        assert torch.isfinite(output.point_data[name]).all(), name
    loss = (
        output.point_data["pressure"].square().sum()
        + output.point_data["wss"].square().sum()
    )
    loss.backward()

    for module_name in ("trace_operator_read_out", "trace_drive_read_out"):
        module = getattr(model, module_name)
        assert module is not None
        total = 0.0
        for parameter_name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"{module_name}.{parameter_name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"{module_name}.{parameter_name}"
            )
            total += float(parameter.grad.abs().sum())
        assert total > 0.0, module_name


@pytest.mark.parametrize("reflection", [False, True])
def test_trace_similarity_and_parity_equivariance(device, reflection):
    """Rotation/reflection + scale + translation covariance, mode on."""
    model = _model(device)

    transform = _orthogonal(device, reflection=reflection)
    translation = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    domain = _domain(device)
    moved = _domain(
        device,
        transform=transform,
        scale=1.7,
        translation=translation,
        reverse_boundary_orientation=reflection,
    )

    with torch.no_grad():
        out = model(domain)
        out_moved = model(moved)

    torch.testing.assert_close(
        out_moved.point_data["pressure"],
        out.point_data["pressure"],
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        out_moved.point_data["wss"],
        torch.einsum("nd,ed->ne", out.point_data["wss"], transform),
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_trace_drive_linearity_linear_mode(device):
    """Linear mode stays exactly drive-linear through both read-outs."""
    model = _model(device, field_mode="linear")

    with torch.no_grad():
        out_base = model(_domain(device, drive_scale=1.0))
        out_scaled = model(_domain(device, drive_scale=3.5))
    for name in _OUTPUT_RANKS:
        torch.testing.assert_close(
            out_scaled.point_data[name],
            3.5 * out_base.point_data[name],
            rtol=1.0e-10,
            atol=1.0e-12,
        )


def test_trace_zero_preservation(device):
    """Zero drive produces exactly zero output: the drive read-out is
    bias-free and linear in the (zero-preserving) encoded drive state."""
    model = _model(device, field_mode="zero_preserving_nonlinear")

    with torch.no_grad():
        output = model(_domain(device, drive_scale=0.0))
    for name in _OUTPUT_RANKS:
        values = output.point_data[name]
        torch.testing.assert_close(values, torch.zeros_like(values), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Chunking, checkpoint compose, and the Module round trip
# ---------------------------------------------------------------------------


def test_trace_decode_chunking_is_pure_memory_control(device):
    """The declared identity map is per absolute index, not per chunk.

    At the decoder level the trace-corrected message is BITWISE chunk
    independent -- the correction writes one row-local constant -- pinned
    in 2D at a realistic query count, the same discipline as the suite's
    existing bitwise independence test (the 3D closed forms' einsum
    lowering carries a pre-existing 1-ulp batch-shape drift on CUDA at
    larger shapes, unrelated to this mode).  At the model level (3D,
    nonzero merged-source trace offset) the outputs match to the house
    tight tolerance: the query-side ``nn.Linear`` lifts are GEMMs whose
    reduction order may depend on the batch shape, exactly as in the
    trace-free model.
    """
    # --- Decoder level, 2D singpair (the recipe trace arm's dictionary).
    torch.manual_seed(31)
    disk_model = MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={"disk": {"drive": {"boundary_value": 0}}},
        field_mode="linear",
        query_decoder="kernel",
        trace_of="disk",
        kernel_mlp_members=0,
        kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
    ).to(device=device, dtype=torch.float64)
    for module in disk_model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    disk_model.eval()
    boundary = _circle_boundary(24, device)
    generator = torch.Generator().manual_seed(32)
    values = torch.randn(24, generator=generator, dtype=torch.float64).to(device)
    disk_domain = DomainMesh(
        interior=Mesh(points=boundary.cell_centroids.clone()),
        boundaries={"disk": boundary.with_data(cell_data={"boundary_value": values})},
    )
    with torch.no_grad():
        disk_encoded = disk_model.encode(disk_domain)
        assert disk_encoded.trace_slice == slice(0, 24)
        normalized = (
            disk_domain.interior.points - disk_encoded.center
        ) / disk_encoded.reference_length
        self_indices = torch.arange(24, device=normalized.device)
        full = disk_model.kernel_decoder(
            normalized, disk_encoded.kernel_cache, self_indices
        )
        split = 7
        first = disk_model.kernel_decoder(
            normalized[:split], disk_encoded.kernel_cache, self_indices[:split]
        )
        second = disk_model.kernel_decoder(
            normalized[split:], disk_encoded.kernel_cache, self_indices[split:]
        )
    assert torch.equal(full.scalars, torch.cat((first.scalars, second.scalars), dim=0))
    assert torch.equal(full.vectors, torch.cat((first.vectors, second.vectors), dim=0))

    # --- Model level, 3D: the trace boundary starts at a nonzero offset of
    # the merged source, and chunk size never changes the operator.
    model = _model(device)
    domain = _domain(device)
    encoded = model.encode(domain)
    assert encoded.trace_slice == slice(8, 26)  # "far" (8) precedes "vehicle" (18)
    with torch.no_grad():
        chunked = model.decode(encoded)  # query_chunk_size=5 -> four chunks
        model.query_chunk_size = 1 << 20
        whole = model.decode(encoded)
    for name in _OUTPUT_RANKS:
        torch.testing.assert_close(
            chunked.point_data[name],
            whole.point_data[name],
            rtol=1.0e-9,
            atol=1.0e-11,
        )


def test_trace_composes_with_checkpointed_chunks_bitwise(device):
    """Checkpointed decode chunks recompute the side-corrected members
    exactly: outputs and every gradient are bitwise unchanged."""
    plain = _model(device, kernel_checkpoint_query_chunks=False)
    checkpointed = _model(device, kernel_checkpoint_query_chunks=True)
    checkpointed.load_state_dict(plain.state_dict())

    domain = _domain(device)
    results = {}
    for label, model in (("plain", plain), ("checkpointed", checkpointed)):
        output = model(domain)
        loss = (
            output.point_data["pressure"].square().sum()
            + output.point_data["wss"].square().sum()
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        results[label] = (
            {name: output.point_data[name].detach() for name in _OUTPUT_RANKS},
            {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            },
        )

    plain_out, plain_grads = results["plain"]
    ckpt_out, ckpt_grads = results["checkpointed"]
    for name in _OUTPUT_RANKS:
        assert torch.equal(plain_out[name], ckpt_out[name]), name
    assert set(plain_grads) == set(ckpt_grads)
    for name in plain_grads:
        assert torch.equal(plain_grads[name], ckpt_grads[name]), name


def test_trace_checkpoint_roundtrip(device, tmp_path):
    """Trace models survive the Module save / from_checkpoint round trip."""
    model = _model(device, dtype=torch.float32)
    domain = _domain(device, dtype=torch.float32)
    with torch.no_grad():
        expected = model(domain)

    checkpoint = str(tmp_path / "trace.mdlus")
    model.save(checkpoint)
    restored = MeshTransformer.from_checkpoint(checkpoint).to(device)
    restored.eval()

    assert restored.trace_of == "vehicle"
    assert restored.trace_operator_read_out is not None
    assert restored.trace_drive_read_out is not None
    with torch.no_grad():
        actual = restored(domain)
    for name in _OUTPUT_RANKS:
        assert torch.equal(actual.point_data[name], expected.point_data[name]), name
