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

"""Contracts for the local-corrector probe block (task #53;
``kernel_local_pair_features``) and the dimension-generic subtended angle.

Three probe modes share one appended five-channel block on the
smooth-member MLP's per-pair features: ``"windowed"`` (probe A,
kernel-side entry of the P2 local scalars under the smooth full-support
window), ``"near_only"`` (probe B, compact near-field support via a C^1
smoothstep in the subtended angle), and ``"global_control"`` (probe C,
matched parameter count, per-sample pooled content only).  Pinned here:

- the subtended angle is dimensionless, DIMENSION-GENERIC
  (:math:`h=\\mu^{1/m}`, never an area over a distance), and invariant
  under uniform coordinate scaling, in both 2D and 3D;
- knob-off is bitwise the pre-knob model (state dict and outputs);
- all three modes construct, run forward/backward finitely, with
  gradients reaching the widened member MLP, at IDENTICAL parameter
  counts;
- loud failures: no ``trace_of``, moment decoder, member-less decoder;
- ``near_only`` support is exactly compact (all-far means an all-zero
  block) and ``global_control`` carries exactly the pooled scalars;
- similarity + parity equivariance survive with each local mode on.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    PairInvariantFeatures,
    subtended_angle,
)
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

_BOUNDARY_RANKS = {
    "vehicle": {"operator": {}, "drive": {}},
    "far": {"operator": {}, "drive": {"forcing": 0}},
}
_GLOBAL_RANKS = {"operator": {}, "drive": {"flow_direction": 1}}
_OUTPUT_RANKS = {"pressure": 0, "wss": 1}

_MODES = ("windowed", "near_only", "global_control")


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
    generator = torch.Generator(device="cpu").manual_seed(431)
    transform, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform.to(device)


def _model(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
    seed: int = 1201,
    **overrides,
) -> MeshTransformer:
    torch.manual_seed(seed)
    kwargs = dict(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        field_mode="zero_preserving_nonlinear",
        query_decoder="kernel",
        trace_of="vehicle",
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        # Several decode chunks over the 18-cell trace boundary, so the
        # chunk-local own-cell gather is exercised across chunk seams.
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
    transform: torch.Tensor | None = None,
    scale: float = 1.0,
    translation: torch.Tensor | None = None,
    reverse_boundary_orientation: bool = False,
) -> DomainMesh:
    """Two-boundary domain whose interior IS the vehicle's cell centroids."""
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
    forcing = torch.randn(far_cells.shape[0], generator=generator, dtype=dtype).to(
        device
    )
    flow_direction = torch.tensor([0.9, 0.3, -0.2], device=device, dtype=dtype)

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
    return DomainMesh(
        interior=Mesh(points=vehicle.cell_centroids.clone()),
        boundaries={
            "vehicle": vehicle,
            "far": Mesh(points=far_points, cells=far_cells).with_data(
                cell_data={"forcing": forcing}
            ),
        },
        global_data={"flow_direction": flow_direction},
    )


# ---------------------------------------------------------------------------
# The subtended angle: dimensionless, dimension-generic, scale-invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_manifold_dims", [1, 2])
def test_subtended_angle_scale_invariant_and_dimension_generic(
    device, n_manifold_dims
):
    """theta is invariant under uniform coordinate scaling in 2D and 3D.

    Scaling coordinates by ``s`` scales squared distances by ``s**2`` and
    cell measures by ``s**m`` (length of a segment, area of a triangle);
    ``h = measure**(1/m)`` makes both numerator and denominator lengths,
    so theta is a pure, dimensionless ratio -- never an area divided by a
    distance.
    """
    generator = torch.Generator(device="cpu").manual_seed(11)
    squared = (
        torch.rand(7, 9, generator=generator, dtype=torch.float64) + 0.05
    ).to(device)
    measures = (
        torch.rand(9, generator=generator, dtype=torch.float64) + 0.05
    ).to(device)
    theta = subtended_angle(squared, measures, n_manifold_dims)
    for s in (0.037, 1.0, 260.0):
        scaled = subtended_angle(
            squared * s**2, measures * s**n_manifold_dims, n_manifold_dims
        )
        torch.testing.assert_close(scaled, theta, rtol=1.0e-12, atol=1.0e-14)

    # Dimensional genericity, checked against the hand-built ratio: a
    # single triangle of area A seen from distance d subtends
    # sqrt(A) / d, and a segment of length L subtends L / d.
    measure = torch.tensor([0.49], dtype=torch.float64, device=device)
    squared_distance = torch.tensor([[4.0]], dtype=torch.float64, device=device)
    expected = (0.49 ** (1.0 / n_manifold_dims)) / 2.0
    value = subtended_angle(squared_distance, measure, n_manifold_dims)
    torch.testing.assert_close(
        value,
        torch.full_like(value, expected),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_subtended_angle_rejects_bad_shapes(device):
    squared = torch.rand(3, 4, device=device)
    with pytest.raises(ValueError, match="cell_measures"):
        subtended_angle(squared, torch.rand(5, device=device), 2)
    with pytest.raises(ValueError, match="n_manifold_dims"):
        subtended_angle(squared, torch.rand(4, device=device), 0)


# ---------------------------------------------------------------------------
# Default-off identity and loud failures.
# ---------------------------------------------------------------------------


def test_knob_off_is_bitwise_default(device):
    baseline = _model(device)
    explicit_off = _model(device, kernel_local_pair_features=None)

    base_state = baseline.state_dict()
    off_state = explicit_off.state_dict()
    assert base_state.keys() == off_state.keys()
    for key in base_state:
        torch.testing.assert_close(
            off_state[key], base_state[key], rtol=0.0, atol=0.0
        )

    domain = _domain(device)
    with torch.no_grad():
        out = baseline(domain)
        out_off = explicit_off(domain)
    torch.testing.assert_close(
        out_off.point_data["pressure"],
        out.point_data["pressure"],
        rtol=0.0,
        atol=0.0,
    )


def test_knob_failure_modes(device):
    with pytest.raises(ValueError, match="requires trace_of"):
        _model(device, kernel_local_pair_features="windowed", trace_of=None)
    with pytest.raises(ValueError, match="query_decoder"):
        _model(
            device,
            kernel_local_pair_features="windowed",
            query_decoder="moment",
        )
    with pytest.raises(ValueError, match="mlp_members > 0"):
        _model(
            device,
            kernel_local_pair_features="windowed",
            kernel_mlp_members=0,
        )
    with pytest.raises(ValueError, match="local_pair_features must be one of"):
        _model(device, kernel_local_pair_features="banana")


# ---------------------------------------------------------------------------
# All three modes: construct, run, matched parameter counts, live grads.
# ---------------------------------------------------------------------------


def test_probe_modes_run_and_match_parameter_counts(device):
    counts = {}
    for mode in _MODES:
        model = _model(device, kernel_local_pair_features=mode)
        counts[mode] = sum(p.numel() for p in model.parameters())
        domain = _domain(device)
        out = model(domain)
        pressure = out.point_data["pressure"]
        assert torch.isfinite(pressure).all()
        loss = pressure.square().sum() + out.point_data["wss"].square().sum()
        loss.backward()
        first_linear = model.kernel_decoder.member_mlp[0]
        assert first_linear.weight.grad is not None
        assert torch.isfinite(first_linear.weight.grad).all()
        # The widened input columns (the probe block) receive gradient in
        # the local modes: locality is live, not decorative.
        probe_columns = first_linear.weight.grad[:, -5:]
        if mode != "global_control":
            assert probe_columns.abs().sum() > 0.0
    assert counts["windowed"] == counts["near_only"] == counts["global_control"]

    baseline = _model(device)
    assert counts["windowed"] > sum(p.numel() for p in baseline.parameters())


# ---------------------------------------------------------------------------
# Window semantics: compact support and pooled control, white-box.
# ---------------------------------------------------------------------------


def _white_box_block(model, domain):
    encoded = model.encode(domain)
    cache = encoded.kernel_cache
    n_trace = encoded.trace_slice.stop - encoded.trace_slice.start
    queries = (
        encoded.query_mesh.points - encoded.center
    ) / encoded.reference_length
    features = PairInvariantFeatures.compute(
        queries,
        cache.centroids,
        cache.normals,
        cache.pair_vectors,
    )
    self_indices = torch.arange(
        encoded.trace_slice.start,
        encoded.trace_slice.stop,
        device=queries.device,
    )
    assert self_indices.shape[0] == n_trace
    block = model.kernel_decoder._local_pair_feature_block(
        features, cache, self_indices
    )
    return block, encoded.trace_slice.start


def test_near_only_support_is_exactly_compact(device):
    """With a huge theta_c, only SELF-pairs survive the near_only window.

    A trace query sits on its own panel, so its self-pair subtends
    theta -> infinity and is near at any threshold -- physically correct
    (the near set always contains the singular pair).  Every off-diagonal
    pair must be EXACTLY zero: compact support, not decay.
    """
    model = _model(
        device,
        kernel_local_pair_features="near_only",
        kernel_near_theta=1.0e6,
    )
    with torch.no_grad():
        block, trace_start = _white_box_block(model, _domain(device))
    assert block.shape[-1] == 5
    n_queries = block.shape[0]
    self_mask = torch.zeros(
        block.shape[:2], dtype=torch.bool, device=block.device
    )
    rows = torch.arange(n_queries)
    self_mask[rows, trace_start + rows] = True
    off_diagonal = block[~self_mask]
    assert off_diagonal.abs().max().item() == 0.0
    # The self-pair window saturates at exactly one.
    torch.testing.assert_close(
        block[self_mask][..., 0],
        torch.ones_like(block[self_mask][..., 0]),
        rtol=0.0,
        atol=0.0,
    )

    # And with a tiny theta_c every pair saturates the window at one.
    saturated = _model(
        device,
        kernel_local_pair_features="near_only",
        kernel_near_theta=1.0e-9,
        seed=1201,
    )
    with torch.no_grad():
        block_sat, _ = _white_box_block(saturated, _domain(device))
    torch.testing.assert_close(
        block_sat[..., 0],
        torch.ones_like(block_sat[..., 0]),
        rtol=0.0,
        atol=0.0,
    )


def test_global_control_carries_exactly_the_pooled_scalars(device):
    model = _model(device, kernel_local_pair_features="global_control")
    domain = _domain(device)
    with torch.no_grad():
        block, _ = _white_box_block(model, domain)
        encoded = model.encode(domain)
        cache = encoded.kernel_cache
        measure = cache.weights
        pooled = (measure[:, None] * cache.local_scalars).sum(
            dim=0
        ) / measure.sum()
    torch.testing.assert_close(
        block[..., 0], pooled[0].expand_as(block[..., 0]), rtol=1.0e-12, atol=0.0
    )
    torch.testing.assert_close(
        block[..., 1], pooled[1].expand_as(block[..., 1]), rtol=1.0e-12, atol=0.0
    )
    assert block[..., 2:].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Similarity + parity equivariance with the local modes on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["windowed", "near_only"])
@pytest.mark.parametrize("reflection", [False, True])
def test_similarity_and_parity_equivariance_with_probe_on(
    device, mode, reflection
):
    model = _model(device, kernel_local_pair_features=mode)

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
