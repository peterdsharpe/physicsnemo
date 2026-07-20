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

"""Contracts for the CONTRACT-BREAKING DIAGNOSTIC
``diagnostic_local_query_features`` (probe P2 of the H-C decomposition;
book/18-notebook.qmd @sec-nb-aga-fleet).

The knob hands each declared trace query its own cell's local geometry
(unit normal through the existing typed normal channel; log relative area
and nondimensional curvature as two appended scalar channels that are zero
on source rows).  It deliberately breaks the boundary-integral INFORMATION
diet while preserving the similarity/parity contracts.  Pinned here:

- default-off is bitwise the pre-knob model (state dict and outputs);
- the knob requires ``trace_of`` and fails loudly without it, and a
  knob-on model rejects stale encodings that carry no features;
- the features are LIVE: zeroing them changes the output, and gradients
  reach the lift;
- similarity + parity equivariance survive with the knob on (the features
  are typed/invariant by construction: unit normal in the source-side
  channel, log area ratio, curvature scaled by
  ``reference_length**n_manifold_dims``).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

_BOUNDARY_RANKS = {
    "vehicle": {"operator": {}, "drive": {}},
    "far": {"operator": {}, "drive": {"forcing": 0}},
}
_GLOBAL_RANKS = {"operator": {}, "drive": {"flow_direction": 1}}
_OUTPUT_RANKS = {"pressure": 0, "wss": 1}


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


def _model(
    device: torch.device | str,
    *,
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
        # chunk-local feature slicing is exercised.
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
# Default-off identity and loud failures
# ---------------------------------------------------------------------------


def test_knob_off_is_bitwise_default(device):
    """Knob absent and knob=False build and evaluate bitwise identically."""
    baseline = _model(device)
    explicit_off = _model(device, diagnostic_local_query_features=False)

    base_state = baseline.state_dict()
    off_state = explicit_off.state_dict()
    assert base_state.keys() == off_state.keys()
    for key in base_state:
        torch.testing.assert_close(off_state[key], base_state[key], rtol=0.0, atol=0.0)

    domain = _domain(device)
    with torch.no_grad():
        out_base = baseline(domain)
        out_off = explicit_off(domain)
    for name in _OUTPUT_RANKS:
        torch.testing.assert_close(
            out_off.point_data[name],
            out_base.point_data[name],
            rtol=0.0,
            atol=0.0,
        )


def test_knob_requires_trace_of():
    with pytest.raises(ValueError, match="requires trace_of"):
        _model("cpu", trace_of=None, diagnostic_local_query_features=True)


def test_knob_on_rejects_stale_featureless_encoding(device):
    """A knob-on decode must reject an encoding stripped of its features."""
    model = _model(device, diagnostic_local_query_features=True)
    domain = _domain(device)
    with torch.no_grad():
        encoded = model.encode(domain)
    assert encoded.diagnostic_query_features is not None
    stale = dataclasses.replace(encoded, diagnostic_query_features=None)
    with pytest.raises(ValueError, match="carries no diagnostic features"):
        with torch.no_grad():
            model.decode(stale)


# ---------------------------------------------------------------------------
# The features are live
# ---------------------------------------------------------------------------


def test_features_change_output_and_gradients_flow(device):
    """Zeroed features change the output; backward is finite and reaches
    the lift."""
    model = _model(device, diagnostic_local_query_features=True)
    domain = _domain(device)

    encoded = model.encode(domain)
    scalars, normals = encoded.diagnostic_query_features
    assert scalars.shape == (18, 2)
    assert normals.shape == (18, 3)
    assert torch.isfinite(scalars).all()
    # The squashed channels are strictly bounded: unbounded curvature
    # (12 decades on a real vehicle) measurably NaN'd training.
    assert float(scalars.abs().max()) < 1.0
    torch.testing.assert_close(
        normals.norm(dim=-1),
        torch.ones_like(normals[:, 0]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    zeroed = dataclasses.replace(
        encoded,
        diagnostic_query_features=(
            torch.zeros_like(scalars),
            torch.zeros_like(normals),
        ),
    )
    with torch.no_grad():
        out_real = model.decode(encoded)
        out_zeroed = model.decode(zeroed)
    deltas = [
        (out_real.point_data[name] - out_zeroed.point_data[name]).abs().max()
        for name in _OUTPUT_RANKS
    ]
    assert max(float(d) for d in deltas) > 0.0

    output = model(domain)
    loss = sum(output.point_data[name].square().sum() for name in _OUTPUT_RANKS)
    loss.backward()
    total = 0.0
    for name, parameter in model.operator_lift.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        total += float(parameter.grad.abs().sum())
    assert total > 0.0


# ---------------------------------------------------------------------------
# Equivariance survives (the knob breaks the information diet, not the
# transformation contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reflection", [False, True])
def test_similarity_and_parity_equivariance_with_knob_on(device, reflection):
    model = _model(device, diagnostic_local_query_features=True)

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
