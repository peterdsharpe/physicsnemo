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

"""Contracts for the per-boundary moment pool (``per_boundary_moment_pool``).

The pool computes every encoder attention moment per declared boundary and
combines the per-boundary moments through learned dimensionless
per-boundary/per-head log-gains (Green's identity splits over boundary
components, so the decomposition recombined by pure numbers is exact
structure).  The contracts pinned here:

- the default-off knob is bitwise the historical model (state dict and
  outputs);
- at zero gains the pooled moments reproduce the plain quadrature sum
  (up to floating-point summation order);
- the gains are live, trainable parameters;
- similarity covariance, parity typing, drive-linearity (linear mode), and
  zero preservation (nonlinear mode) survive arbitrary gains -- the gain is
  a pure number per (boundary, head);
- pooled models checkpoint-compose through the Module save/from_checkpoint
  round trip;
- malformed segment configurations fail loudly at the attention and block
  levels.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import (
    MeshAttention,
    ScalarVectorState,
)
from physicsnemo.experimental.nn.mesh_attention.block import LinearMeshFieldBlock
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

_BOUNDARY_RANKS = {
    "wall": {"operator": {}, "drive": {"forcing": 0}},
    "far": {"operator": {}, "drive": {}},
}
_GLOBAL_RANKS = {"operator": {}, "drive": {"flow_direction": 1}}
_OUTPUT_RANKS = {"pressure": 0, "velocity": 1}


def _model(
    device: torch.device | str,
    *,
    per_boundary_moment_pool: bool = True,
    per_boundary_moment_pool_balanced: bool = False,
    field_mode: str = "linear",
    query_decoder: str = "moment",
    dtype: torch.dtype = torch.float64,
) -> MeshTransformer:
    torch.manual_seed(732)
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key=None,
        field_mode=field_mode,
        query_decoder=query_decoder,
        per_boundary_moment_pool=per_boundary_moment_pool,
        per_boundary_moment_pool_balanced=per_boundary_moment_pool_balanced,
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=3,
    ).to(device=device, dtype=dtype)
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = dtype
    with torch.no_grad():
        ### Skip the moment-pool gains: they carry the exact init contract
        ### (zero == plain quadrature sum), and skipping them keeps the RNG
        ### stream aligned between pooled and unpooled constructions so the
        ### shared weights match parameter-for-parameter.
        for name, parameter in model.named_parameters():
            if parameter.numel() and not name.endswith("moment_segment_log_gain"):
                parameter.uniform_(-0.2, 0.2)
    model.eval()
    return model


def _set_gains(model: MeshTransformer, values: list[float]) -> None:
    """Deterministically perturb every block's per-boundary log-gains."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("moment_segment_log_gain"):
                pattern = torch.tensor(
                    values, device=parameter.device, dtype=parameter.dtype
                )
                parameter.copy_(pattern[:, None].expand_as(parameter))


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


def _domain(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
    drive_scale: float = 1.0,
    transform: torch.Tensor | None = None,
    scale: float = 1.0,
    translation: torch.Tensor | None = None,
    reverse_boundary_orientation: bool = False,
) -> DomainMesh:
    """Two-boundary domain with a measure-dominant far boundary.

    ``wall`` is a unit patch carrying the scalar drive; ``far`` is a 5x
    scaled patch (25x the measure) with geometry only -- a miniature of the
    tunnel-vs-vehicle measure imbalance the pool exists to address.
    """
    wall_points, wall_cells = _patch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.1),
        (0.0, 1.0, -0.1),
        3,
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
    query_points = torch.tensor(
        [
            [0.18, 0.21, 0.17],
            [0.42, 0.11, 0.23],
            [0.08, 0.56, 0.19],
            [0.25, 0.22, 0.51],
            [0.91, 0.77, 0.62],
        ],
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(88)
    forcing = drive_scale * (
        torch.randn(wall_cells.shape[0], generator=generator, dtype=dtype).to(device)
    )
    flow_direction = drive_scale * torch.tensor(
        [0.9, 0.3, -0.2], device=device, dtype=dtype
    )

    if transform is not None:
        wall_points = torch.einsum("nd,ed->ne", wall_points, transform)
        far_points = torch.einsum("nd,ed->ne", far_points, transform)
        query_points = torch.einsum("nd,ed->ne", query_points, transform)
        flow_direction = torch.einsum("d,ed->e", flow_direction, transform)
    if reverse_boundary_orientation:
        wall_cells = wall_cells[:, [0, 2, 1]]
        far_cells = far_cells[:, [0, 2, 1]]
    if translation is None:
        translation = wall_points.new_zeros(3)
    else:
        translation = translation.to(device=device, dtype=dtype)
    wall_points = scale * wall_points + translation
    far_points = scale * far_points + translation
    query_points = scale * query_points + translation

    return DomainMesh(
        interior=Mesh(points=query_points),
        boundaries={
            "wall": Mesh(
                points=wall_points, cells=wall_cells, cell_data={"forcing": forcing}
            ),
            "far": Mesh(points=far_points, cells=far_cells),
        },
        global_data={"flow_direction": flow_direction},
    )


### ---------------------------------------------------------------------------
### Default-off and init contracts
### ---------------------------------------------------------------------------


def test_pool_off_is_bitwise_default(device):
    """Knob off == knob absent: identical parameters and identical outputs."""
    baseline = _model(device, per_boundary_moment_pool=False)
    torch.manual_seed(732)
    implicit = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key=None,
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=3,
    ).to(device=device, dtype=torch.float64)
    for module in implicit.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    with torch.no_grad():
        for parameter in implicit.parameters():
            if parameter.numel():
                parameter.uniform_(-0.2, 0.2)
    implicit.eval()

    baseline_state = baseline.state_dict()
    implicit_state = implicit.state_dict()
    assert set(baseline_state) == set(implicit_state)
    assert not any("moment_segment_log_gain" in key for key in baseline_state)
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


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_pool_init_reproduces_unpooled(device, query_decoder):
    """Zero gains reproduce the plain quadrature sum (up to summation order)."""
    pooled = _model(device, per_boundary_moment_pool=True, query_decoder=query_decoder)
    plain = _model(device, per_boundary_moment_pool=False, query_decoder=query_decoder)
    ### Identical seeded construction: shared weights must agree exactly
    ### (the gain parameters are zero-initialized and consume no RNG).
    plain_state = plain.state_dict()
    for key, value in pooled.state_dict().items():
        if "moment_segment_log_gain" in key:
            assert torch.equal(value, torch.zeros_like(value)), key
        else:
            assert torch.equal(value, plain_state[key]), key

    domain = _domain(device)
    with torch.no_grad():
        out_pooled = pooled(domain)
        out_plain = plain(domain)
    for name in _OUTPUT_RANKS:
        torch.testing.assert_close(
            out_pooled.point_data[name],
            out_plain.point_data[name],
            rtol=1.0e-9,
            atol=1.0e-11,
        )


def test_pool_gains_are_live_and_trainable(device):
    """Gains receive finite nonzero gradients and change the output."""
    model = _model(device, per_boundary_moment_pool=True)
    domain = _domain(device)

    output = model(domain)
    loss = (
        output.point_data["pressure"].square().sum()
        + output.point_data["velocity"].square().sum()
    )
    loss.backward()
    gain_names = [
        name
        for name, parameter in model.named_parameters()
        if name.endswith("moment_segment_log_gain")
    ]
    ### One gain per operator / drive / query block (2 query layers).
    assert len(gain_names) == 4
    for name, parameter in model.named_parameters():
        if name.endswith("moment_segment_log_gain"):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert float(parameter.grad.abs().sum()) > 0.0, name

    with torch.no_grad():
        base = model(domain)
    _set_gains(model, [0.0, -8.0])  # suppress the measure-dominant far boundary
    with torch.no_grad():
        suppressed = model(domain)
    assert not torch.allclose(
        base.point_data["pressure"], suppressed.point_data["pressure"]
    )


### ---------------------------------------------------------------------------
### Invariance contracts at arbitrary gains
### ---------------------------------------------------------------------------


@pytest.mark.parametrize("reflection", [False, True])
def test_pool_similarity_and_parity_equivariance(device, reflection):
    """Rotation/reflection + scale + translation covariance with live gains."""
    model = _model(device, per_boundary_moment_pool=True)
    _set_gains(model, [0.7, -1.3])

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
        out_moved.point_data["velocity"],
        torch.einsum("nd,ed->ne", out.point_data["velocity"], transform),
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_pool_drive_linearity_linear_mode(device):
    """Linear mode stays exactly drive-linear: the gain is drive-independent."""
    model = _model(device, per_boundary_moment_pool=True, field_mode="linear")
    _set_gains(model, [0.5, -2.5])

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


def test_pool_zero_preservation(device):
    """Zero drive produces exactly zero output through gained moments."""
    model = _model(
        device,
        per_boundary_moment_pool=True,
        field_mode="zero_preserving_nonlinear",
    )
    _set_gains(model, [1.1, -0.9])

    with torch.no_grad():
        output = model(_domain(device, drive_scale=0.0))
    for name in _OUTPUT_RANKS:
        values = output.point_data[name]
        torch.testing.assert_close(values, torch.zeros_like(values), rtol=0.0, atol=0.0)


### ---------------------------------------------------------------------------
### Checkpoint compose
### ---------------------------------------------------------------------------


def test_pool_checkpoint_roundtrip(device, tmp_path):
    """Pooled models survive the Module save / from_checkpoint round trip."""
    model = _model(device, per_boundary_moment_pool=True, dtype=torch.float32)
    _set_gains(model, [0.3, -4.0])
    domain = _domain(device, dtype=torch.float32)
    with torch.no_grad():
        expected = model(domain)

    checkpoint = str(tmp_path / "pooled.mdlus")
    model.save(checkpoint)
    restored = MeshTransformer.from_checkpoint(checkpoint).to(device)
    restored.eval()

    assert restored.per_boundary_moment_pool is True
    gains = dict(restored.named_parameters())
    key = "drive_blocks.0.moment_segment_log_gain"
    assert key in gains and gains[key].shape == (2, 2)
    with torch.no_grad():
        actual = restored(domain)
    for name in _OUTPUT_RANKS:
        assert torch.equal(actual.point_data[name], expected.point_data[name]), name


### ---------------------------------------------------------------------------
### Validation contracts
### ---------------------------------------------------------------------------


def test_segment_validation_errors(device):
    """Malformed segment configurations fail loudly, not silently."""
    torch.manual_seed(11)
    attention = MeshAttention(
        query_scalar_dim=2,
        query_vector_dim=1,
        key_scalar_dim=2,
        key_vector_dim=1,
        value_scalar_dim=2,
        value_vector_dim=1,
        out_scalar_dim=2,
        out_vector_dim=1,
        heads=2,
        scalar_rank=2,
        vector_rank=1,
        scalar_value_dim=2,
        vector_value_dim=1,
    ).to(device)
    points, cells = _patch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        3,
        device=device,
        dtype=torch.float32,
    )
    mesh = Mesh(points=points, cells=cells)
    n = mesh.n_cells
    state = ScalarVectorState(
        torch.randn(n, 2, device=device), torch.randn(n, 1, 3, device=device)
    )
    good_segments = (slice(0, 3), slice(3, n))
    good_gain = torch.zeros(2, 2, device=device)

    ### Well-formed segments work.
    attention.build_moments(mesh, state, state, good_segments, good_gain)

    with pytest.raises(ValueError, match="provided together"):
        attention.build_moments(mesh, state, state, segments=good_segments)
    with pytest.raises(ValueError, match="shape"):
        attention.build_moments(
            mesh, state, state, good_segments, torch.zeros(3, 2, device=device)
        )
    with pytest.raises(ValueError, match="does not begin at"):
        attention.build_moments(
            mesh, state, state, (slice(0, 3), slice(4, n)), good_gain
        )
    with pytest.raises(ValueError, match="cover every source cell"):
        attention.build_moments(
            mesh, state, state, (slice(0, 3), slice(3, n - 1)), good_gain
        )

    ### Blocks constructed without segments reject segmented calls.
    block = LinearMeshFieldBlock(2, 1, 2, 1, heads=2, scalar_rank=2, vector_rank=1).to(
        device
    )
    field = ScalarVectorState(
        torch.randn(n, 2, device=device), torch.randn(n, 1, 3, device=device)
    )
    with pytest.raises(ValueError, match="n_moment_segments=0"):
        block.build_source_moments(mesh, state, field, moment_segments=good_segments)


def test_balanced_pool_requires_pool(device):
    """The balanced flag without the per-boundary pool is rejected."""
    with pytest.raises(ValueError, match="requires"):
        _model(
            device,
            per_boundary_moment_pool=False,
            per_boundary_moment_pool_balanced=True,
        )


def test_balanced_pool_zero_gain_reweights_and_reparameterizes(device):
    """The balanced pool (external-review arm) at zero gains weights each
    boundary equally (differing from the plain sum when boundary measures
    differ), and setting the gains to ln(A_s) - ln(mean A) cancels the
    offset exactly -- the balanced arm is a reparameterized initialization
    of the same hypothesis class, not a different operator."""
    balanced = _model(device, per_boundary_moment_pool_balanced=True)
    plain = _model(device, per_boundary_moment_pool=False)
    domain = _domain(device)

    measures = torch.stack(
        [
            domain.boundaries[name].cell_areas.sum()
            for name in balanced.boundary_names
        ]
    )
    assert not torch.allclose(measures, measures.mean().expand_as(measures)), (
        "fixture boundaries must have unequal measures for this test to "
        "discriminate"
    )

    with torch.no_grad():
        out_balanced = balanced(domain)
        out_plain = plain(domain)
    assert any(
        not torch.allclose(
            out_balanced.point_data[name], out_plain.point_data[name]
        )
        for name in _OUTPUT_RANKS
    ), "balanced pool at zero gains must differ from the plain sum"

    _set_gains(balanced, (measures.log() - measures.mean().log()).tolist())
    with torch.no_grad():
        out_cancelled = balanced(domain)
    for name in _OUTPUT_RANKS:
        torch.testing.assert_close(
            out_cancelled.point_data[name],
            out_plain.point_data[name],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
