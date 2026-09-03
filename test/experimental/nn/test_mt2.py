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

"""Contract tests for MeshTransformer2: every guarantee is exact by construction."""

import pytest
import torch

from physicsnemo.experimental.nn.mt2 import MeshTransformer2


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = MeshTransformer2(hidden=64, n_layers=3, n_slices=32).double().eval()
    n = 500
    ### Anisotropic cloud: the principal-axis frame is exactly covariant
    ### only where the covariance spectrum is non-degenerate (generic for
    ### vehicle geometry; an isotropic cloud is the degenerate corner).
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
    return m, pts, nrm, drv, w, base


def _split(o):
    return o[..., :1], o[..., 1:4]


def test_rotation_equivariance(setup):
    m, pts, nrm, drv, w, base = setup
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_translation_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        tr = m(pts + shift, nrm, drv, w)
    assert torch.allclose(tr, base, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_drive_degree_one(setup, k):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        sc = m(pts, nrm, drv * k, w)
    assert torch.allclose(sc, base * k, atol=1e-10)


def test_measure_weight_scale_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        ws = m(pts, nrm, drv, w * 137.0)
    assert torch.allclose(ws, base, atol=1e-9)


def test_collated_input_shapes(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        out = m(pts, nrm, drv[:, None, :], w[..., None].squeeze(-1))
    assert torch.allclose(out, base, atol=1e-12)


@pytest.fixture
def setup_local():
    torch.manual_seed(0)
    m = (
        MeshTransformer2(
            hidden=64, n_layers=2, n_slices=16,
            use_local_features=True, local_radii=(0.5, 1.5),
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
    return m, pts, nrm, drv, w, base


def test_local_rotation_equivariance(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_local_drive_degree_one(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    with torch.no_grad():
        sc = m(pts, nrm, drv * 2.0, w)
    assert torch.allclose(sc, base * 2.0, atol=1e-10)


@pytest.fixture
def setup_qi():
    torch.manual_seed(0)
    m = (
        MeshTransformer2(
            hidden=64, n_layers=2, n_slices=16,
            query_independent=True, n_decoder_layers=2,
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return m, pts, nrm, drv, w


def test_query_set_independence(setup_qi):
    """THE v5a contract: same source, different query companions ->
    bitwise-identical predictions at shared queries."""
    m, pts, nrm, drv, w = setup_qi
    qa = pts[:, :50]
    na = nrm[:, :50]
    q_big = torch.cat([qa, pts[:, 200:300]], dim=1)
    n_big = torch.cat([na, nrm[:, 200:300]], dim=1)
    with torch.no_grad():
        out_small = m(pts, nrm, drv, w, query_points=qa, query_normals=na)
        out_big = m(pts, nrm, drv, w, query_points=q_big, query_normals=n_big)
    ### Mathematically exact; allclose(1e-12) rather than bitwise because
    ### GEMM tiling reorders reductions when the query count changes.
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)


def test_qi_rotation_equivariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_qi_drive_degree_one(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        sc = m(pts, nrm, drv * 2.0, w)
    assert torch.allclose(sc, base * 2.0, atol=1e-10)


def test_boundary_scalar_channel_contracts():
    torch.manual_seed(0)
    m = (
        MeshTransformer2(
            hidden=64, n_layers=2, n_slices=16, n_boundary_scalars=2
        )
        .double()
        .eval()
    )
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    bs = torch.randn(1, n, 2, dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w, boundary_scalars=bs)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w, boundary_scalars=bs)
    assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)


def test_scale_conditioning_rotation_equivariance():
    """M1 arm: the log-size scalar breaks scale equivariance by design but
    must leave rotation equivariance and translation invariance exact."""
    torch.manual_seed(0)
    m = (
        MeshTransformer2(hidden=64, n_layers=2, n_slices=16, scale_conditioning=True)
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w)
        double = m(pts * 2.0, nrm, drv, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    ### the flag must actually break scale equivariance (else it is inert)
    assert not torch.allclose(double, base, atol=1e-6)


def test_anchor_conditioned_decode_query_independence():
    """v5a4 contract: with a fixed source cloud, the interacting core runs on
    a deterministic anchor subset at eval, so predictions at shared queries
    must not depend on the companion query set."""
    torch.manual_seed(0)
    m = (
        MeshTransformer2(
            hidden=64, n_layers=2, n_slices=16,
            query_independent=True, n_decoder_layers=2, n_anchors=100,
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qa, na = pts[:, :50], nrm[:, :50]
    q_big = torch.cat([qa, pts[:, 200:300]], dim=1)
    n_big = torch.cat([na, nrm[:, 200:300]], dim=1)
    with torch.no_grad():
        out_small = m(pts, nrm, drv, w, query_points=qa, query_normals=na)
        out_big = m(pts, nrm, drv, w, query_points=q_big, query_normals=n_big)
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)

    ### rotation equivariance must survive the anchor subset
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_parity_fix_reflection_equivariance():
    """M3 audit fix: with parity_fix=True the full output must be exactly
    reflection-covariant (scalars invariant, vectors mirrored) -- the parity
    covariance of Navier-Stokes. Without the fix the e_phi pseudovector
    channels break this; the test also asserts the defect is real so the
    fix cannot silently become inert."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))  # mirror

    torch.manual_seed(1)
    fixed = MeshTransformer2(
        hidden=64, n_layers=2, n_slices=16, parity_fix=True, parity_gate_scale=0.1
    )
    fixed = fixed.double().eval()
    torch.manual_seed(1)
    broken = MeshTransformer2(hidden=64, n_layers=2, n_slices=16).double().eval()

    with torch.no_grad():
        base = fixed(pts, nrm, drv, w)
        mirr = fixed(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        base_b = broken(pts, nrm, drv, w)
        mirr_b = broken(pts @ M.T, nrm @ M.T, drv @ M.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ M.T, atol=1e-10)
    ### the unfixed head must violate reflection covariance on vectors
    _, v0b = _split(base_b)
    _, v1b = _split(mirr_b)
    assert not torch.allclose(v1b, v0b @ M.T, atol=1e-6)


@pytest.mark.parametrize("basis", ["true5", "true7"])
def test_true_vector_basis_reflection_and_rotation(basis):
    """L2 arms: the all-true-vector bases must be exactly reflection-covariant
    (no gate) and rotation-equivariant."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = MeshTransformer2(hidden=64, n_layers=2, n_slices=16, vector_basis=basis)
    m = m.double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        mirr = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    p2, v2 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ M.T, atol=1e-10)
    assert torch.allclose(p2, p0, atol=1e-10) and torch.allclose(v2, v0 @ q.T, atol=1e-10)


def test_odd_head_reflection_rotation_and_translation():
    """W2 arm: odd-coefficient head must be exactly O(3)-covariant and
    translation-invariant, and must actually use the e_phi channels."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = MeshTransformer2(hidden=64, n_layers=2, n_slices=16, odd_head=True).double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        mirr = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    p2, v2 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ M.T, atol=1e-10)
    assert torch.allclose(p2, p0, atol=1e-10) and torch.allclose(v2, v0 @ q.T, atol=1e-10)
    ### the odd channels must be live once the (zero-initialized) gate is
    ### non-zero, and must remain exactly reflection-covariant
    with torch.no_grad():
        m.odd_gate.weight.normal_(0.0, 0.1)
        base2 = m(pts, nrm, drv, w)
        mirr2 = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
    _, v0b = _split(base2)
    _, v1b = _split(mirr2)
    assert not torch.allclose(v0b, v0, atol=1e-6)
    assert torch.allclose(v1b, v0b @ M.T, atol=1e-10)


@pytest.mark.parametrize("kw", [{"odd_head": True}, {"parity_fix": True}, {"vector_basis": "true7"}])
def test_head_variants_run_under_bf16_autocast(kw):
    """Mixed-precision smoke: every head variant must survive bf16 autocast
    (the odd-coefficient head once failed with a dtype mismatch at step 0)."""
    torch.manual_seed(0)
    m = MeshTransformer2(hidden=32, n_layers=1, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(pts, nrm, drv, w)
    out.float().sum().backward()
    assert torch.isfinite(out.float()).all()


def test_similarity_gauge_geometric_scale_equivariance():
    """S1: with the measure-weighted gauge, a geometric rescale (points x k,
    areas x k^2) must leave the output exactly unchanged; the default gauge
    must NOT (documenting that the original model is not scale-equivariant)."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    torch.manual_seed(1)
    mg = MeshTransformer2(hidden=64, n_layers=2, n_slices=16, similarity_gauge=True).double().eval()
    torch.manual_seed(1)
    m0 = MeshTransformer2(hidden=64, n_layers=2, n_slices=16).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        a = mg(pts, nrm, drv, w)
        bsc = mg(k * pts + shift, nrm, drv, k * k * w)
        a0 = m0(pts, nrm, drv, w)
        b0 = m0(k * pts, nrm, drv, k * k * w)
    assert torch.allclose(bsc, a, atol=1e-10)
    assert not torch.allclose(b0, a0, atol=1e-3)
