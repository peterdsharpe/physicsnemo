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

"""Contract tests for ISLA (Invariant Slice Attention): every guarantee is exact by construction."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA

# The contract tests below were written for the centered construction (plain-mean
# centre, constant reference_length), which was the class default until 2026-09-11.
# The class default is now the relative frame with the total-measure scale (the
# reference configuration); these tests pin the centered construction explicitly so
# they keep verifying it, and test_default_is_relative_total_measure covers the default.
_CENTERED = dict(frame_mode="centered", scale_mode="reference_length")



@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32).double().eval()
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
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    return m, pts, nrm, drv, w, base


def _split(o):
    return o[..., :1], o[..., 1:4]


def test_rotation_equivariance(setup):
    m, pts, nrm, drv, w, base = setup
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_translation_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        tr = m(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w)
    assert torch.allclose(tr, base, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_global_vector_magnitude_invariance(setup, k):
    """Global vector inputs enter as unit directions: rescaling one leaves every
    output unchanged (a physically meaningful magnitude is a global scalar input)."""
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        sc = m(points=pts, normals=nrm, global_vectors=drv * k, measure_weights=w)
    assert torch.allclose(sc, base, atol=1e-10)


def test_measure_weight_scale_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        ws = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w * 137.0)
    assert torch.allclose(ws, base, atol=1e-9)


def test_collated_input_shapes(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        out = m(points=pts, normals=nrm, global_vectors=drv[:, None, :], measure_weights=w[..., None].squeeze(-1))
    assert torch.allclose(out, base, atol=1e-12)


@pytest.fixture
def setup_local():
    torch.manual_seed(0)
    m = (
        ISLA(**_CENTERED, 
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
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    return m, pts, nrm, drv, w, base


def test_local_rotation_equivariance(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_local_global_vector_magnitude_invariance(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    with torch.no_grad():
        sc = m(points=pts, normals=nrm, global_vectors=drv * 2.0, measure_weights=w)
    assert torch.allclose(sc, base, atol=1e-10)


@pytest.fixture
def setup_qi():
    torch.manual_seed(0)
    m = (
        ISLA(**_CENTERED, 
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
        out_small = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qa, query_normals=na)
        out_big = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q_big, query_normals=n_big)
    ### Mathematically exact; allclose(1e-12) rather than bitwise because
    ### GEMM tiling reorders reductions when the query count changes.
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)


def test_qi_rotation_equivariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_qi_global_vector_magnitude_invariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        sc = m(points=pts, normals=nrm, global_vectors=drv * 2.0, measure_weights=w)
    assert torch.allclose(sc, base, atol=1e-10)


def test_boundary_scalar_channel_contracts():
    torch.manual_seed(0)
    m = (
        ISLA(**_CENTERED, 
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
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, boundary_scalars=bs)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, boundary_scalars=bs)
    assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)


def test_scale_conditioning_rotation_equivariance():
    """M1 arm: the log-size scalar breaks scale equivariance by design but
    must leave rotation equivariance and translation invariance exact."""
    torch.manual_seed(0)
    m = (
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, scale_conditioning=True)
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
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
        double = m(points=pts * 2.0, normals=nrm, global_vectors=drv, measure_weights=w)
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
        ISLA(**_CENTERED, 
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
        out_small = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qa, query_normals=na)
        out_big = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q_big, query_normals=n_big)
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)

    ### rotation equivariance must survive the anchor subset
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
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
    fixed = ISLA(**_CENTERED, 
        hidden=64, n_layers=2, n_slices=16, parity_fix=True, parity_gate_scale=0.1
    )
    fixed = fixed.double().eval()
    torch.manual_seed(1)
    broken = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()

    with torch.no_grad():
        base = fixed(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        mirr = fixed(points=pts @ M.T, normals=nrm @ M.T, global_vectors=drv @ M.T, measure_weights=w)
        base_b = broken(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        mirr_b = broken(points=pts @ M.T, normals=nrm @ M.T, global_vectors=drv @ M.T, measure_weights=w)
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
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, vector_basis=basis)
    m = m.double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        mirr = m(points=pts @ M.T, normals=nrm @ M.T, global_vectors=drv @ M.T, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
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
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, odd_head=True).double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        mirr = m(points=pts @ M.T, normals=nrm @ M.T, global_vectors=drv @ M.T, measure_weights=w)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    p2, v2 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ M.T, atol=1e-10)
    assert torch.allclose(p2, p0, atol=1e-10) and torch.allclose(v2, v0 @ q.T, atol=1e-10)
    ### the odd channels must be live once the (zero-initialized) gate is
    ### non-zero, and must remain exactly reflection-covariant
    with torch.no_grad():
        m.odd_gate.weight.normal_(0.0, 0.1)
        base2 = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        mirr2 = m(points=pts @ M.T, normals=nrm @ M.T, global_vectors=drv @ M.T, measure_weights=w)
    _, v0b = _split(base2)
    _, v1b = _split(mirr2)
    assert not torch.allclose(v0b, v0, atol=1e-6)
    assert torch.allclose(v1b, v0b @ M.T, atol=1e-10)


@pytest.mark.parametrize("kw", [{"odd_head": True}, {"parity_fix": True}, {"vector_basis": "true7"}])
def test_head_variants_run_under_bf16_autocast(kw):
    """Mixed-precision smoke: every head variant must survive bf16 autocast
    (the odd-coefficient head once failed with a dtype mismatch at step 0)."""
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=1, n_slices=8, **{**_CENTERED, **kw})
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
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
    mg = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, similarity_gauge=True).double().eval()
    torch.manual_seed(1)
    m0 = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        a = mg(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        bsc = mg(points=k * pts + shift, normals=nrm, global_vectors=drv, measure_weights=k * k * w)
        a0 = m0(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        b0 = m0(points=k * pts, normals=nrm, global_vectors=drv, measure_weights=k * k * w)
    assert torch.allclose(bsc, a, atol=1e-10)
    assert not torch.allclose(b0, a0, atol=1e-3)


def test_raw_coord_channel_breaks_equivariance_by_design():
    """Branch-B discriminator flag: must run, must change the output, and must
    NOT be rotation-equivariant (that is the point); default stays exact."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, similarity_gauge=True,
                         raw_coord_channel=True).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    p0, _ = _split(base)
    p1, _ = _split(rot)
    assert torch.isfinite(base).all()
    assert not torch.allclose(p1, p0, atol=1e-3)


def test_interior_queries_contracts():
    """V0 boundary->interior mode: off-surface queries without normals must be
    exactly rotation/translation-equivariant, query-set independent, and
    finite; the derived proxy normal must not be degenerate."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0  # interior/exterior points, no normals
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, query_independent=True,
                         n_decoder_layers=2, interior_queries=True, similarity_gauge=True,
                         out_scalars=1, out_vectors=1).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, query_points=qpts @ q.T + shift)
        sub = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts[:, :50])
    assert torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(sub, base[:, :50], atol=1e-12, rtol=0.0)


def test_latent_volume_tokens_contracts():
    """Branch V: latent volume tokens keep exact SE(3) covariance and query
    independence for interior queries, and are live (change the output)."""
    torch.manual_seed(0)
    n, nq = 400, 120
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    kw = dict(hidden=64, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2,
              interior_queries=True, similarity_gauge=True, out_scalars=1, out_vectors=1)
    torch.manual_seed(1)
    m = ISLA(latent_volume_tokens=True, **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(1)
    m0 = ISLA(latent_volume_tokens=False, **{**_CENTERED, **kw}).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, query_points=qpts @ q.T + shift)
        sub = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts[:, :40])
        plain = m0(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts)
    assert torch.isfinite(base).all()
    p0, v0 = _split(base); p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(sub, base[:, :40], atol=1e-12, rtol=0.0)
    assert not torch.allclose(base, plain, atol=1e-6)


def _qt_case(n=400, nq=150, seed=0):
    torch.manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    qnrm = torch.nn.functional.normalize(torch.randn(1, nq, 3, dtype=torch.float64), dim=-1)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    return pts, nrm, drv, w, qpts, qnrm, q, shift


@pytest.mark.parametrize("flag", ["wake_tokens", "latent_volume_tokens"])
def test_context_tokens_with_query_tokens_contracts(flag):
    """Wake tokens (2026-09-10) and latent volume tokens in the interacting
    query-token mode: exact SE(3) covariance, correct output count (only the
    queries are read), invariance to a uniform rescale of the measure weights
    and to splitting every surface token into two half-weight copies
    (Horvitz-Thompson refinement), liveness, gradients everywhere, and the
    constructor contract (some interior query path is required)."""
    pts, nrm, drv, w, qpts, qnrm, q, shift = _qt_case()
    # query_mass="source_total" is the refinement-invariant convention (the one D1 uses); the
    # default per-token mean is not invariant to splitting tokens, independently of this flag.
    kw = dict(hidden=64, n_layers=2, n_slices=16, query_tokens=True, similarity_gauge=True,
              query_mass="source_total", out_scalars=1, out_vectors=1)
    torch.manual_seed(1)
    m = ISLA(**{flag: True}, **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(1)
    m0 = ISLA(**{**_CENTERED, **kw}).double().eval()
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, query_points=qpts @ q.T + shift, query_normals=qnrm @ q.T)
        scaled_w = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w * 7.3, query_points=qpts, query_normals=qnrm)
        split = m(points=torch.cat([pts, pts], 1), normals=torch.cat([nrm, nrm], 1), global_vectors=drv, measure_weights=torch.cat([w, w], 1) / 2,
                  query_points=qpts, query_normals=qnrm)
        plain = m0(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
    assert base.shape == (1, qpts.shape[1], 4) and torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(scaled_w, base, atol=1e-10)
    assert torch.allclose(split, base, atol=1e-8)
    assert not torch.allclose(base, plain, atol=1e-6)
    m.train()
    out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
    out.square().mean().backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, **{flag: True})


def test_wake_tokens_extent_is_sampling_invariant():
    """The wake tokens' positions depend on the surface only through its
    measure-weighted drive-aligned extent and centre. A 10:1 biased resample
    of the surface with exact inverse-inclusion weights reproduces that
    extent (checked directly on the weighted statistic) and yields a finite
    prediction at fixed downstream queries."""
    torch.manual_seed(3)
    n = 4000
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(pts, dim=-1)
    drv = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    w = torch.ones(1, n, dtype=torch.float64)
    qpts = torch.randn(1, 60, 3, dtype=torch.float64) * 4.0 + torch.tensor([6.0, 0.0, 0.0], dtype=torch.float64)
    qnrm = torch.nn.functional.normalize(torch.randn(1, 60, 3, dtype=torch.float64), dim=-1)
    torch.manual_seed(1)
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, query_tokens=True, similarity_gauge=True, wake_tokens=True,
             query_mass="source_total", out_scalars=1, out_vectors=1).double().eval()
    front = pts[0, :, 0] < pts[0, :, 0].median()
    pi = torch.where(front, torch.full((n,), 0.5, dtype=torch.float64), torch.full((n,), 0.05, dtype=torch.float64))
    keep = torch.rand(n, dtype=torch.float64) < pi
    with torch.no_grad():
        full = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
        biased = m(points=pts[:, keep], normals=nrm[:, keep], global_vectors=drv, measure_weights=w[:, keep] / pi[keep], query_points=qpts, query_normals=qnrm)
    s = pts[0, :, 0]
    s_b, wb = s[keep], 1.0 / pi[keep]
    ell_full = s.std(unbiased=False)
    mean_b = (wb * s_b).sum() / wb.sum()
    ell_b = torch.sqrt((wb * (s_b - mean_b) ** 2).sum() / wb.sum())
    assert torch.isfinite(full).all() and torch.isfinite(biased).all()
    assert abs(ell_b - ell_full) / ell_full < 0.05


def test_a35b_ablation_flags_run_and_differ():
    """A35b: raw seeds break equivariance by design; no-relational-geo stays
    exactly equivariant; both run, are finite, and change the output."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    torch.manual_seed(1)
    ref = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    nogeo = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, use_relational_geo=False).double().eval()
    raw = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, seed_mode="raw").double().eval()
    with torch.no_grad():
        o_ref = ref(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        o_ng = nogeo(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        o_ng_rot = nogeo(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
        o_raw = raw(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        o_raw_rot = raw(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
    assert torch.isfinite(o_ng).all() and torch.isfinite(o_raw).all()
    assert not torch.allclose(o_ng, o_ref, atol=1e-6)
    p0, v0 = _split(o_ng); p1, v1 = _split(o_ng_rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    pr0, _ = _split(o_raw); pr1, _ = _split(o_raw_rot)
    assert not torch.allclose(pr1, pr0, atol=1e-3)


@pytest.mark.parametrize("kw", [{}, {"use_relational_geo": False}, {"seed_mode": "raw"},
                                {"odd_head": True}, {"similarity_gauge": True},
                                {"frame_mode": "relative"},
                                {"frame_mode": "relative", "scale_mode": "total_measure",
                                 "query_independent": True, "n_decoder_layers": 1}])
def test_all_parameters_receive_gradients(kw):
    """DDP requires every parameter to take part in the loss; a module built
    but skipped in forward crashes distributed training (A35b nogeo incident)."""
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, **{**_CENTERED, **kw})
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w).sum().backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    assert not unused, unused


def test_geo_pool_then_project_is_exact():
    """Pooling the 8 relational invariants over slices and then projecting is
    exactly the old project-then-pool (the projection is affine and the slice
    mix is a softmax over slices, so the bias passes through unchanged). This
    is the identity behind the memory saving: the saved activation is
    (B, N, 8) instead of (B, N, S, hidden/2)."""
    from physicsnemo.experimental.nn.isla.model import _ReadBlock, _SliceBlock

    torch.manual_seed(0)
    for blk in (_SliceBlock(64, 32).double(), _ReadBlock(64, 32).double()):
        geo = torch.randn(2, 50, 32, blk.n_geo, dtype=torch.float64)
        mix = torch.softmax(torch.randn(2, 50, 32, dtype=torch.float64), dim=-1)
        old = torch.einsum("bns,bnsg->bng", mix, blk.geo_feat(geo))
        new = blk.geo_feat(torch.einsum("bns,bnsg->bng", mix, geo))
        assert torch.allclose(old, new, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("extra", [{}, {"query_independent": True, "interior_queries": True}])
def test_geo_checkpoint_is_exact(extra):
    """Recomputing the per-slice invariants in backward (geo_checkpoint=True)
    changes what autograd stores, not what it computes: identical forward
    output and gradients (to roundoff) against the stored-activation path,
    for the encoder blocks and the passive decoder blocks."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8, **extra)
    ref = ISLA(**{**_CENTERED, **kw}).double()
    ckp = ISLA(geo_checkpoint=True, **{**_CENTERED, **kw}).double()
    ckp.load_state_dict(ref.state_dict())
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 40, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    fk = {}
    if extra:
        fk = dict(query_points=torch.randn(1, 17, 3, dtype=torch.float64), query_normals=torch.nn.functional.normalize(torch.randn(1, 17, 3, dtype=torch.float64), dim=-1))
    outs = []
    for m in (ref, ckp):
        out = m(points=pts, normals=nrm, global_vectors=drive, measure_weights=w, **fk)
        out.square().sum().backward()
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    for (n, p), (_, q) in zip(ref.named_parameters(), ckp.named_parameters()):
        assert p.grad is not None and q.grad is not None, n
        assert torch.allclose(p.grad, q.grad, atol=1e-11, rtol=1e-9), n


def test_query_scalars_contracts():
    """Per-query scalar inputs on the query tokens (e.g. the signed distance
    to the wall): exact rotation/translation covariance is untouched (scalars
    are invariants), the scalar channel changes the output and receives
    gradients, geometric-scale equivariance holds under the similarity gauge
    with the "length" scaling, and misuse raises."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8,
              query_tokens=True, similarity_gauge=True, n_query_scalars=1)
    m = ISLA(**{**_CENTERED, **kw}).double()
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 40, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64)
    qn = torch.nn.functional.normalize(torch.randn(1, 12, 3, dtype=torch.float64), dim=-1)
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    out = m(points=pts, normals=nrm, global_vectors=drive, measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf)
    assert out.shape == (1, 12, 4)
    ### scalar channel is live
    out2 = m(points=pts, normals=nrm, global_vectors=drive, measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf * 2)
    assert not torch.allclose(out, out2)
    ### rotation + translation covariance (scalars ride along unchanged)
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(points=pts @ R.T + t, normals=nrm @ R.T, global_vectors=drive @ R.T, measure_weights=w, query_points=q @ R.T + t,
            query_normals=qn @ R.T, query_scalars=sdf)
    assert torch.allclose(rot[..., 0], out[..., 0], atol=1e-10)
    assert torch.allclose(rot[..., 1:], out[..., 1:] @ R.T, atol=1e-10)
    ### geometric-scale equivariance: lengths scale, so must the scalar
    s = 3.7
    sc = m(points=pts * s, normals=nrm, global_vectors=drive, measure_weights=w * s**2, query_points=q * s, query_normals=qn,
           query_scalars=sdf * s)
    assert torch.allclose(sc, out, atol=1e-10)
    ### gradients reach the scalar embedding
    out.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt_scalar_embed.parameters())
    with pytest.raises(ValueError):
        m(points=pts, normals=nrm, global_vectors=drive, measure_weights=w, query_points=q, query_normals=qn)
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, out_scalars=1, out_vectors=1, hidden=32, n_layers=1, n_slices=8, n_query_scalars=1)


def test_query_local_features_contracts():
    """Local surface-patch features on the query tokens: exact rotation,
    translation and (with the similarity gauge) scale covariance hold, the
    channel changes the output and receives gradients, and it requires
    query_tokens."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8,
              query_tokens=True, similarity_gauge=True, n_query_scalars=1,
              query_local_features=True, query_local_radii=(0.1, 0.3))
    m = ISLA(**{**_CENTERED, **kw}).double()
    m0 = ISLA(**{**_CENTERED, **kw, "query_local_features": False}).double()
    pts = torch.randn(1, 60, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 60, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64) * 0.5
    qn = torch.nn.functional.normalize(torch.randn(1, 12, 3, dtype=torch.float64), dim=-1)
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    args = dict(measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf)
    out = m(points=pts, normals=nrm, global_vectors=drive, **args)
    assert out.shape == (1, 12, 4)
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(points=pts @ R.T + t, normals=nrm @ R.T, global_vectors=drive @ R.T, measure_weights=w, query_points=q @ R.T + t,
            query_normals=qn @ R.T, query_scalars=sdf)
    assert torch.allclose(rot[..., 0], out[..., 0], atol=1e-10)
    assert torch.allclose(rot[..., 1:], out[..., 1:] @ R.T, atol=1e-10)
    s = 2.3
    sc = m(points=pts * s, normals=nrm, global_vectors=drive, measure_weights=w * s**2, query_points=q * s, query_normals=qn,
           query_scalars=sdf * s)
    assert torch.allclose(sc, out, atol=1e-10)
    out.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt_local_embed.parameters())
    ### channel is live: zeroing its embedding output recovers the no-channel model's function
    m0.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("qt_local_embed")})
    assert not torch.allclose(m0(points=pts, normals=nrm, global_vectors=drive, **args), out)
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, out_scalars=1, out_vectors=1, hidden=32, n_layers=1, n_slices=8, query_local_features=True)


def test_legacy_name_is_an_alias():
    """The previous name and import path keep working (cluster configs, checkpoints)."""
    from physicsnemo.experimental.nn import MeshTransformer2 as legacy_top
    from physicsnemo.experimental.nn.mt2 import MeshTransformer2 as legacy_path

    assert legacy_top is ISLA and legacy_path is ISLA


def test_query_tokens_contracts():
    """Query-token mode (interior queries as interacting tokens): exactly
    rotation/translation-covariant, finite, live (differs from the passive
    interior decode), and -- by design -- NOT query-set independent."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    qnrm = torch.nn.functional.normalize(torch.randn(1, nq, 3, dtype=torch.float64), dim=-1)
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, query_tokens=True, similarity_gauge=True,
             out_scalars=1, out_vectors=1).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
        rot = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, query_points=qpts @ q.T + shift, query_normals=qnrm @ q.T)
        sub = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts[:, :50], query_normals=qnrm[:, :50])
    assert base.shape == (1, nq, 4) and torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    # interacting mode: predictions at shared queries depend on the query set
    assert not torch.allclose(sub, base[:, :50], atol=1e-6)
    # every parameter receives a gradient (DDP safety)
    m.train()
    out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=qpts, query_normals=qnrm)
    out.square().mean().backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, query_tokens=True, query_independent=True)


@pytest.mark.parametrize("extra", [{}, {"n_query_scalars": 1, "query_scalar_scale": "length"}])
def test_query_mass_source_total_refinement_invariance(extra):
    """Audit 2026-09-08: splitting every source token into two half-weight
    copies leaves the discrete source measure unchanged (positions, normals,
    total area, every integral). With query_mass="source_total" the query
    tokens' weight is a fraction of the total source measure, so the output
    is exactly unchanged and the similarity-gauge scale contract still holds;
    the default "geometric_mean" is the trained convention and is asserted to
    keep its (refinement-dependent) formula so checkpoints reproduce."""
    torch.manual_seed(42)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q, qn = pts[:, :17], nrm[:, :17]
    fk = dict(query_points=q, query_normals=qn)
    if extra:
        fk["query_scalars"] = q.norm(dim=-1)
    kw = dict(hidden=32, n_layers=2, n_slices=8, similarity_gauge=True, query_tokens=True, **extra)
    torch.manual_seed(0)
    total = ISLA(query_mass="source_total", **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(0)
    default = ISLA(**{**_CENTERED, **kw}).double().eval()
    refined = dict(points=pts.repeat_interleave(2, 1), normals=nrm.repeat_interleave(2, 1), global_vectors=drv,
                   measure_weights=w.repeat_interleave(2, 1) / 2)
    s = 3.7
    scaled_fk = {**fk, "query_points": q * s}
    if extra:
        scaled_fk["query_scalars"] = fk["query_scalars"] * s
    with torch.no_grad():
        a = total(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk)
        b = total(**refined, **fk)
        c = total(points=pts * s, normals=nrm, global_vectors=drv, measure_weights=w * s**2, **scaled_fk)
        a0 = default(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk)
        b0 = default(**refined, **fk)
    assert torch.allclose(b, a, atol=1e-10, rtol=0.0)
    assert torch.allclose(c, a, atol=1e-10, rtol=0.0)
    ### regression guard: the default convention is unchanged (and therefore
    ### still refinement-dependent); the two conventions are not the same model
    assert not torch.allclose(b0, a0, atol=1e-3)
    assert not torch.allclose(a0, a, atol=1e-6)
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, query_mass="source_total")
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, query_tokens=True, query_mass="mean")


@pytest.mark.parametrize("extra", [{}, {"query_independent": True, "n_decoder_layers": 2}])
def test_similarity_gauge_local_features_scale_equivariance(extra):
    """Audit 2026-09-08: under the similarity gauge the surface patch
    integrals normalize the measure weights to fractions of the total, so a
    geometric rescale (points x k, areas x k^2) leaves log(mass) -- and the
    output -- exactly unchanged, for the encoder seeds and the passive
    decoder's query-side patch integrals alike."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(hidden=64, n_layers=2, n_slices=16, similarity_gauge=True,
             use_local_features=True, local_radii=(0.5, 1.5), **{**_CENTERED, **extra}).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    fk = dict(query_points=pts[:, :50], query_normals=nrm[:, :50]) if extra else {}
    fk_sc = {**fk, "query_points": k * fk["query_points"] + shift} if extra else {}
    with torch.no_grad():
        a = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk)
        b = m(points=k * pts + shift, normals=nrm, global_vectors=drv, measure_weights=k * k * w, **fk_sc)
    assert torch.allclose(b, a, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("kw", [{"scale_conditioning": True}, {"seed_mode": "raw"},
                                {"raw_coord_channel": True}, {"odd_head": True},
                                {"odd_head": True, "n_anchors": 30}])
def test_passive_decode_seed_and_head_options(kw):
    """Audit 2026-09-08: passive decoding at a query set whose size differs
    from the source used to fail with a shape error for every seed-widening
    option (the query seed lacked the extra channels) and for the odd head
    (it read the source normal and count after they were overwritten by the
    query-side values). Now: finite outputs of the right shape, and the
    passive contract -- a prediction at one query does not depend on which
    other points are queried -- holds to 1e-12."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = torch.randn(1, 17, 3, dtype=torch.float64) * 2.0
    qn = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2, **{**_CENTERED, **kw})
    m = m.double().eval()
    if kw.get("odd_head"):
        with torch.no_grad():
            m.odd_gate.weight.normal_(0.0, 0.1)  # make the odd channels live
    with torch.no_grad():
        out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q, query_normals=qn)
        sub = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q[:, :8], query_normals=qn[:, :8])
    assert out.shape == (1, 17, 4) and torch.isfinite(out).all()
    assert torch.allclose(sub, out[:, :8], atol=1e-12, rtol=0.0)


def test_passive_decode_boundary_scalars():
    """Audit 2026-09-08: boundary scalars are per-boundary-cell data. Passive
    decoding of the boundary itself (query_points=None) carries them into the
    query seeds and runs; distinct query points carry zeros in the channel
    (GLOBAL INPUTS, 2026-09-14) and decode without a shape error."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    bs = torch.randn(1, 60, 2, dtype=torch.float64)
    m = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2,
             n_boundary_scalars=2).double().eval()
    with torch.no_grad():
        out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, boundary_scalars=bs)
        out2 = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, boundary_scalars=bs * 2)
    assert out.shape == (1, 60, 4) and torch.isfinite(out).all()
    assert not torch.allclose(out, out2, atol=1e-6)  # the channel is live on the query side
    with torch.no_grad():
        out_q = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, boundary_scalars=bs, query_points=pts[:, :17] * 0.5, query_normals=nrm[:, :17])
    assert out_q.shape == (1, 17, 4) and torch.isfinite(out_q).all()


def _rot_z(deg):
    c, s = torch.cos(torch.deg2rad(torch.tensor(deg, dtype=torch.float64))), torch.sin(
        torch.deg2rad(torch.tensor(deg, dtype=torch.float64))
    )
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _collision_pair(angles_a, angles_b):
    """Two five-component arrangements with zero transverse first moment about an
    axial drive (audit 2026-09-08, item 3): first-moment anchors cannot tell them apart."""
    normal = torch.tensor(
        [[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)], dtype=torch.float64
    ) / 3**0.5
    point = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64) + 0.15 * normal

    def cloud(angles):
        ps, ns = [], []
        for a in angles:
            R = _rot_z(a)
            ps.append(point @ R.T)
            ns.append(normal @ R.T)
        return torch.cat(ps)[None], torch.cat(ns)[None]

    (p1, n1), (p2, n2) = cloud(angles_a), cloud(angles_b)
    d = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    w = torch.full((1, 40), 4 * torch.pi * 0.15**2 / 8, dtype=torch.float64)
    return p1, n1, p2, n2, d, w


@pytest.mark.parametrize("extra", [{}, {"similarity_gauge": True}, {"geo_checkpoint": True}])
def test_second_moment_features_contracts(extra):
    """MOM2 channel: exact SE(3) covariance, global-vector magnitude invariance, measure-scale
    invariance, and (with the gauge) geometric-scale equivariance."""
    torch.manual_seed(0)
    m = ISLA(hidden=64, n_layers=3, n_slices=32, second_moment_features=True, **{**_CENTERED, **extra}).double().eval()
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        moved = m(points=pts @ q.T + shift, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
        rescaled_w = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=3.7 * w)
    p0, v0 = _split(base)
    p1, v1 = _split(moved)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(rescaled_w, base, atol=1e-10)
    if extra.get("similarity_gauge"):
        with torch.no_grad():
            scaled = m(points=2.7 * pts, normals=nrm, global_vectors=drv, measure_weights=2.7**2 * w)
        assert torch.allclose(scaled, base, atol=1e-10)
    # the channel is live: outputs differ from the eight-invariant model with the same seed
    torch.manual_seed(0)
    m8 = ISLA(hidden=64, n_layers=3, n_slices=32, **{**_CENTERED, **extra}).double().eval()
    assert m8.blocks[0].geo_logit.in_features == 8 and m.blocks[0].geo_logit.in_features == 10


def test_second_moment_features_separate_first_moment_collision():
    """The audit's counterexample: identical eight-invariant outputs on the common
    component, separated once the second-moment channel is on."""
    p1, n1, p2, n2, d, w = _collision_pair([0, 120, 240, 27, 207], [0, 120, 240, 43, 223])
    torch.manual_seed(0)
    base = ISLA(**_CENTERED, hidden=64, n_layers=4, n_slices=32).double().eval()
    torch.manual_seed(0)
    mom2 = ISLA(**_CENTERED, hidden=64, n_layers=4, n_slices=32, second_moment_features=True).double().eval()
    with torch.no_grad():
        a0, b0 = base(points=p1, normals=n1, global_vectors=d, measure_weights=w), base(points=p2, normals=n2, global_vectors=d, measure_weights=w)
        a2, b2 = mom2(points=p1, normals=n1, global_vectors=d, measure_weights=w), mom2(points=p2, normals=n2, global_vectors=d, measure_weights=w)
    assert (a0[:, :8] - b0[:, :8]).abs().max() < 1e-12  # blind by construction
    assert (a2[:, :8] - b2[:, :8]).abs().max() > 1e-4  # separated
    # congruent control: the same arrangement rotated about the drive is identical
    R = _rot_z(37.0)
    with torch.no_grad():
        rot = mom2(points=p1 @ R.T, normals=n1 @ R.T, global_vectors=d @ R.T, measure_weights=w)
    assert torch.allclose(rot[..., :1], a2[..., :1], atol=1e-10)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_measure_weight_power_contracts(alpha):
    """Tempered routing measure w^alpha: alpha=0 equals use_measure_weights=False exactly,
    alpha=1 is the default, every alpha keeps SE(3) covariance and measure-scale invariance."""
    torch.manual_seed(0)
    m = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32, measure_weight_power=alpha).double().eval()
    torch.manual_seed(0)
    m_off = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32, use_measure_weights=False).double().eval()
    torch.manual_seed(0)
    m_ref = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32).double().eval()
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        rot = m(points=pts @ q.T + 2.0, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
        scaled_w = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=4.1 * w)
        off = m_off(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        ref = m_ref(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    p0, v0 = _split(base); p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(scaled_w, base, atol=1e-10)
    if alpha == 0.0:
        assert torch.allclose(base, off, atol=1e-12)
    else:
        assert not torch.allclose(base, ref, atol=1e-6) and not torch.allclose(base, off, atol=1e-6)


@pytest.mark.parametrize("kw", [{"query_density_feature": True}, {"query_neighbor_features": True},
                                {"query_density_feature": True, "query_neighbor_features": True, "n_query_scalars": 1}])
def test_query_cloud_channels_contracts(kw):
    """QTDENS channels (density / neighbour aggregation over the query cloud): exact SE(3)
    covariance, global-vector magnitude invariance, measure-scale invariance, gauge scale equivariance, and
    a live channel (outputs differ from the plain query-token model with the same seed)."""
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, query_tokens=True, similarity_gauge=True, **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(0)
    m0 = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8, query_tokens=True, similarity_gauge=True,
              n_query_scalars=kw.get("n_query_scalars", 0)).double().eval()
    torch.manual_seed(1)
    p = torch.randn(1, 80, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    n = torch.nn.functional.normalize(torch.randn_like(p), dim=-1)
    d = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 80, dtype=torch.float64) + 0.5
    q = torch.randn(1, 40, 3, dtype=torch.float64) * 0.5
    qn = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    extra = {"query_scalars": q.norm(dim=-1, keepdim=True)} if kw.get("n_query_scalars") else {}
    R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=p, normals=n, global_vectors=d, measure_weights=w, query_points=q, query_normals=qn, **extra)
        moved = m(points=p @ R.T + t, normals=n @ R.T, global_vectors=d @ R.T, measure_weights=w, query_points=q @ R.T + t, query_normals=qn @ R.T, **extra)
        scaled_w = m(points=p, normals=n, global_vectors=d, measure_weights=2.5 * w, query_points=q, query_normals=qn, **extra)
        sc_extra = {"query_scalars": 1.7 * extra["query_scalars"]} if extra else {}
        scaled = m(points=1.7 * p, normals=n, global_vectors=d, measure_weights=1.7**2 * w, query_points=1.7 * q, query_normals=qn, **sc_extra)
        plain = m0(points=p, normals=n, global_vectors=d, measure_weights=w, query_points=q, query_normals=qn, **extra)
    s0, v0 = base[..., :1], base[..., 1:4]
    s1, v1 = moved[..., :1], moved[..., 1:4]
    assert torch.allclose(s1, s0, atol=1e-10) and torch.allclose(v1, v0 @ R.T, atol=1e-10)
    assert torch.allclose(scaled_w, base, atol=1e-10)
    assert torch.allclose(scaled, base, atol=1e-9)
    assert (plain - base).abs().max() > 1e-6
    with torch.no_grad():
        dens, nbr = m._query_cloud_invariants(q - q.mean(1, keepdim=True), qn, d[:, None, None].expand(1, 40, 1, 3), None)
    assert dens.shape == (1, 40, 2) and (nbr is None or nbr.shape[-1] == 8)


def _biased_poisson_subsample(pts, w, n_expected, bias, generator):
    """10:1 Poisson subsample with exact Horvitz-Thompson weights, mirroring the
    recipe's PoissonBiasedSubsampleMesh (bias toward x below the median)."""
    x = pts[0, :, 0]
    b = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    pi = (n_expected / b.sum() * b).clamp(max=1.0)
    keep = torch.rand(x.shape[0], dtype=pi.dtype, generator=generator) < pi
    idx = keep.nonzero(as_tuple=True)[0]
    return pts[:, idx], (w[:, idx] / pi[idx]), idx


def test_center_mode_contracts():
    """center_mode='plain' is the pre-flag model; 'measure' with uniform weights
    equals 'plain' to 1e-12; both modes are translation invariant."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    torch.manual_seed(1)
    m_default = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    m_plain = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, center_mode="plain").double().eval()
    torch.manual_seed(1)
    m_meas = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, center_mode="measure").double().eval()
    with torch.no_grad():
        a = m_default(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        a_plain = m_plain(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        a_meas = m_meas(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        a_meas_unif = m_meas(points=pts, normals=nrm, global_vectors=drv, measure_weights=torch.ones_like(w))
        a_plain_unif = m_plain(points=pts, normals=nrm, global_vectors=drv, measure_weights=torch.ones_like(w))
        a_meas_now = m_meas(points=pts, normals=nrm, global_vectors=drv, measure_weights=None)
        a_plain_now = m_plain(points=pts, normals=nrm, global_vectors=drv, measure_weights=None)
    assert torch.equal(a_plain, a)
    assert torch.allclose(a_meas_unif, a_plain_unif, atol=1e-12)
    assert torch.allclose(a_meas_now, a_plain_now, atol=1e-12)
    assert not torch.allclose(a_meas, a_plain, atol=1e-6)  # non-uniform weights: a live channel
    with torch.no_grad():
        assert torch.allclose(m_plain(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w), a_plain, atol=1e-10)
        assert torch.allclose(m_meas(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w), a_meas, atol=1e-10)
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, center_mode="weighted")


def test_center_mode_measure_is_sampling_bias_robust():
    """The discriminating contract. Under a 10:1 biased Poisson subsample with
    exact HT weights the measure-weighted centroid stays near the full-cloud
    weighted centroid while the plain mean moves by a large fraction of the
    cloud radius; and a measure-centered ISLA's predictions at fixed queries
    move less between a uniform and a biased subsample than a plain-centered
    one's (ordering only, several seeds)."""
    torch.manual_seed(0)
    n_full = 100000
    pts = torch.randn(1, n_full, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n_full, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n_full, dtype=torch.float64) + 0.5
    w_n = w / w.sum()
    c_full = (w_n[..., None] * pts).sum(dim=1)
    radius = (w_n * (pts - c_full[:, None]).norm(dim=-1)).sum()
    q = pts[:, :40]
    qn = nrm[:, :40]
    torch.manual_seed(1)
    m_plain = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=1).double().eval()
    torch.manual_seed(1)
    m_meas = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=1,
                  center_mode="measure").double().eval()
    n_sub = 10000  # the probe's token budget; ~900 tokens land in the 10x-undersampled half
    wins = 0
    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        p_u, w_u, iu = _biased_poisson_subsample(pts, w, n_sub, 1.0, g)
        p_b, w_b, ib = _biased_poisson_subsample(pts, w, n_sub, 10.0, g)
        c_meas = ((w_b / w_b.sum())[..., None] * p_b).sum(dim=1)
        c_plain = p_b.mean(dim=1)
        assert (c_meas - c_full).norm() < 0.15 * radius  # ~4 sigma of the HT estimate
        assert (c_plain - c_full).norm() > 0.3 * radius
        with torch.no_grad():
            o_pu = m_plain(points=p_u, normals=nrm[:, iu], global_vectors=drv, measure_weights=w_u, query_points=q, query_normals=qn)
            o_pb = m_plain(points=p_b, normals=nrm[:, ib], global_vectors=drv, measure_weights=w_b, query_points=q, query_normals=qn)
            o_mu = m_meas(points=p_u, normals=nrm[:, iu], global_vectors=drv, measure_weights=w_u, query_points=q, query_normals=qn)
            o_mb = m_meas(points=p_b, normals=nrm[:, ib], global_vectors=drv, measure_weights=w_b, query_points=q, query_normals=qn)
        wins += int((o_mb - o_mu).norm() < (o_pb - o_pu).norm())
    assert wins == 4


def _frame_cloud(n=300, seed=0):
    torch.manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64) + 5.0
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return pts, nrm, drv, w


def _rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_global_frame_contracts():
    """FRAME-FULL: center_mode='global' / scale_mode='global' read the frame from
    forward arguments. Supplying the plain mean and the constant reference length
    reproduces the default model; the frame must be supplied; translation
    equivariance holds with the supplied center translated along; rotation
    equivariance with the center rotated along."""
    pts, nrm, drv, w = _frame_cloud()
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    torch.manual_seed(1)
    m_plain = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    m_c = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, center_mode="global").double().eval()
    torch.manual_seed(1)
    m_cs = ISLA(frame_mode="centered", hidden=64, n_layers=2, n_slices=16, center_mode="global", scale_mode="global").double().eval()
    c = pts.mean(dim=1)  # (1, 3)
    s = torch.full((1,), 8.0, dtype=torch.float64)
    with torch.no_grad():
        base = m_plain(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        assert torch.allclose(m_c(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c), base, atol=1e-12)
        assert torch.allclose(m_cs(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c, frame_scale=s), base, atol=1e-12)
        with pytest.raises(ValueError):
            m_c(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        with pytest.raises(ValueError):
            m_cs(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c)
        c2 = c + torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float64)  # any supplied frame
        a = m_cs(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c2, frame_scale=s * 1.3)
        assert torch.allclose(m_cs(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c2 + shift, frame_scale=s * 1.3), a, atol=1e-10)
        q = _rotation()
        rot = m_cs(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w, frame_center=c2 @ q.T, frame_scale=s * 1.3)
        assert torch.allclose(rot[..., :1], a[..., :1], atol=1e-10)
        assert torch.allclose(rot[..., 1:4], a[..., 1:4] @ q.T, atol=1e-10)
        ### the supplied frame is a live input, not ignored
        assert not torch.allclose(m_cs(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, frame_center=c2, frame_scale=s), base, atol=1e-6)
    with pytest.raises(ValueError):
        ISLA(frame_mode="centered", hidden=64, n_layers=2, n_slices=16, similarity_gauge=True, scale_mode="global")
    with pytest.raises(ValueError):
        ISLA(frame_mode="centered", hidden=64, n_layers=2, n_slices=16, scale_mode="rms")


def test_relative_frame_contracts():
    """RELFRAME: no frame origin. Exact translation invariance without centering
    (1e-12, fp64), rotation equivariance, the reduced feature widths (seeds 5 ->
    1, relational 8 -> 6 in the slice blocks and the passive read blocks),
    geometric scale equivariance under scale_mode='total_measure' (points x s,
    weights x s^2), and the excluded combinations."""
    pts, nrm, drv, w = _frame_cloud()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=torch.float64)
    torch.manual_seed(1)
    m = ISLA(hidden=64, n_layers=2, n_slices=16, frame_mode="relative", query_independent=True,
             n_decoder_layers=1).double().eval()
    torch.manual_seed(1)
    m_tm = ISLA(hidden=64, n_layers=2, n_slices=16, frame_mode="relative", scale_mode="total_measure").double().eval()
    torch.manual_seed(1)
    m_def = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    assert m.embed[0].in_features == 1 and m_def.embed[0].in_features == 5
    assert m.blocks[0].geo_logit.in_features == 6 and m_def.blocks[0].geo_logit.in_features == 8
    assert m.read_blocks[0].geo_logit.in_features == 6
    with torch.no_grad():
        for model in (m, m_tm):
            base = model(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
            assert torch.isfinite(base).all()
            assert torch.allclose(model(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w), base, atol=1e-12)
            q = _rotation()
            rot = model(points=pts @ q.T, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
            assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
            assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)
        base = m_tm(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        assert torch.allclose(m_tm(points=pts * 2.5, normals=nrm, global_vectors=drv, measure_weights=w * 2.5**2), base, atol=1e-10)
        assert not torch.allclose(m_tm(points=pts * 2.5, normals=nrm, global_vectors=drv, measure_weights=w), base, atol=1e-3)  # the total measure IS the scale
        with pytest.raises(ValueError):
            m_tm(points=pts, normals=nrm, global_vectors=drv, measure_weights=None)
    for bad in ({"similarity_gauge": True}, {"odd_head": True}, {"seed_mode": "raw"},
                {"center_mode": "measure"}, {"scale_conditioning": True}):
        with pytest.raises(ValueError):
            ISLA(hidden=64, n_layers=2, n_slices=16, frame_mode="relative", **bad)
    with pytest.raises(ValueError):
        ISLA(hidden=64, n_layers=2, n_slices=16, frame_mode="absolute")


@pytest.mark.parametrize("kw", [{"center_mode": "global", "scale_mode": "global"},
                                {"frame_mode": "relative"},
                                {"frame_mode": "relative", "scale_mode": "total_measure"}])
def test_frame_modes_are_sampling_consistent(kw):
    """The discriminating contract for both frame programs. Under a 10:1 biased
    Poisson subsample with exact HT weights, a model whose frame carries no
    sample statistic (FRAME-FULL: the full-cloud frame supplied; RELFRAME: no
    frame) moves its predictions at fixed queries between the uniform and the
    biased draw by no more than the measure noise -- the same order as between two
    independent uniform draws -- and far less than the plain-centred model."""
    torch.manual_seed(0)
    n_full = 100000
    pts = torch.randn(1, n_full, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n_full, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n_full, dtype=torch.float64) + 0.5
    w_n = w / w.sum()
    c_full = (w_n[..., None] * pts).sum(dim=1)
    s_full = (w_n * (pts - c_full[:, None]).square().sum(-1)).sum().sqrt().reshape(1)
    frame = dict(frame_center=c_full, frame_scale=s_full) if kw.get("center_mode") == "global" else {}
    q, qn = pts[:, :40], nrm[:, :40]
    ### reference_length 1 puts r at the cloud's scale (the plain model must feel
    ### the centroid shift); local_readout_rho 1 keeps the passive readout a smooth
    ### measure-weighted average on this volumetric toy cloud (its surface default,
    ### 0.02, underflows to the clamp here and its noise would swamp the frame).
    common = dict(hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=1,
                  reference_length=1.0, local_readout_rho=1.0)
    torch.manual_seed(1)
    m_plain = ISLA(**{**_CENTERED, **common}).double().eval()
    torch.manual_seed(1)
    m = ISLA(**{**_CENTERED, **common, **kw}).double().eval()
    n_sub = 10000
    d_bias, d_noise, d_plain, d_plain_noise = [], [], [], []
    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        p_u, w_u, iu = _biased_poisson_subsample(pts, w, n_sub, 1.0, g)
        p_u2, w_u2, iu2 = _biased_poisson_subsample(pts, w, n_sub, 1.0, g)
        p_b, w_b, ib = _biased_poisson_subsample(pts, w, n_sub, 10.0, g)
        with torch.no_grad():
            o_pu = m_plain(points=p_u, normals=nrm[:, iu], global_vectors=drv, measure_weights=w_u, query_points=q, query_normals=qn)
            o_pu2 = m_plain(points=p_u2, normals=nrm[:, iu2], global_vectors=drv, measure_weights=w_u2, query_points=q, query_normals=qn)
            o_pb = m_plain(points=p_b, normals=nrm[:, ib], global_vectors=drv, measure_weights=w_b, query_points=q, query_normals=qn)
            o_u = m(points=p_u, normals=nrm[:, iu], global_vectors=drv, measure_weights=w_u, query_points=q, query_normals=qn, **frame)
            o_u2 = m(points=p_u2, normals=nrm[:, iu2], global_vectors=drv, measure_weights=w_u2, query_points=q, query_normals=qn, **frame)
            o_b = m(points=p_b, normals=nrm[:, ib], global_vectors=drv, measure_weights=w_b, query_points=q, query_normals=qn, **frame)
        d_bias.append(float((o_b - o_u).norm()))
        d_noise.append(float((o_u2 - o_u).norm()))
        d_plain.append(float((o_pb - o_pu).norm()))
        d_plain_noise.append(float((o_pu2 - o_pu).norm()))
        assert d_bias[-1] < 0.4 * d_plain[-1], (d_bias, d_plain)
    ### measure-noise level: the biased draw's HT estimates have a larger variance
    ### than a uniform draw's (10x fewer points in the undersampled half), so the
    ### bias-vs-uniform difference may exceed the uniform-vs-uniform one by a
    ### bounded factor -- not the order of magnitude a frame shift produces.
    ratio = sum(d_bias) / sum(d_noise)
    ratio_plain = sum(d_plain) / sum(d_plain_noise)
    assert ratio < 4.0 < ratio_plain, (ratio, ratio_plain)


def test_default_is_relative_total_measure():
    """The class default is the reference configuration decided on 2026-09-11:
    the relative frame (no centroid anywhere; one seed invariant n.d and six
    relational invariants) with the total-measure scale (positions divided by
    the square root of the total quadrature measure). Checkpoints store their
    constructor arguments, so models saved under the earlier defaults are
    unaffected; this test pins the default itself."""
    m = ISLA(hidden=32, n_layers=2, n_slices=8)
    assert m.frame_mode == "relative" and m.scale_mode == "total_measure" and m.relative_frame
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6
    m_c = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8)
    assert m_c.embed[0].in_features == 5 and m_c.blocks[0].geo_logit.in_features == 8


def test_constructor_and_forward_are_keyword_only():
    """Every constructor argument and every forward input is keyword-only
    (ruling 2026-09-11: inputs read as the recipe's forward_kwargs mapping and
    models can be hot-swapped without positional bookkeeping)."""
    pts = torch.randn(1, 16, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 16, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 16) + 0.5
    with pytest.raises(TypeError):
        ISLA(1, 1)
    m = ISLA(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(TypeError):
        m(pts, nrm, drv, w)
    out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    assert out.shape == (1, 16, 4) and torch.isfinite(out).all()
