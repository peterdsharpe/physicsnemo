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
from physicsnemo.experimental.nn.isla.model import _REMOVED_OPTIONS, RESEARCH_TAG

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
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
        )
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
        out = m(
            points=pts,
            normals=nrm,
            global_vectors=drv[:, None, :],
            measure_weights=w[..., None].squeeze(-1),
        )
    assert torch.allclose(out, base, atol=1e-12)


@pytest.fixture
def setup_qi():
    torch.manual_seed(0)
    m = (
        ISLA(
            **_CENTERED,
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_independent=True,
            n_decoder_layers=2,
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
        out_small = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=qa,
            query_normals=na,
        )
        out_big = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=q_big,
            query_normals=n_big,
        )
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
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
        )
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
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, n_boundary_scalars=2)
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
        base = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            boundary_scalars=bs,
        )
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
            boundary_scalars=bs,
        )
    assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)


@pytest.mark.parametrize(
    "kw", [{}, {"frame_mode": "relative", "scale_mode": "total_measure"}]
)
def test_head_runs_under_bf16_autocast(kw):
    """Mixed-precision smoke: the head must survive bf16 autocast in both frames
    (the geometry is kept in the input precision by construction)."""
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
    mg = (
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, similarity_gauge=True)
        .double()
        .eval()
    )
    torch.manual_seed(1)
    m0 = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        a = mg(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        bsc = mg(
            points=k * pts + shift,
            normals=nrm,
            global_vectors=drv,
            measure_weights=k * k * w,
        )
        a0 = m0(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        b0 = m0(
            points=k * pts, normals=nrm, global_vectors=drv, measure_weights=k * k * w
        )
    assert torch.allclose(bsc, a, atol=1e-10)
    assert not torch.allclose(b0, a0, atol=1e-3)


def test_passive_interior_queries_contracts():
    """Passive decoding at off-surface query points: with the query normals
    supplied (e.g. the SDF gradient) the prediction is exactly rotation- and
    translation-covariant, query-set independent and finite; distinct query
    points WITHOUT normals are refused with a clear error."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0  # interior/exterior points
    qnrm = torch.nn.functional.normalize(
        torch.randn(1, nq, 3, dtype=torch.float64), dim=-1
    )
    m = (
        ISLA(
            **_CENTERED,
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_independent=True,
            n_decoder_layers=2,
            similarity_gauge=True,
            out_scalars=1,
            out_vectors=1,
        )
        .double()
        .eval()
    )
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=qpts,
            query_normals=qnrm,
        )
        rot = m(
            points=pts @ q.T + shift,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
            query_points=qpts @ q.T + shift,
            query_normals=qnrm @ q.T,
        )
        sub = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=qpts[:, :50],
            query_normals=qnrm[:, :50],
        )
    assert base.shape == (1, nq, 4) and torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(sub, base[:, :50], atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError, match="pass query_normals"):
        with torch.no_grad():
            m(
                points=pts,
                normals=nrm,
                global_vectors=drv,
                measure_weights=w,
                query_points=qpts,
            )


def test_no_relational_geo_ablation_runs_and_differs():
    """A35b: use_relational_geo=False (Transolver-style feature-only slicing)
    stays exactly equivariant, runs, is finite, and changes the output."""
    torch.manual_seed(0)
    n = 300
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
    torch.manual_seed(1)
    ref = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    nogeo = (
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, use_relational_geo=False)
        .double()
        .eval()
    )
    with torch.no_grad():
        o_ref = ref(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        o_ng = nogeo(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        o_ng_rot = nogeo(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
        )
    assert torch.isfinite(o_ng).all()
    assert not torch.allclose(o_ng, o_ref, atol=1e-6)
    p0, v0 = _split(o_ng)
    p1, v1 = _split(o_ng_rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(
        v1, v0 @ q.T, atol=1e-10
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"use_relational_geo": False},
        {"similarity_gauge": True},
        {"frame_mode": "relative"},
        {
            "frame_mode": "relative",
            "scale_mode": "total_measure",
            "query_independent": True,
            "n_decoder_layers": 1,
        },
    ],
)
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


@pytest.mark.parametrize("extra", [{}, {"query_independent": True}])
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
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 40, 3, dtype=torch.float64), dim=-1
    )
    drive = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    fk = {}
    if extra:
        fk = dict(
            query_points=torch.randn(1, 17, 3, dtype=torch.float64),
            query_normals=torch.nn.functional.normalize(
                torch.randn(1, 17, 3, dtype=torch.float64), dim=-1
            ),
        )
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
    kw = dict(
        out_scalars=1,
        out_vectors=1,
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_tokens=True,
        similarity_gauge=True,
        n_query_scalars=1,
    )
    m = ISLA(**{**_CENTERED, **kw}).double()
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 40, 3, dtype=torch.float64), dim=-1
    )
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64)
    qn = torch.nn.functional.normalize(
        torch.randn(1, 12, 3, dtype=torch.float64), dim=-1
    )
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    out = m(
        points=pts,
        normals=nrm,
        global_vectors=drive,
        measure_weights=w,
        query_points=q,
        query_normals=qn,
        query_scalars=sdf,
    )
    assert out.shape == (1, 12, 4)
    ### scalar channel is live
    out2 = m(
        points=pts,
        normals=nrm,
        global_vectors=drive,
        measure_weights=w,
        query_points=q,
        query_normals=qn,
        query_scalars=sdf * 2,
    )
    assert not torch.allclose(out, out2)
    ### rotation + translation covariance (scalars ride along unchanged)
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(
        points=pts @ R.T + t,
        normals=nrm @ R.T,
        global_vectors=drive @ R.T,
        measure_weights=w,
        query_points=q @ R.T + t,
        query_normals=qn @ R.T,
        query_scalars=sdf,
    )
    assert torch.allclose(rot[..., 0], out[..., 0], atol=1e-10)
    assert torch.allclose(rot[..., 1:], out[..., 1:] @ R.T, atol=1e-10)
    ### geometric-scale equivariance: lengths scale, so must the scalar
    s = 3.7
    sc = m(
        points=pts * s,
        normals=nrm,
        global_vectors=drive,
        measure_weights=w * s**2,
        query_points=q * s,
        query_normals=qn,
        query_scalars=sdf * s,
    )
    assert torch.allclose(sc, out, atol=1e-10)
    ### gradients reach the scalar embedding
    out.square().sum().backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.qt_scalar_embed.parameters()
    )
    with pytest.raises(ValueError):
        m(
            points=pts,
            normals=nrm,
            global_vectors=drive,
            measure_weights=w,
            query_points=q,
            query_normals=qn,
        )
    with pytest.raises(ValueError):
        ISLA(
            **_CENTERED,
            out_scalars=1,
            out_vectors=1,
            hidden=32,
            n_layers=1,
            n_slices=8,
            n_query_scalars=1,
        )


def test_query_tokens_contracts():
    """Query-token mode (interior queries as interacting tokens): exactly
    rotation/translation-covariant, finite, live (differs from the passive
    interior decode), and -- by design -- NOT query-set independent."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    qnrm = torch.nn.functional.normalize(
        torch.randn(1, nq, 3, dtype=torch.float64), dim=-1
    )
    m = (
        ISLA(
            **_CENTERED,
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_tokens=True,
            similarity_gauge=True,
            out_scalars=1,
            out_vectors=1,
        )
        .double()
        .eval()
    )
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=qpts,
            query_normals=qnrm,
        )
        rot = m(
            points=pts @ q.T + shift,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
            query_points=qpts @ q.T + shift,
            query_normals=qnrm @ q.T,
        )
        sub = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            query_points=qpts[:, :50],
            query_normals=qnrm[:, :50],
        )
    assert base.shape == (1, nq, 4) and torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    # interacting mode: predictions at shared queries depend on the query set
    assert not torch.allclose(sub, base[:, :50], atol=1e-6)
    # every parameter receives a gradient (DDP safety)
    m.train()
    out = m(
        points=pts,
        normals=nrm,
        global_vectors=drv,
        measure_weights=w,
        query_points=qpts,
        query_normals=qnrm,
    )
    out.square().mean().backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        ISLA(
            **_CENTERED,
            hidden=32,
            n_layers=1,
            n_slices=8,
            query_tokens=True,
            query_independent=True,
        )


@pytest.mark.parametrize(
    "extra", [{}, {"n_query_scalars": 1, "query_scalar_scale": "length"}]
)
def test_query_mass_source_total_refinement_invariance(extra):
    """Audit 2026-09-08: splitting every source token into two half-weight
    copies leaves the discrete source measure unchanged (positions, normals,
    total area, every integral). With query_mass="source_total" the query
    tokens' weight is a fraction of the total source measure, so the output
    is exactly unchanged and the similarity-gauge scale contract still holds;
    the default "geometric_mean" is the trained convention and is asserted to
    keep its (refinement-dependent) formula so checkpoints reproduce."""
    torch.manual_seed(42)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q, qn = pts[:, :17], nrm[:, :17]
    fk = dict(query_points=q, query_normals=qn)
    if extra:
        fk["query_scalars"] = q.norm(dim=-1)
    kw = dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        similarity_gauge=True,
        query_tokens=True,
        **extra,
    )
    torch.manual_seed(0)
    total = ISLA(query_mass="source_total", **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(0)
    default = ISLA(**{**_CENTERED, **kw}).double().eval()
    refined = dict(
        points=pts.repeat_interleave(2, 1),
        normals=nrm.repeat_interleave(2, 1),
        global_vectors=drv,
        measure_weights=w.repeat_interleave(2, 1) / 2,
    )
    s = 3.7
    scaled_fk = {**fk, "query_points": q * s}
    if extra:
        scaled_fk["query_scalars"] = fk["query_scalars"] * s
    with torch.no_grad():
        a = total(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk)
        b = total(**refined, **fk)
        c = total(
            points=pts * s,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w * s**2,
            **scaled_fk,
        )
        a0 = default(
            points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk
        )
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
        ISLA(
            **_CENTERED,
            hidden=32,
            n_layers=1,
            n_slices=8,
            query_tokens=True,
            query_mass="mean",
        )


@pytest.mark.parametrize(
    "extra", [{}, {"query_independent": True, "n_decoder_layers": 2}]
)
def test_similarity_gauge_scale_equivariance_with_passive_decoder(extra):
    """Under the similarity gauge a geometric rescale (points x k, areas x k^2)
    leaves the output exactly unchanged, for the encoder alone and with the
    passive decoder reading queries that are rescaled along."""
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
    m = (
        ISLA(
            hidden=64,
            n_layers=2,
            n_slices=16,
            similarity_gauge=True,
            **{**_CENTERED, **extra},
        )
        .double()
        .eval()
    )
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    fk = dict(query_points=pts[:, :50], query_normals=nrm[:, :50]) if extra else {}
    fk_sc = {**fk, "query_points": k * fk["query_points"] + shift} if extra else {}
    with torch.no_grad():
        a = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **fk)
        b = m(
            points=k * pts + shift,
            normals=nrm,
            global_vectors=drv,
            measure_weights=k * k * w,
            **fk_sc,
        )
    assert torch.allclose(b, a, atol=1e-10, rtol=0.0)


def test_passive_decode_boundary_scalars():
    """Audit 2026-09-08: boundary scalars are per-boundary-cell data. Passive
    decoding of the boundary itself (query_points=None) carries them into the
    query seeds and runs; distinct query points carry zeros in the channel
    (GLOBAL INPUTS, 2026-09-14) and decode without a shape error."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    bs = torch.randn(1, 60, 2, dtype=torch.float64)
    m = (
        ISLA(
            **_CENTERED,
            hidden=32,
            n_layers=2,
            n_slices=8,
            query_independent=True,
            n_decoder_layers=2,
            n_boundary_scalars=2,
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        out = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            boundary_scalars=bs,
        )
        out2 = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            boundary_scalars=bs * 2,
        )
    assert out.shape == (1, 60, 4) and torch.isfinite(out).all()
    assert not torch.allclose(
        out, out2, atol=1e-6
    )  # the channel is live on the query side
    with torch.no_grad():
        out_q = m(
            points=pts,
            normals=nrm,
            global_vectors=drv,
            measure_weights=w,
            boundary_scalars=bs,
            query_points=pts[:, :17] * 0.5,
            query_normals=nrm[:, :17],
        )
    assert out_q.shape == (1, 17, 4) and torch.isfinite(out_q).all()


def _biased_poisson_subsample(pts, w, n_expected, bias, generator):
    """10:1 Poisson subsample with exact Horvitz-Thompson weights, mirroring the
    recipe's PoissonBiasedSubsampleMesh (bias toward x below the median)."""
    x = pts[0, :, 0]
    b = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    pi = (n_expected / b.sum() * b).clamp(max=1.0)
    keep = torch.rand(x.shape[0], dtype=pi.dtype, generator=generator) < pi
    idx = keep.nonzero(as_tuple=True)[0]
    return pts[:, idx], (w[:, idx] / pi[idx]), idx


def _frame_cloud(n=300, seed=0):
    torch.manual_seed(seed)
    pts = (
        torch.randn(1, n, 3, dtype=torch.float64)
        * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
        + 5.0
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return pts, nrm, drv, w


def _rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_relative_frame_contracts():
    """RELFRAME: no frame origin. Exact translation invariance without centering
    (1e-12, fp64), rotation equivariance, the reduced feature widths (seeds 5 ->
    1, relational 8 -> 6 in the slice blocks and the passive read blocks),
    geometric scale equivariance under scale_mode='total_measure' (points x s,
    weights x s^2), and the excluded combinations."""
    pts, nrm, drv, w = _frame_cloud()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=torch.float64)
    torch.manual_seed(1)
    m = (
        ISLA(
            hidden=64,
            n_layers=2,
            n_slices=16,
            frame_mode="relative",
            query_independent=True,
            n_decoder_layers=1,
        )
        .double()
        .eval()
    )
    torch.manual_seed(1)
    m_tm = (
        ISLA(
            hidden=64,
            n_layers=2,
            n_slices=16,
            frame_mode="relative",
            scale_mode="total_measure",
        )
        .double()
        .eval()
    )
    torch.manual_seed(1)
    m_def = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double().eval()
    assert m.embed[0].in_features == 1 and m_def.embed[0].in_features == 5
    assert (
        m.blocks[0].geo_logit.in_features == 6
        and m_def.blocks[0].geo_logit.in_features == 8
    )
    assert m.read_blocks[0].geo_logit.in_features == 6
    with torch.no_grad():
        for model in (m, m_tm):
            base = model(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
            assert torch.isfinite(base).all()
            assert torch.allclose(
                model(
                    points=pts + shift,
                    normals=nrm,
                    global_vectors=drv,
                    measure_weights=w,
                ),
                base,
                atol=1e-12,
            )
            q = _rotation()
            rot = model(
                points=pts @ q.T,
                normals=nrm @ q.T,
                global_vectors=drv @ q.T,
                measure_weights=w,
            )
            assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
            assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)
        base = m_tm(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        assert torch.allclose(
            m_tm(
                points=pts * 2.5,
                normals=nrm,
                global_vectors=drv,
                measure_weights=w * 2.5**2,
            ),
            base,
            atol=1e-10,
        )
        assert not torch.allclose(
            m_tm(points=pts * 2.5, normals=nrm, global_vectors=drv, measure_weights=w),
            base,
            atol=1e-3,
        )  # the total measure IS the scale
        with pytest.raises(ValueError):
            m_tm(points=pts, normals=nrm, global_vectors=drv, measure_weights=None)
    with pytest.raises(ValueError):
        ISLA(
            hidden=64,
            n_layers=2,
            n_slices=16,
            frame_mode="relative",
            similarity_gauge=True,
        )
    with pytest.raises(ValueError):
        ISLA(hidden=64, n_layers=2, n_slices=16, frame_mode="absolute")
    with pytest.raises(ValueError):
        ISLA(
            hidden=64, n_layers=2, n_slices=16, frame_mode="centered", scale_mode="rms"
        )
    with pytest.raises(ValueError):
        ISLA(
            hidden=64,
            n_layers=2,
            n_slices=16,
            frame_mode="centered",
            similarity_gauge=True,
            scale_mode="total_measure",
        )


@pytest.mark.parametrize(
    "kw",
    [
        {"frame_mode": "relative"},
        {"frame_mode": "relative", "scale_mode": "total_measure"},
    ],
)
def test_relative_frame_is_sampling_consistent(kw):
    """The discriminating contract of the relative frame. Under a 10:1 biased
    Poisson subsample with exact HT weights, a model whose frame carries no
    sample statistic moves its predictions at fixed queries between the uniform
    and the biased draw by no more than the measure noise -- the same order as
    between two independent uniform draws -- and far less than the plain-centred
    model."""
    torch.manual_seed(0)
    n_full = 100000
    pts = torch.randn(1, n_full, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n_full, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n_full, dtype=torch.float64) + 0.5
    q, qn = pts[:, :40], nrm[:, :40]
    ### reference_length 1 puts r at the cloud's scale (the plain model must feel
    ### the centroid shift); local_readout_rho 1 keeps the passive readout a smooth
    ### measure-weighted average on this volumetric toy cloud (its surface default,
    ### 0.02, underflows to the clamp here and its noise would swamp the frame).
    common = dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_independent=True,
        n_decoder_layers=1,
        reference_length=1.0,
        local_readout_rho=1.0,
    )
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
            o_pu = m_plain(
                points=p_u,
                normals=nrm[:, iu],
                global_vectors=drv,
                measure_weights=w_u,
                query_points=q,
                query_normals=qn,
            )
            o_pu2 = m_plain(
                points=p_u2,
                normals=nrm[:, iu2],
                global_vectors=drv,
                measure_weights=w_u2,
                query_points=q,
                query_normals=qn,
            )
            o_pb = m_plain(
                points=p_b,
                normals=nrm[:, ib],
                global_vectors=drv,
                measure_weights=w_b,
                query_points=q,
                query_normals=qn,
            )
            o_u = m(
                points=p_u,
                normals=nrm[:, iu],
                global_vectors=drv,
                measure_weights=w_u,
                query_points=q,
                query_normals=qn,
            )
            o_u2 = m(
                points=p_u2,
                normals=nrm[:, iu2],
                global_vectors=drv,
                measure_weights=w_u2,
                query_points=q,
                query_normals=qn,
            )
            o_b = m(
                points=p_b,
                normals=nrm[:, ib],
                global_vectors=drv,
                measure_weights=w_b,
                query_points=q,
                query_normals=qn,
            )
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
    assert (
        m.frame_mode == "relative"
        and m.scale_mode == "total_measure"
        and m.relative_frame
    )
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


def test_legacy_options_former_defaults_are_accepted():
    """Checkpoints written by the research class recorded every removed option;
    one recorded at its former default describes this very network and must
    build the same model (same parameters, same output), with the legacy name
    dropped from the recorded constructor arguments. Sequence defaults arrive
    as lists after the JSON round trip and are accepted too."""
    pts = torch.randn(1, 16, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 16, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 16) + 0.5
    torch.manual_seed(0)
    ref = ISLA(hidden=32, n_layers=1, n_slices=8).eval()
    all_defaults = dict(_REMOVED_OPTIONS)
    all_defaults["local_radii"] = list(
        all_defaults["local_radii"]
    )  # as a JSON round trip delivers it
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=1, n_slices=8, **all_defaults).eval()
    assert m.state_dict().keys() == ref.state_dict().keys()
    with torch.no_grad():
        assert torch.equal(
            m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
            ref(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
        )
    assert not set(_REMOVED_OPTIONS) & set(m._args["__args__"])
    assert m._args["__args__"]["hidden"] == 32


@pytest.mark.parametrize(
    "kw",
    [
        {"odd_head": True},
        {"center_mode": "measure"},
        {"anchor_topk": 4},
        {"local_radii": (0.5, 1.5)},
        {"routing_logit_scale": 0.5},
        {"scale_mode": "global"},
    ],
)
def test_legacy_options_non_default_values_raise_naming_the_tag(kw):
    """A removed option at a non-default value cannot be reproduced by the lean
    class: the error names the option and the tag where the research class lives."""
    name = next(iter(kw))
    with pytest.raises(ValueError, match=RESEARCH_TAG) as err:
        ISLA(hidden=32, n_layers=1, n_slices=8, **kw)
    assert name in str(err.value)


def test_unknown_constructor_argument_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonsense'"):
        ISLA(hidden=32, n_layers=1, n_slices=8, nonsense=1)


def test_multi_head_routing_keeps_the_contracts():
    """FORM-HEADS: n_heads > 1 routes each token several ways per block; every guarantee of the
    single routing (rotation equivariance, translation invariance, measure-weight scale
    invariance) must hold exactly, and the default n_heads=1 keeps the parameter names."""
    torch.manual_seed(0)
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, n_heads=4).double().eval()
    n = 300
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
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_vectors=drv @ q.T,
            measure_weights=w,
        )
        tr = m(points=pts + shift, normals=nrm, global_vectors=drv, measure_weights=w)
        ws = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w * 137.0)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.isfinite(base).all()
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(
        v1, v0 @ q.T, atol=1e-10
    )
    assert torch.allclose(tr, base, atol=1e-10)
    assert torch.allclose(ws, base, atol=1e-9)
    single = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16).double()
    assert single.n_heads == 1
    assert set(single.state_dict()) == set(
        m.state_dict()
    )  # same parameter names; shapes differ
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, n_heads=5)
