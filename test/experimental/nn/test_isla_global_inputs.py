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

"""ISLA global inputs (2026-09-14): zero or more global vector inputs and zero
or more global scalar inputs. The contracts tested here are exact (float64,
tolerance 1e-10): SE(3) covariance and measure-scale invariance for K = 0 and
K = 2 vectors, magnitude invariance of every vector, scalar conditioning that
changes the output while preserving every covariance contract, the K = 1
parameter layout of the former single-vector model, and the one-vector-only
variants refusing K != 1."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA

D = torch.float64
_CENTERED = dict(frame_mode="centered", scale_mode="reference_length")


def _rotation(seed: int):
    R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(seed)))
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R


def _sample(b=2, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(b, n, 3, dtype=D, generator=g)
    nrm = torch.nn.functional.normalize(torch.randn(b, n, 3, dtype=D, generator=g), dim=-1)
    w = torch.rand(b, n, dtype=D, generator=g) + 0.1
    return pts, nrm, w


def _model(**kw):
    torch.manual_seed(0)
    return ISLA(hidden=32, n_layers=2, n_slices=8, out_scalars=1, out_vectors=1, **kw).to(D).eval()


def _assert_se3(m, pts, nrm, w, gv, gs, extra=None, atol=1e-10):
    extra = extra or {}
    R = _rotation(5)
    t = torch.tensor([2.0, -5.0, 1.0], dtype=D)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv, global_scalars=gs, **extra)
        moved_extra = {
            k: (v @ R.T + t if k.endswith("points") else (v @ R.T if k.endswith("normals") else v))
            for k, v in extra.items()
        }  # scalars (query_scalars, boundary_scalars) are invariants and pass through unchanged
        moved = m(
            points=pts @ R.T + t, normals=nrm @ R.T, measure_weights=w,
            global_vectors=None if gv is None else gv @ R.T, global_scalars=gs, **moved_extra,
        )
    assert torch.allclose(moved[..., :1], base[..., :1], atol=atol)
    assert torch.allclose(moved[..., 1:4], base[..., 1:4] @ R.T, atol=atol)
    if m.scale_mode != "total_measure":
        ### Measure-scale invariance holds where the length unit is constant; the
        ### reference configuration's unit is sqrt(total measure), so there the
        ### weights carry physical area and a rescale is a different geometry.
        with torch.no_grad():
            rescaled = m(points=pts, normals=nrm, measure_weights=137.0 * w, global_vectors=gv, global_scalars=gs, **extra)
        assert torch.allclose(rescaled, base, atol=1e-9)
    return base


@pytest.mark.parametrize("frame", ["relative", "centered"])
def test_zero_global_vectors(frame):
    """K = 0: no direction anywhere in the problem (steady conduction with pinned
    boundary values). The seed is constant in the relative frame and the model is
    still exactly SE(3)-covariant and measure-scale invariant."""
    kw = {} if frame == "relative" else _CENTERED
    m = _model(n_global_vectors=0, **kw)
    pts, nrm, w = _sample()
    if frame == "relative":
        assert m.constant_seed
        assert m.embed[0].in_features == 1
    else:
        assert m.embed[0].in_features == 3  # |r|, log|r|, rhat.n
    assert m.n_basis == 4  # n, u, and the two complements of (u, n)
    base = _assert_se3(m, pts, nrm, w, None, None)
    assert torch.isfinite(base).all()
    with pytest.raises(ValueError, match="n_global_vectors=0"):
        m(points=pts, normals=nrm, measure_weights=w, global_vectors=torch.randn(2, 1, 3, dtype=D))


@pytest.mark.parametrize("frame", ["relative", "centered"])
def test_two_global_vectors(frame):
    """K = 2 (a freestream and a gravity direction): both vectors rotate with the
    geometry, the output rotates with them, and rescaling either vector's magnitude
    changes nothing."""
    kw = {} if frame == "relative" else _CENTERED
    m = _model(n_global_vectors=2, **kw)
    pts, nrm, w = _sample()
    gv = torch.randn(2, 2, 3, dtype=D)
    assert m.embed[0].in_features == (2 if frame == "relative" else 7)
    assert m.n_basis == 10
    base = _assert_se3(m, pts, nrm, w, gv, None)
    with torch.no_grad():
        scaled = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv * torch.tensor([[[3.0], [0.2]]], dtype=D))
        swapped = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv.flip(1))
        shared = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv[:1])
    assert torch.allclose(scaled, base, atol=1e-10)
    assert not torch.allclose(swapped, base, atol=1e-3)  # the two inputs are distinct channels
    assert torch.allclose(shared[:1], base[:1], atol=1e-10)  # a single set is shared across the batch
    with pytest.raises(ValueError, match="multiple of 2 vectors"):
        m(points=pts, normals=nrm, measure_weights=w, global_vectors=torch.randn(5, 3, dtype=D))


def test_two_vectors_passive_and_query_token_paths():
    """The interior paths carry the K vectors to the query side as well."""
    pts, nrm, w = _sample()
    q = torch.randn(2, 40, 3, dtype=D)
    qn = torch.nn.functional.normalize(torch.randn(2, 40, 3, dtype=D), dim=-1)
    gv = torch.randn(2, 2, 3, dtype=D)
    m_passive = _model(n_global_vectors=2, query_independent=True, interior_queries=True, n_decoder_layers=1)
    _assert_se3(m_passive, pts, nrm, w, gv, None, extra=dict(query_points=q))
    m_qt = _model(n_global_vectors=2, query_tokens=True, query_mass="source_total")
    _assert_se3(m_qt, pts, nrm, w, gv, None, extra=dict(query_points=q, query_normals=qn))
    ### the neighbour channel searches neighbours in float32 (pre-existing), so its
    ### covariance holds to ~1e-7 at any K; the K-wide layout is what is tested here
    m_nbr = _model(n_global_vectors=2, query_tokens=True, query_mass="source_total", query_neighbor_features=True)
    assert m_nbr.qt_neighbor_embed[0].in_features == 4 + 2 * 2 + 2
    _assert_se3(m_nbr, pts, nrm, w, gv, None, extra=dict(query_points=q, query_normals=qn), atol=1e-5)


def test_global_scalars_condition_every_token():
    """S = 2 scalar inputs (a diffusivity and a modulus, say) change the output,
    reach the query side too, and leave every covariance contract intact."""
    pts, nrm, w = _sample()
    m = _model(n_global_scalars=2)
    assert m.embed[0].in_features == 1 + 2  # n.g plus the two scalars
    gv = torch.randn(2, 1, 3, dtype=D)
    gs = torch.tensor([[0.3, -1.2], [2.0, 0.1]], dtype=D)
    base = _assert_se3(m, pts, nrm, w, gv, gs)
    with torch.no_grad():
        other = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv, global_scalars=gs + 1.0)
        shared = m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv, global_scalars=gs[:1])
    assert not torch.allclose(other, base, atol=1e-3)
    assert torch.allclose(shared[:1], base[:1], atol=1e-10)
    with pytest.raises(ValueError, match="needs global_scalars"):
        m(points=pts, normals=nrm, measure_weights=w, global_vectors=gv)
    m0 = _model()
    with pytest.raises(ValueError, match="n_global_scalars=0"):
        m0(points=pts, normals=nrm, measure_weights=w, global_vectors=gv, global_scalars=gs)
    ### the passive and interacting query paths take the scalars as well
    q = torch.randn(2, 30, 3, dtype=D)
    qn = torch.nn.functional.normalize(torch.randn(2, 30, 3, dtype=D), dim=-1)
    m_passive = _model(n_global_scalars=2, query_independent=True, interior_queries=True, n_decoder_layers=1)
    _assert_se3(m_passive, pts, nrm, w, gv, gs, extra=dict(query_points=q))
    m_qt = _model(n_global_scalars=2, query_tokens=True, query_mass="source_total")
    _assert_se3(m_qt, pts, nrm, w, gv, gs, extra=dict(query_points=q, query_normals=qn))


def test_no_global_inputs_at_all_with_scalars_only():
    """K = 0 with S = 1: a parameterized PDE family with no preferred direction."""
    pts, nrm, w = _sample()
    m = _model(n_global_vectors=0, n_global_scalars=1)
    assert not m.constant_seed and m.embed[0].in_features == 1
    gs = torch.tensor([[0.5], [1.5]], dtype=D)
    base = _assert_se3(m, pts, nrm, w, None, gs)
    with torch.no_grad():
        other = m(points=pts, normals=nrm, measure_weights=w, global_scalars=2.0 * gs)
    assert not torch.allclose(other, base, atol=1e-3)


def test_one_vector_layout_matches_the_former_single_vector_model():
    """K = 1, S = 0 (the defaults) must keep the former parameter layout so that
    every saved checkpoint loads: seed width 1 (relative) / 5 (centered), geometry
    width 6 / 8, head basis 7 (globe7), 5 (true5), 7 (true7)."""
    m = _model()
    assert (m.n_global_vectors, m.n_global_scalars) == (1, 0)
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6 and m.n_basis == 7
    mc = _model(**_CENTERED)
    assert mc.embed[0].in_features == 5 and mc.blocks[0].geo_logit.in_features == 8
    assert _model(vector_basis="true5").n_basis == 5
    assert _model(vector_basis="true7").n_basis == 7
    assert _model(second_moment_features=True).blocks[0].geo_logit.in_features == 8
    assert _model(use_local_features=True, **_CENTERED).embed[0].in_features == 5 + 7 * 2


def test_legacy_vector_layouts_still_accepted():
    """For K = 1 the recipe delivers (3,), (B, 3) or (B, 1, 3); all are one vector."""
    pts, nrm, w = _sample()
    m = _model()
    d = torch.tensor([1.0, 0.3, 0.0], dtype=D)
    with torch.no_grad():
        a = m(points=pts, normals=nrm, measure_weights=w, global_vectors=d)
        b = m(points=pts, normals=nrm, measure_weights=w, global_vectors=d[None].expand(2, 3))
        c = m(points=pts, normals=nrm, measure_weights=w, global_vectors=d[None, None].expand(2, 1, 3))
    assert torch.equal(a, b) and torch.equal(a, c)
    with pytest.raises(ValueError, match="needs global_vectors"):
        m(points=pts, normals=nrm, measure_weights=w)


@pytest.mark.parametrize(
    "kw",
    [dict(geo_kernel="fused"), dict(odd_head=True, **_CENTERED), dict(parity_fix=True, **_CENTERED),
     dict(latent_volume_tokens=True, query_independent=True), dict(wake_tokens=True, query_independent=True)],
)
def test_one_vector_only_variants_refuse_other_counts(kw):
    for k in (0, 2):
        with pytest.raises(ValueError, match="exactly one global vector"):
            ISLA(hidden=32, n_layers=1, n_slices=8, n_global_vectors=k, **kw)


def test_negative_counts_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ISLA(hidden=32, n_layers=1, n_slices=8, n_global_vectors=-1)


def test_boundary_data_reaches_every_interior_path():
    """A boundary-value problem needs the per-cell boundary data (Dirichlet trace)
    on the boundary tokens and zeros on every token that is not a boundary cell.
    All three interior paths accept it, the covariance contracts hold, and the
    interior prediction depends on the boundary data."""
    pts, nrm, w = _sample()
    bsc = torch.randn(2, pts.shape[1], 1, dtype=D)
    q = torch.randn(2, 30, 3, dtype=D) * 0.5
    qn = torch.nn.functional.normalize(torch.randn(2, 30, 3, dtype=D), dim=-1)
    gs = torch.tensor([[0.7], [2.0]], dtype=D)
    arms = [
        (_model(n_global_vectors=0, n_global_scalars=1, n_boundary_scalars=1, query_tokens=True, query_mass="source_total"),
         dict(query_points=q, query_normals=qn)),
        (_model(n_global_vectors=0, n_global_scalars=1, n_boundary_scalars=1, query_independent=True, interior_queries=True, n_decoder_layers=1),
         dict(query_points=q)),
        (_model(n_global_vectors=0, n_global_scalars=1, n_boundary_scalars=1, query_independent=True, support_tokens=True, query_mass="source_total", n_decoder_layers=1),
         dict(query_points=q, query_normals=qn, support_points=q, support_normals=qn)),
    ]
    for m, extra in arms:
        assert m.embed[0].in_features == 2  # boundary scalar + global scalar (no vectors, relative frame)
        base = _assert_se3(m, pts, nrm, w, None, gs, extra=dict(extra, boundary_scalars=bsc))
        with torch.no_grad():
            other = m(points=pts, normals=nrm, measure_weights=w, global_scalars=gs, boundary_scalars=bsc + 1.0, **extra)
        assert base.shape[1] == 30 and not torch.allclose(other, base, atol=1e-3)
        continue
        assert base.shape[1] == 30 and not torch.allclose(other, base, atol=1e-3)


def test_routing_logit_scale_default_is_identity_and_softens_below_one():
    """routing_logit_scale=1.0 is the trained model bitwise; a value below one keeps
    the token->slice routing softer (higher entropy) and preserves every contract."""
    import math
    import physicsnemo.experimental.nn.isla.model as M
    pts, nrm, w = _sample()
    gv = torch.randn(2, 1, 3, dtype=D)
    m1 = _model(); m1b = _model(routing_logit_scale=1.0); m_soft = _model(routing_logit_scale=0.25)
    m_soft.load_state_dict(m1.state_dict())
    with torch.no_grad():
        o1 = m1(points=pts, normals=nrm, measure_weights=w, global_vectors=gv)
        o1b = m1b(points=pts, normals=nrm, measure_weights=w, global_vectors=gv)
    assert torch.equal(o1, o1b)
    ents = {}
    orig = M._SliceBlock.forward
    def spy(self, h, log_w, r, n_hat, g_hat, eps):
        logits = self.assign(self.norm_assign(h)) * self.routing_logit_scale
        mix = torch.softmax(logits, dim=-1)
        ents.setdefault(self.routing_logit_scale, []).append(-(mix * (mix + 1e-12).log()).sum(-1).mean().item() / math.log(mix.shape[-1]))
        return orig(self, h, log_w, r, n_hat, g_hat, eps)
    M._SliceBlock.forward = spy
    try:
        with torch.no_grad():
            m1(points=pts, normals=nrm, measure_weights=w, global_vectors=gv)
            o_soft = m_soft(points=pts, normals=nrm, measure_weights=w, global_vectors=gv)
    finally:
        M._SliceBlock.forward = orig
    assert sum(ents[0.25]) / len(ents[0.25]) > sum(ents[1.0]) / len(ents[1.0])
    assert not torch.allclose(o_soft, o1, atol=1e-6)
    _assert_se3(m_soft, pts, nrm, w, gv, None)
    with pytest.raises(ValueError, match="positive"):
        ISLA(hidden=32, n_layers=1, n_slices=8, routing_logit_scale=0.0)
