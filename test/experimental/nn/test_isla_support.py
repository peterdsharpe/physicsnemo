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

"""Contract tests for ISLA's support-token mode (transfer program D1, 2026-09-09):
a problem-derived interior SUPPORT set interacts in the encoder, the requested
queries are decoded passively (with their SDF scalar) and therefore cannot
depend on one another."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA

# The contract tests below were written for the centered construction (plain-mean
# centre, constant reference_length), which was the class default until 2026-09-11.
# The class default is now the relative frame with the total-measure scale (the
# reference configuration); these tests pin the centered construction explicitly so
# they keep verifying it, and test_default_is_relative_total_measure covers the default.
_CENTERED = dict(frame_mode="centered", scale_mode="reference_length")


D = torch.float64


def _cloud(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=D, generator=g) * torch.tensor([3.0, 2.0, 1.0], dtype=D)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=D, generator=g), dim=-1)
    w = torch.rand(1, n, dtype=D, generator=g) + 0.5
    return pts, nrm, w


def _interior(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, n, 3, dtype=D, generator=g) * 4.0
    qn = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=D, generator=g), dim=-1)
    qs = q.norm(dim=-1, keepdim=True) * 0.3 + 0.05  # a positive "signed distance"
    return q, qn, qs


def _model(**kw):
    torch.manual_seed(0)
    return ISLA(hidden=64, n_layers=2, n_slices=16, query_independent=True, n_decoder_layers=3,
                support_tokens=True, n_query_scalars=1, query_mass="source_total", **{**_CENTERED, **kw}).double().eval()


@pytest.fixture
def setup():
    m = _model()
    pts, nrm, w = _cloud(300)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=D, generator=torch.Generator().manual_seed(3)), dim=-1)
    sp, sn, ss = _interior(120, seed=1)
    q, qn, qs = _interior(80, seed=2)
    return m, pts, nrm, drv, w, (sp, sn, ss), (q, qn, qs)


def _run(m, pts, nrm, drv, w, S, Q):
    sp, sn, ss = S
    q, qn, qs = Q
    return m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q, query_normals=qn, query_scalars=qs,
             support_points=sp, support_normals=sn, support_scalars=ss)


def test_query_independence_with_fixed_support(setup):
    """The contract: with surface and support fixed, a query's prediction does not
    depend on which other queries are requested (exact up to GEMM reordering)."""
    m, pts, nrm, drv, w, S, Q = setup
    q, qn, qs = Q
    with torch.no_grad():
        alone = _run(m, pts, nrm, drv, w, S, (q[:, :1], qn[:, :1], qs[:, :1]))
        small = _run(m, pts, nrm, drv, w, S, (q[:, :20], qn[:, :20], qs[:, :20]))
        big = _run(m, pts, nrm, drv, w, S, Q)
        other = _interior(500, seed=9)
        mixed = _run(m, pts, nrm, drv, w, S, (torch.cat([q[:, :20], other[0]], 1), torch.cat([qn[:, :20], other[1]], 1),
                                              torch.cat([qs[:, :20], other[2]], 1)))
    assert torch.allclose(alone, big[:, :1], atol=1e-12, rtol=0.0)
    assert torch.allclose(small, big[:, :20], atol=1e-12, rtol=0.0)
    assert torch.allclose(small, mixed[:, :20], atol=1e-12, rtol=0.0)


def test_support_dependence_is_real(setup):
    """Documented (not a defect): the support set defines the computational state,
    so changing it changes predictions. Contrast with the query independence above."""
    m, pts, nrm, drv, w, S, Q = setup
    with torch.no_grad():
        a = _run(m, pts, nrm, drv, w, S, Q)
        b = _run(m, pts, nrm, drv, w, _interior(120, seed=7), Q)
    assert not torch.allclose(a, b, atol=1e-6)


def test_se3_covariance_and_global_vector_magnitude(setup):
    m, pts, nrm, drv, w, S, Q = setup
    sp, sn, ss = S
    q, qn, qs = Q
    R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(5)))
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([2.0, -5.0, 1.0], dtype=D)
    with torch.no_grad():
        base = _run(m, pts, nrm, drv, w, S, Q)
        moved = _run(m, pts @ R.T + t, nrm @ R.T, drv @ R.T, w, (sp @ R.T + t, sn @ R.T, ss), (q @ R.T + t, qn @ R.T, qs))
        scaled_drive = _run(m, pts, nrm, 2.5 * drv, w, S, Q)
    p0, v0 = base[..., :1], base[..., 1:4]
    p1, v1 = moved[..., :1], moved[..., 1:4]
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ R.T, atol=1e-10)
    assert torch.allclose(scaled_drive, base, atol=1e-10)  # unit direction inside; magnitude has no effect


def test_measure_scale_and_refinement_invariance(setup):
    """Multiplying every source weight by a constant, or splitting every source
    token into two half-weight copies (the same discrete measure), leaves the
    output unchanged: support mass is source_total, routing is quadrature-mean,
    the kernel readout is measure-weighted."""
    m, pts, nrm, drv, w, S, Q = setup
    with torch.no_grad():
        base = _run(m, pts, nrm, drv, w, S, Q)
        scaled = _run(m, pts, nrm, drv, 4.2 * w, S, Q)
        split = _run(m, pts.repeat_interleave(2, 1), nrm.repeat_interleave(2, 1), drv, w.repeat_interleave(2, 1) / 2, S, Q)
    assert torch.allclose(scaled, base, atol=1e-10)
    assert torch.allclose(split, base, atol=1e-10)


def test_similarity_gauge_scale_equivariance():
    m = _model(similarity_gauge=True)
    pts, nrm, w = _cloud(300)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=D, generator=torch.Generator().manual_seed(3)), dim=-1)
    S, Q = _interior(120, 1), _interior(80, 2)
    k = 2.7
    with torch.no_grad():
        base = _run(m, pts, nrm, drv, w, S, Q)
        scaled = _run(m, k * pts, nrm, drv, k**2 * w, (k * S[0], S[1], k * S[2]), (k * Q[0], Q[1], k * Q[2]))
    assert torch.allclose(scaled, base, atol=1e-10)


def test_geo_checkpoint_is_exact_with_support(setup):
    m, pts, nrm, drv, w, S, Q = setup
    torch.manual_seed(0)
    m2 = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, query_independent=True, n_decoder_layers=3,
              support_tokens=True, n_query_scalars=1, query_mass="source_total", geo_checkpoint=True).double().eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        a = _run(m, pts, nrm, drv, w, S, Q)
        b = _run(m2, pts, nrm, drv, w, S, Q)
    assert torch.equal(a, b)


def test_all_parameters_receive_gradients(setup):
    m, pts, nrm, drv, w, S, Q = setup
    m.train()
    out = _run(m, pts, nrm, drv, w, S, Q)
    out.square().mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_decoder_depth_is_configurable():
    torch.manual_seed(0)
    m12 = ISLA(**_CENTERED, hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=12,
               support_tokens=True, n_query_scalars=1, query_mass="source_total")
    assert len(m12.read_blocks) == 12
    n_read = sum(p.numel() for p in m12.read_blocks.parameters())
    n_enc = sum(p.numel() for p in m12.blocks.parameters())
    assert n_read > n_enc  # 12 read blocks against 2 encoder blocks at this size


def test_option_validation():
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, support_tokens=True)  # needs query_independent
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=8, support_tokens=True, query_independent=True, query_tokens=True)
    m = _model()
    pts, nrm, w = _cloud(50)
    drv = torch.tensor([[0.0, 0.0, 1.0]], dtype=D)
    S, Q = _interior(10, 1), _interior(5, 2)
    with pytest.raises(ValueError):  # support without scalars while n_query_scalars > 0
        m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=Q[0], query_normals=Q[1], query_scalars=Q[2],
          support_points=S[0], support_normals=S[1])


def test_passive_queries_take_scalars_without_support():
    """The read path accepts the query SDF scalar on its own (n_query_scalars with
    query_independent=True), which the earlier passive arm could not."""
    torch.manual_seed(0)
    m = ISLA(**_CENTERED, hidden=64, n_layers=2, n_slices=16, query_independent=True, n_decoder_layers=2, n_query_scalars=1).double().eval()
    pts, nrm, w = _cloud(200)
    drv = torch.tensor([[0.0, 0.0, 1.0]], dtype=D)
    q, qn, qs = _interior(30, 2)
    with torch.no_grad():
        a = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q, query_normals=qn, query_scalars=qs)
        b = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, query_points=q, query_normals=qn, query_scalars=2 * qs)
    assert a.shape == (1, 30, 4) and not torch.allclose(a, b, atol=1e-6)
