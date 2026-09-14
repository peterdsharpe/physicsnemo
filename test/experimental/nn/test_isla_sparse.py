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

"""Contract tests for ISLA's sparse anchor routing (``anchor_topk``, SPARSE 2026-09-09)."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA

# The contract tests below were written for the centered construction (plain-mean
# centre, constant reference_length), which was the class default until 2026-09-11.
# The class default is now the relative frame with the total-measure scale (the
# reference configuration); these tests pin the centered construction explicitly so
# they keep verifying it, and test_default_is_relative_total_measure covers the default.
_CENTERED = dict(frame_mode="centered", scale_mode="reference_length")



def _cloud(n=300, seed=1):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, generator=g, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, generator=g, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, generator=g, dtype=torch.float64) + 0.5
    return pts, nrm, drv, w


def _split(o):
    return o[..., :1], o[..., 1:4]


@pytest.mark.parametrize(
    "extra",
    [{}, {"geo_checkpoint": True}, {"second_moment_features": True}, {"similarity_gauge": True}],
)
def test_anchor_topk_full_is_exact(extra):
    """At k = n_slices the sparse routing reproduces the dense model to roundoff."""
    pts, nrm, drv, w = _cloud()
    torch.manual_seed(0)
    dense = ISLA(hidden=64, n_layers=3, n_slices=32, **{**_CENTERED, **extra}).double().eval()
    sparse = ISLA(hidden=64, n_layers=3, n_slices=32, anchor_topk=32, **{**_CENTERED, **extra}).double().eval()
    sparse.load_state_dict(dense.state_dict())
    with torch.no_grad():
        a, b = dense(points=pts, normals=nrm, global_vectors=drv, measure_weights=w), sparse(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    assert torch.allclose(a, b, atol=1e-10, rtol=0)


@pytest.mark.parametrize("extra", [{}, {"similarity_gauge": True}, {"geo_checkpoint": True}])
def test_anchor_topk_contracts(extra):
    """k < n_slices: exact SE(3) covariance, global-vector magnitude invariance, measure-scale invariance,
    gauge scale equivariance; the routing is genuinely sparse and gradients flow."""
    pts, nrm, drv, w = _cloud()
    torch.manual_seed(0)
    m = ISLA(hidden=64, n_layers=3, n_slices=32, anchor_topk=8, **{**_CENTERED, **extra}).double().eval()
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
    torch.manual_seed(0)
    dense = ISLA(hidden=64, n_layers=3, n_slices=32, **{**_CENTERED, **extra}).double().eval()
    with torch.no_grad():
        ref = dense(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    assert not torch.allclose(ref, base, atol=1e-6)  # k = 8 of 32 anchors changes the output
    m.train()
    out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    out.square().mean().backward()
    assert all(p.grad is not None for name, p in m.named_parameters() if "geo_" in name)


def test_anchor_topk_validation():
    with pytest.raises(ValueError):
        ISLA(**_CENTERED, hidden=32, n_layers=1, n_slices=16, anchor_topk=17)
