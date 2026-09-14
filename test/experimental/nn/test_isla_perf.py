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

"""ISLA-PERF: the fast point softmax (reduction along a contiguous last dimension) is
the same arithmetic as the original middle-dimension softmax. Exactness is checked in
float64 for every configuration that owns a point softmax, and the float32 difference
is bounded at roundoff."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA
from physicsnemo.experimental.nn.isla.model import _softmax_over_points

# The contract tests below were written for the centered construction (plain-mean
# centre, constant reference_length), which was the class default until 2026-09-11.
# The class default is now the relative frame with the total-measure scale (the
# reference configuration); these tests pin the centered construction explicitly so
# they keep verifying it, and test_default_is_relative_total_measure covers the default.
_CENTERED = dict(frame_mode="centered", scale_mode="reference_length")



def _cloud(n=300, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    pts = (torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])).to(dtype)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1).to(dtype)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, generator=g), dim=-1).to(dtype)
    w = (torch.rand(1, n, generator=g) + 0.5).to(dtype)
    return pts, nrm, drv, w


def test_softmax_over_points_matches_middle_dim_softmax():
    x = torch.randn(2, 500, 64, dtype=torch.float64)
    assert torch.allclose(_softmax_over_points(x, fast=True), torch.softmax(x, dim=1), atol=1e-14)
    assert torch.equal(_softmax_over_points(x, fast=False), torch.softmax(x, dim=1))
    x32 = x.float()
    rel = ((_softmax_over_points(x32) - torch.softmax(x32, dim=1)).norm() / torch.softmax(x32, dim=1).norm()).item()
    assert rel < 1e-6


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"similarity_gauge": True, "geo_checkpoint": True},
        {"query_independent": True, "n_decoder_layers": 2},
        {"query_independent": True, "n_decoder_layers": 2, "latent_volume_tokens": True},
        {"odd_head": True},
        {"query_tokens": True},
    ],
)
def test_fast_point_softmax_is_exact_in_float64(kw):
    """Every configuration with a point softmax: fast and native paths agree to roundoff
    in float64 with identical weights and inputs."""
    pts, nrm, drv, w = _cloud()
    torch.manual_seed(0)
    fast = ISLA(hidden=64, n_layers=3, n_slices=32, fast_point_softmax=True, **{**_CENTERED, **kw}).double().eval()
    torch.manual_seed(0)
    native = ISLA(hidden=64, n_layers=3, n_slices=32, fast_point_softmax=False, **{**_CENTERED, **kw}).double().eval()
    native.load_state_dict(fast.state_dict())
    extra = {}
    if kw.get("query_tokens"):
        extra = dict(query_points=pts[:, :50], query_normals=nrm[:, :50])
    elif kw.get("query_independent"):
        extra = dict(query_points=pts[:, :50] + 0.3, query_normals=nrm[:, :50])
    with torch.no_grad():
        a = fast(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **extra)
        b = native(points=pts, normals=nrm, global_vectors=drv, measure_weights=w, **extra)
    assert torch.allclose(a, b, atol=1e-11), float((a - b).abs().max())


def test_fast_point_softmax_float32_difference_is_roundoff():
    pts, nrm, drv, w = _cloud(dtype=torch.float32)
    torch.manual_seed(0)
    fast = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32, fast_point_softmax=True).eval()
    torch.manual_seed(0)
    native = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32, fast_point_softmax=False).eval()
    native.load_state_dict(fast.state_dict())
    with torch.no_grad():
        a, b = fast(points=pts, normals=nrm, global_vectors=drv, measure_weights=w), native(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    assert ((a - b).norm() / b.norm()).item() < 1e-5


def test_fast_point_softmax_keeps_contracts():
    """SE(3) covariance and measure-scale invariance hold on the fast path."""
    pts, nrm, drv, w = _cloud()
    torch.manual_seed(0)
    m = ISLA(**_CENTERED, hidden=64, n_layers=3, n_slices=32, fast_point_softmax=True).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
        moved = m(points=pts @ q.T + 1.5, normals=nrm @ q.T, global_vectors=drv @ q.T, measure_weights=w)
        scaled_w = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=2.5 * w)
    assert torch.allclose(moved[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(moved[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)
    assert torch.allclose(scaled_w, base, atol=1e-10)
