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

"""KERNEL STUDY (2026-09-13): the implementation options of ISLA's geometry region
compute the same thing as the eager reference (geo_kernel="eager", geo_checkpoint=True,
fast_point_softmax=True).

Bars from the study brief, applied to the reference configuration's arithmetic:
the fused Triton region reproduces the eager forward to 1e-6 relative in float32 and
the parameter gradients to 1e-5 relative; in float64 the two agree to 1e-11 (same
arithmetic, different summation order). The same bars hold for the whole model
compiled with torch.compile, which is the recipe's default (compile: true) and must
stay free of graph breaks."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA
from physicsnemo.experimental.nn.isla.model import _geo_region

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")

REF = dict(
    hidden=64,
    n_layers=3,
    n_slices=32,
    out_scalars=1,
    out_vectors=1,
    geo_checkpoint=True,
    fast_point_softmax=True,
)


def _cloud(n=700, batch=1, seed=0, device="cuda", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    pts = (torch.randn(batch, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])).to(
        device, dtype
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(batch, n, 3, generator=g), dim=-1
    ).to(device, dtype)
    drv = torch.nn.functional.normalize(torch.randn(batch, 3, generator=g), dim=-1).to(
        device, dtype
    )
    w = (torch.rand(batch, n, generator=g) + 0.5).to(device, dtype)
    tgt = torch.randn(batch, n, 4, generator=g).to(device, dtype)
    return dict(points=pts, normals=nrm, global_vectors=drv, measure_weights=w), tgt


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _run(model, inputs, target):
    model.zero_grad(set_to_none=True)
    out = model(**inputs)
    torch.nn.functional.mse_loss(out.float(), target.float()).backward()
    grads = {
        k.replace("_orig_mod.", ""): p.grad.detach().clone()
        for k, p in model.named_parameters()
    }
    return out.detach(), grads


def _pair(dtype, device="cuda", **kw):
    torch.manual_seed(0)
    ref = ISLA(**REF).to(device, dtype)
    torch.manual_seed(0)
    opt = ISLA(**{**REF, **kw}).to(device, dtype)
    opt.load_state_dict(ref.state_dict())
    return ref, opt


def _assert_same(ref, opt, dtype, out_tol, grad_tol, n=700, batch=1):
    inputs, tgt = _cloud(n, batch, dtype=dtype)
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert _rel(o_opt, o_ref) < out_tol, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue  # geo_logit.bias: a constant over slices cancels in both softmaxes; its gradient is roundoff
        assert _rel(g_opt[k], g) < grad_tol, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- fused region
@cuda_only
@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fused_geo_region_matches_eager_region(relative, dtype):
    """Region level: forward outputs (bias, mix, pooled) and every input gradient."""
    from physicsnemo.experimental.nn.isla.geo_kernel import fused_geo_region

    torch.backends.cuda.matmul.allow_tf32 = False
    b, n, s = 2, 500, 40
    g = torch.Generator().manual_seed(0)

    def mk(*shape):
        return (
            (torch.randn(*shape, generator=g) * 0.1).to("cuda", dtype).requires_grad_()
        )

    r, z = mk(b, n, 3), mk(b, s, 3)
    nh = (
        torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    ms = (
        torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    d = (
        torch.nn.functional.normalize(torch.randn(b, 1, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    logits = torch.randn(b, n, s, generator=g).to("cuda", dtype).requires_grad_()
    torch.manual_seed(0)
    lin = torch.nn.Linear(6 if relative else 8, 1).to("cuda", dtype)
    leaves = (lin.weight, lin.bias, logits, r, nh, d, z, ms)
    wts = [
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, lin.in_features, generator=g).to("cuda", dtype),
    ]

    def run(fn, *extra):
        for t in leaves:
            t.grad = None
        d_n = d.expand(b, n, 3)
        ### the eager region takes the K global vectors as (B, N, K, 3); the fused kernel is single-vector (B, N, 3)
        outs = fn(
            lin,
            logits,
            r,
            nh,
            d_n[:, :, None] if fn is _geo_region else d_n,
            z,
            ms,
            1e-12,
            *extra,
        )
        sum((o * w).sum() for o, w in zip(outs, wts)).backward()
        return [o.detach() for o in outs], [t.grad.detach().clone() for t in leaves]

    o_ref, g_ref = run(_geo_region, relative)
    o_f, g_f = run(fused_geo_region, relative)
    tol = 1e-11 if dtype == torch.float64 else 1e-6
    for a, bb in zip(o_f, o_ref):
        assert _rel(a, bb) < tol, _rel(a, bb)
    for a, bb in zip(g_f, g_ref):
        assert _rel(a, bb) < (1e-11 if dtype == torch.float64 else 1e-5), _rel(a, bb)


@cuda_only
def test_fused_geo_region_bf16_autocast_forward_matches_eager():
    """Under bf16 autocast the fused forward makes the same dtype decisions as eager
    (bf16 Linear on bf16-rounded invariants, bf16 pre-logit sum, fp32 softmax), so the
    two forwards agree to bf16 rounding-boundary flips."""
    from physicsnemo.experimental.nn.isla.geo_kernel import fused_geo_region

    b, n, s = 1, 800, 64
    g = torch.Generator().manual_seed(1)
    r = (torch.randn(b, n, 3, generator=g) * 0.1).cuda()
    z = (torch.randn(b, s, 3, generator=g) * 0.1).cuda()
    nh = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1).cuda()
    ms = torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1).cuda()
    d = (
        torch.nn.functional.normalize(torch.randn(b, 1, 3, generator=g), dim=-1)
        .cuda()
        .expand(b, n, 3)
    )
    logits = torch.randn(b, n, s, generator=g).cuda().bfloat16()
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _geo_region(
            lin, logits, r, nh, d[:, :, None], z, ms, 1e-12, True
        )  # eager region takes (B, N, K, 3)
        fused = fused_geo_region(lin, logits, r, nh, d, z, ms, 1e-12, True)
    assert [t.dtype for t in fused] == [t.dtype for t in ref]
    for a, bb in zip(fused, ref):
        assert _rel(a, bb) < 1e-4, _rel(a, bb)


# --------------------------------------------------------------------------- model level
@cuda_only
def test_fused_geo_kernel_model_exact_in_float64():
    ref, opt = _pair(torch.float64, geo_kernel="fused")
    _assert_same(ref, opt, torch.float64, 1e-11, 1e-11)


@cuda_only
def test_fused_geo_kernel_model_float32_bars():
    torch.backends.cuda.matmul.allow_tf32 = False
    ref, opt = _pair(torch.float32, geo_kernel="fused")
    _assert_same(ref, opt, torch.float32, 1e-6, 1e-5)


@cuda_only
def test_fused_geo_kernel_passive_decoder_and_batch():
    """The read blocks use the fused region too (query_independent), and batch > 1."""
    torch.backends.cuda.matmul.allow_tf32 = False
    kw = dict(query_independent=True, n_decoder_layers=2)
    torch.manual_seed(0)
    ref = ISLA(**REF, **kw).cuda()
    torch.manual_seed(0)
    opt = ISLA(**REF, **kw, geo_kernel="fused").cuda()
    opt.load_state_dict(ref.state_dict())
    inputs, tgt = _cloud(600, batch=2)
    q = dict(
        query_points=inputs["points"][:, :200] + 0.3,
        query_normals=inputs["normals"][:, :200],
    )
    ref.zero_grad()
    opt.zero_grad()
    a = ref(**inputs, **q)
    b = opt(**inputs, **q)
    assert _rel(b.detach(), a.detach()) < 1e-6
    (a * tgt[:, :200]).sum().backward()
    (b * tgt[:, :200]).sum().backward()
    total = torch.cat(
        [p.grad.flatten() for p in ref.parameters() if p.grad is not None]
    ).norm()
    for (k, p), (_, q_) in zip(ref.named_parameters(), opt.named_parameters()):
        if p.grad is not None and p.grad.norm() > 1e-6 * total:
            assert _rel(q_.grad, p.grad) < 1e-5, k


@cuda_only
def test_fused_geo_kernel_keeps_contracts():
    """SE(3) covariance and measure-scale invariance on the fused path (float64), in the
    centered frame (the total-measure scale of the default frame is not measure-scale
    invariant by design, see test_isla_perf), which also runs the 8-invariant kernel."""
    torch.manual_seed(0)
    m = (
        ISLA(
            **REF,
            geo_kernel="fused",
            frame_mode="centered",
            scale_mode="reference_length",
        )
        .cuda()
        .double()
        .eval()
    )
    inputs, _ = _cloud(500, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device="cuda"))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(**inputs)
        moved = m(
            points=inputs["points"] @ q.T + 1.5,
            normals=inputs["normals"] @ q.T,
            global_vectors=inputs["global_vectors"] @ q.T,
            measure_weights=inputs["measure_weights"],
        )
        scaled = m(**{**inputs, "measure_weights": 2.5 * inputs["measure_weights"]})
    assert torch.allclose(moved[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(moved[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)
    assert torch.allclose(scaled, base, atol=1e-10)


def test_geo_kernel_argument_is_validated():
    with pytest.raises(ValueError):
        ISLA(**REF, geo_kernel="triton")


# --------------------------------------------------------------------------- torch.compile
@cuda_only
@pytest.mark.parametrize("geo_kernel", ["eager", "fused"])
def test_torch_compile_fullgraph_matches_eager(geo_kernel):
    """The recipe compiles the whole model (compile: true); it must stay one graph and
    reproduce the eager reference within the study's float32 bars."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.reset()
    ref, opt = _pair(torch.float32, geo_kernel=geo_kernel)
    inputs, _ = _cloud(300)
    ex = torch._dynamo.explain(opt)(**inputs)
    assert ex.graph_break_count == 0, ex.break_reasons
    compiled = torch.compile(opt, fullgraph=True)
    _assert_same(ref, compiled, torch.float32, 1e-6, 1e-5, n=300)
