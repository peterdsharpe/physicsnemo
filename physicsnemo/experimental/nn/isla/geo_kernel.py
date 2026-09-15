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

r"""Fused Triton kernel for ISLA's per-layer geometry region (KERNEL STUDY, 2026-09-13).

The dense geometry region of a slice block (``model._geo_region``) evaluates the
point-anchor relational invariants on the (B, N, S) point-slice grid, the 6->1 (or
8->1) routing-bias Linear on them, the point->slice softmax of the pre-logits plus
that bias, and the mix-weighted pooling of the invariants back to the points. In
eager PyTorch every intermediate of that computation -- rel (B,N,S,3), dist, rel_hat,
the four dot products and the concatenated (B,N,S,geo) invariants -- is written to
memory and read back; torch.compile fuses the elementwise chain but still
materializes the (B,N,S,geo) invariants (once in float32 and once in bfloat16 for the
GEMM) in forward and again in backward. The region is memory-bound, so its cost is
that traffic.

The two kernels here evaluate the same arithmetic on register tiles of
``BLOCK_N`` points x all ``S`` slices and never write any (B,N,S,.) intermediate:
the forward reads the pre-logits once and writes the routing bias and the mix once
(plus the (B,N,geo) pooled invariants); the backward re-derives the invariants from
the (B,N,3) / (B,S,3) geometry inside the tile, reads the three incoming gradients
once and writes the pre-logit gradient once. Everything else the region touches is
O(B N) or O(B S). The per-slice reductions over points (gradients of the anchors and
of the Linear) are written as per-tile partial sums and reduced by torch, so the
result is deterministic (no atomics).

Same arithmetic, same dtype decisions as the eager region under autocast (the
invariants and the Linear operands are rounded to bfloat16 where autocast would run
the Linear and the pooling GEMM in bfloat16, the pre-logit + bias sum is rounded to
the pre-logits' dtype, the softmax runs in float32), so the fused forward reproduces
the eager forward to floating-point reordering. In the backward the reductions
accumulate in float32 where autocast's GEMM backward would have rounded to bfloat16,
which is the one place the two paths differ by more than reordering (the fused
gradient is the more precise one). Float64 is supported for exactness tests.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with the CUDA wheels of torch
    HAS_TRITON = False

MAX_SLICES = 1024


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _rn(x, ROUND_BF16: tl.constexpr):
        """Round-trip through bfloat16 where autocast would have rounded."""
        if ROUND_BF16:
            return x.to(tl.bfloat16).to(tl.float32)
        else:
            return x

    @triton.jit
    def _div(a, b, F64: tl.constexpr):
        # correctly rounded division, as PyTorch's; Triton's ``/`` on fp32 is the approximate div.full
        if F64:
            return a / b
        else:
            return tl.math.div_rn(a, b)

    @triton.jit
    def _sqrt(x, F64: tl.constexpr):
        if F64:
            return libdevice.sqrt(x)
        else:
            return tl.math.sqrt_rn(x)

    @triton.jit
    def _load3(ptr, offs, mask):
        return (
            tl.load(ptr + offs, mask=mask, other=0.0),
            tl.load(ptr + offs + 1, mask=mask, other=0.0),
            tl.load(ptr + offs + 2, mask=mask, other=0.0),
        )

    @triton.jit
    def _geo_fwd_kernel(
        r_ptr,
        n_ptr,
        d_ptr,
        z_ptr,
        m_ptr,
        logit_ptr,
        w_ptr,
        b_ptr,
        bias_out_ptr,
        mix_out_ptr,
        pooled_out_ptr,
        N,
        S,
        eps,
        r_sb,
        r_sn,
        n_sb,
        n_sn,
        d_sb,
        d_sn,
        z_sb,
        z_ss,
        m_sb,
        m_ss,
        l_sb,
        l_sn,
        bo_sb,
        bo_sn,
        mo_sb,
        mo_sn,
        po_sb,
        po_sn,
        RELATIVE: tl.constexpr,
        LIN_BF16: tl.constexpr,
        LOGIT_BF16: tl.constexpr,
        F64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_s = tl.arange(0, BLOCK_S)
        mask_n = offs_n < N
        mask_s = offs_s < S
        mask = mask_n[:, None] & mask_s[None, :]

        r0, r1, r2 = _load3(r_ptr, pid_b * r_sb + offs_n * r_sn, mask_n)
        n0, n1, n2 = _load3(n_ptr, pid_b * n_sb + offs_n * n_sn, mask_n)
        d0, d1, d2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn, mask_n)
        z0, z1, z2 = _load3(z_ptr, pid_b * z_sb + offs_s * z_ss, mask_s)
        m0, m1, m2 = _load3(m_ptr, pid_b * m_sb + offs_s * m_ss, mask_s)

        rel0 = r0[:, None] - z0[None, :]
        rel1 = r1[:, None] - z1[None, :]
        rel2 = r2[:, None] - z2[None, :]
        dist_raw = _sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2, F64)
        dist = tl.maximum(dist_raw, eps)
        h0 = _div(rel0, dist, F64)
        h1 = _div(rel1, dist, F64)
        h2 = _div(rel2, dist, F64)
        g0 = dist
        g1 = libdevice.log(dist)
        g2 = h0 * d0[:, None] + h1 * d1[:, None] + h2 * d2[:, None]
        g3 = h0 * n0[:, None] + h1 * n1[:, None] + h2 * n2[:, None]
        g4 = h0 * m0[None, :] + h1 * m1[None, :] + h2 * m2[None, :]
        g5 = (
            n0[:, None] * m0[None, :]
            + n1[:, None] * m1[None, :]
            + n2[:, None] * m2[None, :]
        )

        w0 = _rn(tl.load(w_ptr + 0), LIN_BF16)
        w1 = _rn(tl.load(w_ptr + 1), LIN_BF16)
        w2 = _rn(tl.load(w_ptr + 2), LIN_BF16)
        w3 = _rn(tl.load(w_ptr + 3), LIN_BF16)
        w4 = _rn(tl.load(w_ptr + 4), LIN_BF16)
        w5 = _rn(tl.load(w_ptr + 5), LIN_BF16)
        bb = _rn(tl.load(b_ptr), LIN_BF16)
        q0 = _rn(g0, LIN_BF16)
        q1 = _rn(g1, LIN_BF16)
        q2 = _rn(g2, LIN_BF16)
        q3 = _rn(g3, LIN_BF16)
        q4 = _rn(g4, LIN_BF16)
        q5 = _rn(g5, LIN_BF16)
        bias = q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4 + q5 * w5
        if not RELATIVE:
            zmag = tl.maximum(_sqrt(z0 * z0 + z1 * z1 + z2 * z2, F64), eps)
            zh0 = _div(z0, zmag, F64)
            zh1 = _div(z1, zmag, F64)
            zh2 = _div(z2, zmag, F64)
            g6 = tl.zeros_like(g0) + zmag[None, :]
            g7 = (
                zh0[None, :] * d0[:, None]
                + zh1[None, :] * d1[:, None]
                + zh2[None, :] * d2[:, None]
            )
            w6 = _rn(tl.load(w_ptr + 6), LIN_BF16)
            w7 = _rn(tl.load(w_ptr + 7), LIN_BF16)
            q6 = _rn(g6, LIN_BF16)
            q7 = _rn(g7, LIN_BF16)
            bias = bias + q6 * w6 + q7 * w7
        bias = _rn(bias + bb, LIN_BF16)

        l_off = pid_b * l_sb + offs_n[:, None] * l_sn + offs_s[None, :]
        logit = tl.load(logit_ptr + l_off, mask=mask, other=0.0).to(bias.dtype)
        x = _rn(logit + bias, LOGIT_BF16)
        x = tl.where(mask_s[None, :], x, float("-inf"))
        mx = tl.max(x, axis=1)
        e = libdevice.exp(x - mx[:, None])
        ssum = tl.sum(e, axis=1)
        mix = _div(e, ssum[:, None], F64)

        tl.store(
            bias_out_ptr + pid_b * bo_sb + offs_n[:, None] * bo_sn + offs_s[None, :],
            bias.to(bias_out_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            mix_out_ptr + pid_b * mo_sb + offs_n[:, None] * mo_sn + offs_s[None, :],
            mix.to(mix_out_ptr.dtype.element_ty),
            mask=mask,
        )

        mq = _rn(mix, LIN_BF16)
        po = pooled_out_ptr + pid_b * po_sb + offs_n * po_sn
        ot = pooled_out_ptr.dtype.element_ty
        tl.store(po + 0, _rn(tl.sum(mq * q0, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 1, _rn(tl.sum(mq * q1, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 2, _rn(tl.sum(mq * q2, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 3, _rn(tl.sum(mq * q3, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 4, _rn(tl.sum(mq * q4, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 5, _rn(tl.sum(mq * q5, axis=1), LIN_BF16).to(ot), mask=mask_n)
        if not RELATIVE:
            tl.store(po + 6, _rn(tl.sum(mq * q6, axis=1), LIN_BF16).to(ot), mask=mask_n)
            tl.store(po + 7, _rn(tl.sum(mq * q7, axis=1), LIN_BF16).to(ot), mask=mask_n)

    @triton.jit
    def _geo_bwd_kernel(
        r_ptr,
        n_ptr,
        d_ptr,
        z_ptr,
        m_ptr,
        logit_ptr,
        w_ptr,
        b_ptr,
        gbias_ptr,
        gmix_ptr,
        gpooled_ptr,
        glogit_ptr,
        gr_ptr,
        gn_ptr,
        gd_ptr,
        gz_part_ptr,
        gm_part_ptr,
        gw_part_ptr,
        gb_part_ptr,
        N,
        S,
        eps,
        n_tiles,
        r_sb,
        r_sn,
        n_sb,
        n_sn,
        d_sb,
        d_sn,
        z_sb,
        z_ss,
        m_sb,
        m_ss,
        l_sb,
        l_sn,
        gb_sb,
        gb_sn,
        gm_sb,
        gm_sn,
        gp_sb,
        gp_sn,
        gl_sb,
        gl_sn,
        RELATIVE: tl.constexpr,
        LIN_BF16: tl.constexpr,
        LOGIT_BF16: tl.constexpr,
        F64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_s = tl.arange(0, BLOCK_S)
        mask_n = offs_n < N
        mask_s = offs_s < S
        mask = mask_n[:, None] & mask_s[None, :]

        r0, r1, r2 = _load3(r_ptr, pid_b * r_sb + offs_n * r_sn, mask_n)
        n0, n1, n2 = _load3(n_ptr, pid_b * n_sb + offs_n * n_sn, mask_n)
        d0, d1, d2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn, mask_n)
        z0, z1, z2 = _load3(z_ptr, pid_b * z_sb + offs_s * z_ss, mask_s)
        m0, m1, m2 = _load3(m_ptr, pid_b * m_sb + offs_s * m_ss, mask_s)

        # ---- recompute the forward tile (identical to _geo_fwd_kernel)
        rel0 = r0[:, None] - z0[None, :]
        rel1 = r1[:, None] - z1[None, :]
        rel2 = r2[:, None] - z2[None, :]
        dist_raw = _sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2, F64)
        dist = tl.maximum(dist_raw, eps)
        h0 = _div(rel0, dist, F64)
        h1 = _div(rel1, dist, F64)
        h2 = _div(rel2, dist, F64)
        g0 = dist
        g1 = libdevice.log(dist)
        g2 = h0 * d0[:, None] + h1 * d1[:, None] + h2 * d2[:, None]
        g3 = h0 * n0[:, None] + h1 * n1[:, None] + h2 * n2[:, None]
        g4 = h0 * m0[None, :] + h1 * m1[None, :] + h2 * m2[None, :]
        g5 = (
            n0[:, None] * m0[None, :]
            + n1[:, None] * m1[None, :]
            + n2[:, None] * m2[None, :]
        )
        w0 = _rn(tl.load(w_ptr + 0), LIN_BF16)
        w1 = _rn(tl.load(w_ptr + 1), LIN_BF16)
        w2 = _rn(tl.load(w_ptr + 2), LIN_BF16)
        w3 = _rn(tl.load(w_ptr + 3), LIN_BF16)
        w4 = _rn(tl.load(w_ptr + 4), LIN_BF16)
        w5 = _rn(tl.load(w_ptr + 5), LIN_BF16)
        bb = _rn(tl.load(b_ptr), LIN_BF16)
        q0 = _rn(g0, LIN_BF16)
        q1 = _rn(g1, LIN_BF16)
        q2 = _rn(g2, LIN_BF16)
        q3 = _rn(g3, LIN_BF16)
        q4 = _rn(g4, LIN_BF16)
        q5 = _rn(g5, LIN_BF16)
        bias = q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4 + q5 * w5
        if not RELATIVE:
            zmag_raw = _sqrt(z0 * z0 + z1 * z1 + z2 * z2, F64)
            zmag = tl.maximum(zmag_raw, eps)
            zh0 = _div(z0, zmag, F64)
            zh1 = _div(z1, zmag, F64)
            zh2 = _div(z2, zmag, F64)
            g6 = tl.zeros_like(g0) + zmag[None, :]
            g7 = (
                zh0[None, :] * d0[:, None]
                + zh1[None, :] * d1[:, None]
                + zh2[None, :] * d2[:, None]
            )
            w6 = _rn(tl.load(w_ptr + 6), LIN_BF16)
            w7 = _rn(tl.load(w_ptr + 7), LIN_BF16)
            q6 = _rn(g6, LIN_BF16)
            q7 = _rn(g7, LIN_BF16)
            bias = bias + q6 * w6 + q7 * w7
        bias = _rn(bias + bb, LIN_BF16)
        l_off = pid_b * l_sb + offs_n[:, None] * l_sn + offs_s[None, :]
        logit = tl.load(logit_ptr + l_off, mask=mask, other=0.0).to(bias.dtype)
        x = _rn(logit + bias, LOGIT_BF16)
        x = tl.where(mask_s[None, :], x, float("-inf"))
        mx = tl.max(x, axis=1)
        e = libdevice.exp(x - mx[:, None])
        ssum = tl.sum(e, axis=1)
        mix = _div(e, ssum[:, None], F64)

        # ---- incoming gradients
        gbias_in = tl.load(
            gbias_ptr + pid_b * gb_sb + offs_n[:, None] * gb_sn + offs_s[None, :],
            mask=mask,
            other=0.0,
        ).to(bias.dtype)
        gmix_in = tl.load(
            gmix_ptr + pid_b * gm_sb + offs_n[:, None] * gm_sn + offs_s[None, :],
            mask=mask,
            other=0.0,
        ).to(bias.dtype)
        gp = gpooled_ptr + pid_b * gp_sb + offs_n * gp_sn
        gp0 = tl.load(gp + 0, mask=mask_n, other=0.0).to(bias.dtype)
        gp1 = tl.load(gp + 1, mask=mask_n, other=0.0).to(bias.dtype)
        gp2 = tl.load(gp + 2, mask=mask_n, other=0.0).to(bias.dtype)
        gp3 = tl.load(gp + 3, mask=mask_n, other=0.0).to(bias.dtype)
        gp4 = tl.load(gp + 4, mask=mask_n, other=0.0).to(bias.dtype)
        gp5 = tl.load(gp + 5, mask=mask_n, other=0.0).to(bias.dtype)
        # pooled = sum_s mix_s geo_s  ->  d/dmix_s = sum_g gp_g geo_gs ; d/dgeo_gs = mix_s gp_g
        gmix = (
            gmix_in
            + gp0[:, None] * q0
            + gp1[:, None] * q1
            + gp2[:, None] * q2
            + gp3[:, None] * q3
            + gp4[:, None] * q4
            + gp5[:, None] * q5
        )
        if not RELATIVE:
            gp6 = tl.load(gp + 6, mask=mask_n, other=0.0).to(bias.dtype)
            gp7 = tl.load(gp + 7, mask=mask_n, other=0.0).to(bias.dtype)
            gmix = gmix + gp6[:, None] * q6 + gp7[:, None] * q7
        # softmax backward over slices
        dot = tl.sum(gmix * mix, axis=1)
        glogit = mix * (gmix - dot[:, None])
        glogit = tl.where(mask, glogit, 0.0)
        tl.store(
            glogit_ptr + pid_b * gl_sb + offs_n[:, None] * gl_sn + offs_s[None, :],
            glogit.to(glogit_ptr.dtype.element_ty),
            mask=mask,
        )
        gb = glogit + gbias_in  # total gradient on the routing bias
        # Linear backward: d/dgeo_g = gb w_g ; d/dw_g = sum gb geo_g ; d/db = sum gb
        gg0 = gb * w0 + mix * gp0[:, None]
        gg1 = gb * w1 + mix * gp1[:, None]
        gg2 = gb * w2 + mix * gp2[:, None]
        gg3 = gb * w3 + mix * gp3[:, None]
        gg4 = gb * w4 + mix * gp4[:, None]
        gg5 = gb * w5 + mix * gp5[:, None]
        tile = pid_b * n_tiles + pid_n
        NG: tl.constexpr = 6 if RELATIVE else 8
        gw = gw_part_ptr + tile * NG
        tl.store(gw + 0, tl.sum(tl.sum(gb * q0, axis=1), axis=0))
        tl.store(gw + 1, tl.sum(tl.sum(gb * q1, axis=1), axis=0))
        tl.store(gw + 2, tl.sum(tl.sum(gb * q2, axis=1), axis=0))
        tl.store(gw + 3, tl.sum(tl.sum(gb * q3, axis=1), axis=0))
        tl.store(gw + 4, tl.sum(tl.sum(gb * q4, axis=1), axis=0))
        tl.store(gw + 5, tl.sum(tl.sum(gb * q5, axis=1), axis=0))
        tl.store(gb_part_ptr + tile, tl.sum(tl.sum(gb, axis=1), axis=0))
        # ---- invariants backward
        # v = d(loss)/d(rel_hat)
        v0 = gg2 * d0[:, None] + gg3 * n0[:, None] + gg4 * m0[None, :]
        v1 = gg2 * d1[:, None] + gg3 * n1[:, None] + gg4 * m1[None, :]
        v2 = gg2 * d2[:, None] + gg3 * n2[:, None] + gg4 * m2[None, :]
        vdoth = v0 * h0 + v1 * h1 + v2 * h2
        # rel_hat = rel / dist: numerator -> v/dist ; denominator -> -(v.rel)/dist^2 = -vdoth/dist
        gdist = gg0 + _div(gg1, dist, F64) - _div(vdoth, dist, F64)
        # dist = clamp_min(|rel|, eps): gradient passes where |rel| >= eps; |rel| backward = rel/|rel|
        ok = dist_raw >= eps
        safe = tl.where(dist_raw > 0, dist_raw, 1.0)
        gscale = tl.where(ok, _div(gdist, safe, F64), 0.0)
        grel0 = _div(v0, dist, F64) + gscale * rel0
        grel1 = _div(v1, dist, F64) + gscale * rel1
        grel2 = _div(v2, dist, F64) + gscale * rel2
        grel0 = tl.where(mask, grel0, 0.0)
        grel1 = tl.where(mask, grel1, 0.0)
        grel2 = tl.where(mask, grel2, 0.0)
        gr = gr_ptr + pid_b * r_sb + offs_n * r_sn
        tl.store(gr + 0, tl.sum(grel0, axis=1), mask=mask_n)
        tl.store(gr + 1, tl.sum(grel1, axis=1), mask=mask_n)
        tl.store(gr + 2, tl.sum(grel2, axis=1), mask=mask_n)
        gz_part = gz_part_ptr + tile * S * 3 + offs_s * 3
        gz0 = -tl.sum(grel0, axis=0)
        gz1 = -tl.sum(grel1, axis=0)
        gz2 = -tl.sum(grel2, axis=0)
        # n_hat: g3 = rel_hat.n ; g5 = n.m
        gn = gn_ptr + pid_b * n_sb + offs_n * n_sn
        tl.store(
            gn + 0,
            tl.sum(tl.where(mask, gg3 * h0 + gg5 * m0[None, :], 0.0), axis=1),
            mask=mask_n,
        )
        tl.store(
            gn + 1,
            tl.sum(tl.where(mask, gg3 * h1 + gg5 * m1[None, :], 0.0), axis=1),
            mask=mask_n,
        )
        tl.store(
            gn + 2,
            tl.sum(tl.where(mask, gg3 * h2 + gg5 * m2[None, :], 0.0), axis=1),
            mask=mask_n,
        )
        # m_s: g4 = rel_hat.m ; g5 = n.m
        gm_part = gm_part_ptr + tile * S * 3 + offs_s * 3
        tl.store(
            gm_part + 0,
            tl.sum(tl.where(mask, gg4 * h0 + gg5 * n0[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        tl.store(
            gm_part + 1,
            tl.sum(tl.where(mask, gg4 * h1 + gg5 * n1[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        tl.store(
            gm_part + 2,
            tl.sum(tl.where(mask, gg4 * h2 + gg5 * n2[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        # d: g2 = rel_hat.d (+ g7 = zhat.d)
        gd0 = gg2 * h0
        gd1 = gg2 * h1
        gd2 = gg2 * h2
        if not RELATIVE:
            gg6 = gb * w6 + mix * gp6[:, None]
            gg7 = gb * w7 + mix * gp7[:, None]
            tl.store(gw + 6, tl.sum(tl.sum(gb * q6, axis=1), axis=0))
            tl.store(gw + 7, tl.sum(tl.sum(gb * q7, axis=1), axis=0))
            gd0 = gd0 + gg7 * zh0[None, :]
            gd1 = gd1 + gg7 * zh1[None, :]
            gd2 = gd2 + gg7 * zh2[None, :]
            # z through zmag (g6) and zhat (g7): zhat = z / zmag
            u0 = tl.sum(tl.where(mask, gg7 * d0[:, None], 0.0), axis=0)
            u1 = tl.sum(tl.where(mask, gg7 * d1[:, None], 0.0), axis=0)
            u2 = tl.sum(tl.where(mask, gg7 * d2[:, None], 0.0), axis=0)
            udotzh = u0 * zh0 + u1 * zh1 + u2 * zh2
            gzmag = tl.sum(tl.where(mask, gg6, 0.0), axis=0) - _div(udotzh, zmag, F64)
            zok = zmag_raw >= eps
            zsafe = tl.where(zmag_raw > 0, zmag_raw, 1.0)
            zscale = tl.where(zok, _div(gzmag, zsafe, F64), 0.0)
            gz0 = gz0 + _div(u0, zmag, F64) + zscale * z0
            gz1 = gz1 + _div(u1, zmag, F64) + zscale * z1
            gz2 = gz2 + _div(u2, zmag, F64) + zscale * z2
        tl.store(gz_part + 0, gz0, mask=mask_s)
        tl.store(gz_part + 1, gz1, mask=mask_s)
        tl.store(gz_part + 2, gz2, mask=mask_s)
        gd = (
            gd_ptr + pid_b * n_sb + offs_n * n_sn
        )  # grad_d is allocated with n_hat's layout (B, N, 3)
        tl.store(gd + 0, tl.sum(tl.where(mask, gd0, 0.0), axis=1), mask=mask_n)
        tl.store(gd + 1, tl.sum(tl.where(mask, gd1, 0.0), axis=1), mask=mask_n)
        tl.store(gd + 2, tl.sum(tl.where(mask, gd2, 0.0), axis=1), mask=mask_n)


def _check(r, n_hat, d_hat, z_pos, m_s, logits_pre, weight, bias):
    b, n, _ = r.shape
    s = z_pos.shape[1]
    if not (r.is_cuda and r.dtype in (torch.float32, torch.float64)):
        raise ValueError("fused geo region needs CUDA float32/float64 geometry")
    if s > MAX_SLICES:
        raise ValueError(
            f"fused geo region supports at most {MAX_SLICES} slices, got {s}"
        )
    if logits_pre.shape != (b, n, s) or logits_pre.stride(-1) != 1:
        raise ValueError(
            "logits_pre must be (B, N, S) with a contiguous last dimension"
        )
    for t in (n_hat, d_hat):
        if t.shape != (b, n, 3) or t.stride(-1) != 1:
            raise ValueError("r, n_hat, d_hat must be (B, N, 3) with unit last stride")
    if (
        r.stride(-1) != 1
        or m_s.shape != (b, s, 3)
        or z_pos.shape != (b, s, 3)
        or z_pos.stride(-1) != 1
        or m_s.stride(-1) != 1
    ):
        raise ValueError("geometry tensors must have unit last stride")
    if weight.shape[-1] not in (6, 8) or bias.numel() != 1:
        raise ValueError("routing Linear must be 6->1 or 8->1")


#: Launch configuration (points per program, warps per program); ``None`` picks the
#: defaults below, which were tuned on an RTX 4090 (kernel study 2026-09-13).
FWD_BLOCK_N: int | None = None
BWD_BLOCK_N: int | None = None
NUM_WARPS: int = 4


def _fwd_block(s):
    if FWD_BLOCK_N is not None:
        return FWD_BLOCK_N
    return 8 if s <= 256 else 4 if s <= 512 else 2


def _bwd_block(s):
    if BWD_BLOCK_N is not None:
        return BWD_BLOCK_N
    return 4 if s <= 256 else 2 if s <= 512 else 1


@torch.library.custom_op("physicsnemo::isla_geo_region_fwd", mutates_args=())
def _geo_region_fwd(
    logits_pre: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    d_hat: torch.Tensor,
    z_pos: torch.Tensor,
    m_s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    relative: bool,
    lin_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(r, n_hat, d_hat, z_pos, m_s, logits_pre, weight, bias)
    b, n, s = logits_pre.shape
    ng = 6 if relative else 8
    out_dt = torch.bfloat16 if lin_bf16 else r.dtype
    bias_out = torch.empty(b, n, s, device=r.device, dtype=out_dt)
    mix_out = torch.empty(b, n, s, device=r.device, dtype=r.dtype)
    pooled = torch.empty(b, n, ng, device=r.device, dtype=out_dt)
    w = weight.reshape(-1).to(r.dtype).contiguous()
    bb = bias.reshape(-1).to(r.dtype).contiguous()
    block_n, block_s = _fwd_block(s), _next_pow2(s)
    grid = (triton.cdiv(n, block_n), b)
    _geo_fwd_kernel[grid](
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        logits_pre,
        w,
        bb,
        bias_out,
        mix_out,
        pooled,
        n,
        s,
        float(eps),
        r.stride(0),
        r.stride(1),
        n_hat.stride(0),
        n_hat.stride(1),
        d_hat.stride(0),
        d_hat.stride(1),
        z_pos.stride(0),
        z_pos.stride(1),
        m_s.stride(0),
        m_s.stride(1),
        logits_pre.stride(0),
        logits_pre.stride(1),
        bias_out.stride(0),
        bias_out.stride(1),
        mix_out.stride(0),
        mix_out.stride(1),
        pooled.stride(0),
        pooled.stride(1),
        RELATIVE=relative,
        LIN_BF16=lin_bf16,
        LOGIT_BF16=logits_pre.dtype == torch.bfloat16,
        F64=r.dtype == torch.float64,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        num_warps=NUM_WARPS,
    )
    return bias_out, mix_out, pooled


@_geo_region_fwd.register_fake
def _(logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias, eps, relative, lin_bf16):
    b, n, s = logits_pre.shape
    out_dt = torch.bfloat16 if lin_bf16 else r.dtype
    return (
        torch.empty(b, n, s, device=r.device, dtype=out_dt),
        torch.empty(b, n, s, device=r.device, dtype=r.dtype),
        torch.empty(b, n, 6 if relative else 8, device=r.device, dtype=out_dt),
    )


@torch.library.custom_op("physicsnemo::isla_geo_region_bwd", mutates_args=())
def _geo_region_bwd(
    logits_pre: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    d_hat: torch.Tensor,
    z_pos: torch.Tensor,
    m_s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_bias: torch.Tensor,
    grad_mix: torch.Tensor,
    grad_pooled: torch.Tensor,
    eps: float,
    relative: bool,
    lin_bf16: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    b, n, s = logits_pre.shape
    ng = 6 if relative else 8
    grad_bias = grad_bias.contiguous()
    grad_mix = grad_mix.contiguous()
    grad_pooled = grad_pooled.contiguous()
    block_n, block_s = _bwd_block(s), _next_pow2(s)
    n_tiles = triton.cdiv(n, block_n)
    glogit = torch.empty_like(logits_pre)
    gr = torch.empty(b, n, 3, device=r.device, dtype=r.dtype)
    gn = torch.empty(b, n, 3, device=r.device, dtype=r.dtype)
    gd = torch.empty(b, n, 3, device=r.device, dtype=r.dtype)
    gz_part = torch.empty(b, n_tiles, s, 3, device=r.device, dtype=r.dtype)
    gm_part = torch.empty(b, n_tiles, s, 3, device=r.device, dtype=r.dtype)
    gw_part = torch.empty(b, n_tiles, ng, device=r.device, dtype=r.dtype)
    gb_part = torch.empty(b, n_tiles, device=r.device, dtype=r.dtype)
    w = weight.reshape(-1).to(r.dtype).contiguous()
    bb = bias.reshape(-1).to(r.dtype).contiguous()
    _geo_bwd_kernel[(n_tiles, b)](
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        logits_pre,
        w,
        bb,
        grad_bias,
        grad_mix,
        grad_pooled,
        glogit,
        gr,
        gn,
        gd,
        gz_part,
        gm_part,
        gw_part,
        gb_part,
        n,
        s,
        float(eps),
        n_tiles,
        r.stride(0),
        r.stride(1),
        gn.stride(0),
        gn.stride(1),
        d_hat.stride(0),
        d_hat.stride(1),
        z_pos.stride(0),
        z_pos.stride(1),
        m_s.stride(0),
        m_s.stride(1),
        logits_pre.stride(0),
        logits_pre.stride(1),
        grad_bias.stride(0),
        grad_bias.stride(1),
        grad_mix.stride(0),
        grad_mix.stride(1),
        grad_pooled.stride(0),
        grad_pooled.stride(1),
        glogit.stride(0),
        glogit.stride(1),
        RELATIVE=relative,
        LIN_BF16=lin_bf16,
        LOGIT_BF16=logits_pre.dtype == torch.bfloat16,
        F64=r.dtype == torch.float64,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        num_warps=NUM_WARPS,
    )
    gz = gz_part.sum(dim=1)
    gm = gm_part.sum(dim=1)
    gw = gw_part.sum(dim=(0, 1)).reshape(weight.shape).to(weight.dtype)
    gbias = gb_part.sum().reshape(bias.shape).to(bias.dtype)
    return glogit, gr, gn, gd, gz, gm, gw, gbias


@_geo_region_bwd.register_fake
def _(
    logits_pre,
    r,
    n_hat,
    d_hat,
    z_pos,
    m_s,
    weight,
    bias,
    grad_bias,
    grad_mix,
    grad_pooled,
    eps,
    relative,
    lin_bf16,
):
    b, n, s = logits_pre.shape
    e = lambda *shape, dt=r.dtype: torch.empty(*shape, device=r.device, dtype=dt)  # noqa: E731
    return (
        torch.empty_like(logits_pre),
        e(b, n, 3),
        e(b, n, 3),
        e(b, n, 3),
        e(b, s, 3),
        e(b, s, 3),
        e(*weight.shape, dt=weight.dtype),
        e(*bias.shape, dt=bias.dtype),
    )


def _setup_context(ctx, inputs, output):
    logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias, eps, relative, lin_bf16 = (
        inputs
    )
    ctx.save_for_backward(logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias)
    ctx.eps, ctx.relative, ctx.lin_bf16 = eps, relative, lin_bf16


def _backward(ctx, grad_bias, grad_mix, grad_pooled):
    logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias = ctx.saved_tensors
    if grad_bias is None:
        grad_bias = torch.zeros_like(logits_pre)
    if grad_mix is None:
        grad_mix = torch.zeros(logits_pre.shape, device=r.device, dtype=r.dtype)
    if grad_pooled is None:
        grad_pooled = torch.zeros(
            logits_pre.shape[0],
            logits_pre.shape[1],
            weight.shape[-1],
            device=r.device,
            dtype=r.dtype,
        )
    glogit, gr, gn, gd, gz, gm, gw, gb = _geo_region_bwd(
        logits_pre,
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        weight,
        bias,
        grad_bias,
        grad_mix,
        grad_pooled,
        ctx.eps,
        ctx.relative,
        ctx.lin_bf16,
    )
    return glogit, gr, gn, gd, gz, gm, gw, gb, None, None, None


_geo_region_fwd.register_autograd(_backward, setup_context=_setup_context)


def fused_geo_region(
    lin: torch.nn.Linear,
    logits_pre,
    r,
    n_hat,
    d_hat,
    z_pos,
    m_s,
    eps: float,
    relative: bool = False,
):
    """Drop-in for ``model._geo_region``: returns (bias (B,N,S), mix (B,N,S),
    pooled (B,N,geo)) from one fused kernel."""
    if not HAS_TRITON:
        raise RuntimeError("fused geo region requires triton")
    lin_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    if torch.is_autocast_enabled("cuda") and not lin_bf16:
        raise RuntimeError("fused geo region supports autocast in bfloat16 only")
    return _geo_region_fwd(
        logits_pre,
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        lin.weight,
        lin.bias,
        float(eps),
        bool(relative),
        bool(lin_bf16),
    )
