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

r"""ISLA: Invariant Slice Attention.

An SE(3)-equivariant soft-slice transformer for steady boundary-value PDE
surrogates. A boundary sample (positions, unit normals, quadrature measures,
optional per-cell boundary data) and the problem's *global inputs* -- zero or
more global vector inputs (``n_global_vectors``; a freestream direction, a
gravity or applied-field direction) and zero or more global scalar inputs
(``n_global_scalars``; a diffusivity, a Reynolds number, a modulus) -- go in;
fields at the boundary (and, in the interior modes, at arbitrary query
points) come out. The name states the design principle: the attention
operates only on invariants of the per-point vector set
:math:`\{r_i, n_i, \hat g_1, \dots, \hat g_K\}` (scaled position, unit
normal, unit global vector inputs) together with the global scalars, and the
frame is re-attached only at the vector heads, so exact
rotation and translation covariance is paid once at the network's edges
instead of in every layer. By default (``frame_mode="relative"``,
``scale_mode="total_measure"``, the reference configuration) positions
enter only as point-to-anchor differences and the length scale is the
square root of the total quadrature measure of the sample, so no centroid,
no sample statistic and no per-dataset reference length appears anywhere in
the forward pass and the frame carries no information about where the
mesher placed its cells. ``frame_mode="centered"`` restores the earlier
construction (plain-mean centering, constant ``reference_length``), which
the similarity gauge, ``odd_head``, ``seed_mode="raw"`` and
``scale_conditioning`` require.

Contracts, all by construction rather than per-layer enforcement:

- **Rotation/translation equivariance.** The backbone sees only
  invariants; vector outputs are expanded in the input vector set plus its
  spherical-basis complements with invariant coefficients. With
  ``similarity_gauge=True`` the gauge (centroid and reference length) is
  derived from the measure-weighted geometry and the model is additionally
  equivariant to geometric scale.
- **Measure-aware aggregation.** Slice states are quadrature-weighted
  means, so the routing reads an (unbiasedly) sampled integral rather
  than a raw point population.
- **Query independence (optional).** With ``query_independent=True``
  queries are decoded by passive read blocks and a prediction at one point
  does not depend on which other points are queried.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping.

Global vector inputs are normalized to unit directions inside the model
(a direction is what the invariants consume); a physically meaningful
magnitude belongs among the global scalar inputs. With one global vector and
no global scalars (the defaults) the network is parameter-for-parameter the
former single-vector model, so every saved checkpoint loads unchanged.

``MeshTransformer2`` is retained as a backward-compatible alias of
:class:`ISLA`.
"""

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.functional.equivariant_ops import spherical_basis


def _softmax_over_points(x: Float[torch.Tensor, "batch tokens slices"], fast: bool = True):
    """Softmax over the point dimension (dim 1) of a (B, N, S) tensor.

    ISLA-PERF (2026-09-09): PyTorch serves a softmax over a *middle* dimension
    with its "spatial" kernel, which parallelizes over the B*S columns only and
    walks the N points serially. On a GB300 that kernel was 85-95% of ISLA's
    training step (203 of 238 ms at 10k tokens, 3.3 of 3.5 s at 80k) and the
    source of the superlinear token scaling. Reducing along a contiguous last
    dimension instead uses the row-parallel kernel. Same arithmetic; the
    floating-point summation order differs (roundoff). ``fast=False`` keeps
    the original kernel for bitwise reproduction of checkpoints trained before
    2026-09-10 (bf16 outputs differ 1-1.5% between kernels; float32 model outputs
    agree to 1e-6). On raw softmax weights the two kernels are not equally
    precise: against a float64 reference the middle-dimension kernel carries a
    ~1e-4 absolute float32 floor at every N from 40k to 100k, the row-parallel
    kernel ~1e-7 (measured on a GB300, 2026-09-10; no defect at any N, including
    across 2^16 points). Which kernel a model runs is decided by this flag at
    instantiation, not by the checkpoint: a checkpoint written before the flag
    existed takes the current default when loaded.
    """
    if not fast:
        return torch.softmax(x, dim=1)
    return torch.softmax(x.transpose(1, 2), dim=-1).transpose(1, 2)


def _relational_invariants(
    r: Float[torch.Tensor, "batch tokens 3"],
    n_hat: Float[torch.Tensor, "batch tokens 3"],
    g_hat: Float[torch.Tensor, "batch tokens vectors 3"],
    z_pos: Float[torch.Tensor, "batch slices 3"],
    m_s: Float[torch.Tensor, "batch slices 3"],
    eps: float,
    c_s: Float[torch.Tensor, "batch slices 3 3"] | None = None,
    relative: bool = False,
) -> Float[torch.Tensor, "batch tokens slices geo"]:
    """The point-anchor invariants (v3b set, generalized to K global vector
    inputs): distance and its log, the unit relative vector dotted with each
    global vector (K terms), with the point normal and with the anchor normal,
    the point normal dotted with the anchor normal, and (centered frame only)
    the anchor radius and the anchor direction dotted with each global vector
    (1 + K terms). Width 5 + K (+ 1 + K centered; + 2 with ``c_s``); for K = 1
    this is the original 6/8-wide set in its original order. Shared by the
    encoder slice blocks and the passive decoder blocks. With ``c_s`` (the
    per-slice second-moment tensor about the anchor; MOM2, 2026-09-08) two more
    invariants are appended: rel_hat^T C_s rel_hat and tr C_s.

    ``relative`` (RELFRAME, 2026-09-10) drops the invariants that refer to the
    frame origin, the anchor radius ``|z_s|`` and the anchor direction
    ``z_hat_s . g_k``, leaving the point-anchor terms (plus the second-moment
    pair), which depend on the anchors' positions relative to the point only."""
    ### Anchors are either shared by all points, z_pos (B, S, 3), or gathered per
    ### point for sparse routing (SPARSE, 2026-09-09), z_pos (B, N, k, 3); the
    ### same arithmetic serves both (the shared case broadcasts over points).
    per_point = z_pos.dim() == 4
    z = z_pos if per_point else z_pos[:, None, :, :]
    m = m_s if per_point else m_s[:, None, :, :]
    rel = r[:, :, None, :] - z  # (B, N, S|k, 3)
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
    rel_hat = rel / dist
    n_exp = n_hat[:, :, None, :]
    feats = [
        dist,
        torch.log(dist),
        ### Broadcast multiply-and-sum, not einsum: einsum is autocast to bf16 and
        ### the geometry must stay in the input precision (for K = 1 this is the
        ### former (rel_hat * d).sum(-1) arithmetic exactly).
        (rel_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),  # (B, N, S|k, K)
        (rel_hat * n_exp).sum(-1, keepdim=True),
        (rel_hat * m).sum(-1, keepdim=True),
        (n_exp * m).sum(-1, keepdim=True),
    ]
    if not relative:
        z_mag = z.norm(dim=-1, keepdim=True).clamp_min(eps)
        z_hat = (z / z_mag).expand(rel.shape[0], rel.shape[1], -1, 3)
        feats += [
            z_mag.expand(rel.shape[0], rel.shape[1], -1, 1),
            (z_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),
        ]
    if c_s is not None:
        ### Second-moment channel: the anchor's covariance seen from the point.
        ### Both quantities are invariant (C_s is a rank-2 equivariant tensor
        ### about a translation-covariant anchor); they carry the transverse
        ### arrangement that first-moment anchors lose whenever the routed
        ### points' transverse first moment vanishes (audit 2026-09-08, item 3).
        if per_point:
            quad = torch.einsum("bnki,bnkij,bnkj->bnk", rel_hat, c_s, rel_hat)[..., None]
            trace = c_s.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None]
        else:
            quad = torch.einsum("bnsi,bsij,bnsj->bns", rel_hat, c_s, rel_hat)[..., None]
            trace = c_s.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, :, None]
        feats += [quad, trace.expand(rel.shape[0], rel.shape[1], -1, 1)]
    return torch.cat(feats, dim=-1)


def _geo_region(lin: nn.Linear, logits_pre, r, n_hat, g_hat, z_pos, m_s, eps: float, c_s=None,
                relative: bool = False):
    """One recompute region per layer: the per-slice routing bias from the
    invariants and the invariants pooled over slices by the resulting
    point->slice mix. Returns (bias (B,N,S), mix (B,N,S), pooled (B,N,8)); the
    (B,N,S,8) invariants and their (B,N,S,3) intermediates never leave the
    region, so under checkpointing they are rebuilt in backward, not stored."""
    geo = _relational_invariants(r, n_hat, g_hat, z_pos, m_s, eps, c_s, relative)
    bias = lin(geo).squeeze(-1)
    mix = torch.softmax(logits_pre + bias, dim=-1)  # normalized over slices
    return bias, mix, torch.einsum("bns,bnsg->bng", mix, geo)


def _fused_geo_region():
    """The Triton-fused dense geometry region (geo_kernel="fused"); imported lazily so
    that the module loads without triton and the eager path never touches it."""
    from .geo_kernel import fused_geo_region

    return fused_geo_region


def _geo_region_sparse(lin: nn.Linear, logits_pre, r, n_hat, g_hat, z_pos, m_s, eps: float, c_s, k: int,
                       relative: bool = False):
    """SPARSE (2026-09-09): the recompute region of _geo_region restricted to each
    point's k nearest anchors. Invariants, routing bias and the point->slice mix
    exist only on the (B, N, k) selected anchors; the bias is scattered back into
    a dense (B, N, S) tensor with a large negative fill for unselected anchors, so
    the point->slice softmax and the slice-state assignment exclude them. At
    k = n_slices every anchor is selected and the result equals _geo_region to
    roundoff (the pooled sum runs in a different order). Returns
    (bias_full (B,N,S) with the fill, mix_full (B,N,S) with zeros, pooled (B,N,geo))."""
    b, n, s = logits_pre.shape
    with torch.no_grad():
        d2 = torch.cdist(r.float(), z_pos.float())  # (B, N, S)
        idx = d2.topk(k, dim=-1, largest=False).indices  # (B, N, k)
    bidx = torch.arange(b, device=r.device)[:, None, None]
    z_k = z_pos[bidx, idx]  # (B, N, k, 3)
    m_k = m_s[bidx, idx]
    c_k = c_s[bidx, idx] if c_s is not None else None  # (B, N, k, 3, 3)
    geo = _relational_invariants(r, n_hat, g_hat, z_k, m_k, eps, c_k, relative)  # (B, N, k, geo)
    bias_k = lin(geo).squeeze(-1)  # (B, N, k)
    logits_k = torch.gather(logits_pre, -1, idx) + bias_k
    mix_k = torch.softmax(logits_k, dim=-1)
    pooled = torch.einsum("bnk,bnkg->bng", mix_k, geo)
    neg = -1e4 if logits_pre.dtype in (torch.float16, torch.bfloat16) else -1e9
    bias_full = logits_pre.new_full((b, n, s), neg).scatter(-1, idx, bias_k.to(logits_pre.dtype))
    mix_full = logits_pre.new_zeros((b, n, s)).scatter(-1, idx, mix_k.to(logits_pre.dtype))
    return bias_full, mix_full, pooled


def _geo_width(n_global_vectors: int, relative: bool, second_moment: bool = False) -> int:
    """Width of _relational_invariants: dist, log dist, K rel.g_k, rel.n, rel.m,
    n.m (5 + K); centered adds |z_s| and K zhat_s.g_k; MOM2 adds two. K = 1
    gives the original 6 (relative) / 8 (centered)."""
    k = int(n_global_vectors)
    return 5 + k + (0 if relative else 1 + k) + (2 if second_moment else 0)


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4,
                 use_relational_geo: bool = True, geo_checkpoint: bool = False,
                 second_moment: bool = False, anchor_topk: int = 0,
                 fast_point_softmax: bool = True, relative_frame: bool = False,
                 geo_kernel: str = "eager", n_global_vectors: int = 1) -> None:
        super().__init__()
        self.use_relational_geo = use_relational_geo
        self.fast_point_softmax = bool(fast_point_softmax)
        ### KERNEL STUDY (2026-09-13): "fused" evaluates the dense geometry region
        ### (invariants -> routing bias -> point->slice mix -> pooled invariants) in
        ### one Triton kernel per direction that never materializes a (B,N,S,.)
        ### tensor (see geo_kernel.py); same arithmetic as _geo_region, recompute
        ### built in, so geo_checkpoint is moot on that path. Falls back to the
        ### eager region for sparse routing and the second-moment channel.
        self.geo_kernel = geo_kernel
        ### RELFRAME (2026-09-10): without a frame origin the two origin-referring
        ### invariants (|z_s|, zhat_s.g_k) are gone and the geo width is 5 + K (+2 MOM2).
        self.relative_frame = bool(relative_frame)
        ### SPARSE (2026-09-09): route each point to its anchor_topk nearest
        ### anchors only (0 = dense). The (B, N, S, geo) invariants and their
        ### (B, N, S, 3) intermediates, the dominant cost of a slice block,
        ### shrink to (B, N, k, .); slice states are still formed from every point
        ### that selected the slice. Exact at k = n_slices.
        self.anchor_topk = int(anchor_topk)
        ### MOM2 (2026-09-08): per-slice second-moment tensor about the anchor,
        ### read at each point as rel_hat^T C_s rel_hat and tr C_s (two more
        ### invariants). Restores the transverse arrangement that first-moment
        ### anchors cannot see. Flag-gated; off reproduces the v3b set exactly.
        self.second_moment = bool(second_moment)
        self.n_geo = _geo_width(n_global_vectors, self.relative_frame, self.second_moment)
        ### Activation recompute (2026-09-07 memory attribution): the per-slice
        ### geometry tensors -- rel (B,N,S,3), rel_hat, dist and two bf16 copies
        ### of the (B,N,S,8) invariants -- are 76% of ISLA's saved activations
        ### at 10k tokens. With geo_checkpoint the invariants are rebuilt from
        ### (r, n_hat, g_hat, z_pos, m_s) inside backward instead of stored;
        ### the forward is bitwise unchanged (same ops, same order).
        self.geo_checkpoint = bool(geo_checkpoint)
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        ### Relational geometry (v2): per-slice equivariant anchors and
        ### point-anchor invariants -- many local, data-adaptive reference
        ### points instead of any global frame (smooth by construction).
        ### Built only when used: parameters that never receive gradients
        ### crash DDP (A35b nogeo, 2026-09-05).
        self.geo_width = hidden // 2
        if use_relational_geo:
            self.geo_logit = nn.Linear(self.n_geo, 1)
            self.geo_feat = nn.Linear(self.n_geo, self.geo_width)
        self.slice_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )
        self.broadcast = nn.Linear(2 * hidden + hidden // 2, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(
        self,
        h: Float[torch.Tensor, "batch tokens hidden"],
        log_w: Float[torch.Tensor, "batch tokens 1"],
        r: Float[torch.Tensor, "batch tokens 3"],
        n_hat: Float[torch.Tensor, "batch tokens 3"],
        g_hat: Float[torch.Tensor, "batch tokens vectors 3"],
        eps: float,
    ) -> Float[torch.Tensor, "batch tokens hidden"]:
        ### Soft assignment of points to slices; measure weights enter as a
        ### log-space bias so slice states are quadrature-weighted means.
        logits = self.assign(self.norm_assign(h))  # (B, N, S)
        a = _softmax_over_points(logits + log_w, self.fast_point_softmax)  # normalized over points
        ### Equivariant anchors: weighted mean position AND mean normal
        ### direction per slice (v3b) -- anchors gain orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, 3)
        m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)
        ### Geometry refines the routing and the readback. A35b ablation:
        ### use_relational_geo=False removes the anchor-relational invariants
        ### from routing and readback (Transolver-style feature-only slicing).
        if self.use_relational_geo:
            ### Pool the 8 invariants over slices FIRST, then project: exactly
            ### equal to projecting then pooling (the projection is affine and
            ### point_mix sums to one over slices), but the saved activation is
            ### (B, N, 8) instead of (B, N, S, hidden/2) -- ~0.5 GB per layer at
            ### 10k tokens, 256 slices, hidden 192 (A35b memory derivation).
            c_s = None
            if self.second_moment:
                ### C_s = E_a[r r^T] - z_s z_s^T under the point->slice weights a
                ### (which sum to one over points): no (B, N, S, .) intermediate.
                rr = (r[:, :, :, None] * r[:, :, None, :]).reshape(r.shape[0], r.shape[1], 9)
                c_s = torch.einsum("bns,bnk->bsk", a, rr).reshape(r.shape[0], -1, 3, 3)
                c_s = c_s - z_pos[:, :, :, None] * z_pos[:, :, None, :]
            if self.anchor_topk and self.anchor_topk < logits.shape[-1]:
                region, geo_args = _geo_region_sparse, (self.geo_logit, logits, r, n_hat, g_hat, z_pos, m_s, eps, c_s, self.anchor_topk, self.relative_frame)
            elif self.geo_kernel == "fused" and c_s is None:
                ### The fused kernel is written for one global vector (enforced at construction).
                region, geo_args = _fused_geo_region(), (self.geo_logit, logits, r, n_hat, g_hat[:, :, 0], z_pos, m_s, eps, self.relative_frame)
            else:
                region, geo_args = _geo_region, (self.geo_logit, logits, r, n_hat, g_hat, z_pos, m_s, eps, c_s, self.relative_frame)
            if self.geo_checkpoint and region is _geo_region:
                bias, point_mix, pooled = checkpoint(region, *geo_args, use_reentrant=False)
            else:
                bias, point_mix, pooled = region(*geo_args)
            logits = logits + bias
        else:
            point_mix = torch.softmax(logits, dim=-1)  # normalized over slices
        a = _softmax_over_points(logits + log_w, self.fast_point_softmax)
        z = torch.einsum("bns,bnh->bsh", a, h)  # slice states
        z = z + self.slice_mlp(z)
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        if self.use_relational_geo:
            geo_pool = self.geo_feat(pooled)
        else:
            geo_pool = h.new_zeros(h.shape[0], h.shape[1], self.geo_width)
        h = h + self.broadcast(torch.cat([h, back, geo_pool], dim=-1))
        return h + self.mlp(self.norm_mlp(h))




class _ReadBlock(nn.Module):
    """Passive decoder layer (v5a): queries read encoder slices/anchors,
    never write. Removing the write-back is what makes predictions at one
    query independent of every other query (given a fixed source sample)."""

    # v5a2: the full v3b relational-feature set (thin decoder pipes collapse
    # training -- measured twice now, v3a and v5a-v1)

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4,
                 geo_checkpoint: bool = False, relative_frame: bool = False,
                 geo_kernel: str = "eager", n_global_vectors: int = 1) -> None:
        super().__init__()
        self.geo_checkpoint = bool(geo_checkpoint)
        self.relative_frame = bool(relative_frame)
        self.geo_kernel = geo_kernel  # see _SliceBlock
        self.n_geo = n_geo = _geo_width(n_global_vectors, self.relative_frame)
        self.norm = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        self.geo_logit = nn.Linear(n_geo, 1)
        self.geo_feat = nn.Linear(n_geo, hidden // 2)
        self.broadcast = nn.Linear(2 * hidden + hidden // 2, hidden)
        self.local_read = nn.Linear(2 * hidden, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(self, q_h, q_r, q_n, q_g, z_states, z_pos, m_s, eps,
                src_r=None, src_h=None, src_w=None, local_rho=None, kernel_logspace=False):
        logits_pre = self.assign(self.norm(q_h))
        if self.geo_kernel == "fused":
            _, mix, pooled = _fused_geo_region()(self.geo_logit, logits_pre, q_r, q_n, q_g[:, :, 0], z_pos, m_s, eps, self.relative_frame)
        elif self.geo_checkpoint:
            _, mix, pooled = checkpoint(_geo_region, self.geo_logit, logits_pre, q_r, q_n, q_g, z_pos, m_s, eps, None, self.relative_frame, use_reentrant=False)
        else:
            _, mix, pooled = _geo_region(self.geo_logit, logits_pre, q_r, q_n, q_g, z_pos, m_s, eps, None, self.relative_frame)
        back = torch.einsum("bqs,bsh->bqh", mix, z_states)
        geo_pool = self.geo_feat(pooled)  # pool-then-project (exact)
        q_h = q_h + self.broadcast(torch.cat([q_h, back, geo_pool], dim=-1))
        if src_h is not None:
            ### v5a3: local token readout -- the per-point detail 256 slice
            ### states cannot carry. Measure-weighted Gaussian kernel over
            ### SOURCE positions attending to encoder states; queries still
            ### never write, so query-independence is preserved exactly.
            local = _kernel_readout(q_r, src_r, src_h, src_w, local_rho, eps, logspace=kernel_logspace)
            q_h = q_h + self.local_read(torch.cat([q_h, local], dim=-1))
        return q_h + self.mlp(self.norm_mlp(q_h))



def _kernel_readout(q_r, src_r, src_h, src_w, rho, eps, logspace: bool = False):
    """Measure-weighted Gaussian-kernel average of source states at query
    positions, row-chunked. Passive: a pure function of the source.

    ``logspace=False`` is the legacy form (kernel mass clamped at ``eps``):
    for a query farther than a few ``rho`` from every source the mass
    underflows, the clamp takes over, and the readout scales with the
    absolute source weights (measured 2026-09-09: a 4.2x weight rescale moved
    the passive output by 0.16). Kept so trained passive checkpoints
    reproduce. ``logspace=True`` (support-token mode) evaluates the same
    average as a softmax over sources with logits -d^2/rho^2 + log w, which
    is exactly invariant to rescaling the source measure and well defined
    everywhere; where the mass is not tiny the two agree to roundoff, and
    where it is, the log-space form attends to the nearest sources instead of
    returning a clamp-scaled value. ``src_w`` is the log-weight when
    ``logspace`` is set."""
    b, nq, _ = q_r.shape
    outs = []
    chunk = 4096
    for i0 in range(0, nq, chunk):
        d2 = torch.cdist(q_r[:, i0 : i0 + chunk], src_r).square()
        if logspace:
            logits = -d2 / (rho * rho) + src_w[:, None, :]
            k = torch.softmax(logits, dim=-1)
            outs.append(torch.einsum("bcn,bnh->bch", k, src_h))
        else:
            k = torch.exp(-d2 / (rho * rho)) * src_w[:, None, :]
            mass = k.sum(-1, keepdim=True).clamp_min(eps)
            outs.append(torch.einsum("bcn,bnh->bch", k, src_h) / mass)
    return torch.cat(outs, dim=1)

class ISLA(Module):
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges (see module docs)."""

    class MetaData(ModelMetaData):
        jit: bool = False
        amp: bool = True

    #: Class names this architecture was saved under before it was renamed; the
    #: checkpoint loader searches these when no file exists under the current
    #: name (see physicsnemo.utils.checkpoint._legacy_checkpoint_filename), so
    #: checkpoints written as ``MeshTransformer2.*.mdlus`` still load into ISLA.
    _legacy_class_names: tuple[str, ...] = ("MeshTransformer2",)

    def __init__(
        self,
        *,
        out_scalars: int = 1,
        out_vectors: int = 1,
        hidden: int = 256,
        n_layers: int = 12,
        n_slices: int = 256,
        mlp_ratio: int = 4,
        reference_length: float = 8.0,
        use_measure_weights: bool = True,
        measure_weight_power: float = 1.0,
        fast_point_softmax: bool = True,
        use_local_features: bool = False,
        local_radii: tuple[float, ...] = (0.01, 0.03),
        n_boundary_scalars: int = 0,
        n_global_vectors: int = 1,
        n_global_scalars: int = 0,
        parity_fix: bool = False,
        parity_gate_scale: float = 0.0,
        vector_basis: str = "globe7",
        odd_head: bool = False,
        similarity_gauge: bool = False,
        raw_coord_channel: bool = False,
        interior_queries: bool = False,
        anchor_normal_rho: float = 0.25,
        latent_volume_tokens: bool = False,
        lvt_offsets: tuple = (0.5, 1.0, 2.0),
        wake_tokens: bool = False,
        wake_offsets: tuple = (1.0, 2.0, 4.0),
        seed_mode: str = "invariant",
        use_relational_geo: bool = True,
        scale_conditioning: bool = False,
        query_independent: bool = False,
        n_anchors: int = 0,
        n_decoder_layers: int = 4,
        local_readout_rho: float = 0.02,
        query_tokens: bool = False,
        geo_checkpoint: bool = False,
        n_query_scalars: int = 0,
        query_scalar_scale: str = "length",
        query_local_features: bool = False,
        query_local_radii: tuple[float, ...] = (0.05, 0.15, 0.5),
        query_mass: str = "geometric_mean",
        second_moment_features: bool = False,
        anchor_topk: int = 0,
        support_tokens: bool = False,
        query_density_feature: bool = False,
        query_density_radius: float = 0.05,
        query_neighbor_features: bool = False,
        query_neighbor_k: int = 16,
        center_mode: str = "plain",
        frame_mode: str = "relative",
        scale_mode: str = "total_measure",
        geo_kernel: str = "eager",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### KERNEL STUDY (2026-09-13): geo_kernel selects the implementation of the
        ### per-layer geometry region, not its arithmetic. "eager" is the PyTorch
        ### region (with geo_checkpoint deciding whether its (B,N,S,.) intermediates
        ### are stored or rebuilt in backward); "fused" is the Triton kernel of
        ### geo_kernel.py, which reads the pre-logits once and writes the bias and
        ### mix once per direction, with the recompute built in (CUDA only; the
        ### sparse-routing and second-moment variants stay eager).
        if geo_kernel not in ("eager", "fused"):
            raise ValueError(f"geo_kernel must be 'eager' or 'fused', got {geo_kernel!r}")
        self.geo_kernel = geo_kernel
        ### GLOBAL INPUTS (2026-09-14, ruling: the architecture targets steady
        ### boundary-value problems in general, so the problem's global data
        ### are zero or more global VECTOR inputs (each enters as a unit
        ### direction: one seed term n.g_k per token, one direction cosine
        ### rel.g_k per point-anchor pair, and the head basis gains g_k with the
        ### spherical complements of (u, g_k)) and zero or more global SCALAR
        ### inputs (PDE parameters; appended to every token's seed features).
        ### n_global_vectors=1, n_global_scalars=0 reproduces the former
        ### single-vector model parameter-for-parameter, which is why they are
        ### the defaults: checkpoints store their constructor arguments.
        self.n_global_vectors = int(n_global_vectors)
        self.n_global_scalars = int(n_global_scalars)
        if self.n_global_vectors < 0 or self.n_global_scalars < 0:
            raise ValueError("n_global_vectors and n_global_scalars must be non-negative")
        if self.n_global_vectors != 1:
            one_vector_only = {
                "geo_kernel='fused'": geo_kernel == "fused",
                "odd_head": odd_head,
                "parity_fix": parity_fix,
                "latent_volume_tokens": latent_volume_tokens,
                "wake_tokens": wake_tokens,
            }
            bad = [k for k, v in one_vector_only.items() if v]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} are written for exactly one global vector input; "
                    f"got n_global_vectors={self.n_global_vectors}"
                )
        ### RELFRAME (2026-09-10, ruling: no sample statistic may enter the
        ### flagship's frame; 2026-09-11: frame_mode="relative" and
        ### scale_mode="total_measure" became the class defaults, the
        ### reference configuration; checkpoints store their constructor
        ### arguments, so models saved under the old defaults load unchanged).
        ### frame_mode="relative" removes the frame origin
        ### altogether: r = points / L with no centering. The six scalars that
        ### referred to the centroid are gone -- the four seed features |r|,
        ### log|r|, rhat.g, rhat.n (seeds reduce to n.g_k; the measure enters the
        ### routing as before) and the relational invariants |z_s| and
        ### zhat_s.g_k (see _relational_invariants). The vector head's radial
        ### basis vector rhat becomes the direction from the point's soft slice
        ### anchor (a measure-weighted mean, the same construction as the
        ### relational anchors) so the head stays translation covariant.
        ### Exact translation invariance holds without any centering.
        ### scale_mode="total_measure" divides positions by sqrt(sum of the
        ### measure weights) per sample -- the total surface area, an integral
        ### of the geometry that is consistent under Horvitz-Thompson weights
        ### -- instead of the constant reference_length.
        if frame_mode not in ("centered", "relative"):
            raise ValueError(f"frame_mode must be 'centered' or 'relative', got {frame_mode!r}")
        self.frame_mode = frame_mode
        self.relative_frame = frame_mode == "relative"
        if scale_mode not in ("reference_length", "total_measure", "global"):
            raise ValueError(
                f"scale_mode must be 'reference_length', 'total_measure' or 'global', got {scale_mode!r}"
            )
        self.scale_mode = scale_mode
        if similarity_gauge and scale_mode != "reference_length":
            raise ValueError("similarity_gauge sets its own scale; use scale_mode='reference_length'")
        if self.relative_frame:
            if center_mode != "plain":
                raise ValueError(
                    "frame_mode='relative' (the default) has no center; pass frame_mode='centered' to use center_mode"
                )
            if similarity_gauge or odd_head or seed_mode != "invariant" or scale_conditioning:
                raise ValueError(
                    "frame_mode='relative' (the default) excludes similarity_gauge, odd_head, seed_mode='raw' "
                    "and scale_conditioning (each reads a position relative to a frame origin); "
                    "pass frame_mode='centered' (and scale_mode='reference_length') to use them"
                )
        ### CENTER (2026-09-10): the constant gauge (similarity_gauge=False,
        ### the reference configuration) centers by the PLAIN mean of the
        ### sampled points. The routing softmax adds raw log-weights (invariant
        ### to uniform rescaling), the slice moments are attention-normalized,
        ### and the reference configuration has no local features, so the
        ### unweighted centroid is the ONLY sampling-distribution-dependent
        ### quantity in its forward pass: under a 10:1 sampling bias toward one
        ### half of the body the plain mean moves by a large fraction of the
        ### body length, every r shifts, and all relational invariants leave
        ### their trained range. "measure" centers by the normalized measure
        ### weights (the similarity-gauge centroid formula; plain mean when no
        ### weights are given) while keeping the constant length scale.
        ### FRAME-FULL (2026-09-10, diagnostic of the frame-variance mechanism):
        ### "global" reads the center from the forward argument ``frame_center``
        ### (B, 3), computed once per case from the FULL surface geometry by the
        ### data pipeline; the frame of a surface is a property of the geometry,
        ### so supplied this way it has no sampling dependence and zero estimator
        ### variance (a centroid estimated from 10k area-weighted samples of a
        ### multi-element mesh whose cell areas span orders of magnitude does
        ### not). scale_mode="global" likewise divides by ``frame_scale`` (B,).
        if center_mode not in ("plain", "measure", "global"):
            raise ValueError(f"center_mode must be 'plain', 'measure' or 'global', got {center_mode!r}")
        self.center_mode = center_mode
        ### Density-factorial knob (prereg 3f4e4af7 follow-up): with False,
        ### the assignment softmax ignores quadrature weights entirely,
        ### isolating the measure-bias pathway of density sensitivity.
        self.use_measure_weights = use_measure_weights
        ### MEAS-METRIC (2026-09-09): temper the routing measure toward uniform,
        ### w -> w^alpha (alpha = 1 the quadrature measure, 0 uniform). Measure-scale
        ### invariance is kept for every alpha (a common factor c^alpha cancels in
        ### the softmax); alpha = 0 reproduces use_measure_weights=False exactly.
        self.measure_weight_power = float(measure_weight_power)
        ### ISLA-PERF (2026-09-09): point softmaxes reduce along a contiguous last
        ### dimension (see _softmax_over_points); False reproduces the original
        ### middle-dimension kernel bitwise (roundoff-level difference otherwise).
        ### Default True since 2026-09-10: training-neutral on DrivAerML (+1.8% in
        ### float32, inside the 3% bar) at 0.59x step time and 0.4x memory; the two
        ### kernels agree to 1e-6 in float32 evaluation. Checkpoints trained before
        ### 2026-09-10 used the reference kernel.
        self.fast_point_softmax = bool(fast_point_softmax)
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.reference_length = float(reference_length)
        self.eps = eps
        ### v4 EXPERIMENT (prereg f16bef42, flag-gated, default off): local
        ### measure-weighted patch integrals at physical radii -- the
        ### family-portable channel. 7 invariants per radius.
        self.use_local_features = use_local_features
        self.local_radii = tuple(float(x) for x in local_radii)
        ### G3 experiment channel (prereg pending): per-point boundary-condition
        ### scalars (e.g. a Dirichlet trace). Scalars are invariants, so every
        ### contract is untouched; they simply widen the seed features.
        self.n_boundary_scalars = int(n_boundary_scalars)
        ### M1 experiment (lit synthesis 2026-08-20): deliberately BREAK exact
        ### scale equivariance with a log-size scalar (Reynolds proxy) --
        ### the Petrache-Trivedi over-symmetrization test.
        self.scale_conditioning = scale_conditioning
        ### M3 audit fix (2026-08-20): the e_phi basis complements are cross
        ### products (pseudovectors) while the trunk's coefficients are
        ### parity-even, so the vector head violated reflection equivariance.
        ### Gating the e_phi coefficients with the smooth pseudoscalar
        ### r_hat . (n_hat x g_hat) restores exact parity covariance. Off by
        ### default so frozen checkpoints keep their trained behavior.
        self.parity_fix = parity_fix
        ### W1' (instrument wave follow-up): the raw pseudoscalar gate p also
        ### modulates MAGNITUDE (|p| ~ 0 wherever r, n, d are near-coplanar,
        ### e.g. the symmetry plane), which W1 showed costs ~19% wall-shear
        ### accuracy. tanh(p / scale) keeps the odd sign structure (exact
        ### reflection covariance) with unit magnitude away from p = 0.
        self.parity_gate_scale = float(parity_gate_scale)
        ### v5a4 experiment: AB-UPT-style anchor-conditioned decode -- only a
        ### fixed-size anchor subset runs the interacting encoder; all points
        ### decode through the read-only path. The anchor count is absolute
        ### (not a fraction) so the anchor set cannot depend on the query set,
        ### which is the query-independence contract. 0 disables (v5a3).
        self.n_anchors = int(n_anchors)
        ### A35b ablations: seed_mode="raw" replaces the {r,n,g_k} invariant
        ### seeds with the raw vectors [r, n, g_1..g_K] (GeoTransolver-style inputs);
        ### use_relational_geo=False removes anchor geometry from the slices.
        if seed_mode not in ("invariant", "raw"):
            raise ValueError(f"unknown seed_mode {seed_mode!r}")
        self.seed_mode = seed_mode
        K = self.n_global_vectors
        if seed_mode == "invariant":
            n_base = K if self.relative_frame else 3 + 2 * K
        else:
            n_base = 6 + 3 * K
        n_seed = (n_base + ((5 + 2 * K) * len(self.local_radii) if use_local_features else 0)
                  + self.n_boundary_scalars + (1 if scale_conditioning else 0)
                  + (6 if raw_coord_channel else 0) + self.n_global_scalars)
        ### With no global inputs and no other seed channel the seed is empty;
        ### every token then starts from one learned embedding and all
        ### separation comes from the slice blocks' relational anchors.
        self.constant_seed = n_seed == 0
        if self.constant_seed:
            n_seed = 1
        ### Seed invariants of {r, n, g_k} plus the global scalars; separation
        ### comes from the slice blocks' relational anchors (v2), not from these.
        self.embed = nn.Sequential(
            nn.Linear(n_seed, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        ### MOM2 (audit 2026-09-08, item 3; notebook #sec-nb-mom2-prereg): the
        ### per-slice second-moment channel. Off by default so every trained
        ### checkpoint reproduces exactly.
        self.second_moment_features = bool(second_moment_features)
        ### SPARSE (2026-09-09; notebook #sec-nb-sparse-routing-prereg): each point
        ### evaluates relational geometry against its anchor_topk nearest anchors
        ### only. 0 (default) is the dense model; k = n_slices reproduces it.
        self.anchor_topk = int(anchor_topk)
        if self.anchor_topk < 0 or self.anchor_topk > n_slices:
            raise ValueError("anchor_topk must lie in [0, n_slices]")
        self.blocks = nn.ModuleList(
            _SliceBlock(hidden, n_slices, mlp_ratio, use_relational_geo=use_relational_geo,
                        geo_checkpoint=geo_checkpoint, second_moment=self.second_moment_features,
                        anchor_topk=self.anchor_topk,
                        fast_point_softmax=self.fast_point_softmax,
                        relative_frame=self.relative_frame, geo_kernel=geo_kernel,
                        n_global_vectors=K)
            for _ in range(n_layers)
        )
        if self.relative_frame:
            ### Head frame (RELFRAME): per-point soft slice anchor for the
            ### radial basis vector, the odd head's construction.
            self.frame_assign = nn.Linear(hidden, n_slices)
        ### v5a EXPERIMENT (flag-gated, default off): encode/decode split.
        ### Queries decode passively from final encoder slices and anchors:
        ### query-independent by construction given the source sample.
        self.query_independent = query_independent
        self.local_readout_rho = float(local_readout_rho)
        if query_independent:
            self.final_assign = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, n_slices)
            )
            self.read_blocks = nn.ModuleList(
                _ReadBlock(hidden, n_slices, mlp_ratio, geo_checkpoint=geo_checkpoint,
                           relative_frame=self.relative_frame, geo_kernel=geo_kernel,
                           n_global_vectors=K)
                for _ in range(n_decoder_layers)
            )
        self.norm_out = nn.LayerNorm(hidden)
        ### Vector head: coefficients over {g_1..g_K, n, rhat} plus the
        ### spherical-basis complements of (rhat, n) and of each (rhat, g_k) --
        ### the GLOBE multi-vector expansion (4 + 3K basis vectors; 7 for K = 1).
        ### L2 experiment (2026-09-02): the two e_phi complements are
        ### pseudovectors. "globe7" is the original basis; "true5" drops them
        ### (capacity control); "true7" replaces them with the TRUE vectors
        ### e_phi_n x g_hat and e_phi_g x n_hat (pseudo x true = true), giving
        ### exact reflection covariance with no gating and no lost channel.
        if vector_basis not in ("globe7", "true5", "true7"):
            raise ValueError(f"unknown vector_basis {vector_basis!r}")
        self.vector_basis = vector_basis
        if vector_basis == "true5":
            self.n_basis = 3 + 2 * K
        elif vector_basis == "true7":
            self.n_basis = 3 + 4 * K
        else:
            self.n_basis = 4 + 3 * K
        self.head = nn.Linear(hidden, out_scalars + out_vectors * self.n_basis)
        ### W2 (2026-09-02): odd-coefficient head. {r,n,g} span R^3, so the
        ### e_phi (pseudovector) direction is reachable COVARIANTLY only with a
        ### parity-odd coefficient, and the trunk emits even invariants only.
        ### Build pseudoscalars from {r, n, g} and the point's soft slice
        ### anchor (weighted anchor position z and normal m, both true
        ### vectors), and set coeff_phi = sum_k p_k * g_k(h). Exactly
        ### reflection-covariant; not killed where any single p_k vanishes.
        self.odd_head = odd_head
        ### S1 (critic review 2026-09-02): the original reduction used an
        ### UNWEIGHTED centroid and a CONSTANT reference length, so the model was
        ### neither measure-complete nor scale-equivariant despite the book's
        ### claims. This gauge uses the measure-weighted centroid and the
        ### measure-weighted RMS radius: exact geometric-scale equivariance
        ### and a density-robust frame.
        self.similarity_gauge = similarity_gauge
        ### Branch-B mechanism discriminator (2026-09-04): append the raw
        ### gauge-normalized coordinates and normal components to the seed
        ### invariants. This DELIBERATELY breaks SE(3) equivariance; it tests
        ### whether GeoTransolver's pointwise raw-coordinate features are what
        ### carry its smaller OOD degradation ratio. Never a product setting.
        self.raw_coord_channel = raw_coord_channel
        ### V0 (boundary->interior, 2026-09-05): off-surface queries carry no
        ### normal. Instead of removing the normal from every query-side
        ### invariant (a thin-pipe rewrite), derive an equivariant proxy normal
        ### per query as the geometric soft assignment of the query position
        ### to the slice anchors, applied to the anchor mean normals m_s.
        ### Exactly SE(3)-covariant, smooth, defined everywhere, no learned
        ### parameters; explicit query_normals (e.g. SDF normals) override it.
        self.interior_queries = interior_queries
        self.anchor_normal_rho = float(anchor_normal_rho)
        ### Branch V (MT3 skeleton addendum 2026-09-05): equivariant LATENT
        ### VOLUME TOKENS. K = n_slices*len(offsets)+1 interacting tokens whose
        ### positions are built covariantly from the boundary alone (slice
        ### anchor + c_j * rho_s along the anchor normal, plus the centroid),
        ### so interior queries can read off-surface context while staying
        ### exactly query-independent. Only meaningful with query_independent.
        self.latent_volume_tokens = bool(latent_volume_tokens)
        self.lvt_offsets = tuple(float(c) for c in lvt_offsets)
        if self.latent_volume_tokens:
            ### 2026-09-10 (transfer program, campaign D follow-up): the tokens
            ### are wired into the encoder token set, so they are equally
            ### meaningful with interacting query tokens; the constraint is that
            ### some interior query path must exist to read off-surface context.
            if not (query_independent or query_tokens):
                raise ValueError("latent_volume_tokens requires query_independent=True or query_tokens=True")
            self.lvt_assign = nn.Linear(hidden, n_slices)
            self.lvt_logw = nn.Parameter(torch.zeros(1))
            self.lvt_embed = nn.Sequential(
                nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
        ### WAKE TOKENS (2026-09-10, transfer program). K = len(wake_offsets)
        ### interacting tokens placed on the global vector's axis through the
        ### centering point, downstream at c_k * ell, where ell is the
        ### measure-weighted RMS extent of the surface along it (a body half-length that is
        ### invariant to how the surface was sampled, by the Horvitz-Thompson
        ### weights). Mechanism: far from the body every surface anchor is at
        ### nearly the same distance and direction, so a query's relational
        ### invariants lose resolution exactly in the outer wake, where the
        ### band decomposition of 2026-09-10 locates ISLA's eddy-viscosity
        ### deficit; wake tokens give the slices anchors whose relative geometry
        ### varies along the wake. Their routing weight is a learned fraction of
        ### the total surface measure (the "source_total" convention), so the
        ### measure-scale and source-refinement contracts hold exactly. Built
        ### from the global vector and the weighted surface geometry alone:
        ### covariant, query-independent, discretization-invariant. Off by
        ### default; written for one global vector input.
        self.wake_tokens = bool(wake_tokens)
        self.wake_offsets = tuple(float(c) for c in wake_offsets)
        if self.wake_tokens:
            if not (query_independent or query_tokens):
                raise ValueError("wake_tokens requires query_independent=True or query_tokens=True")
            if not self.wake_offsets:
                raise ValueError("wake_tokens needs at least one offset")
            self.wk_logw = nn.Parameter(torch.zeros(1))
            self.wk_embed = nn.Sequential(
                nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
        ### QUERY TOKENS (boundary->interior exploration, 2026-09-07): interior
        ### query points join the encoder as INTERACTING tokens, carrying the
        ### same {q, n_q, g_k} invariant seeds as the surface tokens (the
        ### query normal must be supplied, e.g. the SDF gradient), a learned
        ### measure weight and a learned token-type offset. This is
        ### GeoTransolver's interior mechanism (queries as tokens) on ISLA's
        ### equivariant routing: exactly SE(3)-covariant, NOT query-independent.
        ### Outputs are returned for the query tokens only.
        self.query_tokens = bool(query_tokens)
        if self.query_tokens:
            if query_independent:
                raise ValueError("query_tokens is an interacting mode; set query_independent=False")
            if use_local_features or raw_coord_channel or scale_conditioning or seed_mode != "invariant":
                raise ValueError("query_tokens supports the plain invariant seed set (plus boundary and global scalars) only")
            self.qt_logw = nn.Parameter(torch.zeros(1))
            self.qt_type = nn.Parameter(torch.zeros(hidden))
        ### Query-token measure weight (audit 2026-09-08). "geometric_mean":
        ### each query's log-weight is qt_logw + the MEAN surface log-weight.
        ### That depends on how the source measure is represented -- splitting
        ### every surface token into two half-weight copies leaves positions,
        ### normals, total area and every integral unchanged but halves each
        ### query weight (3.7%-9.1% output change in the probe). It is kept as
        ### the default only so that trained query-token checkpoints reproduce
        ### exactly. "source_total": the queries' total weight is a learned
        ### fraction exp(qt_logw) of the TOTAL source measure, split equally
        ### over the queries -- invariant to any re-representation of the same
        ### discrete measure, and it scales like an area, so the similarity
        ### gauge scale contract still holds. The successor recipe should use
        ### "source_total".
        if query_mass not in ("geometric_mean", "source_total"):
            raise ValueError(f"unknown query_mass {query_mass!r}")
        if query_mass != "geometric_mean" and not (self.query_tokens or support_tokens):
            raise ValueError("query_mass requires query_tokens=True or support_tokens=True")
        self.query_mass = query_mass
        ### SUPPORT TOKENS (transfer program D1, 2026-09-09; audit agenda
        ### sec-direction-support): a problem-derived SUPPORT set (e.g. a fixed
        ### per-case sample of interior points with their signed distance) joins
        ### the encoder as interacting tokens exactly like query tokens do, while
        ### the requested output points are decoded PASSIVELY through the read
        ### blocks (query_independent=True). The prediction at a query therefore
        ### cannot depend on which other queries are requested, only on the
        ### case (surface + support), which is the deployment contract the
        ### interacting query-token configuration lacks. The support tokens'
        ### total routing weight follows `query_mass` over the support set
        ### ("source_total" recommended: invariant to refinement of the source
        ### measure).
        self.support_tokens = bool(support_tokens)
        if self.support_tokens:
            if not query_independent:
                raise ValueError("support_tokens requires query_independent=True (passive read blocks decode the queries)")
            if self.query_tokens or self.latent_volume_tokens or self.n_anchors or wake_tokens:
                raise ValueError("support_tokens excludes query_tokens, latent_volume_tokens, wake_tokens and n_anchors")
            if use_local_features or raw_coord_channel or scale_conditioning or seed_mode != "invariant":
                raise ValueError("support_tokens supports the plain invariant seed set (plus boundary and global scalars) only")
            self.sp_logw = nn.Parameter(torch.zeros(1))
            self.sp_type = nn.Parameter(torch.zeros(hidden))
        ### Optional per-query scalar inputs for the query tokens (e.g. the
        ### signed distance to the wall, which GeoTransolver's volume
        ### configuration receives at every interior point). Scalars are
        ### invariants, so every covariance contract is untouched; with
        ### query_scalar_scale="length" each scalar s is divided by the gauge
        ### and entered as [s, sign(s) log(|s|+eps)], keeping geometric-scale
        ### equivariance when the similarity gauge is on.
        self.n_query_scalars = int(n_query_scalars)
        self.query_scalar_scale = query_scalar_scale
        if self.n_query_scalars:
            if not (self.query_tokens or query_independent):
                raise ValueError("n_query_scalars requires query_tokens=True or query_independent=True")
            if query_scalar_scale not in ("length", "none"):
                raise ValueError(f"unknown query_scalar_scale {query_scalar_scale!r}")
            width = 2 * self.n_query_scalars if query_scalar_scale == "length" else self.n_query_scalars
            if self.query_tokens:
                self.qt_scalar_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
            if query_independent:
                ### Passive queries take the same per-query scalars (e.g. the
                ### signed distance) through their own embedding; the read
                ### blocks then see an input matched to the interacting arm's.
                self.rq_scalar_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
            if self.support_tokens:
                self.sp_scalar_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
        ### Optional local surface-patch features on the query tokens: the
        ### same measure-weighted Gaussian patch integrals of the SURFACE
        ### sample around each query that the passive decoder can use
        ### (_local_invariants_at), at radii in gauge units, entered
        ### additively through their own embedding. Invariant by
        ### construction (integrals of equivariant vectors projected on the
        ### query normal and the global vectors), so every covariance contract holds.
        ### Target: the eddy-viscosity deficit to GeoTransolver-volume, whose
        ### six-radius local features are the one input class ISLA lacked.
        self.query_local_features = bool(query_local_features)
        self.query_local_radii = tuple(float(x) for x in query_local_radii)
        if self.query_local_features:
            if not self.query_tokens:
                raise ValueError("query_local_features requires query_tokens=True")
            self.qt_local_embed = nn.Sequential(
                nn.Linear(7 * len(self.query_local_radii), hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
        ### QTDENS (2026-09-09, notebook #sec-nb-qt-density-prereg): two flag-gated
        ### query-side channels that read the INTERIOR sample itself, the input
        ### class GeoTransolver-volume's local features aggregate over. Both are
        ### exact invariants of the gauge-normalized query cloud (distances in
        ### gauge units, dot products of equivariant unit vectors), so every
        ### covariance contract holds; both make the prediction depend on the
        ### query cloud, as the query-token mode already does.
        ### (a) density: the number of other queries within query_density_radius
        ###     (gauge units), entered as [log(1+c), log(1+c) - log(n_queries)].
        ###     A pure sampling-density (mesh-scale) proxy with no physics.
        ### (b) neighbours: mean over the k nearest other queries of the pair
        ###     invariants {|dr|, log|dr|, dr_hat.n_q, dr_hat.g_k, n_j.n_q, n_j.g_k,
        ###     (s_j - s_q)/gauge if query scalars are given} plus the k-th
        ###     neighbour distance and its log.
        self.query_density_feature = bool(query_density_feature)
        self.query_density_radius = float(query_density_radius)
        self.query_neighbor_features = bool(query_neighbor_features)
        self.query_neighbor_k = int(query_neighbor_k)
        if self.query_density_feature:
            if not self.query_tokens:
                raise ValueError("query_density_feature requires query_tokens=True")
            self.qt_density_embed = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if self.query_neighbor_features:
            if not self.query_tokens:
                raise ValueError("query_neighbor_features requires query_tokens=True")
            n_nbr = 4 + 2 * K + (1 if self.n_query_scalars else 0) + 2
            self.qt_neighbor_embed = nn.Sequential(nn.Linear(n_nbr, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if odd_head:
            self.N_ODD = 7
            self.odd_assign = nn.Linear(hidden, n_slices)
            self.odd_gate = nn.Linear(hidden, out_vectors * 2 * self.N_ODD)
            ### W2' (2026-09-02): W2 collapsed to a near-mean predictor
            ### (pressure 0.77). Zero-init the odd gate so training starts as
            ### the stable true5 head and the odd channels grow from zero, and
            ### bound the pseudoscalars to [-1, 1] by normalizing the anchor
            ### vectors before the triple products.
            nn.init.zeros_(self.odd_gate.weight)
            nn.init.zeros_(self.odd_gate.bias)


    @staticmethod
    def _dots(x, g_hat):
        """x (B, N, 3) against the K unit global vectors g_hat (B, N, K, 3) -> (B, N, K).
        Multiply-and-sum rather than einsum so autocast leaves the geometry in fp32."""
        return (x[:, :, None, :] * g_hat).sum(-1)

    def _seed_invariants(self, mag, hat, n_hat, g_hat):
        """The per-token seed invariants of {r, n, g_k}: |r|, log|r|, rhat.g_k (K),
        rhat.n, n.g_k (K) -- or the K terms n.g_k alone in the relative frame,
        where r has no origin (an empty tensor when K = 0)."""
        if self.relative_frame:
            return self._dots(n_hat, g_hat)
        return torch.cat(
            [
                mag,
                torch.log(mag),
                self._dots(hat, g_hat),
                (hat * n_hat).sum(-1, keepdim=True),
                self._dots(n_hat, g_hat),
            ],
            dim=-1,
        )

    def _with_empty_boundary_data(self, inv, n_tokens: int):
        """Boundary-condition data (``boundary_scalars``) belong to boundary cells;
        tokens that are not boundary cells (support tokens, interior query tokens,
        passive interior queries) carry zeros in that channel so the shared seed
        layout is kept (GLOBAL INPUTS, 2026-09-14: the Dirichlet-data channel must
        reach the interior paths for a boundary-value problem to be posed)."""
        if not self.n_boundary_scalars:
            return inv
        return torch.cat([inv, inv.new_zeros(inv.shape[0], n_tokens, self.n_boundary_scalars)], dim=-1)

    def _with_global_scalars(self, inv, g_scalars, n_tokens: int):
        """Append the global scalar inputs (B, S) to every token's seed features;
        substitute the constant seed when the feature set is empty."""
        b = inv.shape[0]
        if g_scalars is not None and g_scalars.shape[-1]:
            inv = torch.cat([inv, g_scalars[:, None, :].expand(b, n_tokens, -1).to(inv.dtype)], dim=-1)
        if self.constant_seed:
            inv = inv.new_ones(b, n_tokens, 1)
        return inv

    def _local_invariants_at(self, q_r, q_n, q_g, src_r, src_n, log_w, radii=None,
                             normalize_weights=False):
        """Query-passive variant: patch integrals of the SOURCE sample
        evaluated at arbitrary query positions (radii default to
        ``self.local_radii``; the query-token channel passes its own and
        normalizes the measure weights to fractions of the total surface
        measure, so that log(mass) is invariant to geometric scale)."""
        b, nq, _ = q_r.shape
        w = torch.exp(log_w.squeeze(-1))
        if normalize_weights:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        feats = []
        chunk = 4096
        for rho in (self.local_radii if radii is None else radii):
            outs = []
            for i0 in range(0, nq, chunk):
                ri = q_r[:, i0 : i0 + chunk]
                d2 = torch.cdist(ri, src_r).square()
                k = torch.exp(-d2 / (rho * rho)) * w[:, None, :]
                mass = k.sum(-1, keepdim=True).clamp_min(self.eps)
                nbar = torch.einsum("bcn,bnk->bck", k, src_n) / mass
                delta = (torch.einsum("bcn,bnk->bck", k, src_r) / mass) - ri
                ni = q_n[:, i0 : i0 + chunk]
                gi = q_g[:, i0 : i0 + chunk]
                outs.append(
                    torch.cat(
                        [
                            (nbar * ni).sum(-1, keepdim=True),
                            self._dots(nbar, gi),
                            nbar.norm(dim=-1, keepdim=True),
                            (delta * ni).sum(-1, keepdim=True) / rho,
                            self._dots(delta, gi) / rho,
                            delta.norm(dim=-1, keepdim=True) / rho,
                            torch.log(mass),
                        ],
                        dim=-1,
                    )
                )
            feats.append(torch.cat(outs, dim=1))
        return torch.cat(feats, dim=-1)

    def _local_invariants(
        self,
        r: Float[torch.Tensor, "batch tokens 3"],
        n_hat: Float[torch.Tensor, "batch tokens 3"],
        g_hat: Float[torch.Tensor, "batch tokens vectors 3"],
        log_w: Float[torch.Tensor, "batch tokens 1"],
        normalize_weights: bool = False,
    ) -> Float[torch.Tensor, "batch tokens feats"]:
        """Measure-weighted Gaussian patch integrals at fixed physical radii
        (5 + 2K invariants per radius; 7 for one global vector).

        Exactly equivariant (integrals of equivariant vectors, projected on
        n_i and the g_k); unbiased under HT sampling via the measure weights;
        row-chunked so the pairwise kernel never materializes at full size.

        With ``normalize_weights`` the weights are fractions of the total
        measure, so log(mass) is invariant to geometric scale (areas x k^2);
        the radii are already in gauge units. The caller passes
        ``similarity_gauge`` here (audit 2026-09-08): the raw-weight formula
        is kept for the non-gauge path so that the trained DrivAerML
        "local features" arm reproduces bit-identically, and only the gauge
        path -- whose scale contract the raw log(mass) broke -- changes.
        """
        b, n, _ = r.shape
        w = torch.exp(log_w.squeeze(-1))  # (B, N) relative measure weights
        if normalize_weights:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        feats = []
        chunk = 4096
        for rho in self.local_radii:
            outs = []
            for i0 in range(0, n, chunk):
                ri = r[:, i0 : i0 + chunk]  # (B, C, 3)
                d2 = torch.cdist(ri, r).square()  # (B, C, N)
                k = torch.exp(-d2 / (rho * rho)) * w[:, None, :]
                mass = k.sum(-1, keepdim=True).clamp_min(self.eps)  # (B, C, 1)
                nbar = torch.einsum("bcn,bnk->bck", k, n_hat) / mass
                delta = (torch.einsum("bcn,bnk->bck", k, r) / mass) - ri
                ni = n_hat[:, i0 : i0 + chunk]
                gi = g_hat[:, i0 : i0 + chunk]
                outs.append(
                    torch.cat(
                        [
                            (nbar * ni).sum(-1, keepdim=True),
                            self._dots(nbar, gi),
                            nbar.norm(dim=-1, keepdim=True),
                            (delta * ni).sum(-1, keepdim=True) / rho,
                            self._dots(delta, gi) / rho,
                            delta.norm(dim=-1, keepdim=True) / rho,
                            torch.log(mass),
                        ],
                        dim=-1,
                    )
                )
            feats.append(torch.cat(outs, dim=1))
        return torch.cat(feats, dim=-1)

    def _query_cloud_invariants(self, q_r, q_nhat, q_g, qs_raw=None, chunk: int = 2048):
        """QTDENS channels from the gauge-normalized query cloud alone (see __init__).
        Returns (density (B,Q,2), neighbours (B,Q,n_nbr)); either may be unused."""
        bq, nq, _ = q_r.shape
        k = min(self.query_neighbor_k, max(nq - 1, 1))
        rho = self.query_density_radius
        dens_out, nbr_out = [], []
        for i0 in range(0, nq, chunk):
            qi = q_r[:, i0:i0 + chunk]
            d2 = torch.cdist(qi.float(), q_r.float()).square()  # (B, c, Q)
            ar = torch.arange(i0, min(i0 + chunk, nq), device=q_r.device)
            d2[:, torch.arange(len(ar), device=q_r.device), ar] = float("inf")  # exclude self
            count = (d2 < rho * rho).sum(-1, keepdim=True).to(q_r.dtype)
            logc = torch.log1p(count)
            dens_out.append(torch.cat([logc, logc - math.log(nq)], dim=-1))
            if self.query_neighbor_features:
                dk, idx = torch.topk(d2, k, dim=-1, largest=False)  # (B, c, k)
                nb_r = torch.gather(q_r[:, None].expand(bq, len(ar), nq, 3), 2, idx[..., None].expand(bq, len(ar), k, 3))
                nb_n = torch.gather(q_nhat[:, None].expand(bq, len(ar), nq, 3), 2, idx[..., None].expand(bq, len(ar), k, 3))
                rel = nb_r - qi[:, :, None, :]
                dist = rel.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                rel_hat = rel / dist
                nq_e = q_nhat[:, i0:i0 + chunk, None, :]
                g_e = q_g[:, i0:i0 + chunk]  # (B, c, K, 3)
                feats = [dist, torch.log(dist), (rel_hat * nq_e).sum(-1, keepdim=True),
                         (rel_hat[:, :, :, None, :] * g_e[:, :, None, :, :]).sum(-1),
                         (nb_n * nq_e).sum(-1, keepdim=True),
                         (nb_n[:, :, :, None, :] * g_e[:, :, None, :, :]).sum(-1)]
                if qs_raw is not None:
                    nb_s = torch.gather(qs_raw[:, None, :, :1].expand(bq, len(ar), nq, 1), 2, idx[..., None])
                    feats.append(nb_s - qs_raw[:, i0:i0 + chunk, None, :1])
                per = torch.cat(feats, dim=-1).mean(dim=2)  # (B, c, 6[+1])
                dk_k = dk[..., -1:].clamp_min(self.eps * self.eps).sqrt().to(q_r.dtype)
                nbr_out.append(torch.cat([per, dk_k, torch.log(dk_k)], dim=-1))
        dens = torch.cat(dens_out, dim=1)
        nbr = torch.cat(nbr_out, dim=1) if nbr_out else None
        return dens, nbr

    def forward(
        self,
        *,
        points: Float[torch.Tensor, "batch tokens 3"],
        normals: Float[torch.Tensor, "batch tokens 3"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
        global_vectors: Float[torch.Tensor, "batch vectors 3"] | None = None,
        global_scalars: Float[torch.Tensor, "batch scalars"] | None = None,
        boundary_scalars: Float[torch.Tensor, "batch tokens n_bscalars"] | None = None,
        query_points: Float[torch.Tensor, "batch queries 3"] | None = None,
        query_normals: Float[torch.Tensor, "batch queries 3"] | None = None,
        query_scalars: Float[torch.Tensor, "batch queries n_qscalars"] | None = None,
        support_points: Float[torch.Tensor, "batch support 3"] | None = None,
        support_normals: Float[torch.Tensor, "batch support 3"] | None = None,
        support_scalars: Float[torch.Tensor, "batch support n_qscalars"] | None = None,
        frame_center: Float[torch.Tensor, "batch 3"] | None = None,
        frame_scale: Float[torch.Tensor, " batch"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens out_dim"]:
        if points.ndim == 2:
            points = points[None]
            normals = normals[None]
        b, n, _ = points.shape
        ### Global vector inputs: (B, K, 3), or any layout with B*K*3 or K*3
        ### elements ((K, 3), and for K = 1 (B, 3), (B, 1, 3) or (3,)); a
        ### single set is shared across the batch. Each is normalized to a unit
        ### direction; the magnitude, if it means anything, is a global scalar.
        K = self.n_global_vectors
        if K == 0:
            if global_vectors is not None and global_vectors.numel():
                raise ValueError("this model was built with n_global_vectors=0; pass no global_vectors")
            g_unit = points.new_zeros(b, 0, 3)
        else:
            if global_vectors is None:
                raise ValueError(f"n_global_vectors={K} needs global_vectors of shape (batch, {K}, 3)")
            if global_vectors.numel() % (3 * K):
                raise ValueError(
                    f"global_vectors has {global_vectors.numel()} elements, not a multiple of {K} vectors x 3"
                )
            gv = global_vectors.reshape(-1, K, 3).to(points.dtype)
            if gv.shape[0] == 1 and b != 1:
                gv = gv.expand(b, K, 3)
            elif gv.shape[0] != b:
                raise ValueError(f"global_vectors gives {gv.shape[0]} sets of {K} vectors for a batch of {b}")
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, K, 3)
        g_hat = g_unit[:, None].expand(b, n, K, 3)
        ### Global scalar inputs: (B, S) or (S,) shared across the batch.
        S = self.n_global_scalars
        if S == 0:
            if global_scalars is not None and global_scalars.numel():
                raise ValueError("this model was built with n_global_scalars=0; pass no global_scalars")
            g_scalars = None
        else:
            if global_scalars is None:
                raise ValueError(f"n_global_scalars={S} needs global_scalars of shape (batch, {S})")
            g_scalars = global_scalars.reshape(-1, S).to(points.dtype)
            if g_scalars.shape[0] == 1 and b != 1:
                g_scalars = g_scalars.expand(b, S)
            elif g_scalars.shape[0] != b:
                raise ValueError(f"global_scalars gives {g_scalars.shape[0]} rows for a batch of {b}")

        ### Similarity reduction: center by the plain mean, scale by L_ref.
        if self.similarity_gauge or self.center_mode == "measure":
            w_raw = (
                measure_weights.reshape(b, n, 1).to(points.dtype)
                if measure_weights is not None
                else torch.ones(b, n, 1, dtype=points.dtype, device=points.device)
            )
            w_n = w_raw / w_raw.sum(dim=1, keepdim=True).clamp_min(self.eps)
        if self.relative_frame:
            center = points.new_zeros(b, 1, 3)  # RELFRAME: no frame origin
        elif self.center_mode == "global":
            if frame_center is None:
                raise ValueError("center_mode='global' needs frame_center (B, 3); the config must supply it")
            center = frame_center.reshape(b, 1, 3).to(points.dtype)
        elif self.similarity_gauge or self.center_mode == "measure":
            center = (w_n * points).sum(dim=1, keepdim=True)
        else:
            center = points.mean(dim=1, keepdim=True)
        if self.scale_mode == "global":
            if frame_scale is None:
                raise ValueError("scale_mode='global' needs frame_scale (B,); the config must supply it")
            gauge = frame_scale.reshape(b, 1, 1).to(points.dtype)
        elif self.scale_mode == "total_measure":
            if measure_weights is None:
                raise ValueError("scale_mode='total_measure' needs measure_weights")
            gauge = measure_weights.reshape(b, n, 1).to(points.dtype).sum(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
        elif self.similarity_gauge:
            gauge = (
                (w_n * (points - center).square().sum(-1, keepdim=True)).sum(dim=1, keepdim=True)
            ).sqrt().clamp_min(self.eps)  # (B,1,1) weighted RMS radius
        else:
            gauge = self.reference_length
        r = (points - center) / gauge
        r_mag = r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        r_hat = r / r_mag
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        if measure_weights is None or not self.use_measure_weights:
            log_w = points.new_zeros(b, n, 1)
        else:
            measure_weights = measure_weights.reshape(b, n)
            log_w = torch.log(measure_weights.clamp_min(self.eps))[..., None]
            if self.measure_weight_power != 1.0:
                log_w = log_w * self.measure_weight_power

        if self.seed_mode == "raw":
            invariants = torch.cat([r, n_hat, g_hat.reshape(b, n, 3 * K)], dim=-1)
        else:
            invariants = self._seed_invariants(r_mag, r_hat, n_hat, g_hat)
        if self.use_local_features:
            invariants = torch.cat(
                [invariants, self._local_invariants(r, n_hat, g_hat, log_w,
                                                    normalize_weights=self.similarity_gauge)],
                dim=-1,
            )
        if self.n_boundary_scalars:
            bs = boundary_scalars.reshape(b, n, self.n_boundary_scalars)
            invariants = torch.cat([invariants, bs.to(invariants.dtype)], dim=-1)
        if self.raw_coord_channel:
            invariants = torch.cat([invariants, r, n_hat], dim=-1)
        if self.scale_conditioning:
            raw_scale = (points - center).norm(dim=-1).mean(dim=1, keepdim=True)
            log_s = torch.log(raw_scale.clamp_min(self.eps))[..., None]
            invariants = torch.cat(
                [invariants, log_s.expand(b, n, 1)], dim=-1
            )
        invariants = self._with_global_scalars(invariants, g_scalars, n)
        h = self.embed(invariants)

        ### n_boundary counts the SURFACE tokens only: the surface measure
        ### statistics that the support, query and wake tokens' routing
        ### weights refer to, and the surface-only local invariants, must not
        ### see the constructed context tokens appended below.
        n_boundary = n
        if self.latent_volume_tokens and (self.query_independent or self.query_tokens):
            ### pre-encoder geometric slice assignment -> anchors z0, m0, rho0
            a0 = _softmax_over_points(self.lvt_assign(h) + log_w, self.fast_point_softmax)  # (B,N,S)
            z0 = torch.einsum("bns,bnc->bsc", a0, r)
            m0 = torch.einsum("bns,bnc->bsc", a0, n_hat)
            m0 = m0 / m0.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            d2 = (r[:, :, None, :] - z0[:, None, :, :]).square().sum(-1)  # (B,N,S)
            rho0 = torch.einsum("bns,bns->bs", a0, d2).clamp_min(self.eps).sqrt()  # (B,S)
            S = z0.shape[1]
            pos, mtok, ctok, rtok = [], [], [], []
            for c in self.lvt_offsets:
                pos.append(z0 + c * rho0[..., None] * m0)
                mtok.append(m0)
                ctok.append(torch.full_like(rho0, c))
                rtok.append(rho0)
            pos.append(torch.zeros_like(z0[:, :1]))            # centroid token
            mtok.append(g_hat[:, :1, 0])                       # covariant placeholder normal (one global vector)
            ctok.append(torch.zeros_like(rho0[:, :1]))
            rtok.append(rho0.mean(dim=1, keepdim=True))
            p_l = torch.cat(pos, dim=1)                        # (B,K,3)
            m_l = torch.cat(mtok, dim=1)
            c_l = torch.cat(ctok, dim=1)[..., None]
            rho_l = torch.cat(rtok, dim=1)[..., None]
            K_l = p_l.shape[1]
            d_l = g_hat[:, :1, 0].expand(b, K_l, 3)
            p_mag = p_l.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            p_hat = p_l / p_mag
            inv_l = torch.cat(
                [
                    p_mag, torch.log(p_mag),
                    (p_hat * d_l).sum(-1, keepdim=True),
                    (p_hat * m_l).sum(-1, keepdim=True),
                    (m_l * d_l).sum(-1, keepdim=True),
                    c_l, rho_l,
                ],
                dim=-1,
            )
            h = torch.cat([h, self.lvt_embed(inv_l)], dim=1)
            r = torch.cat([r, p_l], dim=1)
            n_hat = torch.cat([n_hat, m_l], dim=1)
            g_hat = torch.cat([g_hat, d_l[:, :, None]], dim=1)
            lvt_w = self.lvt_logw.to(log_w.dtype)
            if self.query_tokens:
                ### Query-token mode (2026-09-10): the volume tokens' total weight
                ### is a learned fraction of the total surface measure, so the
                ### measure-scale and refinement contracts hold exactly. The
                ### passive path keeps its original absolute weight so that
                ### existing checkpoints evaluate unchanged.
                lvt_w = lvt_w + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True) - math.log(K_l)
            log_w = torch.cat([log_w, lvt_w.expand(b, K_l, 1)], dim=1)
            n = n + K_l

        if self.wake_tokens and (self.query_independent or self.query_tokens):
            ### Wake tokens (see __init__): context downstream of the body along
            ### the (single) global vector input, built from the surface tokens'
            ### weighted geometry.
            r_b, logw_b = r[:, :n_boundary], log_w[:, :n_boundary]
            d_b = g_unit[:, :1, :]  # (B,1,3) the one global vector
            s_b = (r_b * d_b).sum(-1, keepdim=True)  # (B,N,1) coordinate along it
            w_b = torch.softmax(logw_b, dim=1)  # normalized measure, scale-free
            s_mean = (w_b * s_b).sum(dim=1, keepdim=True)  # (B,1,1)
            ell = ((w_b * (s_b - s_mean).square()).sum(dim=1, keepdim=True)).sqrt().clamp_min(self.eps)
            Kw = len(self.wake_offsets)
            c_w = torch.tensor(self.wake_offsets, dtype=r.dtype, device=r.device).view(1, Kw, 1)
            p_w = (s_mean + c_w * ell) * d_b  # (B,Kw,3) on the global vector's axis
            m_w = d_b.expand(b, Kw, 3)  # covariant placeholder normal
            d_w = d_b.expand(b, Kw, 3)
            p_mag_w = p_w.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            p_hat_w = p_w / p_mag_w
            inv_w = torch.cat(
                [
                    p_mag_w, torch.log(p_mag_w),
                    (p_hat_w * d_w).sum(-1, keepdim=True),
                    (p_hat_w * m_w).sum(-1, keepdim=True),
                    (m_w * d_w).sum(-1, keepdim=True),
                    c_w.expand(b, Kw, 1), ell.expand(b, Kw, 1),
                ],
                dim=-1,
            )
            h = torch.cat([h, self.wk_embed(inv_w.to(h.dtype))], dim=1)
            r = torch.cat([r, p_w], dim=1)
            n_hat = torch.cat([n_hat, m_w], dim=1)
            g_hat = torch.cat([g_hat, d_w[:, :, None]], dim=1)
            ### total wake weight = learned fraction of the total surface measure
            wk_logw = (self.wk_logw.to(log_w.dtype)
                       + torch.logsumexp(logw_b, dim=1, keepdim=True) - math.log(Kw))
            log_w = torch.cat([log_w, wk_logw.expand(b, Kw, 1)], dim=1)
            n = n + Kw

        n_surface = n
        if self.support_tokens and support_points is not None:
            ### Support tokens (see __init__): interacting interior tokens that
            ### are a function of the case, not of the requested queries.
            if support_normals is None:
                raise ValueError("support_tokens needs support_normals (e.g. the SDF gradient at each support point)")
            s_pts = support_points
            bs_, ns_, _ = s_pts.shape
            s_r = (s_pts - center) / gauge
            s_mag = s_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            s_rhat = s_r / s_mag
            s_nhat = support_normals / support_normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            s_g = g_unit[:, None].expand(bs_, ns_, K, 3)
            s_inv = self._with_global_scalars(
                self._with_empty_boundary_data(self._seed_invariants(s_mag, s_rhat, s_nhat, s_g), ns_), g_scalars, ns_)
            h_s = self.embed(s_inv) + self.sp_type.to(h.dtype)
            if self.n_query_scalars:
                if support_scalars is None:
                    raise ValueError("n_query_scalars > 0 with support_tokens needs support_scalars")
                ss = support_scalars.reshape(bs_, ns_, self.n_query_scalars).to(s_inv.dtype)
                if self.query_scalar_scale == "length":
                    ss = ss / gauge
                    ss = torch.cat([ss, torch.sign(ss) * torch.log(ss.abs() + self.eps)], dim=-1)
                h_s = h_s + self.sp_scalar_embed(ss).to(h_s.dtype)
            elif support_scalars is not None:
                raise ValueError("support_scalars given but n_query_scalars == 0")
            if self.query_mass == "source_total":
                s_logw = (self.sp_logw.to(log_w.dtype)
                          + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True) - math.log(ns_))
            else:
                s_logw = self.sp_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(dim=1, keepdim=True)
            h = torch.cat([h, h_s], dim=1)
            r = torch.cat([r, s_r], dim=1)
            n_hat = torch.cat([n_hat, s_nhat], dim=1)
            g_hat = torch.cat([g_hat, s_g], dim=1)
            log_w = torch.cat([log_w, s_logw.expand(b, ns_, 1)], dim=1)
            n = n + ns_
        qt_active = self.query_tokens and query_points is not None
        if qt_active:
            ### Interior queries as interacting tokens (see __init__).
            if query_normals is None:
                raise ValueError("query_tokens needs query_normals (e.g. the SDF gradient at each query)")
            q_pts = query_points
            bq, nq, _ = q_pts.shape
            q_r = (q_pts - center) / gauge
            q_mag = q_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_rhat = q_r / q_mag
            q_nhat = query_normals / query_normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_g = g_unit[:, None].expand(bq, nq, K, 3)
            q_inv = self._with_global_scalars(
                self._with_empty_boundary_data(self._seed_invariants(q_mag, q_rhat, q_nhat, q_g), nq), g_scalars, nq)
            h_q = self.embed(q_inv) + self.qt_type.to(h.dtype)
            if self.n_query_scalars:
                if query_scalars is None:
                    raise ValueError("n_query_scalars > 0 needs query_scalars")
                qs = query_scalars.reshape(bq, nq, self.n_query_scalars).to(q_inv.dtype)
                if self.query_scalar_scale == "length":
                    qs = qs / gauge
                    qs = torch.cat([qs, torch.sign(qs) * torch.log(qs.abs() + self.eps)], dim=-1)
                h_q = h_q + self.qt_scalar_embed(qs).to(h_q.dtype)
            elif query_scalars is not None:
                raise ValueError("query_scalars given but n_query_scalars == 0")
            if self.query_local_features:
                ### Patch integrals of the surface sample around each query
                ### (surface tokens only: the first n_surface entries).
                q_loc = self._local_invariants_at(
                    q_r, q_nhat, q_g, r[:, :n_boundary], n_hat[:, :n_boundary], log_w[:, :n_boundary],
                    radii=self.query_local_radii, normalize_weights=True,
                )
                h_q = h_q + self.qt_local_embed(q_loc.to(q_inv.dtype)).to(h_q.dtype)
            if self.query_density_feature or self.query_neighbor_features:
                qs_raw = None
                if self.n_query_scalars and query_scalars is not None:
                    qs_raw = query_scalars.reshape(bq, nq, self.n_query_scalars).to(q_r.dtype) / gauge
                dens, nbr = self._query_cloud_invariants(q_r, q_nhat, q_g, qs_raw)
                if self.query_density_feature:
                    h_q = h_q + self.qt_density_embed(dens.to(q_inv.dtype)).to(h_q.dtype)
                if self.query_neighbor_features:
                    h_q = h_q + self.qt_neighbor_embed(nbr.to(q_inv.dtype)).to(h_q.dtype)
            h = torch.cat([h, h_q], dim=1)
            r = torch.cat([r, q_r], dim=1)
            n_hat = torch.cat([n_hat, q_nhat], dim=1)
            g_hat = torch.cat([g_hat, q_g], dim=1)
            ### The query tokens' routing weight is learned RELATIVE to the
            ### surface measure (mean surface log-weight), so that rescaling
            ### the geometry (areas x k^2) shifts every token's log-weight by
            ### the same 2 log k and the similarity-gauge scale contract holds
            ### exactly; an absolute learned weight broke it (2026-09-07).
            ### "source_total" (see __init__) ties the queries' total weight to
            ### the total surface measure instead of the per-token mean.
            if self.query_mass == "source_total":
                q_logw = (self.qt_logw.to(log_w.dtype)
                          + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True) - math.log(nq))
            else:
                q_logw = self.qt_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(dim=1, keepdim=True)
            log_w = torch.cat([log_w, q_logw.expand(b, nq, 1)], dim=1)
            n = n + nq

        if not (self.query_independent and 0 < self.n_anchors < n):
            for block in self.blocks:
                h = block(h, log_w, r, n_hat, g_hat, self.eps)

        ### The odd head builds its slice anchors from the SOURCE tokens
        ### (src_r, src_n, src_logw, and h); the branches below overwrite
        ### r_hat/n_hat/g_out/n with the query-side values, so the source
        ### tensors are captured here (audit 2026-09-08: the head used to read
        ### the overwritten n_hat and n and failed whenever the query count
        ### differed from the source count).
        src_r, src_n, src_logw = r, n_hat, log_w
        r_out = r  # positions of the tokens the head reads (RELFRAME radial basis)
        g_out = g_hat  # the global vectors seen by the tokens the head reads
        if qt_active:
            ### Heads read the query tokens only (surface tokens were context).
            h_out = h[:, n_surface:]
            r_hat, n_hat, g_out, b, n = q_rhat, q_nhat, q_g, bq, nq
            r_out = q_r
        elif self.query_independent:
            if 0 < self.n_anchors < n:
                ### v5a4: the interacting core is a random anchor subset in
                ### training (deterministic prefix at eval), so predictions at
                ### non-anchor points are query-independent given the anchors.
                n_anchor = self.n_anchors
                if self.training:
                    idx = torch.randperm(n, device=points.device)[:n_anchor]
                else:
                    idx = torch.arange(n_anchor, device=points.device)
                h = h[:, idx]
                r_enc, n_enc = r[:, idx], n_hat[:, idx]
                g_enc = g_hat[:, idx]
                log_w_enc = log_w[:, idx]
            else:
                r_enc, n_enc, g_enc, log_w_enc = r, n_hat, g_hat, log_w
            ### Final encoder slice states and anchors (read-only for queries).
            if 0 < self.n_anchors < n:
                for block in self.blocks:
                    h = block(h, log_w_enc, r_enc, n_enc, g_enc, self.eps)
            logits = self.final_assign(h)
            a = _softmax_over_points(logits + (log_w_enc if 0 < self.n_anchors < n else log_w), self.fast_point_softmax)
            r_src = r_enc if 0 < self.n_anchors < n else r
            n_src = n_enc if 0 < self.n_anchors < n else n_hat
            z_states = torch.einsum("bns,bnh->bsh", a, h)
            z_pos = torch.einsum("bns,bnc->bsc", a, r_src)
            m_s = torch.einsum("bns,bnc->bsc", a, n_src)
            m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            if query_points is None:
                q_pts, q_nrm = points, normals
            else:
                q_pts = query_points
                if query_normals is not None:
                    q_nrm = query_normals
                elif self.interior_queries:
                    q_nrm = None  # derived from the anchors below
                elif query_points.shape[1] != normals.shape[1]:
                    raise ValueError(
                        "query_points are not the boundary points: pass query_normals (e.g. the SDF "
                        "gradient) or set interior_queries=True to derive a normal from the anchors"
                    )
                else:
                    q_nrm = normals
            q_r = (q_pts - center) / gauge
            q_mag = q_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_rhat = q_r / q_mag
            if q_nrm is None:
                d2 = torch.cdist(q_r, z_pos).square()  # (B, Nq, S)
                a_q = torch.softmax(-d2 / (self.anchor_normal_rho ** 2), dim=-1)
                q_nrm = torch.einsum("bqs,bsc->bqc", a_q, m_s)
            q_nhat = q_nrm / q_nrm.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            bq, nq, _ = q_pts.shape
            q_g = g_unit[:, None].expand(bq, nq, K, 3)
            if self.seed_mode == "raw":
                q_inv = torch.cat([q_r, q_nhat, q_g.reshape(bq, nq, 3 * K)], dim=-1)
            else:
                q_inv = self._seed_invariants(q_mag, q_rhat, q_nhat, q_g)
            if self.use_local_features:
                ### Local integrals read the SOURCE sample -- query-passive.
                q_inv = torch.cat(
                    [q_inv, self._local_invariants_at(q_r, q_nhat, q_g, r, n_hat, log_w,
                                                      normalize_weights=self.similarity_gauge)],
                    dim=-1,
                )
            ### The remaining seed channels must match the encoder's seed
            ### layout (audit 2026-09-08: passive decoding used to fail with a
            ### shape error for every option below). Boundary scalars are
            ### per-boundary-cell data: the queries carry them only when the
            ### queries ARE the boundary points, and zeros otherwise.
            if self.n_boundary_scalars:
                if query_points is None:
                    q_inv = torch.cat([q_inv, bs.to(q_inv.dtype)], dim=-1)
                else:
                    q_inv = self._with_empty_boundary_data(q_inv, nq)
            if self.raw_coord_channel:
                q_inv = torch.cat([q_inv, q_r, q_nhat], dim=-1)
            if self.scale_conditioning:
                q_inv = torch.cat([q_inv, log_s.expand(bq, nq, 1)], dim=-1)
            q_inv = self._with_global_scalars(q_inv, g_scalars, nq)
            q_h = self.embed(q_inv)
            if self.n_query_scalars:
                if query_scalars is None:
                    raise ValueError("n_query_scalars > 0 needs query_scalars")
                qs = query_scalars.reshape(bq, nq, self.n_query_scalars).to(q_inv.dtype)
                if self.query_scalar_scale == "length":
                    qs = qs / gauge
                    qs = torch.cat([qs, torch.sign(qs) * torch.log(qs.abs() + self.eps)], dim=-1)
                q_h = q_h + self.rq_scalar_embed(qs).to(q_h.dtype)
            elif query_scalars is not None:
                raise ValueError("query_scalars given but n_query_scalars == 0")
            src_logw = log_w_enc if 0 < self.n_anchors < n else log_w
            src_r, src_n = r_src, n_src
            ### Support-token mode uses the exactly measure-invariant log-space
            ### kernel (see _kernel_readout); the legacy passive path keeps the
            ### clamped form so its trained checkpoints reproduce.
            logspace = self.support_tokens
            src_w = src_logw.squeeze(-1) if logspace else torch.exp(src_logw.squeeze(-1))
            for rb in self.read_blocks:
                q_h = rb(
                    q_h, q_r, q_nhat, q_g, z_states, z_pos, m_s, self.eps,
                    src_r=r_src, src_h=h, src_w=src_w,
                    local_rho=self.local_readout_rho, kernel_logspace=logspace,
                )
            h_out, r_hat, n_hat, g_out, b, n = q_h, q_rhat, q_nhat, q_g, bq, nq
            r_out = q_r
        else:
            h_out = h
        if self.relative_frame:
            ### RELFRAME head frame: the radial basis vector is the direction from
            ### the token's soft slice anchor (measure-weighted mean of source
            ### positions) instead of from a centroid; translation covariant.
            a_f = _softmax_over_points(self.frame_assign(h) + src_logw, self.fast_point_softmax)
            z_f = torch.einsum("bns,bnc->bsc", a_f, src_r)
            b_f = torch.softmax(self.frame_assign(h_out), dim=-1)
            u = r_out - torch.einsum("bns,bsc->bnc", b_f, z_f).to(r_out.dtype)
            r_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        out = self.head(self.norm_out(h_out))

        scalars = out[..., : self.out_scalars]
        coeffs = out[..., self.out_scalars :].reshape(
            b, n, self.out_vectors, self.n_basis
        )
        ### GLOBE-style expansion: input vectors + spherical complements of
        ### the (r_hat, n_hat) pair and of each (r_hat, g_k) pair. Exactly
        ### equivariant; non-orthogonal inputs span via the complements. For
        ### K = 1 the stacking order is the original [g, n, r, e_th_n, e_ph_n,
        ### e_th_g, e_ph_g] (true5/true7 likewise), so trained heads reproduce.
        gs = [g_out[:, :, k] for k in range(self.n_global_vectors)]
        _, e_th_n, e_ph_n = spherical_basis(r_hat, n_hat, normalize_basis_vectors=False)
        sb = [spherical_basis(r_hat, g, normalize_basis_vectors=False) for g in gs]
        if self.vector_basis == "true5":
            basis = gs + [n_hat, r_hat, e_th_n] + [e_th for _, e_th, _ in sb]
        elif self.vector_basis == "true7":
            basis = (gs + [n_hat, r_hat, e_th_n] + [e_th for _, e_th, _ in sb]
                     + [torch.linalg.cross(e_ph_n, g, dim=-1) for g in gs]
                     + [torch.linalg.cross(e_ph, n_hat, dim=-1) for _, _, e_ph in sb])
        else:
            basis = gs + [n_hat, r_hat, e_th_n, e_ph_n]
            for _, e_th, e_ph in sb:
                basis += [e_th, e_ph]
        basis = torch.stack(basis, dim=-2)  # (B, N, n_basis, 3)
        if self.odd_head and self.vector_basis == "globe7":
            d_hat = gs[0]  # odd head: one global vector (enforced at construction)
            ### per-point soft slice anchor (true vectors, equivariant)
            lg = self.odd_assign(h)
            a_s = _softmax_over_points(lg + src_logw, self.fast_point_softmax)  # slices over source points
            z_s = torch.einsum("bns,bnc->bsc", a_s, src_r)
            m_s = torch.einsum("bns,bnc->bsc", a_s, src_n)
            b_q = torch.softmax(self.odd_assign(h_out), dim=-1)  # point over slices
            ### autocast may emit bf16 from einsum; keep geometry in fp32.
            z_q = torch.einsum("bns,bsc->bnc", b_q, z_s).to(r_hat.dtype)
            m_q = torch.einsum("bns,bsc->bnc", b_q, m_s).to(r_hat.dtype)
            z_q = z_q / z_q.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            m_q = m_q / m_q.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            def trip(u, v, w):
                return (u * torch.linalg.cross(v, w, dim=-1)).sum(-1, keepdim=True)
            pseudo = torch.cat(
                [
                    trip(r_hat, n_hat, d_hat),
                    trip(r_hat, n_hat, z_q), trip(r_hat, n_hat, m_q),
                    trip(n_hat, d_hat, z_q), trip(n_hat, d_hat, m_q),
                    trip(r_hat, d_hat, z_q), trip(r_hat, d_hat, m_q),
                ],
                dim=-1,
            )  # (B, N, K) all parity-odd, rotation-invariant
            g = self.odd_gate(self.norm_out(h_out)).reshape(
                b, n, self.out_vectors, 2, self.N_ODD
            )
            ### mixed precision: pseudo is fp32 geometry, g may be bf16 under
            ### autocast; contract in fp32 and cast back to the head's dtype.
            odd_coeff = torch.einsum(
                "bnvjk,bnk->bnvj", g.to(pseudo.dtype), pseudo
            ).to(coeffs.dtype)  # (B,N,V,2)
            coeffs = coeffs.clone()
            coeffs[..., 4] = odd_coeff[..., 0]
            coeffs[..., 6] = odd_coeff[..., 1]
        if self.parity_fix and self.vector_basis == "globe7":
            p_odd = (
                r_hat * torch.linalg.cross(n_hat, gs[0], dim=-1)
            ).sum(-1)[..., None, None]  # (B, N, 1, 1), parity-odd invariant
            if self.parity_gate_scale > 0:
                p_odd = torch.tanh(p_odd / self.parity_gate_scale)
            gate = torch.ones_like(coeffs[..., :1, :]).expand_as(coeffs).clone()
            gate[..., 4] = p_odd[..., 0]
            gate[..., 6] = p_odd[..., 0]
            coeffs = coeffs * gate
        vectors = torch.einsum("bnvk,bnkc->bnvc", coeffs, basis)

        return torch.cat(
            [scalars, vectors.reshape(b, n, self.out_vectors * 3)], dim=-1
        )


#: Backward-compatible alias for the architecture's previous name.
MeshTransformer2 = ISLA
