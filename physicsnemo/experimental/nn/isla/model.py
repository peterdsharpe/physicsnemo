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
frame is re-attached only at the vector heads, so exact rotation and
translation covariance is paid once at the network's edges instead of in
every layer.

**Reference configuration** (the defaults): ``frame_mode="relative"`` with
``scale_mode="total_measure"``. Positions enter only as point-to-anchor
differences and the length scale is the square root of the total quadrature
measure of the sample, so no centroid, no sample statistic and no
per-dataset reference length appears anywhere in the forward pass and the
frame carries no information about where the mesher placed its cells.

**Frame variants.** ``frame_mode="centered"`` centers by the plain mean of
the sampled points and scales by the constant ``reference_length``
(``scale_mode="reference_length"``); ``similarity_gauge=True`` on top of it
derives the centroid and the length scale from the measure-weighted geometry
and makes the model additionally equivariant to geometric scale.

**Interior modes.** ``query_tokens=True`` lets interior query points join
the encoder as interacting tokens (their normal, e.g. the SDF gradient, must
be supplied); ``query_independent=True`` decodes queries passively through
read-only blocks, so a prediction at one point does not depend on which
other points are queried; ``support_tokens=True`` (with
``query_independent``) adds a problem-derived interior support set as
interacting tokens while the requested queries stay passive. Per-query
scalars (``n_query_scalars``, e.g. the signed distance) and per-cell
boundary scalars (``n_boundary_scalars``) are invariants and widen the seed
features without touching any covariance contract.

**Global inputs.** Global vector inputs are normalized to unit directions
inside the model (a direction is what the invariants consume); a physically
meaningful magnitude belongs among the global scalar inputs. With one
global vector and no global scalars (the defaults) the network is
parameter-for-parameter the former single-vector model.

**Kernels.** ``geo_kernel="fused"`` evaluates the per-layer geometry region
in a Triton kernel (CUDA, one global vector); ``geo_checkpoint`` rebuilds the
eager region's intermediates in backward instead of storing them. Both are
exact opt-ins that leave the arithmetic unchanged.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping.

**Research ablations.** The options studied and retired during ISLA's
development (parity gates, odd heads, alternative vector bases, local patch
features, latent volume and wake tokens, sparse routing, second-moment
invariants, derived interior normals, anchor-conditioned decode, query-cloud
channels, measure tempering, routing temperature, alternative centers and a
supplied frame) live at git tag ``isla-research-full``. Checkpoints that
recorded such an option at its former default still load; one that recorded
a non-default value raises and names the tag (see ``_REMOVED_OPTIONS``).

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

#: Git tag at which the full research class (every ablation option) is preserved.
RESEARCH_TAG = "isla-research-full"

#: Constructor options removed from the mainline class, with their former
#: defaults. Checkpoints store their constructor arguments, so a checkpoint
#: written by the research class may carry any of these; one that carries the
#: former default is accepted silently (it describes the same network), one
#: that carries anything else cannot be reproduced by this class and raises.
_REMOVED_OPTIONS: dict[str, object] = {
    "measure_weight_power": 1.0,
    "routing_logit_scale": 1.0,
    "use_local_features": False,
    "local_radii": (0.01, 0.03),
    "parity_fix": False,
    "parity_gate_scale": 0.0,
    "vector_basis": "globe7",
    "odd_head": False,
    "raw_coord_channel": False,
    "interior_queries": False,
    "anchor_normal_rho": 0.25,
    "latent_volume_tokens": False,
    "lvt_offsets": (0.5, 1.0, 2.0),
    "wake_tokens": False,
    "wake_offsets": (1.0, 2.0, 4.0),
    "seed_mode": "invariant",
    "scale_conditioning": False,
    "n_anchors": 0,
    "query_local_features": False,
    "query_local_radii": (0.05, 0.15, 0.5),
    "second_moment_features": False,
    "anchor_topk": 0,
    "query_density_feature": False,
    "query_density_radius": 0.05,
    "query_neighbor_features": False,
    "query_neighbor_k": 16,
    "center_mode": "plain",
}


def _is_former_default(name: str, value) -> bool:
    """Whether ``value`` equals the former default of removed option ``name``.
    Sequence defaults are compared as tuples: checkpoint arguments round-trip
    through JSON, which turns tuples into lists."""
    default = _REMOVED_OPTIONS[name]
    if isinstance(default, (tuple, list)):
        return isinstance(value, (tuple, list)) and tuple(value) == tuple(default)
    return value == default


def _check_legacy_options(legacy_options: dict) -> None:
    """Validate the ``**legacy_options`` catch-all of :class:`ISLA`: names that
    were never options raise ``TypeError`` (as an unknown keyword would), removed
    options at their former default pass, anything else raises ``ValueError``
    naming the option and the research tag."""
    for name, value in legacy_options.items():
        if name not in _REMOVED_OPTIONS:
            raise TypeError(
                f"ISLA.__init__() got an unexpected keyword argument {name!r}"
            )
        if not _is_former_default(name, value):
            raise ValueError(
                f"{name}={value!r} is a research option that was removed from ISLA "
                f"(its former default {_REMOVED_OPTIONS[name]!r} is the only accepted value); "
                f"the full research class is preserved at git tag {RESEARCH_TAG!r}"
            )


def _softmax_over_points(
    x: Float[torch.Tensor, "batch tokens slices"], fast: bool = True
):
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
    relative: bool = False,
) -> Float[torch.Tensor, "batch tokens slices geo"]:
    """The point-anchor invariants (v3b set, generalized to K global vector
    inputs): distance and its log, the unit relative vector dotted with each
    global vector (K terms), with the point normal and with the anchor normal,
    the point normal dotted with the anchor normal, and (centered frame only)
    the anchor radius and the anchor direction dotted with each global vector
    (1 + K terms). Width 5 + K (+ 1 + K centered); for K = 1 this is the
    original 6/8-wide set in its original order. Shared by the encoder slice
    blocks and the passive decoder blocks.

    ``relative`` (RELFRAME, 2026-09-10) drops the invariants that refer to the
    frame origin, the anchor radius ``|z_s|`` and the anchor direction
    ``z_hat_s . g_k``, leaving the point-anchor terms, which depend on the
    anchors' positions relative to the point only."""
    z = z_pos[:, None, :, :]
    m = m_s[:, None, :, :]
    rel = r[:, :, None, :] - z  # (B, N, S, 3)
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
    rel_hat = rel / dist
    n_exp = n_hat[:, :, None, :]
    feats = [
        dist,
        torch.log(dist),
        ### Broadcast multiply-and-sum, not einsum: einsum is autocast to bf16 and
        ### the geometry must stay in the input precision (for K = 1 this is the
        ### former (rel_hat * d).sum(-1) arithmetic exactly).
        (rel_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),  # (B, N, S, K)
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
    return torch.cat(feats, dim=-1)


def _geo_region(
    lin: nn.Linear,
    logits_pre,
    r,
    n_hat,
    g_hat,
    z_pos,
    m_s,
    eps: float,
    relative: bool = False,
):
    """One recompute region per layer: the per-slice routing bias from the
    invariants and the invariants pooled over slices by the resulting
    point->slice mix. Returns (bias (B,N,S), mix (B,N,S), pooled (B,N,geo)); the
    (B,N,S,geo) invariants and their (B,N,S,3) intermediates never leave the
    region, so under checkpointing they are rebuilt in backward, not stored."""
    geo = _relational_invariants(r, n_hat, g_hat, z_pos, m_s, eps, relative)
    bias = lin(geo).squeeze(-1)
    mix = torch.softmax(logits_pre + bias, dim=-1)  # normalized over slices
    return bias, mix, torch.einsum("bns,bnsg->bng", mix, geo)


def _fused_geo_region():
    """The Triton-fused dense geometry region (geo_kernel="fused"); imported lazily so
    that the module loads without triton and the eager path never touches it."""
    from .geo_kernel import fused_geo_region

    return fused_geo_region


def _geo_width(n_global_vectors: int, relative: bool) -> int:
    """Width of _relational_invariants: dist, log dist, K rel.g_k, rel.n, rel.m,
    n.m (5 + K); centered adds |z_s| and K zhat_s.g_k. K = 1 gives the original
    6 (relative) / 8 (centered)."""
    k = int(n_global_vectors)
    return 5 + k + (0 if relative else 1 + k)


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mlp_ratio: int = 4,
        use_relational_geo: bool = True,
        geo_checkpoint: bool = False,
        fast_point_softmax: bool = True,
        relative_frame: bool = False,
        geo_kernel: str = "eager",
        n_global_vectors: int = 1,
    ) -> None:
        super().__init__()
        self.use_relational_geo = use_relational_geo
        self.fast_point_softmax = bool(fast_point_softmax)
        ### KERNEL STUDY (2026-09-13): "fused" evaluates the dense geometry region
        ### (invariants -> routing bias -> point->slice mix -> pooled invariants) in
        ### one Triton kernel per direction that never materializes a (B,N,S,.)
        ### tensor (see geo_kernel.py); same arithmetic as _geo_region, recompute
        ### built in, so geo_checkpoint is moot on that path.
        self.geo_kernel = geo_kernel
        ### RELFRAME (2026-09-10): without a frame origin the two origin-referring
        ### invariants (|z_s|, zhat_s.g_k) are gone and the geo width is 5 + K.
        self.relative_frame = bool(relative_frame)
        self.n_geo = _geo_width(n_global_vectors, self.relative_frame)
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
        a = _softmax_over_points(
            logits + log_w, self.fast_point_softmax
        )  # normalized over points
        ### Equivariant anchors: weighted mean position AND mean normal
        ### direction per slice (v3b) -- anchors gain orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, 3)
        m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)
        ### Geometry refines the routing and the readback. A35b ablation:
        ### use_relational_geo=False removes the anchor-relational invariants
        ### from routing and readback (Transolver-style feature-only slicing).
        if self.use_relational_geo:
            ### Pool the invariants over slices FIRST, then project: exactly
            ### equal to projecting then pooling (the projection is affine and
            ### point_mix sums to one over slices), but the saved activation is
            ### (B, N, geo) instead of (B, N, S, hidden/2) -- ~0.5 GB per layer at
            ### 10k tokens, 256 slices, hidden 192 (A35b memory derivation).
            if self.geo_kernel == "fused":
                ### The fused kernel is written for one global vector (enforced at construction).
                bias, point_mix, pooled = _fused_geo_region()(
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    g_hat[:, :, 0],
                    z_pos,
                    m_s,
                    eps,
                    self.relative_frame,
                )
            elif self.geo_checkpoint:
                bias, point_mix, pooled = checkpoint(
                    _geo_region,
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    g_hat,
                    z_pos,
                    m_s,
                    eps,
                    self.relative_frame,
                    use_reentrant=False,
                )
            else:
                bias, point_mix, pooled = _geo_region(
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    g_hat,
                    z_pos,
                    m_s,
                    eps,
                    self.relative_frame,
                )
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

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mlp_ratio: int = 4,
        geo_checkpoint: bool = False,
        relative_frame: bool = False,
        geo_kernel: str = "eager",
        n_global_vectors: int = 1,
    ) -> None:
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

    def forward(
        self,
        q_h,
        q_r,
        q_n,
        q_g,
        z_states,
        z_pos,
        m_s,
        eps,
        src_r=None,
        src_h=None,
        src_w=None,
        local_rho=None,
        kernel_logspace=False,
    ):
        logits_pre = self.assign(self.norm(q_h))
        if self.geo_kernel == "fused":
            _, mix, pooled = _fused_geo_region()(
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                q_g[:, :, 0],
                z_pos,
                m_s,
                eps,
                self.relative_frame,
            )
        elif self.geo_checkpoint:
            _, mix, pooled = checkpoint(
                _geo_region,
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                q_g,
                z_pos,
                m_s,
                eps,
                self.relative_frame,
                use_reentrant=False,
            )
        else:
            _, mix, pooled = _geo_region(
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                q_g,
                z_pos,
                m_s,
                eps,
                self.relative_frame,
            )
        back = torch.einsum("bqs,bsh->bqh", mix, z_states)
        geo_pool = self.geo_feat(pooled)  # pool-then-project (exact)
        q_h = q_h + self.broadcast(torch.cat([q_h, back, geo_pool], dim=-1))
        if src_h is not None:
            ### v5a3: local token readout -- the per-point detail 256 slice
            ### states cannot carry. Measure-weighted Gaussian kernel over
            ### SOURCE positions attending to encoder states; queries still
            ### never write, so query-independence is preserved exactly.
            local = _kernel_readout(
                q_r, src_r, src_h, src_w, local_rho, eps, logspace=kernel_logspace
            )
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
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges.

    A soft-slice transformer for steady boundary-value problems whose attention
    operates only on invariants of the per-point vector set and whose vector
    outputs are re-assembled in an equivariant basis at the heads, so exact
    rotation and translation covariance holds by construction (see the module
    docstring for the contracts and the reference configuration). Every
    argument is keyword-only.

    Args:
        out_scalars: Number of scalar output fields per token (e.g. pressure).
        out_vectors: Number of vector output fields per token (e.g. wall shear);
            each is expanded in the ``4 + 3K`` equivariant head basis.
        hidden: Token width of the slice blocks.
        n_layers: Number of slice-attention blocks in the encoder.
        n_slices: Number of soft slices (data-adaptive anchors) per block.
        mlp_ratio: Expansion ratio of the per-token and per-slice MLPs.
        reference_length: Constant length unit of the centered frame when
            ``scale_mode="reference_length"``; unused by the reference
            configuration (``scale_mode="total_measure"``).
        use_measure_weights: Route with the quadrature measure as a log-space
            bias so slice states are measure-weighted (quadrature) means. ``False``
            is the "weights-off" ablation, which reads the sampling density.
        fast_point_softmax: Use the row-parallel softmax kernel over points
            (``True``, the default since 2026-09-10) or the reference
            middle-dimension kernel (roundoff-level difference).
        n_boundary_scalars: Number of per-cell boundary-condition scalars
            (``boundary_scalars``, e.g. a Dirichlet trace) appended to the seeds;
            tokens that are not boundary cells carry zeros in the channel.
        n_global_vectors: Number ``K >= 0`` of global vector inputs
            (``global_vectors`` of shape ``(B, K, 3)``); each enters as a unit
            direction through one seed cosine, one relational direction cosine
            and three head basis vectors. ``1`` (the external-aerodynamics
            freestream direction) reproduces the former single-vector model.
        n_global_scalars: Number ``S >= 0`` of global scalar inputs
            (``global_scalars`` of shape ``(B, S)``, e.g. PDE parameters),
            appended to every token's seed features.
        similarity_gauge: Centered-frame variant whose centroid and length unit
            are the measure-weighted centroid and RMS radius of the sample
            (adds equivariance to geometric scale); requires
            ``frame_mode="centered"`` and ``scale_mode="reference_length"``.
        use_relational_geo: Feed the point-anchor relational invariants to the
            routing bias and the read-back (``False`` is the feature-only
            slicing ablation).
        query_independent: Decode queries passively through read blocks, so a
            prediction at one point does not depend on which other points are
            queried (given the boundary sample).
        n_decoder_layers: Number of passive read blocks when
            ``query_independent`` is set.
        local_readout_rho: Radius (in frame units) of the measure-weighted
            Gaussian local readout of the passive decoder.
        query_tokens: Admit ``query_points`` as interacting tokens (the interior
            reference configuration; needs ``query_normals``).
        geo_checkpoint: Rebuild the per-slice geometric invariants in the
            backward pass instead of storing them (bitwise-identical forward and
            gradients, lower peak memory).
        n_query_scalars: Number of per-query scalars (``query_scalars``, e.g.
            the signed distance to the wall) for the interior modes.
        query_scalar_scale: ``"length"`` divides query scalars by the frame's
            length unit and enters ``[s, sign(s) log(|s| + eps)]``; ``"none"``
            enters them raw.
        query_mass: Routing weight of interior/support tokens:
            ``"geometric_mean"`` (each query carries the mean boundary
            log-weight plus a learned offset; kept for trained checkpoints) or
            ``"source_total"`` (the tokens share a learned fraction of the total
            boundary measure; invariant to re-representing the same measure).
        support_tokens: With ``query_independent``, admit a per-case
            computational support (``support_points`` / ``support_normals`` /
            ``support_scalars``) as interacting tokens next to the boundary.
        frame_mode: ``"relative"`` (reference): positions enter only as
            point-to-anchor differences, no centroid anywhere; ``"centered"``:
            centre on the plain mean of the sampled points (the constant-gauge
            and similarity-gauge variants).
        scale_mode: ``"total_measure"`` (reference): divide positions by the
            square root of the total quadrature measure, an integral of the
            geometry; ``"reference_length"``: divide by ``reference_length``.
        geo_kernel: ``"eager"`` (default, the reference implementation) or
            ``"fused"`` (exact Triton kernel for the per-layer geometry region,
            CUDA only, opt-in; one global vector only).
        eps: Numerical floor for norms and logarithms.
        **legacy_options: Research options removed from the mainline on
            2026-09-15 (see ``_REMOVED_OPTIONS``). A checkpoint that recorded one
            at its former default loads unchanged; any other value raises a
            ``ValueError`` naming the git tag ``isla-research-full`` where the
            option remains runnable.

    Forward inputs (all keyword-only): ``points`` and ``normals`` of shape
    ``(B, N, 3)``; ``measure_weights`` ``(B, N)`` (required by
    ``scale_mode="total_measure"``; Horvitz-Thompson corrected so they sum to
    the boundary measure); ``global_vectors`` ``(B, K, 3)``; ``global_scalars``
    ``(B, S)``; ``boundary_scalars`` ``(B, N, n_boundary_scalars)``;
    ``query_points`` / ``query_normals`` ``(B, Q, 3)`` and ``query_scalars``
    ``(B, Q, n_query_scalars)`` for the interior modes; ``support_points`` /
    ``support_normals`` ``(B, M, 3)`` and ``support_scalars``
    ``(B, M, n_query_scalars)`` with ``support_tokens``. Returns
    ``(B, N_out, out_scalars + 3 * out_vectors)`` with ``N_out`` the boundary
    token count, or the query count in the interior modes.
    """

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
        fast_point_softmax: bool = True,
        n_boundary_scalars: int = 0,
        n_global_vectors: int = 1,
        n_global_scalars: int = 0,
        similarity_gauge: bool = False,
        use_relational_geo: bool = True,
        query_independent: bool = False,
        n_decoder_layers: int = 4,
        local_readout_rho: float = 0.02,
        query_tokens: bool = False,
        geo_checkpoint: bool = False,
        n_query_scalars: int = 0,
        query_scalar_scale: str = "length",
        query_mass: str = "geometric_mean",
        support_tokens: bool = False,
        frame_mode: str = "relative",
        scale_mode: str = "total_measure",
        geo_kernel: str = "eager",
        eps: float = 1e-12,
        **legacy_options,
    ) -> None:
        ### Removed research options (see _REMOVED_OPTIONS): a checkpoint that
        ### recorded one at its former default describes this very network and
        ### loads; any other value raises. Accepted names are dropped from the
        ### recorded constructor arguments so a re-saved checkpoint is lean.
        _check_legacy_options(legacy_options)
        for name in legacy_options:
            self._args["__args__"].pop(name, None)
        super().__init__(meta=self.MetaData())
        ### KERNEL STUDY (2026-09-13): geo_kernel selects the implementation of the
        ### per-layer geometry region, not its arithmetic. "eager" is the PyTorch
        ### region (with geo_checkpoint deciding whether its (B,N,S,.) intermediates
        ### are stored or rebuilt in backward); "fused" is the Triton kernel of
        ### geo_kernel.py, which reads the pre-logits once and writes the bias and
        ### mix once per direction, with the recompute built in (CUDA only).
        if geo_kernel not in ("eager", "fused"):
            raise ValueError(
                f"geo_kernel must be 'eager' or 'fused', got {geo_kernel!r}"
            )
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
            raise ValueError(
                "n_global_vectors and n_global_scalars must be non-negative"
            )
        if self.n_global_vectors != 1 and geo_kernel == "fused":
            raise ValueError(
                f"geo_kernel='fused' is written for exactly one global vector input; "
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
            raise ValueError(
                f"frame_mode must be 'centered' or 'relative', got {frame_mode!r}"
            )
        self.frame_mode = frame_mode
        self.relative_frame = frame_mode == "relative"
        if scale_mode not in ("reference_length", "total_measure"):
            hint = (
                f" (scale_mode='global' is a research option preserved at git tag {RESEARCH_TAG!r})"
                if scale_mode == "global"
                else ""
            )
            raise ValueError(
                f"scale_mode must be 'reference_length' or 'total_measure', got {scale_mode!r}{hint}"
            )
        self.scale_mode = scale_mode
        if similarity_gauge and scale_mode != "reference_length":
            raise ValueError(
                "similarity_gauge sets its own scale; use scale_mode='reference_length'"
            )
        if self.relative_frame and similarity_gauge:
            raise ValueError(
                "frame_mode='relative' (the default) excludes similarity_gauge (it reads a position "
                "relative to a frame origin); pass frame_mode='centered' and scale_mode='reference_length' to use it"
            )
        ### CENTER (2026-09-10): the constant gauge (similarity_gauge=False) in
        ### the centered frame centers by the PLAIN mean of the sampled points.
        ### The routing softmax adds raw log-weights (invariant to uniform
        ### rescaling) and the slice moments are attention-normalized, so the
        ### unweighted centroid is the ONLY sampling-distribution-dependent
        ### quantity in that forward pass -- the reason the relative frame is
        ### the reference configuration.
        ### Density-factorial knob (prereg 3f4e4af7 follow-up): with False,
        ### the assignment softmax ignores quadrature weights entirely,
        ### isolating the measure-bias pathway of density sensitivity.
        self.use_measure_weights = use_measure_weights
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
        ### G3 experiment channel (prereg pending): per-point boundary-condition
        ### scalars (e.g. a Dirichlet trace). Scalars are invariants, so every
        ### contract is untouched; they simply widen the seed features.
        self.n_boundary_scalars = int(n_boundary_scalars)
        ### Seed width: the {r, n, g_k} invariants (K terms n.g_k in the relative
        ### frame; |r|, log|r|, rhat.g_k, rhat.n, n.g_k centered), the boundary
        ### scalars and the global scalars.
        K = self.n_global_vectors
        n_base = K if self.relative_frame else 3 + 2 * K
        n_seed = n_base + self.n_boundary_scalars + self.n_global_scalars
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
        self.blocks = nn.ModuleList(
            _SliceBlock(
                hidden,
                n_slices,
                mlp_ratio,
                use_relational_geo=use_relational_geo,
                geo_checkpoint=geo_checkpoint,
                fast_point_softmax=self.fast_point_softmax,
                relative_frame=self.relative_frame,
                geo_kernel=geo_kernel,
                n_global_vectors=K,
            )
            for _ in range(n_layers)
        )
        if self.relative_frame:
            ### Head frame (RELFRAME): per-point soft slice anchor for the
            ### radial basis vector.
            self.frame_assign = nn.Linear(hidden, n_slices)
        ### v5a (query_independent): encode/decode split. Queries decode
        ### passively from final encoder slices and anchors: query-independent
        ### by construction given the source sample.
        self.query_independent = query_independent
        self.local_readout_rho = float(local_readout_rho)
        if query_independent:
            self.final_assign = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, n_slices)
            )
            self.read_blocks = nn.ModuleList(
                _ReadBlock(
                    hidden,
                    n_slices,
                    mlp_ratio,
                    geo_checkpoint=geo_checkpoint,
                    relative_frame=self.relative_frame,
                    geo_kernel=geo_kernel,
                    n_global_vectors=K,
                )
                for _ in range(n_decoder_layers)
            )
        self.norm_out = nn.LayerNorm(hidden)
        ### Vector head: coefficients over {g_1..g_K, n, rhat} plus the
        ### spherical-basis complements of (rhat, n) and of each (rhat, g_k) --
        ### the GLOBE multi-vector expansion (4 + 3K basis vectors; 7 for K = 1).
        self.n_basis = 4 + 3 * K
        self.head = nn.Linear(hidden, out_scalars + out_vectors * self.n_basis)
        ### S1 (critic review 2026-09-02): the centered frame's plain-mean
        ### centroid and constant reference length make the model neither
        ### measure-complete nor scale-equivariant. This gauge uses the
        ### measure-weighted centroid and the measure-weighted RMS radius:
        ### exact geometric-scale equivariance and a density-robust frame.
        self.similarity_gauge = similarity_gauge
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
                raise ValueError(
                    "query_tokens is an interacting mode; set query_independent=False"
                )
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
            raise ValueError(
                "query_mass requires query_tokens=True or support_tokens=True"
            )
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
                raise ValueError(
                    "support_tokens requires query_independent=True (passive read blocks decode the queries)"
                )
            if self.query_tokens:
                raise ValueError("support_tokens excludes query_tokens")
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
                raise ValueError(
                    "n_query_scalars requires query_tokens=True or query_independent=True"
                )
            if query_scalar_scale not in ("length", "none"):
                raise ValueError(f"unknown query_scalar_scale {query_scalar_scale!r}")
            width = (
                2 * self.n_query_scalars
                if query_scalar_scale == "length"
                else self.n_query_scalars
            )
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
        return torch.cat(
            [inv, inv.new_zeros(inv.shape[0], n_tokens, self.n_boundary_scalars)],
            dim=-1,
        )

    def _with_global_scalars(self, inv, g_scalars, n_tokens: int):
        """Append the global scalar inputs (B, S) to every token's seed features;
        substitute the constant seed when the feature set is empty."""
        b = inv.shape[0]
        if g_scalars is not None and g_scalars.shape[-1]:
            inv = torch.cat(
                [inv, g_scalars[:, None, :].expand(b, n_tokens, -1).to(inv.dtype)],
                dim=-1,
            )
        if self.constant_seed:
            inv = inv.new_ones(b, n_tokens, 1)
        return inv

    def _query_scalar_features(self, scalars, b: int, n: int, gauge, dtype):
        """Per-query scalars (B, N, n_query_scalars) as seed-side features: divided
        by the gauge and paired with a signed log under ``query_scalar_scale="length"``."""
        qs = scalars.reshape(b, n, self.n_query_scalars).to(dtype)
        if self.query_scalar_scale == "length":
            qs = qs / gauge
            qs = torch.cat(
                [qs, torch.sign(qs) * torch.log(qs.abs() + self.eps)], dim=-1
            )
        return qs

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
                raise ValueError(
                    "this model was built with n_global_vectors=0; pass no global_vectors"
                )
            g_unit = points.new_zeros(b, 0, 3)
        else:
            if global_vectors is None:
                raise ValueError(
                    f"n_global_vectors={K} needs global_vectors of shape (batch, {K}, 3)"
                )
            if global_vectors.numel() % (3 * K):
                raise ValueError(
                    f"global_vectors has {global_vectors.numel()} elements, not a multiple of {K} vectors x 3"
                )
            gv = global_vectors.reshape(-1, K, 3).to(points.dtype)
            if gv.shape[0] == 1 and b != 1:
                gv = gv.expand(b, K, 3)
            elif gv.shape[0] != b:
                raise ValueError(
                    f"global_vectors gives {gv.shape[0]} sets of {K} vectors for a batch of {b}"
                )
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, K, 3)
        g_hat = g_unit[:, None].expand(b, n, K, 3)
        ### Global scalar inputs: (B, S) or (S,) shared across the batch.
        S = self.n_global_scalars
        if S == 0:
            if global_scalars is not None and global_scalars.numel():
                raise ValueError(
                    "this model was built with n_global_scalars=0; pass no global_scalars"
                )
            g_scalars = None
        else:
            if global_scalars is None:
                raise ValueError(
                    f"n_global_scalars={S} needs global_scalars of shape (batch, {S})"
                )
            g_scalars = global_scalars.reshape(-1, S).to(points.dtype)
            if g_scalars.shape[0] == 1 and b != 1:
                g_scalars = g_scalars.expand(b, S)
            elif g_scalars.shape[0] != b:
                raise ValueError(
                    f"global_scalars gives {g_scalars.shape[0]} rows for a batch of {b}"
                )

        ### Frame: no origin in the relative frame; the measure-weighted centroid
        ### under the similarity gauge; the plain mean otherwise. Scale: the square
        ### root of the total measure, the gauge's weighted RMS radius, or L_ref.
        if self.similarity_gauge:
            w_raw = (
                measure_weights.reshape(b, n, 1).to(points.dtype)
                if measure_weights is not None
                else torch.ones(b, n, 1, dtype=points.dtype, device=points.device)
            )
            w_n = w_raw / w_raw.sum(dim=1, keepdim=True).clamp_min(self.eps)
        if self.relative_frame:
            center = points.new_zeros(b, 1, 3)  # RELFRAME: no frame origin
        elif self.similarity_gauge:
            center = (w_n * points).sum(dim=1, keepdim=True)
        else:
            center = points.mean(dim=1, keepdim=True)
        if self.scale_mode == "total_measure":
            if measure_weights is None:
                raise ValueError("scale_mode='total_measure' needs measure_weights")
            gauge = (
                measure_weights.reshape(b, n, 1)
                .to(points.dtype)
                .sum(dim=1, keepdim=True)
                .clamp_min(self.eps)
                .sqrt()
            )
        elif self.similarity_gauge:
            gauge = (
                (
                    (w_n * (points - center).square().sum(-1, keepdim=True)).sum(
                        dim=1, keepdim=True
                    )
                )
                .sqrt()
                .clamp_min(self.eps)
            )  # (B,1,1) weighted RMS radius
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

        invariants = self._seed_invariants(r_mag, r_hat, n_hat, g_hat)
        if self.n_boundary_scalars:
            bs = boundary_scalars.reshape(b, n, self.n_boundary_scalars)
            invariants = torch.cat([invariants, bs.to(invariants.dtype)], dim=-1)
        invariants = self._with_global_scalars(invariants, g_scalars, n)
        h = self.embed(invariants)

        ### n_boundary counts the SURFACE tokens only: the surface measure
        ### statistics that the support and query tokens' routing weights refer
        ### to must not see the tokens appended below.
        n_boundary = n
        if self.support_tokens and support_points is not None:
            ### Support tokens (see __init__): interacting interior tokens that
            ### are a function of the case, not of the requested queries.
            if support_normals is None:
                raise ValueError(
                    "support_tokens needs support_normals (e.g. the SDF gradient at each support point)"
                )
            s_pts = support_points
            bs_, ns_, _ = s_pts.shape
            s_r = (s_pts - center) / gauge
            s_mag = s_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            s_rhat = s_r / s_mag
            s_nhat = support_normals / support_normals.norm(
                dim=-1, keepdim=True
            ).clamp_min(self.eps)
            s_g = g_unit[:, None].expand(bs_, ns_, K, 3)
            s_inv = self._with_global_scalars(
                self._with_empty_boundary_data(
                    self._seed_invariants(s_mag, s_rhat, s_nhat, s_g), ns_
                ),
                g_scalars,
                ns_,
            )
            h_s = self.embed(s_inv) + self.sp_type.to(h.dtype)
            if self.n_query_scalars:
                if support_scalars is None:
                    raise ValueError(
                        "n_query_scalars > 0 with support_tokens needs support_scalars"
                    )
                ss = self._query_scalar_features(
                    support_scalars, bs_, ns_, gauge, s_inv.dtype
                )
                h_s = h_s + self.sp_scalar_embed(ss).to(h_s.dtype)
            elif support_scalars is not None:
                raise ValueError("support_scalars given but n_query_scalars == 0")
            if self.query_mass == "source_total":
                s_logw = (
                    self.sp_logw.to(log_w.dtype)
                    + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True)
                    - math.log(ns_)
                )
            else:
                s_logw = self.sp_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(
                    dim=1, keepdim=True
                )
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
                raise ValueError(
                    "query_tokens needs query_normals (e.g. the SDF gradient at each query)"
                )
            q_pts = query_points
            bq, nq, _ = q_pts.shape
            q_r = (q_pts - center) / gauge
            q_mag = q_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_rhat = q_r / q_mag
            q_nhat = query_normals / query_normals.norm(dim=-1, keepdim=True).clamp_min(
                self.eps
            )
            q_g = g_unit[:, None].expand(bq, nq, K, 3)
            q_inv = self._with_global_scalars(
                self._with_empty_boundary_data(
                    self._seed_invariants(q_mag, q_rhat, q_nhat, q_g), nq
                ),
                g_scalars,
                nq,
            )
            h_q = self.embed(q_inv) + self.qt_type.to(h.dtype)
            if self.n_query_scalars:
                if query_scalars is None:
                    raise ValueError("n_query_scalars > 0 needs query_scalars")
                qs = self._query_scalar_features(
                    query_scalars, bq, nq, gauge, q_inv.dtype
                )
                h_q = h_q + self.qt_scalar_embed(qs).to(h_q.dtype)
            elif query_scalars is not None:
                raise ValueError("query_scalars given but n_query_scalars == 0")
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
                q_logw = (
                    self.qt_logw.to(log_w.dtype)
                    + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True)
                    - math.log(nq)
                )
            else:
                q_logw = self.qt_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(
                    dim=1, keepdim=True
                )
            log_w = torch.cat([log_w, q_logw.expand(b, nq, 1)], dim=1)
            n = n + nq

        for block in self.blocks:
            h = block(h, log_w, r, n_hat, g_hat, self.eps)

        ### The head frame builds its slice anchors from the SOURCE tokens
        ### (src_r, src_logw, and h); the branches below overwrite r_hat, n_hat,
        ### g_out and n with the query-side values, so the source tensors are
        ### captured here.
        src_r, src_logw = r, log_w
        r_out = r  # positions of the tokens the head reads (RELFRAME radial basis)
        g_out = g_hat  # the global vectors seen by the tokens the head reads
        if qt_active:
            ### Heads read the query tokens only (surface tokens were context).
            h_out = h[:, n_boundary:]
            r_hat, n_hat, g_out, b, n = q_rhat, q_nhat, q_g, bq, nq
            r_out = q_r
        elif self.query_independent:
            ### Final encoder slice states and anchors (read-only for queries).
            logits = self.final_assign(h)
            a = _softmax_over_points(logits + log_w, self.fast_point_softmax)
            z_states = torch.einsum("bns,bnh->bsh", a, h)
            z_pos = torch.einsum("bns,bnc->bsc", a, r)
            m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
            m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            if query_points is None:
                q_pts, q_nrm = points, normals
            else:
                q_pts = query_points
                if query_normals is not None:
                    q_nrm = query_normals
                elif query_points.shape[1] != normals.shape[1]:
                    raise ValueError(
                        "query_points are not the boundary points: pass query_normals (e.g. the SDF "
                        "gradient at each query)"
                    )
                else:
                    q_nrm = normals
            q_r = (q_pts - center) / gauge
            q_mag = q_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_rhat = q_r / q_mag
            q_nhat = q_nrm / q_nrm.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            bq, nq, _ = q_pts.shape
            q_g = g_unit[:, None].expand(bq, nq, K, 3)
            q_inv = self._seed_invariants(q_mag, q_rhat, q_nhat, q_g)
            ### The remaining seed channels must match the encoder's seed
            ### layout. Boundary scalars are per-boundary-cell data: the
            ### queries carry them only when the queries ARE the boundary
            ### points, and zeros otherwise.
            if self.n_boundary_scalars:
                if query_points is None:
                    q_inv = torch.cat([q_inv, bs.to(q_inv.dtype)], dim=-1)
                else:
                    q_inv = self._with_empty_boundary_data(q_inv, nq)
            q_inv = self._with_global_scalars(q_inv, g_scalars, nq)
            q_h = self.embed(q_inv)
            if self.n_query_scalars:
                if query_scalars is None:
                    raise ValueError("n_query_scalars > 0 needs query_scalars")
                qs = self._query_scalar_features(
                    query_scalars, bq, nq, gauge, q_inv.dtype
                )
                q_h = q_h + self.rq_scalar_embed(qs).to(q_h.dtype)
            elif query_scalars is not None:
                raise ValueError("query_scalars given but n_query_scalars == 0")
            ### Support-token mode uses the exactly measure-invariant log-space
            ### kernel (see _kernel_readout); the legacy passive path keeps the
            ### clamped form so its trained checkpoints reproduce.
            logspace = self.support_tokens
            src_w = (
                src_logw.squeeze(-1) if logspace else torch.exp(src_logw.squeeze(-1))
            )
            for rb in self.read_blocks:
                q_h = rb(
                    q_h,
                    q_r,
                    q_nhat,
                    q_g,
                    z_states,
                    z_pos,
                    m_s,
                    self.eps,
                    src_r=r,
                    src_h=h,
                    src_w=src_w,
                    local_rho=self.local_readout_rho,
                    kernel_logspace=logspace,
                )
            h_out, r_hat, n_hat, g_out, b, n = q_h, q_rhat, q_nhat, q_g, bq, nq
            r_out = q_r
        else:
            h_out = h
        if self.relative_frame:
            ### RELFRAME head frame: the radial basis vector is the direction from
            ### the token's soft slice anchor (measure-weighted mean of source
            ### positions) instead of from a centroid; translation covariant.
            a_f = _softmax_over_points(
                self.frame_assign(h) + src_logw, self.fast_point_softmax
            )
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
        ### e_th_g, e_ph_g], so trained heads reproduce.
        gs = [g_out[:, :, k] for k in range(self.n_global_vectors)]
        _, e_th_n, e_ph_n = spherical_basis(r_hat, n_hat, normalize_basis_vectors=False)
        basis = gs + [n_hat, r_hat, e_th_n, e_ph_n]
        for g in gs:
            _, e_th, e_ph = spherical_basis(r_hat, g, normalize_basis_vectors=False)
            basis += [e_th, e_ph]
        basis = torch.stack(basis, dim=-2)  # (B, N, n_basis, 3)
        vectors = torch.einsum("bnvk,bnkc->bnvc", coeffs, basis)

        return torch.cat([scalars, vectors.reshape(b, n, self.out_vectors * 3)], dim=-1)


#: Backward-compatible alias for the architecture's previous name.
MeshTransformer2 = ISLA
