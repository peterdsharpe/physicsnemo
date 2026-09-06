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

r"""MeshTransformer2 (MT2), stage-0 prototype.

Thesis under test (preregistration mt2_stage0, sha c73bf8b6): soft-slice
global routing retains its measured power when fed exclusively
similarity-invariant features, so exact equivariance can be paid once at
the network's edges instead of in every layer.

Contracts, all by construction rather than per-layer enforcement:

- **Similarity equivariance.** The backbone sees only invariants of the
  per-point vector set :math:`\{r_i, n_i, d\}` (centered/scaled relative
  position, unit normal, unit drive direction); vector outputs are
  expanded in that set plus its spherical-basis complements (the GLOBE
  multi-vector treatment) with invariant coefficients.
- **Drive degree one.** Outputs are scaled by :math:`\lVert d \rVert`;
  the backbone sees only the direction, so a :math:`k\times` drive input
  moves every output by exactly :math:`k\times`.
- **Measure-aware aggregation.** Slice states are quadrature-weighted
  means, so the routing reads an (unbiasedly) sampled integral rather
  than a raw point population.
"""

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.functional.equivariant_ops import spherical_basis


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    N_GEO = 8  # v3b: dist, log dist, rel dots (d, n, m_s), n.m_s, |z_s|, zhat_s.d

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4,
                 use_relational_geo: bool = True) -> None:
        super().__init__()
        self.use_relational_geo = use_relational_geo
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        ### Relational geometry (v2): per-slice equivariant anchors and
        ### point-anchor invariants -- many local, data-adaptive reference
        ### points instead of any global frame (smooth by construction).
        ### Built only when used: parameters that never receive gradients
        ### crash DDP (A35b nogeo, 2026-09-05).
        self.geo_width = hidden // 2
        if use_relational_geo:
            self.geo_logit = nn.Linear(self.N_GEO, 1)
            self.geo_feat = nn.Linear(self.N_GEO, self.geo_width)
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
        d_hat: Float[torch.Tensor, "batch tokens 3"],
        eps: float,
    ) -> Float[torch.Tensor, "batch tokens hidden"]:
        ### Soft assignment of points to slices; measure weights enter as a
        ### log-space bias so slice states are quadrature-weighted means.
        logits = self.assign(self.norm_assign(h))  # (B, N, S)
        a = torch.softmax(logits + log_w, dim=1)  # normalized over points
        ### Equivariant anchors: weighted mean position AND mean normal
        ### direction per slice (v3b) -- anchors gain orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, 3)
        m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)
        z_mag = z_pos.norm(dim=-1, keepdim=True).clamp_min(eps)
        z_hat = z_pos / z_mag
        rel = r[:, :, None, :] - z_pos[:, None, :, :]  # (B, N, S, 3)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
        rel_hat = rel / dist
        n_exp = n_hat[:, :, None, :]
        geo = torch.cat(
            [
                dist,
                torch.log(dist),
                (rel_hat * d_hat[:, :, None, :]).sum(-1, keepdim=True),
                (rel_hat * n_exp).sum(-1, keepdim=True),
                (rel_hat * m_s[:, None, :, :]).sum(-1, keepdim=True),
                (n_exp * m_s[:, None, :, :]).sum(-1, keepdim=True),
                z_mag[:, None, :, :].expand(rel.shape[0], rel.shape[1], -1, 1),
                (z_hat[:, None, :, :] * d_hat[:, :, None, :]).sum(-1, keepdim=True),
            ],
            dim=-1,
        )  # (B, N, S, 8) invariants
        ### Geometry refines the routing and the readback. A35b ablation:
        ### use_relational_geo=False removes the anchor-relational invariants
        ### from routing and readback (Transolver-style feature-only slicing).
        if self.use_relational_geo:
            logits = logits + self.geo_logit(geo).squeeze(-1)
        a = torch.softmax(logits + log_w, dim=1)
        z = torch.einsum("bns,bnh->bsh", a, h)  # slice states
        z = z + self.slice_mlp(z)
        point_mix = torch.softmax(logits, dim=-1)  # normalized over slices
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        if self.use_relational_geo:
            ### Pool the 8 invariants over slices FIRST, then project: exactly
            ### equal to projecting then pooling (the projection is affine and
            ### point_mix sums to one over slices), but the saved activation is
            ### (B, N, 8) instead of (B, N, S, hidden/2) -- ~0.5 GB per layer at
            ### 10k tokens, 256 slices, hidden 192 (A35b memory derivation).
            geo_pool = self.geo_feat(torch.einsum("bns,bnsg->bng", point_mix, geo))
        else:
            geo_pool = h.new_zeros(h.shape[0], h.shape[1], self.geo_width)
        h = h + self.broadcast(torch.cat([h, back, geo_pool], dim=-1))
        return h + self.mlp(self.norm_mlp(h))




class _ReadBlock(nn.Module):
    """Passive decoder layer (v5a): queries read encoder slices/anchors,
    never write. Removing the write-back is what makes predictions at one
    query independent of every other query (given a fixed source sample)."""

    N_GEO = 8  # v5a2: the full v3b relational-feature set (thin decoder
    # pipes collapse training -- measured twice now, v3a and v5a-v1)

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        self.geo_logit = nn.Linear(self.N_GEO, 1)
        self.geo_feat = nn.Linear(self.N_GEO, hidden // 2)
        self.broadcast = nn.Linear(2 * hidden + hidden // 2, hidden)
        self.local_read = nn.Linear(2 * hidden, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(self, q_h, q_r, q_n, q_d, z_states, z_pos, m_s, eps,
                src_r=None, src_h=None, src_w=None, local_rho=None):
        rel = q_r[:, :, None, :] - z_pos[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
        rel_hat = rel / dist
        z_mag = z_pos.norm(dim=-1, keepdim=True).clamp_min(eps)
        z_hat = z_pos / z_mag
        n_exp = q_n[:, :, None, :]
        geo = torch.cat(
            [
                dist,
                torch.log(dist),
                (rel_hat * q_d[:, :, None, :]).sum(-1, keepdim=True),
                (rel_hat * n_exp).sum(-1, keepdim=True),
                (rel_hat * m_s[:, None, :, :]).sum(-1, keepdim=True),
                (n_exp * m_s[:, None, :, :]).sum(-1, keepdim=True),
                z_mag[:, None, :, :].expand(rel.shape[0], rel.shape[1], -1, 1),
                (z_hat[:, None, :, :] * q_d[:, :, None, :]).sum(-1, keepdim=True),
            ],
            dim=-1,
        )
        logits = self.assign(self.norm(q_h)) + self.geo_logit(geo).squeeze(-1)
        mix = torch.softmax(logits, dim=-1)
        back = torch.einsum("bqs,bsh->bqh", mix, z_states)
        geo_pool = self.geo_feat(torch.einsum("bqs,bqsg->bqg", mix, geo))  # pool-then-project (exact)
        q_h = q_h + self.broadcast(torch.cat([q_h, back, geo_pool], dim=-1))
        if src_h is not None:
            ### v5a3: local token readout -- the per-point detail 256 slice
            ### states cannot carry. Measure-weighted Gaussian kernel over
            ### SOURCE positions attending to encoder states; queries still
            ### never write, so query-independence is preserved exactly.
            local = _kernel_readout(q_r, src_r, src_h, src_w, local_rho, eps)
            q_h = q_h + self.local_read(torch.cat([q_h, local], dim=-1))
        return q_h + self.mlp(self.norm_mlp(q_h))



def _kernel_readout(q_r, src_r, src_h, src_w, rho, eps):
    """Measure-weighted Gaussian-kernel average of source states at query
    positions, row-chunked. Passive: a pure function of the source."""
    b, nq, _ = q_r.shape
    outs = []
    chunk = 4096
    for i0 in range(0, nq, chunk):
        d2 = torch.cdist(q_r[:, i0 : i0 + chunk], src_r).square()
        k = torch.exp(-d2 / (rho * rho)) * src_w[:, None, :]
        mass = k.sum(-1, keepdim=True).clamp_min(eps)
        outs.append(torch.einsum("bcn,bnh->bch", k, src_h) / mass)
    return torch.cat(outs, dim=1)

class MeshTransformer2(Module):
    r"""Stage-0 MT2: invariant backbone, equivariant edges (see module docs)."""

    class MetaData(ModelMetaData):
        jit: bool = False
        amp: bool = True

    def __init__(
        self,
        out_scalars: int = 1,
        out_vectors: int = 1,
        hidden: int = 256,
        n_layers: int = 12,
        n_slices: int = 256,
        mlp_ratio: int = 4,
        reference_length: float = 8.0,
        use_measure_weights: bool = True,
        use_local_features: bool = False,
        local_radii: tuple[float, ...] = (0.01, 0.03),
        n_boundary_scalars: int = 0,
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
        seed_mode: str = "invariant",
        use_relational_geo: bool = True,
        scale_conditioning: bool = False,
        query_independent: bool = False,
        n_anchors: int = 0,
        n_decoder_layers: int = 4,
        local_readout_rho: float = 0.02,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### Density-factorial knob (prereg 3f4e4af7 follow-up): with False,
        ### the assignment softmax ignores quadrature weights entirely,
        ### isolating the measure-bias pathway of density sensitivity.
        self.use_measure_weights = use_measure_weights
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
        ### r_hat . (n_hat x d_hat) restores exact parity covariance. Off by
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
        ### A35b ablations: seed_mode="raw" replaces the five {r,n,d} invariant
        ### seeds with the raw vectors [r, n, d] (GeoTransolver-style inputs);
        ### use_relational_geo=False removes anchor geometry from the slices.
        if seed_mode not in ("invariant", "raw"):
            raise ValueError(f"unknown seed_mode {seed_mode!r}")
        self.seed_mode = seed_mode
        n_base = 5 if seed_mode == "invariant" else 9
        n_seed = (n_base + (7 * len(self.local_radii) if use_local_features else 0)
                  + self.n_boundary_scalars + (1 if scale_conditioning else 0)
                  + (6 if raw_coord_channel else 0))
        ### Seed invariants of {r, n, d}; separation comes from the slice
        ### blocks' relational anchors (v2), not from these.
        self.embed = nn.Sequential(
            nn.Linear(n_seed, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            _SliceBlock(hidden, n_slices, mlp_ratio, use_relational_geo=use_relational_geo)
            for _ in range(n_layers)
        )
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
                _ReadBlock(hidden, n_slices, mlp_ratio) for _ in range(n_decoder_layers)
            )
        self.norm_out = nn.LayerNorm(hidden)
        ### Vector head: coefficients over {d, n, rhat} plus the
        ### spherical-basis complements of (rhat, n) and (rhat, d) -- the
        ### GLOBE multi-vector expansion (7 basis vectors).
        ### L2 experiment (2026-09-02): the two e_phi complements are
        ### pseudovectors. "globe7" is the original basis; "true5" drops them
        ### (capacity control); "true7" replaces them with the TRUE vectors
        ### e_phi_n x d_hat and e_phi_d x n_hat (pseudo x true = true), giving
        ### exact reflection covariance with no gating and no lost channel.
        if vector_basis not in ("globe7", "true5", "true7"):
            raise ValueError(f"unknown vector_basis {vector_basis!r}")
        self.vector_basis = vector_basis
        self.n_basis = 5 if vector_basis == "true5" else 7
        self.head = nn.Linear(hidden, out_scalars + out_vectors * self.n_basis)
        ### W2 (2026-09-02): odd-coefficient head. {r,n,d} span R^3, so the
        ### e_phi (pseudovector) direction is reachable COVARIANTLY only with a
        ### parity-odd coefficient, and the trunk emits even invariants only.
        ### Build K pseudoscalars from {r, n, d} and the point's soft slice
        ### anchor (weighted anchor position z and normal m, both true
        ### vectors), and set coeff_phi = sum_k p_k * g_k(h). Exactly
        ### reflection-covariant; not killed where any single p_k vanishes.
        self.odd_head = odd_head
        ### S1 (critic review 2026-09-02): the original reduction used an
        ### UNWEIGHTED centroid and a CONSTANT reference length, so MT2 was
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
            if not query_independent:
                raise ValueError("latent_volume_tokens requires query_independent=True")
            self.lvt_assign = nn.Linear(hidden, n_slices)
            self.lvt_logw = nn.Parameter(torch.zeros(1))
            self.lvt_embed = nn.Sequential(
                nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
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


    def _local_invariants_at(self, q_r, q_n, q_d, src_r, src_n, log_w):
        """Query-passive variant: patch integrals of the SOURCE sample
        evaluated at arbitrary query positions."""
        b, nq, _ = q_r.shape
        w = torch.exp(log_w.squeeze(-1))
        feats = []
        chunk = 4096
        for rho in self.local_radii:
            outs = []
            for i0 in range(0, nq, chunk):
                ri = q_r[:, i0 : i0 + chunk]
                d2 = torch.cdist(ri, src_r).square()
                k = torch.exp(-d2 / (rho * rho)) * w[:, None, :]
                mass = k.sum(-1, keepdim=True).clamp_min(self.eps)
                nbar = torch.einsum("bcn,bnk->bck", k, src_n) / mass
                delta = (torch.einsum("bcn,bnk->bck", k, src_r) / mass) - ri
                ni = q_n[:, i0 : i0 + chunk]
                di = q_d[:, i0 : i0 + chunk]
                outs.append(
                    torch.cat(
                        [
                            (nbar * ni).sum(-1, keepdim=True),
                            (nbar * di).sum(-1, keepdim=True),
                            nbar.norm(dim=-1, keepdim=True),
                            (delta * ni).sum(-1, keepdim=True) / rho,
                            (delta * di).sum(-1, keepdim=True) / rho,
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
        d_hat: Float[torch.Tensor, "batch tokens 3"],
        log_w: Float[torch.Tensor, "batch tokens 1"],
    ) -> Float[torch.Tensor, "batch tokens feats"]:
        """Measure-weighted Gaussian patch integrals at fixed physical radii.

        Exactly equivariant (integrals of equivariant vectors, projected on
        n_i and d); unbiased under HT sampling via the measure weights;
        row-chunked so the pairwise kernel never materializes at full size.
        """
        b, n, _ = r.shape
        w = torch.exp(log_w.squeeze(-1))  # (B, N) relative measure weights
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
                di = d_hat[:, i0 : i0 + chunk]
                outs.append(
                    torch.cat(
                        [
                            (nbar * ni).sum(-1, keepdim=True),
                            (nbar * di).sum(-1, keepdim=True),
                            nbar.norm(dim=-1, keepdim=True),
                            (delta * ni).sum(-1, keepdim=True) / rho,
                            (delta * di).sum(-1, keepdim=True) / rho,
                            delta.norm(dim=-1, keepdim=True) / rho,
                            torch.log(mass),
                        ],
                        dim=-1,
                    )
                )
            feats.append(torch.cat(outs, dim=1))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        points: Float[torch.Tensor, "batch tokens 3"],
        normals: Float[torch.Tensor, "batch tokens 3"],
        drive: Float[torch.Tensor, "batch 3"] | Float[torch.Tensor, " 3"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
        boundary_scalars: Float[torch.Tensor, "batch tokens n_bscalars"] | None = None,
        query_points: Float[torch.Tensor, "batch queries 3"] | None = None,
        query_normals: Float[torch.Tensor, "batch queries 3"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens out_dim"]:
        if points.ndim == 2:
            points = points[None]
            normals = normals[None]
        b, n, _ = points.shape
        ### Tolerate any drive layout the recipe delivers: (3,), (B, 3), or
        ### collated (B, 1, 3).
        drive = drive.reshape(-1, 3)
        if drive.shape[0] != b:
            drive = drive.expand(b, 3)

        ### Drive-degree-one bypass: backbone sees the direction only.
        drive_mag = drive.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # (B,1)
        d_hat = (drive / drive_mag)[:, None, :].expand(b, n, 3)

        ### Similarity reduction: center by the plain mean, scale by L_ref.
        if self.similarity_gauge:
            w_raw = (
                measure_weights.reshape(b, n, 1).to(points.dtype)
                if measure_weights is not None
                else torch.ones(b, n, 1, dtype=points.dtype, device=points.device)
            )
            w_n = w_raw / w_raw.sum(dim=1, keepdim=True).clamp_min(self.eps)
            center = (w_n * points).sum(dim=1, keepdim=True)
            gauge = (
                (w_n * (points - center).square().sum(-1, keepdim=True)).sum(dim=1, keepdim=True)
            ).sqrt().clamp_min(self.eps)  # (B,1,1) weighted RMS radius
        else:
            center = points.mean(dim=1, keepdim=True)
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

        if self.seed_mode == "raw":
            invariants = torch.cat([r, n_hat, d_hat], dim=-1)
        else:
            invariants = torch.cat(
                [
                    r_mag,
                    torch.log(r_mag),
                    (r_hat * d_hat).sum(-1, keepdim=True),
                    (r_hat * n_hat).sum(-1, keepdim=True),
                    (n_hat * d_hat).sum(-1, keepdim=True),
                ],
                dim=-1,
            )
        if self.use_local_features:
            invariants = torch.cat(
                [invariants, self._local_invariants(r, n_hat, d_hat, log_w)],
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
        h = self.embed(invariants)

        if self.latent_volume_tokens and self.query_independent:
            ### pre-encoder geometric slice assignment -> anchors z0, m0, rho0
            a0 = torch.softmax(self.lvt_assign(h) + log_w, dim=1)  # (B,N,S)
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
            mtok.append(d_hat[:, :1])                          # covariant placeholder normal
            ctok.append(torch.zeros_like(rho0[:, :1]))
            rtok.append(rho0.mean(dim=1, keepdim=True))
            p_l = torch.cat(pos, dim=1)                        # (B,K,3)
            m_l = torch.cat(mtok, dim=1)
            c_l = torch.cat(ctok, dim=1)[..., None]
            rho_l = torch.cat(rtok, dim=1)[..., None]
            K = p_l.shape[1]
            d_l = d_hat[:, :1].expand(b, K, 3)
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
            d_hat = torch.cat([d_hat, d_l], dim=1)
            log_w = torch.cat([log_w, self.lvt_logw.to(log_w.dtype).expand(b, K, 1)], dim=1)
            n = n + K

        if not (self.query_independent and 0 < self.n_anchors < n):
            for block in self.blocks:
                h = block(h, log_w, r, n_hat, d_hat, self.eps)

        if self.query_independent:
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
                d_enc = d_hat[:, idx]
                log_w_enc = log_w[:, idx]
            else:
                r_enc, n_enc, d_enc, log_w_enc = r, n_hat, d_hat, log_w
            ### Final encoder slice states and anchors (read-only for queries).
            if 0 < self.n_anchors < n:
                for block in self.blocks:
                    h = block(h, log_w_enc, r_enc, n_enc, d_enc, self.eps)
            logits = self.final_assign(h)
            a = torch.softmax(logits + (log_w_enc if 0 < self.n_anchors < n else log_w), dim=1)
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
            q_d = (drive / drive_mag)[:, None, :].expand(bq, nq, 3)
            q_inv = torch.cat(
                [
                    q_mag,
                    torch.log(q_mag),
                    (q_rhat * q_d).sum(-1, keepdim=True),
                    (q_rhat * q_nhat).sum(-1, keepdim=True),
                    (q_nhat * q_d).sum(-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.use_local_features:
                ### Local integrals read the SOURCE sample -- query-passive.
                q_inv = torch.cat(
                    [q_inv, self._local_invariants_at(q_r, q_nhat, q_d, r, n_hat, log_w)],
                    dim=-1,
                )
            q_h = self.embed(q_inv)
            src_w = torch.exp((log_w_enc if 0 < self.n_anchors < n else log_w).squeeze(-1))
            for rb in self.read_blocks:
                q_h = rb(
                    q_h, q_r, q_nhat, q_d, z_states, z_pos, m_s, self.eps,
                    src_r=r_src, src_h=h, src_w=src_w,
                    local_rho=self.local_readout_rho,
                )
            h_out, r_hat, n_hat, d_hat, b, n = q_h, q_rhat, q_nhat, q_d, bq, nq
        else:
            h_out = h
        out = self.head(self.norm_out(h_out))

        scalars = out[..., : self.out_scalars]
        coeffs = out[..., self.out_scalars :].reshape(
            b, n, self.out_vectors, self.n_basis
        )
        ### GLOBE-style expansion: input vectors + spherical complements of
        ### the (r_hat, n_hat) and (r_hat, d_hat) pairs. Exactly equivariant;
        ### non-orthogonal inputs span via the complements.
        _, e_th_n, e_ph_n = spherical_basis(r_hat, n_hat, normalize_basis_vectors=False)
        _, e_th_d, e_ph_d = spherical_basis(r_hat, d_hat, normalize_basis_vectors=False)
        if self.vector_basis == "true5":
            basis = torch.stack([d_hat, n_hat, r_hat, e_th_n, e_th_d], dim=-2)
        elif self.vector_basis == "true7":
            t_n = torch.linalg.cross(e_ph_n, d_hat, dim=-1)
            t_d = torch.linalg.cross(e_ph_d, n_hat, dim=-1)
            basis = torch.stack(
                [d_hat, n_hat, r_hat, e_th_n, e_th_d, t_n, t_d], dim=-2
            )
        else:
            basis = torch.stack(
                [d_hat, n_hat, r_hat, e_th_n, e_ph_n, e_th_d, e_ph_d], dim=-2
            )  # (B, N, 7, 3)
        if self.odd_head and self.vector_basis == "globe7":
            ### per-point soft slice anchor (true vectors, equivariant)
            src_r = r_src if (self.query_independent and 0 < self.n_anchors < n) else r
            src_n = n_src if (self.query_independent and 0 < self.n_anchors < n) else n_hat
            src_h = h
            src_logw = log_w_enc if (self.query_independent and 0 < self.n_anchors < n) else log_w
            lg = self.odd_assign(src_h)
            a_s = torch.softmax(lg + src_logw, dim=1)  # slices over source points
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
                r_hat * torch.linalg.cross(n_hat, d_hat, dim=-1)
            ).sum(-1)[..., None, None]  # (B, N, 1, 1), parity-odd invariant
            if self.parity_gate_scale > 0:
                p_odd = torch.tanh(p_odd / self.parity_gate_scale)
            gate = torch.ones_like(coeffs[..., :1, :]).expand_as(coeffs).clone()
            gate[..., 4] = p_odd[..., 0]
            gate[..., 6] = p_odd[..., 0]
            coeffs = coeffs * gate
        vectors = torch.einsum("bnvk,bnkc->bnvc", coeffs, basis)

        out_fields = torch.cat(
            [scalars, vectors.reshape(b, n, self.out_vectors * 3)], dim=-1
        )
        ### Degree-one contract: every output scales with the drive magnitude.
        return out_fields * drive_mag[:, None, :]
