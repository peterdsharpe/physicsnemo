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

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        ### Relational geometry (v2): per-slice equivariant anchors and
        ### point-anchor invariants -- many local, data-adaptive reference
        ### points instead of any global frame (smooth by construction).
        self.geo_logit = nn.Linear(self.N_GEO, 1)
        self.geo_feat = nn.Linear(self.N_GEO, hidden // 2)
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
        ### Geometry refines the routing and the readback.
        logits = logits + self.geo_logit(geo).squeeze(-1)
        a = torch.softmax(logits + log_w, dim=1)
        z = torch.einsum("bns,bnh->bsh", a, h)  # slice states
        z = z + self.slice_mlp(z)
        point_mix = torch.softmax(logits, dim=-1)  # normalized over slices
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        geo_pool = torch.einsum("bns,bnsg->bng", point_mix, self.geo_feat(geo))
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
        geo_pool = torch.einsum("bqs,bqsg->bqg", mix, self.geo_feat(geo))
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
        n_seed = (5 + (7 * len(self.local_radii) if use_local_features else 0)
                  + self.n_boundary_scalars + (1 if scale_conditioning else 0))
        ### Seed invariants of {r, n, d}; separation comes from the slice
        ### blocks' relational anchors (v2), not from these.
        self.embed = nn.Sequential(
            nn.Linear(n_seed, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            _SliceBlock(hidden, n_slices, mlp_ratio) for _ in range(n_layers)
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
        if odd_head:
            self.N_ODD = 7
            self.odd_assign = nn.Linear(hidden, n_slices)
            self.odd_gate = nn.Linear(hidden, out_vectors * 2 * self.N_ODD)


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
        center = points.mean(dim=1, keepdim=True)
        r = (points - center) / self.reference_length
        r_mag = r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        r_hat = r / r_mag
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        if measure_weights is None or not self.use_measure_weights:
            log_w = points.new_zeros(b, n, 1)
        else:
            measure_weights = measure_weights.reshape(b, n)
            log_w = torch.log(measure_weights.clamp_min(self.eps))[..., None]

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
        if self.scale_conditioning:
            raw_scale = (points - center).norm(dim=-1).mean(dim=1, keepdim=True)
            log_s = torch.log(raw_scale.clamp_min(self.eps))[..., None]
            invariants = torch.cat(
                [invariants, log_s.expand(b, n, 1)], dim=-1
            )
        h = self.embed(invariants)


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
                q_nrm = query_normals if query_normals is not None else normals
            q_r = (q_pts - center) / self.reference_length
            q_mag = q_r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_rhat = q_r / q_mag
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
            z_q = torch.einsum("bns,bsc->bnc", b_q, z_s)
            m_q = torch.einsum("bns,bsc->bnc", b_q, m_s)
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
            odd_coeff = torch.einsum("bnvjk,bnk->bnvj", g, pseudo)  # (B,N,V,2)
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
