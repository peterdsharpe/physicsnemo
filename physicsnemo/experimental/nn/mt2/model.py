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

    N_GEO = 3  # |r - z_s|, rhat_is . d, rhat_is . n

    def __init__(self, hidden: int, n_slices: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = nn.Linear(hidden, n_slices)
        ### Relational geometry (v2): per-slice equivariant anchors and
        ### point-anchor invariants -- many local, data-adaptive reference
        ### points instead of any global frame (smooth by construction).
        self.geo_logit = nn.Linear(self.N_GEO, 1)
        self.geo_feat = nn.Linear(self.N_GEO, hidden // 4)
        self.slice_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )
        self.broadcast = nn.Linear(2 * hidden + hidden // 4, hidden)
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
        ### Equivariant anchors: weighted mean position per slice.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, 3)
        rel = r[:, :, None, :] - z_pos[:, None, :, :]  # (B, N, S, 3)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
        rel_hat = rel / dist
        geo = torch.cat(
            [
                dist,
                (rel_hat * d_hat[:, :, None, :]).sum(-1, keepdim=True),
                (rel_hat * n_hat[:, :, None, :]).sum(-1, keepdim=True),
            ],
            dim=-1,
        )  # (B, N, S, 3) invariants
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
        eps: float = 1e-12,
    ) -> None:
        super().__init__(meta=self.MetaData())
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.reference_length = float(reference_length)
        self.eps = eps
        ### Seed invariants of {r, n, d}; separation comes from the slice
        ### blocks' relational anchors (v2), not from these.
        self.embed = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            _SliceBlock(hidden, n_slices, mlp_ratio) for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(hidden)
        ### Vector head: coefficients over {d, n, rhat} plus the
        ### spherical-basis complements of (rhat, n) and (rhat, d) -- the
        ### GLOBE multi-vector expansion (7 basis vectors).
        self.n_basis = 7
        self.head = nn.Linear(hidden, out_scalars + out_vectors * self.n_basis)

    def forward(
        self,
        points: Float[torch.Tensor, "batch tokens 3"],
        normals: Float[torch.Tensor, "batch tokens 3"],
        drive: Float[torch.Tensor, "batch 3"] | Float[torch.Tensor, " 3"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
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
        r = (points - points.mean(dim=1, keepdim=True)) / self.reference_length
        r_mag = r.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        r_hat = r / r_mag
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)

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
        h = self.embed(invariants)

        if measure_weights is None:
            log_w = h.new_zeros(b, n, 1)
        else:
            measure_weights = measure_weights.reshape(b, n)
            log_w = torch.log(measure_weights.clamp_min(self.eps))[..., None]

        for block in self.blocks:
            h = block(h, log_w, r, n_hat, d_hat, self.eps)
        out = self.head(self.norm_out(h))

        scalars = out[..., : self.out_scalars]
        coeffs = out[..., self.out_scalars :].reshape(
            b, n, self.out_vectors, self.n_basis
        )
        ### GLOBE-style expansion: input vectors + spherical complements of
        ### the (r_hat, n_hat) and (r_hat, d_hat) pairs. Exactly equivariant;
        ### non-orthogonal inputs span via the complements.
        _, e_th_n, e_ph_n = spherical_basis(r_hat, n_hat, normalize_basis_vectors=False)
        _, e_th_d, e_ph_d = spherical_basis(r_hat, d_hat, normalize_basis_vectors=False)
        basis = torch.stack(
            [d_hat, n_hat, r_hat, e_th_n, e_ph_n, e_th_d, e_ph_d], dim=-2
        )  # (B, N, 7, 3)
        vectors = torch.einsum("bnvk,bnkc->bnvc", coeffs, basis)

        out_fields = torch.cat(
            [scalars, vectors.reshape(b, n, self.out_vectors * 3)], dim=-1
        )
        ### Degree-one contract: every output scales with the drive magnitude.
        return out_fields * drive_mag[:, None, :]
