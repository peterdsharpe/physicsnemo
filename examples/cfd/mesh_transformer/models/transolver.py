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

r"""Faithful Transolver baseline for the Laplace boundary-to-interior benchmark.

Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General
Geometries" (ICML 2024).  :class:`PhysicsAttention`, :class:`TransolverBlock`,
and :class:`Transolver` below are a line-for-line functional port of the
official irregular-mesh implementation (``Physics_Attention_Irregular_Mesh``
and ``Transolver_Irregular_Mesh.Model`` from
https://github.com/thuml/Transolver, MIT license, THUML @ Tsinghua
University), kept numerically identical to the upstream code rather than
"improved":

* Per-head soft assignment of the ``N`` tokens to ``M`` learnable slices via a
  linear map on head features, divided by a **learnable per-head temperature**
  (initialized at 0.5, unclamped, exactly as upstream) and softmaxed over
  slices; weighted mean-pooling (epsilon ``1e-5``) to ``M`` slice tokens;
  standard scaled dot-product attention among slice tokens; broadcast back to
  tokens with the transposed assignment weights; concatenate heads and apply
  an output projection.
* Pre-LayerNorm block structure: ``LN -> Physics-Attention -> residual`` then
  ``LN -> two-layer GELU MLP -> residual``; the final block additionally emits
  the output field through one more LayerNorm and a linear head
  (``last_layer=True`` upstream).
* Embedding: a two-layer GELU MLP (width ``2 * hidden``) on the concatenated
  coordinate and function features; upstream's ``placeholder`` parameter is
  created (it is part of the official parameter budget) but inert here because
  this benchmark always supplies function features, matching the official
  ``fun_dim > 0`` experiments.
* Initialization quirk preserved: upstream applies orthogonal initialization
  to the slice projection inside the layer constructor and *then* overwrites
  every linear weight (including that one) with truncated-normal ``std=0.02``
  in ``Model.initialize_weights``.  We replicate the same ordering, hence the
  same net initialization.

Benchmark adaptation
--------------------
Our problem maps ``{boundary geometry + Dirichlet values}`` to the interior
potential at query points; Transolver's native interface is a fixed point set
with per-point features.  Every adaptation ambiguity is resolved in
Transolver's favor:

* **One shared token sequence.**  The token set is the union of the boundary
  cells and the interior query points.  This grants Transolver full
  boundary-to-query, query-to-boundary, and query-to-query interaction --
  strictly more information flow than the MeshTransformer, whose query
  decodings are independent of one another.  The cost is that a query's
  prediction depends on which other queries are in the batch (query
  independence is broken); we accept and record that contract break as the
  steel-man choice.
* **Token features (7 channels).**  Coordinates are the ``space_dim=2``
  channels; the ``fun_dim=5`` channels are: outward unit normal (2),
  dimensionless cell measure (panel length over ``reference_length``),
  Dirichlet boundary value, and a binary boundary/query indicator (1 for
  boundary cells, 0 for queries).  Query tokens carry zeros for the normal,
  measure, and value channels.  Boundary tokens sit at panel centroids.
* **Normalization.**  Coordinates are centered on the measure-weighted
  boundary centroid and divided by the declared ``reference_length`` -- the
  identical normalization used by the :class:`models.InvariantPairKernel`
  control -- so Transolver is not penalized by the benchmark's unit
  conventions.  The normalized *raw coordinates are features*: Transolver is
  deliberately not translation/rotation invariant, which is exactly the
  contrast this baseline exists to measure.
* **Read-out.**  The scalar ``potential`` is read at the query token
  positions of the final block's linear head.  Boundary-token outputs are
  discarded.
* **No physics scaffolding.**  Boundary values enter raw (no mean lifting, no
  linearity structure); superposition and similarity contracts are therefore
  empirical measurements, as for the other external controls.
* **Batching.**  The benchmark presents one variable-size ``DomainMesh`` at a
  time, so the model runs with batch size 1.

Capacity presets
----------------
``"matched"`` (hidden 64, 4 heads of dim 16, 3 blocks, ``M=64`` slices,
``mlp_ratio=2``) has 103,053 parameters, within 1.2% of the 104,261-parameter
``mesh_transformer_kernel_singonly`` reference arm.  ``"native"`` (hidden 256,
8 heads of dim 32, 8 blocks, ``M=32`` slices, ``mlp_ratio=1``) has 2,809,409
parameters, matching Transolver's published PDE-benchmark scale (the official
irregular-mesh elasticity run uses hidden 128 / 8 heads / 8 layers / 64
slices at ~0.7M parameters; the larger published width 256 with the official
default ``mlp_ratio=1`` is used here to land in the 1-3M published range).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from models import _benchmark_boundary, _prediction_mesh, _reference_length
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh


class PhysicsAttention(nn.Module):
    r"""Official Transolver physics attention for irregular meshes.

    Functional port of ``Physics_Attention_Irregular_Mesh`` from the official
    repository: slice, attend among slice tokens, deslice.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        slice_num: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1 or heads < 1 or dim_head < 1 or slice_num < 1:
            raise ValueError("dim, heads, dim_head, and slice_num must be positive")
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Learnable per-head slice temperature, initialized at 0.5 and never
        # clamped, exactly as in the official implementation.
        self.temperature = nn.Parameter(0.5 * torch.ones(1, heads, 1, 1))

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        # Upstream initializes this orthogonally here; the model-level
        # truncated-normal initialization later overwrites it (see module
        # docstring).  Both steps are kept to replicate upstream exactly.
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token features ``(B, N, C)`` to attended features ``(B, N, C)``."""

        batch, n_tokens, _ = x.shape

        # (1) Slice: soft-assign every token to M slices per head and pool.
        fx_mid = (
            self.in_project_fx(x)
            .reshape(batch, n_tokens, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B, H, N, D)
        x_mid = (
            self.in_project_x(x)
            .reshape(batch, n_tokens, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B, H, N, D)
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # (B, H, N, M)
        slice_norm = slice_weights.sum(2)  # (B, H, M)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5)[:, :, :, None]  # (B, H, M, D)

        # (2) Standard scaled dot-product attention among the M slice tokens.
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attention = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attention, v_slice_token)  # (B, H, M, D)

        # (3) Deslice: broadcast back with the transposed assignment weights.
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(batch, n_tokens, -1)  # (B, N, C)
        return self.to_out(out_x)


class _FeedForward(nn.Module):
    """Official two-layer GELU MLP (upstream ``MLP(..., n_layers=0, res=False)``)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransolverBlock(nn.Module):
    """Official pre-LayerNorm Transolver block, with the last-layer head."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float = 0.0,
        last_layer: bool = False,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim,
            heads=heads,
            dim_head=hidden_dim // heads,
            slice_num=slice_num,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = _FeedForward(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    """Official Transolver model for irregular point sets."""

    def __init__(
        self,
        *,
        space_dim: int,
        fun_dim: int,
        out_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.preprocess = _FeedForward(space_dim + fun_dim, hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    slice_num=slice_num,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    out_dim=out_dim,
                    last_layer=(index == layers - 1),
                )
                for index in range(layers)
            ]
        )
        # Upstream calls initialize_weights() after building the blocks, which
        # overwrites the slice projections' orthogonal initialization; the
        # ordering is preserved deliberately.
        self.initialize_weights()
        self.placeholder = nn.Parameter((1.0 / hidden_dim) * torch.rand(hidden_dim))

    def initialize_weights(self) -> None:
        """Apply the official truncated-normal/zeros initialization."""

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        """Map coordinates ``(B, N, space_dim)`` and features ``(B, N, fun_dim)``
        to the output field ``(B, N, out_dim)``."""

        if fx is not None:
            fx = self.preprocess(torch.cat((x, fx), dim=-1))
        else:
            fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx


def build_token_sequence(
    domain: DomainMesh,
    *,
    device: torch.device,
    dtype: torch.dtype,
    global_feature_keys: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the shared boundary+query token sequence of the adaptation contract.

    Returns the normalized coordinates ``(1, S + Q, D)``, the token features
    ``(1, S + Q, D + 3 + len(global_feature_keys))``, and the number ``S`` of
    boundary tokens; the module docstring documents the normalization and
    feature layout for the 2D bank (``D = 2``, 5 function channels).  The
    layout generalizes verbatim to any spatial dimension: the outward unit
    normal contributes ``D`` channels and the cell measure is
    nondimensionalized by ``reference_length ** (D - 1)`` (panel length over
    length in 2D, triangle area over length squared in 3D).  Each key in
    ``global_feature_keys`` appends one scalar from ``domain.global_data`` as
    a *constant* extra channel on every token (boundary and query alike) --
    the steel-man injection for global operator parameters such as the
    screened-Laplace :math:`\\tilde\\kappa`, mirroring how the official
    Transolver feeds global design parameters as per-point features so the
    scalar is visible to slice assignment and attention in every block.  Both
    the ported arm here and the in-tree arm (:mod:`transolver_intree`) consume
    exactly this sequence, so the arms differ only in model internals.
    """

    boundary = _benchmark_boundary(domain)
    length = _reference_length(domain)
    n_dims = boundary.points.shape[-1]
    with torch.autocast(device_type=boundary.points.device.type, enabled=False):
        measures = boundary.cell_areas / length ** (n_dims - 1)
        center = torch.einsum("s,sd->d", measures, boundary.cell_centroids)
        center = center / measures.sum()
        source_points = (boundary.cell_centroids - center) / length
        query_points = (domain.interior.points - center) / length
        normals = boundary.cell_normals
    values = boundary.cell_data["boundary_value"]

    to_model = {"device": device, "dtype": dtype}
    source_points = source_points.to(**to_model)
    query_points = query_points.to(**to_model)
    boundary_features = torch.cat(
        (
            normals.to(**to_model),
            measures.to(**to_model)[:, None],
            values.to(**to_model)[:, None],
            source_points.new_ones(source_points.shape[0], 1),
        ),
        dim=-1,
    )
    query_features = query_points.new_zeros(query_points.shape[0], n_dims + 3)

    coordinates = torch.cat((source_points, query_points), dim=0)
    features = torch.cat((boundary_features, query_features), dim=0)
    if global_feature_keys:
        extras = torch.stack(
            [
                domain.global_data[key].reshape(()).to(**to_model)
                for key in global_feature_keys
            ]
        )
        features = torch.cat(
            (features, extras[None, :].expand(features.shape[0], -1)), dim=-1
        )
    return coordinates[None], features[None], source_points.shape[0]


class TransolverLaplaceAdapter(nn.Module):
    """Run the official Transolver on the exact conformal-Laplace protocol.

    Boundary cells and interior queries form one shared token sequence (see
    module docstring for the full adaptation contract); the prediction is read
    at the query tokens.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = Transolver(
            space_dim=2,
            # Outward normal (2), dimensionless cell measure, boundary value,
            # and the binary boundary/query indicator.
            fun_dim=5,
            out_dim=1,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        parameter = next(self.model.parameters())
        coordinates, features, n_boundary = build_token_sequence(
            domain, device=parameter.device, dtype=parameter.dtype
        )
        output = self.model(coordinates, features)  # (1, S + Q, 1)
        potential = output[0, n_boundary:, 0]
        return _prediction_mesh(domain, potential)


@dataclass(frozen=True)
class TransolverPreset:
    """One declared Transolver capacity setting."""

    hidden_dim: int
    layers: int
    heads: int
    slice_num: int
    mlp_ratio: int


TRANSOLVER_PRESETS: dict[str, TransolverPreset] = {
    # 103,053 parameters: within 1.2% of the 104,261-parameter
    # mesh_transformer_kernel_singonly reference arm.
    "matched": TransolverPreset(
        hidden_dim=64, layers=3, heads=4, slice_num=64, mlp_ratio=2
    ),
    # 2,809,409 parameters: Transolver's published PDE-benchmark scale
    # (hidden 256, 8 heads, 8 layers, 32 slices, official default mlp_ratio=1).
    "native": TransolverPreset(
        hidden_dim=256, layers=8, heads=8, slice_num=32, mlp_ratio=1
    ),
}


def build_transolver(preset: str) -> TransolverLaplaceAdapter:
    """Construct the benchmark-adapted Transolver at one declared capacity."""

    try:
        settings = TRANSOLVER_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown Transolver preset {preset!r}; "
            f"expected one of {tuple(TRANSOLVER_PRESETS)}"
        ) from None
    return TransolverLaplaceAdapter(**asdict(settings))


__all__ = [
    "PhysicsAttention",
    "Transolver",
    "TransolverBlock",
    "TransolverLaplaceAdapter",
    "TransolverPreset",
    "TRANSOLVER_PRESETS",
    "build_token_sequence",
    "build_transolver",
]
