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

r"""Mesh attention: hierarchical, equivariant, distance-decaying attention on meshes.

This module provides :class:`MeshAttention`, an attention layer whose tokens are
mesh cells (or, more generally, points carrying a quadrature weight).  It is the
attention-mechanism reinterpretation of GLOBE's dual-tree Barnes-Hut kernel
machinery:

- The attention weight between two tokens decays with physical distance via a
  learnable radial envelope, so the interaction matrix is hierarchically
  low-rank off-diagonal and can be evaluated in :math:`O(N \log N)` using the
  same :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` dual-tree
  traversal that powers GLOBE.
- The operator is **unnormalized** (no softmax): it is a learnable integral
  operator ``o(x_i) = sum_j w_ij v_j``, the physically-correct object for
  PDE-operator learning (cf. the Galerkin Transformer, Cao NeurIPS 2021).
- It is **equivariant** by construction: attention logits are built from
  invariants (a content dot-product of scalar features plus a distance decay),
  and values carry a separate scalar (invariant) and vector (equivariant) path,
  so scalar outputs are invariant and vector outputs are rotation/parity
  equivariant.

The near-field-exact / far-field-low-rank decomposition follows the
fast-multipole family of efficient transformers (FMMformer, H-Transformer,
Fast Multipole Attention), here generalized to 3D unstructured meshes with a
physically-grounded score.

See Also
--------
:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` : the spatial tree
    used to accelerate the attention.
:class:`~physicsnemo.experimental.nn.flare_attention.FLARE` : a different
    linear-cost attention (global query routing) in the same package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.nn import Mlp
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.cluster_tree import (
    ClusterTree,
    DualInteractionPlan,
    SourceAggregates,
)

QKNorm = Literal["layernorm", "cosine", "none"]
FarField = Literal["m0", "m0+m1"]


class RadialDecay(nn.Module):
    r"""Per-head learnable radial decay envelope ``g_h(r)``.

    Computes a smooth, monotonically-decreasing function of squared distance,
    one independent length scale per attention head:

    .. math::

        g_h(r) = \left(1 + (r / \ell_h)^2\right)^{-p}

    with ``g_h(0) = 1`` (strong local attention) and algebraic far-field decay
    ``g_h(r) ~ r^{-2p}``.  The envelope is integrable over :math:`\mathbb{R}^D`
    when ``2p > D``; the default ``p = 2`` therefore works for 2D and 3D.  The
    per-head length scale :math:`\ell_h` is stored as ``log_lengthscale`` to
    keep it strictly positive, and is initialized across a geometric range so
    that heads naturally specialize to different spatial scales ("heads as
    scales", replacing GLOBE's explicit multiscale branches).

    Parameters
    ----------
    num_heads : int
        Number of attention heads (independent length scales).
    p : float, optional, default=2.0
        Decay exponent. For integrability in :math:`D` dims, use ``2p > D``.
    lengthscale_init : float | Sequence[float], optional, default=1.0
        Initial length scale(s). A scalar is spread geometrically across heads
        (``[0.1, ..., 10] * lengthscale_init``); a sequence sets each head
        explicitly and must have length ``num_heads``.

    Forward
    -------
    r_sq : Float[torch.Tensor, "..."]
        Squared distances of arbitrary shape :math:`(\dots)`.

    Outputs
    -------
    Float[torch.Tensor, "... heads"]
        Decay values of shape :math:`(\dots, H)`.
    """

    def __init__(
        self,
        num_heads: int,
        p: float = 2.0,
        lengthscale_init: float | Sequence[float] = 1.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads=!r}")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p=!r}")

        self.num_heads = num_heads
        self.p = float(p)

        if isinstance(lengthscale_init, (int, float)):
            if num_heads == 1:
                lengthscales = torch.tensor([float(lengthscale_init)])
            else:
                # Geometric spread from 0.1x to 10x the nominal scale.
                lengthscales = float(lengthscale_init) * torch.logspace(
                    -1.0, 1.0, num_heads
                )
        else:
            lengthscales = torch.as_tensor(lengthscale_init, dtype=torch.float32)
            if lengthscales.shape != (num_heads,):
                raise ValueError(
                    f"lengthscale_init sequence must have length {num_heads=!r}, "
                    f"got shape {tuple(lengthscales.shape)}"
                )
        if (lengthscales <= 0).any():
            raise ValueError("All length scales must be positive.")

        self.log_lengthscale = nn.Parameter(lengthscales.log())

    def forward(
        self, r_sq: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... heads"]:
        r"""Evaluate the per-head decay at squared distances ``r_sq``."""
        # inv_l2_h = 1 / ell_h^2, shape (H,)
        inv_l2 = torch.exp(-2.0 * self.log_lengthscale)
        # (..., 1) * (H,) -> (..., H)
        scaled = r_sq.unsqueeze(-1) * inv_l2
        return (1.0 + scaled).pow(-self.p)


class MeshAttention(nn.Module):
    r"""Hierarchical, equivariant, distance-decaying attention over mesh tokens.

    Tokens are mesh cells (or points) with positions :math:`x_i` (cell
    centroids) and quadrature weights :math:`\alpha_i` (cell areas).  Token
    state is *rank-typed*: ``scalars`` (rotation-invariant) and ``vectors``
    (rotation-equivariant).  For attention heads :math:`h = 1 \dots H` with head
    dimension :math:`d`, the layer computes the unnormalized integral operator

    .. math::

        o_{i,h} = \sum_j w^h_{ij}\, v_{j,h},
        \qquad
        w^h_{ij} = \left(a_h\, \langle \phi(q^h_i), \phi(k^h_j)\rangle
                          + b_h\right)\, g_h(\lVert x_i - x_j \rVert)\, \alpha_j,

    where :math:`q, k` are projected from ``scalars`` (so the score is
    invariant), :math:`\phi` is the per-token normalization selected by
    ``qk_norm``, :math:`g_h` is a learnable radial decay
    (:class:`RadialDecay`), and :math:`a_h, b_h` are learnable per-head content
    and baseline gains.  The value :math:`v_{j,h}` packs a scalar part (from
    ``scalars``) and an equivariant vector part (a bias-free linear combination
    of ``vectors``); because :math:`w^h_{ij}` is invariant, scalar outputs are
    invariant and vector outputs are equivariant.

    The sum is evaluated in :math:`O(N \log N)` via a dual-tree Barnes-Hut
    traversal: near pairs are exact and carry the full (content + baseline)
    weight, while far interactions keep only the content-free baseline term and
    are approximated by area-weighted cluster monopoles (``far_field="m0"``).
    Setting ``theta=0`` (or calling :meth:`forward_reference`) makes every
    interaction exact and recovers dense attention; this is the correctness
    oracle.

    Parameters
    ----------
    scalar_dim : int
        Number of input scalar (invariant) channels.
    vector_dim : int, optional, default=0
        Number of input vector (equivariant) channels. ``0`` disables the
        vector path.
    heads : int, optional, default=8
        Number of attention heads.
    dim_head : int, optional, default=32
        Dimension of each attention head (and of the per-head scalar value).
    out_scalar_dim : int | None, optional, default=None
        Number of output scalar channels. Defaults to ``scalar_dim``.
    out_vector_dim : int | None, optional, default=None
        Number of output vector channels. Defaults to ``vector_dim``. Must be
        ``0`` if ``vector_dim == 0`` (equivariant vectors cannot be created
        from scalars).
    vector_invariants : bool, optional, default=True
        If ``True`` (and ``vector_dim > 0``), augment the scalar features with
        rotation-invariant features derived from the input vectors (the upper
        triangle of their per-token Gram matrix) before projecting queries,
        keys, and scalar values. This lets vectors influence the attention
        scores and scalar outputs while preserving equivariance.
    qk_norm : {"layernorm", "cosine", "none"}, optional, default="layernorm"
        Per-token normalization :math:`\phi` applied to queries/keys before the
        content dot-product. ``"layernorm"`` (the default, following the
        Galerkin Transformer) preserves magnitude/scale propagation;
        ``"cosine"`` (Swin V2) hard-bounds the score to :math:`[-1, 1]`;
        ``"none"`` is the raw dot-product.
    decay_p : float, optional, default=2.0
        Exponent of the radial decay envelope (see :class:`RadialDecay`).
    lengthscale_init : float | Sequence[float], optional, default=1.0
        Initial per-head decay length scale(s) (see :class:`RadialDecay`).
        Positions should be nondimensionalized so this is :math:`O(1)`.
    far_field : {"m0", "m0+m1"}, optional, default="m0"
        Far-field model. ``"m0"`` keeps only the content-free baseline at range
        (cheap, content selection is local). ``"m0+m1"`` additionally carries
        the content moment at range (not yet implemented; reserved).
    mass_normalize : bool, optional, default=False
        If ``True``, divide the output by the content-free envelope mass
        :math:`\sum_j g_h(r_{ij})\,\alpha_j` (a geometry-only, Galerkin-style
        normalizer; **not** a softmax). Off by default to keep the pure
        integral-operator form.
    content_gain_init : float, optional, default=0.1
        Initial value of the per-head content gain :math:`a_h` (small, so the
        layer starts close to a geometric value-smoother).
    baseline_gain_init : float, optional, default=1.0
        Initial value of the per-head baseline gain :math:`b_h`.
    eps : float, optional, default=1e-6
        Numerical floor used by ``cosine`` normalization and ``mass_normalize``.

    Forward
    -------
    scalars : Float[torch.Tensor, "n_src scalar_dim"]
        Source (key/value) scalar features, shape :math:`(N_s, C_s)`.
    vectors : Float[torch.Tensor, "n_src vector_dim n_dims"] | None
        Source vector features, shape :math:`(N_s, V, D)`, or ``None`` if
        ``vector_dim == 0``.
    positions : Float[torch.Tensor, "n_src n_dims"]
        Source point coordinates, shape :math:`(N_s, D)`.
    areas : Float[torch.Tensor, "n_src"] | None, optional
        Per-source quadrature weights, shape :math:`(N_s,)`. Defaults to ones.
    source_tree, target_tree : ClusterTree | None, optional
        Precomputed source/target trees. Built on the fly if ``None``.
    plan : DualInteractionPlan | None, optional
        Precomputed dual-tree plan. Computed on the fly if ``None``.
    query_scalars : Float[torch.Tensor, "n_tgt scalar_dim"] | None, optional
        Query scalar features for cross-attention. If ``None``, the layer does
        self-attention (queries equal sources).
    query_vectors : Float[torch.Tensor, "n_tgt vector_dim n_dims"] | None, optional
        Query vector features for cross-attention, used only to form query-side
        vector invariants when ``vector_invariants`` is enabled. ``None`` (the
        default) zero-fills that block for query points without vectors.
    query_positions : Float[torch.Tensor, "n_tgt n_dims"] | None, optional
        Query coordinates for cross-attention. Defaults to ``positions``.
    theta : float, optional, default=1.0
        Barnes-Hut opening angle (larger = more far-field approximation).
    source_aggregates : SourceAggregates | None, optional
        Precomputed per-node source aggregates. Computed on the fly if ``None``.

    Outputs
    -------
    tuple[Float[torch.Tensor, "n_tgt out_scalar_dim"], Float[torch.Tensor, "n_tgt out_vector_dim n_dims"] | None]
        The output scalar features and (if ``out_vector_dim > 0``) vector
        features at the query points; the vector output is ``None`` otherwise.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import MeshAttention
    >>> layer = MeshAttention(scalar_dim=16, vector_dim=2, heads=4, dim_head=8)
    >>> n, d = 200, 3
    >>> scalars = torch.randn(n, 16)
    >>> vectors = torch.randn(n, 2, d)
    >>> positions = torch.randn(n, d)
    >>> out_s, out_v = layer(scalars, vectors, positions)
    >>> out_s.shape, out_v.shape
    (torch.Size([200, 16]), torch.Size([200, 2, 3]))
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int = 0,
        heads: int = 8,
        dim_head: int = 32,
        out_scalar_dim: int | None = None,
        out_vector_dim: int | None = None,
        vector_invariants: bool = True,
        qk_norm: QKNorm = "layernorm",
        decay_p: float = 2.0,
        lengthscale_init: float | Sequence[float] = 1.0,
        far_field: FarField = "m0",
        mass_normalize: bool = False,
        content_gain_init: float = 0.1,
        baseline_gain_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if out_scalar_dim is None:
            out_scalar_dim = scalar_dim
        if out_vector_dim is None:
            out_vector_dim = vector_dim
        if vector_dim == 0 and out_vector_dim != 0:
            raise ValueError(
                "out_vector_dim must be 0 when vector_dim == 0: equivariant "
                "vector outputs cannot be created from scalar inputs alone "
                f"(got vector_dim=0, out_vector_dim={out_vector_dim})."
            )
        if qk_norm not in ("layernorm", "cosine", "none"):
            raise ValueError(
                f"qk_norm must be 'layernorm', 'cosine', or 'none', got {qk_norm!r}"
            )
        if far_field not in ("m0", "m0+m1"):
            raise ValueError(f"far_field must be 'm0' or 'm0+m1', got {far_field!r}")
        if far_field == "m0+m1":
            raise NotImplementedError(
                "far_field='m0+m1' (content-carrying far field) is reserved but "
                "not yet implemented; use 'm0'."
            )

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        self.qk_norm = qk_norm
        self.far_field = far_field
        self.mass_normalize = mass_normalize
        self.eps = eps

        inner_dim = heads * dim_head

        # Optionally augment the invariant scalar features with rotation-
        # invariant features derived from the input vectors (the upper triangle
        # of their per-token Gram matrix). This lets vectors inform the
        # attention scores and the scalar value path while preserving
        # equivariance (dot products of vectors are O(D)-invariant).
        self.vector_invariants = bool(vector_invariants) and vector_dim > 0
        self._n_vec_inv = (
            vector_dim * (vector_dim + 1) // 2 if self.vector_invariants else 0
        )
        qk_in_dim = scalar_dim + self._n_vec_inv

        # Query/key/scalar-value projections (from invariant features).
        self.to_q = nn.Linear(qk_in_dim, inner_dim)
        self.to_k = nn.Linear(qk_in_dim, inner_dim)
        self.to_v = nn.Linear(qk_in_dim, inner_dim)

        # Equivariant vector value: a bias-free linear mix of input vector
        # channels into one value-vector per head. Shape (H, V); applied as
        # ``einsum('hv,nvd->nhd')`` so the spatial dimension d rides along and
        # the map is dimension-generic and equivariant.
        if vector_dim > 0:
            self.vec_value = nn.Parameter(
                torch.randn(heads, vector_dim) / math.sqrt(vector_dim)
            )
        else:
            self.register_parameter("vec_value", None)

        # Q/K normalization phi.
        if qk_norm == "layernorm":
            self.q_norm: nn.Module = nn.LayerNorm(dim_head)
            self.k_norm: nn.Module = nn.LayerNorm(dim_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        # Cosine is already bounded; raw/layernorm get the conventional 1/sqrt(d).
        self.score_scale = 1.0 if qk_norm == "cosine" else dim_head**-0.5

        # Per-head content (a_h) and baseline (b_h) gains.
        self.content_gain = nn.Parameter(torch.full((heads,), float(content_gain_init)))
        self.baseline_gain = nn.Parameter(
            torch.full((heads,), float(baseline_gain_init))
        )

        self.decay = RadialDecay(heads, p=decay_p, lengthscale_init=lengthscale_init)

        # Output projections.
        self.to_out_scalar = nn.Linear(inner_dim, out_scalar_dim)
        if out_vector_dim > 0:
            # Equivariant head-combine for the vector output, (out_V, H), no bias.
            self.vec_out = nn.Parameter(
                torch.randn(out_vector_dim, heads) / math.sqrt(heads)
            )
        else:
            self.register_parameter("vec_out", None)

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _norm_q(
        self, q: Float[torch.Tensor, "n heads dim"]
    ) -> Float[torch.Tensor, "n heads dim"]:
        if self.qk_norm == "cosine":
            return F.normalize(q, dim=-1, eps=self.eps)
        return self.q_norm(q)

    def _norm_k(
        self, k: Float[torch.Tensor, "n heads dim"]
    ) -> Float[torch.Tensor, "n heads dim"]:
        if self.qk_norm == "cosine":
            return F.normalize(k, dim=-1, eps=self.eps)
        return self.k_norm(k)

    def _vector_invariants(
        self, vectors: Float[torch.Tensor, "n vector_dim n_dims"]
    ) -> Float[torch.Tensor, "n n_invariants"]:
        """Rotation-invariant per-token features: the Gram upper triangle.

        Returns the :math:`V(V+1)/2` distinct dot products
        :math:`\\langle v_c, v_{c'} \\rangle` (``c <= c'``) per token, which are
        invariant under any orthogonal transform of the spatial axes.
        """
        gram = torch.einsum("nvd,nwd->nvw", vectors, vectors)  # (n, V, V)
        iu = torch.triu_indices(
            self.vector_dim, self.vector_dim, device=vectors.device
        )
        return gram[:, iu[0], iu[1]]  # (n, V*(V+1)/2)

    def _augment(
        self,
        scalars: Float[torch.Tensor, "n scalar_dim"],
        vectors: Float[torch.Tensor, "n vector_dim n_dims"] | None,
    ) -> Float[torch.Tensor, "n aug_dim"]:
        """Concatenate vector-derived invariants onto the scalar features.

        A no-op when ``vector_invariants`` is disabled. When enabled but
        ``vectors is None`` (e.g. cross-attention query points without vector
        features), the invariant block is zero-filled.
        """
        if not self.vector_invariants:
            return scalars
        if vectors is None:
            inv = scalars.new_zeros(scalars.shape[0], self._n_vec_inv)
        else:
            inv = self._vector_invariants(vectors)
        return torch.cat([scalars, inv], dim=-1)

    def _project_qk(
        self,
        query_scalars: Float[torch.Tensor, "n_tgt aug_dim"],
        key_scalars: Float[torch.Tensor, "n_src aug_dim"],
    ) -> tuple[
        Float[torch.Tensor, "n_tgt heads dim"], Float[torch.Tensor, "n_src heads dim"]
    ]:
        """Project and normalize queries (from targets) and keys (from sources).

        Inputs are the (possibly vector-invariant-augmented) invariant features.
        """
        H, d = self.heads, self.dim_head
        q = self._norm_q(self.to_q(query_scalars).reshape(-1, H, d))
        k = self._norm_k(self.to_k(key_scalars).reshape(-1, H, d))
        return q, k

    def _project_value(
        self,
        key_scalars: Float[torch.Tensor, "n_src aug_dim"],
        key_vectors: Float[torch.Tensor, "n_src vector_dim n_dims"] | None,
    ) -> Float[torch.Tensor, "n_src heads value_dim"]:
        r"""Build the packed per-head value ``[scalar_value | vector_value]``.

        The scalar value is a standard linear projection (dimension
        ``dim_head``); the vector value is a bias-free, equivariant linear
        combination of the source vector channels (dimension :math:`D`). They
        are concatenated so a single weighted sum produces both output paths,
        with the vector sub-block remaining equivariant because the attention
        weights are invariant.
        """
        H, d = self.heads, self.dim_head
        n_src = key_scalars.shape[0]
        v_scalar = self.to_v(key_scalars).reshape(n_src, H, d)
        if self.vector_dim > 0:
            # (H, V) x (n_src, V, D) -> (n_src, H, D); equivariant in D.
            v_vector = torch.einsum("hv,nvd->nhd", self.vec_value, key_vectors)
            return torch.cat([v_scalar, v_vector], dim=-1)
        return v_scalar

    def _split_output(
        self, out: Float[torch.Tensor, "n_tgt heads value_dim"]
    ) -> tuple[
        Float[torch.Tensor, "n_tgt out_scalar_dim"],
        Float[torch.Tensor, "n_tgt out_vector_dim n_dims"] | None,
    ]:
        """Split the packed per-head output into scalar and vector outputs."""
        H, d = self.heads, self.dim_head
        n_tgt = out.shape[0]
        out_scalar = self.to_out_scalar(out[..., :d].reshape(n_tgt, H * d))
        if self.out_vector_dim > 0:
            # (out_V, H) x (n_tgt, H, D) -> (n_tgt, out_V, D); equivariant.
            out_vector = torch.einsum("mh,nhd->nmd", self.vec_out, out[..., d:])
            return out_scalar, out_vector
        return out_scalar, None

    # ------------------------------------------------------------------
    # Dense reference (correctness oracle; also fine for small inputs)
    # ------------------------------------------------------------------

    def _dense_weighted_values(
        self,
        q: Float[torch.Tensor, "n_tgt heads dim"],
        k: Float[torch.Tensor, "n_src heads dim"],
        value: Float[torch.Tensor, "n_src heads value_dim"],
        query_positions: Float[torch.Tensor, "n_tgt n_dims"],
        key_positions: Float[torch.Tensor, "n_src n_dims"],
        areas: Float[torch.Tensor, " n_src"],
    ) -> Float[torch.Tensor, "n_tgt heads value_dim"]:
        r"""Brute-force :math:`O(N^2)` evaluation of ``sum_j w_ij v_j``.

        This materializes the full :math:`(N_t, N_s, H)` weight matrix and is
        used as the correctness oracle (it equals the hierarchical forward at
        ``theta = 0``) and as a fallback for small problems.
        """
        # Content similarity <phi(q), phi(k)>, scaled. (N_t, N_s, H)
        score = torch.einsum("thd,shd->tsh", q, k) * self.score_scale
        # Squared distances and the per-head decay. (N_t, N_s) -> (N_t, N_s, H)
        r_sq = (query_positions[:, None, :] - key_positions[None, :, :]).pow(2).sum(-1)
        g = self.decay(r_sq)
        # Geometric/quadrature factor g * alpha. (N_t, N_s, H)
        geom = g * areas[None, :, None]
        # Full weight (a*score + b) * g * alpha. (N_t, N_s, H)
        weight = (self.content_gain * score + self.baseline_gain) * geom
        out = torch.einsum("tsh,shf->thf", weight, value)
        if self.mass_normalize:
            mass = geom.sum(dim=1)  # (N_t, H)
            out = out / (mass.unsqueeze(-1) + self.eps)
        return out

    # ------------------------------------------------------------------
    # Hierarchical (dual-tree) evaluation
    # ------------------------------------------------------------------

    def _hierarchical_weighted_values(
        self,
        q: Float[torch.Tensor, "n_tgt heads dim"],
        k: Float[torch.Tensor, "n_src heads dim"],
        value: Float[torch.Tensor, "n_src heads value_dim"],
        query_positions: Float[torch.Tensor, "n_tgt n_dims"],
        key_positions: Float[torch.Tensor, "n_src n_dims"],
        areas: Float[torch.Tensor, " n_src"],
        source_tree: ClusterTree,
        target_tree: ClusterTree,
        plan: DualInteractionPlan,
        source_aggregates: SourceAggregates | None,
    ) -> Float[torch.Tensor, "n_tgt heads value_dim"]:
        r"""Dual-tree Barnes-Hut evaluation of ``sum_j w_ij v_j``.

        The operator is split into

        * a **content-free baseline** ``B_i = sum_j g_h(r_ij) alpha_j v_j``,
          evaluated over all sources using the four dual-tree interaction
          categories (near pairs exact; far interactions via area-weighted
          cluster monopoles ``M0``), and
        * a **near-only content correction**
          ``C_i = sum_{j near i} <phi(q_i),phi(k_j)> g_h(r_ij) alpha_j v_j``.

        The result is ``a_h * C + b_h * B`` (optionally divided by the
        content-free mass).  A unit column is appended to the value so the
        baseline pass simultaneously accumulates the envelope mass used by
        ``mass_normalize``.
        """
        H = self.heads
        n_tgt = query_positions.shape[0]
        n_src = key_positions.shape[0]
        device = query_positions.device
        Fv = value.shape[-1]

        # Augment the value with a per-head unit column so the same baseline
        # accumulation also yields the envelope mass Z = sum_j g_h * alpha.
        ones_col = areas.new_ones(n_src, H, 1)
        value_aug = torch.cat([value, ones_col], dim=-1)  # (N_s, H, Fv+1)
        Fa = Fv + 1

        # Per-node area-weighted source aggregates (centroids + value means).
        if source_aggregates is None:
            source_aggregates = source_tree.compute_source_aggregates(
                source_points=key_positions,
                areas=areas,
                source_data=TensorDict(
                    {"value": value_aug}, batch_size=[n_src], device=device
                ),
            )
        src_centroids = source_aggregates.node_centroid  # (n_src_nodes, D)
        # node_source_data holds the area-weighted MEAN; multiply by total area
        # to recover the area-weighted SUM (the M0 monopole moment).
        node_mean = source_aggregates.node_source_data["value"]  # (n_src_nodes,H,Fa)
        m0 = node_mean * source_tree.node_total_area[:, None, None]

        # Target node centroids (reused from sources for self-interaction).
        if target_tree is source_tree:
            tgt_centroids = src_centroids
        else:
            tgt_centroids = target_tree.compute_source_aggregates(
                source_points=query_positions,
                areas=query_positions.new_ones(n_tgt),
                source_data=None,
            ).node_centroid

        # Flat accumulation buffers (index_add_ wants 2D).
        buf_baseline = query_positions.new_zeros(n_tgt, H * Fa)
        buf_content = query_positions.new_zeros(n_tgt, H * Fv)

        def _scatter(buf: torch.Tensor, tgt_ids: torch.Tensor, contrib: torch.Tensor):
            # contrib: (n, H, F) -> (n, H*F); accumulate at tgt_ids.
            buf.index_add_(0, tgt_ids, contrib.reshape(contrib.shape[0], -1))

        # --- Phase: near (exact individual pairs) -------------------------
        # Carries BOTH the baseline (g*alpha*v) and the content correction
        # (<q,k>*g*alpha*v); the shared g*alpha*v factor is computed once.
        nt = plan.near_target_ids
        ns = plan.near_source_ids
        if nt.numel() > 0:
            r_sq = (query_positions[nt] - key_positions[ns]).pow(2).sum(-1)
            g = self.decay(r_sq)  # (n_near, H)
            g_alpha = g * areas[ns][:, None]  # (n_near, H)
            base_pair = g_alpha[:, :, None] * value_aug[ns]  # (n_near, H, Fa)
            _scatter(buf_baseline, nt, base_pair)
            score = (q[nt] * k[ns]).sum(-1) * self.score_scale  # (n_near, H)
            content_pair = (score * g_alpha)[:, :, None] * value[ns]  # (n_near,H,Fv)
            _scatter(buf_content, nt, content_pair)

        # --- Phase: (near target, far source node) ------------------------
        # Individual target attends to a source-cluster monopole. Baseline only.
        nf_t = plan.nf_target_ids
        nf_sn = plan.nf_source_node_ids
        if nf_t.numel() > 0:
            r_sq = (query_positions[nf_t] - src_centroids[nf_sn]).pow(2).sum(-1)
            g = self.decay(r_sq)  # (n_nf, H)
            _scatter(buf_baseline, nf_t, g[:, :, None] * m0[nf_sn])

        # --- Phase: (far target node, far source node) --------------------
        # Cluster-cluster monopole, broadcast to all targets in the node.
        far_tn = plan.far_target_node_ids
        far_sn = plan.far_source_node_ids
        if far_tn.numel() > 0:
            r_sq = (tgt_centroids[far_tn] - src_centroids[far_sn]).pow(2).sum(-1)
            g = self.decay(r_sq)  # (n_far, H)
            contrib = g[:, :, None] * m0[far_sn]  # (n_far, H, Fa)
            starts = target_tree.node_range_start[far_tn]
            counts = target_tree.node_range_count[far_tn]
            positions, pair_ids = _ragged_arange(starts, counts)
            expanded_tgt = target_tree.sorted_source_order[positions]
            _scatter(buf_baseline, expanded_tgt, contrib[pair_ids])

        # --- Phase: (far target node, near source) ------------------------
        # Individual source evaluated at the target-node centroid, broadcast to
        # the stage-1 survivor targets. Baseline only.
        fn_tn = plan.fn_target_node_ids
        fn_s = plan.fn_source_ids
        if fn_tn.numel() > 0:
            r_sq = (tgt_centroids[fn_tn] - key_positions[fn_s]).pow(2).sum(-1)
            g = self.decay(r_sq)  # (n_fn, H)
            g_alpha = g * areas[fn_s][:, None]
            contrib = g_alpha[:, :, None] * value_aug[fn_s]  # (n_fn, H, Fa)
            positions, pair_ids = _ragged_arange(
                plan.fn_broadcast_starts, plan.fn_broadcast_counts
            )
            expanded_tgt = plan.fn_broadcast_targets[positions]
            _scatter(buf_baseline, expanded_tgt, contrib[pair_ids])

        # --- Combine ------------------------------------------------------
        baseline = buf_baseline.reshape(n_tgt, H, Fa)
        content = buf_content.reshape(n_tgt, H, Fv)
        base_value = baseline[..., :Fv]  # (N_t, H, Fv)
        mass = baseline[..., Fv:]  # (N_t, H, 1) = sum_j g_h * alpha
        out = (
            self.content_gain[None, :, None] * content
            + self.baseline_gain[None, :, None] * base_value
        )
        if self.mass_normalize:
            out = out / (mass + self.eps)
        return out

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        scalars: Float[torch.Tensor, "n_src scalar_dim"],
        vectors: Float[torch.Tensor, "n_src vector_dim n_dims"] | None = None,
        positions: Float[torch.Tensor, "n_src n_dims"] = None,  # ty: ignore[invalid-parameter-default]
        areas: Float[torch.Tensor, " n_src"] | None = None,
        *,
        source_tree: ClusterTree | None = None,
        target_tree: ClusterTree | None = None,
        plan: DualInteractionPlan | None = None,
        query_scalars: Float[torch.Tensor, "n_tgt scalar_dim"] | None = None,
        query_vectors: Float[torch.Tensor, "n_tgt vector_dim n_dims"] | None = None,
        query_positions: Float[torch.Tensor, "n_tgt n_dims"] | None = None,
        theta: float = 1.0,
        source_aggregates: SourceAggregates | None = None,
    ) -> tuple[
        Float[torch.Tensor, "n_tgt out_scalar_dim"],
        Float[torch.Tensor, "n_tgt out_vector_dim n_dims"] | None,
    ]:
        r"""Evaluate mesh attention (hierarchical dual-tree).

        With only ``scalars``/``vectors``/``positions`` given, performs
        self-attention. Provide ``query_scalars``/``query_positions`` (and
        optionally a distinct ``target_tree``) for cross-attention to a separate
        set of query points. Trees and the interaction plan are built on the fly
        when not supplied; supply precomputed ones to amortize across layers.

        See the class docstring for parameter and output details.
        """
        if positions is None:
            raise ValueError("positions is required")
        self._validate_inputs(scalars, vectors, positions)

        if areas is None:
            areas = positions.new_ones(positions.shape[0])
        self_attention = query_scalars is None
        if query_scalars is None:
            query_scalars = scalars
        if query_positions is None:
            query_positions = positions

        # Augment invariant features with vector-derived invariants (when
        # enabled), then project (queries from targets; keys/values from
        # sources). For self-attention the query and source augmentations are
        # identical and computed once.
        aug_key = self._augment(scalars, vectors)
        aug_query = (
            aug_key
            if self_attention
            else self._augment(query_scalars, query_vectors)
        )
        q, k = self._project_qk(aug_query, key_scalars=aug_key)
        value = self._project_value(aug_key, vectors)

        # Build spatial structures on the fly if not provided.
        if source_tree is None:
            source_tree = ClusterTree.from_points(positions, areas=areas)
        if target_tree is None:
            target_tree = (
                source_tree
                if self_attention
                else ClusterTree.from_points(query_positions)
            )
        if plan is None:
            plan = source_tree.find_dual_interaction_pairs(
                target_tree=target_tree, theta=theta
            )

        out = self._hierarchical_weighted_values(
            q, k, value, query_positions, positions, areas,
            source_tree, target_tree, plan, source_aggregates,
        )
        return self._split_output(out)

    def forward_reference(
        self,
        scalars: Float[torch.Tensor, "n_src scalar_dim"],
        vectors: Float[torch.Tensor, "n_src vector_dim n_dims"] | None = None,
        positions: Float[torch.Tensor, "n_src n_dims"] = None,  # ty: ignore[invalid-parameter-default]
        areas: Float[torch.Tensor, " n_src"] | None = None,
        *,
        query_scalars: Float[torch.Tensor, "n_tgt scalar_dim"] | None = None,
        query_vectors: Float[torch.Tensor, "n_tgt vector_dim n_dims"] | None = None,
        query_positions: Float[torch.Tensor, "n_tgt n_dims"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "n_tgt out_scalar_dim"],
        Float[torch.Tensor, "n_tgt out_vector_dim n_dims"] | None,
    ]:
        r"""Dense :math:`O(N^2)` reference forward (correctness oracle).

        Computes the exact attention with no tree approximation, equal to
        :meth:`forward` at ``theta = 0``. Intended for tests and small inputs.
        """
        if positions is None:
            raise ValueError("positions is required")
        self._validate_inputs(scalars, vectors, positions)
        if areas is None:
            areas = positions.new_ones(positions.shape[0])
        self_attention = query_scalars is None
        if query_scalars is None:
            query_scalars = scalars
        if query_positions is None:
            query_positions = positions

        aug_key = self._augment(scalars, vectors)
        aug_query = (
            aug_key
            if self_attention
            else self._augment(query_scalars, query_vectors)
        )
        q, k = self._project_qk(aug_query, key_scalars=aug_key)
        value = self._project_value(aug_key, vectors)
        out = self._dense_weighted_values(
            q, k, value, query_positions, positions, areas
        )
        return self._split_output(out)

    def _validate_inputs(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        """Eager-mode shape/feature validation (skipped under torch.compile)."""
        if torch.compiler.is_compiling():
            return
        if scalars.ndim != 2 or scalars.shape[-1] != self.scalar_dim:
            raise ValueError(
                f"Expected scalars of shape (N, {self.scalar_dim}), "
                f"got {tuple(scalars.shape)}"
            )
        if positions.ndim != 2:
            raise ValueError(
                f"Expected positions of shape (N, D), got {tuple(positions.shape)}"
            )
        if self.vector_dim > 0:
            if vectors is None:
                raise ValueError(
                    f"This layer expects {self.vector_dim} vector channels, "
                    "but vectors=None was given."
                )
            if vectors.ndim != 3 or vectors.shape[1] != self.vector_dim:
                raise ValueError(
                    f"Expected vectors of shape (N, {self.vector_dim}, D), "
                    f"got {tuple(vectors.shape)}"
                )
            if vectors.shape[-1] != positions.shape[-1]:
                raise ValueError(
                    f"vectors spatial dim {vectors.shape[-1]} != positions "
                    f"spatial dim {positions.shape[-1]}"
                )
        elif vectors is not None:
            raise ValueError(
                "This layer was constructed with vector_dim=0 but received "
                "non-None vectors."
            )


class MeshTransformerBlock(nn.Module):
    r"""Pre-norm transformer block built on :class:`MeshAttention`.

    Applies mesh self-attention and a token-wise MLP with residual
    connections:

    .. math::

        s &\leftarrow s + \operatorname{MeshAttn}(\operatorname{LN}(s), v)_{\text{scalar}} \\
        v &\leftarrow v + \operatorname{MeshAttn}(\operatorname{LN}(s), v)_{\text{vector}} \\
        s &\leftarrow s + \operatorname{MLP}(\operatorname{LN}(s))

    The MLP and layer norms act only on the rotation-invariant scalar stream;
    the equivariant vector stream is updated solely through the attention
    residual, so the whole block is :math:`O(D)`-equivariant. The spatial tree
    and interaction plan depend only on geometry, so they can be built once and
    shared across a stack of blocks.

    Parameters
    ----------
    scalar_dim : int
        Number of scalar (invariant) channels; preserved across the block.
    vector_dim : int, optional, default=0
        Number of vector (equivariant) channels; preserved across the block.
    heads : int, optional, default=8
        Number of attention heads.
    dim_head : int, optional, default=32
        Dimension per attention head.
    mlp_ratio : int, optional, default=4
        Hidden width of the scalar MLP as a multiple of ``scalar_dim``.
    act : str, optional, default="gelu"
        Activation function for the MLP.
    qk_norm : {"layernorm", "cosine", "none"}, optional, default="layernorm"
        Query/key normalization passed to :class:`MeshAttention`.
    vector_invariants : bool, optional, default=True
        Whether attention augments scalars with vector-derived invariants.
    decay_p : float, optional, default=2.0
        Radial decay exponent for the attention envelope.
    lengthscale_init : float | Sequence[float], optional, default=1.0
        Initial per-head decay length scale(s).
    mass_normalize : bool, optional, default=False
        Whether attention divides by the content-free envelope mass.

    Forward
    -------
    scalars : Float[torch.Tensor, "n scalar_dim"]
        Scalar token features.
    vectors : Float[torch.Tensor, "n vector_dim n_dims"] | None
        Vector token features, or ``None`` when ``vector_dim == 0``.
    positions : Float[torch.Tensor, "n n_dims"]
        Token coordinates (cell centroids).
    areas : Float[torch.Tensor, "n"] | None, optional
        Per-token quadrature weights. Defaults to ones.
    source_tree : ClusterTree | None, optional
        Precomputed cluster tree (built on the fly if ``None``). Reuse across
        a stack of blocks for efficiency.
    plan : DualInteractionPlan | None, optional
        Precomputed dual-tree plan (computed on the fly if ``None``).
    theta : float, optional, default=1.0
        Barnes-Hut opening angle.

    Outputs
    -------
    tuple[Float[torch.Tensor, "n scalar_dim"], Float[torch.Tensor, "n vector_dim n_dims"] | None]
        Updated scalar and vector streams, with the same shapes as the inputs.

    See Also
    --------
    :class:`MeshAttention` : the attention layer this block wraps.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import MeshTransformerBlock
    >>> block = MeshTransformerBlock(scalar_dim=16, vector_dim=2, heads=4, dim_head=8)
    >>> s = torch.randn(128, 16)
    >>> v = torch.randn(128, 2, 3)
    >>> p = torch.randn(128, 3)
    >>> s_out, v_out = block(s, v, p)
    >>> s_out.shape, v_out.shape
    (torch.Size([128, 16]), torch.Size([128, 2, 3]))
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int = 0,
        heads: int = 8,
        dim_head: int = 32,
        mlp_ratio: int = 4,
        act: str = "gelu",
        qk_norm: QKNorm = "layernorm",
        vector_invariants: bool = True,
        decay_p: float = 2.0,
        lengthscale_init: float | Sequence[float] = 1.0,
        mass_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(scalar_dim)
        self.attn = MeshAttention(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            heads=heads,
            dim_head=dim_head,
            out_scalar_dim=scalar_dim,
            out_vector_dim=vector_dim,
            vector_invariants=vector_invariants,
            qk_norm=qk_norm,
            decay_p=decay_p,
            lengthscale_init=lengthscale_init,
            mass_normalize=mass_normalize,
        )
        self.ln_mlp = nn.LayerNorm(scalar_dim)
        self.mlp = Mlp(
            in_features=scalar_dim,
            hidden_features=scalar_dim * mlp_ratio,
            out_features=scalar_dim,
            act_layer=act,
        )

    def forward(
        self,
        scalars: Float[torch.Tensor, "n scalar_dim"],
        vectors: Float[torch.Tensor, "n vector_dim n_dims"] | None = None,
        positions: Float[torch.Tensor, "n n_dims"] = None,  # ty: ignore[invalid-parameter-default]
        areas: Float[torch.Tensor, " n"] | None = None,
        *,
        source_tree: ClusterTree | None = None,
        plan: DualInteractionPlan | None = None,
        theta: float = 1.0,
    ) -> tuple[
        Float[torch.Tensor, "n scalar_dim"],
        Float[torch.Tensor, "n vector_dim n_dims"] | None,
    ]:
        r"""Apply the block (self-attention + MLP) with residual connections."""
        if positions is None:
            raise ValueError("positions is required")

        # Self-attention on the pre-normalized scalar stream (queries == keys ==
        # values == this token set). Vectors are passed raw; the attention layer
        # forms its own equivariant value and invariants from them.
        attn_scalars, attn_vectors = self.attn(
            self.ln_attn(scalars),
            vectors,
            positions,
            areas,
            source_tree=source_tree,
            plan=plan,
            theta=theta,
        )
        scalars = scalars + attn_scalars
        if vectors is not None and attn_vectors is not None:
            vectors = vectors + attn_vectors

        # Token-wise MLP on the invariant scalar stream.
        scalars = scalars + self.mlp(self.ln_mlp(scalars))
        return scalars, vectors
