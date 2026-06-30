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

r"""Pre-norm transformer block over mesh tokens, built on :class:`MeshAttention`.

See :class:`MeshTransformerBlock`; the package ``README.md`` gives the full
design write-up (how it works, motivation, and tradeoffs).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.mesh.spatial.cluster_tree import ClusterTree, DualInteractionPlan
from physicsnemo.nn import Mlp

from .attention import MeshAttention, QKNorm


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

    .. note::

        Known limitation: there is no scale control on the equivariant path.
        The vector stream receives residual updates from an *unnormalized*
        integral operator with no norm of its own, and the vector-derived Gram
        invariants entering attention are unnormalized quadratic features
        concatenated onto layer-normed scalars. In deep stacks the vector
        magnitudes (and with them the Gram features) can drift or grow;
        monitor them, and consider an equivariant vector norm (PaiNN-style:
        rescale each vector channel by the RMS of its norms) if this becomes
        a problem in practice.

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
    leaf_size : int, optional, default=1
        Leaf size for trees built on the fly (see :class:`MeshAttention`).

    Forward
    -------
    scalars : Float[torch.Tensor, "n scalar_dim"]
        Scalar token features.
    positions : Float[torch.Tensor, "n n_dims"]
        Token coordinates (cell centroids).
    vectors : Float[torch.Tensor, "n vector_dim n_dims"] | None
        Vector token features, or ``None`` when ``vector_dim == 0``.
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
    >>> s_out, v_out = block(s, p, v)
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
        leaf_size: int = 1,
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
            leaf_size=leaf_size,
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
        positions: Float[torch.Tensor, "n n_dims"],
        vectors: Float[torch.Tensor, "n vector_dim n_dims"] | None = None,
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
        # Self-attention on the pre-normalized scalar stream (queries == keys ==
        # values == this token set). Vectors are passed raw; the attention layer
        # forms its own equivariant value and invariants from them.
        attn_scalars, attn_vectors = self.attn(
            self.ln_attn(scalars),
            positions,
            vectors,
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
