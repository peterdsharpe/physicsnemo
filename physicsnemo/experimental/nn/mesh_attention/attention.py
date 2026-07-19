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

r"""Global, rank-typed signed moment attention on a boundary mesh.

The mathematical operator in this module is deliberately small.  A source
``Mesh`` supplies the quadrature measure and scalar/vector key and value
features supply a finite-rank kernel.  Source moments are formed once and may
then be evaluated at any number of receivers without coupling the receivers to
one another.

No spatial neighbourhood, radial cutoff, softmax, or tree is part of the
operator.  Hierarchical acceleration for a future non-separable kernel belongs
behind a separate numerical backend and must converge to a dense oracle.

The pseudoscalar (``0o``) sector
--------------------------------

The type system carries three sectors: true scalars (``0e``), polar vectors
(``1o``), and -- in two spatial dimensions only -- pseudoscalars (``0o``,
rotation invariant, sign-flipping under reflection).  The pseudoscalar sector
is the program's first type-system extension, forced by two measured failures
of the original ``{0e, 1o}`` system on the exterior potential-flow benchmark:

1. **Streamfunction targets are pseudoscalars and produced a provable
   no-response.**  Every scalar the ``{0e, 1o}`` system can emit is built
   from dot products of polar vectors and is therefore mirror-even, while
   the disturbance streamfunction is mirror-odd (its uniform-flow part is
   the wedge :math:`U \wedge x`).  The only O(2)-equivariant fit of an odd
   target by an even model is the zero function: every trained global-drive
   arm sat at relative L2 :math:`\approx 1.0`.
2. **Even with polar-vector outputs, circulation was unreachable.**  The
   circulation component of the velocity is
   :math:`\Gamma\, x^\perp / (2\pi\lvert x\rvert^2)`.  With :math:`\Gamma`
   typed as a true scalar the product :math:`\Gamma\, x^\perp` is axial
   (parity-odd where a polar vector must be parity-even under the combined
   data-and-frame mirror) and hence unrepresentable; circulation-OOD was
   pinned at relative L2 0.647 across all velocity arms and seeds while
   in-distribution reached 0.14.

The complete closed product set over ``{0e (s), 0o (p), 1o (v)}`` in 2D adds
exactly four typed products to the existing dot-product algebra:

- wedge: :math:`v \wedge w = v_x w_y - v_y w_x \to 0o`
  (:func:`_wedge_invariants`, :func:`_pair_wedges`);
- rotation: :math:`p \cdot v^\perp` with :math:`v^\perp = (-v_y, v_x)
  \to 1o` (:func:`_vector_perp`; how :math:`\Gamma\,x^\perp` becomes
  representable);
- :math:`p \cdot q \to 0e` (:func:`_pseudo_pair_invariants`);
- :math:`s \cdot p \to 0o` (plain products against pseudo bases).

Pseudoscalars behave exactly like scalars under rotations, so the attention
and moment machinery treats them identically (they ride alongside the scalar
value features); only invariant formation and the product rules differ.  In
3D the analogous parity-odd object is the axial vector, which is out of
scope: pseudoscalar channels are rejected outside 2D rather than silently
mistreated.  With every pseudo width at zero this module is bitwise
identical to its pre-extension behavior.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.integration import (
    _integrate_weighted_moment,
    integrate_moment,
)


@dataclass(frozen=True)
class ScalarVectorState:
    r"""A packed collection of invariant scalars, polar vectors, and (in 2D)
    pseudoscalars.

    ``scalars`` has shape ``(N, C_s)``, ``vectors`` has shape ``(N, C_v, D)``,
    and ``pseudos`` has shape ``(N, C_p)``.  Zero channels of any sector are
    represented by an empty tensor, not ``None``; this keeps compiled call
    signatures stable.  ``pseudos`` may be omitted at construction, in which
    case it materializes as a zero-width tensor.

    In representation-theoretic terms, this state carries up to three O(D)
    irreducible-representation sectors in Cartesian basis: ``scalars`` is the
    even-parity trivial irrep (``0e``; true scalars), ``vectors`` is the
    odd-parity defining irrep (``1o``; polar vectors), and ``pseudos`` is the
    odd-parity trivial irrep (``0o``; rotation invariant, sign-flipping under
    reflection) -- supported in two spatial dimensions only (see the module
    docstring for the measured failures that forced this first type-system
    extension).  Axial vectors (``1e``; the 3D analogue of the pseudoscalar)
    and higher orders remain deliberately not representable: any new physical
    type must arrive as a new typed sector with its own transformation law,
    never packed into an existing tensor.  This per-sector packing is the
    "Irreps" abstraction of libraries such as e3nn, minus the dependency; a
    future migration is a container swap.
    """

    scalars: Float[torch.Tensor, "n scalar_channels"]
    vectors: Float[torch.Tensor, "n vector_channels spatial_dims"]
    pseudos: Float[torch.Tensor, "n pseudo_channels"] | None = None

    def __post_init__(self) -> None:
        """Materialize an omitted pseudo sector as a zero-width tensor."""
        if self.pseudos is None:
            object.__setattr__(
                self,
                "pseudos",
                self.scalars.new_empty(self.scalars.shape[0], 0),
            )

    @property
    def n_entities(self) -> int:
        """Number of entities ``N`` (rows) carried by every sector."""
        return self.scalars.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        """Spatial dimension ``D`` of the vector sector."""
        return self.vectors.shape[-1]

    def validate(self, *, label: str = "state") -> None:
        """Raise ``ValueError`` unless all sectors have consistent shapes,
        entity counts, devices, and dtypes.  ``label`` names the offending
        state in error messages."""
        if self.scalars.ndim != 2:
            raise ValueError(
                f"{label}.scalars must have shape (N, C), got "
                f"{tuple(self.scalars.shape)}"
            )
        if self.vectors.ndim != 3:
            raise ValueError(
                f"{label}.vectors must have shape (N, C, D), got "
                f"{tuple(self.vectors.shape)}"
            )
        if self.pseudos.ndim != 2:
            raise ValueError(
                f"{label}.pseudos must have shape (N, C), got "
                f"{tuple(self.pseudos.shape)}"
            )
        if self.scalars.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"{label} scalar/vector entity counts differ: "
                f"{self.scalars.shape[0]} != {self.vectors.shape[0]}"
            )
        if self.scalars.shape[0] != self.pseudos.shape[0]:
            raise ValueError(
                f"{label} scalar/pseudoscalar entity counts differ: "
                f"{self.scalars.shape[0]} != {self.pseudos.shape[0]}"
            )
        if self.scalars.device != self.vectors.device:
            raise ValueError(f"{label} scalar/vector devices differ")
        if self.scalars.dtype != self.vectors.dtype:
            raise ValueError(f"{label} scalar/vector dtypes differ")
        if self.scalars.device != self.pseudos.device:
            raise ValueError(f"{label} scalar/pseudoscalar devices differ")
        if self.scalars.dtype != self.pseudos.dtype:
            raise ValueError(f"{label} scalar/pseudoscalar dtypes differ")

    @classmethod
    def zeros(
        cls,
        n_entities: int,
        scalar_channels: int,
        vector_channels: int,
        n_spatial_dims: int,
        *,
        pseudo_channels: int = 0,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "ScalarVectorState":
        """Construct an all-zeros state with the given sector widths."""
        return cls(
            scalars=torch.zeros(
                n_entities, scalar_channels, device=device, dtype=dtype
            ),
            vectors=torch.zeros(
                n_entities,
                vector_channels,
                n_spatial_dims,
                device=device,
                dtype=dtype,
            ),
            pseudos=torch.zeros(
                n_entities, pseudo_channels, device=device, dtype=dtype
            ),
        )

    def cat(self, other: "ScalarVectorState") -> "ScalarVectorState":
        """Concatenate two states channel-wise (same entities, more channels
        per sector); entity counts and spatial dims must match."""
        if self.n_entities != other.n_entities:
            raise ValueError("Cannot concatenate states with different entity counts")
        if self.n_spatial_dims != other.n_spatial_dims:
            raise ValueError("Cannot concatenate states with different spatial dims")
        return ScalarVectorState(
            torch.cat((self.scalars, other.scalars), dim=-1),
            torch.cat((self.vectors, other.vectors), dim=1),
            torch.cat((self.pseudos, other.pseudos), dim=-1),
        )

    def slice(
        self, item: slice | Int[torch.Tensor, " m"] | Bool[torch.Tensor, " n"]
    ) -> "ScalarVectorState":
        """Index all sectors along the entity axis with the same ``item``
        (a slice, integer index tensor, or boolean mask)."""
        return ScalarVectorState(
            self.scalars[item], self.vectors[item], self.pseudos[item]
        )


@dataclass(frozen=True)
class TypedQK:
    r"""Projected per-head query or key features, one sector per parity.

    ``scalars`` carries the rotation-invariant (``0e``) rank channels and
    ``vectors`` the polar-vector (``1o``) rank channels; the typed attention
    score contracts each sector only against its like-typed partner, which is
    what keeps the score itself invariant.
    """

    scalars: Float[torch.Tensor, "n heads scalar_rank"]
    vectors: Float[torch.Tensor, "n heads vector_rank spatial_dims"]


@dataclass(frozen=True)
class TypedValues:
    r"""Projected per-head value features.

    ``scalars`` packs the rotation-invariant value features as
    ``(N, H, F_s + F_p)``: true-scalar (``0e``) features first, then
    pseudoscalar (``0o``) features.  Both are invariant under rotation and
    multiply the invariant pair score identically, so the moment machinery
    treats them as one block; they are split again -- with separate output
    maps that never mix the two parities -- at read-out.
    """

    scalars: Float[torch.Tensor, "n heads value_scalars"]
    vectors: Float[torch.Tensor, "n heads value_vectors spatial_dims"]


@dataclass(frozen=True)
class AttentionMoments:
    r"""Quadrature-integrated source moments for typed attention.

    The scalar-value axis has width ``F_s + F_p``: true-scalar value features
    followed by pseudoscalar value features (see :class:`TypedValues`).
    """

    scalar_key_scalar_value: Float[torch.Tensor, "heads scalar_rank value_scalars"]
    vector_key_scalar_value: Float[
        torch.Tensor, "heads vector_rank spatial_dims value_scalars"
    ]
    scalar_key_vector_value: Float[
        torch.Tensor, "heads scalar_rank value_vectors spatial_dims"
    ]
    vector_key_vector_value: Float[
        torch.Tensor, "heads vector_rank spatial_dims value_vectors spatial_dims"
    ]


def _mix_channels(
    weight: Float[torch.Tensor, "channels_out channels_in"],
    tensor: Float[torch.Tensor, "n channels_in *trailing"],
) -> Float[torch.Tensor, "n channels_out *trailing"]:
    """``einsum("oc,nc...->no...", weight, tensor)`` as one GEMM.

    The einsum form lowers to a batch-``n`` bmm of tiny ``(O,C)@(C,·)``
    items, which cuBLAS decomposes into one gemv launch per entity --
    measured at ~10^5 launches per training step at mesh scale (the
    dominant kernel count in every arm; 2026-07-11 decode profile).
    Flattening the trailing axes into the GEMM M-dimension performs the
    SAME contraction over ``c`` (algebraically identical; the per-output
    accumulation runs inside one cuBLAS GEMM instead of one gemv per
    entity) in a single launch.  The transpose copy this costs is one
    bandwidth pass -- negligible against the launch storm it removes.
    """
    n = tensor.shape[0]
    trailing = tensor.shape[2:]
    if tensor.numel() == 0 or weight.numel() == 0:
        # Degenerate channel sets: the reshape below cannot infer a
        # zero-element trailing block; the einsum reference is free at
        # this size and keeps the exact output semantics.
        return torch.einsum("oc,nc...->no...", weight, tensor)
    flat = tensor.reshape(n, tensor.shape[1], -1)  # (N, C, T)
    columns = flat.transpose(1, 2).reshape(-1, flat.shape[1])  # (N*T, C)
    mixed = columns @ weight.transpose(0, 1)  # (N*T, O)
    return (
        mixed.reshape(n, -1, weight.shape[0])
        .transpose(1, 2)
        .reshape(n, weight.shape[0], *trailing)
    )


def _gram_invariants(
    vectors: Float[torch.Tensor, "n channels spatial_dims"],
) -> Float[torch.Tensor, "n gram_channels"]:
    """Return the upper triangle of each per-entity vector Gram matrix.

    This is the ``1o x 1o -> 0e`` Clebsch-Gordan contraction (up to a fixed
    normalization); the upper triangle is complete because the antisymmetric
    combination is parity-odd -- in 2D it is the wedge, which feeds the
    pseudoscalar sector via :func:`_wedge_invariants` when that sector is
    enabled, and in 3D it is the axial sector this state does not carry.
    """
    n, channels, _ = vectors.shape
    if channels == 0:
        return vectors.new_empty(n, 0)
    gram = torch.einsum("ncd,ned->nce", vectors, vectors)
    rows, cols = torch.triu_indices(channels, channels, device=vectors.device)
    return gram[:, rows, cols]


def _pseudo_pair_invariants(
    pseudos: Float[torch.Tensor, "n channels"],
) -> Float[torch.Tensor, "n pair_channels"]:
    """Return the upper triangle (with diagonal) of per-entity ``p_i p_j``.

    This is the ``0o x 0o -> 0e`` product: each entry is even under
    reflection, so the result may feed any true-scalar (invariant) path.
    """
    n, channels = pseudos.shape
    if channels == 0:
        return pseudos.new_empty(n, 0)
    outer = pseudos[:, :, None] * pseudos[:, None, :]
    rows, cols = torch.triu_indices(channels, channels, device=pseudos.device)
    return outer[:, rows, cols]


def _require_planar(
    vectors: Float[torch.Tensor, "n channels spatial_dims"], *, operation: str
) -> None:
    """Reject non-2D inputs to the parity-odd planar products."""
    if vectors.shape[-1] != 2:
        raise ValueError(
            f"{operation} requires 2D (planar) vectors: the pseudoscalar "
            "(0o) sector is two-dimensional by design, and in 3D the "
            "analogous parity-odd object is the axial vector, which is out "
            f"of scope; got {vectors.shape[-1]} spatial dimensions"
        )


def _pair_wedges(
    first: Float[torch.Tensor, "n channels_1 2"],
    second: Float[torch.Tensor, "n channels_2 2"],
) -> Float[torch.Tensor, "n channels_1 channels_2"]:
    r"""All pairwise 2D wedges ``first_i ∧ second_j`` per entity.

    ``first`` is ``(N, C_1, 2)`` and ``second`` is ``(N, C_2, 2)``; the result
    is ``(N, C_1, C_2)`` with entries :math:`a_x b_y - a_y b_x`.  This is the
    ``1o x 1o -> 0o`` product: rotation invariant, sign-flipping under
    reflection (``det`` of the orthogonal map).
    """
    _require_planar(first, operation="_pair_wedges")
    _require_planar(second, operation="_pair_wedges")
    return (
        first[:, :, None, 0] * second[:, None, :, 1]
        - first[:, :, None, 1] * second[:, None, :, 0]
    )


def _wedge_invariants(
    vectors: Float[torch.Tensor, "n channels 2"],
) -> Float[torch.Tensor, "n wedge_channels"]:
    r"""Strict-upper-triangle wedges ``v_i ∧ v_j`` (``i < j``) per entity.

    The pseudoscalar (``0o``) companion of :func:`_gram_invariants`: the
    antisymmetric half of the ``1o x 1o`` product, complete in 2D with the
    strict upper triangle because the wedge is antisymmetric and its diagonal
    vanishes.
    """
    _require_planar(vectors, operation="_wedge_invariants")
    n, channels, _ = vectors.shape
    if channels < 2:
        return vectors.new_empty(n, 0)
    wedges = _pair_wedges(vectors, vectors)
    rows, cols = torch.triu_indices(channels, channels, offset=1, device=vectors.device)
    return wedges[:, rows, cols]


def _vector_perp(
    vectors: Float[torch.Tensor, "n channels 2"],
) -> Float[torch.Tensor, "n channels 2"]:
    r"""Rotate each 2D vector by ``+90``: ``v = (v_x, v_y) -> (-v_y, v_x)``.

    ``v^\perp`` transforms as an *axial* vector (``R v^\perp`` times
    ``det R``), so it must always be paired with exactly one pseudoscalar
    coefficient -- the ``0o x 1o -> 1o`` rotation product -- before it may
    join a polar-vector channel.
    """
    _require_planar(vectors, operation="_vector_perp")
    return torch.stack((-vectors[..., 1], vectors[..., 0]), dim=-1)


class TypedProjection(nn.Module):
    r"""Project scalar/pseudoscalar/vector state without mixing parities.

    The vector path (channel mixing with one shared weight per Cartesian
    component, no bias) is exactly the equivariant linear map Schur's lemma
    permits on an isotypic component; the scalar path additionally lifts the
    quadratic invariants (``1o x 1o -> 0e`` Gram products and, with pseudo
    inputs, ``0o x 0o -> 0e`` pair products), making the whole map linear
    plus one quadratic invariant lift.  Pseudoscalar outputs are a bias-free
    linear map of the pseudo inputs plus the quadratic ``1o x 1o -> 0o``
    wedge invariants of the input vectors (2D only); a bias is forbidden on
    that path because a constant does not flip under reflection.

    ``include_vector_invariants`` gates *every* quadratic lift -- vector
    Grams, pseudo pair products, and vector wedges -- so field-linear value
    paths (superposition contracts) that set it ``False`` remain exactly
    linear in all three sectors.
    """

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        *,
        scalar_bias: bool,
        include_vector_invariants: bool = True,
        pseudo_in: int = 0,
        pseudo_out: int = 0,
    ) -> None:
        """Build the per-sector maps for the given channel widths.

        ``scalar_bias`` gates the bias on the scalar path only (vector and
        pseudo paths are structurally bias-free); ``include_vector_invariants``
        gates every quadratic lift as described in the class docstring.
        """
        super().__init__()
        if pseudo_in < 0 or pseudo_out < 0:
            raise ValueError("pseudo channel counts must be non-negative")
        self.scalar_in = scalar_in
        self.vector_in = vector_in
        self.scalar_out = scalar_out
        self.vector_out = vector_out
        self.pseudo_in = pseudo_in
        self.pseudo_out = pseudo_out
        self.include_vector_invariants = include_vector_invariants
        n_invariants = (
            vector_in * (vector_in + 1) // 2 if include_vector_invariants else 0
        )
        n_pseudo_invariants = (
            pseudo_in * (pseudo_in + 1) // 2 if include_vector_invariants else 0
        )
        self.scalar = (
            nn.Linear(
                scalar_in + n_invariants + n_pseudo_invariants,
                scalar_out,
                bias=scalar_bias,
            )
            if scalar_out
            else None
        )
        if vector_out and not vector_in:
            raise ValueError(
                "TypedProjection cannot create a vector without an input vector basis"
            )
        if vector_out:
            self.vector_weight = nn.Parameter(
                torch.randn(vector_out, vector_in) / math.sqrt(max(vector_in, 1))
            )
        else:
            self.register_parameter("vector_weight", None)
        n_wedges = vector_in * (vector_in - 1) // 2 if include_vector_invariants else 0
        self._n_pseudo_features = pseudo_in + n_wedges
        if pseudo_out and not self._n_pseudo_features:
            raise ValueError(
                "TypedProjection cannot create a pseudoscalar without a "
                "pseudo or vector-pair (wedge) input basis"
            )
        if pseudo_out:
            self.pseudo_weight = nn.Parameter(
                torch.randn(pseudo_out, self._n_pseudo_features)
                / math.sqrt(self._n_pseudo_features)
            )
        else:
            self.register_parameter("pseudo_weight", None)

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        """Project ``state`` to the configured output widths, sector by
        sector, without mixing parities (see the class docstring for the
        exact maps applied to each sector)."""
        if state.pseudos.shape[1] != self.pseudo_in:
            raise ValueError(
                f"state has {state.pseudos.shape[1]} pseudoscalar channels; "
                f"expected {self.pseudo_in}"
            )
        if self.scalar is not None:
            scalar_input = state.scalars
            if self.include_vector_invariants:
                scalar_parts = [scalar_input, _gram_invariants(state.vectors)]
                if self.pseudo_in:
                    scalar_parts.append(_pseudo_pair_invariants(state.pseudos))
                scalar_input = torch.cat(scalar_parts, dim=-1)
            scalars = self.scalar(scalar_input)
        else:
            scalars = None
        if self.vector_out:
            vectors = _mix_channels(self.vector_weight, state.vectors)
        else:
            vectors = state.vectors.new_empty(state.n_entities, 0, state.n_spatial_dims)
        if self.pseudo_out:
            pseudo_parts = []
            if self.pseudo_in:
                pseudo_parts.append(state.pseudos)
            if self.include_vector_invariants and self.vector_in >= 2:
                pseudo_parts.append(_wedge_invariants(state.vectors))
            pseudo_features = (
                pseudo_parts[0]
                if len(pseudo_parts) == 1
                else torch.cat(pseudo_parts, dim=-1)
            )
            # Plain matrix product; einsum routed this through the batched
            # path on some backends -- keep it an explicit single GEMM.
            pseudos = pseudo_features @ self.pseudo_weight.transpose(0, 1)
        else:
            pseudos = state.scalars.new_empty(state.n_entities, 0)
        if scalars is None:
            scalars = vectors.new_empty(state.n_entities, 0)
        else:
            vectors = vectors.to(dtype=scalars.dtype)
        return ScalarVectorState(scalars, vectors, pseudos.to(dtype=scalars.dtype))


class MeshAttention(nn.Module):
    r"""Exact global signed moment attention for scalar and polar-vector fields.

    Queries and keys may contain rank-0 and rank-1 channels.  The invariant
    pair coefficient is

    .. math::

        a_{ijh}=q^0_{ih}\cdot k^0_{jh}
        +\sum_r q^1_{ihr}\cdot k^1_{jhr}.

    Values retain their scalar/vector type.  Associativity evaluates the dense
    quadrature sum without constructing an ``N_target x N_source`` matrix.
    ``entity_chunk_size`` bounds live projection workspace in inference.  With
    autograd enabled, PyTorch retains each chunk's saved activations for the
    backward pass, so total saved activation memory remains linear in entity
    count rather than being bounded by one chunk.

    Pseudoscalar (``0o``) channels (2D only; all pseudo widths default to
    zero, which is bitwise identical to the pre-extension layer): query/key
    pseudo channels enter the pair coefficient only through their invariant
    ``0o x 0o -> 0e`` pair products inside the scalar-rank projection, so the
    coefficient stays a true scalar.  Pseudo *value* channels are rotation
    invariant exactly like scalar values, so they ride the scalar-value
    moment machinery (packed after the scalar features; see
    :class:`TypedValues`) and are split back out at read-out through a
    dedicated bias-free output map -- a bias, or any linear mixing with true
    scalars, would break the reflection sign flip.
    """

    def __init__(
        self,
        *,
        query_scalar_dim: int,
        query_vector_dim: int,
        key_scalar_dim: int,
        key_vector_dim: int,
        value_scalar_dim: int,
        value_vector_dim: int,
        out_scalar_dim: int,
        out_vector_dim: int,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        scalar_value_dim: int = 8,
        vector_value_dim: int = 4,
        qk_scalar_bias: bool = True,
        value_scalar_bias: bool = False,
        value_include_vector_invariants: bool = True,
        output_scalar_bias: bool = False,
        accumulation_dtype: torch.dtype | None = torch.float32,
        entity_chunk_size: int | None = 65536,
        query_pseudo_dim: int = 0,
        key_pseudo_dim: int = 0,
        value_pseudo_dim: int = 0,
        out_pseudo_dim: int = 0,
        pseudo_value_dim: int = 0,
    ) -> None:
        """Configure head count, per-sector query/key/value/output widths,
        attention ranks, and the memory knobs (``accumulation_dtype``,
        ``entity_chunk_size``); see the class docstring for semantics."""
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if scalar_rank < 0 or vector_rank < 0:
            raise ValueError("attention ranks must be non-negative")
        if scalar_rank + vector_rank == 0:
            raise ValueError("at least one scalar or vector key rank is required")
        if scalar_value_dim < 0 or vector_value_dim < 0:
            raise ValueError("value dimensions must be non-negative")
        if (
            min(
                query_pseudo_dim,
                key_pseudo_dim,
                value_pseudo_dim,
                out_pseudo_dim,
                pseudo_value_dim,
            )
            < 0
        ):
            raise ValueError("pseudoscalar dimensions must be non-negative")
        if entity_chunk_size is not None and (
            isinstance(entity_chunk_size, bool)
            or not isinstance(entity_chunk_size, int)
            or entity_chunk_size < 1
        ):
            raise ValueError("entity_chunk_size must be a positive integer or None")

        self.heads = heads
        self.query_scalar_dim = query_scalar_dim
        self.query_vector_dim = query_vector_dim
        self.key_scalar_dim = key_scalar_dim
        self.key_vector_dim = key_vector_dim
        self.value_scalar_dim = value_scalar_dim
        self.value_vector_dim = value_vector_dim
        self.scalar_rank = scalar_rank
        self.vector_rank = vector_rank
        self.scalar_value_dim = scalar_value_dim
        self.vector_value_dim = vector_value_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        self.query_pseudo_dim = query_pseudo_dim
        self.key_pseudo_dim = key_pseudo_dim
        self.value_pseudo_dim = value_pseudo_dim
        self.out_pseudo_dim = out_pseudo_dim
        self.pseudo_value_dim = pseudo_value_dim
        self.accumulation_dtype = accumulation_dtype
        self.entity_chunk_size = entity_chunk_size

        self.query_projection = TypedProjection(
            query_scalar_dim,
            query_vector_dim,
            heads * scalar_rank,
            heads * vector_rank,
            scalar_bias=qk_scalar_bias,
            pseudo_in=query_pseudo_dim,
        )
        self.key_projection = TypedProjection(
            key_scalar_dim,
            key_vector_dim,
            heads * scalar_rank,
            heads * vector_rank,
            scalar_bias=qk_scalar_bias,
            pseudo_in=key_pseudo_dim,
        )
        self.value_projection = TypedProjection(
            value_scalar_dim,
            value_vector_dim,
            heads * scalar_value_dim,
            heads * vector_value_dim,
            scalar_bias=value_scalar_bias,
            include_vector_invariants=value_include_vector_invariants,
            pseudo_in=value_pseudo_dim,
            pseudo_out=heads * pseudo_value_dim,
        )

        if out_scalar_dim and not scalar_value_dim:
            raise ValueError("Scalar output requires at least one scalar value channel")
        self.scalar_output = (
            nn.Linear(
                heads * scalar_value_dim,
                out_scalar_dim,
                bias=output_scalar_bias,
            )
            if out_scalar_dim
            else None
        )
        if out_vector_dim and not vector_value_dim:
            raise ValueError("Vector output requires at least one vector value channel")
        if out_vector_dim:
            self.vector_output_weight = nn.Parameter(
                torch.randn(out_vector_dim, heads, vector_value_dim)
                / math.sqrt(max(heads * vector_value_dim, 1))
            )
        else:
            self.register_parameter("vector_output_weight", None)
        if out_pseudo_dim and not pseudo_value_dim:
            raise ValueError(
                "Pseudoscalar output requires at least one pseudo value channel"
            )
        # Never a bias: a constant offset does not flip under reflection.
        self.pseudo_output = (
            nn.Linear(heads * pseudo_value_dim, out_pseudo_dim, bias=False)
            if out_pseudo_dim
            else None
        )

    def _accumulation_type(self, *tensors: torch.Tensor) -> torch.dtype:
        """Promote inputs with a precision floor, never downcast FP64."""
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        if self.accumulation_dtype is not None:
            dtype = torch.promote_types(dtype, self.accumulation_dtype)
        return dtype

    @staticmethod
    def _validate_projection_state(
        state: ScalarVectorState,
        *,
        scalar_dim: int,
        vector_dim: int,
        pseudo_dim: int,
        label: str,
    ) -> None:
        """Validate ``state`` and check each sector's channel width against
        the projection's declared input widths."""
        state.validate(label=label)
        if state.scalars.shape[1] != scalar_dim:
            raise ValueError(
                f"{label}.scalars has {state.scalars.shape[1]} channels; "
                f"expected {scalar_dim}"
            )
        if state.vectors.shape[1] != vector_dim:
            raise ValueError(
                f"{label}.vectors has {state.vectors.shape[1]} channels; "
                f"expected {vector_dim}"
            )
        if state.pseudos.shape[1] != pseudo_dim:
            raise ValueError(
                f"{label}.pseudos has {state.pseudos.shape[1]} channels; "
                f"expected {pseudo_dim}"
            )

    def project_queries(self, state: ScalarVectorState) -> TypedQK:
        """Project a query state to per-head typed rank channels.

        The :math:`1/\\sqrt{R_s + D R_v}` score scale is folded into the
        query side once, so key projections and moments stay unscaled.
        """
        self._validate_projection_state(
            state,
            scalar_dim=self.query_scalar_dim,
            vector_dim=self.query_vector_dim,
            pseudo_dim=self.query_pseudo_dim,
            label="query_state",
        )
        projected = self.query_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        # The vector ranks contain D Cartesian components, so the invariant
        # signed dot product has R_s + D R_v independently varying terms.
        score_scale = 1.0 / math.sqrt(max(self.scalar_rank + d * self.vector_rank, 1))
        return TypedQK(
            projected.scalars.reshape(n, self.heads, self.scalar_rank) * score_scale,
            projected.vectors.reshape(n, self.heads, self.vector_rank, d) * score_scale,
        )

    def project_keys(self, state: ScalarVectorState) -> TypedQK:
        """Project a source state to per-head typed key rank channels
        (unscaled; the score scale lives on the query side)."""
        self._validate_projection_state(
            state,
            scalar_dim=self.key_scalar_dim,
            vector_dim=self.key_vector_dim,
            pseudo_dim=self.key_pseudo_dim,
            label="key_state",
        )
        projected = self.key_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        return TypedQK(
            projected.scalars.reshape(n, self.heads, self.scalar_rank),
            projected.vectors.reshape(n, self.heads, self.vector_rank, d),
        )

    def project_values(self, state: ScalarVectorState) -> TypedValues:
        """Project a source state to per-head typed value features; pseudo
        value channels are packed after the scalar features (they share the
        invariant-value moment machinery; see the class docstring)."""
        self._validate_projection_state(
            state,
            scalar_dim=self.value_scalar_dim,
            vector_dim=self.value_vector_dim,
            pseudo_dim=self.value_pseudo_dim,
            label="value_state",
        )
        projected = self.value_projection(state)
        n, d = state.n_entities, state.n_spatial_dims
        scalars = projected.scalars.reshape(n, self.heads, self.scalar_value_dim)
        if self.pseudo_value_dim:
            # Pseudo value features are rotation invariant like scalars, so
            # they share the invariant-value moment machinery; the parity
            # split is restored at read-out (see the class docstring).
            scalars = torch.cat(
                (
                    scalars,
                    projected.pseudos.reshape(n, self.heads, self.pseudo_value_dim),
                ),
                dim=-1,
            )
        return TypedValues(
            scalars,
            projected.vectors.reshape(n, self.heads, self.vector_value_dim, d),
        )

    def build_moments(
        self,
        source_mesh: Mesh,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
        segments: Sequence[slice] | None = None,
        segment_log_gain: Float[torch.Tensor, "n_segments heads"] | None = None,
        segment_measure_balance: bool = False,
    ) -> AttentionMoments:
        r"""Project and quadrature-integrate keys and values once.

        With the default ``segments=None`` the moments are the plain
        quadrature sum over every source cell -- byte-identical to the
        historical operator.  Supplying ``segments`` (a contiguous,
        non-overlapping partition of the source cells, e.g. one slice per
        boundary component) together with ``segment_log_gain`` (shape
        ``(len(segments), heads)``) computes each segment's moments
        separately and combines them as

        .. math:: M = \sum_s e^{g_s}\, M_s
                    = \sum_s e^{g_s + \ln A_s}\, \bar M_s,

        where :math:`\bar M_s = M_s / A_s` is the segment's
        measure-averaged moment and :math:`A_s` its total measure: the
        learned, dimensionless per-segment/per-head gain shifts each
        segment's log-measure weight additively, so a measure imbalance
        between segments (e.g. tunnel panels vs vehicle cells) is a
        log-scale parameter away rather than a feature-magnitude fight.
        At ``segment_log_gain = 0`` the combination reproduces the plain
        sum exactly (up to floating-point summation order).  The gain is a
        pure number per (segment, head): similarity covariance, parity
        typing, and drive-linearity of the moments are unchanged.

        ``segment_measure_balance=True`` (external-review balanced arm)
        shifts each segment's effective log-gain by
        :math:`\ln\bar A - \ln A_s` (:math:`\bar A` the mean segment
        measure of the sample), so at zero gains every segment contributes
        its measure-AVERAGED moment scaled by the common :math:`\bar A` --
        equal weight per boundary instead of raw measure dominance.  The
        offset is a per-sample dimensionless measure ratio (similarity
        covariance unchanged), and the learned gains can recover the plain
        sum (:math:`g_s = \ln A_s - \ln\bar A`), so the balanced arm is a
        reparameterized initialization, not a smaller hypothesis class.
        Default ``False`` is bitwise the historical pool.
        """
        if source_mesh.n_cells != key_state.n_entities:
            raise ValueError("source Mesh cell count must match key state entity count")
        if key_state.n_entities != value_state.n_entities:
            raise ValueError("key and value entity counts must match")
        if source_mesh.n_spatial_dims != key_state.n_spatial_dims:
            raise ValueError("source Mesh and key state spatial dims differ")
        if source_mesh.n_spatial_dims != value_state.n_spatial_dims:
            raise ValueError("source Mesh and value state spatial dims differ")
        if (segments is None) != (segment_log_gain is None):
            raise ValueError(
                "segments and segment_log_gain must be provided together: "
                "the segmented moment pool is defined by both the source "
                "partition and its per-segment log-gains"
            )
        if segment_measure_balance and segments is None:
            raise ValueError(
                "segment_measure_balance requires segments: the balance is "
                "an offset on the per-segment log-gains"
            )
        if segments is not None:
            return self._build_segmented_moments(
                source_mesh,
                key_state,
                value_state,
                segments,
                segment_log_gain,
                measure_balance=segment_measure_balance,
            )

        # Attention heads are aligned groups, not axes to outer-product with
        # one another. The Mesh owns quadrature measure; the shared weighted
        # primitive owns NaN and accumulation policy.
        def _moments_from_projected(
            keys: TypedQK,
            values: TypedValues,
            weights: Float[torch.Tensor, " n"] | None,
        ) -> AttentionMoments:
            """Integrate one (possibly chunk-sliced) projected source set."""
            # Cartesian components are independently varying finite-rank
            # features in the signed kernel. Flatten them next to the scalar
            # features so all four typed key/value moments share one weighted
            # matrix multiplication. Slicing the joint moment back into typed
            # blocks preserves the public representation and evaluation math.
            key_features = torch.cat(
                (keys.scalars, keys.vectors.flatten(start_dim=2)), dim=-1
            )
            value_features = torch.cat(
                (values.scalars, values.vectors.flatten(start_dim=2)), dim=-1
            )
            if weights is None:
                joint_moment = integrate_moment(
                    source_mesh,
                    key_features,
                    value_features,
                    aligned_dims=1,
                    accumulation_dtype=self.accumulation_dtype,
                    nan_policy="propagate",
                )
            else:
                joint_moment = _integrate_weighted_moment(
                    key_features,
                    value_features,
                    weights,
                    aligned_dims=1,
                    accumulation_dtype=self.accumulation_dtype,
                    nan_policy="propagate",
                )

            scalar_rank = self.scalar_rank
            # Invariant value features: scalar values then pseudo values.
            scalar_value_dim = self.scalar_value_dim + self.pseudo_value_dim
            spatial_dim = keys.vectors.shape[-1]
            return AttentionMoments(
                scalar_key_scalar_value=joint_moment[
                    :, :scalar_rank, :scalar_value_dim
                ],
                vector_key_scalar_value=joint_moment[
                    :, scalar_rank:, :scalar_value_dim
                ].reshape(
                    self.heads,
                    self.vector_rank,
                    spatial_dim,
                    scalar_value_dim,
                ),
                scalar_key_vector_value=joint_moment[
                    :, :scalar_rank, scalar_value_dim:
                ].reshape(
                    self.heads,
                    self.scalar_rank,
                    self.vector_value_dim,
                    spatial_dim,
                ),
                vector_key_vector_value=joint_moment[
                    :, scalar_rank:, scalar_value_dim:
                ].reshape(
                    self.heads,
                    self.vector_rank,
                    spatial_dim,
                    self.vector_value_dim,
                    spatial_dim,
                ),
            )

        chunk_size = self.entity_chunk_size
        if chunk_size is None or key_state.n_entities <= chunk_size:
            return _moments_from_projected(
                self.project_keys(key_state),
                self.project_values(value_state),
                None,
            )

        accumulated: AttentionMoments | None = None
        weights = source_mesh.cell_areas
        for start in range(0, key_state.n_entities, chunk_size):
            item = slice(start, min(start + chunk_size, key_state.n_entities))
            chunk_moments = _moments_from_projected(
                self.project_keys(key_state.slice(item)),
                self.project_values(value_state.slice(item)),
                weights[item],
            )
            if accumulated is None:
                accumulated = chunk_moments
            else:
                accumulated = AttentionMoments(
                    accumulated.scalar_key_scalar_value
                    + chunk_moments.scalar_key_scalar_value,
                    accumulated.vector_key_scalar_value
                    + chunk_moments.vector_key_scalar_value,
                    accumulated.scalar_key_vector_value
                    + chunk_moments.scalar_key_vector_value,
                    accumulated.vector_key_vector_value
                    + chunk_moments.vector_key_vector_value,
                )
        if accumulated is None:
            raise RuntimeError("Cannot build attention moments from an empty source")
        return accumulated

    def _build_segmented_moments(
        self,
        source_mesh: Mesh,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
        segments: Sequence[slice],
        segment_log_gain: Float[torch.Tensor, "n_segments heads"],
        measure_balance: bool = False,
    ) -> AttentionMoments:
        r"""Combine per-segment moments with dimensionless per-head gains.

        See :meth:`build_moments` for the operator definition.  Segments
        must be non-empty, unit-stride, and contiguously partition
        ``[0, n_entities)``; the moments are linear over source cells, so
        the per-segment computation reuses the plain path on cell slices
        and the gained sum reproduces the plain moments exactly when every
        gain is zero (up to summation order).
        """
        if len(segments) == 0:
            raise ValueError("segments must contain at least one slice")
        if not isinstance(segment_log_gain, torch.Tensor) or tuple(
            segment_log_gain.shape
        ) != (len(segments), self.heads):
            actual = (
                tuple(segment_log_gain.shape)
                if isinstance(segment_log_gain, torch.Tensor)
                else type(segment_log_gain).__name__
            )
            raise ValueError(
                f"segment_log_gain must be a tensor of shape "
                f"(len(segments), heads) = ({len(segments)}, {self.heads}), "
                f"got {actual}"
            )
        expected_start = 0
        for segment in segments:
            if not isinstance(segment, slice) or segment.step not in (None, 1):
                raise ValueError(
                    f"segments must be unit-stride slices, got {segment!r}"
                )
            if segment.start != expected_start or segment.stop <= segment.start:
                raise ValueError(
                    "segments must be non-empty and contiguously partition "
                    f"the source cells starting at 0; segment {segment!r} "
                    f"does not begin at {expected_start}"
                )
            expected_start = segment.stop
        if expected_start != key_state.n_entities:
            raise ValueError(
                f"segments end at {expected_start} but the source has "
                f"{key_state.n_entities} cells; segments must cover every "
                "source cell exactly once"
            )

        if measure_balance:
            # Balanced pool: offset each segment's log-gain by
            # ln(mean measure) - ln(segment measure), computed per sample
            # from the Mesh quadrature measure (a dimensionless ratio, so
            # similarity covariance is untouched; see build_moments).
            areas = source_mesh.cell_areas
            segment_measure = torch.stack(
                [areas[segment].sum() for segment in segments]
            )
            log_offset = segment_measure.mean().log() - segment_measure.log()
            segment_log_gain = segment_log_gain + log_offset.to(
                segment_log_gain.dtype
            )[:, None]
        combined: list[torch.Tensor] | None = None
        for index, segment in enumerate(segments):
            part = self.build_moments(
                source_mesh.slice_cells(segment),
                key_state.slice(segment),
                value_state.slice(segment),
            )
            gain = torch.exp(segment_log_gain[index]).to(
                part.scalar_key_scalar_value.dtype
            )
            gained = [
                part.scalar_key_scalar_value * gain[:, None, None],
                part.vector_key_scalar_value * gain[:, None, None, None],
                part.scalar_key_vector_value * gain[:, None, None, None],
                part.vector_key_vector_value * gain[:, None, None, None, None],
            ]
            combined = (
                gained
                if combined is None
                else [total + term for total, term in zip(combined, gained)]
            )
        return AttentionMoments(*combined)

    def evaluate_moments(
        self,
        query_state: ScalarVectorState,
        moments: AttentionMoments,
    ) -> ScalarVectorState:
        r"""Evaluate cached source moments independently at each receiver."""
        d = query_state.n_spatial_dims
        invariant_value_dim = self.scalar_value_dim + self.pseudo_value_dim
        expected_shapes = (
            (self.heads, self.scalar_rank, invariant_value_dim),
            (self.heads, self.vector_rank, d, invariant_value_dim),
            (self.heads, self.scalar_rank, self.vector_value_dim, d),
            (self.heads, self.vector_rank, d, self.vector_value_dim, d),
        )
        actual_shapes = (
            tuple(moments.scalar_key_scalar_value.shape),
            tuple(moments.vector_key_scalar_value.shape),
            tuple(moments.scalar_key_vector_value.shape),
            tuple(moments.vector_key_vector_value.shape),
        )
        if actual_shapes != expected_shapes:
            raise ValueError(
                "AttentionMoments are incompatible with this layer/query; "
                f"expected {expected_shapes}, got {actual_shapes}"
            )
        moment_tensors = (
            moments.scalar_key_scalar_value,
            moments.vector_key_scalar_value,
            moments.scalar_key_vector_value,
            moments.vector_key_vector_value,
        )
        if any(
            tensor.device != query_state.scalars.device for tensor in moment_tensors
        ):
            raise ValueError("AttentionMoments and query_state must share a device")
        chunk_size = self.entity_chunk_size
        if chunk_size is not None and query_state.n_entities > chunk_size:
            outputs = [
                self.evaluate_moments(
                    query_state.slice(
                        slice(
                            start,
                            min(start + chunk_size, query_state.n_entities),
                        )
                    ),
                    moments,
                )
                for start in range(0, query_state.n_entities, chunk_size)
            ]
            return ScalarVectorState(
                torch.cat([output.scalars for output in outputs], dim=0),
                torch.cat([output.vectors for output in outputs], dim=0),
                torch.cat([output.pseudos for output in outputs], dim=0),
            )
        queries = self.project_queries(query_state)
        output_dtype = query_state.scalars.dtype
        dtype = self._accumulation_type(
            queries.scalars,
            queries.vectors,
            moments.scalar_key_scalar_value,
            moments.vector_key_scalar_value,
            moments.scalar_key_vector_value,
            moments.vector_key_vector_value,
        )
        qs = queries.scalars.to(dtype)
        qv = queries.vectors.to(dtype)
        scalar_key_scalar_value = moments.scalar_key_scalar_value.to(dtype)
        vector_key_scalar_value = moments.vector_key_scalar_value.to(dtype)
        scalar_key_vector_value = moments.scalar_key_vector_value.to(dtype)
        vector_key_vector_value = moments.vector_key_vector_value.to(dtype)

        with torch.autocast(device_type=qs.device.type, enabled=False):
            scalar_heads = torch.einsum(
                "nhr,hrf->nhf", qs, scalar_key_scalar_value
            ) + torch.einsum("nhrd,hrdf->nhf", qv, vector_key_scalar_value)
            vector_heads = torch.einsum(
                "nhr,hrfd->nhfd", qs, scalar_key_vector_value
            ) + torch.einsum("nhrd,hrdfe->nhfe", qv, vector_key_vector_value)

        return self._typed_read_out(
            scalar_heads, vector_heads, query_state, output_dtype
        )

    def _typed_read_out(
        self,
        scalar_heads: Float[torch.Tensor, "n heads value_scalars"],
        vector_heads: Float[torch.Tensor, "n heads value_vectors spatial_dims"],
        query_state: ScalarVectorState,
        output_dtype: torch.dtype,
    ) -> ScalarVectorState:
        r"""Split the invariant heads by parity and apply the typed outputs.

        ``scalar_heads`` carries the joint invariant value features
        ``(N, H, F_s + F_p)``; scalar and pseudoscalar features receive
        separate output maps because a shared linear map would mix parities.
        """
        n = query_state.n_entities
        if self.pseudo_value_dim:
            pseudo_heads = scalar_heads[..., self.scalar_value_dim :]
            scalar_heads = scalar_heads[..., : self.scalar_value_dim]
        else:
            pseudo_heads = None
        scalars = (
            self.scalar_output(
                scalar_heads.to(output_dtype).reshape(
                    n,
                    self.heads * self.scalar_value_dim,
                )
            )
            if self.scalar_output is not None
            else query_state.scalars.new_empty(n, 0)
        )
        if self.out_vector_dim:
            vectors = torch.einsum(
                "ohf,nhfd->nod",
                self.vector_output_weight,
                vector_heads.to(output_dtype),
            )
        else:
            vectors = query_state.vectors.new_empty(n, 0, query_state.n_spatial_dims)
        if self.pseudo_output is not None:
            pseudos = self.pseudo_output(
                pseudo_heads.to(output_dtype).reshape(
                    n,
                    self.heads * self.pseudo_value_dim,
                )
            )
        else:
            pseudos = scalars.new_empty(n, 0)
        return ScalarVectorState(
            scalars,
            vectors.to(dtype=scalars.dtype),
            pseudos.to(dtype=scalars.dtype),
        )

    def forward(
        self,
        source_mesh: Mesh,
        query_state: ScalarVectorState,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
        segments: Sequence[slice] | None = None,
        segment_log_gain: Float[torch.Tensor, "n_segments heads"] | None = None,
        segment_measure_balance: bool = False,
    ) -> ScalarVectorState:
        """Full attention pass: :meth:`build_moments` over the source, then
        :meth:`evaluate_moments` at every query entity.  ``segments`` /
        ``segment_log_gain`` enable the per-segment moment pool (see
        :meth:`build_moments`)."""
        return self.evaluate_moments(
            query_state,
            self.build_moments(
                source_mesh,
                key_state,
                value_state,
                segments=segments,
                segment_log_gain=segment_log_gain,
                segment_measure_balance=segment_measure_balance,
            ),
        )

    def forward_reference(
        self,
        source_mesh: Mesh,
        query_state: ScalarVectorState,
        key_state: ScalarVectorState,
        value_state: ScalarVectorState,
    ) -> ScalarVectorState:
        r"""Dense all-pairs oracle for values and gradient tests."""
        if source_mesh.n_cells != key_state.n_entities:
            raise ValueError("source Mesh cell count must match key state entity count")
        if key_state.n_entities != value_state.n_entities:
            raise ValueError("key and value entity counts must match")
        if (
            source_mesh.n_spatial_dims != key_state.n_spatial_dims
            or source_mesh.n_spatial_dims != value_state.n_spatial_dims
            or source_mesh.n_spatial_dims != query_state.n_spatial_dims
        ):
            raise ValueError(
                "source Mesh, query, key, and value spatial dimensions must match"
            )
        q = self.project_queries(query_state)
        k = self.project_keys(key_state)
        v = self.project_values(value_state)
        dtype = self._accumulation_type(
            q.scalars,
            q.vectors,
            k.scalars,
            k.vectors,
            v.scalars,
            v.vectors,
            source_mesh.cell_areas,
        )
        with torch.autocast(device_type=q.scalars.device.type, enabled=False):
            score = torch.einsum(
                "mhr,nhr->mnh", q.scalars.to(dtype), k.scalars.to(dtype)
            ) + torch.einsum("mhrd,nhrd->mnh", q.vectors.to(dtype), k.vectors.to(dtype))
            weighted_score = score * source_mesh.cell_areas.to(dtype)[None, :, None]
            scalar_heads = torch.einsum(
                "mnh,nhf->mhf", weighted_score, v.scalars.to(dtype)
            )
            vector_heads = torch.einsum(
                "mnh,nhfd->mhfd", weighted_score, v.vectors.to(dtype)
            )
        output_dtype = query_state.scalars.dtype
        return self._typed_read_out(
            scalar_heads, vector_heads, query_state, output_dtype
        )


__all__ = [
    "AttentionMoments",
    "MeshAttention",
    "ScalarVectorState",
    "TypedProjection",
]
