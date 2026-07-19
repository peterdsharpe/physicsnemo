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

r"""Equivariant residual blocks for global boundary-to-boundary operators.

The geometry and physical-field streams are intentionally separate.  Geometry
blocks are nonlinear.  Linear field blocks are exactly linear in their field
argument at fixed geometry; nonlinear field blocks are structurally
zero-preserving; the quadratic read-in (:class:`QuadraticFieldReadIn`) is the
third field-mode class, exactly a polynomial of DECLARED degree two in the
field.  Keeping these as different Python classes makes it difficult for a
future normalization, activation, or bias to silently invalidate the
linear-mode (or declared-degree) contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.mesh import Mesh

from .attention import (
    AttentionMoments,
    MeshAttention,
    ScalarVectorState,
    TypedProjection,
    _gram_invariants,
    _mix_channels,
    _pair_wedges,
    _pseudo_pair_invariants,
    _vector_perp,
    _wedge_invariants,
)


def _add(left: ScalarVectorState, right: ScalarVectorState) -> ScalarVectorState:
    """Add two identically typed states with an informative shape failure."""
    if left.scalars.shape != right.scalars.shape:
        raise ValueError(
            "Scalar residual shapes differ: "
            f"{tuple(left.scalars.shape)} != {tuple(right.scalars.shape)}"
        )
    if left.vectors.shape != right.vectors.shape:
        raise ValueError(
            "Vector residual shapes differ: "
            f"{tuple(left.vectors.shape)} != {tuple(right.vectors.shape)}"
        )
    if left.pseudos.shape != right.pseudos.shape:
        raise ValueError(
            "Pseudoscalar residual shapes differ: "
            f"{tuple(left.pseudos.shape)} != {tuple(right.pseudos.shape)}"
        )
    return ScalarVectorState(
        left.scalars + right.scalars,
        left.vectors + right.vectors,
        left.pseudos + right.pseudos,
    )


def _apply_coefficients(
    coefficients: Float[torch.Tensor, "n channels_out basis"],
    basis: Float[torch.Tensor, "n basis spatial_dims"],
) -> Float[torch.Tensor, "n channels_out spatial_dims"]:
    """``einsum("nog,ngd->nod", coefficients, basis)`` without per-item gemv.

    The per-entity coefficient matrices are data-dependent, so this
    contraction cannot fold into one shared GEMM -- but as an einsum/bmm
    it decomposes into one tiny gemv launch per entity (the encoder
    launch storm; 2026-07-11 decode profile).  With ``g`` at geometry-
    basis size (single digits) and ``d`` spatial, a broadcast multiply
    plus a ``g``-axis sum computes the same sums in two big kernels.
    """
    return (coefficients.unsqueeze(-1) * basis[:, None, :, :]).sum(dim=2)


class TypedRMSNorm(nn.Module):
    r"""O(D)-equivariant normalization for a nonlinear geometry state.

    Scalars receive ordinary RMS normalization.  All vector channels at one
    entity share an invariant RMS denominator, so Cartesian components are
    never mixed and rotations/reflections commute with this operation.

    This class is only used on the geometry stream.  It must not be inserted
    in :class:`LinearMeshFieldBlock`, where normalization by field amplitude
    would violate superposition.
    """

    def __init__(self, scalar_dim: int, vector_dim: int, eps: float = 1.0e-8) -> None:
        """Allocate per-channel gain parameters for the given sector widths."""
        super().__init__()
        self.eps = eps
        self.scalar_weight = nn.Parameter(torch.ones(scalar_dim))
        self.vector_weight = nn.Parameter(torch.ones(vector_dim))

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        """RMS-normalize each sector equivariantly (see class docstring);
        rejects states carrying a pseudoscalar sector."""
        if state.pseudos.shape[-1]:
            raise ValueError(
                "TypedRMSNorm is a geometry-stream module and the geometry "
                "state carries no pseudoscalar sector; got "
                f"{state.pseudos.shape[-1]} pseudo channels"
            )
        if state.scalars.shape[-1]:
            scalar_rms = state.scalars.square().mean(dim=-1, keepdim=True)
            scalars = state.scalars * torch.rsqrt(scalar_rms + self.eps)
            scalars = scalars * self.scalar_weight
        else:
            scalars = state.scalars

        if state.vectors.shape[1]:
            vector_rms = state.vectors.square().mean(dim=(1, 2), keepdim=True)
            vectors = state.vectors * torch.rsqrt(vector_rms + self.eps)
            vectors = vectors * self.vector_weight[None, :, None]
        else:
            vectors = state.vectors
        return ScalarVectorState(scalars, vectors.to(dtype=scalars.dtype))


class StateLayerScale(nn.Module):
    """Learned per-channel residual scales that preserve tensor type.

    A multiplicative true-scalar scale is parity-safe on every sector, so
    pseudoscalar channels receive the same treatment as scalars; the pseudo
    scale parameter exists only when ``pseudo_dim`` is positive, keeping the
    default state dict bitwise identical to the pre-extension module.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        init: float = 1.0e-2,
        *,
        pseudo_dim: int = 0,
    ) -> None:
        """Allocate per-channel residual scales initialized to ``init``."""
        super().__init__()
        self.scalar_scale = nn.Parameter(torch.full((scalar_dim,), init))
        self.vector_scale = nn.Parameter(torch.full((vector_dim,), init))
        if pseudo_dim:
            self.pseudo_scale = nn.Parameter(torch.full((pseudo_dim,), init))
        else:
            self.register_parameter("pseudo_scale", None)

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        """Scale every channel of every sector by its learned coefficient."""
        scalars = state.scalars * self.scalar_scale
        # The zero-width passthrough still tracks the scalars' (possibly
        # autocast-promoted) dtype so the state stays internally consistent.
        pseudos = (
            state.pseudos * self.pseudo_scale
            if self.pseudo_scale is not None
            else state.pseudos.to(dtype=scalars.dtype)
        )
        return ScalarVectorState(
            scalars,
            state.vectors * self.vector_scale[None, :, None],
            pseudos,
        )


class GeometryFeedForward(nn.Module):
    """A small nonlinear O(D)-equivariant pointwise geometry network."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        """Build the typed input/output projections and the scalar-to-vector
        gate at ``hidden_ratio`` times the state widths."""
        super().__init__()
        hidden_scalar = max(hidden_ratio * scalar_dim, 1)
        hidden_vector = max(hidden_ratio * vector_dim, vector_dim)
        self.input = TypedProjection(
            scalar_dim,
            vector_dim,
            hidden_scalar,
            hidden_vector,
            scalar_bias=True,
        )
        self.vector_gate = (
            nn.Linear(hidden_scalar, hidden_vector) if hidden_vector else None
        )
        self.output = TypedProjection(
            hidden_scalar,
            hidden_vector,
            scalar_dim,
            vector_dim,
            scalar_bias=True,
        )

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        """Apply the equivariant MLP: typed lift, SiLU on scalars, an
        invariant sigmoid gate on vector channels, typed read-out."""
        hidden = self.input(state)
        scalars = torch.nn.functional.silu(hidden.scalars)
        if hidden.vectors.shape[1]:
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for non-empty vector state")
            gates = torch.sigmoid(self.vector_gate(scalars))
            vectors = hidden.vectors * gates[:, :, None]
        else:
            vectors = hidden.vectors
        return self.output(ScalarVectorState(scalars, vectors))


class GeometryConditionedLinear(nn.Module):
    r"""An O(D)-equivariant map linear in ``field`` at fixed ``geometry``.

    Direct scalar and vector channel mixing is augmented by the two elementary
    geometry-mediated type changes: vector--geometry dot products create
    scalars, and scalar coefficients multiplying geometry vectors create
    vectors.  Invariant geometry gates modulate the result but never read the
    field, so the complete map remains exactly linear in ``field``.

    In Clebsch-Gordan terms the five branches realize ``0e -> 0e`` and
    ``1o -> 1o`` Schur channel mixing, ``1o x 1o -> 0e``
    (``scalar_from_vector_dots``), ``0e x 1o -> 1o`` (``vector_from_scalar``),
    and ``(1o x 1o -> 0e) x 1o -> 1o`` (``vector_from_vector_dots``).  For
    ``l <= 1`` inputs and outputs under O(D) this set is complete: the
    remaining formal paths are parity-forbidden (Levi-Civita) or reduce to
    these by isotropic-tensor identities.

    With the 2D pseudoscalar sector enabled (``field_pseudo_dim`` or
    ``out_pseudo_dim`` positive; both default to zero, which is bitwise
    identical to the pre-extension module), five additional branches close
    the product set over ``{0e, 0o, 1o}``, all still linear in ``field``:
    ``0o -> 0o`` Schur mixing (``pseudo_from_pseudo``),
    ``1o x 1o -> 0o`` wedges of field vectors against geometry vectors
    (``pseudo_from_vector_wedges``),
    ``0e x 0o -> 0o`` field scalars times geometry wedge pairs
    (``pseudo_from_scalar_wedges``),
    ``0o x 0o -> 0e`` field pseudos times geometry wedge pairs
    (``scalar_from_pseudo_wedges``), and the rotation product
    ``0o x 1o -> 1o`` of field pseudos with perpendicular geometry vectors
    (``vector_from_pseudo`` -- the branch that makes e.g. the circulation
    velocity :math:`\Gamma\,x^\perp/(2\pi|x|^2)` representable).  The
    geometry state itself carries no pseudoscalar sector (the operator
    stream is parity-even) and this is enforced.

    ``bounded_gate_invariants`` (default ``False``: bitwise identical to the
    historical module -- the knob adds no parameters and changes no
    initialization or RNG consumption) feeds every invariant gate a
    *compactified* copy of the geometry invariants: each geometry vector
    channel is mapped through :math:`v \mapsto v/\sqrt{1+|v|^2}` before its
    Gram products are taken -- so the Gram diagonal becomes the compactified
    radial invariant :math:`|v|^2/(1+|v|^2)` and every off-diagonal a
    correlation-like product bounded by one -- and each geometry scalar
    through the smooth odd map :math:`s \mapsto s/\sqrt{1+s^2}`.  The linear
    branches are untouched: they keep the raw geometry, so all radial
    structure keeps flowing through the non-saturating, field-linear paths.
    Equivariance is unchanged (the compactification rescales each vector by
    a function of its own invariant norm) and every map stays analytic.

    Why (measured far-field gate collapse): on the exterior potential-flow
    benchmark the trained ``MeshTransformer`` output projection's vector
    gate reads unbounded query-position invariants (:math:`|x|^2`-type,
    entering both through the operator lift's quadratic invariant lift into
    the geometry scalars and through the geometry-vector Grams), and the
    sigmoid saturates doubly-exponentially beyond the training annulus:
    RMS gate 0.52 at :math:`r=1.1`, 0.010 at :math:`r=4` (the training
    edge), ``2.3e-17`` at :math:`r=12`, while the kernel dictionary's
    member basis extrapolates exactly.  With every gate input bounded in
    ``[-1, 1]`` the gate pre-activation is bounded by
    :math:`\lVert W\rVert_1+|b|` for *all* inputs, so no saturation regime
    exists in query radius: as :math:`|x|\to\infty` the compactified
    invariants converge to their angular limits and the gate converges
    smoothly to a direction-dependent constant inside a compact subset of
    ``(0, 2)``.  Rejected alternative -- conditioning the gates on
    radius-free operator state only: by the time this module sees its
    geometry, the operator lift has mixed the query position into every
    scalar and vector channel, so a radius-free gate input would require a
    second position-free geometry stream through the whole model (far more
    invasive), and it would also delete the near-field radial gate
    structure the trained arms demonstrably use inside the annulus.

    Pre-registered test (restated from the far-field strong-inference study
    before the retraining run): the fixed arm -- identical to
    ``mesh_transformer_kernel_singpair_pseudo`` except this knob on the
    output projection -- trained on the ORIGINAL annulus
    :math:`r/L\in[1.05,4]` (3000 steps, seed 17, singpair_pseudo velocity
    family) must (a) restore the prediction's far-band decay exponent to
    :math:`\approx-2` beyond :math:`r=4` (mean far-band exponent delta
    ``|pred - exact| <= 0.5``, which the collapsed baseline fails at delta
    :math:`\approx-24`), and (b) drop the ``farfield_queries`` relative L2
    substantially below the 0.694 baseline (``< 0.35`` = supported,
    ``< 0.10`` = fully fixed), with in-distribution error within
    :math:`\pm 0.01` of the baseline 0.027.  Falsifier: no improvement
    means the gate collapse is a symptom, not the cause.

    MEASURED OUTCOME (p17_ff_boundedgate_s17, 2026-07-04): the falsifier
    fired.  The knob behaves exactly as designed -- the traced gate no
    longer saturates (RMS 0.375 at :math:`r=1.1` to 0.019 at :math:`r=12`,
    a smooth nonzero limit) and in-distribution error *improved* to 0.0215
    (circulation OOD 0.018 vs 0.033) -- but ``farfield_queries`` got worse,
    0.694 to 2.98, because the prediction now *grows* far out (far-band
    exponents up to +6 vs the exact -2; mean signed far delta +4.9, flipped
    from the baseline's -24 collapse).  The branch decomposition localizes
    the growth in the direct-drive output branches that multiply lifted
    drive coefficients by the raw geometry vectors
    (``vector_from_scalar``/``vector_from_pseudo``: the lifted drive
    scalar/pseudo norms grow like :math:`r`--:math:`r^2` and the geometry
    vectors like :math:`r`).  Interpretation: the baseline's gate
    saturation was real but load-bearing -- the trained model used the
    sigmoid tail as its only radial decay device to suppress these
    polynomially growing linear paths, and bounding the gate unmasked
    them.  The root far-field pathology is therefore the *pair* of
    unbounded query-radius dependences (saturating gate AND polynomially
    growing geometry-vector branches), not the gate alone.  The knob is
    kept (default off) as the measured, tested half of that fix and as the
    instrument that isolated the remaining half.

    RESOLUTION (third rung of the ladder): the remaining half cannot be
    fixed inside this module -- the growing geometry vectors are the
    module's *linear-branch inputs*, and compactifying them here would
    change the meaning of the geometry state for every caller.  The
    principled fix bounds at the source instead:
    ``MeshTransformer(bounded_query_geometry=True)`` injects the
    compactified query position :math:`\hat x = x/\sqrt{1+|x|^2}` into the
    query operator state, so both unbounded radial dependences (the gate
    inputs AND the geometry vectors these branches multiply) become
    bounded functions of query radius before this module ever sees them,
    while the kernel dictionary's exact members keep the raw coordinates.
    See the ``MeshTransformer`` constructor docstring for the full design
    note and that knob's pre-registered test.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        out_scalar_dim: int,
        out_vector_dim: int,
        *,
        field_pseudo_dim: int = 0,
        out_pseudo_dim: int = 0,
        bounded_gate_invariants: bool = False,
    ) -> None:
        """Allocate the Clebsch-Gordan branch maps that exist for the given
        sector widths (each branch parameter is created only when its input
        and output widths are positive, keeping narrower configurations'
        state dicts unchanged) and the zero-initialized invariant gates."""
        super().__init__()
        if field_pseudo_dim < 0 or out_pseudo_dim < 0:
            raise ValueError("pseudoscalar channel counts must be non-negative")
        self.bounded_gate_invariants = bool(bounded_gate_invariants)
        if out_vector_dim and not (
            field_vector_dim
            or ((field_scalar_dim or field_pseudo_dim) and geometry_vector_dim)
        ):
            raise ValueError(
                "A vector output requires a field-vector or geometry-vector basis"
            )
        n_geometry_wedges = geometry_vector_dim * (geometry_vector_dim - 1) // 2
        if out_pseudo_dim and not (
            field_pseudo_dim
            or (field_vector_dim and geometry_vector_dim)
            or (field_scalar_dim and n_geometry_wedges)
        ):
            raise ValueError(
                "A pseudoscalar output requires a field-pseudo basis, a "
                "field-vector x geometry-vector wedge basis, or a "
                "field-scalar x geometry-wedge basis"
            )
        self.geometry_vector_dim = geometry_vector_dim
        self.geometry_scalar_dim = geometry_scalar_dim
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.field_pseudo_dim = field_pseudo_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        self.out_pseudo_dim = out_pseudo_dim
        self._n_geometry_wedges = n_geometry_wedges

        geometry_invariants = (
            geometry_scalar_dim + geometry_vector_dim * (geometry_vector_dim + 1) // 2
        )
        self.scalar_gate = (
            nn.Linear(geometry_invariants, out_scalar_dim) if out_scalar_dim else None
        )
        self.vector_gate = (
            nn.Linear(geometry_invariants, out_vector_dim) if out_vector_dim else None
        )
        if self.scalar_gate is not None:
            nn.init.zeros_(self.scalar_gate.weight)
            nn.init.zeros_(self.scalar_gate.bias)
        if self.vector_gate is not None:
            nn.init.zeros_(self.vector_gate.weight)
            nn.init.zeros_(self.vector_gate.bias)

        self.scalar_from_scalar = (
            nn.Linear(field_scalar_dim, out_scalar_dim, bias=False)
            if field_scalar_dim and out_scalar_dim
            else None
        )
        if out_scalar_dim and field_vector_dim and geometry_vector_dim:
            self.scalar_from_vector_dots = nn.Linear(
                field_vector_dim * geometry_vector_dim,
                out_scalar_dim,
                bias=False,
            )
        else:
            self.scalar_from_vector_dots = None

        if out_vector_dim and field_vector_dim:
            self.vector_from_vector = nn.Parameter(
                torch.randn(out_vector_dim, field_vector_dim)
                / math.sqrt(field_vector_dim)
            )
        else:
            self.register_parameter("vector_from_vector", None)

        if out_vector_dim and field_vector_dim and geometry_vector_dim:
            self.vector_from_vector_dots = nn.Linear(
                field_vector_dim * geometry_vector_dim,
                out_vector_dim * geometry_vector_dim,
                bias=False,
            )
        else:
            self.vector_from_vector_dots = None

        if out_vector_dim and field_scalar_dim and geometry_vector_dim:
            self.vector_from_scalar = nn.Linear(
                field_scalar_dim,
                out_vector_dim * geometry_vector_dim,
                bias=False,
            )
        else:
            self.vector_from_scalar = None

        # Pseudoscalar-sector branches (all parameters exist only when the
        # relevant widths are positive, keeping the default bitwise).  Every
        # branch is bias-free and linear in the field; the invariant gate is
        # the only affine ingredient and it never reads the field.
        if out_scalar_dim and field_pseudo_dim and n_geometry_wedges:
            # 0o x 0o -> 0e: field pseudos against geometry wedge pairs.
            self.scalar_from_pseudo_wedges = nn.Linear(
                field_pseudo_dim * n_geometry_wedges,
                out_scalar_dim,
                bias=False,
            )
        else:
            self.scalar_from_pseudo_wedges = None
        if out_vector_dim and field_pseudo_dim and geometry_vector_dim:
            # 0o x 1o -> 1o: the rotation product against perpendicular
            # geometry vectors.
            self.vector_from_pseudo = nn.Linear(
                field_pseudo_dim,
                out_vector_dim * geometry_vector_dim,
                bias=False,
            )
        else:
            self.vector_from_pseudo = None
        if out_pseudo_dim and field_pseudo_dim:
            # 0o -> 0o Schur channel mixing.
            self.pseudo_from_pseudo = nn.Parameter(
                torch.randn(out_pseudo_dim, field_pseudo_dim)
                / math.sqrt(field_pseudo_dim)
            )
        else:
            self.register_parameter("pseudo_from_pseudo", None)
        if out_pseudo_dim and field_vector_dim and geometry_vector_dim:
            # 1o x 1o -> 0o: wedges of field vectors with geometry vectors.
            self.pseudo_from_vector_wedges = nn.Linear(
                field_vector_dim * geometry_vector_dim,
                out_pseudo_dim,
                bias=False,
            )
        else:
            self.pseudo_from_vector_wedges = None
        if out_pseudo_dim and field_scalar_dim and n_geometry_wedges:
            # 0e x 0o -> 0o: field scalars against geometry wedge pairs.
            self.pseudo_from_scalar_wedges = nn.Linear(
                field_scalar_dim * n_geometry_wedges,
                out_pseudo_dim,
                bias=False,
            )
        else:
            self.pseudo_from_scalar_wedges = None
        self.pseudo_gate = (
            nn.Linear(geometry_invariants, out_pseudo_dim) if out_pseudo_dim else None
        )
        if self.pseudo_gate is not None:
            nn.init.zeros_(self.pseudo_gate.weight)
            nn.init.zeros_(self.pseudo_gate.bias)

    def _geometry_invariants(
        self, geometry: ScalarVectorState
    ) -> Float[torch.Tensor, "n geometry_invariants"]:
        """Invariants fed to the sigmoid gates (never to the linear branches).

        With ``bounded_gate_invariants`` every entry is compactified into
        ``[-1, 1]`` (see the class docstring): vectors are rescaled by
        ``1/sqrt(1 + |v|^2)`` before their Grams so the diagonal becomes
        ``|v|^2 / (1 + |v|^2)``, and scalars pass through ``s/sqrt(1+s^2)``.
        Both maps are analytic and equivariant, so no saturation regime
        exists in any unbounded (e.g. query-radius) direction.
        """
        if not self.bounded_gate_invariants:
            return torch.cat(
                (geometry.scalars, _gram_invariants(geometry.vectors)), dim=-1
            )
        scalars = geometry.scalars * torch.rsqrt(1.0 + geometry.scalars.square())
        vectors = geometry.vectors * torch.rsqrt(
            1.0 + geometry.vectors.square().sum(dim=-1, keepdim=True)
        )
        return torch.cat((scalars, _gram_invariants(vectors)), dim=-1)

    def forward(
        self,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        """Evaluate every existing branch and gate; exactly linear in
        ``field`` at fixed ``geometry`` (see the class docstring for the
        branch-by-branch decomposition)."""
        if geometry.n_entities != field.n_entities:
            raise ValueError("geometry and field entity counts must match")
        if geometry.n_spatial_dims != field.n_spatial_dims:
            raise ValueError("geometry and field spatial dimensions must match")
        if geometry.scalars.shape[1] != self.geometry_scalar_dim:
            raise ValueError("geometry has the wrong number of scalar channels")
        if geometry.vectors.shape[1] != self.geometry_vector_dim:
            raise ValueError("geometry has the wrong number of vector channels")
        if geometry.pseudos.shape[1]:
            raise ValueError(
                "the geometry state carries no pseudoscalar sector; got "
                f"{geometry.pseudos.shape[1]} pseudo channels"
            )
        if field.scalars.shape[1] != self.field_scalar_dim:
            raise ValueError("field has the wrong number of scalar channels")
        if field.vectors.shape[1] != self.field_vector_dim:
            raise ValueError("field has the wrong number of vector channels")
        if field.pseudos.shape[1] != self.field_pseudo_dim:
            raise ValueError("field has the wrong number of pseudoscalar channels")
        invariants = self._geometry_invariants(geometry)
        geometry_wedges = (
            _wedge_invariants(geometry.vectors)
            if (
                self.scalar_from_pseudo_wedges is not None
                or self.pseudo_from_scalar_wedges is not None
            )
            else None
        )

        scalar_terms: list[torch.Tensor] = []
        if self.scalar_from_scalar is not None:
            scalar_terms.append(self.scalar_from_scalar(field.scalars))
        if self.scalar_from_vector_dots is not None:
            dots = torch.einsum(
                "nfd,ngd->nfg", field.vectors, geometry.vectors
            ).flatten(1)
            scalar_terms.append(self.scalar_from_vector_dots(dots))
        if self.scalar_from_pseudo_wedges is not None:
            products = (
                field.pseudos[:, :, None] * geometry_wedges[:, None, :]
            ).flatten(1)
            scalar_terms.append(self.scalar_from_pseudo_wedges(products))
        if self.out_scalar_dim:
            if self.scalar_gate is None:
                raise RuntimeError("scalar gate missing for scalar output")
            scalar_gates = 2.0 * torch.sigmoid(self.scalar_gate(invariants))
            if scalar_terms:
                scalars = scalar_terms[0]
                for term in scalar_terms[1:]:
                    scalars = scalars + term
            else:
                scalars = torch.zeros_like(scalar_gates)
            # 2 sigmoid(0) = 1: conditioning begins as a neutral
            # multiplicative gate while retaining an unrestricted,
            # geometry-dependent derivative.
            scalars = scalars * scalar_gates
        else:
            scalars = None

        if self.out_vector_dim:
            vector_terms: list[torch.Tensor] = []
            if self.vector_from_vector is not None:
                vector_terms.append(
                    _mix_channels(self.vector_from_vector, field.vectors)
                )
            if self.vector_from_vector_dots is not None:
                dots = torch.einsum(
                    "nfd,ngd->nfg", field.vectors, geometry.vectors
                ).flatten(1)
                coefficients = self.vector_from_vector_dots(dots).reshape(
                    field.n_entities,
                    self.out_vector_dim,
                    self.geometry_vector_dim,
                )
                vector_terms.append(_apply_coefficients(coefficients, geometry.vectors))
            if self.vector_from_scalar is not None:
                coefficients = self.vector_from_scalar(field.scalars).reshape(
                    field.n_entities,
                    self.out_vector_dim,
                    self.geometry_vector_dim,
                )
                vector_terms.append(_apply_coefficients(coefficients, geometry.vectors))
            if self.vector_from_pseudo is not None:
                # Rotation product: the perpendicular of a polar vector is
                # axial, and its pairing with exactly one pseudoscalar
                # coefficient is polar again.
                coefficients = self.vector_from_pseudo(field.pseudos).reshape(
                    field.n_entities,
                    self.out_vector_dim,
                    self.geometry_vector_dim,
                )
                vector_terms.append(
                    _apply_coefficients(coefficients, _vector_perp(geometry.vectors))
                )
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for vector output")
            vector_gates = 2.0 * torch.sigmoid(self.vector_gate(invariants))
            if vector_terms:
                vectors = vector_terms[0]
                for term in vector_terms[1:]:
                    vectors = vectors + term
            else:
                vectors = vector_gates.new_zeros(
                    field.n_entities,
                    self.out_vector_dim,
                    field.n_spatial_dims,
                )
            vectors = vectors * vector_gates[:, :, None]
        else:
            vectors = field.vectors.new_empty(field.n_entities, 0, field.n_spatial_dims)

        if self.out_pseudo_dim:
            pseudo_terms: list[torch.Tensor] = []
            if self.pseudo_from_pseudo is not None:
                # Plain matrix product; keep it an explicit single GEMM.
                pseudo_terms.append(
                    field.pseudos @ self.pseudo_from_pseudo.transpose(0, 1)
                )
            if self.pseudo_from_vector_wedges is not None:
                wedges = _pair_wedges(field.vectors, geometry.vectors).flatten(1)
                pseudo_terms.append(self.pseudo_from_vector_wedges(wedges))
            if self.pseudo_from_scalar_wedges is not None:
                products = (
                    field.scalars[:, :, None] * geometry_wedges[:, None, :]
                ).flatten(1)
                pseudo_terms.append(self.pseudo_from_scalar_wedges(products))
            if self.pseudo_gate is None:
                raise RuntimeError("pseudo gate missing for pseudoscalar output")
            pseudo_gates = 2.0 * torch.sigmoid(self.pseudo_gate(invariants))
            if pseudo_terms:
                pseudos = pseudo_terms[0]
                for term in pseudo_terms[1:]:
                    pseudos = pseudos + term
            else:
                pseudos = pseudo_gates.new_zeros(field.n_entities, self.out_pseudo_dim)
            pseudos = pseudos * pseudo_gates
        else:
            pseudos = field.scalars.new_empty(field.n_entities, 0)

        if scalars is None:
            scalars = vectors.new_empty(field.n_entities, 0)
        else:
            vectors = vectors.to(dtype=scalars.dtype)
        return ScalarVectorState(scalars, vectors, pseudos.to(dtype=scalars.dtype))


class ZeroPreservingFeedForward(nn.Module):
    """Nonlinear equivariant update whose output is exactly zero at zero field.

    Pseudoscalar field channels are gated by a sigmoid of true-scalar
    invariants (which, with a pseudo sector present, include the hidden
    ``0o x 0o -> 0e`` pair products) -- an invariant multiplicative gate is
    parity-safe, while feeding a pseudoscalar directly through a nonlinearity
    would not be.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        field_pseudo_dim: int = 0,
    ) -> None:
        """Build the lift/gate/project sandwich: two geometry-conditioned
        linear maps around invariant sigmoid gates (all zero-preserving)."""
        super().__init__()
        self.field_pseudo_dim = field_pseudo_dim
        self.lift = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_pseudo_dim=field_pseudo_dim,
            out_pseudo_dim=field_pseudo_dim,
        )
        invariant_dim = (
            geometry_scalar_dim
            + geometry_vector_dim * (geometry_vector_dim + 1) // 2
            + field_scalar_dim
            + field_vector_dim * (field_vector_dim + 1) // 2
            + field_pseudo_dim * (field_pseudo_dim + 1) // 2
        )
        self.scalar_gate = nn.Linear(invariant_dim, field_scalar_dim)
        self.vector_gate = (
            nn.Linear(invariant_dim, field_vector_dim) if field_vector_dim else None
        )
        self.pseudo_gate = (
            nn.Linear(invariant_dim, field_pseudo_dim) if field_pseudo_dim else None
        )
        self.project = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_pseudo_dim=field_pseudo_dim,
            out_pseudo_dim=field_pseudo_dim,
        )

    def forward(
        self,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        """Lift the field, gate every sector by mixed geometry+field
        invariants, and project back; output vanishes exactly at zero
        field (each term carries at least one factor of the field)."""
        hidden = self.lift(geometry, field)
        invariant_parts = [
            geometry.scalars,
            _gram_invariants(geometry.vectors),
            hidden.scalars,
            _gram_invariants(hidden.vectors),
        ]
        if self.field_pseudo_dim:
            invariant_parts.append(_pseudo_pair_invariants(hidden.pseudos))
        invariants = torch.cat(invariant_parts, dim=-1)
        scalars = hidden.scalars * torch.sigmoid(self.scalar_gate(invariants))
        vectors = hidden.vectors
        if vectors.shape[1]:
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for non-empty vector state")
            vectors = vectors * torch.sigmoid(self.vector_gate(invariants))[:, :, None]
        pseudos = hidden.pseudos
        if self.field_pseudo_dim:
            if self.pseudo_gate is None:
                raise RuntimeError("pseudo gate missing for non-empty pseudo state")
            pseudos = pseudos * torch.sigmoid(self.pseudo_gate(invariants))
        return self.project(
            geometry,
            ScalarVectorState(scalars, vectors, pseudos.to(dtype=scalars.dtype)),
        )


class QuadraticFieldReadIn(nn.Module):
    r"""Pointwise field composition of DECLARED polynomial drive degree two.

    The third field-mode class, extending the two-modes-two-classes
    discipline (``LinearMeshFieldBlock`` / ``NonlinearZeroMeshFieldBlock``)
    to "one Python class per declared drive-degree law": selected by
    ``MeshTransformer(field_mode="quadratic")``, never by a flag on the
    nonlinear class.  Given a field state :math:`u` that is EXACTLY linear
    (and zero-preserving) in the drive at fixed geometry -- which the
    quadratic mode guarantees by routing the entire drive path through the
    existing drive-linear machinery -- this module emits

    .. math::

       F \;=\; u \;+\; s \odot g(\mathrm{op}) \odot B\big(L_2 u,\ L_3 u\big),

    where :math:`L_2, L_3` are bias-free typed linear maps (exactly linear,
    parity-preserving factor projections), :math:`B` is a BILINEAR typed
    product drawn from the closed ``{0e, 0o, 1o}`` product set already used
    by :class:`GeometryConditionedLinear` (``0e x 0e -> 0e`` scalar pairs,
    ``1o . 1o -> 0e`` dots, ``0o x 0o -> 0e`` pseudo pairs,
    ``0e x 1o -> 1o`` scalar-vector products, ``0o x 1o^\perp -> 1o``
    rotation products, ``1o ^ 1o -> 0o`` wedges, and ``0e x 0o -> 0o``
    scalar-pseudo products), :math:`g` are sigmoid gates of geometry
    (operator) invariants that never read the field, and :math:`s` is a
    small learned per-channel scale.  Because every learned ingredient is
    either linear in the field, bilinear in the field, or field-independent,
    :math:`F` is EXACTLY a polynomial of degree :math:`\le 2` in the drive
    -- for any weights, by construction; the property is provable and
    machine-precision testable (scale the drive by :math:`\alpha` and the
    output is exactly :math:`c_1\alpha + c_2\alpha^2`).  Zero preservation
    is inherited: :math:`u = 0` at zero drive annihilates every term.

    Why the quadratic law lives in a single READ-IN composition rather than
    in a stackable residual block: a same-interface block whose update adds
    :math:`B(F, F)` composes to degree :math:`2^k` after :math:`k` layers
    -- exactly the implicit-degree escalation this class exists to forbid
    (the iteration-34 diagnosis measured effective drive degree ~21 in the
    ``zero_preserving_nonlinear`` mode against targets of degree 1 and 2).
    Degree closure under stacking would require a graded (linear, quadratic)
    state pair threaded through every block and the kernel decoder -- a far
    larger interface change buying nothing the single composition does not
    already provide: with drive-linear states available AT THE QUERY, one
    bilinear composition spans exactly the products of boundary integrals
    (e.g. a Bernoulli pressure :math:`|u(x)|^2`) that degree-2 targets
    need.  The composition is therefore applied once, to the assembled
    query field state (direct drive lift plus kernel/moment message),
    immediately before the drive-linear output projection.

    The generalization to declared degree :math:`k` is the same
    construction with a degree-graded tuple of states and :math:`k - 1`
    bilinear compositions (each pairing grades that sum to the target
    grade); this class is the first instance, :math:`k = 2`, matching the
    degree of every current benchmark target.

    Parameters
    ----------
    geometry_scalar_dim, geometry_vector_dim : int
        Channel counts of the (query-side) operator state whose invariants
        feed the gates.
    field_scalar_dim, field_vector_dim : int
        Channel counts of the field state; the emitted state carries the
        same typed channel counts (residual form).
    field_pseudo_dim : int, optional
        Pseudoscalar (``0o``, 2D-only) channel count of the field state.
        The parity-odd product branches exist only when positive, keeping
        the pseudo-free state dict minimal.
    factor_scalar_dim, factor_vector_dim : int or None, optional
        Widths of the bilinear factor projections :math:`L_2, L_3`.  The
        defaults (``field_scalar_dim // 4`` and ``field_vector_dim // 2``,
        floored at one where the sector exists) bound the pair-product
        feature count while leaving every product type represented.
    layer_scale : float, optional
        Initial per-channel scale of the quadratic term.  The default
        ``1e-2`` matches the residual-update discipline used throughout the
        stack: the mode starts indistinguishable from the linear machinery
        and the quadratic component is learned, not imposed.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        field_pseudo_dim: int = 0,
        factor_scalar_dim: int | None = None,
        factor_vector_dim: int | None = None,
        layer_scale: float = 1.0e-2,
    ) -> None:
        """Build the two exactly-linear factor maps, the bilinear product
        branches over ``{0e, 0o, 1o}``, the field-blind operator gates,
        and the layer scale (see the class docstring for the law)."""
        super().__init__()
        if field_scalar_dim < 1:
            raise ValueError("field_scalar_dim must be positive")
        if field_vector_dim < 0 or field_pseudo_dim < 0:
            raise ValueError("field channel counts must be non-negative")
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.field_pseudo_dim = field_pseudo_dim
        factor_scalar = (
            max(field_scalar_dim // 4, 1)
            if factor_scalar_dim is None
            else factor_scalar_dim
        )
        factor_vector = (
            (max(field_vector_dim // 2, 1) if field_vector_dim else 0)
            if factor_vector_dim is None
            else factor_vector_dim
        )
        if factor_scalar < 1:
            raise ValueError("factor_scalar_dim must be positive")
        if factor_vector and not field_vector_dim:
            raise ValueError("factor_vector_dim requires a field-vector input basis")
        self.factor_scalar_dim = factor_scalar
        self.factor_vector_dim = factor_vector

        # The two drive-linear factor maps L2, L3: bias-free typed linear
        # projections with every quadratic invariant lift disabled, so each
        # factor is EXACTLY linear and homogeneous in the field.
        def _factor() -> TypedProjection:
            """One bias-free, invariant-lift-free (hence exactly field-linear
            and homogeneous) typed factor projection."""
            return TypedProjection(
                field_scalar_dim,
                field_vector_dim,
                factor_scalar,
                factor_vector,
                scalar_bias=False,
                include_vector_invariants=False,
                pseudo_in=field_pseudo_dim,
                pseudo_out=field_pseudo_dim,
            )

        self.left_factor = _factor()
        self.right_factor = _factor()

        # Bilinear product branches (all bias-free; every parameter is a
        # coefficient of one degree-2 monomial in the field).
        self.scalar_from_scalar_pairs = nn.Linear(
            factor_scalar * factor_scalar, field_scalar_dim, bias=False
        )
        self.scalar_from_vector_dots = (
            nn.Linear(factor_vector * factor_vector, field_scalar_dim, bias=False)
            if factor_vector
            else None
        )
        self.scalar_from_pseudo_pairs = (
            nn.Linear(field_pseudo_dim * field_pseudo_dim, field_scalar_dim, bias=False)
            if field_pseudo_dim
            else None
        )
        if field_vector_dim and factor_vector:
            self.vector_from_scalar_left = nn.Linear(
                factor_scalar, field_vector_dim * factor_vector, bias=False
            )
            self.vector_from_scalar_right = nn.Linear(
                factor_scalar, field_vector_dim * factor_vector, bias=False
            )
        else:
            self.vector_from_scalar_left = None
            self.vector_from_scalar_right = None
        if field_vector_dim and factor_vector and field_pseudo_dim:
            # 0o x 1o -> 1o rotation products (2D only; the model-level
            # pseudo coherence rules guarantee planarity here).
            self.vector_from_pseudo_left = nn.Linear(
                field_pseudo_dim, field_vector_dim * factor_vector, bias=False
            )
            self.vector_from_pseudo_right = nn.Linear(
                field_pseudo_dim, field_vector_dim * factor_vector, bias=False
            )
        else:
            self.vector_from_pseudo_left = None
            self.vector_from_pseudo_right = None
        if field_pseudo_dim:
            self.pseudo_from_vector_wedges = (
                nn.Linear(factor_vector * factor_vector, field_pseudo_dim, bias=False)
                if factor_vector
                else None
            )
            self.pseudo_from_scalar_pseudo_left = nn.Linear(
                factor_scalar * field_pseudo_dim, field_pseudo_dim, bias=False
            )
            self.pseudo_from_scalar_pseudo_right = nn.Linear(
                factor_scalar * field_pseudo_dim, field_pseudo_dim, bias=False
            )
        else:
            self.pseudo_from_vector_wedges = None
            self.pseudo_from_scalar_pseudo_left = None
            self.pseudo_from_scalar_pseudo_right = None

        # Operator conditioning: invariant sigmoid gates, zero-initialized to
        # the neutral value 1 (2 sigmoid(0)), exactly the
        # GeometryConditionedLinear gate discipline.  Gates never read the
        # field, so conditioning cannot raise the drive degree.
        geometry_invariants = (
            geometry_scalar_dim + geometry_vector_dim * (geometry_vector_dim + 1) // 2
        )
        self.scalar_gate = nn.Linear(geometry_invariants, field_scalar_dim)
        self.vector_gate = (
            nn.Linear(geometry_invariants, field_vector_dim)
            if field_vector_dim
            else None
        )
        self.pseudo_gate = (
            nn.Linear(geometry_invariants, field_pseudo_dim)
            if field_pseudo_dim
            else None
        )
        for gate in (self.scalar_gate, self.vector_gate, self.pseudo_gate):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
        self.scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=layer_scale,
            pseudo_dim=field_pseudo_dim,
        )

    def forward(
        self,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
    ) -> ScalarVectorState:
        """Emit ``field + scale * gate(op) * B(L2 field, L3 field)`` — the
        declared degree-2 composition; exactly quadratic in the drive when
        ``field`` is drive-linear (see the class docstring)."""
        if geometry.n_entities != field.n_entities:
            raise ValueError("geometry and field entity counts must match")
        if field.scalars.shape[1] != self.field_scalar_dim:
            raise ValueError("field has the wrong number of scalar channels")
        if field.vectors.shape[1] != self.field_vector_dim:
            raise ValueError("field has the wrong number of vector channels")
        if field.pseudos.shape[1] != self.field_pseudo_dim:
            raise ValueError("field has the wrong number of pseudoscalar channels")
        if geometry.pseudos.shape[1]:
            raise ValueError(
                "the geometry state carries no pseudoscalar sector; got "
                f"{geometry.pseudos.shape[1]} pseudo channels"
            )
        n = field.n_entities
        left = self.left_factor(field)
        right = self.right_factor(field)
        invariants = torch.cat(
            (geometry.scalars, _gram_invariants(geometry.vectors)), dim=-1
        )

        scalar_terms = [
            self.scalar_from_scalar_pairs(
                (left.scalars[:, :, None] * right.scalars[:, None, :]).flatten(1)
            )
        ]
        if self.scalar_from_vector_dots is not None:
            dots = torch.einsum("nfd,ngd->nfg", left.vectors, right.vectors)
            scalar_terms.append(self.scalar_from_vector_dots(dots.flatten(1)))
        if self.scalar_from_pseudo_pairs is not None:
            scalar_terms.append(
                self.scalar_from_pseudo_pairs(
                    (left.pseudos[:, :, None] * right.pseudos[:, None, :]).flatten(1)
                )
            )
        scalars = scalar_terms[0]
        for term in scalar_terms[1:]:
            scalars = scalars + term
        scalars = scalars * (2.0 * torch.sigmoid(self.scalar_gate(invariants)))

        if self.field_vector_dim:
            vector_terms: list[torch.Tensor] = []
            for coefficients_map, coefficient_source, vector_source, perp in (
                (self.vector_from_scalar_left, left.scalars, right.vectors, False),
                (self.vector_from_scalar_right, right.scalars, left.vectors, False),
                (self.vector_from_pseudo_left, left.pseudos, right.vectors, True),
                (self.vector_from_pseudo_right, right.pseudos, left.vectors, True),
            ):
                if coefficients_map is None:
                    continue
                coefficients = coefficients_map(coefficient_source).reshape(
                    n, self.field_vector_dim, self.factor_vector_dim
                )
                basis = _vector_perp(vector_source) if perp else vector_source
                vector_terms.append(_apply_coefficients(coefficients, basis))
            if vector_terms:
                vectors = vector_terms[0]
                for term in vector_terms[1:]:
                    vectors = vectors + term
            else:
                vectors = field.vectors.new_zeros(
                    n, self.field_vector_dim, field.n_spatial_dims
                )
            if self.vector_gate is None:
                raise RuntimeError("vector gate missing for vector field")
            vectors = vectors * (
                2.0 * torch.sigmoid(self.vector_gate(invariants))[:, :, None]
            )
        else:
            vectors = field.vectors.new_empty(n, 0, field.n_spatial_dims)

        if self.field_pseudo_dim:
            pseudo_terms: list[torch.Tensor] = []
            if self.pseudo_from_vector_wedges is not None:
                wedges = _pair_wedges(left.vectors, right.vectors)
                pseudo_terms.append(self.pseudo_from_vector_wedges(wedges.flatten(1)))
            pseudo_terms.append(
                self.pseudo_from_scalar_pseudo_left(
                    (left.scalars[:, :, None] * right.pseudos[:, None, :]).flatten(1)
                )
            )
            pseudo_terms.append(
                self.pseudo_from_scalar_pseudo_right(
                    (right.scalars[:, :, None] * left.pseudos[:, None, :]).flatten(1)
                )
            )
            pseudos = pseudo_terms[0]
            for term in pseudo_terms[1:]:
                pseudos = pseudos + term
            if self.pseudo_gate is None:
                raise RuntimeError("pseudo gate missing for pseudo field")
            pseudos = pseudos * (2.0 * torch.sigmoid(self.pseudo_gate(invariants)))
        else:
            pseudos = scalars.new_empty(n, 0)

        quadratic = self.scale(
            ScalarVectorState(
                scalars,
                vectors.to(dtype=scalars.dtype),
                pseudos.to(dtype=scalars.dtype),
            )
        )
        return _add(field, quadratic)


def _init_moment_segment_gain(
    module: nn.Module, n_moment_segments: int, heads: int
) -> None:
    """Attach the per-segment moment-pool log-gain parameter (or ``None``).

    ``n_moment_segments == 0`` (the default) registers ``None``: no
    parameter is created, the state dict is unchanged, and the block is
    bitwise the historical one.  A positive count creates a zero-initialized
    ``(n_moment_segments, heads)`` log-gain, so the pooled combination
    reproduces the plain quadrature sum at initialization (see
    :meth:`MeshAttention.build_moments`).
    """
    if not isinstance(n_moment_segments, int) or isinstance(n_moment_segments, bool):
        raise TypeError(
            f"n_moment_segments must be an integer, got {n_moment_segments!r}"
        )
    if n_moment_segments < 0:
        raise ValueError(
            f"n_moment_segments must be nonnegative, got {n_moment_segments}"
        )
    if n_moment_segments:
        module.moment_segment_log_gain = nn.Parameter(
            torch.zeros(n_moment_segments, heads)
        )
    else:
        module.register_parameter("moment_segment_log_gain", None)


def _init_moment_pool_balance(
    module: nn.Module, moment_pool_balanced: bool, n_moment_segments: int
) -> None:
    """Validate and store the balanced-pool flag (external-review balanced
    arm). ``True`` offsets each segment's log-gain by ln(mean measure) -
    ln(segment measure) at pool time (see ``MeshAttention.build_moments``);
    it requires the per-segment pool to exist. Default ``False`` is bitwise
    the historical pool."""
    if not isinstance(moment_pool_balanced, bool):
        raise ValueError(
            f"moment_pool_balanced must be a bool, got {moment_pool_balanced!r}"
        )
    if moment_pool_balanced and not n_moment_segments:
        raise ValueError(
            "moment_pool_balanced=True requires n_moment_segments > 0: the "
            "measure balance offsets the per-segment moment-pool log-gains, "
            "so without segments there is nothing to balance"
        )
    module.moment_pool_balanced = moment_pool_balanced


def _moment_segment_gain(
    module: nn.Module, moment_segments: Sequence[slice] | None
) -> Float[torch.Tensor, "n_segments heads"] | None:
    """Resolve the log-gain for a forward call, rejecting mismatched use."""
    if moment_segments is None:
        return None
    gain = module.moment_segment_log_gain
    if gain is None:
        raise ValueError(
            f"{type(module).__name__} received moment_segments but was "
            "constructed with n_moment_segments=0; construct the block with "
            "the segment count to enable the per-segment moment pool"
        )
    return gain


class MeshOperatorBlock(nn.Module):
    """Nonlinear global self-interaction block for operator geometry.

    ``n_moment_segments`` (default 0: bitwise pre-extension behavior) equips
    the block with per-segment, per-head moment-pool log-gains; forward
    calls may then pass ``moment_segments`` (one slice per source segment,
    e.g. per boundary component) to combine per-segment attention moments
    through the learned dimensionless gains.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        hidden_ratio: int = 2,
        layer_scale: float = 1.0e-2,
        entity_chunk_size: int | None = 65536,
        n_moment_segments: int = 0,
        moment_pool_balanced: bool = False,
    ) -> None:
        """Assemble the pre-norm self-attention + feed-forward residual pair
        (both residuals layer-scaled) at the given state widths."""
        super().__init__()
        self.attention_norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.attention = MeshAttention(
            query_scalar_dim=scalar_dim,
            query_vector_dim=vector_dim,
            key_scalar_dim=scalar_dim,
            key_vector_dim=vector_dim,
            value_scalar_dim=scalar_dim,
            value_vector_dim=vector_dim,
            out_scalar_dim=scalar_dim,
            out_vector_dim=vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank if vector_dim else 0,
            scalar_value_dim=max(scalar_dim // heads, 1),
            vector_value_dim=max(vector_dim // heads, 1) if vector_dim else 0,
            value_scalar_bias=True,
            output_scalar_bias=True,
            entity_chunk_size=entity_chunk_size,
        )
        self.attention_scale = StateLayerScale(scalar_dim, vector_dim, init=layer_scale)
        self.feed_forward_norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.feed_forward = GeometryFeedForward(
            scalar_dim, vector_dim, hidden_ratio=hidden_ratio
        )
        self.feed_forward_scale = StateLayerScale(
            scalar_dim, vector_dim, init=layer_scale
        )
        _init_moment_segment_gain(self, n_moment_segments, heads)
        _init_moment_pool_balance(self, moment_pool_balanced, n_moment_segments)

    def forward(
        self,
        source_mesh: Mesh,
        state: ScalarVectorState,
        moment_segments: Sequence[slice] | None = None,
    ) -> ScalarVectorState:
        """One residual step of global typed self-attention over the source
        mesh, then one residual typed feed-forward step."""
        normalized = self.attention_norm(state)
        state = _add(
            state,
            self.attention_scale(
                self.attention(
                    source_mesh,
                    normalized,
                    normalized,
                    normalized,
                    segments=moment_segments,
                    segment_log_gain=_moment_segment_gain(self, moment_segments),
                    segment_measure_balance=self.moment_pool_balanced,
                )
            ),
        )
        return _add(
            state,
            self.feed_forward_scale(self.feed_forward(self.feed_forward_norm(state))),
        )


class PointwiseGeometryBlock(nn.Module):
    """Nonlinear typed feature map with no interaction or spatial cutoff."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        hidden_ratio: int = 2,
        layer_scale: float = 1.0e-2,
    ) -> None:
        """Assemble the norm → feed-forward → layer-scale residual unit."""
        super().__init__()
        self.norm = TypedRMSNorm(scalar_dim, vector_dim)
        self.feed_forward = GeometryFeedForward(
            scalar_dim, vector_dim, hidden_ratio=hidden_ratio
        )
        self.scale = StateLayerScale(scalar_dim, vector_dim, init=layer_scale)

    def forward(self, state: ScalarVectorState) -> ScalarVectorState:
        """One residual pointwise equivariant feed-forward step."""
        return _add(
            state,
            self.scale(self.feed_forward(self.norm(state))),
        )


class LinearMeshFieldBlock(nn.Module):
    r"""Global field block with an exact fixed-geometry superposition law.

    ``field_pseudo_dim`` (default 0: bitwise pre-extension behavior) adds 2D
    pseudoscalar field channels.  They ride the invariant value path of the
    attention and the pseudo branches of the pointwise map; with
    ``value_include_vector_invariants=False`` the pseudo value projection is
    a bias-free linear map of the pseudo field alone, so the superposition
    law extends to the pseudo sector exactly.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        layer_scale: float = 1.0e-2,
        message_layer_scale: float | None = None,
        entity_chunk_size: int | None = 65536,
        field_pseudo_dim: int = 0,
        n_moment_segments: int = 0,
        moment_pool_balanced: bool = False,
    ) -> None:
        """Assemble the superposition-preserving unit: geometry-keyed
        attention with strictly field-linear values, a geometry-conditioned
        pointwise map, and layer scales (all bias-free on field paths)."""
        super().__init__()
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.field_pseudo_dim = field_pseudo_dim
        self.attention = MeshAttention(
            query_scalar_dim=geometry_scalar_dim,
            query_vector_dim=geometry_vector_dim,
            key_scalar_dim=geometry_scalar_dim,
            key_vector_dim=geometry_vector_dim,
            value_scalar_dim=field_scalar_dim,
            value_vector_dim=field_vector_dim,
            out_scalar_dim=field_scalar_dim,
            out_vector_dim=field_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank if geometry_vector_dim else 0,
            scalar_value_dim=max(field_scalar_dim // heads, 1),
            vector_value_dim=(
                max(field_vector_dim // heads, 1) if field_vector_dim else 0
            ),
            value_scalar_bias=False,
            value_include_vector_invariants=False,
            output_scalar_bias=False,
            entity_chunk_size=entity_chunk_size,
            value_pseudo_dim=field_pseudo_dim,
            out_pseudo_dim=field_pseudo_dim,
            pseudo_value_dim=(
                max(field_pseudo_dim // heads, 1) if field_pseudo_dim else 0
            ),
        )
        self.pointwise = GeometryConditionedLinear(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_pseudo_dim=field_pseudo_dim,
            out_pseudo_dim=field_pseudo_dim,
        )
        self.message_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=(layer_scale if message_layer_scale is None else message_layer_scale),
            pseudo_dim=field_pseudo_dim,
        )
        self.pointwise_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=layer_scale,
            pseudo_dim=field_pseudo_dim,
        )
        _init_moment_segment_gain(self, n_moment_segments, heads)
        _init_moment_pool_balance(self, moment_pool_balanced, n_moment_segments)

    def build_source_moments(
        self,
        source_mesh: Mesh,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        moment_segments: Sequence[slice] | None = None,
    ) -> AttentionMoments:
        """Compress one global source integral for reuse by many queries."""
        return self.attention.build_moments(
            source_mesh,
            source_geometry,
            source_field,
            segments=moment_segments,
            segment_log_gain=_moment_segment_gain(self, moment_segments),
            segment_measure_balance=self.moment_pool_balanced,
        )

    def evaluate_cross(
        self,
        query_geometry: ScalarVectorState,
        moments: AttentionMoments,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        """Evaluate precomputed source moments at independent receivers."""
        message = self.message_scale(
            self.attention.evaluate_moments(query_geometry, moments)
        )
        state = message if query_field is None else _add(query_field, message)
        return _add(
            state,
            self.pointwise_scale(self.pointwise(query_geometry, state)),
        )

    def cross(
        self,
        source_mesh: Mesh,
        query_geometry: ScalarVectorState,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        query_field: ScalarVectorState | None = None,
        moment_segments: Sequence[slice] | None = None,
    ) -> ScalarVectorState:
        """Source-to-query pass: :meth:`build_source_moments` then
        :meth:`evaluate_cross` (fused convenience form)."""
        return self.evaluate_cross(
            query_geometry,
            self.build_source_moments(
                source_mesh,
                source_geometry,
                source_field,
                moment_segments=moment_segments,
            ),
            query_field,
        )

    def forward(
        self,
        source_mesh: Mesh,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
        moment_segments: Sequence[slice] | None = None,
    ) -> ScalarVectorState:
        """Self-interaction form of :meth:`cross` (source = query set)."""
        return self.cross(
            source_mesh,
            geometry,
            geometry,
            field,
            field,
            moment_segments=moment_segments,
        )


class NonlinearZeroMeshFieldBlock(nn.Module):
    r"""Global content-dependent field block with exact zero preservation.

    ``field_pseudo_dim`` (default 0: bitwise pre-extension behavior) adds 2D
    pseudoscalar field channels.  Keys and queries read them only through
    invariant ``0o x 0o -> 0e`` pair products; value pseudo features may
    additionally use field-vector wedges.  Both are at least linear in the
    field, so zero drive still produces an exactly zero message.
    """

    def __init__(
        self,
        geometry_scalar_dim: int,
        geometry_vector_dim: int,
        field_scalar_dim: int,
        field_vector_dim: int,
        *,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        layer_scale: float = 1.0e-2,
        message_layer_scale: float | None = None,
        entity_chunk_size: int | None = 65536,
        field_pseudo_dim: int = 0,
        n_moment_segments: int = 0,
        moment_pool_balanced: bool = False,
    ) -> None:
        """Assemble the content-dependent unit: attention keyed on the
        concatenated geometry+field state (values field-only, so zero field
        gives a zero message), a zero-preserving pointwise map, and layer
        scales."""
        super().__init__()
        self.field_scalar_dim = field_scalar_dim
        self.field_vector_dim = field_vector_dim
        self.field_pseudo_dim = field_pseudo_dim
        combined_scalar = geometry_scalar_dim + field_scalar_dim
        combined_vector = geometry_vector_dim + field_vector_dim
        self.attention = MeshAttention(
            query_scalar_dim=combined_scalar,
            query_vector_dim=combined_vector,
            key_scalar_dim=combined_scalar,
            key_vector_dim=combined_vector,
            value_scalar_dim=field_scalar_dim,
            value_vector_dim=field_vector_dim,
            out_scalar_dim=field_scalar_dim,
            out_vector_dim=field_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=(vector_rank if geometry_vector_dim + field_vector_dim else 0),
            scalar_value_dim=max(field_scalar_dim // heads, 1),
            vector_value_dim=(
                max(field_vector_dim // heads, 1) if field_vector_dim else 0
            ),
            value_scalar_bias=False,
            value_include_vector_invariants=True,
            output_scalar_bias=False,
            entity_chunk_size=entity_chunk_size,
            query_pseudo_dim=field_pseudo_dim,
            key_pseudo_dim=field_pseudo_dim,
            value_pseudo_dim=field_pseudo_dim,
            out_pseudo_dim=field_pseudo_dim,
            pseudo_value_dim=(
                max(field_pseudo_dim // heads, 1) if field_pseudo_dim else 0
            ),
        )
        self.pointwise = ZeroPreservingFeedForward(
            geometry_scalar_dim,
            geometry_vector_dim,
            field_scalar_dim,
            field_vector_dim,
            field_pseudo_dim=field_pseudo_dim,
        )
        self.message_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=(layer_scale if message_layer_scale is None else message_layer_scale),
            pseudo_dim=field_pseudo_dim,
        )
        self.pointwise_scale = StateLayerScale(
            field_scalar_dim,
            field_vector_dim,
            init=layer_scale,
            pseudo_dim=field_pseudo_dim,
        )
        _init_moment_segment_gain(self, n_moment_segments, heads)
        _init_moment_pool_balance(self, moment_pool_balanced, n_moment_segments)

    def build_source_moments(
        self,
        source_mesh: Mesh,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        moment_segments: Sequence[slice] | None = None,
    ) -> AttentionMoments:
        """Compress content-dependent source keys and values for query reuse."""
        return self.attention.build_moments(
            source_mesh,
            source_geometry.cat(source_field),
            source_field,
            segments=moment_segments,
            segment_log_gain=_moment_segment_gain(self, moment_segments),
            segment_measure_balance=self.moment_pool_balanced,
        )

    def evaluate_cross(
        self,
        query_geometry: ScalarVectorState,
        moments: AttentionMoments,
        query_field: ScalarVectorState | None = None,
    ) -> ScalarVectorState:
        """Evaluate cached source moments while preserving the zero solution."""
        if query_field is None:
            query_field = ScalarVectorState.zeros(
                query_geometry.n_entities,
                self.field_scalar_dim,
                self.field_vector_dim,
                query_geometry.n_spatial_dims,
                pseudo_channels=self.field_pseudo_dim,
                device=query_geometry.scalars.device,
                dtype=query_geometry.scalars.dtype,
            )
            residual = False
        else:
            residual = True
        message = self.message_scale(
            self.attention.evaluate_moments(query_geometry.cat(query_field), moments)
        )
        state = _add(query_field, message) if residual else message
        return _add(
            state,
            self.pointwise_scale(self.pointwise(query_geometry, state)),
        )

    def cross(
        self,
        source_mesh: Mesh,
        query_geometry: ScalarVectorState,
        source_geometry: ScalarVectorState,
        source_field: ScalarVectorState,
        query_field: ScalarVectorState | None = None,
        moment_segments: Sequence[slice] | None = None,
    ) -> ScalarVectorState:
        """Source-to-query pass: :meth:`build_source_moments` then
        :meth:`evaluate_cross` (fused convenience form)."""
        return self.evaluate_cross(
            query_geometry,
            self.build_source_moments(
                source_mesh,
                source_geometry,
                source_field,
                moment_segments=moment_segments,
            ),
            query_field,
        )

    def forward(
        self,
        source_mesh: Mesh,
        geometry: ScalarVectorState,
        field: ScalarVectorState,
        moment_segments: Sequence[slice] | None = None,
    ) -> ScalarVectorState:
        """Self-interaction form of :meth:`cross` (source = query set)."""
        return self.cross(
            source_mesh,
            geometry,
            geometry,
            field,
            field,
            moment_segments=moment_segments,
        )


__all__ = [
    "GeometryConditionedLinear",
    "LinearMeshFieldBlock",
    "MeshOperatorBlock",
    "NonlinearZeroMeshFieldBlock",
    "PointwiseGeometryBlock",
    "QuadraticFieldReadIn",
    "TypedRMSNorm",
    "ZeroPreservingFeedForward",
]
