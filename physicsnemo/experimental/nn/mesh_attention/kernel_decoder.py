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

r"""Kernel-basis query decoding for boundary-driven mesh transformers.

The separable moment decoder in :mod:`.attention` provably cannot produce
angular orders :math:`m\ge3` in the scalar linear-mode setting analyzed in
README section 6.2: with the centered query position as the only query-side
polar vector, every scalar output is a radial function times at most two
copies of that vector.  This module supplies the alternative query decoder
that removes the ceiling.  The boundary-to-query message becomes a genuine
pair kernel,

.. math::

   u_{hf}(x) = \sum_j \kappa_h(x, y_j)\, V_{jhf},
   \qquad
   \kappa_h(x, y_j) = \sum_m C_{mh}(\mathrm{op}_j)\, \varphi_m(x, y_j),

evaluated **densely** over query--source pairs, in query chunks.  The cost is
:math:`O(N_qN_s)` per decode -- a deliberate, documented trade against the
separable decoder's :math:`O(N_q+N_s)`; chunking bounds memory only and never
changes the operator.  A hierarchical (FMM-style) backend would be a separate
numerical backend converging to this dense oracle.

Member dictionary :math:`\varphi`
---------------------------------

- **Exact double-layer member.**  The exact integral over each boundary cell
  of the free-space double-layer singularity
  :math:`\partial G/\partial n_y` (with :math:`-\Delta G=\delta`), i.e.
  :math:`n\cdot(x-y)/(2\pi|x-y|^2)` over straight segments in 2D (signed
  subtended angle, including the :math:`\sigma=n\times\tau` orientation
  factor) and :math:`n\cdot(x-y)/(4\pi|x-y|^3)` over flat triangles in 3D
  (van Oosterom--Strackee signed solid angle).  This member's value **is**
  the cell-integrated influence with the **geometric** panel measure
  included; it is never multiplied by geometric area again. A public
  dimensionless representation/inclusion factor still multiplies this exact
  panel integral once. Exact integration matters:
  midpoint quadrature of a singular kernel produces uncontrolled
  near-boundary error that a learned kernel then mollifies away at the cost
  of operator fidelity.
- **Exact single-layer member** (``include_single_layer_member=True``).  The
  exact integral over each boundary cell of the free-space Green's function
  itself: :math:`-\log(|x-y|/L_{\mathrm{ref}})/(2\pi)` over straight
  segments in 2D and :math:`1/(4\pi|x-y|)` over flat triangles in 3D.  The
  physics motivation is completeness on multiply connected domains: a
  double-layer representation cannot carry net flux through handles -- the
  topological (winding) component of a harmonic field, e.g.
  :math:`u=a+b\log r` on an annulus, has **zero** double-layer
  representation -- and Green's representation theorem requires both layers.
  The worst benchmark tier (3D shell topology) is exactly this deficiency.
  Like the double-layer member, the value is the exact cell-integrated
  influence with geometric panel measure included, then receives any public
  dimensionless representation/inclusion factor exactly once. Singular
  kernels still use singular quadrature. Unlike the double-layer member, the
  single layer is orientation independent (no :math:`\sigma` factor; it
  never reads the cell normal).
- **Monopole-free single layer** (``monopole_free_single_layer=True``;
  requires the single-layer member).  A measure-weighted rank-one deflation
  of the single-layer member column,

  .. math::

     \tilde\varphi_{\mathrm{SL}}(x, y_j)
     = \varphi_{\mathrm{SL}}(x, y_j)
     - \frac{w_j}{\sum_k w_k}\sum_k \varphi_{\mathrm{SL}}(x, y_k),

  which is algebraically identical to projecting the *effective* conditioned
  single-layer charge density :math:`\rho_j = C_{\mathrm{SL},jh}V_{jhf}` onto
  the zero-net-charge subspace :math:`\sum_j w_j\rho_j = 0` for every head
  and value channel simultaneously (the correction couples through the two
  sums, so deflating the member column applies the exact density projection
  without touching the factored coefficient/value tensors).  Each deflated
  member is the potential of a unit-density panel minus the same total
  charge spread uniformly over the whole boundary -- exactly zero net
  monopole per member, so the single layer's monopole tail
  (:math:`\log r` in 2D, :math:`1/r` in 3D) is dead *by construction*, for
  any conditioned coefficients and values; the leading survivor is the
  zero-mean (dipole-and-up) content, one full order down.  A pleasant
  corollary in 2D: the additive log-scale gauge
  :math:`\log(\lVert x-y\rVert/L_{\mathrm{ref}})` cancels in the deflation,
  so the deflated member is exactly similarity *invariant* (the raw 2D
  single layer is only similarity covariant).  PHYSICS LICENSE: for the
  exterior disturbance field of a closed body with no net flux the monopole
  (net source strength) vanishes identically, so the projection removes
  only unphysical directions.  It is NOT licensed -- and must stay opt-in,
  default off -- for problems with genuine net flux: screened problems,
  volumetric sources or sinks, heat flux from a hot body (Neumann data with
  nonzero mean), and multiply connected domains where the single-layer
  member exists precisely to carry net flux through handles (the
  :math:`u=a+b\log r` winding component on an annulus is exactly what this
  knob kills).  History (far-field ladder, iteration 30): the trained
  singpair arms fit the annulus by near-cancellation of the single-layer
  :math:`\log r` tail against other O(1) pieces (measured state norm 2.4 at
  r=12 while the exact field decays like :math:`r^{-2}`), and the
  cancellation degrades far out; this knob replaces fitted cancellation
  with structural absence.
- **Smooth members.**  Low-order polynomial members :math:`\{1,\ b,\ a\}` of
  the pair invariants (for interpretability) plus a small SiLU MLP of all
  pair invariants (the general angular content).  Smooth members are
  evaluated at cell centroids and multiplied by the cell measure
  :math:`w_j`: midpoint quadrature is consistent for smooth integrands, so
  only the singular member needs the exact closed form.  Both smooth
  families are ablation knobs: ``mlp_members=0`` removes the learned
  members and ``include_polynomial_members=False`` removes the polynomial
  members; with both off the dictionary is the exact singular member(s)
  alone (one, or two with ``include_single_layer_member=True``).
- **Auxiliary-scale invariants for the smooth members**
  (``auxiliary_scale=True``).  Some problems carry a second physical length
  scale the reference-length gauge cannot see.  Motivating measurement
  (AirFRANS, steady RANS at :math:`\mathrm{Re}\sim4\times10^6`): the
  velocity error concentrates at the wall -- 49% of the MSE inside
  :math:`d/c<10^{-4}` -- because the boundary layer lives at
  :math:`\delta/c\sim\mathrm{Re}^{-1/2}\approx5\times10^{-4}`, a scale the
  kernels previously saw only through per-source scalar conditioning while
  every pair-radial feature was chord-scale.  The fix is a per-problem
  CONTRACT, not a learned feature: the caller declares a
  similarity-covariant auxiliary length scale
  :math:`\delta=\lambda\,L_{\mathrm{ref}}` through the dimensionless
  per-case global input :math:`\lambda` (e.g. :math:`\mathrm{Re}^{-1/2}`),
  and the pair MLP additionally receives the same normalized-frame
  invariants rescaled to the :math:`\delta` gauge --
  :math:`a/\lambda^2`, :math:`b/\lambda`, and :math:`v_c\cdot r/\lambda`
  -- appended after the base block.  ONLY the learned smooth members read
  the auxiliary block; the exact singular members and the polynomial
  members are untouched (hence ``mlp_members>0`` is required: without MLP
  members the declared scale has no carrier).  Because :math:`r` is
  already :math:`L_{\mathrm{ref}}`-normalized and :math:`\lambda` is a
  declared dimensionless input, the auxiliary invariants are similarity
  invariant exactly like the base ones, and scalar division changes no
  transformation law (no new typed content, no parity change).
- **Log-radial pair features for the smooth members**
  (``log_radial_features=True``).  Any power-law auxiliary length scale
  :math:`\delta=L_{\mathrm{ref}}\,\Pi^\alpha` built from a dimensionless
  group :math:`\Pi` is LINEAR in log space:
  :math:`\ln(r/\delta)=\ln r-\alpha\ln\Pi`.  The smooth members already
  receive dimensionless-group conditioning through the operator-scalar
  coefficient map (e.g. :math:`\ln\mathrm{Re}` on AirFRANS); this knob
  additionally hands the pair MLP the radial coordinate in the same log
  space -- ``ln(a + eps)`` (with :math:`a=\lVert r\rVert^2`, so
  :math:`\ln a=2\ln\lVert r\rVert` linearizes every radial power law) --
  together with the scale-free normalized alignments ``b / sqrt(a + eps)``
  (bounded, cosine-like) and ``v_c . r / sqrt(a + eps)``, which decouple
  angular content from radial magnitude so the MLP need not disentangle
  them multiplicatively.  ANY power-law scale thereby becomes learnable:
  no declared exponent, no semantically named input, PDE-general.
  HISTORY (H4 -> H4-L, the pre-registered V4 follow-up): the
  declared-viscous-scale contract above (``auxiliary_scale`` with
  :math:`\lambda=\mathrm{Re}^{-1/2}`) was REFUTED by its own capacity
  control -- plain MLP members beat the declared arm ~6x on AirFRANS
  velocity -- and the declaration self-sabotaged numerically by feeding
  the MLP invariants divided by :math:`\lambda\sim5\times10^{-4}`
  (features scaled by ~4e6).  The refuted knob stays runnable; this knob
  keeps the winning members and makes the scale a learnable exponent in
  log space instead of a declared ratio.  Parity and similarity: all
  three features are similarity invariant in the normalized frame exactly
  like their parents (elementwise ``log`` and division by the even,
  positive ``sqrt(a + eps)`` change no transformation law) and
  parity-typed identically -- ``ln(a + eps)`` is even like :math:`a`;
  the normalized alignments are odd in the normal and in :math:`v_c`
  exactly like :math:`b` and :math:`v_c\cdot r`.  ``eps`` is the fixed
  tiny constant :data:`_LOG_RADIAL_EPS`: :math:`a` is a normalized-frame
  gauge quantity of order one, and a query has :math:`a=0` only at
  measure-zero coincidence with a source centroid (on-panel queries are
  otherwise finite), so the floor only guards the log there.  Only the
  learned smooth members read the block (``mlp_members > 0`` required);
  it composes freely with ``auxiliary_scale`` as an independent appended
  feature block (the MLP input widths add).

Declared boundary-trace self-entries
------------------------------------

For boundary-to-boundary tasks the caller may declare, per decode call
(``self_indices`` in :meth:`KernelBasisCrossDecoder.forward`), that query
``i`` lies ON source panel ``self_indices[i]``.  The exact double-layer
member is discontinuous across its own panel -- the jump relation of
potential theory -- and evaluated exactly on the panel its closed form
returns an accidental signed-zero branch of ``atan2`` rather than the
principal value; with the declaration, the own-panel entries are replaced
by the exact one-sided limit of the declared trace side
(:func:`exterior_trace_self_entries` carries the full jump-relation
analysis, including why the single-layer member needs no value
correction).  The model-level declaration is
:class:`~physicsnemo.experimental.nn.mesh_attention.model.MeshTransformer`'s
``trace_of``.  The default ``self_indices=None`` is bitwise identical to
the pre-extension decoder.

The coefficients :math:`C_{mh}` are a linear map of each source token's
operator-state invariants (scalars plus vector Gram invariants), so the
globally encoded boundary conditions the kernel while the kernel never reads
absolute positions or orientations.  All pair features are joint O(D)
invariants of :math:`\{x-y,\ n_y,\ \text{source-state vectors}\}`; evaluated
in the model's normalized frame they are also translation invariant and
similarity covariant.

Field-mode classes
------------------

Mirroring the two field-block classes, the two linearity disciplines are
separate Python classes rather than flags:

- :class:`LinearKernelBasisCrossDecoder`: the kernel reads only the operator
  state; values are a bias-free linear projection of the drive state.  The
  message is exactly linear in the drive at fixed geometry.
- :class:`NonlinearZeroKernelBasisCrossDecoder`: the kernel may additionally
  read drive-state invariants (making the map nonlinear in the drive), but
  values remain bias-free and drive-linear-or-higher, so zero drive still
  produces an exactly zero message.

Each query row of the dense evaluation depends only on that query point and
the cached source quantities, so query points never interact and the decoded
value at a point does not depend on which other points are requested.  The
strength of that independence is a SPLIT contract (2026-07-12):

- **Closed forms are bitwise.**  The exact double/single-layer members,
  their jump relations, and the pair invariants use broadcast
  multiply-plus-fixed-axis-sum contractions whose per-row reduction order
  ignores the batch shape, so their query-set independence is exact to the
  bit and physically licensed.
- **Learned member MLPs are tolerance-level.**  Their
  :class:`_RowStableLinear` layers run plain GEMMs, whose per-row
  rounding may vary with the chunk shape at the fp-reorder scale; these are
  learned smooth functions with no analytic identity to preserve.  The
  bitwise per-channel reference contraction survives only as the test
  suite's arbiter (:meth:`_RowStableLinear.reference_forward`); it defines
  the reference accumulation order that the tolerance tests compare
  against and is not reachable from any model configuration (ratified
  2026-07-13, demoting the earlier user-facing reference mode).

Query chunking remains a pure memory control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import (
    MEASURE_WEIGHTS_KEY,
    cell_measure_weights,
)

from .attention import (
    ScalarVectorState,
    TypedProjection,
    _gram_invariants,
    _pseudo_pair_invariants,
)
from .block import StateLayerScale

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Additive floor inside the log-radial features' ``log`` and ``sqrt``
#: arguments.  ``a`` lives in the model's normalized frame (gauge units of
#: order one), so a fixed tiny constant -- representable in both fp32 and
#: fp64 -- is scale-appropriate; it matters only at the measure-zero
#: coincidence of a query point with a source centroid, where it guards the
#: log without perturbing any resolvable pair.
_LOG_RADIAL_EPS = 1.0e-24

#: Boundary cell vertex counts admitting an exact double-layer member.
_EXACT_MEMBER_VERTICES = {2: 2, 3: 3}

#: Rank-block size for the Barnes--Hut lazy segment folds (power of two).
#: Peak decode memory scales as ``n_queries * _BH_PAIR_BLOCK * heads *
#: value_channels``; the pair count only sets how many sequential blocks
#: run.  Must stay a fixed constant: the fold shape is part of the bitwise
#: query-set-independence contract (see ``barnes_hut.segment_sum_by_query``).
_BH_PAIR_BLOCK = 32

#: Local-corrector probe modes (task #53).  "windowed": local scalars enter
#: every pair, weighted by the smooth full-support window theta/(1+theta).
#: "near_only": identical channels weighted by a C^1 smoothstep that is
#: EXACTLY zero for theta <= near_theta/2 (compact near-field support).
#: "global_control": identical channel WIDTH carrying only per-sample
#: measure-weighted pooled scalars (zero per-pair locality) -- the sharing
#: control at matched parameter count.
_LOCAL_PAIR_FEATURE_MODES = {"windowed", "near_only", "global_control"}

#: Probe block width: [window, w*l_q, w*k_q, w*l_s, w*k_s] for the local
#: modes; [pool_l_s, pool_k_s, 0, 0, 0] broadcast for the global control.
_LOCAL_PAIR_FEATURE_WIDTH = 5


class _RowStableLinear(nn.Module):
    r"""The member-MLP linear map, plus the test suite's reference arbiter.

    The runtime path is a plain GEMM.  Per the split contract (2026-07-12,
    logged in the program notebook): the decoder's *closed forms* keep
    bitwise query-set independence -- their exactness is physically
    licensed -- while the *learned member MLPs* served by this layer hold
    it only to fp-reorder tolerance, because GEMM accumulation order can
    change with the number of rows.

    :meth:`reference_forward` is the TEST-INTERNAL reference
    implementation (ratified 2026-07-13): a per-channel multiply-plus-
    fixed-axis-sum whose per-row reduction order is bitwise independent of
    the batch shape.  It defines the reference accumulation order that the
    chunk-tolerance test measures the GEMM against, and doubles as the
    debugging arbiter for suspected chunk-sensitivity bugs (rerun under
    ``reference_forward``: any remaining drift is a real bug, not
    reorder noise).  It re-reads its input once per output unit (measured:
    ~26 s and ~48x redundant memory traffic per training step at DrivAerML
    scale, 2026-07-11 decode profile) and is deliberately NOT reachable
    from any model configuration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        """Allocate the ``(out_features, in_features)`` weight (Kaiming-style
        scale) and optional bias shared by both forward paths."""
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(
        self, features: Float[torch.Tensor, "*batch in_features"]
    ) -> Float[torch.Tensor, "*batch out_features"]:
        """Runtime path: one GEMM (fp-reorder tolerance across batch shapes;
        see the class docstring for the split contract)."""
        # The bias is added OUTSIDE the GEMM, mirroring reference_forward,
        # so the two paths differ only in the contraction kernel.
        output = features @ self.weight.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def reference_forward(
        self, features: Float[torch.Tensor, "*batch in_features"]
    ) -> Float[torch.Tensor, "*batch out_features"]:
        """Test-internal arbiter: bitwise batch-shape-independent contraction."""
        columns = [(features * row).sum(dim=-1) for row in self.weight.unbind(dim=0)]
        output = torch.stack(columns, dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output


@dataclass(frozen=True)
class PairInvariantFeatures:
    r"""Joint O(D) invariants of a query chunk against every boundary source.

    For normalized query points :math:`x`, source centroids :math:`y`, source
    normals :math:`n`, and per-source state vectors :math:`v_c`, the stored
    invariants of the displacement :math:`r=x-y` are

    .. math::

       a = \lVert r\rVert^2,\qquad
       b = n\cdot r,\qquad
       v_c\cdot r .

    No absolute position, orientation, or Cartesian component appears:
    rotating or reflecting all geometry and vectors together leaves every
    entry unchanged, and translations cancel in :math:`r`.

    With a declared auxiliary length scale (``auxiliary_scale`` passed to
    :meth:`compute`, the dimensionless per-case ratio
    :math:`\lambda=\delta/L_{\mathrm{ref}}`), the same invariants of the
    SAME displacement are additionally stored rescaled to the
    :math:`\delta` gauge,

    .. math::

       a/\lambda^2,\qquad b/\lambda,\qquad v_c\cdot r/\lambda ,

    so radial structure at the declared auxiliary scale (e.g. a boundary
    layer at :math:`\delta=\mathrm{Re}^{-1/2}L_{\mathrm{ref}}`) appears at
    order one to the smooth-member MLP.  Because :math:`r` is already
    normalized by the reference length and :math:`\lambda` is a declared
    dimensionless input, the auxiliary block is similarity invariant
    exactly like the base block, and scalar division preserves every
    transformation law (no new typed content).  At :math:`\lambda=1` the
    auxiliary block equals the base block bitwise.

    With ``log_radial=True`` passed to :meth:`compute`, a log-radial block
    of the SAME invariants is additionally stored,

    .. math::

       \ln(a+\epsilon),\qquad
       b/\sqrt{a+\epsilon},\qquad
       v_c\cdot r/\sqrt{a+\epsilon},

    with :math:`\epsilon` the fixed floor :data:`_LOG_RADIAL_EPS`.
    :math:`\ln a=2\ln\lVert r\rVert` makes every radial power law -- and
    hence any power-law auxiliary scale,
    :math:`\ln(r/(L_{\mathrm{ref}}\Pi^\alpha))=\ln r-\alpha\ln\Pi` for a
    dimensionless group :math:`\Pi` -- a LINEAR function of features the
    smooth-member MLP sees, while the normalized alignments carry the
    angular content scale-free (bounded: :math:`|b|\le\lVert r\rVert` for
    a unit normal), so the MLP need not disentangle angle from radius
    multiplicatively.  Parity typing follows the parents exactly:
    :math:`\ln(a+\epsilon)` is even like :math:`a`; the normalized
    alignments are odd in the normal and in :math:`v_c` exactly like
    :math:`b` and :math:`v_c\cdot r` (division by the even, positive
    :math:`\sqrt{a+\epsilon}` changes no transformation law), and every
    entry is similarity invariant in the normalized frame like the base
    block.
    """

    squared_distance: Float[torch.Tensor, "q s"]
    normal_alignment: Float[torch.Tensor, "q s"]
    vector_alignments: Float[torch.Tensor, "q s channels"]
    # Auxiliary-scale invariant block; ``None`` unless a declared auxiliary
    # scale was supplied to ``compute``.
    auxiliary_squared_distance: Float[torch.Tensor, "q s"] | None = None
    auxiliary_normal_alignment: Float[torch.Tensor, "q s"] | None = None
    auxiliary_vector_alignments: Float[torch.Tensor, "q s channels"] | None = None
    # Log-radial feature block; ``None`` unless ``compute`` was asked for it
    # (``log_radial=True``), which keeps older callers bitwise untouched.
    log_squared_distance: Float[torch.Tensor, "q s"] | None = None
    normalized_normal_alignment: Float[torch.Tensor, "q s"] | None = None
    normalized_vector_alignments: Float[torch.Tensor, "q s channels"] | None = None

    @classmethod
    def compute(
        cls,
        query_points: Float[torch.Tensor, "q spatial_dims"],
        source_centroids: Float[torch.Tensor, "s spatial_dims"],
        source_normals: Float[torch.Tensor, "s spatial_dims"],
        source_vectors: Float[torch.Tensor, "s channels spatial_dims"],
        auxiliary_scale: Float[torch.Tensor, ""] | None = None,
        log_radial: bool = False,
    ) -> "PairInvariantFeatures":
        """Build the per-pair invariants for one query chunk."""
        if query_points.ndim != 2 or source_centroids.ndim != 2:
            raise ValueError("query_points and source_centroids must be (N, D)")
        if query_points.shape[-1] != source_centroids.shape[-1]:
            raise ValueError("query and source spatial dimensions differ")
        if source_normals.shape != source_centroids.shape:
            raise ValueError("source_normals must match source_centroids shape")
        if (
            source_vectors.ndim != 3
            or source_vectors.shape[0] != (source_centroids.shape[0])
        ):
            raise ValueError("source_vectors must have shape (S, C, D)")
        if auxiliary_scale is not None and auxiliary_scale.ndim != 0:
            raise ValueError("auxiliary_scale must be a 0-dimensional tensor")
        displacement = query_points[:, None, :] - source_centroids[None, :, :]
        squared_distance = displacement.square().sum(dim=-1)
        # Broadcast multiply-plus-fixed-axis-sum for every query-axis
        # contraction, NEVER einsum: ``einsum("qsd,scd->qsc", ...)`` lowers
        # to a batched GEMM and ``einsum("qsd,sd->qs", ...)`` to a batched
        # cuBLAS gemv, and both pick reduction internals from the batch
        # shape (measured: Q=1 vs Q=40 rows differ at 1 ulp on CUDA),
        # breaking the decoder's bitwise query-set-independence contract.
        normal_alignment = (displacement * source_normals[None, :, :]).sum(dim=-1)
        vector_alignments = (
            displacement[:, :, None, :] * source_vectors[None, :, :, :]
        ).sum(dim=-1)
        auxiliary_kwargs = {}
        if auxiliary_scale is not None:
            # Elementwise division by a scalar keeps every query row's
            # arithmetic independent of the batch shape, so the bitwise
            # query-set-independence contract is untouched.
            auxiliary_kwargs = dict(
                auxiliary_squared_distance=squared_distance / auxiliary_scale.square(),
                auxiliary_normal_alignment=normal_alignment / auxiliary_scale,
                auxiliary_vector_alignments=vector_alignments / auxiliary_scale,
            )
        log_radial_kwargs = {}
        if log_radial:
            # Elementwise per-pair maps of the invariants computed above: no
            # batch-shape-dependent reduction is introduced, so the bitwise
            # query-set-independence contract is untouched, and negating a
            # parent (normal or state vector) negates its normalized child
            # exactly (IEEE negation and division are sign-exact).
            radius = torch.sqrt(squared_distance + _LOG_RADIAL_EPS)
            log_radial_kwargs = dict(
                log_squared_distance=torch.log(squared_distance + _LOG_RADIAL_EPS),
                normalized_normal_alignment=normal_alignment / radius,
                normalized_vector_alignments=vector_alignments / radius.unsqueeze(-1),
            )
        return cls(
            squared_distance=squared_distance,
            normal_alignment=normal_alignment,
            vector_alignments=vector_alignments,
            **auxiliary_kwargs,
            **log_radial_kwargs,
        )

    def stacked(self) -> Float[torch.Tensor, "q s features"]:
        """Return all invariants stacked as ``(Q, S, F)`` features.

        The base block contributes ``2 + C`` features; the auxiliary block
        (when present) appends another ``2 + C`` AFTER it, and the
        log-radial block (when present) appends a final ``2 + C`` after
        every other block, each in the same ``(a, b, alignments)`` order.
        """
        parts = [
            self.squared_distance.unsqueeze(-1),
            self.normal_alignment.unsqueeze(-1),
            self.vector_alignments,
        ]
        if self.auxiliary_squared_distance is not None:
            parts.extend(
                (
                    self.auxiliary_squared_distance.unsqueeze(-1),
                    self.auxiliary_normal_alignment.unsqueeze(-1),
                    self.auxiliary_vector_alignments,
                )
            )
        if self.log_squared_distance is not None:
            parts.extend(
                (
                    self.log_squared_distance.unsqueeze(-1),
                    self.normalized_normal_alignment.unsqueeze(-1),
                    self.normalized_vector_alignments,
                )
            )
        return torch.cat(parts, dim=-1)


def subtended_angle(
    squared_distance: Float[torch.Tensor, "q s"],
    cell_measures: Float[torch.Tensor, " s"],
    n_manifold_dims: int,
) -> Float[torch.Tensor, "q s"]:
    r"""Apparent subtended angle :math:`\theta \approx h/d` of each source panel.

    ``squared_distance`` is the pairwise :math:`\lVert x_q - y_s\rVert^2`
    with shape ``(Q, S)``; ``cell_measures`` the per-source cell measures
    (length of a segment, area of a triangle) with shape ``(S,)``.  The
    panel's linear size is derived DIMENSION-GENERICALLY from its measure,

    .. math:: h_s = \mu_s^{1/m},\qquad
              \theta_{qs} = h_s\,/\,\sqrt{a_{qs}+\epsilon},

    with :math:`m` the manifold dimension (1 for boundary curves in 2D, 2
    for boundary surfaces in 3D) -- never an area divided by a distance.
    Both :math:`h` and :math:`d` are lengths in the same frame, so
    :math:`\theta` is dimensionless and invariant under uniform coordinate
    scaling (scaling points by :math:`s` scales :math:`\mu` by
    :math:`s^m`, hence :math:`h` and :math:`d` both by :math:`s`); it is
    trivially invariant under rotations, reflections, and translations.
    This is the Barnes-Hut multipole acceptance criterion: smooth far-field
    representations err as powers of :math:`h/d`, so "near field" defined
    as a :math:`\theta` threshold is exactly "where smooth approximations
    fail," adaptively in the local mesh resolution.

    Every operation is an elementwise per-pair map (no reduction), so the
    decoder's query-set-independence contract is untouched.
    """
    if n_manifold_dims < 1:
        raise ValueError(
            f"n_manifold_dims must be a positive integer, got {n_manifold_dims}"
        )
    if cell_measures.ndim != 1 or squared_distance.shape[-1] != cell_measures.shape[0]:
        raise ValueError(
            "cell_measures must be (S,) matching squared_distance's source "
            f"axis; got {tuple(cell_measures.shape)} against "
            f"{tuple(squared_distance.shape)}"
        )
    panel_size = cell_measures.clamp_min(0.0).pow(1.0 / n_manifold_dims)
    distance = torch.sqrt(squared_distance + _LOG_RADIAL_EPS)
    return panel_size[None, :] / distance


def _segment_double_layer_member(
    query_points: Float[torch.Tensor, "q 2"],
    panel_vertices: Float[torch.Tensor, "s 2 2"],
    cell_normals: Float[torch.Tensor, "s 2"],
) -> Float[torch.Tensor, "q s"]:
    r"""Exact straight-segment integrals of the 2D double-layer singularity.

    Entry ``(i, j)`` is :math:`\int_{P_j} n\cdot(x_i-y)\,/\,
    (2\pi\lVert x_i-y\rVert^2)\,ds_y`, evaluated in closed form as the signed
    subtended angle.  The orientation factor :math:`\sigma=n\times\tau=\pm1`
    makes the value odd in the supplied normal, as the double layer requires,
    independently of the panel's vertex ordering.  The boundary measure is
    included; at an interior point of a closed outward-oriented boundary the
    entries sum to exactly :math:`-1` (Gauss).
    """
    start_vector = panel_vertices[None, :, 0, :] - query_points[:, None, :]
    end_vector = panel_vertices[None, :, 1, :] - query_points[:, None, :]
    cross = (
        start_vector[..., 0] * end_vector[..., 1]
        - start_vector[..., 1] * end_vector[..., 0]
    )
    dot = torch.sum(start_vector * end_vector, dim=-1)
    tangent = panel_vertices[:, 1, :] - panel_vertices[:, 0, :]
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    sigma = cell_normals[:, 0] * tangent[:, 1] - cell_normals[:, 1] * tangent[:, 0]
    return -sigma[None, :] * torch.atan2(cross, dot) / _TWO_PI


def _triangle_double_layer_member(
    query_points: Float[torch.Tensor, "q 3"],
    panel_vertices: Float[torch.Tensor, "s 3 3"],
    cell_normals: Float[torch.Tensor, "s 3"],
) -> Float[torch.Tensor, "q s"]:
    r"""Exact flat-triangle integrals of the 3D double-layer singularity.

    Entry ``(i, j)`` is :math:`\int_{T_j} n\cdot(x_i-y)\,/\,
    (4\pi\lVert x_i-y\rVert^3)\,dS_y`, the van Oosterom--Strackee signed
    solid angle divided by :math:`-4\pi`.  The vertex winding fixes the
    orientation of the closed form; the sign factor against the supplied cell
    normals keeps the member odd in the normal even if a mesh ever carries
    normals that disagree with its winding.  The boundary measure is
    included; at an interior point of a closed outward-oriented boundary the
    entries sum to exactly :math:`-1` (Gauss).
    """
    a = panel_vertices[None, :, 0, :] - query_points[:, None, :]
    b = panel_vertices[None, :, 1, :] - query_points[:, None, :]
    c = panel_vertices[None, :, 2, :] - query_points[:, None, :]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    # Elementwise multiply-plus-fixed-axis-sum, NOT einsum("qsd,qsd->qs"):
    # that einsum lowers to a batch=Q*S bmm of (1,3)@(3,1) items, which
    # cuBLAS decomposes into one gemv launch per PAIR -- measured as the
    # dominant kernel-launch count of every decode step (2026-07-11
    # profile: ~10^5 launches, ~7.4 s/step at 20k sources).  The mul+sum
    # form computes the same 3-term dot per pair in two dense kernels and
    # keeps the per-row reduction independent of the batch shape (the
    # decoder's bitwise query-set-independence contract).
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = (
        la * lb * lc
        + (a * b).sum(dim=-1) * lc
        + (b * c).sum(dim=-1) * la
        + (c * a).sum(dim=-1) * lb
    )
    winding_normal = torch.cross(
        panel_vertices[:, 1, :] - panel_vertices[:, 0, :],
        panel_vertices[:, 2, :] - panel_vertices[:, 0, :],
        dim=-1,
    )
    sigma = torch.sign(torch.einsum("sd,sd->s", winding_normal, cell_normals))
    return -sigma[None, :] * 2.0 * torch.atan2(numerator, denominator) / _FOUR_PI


def exact_double_layer_member(
    query_points: Float[torch.Tensor, "q spatial_dims"],
    panel_vertices: Float[torch.Tensor, "s vertices spatial_dims"],
    cell_normals: Float[torch.Tensor, "s spatial_dims"],
) -> Float[torch.Tensor, "q s"]:
    r"""Dimension-dispatched exact cell integrals of the double layer.

    ``panel_vertices`` has shape ``(S, 2, 2)`` for straight 2D segments or
    ``(S, 3, 3)`` for flat 3D triangles.  Values are similarity invariant
    (angles) and translation invariant, and geometric panel measure is
    included -- do not multiply the result by geometric area again. A
    dimensionless public measure factor may still be applied once.
    """
    n_dims = query_points.shape[-1]
    expected = _EXACT_MEMBER_VERTICES.get(n_dims)
    if expected is None:
        raise ValueError(
            "the exact double-layer member is implemented for 2D segments and "
            f"3D triangles only, got {n_dims} spatial dimensions"
        )
    if panel_vertices.ndim != 3 or panel_vertices.shape[1] != expected:
        raise ValueError(
            f"panel_vertices must have shape (S, {expected}, {n_dims}) in "
            f"{n_dims}D, got {tuple(panel_vertices.shape)}"
        )
    if n_dims == 2:
        return _segment_double_layer_member(query_points, panel_vertices, cell_normals)
    return _triangle_double_layer_member(query_points, panel_vertices, cell_normals)


def _segment_single_layer_member(
    query_points: Float[torch.Tensor, "q 2"],
    panel_vertices: Float[torch.Tensor, "s 2 2"],
) -> Float[torch.Tensor, "q s"]:
    r"""Exact straight-segment integrals of the 2D single-layer kernel.

    Entry ``(i, j)`` is :math:`\int_{P_j} -\log\lVert x_i-y\rVert
    /(2\pi)\,ds_y`, the potential of a unit-density source panel, in the
    classical panel-method closed form (e.g. Katz & Plotkin, *Low-Speed
    Aerodynamics*, constant-strength source panel; Hess & Smith 1967).  In
    local panel coordinates with tangential query offsets
    :math:`\xi_{1,2}` from the two endpoints, perpendicular offset
    :math:`\eta`, and endpoint distances :math:`r_{1,2}`,

    .. math::

       \int_P \log\lVert x-y\rVert\,ds_y
       = \xi_1\log r_1 - \xi_2\log r_2 - \ell
         + \eta\,(\theta_2-\theta_1),
       \qquad \theta_k=\operatorname{atan2}(\eta,\xi_k).

    The value is finite for every query point, including on the panel and on
    its extension (the endpoint logs are clamped so :math:`0\cdot\log 0`
    limits evaluate to the exact finite integral).  In contrast to the
    double-layer member there is **no** :math:`\sigma=n\times\tau`
    orientation factor: the single layer is even in the normal and never
    reads it, so vertex ordering and normal orientation are irrelevant by
    construction.

    The logarithm's argument is the distance in the decoder's normalized
    frame, where coordinates are already divided by the model reference
    length; the additive log-scale gauge is therefore fixed as
    :math:`\log(\lVert x-y\rVert/L_{\mathrm{ref}})` and the member is
    dimensionless. Geometric panel measure is included -- do not multiply by
    geometric area again. A dimensionless public measure factor may still be
    applied once.
    """
    start_vector = panel_vertices[None, :, 0, :] - query_points[:, None, :]
    end_vector = panel_vertices[None, :, 1, :] - query_points[:, None, :]
    edge = panel_vertices[:, 1, :] - panel_vertices[:, 0, :]
    length = edge.norm(dim=-1)
    tangent = edge / length[:, None]
    # Local coordinates of the query point relative to the panel: xi_k is the
    # tangential offset from endpoint k and eta the perpendicular offset (its
    # sign cancels in eta * (theta_2 - theta_1), so no orientation enters).
    # Mul+sum, not einsum("qsd,sd->qs"): the einsum's batched-gemv lowering
    # is batch-shape dependent at 1 ulp on CUDA (query-set independence).
    xi_start = -(start_vector * tangent[None, :, :]).sum(dim=-1)
    xi_end = -(end_vector * tangent[None, :, :]).sum(dim=-1)
    eta = (
        tangent[None, :, 1] * start_vector[..., 0]
        - tangent[None, :, 0] * start_vector[..., 1]
    )
    tiny = torch.finfo(query_points.dtype).tiny
    log_start = start_vector.norm(dim=-1).clamp_min(tiny).log()
    log_end = end_vector.norm(dim=-1).clamp_min(tiny).log()
    subtended = torch.atan2(eta, xi_end) - torch.atan2(eta, xi_start)
    integral = (
        xi_start * log_start - xi_end * log_end - length[None, :] + eta * subtended
    )
    return -integral / _TWO_PI


def _triangle_single_layer_member(
    query_points: Float[torch.Tensor, "q 3"],
    panel_vertices: Float[torch.Tensor, "s 3 3"],
) -> Float[torch.Tensor, "q s"]:
    r"""Exact flat-triangle integrals of the 3D single-layer kernel.

    Entry ``(i, j)`` is :math:`\int_{T_j} 1/(4\pi\lVert x_i-y\rVert)\,dS_y`,
    the potential of a unit-density source triangle, via the classical
    per-edge closed form (Hess & Smith 1967; Newman 1986, *Distributions of
    sources and normal dipoles over a quadrilateral panel*, specialized to
    triangles; equivalently Wilton et al. 1984):

    .. math::

       \int_T \frac{dS_y}{\lVert x-y\rVert}
       = \sum_{\text{edges}} t_e\,
         \bigl[\operatorname{asinh}(s_e^+/\mu_e)
              -\operatorname{asinh}(s_e^-/\mu_e)\bigr]
       - |h|\,\Omega,

    where, per edge, :math:`s_e^\pm` are the tangential offsets of the query
    point from the edge endpoints, :math:`\mu_e` its distance to the edge
    line, and :math:`t_e` the in-plane signed distance from its projection to
    the edge; :math:`h` is the height above the triangle plane and
    :math:`\Omega\in[0,2\pi]` the unsigned van Oosterom--Strackee solid
    angle.  The ``asinh`` difference is the numerically stable form of the
    textbook edge log :math:`\ln((r_a+r_b+\ell)/(r_a+r_b-\ell))`: it suffers
    no cancellation for far or on-extension queries, and the
    :math:`t_e\log(1/\mu_e)` limits vanish because
    :math:`\mu_e\ge|t_e|`.  The value is finite for every query point,
    including on the triangle.

    In contrast to the double-layer member there is **no** orientation
    factor: only :math:`|h|` and :math:`|\Omega|` appear and the in-plane
    edge distances :math:`t_e` are paired with the winding normal
    consistently, so the member never reads the cell normal and vertex
    winding is irrelevant by construction. Geometric panel measure is
    included -- do not multiply by geometric area again. A dimensionless
    public measure factor may still be applied once.
    """
    a = panel_vertices[None, :, 0, :] - query_points[:, None, :]
    b = panel_vertices[None, :, 1, :] - query_points[:, None, :]
    c = panel_vertices[None, :, 2, :] - query_points[:, None, :]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    winding_normal = torch.cross(
        panel_vertices[:, 1, :] - panel_vertices[:, 0, :],
        panel_vertices[:, 2, :] - panel_vertices[:, 0, :],
        dim=-1,
    )
    unit_normal = winding_normal / winding_normal.norm(dim=-1, keepdim=True)
    # Mul+sum, not einsum("qsd,sd->qs"), here and per edge below: the
    # einsum's batched-gemv lowering is batch-shape dependent at 1 ulp on
    # CUDA (measured at Q=1), breaking bitwise query-set independence and
    # bitwise parity with the pairwise Barnes--Hut near field.
    height = (a * unit_normal[None, :, :]).sum(dim=-1)
    tiny = torch.finfo(query_points.dtype).tiny
    relative = (a, b, c)
    edge_terms = query_points.new_zeros(a.shape[:2])
    for start, end in ((0, 1), (1, 2), (2, 0)):
        p = relative[start]
        q = relative[end]
        edge = panel_vertices[:, end, :] - panel_vertices[:, start, :]
        edge_tangent = edge / edge.norm(dim=-1, keepdim=True)
        s_start = (p * edge_tangent[None, :, :]).sum(dim=-1)
        s_end = (q * edge_tangent[None, :, :]).sum(dim=-1)
        # Distance from the query point to the edge *line* (includes the
        # out-of-plane component: mu^2 = t^2 + h^2 >= t^2, which is what
        # makes t * asinh(s/mu) vanish smoothly near the edge).
        mu = (p - s_start.unsqueeze(-1) * edge_tangent[None, :, :]).norm(dim=-1)
        mu = mu.clamp_min(tiny)
        in_plane_distance = (
            p * torch.cross(edge_tangent, unit_normal, dim=-1)[None, :, :]
        ).sum(dim=-1)
        edge_terms = edge_terms + in_plane_distance * (
            torch.asinh(s_end / mu) - torch.asinh(s_start / mu)
        )
    # Elementwise multiply-plus-fixed-axis-sum, NOT einsum("qsd,qsd->qs"):
    # that einsum lowers to a batch=Q*S bmm of (1,3)@(3,1) items, which
    # cuBLAS decomposes into one gemv launch per PAIR -- measured as the
    # dominant kernel-launch count of every decode step (2026-07-11
    # profile: ~10^5 launches, ~7.4 s/step at 20k sources).  The mul+sum
    # form computes the same 3-term dot per pair in two dense kernels and
    # keeps the per-row reduction independent of the batch shape (the
    # decoder's bitwise query-set-independence contract).
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = (
        la * lb * lc
        + (a * b).sum(dim=-1) * lc
        + (b * c).sum(dim=-1) * la
        + (c * a).sum(dim=-1) * lb
    )
    solid_angle = 2.0 * torch.atan2(numerator, denominator).abs()
    return (edge_terms - height.abs() * solid_angle) / _FOUR_PI


def exact_single_layer_member(
    query_points: Float[torch.Tensor, "q spatial_dims"],
    panel_vertices: Float[torch.Tensor, "s vertices spatial_dims"],
) -> Float[torch.Tensor, "q s"]:
    r"""Dimension-dispatched exact cell integrals of the single layer.

    ``panel_vertices`` has shape ``(S, 2, 2)`` for straight 2D segments or
    ``(S, 3, 3)`` for flat 3D triangles.  Entry ``(i, j)`` is the exact
    integral of the free-space Green's function (:math:`-\Delta G=\delta`)
    over cell :math:`j` evaluated at query point :math:`i`:
    :math:`-\log(\lVert x-y\rVert/L_{\mathrm{ref}})/(2\pi)` in 2D and
    :math:`1/(4\pi\lVert x-y\rVert)` in 3D, with geometric panel measure
    included -- do not multiply the result by geometric area again. A
    dimensionless public measure factor may still be applied once.

    Unlike :func:`exact_double_layer_member` there is no normal argument:
    the single layer is orientation independent (a recurring failure mode
    for the double layer was orientation sign bugs; here no orientation
    exists to get wrong).  This member is what completes the dictionary on
    multiply connected domains, where the double layer alone cannot
    represent the net-flux (winding) component of the field.

    Values are evaluated in the model's normalized frame; in 2D the
    logarithm's additive scale gauge is thereby fixed to the model reference
    length (a bare log of a dimensional distance would be a unit-bearing
    bug), and the member scales covariantly rather than invariantly under
    similarity maps.
    """
    n_dims = query_points.shape[-1]
    expected = _EXACT_MEMBER_VERTICES.get(n_dims)
    if expected is None:
        raise ValueError(
            "the exact single-layer member is implemented for 2D segments and "
            f"3D triangles only, got {n_dims} spatial dimensions"
        )
    if panel_vertices.ndim != 3 or panel_vertices.shape[1] != expected:
        raise ValueError(
            f"panel_vertices must have shape (S, {expected}, {n_dims}) in "
            f"{n_dims}D, got {tuple(panel_vertices.shape)}"
        )
    if n_dims == 2:
        return _segment_single_layer_member(query_points, panel_vertices)
    return _triangle_single_layer_member(query_points, panel_vertices)


def exterior_trace_self_entries(
    double_layer: Float[torch.Tensor, "q s"],
    self_indices: Int[torch.Tensor, " q"],
) -> Float[torch.Tensor, "q s"]:
    r"""Replace declared own-panel double-layer entries with the exterior trace.

    ``double_layer`` is a ``(Q, S)`` matrix from
    :func:`exact_double_layer_member`; ``self_indices`` is a ``(Q,)``
    ``torch.long`` tensor declaring that query ``i`` lies ON source panel
    ``self_indices[i]``.  The returned matrix equals the input everywhere
    except the declared entries ``(i, self_indices[i])``, which are set to
    exactly ``+1/2`` -- the one-sided limit of the own-panel integral from
    the side the supplied cell normal points toward.

    **The jump relation, in this decoder's normalization.**  The
    cell-integrated double layer :math:`\int_{P}\partial G/\partial n_y\,ds`
    (rows sum to :math:`-1` at interior points of a closed outward-oriented
    boundary; Gauss) is discontinuous exactly across its own panel.  For a
    FLAT panel (2D straight segment or 3D flat triangle) the on-panel
    principal value of the own-panel integral is exactly zero
    (:math:`n\cdot(x-y)\equiv 0` for :math:`x, y` in the panel's own
    plane), and the two one-sided limits are exactly :math:`\pm 1/2`
    independent of panel shape, size, and ambient dimension: the panel
    subtends the half angle :math:`\pi` of :math:`2\pi` (2D) or the half
    solid angle :math:`2\pi` of :math:`4\pi` (3D) from a point approaching
    its own interior, with the sign :math:`+1/2` on the side the supplied
    normal points toward (:math:`n\cdot(x-y)>0` there).  Equivalently, by
    the Gauss bookkeeping on a closed outward boundary: the smooth-point
    principal-value row sum is :math:`-1/2`, so the own-panel exterior
    limit is :math:`0-(-1/2)=+1/2` and the interior limit
    :math:`-1-(-1/2)=-1/2`.

    **Why a replacement is needed at all.**  Evaluated exactly ON the
    panel, the closed forms do not return the principal value: the
    ``atan2`` argument pair degenerates to ``(signed zero, negative)``, so
    the member lands on an accidental :math:`\pm 1/2` branch decided by
    floating-point sign bits (measured on the DrivAerML surface task: the
    INTERIOR branch, while the physical surface values are the exterior
    trace -- the GeoTransolver-gap verdict's H-D mechanism).  No learned
    coefficient can repair a term that flips sign under infinitesimal
    displacement; the declared replacement serves the correct branch
    exactly.  Because the limit is the constant :math:`+1/2` for any flat
    panel, the replaced entries carry zero geometry gradient -- which is
    also the analytically correct sensitivity.

    **Why the single layer receives no correction.**  The single-layer
    potential is continuous across its own layer -- its VALUE has no jump
    (only its normal derivative jumps, by minus the density), and the
    exact closed forms in :func:`exact_single_layer_member` are finite on
    the panel and already evaluate the common two-sided limit, in both 2D
    and 3D.  Smooth (polynomial and MLP) members are continuous functions
    of the pair invariants and are likewise untouched.

    The replacement writes one entry per query row from that row's own
    declared index, so the bitwise query-set-independence contract is
    preserved (each row's arithmetic never reads another row).
    """
    if double_layer.ndim != 2:
        raise ValueError(
            f"double_layer must have shape (Q, S), got {tuple(double_layer.shape)}"
        )
    if self_indices.ndim != 1 or self_indices.shape[0] != double_layer.shape[0]:
        raise ValueError(
            "self_indices must have shape (Q,) matching the query count, got "
            f"{tuple(self_indices.shape)} for {double_layer.shape[0]} queries"
        )
    if self_indices.dtype != torch.long:
        raise ValueError(
            f"self_indices must be a torch.long index tensor, got {self_indices.dtype}"
        )
    rows = torch.arange(double_layer.shape[0], device=double_layer.device)
    return double_layer.index_put((rows, self_indices), double_layer.new_full((), 0.5))


@dataclass(frozen=True)
class KernelDecoderCache:
    r"""Source-side quantities cached by ``encode`` for kernel decoding.

    Everything is expressed in the model's normalized frame.  The cache is
    query independent: it contains per-source geometry (cell vertices,
    centroids, normals, geometric panel extent), the dimensionless public
    measure factors and their resulting effective quadrature measure, the
    per-source kernel coefficients :math:`C_{jmh}`, the state vectors whose
    pair alignments feed the smooth members, and the projected drive values.
    """

    panel_vertices: Float[torch.Tensor, "s vertices spatial_dims"]
    centroids: Float[torch.Tensor, "s spatial_dims"]
    normals: Float[torch.Tensor, "s spatial_dims"]
    # Historical public field: geometric panel area. Keep both its meaning
    # and constructor position so ``weights=`` and positional calls from
    # experimental users remain behaviorally compatible.
    weights: Float[torch.Tensor, " s"]
    pair_vectors: Float[torch.Tensor, "s channels spatial_dims"]
    coefficients: Float[torch.Tensor, "s members heads"]
    value_scalars: Float[torch.Tensor, "s heads value_scalars"]
    value_vectors: Float[torch.Tensor, "s heads value_vectors spatial_dims"]
    # Pseudoscalar (0o) value features; ``None`` only for caches built before
    # the pseudo sector existed (equivalent to zero width).
    value_pseudos: Float[torch.Tensor, "s heads value_pseudos"] | None = None
    # Declared auxiliary length-scale ratio lambda = delta / L_ref (a
    # dimensionless 0-dim tensor, a physical declaration read from the raw
    # global operator input, never a learned feature); ``None`` for caches
    # built without the auxiliary-scale contract, which keeps caches from
    # older decoders valid exactly like ``value_pseudos`` does.
    auxiliary_scale: Float[torch.Tensor, ""] | None = None
    # Per-source squashed local-geometry scalars ``(S, 2)`` --
    # [log-relative-measure, nondimensional curvature], both bounded by the
    # smooth fixed maps documented in ``MeshTransformer.encode`` -- consumed
    # only by the local-corrector probe block (``local_pair_features``);
    # ``None`` keeps caches from older decoders valid.
    local_scalars: Float[torch.Tensor, "s 2"] | None = None
    # Barnes--Hut backend state (``decode_backend="barnes_hut"`` only):
    # the source cluster tree and the channel-resolved per-node far-field
    # aggregates (see ``barnes_hut.py``).  Geometry + learned-density
    # content, built once per encode; ``None`` on dense caches.
    bh_tree: "object | None" = None
    bh_aggregates: "object | None" = None
    # Added after every historical field to preserve positional construction.
    # ``None`` denotes an old/no-weight cache, where the dimensionless
    # representation factor was implicitly one.
    measure_factors: Float[torch.Tensor, " s"] | None = None

    @property
    def quadrature_measures(self) -> Float[torch.Tensor, " s"]:
        """Effective quadrature measure: panel area times public factor."""
        if self.measure_factors is None:
            return self.weights
        return self.weights * self.measure_factors

    @property
    def geometric_panel_areas(self) -> Float[torch.Tensor, " s"]:
        """Explicit alias for the historical geometric-area field."""
        return self.weights

    @property
    def panel_areas(self) -> Float[torch.Tensor, " s"]:
        """Named geometric-panel alias introduced with the measure split."""
        return self.weights

    @property
    def representation_measure_factors(self) -> Float[torch.Tensor, " s"]:
        """Dimensionless public factors (unit factors for old caches)."""
        if self.measure_factors is None:
            return torch.ones_like(self.weights)
        return self.measure_factors


class KernelBasisCrossDecoder(nn.Module):
    r"""Dense operator-conditioned pair-kernel boundary-to-query message.

    This base class owns the shared machinery; instantiate
    :class:`LinearKernelBasisCrossDecoder` or
    :class:`NonlinearZeroKernelBasisCrossDecoder`, which fix the linearity
    class through separate code paths.

    Parameters
    ----------
    n_spatial_dims : int
        Ambient dimension; 2 (segment boundaries) or 3 (triangle boundaries).
    operator_scalar_dim, operator_vector_dim : int
        Channel counts of the encoded source operator state.
    drive_scalar_dim, drive_vector_dim : int
        Channel counts of the encoded source drive state; the emitted message
        carries the same typed channel counts.
    drive_pseudo_dim : int
        Pseudoscalar (``0o``, 2D-only) channel count of the drive state; the
        message carries the same pseudo width.  Pseudo values are rotation
        invariant like scalar values, so they multiply the same invariant
        kernel and are read out through a dedicated bias-free output map that
        never mixes them with true scalars.  The default 0 is bitwise
        identical to the pre-extension decoder.
    heads : int
        Number of independent kernels; each head owns its own member
        coefficients and value channels.
    include_double_layer_member : bool
        Whether the exact double-layer member -- the dictionary's default
        base member, the exact cell-integrated free-space double-layer
        singularity -- joins the dictionary.  Default ``True`` preserves
        every existing configuration bitwise.  ``False`` (with the single
        layer also off) is the MLP-only ablation: the dictionary carries no
        exact singular structure, the thesis-critical control separating
        boundary-integral physics from generic learned pairwise capacity
        (external-review P0).  In trace mode the exterior jump correction
        rides this member, so with it off no self-entry correction is
        applied.
    include_polynomial_members : bool
        Whether the fixed polynomial smooth members :math:`\{1,\ b,\ a\}`
        join the dictionary.  ``False`` is the polynomial-off ablation;
        combined with ``mlp_members=0`` the dictionary is the exact singular
        member alone.
    include_single_layer_member : bool
        Whether the exact single-layer (monopole) member joins the exact
        double-layer member in the dictionary.  Default ``False`` preserves
        the pruned two-family dictionary bitwise; the science arm turns it
        on because a pure double-layer representation is incomplete on
        multiply connected domains (it cannot carry net flux through
        handles; see the module docstring).
    monopole_free_single_layer : bool
        Deflate the exact single-layer member column by its measure-weighted
        boundary mean (see the module docstring), so every deflated member
        carries exactly zero net charge and the single layer's monopole tail
        (:math:`\log r` in 2D, :math:`1/r` in 3D) is structurally absent for
        any conditioned coefficients.  Physically licensed for exterior
        disturbance fields with zero net flux; NOT licensed for problems
        with genuine net flux (sources, screened operators, flux through
        handles of multiply connected domains) -- hence default ``False``
        (bitwise identical to the undeflated dictionary; the knob adds no
        parameters).  Requires ``include_single_layer_member=True``.
    auxiliary_scale : bool
        Accept a declared per-problem auxiliary length scale
        :math:`\delta=\lambda\,L_{\mathrm{ref}}` (the dimensionless ratio
        :math:`\lambda` rides in :attr:`KernelDecoderCache.auxiliary_scale`)
        and append a second copy of every pair invariant rescaled to the
        :math:`\delta` gauge -- :math:`a/\lambda^2`, :math:`b/\lambda`,
        :math:`v_c\cdot r/\lambda` -- AFTER the base block of the
        smooth-member MLP input (whose width doubles accordingly).  The
        auxiliary block feeds ONLY the learned smooth members: the exact
        singular members, the polynomial members, and the coefficient map
        never read it, so the knob requires ``mlp_members > 0`` (rejected
        otherwise -- there would be no carrier).  PHYSICS LICENSE: declared
        per problem when a second physical scale exists that the gauge
        cannot see, e.g. a turbulent boundary layer at
        :math:`\delta/c\sim\mathrm{Re}^{-1/2}` (AirFRANS measured 49% of
        velocity MSE inside :math:`d/c<10^{-4}`, unreachable by chord-scale
        radial features; see the module docstring).  :math:`\lambda` is a
        physical declaration like a reference length, never a learned
        feature; similarity covariance is preserved because :math:`r` is
        already :math:`L_{\mathrm{ref}}`-normalized and :math:`\lambda` is
        dimensionless.  Default ``False`` adds no parameters and is bitwise
        identical to the pre-extension decoder (state dict and outputs).
    log_radial_features : bool
        Append the log-radial pair-feature block -- ``ln(a + eps)`` and the
        scale-free normalized alignments ``b / sqrt(a + eps)`` and
        ``v_c . r / sqrt(a + eps)``, with ``eps`` the fixed floor
        :data:`_LOG_RADIAL_EPS` -- AFTER every other block of the
        smooth-member MLP input (whose width grows by one base-block copy).
        PHYSICS LICENSE: a power-law auxiliary scale
        :math:`\delta=L_{\mathrm{ref}}\,\Pi^\alpha` is linear in log space
        (:math:`\ln(r/\delta)=\ln r-\alpha\ln\Pi`), so with the
        dimensionless-group conditioning the members already receive
        through the coefficient map, ``ln a`` makes ANY power-law scale
        learnable -- no declared exponent, no semantic naming, PDE-general.
        This is the pre-registered follow-up (H4-L / V4) to the REFUTED
        declared-scale experiment above: ``auxiliary_scale`` with
        :math:`\lambda=\mathrm{Re}^{-1/2}` lost to its own plain-member
        capacity control ~6x on AirFRANS velocity (see the module
        docstring), so the exponent becomes learnable structure instead of
        a declaration.  The block feeds ONLY the learned smooth members,
        hence ``mlp_members > 0`` is required (no carrier otherwise --
        mirroring ``auxiliary_scale``); it composes freely with
        ``auxiliary_scale`` as an independent feature block (widths add).
        Every feature is similarity invariant in the normalized frame and
        parity-typed identically to its parent (``ln(a + eps)`` even like
        ``a``; the normalized alignments odd in the normal and state
        vector exactly like ``b`` and ``v . r``).  Default ``False`` adds
        no parameters and is bitwise identical to the pre-extension
        decoder (state dict and outputs).
    mlp_members : int
        Number of learned smooth dictionary members produced by the pair MLP.
    mlp_hidden_dim : int
        Hidden width of the two-layer SiLU pair MLP.
    query_chunk_size : int
        Queries evaluated per dense chunk.  Memory only: every query row is
        computed independently, so chunking never changes the operator.
    checkpoint_query_chunks : bool
        Recompute each query chunk's dense pair activations in the backward
        pass (:func:`torch.utils.checkpoint.checkpoint`,
        ``use_reentrant=False``) instead of retaining them.  Without this,
        chunking bounds only the *forward* peak: autograd retains every
        chunk's :math:`O(\text{chunk}\times N_s)` intermediates, so training
        memory grows as :math:`O(N_qN_s)` — the measured product-scope wall
        (:math:`\approx742` GiB at :math:`10^5` sources
        :math:`\times\ 10^6` queries).  With it, retained decode activations
        drop to the chunk being recomputed plus the :math:`O(N_q)` outputs,
        at the price of one extra decode forward inside backward.  The
        recomputation is exact: the decoder is RNG-free and every chunk op
        is deterministic for fixed input shapes, so recomputed activations
        — and therefore gradients — are bitwise identical to the retained
        ones.  Default ``False`` preserves the historical autograd graph
        exactly.
    accumulation_dtype : torch.dtype or None
        Precision floor for the dense contraction, mirroring
        :class:`.attention.MeshAttention`.

    Notes
    -----
    One decode costs :math:`O(N_qN_s)` work and, per chunk,
    :math:`O(\text{chunk}\times N_s)` memory.  This is the documented price
    of a nonseparable kernel; see the module docstring.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        operator_scalar_dim: int,
        operator_vector_dim: int,
        drive_scalar_dim: int,
        drive_vector_dim: int,
        heads: int = 4,
        include_double_layer_member: bool = True,
        include_polynomial_members: bool = True,
        include_single_layer_member: bool = False,
        monopole_free_single_layer: bool = False,
        auxiliary_scale: bool = False,
        log_radial_features: bool = False,
        mlp_members: int = 8,
        mlp_hidden_dim: int = 48,
        query_chunk_size: int = 2048,
        checkpoint_query_chunks: bool = False,
        accumulation_dtype: torch.dtype | None = torch.float32,
        drive_pseudo_dim: int = 0,
        local_pair_features: str | None = None,
        near_theta: float = 0.25,
        decode_backend: str = "dense",
        bh_theta: float = 0.5,
        bh_leaf_size: int = 32,
    ) -> None:
        """Validate the configuration and build the member dictionary,
        coefficient map, value projection, and typed output maps.  Every
        parameter is documented in the class docstring's Parameters
        section; validation here enforces the coherence rules stated there
        (dimension dispatch, pseudo-sector planarity, member carriers for
        the feature blocks, probe-mode names)."""
        super().__init__()
        if n_spatial_dims not in _EXACT_MEMBER_VERTICES:
            raise ValueError(
                "KernelBasisCrossDecoder requires n_spatial_dims 2 or 3: the "
                "exact double-layer member is dimension-dispatched to segment "
                "and triangle quadrature"
            )
        if not isinstance(include_double_layer_member, bool):
            raise ValueError("include_double_layer_member must be a bool")
        if not isinstance(include_polynomial_members, bool):
            raise ValueError("include_polynomial_members must be a bool")
        if not (
            include_double_layer_member
            or include_single_layer_member
            or include_polynomial_members
            or mlp_members > 0
        ):
            raise ValueError(
                "the member dictionary is empty: enable at least one of the "
                "exact double-layer member, the exact single-layer member, "
                "the polynomial members, or mlp_members > 0"
            )
        if not isinstance(include_single_layer_member, bool):
            raise ValueError("include_single_layer_member must be a bool")
        if not isinstance(monopole_free_single_layer, bool):
            raise ValueError("monopole_free_single_layer must be a bool")
        if not isinstance(auxiliary_scale, bool):
            raise ValueError("auxiliary_scale must be a bool")
        if not isinstance(log_radial_features, bool):
            raise ValueError("log_radial_features must be a bool")
        if not isinstance(checkpoint_query_chunks, bool):
            raise ValueError("checkpoint_query_chunks must be a bool")
        if monopole_free_single_layer and not include_single_layer_member:
            raise ValueError(
                "monopole_free_single_layer deflates the exact single-layer "
                "member and therefore requires "
                "include_single_layer_member=True; without that member there "
                "is no monopole to control"
            )
        if auxiliary_scale and mlp_members == 0:
            raise ValueError(
                "auxiliary_scale=True requires mlp_members > 0: the auxiliary "
                "r/delta-rescaled invariants feed only the learned "
                "smooth-member MLP (the exact singular and polynomial members "
                "never read them), so without MLP members the declared scale "
                "has no carrier"
            )
        if log_radial_features and mlp_members == 0:
            raise ValueError(
                "log_radial_features=True requires mlp_members > 0: the "
                "log-radial pair features feed only the learned "
                "smooth-member MLP (the exact singular and polynomial "
                "members never read them), so without MLP members the "
                "features have no carrier"
            )
        if local_pair_features is not None:
            if local_pair_features not in _LOCAL_PAIR_FEATURE_MODES:
                raise ValueError(
                    "local_pair_features must be one of "
                    f"{sorted(_LOCAL_PAIR_FEATURE_MODES)} or None, got "
                    f"{local_pair_features!r}"
                )
            if mlp_members == 0:
                raise ValueError(
                    "local_pair_features requires mlp_members > 0: the "
                    "probe block feeds only the learned smooth-member MLP "
                    "(the exact singular and polynomial members never read "
                    "it), so without MLP members it has no carrier"
                )
        if not (
            isinstance(near_theta, float)
            and math.isfinite(near_theta)
            and near_theta > 0.0
        ):
            raise ValueError(
                f"near_theta must be a finite positive float, got {near_theta!r}"
            )
        if decode_backend not in ("dense", "barnes_hut"):
            raise ValueError(
                'decode_backend must be "dense" or "barnes_hut", got '
                f"{decode_backend!r}"
            )
        if decode_backend == "barnes_hut":
            # v1 scope of the hierarchical backend (task #41, design doc):
            # 3D triangle boundaries, the two exact singular members plus
            # smooth MLP members.  Each rejection names the missing
            # far-field treatment rather than silently approximating.
            if n_spatial_dims != 3:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' is implemented for 3D "
                    "triangle boundaries only (v1; the 2D complex-moment "
                    "expansion is designed but unbuilt)"
                )
            if not include_double_layer_member:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' v1 builds its far field "
                    "around the exact double-layer member; the MLP-only "
                    "ablation (include_double_layer_member=False) is a "
                    "dense-backend science arm"
                )
            if include_polynomial_members:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' has no far-field treatment "
                    "for the polynomial members (their {1, b, a} kernels do "
                    "not decay); disable include_polynomial_members or use "
                    "the dense backend"
                )
            if monopole_free_single_layer:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' does not implement the "
                    "monopole-free deflation (a global rank-one coupling "
                    "across all sources); use the dense backend"
                )
            if local_pair_features is not None:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' does not carry the "
                    "local_pair_features probe block (diagnostic-only, "
                    "retired per @sec-nb-probes-verdict); use the dense "
                    "backend"
                )
            if not (
                isinstance(bh_theta, float)
                and math.isfinite(bh_theta)
                and bh_theta > 0.0
            ):
                raise ValueError(
                    f"bh_theta must be a finite positive float, got {bh_theta!r}"
                )
            if isinstance(bh_leaf_size, bool) or (
                not isinstance(bh_leaf_size, int) or bh_leaf_size < 1
            ):
                raise ValueError(
                    f"bh_leaf_size must be an integer >= 1, got {bh_leaf_size!r}"
                )
            if checkpoint_query_chunks:
                raise NotImplementedError(
                    "decode_backend='barnes_hut' does not compose with "
                    "checkpoint_query_chunks in v1 (the checkpoint boundary "
                    "packs dense cache tensors only).  BH memory is sparse "
                    "-- O(near pairs + nodes) instead of O(chunk x S) -- so "
                    "the checkpoint is typically unnecessary; disable it"
                )
        # The two vector channel counts may be zero independently: a
        # vector-less operator state simply contributes no pair alignments or
        # Gram invariants, and a vector-less drive state carries no vector
        # value channels (``vector_value_dim=0`` below).  The joint
        # scalar-only coherence rule lives in :class:`.model.MeshTransformer`.
        for name, value, minimum in (
            ("operator_scalar_dim", operator_scalar_dim, 1),
            ("operator_vector_dim", operator_vector_dim, 0),
            ("drive_scalar_dim", drive_scalar_dim, 1),
            ("drive_vector_dim", drive_vector_dim, 0),
            ("drive_pseudo_dim", drive_pseudo_dim, 0),
            ("heads", heads, 1),
            ("mlp_members", mlp_members, 0),
            ("mlp_hidden_dim", mlp_hidden_dim, 1),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (value < minimum)
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if drive_pseudo_dim and n_spatial_dims != 2:
            raise ValueError(
                "pseudoscalar drive channels are two-dimensional by design "
                "(in 3D the analogous object is the axial vector, which is "
                f"out of scope); got n_spatial_dims={n_spatial_dims}"
            )

        self.n_spatial_dims = n_spatial_dims
        self.operator_scalar_dim = operator_scalar_dim
        self.operator_vector_dim = operator_vector_dim
        self.drive_scalar_dim = drive_scalar_dim
        self.drive_vector_dim = drive_vector_dim
        self.drive_pseudo_dim = drive_pseudo_dim
        self.heads = heads
        self.decode_backend = decode_backend
        self.bh_theta = bh_theta
        self.bh_leaf_size = bh_leaf_size
        self.include_polynomial_members = include_polynomial_members
        self.include_double_layer_member = include_double_layer_member
        self.include_single_layer_member = include_single_layer_member
        self.monopole_free_single_layer = monopole_free_single_layer
        self.auxiliary_scale = auxiliary_scale
        self.log_radial_features = log_radial_features
        self.local_pair_features = local_pair_features
        self.near_theta = near_theta
        self.mlp_members = mlp_members
        # Members: exact double layer, exact single layer (optional),
        # polynomial {1, b, a} (optional), learned MLP.
        # ``include_polynomial_members=False`` with ``mlp_members=0`` is the
        # singular-only ablation: the dictionary is the exact double-layer
        # member alone (plus the exact single-layer member when enabled --
        # the two-member "singpair" science arm), and the decoder must still
        # construct and run for those science arms.  The converse
        # ``include_double_layer_member=False`` (with the single layer also
        # off) is the MLP-only ablation: no exact singular structure at all,
        # the thesis-critical control for attributing industrial gains to
        # boundary-integral physics rather than generic pairwise capacity.
        self.n_members = (
            (1 if include_double_layer_member else 0)
            + (1 if include_single_layer_member else 0)
            + (3 if include_polynomial_members else 0)
            + mlp_members
        )
        self.scalar_value_dim = max(drive_scalar_dim // heads, 1)
        # Mirror the field blocks: no drive vector channels means no vector
        # value channels (a positive width here would ask TypedProjection to
        # create vectors without an input vector basis).
        self.vector_value_dim = (
            max(drive_vector_dim // heads, 1) if drive_vector_dim else 0
        )
        self.pseudo_value_dim = (
            max(drive_pseudo_dim // heads, 1) if drive_pseudo_dim else 0
        )
        self.query_chunk_size = query_chunk_size
        self.checkpoint_query_chunks = checkpoint_query_chunks
        self.accumulation_dtype = accumulation_dtype

        pair_features = 2 + self._pair_vector_channels()
        if auxiliary_scale:
            # The duplicated invariant block at the declared delta scale
            # doubles the smooth-member MLP's input width; nothing else in
            # the decoder reads the auxiliary invariants.
            pair_features *= 2
        if log_radial_features:
            # The log-radial block appends one more base-width copy of the
            # pair invariants (ln a, b/sqrt a, v.r/sqrt a) after every other
            # block; nothing else in the decoder reads it, and it composes
            # with the auxiliary block as independent widths.
            pair_features += 2 + self._pair_vector_channels()
        if local_pair_features is not None:
            # Probe block (task #53): five even similarity-invariant scalar
            # channels appended AFTER every other block; nothing else in the
            # decoder reads them.  All three modes share the width so their
            # parameter counts match exactly (the sharing-control comparison
            # requires it).
            pair_features += _LOCAL_PAIR_FEATURE_WIDTH
        final = _RowStableLinear(
            mlp_hidden_dim,
            mlp_members,
            bias=False,
        )
        # Small final-layer initialization: the learned smooth members start
        # near zero while the exact and polynomial members carry the initial
        # read-in, matching the benchmark-winning pair-kernel initialization.
        nn.init.normal_(final.weight, std=1.0e-2 / math.sqrt(mlp_hidden_dim))
        # ``mlp_members == 0`` is the dictionary-only ablation: the kernel is
        # spanned by the exact singular member and the polynomial members
        # alone, isolating structured-member behavior from learned smooth
        # corrections (the freq-OOD "dictionary-selection tension" probe).
        self.member_mlp = (
            nn.Sequential(
                _RowStableLinear(pair_features, mlp_hidden_dim),
                nn.SiLU(),
                _RowStableLinear(mlp_hidden_dim, mlp_hidden_dim),
                nn.SiLU(),
                final,
            )
            if mlp_members
            else None
        )
        # A bias keeps an operator-independent kernel component available; it
        # reads only source invariants, never the drive, in the linear class.
        self.coefficient_map = nn.Linear(
            self._source_invariant_channels(), self.n_members * heads
        )
        self.value_projection = TypedProjection(
            drive_scalar_dim,
            drive_vector_dim,
            heads * self.scalar_value_dim,
            heads * self.vector_value_dim,
            scalar_bias=False,
            include_vector_invariants=self._value_includes_vector_invariants(),
            pseudo_in=drive_pseudo_dim,
            pseudo_out=heads * self.pseudo_value_dim,
        )
        # Output maps are explicit bias-free parameter contractions rather
        # than ``nn.Linear`` so the whole decoder evaluates each query row
        # with batch-size-independent reduction order (see ``forward``).
        self.scalar_output_weight = nn.Parameter(
            torch.randn(drive_scalar_dim, heads * self.scalar_value_dim)
            / math.sqrt(heads * self.scalar_value_dim)
        )
        self.vector_output_weight = nn.Parameter(
            torch.randn(drive_vector_dim, heads, self.vector_value_dim)
            / math.sqrt(max(heads * self.vector_value_dim, 1))
        )
        # Registered only when the pseudo sector exists so the default state
        # dict stays bitwise identical to the pre-extension decoder.
        if drive_pseudo_dim:
            self.pseudo_output_weight = nn.Parameter(
                torch.randn(drive_pseudo_dim, heads * self.pseudo_value_dim)
                / math.sqrt(heads * self.pseudo_value_dim)
            )
        else:
            self.register_parameter("pseudo_output_weight", None)
        # The kernel message initializes the query field state (a read-in,
        # not a perturbative residual), so its learnable per-channel scale
        # starts at one rather than at the small residual LayerScale value.
        self.message_scale = StateLayerScale(
            drive_scalar_dim,
            drive_vector_dim,
            init=1.0,
            pseudo_dim=drive_pseudo_dim,
        )

    # ------------------------------------------------------------------
    # Linearity-critical hooks; implemented by the two concrete classes.
    # ------------------------------------------------------------------
    def _pair_vector_channels(self) -> int:
        """Number of per-source state vectors entering the pair invariants
        (fixes which states the kernel may read; linearity-critical)."""
        raise NotImplementedError(
            "instantiate LinearKernelBasisCrossDecoder or "
            "NonlinearZeroKernelBasisCrossDecoder"
        )

    def _source_invariant_channels(self) -> int:
        """Width of the invariant features feeding the coefficient map
        (linearity-critical: fixes what may condition the kernel)."""
        raise NotImplementedError(
            "instantiate LinearKernelBasisCrossDecoder or "
            "NonlinearZeroKernelBasisCrossDecoder"
        )

    def _kernel_pair_vectors(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s channels spatial_dims"]:
        """Per-source vectors whose alignments with the displacement join
        the pair invariants; which states contribute fixes the linearity
        class (operator-only keeps the kernel drive-independent)."""
        raise NotImplementedError

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s invariants"]:
        """Per-source invariant features conditioning the member
        coefficients; which states contribute fixes the linearity class."""
        raise NotImplementedError

    def _value_includes_vector_invariants(self) -> bool:
        """Whether the value projection may lift quadratic invariants
        (``False`` keeps values exactly drive-linear)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery.
    # ------------------------------------------------------------------
    def build_source_cache(
        self,
        source_mesh: Mesh,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
        *,
        auxiliary_scale: Float[torch.Tensor, ""] | None = None,
        local_scalars: Float[torch.Tensor, "s 2"] | None = None,
    ) -> KernelDecoderCache:
        """Cache every query-independent source quantity once per encode.

        ``auxiliary_scale`` is the declared dimensionless per-case ratio
        :math:`\\lambda=\\delta/L_{\\mathrm{ref}}` (a positive finite 0-dim
        tensor), required exactly when the decoder was constructed with
        ``auxiliary_scale=True`` and rejected otherwise.
        """
        if self.auxiliary_scale:
            if auxiliary_scale is None:
                raise ValueError(
                    "this decoder was constructed with auxiliary_scale=True "
                    "and requires the declared per-case scale tensor "
                    "(auxiliary_scale=...) when building its source cache"
                )
            if (
                not isinstance(auxiliary_scale, torch.Tensor)
                or auxiliary_scale.ndim != 0
            ):
                raise ValueError(
                    "auxiliary_scale must be a 0-dimensional tensor holding "
                    "the dimensionless ratio lambda = delta / L_ref"
                )
            if not torch.compiler.is_compiling() and (
                not torch.isfinite(auxiliary_scale).item()
                or auxiliary_scale.item() <= 0.0
            ):
                raise ValueError(
                    "auxiliary_scale must be finite and positive: it declares "
                    "a physical length-scale ratio delta / L_ref"
                )
        elif auxiliary_scale is not None:
            raise ValueError(
                "auxiliary_scale was supplied but this decoder was "
                "constructed with auxiliary_scale=False and would silently "
                "ignore the declared scale"
            )
        if source_mesh.n_spatial_dims != self.n_spatial_dims:
            raise ValueError("source mesh has the wrong spatial dimension")
        if source_mesh.n_cells != operator_state.n_entities:
            raise ValueError("source mesh and operator state entity counts differ")
        if operator_state.n_entities != drive_state.n_entities:
            raise ValueError("operator and drive state entity counts differ")
        expected_vertices = _EXACT_MEMBER_VERTICES[self.n_spatial_dims]
        if source_mesh.cells.shape[-1] != expected_vertices:
            raise ValueError(
                "the kernel query decoder requires boundary cells with "
                f"{expected_vertices} vertices in {self.n_spatial_dims}D "
                "(straight segments / flat triangles) for its exact "
                f"double-layer member, got cells with "
                f"{source_mesh.cells.shape[-1]} vertices"
            )
        if self.local_pair_features is not None and local_scalars is None:
            raise ValueError(
                "local_pair_features is enabled but no per-source local "
                "scalars were supplied to the cache build"
            )
        if local_scalars is not None and (
            local_scalars.ndim != 2 or local_scalars.shape != (source_mesh.n_cells, 2)
        ):
            raise ValueError(
                "local_scalars must have shape (n_cells, 2), got "
                f"{tuple(local_scalars.shape)}"
            )
        values = self.value_projection(drive_state)
        n = drive_state.n_entities
        cache = KernelDecoderCache(
            panel_vertices=source_mesh.points[source_mesh.cells],
            centroids=source_mesh.cell_centroids,
            normals=source_mesh.cell_normals,
            # Keep physical panel extent separate from represented measure.
            # Only panel_areas may define near-field geometry (for example
            # panel_size / distance); measure_factors multiply exact panel
            # integrals once, and quadrature_measures drive every midpoint
            # quadrature.
            weights=source_mesh.cell_areas,
            measure_factors=(
                cell_measure_weights(source_mesh)
                if MEASURE_WEIGHTS_KEY in source_mesh.cell_data
                else None
            ),
            pair_vectors=self._kernel_pair_vectors(operator_state, drive_state),
            coefficients=self.coefficient_map(
                self._kernel_source_invariants(operator_state, drive_state)
            ).reshape(n, self.n_members, self.heads),
            value_scalars=values.scalars.reshape(n, self.heads, self.scalar_value_dim),
            value_vectors=values.vectors.reshape(
                n, self.heads, self.vector_value_dim, self.n_spatial_dims
            ),
            value_pseudos=values.pseudos.reshape(n, self.heads, self.pseudo_value_dim),
            auxiliary_scale=auxiliary_scale,
            local_scalars=local_scalars,
        )
        if self.decode_backend == "barnes_hut":
            # Geometry-static tree plus channel-resolved far-field
            # aggregates, once per encode (barnes_hut module header).  The
            # tree indices are not differentiated (geometry is not a
            # training variable); the aggregates are linear in the learned
            # coefficients/values and carry gradients exactly.
            from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

            from . import barnes_hut as _bh

            with torch.autocast(device_type=cache.centroids.device.type, enabled=False):
                tree = ClusterTree.from_points(
                    cache.centroids,
                    leaf_size=self.bh_leaf_size,
                    areas=cache.geometric_panel_areas,
                )
                aggregates = _bh.build_node_aggregates(
                    tree,
                    measures=cache.quadrature_measures,
                    centroids=cache.centroids,
                    normals=cache.normals,
                    pair_vectors=cache.pair_vectors,
                    coefficients=cache.coefficients,
                    value_scalars=cache.value_scalars,
                    value_vectors=cache.value_vectors,
                    value_pseudos=(
                        cache.value_pseudos if self.pseudo_value_dim else None
                    ),
                    include_single_layer=self.include_single_layer_member,
                    n_smooth_members=self.mlp_members,
                )
            cache = dataclass_replace(cache, bh_tree=tree, bh_aggregates=aggregates)
        return cache

    def _accumulation_type(self, *tensors: torch.Tensor) -> torch.dtype:
        """Promote inputs with a precision floor, never downcast FP64."""
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        if self.accumulation_dtype is not None:
            dtype = torch.promote_types(dtype, self.accumulation_dtype)
        return dtype

    def _local_pair_feature_block(
        self,
        features: PairInvariantFeatures,
        cache: KernelDecoderCache,
        self_indices: Int[torch.Tensor, " q"] | None,
    ) -> Float[torch.Tensor, "q s 5"]:
        r"""Local-corrector probe block (task #53), ``(Q, S, 5)``.

        All channels are even, similarity-invariant, bounded scalars, built
        from elementwise per-pair maps plus (for the local modes) a per-row
        own-cell gather through the declared trace identity map -- no
        batch-shape-dependent reduction, so query-set independence is
        untouched.  The ``global_control`` mode's pooled scalars are
        measure-weighted means over the FIXED source axis, identical for
        every chunk by construction.
        """
        squared_distance = features.squared_distance
        if self.local_pair_features == "global_control":
            # Sharing control: identical width, zero per-pair locality.
            measure = cache.quadrature_measures.to(cache.local_scalars.dtype)
            pooled = (measure[:, None] * cache.local_scalars).sum(
                dim=0
            ) / measure.sum().clamp_min(torch.finfo(measure.dtype).tiny)
            block = squared_distance.new_zeros(
                (*squared_distance.shape, _LOCAL_PAIR_FEATURE_WIDTH)
            )
            block[..., 0] = pooled[0]
            block[..., 1] = pooled[1]
            return block
        theta = subtended_angle(
            squared_distance, cache.geometric_panel_areas, self.n_spatial_dims - 1
        )
        if self.local_pair_features == "windowed":
            # Smooth full-support window: bounded to [0, 1), -> 0 as
            # d -> infinity, -> 1 on contact; no free constant.
            window = theta / (1.0 + theta)
        else:  # "near_only"
            # C^1 smoothstep, EXACTLY zero for theta <= near_theta / 2 and
            # one for theta >= near_theta: compact near-field support.
            ramp = torch.clamp(
                (theta - 0.5 * self.near_theta) / (0.5 * self.near_theta),
                min=0.0,
                max=1.0,
            )
            window = ramp.square() * (3.0 - 2.0 * ramp)
        source_scalars = cache.local_scalars.to(window.dtype)  # (S, 2)
        # Query-side scalars via the declared trace identity map: row i
        # reads exactly its own declared cell -- a per-row gather.
        query_scalars = source_scalars[self_indices]  # (Q, 2)
        return torch.stack(
            (
                window,
                window * query_scalars[:, 0:1],
                window * query_scalars[:, 1:2],
                window * source_scalars[None, :, 0],
                window * source_scalars[None, :, 1],
            ),
            dim=-1,
        )

    def _evaluate_chunk(
        self,
        query_points: Float[torch.Tensor, "q spatial_dims"],
        cache: KernelDecoderCache,
        self_indices: Int[torch.Tensor, " q"] | None = None,
    ) -> ScalarVectorState:
        """Evaluate the full dense pair-kernel message for one query chunk
        (exact members, optional smooth members, typed read-out); the
        non-checkpointed inner loop of :meth:`forward`."""
        if self.decode_backend == "barnes_hut":
            return self._evaluate_chunk_barnes_hut(query_points, cache, self_indices)
        # Geometry-derived pair quantities keep the input geometry precision
        # even under an ambient autocast scope: the exact member and the
        # invariants are numerical mesh operations, not learned layers.
        device_type = query_points.device.type
        # Pair invariants feed only the smooth members (polynomial and MLP).
        # On the singular-only and singpair arms neither exists, and the
        # dense O(Q x S) invariant tensors were measured dead work in the
        # decode hot loop (engineering review, "dead work in the dense hot
        # loop") -- skip them entirely; the exact members never read them.
        needs_pair_features = (
            self.member_mlp is not None or self.include_polynomial_members
        )
        with torch.autocast(device_type=device_type, enabled=False):
            features = (
                PairInvariantFeatures.compute(
                    query_points,
                    cache.centroids,
                    cache.normals,
                    cache.pair_vectors,
                    # The declared auxiliary scale reaches only the pair
                    # invariants' auxiliary block (and through it only the
                    # smooth-member MLP below); the exact singular members and
                    # polynomial members never read it.
                    auxiliary_scale=(
                        cache.auxiliary_scale if self.auxiliary_scale else None
                    ),
                    # The log-radial block is derived elementwise from the same
                    # chunk quantities (no new cache tensor, no new checkpoint
                    # argument) and likewise reaches only the smooth-member MLP.
                    log_radial=self.log_radial_features,
                )
                if needs_pair_features
                else None
            )
            if self.include_double_layer_member:
                singular = exact_double_layer_member(
                    query_points, cache.panel_vertices, cache.normals
                )
                # The closed form already integrates over the geometric
                # panel. Public inclusion/representation factors multiply
                # that exact integral once; effective area would count the
                # geometric panel a second time.
                singular = (
                    singular
                    * cache.representation_measure_factors.to(singular.dtype)[None, :]
                )
                if self_indices is not None:
                    # Declared boundary-trace queries: the closed form lands
                    # on an accidental signed-zero branch of the jump
                    # discontinuity at its own panel; replace the (query, own
                    # panel) entries with the exact exterior one-sided limit
                    # +1/2 (see exterior_trace_self_entries for the jump
                    # relation). Apply this identity term after the public
                    # factor so the local +1/2 jump is not Horvitz--Thompson
                    # reweighted. The single-layer member below is continuous
                    # across the boundary -- only its normal derivative jumps
                    # -- so its value needs, and receives, no correction.
                    singular = exterior_trace_self_entries(singular, self_indices)
                singular = singular.unsqueeze(-1)
            else:
                # MLP-only ablation (external-review P0): no exact
                # double-layer stack; the single layer below may still open
                # the singular block.
                singular = None
            if self.include_single_layer_member:
                # Second exact singular member: the single layer never reads
                # the cell normals (orientation independent, unlike the
                # sigma-carrying double layer above) and, like the double
                # layer, already includes the boundary measure.
                single_layer = exact_single_layer_member(
                    query_points, cache.panel_vertices
                )
                single_layer = (
                    single_layer
                    * cache.representation_measure_factors.to(single_layer.dtype)[
                        None, :
                    ]
                )
                if self.monopole_free_single_layer:
                    # Measure-weighted rank-one deflation (module docstring):
                    # subtract each member's share of the uniform-density
                    # boundary potential so every deflated member carries
                    # exactly zero net charge -- the exact projection of the
                    # conditioned single-layer density onto the zero-net-flux
                    # subspace, killing the monopole tail (log r in 2D, 1/r
                    # in 3D) by construction.  The reduction is a fixed-axis
                    # per-query-row sum, so bitwise query-set independence is
                    # preserved, and the map is linear and differentiable.
                    weights = cache.quadrature_measures.to(single_layer.dtype)
                    single_layer = (
                        single_layer
                        - single_layer.sum(dim=-1, keepdim=True)
                        * (weights / weights.sum())[None, :]
                    )
                single_layer = single_layer.unsqueeze(-1)
                singular = (
                    single_layer
                    if singular is None
                    else torch.cat((singular, single_layer), dim=-1)
                )
            polynomial = (
                torch.stack(
                    (
                        torch.ones_like(features.squared_distance),
                        features.normal_alignment,
                        features.squared_distance,
                    ),
                    dim=-1,
                )
                if self.include_polynomial_members
                else None
            )
            probe_block = (
                self._local_pair_feature_block(features, cache, self_indices)
                if self.local_pair_features is not None
                else None
            )
        if self.member_mlp is not None:
            mlp_input = features.stacked()
            if probe_block is not None:
                # The probe block joins AFTER every other feature block
                # (base, auxiliary, log-radial), mirroring their append
                # discipline; only the smooth-member MLP reads it.
                mlp_input = torch.cat(
                    (mlp_input, probe_block.to(mlp_input.dtype)), dim=-1
                )
            learned = self.member_mlp(mlp_input)
            smooth_parts = (
                torch.cat((polynomial.to(learned.dtype), learned), dim=-1)
                if polynomial is not None
                else learned
            )
        else:
            smooth_parts = polynomial
        if smooth_parts is not None:
            # Smooth members use consistent midpoint quadrature (value at the
            # centroid times effective cell measure); singular members
            # already contain geometric panel integration and received only
            # their dimensionless public factor above.
            smooth = (
                smooth_parts
                * cache.quadrature_measures.to(smooth_parts.dtype)[None, :, None]
            )
            members = (
                smooth
                if singular is None
                else torch.cat((singular.to(smooth.dtype), smooth), dim=-1)
            )
        else:
            # Singular-only ablation: no smooth members exist, so the member
            # axis carries the exact singular member(s) alone.  (The
            # both-empty case is rejected at construction: the member
            # dictionary must be non-empty.)
            members = singular

        dtype = self._accumulation_type(
            members,
            cache.coefficients,
            cache.value_scalars,
            cache.value_vectors,
        )
        # Broadcast multiply-plus-sum keeps each query row's floating-point
        # reduction order independent of how many queries share the batch (a
        # batched-GEMM lowering may reorder the contraction by shape), which
        # preserves exact query-set independence.
        with torch.autocast(device_type=device_type, enabled=False):
            kernel = (
                members.to(dtype).unsqueeze(-1)
                * cache.coefficients.to(dtype)[None, :, :, :]
            ).sum(dim=2)  # (Q, S, H)
            scalar_heads = (
                kernel.unsqueeze(-1) * cache.value_scalars.to(dtype)[None, :, :, :]
            ).sum(dim=1)  # (Q, H, F_s)
            vector_heads = (
                kernel[..., None, None]
                * cache.value_vectors.to(dtype)[None, :, :, :, :]
            ).sum(dim=1)  # (Q, H, F_v, D)
            pseudo_heads = (
                (
                    kernel.unsqueeze(-1) * cache.value_pseudos.to(dtype)[None, :, :, :]
                ).sum(dim=1)  # (Q, H, F_p)
                if self.pseudo_value_dim
                else None
            )
        output_dtype = query_points.dtype
        heads_flat = scalar_heads.to(output_dtype).reshape(
            query_points.shape[0], self.heads * self.scalar_value_dim
        )
        scalars = (heads_flat[:, None, :] * self.scalar_output_weight[None, :, :]).sum(
            dim=-1
        )
        vectors = (
            self.vector_output_weight[None, :, :, :, None]
            * vector_heads.to(output_dtype)[:, None, :, :, :]
        ).sum(dim=(2, 3))
        if self.drive_pseudo_dim:
            # Same row-stable broadcast-and-sum contraction as the scalar
            # output; a separate map keeps the two parities unmixed.
            pseudo_flat = pseudo_heads.to(output_dtype).reshape(
                query_points.shape[0], self.heads * self.pseudo_value_dim
            )
            pseudos = (
                pseudo_flat[:, None, :] * self.pseudo_output_weight[None, :, :]
            ).sum(dim=-1)
        else:
            pseudos = scalars.new_empty(query_points.shape[0], 0)
        return self.message_scale(
            ScalarVectorState(
                scalars,
                vectors.to(dtype=scalars.dtype),
                pseudos.to(dtype=scalars.dtype),
            )
        )

    def _evaluate_chunk_barnes_hut(
        self,
        query_points: Float[torch.Tensor, "q spatial_dims"],
        cache: KernelDecoderCache,
        self_indices: Int[torch.Tensor, " q"] | None = None,
    ) -> ScalarVectorState:
        r"""Hierarchical (Barnes--Hut) evaluation of one query chunk.

        Near pairs run the SAME closed forms and smooth-member MLP as the
        dense path, pairwise (per-pair values bitwise identical -- pinned by
        ``test_barnes_hut.py``); far (query, node) pairs read the cache's
        channel-resolved node aggregates against analytic point kernels
        (exact members) and the node's virtual source (smooth members).
        See ``barnes_hut.py`` for the contract; the output projection tail
        deliberately mirrors :meth:`_evaluate_chunk` (kept duplicated so the
        bitwise-sensitive dense path is never touched; the equivalence
        tests couple the two).
        """
        from . import barnes_hut as _bh

        if cache.bh_tree is None or cache.bh_aggregates is None:
            raise ValueError(
                "decode_backend='barnes_hut' requires a cache built by this "
                "decoder's build_source_cache (bh_tree/bh_aggregates missing "
                "-- the cache came from a dense-backend build)"
            )
        tree = cache.bh_tree
        agg = cache.bh_aggregates
        device_type = query_points.device.type
        n_queries = query_points.shape[0]

        def _pair_features(
            displacement: torch.Tensor,  # (P, D)
            normals: torch.Tensor,  # (P, D)
            pair_vectors: torch.Tensor,  # (P, C, D)
        ) -> torch.Tensor:
            """Per-pair mirror of ``PairInvariantFeatures`` + ``stacked()``:
            same invariants, same block order (base, auxiliary, log-radial),
            elementwise ops only."""
            squared_distance = displacement.square().sum(dim=-1)
            normal_alignment = (displacement * normals).sum(dim=-1)
            vector_alignments = (displacement[:, None, :] * pair_vectors).sum(dim=-1)
            parts = [
                squared_distance.unsqueeze(-1),
                normal_alignment.unsqueeze(-1),
                vector_alignments,
            ]
            if self.auxiliary_scale:
                lam = cache.auxiliary_scale
                parts.extend(
                    (
                        (squared_distance / lam.square()).unsqueeze(-1),
                        (normal_alignment / lam).unsqueeze(-1),
                        vector_alignments / lam,
                    )
                )
            if self.log_radial_features:
                radius = torch.sqrt(squared_distance + _LOG_RADIAL_EPS)
                parts.extend(
                    (
                        torch.log(squared_distance + _LOG_RADIAL_EPS).unsqueeze(-1),
                        (normal_alignment / radius).unsqueeze(-1),
                        vector_alignments / radius.unsqueeze(-1),
                    )
                )
            return torch.cat(parts, dim=-1)

        with torch.autocast(device_type=device_type, enabled=False):
            part = _bh.single_tree_partition(query_points, tree, self.bh_theta)
        nq, ns = part.near_query, part.near_source
        fq, fn = part.far_query, part.far_node

        dtype = self._accumulation_type(
            query_points,
            cache.coefficients,
            cache.value_scalars,
            cache.value_vectors,
        )
        n_heads = self.heads
        f_s = self.scalar_value_dim
        f_v = self.vector_value_dim
        f_p = self.pseudo_value_dim
        has_pseudos = bool(f_p)
        n_dims = query_points.shape[-1]
        # One-time dtype promotion of every gathered operand (no-ops when
        # the cache already carries the accumulation dtype).
        coefficients = cache.coefficients.to(dtype)
        value_scalars = cache.value_scalars.to(dtype)
        value_vectors = cache.value_vectors.to(dtype)
        value_pseudos = (
            cache.value_pseudos.to(dtype) if cache.value_pseudos is not None else None
        )

        # Both fields reduce through the lazy block-fused segment fold: each
        # rank block gathers at most (n_queries * block) pairs, computes the
        # per-pair typed rows [scalar | vector | pseudo] flattened to one
        # (nb, C) tensor, and folds immediately -- nothing (P, H, F)-sized
        # is ever materialized, so memory is bounded by the block size, not
        # the pair count.

        def _typed_row(kernel_p: torch.Tensor, ns_s: torch.Tensor) -> torch.Tensor:
            # Explicit widths: -1 inference fails on empty selections.
            nb = kernel_p.shape[0]
            parts = [
                (kernel_p.unsqueeze(-1) * value_scalars[ns_s]).reshape(
                    nb, n_heads * f_s
                ),
                (kernel_p[:, :, None, None] * value_vectors[ns_s]).reshape(
                    nb, n_heads * f_v * n_dims
                ),
            ]
            if has_pseudos:
                parts.append(
                    (kernel_p.unsqueeze(-1) * value_pseudos[ns_s]).reshape(
                        nb, n_heads * f_p
                    )
                )
            return torch.cat(parts, dim=-1)

        def _near_rows(sel: torch.Tensor) -> torch.Tensor:
            nq_s = nq[sel]
            ns_s = ns[sel]
            with torch.autocast(device_type=device_type, enabled=False):
                pieces = []
                dl = _bh.pair_triangle_double_layer(
                    query_points[nq_s],
                    cache.panel_vertices[ns_s],
                    cache.normals[ns_s],
                )
                dl = dl * cache.representation_measure_factors[ns_s].to(dl.dtype)
                if self_indices is not None:
                    own = ns_s == self_indices[nq_s]
                    dl = torch.where(own, dl.new_full((), 0.5), dl)
                pieces.append(dl.unsqueeze(-1))
                if self.include_single_layer_member:
                    sl = _bh.pair_triangle_single_layer(
                        query_points[nq_s], cache.panel_vertices[ns_s]
                    )
                    sl = sl * cache.representation_measure_factors[ns_s].to(sl.dtype)
                    pieces.append(sl.unsqueeze(-1))
                displacement = query_points[nq_s] - cache.centroids[ns_s]
            if self.member_mlp is not None:
                # Under ambient autocast, mirroring the dense path.
                feats = _pair_features(
                    displacement, cache.normals[ns_s], cache.pair_vectors[ns_s]
                )
                learned = self.member_mlp(feats)
                pieces.append(
                    learned * cache.quadrature_measures[ns_s].to(learned.dtype)[:, None]
                )
            with torch.autocast(device_type=device_type, enabled=False):
                members = torch.cat([p.to(pieces[0].dtype) for p in pieces], dim=-1)
                kernel_p = (members.to(dtype).unsqueeze(-1) * coefficients[ns_s]).sum(
                    dim=1
                )  # (nb, H)
                return _typed_row(kernel_p, ns_s)

        def _far_rows(sel: torch.Tensor) -> torch.Tensor:
            fq_s = fq[sel]
            fn_s = fn[sel]
            nb = sel.shape[0]
            with torch.autocast(device_type=device_type, enabled=False):
                r_vec = query_points[fq_s] - agg.centroid[fn_s]
                r = torch.sqrt(r_vec.square().sum(dim=-1)).clamp_min(
                    torch.finfo(query_points.dtype).tiny
                )
                # Double layer far limit: sum(A n rho) . (x - c)/(4 pi r^3),
                # contracted per dipole component so gathers stay (nb, H, F).
                dl_kernel = (r_vec / (_FOUR_PI * r.pow(3))[:, None]).to(dtype)
                scal = dl_kernel.new_zeros((nb, n_heads, f_s))
                vec = dl_kernel.new_zeros((nb, n_heads, f_v, n_dims))
                pse = dl_kernel.new_zeros((nb, n_heads, f_p)) if has_pseudos else None
                for d in range(n_dims):
                    w = dl_kernel[:, d]
                    scal = scal + w[:, None, None] * agg.dl_scalars.to(dtype)[fn_s, d]
                    vec = (
                        vec + w[:, None, None, None] * agg.dl_vectors.to(dtype)[fn_s, d]
                    )
                    if pse is not None and agg.dl_pseudos is not None:
                        pse = pse + w[:, None, None] * agg.dl_pseudos.to(dtype)[fn_s, d]
                if self.include_single_layer_member:
                    sl_kernel = (1.0 / (_FOUR_PI * r)).to(dtype)
                    scal = (
                        scal + sl_kernel[:, None, None] * agg.sl_scalars.to(dtype)[fn_s]
                    )
                    vec = (
                        vec
                        + sl_kernel[:, None, None, None]
                        * agg.sl_vectors.to(dtype)[fn_s]
                    )
                    if pse is not None and agg.sl_pseudos is not None:
                        pse = (
                            pse
                            + sl_kernel[:, None, None] * agg.sl_pseudos.to(dtype)[fn_s]
                        )
            if self.member_mlp is not None:
                # Virtual-source smooth members under ambient autocast.
                far_feats = _pair_features(
                    query_points[fq_s] - agg.centroid[fn_s],
                    agg.unit_normal[fn_s],
                    agg.mean_pair_vectors[fn_s],
                )
                far_learned = self.member_mlp(far_feats).to(dtype)  # (nb, M)
                with torch.autocast(device_type=device_type, enabled=False):
                    for m in range(self.mlp_members):
                        w = far_learned[:, m]
                        scal = (
                            scal
                            + w[:, None, None] * agg.smooth_scalars.to(dtype)[fn_s, m]
                        )
                        vec = (
                            vec
                            + w[:, None, None, None]
                            * agg.smooth_vectors.to(dtype)[fn_s, m]
                        )
                        if pse is not None and agg.smooth_pseudos is not None:
                            pse = (
                                pse
                                + w[:, None, None]
                                * agg.smooth_pseudos.to(dtype)[fn_s, m]
                            )
            parts = [
                scal.reshape(nb, n_heads * f_s),
                vec.reshape(nb, n_heads * f_v * n_dims),
            ]
            if pse is not None:
                parts.append(pse.reshape(nb, n_heads * f_p))
            return torch.cat(parts, dim=-1)

        combined = _bh.segment_sum_by_query(
            _near_rows, nq, n_queries, block=_BH_PAIR_BLOCK, checkpoint_blocks=True
        ) + _bh.segment_sum_by_query(
            _far_rows, fq, n_queries, block=_BH_PAIR_BLOCK, checkpoint_blocks=True
        )
        split_s = n_heads * f_s
        split_v = split_s + n_heads * f_v * n_dims
        scalar_heads = combined[:, :split_s].reshape(n_queries, n_heads, f_s)
        vector_heads = combined[:, split_s:split_v].reshape(
            n_queries, n_heads, f_v, n_dims
        )
        pseudo_heads = (
            combined[:, split_v:].reshape(n_queries, n_heads, f_p)
            if has_pseudos
            else None
        )

        # ---- Typed output projection (mirrors _evaluate_chunk's tail) -----
        output_dtype = query_points.dtype
        heads_flat = scalar_heads.to(output_dtype).reshape(
            n_queries, self.heads * self.scalar_value_dim
        )
        scalars = (heads_flat[:, None, :] * self.scalar_output_weight[None, :, :]).sum(
            dim=-1
        )
        vectors = (
            self.vector_output_weight[None, :, :, :, None]
            * vector_heads.to(output_dtype)[:, None, :, :, :]
        ).sum(dim=(2, 3))
        if self.drive_pseudo_dim:
            pseudo_flat = pseudo_heads.to(output_dtype).reshape(
                n_queries, self.heads * self.pseudo_value_dim
            )
            pseudos = (
                pseudo_flat[:, None, :] * self.pseudo_output_weight[None, :, :]
            ).sum(dim=-1)
        else:
            pseudos = scalars.new_empty(n_queries, 0)
        return self.message_scale(
            ScalarVectorState(
                scalars,
                vectors.to(dtype=scalars.dtype),
                pseudos.to(dtype=scalars.dtype),
            )
        )

    def _checkpointed_chunk(
        self,
        query_points: Float[torch.Tensor, "q spatial_dims"],
        cache: KernelDecoderCache,
        self_indices: Int[torch.Tensor, " q"] | None = None,
    ) -> ScalarVectorState:
        """:meth:`_evaluate_chunk` under gradient checkpointing: the chunk's
        dense O(Q x S) activations are recomputed in backward instead of
        stored, bounding training memory by one chunk (bitwise identical to
        the plain path; see the module docstring's memory section)."""
        # Autograd connects only through tensors passed as explicit
        # checkpoint arguments, so the cache is unpacked here and rebuilt
        # inside; parameters used within ``_evaluate_chunk`` receive
        # gradients through the non-reentrant path.  ``preserve_rng_state``
        # is off because the decoder is RNG-free (no dropout anywhere), so
        # the recomputation is deterministic without the state round-trip.
        # The optional cache fields (pseudo values, declared auxiliary
        # scale) ride through the same explicit argument packing so their
        # gradients survive checkpointing; the declared trace indices are
        # gradient-free integers but ride the same packing for uniformity.
        has_pseudos = cache.value_pseudos is not None
        has_auxiliary = cache.auxiliary_scale is not None
        has_local = cache.local_scalars is not None
        has_self_indices = self_indices is not None

        def run(
            points: torch.Tensor,
            panel_vertices: torch.Tensor,
            centroids: torch.Tensor,
            normals: torch.Tensor,
            panel_areas: torch.Tensor,
            measure_factors: torch.Tensor,
            pair_vectors: torch.Tensor,
            coefficients: torch.Tensor,
            value_scalars: torch.Tensor,
            value_vectors: torch.Tensor,
            *optional: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            extras = list(optional)
            message = self._evaluate_chunk(
                points,
                KernelDecoderCache(
                    panel_vertices=panel_vertices,
                    centroids=centroids,
                    normals=normals,
                    weights=panel_areas,
                    measure_factors=measure_factors,
                    pair_vectors=pair_vectors,
                    coefficients=coefficients,
                    value_scalars=value_scalars,
                    value_vectors=value_vectors,
                    value_pseudos=extras.pop(0) if has_pseudos else None,
                    auxiliary_scale=extras.pop(0) if has_auxiliary else None,
                    local_scalars=extras.pop(0) if has_local else None,
                ),
                self_indices=extras.pop(0) if has_self_indices else None,
            )
            return message.scalars, message.vectors, message.pseudos

        tensors = (
            query_points,
            cache.panel_vertices,
            cache.centroids,
            cache.normals,
            cache.geometric_panel_areas,
            cache.representation_measure_factors,
            cache.pair_vectors,
            cache.coefficients,
            cache.value_scalars,
            cache.value_vectors,
        )
        if has_pseudos:
            tensors = tensors + (cache.value_pseudos,)
        if has_auxiliary:
            tensors = tensors + (cache.auxiliary_scale,)
        if has_local:
            tensors = tensors + (cache.local_scalars,)
        if has_self_indices:
            tensors = tensors + (self_indices,)
        scalars, vectors, out_pseudos = torch.utils.checkpoint.checkpoint(
            run,
            *tensors,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        return ScalarVectorState(scalars, vectors, out_pseudos)

    def forward(
        self,
        query_points: Float[torch.Tensor, "q spatial_dims"],
        cache: KernelDecoderCache,
        self_indices: Int[torch.Tensor, " q"] | None = None,
    ) -> ScalarVectorState:
        """Evaluate the dense pair-kernel message at independent query points.

        ``self_indices`` (optional, ``(Q,)`` ``torch.long``) declares that
        query ``i`` lies ON source panel ``self_indices[i]``: the exact
        double-layer member's own-panel entries are then replaced by the
        exterior one-sided limit of the jump relation (see
        :func:`exterior_trace_self_entries`; the single-layer member is
        continuous across the boundary and is untouched).  The default
        ``None`` is bitwise identical to the pre-extension decode.  Query
        independence becomes independence *given the declared identity
        map*: each row still never reads another row, and chunking remains
        a pure memory control.
        """
        if query_points.ndim != 2 or query_points.shape[-1] != self.n_spatial_dims:
            raise ValueError(
                f"query_points must have shape (Q, {self.n_spatial_dims}), got "
                f"{tuple(query_points.shape)}"
            )
        if not isinstance(cache, KernelDecoderCache):
            raise TypeError("cache must be a KernelDecoderCache from encode")
        expected = (
            cache.coefficients.shape[0],
            self.n_members,
            self.heads,
        )
        if tuple(cache.coefficients.shape) != expected:
            raise ValueError(
                "KernelDecoderCache is incompatible with this decoder; "
                f"expected coefficients of shape {expected}, got "
                f"{tuple(cache.coefficients.shape)}"
            )
        if self.pseudo_value_dim and (
            cache.value_pseudos is None
            or tuple(cache.value_pseudos.shape)
            != (
                cache.coefficients.shape[0],
                self.heads,
                self.pseudo_value_dim,
            )
        ):
            raise ValueError(
                "KernelDecoderCache carries no compatible pseudoscalar value "
                "features; it was built by a decoder without this decoder's "
                "pseudo sector"
            )
        if self.auxiliary_scale and cache.auxiliary_scale is None:
            raise ValueError(
                "KernelDecoderCache carries no declared auxiliary scale; it "
                "was built by a decoder without this decoder's "
                "auxiliary-scale contract"
            )
        if self.local_pair_features is not None:
            if cache.local_scalars is None:
                raise ValueError(
                    "local_pair_features is enabled but the cache carries no "
                    "per-source local scalars; it was built by a decoder (or "
                    "model) without the probe contract -- re-encode"
                )
            if self.local_pair_features != "global_control" and self_indices is None:
                raise ValueError(
                    "local_pair_features mode "
                    f"{self.local_pair_features!r} reads query-side local "
                    "scalars through the declared trace identity map and "
                    "therefore requires self_indices (trace mode)"
                )
        n_queries = query_points.shape[0]
        if self_indices is not None:
            if (
                not isinstance(self_indices, torch.Tensor)
                or self_indices.ndim != 1
                or self_indices.dtype != torch.long
            ):
                raise ValueError(
                    "self_indices must be a 1-D torch.long tensor of declared "
                    "own-panel source indices, one per query point"
                )
            if self_indices.shape[0] != n_queries:
                raise ValueError(
                    f"self_indices declares {self_indices.shape[0]} own-panel "
                    f"indices for {n_queries} query points; the boundary-trace "
                    "declaration requires exactly one source cell per query"
                )
            if self_indices.device != query_points.device:
                raise ValueError("self_indices must share the query device")
            n_sources = cache.coefficients.shape[0]
            if (
                not torch.compiler.is_compiling()
                and n_queries
                and (
                    int(self_indices.min()) < 0 or int(self_indices.max()) >= n_sources
                )
            ):
                raise ValueError(
                    "self_indices must index the cached source cells "
                    f"[0, {n_sources}); got values outside that range"
                )
        if n_queries == 0:
            return ScalarVectorState.zeros(
                0,
                self.drive_scalar_dim,
                self.drive_vector_dim,
                self.n_spatial_dims,
                pseudo_channels=self.drive_pseudo_dim,
                device=query_points.device,
                dtype=query_points.dtype,
            )
        # Checkpointing is a training-memory control only: under no_grad or
        # inference the plain chunk evaluation already peaks at one chunk.
        evaluate = (
            self._checkpointed_chunk
            if self.checkpoint_query_chunks and torch.is_grad_enabled()
            else self._evaluate_chunk
        )
        outputs = [
            evaluate(
                query_points[start : start + self.query_chunk_size],
                cache,
                None
                if self_indices is None
                else self_indices[start : start + self.query_chunk_size],
            )
            for start in range(0, n_queries, self.query_chunk_size)
        ]
        if len(outputs) == 1:
            return outputs[0]
        return ScalarVectorState(
            torch.cat([output.scalars for output in outputs], dim=0),
            torch.cat([output.vectors for output in outputs], dim=0),
            torch.cat([output.pseudos for output in outputs], dim=0),
        )


class LinearKernelBasisCrossDecoder(KernelBasisCrossDecoder):
    r"""Kernel decoder whose message is exactly linear in the drive.

    The kernel reads only the operator state: pair alignments use the
    operator vectors and the member coefficients use operator scalars plus
    operator-vector Gram invariants.  Values are a bias-free linear
    projection of the drive with vector Gram invariants disabled, so at fixed
    geometry the message obeys exact superposition and maps zero drive to
    zero, up to floating point.
    """

    def _pair_vector_channels(self) -> int:
        """Operator vectors only (drive-blind kernel)."""
        return self.operator_vector_dim

    def _source_invariant_channels(self) -> int:
        """Width of the drive-blind invariant feature set."""
        return (
            self.operator_scalar_dim
            + self.operator_vector_dim * (self.operator_vector_dim + 1) // 2
        )

    def _kernel_pair_vectors(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s channels spatial_dims"]:
        """Operator vectors only: the kernel never reads the drive."""
        return operator_state.vectors

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s invariants"]:
        """Operator scalars plus operator-vector Grams: drive-blind."""
        return torch.cat(
            (operator_state.scalars, _gram_invariants(operator_state.vectors)),
            dim=-1,
        )

    def _value_includes_vector_invariants(self) -> bool:
        """Disabled: values stay exactly drive-linear."""
        return False


class NonlinearZeroKernelBasisCrossDecoder(KernelBasisCrossDecoder):
    r"""Kernel decoder with drive-dependent kernel and exact zero preservation.

    The kernel additionally reads drive invariants: drive vectors join the
    pair alignments and drive scalars plus drive-vector Gram invariants --
    and, with a pseudo sector, the invariant ``0o x 0o -> 0e`` drive-pseudo
    pair products -- join the member-coefficient inputs, so the map is
    nonlinear in the drive.  Values remain bias-free (linear plus drive
    quadratic invariants), so a zero drive still produces an exactly zero
    message; superposition is deliberately not claimed.
    """

    def _pair_vector_channels(self) -> int:
        """Operator vectors only (drive-blind kernel)."""
        return self.operator_vector_dim + self.drive_vector_dim

    def _source_invariant_channels(self) -> int:
        """Width of the drive-inclusive invariant feature set."""
        return (
            self.operator_scalar_dim
            + self.operator_vector_dim * (self.operator_vector_dim + 1) // 2
            + self.drive_scalar_dim
            + self.drive_vector_dim * (self.drive_vector_dim + 1) // 2
            + self.drive_pseudo_dim * (self.drive_pseudo_dim + 1) // 2
        )

    def _kernel_pair_vectors(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s channels spatial_dims"]:
        """Operator AND drive vectors: the kernel is drive-dependent."""
        return torch.cat((operator_state.vectors, drive_state.vectors), dim=1)

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> Float[torch.Tensor, "s invariants"]:
        """Operator and drive invariants (incl. pseudo pair products)."""
        parts = [
            operator_state.scalars,
            _gram_invariants(operator_state.vectors),
            drive_state.scalars,
            _gram_invariants(drive_state.vectors),
        ]
        if self.drive_pseudo_dim:
            parts.append(_pseudo_pair_invariants(drive_state.pseudos))
        return torch.cat(parts, dim=-1)

    def _value_includes_vector_invariants(self) -> bool:
        """Enabled: quadratic drive invariants still vanish at zero drive."""
        return True


__all__ = [
    "KernelBasisCrossDecoder",
    "KernelDecoderCache",
    "LinearKernelBasisCrossDecoder",
    "NonlinearZeroKernelBasisCrossDecoder",
    "PairInvariantFeatures",
    "exact_double_layer_member",
    "exact_single_layer_member",
    "exterior_trace_self_entries",
]
