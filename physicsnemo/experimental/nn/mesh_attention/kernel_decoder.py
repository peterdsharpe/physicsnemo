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
  the cell-integrated influence with the boundary measure included; it is
  never multiplied by the cell weight again.  Exact integration matters:
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
  influence with the boundary measure included, never reweighted, because
  singular kernels need singular quadrature.  Unlike the double-layer
  member, the single layer is orientation independent (no :math:`\sigma`
  factor; it never reads the cell normal).
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
implementation additionally avoids batch-shape-dependent GEMM reductions on
the query axis (broadcast multiply-plus-sum contractions and
:class:`_RowStableLinear` pair layers), making that independence bitwise
rather than merely tight, and making query chunking a pure memory control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.mesh import Mesh

from .attention import (
    ScalarVectorState,
    TypedProjection,
    _gram_invariants,
    _pseudo_pair_invariants,
)
from .block import StateLayerScale

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Boundary cell vertex counts admitting an exact double-layer member.
_EXACT_MEMBER_VERTICES = {2: 2, 3: 3}


class _RowStableLinear(nn.Module):
    r"""A linear map whose per-row reduction order ignores the batch shape.

    ``nn.Linear`` dispatches to GEMM kernels whose accumulation order can
    change with the number of rows, so evaluating a query subset may differ
    from the corresponding rows of a full batch by roundoff.  Contracting one
    output channel at a time with an elementwise multiply and a fixed-axis
    sum keeps every row's result bitwise independent of the other rows in
    the batch, which is what makes the decoder's query-set independence
    exact rather than merely tight.  Input feature counts here are small
    (pair invariants and a narrow hidden width), so the per-channel loop
    costs only a bounded number of fused elementwise reductions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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
    """

    squared_distance: torch.Tensor  # (Q, S)
    normal_alignment: torch.Tensor  # (Q, S)
    vector_alignments: torch.Tensor  # (Q, S, C)

    @classmethod
    def compute(
        cls,
        query_points: torch.Tensor,
        source_centroids: torch.Tensor,
        source_normals: torch.Tensor,
        source_vectors: torch.Tensor,
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
        displacement = query_points[:, None, :] - source_centroids[None, :, :]
        return cls(
            squared_distance=displacement.square().sum(dim=-1),
            normal_alignment=torch.einsum("qsd,sd->qs", displacement, source_normals),
            # Broadcast multiply-plus-sum rather than
            # ``einsum("qsd,scd->qsc", ...)``: that einsum lowers to a
            # batched GEMM whose per-row reduction can change with the
            # query-chunk shape (measured 1-ulp drift on CUDA), breaking the
            # decoder's bitwise query-set-independence contract.  The
            # ``qsd,sd->qs`` contractions lower to stable mul-plus-sum
            # reductions and may remain einsums.
            vector_alignments=(
                displacement[:, :, None, :] * source_vectors[None, :, :, :]
            ).sum(dim=-1),
        )

    def stacked(self) -> torch.Tensor:
        """Return all invariants stacked as ``(Q, S, 2 + C)`` features."""
        return torch.cat(
            (
                self.squared_distance.unsqueeze(-1),
                self.normal_alignment.unsqueeze(-1),
                self.vector_alignments,
            ),
            dim=-1,
        )


def _segment_double_layer_member(
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
    cell_normals: torch.Tensor,
) -> torch.Tensor:
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
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
    cell_normals: torch.Tensor,
) -> torch.Tensor:
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
    numerator = torch.einsum("qsd,qsd->qs", a, torch.cross(b, c, dim=-1))
    denominator = (
        la * lb * lc
        + torch.einsum("qsd,qsd->qs", a, b) * lc
        + torch.einsum("qsd,qsd->qs", b, c) * la
        + torch.einsum("qsd,qsd->qs", c, a) * lb
    )
    winding_normal = torch.cross(
        panel_vertices[:, 1, :] - panel_vertices[:, 0, :],
        panel_vertices[:, 2, :] - panel_vertices[:, 0, :],
        dim=-1,
    )
    sigma = torch.sign(torch.einsum("sd,sd->s", winding_normal, cell_normals))
    return -sigma[None, :] * 2.0 * torch.atan2(numerator, denominator) / _FOUR_PI


def exact_double_layer_member(
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
    cell_normals: torch.Tensor,
) -> torch.Tensor:
    r"""Dimension-dispatched exact cell integrals of the double layer.

    ``panel_vertices`` has shape ``(S, 2, 2)`` for straight 2D segments or
    ``(S, 3, 3)`` for flat 3D triangles.  Values are similarity invariant
    (angles) and translation invariant, and the measure is included -- do not
    multiply the result by cell weights again.
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
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
) -> torch.Tensor:
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
    dimensionless.  The boundary measure is included -- do not multiply by
    cell weights again.
    """
    start_vector = panel_vertices[None, :, 0, :] - query_points[:, None, :]
    end_vector = panel_vertices[None, :, 1, :] - query_points[:, None, :]
    edge = panel_vertices[:, 1, :] - panel_vertices[:, 0, :]
    length = edge.norm(dim=-1)
    tangent = edge / length[:, None]
    # Local coordinates of the query point relative to the panel: xi_k is the
    # tangential offset from endpoint k and eta the perpendicular offset (its
    # sign cancels in eta * (theta_2 - theta_1), so no orientation enters).
    xi_start = -torch.einsum("qsd,sd->qs", start_vector, tangent)
    xi_end = -torch.einsum("qsd,sd->qs", end_vector, tangent)
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
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
) -> torch.Tensor:
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
    winding is irrelevant by construction.  The boundary measure is included
    -- do not multiply by cell weights again.
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
    height = torch.einsum("qsd,sd->qs", a, unit_normal)
    tiny = torch.finfo(query_points.dtype).tiny
    relative = (a, b, c)
    edge_terms = query_points.new_zeros(a.shape[:2])
    for start, end in ((0, 1), (1, 2), (2, 0)):
        p = relative[start]
        q = relative[end]
        edge = panel_vertices[:, end, :] - panel_vertices[:, start, :]
        edge_tangent = edge / edge.norm(dim=-1, keepdim=True)
        s_start = torch.einsum("qsd,sd->qs", p, edge_tangent)
        s_end = torch.einsum("qsd,sd->qs", q, edge_tangent)
        # Distance from the query point to the edge *line* (includes the
        # out-of-plane component: mu^2 = t^2 + h^2 >= t^2, which is what
        # makes t * asinh(s/mu) vanish smoothly near the edge).
        mu = (p - s_start.unsqueeze(-1) * edge_tangent[None, :, :]).norm(dim=-1)
        mu = mu.clamp_min(tiny)
        in_plane_distance = torch.einsum(
            "qsd,sd->qs", p, torch.cross(edge_tangent, unit_normal, dim=-1)
        )
        edge_terms = edge_terms + in_plane_distance * (
            torch.asinh(s_end / mu) - torch.asinh(s_start / mu)
        )
    numerator = torch.einsum("qsd,qsd->qs", a, torch.cross(b, c, dim=-1))
    denominator = (
        la * lb * lc
        + torch.einsum("qsd,qsd->qs", a, b) * lc
        + torch.einsum("qsd,qsd->qs", b, c) * la
        + torch.einsum("qsd,qsd->qs", c, a) * lb
    )
    solid_angle = 2.0 * torch.atan2(numerator, denominator).abs()
    return (edge_terms - height.abs() * solid_angle) / _FOUR_PI


def exact_single_layer_member(
    query_points: torch.Tensor,
    panel_vertices: torch.Tensor,
) -> torch.Tensor:
    r"""Dimension-dispatched exact cell integrals of the single layer.

    ``panel_vertices`` has shape ``(S, 2, 2)`` for straight 2D segments or
    ``(S, 3, 3)`` for flat 3D triangles.  Entry ``(i, j)`` is the exact
    integral of the free-space Green's function (:math:`-\Delta G=\delta`)
    over cell :math:`j` evaluated at query point :math:`i`:
    :math:`-\log(\lVert x-y\rVert/L_{\mathrm{ref}})/(2\pi)` in 2D and
    :math:`1/(4\pi\lVert x-y\rVert)` in 3D, with the boundary measure
    included -- do not multiply the result by cell weights again.

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


@dataclass(frozen=True)
class KernelDecoderCache:
    r"""Source-side quantities cached by ``encode`` for kernel decoding.

    Everything is expressed in the model's normalized frame.  The cache is
    query independent: it contains per-source geometry (cell vertices,
    centroids, normals, measure), the per-source kernel coefficients
    :math:`C_{jmh}`, the state vectors whose pair alignments feed the smooth
    members, and the projected drive values.
    """

    panel_vertices: torch.Tensor  # (S, vertices, D)
    centroids: torch.Tensor  # (S, D)
    normals: torch.Tensor  # (S, D)
    weights: torch.Tensor  # (S,)
    pair_vectors: torch.Tensor  # (S, C, D)
    coefficients: torch.Tensor  # (S, M, H)
    value_scalars: torch.Tensor  # (S, H, F_s)
    value_vectors: torch.Tensor  # (S, H, F_v, D)
    # Pseudoscalar (0o) value features; ``None`` only for caches built before
    # the pseudo sector existed (equivalent to zero width).
    value_pseudos: torch.Tensor | None = None


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
    mlp_members : int
        Number of learned smooth dictionary members produced by the pair MLP.
    mlp_hidden_dim : int
        Hidden width of the two-layer SiLU pair MLP.
    query_chunk_size : int
        Queries evaluated per dense chunk.  Memory only: every query row is
        computed independently, so chunking never changes the operator.
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
        include_polynomial_members: bool = True,
        include_single_layer_member: bool = False,
        monopole_free_single_layer: bool = False,
        mlp_members: int = 8,
        mlp_hidden_dim: int = 48,
        query_chunk_size: int = 2048,
        accumulation_dtype: torch.dtype | None = torch.float32,
        drive_pseudo_dim: int = 0,
    ) -> None:
        super().__init__()
        if n_spatial_dims not in _EXACT_MEMBER_VERTICES:
            raise ValueError(
                "KernelBasisCrossDecoder requires n_spatial_dims 2 or 3: the "
                "exact double-layer member is dimension-dispatched to segment "
                "and triangle quadrature"
            )
        if not isinstance(include_polynomial_members, bool):
            raise ValueError("include_polynomial_members must be a bool")
        if not isinstance(include_single_layer_member, bool):
            raise ValueError("include_single_layer_member must be a bool")
        if not isinstance(monopole_free_single_layer, bool):
            raise ValueError("monopole_free_single_layer must be a bool")
        if monopole_free_single_layer and not include_single_layer_member:
            raise ValueError(
                "monopole_free_single_layer deflates the exact single-layer "
                "member and therefore requires "
                "include_single_layer_member=True; without that member there "
                "is no monopole to control"
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
        self.include_polynomial_members = include_polynomial_members
        self.include_single_layer_member = include_single_layer_member
        self.monopole_free_single_layer = monopole_free_single_layer
        self.mlp_members = mlp_members
        # Members: exact double layer, exact single layer (optional),
        # polynomial {1, b, a} (optional), learned MLP.
        # ``include_polynomial_members=False`` with ``mlp_members=0`` is the
        # singular-only ablation: the dictionary is the exact double-layer
        # member alone (plus the exact single-layer member when enabled --
        # the two-member "singpair" science arm), and the decoder must still
        # construct and run for those science arms.
        self.n_members = (
            1
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
        self.accumulation_dtype = accumulation_dtype

        pair_features = 2 + self._pair_vector_channels()
        final = _RowStableLinear(mlp_hidden_dim, mlp_members, bias=False)
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
        raise NotImplementedError(
            "instantiate LinearKernelBasisCrossDecoder or "
            "NonlinearZeroKernelBasisCrossDecoder"
        )

    def _source_invariant_channels(self) -> int:
        raise NotImplementedError(
            "instantiate LinearKernelBasisCrossDecoder or "
            "NonlinearZeroKernelBasisCrossDecoder"
        )

    def _kernel_pair_vectors(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _value_includes_vector_invariants(self) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery.
    # ------------------------------------------------------------------
    def build_source_cache(
        self,
        source_mesh: Mesh,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> KernelDecoderCache:
        """Cache every query-independent source quantity once per encode."""
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
        values = self.value_projection(drive_state)
        n = drive_state.n_entities
        return KernelDecoderCache(
            panel_vertices=source_mesh.points[source_mesh.cells],
            centroids=source_mesh.cell_centroids,
            normals=source_mesh.cell_normals,
            weights=source_mesh.cell_areas,
            pair_vectors=self._kernel_pair_vectors(operator_state, drive_state),
            coefficients=self.coefficient_map(
                self._kernel_source_invariants(operator_state, drive_state)
            ).reshape(n, self.n_members, self.heads),
            value_scalars=values.scalars.reshape(n, self.heads, self.scalar_value_dim),
            value_vectors=values.vectors.reshape(
                n, self.heads, self.vector_value_dim, self.n_spatial_dims
            ),
            value_pseudos=values.pseudos.reshape(n, self.heads, self.pseudo_value_dim),
        )

    def _accumulation_type(self, *tensors: torch.Tensor) -> torch.dtype:
        """Promote inputs with a precision floor, never downcast FP64."""
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        if self.accumulation_dtype is not None:
            dtype = torch.promote_types(dtype, self.accumulation_dtype)
        return dtype

    def _evaluate_chunk(
        self,
        query_points: torch.Tensor,
        cache: KernelDecoderCache,
    ) -> ScalarVectorState:
        # Geometry-derived pair quantities keep the input geometry precision
        # even under an ambient autocast scope: the exact member and the
        # invariants are numerical mesh operations, not learned layers.
        device_type = query_points.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            features = PairInvariantFeatures.compute(
                query_points,
                cache.centroids,
                cache.normals,
                cache.pair_vectors,
            )
            singular = exact_double_layer_member(
                query_points, cache.panel_vertices, cache.normals
            ).unsqueeze(-1)
            if self.include_single_layer_member:
                # Second exact singular member: the single layer never reads
                # the cell normals (orientation independent, unlike the
                # sigma-carrying double layer above) and, like the double
                # layer, already includes the boundary measure.
                single_layer = exact_single_layer_member(
                    query_points, cache.panel_vertices
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
                    weights = cache.weights.to(single_layer.dtype)
                    single_layer = (
                        single_layer
                        - single_layer.sum(dim=-1, keepdim=True)
                        * (weights / weights.sum())[None, :]
                    )
                singular = torch.cat(
                    (singular, single_layer.unsqueeze(-1)),
                    dim=-1,
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
        if self.member_mlp is not None:
            learned = self.member_mlp(features.stacked())
            smooth_parts = (
                torch.cat((polynomial.to(learned.dtype), learned), dim=-1)
                if polynomial is not None
                else learned
            )
        else:
            smooth_parts = polynomial
        if smooth_parts is not None:
            # Smooth members use consistent midpoint quadrature (value at the
            # centroid times the cell measure); the singular members are
            # already exact integrals with measure included and are not
            # weighted again.
            smooth = smooth_parts * cache.weights.to(smooth_parts.dtype)[None, :, None]
            members = torch.cat((singular.to(smooth.dtype), smooth), dim=-1)
        else:
            # Singular-only ablation: no smooth members exist, so the member
            # axis carries the exact singular member(s) alone.
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

    def forward(
        self,
        query_points: torch.Tensor,
        cache: KernelDecoderCache,
    ) -> ScalarVectorState:
        """Evaluate the dense pair-kernel message at independent query points."""
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
        n_queries = query_points.shape[0]
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
        outputs = [
            self._evaluate_chunk(
                query_points[start : start + self.query_chunk_size], cache
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
        return self.operator_vector_dim

    def _source_invariant_channels(self) -> int:
        return (
            self.operator_scalar_dim
            + self.operator_vector_dim * (self.operator_vector_dim + 1) // 2
        )

    def _kernel_pair_vectors(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> torch.Tensor:
        return operator_state.vectors

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> torch.Tensor:
        return torch.cat(
            (operator_state.scalars, _gram_invariants(operator_state.vectors)),
            dim=-1,
        )

    def _value_includes_vector_invariants(self) -> bool:
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
        return self.operator_vector_dim + self.drive_vector_dim

    def _source_invariant_channels(self) -> int:
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
    ) -> torch.Tensor:
        return torch.cat((operator_state.vectors, drive_state.vectors), dim=1)

    def _kernel_source_invariants(
        self,
        operator_state: ScalarVectorState,
        drive_state: ScalarVectorState,
    ) -> torch.Tensor:
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
        return True


__all__ = [
    "KernelBasisCrossDecoder",
    "KernelDecoderCache",
    "LinearKernelBasisCrossDecoder",
    "NonlinearZeroKernelBasisCrossDecoder",
    "PairInvariantFeatures",
    "exact_double_layer_member",
    "exact_single_layer_member",
]
