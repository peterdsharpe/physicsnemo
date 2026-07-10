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

r"""A learned boundary-integral-equation model with one kernel used twice.

Hypothesis under test
---------------------

The analytic layer-potential controls showed that, once the propagation
kernel is *correct*, the boundary density solve is nearly trivial to learn
and the model generalizes across boundary-data frequencies.  The generic
dense pair kernel showed the converse: excellent supervised values but no
operator fidelity (large interior Laplacian residual, poor frequency
generalization).  This module tests whether the *factorization* -- not the
analytic kernel -- is what buys operator generalization.

One learned invariant pair kernel :math:`\kappa_\theta` is used twice:

1. **Boundary self-solve.**  With panel measure :math:`w_j` and mean-free
   Dirichlet data :math:`\tilde g`, form the learned trace operator

   .. math::

      (\tilde T\mu)_i = \tfrac12\mu_i + \sum_{j\ne i}
          w_j\,\kappa_\theta(y_i - y_j,\; n_j)\,\mu_j,

   and apply :math:`p` drive-linear Richardson steps

   .. math::

      \mu_{k+1} = \mu_k + \alpha_k\,(\tilde g - \tilde T\mu_k),
      \qquad \mu_0 = \tilde g .

2. **Propagation.**  Evaluate the *same* kernel at interior queries,

   .. math::

      u(x) = \bar g + \sum_j w_j\,\kappa_\theta(x - y_j,\; n_j)\,\mu_j .

The kernel sees only the joint O(2) invariants
:math:`(\lVert r\rVert^2,\; n\cdot r)` of the normalized displacement and
the source normal -- the same feature set as ``InvariantPairKernel`` in
``models.py``, which is therefore the exact ``n_iterations = 0`` control.

Structural notes
----------------

- The diagonal jump constant :math:`\tfrac12` is a *gauge choice*, not a
  physical prior: rescaling :math:`(\beta, \kappa) \to (c\beta, c\kappa)`
  rescales :math:`\mu` by :math:`1/c` and leaves the tied propagation
  invariant, so any nonzero constant is equivalent.  One half matches the
  analytic double-layer convention used by the benchmark controls.
- The diagonal of the learned self-influence is zeroed, mirroring the
  vanishing Cauchy principal value of an odd kernel over a flat self-panel.
- The kernel and relaxations never see the drive, so the map from Dirichlet
  data to potential is exactly linear at fixed geometry: superposition and
  zero-drive behavior hold to floating point.
- Every query point is evaluated independently against the cached density;
  predictions at a point do not depend on which other points are requested.
- ``collocation_residual`` reports the model's self-declared boundary trace
  residual :math:`\tilde T\mu - \tilde g`, the same convention as the
  analytic layer-potential controls.

This is a benchmark-local research prototype, not a proposed public API.
"""

from __future__ import annotations

import math

import torch
from models import (
    _benchmark_boundary,
    _constant_exact_boundary_mean,
    _prediction_mesh,
    _reference_length,
)
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh


def _invariant_kernel_mlp(hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    """Return the two-invariant kernel MLP used by ``InvariantPairKernel``."""

    layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
    for _ in range(hidden_layers - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
    final = nn.Linear(hidden_dim, 1, bias=False)
    nn.init.normal_(final.weight, std=1.0e-2 / math.sqrt(hidden_dim))
    layers.append(final)
    return nn.Sequential(*layers)


def _pair_invariants(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Stack the joint O(2) invariants (|r|^2, n_source . r) per pair."""

    return torch.stack(
        (
            displacement.square().sum(dim=-1),
            torch.einsum("qsd,sd->qs", displacement, normals),
        ),
        dim=-1,
    )


class _MlpPairKernel(nn.Module):
    """Free-form invariant kernel: an MLP of (|r|^2, n . r)."""

    def __init__(self, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.mlp = _invariant_kernel_mlp(hidden_dim, hidden_layers)

    def forward(
        self, displacement: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(_pair_invariants(displacement, normals)).squeeze(-1)


class _HarmonicPairKernel(nn.Module):
    r"""Exactly harmonic pair kernel: a real Laurent series in the panel frame.

    With :math:`b_\parallel = n\cdot r` and the pseudoscalar
    :math:`b_\perp = n\times r`, the complex variable
    :math:`\zeta = b_\parallel + i b_\perp` is antiholomorphic in the query
    coordinate, so for any *real* coefficients

    .. math::

       \kappa(r, n)
       = \sum_{k=1}^{K_s} c_k\,\mathrm{Re}\,\zeta^{-k}
       + \sum_{k=0}^{K_r} d_k\,\mathrm{Re}\,\zeta^{k}

    is exactly harmonic in the query point.  Real coefficients make
    :math:`\kappa` even in :math:`b_\perp`, hence a genuine O(2) joint
    invariant (reflection safe).  The exact interior Dirichlet double-layer
    kernel is the single member :math:`c_1 = 1/(2\pi)`; the basis is the local
    multipole family centred at *each source point*, so there is no global
    expansion centre.  Coincident points (only the zeroed self-influence
    diagonal in practice) return zero.
    """

    def __init__(self, singular_orders: int = 2, regular_orders: int = 2) -> None:
        super().__init__()
        for name, value in (
            ("singular_orders", singular_orders),
            ("regular_orders", regular_orders),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if singular_orders < 1:
            raise ValueError(
                "at least one singular order is required to represent a "
                "decaying boundary kernel"
            )
        self.singular_orders = singular_orders
        self.regular_orders = regular_orders
        n_coefficients = singular_orders + regular_orders + 1
        self.coefficients = nn.Parameter(1.0e-2 * torch.randn(n_coefficients))

    def forward(
        self, displacement: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        parallel = torch.einsum("qsd,sd->qs", displacement, normals)
        perpendicular = (
            displacement[..., 0] * normals[None, :, 1]
            - displacement[..., 1] * normals[None, :, 0]
        )
        zeta = torch.complex(parallel, perpendicular)
        coincident = (parallel == 0.0) & (perpendicular == 0.0)
        safe = torch.where(coincident, torch.ones_like(zeta), zeta)

        coefficients = self.coefficients.to(dtype=parallel.dtype)
        result = torch.zeros_like(parallel)
        inverse = safe.reciprocal()
        power = torch.ones_like(safe)
        for order in range(self.singular_orders):
            power = power * inverse
            result = result + coefficients[order] * power.real
        power = torch.ones_like(safe)
        result = result + coefficients[self.singular_orders] * power.real
        for order in range(self.regular_orders):
            power = power * safe
            result = result + (
                coefficients[self.singular_orders + 1 + order] * power.real
            )
        return torch.where(coincident, torch.zeros_like(result), result)


class SelfConsistentPairKernel(nn.Module):
    r"""Learned BIE: one invariant pair kernel for density solve and decoding.

    Parameters
    ----------
    hidden_dim, hidden_layers : int
        Kernel MLP shape; defaults match ``InvariantPairKernel`` exactly.
    n_iterations : int
        Richardson steps ``p``.  ``0`` disables the self-solve and reduces the
        architecture to ``InvariantPairKernel`` (the factorial control).
    tied : bool
        If ``False``, an independent kernel parameterizes the boundary
        self-solve (ablation isolating tying from iteration).
    query_chunk_size : int
        Queries evaluated per chunk; changes memory only.

    .. warning:: FROZEN LADDER RECORD (2026-07-07, executing the engineering
       review).  This class is the *executable record* of the learned-BIE
       falsification ladder (book chapter "The road to a learned BIE"), kept
       runnable so every refuted configuration stays reproducible.  Its
       options are NOT recommendations: ``trace_loss``, ``kernel_pde_loss``,
       and ``tied=False`` each reproduce a configuration the ladder
       *refuted*, and the midpoint-quadrature ``kernel_family`` variants
       lose to exact panel quadrature (:class:`HarmonicPanelBIE`) by an
       order of magnitude near boundaries.  For anything other than
       reproducing the ladder, use :class:`HarmonicPanelBIE` (Dirichlet) or
       :class:`NeumannHarmonicPanelBIE` (Neumann).  The module is
       maintenance-frozen: bug fixes only, no new options.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        n_iterations: int = 8,
        tied: bool = True,
        initial_relaxation: float = 1.0,
        query_chunk_size: int = 1024,
        kernel_family: str = "mlp",
        trace_loss: bool = False,
        kernel_pde_loss: bool = False,
        pde_pair_samples: int = 512,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("hidden_dim", hidden_dim, 1),
            ("hidden_layers", hidden_layers, 1),
            ("n_iterations", n_iterations, 0),
            ("query_chunk_size", query_chunk_size, 1),
            ("pde_pair_samples", pde_pair_samples, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(initial_relaxation):
            raise ValueError("initial_relaxation must be finite")
        if trace_loss and not n_iterations:
            raise ValueError(
                "trace_loss requires n_iterations >= 1: without a self-solve, "
                "the trace equation has no learned density to constrain"
            )
        if kernel_family not in ("mlp", "harmonic"):
            raise ValueError("kernel_family must be 'mlp' or 'harmonic'")
        if kernel_family == "harmonic" and kernel_pde_loss:
            raise ValueError(
                "kernel_pde_loss is redundant for the harmonic family: the "
                "kernel is exactly harmonic by construction"
            )

        def _make_kernel() -> nn.Module:
            if kernel_family == "harmonic":
                return _HarmonicPairKernel()
            return _MlpPairKernel(hidden_dim, hidden_layers)

        self.kernel_family = kernel_family
        self.kernel = _make_kernel()
        self.solve_kernel = self.kernel if tied else _make_kernel()
        self.tied = tied
        self.n_iterations = n_iterations
        if n_iterations:
            self.relaxation = nn.Parameter(
                torch.full((n_iterations,), float(initial_relaxation))
            )
        else:
            self.register_parameter("relaxation", None)
        self.query_chunk_size = query_chunk_size
        self.trace_loss = trace_loss
        self.kernel_pde_loss = kernel_pde_loss
        self.pde_pair_samples = pde_pair_samples

    @staticmethod
    def _normalized_geometry(
        domain: DomainMesh,
    ) -> tuple[Mesh, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boundary plus dimensionless weights, frame, and sources."""

        boundary = _benchmark_boundary(domain)
        length = _reference_length(domain)
        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            total_measure = weights.sum()
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / total_measure
            source_points = (boundary.cell_centroids - center) / length
        return boundary, weights, center, length, source_points

    def _self_influence(
        self,
        source_points: torch.Tensor,
        normals: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return the measure-weighted boundary self-influence ``K̃``."""

        displacement = source_points[:, None, :] - source_points[None, :, :]
        influence = self.solve_kernel(displacement, normals)
        # Zero self-influence mirrors the vanishing flat-panel principal
        # value; the analytic interior jump enters as the explicit 1/2.
        influence = influence * (
            1.0
            - torch.eye(
                influence.shape[0],
                device=influence.device,
                dtype=influence.dtype,
            )
        )
        return influence * weights[None, :]

    def _solve_density(
        self,
        source_points: torch.Tensor,
        normals: torch.Tensor,
        weights: torch.Tensor,
        residual_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return the Richardson density; exactly linear in the drive."""

        if not self.n_iterations:
            return residual_values
        influence = self._self_influence(source_points, normals, weights)
        density = residual_values
        relaxation = self.relaxation.to(dtype=residual_values.dtype)
        for step in relaxation.unbind():
            trace = 0.5 * density + influence @ density
            density = density + step * (residual_values - trace)
        return density

    def _boundary_state(
        self, domain: DomainMesh
    ) -> tuple[
        Mesh,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        boundary, weights, center, length, source_points = self._normalized_geometry(
            domain
        )
        normals = boundary.cell_normals
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual_values = values - mean
        density = self._solve_density(source_points, normals, weights, residual_values)
        return boundary, weights, center, length, source_points, mean, density

    def forward(self, domain: DomainMesh) -> Mesh:
        (
            boundary,
            weights,
            center,
            length,
            source_points,
            mean,
            density,
        ) = self._boundary_state(domain)
        normals = boundary.cell_normals
        query_points = (domain.interior.points - center) / length

        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            query = query_points[start : start + self.query_chunk_size]
            displacement = query[:, None, :] - source_points[None, :, :]
            pair_kernel = self.kernel(displacement, normals)
            chunks.append(
                mean + torch.einsum("qs,s,s->q", pair_kernel, weights, density)
            )

        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)

    def collocation_residual(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the self-declared trace residual :math:`\tilde T\mu-\tilde g`.

        With the analytic constant lift, this equals the residual of the full
        model trace against the prescribed boundary values.  When
        ``n_iterations == 0`` there is no self-declared trace operator in the
        architecture; the tied kernel's trace operator is still reported so
        the diagnostic remains defined.
        """

        boundary, weights, _, _, source_points, mean, density = self._boundary_state(
            domain
        )
        normals = boundary.cell_normals
        values = boundary.cell_data["boundary_value"]
        residual_values = values - mean

        influence = self._self_influence(source_points, normals, weights)
        trace = 0.5 * density + influence @ density
        return trace - residual_values

    def _kernel_laplacian_penalty(
        self,
        query_points: torch.Tensor,
        source_points: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return the scale-free kernel harmonicity residual on sampled pairs.

        For :math:`\kappa(a, b)` with invariants :math:`a=\lVert r\rVert^2` and
        :math:`b=n\cdot r`, the ambient 2D Laplacian in the query coordinate is

        .. math::

           \nabla_x^2\kappa
           = 4a\,\kappa_{aa} + 4b\,\kappa_{ab} + \kappa_{bb} + 4\kappa_a .

        The penalty is ``mean((a * laplacian)^2)``: the conformal weight
        ``a = |r|^2`` makes the residual dimensionless relative to a
        ``1/|r|``-type kernel and keeps near-singular pairs from dominating.
        Pairs are subsampled from the actual query--source displacements of the
        current case, so no sampling length scale is introduced.
        """

        displacement = (query_points[:, None, :] - source_points[None, :, :]).reshape(
            -1, 2
        )
        pair_normals = (
            normals[None, :, :].expand(query_points.shape[0], -1, -1).reshape(-1, 2)
        )
        n_pairs = displacement.shape[0]
        if n_pairs > self.pde_pair_samples:
            index = torch.randint(
                n_pairs, (self.pde_pair_samples,), device=displacement.device
            )
            displacement = displacement[index]
            pair_normals = pair_normals[index]

        features = torch.stack(
            (
                displacement.square().sum(dim=-1),
                torch.einsum("pd,pd->p", displacement, pair_normals),
            ),
            dim=-1,
        ).requires_grad_(True)
        kappa = self.kernel.mlp(features).squeeze(-1)
        (first,) = torch.autograd.grad(kappa.sum(), features, create_graph=True)
        kappa_a = first[..., 0]
        (second_a,) = torch.autograd.grad(kappa_a.sum(), features, create_graph=True)
        (second_b,) = torch.autograd.grad(
            first[..., 1].sum(), features, create_graph=True
        )
        a = features[..., 0].detach()
        b = features[..., 1].detach()
        laplacian = (
            4.0 * a * second_a[..., 0]
            + 4.0 * b * second_a[..., 1]
            + second_b[..., 1]
            + 4.0 * kappa_a
        )
        return (a * laplacian).square().mean()

    def auxiliary_loss(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the enabled dimensionless structural penalties.

        - ``trace_loss``: relative measure-weighted trace residual
          :math:`\sum w(\tilde T\mu-\tilde g)^2 / \sum w\tilde g^2` -- the
          model's own boundary condition, enforced through its self-declared
          trace operator.
        - ``kernel_pde_loss``: the harmonicity residual of the propagation
          kernel itself.  This uses only the PDE operator (which defines the
          problem), never its Green function; a harmonic kernel makes the
          entire predicted field harmonic for any density, after which the
          maximum principle bounds interior error by boundary-trace error.
        """

        terms: list[torch.Tensor] = []
        if self.trace_loss:
            boundary, weights, _, _, source_points, mean, density = (
                self._boundary_state(domain)
            )
            values = boundary.cell_data["boundary_value"]
            residual_values = values - mean
            influence = self._self_influence(
                source_points, boundary.cell_normals, weights
            )
            trace_residual = 0.5 * density + influence @ density - residual_values
            terms.append(
                torch.sum(weights * trace_residual.square())
                / torch.sum(weights * residual_values.square()).clamp_min(1.0e-30)
            )
        if self.kernel_pde_loss:
            boundary, weights, center, length, source_points = (
                self._normalized_geometry(domain)
            )
            query_points = (domain.interior.points - center) / length
            terms.append(
                self._kernel_laplacian_penalty(
                    query_points, source_points, boundary.cell_normals
                )
            )
        if not terms:
            raise RuntimeError(
                "auxiliary_loss called with neither trace_loss nor "
                "kernel_pde_loss enabled"
            )
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total


class HarmonicPanelBIE(nn.Module):
    r"""Learned harmonic layer potential with exact straight-panel quadrature.

    The pointwise kernel is the admissible harmonic family

    .. math::

       \kappa(r, n) = c_1\,\mathrm{Re}\,\zeta^{-1}
           + \sum_{k=0}^{K} d_k\,\mathrm{Re}\,\zeta^{k},
       \qquad \zeta = n\cdot r + i\,n\times r,

    i.e. a learned multiple of the double-layer singularity plus entire
    harmonic corrections.  Hypersingular orders (:math:`\zeta^{-k}`,
    :math:`k\ge2`) are excluded because their flat-panel principal value does
    not exist.  On a straight panel :math:`\zeta` is affine in arclength with
    unit slope, so every term integrates in closed form:

    .. math::

       \int_{\text{panel}} \mathrm{Re}\,F(\zeta)\,ds
           = -\sigma\,\mathrm{Im}\left[G(\zeta_{\text{end}})
           - G(\zeta_{\text{start}})\right],
       \qquad G' = F,\ \sigma = n\times\tau ,

    which for the singular term reduces to the signed subtended angle
    :math:`c_1\,\mathrm{atan2}(\mathrm{cross},\mathrm{dot})` used by the
    analytic controls.  There is no near-boundary midpoint-quadrature error:
    the exact analytic Richardson control is the single member
    :math:`c_1 = -1/(2\pi),\ d = 0` of this family, so the oracle is inside
    the learnable set.

    Density solve and constant lift follow ``SelfConsistentPairKernel``:
    :math:`p` drive-linear Richardson steps through
    :math:`\tilde T = \tfrac12 I + \tilde K` where :math:`\tilde K` collects
    the exact panel integrals at panel midpoints (flat-panel principal value
    of the singular term is zero; the entire terms use the same closed form).
    The map from Dirichlet data to potential is exactly linear at fixed
    geometry, queries never interact, and all features are joint O(2)
    invariants of relative geometry.

    DEFAULTS (changed 2026-07-07, executing the engineering review's
    "evidence-orphaned defaults" item): the pruning study proved
    ``regular_orders=0`` with a single shared relaxation lossless (identical
    to four decimals) and *better* than the 13-parameter original, so the
    pruned 3-parameter configuration is now the default.  The archived
    13-parameter arm remains reproducible by name: ``train.py`` pins
    ``harmonic_panel_bie`` to the explicit historical configuration
    (``regular_orders=3, shared_relaxation=False``), and every archived run
    records its full config either way.
    """

    def __init__(
        self,
        *,
        regular_orders: int = 0,
        n_iterations: int = 8,
        initial_relaxation: float = 1.0,
        shared_relaxation: bool = True,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("regular_orders", regular_orders, 0),
            ("n_iterations", n_iterations, 0),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(initial_relaxation):
            raise ValueError("initial_relaxation must be finite")
        self.regular_orders = regular_orders
        self.n_iterations = n_iterations
        self.shared_relaxation = shared_relaxation
        self.singular_coefficient = nn.Parameter(1.0e-2 * torch.randn(()))
        self.regular_coefficients = nn.Parameter(
            1.0e-2 * torch.randn(regular_orders + 1)
        )
        if n_iterations:
            # Every converged run learned a uniform per-step relaxation, so a
            # single shared scalar is the ablation-motivated default candidate.
            shape = () if shared_relaxation else (n_iterations,)
            self.relaxation = nn.Parameter(torch.full(shape, float(initial_relaxation)))
        else:
            self.register_parameter("relaxation", None)
        self.query_chunk_size = query_chunk_size

    def _influence(
        self,
        query_points: torch.Tensor,
        panel_start: torch.Tensor,
        panel_end: torch.Tensor,
        normals: torch.Tensor,
        *,
        zero_singular_diagonal: bool = False,
    ) -> torch.Tensor:
        """Return exact panel-integrated influences with measure included."""

        start_vector = panel_start.unsqueeze(0) - query_points.unsqueeze(1)
        end_vector = panel_end.unsqueeze(0) - query_points.unsqueeze(1)
        cross = (
            start_vector[..., 0] * end_vector[..., 1]
            - start_vector[..., 1] * end_vector[..., 0]
        )
        dot = torch.sum(start_vector * end_vector, dim=-1)
        tangent = panel_end - panel_start
        tangent = tangent / tangent.norm(dim=-1, keepdim=True)
        sigma = (
            normals[:, 0] * tangent[:, 1] - normals[:, 1] * tangent[:, 0]
        )  # n x tau = +-1 per panel; makes every term odd in n as required
        subtended = -sigma[None, :] * torch.atan2(cross, dot)
        if zero_singular_diagonal:
            subtended = subtended * (
                1.0
                - torch.eye(
                    subtended.shape[0],
                    device=subtended.device,
                    dtype=subtended.dtype,
                )
            )
        coefficients = self.regular_coefficients.to(dtype=query_points.dtype)
        result = self.singular_coefficient.to(dtype=query_points.dtype) * subtended

        def zeta_of(displacement_to_query: torch.Tensor) -> torch.Tensor:
            parallel = torch.einsum("qsd,sd->qs", -displacement_to_query, normals)
            perpendicular = -(
                displacement_to_query[..., 0] * normals[None, :, 1]
                - displacement_to_query[..., 1] * normals[None, :, 0]
            )
            return torch.complex(parallel, perpendicular)

        # zeta = n.(x - y) + i n x (x - y); the vectors above are (y - x).
        zeta_start = zeta_of(start_vector)
        zeta_end = zeta_of(end_vector)
        power_start = torch.ones_like(zeta_start)
        power_end = torch.ones_like(zeta_end)
        for order in range(self.regular_orders + 1):
            power_start = power_start * zeta_start
            power_end = power_end * zeta_end
            antiderivative = (power_end - power_start) / (order + 1)
            result = result + coefficients[order] * (
                sigma[None, :] * antiderivative.imag
            )
        return result

    @staticmethod
    def _normalized_panels(
        domain: DomainMesh,
    ) -> tuple[
        Mesh, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        boundary = _benchmark_boundary(domain)
        length = _reference_length(domain)
        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            total_measure = weights.sum()
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / total_measure
            vertices = boundary.points[boundary.cells]
            panel_start = (vertices[:, 0] - center) / length
            panel_end = (vertices[:, 1] - center) / length
            midpoints = (boundary.cell_centroids - center) / length
        return boundary, weights, center, length, panel_start, panel_end, midpoints

    def _boundary_state(
        self, domain: DomainMesh
    ) -> tuple[
        Mesh,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            boundary,
            weights,
            center,
            length,
            panel_start,
            panel_end,
            midpoints,
        ) = self._normalized_panels(domain)
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual_values = values - mean

        density = residual_values
        if self.n_iterations:
            influence = self._influence(
                midpoints,
                panel_start,
                panel_end,
                boundary.cell_normals,
                zero_singular_diagonal=True,
            )
            relaxation = self.relaxation.to(dtype=residual_values.dtype)
            if self.shared_relaxation:
                relaxation = relaxation.expand(self.n_iterations)
            for step in relaxation.unbind():
                trace = 0.5 * density + influence @ density
                density = density + step * (residual_values - trace)
        return boundary, center, length, panel_start, panel_end, mean, density

    def forward(self, domain: DomainMesh) -> Mesh:
        (
            boundary,
            center,
            length,
            panel_start,
            panel_end,
            mean,
            density,
        ) = self._boundary_state(domain)
        query_points = (domain.interior.points - center) / length

        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            influence = self._influence(
                query_points[start : start + self.query_chunk_size],
                panel_start,
                panel_end,
                boundary.cell_normals,
            )
            chunks.append(mean + influence @ density)
        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)

    def collocation_residual(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the self-declared trace residual (see the class caveat).

        For a converged Richardson solve this residual is small by
        construction regardless of the learned coefficients; it certifies the
        solve, and certifies boundary fidelity only insofar as the imposed
        ``1/2`` jump matches the kernel's actual jump (exactly the
        self-consistent point :math:`c_1 = -1/(2\pi)`).
        """

        (
            boundary,
            _,
            _,
            panel_start,
            panel_end,
            mean,
            density,
        ) = self._boundary_state(domain)
        _, _, _, _, _, _, midpoints = self._normalized_panels(domain)
        values = boundary.cell_data["boundary_value"]
        residual_values = values - mean
        influence = self._influence(
            midpoints,
            panel_start,
            panel_end,
            boundary.cell_normals,
            zero_singular_diagonal=True,
        )
        return 0.5 * density + influence @ density - residual_values


def _neumann_benchmark_boundary(domain: DomainMesh) -> Mesh:
    """Return the sole benchmark boundary, requiring Neumann flux data."""

    if set(domain.boundaries.keys()) != {"dirichlet"}:
        raise ValueError("benchmark domains must contain only a 'dirichlet' boundary")
    boundary = domain.boundaries["dirichlet"]
    if "boundary_flux" not in boundary.cell_data:
        raise ValueError(
            "Neumann models require cell_data['boundary_flux']; generate "
            "samples with the Neumann problem (train.py --problem neumann)"
        )
    return boundary


class NeumannHarmonicPanelBIE(nn.Module):
    r"""Learned Neumann layer potential with exact straight-panel quadrature.

    The pointwise kernel extends the Dirichlet family of
    :class:`HarmonicPanelBIE` with single-layer content,

    .. math::

       \kappa(r, n) = c_0\,\mathrm{Re}\log\zeta
           + c_1\,\mathrm{Re}\,\zeta^{-1}
           + \sum_{k=0}^{K} d_k\,\mathrm{Re}\,\zeta^{k},
       \qquad \zeta = n\cdot r + i\,n\times r .

    ``Re log zeta = log|zeta| = log|r|`` is harmonic away from the source; the
    normalized frame (coordinates divided by the reference length) makes the
    log argument dimensionless, so the term is O(2)+similarity legitimate.
    Every term integrates over a straight panel in closed form through
    :math:`\int \mathrm{Re}\,F(\zeta)\,ds = \sigma\,\mathrm{Im}[\Delta G]`
    with :math:`G' = F` and :math:`\sigma = n\times\tau` (in this file's
    frame :math:`d\zeta/ds = +i\sigma`, matching the entire-order terms of
    :class:`HarmonicPanelBIE`).  For the log term
    :math:`G = \zeta\log\zeta - \zeta`; because the panel's :math:`\zeta`-path
    is a vertical segment that may cross the principal branch cut, the
    imaginary part is evaluated through the split
    :math:`\mathrm{Im}[\zeta\log\zeta]
    = \mathrm{Re}\,\zeta\,\arg\zeta + \mathrm{Im}\,\zeta\,\log|\zeta|`
    with the *continuous* argument change
    :math:`\Delta\theta = \arg(\zeta_{\text{end}}\bar\zeta_{\text{start}})`
    (a straight segment not through the origin subtends less than
    :math:`\pi`), which is verified against brute-force quadrature in the
    tests.  The same closed form evaluates the finite self-panel log integral
    :math:`\int \log|\zeta|\,ds = L(\log(L/2) - 1)` exactly, because the
    :math:`\mathrm{Re}\,\zeta\cdot\Delta\theta` term carries the factor
    :math:`\mathrm{Re}\,\zeta = n\cdot r = 0` on the panel itself.

    Density solve
    -------------
    ``p`` drive-linear Richardson steps on the model's own Neumann trace
    operator :math:`\tilde T = -\tfrac12 I + \tilde K'`, where
    :math:`\tilde K'_{ij} = n_i\cdot\nabla_{x_i}\,I_j(x_i)` is the normal
    derivative of the exact panel-integrated influence at panel midpoints,
    obtained by automatic differentiation of the influence in the query point
    (exact, no new quadrature), with the diagonal zeroed to mirror the
    vanishing flat-panel principal value of the single-layer normal
    derivative.  The ``+1/2 I - K`` orientation is a documented gauge convention chosen so
    Richardson relaxation is convergent near zero kernel (matching the
    Dirichlet family); it flips the self-consistent oracle sign,
    not a physical prior: with this sign the analytic interior Neumann solve
    sits at :math:`c_0 = +1/(2\pi)` (kernel :math:`+\log|r|/(2\pi)`), for
    which :math:`\tilde T = -(\tfrac12 I + K'_{\mathrm{analytic}})` and the
    propagated field equals the classical single-layer solution
    :math:`S[\sigma_{\mathrm{analytic}}]` with the internal density flipped in
    sign.  Flipping both the jump sign and :math:`c_0` yields the identical
    operator; the oracle test pins this convention against the analytic disk
    solution.  The flux data are made discretely compatible (panel-measure
    weighted mean removed) as an exact linear operation before the solve; the
    generator already guarantees this, so it is defensive only.

    Gauge
    -----
    Neumann data determine the potential only up to a constant.  The model
    reports the potential relative to its own boundary mean:
    :math:`u(x) = u_{\mathrm{raw}}(x) - \langle u_{\mathrm{raw}}
    \rangle_{\partial\Omega}` with the boundary mean computed from the exact
    panel-integrated trace at panel midpoints (singular-term diagonal zeroed
    -- the flat-panel principal value -- log and entire diagonals by their
    exact finite self-panel integrals).  This is a boundary integral of the
    density alone, so query points never interact and predictions at a point
    do not depend on which other points are requested.  It matches the
    generator's gauge-fixed targets ``u - mean_boundary(u)``.

    Exact contracts: linearity in the flux data (kernel and relaxations never
    see the drive), joint O(2)+similarity invariance, exact query-set
    independence, and harmonicity of the propagated field by construction.
    There is no constant lift: constants are gauged out on both sides.
    """

    def __init__(
        self,
        *,
        regular_orders: int = 3,
        n_iterations: int = 8,
        initial_relaxation: float = 1.0,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("regular_orders", regular_orders, 0),
            ("n_iterations", n_iterations, 0),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(initial_relaxation):
            raise ValueError("initial_relaxation must be finite")
        self.regular_orders = regular_orders
        self.n_iterations = n_iterations
        self.log_coefficient = nn.Parameter(1.0e-2 * torch.randn(()))
        self.singular_coefficient = nn.Parameter(1.0e-2 * torch.randn(()))
        self.regular_coefficients = nn.Parameter(
            1.0e-2 * torch.randn(regular_orders + 1)
        )
        if n_iterations:
            self.relaxation = nn.Parameter(
                torch.full((n_iterations,), float(initial_relaxation))
            )
        else:
            self.register_parameter("relaxation", None)
        self.query_chunk_size = query_chunk_size

    def _influence(
        self,
        query_points: torch.Tensor,
        panel_start: torch.Tensor,
        panel_end: torch.Tensor,
        normals: torch.Tensor,
        *,
        zero_singular_diagonal: bool = False,
        zero_diagonal: bool = False,
    ) -> torch.Tensor:
        """Return exact panel-integrated influences with measure included.

        ``zero_singular_diagonal`` zeroes only the subtended-angle term on the
        diagonal (the flat-panel principal value), keeping the finite log and
        entire self-panel integrals: this is the potential-trace convention.
        ``zero_diagonal`` zeroes every term on the diagonal: this is the
        flux-trace convention, applied before differentiation in the query.
        """

        start_vector = panel_start.unsqueeze(0) - query_points.unsqueeze(1)
        end_vector = panel_end.unsqueeze(0) - query_points.unsqueeze(1)
        cross = (
            start_vector[..., 0] * end_vector[..., 1]
            - start_vector[..., 1] * end_vector[..., 0]
        )
        dot = torch.sum(start_vector * end_vector, dim=-1)
        tangent = panel_end - panel_start
        tangent = tangent / tangent.norm(dim=-1, keepdim=True)
        sigma = (
            normals[:, 0] * tangent[:, 1] - normals[:, 1] * tangent[:, 0]
        )  # n x tau = +-1 per panel; makes every term odd in n as required

        def eye_mask(reference: torch.Tensor) -> torch.Tensor:
            return 1.0 - torch.eye(
                reference.shape[0],
                device=reference.device,
                dtype=reference.dtype,
            )

        subtended = -sigma[None, :] * torch.atan2(cross, dot)
        if zero_singular_diagonal:
            subtended = subtended * eye_mask(subtended)

        def zeta_of(displacement_to_query: torch.Tensor) -> torch.Tensor:
            parallel = torch.einsum("qsd,sd->qs", -displacement_to_query, normals)
            perpendicular = -(
                displacement_to_query[..., 0] * normals[None, :, 1]
                - displacement_to_query[..., 1] * normals[None, :, 0]
            )
            return torch.complex(parallel, perpendicular)

        # zeta = n.(x - y) - i n x (x - y); the vectors above are (y - x).
        # This is the conjugate of the class docstring's frame; every kernel
        # term (log|zeta|, Re zeta^k) is conjugation-even, so pointwise values
        # agree, and the +i sigma path direction below fixes the
        # antiderivative signs consistently.
        zeta_start = zeta_of(start_vector)
        zeta_end = zeta_of(end_vector)

        # Log term: sigma * Im[Delta(zeta log zeta - zeta)] with the
        # continuous branch.  In this file's frame zeta = n.r - i (n x r), so
        # d(zeta)/ds = +i sigma along the panel and the closed form carries
        # +sigma, exactly like the entire orders below.  Along the panel
        # Re(zeta) is constant and the zeta-path is a straight vertical
        # segment not through the origin, so the continuous argument change
        # is the principal arg of zeta_end * conj(zeta_start) (magnitude
        # below pi).  Splitting
        # Im[zeta log zeta] = Re(zeta) arg(zeta) + Im(zeta) log|zeta|
        # avoids the principal-branch jump of log across the negative real
        # axis.  The (Re zeta_end - Re zeta_start) term vanishes analytically
        # and only cancels floating-point noise in the constant real part.
        # On the self panel Re(zeta) = 0, so the arg terms drop out and the
        # same expression evaluates the finite self integral
        # L (log(L/2) - 1) exactly.
        theta_start = torch.atan2(zeta_start.imag, zeta_start.real)
        relative = zeta_end * zeta_start.conj()
        delta_theta = torch.atan2(relative.imag, relative.real)
        imag_delta_antiderivative = (
            (zeta_end.real - zeta_start.real) * theta_start
            + zeta_end.real * delta_theta
            + zeta_end.imag * torch.log(zeta_end.abs())
            - zeta_start.imag * torch.log(zeta_start.abs())
            - (zeta_end.imag - zeta_start.imag)
        )
        log_term = sigma[None, :] * imag_delta_antiderivative

        result = (
            self.log_coefficient.to(dtype=query_points.dtype) * log_term
            + self.singular_coefficient.to(dtype=query_points.dtype) * subtended
        )
        coefficients = self.regular_coefficients.to(dtype=query_points.dtype)
        power_start = torch.ones_like(zeta_start)
        power_end = torch.ones_like(zeta_end)
        for order in range(self.regular_orders + 1):
            power_start = power_start * zeta_start
            power_end = power_end * zeta_end
            antiderivative = (power_end - power_start) / (order + 1)
            result = result + coefficients[order] * (
                sigma[None, :] * antiderivative.imag
            )
        if zero_diagonal:
            result = result * eye_mask(result)
        return result

    @staticmethod
    def _normalized_panels(
        domain: DomainMesh,
    ) -> tuple[
        Mesh, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        boundary = _neumann_benchmark_boundary(domain)
        length = _reference_length(domain)
        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            total_measure = weights.sum()
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / total_measure
            vertices = boundary.points[boundary.cells]
            panel_start = (vertices[:, 0] - center) / length
            panel_end = (vertices[:, 1] - center) / length
            midpoints = (boundary.cell_centroids - center) / length
        return boundary, weights, center, length, panel_start, panel_end, midpoints

    def _apply_flux_trace(
        self,
        midpoints: torch.Tensor,
        panel_start: torch.Tensor,
        panel_end: torch.Tensor,
        normals: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return :math:`\tilde K'\sigma` by differentiating the influence.

        Row ``i`` of the diagonal-zeroed influence depends only on query
        ``i``, so one gradient of ``sum_i (I sigma)_i`` with respect to fresh
        leaf queries yields every row's query-gradient at once.  The fresh
        leaf keeps earlier Richardson iterates (which may depend on previous
        trace applications) out of the differentiation, and
        ``create_graph`` follows the ambient grad mode so training
        backpropagates through the operator.
        """

        create_graph = torch.is_grad_enabled()
        queries = midpoints.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            influence = self._influence(
                queries, panel_start, panel_end, normals, zero_diagonal=True
            )
            (gradient,) = torch.autograd.grad(
                (influence @ density).sum(),
                queries,
                create_graph=create_graph,
            )
        return torch.einsum("qd,qd->q", gradient, normals)

    def _flux_trace_matrix(
        self,
        midpoints: torch.Tensor,
        panel_start: torch.Tensor,
        panel_end: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        r"""Materialize :math:`\tilde K'` column by column (diagnostics only)."""

        queries = midpoints.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            influence = self._influence(
                queries, panel_start, panel_end, normals, zero_diagonal=True
            )
            columns = []
            for panel in range(influence.shape[1]):
                (gradient,) = torch.autograd.grad(
                    influence[:, panel].sum(), queries, retain_graph=True
                )
                columns.append(torch.einsum("qd,qd->q", gradient, normals))
        return torch.stack(columns, dim=-1)

    def _boundary_state(
        self, domain: DomainMesh
    ) -> tuple[
        Mesh,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            boundary,
            weights,
            center,
            length,
            panel_start,
            panel_end,
            midpoints,
        ) = self._normalized_panels(domain)
        # Physical flux carries units 1/length; the normalized-frame datum is
        # length * flux, which is what makes the output similarity invariant.
        flux = boundary.cell_data["boundary_flux"] * length
        # Defensive exact-linear compatibility correction; the generator
        # already removes the discrete deficit, so this is a numerical no-op.
        flux = flux - torch.sum(weights * flux) / weights.sum()

        density = flux
        if self.n_iterations:
            normals = boundary.cell_normals
            relaxation = self.relaxation.to(dtype=flux.dtype)
            for step in relaxation.unbind():
                trace = 0.5 * density + self._apply_flux_trace(
                    midpoints, panel_start, panel_end, normals, density
                )
                density = density + step * (flux - trace)
        return (
            boundary,
            weights,
            center,
            length,
            panel_start,
            panel_end,
            midpoints,
            flux,
            density,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        (
            boundary,
            weights,
            center,
            length,
            panel_start,
            panel_end,
            midpoints,
            _,
            density,
        ) = self._boundary_state(domain)
        normals = boundary.cell_normals

        # Boundary mean of the raw potential, via the exact panel-integrated
        # trace at midpoints (finite log/entire self-panel integrals kept,
        # singular principal value zeroed).  A pure boundary functional of the
        # density: the gauge cannot couple query points.
        trace_influence = self._influence(
            midpoints, panel_start, panel_end, normals, zero_singular_diagonal=True
        )
        gauge = torch.sum(weights * (trace_influence @ density)) / weights.sum()

        query_points = (domain.interior.points - center) / length
        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            influence = self._influence(
                query_points[start : start + self.query_chunk_size],
                panel_start,
                panel_end,
                normals,
            )
            # einsum keeps the per-query reduction independent of how many
            # queries share the batch (the influence tensor is a strided
            # complex-view composite, where mm kernel selection would
            # otherwise vary with the batch shape).
            chunks.append(torch.einsum("qs,s->q", influence, density) - gauge)
        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)

    def collocation_residual(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the self-declared flux-trace residual :math:`\tilde T\sigma - \tilde f`.

        The residual is expressed in the normalized frame (flux times
        reference length).  As for the Dirichlet sibling, a converged
        Richardson solve makes this small regardless of the learned
        coefficients; it certifies the solve, and certifies boundary fidelity
        only insofar as the imposed ``+1/2 I - K`` orientation matches the kernel's actual
        jump (exactly the self-consistent point ``c0 = 1/(2 pi)``).
        """

        (
            boundary,
            _,
            _,
            _,
            panel_start,
            panel_end,
            midpoints,
            flux,
            density,
        ) = self._boundary_state(domain)
        trace = 0.5 * density + self._apply_flux_trace(
            midpoints, panel_start, panel_end, boundary.cell_normals, density
        )
        return trace - flux


__all__ = [
    "HarmonicPanelBIE",
    "NeumannHarmonicPanelBIE",
    "SelfConsistentPairKernel",
]
