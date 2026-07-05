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

r"""PDE-conforming controls for the two-dimensional Laplace benchmark.

This module deliberately lives in the example rather than in PhysicsNeMo's
generic layer library.  It represents a harmonic function by the sign-normalized
double-layer potential

.. math::

    (Q\mu)(x) = \int_{\partial\Omega}
        \frac{n_y\mathbin{\cdot}(y-x)}{2\pi\lVert y-x\rVert^2}
        \mu(y)\,ds_y.

For an outward-oriented, smooth closed boundary its interior trace is
``(1/2 I + K) mu``.  Consequently, the interior Dirichlet density is obtained
from the second-kind equation

.. math::

    (\tfrac12 I + K)\mu = g.

The sign convention makes ``Q[1] = 1`` in the interior.  Unlike a 2D
single-layer potential, this representation needs neither a logarithmic
reference length nor a constant-mode gauge.

The benchmark boundary consists of oriented straight panels with piecewise
constant data.  Rather than sample the nearly singular kernel at panel
midpoints, :func:`double_layer_influence` analytically integrates each panel:
its contribution is the signed angle subtended at the query divided by
``2*pi``.  The cell measure is therefore included exactly, including in the
near-boundary regime.  Boundary collocation uses panel midpoints, the exact
off-diagonal panel integrals, the Cauchy principal value zero on each flat
self-panel, and the analytic ``1/2`` interior jump.

All controls are exactly linear in the prescribed boundary values.  Their
geometry uses only relative vectors and signed angles, so a consistently
oriented mesh gives exact translation, positive-scale, rotation, and reflection
invariance up to floating-point arithmetic.  The query evaluation is chunked;
collocation assembly/storage are quadratic in panel count and the dense solve
is cubic, so the solved-density path is intended only as a diagnostic control.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

if TYPE_CHECKING:
    from models import MeshTransformerConfig

_TWO_PI: Final[float] = 2.0 * math.pi


def _validate_boundary(boundary: Mesh) -> None:
    if boundary.n_spatial_dims != 2 or boundary.n_manifold_dims != 1:
        raise ValueError(
            "the Laplace double-layer control requires oriented line panels in 2D"
        )
    if boundary.n_cells < 3:
        raise ValueError("a closed boundary requires at least three panels")
    if not torch.compiler.is_compiling():
        areas = boundary.cell_areas
        if torch.any(~torch.isfinite(areas) | (areas <= 0.0)).item():
            raise ValueError("all boundary panels must have finite positive measure")


def _validate_queries(boundary: Mesh, query_points: torch.Tensor) -> None:
    if query_points.ndim != 2 or query_points.shape[-1] != 2:
        raise ValueError("query_points must have shape (n_query, 2)")
    if not query_points.is_floating_point():
        raise TypeError("query_points must be floating point")
    if query_points.device != boundary.points.device:
        raise ValueError("query points and boundary must share a device")
    if query_points.dtype != boundary.points.dtype:
        raise ValueError("query points and boundary must share a dtype")


def _panel_endpoints(boundary: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = boundary.points[boundary.cells]
    return vertices[:, 0], vertices[:, 1]


def _panel_influence_unchecked(
    panel_start: torch.Tensor,
    panel_end: torch.Tensor,
    query_points: torch.Tensor,
) -> torch.Tensor:
    """Return exact constant-panel influences without singularity checks."""

    start_vector = panel_start.unsqueeze(0) - query_points.unsqueeze(1)
    end_vector = panel_end.unsqueeze(0) - query_points.unsqueeze(1)
    cross = (
        start_vector[..., 0] * end_vector[..., 1]
        - start_vector[..., 1] * end_vector[..., 0]
    )
    dot = torch.sum(start_vector * end_vector, dim=-1)
    return -torch.atan2(cross, dot) / _TWO_PI


def _query_on_panel(
    panel_start: torch.Tensor,
    panel_end: torch.Tensor,
    query_points: torch.Tensor,
) -> torch.Tensor:
    """Identify exact point-on-segment singularities without a length scale."""

    start_vector = panel_start.unsqueeze(0) - query_points.unsqueeze(1)
    end_vector = panel_end.unsqueeze(0) - query_points.unsqueeze(1)
    cross = (
        start_vector[..., 0] * end_vector[..., 1]
        - start_vector[..., 1] * end_vector[..., 0]
    )
    dot = torch.sum(start_vector * end_vector, dim=-1)
    return (cross == 0.0) & (dot <= 0.0)


def double_layer_influence(
    boundary: Mesh,
    query_points: torch.Tensor,
) -> torch.Tensor:
    r"""Return exact piecewise-constant panel influences at interior queries.

    The returned matrix has shape ``(n_query, n_panels)``.  Entry ``(i, j)``
    is the analytic integral of

    ``n_j dot (y - query_i) / (2 pi |y - query_i|**2)``

    over panel ``j``.  Thus multiplication by a panelwise density performs the
    boundary integral; no additional quadrature weight may be applied.

    Queries lying exactly on a panel are rejected because a boundary value
    requires a side limit and jump relation, not pointwise kernel evaluation.
    Use :func:`double_layer_collocation_matrix` for the interior boundary trace.
    """

    _validate_boundary(boundary)
    _validate_queries(boundary, query_points)
    panel_start, panel_end = _panel_endpoints(boundary)
    if (
        not torch.compiler.is_compiling()
        and torch.any(_query_on_panel(panel_start, panel_end, query_points)).item()
    ):
        raise ValueError(
            "double-layer evaluation is singular on a panel; use the boundary "
            "trace/collocation operator instead"
        )
    return _panel_influence_unchecked(panel_start, panel_end, query_points)


def double_layer_collocation_matrix(boundary: Mesh) -> torch.Tensor:
    r"""Discretize the interior trace ``1/2 I + K`` at panel midpoints.

    Densities and Dirichlet data are piecewise constant per panel.  Off-diagonal
    entries are exact panel integrals.  The diagonal principal-value integral
    over a straight self-panel is zero, while ``1/2`` is the analytic interior
    jump.  No diagonal regularization is added: this is a second-kind interior
    Dirichlet equation and has no single-layer constant-mode gauge.
    """

    _validate_boundary(boundary)
    panel_start, panel_end = _panel_endpoints(boundary)
    matrix = _panel_influence_unchecked(panel_start, panel_end, boundary.cell_centroids)
    matrix = matrix.clone()
    matrix.diagonal().fill_(0.5)
    return matrix


def solve_double_layer_density(
    boundary: Mesh,
    boundary_values: torch.Tensor,
) -> torch.Tensor:
    r"""Solve ``(1/2 I + K) mu = g`` for one or more right-hand sides.

    ``boundary_values`` must have leading dimension ``boundary.n_cells``; any
    trailing dimensions are flattened into independent right-hand sides and
    restored on return.  A failed or ill-conditioned solve is intentionally
    surfaced by ``torch.linalg.solve`` rather than hidden by
    geometry-dependent regularization.
    """

    _validate_density(boundary, boundary_values, name="boundary_values")
    system = double_layer_collocation_matrix(boundary)
    if boundary_values.ndim == 1:
        return torch.linalg.solve(system, boundary_values)
    shape = boundary_values.shape
    density = torch.linalg.solve(system, boundary_values.reshape(shape[0], -1))
    return density.reshape(shape)


def evaluate_double_layer(
    boundary: Mesh,
    density: torch.Tensor,
    query_points: torch.Tensor,
    *,
    query_chunk_size: int = 1024,
) -> torch.Tensor:
    r"""Evaluate a panelwise-constant double-layer density in query chunks."""

    _validate_density(boundary, density, name="density")
    _validate_queries(boundary, query_points)
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size < 1
    ):
        raise ValueError("query_chunk_size must be a positive integer")

    chunks: list[torch.Tensor] = []
    for start in range(0, query_points.shape[0], query_chunk_size):
        influence = double_layer_influence(
            boundary, query_points[start : start + query_chunk_size]
        )
        chunks.append(torch.tensordot(influence, density, dims=([1], [0])))
    if chunks:
        return torch.cat(chunks, dim=0)
    return density.new_empty((0, *density.shape[1:]))


def _validate_density(boundary: Mesh, density: torch.Tensor, *, name: str) -> None:
    _validate_boundary(boundary)
    if density.ndim < 1 or density.shape[0] != boundary.n_cells:
        raise ValueError(
            f"{name} must have leading dimension boundary.n_cells ({boundary.n_cells})"
        )
    if not density.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if density.device != boundary.points.device:
        raise ValueError(f"{name} and boundary must share a device")
    if density.dtype != boundary.points.dtype:
        raise ValueError(f"{name} and boundary must share a dtype")


def _benchmark_boundary(domain: DomainMesh, boundary_value_key: str) -> Mesh:
    if set(domain.boundaries.keys()) != {"dirichlet"}:
        raise ValueError("benchmark domains must contain only a 'dirichlet' boundary")
    boundary = domain.boundaries["dirichlet"]
    _validate_boundary(boundary)
    if boundary_value_key not in boundary.cell_data:
        raise ValueError(f"dirichlet.cell_data must contain {boundary_value_key!r}")
    values = boundary.cell_data[boundary_value_key]
    if values.ndim != 1:
        raise ValueError(f"{boundary_value_key!r} must be a scalar cell field")
    _validate_density(boundary, values, name=boundary_value_key)
    return boundary


def _prediction_mesh(
    domain: DomainMesh, potential: torch.Tensor, output_key: str
) -> Mesh:
    return domain.interior.with_data(
        point_data={output_key: potential},
        cell_data={},
        global_data=domain.global_data,
    )


class _DoubleLayerControl(nn.Module):
    """Common domain adapter for benchmark-local double-layer controls."""

    def __init__(
        self,
        *,
        query_chunk_size: int = 1024,
        boundary_value_key: str = "boundary_value",
        output_key: str = "potential",
    ) -> None:
        super().__init__()
        if (
            isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        if not boundary_value_key or not output_key:
            raise ValueError("field keys must be nonempty")
        self.query_chunk_size = query_chunk_size
        self.boundary_value_key = boundary_value_key
        self.output_key = output_key

    def density(self, boundary: Mesh, boundary_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain, self.boundary_value_key)
        values = boundary.cell_data[self.boundary_value_key]
        mean = _constant_exact_boundary_mean(boundary.cell_areas, values)
        residual_values = values - mean
        density = self.density(boundary, residual_values)
        residual_potential = evaluate_double_layer(
            boundary,
            density,
            domain.interior.points,
            query_chunk_size=self.query_chunk_size,
        )
        potential = mean + residual_potential
        return _prediction_mesh(domain, potential, self.output_key)

    def collocation_residual(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return ``(1/2 I + K) mu_theta[g] - g`` panelwise."""

        boundary = _benchmark_boundary(domain, self.boundary_value_key)
        values = boundary.cell_data[self.boundary_value_key]
        mean = _constant_exact_boundary_mean(boundary.cell_areas, values)
        residual_values = values - mean
        density = self.density(boundary, residual_values)
        return double_layer_collocation_matrix(boundary) @ density - residual_values

    def collocation_loss(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the boundary-measure-weighted mean squared trace residual."""

        boundary = _benchmark_boundary(domain, self.boundary_value_key)
        residual = self.collocation_residual(domain)
        weights = boundary.cell_areas
        return torch.sum(weights * residual.square()) / weights.sum()


class DirectDoubleLayerPotential(_DoubleLayerControl):
    r"""Use the mean-free prescribed values directly as layer density.

    This parameter-free control tests whether merely supplying an analytic
    harmonic decoder is sufficient. The constant component is lifted
    analytically, while ``g - g_bar`` is used directly as density. It therefore
    reproduces constants even for smooth-domain queries outside the chordal
    input polygon, but it does *not* generally satisfy the nonconstant boundary
    trace because the residual Dirichlet value is not generally the solution
    density.
    """

    def density(self, boundary: Mesh, boundary_values: torch.Tensor) -> torch.Tensor:
        """Return the prescribed boundary values unchanged as the density."""

        del boundary
        return boundary_values


class SolvedDoubleLayerPotential(_DoubleLayerControl):
    r"""Solve the dense second-kind boundary equation before evaluation.

    The constant component is lifted analytically and the mean-free density is
    solved from the second-kind system. This is the parameter-free
    PDE-conforming reference. Its quadratic memory and cubic dense-solve cost
    make it a diagnostic oracle, not a proposed scalable architecture.
    """

    def density(self, boundary: Mesh, boundary_values: torch.Tensor) -> torch.Tensor:
        """Return the density from the exact dense collocation solve."""

        return solve_double_layer_density(boundary, boundary_values)


class LearnedDensityDoubleLayerPotential(_DoubleLayerControl):
    r"""Learn a linear Richardson processor for the double-layer density.

    Starting from the direct density ``mu_0 = g``, iteration ``j`` applies

    .. math::

        \mu_{j+1} = \mu_j + \alpha_j
            \left[g-(\tfrac12I+K)\mu_j\right].

    The scalar relaxations ``alpha_j`` are the only trainable parameters and
    may be optimized with :meth:`collocation_loss`.  For every parameter value,
    the map from ``g`` to ``mu`` and hence to the interior potential is exactly
    linear.  The processor is mesh-resolution independent and uses no learned
    coordinate frame or interaction length.  It is intentionally a minimal
    learned-density control rather than a claim that Richardson iteration is
    the best production boundary encoder.
    """

    def __init__(
        self,
        *,
        n_iterations: int = 8,
        initial_relaxation: float = 1.0,
        query_chunk_size: int = 1024,
        boundary_value_key: str = "boundary_value",
        output_key: str = "potential",
    ) -> None:
        super().__init__(
            query_chunk_size=query_chunk_size,
            boundary_value_key=boundary_value_key,
            output_key=output_key,
        )
        if (
            isinstance(n_iterations, bool)
            or not isinstance(n_iterations, int)
            or n_iterations < 1
        ):
            raise ValueError("n_iterations must be a positive integer")
        if not math.isfinite(initial_relaxation):
            raise ValueError("initial_relaxation must be finite")
        self.relaxation = nn.Parameter(
            torch.full((n_iterations,), float(initial_relaxation))
        )

    def density(self, boundary: Mesh, boundary_values: torch.Tensor) -> torch.Tensor:
        """Apply learned linear residual iterations to the boundary data."""

        system = double_layer_collocation_matrix(boundary)
        density = boundary_values
        relaxation = self.relaxation.to(dtype=boundary_values.dtype)
        for step in relaxation.unbind():
            density = density + step * (boundary_values - system @ density)
        return density


def _constant_exact_boundary_mean(
    weights: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    r"""Return a linear weighted mean that reproduces constants bit-for-bit.

    Writing the mean relative to one boundary value avoids the independent
    rounding of ``sum(weights * constant)`` and ``constant * sum(weights)``.
    The expression remains a linear functional of ``values``.
    """

    anchor = values[0]
    return anchor + torch.sum(weights * (values - anchor)) / weights.sum()


class EncodedDoubleLayerPotential(nn.Module):
    r"""MeshTransformer boundary encoder with an analytic harmonic decoder.

    This control retains the current MeshTransformer source-side architecture:
    a nonlinear operator/geometry stream and an exactly drive-linear boundary
    stream.  The stock query-attention blocks are discarded at construction.
    Its geometry-conditioned scalar output projection is instead applied
    pointwise to the encoded boundary states to produce a panel density.  Thus
    every registered trainable parameter participates in boundary encoding or
    density projection; no unused stock query decoder is carried by the model.

    Constant lifting is structural.  For boundary data ``g``, let ``g_bar`` be
    its panel-measure mean and encode only ``g - g_bar``.  The model evaluates

    .. math::

        u_\theta(x) = \bar g + Q\rho_\theta[g-\bar g](x).

    The density processor is linear in its residual drive at fixed geometry,
    so the full map is exactly linear in ``g``.  A constant trace has exactly
    zero residual input and therefore produces ``u = g_bar`` independently of
    the learned parameters.  The corresponding total panel density is
    ``g_bar + rho_theta``.

    This remains a benchmark-local hybrid: MeshTransformer learns the global
    boundary density, while the exact 2D double-layer kernel supplies harmonic
    propagation to arbitrary interior queries.  Its boundary encoding is
    linear in entity count at fixed feature order, but exact evaluation is
    dense in query--panel pairs.
    """

    def __init__(
        self,
        config: MeshTransformerConfig | None = None,
        *,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        # Delayed import keeps this example module acyclic if models.py later
        # chooses to re-export the benchmark-local PDE controls.
        from models import MeshTransformerConfig, build_mesh_transformer

        config = MeshTransformerConfig() if config is None else config
        encoder = build_mesh_transformer(config)

        # `encode` naturally supports no query blocks: it then returns the
        # source operator/drive states and an empty moment tuple.  Replacing the
        # ModuleList unregisters (rather than merely freezing) every stock
        # query-attention parameter.  Move the scalar output projection out of
        # the stock model and repurpose it as the explicitly named,
        # geometry-conditioned density head below.
        density_projection = encoder.output_projection
        encoder.query_blocks = nn.ModuleList()
        encoder.output_projection = None
        encoder.query_layers = 0
        self.encoder = encoder
        self.density_projection = density_projection

        if query_chunk_size is None:
            query_chunk_size = config.query_chunk_size
        if (
            isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        self.query_chunk_size = query_chunk_size

    def _encode_residual(
        self, domain: DomainMesh
    ) -> tuple[Mesh, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary = _benchmark_boundary(domain, "boundary_value")
        values = boundary.cell_data["boundary_value"]
        weights = boundary.cell_areas
        mean = _constant_exact_boundary_mean(weights, values)
        residual_values = values - mean
        residual_boundary = boundary.with_data(
            cell_data={"boundary_value": residual_values}
        )
        residual_domain = DomainMesh(
            interior=domain.interior,
            boundaries={"dirichlet": residual_boundary},
            global_data=domain.global_data,
        )
        encoded = self.encoder.encode(residual_domain)
        projected = self.density_projection(encoded.operator_state, encoded.drive_state)
        if projected.scalars.shape != (boundary.n_cells, 1):
            raise RuntimeError("density projection must produce one scalar per panel")
        residual_density = projected.scalars[:, 0]
        return boundary, mean, residual_values, residual_density

    def panel_density(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the lifted total density ``g_bar + rho_theta`` panelwise."""

        _, mean, _, residual_density = self._encode_residual(domain)
        return mean + residual_density

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary, mean, _, residual_density = self._encode_residual(domain)
        residual_potential = evaluate_double_layer(
            boundary,
            residual_density,
            domain.interior.points,
            query_chunk_size=self.query_chunk_size,
        )
        # Add the constant solution explicitly instead of relying on a winding
        # test for the polygonal approximation.  This remains exact even for a
        # query infinitesimally outside the discrete polygon but inside the
        # smooth generating domain.
        potential = mean + residual_potential
        return _prediction_mesh(domain, potential, "potential")

    def collocation_residual(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the lifted interior-trace residual on boundary panels."""

        boundary, _, residual_values, residual_density = self._encode_residual(domain)
        return (
            double_layer_collocation_matrix(boundary) @ residual_density
            - residual_values
        )

    def collocation_loss(self, domain: DomainMesh) -> torch.Tensor:
        r"""Return the panel-measure-weighted mean squared trace residual."""

        boundary = _benchmark_boundary(domain, "boundary_value")
        residual = self.collocation_residual(domain)
        weights = boundary.cell_areas
        return torch.sum(weights * residual.square()) / weights.sum()


__all__ = [
    "DirectDoubleLayerPotential",
    "EncodedDoubleLayerPotential",
    "LearnedDensityDoubleLayerPotential",
    "SolvedDoubleLayerPotential",
    "double_layer_collocation_matrix",
    "double_layer_influence",
    "evaluate_double_layer",
    "solve_double_layer_density",
]
