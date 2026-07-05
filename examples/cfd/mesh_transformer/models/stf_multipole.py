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

r"""Benchmark-local planar symmetric-trace-free multipole decoder.

For a planar vector :math:`x=(x_1,x_2)`, the two independent Cartesian
components of its rank-:math:`\ell` symmetric-trace-free (STF) power are

.. math::

   h_\ell(x) = (\operatorname{Re}(x_1+i x_2)^\ell,
                    \operatorname{Im}(x_1+i x_2)^\ell).

This is a polynomial Cartesian tensor representation, not an axis-wise
positional encoding.  Under any :math:`Q\in O(2)`, ``h_l(Q @ x)`` is related
to ``h_l(x)`` by an orthogonal two-dimensional irrep.  Consequently, an inner
product between source and query order-:math:`\ell` features is invariant
under rotations *and* reflections.

The model integrates one finite collection of boundary multipoles and reuses
it at every query point.  Its work and storage are therefore
:math:`O(C\ell_{\max}(N_s+N_q))` for ``C=channels_per_order``.  There are no
query-source pairs, spatial cutoffs, absolute coordinates, or Fourier-axis
features.  This module deliberately remains inside the example until the
higher-order construction earns a core API through the benchmark gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from models import _constant_exact_boundary_mean
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh


def planar_stf_coordinates(vectors: torch.Tensor, order: int) -> torch.Tensor:
    r"""Return the two Cartesian coordinates of ``STF(vectors**order)``.

    The output has shape ``vectors.shape[:-1] + (2,)``.  The recurrence is
    polynomial and avoids complex tensors, making gradients at the origin
    well-defined.  Order zero is intentionally excluded because its scalar
    irrep is handled by the quadrature-weighted boundary mean.
    """

    if vectors.ndim < 1 or vectors.shape[-1] != 2:
        raise ValueError("vectors must have final dimension two")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    real = vectors[..., 0]
    imaginary = vectors[..., 1]
    base_real = real
    base_imaginary = imaginary
    for _ in range(1, order):
        real, imaginary = (
            base_real * real - base_imaginary * imaginary,
            base_imaginary * real + base_real * imaginary,
        )
    return torch.stack((real, imaginary), dim=-1)


def _planar_stf_sequence(
    vectors: torch.Tensor,
    maximum_order: int,
) -> tuple[torch.Tensor, ...]:
    """Compute every regular planar STF power in one linear recurrence."""

    if vectors.ndim < 1 or vectors.shape[-1] != 2:
        raise ValueError("vectors must have final dimension two")
    if (
        isinstance(maximum_order, bool)
        or not isinstance(maximum_order, int)
        or maximum_order < 1
    ):
        raise ValueError("maximum_order must be a positive integer")
    base_real = vectors[..., 0]
    base_imaginary = vectors[..., 1]
    real = base_real
    imaginary = base_imaginary
    result: list[torch.Tensor] = []
    for order in range(1, maximum_order + 1):
        result.append(torch.stack((real, imaginary), dim=-1))
        if order < maximum_order:
            real, imaginary = (
                base_real * real - base_imaginary * imaginary,
                base_imaginary * real + base_real * imaginary,
            )
    return tuple(result)


class _ResidualInvariantGate(nn.Module):
    """Geometry-only scalar gate initialized to the constant function one."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        final = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, invariants: torch.Tensor) -> torch.Tensor:
        """Evaluate invariant scalar channels without changing tensor type."""

        return 1.0 + self.network(invariants)


@dataclass(frozen=True)
class STFMultipoleEncoding:
    r"""Reusable quadrature encoding of one Dirichlet boundary.

    ``moments[ell - 1]`` has shape ``(channels_per_order, 2)`` and transforms
    in the planar order-``ell`` STF irrep.  All remaining members are scalar
    invariants or physical vectors with explicit similarity scaling.
    """

    center: torch.Tensor
    reference_length: torch.Tensor
    normalized_measure: torch.Tensor
    boundary_mean: torch.Tensor
    moments: tuple[torch.Tensor, ...]


class STFMultipolePotential(nn.Module):
    r"""Linear-in-drive, similarity-invariant planar multipole potential.

    Boundary coordinates are centered at their quadrature centroid and divided
    by ``reference_length``.  Per-order source gates depend only on
    ``(|y|**2, n dot y, normalized boundary measure)``; query gates depend only
    on ``(|x|**2, normalized boundary measure)``.  These are dimensionless
    joint :math:`O(2)` invariants.  A boundary drive enters only after every
    gate has been computed, so no nonlinear path can violate superposition.

    The exact boundary mean is lifted out before higher moments are formed.
    Thus every constant Dirichlet condition is reproduced exactly on every
    geometry.  At initialization, each order has total coefficient
    :math:`1/\pi`, the continuous unit-disk Poisson coefficient.  Orders above
    ``lmax`` are absent by construction rather than merely discouraged.
    """

    _SUPPORTED_LMAX = (1, 2, 4)

    def __init__(
        self,
        *,
        lmax: int,
        channels_per_order: int = 2,
        hidden_dim: int = 32,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if isinstance(lmax, bool) or lmax not in self._SUPPORTED_LMAX:
            raise ValueError(f"lmax must be one of {self._SUPPORTED_LMAX}")
        for name, value in (
            ("channels_per_order", channels_per_order),
            ("hidden_dim", hidden_dim),
            ("hidden_layers", hidden_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.lmax = lmax
        self.channels_per_order = channels_per_order
        self.source_gates = nn.ModuleList(
            _ResidualInvariantGate(3, channels_per_order, hidden_dim, hidden_layers)
            for _ in range(lmax)
        )
        self.query_gates = nn.ModuleList(
            _ResidualInvariantGate(2, channels_per_order, hidden_dim, hidden_layers)
            for _ in range(lmax)
        )
        coefficients = torch.full(
            (lmax, channels_per_order),
            1.0 / (math.pi * channels_per_order),
        )
        self.coefficients = nn.Parameter(coefficients)

    def encode_boundary(
        self,
        boundary: Mesh,
        reference_length: torch.Tensor,
    ) -> STFMultipoleEncoding:
        """Integrate reusable STF moments from one boundary mesh."""

        self._validate_boundary(boundary, reference_length)
        weights = boundary.cell_areas / reference_length
        normalized_measure = weights.sum()
        center = torch.einsum("s,sd->d", weights, boundary.cell_centroids)
        center = center / normalized_measure
        source = (boundary.cell_centroids - center) / reference_length
        normals = boundary.cell_normals
        values = boundary.cell_data["boundary_value"]
        boundary_mean = _constant_exact_boundary_mean(weights, values)
        residual = values - boundary_mean

        repeated_measure = normalized_measure.expand(source.shape[0])
        source_invariants = torch.stack(
            (
                source.square().sum(dim=-1),
                torch.sum(normals * source, dim=-1),
                repeated_measure,
            ),
            dim=-1,
        )
        moments: list[torch.Tensor] = []
        weighted_residual = weights * residual
        source_stfs = _planar_stf_sequence(source, self.lmax)
        for gate, stf in zip(self.source_gates, source_stfs, strict=True):
            scalar_channels = gate(source_invariants)
            moments.append(
                torch.einsum("s,sc,sa->ca", weighted_residual, scalar_channels, stf)
            )

        return STFMultipoleEncoding(
            center=center,
            reference_length=reference_length,
            normalized_measure=normalized_measure,
            boundary_mean=boundary_mean,
            moments=tuple(moments),
        )

    def decode_points(
        self,
        points: torch.Tensor,
        encoding: STFMultipoleEncoding,
    ) -> torch.Tensor:
        """Evaluate an encoded boundary at arbitrary physical query points."""

        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("query points must have shape (n_query, 2)")
        if (
            points.device != encoding.center.device
            or points.dtype != encoding.center.dtype
        ):
            raise ValueError(
                "query points and boundary encoding must share dtype/device"
            )
        if len(encoding.moments) != self.lmax:
            raise ValueError("encoding order does not match this model")

        query = (points - encoding.center) / encoding.reference_length
        repeated_measure = encoding.normalized_measure.expand(points.shape[0])
        query_invariants = torch.stack(
            (query.square().sum(dim=-1), repeated_measure), dim=-1
        )
        potential = encoding.boundary_mean.expand(points.shape[0])
        query_stfs = _planar_stf_sequence(query, self.lmax)
        for order, (gate, moment, stf) in enumerate(
            zip(self.query_gates, encoding.moments, query_stfs, strict=True),
            start=1,
        ):
            scalar_channels = gate(query_invariants)
            contractions = torch.einsum("qa,ca->qc", stf, moment)
            potential = potential + torch.einsum(
                "c,qc,qc->q",
                self.coefficients[order - 1],
                scalar_channels,
                contractions,
            )
        return potential

    def forward(self, domain: DomainMesh) -> Mesh:
        """Map a benchmark ``DomainMesh`` to an interior potential ``Mesh``."""

        if set(domain.boundaries.keys()) != {"dirichlet"}:
            raise ValueError("domain must contain only a 'dirichlet' boundary")
        try:
            reference_length = domain.global_data["reference_length"].reshape(())
        except KeyError:
            raise ValueError(
                "domain.global_data must contain 'reference_length'"
            ) from None
        encoding = self.encode_boundary(
            domain.boundaries["dirichlet"], reference_length
        )
        potential = self.decode_points(domain.interior.points, encoding)
        return domain.interior.with_data(
            point_data={"potential": potential},
            cell_data={},
            global_data=domain.global_data,
        )

    @staticmethod
    def _validate_boundary(boundary: Mesh, reference_length: torch.Tensor) -> None:
        """Validate the narrow benchmark contract before forming moments."""

        if boundary.n_spatial_dims != 2 or boundary.n_manifold_dims != 1:
            raise ValueError("dirichlet boundary must be an edge mesh in 2D")
        if "boundary_value" not in boundary.cell_data:
            raise ValueError("boundary.cell_data must contain 'boundary_value'")
        values = boundary.cell_data["boundary_value"]
        if values.ndim != 1 or values.shape[0] != boundary.n_cells:
            raise ValueError("boundary_value must have shape (n_boundary_cells,)")
        if reference_length.ndim != 0:
            raise ValueError("reference_length must be scalar")
        if (
            boundary.points.device != reference_length.device
            or boundary.points.dtype != reference_length.dtype
        ):
            raise ValueError("boundary and reference_length must share dtype/device")
        if not torch.compiler.is_compiling():
            if (
                not torch.isfinite(reference_length).item()
                or reference_length.item() <= 0
            ):
                raise ValueError("reference_length must be finite and positive")
            areas = boundary.cell_areas
            if not torch.isfinite(areas).all().item() or torch.any(areas <= 0).item():
                raise ValueError("boundary cell measures must be finite and positive")


__all__ = [
    "STFMultipoleEncoding",
    "STFMultipolePotential",
    "planar_stf_coordinates",
]
