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

r"""Exact variable-geometry Dirichlet--Laplace samples in two dimensions.

The canonical domain is the image of the unit disk under

.. math::

   F(z) = z + \sum_{m \in S} a_m z^m,

where the coefficients obey ``sum(m * abs(a_m)) < 1``.  Consequently,
``abs(F'(z) - 1) < 1`` throughout the closed disk.  In particular, the
derivative cannot vanish and ``Re(F') > 0``.  Integrating ``F'`` along the
line segment between any two disk points proves that ``F`` is injective and
gives the quantitative bi-Lipschitz bounds

.. math::

   (1-\kappa)|z_1-z_2| \leq |F(z_1)-F(z_2)|
   \leq (1+\kappa)|z_1-z_2|.

For a holomorphic polynomial ``H``, the scalar field

.. math::

   u\left(t + L R F(z)\right) = \operatorname{Re} H(z)

solves the homogeneous Laplace equation exactly.  Here ``R`` is any element
of O(2), ``L > 0``, and ``t`` is a translation.  The generated labels therefore
have no numerical PDE-solver error.  Polygonal boundary panels are solely an
input discretization of this known continuous problem.

Only physical coordinates, cell-centered Dirichlet values, the reference
length, targets, and loss metadata are placed in the :class:`DomainMesh`.
Conformal preimages and generating coefficients remain on the returned sample
object so a model cannot accidentally consume them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from physicsnemo.mesh import DomainMesh, Mesh


def _validate_real_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    _validate_real_dtype(dtype)
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}")


def _generator(seed: int) -> torch.Generator:
    _validate_seed(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def _canonical_modes(modes: Sequence[int], *, minimum: int) -> tuple[int, ...]:
    result = tuple(modes)
    if any(isinstance(mode, bool) or not isinstance(mode, int) for mode in result):
        raise TypeError("modes must contain only integers")
    if any(mode < minimum for mode in result):
        raise ValueError(f"modes must be at least {minimum}")
    if len(set(result)) != len(result):
        raise ValueError("modes must be unique")
    if tuple(sorted(result)) != result:
        raise ValueError("modes must be strictly increasing")
    return result


def complex_to_points(values: torch.Tensor) -> torch.Tensor:
    r"""Convert a complex tensor to Cartesian points with final dimension two."""
    if not torch.is_complex(values):
        raise TypeError("values must be a complex tensor")
    return torch.stack((values.real, values.imag), dim=-1)


def points_to_complex(points: torch.Tensor) -> torch.Tensor:
    r"""Convert Cartesian points with final dimension two to complex values."""
    if points.shape[-1:] != (2,):
        raise ValueError("points must have final dimension two")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points must use float32 or float64")
    return torch.complex(points[..., 0], points[..., 1])


def unit_circle(angles: torch.Tensor) -> torch.Tensor:
    r"""Return ``exp(i * angles)`` while preserving device and precision."""
    if angles.dtype not in (torch.float32, torch.float64):
        raise ValueError("angles must use float32 or float64")
    return torch.polar(torch.ones_like(angles), angles)


@dataclass(frozen=True)
class ConformalGeometry:
    r"""Coefficients of a certified nondegenerate disk map.

    ``modes`` contains distinct increasing integers at least two and
    ``coefficients[j]`` is :math:`a_{m_j}`.  The empty mode tuple represents
    the unit disk.  The strict coefficient bound is checked at construction.
    """

    modes: tuple[int, ...]
    coefficients: torch.Tensor

    def __post_init__(self) -> None:
        modes = _canonical_modes(self.modes, minimum=2)
        object.__setattr__(self, "modes", modes)
        if not torch.is_complex(self.coefficients):
            raise TypeError("geometry coefficients must be a complex tensor")
        if self.coefficients.ndim != 1 or self.coefficients.shape[0] != len(modes):
            raise ValueError("geometry coefficients must have shape (len(modes),)")
        if not torch.isfinite(self.coefficients).all().item():
            raise ValueError("geometry coefficients must be finite")
        if self.deformation_bound.item() >= 1.0:
            raise ValueError("sum(m * abs(a_m)) must be strictly less than one")

    @property
    def real_dtype(self) -> torch.dtype:
        """Return the real dtype underlying the complex coefficients."""

        return self.coefficients.real.dtype

    @property
    def device(self) -> torch.device:
        """Return the device holding the geometry coefficients."""

        return self.coefficients.device

    @property
    def deformation_bound(self) -> torch.Tensor:
        r"""Return :math:`\kappa=\sum_m m|a_m|`."""
        if not self.modes:
            return self.coefficients.real.new_zeros(())
        modes = self.coefficients.real.new_tensor(self.modes)
        return torch.sum(modes * torch.abs(self.coefficients))


@dataclass(frozen=True)
class HarmonicDrive:
    r"""Coefficients of ``Re(c_0 + sum(c_k z**k))`` on the disk."""

    constant: torch.Tensor
    modes: tuple[int, ...]
    coefficients: torch.Tensor

    def __post_init__(self) -> None:
        modes = _canonical_modes(self.modes, minimum=1)
        object.__setattr__(self, "modes", modes)
        if self.constant.ndim != 0 or torch.is_complex(self.constant):
            raise ValueError("drive constant must be a real scalar tensor")
        if self.constant.dtype not in (torch.float32, torch.float64):
            raise ValueError("drive constant must use float32 or float64")
        if not torch.is_complex(self.coefficients):
            raise TypeError("drive coefficients must be a complex tensor")
        if self.coefficients.ndim != 1 or self.coefficients.shape[0] != len(modes):
            raise ValueError("drive coefficients must have shape (len(modes),)")
        if (
            self.coefficients.real.dtype != self.constant.dtype
            or self.coefficients.device != self.constant.device
        ):
            raise ValueError(
                "drive constant and coefficients must share dtype and device"
            )
        if (
            not torch.isfinite(self.constant).item()
            or not torch.isfinite(self.coefficients).all().item()
        ):
            raise ValueError("drive coefficients must be finite")

    @property
    def boundary_rms(self) -> torch.Tensor:
        r"""Exact RMS of the real trace on the unit circle."""
        energy = self.constant.square()
        if self.coefficients.numel():
            energy = energy + 0.5 * torch.sum(torch.abs(self.coefficients).square())
        return torch.sqrt(energy)


@dataclass(frozen=True)
class SimilarityTransform:
    r"""A physical transform ``x -> translation + scale * rotation @ x``.

    Despite the field name ``rotation``, any orthogonal 2-by-2 matrix is
    accepted, including reflections.
    """

    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor

    def __post_init__(self) -> None:
        if self.scale.ndim != 0 or self.scale.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("scale must be a float32 or float64 scalar tensor")
        if self.rotation.shape != (2, 2) or self.translation.shape != (2,):
            raise ValueError(
                "rotation and translation must have shapes (2, 2) and (2,)"
            )
        if (
            self.rotation.dtype != self.scale.dtype
            or self.translation.dtype != self.scale.dtype
            or self.rotation.device != self.scale.device
            or self.translation.device != self.scale.device
        ):
            raise ValueError("similarity tensors must share dtype and device")
        if not (
            torch.isfinite(self.scale).item()
            and torch.isfinite(self.rotation).all().item()
            and torch.isfinite(self.translation).all().item()
        ):
            raise ValueError("similarity tensors must be finite")
        if self.scale.item() <= 0.0:
            raise ValueError("scale must be positive")
        identity = torch.eye(2, dtype=self.scale.dtype, device=self.scale.device)
        tolerance = 2.0e-5 if self.scale.dtype == torch.float32 else 2.0e-12
        if not torch.allclose(
            self.rotation.T @ self.rotation,
            identity,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("rotation must be an O(2) matrix")

    @property
    def determinant(self) -> torch.Tensor:
        """Return the orientation sign of the orthogonal transformation."""

        return torch.linalg.det(self.rotation)


@dataclass(frozen=True)
class ConformalLaplaceSample:
    r"""A discretized problem plus its private analytic generating state."""

    domain: DomainMesh
    geometry: ConformalGeometry
    drive: HarmonicDrive
    similarity: SimilarityTransform
    query_preimages: torch.Tensor
    boundary_midpoint_preimages: torch.Tensor

    @property
    def target(self) -> torch.Tensor:
        """Return the sampled interior potential used only as a target."""

        return self.domain.interior.point_data["potential"]

    @property
    def area_jacobian(self) -> torch.Tensor:
        """Return the physical area weight for each interior query."""

        return self.domain.interior.point_data["area_jacobian"]


def identity_similarity(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SimilarityTransform:
    r"""Construct the identity physical similarity transform."""
    _validate_real_dtype(dtype)
    return SimilarityTransform(
        scale=torch.ones((), device=device, dtype=dtype),
        rotation=torch.eye(2, device=device, dtype=dtype),
        translation=torch.zeros(2, device=device, dtype=dtype),
    )


def sample_geometry(
    seed: int,
    *,
    modes: Sequence[int] = (2, 3),
    deformation_range: tuple[float, float] = (0.05, 0.35),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ConformalGeometry:
    r"""Sample conformal coefficients with a prescribed injectivity bound.

    The sampled coefficients are rescaled so ``sum(m * abs(a_m))`` equals a
    uniformly sampled value in ``deformation_range``.  Random numbers are
    generated on CPU in float64 before conversion, making a seed independent
    of the requested execution device and floating-point precision.
    """
    _validate_real_dtype(dtype)
    modes = _canonical_modes(modes, minimum=2)
    if not modes:
        raise ValueError("sample_geometry requires at least one mode")
    lower, upper = deformation_range
    if not (
        math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower <= upper < 1.0
    ):
        raise ValueError("deformation_range must satisfy 0 <= lower <= upper < 1")

    generator = _generator(seed)
    raw = torch.complex(
        torch.randn(len(modes), generator=generator, dtype=torch.float64),
        torch.randn(len(modes), generator=generator, dtype=torch.float64),
    )
    weighted_norm = torch.sum(raw.abs() * raw.real.new_tensor(modes))
    deformation = lower + (upper - lower) * torch.rand(
        (), generator=generator, dtype=torch.float64
    )
    coefficients = raw * (deformation / weighted_norm)
    return ConformalGeometry(
        modes=modes,
        coefficients=coefficients.to(device=device, dtype=_complex_dtype(dtype)),
    )


def sample_drive(
    seed: int,
    *,
    modes: Sequence[int] = tuple(range(1, 9)),
    regularity: float = 2.0,
    boundary_rms: float = 1.0,
    include_constant: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> HarmonicDrive:
    r"""Sample and normalize a smooth random Dirichlet trace.

    Mode ``k`` is initially weighted by ``k**(-regularity)``.  The complete
    real trace is then normalized to the requested exact unit-circle RMS.
    """
    _validate_real_dtype(dtype)
    modes = _canonical_modes(modes, minimum=1)
    if not modes and not include_constant:
        raise ValueError("the drive must contain at least one degree of freedom")
    if not math.isfinite(regularity):
        raise ValueError("regularity must be finite")
    if not math.isfinite(boundary_rms) or boundary_rms <= 0.0:
        raise ValueError("boundary_rms must be finite and positive")

    generator = _generator(seed)
    constant = (
        torch.randn((), generator=generator, dtype=torch.float64)
        if include_constant
        else torch.zeros((), dtype=torch.float64)
    )
    if modes:
        raw = torch.complex(
            torch.randn(len(modes), generator=generator, dtype=torch.float64),
            torch.randn(len(modes), generator=generator, dtype=torch.float64),
        )
        weights = raw.real.new_tensor(modes).pow(-regularity)
        coefficients = raw * weights
    else:
        coefficients = torch.empty(0, dtype=torch.complex128)
    energy = constant.square() + 0.5 * torch.sum(coefficients.abs().square())
    normalization = boundary_rms / torch.sqrt(energy)
    return HarmonicDrive(
        constant=(constant * normalization).to(device=device, dtype=dtype),
        modes=modes,
        coefficients=(coefficients * normalization).to(
            device=device, dtype=_complex_dtype(dtype)
        ),
    )


def sample_similarity(
    seed: int,
    *,
    scale_range: tuple[float, float] = (0.5, 2.0),
    translation_extent: float = 2.0,
    reflection: bool | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SimilarityTransform:
    r"""Sample a deterministic O(2), log-uniform-scale, translation transform."""
    _validate_real_dtype(dtype)
    lower, upper = scale_range
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower <= upper):
        raise ValueError("scale_range must satisfy 0 < lower <= upper")
    if not math.isfinite(translation_extent) or translation_extent < 0.0:
        raise ValueError("translation_extent must be finite and non-negative")
    if reflection is not None and not isinstance(reflection, bool):
        raise TypeError("reflection must be a bool or None")

    generator = _generator(seed)
    angle = 2.0 * math.pi * torch.rand((), generator=generator, dtype=torch.float64)
    log_scale = math.log(lower) + math.log(upper / lower) * torch.rand(
        (), generator=generator, dtype=torch.float64
    )
    scale = torch.exp(log_scale)
    translation = (
        2.0 * torch.rand(2, generator=generator, dtype=torch.float64) - 1.0
    ) * (translation_extent * scale)
    if reflection is None:
        reflection = bool(
            torch.randint(0, 2, (), generator=generator, dtype=torch.int64).item()
        )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        )
    )
    if reflection:
        rotation = rotation @ torch.diag(rotation.new_tensor([-1.0, 1.0]))
    return SimilarityTransform(
        scale=scale.to(device=device, dtype=dtype),
        rotation=rotation.to(device=device, dtype=dtype),
        translation=translation.to(device=device, dtype=dtype),
    )


def _integer_powers(values: torch.Tensor, modes: tuple[int, ...]) -> torch.Tensor:
    """Evaluate integer powers without complex-log derivatives at zero.

    Passing a tensor exponent to ``Tensor.pow`` selects the general complex
    power implementation, whose autograd formula contains ``log(values)`` and
    therefore produces NaNs at zero even when every exponent is an integer.
    Python integer exponents use polynomial multiplication rules instead.
    """
    return torch.stack(tuple(values**mode for mode in modes), dim=-1)


def conformal_map(geometry: ConformalGeometry, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the canonical map ``F(z)``."""
    _validate_complex_argument(geometry, z)
    if not geometry.modes:
        return z
    return z + torch.sum(
        geometry.coefficients * _integer_powers(z, geometry.modes), dim=-1
    )


def conformal_derivative(geometry: ConformalGeometry, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate ``F'(z)``."""
    _validate_complex_argument(geometry, z)
    if not geometry.modes:
        return torch.ones_like(z)
    modes = z.real.new_tensor(geometry.modes)
    return 1.0 + torch.sum(
        modes
        * geometry.coefficients
        * _integer_powers(z, tuple(mode - 1 for mode in geometry.modes)),
        dim=-1,
    )


def evaluate_potential(drive: HarmonicDrive, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the exact harmonic target ``Re(H(z))``."""
    _validate_drive_argument(drive, z)
    if not drive.modes:
        return drive.constant.expand(z.shape)
    holomorphic = drive.constant + torch.sum(
        drive.coefficients * _integer_powers(z, drive.modes), dim=-1
    )
    return holomorphic.real


def evaluate_flux(
    geometry: ConformalGeometry,
    drive: HarmonicDrive,
    similarity: SimilarityTransform,
    boundary_preimages: torch.Tensor,
    physical_normals: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the exact physical normal flux :math:`\partial u/\partial n`.

    The solution is ``u(x) = Re H(z)`` with ``x = T(F(z))``, ``T`` the physical
    similarity.  In the intermediate frame ``w = F(z)`` the field is
    ``Re G(w)`` with ``G = H \circ F^{-1}``, so the gradient is the vector form
    of the conjugate derivative,

    .. math::

       \nabla_w u = (\operatorname{Re} g,\ -\operatorname{Im} g),
       \qquad g = G'(w) = H'(z) / F'(z),

    and the physical gradient under ``x = t + s R w`` (any orthogonal ``R``,
    including reflections) is :math:`\nabla_x u = (1/s)\,R\,\nabla_w u`.  The
    returned flux is ``physical_normals . grad_x u`` at each supplied preimage,
    with the caller choosing the normal convention (the benchmark generator
    uses the exact outward normals of the continuous curve).  A drive without
    nonconstant modes has identically zero flux.
    """

    _validate_geometry_drive(geometry, drive)
    _validate_compatible(geometry, similarity)
    _validate_complex_argument(geometry, boundary_preimages)
    if physical_normals.shape != boundary_preimages.shape + (2,):
        raise ValueError(
            "physical_normals must have shape (*boundary_preimages.shape, 2)"
        )
    if (
        physical_normals.dtype != geometry.real_dtype
        or physical_normals.device != geometry.device
    ):
        raise ValueError("physical_normals and geometry must share dtype and device")
    if not drive.modes:
        return torch.zeros(
            boundary_preimages.shape,
            device=geometry.device,
            dtype=geometry.real_dtype,
        )

    modes = boundary_preimages.real.new_tensor(drive.modes)
    holomorphic_derivative = torch.sum(
        modes
        * drive.coefficients
        * _integer_powers(boundary_preimages, tuple(mode - 1 for mode in drive.modes)),
        dim=-1,
    )
    conjugate_gradient = holomorphic_derivative / conformal_derivative(
        geometry, boundary_preimages
    )
    intermediate_gradient = torch.stack(
        (conjugate_gradient.real, -conjugate_gradient.imag), dim=-1
    )
    physical_gradient = (
        torch.einsum("ed,...d->...e", similarity.rotation, intermediate_gradient)
        / similarity.scale
    )
    return torch.sum(physical_normals * physical_gradient, dim=-1)


def build_neumann_domain_sample(
    geometry: ConformalGeometry,
    drive: HarmonicDrive,
    *,
    n_boundary: int = 128,
    n_query: int = 512,
    query_seed: int = 0,
    query_preimages: torch.Tensor | None = None,
    similarity: SimilarityTransform | None = None,
) -> ConformalLaplaceSample:
    r"""Build the Neumann variant of one exact conformal Laplace problem.

    The mesh geometry, query sampling, and metadata match
    :func:`build_domain_sample` exactly; only the boundary data and the gauge
    of the interior target change:

    ``boundary_flux`` (boundary ``cell_data``)
        Exact continuous flux ``n . grad u`` sampled at parameter-space panel
        midpoints with the exact outward normal of the continuous curve, then
        made discretely compatible.  The continuum flux of a harmonic field
        integrates to zero over the boundary, but midpoint sampling leaves an
        ``O(h**2)`` quadrature deficit; the panel-measure-weighted mean
        ``sum(w * flux) / sum(w)`` is subtracted so ``sum(w * flux) == 0``
        exactly and the discrete Neumann problem stays solvable.
    ``boundary_value`` (boundary ``cell_data``)
        Gauge-fixed Dirichlet trace samples ``g - u_bar``, retained only for
        boundary-range diagnostics.  Neumann models must not read this field.
    ``potential`` (interior ``point_data``)
        Gauge-fixed target ``u - u_bar`` where
        ``u_bar = sum(w_j * u(midpoint_j)) / sum(w_j)`` is the discrete
        boundary-measure mean of the exact potential -- a property of the
        solution and its boundary quadrature, never of the query set.  Neumann
        data determine ``u`` only up to a constant, and this gauge matches the
        model-side convention of reporting the potential relative to its own
        boundary mean.

    The sole boundary keeps the benchmark's fixed ``"dirichlet"`` mesh key so
    every piece of mesh plumbing is shared between problems; the boundary
    condition type is carried entirely by the ``cell_data`` key.
    """

    sample = build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        n_query=n_query,
        query_seed=query_seed,
        query_preimages=query_preimages,
        similarity=similarity,
    )
    boundary = sample.domain.boundaries["dirichlet"]
    weights = boundary.cell_areas
    exact_normals = boundary_outward_normals(
        geometry,
        torch.angle(sample.boundary_midpoint_preimages),
        sample.similarity,
    )
    flux = evaluate_flux(
        geometry,
        drive,
        sample.similarity,
        sample.boundary_midpoint_preimages,
        exact_normals,
    )
    flux = flux - torch.sum(weights * flux) / weights.sum()

    values = boundary.cell_data["boundary_value"]
    boundary_mean = torch.sum(weights * values) / weights.sum()
    neumann_boundary = boundary.with_data(
        cell_data={
            "boundary_flux": flux,
            "boundary_value": values - boundary_mean,
        }
    )
    interior = sample.domain.interior
    point_data = dict(interior.point_data.items())
    point_data["potential"] = point_data["potential"] - boundary_mean
    domain = DomainMesh(
        interior=interior.with_data(point_data=point_data),
        boundaries={"dirichlet": neumann_boundary},
        global_data=sample.domain.global_data,
    )
    return ConformalLaplaceSample(
        domain=domain,
        geometry=sample.geometry,
        drive=sample.drive,
        similarity=sample.similarity,
        query_preimages=sample.query_preimages,
        boundary_midpoint_preimages=sample.boundary_midpoint_preimages,
    )


def apply_similarity(
    points: torch.Tensor, transform: SimilarityTransform
) -> torch.Tensor:
    r"""Apply a physical similarity to Cartesian points."""
    if points.shape[-1:] != (2,):
        raise ValueError("points must have final dimension two")
    if points.dtype != transform.scale.dtype or points.device != transform.scale.device:
        raise ValueError("points and transform must share dtype and device")
    rotated = torch.einsum("...d,ed->...e", points, transform.rotation)
    return transform.translation + transform.scale * rotated


def map_to_physical(
    geometry: ConformalGeometry,
    z: torch.Tensor,
    similarity: SimilarityTransform,
) -> torch.Tensor:
    r"""Map disk coordinates through ``F`` and the physical similarity."""
    _validate_compatible(geometry, similarity)
    return apply_similarity(complex_to_points(conformal_map(geometry, z)), similarity)


def physical_area_jacobian(
    geometry: ConformalGeometry,
    z: torch.Tensor,
    similarity: SimilarityTransform,
) -> torch.Tensor:
    r"""Return ``L**2 * abs(F'(z))**2`` for physical-area quadrature."""
    _validate_compatible(geometry, similarity)
    return similarity.scale.square() * conformal_derivative(geometry, z).abs().square()


def boundary_outward_normals(
    geometry: ConformalGeometry,
    angles: torch.Tensor,
    similarity: SimilarityTransform,
) -> torch.Tensor:
    r"""Evaluate exact outward unit normals on the transformed boundary."""
    _validate_compatible(geometry, similarity)
    z = unit_circle(angles)
    # For the counterclockwise curve F(exp(i theta)), the clockwise rotation
    # of its tangent is z F'(z), which points outward.
    canonical = complex_to_points(z * conformal_derivative(geometry, z))
    canonical = canonical / torch.linalg.vector_norm(canonical, dim=-1, keepdim=True)
    return torch.einsum("...d,ed->...e", canonical, similarity.rotation)


def sample_disk_preimages(
    seed: int,
    n_points: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""Sample points uniformly with respect to unit-disk area."""
    _validate_real_dtype(dtype)
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    generator = _generator(seed)
    radii = torch.sqrt(torch.rand(n_points, generator=generator, dtype=torch.float64))
    angles = (
        2.0 * math.pi * torch.rand(n_points, generator=generator, dtype=torch.float64)
    )
    return torch.polar(radii, angles).to(device=device, dtype=_complex_dtype(dtype))


def build_domain_sample(
    geometry: ConformalGeometry,
    drive: HarmonicDrive,
    *,
    n_boundary: int = 128,
    n_query: int = 512,
    query_seed: int = 0,
    query_preimages: torch.Tensor | None = None,
    similarity: SimilarityTransform | None = None,
) -> ConformalLaplaceSample:
    r"""Build a polygonal boundary and exact interior targets.

    Boundary values are samples of the continuous trace at parameter-space
    panel midpoints.  Query points are sampled uniformly in disk area unless
    explicit complex ``query_preimages`` are supplied.  The interior metadata
    contains

    ``potential``
        Exact dimensionless target.
    ``area_jacobian``
        Physical Jacobian ``L**2 * abs(F')**2`` for area-weighted losses.
    ``preimage_radius``
        Radius used only for stratified evaluation.  Like all interior data,
        it is not a MeshTransformer input.
    """
    if (
        isinstance(n_boundary, bool)
        or not isinstance(n_boundary, int)
        or n_boundary < 3
    ):
        raise ValueError("n_boundary must be an integer of at least three")
    if isinstance(n_query, bool) or not isinstance(n_query, int) or n_query < 1:
        raise ValueError("n_query must be a positive integer")
    _validate_geometry_drive(geometry, drive)
    similarity = (
        identity_similarity(device=geometry.device, dtype=geometry.real_dtype)
        if similarity is None
        else similarity
    )
    _validate_compatible(geometry, similarity)

    angles = (
        2.0
        * math.pi
        * torch.arange(n_boundary, device=geometry.device, dtype=geometry.real_dtype)
        / n_boundary
    )
    boundary_preimages = unit_circle(angles)
    boundary_points = map_to_physical(geometry, boundary_preimages, similarity)
    indices = torch.arange(n_boundary, device=geometry.device)
    successors = torch.roll(indices, -1)
    # Mesh edge normals are a counterclockwise rotation of the directed edge.
    # An orientation-preserving physical map needs clockwise cell winding for
    # outward normals; a reflection reverses the vertex traversal and therefore
    # needs the opposite cell winding.
    if similarity.determinant.item() > 0.0:
        cells = torch.stack((successors, indices), dim=-1)
    else:
        cells = torch.stack((indices, successors), dim=-1)

    midpoint_angles = angles + math.pi / n_boundary
    midpoint_preimages = unit_circle(midpoint_angles)
    boundary_values = evaluate_potential(drive, midpoint_preimages)
    boundary = Mesh(
        points=boundary_points,
        cells=cells,
        cell_data={"boundary_value": boundary_values},
    )

    if query_preimages is None:
        query_preimages = sample_disk_preimages(
            query_seed,
            n_query,
            device=geometry.device,
            dtype=geometry.real_dtype,
        )
    else:
        _validate_complex_argument(geometry, query_preimages)
        if query_preimages.ndim != 1:
            raise ValueError("query_preimages must have shape (n_query,)")
        if query_preimages.numel() < 1:
            raise ValueError("query_preimages must be nonempty")
        tolerance = 2.0e-6 if geometry.real_dtype == torch.float32 else 2.0e-13
        if torch.any(query_preimages.abs() > 1.0 + tolerance).item():
            raise ValueError("query_preimages must lie in the closed unit disk")

    query_points = map_to_physical(geometry, query_preimages, similarity)
    target = evaluate_potential(drive, query_preimages)
    jacobian = physical_area_jacobian(geometry, query_preimages, similarity)
    interior = Mesh(
        points=query_points,
        point_data={
            "potential": target,
            "area_jacobian": jacobian,
            "preimage_radius": query_preimages.abs(),
        },
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={"reference_length": similarity.scale},
    )
    return ConformalLaplaceSample(
        domain=domain,
        geometry=geometry,
        drive=drive,
        similarity=similarity,
        query_preimages=query_preimages,
        boundary_midpoint_preimages=midpoint_preimages,
    )


def transform_sample(
    sample: ConformalLaplaceSample,
    similarity: SimilarityTransform,
) -> ConformalLaplaceSample:
    r"""Rebuild one continuous problem under a different physical similarity."""
    return build_domain_sample(
        sample.geometry,
        sample.drive,
        n_boundary=sample.domain.boundaries["dirichlet"].n_cells,
        query_preimages=sample.query_preimages,
        similarity=similarity,
    )


def _validate_complex_argument(
    geometry: ConformalGeometry, values: torch.Tensor
) -> None:
    if not torch.is_complex(values):
        raise TypeError("disk coordinates must be a complex tensor")
    if values.real.dtype != geometry.real_dtype or values.device != geometry.device:
        raise ValueError("disk coordinates and geometry must share dtype and device")


def _validate_drive_argument(drive: HarmonicDrive, values: torch.Tensor) -> None:
    if not torch.is_complex(values):
        raise TypeError("disk coordinates must be a complex tensor")
    if (
        values.real.dtype != drive.constant.dtype
        or values.device != drive.constant.device
    ):
        raise ValueError("disk coordinates and drive must share dtype and device")


def _validate_compatible(
    geometry: ConformalGeometry, similarity: SimilarityTransform
) -> None:
    if (
        geometry.real_dtype != similarity.scale.dtype
        or geometry.device != similarity.scale.device
    ):
        raise ValueError("geometry and similarity must share dtype and device")


def _validate_geometry_drive(geometry: ConformalGeometry, drive: HarmonicDrive) -> None:
    if (
        geometry.real_dtype != drive.constant.dtype
        or geometry.device != drive.constant.device
    ):
        raise ValueError("geometry and drive must share dtype and device")


__all__ = [
    "ConformalGeometry",
    "ConformalLaplaceSample",
    "HarmonicDrive",
    "SimilarityTransform",
    "apply_similarity",
    "boundary_outward_normals",
    "build_domain_sample",
    "build_neumann_domain_sample",
    "complex_to_points",
    "conformal_derivative",
    "conformal_map",
    "evaluate_flux",
    "evaluate_potential",
    "identity_similarity",
    "map_to_physical",
    "physical_area_jacobian",
    "points_to_complex",
    "sample_disk_preimages",
    "sample_drive",
    "sample_geometry",
    "sample_similarity",
    "transform_sample",
    "unit_circle",
]
