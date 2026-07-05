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

r"""Rotational steady Euler on disks: exact vorticity, exact labels, no solver.

This is the program's first genuinely ROTATIONAL steady Euler family.  All
previous fluid suites (potential flow, ``euler_bernoulli``) are irrotational
-- the vorticity is identically zero and the velocity is harmonic.  Here the
vorticity is nonzero *everywhere* and the momentum balance is genuinely
nonlinear, yet every label is closed form.

**Physics (restated and verified).**  2D steady incompressible Euler is

.. math::

   (u\cdot\nabla)u + \nabla p = 0, \qquad \nabla\cdot u = 0,

with density 1.  Introduce the streamfunction :math:`\psi` via

.. math::

   u = (\partial\psi/\partial y,\; -\partial\psi/\partial x),

so incompressibility holds identically and the scalar vorticity is
:math:`\omega = \partial_x v - \partial_y u = -\Delta\psi`.  Steady vorticity
transport, :math:`u\cdot\nabla\omega = 0`, says :math:`\omega` is constant
along streamlines, so (wherever streamlines foliate the region)
:math:`\omega = F(\psi)`.  For the LINEAR choice :math:`F(\psi) = c^2\psi`
the streamfunction solves the HELMHOLTZ equation

.. math::

   \Delta\psi + c^2\psi = 0

-- the oscillatory sibling of the screened testbed's modified-Helmholtz
equation, with :math:`J_m` Bessel series on disks where screened has
:math:`I_m/K_m`.  The pressure follows from the rotational Bernoulli
relation.  Using :math:`(u\cdot\nabla)u = \nabla(|u|^2/2) - u\times\omega`
and :math:`u\times(\omega\hat z) = \omega\,(v, -u) = -\omega\,\nabla\psi`
(both exact identities for the streamfunction convention above), momentum
becomes

.. math::

   \nabla\Big(p + \tfrac12|u|^2\Big) = -\,\omega\,\nabla\psi
   = -F(\psi)\,\nabla\psi ,

so :math:`H = p + |u|^2/2` is a function of :math:`\psi` alone with
:math:`dH/d\psi = -F(\psi)`; for :math:`F(\psi) = c^2\psi`,

.. math::

   H(\psi) = H_0 - \tfrac12 c^2 \psi^2
   \quad\Longrightarrow\quad
   p = H_0 - \tfrac12 c^2\psi^2 - \tfrac12|u|^2 ,

with the additive pressure gauge fixed here by :math:`H_0 = 0`.  This
derivation is VERIFIED numerically, not just asserted: the mandatory
certification of this family is the float64 autograd residual of the steady
Euler momentum equation, :math:`\|(u\cdot\nabla)u + \nabla p\| /
\|(u\cdot\nabla)u\| \le 10^{-10}` at random interior points across every
split (see :func:`euler_momentum_residual` and the tests), together with
exact incompressibility and the Helmholtz residual of :math:`\psi`.

**Exact solutions.**  On a disk of radius :math:`R` centered at
:math:`x_0`, with polar coordinates :math:`(r, \theta)` about the center,

.. math::

   \psi(r,\theta) = \sum_m \operatorname{Re}\!\big[c_m e^{im\theta}\big]\,
   \frac{J_m(c\,r)}{J_m(c\,R)} ,

each mode solving the Helmholtz equation exactly.  The implementation uses
the entire (Cartesian-smooth) form :math:`J_m(cr)e^{im\theta} =
(w/R)^m\,S_m(t)\,\cdot(\text{normalization})` with :math:`w = \tilde x +
i\tilde y`, :math:`t = (c/2)^2|w|^2`, and the alternating series
:math:`S_m(t) = \sum_k (-t)^k / (k!\,(k+m)!)`, which is autograd-smooth
everywhere including the disk center.  The series is accurate to float64
machine precision for arguments :math:`c\,r \le 6` and orders
:math:`m \le 10` (documented validity range, asserted by the builder; the
family's parameter band keeps :math:`c\,R < 2.405` anyway).

**Parity typing (the iteration-21/23 typing, exactly).**  Under a
reflection of frame and data, :math:`\psi` is a PSEUDOSCALAR (it flips
sign), the velocity :math:`u` is a polar vector, and the pressure is a true
scalar (it is even, being quadratic in :math:`\psi` and :math:`u`).  The
boundary drive of this family is the polar-vector boundary velocity, so --
unlike Family A and unlike the typed-circulation extension -- no declared
datum is itself pseudoscalar and no parity WALL exists for the velocity
head: polar in, polar out is representable with dot-product typing.  The
``*_pseudo`` arm instead probes whether internal pseudoscalar channels
(``drive_pseudo_dim > 0``: pseudo state arising from wedges of the vector
drive against geometry) help, because the natural latent of this problem --
the streamfunction itself -- is exactly such a wedge-sourced pseudoscalar.

**Model-facing data.**  The model sees only physical mesh data:

- ``boundary_velocity`` (rank 1) on boundary cells -- the exact velocity at
  panel midpoints, the family's drive;
- the dimensionless vorticity coupling :math:`\tilde c = c\,R` as a GLOBAL
  OPERATOR SCALAR in ``global_data["vorticity_coupling"]`` (the parameter
  axis of this family, exactly as the screened testbed's
  :math:`\tilde\kappa`); the reference length is the disk radius, so the
  supplied value is exactly :math:`c\,L_{\mathrm{ref}}`;
- interior query points.  Targets (interior ``point_data``):
  ``{velocity: rank 1, pressure: rank 0}`` -- the second multi-field family,
  with the same conventions as ``euler_bernoulli``.

**Well-posedness, stated honestly.**  Boundary-value problems for steady
Euler are subtle in general: with through-flow, vorticity must be specified
on inflow, and neither existence nor uniqueness is generic.  This family
does NOT claim otherwise.  It poses the LEARNING problem
:math:`\{\text{boundary velocity}, \tilde c\} \to \text{interior fields}`
on a family of certified exact solutions.  Within the sampled modal family
the map is well defined provided :math:`\tilde c` stays strictly below the
first Dirichlet eigenvalue of the disk: at :math:`\tilde c = j_{m,k}` (zeros
of :math:`J_m`; the smallest is :math:`j_{0,1} \approx 2.404826`) the
homogeneous Helmholtz problem acquires a nontrivial interior eigenfunction
with zero boundary trace, so the interior is NOT determined by boundary
data, and the mode normalization above divides by :math:`J_m(\tilde c)\to
0`.  Every split therefore keeps :math:`\tilde c` in a conservative band
below :math:`j_{0,1}`, with a near-eigenvalue tier retained only because its
labels remain certified (the series is exact; only the interior/boundary
amplitude ratio grows).

THE PRE-REGISTERED LINEAR WALL (logged before the first
``euler_rotational`` training run).  The boundary velocity is the family's
only drive, the exact interior velocity is exactly LINEAR in it (both are
linear in the mode coefficients), and the pressure
:math:`p = -\tfrac12 c^2\psi^2 - \tfrac12|u|^2` is exactly EVEN
(drive-quadratic).  The mode phases are uniform, so the training drive
distribution is symmetric under negation.  A ``field_mode="linear"`` arm is
exactly odd in the drive, and the :math:`L^2`-optimal odd fit of an even
target over a negation-symmetric distribution is the zero function: the
linear control's pressure error must sit pinned at relative L2
:math:`\approx 1.0` while its velocity error matches the nonlinear arm's.
This is the third structural wall instance (after the Liouville
superposition wall and the ``euler_bernoulli`` pressure-parity wall);
``mt_singpair_linear`` is registered purely to MEASURE it.

ARMS.  All transformer arms use the pruned two-member "singpair" kernel
dictionary and the flipped one-head reference configuration of iteration 32
(heads 1, ranks 48/16 at the reference total score capacity):

- ``mt_singpair_nl`` -- ``zero_preserving_nonlinear``: the historical
  primary arm; before iteration 35 the only mode in which the
  drive-quadratic pressure was representable.
- ``mt_singpair_nl_pseudo`` -- nonlinear plus ``drive_pseudo_dim=8``: the
  :math:`\psi`-parity probe (internal pseudoscalar channels for the
  wedge-sourced streamfunction latent; no wall is claimed for this arm).
- ``mt_singpair_linear`` -- the pre-registered wall control (see above).
- ``mt_singpair_q2`` -- ``field_mode="quadratic"`` (iteration 35, the fix
  rung of the nonlinear-fragility ladder): the drive degree is DECLARED at
  exactly the targets' degrees (velocity 1, pressure 2).  Iteration 34
  localized the near-eigenvalue detonation of the nonlinear arms
  (:math:`10^6`--:math:`10^{13}` at full training) in the read-in's
  implicit drive degree ~21 riding the physical :math:`1/J_0(\tilde c)`
  drive amplification.  Pre-registered (before the q2 retraining):
  near-eigenvalue combined error < 1.0 (stretch: the ~0.24--0.30
  renormalized-ordinary level of iteration 34), in-distribution within 2x
  seed sd of the nonlinear arms, and the drive-scaling structural degree
  test at machine precision.
- ``mt_singpair_q2_pseudo`` -- quadratic plus ``drive_pseudo_dim=8`` (the
  degree and parity probes composed); registered because the composition
  is one knob, not part of the pre-registered iteration 35 run matrix.
- ``boundary_mean`` -- the parameter-free floor: the boundary-measure mean
  of the declared per-field drive.  The velocity drive's mean is a constant
  vector; pressure has no boundary drive, so its floor is zero and its
  relative L2 is exactly 1.0 by construction.  (A constant velocity with
  zero pressure is itself an exact Euler solution, so the floor's momentum
  residual is exactly zero -- calibrated in the tests.)

SPLITS.  ``in_distribution`` (:math:`\tilde c \in [0.5, 1.8]`, drive modes
0--3), ``unseen_coupling`` (:math:`\tilde c \in [1.9, 2.2]`: parameter OOD
within the safe band), ``unseen_modes`` (drive modes 4--6), and
``near_eigenvalue`` (:math:`\tilde c \in [2.25, 2.38]`, approaching
:math:`j_{0,1} = 2.4048...`; labels stay certified -- asserted by the
certification tests -- but the :math:`m=0` mode's radial profile is
amplified by :math:`1/J_0(\tilde c)` relative to its boundary trace, up
to :math:`\approx 77` at the band edge :math:`\tilde c = 2.38`).  Every split reports per-field relative
L2 (``<split>/velocity``, ``<split>/pressure``) plus the combined
concatenated-field norm, and the training loss is the equal-weight mean of
per-field relative squared errors, exactly as in ``euler_bernoulli``.

This is a benchmark-local research prototype, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import _paths  # noqa: F401
import torch
from models import MeshTransformerConfig
from potential_flow import _KERNEL_ARM_KWARGS, _ONE_HEAD_CONFIG
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

FAMILY = "euler_rotational"

# The multi-field output declaration: one polar vector, one true scalar.
OUTPUT_FIELD_RANKS: dict[str, int] = {"velocity": 1, "pressure": 0}

#: First zero of J_0 -- the square root of the disk's first Dirichlet
#: eigenvalue of -Delta (radius 1).  At coupling = j_{m,k} the interior
#: Helmholtz solution is not determined by boundary data (a nontrivial
#: eigenfunction has zero trace) and the mode normalization divides by
#: J_m(coupling) -> 0, so the whole family stays strictly below this value.
FIRST_DIRICHLET_EIGENVALUE = 2.4048255576957727

#: Documented validity range of the J_m power series (float64 machine
#: precision; the alternating series loses digits only for much larger
#: arguments).  The eigenvalue guard keeps arguments below 2.405 anyway.
_SERIES_MAX_ARGUMENT = 6.0
_SERIES_MAX_ORDER = 10

# The pruned two-member kernel dictionary shared by every transformer arm.
_SINGPAIR_KWARGS = _KERNEL_ARM_KWARGS["mesh_transformer_kernel_singpair"]

# Drive/operator schema: the polar-vector boundary velocity is the drive on
# cells; the dimensionless vorticity coupling is a global OPERATOR scalar
# (the parameter axis, like the screened testbed's kappa_tilde).
_SCHEMA: dict[str, dict] = {
    "boundary_field_ranks": {
        "dirichlet": {"operator": {}, "drive": {"boundary_velocity": 1}}
    },
    "global_field_ranks": {"operator": {"vorticity_coupling": 0}, "drive": {}},
}


def bessel_j(order: int, x: torch.Tensor, *, terms: int = 40) -> torch.Tensor:
    r"""Series :math:`J_m(x)=\sum_k (-1)^k (x/2)^{2k+m}/(k!\,(k+m)!)`.

    Accurate to float64 machine precision for ``x <= 6`` and ``order <= 10``
    (documented validity range; asserted by the sample builder through the
    eigenvalue guard, which keeps arguments below 2.405).
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    half = 0.5 * x
    term = half.pow(order) / math.factorial(order)
    total = term
    for k in range(1, terms):
        term = term * (-half * half) / (k * (k + order))
        total = total + term
    return total


def _series_pair(
    order: int, t: torch.Tensor, *, terms: int = 40
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return :math:`S_m(t)=\sum_k (-t)^k/(k!\,(k+m)!)` and its derivative.

    :math:`J_m(x) = (x/2)^m S_m((x/2)^2)`; the series is entire in ``t``, so
    fields built from it are smooth in Cartesian coordinates everywhere,
    including the disk center.
    """

    value = torch.full_like(t, 1.0 / math.factorial(order))
    derivative = torch.zeros_like(t)
    term = value.clone()
    dterm = torch.full_like(t, -1.0 / math.factorial(1 + order))
    for k in range(1, terms):
        if k > 1:
            dterm = dterm * (-t) / ((k - 1) * (k + order))
        derivative = derivative + dterm
        term = term * (-t) / (k * (k + order))
        value = value + term
    return value, derivative


@dataclass(frozen=True)
class RotationalFlow:
    """One exact rotational Euler flow (the label generator's parameters).

    ``coefficients`` maps each angular mode to its complex amplitude; the
    boundary trace of the streamfunction of mode ``m`` is exactly
    ``Re[c_m e^{im theta}]``.  ``coupling`` is the dimensionless
    ``c * radius``; the dimensional vorticity coupling is
    ``c = coupling / radius``.
    """

    center: torch.Tensor  # (2,), float64
    radius: float
    coupling: float
    coefficients: dict[int, torch.Tensor]  # complex128 scalars


def _fields(
    flow: RotationalFlow, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form streamfunction and velocity at ``(n, 2)`` points.

    Differentiable in ``points`` (the series is polynomial in the local
    Cartesian coordinates), so autograd on top of this function yields exact
    derivatives of the exact fields -- the certification path.
    """

    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape (n, 2)")
    dtype = points.dtype
    center = flow.center.to(device=points.device, dtype=dtype)
    local = points - center[None, :]
    w = torch.complex(local[..., 0], local[..., 1])
    c = flow.coupling / flow.radius
    t = (0.25 * c * c) * (local.square().sum(dim=-1))
    t_boundary = torch.tensor((0.5 * flow.coupling) ** 2, dtype=dtype)
    psi = torch.zeros_like(t)
    dpsi_dx = torch.zeros_like(t)
    dpsi_dy = torch.zeros_like(t)
    for order, coefficient in flow.coefficients.items():
        if order > _SERIES_MAX_ORDER:
            raise ValueError("mode order exceeds the series validity range")
        series, dseries = _series_pair(order, t)
        series_boundary, _ = _series_pair(order, t_boundary)
        coefficient = coefficient.to(torch.complex128 if dtype == torch.float64
                                     else torch.complex64)
        # Holomorphic angular factor A(w) = c_m (w/R)^m and its derivative.
        if order == 0:
            angular = coefficient * torch.ones_like(w)
            angular_prime = torch.zeros_like(w)
        else:
            scale = flow.radius**order
            angular = coefficient * w.pow(order) / scale
            angular_prime = coefficient * order * w.pow(order - 1) / scale
        radial = series / series_boundary
        dradial = dseries / series_boundary
        psi = psi + angular.real * radial
        # d Re[A]/dx = Re[A'], d Re[A]/dy = -Im[A']; dt/dx = (c^2/2) x_local.
        chain = angular.real * dradial * (0.5 * c * c)
        dpsi_dx = dpsi_dx + angular_prime.real * radial + chain * local[..., 0]
        dpsi_dy = dpsi_dy - angular_prime.imag * radial + chain * local[..., 1]
    velocity_ = torch.stack((dpsi_dy, -dpsi_dx), dim=-1)
    return psi, velocity_


def streamfunction(flow: RotationalFlow, points: torch.Tensor) -> torch.Tensor:
    """Exact streamfunction (a pseudoscalar field) at ``(n, 2)`` points."""

    return _fields(flow, points)[0]


def velocity(flow: RotationalFlow, points: torch.Tensor) -> torch.Tensor:
    """Exact velocity ``u = (psi_y, -psi_x)`` at ``(n, 2)`` points."""

    return _fields(flow, points)[1]


def vorticity(flow: RotationalFlow, points: torch.Tensor) -> torch.Tensor:
    r"""Exact vorticity :math:`\omega = -\Delta\psi = c^2\psi` (nonzero!)."""

    c = flow.coupling / flow.radius
    return (c * c) * streamfunction(flow, points)


def pressure(flow: RotationalFlow, points: torch.Tensor) -> torch.Tensor:
    r"""Exact pressure from the rotational Bernoulli relation (gauge H0=0).

    .. math::

       p = -\tfrac12 c^2 \psi^2 - \tfrac12 |u|^2 ,

    i.e. :math:`H(\psi) = H_0 - c^2\psi^2/2` with :math:`H_0 = 0`; exactly
    even (drive-quadratic) in the mode coefficients because :math:`\psi` and
    :math:`u` are exactly odd.
    """

    psi, u = _fields(flow, points)
    c = flow.coupling / flow.radius
    return -0.5 * (c * c) * psi.square() - 0.5 * u.square().sum(dim=-1)


def euler_momentum_residual(flow: RotationalFlow, points: torch.Tensor) -> float:
    r"""Normalized steady-Euler momentum residual of the EXACT fields.

    ``|| (u . grad) u + grad p || / || (u . grad) u ||`` with every
    derivative taken by float64 autograd through the closed-form fields --
    the family's headline certification.  The exact labels must score
    float-roundoff zero (<= 1e-10 asserted in the tests); a wrong Bernoulli
    derivation would score O(1).
    """

    pts = points.detach().to(torch.float64).clone().requires_grad_(True)
    psi, u = _fields(flow, pts)
    c = flow.coupling / flow.radius
    p = -0.5 * (c * c) * psi.square() - 0.5 * u.square().sum(dim=-1)
    (grad_p,) = torch.autograd.grad(p.sum(), pts, create_graph=True)
    advection = torch.zeros_like(u)
    for component in range(2):
        (grad_u,) = torch.autograd.grad(
            u[:, component].sum(), pts, create_graph=True
        )
        advection[:, component] = (u * grad_u).sum(dim=-1)
    residual = (advection + grad_p).detach()
    return float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(advection.detach()).clamp_min(1.0e-30)
    )


def divergence_residual(flow: RotationalFlow, points: torch.Tensor) -> float:
    """Normalized incompressibility residual ``||div u|| L / ||u||`` (exact 0)."""

    pts = points.detach().to(torch.float64).clone().requires_grad_(True)
    u = _fields(flow, pts)[1]
    divergence = torch.zeros(pts.shape[0], dtype=torch.float64)
    for component in range(2):
        (grad_u,) = torch.autograd.grad(
            u[:, component].sum(), pts, create_graph=True
        )
        divergence = divergence + grad_u[:, component]
    return float(
        flow.radius
        * torch.linalg.vector_norm(divergence.detach())
        / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
    )


def helmholtz_residual(flow: RotationalFlow, points: torch.Tensor) -> float:
    r"""Normalized residual ``||(lap + c^2) psi|| L^2 / ||psi||`` (exact 0)."""

    pts = points.detach().to(torch.float64).clone().requires_grad_(True)
    psi = _fields(flow, pts)[0]
    (grad,) = torch.autograd.grad(psi.sum(), pts, create_graph=True)
    laplacian = torch.zeros_like(psi)
    for component in range(2):
        (second,) = torch.autograd.grad(
            grad[:, component].sum(), pts, create_graph=True
        )
        laplacian = laplacian + second[:, component]
    c = flow.coupling / flow.radius
    residual = ((laplacian + c * c * psi) * flow.radius**2).detach()
    return float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(psi.detach()).clamp_min(1.0e-30)
    )


@dataclass(frozen=True)
class EulerRotationalSample:
    """One exact rotational-Euler multi-field problem on a disk."""

    domain: DomainMesh
    targets: dict[str, torch.Tensor]
    flow: RotationalFlow


def build_euler_rotational_sample(
    seed: int,
    *,
    coupling_range: tuple[float, float],
    modes: tuple[int, ...],
    n_boundary: int = 64,
    n_query: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EulerRotationalSample:
    """Sample a disk, vorticity coupling, and balanced drive spectrum.

    Geometry randomization follows the screened testbed exactly: radius,
    center, and boundary-panel rotation offset are uniform; the solution's
    orientation is randomized through the mode phases.  All labels are
    computed in float64 before the final device/dtype cast.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(
            torch.empty((), dtype=torch.float64).uniform_(
                low, high, generator=generator
            )
        )

    radius = uniform(0.5, 2.0)
    center = torch.tensor(
        [uniform(-2.0, 2.0), uniform(-2.0, 2.0)], dtype=torch.float64
    )
    offset = uniform(0.0, 2.0 * math.pi)
    coupling = uniform(*coupling_range)
    if coupling >= FIRST_DIRICHLET_EIGENVALUE:
        raise ValueError(
            "coupling reaches the first Dirichlet eigenvalue j_{0,1}: the "
            "interior solution is not determined by boundary data there"
        )
    if coupling > _SERIES_MAX_ARGUMENT:
        raise ValueError("coupling exceeds the Bessel-series validity range")
    if max(modes) > _SERIES_MAX_ORDER:
        raise ValueError("mode order exceeds the Bessel-series validity range")

    coefficients = {
        m: torch.polar(
            torch.tensor(1.0 / math.sqrt(len(modes)), dtype=torch.float64),
            torch.tensor(uniform(0.0, 2.0 * math.pi), dtype=torch.float64),
        )
        for m in modes
    }
    flow = RotationalFlow(
        center=center, radius=radius, coupling=coupling, coefficients=coefficients
    )

    vertex_angles = (
        offset
        + 2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    )
    points = center + radius * torch.stack(
        (vertex_angles.cos(), vertex_angles.sin()), dim=-1
    )
    index = torch.arange(n_boundary)
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    midpoint_angles = vertex_angles + math.pi / n_boundary
    midpoints = center + radius * torch.stack(
        (midpoint_angles.cos(), midpoint_angles.sin()), dim=-1
    )
    boundary_velocity = velocity(flow, midpoints)

    query_r = (
        radius
        * 0.95
        * torch.sqrt(torch.rand(n_query, dtype=torch.float64, generator=generator))
    )
    query_theta = (
        2.0 * math.pi * torch.rand(n_query, dtype=torch.float64, generator=generator)
    )
    query_points = center + torch.stack(
        (query_r * query_theta.cos(), query_r * query_theta.sin()), dim=-1
    )
    target_velocity = velocity(flow, query_points)
    target_pressure = pressure(flow, query_points)
    if not bool(torch.isfinite(target_velocity).all()) or not bool(
        torch.isfinite(target_pressure).all()
    ):
        raise RuntimeError("the exact targets contain non-finite values")

    boundary = Mesh(
        points=points.to(device=device, dtype=dtype),
        cells=cells.to(device=device),
        cell_data={
            "boundary_velocity": boundary_velocity.to(device=device, dtype=dtype)
        },
    )
    interior = Mesh(points=query_points.to(device=device, dtype=dtype))
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": torch.tensor(radius, device=device, dtype=dtype),
            "vorticity_coupling": torch.tensor(
                coupling, device=device, dtype=dtype
            ),
        },
    )
    return EulerRotationalSample(
        domain=domain,
        targets={
            "velocity": target_velocity.to(device=device, dtype=dtype),
            "pressure": target_pressure.to(device=device, dtype=dtype),
        },
        flow=flow,
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

SPLITS: dict[str, dict] = {
    "in_distribution": {"coupling_range": (0.5, 1.8), "modes": (0, 1, 2, 3)},
    "unseen_coupling": {"coupling_range": (1.9, 2.2), "modes": (0, 1, 2, 3)},
    "unseen_modes": {"coupling_range": (0.5, 1.8), "modes": (4, 5, 6)},
    "near_eigenvalue": {"coupling_range": (2.25, 2.38), "modes": (0, 1, 2, 3)},
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class DriveBoundaryMean(nn.Module):
    """Parameter-free per-field boundary-mean floor.

    Predicts the boundary-measure mean of the declared per-field drive at
    every query: the weighted mean of ``boundary_velocity`` for the velocity
    (a constant polar vector -- the only O(2)-equivariant constant the data
    supplies) and zero for the pressure, which has no boundary drive, so its
    relative L2 is exactly 1.0 by construction.  A constant velocity with
    zero pressure gradient is itself an exact steady Euler solution, so this
    floor's momentum residual is exactly zero -- uninformative but
    consistent, the same pattern as the harmonic constant floors.
    """

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = domain.boundaries["dirichlet"]
        weights = boundary.cell_areas
        drive = boundary.cell_data["boundary_velocity"]
        mean = (weights[:, None] * drive).sum(dim=0) / weights.sum()
        n = domain.interior.n_points
        return domain.interior.with_data(
            point_data={
                "velocity": mean[None, :].repeat(n, 1),
                "pressure": drive.new_zeros((n,)),
            },
            cell_data={},
            global_data=domain.global_data,
        )


MODEL_NAMES: tuple[str, ...] = (
    "boundary_mean",
    "mt_singpair_linear",
    "mt_singpair_nl",
    "mt_singpair_nl_pseudo",
    "mt_singpair_q2",
    "mt_singpair_q2_pseudo",
)

#: Field mode and pseudo sector width per transformer arm.
_ARM_MODES: dict[str, tuple[str, int]] = {
    "mt_singpair_linear": ("linear", 0),
    "mt_singpair_nl": ("zero_preserving_nonlinear", 0),
    "mt_singpair_nl_pseudo": ("zero_preserving_nonlinear", 8),
    "mt_singpair_q2": ("quadratic", 0),
    "mt_singpair_q2_pseudo": ("quadratic", 8),
}


def _build_model(model_name: str) -> nn.Module:
    """Instantiate one arm of the rotational multi-field comparison.

    All transformer arms share the singpair kernel dictionary, the
    multi-field output declaration, the drive/operator schema, and the
    flipped one-head reference configuration (iteration 32: heads 1, ranks
    48/16 at the reference total score capacity); they differ only in field
    mode and in the presence of the pseudo channel sector.
    """

    if model_name == "boundary_mean":
        return DriveBoundaryMean()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}")
    capacity = asdict(MeshTransformerConfig())
    capacity.update(_ONE_HEAD_CONFIG)
    kernel_kwargs = dict(_SINGPAIR_KWARGS)
    field_mode, drive_pseudo_dim = _ARM_MODES[model_name]
    if drive_pseudo_dim:
        kernel_kwargs["drive_pseudo_dim"] = drive_pseudo_dim
    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks=dict(OUTPUT_FIELD_RANKS),
        boundary_field_ranks=_SCHEMA["boundary_field_ranks"],
        global_field_ranks=_SCHEMA["global_field_ranks"],
        reference_length_key="reference_length",
        field_mode=field_mode,
        query_decoder="kernel",
        **kernel_kwargs,
        **capacity,
    )


# ---------------------------------------------------------------------------
# Metrics, evaluation, and driver
# ---------------------------------------------------------------------------


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _combined_relative_l2(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> float:
    """Relative L2 over the concatenation of every output field."""

    numerator = sum(
        float((predictions[name] - targets[name]).square().sum()) for name in targets
    )
    denominator = sum(float(targets[name].square().sum()) for name in targets)
    return math.sqrt(numerator / max(denominator, 1.0e-30))


def _predictions(model: nn.Module, domain: DomainMesh) -> dict[str, torch.Tensor]:
    point_data = model(domain).point_data
    return {name: point_data[name] for name in OUTPUT_FIELD_RANKS}


def _multi_field_loss(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Equal-weight mean of per-field relative squared errors."""

    terms = [
        torch.sum((predictions[name] - targets[name]).square())
        / torch.sum(targets[name].square()).clamp_min(1.0e-30)
        for name in targets
    ]
    return sum(terms) / len(terms)


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    *,
    eval_seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Per-field and combined relative L2 per split, on frozen banks."""

    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS.items())):
        errors: dict[str, list[float]] = {"": [], "/velocity": [], "/pressure": []}
        for case in range(n_cases):
            sample = build_euler_rotational_sample(
                eval_seed + 7919 * case + 1_000_003 * split_index,
                device=device,
                dtype=dtype,
                **spec,
            )
            predictions = _predictions(model, sample.domain)
            errors[""].append(_combined_relative_l2(predictions, sample.targets))
            for field in OUTPUT_FIELD_RANKS:
                errors[f"/{field}"].append(
                    _relative_l2(predictions[field], sample.targets[field])
                )
        for suffix, values in errors.items():
            report[name + suffix] = sum(values) / len(values)
    return report


def _prediction_derivative_residuals(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
    split: str,
) -> tuple[float, float, float]:
    """One split's (helmholtz, momentum, divergence) prediction residuals.

    Float64 autograd at 32 interior queries, two cases (the bank's
    fidelity-block convention).  The velocity of every exact solution
    satisfies the VECTOR Helmholtz equation componentwise
    (``lap u = -c^2 u``, because derivatives of the Helmholtz
    streamfunction commute with the Laplacian), which is the licensed
    per-field strong form here; the cross-field strong form is the steady
    Euler momentum residual of the model's own two output fields, normalized
    by the exact advection norm; incompressibility is per-field for the
    velocity.  Exact labels score float-noise zero on all three.
    """

    spec = SPLITS[split]
    model_fp64 = model.double()
    model_fp64.eval()
    helmholtz_values, momentum_values, divergence_values = [], [], []
    for case in range(2):
        sample = build_euler_rotational_sample(
            seed + case, n_query=32, device=device, dtype=torch.float64, **spec
        )
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        out = model_fp64(domain).point_data
        u, p = out["velocity"], out["pressure"]
        length = sample.domain.global_data["reference_length"].reshape(())
        coupling = sample.domain.global_data["vorticity_coupling"].reshape(())
        laplacian = torch.zeros_like(u)
        advection = torch.zeros_like(u)
        divergence = torch.zeros(points.shape[0], dtype=torch.float64, device=device)
        if u.grad_fn is not None:
            for component in range(2):
                (gradient,) = torch.autograd.grad(
                    u[:, component].sum(),
                    points,
                    create_graph=True,
                    allow_unused=True,
                )
                if gradient is None:
                    continue
                advection[:, component] = (u * gradient).sum(dim=-1)
                divergence = divergence + gradient[:, component]
                for direction in range(2):
                    (second,) = torch.autograd.grad(
                        gradient[:, direction].sum(),
                        points,
                        create_graph=True,
                        allow_unused=True,
                    )
                    if second is not None:
                        laplacian[:, component] = (
                            laplacian[:, component] + second[:, direction]
                        )
        if p.grad_fn is not None:
            (grad_p,) = torch.autograd.grad(
                p.sum(), points, create_graph=True, allow_unused=True
            )
        else:
            grad_p = None
        if grad_p is None:
            grad_p = torch.zeros_like(u)
        momentum = advection.detach() + grad_p.detach()
        exact_scale = _exact_advection_norm(sample.flow, points.detach())
        helmholtz = (laplacian.detach() + (coupling / length) ** 2 * u.detach()) * (
            length**2
        )
        helmholtz_values.append(
            float(
                torch.linalg.vector_norm(helmholtz)
                / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
            )
        )
        momentum_values.append(
            float(torch.linalg.vector_norm(momentum) / max(exact_scale, 1.0e-30))
        )
        divergence_values.append(
            float(
                length.detach()
                * torch.linalg.vector_norm(divergence.detach())
                / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
            )
        )
    return (
        sum(helmholtz_values) / len(helmholtz_values),
        sum(momentum_values) / len(momentum_values),
        sum(divergence_values) / len(divergence_values),
    )


def _exact_advection_norm(flow: RotationalFlow, points: torch.Tensor) -> float:
    """The exact flow's ``||(u . grad) u||`` at the given points (the scale)."""

    pts = points.detach().to(torch.float64).clone().requires_grad_(True)
    u = _fields(flow, pts)[1]
    advection = torch.zeros_like(u)
    for component in range(2):
        (grad_u,) = torch.autograd.grad(
            u[:, component].sum(), pts, create_graph=True
        )
        advection[:, component] = (u * grad_u).sum(dim=-1)
    return float(torch.linalg.vector_norm(advection.detach()))


def fidelity_metrics(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split: the per-field vector-Helmholtz strong-form residual of the
    predicted velocity, the cross-field steady-Euler momentum residual of
    the model's own (velocity, pressure) pair, and the incompressibility
    residual -- all float64 autograd, two cases and 32 queries per split, so
    the block stays cheap relative to training.  No maximum principle is
    licensed: the velocity components are Helmholtz (oscillatory), not
    harmonic, and the pressure is a quadratic composite.
    """

    residuals = {
        name: _prediction_derivative_residuals(
            model, seed=seed, device=device, split=name
        )
        for name in sorted(SPLITS)
    }
    return {
        "pde_residual": {name: values[0] for name, values in residuals.items()},
        "pde_residual_note": (
            "||(lap + (c~/L)^2) u_pred|| L^2 / ||u_pred||: the exact velocity "
            "satisfies the vector Helmholtz equation componentwise and scores "
            "~0; a constant-velocity prediction scores exactly c~^2 (the "
            "boundary_mean calibration)"
        ),
        "momentum_residual": {name: values[1] for name, values in residuals.items()},
        "momentum_residual_note": (
            "||(u_pred . grad) u_pred + grad p_pred|| / ||(u . grad) u_exact||: "
            "the steady Euler momentum residual of the model's own two output "
            "fields -- the cross-field strong form; exact labels score ~0"
        ),
        "divergence_residual": {
            name: values[2] for name, values in residuals.items()
        },
        "divergence_residual_note": (
            "||div u_pred|| L / ||u_pred||; the exact velocity is "
            "divergence-free by construction"
        ),
        "max_principle_violation": None,
        "max_principle_note": (
            "n/a: the velocity components satisfy the oscillatory Helmholtz "
            "equation (no maximum principle) and the pressure is a quadratic "
            "composite, so no principle is licensed"
        ),
    }


_EQUATION = (
    "euler/rotational (interior disk, exact): 2D steady incompressible Euler "
    "with omega = c^2 psi (vorticity linear in the streamfunction), so "
    "lap psi + c^2 psi = 0 (Helmholtz, Bessel J series); u = (psi_y, -psi_x); "
    "p = -c^2 psi^2/2 - |u|^2/2 from the rotational Bernoulli relation "
    "H(psi) = H0 - c^2 psi^2/2 (gauge H0 = 0); targets = {velocity (rank 1, "
    "drive-linear), pressure (rank 0, drive-quadratic)}"
)

_DESIGN_NOTES = {
    "pre_registered_prediction": (
        "logged before the first euler_rotational training run: the "
        "field_mode='linear' arm is exactly odd in the boundary-velocity "
        "drive while the pressure target is exactly even (drive-quadratic), "
        "and the drive distribution is negation-symmetric, so its pressure "
        "relative L2 must stay pinned at ~1.0 (trained no-response) -- the "
        "third structural wall instance -- while its velocity should match "
        "the nonlinear arm's; the nonlinear arms' pressure error should drop"
    ),
    "well_posedness_note": (
        "steady Euler boundary-value problems are subtle in general; this "
        "family poses the LEARNING problem {boundary velocity, c~} -> "
        "interior fields on a family of certified exact solutions, with "
        "every split strictly below the first disk Dirichlet eigenvalue "
        "j_{0,1} = 2.404826 (at eigenvalues the interior is not determined "
        "by boundary data), not a claim about general Euler well-posedness"
    ),
    "pseudo_arm_note": (
        "no wall is claimed for the pseudo arm: the boundary-velocity drive "
        "and velocity target are both polar, so the map is representable "
        "with dot-only typing; drive_pseudo_dim=8 probes whether internal "
        "wedge-sourced pseudoscalar channels (the type of the streamfunction "
        "latent) help"
    ),
}


def run_experiment(
    *,
    model_name: str,
    steps: int,
    seed: int,
    device: str,
    output_dir: str,
    eval_cases: int = 16,
    save_checkpoint: bool = False,
    snapshot_steps: tuple[int, ...] = (),
) -> dict:
    """Train one arm on the in-distribution split and report every metric.

    ``save_checkpoint`` (default off: bitwise-identical runs and reports)
    additionally writes the reported (best-validation) state dict next to the
    JSON report so analysis scripts can reload the trained arm;
    ``snapshot_steps`` writes the RAW (not best-validation) training state at
    the named steps.
    """

    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    train_spec = SPLITS["in_distribution"]
    model = _build_model(model_name).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_lookup = set(snapshot_steps)

    def _save_state(name: str, extra: dict) -> None:
        torch.save(
            {
                "model": model_name,
                "family": FAMILY,
                "seed": seed,
                "steps": steps,
                "state_dict": {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                },
                **extra,
            },
            out / name,
        )

    best_state, best_val, history = None, float("inf"), []
    start_time = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-6)
        for step in range(1, steps + 1):
            model.train()
            sample = build_euler_rotational_sample(
                seed + 104_729 * step, device=device_t, dtype=dtype, **train_spec
            )
            loss = _multi_field_loss(
                _predictions(model, sample.domain), sample.targets
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step in snapshot_lookup:
                _save_state(
                    f"{FAMILY}_{model_name}_seed{seed}_step{step}.pt",
                    {"snapshot_step": step},
                )
            if step % 250 == 0 or step == steps:
                validation = evaluate_splits(
                    model,
                    eval_seed=71_000_011,
                    n_cases=4,
                    device=device_t,
                    dtype=dtype,
                )
                score = 0.5 * (
                    validation["in_distribution/velocity"]
                    + validation["in_distribution/pressure"]
                )
                history.append(
                    {
                        "step": step,
                        "validation_mean_per_field_relative_l2": score,
                        "validation_velocity": validation["in_distribution/velocity"],
                        "validation_pressure": validation["in_distribution/pressure"],
                    }
                )
                if score < best_val:
                    best_val = score
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        validation = evaluate_splits(
            model, eval_seed=71_000_011, n_cases=4, device=device_t, dtype=dtype
        )
        best_val = 0.5 * (
            validation["in_distribution/velocity"]
            + validation["in_distribution/pressure"]
        )

    checkpoint_name = None
    if save_checkpoint:
        checkpoint_name = f"{FAMILY}_{model_name}_seed{seed}.pt"
        _save_state(checkpoint_name, {})

    report = {
        "model": model_name,
        "family": FAMILY,
        "equation": _EQUATION,
        "seed": seed,
        "steps": steps,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_validation_mean_per_field_relative_l2": best_val,
        "checkpoint": checkpoint_name,
        "snapshot_steps": sorted(snapshot_lookup),
        "splits": evaluate_splits(
            model,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            device=device_t,
            dtype=dtype,
        ),
        "design_notes": _DESIGN_NOTES,
        "fidelity": fidelity_metrics(model, seed=83_000_019, device=device_t),
        "state": {
            k: v.tolist() for k, v in model.state_dict().items() if v.numel() <= 16
        },
    }
    (out / f"{FAMILY}_{model_name}_seed{seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-cases", type=int, default=16)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
        eval_cases=arguments.eval_cases,
        save_checkpoint=arguments.save_checkpoint,
        snapshot_steps=tuple(arguments.snapshot_steps),
    )
    print(json.dumps({k: result[k] for k in ("model", "family", "splits")}))
