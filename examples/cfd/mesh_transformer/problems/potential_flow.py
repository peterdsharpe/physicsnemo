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

r"""Fluid-flavored exterior benchmarks: 2D potential flow past a body, plus a
manufactured multi-scale boundary-layer tier.

Every earlier 2D family in this example is an *interior* Dirichlet or Neumann
problem driven by scalar boundary data.  This file adds the program's first
fluid-flavored well-posedness variant: an **exterior** domain whose drive is a
**global far-field vector**, and (tier two) a target with two intrinsic length
scales.  It is the entry point toward steady Euler/Navier--Stokes benchmarks.

**Family A -- exterior potential flow** (``family="potential_flow"``):
incompressible potential flow past a smooth closed body.  The body is the
image of the unit circle under a certified exterior conformal map

.. math::

   G(z) = z + \sum_{m \in S} b_m z^{-m}, \qquad \sum_m m\,|b_m| < 1,

followed by an orientation-preserving similarity (scale ``L``, rotation,
translation).  The coefficient bound plays the same role as in
``conformal_laplace``: since :math:`|z_1^{-m} - z_2^{-m}| \le m|z_1 - z_2|`
for :math:`|z_i| \ge 1`, it gives :math:`|G(z_1)-G(z_2)| \ge
(1-\kappa)|z_1-z_2|` on the closed exterior, so ``G`` is injective and the
body curve is simple *by certificate*, not by luck.  (The interior maps
:math:`z + \sum a_m z^m` of the 2D bank cannot be reused verbatim: a
polynomial is never injective near infinity, so exterior univalence forces
the negative-power modes.)  Higher modes and larger deformation give the
"wilder" star-like bodies of the OOD splits.

The exact flow is the circle theorem pushed through ``G``: around the unit
disk the complex potential is
:math:`F(\zeta) = \bar A \zeta + A/\zeta - (i\Gamma/2\pi)\log\zeta`, and a
conformal map preserves both harmonicity and the streamline property of the
body.  The **target is the nondimensional disturbance streamfunction**

.. math::

   \psi_d^*(\zeta) = \operatorname{Im}\!\big[u_c/\zeta
       - \bar u_c \textstyle\sum_m b_m \zeta^{-m}\big]
       - \frac{\Gamma^*}{2\pi}\,\ln|\zeta| ,

where ``u_c`` is the far-field direction expressed in the canonical frame and
:math:`\Gamma^*` the nondimensional circulation.  The streamfunction (not the
velocity potential) is the clean single-valued field here: with circulation
the potential is *multivalued* (its vortex part is :math:`\propto\arg\zeta`)
while :math:`\psi_d^*` stays single-valued; impermeability
:math:`n\cdot\nabla\phi=0` is equivalent to the body being the streamline
:math:`\psi^*_{\rm total}=0`, which is checked exactly.  Predicting the
velocity (a rank-1 *output*) is Family A' below.

MODEL-FACING PROBLEM.  The impermeability condition is homogeneous, so the
boundary carries **no scalar drive**: the drive is the far-field velocity
``U`` (|U| = 1, direction random) as a **global rank-1 drive field**
(``freestream_velocity``) plus the circulation scalar.  This is the first
benchmark where the MeshTransformer's vector-drive path is structurally
necessary: scalar-only arms (``operator_vector_dim=0``) reject rank-1 drives
by design, so every registered transformer arm here is a vector-channel arm.
For the scalar baselines (``boundary_mean``, ``pair_kernel``) the drive is
*scalarized* onto the boundary as ``boundary_value`` -- the exact Dirichlet
trace of the disturbance streamfunction,
:math:`-\operatorname{Im}[\bar u_c G(\zeta)]` on ``|zeta|=1``, which is just
the negated uniform-stream streamfunction sampled on the body (no solve
involved, equivalent data to ``{geometry, U}``).  Because the vortex term
vanishes on the body, this trace is **independent of circulation**: the
trace-driven baselines are structurally circulation-blind, which is part of
the benchmark's design (transformer arms receive :math:`\Gamma^*` as a
global scalar drive).  Note :math:`\Gamma^*` and :math:`\psi^*` are
pseudoscalars, so this family uses orientation-preserving similarities only.

THE PSEUDOSCALAR WALL.  The fluid benchmark's first training runs found
every global-drive transformer arm pinned at relative L2 :math:`\approx 1.0`
on Family A -- a trained *no-response*, not an optimization failure.  The
reason is representation-theoretic.  The target :math:`\psi_d^*` is a 2D
**pseudoscalar**: its uniform-flow part is the wedge
:math:`\psi_\infty = U \wedge (x - x_0)`, so mirroring the whole problem
(body, far field, queries) flips the target's sign.  The MeshTransformer's
type system carries only true scalars and polar vectors, and every scalar it
can emit is built from **dot products of polar vectors** -- such an output
equals its mirror image.  One line of group theory then forces the observed
wall: an O(2)-equivariant scalar-output model obeys
:math:`N[\sigma\cdot\text{data}](\sigma x) = N[\text{data}](x)`, while the
exact map obeys
:math:`\psi_d[\sigma\cdot\text{data}](\sigma x) = -\psi_d[\text{data}](x)`;
the only equivariant fit of an odd target by an even model is the zero
function, whose relative L2 is exactly one.  The wall was the type contract
enforcing itself, not a capacity or optimization deficiency.  (The
scalarized trace does not save the global-drive arms: they receive ``U`` as
a polar vector, and the pseudoscalar sign lives in the :math:`U \to \psi_d`
*map*, not in any drive value they are given.)  Two representable
reformulations resolve the wall without weakening the type system:

**Family A' -- velocity output** (``family="potential_flow_velocity"``):
identical geometry, drive schema, splits, and queries, but the target is the
nondimensional disturbance **velocity** :math:`u_d/|U| = \nabla\phi_d/|U|`
-- a **polar vector**, computed exactly from the complex-derivative
push-forward :math:`u - iv = W_d'(\zeta)\,/\,(e^{i\alpha} G'(\zeta))` of the
circle-theorem disturbance potential (the similarity scale cancels against
nondimensionalization).  The velocity is single-valued even with circulation
(only the *potential* is multivalued), and a polar-vector output transforms
with the mirror, :math:`u_d[\sigma\cdot\text{data}](\sigma x) = \sigma\,
u_d[\text{data}](x)`, so the parity obstruction vanishes.  Transformer arms
are wired with ``output_field_ranks={"velocity": 1}``; the kernel decoder
carries rank-1 outputs end-to-end through its vector value channels.  The
scalar controls (``boundary_mean``, ``pair_kernel``) cannot emit vectors and
are not registered for this family.

**Trace-driven arms** (``mesh_transformer_kernel_trace`` and
``mesh_transformer_kernel_singpair_trace``, Family A only): the transformer
receives the SAME certified scalarized drive the controls get -- the
disturbance-trace ``boundary_value`` as a per-cell scalar drive on the body
-- plus :math:`\Gamma^*` as a global scalar, and NOT the raw far-field
vector.  A scalar boundary drive flips sign together with the target under
mirroring, so the pseudoscalar-ness arrives baked into the *data* exactly as
for ``pair_kernel``, and the original :math:`\psi_d^*` target becomes
representable.  These arms separate "cannot represent the target type"
(fixed by Family A') from "cannot do the far-field-to-trace encode-solve"
(tested here); unlike the controls, they are not circulation-blind.

**Typed-circulation arms** (``mesh_transformer_kernel_pseudo`` and
``mesh_transformer_kernel_singpair_pseudo``, Family A' only): the velocity
output removes the *output-type* obstruction but leaves a second, subtler
wall on the *input* side.  The circulation part of the exact velocity is
:math:`\Gamma\,x^\perp/(2\pi|x|^2)`; with :math:`\Gamma` typed as a true
scalar that product is axial and hence unrepresentable by an
O(2)-equivariant polar-vector output -- measured as the ``circulation_ood``
split pinned at relative L2 0.647 across all velocity arms and seeds while
``in_distribution`` reached 0.14.  These arms declare the circulation with
the type system's new 2D pseudoscalar rank token,
``{"freestream_velocity": 1, "circulation": "0o"}``, so the rotation
product :math:`p\,x^\perp` of the pseudo sector carries it (see
:data:`_PSEUDO_SCHEMA` and
:mod:`physicsnemo.experimental.nn.mesh_attention.attention` for the full
failure record and product set).  Every pre-existing arm is untouched: the
pseudo sector is off (bitwise) unless declared.

**Family B -- boundary-layer tier** (``family="boundary_layer"``): a
manufactured multi-scale field

.. math::

   u^*(\zeta) = \psi_d^*(\zeta)\big|_{\Gamma=0}
       + f(\theta)\, e^{-(r-1)/\delta}, \qquad \zeta = r e^{i\theta},

posing exactly the representation problem GLOBE met with multi-scale kernels:
a thin exponential wall layer (:math:`\delta/L \ll 1`) superposed on a smooth
far field.  **This is not a Navier--Stokes solution** -- it is exact by
construction and makes no physics claim beyond the two-scale structure; the
Euler/N--S suites with a numerical solver come later.  The wall coordinates
are conformal: ``r - 1`` is a smooth (real-analytic) wall-distance coordinate
that matches physical distance to first order (physical distance
:math:`\approx L\,|G'(e^{i\theta})|\,(r-1)`, with ``|G'|`` pinned to
:math:`1\pm\kappa` by the certificate), and :math:`\theta` is the arc-length
parameter up to the same ``|G'|`` warp.  Conformal coordinates are chosen
over the true nearest-point distance because they are smooth across the
medial axis and exactly evaluable, keeping the benchmark solver-free.

The BC data determine the target: the amplitude profile ``f`` is band-limited
in :math:`\theta`, provided **as boundary cell data** (``layer_amplitude`` at
panel midpoints), :math:`\delta` is a **global operator scalar**
(``layer_thickness``, nondimensional), and ``U`` is the same global vector
drive.  The target is linear in ``(U, f)`` jointly and nonlinear in
:math:`\delta`, matching the drive/operator role split.  Half the queries are
sampled inside the layer (:math:`r-1 < 3\delta`), half in the far field, and
every split is reported with separate ``near_wall`` / ``far_field`` buckets
-- the multi-scale diagnostic.  The trace-driven baselines additionally never
see :math:`\delta` (they have no operator slot), so they are thickness-blind
near the wall by construction.

Certification (build time + tests): the coefficient bound, an all-pairs
simplicity check of the panel loop, winding checks (body encloses its anchor;
far queries are exterior), exact query membership via
:math:`|\zeta|\ge 1+\text{margin}` under the injectivity certificate, the
exact on-body streamline identity, an FD Laplacian check of the potential
part in *physical* coordinates (via Newton inversion of the map), and a
closed-form recomputation of the boundary-layer target from independently
inverted preimages.  The velocity target adds: agreement of the
:math:`dW/dz` push-forward with central FD gradients of the streamfunction
(with circulation) and of the potential (at zero circulation) in physical
coordinates to relative :math:`\le 10^{-5}`, a recomputation from
independently inverted preimages, and an explicit mirror test certifying
that :math:`\psi_d^*` flips sign while :math:`u_d` transforms as a polar
vector.  The trace-driven arms reuse the existing certified trace unchanged.

Conventions follow the linear benchmarks: panels at equal canonical parameter
with data at true parameter midpoints, the single boundary keeps the bank's
fixed ``"dirichlet"`` mesh key (the BC type lives in the data), cell winding
is counterclockwise so normals point *out of the fluid domain* (into the
body, as for the excluded disks of the encoder-stress family),
``reference_length`` is the similarity scale, and all generation is float64
on CPU with a device/dtype cast at the end.  This is a benchmark-local
research prototype, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace

import _paths  # noqa: F401
import torch
from conformal_laplace import (
    SimilarityTransform,
    complex_to_points,
    points_to_complex,
    sample_similarity,
    unit_circle,
)
from encoder_stress import mesh_cell_winding, polyline_is_simple, winding_number
from models import (
    BoundaryMean,
    InvariantPairKernel,
    MeshTransformerConfig,
    build_mesh_transformer,
)
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

_TWO_PI = 2.0 * math.pi


def _substream(seed: int, stream: int) -> int:
    """Derive independent deterministic seeds without mutable RNG state."""

    return seed + 15_485_863 * stream


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(
        torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator)
    )


# ---------------------------------------------------------------------------
# Exterior conformal bodies (star family, certified univalent outside the disk)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExteriorBody:
    r"""A certified exterior map plus its physical similarity placement.

    The canonical map is :math:`G(z) = z + \sum_m b_m z^{-m}` on
    :math:`|z| \ge 1` with the strict bound :math:`\sum_m m|b_m| < 1`, which
    certifies both :math:`G' \neq 0` and injectivity on the closed exterior
    (see the module docstring).  ``similarity`` must be orientation
    preserving: the complex push-forward of the circle-theorem flow -- and the
    pseudoscalar circulation/streamfunction pair -- assume a proper rotation.
    """

    modes: tuple[int, ...]
    coefficients: torch.Tensor  # complex128, shape (len(modes),)
    similarity: SimilarityTransform  # float64, det(rotation) = +1

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if any(isinstance(m, bool) or not isinstance(m, int) for m in modes):
            raise TypeError("modes must contain only integers")
        if any(m < 1 for m in modes):
            raise ValueError("exterior map modes must be at least one")
        if tuple(sorted(set(modes))) != modes:
            raise ValueError("modes must be unique and strictly increasing")
        object.__setattr__(self, "modes", modes)
        if not torch.is_complex(self.coefficients):
            raise TypeError("coefficients must be a complex tensor")
        if self.coefficients.shape != (len(modes),):
            raise ValueError("coefficients must have shape (len(modes),)")
        if not torch.isfinite(torch.view_as_real(self.coefficients)).all().item():
            raise ValueError("coefficients must be finite")
        if float(self.deformation_bound) >= 1.0:
            raise ValueError("sum(m * abs(b_m)) must be strictly less than one")
        if float(self.similarity.determinant) <= 0.0:
            raise ValueError("exterior bodies require an orientation-preserving map")

    @property
    def deformation_bound(self) -> torch.Tensor:
        r"""Return :math:`\kappa = \sum_m m |b_m|` (the injectivity margin)."""

        if not self.modes:
            return self.coefficients.real.new_zeros(())
        modes = self.coefficients.real.new_tensor(self.modes)
        return torch.sum(modes * torch.abs(self.coefficients))

    @property
    def rotation_factor(self) -> torch.Tensor:
        r"""Return the similarity rotation as the unit complex ``exp(i alpha)``."""

        rotation = self.similarity.rotation
        return torch.complex(rotation[0, 0], rotation[1, 0])

    @property
    def translation_complex(self) -> torch.Tensor:
        """Return the similarity translation as a complex scalar."""

        translation = self.similarity.translation
        return torch.complex(translation[0], translation[1])


def _negative_power_series(body: ExteriorBody, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate :math:`\sum_m b_m z^{-m}` (the map's deviation from identity)."""

    if not body.modes:
        return torch.zeros_like(z)
    powers = torch.stack(tuple(z**-mode for mode in body.modes), dim=-1)
    return torch.sum(body.coefficients * powers, dim=-1)


def exterior_map(body: ExteriorBody, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the canonical exterior map :math:`G(z)` for ``|z| >= 1``."""

    return z + _negative_power_series(body, z)


def exterior_map_derivative(body: ExteriorBody, z: torch.Tensor) -> torch.Tensor:
    r"""Evaluate :math:`G'(z) = 1 - \sum_m m\, b_m z^{-(m+1)}`."""

    if not body.modes:
        return torch.ones_like(z)
    modes = z.real.new_tensor(body.modes)
    powers = torch.stack(tuple(z ** -(mode + 1) for mode in body.modes), dim=-1)
    return 1.0 - torch.sum(modes * body.coefficients * powers, dim=-1)


def body_to_physical(body: ExteriorBody, z: torch.Tensor) -> torch.Tensor:
    """Map canonical complex coordinates to physical complex coordinates."""

    scale = body.similarity.scale.to(torch.float64)
    factor = scale * body.rotation_factor
    return body.translation_complex + factor * exterior_map(body, z)


def invert_body_map(
    body: ExteriorBody,
    physical: torch.Tensor,
    *,
    max_iterations: int = 64,
    tolerance: float = 1.0e-13,
) -> torch.Tensor:
    r"""Newton-invert the physical placement map at exterior points.

    Because :math:`|G - \mathrm{id}| < 1` and :math:`|G' - 1| < 1` on the
    closed exterior, Newton from the similarity-only initial guess converges
    for every point of the fluid domain.  The residual is verified after the
    loop; failure raises rather than returning silently wrong preimages.
    """

    scale = body.similarity.scale.to(torch.float64)
    factor = scale * body.rotation_factor
    w = (physical - body.translation_complex) / factor
    z = w.clone()
    for _ in range(max_iterations):
        residual = exterior_map(body, z) - w
        if float(residual.abs().max()) < tolerance:
            break
        z = z - residual / exterior_map_derivative(body, z)
    residual = exterior_map(body, z) - w
    if not float(residual.abs().max()) < 100.0 * tolerance:
        raise RuntimeError("Newton inversion of the exterior map did not converge")
    return z


def sample_exterior_body(
    coefficient_seed: int,
    similarity_seed: int,
    *,
    modes: tuple[int, ...],
    deformation_range: tuple[float, float],
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> ExteriorBody:
    r"""Sample a certified exterior body (star family).

    Coefficients are complex gaussians rescaled so :math:`\sum m|b_m|` equals
    a uniform draw from ``deformation_range`` (which must stay strictly below
    one -- the certificate).  The similarity is rotation + log-uniform scale +
    translation, reflections excluded.
    """

    lower, upper = deformation_range
    if not (0.0 <= lower <= upper < 1.0):
        raise ValueError("deformation_range must satisfy 0 <= lower <= upper < 1")
    generator = _generator(coefficient_seed)
    raw = torch.complex(
        torch.randn(len(modes), generator=generator, dtype=torch.float64),
        torch.randn(len(modes), generator=generator, dtype=torch.float64),
    )
    weighted_norm = torch.sum(raw.abs() * raw.real.new_tensor(modes)).clamp_min(1e-30)
    deformation = _uniform(generator, lower, upper)
    coefficients = raw * (deformation / weighted_norm)
    similarity = sample_similarity(
        similarity_seed,
        scale_range=scale_range,
        translation_extent=1.0,
        reflection=False,
    )
    return ExteriorBody(
        modes=tuple(modes), coefficients=coefficients, similarity=similarity
    )


# ---------------------------------------------------------------------------
# Exact flow: circle theorem pushed through the exterior map
# ---------------------------------------------------------------------------


def disturbance_streamfunction(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    r"""Nondimensional disturbance streamfunction at canonical points.

    .. math::

       \psi_d^*(\zeta) = \operatorname{Im}\big[u_c/\zeta
           - \bar u_c \sum_m b_m \zeta^{-m}\big]
           - (\Gamma^*/2\pi) \ln|\zeta| .

    This is :math:`(\psi_{\rm total} - \psi_\infty)/(L\,|U|)` with the
    uniform stream anchored at the body translation; it is the imaginary part
    of a holomorphic function plus a single-valued vortex term, hence exactly
    harmonic in physical coordinates for :math:`|\zeta| > 1` -- including
    when the (multivalued) disturbance *potential* would not be usable.
    """

    u_c = canonical_freestream
    holomorphic = u_c / z - torch.conj(u_c) * _negative_power_series(body, z)
    return holomorphic.imag - (circulation / _TWO_PI) * torch.log(z.abs())


def total_streamfunction(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    r"""Nondimensional total streamfunction; exactly zero on the body curve."""

    uniform = (torch.conj(canonical_freestream) * exterior_map(body, z)).imag
    disturbance = disturbance_streamfunction(body, canonical_freestream, circulation, z)
    return disturbance + uniform


def streamfunction_at_physical(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    points: torch.Tensor,
) -> torch.Tensor:
    """Disturbance streamfunction at physical (n, 2) points via Newton inversion.

    Used by the certification tests to finite-difference the target in
    *physical* coordinates, independently of the stored preimages.
    """

    z = invert_body_map(body, points_to_complex(points))
    return disturbance_streamfunction(body, canonical_freestream, circulation, z)


def disturbance_complex_potential(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    r"""Nondimensional disturbance complex potential at canonical points.

    .. math::

       W_d(\zeta) = u_c/\zeta - \bar u_c \sum_m b_m \zeta^{-m}
           - \frac{i\Gamma^*}{2\pi}\,\operatorname{Log}\zeta .

    ``Im`` of this is exactly :func:`disturbance_streamfunction` (the vortex
    term's imaginary part, :math:`-(\Gamma^*/2\pi)\ln|\zeta|`, is single
    valued for any branch).  ``Re`` is the disturbance velocity *potential*
    :math:`\phi_d^*`, which with circulation is **multivalued** -- the
    principal branch used here has a cut along the canonical negative real
    axis.  Certification therefore differentiates ``Re`` only at zero
    circulation, where :math:`W_d` is holomorphic and single valued on the
    whole exterior.
    """

    u_c = canonical_freestream
    holomorphic = u_c / z - torch.conj(u_c) * _negative_power_series(body, z)
    return holomorphic - 1j * (circulation / _TWO_PI) * torch.log(z)


def disturbance_complex_velocity(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    r"""Nondimensional physical-frame disturbance velocity ``u* + i v*``.

    The complex velocity is the derivative of the physical disturbance
    potential with respect to the physical coordinate
    :math:`w = t + L e^{i\alpha} G(\zeta)`:

    .. math::

       u_d - i v_d = \frac{dW_{d,\rm phys}}{dw}
           = |U|\,\frac{W_d'(\zeta)}{e^{i\alpha}\,G'(\zeta)} ,

    the similarity scale :math:`L` cancelling against the potential's
    nondimensionalization, so the returned quantity is
    :math:`(u_d + i v_d)/|U|` -- the conjugate of the expression above.  With

    .. math::

       W_d'(\zeta) = -u_c/\zeta^2 + \bar u_c\,(1 - G'(\zeta))
           - \frac{i\Gamma^*}{2\pi\zeta}

    (using :math:`\sum_m m\,b_m\zeta^{-(m+1)} = 1 - G'`), every term is
    single valued **including the circulation term** -- only the potential,
    never the velocity, is multivalued -- and :math:`G' \neq 0` on the closed
    exterior by the injectivity certificate.  Unlike the pseudoscalar
    :math:`\psi_d^*`, the velocity is a **polar vector**: mirroring the whole
    problem mirrors it, which is what makes it representable by the
    MeshTransformer's dot-product type system (see the module docstring).
    """

    u_c = canonical_freestream
    derivative = exterior_map_derivative(body, z)
    w_prime = (
        -u_c / (z * z)
        + torch.conj(u_c) * (1.0 - derivative)
        - 1j * (circulation / _TWO_PI) / z
    )
    return torch.conj(w_prime / (body.rotation_factor * derivative))


def disturbance_velocity(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    """Disturbance velocity as ``(n, 2)`` physical-frame Cartesian vectors."""

    return complex_to_points(
        disturbance_complex_velocity(body, canonical_freestream, circulation, z)
    )


def velocity_at_physical(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    points: torch.Tensor,
) -> torch.Tensor:
    """Disturbance velocity at physical ``(n, 2)`` points via Newton inversion.

    Used by the certification tests to recompute the velocity target from
    independently inverted preimages and to compare the exact push-forward
    against finite differences of the streamfunction/potential in physical
    coordinates.
    """

    z = invert_body_map(body, points_to_complex(points))
    return disturbance_velocity(body, canonical_freestream, circulation, z)


# ---------------------------------------------------------------------------
# Manufactured boundary layer (Family B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerProfile:
    r"""Band-limited amplitude ``f(theta)`` of the manufactured wall layer.

    ``f(theta) = constant + sum_k (cosine_k cos k theta + sine_k sin k theta)``
    over the declared modes; the exact circle RMS is
    ``sqrt(constant**2 + 0.5 * sum(cosine**2 + sine**2))``.
    """

    constant: float
    modes: tuple[int, ...]
    cosine: torch.Tensor  # float64, shape (len(modes),)
    sine: torch.Tensor  # float64, shape (len(modes),)

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in modes):
            raise ValueError("layer modes must be integers of at least one")
        if tuple(sorted(set(modes))) != modes:
            raise ValueError("layer modes must be unique and strictly increasing")
        object.__setattr__(self, "modes", modes)
        expected = (len(modes),)
        if self.cosine.shape != expected or self.sine.shape != expected:
            raise ValueError("cosine and sine must have shape (len(modes),)")

    @property
    def circle_rms(self) -> float:
        """Exact RMS of the profile over the canonical angle."""

        energy = self.constant**2 + 0.5 * float(
            (self.cosine.square() + self.sine.square()).sum()
        )
        return math.sqrt(energy)


def sample_layer_profile(
    seed: int,
    *,
    modes: tuple[int, ...],
    include_constant: bool,
) -> LayerProfile:
    """Sample a unit-RMS band-limited amplitude profile (flat in-band weights)."""

    generator = _generator(seed)
    constant = (
        float(torch.randn((), generator=generator, dtype=torch.float64))
        if include_constant
        else 0.0
    )
    cosine = torch.randn(len(modes), generator=generator, dtype=torch.float64)
    sine = torch.randn(len(modes), generator=generator, dtype=torch.float64)
    raw = LayerProfile(constant=constant, modes=tuple(modes), cosine=cosine, sine=sine)
    normalization = 1.0 / max(raw.circle_rms, 1.0e-30)
    return LayerProfile(
        constant=constant * normalization,
        modes=tuple(modes),
        cosine=cosine * normalization,
        sine=sine * normalization,
    )


def evaluate_layer_profile(profile: LayerProfile, theta: torch.Tensor) -> torch.Tensor:
    """Evaluate ``f(theta)`` (branch independent: only ``cos``/``sin`` enter)."""

    value = torch.full_like(theta, profile.constant)
    for index, mode in enumerate(profile.modes):
        value = (
            value
            + profile.cosine[index] * torch.cos(mode * theta)
            + profile.sine[index] * torch.sin(mode * theta)
        )
    return value


def boundary_layer_term(
    profile: LayerProfile, layer_thickness: float, z: torch.Tensor
) -> torch.Tensor:
    r"""Evaluate the exact wall-layer term ``f(theta) exp(-(r-1)/delta)``."""

    radius = z.abs()
    return evaluate_layer_profile(profile, torch.atan2(z.imag, z.real)) * torch.exp(
        -(radius - 1.0) / layer_thickness
    )


# ---------------------------------------------------------------------------
# Sample assembly and certification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PotentialFlowSample:
    """One exact exterior-flow problem plus its certification payload.

    ``layer_thickness`` is ``nan`` and ``layer_profile``/``near_wall_mask``
    are ``None`` for the pure potential-flow families.  ``target`` has shape
    ``(n_query,)`` for the pseudoscalar streamfunction families and
    ``(n_query, 2)`` for the polar-vector ``potential_flow_velocity`` family.
    """

    domain: DomainMesh
    target: torch.Tensor
    family: str
    body: ExteriorBody
    freestream: torch.Tensor  # (2,) float64 physical unit vector
    canonical_freestream: torch.Tensor  # complex128 scalar u_c
    circulation: float  # nondimensional Gamma*
    layer_thickness: float
    layer_profile: LayerProfile | None
    query_preimages: torch.Tensor  # complex128 (n_query,), cpu
    boundary_midpoint_preimages: torch.Tensor  # complex128 (n_boundary,), cpu
    boundary_loop: torch.Tensor  # (n_boundary, 2) float64 CCW physical vertices
    boundary_midpoints: torch.Tensor  # (n_boundary, 2) float64 physical
    near_wall_mask: torch.Tensor | None  # bool (n_query,) on the sample device


def _boundary_panels(
    body: ExteriorBody, n_boundary: int, phase: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-parameter panel vertices/midpoints (physical) and their preimages."""

    steps = torch.arange(n_boundary, dtype=torch.float64)
    vertex_angles = _TWO_PI * (steps + phase) / n_boundary
    midpoint_angles = vertex_angles + math.pi / n_boundary
    vertex_preimages = unit_circle(vertex_angles)
    midpoint_preimages = unit_circle(midpoint_angles)
    vertices = complex_to_points(body_to_physical(body, vertex_preimages))
    midpoints = complex_to_points(body_to_physical(body, midpoint_preimages))
    return vertices, midpoints, vertex_preimages, midpoint_preimages


def _certify_sample(
    body: ExteriorBody,
    loop: torch.Tensor,
    boundary: Mesh,
    query_preimages: torch.Tensor,
    query_points: torch.Tensor,
    target: torch.Tensor,
    *,
    min_query_radius: float,
) -> None:
    """Build-time certification of geometry, membership, and finiteness.

    Query membership in the fluid is *exact*: the injectivity certificate
    plus ``|zeta| >= 1 + margin``.  The panel-polygon winding test is applied
    only to queries with ``|zeta| >= 1.2``, where chord sag cannot flip
    membership; the polygon is an input discretization, not the geometry.
    """

    if float(body.deformation_bound) >= 1.0:
        raise RuntimeError("exterior map lost its injectivity certificate")
    if not polyline_is_simple(loop):
        raise RuntimeError("the body panel loop self-intersects")
    anchor = body.similarity.translation[None, :]
    if int(winding_number(loop, anchor)[0]) != 1:
        raise RuntimeError("the body loop must wind +1 (CCW) around its anchor")
    if float(query_preimages.abs().min()) < min_query_radius - 1.0e-12:
        raise RuntimeError("a query preimage violates the exterior margin")
    far = query_preimages.abs() >= 1.2
    if bool(far.any()):
        winding = mesh_cell_winding(boundary, query_points[far])
        if not bool((winding == 0).all()):
            raise RuntimeError("a far query point is not exterior to the panel loop")
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("the exact target contains non-finite values")


def _finalize_sample(
    *,
    family: str,
    body: ExteriorBody,
    freestream: torch.Tensor,
    canonical_freestream: torch.Tensor,
    circulation: float,
    layer_thickness: float,
    layer_profile: LayerProfile | None,
    loop: torch.Tensor,
    midpoints: torch.Tensor,
    midpoint_preimages: torch.Tensor,
    query_preimages: torch.Tensor,
    query_points: torch.Tensor,
    target: torch.Tensor,
    boundary_cell_data: dict[str, torch.Tensor],
    global_scalars: dict[str, float],
    near_wall_mask: torch.Tensor | None,
    min_query_radius: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PotentialFlowSample:
    n = loop.shape[0]
    index = torch.arange(n)
    # Counterclockwise cells: mesh edge normals (a CCW rotation of the
    # directed edge) then point *into* the body, i.e. out of the fluid
    # domain -- the excluded-disk convention of the encoder-stress family.
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    certification_boundary = Mesh(points=loop, cells=cells)
    _certify_sample(
        body,
        loop,
        certification_boundary,
        query_preimages,
        query_points,
        target,
        min_query_radius=min_query_radius,
    )
    boundary = Mesh(
        points=loop.to(device=device, dtype=dtype),
        cells=cells.to(device),
        cell_data={
            key: value.to(device=device, dtype=dtype)
            for key, value in boundary_cell_data.items()
        },
    )
    interior = Mesh(points=query_points.to(device=device, dtype=dtype))
    global_data = {
        "reference_length": body.similarity.scale.to(device=device, dtype=dtype),
        "freestream_velocity": freestream.to(device=device, dtype=dtype),
    }
    for key, value in global_scalars.items():
        global_data[key] = torch.tensor(value, device=device, dtype=dtype)
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data=global_data,
    )
    return PotentialFlowSample(
        domain=domain,
        target=target.to(device=device, dtype=dtype),
        family=family,
        body=body,
        freestream=freestream,
        canonical_freestream=canonical_freestream,
        circulation=circulation,
        layer_thickness=layer_thickness,
        layer_profile=layer_profile,
        query_preimages=query_preimages,
        boundary_midpoint_preimages=midpoint_preimages,
        boundary_loop=loop,
        boundary_midpoints=midpoints,
        near_wall_mask=(
            None if near_wall_mask is None else near_wall_mask.to(device=device)
        ),
    )


def _sample_freestream(
    seed: int, body: ExteriorBody, circulation_magnitude_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Unit far-field vector, its canonical complex form, and Gamma*."""

    generator = _generator(seed)
    beta = _uniform(generator, 0.0, _TWO_PI)
    freestream = torch.tensor([math.cos(beta), math.sin(beta)], dtype=torch.float64)
    canonical = torch.conj(body.rotation_factor) * torch.complex(
        freestream[0], freestream[1]
    )
    magnitude = _uniform(generator, *circulation_magnitude_range)
    sign = 1.0 if _uniform(generator, 0.0, 1.0) < 0.5 else -1.0
    return freestream, canonical, sign * magnitude


def _annulus_preimages(
    seed: int, n_query: int, radius_range: tuple[float, float]
) -> torch.Tensor:
    """Log-uniform-in-radius, uniform-in-angle exterior canonical points."""

    generator = _generator(seed)
    log_low, log_high = math.log(radius_range[0]), math.log(radius_range[1])
    radii = torch.exp(
        log_low
        + (log_high - log_low)
        * torch.rand(n_query, generator=generator, dtype=torch.float64)
    )
    angles = _TWO_PI * torch.rand(n_query, generator=generator, dtype=torch.float64)
    return torch.polar(radii, angles)


def _build_exterior_flow_sample(
    seed: int,
    *,
    family: str,
    modes: tuple[int, ...],
    deformation_range: tuple[float, float],
    circulation_magnitude_range: tuple[float, float],
    query_radius_range: tuple[float, float],
    n_boundary: int,
    n_query: int,
    scale_range: tuple[float, float],
    device: torch.device | str,
    dtype: torch.dtype,
) -> PotentialFlowSample:
    """Shared exterior-flow sample core for Families A and A'.

    The two families share every random substream (body, freestream,
    circulation, queries, panel phase), so a ``potential_flow`` sample and a
    ``potential_flow_velocity`` sample at the same seed describe the *same*
    flow and differ only in which exact field is the target.
    """

    body = sample_exterior_body(
        _substream(seed, 0),
        _substream(seed, 1),
        modes=modes,
        deformation_range=deformation_range,
        scale_range=scale_range,
    )
    freestream, canonical_freestream, circulation = _sample_freestream(
        _substream(seed, 2), body, circulation_magnitude_range
    )
    query_preimages = _annulus_preimages(
        _substream(seed, 3), n_query, query_radius_range
    )
    phase = _uniform(_generator(_substream(seed, 4)), 0.0, 1.0)
    loop, midpoints, _, midpoint_preimages = _boundary_panels(body, n_boundary, phase)
    query_points = complex_to_points(body_to_physical(body, query_preimages))
    if family == "potential_flow_velocity":
        target = disturbance_velocity(
            body, canonical_freestream, circulation, query_preimages
        )
    else:
        target = disturbance_streamfunction(
            body, canonical_freestream, circulation, query_preimages
        )
    trace = disturbance_streamfunction(
        body, canonical_freestream, circulation, midpoint_preimages
    )
    return _finalize_sample(
        family=family,
        body=body,
        freestream=freestream,
        canonical_freestream=canonical_freestream,
        circulation=circulation,
        layer_thickness=math.nan,
        layer_profile=None,
        loop=loop,
        midpoints=midpoints,
        midpoint_preimages=midpoint_preimages,
        query_preimages=query_preimages,
        query_points=query_points,
        target=target,
        boundary_cell_data={"boundary_value": trace},
        global_scalars={"circulation": circulation},
        near_wall_mask=None,
        min_query_radius=query_radius_range[0],
        device=device,
        dtype=dtype,
    )


def build_potential_flow_sample(
    seed: int,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    deformation_range: tuple[float, float] = (0.05, 0.35),
    circulation_magnitude_range: tuple[float, float] = (0.0, 1.5),
    query_radius_range: tuple[float, float] = (1.05, 4.0),
    n_boundary: int = 160,
    n_query: int = 256,
    scale_range: tuple[float, float] = (0.5, 2.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PotentialFlowSample:
    """Build one exact exterior potential-flow problem (Family A).

    The target is the nondimensional disturbance streamfunction at canonical
    annulus queries (log-uniform radius) pushed to physical coordinates.  The
    ``boundary_value`` cell data is the exact scalarized drive (the
    disturbance trace, circulation-independent); the global-drive transformer
    arms do not declare it and consume ``freestream_velocity``/
    ``circulation`` from ``global_data`` instead, while the trace-driven arms
    declare it (plus ``circulation``) precisely because the pseudoscalar
    target is unrepresentable from the polar-vector drive alone (see the
    module docstring).
    """

    return _build_exterior_flow_sample(
        seed,
        family="potential_flow",
        modes=modes,
        deformation_range=deformation_range,
        circulation_magnitude_range=circulation_magnitude_range,
        query_radius_range=query_radius_range,
        n_boundary=n_boundary,
        n_query=n_query,
        scale_range=scale_range,
        device=device,
        dtype=dtype,
    )


def build_potential_flow_velocity_sample(
    seed: int,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    deformation_range: tuple[float, float] = (0.05, 0.35),
    circulation_magnitude_range: tuple[float, float] = (0.0, 1.5),
    query_radius_range: tuple[float, float] = (1.05, 4.0),
    n_boundary: int = 160,
    n_query: int = 256,
    scale_range: tuple[float, float] = (0.5, 2.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PotentialFlowSample:
    r"""Build one exact exterior-flow problem with a velocity target (A').

    Same flow, geometry, drive data, and random substreams as
    :func:`build_potential_flow_sample`; only the target changes, to the
    nondimensional disturbance **velocity** ``(n_query, 2)`` -- exact via the
    complex-derivative push-forward :math:`W_d'(\zeta)/(e^{i\alpha}
    G'(\zeta))`.  Background: the streamfunction target of Family A is a 2D
    pseudoscalar (mirror-odd; its uniform-flow part is the wedge
    :math:`U \wedge x`), which the MeshTransformer's dot-product-only scalar
    outputs provably cannot represent -- the trained relative L2
    :math:`\approx 1` wall.  The velocity is a polar vector and hence
    representable; it is also single-valued even with circulation, unlike
    the velocity potential.
    """

    return _build_exterior_flow_sample(
        seed,
        family="potential_flow_velocity",
        modes=modes,
        deformation_range=deformation_range,
        circulation_magnitude_range=circulation_magnitude_range,
        query_radius_range=query_radius_range,
        n_boundary=n_boundary,
        n_query=n_query,
        scale_range=scale_range,
        device=device,
        dtype=dtype,
    )


def build_boundary_layer_sample(
    seed: int,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    deformation_range: tuple[float, float] = (0.05, 0.35),
    delta_range: tuple[float, float] = (0.02, 0.05),
    amplitude_modes: tuple[int, ...] = (1, 2, 3, 4),
    amplitude_include_constant: bool = True,
    outer_query_radius: float = 4.0,
    n_boundary: int = 192,
    n_query: int = 256,
    scale_range: tuple[float, float] = (0.5, 2.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PotentialFlowSample:
    r"""Build one manufactured multi-scale boundary-layer problem (Family B).

    Half the queries sit inside the wall layer (:math:`r - 1 \in
    [0.05, 2.95]\,\delta`), half in the far field (log-uniform in
    :math:`[1 + 3.05\delta,\ \text{outer}]`), so the ``near_wall_mask``
    (:math:`r - 1 < 3\delta`) splits them exactly.  Circulation is zero in
    this family; the smooth part of the target is the Family A disturbance
    streamfunction.
    """

    body = sample_exterior_body(
        _substream(seed, 0),
        _substream(seed, 1),
        modes=modes,
        deformation_range=deformation_range,
        scale_range=scale_range,
    )
    freestream, canonical_freestream, _ = _sample_freestream(
        _substream(seed, 2), body, (0.0, 0.0)
    )
    circulation = 0.0
    generator = _generator(_substream(seed, 5))
    delta = _uniform(generator, *delta_range)
    profile = sample_layer_profile(
        _substream(seed, 6),
        modes=amplitude_modes,
        include_constant=amplitude_include_constant,
    )

    n_near = n_query // 2
    query_generator = _generator(_substream(seed, 3))
    near_wall = 1.0 + delta * (
        0.05 + 2.90 * torch.rand(n_near, generator=query_generator, dtype=torch.float64)
    )
    log_low = math.log(1.0 + 3.05 * delta)
    log_high = math.log(outer_query_radius)
    far_field = torch.exp(
        log_low
        + (log_high - log_low)
        * torch.rand(n_query - n_near, generator=query_generator, dtype=torch.float64)
    )
    radii = torch.cat((near_wall, far_field))
    angles = _TWO_PI * torch.rand(
        n_query, generator=query_generator, dtype=torch.float64
    )
    query_preimages = torch.polar(radii, angles)
    near_wall_mask = query_preimages.abs() - 1.0 < 3.0 * delta

    phase = _uniform(_generator(_substream(seed, 4)), 0.0, 1.0)
    loop, midpoints, _, midpoint_preimages = _boundary_panels(body, n_boundary, phase)
    query_points = complex_to_points(body_to_physical(body, query_preimages))
    smooth = disturbance_streamfunction(
        body, canonical_freestream, circulation, query_preimages
    )
    target = smooth + boundary_layer_term(profile, delta, query_preimages)
    midpoint_angles = torch.atan2(midpoint_preimages.imag, midpoint_preimages.real)
    layer_amplitude = evaluate_layer_profile(profile, midpoint_angles)
    trace = (
        disturbance_streamfunction(
            body, canonical_freestream, circulation, midpoint_preimages
        )
        + layer_amplitude
    )
    return _finalize_sample(
        family="boundary_layer",
        body=body,
        freestream=freestream,
        canonical_freestream=canonical_freestream,
        circulation=circulation,
        layer_thickness=delta,
        layer_profile=profile,
        loop=loop,
        midpoints=midpoints,
        midpoint_preimages=midpoint_preimages,
        query_preimages=query_preimages,
        query_points=query_points,
        target=target,
        boundary_cell_data={
            "boundary_value": trace,
            "layer_amplitude": layer_amplitude,
        },
        global_scalars={"layer_thickness": delta},
        near_wall_mask=near_wall_mask,
        min_query_radius=1.0 + 0.049 * delta,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Splits, model registry, evaluation, and driver
# ---------------------------------------------------------------------------

SPLITS: dict[str, dict[str, dict]] = {
    "potential_flow": {
        "in_distribution": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "circulation_magnitude_range": (0.0, 1.5),
            "query_radius_range": (1.05, 4.0),
        },
        "unseen_geometry_modes": {
            "modes": (4, 5, 6),
            "deformation_range": (0.05, 0.35),
            "circulation_magnitude_range": (0.0, 1.5),
            "query_radius_range": (1.05, 4.0),
        },
        "wilder_shapes": {
            "modes": (1, 2, 3),
            "deformation_range": (0.45, 0.70),
            "circulation_magnitude_range": (0.0, 1.5),
            "query_radius_range": (1.05, 4.0),
        },
        "circulation_ood": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "circulation_magnitude_range": (2.5, 4.5),
            "query_radius_range": (1.05, 4.0),
        },
        "farfield_queries": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "circulation_magnitude_range": (0.0, 1.5),
            "query_radius_range": (4.0, 8.0),
        },
    },
    "boundary_layer": {
        "in_distribution": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "delta_range": (0.02, 0.05),
            "amplitude_modes": (1, 2, 3, 4),
            "amplitude_include_constant": True,
        },
        "thinner_layer": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "delta_range": (0.005, 0.012),
            "amplitude_modes": (1, 2, 3, 4),
            "amplitude_include_constant": True,
        },
        "unseen_amplitude_frequencies": {
            "modes": (1, 2, 3),
            "deformation_range": (0.05, 0.35),
            "delta_range": (0.02, 0.05),
            "amplitude_modes": (6, 7, 8, 9, 10),
            "amplitude_include_constant": False,
        },
    },
}

# Family A' reuses the Family A splits verbatim: same flows, same OOD axes,
# different exact target field (disturbance velocity instead of the
# pseudoscalar streamfunction).
SPLITS["potential_flow_velocity"] = {
    name: dict(spec) for name, spec in SPLITS["potential_flow"].items()
}

SAMPLE_BUILDERS = {
    "potential_flow": build_potential_flow_sample,
    "potential_flow_velocity": build_potential_flow_velocity_sample,
    "boundary_layer": build_boundary_layer_sample,
}

# Model-facing schemas.  Family A: no boundary drive (impermeability is
# homogeneous); the drive is the global far-field vector plus circulation.
# Family A' (velocity target) keeps the identical drive schema.  Family B:
# the layer amplitude is a scalar boundary drive, the thickness is a global
# *operator* scalar (the target is nonlinear in delta but linear in (U, f)
# jointly), and the far field remains a global vector drive.
_FAMILY_SCHEMAS: dict[str, dict] = {
    "potential_flow": {
        "boundary_field_ranks": {"dirichlet": {"operator": {}, "drive": {}}},
        "global_field_ranks": {
            "operator": {},
            "drive": {"circulation": 0, "freestream_velocity": 1},
        },
    },
    "potential_flow_velocity": {
        "boundary_field_ranks": {"dirichlet": {"operator": {}, "drive": {}}},
        "global_field_ranks": {
            "operator": {},
            "drive": {"circulation": 0, "freestream_velocity": 1},
        },
    },
    "boundary_layer": {
        "boundary_field_ranks": {
            "dirichlet": {"operator": {}, "drive": {"layer_amplitude": 0}}
        },
        "global_field_ranks": {
            "operator": {"layer_thickness": 0},
            "drive": {"freestream_velocity": 1},
        },
    },
}

# The trace-driven transformer arms (Family A only) receive the SAME
# certified scalarized drive the controls get -- the disturbance trace as a
# per-cell scalar boundary drive -- plus the circulation as a global scalar,
# and deliberately NOT the raw far-field vector: under mirroring a scalar
# boundary drive flips sign together with the pseudoscalar target, so the
# odd part of the solution map arrives baked into the data exactly as for
# ``pair_kernel`` (which fits it), isolating the far-field-to-trace
# encode-solve from the type obstruction fixed by Family A'.
_TRACE_SCHEMA: dict[str, dict] = {
    "boundary_field_ranks": {
        "dirichlet": {"operator": {}, "drive": {"boundary_value": 0}}
    },
    "global_field_ranks": {"operator": {}, "drive": {"circulation": 0}},
}

# THE SECOND PSEUDOSCALAR WALL, AND THE TYPED-CIRCULATION ARMS.  Even with
# the polar-vector output of Family A', the *circulation part* of the exact
# velocity is Gamma * x_perp / (2 pi |x|^2).  With Gamma declared as a true
# scalar (rank 0) that product is axial -- under the combined data-and-frame
# mirror it acquires the wrong sign for a polar output -- so an
# O(2)-equivariant model provably cannot emit it, and the measured velocity
# arms sat at relative L2 0.647 on the ``circulation_ood`` split across
# every arm and seed while ``in_distribution`` reached 0.14 (the 0.647 floor
# is the norm fraction of the unrepresentable circulation component on that
# split).  The ``*_pseudo`` arms below (Family A' only) fix the *type* of
# the datum instead of re-encoding the data: circulation is declared with
# the MeshTransformer's pseudoscalar rank token ``"0o"``
# (``{"freestream_velocity": 1, "circulation": "0o"}``) and the model runs
# with ``drive_pseudo_dim`` pseudo channels, whose rotation product
# ``p * x_perp`` makes Gamma * x_perp representable.  PRE-REGISTERED
# PREDICTION (logged before the first typed-Gamma training run): with Gamma
# declared pseudoscalar, ``circulation_ood`` leaves the 0.647 floor and
# joins the in-distribution error level.
_PSEUDO_SCHEMA: dict[str, dict] = {
    "boundary_field_ranks": {"dirichlet": {"operator": {}, "drive": {}}},
    "global_field_ranks": {
        "operator": {},
        "drive": {"circulation": "0o", "freestream_velocity": 1},
    },
}

# Pseudo channel width for the typed-circulation arms; every other setting
# is inherited unchanged from the shared MeshTransformerConfig.
_PSEUDO_ARM_KWARGS: dict[str, int] = {"drive_pseudo_dim": 8}

# One-head capacity trade for the ``*_h1`` probe arms (iteration 32,
# pre-registered in ``studies/one_head_cross_suite.py``): one full-width
# attention head at quadrupled per-head ranks, holding the total score
# capacity heads * (scalar_rank + D * vector_rank) at the reference value
# 4*(12 + 2*4) = 1*(48 + 2*16) = 80, so the score-coefficient class is
# matched by construction and the arm differs only in the un-blocked
# per-head value/output structure.  The pseudo sector rides the scalar
# moment machinery, so ``drive_pseudo_dim`` is untouched (its per-head
# split 8/heads changes from 2 to 8 exactly as the scalar channels do).
_ONE_HEAD_CONFIG: dict[str, int] = {"heads": 1, "scalar_rank": 48, "vector_rank": 16}

# Named point prediction per family.  The scalar families predict the
# pseudoscalar streamfunction under the bank's historical "potential" key;
# Family A' predicts the rank-1 disturbance velocity.
_OUTPUT_FIELD_RANKS: dict[str, dict[str, int]] = {
    "potential_flow": {"potential": 0},
    "potential_flow_velocity": {"velocity": 1},
    "boundary_layer": {"potential": 0},
}

_TARGET_KEYS = {
    family: next(iter(spec)) for family, spec in _OUTPUT_FIELD_RANKS.items()
}

# Kernel-dictionary ablation kwargs shared by every transformer arm family.
_KERNEL_ARM_KWARGS: dict[str, dict] = {
    "mesh_transformer_kernel": {},
    "mesh_transformer_kernel_singonly": {
        "kernel_mlp_members": 0,
        "kernel_include_polynomial_members": False,
    },
    "mesh_transformer_kernel_singpair": {
        "kernel_mlp_members": 0,
        "kernel_include_polynomial_members": False,
        "kernel_include_single_layer_member": True,
    },
}

# Per-family registries.  The scalar controls cannot emit a rank-1 output,
# so they are absent from the velocity family; the trace-driven arms exist
# only where the certified trace is the load-bearing reformulation (Family
# A, whose global-drive target is otherwise unrepresentable).
FAMILY_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "potential_flow": (
        "boundary_mean",
        "pair_kernel",
        "mesh_transformer_kernel",
        "mesh_transformer_kernel_singonly",
        "mesh_transformer_kernel_singpair",
        "mesh_transformer_kernel_trace",
        "mesh_transformer_kernel_singpair_trace",
    ),
    "potential_flow_velocity": (
        "mesh_transformer_kernel",
        "mesh_transformer_kernel_singonly",
        "mesh_transformer_kernel_singpair",
        "mesh_transformer_kernel_pseudo",
        "mesh_transformer_kernel_singpair_pseudo",
        "mesh_transformer_kernel_singpair_pseudo_h1",
    ),
    "boundary_layer": (
        "boundary_mean",
        "pair_kernel",
        "mesh_transformer_kernel",
        "mesh_transformer_kernel_singonly",
        "mesh_transformer_kernel_singpair",
    ),
}

MODEL_NAMES = tuple(
    dict.fromkeys(name for names in FAMILY_MODEL_NAMES.values() for name in names)
)


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _build_transformer_arm(
    family: str,
    schema: dict,
    kernel_kwargs: dict,
    config: MeshTransformerConfig | None = None,
) -> nn.Module:
    """Construct one MeshTransformer arm with the family's output type.

    Scalar-output families go through the shared ``models.py`` passthrough
    (bitwise-identical to the historical arms).  The velocity family needs
    ``output_field_ranks={"velocity": 1}`` -- a rank-1 polar-vector
    prediction, which the kernel decoder carries end-to-end through its
    vector value channels -- so it constructs the model directly with
    otherwise identical conventions.  ``config`` (default ``None``: the
    shared benchmark capacity, bitwise-identical arms) lets the ``*_h1``
    probe arms swap in the one-head capacity trade without touching any
    other convention.
    """

    if config is None:
        config = MeshTransformerConfig()
    output_field_ranks = _OUTPUT_FIELD_RANKS[family]
    if output_field_ranks == {"potential": 0}:
        return build_mesh_transformer(
            config,
            query_decoder="kernel",
            **kernel_kwargs,
            **schema,
        )
    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks=dict(output_field_ranks),
        boundary_field_ranks=schema["boundary_field_ranks"],
        global_field_ranks=schema["global_field_ranks"],
        reference_length_key="reference_length",
        field_mode="linear",
        query_decoder="kernel",
        **kernel_kwargs,
        **asdict(config),
    )


def _build_model(
    model_name: str,
    family: str,
    *,
    bounded_gates: bool = False,
    bounded_query: bool = False,
    decaying_drive: bool = False,
    monopole_free_sl: bool = False,
) -> nn.Module:
    """Instantiate one arm of the vector-drive comparison.

    ``boundary_mean`` and ``pair_kernel`` consume the scalarized drive trace
    (``boundary_value``); they are circulation-blind by construction and, in
    Family B, thickness-blind.  The global-drive transformer arms consume the
    rank-1 far field -- scalar-only ablations reject it by design, so there
    is no scalar-only transformer arm in this registry.  The ``*_trace``
    arms swap the global drive for the certified trace plus circulation
    (:data:`_TRACE_SCHEMA`), and the velocity family's arms predict a rank-1
    output; both are the pseudoscalar-wall reformulations described in the
    module docstring.  The ``*_pseudo`` arms (velocity family only) instead
    keep the raw global drive but declare the circulation with the ``"0o"``
    pseudoscalar rank token and switch on the pseudo channel sector
    (:data:`_PSEUDO_SCHEMA`, :data:`_PSEUDO_ARM_KWARGS`) -- the
    typed-circulation fix for the 0.647 ``circulation_ood`` floor.  The
    ``*_h1`` suffix (velocity family, ``singpair_pseudo`` dictionary only)
    additionally applies the one-head capacity trade
    (:data:`_ONE_HEAD_CONFIG`) at fixed total score capacity -- the
    iteration-32 cross-suite confirmation arm of the iteration-31 one-head
    finding; everything else about the arm is bitwise ``singpair_pseudo``.

    ``bounded_gates`` (default ``False``: bitwise-identical arms) switches
    on ``bounded_output_gate_invariants`` on the transformer arms -- the
    far-field gate-collapse fix (compactified query invariants feeding the
    output projection's sigmoid gates; see ``GeometryConditionedLinear``).
    Measured alone it fired its falsifier: the gate collapse had been
    suppressing polynomially growing direct-drive branches, and
    ``farfield_queries`` went 0.694 -> 2.98.  ``bounded_query`` (default
    ``False``: bitwise-identical arms) is the source-side completion --
    ``MeshTransformer(bounded_query_geometry=True)`` injects the
    compactified query position x/sqrt(1+|x|^2) into the query operator
    state, bounding every learned query-radius dependence (gates AND
    direct-drive geometry vectors) at once, while the kernel dictionary's
    exact members keep the raw coordinates.  ``decaying_drive`` and
    ``monopole_free_sl`` (both default ``False``: bitwise-identical arms)
    are iteration 30's decay structure -- bounding alone plateaued
    ``farfield_queries`` at 0.452 because bounded is not decaying.
    ``decaying_drive`` switches on
    ``MeshTransformer(decaying_direct_drive=True)``: the query-side
    direct-drive contribution is multiplied by the fixed analytic envelope
    1/(1+|x|^2) of the raw query radius (the exterior-expansion leading
    order of the zero-net-flux disturbance velocity), so the direct term
    decays at the physical rate while the exact members stay the sole
    carriers of the slower (circulation r^-1) tails.  ``monopole_free_sl``
    switches on ``kernel_monopole_free_single_layer=True``: the exact
    single-layer member is deflated to zero net charge, structurally
    killing the log-r monopole tail that iteration 29 measured being fit
    by near-cancellation (licensed here because the disturbance field of a
    closed body with no net flux has no monopole; NOT licensed for
    problems with genuine net flux, e.g. screened or source-driven ones).
    All four knobs apply to any transformer arm (``monopole_free_sl`` to
    the singpair dictionaries carrying the single-layer member); the
    baseline arms ignore radius by construction and reject the flags.
    """

    if family not in FAMILY_MODEL_NAMES:
        raise ValueError(f"unknown family {family!r}")
    if model_name not in FAMILY_MODEL_NAMES[family]:
        raise ValueError(
            f"model {model_name!r} is not registered for family {family!r}"
        )
    if model_name in ("boundary_mean", "pair_kernel"):
        flags = {
            "bounded_gates": bounded_gates,
            "bounded_query": bounded_query,
            "decaying_drive": decaying_drive,
            "monopole_free_sl": monopole_free_sl,
        }
        if any(flags.values()):
            flag = next(name for name, value in flags.items() if value)
            raise ValueError(
                f"{flag} applies to transformer arms only, not {model_name!r}"
            )
        return BoundaryMean() if model_name == "boundary_mean" else InvariantPairKernel()
    one_head = model_name.endswith("_h1")
    stem = model_name.removesuffix("_h1")
    trace_driven = stem.endswith("_trace")
    pseudo_typed = stem.endswith("_pseudo")
    base_name = stem.removesuffix("_trace").removesuffix("_pseudo")
    if trace_driven:
        schema = _TRACE_SCHEMA
    elif pseudo_typed:
        schema = _PSEUDO_SCHEMA
    else:
        schema = _FAMILY_SCHEMAS[family]
    kernel_kwargs = dict(_KERNEL_ARM_KWARGS[base_name])
    if pseudo_typed:
        # The typed-circulation arms declare Gamma with the "0o" rank token,
        # which requires the pseudo channel sector to be switched on.
        kernel_kwargs.update(_PSEUDO_ARM_KWARGS)
    if bounded_gates:
        # Added only when on so the default arm's construction call is
        # byte-for-byte the historical one.
        kernel_kwargs["bounded_output_gate_invariants"] = True
    if bounded_query:
        kernel_kwargs["bounded_query_geometry"] = True
    if decaying_drive:
        kernel_kwargs["decaying_direct_drive"] = True
    if monopole_free_sl:
        # The decoder itself rejects arms lacking the single-layer member
        # (there is no monopole to control without it).
        kernel_kwargs["kernel_monopole_free_single_layer"] = True
    config = replace(MeshTransformerConfig(), **_ONE_HEAD_CONFIG) if one_head else None
    return _build_transformer_arm(family, schema, kernel_kwargs, config=config)


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    *,
    family: str,
    eval_seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Mean relative L2 per split on frozen, deterministic evaluation banks.

    For the velocity family the relative L2 is taken over the full vector
    field (Frobenius norm across points and components).  For the
    boundary-layer family every split additionally reports
    ``<split>/near_wall`` and ``<split>/far_field`` bucket errors -- the
    multi-scale diagnostic (bucket-restricted relative L2, averaged over
    cases).
    """

    build = SAMPLE_BUILDERS[family]
    target_key = _TARGET_KEYS[family]
    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS[family].items())):
        errors: dict[str, list[float]] = {"": []}
        for case in range(n_cases):
            sample = build(
                eval_seed + 7919 * case + 1_000_003 * split_index,
                device=device,
                dtype=dtype,
                **spec,
            )
            prediction = model(sample.domain).point_data[target_key]
            errors[""].append(_relative_l2(prediction, sample.target))
            if sample.near_wall_mask is not None:
                for bucket, mask in (
                    ("/near_wall", sample.near_wall_mask),
                    ("/far_field", ~sample.near_wall_mask),
                ):
                    errors.setdefault(bucket, []).append(
                        _relative_l2(prediction[mask], sample.target[mask])
                    )
        for suffix, values in errors.items():
            report[name + suffix] = sum(values) / len(values)
    return report


def pde_residual(
    model: nn.Module,
    *,
    family: str,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    r"""Return ``||lap u_pred|| * L^2 / ||u_pred||`` via autograd (float64).

    Meaningful for the two harmonic exterior families: the streamfunction
    target of ``potential_flow`` is harmonic, and the disturbance-velocity
    target of ``potential_flow_velocity`` is harmonic componentwise
    (:math:`dW/dz` is holomorphic), so the residual sums the componentwise
    Laplacian energy over the output's trailing components.  It diagnoses
    *harmonicity* of the prediction, not accuracy (the exact solution and any
    harmonic model score float-noise zero).  The boundary-layer target is
    deliberately non-harmonic, so the driver skips this diagnostic there.
    Float64, two cases of the requested split (default in-distribution, the
    historical convention), 32 interior queries.
    """

    if family not in ("potential_flow", "potential_flow_velocity"):
        raise ValueError(
            "pde_residual applies only to the harmonic potential_flow and "
            "potential_flow_velocity families"
        )
    build = SAMPLE_BUILDERS[family]
    target_key = _TARGET_KEYS[family]
    spec = SPLITS[family][split]
    model.eval()
    residuals = []
    for case in range(2):
        sample = build(
            seed + case, n_query=32, device=device, dtype=torch.float64, **spec
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        u = model_fp64(domain).point_data[target_key]
        components = u.reshape(u.shape[0], -1)
        laplacian = torch.zeros_like(components)
        if u.grad_fn is not None:
            for output in range(components.shape[1]):
                (gradient,) = torch.autograd.grad(
                    components[:, output].sum(),
                    points,
                    create_graph=True,
                    allow_unused=True,
                )
                if gradient is None:
                    continue
                for component in range(2):
                    (second,) = torch.autograd.grad(
                        gradient[:, component].sum(),
                        points,
                        create_graph=True,
                        allow_unused=True,
                    )
                    if second is not None:
                        laplacian[:, output] = (
                            laplacian[:, output] + second[:, component]
                        )
        length = sample.domain.global_data["reference_length"].reshape(())
        residual = laplacian.detach() * length**2
        residuals.append(
            float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


def fidelity_metrics(
    model: nn.Module,
    *,
    family: str,
    seed: int,
    device: torch.device,
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split, the strong-form residual under the driver's existing
    convention (:func:`pde_residual`: float64 autograd, two cases, 32
    interior queries -- deliberately subsampled so the block stays cheap
    relative to training) for the two harmonic families; the manufactured
    boundary-layer target is non-harmonic, so its entry is ``None``.  No
    maximum-principle violation is licensed on these exterior problems: the
    drive is global (freestream and circulation) rather than a prescribed
    Dirichlet range, the circulation term is unbounded at infinity, and the
    velocity/pressure targets are not scalar harmonic traces.
    """

    harmonic_family = family in ("potential_flow", "potential_flow_velocity")
    return {
        "pde_residual": (
            {
                name: pde_residual(
                    model, family=family, seed=seed, device=device, split=name
                )
                for name in sorted(SPLITS[family])
            }
            if harmonic_family
            else None
        ),
        "pde_residual_note": (
            "||lap u|| L^2 / ||u|| (componentwise for the velocity family) "
            "via float64 autograd at 32 interior points on two cases per "
            "split; harmonicity of the prediction, not accuracy -- the exact "
            "solution and any harmonic model score ~0; None for the "
            "deliberately non-harmonic boundary-layer family"
        ),
        "max_principle_violation": None,
        "max_principle_note": (
            "n/a: exterior problems driven by global data (freestream, "
            "circulation) supply no prescribed Dirichlet range to bound the "
            "interior, and the circulation term is unbounded at infinity"
        ),
    }


_EQUATIONS = {
    "potential_flow": (
        "laplace (exterior): potential flow past a body; target = "
        "nondimensional disturbance streamfunction (2d pseudoscalar)"
    ),
    "potential_flow_velocity": (
        "laplace (exterior): potential flow past a body; target = "
        "nondimensional disturbance velocity (polar vector, exact dW/dz "
        "push-forward)"
    ),
    "boundary_layer": (
        "manufactured multi-scale: potential flow + f(s) exp(-d/delta) wall "
        "layer (exact by construction; not a navier-stokes solution)"
    ),
}


def widened_training_spec(family: str, train_query_outer: float) -> dict:
    """Return the in-distribution training spec with a widened query annulus.

    This is the M1 (annulus-coverage) discriminator knob of the far-field
    strong-inference study: it changes the radial range of the TRAINING
    queries only -- every evaluation split in :data:`SPLITS` (including
    ``farfield_queries``) is deliberately untouched, so a retrained arm is
    scored against the identical frozen banks as the default arm.  For the
    exterior-flow families the training annulus becomes ``(inner,
    train_query_outer)`` in canonical preimage radius; for the boundary-layer
    family the analogous ``outer_query_radius`` is overridden.
    """

    if family not in SPLITS:
        raise ValueError(f"unknown family {family!r}")
    outer = float(train_query_outer)
    spec = dict(SPLITS[family]["in_distribution"])
    if family == "boundary_layer":
        if not outer > 1.0:
            raise ValueError("train_query_outer must exceed the unit body radius")
        spec["outer_query_radius"] = outer
    else:
        inner, _ = spec["query_radius_range"]
        if not outer > inner:
            raise ValueError(
                f"train_query_outer must exceed the inner query radius {inner}"
            )
        spec["query_radius_range"] = (inner, outer)
    return spec


def run_experiment(
    *,
    model_name: str,
    family: str,
    steps: int,
    seed: int,
    device: str,
    output_dir: str,
    eval_cases: int = 16,
    train_query_outer: float | None = None,
    save_checkpoint: bool = False,
    bounded_gates: bool = False,
    bounded_query: bool = False,
    decaying_drive: bool = False,
    monopole_free_sl: bool = False,
) -> dict:
    """Train one arm on the family's in-distribution split and report.

    ``train_query_outer`` (default ``None``: bitwise-identical training
    stream to the historical runs) widens the training-query annulus via
    :func:`widened_training_spec` without touching any evaluation split.
    ``save_checkpoint`` (default off) additionally writes the reported
    (best-validation) state dict next to the JSON report so analysis scripts
    can reload the trained arm.  ``bounded_gates`` (default off:
    bitwise-identical arms) is the far-field gate-collapse fix,
    ``bounded_query`` the source-side compactified-query-position completion
    of it, and ``decaying_drive``/``monopole_free_sl`` iteration 30's
    far-field decay structure -- the analytic 1/(1+|x|^2) direct-drive
    envelope and the zero-net-charge single-layer deflation
    (:func:`_build_model`); all are recorded in the report and checkpoint so
    analysis scripts rebuild the matching architecture (none of the knobs
    adds parameters, so the state dicts are interchangeable and the flags
    alone select the parameterization).
    """

    if family not in SPLITS:
        raise ValueError(f"unknown family {family!r}")
    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    build = SAMPLE_BUILDERS[family]
    target_key = _TARGET_KEYS[family]
    train_spec = (
        SPLITS[family]["in_distribution"]
        if train_query_outer is None
        else widened_training_spec(family, train_query_outer)
    )
    model = _build_model(
        model_name,
        family,
        bounded_gates=bounded_gates,
        bounded_query=bounded_query,
        decaying_drive=decaying_drive,
        monopole_free_sl=monopole_free_sl,
    ).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]

    best_state, best_val, history = None, float("inf"), []
    start_time = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-6)
        for step in range(1, steps + 1):
            model.train()
            sample = build(
                seed + 104_729 * step, device=device_t, dtype=dtype, **train_spec
            )
            prediction = model(sample.domain).point_data[target_key]
            loss = torch.sum((prediction - sample.target).square()) / torch.sum(
                sample.target.square()
            ).clamp_min(1.0e-30)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 250 == 0 or step == steps:
                validation = evaluate_splits(
                    model,
                    family=family,
                    eval_seed=71_000_011,
                    n_cases=4,
                    device=device_t,
                    dtype=dtype,
                )["in_distribution"]
                history.append({"step": step, "validation_relative_l2": validation})
                if validation < best_val:
                    best_val = validation
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_val = evaluate_splits(
            model,
            family=family,
            eval_seed=71_000_011,
            n_cases=4,
            device=device_t,
            dtype=dtype,
        )["in_distribution"]

    report = {
        "model": model_name,
        "family": family,
        "equation": _EQUATIONS[family],
        "seed": seed,
        "steps": steps,
        "train_query_outer": train_query_outer,
        "bounded_gates": bounded_gates,
        "bounded_query": bounded_query,
        "decaying_drive": decaying_drive,
        "monopole_free_sl": monopole_free_sl,
        "train_spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in train_spec.items()
        },
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_validation_relative_l2": best_val,
        "splits": evaluate_splits(
            model,
            family=family,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            device=device_t,
            dtype=dtype,
        ),
        "pde_residual": (
            pde_residual(model, family=family, seed=83_000_019, device=device_t)
            if family == "potential_flow"
            else None
        ),
        "pde_residual_scale_note": (
            "||lap u|| L^2 / ||u||: harmonicity of the prediction, not "
            "accuracy; reported only for potential_flow (the boundary-layer "
            "target is non-harmonic by construction)"
        ),
        "fidelity": fidelity_metrics(
            model, family=family, seed=83_000_019, device=device_t
        ),
        "state": {
            k: v.tolist() for k, v in model.state_dict().items() if v.numel() <= 16
        },
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_name = None
    if save_checkpoint:
        checkpoint_name = f"{family}_{model_name}_seed{seed}.pt"
        torch.save(
            {
                "model": model_name,
                "family": family,
                "seed": seed,
                "steps": steps,
                "train_query_outer": train_query_outer,
                "bounded_gates": bounded_gates,
                "bounded_query": bounded_query,
                "decaying_drive": decaying_drive,
                "monopole_free_sl": monopole_free_sl,
                "state_dict": model.state_dict(),
            },
            out / checkpoint_name,
        )
    report["checkpoint"] = checkpoint_name
    (out / f"{family}_{model_name}_seed{seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--family",
        required=True,
        choices=("potential_flow", "potential_flow_velocity", "boundary_layer"),
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-cases", type=int, default=16)
    parser.add_argument(
        "--train-query-outer",
        type=float,
        default=None,
        help=(
            "widen the TRAINING query annulus's outer radius (canonical "
            "preimage units); evaluation splits are untouched -- the M1 "
            "annulus-coverage discriminator knob"
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="write the best-validation state dict next to the JSON report",
    )
    parser.add_argument(
        "--bounded-gates",
        action="store_true",
        help=(
            "compactify the query invariants feeding the output projection's "
            "sigmoid gates (bounded_output_gate_invariants) -- the far-field "
            "gate-collapse fix; default off is bitwise-identical arms"
        ),
    )
    parser.add_argument(
        "--bounded-query",
        action="store_true",
        help=(
            "inject the compactified query position x/sqrt(1+|x|^2) into the "
            "query operator state (bounded_query_geometry) -- the source-side "
            "completion of the far-field fix; the kernel dictionary's exact "
            "members keep the raw coordinates; default off is "
            "bitwise-identical arms"
        ),
    )
    parser.add_argument(
        "--decaying-drive",
        action="store_true",
        help=(
            "multiply the query-side direct-drive contribution by the fixed "
            "analytic envelope 1/(1+|x|^2) of the raw query radius "
            "(decaying_direct_drive) -- the exterior-expansion leading order "
            "of the zero-net-flux disturbance velocity; iteration 30's decay "
            "structure (bounded is not decaying); default off is "
            "bitwise-identical arms"
        ),
    )
    parser.add_argument(
        "--monopole-free-sl",
        action="store_true",
        help=(
            "deflate the exact single-layer member to zero net charge "
            "(kernel_monopole_free_single_layer), structurally killing its "
            "log-r monopole tail -- licensed for zero-net-flux exterior "
            "disturbance fields only; requires a singpair arm; default off "
            "is bitwise-identical arms"
        ),
    )
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        family=arguments.family,
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
        eval_cases=arguments.eval_cases,
        train_query_outer=arguments.train_query_outer,
        save_checkpoint=arguments.save_checkpoint,
        bounded_gates=arguments.bounded_gates,
        bounded_query=arguments.bounded_query,
        decaying_drive=arguments.decaying_drive,
        monopole_free_sl=arguments.monopole_free_sl,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("model", "family", "splits", "pde_residual")}
        )
    )
