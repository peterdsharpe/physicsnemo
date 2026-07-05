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

r"""Multi-field exterior benchmark: velocity **and** Bernoulli pressure.

This is the program's first MULTI-FIELD (multivariate-output) family: 2D
steady incompressible flow past a certified star body, predicting **both**
the rank-1 disturbance velocity and the rank-0 pressure coefficient at the
same exterior query points,

.. math::

   \text{targets} = \{\,u_d^*\ (\text{rank 1}),\ p^*\ (\text{rank 0})\,\}.

Everything geometric and kinematic is *imported* from
:mod:`potential_flow` -- the certified exterior conformal maps, the circle
theorem push-forward, the splits, and the certified velocity target of
Family A' (``potential_flow_velocity``).  Seed for seed, an
``euler_bernoulli`` sample describes bitwise the *same flow* as a Family A'
sample; this module adds only the second output field.

**The pressure label is exact by construction.**  For steady incompressible
potential flow, Bernoulli's principle along any streamline (all streamlines
share the far-field reference here) gives, in nondimensional form (density
1, far-field pressure reference, velocities scaled by :math:`|U|`),

.. math::

   p^* = \tfrac12\big(|U^*|^2 - |u^*_{\rm total}|^2\big),
   \qquad u^*_{\rm total} = U^* + u_d^*,\quad |U^*| = 1,

with :math:`u_d^*` the exact disturbance velocity already computed by the
potential-flow machinery (the :math:`dW/dz` push-forward through the
certified exterior map, circulation included).  No numerical solver enters:
the multi-field target is closed-form, and the label chain is certified by
recomputing :math:`u_d^*` from independently Newton-inverted preimages and
verifying the Bernoulli identity at :math:`10^{-12}` in float64.

**Why this family exists: the pressure is drive-QUADRATIC.**  Expanding,

.. math::

   p^* = -\,U^* \cdot u_d^* \;-\; \tfrac12 |u_d^*|^2 ,

and :math:`u_d^*` is jointly linear in the drive :math:`(U, \Gamma^*)`, so
:math:`p^*` is an exactly *even* (quadratic) function of the drive while the
velocity is exactly *odd* (linear).  This is the first benchmark whose
target is intrinsically drive-nonlinear while the geometry family stays
certified-exact -- the natural bridge toward Euler/RANS: multi-field output
plus nonlinear drive coupling, before any CFD solver enters.

THE PRE-REGISTERED LINEAR WALL (logged before the first ``euler_bernoulli``
training run).  A ``field_mode="linear"`` arm's drive-to-output map is
exactly linear, hence exactly odd: :math:`N[-d] = -N[d]`.  The pressure
target is exactly even in the drive, and the training drive distribution is
symmetric under negation (uniform far-field angle, sign-symmetric
circulation), so the L2-optimal odd fit of the even pressure target is the
zero function -- the linear control's pressure error must sit pinned at
relative L2 :math:`\approx 1.0` (a trained no-response, exactly like the
Liouville superposition wall and the Family A pseudoscalar wall), while its
velocity error should match the nonlinear arm's (the velocity is drive-odd
and hence fully representable in linear mode).  The
``zero_preserving_nonlinear`` arms can form drive-quadratic invariants
(drive-scalar products and drive-vector Gram invariants in the blocks and
kernel decoder), so the pressure is representable there and its error should
drop with training.  ``mt_singpair_linear`` is registered purely to MEASURE
that wall.

ARMS.  All transformer arms use the pruned two-member "singpair" kernel
dictionary (exact double layer + exact single layer, no smooth members) and
declare ``output_field_ranks={"velocity": 1, "pressure": 0}`` -- the first
multi-field output, unpacked by the model's ``FieldLayout`` into named
point fields:

- ``mt_singpair_nl`` -- ``zero_preserving_nonlinear``: the historical
  primary arm; before iteration 35 it was the only mode in which the
  drive-quadratic pressure was representable at all.
- ``mt_singpair_linear`` -- the pre-registered wall control (see above).
- ``mt_singpair_nl_pseudo`` -- nonlinear plus the typed-circulation
  extension of iteration 23: circulation declared with the ``"0o"``
  pseudoscalar rank token and ``drive_pseudo_dim=8``, so the axial
  circulation velocity :math:`\Gamma\,x^\perp/(2\pi|x|^2)` is representable
  (measured on Family A' as the ``circulation_ood`` split leaving the 0.647
  floor).  Parity machinery should matter for the *velocity*; the pressure
  is parity-even and needs only the nonlinear sector.
- ``mt_singpair_q2`` -- ``field_mode="quadratic"`` (iteration 35, the fix
  rung of the nonlinear-fragility ladder): the drive degree is DECLARED at
  exactly the targets' degrees (velocity 1, pressure 2) instead of being
  left implicit -- iteration 34's amplitude probe measured effective drive
  degree ~21 in the nonlinear arms, the engine of the circulation-OOD
  blowup.  Pre-registered (before the q2 retraining): circulation-OOD must
  fall below the nonlinear arm's 0.35 toward the ID level (bar < 0.15),
  in-distribution within 2x seed sd of the nonlinear arms, and the
  drive-scaling structural degree test must pass at machine precision.
- ``mt_singpair_q2_pseudo`` -- the quadratic mode composed with the typed
  circulation (declared degree AND declared parity), registered because the
  composition is one schema swap; not part of the pre-registered iteration
  35 run matrix.

CONTROLS.  ``boundary_mean`` generalizes the scalar bank's parameter-free
constant floor per field: the boundary drive of this family is empty
(impermeability is homogeneous), so the boundary-measure mean of the
declared per-field drive is the far-field reference constant -- the zero
disturbance velocity and the zero Bernoulli pressure (also the unique
parameter-free O(2)-equivariant constants).  Its per-field relative L2 is
exactly 1.0 by construction: the no-response floor every trained arm must
beat.  A trace-informed ``pair_kernel`` (scalarized trace per field) is
deliberately NOT registered: the Family A scalarization is certified
"equivalent data, no solve" because the disturbance-streamfunction trace is
the negated uniform-stream streamfunction at physical body points, but the
on-body *pressure* trace requires the surface speed
:math:`|W'(\zeta)|/|G'(\zeta)|` -- the solution itself -- so a per-field
pressure trace would leak the answer and certify nothing.

SPLITS AND METRICS.  The five exterior-flow splits are reused verbatim
(``in_distribution``, ``unseen_geometry_modes``, ``wilder_shapes``,
``circulation_ood``, ``farfield_queries``).  Every split reports per-field
relative L2 (``<split>/velocity``, ``<split>/pressure``) plus the combined
norm (``<split>``: relative L2 over the concatenation of both fields).  The
training loss is the equal-weight mean of per-field relative squared errors,
so the smaller-norm pressure field cannot be silently ignored.

Certification (tests): the Bernoulli identity against independently
recomputed :math:`|u|` at the sample's query points (:math:`10^{-12}`,
float64); the velocity certification reused bitwise from the potential-flow
machinery (same-seed target equality with Family A'); an on-body
impermeability check through the velocity itself (the exact-map tangent
:math:`t \propto i\zeta e^{i\alpha}G'(\zeta)` satisfies
:math:`\operatorname{Im}[\bar t\,u_{\rm total}] = 0` on :math:`|\zeta|=1`);
and the exact drive-parity identities (velocity odd, pressure even) that
underwrite the pre-registered wall.  This is a benchmark-local research
prototype, not a proposed public API.
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
from potential_flow import (
    _FAMILY_SCHEMAS,
    _KERNEL_ARM_KWARGS,
    _PSEUDO_ARM_KWARGS,
    _PSEUDO_SCHEMA,
    ExteriorBody,
    PotentialFlowSample,
    build_potential_flow_velocity_sample,
    disturbance_complex_velocity,
    disturbance_velocity,
    exterior_map_derivative,
    velocity_at_physical,
)
from potential_flow import (
    SPLITS as _POTENTIAL_FLOW_SPLITS,
)
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

FAMILY = "euler_bernoulli"

# The multi-field output declaration: one polar vector, one true scalar.
OUTPUT_FIELD_RANKS: dict[str, int] = {"velocity": 1, "pressure": 0}

# The exterior-flow splits, reused verbatim (same flows, same OOD axes).
SPLITS: dict[str, dict] = {
    name: dict(spec) for name, spec in _POTENTIAL_FLOW_SPLITS["potential_flow"].items()
}

# Drive schemas, imported from the potential-flow registry: the untyped
# global-drive schema (far-field vector + true-scalar circulation) and the
# typed-circulation ("0o") schema of the iteration-23 pseudo arms.
_UNTYPED_SCHEMA = _FAMILY_SCHEMAS["potential_flow_velocity"]

# The pruned two-member kernel dictionary shared by every arm here.
_SINGPAIR_KWARGS = _KERNEL_ARM_KWARGS["mesh_transformer_kernel_singpair"]


# ---------------------------------------------------------------------------
# Exact Bernoulli pressure
# ---------------------------------------------------------------------------


def bernoulli_pressure(
    freestream: torch.Tensor, disturbance: torch.Tensor
) -> torch.Tensor:
    r"""Nondimensional Bernoulli pressure from the disturbance velocity.

    .. math::

       p^* = \tfrac12\big(|U^*|^2 - |U^* + u_d^*|^2\big)
           = \tfrac12\big(|U|^2 - |u_{\rm total}|^2\big)\big/|U|^2 ,

    with ``freestream`` the physical unit far-field vector ``(2,)`` and
    ``disturbance`` the nondimensional disturbance velocity ``(n, 2)``.
    Density 1, far-field pressure reference; exactly even (quadratic) in the
    joint drive :math:`(U, \Gamma^*)` because :math:`u_d^*` is exactly odd.
    """

    if freestream.shape != (2,):
        raise ValueError("freestream must have shape (2,)")
    if disturbance.ndim != 2 or disturbance.shape[-1] != 2:
        raise ValueError("disturbance must have shape (n, 2)")
    total = freestream[None, :].to(disturbance.dtype) + disturbance
    return 0.5 * (
        freestream.square().sum().to(disturbance.dtype) - total.square().sum(dim=-1)
    )


def pressure_at_physical(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    freestream: torch.Tensor,
    circulation: float,
    points: torch.Tensor,
) -> torch.Tensor:
    """Bernoulli pressure at physical ``(n, 2)`` points, independently.

    Recomputes the disturbance velocity from independently Newton-inverted
    preimages (:func:`potential_flow.velocity_at_physical`) and applies the
    Bernoulli identity -- the certification path for the pressure label.
    """

    return bernoulli_pressure(
        freestream,
        velocity_at_physical(body, canonical_freestream, circulation, points),
    )


def body_tangency_residual(
    body: ExteriorBody,
    canonical_freestream: torch.Tensor,
    circulation: float,
    z: torch.Tensor,
) -> torch.Tensor:
    r"""On-body impermeability residual through the velocity itself.

    At :math:`\zeta = e^{i\theta}` the physical body tangent is
    :math:`t(\zeta) \propto i\,\zeta\,e^{i\alpha}G'(\zeta)` (the
    :math:`\theta`-derivative of the physical curve; :math:`|t| = |G'| \ge
    1-\kappa > 0` by the certificate).  Impermeability of the exact flow is
    :math:`\operatorname{Im}[\bar t\,u_{\rm total}] = 0` exactly on the body
    -- the velocity-level statement of the body being a streamline.  Returns
    the normalized residual :math:`\operatorname{Im}[\bar t\,u_{\rm
    total}]/|t|` at the given on-circle preimages.
    """

    freestream_complex = body.rotation_factor * canonical_freestream
    total = freestream_complex + disturbance_complex_velocity(
        body, canonical_freestream, circulation, z
    )
    tangent = 1j * z * body.rotation_factor * exterior_map_derivative(body, z)
    return (torch.conj(tangent) * total).imag / tangent.abs()


# ---------------------------------------------------------------------------
# Sample assembly (delegates to the certified Family A' builder)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EulerBernoulliSample:
    """One exact multi-field exterior-flow problem.

    ``flow`` is the underlying certified Family A' sample (same seed, same
    flow, velocity target, and full certification payload); ``targets`` maps
    each declared output field to its exact label on the shared queries.
    """

    flow: PotentialFlowSample
    targets: dict[str, torch.Tensor]

    @property
    def domain(self) -> DomainMesh:
        return self.flow.domain

    @property
    def body(self) -> ExteriorBody:
        return self.flow.body

    @property
    def freestream(self) -> torch.Tensor:
        return self.flow.freestream

    @property
    def canonical_freestream(self) -> torch.Tensor:
        return self.flow.canonical_freestream

    @property
    def circulation(self) -> float:
        return self.flow.circulation

    @property
    def query_preimages(self) -> torch.Tensor:
        return self.flow.query_preimages


def build_euler_bernoulli_sample(
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
) -> EulerBernoulliSample:
    """Build one exact multi-field exterior-flow problem.

    Delegates the flow, geometry, certification, and velocity target to
    :func:`potential_flow.build_potential_flow_velocity_sample` (bitwise the
    same flow at the same seed) and adds the exact Bernoulli pressure label,
    computed in float64 before the final device/dtype cast.
    """

    flow = build_potential_flow_velocity_sample(
        seed,
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
    disturbance = disturbance_velocity(
        flow.body, flow.canonical_freestream, flow.circulation, flow.query_preimages
    )
    pressure = bernoulli_pressure(flow.freestream, disturbance)
    if not bool(torch.isfinite(pressure).all()):
        raise RuntimeError("the exact pressure target contains non-finite values")
    return EulerBernoulliSample(
        flow=flow,
        targets={
            "velocity": flow.target,
            "pressure": pressure.to(device=device, dtype=dtype),
        },
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class FarFieldBoundaryMean(nn.Module):
    """Parameter-free per-field constant floor (the multi-field boundary mean).

    The scalar bank's ``BoundaryMean`` predicts the boundary-measure mean of
    the per-field boundary drive.  This family's boundary drive is empty
    (impermeability is homogeneous), so that mean is the far-field reference
    constant of each field: the zero disturbance velocity and the zero
    Bernoulli pressure -- also the unique parameter-free O(2)-equivariant
    constants (a nonzero constant vector would require a preferred
    direction).  Per-field relative L2 is exactly 1.0 by construction; this
    is the no-response floor the trained arms must beat.
    """

    def forward(self, domain: DomainMesh) -> Mesh:
        points = domain.interior.points
        n = domain.interior.n_points
        return domain.interior.with_data(
            point_data={
                "velocity": points.new_zeros((n, 2)),
                "pressure": points.new_zeros((n,)),
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

#: Field mode and drive schema per transformer arm (see the module
#: docstring's ARMS section for what each arm measures).
_ARM_MODES: dict[str, tuple[str, str]] = {
    "mt_singpair_linear": ("linear", "untyped"),
    "mt_singpair_nl": ("zero_preserving_nonlinear", "untyped"),
    "mt_singpair_nl_pseudo": ("zero_preserving_nonlinear", "pseudo"),
    "mt_singpair_q2": ("quadratic", "untyped"),
    "mt_singpair_q2_pseudo": ("quadratic", "pseudo"),
}


def _build_model(model_name: str) -> nn.Module:
    """Instantiate one arm of the multi-field comparison.

    All transformer arms share the singpair kernel dictionary and the
    multi-field output declaration; they differ only in field mode and in
    the circulation's declared type (see the module docstring).
    """

    if model_name == "boundary_mean":
        return FarFieldBoundaryMean()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}")
    kernel_kwargs = dict(_SINGPAIR_KWARGS)
    field_mode, schema_name = _ARM_MODES[model_name]
    if schema_name == "pseudo":
        schema = _PSEUDO_SCHEMA
        kernel_kwargs.update(_PSEUDO_ARM_KWARGS)
    else:
        schema = _UNTYPED_SCHEMA
    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks=dict(OUTPUT_FIELD_RANKS),
        boundary_field_ranks=schema["boundary_field_ranks"],
        global_field_ranks=schema["global_field_ranks"],
        reference_length_key="reference_length",
        field_mode=field_mode,
        query_decoder="kernel",
        **kernel_kwargs,
        **asdict(MeshTransformerConfig()),
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
    """Equal-weight mean of per-field relative squared errors.

    Per-field normalization keeps the smaller-norm pressure field from being
    drowned by the velocity in the combined objective -- the wall must be
    measurable, not masked.
    """

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
    """Per-field and combined relative L2 per split, on frozen banks.

    Report keys: ``<split>/velocity`` and ``<split>/pressure`` (per-field
    relative L2, averaged over cases) and ``<split>`` (the combined
    concatenated-field norm).
    """

    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS.items())):
        errors: dict[str, list[float]] = {"": [], "/velocity": [], "/pressure": []}
        for case in range(n_cases):
            sample = build_euler_bernoulli_sample(
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


def pde_residual(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    r"""Return ``||lap u_d|| * L^2 / ||u_d||`` for the predicted velocity.

    The exact disturbance velocity is harmonic componentwise
    (:math:`\overline{dW/dz}` has holomorphic conjugate, so both Cartesian
    components are real harmonic), which is the strong form this multi-field
    problem licenses on its rank-1 output; the exact target scores
    float-noise zero.  Same convention and cost as the scalar drivers:
    float64 autograd, two cases of the requested split, 32 interior queries.
    The Bernoulli pressure is *not* harmonic; its licensed check is the
    algebraic :func:`bernoulli_consistency` residual instead.
    """

    spec = SPLITS[split]
    model.eval()
    residuals = []
    for case in range(2):
        sample = build_euler_bernoulli_sample(
            seed + case, n_query=32, device=device, dtype=torch.float64, **spec
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        velocity = model_fp64(domain).point_data["velocity"]
        laplacian = torch.zeros_like(velocity)
        if velocity.grad_fn is not None:
            for output in range(2):
                (gradient,) = torch.autograd.grad(
                    velocity[:, output].sum(),
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
                / torch.linalg.vector_norm(velocity.detach()).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


@torch.no_grad()
def bernoulli_consistency(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    r"""Cross-field Bernoulli residual of the *prediction*, per split.

    ``||p_pred - p_bernoulli(u_pred)|| / ||p_exact||`` where
    :math:`p_{\rm bernoulli}(u) = (|U|^2 - |U + u|^2)/2` is the exact
    algebraic strong form linking the two output fields (density 1,
    far-field reference).  The exact multi-field labels satisfy it to
    float64 roundoff by construction, so this measures whether the model's
    two heads describe one flow rather than two independent fits.
    Normalized by the exact pressure target's norm (the field's natural
    scale; the implied pressure of a degenerate zero-velocity prediction
    would otherwise have no scale).  Two cases per split, float64.
    """

    spec = SPLITS[split]
    model_fp64 = model.double()
    model_fp64.eval()
    residuals = []
    for case in range(2):
        sample = build_euler_bernoulli_sample(
            seed + case, device=device, dtype=torch.float64, **spec
        )
        predictions = _predictions(model_fp64, sample.domain)
        # The sample's certification-side freestream lives on the CPU in
        # float64; the prediction lives on the model device.
        implied = bernoulli_pressure(
            sample.freestream.to(device=predictions["velocity"].device),
            predictions["velocity"],
        )
        residuals.append(
            float(
                torch.linalg.vector_norm(predictions["pressure"] - implied)
                / torch.linalg.vector_norm(
                    sample.targets["pressure"]
                ).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


def fidelity_metrics(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split: the velocity-harmonicity strong-form residual
    (:func:`pde_residual`) and the cross-field Bernoulli consistency
    residual (:func:`bernoulli_consistency`), both deliberately subsampled
    (two cases per split; 32 queries for the autograd residual) so the block
    stays cheap relative to training.  No maximum-principle violation is
    licensed: the velocity target is a vector field and the Bernoulli
    pressure is not harmonic.
    """

    return {
        "pde_residual": {
            name: pde_residual(model, seed=seed, device=device, split=name)
            for name in sorted(SPLITS)
        },
        "pde_residual_note": (
            "velocity harmonicity ||lap u_d|| L^2 / ||u_d|| (componentwise, "
            "float64 autograd, 32 interior points, two cases per split); "
            "the exact disturbance velocity scores ~0.  The pressure field "
            "is checked through 'bernoulli_consistency' instead"
        ),
        "bernoulli_consistency": {
            name: bernoulli_consistency(model, seed=seed, device=device, split=name)
            for name in sorted(SPLITS)
        },
        "bernoulli_consistency_note": (
            "||p_pred - (|U|^2 - |U + u_pred|^2)/2|| / ||p_exact||: the "
            "algebraic Bernoulli identity applied to the model's own "
            "velocity, two cases per split; exact labels score ~0"
        ),
        "max_principle_violation": None,
        "max_principle_note": (
            "n/a: the velocity target is a polar vector field and the "
            "Bernoulli pressure is not harmonic, so no maximum principle is "
            "licensed"
        ),
    }


_EQUATION = (
    "euler/bernoulli (exterior, exact): steady incompressible potential flow "
    "past a body; targets = {disturbance velocity (rank 1), bernoulli "
    "pressure p* = (|U|^2 - |u_total|^2)/2 (rank 0, drive-quadratic)}"
)

_DESIGN_NOTES = {
    "pre_registered_prediction": (
        "logged before the first euler_bernoulli training run: the "
        "field_mode='linear' arm is exactly odd in the drive while the "
        "pressure target is exactly even, so its pressure relative L2 must "
        "stay pinned at ~1.0 (trained no-response); its velocity should "
        "match the nonlinear arm, whose pressure error should drop"
    ),
    "control_note": (
        "trace-informed pair_kernel deliberately not registered: a "
        "certified 'equivalent data, no solve' scalarized pressure trace "
        "does not exist -- the on-body pressure requires the surface speed "
        "|W'|/|G'|, i.e. the solution itself"
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
    the named steps -- e.g. ``(200,)`` captures the iteration-25 early-training
    regime in which the circulation-OOD blowup was observed.
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
            sample = build_euler_bernoulli_sample(
                seed + 104_729 * step, device=device_t, dtype=dtype, **train_spec
            )
            loss = _multi_field_loss(_predictions(model, sample.domain), sample.targets)
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
    parser.add_argument("--steps", type=int, default=1500)
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
