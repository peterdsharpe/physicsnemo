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

r"""Nonlinear elliptic testbed: the 2D Liouville equation on deformed disks.

The benchmark's first **nonlinear** homogeneous elliptic problem is

.. math::

   \Delta u + 2\,e^{u} = 0 \quad \text{on } \Omega \subset \mathbb{R}^2 ,

whose exact solutions are the constant-curvature (:math:`K=+1`) metric
potentials

.. math::

   u(w) = \log \frac{4\,|f'(w)|^2}{\left(1 + |f(w)|^2\right)^2}

for any holomorphic :math:`f` with :math:`f' \neq 0` on :math:`\Omega`
(the pullback of the spherical metric under :math:`f`).  The constant ``2``
convention was verified by autograd: :math:`\Delta u + c\,e^u` at random
interior points is machine zero (relative residual below ``1e-11`` in
float64) only for :math:`c = 2` among candidates :math:`\{1, 2, 4, 8\}`;
the others leave :math:`O(1)` residuals.  The same check is a unit test.

Why this benchmark exists: every previously studied model in this example is
structured around **linear** boundary-to-interior maps (mean lift plus a
kernel that is linear in the Dirichlet data; the harmonic-panel BIE is in
addition *harmonic by construction*).  The Liouville solution is not
harmonic -- :math:`\Delta u = -2e^u < 0` everywhere, so ``u`` is strictly
superharmonic and its interior values strictly exceed the harmonic extension
of its own trace.  Measuring exactly how the linear-PDE-structured baselines
fail here quantifies the "nonlinearity gap" that motivates the next
architecture iteration.  This is a benchmark plus baseline study, not a new
solver.

Sample construction (mirrors the linear benchmark's conventions):

- Geometry reuses :func:`conformal_laplace.sample_geometry` (deformed disks
  :math:`F(z) = z + \sum a_m z^m`, identity physical similarity, hence
  ``reference_length == 1``) with the same train/OOD geometry-mode splits.
- The holomorphic ``f`` is drawn from two families, both guaranteed to have
  a nonvanishing derivative on the domain: a Möbius transform of an affine
  map, ``f = t + mu / (a w + b - p)``, and a Möbius transform of an
  exponential, ``f = t + mu / (exp(alpha w + beta) - p)`` with
  ``|alpha| <= 1.5``.  The Möbius pole ``p`` is placed at distance greater
  than three from the inner map's image of the domain, so ``f`` is
  holomorphic and ``f' = -mu g' / (g - p)^2`` never vanishes.
- Coefficient magnitudes are balanced so ``u`` is :math:`O(1)`: the Möbius
  residue is scaled so ``|f'(0)|`` is a sampled :math:`O(1)` magnitude and
  the constant is chosen so ``|f(0)| <= 1``.  Samples with
  ``max |u| > 4`` over a probe set (boundary midpoints, interior queries,
  and extra interior probes) are rejected and redrawn deterministically.
- ``boundary_value`` is the exact ``u`` at panel *parameter* midpoints
  (Dirichlet data); interior targets are the exact ``u`` at the query
  points.  The :class:`~physicsnemo.mesh.DomainMesh` layout (single
  ``"dirichlet"`` boundary with ``boundary_value`` cell data,
  ``reference_length`` global data) matches the linear benchmark, so the
  baseline model classes are reused unmodified.

This is a benchmark-local research prototype, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import _paths  # noqa: F401
import torch
from conformal_laplace import (
    ConformalGeometry,
    HarmonicDrive,
    build_domain_sample,
    map_to_physical,
    points_to_complex,
    sample_disk_preimages,
    sample_geometry,
)
from models import BoundaryMean, InvariantPairKernel
from self_consistent_kernel import HarmonicPanelBIE
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

LIOUVILLE_CONSTANT = 2.0
"""Verified convention: ``lap u + 2 exp(u) = 0`` for the metric potential."""

_AMPLITUDE_CAP = 4.0
_POLE_CLEARANCE = 3.0
_FAMILIES = ("moebius_affine", "moebius_exp")


def _substream(seed: int, stream: int) -> int:
    """Derive independent deterministic seeds without mutable RNG state."""

    return seed + 15_485_863 * stream


@dataclass(frozen=True)
class LiouvilleField:
    r"""Coefficients of one holomorphic ``f`` with certified ``f' != 0``.

    The map is ``f(w) = constant + residue / (g(w) - pole)`` where the inner
    map ``g`` is ``linear_coefficient * w + offset`` for family
    ``"moebius_affine"`` and ``exp(linear_coefficient * w + offset)`` for
    family ``"moebius_exp"``.  All coefficient tensors are complex scalars.

    ``domain_radius_bound`` is an upper bound on ``|w|`` over the physical
    domain; construction validates that the Möbius pole keeps at least
    distance three from the inner map's image of that disk, which together
    with ``residue != 0`` and ``linear_coefficient != 0`` guarantees
    ``f' = -residue * g' / (g - pole)^2`` never vanishes on the domain.
    """

    family: str
    linear_coefficient: torch.Tensor
    offset: torch.Tensor
    pole: torch.Tensor
    residue: torch.Tensor
    constant: torch.Tensor
    domain_radius_bound: float

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got {self.family!r}")
        for name in ("linear_coefficient", "offset", "pole", "residue", "constant"):
            value = getattr(self, name)
            if not torch.is_complex(value) or value.ndim != 0:
                raise ValueError(f"{name} must be a complex scalar tensor")
            if (
                not torch.isfinite(value.real).item()
                or not torch.isfinite(value.imag).item()
            ):
                raise ValueError(f"{name} must be finite")
        if (
            not math.isfinite(self.domain_radius_bound)
            or self.domain_radius_bound <= 0.0
        ):
            raise ValueError("domain_radius_bound must be finite and positive")
        if self.linear_coefficient.abs().item() < 1.0e-6:
            raise ValueError(
                "linear_coefficient must be bounded away from zero (f' guard)"
            )
        if self.residue.abs().item() < 1.0e-9:
            raise ValueError("residue must be bounded away from zero (f' guard)")
        if self.pole.abs().item() < self.image_radius_bound + _POLE_CLEARANCE - 1.0e-9:
            raise ValueError(
                "the Moebius pole must keep distance at least "
                f"{_POLE_CLEARANCE} from the inner map's image of the domain"
            )

    @property
    def image_radius_bound(self) -> float:
        """Return an upper bound on ``|g(w)|`` over ``|w| <= radius bound``."""

        if self.family == "moebius_affine":
            return float(
                self.linear_coefficient.abs().item() * self.domain_radius_bound
                + self.offset.abs().item()
            )
        return float(
            math.exp(
                self.offset.real.item()
                + self.linear_coefficient.abs().item() * self.domain_radius_bound
            )
        )


def _inner_map(field: LiouvilleField, w: torch.Tensor) -> torch.Tensor:
    if field.family == "moebius_affine":
        return field.linear_coefficient * w + field.offset
    return torch.exp(field.linear_coefficient * w + field.offset)


def _inner_derivative(field: LiouvilleField, w: torch.Tensor) -> torch.Tensor:
    if field.family == "moebius_affine":
        return field.linear_coefficient.expand(w.shape).clone()
    return field.linear_coefficient * torch.exp(
        field.linear_coefficient * w + field.offset
    )


def field_value(field: LiouvilleField, w: torch.Tensor) -> torch.Tensor:
    r"""Evaluate ``f(w)`` at complex points ``w``."""

    return field.constant + field.residue / (_inner_map(field, w) - field.pole)


def field_derivative(field: LiouvilleField, w: torch.Tensor) -> torch.Tensor:
    r"""Evaluate ``f'(w)`` (nonvanishing on the certified domain)."""

    inner = _inner_map(field, w)
    return -field.residue * _inner_derivative(field, w) / (inner - field.pole) ** 2


def liouville_solution(field: LiouvilleField, w: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the exact potential ``log(4 |f'|^2 / (1 + |f|^2)^2)``.

    Composed from real logarithms of positive quantities for numerical
    stability and clean autograd; ``|f'|^2 > 0`` on the certified domain.
    """

    value = field_value(field, w)
    derivative = field_derivative(field, w)
    derivative_abs2 = derivative.real.square() + derivative.imag.square()
    value_abs2 = value.real.square() + value.imag.square()
    return math.log(4.0) + torch.log(derivative_abs2) - 2.0 * torch.log1p(value_abs2)


def liouville_pde_residual(
    field: LiouvilleField,
    points: torch.Tensor,
    *,
    constant: float = LIOUVILLE_CONSTANT,
) -> torch.Tensor:
    r"""Return ``lap u + constant * exp(u)`` at real points via autograd.

    For the exact solution and ``constant == 2`` this is machine zero; other
    constants leave :math:`O(e^u)` residuals, which is how the convention was
    verified.
    """

    if points.shape[-1:] != (2,):
        raise ValueError("points must have final dimension two")
    coordinates = points.detach().clone().requires_grad_(True)
    w = torch.complex(coordinates[..., 0], coordinates[..., 1])
    u = liouville_solution(field, w)
    (gradient,) = torch.autograd.grad(u.sum(), coordinates, create_graph=True)
    laplacian = torch.zeros_like(u)
    for component in range(2):
        (second,) = torch.autograd.grad(
            gradient[..., component].sum(), coordinates, create_graph=True
        )
        laplacian = laplacian + second[..., component]
    return laplacian + constant * torch.exp(u)


def sample_liouville_field(
    seed: int,
    *,
    domain_radius_bound: float,
) -> LiouvilleField:
    r"""Sample one certified field with magnitudes balanced so ``u`` is O(1).

    The family is drawn uniformly.  The residue is scaled so ``|f'(0)|`` is a
    log-uniform :math:`O(1)` magnitude and the constant places ``f(0)``
    inside the unit disk; both choices keep ``u(0)`` within roughly
    ``[-2.5, 2.5]`` before the generator's explicit amplitude rejection.
    Random numbers come from a CPU float64 generator, so a seed identifies
    one field independent of execution device.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(
            torch.empty((), dtype=torch.float64).uniform_(
                low, high, generator=generator
            )
        )

    def unit_phase() -> torch.Tensor:
        angle = torch.tensor(uniform(0.0, 2.0 * math.pi), dtype=torch.float64)
        return torch.polar(torch.ones((), dtype=torch.float64), angle)

    family = _FAMILIES[
        int(torch.randint(0, 2, (), generator=generator, dtype=torch.int64))
    ]
    if family == "moebius_affine":
        linear = unit_phase() * math.exp(uniform(math.log(0.5), math.log(1.5)))
        offset = unit_phase() * uniform(0.0, 0.8)
        inner_at_zero = offset
        inner_derivative_at_zero = linear
        image_bound = linear.abs().item() * domain_radius_bound + offset.abs().item()
    else:
        linear = unit_phase() * math.exp(uniform(math.log(0.3), math.log(1.5)))
        offset = torch.complex(
            torch.tensor(uniform(-0.5, 0.5), dtype=torch.float64),
            torch.tensor(uniform(0.0, 2.0 * math.pi), dtype=torch.float64),
        )
        inner_at_zero = torch.exp(offset)
        inner_derivative_at_zero = linear * inner_at_zero
        image_bound = math.exp(
            offset.real.item() + linear.abs().item() * domain_radius_bound
        )

    pole = unit_phase() * (image_bound + _POLE_CLEARANCE + uniform(0.1, 3.0))
    derivative_magnitude = math.exp(uniform(math.log(0.4), math.log(2.5)))
    # f'(0) = -residue * g'(0) / (g(0) - pole)^2; scale the residue so
    # |f'(0)| equals the sampled magnitude with a free phase.
    residue = (
        derivative_magnitude
        * unit_phase()
        * (inner_at_zero - pole) ** 2
        / inner_derivative_at_zero
    )
    value_at_zero = unit_phase() * uniform(0.0, 1.0)
    constant = value_at_zero - residue / (inner_at_zero - pole)
    return LiouvilleField(
        family=family,
        linear_coefficient=linear,
        offset=offset,
        pole=pole,
        residue=residue,
        constant=constant,
        domain_radius_bound=domain_radius_bound,
    )


def _zero_drive() -> HarmonicDrive:
    """Placeholder drive: mesh construction is reused, its values are not."""

    return HarmonicDrive(
        constant=torch.zeros((), dtype=torch.float64),
        modes=(),
        coefficients=torch.empty(0, dtype=torch.complex128),
    )


@dataclass(frozen=True)
class LiouvilleSample:
    """One exact Liouville Dirichlet problem plus its private generator state."""

    domain: DomainMesh
    target: torch.Tensor
    field: LiouvilleField
    geometry: ConformalGeometry


def build_liouville_sample(
    seed: int,
    *,
    geometry_modes: tuple[int, ...] = (2, 3),
    deformation_range: tuple[float, float] = (0.05, 0.35),
    n_boundary: int = 64,
    n_query: int = 128,
    n_extra_probes: int = 256,
    max_attempts: int = 64,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LiouvilleSample:
    r"""Build one exact Liouville problem on a deformed disk.

    Geometry and mesh conventions (panel winding, parameter-midpoint boundary
    sampling, area-uniform disk queries) are inherited from
    :func:`conformal_laplace.build_domain_sample` with the identity physical
    similarity, so ``reference_length == 1``.  All exact values are computed
    in float64 and cast to the requested device and dtype at the end.

    Fields whose ``max |u|`` over the probe set exceeds ``4`` are rejected
    and redrawn from a deterministic per-attempt seed stream, so a seed
    always identifies the same accepted sample.
    """

    geometry = sample_geometry(
        _substream(seed, 0),
        modes=geometry_modes,
        deformation_range=deformation_range,
        dtype=torch.float64,
    )
    base = build_domain_sample(
        geometry,
        _zero_drive(),
        n_boundary=n_boundary,
        n_query=n_query,
        query_seed=_substream(seed, 1),
    )
    # |F(z)| <= |z| + sum |a_m| <= 1 + sum |a_m| on the closed unit disk.
    radius_bound = float(1.0 + geometry.coefficients.abs().sum().item())

    midpoint_w = points_to_complex(
        map_to_physical(geometry, base.boundary_midpoint_preimages, base.similarity)
    )
    query_w = points_to_complex(base.domain.interior.points)
    probe_w = points_to_complex(
        map_to_physical(
            geometry,
            sample_disk_preimages(_substream(seed, 2), n_extra_probes),
            base.similarity,
        )
    )

    field: LiouvilleField | None = None
    boundary_values = target = None
    for attempt in range(max_attempts):
        candidate = sample_liouville_field(
            _substream(seed, 3 + attempt),
            domain_radius_bound=radius_bound,
        )
        boundary_values = liouville_solution(candidate, midpoint_w)
        target = liouville_solution(candidate, query_w)
        probes = liouville_solution(candidate, probe_w)
        amplitude = max(
            boundary_values.abs().max().item(),
            target.abs().max().item(),
            probes.abs().max().item(),
        )
        if amplitude <= _AMPLITUDE_CAP:
            field = candidate
            break
    if field is None:
        raise RuntimeError(
            f"no field with max |u| <= {_AMPLITUDE_CAP} found in "
            f"{max_attempts} attempts for seed {seed}"
        )

    base_boundary = base.domain.boundaries["dirichlet"]
    boundary = Mesh(
        points=base_boundary.points.to(device=device, dtype=dtype),
        cells=base_boundary.cells.to(device=device),
        cell_data={"boundary_value": boundary_values.to(device=device, dtype=dtype)},
    )
    interior = Mesh(
        points=base.domain.interior.points.to(device=device, dtype=dtype),
        point_data={"potential": target.to(device=device, dtype=dtype)},
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": base.similarity.scale.to(device=device, dtype=dtype)
        },
    )
    return LiouvilleSample(
        domain=domain,
        target=target.to(device=device, dtype=dtype),
        field=field,
        geometry=geometry,
    )


SPLITS: dict[str, dict] = {
    "in_distribution": {
        "geometry_modes": (2, 3),
        "deformation_range": (0.05, 0.35),
    },
    "unseen_geometry_modes": {
        "geometry_modes": (4, 5),
        "deformation_range": (0.05, 0.35),
    },
    "stronger_deformation": {
        "geometry_modes": (2, 3),
        "deformation_range": (0.45, 0.65),
    },
}


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _build_model(model_name: str) -> nn.Module:
    """Instantiate one linear-PDE-structured baseline (reused unmodified)."""

    if model_name == "pair_kernel":
        return InvariantPairKernel()
    if model_name == "harmonic_panel_bie":
        return HarmonicPanelBIE(regular_orders=0, shared_relaxation=True)
    if model_name == "boundary_mean":
        return BoundaryMean()
    if model_name.startswith("mesh_transformer_kernel_nl"):
        # Nonlinear-mode kernel decoder: the kernel may read drive invariants,
        # so the model is NOT drive-linear — the arm that can, in principle,
        # cross the superposition wall of the linear baselines (~0.79).
        # The _nomlp variant tests whether unstructured smooth members "earn
        # their place" on a nonlinear PDE (they were OOD-destructive and
        # ID-neutral on linear Laplace).  The _singpair variant is the
        # two-member exact singular dictionary (double layer + single layer):
        # it is the TWO-STREAM arm of the iteration-26 stream-separation
        # study, paired at matched parameters (171,449 vs 171,481) with the
        # single-stream control below.
        from models import MeshTransformerConfig, build_mesh_transformer

        return build_mesh_transformer(
            MeshTransformerConfig(),
            query_decoder="kernel",
            kernel_mlp_members=(
                0 if model_name.endswith(("_nomlp", "_singonly", "_singpair")) else 8
            ),
            kernel_include_polynomial_members=not model_name.endswith(
                ("_singonly", "_singpair")
            ),
            kernel_include_single_layer_member=model_name.endswith("_singpair"),
            field_mode="zero_preserving_nonlinear",
        )
    if model_name == "mesh_transformer_single_stream_nl":
        # MEASUREMENT-ONLY control (never a production mode): boundary values
        # fused into the nonlinear operator stream, constant unit drive; the
        # single-stream arm of the stream-separation study, paired with
        # "mesh_transformer_kernel_nl_singpair".  On this nonlinear PDE the
        # two-stream zero-drive contract is a *misspecification* (the exact
        # zero-Dirichlet Liouville solution is nonzero), so this is the
        # problem where single-stream fusion might legitimately win.  See
        # models.SingleStreamFusionControl for the pre-registered hypotheses.
        from models import MeshTransformerConfig, build_single_stream_control

        return build_single_stream_control(
            MeshTransformerConfig(),
            field_mode="zero_preserving_nonlinear",
        )
    if model_name == "mesh_transformer_kernel_linear":
        from models import MeshTransformerConfig, build_mesh_transformer

        return build_mesh_transformer(MeshTransformerConfig(), query_decoder="kernel")
    raise ValueError(f"unknown model {model_name!r}")


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    *,
    eval_seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Mean relative L2 per split on frozen, deterministic evaluation banks."""

    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS.items())):
        errors = []
        for case in range(n_cases):
            sample = build_liouville_sample(
                eval_seed + 7919 * case + 1_000_003 * split_index,
                geometry_modes=spec["geometry_modes"],
                deformation_range=spec["deformation_range"],
                device=device,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        report[name] = sum(errors) / len(errors)
    return report


@torch.no_grad()
def zero_drive_response(
    model: nn.Module,
    *,
    seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    r"""RMS interior response to identically zero Dirichlet data.

    First-class number for the stream-separation study.  Zero-preserving
    two-stream models return exactly zero by construction; the single-stream
    control does not.  Interpretation is PDE-specific: for *this* nonlinear
    equation the exact zero-Dirichlet solution is nonzero (on the unit disk
    it is ``u = log 4 - 2 log(1 + |w|**2)``, so ``u(0) = log 4 ~ 1.386`` and
    the area-mean is ``2 - 2 log 2 ~ 0.614``), so a structural zero here is a
    misspecification, not a virtue.  Unnormalized RMS is reported because the
    reference solution scale is O(1) by the generator's amplitude cap.
    """

    model.eval()
    spec = SPLITS["in_distribution"]
    responses = []
    for case in range(n_cases):
        sample = build_liouville_sample(
            seed + 7919 * case,
            geometry_modes=spec["geometry_modes"],
            deformation_range=spec["deformation_range"],
            device=device,
            dtype=dtype,
        )
        boundary = sample.domain.boundaries["dirichlet"]
        zero_domain = DomainMesh(
            interior=sample.domain.interior.with_data(
                point_data={}, cell_data={}, global_data={}
            ),
            boundaries={
                "dirichlet": boundary.with_data(
                    cell_data={
                        "boundary_value": torch.zeros_like(
                            boundary.cell_data["boundary_value"]
                        )
                    }
                )
            },
            global_data=sample.domain.global_data,
        )
        prediction = model(zero_domain).point_data["potential"]
        responses.append(float(prediction.square().mean().sqrt()))
    return sum(responses) / len(responses)


def pde_residual(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
    split: str = "in_distribution",
) -> float:
    r"""Return ``||lap u + 2 exp(u)|| / ||exp(u)||`` on predictions, autograd.

    Computed in float64 at 32 interior points on two cases of the requested
    split (default in-distribution, the historical convention), with ``u``
    the model's prediction (``reference_length == 1``, so no explicit
    nondimensionalization factor remains).  Scale calibration: the exact
    solution scores ``0`` and any harmonic prediction (``lap u == 0``) scores
    exactly ``2``.
    """

    model.eval()
    residuals = []
    spec = SPLITS[split]
    for case in range(2):
        sample = build_liouville_sample(
            seed + case,
            geometry_modes=spec["geometry_modes"],
            deformation_range=spec["deformation_range"],
            n_query=32,
            device=device,
            dtype=torch.float64,
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        u = model_fp64(domain).point_data["potential"]
        # Constant predictors (BoundaryMean) never touch the interior points,
        # so their prediction carries no graph at all; their laplacian is
        # identically zero.  allow_unused covers partial dependence.
        laplacian = torch.zeros_like(u)
        if u.grad_fn is not None:
            (gradient,) = torch.autograd.grad(
                u.sum(), points, create_graph=True, allow_unused=True
            )
            if gradient is not None:
                for component in range(2):
                    (second,) = torch.autograd.grad(
                        gradient[:, component].sum(),
                        points,
                        create_graph=True,
                        allow_unused=True,
                    )
                    if second is not None:
                        laplacian = laplacian + second[:, component]
        source = torch.exp(u.detach())
        residual = laplacian.detach() + LIOUVILLE_CONSTANT * source
        residuals.append(
            float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(source).clamp_min(1.0e-30)
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

    Per split, the strong-form residual under the driver's existing
    convention (:func:`pde_residual`: float64 autograd, two cases, 32
    interior queries -- deliberately subsampled so the block stays cheap
    relative to training).  The harmonic maximum principle does not apply to
    the nonlinear Liouville operator, so the violation entry is marked
    inapplicable rather than reported.
    """

    return {
        "pde_residual": {
            name: pde_residual(model, seed=seed, device=device, split=name)
            for name in sorted(SPLITS)
        },
        "pde_residual_note": (
            "||lap u + 2 e^u|| / ||e^u|| via float64 autograd at 32 interior "
            "points on two cases per split (the top-level 'pde_residual' is "
            "the in_distribution entry); exact solution scores 0, any "
            "harmonic prediction scores exactly 2"
        ),
        "max_principle_violation": None,
        "max_principle_note": (
            "n/a: Liouville is nonlinear (lap u = -2 e^u < 0), so the "
            "two-sided harmonic maximum principle does not apply and no "
            "boundary-range violation metric is licensed"
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
) -> dict:
    """Train one baseline on in-distribution Liouville samples and report."""

    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    model = _build_model(model_name).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]
    train_spec = SPLITS["in_distribution"]

    best_state, best_val, history = None, float("inf"), []
    start_time = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-6)
        for step in range(1, steps + 1):
            model.train()
            sample = build_liouville_sample(
                seed + 104_729 * step,
                geometry_modes=train_spec["geometry_modes"],
                deformation_range=train_spec["deformation_range"],
                device=device_t,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
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
            model, eval_seed=71_000_011, n_cases=4, device=device_t, dtype=dtype
        )["in_distribution"]

    # The zero-drive audit must run before pde_residual: pde_residual
    # converts the model to float64 in place, while this audit (like the
    # split evaluation) uses the float32 deployment dtype.
    zero_drive_rms = zero_drive_response(
        model,
        seed=89_000_023,
        n_cases=4,
        device=device_t,
        dtype=dtype,
    )
    report = {
        "model": model_name,
        "equation": "liouville: lap u + 2 exp(u) = 0",
        "seed": seed,
        "steps": steps,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_validation_relative_l2": best_val,
        "splits": evaluate_splits(
            model,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            device=device_t,
            dtype=dtype,
        ),
        "pde_residual": pde_residual(model, seed=83_000_019, device=device_t),
        "pde_residual_scale_note": (
            "||lap u + 2 e^u|| / ||e^u||: exact solution scores 0; any "
            "harmonic prediction scores exactly 2"
        ),
        "fidelity": fidelity_metrics(model, seed=83_000_019, device=device_t),
        "zero_drive_response_rms": zero_drive_rms,
        "zero_drive_scale_note": (
            "RMS interior prediction for boundary_value == 0; "
            "zero-preserving two-stream models score exactly 0, but the "
            "exact zero-Dirichlet Liouville solution is nonzero (u(0) = "
            "log 4 on the unit disk), so 0 is a misspecification here"
        ),
        "state": {
            k: v.tolist() for k, v in model.state_dict().items() if v.numel() <= 16
        },
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "pair_kernel",
            "harmonic_panel_bie",
            "boundary_mean",
            "mesh_transformer_kernel_linear",
            "mesh_transformer_kernel_nl",
            "mesh_transformer_kernel_nl_nomlp",
            "mesh_transformer_kernel_nl_singonly",
            "mesh_transformer_kernel_nl_singpair",
            "mesh_transformer_single_stream_nl",
        ),
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
    )
    print(json.dumps({k: result[k] for k in ("model", "splits", "pde_residual")}))
