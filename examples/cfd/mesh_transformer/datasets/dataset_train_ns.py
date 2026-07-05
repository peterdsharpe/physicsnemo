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

r"""Fixed-dataset training on the cataloged Navier-Stokes benchmark.

The multi-field sibling of :mod:`dataset_train` for the ``ns_cavity_star``
family: identical fixed-dataset protocol (AdamW at ``3e-4`` / weight decay
``1e-6``, gradient clipping at ``1.0``, reproducibly shuffled epochs,
gradient accumulation over ``--batch-cases``, best-validation checkpoint
restored before the final split evaluation), with the multi-field loss and
metrics conventions of the exact-label fluid suites
(``euler_bernoulli`` / ``euler_rotational``): the training loss is the
equal-weight mean of per-field relative squared errors and every evaluation
reports per-field relative L2 (``velocity``, ``pressure``), the combined
concatenated-field norm, and -- new for this suite -- **near-wall versus
interior query buckets** (split by the cataloged per-query wall distance at
the family's ``interior_margin``), the boundary-layer diagnostic.

Because the labels come from a solver on a genuinely nonlinear PDE, every
evaluation also records **per-case results with the case Reynolds number**,
so the Re-dependence of each arm's error (the pre-registered linear-arm
wall growth) is measurable from the report JSON without re-running.

Every report also carries an **operator-fidelity block** (added iteration
37; the iteration-36 archived runs predate it): the strong-form momentum
residual of the model's OWN prediction,
:math:`\tilde\nu\,\Delta u - (u\cdot\nabla)u - \nabla p` by float64
autograd at a deterministic 32-query interior subsample of two cases per
split, normalized by the prediction's advection norm
:math:`\|(u\cdot\nabla)u\|` (the euler-driver convention; no closed-form
scale exists on a solver-labeled family), plus the matching divergence
norm.  The coefficient :math:`\tilde\nu = 1/\mathrm{Re}` is read from the
CASE -- the model never sees the governing equations; the metric does.

ARMS (the iteration-36 comparison; all transformer arms use the pruned
two-member "singpair" kernel dictionary and the flipped one-head reference
configuration of iteration 32, exactly as ``euler_rotational``):

- ``mt_singpair_q2_pseudo`` -- ``field_mode="quadratic"`` plus
  ``drive_pseudo_dim=8``, the iteration-35 flagship.  NOTE the changed
  epistemic status: for the exact Euler families the pressure was *exactly*
  drive-quadratic, so declared degree 2 was a contract; for Navier-Stokes
  the solution map is analytic in the drive with **all** polynomial orders
  present (the Stokes component is drive-linear; each convection correction
  raises the degree), so the declared-degree-2 arm is now an APPROXIMATION
  HYPOTHESIS under test, not a representability guarantee.
- ``mt_singpair_nl_pseudo`` -- ``zero_preserving_nonlinear`` plus pseudo
  channels: the unbounded nonlinear control.
- ``mt_singpair_linear`` -- the PRE-REGISTERED Re-dependent wall control
  (logged before the first ns_cavity_star training run): at fixed operator
  scalar the exact map drive -> (velocity, pressure) is nonlinear in the
  drive for both fields once convection matters, and tends to the exactly
  linear Stokes map as Re -> 0.  Prediction: the linear arm's per-case
  error for BOTH fields grows with the case Reynolds number and approaches
  the nonlinear arms' error at the low-Re end of the band -- a wall whose
  height is a measurable function of Re, unlike the all-or-nothing parity
  and superposition walls.
- ``mt_singpair_linear_pseudo`` -- the iteration-38 CONFOUND CONTROL for
  iteration 37's velocity verdict: ``field_mode="linear"`` (exact
  fixed-geometry drive superposition, same read-in class as the wall
  control) PLUS ``drive_pseudo_dim=8`` -- the same pseudo/parity sector
  the nonlinear arm carries.  Iteration 37 read the flat Stokes-band
  velocity gap as a linear-READ-IN deficit, but its linear arm differed
  from nl_pseudo in parity machinery and parameter count too; this arm
  isolates the read-in class at matched sector structure (params as close
  to nl_pseudo as the mode allows -- the residual difference is the
  nonlinear read-in stack itself).  Pre-registered (logged before the
  first mt_singpair_linear_pseudo training run, iteration 38): if
  linear+pseudo closes the Stokes-band velocity gap (bottom joined bin
  gap to nl_pseudo <= 0.05 against iteration 37's measured 0.184), the
  deficit was parity/capacity and iteration 37's velocity reading is
  REVISED (the linear read-in class is fine where the physics is linear);
  falsifier: bottom-bin gap >= 0.1 means the read-in-class reading
  stands, now unconfounded.
- ``boundary_mean`` -- the parameter-free floor: the boundary-measure mean
  of the velocity drive for the velocity, zero for the pressure (whose
  relative L2 is therefore exactly 1.0 by construction).
- ``transolver_intree_matched`` -- the in-tree Transolver on the same token
  contract as every other suite (coordinates normalized by the
  measure-weighted boundary centroid and reference length; function
  channels: normal, dimensionless measure, the rank-1 boundary velocity,
  boundary/query indicator, plus the global viscosity as a constant
  channel), reading three output channels (velocity, pressure) at the
  query tokens.

The global operator scalar exposed to every arm is the dimensionless
viscosity :math:`\tilde\nu = 1/\mathrm{Re}` -- the coefficient of the
nondimensional momentum equation (unit peak drive speed, unit reference
length), the same "feed the PDE coefficient" convention as the screened
suite's :math:`\tilde\kappa`.  The catalog also stores ``reynolds`` per
case for reporting.

This is a benchmark-local research driver, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import dataset_catalog
import torch
from models import MeshTransformerConfig
from potential_flow import _KERNEL_ARM_KWARGS, _ONE_HEAD_CONFIG
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

FAMILY = "ns_cavity_star"
VALIDATION_SPLIT = "eval_id"
MAX_VALIDATION_CASES = 8

OUTPUT_FIELD_RANKS: dict[str, int] = {"velocity": 1, "pressure": 0}

_SINGPAIR_KWARGS = _KERNEL_ARM_KWARGS["mesh_transformer_kernel_singpair"]

_SCHEMA: dict[str, dict] = {
    "boundary_field_ranks": {
        "dirichlet": {"operator": {}, "drive": {"boundary_velocity": 1}}
    },
    "global_field_ranks": {"operator": {"viscosity": 0}, "drive": {}},
}

MODEL_NAMES: tuple[str, ...] = (
    "boundary_mean",
    "mt_singpair_linear",
    "mt_singpair_linear_pseudo",
    "mt_singpair_nl_pseudo",
    "mt_singpair_q2_pseudo",
    "transolver_intree_matched",
)

#: Field mode and pseudo sector width per MeshTransformer arm.
_ARM_MODES: dict[str, tuple[str, int]] = {
    "mt_singpair_linear": ("linear", 0),
    "mt_singpair_linear_pseudo": ("linear", 8),
    "mt_singpair_nl_pseudo": ("zero_preserving_nonlinear", 8),
    "mt_singpair_q2_pseudo": ("quadratic", 8),
}

_DESIGN_NOTES = {
    "pre_registered_linear_wall": (
        "logged before the first ns_cavity_star training run: at fixed "
        "operator scalar the exact solution map is nonlinear in the drive "
        "for BOTH velocity and pressure once convection matters, and tends "
        "to the exactly linear Stokes map as Re -> 0; the linear arm's "
        "per-case relative L2 for both fields should therefore GROW with "
        "the case Reynolds number and approach the nonlinear arms' at the "
        "low-Re end -- measured from the per_case blocks of the reports"
    ),
    "q2_approximation_note": (
        "for Navier-Stokes the declared-degree-2 arm is an APPROXIMATION "
        "hypothesis, not a contract: the solution map is analytic in the "
        "drive with all polynomial orders present (Stokes part linear, "
        "each convection correction raising the degree); pre-registered "
        "read-out: q2_pseudo within 2x seed sd of nl_pseudo on eval_id "
        "means the degree-2 truncation suffices at these Re"
    ),
    "boundary_mean_floor_note": (
        "the pressure has no boundary drive, so the floor predicts zero "
        "and its pressure relative L2 is exactly 1.0 by construction"
    ),
    "linear_pseudo_confound_check": (
        "iteration 38, logged before the first mt_singpair_linear_pseudo "
        "training run: iteration 37's velocity verdict (flat Stokes-band "
        "gap = linear-read-in deficit) was confounded -- the linear arm "
        "also lacked the pseudo sector and 2.2x the parameters.  The "
        "linear_pseudo arm (field_mode='linear', drive_pseudo_dim=8) is "
        "the fair control.  Pre-registered outcomes on the joined "
        "gap-vs-Re curve (gap = linear_pseudo minus nl_pseudo, per log-Re "
        "bin, seeds pooled, eval_id + eval_unseen_Re): bottom joined bin "
        "velocity gap <= 0.05 REVISES iteration 37 (deficit was "
        "parity/capacity, the linear read-in class is fine where the "
        "physics is linear); bottom-bin gap >= 0.1 means the "
        "read-in-class reading STANDS, now unconfounded.  Either outcome "
        "is progress; both were stated in advance"
    ),
}


class DriveBoundaryMean(nn.Module):
    """Parameter-free per-field boundary-mean floor (euler conventions)."""

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


def _ns_token_sequence(
    domain: DomainMesh,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Shared boundary+query token sequence for the Transolver arm.

    The exact adaptation contract of ``transolver.build_token_sequence``,
    lifted to the rank-1 drive: coordinates are centered at the
    measure-weighted boundary centroid and normalized by the reference
    length; the function channels are the outward unit normal (2), the
    dimensionless cell measure (1), the boundary velocity (2), the
    boundary/query indicator (1), and the global dimensionless viscosity as
    a constant channel on every token (1) -- 7 channels total, zeros on
    query tokens except the viscosity.
    """

    boundary = domain.boundaries["dirichlet"]
    length = domain.global_data["reference_length"].reshape(())
    with torch.autocast(device_type=boundary.points.device.type, enabled=False):
        measures = boundary.cell_areas / length
        center = torch.einsum("s,sd->d", measures, boundary.cell_centroids)
        center = center / measures.sum()
        source_points = (boundary.cell_centroids - center) / length
        query_points = (domain.interior.points - center) / length
        normals = boundary.cell_normals
    velocity = boundary.cell_data["boundary_velocity"]
    viscosity = domain.global_data["viscosity"].reshape(())

    to_model = {"device": device, "dtype": dtype}
    source_points = source_points.to(**to_model)
    query_points = query_points.to(**to_model)
    boundary_features = torch.cat(
        (
            normals.to(**to_model),
            measures.to(**to_model)[:, None],
            velocity.to(**to_model),
            source_points.new_ones(source_points.shape[0], 1),
        ),
        dim=-1,
    )
    query_features = query_points.new_zeros(query_points.shape[0], 6)
    coordinates = torch.cat((source_points, query_points), dim=0)
    features = torch.cat((boundary_features, query_features), dim=0)
    features = torch.cat(
        (
            features,
            viscosity.to(**to_model).expand(features.shape[0])[:, None],
        ),
        dim=-1,
    )
    return coordinates[None], features[None], source_points.shape[0]


class InTreeTransolverNSAdapter(nn.Module):
    """The in-tree Transolver on the multi-field N-S benchmark protocol."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from physicsnemo.models.transolver import Transolver

        self.model = Transolver(
            functional_dim=7,
            embedding_dim=2,
            out_dim=3,
            n_layers=layers,
            n_hidden=hidden_dim,
            n_head=heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            structured_shape=None,
            unified_pos=False,
            use_te=False,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        parameter = next(self.model.parameters())
        coordinates, features, n_boundary = _ns_token_sequence(
            domain, device=parameter.device, dtype=parameter.dtype
        )
        output = self.model(features, embedding=coordinates)  # (1, S + Q, 3)
        return domain.interior.with_data(
            point_data={
                "velocity": output[0, n_boundary:, 0:2],
                "pressure": output[0, n_boundary:, 2],
            },
            cell_data={},
            global_data=domain.global_data,
        )


def make_ns_model(model_name: str) -> nn.Module:
    """Instantiate one arm of the N-S fixed-dataset comparison."""

    if model_name == "boundary_mean":
        return DriveBoundaryMean()
    if model_name == "transolver_intree_matched":
        from transolver import TRANSOLVER_PRESETS

        return InTreeTransolverNSAdapter(**asdict(TRANSOLVER_PRESETS["matched"]))
    if model_name not in _ARM_MODES:
        raise ValueError(f"unknown model {model_name!r}")

    from physicsnemo.experimental.nn import MeshTransformer

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
# Data plumbing
# ---------------------------------------------------------------------------


class NSCaseBank:
    """Serve cataloged N-S cases as device-resident samples (cached)."""

    def __init__(
        self,
        directory: Path | str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache: bool = True,
    ) -> None:
        self._directory = Path(directory)
        self._device = device
        self._dtype = dtype
        self._cache: dict[int, tuple] | None = {} if cache else None

    def sample(self, index: int):
        """Return ``(domain, targets, wall_distance, reynolds)`` for a case."""

        if self._cache is not None and index in self._cache:
            return self._cache[index]
        case = dataset_catalog.load_case(self._directory, index)
        domain, targets = dataset_catalog.load_ns_domain_sample(
            case, device=self._device, dtype=self._dtype
        )
        wall_distance = torch.from_numpy(case.arrays["query_wall_distance"].copy()).to(
            device=self._device, dtype=self._dtype
        )
        entry = (domain, targets, wall_distance, float(case.params["reynolds"]))
        if self._cache is not None:
            self._cache[index] = entry
        return entry


def _predictions(model: nn.Module, domain: DomainMesh) -> dict[str, torch.Tensor]:
    """Predict with the targets stripped at the benchmark boundary."""

    model_domain = DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}, global_data={}),
        boundaries=domain.boundaries,
        global_data=domain.global_data,
    )
    point_data = model(model_domain).point_data
    return {name: point_data[name] for name in OUTPUT_FIELD_RANKS}


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _combined_relative_l2(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> float:
    numerator = sum(
        float((predictions[name] - targets[name]).square().sum()) for name in targets
    )
    denominator = sum(float(targets[name].square().sum()) for name in targets)
    return math.sqrt(numerator / max(denominator, 1.0e-30))


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
def _evaluate_cases(
    model: nn.Module,
    bank: NSCaseBank,
    indices: list[int],
    *,
    near_threshold: float,
) -> tuple[dict[str, float], list[dict]]:
    """Aggregate and per-case metrics over a fixed case-index list."""

    model.eval()
    sums: dict[str, list[float]] = {}
    per_case: list[dict] = []
    for index in indices:
        domain, targets, wall_distance, reynolds = bank.sample(index)
        predictions = _predictions(model, domain)
        near = wall_distance < near_threshold
        record: dict = {"case": index, "reynolds": reynolds}
        record["combined"] = _combined_relative_l2(predictions, targets)
        for name in OUTPUT_FIELD_RANKS:
            record[name] = _relative_l2(predictions[name], targets[name])
            for bucket, mask in (("near_wall", near), ("interior", ~near)):
                if bool(mask.any()):
                    record[f"{name}/{bucket}"] = _relative_l2(
                        predictions[name][mask], targets[name][mask]
                    )
        per_case.append(record)
        for key, value in record.items():
            if key in ("case", "reynolds"):
                continue
            sums.setdefault(key, []).append(value)
    aggregate = {key: sum(values) / len(values) for key, values in sums.items()}
    return aggregate, per_case


# ---------------------------------------------------------------------------
# Operator-fidelity block (strong-form residuals of the model's own outputs)
# ---------------------------------------------------------------------------

FIDELITY_CASES_PER_SPLIT = 2
FIDELITY_QUERIES = 32


def _case_fidelity_residuals(
    model_fp64: nn.Module,
    case: "dataset_catalog.CatalogCase",
    *,
    device: torch.device,
) -> tuple[float, float]:
    """One case's (momentum, divergence) strong-form prediction residuals.

    Float64 autograd at a deterministic :data:`FIDELITY_QUERIES`-point
    subsample of the case's *interior-bucket* queries (near-wall points
    excluded: the strong form is evaluated where the solution is smooth on
    the model's own scale).  The momentum residual is

    .. math:: \\|\\tilde\\nu\\,\\Delta u - (u\\cdot\\nabla)u - \\nabla p\\|
              \\;/\\; \\|(u\\cdot\\nabla)u\\|

    with every field the model's own output and :math:`\\tilde\\nu = 1/Re`
    read from the CASE -- the model never sees the governing equations, the
    metric does.  The normalizer is the prediction's own advection norm
    (the euler-driver convention; on a solver-labeled family no closed-form
    advection scale exists).  The divergence residual is
    ``L * ||div u|| / ||u||``.  A solver-exact field scores ~0 on both;
    note a constant velocity with zero pressure also zeroes the momentum
    strong form (it fails the *boundary conditions*, not the interior PDE),
    so the block complements -- never replaces -- the relative-L2 columns.
    """

    domain, _ = dataset_catalog.load_ns_domain_sample(
        case, device=device, dtype=torch.float64
    )
    wall = torch.from_numpy(case.arrays["query_wall_distance"].copy())
    margin = float(case.params["queries"]["interior_margin"])
    interior = torch.nonzero(wall >= margin).squeeze(-1)
    generator = torch.Generator().manual_seed(case.index)
    subsample = interior[
        torch.randperm(interior.numel(), generator=generator)[:FIDELITY_QUERIES]
    ].to(device)
    points = domain.interior.points[subsample].clone().requires_grad_(True)
    fidelity_domain = DomainMesh(
        interior=Mesh(points=points),
        boundaries=dict(domain.boundaries.items()),
        global_data=domain.global_data,
    )
    out = model_fp64(fidelity_domain).point_data
    u, p = out["velocity"], out["pressure"]

    laplacian = torch.zeros_like(u)
    advection = torch.zeros_like(u)
    divergence = torch.zeros(points.shape[0], dtype=torch.float64, device=device)
    if u.grad_fn is not None:
        for component in range(2):
            (gradient,) = torch.autograd.grad(
                u[:, component].sum(), points, create_graph=True, allow_unused=True
            )
            if gradient is None:
                continue
            advection[:, component] = (u * gradient).sum(dim=-1)
            divergence = divergence + gradient[:, component]
            if gradient.grad_fn is None:
                continue  # first derivative constant in x: Laplacian is zero
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
    grad_p = None
    if p.grad_fn is not None:
        (grad_p,) = torch.autograd.grad(
            p.sum(), points, create_graph=True, allow_unused=True
        )
    if grad_p is None:
        grad_p = torch.zeros_like(u)

    viscosity = domain.global_data["viscosity"].reshape(()).to(torch.float64)
    length = domain.global_data["reference_length"].reshape(())
    momentum = (viscosity * laplacian - advection - grad_p).detach()
    momentum_residual = float(
        torch.linalg.vector_norm(momentum)
        / torch.linalg.vector_norm(advection.detach()).clamp_min(1.0e-30)
    )
    divergence_residual = float(
        length.detach()
        * torch.linalg.vector_norm(divergence.detach())
        / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
    )
    return momentum_residual, divergence_residual


def fidelity_metrics(
    model: nn.Module,
    *,
    dataset_dir: Path | str,
    manifest: dict,
    split_names: list[str],
    device: torch.device,
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per evaluation split: the strong-form N-S momentum residual and the
    divergence norm of the model's own (velocity, pressure) prediction --
    float64 autograd, the first :data:`FIDELITY_CASES_PER_SPLIT` cases per
    split and :data:`FIDELITY_QUERIES` interior queries per case, so the
    block stays cheap relative to training.  Restores the model to float32
    before returning.
    """

    model_fp64 = model.double()
    model_fp64.eval()
    momentum: dict[str, float] = {}
    divergence: dict[str, float] = {}
    try:
        for name in split_names:
            indices = list(dataset_catalog.split_indices(manifest, name))[
                :FIDELITY_CASES_PER_SPLIT
            ]
            values = [
                _case_fidelity_residuals(
                    model_fp64,
                    dataset_catalog.load_case(dataset_dir, index),
                    device=device,
                )
                for index in indices
            ]
            momentum[name] = sum(v[0] for v in values) / len(values)
            divergence[name] = sum(v[1] for v in values) / len(values)
    finally:
        model.float()
    return {
        "momentum_residual": momentum,
        "momentum_residual_note": (
            "||nu~ lap u_pred - (u_pred . grad) u_pred - grad p_pred|| / "
            "||(u_pred . grad) u_pred||: the strong-form steady N-S momentum "
            "residual of the model's own two output fields (float64 "
            "autograd, 32 interior queries, two cases per split); nu~ = 1/Re "
            "comes from the CASE, not the model -- the model never sees the "
            "governing equations, the metric does.  The normalizer is the "
            "prediction's own advection norm (no closed-form scale exists on "
            "a solver-labeled family); solver-exact fields score ~0"
        ),
        "divergence_residual": divergence,
        "divergence_residual_note": (
            "||div u_pred|| L / ||u_pred||: the incompressibility strong "
            "form of the predicted velocity at the same subsample"
        ),
        "n_cases_per_split": FIDELITY_CASES_PER_SPLIT,
        "n_queries": FIDELITY_QUERIES,
    }


def run_experiment(
    *,
    dataset_dir: Path | str,
    model_name: str,
    epochs: int,
    seed: int,
    device: str,
    output_dir: Path | str,
    batch_cases: int = 1,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-6,
    cache: bool = True,
) -> dict:
    """Train one arm on the cataloged N-S dataset for a fixed epoch count."""

    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if batch_cases < 1:
        raise ValueError("batch_cases must be positive")

    dataset_dir = Path(dataset_dir)
    manifest = dataset_catalog.load_manifest(dataset_dir)
    if manifest["family"] != FAMILY:
        raise dataset_catalog.CatalogError(
            f"dataset family {manifest['family']!r} is not {FAMILY!r}"
        )
    near_threshold = float(manifest["generator_settings"].get("interior_margin", 0.12))
    train_indices = list(dataset_catalog.split_indices(manifest, "train"))
    eval_split_names = sorted(name for name in manifest["splits"] if name != "train")
    if VALIDATION_SPLIT not in eval_split_names:
        raise dataset_catalog.CatalogError(
            f"manifest defines no {VALIDATION_SPLIT!r} split for validation"
        )
    validation_indices = list(
        dataset_catalog.split_indices(manifest, VALIDATION_SPLIT)
    )[:MAX_VALIDATION_CASES]

    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    model = make_ns_model(model_name).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]
    bank = NSCaseBank(dataset_dir, device=device_t, dtype=dtype, cache=cache)

    def validation_score() -> float:
        aggregate, _ = _evaluate_cases(
            model, bank, validation_indices, near_threshold=near_threshold
        )
        return aggregate["combined"]

    history: list[dict[str, float | int]] = []
    best_state, best_val, best_epoch = None, float("inf"), 0
    start_time = time.perf_counter()
    if parameters and epochs > 0:
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        shuffler = torch.Generator(device="cpu").manual_seed(seed)
        validate_every = max(1, epochs // 12)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_indices), generator=shuffler).tolist()
            losses: list[float] = []
            for batch_start in range(0, len(order), batch_cases):
                batch = order[batch_start : batch_start + batch_cases]
                optimizer.zero_grad(set_to_none=True)
                for position in batch:
                    domain, targets, _, _ = bank.sample(train_indices[position])
                    loss = _multi_field_loss(_predictions(model, domain), targets)
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach().cpu()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if epoch % validate_every == 0 or epoch == epochs:
                validation = validation_score()
                record["validation_combined_relative_l2"] = validation
                if validation < best_val:
                    best_val = validation
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            history.append(record)
            print(json.dumps(record), flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_val = validation_score()

    splits: dict[str, dict] = {}
    per_case: dict[str, list[dict]] = {}
    for name in eval_split_names:
        aggregate, cases = _evaluate_cases(
            model,
            bank,
            list(dataset_catalog.split_indices(manifest, name)),
            near_threshold=near_threshold,
        )
        splits[name] = aggregate
        per_case[name] = cases

    fidelity = fidelity_metrics(
        model,
        dataset_dir=dataset_dir,
        manifest=manifest,
        split_names=eval_split_names,
        device=device_t,
    )

    report = {
        "model": model_name,
        "family": FAMILY,
        "dataset": {
            "family": manifest["family"],
            "version": manifest["version"],
            "path": str(dataset_dir.resolve()),
            "n_cases": manifest["n_cases"],
        },
        "seed": seed,
        "epochs": epochs,
        "batch_cases": batch_cases,
        "n_train_cases": len(train_indices),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": history,
        "best_validation_combined_relative_l2": best_val,
        "best_epoch": best_epoch,
        "validation_split": VALIDATION_SPLIT,
        "validation_cases": len(validation_indices),
        "near_wall_threshold": near_threshold,
        "splits": splits,
        "per_case": per_case,
        "fidelity": fidelity,
        "design_notes": _DESIGN_NOTES,
        "split_sizes": {
            name: spec["stop"] - spec["start"]
            for name, spec in manifest["splits"].items()
        },
        "verification": manifest["verification"],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_experiment(
        dataset_dir=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch_cases=args.batch_cases,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        cache=not args.no_cache,
    )
    print(
        json.dumps(
            {
                "model": report["model"],
                "best_validation_combined_relative_l2": report[
                    "best_validation_combined_relative_l2"
                ],
                "splits": {
                    name: {
                        key: values[key] for key in ("combined", "velocity", "pressure")
                    }
                    for name, values in report["splits"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
