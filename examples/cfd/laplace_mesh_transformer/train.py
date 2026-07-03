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

"""Train and evaluate mesh-native surrogates on exact Laplace problems.

This script intentionally uses one variable-size ``DomainMesh`` at a time.
Synthetic cases are generated online, so an epoch over a small fixed geometry
set cannot masquerade as geometric generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from conformal_laplace import (
    ConformalGeometry,
    ConformalLaplaceSample,
    HarmonicDrive,
    SimilarityTransform,
    build_domain_sample,
    build_neumann_domain_sample,
    evaluate_potential,
    sample_drive,
    sample_geometry,
    sample_similarity,
    unit_circle,
)
from metrics import (
    aggregate_metrics,
    case_metrics,
    certified_maximum_principle_violation,
    weighted_relative_l2,
)
from models import (
    BoundaryMean,
    InvariantPairKernel,
    MeshTransformerConfig,
    build_lifted_mesh_transformer,
    build_mesh_transformer,
    parameter_count,
)
from provenance import runtime_environment, source_provenance
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

ModelName = Literal[
    "mesh_transformer",
    "lifted_mesh_transformer",
    "pair_kernel",
    "encoded_pair_kernel",
    "self_consistent_pair_kernel",
    "self_consistent_pair_kernel_untied",
    "self_consistent_pair_kernel_trace",
    "self_consistent_pair_kernel_full",
    "pair_kernel_harmonic",
    "harmonic_kernel_direct",
    "harmonic_kernel_bie",
    "harmonic_kernel_bie_trace",
    "harmonic_panel_direct",
    "harmonic_panel_bie",
    "harmonic_panel_bie_minimal",
    "harmonic_panel_bie_2param",
    "harmonic_panel_bie_p2",
    "neumann_harmonic_panel_direct",
    "neumann_harmonic_panel_bie",
    "neumann_harmonic_panel_bie_minimal",
    "stf_multipole_l1",
    "stf_multipole_l2",
    "stf_multipole_l4",
    "double_layer_direct",
    "double_layer_solved",
    "double_layer_richardson",
    "double_layer_encoded",
    "globe_exact",
    "globe_hierarchical",
    "geotransolver_matched",
    "geotransolver_published_scale",
    "boundary_mean",
]
MODEL_NAMES = (
    "mesh_transformer",
    "lifted_mesh_transformer",
    "pair_kernel",
    "encoded_pair_kernel",
    "self_consistent_pair_kernel",
    "self_consistent_pair_kernel_untied",
    "self_consistent_pair_kernel_trace",
    "self_consistent_pair_kernel_full",
    "pair_kernel_harmonic",
    "harmonic_kernel_direct",
    "harmonic_kernel_bie",
    "harmonic_kernel_bie_trace",
    "harmonic_panel_direct",
    "harmonic_panel_bie",
    "harmonic_panel_bie_minimal",
    "harmonic_panel_bie_2param",
    "harmonic_panel_bie_p2",
    "neumann_harmonic_panel_direct",
    "neumann_harmonic_panel_bie",
    "neumann_harmonic_panel_bie_minimal",
    "stf_multipole_l1",
    "stf_multipole_l2",
    "stf_multipole_l4",
    "double_layer_direct",
    "double_layer_solved",
    "double_layer_richardson",
    "double_layer_encoded",
    "globe_exact",
    "globe_hierarchical",
    "geotransolver_matched",
    "geotransolver_published_scale",
    "boundary_mean",
)
# Models that consume Neumann flux data; every other model reads Dirichlet
# values.  The pairing with --problem is validated explicitly so that a
# missing cell_data key surfaces as a configuration error, never as silent
# training on the wrong boundary condition.
NEUMANN_MODEL_NAMES = (
    "neumann_harmonic_panel_direct",
    "neumann_harmonic_panel_bie",
    "neumann_harmonic_panel_bie_minimal",
)
Problem = Literal["dirichlet", "neumann"]
TrainingDriveDistribution = Literal[
    "boundary_balanced_mixture",
    "disk_interior_balanced_mixture",
    "uniform_pure_mode",
]
TrainingObjective = Literal[
    "auto",
    "interior_supervision",
    "boundary_collocation",
    "interior_plus_auxiliary",
]


@dataclass(frozen=True)
class SplitSpec:
    """One controlled generalization axis."""

    geometry_modes: tuple[int, ...]
    deformation_range: tuple[float, float]
    drive_modes: tuple[int, ...]
    drive_regularity: float
    drive_include_constant: bool


TRAIN_SPLIT = SplitSpec(
    geometry_modes=(2, 3),
    deformation_range=(0.05, 0.35),
    drive_modes=(1, 2, 3, 4),
    # Equal boundary-variance coefficients remove an additional imposed
    # spectral decay. They do not equalize physical interior energy: on the
    # disk, harmonic mode k has area energy proportional to 1 / (k + 1).
    drive_regularity=0.0,
    drive_include_constant=True,
)

EVALUATION_SPLITS: dict[str, SplitSpec] = {
    "interpolation": TRAIN_SPLIT,
    "unseen_geometry_modes": SplitSpec(
        geometry_modes=(4, 5),
        deformation_range=(0.05, 0.35),
        drive_modes=TRAIN_SPLIT.drive_modes,
        drive_regularity=TRAIN_SPLIT.drive_regularity,
        drive_include_constant=TRAIN_SPLIT.drive_include_constant,
    ),
    "stronger_deformation": replace(
        TRAIN_SPLIT,
        deformation_range=(0.45, 0.65),
    ),
    "mixed_geometry_modes": SplitSpec(
        geometry_modes=(2, 3, 4, 5),
        deformation_range=TRAIN_SPLIT.deformation_range,
        drive_modes=TRAIN_SPLIT.drive_modes,
        drive_regularity=TRAIN_SPLIT.drive_regularity,
        drive_include_constant=TRAIN_SPLIT.drive_include_constant,
    ),
    "unseen_boundary_frequencies": replace(
        TRAIN_SPLIT,
        drive_modes=(5, 6, 7, 8),
        drive_regularity=0.0,
        drive_include_constant=False,
    ),
}


TINY_CONFIG = MeshTransformerConfig(
    operator_scalar_dim=16,
    operator_vector_dim=4,
    drive_scalar_dim=24,
    drive_vector_dim=6,
    operator_layers=1,
    drive_layers=1,
    query_layers=1,
    heads=2,
    scalar_rank=6,
    vector_rank=2,
)

# 9,934 trainable parameters versus 9,880 in the default lmax=4 STF model.
# This 0.55% mismatch is recorded rather than hiding ordinary scalar/vector
# capacity behind a much larger control.
STF_MATCHED_CONFIG = MeshTransformerConfig(
    operator_scalar_dim=13,
    operator_vector_dim=3,
    drive_scalar_dim=20,
    drive_vector_dim=4,
    operator_layers=1,
    drive_layers=1,
    query_layers=1,
    heads=2,
    scalar_rank=5,
    vector_rank=2,
)

SHALLOW_CONFIG = MeshTransformerConfig(
    operator_scalar_dim=32,
    operator_vector_dim=8,
    drive_scalar_dim=48,
    drive_vector_dim=12,
    operator_layers=0,
    drive_layers=0,
    query_layers=1,
    heads=4,
    scalar_rank=16,
    vector_rank=8,
)

REFERENCE_CONFIG = MeshTransformerConfig()

LARGE_CONFIG = MeshTransformerConfig(
    operator_scalar_dim=48,
    operator_vector_dim=12,
    drive_scalar_dim=64,
    drive_vector_dim=16,
    operator_layers=3,
    drive_layers=2,
    query_layers=1,
    heads=4,
    scalar_rank=16,
    vector_rank=8,
)

CAPACITY_CONFIGS = {
    "tiny": TINY_CONFIG,
    "stf_matched": STF_MATCHED_CONFIG,
    "shallow": SHALLOW_CONFIG,
    "reference": REFERENCE_CONFIG,
    "large": LARGE_CONFIG,
}


@dataclass(frozen=True)
class RunConfig:
    """Fully specified training and evaluation protocol for one study run."""

    model: ModelName = "mesh_transformer"
    capacity: str = "reference"
    problem: Problem = "dirichlet"
    steps: int = 2000
    cases_per_step: int = 1
    train_boundary_points: int = 64
    train_query_points: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    seed: int = 17
    validation_seed: int = 71_000_011
    evaluation_seed: int = 83_000_019
    report_every: int = 50
    validation_every: int = 250
    validation_cases: int = 8
    evaluation_cases: int = 32
    evaluation_boundary_points: int = 128
    evaluation_query_points: int = 512
    harmonic_cases: int = 2
    matmul_precision: str = "highest"
    training_drive_distribution: TrainingDriveDistribution = "boundary_balanced_mixture"
    training_objective: TrainingObjective = "auto"


def _case_seed(seed: int, case_index: int, stream: int) -> int:
    """Derive independent deterministic streams without mutable RNG state."""

    return seed + 1_000_003 * case_index + 104_729 * stream


def _build_sample(
    problem: Problem,
    geometry: ConformalGeometry,
    drive: HarmonicDrive,
    *,
    n_boundary: int,
    n_query: int = 512,
    query_seed: int = 0,
    query_preimages: torch.Tensor | None = None,
    similarity: SimilarityTransform | None = None,
) -> ConformalLaplaceSample:
    """Build one sample of the declared boundary-condition problem."""

    builder = {
        "dirichlet": build_domain_sample,
        "neumann": build_neumann_domain_sample,
    }
    try:
        build = builder[problem]
    except KeyError:
        raise ValueError(f"unknown problem {problem!r}") from None
    return build(
        geometry,
        drive,
        n_boundary=n_boundary,
        n_query=n_query,
        query_seed=query_seed,
        query_preimages=query_preimages,
        similarity=similarity,
    )


def make_case(
    spec: SplitSpec,
    *,
    seed: int,
    case_index: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> ConformalLaplaceSample:
    """Create one deterministic case whose latent parameters are never inputs."""

    geometry = sample_geometry(
        _case_seed(seed, case_index, 0),
        modes=spec.geometry_modes,
        deformation_range=spec.deformation_range,
        device=device,
        dtype=dtype,
    )
    drive = sample_drive(
        _case_seed(seed, case_index, 1),
        modes=spec.drive_modes,
        regularity=spec.drive_regularity,
        boundary_rms=1.0,
        include_constant=spec.drive_include_constant,
        device=device,
        dtype=dtype,
    )
    return _build_sample(
        problem,
        geometry,
        drive,
        n_boundary=n_boundary,
        n_query=n_query,
        query_seed=_case_seed(seed, case_index, 2),
    )


def make_training_case(
    spec: SplitSpec,
    *,
    distribution: TrainingDriveDistribution,
    seed: int,
    case_index: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> ConformalLaplaceSample:
    r"""Create one case for a controlled boundary-spectrum objective.

    ``disk_interior_balanced_mixture`` fixes coefficient magnitudes so every
    included component has exactly equal area energy on the unit disk, then
    samples only phases/signs. It is an explicit diagnostic approximation on
    deformed domains. ``uniform_pure_mode`` draws one nonconstant mode uniformly
    per case, preventing mixed-loss weighting from hiding a represented mode.
    """

    if distribution == "boundary_balanced_mixture":
        training_spec = spec
    elif distribution == "disk_interior_balanced_mixture":
        base = make_case(
            spec,
            seed=seed,
            case_index=case_index,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        # For u_k(r, theta) = Re(c_k r**k exp(i k theta)), exact unit-disk
        # area energy is pi |c_k|**2 / (2(k+1)); the constant contributes
        # pi c_0**2. Fixed magnitudes |c_k|=sqrt(2(k+1)) and |c_0|=1
        # therefore give every included component equal physical-area energy.
        # Only phases/signs are random, avoiding normalization-induced
        # correlations from merely preweighting Gaussian amplitudes.
        phases = torch.polar(
            torch.ones_like(base.drive.coefficients.real),
            torch.angle(base.drive.coefficients),
        )
        magnitudes = base.drive.coefficients.real.new_tensor(
            [math.sqrt(2.0 * (mode + 1.0)) for mode in base.drive.modes]
        )
        coefficients = phases * magnitudes
        if spec.drive_include_constant:
            constant = torch.where(
                base.drive.constant < 0.0,
                -torch.ones_like(base.drive.constant),
                torch.ones_like(base.drive.constant),
            )
        else:
            constant = torch.zeros_like(base.drive.constant)
        energy = constant.square() + 0.5 * coefficients.abs().square().sum()
        normalization = torch.rsqrt(energy)
        balanced_drive = HarmonicDrive(
            constant=constant * normalization,
            modes=base.drive.modes,
            coefficients=coefficients * normalization,
        )
        return _build_sample(
            problem,
            base.geometry,
            balanced_drive,
            n_boundary=n_boundary,
            query_preimages=base.query_preimages,
            similarity=base.similarity,
        )
    elif distribution == "uniform_pure_mode":
        if not spec.drive_modes:
            raise ValueError("uniform_pure_mode requires at least one drive mode")
        generator = torch.Generator(device="cpu").manual_seed(
            _case_seed(seed, case_index, 3)
        )
        mode_index = int(
            torch.randint(len(spec.drive_modes), (), generator=generator).item()
        )
        training_spec = replace(
            spec,
            drive_modes=(spec.drive_modes[mode_index],),
            drive_regularity=0.0,
            drive_include_constant=False,
        )
    else:
        raise ValueError(f"unknown training drive distribution {distribution!r}")
    return make_case(
        training_spec,
        seed=seed,
        case_index=case_index,
        n_boundary=n_boundary,
        n_query=n_query,
        device=device,
        dtype=dtype,
        problem=problem,
    )


def make_model(model_name: ModelName, capacity: str) -> nn.Module:
    """Construct a benchmark model without silently equalizing unlike costs."""

    if model_name == "mesh_transformer":
        try:
            config = CAPACITY_CONFIGS[capacity]
        except KeyError:
            raise ValueError(f"unknown capacity {capacity!r}") from None
        return build_mesh_transformer(config)
    if model_name == "lifted_mesh_transformer":
        try:
            config = CAPACITY_CONFIGS[capacity]
        except KeyError:
            raise ValueError(f"unknown capacity {capacity!r}") from None
        return build_lifted_mesh_transformer(config)
    if model_name == "pair_kernel":
        return InvariantPairKernel()
    if model_name == "encoded_pair_kernel":
        from research_models import EncodedInvariantPairKernel

        try:
            config = CAPACITY_CONFIGS[capacity]
        except KeyError:
            raise ValueError(f"unknown capacity {capacity!r}") from None
        return EncodedInvariantPairKernel(config)
    if model_name.startswith("neumann_harmonic_panel"):
        from self_consistent_kernel import NeumannHarmonicPanelBIE

        return NeumannHarmonicPanelBIE(
            n_iterations=0 if model_name == "neumann_harmonic_panel_direct" else 8,
            regular_orders=(
                0 if model_name == "neumann_harmonic_panel_bie_minimal" else 3
            ),
        )
    if model_name.startswith("harmonic_panel"):
        from self_consistent_kernel import HarmonicPanelBIE

        settings = {
            "harmonic_panel_direct": dict(n_iterations=0),
            "harmonic_panel_bie": dict(),
            "harmonic_panel_bie_minimal": dict(regular_orders=0),
            "harmonic_panel_bie_2param": dict(
                regular_orders=0, shared_relaxation=True
            ),
            "harmonic_panel_bie_p2": dict(regular_orders=0, n_iterations=2),
        }
        return HarmonicPanelBIE(**settings[model_name])
    if model_name.startswith(("self_consistent_pair_kernel", "harmonic_kernel")) or (
        model_name == "pair_kernel_harmonic"
    ):
        from self_consistent_kernel import SelfConsistentPairKernel

        variants = {
            "self_consistent_pair_kernel": {},
            "self_consistent_pair_kernel_untied": {"tied": False},
            "self_consistent_pair_kernel_trace": {"trace_loss": True},
            "self_consistent_pair_kernel_full": {
                "trace_loss": True,
                "kernel_pde_loss": True,
            },
            "pair_kernel_harmonic": {"n_iterations": 0, "kernel_pde_loss": True},
            "harmonic_kernel_direct": {
                "kernel_family": "harmonic",
                "n_iterations": 0,
            },
            "harmonic_kernel_bie": {"kernel_family": "harmonic"},
            "harmonic_kernel_bie_trace": {
                "kernel_family": "harmonic",
                "trace_loss": True,
            },
        }
        return SelfConsistentPairKernel(**variants[model_name])
    if model_name.startswith("stf_multipole_l"):
        from stf_multipole import STFMultipolePotential

        order = int(model_name.removeprefix("stf_multipole_l"))
        return STFMultipolePotential(lmax=order)
    if model_name == "double_layer_direct":
        from layer_potential import DirectDoubleLayerPotential

        return DirectDoubleLayerPotential()
    if model_name == "double_layer_solved":
        from layer_potential import SolvedDoubleLayerPotential

        return SolvedDoubleLayerPotential()
    if model_name == "double_layer_richardson":
        from layer_potential import LearnedDensityDoubleLayerPotential

        return LearnedDensityDoubleLayerPotential()
    if model_name == "double_layer_encoded":
        from layer_potential import EncodedDoubleLayerPotential

        try:
            config = CAPACITY_CONFIGS[capacity]
        except KeyError:
            raise ValueError(f"unknown capacity {capacity!r}") from None
        return EncodedDoubleLayerPotential(config)
    if model_name in ("globe_exact", "globe_hierarchical"):
        from external_baselines import GlobeLaplaceAdapter

        return GlobeLaplaceAdapter(
            communication_layers=2,
            theta=0.0 if model_name == "globe_exact" else 1.0,
        )
    if model_name == "geotransolver_matched":
        from external_baselines import GeoTransolverLaplaceAdapter

        return GeoTransolverLaplaceAdapter(
            hidden_dim=72,
            layers=2,
            heads=4,
            slices=16,
        )
    if model_name == "geotransolver_published_scale":
        from external_baselines import GeoTransolverLaplaceAdapter

        return GeoTransolverLaplaceAdapter(
            # This is a parameter-scale control, not a reproduction of the
            # paper architecture: the manually chosen multi-radius local path
            # remains disabled by design. At 29.14M parameters it matches the
            # scale of the reported ~29M large aerodynamic model.
            hidden_dim=360,
            layers=20,
            heads=8,
            slices=32,
        )
    if model_name == "boundary_mean":
        return BoundaryMean()
    raise ValueError(f"unknown model {model_name!r}")


def relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    area_jacobian: torch.Tensor,
) -> torch.Tensor:
    """Squared physical-area relative L2 error used as the training loss."""

    numerator = torch.sum(area_jacobian * (prediction - target).square())
    denominator = torch.sum(area_jacobian * target.square())
    return numerator / denominator


ResolvedObjective = Literal[
    "interior_supervision",
    "boundary_collocation",
    "interior_plus_auxiliary",
]


def _resolved_training_objective(
    model: nn.Module,
    requested: TrainingObjective,
) -> ResolvedObjective:
    """Resolve the explicit objective without guessing from model outputs."""

    has_collocation = callable(getattr(model, "collocation_loss", None))
    if requested == "auto":
        return "boundary_collocation" if has_collocation else "interior_supervision"
    if requested == "boundary_collocation" and not has_collocation:
        raise ValueError(
            "boundary_collocation requires a model exposing collocation_loss(domain)"
        )
    if requested == "interior_plus_auxiliary" and not callable(
        getattr(model, "auxiliary_loss", None)
    ):
        raise ValueError(
            "interior_plus_auxiliary requires a model exposing auxiliary_loss(domain)"
        )
    return requested


def _training_loss(
    model: nn.Module,
    sample: ConformalLaplaceSample,
    objective: ResolvedObjective,
) -> torch.Tensor:
    """Evaluate one declared objective while keeping validation unchanged."""

    if objective == "boundary_collocation":
        return model.collocation_loss(sample.domain)  # type: ignore[attr-defined,no-any-return]
    interior = relative_mse(
        _predict(model, sample), sample.target, sample.area_jacobian
    )
    if objective == "interior_plus_auxiliary":
        # Both terms are dimensionless relative residuals of the same solution
        # object, so equal weighting introduces no tuned physical scale.
        return interior + model.auxiliary_loss(sample.domain)  # type: ignore[attr-defined]
    return interior


def _predict(model: nn.Module, sample: ConformalLaplaceSample) -> torch.Tensor:
    # Targets and evaluation-only coordinates live on the generated interior
    # mesh. Strip them at the benchmark boundary instead of relying on every
    # candidate model to ignore unknown point_data correctly.
    model_domain = DomainMesh(
        interior=sample.domain.interior.with_data(
            point_data={}, cell_data={}, global_data={}
        ),
        boundaries=sample.domain.boundaries,
        global_data=sample.domain.global_data,
    )
    return model(model_domain).point_data["potential"]


def _boundary_data_key(problem: Problem) -> str:
    """Return the sole boundary cell_data key a model may consume."""

    return "boundary_flux" if problem == "neumann" else "boundary_value"


def _domain_with_boundary_values(
    domain: DomainMesh,
    values: torch.Tensor,
    *,
    data_key: str = "boundary_value",
) -> DomainMesh:
    """Replace the sole benchmark drive without changing its geometry."""

    boundary = domain.boundaries["dirichlet"]
    return DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}, global_data={}),
        boundaries={"dirichlet": boundary.with_data(cell_data={data_key: values})},
        global_data=domain.global_data,
    )


def _certified_boundary_range(
    drive: HarmonicDrive,
    *,
    n_samples: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Enclose the continuous trace range using samples and a derivative bound.

    For ``g(theta) = Re(c0 + sum ck exp(i k theta))``, the global Lipschitz
    constant is at most ``sum(k * abs(ck))``. Every angle lies within half a
    grid spacing of a sample, making the returned interval a rigorous
    enclosure rather than an extrema heuristic.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    angles = torch.arange(
        n_samples,
        device=drive.constant.device,
        dtype=drive.constant.dtype,
    )
    spacing = 2.0 * torch.pi / n_samples
    values = evaluate_potential(drive, unit_circle(spacing * angles))
    if drive.modes:
        modes = drive.constant.new_tensor(drive.modes)
        derivative_bound = torch.sum(modes * drive.coefficients.abs())
    else:
        derivative_bound = drive.constant.new_zeros(())
    margin = 0.5 * spacing * derivative_bound
    return values.min() - margin, values.max() + margin


@torch.no_grad()
def _evaluate_split_cases(
    model: nn.Module,
    spec: SplitSpec,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> list[dict[str, float]]:
    """Evaluate and retain metrics for every domain in a split.

    The certified maximum-principle diagnostic requires the exact continuous
    Dirichlet trace range; Neumann samples carry gauge-fixed targets whose
    enclosure would need the sample's private gauge, so that single diagnostic
    is reported only for the Dirichlet problem.  The sampled boundary-range
    diagnostic inside ``case_metrics`` remains valid: Neumann samples store
    gauge-fixed trace samples under ``boundary_value`` for diagnostics only.
    """

    model.eval()
    cases: list[dict[str, float]] = []
    for index in range(n_cases):
        sample = make_case(
            spec,
            seed=seed,
            case_index=index,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        prediction = _predict(model, sample)
        metrics = case_metrics(
            prediction,
            sample.target,
            sample.area_jacobian,
            sample.domain.boundaries["dirichlet"].cell_data["boundary_value"],
            sample.query_preimages.abs(),
        )
        if problem == "dirichlet":
            lower, upper = _certified_boundary_range(sample.drive)
            metrics["certified_maximum_principle_violation"] = float(
                certified_maximum_principle_violation(
                    prediction,
                    lower,
                    upper,
                    sample.drive.boundary_rms,
                )
                .detach()
                .cpu()
            )
        cases.append(metrics)
    return cases


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    spec: SplitSpec,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> dict[str, float]:
    """Aggregate per-domain metrics without weighting by mesh size."""

    return aggregate_metrics(
        _evaluate_split_cases(
            model,
            spec,
            seed=seed,
            n_cases=n_cases,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
    )


@torch.no_grad()
def evaluate_similarity_contract(
    model: nn.Module,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> dict[str, float]:
    """Measure paired scalar predictions under large O(2) similarities."""

    model.eval()
    covariance_errors: list[float] = []
    transformed_cases: list[dict[str, float]] = []
    for index in range(n_cases):
        base = make_case(
            TRAIN_SPLIT,
            seed=seed,
            case_index=index,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        transform = sample_similarity(
            _case_seed(seed, index, 3),
            scale_range=(0.2, 5.0),
            translation_extent=4.0,
            reflection=bool(index % 2),
            device=device,
            dtype=dtype,
        )
        transformed = _build_sample(
            problem,
            base.geometry,
            base.drive,
            n_boundary=n_boundary,
            query_preimages=base.query_preimages,
            similarity=transform,
        )
        base_prediction = _predict(model, base)
        transformed_prediction = _predict(model, transformed)
        covariance_errors.append(
            float(
                torch.sqrt(
                    torch.sum(
                        base.area_jacobian
                        * (transformed_prediction - base_prediction).square()
                    )
                    / torch.sum(base.area_jacobian * base.target.square())
                ).cpu()
            )
        )
        transformed_cases.append(
            case_metrics(
                transformed_prediction,
                transformed.target,
                transformed.area_jacobian,
                transformed.domain.boundaries["dirichlet"].cell_data["boundary_value"],
                transformed.query_preimages.abs(),
            )
        )

    result = aggregate_metrics(transformed_cases)
    values = torch.tensor(covariance_errors, dtype=torch.float64)
    result["paired_covariance_error_mean"] = float(values.mean())
    result["paired_covariance_error_max"] = float(values.max())
    return result


@torch.no_grad()
def evaluate_drive_linearity_contract(
    model: nn.Module,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> dict[str, float]:
    r"""Measure zero preservation and superposition at fixed geometry.

    Geometry, quadrature, and query points are identical across each quartet;
    only the boundary drive (Dirichlet values or Neumann flux) changes.  This
    evaluates the complete model rather than inferring linearity from its
    class or parameterization.
    """

    model.eval()
    superposition_errors: list[float] = []
    zero_errors: list[float] = []
    epsilon = torch.finfo(dtype).eps
    coefficients = (0.731, -1.217)
    data_key = _boundary_data_key(problem)
    for index in range(n_cases):
        first = make_case(
            TRAIN_SPLIT,
            seed=seed,
            case_index=index,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        second_drive = sample_drive(
            _case_seed(seed, index, 41),
            modes=TRAIN_SPLIT.drive_modes,
            regularity=TRAIN_SPLIT.drive_regularity,
            boundary_rms=1.0,
            include_constant=TRAIN_SPLIT.drive_include_constant,
            device=device,
            dtype=dtype,
        )
        second = _build_sample(
            problem,
            first.geometry,
            second_drive,
            n_boundary=n_boundary,
            query_preimages=first.query_preimages,
            similarity=first.similarity,
        )
        first_values = first.domain.boundaries["dirichlet"].cell_data[data_key]
        second_values = second.domain.boundaries["dirichlet"].cell_data[data_key]
        first_prediction = model(
            _domain_with_boundary_values(first.domain, first_values, data_key=data_key)
        ).point_data["potential"]
        second_prediction = model(
            _domain_with_boundary_values(first.domain, second_values, data_key=data_key)
        ).point_data["potential"]
        expected = (
            coefficients[0] * first_prediction + coefficients[1] * second_prediction
        )
        combined_values = (
            coefficients[0] * first_values + coefficients[1] * second_values
        )
        actual = model(
            _domain_with_boundary_values(
                first.domain, combined_values, data_key=data_key
            )
        ).point_data["potential"]
        zero = model(
            _domain_with_boundary_values(
                first.domain, torch.zeros_like(first_values), data_key=data_key
            )
        ).point_data["potential"]
        weights = first.area_jacobian
        expected_norm = torch.sqrt(torch.sum(weights * expected.square()))
        error_norm = torch.sqrt(torch.sum(weights * (actual - expected).square()))
        zero_norm = torch.sqrt(torch.sum(weights * zero.square()) / weights.sum())
        superposition_errors.append(
            float((error_norm / expected_norm.clamp_min(epsilon)).cpu())
        )
        zero_errors.append(float(zero_norm.cpu()))

    superposition = torch.tensor(superposition_errors, dtype=torch.float64)
    zero = torch.tensor(zero_errors, dtype=torch.float64)
    return {
        "superposition_relative_l2_mean": float(superposition.mean()),
        "superposition_relative_l2_max": float(superposition.max()),
        "zero_drive_rms_mean": float(zero.mean()),
        "zero_drive_rms_max": float(zero.max()),
    }


@torch.no_grad()
def evaluate_resolution_study(
    model: nn.Module,
    *,
    seed: int,
    n_cases: int,
    resolutions: tuple[int, ...],
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> dict[str, dict[str, float]]:
    """Hold the continuous problems fixed while changing boundary panels."""

    if tuple(sorted(set(resolutions))) != resolutions:
        raise ValueError("resolutions must be unique and increasing")
    model.eval()
    per_resolution: dict[int, list[float]] = {value: [] for value in resolutions}
    changes: dict[int, list[float]] = {value: [] for value in resolutions}
    successive_changes: dict[int, list[float]] = {
        value: [] for value in resolutions[1:]
    }

    for index in range(n_cases):
        base = make_case(
            TRAIN_SPLIT,
            seed=seed,
            case_index=index,
            n_boundary=resolutions[-1],
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        predictions: dict[int, torch.Tensor] = {}
        for resolution in resolutions:
            sample = _build_sample(
                problem,
                base.geometry,
                base.drive,
                n_boundary=resolution,
                query_preimages=base.query_preimages,
            )
            prediction = _predict(model, sample)
            predictions[resolution] = prediction
            per_resolution[resolution].append(
                float(
                    weighted_relative_l2(
                        prediction, sample.target, sample.area_jacobian
                    ).cpu()
                )
            )
        reference = predictions[resolutions[-1]]
        for resolution in resolutions:
            changes[resolution].append(
                float(
                    torch.sqrt(
                        torch.sum(
                            base.area_jacobian
                            * (predictions[resolution] - reference).square()
                        )
                        / torch.sum(base.area_jacobian * base.target.square())
                    ).cpu()
                )
            )
        for previous, resolution in zip(resolutions, resolutions[1:]):
            successive_changes[resolution].append(
                float(
                    torch.sqrt(
                        torch.sum(
                            base.area_jacobian
                            * (predictions[resolution] - predictions[previous]).square()
                        )
                        / torch.sum(base.area_jacobian * base.target.square())
                    ).cpu()
                )
            )

    result: dict[str, dict[str, float]] = {}
    for resolution in resolutions:
        errors = torch.tensor(per_resolution[resolution], dtype=torch.float64)
        deltas = torch.tensor(changes[resolution], dtype=torch.float64)
        result[str(resolution)] = {
            "relative_l2_mean": float(errors.mean()),
            "relative_l2_median": float(torch.quantile(errors, 0.5)),
            "change_from_finest_mean": float(deltas.mean()),
            "change_from_finest_max": float(deltas.max()),
        }
        if resolution in successive_changes:
            successive = torch.tensor(
                successive_changes[resolution], dtype=torch.float64
            )
            result[str(resolution)].update(
                {
                    "change_from_previous_mean": float(successive.mean()),
                    "change_from_previous_max": float(successive.max()),
                }
            )
    return result


@torch.no_grad()
def evaluate_boundary_trace(
    model: nn.Module,
    spec: SplitSpec,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate one common polygon-panel interior-trace contract.

    Generic models are queried at the input panels' geometric centroids.
    Layer-potential models use the analytic interior jump at those same
    collocation points because their pointwise kernel is singular there. Both
    paths compare against the cell-associated Dirichlet value with the same
    panel measure.
    """

    model.eval()
    errors: list[float] = []
    for index in range(n_cases):
        sample = make_case(
            spec,
            seed=seed,
            case_index=index,
            n_boundary=n_boundary,
            n_query=8,
            device=device,
            dtype=dtype,
        )
        boundary = sample.domain.boundaries["dirichlet"]
        weights = boundary.cell_areas
        target = boundary.cell_data["boundary_value"]
        collocation_residual = getattr(model, "collocation_residual", None)
        if callable(collocation_residual):
            error = collocation_residual(sample.domain)
        else:
            trace_domain = DomainMesh(
                interior=Mesh(points=boundary.cell_centroids),
                boundaries=sample.domain.boundaries,
                global_data=sample.domain.global_data,
            )
            prediction = model(trace_domain).point_data["potential"]
            error = prediction - target
        errors.append(
            float(
                torch.sqrt(
                    torch.sum(weights * error.square())
                    / torch.sum(weights * target.square())
                ).cpu()
            )
        )
    values = torch.tensor(errors, dtype=torch.float64)
    return {
        "relative_l2_mean": float(values.mean()),
        "relative_l2_median": float(torch.quantile(values, 0.5)),
        "relative_l2_max": float(values.max()),
    }


@torch.no_grad()
def evaluate_mode_response(
    model: nn.Module,
    *,
    seed: int,
    modes: tuple[int, ...],
    n_geometries: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    r"""Resolve the learned response to individual harmonic boundary modes.

    These modes are evaluation probes in the private reference coordinate,
    never coordinate features supplied to the model.  Mode zero is the
    constant-solution consistency test; modes one through four are represented
    in training, and higher modes expose spectral extrapolation directly.
    """

    if not modes or tuple(sorted(set(modes))) != modes or modes[0] < 0:
        raise ValueError("modes must be unique, increasing, and nonnegative")
    model.eval()
    result: dict[str, dict[str, float]] = {}
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    for mode in modes:
        errors: list[float] = []
        near_boundary_errors: list[float] = []
        for index in range(n_geometries):
            geometry = sample_geometry(
                _case_seed(seed, index, 0),
                modes=TRAIN_SPLIT.geometry_modes,
                deformation_range=TRAIN_SPLIT.deformation_range,
                device=device,
                dtype=dtype,
            )
            if mode == 0:
                drive = HarmonicDrive(
                    constant=torch.ones((), device=device, dtype=dtype),
                    modes=(),
                    coefficients=torch.empty(0, device=device, dtype=complex_dtype),
                )
            else:
                generator = torch.Generator(device="cpu").manual_seed(
                    _case_seed(seed, index, 10 + mode)
                )
                phase = (
                    2.0
                    * torch.pi
                    * torch.rand((), generator=generator, dtype=torch.float64)
                )
                coefficient = torch.polar(
                    torch.full((), 2.0**0.5, dtype=torch.float64), phase
                ).to(device=device, dtype=complex_dtype)
                drive = HarmonicDrive(
                    constant=torch.zeros((), device=device, dtype=dtype),
                    modes=(mode,),
                    coefficients=coefficient[None],
                )
            sample = build_domain_sample(
                geometry,
                drive,
                n_boundary=n_boundary,
                n_query=n_query,
                query_seed=_case_seed(seed, index, 2),
            )
            prediction = _predict(model, sample)
            errors.append(
                float(
                    weighted_relative_l2(
                        prediction, sample.target, sample.area_jacobian
                    ).cpu()
                )
            )
            near_boundary = sample.query_preimages.abs() >= 0.8
            near_boundary_errors.append(
                float(
                    weighted_relative_l2(
                        prediction[near_boundary],
                        sample.target[near_boundary],
                        sample.area_jacobian[near_boundary],
                    ).cpu()
                )
            )
        values = torch.tensor(errors, dtype=torch.float64)
        near_values = torch.tensor(near_boundary_errors, dtype=torch.float64)
        result[str(mode)] = {
            "relative_l2_mean": float(values.mean()),
            "relative_l2_max": float(values.max()),
            "near_boundary_relative_l2_mean": float(near_values.mean()),
        }
    return result


def evaluate_harmonic_residual(
    model: nn.Module,
    *,
    seed: int,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
    problem: Problem = "dirichlet",
) -> dict[str, float]:
    r"""Measure :math:`L^2` of ``reference_length**2 * Laplacian(u)``.

    This is a diagnostic, not a training loss.  A generic learned operator is
    not expected to be exactly harmonic, but a good Laplace surrogate should
    drive this quantity down along with supervised error.  The diagnostic is
    boundary-condition agnostic and applies to both benchmark problems.
    """

    model.eval()
    residuals: list[float] = []
    for index in range(n_cases):
        sample = make_case(
            TRAIN_SPLIT,
            seed=seed,
            case_index=index,
            n_boundary=n_boundary,
            n_query=n_query,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        query_points = sample.domain.interior.points.detach().clone().requires_grad_()
        interior = Mesh(points=query_points)
        domain = DomainMesh(
            interior=interior,
            boundaries=sample.domain.boundaries,
            global_data=sample.domain.global_data,
        )
        prediction = model(domain).point_data["potential"]
        laplacian = _pointwise_laplacian(prediction, query_points)
        reference_length = sample.domain.global_data["reference_length"]
        scaled_residual = reference_length.square() * laplacian
        residuals.append(
            float(
                torch.sqrt(
                    torch.sum(sample.area_jacobian * scaled_residual.square())
                    / torch.sum(sample.area_jacobian * sample.target.square())
                )
                .detach()
                .cpu()
            )
        )

    values = torch.tensor(residuals, dtype=torch.float64)
    return {
        "normalized_laplacian_l2_mean": float(values.mean()),
        "normalized_laplacian_l2_max": float(values.max()),
    }


def _pointwise_laplacian(
    values: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    r"""Return each output's Laplacian with respect to its own coordinate.

    Summing outputs before differentiating is valid only for a strictly
    pointwise decoder. Computing the diagonal Hessian explicitly keeps this
    diagnostic correct if a future benchmark candidate couples query points.
    The evaluation uses at most a few dozen points, so clarity is preferable
    to a more elaborate batched-Jacobian implementation.
    """

    if values.ndim != 1:
        raise ValueError("values must have shape (N,)")
    if coordinates.ndim != 2 or coordinates.shape[0] != values.shape[0]:
        raise ValueError("coordinates must have shape (N, D)")
    if not values.requires_grad:
        return values.new_zeros(values.shape)

    laplacians: list[torch.Tensor] = []
    for index, value in enumerate(values.unbind()):
        gradient = torch.autograd.grad(
            value,
            coordinates,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            laplacians.append(value.new_zeros(()))
            continue

        diagonal = value.new_zeros(())
        for dimension in range(coordinates.shape[1]):
            component = gradient[index, dimension]
            if not component.requires_grad:
                continue
            second_gradient = torch.autograd.grad(
                component,
                coordinates,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if second_gradient is not None:
                diagonal = diagonal + second_gradient[index, dimension]
        laplacians.append(diagonal)
    return torch.stack(laplacians)


@torch.no_grad()
def _validation_objective(
    model: nn.Module,
    config: RunConfig,
    objective: ResolvedObjective,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, float]:
    """Evaluate a held-out score without broadening the training supervision.

    A boundary-collocation candidate must not use interior labels even through
    checkpoint selection. Interior-supervised candidates retain the common
    deployment relative-L2 validation stream.
    """

    if objective in ("interior_supervision", "interior_plus_auxiliary"):
        validation = evaluate_split(
            model,
            TRAIN_SPLIT,
            seed=config.validation_seed,
            n_cases=config.validation_cases,
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=config.problem,
        )
        return "validation_relative_l2", float(validation["relative_l2_mean"])

    model.eval()
    losses: list[float] = []
    for case_index in range(config.validation_cases):
        sample = make_training_case(
            TRAIN_SPLIT,
            distribution=config.training_drive_distribution,
            seed=config.validation_seed,
            case_index=case_index,
            n_boundary=config.evaluation_boundary_points,
            # The collocation objective consumes only the boundary. Retain one
            # query so the DomainMesh contract remains ordinary without
            # spending validation time on unused interior labels.
            n_query=1,
            device=device,
            dtype=dtype,
            problem=config.problem,
        )
        losses.append(float(model.collocation_loss(sample.domain).cpu()))  # type: ignore[attr-defined]
    return "validation_boundary_collocation_mse", sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    config: RunConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, float | int]], dict[str, float | int] | None]:
    """Train online and restore the best state on a fixed validation stream."""

    if parameter_count(model) == 0:
        return [], None
    objective = _resolved_training_objective(model, config.training_objective)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    validation_name, initial_validation_score = _validation_objective(
        model, config, objective, device=device, dtype=dtype
    )
    best_validation = initial_validation_score
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_record: dict[str, float | int] = {
        "step": 0,
        validation_name: best_validation,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    history.append(initial_record)
    print(json.dumps(initial_record), flush=True)
    model.train()

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for local_case in range(config.cases_per_step):
            case_index = (step - 1) * config.cases_per_step + local_case
            sample = make_training_case(
                TRAIN_SPLIT,
                distribution=config.training_drive_distribution,
                seed=config.seed,
                case_index=case_index,
                n_boundary=config.train_boundary_points,
                n_query=config.train_query_points,
                device=device,
                dtype=dtype,
                problem=config.problem,
            )
            loss = _training_loss(model, sample, objective)
            (loss / config.cases_per_step).backward()
            accumulated += float(loss.detach().cpu())
        optimizer.step()

        should_report = step == 1 or step % config.report_every == 0
        should_validate = step == config.steps or step % config.validation_every == 0
        if should_report or should_validate:
            record: dict[str, float | int] = {
                "step": step,
                "train_objective_value": accumulated / config.cases_per_step,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if should_validate:
                _, validation_score = _validation_objective(
                    model, config, objective, device=device, dtype=dtype
                )
                record[validation_name] = validation_score
                if validation_score < best_validation:
                    best_validation = validation_score
                    best_step = step
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                model.train()
            history.append(record)
            print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    return history, {
        "step": best_step,
        validation_name: best_validation,
    }


def evaluate_model(
    model: nn.Module,
    config: RunConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Run the complete accuracy, OOD, physics, and discretization audit.

    The boundary-trace and per-mode-response audits are Dirichlet-specific:
    they compare interior traces against ``boundary_value`` data and include
    the pure-constant mode, which has identically zero gauge-fixed Neumann
    target.  They are cleanly omitted for the Neumann problem rather than
    reinterpreted.  Every other audit (splits, similarity, drive linearity,
    resolution, harmonic residual) is boundary-condition agnostic.
    """

    evaluation_seed = config.evaluation_seed
    problem = config.problem
    split_cases = {
        name: _evaluate_split_cases(
            model,
            spec,
            seed=evaluation_seed + split_index * 100_000,
            n_cases=config.evaluation_cases,
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=problem,
        )
        for split_index, (name, spec) in enumerate(EVALUATION_SPLITS.items())
    }
    splits = {name: aggregate_metrics(cases) for name, cases in split_cases.items()}
    result: dict[str, object] = {
        "splits": splits,
        "split_cases": split_cases,
        "similarity": evaluate_similarity_contract(
            model,
            seed=evaluation_seed + 2_000_000,
            n_cases=max(1, config.evaluation_cases // 4),
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=problem,
        ),
        "drive_linearity": evaluate_drive_linearity_contract(
            model,
            seed=evaluation_seed + 2_500_000,
            n_cases=max(1, config.evaluation_cases // 4),
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=problem,
        ),
        "resolution": evaluate_resolution_study(
            model,
            seed=evaluation_seed + 3_000_000,
            n_cases=max(1, config.evaluation_cases // 8),
            resolutions=(32, 64, 128, 256),
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=problem,
        ),
    }
    if problem == "dirichlet":
        result["boundary_trace"] = {
            name: evaluate_boundary_trace(
                model,
                EVALUATION_SPLITS[name],
                seed=evaluation_seed + 1_000_000 + index * 100_000,
                n_cases=max(1, config.evaluation_cases // 4),
                n_boundary=config.evaluation_boundary_points,
                device=device,
                dtype=dtype,
            )
            for index, name in enumerate(
                (
                    "interpolation",
                    "unseen_geometry_modes",
                    "unseen_boundary_frequencies",
                )
            )
        }
        result["mode_response"] = evaluate_mode_response(
            model,
            seed=evaluation_seed + 5_000_000,
            modes=tuple(range(13)),
            n_geometries=max(2, config.evaluation_cases // 8),
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
        )
    if config.harmonic_cases:
        result["harmonic_residual"] = evaluate_harmonic_residual(
            model,
            seed=evaluation_seed + 4_000_000,
            n_cases=config.harmonic_cases,
            n_boundary=config.evaluation_boundary_points,
            n_query=min(32, config.evaluation_query_points),
            device=device,
            dtype=dtype,
            problem=problem,
        )
    return result


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}")


def _device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="mesh_transformer",
    )
    parser.add_argument(
        "--capacity", choices=tuple(CAPACITY_CONFIGS), default="reference"
    )
    parser.add_argument(
        "--problem",
        choices=("dirichlet", "neumann"),
        default=RunConfig.problem,
        help=(
            "Boundary-condition problem: Dirichlet trace data or "
            "compatibility-corrected Neumann flux data with gauge-fixed targets"
        ),
    )
    parser.add_argument("--steps", type=int, default=RunConfig.steps)
    parser.add_argument("--cases-per-step", type=int, default=RunConfig.cases_per_step)
    parser.add_argument(
        "--train-boundary-points", type=int, default=RunConfig.train_boundary_points
    )
    parser.add_argument(
        "--train-query-points", type=int, default=RunConfig.train_query_points
    )
    parser.add_argument("--learning-rate", type=float, default=RunConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--report-every", type=int, default=RunConfig.report_every)
    parser.add_argument(
        "--validation-every", type=int, default=RunConfig.validation_every
    )
    parser.add_argument(
        "--validation-cases", type=int, default=RunConfig.validation_cases
    )
    parser.add_argument(
        "--training-drive-distribution",
        choices=(
            "boundary_balanced_mixture",
            "disk_interior_balanced_mixture",
            "uniform_pure_mode",
        ),
        default=RunConfig.training_drive_distribution,
        help="Controlled training-spectrum objective; evaluation is unchanged",
    )
    parser.add_argument(
        "--training-objective",
        choices=(
            "auto",
            "interior_supervision",
            "boundary_collocation",
            "interior_plus_auxiliary",
        ),
        default=RunConfig.training_objective,
        help=(
            "Training loss. Auto selects boundary collocation only for models "
            "that expose that PDE-specific contract"
        ),
    )
    parser.add_argument(
        "--validation-seed", type=int, default=RunConfig.validation_seed
    )
    parser.add_argument(
        "--evaluation-seed", type=int, default=RunConfig.evaluation_seed
    )
    parser.add_argument(
        "--evaluation-cases", type=int, default=RunConfig.evaluation_cases
    )
    parser.add_argument(
        "--evaluation-boundary-points",
        type=int,
        default=RunConfig.evaluation_boundary_points,
    )
    parser.add_argument(
        "--evaluation-query-points",
        type=int,
        default=RunConfig.evaluation_query_points,
    )
    parser.add_argument("--harmonic-cases", type=int, default=RunConfig.harmonic_cases)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=RunConfig.matmul_precision,
        help="Explicit torch float32 matmul precision; recorded in the report",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/laplace_mesh_transformer")
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    """Train or evaluate one declared Laplace-surrogate configuration."""

    args = _parse_args()
    positive_counts = (
        args.cases_per_step,
        args.train_boundary_points,
        args.train_query_points,
        getattr(args, "report_every", RunConfig.report_every),
        getattr(args, "validation_every", RunConfig.validation_every),
        getattr(args, "validation_cases", RunConfig.validation_cases),
        args.evaluation_cases,
        args.evaluation_boundary_points,
        args.evaluation_query_points,
    )
    if args.steps < 0 or any(value < 1 for value in positive_counts):
        raise ValueError("steps must be nonnegative and entity/case counts positive")
    if args.harmonic_cases < 0:
        raise ValueError("harmonic_cases must be nonnegative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    problem = getattr(args, "problem", RunConfig.problem)
    if (args.model in NEUMANN_MODEL_NAMES) != (problem == "neumann"):
        raise ValueError(
            f"model {args.model!r} and problem {problem!r} are mismatched: "
            "Neumann models consume cell_data['boundary_flux'] and must run "
            "with --problem neumann; all other models consume "
            "cell_data['boundary_value'] and must run with --problem dirichlet"
        )

    config = RunConfig(
        model=args.model,
        capacity=args.capacity,
        problem=problem,
        steps=args.steps,
        cases_per_step=args.cases_per_step,
        train_boundary_points=args.train_boundary_points,
        train_query_points=args.train_query_points,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        report_every=getattr(args, "report_every", RunConfig.report_every),
        validation_every=getattr(args, "validation_every", RunConfig.validation_every),
        validation_cases=getattr(args, "validation_cases", RunConfig.validation_cases),
        validation_seed=args.validation_seed,
        evaluation_seed=args.evaluation_seed,
        evaluation_cases=args.evaluation_cases,
        evaluation_boundary_points=args.evaluation_boundary_points,
        evaluation_query_points=args.evaluation_query_points,
        harmonic_cases=args.harmonic_cases,
        matmul_precision=args.matmul_precision,
        training_drive_distribution=getattr(
            args,
            "training_drive_distribution",
            RunConfig.training_drive_distribution,
        ),
        training_objective=getattr(
            args,
            "training_objective",
            RunConfig.training_objective,
        ),
    )
    torch.set_float32_matmul_precision(config.matmul_precision)
    model = make_model(config.model, config.capacity).to(device=device, dtype=dtype)
    if args.evaluate_only and args.checkpoint is None and parameter_count(model) > 0:
        raise ValueError("--evaluate-only requires --checkpoint for a learned model")
    input_checkpoint_metadata: dict[str, object] | None = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if (
            checkpoint["model"] != config.model
            or checkpoint["capacity"] != config.capacity
        ):
            raise ValueError("checkpoint model/capacity does not match the command")
        model.load_state_dict(checkpoint["state_dict"])
        input_checkpoint_metadata = {
            "model": checkpoint["model"],
            "capacity": checkpoint["capacity"],
            "run_config": checkpoint.get("run_config"),
            "source": checkpoint.get("source"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.evaluate_only:
        history: list[dict[str, float | int]] = []
        selected_validation: dict[str, float | int] | None = None
    else:
        history, selected_validation = train_model(
            model, config, device=device, dtype=dtype
        )
    evaluation = evaluate_model(model, config, device=device, dtype=dtype)
    elapsed = time.perf_counter() - started
    report_source = source_provenance()
    if input_checkpoint_metadata is not None:
        checkpoint_source = input_checkpoint_metadata["source"]
        checkpoint_digest = (
            checkpoint_source.get("relevant_source_sha256")
            if isinstance(checkpoint_source, dict)
            else None
        )
        evaluator_digest = report_source["relevant_source_sha256"]
        input_checkpoint_metadata["source_matches_evaluator"] = (
            None if checkpoint_digest is None else checkpoint_digest == evaluator_digest
        )

    checkpoint_path: Path | None = None
    if not args.evaluate_only:
        checkpoint_path = args.output_dir / f"{config.model}_{config.capacity}.pt"
        torch.save(
            {
                "model": config.model,
                "capacity": config.capacity,
                "state_dict": model.state_dict(),
                "run_config": asdict(config),
                "source": report_source,
            },
            checkpoint_path,
        )
    report = {
        "run_config": asdict(config),
        "dtype": args.dtype,
        "device": _device_description(device),
        "environment": runtime_environment(device),
        "source": report_source,
        "parameters": parameter_count(model),
        "elapsed_seconds": elapsed,
        "history": history,
        "selected_validation": selected_validation,
        "evaluation": evaluation,
        "input_checkpoint": (
            None if args.checkpoint is None else str(args.checkpoint.resolve())
        ),
        "input_checkpoint_metadata": input_checkpoint_metadata,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    }
    report_path = args.output_dir / f"{config.model}_{config.capacity}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
