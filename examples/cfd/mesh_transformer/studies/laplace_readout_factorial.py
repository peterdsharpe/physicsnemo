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

r"""Factorial test of the two geometry-conditioned scalar readout paths.

The mature two-dimensional Laplace ``mesh_transformer_kernel_singonly`` arm
forms its final scalar prediction from a scalar projection plus a
vector--geometry dot-product projection, then multiplies the sum by a scalar
geometry gate. This study asks which of the latter two paths matters by
crossing them in a 2 x 2 factorial design.

Every arm constructs the complete reference model under the same seed. Only
after construction are disabled paths set to their exact neutral values and
frozen: a zero gate logit gives ``2 * sigmoid(0) == 1``, while a zero
``scalar_from_vector_dots`` map contributes exactly zero. Thus construction,
RNG consumption, module layout, and all shared initial parameters are
identical.

One invocation runs one arm and one seed. The registered experiment uses
seeds 17, 29, 43, 59, and 71; 3,000 online updates; the existing fixed
validation/evaluation streams; all five geometry/frequency splits; the
normalized Laplacian residual; and boundary-resolution transfer at
64/128/256 panels.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from metrics import aggregate_metrics
from models import parameter_count
from provenance import runtime_environment, source_provenance
from torch import nn
from train import (
    EVALUATION_SPLITS,
    RunConfig,
    _evaluate_split_cases,
    evaluate_harmonic_residual,
    evaluate_resolution_study,
    make_model,
    train_model,
)

MODEL = "mesh_transformer_kernel_singonly"
CAPACITY = "reference"
GAUGE = "explicit"
STEPS = 3_000
SEEDS = (17, 29, 43, 59, 71)
RESOLUTIONS = (64, 128, 256)
ACCURACY_DTYPE = torch.float32
RESIDUAL_DTYPE = torch.float64
SPLITS = (
    "interpolation",
    "mixed_geometry_modes",
    "stronger_deformation",
    "unseen_boundary_frequencies",
    "unseen_geometry_modes",
)
_SPLIT_SEED_OFFSETS = {
    "interpolation": 0,
    "unseen_geometry_modes": 100_000,
    "stronger_deformation": 200_000,
    "mixed_geometry_modes": 300_000,
    "unseen_boundary_frequencies": 400_000,
}


@dataclass(frozen=True)
class Arm:
    """One cell in the gate x vector-dot factorial."""

    key: str
    scalar_gate: bool
    scalar_from_vector_dots: bool


ARMS: dict[str, Arm] = {
    arm.key: arm
    for arm in (
        Arm("pure", scalar_gate=False, scalar_from_vector_dots=False),
        Arm("gate_only", scalar_gate=True, scalar_from_vector_dots=False),
        Arm("contraction_only", scalar_gate=False, scalar_from_vector_dots=True),
        Arm("full", scalar_gate=True, scalar_from_vector_dots=True),
    )
}

INTERVENED_PARAMETERS = frozenset(
    {
        "output_projection.scalar_gate.weight",
        "output_projection.scalar_gate.bias",
        "output_projection.scalar_from_vector_dots.weight",
    }
)


def experiment_config(seed: int) -> RunConfig:
    """Return the fixed mature Laplace protocol for one registered seed."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")
    return RunConfig(
        model=MODEL,
        capacity=CAPACITY,
        problem="dirichlet",
        gauge=GAUGE,
        steps=STEPS,
        cases_per_step=1,
        train_boundary_points=64,
        train_query_points=128,
        learning_rate=3.0e-4,
        weight_decay=1.0e-6,
        seed=seed,
        validation_seed=71_000_011,
        evaluation_seed=97_000_037,
        report_every=50,
        validation_every=250,
        validation_cases=8,
        evaluation_cases=64,
        evaluation_boundary_points=128,
        evaluation_query_points=512,
        harmonic_cases=4,
        matmul_precision="highest",
        training_drive_distribution="boundary_balanced_mixture",
        training_objective="interior_supervision",
    )


def _readout_modules(model: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    projection = getattr(model, "output_projection", None)
    scalar_gate = getattr(projection, "scalar_gate", None)
    vector_dots = getattr(projection, "scalar_from_vector_dots", None)
    if not isinstance(scalar_gate, nn.Linear) or not isinstance(vector_dots, nn.Linear):
        raise TypeError(
            f"{MODEL} no longer exposes the two registered scalar readout paths"
        )
    return scalar_gate, vector_dots


def _zero_and_freeze(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
            parameter.requires_grad_(False)


def apply_arm(model: nn.Module, arm: Arm) -> nn.Module:
    """Apply only the two post-construction readout interventions."""

    scalar_gate, vector_dots = _readout_modules(model)
    if not arm.scalar_gate:
        _zero_and_freeze(scalar_gate)
    if not arm.scalar_from_vector_dots:
        _zero_and_freeze(vector_dots)
    return model


def build_arm_model(arm_key: str, seed: int) -> nn.Module:
    """Construct the full registered model, then apply one factorial arm."""

    try:
        arm = ARMS[arm_key]
    except KeyError:
        raise ValueError(
            f"unknown arm {arm_key!r}; choose from {tuple(ARMS)}"
        ) from None
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(MODEL, CAPACITY, GAUGE)
    return apply_arm(model, arm)


def evaluate_readout_factorial(
    model: nn.Module,
    config: RunConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    resolutions: tuple[int, ...] = RESOLUTIONS,
) -> dict[str, object]:
    """Run only the pre-registered accuracy, physics, and transfer checks."""

    if dtype != ACCURACY_DTYPE:
        raise ValueError("the registered accuracy evaluation must use float32")
    if set(_SPLIT_SEED_OFFSETS) != set(EVALUATION_SPLITS):
        raise RuntimeError(
            "the frozen split-to-seed map no longer matches EVALUATION_SPLITS"
        )

    split_cases = {
        name: _evaluate_split_cases(
            model,
            EVALUATION_SPLITS[name],
            seed=config.evaluation_seed + _SPLIT_SEED_OFFSETS[name],
            n_cases=config.evaluation_cases,
            n_boundary=config.evaluation_boundary_points,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=config.problem,
        )
        for name in SPLITS
    }
    splits = {name: aggregate_metrics(cases) for name, cases in split_cases.items()}
    result: dict[str, object] = {
        "accuracy_dtype": "float32",
        "splits": splits,
        "split_cases": split_cases,
        "resolution": evaluate_resolution_study(
            model,
            seed=config.evaluation_seed + 3_000_000,
            n_cases=max(1, config.evaluation_cases // 8),
            resolutions=resolutions,
            n_query=config.evaluation_query_points,
            device=device,
            dtype=dtype,
            problem=config.problem,
        ),
    }
    if config.harmonic_cases:
        residual_model = copy.deepcopy(model).to(device=device, dtype=RESIDUAL_DTYPE)
        result["harmonic_residual"] = {
            "dtype": "float64",
            **evaluate_harmonic_residual(
                residual_model,
                seed=config.evaluation_seed + 4_000_000,
                n_cases=config.harmonic_cases,
                n_boundary=config.evaluation_boundary_points,
                n_query=min(32, config.evaluation_query_points),
                device=device,
                dtype=RESIDUAL_DTYPE,
                problem=config.problem,
            ),
        }
    return result


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with one complete, finite JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_experiment(
    *,
    arm_key: str,
    seed: int,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    """Train, evaluate, and publish one arm/seed report."""

    config = experiment_config(seed)
    torch.set_float32_matmul_precision(config.matmul_precision)
    model = build_arm_model(arm_key, seed).to(device=device, dtype=ACCURACY_DTYPE)

    started = time.perf_counter()
    history, selected_validation = train_model(
        model, config, device=device, dtype=ACCURACY_DTYPE
    )
    evaluation = evaluate_readout_factorial(
        model, config, device=device, dtype=ACCURACY_DTYPE
    )
    elapsed = time.perf_counter() - started

    report: dict[str, object] = {
        "study": "laplace_readout_factorial_v1",
        "arm": asdict(ARMS[arm_key]),
        "run_config": asdict(config),
        "evaluation_protocol": {
            "splits": list(SPLITS),
            "split_seed_offsets": {name: _SPLIT_SEED_OFFSETS[name] for name in SPLITS},
            "resolutions": list(RESOLUTIONS),
            "cases_per_split": config.evaluation_cases,
            "harmonic_cases": config.harmonic_cases,
        },
        "accuracy_dtype": "float32",
        "residual_dtype": "float64",
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "registered_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": parameter_count(model),
        "elapsed_seconds": elapsed,
        "history": history,
        "selected_validation": selected_validation,
        "evaluation": evaluation,
    }
    atomic_write_json(output, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run one process-isolated factorial cell."""

    args = _parse_args()
    output = args.output.expanduser().resolve()
    run_experiment(
        arm_key=args.arm,
        seed=args.seed,
        device=torch.device(args.device),
        output=output,
    )
    print(
        json.dumps(
            {"arm": args.arm, "seed": args.seed, "output": str(output)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
