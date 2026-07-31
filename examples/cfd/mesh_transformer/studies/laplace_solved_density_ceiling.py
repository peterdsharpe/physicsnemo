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

r"""Measure the fixed-carrier solved-density ceiling for the Laplace study.

The control replaces learned boundary-density inference with the dense
second-kind collocation solve, while retaining the same 128 straight panels,
analytic panel propagator, and frozen case/query banks as the scalar-readout
factorial. Field errors are evaluated in float32 for exact pairing with that
experiment. The discrete boundary trace is evaluated separately in float64.
The 64/128/256 ladder holds each continuum problem and query bank fixed while
changing the boundary discretization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from laplace_readout_factorial import (
    _SPLIT_SEED_OFFSETS,
    ACCURACY_DTYPE,
    RESOLUTIONS,
    SEEDS,
    SPLITS,
    atomic_write_json,
    experiment_config,
)
from layer_potential import SolvedDoubleLayerPotential
from metrics import aggregate_metrics
from provenance import runtime_environment, source_provenance
from train import (
    EVALUATION_SPLITS,
    RunConfig,
    _evaluate_split_cases,
    evaluate_boundary_trace,
    evaluate_resolution_study,
)

STUDY = "laplace_solved_density_ceiling_v1"
SOURCE_STUDY = "laplace_readout_factorial_v1"
REFERENCE_SEED = SEEDS[0]
BOUNDARY_TRACE_DTYPE = torch.float64


def evaluation_config() -> RunConfig:
    """Return the frozen factorial configuration that defines the case banks."""

    return experiment_config(REFERENCE_SEED)


def build_model() -> SolvedDoubleLayerPotential:
    """Construct the parameter-free dense collocation control."""

    return SolvedDoubleLayerPotential()


def evaluation_protocol(
    config: RunConfig,
    *,
    resolutions: tuple[int, ...] = RESOLUTIONS,
) -> dict[str, object]:
    """Describe every datum needed to reconstruct the frozen evaluation."""

    return {
        "source_study": SOURCE_STUDY,
        "problem": config.problem,
        "evaluation_seed": config.evaluation_seed,
        "splits": list(SPLITS),
        "split_seed_offsets": {name: _SPLIT_SEED_OFFSETS[name] for name in SPLITS},
        "cases_per_split": config.evaluation_cases,
        "fixed_boundary_points": config.evaluation_boundary_points,
        "query_points_per_case": config.evaluation_query_points,
        "accuracy_dtype": "float32",
        "boundary_trace_dtype": "float64",
        "boundary_trace_cases_per_split": config.evaluation_cases,
        "resolution_seed": config.evaluation_seed + 3_000_000,
        "resolution_cases": max(1, config.evaluation_cases // 8),
        "resolution_query_points": config.evaluation_query_points,
        "resolutions": list(resolutions),
    }


def evaluate_solved_density_ceiling(
    model: SolvedDoubleLayerPotential,
    config: RunConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    resolutions: tuple[int, ...] = RESOLUTIONS,
) -> dict[str, object]:
    """Evaluate fixed-N field error, trace fidelity, and panel refinement."""

    if dtype != ACCURACY_DTYPE:
        raise ValueError("the paired field evaluation must use float32")
    if config.problem != "dirichlet":
        raise ValueError("the solved-density ceiling is defined for Dirichlet data")
    if set(_SPLIT_SEED_OFFSETS) != set(EVALUATION_SPLITS):
        raise RuntimeError(
            "the frozen split-to-seed map no longer matches EVALUATION_SPLITS"
        )

    model = model.to(device=device, dtype=dtype)
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

    trace_model = build_model().to(device=device, dtype=BOUNDARY_TRACE_DTYPE)
    boundary_trace = {
        name: evaluate_boundary_trace(
            trace_model,
            EVALUATION_SPLITS[name],
            seed=config.evaluation_seed + _SPLIT_SEED_OFFSETS[name],
            n_cases=config.evaluation_cases,
            n_boundary=config.evaluation_boundary_points,
            device=device,
            dtype=BOUNDARY_TRACE_DTYPE,
        )
        for name in SPLITS
    }

    return {
        "accuracy_dtype": "float32",
        "splits": {
            name: aggregate_metrics(cases) for name, cases in split_cases.items()
        },
        "split_cases": split_cases,
        "boundary_trace": {
            "dtype": "float64",
            "splits": boundary_trace,
        },
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


def run_study(
    *,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    """Evaluate and atomically publish the single parameter-free report."""

    config = evaluation_config()
    torch.set_float32_matmul_precision(config.matmul_precision)
    model = build_model()

    started = time.perf_counter()
    evaluation = evaluate_solved_density_ceiling(
        model,
        config,
        device=device,
        dtype=ACCURACY_DTYPE,
    )
    elapsed = time.perf_counter() - started

    report: dict[str, object] = {
        "study": STUDY,
        "control": {
            "model": "double_layer_solved",
            "density": "dense_second_kind_collocation",
            "carrier": "piecewise_constant_straight_panels",
            "propagator": "analytic_double_layer_panel_integral",
        },
        "evaluation_protocol": evaluation_protocol(config),
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "registered_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
    }
    atomic_write_json(output, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the registered single-control study."""

    args = _parse_args()
    output = args.output.expanduser().resolve()
    report = run_study(device=torch.device(args.device), output=output)
    print(
        json.dumps(
            {
                "study": report["study"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
