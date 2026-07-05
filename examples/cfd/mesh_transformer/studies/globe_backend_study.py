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

r"""Measure GLOBE's hierarchical approximation with one fixed parameter state.

This diagnostic deliberately separates two questions that independent
training runs cannot separate:

1. How accurate is an exact, learned GLOBE operator on the PDE benchmark?
2. How much does the dual-tree approximation change that *same* operator?

The checkpoint must therefore be an exact ``globe_exact`` training checkpoint.
It is loaded once, and only the Barnes--Hut opening criterion ``theta`` changes
between evaluations.  ``theta`` is a geometric acceptance criterion, not a
certified error tolerance; all approximation errors are measured empirically
against the ``theta=0`` prediction on identical cases.

Example
-------

.. code-block:: bash

   python examples/cfd/mesh_transformer/studies/globe_backend_study.py \
       --checkpoint outputs/globe_exact_reference.pt \
       --device cuda --evaluation-cases 8 \
       --output outputs/globe_backend_sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from conformal_laplace import ConformalLaplaceSample
from metrics import aggregate_metrics, weighted_relative_l2
from provenance import runtime_environment, source_provenance
from torch import nn
from train import (
    EVALUATION_SPLITS,
    RunConfig,
    _predict,
    make_case,
    make_model,
    parameter_count,
)

DEFAULT_THETAS = (0.0, 0.25, 0.5, 1.0)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype_name must be float32 or float64")


def _canonical_thetas(thetas: Sequence[float]) -> tuple[float, ...]:
    if not thetas:
        raise ValueError("thetas must be nonempty")
    converted = tuple(float(theta) for theta in thetas)
    if any(not math.isfinite(theta) or theta < 0.0 for theta in converted):
        raise ValueError("thetas must be finite and nonnegative")
    if len(set(converted)) != len(converted):
        raise ValueError("thetas must be unique")
    if 0.0 not in converted:
        raise ValueError("thetas must include the exact theta=0 baseline")
    return (0.0, *(theta for theta in converted if theta != 0.0))


def _canonical_splits(splits: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(splits) if splits else tuple(EVALUATION_SPLITS)
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation splits must be unique")
    unknown = sorted(set(selected) - EVALUATION_SPLITS.keys())
    if unknown:
        raise ValueError(f"unknown evaluation split(s): {', '.join(unknown)}")
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: nn.Module) -> str:
    """Hash tensor values and metadata without depending on serialization."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.flatten().view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _globe_core(model: nn.Module) -> nn.Module:
    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "theta"):
        raise TypeError("globe_exact must construct an adapter exposing model.theta")
    return core


def load_exact_globe_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    allow_source_mismatch: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load one exact GLOBE checkpoint and audit its source fingerprint."""

    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    try:
        model_name = checkpoint["model"]
        capacity = checkpoint["capacity"]
        state_dict = checkpoint["state_dict"]
    except KeyError as error:
        raise ValueError(
            f"checkpoint is missing required key {error.args[0]!r}"
        ) from error
    if model_name != "globe_exact":
        raise ValueError(
            "backend sweep requires an exact-trained 'globe_exact' checkpoint"
        )
    if not isinstance(capacity, str) or not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint capacity/state_dict metadata is malformed")

    evaluator_source = source_provenance()
    checkpoint_source = checkpoint.get("source")
    checkpoint_digest = (
        checkpoint_source.get("relevant_source_sha256")
        if isinstance(checkpoint_source, Mapping)
        else None
    )
    evaluator_digest = evaluator_source["relevant_source_sha256"]
    source_matches = (
        None if checkpoint_digest is None else checkpoint_digest == evaluator_digest
    )
    if source_matches is False and not allow_source_mismatch:
        raise ValueError(
            "checkpoint source fingerprint differs from the backend evaluator; "
            "pass allow_source_mismatch=True only for an explicitly historical audit"
        )

    run_config = checkpoint.get("run_config")
    checkpoint_seed = (
        run_config.get("seed", 0) if isinstance(run_config, Mapping) else 0
    )
    if isinstance(checkpoint_seed, bool) or not isinstance(checkpoint_seed, int):
        raise ValueError("checkpoint run_config.seed must be an integer")
    torch.manual_seed(checkpoint_seed)
    model = make_model("globe_exact", capacity).to(device=device, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    core = _globe_core(model)
    if float(core.theta) != 0.0:  # type: ignore[attr-defined]
        raise RuntimeError("globe_exact builder did not select exact theta=0")

    metadata = {
        "path": str(checkpoint_path),
        "file_sha256": _file_sha256(checkpoint_path),
        "model": model_name,
        "capacity": capacity,
        "run_config": run_config,
        "checkpoint_source": checkpoint_source,
        "evaluator_source": evaluator_source,
        "source_matches_evaluator": source_matches,
        "source_mismatch_allowed": allow_source_mismatch,
    }
    return model, metadata


def build_evaluation_bank(
    *,
    splits: Sequence[str],
    evaluation_seed: int,
    evaluation_cases: int,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, tuple[ConformalLaplaceSample, ...]]:
    """Build the fixed cases reused without regeneration for every theta."""

    selected = _canonical_splits(splits)
    _positive_integer("evaluation_cases", evaluation_cases)
    _positive_integer("n_boundary", n_boundary)
    _positive_integer("n_query", n_query)
    if n_boundary < 3:
        raise ValueError("n_boundary must be at least three")
    split_indices = {name: index for index, name in enumerate(EVALUATION_SPLITS)}
    return {
        name: tuple(
            make_case(
                EVALUATION_SPLITS[name],
                seed=evaluation_seed + split_indices[name] * 100_000,
                case_index=case_index,
                n_boundary=n_boundary,
                n_query=n_query,
                device=device,
                dtype=dtype,
            )
            for case_index in range(evaluation_cases)
        )
        for name in selected
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_up(
    model: nn.Module,
    bank: Mapping[str, Sequence[ConformalLaplaceSample]],
    *,
    warmup_cases: int,
    device: torch.device,
) -> None:
    if warmup_cases == 0:
        return
    ordered_cases = [sample for cases in bank.values() for sample in cases]
    with torch.no_grad():
        for index in range(warmup_cases):
            _predict(model, ordered_cases[index % len(ordered_cases)])
    _synchronize(device)


def _prediction_delta(
    prediction: torch.Tensor,
    exact_prediction: torch.Tensor,
    target: torch.Tensor,
    area_jacobian: torch.Tensor,
) -> dict[str, float]:
    """Return physical-area-weighted backend deltas with explicit scales."""

    difference_energy = torch.sum(
        area_jacobian * (prediction - exact_prediction).square()
    )
    target_energy = torch.sum(area_jacobian * target.square())
    measure = area_jacobian.sum()
    if not torch.compiler.is_compiling() and (
        target_energy.item() <= 0.0 or measure.item() <= 0.0
    ):
        raise ValueError("backend-delta normalization requires nonzero target/measure")
    return {
        "physical_area_weighted_rms": float(
            torch.sqrt(difference_energy / measure).cpu()
        ),
        "target_normalized_weighted_l2": float(
            torch.sqrt(difference_energy / target_energy).cpu()
        ),
    }


def _evaluate_theta(
    model: nn.Module,
    bank: Mapping[str, Sequence[ConformalLaplaceSample]],
    *,
    theta: float,
    exact_predictions: Mapping[str, Sequence[torch.Tensor]] | None,
    warmup_cases: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, ...]]]:
    core = _globe_core(model)
    core.theta = theta  # type: ignore[attr-defined]
    _warm_up(model, bank, warmup_cases=warmup_cases, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        _synchronize(device)
        baseline_allocation = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocation = None

    split_results: dict[str, Any] = {}
    predictions_by_split: dict[str, tuple[torch.Tensor, ...]] = {}
    total_forward_seconds = 0.0
    with torch.no_grad():
        for split_name, cases in bank.items():
            _synchronize(device)
            started = time.perf_counter()
            predictions = tuple(_predict(model, sample) for sample in cases)
            _synchronize(device)
            forward_seconds = time.perf_counter() - started
            total_forward_seconds += forward_seconds
            predictions_by_split[split_name] = predictions

            target_cases = [
                {
                    "weighted_relative_l2": float(
                        weighted_relative_l2(
                            prediction, sample.target, sample.area_jacobian
                        ).cpu()
                    )
                }
                for prediction, sample in zip(predictions, cases, strict=True)
            ]
            references = (
                predictions
                if exact_predictions is None
                else exact_predictions[split_name]
            )
            delta_cases = [
                _prediction_delta(
                    prediction,
                    exact_prediction,
                    sample.target,
                    sample.area_jacobian,
                )
                for prediction, exact_prediction, sample in zip(
                    predictions, references, cases, strict=True
                )
            ]
            split_results[split_name] = {
                "cases": len(cases),
                "target_error": aggregate_metrics(target_cases),
                "prediction_delta_to_theta_zero": aggregate_metrics(delta_cases),
                "forward_elapsed_synchronized_seconds": forward_seconds,
            }

    if device.type == "cuda":
        _synchronize(device)
        peak_allocation = torch.cuda.max_memory_allocated(device)
        memory = {
            "method": "torch_cuda_allocator",
            "baseline_allocated_bytes": baseline_allocation,
            "peak_allocated_bytes": peak_allocation,
            "peak_increment_bytes": max(0, peak_allocation - baseline_allocation),
        }
    else:
        memory = {
            "method": "not_measured_on_cpu",
            "baseline_allocated_bytes": None,
            "peak_allocated_bytes": None,
            "peak_increment_bytes": None,
        }
    return (
        {
            "theta": theta,
            "splits": split_results,
            "forward_elapsed_synchronized_seconds": total_forward_seconds,
            "cuda_peak_allocation": memory,
        },
        predictions_by_split,
    )


def run_globe_backend_study(
    checkpoint_path: Path,
    *,
    device_name: str = "cpu",
    dtype_name: str = "float32",
    thetas: Sequence[float] = DEFAULT_THETAS,
    splits: Sequence[str] = (),
    evaluation_seed: int = RunConfig.evaluation_seed,
    evaluation_cases: int = 8,
    n_boundary: int = RunConfig.evaluation_boundary_points,
    n_query: int = RunConfig.evaluation_query_points,
    warmup_cases: int = 1,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Run a paired theta sweep and return its self-describing JSON payload."""

    canonical_thetas = _canonical_thetas(thetas)
    canonical_splits = _canonical_splits(splits)
    _positive_integer("evaluation_cases", evaluation_cases)
    _positive_integer("n_boundary", n_boundary)
    _positive_integer("n_query", n_query)
    _nonnegative_integer("warmup_cases", warmup_cases)
    dtype = _dtype(dtype_name)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")

    study_started = time.perf_counter()
    model, checkpoint_metadata = load_exact_globe_checkpoint(
        checkpoint_path,
        device=device,
        dtype=dtype,
        allow_source_mismatch=allow_source_mismatch,
    )
    bank = build_evaluation_bank(
        splits=canonical_splits,
        evaluation_seed=evaluation_seed,
        evaluation_cases=evaluation_cases,
        n_boundary=n_boundary,
        n_query=n_query,
        device=device,
        dtype=dtype,
    )
    state_digest = _state_sha256(model)
    exact_predictions: dict[str, tuple[torch.Tensor, ...]] | None = None
    theta_results: dict[str, Any] = {}
    core = _globe_core(model)
    original_theta = float(core.theta)  # type: ignore[attr-defined]
    try:
        for theta in canonical_thetas:
            result, predictions = _evaluate_theta(
                model,
                bank,
                theta=theta,
                exact_predictions=exact_predictions,
                warmup_cases=warmup_cases,
                device=device,
            )
            if theta == 0.0:
                exact_predictions = predictions
            theta_results[format(theta, ".12g")] = result
    finally:
        core.theta = original_theta  # type: ignore[attr-defined]
    final_state_digest = _state_sha256(model)
    if final_state_digest != state_digest:
        raise RuntimeError(
            "model state changed during the inference-only backend sweep"
        )

    return {
        "schema_version": 1,
        "study": "same-checkpoint GLOBE hierarchical-backend sweep",
        "interpretation": {
            "parameter_state": (
                "one exact-trained checkpoint is loaded once and reused for every theta"
            ),
            "theta": (
                "Barnes--Hut geometric opening criterion; it is not a certified "
                "error tolerance"
            ),
            "reference": (
                "theta=0 exact pair summation on the identical fixed evaluation cases"
            ),
        },
        "checkpoint": checkpoint_metadata,
        "model_state_sha256": state_digest,
        "parameters": parameter_count(model),
        "device": str(device),
        "dtype": dtype_name,
        "environment": runtime_environment(device),
        "evaluation": {
            "seed": evaluation_seed,
            "cases_per_split": evaluation_cases,
            "boundary_points": n_boundary,
            "query_points": n_query,
            "splits": list(canonical_splits),
            "split_seeds": {
                name: evaluation_seed + tuple(EVALUATION_SPLITS).index(name) * 100_000
                for name in canonical_splits
            },
            "warmup_cases_per_theta": warmup_cases,
            "thetas": list(canonical_thetas),
        },
        "theta_results": theta_results,
        "total_wall_seconds": time.perf_counter() - study_started,
    }


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result:
        raise argparse.ArgumentTypeError("list must be nonempty")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--thetas", type=_parse_csv_floats, default=DEFAULT_THETAS)
    parser.add_argument(
        "--split",
        action="append",
        choices=tuple(EVALUATION_SPLITS),
        default=[],
        dest="splits",
        help="evaluation split; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--evaluation-seed", type=int, default=RunConfig.evaluation_seed
    )
    parser.add_argument("--evaluation-cases", type=int, default=8)
    parser.add_argument(
        "--boundary-points", type=int, default=RunConfig.evaluation_boundary_points
    )
    parser.add_argument(
        "--query-points", type=int, default=RunConfig.evaluation_query_points
    )
    parser.add_argument("--warmup-cases", type=int, default=1)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Evaluate one exact-trained GLOBE checkpoint across opening angles."""

    args = _build_parser().parse_args()
    report = run_globe_backend_study(
        args.checkpoint,
        device_name=args.device,
        dtype_name=args.dtype,
        thetas=args.thetas,
        splits=args.splits,
        evaluation_seed=args.evaluation_seed,
        evaluation_cases=args.evaluation_cases,
        n_boundary=args.boundary_points,
        n_query=args.query_points,
        warmup_cases=args.warmup_cases,
        allow_source_mismatch=args.allow_source_mismatch,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(args.output)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_THETAS",
    "build_evaluation_bank",
    "load_exact_globe_checkpoint",
    "run_globe_backend_study",
]
