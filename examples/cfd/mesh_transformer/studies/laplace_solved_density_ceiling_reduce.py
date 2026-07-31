# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the preregistered Laplace solved-density ceiling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from laplace_readout_factorial import SEEDS, SPLITS, atomic_write_json
from laplace_solved_density_ceiling import (
    STUDY,
    evaluation_config,
    evaluation_protocol,
)

REDUCTION_STUDY = "laplace_solved_density_ceiling_reduction_v1"
BASELINE_STUDY = "laplace_readout_factorial_v1"
BASELINE_PROTOCOL_FINGERPRINT = (
    "fefa0ea2f7ece6f34089460466d1564e2cb0351822526cc9084c41579d6e52da"
)
BASELINE_SOURCE_FINGERPRINT = (
    "36a0e7406a9d7e432a458cdff61a17a8fcd34d81f4b6819041795e406e4cab2f"
)
MIXED_GEOMETRY_SPLIT = "mixed_geometry_modes"
PRIMARY_SPLITS = (
    "interpolation",
    "stronger_deformation",
    "unseen_boundary_frequencies",
    "unseen_geometry_modes",
)
DENSITY_DECISION_SPLITS = (
    "interpolation",
    "stronger_deformation",
    "unseen_geometry_modes",
)
DENSITY_RATIO_CEILING = 0.50
TRACE_ERROR_CEILING = 1.0e-10
DISCRETIZATION_RATIO_FLOOR = 0.80


def _require_finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("a geometric mean requires at least one value")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric-mean inputs must be finite and nonnegative")
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _validate_baseline(summary: dict[str, Any]) -> None:
    if summary.get("study") != BASELINE_STUDY:
        raise ValueError("wrong factorial baseline study")
    if summary.get("registered_seeds") != list(SEEDS):
        raise ValueError("factorial baseline does not contain the registered seeds")
    if set(summary.get("registered_arms", {})) != {
        "pure",
        "gate_only",
        "contraction_only",
        "full",
    }:
        raise ValueError("factorial baseline does not contain the registered arms")
    if summary.get("protocol_fingerprint") != BASELINE_PROTOCOL_FINGERPRINT:
        raise ValueError("factorial protocol fingerprint mismatch")
    if summary.get("source_fingerprint") != BASELINE_SOURCE_FINGERPRINT:
        raise ValueError("factorial source fingerprint mismatch")
    if set(summary.get("splits", {})) != set(SPLITS):
        raise ValueError("factorial baseline split set mismatch")


def _validate_oracle(report: dict[str, Any]) -> None:
    if report.get("study") != STUDY:
        raise ValueError("wrong solved-density study")
    if report.get("evaluation_protocol") != evaluation_protocol(evaluation_config()):
        raise ValueError("solved-density evaluation protocol mismatch")
    if report.get("registered_parameters") != 0:
        raise ValueError("solved-density control must have zero registered parameters")
    if report.get("trainable_parameters") != 0:
        raise ValueError("solved-density control must have zero trainable parameters")
    if report.get("environment", {}).get("device") != "cuda":
        raise ValueError("the paired case bank must be generated on CUDA")

    evaluation = report.get("evaluation", {})
    if evaluation.get("accuracy_dtype") != "float32":
        raise ValueError("solved-density field evaluation must use float32")
    if set(evaluation.get("splits", {})) != set(SPLITS):
        raise ValueError("solved-density field split set mismatch")
    trace = evaluation.get("boundary_trace", {})
    if trace.get("dtype") != "float64":
        raise ValueError("solved-density trace evaluation must use float64")
    if set(trace.get("splits", {})) != set(SPLITS):
        raise ValueError("solved-density trace split set mismatch")


def reduce_reports(
    oracle: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen density-versus-discretization decision rule."""

    _validate_oracle(oracle)
    _validate_baseline(baseline)

    ratios: dict[str, dict[str, Any]] = {}
    trace_errors: dict[str, float] = {}
    for split in SPLITS:
        oracle_error = _require_finite_nonnegative(
            oracle["evaluation"]["splits"][split]["relative_l2_mean"],
            name=f"oracle {split} relative_l2_mean",
        )
        pure_errors = {
            str(seed): _require_finite_nonnegative(
                baseline["splits"][split]["arm_seed_relative_l2_means"]["pure"][
                    str(seed)
                ],
                name=f"pure seed {seed} {split} relative_l2_mean",
            )
            for seed in SEEDS
        }
        if any(value == 0.0 for value in pure_errors.values()):
            raise ValueError("pure-model baseline errors must be positive")
        per_seed = {
            seed: oracle_error / pure_error for seed, pure_error in pure_errors.items()
        }
        ratios[split] = {
            "oracle_relative_l2_mean": oracle_error,
            "pure_seed_relative_l2_means": pure_errors,
            "oracle_over_pure_per_seed": per_seed,
            "oracle_over_pure_geometric_mean": _geometric_mean(list(per_seed.values())),
            "role": ("exploratory" if split == MIXED_GEOMETRY_SPLIT else "primary"),
        }
        trace_errors[split] = _require_finite_nonnegative(
            oracle["evaluation"]["boundary_trace"]["splits"][split]["relative_l2_mean"],
            name=f"oracle {split} trace relative_l2_mean",
        )

    density_checks = {
        split: {
            "ratio_at_most_0_50": (
                ratios[split]["oracle_over_pure_geometric_mean"]
                <= DENSITY_RATIO_CEILING
            ),
            "trace_at_most_1e_10": trace_errors[split] <= TRACE_ERROR_CEILING,
        }
        for split in DENSITY_DECISION_SPLITS
    }
    density_passed = all(
        check["ratio_at_most_0_50"] and check["trace_at_most_1e_10"]
        for check in density_checks.values()
    )
    discretization_splits = [
        split
        for split in PRIMARY_SPLITS
        if ratios[split]["oracle_over_pure_geometric_mean"]
        >= DISCRETIZATION_RATIO_FLOOR
    ]
    discretization_passed = len(discretization_splits) >= 3

    if density_passed:
        verdict = "density_inference_is_principal_bottleneck"
    elif discretization_passed:
        verdict = "finite_boundary_discretization_is_principal_bottleneck"
    else:
        verdict = "split_dependent"

    return {
        "study": REDUCTION_STUDY,
        "verdict": verdict,
        "ratios": ratios,
        "trace_relative_l2_means": trace_errors,
        "decision": {
            "density_inference": {
                "passed": density_passed,
                "ratio_ceiling": DENSITY_RATIO_CEILING,
                "trace_error_ceiling": TRACE_ERROR_CEILING,
                "splits": density_checks,
            },
            "boundary_discretization": {
                "passed": discretization_passed,
                "ratio_floor": DISCRETIZATION_RATIO_FLOOR,
                "required_splits": 3,
                "qualifying_primary_splits": discretization_splits,
            },
        },
        "resolution": oracle["evaluation"]["resolution"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = reduce_reports(_load_json(args.oracle), _load_json(args.baseline))
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
