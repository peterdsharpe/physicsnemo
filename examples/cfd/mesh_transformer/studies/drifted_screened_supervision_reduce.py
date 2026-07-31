# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the registered pointwise-versus-solution supervision study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import drifted_screened_principal_part as base
from drifted_screened_supervision import (
    ARMS,
    CHECK_QUADRATURE_ORDER,
    EVALUATION_BOUNDARY_POINTS,
    EVALUATION_CASES,
    EVALUATION_QUERY_POINTS,
    HELD_OUT_SOLUTION_MODES,
    KERNEL_EVALUATION_PAIRS,
    QUADRATURE_ORDER,
    RESOLUTION_CASES,
    RESOLUTIONS,
    SEEDS,
    STUDY,
    TRAIN_BOUNDARY_POINTS,
    TRAIN_QUADRATURE_ORDER,
    TRAIN_QUERY_POINTS,
    TRAIN_SOLUTION_MODES,
    TRAIN_STEPS,
)

ALIGNED_ARMS = ("solution", "hybrid")
OPERATOR_SPLITS = (
    "ood_low_screening",
    "ood_high_screening",
    "ood_high_drift",
)
DECIDING_METRICS = (
    "learned_field_relative_l2",
    "exact_trace_residual_relative_l2",
)


def _numeric_leaves(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from _numeric_leaves(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _pde_value(report: dict[str, Any], split: str, metric: str) -> float:
    return float(report["pde_evaluation"][split]["metrics"][metric]["mean"])


def _boundary_value(report: dict[str, Any], metric: str) -> float:
    return float(report["boundary_spectrum_evaluation"]["metrics"][metric]["mean"])


def _resolution_value(
    report: dict[str, Any],
    *,
    split: str,
    resolution: int,
    metric: str,
) -> float:
    return float(
        report["resolution_evaluation"][split]["resolutions"][str(resolution)][metric]
    )


def _paired_summary(
    candidate: list[float],
    baseline: list[float],
) -> dict[str, Any]:
    if len(candidate) != len(SEEDS) or len(baseline) != len(SEEDS):
        raise ValueError("paired summaries require every registered seed")
    ratios = [
        candidate_value / baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    return {
        "geometric_mean_ratio": _geometric_mean(ratios),
        "candidate_better_seed_count": sum(
            candidate_value < baseline_value
            for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
        ),
        "ratios_by_seed": {
            str(seed): ratio for seed, ratio in zip(SEEDS, ratios, strict=True)
        },
    }


def validate_reports(
    reports: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        if report.get("study") != STUDY:
            raise ValueError("input contains a report from another study")
        key = (str(report["arm"]), int(report["seed"]))
        if key in index:
            raise ValueError(f"duplicate report {key}")
        if not all(math.isfinite(value) for value in _numeric_leaves(report)):
            raise ValueError(f"nonfinite value in report {key}")
        protocol = report["protocol"]
        expected_protocol = {
            "train_steps": TRAIN_STEPS,
            "train_boundary_points": TRAIN_BOUNDARY_POINTS,
            "train_query_points": TRAIN_QUERY_POINTS,
            "train_quadrature_order_per_half_panel": TRAIN_QUADRATURE_ORDER,
            "train_solution_modes": list(TRAIN_SOLUTION_MODES),
            "held_out_solution_modes": list(HELD_OUT_SOLUTION_MODES),
            "evaluation_cases_per_split": EVALUATION_CASES,
            "evaluation_boundary_points": EVALUATION_BOUNDARY_POINTS,
            "evaluation_query_points": EVALUATION_QUERY_POINTS,
            "resolution_cases_per_split": RESOLUTION_CASES,
            "resolutions": list(RESOLUTIONS),
            "quadrature_order_per_half_panel": QUADRATURE_ORDER,
            "check_quadrature_order_per_half_panel": CHECK_QUADRATURE_ORDER,
            "kernel_evaluation_pairs_per_split": KERNEL_EVALUATION_PAIRS,
        }
        for name, value in expected_protocol.items():
            if protocol.get(name) != value:
                raise ValueError(
                    f"report {key} has nonregistered {name}: "
                    f"{protocol.get(name)!r} != {value!r}"
                )
        if set(report["pde_evaluation"]) != set(base.PDE_SPLIT_ORDER):
            raise ValueError(f"report {key} has the wrong PDE splits")
        if set(report["resolution_evaluation"]) != set(base.RESOLUTION_SPLITS):
            raise ValueError(f"report {key} has the wrong resolution splits")
        if report["boundary_spectrum_evaluation"]["solution_modes"] != list(
            HELD_OUT_SOLUTION_MODES
        ):
            raise ValueError(f"report {key} has the wrong held-out solution modes")
        index[key] = report
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        raise ValueError(f"incomplete factorial: missing={missing}, extra={extra}")
    source_hashes = {
        report["source"]["relevant_source_sha256"] for report in index.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("arm reports do not share one source fingerprint")
    return index


def _paired_pde(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    split: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_pde_value(index[(candidate, seed)], split, metric) for seed in SEEDS],
        [_pde_value(index[("pointwise", seed)], split, metric) for seed in SEEDS],
    )


def _paired_boundary(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_boundary_value(index[(candidate, seed)], metric) for seed in SEEDS],
        [_boundary_value(index[("pointwise", seed)], metric) for seed in SEEDS],
    )


def _paired_resolution(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    split: str,
    resolution: int,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [
            _resolution_value(
                index[(candidate, seed)],
                split=split,
                resolution=resolution,
                metric=metric,
            )
            for seed in SEEDS
        ],
        [
            _resolution_value(
                index[("pointwise", seed)],
                split=split,
                resolution=resolution,
                metric=metric,
            )
            for seed in SEEDS
        ],
    )


def _passes_improvement(summary: dict[str, Any]) -> bool:
    return (
        summary["geometric_mean_ratio"] <= 0.7
        and summary["candidate_better_seed_count"] >= 4
    )


def apply_registered_decision(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for arm in ALIGNED_ARMS:
        guards = {
            "in_distribution_field": _paired_pde(
                index,
                candidate=arm,
                split="in_distribution",
                metric="learned_field_relative_l2",
            ),
            "held_out_boundary_spectrum_field": _paired_boundary(
                index,
                candidate=arm,
                metric="learned_field_relative_l2",
            ),
            "near_boundary_field": _paired_pde(
                index,
                candidate=arm,
                split="near_boundary",
                metric="learned_field_relative_l2",
            ),
            "fine_resolution_field": _paired_resolution(
                index,
                candidate=arm,
                split="in_distribution",
                resolution=RESOLUTIONS[-1],
                metric="learned_field_relative_l2",
            ),
        }
        guard_pass = all(
            summary["geometric_mean_ratio"] <= 1.2 for summary in guards.values()
        )
        operator_transfer = {}
        for split in OPERATOR_SPLITS:
            metrics = {
                metric: _paired_pde(
                    index,
                    candidate=arm,
                    split=split,
                    metric=metric,
                )
                for metric in DECIDING_METRICS
            }
            operator_transfer[split] = {
                "metrics": metrics,
                "passes": all(
                    _passes_improvement(summary) for summary in metrics.values()
                ),
            }
        operator_splits_passed = sum(
            result["passes"] for result in operator_transfer.values()
        )
        comparisons[arm] = {
            "guards": guards,
            "guard_pass": guard_pass,
            "operator_transfer": operator_transfer,
            "operator_splits_passed": operator_splits_passed,
            "supervision_claim_earned": guard_pass and operator_splits_passed >= 2,
            "boundary_distribution_overfit": (
                operator_splits_passed >= 2
                and guards["held_out_boundary_spectrum_field"]["geometric_mean_ratio"]
                > 1.5
            ),
        }

    oracle_maximum = max(
        [
            _pde_value(report, split, "oracle_field_relative_l2")
            for report in index.values()
            for split in base.PDE_SPLIT_ORDER
        ]
        + [
            _boundary_value(report, "oracle_field_relative_l2")
            for report in index.values()
        ]
    )
    quadrature_maximum = max(
        [
            case["quadrature_relative_frobenius"]
            for report in index.values()
            for split in base.PDE_SPLIT_ORDER
            for case in report["pde_evaluation"][split]["cases"]
        ]
        + [
            case["quadrature_relative_frobenius"]
            for report in index.values()
            for case in report["boundary_spectrum_evaluation"]["cases"]
        ]
    )
    numerical_sanity = oracle_maximum <= 0.02 and quadrature_maximum <= 0.001

    solution_earned = comparisons["solution"]["supervision_claim_earned"]
    hybrid_earned = comparisons["hybrid"]["supervision_claim_earned"]
    if not numerical_sanity:
        verdict = "numerically_unresolved"
    elif hybrid_earned and not solution_earned:
        verdict = "kernel_identification_and_solution_alignment_are_complementary"
    elif solution_earned or hybrid_earned:
        verdict = "solution_alignment_improves_operator_transfer"
    elif comparisons["solution"]["boundary_distribution_overfit"]:
        verdict = "solution_only_overfits_boundary_distribution"
    else:
        verdict = "supervision_mismatch_not_principal"
    return {
        "verdict": verdict,
        "numerical_sanity": numerical_sanity,
        "oracle_maximum_mean_field_error": oracle_maximum,
        "quadrature_maximum_relative_frobenius": quadrature_maximum,
        "comparisons_to_pointwise": comparisons,
    }


def summarize(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        reports = [index[(arm, seed)] for seed in SEEDS]
        arms[arm] = {
            "parameters": reports[0]["parameters"],
            "learned_singular_coefficients": {
                str(seed): index[(arm, seed)]["learned_singular_coefficient"]
                for seed in SEEDS
            },
            "pde_field_geometric_means": {
                split: _geometric_mean(
                    [
                        _pde_value(report, split, "learned_field_relative_l2")
                        for report in reports
                    ]
                )
                for split in base.PDE_SPLIT_ORDER
            },
            "pde_trace_geometric_means": {
                split: _geometric_mean(
                    [
                        _pde_value(report, split, "exact_trace_residual_relative_l2")
                        for report in reports
                    ]
                )
                for split in base.PDE_SPLIT_ORDER
            },
            "held_out_boundary_field_geometric_mean": _geometric_mean(
                [
                    _boundary_value(report, "learned_field_relative_l2")
                    for report in reports
                ]
            ),
            "held_out_boundary_trace_geometric_mean": _geometric_mean(
                [
                    _boundary_value(report, "exact_trace_residual_relative_l2")
                    for report in reports
                ]
            ),
            "near_singular_kernel_geometric_mean": _geometric_mean(
                [
                    report["kernel_evaluation"]["near_singular"][
                        "scaled_kernel_relative_l2"
                    ]
                    for report in reports
                ]
            ),
        }
    return {
        "study": STUDY,
        "arms": arms,
        "decision": apply_registered_decision(index),
        "source_sha256": next(
            iter(
                {
                    report["source"]["relevant_source_sha256"]
                    for report in index.values()
                }
            )
        ),
    }


def reduce_directory(input_dir: Path) -> dict[str, Any]:
    reports = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open() as stream:
            reports.append(json.load(stream))
    return summarize(validate_reports(reports))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = reduce_directory(args.input_dir)
    base.atomic_write_json(args.output, summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
