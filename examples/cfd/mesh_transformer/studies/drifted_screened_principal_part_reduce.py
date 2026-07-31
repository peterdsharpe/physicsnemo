# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the registered principal-part transfer factorial."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from drifted_screened_principal_part import (
    ARMS,
    CHECK_QUADRATURE_ORDER,
    EVALUATION_BOUNDARY_POINTS,
    EVALUATION_CASES,
    EVALUATION_QUERY_POINTS,
    KERNEL_EVALUATION_PAIRS,
    PDE_SPLIT_ORDER,
    QUADRATURE_ORDER,
    RESOLUTION_CASES,
    RESOLUTIONS,
    SEEDS,
    STUDY,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    atomic_write_json,
)

COMPETITORS = ("free_principal", "fully_learned")
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


def _paired_summary(fixed: list[float], comparison: list[float]) -> dict[str, Any]:
    if len(fixed) != len(SEEDS) or len(comparison) != len(SEEDS):
        raise ValueError("paired summaries require every registered seed")
    ratios = [
        fixed_value / comparison_value
        for fixed_value, comparison_value in zip(fixed, comparison, strict=True)
    ]
    return {
        "geometric_mean_ratio": _geometric_mean(ratios),
        "fixed_better_seed_count": sum(
            fixed_value < comparison_value
            for fixed_value, comparison_value in zip(fixed, comparison, strict=True)
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
            "train_batch_size": TRAIN_BATCH_SIZE,
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
        if set(report["pde_evaluation"]) != set(PDE_SPLIT_ORDER):
            raise ValueError(f"report {key} has the wrong PDE splits")
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


def _arm_reports(
    index: dict[tuple[str, int], dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    return [index[(arm, seed)] for seed in SEEDS]


def _paired_pde(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    competitor: str,
    split: str,
    metric: str,
) -> dict[str, Any]:
    fixed = [
        _pde_value(index[("fixed_principal", seed)], split, metric) for seed in SEEDS
    ]
    comparison = [
        _pde_value(index[(competitor, seed)], split, metric) for seed in SEEDS
    ]
    return _paired_summary(fixed, comparison)


def _paired_resolution(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    competitor: str,
    split: str,
    resolution: int,
    metric: str,
) -> dict[str, Any]:
    fixed = [
        _resolution_value(
            index[("fixed_principal", seed)],
            split=split,
            resolution=resolution,
            metric=metric,
        )
        for seed in SEEDS
    ]
    comparison = [
        _resolution_value(
            index[(competitor, seed)],
            split=split,
            resolution=resolution,
            metric=metric,
        )
        for seed in SEEDS
    ]
    return _paired_summary(fixed, comparison)


def _passes_ratio(summary: dict[str, Any], ceiling: float) -> bool:
    return (
        summary["geometric_mean_ratio"] <= ceiling
        and summary["fixed_better_seed_count"] >= 4
    )


def apply_registered_decision(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arm_id_geometric_means = {
        arm: _geometric_mean(
            [
                _pde_value(
                    index[(arm, seed)], "in_distribution", "learned_field_relative_l2"
                )
                for seed in SEEDS
            ]
        )
        for arm in ARMS
    }
    id_noninferior = arm_id_geometric_means["fixed_principal"] <= 1.2 * min(
        arm_id_geometric_means[competitor] for competitor in COMPETITORS
    )

    near_boundary = {
        competitor: {
            metric: _paired_pde(
                index,
                competitor=competitor,
                split="near_boundary",
                metric=metric,
            )
            for metric in DECIDING_METRICS
        }
        for competitor in COMPETITORS
    }
    fine_resolution = {
        competitor: {
            metric: _paired_resolution(
                index,
                competitor=competitor,
                split="in_distribution",
                resolution=RESOLUTIONS[-1],
                metric=metric,
            )
            for metric in DECIDING_METRICS
        }
        for competitor in COMPETITORS
    }
    near_boundary_pass = all(
        _passes_ratio(summary, 0.5)
        for comparison in near_boundary.values()
        for summary in comparison.values()
    )
    fine_resolution_pass = all(
        _passes_ratio(summary, 0.5)
        for comparison in fine_resolution.values()
        for summary in comparison.values()
    )
    principal_part_earned = (
        id_noninferior and near_boundary_pass and fine_resolution_pass
    )

    operator_transfer: dict[str, Any] = {}
    for split in OPERATOR_SPLITS:
        comparisons = {
            competitor: {
                metric: _paired_pde(
                    index,
                    competitor=competitor,
                    split=split,
                    metric=metric,
                )
                for metric in DECIDING_METRICS
            }
            for competitor in COMPETITORS
        }
        operator_transfer[split] = {
            "comparisons": comparisons,
            "passes": all(
                _passes_ratio(summary, 0.7)
                for comparison in comparisons.values()
                for summary in comparison.values()
            ),
        }
    operator_splits_passed = sum(
        result["passes"] for result in operator_transfer.values()
    )
    operator_transfer_earned = principal_part_earned and operator_splits_passed >= 2

    oracle_max_mean = max(
        _pde_value(report, split, "oracle_field_relative_l2")
        for report in index.values()
        for split in PDE_SPLIT_ORDER
    )
    quadrature_maximum = max(
        case["quadrature_relative_frobenius"]
        for report in index.values()
        for split in PDE_SPLIT_ORDER
        for case in report["pde_evaluation"][split]["cases"]
    )
    numerical_sanity = oracle_max_mean <= 0.02 and quadrature_maximum <= 0.001

    if not numerical_sanity:
        verdict = "numerically_unresolved"
    elif operator_transfer_earned:
        verdict = "fixed_principal_improves_operator_transfer"
    elif principal_part_earned:
        verdict = "fixed_principal_improves_near_trace_only"
    elif not id_noninferior:
        verdict = "fixed_principal_harms_interpolation"
    else:
        verdict = "hard_fixed_coefficient_not_earned"
    return {
        "verdict": verdict,
        "numerical_sanity": numerical_sanity,
        "oracle_max_split_mean_field_error": oracle_max_mean,
        "quadrature_maximum_relative_frobenius": quadrature_maximum,
        "id_field_geometric_means": arm_id_geometric_means,
        "id_noninferior": id_noninferior,
        "near_boundary": near_boundary,
        "near_boundary_pass": near_boundary_pass,
        "fine_resolution": fine_resolution,
        "fine_resolution_pass": fine_resolution_pass,
        "principal_part_earned": principal_part_earned,
        "operator_transfer": operator_transfer,
        "operator_splits_passed": operator_splits_passed,
        "operator_transfer_earned": operator_transfer_earned,
    }


def summarize(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        reports = _arm_reports(index, arm)
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
                for split in PDE_SPLIT_ORDER
            },
            "pde_trace_geometric_means": {
                split: _geometric_mean(
                    [
                        _pde_value(report, split, "exact_trace_residual_relative_l2")
                        for report in reports
                    ]
                )
                for split in PDE_SPLIT_ORDER
            },
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
    paths = sorted(input_dir.glob("*.json"))
    reports = []
    for path in paths:
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
    atomic_write_json(args.output, summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
