# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the registered two-limit asymptotic-carrier study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import drifted_screened_asymptotic as study
import drifted_screened_principal_part as base

BASELINE = "raw_hybrid"
CANDIDATES = ("fixed_carrier", "learned_carrier")
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
    if len(candidate) != len(study.SEEDS) or len(baseline) != len(study.SEEDS):
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
            str(seed): ratio for seed, ratio in zip(study.SEEDS, ratios, strict=True)
        },
    }


def validate_reports(
    reports: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        if report.get("study") != study.STUDY:
            raise ValueError("input contains a report from another study")
        key = (str(report["arm"]), int(report["seed"]))
        if key in index:
            raise ValueError(f"duplicate report {key}")
        if key not in expected:
            raise ValueError(f"unregistered report {key}")
        if not all(math.isfinite(value) for value in _numeric_leaves(report)):
            raise ValueError(f"nonfinite value in report {key}")
        protocol = report["protocol"]
        training_applied = key[0] != "fixed_carrier"
        expected_protocol = {
            "training_applied": training_applied,
            "train_steps": study.TRAIN_STEPS if training_applied else 0,
            "train_boundary_points": study.TRAIN_BOUNDARY_POINTS,
            "train_query_points": study.TRAIN_QUERY_POINTS,
            "train_quadrature_order_per_half_panel": study.TRAIN_QUADRATURE_ORDER,
            "train_solution_modes": list(study.TRAIN_SOLUTION_MODES),
            "held_out_solution_modes": list(study.HELD_OUT_SOLUTION_MODES),
            "evaluation_cases_per_split": study.EVALUATION_CASES,
            "evaluation_boundary_points": study.EVALUATION_BOUNDARY_POINTS,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "resolution_cases_per_split": study.RESOLUTION_CASES,
            "resolutions": list(study.RESOLUTIONS),
            "quadrature_order_per_half_panel": study.QUADRATURE_ORDER,
            "check_quadrature_order_per_half_panel": study.CHECK_QUADRATURE_ORDER,
            "kernel_evaluation_pairs_per_split": study.KERNEL_EVALUATION_PAIRS,
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
            study.HELD_OUT_SOLUTION_MODES
        ):
            raise ValueError(f"report {key} has the wrong held-out solution modes")
        index[key] = report
    missing = expected - set(index)
    if missing:
        raise ValueError(f"incomplete factorial: missing={missing}")
    source_hashes = {
        report["source"]["relevant_source_sha256"] for report in index.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("arm reports do not share one source fingerprint")
    learned_counts = {
        index[(arm, seed)]["parameters"]
        for arm in ("raw_hybrid", "learned_carrier")
        for seed in study.SEEDS
    }
    if len(learned_counts) != 1:
        raise ValueError("learned arms do not have matched capacity")
    if any(index[("fixed_carrier", seed)]["parameters"] != 0 for seed in study.SEEDS):
        raise ValueError("fixed carrier unexpectedly has learned parameters")
    fixed_keys = (
        "parameters",
        "training_history",
        "learned_singular_coefficient",
        "kernel_evaluation",
        "pde_evaluation",
        "boundary_spectrum_evaluation",
        "resolution_evaluation",
    )
    reference = index[("fixed_carrier", study.SEEDS[0])]
    for seed in study.SEEDS[1:]:
        report = index[("fixed_carrier", seed)]
        if any(report[name] != reference[name] for name in fixed_keys):
            raise ValueError("deterministic carrier differs across seed labels")
    return index


def _paired_pde(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    split: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_pde_value(index[(candidate, seed)], split, metric) for seed in study.SEEDS],
        [_pde_value(index[(baseline, seed)], split, metric) for seed in study.SEEDS],
    )


def _paired_boundary(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_boundary_value(index[(candidate, seed)], metric) for seed in study.SEEDS],
        [_boundary_value(index[(baseline, seed)], metric) for seed in study.SEEDS],
    )


def _paired_resolution(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
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
            for seed in study.SEEDS
        ],
        [
            _resolution_value(
                index[(baseline, seed)],
                split=split,
                resolution=resolution,
                metric=metric,
            )
            for seed in study.SEEDS
        ],
    )


def _passes_improvement(summary: dict[str, Any]) -> bool:
    return (
        summary["geometric_mean_ratio"] <= 0.7
        and summary["candidate_better_seed_count"] >= 4
    )


def _candidate_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    guards = {
        "in_distribution_field": _paired_pde(
            index,
            candidate=candidate,
            baseline=baseline,
            split="in_distribution",
            metric="learned_field_relative_l2",
        ),
        "held_out_boundary_spectrum_field": _paired_boundary(
            index,
            candidate=candidate,
            baseline=baseline,
            metric="learned_field_relative_l2",
        ),
        "near_boundary_field": _paired_pde(
            index,
            candidate=candidate,
            baseline=baseline,
            split="near_boundary",
            metric="learned_field_relative_l2",
        ),
        "fine_resolution_field": _paired_resolution(
            index,
            candidate=candidate,
            baseline=baseline,
            split="in_distribution",
            resolution=study.RESOLUTIONS[-1],
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
                candidate=candidate,
                baseline=baseline,
                split=split,
                metric=metric,
            )
            for metric in DECIDING_METRICS
        }
        operator_transfer[split] = {
            "metrics": metrics,
            "passes": all(_passes_improvement(summary) for summary in metrics.values()),
        }
    operator_splits_passed = sum(
        result["passes"] for result in operator_transfer.values()
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "guards": guards,
        "guard_pass": guard_pass,
        "operator_transfer": operator_transfer,
        "operator_splits_passed": operator_splits_passed,
        "transfer_claim_earned": guard_pass and operator_splits_passed >= 2,
    }


def apply_registered_decision(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    comparisons = {
        candidate: _candidate_comparison(
            index,
            candidate=candidate,
            baseline=BASELINE,
        )
        for candidate in CANDIDATES
    }
    learned_vs_fixed = {
        "in_distribution_field": _paired_pde(
            index,
            candidate="learned_carrier",
            baseline="fixed_carrier",
            split="in_distribution",
            metric="learned_field_relative_l2",
        ),
        "held_out_boundary_spectrum_field": _paired_boundary(
            index,
            candidate="learned_carrier",
            baseline="fixed_carrier",
            metric="learned_field_relative_l2",
        ),
        "operator_transfer": {
            split: {
                metric: _paired_pde(
                    index,
                    candidate="learned_carrier",
                    baseline="fixed_carrier",
                    split=split,
                    metric=metric,
                )
                for metric in DECIDING_METRICS
            }
            for split in OPERATOR_SPLITS
        },
    }
    learning_improves_fields = all(
        _passes_improvement(learned_vs_fixed[name])
        for name in (
            "in_distribution_field",
            "held_out_boundary_spectrum_field",
        )
    )
    learning_preserves_operator_splits = all(
        summary["geometric_mean_ratio"] <= 1.2
        for split in OPERATOR_SPLITS
        for summary in learned_vs_fixed["operator_transfer"][split].values()
    )
    learning_complexity_earned = (
        learning_improves_fields and learning_preserves_operator_splits
    )
    learned_vs_fixed.update(
        {
            "learning_improves_fields": learning_improves_fields,
            "learning_preserves_operator_splits": learning_preserves_operator_splits,
            "learning_complexity_earned": learning_complexity_earned,
        }
    )

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

    fixed_earned = comparisons["fixed_carrier"]["transfer_claim_earned"]
    learned_earned = comparisons["learned_carrier"]["transfer_claim_earned"]
    high_screening_only = any(
        comparison["operator_transfer"]["ood_high_screening"]["passes"]
        and comparison["operator_splits_passed"] < 2
        for comparison in comparisons.values()
    )
    if not numerical_sanity:
        verdict = "numerically_unresolved"
    elif learned_earned and learning_complexity_earned:
        verdict = "learned_transition_earned"
    elif fixed_earned:
        verdict = "analytic_two_limit_carrier_sufficient"
    elif learned_earned:
        verdict = "scaffold_passes_but_learning_complexity_not_earned"
    elif high_screening_only:
        verdict = "high_screening_specialist_only"
    else:
        verdict = "two_limit_scaffold_not_sufficient"
    return {
        "verdict": verdict,
        "numerical_sanity": numerical_sanity,
        "oracle_maximum_mean_field_error": oracle_maximum,
        "quadrature_maximum_relative_frobenius": quadrature_maximum,
        "comparisons_to_raw_hybrid": comparisons,
        "learned_vs_fixed_carrier": learned_vs_fixed,
    }


def summarize(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in study.ARMS:
        reports = [index[(arm, seed)] for seed in study.SEEDS]
        arms[arm] = {
            "parameters": reports[0]["parameters"],
            "learned_singular_coefficients": {
                str(seed): index[(arm, seed)]["learned_singular_coefficient"]
                for seed in study.SEEDS
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
        "study": study.STUDY,
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
