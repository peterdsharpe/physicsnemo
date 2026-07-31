# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the frozen nonseparable latent-composition comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_latent_composition as study

FORMAL_SEEDS = (29, 43, 59, 71, 83)
FORMAL_EVALUATION_SEED = study.EVALUATION_SEED + 1_000_000
FORMAL_ORDER_SEED = study.ORDER_SEED + 1_000_000
ORDERED_RATIO_MAX = 0.75
FIXED_NONINFERIORITY_MAX = 1.20
CONTRAST_RECOVERY_MIN = 0.50
GAP_CLOSURE_MIN = 0.50
ORACLE_ERROR_MAX = 0.15
SORTED_CONTRAST_ABS_MAX = 1.0e-12


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _metric(report: dict[str, Any], split: str) -> float:
    return report["split_evaluation"][split]["metrics"]["cross_channel_relative_l2"][
        "mean"
    ]


def _load_reports(input_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.json")):
        with path.open() as stream:
            report = json.load(stream)
        if report.get("study") != study.STUDY:
            raise ValueError(f"{path}: unexpected study {report.get('study')!r}")
        key = (report["arm"], report["seed"])
        if key in reports:
            raise ValueError(f"duplicate report {key}")
        reports[key] = report
    expected = {(arm, seed) for arm in study.LEARNED_ARMS for seed in FORMAL_SEEDS}
    expected.update((arm, FORMAL_SEEDS[0]) for arm in study.ANALYTIC_ARMS)
    missing = sorted(expected - reports.keys())
    unexpected = sorted(reports.keys() - expected)
    if missing or unexpected:
        raise ValueError(f"report mismatch: missing={missing}, unexpected={unexpected}")
    for report in reports.values():
        protocol = report["protocol"]
        if protocol["evaluation_seed"] != FORMAL_EVALUATION_SEED:
            raise ValueError("formal evaluation seed mismatch")
        if protocol["order_seed"] != FORMAL_ORDER_SEED:
            raise ValueError("formal order seed mismatch")
    return reports


def _gap_closure(path_error: float, global_error: float, oracle_error: float) -> float:
    denominator = global_error - oracle_error
    if denominator <= 0.0:
        return -math.inf
    return (global_error - path_error) / denominator


def reduce_reports(input_dir: Path) -> dict[str, Any]:
    reports = _load_reports(input_dir)
    analytic = {arm: reports[(arm, FORMAL_SEEDS[0])] for arm in study.ANALYTIC_ARMS}
    fixed = analytic["fixed_rank4"]
    oracle = analytic["oracle_rank4"]
    oracle_valid = all(
        _metric(oracle, split) <= ORACLE_ERROR_MAX for split in study.SPLITS
    )

    seed_results = []
    for seed in FORMAL_SEEDS:
        global_rank4 = reports[("global_rank4", seed)]
        path_rank4 = reports[("path_rank4", seed)]
        sorted_rank4 = reports[("sorted_rank4", seed)]
        ordered_gates = {
            "id_at_least_25pct_better_than_global": (
                _metric(path_rank4, "in_distribution")
                <= ORDERED_RATIO_MAX * _metric(global_rank4, "in_distribution")
            ),
            "high_at_least_25pct_better_than_global": (
                _metric(path_rank4, "high_heterogeneity")
                <= ORDERED_RATIO_MAX * _metric(global_rank4, "high_heterogeneity")
            ),
            "id_within_20pct_of_fixed": (
                _metric(path_rank4, "in_distribution")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "in_distribution")
            ),
            "high_within_20pct_of_fixed": (
                _metric(path_rank4, "high_heterogeneity")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "high_heterogeneity")
            ),
            "swap_contrast_at_least_half_recovered": (
                path_rank4["order_challenge"]["contrast_recovery_fraction"]
                >= CONTRAST_RECOVERY_MIN
            ),
        }
        fast_gap = _gap_closure(
            _metric(path_rank4, "fast_variation"),
            _metric(global_rank4, "fast_variation"),
            _metric(oracle, "fast_variation"),
        )
        combined_gap = _gap_closure(
            _metric(path_rank4, "combined_shift"),
            _metric(global_rank4, "combined_shift"),
            _metric(oracle, "combined_shift"),
        )
        breadth_gates = {
            "fast_gap_at_least_half_closed": fast_gap >= GAP_CLOSURE_MIN,
            "combined_gap_at_least_half_closed": combined_gap >= GAP_CLOSURE_MIN,
            "fast_within_20pct_of_fixed": (
                _metric(path_rank4, "fast_variation")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "fast_variation")
            ),
            "combined_within_20pct_of_fixed": (
                _metric(path_rank4, "combined_shift")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "combined_shift")
            ),
        }
        sorted_contrast = sorted_rank4["order_challenge"][
            "predicted_contrast_relative_l2"
        ]
        seed_results.append(
            {
                "seed": seed,
                "ordered_gates": ordered_gates,
                "ordered_passed": all(ordered_gates.values()),
                "breadth_gates": breadth_gates,
                "breadth_passed": all(breadth_gates.values()),
                "fast_global_to_oracle_gap_closure": fast_gap,
                "combined_global_to_oracle_gap_closure": combined_gap,
                "sorted_predicted_contrast_relative_l2": sorted_contrast,
                "sorted_control_passed": (
                    abs(sorted_contrast) <= SORTED_CONTRAST_ABS_MAX
                ),
            }
        )

    ordered_seed_count = sum(result["ordered_passed"] for result in seed_results)
    breadth_seed_count = sum(result["breadth_passed"] for result in seed_results)
    sorted_valid = all(result["sorted_control_passed"] for result in seed_results)
    controls_valid = oracle_valid and sorted_valid
    ordered_earned = controls_valid and ordered_seed_count >= 4
    breadth_earned = ordered_earned and breadth_seed_count >= 4
    if not controls_valid:
        verdict = "invalid_controls"
    elif not ordered_earned:
        verdict = "ordered_compression_not_earned"
    elif breadth_earned:
        verdict = "ordered_compression_and_breadth_earned"
    else:
        verdict = "ordered_compression_earned_breadth_refuted"

    aggregate: dict[str, Any] = {}
    for arm in study.ARMS:
        arm_reports = (
            [reports[(arm, seed)] for seed in FORMAL_SEEDS]
            if arm in study.LEARNED_ARMS
            else [analytic[arm]]
        )
        aggregate[arm] = {
            split: {
                "cross_channel_relative_l2_geomean": _geometric_mean(
                    [_metric(report, split) for report in arm_reports]
                )
            }
            for split in study.SPLITS
        }
        aggregate[arm]["order_challenge"] = {
            "paired_residual_relative_l2_geomean": _geometric_mean(
                [
                    report["order_challenge"]["paired_residual_relative_l2"]
                    for report in arm_reports
                ]
            ),
            "contrast_recovery_fraction_range": [
                min(
                    report["order_challenge"]["contrast_recovery_fraction"]
                    for report in arm_reports
                ),
                max(
                    report["order_challenge"]["contrast_recovery_fraction"]
                    for report in arm_reports
                ),
            ],
        }

    return {
        "study": study.STUDY,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
        "formal_order_seed": FORMAL_ORDER_SEED,
        "registered_thresholds": {
            "ordered_ratio_max": ORDERED_RATIO_MAX,
            "fixed_noninferiority_max": FIXED_NONINFERIORITY_MAX,
            "contrast_recovery_min": CONTRAST_RECOVERY_MIN,
            "gap_closure_min": GAP_CLOSURE_MIN,
            "oracle_error_max": ORACLE_ERROR_MAX,
            "sorted_contrast_abs_max": SORTED_CONTRAST_ABS_MAX,
            "required_seed_count": 4,
        },
        "controls": {
            "oracle_valid": oracle_valid,
            "sorted_valid": sorted_valid,
            "all_valid": controls_valid,
        },
        "seed_results": seed_results,
        "ordered_seed_count": ordered_seed_count,
        "breadth_seed_count": breadth_seed_count,
        "ordered_compression_earned": ordered_earned,
        "breadth_earned": breadth_earned,
        "aggregate": aggregate,
        "verdict": verdict,
        "source_hashes": sorted(
            {report["source"]["relevant_source_sha256"] for report in reports.values()}
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = reduce_reports(args.input_dir)
    shared.atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
