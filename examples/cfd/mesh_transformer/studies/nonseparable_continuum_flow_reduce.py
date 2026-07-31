# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the frozen nonseparable continuum-flow comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_continuum_flow as study

FORMAL_SEEDS = (29, 43, 59, 71, 83)
FORMAL_EVALUATION_SEED = study.EVALUATION_SEED + 1_000_000
FORMAL_ORDER_SEED = study.ORDER_SEED + 1_000_000
FIXED_NONINFERIORITY_MAX = 1.20
IMPROVEMENT_RATIO_MAX = 0.75
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
    for path in sorted(input_dir.glob("*_seed*.json")):
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
        recurrence = reports[("gru_path_rank4", seed)]
        sorted_flow = reports[("sorted_flow_rank4", seed)]
        path_flow = reports[("path_flow_rank4", seed)]
        continuum_gates = {
            "id_within_20pct_of_fixed": (
                _metric(path_flow, "in_distribution_8")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "in_distribution_8")
            ),
            "coarse_within_20pct_of_fixed": (
                _metric(path_flow, "coarse_4")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "coarse_4")
            ),
            "refined16_within_20pct_of_fixed": (
                _metric(path_flow, "refined_16")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "refined_16")
            ),
            "refined32_within_20pct_of_fixed": (
                _metric(path_flow, "refined_32")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "refined_32")
            ),
            "refined16_at_least_25pct_better_than_recurrence": (
                _metric(path_flow, "refined_16")
                <= IMPROVEMENT_RATIO_MAX * _metric(recurrence, "refined_16")
            ),
            "refined32_at_least_25pct_better_than_recurrence": (
                _metric(path_flow, "refined_32")
                <= IMPROVEMENT_RATIO_MAX * _metric(recurrence, "refined_32")
            ),
            "all_resolution_shifts_better_than_sorted": all(
                _metric(path_flow, split)
                <= IMPROVEMENT_RATIO_MAX * _metric(sorted_flow, split)
                for split in ("coarse_4", "refined_16", "refined_32")
            ),
        }
        frequency_gates = {
            "fast8_within_20pct_of_fixed": (
                _metric(path_flow, "fast_8")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "fast_8")
            ),
            "fast16_within_20pct_of_fixed": (
                _metric(path_flow, "fast_16")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "fast_16")
            ),
            "fast8_at_least_25pct_better_than_recurrence": (
                _metric(path_flow, "fast_8")
                <= IMPROVEMENT_RATIO_MAX * _metric(recurrence, "fast_8")
            ),
            "fast16_at_least_25pct_better_than_recurrence": (
                _metric(path_flow, "fast_16")
                <= IMPROVEMENT_RATIO_MAX * _metric(recurrence, "fast_16")
            ),
            "fast8_at_least_25pct_better_than_sorted": (
                _metric(path_flow, "fast_8")
                <= IMPROVEMENT_RATIO_MAX * _metric(sorted_flow, "fast_8")
            ),
            "fast16_at_least_25pct_better_than_sorted": (
                _metric(path_flow, "fast_16")
                <= IMPROVEMENT_RATIO_MAX * _metric(sorted_flow, "fast_16")
            ),
        }
        sorted_contrast = sorted_flow["order_challenge"][
            "predicted_contrast_relative_l2"
        ]
        seed_results.append(
            {
                "seed": seed,
                "continuum_gates": continuum_gates,
                "continuum_passed": all(continuum_gates.values()),
                "frequency_gates": frequency_gates,
                "frequency_passed": all(frequency_gates.values()),
                "sorted_predicted_contrast_relative_l2": sorted_contrast,
                "sorted_control_passed": (
                    abs(sorted_contrast) <= SORTED_CONTRAST_ABS_MAX
                ),
            }
        )

    continuum_seed_count = sum(result["continuum_passed"] for result in seed_results)
    frequency_seed_count = sum(result["frequency_passed"] for result in seed_results)
    sorted_valid = all(result["sorted_control_passed"] for result in seed_results)
    controls_valid = oracle_valid and sorted_valid
    continuum_earned = controls_valid and continuum_seed_count >= 4
    frequency_earned = continuum_earned and frequency_seed_count >= 4
    if not controls_valid:
        verdict = "invalid_controls"
    elif not continuum_earned:
        verdict = "continuum_flow_not_earned"
    elif frequency_earned:
        verdict = "continuum_and_frequency_transfer_earned"
    else:
        verdict = "continuum_earned_frequency_transfer_refuted"

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
            "contrast_recovery_fraction_range": [
                min(
                    report["order_challenge"]["contrast_recovery_fraction"]
                    for report in arm_reports
                ),
                max(
                    report["order_challenge"]["contrast_recovery_fraction"]
                    for report in arm_reports
                ),
            ]
        }

    return {
        "study": study.STUDY,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
        "formal_order_seed": FORMAL_ORDER_SEED,
        "registered_thresholds": {
            "fixed_noninferiority_max": FIXED_NONINFERIORITY_MAX,
            "improvement_ratio_max": IMPROVEMENT_RATIO_MAX,
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
        "continuum_seed_count": continuum_seed_count,
        "frequency_seed_count": frequency_seed_count,
        "continuum_earned": continuum_earned,
        "frequency_transfer_earned": frequency_earned,
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
