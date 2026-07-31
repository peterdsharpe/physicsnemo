# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the frozen coefficient-frequency coverage comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_frequency_coverage as study

FORMAL_SEEDS = (29, 43, 59, 71, 83)
FORMAL_EVALUATION_SEED = study.EVALUATION_SEED + 1_000_000
FIXED_NONINFERIORITY_MAX = 1.20
IMPROVEMENT_RATIO_MAX = 0.75
SMOOTH_RETENTION_MAX = 1.20
RESOLUTION_DRIFT_MAX = 1.20
ORACLE_ERROR_MAX = 0.15


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
        narrow = reports[("narrow_flow", seed)]
        broad = reports[("broad_flow", seed)]
        coverage_gates = {
            "smooth_within_20pct_of_fixed": (
                _metric(broad, "smooth_16")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "smooth_16")
            ),
            "covered_fast_within_20pct_of_fixed": (
                _metric(broad, "covered_fast_16")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "covered_fast_16")
            ),
            "covered_fast_at_least_25pct_better_than_narrow": (
                _metric(broad, "covered_fast_16")
                <= IMPROVEMENT_RATIO_MAX * _metric(narrow, "covered_fast_16")
            ),
            "smooth_within_20pct_of_narrow": (
                _metric(broad, "smooth_16")
                <= SMOOTH_RETENTION_MAX * _metric(narrow, "smooth_16")
            ),
        }
        local_law_gates = {
            "unseen16_within_20pct_of_fixed": (
                _metric(broad, "unseen_16")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "unseen_16")
            ),
            "unseen32_within_20pct_of_fixed": (
                _metric(broad, "unseen_32")
                <= FIXED_NONINFERIORITY_MAX * _metric(fixed, "unseen_32")
            ),
            "unseen16_at_least_25pct_better_than_narrow": (
                _metric(broad, "unseen_16")
                <= IMPROVEMENT_RATIO_MAX * _metric(narrow, "unseen_16")
            ),
            "unseen32_at_least_25pct_better_than_narrow": (
                _metric(broad, "unseen_32")
                <= IMPROVEMENT_RATIO_MAX * _metric(narrow, "unseen_32")
            ),
            "unseen32_within_20pct_of_unseen16": (
                _metric(broad, "unseen_32")
                <= RESOLUTION_DRIFT_MAX * _metric(broad, "unseen_16")
            ),
        }
        seed_results.append(
            {
                "seed": seed,
                "coverage_gates": coverage_gates,
                "coverage_passed": all(coverage_gates.values()),
                "local_law_gates": local_law_gates,
                "local_law_passed": all(local_law_gates.values()),
            }
        )

    coverage_seed_count = sum(result["coverage_passed"] for result in seed_results)
    local_law_seed_count = sum(result["local_law_passed"] for result in seed_results)
    coverage_earned = oracle_valid and coverage_seed_count >= 4
    local_law_earned = coverage_earned and oracle_valid and local_law_seed_count >= 4
    if not oracle_valid:
        verdict = "invalid_controls"
    elif not coverage_earned:
        verdict = "coverage_not_earned"
    elif local_law_earned:
        verdict = "coverage_and_local_law_earned"
    else:
        verdict = "coverage_earned_local_law_refuted"

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

    return {
        "study": study.STUDY,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
        "registered_thresholds": {
            "fixed_noninferiority_max": FIXED_NONINFERIORITY_MAX,
            "improvement_ratio_max": IMPROVEMENT_RATIO_MAX,
            "smooth_retention_max": SMOOTH_RETENTION_MAX,
            "resolution_drift_max": RESOLUTION_DRIFT_MAX,
            "oracle_error_max": ORACLE_ERROR_MAX,
            "required_seed_count": 4,
        },
        "controls": {
            "oracle_valid": oracle_valid,
            "all_valid": oracle_valid,
        },
        "seed_results": seed_results,
        "coverage_seed_count": coverage_seed_count,
        "local_law_seed_count": local_law_seed_count,
        "coverage_earned": coverage_earned,
        "local_law_earned": local_law_earned,
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
