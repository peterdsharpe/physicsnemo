# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce independent nonseparable rank-census reports against frozen gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared

STUDY = "nonseparable_rank_census_v1"
EXPECTED_REPORTS = 4
LEVELS = ("0.2", "0.5", "0.8")
STRONG_LEVEL = "0.8"
ORACLE_RANK4_MAX = 0.15
ORACLE_RANK3_MIN = 0.20
BASELINE_TO_ORACLE_MIN = 2.0
CROSS_CHANNEL_STRENGTH_MIN = 0.004


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _load_reports(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("seed*.json"))
    if len(paths) != EXPECTED_REPORTS:
        raise ValueError(f"expected {EXPECTED_REPORTS} reports, found {len(paths)}")
    reports = []
    for path in paths:
        with path.open() as stream:
            report = json.load(stream)
        if report.get("study") != STUDY:
            raise ValueError(f"{path}: unexpected study {report.get('study')!r}")
        reports.append(report)
    seeds = [report["protocol"]["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique, got {seeds}")
    return reports


def _evaluate_report(report: dict[str, Any]) -> dict[str, Any]:
    levels = report["levels"]
    oracle_rank4 = {
        level: levels[level]["ranks"]["4"]["oracle_cross_channel_relative_l2"]["mean"]
        for level in LEVELS
    }
    oracle_rank3 = {
        level: levels[level]["ranks"]["3"]["oracle_cross_channel_relative_l2"]["mean"]
        for level in LEVELS
    }
    strong_oracle = oracle_rank4[STRONG_LEVEL]
    strong_rank4 = levels[STRONG_LEVEL]["ranks"]["4"]
    fixed_ratio = (
        strong_rank4["fixed_truncation_cross_channel_relative_l2"]["mean"]
        / strong_oracle
    )
    global_ratio = (
        strong_rank4["global_eigenspace_cross_channel_relative_l2"]["mean"]
        / strong_oracle
    )
    cross_strength = levels[STRONG_LEVEL]["cross_channel_to_operator_norm_ratio"]
    gates = {
        "rank4_oracle_at_most_15pct_all_levels": all(
            value <= ORACLE_RANK4_MAX for value in oracle_rank4.values()
        ),
        "rank3_oracle_above_20pct_all_levels": all(
            value > ORACLE_RANK3_MIN for value in oracle_rank3.values()
        ),
        "fixed_rank4_at_least_2x_oracle_strong": (
            fixed_ratio >= BASELINE_TO_ORACLE_MIN
        ),
        "global_rank4_at_least_2x_oracle_strong": (
            global_ratio >= BASELINE_TO_ORACLE_MIN
        ),
        "cross_channel_strength_at_least_0p4pct_strong": (
            cross_strength >= CROSS_CHANNEL_STRENGTH_MIN
        ),
    }
    return {
        "seed": report["protocol"]["seed"],
        "oracle_rank3_cross_channel_relative_l2": oracle_rank3,
        "oracle_rank4_cross_channel_relative_l2": oracle_rank4,
        "strong_fixed_to_oracle_ratio": fixed_ratio,
        "strong_global_to_oracle_ratio": global_ratio,
        "strong_cross_channel_to_operator_norm_ratio": cross_strength,
        "gates": gates,
        "passed": all(gates.values()),
    }


def reduce_reports(input_dir: Path) -> dict[str, Any]:
    reports = _load_reports(input_dir)
    replicates = [_evaluate_report(report) for report in reports]
    aggregate: dict[str, Any] = {}
    for level in LEVELS:
        rank3 = [
            report["levels"][level]["ranks"]["3"]["oracle_cross_channel_relative_l2"][
                "mean"
            ]
            for report in reports
        ]
        rank4 = [
            report["levels"][level]["ranks"]["4"]["oracle_cross_channel_relative_l2"][
                "mean"
            ]
            for report in reports
        ]
        fixed = [
            report["levels"][level]["ranks"]["4"][
                "fixed_truncation_cross_channel_relative_l2"
            ]["mean"]
            for report in reports
        ]
        global_eigenspace = [
            report["levels"][level]["ranks"]["4"][
                "global_eigenspace_cross_channel_relative_l2"
            ]["mean"]
            for report in reports
        ]
        strength = [
            report["levels"][level]["cross_channel_to_operator_norm_ratio"]
            for report in reports
        ]
        aggregate[level] = {
            "oracle_rank3_cross_channel_relative_l2_geomean": _geometric_mean(rank3),
            "oracle_rank4_cross_channel_relative_l2_geomean": _geometric_mean(rank4),
            "fixed_rank4_cross_channel_relative_l2_geomean": _geometric_mean(fixed),
            "global_rank4_cross_channel_relative_l2_geomean": _geometric_mean(
                global_eigenspace
            ),
            "cross_channel_to_operator_norm_ratio_geomean": _geometric_mean(strength),
        }
    passed = all(replicate["passed"] for replicate in replicates)
    return {
        "study": STUDY,
        "registered_thresholds": {
            "expected_reports": EXPECTED_REPORTS,
            "oracle_rank4_cross_channel_relative_l2_max": ORACLE_RANK4_MAX,
            "oracle_rank3_cross_channel_relative_l2_strict_min": ORACLE_RANK3_MIN,
            "strong_baseline_to_oracle_ratio_min": BASELINE_TO_ORACLE_MIN,
            "strong_cross_channel_to_operator_norm_ratio_min": (
                CROSS_CHANNEL_STRENGTH_MIN
            ),
        },
        "replicates": replicates,
        "aggregate": aggregate,
        "all_reports_passed": passed,
        "verdict": (
            "rank_four_ceiling_earned" if passed else "rank_four_ceiling_not_earned"
        ),
        "source_hashes": sorted(
            {report["source"]["relevant_source_sha256"] for report in reports}
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
