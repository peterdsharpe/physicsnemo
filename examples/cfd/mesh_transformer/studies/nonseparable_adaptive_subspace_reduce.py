# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the frozen moving-subspace realizability census."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_adaptive_subspace as study

FORMAL_PROFILE_SEEDS = (263_000_239, 269_000_241, 271_000_247, 277_000_251)
SMOOTH_NONINFERIORITY_MAX = 1.20
SHIFT_IMPROVEMENT_RATIO_MAX = 0.75
RESOLUTION_DRIFT_MAX = 1.20
ORACLE_ERROR_MAX = 0.15


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _metric(report: dict[str, Any], split: str, arm: str) -> float:
    return report["split_evaluation"][split]["metrics"][arm][
        "cross_channel_relative_l2"
    ]["mean"]


def _load_reports(input_dir: Path) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("profile_seed*.json")):
        with path.open() as stream:
            report = json.load(stream)
        if report.get("study") != study.STUDY:
            raise ValueError(f"{path}: unexpected study {report.get('study')!r}")
        seed = report["profile_seed"]
        if seed in reports:
            raise ValueError(f"duplicate profile seed {seed}")
        reports[seed] = report
    missing = sorted(set(FORMAL_PROFILE_SEEDS) - reports.keys())
    unexpected = sorted(reports.keys() - set(FORMAL_PROFILE_SEEDS))
    if missing or unexpected:
        raise ValueError(f"report mismatch: missing={missing}, unexpected={unexpected}")
    return reports


def reduce_reports(input_dir: Path) -> dict[str, Any]:
    reports = _load_reports(input_dir)
    batch_results = []
    for profile_seed in FORMAL_PROFILE_SEEDS:
        report = reports[profile_seed]
        oracle_valid = all(
            _metric(report, split, "oracle_rank4") <= ORACLE_ERROR_MAX
            for split in study.SPLITS
        )
        gates = {
            "low_within_20pct_of_fixed": (
                _metric(report, "low_16", "local_connected_rank4")
                <= SMOOTH_NONINFERIORITY_MAX * _metric(report, "low_16", "fixed_rank4")
            ),
            "high32_within_20pct_of_high16": (
                _metric(report, "high_32", "local_connected_rank4")
                <= RESOLUTION_DRIFT_MAX
                * _metric(report, "high_16", "local_connected_rank4")
            ),
        }
        for split in ("mid_16", "high_16", "high_32"):
            for baseline in (
                "fixed_rank4",
                "global_rank4",
                "local_naive_rank4",
            ):
                gates[f"{split}_beats_{baseline}_by_25pct"] = _metric(
                    report, split, "local_connected_rank4"
                ) <= SHIFT_IMPROVEMENT_RATIO_MAX * _metric(report, split, baseline)
        batch_results.append(
            {
                "profile_seed": profile_seed,
                "oracle_valid": oracle_valid,
                "gates": gates,
                "passed": oracle_valid and all(gates.values()),
            }
        )

    controls_valid = all(result["oracle_valid"] for result in batch_results)
    passed_batch_count = sum(result["passed"] for result in batch_results)
    adaptive_subspace_earned = controls_valid and passed_batch_count == len(
        FORMAL_PROFILE_SEEDS
    )
    if not controls_valid:
        verdict = "invalid_controls"
    elif adaptive_subspace_earned:
        verdict = "adaptive_subspace_earned"
    else:
        verdict = "adaptive_subspace_not_earned"

    aggregate = {
        arm: {
            split: {
                "cross_channel_relative_l2_geomean": _geometric_mean(
                    [
                        _metric(reports[seed], split, arm)
                        for seed in FORMAL_PROFILE_SEEDS
                    ]
                )
            }
            for split in study.SPLITS
        }
        for arm in study.ARMS
    }
    return {
        "study": study.STUDY,
        "formal_profile_seeds": list(FORMAL_PROFILE_SEEDS),
        "registered_thresholds": {
            "smooth_noninferiority_max": SMOOTH_NONINFERIORITY_MAX,
            "shift_improvement_ratio_max": SHIFT_IMPROVEMENT_RATIO_MAX,
            "resolution_drift_max": RESOLUTION_DRIFT_MAX,
            "oracle_error_max": ORACLE_ERROR_MAX,
            "required_batch_count": len(FORMAL_PROFILE_SEEDS),
        },
        "batch_results": batch_results,
        "controls_valid": controls_valid,
        "passed_batch_count": passed_batch_count,
        "adaptive_subspace_earned": adaptive_subspace_earned,
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
