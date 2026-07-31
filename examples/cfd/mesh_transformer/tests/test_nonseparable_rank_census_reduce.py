# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered nonseparable rank-census reduction."""

from __future__ import annotations

import json
from pathlib import Path

import nonseparable_rank_census_reduce as reducer
import pytest


def _report(seed: int, *, rank4_strong: float = 0.12) -> dict:
    levels = {}
    for level in reducer.LEVELS:
        rank4 = rank4_strong if level == reducer.STRONG_LEVEL else 0.11
        levels[level] = {
            "cross_channel_to_operator_norm_ratio": (
                0.005 if level == reducer.STRONG_LEVEL else 0.002
            ),
            "ranks": {
                "3": {
                    "oracle_cross_channel_relative_l2": {"mean": 0.24},
                },
                "4": {
                    "oracle_cross_channel_relative_l2": {"mean": rank4},
                    "fixed_truncation_cross_channel_relative_l2": {"mean": 0.30},
                    "global_eigenspace_cross_channel_relative_l2": {"mean": 0.31},
                },
            },
        }
    return {
        "study": reducer.STUDY,
        "protocol": {"seed": seed},
        "levels": levels,
        "source": {"relevant_source_sha256": "abc"},
    }


def _write_reports(root: Path, reports: list[dict]) -> None:
    for report in reports:
        path = root / f"seed{report['protocol']['seed']}.json"
        path.write_text(json.dumps(report))


def test_registered_gates_accept_four_independent_reports(tmp_path: Path) -> None:
    _write_reports(tmp_path, [_report(seed) for seed in (11, 13, 17, 19)])
    summary = reducer.reduce_reports(tmp_path)
    assert summary["all_reports_passed"]
    assert summary["verdict"] == "rank_four_ceiling_earned"
    assert all(replicate["passed"] for replicate in summary["replicates"])


def test_one_failed_report_rejects_the_ceiling(tmp_path: Path) -> None:
    reports = [_report(seed) for seed in (11, 13, 17, 19)]
    reports[-1] = _report(19, rank4_strong=0.16)
    _write_reports(tmp_path, reports)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["all_reports_passed"]
    assert summary["verdict"] == "rank_four_ceiling_not_earned"


def test_reducer_requires_four_unique_reports(tmp_path: Path) -> None:
    _write_reports(tmp_path, [_report(seed) for seed in (11, 13, 17)])
    with pytest.raises(ValueError, match="expected 4 reports"):
        reducer.reduce_reports(tmp_path)
