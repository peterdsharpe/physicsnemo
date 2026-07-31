# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen coefficient-frequency coverage reducer."""

from __future__ import annotations

import json
from pathlib import Path

import nonseparable_frequency_coverage as study
import nonseparable_frequency_coverage_reduce as reducer


def _report(arm: str, seed: int, *, default_error: float) -> dict:
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "protocol": {"evaluation_seed": reducer.FORMAL_EVALUATION_SEED},
        "split_evaluation": {
            split: {
                "metrics": {
                    "cross_channel_relative_l2": {"mean": default_error},
                }
            }
            for split in study.SPLITS
        },
        "source": {"relevant_source_sha256": "abc"},
    }


def _set_metric(report: dict, split: str, value: float) -> None:
    report["split_evaluation"][split]["metrics"]["cross_channel_relative_l2"][
        "mean"
    ] = value


def _write_case(
    root: Path,
    *,
    broad_covered_error: float = 0.22,
    broad_unseen_error: float = 0.22,
) -> None:
    for seed in reducer.FORMAL_SEEDS:
        narrow = _report("narrow_flow", seed, default_error=0.60)
        broad = _report("broad_flow", seed, default_error=0.22)
        _set_metric(broad, "covered_fast_16", broad_covered_error)
        _set_metric(broad, "unseen_16", broad_unseen_error)
        _set_metric(broad, "unseen_32", broad_unseen_error)
        for arm, report in (("narrow_flow", narrow), ("broad_flow", broad)):
            (root / f"{arm}_seed{seed}.json").write_text(json.dumps(report))
    analytic_seed = reducer.FORMAL_SEEDS[0]
    for arm, error in (
        ("diagonal_carrier", 1.0),
        ("fixed_rank4", 0.20),
        ("oracle_rank4", 0.10),
    ):
        report = _report(arm, analytic_seed, default_error=error)
        (root / f"{arm}_seed{analytic_seed}.json").write_text(json.dumps(report))


def test_reducer_accepts_coverage_and_local_law(tmp_path: Path) -> None:
    _write_case(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    summary = reducer.reduce_reports(tmp_path)
    assert summary["coverage_earned"]
    assert summary["local_law_earned"]
    assert summary["verdict"] == "coverage_and_local_law_earned"


def test_reducer_separates_coverage_from_local_law(tmp_path: Path) -> None:
    _write_case(tmp_path, broad_unseen_error=0.40)
    summary = reducer.reduce_reports(tmp_path)
    assert summary["coverage_earned"]
    assert not summary["local_law_earned"]
    assert summary["verdict"] == "coverage_earned_local_law_refuted"


def test_reducer_rejects_failed_coverage(tmp_path: Path) -> None:
    _write_case(tmp_path, broad_covered_error=0.40)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["coverage_earned"]
    assert summary["verdict"] == "coverage_not_earned"
