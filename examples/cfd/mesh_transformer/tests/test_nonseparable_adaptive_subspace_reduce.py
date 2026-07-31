# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen moving-subspace census reducer."""

from __future__ import annotations

import json
from pathlib import Path

import nonseparable_adaptive_subspace as study
import nonseparable_adaptive_subspace_reduce as reducer


def _report(profile_seed: int, *, connected_error: float = 0.25) -> dict:
    arm_errors = {
        "diagonal_carrier": 1.0,
        "fixed_rank4": 0.40,
        "global_rank4": 0.50,
        "local_naive_rank4": 0.60,
        "local_connected_rank4": connected_error,
        "oracle_rank4": 0.10,
    }
    return {
        "study": study.STUDY,
        "profile_seed": profile_seed,
        "split_evaluation": {
            split: {
                "metrics": {
                    arm: {
                        "cross_channel_relative_l2": {"mean": error},
                    }
                    for arm, error in arm_errors.items()
                }
            }
            for split in study.SPLITS
        },
        "source": {"relevant_source_sha256": "abc"},
    }


def _write_case(
    root: Path,
    *,
    connected_error: float = 0.25,
    oracle_error: float = 0.10,
) -> None:
    for seed in reducer.FORMAL_PROFILE_SEEDS:
        report = _report(seed, connected_error=connected_error)
        for split in study.SPLITS:
            report["split_evaluation"][split]["metrics"]["oracle_rank4"][
                "cross_channel_relative_l2"
            ]["mean"] = oracle_error
        (root / f"profile_seed{seed}.json").write_text(json.dumps(report))


def test_reducer_accepts_adaptive_subspace(tmp_path: Path) -> None:
    _write_case(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    summary = reducer.reduce_reports(tmp_path)
    assert summary["adaptive_subspace_earned"]
    assert summary["verdict"] == "adaptive_subspace_earned"


def test_reducer_rejects_weak_adaptive_subspace(tmp_path: Path) -> None:
    _write_case(tmp_path, connected_error=0.35)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["adaptive_subspace_earned"]
    assert summary["verdict"] == "adaptive_subspace_not_earned"


def test_reducer_rejects_invalid_oracle(tmp_path: Path) -> None:
    _write_case(tmp_path, oracle_error=0.20)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["controls_valid"]
    assert summary["verdict"] == "invalid_controls"
