# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen nonseparable continuum-flow reducer."""

from __future__ import annotations

import json
from pathlib import Path

import nonseparable_continuum_flow as study
import nonseparable_continuum_flow_reduce as reducer


def _report(
    arm: str,
    seed: int,
    *,
    default_error: float,
    predicted_contrast: float = 0.2,
) -> dict:
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "protocol": {
            "evaluation_seed": reducer.FORMAL_EVALUATION_SEED,
            "order_seed": reducer.FORMAL_ORDER_SEED,
        },
        "split_evaluation": {
            split: {
                "metrics": {
                    "cross_channel_relative_l2": {"mean": default_error},
                }
            }
            for split in study.SPLITS
        },
        "order_challenge": {
            "contrast_recovery_fraction": 0.6,
            "predicted_contrast_relative_l2": predicted_contrast,
        },
        "source": {"relevant_source_sha256": "abc"},
    }


def _write_case(
    root: Path,
    *,
    flow_fast_error: float = 0.24,
    flow_resolution_error: float = 0.24,
) -> None:
    for seed in reducer.FORMAL_SEEDS:
        reports = {
            "gru_path_rank4": _report(
                "gru_path_rank4",
                seed,
                default_error=0.60,
            ),
            "sorted_flow_rank4": _report(
                "sorted_flow_rank4",
                seed,
                default_error=0.60,
                predicted_contrast=0.0,
            ),
            "path_flow_rank4": _report(
                "path_flow_rank4",
                seed,
                default_error=0.24,
            ),
        }
        path = reports["path_flow_rank4"]
        for split in ("coarse_4", "refined_16", "refined_32"):
            path["split_evaluation"][split]["metrics"]["cross_channel_relative_l2"][
                "mean"
            ] = flow_resolution_error
        for split in ("fast_8", "fast_16"):
            path["split_evaluation"][split]["metrics"]["cross_channel_relative_l2"][
                "mean"
            ] = flow_fast_error
        for arm, report in reports.items():
            (root / f"{arm}_seed{seed}.json").write_text(json.dumps(report))
    analytic_seed = reducer.FORMAL_SEEDS[0]
    for arm, error in (
        ("diagonal_carrier", 1.0),
        ("fixed_rank4", 0.22),
        ("oracle_rank4", 0.10),
    ):
        report = _report(arm, analytic_seed, default_error=error)
        (root / f"{arm}_seed{analytic_seed}.json").write_text(json.dumps(report))


def test_reducer_accepts_both_claims(tmp_path: Path) -> None:
    _write_case(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    summary = reducer.reduce_reports(tmp_path)
    assert summary["continuum_earned"]
    assert summary["frequency_transfer_earned"]
    assert summary["verdict"] == "continuum_and_frequency_transfer_earned"


def test_reducer_separates_continuum_from_frequency(tmp_path: Path) -> None:
    _write_case(tmp_path, flow_fast_error=0.40)
    summary = reducer.reduce_reports(tmp_path)
    assert summary["continuum_earned"]
    assert not summary["frequency_transfer_earned"]
    assert summary["verdict"] == "continuum_earned_frequency_transfer_refuted"


def test_reducer_rejects_weak_continuum_flow(tmp_path: Path) -> None:
    _write_case(tmp_path, flow_resolution_error=0.40)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["continuum_earned"]
    assert summary["verdict"] == "continuum_flow_not_earned"
