# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen nonseparable latent-composition reducer."""

from __future__ import annotations

import json
from pathlib import Path

import nonseparable_latent_composition as study
import nonseparable_latent_composition_reduce as reducer


def _report(
    arm: str,
    seed: int,
    *,
    cross_error: float,
    contrast_recovery: float,
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
                    "cross_channel_relative_l2": {"mean": cross_error},
                }
            }
            for split in study.SPLITS
        },
        "order_challenge": {
            "paired_residual_relative_l2": cross_error,
            "contrast_recovery_fraction": contrast_recovery,
            "predicted_contrast_relative_l2": predicted_contrast,
        },
        "source": {"relevant_source_sha256": "abc"},
    }


def _write_case(
    root: Path,
    *,
    path_error: float = 0.24,
    path_contrast: float = 0.60,
) -> None:
    for seed in reducer.FORMAL_SEEDS:
        reports = {
            "global_full": _report(
                "global_full",
                seed,
                cross_error=0.36,
                contrast_recovery=0.40,
            ),
            "global_rank4": _report(
                "global_rank4",
                seed,
                cross_error=0.40,
                contrast_recovery=0.40,
            ),
            "sorted_rank4": _report(
                "sorted_rank4",
                seed,
                cross_error=0.90,
                contrast_recovery=0.0,
                predicted_contrast=0.0,
            ),
            "path_rank4": _report(
                "path_rank4",
                seed,
                cross_error=path_error,
                contrast_recovery=path_contrast,
            ),
        }
        for arm, report in reports.items():
            (root / f"{arm}_seed{seed}.json").write_text(json.dumps(report))
    analytic_seed = reducer.FORMAL_SEEDS[0]
    for arm, error, contrast in (
        ("diagonal_carrier", 1.0, 0.0),
        ("fixed_rank4", 0.22, 0.65),
        ("oracle_rank4", 0.10, 0.85),
    ):
        report = _report(
            arm,
            analytic_seed,
            cross_error=error,
            contrast_recovery=contrast,
        )
        (root / f"{arm}_seed{analytic_seed}.json").write_text(json.dumps(report))


def test_reducer_accepts_both_claims(tmp_path: Path) -> None:
    _write_case(tmp_path)
    summary = reducer.reduce_reports(tmp_path)
    assert summary["ordered_compression_earned"]
    assert summary["breadth_earned"]
    assert summary["verdict"] == "ordered_compression_and_breadth_earned"


def test_reducer_separates_ordering_from_breadth(tmp_path: Path) -> None:
    _write_case(tmp_path)
    for path in tmp_path.glob("path_rank4_*.json"):
        report = json.loads(path.read_text())
        for split in ("fast_variation", "combined_shift"):
            report["split_evaluation"][split]["metrics"]["cross_channel_relative_l2"][
                "mean"
            ] = 0.35
        path.write_text(json.dumps(report))
    summary = reducer.reduce_reports(tmp_path)
    assert summary["ordered_compression_earned"]
    assert not summary["breadth_earned"]
    assert summary["verdict"] == "ordered_compression_earned_breadth_refuted"


def test_reducer_rejects_weak_ordering(tmp_path: Path) -> None:
    _write_case(tmp_path, path_error=0.34, path_contrast=0.45)
    summary = reducer.reduce_reports(tmp_path)
    assert not summary["ordered_compression_earned"]
    assert summary["verdict"] == "ordered_compression_not_earned"
