# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the canonical screened double-layer study."""

from __future__ import annotations

import screened_canonical_double_layer as study
import torch
from screened_laplace import SPLITS, build_screened_sample

DEVICE = torch.device("cpu")


def test_resolution_sampling_preserves_the_continuum_problem() -> None:
    first = build_screened_sample(
        123,
        **SPLITS["in_distribution"],
        n_boundary=16,
        n_query=32,
        device=DEVICE,
        dtype=torch.float64,
    )
    second = build_screened_sample(
        123,
        **SPLITS["in_distribution"],
        n_boundary=32,
        n_query=32,
        device=DEVICE,
        dtype=torch.float64,
    )

    assert first.kappa_tilde == second.kappa_tilde
    assert torch.equal(first.domain.interior.points, second.domain.interior.points)
    assert torch.equal(first.target, second.target)


def test_one_corrected_case_tracks_the_dense_trace() -> None:
    result = study.evaluate_case(
        seed=456,
        split="ood_low_screening",
        n_boundary=32,
        n_query=64,
        device=DEVICE,
        quadrature_order=32,
        check_quadrature_order=16,
    )

    assert result["dense_trace_relative_l2"] < 1.0e-12
    assert result["iterative_trace_relative_l2"] < 0.01
    assert result["iterative_minus_dense_relative_target_l2"] < 0.01
    assert result["unit_richardson_spectral_radius"] < 0.75


def test_small_study_preserves_schema_without_issuing_registered_verdict() -> None:
    report = study.run_study(
        device=DEVICE,
        evaluation_cases=1,
        evaluation_boundary_points=16,
        evaluation_query_points=16,
        resolution_cases=1,
        resolutions=(16, 32),
        quadrature_order=16,
        check_quadrature_order=8,
    )

    assert report["study"] == study.STUDY
    assert set(report["evaluation"]) == set(study.SPLIT_ORDER)
    assert set(report["resolution"]) == set(study.SPLIT_ORDER)
    assert "decision" not in report
    assert all(
        len(report["evaluation"][split]["cases"]) == 1 for split in study.SPLIT_ORDER
    )


def test_registered_protocol_constants() -> None:
    assert study.EVALUATION_CASES == 64
    assert study.EVALUATION_BOUNDARY_POINTS == 64
    assert study.EVALUATION_QUERY_POINTS == 512
    assert study.RESOLUTION_CASES == 8
    assert study.RESOLUTIONS == (64, 128, 256)
    assert study.QUADRATURE_ORDER == 64
    assert study.CHECK_QUADRATURE_ORDER == 32
    assert study.RICHARDSON_STEPS == 8
