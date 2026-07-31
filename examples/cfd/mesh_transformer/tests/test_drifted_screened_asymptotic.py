# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the two-limit asymptotic-carrier study."""

from __future__ import annotations

import math

import drifted_screened_asymptotic as study
import drifted_screened_principal_part as base
import torch

DEVICE = torch.device("cpu")


def test_carrier_has_the_registered_small_and_large_limits() -> None:
    small = torch.tensor(1.0e-5, dtype=torch.float64)
    large = torch.tensor(40.0, dtype=torch.float64)
    small_carrier = study.matched_screening_carrier(small)
    large_carrier = study.matched_screening_carrier(large)
    large_limit = torch.sqrt(0.5 * math.pi * large) * torch.exp(-large)
    assert torch.allclose(small_carrier, torch.ones_like(small), atol=1.0e-9)
    assert torch.allclose(large_carrier / large_limit, torch.ones_like(large))


def test_carrier_is_not_the_exact_bessel_response() -> None:
    scaled_radius = torch.tensor(1.0, dtype=torch.float64)
    carrier = study.matched_screening_carrier(scaled_radius)
    exact = scaled_radius * torch.special.modified_bessel_k1(scaled_radius)
    assert abs(float(carrier / exact) - 1.0) > 0.05


def test_correction_window_is_bounded_and_vanishes_at_both_limits() -> None:
    scaled_radius = torch.logspace(-6, 6, 20_000, dtype=torch.float64)
    window = study.correction_window(scaled_radius)
    assert float(window.max()) < 0.6
    assert float(window[0]) < 1.0e-10
    assert float(window[-1]) < 1.0e-5


def test_learned_carrier_matches_raw_capacity() -> None:
    raw = base.ScaledKernelModel("free_principal")
    learned = study.LearnedCarrier()
    assert sum(parameter.numel() for parameter in raw.parameters()) == 8_834
    assert sum(parameter.numel() for parameter in learned.parameters()) == 8_834
    assert (
        sum(parameter.numel() for parameter in study.FixedCarrier().parameters()) == 0
    )


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="learned_carrier",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_boundary_points=12,
        train_query_points=12,
        train_quadrature_order=4,
        evaluation_cases=1,
        evaluation_boundary_points=16,
        evaluation_query_points=16,
        resolution_cases=1,
        resolutions=(16, 24),
        quadrature_order=4,
        check_quadrature_order=4,
        kernel_evaluation_pairs=64,
    )
    assert report["study"] == study.STUDY
    assert report["arm"] == "learned_carrier"
    assert report["parameters"] == 8_834
    assert report["protocol"]["training_applied"] is True
    assert set(report["pde_evaluation"]) == set(base.PDE_SPLIT_ORDER)


def test_fixed_carrier_reports_that_training_was_not_applied() -> None:
    report = study.run_arm(
        arm="fixed_carrier",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        evaluation_cases=1,
        evaluation_boundary_points=16,
        evaluation_query_points=16,
        resolution_cases=1,
        resolutions=(16, 24),
        quadrature_order=4,
        check_quadrature_order=4,
        kernel_evaluation_pairs=64,
    )
    assert report["parameters"] == 0
    assert report["training_history"] == []
    assert report["protocol"]["training_applied"] is False
    assert report["protocol"]["train_steps"] == 0


def test_registered_protocol_constants() -> None:
    assert study.ARMS == ("raw_hybrid", "fixed_carrier", "learned_carrier")
    assert study.SEEDS == (17, 29, 43, 59, 71)
    assert study.TRAIN_STEPS == 4_000
    assert study.RESOLUTIONS == (64, 128, 256)
