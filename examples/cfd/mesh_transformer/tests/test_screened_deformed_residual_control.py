# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for deformed screened residual control."""

from __future__ import annotations

import screened_deformed_residual_control as study
import torch

DEVICE = torch.device("cpu")


def test_regular_bessel_field_satisfies_screened_laplace() -> None:
    points = torch.tensor(
        ((0.2, 0.1), (-0.4, 0.3), (0.1, -0.6)),
        dtype=torch.float64,
        requires_grad=True,
    )
    kappa = 1.7
    values = study.regular_screened_field(
        points,
        kappa_tilde=kappa,
        modes=(0, 1, 2, 3),
        phases=torch.tensor((0.2, 0.5, 1.1, 2.0), dtype=torch.float64),
    )
    (gradient,) = torch.autograd.grad(values.sum(), points, create_graph=True)
    laplacian = torch.zeros_like(values)
    for component in range(2):
        (second,) = torch.autograd.grad(
            gradient[:, component].sum(), points, create_graph=True
        )
        laplacian = laplacian + second[:, component]

    residual = laplacian - kappa**2 * values
    assert float(residual.detach().abs().max()) < 1.0e-10


def test_resolution_sampling_preserves_geometry_queries_and_target() -> None:
    first = study.build_deformed_sample(
        123,
        split="stronger_deformation",
        n_boundary=32,
        n_query=64,
        device=DEVICE,
    )
    second = study.build_deformed_sample(
        123,
        split="stronger_deformation",
        n_boundary=64,
        n_query=64,
        device=DEVICE,
    )

    assert first.kappa_tilde == second.kappa_tilde
    assert first.deformation == second.deformation
    assert torch.equal(first.query_points, second.query_points)
    assert torch.equal(first.target, second.target)


def test_one_deformed_case_converges_to_the_dense_field() -> None:
    result = study.evaluate_case(
        seed=456,
        split="unseen_geometry_modes",
        n_boundary=64,
        n_query=64,
        device=DEVICE,
        quadrature_order=32,
        check_quadrature_order=16,
    )

    assert result["converged"] is True
    assert result["iterations"] <= study.MAX_ITERATIONS
    assert result["stopping_trace_relative_l2"] <= study.TRACE_TOLERANCE
    assert result["dense_trace_relative_l2"] < 1.0e-12
    assert result["iterative_minus_dense_relative_target_l2"] < 0.001


def test_small_study_preserves_schema_without_registered_verdict() -> None:
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


def test_registered_protocol_constants() -> None:
    assert study.EVALUATION_CASES == 32
    assert study.EVALUATION_BOUNDARY_POINTS == 128
    assert study.EVALUATION_QUERY_POINTS == 512
    assert study.RESOLUTION_CASES == 8
    assert study.RESOLUTIONS == (64, 128, 256)
    assert study.TRACE_TOLERANCE == 1.0e-6
    assert study.MAX_ITERATIONS == 32
