# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the drifted screened principal-part study."""

from __future__ import annotations

import math

import drifted_screened_principal_part as study
import torch

DEVICE = torch.device("cpu")


def test_regular_field_satisfies_drifted_screened_operator() -> None:
    points = torch.tensor(
        ((0.2, 0.1), (-0.4, 0.3), (0.1, -0.6)),
        dtype=torch.float64,
        requires_grad=True,
    )
    kappa = 1.3
    drift = torch.tensor((0.7, -0.4), dtype=torch.float64)
    values = study.regular_drifted_screened_field(
        points,
        kappa=kappa,
        drift=drift,
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
    residual = laplacian + (gradient * drift).sum(dim=-1) - kappa**2 * values
    assert float(residual.detach().abs().max()) < 1.0e-10


def test_exact_scaled_kernel_has_universal_principal_limit() -> None:
    radius = torch.full((4,), 1.0e-7, dtype=torch.float64)
    normal_dot = torch.tensor((-1.0, -0.3, 0.4, 1.0), dtype=torch.float64)
    drift_dot = torch.tensor((2.0, -1.0, 0.5, -2.0), dtype=torch.float64)
    kappa = torch.tensor((0.1, 1.0, 3.0, 5.0), dtype=torch.float64)
    drift_magnitude = torch.tensor((2.0, 1.0, 0.5, 2.5), dtype=torch.float64)

    actual = study.exact_scaled_double_kernel(
        radius, normal_dot, drift_dot, kappa, drift_magnitude
    )
    expected = study.DOUBLE_COEFFICIENT * normal_dot
    assert torch.allclose(actual, expected, atol=5.0e-8, rtol=5.0e-8)


def test_fixed_arm_enforces_limit_without_constraining_remainder() -> None:
    torch.manual_seed(5)
    model = study.ScaledKernelModel("fixed_principal").double()
    radius = torch.full((8,), 1.0e-9, dtype=torch.float64)
    normal_dot = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    drift_dot = torch.linspace(-2.0, 2.0, 8, dtype=torch.float64)
    kappa = torch.full((8,), 4.0, dtype=torch.float64)
    drift_magnitude = torch.full((8,), 2.5, dtype=torch.float64)

    actual = model(radius, normal_dot, drift_dot, kappa, drift_magnitude)
    expected = study.DOUBLE_COEFFICIENT * normal_dot
    assert torch.allclose(actual, expected, atol=1.0e-8, rtol=1.0e-8)


def test_resolution_sampling_preserves_continuum_problem() -> None:
    first = study.build_pde_sample(
        123,
        split="ood_high_drift",
        n_boundary=32,
        n_query=64,
        device=DEVICE,
    )
    second = study.build_pde_sample(
        123,
        split="ood_high_drift",
        n_boundary=64,
        n_query=64,
        device=DEVICE,
    )
    assert first.kappa == second.kappa
    assert torch.equal(first.drift, second.drift)
    assert torch.equal(first.query_points, second.query_points)
    assert torch.equal(first.target, second.target)


def test_exact_gauge_transformed_layer_reproduces_manufactured_field() -> None:
    sample = study.build_pde_sample(
        456,
        split="in_distribution",
        n_boundary=96,
        n_query=96,
        device=DEVICE,
    )
    trace = study.trace_matrix(
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=24,
        model=None,
    )
    field = study.double_layer_influence(
        sample.query_points,
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=24,
        model=None,
    )
    density = torch.linalg.solve(trace, sample.boundary_values)
    error = study.relative_l2(field @ density, sample.target)
    assert error < 0.01
    assert math.isfinite(float(torch.linalg.cond(trace)))


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="fixed_principal",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_batch_size=32,
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
    assert report["arm"] == "fixed_principal"
    assert set(report["kernel_evaluation"]) == set(study.KERNEL_SPLITS)
    assert set(report["pde_evaluation"]) == set(study.PDE_SPLIT_ORDER)
    assert set(report["resolution_evaluation"]) == set(study.RESOLUTION_SPLITS)


def test_registered_protocol_constants() -> None:
    assert study.ARMS == (
        "fixed_principal",
        "free_principal",
        "fully_learned",
    )
    assert study.SEEDS == (17, 29, 43, 59, 71)
    assert study.TRAIN_STEPS == 8_000
    assert study.RESOLUTIONS == (64, 128, 256)
