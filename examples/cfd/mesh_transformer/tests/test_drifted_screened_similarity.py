# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the drifted-screened similarity-coordinate study."""

from __future__ import annotations

import drifted_screened_principal_part as base
import drifted_screened_similarity as study
import torch

DEVICE = torch.device("cpu")


def test_similarity_features_are_sufficient_for_the_exact_scaled_kernel() -> None:
    radius = torch.tensor((0.01, 0.4, 2.0), dtype=torch.float64)
    normal_dot_direction = torch.tensor((-0.7, 0.2, 0.9), dtype=torch.float64)
    drift_dot_direction = torch.tensor((0.3, -0.8, 1.7), dtype=torch.float64)
    kappa = torch.tensor((0.1, 1.2, 4.5), dtype=torch.float64)
    drift_magnitude = torch.tensor((0.5, 0.9, 2.1), dtype=torch.float64)

    features = base.kernel_features(
        radius,
        normal_dot_direction,
        drift_dot_direction,
        kappa,
        drift_magnitude,
        feature_system="similarity",
    )
    eta = 6.25 * features[:, 3]
    scaled_radius = 12.5 * features[:, 4]
    reconstructed = (
        base.DOUBLE_COEFFICIENT
        * torch.exp(-0.5 * eta)
        * scaled_radius
        * torch.special.modified_bessel_k1(scaled_radius)
        * features[:, 2]
    )
    exact = base.exact_scaled_double_kernel(
        radius,
        normal_dot_direction,
        drift_dot_direction,
        kappa,
        drift_magnitude,
    )
    assert torch.allclose(reconstructed, exact, atol=1.0e-14, rtol=1.0e-14)
    assert torch.allclose(8.0 * features[:, 5], torch.log(scaled_radius))


def test_raw_features_remain_backward_compatible() -> None:
    radius = torch.tensor((0.2, 1.1), dtype=torch.float64)
    normal = torch.tensor((0.4, -0.6), dtype=torch.float64)
    directional_drift = torch.tensor((0.8, -1.2), dtype=torch.float64)
    kappa = torch.tensor((0.7, 1.8), dtype=torch.float64)
    drift = torch.tensor((0.4, 0.9), dtype=torch.float64)
    expected = torch.stack(
        (
            torch.log(radius) / 8.0,
            radius / 3.0,
            normal,
            directional_drift / 2.5,
            kappa / 5.0,
            drift / 2.5,
        ),
        dim=-1,
    )
    actual = base.kernel_features(
        radius,
        normal,
        directional_drift,
        kappa,
        drift,
    )
    assert torch.equal(actual, expected)


def test_coordinate_arms_have_identical_capacity_and_initialization() -> None:
    torch.manual_seed(17)
    raw = base.ScaledKernelModel("free_principal", feature_system="raw")
    torch.manual_seed(17)
    similarity = base.ScaledKernelModel("free_principal", feature_system="similarity")
    assert sum(parameter.numel() for parameter in raw.parameters()) == 8_834
    assert sum(parameter.numel() for parameter in similarity.parameters()) == 8_834
    for raw_parameter, similarity_parameter in zip(
        raw.parameters(), similarity.parameters(), strict=True
    ):
        assert torch.equal(raw_parameter, similarity_parameter)


def test_small_run_preserves_factorial_schema() -> None:
    report = study.run_arm(
        arm="similarity_hybrid",
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
    assert report["arm"] == "similarity_hybrid"
    assert report["feature_system"] == "similarity"
    assert report["loss"] == "hybrid"
    assert set(report["pde_evaluation"]) == set(base.PDE_SPLIT_ORDER)


def test_registered_protocol_constants() -> None:
    assert study.ARMS == (
        "raw_pointwise",
        "similarity_pointwise",
        "raw_hybrid",
        "similarity_hybrid",
    )
    assert study.SEEDS == (17, 29, 43, 59, 71)
    assert study.TRAIN_STEPS == 4_000
    assert study.RESOLUTIONS == (64, 128, 256)
