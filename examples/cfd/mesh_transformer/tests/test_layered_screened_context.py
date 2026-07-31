# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the layered variable-coefficient context study."""

from __future__ import annotations

import layered_screened_context as study
import numpy as np
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _challenge_inputs(n_pairs: int = 128):
    return study.order_challenge_profiles(
        n_pairs,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_layer_transfer_has_unit_determinant() -> None:
    q = torch.linspace(0.1, 8.0, 100, dtype=DTYPE)
    lengths = torch.linspace(0.0, 1.0, 100, dtype=DTYPE)
    determinant = torch.linalg.det(study.layer_transfer(q, lengths))
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=1.0e-12)


def test_optical_carrier_is_exact_for_constant_profiles() -> None:
    generator = torch.Generator().manual_seed(7)
    samples = 256
    coefficient = 0.25 + 2.75 * torch.rand(
        samples,
        1,
        generator=generator,
        dtype=DTYPE,
    )
    profiles = coefficient.expand(-1, study.N_LAYERS)
    modes = torch.randint(0, 8, (samples,), generator=generator).to(DTYPE)
    query_x = torch.rand(samples, generator=generator, dtype=DTYPE)
    sides = torch.randint(0, 2, (samples,), generator=generator)
    exact = study.exact_mode_response(profiles, modes, query_x, sides)
    carrier = study.fixed_optical_response(profiles, modes, query_x, sides)
    assert torch.allclose(exact, carrier, atol=2.0e-13, rtol=2.0e-13)


def test_exact_response_satisfies_both_boundary_contracts() -> None:
    generator = torch.Generator().manual_seed(11)
    profiles = 0.25 + 2.75 * torch.rand(
        128,
        study.N_LAYERS,
        generator=generator,
        dtype=DTYPE,
    )
    modes = torch.randint(0, 8, (128,), generator=generator).to(DTYPE)
    zeros = torch.zeros(128, dtype=DTYPE)
    ones = torch.ones(128, dtype=DTYPE)
    left = torch.zeros(128, dtype=torch.long)
    right = torch.ones(128, dtype=torch.long)
    assert torch.allclose(
        study.exact_mode_response(profiles, modes, zeros, left),
        ones,
        atol=2.0e-13,
    )
    assert torch.allclose(
        study.exact_mode_response(profiles, modes, ones, left),
        zeros,
        atol=2.0e-13,
    )
    assert torch.allclose(
        study.exact_mode_response(profiles, modes, zeros, right),
        zeros,
        atol=2.0e-13,
    )
    assert torch.allclose(
        study.exact_mode_response(profiles, modes, ones, right),
        ones,
        atol=2.0e-13,
    )


def test_transfer_reference_matches_an_independent_finite_difference_solve() -> None:
    profile = np.array([0.8, 0.3, 2.7, 1.1, 1.8, 0.6, 2.2, 1.4])
    mode = 2
    intervals = 4_096
    dx = 1.0 / intervals
    x = np.arange(1, intervals) * dx
    layer = np.minimum((x * study.N_LAYERS).astype(int), study.N_LAYERS - 1)
    q2 = mode**2 + profile[layer] ** 2
    diagonal = 2.0 + dx**2 * q2
    lower = np.full(intervals - 2, -1.0)
    upper = np.full(intervals - 2, -1.0)
    rhs = np.zeros(intervals - 1)
    rhs[0] = 1.0

    for index in range(1, len(diagonal)):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]
    interior = np.empty_like(rhs)
    interior[-1] = rhs[-1] / diagonal[-1]
    for index in range(len(interior) - 2, -1, -1):
        interior[index] = (rhs[index] - upper[index] * interior[index + 1]) / diagonal[
            index
        ]

    query_index = intervals // 2
    finite_difference = interior[query_index - 1]
    transfer = study.exact_mode_response(
        torch.tensor(profile[None, :], dtype=DTYPE),
        torch.tensor([mode], dtype=DTYPE),
        torch.tensor([0.5], dtype=DTYPE),
        torch.tensor([0]),
    )
    assert abs(float(transfer) - finite_difference) < 1.0e-5


def test_order_challenge_is_identical_to_scalar_features_but_not_to_physics() -> None:
    profile_a, profile_b, modes, query_x, sides = _challenge_inputs()
    features_a = study.scalar_features(profile_a, modes, query_x, sides)
    features_b = study.scalar_features(profile_b, modes, query_x, sides)
    assert torch.equal(features_a, features_b)

    target_a = study.exact_mode_response(profile_a, modes, query_x, sides)
    target_b = study.exact_mode_response(profile_b, modes, query_x, sides)
    contrast = torch.sqrt(
        2.0
        * (target_a - target_b).square().sum()
        / (target_a.square() + target_b.square()).sum()
    )
    assert float(contrast) > 0.02

    optical = study.FixedOptical()
    assert torch.equal(
        optical(profile_a, modes, query_x, sides),
        optical(profile_b, modes, query_x, sides),
    )
    tokens_a = study.token_features(profile_a, modes, query_x, sides)
    tokens_b = study.token_features(profile_b, modes, query_x, sides)
    assert not torch.equal(tokens_a, tokens_b)


def test_learned_arm_capacities_are_matched_as_registered() -> None:
    scalar = sum(
        parameter.numel() for parameter in study.ScalarCorrection().parameters()
    )
    carrier = sum(
        parameter.numel()
        for parameter in study.OrderedCorrection(carrier=True).parameters()
    )
    raw = sum(
        parameter.numel()
        for parameter in study.OrderedCorrection(carrier=False).parameters()
    )
    assert carrier == raw == 11_073
    assert scalar == 11_041
    assert abs(scalar / carrier - 1.0) < 0.01
    assert (
        sum(parameter.numel() for parameter in study.FixedOptical().parameters()) == 0
    )


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="ordered_carrier",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_batch_size=32,
        evaluation_profiles=2,
        evaluation_query_points=8,
        order_pairs=32,
    )
    assert report["study"] == study.STUDY
    assert report["arm"] == "ordered_carrier"
    assert report["parameters"] == 11_073
    assert set(report["split_evaluation"]) == set(study.SPLITS)
    assert report["order_challenge"]["scalar_input_max_abs_difference"] == 0.0
    assert report["reference_certification"]["boundary_max_abs_error"] < 1.0e-10
