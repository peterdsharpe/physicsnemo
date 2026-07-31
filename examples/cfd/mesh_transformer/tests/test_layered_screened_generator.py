# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for path-ordered learned local generators."""

from __future__ import annotations

import layered_screened_context as base
import layered_screened_generator as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _random_inputs(samples: int = 128):
    generator = torch.Generator().manual_seed(19)
    profiles = 0.25 + 2.75 * torch.rand(
        samples,
        base.N_LAYERS,
        generator=generator,
        dtype=DTYPE,
    )
    modes = torch.randint(0, 8, (samples,), generator=generator).to(DTYPE)
    query_x = torch.rand(samples, generator=generator, dtype=DTYPE)
    sides = torch.randint(0, 2, (samples,), generator=generator)
    return profiles, modes, query_x, sides


def test_response_from_true_wavenumber_matches_reference() -> None:
    profiles, modes, query_x, sides = _random_inputs()
    q = base.mode_wavenumber(profiles, modes)
    predicted = study.response_from_wavenumber(q, query_x, sides)
    reference = base.exact_mode_response(profiles, modes, query_x, sides)
    assert torch.equal(predicted, reference)


def test_learned_generator_preserves_both_boundary_values() -> None:
    profiles, modes, _, _ = _random_inputs()
    zeros = torch.zeros(len(profiles), dtype=DTYPE)
    ones = torch.ones(len(profiles), dtype=DTYPE)
    left = torch.zeros(len(profiles), dtype=torch.long)
    right = torch.ones(len(profiles), dtype=torch.long)
    model = study.LearnedLocalGenerator(physical_order=True).to(dtype=DTYPE)
    assert torch.allclose(model(profiles, modes, zeros, left), ones, atol=1.0e-12)
    assert torch.allclose(model(profiles, modes, ones, left), zeros, atol=1.0e-12)
    assert torch.allclose(model(profiles, modes, zeros, right), zeros, atol=1.0e-12)
    assert torch.allclose(model(profiles, modes, ones, right), ones, atol=1.0e-12)


def test_sorted_generator_is_exactly_blind_to_order_challenge() -> None:
    profile_a, profile_b, modes, query_x, sides = base.order_challenge_profiles(
        128,
        device=DEVICE,
        dtype=DTYPE,
    )
    model = study.LearnedLocalGenerator(physical_order=False).to(dtype=DTYPE)
    assert torch.equal(
        model(profile_a, modes, query_x, sides),
        model(profile_b, modes, query_x, sides),
    )


def test_physical_and_sorted_generators_have_matched_capacity() -> None:
    physical = sum(
        parameter.numel()
        for parameter in study.LearnedLocalGenerator(physical_order=True).parameters()
    )
    sorted_count = sum(
        parameter.numel()
        for parameter in study.LearnedLocalGenerator(physical_order=False).parameters()
    )
    assert physical == sorted_count == 4_417
    assert (
        sum(parameter.numel() for parameter in study.AnalyticPath().parameters()) == 0
    )


def test_analytic_oracle_has_zero_evaluation_error() -> None:
    model = study.AnalyticPath().to(dtype=DTYPE)
    evaluation = base.evaluate_split(
        model,
        "in_distribution",
        n_profiles=2,
        n_query=8,
        device=DEVICE,
    )
    assert evaluation["metrics"]["field_relative_l2"]["mean"] == 0.0
    order = base.evaluate_order_challenge(model, n_pairs=32, device=DEVICE)
    assert order["paired_field_relative_l2"] == 0.0
    assert order["contrast_relative_l2"] == 0.0


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="path_generator",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_batch_size=32,
        evaluation_profiles=2,
        evaluation_query_points=8,
        order_pairs=32,
    )
    assert report["study"] == study.STUDY
    assert report["arm"] == "path_generator"
    assert report["parameters"] == 4_417
    assert set(report["split_evaluation"]) == set(base.SPLITS)
    assert report["order_challenge"]["scalar_input_max_abs_difference"] == 0.0
