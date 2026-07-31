# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for coupled physical-order composition."""

from __future__ import annotations

import coupled_layered_composition as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(
    n_profiles: int = 16,
    *,
    n_layers: int = study.TRAIN_LAYERS,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(23)
    return study.sample_profiles(
        n_profiles,
        n_layers=n_layers,
        twist_frequencies=study.TRAIN_TWISTS,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_profiles_are_symmetric_positive_definite_and_coupled() -> None:
    profiles = _profiles()
    eigenvalues = torch.linalg.eigvalsh(profiles)
    assert torch.all(eigenvalues > 0.0)
    assert torch.equal(profiles, profiles.transpose(-2, -1))
    assert float(profiles[..., 0, 1].abs().max()) > 0.1


def test_generator_commutator_matches_registered_formula() -> None:
    first, second = _profiles(2)[:, 0]
    a_first = study.generator_matrix(first)
    a_second = study.generator_matrix(second)
    measured = a_second @ a_first - a_first @ a_second
    zeros = torch.zeros(2, 2, dtype=DTYPE)
    expected = torch.cat(
        (
            torch.cat((first - second, zeros), dim=-1),
            torch.cat((zeros, second - first), dim=-1),
        ),
        dim=-2,
    )
    assert torch.allclose(measured, expected, atol=1.0e-14)


def test_exact_kernel_preserves_both_vector_boundaries() -> None:
    profiles = _profiles()
    zeros = torch.zeros(len(profiles), dtype=DTYPE)
    ones = torch.ones_like(zeros)
    identity = torch.eye(2, dtype=DTYPE).expand(len(profiles), 2, 2)
    null = torch.zeros_like(identity)
    assert torch.allclose(
        study.boundary_kernel(profiles, zeros),
        torch.cat((identity, null), dim=-1),
        atol=2.0e-12,
    )
    assert torch.allclose(
        study.boundary_kernel(profiles, ones),
        torch.cat((null, identity), dim=-1),
        atol=2.0e-12,
    )


def test_reference_is_rotation_covariant_and_volume_preserving() -> None:
    certification = study.certify_reference(DEVICE)
    assert certification["boundary_max_abs_error"] <= 1.0e-12
    assert certification["transfer_determinant_max_abs_error"] <= 1.0e-10
    assert certification["rotation_covariance_max_abs_error"] <= 1.0e-12


def test_sorted_generator_is_exactly_blind_to_layer_permutations() -> None:
    generator = torch.Generator().manual_seed(29)
    profile_a, profile_b = study.order_challenge_profiles(
        64,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    query_x = torch.linspace(0.1, 0.9, 64, dtype=DTYPE)
    model = study.LearnedCoupledGenerator(physical_order=False).to(dtype=DTYPE)
    assert torch.equal(
        model(profile_a, query_x),
        model(profile_b, query_x),
    )


def test_learned_generator_is_rotation_equivariant() -> None:
    profiles = _profiles()
    model = study.LearnedCoupledGenerator(physical_order=True).to(dtype=DTYPE)
    angle = torch.tensor(0.61, dtype=DTYPE)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    mapped = model.map_local(profiles)
    rotated = rotation @ profiles @ rotation.T
    mapped_rotated = model.map_local(rotated)
    assert torch.allclose(
        mapped_rotated,
        rotation @ mapped @ rotation.T,
        atol=2.0e-12,
    )


def test_learned_generator_accepts_twice_as_many_layers() -> None:
    profiles = _profiles(n_layers=study.DOUBLED_LAYERS)
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    model = study.LearnedCoupledGenerator(physical_order=True).to(dtype=DTYPE)
    response = model(profiles, query_x)
    assert response.shape == (len(profiles), 2, 4)
    assert torch.isfinite(response).all()


def test_learned_arms_have_comparable_capacity() -> None:
    counts = {
        arm: sum(parameter.numel() for parameter in study.build_model(arm).parameters())
        for arm in study.LEARNED_ARMS
    }
    assert counts["ordered_pool"] == 4_152
    assert counts["sorted_generator"] == counts["path_generator"] == 4_417
    assert max(counts.values()) / min(counts.values()) < 1.07
    assert sum(study.AnalyticPath().parameters(), start=0) == 0


def test_analytic_oracle_has_zero_registered_error() -> None:
    model = study.AnalyticPath().to(dtype=DTYPE)
    evaluation = study.evaluate_split(
        model,
        "held_out_twist",
        n_profiles=2,
        n_query=4,
        device=DEVICE,
    )
    assert evaluation["metrics"]["operator_relative_l2"]["mean"] == 0.0
    assert evaluation["metrics"]["cross_channel_relative_l2"]["mean"] == 0.0
    order = study.evaluate_order_challenge(model, n_pairs=32, device=DEVICE)
    assert order["paired_field_relative_l2"] == 0.0
    assert order["contrast_relative_l2"] == 0.0
    assert order["true_contrast_relative_l2"] > 0.01


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="path_generator",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_batch_size=8,
        evaluation_profiles=2,
        evaluation_query_points=4,
        order_pairs=16,
    )
    assert report["study"] == study.STUDY
    assert report["arm"] == "path_generator"
    assert report["parameters"] == 4_417
    assert set(report["split_evaluation"]) == set(study.SPLITS)
    assert report["order_challenge"]["multiset_max_abs_difference"] == 0.0
    assert report["local_map_evaluation"] is not None
