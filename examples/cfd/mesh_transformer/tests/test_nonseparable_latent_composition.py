# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the nonseparable rank-four latent experiment."""

from __future__ import annotations

import nonseparable_latent_composition as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(n_profiles: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(37)
    return study.sample_profiles(
        n_profiles,
        heterogeneity=0.6,
        x_frequencies=study.TRAIN_X_FREQUENCIES,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_sampled_profiles_are_positive_and_nonseparable() -> None:
    profiles = _profiles()
    assert torch.all(torch.linalg.eigvalsh(profiles) > 0.0)
    assert torch.allclose(profiles, profiles.transpose(-2, -1), atol=1.0e-14)
    diagonal = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    assert float((profiles - diagonal).abs().max()) > 1.0e-3


def test_profile_features_are_finite_and_complete() -> None:
    features = study.profile_features(_profiles())
    expected = study.CHANNELS * (study.CHANNELS + 1) // 2
    assert features.shape == (4, study.LAYERS, expected)
    assert torch.all(torch.isfinite(features))


def test_rank_four_models_enforce_rank_and_boundary_contract() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    for arm in ("global_rank4", "sorted_rank4", "path_rank4"):
        model = study.build_model(arm).to(dtype=DTYPE)
        correction = model(profiles, query_x)
        assert torch.all(
            torch.linalg.matrix_rank(correction, tol=1.0e-10) <= study.RANK
        )
        assert torch.count_nonzero(model(profiles, torch.zeros_like(query_x))) == 0
        assert torch.count_nonzero(model(profiles, torch.ones_like(query_x))) == 0


def test_sorted_path_is_permutation_invariant() -> None:
    profiles = _profiles()
    permuted = profiles[:, torch.tensor((3, 1, 7, 0, 6, 2, 5, 4))]
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    model = study.build_model("sorted_rank4").to(dtype=DTYPE)
    assert torch.allclose(
        model(profiles, query_x),
        model(permuted, query_x),
        atol=1.0e-12,
    )


def test_oracle_rank_four_is_no_worse_than_fixed_rank_four() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    target = study.census.boundary_kernel(profiles, query_x)
    carrier = study.census.diagonal_carrier_kernel(profiles, query_x)
    residual = target - carrier
    oracle = study.oracle_rank_correction(residual)
    fixed = study.analytic_correction(
        "fixed_rank4",
        profiles,
        query_x,
        residual,
    )
    assert torch.linalg.vector_norm(oracle - residual) <= torch.linalg.vector_norm(
        fixed - residual
    )


def test_tiny_training_run_is_finite() -> None:
    model, history = study.train_model(
        "path_rank4",
        seed=study.SEEDS[0],
        device=DEVICE,
        steps=2,
        batch_size=4,
        samples_per_level=2,
    )
    assert isinstance(model, study.PathCorrection)
    assert history[-1]["residual_relative_mse"] >= 0.0
