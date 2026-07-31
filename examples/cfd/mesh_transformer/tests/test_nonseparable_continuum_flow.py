# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the nonseparable continuum-flow experiment."""

from __future__ import annotations

import nonseparable_continuum_flow as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(n_profiles: int = 4, *, n_layers: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(41)
    return study.base.sample_profiles(
        n_profiles,
        heterogeneity=0.6,
        x_frequencies=study.base.TRAIN_X_FREQUENCIES,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
        n_layers=n_layers,
    )


def test_flow_models_enforce_rank_and_boundary_contract() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    for arm in ("sorted_flow_rank4", "path_flow_rank4"):
        model = study.build_model(arm).to(dtype=DTYPE)
        correction = model(profiles, query_x)
        assert torch.all(
            torch.linalg.matrix_rank(correction, tol=1.0e-10) <= study.RANK
        )
        assert torch.count_nonzero(model(profiles, torch.zeros_like(query_x))) == 0
        assert torch.count_nonzero(model(profiles, torch.ones_like(query_x))) == 0


def test_sorted_flow_is_permutation_invariant() -> None:
    profiles = _profiles()
    permuted = profiles[:, torch.tensor((3, 1, 7, 0, 6, 2, 5, 4))]
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    model = study.build_model("sorted_flow_rank4").to(dtype=DTYPE)
    assert torch.allclose(
        model(profiles, query_x),
        model(permuted, query_x),
        atol=1.0e-12,
    )


def test_constant_token_flow_converges_under_refinement() -> None:
    torch.manual_seed(3)
    model = study.LatentFlow(physical_order=True).to(dtype=DTYPE)
    token = torch.randn(1, 1, 24, dtype=DTYPE)
    state_8 = model.integrate_tokens(token.expand(1, 8, 24))
    state_16 = model.integrate_tokens(token.expand(1, 16, 24))
    state_32 = model.integrate_tokens(token.expand(1, 32, 24))
    coarse_change = torch.linalg.vector_norm(state_16 - state_8)
    fine_change = torch.linalg.vector_norm(state_32 - state_16)
    assert fine_change < coarse_change


def test_matched_profile_sampling_is_resolution_consistent() -> None:
    generator_8 = torch.Generator().manual_seed(43)
    generator_16 = torch.Generator().manual_seed(43)
    profiles_8 = study.base.sample_profiles(
        3,
        heterogeneity=0.5,
        x_frequencies=study.base.TRAIN_X_FREQUENCIES,
        generator=generator_8,
        device=DEVICE,
        dtype=DTYPE,
        n_layers=8,
    )
    profiles_16 = study.base.sample_profiles(
        3,
        heterogeneity=0.5,
        x_frequencies=study.base.TRAIN_X_FREQUENCIES,
        generator=generator_16,
        device=DEVICE,
        dtype=DTYPE,
        n_layers=16,
    )
    assert torch.allclose(profiles_8, profiles_16[:, ::2], atol=0.2)


def test_tiny_training_run_is_finite() -> None:
    model, history = study.train_model(
        "path_flow_rank4",
        seed=study.SEEDS[0],
        device=DEVICE,
        steps=2,
        batch_size=4,
        samples_per_level=2,
    )
    assert isinstance(model, study.LatentFlow)
    assert history[-1]["residual_relative_mse"] >= 0.0
