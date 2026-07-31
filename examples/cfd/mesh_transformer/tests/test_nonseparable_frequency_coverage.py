# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the coefficient-frequency coverage experiment."""

from __future__ import annotations

import nonseparable_frequency_coverage as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(n_profiles: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(47)
    return study.base.sample_profiles(
        n_profiles,
        heterogeneity=0.5,
        x_frequencies=(1, 2, 3, 4),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
        n_layers=study.TRAIN_LAYERS,
    )


def test_training_support_changes_profiles_without_changing_shapes() -> None:
    datasets = [
        study.generate_training_dataset(
            x_frequencies=frequencies,
            samples_per_level=2,
            device=DEVICE,
            dtype=DTYPE,
        )
        for frequencies in study.TRAINING_FREQUENCIES.values()
    ]
    narrow_profiles, narrow_query, narrow_target = datasets[0]
    broad_profiles, broad_query, broad_target = datasets[1]
    assert narrow_profiles.shape == broad_profiles.shape == (6, 16, 6, 6)
    assert narrow_query.shape == broad_query.shape == (6,)
    assert narrow_target.shape == broad_target.shape == (6, 6, 12)
    assert not torch.allclose(narrow_profiles, broad_profiles)
    assert torch.isfinite(narrow_target).all()
    assert torch.isfinite(broad_target).all()


def test_flow_retains_rank_and_boundary_contract_at_training_resolution() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    model = study.flow.LatentFlow(physical_order=True).to(dtype=DTYPE)
    correction = model(profiles, query_x)
    assert torch.all(
        torch.linalg.matrix_rank(correction, tol=1.0e-10) <= study.flow.RANK
    )
    assert torch.count_nonzero(model(profiles, torch.zeros_like(query_x))) == 0
    assert torch.count_nonzero(model(profiles, torch.ones_like(query_x))) == 0


def test_tiny_training_runs_are_finite() -> None:
    for arm in study.LEARNED_ARMS:
        model, history = study.train_model(
            arm,
            seed=study.SEEDS[0],
            device=DEVICE,
            steps=2,
            batch_size=4,
            samples_per_level=2,
        )
        assert isinstance(model, study.flow.LatentFlow)
        assert history[-1]["residual_relative_mse"] >= 0.0


def test_analytic_evaluation_covers_both_unseen_resolutions() -> None:
    results = {
        split: study.evaluate_split(
            "fixed_rank4",
            None,
            split,
            n_profiles=2,
            n_query=3,
            evaluation_seed=study.EVALUATION_SEED,
            device=DEVICE,
        )
        for split in ("unseen_16", "unseen_32")
    }
    assert results["unseen_16"]["n_layers"] == 16
    assert results["unseen_32"]["n_layers"] == 32
    for result in results.values():
        assert result["metrics"]["cross_channel_relative_l2"]["mean"] >= 0.0
