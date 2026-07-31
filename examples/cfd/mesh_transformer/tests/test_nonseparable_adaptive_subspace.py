# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the moving-subspace realizability census."""

from __future__ import annotations

import nonseparable_adaptive_subspace as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(n_profiles: int = 3, *, n_layers: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(53)
    return study.base.sample_profiles(
        n_profiles,
        heterogeneity=0.5,
        x_frequencies=(1, 2),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
        n_layers=n_layers,
    )


def test_block_pair_constructs_batched_block_diagonal() -> None:
    first = torch.randn(2, 3, 2, dtype=DTYPE)
    second = torch.randn(2, 3, 2, dtype=DTYPE)
    block = study._block_pair(first, second)
    assert block.shape == (2, 6, 4)
    assert torch.count_nonzero(block[:, :3, 2:]) == 0
    assert torch.count_nonzero(block[:, 3:, :2]) == 0


def test_adaptive_correction_enforces_rank_and_boundary_contract() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    for connected in (False, True):
        correction = study.adaptive_correction(
            profiles,
            query_x,
            connected=connected,
        )
        assert torch.all(torch.linalg.matrix_rank(correction, tol=1.0e-9) <= study.RANK)
        at_left = study.adaptive_correction(
            profiles,
            torch.zeros_like(query_x),
            connected=connected,
        )
        at_right = study.adaptive_correction(
            profiles,
            torch.ones_like(query_x),
            connected=connected,
        )
        assert torch.allclose(at_left, torch.zeros_like(at_left), atol=1.0e-10)
        assert torch.allclose(at_right, torch.zeros_like(at_right), atol=1.0e-9)


def test_constant_local_basis_matches_global_eigenspace() -> None:
    base_profile = _profiles(2, n_layers=1)
    profiles = base_profile.expand(2, 8, 6, 6).clone()
    query_x = torch.tensor((0.3, 0.7), dtype=DTYPE)
    carrier = study.base.census.diagonal_carrier_kernel(profiles, query_x)
    global_prediction = study.base.census.global_eigenspace_kernel(
        profiles,
        query_x,
        rank=study.RANK,
    )
    for connected in (False, True):
        local_prediction = carrier + study.adaptive_correction(
            profiles,
            query_x,
            connected=connected,
        )
        assert torch.allclose(local_prediction, global_prediction, atol=1.0e-9)


def test_tiny_census_is_finite() -> None:
    report = study.run_census(
        profile_seed=59,
        device=DEVICE,
        evaluation_profiles=2,
        evaluation_query_points=3,
    )
    for split in study.SPLITS:
        for arm in study.ARMS:
            metric = report["split_evaluation"][split]["metrics"][arm][
                "cross_channel_relative_l2"
            ]["mean"]
            assert torch.isfinite(torch.tensor(metric))
