# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the nonseparable response-rank census."""

from __future__ import annotations

import nonseparable_rank_census as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _profiles(
    n_profiles: int = 4,
    *,
    heterogeneity: float = 0.5,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    return study.sample_operator_profiles(
        n_profiles,
        n_layers=study.LAYERS,
        heterogeneity=heterogeneity,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_cosine_basis_is_discretely_orthonormal() -> None:
    basis, weight = study.cosine_basis(
        study.TRANSVERSE_POINTS,
        study.CHANNELS,
        device=DEVICE,
        dtype=DTYPE,
    )
    gram = weight * basis.T @ basis
    assert torch.allclose(gram, torch.eye(study.CHANNELS, dtype=DTYPE), atol=1.0e-14)


def test_sampled_operator_is_positive_and_nonseparable() -> None:
    profiles = _profiles()
    assert torch.all(torch.linalg.eigvalsh(profiles) > 0.0)
    assert torch.allclose(profiles, profiles.transpose(-2, -1), atol=1.0e-14)
    off_diagonal = profiles - torch.diag_embed(
        torch.diagonal(profiles, dim1=-2, dim2=-1)
    )
    assert float(off_diagonal.abs().max()) > 1.0e-3
    assert float((profiles[:, 1:] - profiles[:, :-1]).abs().max()) > 1.0e-3


def test_boundary_kernel_preserves_all_boundary_channels() -> None:
    profiles = _profiles()
    zeros = torch.zeros(len(profiles), dtype=DTYPE)
    ones = torch.ones_like(zeros)
    identity = torch.eye(study.CHANNELS, dtype=DTYPE).expand(
        len(profiles),
        study.CHANNELS,
        study.CHANNELS,
    )
    null = torch.zeros_like(identity)
    assert torch.allclose(
        study.boundary_kernel(profiles, zeros),
        torch.cat((identity, null), dim=-1),
        atol=2.0e-11,
    )
    assert torch.allclose(
        study.boundary_kernel(profiles, ones),
        torch.cat((null, identity), dim=-1),
        atol=2.0e-11,
    )


def test_full_rank_reductions_reproduce_exact_kernel() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    target = study.boundary_kernel(profiles, query_x)
    fixed = study.fixed_truncation_kernel(
        profiles,
        query_x,
        rank=study.CHANNELS,
    )
    global_eigenspace = study.global_eigenspace_kernel(
        profiles,
        query_x,
        rank=study.CHANNELS,
    )
    assert torch.allclose(fixed, target, atol=2.0e-11)
    assert torch.allclose(global_eigenspace, target, atol=2.0e-11)


def test_diagonal_carrier_has_no_cross_channel_response() -> None:
    profiles = _profiles()
    query_x = torch.linspace(0.1, 0.9, len(profiles), dtype=DTYPE)
    carrier = study.diagonal_carrier_kernel(profiles, query_x)
    assert torch.count_nonzero(study.cross_channel_part(carrier)) == 0


def test_oracle_rank_error_is_monotone() -> None:
    level = study.census_level(
        0.5,
        n_profiles=2,
        n_query=3,
        seed=study.SEED,
        device=DEVICE,
    )
    errors = [
        level["ranks"][str(rank)]["oracle_best_rank_residual_relative_l2"]["mean"]
        for rank in study.RANKS
    ]
    captured = [
        level["ranks"][str(rank)]["oracle_residual_energy_captured"]["mean"]
        for rank in study.RANKS
    ]
    assert all(first >= second for first, second in zip(errors, errors[1:]))
    assert all(first <= second for first, second in zip(captured, captured[1:]))
    assert errors[-1] < errors[0]
