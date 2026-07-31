# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test a connection-aware moving rank-four subspace on nonseparable fields."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_continuum_flow as flow
import nonseparable_latent_composition as base
import torch
from provenance import runtime_environment, source_provenance

STUDY = "nonseparable_adaptive_subspace_v1"
RANK = base.RANK
ARMS = (
    "diagonal_carrier",
    "fixed_rank4",
    "global_rank4",
    "local_naive_rank4",
    "local_connected_rank4",
    "oracle_rank4",
)
EVALUATION_PROFILES = 128
EVALUATION_QUERY_POINTS = 32

SPLITS: dict[str, tuple[float, tuple[int, ...], int, int]] = {
    "low_16": (0.5, (1, 2), 16, 0),
    "mid_16": (0.5, (3, 4), 16, 10_000),
    "high_16": (0.5, (5, 6), 16, 20_000),
    "high_32": (0.5, (5, 6), 32, 20_000),
}


def _block_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return a batched block diagonal matrix with two equally shaped blocks."""

    zeros = torch.zeros_like(first)
    return torch.cat(
        (
            torch.cat((first, zeros), dim=-1),
            torch.cat((zeros, second), dim=-1),
        ),
        dim=-2,
    )


def local_eigenbasis(profiles: torch.Tensor, *, rank: int = RANK) -> torch.Tensor:
    """Return the lowest local eigenspace in every coefficient layer."""

    _, eigenvectors = torch.linalg.eigh(profiles)
    return eigenvectors[..., :rank]


def _reduced_total_transfer(
    reduced_profiles: torch.Tensor,
    bases: torch.Tensor,
    *,
    connected: bool,
) -> torch.Tensor:
    rank = reduced_profiles.shape[-1]
    n_layers = reduced_profiles.shape[-3]
    matrix = torch.eye(
        2 * rank,
        device=reduced_profiles.device,
        dtype=reduced_profiles.dtype,
    ).expand(*reduced_profiles.shape[:-3], 2 * rank, 2 * rank)
    layer_width = 1.0 / n_layers
    for index in range(n_layers):
        matrix = (
            base.census.layer_transfer(reduced_profiles[..., index, :, :], layer_width)
            @ matrix
        )
        if connected and index + 1 < n_layers:
            overlap = (
                bases[..., index + 1, :, :].transpose(-2, -1) @ bases[..., index, :, :]
            )
            matrix = _block_pair(overlap, overlap) @ matrix
    return matrix


def _reduced_partial_transfer(
    reduced_profiles: torch.Tensor,
    bases: torch.Tensor,
    query_x: torch.Tensor,
    *,
    connected: bool,
) -> torch.Tensor:
    rank = reduced_profiles.shape[-1]
    n_layers = reduced_profiles.shape[-3]
    identity = torch.eye(
        2 * rank,
        device=reduced_profiles.device,
        dtype=reduced_profiles.dtype,
    ).expand(*reduced_profiles.shape[:-3], 2 * rank, 2 * rank)
    matrix = identity
    starts = (
        torch.arange(
            n_layers,
            device=reduced_profiles.device,
            dtype=reduced_profiles.dtype,
        )
        / n_layers
    )
    lengths = (query_x[..., None] - starts).clamp(
        min=0.0,
        max=1.0 / n_layers,
    )
    for index in range(n_layers):
        matrix = (
            base.census.layer_transfer(
                reduced_profiles[..., index, :, :],
                lengths[..., index],
            )
            @ matrix
        )
        if connected and index + 1 < n_layers:
            overlap = (
                bases[..., index + 1, :, :].transpose(-2, -1) @ bases[..., index, :, :]
            )
            transition = _block_pair(overlap, overlap)
            crossed = query_x >= (index + 1) / n_layers
            transition = torch.where(
                crossed[..., None, None],
                transition,
                identity,
            )
            matrix = transition @ matrix
    return matrix


def _query_basis(bases: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
    n_layers = bases.shape[-3]
    n_channels = bases.shape[-2]
    rank = bases.shape[-1]
    indices = torch.floor(query_x * n_layers).to(torch.long).clamp(max=n_layers - 1)
    gather_indices = indices[..., None, None, None].expand(
        *indices.shape,
        1,
        n_channels,
        rank,
    )
    return torch.gather(bases, -3, gather_indices).squeeze(-3)


def reduced_moving_basis_kernel(
    reduced_profiles: torch.Tensor,
    bases: torch.Tensor,
    query_x: torch.Tensor,
    *,
    connected: bool,
) -> torch.Tensor:
    """Solve the reduced boundary problem and lift through its moving bases."""

    rank = reduced_profiles.shape[-1]
    transfer = _reduced_total_transfer(
        reduced_profiles,
        bases,
        connected=connected,
    )
    t11 = transfer[..., :rank, :rank]
    t12 = transfer[..., :rank, rank:]
    identity = torch.eye(
        rank,
        device=reduced_profiles.device,
        dtype=reduced_profiles.dtype,
    ).expand(*reduced_profiles.shape[:-3], rank, rank)
    initial_derivative = torch.linalg.solve(
        t12,
        torch.cat((-t11, identity), dim=-1),
    )
    initial_value = torch.cat((identity, torch.zeros_like(identity)), dim=-1)
    initial_state = torch.cat((initial_value, initial_derivative), dim=-2)
    partial = _reduced_partial_transfer(
        reduced_profiles,
        bases,
        query_x,
        connected=connected,
    )
    reduced_kernel = partial[..., :rank, :] @ initial_state
    query_basis = _query_basis(bases, query_x)
    boundary_basis = _block_pair(
        bases[..., 0, :, :],
        bases[..., -1, :, :],
    )
    return query_basis @ reduced_kernel @ boundary_basis.transpose(-2, -1)


def adaptive_correction(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
    *,
    connected: bool,
    rank: int = RANK,
) -> torch.Tensor:
    """Return a moving-subspace correction above the exact diagonal carrier."""

    bases = local_eigenbasis(profiles, rank=rank)
    reduced_profiles = bases.transpose(-2, -1) @ profiles @ bases
    diagonal_profiles = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    reduced_diagonal = bases.transpose(-2, -1) @ diagonal_profiles @ bases
    return reduced_moving_basis_kernel(
        reduced_profiles,
        bases,
        query_x,
        connected=connected,
    ) - reduced_moving_basis_kernel(
        reduced_diagonal,
        bases,
        query_x,
        connected=connected,
    )


@torch.no_grad()
def evaluate_split(
    split: str,
    *,
    profile_seed: int,
    n_profiles: int,
    n_query: int,
    device: torch.device,
) -> dict[str, Any]:
    heterogeneity, x_frequencies, n_layers, seed_offset = SPLITS[split]
    generator = torch.Generator(device=device)
    generator.manual_seed(profile_seed + seed_offset)
    profiles = base.sample_profiles(
        n_profiles,
        heterogeneity=heterogeneity,
        x_frequencies=x_frequencies,
        generator=generator,
        device=device,
        dtype=torch.float64,
        n_layers=n_layers,
    )
    query_grid = torch.linspace(
        0.03,
        0.97,
        n_query,
        device=device,
        dtype=torch.float64,
    )
    repeated_profiles = profiles.repeat_interleave(n_query, dim=0)
    query_x = query_grid.repeat(n_profiles)
    target = base.census.boundary_kernel(repeated_profiles, query_x)
    carrier = base.census.diagonal_carrier_kernel(repeated_profiles, query_x)
    residual = target - carrier
    predictions = {
        "diagonal_carrier": carrier,
        "fixed_rank4": base.census.fixed_truncation_kernel(
            repeated_profiles,
            query_x,
            rank=RANK,
        ),
        "global_rank4": base.census.global_eigenspace_kernel(
            repeated_profiles,
            query_x,
            rank=RANK,
        ),
        "local_naive_rank4": carrier
        + adaptive_correction(
            repeated_profiles,
            query_x,
            connected=False,
        ),
        "local_connected_rank4": carrier
        + adaptive_correction(
            repeated_profiles,
            query_x,
            connected=True,
        ),
        "oracle_rank4": carrier + base.oracle_rank_correction(residual),
    }
    target = target.reshape(n_profiles, n_query, base.CHANNELS, 2 * base.CHANNELS)
    cross_target = base.census.cross_channel_part(target)
    metrics = {}
    for arm, prediction in predictions.items():
        prediction = prediction.reshape_as(target)
        cross_prediction = base.census.cross_channel_part(prediction)
        metrics[arm] = {
            "operator_relative_l2": flow._profile_relative_metrics(
                prediction,
                target,
            ),
            "cross_channel_relative_l2": flow._profile_relative_metrics(
                cross_prediction,
                cross_target,
            ),
        }
    return {
        "heterogeneity": heterogeneity,
        "x_frequencies": list(x_frequencies),
        "n_layers": n_layers,
        "n_profiles": n_profiles,
        "n_query": n_query,
        "metrics": metrics,
    }


def run_census(
    *,
    profile_seed: int,
    device: torch.device,
    evaluation_profiles: int = EVALUATION_PROFILES,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
) -> dict[str, Any]:
    started = time.time()
    split_evaluation = {}
    for index, split in enumerate(SPLITS, start=1):
        split_evaluation[split] = evaluate_split(
            split,
            profile_seed=profile_seed,
            n_profiles=evaluation_profiles,
            n_query=evaluation_query_points,
            device=device,
        )
        print(
            f"HEARTBEAT phase=evaluate completed_units={index} split={split}",
            flush=True,
        )
    return {
        "study": STUDY,
        "profile_seed": profile_seed,
        "protocol": {
            "evaluation_profiles": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "rank": RANK,
            "dtype_evaluation": "float64",
        },
        "split_evaluation": split_evaluation,
        "elapsed_seconds": time.time() - started,
        "environment": runtime_environment(device),
        "source": source_provenance(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluation-profiles", type=int, default=EVALUATION_PROFILES)
    parser.add_argument(
        "--evaluation-query-points",
        type=int,
        default=EVALUATION_QUERY_POINTS,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_census(
        profile_seed=args.profile_seed,
        device=torch.device(args.device),
        evaluation_profiles=args.evaluation_profiles,
        evaluation_query_points=args.evaluation_query_points,
    )
    shared.atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
