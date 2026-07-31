# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Measure the low-rank ceiling of a nonseparable spectral boundary operator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import torch
from provenance import runtime_environment, source_provenance

STUDY = "nonseparable_rank_census_v1"
CHANNELS = 6
LAYERS = 8
TRANSVERSE_POINTS = 64
PROFILES = 256
QUERY_POINTS = 32
HARMONICS = 3
SEED = 181_000_151
RANKS = tuple(range(1, CHANNELS))
HETEROGENEITY_LEVELS = (0.2, 0.5, 0.8)


def cosine_basis(
    n_points: int,
    n_channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """Return an orthonormal real cosine basis on the periodic interval."""

    y = 2.0 * math.pi * torch.arange(n_points, device=device, dtype=dtype) / n_points
    columns = [torch.full_like(y, 1.0 / math.sqrt(2.0 * math.pi))]
    columns.extend(
        torch.cos(index * y) / math.sqrt(math.pi) for index in range(1, n_channels)
    )
    return torch.stack(columns, dim=-1), 2.0 * math.pi / n_points


def _uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def sample_operator_profiles(
    n_profiles: int,
    *,
    n_layers: int,
    heterogeneity: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    n_channels: int = CHANNELS,
    transverse_points: int = TRANSVERSE_POINTS,
) -> torch.Tensor:
    """Sample Fourier--Galerkin matrices for a positive nonseparable field."""

    basis, weight = cosine_basis(
        transverse_points,
        n_channels,
        device=device,
        dtype=dtype,
    )
    y = (
        2.0
        * math.pi
        * torch.arange(transverse_points, device=device, dtype=dtype)
        / transverse_points
    )
    x = (torch.arange(n_layers, device=device, dtype=dtype) + 0.5) / n_layers
    base = _uniform(
        (n_profiles, 1, 1),
        low=math.log(0.4),
        high=math.log(1.2),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    log_coefficient = base.expand(n_profiles, n_layers, transverse_points).clone()
    for harmonic in range(1, HARMONICS + 1):
        amplitude = _uniform(
            (n_profiles, 1, 1),
            low=0.35 * heterogeneity / harmonic,
            high=heterogeneity / harmonic,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        phase = _uniform(
            (n_profiles, 1, 1),
            low=0.0,
            high=2.0 * math.pi,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        x_frequency = torch.randint(
            1,
            3,
            (n_profiles, 1, 1),
            generator=generator,
            device=device,
        ).to(dtype)
        x_envelope = torch.sin(2.0 * math.pi * x_frequency * x[None, :, None] + phase)
        log_coefficient = log_coefficient + (
            amplitude * x_envelope * torch.cos(harmonic * y)[None, None, :]
        )
    coefficient = torch.exp(log_coefficient)
    reaction = weight * torch.einsum(
        "yi,pny,yj->pnij",
        basis,
        coefficient,
        basis,
    )
    transverse_laplacian = torch.diag(
        torch.arange(n_channels, device=device, dtype=dtype).square()
    )
    return reaction + transverse_laplacian


def generator_matrix(coefficients: torch.Tensor) -> torch.Tensor:
    """Return the first-order block generator for any channel count."""

    n_channels = coefficients.shape[-1]
    batch_shape = coefficients.shape[:-2]
    zeros = torch.zeros(
        *batch_shape,
        n_channels,
        n_channels,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    identity = torch.eye(
        n_channels,
        device=coefficients.device,
        dtype=coefficients.dtype,
    ).expand(*batch_shape, n_channels, n_channels)
    return torch.cat(
        (
            torch.cat((zeros, identity), dim=-1),
            torch.cat((coefficients, zeros), dim=-1),
        ),
        dim=-2,
    )


def layer_transfer(
    coefficients: torch.Tensor,
    length: torch.Tensor | float,
) -> torch.Tensor:
    length_tensor = torch.as_tensor(
        length,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    return torch.matrix_exp(
        generator_matrix(coefficients) * length_tensor[..., None, None]
    )


def total_transfer(profiles: torch.Tensor) -> torch.Tensor:
    n_channels = profiles.shape[-1]
    batch_shape = profiles.shape[:-3]
    matrix = torch.eye(
        2 * n_channels,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*batch_shape, 2 * n_channels, 2 * n_channels)
    layer_width = 1.0 / profiles.shape[-3]
    for index in range(profiles.shape[-3]):
        matrix = layer_transfer(profiles[..., index, :, :], layer_width) @ matrix
    return matrix


def partial_transfer(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    n_layers = profiles.shape[-3]
    n_channels = profiles.shape[-1]
    starts = (
        torch.arange(
            n_layers,
            device=profiles.device,
            dtype=profiles.dtype,
        )
        / n_layers
    )
    lengths = (query_x[..., None] - starts).clamp(
        min=0.0,
        max=1.0 / n_layers,
    )
    batch_shape = profiles.shape[:-3]
    matrix = torch.eye(
        2 * n_channels,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*batch_shape, 2 * n_channels, 2 * n_channels)
    for index in range(n_layers):
        matrix = (
            layer_transfer(
                profiles[..., index, :, :],
                lengths[..., index],
            )
            @ matrix
        )
    return matrix


def boundary_kernel(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    """Return the full boundary-to-interior response matrix."""

    n_channels = profiles.shape[-1]
    transfer = total_transfer(profiles)
    t11 = transfer[..., :n_channels, :n_channels]
    t12 = transfer[..., :n_channels, n_channels:]
    identity = torch.eye(
        n_channels,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*profiles.shape[:-3], n_channels, n_channels)
    initial_derivative = torch.linalg.solve(
        t12,
        torch.cat((-t11, identity), dim=-1),
    )
    initial_value = torch.cat((identity, torch.zeros_like(identity)), dim=-1)
    initial_state = torch.cat((initial_value, initial_derivative), dim=-2)
    return partial_transfer(profiles, query_x)[..., :n_channels, :] @ initial_state


def linear_carrier(
    query_x: torch.Tensor,
    n_channels: int,
) -> torch.Tensor:
    identity = torch.eye(
        n_channels,
        device=query_x.device,
        dtype=query_x.dtype,
    ).expand(*query_x.shape, n_channels, n_channels)
    return torch.cat(
        (
            (1.0 - query_x)[..., None, None] * identity,
            query_x[..., None, None] * identity,
        ),
        dim=-1,
    )


def diagonal_carrier_kernel(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    """Propagate each uncoupled transverse mode with its local diagonal law."""

    diagonal_profiles = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    return boundary_kernel(diagonal_profiles, query_x)


def lift_reduced_correction(
    reduced_correction: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Lift a reduced boundary-kernel correction into the full channel space."""

    boundary_basis = torch.block_diag(basis, basis)
    return basis @ reduced_correction @ boundary_basis.transpose(-2, -1)


def fixed_truncation_kernel(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Use the first coordinate modes as one fixed Galerkin subspace."""

    n_channels = profiles.shape[-1]
    basis = torch.eye(
        n_channels,
        rank,
        device=profiles.device,
        dtype=profiles.dtype,
    )
    reduced_profiles = basis.T @ profiles @ basis
    reduced = boundary_kernel(reduced_profiles, query_x)
    diagonal_profiles = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    reduced_diagonal = basis.T @ diagonal_profiles @ basis
    reduced_carrier = boundary_kernel(reduced_diagonal, query_x)
    return diagonal_carrier_kernel(profiles, query_x) + lift_reduced_correction(
        reduced - reduced_carrier,
        basis,
    )


def global_eigenspace_kernel(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Use each profile's lowest global-average eigenspace at every layer."""

    _, eigenvectors = torch.linalg.eigh(profiles.mean(dim=-3))
    basis = eigenvectors[..., :rank]
    reduced_profiles = (
        basis.transpose(-2, -1)[..., None, :, :] @ profiles @ basis[..., None, :, :]
    )
    reduced = boundary_kernel(reduced_profiles, query_x)
    diagonal_profiles = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    reduced_diagonal = (
        basis.transpose(-2, -1)[..., None, :, :]
        @ diagonal_profiles
        @ basis[..., None, :, :]
    )
    reduced_carrier = boundary_kernel(reduced_diagonal, query_x)
    basis_transpose = basis.transpose(-2, -1)
    zeros = torch.zeros_like(basis_transpose)
    boundary_basis = torch.cat(
        (
            torch.cat((basis_transpose, zeros), dim=-1),
            torch.cat((zeros, basis_transpose), dim=-1),
        ),
        dim=-2,
    )
    lifted = basis @ (reduced - reduced_carrier) @ boundary_basis
    return diagonal_carrier_kernel(profiles, query_x) + lifted


def _relative_profile_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    error = torch.linalg.vector_norm(prediction - target, dim=(1, 2, 3))
    scale = torch.linalg.vector_norm(target, dim=(1, 2, 3)).clamp_min(1.0e-14)
    relative = error / scale
    return {
        "mean": float(relative.mean()),
        "median": float(relative.median()),
        "maximum": float(relative.max()),
    }


def cross_channel_part(kernel: torch.Tensor) -> torch.Tensor:
    """Keep response entries whose output and boundary channels differ."""

    n_channels = kernel.shape[-2]
    channel_mask = ~torch.eye(
        n_channels,
        device=kernel.device,
        dtype=torch.bool,
    )
    split_boundary = kernel.reshape(*kernel.shape[:-2], n_channels, 2, n_channels)
    return (split_boundary * channel_mask[:, None, :]).reshape_as(kernel)


@torch.no_grad()
def census_level(
    heterogeneity: float,
    *,
    n_profiles: int,
    n_query: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + round(100 * heterogeneity))
    profiles = sample_operator_profiles(
        n_profiles,
        n_layers=LAYERS,
        heterogeneity=heterogeneity,
        generator=generator,
        device=device,
        dtype=torch.float64,
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
    target = boundary_kernel(repeated_profiles, query_x).reshape(
        n_profiles,
        n_query,
        CHANNELS,
        2 * CHANNELS,
    )
    carrier = diagonal_carrier_kernel(
        repeated_profiles,
        query_x,
    ).reshape_as(target)
    residual = target - carrier
    left_vectors, singular_values, right_vectors = torch.linalg.svd(
        residual,
        full_matrices=False,
    )
    target_norm = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1.0e-14)
    residual_norm = torch.linalg.vector_norm(residual, dim=(-2, -1)).clamp_min(1.0e-14)
    cross_target = cross_channel_part(target)
    cross_norm = torch.linalg.vector_norm(cross_target, dim=(-2, -1)).clamp_min(1.0e-14)

    ranks: dict[str, Any] = {}
    for rank in RANKS:
        tail = torch.linalg.vector_norm(singular_values[..., rank:], dim=-1)
        best_relative = tail / target_norm
        best_residual_relative = tail / residual_norm
        captured_residual = 1.0 - (tail / residual_norm).square()
        low_rank_residual = (
            left_vectors[..., :, :rank] * singular_values[..., None, :rank]
        ) @ right_vectors[..., :rank, :]
        oracle = carrier + low_rank_residual
        fixed = fixed_truncation_kernel(
            repeated_profiles,
            query_x,
            rank=rank,
        ).reshape_as(target)
        global_eigenspace = global_eigenspace_kernel(
            repeated_profiles,
            query_x,
            rank=rank,
        ).reshape_as(target)
        ranks[str(rank)] = {
            "oracle_best_rank_operator_relative_l2": {
                "mean": float(best_relative.mean()),
                "median": float(best_relative.median()),
                "maximum": float(best_relative.max()),
            },
            "oracle_best_rank_residual_relative_l2": {
                "mean": float(best_residual_relative.mean()),
                "median": float(best_residual_relative.median()),
                "maximum": float(best_residual_relative.max()),
            },
            "oracle_residual_energy_captured": {
                "mean": float(captured_residual.mean()),
                "median": float(captured_residual.median()),
                "minimum": float(captured_residual.min()),
            },
            "oracle_cross_channel_relative_l2": _relative_profile_metrics(
                cross_channel_part(oracle),
                cross_target,
            ),
            "fixed_truncation_operator_relative_l2": _relative_profile_metrics(
                fixed,
                target,
            ),
            "fixed_truncation_residual_relative_l2": _relative_profile_metrics(
                fixed - carrier,
                residual,
            ),
            "fixed_truncation_cross_channel_relative_l2": _relative_profile_metrics(
                cross_channel_part(fixed),
                cross_target,
            ),
            "global_eigenspace_operator_relative_l2": _relative_profile_metrics(
                global_eigenspace,
                target,
            ),
            "global_eigenspace_residual_relative_l2": _relative_profile_metrics(
                global_eigenspace - carrier,
                residual,
            ),
            "global_eigenspace_cross_channel_relative_l2": _relative_profile_metrics(
                cross_channel_part(global_eigenspace),
                cross_target,
            ),
        }
    return {
        "heterogeneity": heterogeneity,
        "coefficient_eigenvalue_minimum": float(torch.linalg.eigvalsh(profiles).min()),
        "coefficient_eigenvalue_maximum": float(torch.linalg.eigvalsh(profiles).max()),
        "diagonal_carrier_operator_relative_l2": _relative_profile_metrics(
            carrier,
            target,
        ),
        "diagonal_carrier_cross_channel_relative_l2": _relative_profile_metrics(
            cross_channel_part(carrier),
            cross_target,
        ),
        "residual_to_operator_norm_ratio": float((residual_norm / target_norm).mean()),
        "cross_channel_to_operator_norm_ratio": float(
            (cross_norm / target_norm).mean()
        ),
        "ranks": ranks,
    }


def run_census(
    *,
    n_profiles: int,
    n_query: int,
    seed: int = SEED,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "study": STUDY,
        "protocol": {
            "channels": CHANNELS,
            "layers": LAYERS,
            "transverse_points": TRANSVERSE_POINTS,
            "profiles_per_level": n_profiles,
            "query_points": n_query,
            "heterogeneity_levels": list(HETEROGENEITY_LEVELS),
            "ranks": list(RANKS),
            "seed": seed,
            "dtype": "float64",
        },
        "levels": {
            str(level): census_level(
                level,
                n_profiles=n_profiles,
                n_query=n_query,
                seed=seed,
                device=device,
            )
            for level in HETEROGENEITY_LEVELS
        },
        "environment": runtime_environment(device),
        "source": source_provenance(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profiles", type=int, default=PROFILES)
    parser.add_argument("--query-points", type=int, default=QUERY_POINTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_census(
        n_profiles=args.profiles,
        n_query=args.query_points,
        seed=args.seed,
        device=torch.device(args.device),
    )
    shared.atomic_write_json(args.output, report)
    print(json.dumps(report["levels"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
