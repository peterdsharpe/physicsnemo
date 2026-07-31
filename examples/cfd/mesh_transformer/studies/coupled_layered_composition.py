# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test physical-order composition in a coupled two-channel boundary system."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import torch
from provenance import runtime_environment, source_provenance
from torch import nn

STUDY = "coupled_layered_composition_v1"
ARMS = (
    "ordered_pool",
    "sorted_generator",
    "path_generator",
    "analytic_path",
)
LEARNED_ARMS = ARMS[:-1]
GENERATOR_ARMS = ("sorted_generator", "path_generator")
SEEDS = (17, 29, 43, 59, 71)

SPLITS = (
    "in_distribution",
    "held_out_twist",
    "doubled_layers",
)
TRAIN_TWISTS = (1, 2)
HELD_OUT_TWISTS = (3, 4)
TRAIN_LAYERS = 8
DOUBLED_LAYERS = 16
TRAIN_STEPS = 4_000
TRAIN_BATCH_SIZE = 1_024
LEARNING_RATE = 1.0e-3
EVALUATION_PROFILES = 128
EVALUATION_QUERY_POINTS = 32
ORDER_PAIRS = 4_096
TOKEN_WIDTH = 36
HEAD_WIDTH = 56
GENERATOR_WIDTH = 64

TRAINING_SEED = 163_000_117
EVALUATION_SEED = 167_000_129
ORDER_SEED = 173_000_137
CERTIFICATION_SEED = 179_000_143


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


def sample_profiles(
    n_profiles: int,
    *,
    n_layers: int,
    twist_frequencies: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample smooth positive-definite two-channel coefficient profiles."""

    if not twist_frequencies:
        raise ValueError("twist_frequencies must not be empty")
    x = (torch.arange(n_layers, device=device, dtype=dtype) + 0.5) / n_layers
    x = x[None, :]
    shape = (n_profiles, 1)
    low_base = _uniform(
        shape,
        low=0.55,
        high=1.05,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    high_base = _uniform(
        shape,
        low=1.60,
        high=2.60,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    low_amplitude = _uniform(
        shape,
        low=0.05,
        high=0.22,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    high_amplitude = _uniform(
        shape,
        low=0.05,
        high=0.22,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    low_phase = _uniform(
        shape,
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    high_phase = _uniform(
        shape,
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    low_wavenumber = low_base * (
        1.0 + low_amplitude * torch.sin(2.0 * math.pi * x + low_phase)
    )
    high_wavenumber = high_base * (
        1.0 + high_amplitude * torch.cos(2.0 * math.pi * x + high_phase)
    )

    frequency_indices = torch.randint(
        len(twist_frequencies),
        (n_profiles, 1),
        generator=generator,
        device=device,
    )
    frequencies = torch.tensor(
        twist_frequencies,
        device=device,
        dtype=dtype,
    )[frequency_indices]
    angle_offset = _uniform(
        shape,
        low=-math.pi,
        high=math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    angle_amplitude = _uniform(
        shape,
        low=0.35,
        high=1.00,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    angle_phase = _uniform(
        shape,
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    angle = angle_offset + angle_amplitude * torch.sin(
        2.0 * math.pi * frequencies * x + angle_phase
    )

    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    low_eigenvalue = low_wavenumber.square()
    high_eigenvalue = high_wavenumber.square()
    q00 = cosine.square() * low_eigenvalue + sine.square() * high_eigenvalue
    q11 = sine.square() * low_eigenvalue + cosine.square() * high_eigenvalue
    q01 = cosine * sine * (low_eigenvalue - high_eigenvalue)
    row0 = torch.stack((q00, q01), dim=-1)
    row1 = torch.stack((q01, q11), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def generator_matrix(coefficients: torch.Tensor) -> torch.Tensor:
    """Return the first-order generator ``[[0, I], [Q, 0]]``."""

    batch_shape = coefficients.shape[:-2]
    zeros = torch.zeros(
        *batch_shape,
        2,
        2,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    identity = torch.eye(
        2,
        device=coefficients.device,
        dtype=coefficients.dtype,
    ).expand(*batch_shape, 2, 2)
    top = torch.cat((zeros, identity), dim=-1)
    bottom = torch.cat((coefficients, zeros), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def layer_transfer(
    coefficients: torch.Tensor,
    length: torch.Tensor | float,
) -> torch.Tensor:
    """Return the exact state transfer through one constant layer."""

    length_tensor = torch.as_tensor(
        length,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    return torch.matrix_exp(
        generator_matrix(coefficients) * length_tensor[..., None, None]
    )


def total_transfer(profiles: torch.Tensor) -> torch.Tensor:
    """Compose all layer transfers in physical order."""

    batch_shape = profiles.shape[:-3]
    matrix = torch.eye(
        4,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*batch_shape, 4, 4)
    layer_width = 1.0 / profiles.shape[-3]
    for index in range(profiles.shape[-3]):
        matrix = layer_transfer(profiles[..., index, :, :], layer_width) @ matrix
    return matrix


def partial_transfer(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    """Compose transfers from the left boundary to each query location."""

    n_layers = profiles.shape[-3]
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
        4,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*batch_shape, 4, 4)
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
    """Map both two-component Dirichlet boundaries to the interior field."""

    transfer = total_transfer(profiles)
    t11 = transfer[..., :2, :2]
    t12 = transfer[..., :2, 2:]
    identity = torch.eye(
        2,
        device=profiles.device,
        dtype=profiles.dtype,
    ).expand(*profiles.shape[:-3], 2, 2)
    initial_derivative = torch.linalg.solve(
        t12,
        torch.cat((-t11, identity), dim=-1),
    )
    initial_value = torch.cat((identity, torch.zeros_like(identity)), dim=-1)
    initial_state = torch.cat((initial_value, initial_derivative), dim=-2)
    return partial_transfer(profiles, query_x)[..., :2, :] @ initial_state


class OrderedPool(nn.Module):
    """Predict the response from position-aware, mean-pooled coefficient tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.token_network = nn.Sequential(
            nn.Linear(4, TOKEN_WIDTH),
            nn.SiLU(),
            nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(TOKEN_WIDTH + 2, HEAD_WIDTH),
            nn.SiLU(),
            nn.Linear(HEAD_WIDTH, 8),
        )

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        n_layers = profiles.shape[-3]
        positions = (
            torch.arange(
                n_layers,
                device=profiles.device,
                dtype=profiles.dtype,
            )
            + 0.5
        ) / n_layers
        positions = positions.expand(*profiles.shape[:-3], n_layers)
        features = torch.stack(
            (
                profiles[..., 0, 0] / 8.0,
                profiles[..., 0, 1] / 8.0,
                profiles[..., 1, 1] / 8.0,
                2.0 * positions - 1.0,
            ),
            dim=-1,
        )
        context = self.token_network(features).mean(dim=-2)
        query_features = torch.stack(
            (2.0 * query_x - 1.0, query_x * (1.0 - query_x)),
            dim=-1,
        )
        residual = 2.0 * torch.tanh(
            self.head(torch.cat((context, query_features), dim=-1))
        ).reshape(*query_x.shape, 2, 4)

        identity = torch.eye(
            2,
            device=profiles.device,
            dtype=profiles.dtype,
        ).expand(*query_x.shape, 2, 2)
        carrier = torch.cat(
            (
                (1.0 - query_x)[..., None, None] * identity,
                query_x[..., None, None] * identity,
            ),
            dim=-1,
        )
        return (
            carrier
            + query_x[..., None, None] * (1.0 - query_x)[..., None, None] * residual
        )


def _sort_profile_matrices(profiles: torch.Tensor) -> torch.Tensor:
    key = (
        profiles[..., 0, 0]
        + math.sqrt(2.0) * profiles[..., 0, 1]
        + math.pi * profiles[..., 1, 1]
    )
    indices = torch.argsort(key, dim=-1)
    return torch.gather(
        profiles,
        -3,
        indices[..., None, None].expand(*indices.shape, 2, 2),
    )


class LearnedCoupledGenerator(nn.Module):
    """Learn an equivariant local spectral map and compose its state updates."""

    def __init__(self, *, physical_order: bool) -> None:
        super().__init__()
        self.physical_order = physical_order
        self.spectral_network = nn.Sequential(
            nn.Linear(2, GENERATOR_WIDTH),
            nn.SiLU(),
            nn.Linear(GENERATOR_WIDTH, GENERATOR_WIDTH),
            nn.SiLU(),
            nn.Linear(GENERATOR_WIDTH, 1),
        )

    def map_local(self, coefficients: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(coefficients)
        wavenumbers = torch.sqrt(eigenvalues.clamp_min(1.0e-12))
        features = torch.stack(
            (
                wavenumbers / 3.0,
                torch.log(wavenumbers) / 4.0,
            ),
            dim=-1,
        )
        mapped_eigenvalues = torch.nn.functional.softplus(
            self.spectral_network(features).squeeze(-1)
        )
        return (
            eigenvectors
            @ torch.diag_embed(mapped_eigenvalues)
            @ eigenvectors.transpose(-2, -1)
        )

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        local_profiles = (
            profiles if self.physical_order else _sort_profile_matrices(profiles)
        )
        return boundary_kernel(self.map_local(local_profiles), query_x)


class AnalyticPath(nn.Module):
    """Exact coupled path product, used only as a numerical oracle."""

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        return boundary_kernel(profiles, query_x)


def build_model(arm: str) -> nn.Module:
    if arm == "ordered_pool":
        return OrderedPool()
    if arm == "sorted_generator":
        return LearnedCoupledGenerator(physical_order=False)
    if arm == "path_generator":
        return LearnedCoupledGenerator(physical_order=True)
    if arm == "analytic_path":
        return AnalyticPath()
    raise ValueError(f"arm must be one of {ARMS}")


def sample_training_batch(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    profiles = sample_profiles(
        batch_size,
        n_layers=TRAIN_LAYERS,
        twist_frequencies=TRAIN_TWISTS,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query_x = _uniform(
        (batch_size,),
        low=0.02,
        high=0.98,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return profiles, query_x


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train one learned arm on the frozen profile sequence."""

    if arm not in LEARNED_ARMS:
        raise ValueError(f"arm must be one of {LEARNED_ARMS}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(arm).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1.0e-6,
    )
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        generator = torch.Generator(device=device)
        generator.manual_seed(TRAINING_SEED + step)
        profiles, query_x = sample_training_batch(
            batch_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            target = boundary_kernel(profiles, query_x)
        prediction = model(profiles, query_x)
        squared_error = (prediction - target).square().sum(dim=(-2, -1))
        squared_scale = target.square().sum(dim=(-2, -1)).clamp_min(1.0e-8)
        loss = (squared_error / squared_scale).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            record = {"step": step, "operator_relative_mse": float(loss.detach())}
            history.append(record)
            print(
                f"HEARTBEAT phase=train completed_units={step} "
                f"arm={arm} seed={seed} loss={record['operator_relative_mse']:.6e}",
                flush=True,
            )
    return model, history


def _profile_relative_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cross_channel: bool,
) -> dict[str, float]:
    if cross_channel:
        mask = torch.tensor(
            (
                (False, True, False, True),
                (True, False, True, False),
            ),
            device=target.device,
        )
        prediction = prediction[..., mask]
        target = target[..., mask]
    dimensions = tuple(range(1, target.ndim))
    error = torch.linalg.vector_norm(prediction - target, dim=dimensions)
    scale = torch.linalg.vector_norm(target, dim=dimensions).clamp_min(1.0e-14)
    relative = error / scale
    return {
        "mean": float(relative.mean()),
        "median": float(relative.median()),
        "maximum": float(relative.max()),
    }


def _split_protocol(split: str) -> tuple[int, tuple[int, ...]]:
    if split == "in_distribution":
        return TRAIN_LAYERS, TRAIN_TWISTS
    if split == "held_out_twist":
        return TRAIN_LAYERS, HELD_OUT_TWISTS
    if split == "doubled_layers":
        return DOUBLED_LAYERS, TRAIN_TWISTS
    raise ValueError(f"unknown split {split}")


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    split: str,
    *,
    n_profiles: int,
    n_query: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one registered coefficient-profile split."""

    n_layers, twists = _split_protocol(split)
    generator = torch.Generator(device=device)
    generator.manual_seed(EVALUATION_SEED + SPLITS.index(split))
    profiles = sample_profiles(
        n_profiles,
        n_layers=n_layers,
        twist_frequencies=twists,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    query_grid = torch.linspace(
        0.02,
        0.98,
        n_query,
        device=device,
        dtype=torch.float64,
    )
    repeated_profiles = profiles.repeat_interleave(n_query, dim=0)
    query_x = query_grid.repeat(n_profiles)
    target = boundary_kernel(repeated_profiles, query_x).reshape(
        n_profiles,
        n_query,
        2,
        4,
    )
    prediction = model(repeated_profiles, query_x).reshape_as(target)
    zero = torch.zeros(n_profiles, device=device, dtype=torch.float64)
    one = torch.ones_like(zero)
    left_boundary = torch.cat(
        (
            torch.eye(2, device=device, dtype=torch.float64),
            torch.zeros(2, 2, device=device, dtype=torch.float64),
        ),
        dim=-1,
    ).expand(n_profiles, 2, 4)
    right_boundary = torch.cat(
        (
            torch.zeros(2, 2, device=device, dtype=torch.float64),
            torch.eye(2, device=device, dtype=torch.float64),
        ),
        dim=-1,
    ).expand(n_profiles, 2, 4)
    boundary_error = max(
        float((model(profiles, zero) - left_boundary).abs().max()),
        float((model(profiles, one) - right_boundary).abs().max()),
    )
    return {
        "n_profiles": n_profiles,
        "n_query": n_query,
        "n_layers": n_layers,
        "twist_frequencies": list(twists),
        "metrics": {
            "operator_relative_l2": _profile_relative_metrics(
                prediction,
                target,
                cross_channel=False,
            ),
            "cross_channel_relative_l2": _profile_relative_metrics(
                prediction,
                target,
                cross_channel=True,
            ),
            "boundary_max_abs_error": boundary_error,
        },
    }


def order_challenge_profiles(
    n_pairs: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return profiles that differ only by two layers' physical order."""

    profile_a = sample_profiles(
        n_pairs,
        n_layers=TRAIN_LAYERS,
        twist_frequencies=TRAIN_TWISTS,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    profile_b = profile_a.clone()
    left_index = 1
    right_index = TRAIN_LAYERS - 2
    profile_b[:, left_index] = profile_a[:, right_index]
    profile_b[:, right_index] = profile_a[:, left_index]
    return profile_a, profile_b


@torch.no_grad()
def evaluate_order_challenge(
    model: nn.Module,
    *,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure recovery of a response change caused only by layer order."""

    generator = torch.Generator(device=device)
    generator.manual_seed(ORDER_SEED)
    profile_a, profile_b = order_challenge_profiles(
        n_pairs,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    query_x = _uniform(
        (n_pairs,),
        low=0.08,
        high=0.92,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    target_a = boundary_kernel(profile_a, query_x)
    target_b = boundary_kernel(profile_b, query_x)
    prediction_a = model(profile_a, query_x)
    prediction_b = model(profile_b, query_x)
    target_delta = target_a - target_b
    prediction_delta = prediction_a - prediction_b
    delta_scale = torch.linalg.vector_norm(target_delta).clamp_min(1.0e-14)
    field_scale = torch.sqrt(
        0.5
        * (
            torch.linalg.vector_norm(target_a).square()
            + torch.linalg.vector_norm(target_b).square()
        )
    ).clamp_min(1.0e-14)
    pair_error = torch.sqrt(
        0.5
        * (
            torch.linalg.vector_norm(prediction_a - target_a).square()
            + torch.linalg.vector_norm(prediction_b - target_b).square()
        )
    )
    contrast_error = (
        torch.linalg.vector_norm(prediction_delta - target_delta) / delta_scale
    )
    sorted_a = _sort_profile_matrices(profile_a)
    sorted_b = _sort_profile_matrices(profile_b)
    return {
        "paired_field_relative_l2": float(pair_error / field_scale),
        "contrast_relative_l2": float(contrast_error),
        "contrast_recovery_fraction": float(1.0 - contrast_error),
        "predicted_contrast_relative_l2": float(
            torch.linalg.vector_norm(prediction_delta) / delta_scale
        ),
        "true_contrast_relative_l2": float(delta_scale / field_scale),
        "multiset_max_abs_difference": float((sorted_a - sorted_b).abs().max()),
    }


@torch.no_grad()
def evaluate_local_map(
    model: nn.Module,
    *,
    device: torch.device,
) -> dict[str, float] | None:
    """Measure the learned generator's local coefficient map."""

    if not isinstance(model, LearnedCoupledGenerator):
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(EVALUATION_SEED + 97)
    profiles = sample_profiles(
        256,
        n_layers=TRAIN_LAYERS,
        twist_frequencies=HELD_OUT_TWISTS,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    target = profiles.reshape(-1, 2, 2)
    prediction = model.map_local(target)
    relative = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(
        target
    )
    return {
        "relative_frobenius": float(relative),
        "maximum_abs_error": float((prediction - target).abs().max()),
    }


@torch.no_grad()
def certify_reference(device: torch.device) -> dict[str, float]:
    """Certify boundary values, unit determinant, and rotation covariance."""

    generator = torch.Generator(device=device)
    generator.manual_seed(CERTIFICATION_SEED)
    profiles = sample_profiles(
        32,
        n_layers=TRAIN_LAYERS,
        twist_frequencies=HELD_OUT_TWISTS,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    query_x = _uniform(
        (32,),
        low=0.0,
        high=1.0,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    zero = torch.zeros(32, device=device, dtype=torch.float64)
    one = torch.ones_like(zero)
    identity = torch.eye(2, device=device, dtype=torch.float64).expand(32, 2, 2)
    zeros = torch.zeros_like(identity)
    boundary_error = max(
        float(
            (boundary_kernel(profiles, zero) - torch.cat((identity, zeros), dim=-1))
            .abs()
            .max()
        ),
        float(
            (boundary_kernel(profiles, one) - torch.cat((zeros, identity), dim=-1))
            .abs()
            .max()
        ),
    )
    determinant_error = float(
        (torch.linalg.det(total_transfer(profiles)) - 1.0).abs().max()
    )

    angle = torch.tensor(0.37, device=device, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    rotated_profiles = rotation @ profiles @ rotation.transpose(-2, -1)
    kernel = boundary_kernel(profiles, query_x)
    rotated_kernel = boundary_kernel(rotated_profiles, query_x)
    boundary_rotation = torch.block_diag(rotation, rotation)
    expected_rotated = rotation @ kernel @ boundary_rotation.transpose(-2, -1)
    rotation_error = float((rotated_kernel - expected_rotated).abs().max())
    return {
        "boundary_max_abs_error": boundary_error,
        "transfer_determinant_max_abs_error": determinant_error,
        "rotation_covariance_max_abs_error": rotation_error,
    }


def run_arm(
    *,
    arm: str,
    seed: int,
    device: torch.device,
    train_steps: int = TRAIN_STEPS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    evaluation_profiles: int = EVALUATION_PROFILES,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
    order_pairs: int = ORDER_PAIRS,
) -> dict[str, Any]:
    """Train and evaluate one registered coupled-composition arm."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    started = time.time()
    if arm == "analytic_path":
        model = build_model(arm)
        history: list[dict[str, float]] = []
    else:
        model, history = train_model(
            arm,
            seed=seed,
            device=device,
            steps=train_steps,
            batch_size=train_batch_size,
        )
    model = model.to(device=device, dtype=torch.float64).eval()
    split_evaluation = {}
    for index, split in enumerate(SPLITS, start=1):
        split_evaluation[split] = evaluate_split(
            model,
            split,
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
        "arm": arm,
        "seed": seed,
        "protocol": {
            "training_applied": arm != "analytic_path",
            "train_steps": train_steps if arm != "analytic_path" else 0,
            "train_batch_size": train_batch_size,
            "train_layers": TRAIN_LAYERS,
            "doubled_layers": DOUBLED_LAYERS,
            "train_twists": list(TRAIN_TWISTS),
            "held_out_twists": list(HELD_OUT_TWISTS),
            "evaluation_profiles_per_split": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "order_pairs": order_pairs,
            "token_width": TOKEN_WIDTH,
            "head_width": HEAD_WIDTH,
            "generator_width": GENERATOR_WIDTH,
            "learning_rate": LEARNING_RATE,
            "dtype_training": "float32",
            "dtype_evaluation": "float64",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": history,
        "split_evaluation": split_evaluation,
        "order_challenge": evaluate_order_challenge(
            model,
            n_pairs=order_pairs,
            device=device,
        ),
        "local_map_evaluation": evaluate_local_map(model, device=device),
        "reference_certification": certify_reference(device),
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "elapsed_seconds": time.time() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--evaluation-profiles",
        type=int,
        default=EVALUATION_PROFILES,
    )
    parser.add_argument(
        "--evaluation-query-points",
        type=int,
        default=EVALUATION_QUERY_POINTS,
    )
    parser.add_argument("--order-pairs", type=int, default=ORDER_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_arm(
        arm=args.arm,
        seed=args.seed,
        device=torch.device(args.device),
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        evaluation_profiles=args.evaluation_profiles,
        evaluation_query_points=args.evaluation_query_points,
        order_pairs=args.order_pairs,
    )
    shared.atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "seed": report["seed"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
