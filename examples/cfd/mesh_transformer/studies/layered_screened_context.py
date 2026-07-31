# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test whether layered elliptic operators require ordered coefficient context."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import _paths  # noqa: F401
import torch
from provenance import runtime_environment, source_provenance
from torch import nn

STUDY = "layered_screened_context_v1"
ARMS = (
    "fixed_optical",
    "scalar_correction",
    "ordered_carrier",
    "ordered_raw",
)
LEARNED_ARMS = ARMS[1:]
ORDERED_ARMS = ARMS[2:]
SEEDS = (17, 29, 43, 59, 71)

N_LAYERS = 8
TRAIN_MODES = (0, 1, 2, 3)
HELD_OUT_MODES = (4, 5, 6, 7)
TRAIN_KAPPA_RANGE = (0.25, 3.0)
LOW_KAPPA_RANGE = (0.05, 0.15)
HIGH_KAPPA_RANGE = (3.5, 5.0)
ORDER_LOW_RANGE = (0.25, 0.5)
ORDER_HIGH_RANGE = (2.75, 3.0)
SPLITS = (
    "in_distribution",
    "held_out_modes",
    "ood_low_coefficient",
    "ood_high_coefficient",
)

TRAINING_SEED = 149_000_081
EVALUATION_SEED = 151_000_087
ORDER_SEED = 157_000_093
TRAIN_STEPS = 4_000
TRAIN_BATCH_SIZE = 2_048
LEARNING_RATE = 1.0e-3
EVALUATION_PROFILES = 128
EVALUATION_QUERY_POINTS = 64
ORDER_PAIRS = 4_096
TOKEN_WIDTH = 48
HEAD_WIDTH = 64
SCALAR_WIDTH = 96
LOG_CORRECTION_SCALE = 10.0


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one complete finite JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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


def layer_transfer(
    q: torch.Tensor,
    length: torch.Tensor | float,
) -> torch.Tensor:
    """Return the exact state-transfer matrix for one constant layer."""

    length_tensor = torch.as_tensor(length, device=q.device, dtype=q.dtype)
    argument = q * length_tensor
    cosine = torch.cosh(argument)
    sine = torch.sinh(argument)
    row_0 = torch.stack((cosine, sine / q), dim=-1)
    row_1 = torch.stack((q * sine, cosine), dim=-1)
    return torch.stack((row_0, row_1), dim=-2)


def mode_wavenumber(
    profiles: torch.Tensor,
    modes: torch.Tensor,
) -> torch.Tensor:
    """Return ``sqrt(n^2 + kappa(x)^2)`` in every layer."""

    return torch.sqrt(profiles.square() + modes[..., None].square())


def lengths_left_of_query(
    query_x: torch.Tensor,
    *,
    n_layers: int,
) -> torch.Tensor:
    """Return how much of each layer lies between zero and each query."""

    layer_width = 1.0 / n_layers
    starts = (
        torch.arange(
            n_layers,
            device=query_x.device,
            dtype=query_x.dtype,
        )
        * layer_width
    )
    return (query_x[..., None] - starts).clamp(min=0.0, max=layer_width)


def total_transfer(q: torch.Tensor) -> torch.Tensor:
    """Compose all layer matrices in physical order."""

    batch_shape = q.shape[:-1]
    matrix = torch.eye(2, device=q.device, dtype=q.dtype).expand(*batch_shape, 2, 2)
    layer_width = 1.0 / q.shape[-1]
    for index in range(q.shape[-1]):
        matrix = layer_transfer(q[..., index], layer_width) @ matrix
    return matrix


def partial_transfer(q: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
    """Compose the transfer from the left boundary to each query."""

    batch_shape = q.shape[:-1]
    matrix = torch.eye(2, device=q.device, dtype=q.dtype).expand(*batch_shape, 2, 2)
    lengths = lengths_left_of_query(query_x, n_layers=q.shape[-1])
    for index in range(q.shape[-1]):
        matrix = layer_transfer(q[..., index], lengths[..., index]) @ matrix
    return matrix


def exact_mode_response(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact Dirichlet-to-interior response of one Fourier mode."""

    q = mode_wavenumber(profiles, modes)
    transfer = total_transfer(q)
    a = transfer[..., 0, 0]
    b = transfer[..., 0, 1]
    left = sides == 0
    initial_value = left.to(dtype=profiles.dtype)
    initial_derivative = torch.where(left, -a / b, 1.0 / b)
    initial_state = torch.stack((initial_value, initial_derivative), dim=-1)
    return (partial_transfer(q, query_x) @ initial_state[..., None])[..., 0, 0]


def optical_summaries(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left, right, and total optical distances."""

    q = mode_wavenumber(profiles, modes)
    n_layers = profiles.shape[-1]
    layer_width = 1.0 / n_layers
    scaled_query = query_x * n_layers
    full_count = torch.floor(scaled_query).to(torch.long).clamp(0, n_layers)
    layer_indices = torch.arange(n_layers, device=profiles.device)
    full_q = torch.where(
        layer_indices < full_count[..., None],
        q,
        torch.zeros_like(q),
    )
    full_sum = torch.sort(full_q, dim=-1).values.sum(dim=-1)
    partial_index = full_count.clamp(max=n_layers - 1)
    partial_q = q.gather(-1, partial_index[..., None]).squeeze(-1)
    partial_length = (query_x - full_count.to(query_x.dtype) * layer_width).clamp(
        min=0.0,
        max=layer_width,
    )
    left = layer_width * full_sum + partial_length * partial_q
    total = layer_width * torch.sort(q, dim=-1).values.sum(dim=-1)
    return left, total - left, total


def fixed_optical_response(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the WKB-like carrier that is exact for constant coefficients."""

    left, right, total = optical_summaries(profiles, modes, query_x)
    source_remainder = torch.where(sides == 0, right, left)
    return torch.sinh(source_remainder) / torch.sinh(total)


def _local_coefficients(
    profiles: torch.Tensor,
    query_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_layers = profiles.shape[-1]
    scaled = query_x * n_layers
    left_index = (torch.ceil(scaled).to(torch.long) - 1).clamp(0, n_layers - 1)
    right_index = torch.floor(scaled).to(torch.long).clamp(0, n_layers - 1)
    return (
        profiles.gather(-1, left_index[..., None]).squeeze(-1),
        profiles.gather(-1, right_index[..., None]).squeeze(-1),
    )


def scalar_features(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Compress a coefficient profile into order-blind optical summaries."""

    optical_left, optical_right, optical_total = optical_summaries(
        profiles,
        modes,
        query_x,
    )
    local_left, local_right = _local_coefficients(profiles, query_x)
    carrier = fixed_optical_response(profiles, modes, query_x, sides)
    side_sign = 2.0 * sides.to(profiles.dtype) - 1.0
    sorted_profiles = torch.sort(profiles, dim=-1).values
    return torch.stack(
        (
            query_x,
            1.0 - query_x,
            side_sign,
            modes / max(HELD_OUT_MODES),
            optical_left / 8.0,
            optical_right / 8.0,
            optical_total / 8.0,
            profiles[..., 0] / 5.0,
            profiles[..., -1] / 5.0,
            local_left / 5.0,
            local_right / 5.0,
            sorted_profiles.mean(dim=-1) / 5.0,
            sorted_profiles.std(dim=-1, unbiased=False) / 3.0,
            sorted_profiles[..., 0] / 5.0,
            sorted_profiles[..., -1] / 5.0,
            carrier,
        ),
        dim=-1,
    )


def raw_global_features(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Return a capacity-matched global context without optical distances."""

    features = scalar_features(profiles, modes, query_x, sides).clone()
    features[..., 4:7] = 0.0
    features[..., -1] = 0.0
    return features


def token_features(
    profiles: torch.Tensor,
    modes: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Attach each coefficient value to its physical position."""

    n_layers = profiles.shape[-1]
    centers = (
        torch.arange(n_layers, device=profiles.device, dtype=profiles.dtype) + 0.5
    ) / n_layers
    centered = 2.0 * centers - 1.0
    relative = centers - query_x[..., None]
    side_sign = 2.0 * sides.to(profiles.dtype) - 1.0
    return torch.stack(
        (
            centered.expand_as(profiles),
            relative,
            profiles / 5.0,
            modes[..., None].expand_as(profiles) / max(HELD_OUT_MODES),
            side_sign[..., None].expand_as(profiles),
            (relative <= 0.0).to(profiles.dtype),
        ),
        dim=-1,
    )


def correction_window(query_x: torch.Tensor, sides: torch.Tensor) -> torch.Tensor:
    """Vanish at the source boundary while preserving the opposite zero trace."""

    return torch.where(sides == 0, query_x, 1.0 - query_x)


def _mlp(input_width: int, hidden_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, 1),
    )


class FixedOptical(nn.Module):
    """Parameter-free optical carrier."""

    def forward(
        self,
        profiles: torch.Tensor,
        modes: torch.Tensor,
        query_x: torch.Tensor,
        sides: torch.Tensor,
    ) -> torch.Tensor:
        return fixed_optical_response(profiles, modes, query_x, sides)


class ScalarCorrection(nn.Module):
    """Learned correction that is blind to layer ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp(16, SCALAR_WIDTH)

    def forward(
        self,
        profiles: torch.Tensor,
        modes: torch.Tensor,
        query_x: torch.Tensor,
        sides: torch.Tensor,
    ) -> torch.Tensor:
        log_correction = self.network(
            scalar_features(profiles, modes, query_x, sides)
        ).squeeze(-1)
        return fixed_optical_response(
            profiles,
            modes,
            query_x,
            sides,
        ) * torch.exp(
            LOG_CORRECTION_SCALE
            * correction_window(query_x, sides)
            * torch.tanh(log_correction)
        )


class OrderedCorrection(nn.Module):
    """Position-aware coefficient encoder with one declared base response."""

    def __init__(self, *, carrier: bool) -> None:
        super().__init__()
        self.carrier = carrier
        self.token_network = nn.Sequential(
            nn.Linear(6, TOKEN_WIDTH),
            nn.SiLU(),
            nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH),
            nn.SiLU(),
        )
        self.head = _mlp(TOKEN_WIDTH + 16, HEAD_WIDTH)

    def forward(
        self,
        profiles: torch.Tensor,
        modes: torch.Tensor,
        query_x: torch.Tensor,
        sides: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.token_network(
            token_features(profiles, modes, query_x, sides)
        ).mean(dim=-2)
        global_features = (
            scalar_features(profiles, modes, query_x, sides)
            if self.carrier
            else raw_global_features(profiles, modes, query_x, sides)
        )
        log_correction = self.head(
            torch.cat((encoded, global_features), dim=-1)
        ).squeeze(-1)
        base = (
            fixed_optical_response(profiles, modes, query_x, sides)
            if self.carrier
            else torch.where(sides == 0, 1.0 - query_x, query_x)
        )
        return base * torch.exp(
            LOG_CORRECTION_SCALE
            * correction_window(query_x, sides)
            * torch.tanh(log_correction)
        )


def build_model(arm: str) -> nn.Module:
    if arm == "fixed_optical":
        return FixedOptical()
    if arm == "scalar_correction":
        return ScalarCorrection()
    if arm == "ordered_carrier":
        return OrderedCorrection(carrier=True)
    if arm == "ordered_raw":
        return OrderedCorrection(carrier=False)
    raise ValueError(f"arm must be one of {ARMS}")


def sample_training_batch(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    profiles = _uniform(
        (batch_size, N_LAYERS),
        low=TRAIN_KAPPA_RANGE[0],
        high=TRAIN_KAPPA_RANGE[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    modes = torch.randint(
        min(TRAIN_MODES),
        max(TRAIN_MODES) + 1,
        (batch_size,),
        generator=generator,
        device=device,
    ).to(dtype)
    query_x = _uniform(
        (batch_size,),
        low=0.02,
        high=0.98,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    sides = torch.randint(
        0,
        2,
        (batch_size,),
        generator=generator,
        device=device,
    )
    return profiles, modes, query_x, sides


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train one registered learned arm on identical deterministic batches."""

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
        inputs = sample_training_batch(
            batch_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            target = exact_mode_response(*inputs)
        prediction = model(*inputs)
        loss = (torch.log(prediction) - torch.log(target)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            record = {"step": step, "log_amplitude_mse": float(loss.detach())}
            history.append(record)
            print(
                f"HEARTBEAT phase=train completed_units={step} "
                f"arm={arm} seed={seed} loss={record['log_amplitude_mse']:.6e}",
                flush=True,
            )
    return model, history


def _split_spec(split: str) -> tuple[tuple[float, float], tuple[int, ...]]:
    if split == "in_distribution":
        return TRAIN_KAPPA_RANGE, TRAIN_MODES
    if split == "held_out_modes":
        return TRAIN_KAPPA_RANGE, HELD_OUT_MODES
    if split == "ood_low_coefficient":
        return LOW_KAPPA_RANGE, TRAIN_MODES
    if split == "ood_high_coefficient":
        return HIGH_KAPPA_RANGE, TRAIN_MODES
    raise ValueError(f"split must be one of {SPLITS}")


def _evaluation_inputs(
    split: str,
    *,
    n_profiles: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    kappa_range, mode_values = _split_spec(split)
    generator = torch.Generator(device=device)
    generator.manual_seed(EVALUATION_SEED + SPLITS.index(split))
    base_profiles = _uniform(
        (n_profiles, N_LAYERS),
        low=kappa_range[0],
        high=kappa_range[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query_values = torch.linspace(0.02, 0.98, n_query, device=device, dtype=dtype)
    mode_tensor = torch.tensor(mode_values, device=device, dtype=dtype)
    side_tensor = torch.arange(2, device=device)
    profiles = base_profiles[:, None, None, None, :].expand(
        -1,
        len(mode_values),
        2,
        n_query,
        -1,
    )
    modes = mode_tensor[None, :, None, None].expand(
        n_profiles,
        -1,
        2,
        n_query,
    )
    sides = side_tensor[None, None, :, None].expand(
        n_profiles,
        len(mode_values),
        -1,
        n_query,
    )
    query_x = query_values[None, None, None, :].expand(
        n_profiles,
        len(mode_values),
        2,
        -1,
    )
    return (
        profiles.reshape(-1, N_LAYERS),
        modes.reshape(-1),
        query_x.reshape(-1),
        sides.reshape(-1),
        mode_values,
    )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    split: str,
    *,
    n_profiles: int,
    n_query: int,
    device: torch.device,
) -> dict[str, Any]:
    inputs = _evaluation_inputs(
        split,
        n_profiles=n_profiles,
        n_query=n_query,
        device=device,
        dtype=torch.float64,
    )
    profiles, modes, query_x, sides, mode_values = inputs
    prediction = model(profiles, modes, query_x, sides)
    target = exact_mode_response(profiles, modes, query_x, sides)
    shape = (n_profiles, len(mode_values), 2, n_query)
    prediction = prediction.reshape(shape)
    target = target.reshape(shape)
    relative = torch.sqrt(
        (prediction - target).square().sum(dim=-1) / target.square().sum(dim=-1)
    ).reshape(-1)
    log_error = (torch.log(prediction) - torch.log(target)).reshape(-1)
    return {
        "modes": list(mode_values),
        "metrics": {
            "field_relative_l2": {
                "mean": float(relative.mean()),
                "median": float(relative.median()),
                "maximum": float(relative.max()),
            },
            "log_amplitude_rmse": float(torch.sqrt(log_error.square().mean())),
        },
    }


def order_challenge_profiles(
    n_pairs: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct profile pairs differing only by two within-range layer orders."""

    generator = torch.Generator(device=device)
    generator.manual_seed(ORDER_SEED)
    profile_a = _uniform(
        (n_pairs, N_LAYERS),
        low=TRAIN_KAPPA_RANGE[0],
        high=TRAIN_KAPPA_RANGE[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    low = _uniform(
        (n_pairs,),
        low=ORDER_LOW_RANGE[0],
        high=ORDER_LOW_RANGE[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    high = _uniform(
        (n_pairs,),
        low=ORDER_HIGH_RANGE[0],
        high=ORDER_HIGH_RANGE[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    profile_a[:, 1] = low
    profile_a[:, 2] = high
    profile_b = profile_a.clone()
    profile_b[:, 1] = high
    profile_b[:, 2] = low
    modes = torch.randint(
        min(TRAIN_MODES),
        max(TRAIN_MODES) + 1,
        (n_pairs,),
        generator=generator,
        device=device,
    ).to(dtype)
    sides = torch.randint(
        0,
        2,
        (n_pairs,),
        generator=generator,
        device=device,
    )
    query_x = torch.full((n_pairs,), 0.5, device=device, dtype=dtype)
    return profile_a, profile_b, modes, query_x, sides


@torch.no_grad()
def evaluate_order_challenge(
    model: nn.Module,
    *,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    profile_a, profile_b, modes, query_x, sides = order_challenge_profiles(
        n_pairs,
        device=device,
        dtype=torch.float64,
    )
    scalar_difference = (
        scalar_features(profile_a, modes, query_x, sides)
        - scalar_features(profile_b, modes, query_x, sides)
    ).abs()
    target_a = exact_mode_response(profile_a, modes, query_x, sides)
    target_b = exact_mode_response(profile_b, modes, query_x, sides)
    prediction_a = model(profile_a, modes, query_x, sides)
    prediction_b = model(profile_b, modes, query_x, sides)
    target_contrast = target_a - target_b
    prediction_contrast = prediction_a - prediction_b
    pair_error = (prediction_a - target_a).square() + (prediction_b - target_b).square()
    pair_target = target_a.square() + target_b.square()
    contrast_error = (prediction_contrast - target_contrast).square()
    contrast_target = target_contrast.square()
    contrast_relative_l2 = torch.sqrt(contrast_error.sum() / contrast_target.sum())
    return {
        "paired_field_relative_l2": float(
            torch.sqrt(pair_error.sum() / pair_target.sum())
        ),
        "contrast_relative_l2": float(contrast_relative_l2),
        "contrast_recovery_fraction": float(1.0 - contrast_relative_l2),
        "true_contrast_relative_l2": float(
            torch.sqrt(2.0 * contrast_target.sum() / pair_target.sum())
        ),
        "predicted_contrast_relative_l2": float(
            torch.sqrt(2.0 * prediction_contrast.square().sum() / pair_target.sum())
        ),
        "scalar_input_max_abs_difference": float(scalar_difference.max()),
    }


@torch.no_grad()
def certify_reference(device: torch.device) -> dict[str, float]:
    """Cross-check transfer labels against exact constant-profile identities."""

    generator = torch.Generator(device=device)
    generator.manual_seed(163_000_109)
    samples = 1_024
    coefficient = _uniform(
        (samples, 1),
        low=TRAIN_KAPPA_RANGE[0],
        high=TRAIN_KAPPA_RANGE[1],
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    profiles = coefficient.expand(-1, N_LAYERS)
    modes = torch.randint(
        0,
        max(HELD_OUT_MODES) + 1,
        (samples,),
        generator=generator,
        device=device,
    ).to(torch.float64)
    query_x = _uniform(
        (samples,),
        low=0.0,
        high=1.0,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    sides = torch.randint(
        0,
        2,
        (samples,),
        generator=generator,
        device=device,
    )
    exact = exact_mode_response(profiles, modes, query_x, sides)
    carrier = fixed_optical_response(profiles, modes, query_x, sides)

    random_profiles = _uniform(
        (samples, N_LAYERS),
        low=TRAIN_KAPPA_RANGE[0],
        high=TRAIN_KAPPA_RANGE[1],
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    random_modes = torch.randint(
        0,
        max(HELD_OUT_MODES) + 1,
        (samples,),
        generator=generator,
        device=device,
    ).to(torch.float64)
    q = mode_wavenumber(random_profiles, random_modes)
    determinant = torch.linalg.det(total_transfer(q))
    left_side = torch.zeros(samples, device=device, dtype=torch.long)
    right_side = torch.ones(samples, device=device, dtype=torch.long)
    zeros = torch.zeros(samples, device=device, dtype=torch.float64)
    ones = torch.ones(samples, device=device, dtype=torch.float64)
    boundary_errors = torch.stack(
        (
            exact_mode_response(random_profiles, random_modes, zeros, left_side) - 1.0,
            exact_mode_response(random_profiles, random_modes, ones, left_side),
            exact_mode_response(random_profiles, random_modes, zeros, right_side),
            exact_mode_response(random_profiles, random_modes, ones, right_side) - 1.0,
        )
    ).abs()
    return {
        "constant_profile_max_abs_error": float((exact - carrier).abs().max()),
        "transfer_determinant_max_abs_error": float((determinant - 1.0).abs().max()),
        "boundary_max_abs_error": float(boundary_errors.max()),
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
    """Train and evaluate one registered arm."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    started = time.time()
    if arm == "fixed_optical":
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
            "training_applied": arm != "fixed_optical",
            "train_steps": train_steps if arm != "fixed_optical" else 0,
            "train_batch_size": train_batch_size,
            "train_modes": list(TRAIN_MODES),
            "held_out_modes": list(HELD_OUT_MODES),
            "n_layers": N_LAYERS,
            "train_kappa_range": list(TRAIN_KAPPA_RANGE),
            "low_kappa_range": list(LOW_KAPPA_RANGE),
            "high_kappa_range": list(HIGH_KAPPA_RANGE),
            "evaluation_profiles_per_split": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "order_pairs": order_pairs,
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
    atomic_write_json(args.output, report)
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
