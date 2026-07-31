# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test a rank-four ordered latent surrogate on nonseparable coefficients."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_rank_census as census
import torch
from provenance import runtime_environment, source_provenance
from torch import nn

STUDY = "nonseparable_latent_composition_v1"
LEARNED_ARMS = (
    "global_full",
    "global_rank4",
    "sorted_rank4",
    "path_rank4",
)
ANALYTIC_ARMS = (
    "diagonal_carrier",
    "fixed_rank4",
    "oracle_rank4",
)
ARMS = LEARNED_ARMS + ANALYTIC_ARMS
SEEDS = (17, 29, 43, 59, 71, 83)

CHANNELS = census.CHANNELS
LAYERS = census.LAYERS
TRANSVERSE_POINTS = census.TRANSVERSE_POINTS
RANK = 4
TRAIN_LEVELS = (0.2, 0.5, 0.8)
TRAIN_X_FREQUENCIES = (1, 2)
FAST_X_FREQUENCIES = (3, 4)
TRAIN_SAMPLES_PER_LEVEL = 8_192
TRAIN_STEPS = 4_000
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3
EVALUATION_PROFILES = 128
EVALUATION_QUERY_POINTS = 32
ORDER_PAIRS = 2_048
GLOBAL_WIDTH = 72
PATH_WIDTH = 64

TRAINING_DATA_SEED = 211_000_171
TRAINING_ORDER_SEED = 223_000_181
EVALUATION_SEED = 227_000_191
ORDER_SEED = 229_000_197

SPLITS: dict[str, tuple[float, tuple[int, ...]]] = {
    "in_distribution": (0.5, TRAIN_X_FREQUENCIES),
    "high_heterogeneity": (1.0, TRAIN_X_FREQUENCIES),
    "fast_variation": (0.5, FAST_X_FREQUENCIES),
    "combined_shift": (1.0, FAST_X_FREQUENCIES),
}


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
    heterogeneity: float,
    x_frequencies: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    n_layers: int = LAYERS,
) -> torch.Tensor:
    """Sample positive Fourier--Galerkin profiles with controlled x variation."""

    if not x_frequencies:
        raise ValueError("x_frequencies must not be empty")
    basis, weight = census.cosine_basis(
        TRANSVERSE_POINTS,
        CHANNELS,
        device=device,
        dtype=dtype,
    )
    y = (
        2.0
        * math.pi
        * torch.arange(TRANSVERSE_POINTS, device=device, dtype=dtype)
        / TRANSVERSE_POINTS
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
    log_coefficient = base.expand(n_profiles, n_layers, TRANSVERSE_POINTS).clone()
    frequency_choices = torch.tensor(
        x_frequencies,
        device=device,
        dtype=dtype,
    )
    for harmonic in range(1, census.HARMONICS + 1):
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
        frequency_indices = torch.randint(
            len(x_frequencies),
            (n_profiles, 1, 1),
            generator=generator,
            device=device,
        )
        x_frequency = frequency_choices[frequency_indices]
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
        torch.arange(CHANNELS, device=device, dtype=dtype).square()
    )
    return reaction + transverse_laplacian


def profile_features(profiles: torch.Tensor) -> torch.Tensor:
    """Represent each symmetric local matrix without its redundant triangle."""

    diagonal = torch.diagonal(profiles, dim1=-2, dim2=-1)
    diagonal_features = torch.log(diagonal.clamp_min(1.0e-12)) / 4.0
    row, column = torch.triu_indices(
        CHANNELS,
        CHANNELS,
        offset=1,
        device=profiles.device,
    )
    scale = torch.sqrt(diagonal[..., row] * diagonal[..., column]).clamp_min(1.0e-12)
    coupling_features = profiles[..., row, column] / scale
    return torch.cat((diagonal_features, coupling_features), dim=-1)


def coupling_scale(profiles: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diag_embed(torch.diagonal(profiles, dim1=-2, dim2=-1))
    off_diagonal = profiles - diagonal
    entries = profiles.shape[-3] * CHANNELS * (CHANNELS - 1)
    return torch.linalg.vector_norm(off_diagonal, dim=(-3, -2, -1)) / math.sqrt(entries)


def _sort_profiles(profiles: torch.Tensor) -> torch.Tensor:
    features = profile_features(profiles)
    weights = torch.linspace(
        1.0,
        math.sqrt(2.0),
        features.shape[-1],
        device=profiles.device,
        dtype=profiles.dtype,
    )
    keys = (features * weights).sum(dim=-1)
    indices = torch.argsort(keys, dim=-1)
    return torch.gather(
        profiles,
        -3,
        indices[..., None, None].expand(*indices.shape, CHANNELS, CHANNELS),
    )


class CorrectionHead(nn.Module):
    """Decode either a full or explicitly rank-four kernel correction."""

    def __init__(self, width: int, *, low_rank: bool) -> None:
        super().__init__()
        self.low_rank = low_rank
        self.output = nn.Linear(width, CHANNELS * 2 * CHANNELS)

    def forward(
        self,
        context: torch.Tensor,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.tanh(self.output(context))
        if self.low_rank:
            split = CHANNELS * RANK
            left = raw[..., :split].reshape(*query_x.shape, CHANNELS, RANK)
            right = raw[..., split:].reshape(*query_x.shape, RANK, 2 * CHANNELS)
            normalized = (left @ right) / math.sqrt(RANK)
        else:
            normalized = raw.reshape(*query_x.shape, CHANNELS, 2 * CHANNELS)
        scale = coupling_scale(profiles) * query_x * (1.0 - query_x)
        return scale[..., None, None] * normalized


class GlobalCorrection(nn.Module):
    """Compress the whole ordered profile once, then decode at the query."""

    def __init__(self, *, low_rank: bool) -> None:
        super().__init__()
        feature_count = LAYERS * (CHANNELS * (CHANNELS + 1) // 2) + 2
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, GLOBAL_WIDTH),
            nn.SiLU(),
            nn.Linear(GLOBAL_WIDTH, GLOBAL_WIDTH),
            nn.SiLU(),
        )
        self.head = CorrectionHead(GLOBAL_WIDTH, low_rank=low_rank)

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        if profiles.shape[-3] != LAYERS:
            raise ValueError(f"expected {LAYERS} layers, got {profiles.shape[-3]}")
        query_features = torch.stack(
            (2.0 * query_x - 1.0, query_x * (1.0 - query_x)),
            dim=-1,
        )
        features = torch.cat(
            (
                profile_features(profiles).flatten(start_dim=-2),
                query_features,
            ),
            dim=-1,
        )
        return self.head(self.encoder(features), profiles, query_x)


class PathCorrection(nn.Module):
    """Update a fixed-width latent state once per physical coefficient layer."""

    def __init__(self, *, physical_order: bool) -> None:
        super().__init__()
        self.physical_order = physical_order
        local_features = CHANNELS * (CHANNELS + 1) // 2 + 3
        self.recurrence = nn.GRUCell(local_features, PATH_WIDTH)
        self.head = CorrectionHead(PATH_WIDTH, low_rank=True)

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        ordered_profiles = profiles if self.physical_order else _sort_profiles(profiles)
        local = profile_features(ordered_profiles)
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
        query = query_x[..., None].expand(*query_x.shape, n_layers)
        tokens = torch.cat(
            (
                local,
                (2.0 * positions - 1.0)[..., None],
                (2.0 * query - 1.0)[..., None],
                (query * (1.0 - query))[..., None],
            ),
            dim=-1,
        )
        state = torch.zeros(
            *profiles.shape[:-3],
            PATH_WIDTH,
            device=profiles.device,
            dtype=profiles.dtype,
        )
        for index in range(n_layers):
            state = self.recurrence(tokens[..., index, :], state)
        return self.head(state, profiles, query_x)


def build_model(arm: str) -> nn.Module:
    if arm == "global_full":
        return GlobalCorrection(low_rank=False)
    if arm == "global_rank4":
        return GlobalCorrection(low_rank=True)
    if arm == "sorted_rank4":
        return PathCorrection(physical_order=False)
    if arm == "path_rank4":
        return PathCorrection(physical_order=True)
    raise ValueError(f"learned arm must be one of {LEARNED_ARMS}")


@torch.no_grad()
def oracle_rank_correction(residual: torch.Tensor, *, rank: int = RANK) -> torch.Tensor:
    left, singular_values, right = torch.linalg.svd(residual, full_matrices=False)
    return (left[..., :, :rank] * singular_values[..., None, :rank]) @ right[
        ..., :rank, :
    ]


@torch.no_grad()
def analytic_correction(
    arm: str,
    profiles: torch.Tensor,
    query_x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    if arm == "diagonal_carrier":
        return torch.zeros_like(residual)
    if arm == "fixed_rank4":
        carrier = census.diagonal_carrier_kernel(profiles, query_x)
        return (
            census.fixed_truncation_kernel(
                profiles,
                query_x,
                rank=RANK,
            )
            - carrier
        )
    if arm == "oracle_rank4":
        return oracle_rank_correction(residual)
    raise ValueError(f"analytic arm must be one of {ANALYTIC_ARMS}")


@torch.no_grad()
def generate_training_dataset(
    *,
    samples_per_level: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    profiles_parts = []
    query_parts = []
    residual_parts = []
    for index, heterogeneity in enumerate(TRAIN_LEVELS):
        generator = torch.Generator(device=device)
        generator.manual_seed(TRAINING_DATA_SEED + index)
        profiles = sample_profiles(
            samples_per_level,
            heterogeneity=heterogeneity,
            x_frequencies=TRAIN_X_FREQUENCIES,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        query_x = _uniform(
            (samples_per_level,),
            low=0.03,
            high=0.97,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        target = census.boundary_kernel(profiles, query_x)
        carrier = census.diagonal_carrier_kernel(profiles, query_x)
        profiles_parts.append(profiles)
        query_parts.append(query_x)
        residual_parts.append(target - carrier)
        print(
            f"HEARTBEAT phase=data completed_units={index + 1} "
            f"heterogeneity={heterogeneity}",
            flush=True,
        )
    return (
        torch.cat(profiles_parts),
        torch.cat(query_parts),
        torch.cat(residual_parts),
    )


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    samples_per_level: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if arm not in LEARNED_ARMS:
        raise ValueError(f"arm must be one of {LEARNED_ARMS}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(arm).to(device=device, dtype=torch.float32)
    profiles, query_x, residual = generate_training_dataset(
        samples_per_level=samples_per_level,
        device=device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1.0e-6,
    )
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        generator = torch.Generator(device=device)
        generator.manual_seed(TRAINING_ORDER_SEED + step)
        indices = torch.randint(
            len(profiles),
            (batch_size,),
            generator=generator,
            device=device,
        )
        target = residual[indices]
        prediction = model(profiles[indices], query_x[indices])
        squared_error = (prediction - target).square().sum(dim=(-2, -1))
        squared_scale = target.square().sum(dim=(-2, -1)).clamp_min(1.0e-12)
        loss = (squared_error / squared_scale).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            record = {"step": step, "residual_relative_mse": float(loss.detach())}
            history.append(record)
            print(
                f"HEARTBEAT phase=train completed_units={step} arm={arm} "
                f"seed={seed} loss={record['residual_relative_mse']:.6e}",
                flush=True,
            )
    return model, history


def _profile_relative_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    dimensions = tuple(range(1, target.ndim))
    error = torch.linalg.vector_norm(prediction - target, dim=dimensions)
    scale = torch.linalg.vector_norm(target, dim=dimensions).clamp_min(1.0e-14)
    relative = error / scale
    return {
        "mean": float(relative.mean()),
        "median": float(relative.median()),
        "maximum": float(relative.max()),
    }


@torch.no_grad()
def evaluate_split(
    arm: str,
    model: nn.Module | None,
    split: str,
    *,
    n_profiles: int,
    n_query: int,
    evaluation_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    heterogeneity, x_frequencies = SPLITS[split]
    generator = torch.Generator(device=device)
    generator.manual_seed(evaluation_seed + tuple(SPLITS).index(split))
    profiles = sample_profiles(
        n_profiles,
        heterogeneity=heterogeneity,
        x_frequencies=x_frequencies,
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
    target = census.boundary_kernel(repeated_profiles, query_x)
    carrier = census.diagonal_carrier_kernel(repeated_profiles, query_x)
    residual = target - carrier
    if model is None:
        prediction_residual = analytic_correction(
            arm,
            repeated_profiles,
            query_x,
            residual,
        )
    else:
        prediction_residual = model(repeated_profiles, query_x)
    prediction = carrier + prediction_residual
    target = target.reshape(n_profiles, n_query, CHANNELS, 2 * CHANNELS)
    carrier = carrier.reshape_as(target)
    residual = residual.reshape_as(target)
    prediction = prediction.reshape_as(target)
    prediction_residual = prediction_residual.reshape_as(target)
    cross_target = census.cross_channel_part(target)
    cross_prediction = census.cross_channel_part(prediction)
    return {
        "heterogeneity": heterogeneity,
        "x_frequencies": list(x_frequencies),
        "n_profiles": n_profiles,
        "n_query": n_query,
        "metrics": {
            "operator_relative_l2": _profile_relative_metrics(prediction, target),
            "residual_relative_l2": _profile_relative_metrics(
                prediction_residual,
                residual,
            ),
            "cross_channel_relative_l2": _profile_relative_metrics(
                cross_prediction,
                cross_target,
            ),
            "carrier_operator_relative_l2": _profile_relative_metrics(
                carrier,
                target,
            ),
            "cross_channel_to_operator_norm_ratio": float(
                torch.linalg.vector_norm(cross_target)
                / torch.linalg.vector_norm(target).clamp_min(1.0e-14)
            ),
        },
    }


def order_challenge_profiles(
    n_pairs: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile_a = sample_profiles(
        n_pairs,
        heterogeneity=0.8,
        x_frequencies=TRAIN_X_FREQUENCIES,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    profile_b = profile_a.clone()
    left_index = 1
    right_index = LAYERS - 2
    profile_b[:, left_index] = profile_a[:, right_index]
    profile_b[:, right_index] = profile_a[:, left_index]
    return profile_a, profile_b


@torch.no_grad()
def evaluate_order_challenge(
    arm: str,
    model: nn.Module | None,
    *,
    n_pairs: int,
    order_seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(order_seed)
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

    def target_and_prediction(
        profiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = census.boundary_kernel(profiles, query_x)
        carrier = census.diagonal_carrier_kernel(profiles, query_x)
        residual = target - carrier
        prediction = (
            analytic_correction(arm, profiles, query_x, residual)
            if model is None
            else model(profiles, query_x)
        )
        return residual, prediction

    target_a, prediction_a = target_and_prediction(profile_a)
    target_b, prediction_b = target_and_prediction(profile_b)
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
    return {
        "paired_residual_relative_l2": float(pair_error / field_scale),
        "contrast_relative_l2": float(contrast_error),
        "contrast_recovery_fraction": float(1.0 - contrast_error),
        "predicted_contrast_relative_l2": float(
            torch.linalg.vector_norm(prediction_delta) / delta_scale
        ),
        "true_contrast_to_residual_norm_ratio": float(delta_scale / field_scale),
        "sorted_input_max_abs_difference": float(
            (_sort_profiles(profile_a) - _sort_profiles(profile_b)).abs().max()
        ),
    }


def run_arm(
    *,
    arm: str,
    seed: int,
    device: torch.device,
    train_steps: int = TRAIN_STEPS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    train_samples_per_level: int = TRAIN_SAMPLES_PER_LEVEL,
    evaluation_profiles: int = EVALUATION_PROFILES,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
    order_pairs: int = ORDER_PAIRS,
    evaluation_seed: int = EVALUATION_SEED,
    order_seed: int = ORDER_SEED,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    started = time.time()
    if arm in LEARNED_ARMS:
        model, history = train_model(
            arm,
            seed=seed,
            device=device,
            steps=train_steps,
            batch_size=train_batch_size,
            samples_per_level=train_samples_per_level,
        )
        model = model.to(device=device, dtype=torch.float64).eval()
    else:
        model = None
        history = []
    split_evaluation = {}
    for index, split in enumerate(SPLITS, start=1):
        split_evaluation[split] = evaluate_split(
            arm,
            model,
            split,
            n_profiles=evaluation_profiles,
            n_query=evaluation_query_points,
            evaluation_seed=evaluation_seed,
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
            "training_applied": arm in LEARNED_ARMS,
            "train_levels": list(TRAIN_LEVELS),
            "train_x_frequencies": list(TRAIN_X_FREQUENCIES),
            "fast_x_frequencies": list(FAST_X_FREQUENCIES),
            "train_samples_per_level": (
                train_samples_per_level if arm in LEARNED_ARMS else 0
            ),
            "train_steps": train_steps if arm in LEARNED_ARMS else 0,
            "train_batch_size": train_batch_size,
            "evaluation_profiles": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "evaluation_seed": evaluation_seed,
            "order_pairs": order_pairs,
            "order_seed": order_seed,
            "rank": RANK,
            "global_width": GLOBAL_WIDTH,
            "path_width": PATH_WIDTH,
            "learning_rate": LEARNING_RATE,
            "dtype_training": "float32",
            "dtype_evaluation": "float64",
        },
        "parameters": (
            sum(parameter.numel() for parameter in model.parameters())
            if model is not None
            else 0
        ),
        "training_history": history,
        "split_evaluation": split_evaluation,
        "order_challenge": evaluate_order_challenge(
            arm,
            model,
            n_pairs=order_pairs,
            order_seed=order_seed,
            device=device,
        ),
        "elapsed_seconds": time.time() - started,
        "environment": runtime_environment(device),
        "source": source_provenance(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--train-samples-per-level",
        type=int,
        default=TRAIN_SAMPLES_PER_LEVEL,
    )
    parser.add_argument("--evaluation-profiles", type=int, default=EVALUATION_PROFILES)
    parser.add_argument(
        "--evaluation-query-points",
        type=int,
        default=EVALUATION_QUERY_POINTS,
    )
    parser.add_argument("--order-pairs", type=int, default=ORDER_PAIRS)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_arm(
        arm=args.arm,
        seed=args.seed,
        device=torch.device(args.device),
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        train_samples_per_level=args.train_samples_per_level,
        evaluation_profiles=args.evaluation_profiles,
        evaluation_query_points=args.evaluation_query_points,
        order_pairs=args.order_pairs,
        evaluation_seed=args.evaluation_seed,
        order_seed=args.order_seed,
    )
    shared.atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
