# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test whether broader coefficient spectra reveal a transferable local flow."""

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
from torch import nn

STUDY = "nonseparable_frequency_coverage_v1"
LEARNED_ARMS = ("narrow_flow", "broad_flow")
ANALYTIC_ARMS = ("diagonal_carrier", "fixed_rank4", "oracle_rank4")
ARMS = LEARNED_ARMS + ANALYTIC_ARMS
SEEDS = base.SEEDS

TRAIN_LAYERS = 16
TRAINING_FREQUENCIES = {
    "narrow_flow": (1, 2),
    "broad_flow": (1, 2, 3, 4),
}
TRAIN_STEPS = flow.TRAIN_STEPS
TRAIN_BATCH_SIZE = flow.TRAIN_BATCH_SIZE
TRAIN_SAMPLES_PER_LEVEL = flow.TRAIN_SAMPLES_PER_LEVEL
EVALUATION_PROFILES = flow.EVALUATION_PROFILES
EVALUATION_QUERY_POINTS = flow.EVALUATION_QUERY_POINTS
LEARNING_RATE = flow.LEARNING_RATE

EVALUATION_SEED = 251_000_227

SPLITS: dict[str, tuple[float, tuple[int, ...], int, str]] = {
    "smooth_16": (0.5, (1, 2), 16, "smooth"),
    "covered_fast_16": (0.5, (3, 4), 16, "covered"),
    "unseen_16": (0.5, (5, 6), 16, "unseen"),
    "unseen_32": (0.5, (5, 6), 32, "unseen"),
}
FAMILY_SEED_OFFSETS = {
    "smooth": 0,
    "covered": 10_000,
    "unseen": 20_000,
}


@torch.no_grad()
def generate_training_dataset(
    *,
    x_frequencies: tuple[int, ...],
    samples_per_level: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a matched-budget dataset with the requested spectral support."""

    profiles_parts = []
    query_parts = []
    residual_parts = []
    for index, heterogeneity in enumerate(base.TRAIN_LEVELS):
        generator = torch.Generator(device=device)
        generator.manual_seed(base.TRAINING_DATA_SEED + index)
        profiles = base.sample_profiles(
            samples_per_level,
            heterogeneity=heterogeneity,
            x_frequencies=x_frequencies,
            generator=generator,
            device=device,
            dtype=dtype,
            n_layers=TRAIN_LAYERS,
        )
        query_x = base._uniform(
            (samples_per_level,),
            low=0.03,
            high=0.97,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        target = base.census.boundary_kernel(profiles, query_x)
        carrier = base.census.diagonal_carrier_kernel(profiles, query_x)
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
    model = flow.LatentFlow(physical_order=True).to(
        device=device,
        dtype=torch.float32,
    )
    profiles, query_x, residual = generate_training_dataset(
        x_frequencies=TRAINING_FREQUENCIES[arm],
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
        generator.manual_seed(base.TRAINING_ORDER_SEED + step)
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


def _family_seed(split: str, evaluation_seed: int) -> int:
    family = SPLITS[split][3]
    return evaluation_seed + FAMILY_SEED_OFFSETS[family]


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
    heterogeneity, x_frequencies, n_layers, _ = SPLITS[split]
    generator = torch.Generator(device=device)
    generator.manual_seed(_family_seed(split, evaluation_seed))
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
    if model is None:
        prediction_residual = base.analytic_correction(
            arm,
            repeated_profiles,
            query_x,
            residual,
        )
    else:
        prediction_residual = model(repeated_profiles, query_x)
    prediction = carrier + prediction_residual
    target = target.reshape(n_profiles, n_query, base.CHANNELS, 2 * base.CHANNELS)
    carrier = carrier.reshape_as(target)
    residual = residual.reshape_as(target)
    prediction = prediction.reshape_as(target)
    prediction_residual = prediction_residual.reshape_as(target)
    cross_target = base.census.cross_channel_part(target)
    cross_prediction = base.census.cross_channel_part(prediction)
    return {
        "heterogeneity": heterogeneity,
        "x_frequencies": list(x_frequencies),
        "n_layers": n_layers,
        "n_profiles": n_profiles,
        "n_query": n_query,
        "metrics": {
            "operator_relative_l2": flow._profile_relative_metrics(
                prediction,
                target,
            ),
            "residual_relative_l2": flow._profile_relative_metrics(
                prediction_residual,
                residual,
            ),
            "cross_channel_relative_l2": flow._profile_relative_metrics(
                cross_prediction,
                cross_target,
            ),
            "carrier_operator_relative_l2": flow._profile_relative_metrics(
                carrier,
                target,
            ),
        },
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
    evaluation_seed: int = EVALUATION_SEED,
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
            "train_layers": TRAIN_LAYERS,
            "train_levels": list(base.TRAIN_LEVELS),
            "train_x_frequencies": (
                list(TRAINING_FREQUENCIES[arm]) if arm in LEARNED_ARMS else []
            ),
            "train_samples_per_level": (
                train_samples_per_level if arm in LEARNED_ARMS else 0
            ),
            "train_steps": train_steps if arm in LEARNED_ARMS else 0,
            "train_batch_size": train_batch_size,
            "evaluation_profiles": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "evaluation_seed": evaluation_seed,
            "rank": flow.RANK,
            "flow_channels": flow.FLOW_CHANNELS,
            "local_width": flow.LOCAL_WIDTH,
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
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
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
        evaluation_seed=args.evaluation_seed,
    )
    shared.atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
