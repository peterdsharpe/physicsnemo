# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Test a path-length-consistent rank-four flow on nonseparable coefficients."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as shared
import nonseparable_latent_composition as base
import torch
from provenance import runtime_environment, source_provenance
from torch import nn

STUDY = "nonseparable_continuum_flow_v1"
LEARNED_ARMS = (
    "gru_path_rank4",
    "sorted_flow_rank4",
    "path_flow_rank4",
)
ANALYTIC_ARMS = (
    "diagonal_carrier",
    "fixed_rank4",
    "oracle_rank4",
)
ARMS = LEARNED_ARMS + ANALYTIC_ARMS
SEEDS = base.SEEDS

RANK = base.RANK
FLOW_CHANNELS = 32
LOCAL_WIDTH = 64
TRAIN_STEPS = base.TRAIN_STEPS
TRAIN_BATCH_SIZE = base.TRAIN_BATCH_SIZE
TRAIN_SAMPLES_PER_LEVEL = base.TRAIN_SAMPLES_PER_LEVEL
EVALUATION_PROFILES = base.EVALUATION_PROFILES
EVALUATION_QUERY_POINTS = base.EVALUATION_QUERY_POINTS
ORDER_PAIRS = base.ORDER_PAIRS
LEARNING_RATE = base.LEARNING_RATE

EVALUATION_SEED = 239_000_211
ORDER_SEED = 241_000_217

SPLITS: dict[str, tuple[float, tuple[int, ...], int, str]] = {
    "in_distribution_8": (0.5, base.TRAIN_X_FREQUENCIES, 8, "smooth"),
    "coarse_4": (0.5, base.TRAIN_X_FREQUENCIES, 4, "smooth"),
    "refined_16": (0.5, base.TRAIN_X_FREQUENCIES, 16, "smooth"),
    "refined_32": (0.5, base.TRAIN_X_FREQUENCIES, 32, "smooth"),
    "fast_8": (0.5, base.FAST_X_FREQUENCIES, 8, "fast"),
    "fast_16": (0.5, base.FAST_X_FREQUENCIES, 16, "fast"),
}


class LatentFlow(nn.Module):
    """Integrate a shared noncommuting affine flow using the true layer width."""

    def __init__(self, *, physical_order: bool) -> None:
        super().__init__()
        self.physical_order = physical_order
        local_features = base.CHANNELS * (base.CHANNELS + 1) // 2 + 3
        local_outputs = RANK * RANK + RANK * FLOW_CHANNELS
        self.local_law = nn.Sequential(
            nn.Linear(local_features, LOCAL_WIDTH),
            nn.SiLU(),
            nn.Linear(LOCAL_WIDTH, local_outputs),
        )
        self.head = base.CorrectionHead(
            RANK * FLOW_CHANNELS,
            low_rank=True,
        )

    def integrate_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Integrate token-conditioned affine dynamics with midpoint updates."""

        n_layers = tokens.shape[-2]
        state = torch.zeros(
            tokens.shape[0],
            RANK,
            FLOW_CHANNELS,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        layer_width = 1.0 / n_layers
        for index in range(n_layers):
            local = self.local_law(tokens[:, index])
            generator = 0.75 * torch.tanh(
                local[:, : RANK * RANK].reshape(-1, RANK, RANK)
            )
            source = torch.tanh(
                local[:, RANK * RANK :].reshape(-1, RANK, FLOW_CHANNELS)
            )
            first_slope = generator @ state + source
            midpoint = state + 0.5 * layer_width * first_slope
            state = state + layer_width * (generator @ midpoint + source)
        return state

    def forward(
        self,
        profiles: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        ordered_profiles = (
            profiles if self.physical_order else base._sort_profiles(profiles)
        )
        local = base.profile_features(ordered_profiles)
        n_layers = profiles.shape[-3]
        positions = (
            torch.arange(
                n_layers,
                device=profiles.device,
                dtype=profiles.dtype,
            )
            + 0.5
        ) / n_layers
        positions = positions.expand(len(profiles), n_layers)
        query = query_x[:, None].expand(len(profiles), n_layers)
        tokens = torch.cat(
            (
                local,
                (2.0 * positions - 1.0)[..., None],
                (2.0 * query - 1.0)[..., None],
                (query * (1.0 - query))[..., None],
            ),
            dim=-1,
        )
        state = self.integrate_tokens(tokens)
        return self.head(state.flatten(start_dim=-2), profiles, query_x)


def build_model(arm: str) -> nn.Module:
    if arm == "gru_path_rank4":
        return base.PathCorrection(physical_order=True)
    if arm == "sorted_flow_rank4":
        return LatentFlow(physical_order=False)
    if arm == "path_flow_rank4":
        return LatentFlow(physical_order=True)
    raise ValueError(f"learned arm must be one of {LEARNED_ARMS}")


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
    profiles, query_x, residual = base.generate_training_dataset(
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


def _family_seed(split: str, evaluation_seed: int) -> int:
    family = SPLITS[split][3]
    return evaluation_seed + (0 if family == "smooth" else 10_000)


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
        },
    }


@torch.no_grad()
def evaluate_order_challenge(
    arm: str,
    model: nn.Module | None,
    *,
    n_pairs: int,
    order_seed: int,
    device: torch.device,
) -> dict[str, float]:
    analytic_arm = arm if model is None else "path_rank4"
    return base.evaluate_order_challenge(
        analytic_arm,
        model,
        n_pairs=n_pairs,
        order_seed=order_seed,
        device=device,
    )


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
            "train_layers": base.LAYERS,
            "train_levels": list(base.TRAIN_LEVELS),
            "train_x_frequencies": list(base.TRAIN_X_FREQUENCIES),
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
            "flow_channels": FLOW_CHANNELS,
            "local_width": LOCAL_WIDTH,
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
