# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test path-ordered composition of a learned local elliptic generator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as base
import torch
from provenance import runtime_environment, source_provenance
from torch import nn

STUDY = "layered_screened_generator_v1"
ARMS = (
    "scalar_correction",
    "ordered_raw",
    "sorted_generator",
    "path_generator",
    "analytic_path",
)
LEARNED_ARMS = ARMS[:-1]
GENERATOR_ARMS = ("sorted_generator", "path_generator")
CONTROL_ARMS = ("scalar_correction", "ordered_raw")
SEEDS = base.SEEDS

TRAIN_STEPS = base.TRAIN_STEPS
TRAIN_BATCH_SIZE = base.TRAIN_BATCH_SIZE
EVALUATION_PROFILES = base.EVALUATION_PROFILES
EVALUATION_QUERY_POINTS = base.EVALUATION_QUERY_POINTS
ORDER_PAIRS = base.ORDER_PAIRS
GENERATOR_WIDTH = 64
MAX_LOCAL_POTENTIAL = 36.0


def response_from_wavenumber(
    q: torch.Tensor,
    query_x: torch.Tensor,
    sides: torch.Tensor,
) -> torch.Tensor:
    """Solve the two-boundary response induced by local wavenumbers."""

    transfer = base.total_transfer(q)
    a = transfer[..., 0, 0]
    b = transfer[..., 0, 1]
    left = sides == 0
    initial_value = left.to(dtype=q.dtype)
    initial_derivative = torch.where(left, -a / b, 1.0 / b)
    initial_state = torch.stack((initial_value, initial_derivative), dim=-1)
    return (base.partial_transfer(q, query_x) @ initial_state[..., None])[..., 0, 0]


class LearnedLocalGenerator(nn.Module):
    """Infer one positive local potential and compose its state maps."""

    def __init__(self, *, physical_order: bool) -> None:
        super().__init__()
        self.physical_order = physical_order
        self.potential_network = nn.Sequential(
            nn.Linear(2, GENERATOR_WIDTH),
            nn.SiLU(),
            nn.Linear(GENERATOR_WIDTH, GENERATOR_WIDTH),
            nn.SiLU(),
            nn.Linear(GENERATOR_WIDTH, 1),
        )

    def forward(
        self,
        profiles: torch.Tensor,
        modes: torch.Tensor,
        query_x: torch.Tensor,
        sides: torch.Tensor,
    ) -> torch.Tensor:
        local_profiles = (
            profiles if self.physical_order else torch.sort(profiles, dim=-1).values
        )
        local_features = torch.stack(
            (
                local_profiles / 5.0,
                torch.log(local_profiles) / 4.0,
            ),
            dim=-1,
        )
        potential = torch.nn.functional.softplus(
            self.potential_network(local_features).squeeze(-1)
        ).clamp(max=MAX_LOCAL_POTENTIAL)
        q = torch.sqrt(modes[..., None].square() + potential)
        return response_from_wavenumber(q, query_x, sides)


class AnalyticPath(nn.Module):
    """Exact path-ordered generator, used only as a numerical oracle."""

    def forward(
        self,
        profiles: torch.Tensor,
        modes: torch.Tensor,
        query_x: torch.Tensor,
        sides: torch.Tensor,
    ) -> torch.Tensor:
        return base.exact_mode_response(profiles, modes, query_x, sides)


def build_model(arm: str) -> nn.Module:
    if arm == "scalar_correction":
        return base.ScalarCorrection()
    if arm == "ordered_raw":
        return base.OrderedCorrection(carrier=False)
    if arm == "sorted_generator":
        return LearnedLocalGenerator(physical_order=False)
    if arm == "path_generator":
        return LearnedLocalGenerator(physical_order=True)
    if arm == "analytic_path":
        return AnalyticPath()
    raise ValueError(f"arm must be one of {ARMS}")


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train one learned arm on the frozen layered-profile sequence."""

    if arm not in LEARNED_ARMS:
        raise ValueError(f"arm must be one of {LEARNED_ARMS}")
    if arm in CONTROL_ARMS:
        return base.train_model(
            arm,
            seed=seed,
            device=device,
            steps=steps,
            batch_size=batch_size,
        )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(arm).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=1.0e-6,
    )
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        generator = torch.Generator(device=device)
        generator.manual_seed(base.TRAINING_SEED + step)
        inputs = base.sample_training_batch(
            batch_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            target = base.exact_mode_response(*inputs)
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
    """Train and evaluate one registered composition arm."""

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
    for index, split in enumerate(base.SPLITS, start=1):
        split_evaluation[split] = base.evaluate_split(
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
            "train_modes": list(base.TRAIN_MODES),
            "held_out_modes": list(base.HELD_OUT_MODES),
            "n_layers": base.N_LAYERS,
            "train_kappa_range": list(base.TRAIN_KAPPA_RANGE),
            "low_kappa_range": list(base.LOW_KAPPA_RANGE),
            "high_kappa_range": list(base.HIGH_KAPPA_RANGE),
            "evaluation_profiles_per_split": evaluation_profiles,
            "evaluation_query_points": evaluation_query_points,
            "order_pairs": order_pairs,
            "learning_rate": base.LEARNING_RATE,
            "generator_width": GENERATOR_WIDTH,
            "dtype_training": "float32",
            "dtype_evaluation": "float64",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": history,
        "split_evaluation": split_evaluation,
        "order_challenge": base.evaluate_order_challenge(
            model,
            n_pairs=order_pairs,
            device=device,
        ),
        "reference_certification": base.certify_reference(device),
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
    base.atomic_write_json(args.output, report)
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
