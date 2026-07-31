# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test whether a fixed elliptic principal part improves kernel transfer."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import numpy as np
import torch
from provenance import runtime_environment, source_provenance
from screened_laplace import modified_bessel_i
from torch import nn

from physicsnemo.mesh import Mesh

STUDY = "drifted_screened_principal_part_v1"
ARMS = ("fixed_principal", "free_principal", "fully_learned")
SEEDS = (17, 29, 43, 59, 71)
TRAINING_SEED = 127_000_051
EVALUATION_SEED = 131_000_057
RESOLUTION_SEED = 137_000_069
TRAIN_STEPS = 8_000
TRAIN_BATCH_SIZE = 4_096
LEARNING_RATE = 1.0e-3
WIDTH = 64
DEPTH = 3
EVALUATION_CASES = 8
EVALUATION_BOUNDARY_POINTS = 96
EVALUATION_QUERY_POINTS = 256
RESOLUTION_CASES = 4
RESOLUTIONS = (64, 128, 256)
QUADRATURE_ORDER = 24
CHECK_QUADRATURE_ORDER = 16
KERNEL_EVALUATION_PAIRS = 65_536
DOUBLE_COEFFICIENT = 1.0 / (2.0 * math.pi)
FEATURE_SYSTEMS = ("raw", "similarity")

TRAINING_RANGES = {
    "radius": (0.04, 2.5),
    "kappa": (0.5, 2.0),
    "drift_magnitude": (0.0, 1.0),
}

KERNEL_SPLITS = {
    "in_distribution": TRAINING_RANGES,
    "near_singular": {
        **TRAINING_RANGES,
        "radius": (5.0e-4, 0.02),
    },
    "ood_low_screening": {
        **TRAINING_RANGES,
        "kappa": (0.05, 0.3),
    },
    "ood_high_screening": {
        **TRAINING_RANGES,
        "kappa": (3.0, 5.0),
    },
    "ood_high_drift": {
        **TRAINING_RANGES,
        "drift_magnitude": (1.5, 2.5),
    },
}

PDE_SPLITS: dict[str, dict[str, Any]] = {
    "in_distribution": {
        "kappa": (0.5, 2.0),
        "drift_magnitude": (0.0, 1.0),
        "deformation": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "query_region": "interior",
    },
    "ood_low_screening": {
        "kappa": (0.05, 0.3),
        "drift_magnitude": (0.0, 1.0),
        "deformation": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "query_region": "interior",
    },
    "ood_high_screening": {
        "kappa": (3.0, 5.0),
        "drift_magnitude": (0.0, 1.0),
        "deformation": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "query_region": "interior",
    },
    "ood_high_drift": {
        "kappa": (0.5, 2.0),
        "drift_magnitude": (1.5, 2.5),
        "deformation": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "query_region": "interior",
    },
    "unseen_geometry": {
        "kappa": (0.5, 2.0),
        "drift_magnitude": (0.0, 1.0),
        "deformation": (0.05, 0.15),
        "geometry_modes": (5, 6, 7),
        "query_region": "interior",
    },
    "near_boundary": {
        "kappa": (0.5, 2.0),
        "drift_magnitude": (0.0, 1.0),
        "deformation": (0.05, 0.15),
        "geometry_modes": (2, 3, 4),
        "query_region": "near_boundary",
    },
}
PDE_SPLIT_ORDER = tuple(PDE_SPLITS)
RESOLUTION_SPLITS = ("in_distribution", "ood_high_drift", "near_boundary")


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


@dataclass(frozen=True)
class KernelBatch:
    radius: torch.Tensor
    normal_dot_direction: torch.Tensor
    drift_dot_direction: torch.Tensor
    kappa: torch.Tensor
    drift_magnitude: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class DriftedScreenedSample:
    boundary: Mesh
    query_points: torch.Tensor
    boundary_values: torch.Tensor
    target: torch.Tensor
    kappa: float
    drift: torch.Tensor


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
        shape, generator=generator, device=device, dtype=dtype
    )


def exact_scaled_double_kernel(
    radius: torch.Tensor,
    normal_dot_direction: torch.Tensor,
    drift_dot_direction: torch.Tensor,
    kappa: torch.Tensor,
    drift_magnitude: torch.Tensor,
) -> torch.Tensor:
    r"""Return ``r K`` for the gauge-transformed double-layer kernel.

    For ``L u = Delta u + b.grad(u) - kappa^2 u``, writing
    ``u(x) = exp(-b.x/2) v(x)`` gives screened Laplace with
    ``lambda^2 = kappa^2 + |b|^2/4``.  Gauge-transforming its canonical
    double layer yields a valid trace with the universal limit
    ``r K -> (n.hat(r))/(2 pi)``.
    """

    lam = torch.sqrt(kappa.square() + 0.25 * drift_magnitude.square())
    scaled_radius = lam * radius
    return (
        DOUBLE_COEFFICIENT
        * torch.exp(-0.5 * radius * drift_dot_direction)
        * scaled_radius
        * torch.special.modified_bessel_k1(scaled_radius)
        * normal_dot_direction
    )


def sample_kernel_batch(
    n_pairs: int,
    *,
    ranges: dict[str, tuple[float, float]],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> KernelBatch:
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    log_radius = _uniform(
        (n_pairs,),
        low=math.log(ranges["radius"][0]),
        high=math.log(ranges["radius"][1]),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    radius = log_radius.exp()
    direction_angle = _uniform(
        (n_pairs,),
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    normal_angle = _uniform(
        (n_pairs,),
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    drift_angle = _uniform(
        (n_pairs,),
        low=0.0,
        high=2.0 * math.pi,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    drift_magnitude = _uniform(
        (n_pairs,),
        low=ranges["drift_magnitude"][0],
        high=ranges["drift_magnitude"][1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    kappa = _uniform(
        (n_pairs,),
        low=ranges["kappa"][0],
        high=ranges["kappa"][1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    normal_dot_direction = torch.cos(normal_angle - direction_angle)
    drift_dot_direction = drift_magnitude * torch.cos(drift_angle - direction_angle)
    target = exact_scaled_double_kernel(
        radius,
        normal_dot_direction,
        drift_dot_direction,
        kappa,
        drift_magnitude,
    )
    return KernelBatch(
        radius=radius,
        normal_dot_direction=normal_dot_direction,
        drift_dot_direction=drift_dot_direction,
        kappa=kappa,
        drift_magnitude=drift_magnitude,
        target=target,
    )


def kernel_features(
    radius: torch.Tensor,
    normal_dot_direction: torch.Tensor,
    drift_dot_direction: torch.Tensor,
    kappa: torch.Tensor,
    drift_magnitude: torch.Tensor,
    *,
    feature_system: str = "raw",
) -> torch.Tensor:
    """Return one registered set of dimensionless rotational invariants."""

    if feature_system == "raw":
        features = (
            torch.log(radius) / 8.0,
            radius / 3.0,
            normal_dot_direction,
            drift_dot_direction / 2.5,
            kappa / 5.0,
            drift_magnitude / 2.5,
        )
    elif feature_system == "similarity":
        lam = torch.sqrt(kappa.square() + 0.25 * drift_magnitude.square())
        scaled_radius = lam * radius
        directional_drift_distance = radius * drift_dot_direction
        features = (
            torch.log(radius) / 8.0,
            radius / 3.0,
            normal_dot_direction,
            directional_drift_distance / 6.25,
            scaled_radius / 12.5,
            torch.log(scaled_radius) / 8.0,
        )
    else:
        raise ValueError(f"feature_system must be one of {FEATURE_SYSTEMS}")
    return torch.stack(features, dim=-1)


def _mlp() -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(6, WIDTH), nn.SiLU()]
    for _ in range(DEPTH - 1):
        layers.extend((nn.Linear(WIDTH, WIDTH), nn.SiLU()))
    output = nn.Linear(WIDTH, 1)
    nn.init.normal_(output.weight, std=1.0e-3)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ScaledKernelModel(nn.Module):
    """One of the three registered representations of ``r K``."""

    def __init__(self, arm: str, *, feature_system: str = "raw") -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if feature_system not in FEATURE_SYSTEMS:
            raise ValueError(f"feature_system must be one of {FEATURE_SYSTEMS}")
        self.arm = arm
        self.feature_system = feature_system
        self.network = _mlp()
        if arm == "free_principal":
            self.singular_coefficient = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("singular_coefficient", None)

    def forward(
        self,
        radius: torch.Tensor,
        normal_dot_direction: torch.Tensor,
        drift_dot_direction: torch.Tensor,
        kappa: torch.Tensor,
        drift_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        learned = self.network(
            kernel_features(
                radius,
                normal_dot_direction,
                drift_dot_direction,
                kappa,
                drift_magnitude,
                feature_system=self.feature_system,
            )
        ).squeeze(-1)
        if self.arm == "fully_learned":
            return learned
        coefficient = (
            DOUBLE_COEFFICIENT
            if self.arm == "fixed_principal"
            else self.singular_coefficient
        )
        return coefficient * normal_dot_direction + radius * learned


def train_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
) -> tuple[ScaledKernelModel, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ScaledKernelModel(arm).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-6
    )
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        batch = sample_kernel_batch(
            batch_size,
            ranges=TRAINING_RANGES,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        prediction = model(
            batch.radius,
            batch.normal_dot_direction,
            batch.drift_dot_direction,
            batch.kappa,
            batch.drift_magnitude,
        )
        loss = (prediction - batch.target).square().mean() / (DOUBLE_COEFFICIENT**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == steps:
            value = float(loss.detach())
            history.append({"step": step, "scaled_kernel_mse": value})
            print(
                f"HEARTBEAT phase=train completed_units={step} "
                f"arm={arm} seed={seed} loss={value:.6e}",
                flush=True,
            )
    return model, history


def regular_drifted_screened_field(
    points: torch.Tensor,
    *,
    kappa: float,
    drift: torch.Tensor,
    modes: tuple[int, ...],
    phases: torch.Tensor,
) -> torch.Tensor:
    """Evaluate an exact globally regular solution of the drifted operator."""

    radius = points.norm(dim=-1)
    angle = torch.atan2(points[:, 1], points[:, 0])
    drift_magnitude = torch.linalg.vector_norm(drift)
    lam = torch.sqrt(
        torch.as_tensor(kappa, device=points.device, dtype=points.dtype).square()
        + 0.25 * drift_magnitude.square()
    )
    screened = torch.zeros_like(radius)
    scale = 1.0 / math.sqrt(len(modes))
    for index, mode in enumerate(modes):
        radial = modified_bessel_i(mode, lam * radius) / modified_bessel_i(mode, lam)
        screened = screened + scale * radial * torch.cos(mode * angle + phases[index])
    return torch.exp(-0.5 * (points * drift).sum(dim=-1)) * screened


def _radial_boundary(
    angles: torch.Tensor,
    *,
    deformation: float,
    modes: tuple[int, ...],
    phases: torch.Tensor,
) -> torch.Tensor:
    radius = torch.ones_like(angles)
    for index, mode in enumerate(modes):
        radius = radius + (deformation / len(modes)) * torch.cos(
            mode * angles + phases[index]
        )
    return radius


def build_pde_sample(
    seed: int,
    *,
    split: str,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    solution_modes: tuple[int, ...] = (0, 1, 2, 3),
) -> DriftedScreenedSample:
    if split not in PDE_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    spec = PDE_SPLITS[split]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(
            torch.empty((), dtype=torch.float64).uniform_(
                low, high, generator=generator
            )
        )

    kappa = uniform(*spec["kappa"])
    drift_magnitude = uniform(*spec["drift_magnitude"])
    drift_angle = uniform(0.0, 2.0 * math.pi)
    drift = drift_magnitude * torch.tensor(
        (math.cos(drift_angle), math.sin(drift_angle)), dtype=torch.float64
    )
    deformation = uniform(*spec["deformation"])
    rotation = uniform(0.0, 2.0 * math.pi)
    geometry_phases = torch.tensor(
        [uniform(0.0, 2.0 * math.pi) for _ in spec["geometry_modes"]],
        dtype=torch.float64,
    )
    solution_phases = torch.tensor(
        [uniform(0.0, 2.0 * math.pi) for _ in solution_modes],
        dtype=torch.float64,
    )

    vertex_angles = (
        rotation
        + 2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    )
    vertex_radius = _radial_boundary(
        vertex_angles,
        deformation=deformation,
        modes=spec["geometry_modes"],
        phases=geometry_phases,
    )
    points = vertex_radius[:, None] * torch.stack(
        (vertex_angles.cos(), vertex_angles.sin()), dim=-1
    )
    indices = torch.arange(n_boundary)
    cells = torch.stack((indices, torch.roll(indices, -1)), dim=-1)
    boundary = Mesh(
        points=points.to(device=device, dtype=dtype),
        cells=cells.to(device=device),
    )

    query_angles = (
        2.0 * math.pi * torch.rand(n_query, dtype=torch.float64, generator=generator)
    )
    if spec["query_region"] == "near_boundary":
        query_fraction = torch.empty(n_query, dtype=torch.float64).uniform_(
            0.97, 0.995, generator=generator
        )
    else:
        query_fraction = 0.9 * torch.sqrt(
            torch.rand(n_query, dtype=torch.float64, generator=generator)
        )
    query_radius = query_fraction * _radial_boundary(
        query_angles,
        deformation=deformation,
        modes=spec["geometry_modes"],
        phases=geometry_phases,
    )
    query_points = query_radius[:, None] * torch.stack(
        (query_angles.cos(), query_angles.sin()), dim=-1
    )

    drift_device = drift.to(device=device, dtype=dtype)
    phases_device = solution_phases.to(device=device, dtype=dtype)
    boundary_values = regular_drifted_screened_field(
        boundary.cell_centroids,
        kappa=kappa,
        drift=drift_device,
        modes=solution_modes,
        phases=phases_device,
    )
    query_device = query_points.to(device=device, dtype=dtype)
    target = regular_drifted_screened_field(
        query_device,
        kappa=kappa,
        drift=drift_device,
        modes=solution_modes,
        phases=phases_device,
    )
    if torch.any(
        torch.sum(boundary.cell_normals * boundary.cell_centroids, dim=-1) >= 0
    ):
        raise RuntimeError("registered domains require inward normals")
    return DriftedScreenedSample(
        boundary=boundary,
        query_points=query_device,
        boundary_values=boundary_values,
        target=target,
        kappa=kappa,
        drift=drift_device,
    )


@lru_cache(maxsize=None)
def _split_gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 4:
        raise ValueError("quadrature order must be at least four")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        np.concatenate(((nodes - 1.0) / 2.0, (nodes + 1.0) / 2.0)),
        np.concatenate((weights / 2.0, weights / 2.0)),
    )


def _learned_scaled_kernel(
    model: ScaledKernelModel,
    radius: torch.Tensor,
    normal_dot_direction: torch.Tensor,
    drift_dot_direction: torch.Tensor,
    kappa: float,
    drift_magnitude: float,
    *,
    chunk_size: int = 262_144,
) -> torch.Tensor:
    shape = radius.shape
    flat_radius = radius.reshape(-1)
    flat_normal = normal_dot_direction.reshape(-1)
    flat_drift = drift_dot_direction.reshape(-1)
    output = torch.empty_like(flat_radius)
    with torch.no_grad():
        for start in range(0, flat_radius.numel(), chunk_size):
            stop = min(start + chunk_size, flat_radius.numel())
            local_radius = flat_radius[start:stop]
            output[start:stop] = model(
                local_radius,
                flat_normal[start:stop],
                flat_drift[start:stop],
                torch.full_like(local_radius, kappa),
                torch.full_like(local_radius, drift_magnitude),
            )
    return output.reshape(shape)


def double_layer_influence(
    query_points: torch.Tensor,
    boundary: Mesh,
    *,
    kappa: float,
    drift: torch.Tensor,
    quadrature_order: int,
    model: ScaledKernelModel | None,
) -> torch.Tensor:
    """Integrate either the exact or learned gauge-transformed double layer."""

    nodes_np, weights_np = _split_gauss_rule(quadrature_order)
    nodes = torch.as_tensor(
        nodes_np, device=query_points.device, dtype=query_points.dtype
    )
    weights = torch.as_tensor(
        weights_np, device=query_points.device, dtype=query_points.dtype
    )
    vertices = boundary.points[boundary.cells]
    midpoint = vertices.mean(dim=1)
    half_edge = 0.5 * (vertices[:, 1] - vertices[:, 0])
    lengths = 2.0 * torch.linalg.vector_norm(half_edge, dim=-1)
    source_points = midpoint[:, None, :] + nodes[None, :, None] * half_edge[:, None, :]
    displacement = query_points[:, None, None, :] - source_points[None, :, :, :]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / radius[..., None]
    normal_dot_direction = torch.einsum(
        "qsgd,sd->qsg", direction, boundary.cell_normals
    )
    drift_dot_direction = torch.einsum("qsgd,d->qsg", direction, drift)
    drift_magnitude = float(torch.linalg.vector_norm(drift))
    if model is None:
        scaled_kernel = exact_scaled_double_kernel(
            radius,
            normal_dot_direction,
            drift_dot_direction,
            torch.full_like(radius, kappa),
            torch.full_like(radius, drift_magnitude),
        )
    else:
        scaled_kernel = _learned_scaled_kernel(
            model,
            radius,
            normal_dot_direction,
            drift_dot_direction,
            kappa,
            drift_magnitude,
        )
    kernel = scaled_kernel / radius
    return (kernel * weights[None, None, :]).sum(dim=-1) * (0.5 * lengths)[None, :]


def trace_matrix(
    boundary: Mesh,
    *,
    kappa: float,
    drift: torch.Tensor,
    quadrature_order: int,
    model: ScaledKernelModel | None,
) -> torch.Tensor:
    influence = double_layer_influence(
        boundary.cell_centroids,
        boundary,
        kappa=kappa,
        drift=drift,
        quadrature_order=quadrature_order,
        model=model,
    )
    return influence + 0.5 * torch.eye(
        boundary.n_cells,
        device=boundary.points.device,
        dtype=boundary.points.dtype,
    )


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float(torch.linalg.vector_norm(actual - expected) / denominator)


def evaluate_pde_case(
    model: ScaledKernelModel,
    *,
    seed: int,
    split: str,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    quadrature_order: int,
    check_quadrature_order: int,
    solution_modes: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, float]:
    sample = build_pde_sample(
        seed,
        split=split,
        n_boundary=n_boundary,
        n_query=n_query,
        device=device,
        solution_modes=solution_modes,
    )
    exact_trace = trace_matrix(
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=quadrature_order,
        model=None,
    )
    check_trace = trace_matrix(
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=check_quadrature_order,
        model=None,
    )
    learned_trace = trace_matrix(
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=quadrature_order,
        model=model,
    )
    exact_field = double_layer_influence(
        sample.query_points,
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=quadrature_order,
        model=None,
    )
    learned_field = double_layer_influence(
        sample.query_points,
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=quadrature_order,
        model=model,
    )
    exact_density = torch.linalg.solve(exact_trace, sample.boundary_values)
    learned_density = torch.linalg.solve(learned_trace, sample.boundary_values)
    oracle_prediction = exact_field @ exact_density
    learned_prediction = learned_field @ learned_density
    return {
        "kappa": sample.kappa,
        "drift_magnitude": float(torch.linalg.vector_norm(sample.drift)),
        "oracle_field_relative_l2": relative_l2(oracle_prediction, sample.target),
        "learned_field_relative_l2": relative_l2(learned_prediction, sample.target),
        "learned_minus_oracle_relative_target_l2": relative_l2(
            learned_prediction, oracle_prediction
        ),
        "exact_trace_residual_relative_l2": relative_l2(
            exact_trace @ learned_density, sample.boundary_values
        ),
        "learned_trace_relative_frobenius": relative_l2(learned_trace, exact_trace),
        "learned_condition_number": float(torch.linalg.cond(learned_trace)),
        "exact_condition_number": float(torch.linalg.cond(exact_trace)),
        "quadrature_relative_frobenius": relative_l2(check_trace, exact_trace),
    }


PDE_METRICS = (
    "oracle_field_relative_l2",
    "learned_field_relative_l2",
    "learned_minus_oracle_relative_target_l2",
    "exact_trace_residual_relative_l2",
    "learned_trace_relative_frobenius",
    "learned_condition_number",
    "exact_condition_number",
    "quadrature_relative_frobenius",
)


def _summary(cases: list[dict[str, float]], key: str) -> dict[str, float]:
    values = torch.tensor([case[key] for case in cases], dtype=torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "maximum": float(values.max()),
    }


def evaluate_kernel_splits(
    model: ScaledKernelModel,
    *,
    device: torch.device,
    n_pairs: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, (split, ranges) in enumerate(KERNEL_SPLITS.items()):
        generator = torch.Generator(device=device).manual_seed(
            EVALUATION_SEED + 1_000_003 * split_index
        )
        batch = sample_kernel_batch(
            n_pairs,
            ranges=ranges,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        with torch.no_grad():
            prediction = model(
                batch.radius,
                batch.normal_dot_direction,
                batch.drift_dot_direction,
                batch.kappa,
                batch.drift_magnitude,
            )
        error = prediction - batch.target
        result[split] = {
            "scaled_kernel_relative_l2": relative_l2(prediction, batch.target),
            "scaled_kernel_rmse_over_principal_scale": float(
                torch.sqrt(error.square().mean()) / DOUBLE_COEFFICIENT
            ),
            "scaled_kernel_max_error_over_principal_scale": float(
                error.abs().max() / DOUBLE_COEFFICIENT
            ),
        }
    return result


def evaluate_pde_splits(
    model: ScaledKernelModel,
    *,
    device: torch.device,
    n_cases: int,
    n_boundary: int,
    n_query: int,
    quadrature_order: int,
    check_quadrature_order: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, split in enumerate(PDE_SPLIT_ORDER):
        cases = [
            evaluate_pde_case(
                model,
                seed=EVALUATION_SEED + 7_919 * case + 1_000_003 * split_index,
                split=split,
                n_boundary=n_boundary,
                n_query=n_query,
                device=device,
                quadrature_order=quadrature_order,
                check_quadrature_order=check_quadrature_order,
            )
            for case in range(n_cases)
        ]
        result[split] = {
            "metrics": {key: _summary(cases, key) for key in PDE_METRICS},
            "cases": cases,
        }
        print(
            f"HEARTBEAT phase=pde completed_units={split_index + 1} split={split}",
            flush=True,
        )
    return result


def evaluate_resolution_splits(
    model: ScaledKernelModel,
    *,
    device: torch.device,
    n_cases: int,
    resolutions: tuple[int, ...],
    n_query: int,
    quadrature_order: int,
    check_quadrature_order: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, split in enumerate(RESOLUTION_SPLITS):
        by_resolution: dict[int, list[dict[str, float]]] = {
            resolution: [] for resolution in resolutions
        }
        for case in range(n_cases):
            seed = RESOLUTION_SEED + 7_919 * case + 1_000_003 * split_index
            samples = [
                build_pde_sample(
                    seed,
                    split=split,
                    n_boundary=resolution,
                    n_query=n_query,
                    device=device,
                )
                for resolution in resolutions
            ]
            if not all(
                torch.equal(samples[0].query_points, sample.query_points)
                and torch.equal(samples[0].target, sample.target)
                for sample in samples[1:]
            ):
                raise RuntimeError("resolution ladder changed the continuum problem")
            for resolution in resolutions:
                by_resolution[resolution].append(
                    evaluate_pde_case(
                        model,
                        seed=seed,
                        split=split,
                        n_boundary=resolution,
                        n_query=n_query,
                        device=device,
                        quadrature_order=quadrature_order,
                        check_quadrature_order=check_quadrature_order,
                    )
                )
        summaries = {
            str(resolution): {
                key: _summary(by_resolution[resolution], key)["mean"]
                for key in (
                    "oracle_field_relative_l2",
                    "learned_field_relative_l2",
                    "exact_trace_residual_relative_l2",
                )
            }
            for resolution in resolutions
        }
        learned_errors = [
            summaries[str(resolution)]["learned_field_relative_l2"]
            for resolution in resolutions
        ]
        result[split] = {
            "resolutions": summaries,
            "learned_field_monotone": all(
                later <= earlier
                for earlier, later in zip(learned_errors, learned_errors[1:])
            ),
        }
        print(
            f"HEARTBEAT phase=resolution "
            f"completed_units={len(PDE_SPLIT_ORDER) + split_index + 1} split={split}",
            flush=True,
        )
    return result


def run_arm(
    *,
    arm: str,
    seed: int,
    device: torch.device,
    train_steps: int = TRAIN_STEPS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    evaluation_cases: int = EVALUATION_CASES,
    evaluation_boundary_points: int = EVALUATION_BOUNDARY_POINTS,
    evaluation_query_points: int = EVALUATION_QUERY_POINTS,
    resolution_cases: int = RESOLUTION_CASES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    quadrature_order: int = QUADRATURE_ORDER,
    check_quadrature_order: int = CHECK_QUADRATURE_ORDER,
    kernel_evaluation_pairs: int = KERNEL_EVALUATION_PAIRS,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    started = time.time()
    model, history = train_model(
        arm,
        seed=seed,
        device=device,
        steps=train_steps,
        batch_size=train_batch_size,
    )
    model = model.to(device=device, dtype=torch.float64).eval()
    report: dict[str, Any] = {
        "study": STUDY,
        "arm": arm,
        "seed": seed,
        "protocol": {
            "training_ranges": TRAINING_RANGES,
            "kernel_splits": KERNEL_SPLITS,
            "pde_splits": PDE_SPLITS,
            "train_steps": train_steps,
            "train_batch_size": train_batch_size,
            "learning_rate": LEARNING_RATE,
            "width": WIDTH,
            "depth": DEPTH,
            "evaluation_cases_per_split": evaluation_cases,
            "evaluation_boundary_points": evaluation_boundary_points,
            "evaluation_query_points": evaluation_query_points,
            "resolution_cases_per_split": resolution_cases,
            "resolutions": list(resolutions),
            "quadrature_order_per_half_panel": quadrature_order,
            "check_quadrature_order_per_half_panel": check_quadrature_order,
            "kernel_evaluation_pairs_per_split": kernel_evaluation_pairs,
            "dtype_training": "float32",
            "dtype_evaluation": "float64",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": history,
        "learned_singular_coefficient": (
            float(model.singular_coefficient.detach())
            if model.singular_coefficient is not None
            else None
        ),
        "kernel_evaluation": evaluate_kernel_splits(
            model,
            device=device,
            n_pairs=kernel_evaluation_pairs,
        ),
        "pde_evaluation": evaluate_pde_splits(
            model,
            device=device,
            n_cases=evaluation_cases,
            n_boundary=evaluation_boundary_points,
            n_query=evaluation_query_points,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        ),
        "resolution_evaluation": evaluate_resolution_splits(
            model,
            device=device,
            n_cases=resolution_cases,
            resolutions=resolutions,
            n_query=evaluation_query_points,
            quadrature_order=quadrature_order,
            check_quadrature_order=check_quadrature_order,
        ),
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "elapsed_seconds": time.time() - started,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--evaluation-cases", type=int, default=EVALUATION_CASES)
    parser.add_argument(
        "--evaluation-boundary-points",
        type=int,
        default=EVALUATION_BOUNDARY_POINTS,
    )
    parser.add_argument(
        "--evaluation-query-points", type=int, default=EVALUATION_QUERY_POINTS
    )
    parser.add_argument("--resolution-cases", type=int, default=RESOLUTION_CASES)
    parser.add_argument("--quadrature-order", type=int, default=QUADRATURE_ORDER)
    parser.add_argument(
        "--check-quadrature-order", type=int, default=CHECK_QUADRATURE_ORDER
    )
    parser.add_argument(
        "--kernel-evaluation-pairs", type=int, default=KERNEL_EVALUATION_PAIRS
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_arm(
        arm=args.arm,
        seed=args.seed,
        device=torch.device(args.device),
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        evaluation_cases=args.evaluation_cases,
        evaluation_boundary_points=args.evaluation_boundary_points,
        evaluation_query_points=args.evaluation_query_points,
        resolution_cases=args.resolution_cases,
        quadrature_order=args.quadrature_order,
        check_quadrature_order=args.check_quadrature_order,
        kernel_evaluation_pairs=args.kernel_evaluation_pairs,
    )
    atomic_write_json(args.output, report)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "output": str(args.output)}))


if __name__ == "__main__":
    main()
