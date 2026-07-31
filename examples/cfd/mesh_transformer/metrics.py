# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics for the conformal Laplace benchmark.

The query points are sampled uniformly in the reference disk.  Consequently,
``area_jacobian`` is the change-of-variables factor needed for physical-domain
area integrals.  All aggregate error metrics in this module use that weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def _validate_vectors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    area_jacobian: torch.Tensor,
) -> None:
    if prediction.ndim != 1 or target.ndim != 1 or area_jacobian.ndim != 1:
        raise ValueError("prediction, target, and area_jacobian must be 1D")
    if prediction.shape != target.shape or target.shape != area_jacobian.shape:
        raise ValueError("prediction, target, and area_jacobian shapes must match")
    if not torch.compiler.is_compiling():
        if not torch.isfinite(prediction).all().item():
            raise ValueError("prediction must be finite")
        if not torch.isfinite(target).all().item():
            raise ValueError("target must be finite")
        if not torch.isfinite(area_jacobian).all().item():
            raise ValueError("area_jacobian must be finite")
        if not (area_jacobian > 0).all().item():
            raise ValueError("area_jacobian must be strictly positive")


def weighted_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    area_jacobian: torch.Tensor,
) -> torch.Tensor:
    r"""Physical-area relative :math:`L^2` error for one domain.

    A zero target has no relative error scale and is rejected explicitly.  The
    benchmark drive sampler normalizes every nonzero boundary condition, so a
    zero denominator indicates an invalid case rather than a numerical corner
    to hide with an arbitrary epsilon.
    """

    _validate_vectors(prediction, target, area_jacobian)
    error_energy = torch.sum(area_jacobian * (prediction - target).square())
    target_energy = torch.sum(area_jacobian * target.square())
    if not torch.compiler.is_compiling() and target_energy.item() <= 0.0:
        raise ValueError("relative L2 error is undefined for a zero target")
    return torch.sqrt(error_energy / target_energy)


def relative_l2(prediction, target) -> float:
    """Plain (unweighted) relative :math:`L^2` error over all elements.

    The one-line reduction dozens of study scripts used to redefine with
    drifting signatures; import this instead of copying it.  Accepts any
    array-like (CPU torch tensors included); returns a Python float in
    float64.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    scale = float(np.linalg.norm(target.ravel()))
    if scale <= 0.0:
        raise ValueError("relative L2 error is undefined for a zero target")
    return float(np.linalg.norm((prediction - target).ravel()) / scale)


def relative_linf(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative pointwise maximum error for one nonzero target."""

    if prediction.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be matching 1D tensors")
    scale = target.abs().max()
    if not torch.compiler.is_compiling() and scale.item() <= 0.0:
        raise ValueError("relative Linf error is undefined for a zero target")
    return (prediction - target).abs().max() / scale


def sampled_boundary_range_violation(
    prediction: torch.Tensor,
    boundary_values: torch.Tensor,
) -> torch.Tensor:
    r"""Normalized excursion beyond the sampled boundary-value range.

    The harmonic maximum principle motivates this diagnostic, but the supplied
    values are the finite boundary-cell samples seen by the surrogate. It is
    therefore deliberately not named an exact maximum-principle test: extrema
    of the continuous trace may lie between samples. The excursion is
    normalized by sampled boundary RMS so constant nonzero data remain
    well-defined.
    """

    if prediction.ndim != 1 or boundary_values.ndim != 1:
        raise ValueError("prediction and boundary_values must be 1D")
    if prediction.numel() == 0 or boundary_values.numel() == 0:
        raise ValueError("prediction and boundary_values must be nonempty")
    upper = torch.relu(prediction.max() - boundary_values.max())
    lower = torch.relu(boundary_values.min() - prediction.min())
    boundary_rms = boundary_values.square().mean().sqrt()
    if not torch.compiler.is_compiling() and boundary_rms.item() <= 0.0:
        raise ValueError("boundary-range normalization requires nonzero data")
    return torch.maximum(upper, lower) / boundary_rms


def certified_maximum_principle_violation(
    prediction: torch.Tensor,
    boundary_lower_bound: torch.Tensor,
    boundary_upper_bound: torch.Tensor,
    boundary_rms: torch.Tensor,
) -> torch.Tensor:
    r"""Violation outside certified bounds on the continuous Dirichlet trace.

    The caller supplies scalar lower and upper enclosures of the exact
    continuous boundary range. A positive result is therefore a genuine
    maximum-principle violation; conservative enclosures may hide a small
    violation but cannot create one.
    """

    if prediction.ndim != 1 or prediction.numel() == 0:
        raise ValueError("prediction must be a nonempty 1D tensor")
    scalars = (boundary_lower_bound, boundary_upper_bound, boundary_rms)
    if any(value.ndim != 0 for value in scalars):
        raise ValueError("boundary bounds and RMS must be scalar tensors")
    if not torch.compiler.is_compiling():
        if not all(torch.isfinite(value).item() for value in scalars):
            raise ValueError("boundary bounds and RMS must be finite")
        if boundary_lower_bound.item() > boundary_upper_bound.item():
            raise ValueError("boundary lower bound exceeds upper bound")
        if boundary_rms.item() <= 0.0:
            raise ValueError("maximum-principle normalization requires nonzero data")
    upper = torch.relu(prediction.max() - boundary_upper_bound)
    lower = torch.relu(boundary_lower_bound - prediction.min())
    return torch.maximum(upper, lower) / boundary_rms


def case_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    area_jacobian: torch.Tensor,
    boundary_values: torch.Tensor,
    preimage_radius: torch.Tensor,
    *,
    boundary_layer_start: float = 0.8,
) -> dict[str, float]:
    """Return complementary accuracy and physics diagnostics for one case."""

    _validate_vectors(prediction, target, area_jacobian)
    if preimage_radius.ndim != 1 or preimage_radius.shape != target.shape:
        raise ValueError("preimage_radius must be 1D and match target")
    if not 0.0 <= boundary_layer_start < 1.0:
        raise ValueError("boundary_layer_start must be in [0, 1)")

    near_boundary = preimage_radius >= boundary_layer_start
    if not near_boundary.any().item():
        raise ValueError("no query points lie in the requested boundary layer")

    return {
        "relative_l2": float(
            weighted_relative_l2(prediction, target, area_jacobian).detach().cpu()
        ),
        "relative_linf": float(relative_linf(prediction, target).detach().cpu()),
        "near_boundary_relative_l2": float(
            weighted_relative_l2(
                prediction[near_boundary],
                target[near_boundary],
                area_jacobian[near_boundary],
            )
            .detach()
            .cpu()
        ),
        "sampled_boundary_range_violation": float(
            sampled_boundary_range_violation(prediction, boundary_values).detach().cpu()
        ),
    }


def aggregate_metrics(cases: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Aggregate case metrics without letting large meshes dominate a split."""

    if not cases:
        raise ValueError("at least one case is required")
    names = tuple(cases[0])
    if any(tuple(case) != names for case in cases):
        raise ValueError("all cases must contain metrics in the same order")

    result: dict[str, float] = {}
    for name in names:
        values = torch.tensor([case[name] for case in cases], dtype=torch.float64)
        if not torch.isfinite(values).all().item():
            raise ValueError(f"metric {name!r} contains a nonfinite value")
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_median"] = float(torch.quantile(values, 0.5))
        result[f"{name}_p90"] = float(torch.quantile(values, 0.9))
    return result


def paired_case_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    resamples: int = 100_000,
    confidence: float = 0.95,
) -> dict[str, float | list[float]]:
    """Estimate a paired case-mean difference and percentile interval.

    Inputs must be aligned continuous cases, for example per-case errors after
    averaging corresponding cases across training seeds for each model. Cases
    are the sampling unit; training seeds are not incorrectly treated as
    independent PDE problems.
    """

    if len(left) != len(right) or not left:
        raise ValueError("left and right must have the same positive length")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    difference = torch.tensor(left, dtype=torch.float64) - torch.tensor(
        right, dtype=torch.float64
    )
    if not torch.isfinite(difference).all().item():
        raise ValueError("paired values must be finite")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_means: list[torch.Tensor] = []
    remaining = resamples
    while remaining:
        chunk = min(remaining, 16_384)
        indices = torch.randint(
            difference.numel(),
            (chunk, difference.numel()),
            generator=generator,
        )
        sample_means.append(difference[indices].mean(dim=1))
        remaining -= chunk
    bootstrap_means = torch.cat(sample_means)
    tail = (1.0 - confidence) / 2.0
    interval = torch.quantile(
        bootstrap_means,
        bootstrap_means.new_tensor((tail, 1.0 - tail)),
    )
    return {
        "mean": float(difference.mean()),
        "case_bootstrap_interval": [float(value) for value in interval],
    }


__all__ = [
    "aggregate_metrics",
    "case_metrics",
    "certified_maximum_principle_violation",
    "paired_case_bootstrap",
    "relative_l2",
    "relative_linf",
    "sampled_boundary_range_violation",
    "weighted_relative_l2",
]
