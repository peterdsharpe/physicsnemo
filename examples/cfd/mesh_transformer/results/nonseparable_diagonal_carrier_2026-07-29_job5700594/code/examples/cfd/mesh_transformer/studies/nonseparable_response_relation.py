# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate a nonseparable response-direction transfer instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import nonseparable_rank_census as spectral
import torch

INPUT_CHANNELS = 8
POTENTIAL_SCALE = 5.0
TARGET_LENGTH = 1.0
TARGET_SCREENING = 0.0
SOURCE_RADIUS = 0.15
BISECTION_STEPS = 32


@dataclass(frozen=True)
class Resolution:
    name: str
    galerkin_channels: int
    longitudinal_layers: int
    transverse_quadrature_points: int
    interior_query_points: int


RESOLUTIONS = (
    Resolution("coarse", 10, 16, 128, 64),
    Resolution("medium", 14, 32, 192, 64),
    Resolution("reference", 18, 64, 256, 64),
)


def coefficient_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the frozen positive, nonseparable coefficient field."""

    log_coefficient = (
        math.log(0.8)
        + torch.sin(2.0 * math.pi * x[:, None] + 0.3) * torch.cos(y)[None, :]
        + 0.45
        * torch.sin(4.0 * math.pi * x[:, None] - 0.8)
        * torch.cos(2.0 * y)[None, :]
        + 0.25
        * torch.sin(2.0 * math.pi * x[:, None] + 1.2)
        * torch.cos(3.0 * y)[None, :]
    )
    return torch.exp(log_coefficient)


class ResponseFamily:
    """Exact cosine-Galerkin boundary response at one resolution."""

    def __init__(
        self,
        resolution: Resolution,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        channels = resolution.galerkin_channels
        if channels < INPUT_CHANNELS:
            raise ValueError("Galerkin channel count is smaller than the drive support")

        basis, weight = spectral.cosine_basis(
            resolution.transverse_quadrature_points,
            channels,
            device=device,
            dtype=dtype,
        )
        y = (
            2.0
            * math.pi
            * torch.arange(
                resolution.transverse_quadrature_points,
                device=device,
                dtype=dtype,
            )
            / resolution.transverse_quadrature_points
        )
        x = (
            torch.arange(
                resolution.longitudinal_layers,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) / resolution.longitudinal_layers
        coefficient = coefficient_field(x, y)
        self.coefficient_min = float(coefficient.min().cpu())
        self.coefficient_max = float(coefficient.max().cpu())
        self.reaction = weight * torch.einsum(
            "yi,ny,yj->nij",
            basis,
            coefficient,
            basis,
        )
        self.laplacian = torch.diag(
            torch.arange(channels, device=device, dtype=dtype).square()
        )
        self.identity = torch.eye(channels, device=device, dtype=dtype)
        self.query_x = (
            (
                torch.arange(
                    resolution.interior_query_points,
                    device=device,
                    dtype=dtype,
                )
                + 0.5
            )
            / resolution.interior_query_points
        )[None, :]

    def response(
        self,
        length: float,
        screening: float,
        *,
        query_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map the eight left-boundary coefficients to all retained modes."""

        profiles = length**2 * (
            self.laplacian
            + screening**2 * self.identity
            + POTENTIAL_SCALE * self.reaction
        )
        kernel = spectral.boundary_kernel(
            profiles[None, ...],
            self.query_x if query_x is None else query_x,
        )
        return kernel[0, ..., :INPUT_CHANNELS]


def column_energies(response: torch.Tensor) -> torch.Tensor:
    """Mean interior energy produced by each unit boundary coefficient."""

    return torch.mean(torch.sum(response.square(), dim=-2), dim=0)


def weighted_squared_norm(response: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Expected squared norm under independent drive variances."""

    return torch.mean(
        torch.sum(response.square() * weights[None, None, :], dim=(-2, -1))
    )


def relative_distance(
    response: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    return math.sqrt(
        float(
            (
                weighted_squared_norm(response - target, weights)
                / weighted_squared_norm(target, weights)
            ).cpu()
        )
    )


def solve_source_parameter(
    family: ResponseFamily,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    parameter: str,
) -> float:
    """Find the one-sided source parameter at the frozen response radius."""

    if parameter == "length":
        lower, upper = TARGET_LENGTH, 2.0

        def response(value: float) -> torch.Tensor:
            return family.response(value, TARGET_SCREENING)

    elif parameter == "screening":
        lower, upper = TARGET_SCREENING, math.pi

        def response(value: float) -> torch.Tensor:
            return family.response(TARGET_LENGTH, value)

    else:
        raise ValueError(f"unknown source parameter {parameter!r}")

    while relative_distance(response(upper), target, weights) < SOURCE_RADIUS:
        upper *= 1.5
    for _ in range(BISECTION_STEPS):
        midpoint = 0.5 * (lower + upper)
        if relative_distance(response(midpoint), target, weights) < SOURCE_RADIUS:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def same_mode_response(response: torch.Tensor) -> torch.Tensor:
    """Retain only the separable same-index input/output response."""

    approximation = torch.zeros_like(response)
    index = torch.arange(INPUT_CHANNELS, device=response.device)
    approximation[:, index, index] = response[:, index, index]
    return approximation


def evaluate_level(
    family: ResponseFamily,
    weights: torch.Tensor,
    *,
    source_length: float,
    source_screening: float,
) -> dict[str, Any]:
    target = family.response(TARGET_LENGTH, TARGET_SCREENING)
    source_p = family.response(source_length, TARGET_SCREENING)
    source_g = family.response(TARGET_LENGTH, source_screening)
    displacement_p = source_p - target
    displacement_g = source_g - target

    inner = torch.mean(
        torch.sum(
            displacement_p
            * displacement_g
            * weights[None, None, :],
            dim=(-2, -1),
        )
    )
    norm_p = weighted_squared_norm(displacement_p, weights)
    norm_g = weighted_squared_norm(displacement_g, weights)
    cosine = float((inner / torch.sqrt(norm_p * norm_g)).clamp(-1.0, 1.0).cpu())
    angle = math.degrees(math.acos(cosine))

    distance_p_by_input = torch.mean(
        torch.sum(displacement_p.square(), dim=-2),
        dim=0,
    )
    distance_g_by_input = torch.mean(
        torch.sum(displacement_g.square(), dim=-2),
        dim=0,
    )
    p_closer = [
        index
        for index in range(INPUT_CHANNELS)
        if distance_p_by_input[index] < distance_g_by_input[index]
    ]
    g_closer = [
        index
        for index in range(INPUT_CHANNELS)
        if distance_g_by_input[index] < distance_p_by_input[index]
    ]

    return {
        "resolution": asdict(family.resolution),
        "coefficient_min": family.coefficient_min,
        "coefficient_max": family.coefficient_max,
        "coefficient_dynamic_range": (
            family.coefficient_max / family.coefficient_min
        ),
        "target_diagonal_demixing_relative_error": relative_distance(
            same_mode_response(target),
            target,
            weights,
        ),
        "source_P_relative_distance": relative_distance(source_p, target, weights),
        "source_G_relative_distance": relative_distance(source_g, target, weights),
        "response_direction_cosine": cosine,
        "response_direction_angle_degrees": angle,
        "source_P_closer_input_modes": p_closer,
        "source_G_closer_input_modes": g_closer,
        "source_P_low_half_distance_fraction": float(
            (
                torch.sum(weights[:4] * distance_p_by_input[:4])
                / torch.sum(weights * distance_p_by_input)
            ).cpu()
        ),
        "source_G_low_half_distance_fraction": float(
            (
                torch.sum(weights[:4] * distance_g_by_input[:4])
                / torch.sum(weights * distance_g_by_input)
            ).cpu()
        ),
    }


def boundary_certification(
    family: ResponseFamily,
    *,
    length: float,
    screening: float,
) -> dict[str, float]:
    query_x = torch.tensor(
        [[0.0, 1.0]],
        device=family.device,
        dtype=family.dtype,
    )
    response = family.response(length, screening, query_x=query_x)
    expected_left = torch.zeros_like(response[0])
    index = torch.arange(INPUT_CHANNELS, device=family.device)
    expected_left[index, index] = 1.0
    return {
        "left_identity_max_abs_error": float(
            torch.max(torch.abs(response[0] - expected_left)).cpu()
        ),
        "right_zero_max_abs_error": float(torch.max(torch.abs(response[1])).cpu()),
    }


def run(*, device: torch.device) -> dict[str, Any]:
    dtype = torch.float64
    families = {
        resolution.name: ResponseFamily(resolution, device=device, dtype=dtype)
        for resolution in RESOLUTIONS
    }
    reference_family = families["reference"]
    reference_target = reference_family.response(TARGET_LENGTH, TARGET_SCREENING)
    weights = 1.0 / column_energies(reference_target)
    weights = weights / weights.mean()
    source_length = solve_source_parameter(
        reference_family,
        reference_target,
        weights,
        parameter="length",
    )
    source_screening = solve_source_parameter(
        reference_family,
        reference_target,
        weights,
        parameter="screening",
    )
    levels = {
        name: evaluate_level(
            family,
            weights,
            source_length=source_length,
            source_screening=source_screening,
        )
        for name, family in families.items()
    }
    reference = levels["reference"]
    medium = levels["medium"]
    reference_pattern = (
        reference["source_P_closer_input_modes"],
        reference["source_G_closer_input_modes"],
    )
    medium_pattern = (
        medium["source_P_closer_input_modes"],
        medium["source_G_closer_input_modes"],
    )
    certification = boundary_certification(
        reference_family,
        length=TARGET_LENGTH,
        screening=TARGET_SCREENING,
    )

    gates = {
        "material_nonseparability": (
            reference["target_diagonal_demixing_relative_error"] >= 0.04
        ),
        "causal_separation": (
            reference["response_direction_angle_degrees"] >= 30.0
        ),
        "spectral_crossover": (
            sum(index < 4 for index in reference["source_P_closer_input_modes"]) >= 3
            and sum(index >= 4 for index in reference["source_G_closer_input_modes"])
            >= 3
        ),
        "nonextreme_drive_distribution": (
            float((weights.max() / weights.min()).cpu()) <= 5.0
        ),
        "source_distance_control": (
            abs(reference["source_P_relative_distance"] - SOURCE_RADIUS) <= 1.0e-7
            and abs(reference["source_G_relative_distance"] - SOURCE_RADIUS)
            <= 1.0e-7
        ),
        "boundary_certification": (
            max(certification.values()) <= 1.0e-9
        ),
        "resolution_stability": (
            abs(
                medium["response_direction_angle_degrees"]
                - reference["response_direction_angle_degrees"]
            )
            <= 2.0
            and abs(
                medium["target_diagonal_demixing_relative_error"]
                - reference["target_diagonal_demixing_relative_error"]
            )
            <= 0.005
            and abs(medium["source_P_relative_distance"] - SOURCE_RADIUS) <= 0.003
            and abs(medium["source_G_relative_distance"] - SOURCE_RADIUS) <= 0.003
            and medium_pattern == reference_pattern
        ),
    }
    return {
        "schema_version": 1,
        "study": "nonseparable_response_relation_v1",
        "preregistration": (
            "results/nonseparable_response_relation_preregistration_2026-07-29.json"
        ),
        "device": str(device),
        "dtype": str(dtype),
        "source_parameters": {
            "source_P_propagation_length": source_length,
            "source_G_screening": source_screening,
            "source_G_screening_over_pi": source_screening / math.pi,
        },
        "drive_variances": [float(value.cpu()) for value in weights],
        "drive_variance_dynamic_range": float(
            (weights.max() / weights.min()).cpu()
        ),
        "boundary_certification": certification,
        "levels": levels,
        "gates": gates,
        "verdict": (
            "advance_to_target_only_saturation_pilot"
            if all(gates.values())
            else "reject_nonseparable_instrument"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(device=torch.device(args.device))
    source_path = Path(__file__)
    result["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "gates": result["gates"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
