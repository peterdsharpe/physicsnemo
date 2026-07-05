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

"""Contracts for the nonlinear Liouville benchmark generator and driver."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from liouville import (  # noqa: E402
    LIOUVILLE_CONSTANT,
    SPLITS,
    LiouvilleField,
    build_liouville_sample,
    field_derivative,
    liouville_pde_residual,
    liouville_solution,
    run_experiment,
    sample_liouville_field,
)


def _fields_covering_both_families(
    *, domain_radius_bound: float = 1.5, per_family: int = 3
) -> list[LiouvilleField]:
    """Deterministically collect sampled fields from both Moebius families."""

    collected: dict[str, list[LiouvilleField]] = {
        "moebius_affine": [],
        "moebius_exp": [],
    }
    for seed in range(200):
        field = sample_liouville_field(seed, domain_radius_bound=domain_radius_bound)
        bucket = collected[field.family]
        if len(bucket) < per_family:
            bucket.append(field)
        if all(len(bucket) == per_family for bucket in collected.values()):
            return [field for bucket in collected.values() for field in bucket]
    raise AssertionError("family sampling failed to produce both families")


def _interior_points(seed: int, n: int) -> torch.Tensor:
    """Random float64 points in the unit disk (inside every certified domain)."""

    generator = torch.Generator().manual_seed(seed)
    radius = 0.95 * torch.sqrt(torch.rand(n, generator=generator, dtype=torch.float64))
    angle = 2.0 * math.pi * torch.rand(n, generator=generator, dtype=torch.float64)
    return torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)


def test_liouville_constant_convention_is_two() -> None:
    """Autograd selects c = 2 in lap u + c exp(u) = 0, machine zero in fp64.

    This is the verification demanded before trusting the label formula: for
    u = log(4 |f'|^2 / (1 + |f|^2)^2) only c = 2 annihilates the residual;
    the other candidate conventions leave O(1) relative residuals.
    """

    for field in _fields_covering_both_families():
        points = _interior_points(11, 24)
        laplacian = liouville_pde_residual(field, points, constant=0.0).detach()
        w = torch.complex(points[:, 0], points[:, 1])
        exp_u = torch.exp(liouville_solution(field, w))
        scale = float(exp_u.abs().max())
        residuals = {
            candidate: float((laplacian + candidate * exp_u).abs().max()) / scale
            for candidate in (1.0, 2.0, 4.0, 8.0)
        }
        assert residuals[2.0] < 1.0e-9
        for candidate in (1.0, 4.0, 8.0):
            assert residuals[candidate] > 0.1
    assert LIOUVILLE_CONSTANT == 2.0


def test_exact_labels_satisfy_the_pde_on_generated_samples() -> None:
    """The sample's own targets obey lap u + 2 exp(u) = 0 to machine precision."""

    for seed in (0, 1, 5):
        sample = build_liouville_sample(seed, dtype=torch.float64)
        residual = liouville_pde_residual(sample.field, sample.domain.interior.points)
        scale = torch.exp(sample.target).abs().max()
        assert float(residual.detach().abs().max() / scale) < 1.0e-9


def test_derivative_nonvanishing_guard() -> None:
    """Construction rejects degenerate maps; sampled maps keep |f'| > 0."""

    reference = sample_liouville_field(7, domain_radius_bound=1.5)

    def rebuild(**overrides: object) -> LiouvilleField:
        state = {
            "family": reference.family,
            "linear_coefficient": reference.linear_coefficient,
            "offset": reference.offset,
            "pole": reference.pole,
            "residue": reference.residue,
            "constant": reference.constant,
            "domain_radius_bound": reference.domain_radius_bound,
        }
        state.update(overrides)
        return LiouvilleField(**state)

    zero = torch.zeros((), dtype=torch.complex128)
    with pytest.raises(ValueError, match="linear_coefficient"):
        rebuild(linear_coefficient=zero)
    with pytest.raises(ValueError, match="residue"):
        rebuild(residue=zero)
    with pytest.raises(ValueError, match="pole"):
        rebuild(pole=torch.complex(torch.tensor(1.0), torch.tensor(0.5)).to(zero.dtype))

    for field in _fields_covering_both_families():
        w = torch.complex(*_interior_points(23, 512).unbind(-1)) * 1.5
        magnitude = field_derivative(field, w).abs()
        assert float(magnitude.min()) > 0.0
        assert torch.isfinite(magnitude).all()


def test_generator_determinism() -> None:
    """The same seed reproduces every tensor of the sample bit-for-bit."""

    first = build_liouville_sample(321)
    second = build_liouville_sample(321)
    assert first.field.family == second.field.family
    assert torch.equal(first.target, second.target)
    assert torch.equal(first.domain.interior.points, second.domain.interior.points)
    assert torch.equal(
        first.domain.boundaries["dirichlet"].points,
        second.domain.boundaries["dirichlet"].points,
    )
    assert torch.equal(
        first.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        second.domain.boundaries["dirichlet"].cell_data["boundary_value"],
    )


def test_sample_layout_and_amplitude_contract() -> None:
    """Samples use the linear benchmark's DomainMesh layout with |u| <= 4."""

    sample = build_liouville_sample(9)
    assert set(sample.domain.boundaries.keys()) == {"dirichlet"}
    boundary = sample.domain.boundaries["dirichlet"]
    assert "boundary_value" in boundary.cell_data
    length = sample.domain.global_data["reference_length"]
    assert float(length) == 1.0
    assert float(sample.target.abs().max()) <= 4.0
    assert float(boundary.cell_data["boundary_value"].abs().max()) <= 4.0
    # The point of the benchmark: the target is *not* harmonic.  Its exact
    # laplacian is -2 exp(u) <= -2 exp(-4), bounded away from zero.
    fp64 = build_liouville_sample(9, dtype=torch.float64)
    laplacian = liouville_pde_residual(
        fp64.field, fp64.domain.interior.points, constant=0.0
    )
    assert float(laplacian.detach().abs().min()) > 2.0 * math.exp(-4.0) * 0.9


def test_driver_smoke_produces_finite_json(tmp_path: Path) -> None:
    """A 50-step CPU run writes a finite report; the trained path is exercised."""

    report = run_experiment(
        model_name="harmonic_panel_bie",
        steps=50,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=2,
    )
    on_disk = json.loads((tmp_path / "harmonic_panel_bie_seed3.json").read_text())
    for payload in (report, on_disk):
        assert payload["parameters"] == 3
        for value in payload["splits"].values():
            assert math.isfinite(value)
        assert math.isfinite(payload["pde_residual"])
        assert math.isfinite(payload["best_validation_relative_l2"])
        # Operator-fidelity block: per-split strong-form residual, and an
        # explicit n/a marker for the (nonlinear-PDE) maximum principle.
        fidelity = payload["fidelity"]
        assert set(fidelity["pde_residual"]) == set(SPLITS)
        for value in fidelity["pde_residual"].values():
            assert math.isfinite(value)
        assert fidelity["max_principle_violation"] is None


def test_boundary_mean_pde_residual_calibration(tmp_path: Path) -> None:
    """The constant baseline scores exactly 2: the harmonic-limit calibration.

    ``||lap u + 2 e^u|| / ||e^u||`` equals 2 for any prediction with zero
    laplacian, so the parameter-free constant baseline pins the diagnostic's
    scale (and exercises the no-trainable-parameter driver path).
    """

    report = run_experiment(
        model_name="boundary_mean",
        steps=0,
        seed=5,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=2,
    )
    assert report["parameters"] == 0
    assert abs(report["pde_residual"] - 2.0) < 1.0e-9
    # The same calibration holds on every split of the fidelity block.
    assert all(
        abs(value - 2.0) < 1.0e-9
        for value in report["fidelity"]["pde_residual"].values()
    )
