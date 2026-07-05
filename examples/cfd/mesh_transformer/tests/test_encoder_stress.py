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

"""Contracts for the encoder-stress Laplace families (multi-body, deep cavity)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from encoder_stress import (  # noqa: E402
    SPLITS,
    CavityGeometry,
    build_deep_cavity_sample,
    build_multi_body_sample,
    cavity_boundary_loop,
    harmonic_gradient,
    harmonic_potential,
    mesh_cell_winding,
    polyline_is_simple,
    run_experiment,
    winding_number,
)

_BUILDERS = {
    "multi_body": build_multi_body_sample,
    "deep_cavity": build_deep_cavity_sample,
}


def _every_split(seeds=(0, 1)):
    for family, builder in _BUILDERS.items():
        for split_name, spec in SPLITS[family].items():
            for seed in seeds:
                yield (
                    family,
                    split_name,
                    builder(seed * 7919 + 11, dtype=torch.float64, **spec),
                )


def _finite_difference_laplacian(
    points: torch.Tensor,
    positions: torch.Tensor,
    strengths: torch.Tensor,
    step: float,
) -> torch.Tensor:
    laplacian = -4.0 * harmonic_potential(points, positions, strengths)
    for offset in ([step, 0.0], [-step, 0.0], [0.0, step], [0.0, -step]):
        shifted = points + torch.tensor(offset, dtype=torch.float64)
        laplacian = laplacian + harmonic_potential(shifted, positions, strengths)
    return laplacian / step**2


def test_manufactured_solutions_are_harmonic() -> None:
    """FD Laplacian at the interior queries vanishes relative to the field scale.

    The scale is the sum of per-charge Hessian magnitudes ``|q_i| / d_i^2``
    (an upper bound on any single Laplacian contribution), so the check is
    meaningful even at gradient stagnation points; the gradient-relative
    ratio is checked as well wherever the gradient is not degenerate.
    """

    step = 1.0e-4
    for family, split_name, sample in _every_split():
        queries = sample.domain.interior.points
        positions = sample.charge_positions
        strengths = sample.charge_strengths
        laplacian = _finite_difference_laplacian(queries, positions, strengths, step)
        distances = (queries[:, None, :] - positions[None, :, :]).norm(dim=-1)
        hessian_scale = (strengths.abs()[None, :] / distances.square()).sum(dim=-1)
        assert float((laplacian.abs() / hessian_scale).max()) < 1.0e-4, (
            family,
            split_name,
        )
        gradient_norm = harmonic_gradient(queries, positions, strengths).norm(dim=-1)
        length = float(sample.domain.global_data["reference_length"])
        healthy = gradient_norm > 1.0e-2
        assert bool(healthy.any())
        ratio = laplacian[healthy].abs() * length / gradient_norm[healthy]
        assert float(ratio.max()) < 1.0e-3, (family, split_name)


def test_queries_and_charges_are_on_the_correct_sides() -> None:
    """Winding of the merged cell-oriented boundary: -1 for queries, 0 for charges.

    The benchmark's clockwise-cell (outward-normal) convention makes every
    closed boundary wind ``-1`` around interior points; charges must be
    outside the domain closure, hence total winding zero.
    """

    for family, split_name, sample in _every_split():
        boundary = sample.domain.boundaries["dirichlet"]
        query_winding = mesh_cell_winding(boundary, sample.domain.interior.points)
        assert bool((query_winding == -1).all()), (family, split_name)
        charge_winding = mesh_cell_winding(boundary, sample.charge_positions)
        assert bool((charge_winding == 0).all()), (family, split_name)


def test_multi_body_charge_placement_and_disjointness() -> None:
    """Two-body samples keep charges in-disk/exterior and boundaries disjoint."""

    for seed in (3, 4, 5):
        sample = build_multi_body_sample(
            seed, dtype=torch.float64, **SPLITS["multi_body"]["narrow_gap"]
        )
        outer, disk_a, disk_b = sample.boundary_loops
        geometry = sample.geometry
        # Disks strictly inside the outer circle and disjoint from each other.
        assert bool((winding_number(outer, disk_a) == 1).all())
        assert bool((winding_number(outer, disk_b) == 1).all())
        assert bool((winding_number(disk_a, disk_b) == 0).all())
        assert bool((winding_number(disk_b, disk_a) == 0).all())
        centers_gap = float(
            (geometry.disk_centers[0] - geometry.disk_centers[1]).norm()
        )
        assert centers_gap > float(geometry.disk_radii.sum())
        # Each charge is inside exactly one disk or outside the outer circle.
        in_a = winding_number(disk_a, sample.charge_positions) == 1
        in_b = winding_number(disk_b, sample.charge_positions) == 1
        in_outer = winding_number(outer, sample.charge_positions) == 1
        exterior = ~in_outer
        assert bool((in_a.int() + in_b.int() + exterior.int() == 1).all())
        # The coupling story requires charges in both disks and one exterior.
        assert bool(in_a.any()) and bool(in_b.any()) and bool(exterior.any())


def test_cavity_curve_is_simple_across_the_difficulty_range() -> None:
    """Non-self-intersection at the corners of the difficulty box and beyond."""

    corner_cases = [
        (0.35, 0.16),
        (0.55, 0.10),
        (0.65, 0.09),
        (0.80, 0.055),
        (0.85, 0.05),  # slightly beyond the deep_slot split
    ]
    for depth_fraction, width_fraction in corner_cases:
        for radius, angle, phase in (
            (1.0, 0.3, 0.0),
            (0.8, 2.1, 0.37),
            (1.4, 4.4, 0.81),
        ):
            geometry = CavityGeometry(
                center=torch.zeros(2, dtype=torch.float64),
                radius=radius,
                slot_depth=depth_fraction * radius,
                slot_half_width=width_fraction * radius,
                fillet_radius=0.6 * width_fraction * radius,
                slot_angle=angle,
            )
            vertices, midpoints, dense = cavity_boundary_loop(
                geometry, n_panels=224, phase=phase
            )
            assert polyline_is_simple(vertices), (depth_fraction, width_fraction)
            assert polyline_is_simple(dense[::16]), (depth_fraction, width_fraction)
            # The curve visits the slot bottom: minimum distance from the
            # center dips to (R - depth) within discretization slack.
            radial = (vertices - geometry.center).norm(dim=-1)
            bottom = radius - geometry.slot_depth
            assert float(radial.min()) < bottom + 0.1 * radius
            assert float(radial.max()) <= radius + 1.0e-9


def test_cavity_slot_charges_are_nestled_inside_the_cavity() -> None:
    """Some charges sit inside the circumscribing circle yet outside the domain."""

    for split_name in ("in_distribution", "deep_slot"):
        for seed in (2, 6):
            sample = build_deep_cavity_sample(
                seed, dtype=torch.float64, **SPLITS["deep_cavity"][split_name]
            )
            geometry = sample.geometry
            in_circle = (sample.charge_positions - geometry.center).norm(
                dim=-1
            ) < geometry.radius
            outside_domain = (
                winding_number(sample.boundary_loops[0], sample.charge_positions) == 0
            )
            assert bool(outside_domain.all())
            assert int(in_circle.sum()) >= 2, (split_name, seed)
    convex = build_deep_cavity_sample(
        2, dtype=torch.float64, **SPLITS["deep_cavity"]["convex_control"]
    )
    radial = (convex.charge_positions - convex.geometry.center).norm(dim=-1)
    assert float(radial.min()) > convex.geometry.radius


def test_boundary_values_match_the_exact_trace_and_cell_order() -> None:
    """cell_data equals u at the stored curve midpoints, aligned with cells."""

    for family, split_name, sample in _every_split(seeds=(0,)):
        boundary = sample.domain.boundaries["dirichlet"]
        recomputed = harmonic_potential(
            sample.boundary_midpoints, sample.charge_positions, sample.charge_strengths
        )
        torch.testing.assert_close(
            boundary.cell_data["boundary_value"], recomputed, rtol=0.0, atol=1.0e-10
        )
        # Chord centroids track the curve midpoints (catches ordering bugs).
        length = float(sample.domain.global_data["reference_length"])
        drift = (boundary.cell_centroids - sample.boundary_midpoints).norm(dim=-1)
        assert float(drift.max()) < 0.05 * length, (family, split_name)


def test_sample_layout_contract() -> None:
    """Single merged dirichlet boundary, expected cell counts, unit-RMS trace."""

    expected_cells = {
        ("multi_body", "in_distribution"): 192,
        ("multi_body", "single_body"): 144,
        ("deep_cavity", "in_distribution"): 224,
    }
    for (family, split_name), n_cells in expected_cells.items():
        sample = _BUILDERS[family](13, **SPLITS[family][split_name])
        assert sample.domain.boundaries.keys() == {"dirichlet"}
        boundary = sample.domain.boundaries["dirichlet"]
        assert boundary.n_cells == n_cells
        assert boundary.points.dtype == torch.float32
        assert sample.domain.interior.n_points == 256
        assert float(sample.domain.global_data["reference_length"]) > 0.0
        values = boundary.cell_data["boundary_value"]
        assert abs(float(values.double().square().mean().sqrt()) - 1.0) < 1.0e-3
        assert float(values.abs().max()) <= 8.0 + 1.0e-4
        assert torch.isfinite(sample.target).all()


def test_generator_determinism() -> None:
    """The same seed reproduces every tensor of the sample bit-for-bit."""

    for builder in _BUILDERS.values():
        first = builder(321)
        second = builder(321)
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
        assert torch.equal(first.charge_positions, second.charge_positions)


def test_polyline_is_simple_detects_crossings() -> None:
    """The certification primitive rejects a figure-eight polygon."""

    square = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    assert polyline_is_simple(square)
    bowtie = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    assert not polyline_is_simple(bowtie)


def test_model_arms_consume_the_merged_boundary() -> None:
    """Registry arms run forward on both families and return finite fields."""

    from encoder_stress import _build_model

    torch.manual_seed(11)
    samples = [build_multi_body_sample(5), build_deep_cavity_sample(5)]
    for name in ("boundary_mean", "pair_kernel", "mesh_transformer_kernel_singonly"):
        model = _build_model(name)
        for sample in samples:
            with torch.no_grad():
                prediction = model(sample.domain).point_data["potential"]
            assert prediction.shape == (256,)
            assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("family", ("multi_body", "deep_cavity"))
def test_driver_smoke_produces_finite_json(tmp_path: Path, family: str) -> None:
    """A zero-step CPU run writes a finite report with the benchmark shape."""

    report = run_experiment(
        model_name="boundary_mean",
        family=family,
        steps=0,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    on_disk = json.loads((tmp_path / f"{family}_boundary_mean_seed3.json").read_text())
    for payload in (report, on_disk):
        assert payload["parameters"] == 0
        assert payload["family"] == family
        assert set(payload["splits"]) == set(SPLITS[family])
        for value in payload["splits"].values():
            assert math.isfinite(value)
        assert math.isfinite(payload["best_validation_relative_l2"])
        # BoundaryMean is constant, hence exactly harmonic.
        assert abs(payload["pde_residual"]) < 1.0e-9
        # Operator-fidelity block: per-split residual (constant -> ~0) and
        # sampled maximum-principle violation (the boundary mean lies inside
        # the sampled trace range, so the violation is exactly zero).
        fidelity = payload["fidelity"]
        assert set(fidelity["pde_residual"]) == set(SPLITS[family])
        assert all(abs(v) < 1.0e-9 for v in fidelity["pde_residual"].values())
        assert set(fidelity["max_principle_violation"]) == set(SPLITS[family])
        assert all(
            v == 0.0 for v in fidelity["max_principle_violation"].values()
        )
