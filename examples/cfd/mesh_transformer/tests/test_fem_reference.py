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

"""Contracts for the FEM reference solver and the dataset catalog.

The manufactured-agreement test runs at the dataset generator's production
settings (``target_h = 0.02``, 2048 boundary vertices) so the accuracy
number quoted for cataloged datasets is exercised directly; the remaining
solver tests use coarser-but-passing settings to keep the suite fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import dataset_catalog  # noqa: E402
import generate_datasets  # noqa: E402
from fem_reference import (  # noqa: E402
    k0_charge_potential,
    log_charge_potential,
    solve_dirichlet,
)

PRODUCTION_TARGET_H = 0.02
PRODUCTION_N_BOUNDARY = 2048

_CHARGE_CENTERS = 2.5 * np.stack(
    (np.cos([0.3, 2.2, 4.4]), np.sin([0.3, 2.2, 4.4])), axis=1
)
_CHARGE_MAGNITUDES = np.array([1.0, -0.7, 0.4])


def _star_loop(
    n: int, *, amplitude: float = 0.3, lobes: int = 5, radius: float = 1.0
) -> np.ndarray:
    """Star-deformed circle ``r(theta) = radius * (1 + amplitude cos(k theta))``."""

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + amplitude * np.cos(lobes * theta))
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)


def _star_interior_queries(
    n: int, *, amplitude: float = 0.3, lobes: int = 5, seed: int = 0
) -> np.ndarray:
    """Area-uniform-ish points strictly inside the star (star-shaped wrt 0)."""

    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    fraction = 0.95 * np.sqrt(rng.uniform(0.0, 1.0, n))
    r = fraction * (1.0 + amplitude * np.cos(lobes * phi))
    return np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


@pytest.mark.parametrize("equation", ["laplace", "screened"])
def test_manufactured_charge_agreement_at_production_settings(equation):
    """Exterior-charge exact solutions reproduced to rel-L2 <= 1e-4.

    Charge-induced Dirichlet traces on a strongly deformed star: the exact
    field is a sum of log (Laplace) or Bessel-K0 (screened) sources placed
    outside the domain, so the trace is far outside the bank's polynomial
    manufactured family.  Production generator settings.
    """

    kappa = 1.3 if equation == "screened" else 0.0
    loop = _star_loop(PRODUCTION_N_BOUNDARY)
    queries = _star_interior_queries(256)

    if equation == "laplace":

        def exact(points):
            return log_charge_potential(points, _CHARGE_CENTERS, _CHARGE_MAGNITUDES)
    else:

        def exact(points):
            return k0_charge_potential(
                points, _CHARGE_CENTERS, _CHARGE_MAGNITUDES, kappa
            )

    solution = solve_dirichlet(
        [loop],
        exact,
        queries,
        equation=equation,
        kappa=kappa,
        target_h=PRODUCTION_TARGET_H,
    )
    error = _relative_l2(solution.u_query, exact(queries))
    assert error <= 1.0e-4, f"{equation}: rel-L2 {error:.3e} exceeds 1e-4"
    assert solution.diagnostics.linear_residual < 1.0e-8


def test_mesh_convergence_is_cubic_in_l2():
    """P2 interior L2 error decreases at (close to) O(h^3) on a smooth case.

    The domain is a fixed 24-gon star polygon represented exactly, so the
    trace is exact at every boundary node and no geometric error pollutes
    the rate; the exact solution (exterior log charges) is analytic on the
    closure.  The observed order over a fourfold h refinement must be at
    least 2.5.
    """

    loop = _star_loop(24, amplitude=0.15, lobes=3)
    queries = _star_interior_queries(256, amplitude=0.15, lobes=3, seed=1)

    def exact(points):
        return log_charge_potential(points, _CHARGE_CENTERS, _CHARGE_MAGNITUDES)

    target = exact(queries)
    h_values = np.array([0.2, 0.1, 0.05])
    errors = []
    for h in h_values:
        solution = solve_dirichlet([loop], exact, queries, target_h=h)
        errors.append(_relative_l2(solution.u_query, target))
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0), f"errors not decreasing: {errors}"
    observed_order = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
    assert observed_order >= 2.5, (
        f"observed L2 order {observed_order:.2f} < 2.5 (errors {errors})"
    )


def test_annulus_hole_exact_log_solution():
    """Outer circle + inner hole reproduces u = a + b log r to rel-L2 <= 1e-4.

    The log term is harmonic only on the multiply-connected domain (it has
    a source inside the hole), so agreement requires the hole to be meshed
    and both boundaries to carry their own Dirichlet data.
    """

    a, b = 0.5, 1.0
    outer = _star_loop(256, amplitude=0.0, lobes=1, radius=1.0)
    inner = _star_loop(128, amplitude=0.0, lobes=1, radius=0.4)

    def exact(points):
        return a + b * np.log(np.linalg.norm(points, axis=1))

    rng = np.random.default_rng(3)
    phi = rng.uniform(0.0, 2.0 * np.pi, 256)
    r = rng.uniform(0.55, 0.85, 256)
    queries = np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1)

    solution = solve_dirichlet([outer, inner], exact, queries, target_h=0.04)
    error = _relative_l2(solution.u_query, exact(queries))
    assert error <= 1.0e-4, f"annulus rel-L2 {error:.3e} exceeds 1e-4"
    # The solve must genuinely be multiply-connected: with the hole absent
    # the same trace data cannot produce the log profile between the rings.
    assert solution.diagnostics.n_dirichlet_nodes > 2 * (256 + 128) - 8


def test_maximum_principle_interior_within_trace_range():
    """FEM interior values stay within [min, max] of the boundary trace."""

    loop = _star_loop(512)
    queries = _star_interior_queries(512, seed=2)

    def trace(points):
        angle = np.arctan2(points[:, 1], points[:, 0])
        return np.sin(3.0 * angle) - 0.25 * np.cos(angle)

    solution = solve_dirichlet([loop], trace, queries, target_h=0.1)
    lo = solution.diagnostics.trace_min
    hi = solution.diagnostics.trace_max
    tolerance = 1.0e-3 * (hi - lo)
    assert solution.u_query.min() >= lo - tolerance
    assert solution.u_query.max() <= hi + tolerance


def test_per_vertex_trace_matches_callable_trace():
    """Per-vertex trace input agrees with the equivalent callable trace."""

    loop = _star_loop(512, amplitude=0.2, lobes=4)
    queries = _star_interior_queries(64, amplitude=0.2, lobes=4, seed=4)

    def trace(points):
        return log_charge_potential(points, _CHARGE_CENTERS, _CHARGE_MAGNITUDES)

    from_callable = solve_dirichlet([loop], trace, queries, target_h=0.1)
    from_vertices = solve_dirichlet([loop], [trace(loop)], queries, target_h=0.1)
    # The vertex path linearizes the trace along each (short) input segment;
    # the two solutions agree up to that interpolation error.
    assert _relative_l2(from_vertices.u_query, from_callable.u_query) < 1.0e-4


def _tiny_settings() -> generate_datasets.GeneratorSettings:
    return generate_datasets.GeneratorSettings(
        n_boundary=16,
        n_query=16,
        n_fem_boundary=128,
        target_h=0.2,
        base_seed=11,
    )


def test_catalog_round_trip_and_manifest_integrity(tmp_path):
    """Generate 3 tiny cases, reload bit-identically, validate the manifest."""

    directory = generate_datasets.generate_dataset(
        family="star_random_trace",
        n_cases=3,
        version="v-test",
        workers=1,
        settings=_tiny_settings(),
        root=tmp_path,
        created="2026-07-03",
    )
    assert directory == tmp_path / "star_random_trace" / "v-test"

    summary = dataset_catalog.validate_catalog(directory)
    assert summary["n_cases"] == 3

    manifest = dataset_catalog.load_manifest(directory)
    assert manifest["created"] == "2026-07-03"
    assert manifest["family"] == "star_random_trace"
    for split, spec in manifest["splits"].items():
        assert 0 <= spec["start"] < spec["stop"] <= 3, split

    # Bit-identical round trip: regenerating a case in-process must produce
    # exactly the arrays reloaded from disk (dtype and bit pattern).
    for index in range(3):
        case = dataset_catalog.load_case(directory, index)
        job = (index, case.params["split"], _tiny_settings())
        regenerated = generate_datasets._generate_case(job)
        for name, value in regenerated["arrays"].items():
            stored = case.arrays[name]
            assert stored.dtype == np.asarray(value).dtype, name
            assert np.array_equal(stored, value), f"{name} not bit-identical"

    # Verification stats recorded per split, with the maximum principle held.
    for split, stats in manifest["verification"].items():
        if not isinstance(stats, dict) or "max_principle_violation_max" not in stats:
            continue
        assert stats["max_principle_violation_max"] <= 1.0e-2

    # Tampering must be detected.
    victim = directory / dataset_catalog.case_filename(1)
    payload = bytearray(victim.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    victim.write_bytes(bytes(payload))
    with pytest.raises(dataset_catalog.CatalogError, match="checksum"):
        dataset_catalog.validate_catalog(directory)


def test_loader_reconstructs_benchmark_domain_mesh(tmp_path):
    """Cataloged cases rebuild the DomainMesh interface used by the bank."""

    import torch

    from physicsnemo.mesh import DomainMesh

    directory = generate_datasets.generate_dataset(
        family="star_random_trace",
        n_cases=3,
        version="v-loader",
        workers=1,
        settings=_tiny_settings(),
        root=tmp_path,
        created="2026-07-03",
    )
    cases = list(dataset_catalog.iter_split(directory, "train"))
    assert len(cases) == 3

    for case in cases:
        domain, target = dataset_catalog.load_domain_sample(case)
        assert isinstance(domain, DomainMesh)
        boundary = domain.boundaries["dirichlet"]
        assert boundary.n_cells == case.arrays["boundary_points"].shape[0]
        assert boundary.cell_data["boundary_value"].shape == (boundary.n_cells,)
        assert torch.allclose(
            target,
            torch.from_numpy(case.arrays["u_query"]).to(target.dtype),
        )
        assert domain.interior.point_data["potential"] is target
        assert float(domain.global_data["reference_length"]) == 1.0

        # Orientation contract of the stored discretization: the divergence
        # theorem on the closed polygon gives sum(measure * centroid . n)
        # equal to twice the enclosed area, so it must be clearly positive
        # when normals point outward.
        centroids = case.arrays["boundary_cell_centroids"]
        normals = case.arrays["boundary_cell_normals"]
        measures = case.arrays["boundary_cell_measures"]
        twice_area = float(np.sum(measures * np.sum(centroids * normals, axis=1)))
        assert twice_area > 1.0

        # The interior target obeys the maximum principle for its trace.
        values = case.arrays["boundary_value"]
        margin = 1.0e-2 * (values.max() - values.min())
        assert case.arrays["u_query"].max() <= values.max() + margin + 0.05
        assert case.arrays["u_query"].min() >= values.min() - margin - 0.05
