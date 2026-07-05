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

"""Verification suite for the Taylor-Hood Navier-Stokes reference solver.

The fast subset (method-of-manufactured-solutions convergence, discrete
balance identities, gauge and compatibility conventions, Newton behavior)
runs in CI; the production-resolution star case is gated behind
``NS_PRODUCTION_TESTS=1`` because a single solve takes about a minute.
"""

import math
import os

import fem_navier_stokes
import numpy as np
import pytest
from fem_navier_stokes import (
    NewtonError,
    manufactured_solution,
    solve_navier_stokes,
)


def test_quadrature_rule_is_exact_to_degree_five():
    """The Dunavant 7-point rule integrates every monomial of degree <= 5.

    Regression guard for the build-time bug this module's history records:
    a wrong pairing of the rule's barycentric constants produced an invalid
    rule that still passed the (symmetry-insensitive) row-sum and
    divergence identity checks while silently breaking Taylor-Hood inf-sup
    coupling at mesh corners.  Exactness over the full monomial basis is
    the check that would have caught it.
    """

    barycentric = fem_navier_stokes._QUAD_BARYCENTRIC
    weights = fem_navier_stokes._QUAD_WEIGHTS
    for i in range(6):
        for j in range(6 - i):
            approximate = float(
                (weights * barycentric[:, 1] ** i * barycentric[:, 2] ** j).sum()
            )
            # int_T l1^i l2^j over the unit reference triangle equals
            # i! j! / (i + j + 2)!; weights sum to one, so compare 2x.
            exact = (
                2.0 * math.factorial(i) * math.factorial(j) / math.factorial(i + j + 2)
            )
            assert abs(approximate - exact) <= 1.0e-14, (i, j)


_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _interior_queries(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.15, 0.85, size=(n, 2))


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _aligned_pressure_error(computed: np.ndarray, exact: np.ndarray) -> float:
    """Gauge-invariant pressure comparison: align means over the query set."""

    computed = computed - computed.mean()
    exact = exact - exact.mean()
    return _relative_l2(computed, exact)


def test_mms_velocity_converges_at_third_order():
    """Manufactured-solution velocity error decreases at (close to) O(h^3).

    The domain is the exactly represented unit square, the manufactured
    fields are globally smooth, and the forcing is analytic, so the P2
    velocity must converge at order ~3 and the P1 pressure at order ~2.
    """

    nu = 0.05
    mms = manufactured_solution(nu)
    queries = _interior_queries()
    exact_velocity = mms["velocity"](queries)
    exact_pressure = mms["pressure"](queries)

    h_values = np.array([0.16, 0.08, 0.04])
    velocity_errors, pressure_errors = [], []
    for h in h_values:
        solution = solve_navier_stokes(
            [_SQUARE],
            mms["velocity"],
            queries,
            viscosity=nu,
            target_h=h,
            forcing=mms["forcing"],
        )
        velocity_errors.append(_relative_l2(solution.velocity_query, exact_velocity))
        pressure_errors.append(
            _aligned_pressure_error(solution.pressure_query, exact_pressure)
        )

    velocity_errors = np.array(velocity_errors)
    pressure_errors = np.array(pressure_errors)
    assert np.all(np.diff(velocity_errors) < 0.0), velocity_errors
    assert np.all(np.diff(pressure_errors) < 0.0), pressure_errors
    velocity_order = np.polyfit(np.log(h_values), np.log(velocity_errors), 1)[0]
    pressure_order = np.polyfit(np.log(h_values), np.log(pressure_errors), 1)[0]
    assert velocity_order >= 2.5, (velocity_order, velocity_errors)
    assert pressure_order >= 1.5, (pressure_order, pressure_errors)
    assert velocity_errors[-1] <= 5.0e-5, velocity_errors


def test_manufactured_forcing_matches_finite_differences():
    """The analytic MMS forcing equals -nu lap u + (u.grad)u + grad p.

    Central finite differences of the manufactured velocity and pressure
    reproduce the analytic forcing to the FD truncation error -- this
    certifies the verification instrument itself (a wrong hand-derived
    forcing would silently weaken every MMS test).
    """

    nu = 0.037
    mms = manufactured_solution(nu)
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.4, 0.9, size=(50, 2))
    step = 1.0e-5

    def shift(dx: float, dy: float) -> np.ndarray:
        return points + np.array([dx, dy])

    u0 = mms["velocity"](points)
    convection = np.zeros_like(u0)
    laplacian = np.zeros_like(u0)
    gradient_p = np.zeros_like(u0)
    for axis, (dx, dy) in enumerate(((step, 0.0), (0.0, step))):
        plus, minus = mms["velocity"](shift(dx, dy)), mms["velocity"](shift(-dx, -dy))
        convection += u0[:, axis : axis + 1] * (plus - minus) / (2.0 * step)
        laplacian += (plus - 2.0 * u0 + minus) / step**2
        gradient_p[:, axis] = (
            mms["pressure"](shift(dx, dy)) - mms["pressure"](shift(-dx, -dy))
        ) / (2.0 * step)
    reconstructed = -nu * laplacian + convection + gradient_p
    error = np.abs(reconstructed - mms["forcing"](points)).max()
    assert error <= 1.0e-5, error


def test_discrete_balances_and_gauge():
    """Momentum balance identity, pressure gauge, and multiplier convention.

    The assembled momentum residual summed over all nodes must equal the
    independently quadratured convective momentum integral minus the
    forcing integral (roundoff-level assembly certificate); the discrete
    mean pressure is zero (the Lagrange gauge); and the multiplier equals
    minus the discrete boundary flux divided by the domain area.
    """

    nu = 0.02
    mms = manufactured_solution(nu)
    solution = solve_navier_stokes(
        [_SQUARE],
        mms["velocity"],
        _interior_queries(50),
        viscosity=nu,
        target_h=0.08,
        forcing=mms["forcing"],
    )
    diagnostics = solution.diagnostics
    assert diagnostics.momentum_balance_error <= 1.0e-11
    assert abs(diagnostics.gauge_mean_pressure) <= 1.0e-12
    assert diagnostics.relative_residual <= 1.0e-10
    # lambda * |Omega| == -boundary flux (the multiplier absorbs the
    # discrete compatibility defect); the unit square has area one.
    assert math.isclose(
        diagnostics.lagrange_multiplier,
        -diagnostics.boundary_flux,
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    )
    # Newton must actually have converged quadratically from Stokes.
    assert diagnostics.newton_iterations <= 6
    assert diagnostics.continuation_solves == 0


def test_tangential_drive_has_tiny_discrete_flux():
    """A tangential-only drive is exactly compatible at the continuous level.

    On the polygonal domain the interpolated trace leaves a small discrete
    flux (reported, absorbed by the multiplier); with a rotating-lid drive
    on a 24-gon disk it must sit orders below the unit drive scale.
    """

    n = 24
    angles = 2.0 * math.pi * np.arange(n) / n
    loop = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tangents = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    queries = 0.5 * _interior_queries(50, seed=1) - 0.25
    solution = solve_navier_stokes(
        [loop],
        [tangents],  # unit tangential drive at every vertex
        queries,
        viscosity=0.05,
        target_h=0.1,
    )
    assert abs(solution.diagnostics.boundary_flux) <= 1.0e-6
    assert solution.diagnostics.momentum_balance_error <= 1.0e-11
    # The rotating-lid disk flow keeps the peak speed at the boundary.
    assert solution.diagnostics.velocity_speed_max <= 1.05


def test_newton_error_reported_not_masked():
    """An unreachable tolerance surfaces as NewtonError (no silent labels)."""

    nu = 0.05
    mms = manufactured_solution(nu)
    with pytest.raises(NewtonError):
        solve_navier_stokes(
            [_SQUARE],
            mms["velocity"],
            _interior_queries(10),
            viscosity=nu,
            target_h=0.2,
            forcing=mms["forcing"],
            newton_tolerance=1.0e-18,  # below float64 roundoff: must fail
            max_newton_iterations=4,
            continuation=False,
        )


@pytest.mark.skipif(
    not os.environ.get("NS_PRODUCTION_TESTS"),
    reason="production-resolution N-S solve takes minutes; "
    "set NS_PRODUCTION_TESTS=1 to run",
)
def test_mms_star_production_resolution():
    """MMS on a production star mesh at the catalog's production settings.

    The dataset generator's production resolution is ``target_h = 0.0105``
    on a 1024-gon star; the manufactured velocity must be reproduced to
    rel-L2 <= 1e-6 there (the suite's MMS bar, met at production
    resolution; the convergence-order test above pins the order).
    """

    import generate_datasets as gd
    import torch
    from conformal_laplace import sample_geometry

    nu = 0.02
    mms = manufactured_solution(nu)
    geometry = sample_geometry(
        7, modes=(2, 3), deformation_range=(0.05, 0.35), dtype=torch.float64
    )
    angles = 2.0 * math.pi * torch.arange(1024, dtype=torch.float64) / 1024
    loop = gd._star_boundary(geometry, angles)
    queries, _ = gd._sample_bucketed_queries(
        11,
        loop,
        n_interior=128,
        n_near=32,
        interior_margin=0.12,
        near_band=(0.02, 0.08),
    )
    solution = solve_navier_stokes(
        [loop],
        mms["velocity"],
        queries,
        viscosity=nu,
        target_h=0.0105,
        forcing=mms["forcing"],
    )
    error = _relative_l2(solution.velocity_query, mms["velocity"](queries))
    assert error <= 1.0e-6, error
