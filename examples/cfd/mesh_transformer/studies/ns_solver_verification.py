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

r"""Verification study for the Navier-Stokes reference solver (iteration 36).

Produces the dated artifact behind the ns_cavity_star suite's solver
credential:

1. **MMS convergence** on the exactly represented unit square: velocity
   order ~3 (P2), pressure order ~2 (P1), against the analytic manufactured
   solution with hand-derived forcing.
2. **MMS on a production star** (1024-gon, train-band deformation): the
   error across resolutions including the catalog's production
   ``target_h = 0.0105``, where the velocity crosses the 1e-6 bar, with
   wall-clock cost per solve (the cost curve that set the resolution).
3. **Reynolds robustness grid**: Newton behavior (iterations,
   backtracking, continuation, failures) across Re x deformation at
   production resolution -- the evidence behind the family's declared
   achievable band (train [10, 200], unseen-Re (200, 300]).

Pressure comparisons align means over the query set (the discrete gauge is
zero mean over the domain; mean alignment is the gauge-invariant metric).

Usage::

    python ns_solver_verification.py --output ../results/ns_solver_verification_<date>.json

This is a benchmark-local research study, not a proposed public API.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from pathlib import Path

import _paths  # noqa: F401
import generate_datasets as gd
import numpy as np
import torch
from conformal_laplace import sample_geometry
from fem_navier_stokes import (
    LINEAR_SOLVER,
    NewtonError,
    manufactured_solution,
    solve_navier_stokes,
)
from provenance import runtime_environment, source_provenance

_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _aligned_pressure_error(computed: np.ndarray, exact: np.ndarray) -> float:
    return _relative_l2(computed - computed.mean(), exact - exact.mean())


def _star_case(seed: int, deformation: tuple[float, float], n_boundary: int):
    geometry = sample_geometry(
        seed, modes=(2, 3), deformation_range=deformation, dtype=torch.float64
    )
    angles = 2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    loop = gd._star_boundary(geometry, angles)
    queries, _ = gd._sample_bucketed_queries(
        seed + 4,
        loop,
        n_interior=128,
        n_near=32,
        interior_margin=0.12,
        near_band=(0.02, 0.08),
    )
    return loop, queries


def mms_square_convergence(h_values: list[float], viscosity: float) -> dict:
    """MMS convergence table and observed orders on the unit square."""

    mms = manufactured_solution(viscosity)
    queries = np.random.default_rng(0).uniform(0.15, 0.85, size=(300, 2))
    exact_velocity = mms["velocity"](queries)
    exact_pressure = mms["pressure"](queries)
    rows = []
    for h in h_values:
        start = time.perf_counter()
        solution = solve_navier_stokes(
            [_SQUARE],
            mms["velocity"],
            queries,
            viscosity=viscosity,
            target_h=h,
            forcing=mms["forcing"],
        )
        rows.append(
            {
                "target_h": h,
                "n_triangles": solution.diagnostics.n_triangles,
                "velocity_rel_l2": _relative_l2(
                    solution.velocity_query, exact_velocity
                ),
                "pressure_rel_l2": _aligned_pressure_error(
                    solution.pressure_query, exact_pressure
                ),
                "newton_iterations": solution.diagnostics.newton_iterations,
                "momentum_balance_error": (solution.diagnostics.momentum_balance_error),
                "seconds": time.perf_counter() - start,
            }
        )
    log_h = np.log([row["target_h"] for row in rows])
    return {
        "viscosity": viscosity,
        "rows": rows,
        "velocity_order": float(
            np.polyfit(log_h, np.log([r["velocity_rel_l2"] for r in rows]), 1)[0]
        ),
        "pressure_order": float(
            np.polyfit(log_h, np.log([r["pressure_rel_l2"] for r in rows]), 1)[0]
        ),
    }


def mms_star_resolutions(h_values: list[float], viscosity: float) -> dict:
    """MMS on the production star at production/verification resolutions."""

    mms = manufactured_solution(viscosity)
    loop, queries = _star_case(7, (0.05, 0.35), 1024)
    exact_velocity = mms["velocity"](queries)
    exact_pressure = mms["pressure"](queries)
    rows = []
    for h in h_values:
        start = time.perf_counter()
        solution = solve_navier_stokes(
            [loop],
            mms["velocity"],
            queries,
            viscosity=viscosity,
            target_h=h,
            forcing=mms["forcing"],
        )
        rows.append(
            {
                "target_h": h,
                "n_triangles": solution.diagnostics.n_triangles,
                "velocity_rel_l2": _relative_l2(
                    solution.velocity_query, exact_velocity
                ),
                "pressure_rel_l2": _aligned_pressure_error(
                    solution.pressure_query, exact_pressure
                ),
                "seconds": time.perf_counter() - start,
            }
        )
    return {"viscosity": viscosity, "n_fem_boundary": 1024, "rows": rows}


def reynolds_robustness_grid(reynolds_values: list[float], target_h: float) -> dict:
    """Newton behavior across Re and deformation at production resolution."""

    cases = []
    for seed, deformation in ((3, (0.05, 0.35)), (5, (0.45, 0.65)), (9, (0.45, 0.65))):
        geometry = sample_geometry(
            seed, modes=(2, 3), deformation_range=deformation, dtype=torch.float64
        )
        angles = 2.0 * math.pi * torch.arange(1024, dtype=torch.float64) / 1024
        loop = gd._star_boundary(geometry, angles)
        dense_arc, total_length = gd._cumulative_arc_length(loop)
        evaluate, _ = gd._sample_arc_drive_profile(
            seed + 100,
            band=(0, 2),
            decay=1.0,
            width_range=(0.25, 0.5),
            total_length=total_length,
        )
        raw = evaluate(dense_arc)
        velocity = (raw / np.abs(raw).max())[:, None] * gd._star_tangents(
            geometry, angles
        )
        queries, _ = gd._sample_bucketed_queries(
            seed + 200,
            loop,
            n_interior=64,
            n_near=16,
            interior_margin=0.12,
            near_band=(0.02, 0.08),
        )
        cases.append((seed, deformation, loop, velocity, queries))

    rows = []
    for reynolds in reynolds_values:
        for seed, deformation, loop, velocity, queries in cases:
            start = time.perf_counter()
            try:
                solution = solve_navier_stokes(
                    [loop],
                    [velocity],
                    queries,
                    viscosity=1.0 / reynolds,
                    target_h=target_h,
                    continuation=True,
                )
                diagnostics = solution.diagnostics
                rows.append(
                    {
                        "reynolds": reynolds,
                        "geometry_seed": seed,
                        "deformation_max": deformation[1],
                        "converged": True,
                        "newton_iterations": diagnostics.newton_iterations,
                        "continuation_solves": diagnostics.continuation_solves,
                        "backtracking_steps": diagnostics.backtracking_steps,
                        "velocity_speed_max": diagnostics.velocity_speed_max,
                        "divergence_l2_normalized": (
                            diagnostics.divergence_l2_normalized
                        ),
                        "seconds": time.perf_counter() - start,
                    }
                )
            except NewtonError as error:
                rows.append(
                    {
                        "reynolds": reynolds,
                        "geometry_seed": seed,
                        "deformation_max": deformation[1],
                        "converged": False,
                        "error": str(error),
                        "seconds": time.perf_counter() - start,
                    }
                )
    return {"target_h": target_h, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON path (default results/ns_solver_verification_<date>.json)",
    )
    parser.add_argument("--skip-re-grid", action="store_true")
    parser.add_argument(
        "--max-reynolds", type=float, default=400.0, help="Re grid upper end"
    )
    arguments = parser.parse_args()

    date = datetime.date.today().isoformat()
    output = arguments.output or (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"ns_solver_verification_{date}.json"
    )

    report = {
        "title": "ns_cavity_star solver verification (iteration 36)",
        "date": date,
        "linear_solver": LINEAR_SOLVER,
        "production_settings": {"target_h": 0.0105, "n_fem_boundary": 1024},
        "mms_square_convergence": mms_square_convergence(
            [0.16, 0.08, 0.04, 0.02], viscosity=0.05
        ),
        "mms_star_resolutions": mms_star_resolutions(
            [0.02, 0.015, 0.0125, 0.0105, 0.01], viscosity=0.02
        ),
        "provenance": source_provenance(),
        "environment": runtime_environment(torch.device("cpu")),
    }
    if not arguments.skip_re_grid:
        grid = [100.0, 200.0, 300.0]
        if arguments.max_reynolds > 300.0:
            grid.append(arguments.max_reynolds)
        report["reynolds_robustness"] = reynolds_robustness_grid(grid, 0.0105)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
