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

"""Precompute dense N-S gallery fields for the book's dataset chapter.

Peter's review of the Navier-Stokes chapter: the sample gallery drawn as
a 160-point cloud is hard to interpret; render the smooth underlying
solution (like the hero figure) and structure the gallery as the SAME
geometry+drive across the Reynolds range (rows of fixed sample, columns
of Re) plus several distinct samples.

This script solves that exact grid with the v1 catalog's own case
construction (``generate_datasets`` samplers, ``solve_navier_stokes``
with ``keep_mesh=True``) at pinned geometry/drive seeds, and writes ONE
compact npz for ``book/data/`` holding per-case P1 vertex fields:

    case{k}_points, case{k}_triangles, case{k}_velocity, case{k}_pressure,
    case{k}_loop, case{k}_wall_speed, case{k}_reynolds, case{k}_seed

plus, for the first case only, the as-posed 160-point sampling
(``asposed_points``, ``asposed_speed``) so the gallery can show what the
model actually receives against the smooth truth.

Run on a cluster CPU allocation (each solve is a sparse Newton at
h=0.02, a few GB peak); local execution is deliberately discouraged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for entry in (_HERE.parent, _HERE.parent / "datasets", _HERE.parent / "problems"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from conformal_laplace import sample_geometry  # noqa: E402
from fem_navier_stokes import solve_navier_stokes  # noqa: E402
from generate_datasets import (  # noqa: E402
    _cumulative_arc_length,
    _sample_arc_drive_profile,
    _star_boundary,
    _star_tangents,
    _substream,
)

#: The v1 train-split construction constants (mirrors _ns_split_spec /
#: NSGeneratorSettings defaults; provenance recorded in the output).
GEOMETRY_MODES = (2, 3)
DEFORMATION_RANGE = (0.05, 0.35)
DRIVE_BAND = (1.0, 6.0)
DRIVE_WIDTH_RANGE = (0.18, 0.45)
DRIVE_DECAY = 2.0
N_FEM_BOUNDARY = 2048
TARGET_H = 0.02

GEOMETRY_SEEDS = (3, 7, 12)
REYNOLDS_VALUES = (10.0, 50.0, 150.0, 300.0)


def build_case(seed: int, reynolds: float, rng_queries: bool = False) -> dict:
    geometry = sample_geometry(
        _substream(seed, 0),
        modes=GEOMETRY_MODES,
        deformation_range=DEFORMATION_RANGE,
        dtype=torch.float64,
    )
    dense_angles = (
        2.0 * math.pi * torch.arange(N_FEM_BOUNDARY, dtype=torch.float64)
        / N_FEM_BOUNDARY
    )
    fem_loop = _star_boundary(geometry, dense_angles)
    dense_arc, total_length = _cumulative_arc_length(fem_loop)
    evaluate_drive, drive_parameters = _sample_arc_drive_profile(
        _substream(seed, 2),
        band=DRIVE_BAND,
        decay=DRIVE_DECAY,
        width_range=DRIVE_WIDTH_RANGE,
        total_length=total_length,
    )
    raw = evaluate_drive(dense_arc)
    peak = float(np.abs(raw).max())
    if peak < 1.0e-8:
        raise RuntimeError(f"degenerate drive at seed {seed}")
    tangents = _star_tangents(geometry, dense_angles)
    fem_velocity = ((raw / peak)[:, None] * tangents).astype(np.float64)

    n_probe = 160 if rng_queries else 4
    probe_rng = np.random.default_rng(_substream(seed, 4))
    # Uniform-in-disc rejection sampling inside the star (matches the v1
    # recipe's interior draw closely enough for an illustrative overlay).
    loop_np = np.asarray(fem_loop, dtype=np.float64)
    lo, hi = loop_np.min(axis=0), loop_np.max(axis=0)
    from matplotlib.path import Path as MplPath

    polygon = MplPath(loop_np)
    queries = []
    while len(queries) < n_probe:
        cand = probe_rng.uniform(lo, hi, size=(4 * n_probe, 2))
        inside = polygon.contains_points(cand, radius=-1e-9)
        queries.extend(cand[inside][: n_probe - len(queries)])
    query_points = np.asarray(queries, dtype=np.float64)

    start = time.perf_counter()
    solution = solve_navier_stokes(
        [loop_np],
        [fem_velocity],
        query_points,
        viscosity=1.0 / reynolds,
        target_h=TARGET_H,
        keep_mesh=True,
    )
    elapsed = time.perf_counter() - start
    n_vertices = solution.vertex_pressure.shape[0]
    return {
        "points": solution.node_points[:n_vertices].astype(np.float32),
        "triangles": solution.triangles.astype(np.int32),
        "velocity": solution.node_velocity[:n_vertices].astype(np.float32),
        "pressure": solution.vertex_pressure.astype(np.float32),
        "loop": loop_np.astype(np.float32),
        "wall_speed": np.abs(raw / peak).astype(np.float32),
        "asposed_points": query_points.astype(np.float32),
        "asposed_speed": np.linalg.norm(
            solution.velocity_query, axis=1
        ).astype(np.float32),
        "meta": {
            "seed": seed,
            "reynolds": reynolds,
            "drive": drive_parameters,
            "elapsed_seconds": elapsed,
            "n_vertices": int(n_vertices),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    payload: dict[str, np.ndarray] = {}
    metas = []
    k = 0
    for seed in GEOMETRY_SEEDS:
        for reynolds in REYNOLDS_VALUES:
            case = build_case(seed, reynolds, rng_queries=(k == 0))
            for key in ("points", "triangles", "velocity", "pressure",
                        "loop", "wall_speed"):
                payload[f"case{k}_{key}"] = case[key]
            if k == 0:
                payload["asposed_points"] = case["asposed_points"]
                payload["asposed_speed"] = case["asposed_speed"]
            metas.append(case["meta"])
            print(json.dumps({"case": k, **case["meta"]}), flush=True)
            k += 1
    payload["n_cases"] = np.int64(k)
    payload["geometry_seeds"] = np.asarray(GEOMETRY_SEEDS, dtype=np.int64)
    payload["reynolds_values"] = np.asarray(REYNOLDS_VALUES)
    payload["__meta__"] = np.frombuffer(
        json.dumps({
            "script": "studies/ns_gallery_fields.py",
            "target_h": TARGET_H,
            "construction": "v1 train-split samplers, pinned seeds",
            "cases": metas,
        }).encode(), dtype=np.uint8)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **payload)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
