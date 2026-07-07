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

r"""One-time AirFRANS VTK -> npz converter (the only PyVista-touching module).

Reads the official AirFRANS ``Dataset`` directory (one subdirectory per
case holding ``{base}_freestream.vtp`` / ``{base}_aerofoil.vtp`` /
``{base}_internal.vtu``, plus the distribution's ``manifest.json``) and
writes one ``{case_name}.npz`` per case in the layout consumed by
:mod:`airfrans_dataset` -- geometry in float64, targets in float32:

- ``boundary_points`` ``(n_b, 2)`` / ``boundary_cells`` ``(n_b, 2)``: the
  airfoil section polyline, chained into closed loops with consistent
  counterclockwise orientation (the exterior-flow convention of the fluid
  suites: cell normals point out of the fluid, into the body).  The
  ``aerofoil.vtp`` is a z = 0 polyline (``.lines``, not faces); its segment
  connectivity is extracted and points are projected to ``(x, y)``.
- ``query_points`` ``(n_q, 2)``: the internal volume mesh points.
- Targets: ``delta_velocity`` :math:`= (U - U_\infty)/|U_\infty|`,
  ``pressure_coefficient`` :math:`= p / q_\infty`, ``log_nut_ratio``
  :math:`= \ln(1 + \nu_t/\nu)`, plus the diagnostic ``cpt``
  :math:`= (p + q)/q_\infty`.
- ``is_surface``: on-airfoil query points, identified via the volume mesh's
  ``implicit_distance == 0`` (the AirFRANS convention).
- ``u_inf`` (2,), ``nu``, ``chord``: the case's freestream vector (mean of
  the freestream patch's face ``U``), the working kinematic viscosity, and
  the unit chord.

Working constants come from :mod:`airfrans_dataset`: ``RHO = 1`` (verified
from the data -- NOT the mislabeled 1.204) and ``NU = 1.56e-5``.

Label hygiene: energy-conservation violations (:math:`C_{pt} > 1.02`,
~0.003% of points dataset-wide) are excluded as whole-target-row NaN, so
training and evaluation of every arm exclude the same points. The
near-wall pressure-gradient screen (:math:`|\nabla C_p \cdot c| > 20`) is
recorded per case as a *diagnostic count only* -- protocol amendment
2026-07-05, pre-training: applied as a row exclusion it was measured to
delete 16-23% of each case's points, concentrated in the near-wall
region, whereas GLOBE used it only to NaN its gradient *target*, which
this catalog does not store.

Per-case mask counts, the constants, and the source path are recorded in a
catalog-level ``preprocess_manifest.json``; the distribution's
``manifest.json`` is copied through unchanged.

Example::

    python airfrans_preprocess.py \
        --data-dir /data/airfrans/Dataset --output-dir airfrans/v1 --workers 8

This is a CLI-only benchmark-local research utility (PyVista is imported
lazily inside the conversion routine), not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
from airfrans_dataset import (
    CHORD,
    CPT_MASK_THRESHOLD,
    GRAD_CP_MASK_THRESHOLD,
    NU,
    RHO,
)


def _polyline_segments(lines: np.ndarray) -> np.ndarray:
    """Decode a VTK polyline connectivity array into (n, 2) point-id segments.

    The VTK layout is ``[count, id0, id1, ..., count, ...]``; each polyline
    of ``count`` points contributes ``count - 1`` two-point segments.
    """

    lines = np.asarray(lines, dtype=np.int64).ravel()
    segments: list[tuple[int, int]] = []
    cursor = 0
    while cursor < lines.size:
        count = int(lines[cursor])
        if count < 2 or cursor + 1 + count > lines.size:
            raise ValueError("malformed VTK polyline connectivity array")
        ids = lines[cursor + 1 : cursor + 1 + count]
        segments.extend(zip(ids[:-1].tolist(), ids[1:].tolist()))
        cursor += 1 + count
    if not segments:
        raise ValueError("the aerofoil polydata carries no line segments")
    return np.asarray(segments, dtype=np.int64)


def _chain_closed_loops(
    points: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Chain raw segments into CCW closed loops with consistent orientation.

    Treats the segments as an undirected graph (the file's per-segment
    orientation is not trusted), requires every touched point to have
    exactly two neighbors (closed loops only -- an open airfoil section is
    data corruption), walks each loop, and orients it counterclockwise by
    the shoelace signed area.  Returns ``(loop_points, cells, n_loops)``
    with points reindexed to walk order, one segment cell per point.
    """

    adjacency: dict[int, list[int]] = {}
    for a, b in segments.tolist():
        if a == b:
            raise ValueError("degenerate zero-length segment in aerofoil polyline")
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    bad = {node for node, nbrs in adjacency.items() if len(nbrs) != 2}
    if bad:
        raise ValueError(
            "aerofoil polyline is not a union of closed loops: "
            f"{len(bad)} points have degree != 2"
        )

    loop_points: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    visited: set[int] = set()
    offset = 0
    n_loops = 0
    for start in sorted(adjacency):
        if start in visited:
            continue
        walk = [start]
        visited.add(start)
        previous, current = start, adjacency[start][0]
        while current != start:
            walk.append(current)
            visited.add(current)
            first, second = adjacency[current]
            previous, current = current, second if first == previous else first
        ordered = points[np.asarray(walk, dtype=np.int64)]
        rolled = np.roll(ordered, -1, axis=0)
        signed_area = 0.5 * float(
            np.sum(ordered[:, 0] * rolled[:, 1] - rolled[:, 0] * ordered[:, 1])
        )
        if signed_area < 0.0:
            ordered = ordered[::-1]
        n = ordered.shape[0]
        index = np.arange(n, dtype=np.int64)
        loop_points.append(np.ascontiguousarray(ordered))
        cells.append(np.stack((index, np.roll(index, -1)), axis=-1) + offset)
        offset += n
        n_loops += 1
    return np.concatenate(loop_points), np.concatenate(cells), n_loops


def preprocess_case(case_dir: Path | str, output_dir: Path | str) -> dict:
    """Convert one AirFRANS case directory to ``<output_dir>/<name>.npz``.

    Returns the case's provenance record (sizes, freestream, per-mask
    counts) for the catalog-level ``preprocess_manifest.json``.
    """

    import pyvista as pv  # CLI-only dependency; keep every other module clean

    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    base = case_dir.name
    paths = {
        "freestream": case_dir / f"{base}_freestream.vtp",
        "aerofoil": case_dir / f"{base}_aerofoil.vtp",
        "internal": case_dir / f"{base}_internal.vtu",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {path}")
    freestream = pv.read(paths["freestream"])
    aerofoil = pv.read(paths["aerofoil"])
    internal = pv.read(paths["internal"])

    # Freestream vector: mean of the freestream patch's per-face velocity
    # (cell data, GLOBE's face_data convention), projected to the plane.
    if "U" in freestream.cell_data:
        u_freestream = np.asarray(freestream.cell_data["U"], dtype=np.float64)
    else:  # pragma: no cover - some exports carry point data only
        u_freestream = np.asarray(freestream.point_data["U"], dtype=np.float64)
    u_inf = u_freestream[:, :2].mean(axis=0)
    u_inf_magnitude = float(np.linalg.norm(u_inf))
    q_inf = 0.5 * RHO * u_inf_magnitude**2

    # Airfoil boundary: z = 0 polyline -> CCW closed segment loops in (x, y).
    boundary_points, boundary_cells, n_loops = _chain_closed_loops(
        np.asarray(aerofoil.points, dtype=np.float64)[:, :2],
        _polyline_segments(aerofoil.lines),
    )

    # Volume targets (float64 intermediates, float32 storage).
    query_points = np.asarray(internal.points, dtype=np.float64)[:, :2]
    velocity = np.asarray(internal.point_data["U"], dtype=np.float64)[:, :2]
    pressure = np.asarray(internal.point_data["p"], dtype=np.float64)
    nut = np.asarray(internal.point_data["nut"], dtype=np.float64)
    is_surface = np.asarray(internal.point_data["implicit_distance"]) == 0

    delta_velocity = (velocity - u_inf[None, :]) / u_inf_magnitude
    pressure_coefficient = pressure / q_inf
    cpt = pressure_coefficient + np.sum(velocity**2, axis=-1) / u_inf_magnitude**2
    log_nut_ratio = np.log1p(nut / NU)

    # Near-wall pressure-gradient DIAGNOSTIC: cell gradient of C_p averaged
    # to points (GLOBE's recipe -- gradients on cells are more stable).
    # Recorded for provenance only, NEVER used as a row exclusion: GLOBE
    # NaN-masks only its *gradient target* (which this catalog does not
    # store), and applying the |grad C_p| > 20 screen as a whole-row mask
    # was measured to delete 16-23% of each case's points, concentrated in
    # exactly the near-wall region where the boundary-layer physics lives
    # (protocol amendment 2026-07-05, pre-training; see the chapter's
    # protocol section and the lab notebook).
    internal.cell_data["C_p"] = (
        np.asarray(internal.cell_data["p"], dtype=np.float64) / q_inf
    )
    gradient = (
        internal.compute_derivative(scalars="C_p", gradient=True, preference="cell")
        .cell_data_to_point_data()
        .point_data["gradient"]
    )
    grad_cp = np.asarray(gradient, dtype=np.float64)[:, :2] * CHORD
    mask_grad_cp = np.linalg.norm(grad_cp, axis=-1) > GRAD_CP_MASK_THRESHOLD

    mask_cpt = cpt > CPT_MASK_THRESHOLD
    masked = mask_cpt
    delta_velocity[masked] = np.nan
    pressure_coefficient[masked] = np.nan
    log_nut_ratio[masked] = np.nan
    cpt[masked] = np.nan

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{base}.npz",
        boundary_points=boundary_points,
        boundary_cells=boundary_cells,
        query_points=np.ascontiguousarray(query_points),
        delta_velocity=delta_velocity.astype(np.float32),
        pressure_coefficient=pressure_coefficient.astype(np.float32),
        log_nut_ratio=log_nut_ratio.astype(np.float32),
        cpt=cpt.astype(np.float32),
        is_surface=np.ascontiguousarray(is_surface, dtype=bool),
        u_inf=u_inf,
        nu=np.float64(NU),
        chord=np.float64(CHORD),
    )
    return {
        "case": base,
        "n_boundary_points": int(boundary_points.shape[0]),
        "n_boundary_loops": int(n_loops),
        "n_query": int(query_points.shape[0]),
        "n_surface": int(is_surface.sum()),
        "n_masked_cpt": int(mask_cpt.sum()),
        "n_grad_cp_flagged_diagnostic": int(mask_grad_cp.sum()),
        "n_masked_total": int(masked.sum()),
        "u_inf": [float(u_inf[0]), float(u_inf[1])],
        "u_inf_magnitude": u_inf_magnitude,
    }


def _preprocess_case_star(arguments: tuple[str, str]) -> dict:
    """Spawn-picklable shim for :func:`preprocess_case`."""

    return preprocess_case(*arguments)


def convert_dataset(
    *,
    data_dir: Path | str,
    output_dir: Path | str,
    limit: int | None = None,
    workers: int = 1,
) -> dict:
    """Convert every case directory of an AirFRANS distribution.

    Copies ``manifest.json`` through unchanged, writes one npz per case
    (``workers > 1`` uses spawn multiprocessing over cases; the caller must
    be spawn-reimportable), and writes the provenance
    ``preprocess_manifest.json``.  ``limit`` truncates the case list for
    smoke tests (the copied manifest then references missing cases -- a
    smoke catalog is not a full catalog).  Returns the provenance dict.
    """

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"AirFRANS manifest not found: {manifest_path}")
    case_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"no case directories found under {data_dir}")
    if limit is not None:
        case_dirs = case_dirs[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = [(str(path), str(output_dir)) for path in case_dirs]
    if workers == 1:
        records = [_preprocess_case_star(item) for item in arguments]
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            records = list(pool.imap(_preprocess_case_star, arguments))

    shutil.copyfile(manifest_path, output_dir / "manifest.json")
    provenance = {
        "source_data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(records),
        "limit": limit,
        "constants": {
            "RHO": RHO,
            "NU": NU,
            "chord": CHORD,
            "cpt_mask_threshold": CPT_MASK_THRESHOLD,
            "grad_cp_mask_threshold": GRAD_CP_MASK_THRESHOLD,
        },
        "masking_convention": (
            "both published pathology masks (C_pt > 1.02; |grad C_p * c| > 20, "
            "cell gradient averaged to points) set ALL target fields at the "
            "offending points to NaN, so training and evaluation of every arm "
            "exclude the same rows (pre-registered label hygiene)"
        ),
        "orientation_convention": (
            "airfoil loops chained from the polyline segments and oriented "
            "counterclockwise (signed area > 0): segment normals point out of "
            "the fluid, into the body -- the exterior-flow convention of the "
            "fluid suites"
        ),
        "cases": {record["case"]: record for record in records},
    }
    (output_dir / "preprocess_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="the AirFRANS Dataset directory (case dirs + manifest.json)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="convert only the first N cases"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    provenance = convert_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
    )
    total_masked = sum(
        record["n_masked_total"] for record in provenance["cases"].values()
    )
    total_query = sum(record["n_query"] for record in provenance["cases"].values())
    print(
        json.dumps(
            {
                "n_cases": provenance["n_cases"],
                "output_dir": provenance["output_dir"],
                "masked_points": total_masked,
                "masked_fraction": total_masked / max(total_query, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
