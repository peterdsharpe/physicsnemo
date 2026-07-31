# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Audit sparse surface representations on the full DrivAerML CFD master mesh.

The full vehicle boundary of each case is the frozen integration master.  Two
piecewise-constant representation spaces are compared at the same cell budget:

* the production-form cyclic block of on-disk triangles, reconstructed by
  ambient nearest-centroid assignment; and
* a deterministic, area-centroidal cover assigned in the preregistered
  augmented centroid-and-normal space.

Both spaces are restricted from, prolonged to, and transferred through the
same full CFD mesh.  Consequently their constants, integrals, and mass-adjoint
transfer identities are auditable without pretending that two sparse triangle
sets have a literal polygon overlap.

This file deliberately imports no PhysicsNeMo repository code.  Production
queries use SciPy's cKDTree; ``--synthetic-smoke`` has a bounded NumPy fallback
so the algebra and JSON path can be exercised in a minimal local environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import scipy
    from scipy.spatial import cKDTree
except ImportError:  # The bounded fallback is used only by the synthetic smoke test.
    scipy = None
    cKDTree = None


ARRAY_PATHS = {
    "points": "points.memmap",
    "cells": "cells.memmap",
    "CpMeanTrim": "cell_data/CpMeanTrim.memmap",
    "wallShearStressMeanTrim": "cell_data/wallShearStressMeanTrim.memmap",
}
FILL_QUANTILES = (0.5, 0.9, 0.95, 0.99)
ALGEBRA_TOLERANCE = 1.0e-12
NUMPY_FALLBACK_MAX_PAIR_DISTANCES = 50_000_000
MAX_RESTRICTION_REPAIRS = 2


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _finite(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Diagnostic produced a non-finite value: {result}")
    return result


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, FILL_QUANTILES)
    return {
        "min": _finite(values.min()),
        "mean": _finite(values.mean(dtype=np.float64)),
        "q50": _finite(quantiles[0]),
        "q90": _finite(quantiles[1]),
        "q95": _finite(quantiles[2]),
        "q99": _finite(quantiles[3]),
        "max": _finite(values.max()),
    }


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[order].astype(np.float64, copy=False)
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    if cumulative[-1] <= 0.0:
        raise ValueError("Weighted quantiles require positive total weight")
    cumulative /= cumulative[-1]
    sorted_values = values[order]
    return {
        f"q{round(100 * quantile):02d}": _finite(
            np.interp(quantile, cumulative, sorted_values)
        )
        for quantile in FILL_QUANTILES
    }


def _relative_vector_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return _finite(float(np.linalg.norm(actual - expected)) / denominator)


@dataclass
class SurfaceCase:
    case_id: str
    points: np.ndarray
    cells: np.ndarray
    cp: np.ndarray
    wss: np.ndarray
    source: dict[str, Any]

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])


@dataclass
class GeometryChunk:
    start: int
    stop: int
    areas: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    cp: np.ndarray
    wss: np.ndarray


@dataclass
class Support:
    name: str
    points: np.ndarray
    normals: np.ndarray
    normal_length_scale: float
    definition: dict[str, Any]

    @property
    def k(self) -> int:
        return int(self.points.shape[0])


@dataclass
class RestrictedFields:
    measures: np.ndarray
    cp: np.ndarray
    wss: np.ndarray


def _triangle_geometry(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = np.cross(edge_1, edge_2)
    twice_area_f32 = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area_f32.astype(np.float64)
    centroids = vertices.mean(axis=1, dtype=np.float32)
    normals = np.zeros_like(centroids)
    np.divide(
        cross,
        twice_area_f32[:, None],
        out=normals,
        where=twice_area_f32[:, None] > 0.0,
    )
    return areas, centroids, normals


def _geometry_chunks(
    case: SurfaceCase,
    chunk_cells: int,
) -> Iterator[GeometryChunk]:
    for start in range(0, case.n_cells, chunk_cells):
        stop = min(start + chunk_cells, case.n_cells)
        cell_indices = np.asarray(case.cells[start:stop], dtype=np.int64)
        vertices = case.points[cell_indices]
        areas, centroids, normals = _triangle_geometry(vertices)
        yield GeometryChunk(
            start=start,
            stop=stop,
            areas=areas,
            centroids=centroids,
            normals=normals,
            cp=np.asarray(case.cp[start:stop], dtype=np.float32),
            wss=np.asarray(case.wss[start:stop], dtype=np.float32),
        )


def _geometry_at_indices(
    case: SurfaceCase,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.asarray(case.cells[indices], dtype=np.int64)
    return _triangle_geometry(case.points[cells])


def _point_bounds(
    points: np.ndarray, chunk_points: int
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(points), chunk_points):
        stop = min(start + chunk_points, len(points))
        chunk = np.asarray(points[start:stop], dtype=np.float32)
        lower = np.minimum(lower, chunk.min(axis=0))
        upper = np.maximum(upper, chunk.max(axis=0))
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Surface point bounds are not finite")
    return lower, upper


def _master_area(
    case: SurfaceCase,
    chunk_cells: int,
) -> tuple[float, int]:
    total = 0.0
    degenerate = 0
    for chunk_number, chunk in enumerate(_geometry_chunks(case, chunk_cells), start=1):
        total += chunk.areas.sum(dtype=np.float64)
        degenerate += int(np.count_nonzero(chunk.areas <= 0.0))
        if chunk_number % 10 == 0 or chunk.stop == case.n_cells:
            _log(
                f"{case.case_id} master_area completed_cells="
                f"{chunk.stop:,}/{case.n_cells:,}"
            )
    if total <= 0.0:
        raise ValueError("Master surface has nonpositive total area")
    return _finite(total), degenerate


def _resolve_vehicle_root(path: Path) -> Path:
    direct = path
    nested = path / "_tensordict" / "boundaries" / "vehicle" / "_tensordict"
    if (direct / "points.memmap").is_file():
        return direct
    if (nested / "points.memmap").is_file():
        return nested
    raise FileNotFoundError(
        f"{path} is neither a vehicle TensorDict root nor a .pdmsh containing one"
    )


def _metadata_dtype(metadata: dict[str, Any], name: str) -> np.dtype[Any]:
    dtype_name = metadata[name]["dtype"]
    expected = {
        "torch.float32": np.dtype("<f4"),
        "torch.int64": np.dtype("<i8"),
    }
    if dtype_name not in expected:
        raise ValueError(
            f"Unsupported {name} dtype in TensorDict metadata: {dtype_name}"
        )
    return expected[dtype_name]


def _load_pdmsh_case(
    case_id: str,
    path: Path,
    *,
    hash_inputs: bool,
) -> SurfaceCase:
    root = _resolve_vehicle_root(path)
    mesh_meta_path = root / "meta.json"
    cell_meta_path = root / "cell_data" / "meta.json"
    mesh_meta = json.loads(mesh_meta_path.read_text())
    cell_meta = json.loads(cell_meta_path.read_text())

    required_mesh = ("points", "cells")
    required_fields = ("CpMeanTrim", "wallShearStressMeanTrim")
    for name in required_mesh:
        if name not in mesh_meta:
            raise KeyError(f"{mesh_meta_path} has no {name!r} entry")
    for name in required_fields:
        if name not in cell_meta:
            raise KeyError(f"{cell_meta_path} has no {name!r} entry")

    n_points = int(mesh_meta["points"]["shape"][0])
    n_cells = int(mesh_meta["cells"]["shape"][0])
    expected_shapes = {
        "points": (n_points, 3),
        "cells": (n_cells, 3),
        "CpMeanTrim": (n_cells,),
        "wallShearStressMeanTrim": (n_cells, 3),
    }
    metadata_by_name = {
        "points": mesh_meta,
        "cells": mesh_meta,
        "CpMeanTrim": cell_meta,
        "wallShearStressMeanTrim": cell_meta,
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(metadata_by_name[name][name]["shape"])
        if actual_shape != expected_shape:
            raise ValueError(
                f"{root / ARRAY_PATHS[name]} has metadata shape {actual_shape}, "
                f"expected {expected_shape}"
            )

    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for name, relative_path in ARRAY_PATHS.items():
        array_path = root / relative_path
        metadata = metadata_by_name[name]
        dtype = _metadata_dtype(metadata, name)
        shape = expected_shapes[name]
        expected_bytes = math.prod(shape) * dtype.itemsize
        stat = array_path.stat()
        if stat.st_size != expected_bytes:
            raise ValueError(
                f"{array_path} has {stat.st_size} bytes, expected {expected_bytes}"
            )
        arrays[name] = np.memmap(array_path, mode="r", dtype=dtype, shape=shape)
        provenance[name] = {
            "path": str(array_path.resolve()),
            "shape": list(shape),
            "dtype": str(dtype),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(array_path) if hash_inputs else None,
        }

    return SurfaceCase(
        case_id=case_id,
        points=arrays["points"],
        cells=arrays["cells"],
        cp=arrays["CpMeanTrim"],
        wss=arrays["wallShearStressMeanTrim"],
        source={
            "kind": "curated_drivaerml_vehicle_tensordict_memmaps",
            "requested_path": str(path),
            "vehicle_root": str(root.resolve()),
            "n_points": n_points,
            "n_cells": n_cells,
            "metadata": {
                "mesh": {
                    "path": str(mesh_meta_path.resolve()),
                    "sha256": _sha256_file(mesh_meta_path),
                },
                "cell_data": {
                    "path": str(cell_meta_path.resolve()),
                    "sha256": _sha256_file(cell_meta_path),
                },
            },
            "arrays": provenance,
        },
    )


class _NeighborIndex:
    def __init__(self, points: np.ndarray, workers: int):
        self.points = np.asarray(points, dtype=np.float32)
        self.workers = workers
        if cKDTree is not None:
            self.tree = cKDTree(self.points)
            self.backend = "scipy.spatial.cKDTree"
        else:
            self.tree = None
            self.backend = "bounded_numpy_pairwise_fallback"

    def query(
        self,
        queries: np.ndarray,
        candidate_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_count = min(candidate_count, len(self.points))
        queries = np.asarray(queries, dtype=np.float32)
        if self.tree is not None:
            distances, indices = self.tree.query(
                queries,
                k=candidate_count,
                workers=self.workers,
            )
            return np.asarray(distances), np.asarray(indices, dtype=np.int64)

        pair_count = len(queries) * len(self.points)
        if pair_count > NUMPY_FALLBACK_MAX_PAIR_DISTANCES:
            raise RuntimeError(
                "SciPy is required for this production-size nearest-neighbor "
                f"query ({pair_count:,} pair distances)"
            )
        query_sq = np.einsum("ij,ij->i", queries, queries, optimize=True)
        point_sq = np.einsum("ij,ij->i", self.points, self.points, optimize=True)
        squared = query_sq[:, None] + point_sq[None, :]
        squared -= 2.0 * (queries @ self.points.T)
        np.maximum(squared, 0.0, out=squared)
        if candidate_count == 1:
            indices = np.argmin(squared, axis=1)
            distances = np.sqrt(squared[np.arange(len(queries)), indices])
            return distances, indices.astype(np.int64, copy=False)

        indices = np.argpartition(
            squared,
            kth=candidate_count - 1,
            axis=1,
        )[:, :candidate_count]
        candidate_squared = np.take_along_axis(squared, indices, axis=1)
        order = np.argsort(candidate_squared, axis=1, kind="stable")
        indices = np.take_along_axis(indices, order, axis=1)
        distances = np.sqrt(np.take_along_axis(candidate_squared, order, axis=1))
        return distances, indices.astype(np.int64, copy=False)


def _search_coordinates(
    points: np.ndarray,
    normals: np.ndarray,
    normal_length_scale: float,
) -> np.ndarray:
    if normal_length_scale == 0.0:
        return np.asarray(points, dtype=np.float32)
    return np.concatenate(
        (
            np.asarray(points, dtype=np.float32),
            normal_length_scale * np.asarray(normals, dtype=np.float32),
        ),
        axis=1,
    )


def _support_index(support: Support, workers: int) -> _NeighborIndex:
    return _NeighborIndex(
        _search_coordinates(
            support.points,
            support.normals,
            support.normal_length_scale,
        ),
        workers,
    )


def _assign(
    index: _NeighborIndex,
    support: Support,
    centroids: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = _search_coordinates(
        centroids,
        normals,
        support.normal_length_scale,
    )
    _, assignment = index.query(queries, 1)
    assignment = np.asarray(assignment, dtype=np.int64)
    selected_dots = np.einsum(
        "ij,ij->i",
        support.normals[assignment],
        normals,
        dtype=np.float64,
        optimize=True,
    )
    delta = centroids - support.points[assignment]
    distances = np.linalg.norm(delta, axis=1)
    return (
        assignment.astype(np.int64, copy=False),
        distances.astype(np.float32, copy=False),
        selected_dots.astype(np.float32, copy=False),
    )


def _cyclic_support(
    case: SurfaceCase,
    k: int,
    start: int,
    seed: int,
    ordinal: int,
) -> Support:
    if not 0 <= start < case.n_cells:
        raise ValueError(
            f"{case.case_id} cyclic start {start} is outside [0, {case.n_cells})"
        )
    indices = (start + np.arange(k, dtype=np.int64)) % case.n_cells
    areas, centroids, normals = _geometry_at_indices(case, indices)
    if np.any(areas <= 0.0):
        raise ValueError("Cyclic support includes a degenerate triangle")
    return Support(
        name="cyclic_sparse",
        points=centroids,
        normals=normals,
        normal_length_scale=0.0,
        definition={
            "selection": (
                "explicit cyclic block mathematically identical to production "
                "cyclic-block indexing"
            ),
            "start": start,
            "frozen_seed": seed,
            "frozen_seed_role": (
                "provenance for the external frozen draw; the explicit start "
                "is authoritative and no RNG is called by this audit"
            ),
            "replicate_ordinal": ordinal,
            "wraps": bool(start + k > case.n_cells),
            "indices_sha256_int64": _sha256_array(indices),
            "first_indices": [int(value) for value in indices[:10]],
            "native_selected_geometric_area": _finite(areas.sum(dtype=np.float64)),
            "assignment": (
                "ambient nearest support centroid; normal mismatch is diagnosed "
                "but does not alter the production-like nearest choice"
            ),
        },
    )


def _morton_keys(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    levels = (1 << 20) - 1
    span = np.maximum(upper - lower, np.finfo(np.float64).tiny)
    normalized = np.clip((points - lower) / span, 0.0, 1.0)
    coordinates = np.asarray(normalized * levels, dtype=np.uint64)
    keys = np.zeros(len(points), dtype=np.uint64)
    for bit in range(20):
        keys |= ((coordinates[:, 0] >> bit) & 1) << (3 * bit)
        keys |= ((coordinates[:, 1] >> bit) & 1) << (3 * bit + 1)
        keys |= ((coordinates[:, 2] >> bit) & 1) << (3 * bit + 2)
    return keys


def _initial_cover(
    case: SurfaceCase,
    *,
    k: int,
    candidate_multiplier: int,
    lower: np.ndarray,
    upper: np.ndarray,
    normal_length_scale: float,
) -> tuple[Support, dict[str, np.ndarray]]:
    candidate_count = min(case.n_cells, candidate_multiplier * k)
    candidate_indices = (
        np.arange(candidate_count, dtype=np.int64) * case.n_cells // candidate_count
    )
    areas, centroids, normals = _geometry_at_indices(case, candidate_indices)
    valid = areas > 0.0
    if np.count_nonzero(valid) < k:
        raise ValueError(
            f"Only {np.count_nonzero(valid)} nondegenerate initialization "
            f"candidates are available for k={k}"
        )
    candidate_indices = candidate_indices[valid]
    centroids = centroids[valid]
    normals = normals[valid]

    morton = _morton_keys(centroids, lower, upper)
    normal_octant = (
        (normals[:, 0] >= 0.0).astype(np.uint8)
        + 2 * (normals[:, 1] >= 0.0).astype(np.uint8)
        + 4 * (normals[:, 2] >= 0.0).astype(np.uint8)
    )
    order = np.lexsort((candidate_indices, morton, normal_octant))
    positions = (np.arange(k, dtype=np.int64) * 2 + 1) * len(order) // (2 * k)
    selected = order[positions]
    support = Support(
        name="normal_aware_centroidal_cover",
        points=centroids[selected].copy(),
        normals=normals[selected].copy(),
        normal_length_scale=normal_length_scale,
        definition={
            "initialization": (
                "evenly spaced on-disk candidates, stratified by normal octant "
                "and 3-D Morton order"
            ),
            "candidate_count": len(candidate_indices),
            "candidate_multiplier": candidate_multiplier,
            "candidate_indices_sha256_int64": _sha256_array(candidate_indices),
            "initial_seed_source_indices_sha256_int64": _sha256_array(
                candidate_indices[selected]
            ),
            "assignment": (
                "exact nearest neighbor in augmented [centroid, lambda * "
                "unit_normal] coordinates"
            ),
            "squared_assignment_cost": ("||x_i-s_j||^2 + lambda^2 ||n_i-m_j||^2"),
            "normal_coordinate_length_scale": normal_length_scale,
            "normal_coordinate_scale_definition": "sqrt(master_area / k)",
        },
    )
    candidates = {
        "indices": candidate_indices,
        "points": centroids,
        "normals": normals,
    }
    return support, candidates


def _farthest_pool(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pool_size: int,
) -> dict[str, np.ndarray]:
    distances = np.concatenate([record[0] for record in records])
    indices = np.concatenate([record[1] for record in records])
    points = np.concatenate([record[2] for record in records])
    normals = np.concatenate([record[3] for record in records])
    keep_count = min(pool_size, len(distances))
    keep = np.argpartition(distances, len(distances) - keep_count)[-keep_count:]
    order = np.lexsort((indices[keep], -distances[keep]))
    keep = keep[order]
    return {
        "distances": distances[keep],
        "indices": indices[keep],
        "points": points[keep],
        "normals": normals[keep],
    }


def _accumulate_assignment(
    case: SurfaceCase,
    support: Support,
    *,
    chunk_cells: int,
    workers: int,
    include_fields: bool,
    farthest_pool_size: int,
    phase: str,
) -> dict[str, Any]:
    index = _support_index(support, workers)
    measures = np.zeros(support.k, dtype=np.float64)
    centroid_sums = np.zeros((support.k, 3), dtype=np.float64)
    normal_sums = np.zeros((support.k, 3), dtype=np.float64)
    cp_sums = np.zeros(support.k, dtype=np.float64)
    wss_sums = np.zeros((support.k, 3), dtype=np.float64)
    farthest_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for chunk_number, chunk in enumerate(_geometry_chunks(case, chunk_cells), start=1):
        assignment, distances, selected_dots = _assign(
            index,
            support,
            chunk.centroids,
            chunk.normals,
        )
        assignment_metric_distances = np.sqrt(
            np.maximum(
                distances.astype(np.float64) ** 2
                + support.normal_length_scale**2
                * 2.0
                * (1.0 - selected_dots.astype(np.float64)),
                0.0,
            )
        )
        measures += np.bincount(
            assignment,
            weights=chunk.areas,
            minlength=support.k,
        )
        for dimension in range(3):
            centroid_sums[:, dimension] += np.bincount(
                assignment,
                weights=chunk.areas * chunk.centroids[:, dimension],
                minlength=support.k,
            )
            normal_sums[:, dimension] += np.bincount(
                assignment,
                weights=chunk.areas * chunk.normals[:, dimension],
                minlength=support.k,
            )
        if include_fields:
            cp_sums += np.bincount(
                assignment,
                weights=chunk.areas * chunk.cp,
                minlength=support.k,
            )
            for dimension in range(3):
                wss_sums[:, dimension] += np.bincount(
                    assignment,
                    weights=chunk.areas * chunk.wss[:, dimension],
                    minlength=support.k,
                )
        if farthest_pool_size > 0:
            local_count = min(
                farthest_pool_size,
                len(assignment_metric_distances),
            )
            local = np.argpartition(
                assignment_metric_distances,
                len(assignment_metric_distances) - local_count,
            )[-local_count:]
            farthest_records.append(
                (
                    assignment_metric_distances[local],
                    chunk.start + local.astype(np.int64),
                    chunk.centroids[local].copy(),
                    chunk.normals[local].copy(),
                )
            )
        if chunk_number % 10 == 0 or chunk.stop == case.n_cells:
            _log(
                f"{case.case_id} {phase} completed_cells="
                f"{chunk.stop:,}/{case.n_cells:,}"
            )

    result: dict[str, Any] = {
        "measures": measures,
        "centroid_sums": centroid_sums,
        "normal_sums": normal_sums,
        "neighbor_backend": index.backend,
    }
    if include_fields:
        result["cp_sums"] = cp_sums
        result["wss_sums"] = wss_sums
    if farthest_records:
        result["farthest"] = _farthest_pool(
            farthest_records,
            farthest_pool_size,
        )
    return result


def _repair_empty_supports(
    support: Support,
    accumulation: dict[str, Any],
) -> tuple[list[int], list[int]]:
    measures = accumulation["measures"]
    empty = np.flatnonzero(measures <= 0.0)
    repaired_indices: list[int] = []
    repaired_master_cells: list[int] = []
    if len(empty):
        farthest = accumulation.get("farthest")
        if farthest is None or len(farthest["indices"]) < len(empty):
            raise ValueError(
                f"{len(empty)} empty cover cells cannot be deterministically repaired"
            )
        used_master_cells: set[int] = set()
        for support_index, pool_index in zip(empty, range(len(empty))):
            master_index = int(farthest["indices"][pool_index])
            if master_index in used_master_cells:
                raise ValueError("Farthest-point repair pool contains duplicates")
            used_master_cells.add(master_index)
            support.points[support_index] = farthest["points"][pool_index]
            support.normals[support_index] = farthest["normals"][pool_index]
            repaired_indices.append(int(support_index))
            repaired_master_cells.append(master_index)
    return repaired_indices, repaired_master_cells


def _update_cover(
    support: Support,
    accumulation: dict[str, Any],
) -> dict[str, Any]:
    measures = accumulation["measures"]
    nonempty = measures > 0.0
    old_points = support.points.copy()
    support.points[nonempty] = (
        accumulation["centroid_sums"][nonempty] / measures[nonempty, None]
    )
    updated_normal = accumulation["normal_sums"][nonempty]
    updated_norm = np.linalg.norm(updated_normal, axis=1)
    valid_normal = updated_norm > 0.0
    nonempty_indices = np.flatnonzero(nonempty)
    support.normals[nonempty_indices[valid_normal]] = (
        updated_normal[valid_normal] / updated_norm[valid_normal, None]
    )
    repaired_indices, repaired_master_cells = _repair_empty_supports(
        support,
        accumulation,
    )

    shifts = np.linalg.norm(support.points - old_points, axis=1)
    return {
        "empty_before_repair": int(np.count_nonzero(~nonempty)),
        "repaired_support_indices": repaired_indices,
        "repair_master_cells": repaired_master_cells,
        "shift": _distribution(shifts),
    }


def _centroidal_cover(
    case: SurfaceCase,
    support: Support,
    *,
    iterations: int,
    chunk_cells: int,
    workers: int,
    repair_pool_size: int,
) -> list[dict[str, Any]]:
    history = []
    for iteration in range(iterations):
        accumulation = _accumulate_assignment(
            case,
            support,
            chunk_cells=chunk_cells,
            workers=workers,
            include_fields=False,
            farthest_pool_size=repair_pool_size,
            phase=f"centroidal_iteration_{iteration + 1}",
        )
        update = _update_cover(support, accumulation)
        update.update(
            {
                "iteration": iteration + 1,
                "represented_measure": _distribution(accumulation["measures"]),
                "neighbor_backend": accumulation["neighbor_backend"],
            }
        )
        history.append(update)
    return history


def _restrict_fields(
    case: SurfaceCase,
    support: Support,
    *,
    chunk_cells: int,
    workers: int,
    repair_pool_size: int,
    allow_repair: bool,
) -> tuple[RestrictedFields, list[dict[str, Any]], str]:
    repair_history = []
    for attempt in range(MAX_RESTRICTION_REPAIRS + 1):
        accumulation = _accumulate_assignment(
            case,
            support,
            chunk_cells=chunk_cells,
            workers=workers,
            include_fields=True,
            farthest_pool_size=repair_pool_size if allow_repair else 0,
            phase=f"{support.name}_restriction_attempt_{attempt + 1}",
        )
        measures = accumulation["measures"]
        empty = np.flatnonzero(measures <= 0.0)
        if not len(empty):
            cp = accumulation["cp_sums"] / measures
            wss = accumulation["wss_sums"] / measures[:, None]
            return (
                RestrictedFields(measures=measures, cp=cp, wss=wss),
                repair_history,
                accumulation["neighbor_backend"],
            )
        if not allow_repair:
            raise ValueError(f"{support.name} has {len(empty)} empty represented cells")
        if attempt == MAX_RESTRICTION_REPAIRS:
            break
        old_points = support.points.copy()
        repaired_indices, repaired_master_cells = _repair_empty_supports(
            support,
            accumulation,
        )
        shifts = np.linalg.norm(support.points - old_points, axis=1)
        repair_history.append(
            {
                "empty_before_repair": int(len(empty)),
                "repaired_support_indices": repaired_indices,
                "repair_master_cells": repaired_master_cells,
                "shift": _distribution(shifts),
                "restriction_attempt": attempt + 1,
                "repair_scope": "empty_supports_only",
            }
        )
    raise ValueError(
        f"{support.name} still has empty represented cells after "
        f"{MAX_RESTRICTION_REPAIRS} repairs"
    )


def _fill_metrics(
    distances: np.ndarray,
    areas: np.ndarray,
    bbox_diagonal: float,
) -> dict[str, Any]:
    uniform = _distribution(distances)
    weighted = _weighted_quantiles(distances, areas)
    normalized_uniform = {
        key: _finite(value / bbox_diagonal) for key, value in uniform.items()
    }
    normalized_weighted = {
        key: _finite(value / bbox_diagonal) for key, value in weighted.items()
    }
    return {
        "definition": (
            "exact ambient-Euclidean distance from every full-master cell "
            "centroid to its assigned representation centroid"
        ),
        "cell_uniform": uniform,
        "area_weighted_quantiles": weighted,
        "normalized_by_master_bbox_diagonal": {
            "cell_uniform": normalized_uniform,
            "area_weighted_quantiles": normalized_weighted,
        },
    }


def _support_probe(support: Support, phase: float) -> np.ndarray:
    points = support.points.astype(np.float64)
    normals = support.normals.astype(np.float64)
    return np.stack(
        (
            1.0 + 0.17 * points[:, 0] - 0.11 * points[:, 2],
            np.sin(phase + 0.31 * points[:, 0] + 0.23 * points[:, 1]),
            0.4 * normals[:, 0] - 0.2 * normals[:, 1] + 0.7 * normals[:, 2],
        ),
        axis=1,
    )


def _projection_result(
    *,
    cp_error: float,
    wss_error: float,
    tangent_wss_error: float,
    cp_truth_norm: float,
    wss_truth_norm: float,
    truth_wss_normal_energy: float,
    raw_projected_wss_normal_energy: float,
    tangent_projected_wss_normal_energy: float,
    total_area: float,
    truth_cp_integral: float,
    projected_cp_integral: float,
    truth_wss_integral: np.ndarray,
    projected_wss_integral: np.ndarray,
    truth_pressure_force: np.ndarray,
    projected_pressure_force: np.ndarray,
    truth_pressure_moment: np.ndarray,
    projected_pressure_moment: np.ndarray,
    cp_residual_sums: np.ndarray,
    wss_residual_sums: np.ndarray,
) -> dict[str, Any]:
    cp_scale = max(
        math.sqrt(cp_truth_norm),
        np.finfo(np.float64).tiny,
    )
    wss_scale = max(
        math.sqrt(wss_truth_norm),
        np.finfo(np.float64).tiny,
    )
    return {
        "target_definition": (
            "area-adjoint restriction of full-master truth followed by "
            "piecewise-constant prolongation"
        ),
        "CpMeanTrim": {
            "relative_l2_floor": _finite(math.sqrt(cp_error) / cp_scale),
            "area_weighted_rms_floor": _finite(math.sqrt(cp_error / total_area)),
            "integral_relative_error": _finite(
                abs(projected_cp_integral - truth_cp_integral)
                / max(abs(truth_cp_integral), cp_scale, 1.0)
            ),
            "restriction_orthogonality_max_abs": _finite(
                np.max(np.abs(cp_residual_sums))
            ),
            "pressure_force": {
                "definition": "integral master_area * Cp * master_unit_normal",
                "truth": [_finite(value) for value in truth_pressure_force],
                "projected": [_finite(value) for value in projected_pressure_force],
                "relative_error": _relative_vector_error(
                    projected_pressure_force,
                    truth_pressure_force,
                ),
            },
            "pressure_moment": {
                "definition": (
                    "integral master_area * cross(master_centroid, "
                    "Cp * master_unit_normal)"
                ),
                "origin": [0.0, 0.0, 0.0],
                "truth": [_finite(value) for value in truth_pressure_moment],
                "projected": [_finite(value) for value in projected_pressure_moment],
                "relative_error": _relative_vector_error(
                    projected_pressure_moment,
                    truth_pressure_moment,
                ),
            },
        },
        "wallShearStressMeanTrim": {
            "raw_p0_relative_l2_floor": _finite(math.sqrt(wss_error) / wss_scale),
            "raw_p0_area_weighted_vector_rms_floor": _finite(
                math.sqrt(wss_error / total_area)
            ),
            "tangent_projected_relative_l2_floor": _finite(
                math.sqrt(tangent_wss_error) / wss_scale
            ),
            "tangent_projected_area_weighted_vector_rms_floor": _finite(
                math.sqrt(tangent_wss_error / total_area)
            ),
            "integral_relative_error": _relative_vector_error(
                projected_wss_integral,
                truth_wss_integral,
            ),
            "restriction_orthogonality_max_vector_norm": _finite(
                np.linalg.norm(wss_residual_sums, axis=1).max()
            ),
            "tangency_relative_normal_l2": {
                "full_master_truth": _finite(
                    math.sqrt(truth_wss_normal_energy) / wss_scale
                ),
                "raw_p0_prolongation": _finite(
                    math.sqrt(raw_projected_wss_normal_energy) / wss_scale
                ),
                "after_explicit_master_tangent_projection": _finite(
                    math.sqrt(tangent_projected_wss_normal_energy) / wss_scale
                ),
            },
        },
    }


def _final_diagnostics(
    case: SurfaceCase,
    supports: tuple[Support, Support],
    restricted: tuple[RestrictedFields, RestrictedFields],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    chunk_cells: int,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support_a, support_b = supports
    fields_a, fields_b = restricted
    index_a = _support_index(support_a, workers)
    index_b = _support_index(support_b, workers)
    bbox_diagonal = float(np.linalg.norm(upper - lower))
    if bbox_diagonal <= 0.0:
        raise ValueError("Master-surface bounding box is degenerate")

    distances_a = np.empty(case.n_cells, dtype=np.float32)
    distances_b = np.empty(case.n_cells, dtype=np.float32)
    metric_distances_a = np.empty(case.n_cells, dtype=np.float32)
    metric_distances_b = np.empty(case.n_cells, dtype=np.float32)
    area_buffer = np.empty(case.n_cells, dtype=np.float64)
    diag_mass_a = np.zeros(support_a.k, dtype=np.float64)
    diag_mass_b = np.zeros(support_b.k, dtype=np.float64)
    negative_area = np.zeros(2, dtype=np.float64)
    negative_cells = np.zeros(2, dtype=np.int64)
    minimum_dot = np.full(2, np.inf, dtype=np.float64)
    cp_error = np.zeros(2, dtype=np.float64)
    wss_error = np.zeros(2, dtype=np.float64)
    tangent_wss_error = np.zeros(2, dtype=np.float64)
    raw_projected_wss_normal_energy = np.zeros(2, dtype=np.float64)
    tangent_projected_wss_normal_energy = np.zeros(2, dtype=np.float64)
    cp_residual = [
        np.zeros(support_a.k, dtype=np.float64),
        np.zeros(support_b.k, dtype=np.float64),
    ]
    wss_residual = [
        np.zeros((support_a.k, 3), dtype=np.float64),
        np.zeros((support_b.k, 3), dtype=np.float64),
    ]
    truth_cp_norm = 0.0
    truth_wss_norm = 0.0
    truth_wss_normal_energy = 0.0
    truth_cp_integral = 0.0
    truth_wss_integral = np.zeros(3, dtype=np.float64)
    projected_cp_integral = np.zeros(2, dtype=np.float64)
    projected_wss_integral = np.zeros((2, 3), dtype=np.float64)
    truth_pressure_force = np.zeros(3, dtype=np.float64)
    truth_pressure_moment = np.zeros(3, dtype=np.float64)
    projected_pressure_force = np.zeros((2, 3), dtype=np.float64)
    projected_pressure_moment = np.zeros((2, 3), dtype=np.float64)
    total_area = 0.0
    degenerate_count = 0

    probe_a = _support_probe(support_a, 0.37)
    probe_b = _support_probe(support_b, 1.13)
    transfer_a_to_b_numerator = np.zeros_like(probe_b)
    transfer_b_to_a_numerator = np.zeros_like(probe_a)
    roundtrip_a_numerator = np.zeros_like(probe_a)
    roundtrip_b_numerator = np.zeros_like(probe_b)
    constant_a_to_b_numerator = np.zeros(support_b.k, dtype=np.float64)
    constant_b_to_a_numerator = np.zeros(support_a.k, dtype=np.float64)
    pythagorean_total_cp = np.zeros(2, dtype=np.float64)
    pythagorean_total_wss = np.zeros(2, dtype=np.float64)
    operator_digest = hashlib.sha256()
    assignment_digests = [hashlib.sha256(), hashlib.sha256()]

    for chunk_number, chunk in enumerate(_geometry_chunks(case, chunk_cells), start=1):
        assignment_a, distance_a, dot_a = _assign(
            index_a,
            support_a,
            chunk.centroids,
            chunk.normals,
        )
        assignment_b, distance_b, dot_b = _assign(
            index_b,
            support_b,
            chunk.centroids,
            chunk.normals,
        )
        assignments = (assignment_a, assignment_b)
        distances = (distance_a, distance_b)
        dots = (dot_a, dot_b)
        representation_fields = (fields_a, fields_b)

        area_buffer[chunk.start : chunk.stop] = chunk.areas
        distances_a[chunk.start : chunk.stop] = distance_a
        distances_b[chunk.start : chunk.stop] = distance_b
        metric_distances_a[chunk.start : chunk.stop] = np.sqrt(
            np.maximum(
                distance_a.astype(np.float64) ** 2
                + support_a.normal_length_scale**2
                * 2.0
                * (1.0 - dot_a.astype(np.float64)),
                0.0,
            )
        )
        metric_distances_b[chunk.start : chunk.stop] = np.sqrt(
            np.maximum(
                distance_b.astype(np.float64) ** 2
                + support_b.normal_length_scale**2
                * 2.0
                * (1.0 - dot_b.astype(np.float64)),
                0.0,
            )
        )
        total_area += chunk.areas.sum(dtype=np.float64)
        degenerate_count += int(np.count_nonzero(chunk.areas <= 0.0))
        truth_cp_norm += np.einsum(
            "i,i,i->",
            chunk.areas,
            chunk.cp,
            chunk.cp,
            dtype=np.float64,
            optimize=True,
        )
        truth_wss_norm += np.einsum(
            "i,ij,ij->",
            chunk.areas,
            chunk.wss,
            chunk.wss,
            dtype=np.float64,
            optimize=True,
        )
        truth_wss_normal = np.einsum(
            "ij,ij->i",
            chunk.wss,
            chunk.normals,
            dtype=np.float64,
            optimize=True,
        )
        truth_wss_normal_energy += np.einsum(
            "i,i,i->",
            chunk.areas,
            truth_wss_normal,
            truth_wss_normal,
            dtype=np.float64,
            optimize=True,
        )
        truth_cp_integral += np.einsum(
            "i,i->",
            chunk.areas,
            chunk.cp,
            dtype=np.float64,
            optimize=True,
        )
        truth_wss_integral += np.einsum(
            "i,ij->j",
            chunk.areas,
            chunk.wss,
            dtype=np.float64,
            optimize=True,
        )
        truth_pressure_traction = chunk.cp[:, None] * chunk.normals
        truth_pressure_force += np.einsum(
            "i,ij->j",
            chunk.areas,
            truth_pressure_traction,
            dtype=np.float64,
            optimize=True,
        )
        truth_pressure_moment += np.einsum(
            "i,ij->j",
            chunk.areas,
            np.cross(chunk.centroids, truth_pressure_traction),
            dtype=np.float64,
            optimize=True,
        )

        for ordinal, (assignment, distance, dot, representation) in enumerate(
            zip(
                assignments,
                distances,
                dots,
                representation_fields,
            )
        ):
            mass = np.bincount(
                assignment,
                weights=chunk.areas,
                minlength=len(representation.measures),
            )
            if ordinal == 0:
                diag_mass_a += mass
            else:
                diag_mass_b += mass
            negative = dot < 0.0
            negative_area[ordinal] += chunk.areas[negative].sum(dtype=np.float64)
            negative_cells[ordinal] += np.count_nonzero(negative)
            minimum_dot[ordinal] = min(minimum_dot[ordinal], float(dot.min()))

            projected_cp = representation.cp[assignment]
            projected_wss = representation.wss[assignment]
            raw_projected_wss_normal = np.einsum(
                "ij,ij->i",
                projected_wss,
                chunk.normals,
                dtype=np.float64,
                optimize=True,
            )
            tangent_projected_wss = (
                projected_wss - raw_projected_wss_normal[:, None] * chunk.normals
            )
            tangent_projected_wss_normal = np.einsum(
                "ij,ij->i",
                tangent_projected_wss,
                chunk.normals,
                dtype=np.float64,
                optimize=True,
            )
            cp_delta = chunk.cp - projected_cp
            wss_delta = chunk.wss - projected_wss
            tangent_wss_delta = chunk.wss - tangent_projected_wss
            cp_error[ordinal] += np.einsum(
                "i,i,i->",
                chunk.areas,
                cp_delta,
                cp_delta,
                dtype=np.float64,
                optimize=True,
            )
            wss_error[ordinal] += np.einsum(
                "i,ij,ij->",
                chunk.areas,
                wss_delta,
                wss_delta,
                dtype=np.float64,
                optimize=True,
            )
            tangent_wss_error[ordinal] += np.einsum(
                "i,ij,ij->",
                chunk.areas,
                tangent_wss_delta,
                tangent_wss_delta,
                dtype=np.float64,
                optimize=True,
            )
            raw_projected_wss_normal_energy[ordinal] += np.einsum(
                "i,i,i->",
                chunk.areas,
                raw_projected_wss_normal,
                raw_projected_wss_normal,
                dtype=np.float64,
                optimize=True,
            )
            tangent_projected_wss_normal_energy[ordinal] += np.einsum(
                "i,i,i->",
                chunk.areas,
                tangent_projected_wss_normal,
                tangent_projected_wss_normal,
                dtype=np.float64,
                optimize=True,
            )
            cp_residual[ordinal] += np.bincount(
                assignment,
                weights=chunk.areas * cp_delta,
                minlength=len(representation.measures),
            )
            for dimension in range(3):
                wss_residual[ordinal][:, dimension] += np.bincount(
                    assignment,
                    weights=chunk.areas * wss_delta[:, dimension],
                    minlength=len(representation.measures),
                )
            projected_cp_integral[ordinal] += np.einsum(
                "i,i->",
                chunk.areas,
                projected_cp,
                dtype=np.float64,
                optimize=True,
            )
            projected_wss_integral[ordinal] += np.einsum(
                "i,ij->j",
                chunk.areas,
                projected_wss,
                dtype=np.float64,
                optimize=True,
            )
            projected_pressure_traction = projected_cp[:, None] * chunk.normals
            projected_pressure_force[ordinal] += np.einsum(
                "i,ij->j",
                chunk.areas,
                projected_pressure_traction,
                dtype=np.float64,
                optimize=True,
            )
            projected_pressure_moment[ordinal] += np.einsum(
                "i,ij->j",
                chunk.areas,
                np.cross(chunk.centroids, projected_pressure_traction),
                dtype=np.float64,
                optimize=True,
            )
            prediction = (probe_a, probe_b)[ordinal]
            prediction_cp_delta = prediction[assignment, 0] - chunk.cp
            prediction_wss_delta = prediction[assignment] - chunk.wss
            pythagorean_total_cp[ordinal] += np.einsum(
                "i,i,i->",
                chunk.areas,
                prediction_cp_delta,
                prediction_cp_delta,
                dtype=np.float64,
                optimize=True,
            )
            pythagorean_total_wss[ordinal] += np.einsum(
                "i,ij,ij->",
                chunk.areas,
                prediction_wss_delta,
                prediction_wss_delta,
                dtype=np.float64,
                optimize=True,
            )
            assignment_digests[ordinal].update(
                memoryview(np.ascontiguousarray(assignment)).cast("B")
            )

        for dimension in range(3):
            transfer_a_to_b_numerator[:, dimension] += np.bincount(
                assignment_b,
                weights=chunk.areas * probe_a[assignment_a, dimension],
                minlength=support_b.k,
            )
            transfer_b_to_a_numerator[:, dimension] += np.bincount(
                assignment_a,
                weights=chunk.areas * probe_b[assignment_b, dimension],
                minlength=support_a.k,
            )
            roundtrip_a_numerator[:, dimension] += np.bincount(
                assignment_a,
                weights=chunk.areas * probe_a[assignment_a, dimension],
                minlength=support_a.k,
            )
            roundtrip_b_numerator[:, dimension] += np.bincount(
                assignment_b,
                weights=chunk.areas * probe_b[assignment_b, dimension],
                minlength=support_b.k,
            )
        constant_a_to_b_numerator += np.bincount(
            assignment_b,
            weights=chunk.areas,
            minlength=support_b.k,
        )
        constant_b_to_a_numerator += np.bincount(
            assignment_a,
            weights=chunk.areas,
            minlength=support_a.k,
        )
        operator_digest.update(memoryview(np.ascontiguousarray(assignment_a)).cast("B"))
        operator_digest.update(memoryview(np.ascontiguousarray(assignment_b)).cast("B"))
        operator_digest.update(memoryview(np.ascontiguousarray(chunk.areas)).cast("B"))
        if chunk_number % 10 == 0 or chunk.stop == case.n_cells:
            _log(
                f"{case.case_id} final_diagnostics completed_cells="
                f"{chunk.stop:,}/{case.n_cells:,}"
            )

    if degenerate_count:
        raise ValueError(f"Master mesh has {degenerate_count} degenerate triangles")
    if np.any(diag_mass_a <= 0.0) or np.any(diag_mass_b <= 0.0):
        raise ValueError("Final common-master maps contain empty representation cells")

    transfer_a_to_b = transfer_a_to_b_numerator / diag_mass_b[:, None]
    transfer_b_to_a = transfer_b_to_a_numerator / diag_mass_a[:, None]
    integral_a = np.einsum("i,ij->j", diag_mass_a, probe_a)
    integral_a_after = np.einsum("i,ij->j", diag_mass_b, transfer_a_to_b)
    integral_b = np.einsum("i,ij->j", diag_mass_b, probe_b)
    integral_b_after = np.einsum("i,ij->j", diag_mass_a, transfer_b_to_a)
    adjoint_left = np.einsum(
        "i,ij,ij->",
        diag_mass_b,
        transfer_a_to_b,
        probe_b,
    )
    adjoint_right = np.einsum(
        "i,ij,ij->",
        diag_mass_a,
        probe_a,
        transfer_b_to_a,
    )
    roundtrip_a = roundtrip_a_numerator / diag_mass_a[:, None]
    roundtrip_b = roundtrip_b_numerator / diag_mass_b[:, None]
    roundtrip_error = max(
        float(np.max(np.abs(roundtrip_a - probe_a))),
        float(np.max(np.abs(roundtrip_b - probe_b))),
    )
    constant_error = max(
        float(np.max(np.abs(constant_a_to_b_numerator / diag_mass_b - 1.0))),
        float(np.max(np.abs(constant_b_to_a_numerator / diag_mass_a - 1.0))),
    )
    represented_cp_error = np.array(
        [
            np.einsum(
                "i,i,i->",
                diag_mass_a,
                probe_a[:, 0] - fields_a.cp,
                probe_a[:, 0] - fields_a.cp,
            ),
            np.einsum(
                "i,i,i->",
                diag_mass_b,
                probe_b[:, 0] - fields_b.cp,
                probe_b[:, 0] - fields_b.cp,
            ),
        ],
        dtype=np.float64,
    )
    represented_wss_error = np.array(
        [
            np.einsum(
                "i,ij,ij->",
                diag_mass_a,
                probe_a - fields_a.wss,
                probe_a - fields_a.wss,
            ),
            np.einsum(
                "i,ij,ij->",
                diag_mass_b,
                probe_b - fields_b.wss,
                probe_b - fields_b.wss,
            ),
        ],
        dtype=np.float64,
    )
    pythagorean_error = max(
        float(
            np.max(
                np.abs(pythagorean_total_cp - represented_cp_error - cp_error)
                / np.maximum(pythagorean_total_cp, 1.0)
            )
        ),
        float(
            np.max(
                np.abs(pythagorean_total_wss - represented_wss_error - wss_error)
                / np.maximum(pythagorean_total_wss, 1.0)
            )
        ),
    )

    measure_error_a = np.max(np.abs(diag_mass_a - fields_a.measures)) / max(
        total_area, 1.0
    )
    measure_error_b = np.max(np.abs(diag_mass_b - fields_b.measures)) / max(
        total_area, 1.0
    )
    algebra = {
        "definition": (
            "M_ij = sum_f area_f 1[A(f)=i,B(f)=j] on this case's full CFD "
            "master; T_A_to_B = D_B^-1 M^T and T_B_to_A = D_A^-1 M"
        ),
        "operator_sha256_streaming_assignment_a_b_area_float64": (
            operator_digest.hexdigest()
        ),
        "measure_total_relative_error": _finite(
            max(
                abs(diag_mass_a.sum() - total_area),
                abs(diag_mass_b.sum() - total_area),
            )
            / total_area
        ),
        "restriction_measure_replay_max_relative_error": _finite(
            max(measure_error_a, measure_error_b)
        ),
        "constant_max_abs_error": _finite(constant_error),
        "representation_roundtrip_max_abs_error": _finite(roundtrip_error),
        "pythagorean_relative_error": _finite(pythagorean_error),
        "a_to_b_vector_integral_relative_error": _relative_vector_error(
            integral_a_after,
            integral_a,
        ),
        "b_to_a_vector_integral_relative_error": _relative_vector_error(
            integral_b_after,
            integral_b,
        ),
        "mass_adjoint_relative_error": _finite(
            abs(adjoint_left - adjoint_right)
            / max(abs(adjoint_left), abs(adjoint_right), 1.0)
        ),
    }
    algebra["max_load_bearing_error"] = max(
        algebra["measure_total_relative_error"],
        algebra["restriction_measure_replay_max_relative_error"],
        algebra["constant_max_abs_error"],
        algebra["representation_roundtrip_max_abs_error"],
        algebra["pythagorean_relative_error"],
        algebra["a_to_b_vector_integral_relative_error"],
        algebra["b_to_a_vector_integral_relative_error"],
        algebra["mass_adjoint_relative_error"],
    )
    algebra["tolerance"] = ALGEBRA_TOLERANCE
    algebra["passed"] = algebra["max_load_bearing_error"] <= ALGEBRA_TOLERANCE
    if not algebra["passed"]:
        raise AssertionError(
            "Common-master algebra gate failed: "
            f"{algebra['max_load_bearing_error']:.3e} > "
            f"{ALGEBRA_TOLERANCE:.3e}"
        )

    representation_results = {}
    for ordinal, (
        support,
        representation,
        distances,
        metric_distances,
        diag_mass,
    ) in enumerate(
        zip(
            supports,
            restricted,
            (distances_a, distances_b),
            (metric_distances_a, metric_distances_b),
            (diag_mass_a, diag_mass_b),
        )
    ):
        representation_results[support.name] = {
            "k": support.k,
            "definition": support.definition,
            "support_points_sha256_float32": _sha256_array(support.points),
            "support_normals_sha256_float32": _sha256_array(support.normals),
            "master_assignment_sha256_int64": assignment_digests[ordinal].hexdigest(),
            "empty_representation_cell_count": int(np.count_nonzero(diag_mass <= 0.0)),
            "represented_measure": _distribution(diag_mass),
            "represented_measure_imbalance": {
                "coefficient_of_variation": _finite(
                    diag_mass.std(dtype=np.float64) / diag_mass.mean(dtype=np.float64)
                ),
                "max_over_min": _finite(diag_mass.max() / diag_mass.min()),
            },
            "fill_distance": _fill_metrics(
                distances,
                area_buffer,
                bbox_diagonal,
            ),
            "assignment_metric_distance": {
                "definition": (
                    "sqrt(||x_i-s_j||^2 + lambda^2 ||n_i-m_j||^2); "
                    "lambda=0 for the ambient cyclic arm"
                ),
                "normal_coordinate_length_scale": (support.normal_length_scale),
                "cell_uniform": _distribution(metric_distances),
                "area_weighted_quantiles": _weighted_quantiles(
                    metric_distances,
                    area_buffer,
                ),
            },
            "normal_assignment": {
                "minimum_selected_dot": _finite(minimum_dot[ordinal]),
                "negative_normal_assignment_cell_count": int(negative_cells[ordinal]),
                "negative_normal_assignment_cell_fraction": _finite(
                    negative_cells[ordinal] / case.n_cells
                ),
                "negative_normal_assignment_area": _finite(negative_area[ordinal]),
                "negative_normal_assignment_area_fraction": _finite(
                    negative_area[ordinal] / total_area
                ),
            },
            "p0_projection_floor": _projection_result(
                cp_error=cp_error[ordinal],
                wss_error=wss_error[ordinal],
                tangent_wss_error=tangent_wss_error[ordinal],
                cp_truth_norm=truth_cp_norm,
                wss_truth_norm=truth_wss_norm,
                truth_wss_normal_energy=truth_wss_normal_energy,
                raw_projected_wss_normal_energy=(
                    raw_projected_wss_normal_energy[ordinal]
                ),
                tangent_projected_wss_normal_energy=(
                    tangent_projected_wss_normal_energy[ordinal]
                ),
                total_area=total_area,
                truth_cp_integral=truth_cp_integral,
                projected_cp_integral=projected_cp_integral[ordinal],
                truth_wss_integral=truth_wss_integral,
                projected_wss_integral=projected_wss_integral[ordinal],
                truth_pressure_force=truth_pressure_force,
                projected_pressure_force=projected_pressure_force[ordinal],
                truth_pressure_moment=truth_pressure_moment,
                projected_pressure_moment=projected_pressure_moment[ordinal],
                cp_residual_sums=cp_residual[ordinal],
                wss_residual_sums=wss_residual[ordinal],
            ),
        }

    cyclic = representation_results["cyclic_sparse"]
    cover = representation_results["normal_aware_centroidal_cover"]
    comparison = {
        "ratio_definition": "normal_aware_centroidal_cover / cyclic_sparse",
        "CpMeanTrim_relative_l2_floor_ratio": _finite(
            cover["p0_projection_floor"]["CpMeanTrim"]["relative_l2_floor"]
            / max(
                cyclic["p0_projection_floor"]["CpMeanTrim"]["relative_l2_floor"],
                np.finfo(np.float64).tiny,
            )
        ),
        "wallShearStressMeanTrim_relative_l2_floor_ratio": _finite(
            cover["p0_projection_floor"]["wallShearStressMeanTrim"][
                "raw_p0_relative_l2_floor"
            ]
            / max(
                cyclic["p0_projection_floor"]["wallShearStressMeanTrim"][
                    "raw_p0_relative_l2_floor"
                ],
                np.finfo(np.float64).tiny,
            )
        ),
        "normalized_area_weighted_fill_q95_ratio": _finite(
            cover["fill_distance"]["normalized_by_master_bbox_diagonal"][
                "area_weighted_quantiles"
            ]["q95"]
            / max(
                cyclic["fill_distance"]["normalized_by_master_bbox_diagonal"][
                    "area_weighted_quantiles"
                ]["q95"],
                np.finfo(np.float64).tiny,
            )
        ),
        "normalized_cell_uniform_fill_q95_ratio_diagnostic": _finite(
            cover["fill_distance"]["normalized_by_master_bbox_diagonal"][
                "cell_uniform"
            ]["q95"]
            / max(
                cyclic["fill_distance"]["normalized_by_master_bbox_diagonal"][
                    "cell_uniform"
                ]["q95"],
                np.finfo(np.float64).tiny,
            )
        ),
        "normalized_fill_max_ratio": _finite(
            cover["fill_distance"]["normalized_by_master_bbox_diagonal"][
                "cell_uniform"
            ]["max"]
            / max(
                cyclic["fill_distance"]["normalized_by_master_bbox_diagonal"][
                    "cell_uniform"
                ]["max"],
                np.finfo(np.float64).tiny,
            )
        ),
    }
    master = {
        "n_cells": case.n_cells,
        "geometric_area": _finite(total_area),
        "bbox_min": [_finite(value) for value in lower],
        "bbox_max": [_finite(value) for value in upper],
        "bbox_diagonal": _finite(bbox_diagonal),
        "degenerate_triangle_count": degenerate_count,
        "CpMeanTrim_area_weighted_squared_norm": _finite(truth_cp_norm),
        "wallShearStressMeanTrim_area_weighted_squared_norm": _finite(truth_wss_norm),
    }
    return {
        "full_master": master,
        "representations": representation_results,
        "common_master_algebra": algebra,
        "comparison": comparison,
    }, {
        "total_area": total_area,
        "neighbor_backends": [index_a.backend, index_b.backend],
    }


def _audit_case(
    case: SurfaceCase,
    args: argparse.Namespace,
    cyclic_replicates: list[tuple[int, int]],
) -> dict[str, Any]:
    started = time.perf_counter()
    k = args.synthetic_k if args.synthetic_smoke else args.k
    if not 0 < k <= case.n_cells:
        raise ValueError(f"Expected 0 < k <= {case.n_cells}, got {k}")
    lower, upper = _point_bounds(case.points, args.point_chunk)
    master_area, degenerate_count = _master_area(
        case,
        args.geometry_chunk_cells,
    )
    if degenerate_count:
        raise ValueError(
            f"{case.case_id} master has {degenerate_count} degenerate triangles"
        )
    bbox_diagonal = float(np.linalg.norm(upper - lower))
    normal_length_scale = math.sqrt(master_area / k)
    _log(
        f"{case.case_id} start n_cells={case.n_cells:,} k={k:,} "
        f"bbox_diagonal={bbox_diagonal:.6g} "
        f"normal_lambda={normal_length_scale:.6g}"
    )

    cover, _ = _initial_cover(
        case,
        k=k,
        candidate_multiplier=args.candidate_multiplier,
        lower=lower,
        upper=upper,
        normal_length_scale=normal_length_scale,
    )
    cover_history = _centroidal_cover(
        case,
        cover,
        iterations=args.lloyd_iterations,
        chunk_cells=args.geometry_chunk_cells,
        workers=args.workers,
        repair_pool_size=args.repair_pool_size,
    )
    cover.definition["centroidal_updates"] = (
        "area-weighted master-cell centroid and unit area-vector normal"
    )
    cover.definition["lloyd_iterations"] = args.lloyd_iterations
    cover.definition["lloyd_history"] = cover_history

    cover_fields, cover_repairs, cover_backend = _restrict_fields(
        case,
        cover,
        chunk_cells=args.geometry_chunk_cells,
        workers=args.workers,
        repair_pool_size=args.repair_pool_size,
        allow_repair=True,
    )
    cover.definition["restriction_empty_cell_repairs"] = cover_repairs

    replicate_results = []
    full_master: dict[str, Any] | None = None
    for ordinal, (cyclic_start, cyclic_seed) in enumerate(cyclic_replicates):
        replicate_started = time.perf_counter()
        cyclic = _cyclic_support(
            case,
            k,
            cyclic_start,
            cyclic_seed,
            ordinal,
        )
        cyclic_fields, cyclic_repairs, cyclic_backend = _restrict_fields(
            case,
            cyclic,
            chunk_cells=args.geometry_chunk_cells,
            workers=args.workers,
            repair_pool_size=0,
            allow_repair=False,
        )
        if cyclic_repairs:
            raise AssertionError("Fixed cyclic support was unexpectedly mutated")
        diagnostics, runtime_details = _final_diagnostics(
            case,
            (cyclic, cover),
            (cyclic_fields, cover_fields),
            lower=lower,
            upper=upper,
            chunk_cells=args.geometry_chunk_cells,
            workers=args.workers,
        )
        observed_master = diagnostics.pop("full_master")
        if full_master is None:
            full_master = observed_master
        elif observed_master != full_master:
            raise AssertionError("Full-master diagnostics changed across replicates")
        replicate_results.append(
            {
                "ordinal": ordinal,
                "frozen_cyclic_start": cyclic_start,
                "frozen_seed": cyclic_seed,
                **diagnostics,
                "execution": {
                    "cyclic_restriction_neighbor_backend": cyclic_backend,
                    "final_neighbor_backends": runtime_details["neighbor_backends"],
                    "runtime_seconds": _finite(time.perf_counter() - replicate_started),
                },
            }
        )
    if full_master is None:
        raise ValueError(f"{case.case_id} has no frozen cyclic replicates")
    replay_error = abs(full_master["geometric_area"] - master_area) / master_area
    if replay_error > ALGEBRA_TOLERANCE:
        raise AssertionError(
            f"Master area changed between passes by {replay_error:.3e}"
        )

    comparison_summary = {
        key: _distribution(
            np.asarray(
                [replicate["comparison"][key] for replicate in replicate_results],
                dtype=np.float64,
            )
        )
        for key in replicate_results[0]["comparison"]
        if key != "ratio_definition"
    }
    return {
        "case_id": case.case_id,
        "source": case.source,
        "effective_k": k,
        "full_master": full_master,
        "normal_coordinate_length_scale": normal_length_scale,
        "normal_coordinate_scale_definition": "sqrt(master_area / k)",
        "cover_construction": cover.definition,
        "cyclic_replicates": replicate_results,
        "replicate_comparison_summary": comparison_summary,
        "execution": {
            "cover_restriction_neighbor_backend": cover_backend,
            "runtime_seconds": _finite(time.perf_counter() - started),
        },
    }


def _synthetic_case() -> SurfaceCase:
    n_u = 32
    n_v = 16
    major_radius = 1.7
    minor_radius = 0.55
    u = 2.0 * np.pi * np.arange(n_u) / n_u
    v = 2.0 * np.pi * np.arange(n_v) / n_v
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = (
        np.stack(
            (
                (major_radius + minor_radius * np.cos(vv)) * np.cos(uu),
                (major_radius + minor_radius * np.cos(vv)) * np.sin(uu),
                minor_radius * np.sin(vv),
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )

    def vertex(i: int, j: int) -> int:
        return (i % n_u) * n_v + (j % n_v)

    cells = []
    for i in range(n_u):
        for j in range(n_v):
            p00 = vertex(i, j)
            p10 = vertex(i + 1, j)
            p11 = vertex(i + 1, j + 1)
            p01 = vertex(i, j + 1)
            cells.append((p00, p10, p11))
            cells.append((p00, p11, p01))
    cell_array = np.asarray(cells, dtype=np.int64)
    _, centroids, normals = _triangle_geometry(points[cell_array])
    cp = (
        1.0
        + 0.23 * centroids[:, 0]
        - 0.17 * centroids[:, 1]
        + 0.11 * centroids[:, 2] ** 2
    ).astype(np.float32)
    raw_wss = np.stack(
        (
            0.4 + 0.3 * centroids[:, 1],
            -0.2 + 0.2 * centroids[:, 2],
            0.1 - 0.25 * centroids[:, 0],
        ),
        axis=1,
    )
    wss = (
        raw_wss
        - np.einsum(
            "ij,ij->i",
            raw_wss,
            normals,
        )[:, None]
        * normals
    )
    wss = wss.astype(np.float32)
    return SurfaceCase(
        case_id="synthetic_torus",
        points=points,
        cells=cell_array,
        cp=cp,
        wss=wss,
        source={
            "kind": "deterministic_synthetic_torus",
            "n_points": len(points),
            "n_cells": len(cell_array),
            "parameters": {
                "n_u": n_u,
                "n_v": n_v,
                "major_radius": major_radius,
                "minor_radius": minor_radius,
            },
            "array_sha256": {
                "points": _sha256_array(points),
                "cells": _sha256_array(cell_array),
                "CpMeanTrim": _sha256_array(cp),
                "wallShearStressMeanTrim": _sha256_array(wss),
            },
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("CASE_ID", "PDMSH"),
        help="Case ID and curated domain .pdmsh path; repeat for each case.",
    )
    parser.add_argument(
        "--cyclic-replicate",
        nargs=3,
        action="append",
        metavar=("CASE_ID", "START", "SEED"),
        help=(
            "Frozen case ID, explicit cyclic start, and provenance seed; "
            "repeat to audit multiple replicates."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10_000)
    parser.add_argument("--candidate-multiplier", type=int, default=16)
    parser.add_argument("--lloyd-iterations", type=int, default=2)
    parser.add_argument("--geometry-chunk-cells", type=int, default=250_000)
    parser.add_argument("--point-chunk", type=int, default=1_000_000)
    parser.add_argument("--repair-pool-size", type=int, default=2_048)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument(
        "--hash-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SHA-256 hash all four production input arrays (default: enabled).",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Audit a deterministic small torus instead of reading .pdmsh files.",
    )
    parser.add_argument("--synthetic-k", type=int, default=64)
    args = parser.parse_args()

    if args.synthetic_smoke and args.case:
        parser.error("--synthetic-smoke cannot be combined with --case")
    if args.synthetic_smoke and args.cyclic_replicate:
        parser.error("--synthetic-smoke cannot be combined with --cyclic-replicate")
    if not args.synthetic_smoke and not args.case:
        parser.error("at least one --case is required outside --synthetic-smoke")
    if not args.synthetic_smoke and not args.cyclic_replicate:
        parser.error(
            "at least one explicit --cyclic-replicate is required outside "
            "--synthetic-smoke"
        )
    if args.candidate_multiplier < 1:
        parser.error("--candidate-multiplier must be at least one")
    if args.lloyd_iterations < 1:
        parser.error("--lloyd-iterations must be at least one")
    if args.geometry_chunk_cells < 1 or args.point_chunk < 1:
        parser.error("chunk sizes must be positive")
    if args.repair_pool_size < 1:
        parser.error("--repair-pool-size must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    if args.synthetic_smoke:
        cases = [_synthetic_case()]
        replicates_by_case = {"synthetic_torus": [(0, 0)]}
    else:
        cases = [
            _load_pdmsh_case(
                case_id,
                Path(path),
                hash_inputs=args.hash_inputs,
            )
            for case_id, path in args.case
        ]
        case_ids = [case.case_id for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("--case IDs must be unique")
        replicates_by_case: dict[str, list[tuple[int, int]]] = {
            case_id: [] for case_id in case_ids
        }
        for case_id, start_text, seed_text in args.cyclic_replicate:
            if case_id not in replicates_by_case:
                raise ValueError(f"--cyclic-replicate names unknown case {case_id!r}")
            try:
                start = int(start_text)
                seed = int(seed_text)
            except ValueError as error:
                raise ValueError(
                    "--cyclic-replicate START and SEED must be integers"
                ) from error
            replicates_by_case[case_id].append((start, seed))
        missing = [
            case_id
            for case_id, replicates in replicates_by_case.items()
            if not replicates
        ]
        if missing:
            raise ValueError(
                "Missing explicit --cyclic-replicate entries for " + ", ".join(missing)
            )

    case_results = [
        _audit_case(case, args, replicates_by_case[case.case_id]) for case in cases
    ]
    script_path = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASSED_SYNTHETIC_COMMON_MASTER_SMOKE"
            if args.synthetic_smoke
            else "PASSED_PRODUCTION_COMMON_MASTER_AUDIT"
        ),
        "scope": (
            "Per-case full curated CFD vehicle meshes are frozen integration "
            "masters. No transfer is performed between different cars."
        ),
        "warnings": [
            "The cyclic representation is a sparse scattered triangle support, "
            "not a full-cover remesh and not a literal overlap surface.",
            "The centroidal cover is a deterministic reconstruction design, "
            "not a trained-model result.",
            "P0 projection floors quantify representation error only; they do "
            "not estimate model error.",
            "Ambient centroid distance is not geodesic surface distance.",
        ],
        "design": {
            "requested_k": args.k,
            "synthetic_k": args.synthetic_k if args.synthetic_smoke else None,
            "explicit_cyclic_replicates": (
                {"synthetic_torus": [{"start": 0, "seed": 0}]}
                if args.synthetic_smoke
                else {
                    case_id: [
                        {"start": start, "seed": seed} for start, seed in replicates
                    ]
                    for case_id, replicates in replicates_by_case.items()
                }
            ),
            "candidate_multiplier": args.candidate_multiplier,
            "lloyd_iterations": args.lloyd_iterations,
            "normal_aware_assignment": {
                "squared_cost": ("||x_i-s_j||^2 + lambda^2 ||n_i-m_j||^2"),
                "lambda": "sqrt(per_case_master_area / k)",
            },
            "geometry_chunk_cells": args.geometry_chunk_cells,
            "point_chunk": args.point_chunk,
            "repair_pool_size": args.repair_pool_size,
            "workers": args.workers,
            "hash_inputs": args.hash_inputs,
        },
        "cases": case_results,
        "provenance": {
            "script": {
                "path": str(script_path),
                "sha256": _sha256_file(script_path),
            },
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "scipy": None if scipy is None else scipy.__version__,
            },
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NODELIST",
                    "SLURM_CPUS_PER_TASK",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "runtime_seconds": _finite(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
