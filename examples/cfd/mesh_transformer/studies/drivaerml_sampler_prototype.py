# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Prototype audit of cyclic-block sampling on raw DrivAerML VTP surfaces.

This is deliberately *not* a production-data audit.  The unified external-aero
recipe reads curated, triangulated ``.pdmsh`` files, while the locally available
files are raw quad-dominant VTP surfaces.  This script triangulates those VTPs
with PyVista and audits the resulting order.  The numbers therefore validate
the diagnostic and expose properties of this raw-derived order; they do not
establish properties of the unavailable curated order.

The cyclic samples themselves use the production
``physicsnemo.datapipes._indexing._cyclic_block_indices`` helper.  Seeded
simple-random samples are a diagnostic comparator, not a proxy for production.

Example
-------
Run from the repository root::

    uv run --no-sync python \
      examples/cfd/mesh_transformer/studies/drivaerml_sampler_prototype.py \
      --surface /data/drivaer_data/run_1/boundary_1.vtp \
      --surface /data/drivaer_data/run_2/boundary_2.vtp \
      --output examples/cfd/mesh_transformer/results/\
drivaerml_raw_vtp_sampler_prototype_2026-07-27.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy
import torch
import vtk
from scipy.spatial import cKDTree

from physicsnemo.datapipes._indexing import _cyclic_block_indices

PROTOTYPE_WARNINGS = [
    "PROTOTYPE ONLY: input order is raw VTP order after PyVista triangulation, "
    "not the unavailable curated .pdmsh order used by the unified recipe.",
    "Only the two locally complete raw surfaces are audited; this is not a "
    "population estimate for the 435-case training split.",
    "Fill distance is ambient Euclidean centroid distance on a frozen subset "
    "of full-surface cells, not exact full-surface or geodesic fill distance.",
    "Simple-random sampling is a diagnostic comparator, not a proxy for the "
    "production cyclic-block design.",
]

DEFAULT_K = (2_500, 5_000, 10_000, 20_000, 40_000)
DEFAULT_ROLLING_RMS_K = (2_500, 10_000, 40_000)
DEFAULT_RANDOM_SEEDS = (17, 42, 137, 20_260_727)
DEFAULT_ORDER_LAGS = (1, 2, 4, 8, 16, 64, 256, 1_024, 4_096, 16_384)
QUANTILES = (0.5, 0.9, 0.95, 0.99)
RMS_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


def _git_output(repo_root: Path, *args: str) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    # All callers pass fixed, source-controlled git arguments.
    completed = subprocess.run(  # noqa: S603
        [git_executable, *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _as_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Diagnostic produced a non-finite value: {result}")
    return result


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order].astype(np.float64, copy=False)
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    cumulative /= cumulative[-1]
    return {
        f"q{round(100 * quantile):02d}": _as_float(
            np.interp(quantile, cumulative, sorted_values)
        )
        for quantile in quantiles
    }


def _cyclic_scalar_correlation(values: np.ndarray, lag: int) -> float:
    n = len(values)
    lag %= n
    mean = values.mean(dtype=np.float64)
    second_moment = (
        np.einsum("i,i->", values, values, dtype=np.float64, optimize=True) / n
    )
    variance = second_moment - mean * mean
    if variance <= 0.0:
        return 0.0
    cross = (
        np.einsum("i,i->", values[:-lag], values[lag:], dtype=np.float64, optimize=True)
        + np.einsum(
            "i,i->", values[-lag:], values[:lag], dtype=np.float64, optimize=True
        )
    ) / n
    return _as_float((cross - mean * mean) / variance)


def _cyclic_vector_correlation(values: np.ndarray, lag: int) -> float:
    """Centered vector autocorrelation, normalized to one at lag zero."""
    n = len(values)
    lag %= n
    mean = values.mean(axis=0, dtype=np.float64)
    second_moment = (
        np.einsum("ij,ij->", values, values, dtype=np.float64, optimize=True) / n
    )
    variance = second_moment - float(mean @ mean)
    if variance <= 0.0:
        return 0.0
    cross = (
        np.einsum(
            "ij,ij->",
            values[:-lag],
            values[lag:],
            dtype=np.float64,
            optimize=True,
        )
        + np.einsum(
            "ij,ij->",
            values[-lag:],
            values[:lag],
            dtype=np.float64,
            optimize=True,
        )
    ) / n
    return _as_float((cross - float(mean @ mean)) / variance)


def _cyclic_vector_dot(values: np.ndarray, lag: int) -> float:
    """Mean uncentered dot product, useful for unit surface normals."""
    n = len(values)
    lag %= n
    cross = np.einsum(
        "ij,ij->",
        values[:-lag],
        values[lag:],
        dtype=np.float64,
        optimize=True,
    ) + np.einsum(
        "ij,ij->",
        values[-lag:],
        values[:lag],
        dtype=np.float64,
        optimize=True,
    )
    return _as_float(cross / n)


def _cyclic_window_sums(values: np.ndarray, k: int) -> np.ndarray:
    """All cyclic length-k window sums, one value for every possible start."""
    n = len(values)
    if not 0 < k <= n:
        raise ValueError(f"Expected 0 < k <= {n}, got {k}")
    prefix = np.empty(n + k + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=prefix[1 : n + 1])
    np.cumsum(values[:k], dtype=np.float64, out=prefix[n + 1 :])
    prefix[n + 1 :] += prefix[n]
    result = prefix[k : k + n] - prefix[:n]
    del prefix
    return result


def _weighted_rms_radius(
    areas: np.ndarray, centroids: np.ndarray, indices: np.ndarray | None = None
) -> float:
    if indices is None:
        weights = areas
        points = centroids
    else:
        weights = areas[indices]
        points = centroids[indices]
    total = weights.sum(dtype=np.float64)
    first = np.einsum("i,ij->j", weights, points, dtype=np.float64, optimize=True)
    second = np.einsum(
        "i,ij,ij->", weights, points, points, dtype=np.float64, optimize=True
    )
    squared = second / total - float((first / total) @ (first / total))
    return _as_float(math.sqrt(max(0.0, squared)))


def _self_check() -> dict[str, Any]:
    values = np.array([0.5, 1.5, 3.0, 4.0, 7.5, 9.0, 12.0], dtype=np.float32)
    for k in (1, 3, 6):
        rolling = _cyclic_window_sums(values, k)
        direct = np.array(
            [
                values[_cyclic_block_indices(len(values), k, start=start).numpy()].sum(
                    dtype=np.float64
                )
                for start in range(len(values))
            ]
        )
        np.testing.assert_allclose(rolling, direct, rtol=0.0, atol=0.0)

    areas = np.array([1.0, 2.0, 1.5, 0.75, 1.25], dtype=np.float32)
    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    k = 3
    s0 = _cyclic_window_sums(areas, k)
    s1 = [_cyclic_window_sums(areas * centroids[:, dim], k) for dim in range(3)]
    s2 = _cyclic_window_sums(areas * np.einsum("ij,ij->i", centroids, centroids), k)
    rolling_rms = np.sqrt(
        np.maximum(
            0.0,
            s2 / s0 - sum(np.square(component / s0) for component in s1),
        )
    )
    direct_rms = np.array(
        [
            _weighted_rms_radius(
                areas,
                centroids,
                _cyclic_block_indices(len(areas), k, start=start).numpy(),
            )
            for start in range(len(areas))
        ]
    )
    np.testing.assert_allclose(rolling_rms, direct_rms, rtol=1e-12, atol=1e-12)
    return {
        "status": "passed",
        "checks": [
            "cyclic rolling sums equal production-helper direct sums",
            "rolling RMS radius equals direct production-helper evaluation",
        ],
    }


def _prepare_surface(
    path: Path, geometry_chunk_cells: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _log(f"Reading {path}")
    source_stat = path.stat()
    source_hash = _sha256_file(path)
    surface = pv.read(path)
    if not isinstance(surface, pv.PolyData):
        raise TypeError(f"Expected PolyData in {path}, got {type(surface).__name__}")

    raw_fields = sorted(surface.cell_data.keys())
    required_fields = ("CpMeanTrim", "wallShearStressMeanTrim")
    missing = [name for name in required_fields if name not in surface.cell_data]
    if missing:
        raise KeyError(f"{path} is missing required cell fields: {missing}")

    raw_n_cells = surface.n_cells
    raw_n_points = surface.n_points
    raw_is_all_triangles = surface.is_all_triangles

    # Preserve only the two labels audited below. This bounds the temporary
    # triangulated dataset without changing geometry or cell order.
    for name in list(surface.cell_data.keys()):
        if name not in required_fields:
            del surface.cell_data[name]
    surface.point_data.clear()
    surface.field_data.clear()

    if surface.is_all_triangles:
        triangles = surface
    else:
        _log(f"Triangulating {raw_n_cells:,} raw cells")
        triangles = surface.triangulate(
            pass_verts=False,
            pass_lines=False,
            inplace=False,
            progress_bar=False,
        )
        del surface
        gc.collect()

    faces = np.asarray(triangles.regular_faces)
    points = np.asarray(triangles.points)
    n_cells = len(faces)
    areas = np.empty(n_cells, dtype=np.float32)
    centroids = np.empty((n_cells, 3), dtype=np.float32)
    normals = np.empty((n_cells, 3), dtype=np.float32)

    _log(f"Computing geometry for {n_cells:,} triangles in chunks")
    for start in range(0, n_cells, geometry_chunk_cells):
        stop = min(start + geometry_chunk_cells, n_cells)
        vertices = points[faces[start:stop]]
        edge_1 = vertices[:, 1] - vertices[:, 0]
        edge_2 = vertices[:, 2] - vertices[:, 0]
        cross = np.cross(edge_1, edge_2)
        twice_area = np.linalg.norm(cross, axis=1)
        areas[start:stop] = 0.5 * twice_area
        centroids[start:stop] = vertices.mean(axis=1)
        normals[start:stop] = 0.0
        np.divide(
            cross,
            twice_area[:, None],
            out=normals[start:stop],
            where=twice_area[:, None] > 0.0,
        )
        if stop == n_cells or stop // geometry_chunk_cells % 10 == 0:
            _log(f"  geometry {stop:,}/{n_cells:,}")

    cp = np.asarray(triangles.cell_data["CpMeanTrim"], dtype=np.float32).copy()
    wss = np.asarray(
        triangles.cell_data["wallShearStressMeanTrim"], dtype=np.float32
    ).copy()
    if cp.shape != (n_cells,) or wss.shape != (n_cells, 3):
        raise ValueError(
            f"Unexpected triangulated label shapes: Cp={cp.shape}, WSS={wss.shape}"
        )

    del triangles, faces, points
    gc.collect()

    arrays = {
        "areas": areas,
        "centroids": centroids,
        "normals": normals,
        "cp": cp,
        "wss": wss,
    }
    metadata = {
        "path": str(path.resolve()),
        "sha256": source_hash,
        "bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "raw_n_points": raw_n_points,
        "raw_n_cells": raw_n_cells,
        "raw_is_all_triangles": raw_is_all_triangles,
        "raw_cell_data_fields": raw_fields,
        "triangulated_n_cells": n_cells,
        "triangulation": (
            "identity"
            if raw_is_all_triangles
            else "pyvista.PolyData.triangulate(pass_verts=False, pass_lines=False)"
        ),
        "geometry_dtype": str(centroids.dtype),
        "degenerate_triangle_count": int(np.count_nonzero(areas == 0.0)),
    }
    return arrays, metadata


def _full_totals(
    areas: np.ndarray,
    centroids: np.ndarray,
    cp: np.ndarray,
    wss: np.ndarray,
    center: np.ndarray,
    bbox_diagonal: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    coefficients = np.array([0.37, -0.21, 0.13], dtype=np.float32)
    affine = (
        1.0 + ((centroids - center.astype(np.float32)) @ coefficients) / bbox_diagonal
    ).astype(np.float32)
    fields = {
        "constant_1": np.ones(len(areas), dtype=np.float32),
        "affine_probe": affine,
        "centroid_x": centroids[:, 0],
        "centroid_y": centroids[:, 1],
        "centroid_z": centroids[:, 2],
        "cp": cp,
        "wss_x": wss[:, 0],
        "wss_y": wss[:, 1],
        "wss_z": wss[:, 2],
    }
    totals = {}
    for name, values in fields.items():
        total = np.einsum("i,i->", areas, values, dtype=np.float64, optimize=True)
        absolute_total = np.einsum(
            "i,i->", areas, np.abs(values), dtype=np.float64, optimize=True
        )
        totals[name] = {
            "full": _as_float(total),
            "full_l1_scale": _as_float(absolute_total),
        }
    return fields, totals


def _linear_total_metrics(
    indices: np.ndarray,
    areas: np.ndarray,
    fields: dict[str, np.ndarray],
    full_totals: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    factor = len(areas) / len(indices)
    retained_areas = areas[indices]
    result: dict[str, dict[str, float | None]] = {}
    for name, values in fields.items():
        bare = np.einsum(
            "i,i->",
            retained_areas,
            values[indices],
            dtype=np.float64,
            optimize=True,
        )
        ht = bare * factor
        full = full_totals[name]["full"]
        l1_scale = full_totals[name]["full_l1_scale"]
        relative_error = None
        if abs(full) > 1e-12 * max(l1_scale, 1.0):
            relative_error = _as_float((ht - full) / full)
        result[name] = {
            "bare": _as_float(bare),
            "ht": _as_float(ht),
            "full": full,
            "ht_relative_error": relative_error,
            "ht_error_normalized_by_full_l1": _as_float(
                (ht - full) / max(l1_scale, np.finfo(np.float64).tiny)
            ),
        }
    return result


def _fill_distance_metrics(
    retained_centroids: np.ndarray,
    query_centroids: np.ndarray,
    query_areas: np.ndarray,
    bbox_diagonal: float,
    workers: int,
) -> dict[str, Any]:
    tree = cKDTree(retained_centroids)
    distances, _ = tree.query(query_centroids, k=1, workers=workers)
    cell_quantiles = np.quantile(distances, QUANTILES)
    result = {
        "query_count": len(query_centroids),
        "mean": _as_float(distances.mean()),
        "rms": _as_float(np.sqrt(np.mean(np.square(distances)))),
        "max": _as_float(distances.max()),
        "cell_uniform_quantiles": {
            f"q{round(100 * quantile):02d}": _as_float(value)
            for quantile, value in zip(QUANTILES, cell_quantiles)
        },
        "area_weighted_quantiles": _weighted_quantiles(
            distances, query_areas, QUANTILES
        ),
    }
    result["normalized_by_bbox_diagonal"] = {
        "mean": _as_float(result["mean"] / bbox_diagonal),
        "rms": _as_float(result["rms"] / bbox_diagonal),
        "max": _as_float(result["max"] / bbox_diagonal),
        "cell_uniform_quantiles": {
            key: _as_float(value / bbox_diagonal)
            for key, value in result["cell_uniform_quantiles"].items()
        },
    }
    return result


def _selection_metrics(
    *,
    sampler: str,
    k: int,
    indices: np.ndarray,
    areas: np.ndarray,
    centroids: np.ndarray,
    fields: dict[str, np.ndarray],
    full_totals: dict[str, dict[str, float]],
    full_area: float,
    full_rms: float,
    query_centroids: np.ndarray,
    query_areas: np.ndarray,
    bbox_diagonal: float,
    workers: int,
    design: dict[str, Any],
) -> dict[str, Any]:
    retained_area = areas[indices].sum(dtype=np.float64)
    ht_area = retained_area * len(areas) / k
    sample_rms = _weighted_rms_radius(areas, centroids, indices)
    return {
        "sampler": sampler,
        "k": k,
        "design": design,
        "selection_sha256_int64": _sha256_array(indices.astype(np.int64, copy=False)),
        "first_indices": [int(value) for value in indices[:10]],
        "retained_geometric_area": _as_float(retained_area),
        "retained_geometric_area_fraction": _as_float(retained_area / full_area),
        "ht_constant_total": _as_float(ht_area),
        "ht_constant_relative_error": _as_float(ht_area / full_area - 1.0),
        "sample_r_rms": sample_rms,
        "r_rms_relative_error": _as_float(sample_rms / full_rms - 1.0),
        "linear_totals": _linear_total_metrics(indices, areas, fields, full_totals),
        "fill_distance": _fill_distance_metrics(
            centroids[indices],
            query_centroids,
            query_areas,
            bbox_diagonal,
            workers,
        ),
    }


def _order_autocorrelation(
    arrays: dict[str, np.ndarray], lags: tuple[int, ...]
) -> dict[str, Any]:
    areas = arrays["areas"]
    centroids = arrays["centroids"]
    normals = arrays["normals"]
    cp = arrays["cp"]
    wss = arrays["wss"]
    wss_magnitude = np.linalg.norm(wss, axis=1)
    result = {}
    for lag in lags:
        if not 0 < lag < len(areas):
            continue
        _log(f"  order autocorrelation lag {lag:,}")
        result[str(lag)] = {
            "area_pearson": _cyclic_scalar_correlation(areas, lag),
            "centroid_centered_vector_correlation": _cyclic_vector_correlation(
                centroids, lag
            ),
            "normal_mean_dot": _cyclic_vector_dot(normals, lag),
            "cp_pearson": _cyclic_scalar_correlation(cp, lag),
            "wss_centered_vector_correlation": _cyclic_vector_correlation(wss, lag),
            "wss_magnitude_pearson": _cyclic_scalar_correlation(wss_magnitude, lag),
        }
    del wss_magnitude
    return {
        "cyclic_lag_definition": "correlate cell i with cell (i + lag) mod N",
        "centered_independence_reference": 0.0,
        "normal_dot_independence_reference": _as_float(
            float(
                normals.mean(axis=0, dtype=np.float64)
                @ normals.mean(axis=0, dtype=np.float64)
            )
        ),
        "lags": result,
    }


def _rolling_rms_statistics(
    areas: np.ndarray,
    centroids: np.ndarray,
    k: int,
    full_rms: float,
) -> dict[str, Any]:
    _log(f"  exact rolling RMS distribution for k={k:,}")
    s0 = _cyclic_window_sums(areas, k)
    s1 = []
    for dim in range(3):
        weighted_coordinate = areas * centroids[:, dim]
        s1.append(_cyclic_window_sums(weighted_coordinate, k))
        del weighted_coordinate
    squared_radius = np.einsum(
        "ij,ij->i", centroids, centroids, dtype=np.float32, optimize=True
    )
    squared_radius *= areas
    s2 = _cyclic_window_sums(squared_radius, k)
    del squared_radius

    np.divide(s2, s0, out=s2)
    for component in s1:
        np.divide(component, s0, out=component)
        np.square(component, out=component)
        s2 -= component
    del s0, s1
    np.maximum(s2, 0.0, out=s2)
    np.sqrt(s2, out=s2)

    quantiles = np.quantile(s2, RMS_QUANTILES)
    mean = s2.mean(dtype=np.float64)
    result = {
        "population": "all possible cyclic starts",
        "n_starts": len(s2),
        "mean": _as_float(mean),
        "std": _as_float(s2.std(dtype=np.float64)),
        "min": _as_float(s2.min()),
        "max": _as_float(s2.max()),
        "quantiles": {
            f"q{round(100 * quantile):02d}": _as_float(value)
            for quantile, value in zip(RMS_QUANTILES, quantiles)
        },
        "full_surface_r_rms": full_rms,
        "mean_bias": _as_float(mean - full_rms),
        "mean_relative_bias": _as_float(mean / full_rms - 1.0),
        "relative_std": _as_float(s2.std(dtype=np.float64) / full_rms),
    }
    del s2
    gc.collect()
    return result


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in sorted({sample["k"] for sample in samples}):
        summary[str(k)] = {}
        for sampler in ("cyclic_block", "simple_random"):
            selected = [
                sample
                for sample in samples
                if sample["k"] == k and sample["sampler"] == sampler
            ]
            area_error = np.abs(
                [sample["ht_constant_relative_error"] for sample in selected]
            )
            rms_error = np.abs([sample["r_rms_relative_error"] for sample in selected])
            fill_q95 = np.array(
                [
                    sample["fill_distance"]["cell_uniform_quantiles"]["q95"]
                    for sample in selected
                ]
            )
            summary[str(k)][sampler] = {
                "n_replicates": len(selected),
                "median_abs_ht_constant_relative_error": _as_float(
                    np.median(area_error)
                ),
                "max_abs_ht_constant_relative_error": _as_float(area_error.max()),
                "median_abs_r_rms_relative_error": _as_float(np.median(rms_error)),
                "max_abs_r_rms_relative_error": _as_float(rms_error.max()),
                "median_fill_distance_q95": _as_float(np.median(fill_q95)),
                "max_fill_distance_q95": _as_float(fill_q95.max()),
            }
    return summary


def _audit_surface(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, source = _prepare_surface(path, args.geometry_chunk_cells)
    areas = arrays["areas"]
    centroids = arrays["centroids"]
    cp = arrays["cp"]
    wss = arrays["wss"]
    n_cells = len(areas)
    if max(args.k) > n_cells:
        raise ValueError(f"Requested k={max(args.k)} exceeds {n_cells} cells in {path}")

    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    full_area = float(areas.sum(dtype=np.float64))
    center = (
        np.einsum("i,ij->j", areas, centroids, dtype=np.float64, optimize=True)
        / full_area
    )
    full_rms = _weighted_rms_radius(areas, centroids)
    fields, full_totals = _full_totals(areas, centroids, cp, wss, center, bbox_diagonal)

    query_count = min(args.fill_query_count, n_cells)
    query_rng = np.random.default_rng(args.fill_query_seed)
    query_indices = np.sort(
        query_rng.choice(n_cells, size=query_count, replace=False, shuffle=False)
    )
    query_centroids = centroids[query_indices]
    query_areas = areas[query_indices]

    _log("Computing full-order autocorrelations")
    order_autocorrelation = _order_autocorrelation(arrays, tuple(args.order_lags))

    samples = []
    for k in args.k:
        cyclic_generator_seed = args.cyclic_start_seed + k
        generator = torch.Generator().manual_seed(cyclic_generator_seed)
        for ordinal in range(args.n_cyclic_starts):
            indices = _cyclic_block_indices(
                n_cells,
                k,
                generator=generator,
                device="cpu",
            ).numpy()
            start = int(indices[0])
            _log(f"  cyclic k={k:,} replicate={ordinal} start={start:,}")
            samples.append(
                _selection_metrics(
                    sampler="cyclic_block",
                    k=k,
                    indices=indices,
                    areas=areas,
                    centroids=centroids,
                    fields=fields,
                    full_totals=full_totals,
                    full_area=full_area,
                    full_rms=full_rms,
                    query_centroids=query_centroids,
                    query_areas=query_areas,
                    bbox_diagonal=bbox_diagonal,
                    workers=args.workers,
                    design={
                        "implementation": (
                            "physicsnemo.datapipes._indexing._cyclic_block_indices"
                        ),
                        "generator_seed": cyclic_generator_seed,
                        "draw_ordinal": ordinal,
                        "start": start,
                        "wraps": bool(start + k > n_cells),
                    },
                )
            )

        for seed in args.random_seeds:
            _log(f"  random k={k:,} seed={seed}")
            random_generator = np.random.default_rng(seed)
            indices = np.sort(
                random_generator.choice(
                    n_cells, size=k, replace=False, shuffle=False
                ).astype(np.int64, copy=False)
            )
            samples.append(
                _selection_metrics(
                    sampler="simple_random",
                    k=k,
                    indices=indices,
                    areas=areas,
                    centroids=centroids,
                    fields=fields,
                    full_totals=full_totals,
                    full_area=full_area,
                    full_rms=full_rms,
                    query_centroids=query_centroids,
                    query_areas=query_areas,
                    bbox_diagonal=bbox_diagonal,
                    workers=args.workers,
                    design={
                        "implementation": (
                            "numpy.random.Generator.choice("
                            "replace=False, shuffle=False)"
                        ),
                        "seed": seed,
                    },
                )
            )

    rolling_rms = {
        str(k): _rolling_rms_statistics(areas, centroids, k, full_rms)
        for k in args.rolling_rms_k
    }

    result = {
        "source": source,
        "full_surface": {
            "n_cells": n_cells,
            "geometric_area": _as_float(full_area),
            "area_weighted_centroid": [_as_float(value) for value in center],
            "r_rms": full_rms,
            "bbox_min": [_as_float(value) for value in bbox_min],
            "bbox_max": [_as_float(value) for value in bbox_max],
            "bbox_diagonal": _as_float(bbox_diagonal),
            "area_min": _as_float(areas.min()),
            "area_median": _as_float(np.median(areas)),
            "area_max": _as_float(areas.max()),
            "full_linear_totals": full_totals,
        },
        "fill_query_subset": {
            "method": "seeded simple random cells without replacement",
            "seed": args.fill_query_seed,
            "count": query_count,
            "indices_sha256_int64": _sha256_array(query_indices),
            "first_indices": [int(value) for value in query_indices[:10]],
        },
        "order_autocorrelation": order_autocorrelation,
        "samples": samples,
        "comparison_summary": _summarize_samples(samples),
        "rolling_r_rms": rolling_rms,
        "runtime_seconds": _as_float(time.perf_counter() - started),
    }

    del arrays, areas, centroids, cp, wss, fields
    gc.collect()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        action="append",
        required=True,
        type=Path,
        help="Raw DrivAerML boundary VTP; repeat for multiple cases.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument(
        "--rolling-rms-k",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROLLING_RMS_K),
    )
    parser.add_argument("--cyclic-start-seed", type=int, default=20_260_727)
    parser.add_argument("--n-cyclic-starts", type=int, default=4)
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANDOM_SEEDS),
    )
    parser.add_argument("--fill-query-count", type=int, default=50_000)
    parser.add_argument("--fill-query-seed", type=int, default=271_828)
    parser.add_argument(
        "--order-lags",
        nargs="+",
        type=int,
        default=list(DEFAULT_ORDER_LAGS),
    )
    parser.add_argument("--geometry-chunk-cells", type=int, default=500_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Workers used by scipy cKDTree queries.",
    )
    args = parser.parse_args()
    args.surface = [path.resolve() for path in args.surface]
    args.output = args.output.resolve()
    for path in args.surface:
        if not path.is_file():
            parser.error(f"Surface does not exist: {path}")
    if len(set(args.surface)) != len(args.surface):
        parser.error("--surface paths must be unique")
    if any(k <= 0 for k in [*args.k, *args.rolling_rms_k]):
        parser.error("All k values must be positive")
    if args.n_cyclic_starts <= 0 or args.fill_query_count <= 0:
        parser.error("Replicate and fill-query counts must be positive")
    return args


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]
    indexing_source = (
        repo_root / "physicsnemo" / "datapipes" / "_indexing.py"
    ).resolve()
    self_check = _self_check()

    results = {
        "schema_version": 1,
        "status": "PROTOTYPE_NOT_PRODUCTION_DATA",
        "warnings": PROTOTYPE_WARNINGS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "repository": str(repo_root),
            "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
            "git_branch": _git_output(repo_root, "branch", "--show-current"),
            "git_status_short": (
                _git_output(repo_root, "status", "--short") or ""
            ).splitlines(),
            "script": {
                "path": str(script_path.relative_to(repo_root)),
                "sha256": _sha256_file(script_path),
            },
            "production_indexing_source": {
                "path": str(indexing_source.relative_to(repo_root)),
                "sha256": _sha256_file(indexing_source),
                "symbol": "_cyclic_block_indices",
            },
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "pyvista": pv.__version__,
                "vtk": vtk.vtkVersion.GetVTKVersion(),
            },
        },
        "design": {
            "k": args.k,
            "rolling_rms_k": args.rolling_rms_k,
            "cyclic_start_seed": args.cyclic_start_seed,
            "n_cyclic_starts": args.n_cyclic_starts,
            "random_seeds": args.random_seeds,
            "fill_query_count": args.fill_query_count,
            "fill_query_seed": args.fill_query_seed,
            "order_lags": args.order_lags,
            "geometry_chunk_cells": args.geometry_chunk_cells,
            "workers": args.workers,
        },
        "self_check": self_check,
        "cases": [],
    }

    for surface in args.surface:
        results["cases"].append(_audit_surface(surface, args))

    results["total_runtime_seconds"] = _as_float(time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
