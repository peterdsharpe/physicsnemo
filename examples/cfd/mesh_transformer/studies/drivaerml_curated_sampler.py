# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Audit cyclic-block sampling directly on curated DrivAerML ``.pdmsh`` files.

This script reads the on-disk vehicle memmaps without converting or reordering
them.  It compares the production cyclic-block design with seeded
simple-random subsets at one fixed cell budget.  The fill-distance diagnostic
is exact for a frozen query subset, but that subset is only a proxy for the
full surface.

Example
-------
Run on a machine with direct access to the curated dataset::

    python drivaerml_curated_sampler.py \
      --case run_1 /data/run_1/domain_run_1.pdmsh 11914080,11878329 \
      --case run_2 /data/run_2/domain_run_2.pdmsh 2471027,12664241 \
      --output curated_sampler.json
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ARRAY_PATHS = {
    "points": "points.memmap",
    "cells": "cells.memmap",
    "CpMeanTrim": "cell_data/CpMeanTrim.memmap",
    "wallShearStressMeanTrim": "cell_data/wallShearStressMeanTrim.memmap",
}
RANDOM_SEEDS = (17, 42, 137, 20_260_727)
FILL_QUANTILES = (0.5, 0.9, 0.95, 0.99)


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _finite_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite diagnostic value: {result}")
    return result


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[str, float]:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative /= cumulative[-1]
    return {
        f"q{round(100 * quantile):02d}": _finite_float(
            np.interp(quantile, cumulative, values[order])
        )
        for quantile in quantiles
    }


def _geometry(
    points: np.ndarray,
    cells: np.ndarray,
    chunk_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(cells)
    areas = np.empty(n_cells, dtype=np.float32)
    centroids = np.empty((n_cells, 3), dtype=np.float32)
    for start in range(0, n_cells, chunk_cells):
        stop = min(start + chunk_cells, n_cells)
        vertices = points[cells[start:stop]]
        edge_1 = vertices[:, 1] - vertices[:, 0]
        edge_2 = vertices[:, 2] - vertices[:, 0]
        areas[start:stop] = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
        centroids[start:stop] = vertices.mean(axis=1)
        print(
            f"geometry_completed_cells={stop}/{n_cells}",
            file=sys.stderr,
            flush=True,
        )
    return areas, centroids


def _weighted_rms_radius(
    areas: np.ndarray,
    centroids: np.ndarray,
    indices: np.ndarray | None = None,
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
    return _finite_float(math.sqrt(max(0.0, squared)))


def _cyclic_scalar_correlation(values: np.ndarray) -> float:
    mean = values.mean(dtype=np.float64)
    second = np.einsum("i,i->", values, values, dtype=np.float64, optimize=True) / len(
        values
    )
    variance = second - mean * mean
    if variance <= 0.0:
        return 0.0
    cross = (
        np.einsum("i,i->", values[:-1], values[1:], dtype=np.float64, optimize=True)
        + float(values[-1]) * float(values[0])
    ) / len(values)
    return _finite_float((cross - mean * mean) / variance)


def _cyclic_vector_correlation(values: np.ndarray) -> float:
    mean = values.mean(axis=0, dtype=np.float64)
    second = np.einsum(
        "ij,ij->", values, values, dtype=np.float64, optimize=True
    ) / len(values)
    variance = second - float(mean @ mean)
    if variance <= 0.0:
        return 0.0
    cross = (
        np.einsum(
            "ij,ij->",
            values[:-1],
            values[1:],
            dtype=np.float64,
            optimize=True,
        )
        + float(values[-1] @ values[0])
    ) / len(values)
    return _finite_float((cross - float(mean @ mean)) / variance)


def _order_diagnostic(
    areas: np.ndarray,
    centroids: np.ndarray,
    cp: np.ndarray,
    wss: np.ndarray,
    random_pair_count: int,
    random_pair_seed: int,
    chunk_cells: int,
) -> dict[str, Any]:
    consecutive_sum = 0.0
    consecutive_count = len(centroids) - 1
    for start in range(0, consecutive_count, chunk_cells):
        stop = min(start + chunk_cells, consecutive_count)
        delta = centroids[start + 1 : stop + 1] - centroids[start:stop]
        consecutive_sum += np.linalg.norm(delta, axis=1).sum(dtype=np.float64)

    generator = np.random.default_rng(random_pair_seed)
    left = generator.integers(0, len(centroids), size=random_pair_count)
    right = generator.integers(0, len(centroids), size=random_pair_count)
    random_distance = np.linalg.norm(centroids[left] - centroids[right], axis=1).mean(
        dtype=np.float64
    )
    consecutive_distance = consecutive_sum / consecutive_count
    return {
        "cyclic_lag": 1,
        "area_pearson": _cyclic_scalar_correlation(areas),
        "centroid_centered_vector_correlation": (_cyclic_vector_correlation(centroids)),
        "CpMeanTrim_pearson": _cyclic_scalar_correlation(cp),
        "wallShearStressMeanTrim_centered_vector_correlation": (
            _cyclic_vector_correlation(wss)
        ),
        "consecutive_centroid_distance_mean": _finite_float(consecutive_distance),
        "random_pair_centroid_distance_mean": _finite_float(random_distance),
        "consecutive_over_random_distance_ratio": _finite_float(
            consecutive_distance / random_distance
        ),
        "random_pair_count": random_pair_count,
        "random_pair_seed": random_pair_seed,
    }


def _nearest_distance_metrics(
    selected_centroids: np.ndarray,
    query_centroids: np.ndarray,
    query_areas: np.ndarray,
    bbox_diagonal: float,
    query_block: int,
) -> dict[str, Any]:
    selected = np.asarray(selected_centroids, dtype=np.float32)
    selected_sq = np.einsum(
        "ij,ij->i", selected, selected, dtype=np.float32, optimize=True
    )
    distances = np.empty(len(query_centroids), dtype=np.float32)
    for start in range(0, len(query_centroids), query_block):
        stop = min(start + query_block, len(query_centroids))
        query = np.asarray(query_centroids[start:stop], dtype=np.float32)
        query_sq = np.einsum("ij,ij->i", query, query, dtype=np.float32, optimize=True)
        squared = query_sq[:, None] + selected_sq[None, :]
        squared -= 2.0 * (query @ selected.T)
        np.maximum(squared, 0.0, out=squared)
        distances[start:stop] = np.sqrt(squared.min(axis=1))
    quantiles = np.quantile(distances, FILL_QUANTILES)
    return {
        "definition": (
            "exact ambient-Euclidean nearest-centroid distance from each "
            "frozen query cell to the selected 40k centroids"
        ),
        "query_count": len(query_centroids),
        "mean": _finite_float(distances.mean(dtype=np.float64)),
        "rms": _finite_float(np.sqrt(np.mean(np.square(distances), dtype=np.float64))),
        "max": _finite_float(distances.max()),
        "cell_uniform_quantiles": {
            f"q{round(100 * quantile):02d}": _finite_float(value)
            for quantile, value in zip(FILL_QUANTILES, quantiles)
        },
        "area_weighted_quantiles": _weighted_quantiles(
            distances, query_areas, FILL_QUANTILES
        ),
        "normalized_by_bbox_diagonal": {
            "mean": _finite_float(distances.mean(dtype=np.float64) / bbox_diagonal),
            "max": _finite_float(distances.max() / bbox_diagonal),
            "q95": _finite_float(quantiles[2] / bbox_diagonal),
        },
    }


def _linear_total(
    values: np.ndarray,
    areas: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | None]:
    full = np.einsum("i,i->", areas, values, dtype=np.float64, optimize=True)
    full_l1 = np.einsum("i,i->", areas, np.abs(values), dtype=np.float64, optimize=True)
    ht = (
        np.einsum(
            "i,i->",
            areas[indices],
            values[indices],
            dtype=np.float64,
            optimize=True,
        )
        * len(areas)
        / len(indices)
    )
    relative_error = None
    if abs(full) > 1e-12 * max(full_l1, 1.0):
        relative_error = _finite_float((ht - full) / full)
    return {
        "full": _finite_float(full),
        "ht": _finite_float(ht),
        "ht_relative_error": relative_error,
        "ht_error_normalized_by_full_l1": _finite_float(
            (ht - full) / max(full_l1, np.finfo(np.float64).tiny)
        ),
    }


def _selection_metrics(
    *,
    sampler: str,
    design: dict[str, Any],
    indices: np.ndarray,
    areas: np.ndarray,
    centroids: np.ndarray,
    cp: np.ndarray,
    wss: np.ndarray,
    full_area: float,
    full_rms: float,
    query_centroids: np.ndarray,
    query_areas: np.ndarray,
    bbox_diagonal: float,
    query_block: int,
) -> dict[str, Any]:
    retained_area = areas[indices].sum(dtype=np.float64)
    ht_area = retained_area * len(areas) / len(indices)
    return {
        "sampler": sampler,
        "design": design,
        "selection_sha256_int64": _sha256_array(indices),
        "first_indices": [int(value) for value in indices[:10]],
        "retained_geometric_area_fraction": _finite_float(retained_area / full_area),
        "ht_constant_relative_error": _finite_float(ht_area / full_area - 1.0),
        "r_rms_relative_error": _finite_float(
            _weighted_rms_radius(areas, centroids, indices) / full_rms - 1.0
        ),
        "linear_totals": {
            "CpMeanTrim": _linear_total(cp, areas, indices),
            "wallShearStressMeanTrim_x": _linear_total(wss[:, 0], areas, indices),
            "wallShearStressMeanTrim_y": _linear_total(wss[:, 1], areas, indices),
            "wallShearStressMeanTrim_z": _linear_total(wss[:, 2], areas, indices),
        },
        "fill_distance_proxy": _nearest_distance_metrics(
            centroids[indices],
            query_centroids,
            query_areas,
            bbox_diagonal,
            query_block,
        ),
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for sampler in ("cyclic_block", "simple_random"):
        selected = [sample for sample in samples if sample["sampler"] == sampler]
        area_errors = np.abs(
            [sample["ht_constant_relative_error"] for sample in selected]
        )
        cp_errors = np.abs(
            [
                sample["linear_totals"]["CpMeanTrim"]["ht_error_normalized_by_full_l1"]
                for sample in selected
            ]
        )
        fill_q95 = np.array(
            [
                sample["fill_distance_proxy"]["normalized_by_bbox_diagonal"]["q95"]
                for sample in selected
            ]
        )
        result[sampler] = {
            "n_replicates": len(selected),
            "median_abs_ht_constant_relative_error": _finite_float(
                np.median(area_errors)
            ),
            "max_abs_ht_constant_relative_error": _finite_float(area_errors.max()),
            "median_abs_Cp_ht_error_normalized_by_full_l1": _finite_float(
                np.median(cp_errors)
            ),
            "max_abs_Cp_ht_error_normalized_by_full_l1": _finite_float(cp_errors.max()),
            "median_normalized_fill_q95": _finite_float(np.median(fill_q95)),
            "max_normalized_fill_q95": _finite_float(fill_q95.max()),
        }
    cyclic = result["cyclic_block"]
    random = result["simple_random"]
    result["cyclic_over_random"] = {
        "median_abs_area_error_ratio": _finite_float(
            cyclic["median_abs_ht_constant_relative_error"]
            / max(
                random["median_abs_ht_constant_relative_error"],
                np.finfo(np.float64).tiny,
            )
        ),
        "median_abs_Cp_error_ratio": _finite_float(
            cyclic["median_abs_Cp_ht_error_normalized_by_full_l1"]
            / max(
                random["median_abs_Cp_ht_error_normalized_by_full_l1"],
                np.finfo(np.float64).tiny,
            )
        ),
        "median_fill_q95_ratio": _finite_float(
            cyclic["median_normalized_fill_q95"] / random["median_normalized_fill_q95"]
        ),
    }
    return result


def _audit_case(
    case_id: str,
    path: Path,
    cyclic_starts: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = path / "_tensordict" / "boundaries" / "vehicle" / "_tensordict"
    mesh_meta = json.loads((root / "meta.json").read_text())
    cell_meta = json.loads((root / "cell_data" / "meta.json").read_text())
    n_points = mesh_meta["points"]["shape"][0]
    n_cells = mesh_meta["cells"]["shape"][0]
    if args.k > n_cells:
        raise ValueError(f"k={args.k} exceeds {case_id} n_cells={n_cells}")

    points = np.memmap(
        root / ARRAY_PATHS["points"],
        dtype="<f4",
        mode="r",
        shape=(n_points, 3),
    )
    cells = np.memmap(
        root / ARRAY_PATHS["cells"],
        dtype="<i8",
        mode="r",
        shape=(n_cells, 3),
    )
    cp = np.memmap(
        root / ARRAY_PATHS["CpMeanTrim"],
        dtype="<f4",
        mode="r",
        shape=(n_cells,),
    )
    wss = np.memmap(
        root / ARRAY_PATHS["wallShearStressMeanTrim"],
        dtype="<f4",
        mode="r",
        shape=(n_cells, 3),
    )
    input_arrays = {
        name: {
            "path": str(root / relative_path),
            "shape": (
                mesh_meta[name]["shape"]
                if name in ("points", "cells")
                else cell_meta[name]["shape"]
            ),
            "dtype": (
                mesh_meta[name]["dtype"]
                if name in ("points", "cells")
                else cell_meta[name]["dtype"]
            ),
            "sha256": _sha256_file(root / relative_path),
        }
        for name, relative_path in ARRAY_PATHS.items()
    }
    areas, centroids = _geometry(points, cells, args.geometry_chunk_cells)
    full_area = areas.sum(dtype=np.float64)
    full_rms = _weighted_rms_radius(areas, centroids)
    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    query_generator = np.random.default_rng(args.fill_query_seed)
    query_indices = np.sort(
        query_generator.choice(
            n_cells,
            size=min(args.fill_query_count, n_cells),
            replace=False,
            shuffle=False,
        )
    )
    query_centroids = centroids[query_indices]
    query_areas = areas[query_indices]

    samples = []
    for ordinal, start in enumerate(cyclic_starts):
        indices = (start + np.arange(args.k, dtype=np.int64)) % n_cells
        samples.append(
            _selection_metrics(
                sampler="cyclic_block",
                design={
                    "implementation": (
                        "mathematically identical to "
                        "physicsnemo.datapipes._indexing."
                        "_cyclic_block_indices for explicit start"
                    ),
                    "start": start,
                    "draw_ordinal": ordinal,
                    "wraps": bool(start + args.k > n_cells),
                },
                indices=indices,
                areas=areas,
                centroids=centroids,
                cp=cp,
                wss=wss,
                full_area=full_area,
                full_rms=full_rms,
                query_centroids=query_centroids,
                query_areas=query_areas,
                bbox_diagonal=bbox_diagonal,
                query_block=args.fill_query_block,
            )
        )
        print(
            f"sample_completed case={case_id} sampler=cyclic ordinal={ordinal}",
            file=sys.stderr,
            flush=True,
        )

    for seed in args.random_seeds:
        generator = np.random.default_rng(seed)
        indices = np.sort(
            generator.choice(
                n_cells,
                size=args.k,
                replace=False,
                shuffle=False,
            ).astype(np.int64, copy=False)
        )
        samples.append(
            _selection_metrics(
                sampler="simple_random",
                design={
                    "implementation": "numpy.random.Generator.choice",
                    "seed": seed,
                },
                indices=indices,
                areas=areas,
                centroids=centroids,
                cp=cp,
                wss=wss,
                full_area=full_area,
                full_rms=full_rms,
                query_centroids=query_centroids,
                query_areas=query_areas,
                bbox_diagonal=bbox_diagonal,
                query_block=args.fill_query_block,
            )
        )
        print(
            f"sample_completed case={case_id} sampler=random seed={seed}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "case_id": case_id,
        "source": {
            "path": str(path),
            "vehicle_arrays": input_arrays,
            "n_points": n_points,
            "n_cells": n_cells,
        },
        "full_surface": {
            "geometric_area": _finite_float(full_area),
            "r_rms": full_rms,
            "bbox_diagonal": _finite_float(bbox_diagonal),
            "degenerate_triangle_count": int(np.count_nonzero(areas == 0.0)),
        },
        "fill_query": {
            "count": len(query_indices),
            "seed": args.fill_query_seed,
            "selection_sha256_int64": _sha256_array(query_indices),
        },
        "order_diagnostic": _order_diagnostic(
            areas,
            centroids,
            cp,
            wss,
            args.random_pair_count,
            args.random_pair_seed,
            args.geometry_chunk_cells,
        ),
        "samples": samples,
        "comparison_summary": _summary(samples),
        "runtime_seconds": _finite_float(time.perf_counter() - started),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("CASE_ID", "PDMSH", "CYCLIC_STARTS_CSV"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=40_000)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=list(RANDOM_SEEDS),
    )
    parser.add_argument("--fill-query-count", type=int, default=2_000)
    parser.add_argument("--fill-query-seed", type=int, default=271_828)
    parser.add_argument("--fill-query-block", type=int, default=100)
    parser.add_argument("--random-pair-count", type=int, default=200_000)
    parser.add_argument("--random-pair-seed", type=int, default=314_159)
    parser.add_argument("--geometry-chunk-cells", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    cases = []
    for case_id, path_arg, starts_arg in args.case:
        starts = [int(value) for value in starts_arg.split(",")]
        if not starts:
            raise ValueError(f"{case_id} has no cyclic starts")
        cases.append(_audit_case(case_id, Path(path_arg), starts, args))

    script_path = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED_DIRECT_CURATED_ORDER_AUDIT",
        "scope": (
            "Exact on-disk curated vehicle order for run_1 and run_2 at "
            "k=40,000; not a population estimate for the 435-case training "
            "split."
        ),
        "warnings": [
            "The fill-distance metric is exact only on a frozen 2,000-cell "
            "query subset, not on every surface cell.",
            "Simple-random sampling is a diagnostic comparator, not the "
            "production sampler.",
            "Four starts/seeds per scheme characterize these two cases only.",
        ],
        "design": {
            "k": args.k,
            "random_seeds": args.random_seeds,
            "fill_query_count": args.fill_query_count,
            "fill_query_seed": args.fill_query_seed,
            "fill_query_block": args.fill_query_block,
            "random_pair_count": args.random_pair_count,
            "random_pair_seed": args.random_pair_seed,
            "geometry_chunk_cells": args.geometry_chunk_cells,
        },
        "cases": cases,
        "provenance": {
            "script": {
                "path": str(script_path),
                "sha256": _sha256_file(script_path),
            },
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "environment_threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "runtime_seconds": _finite_float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
