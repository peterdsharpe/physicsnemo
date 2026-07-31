"""Measure source-to-target morphology relation without reading CFD fields.

The preregistered analysis uses only DrivAerML train geometries and the frozen
SHIFT-SUV ``train_128`` subsets. It intentionally runs before any adaptation
validation outcome is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physicsnemo.datapipes.readers.mesh import _subsample_mesh_cells
from physicsnemo.mesh import Mesh

UPPER_TRIANGLE = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def _stable_seed(base_seed: int, domain: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{domain}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, probability: float
) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = torch.cumsum(weights[order], dim=0)
    threshold = probability * cumulative[-1]
    index = int(torch.searchsorted(cumulative, threshold).clamp_max(len(values) - 1))
    return sorted_values[index]


def _upper_triangle(matrix: torch.Tensor) -> list[float]:
    return [float(matrix[i, j]) for i, j in UPPER_TRIANGLE]


def _descriptor(mesh: Mesh) -> tuple[list[float], dict[str, int]]:
    if mesh.n_spatial_dims != 3 or mesh.n_manifold_dims != 2:
        raise ValueError(
            "Expected a triangulated surface in 3D, got "
            f"manifold={mesh.n_manifold_dims}, spatial={mesh.n_spatial_dims}"
        )
    centroids = mesh.cell_centroids.to(dtype=torch.float64)
    areas = mesh.cell_areas.to(dtype=torch.float64)
    normals = mesh.cell_normals.to(dtype=torch.float64)
    finite = (
        torch.isfinite(centroids).all(dim=1)
        & torch.isfinite(areas)
        & torch.isfinite(normals).all(dim=1)
        & (areas > 0)
    )
    dropped = int((~finite).sum())
    centroids = centroids[finite]
    areas = areas[finite]
    normals = normals[finite]
    if len(areas) < 3:
        raise ValueError("Fewer than three finite, positive-area cells")

    weights = areas / areas.sum()
    center = torch.sum(weights[:, None] * centroids, dim=0)
    centered = centroids - center
    rms_radius = torch.sqrt(torch.sum(weights * torch.sum(centered**2, dim=1)))
    if not torch.isfinite(rms_radius) or rms_radius <= 0:
        raise ValueError(f"Invalid RMS radius: {float(rms_radius)}")
    normalized = centered / rms_radius

    spans = [
        float(
            _weighted_quantile(normalized[:, axis], weights, 0.95)
            - _weighted_quantile(normalized[:, axis], weights, 0.05)
        )
        for axis in range(3)
    ]
    centroid_moment = torch.einsum("n,ni,nj->ij", weights, normalized, normalized)
    normal_moment = torch.einsum("n,ni,nj->ij", weights, normals, normals)
    values = spans + _upper_triangle(centroid_moment) + _upper_triangle(normal_moment)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor contains non-finite values")
    return values, {"retained_cells": len(areas), "dropped_cells": dropped}


def _boundary_path(root: Path, case_id: str) -> Path:
    domain_meshes = sorted((root / case_id).glob("*.pdmsh"))
    if len(domain_meshes) != 1:
        raise ValueError(
            f"{root / case_id}: expected one .pdmsh directory, got {domain_meshes}"
        )
    path = domain_meshes[0] / "_tensordict" / "boundaries" / "vehicle"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _measure_case(
    domain: str,
    root: Path,
    case_id: str,
    sampling_seeds: tuple[int, int],
    sample_cells: int,
) -> dict[str, Any]:
    path = _boundary_path(root, case_id)
    passes = []
    diagnostics = []
    for base_seed in sampling_seeds:
        mesh = Mesh.load(path)
        generator = torch.Generator().manual_seed(
            _stable_seed(base_seed, domain, case_id)
        )
        mesh = _subsample_mesh_cells(mesh, sample_cells, generator=generator)
        values, pass_diagnostics = _descriptor(mesh)
        passes.append(values)
        diagnostics.append(pass_diagnostics)
    array = np.asarray(passes, dtype=np.float64)
    mean = array.mean(axis=0)
    relative_difference = float(
        np.linalg.norm(array[0] - array[1]) / max(np.linalg.norm(mean), 1e-12)
    )
    return {
        "case_id": case_id,
        "descriptor_passes": array.tolist(),
        "descriptor_mean": mean.tolist(),
        "relative_pass_difference": relative_difference,
        "diagnostics": diagnostics,
    }


def _load_cases(
    path: Path, split: str, expected_hash: str, expected_case_hash: str
) -> list[str]:
    if _sha256(path) != expected_hash:
        raise ValueError(f"Manifest hash mismatch: {path}")
    payload = json.loads(path.read_text())
    cases = payload[split]
    if not isinstance(cases, list) or not all(isinstance(case, str) for case in cases):
        raise TypeError(f"{path}:{split} must be a list of case IDs")
    if len(cases) != len(set(cases)):
        raise ValueError(f"{path}:{split} contains duplicate case IDs")
    if _sha256_lines(cases) != expected_case_hash:
        raise ValueError(f"Ordered case-ID hash mismatch: {path}:{split}")
    return cases


def _measure_domain(
    domain: str,
    root: Path,
    case_ids: list[str],
    sampling_seeds: tuple[int, int],
    sample_cells: int,
    workers: int,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _measure_case,
                domain,
                root,
                case_id,
                sampling_seeds,
                sample_cells,
            ): case_id
            for case_id in case_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            records[case_id] = future.result()
            if completed % 25 == 0 or completed == len(case_ids):
                print(f"PROGRESS {domain} {completed}/{len(case_ids)}", flush=True)
    return [records[case_id] for case_id in case_ids]


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(squared, 0.0))


def _energy_from_matrices(
    cross: np.ndarray,
    left_self: np.ndarray,
    right_self: np.ndarray,
    left_indices: np.ndarray | None = None,
    right_indices: np.ndarray | None = None,
) -> float:
    if left_indices is not None:
        cross = cross[np.ix_(left_indices, right_indices)]
        left_self = left_self[np.ix_(left_indices, left_indices)]
        right_self = right_self[np.ix_(right_indices, right_indices)]
    return float(2.0 * cross.mean() - left_self.mean() - right_self.mean())


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    return _energy_from_matrices(
        _pairwise_distances(left, right),
        _pairwise_distances(left, left),
        _pairwise_distances(right, right),
    )


def _analyze(
    records: dict[str, list[dict[str, Any]]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    means = {
        domain: np.asarray(
            [record["descriptor_mean"] for record in domain_records],
            dtype=np.float64,
        )
        for domain, domain_records in records.items()
    }
    source_mean = means["drivaer"].mean(axis=0)
    source_std = means["drivaer"].std(axis=0, ddof=1)
    retained = source_std >= 1e-8
    if not retained.any():
        raise ValueError("All source descriptor features are constant")
    standardized = {
        domain: (values[:, retained] - source_mean[retained]) / source_std[retained]
        for domain, values in means.items()
    }

    source = standardized["drivaer"]
    estate = standardized["estate"]
    fastback = standardized["fastback"]
    source_self = _pairwise_distances(source, source)
    estate_self = _pairwise_distances(estate, estate)
    fastback_self = _pairwise_distances(fastback, fastback)
    source_estate = _pairwise_distances(source, estate)
    source_fastback = _pairwise_distances(source, fastback)
    estate_distance = _energy_from_matrices(
        source_estate, source_self, estate_self
    )
    fastback_distance = _energy_from_matrices(
        source_fastback, source_self, fastback_self
    )

    rng = np.random.default_rng(bootstrap_seed)
    contrasts = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        source_indices = rng.integers(len(source), size=len(source))
        estate_indices = rng.integers(len(estate), size=len(estate))
        fastback_indices = rng.integers(len(fastback), size=len(fastback))
        estate_bootstrap = _energy_from_matrices(
            source_estate,
            source_self,
            estate_self,
            source_indices,
            estate_indices,
        )
        fastback_bootstrap = _energy_from_matrices(
            source_fastback,
            source_self,
            fastback_self,
            source_indices,
            fastback_indices,
        )
        contrasts[replicate] = estate_bootstrap - fastback_bootstrap

    pass_distances = []
    for pass_index in range(2):
        pass_values = {
            domain: np.asarray(
                [
                    record["descriptor_passes"][pass_index]
                    for record in domain_records
                ],
                dtype=np.float64,
            )
            for domain, domain_records in records.items()
        }
        pass_standardized = {
            domain: (values[:, retained] - source_mean[retained])
            / source_std[retained]
            for domain, values in pass_values.items()
        }
        pass_distances.append(
            {
                "pass_index": pass_index,
                "drivaer_to_estate": _energy_distance(
                    pass_standardized["drivaer"], pass_standardized["estate"]
                ),
                "drivaer_to_fastback": _energy_distance(
                    pass_standardized["drivaer"], pass_standardized["fastback"]
                ),
            }
        )

    stability = {}
    sampling_stable = True
    for domain, domain_records in records.items():
        differences = np.asarray(
            [record["relative_pass_difference"] for record in domain_records]
        )
        median = float(np.median(differences))
        p95 = float(np.quantile(differences, 0.95))
        passed = median <= 0.05 and p95 <= 0.15
        sampling_stable &= passed
        stability[domain] = {
            "median_relative_l2": median,
            "p95_relative_l2": p95,
            "passed": passed,
        }

    point_contrast = estate_distance - fastback_distance
    contrast_ci = [float(value) for value in np.quantile(contrasts, (0.025, 0.975))]
    point_order = np.sign(point_contrast)
    pass_order_stable = all(
        np.sign(item["drivaer_to_estate"] - item["drivaer_to_fastback"])
        == point_order
        for item in pass_distances
    )
    nearer = "estate" if point_contrast < 0 else "fastback"
    farther = "fastback" if nearer == "estate" else "estate"
    near_distance = min(estate_distance, fastback_distance)
    far_distance = max(estate_distance, fastback_distance)
    distance_ratio = float(far_distance / max(near_distance, 1e-12))
    ci_excludes_zero = contrast_ci[1] < 0 or contrast_ci[0] > 0
    decisive = (
        sampling_stable
        and pass_order_stable
        and ci_excludes_zero
        and distance_ratio >= 1.10
    )
    return {
        "retained_feature_indices": np.flatnonzero(retained).tolist(),
        "source_standardization": {
            "mean": source_mean.tolist(),
            "std": source_std.tolist(),
        },
        "energy_distances": {
            "drivaer_to_estate": estate_distance,
            "drivaer_to_fastback": fastback_distance,
            "estate_minus_fastback": point_contrast,
            "estate_minus_fastback_bootstrap_95_ci": contrast_ci,
            "farther_over_nearer": distance_ratio,
        },
        "pass_energy_distances": pass_distances,
        "sampling_stability": stability,
        "direction_stability_passed": pass_order_stable,
        "decision": {
            "decisive_morphology_ordering": decisive,
            "closer_family": nearer if decisive else None,
            "farther_family": farther if decisive else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--drivaer-root", type=Path, required=True)
    parser.add_argument("--estate-root", type=Path, required=True)
    parser.add_argument("--fastback-root", type=Path, required=True)
    parser.add_argument("--drivaer-manifest", type=Path, required=True)
    parser.add_argument("--estate-manifest", type=Path, required=True)
    parser.add_argument("--fastback-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    preregistration = json.loads(args.preregistration.read_text())
    declared = preregistration["data"]
    case_sets = {
        "drivaer": _load_cases(
            args.drivaer_manifest,
            "train",
            declared["drivaer_train"]["manifest_sha256"],
            declared["drivaer_train"]["ordered_case_ids_sha256"],
        ),
        "estate": _load_cases(
            args.estate_manifest,
            "train_128",
            declared["shift_suv_estate_train_128"]["manifest_sha256"],
            declared["shift_suv_estate_train_128"]["ordered_case_ids_sha256"],
        ),
        "fastback": _load_cases(
            args.fastback_manifest,
            "train_128",
            declared["shift_suv_fastback_train_128"]["manifest_sha256"],
            declared["shift_suv_fastback_train_128"]["ordered_case_ids_sha256"],
        ),
    }
    expected_counts = {
        "drivaer": declared["drivaer_train"]["cases"],
        "estate": declared["shift_suv_estate_train_128"]["cases"],
        "fastback": declared["shift_suv_fastback_train_128"]["cases"],
    }
    for domain, cases in case_sets.items():
        if len(cases) != expected_counts[domain]:
            raise ValueError(
                f"{domain}: expected {expected_counts[domain]} cases, got {len(cases)}"
            )

    measurement = preregistration["geometry_measurement"]
    sampling_seeds = tuple(measurement["sampling_seeds"])
    if len(sampling_seeds) != 2:
        raise ValueError("Exactly two sampling seeds are required")
    roots = {
        "drivaer": args.drivaer_root,
        "estate": args.estate_root,
        "fastback": args.fastback_root,
    }
    records = {
        domain: _measure_domain(
            domain,
            roots[domain],
            cases,
            sampling_seeds,
            int(measurement["cell_samples_per_pass"]),
            args.workers,
        )
        for domain, cases in case_sets.items()
    }
    bootstrap = preregistration["analysis"]["bootstrap"]
    analysis = _analyze(
        records,
        int(bootstrap["replicates"]),
        int(bootstrap["seed"]),
    )
    result = {
        "status": "complete",
        "preregistration_sha256": _sha256(args.preregistration),
        "script_sha256": _sha256(Path(__file__)),
        "test_cases_accessed": False,
        "validation_cases_accessed": False,
        "case_counts": {domain: len(cases) for domain, cases in case_sets.items()},
        "records": records,
        "analysis": analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "complete", "analysis": analysis}, indent=2))


if __name__ == "__main__":
    main()
