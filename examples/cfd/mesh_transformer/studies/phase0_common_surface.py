"""No-training prototype for conservative common-surface comparison.

The production DrivAerML pipeline still lacks an arbitrary-remesh overlap
map. This diagnostic tests the strongest mapping currently supported by
existing primitives: a frozen fine surface whose cells are assigned to
coarse cluster seeds. Restriction is area-weighted, and prolongation is
piecewise constant through the frozen ancestry map.

The result is deliberately labeled a cluster/ancestry prototype. Passing it
does not license nearest-centroid projection between unrelated meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from provenance import runtime_environment, source_provenance  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral  # noqa: E402
from physicsnemo.mesh.remeshing import CellPartition, partition_cells  # noqa: E402


def _sphere(subdivisions: int) -> Mesh:
    raw = sphere_icosahedral.load(radius=1.0, subdivisions=subdivisions)
    return Mesh(points=raw.points.double(), cells=raw.cells)


def _restrict(
    values: torch.Tensor,
    areas: torch.Tensor,
    partition: CellPartition,
) -> torch.Tensor:
    """Area-adjoint restriction from frozen fine cells to clusters."""
    scalar = values.ndim == 1
    matrix = values[:, None] if scalar else values
    weighted = matrix * areas[:, None]
    sums = matrix.new_zeros((partition.cluster_areas.shape[0], matrix.shape[1]))
    sums.index_add_(0, partition.assignments, weighted)
    nonempty = partition.cluster_areas > 0
    result = sums
    result[nonempty] = result[nonempty] / partition.cluster_areas[nonempty, None]
    result[~nonempty] = 0.0
    return result[:, 0] if scalar else result


def _prolong(values: torch.Tensor, partition: CellPartition) -> torch.Tensor:
    """Piecewise-constant prolongation through the frozen ancestry map."""
    return values[partition.assignments]


def _relative_l2(
    actual: torch.Tensor,
    expected: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    error = actual - expected
    if error.ndim > 1:
        weights = weights[:, None]
    numerator = (weights * error.square()).sum()
    denominator = (
        (weights * expected.square()).sum().clamp_min(torch.finfo(expected.dtype).tiny)
    )
    return float(torch.sqrt(numerator / denominator))


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _scalar_field(points: torch.Tensor) -> torch.Tensor:
    return (
        1.0
        + 0.3 * points[:, 0]
        - 0.2 * points[:, 1]
        + 0.15 * points[:, 2].square()
        + 0.1 * points[:, 0] * points[:, 2]
    )


def _tangent_field(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    raw = torch.stack(
        (
            0.5 + points[:, 1],
            -0.3 + 0.4 * points[:, 2],
            0.2 - points[:, 0],
        ),
        dim=-1,
    )
    return raw - (raw * normals).sum(dim=-1, keepdim=True) * normals


def _partition_diagnostics(fine: Mesh, seeds: torch.Tensor) -> dict[str, Any]:
    partition = partition_cells(fine, seeds)
    areas = fine.cell_areas
    nonempty = partition.cluster_areas > 0

    constant = torch.ones(fine.n_cells, dtype=fine.points.dtype)
    restricted_constant = _restrict(constant, areas, partition)
    constant_error = float((restricted_constant[nonempty] - 1.0).abs().max())

    scalar = _scalar_field(fine.cell_centroids)
    restricted_scalar = _restrict(scalar, areas, partition)
    prolonged_scalar = _prolong(restricted_scalar, partition)
    fine_integral = (areas * scalar).sum()
    coarse_integral = (partition.cluster_areas * restricted_scalar).sum()
    scalar_integral_relative_error = float(
        (coarse_integral - fine_integral).abs() / fine_integral.abs()
    )

    coarse_probe = torch.sin(
        torch.arange(seeds.shape[0], dtype=fine.points.dtype) * 0.37
    )
    coarse_roundtrip = _restrict(_prolong(coarse_probe, partition), areas, partition)
    roundtrip_error = float(
        (coarse_roundtrip[nonempty] - coarse_probe[nonempty]).abs().max()
    )

    normals = fine.cell_normals
    pressure = scalar
    wss = _tangent_field(fine.cell_centroids, normals)
    traction = pressure[:, None] * normals + wss
    restricted_traction = _restrict(traction, areas, partition)
    fine_force = (areas[:, None] * traction).sum(dim=0)
    coarse_force = (partition.cluster_areas[:, None] * restricted_traction).sum(dim=0)
    force_relative_error = float((coarse_force - fine_force).norm() / fine_force.norm())

    fine_moment = (
        areas[:, None] * torch.cross(fine.cell_centroids, traction, dim=-1)
    ).sum(dim=0)
    coarse_moment = (
        partition.cluster_areas[:, None]
        * torch.cross(
            partition.cluster_centroids,
            restricted_traction,
            dim=-1,
        )
    ).sum(dim=0)
    moment_relative_error = float(
        (coarse_moment - fine_moment).norm()
        / fine_moment.norm().clamp_min(torch.finfo(fine.points.dtype).tiny)
    )

    restricted_wss = _restrict(wss, areas, partition)
    coarse_wss = (
        restricted_wss
        - (restricted_wss * partition.cluster_normals).sum(dim=-1, keepdim=True)
        * partition.cluster_normals
    )
    fine_wss = _prolong(coarse_wss, partition)
    fine_wss = fine_wss - (fine_wss * normals).sum(dim=-1, keepdim=True) * normals
    tangency_residual = float((fine_wss * normals).sum(dim=-1).abs().max())

    return {
        "n_fine_cells": fine.n_cells,
        "n_clusters": seeds.shape[0],
        "n_empty_clusters": int((~nonempty).sum()),
        "fine_area": float(areas.sum()),
        "cluster_area": float(partition.cluster_areas.sum()),
        "area_relative_error": float(
            (partition.cluster_areas.sum() - areas.sum()).abs() / areas.sum()
        ),
        "constant_max_abs_error": constant_error,
        "scalar_integral_relative_error": scalar_integral_relative_error,
        "coarse_restrict_prolong_roundtrip_max_abs": roundtrip_error,
        "scalar_piecewise_constant_projection_floor_relative_l2": _relative_l2(
            prolonged_scalar, scalar, areas
        ),
        "force_relative_error": force_relative_error,
        "moment_relative_error_from_piecewise_constant_traction": (
            moment_relative_error
        ),
        "wss_tangency_max_abs": tangency_residual,
        "map_sha256": _tensor_digest(
            fine.points,
            fine.cells,
            seeds,
            partition.assignments,
            partition.cluster_areas,
        ),
        "restricted_scalar": restricted_scalar.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            EXAMPLE_ROOT / "results" / "phase0_common_surface_cluster_2026-07-27.json"
        ),
    )
    args = parser.parse_args()

    coarse = _sphere(1)
    seeds = coarse.cell_centroids
    fine_2 = _sphere(2)
    fine_3 = _sphere(3)
    resolution_2 = _partition_diagnostics(fine_2, seeds)
    resolution_3 = _partition_diagnostics(fine_3, seeds)
    coarse_measures = partition_cells(fine_3, seeds).cluster_areas
    r2 = torch.tensor(resolution_2["restricted_scalar"], dtype=torch.float64)
    r3 = torch.tensor(resolution_3["restricted_scalar"], dtype=torch.float64)
    common_surface_refinement_discrepancy = _relative_l2(
        r2,
        r3,
        coarse_measures,
    )

    result = {
        "status": "phase0_no_training_cluster_ancestry_prototype",
        "date": "2026-07-27",
        "scope": (
            "Conservative restriction/piecewise-constant prolongation on a "
            "frozen cluster ancestry map; not arbitrary-remesh projection."
        ),
        "coarse_seed_cells": coarse.n_cells,
        "resolution_2": resolution_2,
        "resolution_3": resolution_3,
        "common_surface_refinement_discrepancy_relative_l2": (
            common_surface_refinement_discrepancy
        ),
        "gates": {
            "no_empty_clusters": resolution_3["n_empty_clusters"] == 0,
            "area_conserved_below_1e-14": resolution_3["area_relative_error"] < 1.0e-14,
            "constant_preserved_below_1e-14": resolution_3["constant_max_abs_error"]
            < 1.0e-14,
            "integral_preserved_below_1e-14": resolution_3[
                "scalar_integral_relative_error"
            ]
            < 1.0e-14,
            "coarse_roundtrip_below_1e-14": resolution_3[
                "coarse_restrict_prolong_roundtrip_max_abs"
            ]
            < 1.0e-14,
            "force_preserved_below_1e-14": resolution_3["force_relative_error"]
            < 1.0e-14,
            "wss_tangent_below_1e-14": resolution_3["wss_tangency_max_abs"] < 1.0e-14,
        },
        "known_missing_gate": (
            "No conservative overlap/adjoint map exists for unrelated "
            "DrivAerML remeshes; force/moment comparisons there remain blocked."
        ),
        "runtime": runtime_environment(torch.device("cpu")),
        "source_provenance": source_provenance(),
    }
    result["all_in_scope_gates_pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
