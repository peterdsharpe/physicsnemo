# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-carrier convergence audit for the Phase-1 surface-transfer map.

The curved icosphere refinement ladder changes both the physical carrier and
the quadrature.  This study instead freezes the subdivision-3 triangle soup
and recursively splits every flat triangle into four coplanar children.  It
therefore isolates whole-cell closest-face partition aliasing while retaining
the exact 156-face representation used by ``phase1_common_surface.py``.

The screen thresholds below were formalized after an exploratory diagnosis,
not preregistered before that diagnosis.  They are fixed before generation of
the dated artifact emitted by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from common_surface_transfer import (  # noqa: E402
    ReferenceSurfaceMap,
    build_reference_surface_map,
)
from phase1_common_surface import _field_diagnostics, _sphere  # noqa: E402
from provenance import runtime_environment, source_provenance  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import cell_measures  # noqa: E402
from physicsnemo.mesh.remeshing import remesh  # noqa: E402

DATE = "2026-07-27"
DEFAULT_MAX_REFINEMENT_LEVEL = 4
EXPECTED_BASE_REFERENCE_CELLS = 1_280
EXPECTED_REPRESENTATION_CELLS = 156
EXPECTED_LEVEL_0_MAP_SHA256 = (
    "7687e70cef50ccca705ea5d8f80c39f1843d59c214b7f1cf3c7c5821898c36d2"
)

GLOBAL_NORMALIZED_SHAPE_L1_THRESHOLD = 0.005
FIELD_METRIC_RELATIVE_DRIFT_THRESHOLD = 0.02
REQUIRED_SUCCESSIVE_TRANSITIONS = 2
FIELD_METRIC_KEYS = (
    "pressure_projection_floor_relative_l2",
    "wss_tangent_projection_floor_relative_l2",
    "traction_force_projection_relative_error",
    "traction_moment_projection_relative_error",
)


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _mesh_sha256(mesh: Mesh) -> str:
    return _tensor_sha256(mesh.points, mesh.cells)


def _file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _representation(
    reference: Mesh,
    *,
    frozen_path: Path | None,
    export_path: Path | None,
) -> tuple[Mesh, dict[str, Any]]:
    """Load an exact representation or generate and optionally freeze one."""
    if frozen_path is not None and export_path is not None:
        raise ValueError(
            "--representation-npz and --export-representation-npz are "
            "mutually exclusive"
        )

    if frozen_path is None:
        representation = remesh(reference, n_clusters=80)
        source: dict[str, Any] = {
            "kind": "generated_by_pyacvd",
            "requested_vertex_clusters": 80,
        }
        if export_path is not None:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                export_path,
                points=representation.points.detach().cpu().double().numpy(),
                cells=representation.cells.detach().cpu().long().numpy(),
            )
            source["export"] = {
                "path": str(export_path.resolve()),
                "sha256": _file_sha256(export_path),
            }
        return representation, source

    with np.load(frozen_path, allow_pickle=False) as payload:
        if set(payload.files) != {"points", "cells"}:
            raise ValueError(
                f"{frozen_path} must contain exactly 'points' and 'cells', "
                f"got {payload.files}"
            )
        points = torch.from_numpy(payload["points"].copy()).double()
        cells = torch.from_numpy(payload["cells"].copy()).long()
    return Mesh(points=points, cells=cells), {
        "kind": "frozen_npz",
        "path": str(frozen_path.resolve()),
        "sha256": _file_sha256(frozen_path),
    }


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    return {
        "q95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _relative_drift(coarse: float, fine: float) -> float:
    denominator = max(abs(coarse), abs(fine), torch.finfo(torch.float64).tiny)
    return abs(fine - coarse) / denominator


def _split_flat_triangles(reference: Mesh) -> tuple[Mesh, dict[str, float]]:
    """Split each triangle into four children without moving the carrier."""
    triangles = reference.points[reference.cells]
    point_0 = triangles[:, 0]
    point_1 = triangles[:, 1]
    point_2 = triangles[:, 2]
    midpoint_01 = (point_0 + point_1) / 2
    midpoint_12 = (point_1 + point_2) / 2
    midpoint_20 = (point_2 + point_0) / 2

    children = torch.stack(
        (
            torch.stack((point_0, midpoint_01, midpoint_20), dim=1),
            torch.stack((midpoint_01, point_1, midpoint_12), dim=1),
            torch.stack((midpoint_20, midpoint_12, point_2), dim=1),
            torch.stack((midpoint_01, midpoint_12, midpoint_20), dim=1),
        ),
        dim=1,
    )
    points = children.reshape(-1, 3)
    cells = torch.arange(
        points.shape[0],
        dtype=torch.long,
        device=points.device,
    ).reshape(-1, 3)
    refined = Mesh(points=points, cells=cells)

    parent_measures = cell_measures(reference)
    partitioned_measures = cell_measures(refined).reshape(-1, 4).sum(dim=1)
    relative_errors = (partitioned_measures - parent_measures).abs() / parent_measures
    parent_normals = reference.cell_normals
    child_normals = refined.cell_normals.reshape(-1, 4, 3)
    child_centroids = refined.cell_centroids.reshape(-1, 4, 3)
    parent_plane_distance = (
        (child_centroids - point_0[:, None]) * parent_normals[:, None]
    ).sum(dim=-1)
    return refined, {
        "parent_area_partition_max_relative_error": float(relative_errors.max()),
        "total_area_relative_error": float(
            (partitioned_measures.sum() - parent_measures.sum()).abs()
            / parent_measures.sum()
        ),
        "child_centroid_parent_plane_distance_max_abs": float(
            parent_plane_distance.abs().max()
        ),
        "child_parent_normal_dot_min": float(
            (child_normals * parent_normals[:, None]).sum(dim=-1).min()
        ),
    }


def _evaluate_level(
    reference: Mesh,
    representation: Mesh,
    level: int,
    base_area: float,
) -> tuple[ReferenceSurfaceMap, dict[str, Any]]:
    transfer, geometry = build_reference_surface_map(
        reference,
        representation,
        max_distance=0.1,
        min_normal_alignment=0.9,
    )
    total_area = float(transfer.reference_measures.sum())
    fields = _field_diagnostics(reference, representation, transfer)
    return transfer, {
        "level": level,
        "n_reference_cells": reference.n_cells,
        "n_reference_points": reference.points.shape[0],
        "reference_mesh_sha256": _mesh_sha256(reference),
        "map_sha256": transfer.sha256(),
        "total_area": total_area,
        "total_area_relative_drift_from_level_0": abs(total_area - base_area)
        / base_area,
        "represented_measure_min": float(transfer.representation_measures.min()),
        "represented_measure_max": float(transfer.representation_measures.max()),
        "geometry": geometry.summary(),
        "field_force_moment_metrics": fields,
    }


def _transition_diagnostics(
    coarse_transfer: ReferenceSurfaceMap,
    coarse_fields: dict[str, Any],
    fine_transfer: ReferenceSurfaceMap,
    fine_fields: dict[str, Any],
    coarse_level: int,
    split_integrity: dict[str, float],
) -> dict[str, Any]:
    if fine_transfer.n_reference_cells != 4 * coarse_transfer.n_reference_cells:
        raise RuntimeError("uniform refinement did not create four children per parent")

    coarse_measures = coarse_transfer.representation_measures
    fine_measures = fine_transfer.representation_measures
    coarse_shape = coarse_measures / coarse_measures.sum()
    fine_shape = fine_measures / fine_measures.sum()
    normalized_face_absolute_change = (fine_shape - coarse_shape).abs()
    normalized_face_relative_change = normalized_face_absolute_change / fine_shape

    child_assignments = fine_transfer.reference_to_representation.reshape(-1, 4)
    parent_assignments = coarse_transfer.reference_to_representation
    mixed_parents = (child_assignments != child_assignments[:, :1]).any(dim=1)
    child_changed_from_parent = child_assignments != parent_assignments[:, None]
    parents_with_changed_child = child_changed_from_parent.any(dim=1)
    coarse_area = coarse_transfer.reference_measures.sum()
    fine_area = fine_transfer.reference_measures.sum()

    field_drifts = {
        key: _relative_drift(coarse_fields[key], fine_fields[key])
        for key in FIELD_METRIC_KEYS
    }
    normalized_shape_l1 = float(normalized_face_absolute_change.sum())
    passes_global_shape = normalized_shape_l1 < GLOBAL_NORMALIZED_SHAPE_L1_THRESHOLD
    passes_field_metrics = (
        max(field_drifts.values()) < FIELD_METRIC_RELATIVE_DRIFT_THRESHOLD
    )
    return {
        "transition": f"level_{coarse_level}_to_{coarse_level + 1}",
        "represented_measure_normalized_shape_l1": normalized_shape_l1,
        "represented_measure_raw_relative_l1": float(
            (fine_measures - coarse_measures).abs().sum() / fine_area
        ),
        "total_area_relative_change": float(
            (fine_area - coarse_area).abs() / fine_area
        ),
        "normalized_face_measure_absolute_change": _distribution(
            normalized_face_absolute_change
        ),
        "normalized_face_measure_relative_change_to_fine": _distribution(
            normalized_face_relative_change
        ),
        "mixed_parent_count": int(mixed_parents.sum()),
        "mixed_parent_area_fraction": float(
            coarse_transfer.reference_measures[mixed_parents].sum() / coarse_area
        ),
        "parent_with_any_child_assignment_change_area_fraction": float(
            coarse_transfer.reference_measures[parents_with_changed_child].sum()
            / coarse_area
        ),
        "assignment_changes_confined_to_mixed_parents": not bool(
            (parents_with_changed_child & ~mixed_parents).any()
        ),
        "child_area_reassigned_from_parent_fraction": float(
            fine_transfer.reference_measures[
                child_changed_from_parent.reshape(-1)
            ].sum()
            / fine_area
        ),
        "split_integrity": split_integrity,
        "field_force_moment_metric_relative_drift": field_drifts,
        "maximum_field_force_moment_metric_relative_drift": max(field_drifts.values()),
        "screen_components": {
            "global_normalized_shape_l1_below_0p5pct": passes_global_shape,
            "all_field_force_moment_metric_drifts_below_2pct": (passes_field_metrics),
            "joint_transition_pass": passes_global_shape and passes_field_metrics,
        },
    }


def _trailing_true_count(values: list[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def _screen_summary(
    transitions: list[dict[str, Any]],
    max_refinement_level: int,
) -> dict[str, Any]:
    global_passes = [
        transition["screen_components"]["global_normalized_shape_l1_below_0p5pct"]
        for transition in transitions
    ]
    field_passes = [
        transition["screen_components"][
            "all_field_force_moment_metric_drifts_below_2pct"
        ]
        for transition in transitions
    ]
    joint_passes = [
        transition["screen_components"]["joint_transition_pass"]
        for transition in transitions
    ]
    trailing_global = _trailing_true_count(global_passes)
    trailing_field = _trailing_true_count(field_passes)
    trailing_joint = _trailing_true_count(joint_passes)
    passed = trailing_joint >= REQUIRED_SUCCESSIVE_TRANSITIONS

    if passed:
        verdict = "pass"
        interpretation = (
            "The fixed-carrier whole-cell partition passes the declared "
            "two-transition convergence screen."
        )
    elif (
        trailing_global == 1
        and trailing_joint == 1
        and max_refinement_level == DEFAULT_MAX_REFINEMENT_LEVEL
    ):
        verdict = "not_yet_passed"
        interpretation = (
            "Uniform level 4 supplies only one successive global normalized "
            "shape-L1 change below 0.5%; the screen requires two. This is "
            "insufficient evidence to promote a converged quantitative claim."
        )
    else:
        verdict = "fail"
        interpretation = (
            "The finest available transitions do not satisfy the declared "
            "two-transition convergence screen."
        )
    return {
        "verdict": verdict,
        "passed": passed,
        "trailing_global_shape_transitions_passing": trailing_global,
        "trailing_field_metric_transitions_passing": trailing_field,
        "trailing_joint_transitions_passing": trailing_joint,
        "interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=DEFAULT_MAX_REFINEMENT_LEVEL,
        help="maximum uniform in-plane refinement level (default: 4)",
    )
    parser.add_argument(
        "--representation-npz",
        type=Path,
        default=None,
        help="load the exact frozen 156-face representation from this NPZ",
    )
    parser.add_argument(
        "--export-representation-npz",
        type=Path,
        default=None,
        help="export the locally generated representation for a later run",
    )
    args = parser.parse_args()
    if args.max_level < 0:
        parser.error("--max-level must be nonnegative")
    if args.output is None:
        level_suffix = (
            ""
            if args.max_level == DEFAULT_MAX_REFINEMENT_LEVEL
            else f"_level{args.max_level}"
        )
        args.output = (
            EXAMPLE_ROOT
            / "results"
            / f"phase1_fixed_carrier_convergence{level_suffix}_{DATE}.json"
        )

    reference = _sphere(3)
    representation, representation_source = _representation(
        reference,
        frozen_path=args.representation_npz,
        export_path=args.export_representation_npz,
    )
    if reference.n_cells != EXPECTED_BASE_REFERENCE_CELLS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_REFERENCE_CELLS} base cells, "
            f"got {reference.n_cells}"
        )
    if representation.n_cells != EXPECTED_REPRESENTATION_CELLS:
        raise RuntimeError(
            f"expected {EXPECTED_REPRESENTATION_CELLS} representation cells, "
            f"got {representation.n_cells}"
        )

    base_area = float(cell_measures(reference).sum())
    representation_sha256 = _mesh_sha256(representation)
    levels: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_transfer: ReferenceSurfaceMap | None = None
    previous_fields: dict[str, Any] | None = None
    pending_split_integrity: dict[str, float] | None = None

    for level in range(args.max_level + 1):
        transfer, level_record = _evaluate_level(
            reference,
            representation,
            level,
            base_area,
        )
        if level == 0 and transfer.sha256() != EXPECTED_LEVEL_0_MAP_SHA256:
            raise RuntimeError(
                f"level-0 map does not match the Phase-1 artifact: {transfer.sha256()}"
            )
        levels.append(level_record)

        if (
            previous_transfer is not None
            and previous_fields is not None
            and pending_split_integrity is not None
        ):
            transitions.append(
                _transition_diagnostics(
                    previous_transfer,
                    previous_fields,
                    transfer,
                    level_record["field_force_moment_metrics"],
                    level - 1,
                    pending_split_integrity,
                )
            )

        previous_transfer = transfer
        previous_fields = level_record["field_force_moment_metrics"]
        if level < args.max_level:
            reference, pending_split_integrity = _split_flat_triangles(reference)

    maximum_area_drift = max(
        level["total_area_relative_drift_from_level_0"] for level in levels
    )
    integrity_gates = {
        "level_0_map_matches_phase1_artifact": (
            levels[0]["map_sha256"] == EXPECTED_LEVEL_0_MAP_SHA256
        ),
        "representation_has_156_faces": (
            representation.n_cells == EXPECTED_REPRESENTATION_CELLS
        ),
        "fixed_carrier_total_area_relative_drift_below_1e-12": (
            maximum_area_drift < 1.0e-12
        ),
        "all_parent_area_partitions_below_1e-12": all(
            transition["split_integrity"]["parent_area_partition_max_relative_error"]
            < 1.0e-12
            for transition in transitions
        ),
        "all_child_centroids_in_parent_plane_below_1e-12": all(
            transition["split_integrity"][
                "child_centroid_parent_plane_distance_max_abs"
            ]
            < 1.0e-12
            for transition in transitions
        ),
        "all_child_parent_normal_dots_above_1_minus_1e-12": all(
            transition["split_integrity"]["child_parent_normal_dot_min"] > 1.0 - 1.0e-12
            for transition in transitions
        ),
    }
    screen = _screen_summary(transitions, args.max_level)
    result = {
        "status": "phase1_fixed_carrier_common_surface_convergence",
        "date": DATE,
        "scope": (
            "Uniform in-plane quadrature refinement of the frozen "
            "subdivision-3 carrier against the exact 156-face Phase-1 "
            "representation. No curved geometry is introduced after level 0."
        ),
        "hypothesis": (
            "If the earlier refinement drift is whole-cell boundary aliasing, "
            "then fixed-carrier subdivision should localize changes to mixed "
            "parents and reduce represented-measure and field-metric drift."
        ),
        "screen_registration": {
            "status": (
                "formalized_after_exploratory_diagnosis_before_generation_of_"
                "this_dated_artifact"
            ),
            "confirmatory_preregistration": False,
            "global_metric": "represented_measure_normalized_shape_l1",
            "global_metric_threshold": GLOBAL_NORMALIZED_SHAPE_L1_THRESHOLD,
            "field_force_moment_metrics": list(FIELD_METRIC_KEYS),
            "field_metric_relative_drift_threshold": (
                FIELD_METRIC_RELATIVE_DRIFT_THRESHOLD
            ),
            "relative_drift_denominator": ("max(abs(coarse), abs(fine), float64_tiny)"),
            "required_successive_joint_transitions": (REQUIRED_SUCCESSIVE_TRANSITIONS),
        },
        "run_configuration": {
            "max_refinement_level": args.max_level,
            "representation_source": representation_source,
        },
        "construction": {
            "base_surface": "unit icosphere at subdivision 3",
            "refinement": (
                "four coplanar midpoint children per parent; vertices are not "
                "renormalized to the sphere"
            ),
            "maximum_uniform_refinement_level": args.max_level,
            "representation_source_kind": representation_source["kind"],
            "representation_faces": representation.n_cells,
            "representation_mesh_sha256": representation_sha256,
            "expected_level_0_map_sha256": EXPECTED_LEVEL_0_MAP_SHA256,
            "face_relative_change_definition": (
                "abs(fine_normalized_measure - coarse_normalized_measure) / "
                "fine_normalized_measure"
            ),
        },
        "levels": levels,
        "transitions": transitions,
        "integrity_gates": integrity_gates,
        "all_integrity_gates_pass": all(integrity_gates.values()),
        "screen": screen,
        "runtime": runtime_environment(torch.device("cpu")),
        "source_provenance": source_provenance(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
