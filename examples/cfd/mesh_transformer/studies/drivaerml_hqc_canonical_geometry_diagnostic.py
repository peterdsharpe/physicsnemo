# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Test a target-free, center-neutral geometry path for the blocked H-QC audit.

For the four frozen K=2500 canaries, this non-deciding diagnostic constructs
one canonical source geometry directly from the raw selected topology.  The
construction promotes raw coordinates to float64, removes their physical
area-weighted center, derives coherent geometry in that frame, divides by the
pipeline and model length gauges, and casts each resulting field to float32
once.  The same bundle is then injected into the historical primary- and
fixed-center paths.

Two interventions isolate the remaining coordinate route:

* ``canonical_derived`` replaces source centroids, areas, and normals.
* ``canonical_full`` additionally replaces source points.

Both interventions decode at the same canonical trace centroids.  Raw
supervision arrays are never indexed, synthetic placeholders are stripped
before model use, and no supervised metric or H-QC decision statistic is
computed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "hqc_canonical_geometry_diagnostic"
PASSED_STATUS = "PASSED_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC"
FAILED_STATUS = "FAILED_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC"

EXPECTED_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_PRIOR_DIAGNOSTIC_NPZ_SHA256 = (
    "d1e6a9fa1a39aa78a9cca26e52eb783a9e78aecbb961ce917164e25fac75a7ea"
)
EXPECTED_PRIOR_DIAGNOSTIC_JSON_SHA256 = (
    "26aed78264e9fd66f329941ce000fc438cb26f6835f9f05ff128567d29444bf5"
)
CASE_IDS = ("run_118", "run_129", "run_145", "run_149")
RESOLUTION = 2_500
PRECISIONS = ("bfloat16", "float32")
PREDICTION_FIELDS = ("pressure", "wss")
RAW_PLACEHOLDER_FIELDS = {
    "pMeanTrim": "scalar",
    "wallShearStressMeanTrim": "vector",
}
DERIVED_RELATIVE_TOLERANCE = 1.0e-3
UNIT_ABS_TOLERANCE = 1.0e-6
CENTER_ABS_TOLERANCE = 1.0e-6
FORBIDDEN_ARTIFACT_KEY_TOKENS = (
    "target",
    "truth",
    "error",
    "force",
    "area_weighted",
    "endpoint",
    "support",
    "futility",
    "mixed",
    "eligibility",
)


@dataclass(frozen=True)
class CanonicalRawGeometry:
    """Center-neutral geometry retained in physical float64 coordinates."""

    points: torch.Tensor
    cells: torch.Tensor
    centroids: torch.Tensor
    areas: torch.Tensor
    normals: torch.Tensor
    center: torch.Tensor


@dataclass(frozen=True)
class CanonicalSourceBundle:
    """Canonical geometry in the model's internal float32 coordinate frame."""

    points: torch.Tensor
    cells: torch.Tensor
    centroids: torch.Tensor
    areas: torch.Tensor
    normals: torch.Tensor
    physical_center: torch.Tensor
    physical_length: float
    model_reference_length: float


def _load_producer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_hqc_producer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen producer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_positive_scalar(value: Any, label: str) -> float:
    tensor = torch.as_tensor(value).detach().double().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{label} must contain one scalar, got {tuple(tensor.shape)}")
    result = float(tensor.item())
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {result}")
    return result


def _nested_tensor_value(data: Any, dotted_path: str) -> torch.Tensor:
    value = data
    for part in dotted_path.split("."):
        value = value[part]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{dotted_path!r} did not resolve to a tensor")
    return value


def _target_free_subset(
    mesh: Any,
    cell_ids: np.ndarray,
    mesh_type: Any,
) -> Any:
    """Compact selected geometry without indexing any raw data association."""
    ids = torch.as_tensor(np.asarray(cell_ids, dtype=np.int64), dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("cell_ids must be a non-empty vector")
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= mesh.n_cells:
        raise ValueError("cell_ids contain an out-of-range row")
    selected_cells = mesh.cells[ids]
    referenced, inverse = torch.unique(
        selected_cells,
        sorted=True,
        return_inverse=True,
    )
    compacted_cells = inverse.reshape_as(selected_cells)
    n_cells = int(ids.numel())
    placeholders = {
        name: (
            mesh.points.new_zeros(n_cells)
            if rank == "scalar"
            else mesh.points.new_zeros(n_cells, mesh.n_spatial_dims)
        )
        for name, rank in RAW_PLACEHOLDER_FIELDS.items()
    }
    return mesh_type(
        points=mesh.points[referenced],
        cells=compacted_cells,
        point_data={},
        cell_data=placeholders,
        global_data=mesh.global_data,
    )


def _strip_local_data(domain: Any, mesh_type: Any) -> Any:
    """Remove all local fields, including synthetic placeholders, before encode."""
    from physicsnemo.mesh import DomainMesh

    interior = mesh_type(
        points=domain.interior.points,
        cells=domain.interior.cells,
        point_data={},
        cell_data={},
        global_data={},
    )
    boundaries = {
        name: mesh_type(
            points=boundary.points,
            cells=boundary.cells,
            point_data={},
            cell_data={},
            global_data={},
        )
        for name, boundary in domain.boundaries.items()
    }
    stripped = DomainMesh(
        interior=interior,
        boundaries=boundaries,
        global_data=domain.global_data,
    )
    _require_no_local_data(stripped)
    return stripped


def _require_no_local_data(domain: Any) -> None:
    associations = {
        "interior.point_data": domain.interior.point_data,
        "interior.cell_data": domain.interior.cell_data,
    }
    for name, boundary in domain.boundaries.items():
        associations[f"boundaries.{name}.point_data"] = boundary.point_data
        associations[f"boundaries.{name}.cell_data"] = boundary.cell_data
    nonempty = {
        name: sorted(str(key) for key in data.keys())
        for name, data in associations.items()
        if len(data.keys()) != 0
    }
    if nonempty:
        raise ValueError(f"Model domain retains local data fields: {nonempty}")


def _triangle_geometry(
    points: torch.Tensor, cells: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if points.dtype != torch.float64:
        raise TypeError(f"points must be float64, got {points.dtype}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {tuple(points.shape)}")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"cells must have shape (M,3), got {tuple(cells.shape)}")
    vertices = points[cells]
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = torch.linalg.cross(edge_1, edge_2)
    twice_area = torch.linalg.vector_norm(cross, dim=-1)
    if not bool(torch.isfinite(points).all()) or not bool(
        torch.isfinite(twice_area).all()
    ):
        raise ValueError("Canonical geometry contains non-finite values")
    if bool(torch.any(twice_area <= 0.0)):
        raise ValueError("Canonical geometry contains a degenerate triangle")
    return vertices.mean(dim=1), 0.5 * twice_area, cross / twice_area[:, None]


def _build_canonical_raw_geometry(mesh: Any) -> CanonicalRawGeometry:
    """Compute coherent centered P/C/A/N in float64 before CenterMesh."""
    points = mesh.points.detach().to(device="cpu", dtype=torch.float64)
    cells = mesh.cells.detach().to(device="cpu", dtype=torch.long)
    centroids, areas, _ = _triangle_geometry(points, cells)
    total_area = areas.sum()
    if not bool(torch.isfinite(total_area)) or float(total_area.item()) <= 0.0:
        raise ValueError("Canonical source has non-positive total area")
    center = torch.einsum("n,nd->d", areas, centroids) / total_area
    centered_points = points - center
    centered_centroids, centered_areas, centered_normals = _triangle_geometry(
        centered_points,
        cells,
    )
    return CanonicalRawGeometry(
        points=centered_points,
        cells=cells.clone(),
        centroids=centered_centroids,
        areas=centered_areas,
        normals=centered_normals,
        center=center,
    )


def _finish_canonical_bundle(
    raw: CanonicalRawGeometry,
    *,
    physical_length: Any,
    model_reference_length: Any,
) -> CanonicalSourceBundle:
    """Scale in float64, then cast each canonical floating field exactly once."""
    physical = _as_positive_scalar(physical_length, "physical length")
    reference = _as_positive_scalar(
        model_reference_length,
        "model reference length",
    )
    scale = physical * reference
    points64 = raw.points / scale
    centroids64, areas64, normals64 = _triangle_geometry(points64, raw.cells)
    return CanonicalSourceBundle(
        points=points64.to(torch.float32),
        cells=raw.cells.clone(),
        centroids=centroids64.to(torch.float32),
        areas=areas64.to(torch.float32),
        normals=normals64.to(torch.float32),
        physical_center=raw.center.clone(),
        physical_length=physical,
        model_reference_length=reference,
    )


def _bundle_fields(bundle: CanonicalSourceBundle) -> dict[str, torch.Tensor]:
    return {
        "points": bundle.points,
        "centroids": bundle.centroids,
        "areas": bundle.areas,
        "normals": bundle.normals,
    }


def _bundle_difference(
    left: CanonicalSourceBundle,
    right: CanonicalSourceBundle,
) -> dict[str, bool]:
    return {
        "cells": bool(torch.equal(left.cells, right.cells)),
        **{
            name: bool(torch.equal(left_value, _bundle_fields(right)[name]))
            for name, left_value in _bundle_fields(left).items()
        },
        "physical_center": bool(
            torch.equal(left.physical_center, right.physical_center)
        ),
        "physical_length": left.physical_length == right.physical_length,
        "model_reference_length": (
            left.model_reference_length == right.model_reference_length
        ),
    }


def _bundle_validity(
    bundle: CanonicalSourceBundle,
    *,
    expected_cells: torch.Tensor,
) -> dict[str, Any]:
    expected_cells = expected_cells.detach().cpu()
    expected_n_cells = expected_cells.shape[0]
    expected_n_points = int(expected_cells.max().item()) + 1
    expected_shapes = {
        "points": (expected_n_points, 3),
        "cells": (expected_n_cells, 3),
        "centroids": (expected_n_cells, 3),
        "areas": (expected_n_cells,),
        "normals": (expected_n_cells, 3),
    }
    observed = {
        "points": bundle.points,
        "cells": bundle.cells,
        "centroids": bundle.centroids,
        "areas": bundle.areas,
        "normals": bundle.normals,
    }
    shape_checks = {
        name: tuple(observed[name].shape) == shape
        for name, shape in expected_shapes.items()
    }
    finite_checks = {
        name: bool(torch.isfinite(value).all())
        for name, value in _bundle_fields(bundle).items()
    }
    unit_deviation = float(
        torch.max(
            torch.abs(torch.linalg.vector_norm(bundle.normals.double(), dim=-1) - 1.0)
        ).item()
    )
    area_weighted_center = (
        torch.einsum(
            "n,nd->d",
            bundle.areas.double(),
            bundle.centroids.double(),
        )
        / bundle.areas.double().sum()
    )
    center_deviation = float(torch.max(torch.abs(area_weighted_center)).item())
    checks = {
        "shapes": all(shape_checks.values()),
        "topology": bool(torch.equal(bundle.cells.cpu(), expected_cells)),
        "finite": all(finite_checks.values()),
        "positive_areas": bool(torch.all(bundle.areas > 0.0)),
        "unit_normals": unit_deviation <= UNIT_ABS_TOLERANCE,
        "area_centered": center_deviation <= CENTER_ABS_TOLERANCE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "shape_checks": shape_checks,
        "finite_checks": finite_checks,
        "maximum_unit_deviation": unit_deviation,
        "maximum_area_center_deviation": center_deviation,
    }


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().double()
    right64 = right.detach().double()
    numerator = torch.linalg.vector_norm(left64 - right64)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(left64),
            torch.linalg.vector_norm(right64),
        ),
        left64.new_tensor(1.0e-12),
    )
    return float((numerator / denominator).item())


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(
            f"Cannot compare shapes {tuple(left.shape)} and {tuple(right.shape)}"
        )
    delta = left.detach().double() - right.detach().double()
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "exact": bool(torch.equal(left, right)),
        "nonzero_count": int(torch.count_nonzero(delta).item()),
        "maximum_absolute_difference": (
            float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
        ),
        "relative_l2_difference": _relative_l2(left, right),
    }


def _prediction_difference(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {
        field: _tensor_difference(left[field], right[field])
        for field in PREDICTION_FIELDS
    }


def _difference_is_exact(
    difference: Mapping[str, Mapping[str, Any]],
) -> bool:
    return all(bool(difference[field]["exact"]) for field in PREDICTION_FIELDS)


def _difference_within(
    difference: Mapping[str, Mapping[str, Any]],
    tolerance: float,
) -> bool:
    return all(
        float(difference[field]["relative_l2_difference"]) <= tolerance
        for field in PREDICTION_FIELDS
    )


def _extract_prediction(output: Any, n_queries: int) -> dict[str, torch.Tensor]:
    result = {
        field: output.point_data[field].detach().float().clone()
        for field in PREDICTION_FIELDS
    }
    expected_shapes = {
        "pressure": (n_queries,),
        "wss": (n_queries, 3),
    }
    for field, expected in expected_shapes.items():
        value = result[field]
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{field} prediction has shape {tuple(value.shape)}, expected {expected}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{field} prediction contains non-finite values")
    return result


def _apply_canonical_geometry(
    source_mesh: Any,
    bundle: CanonicalSourceBundle,
    mode: str,
) -> None:
    key_sets = {
        "canonical_derived": ("centroids", "areas", "normals"),
        "canonical_full": ("centroids", "areas", "normals"),
    }
    if mode not in key_sets:
        raise ValueError(f"Unknown canonical geometry mode {mode!r}")
    if tuple(source_mesh.points.shape) != tuple(bundle.points.shape):
        raise ValueError("Canonical/source point shapes differ")
    if not torch.equal(source_mesh.cells.detach().cpu(), bundle.cells):
        raise ValueError("Canonical/source compacted topology differs")
    if mode == "canonical_full":
        source_mesh.points.copy_(
            bundle.points.to(
                device=source_mesh.points.device,
                dtype=source_mesh.points.dtype,
            )
        )
    values = _bundle_fields(bundle)
    for key in key_sets[mode]:
        source_mesh._cache["cell", key] = values[key].to(
            device=source_mesh.points.device,
            dtype=source_mesh.points.dtype,
        )


def _encode_with_canonical_geometry(
    runtime: Any,
    domain: Any,
    bundle: CanonicalSourceBundle,
    mode: str,
) -> Any:
    model = runtime.model
    original = model._source_operator_input

    def intercepted(
        model_domain: Any,
        source_mesh: Any,
        boundary_operator: Any,
        global_operator: Any,
    ) -> Any:
        _apply_canonical_geometry(source_mesh, bundle, mode)
        return original(
            model_domain,
            source_mesh,
            boundary_operator,
            global_operator,
        )

    object.__setattr__(model, "_source_operator_input", intercepted)
    try:
        return model.encode(domain)
    finally:
        object.__setattr__(model, "_source_operator_input", original)


def _decode_at_canonical_centroids(
    runtime: Any,
    encoded: Any,
    bundle: CanonicalSourceBundle,
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    neutral = replace(
        encoded,
        center=torch.zeros_like(encoded.center),
        reference_length=torch.ones_like(encoded.reference_length),
    )
    query_mesh = runtime.mesh_type(
        points=bundle.centroids.to(
            device=encoded.source_mesh.points.device,
            dtype=encoded.source_mesh.points.dtype,
        )
    )
    output = runtime.model.decode(neutral, query_mesh)
    checks = {
        "canonical_queries_exact": bool(
            torch.equal(query_mesh.points.detach().cpu(), bundle.centroids)
        ),
        "encoded_center_is_exact_zero": bool(
            torch.equal(neutral.center, torch.zeros_like(neutral.center))
        ),
        "encoded_reference_length_is_exact_one": bool(
            torch.equal(
                neutral.reference_length,
                torch.ones_like(neutral.reference_length),
            )
        ),
    }
    return _extract_prediction(output, bundle.centroids.shape[0]), checks


def _historical_prediction(
    runtime: Any,
    domain: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    encoded = runtime.model.encode(domain)
    return (
        encoded,
        _extract_prediction(
            runtime.model.decode(encoded),
            domain.interior.n_points,
        ),
    )


def _historical_replay(
    runtime: Any,
    *,
    primary_domain: Any,
    fixed_domain: Any,
    precision: str,
    prior_arrays: Mapping[str, np.ndarray],
    prior_prefix: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with torch.no_grad(), runtime.autocast_context(precision):
        encoded_primary, primary = _historical_prediction(runtime, primary_domain)
        encoded_fixed, fixed = _historical_prediction(runtime, fixed_domain)
    prediction_comparisons: dict[str, Any] = {}
    geometry_comparisons: dict[str, Any] = {}
    encoded_by_path = {
        "primary": encoded_primary,
        "fixed": encoded_fixed,
    }
    prediction_by_path = {
        "primary": primary,
        "fixed": fixed,
    }
    geometry_names = {
        "points": lambda encoded: encoded.source_mesh.points,
        "centroids": lambda encoded: encoded.source_mesh.cell_centroids,
        "areas": lambda encoded: encoded.source_mesh.cell_areas,
        "normals": lambda encoded: encoded.source_mesh.cell_normals,
    }
    for path, prediction in prediction_by_path.items():
        prediction_comparisons[path] = {}
        for field in PREDICTION_FIELDS:
            key = f"{prior_prefix}__{precision}_{path}_{field}"
            if key not in prior_arrays:
                raise KeyError(f"Prior diagnostic is missing {key}")
            reference = torch.from_numpy(np.asarray(prior_arrays[key])).to(
                device=prediction[field].device,
                dtype=prediction[field].dtype,
            )
            prediction_comparisons[path][field] = _tensor_difference(
                prediction[field],
                reference,
            )
        geometry_comparisons[path] = {}
        encoded = encoded_by_path[path]
        for name, getter in geometry_names.items():
            value = getter(encoded).detach().float()
            key = f"{prior_prefix}__{precision}_model_{path}_source_{name}"
            if key not in prior_arrays:
                raise KeyError(f"Prior diagnostic is missing {key}")
            reference = torch.from_numpy(np.asarray(prior_arrays[key])).to(
                device=value.device,
                dtype=value.dtype,
            )
            geometry_comparisons[path][name] = _tensor_difference(value, reference)
    passed = all(
        bool(prediction_comparisons[path][field]["exact"])
        for path in ("primary", "fixed")
        for field in PREDICTION_FIELDS
    ) and all(
        bool(geometry_comparisons[path][name]["exact"])
        for path in ("primary", "fixed")
        for name in geometry_names
    )
    arrays = {
        f"historical_{path}_{field}": prediction[field]
        for path, prediction in prediction_by_path.items()
        for field in PREDICTION_FIELDS
    }
    arrays.update(
        {
            f"historical_model_{path}_source_{name}": getter(encoded)
            for path, encoded in encoded_by_path.items()
            for name, getter in geometry_names.items()
        }
    )
    return {
        "job304002_primary_fixed_predictions": prediction_comparisons,
        "job304002_model_source_geometry": geometry_comparisons,
        "passed": passed,
    }, arrays


def _injected_geometry_exact(
    encoded: Any,
    bundle: CanonicalSourceBundle,
    mode: str,
) -> dict[str, bool]:
    checks = {
        "centroids": torch.equal(
            encoded.source_mesh.cell_centroids.detach().cpu(),
            bundle.centroids,
        ),
        "areas": torch.equal(
            encoded.source_mesh.cell_areas.detach().cpu(),
            bundle.areas,
        ),
        "normals": torch.equal(
            encoded.source_mesh.cell_normals.detach().cpu(),
            bundle.normals,
        ),
    }
    if mode == "canonical_full":
        checks["points"] = torch.equal(
            encoded.source_mesh.points.detach().cpu(),
            bundle.points,
        )
    return {name: bool(value) for name, value in checks.items()}


def _run_mode(
    runtime: Any,
    *,
    primary_domain: Any,
    fixed_domain: Any,
    bundle: CanonicalSourceBundle,
    mode: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    encoded_primary = _encode_with_canonical_geometry(
        runtime,
        primary_domain,
        bundle,
        mode,
    )
    primary, primary_decode_checks = _decode_at_canonical_centroids(
        runtime,
        encoded_primary,
        bundle,
    )
    encoded_fixed = _encode_with_canonical_geometry(
        runtime,
        fixed_domain,
        bundle,
        mode,
    )
    fixed, fixed_decode_checks = _decode_at_canonical_centroids(
        runtime,
        encoded_fixed,
        bundle,
    )
    encoded_replay = _encode_with_canonical_geometry(
        runtime,
        primary_domain,
        bundle,
        mode,
    )
    replay, replay_decode_checks = _decode_at_canonical_centroids(
        runtime,
        encoded_replay,
        bundle,
    )

    primary_fixed = _prediction_difference(primary, fixed)
    primary_replay = _prediction_difference(primary, replay)
    primary_injection = _injected_geometry_exact(encoded_primary, bundle, mode)
    fixed_injection = _injected_geometry_exact(encoded_fixed, bundle, mode)
    replay_passed = _difference_is_exact(primary_replay)
    injection_passed = all(primary_injection.values()) and all(fixed_injection.values())
    decode_checks = {
        "primary": primary_decode_checks,
        "fixed": fixed_decode_checks,
        "primary_replay": replay_decode_checks,
    }
    decode_passed = all(
        value
        for path_checks in decode_checks.values()
        for value in path_checks.values()
    )
    if mode == "canonical_derived":
        comparison_passed = _difference_within(
            primary_fixed,
            DERIVED_RELATIVE_TOLERANCE,
        )
    else:
        comparison_passed = _difference_is_exact(primary_fixed)
    summary = {
        "mode": mode,
        "primary_fixed_difference": primary_fixed,
        "primary_replay_difference": primary_replay,
        "primary_replay_exact": replay_passed,
        "injected_geometry_exact": {
            "primary": primary_injection,
            "fixed": fixed_injection,
        },
        "canonical_decode_contract": decode_checks,
        "canonical_decode_contract_passed": decode_passed,
        "comparison_gate": {
            "criterion": (
                "fieldwise_relative_l2_le_1e-3"
                if mode == "canonical_derived"
                else "fieldwise_bitwise_exact"
            ),
            "passed": comparison_passed,
        },
        "passed": (
            replay_passed and injection_passed and decode_passed and comparison_passed
        ),
    }
    arrays = {f"{mode}_primary_{field}": primary[field] for field in PREDICTION_FIELDS}
    arrays.update(
        {f"{mode}_fixed_{field}": fixed[field] for field in PREDICTION_FIELDS}
    )
    arrays.update(
        {f"{mode}_primary_replay_{field}": replay[field] for field in PREDICTION_FIELDS}
    )
    return summary, arrays


def _run_precision(
    runtime: Any,
    *,
    primary_domain: Any,
    fixed_domain: Any,
    bundle: CanonicalSourceBundle,
    precision: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    summaries: dict[str, Any] = {}
    arrays: dict[str, torch.Tensor] = {}
    with torch.no_grad(), runtime.autocast_context(precision):
        for mode in ("canonical_derived", "canonical_full"):
            summary, mode_arrays = _run_mode(
                runtime,
                primary_domain=primary_domain,
                fixed_domain=fixed_domain,
                bundle=bundle,
                mode=mode,
            )
            summaries[mode] = summary
            arrays.update(mode_arrays)
    return {
        "precision": precision,
        "modes": summaries,
        "passed": all(summary["passed"] for summary in summaries.values()),
    }, arrays


def _path_topology_checks(
    subset: Any,
    primary_domain: Any,
    fixed_domain: Any,
) -> dict[str, bool]:
    expected = subset.cells.detach().cpu()
    return {
        "primary_matches_selected": bool(
            torch.equal(
                primary_domain.boundaries["vehicle"].cells.detach().cpu(),
                expected,
            )
        ),
        "fixed_matches_selected": bool(
            torch.equal(
                fixed_domain.boundaries["vehicle"].cells.detach().cpu(),
                expected,
            )
        ),
        "primary_matches_fixed": bool(
            torch.equal(
                primary_domain.boundaries["vehicle"].cells.detach().cpu(),
                fixed_domain.boundaries["vehicle"].cells.detach().cpu(),
            )
        ),
    }


def _prior_geometry_replay(
    *,
    ids: np.ndarray,
    primary_domain: Any,
    fixed_domain: Any,
    prior_arrays: Mapping[str, np.ndarray],
    prior_prefix: str,
) -> dict[str, bool]:
    current = {
        "cell_ids_int64": np.asarray(ids, dtype="<i8"),
        "pipeline_primary_points_float32": (
            primary_domain.boundaries["vehicle"]
            .points.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
        ),
        "pipeline_fixed_points_float32": (
            fixed_domain.boundaries["vehicle"]
            .points.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
        ),
        "pipeline_primary_queries_float32": (
            primary_domain.interior.points.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
        ),
        "pipeline_fixed_queries_float32": (
            fixed_domain.interior.points.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
        ),
    }
    checks: dict[str, bool] = {}
    for name, value in current.items():
        key = f"{prior_prefix}__{name}"
        if key not in prior_arrays:
            raise KeyError(f"Prior diagnostic is missing {key}")
        checks[name] = bool(np.array_equal(value, prior_arrays[key]))
    return checks


def _run_case(
    hqc: Any,
    runtime: Any,
    spec: Any,
    prior_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"CANONICAL_CASE_START case={spec.case_id}", flush=True)
    raw_mesh = runtime.dataset.reader._load_sample(spec.reader_index)
    ids = hqc._cyclic_indices(
        spec.n_master_cells,
        spec.historical_start,
        RESOLUTION,
    )
    ids_10k = hqc._cyclic_indices(
        spec.n_master_cells,
        spec.historical_start,
        hqc.BASELINE_K,
    )
    subset = _target_free_subset(raw_mesh, ids, runtime.mesh_type)
    subset_10k = _target_free_subset(raw_mesh, ids_10k, runtime.mesh_type)

    # This construction deliberately precedes both historical CenterMesh paths.
    raw_canonical = _build_canonical_raw_geometry(subset)
    raw_canonical_replay = _build_canonical_raw_geometry(subset)
    physical_length = _nested_tensor_value(subset.global_data, "L_ref")

    fixed_center = hqc._pipeline_center_on_device(subset_10k, runtime.device)
    primary_with_placeholders, primary_center = hqc._apply_pipeline(
        runtime,
        subset,
        fixed_center=None,
    )
    fixed_with_placeholders, applied_fixed_center = hqc._apply_pipeline(
        runtime,
        subset,
        fixed_center=fixed_center,
    )
    if not torch.equal(applied_fixed_center, fixed_center):
        raise ValueError("Fixed center changed while applying historical pipeline")
    primary_domain = _strip_local_data(
        primary_with_placeholders,
        runtime.mesh_type,
    )
    fixed_domain = _strip_local_data(
        fixed_with_placeholders,
        runtime.mesh_type,
    )

    reference_key = runtime.model.reference_length_key
    if reference_key is None:
        raise ValueError("Canonical diagnostic requires an explicit reference length")
    primary_reference = _nested_tensor_value(
        primary_domain.global_data,
        reference_key,
    )
    fixed_reference = _nested_tensor_value(
        fixed_domain.global_data,
        reference_key,
    )
    if not torch.equal(primary_reference, fixed_reference):
        raise ValueError("Primary/fixed model reference lengths differ")
    bundle = _finish_canonical_bundle(
        raw_canonical,
        physical_length=physical_length,
        model_reference_length=primary_reference,
    )
    replay_bundle = _finish_canonical_bundle(
        raw_canonical_replay,
        physical_length=physical_length,
        model_reference_length=primary_reference,
    )
    replay_checks = _bundle_difference(bundle, replay_bundle)
    bundle_validity = _bundle_validity(
        bundle,
        expected_cells=subset.cells,
    )
    path_topology = _path_topology_checks(
        subset,
        primary_domain,
        fixed_domain,
    )
    prior_prefix = f"case_{spec.cohort_ordinal:02d}_{spec.case_id}"
    prior_geometry_replay = _prior_geometry_replay(
        ids=ids,
        primary_domain=primary_domain,
        fixed_domain=fixed_domain,
        prior_arrays=prior_arrays,
        prior_prefix=prior_prefix,
    )
    construction_passed = all(replay_checks.values())
    topology_passed = all(path_topology.values())
    precision_summaries: dict[str, Any] = {}
    precision_arrays: dict[str, torch.Tensor] = {}
    safe_to_run_model = bundle_validity["passed"] and topology_passed
    if safe_to_run_model:
        for precision in PRECISIONS:
            print(
                f"CANONICAL_PRECISION_START case={spec.case_id} precision={precision}",
                flush=True,
            )
            historical_replay, historical_arrays = _historical_replay(
                runtime,
                primary_domain=primary_domain,
                fixed_domain=fixed_domain,
                precision=precision,
                prior_arrays=prior_arrays,
                prior_prefix=prior_prefix,
            )
            summary, arrays = _run_precision(
                runtime,
                primary_domain=primary_domain,
                fixed_domain=fixed_domain,
                bundle=bundle,
                precision=precision,
            )
            summary["job304002_historical_replay"] = historical_replay
            summary["passed"] = summary["passed"] and historical_replay["passed"]
            precision_summaries[precision] = summary
            arrays.update(historical_arrays)
            precision_arrays.update(
                {f"{precision}_{name}": value for name, value in arrays.items()}
            )

    prior_geometry_passed = all(prior_geometry_replay.values())
    passed = (
        bundle_validity["passed"]
        and construction_passed
        and topology_passed
        and prior_geometry_passed
        and len(precision_summaries) == len(PRECISIONS)
        and all(summary["passed"] for summary in precision_summaries.values())
    )
    result = {
        "case_id": spec.case_id,
        "cohort_ordinal": int(spec.cohort_ordinal),
        "reader_index": int(spec.reader_index),
        "resolution": RESOLUTION,
        "canonical_frame": {
            "construction": (
                "raw selected coordinates promoted to float64; physical "
                "area-weighted center removed; coherent triangle geometry "
                "divided by L_ref*model_reference_length; one float32 cast"
            ),
            "physical_center_float64": [
                float(value) for value in bundle.physical_center.tolist()
            ],
            "physical_length": bundle.physical_length,
            "model_reference_length": bundle.model_reference_length,
            "effective_physical_length": (
                bundle.physical_length * bundle.model_reference_length
            ),
            "queries": "canonical_trace_centroids",
        },
        "historical_centers": {
            "primary_point_mean_float32": [
                float(value) for value in primary_center.detach().cpu().tolist()
            ],
            "fixed_s10000_point_mean_float32": [
                float(value) for value in fixed_center.detach().cpu().tolist()
            ],
        },
        "validity": {
            "canonical_bundle": bundle_validity,
            "canonical_construction_replay": replay_checks,
            "canonical_construction_replay_passed": construction_passed,
            "historical_path_topology": path_topology,
            "historical_path_topology_passed": topology_passed,
            "job304002_geometry_replay": prior_geometry_replay,
            "job304002_geometry_replay_passed": prior_geometry_passed,
            "model_local_data_stripped": True,
            "model_probes_executed": safe_to_run_model,
        },
        "precision_probes": precision_summaries,
        "passed": passed,
    }
    arrays_np: dict[str, np.ndarray] = {
        "selected_cell_ids_int64": np.asarray(ids, dtype="<i8"),
        "canonical_cells_int64": bundle.cells.numpy().astype("<i8", copy=False),
        "canonical_points_float32": bundle.points.numpy().astype("<f4", copy=False),
        "canonical_centroids_float32": bundle.centroids.numpy().astype(
            "<f4",
            copy=False,
        ),
        "canonical_areas_float32": bundle.areas.numpy().astype("<f4", copy=False),
        "canonical_normals_float32": bundle.normals.numpy().astype(
            "<f4",
            copy=False,
        ),
    }
    arrays_np.update(
        {
            name: value.detach().cpu().numpy().astype("<f4", copy=False)
            for name, value in precision_arrays.items()
        }
    )
    print(f"CANONICAL_CASE_DONE case={spec.case_id} passed={passed}", flush=True)
    return result, arrays_np


def _array_manifest(
    hqc: Any,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hqc._sha256_array(value),
        }
        for name, value in sorted(arrays.items())
    }


def _forbidden_artifact_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lower = str(key).lower()
            if any(token in lower for token in FORBIDDEN_ARTIFACT_KEY_TOKENS):
                found.append(path)
            found.extend(_forbidden_artifact_keys(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_artifact_keys(nested, f"{prefix}[{index}]"))
    return found


def _provenance(
    *,
    hqc: Any,
    args: argparse.Namespace,
    npz_path: Path,
) -> dict[str, Any]:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    source_paths = (
        Path("physicsnemo/experimental/nn/mesh_attention/model.py"),
        Path("physicsnemo/experimental/nn/mesh_attention/kernel_decoder.py"),
        Path("physicsnemo/mesh/mesh.py"),
        Path("physicsnemo/datapipes/transforms/mesh/transforms.py"),
    )
    return {
        "command": list(sys.argv),
        "diagnostic_script_path": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": hqc._sha256_file(Path(__file__).resolve()),
        "frozen_producer_path": str(args.producer),
        "frozen_producer_sha256": hqc._sha256_file(args.producer),
        "source_tree_manifest_sha256": hqc._source_tree_manifest_sha256(args.repo_root),
        "selected_source_files": {
            path.as_posix(): hqc._sha256_file(args.repo_root / path)
            for path in source_paths
        },
        "input_hashes": {
            "dataset_manifest": hqc._sha256_file(args.dataset_root / "manifest.json"),
            "dataset_config": hqc._sha256_file(args.dataset_config),
            "resolved_config": hqc._sha256_file(args.resolved_config),
            "model_checkpoint": hqc._sha256_file(
                args.checkpoint_dir / hqc.MODEL_FILENAME
            ),
            "normalization_stats": hqc._sha256_file(
                args.checkpoint_dir / hqc.NORM_STATS_FILENAME
            ),
            "historical_metrics": hqc._sha256_file(args.historical_metrics),
            "prior_diagnostic_json": hqc._sha256_file(args.prior_diagnostic_json),
            "prior_diagnostic_npz": hqc._sha256_file(args.prior_diagnostic_npz),
        },
        "npz_path": str(npz_path),
        "npz_sha256": hqc._sha256_file(npz_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hardware": {
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": [int(capability[0]), int(capability[1])],
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--prior-diagnostic-json", type=Path, required=True)
    parser.add_argument("--prior-diagnostic-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    for name in (
        "producer",
        "repo_root",
        "dataset_root",
        "dataset_config",
        "resolved_config",
        "checkpoint_dir",
        "historical_metrics",
        "prior_diagnostic_json",
        "prior_diagnostic_npz",
        "output_json",
        "output_npz",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for output in (args.output_json, args.output_npz):
        sidecar = output.with_name(f"{output.name}.sha256")
        if output.exists() or sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite output or sidecar: {output}")

    hqc = _load_producer(args.producer)
    producer_sha = hqc._sha256_file(args.producer)
    if producer_sha != EXPECTED_PRODUCER_SHA256:
        raise ValueError(
            f"Frozen producer changed: expected {EXPECTED_PRODUCER_SHA256}, "
            f"got {producer_sha}"
        )
    hqc._validate_frozen_inputs(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
        historical_metrics_path=args.historical_metrics,
    )
    runtime = hqc._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    hqc._validate_reader(runtime)
    hqc._require_sha256(
        args.prior_diagnostic_json,
        EXPECTED_PRIOR_DIAGNOSTIC_JSON_SHA256,
        "Prior corrected center diagnostic JSON",
    )
    hqc._require_sha256(
        args.prior_diagnostic_npz,
        EXPECTED_PRIOR_DIAGNOSTIC_NPZ_SHA256,
        "Prior corrected center diagnostic NPZ",
    )
    prior_summary = json.loads(args.prior_diagnostic_json.read_text())
    prior_contract = (
        prior_summary.get("schema_version") == 2
        and prior_summary.get("artifact_kind") == "hqc_center_cause_diagnostic"
        and prior_summary.get("status") == "PASSED_NONDECIDING_DIAGNOSTIC_VALIDITY"
        and tuple(prior_summary.get("scientific_scope", {}).get("case_ids", ()))
        == CASE_IDS
        and prior_summary.get("validity", {}).get("all_cases_and_precisions_passed")
        is True
    )
    if not prior_contract:
        raise ValueError("Prior corrected center diagnostic contract changed")
    with np.load(args.prior_diagnostic_npz, allow_pickle=False) as archive:
        prior_arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    specs = tuple(
        next(spec for spec in hqc.CASE_SPECS if spec.case_id == case_id)
        for case_id in CASE_IDS
    )
    if tuple(spec.case_id for spec in specs) != CASE_IDS:
        raise ValueError("Canonical diagnostic case order changed")

    cases: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}
    for spec in specs:
        case, arrays = _run_case(hqc, runtime, spec, prior_arrays)
        cases.append(case)
        prefix = f"case_{spec.cohort_ordinal:02d}_{spec.case_id}"
        npz_arrays.update(
            {f"{prefix}__{name}": value for name, value in arrays.items()}
        )
        print(
            f"COMPLETED_UNITS={len(cases)}/{len(specs)} case={spec.case_id}",
            flush=True,
        )

    forbidden_array_keys = [
        key
        for key in npz_arrays
        if any(token in key.lower() for token in FORBIDDEN_ARTIFACT_KEY_TOKENS)
    ]
    if forbidden_array_keys:
        raise ValueError(f"Forbidden NPZ keys: {forbidden_array_keys}")
    array_manifest = _array_manifest(hqc, npz_arrays)
    hqc._atomic_write_npz(args.output_npz, npz_arrays)
    hqc._write_sha256_sidecar(args.output_npz)

    all_valid = all(case["passed"] for case in cases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": PASSED_STATUS if all_valid else FAILED_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "case_ids": list(CASE_IDS),
            "resolution": RESOLUTION,
            "precisions": list(PRECISIONS),
            "supervision_arrays_indexed": False,
            "synthetic_placeholders_stripped_before_model": True,
            "hqc_decision_statistics_computed": False,
            "may_not_be_used_as_hqc_verdict_output": True,
        },
        "contract": {
            "canonical_construction": (
                "float64 raw geometry -> physical area center -> divide by "
                "L_ref*model_reference_length -> one float32 cast"
            ),
            "canonical_derived_fields": ["centroids", "areas", "normals"],
            "canonical_full_fields": [
                "points",
                "centroids",
                "areas",
                "normals",
            ],
            "query_frame": "canonical_trace_centroids",
            "derived_fieldwise_relative_tolerance": DERIVED_RELATIVE_TOLERANCE,
            "full_comparison": "fieldwise_bitwise_exact",
        },
        "validity": {
            "all_cases_and_precisions_passed": all_valid,
            "required_gates": [
                "canonical_construction_replay",
                "shape",
                "topology",
                "finite_positive_unit",
                "job304002_geometry_and_primary_fixed_prediction_replay",
                "primary_replay_exact",
                "canonical_derived_primary_fixed_fieldwise_relative_l2_le_1e-3",
                "canonical_full_primary_fixed_fieldwise_bitwise_exact",
            ],
        },
        "cases": cases,
        "npz_array_manifest": array_manifest,
        "provenance": _provenance(hqc=hqc, args=args, npz_path=args.output_npz),
    }
    forbidden_json_keys = _forbidden_artifact_keys(result)
    if forbidden_json_keys:
        raise ValueError(f"Forbidden JSON keys: {forbidden_json_keys}")
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    hqc._atomic_write_bytes(args.output_json, payload)
    hqc._write_sha256_sidecar(args.output_json)
    print(
        f"{result['status']} json={args.output_json} npz={args.output_npz}",
        flush=True,
    )
    if not all_valid:
        raise RuntimeError("Canonical geometry diagnostic failed a required gate")


if __name__ == "__main__":
    main()
