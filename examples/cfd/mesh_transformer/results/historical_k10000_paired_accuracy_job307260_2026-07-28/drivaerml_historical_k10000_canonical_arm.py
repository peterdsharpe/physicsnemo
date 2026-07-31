# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce one target-blind lane of the DrivAerML K=10k canonical arm.

The producer reconstructs the frozen 36-case geometry cohort without opening
any supervision path.  It constructs a coherent source geometry in float64,
removes the physical area-weighted center, applies the frozen physical and
model length gauges, and casts each floating geometry field to float32 once.
The public ``CanonicalSourceGeometry`` encode path is then evaluated in BF16
and decoded exactly once at the canonical cell centroids.

This process publishes observations and provenance only.  It receives no
legacy predictions, truths, metrics, decision thresholds, or categorical
outcome; the independent paired adjudicator owns all deciding comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_canonical_arm_producer"
STATUS = "COMPLETED_HISTORICAL_K10000_CANONICAL_ARM_PRODUCER"
RESOLUTION = 10_000
REQUESTED_EPOCH = 491
TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}
GLOBAL_FIELD_ORDER = (
    "U_inf_x",
    "U_inf_y",
    "U_inf_z",
    "p_inf",
    "rho_inf",
    "nu",
    "L_ref",
    "U_inf_dir_x",
    "U_inf_dir_y",
    "U_inf_dir_z",
    "reference_length",
)

LEGACY_SUPPORT_FILENAME = "drivaerml_historical_k10000_replay.py"
RUNTIME_HELPER_FILENAME = "drivaerml_historical_k10000_replay_runtime.py"
CANONICAL_HELPER_FILENAME = "drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
EXPECTED_LEGACY_SUPPORT_SHA256 = (
    "bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5"
)
EXPECTED_RUNTIME_HELPER_SHA256 = (
    "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
)
EXPECTED_CANONICAL_HELPER_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)

PAIRING_CONTROL_SUFFIXES = (
    "selected_cell_ids_int64",
    "compacted_cells_int64",
    "raw_points_float32",
    "raw_centroids_float32",
    "native_normals_float32",
    "native_areas_float64",
    "pipeline_boundary_points_float32",
    "pipeline_queries_float32",
    "pipeline_normals_float32",
    "pipeline_globals_float32",
    "pipeline_center_float32",
)
CANONICAL_GEOMETRY_SUFFIXES = (
    "canonical_cells_int64",
    "canonical_points_float32",
    "canonical_centroids_float32",
    "canonical_areas_float32",
    "canonical_normals_float32",
    "canonical_physical_center_float64",
    "canonical_queries_float32",
)
PREDICTION_SUFFIXES = (
    "prediction_pressure_training_float32",
    "prediction_wss_training_float32",
    "prediction_pressure_physical_float32",
    "prediction_wss_physical_float32",
)
CASE_ARRAY_SUFFIXES = (
    *PAIRING_CONTROL_SUFFIXES,
    *CANONICAL_GEOMETRY_SUFFIXES,
    *PREDICTION_SUFFIXES,
)
EXPECTED_SINGLE_RANK_ENVIRONMENT = {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
}


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_module(
    path: Path,
    *,
    expected_sha256: str,
    module_name: str,
    label: str,
) -> ModuleType:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 differs: expected {expected_sha256}, got {observed}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {label} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_support_modules(
    script_path: Path,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    directory = script_path.parent
    legacy = _load_verified_module(
        directory / LEGACY_SUPPORT_FILENAME,
        expected_sha256=EXPECTED_LEGACY_SUPPORT_SHA256,
        module_name="frozen_historical_k10000_replay_support",
        label="Frozen historical replay producer support",
    )
    runtime = _load_verified_module(
        directory / RUNTIME_HELPER_FILENAME,
        expected_sha256=EXPECTED_RUNTIME_HELPER_SHA256,
        module_name="frozen_historical_k10000_replay_runtime",
        label="Frozen historical replay runtime",
    )
    canonical = _load_verified_module(
        directory / CANONICAL_HELPER_FILENAME,
        expected_sha256=EXPECTED_CANONICAL_HELPER_SHA256,
        module_name="frozen_canonical_geometry_helper_v5",
        label="Frozen canonical geometry helper",
    )
    return legacy, runtime, canonical


def _manifest_global_values(case: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = case.get("global_input_values_float32")
    expected = {"U_inf", "p_inf", "rho_inf", "nu", "L_ref"}
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError(f"Frozen global inputs changed for {case.get('case_id')}")
    result = {
        name: np.asarray(values[name], dtype="<f4")
        for name in ("U_inf", "p_inf", "rho_inf", "nu", "L_ref")
    }
    if (
        result["U_inf"].shape != (3,)
        or any(result[name].shape != (1,) for name in expected - {"U_inf"})
        or not all(np.isfinite(value).all() for value in result.values())
    ):
        raise ValueError(
            f"Frozen global input shapes changed for {case.get('case_id')}"
        )
    return {
        "U_inf": result["U_inf"],
        **{
            name: result[name].reshape(())
            for name in ("p_inf", "rho_inf", "nu", "L_ref")
        },
    }


def _load_geometry_only_subset(
    legacy: ModuleType,
    runtime: Any,
    dataset_root: Path,
    spec: Any,
    geometry_case: Mapping[str, Any],
) -> tuple[Any, dict[str, np.ndarray]]:
    """Read only frozen cells, referenced points, and allowlisted globals."""
    case_root = (dataset_root / spec.case_id).resolve(strict=True)
    if geometry_case.get("case_id") != spec.case_id or geometry_case.get(
        "resolved_case_root"
    ) != str(case_root):
        raise ValueError(f"Frozen case-root identity changed for {spec.case_id}")

    mesh_relative = (
        f"domain_{spec.case_id}.pdmsh/_tensordict/boundaries/vehicle/_tensordict"
    )
    cells_relative = f"{mesh_relative}/cells.memmap"
    points_relative = f"{mesh_relative}/points.memmap"
    cells_record = legacy._geometry_file_record(geometry_case, cells_relative)
    points_record = legacy._geometry_file_record(geometry_case, points_relative)
    selected_ids = np.arange(
        spec.historical_start,
        spec.historical_start + RESOLUTION,
        dtype=np.int64,
    )
    selected_cells_payload = legacy._safe_hashed_pread(
        case_root / cells_relative,
        expected_file_size=int(cells_record["size_bytes"]),
        expected_file_sha256=str(cells_record["sha256"]),
        offset=spec.historical_start * 3 * 8,
        count=RESOLUTION * 3 * 8,
    )
    selected_cells = np.frombuffer(
        selected_cells_payload,
        dtype="<i8",
    ).reshape(RESOLUTION, 3)
    n_master_points = int(geometry_case.get("n_master_points", -1))
    if (
        n_master_points <= 0
        or int(selected_cells.min()) < 0
        or int(selected_cells.max()) >= n_master_points
    ):
        raise ValueError(f"Selected connectivity is invalid for {spec.case_id}")
    referenced, inverse = np.unique(
        selected_cells.reshape(-1),
        return_inverse=True,
    )
    compacted_cells = inverse.reshape(RESOLUTION, 3).astype("<i8", copy=False)
    points = legacy._safe_hashed_rows(
        case_root / points_relative,
        expected_file_size=int(points_record["size_bytes"]),
        expected_file_sha256=str(points_record["sha256"]),
        n_rows=n_master_points,
        row_indices=referenced,
    )
    globals_float32 = _manifest_global_values(geometry_case)
    raw_mesh = runtime.mesh_type(
        points=torch.from_numpy(points),
        cells=torch.from_numpy(compacted_cells),
        cell_data={
            "pMeanTrim": torch.zeros(RESOLUTION, dtype=torch.float32),
            "wallShearStressMeanTrim": torch.zeros(
                (RESOLUTION, 3),
                dtype=torch.float32,
            ),
        },
        global_data={
            name: torch.from_numpy(value) for name, value in globals_float32.items()
        },
    )
    if "_measure_weights" in raw_mesh.cell_data.keys():
        raise ValueError(
            "Geometry-only reconstructed boundary unexpectedly carries measure weights"
        )
    return raw_mesh, {
        "selected_cell_ids_int64": selected_ids.astype("<i8", copy=False),
        "compacted_cells_int64": compacted_cells,
        "raw_points_float32": points.astype("<f4", copy=False),
    }


def _canonical_geometry_for_domain(
    runtime: Any,
    domain: Any,
    bundle: Any,
) -> Any:
    from physicsnemo.experimental.nn.mesh_attention import CanonicalSourceGeometry
    from physicsnemo.experimental.nn.mesh_attention import model as model_module

    if CanonicalSourceGeometry is not model_module.CanonicalSourceGeometry:
        raise ValueError("Public CanonicalSourceGeometry export has split identity")
    first_boundary = domain.boundaries[runtime.model.boundary_names[0]]
    device = first_boundary.points.device
    dtype = first_boundary.points.dtype
    return CanonicalSourceGeometry(
        points=bundle.points.to(device=device, dtype=dtype),
        cells=bundle.cells.to(
            device=first_boundary.cells.device,
            dtype=first_boundary.cells.dtype,
        ),
        centroids=bundle.centroids.to(device=device, dtype=dtype),
        areas=bundle.areas.to(device=device, dtype=dtype),
        normals=bundle.normals.to(device=device, dtype=dtype),
        center=torch.zeros(
            first_boundary.n_spatial_dims,
            device=device,
            dtype=dtype,
        ),
        reference_length=torch.ones((), device=device, dtype=dtype),
    )


def _authoritative_cache_checks(
    canonical: ModuleType,
    encoded: Any,
    geometry: Any,
    bundle: Any,
) -> dict[str, Any]:
    exact = canonical._injected_geometry_exact(encoded, bundle, "canonical_full")
    exact["cells"] = canonical._tensor_bitwise_equal(
        encoded.source_mesh.cells.detach().cpu(),
        bundle.cells,
    )
    identity = {
        "points": encoded.source_mesh.points.data_ptr() == geometry.points.data_ptr(),
        "cells": encoded.source_mesh.cells.data_ptr() == geometry.cells.data_ptr(),
        "centroids": (
            encoded.source_mesh.cell_centroids.data_ptr()
            == geometry.centroids.data_ptr()
        ),
        "areas": (
            encoded.source_mesh.cell_areas.data_ptr() == geometry.areas.data_ptr()
        ),
        "normals": (
            encoded.source_mesh.cell_normals.data_ptr() == geometry.normals.data_ptr()
        ),
        "center": encoded.center.data_ptr() == geometry.center.data_ptr(),
        "reference_length": (
            encoded.reference_length.data_ptr() == geometry.reference_length.data_ptr()
        ),
    }
    passed = all(exact.values()) and all(identity.values())
    return {
        "cache_values_bitwise_exact": exact,
        "cache_values_bitwise_exact_passed": all(exact.values()),
        "authoritative_storage_identity": identity,
        "authoritative_storage_identity_passed": all(identity.values()),
        "passed": passed,
    }


def _prediction_tensors(
    runtime: Any,
    output: Any,
    redim: Mapping[str, Any],
    domain: Any,
) -> dict[str, np.ndarray]:
    prediction = runtime.normalize_output(
        output,
        TARGET_CONFIG,
        str(runtime.cfg.output_type),
    ).float()
    expected_shapes = {
        "pressure": (RESOLUTION,),
        "wss": (RESOLUTION, 3),
    }
    for name, expected_shape in expected_shapes.items():
        value = prediction[name]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} prediction has shape {tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} prediction contains non-finite values")
    physical = redim["function"](
        prediction,
        normalizer=redim["normalizer"],
        nondim=redim["nondim"],
        field_types=redim["field_types"],
        global_data=domain.global_data,
    )
    result = {
        "prediction_pressure_training_float32": (
            prediction["pressure"].detach().cpu().numpy().astype("<f4", copy=False)
        ),
        "prediction_wss_training_float32": (
            prediction["wss"].detach().cpu().numpy().astype("<f4", copy=False)
        ),
        "prediction_pressure_physical_float32": (
            physical["pressure"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                "<f4",
                copy=False,
            )
        ),
        "prediction_wss_physical_float32": (
            physical["wss"].detach().float().cpu().numpy().astype("<f4", copy=False)
        ),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Re-dimensionalized canonical prediction is non-finite")
    return result


def _run_canonical_forward(
    canonical: ModuleType,
    runtime: Any,
    domain: Any,
    bundle: Any,
    redim: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    geometry = _canonical_geometry_for_domain(runtime, domain, bundle)
    with torch.no_grad(), runtime.autocast_context("bfloat16"):
        encoded = runtime.model.encode(
            domain,
            canonical_source_geometry=geometry,
        )
        query_mesh = runtime.mesh_type(points=geometry.centroids)
        output = runtime.model.decode(encoded, query_mesh)

    cache_checks = _authoritative_cache_checks(
        canonical,
        encoded,
        geometry,
        bundle,
    )
    decode_checks = {
        "canonical_queries_equal_canonical_centroids_raw_byte_exact": (
            canonical._tensor_bitwise_equal(
                query_mesh.points.detach().cpu(),
                bundle.centroids,
            )
        ),
        "canonical_query_storage_identity": (
            query_mesh.points.data_ptr() == geometry.centroids.data_ptr()
        ),
        "encoded_center_is_raw_positive_zero": bool(
            torch.all(encoded.center == 0.0)
            and torch.all(~torch.signbit(encoded.center))
        ),
        "encoded_reference_length_is_exact_positive_one": bool(
            encoded.reference_length == 1.0
            and not torch.signbit(encoded.reference_length)
        ),
        "trace_query_count_exact": (
            encoded.trace_slice is not None
            and encoded.trace_slice.stop - encoded.trace_slice.start == RESOLUTION
        ),
    }
    if not cache_checks["passed"] or not all(decode_checks.values()):
        raise ValueError(
            "Canonical public encode/decode failed an authoritative geometry gate"
        )
    arrays = _prediction_tensors(runtime, output, redim, domain)
    canonical_queries = (
        query_mesh.points.detach().float().cpu().numpy().astype("<f4", copy=False)
    )
    return (
        arrays,
        {
            "authoritative_cache": cache_checks,
            "decode_contract": decode_checks,
            "decode_contract_passed": all(decode_checks.values()),
        },
        canonical_queries,
    )


def _npz_prefix(ordinal: int, case_id: str) -> str:
    return f"case_{ordinal:02d}_{case_id}"


def _validate_single_rank_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in EXPECTED_SINGLE_RANK_ENVIRONMENT}
    if observed != EXPECTED_SINGLE_RANK_ENVIRONMENT:
        raise ValueError(
            "Canonical arm requires one torchrun rank: "
            f"expected={EXPECTED_SINGLE_RANK_ENVIRONMENT} observed={observed}"
        )
    return EXPECTED_SINGLE_RANK_ENVIRONMENT.copy()


def _run_case(
    legacy: ModuleType,
    runtime_support: ModuleType,
    canonical: ModuleType,
    runtime: Any,
    redim: Mapping[str, Any],
    dataset_root: Path,
    spec: Any,
    geometry_case: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    print(
        f"CANONICAL_ARM_CASE_START ordinal={spec.cohort_ordinal} case={spec.case_id}",
        flush=True,
    )
    raw_mesh, input_arrays = _load_geometry_only_subset(
        legacy,
        runtime,
        dataset_root,
        spec,
        geometry_case,
    )
    raw_centroids, native_normals, native_areas = runtime_support._native_geometry(
        raw_mesh
    )
    raw_canonical = canonical._build_canonical_raw_geometry(raw_mesh)
    raw_canonical_replay = canonical._build_canonical_raw_geometry(raw_mesh)
    physical_length = canonical._nested_tensor_value(raw_mesh.global_data, "L_ref")

    domain_with_placeholders, pipeline_center = runtime_support._apply_pipeline(
        runtime,
        raw_mesh,
        fixed_center=None,
    )
    boundary = domain_with_placeholders.boundaries["vehicle"]
    if tuple(boundary.cells.shape) != (RESOLUTION, 3) or tuple(
        domain_with_placeholders.interior.points.shape
    ) != (RESOLUTION, 3):
        raise ValueError(f"Pipeline topology changed for {spec.case_id}")
    if "_measure_weights" in boundary.cell_data.keys():
        raise ValueError(
            f"Canonical arm boundary carries measure weights for {spec.case_id}"
        )
    pipeline_cells = boundary.cells.detach().cpu().numpy().astype("<i8", copy=False)
    if not np.array_equal(pipeline_cells, input_arrays["compacted_cells_int64"]):
        raise ValueError(f"Pipeline connectivity changed for {spec.case_id}")

    domain = canonical._strip_local_data(
        domain_with_placeholders,
        runtime.mesh_type,
    )
    reference_key = runtime.model.reference_length_key
    if reference_key is None:
        raise ValueError("Canonical arm requires an explicit model reference length")
    model_reference_length = canonical._nested_tensor_value(
        domain.global_data,
        reference_key,
    )
    bundle = canonical._finish_canonical_bundle(
        raw_canonical,
        physical_length=physical_length,
        model_reference_length=model_reference_length,
    )
    replay_bundle = canonical._finish_canonical_bundle(
        raw_canonical_replay,
        physical_length=physical_length,
        model_reference_length=model_reference_length,
    )
    construction_replay = canonical._bundle_difference(bundle, replay_bundle)
    construction_replay_passed = all(construction_replay.values())
    bundle_validity = canonical._bundle_validity(
        bundle,
        expected_cells=boundary.cells,
    )
    if not construction_replay_passed or not bundle_validity["passed"]:
        raise ValueError(f"Canonical construction failed for {spec.case_id}")

    predictions, public_api_validity, canonical_queries = _run_canonical_forward(
        canonical,
        runtime,
        domain,
        bundle,
        redim,
    )
    pipeline_boundary_points = (
        boundary.points.detach().float().cpu().numpy().astype("<f4", copy=False)
    )
    pipeline_queries = (
        domain_with_placeholders.interior.points.detach()
        .float()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
    )
    pipeline_normals = (
        boundary.cell_data["normals"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
    )
    pipeline_globals = legacy._pipeline_globals_float32(domain_with_placeholders)
    case_arrays = {
        "selected_cell_ids_int64": input_arrays["selected_cell_ids_int64"],
        "compacted_cells_int64": pipeline_cells,
        "raw_points_float32": input_arrays["raw_points_float32"],
        "raw_centroids_float32": raw_centroids.astype("<f4", copy=False),
        "native_normals_float32": native_normals.astype("<f4", copy=False),
        "native_areas_float64": native_areas.astype("<f8", copy=False),
        "pipeline_boundary_points_float32": pipeline_boundary_points,
        "pipeline_queries_float32": pipeline_queries,
        "pipeline_normals_float32": pipeline_normals,
        "pipeline_globals_float32": pipeline_globals,
        "pipeline_center_float32": (
            pipeline_center.detach().float().cpu().numpy().astype("<f4", copy=False)
        ),
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
        "canonical_physical_center_float64": (
            bundle.physical_center.numpy().astype("<f8", copy=False)
        ),
        "canonical_queries_float32": canonical_queries,
        **predictions,
    }
    if tuple(case_arrays) != CASE_ARRAY_SUFFIXES:
        raise RuntimeError("Canonical arm case-array schema changed")
    prefix = _npz_prefix(spec.cohort_ordinal, spec.case_id)
    for name, value in case_arrays.items():
        arrays[f"{prefix}__{name}"] = np.ascontiguousarray(value)
    array_hashes = {
        name: legacy._array_sha256(value, value.dtype)
        for name, value in case_arrays.items()
    }
    print(
        f"CANONICAL_ARM_CASE_DONE ordinal={spec.cohort_ordinal} case={spec.case_id}",
        flush=True,
    )
    return {
        "cohort_ordinal": spec.cohort_ordinal,
        "case_id": spec.case_id,
        "reader_index": spec.reader_index,
        "n_master_cells": spec.n_master_cells,
        "historical_start": spec.historical_start,
        "resolution": RESOLUTION,
        "n_compacted_points": int(pipeline_boundary_points.shape[0]),
        "canonical_frame": {
            "construction": (
                "raw selected coordinates promoted to float64; physical "
                "area-weighted center removed; coherent triangle geometry "
                "divided by L_ref*model_reference_length; one float32 cast"
            ),
            "physical_length": bundle.physical_length,
            "model_reference_length": bundle.model_reference_length,
            "query_frame": "canonical_trace_centroids",
        },
        "validity": {
            "geometry_only_input": True,
            "synthetic_placeholders_stripped_before_model": True,
            "measure_weights_absent": True,
            "canonical_bundle": bundle_validity,
            "canonical_construction_replay": construction_replay,
            "canonical_construction_replay_passed": construction_replay_passed,
            "public_api": public_api_validity,
            "passed": (
                bundle_validity["passed"]
                and construction_replay_passed
                and public_api_validity["authoritative_cache"]["passed"]
                and public_api_validity["decode_contract_passed"]
            ),
        },
        "array_sha256": array_hashes,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--lane-label", choices=("A", "B"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve_regular_input(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _resolve_directory_input(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a valid non-symlink directory: {path}")
    return path.resolve(strict=True)


def _provenance(
    legacy: ModuleType,
    runtime_support: ModuleType,
    canonical: ModuleType,
    *,
    args: argparse.Namespace,
    runtime: Any,
    static_inputs: Mapping[str, str],
    geometry_verification: Mapping[str, Any],
    import_provenance: Mapping[str, str],
    rank_environment: Mapping[str, str],
    npz_sha256: str,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve(strict=True)
    support_dir = script_path.parent
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    checkpoint_hashes = {
        "model": legacy._sha256_file(
            args.checkpoint_dir / runtime_support.MODEL_FILENAME
        ),
        "training_state": legacy._sha256_file(
            args.checkpoint_dir / runtime_support.TRAINING_STATE_FILENAME
        ),
        "normalization": legacy._sha256_file(
            args.checkpoint_dir / runtime_support.NORM_STATS_FILENAME
        ),
    }
    return {
        "command": list(sys.argv),
        "producer_path": str(script_path),
        "producer_sha256": legacy._sha256_file(script_path),
        "verified_support": {
            "legacy_replay_producer": {
                "path": str(support_dir / LEGACY_SUPPORT_FILENAME),
                "sha256": EXPECTED_LEGACY_SUPPORT_SHA256,
                "loaded_module": legacy.__name__,
            },
            "runtime": {
                "path": str(support_dir / RUNTIME_HELPER_FILENAME),
                "sha256": EXPECTED_RUNTIME_HELPER_SHA256,
                "loaded_module": runtime_support.__name__,
            },
            "canonical_geometry": {
                "path": str(support_dir / CANONICAL_HELPER_FILENAME),
                "sha256": EXPECTED_CANONICAL_HELPER_SHA256,
                "loaded_module": canonical.__name__,
            },
        },
        "repo_root": str(args.repo_root),
        "dataset_root": str(args.dataset_root),
        "static_inputs": dict(static_inputs),
        "geometry_verification": dict(geometry_verification),
        "import_provenance": dict(import_provenance),
        "checkpoint_sha256": checkpoint_hashes,
        "requested_epoch": REQUESTED_EPOCH,
        "loaded_epoch": int(runtime.loaded_epoch),
        "process": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cuda_visible_devices_token": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "rank_environment": dict(rank_environment),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "slurm_procid": os.environ.get("SLURM_PROCID"),
            "slurm_localid": os.environ.get("SLURM_LOCALID"),
        },
        "npz_sha256": npz_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hardware": {
            "device_index": int(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": [int(capability[0]), int(capability[1])],
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("Canonical arm requires a little-endian host")
    args = _parse_args(argv)
    args.repo_root = _resolve_directory_input(args.repo_root, "Repository root")
    args.dataset_root = _resolve_directory_input(args.dataset_root, "Dataset root")
    args.checkpoint_dir = _resolve_directory_input(
        args.checkpoint_dir,
        "Checkpoint directory",
    )
    for name in (
        "dataset_config",
        "resolved_config",
        "geometry_manifest",
    ):
        setattr(
            args,
            name,
            _resolve_regular_input(getattr(args, name), name.replace("_", " ")),
        )
    args.output_json = Path(os.path.abspath(args.output_json))
    args.output_npz = Path(os.path.abspath(args.output_npz))

    script_path = Path(__file__).resolve(strict=True)
    legacy, runtime_support, canonical = _load_support_modules(script_path)
    rank_environment = _validate_single_rank_environment()
    legacy._validate_output_targets(args.output_json, args.output_npz)
    legacy._validate_case_specs(runtime_support)
    static_inputs = legacy._validate_static_inputs(
        runtime_support,
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config=args.dataset_config,
        resolved_config=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    geometry_verification = legacy._verify_geometry_manifest(
        runtime_support,
        args.geometry_manifest,
        args.dataset_root,
    )
    geometry_cases = geometry_verification.pop("case_records")

    runtime = runtime_support._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    if int(runtime.loaded_epoch) != REQUESTED_EPOCH:
        raise ValueError(
            f"Loaded epoch {runtime.loaded_epoch}, expected {REQUESTED_EPOCH}"
        )
    if str(runtime.cfg.precision) != "bfloat16":
        raise ValueError(f"Canonical arm precision changed: {runtime.cfg.precision}")
    import_provenance = legacy._validate_import_provenance(args.repo_root)
    legacy._validate_reader(runtime)
    redim = legacy._redimensionalization_context(runtime, args.dataset_config)

    arrays: dict[str, np.ndarray] = {}
    cases = [
        _run_case(
            legacy,
            runtime_support,
            canonical,
            runtime,
            redim,
            args.dataset_root,
            spec,
            geometry_case,
            arrays,
        )
        for spec, geometry_case in zip(
            runtime_support.CASE_SPECS,
            geometry_cases,
            strict=True,
        )
    ]
    npz_temporary, npz_sha256 = legacy._prepare_npz_temporary(
        args.output_npz,
        arrays,
    )
    try:
        expected_array_count = len(runtime_support.CASE_SPECS) * len(
            CASE_ARRAY_SUFFIXES
        )
        if (
            len(cases) != len(runtime_support.CASE_SPECS)
            or len(arrays) != expected_array_count
            or not all(case["validity"]["passed"] for case in cases)
        ):
            raise RuntimeError(
                "Canonical arm output coverage/validity changed: "
                f"cases={len(cases)} arrays={len(arrays)}"
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lane_label": args.lane_label,
            "contract": {
                "arm": "canonical_source_geometry",
                "public_api": (
                    "model.encode(domain, canonical_source_geometry=geometry); "
                    "model.decode(encoded, canonical_centroid_query_mesh)"
                ),
                "canonical_source_geometry_present": True,
                "producer_reads_supervision_archive_metrics_or_ceilings": False,
                "resolution": RESOLUTION,
                "precision": "bfloat16",
                "torch_compile": False,
                "requested_epoch": REQUESTED_EPOCH,
                "case_count": len(cases),
                "canonical_construction": (
                    "float64 raw geometry -> physical area center -> divide by "
                    "L_ref*model_reference_length -> one float32 cast"
                ),
                "query_frame": "canonical_trace_centroids",
                "encode_count_per_case": 1,
                "decode_count_per_case": 1,
                "global_field_order": list(GLOBAL_FIELD_ORDER),
                "pairing_control_suffixes": list(PAIRING_CONTROL_SUFFIXES),
                "canonical_geometry_suffixes": list(CANONICAL_GEOMETRY_SUFFIXES),
                "prediction_suffixes": list(PREDICTION_SUFFIXES),
                "measure_weights_required_absent": True,
                "synthetic_placeholders_stripped_before_model": True,
            },
            "summary": {
                "case_count": len(cases),
                "array_count": len(arrays),
                "valid_case_count": sum(case["validity"]["passed"] for case in cases),
            },
            "cases": cases,
            "npz": {
                "filename": args.output_npz.name,
                "sha256": npz_sha256,
                "array_count": len(arrays),
                "array_manifest": legacy._array_manifest(arrays),
            },
            "provenance": _provenance(
                legacy,
                runtime_support,
                canonical,
                args=args,
                runtime=runtime,
                static_inputs=static_inputs,
                geometry_verification=geometry_verification,
                import_provenance=import_provenance,
                rank_environment=rank_environment,
                npz_sha256=npz_sha256,
            ),
        }
        payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        json_sha256 = legacy._publish_output_set(
            output_json=args.output_json,
            json_payload=payload,
            output_npz=args.output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )
    finally:
        npz_temporary.unlink(missing_ok=True)
    print(
        f"{STATUS} lane={args.lane_label} "
        f"json_sha256={json_sha256} npz_sha256={npz_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
