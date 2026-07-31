# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the target-free full-cohort canonical-geometry validity experiment.

This is the preregistered successor to the adjudicated four-canary schema-v5
diagnostic.  It leaves that instrument unchanged and loads its geometry/model
helpers only after verifying their exact source hash.  Four deterministic
lanes cover the frozen 36-case ID-reference cohort at all five nested source
resolutions.

For every case, resolution, and precision, the script constructs one neutral
cast-once canonical source bundle and supplies it symmetrically to the
historical primary- and fixed-center paths.  It decodes both on the coupled
``S_K`` trace, then scores both the whole trace and the fixed row-identity
prefix ``Q=S_2500``.  It does not issue a standalone 2,500-query decode at
larger ``K`` because the frozen model declares ``trace_of='vehicle'`` and
therefore requires one query per encoded trace cell.  The coherent public
``canonical_full`` API is the sole implementation candidate and must be
raw-byte exact on both scored panels.  The earlier private
``canonical_derived`` intervention is intentionally excluded: it is useful
only as a post-failure diagnostic and cannot license implementation.

The script never indexes a raw supervision array and computes no
truth-relative, force, area-objective, H-QC eligibility, support, futility, or
mixed statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import physicsnemo
import physicsnemo.experimental.nn.mesh_attention.model as mesh_attention_model
from physicsnemo.experimental.nn.mesh_attention import CanonicalSourceGeometry

SCHEMA_VERSION = 1
ARTIFACT_KIND = "hqc_canonical_geometry_full_cohort_validity_lane"
VALID_STATUS = "VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_LANE"
INVALID_STATUS = "INVALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_LANE"
CANONICAL_FULL_OUTCOME = "CANONICAL_FULL_VALIDITY_PASS"
CANONICAL_REPAIR_REFUTED = "CANONICAL_FULL_VALIDITY_REFUTED"
INVALID_DIAGNOSTIC = "INVALID_DIAGNOSTIC"
ANCHOR_FULL_AND_DERIVED_OUTCOME = "FULL_AND_DERIVED_PASS"

EXPECTED_HELPER_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)
EXPECTED_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_ANCHOR_JSON_SHA256 = (
    "09e336442881f0641c14c91c17dd80ac440d474f996925cf79f2174bc4cacd88"
)
EXPECTED_ANCHOR_NPZ_SHA256 = (
    "edee836e0cc5c66690276e6787496cbd6b81fb08decb7d54ecf1f36b333ddc9f"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
)
EXPECTED_DATASET_CONFIG_SHA256 = (
    "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
)
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
)
EXPECTED_MODEL_SHA256 = (
    "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
)
EXPECTED_TRAINING_STATE_SHA256 = (
    "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
)
# Filled only after the target-free geometry manifest and the opt-in
# implementation are independently verified; empty values fail closed.
EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256 = (
    "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
)
EXPECTED_EXECUTION_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)

CASE_IDS = (
    "run_118",
    "run_129",
    "run_145",
    "run_149",
    "run_17",
    "run_171",
    "run_18",
    "run_183",
    "run_197",
    "run_202",
    "run_225",
    "run_270",
    "run_271",
    "run_298",
    "run_305",
    "run_320",
    "run_367",
    "run_380",
    "run_382",
    "run_399",
    "run_4",
    "run_409",
    "run_419",
    "run_424",
    "run_429",
    "run_431",
    "run_439",
    "run_465",
    "run_468",
    "run_469",
    "run_478",
    "run_489",
    "run_490",
    "run_495",
    "run_71",
    "run_86",
)
ANCHOR_CASE_IDS = CASE_IDS[:4]
RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
PRECISIONS = ("bfloat16", "float32")
QUERY_PANELS = ("coupled_s_k", "fixed_id_prefix_s2500")
FIXED_QUERY_K = 2_500
LANE_COUNT = 4
PREDICTION_FIELDS = ("pressure", "wss")
ALLOWED_RAW_GLOBAL_FIELDS = ("U_inf", "p_inf", "rho_inf", "nu", "L_ref")
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
FORBIDDEN_INPUT_PATH_PARTS = {
    "cell_data",
    "point_data",
    "interior",
    "pmeantrim",
    "wallshearstressmeantrim",
    "pressure",
    "wss",
}


def _expected_geometry_manifest_paths(case_id: str) -> set[str]:
    domain = Path(f"domain_{case_id}.pdmsh")
    root = domain / "_tensordict"
    vehicle = root / "boundaries" / "vehicle"
    vehicle_td = vehicle / "_tensordict"
    global_data = root / "global_data"
    return {
        path.as_posix()
        for path in (
            domain / "meta.json",
            root / "meta.json",
            root / "boundaries" / "meta.json",
            vehicle / "meta.json",
            vehicle_td / "meta.json",
            vehicle_td / "points.memmap",
            vehicle_td / "cells.memmap",
            global_data / "meta.json",
            *(global_data / f"{name}.memmap" for name in ALLOWED_RAW_GLOBAL_FIELDS),
        )
    }


def _target_free_subset(
    mesh: Any,
    cell_ids: np.ndarray,
    mesh_type: Any,
) -> Any:
    """Compact geometry while reading only an explicit global-data allowlist."""
    ids = torch.as_tensor(np.asarray(cell_ids, dtype=np.int64), dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("cell_ids must be a non-empty vector")
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= mesh.n_cells:
        raise ValueError("cell_ids contain an out-of-range row")
    missing_globals = [
        name for name in ALLOWED_RAW_GLOBAL_FIELDS if name not in mesh.global_data
    ]
    if missing_globals:
        raise KeyError(f"Raw mesh lacks allowed global inputs: {missing_globals}")
    selected_cells = mesh.cells[ids]
    referenced, inverse = torch.unique(
        selected_cells,
        sorted=True,
        return_inverse=True,
    )
    compacted_cells = inverse.reshape_as(selected_cells)
    n_cells = int(ids.numel())
    placeholders = {
        "pMeanTrim": mesh.points.new_zeros(n_cells),
        "wallShearStressMeanTrim": mesh.points.new_zeros(
            n_cells,
            mesh.n_spatial_dims,
        ),
    }
    return mesh_type(
        points=mesh.points[referenced],
        cells=compacted_cells,
        point_data={},
        cell_data=placeholders,
        global_data={
            name: mesh.global_data[name] for name in ALLOWED_RAW_GLOBAL_FIELDS
        },
    )


def _target_free_file_subset(
    dataset_root: Path,
    spec: Any,
    cell_ids: np.ndarray,
    mesh_type: Any,
) -> Any:
    """Read one compact subset through the frozen geometry-only file allowlist."""
    ids = np.asarray(cell_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("cell_ids must be a non-empty vector")
    if int(ids.min()) < 0 or int(ids.max()) >= spec.n_master_cells:
        raise ValueError("cell_ids contain an out-of-range row")

    case_root = (dataset_root / spec.case_id).resolve(strict=True)
    tensor_root = case_root / f"domain_{spec.case_id}.pdmsh" / "_tensordict"
    vehicle_root = tensor_root / "boundaries" / "vehicle" / "_tensordict"
    global_root = tensor_root / "global_data"
    vehicle_meta = _strict_json(vehicle_root / "meta.json")
    point_shape = tuple(vehicle_meta["points"]["shape"])
    cell_shape = tuple(vehicle_meta["cells"]["shape"])
    if (
        len(point_shape) != 2
        or point_shape[1] != 3
        or vehicle_meta["points"].get("dtype") != "torch.float32"
        or cell_shape != (spec.n_master_cells, 3)
        or vehicle_meta["cells"].get("dtype") != "torch.int64"
    ):
        raise ValueError(f"Geometry metadata changed for {spec.case_id}")

    cells_memmap = np.memmap(
        vehicle_root / "cells.memmap",
        mode="r",
        dtype="<i8",
        shape=cell_shape,
    )
    selected_cells = torch.from_numpy(np.array(cells_memmap[ids], copy=True))
    referenced, inverse = torch.unique(
        selected_cells,
        sorted=True,
        return_inverse=True,
    )
    points_memmap = np.memmap(
        vehicle_root / "points.memmap",
        mode="r",
        dtype="<f4",
        shape=point_shape,
    )
    selected_points = torch.from_numpy(
        np.array(points_memmap[referenced.numpy()], copy=True)
    )
    compacted_cells = inverse.reshape_as(selected_cells)
    n_cells = int(ids.size)
    global_data = {}
    for name in ALLOWED_RAW_GLOBAL_FIELDS:
        count = 3 if name == "U_inf" else 1
        values = np.fromfile(
            global_root / f"{name}.memmap",
            dtype="<f4",
            count=count,
        )
        if values.size != count:
            raise ValueError(f"Global input payload changed for {spec.case_id}: {name}")
        tensor = torch.from_numpy(values.copy())
        global_data[name] = tensor if name == "U_inf" else tensor.reshape(())
    placeholders = {
        "pMeanTrim": selected_points.new_zeros(n_cells),
        "wallShearStressMeanTrim": selected_points.new_zeros(
            n_cells,
            selected_points.shape[1],
        ),
    }
    return mesh_type(
        points=selected_points,
        cells=compacted_cells,
        point_data={},
        cell_data=placeholders,
        global_data=global_data,
    )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {path}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )


def _decision_outcome(
    *,
    validity_passed: bool,
    full_passed: bool,
) -> str:
    if not validity_passed:
        return INVALID_DIAGNOSTIC
    if full_passed:
        return CANONICAL_FULL_OUTCOME
    return CANONICAL_REPAIR_REFUTED


def _lane_specs(hqc: Any, lane_ordinal: int, lane_count: int) -> tuple[Any, ...]:
    if lane_count != LANE_COUNT:
        raise ValueError(f"lane-count must equal frozen value {LANE_COUNT}")
    if not 0 <= lane_ordinal < lane_count:
        raise ValueError(f"lane-ordinal {lane_ordinal} outside [0,{lane_count})")
    observed = tuple(spec.case_id for spec in hqc.CASE_SPECS)
    if observed != CASE_IDS:
        raise ValueError("Frozen 36-case cohort or ordering changed")
    specs = tuple(
        spec
        for spec in hqc.CASE_SPECS
        if spec.cohort_ordinal % lane_count == lane_ordinal
    )
    if len(specs) != len(CASE_IDS) // LANE_COUNT:
        raise ValueError(f"Lane {lane_ordinal} does not contain exactly nine cases")
    return specs


def _validate_reader_target_free(runtime: Any, hqc: Any) -> None:
    """Validate reader ordering without traversing any local-data path."""
    paths = tuple(Path(path) for path in runtime.dataset.reader._paths)
    if len(paths) != 484:
        raise ValueError(f"Historical reader found {len(paths)} cases, expected 484")
    for spec in hqc.CASE_SPECS:
        case_path = paths[spec.reader_index]
        if spec.case_id not in case_path.parts:
            raise ValueError(
                f"Reader index {spec.reader_index} resolves to {case_path}, "
                f"not {spec.case_id}"
            )
        vehicle_meta = _strict_json(case_path / "_tensordict" / "meta.json")
        n_cells = int(vehicle_meta["cells"]["shape"][0])
        if n_cells != spec.n_master_cells:
            raise ValueError(
                f"{spec.case_id} has {n_cells} master cells, "
                f"expected {spec.n_master_cells}"
            )


def _anchor_prefix(spec: Any) -> str:
    return f"case_{spec.cohort_ordinal:02d}_{spec.case_id}"


def _unit_array_prefix(spec: Any, resolution: int) -> str:
    return f"{_anchor_prefix(spec)}__k{resolution:05d}"


def _anchor_replay(
    helper: Any,
    *,
    spec: Any,
    resolution: int,
    arrays: Mapping[str, np.ndarray],
    anchor_arrays: Mapping[str, np.ndarray],
    model_probes_executed: bool = True,
) -> dict[str, Any]:
    required = spec.case_id in ANCHOR_CASE_IDS and resolution == RESOLUTIONS[0]
    if not required:
        return {
            "required": False,
            "passed": True,
            "compared_arrays": 0,
        }
    if not model_probes_executed:
        return {
            "required": True,
            "passed": False,
            "compared_arrays": 0,
            "not_executed_reason": "model_preflight_validity_failed",
        }
    prefix = _anchor_prefix(spec)
    comparisons: dict[str, bool] = {}
    for name, value in arrays.items():
        anchor_name = name
        for panel in QUERY_PANELS:
            anchor_name = anchor_name.replace(f"_{panel}_", "_", 1)
        key = f"{prefix}__{anchor_name}"
        if key not in anchor_arrays:
            raise KeyError(f"Adjudicated job-305691 anchor is missing {key}")
        comparisons[name] = helper._array_bitwise_equal(value, anchor_arrays[key])
    expected_names = _expected_unit_array_names()
    if set(comparisons) != expected_names:
        raise ValueError("Anchor replay did not cover the exact unit array schema")
    return {
        "required": True,
        "passed": all(comparisons.values()),
        "compared_arrays": len(comparisons),
        "comparisons": comparisons,
    }


def _expected_unit_array_names() -> set[str]:
    names = {
        "selected_cell_ids_int64",
        "canonical_cells_int64",
        "canonical_points_float32",
        "canonical_centroids_float32",
        "canonical_areas_float32",
        "canonical_normals_float32",
    }
    names.update(
        f"{precision}_canonical_full_{panel}_{path}_{field}"
        for precision in PRECISIONS
        for panel in QUERY_PANELS
        for path in ("primary", "fixed", "primary_replay")
        for field in PREDICTION_FIELDS
    )
    return names


def _validate_anchor_summary(summary: Any) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError("Adjudicated job-305691 JSON must be an object")
    if (
        summary.get("schema_version") != 5
        or summary.get("artifact_kind") != "hqc_canonical_geometry_diagnostic"
        or summary.get("status") != "VALID_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC"
        or summary.get("decision_outcome") != ANCHOR_FULL_AND_DERIVED_OUTCOME
        or tuple(summary.get("scientific_scope", {}).get("case_ids", ()))
        != ANCHOR_CASE_IDS
        or summary.get("scientific_scope", {}).get("resolution") != RESOLUTIONS[0]
        or tuple(summary.get("scientific_scope", {}).get("precisions", ()))
        != PRECISIONS
        or summary.get("validity", {}).get("all_cases_and_precisions_passed")
        is not True
        or summary.get("decision_gates", {}).get("full", {}).get("passed") is not True
    ):
        raise ValueError("Adjudicated job-305691 anchor contract changed")


def _validate_output_targets(*outputs: Path) -> None:
    destinations: list[Path] = []
    for output in outputs:
        output = Path(os.path.abspath(os.path.normpath(output)))
        sidecar = output.with_name(f"{output.name}.sha256")
        destinations.extend((output, sidecar))

    lexical_alias = len(set(destinations)) != len(destinations)
    resolved_destinations = [
        destination.resolve(strict=False) for destination in destinations
    ]
    resolved_alias = len(set(resolved_destinations)) != len(resolved_destinations)
    if lexical_alias or resolved_alias:
        raise ValueError(
            "Output and SHA-256 sidecar paths must be pairwise distinct "
            "after normalization"
        )

    for output in destinations:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output or sidecar: {output}")


def _write_fsynced_temporary(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _prepare_npz_temporary(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary, _sha256_file_local(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file_local(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_if_same_inode(path: Path, reference: Path) -> None:
    try:
        path_stat = path.stat(follow_symlinks=False)
        reference_stat = reference.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (path_stat.st_dev, path_stat.st_ino) == (
        reference_stat.st_dev,
        reference_stat.st_ino,
    ):
        path.unlink()


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in sorted(set(paths)):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_output_set_no_clobber(
    *,
    output_json: Path,
    json_payload: bytes,
    output_npz: Path,
    npz_temporary: Path,
    npz_sha256: str,
) -> str:
    """Publish JSON, NPZ, and both sidecars as one rollback-protected set."""
    json_sha256 = hashlib.sha256(json_payload).hexdigest()
    json_sidecar = output_json.with_name(f"{output_json.name}.sha256")
    npz_sidecar = output_npz.with_name(f"{output_npz.name}.sha256")
    json_sidecar_payload = f"{json_sha256}  {output_json.name}\n".encode("ascii")
    npz_sidecar_payload = f"{npz_sha256}  {output_npz.name}\n".encode("ascii")
    temporaries: dict[Path, Path] = {output_npz: npz_temporary}
    published: list[tuple[Path, Path]] = []
    try:
        _validate_output_targets(output_json, output_npz)
        temporaries[output_json] = _write_fsynced_temporary(
            output_json,
            json_payload,
        )
        temporaries[json_sidecar] = _write_fsynced_temporary(
            json_sidecar,
            json_sidecar_payload,
        )
        temporaries[npz_sidecar] = _write_fsynced_temporary(
            npz_sidecar,
            npz_sidecar_payload,
        )
        for destination in (
            output_npz,
            npz_sidecar,
            output_json,
            json_sidecar,
        ):
            temporary = temporaries[destination]
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        _fsync_directories([path.parent for path in temporaries])
        if (
            _sha256_file_local(output_npz) != npz_sha256
            or _sha256_file_local(output_json) != json_sha256
            or npz_sidecar.read_bytes() != npz_sidecar_payload
            or json_sidecar.read_bytes() != json_sidecar_payload
        ):
            raise OSError("Published canonical-validity artifact hash changed")
    except BaseException:
        for destination, temporary in reversed(published):
            _unlink_if_same_inode(destination, temporary)
        _fsync_directories([path.parent for path in temporaries])
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        _fsync_directories([path.parent for path in temporaries])
    return json_sha256


def _validate_import_provenance(repo_root: Path) -> dict[str, str]:
    expected_package = (repo_root / "physicsnemo" / "__init__.py").resolve(strict=True)
    expected_model = (
        repo_root
        / "physicsnemo"
        / "experimental"
        / "nn"
        / "mesh_attention"
        / "model.py"
    ).resolve(strict=True)
    if physicsnemo.__file__ is None or mesh_attention_model.__file__ is None:
        raise RuntimeError("Imported PhysicsNeMo modules have no filesystem origin")
    observed_package = Path(physicsnemo.__file__).resolve(strict=True)
    observed_model = Path(mesh_attention_model.__file__).resolve(strict=True)
    if observed_package != expected_package:
        raise ValueError(
            f"physicsnemo imported from {observed_package}, expected {expected_package}"
        )
    if observed_model != expected_model:
        raise ValueError(
            "mesh_attention.model imported from "
            f"{observed_model}, expected {expected_model}"
        )
    if CanonicalSourceGeometry is not mesh_attention_model.CanonicalSourceGeometry:
        raise ValueError("Public CanonicalSourceGeometry export has split identity")
    return {
        "physicsnemo_init": str(observed_package),
        "mesh_attention_model": str(observed_model),
        "canonical_geometry_module": CanonicalSourceGeometry.__module__,
    }


def _validate_static_inputs(hqc: Any, args: argparse.Namespace) -> None:
    if len(EXPECTED_EXECUTION_SOURCE_TREE_SHA256) != 64:
        raise RuntimeError("Execution source-tree hash has not been frozen")
    _validate_import_provenance(args.repo_root)
    checks = (
        (
            args.dataset_root / "manifest.json",
            EXPECTED_DATASET_MANIFEST_SHA256,
            "Dataset manifest",
        ),
        (args.dataset_config, EXPECTED_DATASET_CONFIG_SHA256, "Dataset config"),
        (args.resolved_config, EXPECTED_RESOLVED_CONFIG_SHA256, "Resolved config"),
        (
            args.checkpoint_dir / hqc.MODEL_FILENAME,
            EXPECTED_MODEL_SHA256,
            "Model checkpoint",
        ),
        (
            args.checkpoint_dir / hqc.NORM_STATS_FILENAME,
            EXPECTED_NORMALIZATION_SHA256,
            "Normalization state",
        ),
        (
            args.checkpoint_dir / hqc.TRAINING_STATE_FILENAME,
            EXPECTED_TRAINING_STATE_SHA256,
            "Training state",
        ),
    )
    for path, expected, label in checks:
        hqc._require_sha256(path, expected, label)
    observed_tree = hqc._source_tree_manifest_sha256(args.repo_root)
    if observed_tree != EXPECTED_EXECUTION_SOURCE_TREE_SHA256:
        raise ValueError(
            "Execution source tree changed: "
            f"expected {EXPECTED_EXECUTION_SOURCE_TREE_SHA256}, got {observed_tree}"
        )


def _validate_geometry_manifest(
    hqc: Any,
    *,
    path: Path,
    dataset_root: Path,
    lane_specs: Sequence[Any],
) -> dict[str, Any]:
    if len(EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256) != 64:
        raise RuntimeError("Geometry input manifest hash has not been frozen")
    hqc._require_sha256(
        path,
        EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256,
        "Target-free geometry input manifest",
    )
    manifest = _strict_json(path)
    cases = manifest.get("cases")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind")
        != "drivaerml_target_free_geometry_input_manifest"
        or manifest.get("status") != "PASSED_TARGET_FREE_GEOMETRY_INPUT_FREEZE"
        or manifest.get("case_count") != len(CASE_IDS)
        or not isinstance(cases, list)
        or [case.get("case_id") for case in cases] != list(CASE_IDS)
        or manifest.get("dataset_root_resolved") != str(dataset_root)
        or manifest.get("dataset_manifest", {}).get("sha256")
        != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise ValueError("Target-free geometry input manifest contract changed")
    exclusion = manifest.get("target_exclusion", {})
    if (
        exclusion.get("point_data_opened") is not False
        or exclusion.get("cell_data_opened") is not False
        or exclusion.get("interior_opened") is not False
        or exclusion.get("supervision_values_opened") is not False
        or exclusion.get("supervision_values_hashed") is not False
        or exclusion.get("supervision_values_serialized") is not False
    ):
        raise ValueError("Geometry manifest target-exclusion contract changed")

    by_case = {case["case_id"]: case for case in cases}
    verified_files = 0
    for spec in lane_specs:
        case = by_case[spec.case_id]
        if (
            case.get("cohort_ordinal") != spec.cohort_ordinal
            or case.get("reader_index") != spec.reader_index
            or case.get("n_master_cells") != spec.n_master_cells
            or case.get("historical_start") != spec.historical_start
        ):
            raise ValueError(f"Geometry manifest metadata changed for {spec.case_id}")
        case_link = dataset_root / spec.case_id
        if (
            not case_link.is_symlink()
            or os.readlink(case_link) != case.get("symlink_target")
            or str(case_link.resolve(strict=True)) != case.get("resolved_case_root")
        ):
            raise ValueError(f"Geometry manifest symlink changed for {spec.case_id}")
        files = case.get("files")
        expected_files = _expected_geometry_manifest_paths(spec.case_id)
        if not isinstance(files, Mapping) or set(files) != expected_files:
            raise ValueError(
                f"Geometry manifest file inventory changed for {spec.case_id}"
            )
        case_root = case_link.resolve(strict=True)
        for relative, record in files.items():
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or {part.lower() for part in relative_path.parts}.intersection(
                    FORBIDDEN_INPUT_PATH_PARTS
                )
            ):
                raise ValueError(f"Forbidden geometry input path: {relative}")
            input_path = case_root / relative_path
            if input_path.is_symlink() or not input_path.is_file():
                raise ValueError(f"Geometry input is not a regular file: {input_path}")
            if (
                type(record) is not dict
                or type(record.get("size_bytes")) is not int
                or type(record.get("sha256")) is not str
                or input_path.stat().st_size != record["size_bytes"]
                or hqc._sha256_file(input_path) != record["sha256"]
            ):
                raise ValueError(f"Geometry input changed: {input_path}")
            verified_files += 1
    return {
        "manifest_sha256": EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256,
        "lane_cases_verified": len(lane_specs),
        "lane_files_verified": verified_files,
    }


def _canonical_geometry_for_domain(
    runtime: Any,
    domain: Any,
    bundle: Any,
) -> CanonicalSourceGeometry:
    """Move the frozen CPU bundle once into the model domain's tensor contract."""
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


def _authoritative_storage_exact(
    encoded: Any,
    geometry: CanonicalSourceGeometry,
) -> dict[str, bool]:
    """Require the public encode to retain the prescribed tensor storage."""
    return {
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


def _decode_coupled_trace(
    helper: Any,
    runtime: Any,
    encoded: Any,
    queries: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    """Decode the one query count permitted by the frozen trace contract."""
    query_mesh = runtime.mesh_type(
        points=queries.to(
            device=encoded.source_mesh.points.device,
            dtype=encoded.source_mesh.points.dtype,
        )
    )
    output = runtime.model.decode(encoded, query_mesh)
    checks = {
        "canonical_queries_exact": helper._tensor_bitwise_equal(
            query_mesh.points.detach().cpu(),
            queries,
        ),
        "encoded_center_is_raw_positive_zero": helper._tensor_bitwise_equal(
            encoded.center.detach().cpu(),
            torch.zeros_like(encoded.center).cpu(),
        ),
        "encoded_reference_length_is_exact_positive_one": (
            helper._tensor_bitwise_equal(
                encoded.reference_length.detach().cpu(),
                torch.ones_like(encoded.reference_length).cpu(),
            )
        ),
        "trace_query_count_exact": (
            encoded.trace_slice is not None
            and encoded.trace_slice.stop - encoded.trace_slice.start == queries.shape[0]
        ),
    }
    return helper._extract_prediction(output, queries.shape[0]), checks


def _score_query_panels(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Score the whole trace and its frozen 2,500-row identity prefix."""
    panels: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        "coupled_s_k": {
            path: {field: values[field] for field in PREDICTION_FIELDS}
            for path, values in predictions.items()
        }
    }
    for values in predictions.values():
        if any(values[field].shape[0] < FIXED_QUERY_K for field in PREDICTION_FIELDS):
            raise ValueError("Coupled trace is smaller than Q=S_2500")
    panels["fixed_id_prefix_s2500"] = {
        path: {
            field: values[field][:FIXED_QUERY_K].clone() for field in PREDICTION_FIELDS
        }
        for path, values in predictions.items()
    }
    return panels


def _run_full_mode(
    helper: Any,
    runtime: Any,
    *,
    primary_domain: Any,
    fixed_domain: Any,
    bundle: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    geometry = _canonical_geometry_for_domain(runtime, primary_domain, bundle)
    encoded = {
        "primary": runtime.model.encode(
            primary_domain,
            canonical_source_geometry=geometry,
        ),
        "fixed": runtime.model.encode(
            fixed_domain,
            canonical_source_geometry=geometry,
        ),
        "primary_replay": runtime.model.encode(
            primary_domain,
            canonical_source_geometry=geometry,
        ),
    }
    injection = {
        path: helper._injected_geometry_exact(value, bundle, "canonical_full")
        for path, value in encoded.items()
    }
    authoritative_storage = {
        path: _authoritative_storage_exact(value, geometry)
        for path, value in encoded.items()
    }
    injection_passed = all(
        check for path_checks in injection.values() for check in path_checks.values()
    )
    storage_passed = all(
        check
        for path_checks in authoritative_storage.values()
        for check in path_checks.values()
    )

    predictions: dict[str, dict[str, torch.Tensor]] = {}
    decode_checks: dict[str, dict[str, bool]] = {}
    for path, value in encoded.items():
        predictions[path], decode_checks[path] = _decode_coupled_trace(
            helper,
            runtime,
            value,
            bundle.centroids,
        )
    decode_passed = all(
        check
        for path_checks in decode_checks.values()
        for check in path_checks.values()
    )
    panel_predictions = _score_query_panels(predictions)
    panels: dict[str, Any] = {}
    arrays: dict[str, torch.Tensor] = {}
    for panel, scored in panel_predictions.items():
        primary_fixed = helper._prediction_difference(
            scored["primary"],
            scored["fixed"],
        )
        primary_replay = helper._prediction_difference(
            scored["primary"],
            scored["primary_replay"],
        )
        replay_passed = helper._difference_is_exact(primary_replay)
        comparison_passed = helper._difference_is_exact(primary_fixed)
        panels[panel] = {
            "query_count": int(scored["primary"]["pressure"].shape[0]),
            "source": (
                "single_public_decode"
                if panel == "coupled_s_k"
                else "first_2500_rows_of_single_public_decode"
            ),
            "primary_fixed_difference": primary_fixed,
            "primary_replay_difference": primary_replay,
            "primary_replay_exact": replay_passed,
            "comparison_gate": {
                "criterion": "fieldwise_bitwise_exact",
                "passed": comparison_passed,
                "controls_candidate_advance": panel == "coupled_s_k",
            },
            "validity_passed": replay_passed,
        }
        arrays.update(
            {
                f"canonical_full_{panel}_{path}_{field}": prediction[field]
                for path, prediction in scored.items()
                for field in PREDICTION_FIELDS
            }
        )
    prefix_alignment = {
        path: {
            field: helper._tensor_bitwise_equal(
                panel_predictions["fixed_id_prefix_s2500"][path][field],
                panel_predictions["coupled_s_k"][path][field][:FIXED_QUERY_K],
            )
            for field in PREDICTION_FIELDS
        }
        for path in ("primary", "fixed", "primary_replay")
    }
    prefix_alignment_passed = all(
        check
        for path_checks in prefix_alignment.values()
        for check in path_checks.values()
    )
    return {
        "mode": "canonical_full_public_api",
        "injected_geometry_exact": injection,
        "injected_geometry_exact_passed": injection_passed,
        "authoritative_storage_identity": authoritative_storage,
        "authoritative_storage_identity_passed": storage_passed,
        "canonical_decode_contract": decode_checks,
        "canonical_decode_contract_passed": decode_passed,
        "query_panels": panels,
        "fixed_id_prefix_matches_coupled_rows": prefix_alignment,
        "fixed_id_prefix_matches_coupled_rows_passed": prefix_alignment_passed,
        "comparison_gate": {
            "criterion": "whole_trace_fieldwise_bitwise_exact",
            "passed": panels["coupled_s_k"]["comparison_gate"]["passed"],
        },
        "validity_passed": (
            injection_passed
            and storage_passed
            and decode_passed
            and prefix_alignment_passed
            and all(panel["validity_passed"] for panel in panels.values())
        ),
    }, arrays


def _run_precision(
    helper: Any,
    runtime: Any,
    *,
    primary_domain: Any,
    fixed_domain: Any,
    bundle: Any,
    precision: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with torch.no_grad(), runtime.autocast_context(precision):
        full, arrays = _run_full_mode(
            helper,
            runtime,
            primary_domain=primary_domain,
            fixed_domain=fixed_domain,
            bundle=bundle,
        )
    return {
        "precision": precision,
        "canonical_full_public_api": full,
        "validity_passed": full["validity_passed"],
        "decision_gates": {
            "full_passed": full["comparison_gate"]["passed"],
        },
    }, arrays


def _run_resolution(
    helper: Any,
    hqc: Any,
    runtime: Any,
    *,
    spec: Any,
    dataset_root: Path,
    fixed_center: torch.Tensor,
    resolution: int,
    anchor_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(
        f"CANONICAL_VALIDITY_UNIT_START case={spec.case_id} k={resolution}",
        flush=True,
    )
    ids = hqc._cyclic_indices(
        spec.n_master_cells,
        spec.historical_start,
        resolution,
    )
    fixed_q_ids = hqc._cyclic_indices(
        spec.n_master_cells,
        spec.historical_start,
        FIXED_QUERY_K,
    )
    fixed_q_is_exact_prefix = helper._array_bitwise_equal(
        np.asarray(ids[:FIXED_QUERY_K], dtype="<i8"),
        np.asarray(fixed_q_ids, dtype="<i8"),
    )
    subset = _target_free_file_subset(
        dataset_root,
        spec,
        ids,
        runtime.mesh_type,
    )

    raw_canonical = helper._build_canonical_raw_geometry(subset)
    raw_canonical_replay = helper._build_canonical_raw_geometry(subset)
    physical_length = helper._nested_tensor_value(subset.global_data, "L_ref")

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
    if not helper._tensor_bitwise_equal(applied_fixed_center, fixed_center):
        raise ValueError("Fixed center changed while applying historical pipeline")
    primary_domain = helper._strip_local_data(
        primary_with_placeholders,
        runtime.mesh_type,
    )
    fixed_domain = helper._strip_local_data(
        fixed_with_placeholders,
        runtime.mesh_type,
    )

    reference_key = runtime.model.reference_length_key
    if reference_key is None:
        raise ValueError("Canonical validity requires an explicit reference length")
    primary_reference = helper._nested_tensor_value(
        primary_domain.global_data,
        reference_key,
    )
    fixed_reference = helper._nested_tensor_value(
        fixed_domain.global_data,
        reference_key,
    )
    if not helper._tensor_bitwise_equal(primary_reference, fixed_reference):
        raise ValueError("Primary/fixed model reference lengths differ")
    bundle = helper._finish_canonical_bundle(
        raw_canonical,
        physical_length=physical_length,
        model_reference_length=primary_reference,
    )
    replay_bundle = helper._finish_canonical_bundle(
        raw_canonical_replay,
        physical_length=physical_length,
        model_reference_length=primary_reference,
    )
    construction_checks = helper._bundle_difference(bundle, replay_bundle)
    bundle_validity = helper._bundle_validity(
        bundle,
        expected_cells=subset.cells,
    )
    topology_checks = helper._path_topology_checks(
        subset,
        primary_domain,
        fixed_domain,
    )
    construction_passed = all(construction_checks.values())
    topology_passed = all(topology_checks.values()) and fixed_q_is_exact_prefix

    precision_summaries: dict[str, Any] = {}
    precision_arrays: dict[str, torch.Tensor] = {}
    safe_to_run_model = (
        bundle_validity["passed"] and construction_passed and topology_passed
    )
    if safe_to_run_model:
        for precision in PRECISIONS:
            print(
                "CANONICAL_VALIDITY_PRECISION_START "
                f"case={spec.case_id} k={resolution} precision={precision}",
                flush=True,
            )
            summary, arrays = _run_precision(
                helper,
                runtime,
                primary_domain=primary_domain,
                fixed_domain=fixed_domain,
                bundle=bundle,
                precision=precision,
            )
            precision_summaries[precision] = summary
            precision_arrays.update(
                {f"{precision}_{name}": value for name, value in arrays.items()}
            )

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
    if safe_to_run_model and set(arrays_np) != _expected_unit_array_names():
        raise ValueError("Unit NPZ array schema changed")

    anchor_replay = _anchor_replay(
        helper,
        spec=spec,
        resolution=resolution,
        arrays=arrays_np,
        anchor_arrays=anchor_arrays,
        model_probes_executed=safe_to_run_model,
    )
    validity_passed = (
        safe_to_run_model
        and len(precision_summaries) == len(PRECISIONS)
        and all(summary["validity_passed"] for summary in precision_summaries.values())
        and anchor_replay["passed"]
    )
    full_passed = len(precision_summaries) == len(PRECISIONS) and all(
        summary["decision_gates"]["full_passed"]
        for summary in precision_summaries.values()
    )
    result = {
        "case_id": spec.case_id,
        "cohort_ordinal": int(spec.cohort_ordinal),
        "reader_index": int(spec.reader_index),
        "resolution": resolution,
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
            "canonical_construction_replay": construction_checks,
            "canonical_construction_replay_passed": construction_passed,
            "historical_path_topology": topology_checks,
            "historical_path_topology_passed": topology_passed,
            "fixed_q_is_exact_source_prefix": fixed_q_is_exact_prefix,
            "job305691_anchor_replay": anchor_replay,
            "model_local_data_stripped": True,
            "model_probes_executed": safe_to_run_model,
        },
        "precision_probes": precision_summaries,
        "validity_passed": validity_passed,
        "decision_gates": {
            "full_passed": full_passed,
        },
        "decision_outcome": _decision_outcome(
            validity_passed=validity_passed,
            full_passed=full_passed,
        ),
    }
    print(
        "CANONICAL_VALIDITY_UNIT_DONE "
        f"case={spec.case_id} k={resolution} "
        f"validity_passed={validity_passed} "
        f"outcome={result['decision_outcome']}",
        flush=True,
    )
    return result, arrays_np


def _run_case(
    helper: Any,
    hqc: Any,
    runtime: Any,
    *,
    spec: Any,
    dataset_root: Path,
    anchor_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"CANONICAL_VALIDITY_CASE_START case={spec.case_id}", flush=True)
    ids_10k = hqc._cyclic_indices(
        spec.n_master_cells,
        spec.historical_start,
        hqc.BASELINE_K,
    )
    subset_10k = _target_free_file_subset(
        dataset_root,
        spec,
        ids_10k,
        runtime.mesh_type,
    )
    fixed_center = hqc._pipeline_center_on_device(subset_10k, runtime.device)

    resolutions: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for resolution in RESOLUTIONS:
        summary, unit_arrays = _run_resolution(
            helper,
            hqc,
            runtime,
            spec=spec,
            dataset_root=dataset_root,
            fixed_center=fixed_center,
            resolution=resolution,
            anchor_arrays=anchor_arrays,
        )
        resolutions.append(summary)
        prefix = _unit_array_prefix(spec, resolution)
        arrays.update(
            {f"{prefix}__{name}": value for name, value in unit_arrays.items()}
        )

    validity_passed = all(row["validity_passed"] for row in resolutions)
    full_passed = all(row["decision_gates"]["full_passed"] for row in resolutions)
    return {
        "case_id": spec.case_id,
        "cohort_ordinal": int(spec.cohort_ordinal),
        "reader_index": int(spec.reader_index),
        "resolutions": resolutions,
        "validity_passed": validity_passed,
        "decision_gates": {
            "full_passed": full_passed,
        },
        "decision_outcome": _decision_outcome(
            validity_passed=validity_passed,
            full_passed=full_passed,
        ),
    }, arrays


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
    output_npz: Path,
    npz_sha256: str,
) -> dict[str, Any]:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return {
        "command": list(sys.argv),
        "diagnostic_script_path": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": hqc._sha256_file(Path(__file__).resolve()),
        "canonical_helper_path": str(args.canonical_helper),
        "canonical_helper_sha256": hqc._sha256_file(args.canonical_helper),
        "frozen_producer_path": str(args.producer),
        "frozen_producer_sha256": hqc._sha256_file(args.producer),
        "import_provenance": _validate_import_provenance(args.repo_root),
        "source_tree_manifest_sha256": hqc._source_tree_manifest_sha256(args.repo_root),
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
            "training_state": hqc._sha256_file(
                args.checkpoint_dir / hqc.TRAINING_STATE_FILENAME
            ),
            "geometry_input_manifest": hqc._sha256_file(args.geometry_input_manifest),
            "job305691_anchor_json": hqc._sha256_file(args.anchor_json),
            "job305691_anchor_npz": hqc._sha256_file(args.anchor_npz),
        },
        "npz_path": str(output_npz),
        "npz_sha256": npz_sha256,
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
    parser.add_argument("--canonical-helper", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--geometry-input-manifest", type=Path, required=True)
    parser.add_argument("--anchor-json", type=Path, required=True)
    parser.add_argument("--anchor-npz", type=Path, required=True)
    parser.add_argument("--lane-ordinal", type=int, required=True)
    parser.add_argument("--lane-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    for name in (
        "producer",
        "canonical_helper",
        "repo_root",
        "dataset_root",
        "dataset_config",
        "resolved_config",
        "checkpoint_dir",
        "geometry_input_manifest",
        "anchor_json",
        "anchor_npz",
    ):
        path = getattr(args, name)
        if path.is_symlink():
            raise ValueError(f"Input must not be a symlink: {path}")
        setattr(args, name, path.resolve(strict=True))
    for name in ("output_json", "output_npz"):
        setattr(args, name, Path(os.path.abspath(getattr(args, name))))
    _validate_output_targets(args.output_json, args.output_npz)

    for path, expected, label in (
        (args.producer, EXPECTED_PRODUCER_SHA256, "Frozen H-QC producer"),
        (args.canonical_helper, EXPECTED_HELPER_SHA256, "Frozen canonical helper"),
    ):
        observed = _sha256_file_local(path)
        if observed != expected:
            raise ValueError(
                f"{label} SHA-256 differs: expected {expected}, got {observed}"
            )
    hqc = _load_module(args.producer, "frozen_hqc_producer")
    helper = _load_module(args.canonical_helper, "frozen_canonical_helper_v5")
    hqc._require_sha256(
        args.anchor_json,
        EXPECTED_ANCHOR_JSON_SHA256,
        "Adjudicated job-305691 JSON",
    )
    hqc._require_sha256(
        args.anchor_npz,
        EXPECTED_ANCHOR_NPZ_SHA256,
        "Adjudicated job-305691 NPZ",
    )
    specs = _lane_specs(hqc, args.lane_ordinal, args.lane_count)
    _validate_static_inputs(hqc, args)
    geometry_verification = _validate_geometry_manifest(
        hqc,
        path=args.geometry_input_manifest,
        dataset_root=args.dataset_root,
        lane_specs=specs,
    )
    _validate_anchor_summary(_strict_json(args.anchor_json))
    with np.load(args.anchor_npz, allow_pickle=False) as archive:
        anchor_arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }

    runtime = hqc._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    _validate_reader_target_free(runtime, hqc)

    cases: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}
    for completed, spec in enumerate(specs, start=1):
        case, arrays = _run_case(
            helper,
            hqc,
            runtime,
            spec=spec,
            dataset_root=args.dataset_root,
            anchor_arrays=anchor_arrays,
        )
        cases.append(case)
        overlap = set(npz_arrays).intersection(arrays)
        if overlap:
            raise ValueError(f"Duplicate output array keys: {sorted(overlap)}")
        npz_arrays.update(arrays)
        print(
            f"COMPLETED_UNITS={completed}/{len(specs)} "
            f"case={spec.case_id} lane={args.lane_ordinal}",
            flush=True,
        )

    forbidden_array_keys = [
        key
        for key in npz_arrays
        if any(token in key.lower() for token in FORBIDDEN_ARTIFACT_KEY_TOKENS)
    ]
    if forbidden_array_keys:
        raise ValueError(f"Forbidden NPZ keys: {forbidden_array_keys}")

    all_validity_passed = all(case["validity_passed"] for case in cases)
    all_full_passed = all(case["decision_gates"]["full_passed"] for case in cases)
    outcome = _decision_outcome(
        validity_passed=all_validity_passed,
        full_passed=all_full_passed,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": VALID_STATUS if all_validity_passed else INVALID_STATUS,
        "decision_outcome": outcome,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lane": {
            "ordinal": args.lane_ordinal,
            "count": args.lane_count,
        },
        "scientific_scope": {
            "case_ids": [spec.case_id for spec in specs],
            "resolutions": list(RESOLUTIONS),
            "precisions": list(PRECISIONS),
            "licensing_field_tensor_comparisons_per_lane": 180,
            "licensing_field_tensor_comparisons_full_cohort": 720,
            "deduplicated_panel_field_summaries_per_lane": 324,
            "deduplicated_panel_field_summaries_full_cohort": 1296,
            "emitted_panel_field_records_per_lane": 360,
            "emitted_panel_field_records_full_cohort": 1440,
            "prefix_summaries_are_independent_decisions": False,
            "supervision_arrays_indexed": False,
            "supervision_files_opened_by_model_producer": False,
            "raw_dataset_sample_loader_called": False,
            "geometry_only_memmap_allowlist_applied": True,
            "synthetic_placeholders_stripped_before_model": True,
            "hqc_decision_statistics_computed": False,
            "may_not_be_used_as_hqc_verdict_output": True,
        },
        "contract": {
            "canonical_construction": (
                "float64 raw geometry -> physical area center -> divide by "
                "L_ref*model_reference_length -> one float32 cast"
            ),
            "canonical_full_fields": [
                "points",
                "centroids",
                "areas",
                "normals",
            ],
            "query_frame": "canonical_trace_centroids",
            "query_execution": (
                "one S_K trace decode per path; Q=S_2500 is the first 2500 "
                "cell-identity rows and not a standalone decode"
            ),
            "full_comparison": "fieldwise_bitwise_exact",
            "full_candidate_advances_if": (
                "all validity gates and all 720 full-field tensor comparisons "
                "pass across the complete four-lane cohort"
            ),
            "canonical_derived_private_intervention_executed": False,
        },
        "validity": {
            "all_cases_resolutions_and_precisions_passed": all_validity_passed,
            "geometry_input_manifest_lane_verification": geometry_verification,
            "required_gates": [
                "exact_cohort_lane_and_nested_resolution_contract",
                "canonical_construction_replay",
                "shape",
                "topology",
                "finite_positive_unit_centered_geometry",
                "job305691_overlap_replay",
                "primary_replay_exact",
                "public_api_authoritative_storage_identity",
                "public_api_raw_positive_zero_center_and_positive_one_scale",
                "prefix_summary_exactly_slices_the_coupled_trace",
            ],
        },
        "decision_gates": {
            "full": {
                "criterion": (
                    "primary-versus-fixed pressure and WSS raw-byte exact "
                    "over the full trace for every lane case, resolution, "
                    "and precision"
                ),
                "passed": all_full_passed,
                "controls_candidate_advance": True,
            },
        },
        "cases": cases,
        "npz_array_manifest": helper._array_manifest(hqc, npz_arrays),
    }
    npz_temporary, npz_sha256 = _prepare_npz_temporary(
        args.output_npz,
        npz_arrays,
    )
    try:
        result["provenance"] = _provenance(
            hqc=hqc,
            args=args,
            output_npz=args.output_npz,
            npz_sha256=npz_sha256,
        )
        forbidden_json_keys = _forbidden_artifact_keys(result)
        if forbidden_json_keys:
            raise ValueError(f"Forbidden JSON keys: {forbidden_json_keys}")
        payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        _publish_output_set_no_clobber(
            output_json=args.output_json,
            json_payload=payload,
            output_npz=args.output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )
    finally:
        npz_temporary.unlink(missing_ok=True)
    print(
        f"{result['status']} outcome={outcome} "
        f"json={args.output_json} npz={args.output_npz}",
        flush=True,
    )
    if not all_validity_passed:
        raise RuntimeError("Full-cohort canonical geometry failed a validity gate")


if __name__ == "__main__":
    main()
