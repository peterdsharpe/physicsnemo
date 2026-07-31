# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce one archive-blind lane of the historical DrivAerML K=10k replay.

The producer reconstructs the exact frozen cell block from explicitly
allowlisted geometry and selected-target bytes, executes the historical
``model(domain)`` call once per case in BF16, and persists observations.  It
never opens the historical prediction archive or metrics and never publishes a
categorical outcome; an independent reducer owns every deciding comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import platform
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_replay_producer"
STATUS = "COMPLETED_HISTORICAL_K10000_REPLAY_PRODUCER"
RESOLUTION = 10_000
TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}

EXPECTED_HELPER_SHA256 = (
    "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
)
HELPER_FILENAME = "drivaerml_historical_k10000_replay_runtime.py"
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
EXPECTED_TRAINING_STATE_SHA256 = (
    "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
)
EXPECTED_GEOMETRY_MANIFEST_SHA256 = (
    "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
)
EXPECTED_TARGET_INPUT_MANIFEST_SHA256 = (
    "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49"
)
EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256 = (
    "fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10"
)
EXPECTED_EXECUTION_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_CURRENT_INFER_SHA256 = (
    "47aec675e54d58ee4202831ae0d20039b1ff5ec40e69cc8b5087ce00bd5234ed"
)
EXPECTED_CURRENT_MODEL_SOURCE_SHA256 = (
    "9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892"
)

EXPECTED_COHORT_SHA256 = (
    "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
)
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

EXPECTED_CASE_SPECS = (
    (0, "run_118", 21, 17_504_739, 14_045_027),
    (1, "run_129", 33, 16_380_547, 14_700_754),
    (2, "run_145", 51, 15_789_064, 9_195_926),
    (3, "run_149", 55, 18_007_064, 4_452_828),
    (4, "run_17", 77, 19_404_150, 6_369_582),
    (5, "run_171", 79, 18_792_923, 1_320_415),
    (6, "run_18", 88, 14_634_570, 10_215_595),
    (7, "run_183", 92, 14_932_664, 7_635_018),
    (8, "run_197", 107, 18_934_869, 16_494_923),
    (9, "run_202", 114, 17_796_743, 15_267_620),
    (10, "run_225", 136, 15_024_109, 3_789_927),
    (11, "run_270", 185, 18_857_430, 10_967_997),
    (12, "run_271", 186, 16_922_213, 5_453_831),
    (13, "run_298", 212, 15_063_884, 4_943_208),
    (14, "run_305", 221, 18_022_481, 16_998_850),
    (15, "run_320", 237, 16_199_351, 15_062_581),
    (16, "run_367", 285, 18_958_141, 5_352_845),
    (17, "run_380", 298, 19_519_305, 11_721_918),
    (18, "run_382", 300, 16_887_630, 11_083_431),
    (19, "run_399", 318, 16_222_090, 15_155_572),
    (20, "run_4", 319, 16_294_644, 13_228_777),
    (21, "run_409", 329, 16_591_548, 1_346_462),
    (22, "run_419", 340, 14_561_784, 12_777_694),
    (23, "run_424", 346, 16_588_938, 13_358_519),
    (24, "run_429", 351, 17_738_132, 365_298),
    (25, "run_431", 354, 15_747_949, 1_091_720),
    (26, "run_439", 362, 17_809_120, 8_840_407),
    (27, "run_465", 391, 16_443_085, 11_669_428),
    (28, "run_468", 394, 18_343_677, 15_504_945),
    (29, "run_469", 395, 19_780_049, 19_757_508),
    (30, "run_478", 404, 16_648_431, 16_079_300),
    (31, "run_489", 416, 16_063_459, 6_463_342),
    (32, "run_490", 418, 17_847_065, 191_824),
    (33, "run_495", 423, 15_715_663, 11_592_670),
    (34, "run_71", 453, 16_516_082, 2_240_523),
    (35, "run_86", 469, 17_188_261, 4_374_650),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_read_bytes(path: Path, *, chunk_bytes: int = 8 << 20) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, chunk_bytes):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise ValueError(f"Input changed while being read: {path}")
    return b"".join(chunks)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular(path: Path, *, expected_size: int) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size != expected_size:
        os.close(descriptor)
        raise ValueError(
            f"Input size/type changed for {path}: "
            f"observed={observed.st_size} expected={expected_size}"
        )
    return descriptor, observed


def _sha256_descriptor(
    descriptor: int,
    *,
    size: int,
    chunk_bytes: int = 8 << 20,
) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        payload = os.pread(descriptor, min(chunk_bytes, size - offset), offset)
        if not payload:
            raise ValueError(f"Short read while hashing descriptor at {offset}/{size}")
        digest.update(payload)
        offset += len(payload)
    return digest.hexdigest()


def _safe_hashed_pread(
    path: Path,
    *,
    expected_file_size: int,
    expected_file_sha256: str,
    offset: int,
    count: int,
) -> bytes:
    descriptor, before = _open_regular(path, expected_size=expected_file_size)
    try:
        observed_sha256 = _sha256_descriptor(
            descriptor,
            size=expected_file_size,
        )
        payload = os.pread(descriptor, count, offset)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if observed_sha256 != expected_file_sha256:
        raise ValueError(f"Geometry input SHA-256 changed: {path}")
    if len(payload) != count:
        raise ValueError(f"Short selected geometry read from {path}")
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"Geometry input changed while being read: {path}")
    return payload


def _safe_hashed_rows(
    path: Path,
    *,
    expected_file_size: int,
    expected_file_sha256: str,
    n_rows: int,
    row_indices: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(row_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) == 0
        or np.any(indices < 0)
        or np.any(indices >= n_rows)
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("Selected point rows must be nonempty, sorted, and unique")
    if expected_file_size != n_rows * 3 * np.dtype("<f4").itemsize:
        raise ValueError("Point-file size contract changed")
    descriptor, before = _open_regular(path, expected_size=expected_file_size)
    mapping: mmap.mmap | None = None
    try:
        observed_sha256 = _sha256_descriptor(
            descriptor,
            size=expected_file_size,
        )
        mapping = mmap.mmap(descriptor, length=0, access=mmap.ACCESS_READ)
        all_points = np.ndarray((n_rows, 3), dtype="<f4", buffer=mapping)
        selected = np.ascontiguousarray(all_points[indices])
        del all_points
        mapping.close()
        mapping = None
        after = os.fstat(descriptor)
    finally:
        if mapping is not None:
            mapping.close()
        os.close(descriptor)
    if observed_sha256 != expected_file_sha256:
        raise ValueError(f"Geometry input SHA-256 changed: {path}")
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"Geometry input changed while being read: {path}")
    return selected


def _safe_selected_target(
    path: Path,
    *,
    expected_file_size: int,
    offset: int,
    count: int,
    expected_selected_sha256: str,
) -> bytes:
    descriptor, before = _open_regular(path, expected_size=expected_file_size)
    try:
        payload = os.pread(descriptor, count, offset)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != count:
        raise ValueError(f"Short selected target read from {path}")
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"Target input changed while being read: {path}")
    if _sha256_bytes(payload) != expected_selected_sha256:
        raise ValueError(f"Selected target bytes changed: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_safe_read_bytes(path))


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {label}")

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _strict_json(path: Path) -> Any:
    return _strict_json_bytes(_safe_read_bytes(path), str(path))


def _require_sha256(path: Path, expected: str, label: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 differs: expected={expected} observed={observed}"
        )


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _array_bytes(value: np.ndarray, dtype: str | np.dtype[Any]) -> bytes:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    return memoryview(array).cast("B").tobytes()


def _array_sha256(value: np.ndarray, dtype: str | np.dtype[Any]) -> str:
    return _sha256_bytes(_array_bytes(value, dtype))


def _verify_geometry_manifest(
    helper: ModuleType,
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    payload = _safe_read_bytes(manifest_path)
    if _sha256_bytes(payload) != EXPECTED_GEOMETRY_MANIFEST_SHA256:
        raise ValueError("Target-free geometry manifest changed")
    manifest = _strict_json_bytes(payload, str(manifest_path))
    cases = manifest.get("cases")
    expected_ids = [spec[1] for spec in EXPECTED_CASE_SPECS]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind")
        != "drivaerml_target_free_geometry_input_manifest"
        or manifest.get("status") != "PASSED_TARGET_FREE_GEOMETRY_INPUT_FREEZE"
        or manifest.get("case_count") != 36
        or not isinstance(cases, list)
        or [case.get("case_id") for case in cases] != expected_ids
        or manifest.get("cohort_sha256") != EXPECTED_COHORT_SHA256
        or manifest.get("dataset_manifest", {}).get("sha256")
        != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise ValueError("Target-free geometry manifest contract changed")
    exclusions = manifest.get("target_exclusion", {})
    for key in (
        "point_data_opened",
        "cell_data_opened",
        "interior_opened",
        "supervision_values_opened",
        "supervision_values_hashed",
        "supervision_values_serialized",
    ):
        if exclusions.get(key) is not False:
            raise ValueError(f"Target-free geometry exclusion changed for {key}")

    verified_files = 0
    for expected_spec, case in zip(EXPECTED_CASE_SPECS, cases, strict=True):
        ordinal, case_id, reader_index, n_cells, start = expected_spec
        if (
            case.get("cohort_ordinal") != ordinal
            or case.get("reader_index") != reader_index
            or case.get("n_master_cells") != n_cells
            or case.get("historical_start") != start
        ):
            raise ValueError(f"Geometry manifest case metadata changed for {case_id}")
        case_link = dataset_root / case_id
        if (
            not case_link.is_symlink()
            or os.readlink(case_link) != case.get("symlink_target")
            or str(case_link.resolve(strict=True)) != case.get("resolved_case_root")
        ):
            raise ValueError(f"Dataset case symlink changed for {case_id}")
        files = case.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"Geometry file inventory is missing for {case_id}")
        case_root = case_link.resolve(strict=True)
        for relative_text, record in files.items():
            relative = Path(relative_text)
            lowered = {part.lower() for part in relative.parts}
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or lowered.intersection(
                    {
                        "point_data",
                        "cell_data",
                        "interior",
                        "pressure",
                        "wss",
                        "pmeantrim",
                        "wallshearstressmeantrim",
                    }
                )
            ):
                raise ValueError(f"Unsafe target-free geometry path: {relative_text}")
            path = case_root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Geometry input is not a regular file: {path}")
            if not isinstance(record, Mapping):
                raise ValueError(f"Geometry input changed: {path}")
            size_bytes = record.get("size_bytes")
            sha256 = record.get("sha256")
            if (
                type(size_bytes) is not int
                or type(sha256) is not str
                or len(sha256) != 64
                or path.stat(follow_symlinks=False).st_size != size_bytes
            ):
                raise ValueError(f"Geometry input changed: {path}")
            if path.name not in {"points.memmap", "cells.memmap"}:
                if _sha256_bytes(_safe_read_bytes(path)) != sha256:
                    raise ValueError(f"Geometry input changed: {path}")
            verified_files += 1
    return {
        "manifest_sha256": EXPECTED_GEOMETRY_MANIFEST_SHA256,
        "cases_verified": 36,
        "files_verified": verified_files,
        "case_records": cases,
    }


def _verify_target_input_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    payload = _safe_read_bytes(manifest_path)
    if _sha256_bytes(payload) != EXPECTED_TARGET_INPUT_MANIFEST_SHA256:
        raise ValueError("Selected-target input manifest changed")
    manifest = _strict_json_bytes(payload, str(manifest_path))
    cases = manifest.get("cases")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind")
        != "drivaerml_historical_k10000_selected_target_input_manifest"
        or manifest.get("status")
        != "PASSED_HISTORICAL_K10000_SELECTED_TARGET_INPUT_FREEZE"
        or manifest.get("case_count") != 36
        or manifest.get("resolution") != RESOLUTION
        or manifest.get("cohort_sha256") != EXPECTED_COHORT_SHA256
        or manifest.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256
        or not isinstance(cases, list)
        or [case.get("case_id") for case in cases]
        != [spec[1] for spec in EXPECTED_CASE_SPECS]
    ):
        raise ValueError("Selected-target input manifest contract changed")
    exclusions = manifest.get("read_exclusions")
    if not isinstance(exclusions, Mapping) or any(
        exclusions.get(key) is not False
        for key in (
            "interior_opened",
            "model_output_generated",
            "other_cell_data_opened",
            "point_data_opened",
        )
    ):
        raise ValueError("Selected-target input exclusions changed")

    for spec, case in zip(EXPECTED_CASE_SPECS, cases, strict=True):
        ordinal, case_id, reader_index, n_cells, start = spec
        if (
            case.get("cohort_ordinal") != ordinal
            or case.get("reader_index") != reader_index
            or case.get("n_master_cells") != n_cells
            or case.get("historical_start") != start
            or case.get("resolution") != RESOLUTION
        ):
            raise ValueError(f"Selected-target case contract changed for {case_id}")
        case_link = dataset_root / case_id
        if (
            not case_link.is_symlink()
            or os.readlink(case_link) != case.get("symlink_target")
            or str(case_link.resolve(strict=True)) != case.get("resolved_case_root")
        ):
            raise ValueError(f"Selected-target case link changed for {case_id}")
        metadata_record = case.get("cell_data_metadata")
        if not isinstance(metadata_record, Mapping):
            raise ValueError(f"Selected-target metadata record missing for {case_id}")
        metadata_relative = Path(str(metadata_record.get("relative_path", "")))
        if metadata_relative.is_absolute() or ".." in metadata_relative.parts:
            raise ValueError(f"Unsafe selected-target metadata path for {case_id}")
        metadata_path = case_link.resolve(strict=True) / metadata_relative
        metadata_payload = _safe_read_bytes(metadata_path)
        if len(metadata_payload) != metadata_record.get("size_bytes") or _sha256_bytes(
            metadata_payload
        ) != metadata_record.get("sha256"):
            raise ValueError(f"Selected-target metadata changed for {case_id}")
        metadata = _strict_json_bytes(metadata_payload, str(metadata_path))
        targets = case.get("selected_targets")
        if not isinstance(targets, Mapping) or set(targets) != {"pressure", "wss"}:
            raise ValueError(f"Selected-target records changed for {case_id}")
        for name, field, components in (
            ("pressure", "pMeanTrim", 1),
            ("wss", "wallShearStressMeanTrim", 3),
        ):
            record = targets[name]
            expected_relative = (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                f"_tensordict/cell_data/{field}.memmap"
            )
            expected_shape = [n_cells] if components == 1 else [n_cells, components]
            if (
                not isinstance(record, Mapping)
                or record.get("raw_field_name") != field
                or record.get("source_relative_path") != expected_relative
                or record.get("source_offset_bytes") != start * components * 4
                or record.get("selected_size_bytes") != RESOLUTION * components * 4
                or record.get("source_size_bytes") != n_cells * components * 4
                or record.get("selected_dtype") != "float32_little_endian"
                or metadata.get(field, {}).get("shape") != expected_shape
                or metadata.get(field, {}).get("dtype") != "torch.float32"
                or metadata.get(field, {}).get("device") != "cpu"
            ):
                raise ValueError(
                    f"Selected-target field contract changed for {case_id}/{name}"
                )
    return {
        "manifest_sha256": EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
        "cases_verified": 36,
        "selected_ranges_verified": 72,
        "case_records": cases,
    }


def _validate_static_inputs(
    helper: ModuleType,
    *,
    repo_root: Path,
    dataset_root: Path,
    dataset_config: Path,
    resolved_config: Path,
    checkpoint_dir: Path,
) -> dict[str, str]:
    checks = (
        (
            dataset_root / "manifest.json",
            EXPECTED_DATASET_MANIFEST_SHA256,
            "Dataset manifest",
        ),
        (dataset_config, EXPECTED_DATASET_CONFIG_SHA256, "Dataset config"),
        (resolved_config, EXPECTED_RESOLVED_CONFIG_SHA256, "Resolved config"),
        (
            checkpoint_dir / helper.MODEL_FILENAME,
            EXPECTED_MODEL_SHA256,
            "Model checkpoint",
        ),
        (
            checkpoint_dir / helper.TRAINING_STATE_FILENAME,
            EXPECTED_TRAINING_STATE_SHA256,
            "Training state",
        ),
        (
            checkpoint_dir / helper.NORM_STATS_FILENAME,
            EXPECTED_NORMALIZATION_SHA256,
            "Normalization state",
        ),
        (
            repo_root
            / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
            / "src/infer.py",
            EXPECTED_CURRENT_INFER_SHA256,
            "Current inference source",
        ),
        (
            repo_root / "physicsnemo/experimental/nn/mesh_attention/model.py",
            EXPECTED_CURRENT_MODEL_SOURCE_SHA256,
            "Current MeshTransformer source",
        ),
    )
    result: dict[str, str] = {}
    for path, expected, label in checks:
        _require_sha256(path, expected, label)
        result[label] = expected
    observed_tree = helper._source_tree_manifest_sha256(repo_root)
    if observed_tree != EXPECTED_EXECUTION_SOURCE_TREE_SHA256:
        raise ValueError(
            "Current execution source tree changed: "
            f"expected={EXPECTED_EXECUTION_SOURCE_TREE_SHA256} "
            f"observed={observed_tree}"
        )
    result["Current execution source tree"] = observed_tree
    return result


def _validate_import_provenance(repo_root: Path) -> dict[str, str]:
    import infer as recipe_infer

    import physicsnemo
    from physicsnemo.experimental.nn.mesh_attention import model as model_module

    expected = {
        "physicsnemo": (repo_root / "physicsnemo/__init__.py").resolve(strict=True),
        "mesh_transformer_model": (
            repo_root / "physicsnemo/experimental/nn/mesh_attention/model.py"
        ).resolve(strict=True),
        "recipe_infer": (
            repo_root
            / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
            / "src/infer.py"
        ).resolve(strict=True),
    }
    observed = {
        "physicsnemo": Path(physicsnemo.__file__).resolve(strict=True),
        "mesh_transformer_model": Path(model_module.__file__).resolve(strict=True),
        "recipe_infer": Path(recipe_infer.__file__).resolve(strict=True),
    }
    if observed != expected:
        raise ImportError(
            f"Execution import provenance changed: expected={expected} observed={observed}"
        )
    return {name: str(path) for name, path in observed.items()}


def _validate_case_specs(helper: ModuleType) -> None:
    observed = tuple(
        (
            spec.cohort_ordinal,
            spec.case_id,
            spec.reader_index,
            spec.n_master_cells,
            spec.historical_start,
        )
        for spec in helper.CASE_SPECS
    )
    if observed != EXPECTED_CASE_SPECS:
        raise ValueError("Frozen helper CASE_SPECS changed")
    helper._validate_historical_starts()


def _validate_reader(runtime: Any) -> None:
    paths = tuple(Path(path) for path in runtime.dataset.reader._paths)
    if len(paths) != 484:
        raise ValueError(f"Reader found {len(paths)} cases, expected 484")
    for _, case_id, reader_index, n_cells, _ in EXPECTED_CASE_SPECS:
        path = paths[reader_index]
        if case_id not in path.parts:
            raise ValueError(
                f"Reader index {reader_index} resolves to {path}, not {case_id}"
            )
        metadata = _strict_json(path / "_tensordict" / "meta.json")
        observed_n_cells = int(metadata["cells"]["shape"][0])
        if observed_n_cells != n_cells:
            raise ValueError(
                f"{case_id} has {observed_n_cells} master cells, expected {n_cells}"
            )


def _geometry_file_record(
    case_record: Mapping[str, Any],
    relative_path: str,
) -> Mapping[str, Any]:
    files = case_record.get("files")
    record = files.get(relative_path) if isinstance(files, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"Frozen geometry record is missing {relative_path}")
    return record


def _load_explicit_raw_subset(
    runtime: Any,
    dataset_root: Path,
    spec: Any,
    geometry_case: Mapping[str, Any],
    target_case: Mapping[str, Any],
) -> tuple[Any, dict[str, np.ndarray], dict[str, str]]:
    case_root = (dataset_root / spec.case_id).resolve(strict=True)
    if (
        geometry_case.get("case_id") != spec.case_id
        or target_case.get("case_id") != spec.case_id
        or geometry_case.get("resolved_case_root") != str(case_root)
        or target_case.get("resolved_case_root") != str(case_root)
    ):
        raise ValueError(f"Frozen case-root identity changed for {spec.case_id}")

    mesh_relative = (
        f"domain_{spec.case_id}.pdmsh/_tensordict/boundaries/vehicle/_tensordict"
    )
    cells_relative = f"{mesh_relative}/cells.memmap"
    points_relative = f"{mesh_relative}/points.memmap"
    cells_record = _geometry_file_record(geometry_case, cells_relative)
    points_record = _geometry_file_record(geometry_case, points_relative)
    selected_ids = np.arange(
        spec.historical_start,
        spec.historical_start + RESOLUTION,
        dtype=np.int64,
    )
    selected_cells_payload = _safe_hashed_pread(
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
    points = _safe_hashed_rows(
        case_root / points_relative,
        expected_file_size=int(points_record["size_bytes"]),
        expected_file_sha256=str(points_record["sha256"]),
        n_rows=n_master_points,
        row_indices=referenced,
    )

    raw_targets: dict[str, np.ndarray] = {}
    target_hashes: dict[str, str] = {}
    target_records = target_case["selected_targets"]
    for name, shape in (
        ("pressure", (RESOLUTION,)),
        ("wss", (RESOLUTION, 3)),
    ):
        record = target_records[name]
        relative = Path(str(record["source_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe selected-target path for {spec.case_id}/{name}")
        payload = _safe_selected_target(
            case_root / relative,
            expected_file_size=int(record["source_size_bytes"]),
            offset=int(record["source_offset_bytes"]),
            count=int(record["selected_size_bytes"]),
            expected_selected_sha256=str(record["selected_sha256"]),
        )
        raw_targets[name] = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
        target_hashes[f"{name}_selected_sha256"] = _sha256_bytes(payload)

    global_values = geometry_case.get("global_input_values_float32")
    if not isinstance(global_values, Mapping) or set(global_values) != {
        "U_inf",
        "p_inf",
        "rho_inf",
        "nu",
        "L_ref",
    }:
        raise ValueError(f"Frozen global inputs changed for {spec.case_id}")
    manifest_globals_float32 = {
        name: np.asarray(global_values[name], dtype="<f4")
        for name in ("U_inf", "p_inf", "rho_inf", "nu", "L_ref")
    }
    if (
        manifest_globals_float32["U_inf"].shape != (3,)
        or any(
            manifest_globals_float32[name].shape != (1,)
            for name in ("p_inf", "rho_inf", "nu", "L_ref")
        )
        or not all(
            np.isfinite(value).all() for value in manifest_globals_float32.values()
        )
    ):
        raise ValueError(f"Frozen global input shapes changed for {spec.case_id}")
    globals_float32 = {
        "U_inf": manifest_globals_float32["U_inf"],
        **{
            name: manifest_globals_float32[name].reshape(())
            for name in ("p_inf", "rho_inf", "nu", "L_ref")
        },
    }

    raw_mesh = runtime.mesh_type(
        points=torch.from_numpy(points),
        cells=torch.from_numpy(compacted_cells),
        cell_data={
            "pMeanTrim": torch.from_numpy(raw_targets["pressure"]),
            "wallShearStressMeanTrim": torch.from_numpy(raw_targets["wss"]),
        },
        global_data={
            name: torch.from_numpy(value) for name, value in globals_float32.items()
        },
    )
    if "_measure_weights" in raw_mesh.cell_data.keys():
        raise ValueError(
            "Raw reconstructed boundary unexpectedly carries measure weights"
        )
    arrays = {
        "selected_cell_ids_int64": selected_ids.astype("<i8", copy=False),
        "compacted_cells_int64": compacted_cells,
        "raw_target_pressure_float32": raw_targets["pressure"],
        "raw_target_wss_float32": raw_targets["wss"],
    }
    return raw_mesh, arrays, target_hashes


def _pipeline_globals_float32(domain: Any) -> np.ndarray:
    values: list[np.ndarray] = []
    for name, expected_shape in (
        ("U_inf", (3,)),
        ("p_inf", ()),
        ("rho_inf", ()),
        ("nu", ()),
        ("L_ref", ()),
        ("U_inf_dir", (3,)),
        ("reference_length", ()),
    ):
        if name not in domain.global_data:
            raise ValueError(f"Pipeline global field is missing: {name}")
        raw = domain.global_data[name].detach().float().cpu().numpy()
        if raw.shape != expected_shape or not np.isfinite(raw).all():
            raise ValueError(
                f"Pipeline global field changed: {name} shape={raw.shape}, "
                f"expected={expected_shape}"
            )
        value = np.asarray(raw, dtype="<f4").reshape(-1)
        values.append(value)
    result = np.concatenate(values).astype("<f4", copy=False)
    if result.shape != (len(GLOBAL_FIELD_ORDER),):
        raise RuntimeError("Pipeline global serialization contract changed")
    return result


def _redimensionalization_context(runtime: Any, dataset_config: Path) -> dict[str, Any]:
    import datasets as recipe_datasets
    import infer as recipe_infer
    from nondim import NonDimensionalizeByMetadata
    from omegaconf import OmegaConf

    normalizer = recipe_datasets.find_normalizer([runtime.dataset])
    if normalizer is None:
        raise ValueError("Replay dataset has no normalization transform")
    field_types = recipe_infer.build_redim_field_types(OmegaConf.load(dataset_config))
    if field_types != {"pressure": "pressure", "wss": "stress"}:
        raise ValueError(f"Re-dimensionalization field types changed: {field_types}")
    return {
        "function": recipe_infer.redimensionalize,
        "normalizer": normalizer,
        "nondim": NonDimensionalizeByMetadata(fields=field_types),
        "field_types": field_types,
    }


def _run_forward(
    runtime: Any,
    domain: Any,
    redim: Mapping[str, Any],
) -> dict[str, np.ndarray | float]:
    batch = runtime.collate_fn([(domain, {})])
    forward_kwargs = batch["forward_kwargs"]
    if set(forward_kwargs) != {"domain"}:
        raise ValueError(f"Historical forward kwargs changed: {set(forward_kwargs)}")
    if "canonical_source_geometry" in forward_kwargs:
        raise ValueError(
            "Replay forward kwargs unexpectedly contain canonical geometry"
        )
    model_domain = forward_kwargs["domain"]
    if model_domain is not domain:
        raise ValueError("Historical collate replaced the model DomainMesh object")
    _pipeline_globals_float32(model_domain)
    if "_measure_weights" in model_domain.boundaries["vehicle"].cell_data.keys():
        raise ValueError(
            "Historical model-call boundary carries source measure weights"
        )
    with torch.no_grad(), runtime.autocast_context(str(runtime.cfg.precision)):
        output = runtime.model(**forward_kwargs)
    prediction = runtime.normalize_output(
        output,
        TARGET_CONFIG,
        str(runtime.cfg.output_type),
    )
    targets = batch["targets"]
    expected_shapes = {
        "prediction_pressure": (RESOLUTION,),
        "prediction_wss": (RESOLUTION, 3),
        "truth_pressure": (RESOLUTION,),
        "truth_wss": (RESOLUTION, 3),
    }
    tensors = {
        "prediction_pressure": prediction["pressure"].float(),
        "prediction_wss": prediction["wss"].float(),
        "truth_pressure": targets["pressure"].float(),
        "truth_wss": targets["wss"].float(),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected {expected_shapes[name]}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")

    pred_physical = redim["function"](
        prediction,
        normalizer=redim["normalizer"],
        nondim=redim["nondim"],
        field_types=redim["field_types"],
        global_data=domain.global_data,
    )
    truth_physical = redim["function"](
        targets,
        normalizer=redim["normalizer"],
        nondim=redim["nondim"],
        field_types=redim["field_types"],
        global_data=domain.global_data,
    )
    result: dict[str, np.ndarray | float] = {
        "prediction_pressure_training": (
            tensors["prediction_pressure"].detach().cpu().numpy()
        ),
        "prediction_wss_training": tensors["prediction_wss"].detach().cpu().numpy(),
        "truth_pressure_training": tensors["truth_pressure"].detach().cpu().numpy(),
        "truth_wss_training": tensors["truth_wss"].detach().cpu().numpy(),
        "prediction_pressure_physical": (
            pred_physical["pressure"].detach().float().cpu().numpy()
        ),
        "prediction_wss_physical": (
            pred_physical["wss"].detach().float().cpu().numpy()
        ),
        "truth_pressure_physical": (
            truth_physical["pressure"].detach().float().cpu().numpy()
        ),
        "truth_wss_physical": (truth_physical["wss"].detach().float().cpu().numpy()),
    }
    return result


def _npz_prefix(ordinal: int, case_id: str) -> str:
    return f"case_{ordinal:02d}_{case_id}"


def _run_case(
    helper: ModuleType,
    runtime: Any,
    redim: Mapping[str, Any],
    dataset_root: Path,
    spec: Any,
    geometry_case: Mapping[str, Any],
    target_case: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    print(
        f"HISTORICAL_REPLAY_CASE_START ordinal={spec.cohort_ordinal} "
        f"case={spec.case_id}",
        flush=True,
    )
    raw_mesh, input_arrays, target_hashes = _load_explicit_raw_subset(
        runtime,
        dataset_root,
        spec,
        geometry_case,
        target_case,
    )
    raw_centroids, native_normals, native_areas = helper._native_geometry(raw_mesh)
    domain, center = helper._apply_pipeline(
        runtime,
        raw_mesh,
        fixed_center=None,
    )
    boundary = domain.boundaries["vehicle"]
    if tuple(boundary.cells.shape) != (RESOLUTION, 3) or tuple(
        domain.interior.points.shape
    ) != (RESOLUTION, 3):
        raise ValueError(f"Pipeline topology changed for {spec.case_id}")
    measure_weights_absent = "_measure_weights" not in boundary.cell_data.keys()
    if not measure_weights_absent:
        raise ValueError(
            f"Historical replay boundary carries measure weights for {spec.case_id}"
        )
    pipeline_cells = boundary.cells.detach().cpu().numpy().astype("<i8", copy=False)
    if not np.array_equal(pipeline_cells, input_arrays["compacted_cells_int64"]):
        raise ValueError(f"Pipeline connectivity changed for {spec.case_id}")
    current = _run_forward(runtime, domain, redim)

    current_queries = (
        domain.interior.points.detach().float().cpu().numpy().astype("<f4")
    )
    current_normals = (
        boundary.cell_data["normals"].detach().float().cpu().numpy().astype("<f4")
    )
    current_boundary_points = (
        boundary.points.detach().float().cpu().numpy().astype("<f4")
    )
    pipeline_globals = _pipeline_globals_float32(domain)

    prefix = _npz_prefix(spec.cohort_ordinal, spec.case_id)
    case_arrays = {
        "selected_cell_ids_int64": input_arrays["selected_cell_ids_int64"],
        "compacted_cells_int64": pipeline_cells,
        "raw_centroids_float32": raw_centroids.astype("<f4", copy=False),
        "native_normals_float32": native_normals.astype("<f4", copy=False),
        "native_areas_float64": native_areas.astype("<f8", copy=False),
        "raw_target_pressure_float32": input_arrays["raw_target_pressure_float32"],
        "raw_target_wss_float32": input_arrays["raw_target_wss_float32"],
        "pipeline_boundary_points_float32": current_boundary_points,
        "pipeline_queries_float32": current_queries.astype("<f4", copy=False),
        "pipeline_normals_float32": current_normals.astype("<f4", copy=False),
        "pipeline_globals_float32": pipeline_globals,
        "prediction_pressure_training_float32": np.asarray(
            current["prediction_pressure_training"], dtype="<f4"
        ),
        "prediction_wss_training_float32": np.asarray(
            current["prediction_wss_training"], dtype="<f4"
        ),
        "truth_pressure_training_float32": np.asarray(
            current["truth_pressure_training"], dtype="<f4"
        ),
        "truth_wss_training_float32": np.asarray(
            current["truth_wss_training"], dtype="<f4"
        ),
        "prediction_pressure_physical_float32": np.asarray(
            current["prediction_pressure_physical"], dtype="<f4"
        ),
        "prediction_wss_physical_float32": np.asarray(
            current["prediction_wss_physical"], dtype="<f4"
        ),
        "truth_pressure_physical_float32": np.asarray(
            current["truth_pressure_physical"], dtype="<f4"
        ),
        "truth_wss_physical_float32": np.asarray(
            current["truth_wss_physical"], dtype="<f4"
        ),
        "pipeline_center_float32": (
            center.detach().float().cpu().numpy().astype("<f4")
        ),
    }
    for name, value in case_arrays.items():
        arrays[f"{prefix}__{name}"] = np.ascontiguousarray(value)
    array_hashes = {
        name: _array_sha256(value, value.dtype) for name, value in case_arrays.items()
    }
    print(
        f"HISTORICAL_REPLAY_CASE_DONE ordinal={spec.cohort_ordinal} "
        f"case={spec.case_id}",
        flush=True,
    )
    return {
        "cohort_ordinal": spec.cohort_ordinal,
        "case_id": spec.case_id,
        "reader_index": spec.reader_index,
        "n_master_cells": spec.n_master_cells,
        "historical_start": spec.historical_start,
        "resolution": RESOLUTION,
        "n_compacted_points": int(current_boundary_points.shape[0]),
        "measure_weights_absent": measure_weights_absent,
        "target_input_verification": target_hashes,
        "array_sha256": array_hashes,
    }


def _validate_output_targets(*outputs: Path) -> None:
    destinations: list[Path] = []
    for output in outputs:
        normalized = Path(os.path.abspath(os.path.normpath(output)))
        destinations.extend(
            (normalized, normalized.with_name(f"{normalized.name}.sha256"))
        )
    if len(set(destinations)) != len(destinations):
        raise ValueError("Output paths and sidecars must be pairwise distinct")
    if len({path.resolve(strict=False) for path in destinations}) != len(destinations):
        raise ValueError("Resolved output paths and sidecars alias")
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")


def _write_fsynced_temporary(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(name)
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
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary, _sha256_file(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def _publish_output_set(
    *,
    output_json: Path,
    json_payload: bytes,
    output_npz: Path,
    npz_temporary: Path,
    npz_sha256: str,
) -> str:
    json_sha256 = _sha256_bytes(json_payload)
    json_sidecar = output_json.with_name(f"{output_json.name}.sha256")
    npz_sidecar = output_npz.with_name(f"{output_npz.name}.sha256")
    json_sidecar_payload = f"{json_sha256}  {output_json.name}\n".encode("ascii")
    npz_sidecar_payload = f"{npz_sha256}  {output_npz.name}\n".encode("ascii")
    temporaries: dict[Path, Path] = {output_npz: npz_temporary}
    published: list[tuple[Path, Path]] = []
    try:
        _validate_output_targets(output_json, output_npz)
        temporaries[output_json] = _write_fsynced_temporary(output_json, json_payload)
        temporaries[json_sidecar] = _write_fsynced_temporary(
            json_sidecar, json_sidecar_payload
        )
        temporaries[npz_sidecar] = _write_fsynced_temporary(
            npz_sidecar, npz_sidecar_payload
        )
        for destination in (output_npz, npz_sidecar, output_json, json_sidecar):
            temporary = temporaries[destination]
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        _fsync_directories(path.parent for path in temporaries)
        if (
            _sha256_file(output_json) != json_sha256
            or _sha256_file(output_npz) != npz_sha256
            or _safe_read_bytes(json_sidecar) != json_sidecar_payload
            or _safe_read_bytes(npz_sidecar) != npz_sidecar_payload
        ):
            raise OSError("Published replay artifact changed")
    except BaseException:
        for destination, temporary in reversed(published):
            _unlink_if_same_inode(destination, temporary)
        _fsync_directories(path.parent for path in temporaries)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        _fsync_directories(path.parent for path in temporaries)
    return json_sha256


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B")),
        }
        for name, value in sorted(arrays.items())
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--target-input-manifest", type=Path, required=True)
    parser.add_argument("--replay-label", choices=("A", "B"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve_regular_input(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _resolve_directory_input(
    path: Path,
    label: str,
    *,
    allow_symlink: bool = False,
) -> Path:
    if (not allow_symlink and path.is_symlink()) or not path.is_dir():
        raise ValueError(f"{label} must be a valid directory: {path}")
    return path.resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("Historical replay requires a little-endian host")
    args = _parse_args(argv)
    args.repo_root = _resolve_directory_input(args.repo_root, "Repository root")
    args.dataset_root = _resolve_directory_input(
        args.dataset_root,
        "Dataset root",
    )
    args.checkpoint_dir = _resolve_directory_input(
        args.checkpoint_dir,
        "Checkpoint directory",
    )
    for name in (
        "dataset_config",
        "resolved_config",
        "geometry_manifest",
        "target_input_manifest",
    ):
        setattr(
            args,
            name,
            _resolve_regular_input(getattr(args, name), name.replace("_", " ")),
        )
    args.output_json = Path(os.path.abspath(args.output_json))
    args.output_npz = Path(os.path.abspath(args.output_npz))
    _validate_output_targets(args.output_json, args.output_npz)

    helper_path = Path(__file__).resolve(strict=True).with_name(HELPER_FILENAME)
    _require_sha256(helper_path, EXPECTED_HELPER_SHA256, "Blind replay runtime")
    helper = _load_module(helper_path, "frozen_blind_historical_replay_runtime")
    _validate_case_specs(helper)
    static_inputs = _validate_static_inputs(
        helper,
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config=args.dataset_config,
        resolved_config=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    geometry_verification = _verify_geometry_manifest(
        helper,
        args.geometry_manifest,
        args.dataset_root,
    )
    geometry_cases = geometry_verification.pop("case_records")
    target_input_verification = _verify_target_input_manifest(
        args.target_input_manifest,
        args.dataset_root,
    )
    target_cases = target_input_verification.pop("case_records")

    runtime = helper._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    import_provenance = _validate_import_provenance(args.repo_root)
    _validate_reader(runtime)
    if str(runtime.cfg.precision) != "bfloat16":
        raise ValueError(f"Replay precision changed: {runtime.cfg.precision}")
    redim = _redimensionalization_context(runtime, args.dataset_config)

    arrays: dict[str, np.ndarray] = {}
    cases = [
        _run_case(
            helper,
            runtime,
            redim,
            args.dataset_root,
            spec,
            geometry_case,
            target_case,
            arrays,
        )
        for spec, geometry_case, target_case in zip(
            helper.CASE_SPECS,
            geometry_cases,
            target_cases,
            strict=True,
        )
    ]

    npz_temporary, npz_sha256 = _prepare_npz_temporary(args.output_npz, arrays)
    try:
        if len(cases) != 36 or len(arrays) != 36 * 20:
            raise RuntimeError(
                f"Replay output coverage changed: "
                f"cases={len(cases)} arrays={len(arrays)}"
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "replay_label": args.replay_label,
            "contract": {
                "legacy_call": "model(domain)",
                "canonical_source_geometry_present": False,
                "producer_reads_archive_or_metrics": False,
                "candidate_or_canonical_arm_present": False,
                "resolution": RESOLUTION,
                "precision": "bfloat16",
                "case_count": 36,
                "global_field_order": list(GLOBAL_FIELD_ORDER),
                "measure_weights_required_absent": True,
            },
            "summary": {
                "case_count": len(cases),
                "array_count": len(arrays),
                "measure_weights_absent_case_count": sum(
                    case["measure_weights_absent"] for case in cases
                ),
            },
            "cases": cases,
            "npz": {
                "filename": args.output_npz.name,
                "sha256": npz_sha256,
                "array_count": len(arrays),
                "array_manifest": _array_manifest(arrays),
            },
            "provenance": {
                "command": list(sys.argv),
                "producer_path": str(Path(__file__).resolve()),
                "producer_sha256": _sha256_file(Path(__file__).resolve()),
                "helper_path": str(helper_path),
                "helper_sha256": EXPECTED_HELPER_SHA256,
                "repo_root": str(args.repo_root),
                "dataset_root": str(args.dataset_root),
                "static_inputs": static_inputs,
                "geometry_verification": geometry_verification,
                "target_input_verification": target_input_verification,
                "historical_input_freeze_sha256": (
                    EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256
                ),
                "import_provenance": import_provenance,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(runtime.device),
            },
        }
        payload = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        digest = _publish_output_set(
            output_json=args.output_json,
            json_payload=payload,
            output_npz=args.output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )
    finally:
        npz_temporary.unlink(missing_ok=True)
    print(
        f"{STATUS} replay={args.replay_label} "
        f"json_sha256={digest} npz_sha256={npz_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
