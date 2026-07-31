# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Freeze raw targets for the blind, nested-resolution canonical H-QC reducer.

This target-only producer never opens a model, prediction, metric, or decision
threshold.  It reads one ordered cyclic Kmax=40,000 target panel per frozen
case.  Every smaller source panel and the fixed Q=2,500 scoring panel are exact
prefixes of the emitted arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_KIND = "drivaerml_hqc_nested_raw_target_bundle"
STATUS = "PASSED_HQC_NESTED_RAW_TARGET_FREEZE"
RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
MAX_RESOLUTION = max(RESOLUTIONS)
FIXED_QUERY_RESOLUTION = min(RESOLUTIONS)
HISTORICAL_ANCHOR_RESOLUTION = 10_000
EXPECTED_CASE_COUNT = 36
DATASET_MANIFEST_SHA256 = (
    "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
)
GEOMETRY_MANIFEST_SHA256 = (
    "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
)
HISTORICAL_TARGET_MANIFEST_SHA256 = (
    "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49"
)
EXPECTED_COHORT_SHA256 = (
    "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
)
GEOMETRY_ARTIFACT_KIND = "drivaerml_target_free_geometry_input_manifest"
GEOMETRY_STATUS = "PASSED_TARGET_FREE_GEOMETRY_INPUT_FREEZE"
HISTORICAL_TARGET_ARTIFACT_KIND = (
    "drivaerml_historical_k10000_selected_target_input_manifest"
)
HISTORICAL_TARGET_STATUS = "PASSED_HISTORICAL_K10000_SELECTED_TARGET_INPUT_FREEZE"
PHYSICAL_GLOBAL_FIELD_ORDER = (
    "U_inf_x",
    "U_inf_y",
    "U_inf_z",
    "p_inf",
    "rho_inf",
    "nu",
    "L_ref",
)

CASE_SPECS = (
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

TARGETS = {
    "pressure": {
        "field": "pMeanTrim",
        "shape_suffix": (),
        "components": 1,
    },
    "wss": {
        "field": "wallShearStressMeanTrim",
        "shape_suffix": (3,),
        "components": 3,
    },
}


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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _safe_read_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"Input changed while being read: {path}")
    return b"".join(chunks)


def _strict_json_bytes(payload: bytes, *, source: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {source}")

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _cyclic_f32_byte_spans(
    *,
    n_rows: int,
    start_row: int,
    row_count: int,
    components: int,
) -> tuple[tuple[int, int], ...]:
    """Return one or two ordered byte spans for unique cyclic float32 rows."""
    values = {
        "n_rows": n_rows,
        "start_row": start_row,
        "row_count": row_count,
        "components": components,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise TypeError("Cyclic row parameters must be integers")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if components <= 0:
        raise ValueError("components must be positive")
    if start_row < 0 or start_row >= n_rows:
        raise ValueError("start_row must lie within the source")
    if row_count <= 0 or row_count > n_rows:
        raise ValueError("row_count must be in [1, n_rows]")

    row_bytes = components * 4
    tail_rows = min(row_count, n_rows - start_row)
    spans = [(start_row * row_bytes, tail_rows * row_bytes)]
    head_rows = row_count - tail_rows
    if head_rows:
        spans.append((0, head_rows * row_bytes))
    return tuple(spans)


def _safe_cyclic_f32_rows(
    path: Path,
    *,
    n_rows: int,
    start_row: int,
    row_count: int,
    components: int,
) -> tuple[bytes, tuple[tuple[int, int], ...]]:
    """Read cyclic float32 rows from one stable regular-file descriptor."""
    spans = _cyclic_f32_byte_spans(
        n_rows=n_rows,
        start_row=start_row,
        row_count=row_count,
        components=components,
    )
    expected_file_size = n_rows * components * 4
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Target source is not a regular file: {path}")
        if before.st_size != expected_file_size:
            raise ValueError(
                f"Target source size changed: {path} "
                f"({before.st_size} != {expected_file_size})"
            )
        chunks = []
        for offset, count in spans:
            chunk = os.pread(descriptor, count, offset)
            if len(chunk) != count:
                raise ValueError(
                    f"Short target read from {path}: {len(chunk)} != {count}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"Target source changed while being read: {path}")
    return b"".join(chunks), spans


def _cohort_document() -> list[dict[str, int | str]]:
    return [
        {
            "cohort_ordinal": row[0],
            "case_id": row[1],
            "reader_index": row[2],
            "n_master_cells": row[3],
            "historical_start": row[4],
        }
        for row in CASE_SPECS
    ]


def _validate_geometry_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = _safe_read_bytes(path)
    digest = _sha256_bytes(payload)
    if digest != GEOMETRY_MANIFEST_SHA256:
        raise ValueError("Geometry input manifest changed")
    document = _strict_json_bytes(payload, source=str(path))
    if not isinstance(document, Mapping):
        raise ValueError("Geometry input manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_ARTIFACT_KIND,
        "status": GEOMETRY_STATUS,
        "case_count": EXPECTED_CASE_COUNT,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"Geometry input manifest {key!r} changed")
    dataset_record = document.get("dataset_manifest")
    if (
        not isinstance(dataset_record, Mapping)
        or dataset_record.get("sha256") != DATASET_MANIFEST_SHA256
    ):
        raise ValueError("Geometry input manifest dataset identity changed")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("Geometry input manifest case coverage changed")
    return dict(document), digest


def _validate_historical_target_manifest(
    path: Path,
) -> tuple[dict[str, Any], str]:
    payload = _safe_read_bytes(path)
    digest = _sha256_bytes(payload)
    if digest != HISTORICAL_TARGET_MANIFEST_SHA256:
        raise ValueError("Historical K=10k target manifest changed")
    document = _strict_json_bytes(payload, source=str(path))
    if not isinstance(document, Mapping):
        raise ValueError("Historical K=10k target manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "artifact_kind": HISTORICAL_TARGET_ARTIFACT_KIND,
        "status": HISTORICAL_TARGET_STATUS,
        "case_count": EXPECTED_CASE_COUNT,
        "resolution": HISTORICAL_ANCHOR_RESOLUTION,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"Historical K=10k target manifest {key!r} changed")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("Historical K=10k target case coverage changed")
    return dict(document), digest


def _geometry_cases_by_ordinal(
    document: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for raw_case in document["cases"]:
        if not isinstance(raw_case, Mapping):
            raise ValueError("Geometry input case must be a JSON object")
        ordinal = raw_case.get("cohort_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("Geometry input case ordinal is invalid")
        if ordinal in result:
            raise ValueError(f"Duplicate geometry input ordinal {ordinal}")
        result[ordinal] = raw_case
    if set(result) != set(range(EXPECTED_CASE_COUNT)):
        raise ValueError("Geometry input ordinals are incomplete")
    return result


def _historical_target_cases_by_ordinal(
    document: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for raw_case in document["cases"]:
        if not isinstance(raw_case, Mapping):
            raise ValueError("Historical K=10k target case must be a JSON object")
        ordinal = raw_case.get("cohort_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("Historical K=10k target case ordinal is invalid")
        if ordinal in result:
            raise ValueError(f"Duplicate historical target ordinal {ordinal}")
        result[ordinal] = raw_case
    if set(result) != set(range(EXPECTED_CASE_COUNT)):
        raise ValueError("Historical K=10k target ordinals are incomplete")
    return result


def _physical_globals(case: Mapping[str, Any], *, case_id: str) -> np.ndarray:
    values = case.get("global_input_values_float32")
    if not isinstance(values, Mapping) or set(values) != {
        "U_inf",
        "p_inf",
        "rho_inf",
        "nu",
        "L_ref",
    }:
        raise ValueError(f"{case_id} geometry globals changed")
    expected_shapes = {
        "U_inf": (3,),
        "p_inf": (1,),
        "rho_inf": (1,),
        "nu": (1,),
        "L_ref": (1,),
    }
    fields = {}
    for name, shape in expected_shapes.items():
        value = np.asarray(values[name], dtype="<f4")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{case_id} geometry global {name!r} changed")
        fields[name] = value
    result = np.concatenate(
        (
            fields["U_inf"],
            fields["p_inf"],
            fields["rho_inf"],
            fields["nu"],
            fields["L_ref"],
        )
    ).astype("<f4", copy=False)
    q_inf = (
        np.float32(0.5)
        * fields["rho_inf"][0]
        * np.sum(
            fields["U_inf"] * fields["U_inf"],
            dtype=np.float32,
        )
    )
    if not np.isfinite(q_inf) or q_inf <= 0.0:
        raise ValueError(f"{case_id} dynamic-pressure scale is invalid")
    return result


def _selected_ids(n_rows: int, start_row: int) -> np.ndarray:
    _cyclic_f32_byte_spans(
        n_rows=n_rows,
        start_row=start_row,
        row_count=MAX_RESOLUTION,
        components=1,
    )
    tail_rows = min(MAX_RESOLUTION, n_rows - start_row)
    tail = np.arange(start_row, start_row + tail_rows, dtype="<i8")
    head = np.arange(MAX_RESOLUTION - tail_rows, dtype="<i8")
    return np.concatenate((tail, head))


def _prefix_hashes(payload: bytes, *, components: int) -> dict[str, str]:
    return {
        str(resolution): _sha256_bytes(payload[: resolution * components * 4])
        for resolution in RESOLUTIONS
    }


def _array_record(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "nbytes": contiguous.nbytes,
        "sha256": _sha256_bytes(contiguous.tobytes(order="C")),
    }


def _inspect_case(
    dataset_root: Path,
    spec: tuple[int, str, int, int, int],
    geometry_case: Mapping[str, Any],
    historical_target_case: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ordinal, case_id, reader_index, n_cells, start = spec
    expected_geometry = {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": reader_index,
        "n_master_cells": n_cells,
        "historical_start": start,
    }
    for key, value in expected_geometry.items():
        if geometry_case.get(key) != value:
            raise ValueError(f"{case_id} geometry manifest {key!r} changed")
        if historical_target_case.get(key) != value:
            raise ValueError(
                f"{case_id} historical K=10k target manifest {key!r} changed"
            )
    if historical_target_case.get("resolution") != HISTORICAL_ANCHOR_RESOLUTION:
        raise ValueError(f"{case_id} historical K=10k target resolution changed")

    case_link = dataset_root / case_id
    if not case_link.is_symlink():
        raise ValueError(f"Dataset case is not a symlink: {case_link}")
    symlink_target = os.readlink(case_link)
    case_root = case_link.resolve(strict=True)
    if str(case_root) != geometry_case.get("resolved_case_root"):
        raise ValueError(f"{case_id} resolved dataset path changed")
    if str(case_root) != historical_target_case.get("resolved_case_root"):
        raise ValueError(f"{case_id} historical target resolved path changed")
    if symlink_target != geometry_case.get("symlink_target"):
        raise ValueError(f"{case_id} dataset symlink target changed")
    if symlink_target != historical_target_case.get("symlink_target"):
        raise ValueError(f"{case_id} historical target symlink target changed")
    tensor_root = (
        case_root
        / f"domain_{case_id}.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
    )
    metadata_path = tensor_root / "cell_data" / "meta.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"Target metadata is not a regular file: {metadata_path}")
    metadata_payload = _safe_read_bytes(metadata_path)
    metadata = _strict_json_bytes(metadata_payload, source=str(metadata_path))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{case_id} target metadata must be a JSON object")
    historical_metadata = historical_target_case.get("cell_data_metadata")
    if (
        not isinstance(historical_metadata, Mapping)
        or historical_metadata.get("size_bytes") != len(metadata_payload)
        or historical_metadata.get("sha256") != _sha256_bytes(metadata_payload)
    ):
        raise ValueError(f"{case_id} historical target metadata identity changed")
    historical_targets = historical_target_case.get("selected_targets")
    if not isinstance(historical_targets, Mapping) or set(historical_targets) != set(
        TARGETS
    ):
        raise ValueError(f"{case_id} historical target field coverage changed")

    array_prefix = f"case_{ordinal:02d}_{case_id}__"
    ids = _selected_ids(n_cells, start)
    arrays = {
        f"{array_prefix}selected_cell_ids_int64": ids,
        f"{array_prefix}physical_globals_float32": _physical_globals(
            geometry_case,
            case_id=case_id,
        ),
    }
    target_records: dict[str, Any] = {}
    for name, contract in TARGETS.items():
        field = str(contract["field"])
        shape_suffix = tuple(contract["shape_suffix"])
        components = int(contract["components"])
        entry = metadata.get(field)
        expected_shape = [n_cells, *shape_suffix]
        if (
            not isinstance(entry, Mapping)
            or entry.get("shape") != expected_shape
            or entry.get("dtype") != "torch.float32"
            or entry.get("device") != "cpu"
        ):
            raise ValueError(f"{case_id} target metadata changed for {field}")
        source_path = tensor_root / "cell_data" / f"{field}.memmap"
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"Target source is not a regular file: {source_path}")
        payload, spans = _safe_cyclic_f32_rows(
            source_path,
            n_rows=n_cells,
            start_row=start,
            row_count=MAX_RESOLUTION,
            components=components,
        )
        payload_sha256 = _sha256_bytes(payload)
        prefix_hashes = _prefix_hashes(payload, components=components)
        historical_record = historical_targets[name]
        expected_historical_record = {
            "raw_field_name": field,
            "source_relative_path": (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                f"_tensordict/cell_data/{field}.memmap"
            ),
            "source_size_bytes": n_cells * components * 4,
            "source_offset_bytes": start * components * 4,
            "selected_size_bytes": HISTORICAL_ANCHOR_RESOLUTION * components * 4,
            "selected_shape": [HISTORICAL_ANCHOR_RESOLUTION, *shape_suffix],
            "selected_dtype": "float32_little_endian",
        }
        if not isinstance(historical_record, Mapping):
            raise ValueError(f"{case_id} historical {name} record changed")
        for key, expected_value in expected_historical_record.items():
            if historical_record.get(key) != expected_value:
                raise ValueError(f"{case_id} historical {name} target {key!r} changed")
        if (
            historical_record.get("selected_sha256")
            != prefix_hashes[str(HISTORICAL_ANCHOR_RESOLUTION)]
        ):
            raise ValueError(
                f"{case_id} {name} K=10k prefix differs from sealed target manifest"
            )
        value = np.frombuffer(payload, dtype="<f4").reshape(
            MAX_RESOLUTION,
            *shape_suffix,
        )
        if not np.isfinite(value).all():
            raise ValueError(
                f"{case_id} selected raw {name} contains non-finite values"
            )
        arrays[f"{array_prefix}raw_target_{name}_float32"] = value.copy()
        target_records[name] = {
            "raw_field_name": field,
            "source_relative_path": (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                f"_tensordict/cell_data/{field}.memmap"
            ),
            "source_size_bytes": n_cells * components * 4,
            "source_spans_bytes": [
                {"offset": offset, "count": count} for offset, count in spans
            ],
            "selected_shape": [MAX_RESOLUTION, *shape_suffix],
            "selected_dtype": "float32_little_endian",
            "selected_sha256": payload_sha256,
            "prefix_sha256_by_resolution": prefix_hashes,
            "historical_k10000_prefix_authenticated": True,
        }

    selection_hashes = {
        str(resolution): _sha256_bytes(
            np.ascontiguousarray(ids[:resolution]).tobytes(order="C")
        )
        for resolution in RESOLUTIONS
    }
    return (
        {
            "cohort_ordinal": ordinal,
            "case_id": case_id,
            "reader_index": reader_index,
            "n_master_cells": n_cells,
            "historical_start": start,
            "max_resolution": MAX_RESOLUTION,
            "selection": {
                "kind": "ordered_cyclic_prefix_from_historical_k10000_start",
                "wraps": start + MAX_RESOLUTION > n_cells,
                "selected_cell_ids_sha256_by_resolution": selection_hashes,
            },
            "logical_case_symlink": str(case_link),
            "symlink_target": symlink_target,
            "resolved_case_root": str(case_root),
            "cell_data_metadata": {
                "relative_path": (
                    f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                    "_tensordict/cell_data/meta.json"
                ),
                "size_bytes": len(metadata_payload),
                "sha256": _sha256_bytes(metadata_payload),
            },
            "targets": target_records,
        },
        arrays,
    )


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _atomic_publish_bundle(payloads: Mapping[Path, bytes]) -> None:
    if not payloads:
        raise ValueError("No bundle payloads were provided")
    paths = list(payloads)
    if len(set(paths)) != len(paths):
        raise ValueError("Bundle destinations must be distinct")
    for destination in paths:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        if destination.parent.is_symlink() or not destination.parent.is_dir():
            raise ValueError(f"Output directory is invalid: {destination.parent}")

    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for destination, content in payloads.items():
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(name)
            temporaries[destination] = temporary
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for destination in paths:
            temporary = temporaries[destination]
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        for parent in {destination.parent for destination in paths}:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for destination, expected in payloads.items():
            if _safe_read_bytes(destination) != expected:
                raise RuntimeError(
                    f"Published bundle payload verification failed: {destination}"
                )
    except BaseException:
        for destination, temporary in reversed(published):
            try:
                destination_stat = destination.stat(follow_symlinks=False)
                temporary_stat = temporary.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (destination_stat.st_dev, destination_stat.st_ino) == (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            ):
                destination.unlink()
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _validate_output_paths(output_json: Path, output_npz: Path) -> None:
    paths = (
        output_npz,
        output_npz.with_name(f"{output_npz.name}.sha256"),
        output_json,
        output_json.with_name(f"{output_json.name}.sha256"),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Output and sidecar paths must be distinct")
    for destination in paths:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        if destination.parent.is_symlink() or not destination.parent.is_dir():
            raise ValueError(f"Output directory is invalid: {destination.parent}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument(
        "--historical-k10000-target-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset_root_input = Path(os.path.abspath(args.dataset_root))
    if dataset_root_input.is_symlink() or not dataset_root_input.is_dir():
        raise ValueError("Dataset root must be a regular directory")
    dataset_root = dataset_root_input.resolve(strict=True)
    geometry_path = Path(os.path.abspath(args.geometry_manifest))
    historical_target_path = Path(
        os.path.abspath(args.historical_k10000_target_manifest)
    )
    output_json = Path(os.path.abspath(args.output_json))
    output_npz = Path(os.path.abspath(args.output_npz))
    if output_json == output_npz:
        raise ValueError("JSON and NPZ outputs must be distinct")
    _validate_output_paths(output_json, output_npz)

    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest_payload = _safe_read_bytes(dataset_manifest_path)
    if _sha256_bytes(dataset_manifest_payload) != DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest changed")
    dataset_manifest = _strict_json_bytes(
        dataset_manifest_payload,
        source=str(dataset_manifest_path),
    )
    expected_case_ids = [spec[1] for spec in CASE_SPECS]
    cohort_sha256 = _sha256_bytes(_canonical_json_bytes(_cohort_document()))
    if len(CASE_SPECS) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Frozen target cohort size changed")
    if cohort_sha256 != EXPECTED_COHORT_SHA256:
        raise RuntimeError("Frozen target cohort identity changed")
    if (
        not isinstance(dataset_manifest, Mapping)
        or dataset_manifest.get("id_reference") != expected_case_ids
    ):
        raise RuntimeError("Dataset reference cohort order changed")

    geometry_manifest, geometry_digest = _validate_geometry_manifest(geometry_path)
    geometry_cases = _geometry_cases_by_ordinal(geometry_manifest)
    historical_target_manifest, historical_target_digest = (
        _validate_historical_target_manifest(historical_target_path)
    )
    historical_target_cases = _historical_target_cases_by_ordinal(
        historical_target_manifest
    )
    cases = []
    arrays: dict[str, np.ndarray] = {}
    for spec in CASE_SPECS:
        case, case_arrays = _inspect_case(
            dataset_root,
            spec,
            geometry_cases[spec[0]],
            historical_target_cases[spec[0]],
        )
        cases.append(case)
        overlap = set(arrays) & set(case_arrays)
        if overlap:
            raise RuntimeError(f"Duplicate target array keys: {sorted(overlap)}")
        arrays.update(case_arrays)
    if [case["case_id"] for case in cases] != expected_case_ids:
        raise RuntimeError("Target cohort coverage changed")

    array_manifest = {name: _array_record(value) for name, value in arrays.items()}
    npz_payload = _npz_bytes(arrays)
    npz_digest = _sha256_bytes(npz_payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_input": str(dataset_root_input),
        "dataset_root_resolved": str(dataset_root),
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "geometry_manifest": {
            "path": str(geometry_path),
            "sha256": geometry_digest,
        },
        "historical_k10000_target_manifest": {
            "path": str(historical_target_path),
            "sha256": historical_target_digest,
            "prefix_hashes_authenticated": EXPECTED_CASE_COUNT * len(TARGETS),
        },
        "case_count": len(cases),
        "resolutions": list(RESOLUTIONS),
        "max_resolution": MAX_RESOLUTION,
        "fixed_query_resolution": FIXED_QUERY_RESOLUTION,
        "physical_globals": {
            "array_suffix": "physical_globals_float32",
            "field_order": list(PHYSICAL_GLOBAL_FIELD_ORDER),
            "dtype": "float32_little_endian",
            "source": "frozen target-free geometry manifest",
            "transformed_by_target_freezer": False,
        },
        "selection": (
            "one Kmax ordered cyclic panel; every smaller S_k and fixed Q are "
            "exact array prefixes"
        ),
        "read_allowlist": [
            "dataset manifest.json",
            "frozen target-free geometry manifest",
            "frozen historical K=10k target manifest",
            "vehicle cell_data/meta.json",
            "vehicle cell_data/pMeanTrim.memmap selected byte spans only",
            (
                "vehicle cell_data/wallShearStressMeanTrim.memmap selected "
                "byte spans only"
            ),
        ],
        "read_exclusions": {
            "model_opened": False,
            "prediction_opened": False,
            "metric_opened": False,
            "decision_threshold_opened": False,
            "other_cell_data_opened": False,
            "point_data_opened": False,
            "interior_opened": False,
        },
        "publication_contract": {
            "json_manifest_linked_last": True,
            "producer_outputs_are_not_a_commit_marker": True,
            "valid_only_after_external_sidecar_checks_and_done_marker": True,
            "interrupted_partial_bundle_must_not_be_overwritten": True,
        },
        "cases": cases,
        "cohort_sha256": cohort_sha256,
        "npz": {
            "path": str(output_npz),
            "sha256": npz_digest,
            "array_count": len(arrays),
        },
        "array_manifest": array_manifest,
        "provenance": {
            "command": list(os.sys.argv),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_bytes(_safe_read_bytes(Path(__file__))),
            "numpy": np.__version__,
        },
    }
    json_payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    json_digest = _sha256_bytes(json_payload)
    payloads = {
        output_npz: npz_payload,
        output_npz.with_name(f"{output_npz.name}.sha256"): (
            f"{npz_digest}  {output_npz.name}\n".encode("ascii")
        ),
        output_json.with_name(f"{output_json.name}.sha256"): (
            f"{json_digest}  {output_json.name}\n".encode("ascii")
        ),
        # Link the self-describing JSON last. Even then it is evidence only
        # after the wrapper verifies both sidecars and writes its DONE marker.
        output_json: json_payload,
    }
    _atomic_publish_bundle(payloads)
    print(
        f"{STATUS} cases={len(cases)} arrays={len(arrays)} "
        f"json_sha256={json_digest} npz_sha256={npz_digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
