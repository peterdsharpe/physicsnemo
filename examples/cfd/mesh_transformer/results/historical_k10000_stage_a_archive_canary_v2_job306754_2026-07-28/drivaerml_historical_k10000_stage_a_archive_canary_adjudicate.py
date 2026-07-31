# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independently adjudicate the two-lane Stage-A archive-domain canary.

This reducer same-byte loads both process-isolated lane artifacts and only the
two manifest-bound archived prediction payloads for fixed ``run_118``.  It is
the sole publisher of the preregistered four-way categorical result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_stage_a_archive_domain_adjudication"
STATUS = "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_ADJUDICATION"

EXACT_PASS = "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS"  # noqa: S105
VALID_REFUTATION = "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION"
INVALID = "INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT"
INCOMPLETE = "INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY"

PRODUCER_ARTIFACT_KIND = "phase1_historical_k10000_stage_a_archive_domain_producer"
PRODUCER_STATUS = "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_PRODUCER"
EXPECTED_PRODUCER_SHA256 = (
    "b596cb3d4a82b30255324b982f5d84d1260f53963b1697c7b2fd9d12049ed8c0"
)
EXPECTED_HISTORICAL_MANIFEST_SHA256 = (
    "545b1f6e906002231415b84277db00eec04f3666233b8da637514e9077a585eb"
)
EXPECTED_INPUT_FREEZE_SHA256 = (
    "fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10"
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
EXPECTED_CURRENT_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_CURRENT_INFER_SHA256 = (
    "47aec675e54d58ee4202831ae0d20039b1ff5ec40e69cc8b5087ce00bd5234ed"
)
EXPECTED_CURRENT_MODEL_SOURCE_SHA256 = (
    "9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892"
)

CASE_ID = "run_118"
READER_INDEX = 21
RESOLUTION = 10_000
BOUNDARY_POINT_COUNT = 29_949
CASE_DIRECTORY = "00021_run_118_domain_run_118.pdmsh"
PRECISION = "bfloat16"
EPOCH = 491

DIRECT_ARRAY_SPECS: Mapping[str, tuple[Path, tuple[int, ...], np.dtype[Any]]] = {
    "archive_boundary_points_float32": (
        Path(
            f"{CASE_DIRECTORY}/_tensordict/boundaries/vehicle/_tensordict/points.memmap"
        ),
        (BOUNDARY_POINT_COUNT, 3),
        np.dtype("<f4"),
    ),
    "archive_boundary_cells_int64": (
        Path(
            f"{CASE_DIRECTORY}/_tensordict/boundaries/vehicle/_tensordict/cells.memmap"
        ),
        (RESOLUTION, 3),
        np.dtype("<i8"),
    ),
    "archive_query_points_float32": (
        Path(f"{CASE_DIRECTORY}/_tensordict/interior/_tensordict/points.memmap"),
        (RESOLUTION, 3),
        np.dtype("<f4"),
    ),
}
GLOBAL_SHAPES: Mapping[str, tuple[int, ...]] = {
    "L_ref": (),
    "U_inf": (3,),
    "U_inf_dir": (3,),
    "nu": (),
    "p_inf": (),
    "reference_length": (),
    "rho_inf": (),
}
GLOBAL_ROOT = Path(f"{CASE_DIRECTORY}/_tensordict/global_data")
for _field, _shape in GLOBAL_SHAPES.items():
    DIRECT_ARRAY_SPECS[f"archive_global_{_field}_float32"] = (
        GLOBAL_ROOT / f"{_field}.memmap",
        _shape,
        np.dtype("<f4"),
    )

ENCODED_ARRAY_SCHEMAS: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "encoded_source_points_float32": (
        (BOUNDARY_POINT_COUNT, 3),
        np.dtype("<f4"),
    ),
    "encoded_source_cells_int64": ((RESOLUTION, 3), np.dtype("<i8")),
    "encoded_source_centroids_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "encoded_source_areas_float32": ((RESOLUTION,), np.dtype("<f4")),
    "encoded_source_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "encoded_center_float32": ((3,), np.dtype("<f4")),
    "encoded_reference_length_float32": ((), np.dtype("<f4")),
}
PREDICTION_ARRAY_SCHEMAS: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "prediction_pressure_physical_float32": ((RESOLUTION,), np.dtype("<f4")),
    "prediction_wss_physical_float32": ((RESOLUTION, 3), np.dtype("<f4")),
}
ARRAY_SCHEMAS: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    **{name: (shape, dtype) for name, (_, shape, dtype) in DIRECT_ARRAY_SPECS.items()},
    **ENCODED_ARRAY_SCHEMAS,
    **PREDICTION_ARRAY_SCHEMAS,
}

ARCHIVED_PREDICTION_SPECS: Mapping[str, tuple[Path, tuple[int, ...], np.dtype[Any]]] = {
    "pressure": (
        Path(
            f"{CASE_DIRECTORY}/_tensordict/interior/_tensordict/"
            "point_data/pred_pressure.memmap"
        ),
        (RESOLUTION,),
        np.dtype("<f4"),
    ),
    "wss": (
        Path(
            f"{CASE_DIRECTORY}/_tensordict/interior/_tensordict/"
            "point_data/pred_wss.memmap"
        ),
        (RESOLUTION, 3),
        np.dtype("<f4"),
    ),
}

_MANIFEST_LINE = re.compile(rb"^([0-9a-f]{64})  (\./[^\x00\r\n]+)$")


@dataclass(frozen=True)
class Lane:
    label: str
    document: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    json_sha256: str
    npz_sha256: str


@dataclass(frozen=True)
class LoadResult:
    state: str
    value: Any | None
    reason: str | None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _parse_manifest(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_entries: int | None,
) -> dict[str, str]:
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise ValueError(
            "Historical archive manifest changed: "
            f"expected={expected_sha256} observed={digest}"
        )
    lines = payload.splitlines()
    if expected_entries is not None and len(lines) != expected_entries:
        raise ValueError(
            f"Historical manifest has {len(lines)} entries, expected {expected_entries}"
        )
    entries: dict[str, str] = {}
    previous: str | None = None
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed historical manifest line: {line!r}")
        digest_text = match.group(1).decode("ascii")
        relative_text = match.group(2).decode("utf-8")
        relative = Path(relative_text[2:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe historical manifest path: {relative_text}")
        if relative_text in entries:
            raise ValueError(f"Duplicate historical manifest path: {relative_text}")
        if previous is not None and relative_text <= previous:
            raise ValueError("Historical manifest paths are not sorted")
        entries[relative_text] = digest_text
        previous = relative_text
    return entries


def _manifest(
    manifest_path: Path,
    *,
    expected_sha256: str = EXPECTED_HISTORICAL_MANIFEST_SHA256,
    expected_entries: int | None = 1656,
) -> tuple[bytes, dict[str, str]]:
    payload = _safe_read_bytes(manifest_path)
    return payload, _parse_manifest(
        payload,
        expected_sha256=expected_sha256,
        expected_entries=expected_entries,
    )


def _manifest_bound_payload(
    archive_root: Path,
    relative: Path,
    entries: Mapping[str, str],
) -> tuple[bytes, str]:
    if relative not in {spec[0] for spec in ARCHIVED_PREDICTION_SPECS.values()}:
        raise ValueError(f"Reducer attempted a forbidden archive read: {relative}")
    manifest_name = f"./{relative.as_posix()}"
    expected = entries.get(manifest_name)
    if expected is None:
        raise ValueError(f"Archived prediction is not manifest-bound: {manifest_name}")
    payload = _safe_read_bytes(archive_root / relative)
    observed = _sha256_bytes(payload)
    if observed != expected:
        raise ValueError(
            f"Archived prediction changed: {manifest_name}; "
            f"expected={expected} observed={observed}"
        )
    return payload, observed


def _array_from_payload(
    payload: bytes,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    expected_bytes = math.prod(shape) * dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"{label} has {len(payload)} bytes, expected {expected_bytes}")
    value = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(value)


def _load_archived_predictions(
    archive_root: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_HISTORICAL_MANIFEST_SHA256,
    expected_manifest_entries: int | None = 1656,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    manifest_payload, entries = _manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_entries=expected_manifest_entries,
    )
    predictions, record = _load_archived_predictions_from_manifest(
        archive_root,
        manifest_payload,
        entries,
    )
    return predictions, record, entries


def _load_archived_predictions_from_manifest(
    archive_root: Path,
    manifest_payload: bytes,
    entries: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if archive_root.is_symlink():
        raise ValueError(f"Historical archive root is a symlink: {archive_root}")
    if not archive_root.exists():
        raise FileNotFoundError(f"Historical archive root is missing: {archive_root}")
    if not archive_root.is_dir():
        raise ValueError(f"Historical archive root is not a directory: {archive_root}")
    predictions: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for field, (relative, shape, dtype) in ARCHIVED_PREDICTION_SPECS.items():
        payload, digest = _manifest_bound_payload(archive_root, relative, entries)
        predictions[field] = _array_from_payload(
            payload,
            shape=shape,
            dtype=dtype,
            label=f"archived {field}",
        )
        records[field] = {
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(payload),
        }
    return predictions, {
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "manifest_entry_count": len(entries),
        "opened_payload_count": len(records),
        "opened_payloads": records,
    }


def _attested_payload(path: Path) -> tuple[bytes, str]:
    payload = _safe_read_bytes(path)
    digest = _sha256_bytes(payload)
    sidecar = path.with_name(f"{path.name}.sha256")
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if _safe_read_bytes(sidecar) != expected_sidecar:
        raise ValueError(f"SHA-256 sidecar changed for {path}")
    return payload, digest


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_noncategorical_producer(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "outcome" in str(key).lower():
                raise ValueError(f"Producer contains a categorical key at {path}.{key}")
            _require_noncategorical_producer(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_noncategorical_producer(child, f"{path}[{index}]")


def _validate_lane_document(
    document: Mapping[str, Any],
    *,
    label: str,
    npz_path: Path,
    npz_sha256: str,
) -> None:
    contract = document.get("contract", {})
    provenance = document.get("provenance", {})
    frozen = provenance.get("frozen_inputs", {})
    repo_root = provenance.get("repo_root")
    expected_imports = (
        {
            "physicsnemo": f"{repo_root}/physicsnemo/__init__.py",
            "mesh_transformer_model": (
                f"{repo_root}/physicsnemo/experimental/nn/mesh_attention/model.py"
            ),
            "recipe_infer": (
                f"{repo_root}/examples/cfd/external_aerodynamics/"
                "unified_external_aero_recipe/src/infer.py"
            ),
        }
        if isinstance(repo_root, str) and repo_root.startswith("/")
        else None
    )
    process = provenance.get("process", {})
    if (
        document.get("schema_version") != 1
        or document.get("artifact_kind") != PRODUCER_ARTIFACT_KIND
        or document.get("status") != PRODUCER_STATUS
        or document.get("lane_label") != label
        or contract.get("case_id") != CASE_ID
        or contract.get("reader_index") != READER_INDEX
        or contract.get("resolution") != RESOLUTION
        or contract.get("precision") != PRECISION
        or contract.get("compiled_model") is not False
        or contract.get("archive_is_input_oracle_only") is not True
        or contract.get("archived_predictions_opened") is not False
        or contract.get("archived_truth_opened") is not False
        or contract.get("raw_targets_opened") is not False
        or contract.get("historical_manifest_opened") is not False
        or contract.get("input_freeze_record_opened") is not False
        or contract.get("dataset_reader_constructed") is not False
        or contract.get("model_call") != "model(domain)"
        or contract.get("model_call_count") != 1
        or contract.get("model_call_keyword_arguments") != []
        or contract.get("canonical_source_geometry_supplied") is not False
        or contract.get("encoded_geometry_captured_from_single_forward") is not True
        or contract.get("local_data_fields_present") is not False
        or contract.get("measure_weights_present") is not False
        or contract.get("categorical_decision_present") is not False
        or contract.get("process_isolated_lane") is not True
        or contract.get("checkpoint_load_epoch") != EPOCH
        or provenance.get("producer_sha256") != EXPECTED_PRODUCER_SHA256
        or provenance.get("loaded_epoch") != EPOCH
        or frozen.get("resolved_config") != EXPECTED_RESOLVED_CONFIG_SHA256
        or frozen.get("dataset_config") != EXPECTED_DATASET_CONFIG_SHA256
        or frozen.get("model_checkpoint") != EXPECTED_MODEL_SHA256
        or frozen.get("training_state") != EXPECTED_TRAINING_STATE_SHA256
        or frozen.get("normalization_state") != EXPECTED_NORMALIZATION_SHA256
        or frozen.get("current_infer_source") != EXPECTED_CURRENT_INFER_SHA256
        or frozen.get("current_model_source") != EXPECTED_CURRENT_MODEL_SOURCE_SHA256
        or frozen.get("current_execution_source_tree")
        != EXPECTED_CURRENT_SOURCE_TREE_SHA256
        or frozen.get("checkpoint_load_epoch") != EPOCH
        or frozen.get("parameter_count") != 1_278_268
        or frozen.get("model_seed") != 42
        or frozen.get("import_provenance") != expected_imports
        or provenance.get("historical_manifest_sha256")
        != EXPECTED_HISTORICAL_MANIFEST_SHA256
        or provenance.get("input_freeze_record_sha256") != EXPECTED_INPUT_FREEZE_SHA256
        or not isinstance(process, Mapping)
        or not isinstance(process.get("pid"), int)
        or process.get("pid", 0) <= 0
        or not isinstance(process.get("hostname"), str)
        or not process.get("hostname")
        or not isinstance(process.get("slurm_job_id"), str)
        or not process.get("slurm_job_id")
        or not isinstance(process.get("cuda_visible_devices"), str)
        or not process.get("cuda_visible_devices")
        or document.get("npz", {}).get("filename") != npz_path.name
        or document.get("npz", {}).get("sha256") != npz_sha256
        or document.get("npz", {}).get("array_count") != len(ARRAY_SCHEMAS)
    ):
        raise ValueError(f"Stage-A producer lane {label} contract changed")
    _require_noncategorical_producer(document)


def _load_lane(
    json_path: Path,
    npz_path: Path,
    label: str,
    manifest_entries: Mapping[str, str],
) -> Lane:
    json_payload, json_sha256 = _attested_payload(json_path)
    npz_payload, npz_sha256 = _attested_payload(npz_path)
    document = _strict_json_bytes(json_payload, str(json_path))
    if not isinstance(document, Mapping):
        raise ValueError(f"Producer lane {label} JSON is not an object")
    _validate_lane_document(
        document,
        label=label,
        npz_path=npz_path,
        npz_sha256=npz_sha256,
    )

    with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
        if archive.files != list(ARRAY_SCHEMAS):
            raise ValueError(f"Producer lane {label} NPZ inventory changed")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    array_manifest = document.get("npz", {}).get("array_manifest")
    if not isinstance(array_manifest, Mapping) or set(array_manifest) != set(arrays):
        raise ValueError(f"Producer lane {label} array manifest changed")
    for name, value in arrays.items():
        shape, dtype = ARRAY_SCHEMAS[name]
        record = array_manifest[name]
        if (
            value.shape != shape
            or value.dtype != dtype
            or not isinstance(record, Mapping)
            or record.get("shape") != list(shape)
            or record.get("dtype") != str(dtype)
            or record.get("sha256") != _array_sha256(value)
            or (np.issubdtype(dtype, np.floating) and not np.isfinite(value).all())
        ):
            raise ValueError(f"Producer lane {label} array changed: {name}")

    archive_record = document.get("archive_inputs")
    opened = (
        archive_record.get("opened_payloads", {})
        if isinstance(archive_record, Mapping)
        else {}
    )
    if (
        not isinstance(opened, Mapping)
        or archive_record.get("historical_manifest_sha256")
        != EXPECTED_HISTORICAL_MANIFEST_SHA256
        or archive_record.get("historical_manifest_opened") is not False
        or archive_record.get("input_freeze_record_sha256")
        != EXPECTED_INPUT_FREEZE_SHA256
        or archive_record.get("input_freeze_record_opened") is not False
        or archive_record.get("input_hash_binding") != "embedded_sha256_constants"
        or archive_record.get("opened_payload_count") != len(DIRECT_ARRAY_SPECS)
        or set(opened) != set(DIRECT_ARRAY_SPECS)
    ):
        raise ValueError(f"Producer lane {label} archive-input record changed")
    for name, (relative, _, _) in DIRECT_ARRAY_SPECS.items():
        manifest_name = f"./{relative.as_posix()}"
        expected = manifest_entries.get(manifest_name)
        record = opened[name]
        if (
            expected is None
            or not isinstance(record, Mapping)
            or record.get("relative_path") != relative.as_posix()
            or record.get("sha256") != expected
            or record.get("sha256") != _array_sha256(arrays[name])
            or record.get("size_bytes") != arrays[name].nbytes
        ):
            raise ValueError(
                f"Producer lane {label} direct archive binding changed: {name}"
            )
    return Lane(
        label=label,
        document=document,
        arrays=arrays,
        json_sha256=json_sha256,
        npz_sha256=npz_sha256,
    )


def _arrays_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    return bool(
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and memoryview(left_array).cast("B").tobytes()
        == memoryview(right_array).cast("B").tobytes()
    )


def _byte_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ValueError(
            "Byte comparison schema differs: "
            f"{left_array.shape}/{left_array.dtype} != "
            f"{right_array.shape}/{right_array.dtype}"
        )
    records_dtype = np.dtype((np.void, left_array.dtype.itemsize))
    left_records = left_array.reshape(-1).view(records_dtype)
    right_records = right_array.reshape(-1).view(records_dtype)
    differing = int(np.count_nonzero(left_records != right_records))
    maximum = (
        0.0
        if left_array.size == 0
        else float(
            np.max(
                np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
            )
        )
    )
    return {
        "exact": differing == 0,
        "differing_elements_including_signed_zero": differing,
        "maximum_absolute_difference": maximum,
        "left_sha256": _array_sha256(left_array),
        "right_sha256": _array_sha256(right_array),
        "shape": list(left_array.shape),
        "dtype": str(left_array.dtype),
    }


def _decide_complete(
    lane_a: Lane,
    lane_b: Lane,
    archived: Mapping[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    process_a = lane_a.document["provenance"]["process"]
    process_b = lane_b.document["provenance"]["process"]
    process_comparison = {
        "same_slurm_job": process_a["slurm_job_id"] == process_b["slurm_job_id"],
        "same_hostname": process_a["hostname"] == process_b["hostname"],
        "distinct_pid": process_a["pid"] != process_b["pid"],
        "distinct_cuda_visible_devices": (
            process_a["cuda_visible_devices"] != process_b["cuda_visible_devices"]
        ),
        "lane_gpu_tokens_exact": (
            process_a["cuda_visible_devices"] == "0"
            and process_b["cuda_visible_devices"] == "1"
        ),
        "lane_a": dict(process_a),
        "lane_b": dict(process_b),
    }
    if not all(
        process_comparison[key]
        for key in (
            "same_slurm_job",
            "same_hostname",
            "distinct_pid",
            "distinct_cuda_visible_devices",
            "lane_gpu_tokens_exact",
        )
    ):
        return INVALID, {
            "reason": "producer lanes are not proven process/GPU isolated",
            "process_isolation": process_comparison,
            "lane_a_vs_lane_b": {},
            "archive_prediction_comparisons": {},
        }

    lane_comparisons = {
        name: _byte_difference(lane_a.arrays[name], lane_b.arrays[name])
        for name in ARRAY_SCHEMAS
    }
    if not all(record["exact"] for record in lane_comparisons.values()):
        return INVALID, {
            "reason": "process-isolated lanes are not raw-byte deterministic",
            "process_isolation": process_comparison,
            "lane_a_vs_lane_b": lane_comparisons,
            "archive_prediction_comparisons": {},
        }

    archive_comparisons = {
        "pressure": _byte_difference(
            lane_a.arrays["prediction_pressure_physical_float32"],
            archived["pressure"],
        ),
        "wss": _byte_difference(
            lane_a.arrays["prediction_wss_physical_float32"],
            archived["wss"],
        ),
    }
    exact = all(record["exact"] for record in archive_comparisons.values())
    return (
        EXACT_PASS if exact else VALID_REFUTATION,
        {
            "reason": (
                "both deterministic current-model predictions exactly match "
                "the historical archive"
                if exact
                else (
                    "both deterministic current-model predictions disagree "
                    "with the historical archive"
                )
            ),
            "process_isolation": process_comparison,
            "lane_a_vs_lane_b": lane_comparisons,
            "archive_prediction_comparisons": archive_comparisons,
        },
    )


def _capture_load(function: Any, *args: Any) -> LoadResult:
    try:
        return LoadResult("complete", function(*args), None)
    except FileNotFoundError as error:
        return LoadResult("missing", None, str(error))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        return LoadResult("invalid", None, f"{type(error).__name__}: {error}")


def _validate_distinct_lane_artifacts(paths: Sequence[Path]) -> dict[str, Any]:
    destinations: list[Path] = []
    for path in paths:
        normalized = Path(os.path.abspath(os.path.normpath(path)))
        destinations.extend(
            (normalized, normalized.with_name(f"{normalized.name}.sha256"))
        )
    if len(set(destinations)) != len(destinations):
        raise ValueError("Lane artifacts and sidecars are not path-distinct")
    resolved = [path.resolve(strict=False) for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Lane artifacts and sidecars resolve to aliased paths")
    identities: dict[tuple[int, int], Path] = {}
    existing = 0
    for path in destinations:
        if not path.exists() and not path.is_symlink():
            continue
        info = path.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            raise ValueError(
                f"Lane artifacts share an inode: {identities[identity]} and {path}"
            )
        identities[identity] = path
        existing += 1
    return {
        "artifact_and_sidecar_path_count": len(destinations),
        "existing_distinct_inode_count": existing,
    }


def _output_destinations(output: Path) -> tuple[Path, Path]:
    normalized = Path(os.path.abspath(os.path.normpath(output)))
    sidecar = normalized.with_name(f"{normalized.name}.sha256")
    if normalized.resolve(strict=False) == sidecar.resolve(strict=False):
        raise ValueError("Reducer output aliases its sidecar")
    return normalized, sidecar


def _validate_output_target(output: Path) -> tuple[Path, Path]:
    destinations = _output_destinations(output)
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    return destinations


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _publish_json(output: Path, payload: bytes) -> str:
    output, sidecar = _validate_output_target(output)
    digest = _sha256_bytes(payload)
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        temporaries[output] = _write_fsynced_temporary(output, payload)
        temporaries[sidecar] = _write_fsynced_temporary(sidecar, sidecar_payload)
        for destination, temporary in temporaries.items():
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        _fsync_directory(output.parent)
        if (
            _safe_read_bytes(output) != payload
            or _safe_read_bytes(sidecar) != sidecar_payload
        ):
            raise OSError("Published adjudication changed")
    except BaseException:
        for destination, temporary in reversed(published):
            _unlink_if_same_inode(destination, temporary)
        if temporaries:
            _fsync_directory(output.parent)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        if temporaries:
            _fsync_directory(output.parent)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-a-json", type=Path, required=True)
    parser.add_argument("--lane-a-npz", type=Path, required=True)
    parser.add_argument("--lane-b-json", type=Path, required=True)
    parser.add_argument("--lane-b-npz", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument(
        "--historical-predictions-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(os.path.abspath(args.output_json))
    _validate_output_target(output)

    separation_result = _capture_load(
        _validate_distinct_lane_artifacts,
        (
            args.lane_a_json,
            args.lane_a_npz,
            args.lane_b_json,
            args.lane_b_npz,
        ),
    )
    manifest_result = _capture_load(
        _manifest,
        args.historical_predictions_manifest,
    )
    archive_result: LoadResult
    lane_a_result: LoadResult
    lane_b_result: LoadResult
    manifest_entries: Mapping[str, str] = {}
    archive_record: Mapping[str, Any] = {}
    if manifest_result.state == "complete":
        manifest_payload, manifest_entries = manifest_result.value
        archive_result = _capture_load(
            _load_archived_predictions_from_manifest,
            args.historical_predictions,
            manifest_payload,
            manifest_entries,
        )
        lane_a_result = _capture_load(
            _load_lane,
            args.lane_a_json,
            args.lane_a_npz,
            "A",
            manifest_entries,
        )
        lane_b_result = _capture_load(
            _load_lane,
            args.lane_b_json,
            args.lane_b_npz,
            "B",
            manifest_entries,
        )
    else:
        archive_result = manifest_result
        lane_a_result = LoadResult(
            manifest_result.state,
            None,
            "archive manifest unavailable before lane validation",
        )
        lane_b_result = lane_a_result

    results = {
        "lane_artifact_separation": separation_result,
        "archive": archive_result,
        "lane_a": lane_a_result,
        "lane_b": lane_b_result,
    }
    invalid_reasons = [
        f"{name}: {result.reason}"
        for name, result in results.items()
        if result.state == "invalid"
    ]
    missing_reasons = [
        f"{name}: {result.reason}"
        for name, result in results.items()
        if result.state == "missing"
    ]
    comparisons: dict[str, Any] = {
        "reason": "",
        "lane_a_vs_lane_b": {},
        "archive_prediction_comparisons": {},
    }
    if invalid_reasons:
        outcome = INVALID
        comparisons["reason"] = "structural or provenance validation failed"
    elif missing_reasons:
        outcome = INCOMPLETE
        comparisons["reason"] = "one or more required artifacts are absent"
    else:
        archived, archive_record = archive_result.value
        outcome, comparisons = _decide_complete(
            lane_a_result.value,
            lane_b_result.value,
            archived,
        )

    reducer_path = Path(__file__).resolve(strict=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "truth_table": {
            EXACT_PASS: (
                "both complete, valid lanes are byte-identical and both "
                "physical prediction fields equal the archive"
            ),
            VALID_REFUTATION: (
                "both complete, valid lanes are byte-identical and at least "
                "one physical prediction field differs from the archive"
            ),
            INVALID: (
                "a present artifact fails structure/provenance validation or "
                "the two complete lanes are not byte-identical"
            ),
            INCOMPLETE: "at least one required artifact is absent and none is invalid",
        },
        "load_validation": {
            name: {"state": result.state, "reason": result.reason}
            for name, result in results.items()
        },
        "lane_artifact_separation": (
            separation_result.value if separation_result.state == "complete" else None
        ),
        "invalid_reasons": invalid_reasons,
        "missing_reasons": missing_reasons,
        "comparisons": comparisons,
        "archive_predictions": archive_record,
        "provenance": {
            "reducer_path": str(reducer_path),
            "reducer_sha256": _sha256_file(reducer_path),
            "expected_producer_sha256": EXPECTED_PRODUCER_SHA256,
            "historical_manifest_sha256": EXPECTED_HISTORICAL_MANIFEST_SHA256,
            "lane_a_json_sha256": (
                lane_a_result.value.json_sha256
                if lane_a_result.state == "complete"
                else None
            ),
            "lane_a_npz_sha256": (
                lane_a_result.value.npz_sha256
                if lane_a_result.state == "complete"
                else None
            ),
            "lane_b_json_sha256": (
                lane_b_result.value.json_sha256
                if lane_b_result.state == "complete"
                else None
            ),
            "lane_b_npz_sha256": (
                lane_b_result.value.npz_sha256
                if lane_b_result.state == "complete"
                else None
            ),
        },
    }
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    digest = _publish_json(output, payload)
    print(f"{STATUS} outcome={outcome} json_sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
