# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independently adjudicate the historical K=10,000 legacy replay.

The producer is deliberately observational: it emits two process-isolated
JSON/NPZ replicas and never reads the historical predictions or metrics.  This
reducer is the sole publisher of a categorical outcome.  It loads every
deciding artifact from sidecar-bound bytes, verifies the immutable historical
archive against its frozen file manifest, and recomputes every comparison and
metric.  Its corrected surface weighting and training/physical consistency
checks are independently reconstructed from that archive plus the hash-bound
normalization state.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_replay_adjudication"
VALID_STATUS = "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
INVALID_STATUS = "INVALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
INCOMPLETE_STATUS = "INCOMPLETE_HISTORICAL_K10000_REPLAY_ADJUDICATION"

EXACT_OUTCOME = "EXACT_HISTORICAL_REPLAY_PASS"
VALID_REFUTATION = "VALID_EXACT_REPLAY_REFUTATION"
INVALID_REPLAY = "INVALID_REPLAY"
INCOMPLETE_REPLAY = "INCOMPLETE_REPLAY"

RESOLUTION = 10_000
CASE_COUNT = 36
ARRAYS_PER_CASE = 20
EXPECTED_PRODUCER_SHA256 = (
    "bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5"
)
EXPECTED_HISTORICAL_MANIFEST_SHA256 = (
    "545b1f6e906002231415b84277db00eec04f3666233b8da637514e9077a585eb"
)
EXPECTED_HISTORICAL_METRICS_SHA256 = (
    "423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a"
)
EXPECTED_TARGET_INPUT_MANIFEST_SHA256 = (
    "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49"
)
EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256 = (
    "fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10"
)
EXPECTED_GEOMETRY_MANIFEST_SHA256 = (
    "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
)
EXPECTED_HELPER_SHA256 = (
    "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
)
EXPECTED_CURRENT_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_CURRENT_MODEL_SOURCE_SHA256 = (
    "9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892"
)
EXPECTED_MODEL_CHECKPOINT_SHA256 = (
    "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
)
EXPECTED_NORMALIZATION_STATE_SHA256 = (
    "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
)
ARCHIVED_PRESSURE_MEAN = 0.16716310713026258
PRESSURE_CASE_ABS_TOLERANCE = 1.0e-3
PRESSURE_MEAN_ABS_TOLERANCE = 5.0e-6
PIPELINE_NORMAL_ABS_TOLERANCE = 2.0e-6
# All 36 frozen cases have p_inf=0.  Exhaustive adjacent-float32-preimage
# analysis over every archived prediction and truth found a worst 1.100e-6
# distance from this float64 inverse, so 2e-6 leaves >1.8x empirical margin.
TRAINING_PHYSICAL_ABS_TOLERANCE = 2.0e-6
TRAINING_PHYSICAL_REL_TOLERANCE = 0.0
NORMALIZATION_EPSILON = 1.0e-8
NONINFERIORITY_RATIO = 1.02

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
ARCHIVED_GLOBAL_FIELDS = (
    ("U_inf", (3,)),
    ("p_inf", ()),
    ("rho_inf", ()),
    ("nu", ()),
    ("L_ref", ()),
    ("U_inf_dir", (3,)),
    ("reference_length", ()),
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

# ``None`` marks the one per-case dynamic dimension.
ARRAY_SCHEMAS: Mapping[str, tuple[tuple[int | None, ...], np.dtype[Any]]] = {
    "selected_cell_ids_int64": ((RESOLUTION,), np.dtype("<i8")),
    "compacted_cells_int64": ((RESOLUTION, 3), np.dtype("<i8")),
    "raw_centroids_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "native_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "native_areas_float64": ((RESOLUTION,), np.dtype("<f8")),
    "raw_target_pressure_float32": ((RESOLUTION,), np.dtype("<f4")),
    "raw_target_wss_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_boundary_points_float32": ((None, 3), np.dtype("<f4")),
    "pipeline_queries_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_globals_float32": ((len(GLOBAL_FIELD_ORDER),), np.dtype("<f4")),
    "prediction_pressure_training_float32": ((RESOLUTION,), np.dtype("<f4")),
    "prediction_wss_training_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "truth_pressure_training_float32": ((RESOLUTION,), np.dtype("<f4")),
    "truth_wss_training_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "prediction_pressure_physical_float32": ((RESOLUTION,), np.dtype("<f4")),
    "prediction_wss_physical_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "truth_pressure_physical_float32": ((RESOLUTION,), np.dtype("<f4")),
    "truth_wss_physical_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_center_float32": ((3,), np.dtype("<f4")),
}

_MANIFEST_LINE = re.compile(rb"^([0-9a-f]{64})  (\./[^\x00\r\n]+)$")
_HEX_DIGITS = frozenset("0123456789abcdef")


class ArtifactUnavailable(ValueError):
    """A required artifact cannot be read completely."""


class ArtifactInvalid(ValueError):
    """A readable artifact violates the frozen experiment contract."""


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactInvalid(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    return bool(
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes() == right_array.tobytes()
    )


def _byte_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ArtifactInvalid(
            "Byte comparison schema differs: "
            f"{left_array.shape}/{left_array.dtype} != "
            f"{right_array.shape}/{right_array.dtype}"
        )
    records_dtype = np.dtype((np.void, left_array.dtype.itemsize))
    left_records = left_array.reshape(-1).view(records_dtype)
    right_records = right_array.reshape(-1).view(records_dtype)
    count = int(np.count_nonzero(left_records != right_records))
    maximum = (
        0.0
        if left_array.size == 0
        else float(
            np.max(
                np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
            )
        )
    )
    if not math.isfinite(maximum):
        raise ArtifactInvalid("Byte comparison contains a non-finite difference")
    return {
        "exact": count == 0,
        "differing_elements_including_signed_zero": count,
        "maximum_absolute_difference": maximum,
        "left_sha256": _array_sha256(left_array),
        "right_sha256": _array_sha256(right_array),
        "shape": list(left_array.shape),
        "dtype": str(left_array.dtype),
    }


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.shape != right64.shape:
        raise ArtifactInvalid(f"Shape mismatch: {left64.shape} != {right64.shape}")
    result = 0.0 if left64.size == 0 else float(np.max(np.abs(left64 - right64)))
    if not math.isfinite(result):
        raise ArtifactInvalid("Difference is non-finite")
    return result


def _read_regular_file_bytes(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    """Read a stable regular file without following any path-component symlink."""

    lexical = Path(os.path.abspath(path))
    parts = lexical.parts
    if not parts or parts[0] != os.sep:
        raise ArtifactUnavailable(f"{label} does not resolve to an absolute path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        leaf_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = leaf_descriptor
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactUnavailable(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (
            identity != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ArtifactUnavailable(f"{label} changed while it was being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ArtifactUnavailable(f"{label} size changed while it was being read")
        return payload, identity
    except ArtifactUnavailable:
        raise
    except OSError as error:
        raise ArtifactUnavailable(
            f"{label} is missing, unreadable, or traverses a symlink"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_verified_artifact_bytes(path: Path, label: str) -> tuple[bytes, str]:
    """Load one payload and its canonical sidecar exactly once each."""

    lexical = Path(os.path.abspath(path))
    payload, _ = _read_regular_file_bytes(lexical, label)
    sidecar, _ = _read_regular_file_bytes(
        lexical.with_name(f"{lexical.name}.sha256"),
        f"{label} sidecar",
    )
    observed = _sha256_bytes(payload)
    expected = f"{observed}  {lexical.name}\n".encode("ascii")
    if sidecar != expected:
        raise ArtifactUnavailable(f"{label} sidecar is missing, stale, or malformed")
    return payload, observed


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactUnavailable(f"Duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ArtifactUnavailable(f"Non-finite JSON token {value!r} in {label}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ArtifactUnavailable:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactUnavailable(f"Unreadable JSON artifact {label}") from error


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} key set differs")


def _contains_value(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(nested, expected) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_value(nested, expected) for nested in value)
    return False


def _reject_producer_conclusions(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(
                token in lowered for token in ("outcome", "decision", "comparison")
            ) or ("metric" in lowered and key != "producer_reads_archive_or_metrics"):
                raise ArtifactInvalid(
                    f"Producer contains forbidden conclusion key {path}"
                )
            _reject_producer_conclusions(nested, path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_producer_conclusions(nested, f"{prefix}[{index}]")


def _case_prefix(ordinal: int, case_id: str) -> str:
    return f"case_{ordinal:02d}_{case_id}"


def _expected_array_names() -> tuple[str, ...]:
    return tuple(
        f"{_case_prefix(ordinal, case_id)}__{suffix}"
        for ordinal, case_id, _, _, _ in CASE_SPECS
        for suffix in ARRAY_SCHEMAS
    )


def _load_npz_bytes(
    payload: bytes,
    label: str,
    expected_names: Sequence[str],
) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [record.filename for record in archive.infolist()]
            expected_members = {f"{name}.npy" for name in expected_names}
            if len(members) != len(set(members)):
                raise ArtifactInvalid(f"{label} contains duplicate ZIP members")
            if set(members) != expected_members:
                raise ArtifactInvalid(f"{label} array key set differs")
            if any(
                "/" in member or "\\" in member or not member.endswith(".npy")
                for member in members
            ):
                raise ArtifactInvalid(f"{label} contains an unsafe member name")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ArtifactUnavailable(f"{label} has corrupt ZIP member {corrupt}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(expected_names):
                raise ArtifactInvalid(f"{label} NumPy array key set differs")
            return {
                name: np.ascontiguousarray(np.array(archive[name], copy=True))
                for name in expected_names
            }
    except (ArtifactInvalid, ArtifactUnavailable):
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactUnavailable(f"Unreadable NPZ artifact {label}") from error


def _shape_matches(shape: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    return len(shape) == len(expected) and all(
        expected_value is None or observed == expected_value
        for observed, expected_value in zip(shape, expected, strict=True)
    )


def _validate_case_array_schema(
    arrays: Mapping[str, np.ndarray],
    *,
    spec: tuple[int, str, int, int, int],
    n_compacted_points: int,
) -> None:
    ordinal, case_id, _, _, start = spec
    prefix = _case_prefix(ordinal, case_id)
    for suffix, (shape, dtype) in ARRAY_SCHEMAS.items():
        value = arrays[f"{prefix}__{suffix}"]
        _require(
            _shape_matches(value.shape, shape) and value.dtype == dtype,
            f"Array schema differs for {prefix}__{suffix}",
        )
        if np.issubdtype(value.dtype, np.floating):
            _require(
                bool(np.isfinite(value).all()),
                f"Array contains non-finite values: {prefix}__{suffix}",
            )
    points = arrays[f"{prefix}__pipeline_boundary_points_float32"]
    _require(
        type(n_compacted_points) is int
        and 3 <= n_compacted_points <= 3 * RESOLUTION
        and points.shape == (n_compacted_points, 3),
        f"Compacted-point count differs for {case_id}",
    )
    cells = arrays[f"{prefix}__compacted_cells_int64"]
    _require(
        bool(np.all(cells >= 0))
        and int(cells.max()) == n_compacted_points - 1
        and np.array_equal(np.unique(cells), np.arange(n_compacted_points)),
        f"Compacted connectivity is not dense for {case_id}",
    )
    expected_ids = np.arange(start, start + RESOLUTION, dtype="<i8")
    _require(
        _array_exact(
            arrays[f"{prefix}__selected_cell_ids_int64"],
            expected_ids,
        ),
        f"Selected cell IDs differ for {case_id}",
    )


def _validate_array_manifest(
    manifest: Any,
    arrays: Mapping[str, np.ndarray],
) -> None:
    manifest = _mapping(manifest, "Producer NPZ array manifest")
    _require(set(manifest) == set(arrays), "Producer array manifest key set differs")
    for name, value in arrays.items():
        record = _mapping(manifest[name], f"Producer array manifest {name}")
        _exact_keys(record, {"shape", "dtype", "sha256"}, f"Array manifest {name}")
        _require(
            record.get("shape") == list(value.shape)
            and record.get("dtype") == str(value.dtype)
            and record.get("sha256") == _array_sha256(value),
            f"Producer array manifest differs for {name}",
        )


def _load_target_manifest(path: Path) -> tuple[dict[str, Mapping[str, Any]], str]:
    payload, digest = _load_verified_artifact_bytes(path, "Selected-target manifest")
    _require(
        digest == EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
        "Selected-target manifest SHA-256 differs",
    )
    document = _mapping(
        _strict_json_bytes(payload, str(path)),
        "Selected-target manifest",
    )
    cases = document.get("cases")
    _require(
        document.get("schema_version") == 1
        and document.get("artifact_kind")
        == "drivaerml_historical_k10000_selected_target_input_manifest"
        and document.get("status")
        == "PASSED_HISTORICAL_K10000_SELECTED_TARGET_INPUT_FREEZE"
        and document.get("case_count") == CASE_COUNT
        and document.get("resolution") == RESOLUTION
        and isinstance(cases, list)
        and len(cases) == CASE_COUNT,
        "Selected-target manifest contract differs",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for spec, case in zip(CASE_SPECS, cases, strict=True):
        ordinal, case_id, reader_index, n_cells, start = spec
        case = _mapping(case, f"Selected-target case {case_id}")
        _require(
            (
                case.get("cohort_ordinal"),
                case.get("case_id"),
                case.get("reader_index"),
                case.get("n_master_cells"),
                case.get("historical_start"),
            )
            == (ordinal, case_id, reader_index, n_cells, start),
            f"Selected-target case identity differs for {case_id}",
        )
        targets = _mapping(
            case.get("selected_targets"),
            f"Selected-target records {case_id}",
        )
        _require(
            set(targets) == {"pressure", "wss"}, f"Target fields differ for {case_id}"
        )
        for field, shape in (
            ("pressure", [RESOLUTION]),
            ("wss", [RESOLUTION, 3]),
        ):
            record = _mapping(targets[field], f"Selected target {case_id} {field}")
            _require(
                record.get("selected_shape") == shape
                and record.get("selected_dtype") == "float32_little_endian"
                and _is_sha256(record.get("selected_sha256")),
                f"Selected-target record differs for {case_id} {field}",
            )
        result[case_id] = targets
    return result, digest


def _load_normalization_state(
    path: Path,
) -> tuple[dict[str, np.ndarray | float], str]:
    """Load the one frozen WSS normalizer from hash-bound bytes."""

    payload, _ = _read_regular_file_bytes(path, "Normalization state")
    digest = _sha256_bytes(payload)
    _require(
        digest == EXPECTED_NORMALIZATION_STATE_SHA256,
        "Normalization-state SHA-256 differs",
    )
    try:
        state = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ArtifactInvalid(
            "Normalization state cannot be decoded with weights_only=True"
        ) from error
    state = _mapping(state, "Normalization state")
    _exact_keys(state, {"wss"}, "Normalization state")
    wss = _mapping(state["wss"], "Normalization state wss")
    _exact_keys(wss, {"type", "mean", "std"}, "Normalization state wss")
    mean = wss["mean"]
    std = wss["std"]
    _require(
        wss["type"] == "vector"
        and isinstance(mean, torch.Tensor)
        and mean.device.type == "cpu"
        and mean.dtype == torch.float32
        and tuple(mean.shape) == (3,)
        and not mean.requires_grad
        and isinstance(std, torch.Tensor)
        and std.device.type == "cpu"
        and std.dtype == torch.float32
        and tuple(std.shape) == ()
        and not std.requires_grad,
        "Normalization-state schema differs",
    )
    mean_array = np.ascontiguousarray(mean.detach().numpy().astype("<f8"))
    std_value = float(std.item())
    _require(
        bool(np.isfinite(mean_array).all())
        and bool(np.array_equal(mean_array, np.zeros(3, dtype="<f8")))
        and math.isfinite(std_value)
        and std_value == float(np.float32(0.00313)),
        "Normalization-state values differ",
    )
    return {
        "wss_mean": mean_array,
        "wss_std": std_value,
    }, digest


def _validate_producer_document(
    document: Mapping[str, Any],
    *,
    expected_label: str,
    npz_path: Path,
    npz_sha256: str,
    arrays: Mapping[str, np.ndarray],
    target_records: Mapping[str, Mapping[str, Any]],
) -> None:
    _reject_producer_conclusions(document)
    _exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "generated_at_utc",
            "replay_label",
            "contract",
            "summary",
            "cases",
            "npz",
            "provenance",
        },
        f"Producer {expected_label}",
    )
    _require(
        document.get("schema_version") == 1
        and document.get("artifact_kind") == "phase1_historical_k10000_replay_producer"
        and document.get("status") == "COMPLETED_HISTORICAL_K10000_REPLAY_PRODUCER"
        and document.get("replay_label") == expected_label,
        f"Producer {expected_label} identity differs",
    )
    generated = document.get("generated_at_utc")
    _require(type(generated) is str, f"Producer {expected_label} timestamp differs")
    try:
        parsed_generated = datetime.fromisoformat(generated)
    except ValueError as error:
        raise ArtifactInvalid(
            f"Producer {expected_label} timestamp is invalid"
        ) from error
    _require(
        parsed_generated.tzinfo is not None,
        f"Producer {expected_label} timestamp is not timezone-aware",
    )

    contract = _mapping(document.get("contract"), f"Producer {expected_label} contract")
    _exact_keys(
        contract,
        {
            "legacy_call",
            "canonical_source_geometry_present",
            "producer_reads_archive_or_metrics",
            "candidate_or_canonical_arm_present",
            "resolution",
            "precision",
            "case_count",
            "global_field_order",
            "measure_weights_required_absent",
        },
        f"Producer {expected_label} contract",
    )
    _require(
        contract
        == {
            "legacy_call": "model(domain)",
            "canonical_source_geometry_present": False,
            "producer_reads_archive_or_metrics": False,
            "candidate_or_canonical_arm_present": False,
            "resolution": RESOLUTION,
            "precision": "bfloat16",
            "case_count": CASE_COUNT,
            "global_field_order": list(GLOBAL_FIELD_ORDER),
            "measure_weights_required_absent": True,
        },
        f"Producer {expected_label} execution contract differs",
    )
    summary = _mapping(document.get("summary"), f"Producer {expected_label} summary")
    _exact_keys(
        summary,
        {"case_count", "array_count", "measure_weights_absent_case_count"},
        f"Producer {expected_label} summary",
    )
    _require(
        summary
        == {
            "case_count": CASE_COUNT,
            "array_count": CASE_COUNT * ARRAYS_PER_CASE,
            "measure_weights_absent_case_count": CASE_COUNT,
        },
        f"Producer {expected_label} summary differs",
    )

    cases = document.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == CASE_COUNT,
        f"Producer {expected_label} case count differs",
    )
    for spec, case in zip(CASE_SPECS, cases, strict=True):
        ordinal, case_id, reader_index, n_cells, start = spec
        case = _mapping(case, f"Producer {expected_label} case {case_id}")
        _exact_keys(
            case,
            {
                "cohort_ordinal",
                "case_id",
                "reader_index",
                "n_master_cells",
                "historical_start",
                "resolution",
                "n_compacted_points",
                "measure_weights_absent",
                "target_input_verification",
                "array_sha256",
            },
            f"Producer {expected_label} case {case_id}",
        )
        _require(
            (
                case.get("cohort_ordinal"),
                case.get("case_id"),
                case.get("reader_index"),
                case.get("n_master_cells"),
                case.get("historical_start"),
                case.get("resolution"),
                case.get("measure_weights_absent"),
            )
            == (ordinal, case_id, reader_index, n_cells, start, RESOLUTION, True),
            f"Producer {expected_label} case identity differs for {case_id}",
        )
        verification = _mapping(
            case.get("target_input_verification"),
            f"Producer {expected_label} target verification {case_id}",
        )
        _exact_keys(
            verification,
            {"pressure_selected_sha256", "wss_selected_sha256"},
            f"Producer {expected_label} target verification {case_id}",
        )
        expected_target_hashes = {
            "pressure_selected_sha256": target_records[case_id]["pressure"][
                "selected_sha256"
            ],
            "wss_selected_sha256": target_records[case_id]["wss"]["selected_sha256"],
        }
        _require(
            dict(verification) == expected_target_hashes,
            f"Producer {expected_label} target verification differs for {case_id}",
        )
        _validate_case_array_schema(
            arrays,
            spec=spec,
            n_compacted_points=case.get("n_compacted_points"),
        )
        prefix = _case_prefix(ordinal, case_id)
        expected_case_hashes = {
            suffix: _array_sha256(arrays[f"{prefix}__{suffix}"])
            for suffix in ARRAY_SCHEMAS
        }
        hashes = _mapping(
            case.get("array_sha256"),
            f"Producer {expected_label} case hashes {case_id}",
        )
        _require(
            dict(hashes) == expected_case_hashes,
            f"Producer {expected_label} case hashes differ for {case_id}",
        )
        _require(
            expected_case_hashes["raw_target_pressure_float32"]
            == expected_target_hashes["pressure_selected_sha256"]
            and expected_case_hashes["raw_target_wss_float32"]
            == expected_target_hashes["wss_selected_sha256"],
            f"Producer {expected_label} raw target bytes differ for {case_id}",
        )

    npz = _mapping(document.get("npz"), f"Producer {expected_label} NPZ")
    _exact_keys(
        npz,
        {"filename", "sha256", "array_count", "array_manifest"},
        f"Producer {expected_label} NPZ",
    )
    _require(
        npz.get("filename") == npz_path.name
        and npz.get("sha256") == npz_sha256
        and npz.get("array_count") == CASE_COUNT * ARRAYS_PER_CASE,
        f"Producer {expected_label} NPZ identity differs",
    )
    _validate_array_manifest(npz.get("array_manifest"), arrays)

    provenance = _mapping(
        document.get("provenance"),
        f"Producer {expected_label} provenance",
    )
    _exact_keys(
        provenance,
        {
            "command",
            "producer_path",
            "producer_sha256",
            "helper_path",
            "helper_sha256",
            "repo_root",
            "dataset_root",
            "static_inputs",
            "geometry_verification",
            "target_input_verification",
            "historical_input_freeze_sha256",
            "import_provenance",
            "python",
            "numpy",
            "torch",
            "cuda_runtime",
            "device_name",
        },
        f"Producer {expected_label} provenance",
    )
    _require(
        provenance.get("producer_sha256") == EXPECTED_PRODUCER_SHA256
        and provenance.get("helper_sha256") == EXPECTED_HELPER_SHA256
        and provenance.get("historical_input_freeze_sha256")
        == EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256,
        f"Producer {expected_label} source provenance differs",
    )
    for expected_hash, label in (
        (EXPECTED_CURRENT_SOURCE_TREE_SHA256, "current execution source"),
        (EXPECTED_CURRENT_MODEL_SOURCE_SHA256, "current model source"),
        (EXPECTED_MODEL_CHECKPOINT_SHA256, "model checkpoint"),
        (EXPECTED_GEOMETRY_MANIFEST_SHA256, "geometry manifest"),
        (EXPECTED_TARGET_INPUT_MANIFEST_SHA256, "selected-target manifest"),
    ):
        _require(
            _contains_value(provenance, expected_hash),
            f"Producer {expected_label} does not bind the {label}",
        )


def _load_producer(
    json_path: Path,
    npz_path: Path,
    expected_label: str,
    target_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_payload, json_sha256 = _load_verified_artifact_bytes(
        json_path,
        f"Producer {expected_label} JSON",
    )
    npz_payload, npz_sha256 = _load_verified_artifact_bytes(
        npz_path,
        f"Producer {expected_label} NPZ",
    )
    document = _mapping(
        _strict_json_bytes(json_payload, str(json_path)),
        f"Producer {expected_label}",
    )
    arrays = _load_npz_bytes(
        npz_payload,
        f"Producer {expected_label} NPZ",
        _expected_array_names(),
    )
    _validate_producer_document(
        document,
        expected_label=expected_label,
        npz_path=npz_path,
        npz_sha256=npz_sha256,
        arrays=arrays,
        target_records=target_records,
    )
    return {
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
    }, arrays


def _parse_manifest_payload(payload: bytes) -> list[tuple[str, str]]:
    _require(
        _sha256_bytes(payload) == EXPECTED_HISTORICAL_MANIFEST_SHA256,
        "Frozen historical prediction manifest changed",
    )
    entries: list[tuple[str, str]] = []
    for line in payload.splitlines():
        match = _MANIFEST_LINE.fullmatch(line)
        _require(match is not None, f"Malformed historical manifest line: {line!r}")
        relative = match.group(2).decode("utf-8")
        path = Path(relative[2:])
        _require(
            not path.is_absolute() and ".." not in path.parts,
            f"Unsafe historical manifest path: {relative}",
        )
        entries.append((relative, match.group(1).decode("ascii")))
    names = [name for name, _ in entries]
    _require(
        len(entries) == 1656 and len(set(names)) == 1656 and names == sorted(names),
        "Frozen historical prediction manifest inventory changed",
    )
    return entries


def _directory_inventory(root: Path, label: str) -> list[str]:
    lexical = Path(os.path.abspath(root))
    try:
        metadata = lexical.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactUnavailable(f"{label} directory is unavailable") from error
    if lexical.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactUnavailable(f"{label} is not a regular directory")
    result: list[str] = []
    try:
        for path in lexical.rglob("*"):
            if path.is_symlink():
                raise ArtifactInvalid(f"{label} contains a symlink: {path}")
            if path.is_file():
                result.append(f"./{path.relative_to(lexical).as_posix()}")
    except OSError as error:
        raise ArtifactUnavailable(f"{label} inventory is unreadable") from error
    return sorted(result)


def _load_verified_tree(
    root: Path,
    entries: Sequence[tuple[str, str]],
    *,
    label: str,
    retain_payloads: bool,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    expected_names = [name for name, _ in entries]
    observed_names = _directory_inventory(root, label)
    missing = sorted(set(expected_names) - set(observed_names))
    extras = sorted(set(observed_names) - set(expected_names))
    if missing:
        raise ArtifactUnavailable(
            f"{label} is incomplete; first missing files: {missing[:5]}"
        )
    _require(not extras, f"{label} has unexpected files: {extras[:5]}")
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    for relative, expected_hash in entries:
        payload, _ = _read_regular_file_bytes(
            Path(root) / relative[2:],
            f"{label} {relative}",
        )
        _require(
            _sha256_bytes(payload) == expected_hash,
            f"{label} content differs for {relative}",
        )
        total_bytes += len(payload)
        if retain_payloads:
            payloads[relative] = payload
    _require(total_bytes == 44_790_588, f"{label} byte count differs")
    return payloads, {
        "exact": True,
        "files_compared": len(entries),
        "bytes_compared": total_bytes,
        "mismatch_count": 0,
    }


def _tree_payload(tree: Mapping[str, bytes], relative: str) -> bytes:
    try:
        return tree[relative]
    except KeyError as error:
        raise ArtifactInvalid(
            f"Historical archive field is absent: {relative}"
        ) from error


def _archived_array(
    tree: Mapping[str, bytes],
    *,
    tensor_root: str,
    metadata: Mapping[str, Any],
    field: str,
    relative: str,
    shape: tuple[int, ...],
    dtype_name: str,
) -> np.ndarray:
    entry = metadata.get(field)
    _require(
        isinstance(entry, Mapping)
        and tuple(entry.get("shape", ())) == shape
        and entry.get("dtype") == dtype_name
        and entry.get("device") == "cpu",
        f"Archived metadata changed for {field}",
    )
    dtype = {
        "torch.float32": np.dtype("<f4"),
        "torch.int64": np.dtype("<i8"),
    }[dtype_name]
    payload = _tree_payload(tree, f"{tensor_root}/{relative}")
    _require(
        len(payload) == math.prod(shape) * dtype.itemsize,
        f"Archived payload size changed for {field}",
    )
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


def _archived_metadata(
    tree: Mapping[str, bytes],
    relative: str,
) -> Mapping[str, Any]:
    return _mapping(
        _strict_json_bytes(_tree_payload(tree, relative), relative),
        f"Archived metadata {relative}",
    )


def _load_archived_case(
    tree: Mapping[str, bytes],
    *,
    reader_index: int,
    case_id: str,
) -> dict[str, np.ndarray]:
    case = f"./{reader_index:05d}_{case_id}_domain_{case_id}.pdmsh"
    interior = f"{case}/_tensordict/interior/_tensordict"
    boundary = f"{case}/_tensordict/boundaries/vehicle/_tensordict"
    interior_meta = _archived_metadata(tree, f"{interior}/meta.json")
    point_meta = _archived_metadata(tree, f"{interior}/point_data/meta.json")
    boundary_meta = _archived_metadata(tree, f"{boundary}/meta.json")
    normal_meta = _archived_metadata(tree, f"{boundary}/cell_data/meta.json")
    global_meta = _archived_metadata(tree, f"{boundary}/global_data/meta.json")
    points_entry = boundary_meta.get("points")
    _require(
        isinstance(points_entry, Mapping)
        and isinstance(points_entry.get("shape"), list)
        and len(points_entry["shape"]) == 2
        and points_entry["shape"][1] == 3
        and type(points_entry["shape"][0]) is int
        and 3 <= points_entry["shape"][0] <= 3 * RESOLUTION,
        "Archived boundary-point metadata changed",
    )
    n_points = points_entry["shape"][0]
    globals_flat: list[np.ndarray] = []
    for field, shape in ARCHIVED_GLOBAL_FIELDS:
        globals_flat.append(
            _archived_array(
                tree,
                tensor_root=f"{boundary}/global_data",
                metadata=global_meta,
                field=field,
                relative=f"{field}.memmap",
                shape=shape,
                dtype_name="torch.float32",
            ).reshape(-1)
        )
    globals_array = np.ascontiguousarray(np.concatenate(globals_flat).astype("<f4"))
    _require(
        globals_array.shape == (len(GLOBAL_FIELD_ORDER),),
        "Archived global field order changed",
    )
    return {
        "boundary_points": _archived_array(
            tree,
            tensor_root=boundary,
            metadata=boundary_meta,
            field="points",
            relative="points.memmap",
            shape=(n_points, 3),
            dtype_name="torch.float32",
        ),
        "cells": _archived_array(
            tree,
            tensor_root=boundary,
            metadata=boundary_meta,
            field="cells",
            relative="cells.memmap",
            shape=(RESOLUTION, 3),
            dtype_name="torch.int64",
        ),
        "normals": _archived_array(
            tree,
            tensor_root=f"{boundary}/cell_data",
            metadata=normal_meta,
            field="normals",
            relative="normals.memmap",
            shape=(RESOLUTION, 3),
            dtype_name="torch.float32",
        ),
        "query_points": _archived_array(
            tree,
            tensor_root=interior,
            metadata=interior_meta,
            field="points",
            relative="points.memmap",
            shape=(RESOLUTION, 3),
            dtype_name="torch.float32",
        ),
        "pred_pressure": _archived_array(
            tree,
            tensor_root=f"{interior}/point_data",
            metadata=point_meta,
            field="pred_pressure",
            relative="pred_pressure.memmap",
            shape=(RESOLUTION,),
            dtype_name="torch.float32",
        ),
        "pred_wss": _archived_array(
            tree,
            tensor_root=f"{interior}/point_data",
            metadata=point_meta,
            field="pred_wss",
            relative="pred_wss.memmap",
            shape=(RESOLUTION, 3),
            dtype_name="torch.float32",
        ),
        "true_pressure": _archived_array(
            tree,
            tensor_root=f"{interior}/point_data",
            metadata=point_meta,
            field="true_pressure",
            relative="true_pressure.memmap",
            shape=(RESOLUTION,),
            dtype_name="torch.float32",
        ),
        "true_wss": _archived_array(
            tree,
            tensor_root=f"{interior}/point_data",
            metadata=point_meta,
            field="true_wss",
            relative="true_wss.memmap",
            shape=(RESOLUTION, 3),
            dtype_name="torch.float32",
        ),
        "globals": globals_array,
    }


def _historical_metric_rows(payload: bytes) -> dict[str, Mapping[str, Any]]:
    _require(
        _sha256_bytes(payload) == EXPECTED_HISTORICAL_METRICS_SHA256,
        "Historical metrics SHA-256 differs",
    )
    rows: dict[str, Mapping[str, Any]] = {}
    summary: Mapping[str, Any] | None = None
    expected_ids = [spec[1] for spec in CASE_SPECS]
    for ordinal, line in enumerate(payload.splitlines()):
        record = _mapping(
            _strict_json_bytes(line, f"historical metrics line {ordinal + 1}"),
            f"Historical metrics line {ordinal + 1}",
        )
        if record.get("phase") == "infer_step":
            matches = [
                case_id
                for case_id in expected_ids
                if f"_{case_id}_domain_{case_id}" in str(record.get("sample_id"))
            ]
            _require(
                len(matches) == 1 and matches[0] not in rows,
                "Historical metrics case mapping changed",
            )
            rows[matches[0]] = record
        elif record.get("phase") == "infer_summary":
            _require(summary is None, "Historical metrics has duplicate summary")
            summary = record
    _require(
        list(rows) == expected_ids and summary is not None,
        "Historical metrics cohort or summary changed",
    )
    try:
        observed_mean = float(summary["metrics"]["pressure_l2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactInvalid("Historical pressure summary changed") from error
    _require(
        math.isfinite(observed_mean)
        and abs(observed_mean - ARCHIVED_PRESSURE_MEAN) <= PRESSURE_MEAN_ABS_TOLERANCE,
        "Historical pressure summary changed",
    )
    return rows


def _derive_archive_triangle_geometry(
    boundary_points: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive unit normals and normalized triangle-area weights in float64."""

    points64 = np.asarray(boundary_points, dtype=np.float64)
    cells64 = np.asarray(cells, dtype=np.int64)
    _require(
        points64.ndim == 2
        and points64.shape[1:] == (3,)
        and points64.shape[0] >= 3
        and cells64.shape == (RESOLUTION, 3)
        and bool(np.isfinite(points64).all()),
        "Archived triangle geometry schema or coordinates are invalid",
    )
    _require(
        bool(np.all(cells64 >= 0)) and bool(np.all(cells64 < points64.shape[0])),
        "Archived triangle connectivity is out of bounds",
    )
    vertices = points64[cells64]
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    crosses = np.cross(edge_1, edge_2)
    twice_areas = np.linalg.norm(crosses, axis=1)
    _require(
        bool(np.isfinite(crosses).all())
        and bool(np.isfinite(twice_areas).all())
        and bool(np.all(twice_areas > 0.0)),
        "Archived boundary contains a non-finite or degenerate triangle",
    )
    total_twice_area = float(np.sum(twice_areas, dtype=np.float64))
    _require(
        math.isfinite(total_twice_area) and total_twice_area > 0.0,
        "Archived total triangle area is invalid",
    )
    normals = np.ascontiguousarray(crosses / twice_areas[:, None])
    normalized_weights = np.ascontiguousarray(twice_areas / total_twice_area)
    _require(
        bool(np.isfinite(normals).all())
        and bool(np.isfinite(normalized_weights).all())
        and bool(np.all(normalized_weights > 0.0)),
        "Derived archive triangle geometry is invalid",
    )
    return normals, normalized_weights


def _normal_comparison(
    observed: np.ndarray,
    archive_derived: np.ndarray,
) -> dict[str, Any]:
    maximum = _maximum_absolute_difference(observed, archive_derived)
    return {
        "maximum_absolute_difference": maximum,
        "absolute_tolerance": PIPELINE_NORMAL_ABS_TOLERANCE,
        "passed": maximum <= PIPELINE_NORMAL_ABS_TOLERANCE,
    }


def _producer_weight_diagnostic(
    native_areas: np.ndarray,
    archive_normalized_weights: np.ndarray,
) -> dict[str, Any]:
    areas64 = np.asarray(native_areas, dtype=np.float64)
    values_valid = bool(
        areas64.shape == (RESOLUTION,)
        and bool(np.isfinite(areas64).all())
        and bool(np.all(areas64 > 0.0))
    )
    if values_valid:
        scaled_areas = areas64 / float(np.max(areas64))
        scaled_total = float(np.sum(scaled_areas, dtype=np.float64))
    else:
        scaled_areas = areas64
        scaled_total = math.nan
    computable = values_valid and math.isfinite(scaled_total) and scaled_total > 0.0
    if not computable:
        return {
            "computable": False,
            "maximum_absolute_difference": None,
            "producer_normalized_weight_sum": None,
            "archive_derived_normalized_weight_sum": float(
                np.sum(archive_normalized_weights, dtype=np.float64)
            ),
            "deciding": False,
        }
    producer_weights = scaled_areas / scaled_total
    return {
        "computable": True,
        "maximum_absolute_difference": _maximum_absolute_difference(
            producer_weights,
            archive_normalized_weights,
        ),
        "producer_normalized_weight_sum": float(
            np.sum(producer_weights, dtype=np.float64)
        ),
        "archive_derived_normalized_weight_sum": float(
            np.sum(archive_normalized_weights, dtype=np.float64)
        ),
        "deciding": False,
    }


def _archive_freestream_scales(globals_array: np.ndarray) -> tuple[float, float]:
    globals64 = np.asarray(globals_array, dtype=np.float64)
    _require(
        globals64.shape == (len(GLOBAL_FIELD_ORDER),)
        and bool(np.isfinite(globals64).all()),
        "Archived global inputs are invalid",
    )
    velocity = globals64[:3]
    p_inf = float(globals64[3])
    rho_inf = float(globals64[4])
    q_inf = 0.5 * rho_inf * float(np.dot(velocity, velocity))
    _require(
        math.isfinite(q_inf) and q_inf > 0.0 and math.isfinite(p_inf) and rho_inf > 0.0,
        "Archived freestream scales are invalid",
    )
    return q_inf, p_inf


def _reconstruct_training_array(
    physical: np.ndarray,
    *,
    field: str,
    q_inf: float,
    p_inf: float,
    normalization_state: Mapping[str, np.ndarray | float],
) -> np.ndarray:
    physical64 = np.asarray(physical, dtype=np.float64)
    _require(
        bool(np.isfinite(physical64).all()),
        f"Producer {field} physical array is non-finite",
    )
    if field == "pressure":
        reconstructed = (physical64 - p_inf) / q_inf
    elif field == "wss":
        mean = np.asarray(normalization_state["wss_mean"], dtype=np.float64)
        scale = float(normalization_state["wss_std"]) + NORMALIZATION_EPSILON
        _require(
            mean.shape == (3,)
            and bool(np.isfinite(mean).all())
            and math.isfinite(scale)
            and scale > 0.0,
            "Loaded WSS normalization state is invalid",
        )
        reconstructed = (physical64 / q_inf - mean) / scale
    else:
        raise ArtifactInvalid(f"Unknown training field {field!r}")
    _require(
        bool(np.isfinite(reconstructed).all()),
        f"Reconstructed {field} training array is non-finite",
    )
    return np.ascontiguousarray(reconstructed)


def _training_physical_comparison(
    training: np.ndarray,
    physical: np.ndarray,
    *,
    field: str,
    q_inf: float,
    p_inf: float,
    normalization_state: Mapping[str, np.ndarray | float],
) -> tuple[np.ndarray, dict[str, Any]]:
    reconstructed = _reconstruct_training_array(
        physical,
        field=field,
        q_inf=q_inf,
        p_inf=p_inf,
        normalization_state=normalization_state,
    )
    training64 = np.asarray(training, dtype=np.float64)
    _require(
        training64.shape == reconstructed.shape and bool(np.isfinite(training64).all()),
        f"Producer {field} training array is invalid",
    )
    difference = np.abs(training64 - reconstructed)
    allowed = TRAINING_PHYSICAL_ABS_TOLERANCE + (
        TRAINING_PHYSICAL_REL_TOLERANCE * np.abs(training64)
    )
    passed = bool(np.all(difference <= allowed))
    return reconstructed, {
        "maximum_absolute_difference": (
            0.0 if difference.size == 0 else float(np.max(difference))
        ),
        "maximum_allowed_difference": (
            0.0 if allowed.size == 0 else float(np.max(allowed))
        ),
        "absolute_tolerance": TRAINING_PHYSICAL_ABS_TOLERANCE,
        "relative_tolerance": TRAINING_PHYSICAL_REL_TOLERANCE,
        "tolerance_basis": (
            ">1.8x the exhaustive worst adjacent-float32-preimage distance "
            "across all 36 frozen p_inf=0 prediction and truth arrays"
        ),
        "passed": passed,
    }


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    result = float(np.linalg.norm(prediction64 - truth64)) / (
        float(np.linalg.norm(truth64)) + 1.0e-8
    )
    if not math.isfinite(result):
        raise ArtifactInvalid("Relative L2 is non-finite")
    return result


def _weighted_relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    normalized_weights: np.ndarray,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    _require(
        weights.shape == (RESOLUTION,)
        and bool(np.isfinite(weights).all())
        and bool(np.all(weights > 0.0))
        and abs(float(np.sum(weights, dtype=np.float64)) - 1.0) <= 1.0e-12,
        "Normalized archive-derived area weights are invalid",
    )
    if prediction64.ndim == 2:
        weights = weights[:, None]
    numerator = math.sqrt(
        float(np.sum(weights * (prediction64 - truth64) ** 2, dtype=np.float64))
    )
    denominator = (
        math.sqrt(float(np.sum(weights * truth64**2, dtype=np.float64))) + 1.0e-8
    )
    result = numerator / denominator
    if not math.isfinite(result):
        raise ArtifactInvalid("Area-weighted relative L2 is non-finite")
    return result


def _historical_pointwise_wss(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    numerator = np.linalg.norm(prediction64 - truth64, axis=-1)
    denominator = np.linalg.norm(truth64, axis=-1) + 1.0e-8
    result = float(np.mean(numerator / denominator, dtype=np.float64))
    if not math.isfinite(result):
        raise ArtifactInvalid("Historical pointwise WSS metric is non-finite")
    return result


def _adjudicate_case(
    *,
    spec: tuple[int, str, int, int, int],
    arrays_a: Mapping[str, np.ndarray],
    arrays_b: Mapping[str, np.ndarray],
    archived: Mapping[str, np.ndarray],
    metric_row: Mapping[str, Any],
    target_record: Mapping[str, Any],
    normalization_state: Mapping[str, np.ndarray | float],
) -> dict[str, Any]:
    ordinal, case_id, reader_index, _, start = spec
    prefix = _case_prefix(ordinal, case_id)
    a = {suffix: arrays_a[f"{prefix}__{suffix}"] for suffix in ARRAY_SCHEMAS}
    b = {suffix: arrays_b[f"{prefix}__{suffix}"] for suffix in ARRAY_SCHEMAS}
    replica_mismatches = [
        suffix for suffix in ARRAY_SCHEMAS if not _array_exact(a[suffix], b[suffix])
    ]
    replicas_exact = not replica_mismatches

    expected_ids = np.arange(start, start + RESOLUTION, dtype="<i8")
    selection_exact = _array_exact(a["selected_cell_ids_int64"], expected_ids)
    target_hashes_exact = bool(
        _array_sha256(a["raw_target_pressure_float32"])
        == target_record["pressure"]["selected_sha256"]
        and _array_sha256(a["raw_target_wss_float32"])
        == target_record["wss"]["selected_sha256"]
    )
    connectivity = _byte_difference(a["compacted_cells_int64"], archived["cells"])
    boundary_points = _byte_difference(
        a["pipeline_boundary_points_float32"],
        archived["boundary_points"],
    )
    globals_comparison = _byte_difference(
        a["pipeline_globals_float32"],
        archived["globals"],
    )
    query_comparison = _byte_difference(
        a["pipeline_queries_float32"],
        archived["query_points"],
    )
    query_max_abs = query_comparison["maximum_absolute_difference"]
    archive_derived_normals, archive_normalized_weights = (
        _derive_archive_triangle_geometry(
            archived["boundary_points"],
            archived["cells"],
        )
    )
    archive_stored_normal_control = _normal_comparison(
        archived["normals"],
        archive_derived_normals,
    )
    normal_max_abs = _maximum_absolute_difference(
        a["pipeline_normals_float32"],
        archived["normals"],
    )
    producer_geometry_diagnostics = {
        "pipeline_normals_vs_archive_derived": _normal_comparison(
            a["pipeline_normals_float32"],
            archive_derived_normals,
        ),
        "native_normals_vs_archive_derived": _normal_comparison(
            a["native_normals_float32"],
            archive_derived_normals,
        ),
        "native_normalized_weights_vs_archive_derived": (
            _producer_weight_diagnostic(
                a["native_areas_float64"],
                archive_normalized_weights,
            )
        ),
        "deciding": False,
    }
    truth_comparisons = {
        "pressure": _byte_difference(
            a["truth_pressure_physical_float32"],
            archived["true_pressure"],
        ),
        "wss": _byte_difference(
            a["truth_wss_physical_float32"],
            archived["true_wss"],
        ),
    }
    prediction_comparisons = {
        "pressure": _byte_difference(
            a["prediction_pressure_physical_float32"],
            archived["pred_pressure"],
        ),
        "wss": _byte_difference(
            a["prediction_wss_physical_float32"],
            archived["pred_wss"],
        ),
    }
    truth_exact = all(row["exact"] for row in truth_comparisons.values())
    predictions_exact = all(row["exact"] for row in prediction_comparisons.values())

    q_inf, p_inf = _archive_freestream_scales(archived["globals"])
    training_physical_comparisons: dict[str, dict[str, Any]] = {}
    for role in ("prediction", "truth"):
        for field in ("pressure", "wss"):
            name = f"{role}_{field}"
            _, comparison = _training_physical_comparison(
                a[f"{name}_training_float32"],
                a[f"{name}_physical_float32"],
                field=field,
                q_inf=q_inf,
                p_inf=p_inf,
                normalization_state=normalization_state,
            )
            training_physical_comparisons[name] = comparison
    training_physical_consistent = all(
        comparison["passed"] for comparison in training_physical_comparisons.values()
    )
    inputs_valid = bool(
        replicas_exact
        and selection_exact
        and target_hashes_exact
        and connectivity["exact"]
        and boundary_points["exact"]
        and globals_comparison["exact"]
        and query_comparison["exact"]
        and normal_max_abs <= PIPELINE_NORMAL_ABS_TOLERANCE
        and archive_stored_normal_control["passed"]
        and truth_exact
        and training_physical_consistent
    )

    metrics: dict[str, float] | None = None
    if training_physical_consistent:
        pred_pressure = a["prediction_pressure_training_float32"]
        pred_wss = a["prediction_wss_training_float32"]
        true_pressure = a["truth_pressure_training_float32"]
        true_wss = a["truth_wss_training_float32"]
        metrics = {
            "uniform_pressure_relative_l2": _relative_l2(
                pred_pressure,
                true_pressure,
            ),
            "uniform_wss_frobenius_relative_l2": _relative_l2(
                pred_wss,
                true_wss,
            ),
            "archive_normalized_area_weighted_pressure_relative_l2": (
                _weighted_relative_l2(
                    pred_pressure,
                    true_pressure,
                    archive_normalized_weights,
                )
            ),
            "archive_normalized_area_weighted_wss_frobenius_relative_l2": (
                _weighted_relative_l2(
                    pred_wss,
                    true_wss,
                    archive_normalized_weights,
                )
            ),
            "historical_pointwise_mean_wss_relative_l2_descriptive": (
                _historical_pointwise_wss(pred_wss, true_wss)
            ),
        }
    try:
        archived_pressure = float(metric_row["metrics"]["pressure_l2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactInvalid(
            f"Historical pressure metric changed for {case_id}"
        ) from error
    _require(
        math.isfinite(archived_pressure),
        f"Historical pressure metric is non-finite for {case_id}",
    )
    recomputed_pressure = (
        None if metrics is None else metrics["uniform_pressure_relative_l2"]
    )
    pressure_difference = (
        None
        if recomputed_pressure is None
        else abs(recomputed_pressure - archived_pressure)
    )
    pressure_metric_passed = bool(
        pressure_difference is not None
        and pressure_difference <= PRESSURE_CASE_ABS_TOLERANCE
    )
    if not inputs_valid:
        case_outcome = INVALID_REPLAY
    elif predictions_exact and pressure_metric_passed:
        case_outcome = "EXACT_CASE_REPLAY_PASS"
    else:
        case_outcome = "VALID_EXACT_CASE_REPLAY_REFUTATION"
    return {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": reader_index,
        "historical_start": start,
        "replicas_exact": replicas_exact,
        "replica_mismatch_arrays": replica_mismatches,
        "input_parity": {
            "selection_exact": selection_exact,
            "selected_target_bytes_exact": target_hashes_exact,
            "compacted_connectivity": connectivity,
            "pipeline_boundary_points": boundary_points,
            "pipeline_globals": globals_comparison,
            "pipeline_queries": query_comparison,
            "query_coordinate_max_abs": query_max_abs,
            "query_coordinate_requirement": ("raw-byte exact, including signed zero"),
            "pipeline_normal_max_abs": normal_max_abs,
            "pipeline_normal_abs_tolerance": PIPELINE_NORMAL_ABS_TOLERANCE,
            "archive_geometry_control": {
                "derived_from_manifest_bound_boundary_points_and_cells": True,
                "derivation_dtype": "float64",
                "stored_normals_vs_derived": archive_stored_normal_control,
                "normalized_area_weights_sum": float(
                    np.sum(archive_normalized_weights, dtype=np.float64)
                ),
                "absolute_areas_or_forces_supported": False,
            },
            "producer_geometry_diagnostics": producer_geometry_diagnostics,
            "truth_chain_control": truth_comparisons,
            "truth_chain_control_exact": truth_exact,
            "training_physical_chain_control": {
                "q_inf_from_archived_globals": q_inf,
                "p_inf_from_archived_globals": p_inf,
                "comparisons": training_physical_comparisons,
                "corrected_metric_input": (
                    "producer training arrays only after all comparisons pass"
                ),
                "passed": training_physical_consistent,
            },
            "passed": inputs_valid,
        },
        "historical_prediction_comparison": prediction_comparisons,
        "historical_predictions_exact": predictions_exact,
        "historical_pressure_metric": {
            "archived": archived_pressure,
            "independently_recomputed": recomputed_pressure,
            "absolute_difference": pressure_difference,
            "absolute_tolerance": PRESSURE_CASE_ABS_TOLERANCE,
            "passed": pressure_metric_passed,
        },
        "corrected_baseline_metrics": metrics,
        "outcome": case_outcome,
    }


def _classify_complete(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    inputs_valid = bool(
        len(cases) == CASE_COUNT
        and all(case["input_parity"]["passed"] for case in cases)
    )
    if not inputs_valid:
        return INVALID_STATUS, INVALID_REPLAY
    predictions_exact = all(case["historical_predictions_exact"] for case in cases)
    pressure_cases_passed = all(
        case["historical_pressure_metric"]["passed"] for case in cases
    )
    pressure_mean = float(
        np.mean(
            [
                case["historical_pressure_metric"]["independently_recomputed"]
                for case in cases
            ],
            dtype=np.float64,
        )
    )
    pressure_mean_passed = (
        abs(pressure_mean - ARCHIVED_PRESSURE_MEAN) <= PRESSURE_MEAN_ABS_TOLERANCE
    )
    if predictions_exact and pressure_cases_passed and pressure_mean_passed:
        return VALID_STATUS, EXACT_OUTCOME
    return VALID_STATUS, VALID_REFUTATION


def _base_result(
    *,
    status: str,
    outcome: str,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_outcome": outcome,
        "failures": list(failures),
    }


def adjudicate(
    *,
    producer_path: Path,
    producer_a_json: Path,
    producer_a_npz: Path,
    producer_b_json: Path,
    producer_b_npz: Path,
    target_input_manifest: Path,
    historical_predictions: Path,
    historical_manifest: Path,
    historical_metrics: Path,
    normalization_state: Path,
) -> dict[str, Any]:
    """Return pass, valid refutation, invalid, or incomplete without ambiguity."""

    if not _is_sha256(EXPECTED_PRODUCER_SHA256):
        return _base_result(
            status=INVALID_STATUS,
            outcome=INVALID_REPLAY,
            failures=[{"kind": "unresolved_producer_sha256_binding"}],
        )
    try:
        producer_payload, _ = _read_regular_file_bytes(
            producer_path,
            "Replay producer source",
        )
        _require(
            _sha256_bytes(producer_payload) == EXPECTED_PRODUCER_SHA256,
            "Replay producer source SHA-256 differs",
        )
        target_records, target_manifest_sha = _load_target_manifest(
            target_input_manifest
        )
        normalization, normalization_sha = _load_normalization_state(
            normalization_state
        )
        producer_a, arrays_a = _load_producer(
            producer_a_json,
            producer_a_npz,
            "A",
            target_records,
        )
        producer_b, arrays_b = _load_producer(
            producer_b_json,
            producer_b_npz,
            "B",
            target_records,
        )
        manifest_payload, manifest_sha = _load_verified_artifact_bytes(
            historical_manifest,
            "Historical prediction manifest",
        )
        entries = _parse_manifest_payload(manifest_payload)
        historical_tree, historical_tree_summary = _load_verified_tree(
            historical_predictions,
            entries,
            label="Historical prediction archive",
            retain_payloads=True,
        )
        metrics_payload, _ = _read_regular_file_bytes(
            historical_metrics,
            "Historical metrics",
        )
        historical_rows = _historical_metric_rows(metrics_payload)
        cases = [
            _adjudicate_case(
                spec=spec,
                arrays_a=arrays_a,
                arrays_b=arrays_b,
                archived=_load_archived_case(
                    historical_tree,
                    reader_index=spec[2],
                    case_id=spec[1],
                ),
                metric_row=historical_rows[spec[1]],
                target_record=target_records[spec[1]],
                normalization_state=normalization,
            )
            for spec in CASE_SPECS
        ]
    except ArtifactUnavailable as failure:
        return _base_result(
            status=INCOMPLETE_STATUS,
            outcome=INCOMPLETE_REPLAY,
            failures=[{"kind": "unavailable_artifact", "message": str(failure)}],
        )
    except ArtifactInvalid as failure:
        return _base_result(
            status=INVALID_STATUS,
            outcome=INVALID_REPLAY,
            failures=[{"kind": "invalid_artifact_or_control", "message": str(failure)}],
        )

    status, outcome = _classify_complete(cases)
    corrected_metrics_available = all(
        case["corrected_baseline_metrics"] is not None for case in cases
    )
    pressure_mean = (
        float(
            np.mean(
                [
                    case["historical_pressure_metric"]["independently_recomputed"]
                    for case in cases
                ],
                dtype=np.float64,
            )
        )
        if corrected_metrics_available
        else None
    )
    metric_names = (
        "uniform_pressure_relative_l2",
        "uniform_wss_frobenius_relative_l2",
        "archive_normalized_area_weighted_pressure_relative_l2",
        "archive_normalized_area_weighted_wss_frobenius_relative_l2",
        "historical_pointwise_mean_wss_relative_l2_descriptive",
    )
    baseline_means = (
        {
            name: float(
                np.mean(
                    [case["corrected_baseline_metrics"][name] for case in cases],
                    dtype=np.float64,
                )
            )
            for name in metric_names
        }
        if corrected_metrics_available
        else None
    )
    result = _base_result(status=status, outcome=outcome, failures=[])
    result.update(
        {
            "decision_gates": {
                "historical_archive_exact": historical_tree_summary["exact"],
                "current_replicas_exact": all(case["replicas_exact"] for case in cases),
                "selected_target_bytes_exact": all(
                    case["input_parity"]["selected_target_bytes_exact"]
                    for case in cases
                ),
                "model_consumed_archive_fields_parity_passed": all(
                    case["input_parity"]["compacted_connectivity"]["exact"]
                    and case["input_parity"]["pipeline_boundary_points"]["exact"]
                    and case["input_parity"]["pipeline_globals"]["exact"]
                    and case["input_parity"]["pipeline_queries"]["exact"]
                    and case["input_parity"]["pipeline_normal_max_abs"]
                    <= PIPELINE_NORMAL_ABS_TOLERANCE
                    for case in cases
                ),
                "model_consumed_archive_fields_scope": {
                    "only_fields_consumed_by_model": True,
                    "fields": [
                        "boundary connectivity",
                        "boundary points",
                        "boundary normals",
                        "global inputs",
                        "query points",
                    ],
                    "archive_derived_target_measure_weights_excluded": True,
                },
                "archive_geometry_controls_passed": all(
                    case["input_parity"]["archive_geometry_control"][
                        "stored_normals_vs_derived"
                    ]["passed"]
                    for case in cases
                ),
                "training_physical_chain_controls_passed": all(
                    case["input_parity"]["training_physical_chain_control"]["passed"]
                    for case in cases
                ),
                "truth_chain_control_exact": all(
                    case["input_parity"]["truth_chain_control_exact"] for case in cases
                ),
                "all_input_parity_passed": all(
                    case["input_parity"]["passed"] for case in cases
                ),
                "all_historical_predictions_exact": all(
                    case["historical_predictions_exact"] for case in cases
                ),
                "all_pressure_case_metrics_passed": all(
                    case["historical_pressure_metric"]["passed"] for case in cases
                ),
                "pressure_mean_passed": (
                    pressure_mean is not None
                    and abs(pressure_mean - ARCHIVED_PRESSURE_MEAN)
                    <= PRESSURE_MEAN_ABS_TOLERANCE
                ),
            },
            "pressure_replay": {
                "independently_recomputed_mean": pressure_mean,
                "archived_mean": ARCHIVED_PRESSURE_MEAN,
                "absolute_difference": (
                    None
                    if pressure_mean is None
                    else abs(pressure_mean - ARCHIVED_PRESSURE_MEAN)
                ),
                "absolute_tolerance": PRESSURE_MEAN_ABS_TOLERANCE,
                "passed": (
                    pressure_mean is not None
                    and abs(pressure_mean - ARCHIVED_PRESSURE_MEAN)
                    <= PRESSURE_MEAN_ABS_TOLERANCE
                ),
            },
            "corrected_stage2_baseline": {
                "licensed": outcome == EXACT_OUTCOME and corrected_metrics_available,
                "metrics_available": corrected_metrics_available,
                "case_count": CASE_COUNT,
                "means": baseline_means,
                "noninferiority_ratio": NONINFERIORITY_RATIO,
                "prospective_absolute_ceilings": {
                    name: NONINFERIORITY_RATIO * value
                    for name, value in (baseline_means or {}).items()
                    if name != "historical_pointwise_mean_wss_relative_l2_descriptive"
                },
                "area_weight_scope": (
                    "normalized relative triangle weights derived in float64 "
                    "from manifest-bound archived boundary geometry"
                ),
                "does_not_support": [
                    "absolute native areas",
                    "absolute force integration",
                ],
                "historical_wss_warning": (
                    "historical_pointwise_mean_wss_relative_l2_descriptive is "
                    "the confirmed historical reduction bug and is not a "
                    "licensing endpoint"
                ),
            },
            "cases": cases,
            "tree_controls": {
                "historical_archive": historical_tree_summary,
            },
            "provenance": {
                "reducer_path": str(Path(__file__).resolve()),
                "reducer_sha256": _sha256_bytes(
                    _read_regular_file_bytes(
                        Path(__file__).resolve(),
                        "Replay adjudicator source",
                    )[0]
                ),
                "producer_sha256": EXPECTED_PRODUCER_SHA256,
                "producer_a_json_sha256": producer_a["json_sha256"],
                "producer_a_npz_sha256": producer_a["npz_sha256"],
                "producer_b_json_sha256": producer_b["json_sha256"],
                "producer_b_npz_sha256": producer_b["npz_sha256"],
                "target_input_manifest_sha256": target_manifest_sha,
                "historical_manifest_sha256": manifest_sha,
                "historical_metrics_sha256": EXPECTED_HISTORICAL_METRICS_SHA256,
                "normalization_state_sha256": normalization_sha,
            },
            "scientific_scope": {
                "supports": (
                    "reproducibility of one frozen epoch-491 K=10000 legacy "
                    "baseline on the 36-case ID-reference cohort"
                ),
                "does_not_support": [
                    "canonical-path accuracy",
                    "noninferiority",
                    "superiority",
                    "other resolutions",
                    "OOD or population generalization",
                    "training-time invariance",
                    "architecture claims",
                    "H-QC mechanism claims",
                    "absolute native areas or forces",
                ],
            },
        }
    )
    return result


def _atomic_publish(path: Path, payload: bytes) -> str:
    output = Path(os.path.abspath(path))
    sidecar = output.with_name(f"{output.name}.sha256")
    for destination in (output, sidecar):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    temporaries: dict[Path, tuple[Path, tuple[int, int]]] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for destination, content in (
            (output, payload),
            (sidecar, sidecar_payload),
        ):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                metadata = os.fstat(stream.fileno())
            temporaries[destination] = (
                temporary,
                (metadata.st_dev, metadata.st_ino),
            )
        for destination in (output, sidecar):
            temporary, _ = temporaries[destination]
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        for destination, content in (
            (output, payload),
            (sidecar, sidecar_payload),
        ):
            observed, identity = _read_regular_file_bytes(
                destination,
                f"Published {destination.name}",
            )
            _, expected_identity = temporaries[destination]
            if observed != content or identity != expected_identity:
                raise OSError(f"Published output changed: {destination}")
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
        for temporary, _ in temporaries.values():
            temporary.unlink(missing_ok=True)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-a-json", type=Path, required=True)
    parser.add_argument("--producer-a-npz", type=Path, required=True)
    parser.add_argument("--producer-b-json", type=Path, required=True)
    parser.add_argument("--producer-b-npz", type=Path, required=True)
    parser.add_argument("--target-input-manifest", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--normalization-state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = adjudicate(
        producer_path=args.producer,
        producer_a_json=args.producer_a_json,
        producer_a_npz=args.producer_a_npz,
        producer_b_json=args.producer_b_json,
        producer_b_npz=args.producer_b_npz,
        target_input_manifest=args.target_input_manifest,
        historical_predictions=args.historical_predictions,
        historical_manifest=args.historical_manifest,
        historical_metrics=args.historical_metrics,
        normalization_state=args.normalization_state,
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    digest = _atomic_publish(args.output_json, payload)
    print(
        f"{result['status']} outcome={result['decision_outcome']} json_sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
