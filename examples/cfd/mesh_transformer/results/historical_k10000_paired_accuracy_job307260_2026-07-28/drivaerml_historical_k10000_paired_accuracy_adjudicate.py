# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Adjudicate the paired K=10k canonical-versus-legacy accuracy experiment.

Only this reducer receives both arms and the frozen Stage-B license.  The two
canonical producers remain target-blind.  A fresh unchanged-legacy sentinel is
an execution control only: all of its arrays must reproduce the sealed Stage-B
arrays exactly, but it cannot redefine the preregistered baseline or ceilings.

The four deciding endpoints are cohort means of per-case relative L2 metrics.
Both arms use the sealed Stage-B training truths and the same normalized
triangle-area weights, independently derived in float64 from the sealed
Stage-B boundary points and connectivity.  Canonical A/B disagreement or any
input-control drift is invalidity.  A reproducible canonical metric above one
or more inclusive frozen ceilings is a valid scientific refutation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import drivaerml_historical_k10000_replay_adjudicate as stage_b
import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_paired_accuracy_adjudication"
VALID_STATUS = "VALID_CANONICAL_NONINFERIORITY_ADJUDICATION"
INVALID_STATUS = "INVALID_CANONICAL_NONINFERIORITY_ADJUDICATION"
INCOMPLETE_STATUS = "INCOMPLETE_CANONICAL_NONINFERIORITY_ADJUDICATION"

NONINFERIORITY_SUCCESS_OUTCOME = "CANONICAL_NONINFERIORITY_PASS"
VALID_REFUTATION = "VALID_CANONICAL_NONINFERIORITY_REFUTATION"
INVALID_COMPARISON = "INVALID_CANONICAL_COMPARISON"
INCOMPLETE_COMPARISON = "INCOMPLETE_CANONICAL_COMPARISON"

RESOLUTION = 10_000
CASE_COUNT = 36
ARRAYS_PER_CANONICAL_CASE = 22
ARRAYS_PER_LEGACY_CASE = 20
REQUESTED_EPOCH = 491
PHYSICAL_LENGTH = 5.0
MODEL_REFERENCE_LENGTH = 8.0
CANONICAL_LENGTH_SCALE = PHYSICAL_LENGTH * MODEL_REFERENCE_LENGTH
CANONICAL_PHYSICAL_INVERSE_ABS_TOLERANCE = 2.0e-6
TRAINING_PHYSICAL_ABS_TOLERANCE = 2.0e-6
NORMALIZATION_EPSILON = 1.0e-8
WSS_NORMALIZATION_STD = float(np.float32(0.00313))

EXPECTED_CANONICAL_PRODUCER_SHA256 = (
    "86455a2fecd018896ecdc7989b9ea558ad8c72fd71d0d3f2f7f35d0c6f429d66"
)
EXPECTED_LEGACY_PRODUCER_SHA256 = (
    "bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5"
)
EXPECTED_RUNTIME_HELPER_SHA256 = (
    "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
)
EXPECTED_CANONICAL_HELPER_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)
EXPECTED_STAGE_B_ADJUDICATION_SHA256 = (
    "6a2d2274611e7be533a18ba9b68f7793ccd3a4777e96df389552d55f64839a1b"
)
EXPECTED_STAGE_B_RESULT_SHA256 = (
    "89b861cc0b0ab3f0b944f96038e41814e01175c075fed75fd828e95aaebbf84d"
)
EXPECTED_STAGE_B_TREE_MANIFEST_SHA256 = (
    "79e40c8117b47dafcb755faf4c16e00eb117fd3b2880c71bdbf23d51d7591810"
)
EXPECTED_STAGE_B_A_JSON_SHA256 = (
    "83f0d53326a128def18d538d427ac48b11a0d789417366cfd7c1dc020b193d67"
)
EXPECTED_STAGE_B_B_JSON_SHA256 = (
    "c7d1254335ac381524dace6c9b6b45c0ae1e275df99b9ca505cbe2395c957b19"
)
EXPECTED_STAGE_B_NPZ_SHA256 = (
    "38353473af9eba1e606e170c70c5d6ed4d7fd736d631c6388dd2244dc1e203c4"
)
EXPECTED_STAGE_B_REDUCER_SHA256 = (
    "4b4f4c893ca154b589b9d1276bf4f43cd52008664f75f3c56fcc66aa26ce8f68"
)
EXPECTED_MODEL_CHECKPOINT_SHA256 = (
    "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
)
EXPECTED_TRAINING_STATE_SHA256 = (
    "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
)
EXPECTED_NORMALIZATION_STATE_SHA256 = (
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
EXPECTED_MODEL_SOURCE_SHA256 = (
    "9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892"
)
EXPECTED_DATASET_CONFIG_SHA256 = (
    "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
)
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
)

CASE_SPECS = stage_b.CASE_SPECS
GLOBAL_FIELD_ORDER = stage_b.GLOBAL_FIELD_ORDER

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
# raw_points is newly persisted by the canonical arm and is independently
# checked against its historical-pipeline image.  The other controls are
# byte-paired directly to the sealed Stage-B arrays.
LEGACY_SHARED_CONTROL_SUFFIXES = tuple(
    suffix for suffix in PAIRING_CONTROL_SUFFIXES if suffix != "raw_points_float32"
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
CANONICAL_ARRAY_SUFFIXES = (
    *PAIRING_CONTROL_SUFFIXES,
    *CANONICAL_GEOMETRY_SUFFIXES,
    *PREDICTION_SUFFIXES,
)

CANONICAL_ARRAY_SCHEMAS: Mapping[
    str,
    tuple[tuple[int | None, ...], np.dtype[Any]],
] = {
    "selected_cell_ids_int64": ((RESOLUTION,), np.dtype("<i8")),
    "compacted_cells_int64": ((RESOLUTION, 3), np.dtype("<i8")),
    "raw_points_float32": ((None, 3), np.dtype("<f4")),
    "raw_centroids_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "native_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "native_areas_float64": ((RESOLUTION,), np.dtype("<f8")),
    "pipeline_boundary_points_float32": ((None, 3), np.dtype("<f4")),
    "pipeline_queries_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "pipeline_globals_float32": (
        (len(GLOBAL_FIELD_ORDER),),
        np.dtype("<f4"),
    ),
    "pipeline_center_float32": ((3,), np.dtype("<f4")),
    "canonical_cells_int64": ((RESOLUTION, 3), np.dtype("<i8")),
    "canonical_points_float32": ((None, 3), np.dtype("<f4")),
    "canonical_centroids_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "canonical_areas_float32": ((RESOLUTION,), np.dtype("<f4")),
    "canonical_normals_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "canonical_physical_center_float64": ((3,), np.dtype("<f8")),
    "canonical_queries_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "prediction_pressure_training_float32": ((RESOLUTION,), np.dtype("<f4")),
    "prediction_wss_training_float32": ((RESOLUTION, 3), np.dtype("<f4")),
    "prediction_pressure_physical_float32": ((RESOLUTION,), np.dtype("<f4")),
    "prediction_wss_physical_float32": ((RESOLUTION, 3), np.dtype("<f4")),
}

DECIDING_METRICS = (
    "uniform_pressure_relative_l2",
    "uniform_wss_frobenius_relative_l2",
    "archive_normalized_area_weighted_pressure_relative_l2",
    "archive_normalized_area_weighted_wss_frobenius_relative_l2",
)
DESCRIPTIVE_WSS_METRIC = "historical_pointwise_mean_wss_relative_l2_descriptive"
FROZEN_BASELINE_MEANS = {
    "uniform_pressure_relative_l2": 0.1671630996206787,
    "uniform_wss_frobenius_relative_l2": 0.27544940646582583,
    "archive_normalized_area_weighted_pressure_relative_l2": (0.18192540063679644),
    "archive_normalized_area_weighted_wss_frobenius_relative_l2": (0.21081553813258389),
}
FROZEN_CEILINGS = {
    "uniform_pressure_relative_l2": 0.1705063616130923,
    "uniform_wss_frobenius_relative_l2": 0.2809583945951423,
    "archive_normalized_area_weighted_pressure_relative_l2": (0.18556390864953237),
    "archive_normalized_area_weighted_wss_frobenius_relative_l2": (0.21503184889523558),
}
FROZEN_DESCRIPTIVE_WSS_MEAN = 0.7052652041675186
BASELINE_RECOMPUTE_ABS_TOLERANCE = 5.0e-15

_TREE_MANIFEST_LINE = re.compile(rb"^([0-9a-f]{64})  (\./[^\x00\r\n]+)$")
_EXPECTED_RANK_ENVIRONMENT = {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
}

ArtifactUnavailable = stage_b.ArtifactUnavailable
ArtifactInvalid = stage_b.ArtifactInvalid


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactInvalid(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} key set differs")


def _is_sha256(value: Any) -> bool:
    return stage_b._is_sha256(value)


def _sha256_bytes(value: bytes) -> str:
    return stage_b._sha256_bytes(value)


def _array_sha256(value: np.ndarray) -> str:
    return stage_b._array_sha256(value)


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return stage_b._array_exact(left, right)


def _contains_value(value: Any, expected: str) -> bool:
    return stage_b._contains_value(value, expected)


def _reject_canonical_producer_conclusions(
    value: Any,
    prefix: str = "",
) -> None:
    """Reject producer-side conclusions while allowing its one false audit key."""

    allowed_metric_key = "producer_reads_supervision_archive_metrics_or_ceilings"
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(
                token in lowered for token in ("outcome", "decision", "comparison")
            ) or ("metric" in lowered and key != allowed_metric_key):
                raise ArtifactInvalid(
                    f"Canonical producer contains forbidden conclusion key {path}"
                )
            _reject_canonical_producer_conclusions(nested, path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_canonical_producer_conclusions(nested, f"{prefix}[{index}]")


def _case_prefix(ordinal: int, case_id: str) -> str:
    return f"case_{ordinal:02d}_{case_id}"


def _canonical_array_names() -> tuple[str, ...]:
    return tuple(
        f"{_case_prefix(ordinal, case_id)}__{suffix}"
        for ordinal, case_id, _, _, _ in CASE_SPECS
        for suffix in CANONICAL_ARRAY_SUFFIXES
    )


def _legacy_array_names() -> tuple[str, ...]:
    return tuple(
        f"{_case_prefix(ordinal, case_id)}__{suffix}"
        for ordinal, case_id, _, _, _ in CASE_SPECS
        for suffix in stage_b.ARRAY_SCHEMAS
    )


def _verified_payload(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    payload, digest = stage_b._load_verified_artifact_bytes(path, label)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"{label} SHA-256 differs")
    return payload, digest


def _strict_document(payload: bytes, label: str) -> Mapping[str, Any]:
    return _mapping(stage_b._strict_json_bytes(payload, label), label)


def _read_source(path: Path, label: str, expected_sha256: str) -> str:
    payload, _ = stage_b._read_regular_file_bytes(path, label)
    digest = _sha256_bytes(payload)
    _require(_is_sha256(expected_sha256), f"{label} binding is unresolved")
    _require(digest == expected_sha256, f"{label} SHA-256 differs")
    return digest


def _shape_matches(
    observed: tuple[int, ...],
    expected: tuple[int | None, ...],
) -> bool:
    return len(observed) == len(expected) and all(
        wanted is None or actual == wanted
        for actual, wanted in zip(observed, expected, strict=True)
    )


def _validate_array_manifest(
    manifest_value: Any,
    arrays: Mapping[str, np.ndarray],
    label: str,
) -> None:
    manifest = _mapping(manifest_value, label)
    _require(set(manifest) == set(arrays), f"{label} key set differs")
    for name, value in arrays.items():
        record = _mapping(manifest[name], f"{label} {name}")
        _exact_keys(record, {"shape", "dtype", "sha256"}, f"{label} {name}")
        _require(
            record
            == {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            },
            f"{label} differs for {name}",
        )


def _validate_canonical_case_schema(
    arrays: Mapping[str, np.ndarray],
    *,
    spec: tuple[int, str, int, int, int],
    n_compacted_points: Any,
) -> None:
    ordinal, case_id, _, _, start = spec
    prefix = _case_prefix(ordinal, case_id)
    _require(
        type(n_compacted_points) is int and 3 <= n_compacted_points <= 3 * RESOLUTION,
        f"Canonical compacted-point count differs for {case_id}",
    )
    for suffix, (shape, dtype) in CANONICAL_ARRAY_SCHEMAS.items():
        value = arrays[f"{prefix}__{suffix}"]
        _require(
            value.dtype == dtype and _shape_matches(value.shape, shape),
            f"Canonical array schema differs for {prefix}__{suffix}",
        )
        if np.issubdtype(value.dtype, np.floating):
            _require(
                bool(np.isfinite(value).all()),
                f"Canonical array contains non-finite values: {prefix}__{suffix}",
            )
    for suffix in (
        "raw_points_float32",
        "pipeline_boundary_points_float32",
        "canonical_points_float32",
    ):
        _require(
            arrays[f"{prefix}__{suffix}"].shape == (n_compacted_points, 3),
            f"Canonical point count differs for {case_id} {suffix}",
        )
    expected_ids = np.arange(start, start + RESOLUTION, dtype="<i8")
    _require(
        _array_exact(
            arrays[f"{prefix}__selected_cell_ids_int64"],
            expected_ids,
        ),
        f"Canonical selected IDs differ for {case_id}",
    )
    cells = arrays[f"{prefix}__compacted_cells_int64"]
    _require(
        bool(np.all(cells >= 0))
        and int(cells.max()) == n_compacted_points - 1
        and np.array_equal(np.unique(cells), np.arange(n_compacted_points)),
        f"Canonical compacted connectivity is not dense for {case_id}",
    )
    _require(
        bool(np.all(arrays[f"{prefix}__native_areas_float64"] > 0.0))
        and bool(np.all(arrays[f"{prefix}__canonical_areas_float32"] > 0.0)),
        f"Canonical areas are not strictly positive for {case_id}",
    )


def _validate_timestamp(value: Any, label: str) -> None:
    _require(type(value) is str, f"{label} timestamp differs")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ArtifactInvalid(f"{label} timestamp is invalid") from error
    _require(parsed.tzinfo is not None, f"{label} timestamp is not timezone-aware")


def _validate_canonical_document(
    document: Mapping[str, Any],
    *,
    lane_label: str,
    npz_path: Path,
    npz_sha256: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    _reject_canonical_producer_conclusions(document)
    _exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "generated_at_utc",
            "lane_label",
            "contract",
            "summary",
            "cases",
            "npz",
            "provenance",
        },
        f"Canonical lane {lane_label}",
    )
    _require(
        document.get("schema_version") == 1
        and document.get("artifact_kind")
        == "phase1_historical_k10000_canonical_arm_producer"
        and document.get("status")
        == "COMPLETED_HISTORICAL_K10000_CANONICAL_ARM_PRODUCER"
        and document.get("lane_label") == lane_label,
        f"Canonical lane {lane_label} identity differs",
    )
    _validate_timestamp(
        document.get("generated_at_utc"), f"Canonical lane {lane_label}"
    )

    contract = _mapping(
        document.get("contract"), f"Canonical lane {lane_label} contract"
    )
    _require(
        contract.get("arm") == "canonical_source_geometry"
        and contract.get("public_api")
        == (
            "model.encode(domain, canonical_source_geometry=geometry); "
            "model.decode(encoded, canonical_centroid_query_mesh)"
        )
        and contract.get("canonical_source_geometry_present") is True
        and contract.get("producer_reads_supervision_archive_metrics_or_ceilings")
        is False
        and contract.get("resolution") == RESOLUTION
        and contract.get("precision") == "bfloat16"
        and contract.get("torch_compile") is False
        and contract.get("requested_epoch") == REQUESTED_EPOCH
        and contract.get("case_count") == CASE_COUNT
        and contract.get("canonical_construction")
        == (
            "float64 raw geometry -> physical area center -> divide by "
            "L_ref*model_reference_length -> one float32 cast"
        )
        and contract.get("query_frame") == "canonical_trace_centroids"
        and contract.get("encode_count_per_case") == 1
        and contract.get("decode_count_per_case") == 1
        and contract.get("global_field_order") == list(GLOBAL_FIELD_ORDER)
        and contract.get("pairing_control_suffixes") == list(PAIRING_CONTROL_SUFFIXES)
        and contract.get("canonical_geometry_suffixes")
        == list(CANONICAL_GEOMETRY_SUFFIXES)
        and contract.get("prediction_suffixes") == list(PREDICTION_SUFFIXES)
        and contract.get("measure_weights_required_absent") is True
        and contract.get("synthetic_placeholders_stripped_before_model") is True,
        f"Canonical lane {lane_label} execution contract differs",
    )
    summary = _mapping(document.get("summary"), f"Canonical lane {lane_label} summary")
    _require(
        summary
        == {
            "case_count": CASE_COUNT,
            "array_count": CASE_COUNT * ARRAYS_PER_CANONICAL_CASE,
            "valid_case_count": CASE_COUNT,
        },
        f"Canonical lane {lane_label} summary differs",
    )

    cases = document.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == CASE_COUNT,
        f"Canonical lane {lane_label} case count differs",
    )
    for spec, case_value in zip(CASE_SPECS, cases, strict=True):
        ordinal, case_id, reader_index, n_cells, start = spec
        case = _mapping(
            case_value,
            f"Canonical lane {lane_label} case {case_id}",
        )
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
                "canonical_frame",
                "validity",
                "array_sha256",
            },
            f"Canonical lane {lane_label} case {case_id}",
        )
        _require(
            (
                case.get("cohort_ordinal"),
                case.get("case_id"),
                case.get("reader_index"),
                case.get("n_master_cells"),
                case.get("historical_start"),
                case.get("resolution"),
            )
            == (ordinal, case_id, reader_index, n_cells, start, RESOLUTION),
            f"Canonical lane {lane_label} case identity differs for {case_id}",
        )
        validity = _mapping(
            case.get("validity"),
            f"Canonical lane {lane_label} validity {case_id}",
        )
        public_api = _mapping(
            validity.get("public_api"),
            f"Canonical lane {lane_label} public API {case_id}",
        )
        decode = _mapping(
            public_api.get("decode_contract"),
            f"Canonical lane {lane_label} decode contract {case_id}",
        )
        _require(
            validity.get("geometry_only_input") is True
            and validity.get("synthetic_placeholders_stripped_before_model") is True
            and validity.get("measure_weights_absent") is True
            and validity.get("canonical_construction_replay_passed") is True
            and validity.get("passed") is True
            and decode.get("canonical_queries_equal_canonical_centroids_raw_byte_exact")
            is True
            and decode.get("canonical_query_storage_identity") is True,
            f"Canonical lane {lane_label} validity differs for {case_id}",
        )
        _validate_canonical_case_schema(
            arrays,
            spec=spec,
            n_compacted_points=case.get("n_compacted_points"),
        )
        prefix = _case_prefix(ordinal, case_id)
        expected_hashes = {
            suffix: _array_sha256(arrays[f"{prefix}__{suffix}"])
            for suffix in CANONICAL_ARRAY_SUFFIXES
        }
        _require(
            dict(_mapping(case.get("array_sha256"), "Canonical case hashes"))
            == expected_hashes,
            f"Canonical lane {lane_label} array hashes differ for {case_id}",
        )

    npz = _mapping(document.get("npz"), f"Canonical lane {lane_label} NPZ")
    _require(
        npz.get("filename") == npz_path.name
        and npz.get("sha256") == npz_sha256
        and npz.get("array_count") == CASE_COUNT * ARRAYS_PER_CANONICAL_CASE,
        f"Canonical lane {lane_label} NPZ identity differs",
    )
    _validate_array_manifest(
        npz.get("array_manifest"),
        arrays,
        f"Canonical lane {lane_label} array manifest",
    )

    provenance = _mapping(
        document.get("provenance"),
        f"Canonical lane {lane_label} provenance",
    )
    _require(
        provenance.get("producer_sha256") == EXPECTED_CANONICAL_PRODUCER_SHA256
        and provenance.get("requested_epoch") == REQUESTED_EPOCH
        and provenance.get("loaded_epoch") == REQUESTED_EPOCH
        and provenance.get("npz_sha256") == npz_sha256,
        f"Canonical lane {lane_label} producer provenance differs",
    )
    for expected, name in (
        (EXPECTED_LEGACY_PRODUCER_SHA256, "legacy support"),
        (EXPECTED_RUNTIME_HELPER_SHA256, "runtime helper"),
        (EXPECTED_CANONICAL_HELPER_SHA256, "canonical helper"),
        (EXPECTED_EXECUTION_SOURCE_TREE_SHA256, "execution source tree"),
        (EXPECTED_MODEL_SOURCE_SHA256, "model source"),
        (EXPECTED_MODEL_CHECKPOINT_SHA256, "checkpoint"),
        (EXPECTED_TRAINING_STATE_SHA256, "training state"),
        (EXPECTED_NORMALIZATION_STATE_SHA256, "normalization state"),
        (EXPECTED_GEOMETRY_MANIFEST_SHA256, "geometry manifest"),
        (EXPECTED_DATASET_CONFIG_SHA256, "dataset config"),
        (EXPECTED_DATASET_MANIFEST_SHA256, "dataset manifest"),
        (EXPECTED_RESOLVED_CONFIG_SHA256, "resolved config"),
    ):
        _require(
            _contains_value(provenance, expected),
            f"Canonical lane {lane_label} does not bind {name}",
        )
    process = _mapping(
        provenance.get("process"),
        f"Canonical lane {lane_label} process provenance",
    )
    _require(
        type(process.get("hostname")) is str
        and bool(process.get("hostname"))
        and type(process.get("pid")) is int
        and process.get("pid") > 0
        and type(process.get("parent_pid")) is int
        and process.get("parent_pid") > 0
        and process.get("cuda_visible_devices_token")
        == ("0" if lane_label == "A" else "1")
        and process.get("rank_environment") == _EXPECTED_RANK_ENVIRONMENT
        and type(process.get("slurm_job_id")) is str
        and bool(process.get("slurm_job_id")),
        f"Canonical lane {lane_label} process provenance differs",
    )


def _load_canonical_lane(
    json_path: Path,
    npz_path: Path,
    lane_label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_payload, json_sha256 = _verified_payload(
        json_path,
        f"Canonical lane {lane_label} JSON",
    )
    npz_payload, npz_sha256 = _verified_payload(
        npz_path,
        f"Canonical lane {lane_label} NPZ",
    )
    document = _strict_document(json_payload, f"Canonical lane {lane_label}")
    arrays = stage_b._load_npz_bytes(
        npz_payload,
        f"Canonical lane {lane_label} NPZ",
        _canonical_array_names(),
    )
    _validate_canonical_document(
        document,
        lane_label=lane_label,
        npz_path=npz_path,
        npz_sha256=npz_sha256,
        arrays=arrays,
    )
    return {
        "document": document,
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
    }, arrays


def _load_sealed_stage_b_lane(
    json_path: Path,
    npz_path: Path,
    lane_label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_json = (
        EXPECTED_STAGE_B_A_JSON_SHA256
        if lane_label == "A"
        else EXPECTED_STAGE_B_B_JSON_SHA256
    )
    json_payload, json_sha256 = _verified_payload(
        json_path,
        f"Sealed Stage-B lane {lane_label} JSON",
        expected_sha256=expected_json,
    )
    npz_payload, npz_sha256 = _verified_payload(
        npz_path,
        f"Sealed Stage-B lane {lane_label} NPZ",
        expected_sha256=EXPECTED_STAGE_B_NPZ_SHA256,
    )
    document = _strict_document(json_payload, f"Sealed Stage-B lane {lane_label}")
    arrays = stage_b._load_npz_bytes(
        npz_payload,
        f"Sealed Stage-B lane {lane_label} NPZ",
        _legacy_array_names(),
    )
    return {
        "document": document,
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
    }, arrays


def _target_records_from_sealed(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for ordinal, case_id, _, _, _ in CASE_SPECS:
        prefix = _case_prefix(ordinal, case_id)
        records[case_id] = {
            "pressure": {
                "selected_sha256": _array_sha256(
                    arrays[f"{prefix}__raw_target_pressure_float32"]
                )
            },
            "wss": {
                "selected_sha256": _array_sha256(
                    arrays[f"{prefix}__raw_target_wss_float32"]
                )
            },
        }
    return records


def _load_fresh_legacy_sentinel(
    json_path: Path,
    npz_path: Path,
    target_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_payload, json_sha256 = _verified_payload(
        json_path,
        "Fresh legacy sentinel JSON",
    )
    npz_payload, npz_sha256 = _verified_payload(
        npz_path,
        "Fresh legacy sentinel NPZ",
    )
    document = _strict_document(json_payload, "Fresh legacy sentinel")
    arrays = stage_b._load_npz_bytes(
        npz_payload,
        "Fresh legacy sentinel NPZ",
        _legacy_array_names(),
    )
    stage_b._validate_producer_document(
        document,
        expected_label="A",
        npz_path=npz_path,
        npz_sha256=npz_sha256,
        arrays=arrays,
        target_records=target_records,
    )
    return {
        "document": document,
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
    }, arrays


def _parse_tree_manifest(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in payload.splitlines():
        match = _TREE_MANIFEST_LINE.fullmatch(line)
        _require(match is not None, "Sealed Stage-B tree manifest is malformed")
        digest = match.group(1).decode("ascii")
        path = match.group(2).decode("utf-8")
        _require(path not in result, "Sealed Stage-B tree manifest has duplicates")
        result[path] = digest
    expected = {
        "./artifacts/historical_k10000_replay_adjudication.json": (
            EXPECTED_STAGE_B_ADJUDICATION_SHA256
        ),
        "./artifacts/historical_k10000_replay_lane_A.json": (
            EXPECTED_STAGE_B_A_JSON_SHA256
        ),
        "./artifacts/historical_k10000_replay_lane_B.json": (
            EXPECTED_STAGE_B_B_JSON_SHA256
        ),
        "./artifacts/historical_k10000_replay_lane_A.npz": (
            EXPECTED_STAGE_B_NPZ_SHA256
        ),
        "./artifacts/historical_k10000_replay_lane_B.npz": (
            EXPECTED_STAGE_B_NPZ_SHA256
        ),
        "./drivaerml_historical_k10000_replay.py": (EXPECTED_LEGACY_PRODUCER_SHA256),
        "./drivaerml_historical_k10000_replay_adjudicate.py": (
            EXPECTED_STAGE_B_REDUCER_SHA256
        ),
    }
    _require(
        all(result.get(path) == digest for path, digest in expected.items()),
        "Sealed Stage-B tree manifest does not bind required artifacts",
    )
    return result


def _validate_stage_b_license(
    adjudication: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> None:
    _require(
        adjudication.get("schema_version") == 1
        and adjudication.get("artifact_kind")
        == "phase1_historical_k10000_replay_adjudication"
        and adjudication.get("status") == "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
        and adjudication.get("decision_outcome") == "EXACT_HISTORICAL_REPLAY_PASS"
        and adjudication.get("failures") == [],
        "Sealed Stage-B adjudication is not an exact valid replay",
    )
    gates = _mapping(adjudication.get("decision_gates"), "Stage-B decision gates")
    _require(
        all(
            value is True
            for key, value in gates.items()
            if key != "model_consumed_archive_fields_scope"
        ),
        "Sealed Stage-B decision gate is not true",
    )
    baseline = _mapping(
        adjudication.get("corrected_stage2_baseline"),
        "Stage-B corrected baseline",
    )
    _require(
        baseline.get("licensed") is True
        and baseline.get("metrics_available") is True
        and baseline.get("case_count") == CASE_COUNT
        and all(
            baseline.get("means", {}).get(name) == value
            for name, value in FROZEN_BASELINE_MEANS.items()
        )
        and baseline.get("means", {}).get(DESCRIPTIVE_WSS_METRIC)
        == FROZEN_DESCRIPTIVE_WSS_MEAN
        and baseline.get("prospective_absolute_ceilings") == FROZEN_CEILINGS,
        "Sealed Stage-B corrected baseline or ceilings differ",
    )
    provenance = _mapping(
        adjudication.get("provenance"),
        "Stage-B adjudication provenance",
    )
    _require(
        provenance.get("reducer_sha256") == EXPECTED_STAGE_B_REDUCER_SHA256
        and provenance.get("producer_sha256") == EXPECTED_LEGACY_PRODUCER_SHA256
        and provenance.get("producer_a_json_sha256") == EXPECTED_STAGE_B_A_JSON_SHA256
        and provenance.get("producer_b_json_sha256") == EXPECTED_STAGE_B_B_JSON_SHA256
        and provenance.get("producer_a_npz_sha256") == EXPECTED_STAGE_B_NPZ_SHA256
        and provenance.get("producer_b_npz_sha256") == EXPECTED_STAGE_B_NPZ_SHA256
        and provenance.get("target_input_manifest_sha256")
        == EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
        "Sealed Stage-B adjudication provenance differs",
    )

    _require(
        result_record.get("schema_version") == 2
        and result_record.get("artifact_kind")
        == "phase1_historical_k10000_stage_b_replay_result"
        and result_record.get("status") == "COMPLETED_EXACT_HISTORICAL_REPLAY_PASS",
        "Stage-B result record identity differs",
    )
    job = _mapping(result_record.get("job"), "Stage-B result job")
    outcome = _mapping(result_record.get("outcome"), "Stage-B result outcome")
    lanes = _mapping(result_record.get("lanes"), "Stage-B result lanes")
    sealed = _mapping(
        result_record.get("sealed_local_copy"),
        "Stage-B sealed copy",
    )
    license_record = _mapping(result_record.get("license"), "Stage-B license")
    _require(
        job.get("job_id") == 306814
        and job.get("state") == "COMPLETED"
        and job.get("exit_code") == "0:0"
        and outcome.get("categorical") == "EXACT_HISTORICAL_REPLAY_PASS"
        and outcome.get("adjudication_sha256") == EXPECTED_STAGE_B_ADJUDICATION_SHA256
        and outcome.get("adjudication_status")
        == "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
        and outcome.get("failure_count") == 0
        and lanes.get("lane_a_json_sha256") == EXPECTED_STAGE_B_A_JSON_SHA256
        and lanes.get("lane_b_json_sha256") == EXPECTED_STAGE_B_B_JSON_SHA256
        and lanes.get("lane_npz_sha256") == EXPECTED_STAGE_B_NPZ_SHA256
        and lanes.get("all_720_arrays_raw_byte_exact") is True
        and lanes.get("including_signed_zero") is True
        and sealed.get("tree_manifest_sha256") == EXPECTED_STAGE_B_TREE_MANIFEST_SHA256
        and sealed.get("local_remote_tree_digest_exact") is True
        and sealed.get("all_original_sidecars_verify") is True
        and license_record.get("status")
        == "ACTIVE_FOR_SEPARATE_PAIRED_CANONICAL_COMPARISON_ONLY"
        and result_record.get("independent_postrun_artifact_audit") == "GO"
        and result_record.get("independent_postrun_metrics_audit") == "GO",
        "Stage-B result record does not grant the paired-comparison license",
    )
    record_baseline = _mapping(
        result_record.get("corrected_frozen_legacy_baseline"),
        "Stage-B result baseline",
    )
    _require(
        record_baseline.get("licensed") is True
        and record_baseline.get("means") == FROZEN_BASELINE_MEANS
        and record_baseline.get("prospective_absolute_ceilings") == FROZEN_CEILINGS,
        "Stage-B result baseline differs",
    )


def _load_stage_b_license_artifacts(
    *,
    adjudication_path: Path,
    result_path: Path,
    tree_manifest_path: Path,
) -> dict[str, Any]:
    adjudication_payload, adjudication_sha256 = _verified_payload(
        adjudication_path,
        "Sealed Stage-B adjudication",
        expected_sha256=EXPECTED_STAGE_B_ADJUDICATION_SHA256,
    )
    result_payload, result_sha256 = _verified_payload(
        result_path,
        "Sealed Stage-B result record",
        expected_sha256=EXPECTED_STAGE_B_RESULT_SHA256,
    )
    tree_payload, tree_sha256 = _verified_payload(
        tree_manifest_path,
        "Sealed Stage-B tree manifest",
        expected_sha256=EXPECTED_STAGE_B_TREE_MANIFEST_SHA256,
    )
    adjudication = _strict_document(
        adjudication_payload,
        "Sealed Stage-B adjudication",
    )
    result_record = _strict_document(result_payload, "Sealed Stage-B result record")
    _parse_tree_manifest(tree_payload)
    _validate_stage_b_license(adjudication, result_record)
    return {
        "adjudication_sha256": adjudication_sha256,
        "result_sha256": result_sha256,
        "tree_manifest_sha256": tree_sha256,
    }


def _mismatch_names(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> list[str]:
    return [name for name in names if not _array_exact(left[name], right[name])]


def _derive_normalized_area_weights(
    boundary_points: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    points64 = np.asarray(boundary_points, dtype=np.float64)
    cells64 = np.asarray(cells, dtype=np.int64)
    _require(
        points64.ndim == 2
        and points64.shape[1:] == (3,)
        and cells64.shape == (RESOLUTION, 3)
        and bool(np.isfinite(points64).all())
        and bool(np.all(cells64 >= 0))
        and bool(np.all(cells64 < points64.shape[0])),
        "Stage-B triangle geometry is invalid",
    )
    vertices = points64[cells64]
    crosses = np.cross(
        vertices[:, 1] - vertices[:, 0],
        vertices[:, 2] - vertices[:, 0],
    )
    twice_areas = np.linalg.norm(crosses, axis=1)
    _require(
        bool(np.isfinite(twice_areas).all()) and bool(np.all(twice_areas > 0.0)),
        "Stage-B triangle geometry is degenerate",
    )
    total = float(np.sum(twice_areas, dtype=np.float64))
    _require(math.isfinite(total) and total > 0.0, "Stage-B total area is invalid")
    weights = np.ascontiguousarray(twice_areas / total)
    _require(
        abs(float(np.sum(weights, dtype=np.float64)) - 1.0) <= 1.0e-12,
        "Stage-B normalized area weights do not sum to one",
    )
    return weights


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    _require(
        prediction64.shape == truth64.shape
        and bool(np.isfinite(prediction64).all())
        and bool(np.isfinite(truth64).all()),
        "Uniform relative-L2 inputs are invalid",
    )
    result = float(np.linalg.norm(prediction64 - truth64)) / (
        float(np.linalg.norm(truth64)) + 1.0e-8
    )
    _require(math.isfinite(result), "Uniform relative L2 is non-finite")
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
        prediction64.shape == truth64.shape
        and weights.shape == (RESOLUTION,)
        and bool(np.isfinite(prediction64).all())
        and bool(np.isfinite(truth64).all())
        and bool(np.isfinite(weights).all())
        and bool(np.all(weights > 0.0))
        and abs(float(np.sum(weights, dtype=np.float64)) - 1.0) <= 1.0e-12,
        "Weighted relative-L2 inputs are invalid",
    )
    expanded = weights[:, None] if prediction64.ndim == 2 else weights
    numerator = math.sqrt(
        float(
            np.sum(
                expanded * (prediction64 - truth64) ** 2,
                dtype=np.float64,
            )
        )
    )
    denominator = (
        math.sqrt(float(np.sum(expanded * truth64**2, dtype=np.float64))) + 1.0e-8
    )
    result = numerator / denominator
    _require(math.isfinite(result), "Weighted relative L2 is non-finite")
    return result


def _historical_pointwise_wss(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    _require(
        prediction64.shape == truth64.shape
        and prediction64.ndim == 2
        and prediction64.shape[1] == 3,
        "Pointwise WSS inputs are invalid",
    )
    numerator = np.linalg.norm(prediction64 - truth64, axis=-1)
    denominator = np.linalg.norm(truth64, axis=-1) + 1.0e-8
    result = float(np.mean(numerator / denominator, dtype=np.float64))
    _require(math.isfinite(result), "Pointwise WSS metric is non-finite")
    return result


def _case_metrics(
    *,
    prediction_pressure: np.ndarray,
    prediction_wss: np.ndarray,
    truth_pressure: np.ndarray,
    truth_wss: np.ndarray,
    normalized_weights: np.ndarray,
) -> dict[str, float]:
    return {
        "uniform_pressure_relative_l2": _relative_l2(
            prediction_pressure,
            truth_pressure,
        ),
        "uniform_wss_frobenius_relative_l2": _relative_l2(
            prediction_wss,
            truth_wss,
        ),
        "archive_normalized_area_weighted_pressure_relative_l2": (
            _weighted_relative_l2(
                prediction_pressure,
                truth_pressure,
                normalized_weights,
            )
        ),
        "archive_normalized_area_weighted_wss_frobenius_relative_l2": (
            _weighted_relative_l2(
                prediction_wss,
                truth_wss,
                normalized_weights,
            )
        ),
        DESCRIPTIVE_WSS_METRIC: _historical_pointwise_wss(
            prediction_wss,
            truth_wss,
        ),
    }


def _reconstruct_canonical_geometry(
    raw_points: np.ndarray,
    cells: np.ndarray,
) -> dict[str, np.ndarray]:
    """Independently repeat the frozen float64 one-cast geometry definition."""

    points = torch.from_numpy(np.ascontiguousarray(raw_points)).to(torch.float64)
    topology = torch.from_numpy(np.ascontiguousarray(cells)).to(torch.long)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and topology.shape == (RESOLUTION, 3)
        and bool(torch.isfinite(points).all())
        and int(topology.min().item()) >= 0
        and int(topology.max().item()) < points.shape[0],
        "Raw canonical geometry is invalid",
    )
    vertices = points[topology]
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = torch.linalg.cross(edge_1, edge_2)
    twice_area = torch.linalg.vector_norm(cross, dim=-1)
    _require(
        bool(torch.isfinite(twice_area).all()) and bool(torch.all(twice_area > 0.0)),
        "Raw canonical geometry contains a degenerate triangle",
    )
    centroids = vertices.mean(dim=1)
    areas = 0.5 * twice_area
    total_area = areas.sum()
    center = torch.einsum("n,nd->d", areas, centroids) / total_area
    return {
        "canonical_cells_int64": np.ascontiguousarray(
            topology.numpy().astype("<i8", copy=False)
        ),
        "canonical_points_float32": np.ascontiguousarray(
            ((points - center) / CANONICAL_LENGTH_SCALE)
            .to(torch.float32)
            .numpy()
            .astype("<f4", copy=False)
        ),
        "canonical_centroids_float32": np.ascontiguousarray(
            ((centroids - center) / CANONICAL_LENGTH_SCALE)
            .to(torch.float32)
            .numpy()
            .astype("<f4", copy=False)
        ),
        "canonical_areas_float32": np.ascontiguousarray(
            (areas / (CANONICAL_LENGTH_SCALE**2))
            .to(torch.float32)
            .numpy()
            .astype("<f4", copy=False)
        ),
        "canonical_normals_float32": np.ascontiguousarray(
            (cross / twice_area[:, None])
            .to(torch.float32)
            .numpy()
            .astype("<f4", copy=False)
        ),
        "canonical_physical_center_float64": np.ascontiguousarray(
            center.numpy().astype("<f8", copy=False)
        ),
    }


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    _require(left64.shape == right64.shape, "Difference shape differs")
    difference = np.abs(left64 - right64)
    result = 0.0 if difference.size == 0 else float(np.max(difference))
    _require(math.isfinite(result), "Difference is non-finite")
    return result


def _training_physical_prediction_control(
    *,
    pressure_training: np.ndarray,
    pressure_physical: np.ndarray,
    wss_training: np.ndarray,
    wss_physical: np.ndarray,
    globals_array: np.ndarray,
) -> dict[str, Any]:
    q_inf, p_inf = stage_b._archive_freestream_scales(globals_array)
    reconstructed_pressure = (
        np.asarray(pressure_physical, dtype=np.float64) - p_inf
    ) / q_inf
    reconstructed_wss = (
        np.asarray(wss_physical, dtype=np.float64)
        / q_inf
        / (WSS_NORMALIZATION_STD + NORMALIZATION_EPSILON)
    )
    pressure_difference = _maximum_absolute_difference(
        pressure_training,
        reconstructed_pressure,
    )
    wss_difference = _maximum_absolute_difference(
        wss_training,
        reconstructed_wss,
    )
    return {
        "pressure_maximum_absolute_difference": pressure_difference,
        "wss_maximum_absolute_difference": wss_difference,
        "absolute_tolerance": TRAINING_PHYSICAL_ABS_TOLERANCE,
        "passed": (
            pressure_difference <= TRAINING_PHYSICAL_ABS_TOLERANCE
            and wss_difference <= TRAINING_PHYSICAL_ABS_TOLERANCE
        ),
    }


def _adjudicate_case(
    *,
    spec: tuple[int, str, int, int, int],
    canonical_arrays: Mapping[str, np.ndarray],
    stage_b_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    ordinal, case_id, reader_index, _, start = spec
    prefix = _case_prefix(ordinal, case_id)
    canonical = {
        suffix: canonical_arrays[f"{prefix}__{suffix}"]
        for suffix in CANONICAL_ARRAY_SUFFIXES
    }
    legacy = {
        suffix: stage_b_arrays[f"{prefix}__{suffix}"]
        for suffix in stage_b.ARRAY_SCHEMAS
    }
    shared_mismatches = [
        suffix
        for suffix in LEGACY_SHARED_CONTROL_SUFFIXES
        if not _array_exact(canonical[suffix], legacy[suffix])
    ]
    reconstructed = _reconstruct_canonical_geometry(
        canonical["raw_points_float32"],
        canonical["compacted_cells_int64"],
    )
    canonical_geometry_mismatches = [
        suffix
        for suffix, expected in reconstructed.items()
        if not _array_exact(canonical[suffix], expected)
    ]
    queries_exact = _array_exact(
        canonical["canonical_queries_float32"],
        canonical["canonical_centroids_float32"],
    )
    inverse_queries = np.asarray(
        canonical["canonical_queries_float32"], dtype=np.float64
    ) * CANONICAL_LENGTH_SCALE + np.asarray(
        canonical["canonical_physical_center_float64"],
        dtype=np.float64,
    )
    inverse_query_max_abs = _maximum_absolute_difference(
        inverse_queries,
        canonical["raw_centroids_float32"],
    )
    pipeline_raw_points = np.asarray(
        canonical["pipeline_boundary_points_float32"],
        dtype=np.float64,
    ) * PHYSICAL_LENGTH + np.asarray(
        canonical["pipeline_center_float32"], dtype=np.float64
    )
    pipeline_raw_point_max_abs = _maximum_absolute_difference(
        pipeline_raw_points,
        canonical["raw_points_float32"],
    )
    physical_chain = _training_physical_prediction_control(
        pressure_training=canonical["prediction_pressure_training_float32"],
        pressure_physical=canonical["prediction_pressure_physical_float32"],
        wss_training=canonical["prediction_wss_training_float32"],
        wss_physical=canonical["prediction_wss_physical_float32"],
        globals_array=legacy["pipeline_globals_float32"],
    )
    controls_passed = bool(
        not shared_mismatches
        and not canonical_geometry_mismatches
        and queries_exact
        and inverse_query_max_abs <= CANONICAL_PHYSICAL_INVERSE_ABS_TOLERANCE
        and pipeline_raw_point_max_abs <= CANONICAL_PHYSICAL_INVERSE_ABS_TOLERANCE
        and physical_chain["passed"]
    )

    weights = _derive_normalized_area_weights(
        legacy["pipeline_boundary_points_float32"],
        legacy["compacted_cells_int64"],
    )
    truth_pressure = legacy["truth_pressure_training_float32"]
    truth_wss = legacy["truth_wss_training_float32"]
    legacy_metrics = _case_metrics(
        prediction_pressure=legacy["prediction_pressure_training_float32"],
        prediction_wss=legacy["prediction_wss_training_float32"],
        truth_pressure=truth_pressure,
        truth_wss=truth_wss,
        normalized_weights=weights,
    )
    canonical_metrics = _case_metrics(
        prediction_pressure=canonical["prediction_pressure_training_float32"],
        prediction_wss=canonical["prediction_wss_training_float32"],
        truth_pressure=truth_pressure,
        truth_wss=truth_wss,
        normalized_weights=weights,
    )
    return {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": reader_index,
        "historical_start": start,
        "validity_controls": {
            "shared_precanonical_inputs_raw_byte_exact": not shared_mismatches,
            "shared_precanonical_input_mismatches": shared_mismatches,
            "canonical_geometry_independently_reconstructed_raw_byte_exact": (
                not canonical_geometry_mismatches
            ),
            "canonical_geometry_mismatches": canonical_geometry_mismatches,
            "canonical_queries_equal_centroids_raw_byte_exact": queries_exact,
            "canonical_query_inverse_maximum_absolute_difference": (
                inverse_query_max_abs
            ),
            "pipeline_raw_point_inverse_maximum_absolute_difference": (
                pipeline_raw_point_max_abs
            ),
            "physical_inverse_absolute_tolerance": (
                CANONICAL_PHYSICAL_INVERSE_ABS_TOLERANCE
            ),
            "canonical_prediction_training_physical_chain": physical_chain,
            "passed": controls_passed,
        },
        "normalized_area_weight_sum": float(np.sum(weights, dtype=np.float64)),
        "legacy_metrics": legacy_metrics,
        "canonical_metrics": canonical_metrics,
        "canonical_minus_legacy_metric_deltas": {
            name: canonical_metrics[name] - legacy_metrics[name]
            for name in DECIDING_METRICS
        },
        "per_case_metrics_deciding": False,
        "casewise_metric_deltas_deciding": False,
    }


def _cohort_means(
    cases: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, float]:
    names = (*DECIDING_METRICS, DESCRIPTIVE_WSS_METRIC)
    return {
        name: float(
            np.mean(
                [case[f"{arm}_metrics"][name] for case in cases],
                dtype=np.float64,
            )
        )
        for name in names
    }


def _classify_canonical_means(
    canonical_means: Mapping[str, float],
) -> tuple[str, dict[str, dict[str, Any]]]:
    gates = {
        name: {
            "canonical_mean": float(canonical_means[name]),
            "inclusive_ceiling": ceiling,
            "passed": float(canonical_means[name]) <= ceiling,
        }
        for name, ceiling in FROZEN_CEILINGS.items()
    }
    outcome = (
        NONINFERIORITY_SUCCESS_OUTCOME
        if all(record["passed"] for record in gates.values())
        else VALID_REFUTATION
    )
    return outcome, gates


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
    canonical_producer: Path,
    canonical_a_json: Path,
    canonical_a_npz: Path,
    canonical_b_json: Path,
    canonical_b_npz: Path,
    legacy_producer: Path,
    fresh_legacy_sentinel_json: Path,
    fresh_legacy_sentinel_npz: Path,
    sealed_stage_b_a_json: Path,
    sealed_stage_b_a_npz: Path,
    sealed_stage_b_b_json: Path,
    sealed_stage_b_b_npz: Path,
    sealed_stage_b_adjudication: Path,
    sealed_stage_b_result: Path,
    sealed_stage_b_tree_manifest: Path,
) -> dict[str, Any]:
    """Return a categorical result without publishing it."""

    try:
        imported_stage_b_reducer_sha256 = _read_source(
            Path(stage_b.__file__),
            "Imported Stage-B reducer source",
            EXPECTED_STAGE_B_REDUCER_SHA256,
        )
        canonical_producer_sha256 = _read_source(
            canonical_producer,
            "Canonical producer source",
            EXPECTED_CANONICAL_PRODUCER_SHA256,
        )
        legacy_producer_sha256 = _read_source(
            legacy_producer,
            "Legacy producer source",
            EXPECTED_LEGACY_PRODUCER_SHA256,
        )
        license_provenance = _load_stage_b_license_artifacts(
            adjudication_path=sealed_stage_b_adjudication,
            result_path=sealed_stage_b_result,
            tree_manifest_path=sealed_stage_b_tree_manifest,
        )
        sealed_a, sealed_arrays_a = _load_sealed_stage_b_lane(
            sealed_stage_b_a_json,
            sealed_stage_b_a_npz,
            "A",
        )
        sealed_b, sealed_arrays_b = _load_sealed_stage_b_lane(
            sealed_stage_b_b_json,
            sealed_stage_b_b_npz,
            "B",
        )
        legacy_names = _legacy_array_names()
        sealed_replica_mismatches = _mismatch_names(
            sealed_arrays_a,
            sealed_arrays_b,
            legacy_names,
        )
        _require(
            not sealed_replica_mismatches,
            "Sealed Stage-B lane arrays are not raw-byte exact",
        )
        target_records = _target_records_from_sealed(sealed_arrays_a)
        sentinel, sentinel_arrays = _load_fresh_legacy_sentinel(
            fresh_legacy_sentinel_json,
            fresh_legacy_sentinel_npz,
            target_records,
        )
        sentinel_mismatches_a = _mismatch_names(
            sentinel_arrays,
            sealed_arrays_a,
            legacy_names,
        )
        sentinel_mismatches_b = _mismatch_names(
            sentinel_arrays,
            sealed_arrays_b,
            legacy_names,
        )
        _require(
            not sentinel_mismatches_a and not sentinel_mismatches_b,
            "Fresh legacy sentinel differs from sealed Stage-B arrays",
        )

        canonical_a, canonical_arrays_a = _load_canonical_lane(
            canonical_a_json,
            canonical_a_npz,
            "A",
        )
        canonical_b, canonical_arrays_b = _load_canonical_lane(
            canonical_b_json,
            canonical_b_npz,
            "B",
        )
        canonical_names = _canonical_array_names()
        canonical_replica_mismatches = _mismatch_names(
            canonical_arrays_a,
            canonical_arrays_b,
            canonical_names,
        )
        _require(
            not canonical_replica_mismatches,
            "Canonical A/B arrays are not raw-byte exact including signed zero",
        )
        process_a = _mapping(
            canonical_a["document"].get("provenance", {}).get("process"),
            "Canonical A process provenance",
        )
        process_b = _mapping(
            canonical_b["document"].get("provenance", {}).get("process"),
            "Canonical B process provenance",
        )
        _require(
            type(process_a.get("pid")) is int
            and process_a.get("pid") > 0
            and type(process_b.get("pid")) is int
            and process_b.get("pid") > 0
            and process_a.get("pid") != process_b.get("pid")
            and process_a.get("hostname")
            and process_a.get("hostname") == process_b.get("hostname")
            and process_a.get("cuda_visible_devices_token") == "0"
            and process_b.get("cuda_visible_devices_token") == "1"
            and process_a.get("rank_environment") == _EXPECTED_RANK_ENVIRONMENT
            and process_b.get("rank_environment") == _EXPECTED_RANK_ENVIRONMENT
            and process_a.get("slurm_job_id")
            and process_a.get("slurm_job_id") == process_b.get("slurm_job_id"),
            "Canonical A/B process-isolation provenance differs",
        )

        cases = [
            _adjudicate_case(
                spec=spec,
                canonical_arrays=canonical_arrays_a,
                stage_b_arrays=sealed_arrays_a,
            )
            for spec in CASE_SPECS
        ]
        _require(
            len(cases) == CASE_COUNT
            and all(case["validity_controls"]["passed"] for case in cases),
            "Canonical input, frame, or prediction-chain validity control failed",
        )
        legacy_means = _cohort_means(cases, "legacy")
        canonical_means = _cohort_means(cases, "canonical")
        baseline_differences = {
            name: abs(legacy_means[name] - frozen)
            for name, frozen in FROZEN_BASELINE_MEANS.items()
        }
        _require(
            all(
                difference <= BASELINE_RECOMPUTE_ABS_TOLERANCE
                for difference in baseline_differences.values()
            ),
            "Independently recomputed Stage-B baseline differs from the license",
        )
    except ArtifactUnavailable as failure:
        return _base_result(
            status=INCOMPLETE_STATUS,
            outcome=INCOMPLETE_COMPARISON,
            failures=[{"kind": "unavailable_artifact", "message": str(failure)}],
        )
    except ArtifactInvalid as failure:
        return _base_result(
            status=INVALID_STATUS,
            outcome=INVALID_COMPARISON,
            failures=[
                {
                    "kind": "invalid_artifact_or_control",
                    "message": str(failure),
                }
            ],
        )

    outcome, metric_gates = _classify_canonical_means(canonical_means)
    result = _base_result(status=VALID_STATUS, outcome=outcome, failures=[])
    result.update(
        {
            "decision_gates": {
                "stage_b_exact_license_active": True,
                "sealed_stage_b_replicas_raw_byte_exact": True,
                "fresh_legacy_sentinel_exact_to_both_sealed_lanes": True,
                "canonical_replicas_all_arrays_raw_byte_exact_including_signed_zero": (
                    True
                ),
                "canonical_processes_distinct_and_gpu_pinned": True,
                "all_precanonical_input_and_frame_controls_passed": True,
                "stage_b_baseline_independently_recomputed": True,
                "four_inclusive_noninferiority_endpoints": metric_gates,
                "all_four_noninferiority_endpoints_passed": (
                    outcome == NONINFERIORITY_SUCCESS_OUTCOME
                ),
                "per_case_deciding_gates_present": False,
            },
            "frozen_legacy_baseline": {
                "licensed_means": FROZEN_BASELINE_MEANS,
                "independently_recomputed_means": {
                    name: legacy_means[name] for name in DECIDING_METRICS
                },
                "absolute_recompute_differences": baseline_differences,
                "recompute_absolute_tolerance": (BASELINE_RECOMPUTE_ABS_TOLERANCE),
                "prospective_absolute_ceilings": FROZEN_CEILINGS,
            },
            "canonical_arm": {
                "cohort_means": {
                    name: canonical_means[name] for name in DECIDING_METRICS
                },
                "descriptive_pointwise_wss_mean": canonical_means[
                    DESCRIPTIVE_WSS_METRIC
                ],
                "descriptive_pointwise_wss_deciding": False,
                "endpoint_gates": metric_gates,
            },
            "confirmed_nonlicensing_diagnostic": {
                "name": DESCRIPTIVE_WSS_METRIC,
                "legacy_recomputed_mean": legacy_means[DESCRIPTIVE_WSS_METRIC],
                "frozen_historical_value": FROZEN_DESCRIPTIVE_WSS_MEAN,
                "status": "confirmed wrong WSS reduction; descriptive only",
                "deciding": False,
            },
            "cases": cases,
            "provenance": {
                "reducer_path": str(Path(__file__).resolve()),
                "reducer_sha256": _sha256_bytes(
                    stage_b._read_regular_file_bytes(
                        Path(__file__).resolve(),
                        "Paired adjudicator source",
                    )[0]
                ),
                "imported_stage_b_reducer_sha256": (imported_stage_b_reducer_sha256),
                "canonical_producer_sha256": canonical_producer_sha256,
                "legacy_producer_sha256": legacy_producer_sha256,
                "canonical_a_json_sha256": canonical_a["json_sha256"],
                "canonical_a_npz_sha256": canonical_a["npz_sha256"],
                "canonical_b_json_sha256": canonical_b["json_sha256"],
                "canonical_b_npz_sha256": canonical_b["npz_sha256"],
                "fresh_legacy_sentinel_json_sha256": sentinel["json_sha256"],
                "fresh_legacy_sentinel_npz_sha256": sentinel["npz_sha256"],
                "sealed_stage_b_a_json_sha256": sealed_a["json_sha256"],
                "sealed_stage_b_a_npz_sha256": sealed_a["npz_sha256"],
                "sealed_stage_b_b_json_sha256": sealed_b["json_sha256"],
                "sealed_stage_b_b_npz_sha256": sealed_b["npz_sha256"],
                **license_provenance,
            },
            "scientific_scope": {
                "supports": (
                    (
                        "canonical arm meets all four frozen 1.02 ceilings on "
                        "the fixed 36-case epoch-491 K=10000 ID-reference cohort"
                    )
                    if outcome == NONINFERIORITY_SUCCESS_OUTCOME
                    else (
                        "the fixed-cohort paired experiment validly refutes "
                        "canonical noninferiority on at least one frozen endpoint"
                    )
                ),
                "does_not_support": [
                    "superiority",
                    "other resolutions",
                    "OOD or population generalization",
                    "training-time invariance",
                    "causal architecture or H-QC mechanism claims",
                    "absolute native areas or forces",
                ],
            },
            "execution_control_limitation": (
                "the fresh legacy sentinel producer does not persist PID/GPU/job "
                "fields; its separate setsid process tree, CUDA token, log, and "
                "heartbeat are attested by the frozen wrapper"
            ),
        }
    )
    return result


def _atomic_publish(path: Path, payload: bytes) -> str:
    return stage_b._atomic_publish(path, payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-producer", type=Path, required=True)
    parser.add_argument("--canonical-a-json", type=Path, required=True)
    parser.add_argument("--canonical-a-npz", type=Path, required=True)
    parser.add_argument("--canonical-b-json", type=Path, required=True)
    parser.add_argument("--canonical-b-npz", type=Path, required=True)
    parser.add_argument("--legacy-producer", type=Path, required=True)
    parser.add_argument(
        "--fresh-legacy-sentinel-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-legacy-sentinel-npz",
        type=Path,
        required=True,
    )
    parser.add_argument("--sealed-stage-b-a-json", type=Path, required=True)
    parser.add_argument("--sealed-stage-b-a-npz", type=Path, required=True)
    parser.add_argument("--sealed-stage-b-b-json", type=Path, required=True)
    parser.add_argument("--sealed-stage-b-b-npz", type=Path, required=True)
    parser.add_argument(
        "--sealed-stage-b-adjudication",
        type=Path,
        required=True,
    )
    parser.add_argument("--sealed-stage-b-result", type=Path, required=True)
    parser.add_argument(
        "--sealed-stage-b-tree-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = adjudicate(
        canonical_producer=args.canonical_producer,
        canonical_a_json=args.canonical_a_json,
        canonical_a_npz=args.canonical_a_npz,
        canonical_b_json=args.canonical_b_json,
        canonical_b_npz=args.canonical_b_npz,
        legacy_producer=args.legacy_producer,
        fresh_legacy_sentinel_json=args.fresh_legacy_sentinel_json,
        fresh_legacy_sentinel_npz=args.fresh_legacy_sentinel_npz,
        sealed_stage_b_a_json=args.sealed_stage_b_a_json,
        sealed_stage_b_a_npz=args.sealed_stage_b_a_npz,
        sealed_stage_b_b_json=args.sealed_stage_b_b_json,
        sealed_stage_b_b_npz=args.sealed_stage_b_b_npz,
        sealed_stage_b_adjudication=args.sealed_stage_b_adjudication,
        sealed_stage_b_result=args.sealed_stage_b_result,
        sealed_stage_b_tree_manifest=args.sealed_stage_b_tree_manifest,
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
    if result["decision_outcome"] == INVALID_COMPARISON:
        return 2
    if result["decision_outcome"] == INCOMPLETE_COMPARISON:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
