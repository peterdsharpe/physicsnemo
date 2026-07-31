# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Apply the pre-registered 36-case fixed-query H-QC decision rule.

This is a pure, fail-closed reducer.  It accepts one or more completed
producer-lane JSON artifacts, validates their frozen identities and integrity,
and publishes one verdict without replacing prior evidence.  It never runs the
model and it never chooses between the two predeclared centering paths.

Run from the repository root after every producer lane is present::

    python3 examples/cfd/mesh_transformer/studies/phase1_hqc_verdict.py \
      /path/to/phase1_hqc_lane_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import struct
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_DATE = "2026-07-27"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_ARTIFACT_KIND = "phase1_hqc_producer_lane"
EXPECTED_PRODUCER_STATUS = "PASSED_HQC_PRODUCER_LANE"

RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
BASELINE_K = 10_000
FIXED_QUERY_K = 2_500
ENDPOINTS = (2_500, 40_000)

EXPECTED_CASES = (
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
EXPECTED_READER_INDICES = dict(
    zip(
        EXPECTED_CASES,
        (
            21,
            33,
            51,
            55,
            77,
            79,
            88,
            92,
            107,
            114,
            136,
            185,
            186,
            212,
            221,
            237,
            285,
            298,
            300,
            318,
            319,
            329,
            340,
            346,
            351,
            354,
            362,
            391,
            394,
            395,
            404,
            416,
            418,
            423,
            453,
            469,
        ),
        strict=True,
    )
)
EXPECTED_MASTER_CELLS = dict(
    zip(
        EXPECTED_CASES,
        (
            17_504_739,
            16_380_547,
            15_789_064,
            18_007_064,
            19_404_150,
            18_792_923,
            14_634_570,
            14_932_664,
            18_934_869,
            17_796_743,
            15_024_109,
            18_857_430,
            16_922_213,
            15_063_884,
            18_022_481,
            16_199_351,
            18_958_141,
            19_519_305,
            16_887_630,
            16_222_090,
            16_294_644,
            16_591_548,
            14_561_784,
            16_588_938,
            17_738_132,
            15_747_949,
            17_809_120,
            16_443_085,
            18_343_677,
            19_780_049,
            16_648_431,
            16_063_459,
            17_847_065,
            15_715_663,
            16_516_082,
            17_188_261,
        ),
        strict=True,
    )
)
EXPECTED_HISTORICAL_STARTS = dict(
    zip(
        EXPECTED_CASES,
        (
            14_045_027,
            14_700_754,
            9_195_926,
            4_452_828,
            6_369_582,
            1_320_415,
            10_215_595,
            7_635_018,
            16_494_923,
            15_267_620,
            3_789_927,
            10_967_997,
            5_453_831,
            4_943_208,
            16_998_850,
            15_062_581,
            5_352_845,
            11_721_918,
            11_083_431,
            15_155_572,
            13_228_777,
            1_346_462,
            12_777_694,
            13_358_519,
            365_298,
            1_091_720,
            8_840_407,
            11_669_428,
            15_504_945,
            19_757_508,
            16_079_300,
            6_463_342,
            191_824,
            11_592_670,
            2_240_523,
            4_374_650,
        ),
        strict=True,
    )
)
ARCHIVED_UNIFORM_PRESSURE_BY_CASE = dict(
    zip(
        EXPECTED_CASES,
        (
            0.15937316417694092,
            0.18477365374565125,
            0.15804672241210938,
            0.15877088904380798,
            0.15290716290473938,
            0.18428756296634674,
            0.1887969821691513,
            0.15883244574069977,
            0.15645363926887512,
            0.17767199873924255,
            0.1639585644006729,
            0.16175809502601624,
            0.1604095846414566,
            0.14261627197265625,
            0.15814006328582764,
            0.1746961772441864,
            0.1645786166191101,
            0.15864448249340057,
            0.19035394489765167,
            0.16157667338848114,
            0.153878316283226,
            0.16156354546546936,
            0.1703338921070099,
            0.15645885467529297,
            0.15831957757472992,
            0.18941590189933777,
            0.17977391183376312,
            0.16812270879745483,
            0.17494602501392365,
            0.187981516122818,
            0.14168570935726166,
            0.1817486435174942,
            0.1773087978363037,
            0.18593141436576843,
            0.1548524796962738,
            0.15890362858772278,
        ),
        strict=True,
    )
)

ARCHIVED_UNIFORM_PRESSURE_MEAN = 0.16716310713026258
ARCHIVED_MEAN_ABS_TOLERANCE = 5.0e-6
ARCHIVED_CASE_ABS_TOLERANCE = 1.0e-3
RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE = 1.0e-6
CENTER_METRIC_RELATIVE_TOLERANCE = 1.0e-3
ARCHIVED_PIPELINE_NORMAL_ABS_TOLERANCE = 2.0e-6
PIPELINE_NORMAL_GEOMETRY_ABS_TOLERANCE = 5.0e-7
PIPELINE_NORMAL_UNIT_ABS_TOLERANCE = 5.0e-6
BASELINE_COMPARABILITY_BOUNDS = (0.5, 2.0)
CLIFF_RATIO_THRESHOLD = 2.0
CLIFF_CASE_COUNT_THRESHOLD = 24
SUPPORT_ATTENUATION_FRACTION = 0.25
SUPPORT_FIXED_RATIO_THRESHOLD = 1.25
SUPPORT_FAVORABLE_COUNT_THRESHOLD = 27
FUTILITY_RETENTION_FRACTION = 0.5
FUTILITY_FIXED_40K_RATIO_THRESHOLD = 2.0
AREA_NEARLY_FLAT_RATIO_THRESHOLD = 1.25
EXPECTED_READER_SEED_FORK_CHAIN_SHA256 = (
    "210ef25300498d42294774cd5ea8f04dd002c94eab25a9e54f17078840faa3b1"
)

# Frozen only after the corrected schema-v2 producer's focused suite,
# formatting, and source audit.
EXPECTED_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_PRODUCER_CONFIG_SHA256 = (
    "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
)

METRIC_DEFINITIONS = {
    "pressure_relative_l2": (
        "sqrt(sum((prediction-truth)^2))/(sqrt(sum(truth^2))+1e-8)"
    ),
    "signed_centered_correlation": (
        "clamp(cos,-1,1); cos=dot(pred-mean(pred),truth-mean(truth))/"
        "(norm(pred-mean(pred))*norm(truth-mean(truth)))"
    ),
    "positive_gain_pattern_error": ("sqrt(1-max(signed_centered_correlation,0)^2)"),
    "amplitude_ratio": "sqrt(mean(prediction^2))/sqrt(mean(truth^2))",
    "wss_frobenius_relative_l2": ("norm(prediction-truth,F)/(norm(truth,F)+1e-8)"),
    "wss_normal_energy": (
        "sqrt(sum((prediction_i dot native_unit_normal_i)^2))/(norm(prediction,F)+1e-8)"
    ),
    "scaled_subset_pressure_force_relative_error": (
        "norm((N/m)*sum(A_i*prediction_i*n_i)-(N/m)*sum(A_i*truth_i*n_i))/"
        "(norm((N/m)*sum(A_i*truth_i*n_i))+1e-12)"
    ),
    "area_weighted_pressure_relative_l2": (
        "sqrt(sum(w_i*(prediction_i-truth_i)^2))/"
        "(sqrt(sum(w_i*truth_i^2))+1e-8); w_i=A_i/sum(A)"
    ),
    "center_prediction_relative_l2_difference": (
        "norm(primary-fixed_center)/max(norm(primary),norm(fixed_center),1e-8)"
    ),
    "center_metric_relative_change": (
        "abs(primary-fixed_center)/max(abs(primary),abs(fixed_center),1e-12)"
    ),
}
HISTORICAL_SIGNATURE_ALGORITHM = (
    "sha256(canonical-json mapping of: ordered raw-cell-id SHA256; ordered "
    "compacted-connectivity SHA256; raw-master training-pressure, normalized-WSS, "
    "native-normal, and native-area component SHA256 values; and SHA256 of "
    "canonical raw reconstructed nondimensional query points float32 after "
    "subtracting query[0]); the archived signature repeats this canonical raw "
    "identity only after independently gating saved compact connectivity exactly "
    "and saved-vs-reconstructed unquantized coordinates <=1e-6, pressure <=1e-3, "
    "WSS <=1e-5, and saved-vs-current pipeline normals <=2e-6. Independent "
    "rounded-coordinate hash equality is deliberately "
    "not used because discontinuous bin boundaries failed on a validated 8.94e-8 "
    "coordinate perturbation."
)

FROZEN_CONTRACT = {
    "hypothesis_id": "H-QC",
    "cohort_name": "id_reference",
    "cohort_case_ids": list(EXPECTED_CASES),
    "manifest_sha256": (
        "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
    ),
    "historical_metrics_sha256": (
        "423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a"
    ),
    "resolved_config_sha256": (
        "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
    ),
    "dataset_config_sha256": (
        "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
    ),
    "execution_source_tree_manifest_sha256": (
        "fa6a7b683fa9aa02e4537ef69e8e977906df7c9fa6964cb759edfcee8d7b90cd"
    ),
    "run_id": "t2_mesh_transformer_surface_flagship_seed42",
    "epoch": 491,
    "model_filename": "MeshTransformer.0.491.mdlus",
    "model_sha256": (
        "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
    ),
    "training_state_filename": "checkpoint.0.491.pt",
    "training_state_sha256": (
        "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
    ),
    "norm_stats_filename": "norm_stats.pt",
    "norm_stats_sha256": (
        "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
    ),
    "reader_seed": 42,
    "reader_generator_seed": 45,
    "resolutions": list(RESOLUTIONS),
    "baseline_k": BASELINE_K,
    "fixed_query_k": FIXED_QUERY_K,
    "source_selection": "nested_cyclic_prefix_from_historical_10000_start",
    "coupled_query": "S_k",
    "fixed_query": "Q=S_2500",
    "center_primary": "per_resolution_compacted_vertex_centroid",
    "center_use_area_weighting": False,
    "center_diagnostic": "fixed_S_10000_compacted_vertex_centroid",
    "trace_model": True,
    "own_cell_typed_readout": True,
    "self_correction": 0.5,
    "inference_compile": False,
    "precision": "bfloat16",
    "parameter_count": 1_278_268,
    "archived_uniform_pressure_mean": ARCHIVED_UNIFORM_PRESSURE_MEAN,
    "archived_mean_abs_tolerance": ARCHIVED_MEAN_ABS_TOLERANCE,
    "archived_case_abs_tolerance": ARCHIVED_CASE_ABS_TOLERANCE,
    "raw_frame_reconstruction_abs_tolerance": (RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE),
    "center_metric_relative_tolerance": CENTER_METRIC_RELATIVE_TOLERANCE,
    "archived_pipeline_normal_abs_tolerance": (ARCHIVED_PIPELINE_NORMAL_ABS_TOLERANCE),
    "pipeline_normal_geometry_abs_tolerance": (PIPELINE_NORMAL_GEOMETRY_ABS_TOLERANCE),
    "pipeline_normal_unit_abs_tolerance": PIPELINE_NORMAL_UNIT_ABS_TOLERANCE,
    "historical_signature_algorithm": HISTORICAL_SIGNATURE_ALGORITHM,
    "metric_definitions": METRIC_DEFINITIONS,
}

FULL_METRIC_KEYS = (
    "pressure_relative_l2",
    "signed_centered_correlation",
    "positive_gain_pattern_error",
    "amplitude_ratio",
    "wss_frobenius_relative_l2",
    "wss_normal_energy",
    "scaled_subset_pressure_force_relative_error",
)
CENTER_DIAGNOSTIC_KEYS = (
    "pressure_prediction_relative_l2_difference",
    "uniform_pressure_error_relative_change",
    "area_pressure_error_relative_change",
)
NPZ_FIELDS = (
    "raw_cell_ids_int64",
    "compacted_cells_int64",
    "raw_centroids_float32",
    "native_normals_float32",
    "native_areas_float64",
    "truth_pressure_float32",
    "truth_wss_float32",
    "primary_query_points_float32",
    "primary_pipeline_normals_float32",
    "primary_pressure_float32",
    "primary_wss_float32",
    "fixed_center_query_points_float32",
    "fixed_center_pipeline_normals_float32",
    "fixed_center_pressure_float32",
    "fixed_center_wss_float32",
)
METRIC_RECOMPUTATION_REL_TOLERANCE = 1.0e-10
METRIC_RECOMPUTATION_ABS_TOLERANCE = 1.0e-12
PIPELINE_NORMAL_DIAGNOSTIC_RECOMPUTATION_ABS_TOLERANCE = 5.0e-7
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class VerdictInputError(ValueError):
    """An input artifact violates the frozen H-QC protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one JSON artifact without replacing prior evidence."""

    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_output_path(output: Path, inputs: list[Path]) -> None:
    output_resolved = output.resolve()
    for input_path in inputs:
        for evidence_path in (input_path, input_path.with_suffix(".npz")):
            if output_resolved == evidence_path.resolve():
                raise VerdictInputError("H-QC output path must not alias an input")
            if (
                output.exists()
                and evidence_path.exists()
                and os.path.samefile(output, evidence_path)
            ):
                raise VerdictInputError("H-QC output path must not alias an input")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerdictInputError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerdictInputError(f"{context} must be a JSON array")
    return value


def _exact_keys(value: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise VerdictInputError(
            f"{context} has wrong keys; missing={missing}, unexpected={unexpected}"
        )


def _finite_number(
    value: Any,
    context: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerdictInputError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VerdictInputError(f"{context} must be finite")
    if nonnegative and result < 0.0:
        raise VerdictInputError(f"{context} must be nonnegative")
    if positive and result <= 0.0:
        raise VerdictInputError(f"{context} must be positive")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerdictInputError(f"{context} must be a nonnegative integer")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise VerdictInputError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerdictInputError(f"{context} must be a nonempty string")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise VerdictInputError(f"{context} must be a boolean")
    return value


def _timestamp(value: Any, context: str) -> str:
    result = _nonempty_string(value, context)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerdictInputError(f"{context} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise VerdictInputError(f"{context} must include a UTC offset")
    return result


def _vector3(value: Any, context: str) -> list[float]:
    raw = _sequence(value, context)
    if len(raw) != 3:
        raise VerdictInputError(f"{context} must contain exactly three values")
    return [
        _finite_number(item, f"{context}[{index}]") for index, item in enumerate(raw)
    ]


@lru_cache(maxsize=None)
def _cyclic_indices_sha256(n_cells: int, start: int, k: int) -> str:
    """Reconstruct the little-endian ordered int64 cyclic-selection digest."""

    digest = hashlib.sha256()
    chunk_size = 4096
    for offset in range(0, k, chunk_size):
        stop = min(offset + chunk_size, k)
        values = tuple((start + index) % n_cells for index in range(offset, stop))
        digest.update(struct.pack(f"<{len(values)}q", *values))
    return digest.hexdigest()


def _validate_provenance(value: Any, context: str) -> dict[str, Any]:
    provenance = _mapping(value, context)
    _exact_keys(
        provenance,
        (
            "producer_path",
            "producer_sha256",
            "config_sha256",
            "command",
            "python",
            "platform",
            "versions",
            "hardware",
            "rescoring_npz_path",
            "rescoring_npz_sha256",
        ),
        context,
    )
    if EXPECTED_PRODUCER_SHA256 == "0" * 64:
        raise VerdictInputError(
            "H-QC reducer has not frozen the final producer SHA-256"
        )
    if EXPECTED_PRODUCER_CONFIG_SHA256 == "0" * 64:
        raise VerdictInputError(
            "H-QC reducer has not frozen the final producer config SHA-256"
        )
    producer_sha256 = _sha256(
        provenance["producer_sha256"], f"{context}.producer_sha256"
    )
    if producer_sha256 != EXPECTED_PRODUCER_SHA256:
        raise VerdictInputError(f"{context}.producer_sha256 is not frozen")
    config_sha256 = _sha256(provenance["config_sha256"], f"{context}.config_sha256")
    if config_sha256 != EXPECTED_PRODUCER_CONFIG_SHA256:
        raise VerdictInputError(f"{context}.config_sha256 is not frozen")

    command = _sequence(provenance["command"], f"{context}.command")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise VerdictInputError(f"{context}.command must contain nonempty strings")
    versions = _mapping(provenance["versions"], f"{context}.versions")
    _exact_keys(versions, ("numpy", "torch", "physicsnemo"), f"{context}.versions")
    validated_versions = {
        key: _nonempty_string(value, f"{context}.versions.{key}")
        for key, value in versions.items()
    }
    hardware = _mapping(provenance["hardware"], f"{context}.hardware")
    _exact_keys(
        hardware,
        (
            "cuda_runtime",
            "visible_cuda_device_count",
            "cuda_device_name",
            "cuda_device_capability",
            "cudnn_version",
        ),
        f"{context}.hardware",
    )
    visible_devices = _nonnegative_int(
        hardware["visible_cuda_device_count"],
        f"{context}.hardware.visible_cuda_device_count",
    )
    if visible_devices == 0:
        raise VerdictInputError(
            f"{context}.hardware.visible_cuda_device_count must be positive"
        )
    capability = _sequence(
        hardware["cuda_device_capability"],
        f"{context}.hardware.cuda_device_capability",
    )
    if len(capability) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in capability
    ):
        raise VerdictInputError(
            f"{context}.hardware.cuda_device_capability must be two integers"
        )
    cudnn_version = _nonnegative_int(
        hardware["cudnn_version"],
        f"{context}.hardware.cudnn_version",
    )
    if cudnn_version == 0:
        raise VerdictInputError(f"{context}.hardware.cudnn_version must be positive")
    validated_hardware = {
        "cuda_runtime": _nonempty_string(
            hardware["cuda_runtime"],
            f"{context}.hardware.cuda_runtime",
        ),
        "visible_cuda_device_count": visible_devices,
        "cuda_device_name": _nonempty_string(
            hardware["cuda_device_name"],
            f"{context}.hardware.cuda_device_name",
        ),
        "cuda_device_capability": capability,
        "cudnn_version": cudnn_version,
    }
    return {
        "producer_path": _nonempty_string(
            provenance["producer_path"], f"{context}.producer_path"
        ),
        "producer_sha256": producer_sha256,
        "config_sha256": config_sha256,
        "command": command,
        "python": _nonempty_string(provenance["python"], f"{context}.python"),
        "platform": _nonempty_string(provenance["platform"], f"{context}.platform"),
        "versions": validated_versions,
        "hardware": validated_hardware,
        "rescoring_npz_path": _nonempty_string(
            provenance["rescoring_npz_path"],
            f"{context}.rescoring_npz_path",
        ),
        "rescoring_npz_sha256": _sha256(
            provenance["rescoring_npz_sha256"],
            f"{context}.rescoring_npz_sha256",
        ),
    }


def _validate_source_identity(value: Any, context: str) -> dict[str, str]:
    source = _mapping(value, context)
    keys = (
        "metadata_sha256",
        "points_sha256",
        "cells_sha256",
        "pressure_sha256",
        "wss_sha256",
    )
    _exact_keys(source, keys, context)
    return {key: _sha256(source[key], f"{context}.{key}") for key in keys}


def _validate_historical(
    value: Any,
    context: str,
    *,
    case_id: str,
    expected_selection_sha256: str,
) -> dict[str, Any]:
    historical = _mapping(value, context)
    keys = (
        "reader_seed_fork_chain_sha256",
        "seed_fork_chain_replayed",
        "selection_sha256_int64",
        "canonical_reconstructed_signature_sha256",
        "saved_artifact_coordinate_max_abs_error",
        "saved_artifact_pipeline_normals_max_abs_error",
        "saved_artifact_parity_passed",
        "exact_archived_row_available",
        "archived_uniform_pressure_relative_l2",
    )
    _exact_keys(historical, keys, context)
    selection_sha256 = _sha256(
        historical["selection_sha256_int64"],
        f"{context}.selection_sha256_int64",
    )
    if selection_sha256 != expected_selection_sha256:
        raise VerdictInputError(
            f"{context}.selection_sha256_int64 does not match start/N/10k"
        )
    parity_error = _finite_number(
        historical["saved_artifact_coordinate_max_abs_error"],
        f"{context}.saved_artifact_coordinate_max_abs_error",
        nonnegative=True,
    )
    pipeline_normal_error = _finite_number(
        historical["saved_artifact_pipeline_normals_max_abs_error"],
        f"{context}.saved_artifact_pipeline_normals_max_abs_error",
        nonnegative=True,
    )
    parity_recomputed = (
        parity_error <= RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE
        and pipeline_normal_error <= ARCHIVED_PIPELINE_NORMAL_ABS_TOLERANCE
    )
    parity_reported = _boolean(
        historical["saved_artifact_parity_passed"],
        f"{context}.saved_artifact_parity_passed",
    )
    if parity_reported is not parity_recomputed:
        raise VerdictInputError(
            f"{context}.saved_artifact_parity_passed disagrees with its error"
        )
    if historical["exact_archived_row_available"] is not True:
        raise VerdictInputError(
            f"{context}.exact_archived_row_available must be true for this cohort"
        )
    archived = _finite_number(
        historical["archived_uniform_pressure_relative_l2"],
        f"{context}.archived_uniform_pressure_relative_l2",
        positive=True,
    )
    expected_archived = ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case_id]
    if not math.isclose(archived, expected_archived, rel_tol=0.0, abs_tol=1.0e-15):
        raise VerdictInputError(
            f"{context}.archived_uniform_pressure_relative_l2 "
            "differs from the frozen historical row"
        )
    reader_chain_sha256 = _sha256(
        historical["reader_seed_fork_chain_sha256"],
        f"{context}.reader_seed_fork_chain_sha256",
    )
    if reader_chain_sha256 != EXPECTED_READER_SEED_FORK_CHAIN_SHA256:
        raise VerdictInputError(
            f"{context}.reader_seed_fork_chain_sha256 is not frozen"
        )
    return {
        "reader_seed_fork_chain_sha256": reader_chain_sha256,
        "seed_fork_chain_replayed": _boolean(
            historical["seed_fork_chain_replayed"],
            f"{context}.seed_fork_chain_replayed",
        ),
        "selection_sha256_int64": selection_sha256,
        "canonical_reconstructed_signature_sha256": _sha256(
            historical["canonical_reconstructed_signature_sha256"],
            f"{context}.canonical_reconstructed_signature_sha256",
        ),
        "saved_artifact_coordinate_max_abs_error": parity_error,
        "saved_artifact_pipeline_normals_max_abs_error": pipeline_normal_error,
        "saved_artifact_parity_passed": parity_reported,
        "exact_archived_row_available": True,
        "archived_uniform_pressure_relative_l2": archived,
    }


def _validate_fixed_q(
    value: Any,
    context: str,
    *,
    expected_q_sha256: str,
) -> dict[str, Any]:
    fixed_q = _mapping(value, context)
    keys = (
        "raw_cell_ids_sha256_int64",
        "truth_pressure_sha256_float32",
        "normals_sha256_float32",
        "native_areas_sha256_float64",
        "truth_rms",
        "native_area",
        "mean_native_cell_area",
        "identity_checks_passed",
    )
    _exact_keys(fixed_q, keys, context)
    raw_ids_sha256 = _sha256(
        fixed_q["raw_cell_ids_sha256_int64"],
        f"{context}.raw_cell_ids_sha256_int64",
    )
    if raw_ids_sha256 != expected_q_sha256:
        raise VerdictInputError(
            f"{context}.raw_cell_ids_sha256_int64 does not match start/N/Q"
        )
    return {
        "raw_cell_ids_sha256_int64": raw_ids_sha256,
        "truth_pressure_sha256_float32": _sha256(
            fixed_q["truth_pressure_sha256_float32"],
            f"{context}.truth_pressure_sha256_float32",
        ),
        "normals_sha256_float32": _sha256(
            fixed_q["normals_sha256_float32"],
            f"{context}.normals_sha256_float32",
        ),
        "native_areas_sha256_float64": _sha256(
            fixed_q["native_areas_sha256_float64"],
            f"{context}.native_areas_sha256_float64",
        ),
        "truth_rms": _finite_number(
            fixed_q["truth_rms"], f"{context}.truth_rms", positive=True
        ),
        "native_area": _finite_number(
            fixed_q["native_area"], f"{context}.native_area", positive=True
        ),
        "mean_native_cell_area": _finite_number(
            fixed_q["mean_native_cell_area"],
            f"{context}.mean_native_cell_area",
            positive=True,
        ),
        "identity_checks_passed": _boolean(
            fixed_q["identity_checks_passed"],
            f"{context}.identity_checks_passed",
        ),
    }


def _validate_s10k_reference(value: Any, context: str) -> dict[str, float]:
    reference = _mapping(value, context)
    _exact_keys(
        reference,
        ("truth_rms", "native_area", "mean_native_cell_area"),
        context,
    )
    return {
        key: _finite_number(reference[key], f"{context}.{key}", positive=True)
        for key in ("truth_rms", "native_area", "mean_native_cell_area")
    }


def _validate_centers(value: Any, context: str) -> dict[str, Any]:
    centers = _mapping(value, context)
    _exact_keys(
        centers,
        (
            "fixed_s10k",
            "primary_by_k",
            "raw_frame_q_reconstruction_max_abs",
            "by_k",
            "passed",
        ),
        context,
    )
    expected_k_keys = tuple(str(k) for k in RESOLUTIONS)
    primary_by_k = _mapping(centers["primary_by_k"], f"{context}.primary_by_k")
    _exact_keys(primary_by_k, expected_k_keys, f"{context}.primary_by_k")
    validated_primary = {
        key: _vector3(primary_by_k[key], f"{context}.primary_by_k.{key}")
        for key in expected_k_keys
    }
    diagnostics_by_k = _mapping(centers["by_k"], f"{context}.by_k")
    _exact_keys(diagnostics_by_k, expected_k_keys, f"{context}.by_k")
    validated_diagnostics = {}
    for key in expected_k_keys:
        k_context = f"{context}.by_k.{key}"
        by_arm = _mapping(diagnostics_by_k[key], k_context)
        _exact_keys(by_arm, ("coupled", "fixed_q"), k_context)
        validated_diagnostics[key] = {}
        for arm in ("coupled", "fixed_q"):
            diagnostic_context = f"{k_context}.{arm}"
            diagnostic = _mapping(by_arm[arm], diagnostic_context)
            _exact_keys(diagnostic, CENTER_DIAGNOSTIC_KEYS, diagnostic_context)
            validated_diagnostics[key][arm] = {
                name: _finite_number(
                    diagnostic[name],
                    f"{diagnostic_context}.{name}",
                    nonnegative=True,
                )
                for name in CENTER_DIAGNOSTIC_KEYS
            }
    raw_error = _finite_number(
        centers["raw_frame_q_reconstruction_max_abs"],
        f"{context}.raw_frame_q_reconstruction_max_abs",
        nonnegative=True,
    )
    threshold_checks = {
        "raw_frame_q_reconstruction": (
            raw_error <= RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE
        ),
        **{
            f"{k}.{arm}.{name}": value <= CENTER_METRIC_RELATIVE_TOLERANCE
            for k, by_arm in validated_diagnostics.items()
            for arm, diagnostics in by_arm.items()
            for name, value in diagnostics.items()
        },
    }
    recomputed_passed = all(threshold_checks.values())
    reported_passed = _boolean(centers["passed"], f"{context}.passed")
    if reported_passed is not recomputed_passed:
        raise VerdictInputError(f"{context}.passed disagrees with its diagnostics")
    return {
        "fixed_s10k": _vector3(centers["fixed_s10k"], f"{context}.fixed_s10k"),
        "primary_by_k": validated_primary,
        "raw_frame_q_reconstruction_max_abs": raw_error,
        "by_k": validated_diagnostics,
        "threshold_checks": threshold_checks,
        "passed": recomputed_passed,
    }


def _validate_full_metrics(value: Any, context: str) -> dict[str, float]:
    metrics = _mapping(value, context)
    _exact_keys(metrics, FULL_METRIC_KEYS, context)
    validated = {}
    for key in FULL_METRIC_KEYS:
        validated[key] = _finite_number(
            metrics[key],
            f"{context}.{key}",
            positive=key == "pressure_relative_l2",
            nonnegative=key
            not in ("pressure_relative_l2", "signed_centered_correlation"),
        )
    correlation = validated["signed_centered_correlation"]
    if not -1.0 <= correlation <= 1.0:
        raise VerdictInputError(
            f"{context}.signed_centered_correlation must lie in [-1, 1]"
        )
    return validated


def _validate_metrics(value: Any, context: str) -> dict[str, Any]:
    metrics = _mapping(value, context)
    _exact_keys(metrics, ("uniform", "area_weighted"), context)

    uniform = _mapping(metrics["uniform"], f"{context}.uniform")
    _exact_keys(uniform, ("coupled", "fixed_q"), f"{context}.uniform")
    validated_uniform = {
        arm: _validate_full_metrics(
            uniform[arm],
            f"{context}.uniform.{arm}",
        )
        for arm in ("coupled", "fixed_q")
    }

    area = _mapping(metrics["area_weighted"], f"{context}.area_weighted")
    _exact_keys(area, ("coupled", "fixed_q"), f"{context}.area_weighted")
    validated_area = {}
    for arm in ("coupled", "fixed_q"):
        arm_context = f"{context}.area_weighted.{arm}"
        arm_metrics = _mapping(area[arm], arm_context)
        _exact_keys(arm_metrics, ("pressure_relative_l2",), arm_context)
        validated_area[arm] = {
            "pressure_relative_l2": _finite_number(
                arm_metrics["pressure_relative_l2"],
                f"{arm_context}.pressure_relative_l2",
                positive=True,
            )
        }
    return {"uniform": validated_uniform, "area_weighted": validated_area}


def _validate_normal_diagnostics(value: Any, context: str) -> dict[str, Any]:
    diagnostics = _mapping(value, context)
    _exact_keys(diagnostics, ("primary", "fixed_center"), context)
    validated: dict[str, dict[str, float]] = {}
    for arm in ("primary", "fixed_center"):
        arm_context = f"{context}.{arm}"
        arm_diagnostics = _mapping(diagnostics[arm], arm_context)
        keys = (
            "max_unit_norm_abs_error",
            "max_geometry_reconstruction_abs_error",
            "min_native_dot",
        )
        _exact_keys(arm_diagnostics, keys, arm_context)
        unit_error = _finite_number(
            arm_diagnostics["max_unit_norm_abs_error"],
            f"{arm_context}.max_unit_norm_abs_error",
            nonnegative=True,
        )
        geometry_error = _finite_number(
            arm_diagnostics["max_geometry_reconstruction_abs_error"],
            f"{arm_context}.max_geometry_reconstruction_abs_error",
            nonnegative=True,
        )
        min_native_dot = _finite_number(
            arm_diagnostics["min_native_dot"],
            f"{arm_context}.min_native_dot",
        )
        if unit_error > PIPELINE_NORMAL_UNIT_ABS_TOLERANCE:
            raise VerdictInputError(
                f"{arm_context}.max_unit_norm_abs_error exceeds the frozen tolerance"
            )
        if geometry_error > PIPELINE_NORMAL_GEOMETRY_ABS_TOLERANCE:
            raise VerdictInputError(
                f"{arm_context}.max_geometry_reconstruction_abs_error "
                "exceeds the frozen tolerance"
            )
        if min_native_dot <= 0.0:
            raise VerdictInputError(
                f"{arm_context}.min_native_dot must preserve native orientation"
            )
        if min_native_dot > 1.0 + 2.0 * PIPELINE_NORMAL_UNIT_ABS_TOLERANCE:
            raise VerdictInputError(
                f"{arm_context}.min_native_dot is incompatible with unit normals"
            )
        validated[arm] = {
            "max_unit_norm_abs_error": unit_error,
            "max_geometry_reconstruction_abs_error": geometry_error,
            "min_native_dot": min_native_dot,
        }
    return validated


def _validate_resolution(
    value: Any,
    context: str,
    *,
    expected_k: int,
    expected_selection_sha256: str,
    expected_q_sha256: str,
) -> dict[str, Any]:
    resolution = _mapping(value, context)
    _exact_keys(
        resolution,
        (
            "k",
            "selection",
            "normal_diagnostics",
            "metrics",
            "finite_checks_passed",
        ),
        context,
    )
    if resolution["k"] != expected_k:
        raise VerdictInputError(f"{context}.k must be {expected_k}")
    selection = _mapping(resolution["selection"], f"{context}.selection")
    _exact_keys(
        selection,
        (
            "cell_ids_sha256_int64",
            "q_prefix_sha256_int64",
            "nested_prefix_passed",
        ),
        f"{context}.selection",
    )
    selection_sha256 = _sha256(
        selection["cell_ids_sha256_int64"],
        f"{context}.selection.cell_ids_sha256_int64",
    )
    if selection_sha256 != expected_selection_sha256:
        raise VerdictInputError(
            f"{context}.selection.cell_ids_sha256_int64 does not match start/N/k"
        )
    q_prefix_sha256 = _sha256(
        selection["q_prefix_sha256_int64"],
        f"{context}.selection.q_prefix_sha256_int64",
    )
    if q_prefix_sha256 != expected_q_sha256:
        raise VerdictInputError(
            f"{context}.selection.q_prefix_sha256_int64 does not preserve Q"
        )
    return {
        "k": expected_k,
        "selection": {
            "cell_ids_sha256_int64": selection_sha256,
            "q_prefix_sha256_int64": q_prefix_sha256,
            "nested_prefix_passed": _boolean(
                selection["nested_prefix_passed"],
                f"{context}.selection.nested_prefix_passed",
            ),
        },
        "normal_diagnostics": _validate_normal_diagnostics(
            resolution["normal_diagnostics"],
            f"{context}.normal_diagnostics",
        ),
        "metrics": _validate_metrics(resolution["metrics"], f"{context}.metrics"),
        "finite_checks_passed": _boolean(
            resolution["finite_checks_passed"],
            f"{context}.finite_checks_passed",
        ),
    }


def _validate_case(value: Any, context: str) -> dict[str, Any]:
    case = _mapping(value, context)
    _exact_keys(
        case,
        (
            "cohort_ordinal",
            "case_id",
            "reader_index",
            "n_master_cells",
            "historical_start",
            "source_identity",
            "historical_10k",
            "fixed_q",
            "s10k_reference",
            "centers",
            "resolutions",
        ),
        context,
    )
    ordinal = _nonnegative_int(case["cohort_ordinal"], f"{context}.cohort_ordinal")
    if ordinal >= len(EXPECTED_CASES):
        raise VerdictInputError(f"{context}.cohort_ordinal is out of range")
    case_id = case["case_id"]
    if case_id != EXPECTED_CASES[ordinal]:
        raise VerdictInputError(
            f"{context}.case_id does not match its frozen cohort ordinal"
        )
    if case["reader_index"] != EXPECTED_READER_INDICES[case_id]:
        raise VerdictInputError(f"{context}.reader_index is not frozen")
    n_master_cells = _nonnegative_int(
        case["n_master_cells"], f"{context}.n_master_cells"
    )
    if n_master_cells != EXPECTED_MASTER_CELLS[case_id]:
        raise VerdictInputError(f"{context}.n_master_cells is not frozen")
    historical_start = _nonnegative_int(
        case["historical_start"], f"{context}.historical_start"
    )
    if historical_start != EXPECTED_HISTORICAL_STARTS[case_id]:
        raise VerdictInputError(f"{context}.historical_start is not frozen")

    selection_hashes = {
        k: _cyclic_indices_sha256(n_master_cells, historical_start, k)
        for k in RESOLUTIONS
    }
    source_identity = _validate_source_identity(
        case["source_identity"], f"{context}.source_identity"
    )
    historical = _validate_historical(
        case["historical_10k"],
        f"{context}.historical_10k",
        case_id=case_id,
        expected_selection_sha256=selection_hashes[BASELINE_K],
    )
    fixed_q = _validate_fixed_q(
        case["fixed_q"],
        f"{context}.fixed_q",
        expected_q_sha256=selection_hashes[FIXED_QUERY_K],
    )
    s10k_reference = _validate_s10k_reference(
        case["s10k_reference"],
        f"{context}.s10k_reference",
    )
    expected_q_mean_area = fixed_q["native_area"] / FIXED_QUERY_K
    if not math.isclose(
        fixed_q["mean_native_cell_area"],
        expected_q_mean_area,
        rel_tol=1.0e-14,
        abs_tol=0.0,
    ):
        raise VerdictInputError(
            f"{context}.fixed_q mean native area disagrees with total/count"
        )
    expected_s10k_mean_area = s10k_reference["native_area"] / BASELINE_K
    if not math.isclose(
        s10k_reference["mean_native_cell_area"],
        expected_s10k_mean_area,
        rel_tol=1.0e-14,
        abs_tol=0.0,
    ):
        raise VerdictInputError(
            f"{context}.s10k_reference mean native area disagrees with total/count"
        )
    centers = _validate_centers(case["centers"], f"{context}.centers")

    raw_resolutions = _sequence(case["resolutions"], f"{context}.resolutions")
    if len(raw_resolutions) != len(RESOLUTIONS):
        raise VerdictInputError(f"{context}.resolutions must contain exactly five rows")
    resolutions = [
        _validate_resolution(
            raw_resolution,
            f"{context}.resolutions[{index}]",
            expected_k=k,
            expected_selection_sha256=selection_hashes[k],
            expected_q_sha256=selection_hashes[FIXED_QUERY_K],
        )
        for index, (raw_resolution, k) in enumerate(
            zip(raw_resolutions, RESOLUTIONS, strict=True)
        )
    ]
    return {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": EXPECTED_READER_INDICES[case_id],
        "n_master_cells": n_master_cells,
        "historical_start": historical_start,
        "source_identity": source_identity,
        "historical_10k": historical,
        "fixed_q": fixed_q,
        "s10k_reference": s10k_reference,
        "centers": centers,
        "resolutions": resolutions,
    }


def _sha256_array(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    numerator = math.sqrt(
        float(np.sum((prediction64 - truth64) ** 2, dtype=np.float64))
    )
    denominator = math.sqrt(float(np.sum(truth64**2, dtype=np.float64))) + epsilon
    result = numerator / denominator
    if not math.isfinite(result):
        raise VerdictInputError("recomputed relative L2 is non-finite")
    return result


def _recompute_metric_bundle(
    *,
    prediction_pressure: np.ndarray,
    truth_pressure: np.ndarray,
    prediction_wss: np.ndarray,
    truth_wss: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    n_master_cells: int,
    context: str,
) -> dict[str, dict[str, float]]:
    pred_p = np.asarray(prediction_pressure, dtype=np.float64)
    true_p = np.asarray(truth_pressure, dtype=np.float64)
    pred_wss = np.asarray(prediction_wss, dtype=np.float64)
    true_wss = np.asarray(truth_wss, dtype=np.float64)
    normals64 = np.asarray(normals, dtype=np.float64)
    areas64 = np.asarray(areas, dtype=np.float64)
    m = len(pred_p)
    expected_shapes = {
        "truth_pressure": (m,),
        "prediction_wss": (m, 3),
        "truth_wss": (m, 3),
        "normals": (m, 3),
        "areas": (m,),
    }
    actual_shapes = {
        "truth_pressure": true_p.shape,
        "prediction_wss": pred_wss.shape,
        "truth_wss": true_wss.shape,
        "normals": normals64.shape,
        "areas": areas64.shape,
    }
    invalid_shapes = {
        key: (actual_shapes[key], expected)
        for key, expected in expected_shapes.items()
        if actual_shapes[key] != expected
    }
    if pred_p.shape != (m,) or invalid_shapes:
        raise VerdictInputError(
            f"{context} metric-array shapes are inconsistent: {invalid_shapes}"
        )
    values = (pred_p, true_p, pred_wss, true_wss, normals64, areas64)
    if not all(np.isfinite(value).all() for value in values):
        raise VerdictInputError(f"{context} metric arrays must be finite")
    if np.any(areas64 <= 0.0):
        raise VerdictInputError(f"{context} native areas must be positive")

    pred_centered = pred_p - pred_p.mean(dtype=np.float64)
    true_centered = true_p - true_p.mean(dtype=np.float64)
    pred_centered_norm = float(np.linalg.norm(pred_centered))
    true_centered_norm = float(np.linalg.norm(true_centered))
    if pred_centered_norm <= 0.0 or true_centered_norm <= 0.0:
        raise VerdictInputError(f"{context} centered pressure norm is zero")
    cosine = float(np.dot(pred_centered, true_centered))
    cosine /= pred_centered_norm * true_centered_norm
    signed_correlation = min(1.0, max(-1.0, cosine))
    positive_correlation = max(0.0, signed_correlation)
    pattern_error = math.sqrt(max(0.0, 1.0 - positive_correlation**2))

    true_rms = math.sqrt(float(np.mean(true_p**2, dtype=np.float64)))
    if true_rms <= 0.0:
        raise VerdictInputError(f"{context} pressure truth RMS is zero")
    amplitude_ratio = math.sqrt(float(np.mean(pred_p**2, dtype=np.float64))) / true_rms

    normal_component = np.einsum(
        "ij,ij->i",
        pred_wss,
        normals64,
        dtype=np.float64,
        optimize=True,
    )
    wss_normal_energy = float(np.linalg.norm(normal_component)) / (
        float(np.linalg.norm(pred_wss)) + 1.0e-8
    )
    scale = float(n_master_cells) / float(m)
    pred_force = scale * np.sum(
        areas64[:, None] * pred_p[:, None] * normals64,
        axis=0,
        dtype=np.float64,
    )
    true_force = scale * np.sum(
        areas64[:, None] * true_p[:, None] * normals64,
        axis=0,
        dtype=np.float64,
    )
    force_error = float(np.linalg.norm(pred_force - true_force)) / (
        float(np.linalg.norm(true_force)) + 1.0e-12
    )

    area_sum = float(areas64.sum(dtype=np.float64))
    weights = areas64 / area_sum
    area_pressure = math.sqrt(
        float(np.sum(weights * (pred_p - true_p) ** 2, dtype=np.float64))
    ) / (math.sqrt(float(np.sum(weights * true_p**2, dtype=np.float64))) + 1.0e-8)
    result = {
        "uniform": {
            "pressure_relative_l2": _relative_l2(pred_p, true_p),
            "signed_centered_correlation": signed_correlation,
            "positive_gain_pattern_error": pattern_error,
            "amplitude_ratio": amplitude_ratio,
            "wss_frobenius_relative_l2": _relative_l2(
                pred_wss.ravel(), true_wss.ravel()
            ),
            "wss_normal_energy": wss_normal_energy,
            "scaled_subset_pressure_force_relative_error": force_error,
        },
        "area_weighted": {"pressure_relative_l2": area_pressure},
    }
    flattened_metrics = [
        metric for metrics in result.values() for metric in metrics.values()
    ]
    if (
        not all(math.isfinite(metric) for metric in flattened_metrics)
        or any(
            metric < 0.0
            for name, metric in result["uniform"].items()
            if name != "signed_centered_correlation"
        )
        or any(metric < 0.0 for metric in result["area_weighted"].values())
    ):
        raise VerdictInputError(f"{context} recomputed metrics are invalid")
    if result["uniform"]["pressure_relative_l2"] <= 0.0:
        raise VerdictInputError(
            f"{context} recomputed pressure relative L2 must be positive"
        )
    return result


def _recompute_center_diagnostic(
    primary_pressure: np.ndarray,
    fixed_pressure: np.ndarray,
    primary_metrics: dict[str, dict[str, float]],
    fixed_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    primary64 = np.asarray(primary_pressure, dtype=np.float64)
    fixed64 = np.asarray(fixed_pressure, dtype=np.float64)
    prediction_difference = float(np.linalg.norm(primary64 - fixed64)) / max(
        float(np.linalg.norm(primary64)),
        float(np.linalg.norm(fixed64)),
        1.0e-8,
    )

    def relative_change(primary: float, fixed: float) -> float:
        return abs(primary - fixed) / max(abs(primary), abs(fixed), 1.0e-12)

    return {
        "pressure_prediction_relative_l2_difference": prediction_difference,
        "uniform_pressure_error_relative_change": relative_change(
            primary_metrics["uniform"]["pressure_relative_l2"],
            fixed_metrics["uniform"]["pressure_relative_l2"],
        ),
        "area_pressure_error_relative_change": relative_change(
            primary_metrics["area_weighted"]["pressure_relative_l2"],
            fixed_metrics["area_weighted"]["pressure_relative_l2"],
        ),
    }


def _compare_recomputed_number(
    reported: float,
    recomputed: float,
    context: str,
) -> tuple[float, float]:
    absolute_difference = abs(reported - recomputed)
    scale = max(abs(reported), abs(recomputed))
    relative_difference = absolute_difference / max(scale, 1.0e-300)
    tolerance = (
        METRIC_RECOMPUTATION_ABS_TOLERANCE + METRIC_RECOMPUTATION_REL_TOLERANCE * scale
    )
    if absolute_difference > tolerance:
        raise VerdictInputError(f"{context} differs from NPZ recomputation")
    return absolute_difference, relative_difference


def _compare_metric_bundle(
    reported: dict[str, Any],
    recomputed: dict[str, dict[str, float]],
    context: str,
) -> tuple[float, float]:
    differences = []
    for weighting in ("uniform", "area_weighted"):
        for name, recomputed_value in recomputed[weighting].items():
            differences.append(
                _compare_recomputed_number(
                    reported[weighting][name],
                    recomputed_value,
                    f"{context}.{weighting}.{name}",
                )
            )
    return (
        max((difference[0] for difference in differences), default=0.0),
        max((difference[1] for difference in differences), default=0.0),
    )


def _compare_pipeline_normal_diagnostic(
    reported: float,
    recomputed: float,
    context: str,
) -> float:
    """Compare a CUDA-float32 diagnostic with independent NumPy-float64 replay."""
    absolute_difference = abs(reported - recomputed)
    if absolute_difference > PIPELINE_NORMAL_DIAGNOSTIC_RECOMPUTATION_ABS_TOLERANCE:
        raise VerdictInputError(
            f"{context} differs from NPZ recomputation: "
            f"reported={reported:.17g}, recomputed={recomputed:.17g}, "
            f"abs_diff={absolute_difference:.3e}, tolerance="
            f"{PIPELINE_NORMAL_DIAGNOSTIC_RECOMPUTATION_ABS_TOLERANCE:.3e}"
        )
    return absolute_difference


def _npz_key(case_ordinal: int, k: int, field: str) -> str:
    return f"case_{case_ordinal:02d}__k_{k:05d}__{field}"


def _npz_array(
    archive: Any,
    key: str,
    *,
    shape: tuple[int, ...],
    dtype: str,
    context: str,
) -> np.ndarray:
    try:
        array = np.asarray(archive[key])
    except (OSError, ValueError, KeyError) as error:
        raise VerdictInputError(f"cannot load {context}: {error}") from error
    expected_dtype = np.dtype(dtype)
    if array.shape != shape:
        raise VerdictInputError(f"{context} has shape {array.shape}, expected {shape}")
    if array.dtype != expected_dtype:
        raise VerdictInputError(
            f"{context} has dtype {array.dtype}, expected {expected_dtype}"
        )
    if not array.flags.c_contiguous:
        raise VerdictInputError(f"{context} must be C-contiguous")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise VerdictInputError(f"{context} contains non-finite values")
    return array


def _recompute_pipeline_normal_diagnostics(
    pipeline_normals: np.ndarray,
    native_normals: np.ndarray,
    context: str,
) -> dict[str, float]:
    pipeline64 = np.asarray(pipeline_normals, dtype=np.float64)
    native64 = np.asarray(native_normals, dtype=np.float64)
    if pipeline64.shape != native64.shape or pipeline64.ndim != 2:
        raise VerdictInputError(
            f"{context} pipeline/native normal shapes must match, got "
            f"{pipeline64.shape} and {native64.shape}"
        )
    norms = np.linalg.norm(pipeline64, axis=1)
    max_unit_error = float(np.max(np.abs(norms - 1.0)))
    if max_unit_error > PIPELINE_NORMAL_UNIT_ABS_TOLERANCE:
        raise VerdictInputError(
            f"{context} pipeline normals are not unit length: "
            f"max_abs_error={max_unit_error:.3e}"
        )
    native_dots = np.einsum(
        "ij,ij->i",
        pipeline64,
        native64,
        dtype=np.float64,
        optimize=True,
    )
    min_native_dot = float(np.min(native_dots))
    if min_native_dot <= 0.0:
        raise VerdictInputError(
            f"{context} pipeline normals do not preserve native orientation: "
            f"min_dot={min_native_dot:.17g}"
        )
    return {
        "max_unit_norm_abs_error": max_unit_error,
        "min_native_dot": min_native_dot,
    }


def _recover_center_and_scale(
    raw_centroids: np.ndarray,
    query_points: np.ndarray,
    context: str,
) -> tuple[np.ndarray, float, float]:
    raw64 = np.asarray(raw_centroids, dtype=np.float64)
    query64 = np.asarray(query_points, dtype=np.float64)
    raw_centered = raw64 - raw64.mean(axis=0)
    query_centered = query64 - query64.mean(axis=0)
    denominator = float(np.sum(query_centered**2, dtype=np.float64))
    if denominator <= 0.0:
        raise VerdictInputError(f"{context} query coordinates have zero variance")
    scale = float(np.sum(raw_centered * query_centered, dtype=np.float64) / denominator)
    if not math.isfinite(scale) or scale <= 0.0:
        raise VerdictInputError(f"{context} recovered coordinate scale is invalid")
    center = np.mean(raw64 - scale * query64, axis=0, dtype=np.float64)
    residual = float(np.max(np.abs(raw64 - (scale * query64 + center[None, :]))))
    if residual > RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE:
        raise VerdictInputError(
            f"{context} raw/query reconstruction residual {residual:.3e} "
            f"exceeds {RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE:.1e}"
        )
    return center, scale, residual


def _validate_rescoring_npz(
    json_path: Path,
    cases: list[dict[str, Any]],
    provenance: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise VerdictInputError(
            f"{context} is missing fetched sibling rescoring NPZ {npz_path}"
        )
    if os.path.samefile(json_path, npz_path):
        raise VerdictInputError(f"{context} JSON and NPZ must not alias")
    if Path(provenance["rescoring_npz_path"]).name != npz_path.name:
        raise VerdictInputError(
            f"{context}.provenance.rescoring_npz_path has the wrong sibling name"
        )
    observed_sha256 = _sha256_file(npz_path)
    if observed_sha256 != provenance["rescoring_npz_sha256"]:
        raise VerdictInputError(f"{context} sibling NPZ SHA-256 does not match JSON")

    expected_keys = {
        _npz_key(case["cohort_ordinal"], k, field)
        for case in cases
        for k in RESOLUTIONS
        for field in NPZ_FIELDS
    }
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise VerdictInputError(f"cannot open {npz_path}: {error}") from error

    max_metric_abs_difference = 0.0
    max_metric_relative_difference = 0.0
    max_center_abs_difference = 0.0
    max_coordinate_reconstruction_residual = 0.0
    max_pipeline_normal_unit_abs_error = 0.0
    min_pipeline_native_dot = math.inf
    max_reported_pipeline_normal_geometry_abs_error = 0.0
    recovered_scales = []
    try:
        if len(archive.files) != len(set(archive.files)):
            raise VerdictInputError(f"{context} NPZ contains duplicate keys")
        observed_keys = set(archive.files)
        if observed_keys != expected_keys:
            raise VerdictInputError(
                f"{context} NPZ has wrong keys; "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"unexpected={sorted(observed_keys - expected_keys)}"
            )

        for case in cases:
            ordinal = case["cohort_ordinal"]
            case_context = f"{context}.npz:{case['case_id']}"
            rows_by_k = {row["k"]: row for row in case["resolutions"]}
            q_reference: dict[str, np.ndarray] | None = None
            baseline_arrays: dict[str, np.ndarray] | None = None
            for k in RESOLUTIONS:
                shapes_and_dtypes = {
                    "raw_cell_ids_int64": ((k,), "<i8"),
                    "compacted_cells_int64": ((k, 3), "<i8"),
                    "raw_centroids_float32": ((k, 3), "<f4"),
                    "native_normals_float32": ((k, 3), "<f4"),
                    "native_areas_float64": ((k,), "<f8"),
                    "truth_pressure_float32": ((k,), "<f4"),
                    "truth_wss_float32": ((k, 3), "<f4"),
                    "primary_query_points_float32": ((k, 3), "<f4"),
                    "primary_pipeline_normals_float32": ((k, 3), "<f4"),
                    "primary_pressure_float32": ((k,), "<f4"),
                    "primary_wss_float32": ((k, 3), "<f4"),
                    "fixed_center_query_points_float32": ((k, 3), "<f4"),
                    "fixed_center_pipeline_normals_float32": ((k, 3), "<f4"),
                    "fixed_center_pressure_float32": ((k,), "<f4"),
                    "fixed_center_wss_float32": ((k, 3), "<f4"),
                }
                arrays = {
                    field: _npz_array(
                        archive,
                        _npz_key(ordinal, k, field),
                        shape=shape,
                        dtype=dtype,
                        context=f"{case_context}.k{k}.{field}",
                    )
                    for field, (shape, dtype) in shapes_and_dtypes.items()
                }

                expected_ids = (
                    case["historical_start"] + np.arange(k, dtype=np.int64)
                ) % case["n_master_cells"]
                if not np.array_equal(arrays["raw_cell_ids_int64"], expected_ids):
                    raise VerdictInputError(
                        f"{case_context}.k{k} raw cell IDs are not the frozen prefix"
                    )
                cells = arrays["compacted_cells_int64"]
                if np.any(cells < 0) or np.any(
                    (cells[:, 0] == cells[:, 1])
                    | (cells[:, 0] == cells[:, 2])
                    | (cells[:, 1] == cells[:, 2])
                ):
                    raise VerdictInputError(
                        f"{case_context}.k{k} compacted triangles are invalid"
                    )
                areas = arrays["native_areas_float64"]
                if np.any(areas <= 0.0):
                    raise VerdictInputError(
                        f"{case_context}.k{k} native areas must be positive"
                    )
                normal_norms = np.linalg.norm(
                    arrays["native_normals_float32"].astype(np.float64),
                    axis=1,
                )
                if not np.allclose(
                    normal_norms,
                    1.0,
                    rtol=5.0e-6,
                    atol=5.0e-6,
                ):
                    raise VerdictInputError(
                        f"{case_context}.k{k} native normals are not unit length"
                    )
                for arm, field in (
                    ("primary", "primary_pipeline_normals_float32"),
                    ("fixed_center", "fixed_center_pipeline_normals_float32"),
                ):
                    recomputed_normal_diagnostics = (
                        _recompute_pipeline_normal_diagnostics(
                            arrays[field],
                            arrays["native_normals_float32"],
                            f"{case_context}.k{k}.{arm}",
                        )
                    )
                    reported_normal_diagnostics = rows_by_k[k]["normal_diagnostics"][
                        arm
                    ]
                    for name, recomputed in recomputed_normal_diagnostics.items():
                        _compare_pipeline_normal_diagnostic(
                            reported_normal_diagnostics[name],
                            recomputed,
                            f"{case_context}.k{k}.normal_diagnostics.{arm}.{name}",
                        )
                    max_pipeline_normal_unit_abs_error = max(
                        max_pipeline_normal_unit_abs_error,
                        recomputed_normal_diagnostics["max_unit_norm_abs_error"],
                    )
                    min_pipeline_native_dot = min(
                        min_pipeline_native_dot,
                        recomputed_normal_diagnostics["min_native_dot"],
                    )
                    max_reported_pipeline_normal_geometry_abs_error = max(
                        max_reported_pipeline_normal_geometry_abs_error,
                        reported_normal_diagnostics[
                            "max_geometry_reconstruction_abs_error"
                        ],
                    )
                if k == BASELINE_K and not np.array_equal(
                    arrays["primary_pipeline_normals_float32"],
                    arrays["fixed_center_pipeline_normals_float32"],
                ):
                    raise VerdictInputError(
                        f"{case_context}.k{k} primary/fixed-center pipeline "
                        "normals differ at the shared S10k center"
                    )

                fixed_identity_fields = (
                    "raw_cell_ids_int64",
                    "raw_centroids_float32",
                    "native_normals_float32",
                    "native_areas_float64",
                    "truth_pressure_float32",
                    "truth_wss_float32",
                    "fixed_center_query_points_float32",
                    "fixed_center_pipeline_normals_float32",
                )
                current_q = {
                    field: arrays[field][:FIXED_QUERY_K]
                    for field in fixed_identity_fields
                }
                if q_reference is None:
                    q_reference = {
                        field: np.array(value, copy=True)
                        for field, value in current_q.items()
                    }
                else:
                    for field, reference in q_reference.items():
                        if not np.array_equal(current_q[field], reference):
                            raise VerdictInputError(
                                f"{case_context}.k{k} fixed-Q changed for {field}"
                            )

                primary_metrics = _recompute_metric_bundle(
                    prediction_pressure=arrays["primary_pressure_float32"],
                    truth_pressure=arrays["truth_pressure_float32"],
                    prediction_wss=arrays["primary_wss_float32"],
                    truth_wss=arrays["truth_wss_float32"],
                    normals=arrays["native_normals_float32"],
                    areas=areas,
                    n_master_cells=case["n_master_cells"],
                    context=f"{case_context}.k{k}.primary.coupled",
                )
                primary_q_metrics = _recompute_metric_bundle(
                    prediction_pressure=arrays["primary_pressure_float32"][
                        :FIXED_QUERY_K
                    ],
                    truth_pressure=arrays["truth_pressure_float32"][:FIXED_QUERY_K],
                    prediction_wss=arrays["primary_wss_float32"][:FIXED_QUERY_K],
                    truth_wss=arrays["truth_wss_float32"][:FIXED_QUERY_K],
                    normals=arrays["native_normals_float32"][:FIXED_QUERY_K],
                    areas=areas[:FIXED_QUERY_K],
                    n_master_cells=case["n_master_cells"],
                    context=f"{case_context}.k{k}.primary.fixed_q",
                )
                fixed_metrics = _recompute_metric_bundle(
                    prediction_pressure=arrays["fixed_center_pressure_float32"],
                    truth_pressure=arrays["truth_pressure_float32"],
                    prediction_wss=arrays["fixed_center_wss_float32"],
                    truth_wss=arrays["truth_wss_float32"],
                    normals=arrays["native_normals_float32"],
                    areas=areas,
                    n_master_cells=case["n_master_cells"],
                    context=f"{case_context}.k{k}.fixed_center.coupled",
                )
                fixed_q_center_metrics = _recompute_metric_bundle(
                    prediction_pressure=arrays["fixed_center_pressure_float32"][
                        :FIXED_QUERY_K
                    ],
                    truth_pressure=arrays["truth_pressure_float32"][:FIXED_QUERY_K],
                    prediction_wss=arrays["fixed_center_wss_float32"][:FIXED_QUERY_K],
                    truth_wss=arrays["truth_wss_float32"][:FIXED_QUERY_K],
                    normals=arrays["native_normals_float32"][:FIXED_QUERY_K],
                    areas=areas[:FIXED_QUERY_K],
                    n_master_cells=case["n_master_cells"],
                    context=f"{case_context}.k{k}.fixed_center.fixed_q",
                )

                reported = rows_by_k[k]["metrics"]
                coupled_difference = _compare_metric_bundle(
                    {
                        "uniform": reported["uniform"]["coupled"],
                        "area_weighted": reported["area_weighted"]["coupled"],
                    },
                    primary_metrics,
                    f"{case_context}.k{k}.metrics.coupled",
                )
                fixed_q_difference = _compare_metric_bundle(
                    {
                        "uniform": reported["uniform"]["fixed_q"],
                        "area_weighted": reported["area_weighted"]["fixed_q"],
                    },
                    primary_q_metrics,
                    f"{case_context}.k{k}.metrics.fixed_q",
                )
                max_metric_abs_difference = max(
                    max_metric_abs_difference,
                    coupled_difference[0],
                    fixed_q_difference[0],
                )
                max_metric_relative_difference = max(
                    max_metric_relative_difference,
                    coupled_difference[1],
                    fixed_q_difference[1],
                )

                if k == FIXED_QUERY_K:
                    for weighting in ("uniform", "area_weighted"):
                        for name, coupled_value in reported[weighting][
                            "coupled"
                        ].items():
                            fixed_value = reported[weighting]["fixed_q"][name]
                            _compare_recomputed_number(
                                coupled_value,
                                fixed_value,
                                (
                                    f"{case_context}.k{k}.{weighting}.{name} "
                                    "coupled/fixed-Q identity"
                                ),
                            )

                center_diagnostics = {
                    "coupled": _recompute_center_diagnostic(
                        arrays["primary_pressure_float32"],
                        arrays["fixed_center_pressure_float32"],
                        primary_metrics,
                        fixed_metrics,
                    ),
                    "fixed_q": _recompute_center_diagnostic(
                        arrays["primary_pressure_float32"][:FIXED_QUERY_K],
                        arrays["fixed_center_pressure_float32"][:FIXED_QUERY_K],
                        primary_q_metrics,
                        fixed_q_center_metrics,
                    ),
                }
                for arm, diagnostics in center_diagnostics.items():
                    for name, value in diagnostics.items():
                        difference = _compare_recomputed_number(
                            case["centers"]["by_k"][str(k)][arm][name],
                            value,
                            (f"{case_context}.k{k}.center_diagnostic.{arm}.{name}"),
                        )
                        max_center_abs_difference = max(
                            max_center_abs_difference, difference[0]
                        )

                primary_center, primary_scale, primary_residual = (
                    _recover_center_and_scale(
                        arrays["raw_centroids_float32"],
                        arrays["primary_query_points_float32"],
                        f"{case_context}.k{k}.primary_center",
                    )
                )
                fixed_center, fixed_scale, fixed_residual = _recover_center_and_scale(
                    arrays["raw_centroids_float32"],
                    arrays["fixed_center_query_points_float32"],
                    f"{case_context}.k{k}.fixed_center",
                )
                if not math.isclose(
                    primary_scale,
                    fixed_scale,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                ):
                    raise VerdictInputError(
                        f"{case_context}.k{k} centering paths changed coordinate scale"
                    )
                recovered_scales.extend((primary_scale, fixed_scale))
                max_coordinate_reconstruction_residual = max(
                    max_coordinate_reconstruction_residual,
                    primary_residual,
                    fixed_residual,
                )
                for label, recovered, reported_center in (
                    (
                        "primary",
                        primary_center,
                        case["centers"]["primary_by_k"][str(k)],
                    ),
                    ("fixed_s10k", fixed_center, case["centers"]["fixed_s10k"]),
                ):
                    center_difference = float(
                        np.max(
                            np.abs(
                                recovered
                                - np.asarray(reported_center, dtype=np.float64)
                            )
                        )
                    )
                    if center_difference > RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE:
                        raise VerdictInputError(
                            f"{case_context}.k{k}.{label} center differs from NPZ "
                            f"by {center_difference:.3e}"
                        )
                    max_center_abs_difference = max(
                        max_center_abs_difference, center_difference
                    )
                if rows_by_k[k]["finite_checks_passed"] is not True:
                    raise VerdictInputError(
                        f"{case_context}.k{k} finite flag disagrees with NPZ"
                    )
                if rows_by_k[k]["selection"]["nested_prefix_passed"] is not True:
                    raise VerdictInputError(
                        f"{case_context}.k{k} prefix flag disagrees with NPZ"
                    )
                if k == BASELINE_K:
                    baseline_arrays = {
                        field: arrays[field]
                        for field in (
                            "truth_pressure_float32",
                            "native_areas_float64",
                        )
                    }

            if q_reference is None or baseline_arrays is None:
                raise AssertionError("five NPZ resolutions must define Q and S10k")
            q_pressure = q_reference["truth_pressure_float32"]
            q_normals = q_reference["native_normals_float32"]
            q_areas = q_reference["native_areas_float64"]
            fixed_q = case["fixed_q"]
            expected_q_hashes = {
                "raw_cell_ids_sha256_int64": _sha256_array(
                    q_reference["raw_cell_ids_int64"], "<i8"
                ),
                "truth_pressure_sha256_float32": _sha256_array(q_pressure, "<f4"),
                "normals_sha256_float32": _sha256_array(q_normals, "<f4"),
                "native_areas_sha256_float64": _sha256_array(q_areas, "<f8"),
            }
            for name, expected_hash in expected_q_hashes.items():
                if fixed_q[name] != expected_hash:
                    raise VerdictInputError(
                        f"{case_context}.fixed_q.{name} differs from NPZ"
                    )
            q_truth_rms = math.sqrt(
                float(np.mean(q_pressure.astype(np.float64) ** 2, dtype=np.float64))
            )
            q_native_area = float(q_areas.sum(dtype=np.float64))
            for name, recomputed in (
                ("truth_rms", q_truth_rms),
                ("native_area", q_native_area),
                ("mean_native_cell_area", q_native_area / FIXED_QUERY_K),
            ):
                _compare_recomputed_number(
                    fixed_q[name],
                    recomputed,
                    f"{case_context}.fixed_q.{name}",
                )
            baseline_pressure = baseline_arrays["truth_pressure_float32"]
            baseline_areas = baseline_arrays["native_areas_float64"]
            baseline_truth_rms = math.sqrt(
                float(
                    np.mean(
                        baseline_pressure.astype(np.float64) ** 2,
                        dtype=np.float64,
                    )
                )
            )
            baseline_native_area = float(baseline_areas.sum(dtype=np.float64))
            for name, recomputed in (
                ("truth_rms", baseline_truth_rms),
                ("native_area", baseline_native_area),
                ("mean_native_cell_area", baseline_native_area / BASELINE_K),
            ):
                _compare_recomputed_number(
                    case["s10k_reference"][name],
                    recomputed,
                    f"{case_context}.s10k_reference.{name}",
                )
            if fixed_q["identity_checks_passed"] is not True:
                raise VerdictInputError(
                    f"{case_context}.fixed_q identity flag disagrees with NPZ"
                )
    except VerdictInputError:
        raise
    except (OSError, ValueError, TypeError, MemoryError) as error:
        raise VerdictInputError(f"cannot validate {npz_path}: {error}") from error
    finally:
        archive.close()

    return {
        "path": str(npz_path.resolve()),
        "sha256": observed_sha256,
        "key_count": len(expected_keys),
        "case_count": len(cases),
        "all_arrays_finite_with_frozen_shapes_and_dtypes": True,
        "all_json_metrics_recomputed": True,
        "all_fixed_center_diagnostics_recomputed": True,
        "all_pipeline_normal_npz_diagnostics_recomputed": True,
        "all_reported_pipeline_geometry_reconstruction_checks_passed": True,
        "fixed_center_q_pipeline_normals_exact_across_k": True,
        "baseline_pipeline_normals_exact_between_center_arms": True,
        "all_q_and_s10k_references_recomputed": True,
        "max_metric_absolute_difference": max_metric_abs_difference,
        "max_metric_relative_difference": max_metric_relative_difference,
        "max_center_absolute_difference": max_center_abs_difference,
        "max_pipeline_normal_unit_abs_error": (max_pipeline_normal_unit_abs_error),
        "min_pipeline_native_dot": min_pipeline_native_dot,
        "max_reported_pipeline_normal_geometry_abs_error": (
            max_reported_pipeline_normal_geometry_abs_error
        ),
        "max_coordinate_reconstruction_residual": (
            max_coordinate_reconstruction_residual
        ),
        "recovered_coordinate_scale_range": [
            min(recovered_scales),
            max(recovered_scales),
        ],
    }


def _validate_lane(
    path: Path,
    artifact: dict[str, Any],
    artifact_sha256: str,
) -> dict[str, Any]:
    context = str(path)
    _exact_keys(
        artifact,
        (
            "schema_version",
            "artifact_kind",
            "status",
            "generated_at_utc",
            "lane",
            "frozen",
            "cases",
            "provenance",
        ),
        context,
    )
    if artifact["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise VerdictInputError(
            f"{context}.schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if artifact["artifact_kind"] != EXPECTED_ARTIFACT_KIND:
        raise VerdictInputError(f"{context}.artifact_kind is not frozen")
    if artifact["status"] != EXPECTED_PRODUCER_STATUS:
        raise VerdictInputError(f"{context}.status must be {EXPECTED_PRODUCER_STATUS}")
    generated_at_utc = _timestamp(
        artifact["generated_at_utc"], f"{context}.generated_at_utc"
    )
    if artifact["frozen"] != FROZEN_CONTRACT:
        raise VerdictInputError(f"{context}.frozen differs from the H-QC contract")

    lane = _mapping(artifact["lane"], f"{context}.lane")
    _exact_keys(lane, ("ordinal", "count"), f"{context}.lane")
    ordinal = _nonnegative_int(lane["ordinal"], f"{context}.lane.ordinal")
    count = _nonnegative_int(lane["count"], f"{context}.lane.count")
    if count == 0 or ordinal >= count:
        raise VerdictInputError(f"{context}.lane ordinal/count are invalid")

    raw_cases = _sequence(artifact["cases"], f"{context}.cases")
    if not raw_cases:
        raise VerdictInputError(f"{context}.cases must not be empty")
    cases = [
        _validate_case(raw_case, f"{context}.cases[{index}]")
        for index, raw_case in enumerate(raw_cases)
    ]
    provenance = _validate_provenance(artifact["provenance"], f"{context}.provenance")
    rescoring_validation = _validate_rescoring_npz(
        path,
        cases,
        provenance,
        context,
    )
    return {
        "ordinal": ordinal,
        "count": count,
        "input_artifact": {
            "path": str(path.resolve()),
            "sha256": artifact_sha256,
            "generated_at_utc": generated_at_utc,
        },
        "provenance": provenance,
        "rescoring_validation": rescoring_validation,
        "cases": cases,
    }


def _pressure(case: dict[str, Any], k: int, weighting: str, arm: str) -> float:
    row = case["resolutions"][RESOLUTIONS.index(k)]
    return row["metrics"][weighting][arm]["pressure_relative_l2"]


def _median(values: list[float]) -> float:
    if len(values) != len(EXPECTED_CASES):
        raise AssertionError("all H-QC medians must use the complete cohort")
    return float(statistics.median(values))


def _common_eligibility(cases: list[dict[str, Any]]) -> dict[str, Any]:
    archived_replay = {}
    support_ratios = {}
    for case in cases:
        case_id = case["case_id"]
        replay = _pressure(case, BASELINE_K, "uniform", "coupled")
        archived = ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case_id]
        replay_abs_error = abs(replay - archived)
        q_truth_ratio = (
            case["fixed_q"]["truth_rms"] / case["s10k_reference"]["truth_rms"]
        )
        q_mean_area_ratio = (
            case["fixed_q"]["mean_native_cell_area"]
            / case["s10k_reference"]["mean_native_cell_area"]
        )
        archived_replay[case_id] = {
            "replayed_uniform_pressure_relative_l2": replay,
            "archived_uniform_pressure_relative_l2": archived,
            "absolute_error": replay_abs_error,
            "passed": replay_abs_error <= ARCHIVED_CASE_ABS_TOLERANCE,
        }
        support_ratios[case_id] = {
            "q_over_s10k_truth_rms": q_truth_ratio,
            "q_over_s10k_mean_native_cell_area": q_mean_area_ratio,
            "q_over_s10k_total_native_area_diagnostic": (
                case["fixed_q"]["native_area"] / case["s10k_reference"]["native_area"]
            ),
            "truth_rms_within_factor_two": 0.5 <= q_truth_ratio <= 2.0,
            "mean_native_cell_area_within_factor_two": (
                0.5 <= q_mean_area_ratio <= 2.0
            ),
        }

    replay_mean = statistics.fmean(
        _pressure(case, BASELINE_K, "uniform", "coupled") for case in cases
    )
    replay_mean_abs_error = abs(replay_mean - ARCHIVED_UNIFORM_PRESSURE_MEAN)
    checks = {
        "seed_fork_chain_replayed_all_cases": all(
            case["historical_10k"]["seed_fork_chain_replayed"] for case in cases
        ),
        "saved_10k_parity_all_cases": all(
            case["historical_10k"]["saved_artifact_parity_passed"] for case in cases
        ),
        "q_identity_all_cases": all(
            case["fixed_q"]["identity_checks_passed"] for case in cases
        ),
        "nested_prefix_all_cases_all_k": all(
            row["selection"]["nested_prefix_passed"]
            for case in cases
            for row in case["resolutions"]
        ),
        "finite_all_cases_all_k": all(
            row["finite_checks_passed"] for case in cases for row in case["resolutions"]
        ),
        "q_truth_rms_within_factor_two_all_cases": all(
            row["truth_rms_within_factor_two"] for row in support_ratios.values()
        ),
        "q_mean_native_cell_area_within_factor_two_all_cases": all(
            row["mean_native_cell_area_within_factor_two"]
            for row in support_ratios.values()
        ),
        "archived_10k_case_replay_all_cases": all(
            row["passed"] for row in archived_replay.values()
        ),
        "archived_10k_mean_replay": (
            replay_mean_abs_error <= ARCHIVED_MEAN_ABS_TOLERANCE
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "archived_replay": {
            "cohort_mean": replay_mean,
            "archived_cohort_mean": ARCHIVED_UNIFORM_PRESSURE_MEAN,
            "absolute_error": replay_mean_abs_error,
            "tolerance": ARCHIVED_MEAN_ABS_TOLERANCE,
            "per_case_tolerance": ARCHIVED_CASE_ABS_TOLERANCE,
            "per_case": archived_replay,
        },
        "q_support_ratios": support_ratios,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _metric_panel(
    cases: list[dict[str, Any]],
    weighting: str,
    common_eligibility: dict[str, Any],
) -> dict[str, Any]:
    per_case_logs: dict[str, dict[str, dict[str, float]]] = {}
    for case in cases:
        case_logs = {}
        for arm in ("coupled", "fixed_q"):
            baseline = _pressure(case, BASELINE_K, weighting, arm)
            case_logs[arm] = {
                str(endpoint): math.log(
                    _pressure(case, endpoint, weighting, arm) / baseline
                )
                for endpoint in ENDPOINTS
            }
        per_case_logs[case["case_id"]] = case_logs

    baseline_ratios = [
        _pressure(case, BASELINE_K, weighting, "fixed_q")
        / _pressure(case, BASELINE_K, weighting, "coupled")
        for case in cases
    ]
    baseline_ratio_median = _median(baseline_ratios)
    baseline_comparable = (
        BASELINE_COMPARABILITY_BOUNDS[0]
        <= baseline_ratio_median
        <= BASELINE_COMPARABILITY_BOUNDS[1]
    )

    endpoint_rows = {}
    cliff_eligible = True
    for endpoint in ENDPOINTS:
        coupled_logs = [
            per_case_logs[case["case_id"]]["coupled"][str(endpoint)] for case in cases
        ]
        fixed_logs = [
            per_case_logs[case["case_id"]]["fixed_q"][str(endpoint)] for case in cases
        ]
        coupled_log_median = _median(coupled_logs)
        fixed_log_median = _median(fixed_logs)
        coupled_ratio_count = sum(
            _pressure(case, endpoint, weighting, "coupled")
            / _pressure(case, BASELINE_K, weighting, "coupled")
            >= CLIFF_RATIO_THRESHOLD
            for case in cases
        )
        favorable_count = sum(
            fixed < coupled
            for fixed, coupled in zip(fixed_logs, coupled_logs, strict=True)
        )
        fixed_endpoint_ratios = [
            _pressure(case, endpoint, weighting, "fixed_q")
            / _pressure(case, BASELINE_K, weighting, "fixed_q")
            for case in cases
        ]
        fixed_endpoint_ratio_median = _median(fixed_endpoint_ratios)

        cliff_log_passed = coupled_log_median >= math.log(CLIFF_RATIO_THRESHOLD)
        cliff_count_passed = coupled_ratio_count >= CLIFF_CASE_COUNT_THRESHOLD
        endpoint_cliff_eligible = cliff_log_passed and cliff_count_passed
        cliff_eligible = cliff_eligible and endpoint_cliff_eligible

        attenuation_passed = max(0.0, fixed_log_median) <= (
            SUPPORT_ATTENUATION_FRACTION * coupled_log_median
        )
        fixed_cap_passed = fixed_log_median <= math.log(SUPPORT_FIXED_RATIO_THRESHOLD)
        favorable_count_passed = favorable_count >= SUPPORT_FAVORABLE_COUNT_THRESHOLD
        support_passed = (
            attenuation_passed and fixed_cap_passed and favorable_count_passed
        )
        retained_half_or_more = max(0.0, fixed_log_median) >= (
            FUTILITY_RETENTION_FRACTION * coupled_log_median
        )
        fixed_40k_doubled = (
            endpoint == 40_000
            and fixed_endpoint_ratio_median >= FUTILITY_FIXED_40K_RATIO_THRESHOLD
        )
        endpoint_rows[str(endpoint)] = {
            "cohort_median_log_ratio_to_10k": {
                "coupled": coupled_log_median,
                "fixed_q": fixed_log_median,
            },
            "cohort_median_ratio_to_10k": {
                "coupled": _median(
                    [
                        _pressure(case, endpoint, weighting, "coupled")
                        / _pressure(case, BASELINE_K, weighting, "coupled")
                        for case in cases
                    ]
                ),
                "fixed_q": fixed_endpoint_ratio_median,
            },
            "coupled_cases_at_least_2x": coupled_ratio_count,
            "favorable_paired_reduction_count": favorable_count,
            "eligibility": {
                "median_coupled_log_at_least_log2": cliff_log_passed,
                "at_least_24_cases_coupled_at_least_2x": cliff_count_passed,
                "passed": endpoint_cliff_eligible,
            },
            "support": {
                "fixed_positive_log_at_most_quarter_coupled": attenuation_passed,
                "fixed_log_at_most_log1p25": fixed_cap_passed,
                "at_least_27_favorable_pairs": favorable_count_passed,
                "passed": support_passed,
            },
            "futility": {
                "fixed_positive_log_retains_at_least_half_coupled": (
                    retained_half_or_more
                ),
                "fixed_40k_over_10k_median_at_least_2x": fixed_40k_doubled,
                "triggered": retained_half_or_more or fixed_40k_doubled,
            },
            "area_coupled_nearly_flat": (
                weighting == "area_weighted"
                and coupled_log_median <= math.log(AREA_NEARLY_FLAT_RATIO_THRESHOLD)
            ),
        }

    eligible = common_eligibility["passed"] and baseline_comparable and cliff_eligible
    both_support = all(
        endpoint_rows[str(endpoint)]["support"]["passed"] for endpoint in ENDPOINTS
    )
    any_futility = any(
        endpoint_rows[str(endpoint)]["futility"]["triggered"] for endpoint in ENDPOINTS
    )
    if not eligible:
        outcome = "INELIGIBLE"
    elif both_support:
        outcome = "SUPPORTED"
    elif any_futility:
        outcome = "FUTILE"
    else:
        outcome = "MIXED"
    nearly_flat_endpoints = [
        endpoint
        for endpoint in ENDPOINTS
        if endpoint_rows[str(endpoint)]["area_coupled_nearly_flat"]
    ]
    failed_eligibility = list(common_eligibility["failed_checks"])
    if not baseline_comparable:
        failed_eligibility.append("baseline_fixed_over_coupled_median_in_[0.5,2]")
    failed_eligibility.extend(
        f"coupled_cliff_at_{endpoint}"
        for endpoint in ENDPOINTS
        if not endpoint_rows[str(endpoint)]["eligibility"]["passed"]
    )
    return {
        "weighting": weighting,
        "per_case_log_ratios": per_case_logs,
        "eligibility": {
            "passed": eligible,
            "common_stage0_passed": common_eligibility["passed"],
            "baseline_fixed_over_coupled_ratios": baseline_ratios,
            "baseline_fixed_over_coupled_median": baseline_ratio_median,
            "baseline_comparability_bounds": list(BASELINE_COMPARABILITY_BOUNDS),
            "baseline_comparable": baseline_comparable,
            "coupled_cliff_both_endpoints": cliff_eligible,
            "failed_checks": failed_eligibility,
        },
        "endpoints": endpoint_rows,
        "outcome": outcome,
        "both_endpoints_support": both_support if eligible else False,
        "any_endpoint_futility": any_futility if eligible else False,
        "area_coupled_nearly_flat_endpoints": nearly_flat_endpoints,
    }


def _case_verdict_rows(
    cases: list[dict[str, Any]],
    common_eligibility: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        case_id = case["case_id"]
        pressure_rows = [
            {
                "k": resolution["k"],
                "selection": resolution["selection"],
                "pressure_relative_l2": {
                    weighting: {
                        arm: resolution["metrics"][weighting][arm][
                            "pressure_relative_l2"
                        ]
                        for arm in ("coupled", "fixed_q")
                    }
                    for weighting in ("uniform", "area_weighted")
                },
                "ordered_secondary_diagnostics": {
                    arm: {
                        key: resolution["metrics"]["uniform"][arm][key]
                        for key in FULL_METRIC_KEYS
                        if key != "pressure_relative_l2"
                    }
                    for arm in ("coupled", "fixed_q")
                },
                "finite_checks_passed": resolution["finite_checks_passed"],
            }
            for resolution in case["resolutions"]
        ]
        rows.append(
            {
                "cohort_ordinal": case["cohort_ordinal"],
                "case_id": case_id,
                "reader_index": case["reader_index"],
                "n_master_cells": case["n_master_cells"],
                "historical_start": case["historical_start"],
                "source_identity": case["source_identity"],
                "historical_10k": case["historical_10k"],
                "fixed_q": case["fixed_q"],
                "s10k_reference": case["s10k_reference"],
                "q_support_ratios": common_eligibility["q_support_ratios"][case_id],
                "archived_replay": common_eligibility["archived_replay"]["per_case"][
                    case_id
                ],
                "centering_diagnostic": case["centers"],
                "resolution_rows": pressure_rows,
            }
        )
    return rows


def _overall_interpretation(
    uniform: dict[str, Any],
    area: dict[str, Any],
    *,
    centering_passed: bool,
) -> tuple[str, str, bool]:
    if not centering_passed:
        return (
            "BLOCKED_HQC_CENTERING",
            (
                "Resolve the preprocessing/centering discrepancy without selecting "
                "a centering path post hoc, then rerun the frozen panel."
            ),
            False,
        )
    if uniform["outcome"] == "INELIGIBLE":
        return (
            "INELIGIBLE_HQC_PANEL",
            "Repair or expand the instrument; do not reinterpret this as a result.",
            False,
        )
    if uniform["outcome"] == "FUTILE":
        return (
            "FUTILE_HQC_DOMINANT_MECHANISM",
            (
                "Together with the failed H4 coverage gate, redirect architecture "
                "work toward a fixed latent carrier or another source-to-latent "
                "representation; do not reopen covering."
            ),
            False,
        )
    if uniform["outcome"] == "MIXED":
        return (
            "MIXED_HQC",
            "Do not advance the matched-training study until the mixed mechanism is resolved.",
            False,
        )
    if area["outcome"] == "SUPPORTED":
        return (
            "SUPPORTED_HQC_DUAL_WEIGHTING",
            (
                "Run the pre-registered matched training study with independent "
                "source and target draws, then test a dual-support decoder."
            ),
            True,
        )
    nearly_flat_endpoints = set(area["area_coupled_nearly_flat_endpoints"])
    if nearly_flat_endpoints == set(ENDPOINTS):
        return (
            "SUPPORTED_HQC_UNIFORM_AREA_NEARLY_FLAT",
            (
                "Record the archived cliff as primarily a uniform-token objective "
                "artifact; do not claim dominance for the physically scored cliff."
            ),
            False,
        )
    if nearly_flat_endpoints:
        return (
            "SUPPORTED_HQC_UNIFORM_ONLY_METRIC_SPECIFIC",
            (
                "Only one area-weighted endpoint is nearly flat; treat the result "
                "as mixed/metric-specific and do not make a physical dual-support "
                "claim."
            ),
            False,
        )
    if area["outcome"] == "INELIGIBLE":
        return (
            "SUPPORTED_HQC_UNIFORM_AREA_INELIGIBLE",
            (
                "Repair or expand the area-weighted instrument before any "
                "physical dual-support claim."
            ),
            False,
        )
    return (
        "SUPPORTED_HQC_UNIFORM_ONLY_METRIC_SPECIFIC",
        (
            "Treat H-QC as metric-specific; it does not license a dual-support "
            "physical model claim."
        ),
        False,
    )


def build_verdict(artifact_paths: list[Path]) -> dict[str, Any]:
    """Load, validate, and aggregate the complete H-QC producer panel."""

    if not artifact_paths:
        raise VerdictInputError("H-QC requires at least one producer lane")
    resolved = [path.resolve() for path in artifact_paths]
    if len(set(resolved)) != len(resolved):
        raise VerdictInputError("H-QC input artifact paths must be distinct")
    inodes = []
    for path in artifact_paths:
        try:
            stat = path.stat()
        except OSError as error:
            raise VerdictInputError(f"cannot stat {path}: {error}") from error
        inodes.append((stat.st_dev, stat.st_ino))
    if len(set(inodes)) != len(inodes):
        raise VerdictInputError("H-QC input artifacts must not be aliases")

    lanes_by_ordinal = {}
    for path in artifact_paths:
        try:
            artifact_bytes = path.read_bytes()
            artifact = json.loads(artifact_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerdictInputError(f"cannot read {path}: {error}") from error
        lane = _validate_lane(
            path,
            _mapping(artifact, str(path)),
            hashlib.sha256(artifact_bytes).hexdigest(),
        )
        if lane["ordinal"] in lanes_by_ordinal:
            raise VerdictInputError(f"duplicate H-QC lane {lane['ordinal']}")
        lanes_by_ordinal[lane["ordinal"]] = lane

    lane_counts = {lane["count"] for lane in lanes_by_ordinal.values()}
    if len(lane_counts) != 1:
        raise VerdictInputError("H-QC lanes declare inconsistent lane counts")
    lane_count = lane_counts.pop()
    if set(lanes_by_ordinal) != set(range(lane_count)):
        raise VerdictInputError("H-QC lane ordinals must be complete from 0 to count-1")
    lanes = [lanes_by_ordinal[ordinal] for ordinal in range(lane_count)]
    npz_inodes = []
    for lane in lanes:
        npz_path = Path(lane["rescoring_validation"]["path"])
        stat = npz_path.stat()
        npz_inodes.append((stat.st_dev, stat.st_ino))
    if len(set(npz_inodes)) != len(npz_inodes):
        raise VerdictInputError("H-QC sibling NPZ artifacts must not be aliases")
    if set(npz_inodes) & set(inodes):
        raise VerdictInputError("H-QC JSON and NPZ artifacts must not alias")

    stable_provenance = {
        key: lanes[0]["provenance"][key]
        for key in (
            "producer_path",
            "producer_sha256",
            "config_sha256",
            "python",
            "platform",
            "versions",
            "hardware",
        )
    }
    for lane in lanes[1:]:
        observed = {key: lane["provenance"][key] for key in stable_provenance}
        if observed != stable_provenance:
            raise VerdictInputError(
                "H-QC lanes disagree on producer/config/environment provenance"
            )

    cases_by_ordinal = {}
    source_identities = set()
    q_identities = set()
    for lane in lanes:
        for case in lane["cases"]:
            ordinal = case["cohort_ordinal"]
            if ordinal in cases_by_ordinal:
                raise VerdictInputError(f"duplicate H-QC cohort ordinal {ordinal}")
            cases_by_ordinal[ordinal] = case
            source_tuple = tuple(case["source_identity"].values())
            if source_tuple in source_identities:
                raise VerdictInputError(
                    "H-QC cases contain duplicate source identities"
                )
            source_identities.add(source_tuple)
            q_hash = case["fixed_q"]["raw_cell_ids_sha256_int64"]
            if q_hash in q_identities:
                raise VerdictInputError(
                    "H-QC cases contain duplicate fixed-Q identities"
                )
            q_identities.add(q_hash)
    if set(cases_by_ordinal) != set(range(len(EXPECTED_CASES))):
        raise VerdictInputError("H-QC lanes do not contain the exact 36-case cohort")
    cases = [cases_by_ordinal[index] for index in range(len(EXPECTED_CASES))]

    common_eligibility = _common_eligibility(cases)
    uniform_panel = _metric_panel(cases, "uniform", common_eligibility)
    area_panel = _metric_panel(cases, "area_weighted", common_eligibility)
    centering_passed = all(case["centers"]["passed"] for case in cases)
    status, next_action, dual_weighting_supported = _overall_interpretation(
        uniform_panel,
        area_panel,
        centering_passed=centering_passed,
    )

    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "hypothesis": {
            "id": "H-QC",
            "statement": (
                "At fixed physical trace cells, changing cyclic source-cell count "
                "leaves pressure error approximately stable or convergent."
            ),
        },
        "decision_rule": {
            "frozen_before_deciding_output_on": PROTOCOL_DATE,
            "cohort_case_ids": list(EXPECTED_CASES),
            "resolutions": list(RESOLUTIONS),
            "baseline_k": BASELINE_K,
            "endpoints": list(ENDPOINTS),
            "thresholds": {
                "archived_mean_abs_tolerance": ARCHIVED_MEAN_ABS_TOLERANCE,
                "archived_case_abs_tolerance": ARCHIVED_CASE_ABS_TOLERANCE,
                "raw_frame_reconstruction_abs_tolerance": (
                    RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE
                ),
                "center_metric_relative_tolerance": (CENTER_METRIC_RELATIVE_TOLERANCE),
                "baseline_comparability_bounds": list(BASELINE_COMPARABILITY_BOUNDS),
                "coupled_cliff_ratio": CLIFF_RATIO_THRESHOLD,
                "coupled_cliff_case_count": CLIFF_CASE_COUNT_THRESHOLD,
                "support_attenuation_fraction": SUPPORT_ATTENUATION_FRACTION,
                "support_fixed_ratio": SUPPORT_FIXED_RATIO_THRESHOLD,
                "support_favorable_case_count": SUPPORT_FAVORABLE_COUNT_THRESHOLD,
                "futility_retention_fraction": FUTILITY_RETENTION_FRACTION,
                "futility_fixed_40k_ratio": FUTILITY_FIXED_40K_RATIO_THRESHOLD,
                "area_nearly_flat_ratio": AREA_NEARLY_FLAT_RATIO_THRESHOLD,
            },
            "metric_definitions": METRIC_DEFINITIONS,
        },
        "input_lanes": [
            {
                "lane": {"ordinal": lane["ordinal"], "count": lane["count"]},
                "input_artifact": lane["input_artifact"],
                "command": lane["provenance"]["command"],
                "rescoring_npz": {
                    "producer_path": lane["provenance"]["rescoring_npz_path"],
                    **lane["rescoring_validation"],
                },
                "case_ordinals": [case["cohort_ordinal"] for case in lane["cases"]],
            }
            for lane in lanes
        ],
        "common_stage0_eligibility": common_eligibility,
        "centering": {
            "passed": centering_passed,
            "failed_case_ids": [
                case["case_id"] for case in cases if not case["centers"]["passed"]
            ],
            "failure_blocks_hqc_without_center_selection": True,
        },
        "metric_panels": {
            "uniform": uniform_panel,
            "area_weighted": area_panel,
        },
        "cases": _case_verdict_rows(cases, common_eligibility),
        "uniform_hqc_supported": uniform_panel["outcome"] == "SUPPORTED",
        "dual_weighting_supported": dual_weighting_supported,
        "licensed_wording": (
            "query co-sampling dominates the physically scored cliff"
            if dual_weighting_supported
            else None
        ),
        "pre_registered_next_action": next_action,
        "limitations": [
            (
                "A pass establishes an evaluation-side mechanism; it does not show "
                "that training-time query co-sampling or cross-family transfer is solved."
            ),
            (
                "Ordered secondary diagnostics are reported but cannot rescue a "
                "failed pressure gate."
            ),
            (
                "The H4 two-case/four-start panel cannot add population support "
                "before this 36-case panel passes."
            ),
        ],
        "integrity": {
            "all_input_artifacts_hashed": True,
            "complete_disjoint_lane_ordinals": True,
            "complete_unique_36_case_cohort": True,
            "ordered_cyclic_selection_hashes_reconstructed": True,
            "fixed_q_prefix_hashes_verified_all_k": True,
            "sibling_npz_sha256_verified_all_lanes": True,
            "all_deciding_metrics_recomputed_from_npz": True,
            "all_center_diagnostics_recomputed_from_npz": True,
            "frozen_contract": FROZEN_CONTRACT,
            "producer_provenance": stable_provenance,
        },
        "provenance": {
            "script": {
                "path": str(script_path),
                "sha256": _sha256_file(script_path),
            },
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _parse_args() -> argparse.Namespace:
    example_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        metavar="LANE_JSON",
        help="Every completed H-QC producer-lane artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(example_root / "results" / f"phase1_hqc_verdict_{PROTOCOL_DATE}.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_output_path(args.output, args.artifacts)
    result = build_verdict(args.artifacts)
    _write_json_once(args.output, result)
    print(
        f"{result['status']} "
        f"uniform={result['metric_panels']['uniform']['outcome']} "
        f"area={result['metric_panels']['area_weighted']['outcome']} "
        f"artifact={args.output}"
    )


if __name__ == "__main__":
    main()
