# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Adjudicate the four-lane canonical-geometry validity experiment.

The producer emits one JSON/NPZ pair for each of four deterministic lanes.
This reducer verifies those artifacts independently and recomputes the
primary/fixed and primary/replay exact comparisons from persisted raw bytes.
It never averages across cases, resolutions, precisions, panels, or fields.

Producer and execution-source hashes are command-line inputs so each frozen
launch package binds the reducer to the exact staged producer and source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

LANE_SCHEMA_VERSION = 1
LANE_ARTIFACT_KIND = "hqc_canonical_geometry_full_cohort_validity_lane"
ADJUDICATION_SCHEMA_VERSION = 1
ADJUDICATION_ARTIFACT_KIND = "hqc_canonical_geometry_full_cohort_validity_adjudication"

LANE_COUNT = 4
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
RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
FIXED_QUERY_K = 2_500
PRECISIONS = ("bfloat16", "float32")
QUERY_PANELS = ("coupled_s_k", "fixed_id_prefix_s2500")
PATHS = ("primary", "fixed", "primary_replay")
FIELDS = ("pressure", "wss")
GEOMETRY_FIELDS = ("points", "centroids", "areas", "normals")
STORAGE_FIELDS = (
    "points",
    "cells",
    "centroids",
    "areas",
    "normals",
    "center",
    "reference_length",
)
DECODE_CHECKS = (
    "canonical_queries_exact",
    "encoded_center_is_raw_positive_zero",
    "encoded_reference_length_is_exact_positive_one",
    "trace_query_count_exact",
)
CONSTRUCTION_CHECKS = (
    "cells",
    "points",
    "centroids",
    "areas",
    "normals",
    "physical_center",
    "physical_length",
    "model_reference_length",
)
TOPOLOGY_CHECKS = (
    "primary_matches_selected",
    "fixed_matches_selected",
    "primary_matches_fixed",
)
UNIT_ABS_TOLERANCE = 1.0e-6
CENTER_ABS_TOLERANCE = 1.0e-6
EXPECTED_PHYSICAL_LENGTH = 5.0
EXPECTED_MODEL_REFERENCE_LENGTH = 8.0
EXPECTED_CASE_METADATA = (
    (21, 17_504_739, 14_045_027),
    (33, 16_380_547, 14_700_754),
    (51, 15_789_064, 9_195_926),
    (55, 18_007_064, 4_452_828),
    (77, 19_404_150, 6_369_582),
    (79, 18_792_923, 1_320_415),
    (88, 14_634_570, 10_215_595),
    (92, 14_932_664, 7_635_018),
    (107, 18_934_869, 16_494_923),
    (114, 17_796_743, 15_267_620),
    (136, 15_024_109, 3_789_927),
    (185, 18_857_430, 10_967_997),
    (186, 16_922_213, 5_453_831),
    (212, 15_063_884, 4_943_208),
    (221, 18_022_481, 16_998_850),
    (237, 16_199_351, 15_062_581),
    (285, 18_958_141, 5_352_845),
    (298, 19_519_305, 11_721_918),
    (300, 16_887_630, 11_083_431),
    (318, 16_222_090, 15_155_572),
    (319, 16_294_644, 13_228_777),
    (329, 16_591_548, 1_346_462),
    (340, 14_561_784, 12_777_694),
    (346, 16_588_938, 13_358_519),
    (351, 17_738_132, 365_298),
    (354, 15_747_949, 1_091_720),
    (362, 17_809_120, 8_840_407),
    (391, 16_443_085, 11_669_428),
    (394, 18_343_677, 15_504_945),
    (395, 19_780_049, 19_757_508),
    (404, 16_648_431, 16_079_300),
    (416, 16_063_459, 6_463_342),
    (418, 17_847_065, 191_824),
    (423, 15_715_663, 11_592_670),
    (453, 16_516_082, 2_240_523),
    (469, 17_188_261, 4_374_650),
)

EXPECTED_CONTRACT = {
    "canonical_construction": (
        "float64 raw geometry -> physical area center -> divide by "
        "L_ref*model_reference_length -> one float32 cast"
    ),
    "canonical_full_fields": list(GEOMETRY_FIELDS),
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
}
EXPECTED_REQUIRED_GATES = [
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
]
FULL_GATE_CRITERION = (
    "primary-versus-fixed pressure and WSS raw-byte exact over the full "
    "trace for every lane case, resolution, and precision"
)

VALID_LANE_STATUS = "VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_LANE"
INVALID_LANE_STATUS = "INVALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_LANE"
VALID_ADJUDICATION_STATUS = "VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT"
INVALID_ADJUDICATION_STATUS = "INVALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT"
INCOMPLETE_ADJUDICATION_STATUS = (
    "INCOMPLETE_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT"
)
EXACT_OUTCOME = "CANONICAL_FULL_VALIDITY_PASS"
REFUTED_OUTCOME = "CANONICAL_FULL_VALIDITY_REFUTED"
INVALID_DIAGNOSTIC = "INVALID_DIAGNOSTIC"
INVALID_OUTCOME = "INVALID_COHORT"
INCOMPLETE_OUTCOME = "INCOMPLETE_COHORT"

EXPECTED_CANONICAL_HELPER_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)
EXPECTED_FROZEN_HQC_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_STABLE_INPUT_HASHES = {
    "dataset_manifest": (
        "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
    ),
    "dataset_config": (
        "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
    ),
    "resolved_config": (
        "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
    ),
    "model_checkpoint": (
        "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
    ),
    "normalization_stats": (
        "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
    ),
    "training_state": (
        "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
    ),
    "job305691_anchor_json": (
        "09e336442881f0641c14c91c17dd80ac440d474f996925cf79f2174bc4cacd88"
    ),
    "job305691_anchor_npz": (
        "edee836e0cc5c66690276e6787496cbd6b81fb08decb7d54ecf1f36b333ddc9f"
    ),
}
FORBIDDEN_KEY_TOKENS = (
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
HEX_DIGITS = frozenset("0123456789abcdef")


class LaneUnavailable(ValueError):
    """A lane cannot be read and therefore cannot support a cohort verdict."""


class LaneInvalid(ValueError):
    """A readable lane violates the experiment or artifact contract."""


@dataclass(frozen=True)
class ExpectedProvenance:
    """Hashes frozen by the launch package rather than the reducer source."""

    lane_producer_sha256: str
    source_tree_sha256: str
    geometry_input_manifest_sha256: str

    def validate(self) -> None:
        for label, value in (
            ("lane producer", self.lane_producer_sha256),
            ("source tree", self.source_tree_sha256),
            ("geometry input manifest", self.geometry_input_manifest_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Expected {label} SHA-256 is not frozen")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaneInvalid(message)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _array_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _read_regular_file_bytes(
    path: Path,
    label: str,
) -> tuple[bytes, tuple[int, int]]:
    """Read one regular file without following a symlink in any path component."""

    lexical = Path(os.path.abspath(path))
    parts = lexical.parts
    if not parts or parts[0] != os.sep:
        raise LaneUnavailable(f"{label} does not resolve to an absolute path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        leaf_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = leaf_descriptor
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LaneUnavailable(f"{label} is not a regular file")
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
            raise LaneUnavailable(f"{label} changed while it was being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise LaneUnavailable(f"{label} size changed while it was being read")
        return payload, identity
    except LaneUnavailable:
        raise
    except OSError as error:
        raise LaneUnavailable(
            f"{label} is missing, unreadable, or traverses a symlink"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_verified_artifact_bytes(
    path: Path,
    label: str,
) -> tuple[bytes, str]:
    lexical = Path(os.path.abspath(path))
    payload, _ = _read_regular_file_bytes(lexical, label)
    sidecar_payload, _ = _read_regular_file_bytes(
        lexical.with_name(f"{lexical.name}.sha256"),
        f"{lexical.name} sidecar",
    )
    try:
        line = sidecar_payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise LaneUnavailable(f"Unreadable sidecar for {lexical.name}") from error
    parts = line.removesuffix("\n").split("  ")
    if (
        not line.endswith("\n")
        or line.count("\n") != 1
        or len(parts) != 2
        or not _is_sha256(parts[0])
        or parts[1] != lexical.name
    ):
        raise LaneUnavailable(f"Malformed sidecar for {lexical.name}")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != parts[0]:
        raise LaneUnavailable(f"Sidecar digest differs for {lexical.name}")
    if sidecar_payload != f"{observed}  {lexical.name}\n".encode("ascii"):
        raise LaneUnavailable(f"Non-canonical sidecar for {lexical.name}")
    return payload, observed


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaneUnavailable(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise LaneUnavailable(f"JSON contains non-finite token {value!r}")


def _parse_finite_float(value: str) -> float:
    decimal = Decimal(value)
    parsed = float(decimal)
    if not math.isfinite(parsed) or (decimal != 0 and parsed == 0.0):
        raise LaneUnavailable(f"JSON float is outside finite binary64: {value}")
    return parsed


def _load_json_bytes(payload: bytes, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except LaneUnavailable:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaneUnavailable(f"Unreadable JSON artifact {name}") from error
    if not isinstance(value, Mapping):
        raise LaneInvalid(f"{name} top level is not an object")
    return value


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in FORBIDDEN_KEY_TOKENS):
                found.append(path)
            found.extend(_forbidden_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_paths(nested, f"{prefix}[{index}]"))
    return found


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} key set differs")


def _expected_lane_cases(lane_ordinal: int) -> tuple[tuple[int, str], ...]:
    return tuple(
        (ordinal, case_id)
        for ordinal, case_id in enumerate(CASE_IDS)
        if ordinal % LANE_COUNT == lane_ordinal
    )


def _unit_prefix(ordinal: int, case_id: str, resolution: int) -> str:
    return f"case_{ordinal:02d}_{case_id}__k{resolution:05d}"


def _unit_array_names(ordinal: int, case_id: str, resolution: int) -> tuple[str, ...]:
    prefix = _unit_prefix(ordinal, case_id, resolution)
    names = [
        f"{prefix}__selected_cell_ids_int64",
        f"{prefix}__canonical_cells_int64",
        f"{prefix}__canonical_points_float32",
        f"{prefix}__canonical_centroids_float32",
        f"{prefix}__canonical_areas_float32",
        f"{prefix}__canonical_normals_float32",
    ]
    names.extend(
        f"{prefix}__{precision}_canonical_full_{panel}_{path}_{field}"
        for precision in PRECISIONS
        for panel in QUERY_PANELS
        for path in PATHS
        for field in FIELDS
    )
    return tuple(names)


def _expected_array_names(lane_ordinal: int) -> tuple[str, ...]:
    return tuple(
        name
        for ordinal, case_id in _expected_lane_cases(lane_ordinal)
        for resolution in RESOLUTIONS
        for name in _unit_array_names(ordinal, case_id, resolution)
    )


def _expected_anchor_array_names() -> tuple[str, ...]:
    names: list[str] = []
    for ordinal, case_id in enumerate(CASE_IDS[:4]):
        prefix = f"case_{ordinal:02d}_{case_id}"
        names.extend(
            f"{prefix}__{name}"
            for name in (
                "selected_cell_ids_int64",
                "canonical_cells_int64",
                "canonical_points_float32",
                "canonical_centroids_float32",
                "canonical_areas_float32",
                "canonical_normals_float32",
            )
        )
        for precision in PRECISIONS:
            names.extend(
                f"{prefix}__{precision}_canonical_{mode}_{path}_{field}"
                for mode in ("derived", "full")
                for path in PATHS
                for field in FIELDS
            )
            names.extend(
                f"{prefix}__{precision}_historical_{path}_{field}"
                for path in ("primary", "fixed")
                for field in FIELDS
            )
            names.extend(
                f"{prefix}__{precision}_historical_model_{path}_source_{field}"
                for path in ("primary", "fixed")
                for field in GEOMETRY_FIELDS
            )
    return tuple(names)


def _load_npz_bytes(
    payload: bytes,
    name: str,
    expected_names: Sequence[str],
) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [info.filename for info in archive.infolist()]
            expected_members = {f"{name}.npy" for name in expected_names}
            if len(members) != len(set(members)):
                raise LaneInvalid(f"{name} contains duplicate ZIP members")
            if set(members) != expected_members:
                raise LaneInvalid(f"{name} array key set differs")
            if any(
                "/" in member or "\\" in member or not member.endswith(".npy")
                for member in members
            ):
                raise LaneInvalid(f"{name} contains an invalid member name")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise LaneUnavailable(f"{name} has a corrupt ZIP member: {bad_member}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(expected_names):
                raise LaneInvalid(f"{name} NumPy array key set differs")
            return {name: np.array(archive[name], copy=True) for name in expected_names}
    except (LaneInvalid, LaneUnavailable):
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise LaneUnavailable(f"Unreadable NPZ artifact {name}") from error


def _validate_manifest(
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    manifest = _mapping(summary.get("npz_array_manifest"), "NPZ array manifest")
    _require(set(manifest) == set(arrays), "NPZ array manifest key set differs")
    for name, array in arrays.items():
        record = _mapping(manifest.get(name), f"Array manifest record {name}")
        _exact_keys(record, {"shape", "dtype", "sha256"}, f"Array manifest {name}")
        _require(
            record.get("shape") == list(array.shape),
            f"Array manifest shape differs for {name}",
        )
        _require(
            record.get("dtype") == str(array.dtype),
            f"Array manifest dtype differs for {name}",
        )
        _require(
            record.get("sha256") == _sha256_array(array),
            f"Array manifest hash differs for {name}",
        )


def _validate_provenance(
    summary: Mapping[str, Any],
    *,
    npz_sha256: str,
    expected: ExpectedProvenance,
) -> None:
    provenance = _mapping(summary.get("provenance"), "Lane provenance")
    _exact_keys(
        provenance,
        {
            "command",
            "diagnostic_script_path",
            "diagnostic_script_sha256",
            "canonical_helper_path",
            "canonical_helper_sha256",
            "frozen_producer_path",
            "frozen_producer_sha256",
            "import_provenance",
            "source_tree_manifest_sha256",
            "input_hashes",
            "npz_path",
            "npz_sha256",
            "slurm_job_id",
            "python",
            "platform",
            "numpy",
            "torch",
            "hardware",
        },
        "Lane provenance",
    )
    _require(
        provenance.get("diagnostic_script_sha256") == expected.lane_producer_sha256,
        "Lane producer hash differs",
    )
    _require(
        provenance.get("source_tree_manifest_sha256") == expected.source_tree_sha256,
        "Execution source-tree hash differs",
    )
    _require(
        provenance.get("canonical_helper_sha256") == EXPECTED_CANONICAL_HELPER_SHA256,
        "Canonical helper hash differs",
    )
    _require(
        provenance.get("frozen_producer_sha256") == EXPECTED_FROZEN_HQC_PRODUCER_SHA256,
        "Frozen H-QC producer hash differs",
    )
    _require(provenance.get("npz_sha256") == npz_sha256, "NPZ provenance hash differs")
    input_hashes = _mapping(provenance.get("input_hashes"), "Input hashes")
    expected_inputs = {
        **EXPECTED_STABLE_INPUT_HASHES,
        "geometry_input_manifest": expected.geometry_input_manifest_sha256,
    }
    _require(dict(input_hashes) == expected_inputs, "Input provenance hashes differ")


def _validate_scope(
    summary: Mapping[str, Any],
    *,
    lane_ordinal: int,
) -> None:
    scope = _mapping(summary.get("scientific_scope"), "Scientific scope")
    _exact_keys(
        scope,
        {
            "case_ids",
            "resolutions",
            "precisions",
            "licensing_field_tensor_comparisons_per_lane",
            "licensing_field_tensor_comparisons_full_cohort",
            "deduplicated_panel_field_summaries_per_lane",
            "deduplicated_panel_field_summaries_full_cohort",
            "emitted_panel_field_records_per_lane",
            "emitted_panel_field_records_full_cohort",
            "prefix_summaries_are_independent_decisions",
            "supervision_arrays_indexed",
            "supervision_files_opened_by_model_producer",
            "raw_dataset_sample_loader_called",
            "geometry_only_memmap_allowlist_applied",
            "synthetic_placeholders_stripped_before_model",
            "hqc_decision_statistics_computed",
            "may_not_be_used_as_hqc_verdict_output",
        },
        "Scientific scope",
    )
    expected_cases = [case_id for _, case_id in _expected_lane_cases(lane_ordinal)]
    _require(scope.get("case_ids") == expected_cases, "Lane case scope differs")
    _require(scope.get("resolutions") == list(RESOLUTIONS), "Resolution scope differs")
    _require(scope.get("precisions") == list(PRECISIONS), "Precision scope differs")
    _require(
        scope.get("supervision_arrays_indexed") is False,
        "Lane does not attest target-free execution",
    )
    _require(
        scope.get("hqc_decision_statistics_computed") is False,
        "Lane computed out-of-scope H-QC statistics",
    )
    _require(
        scope.get("prefix_summaries_are_independent_decisions") is False,
        "Fixed-prefix summaries became independent decisions",
    )
    comparisons_per_lane = (
        len(expected_cases) * len(RESOLUTIONS) * len(PRECISIONS) * len(FIELDS)
    )
    _require(
        scope.get("licensing_field_tensor_comparisons_per_lane")
        == comparisons_per_lane,
        "Lane licensing-comparison count differs",
    )
    _require(
        scope.get("licensing_field_tensor_comparisons_full_cohort")
        == comparisons_per_lane * LANE_COUNT,
        "Full-cohort licensing-comparison count differs",
    )
    deduplicated_per_lane = (
        len(expected_cases)
        * len(PRECISIONS)
        * len(FIELDS)
        * (1 + 2 * (len(RESOLUTIONS) - 1))
    )
    emitted_per_lane = (
        len(expected_cases)
        * len(RESOLUTIONS)
        * len(PRECISIONS)
        * len(QUERY_PANELS)
        * len(FIELDS)
    )
    _require(
        scope.get("deduplicated_panel_field_summaries_per_lane")
        == deduplicated_per_lane
        and scope.get("deduplicated_panel_field_summaries_full_cohort")
        == deduplicated_per_lane * LANE_COUNT,
        "Deduplicated panel-summary count differs",
    )
    _require(
        scope.get("emitted_panel_field_records_per_lane") == emitted_per_lane
        and scope.get("emitted_panel_field_records_full_cohort")
        == emitted_per_lane * LANE_COUNT,
        "Emitted panel-record count differs",
    )
    for key in (
        "supervision_files_opened_by_model_producer",
        "raw_dataset_sample_loader_called",
    ):
        _require(scope.get(key) is False, f"{key} target-free attestation differs")
    for key in (
        "geometry_only_memmap_allowlist_applied",
        "synthetic_placeholders_stripped_before_model",
        "may_not_be_used_as_hqc_verdict_output",
    ):
        _require(scope.get(key) is True, f"{key} target-free attestation differs")


def _finite_number(value: Any, label: str) -> float:
    _require(
        type(value) in {int, float} and math.isfinite(float(value)),
        f"{label} is not a finite number",
    )
    return float(value)


def _validate_true_bool_mapping(
    value: Any,
    expected_keys: Sequence[str],
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    _exact_keys(mapping, set(expected_keys), label)
    for key in expected_keys:
        _require(mapping.get(key) is True, f"{label}.{key} is not true")
    return mapping


def _relative_unit_array_names() -> tuple[str, ...]:
    names = [
        "selected_cell_ids_int64",
        "canonical_cells_int64",
        "canonical_points_float32",
        "canonical_centroids_float32",
        "canonical_areas_float32",
        "canonical_normals_float32",
    ]
    names.extend(
        f"{precision}_canonical_full_{panel}_{path}_{field}"
        for precision in PRECISIONS
        for panel in QUERY_PANELS
        for path in PATHS
        for field in FIELDS
    )
    return tuple(names)


def _validate_difference(
    value: Any,
    *,
    expected_shape: list[int],
    label: str,
) -> Mapping[str, Any]:
    difference = _mapping(value, label)
    _exact_keys(
        difference,
        {
            "shape",
            "left_dtype",
            "right_dtype",
            "exact",
            "nonzero_count",
            "maximum_absolute_difference",
            "relative_l2_difference",
        },
        label,
    )
    _require(difference.get("shape") == expected_shape, f"{label} shape differs")
    for key in ("left_dtype", "right_dtype"):
        dtype = difference.get(key)
        _require(
            dtype == "torch.float32",
            f"{label} {key} differs",
        )
    _require(type(difference.get("exact")) is bool, f"{label} exact flag differs")
    nonzero_count = difference.get("nonzero_count")
    _require(
        type(nonzero_count) is int and nonzero_count >= 0,
        f"{label} nonzero count differs",
    )
    maximum = _finite_number(
        difference.get("maximum_absolute_difference"),
        f"{label} maximum absolute difference",
    )
    relative = _finite_number(
        difference.get("relative_l2_difference"),
        f"{label} relative L2 difference",
    )
    _require(maximum >= 0.0 and relative >= 0.0, f"{label} has a negative metric")
    if difference.get("exact") is True:
        _require(
            nonzero_count == 0 and maximum == 0.0 and relative == 0.0,
            f"{label} exact flag contradicts its metrics",
        )
    return difference


def _validate_anchor_replay(
    value: Any,
    *,
    ordinal: int,
    resolution: int,
    label: str,
) -> None:
    anchor = _mapping(value, label)
    required = ordinal < 4 and resolution == RESOLUTIONS[0]
    expected_keys = {"required", "passed", "compared_arrays"}
    if required:
        expected_keys.add("comparisons")
    _exact_keys(anchor, expected_keys, label)
    _require(anchor.get("required") is required, f"{label} requirement differs")
    _require(anchor.get("passed") is True, f"{label} did not pass")
    expected_count = len(_relative_unit_array_names()) if required else 0
    _require(
        anchor.get("compared_arrays") == expected_count,
        f"{label} comparison count differs",
    )
    if required:
        _validate_true_bool_mapping(
            anchor.get("comparisons"),
            _relative_unit_array_names(),
            f"{label} comparisons",
        )


def _validate_bundle_record(value: Any, label: str) -> None:
    bundle = _mapping(value, label)
    _exact_keys(
        bundle,
        {
            "passed",
            "checks",
            "shape_checks",
            "finite_checks",
            "maximum_unit_deviation",
            "maximum_area_center_deviation",
        },
        label,
    )
    _require(bundle.get("passed") is True, f"{label} did not pass")
    _validate_true_bool_mapping(
        bundle.get("checks"),
        (
            "shapes",
            "topology",
            "finite",
            "positive_areas",
            "unit_normals",
            "area_centered",
        ),
        f"{label} checks",
    )
    _validate_true_bool_mapping(
        bundle.get("shape_checks"),
        ("points", "cells", "centroids", "areas", "normals"),
        f"{label} shape checks",
    )
    _validate_true_bool_mapping(
        bundle.get("finite_checks"),
        GEOMETRY_FIELDS,
        f"{label} finite checks",
    )
    unit_deviation = _finite_number(
        bundle.get("maximum_unit_deviation"),
        f"{label} maximum unit deviation",
    )
    center_deviation = _finite_number(
        bundle.get("maximum_area_center_deviation"),
        f"{label} maximum area-center deviation",
    )
    _require(
        0.0 <= unit_deviation <= UNIT_ABS_TOLERANCE,
        f"{label} unit-normal deviation exceeds tolerance",
    )
    _require(
        0.0 <= center_deviation <= CENTER_ABS_TOLERANCE,
        f"{label} area-center deviation exceeds tolerance",
    )


def _validate_panel_record(
    value: Any,
    *,
    panel: str,
    query_count: int,
    label: str,
) -> None:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "query_count",
            "source",
            "primary_fixed_difference",
            "primary_replay_difference",
            "primary_replay_exact",
            "comparison_gate",
            "validity_passed",
        },
        label,
    )
    _require(record.get("query_count") == query_count, f"{label} query count differs")
    expected_source = (
        "single_public_decode"
        if panel == "coupled_s_k"
        else "first_2500_rows_of_single_public_decode"
    )
    _require(record.get("source") == expected_source, f"{label} source differs")
    expected_shapes = {
        "pressure": [query_count],
        "wss": [query_count, 3],
    }
    primary_fixed = _mapping(
        record.get("primary_fixed_difference"),
        f"{label} primary/fixed difference",
    )
    primary_replay = _mapping(
        record.get("primary_replay_difference"),
        f"{label} primary/replay difference",
    )
    _exact_keys(primary_fixed, set(FIELDS), f"{label} primary/fixed difference")
    _exact_keys(primary_replay, set(FIELDS), f"{label} primary/replay difference")
    for field in FIELDS:
        _validate_difference(
            primary_fixed[field],
            expected_shape=expected_shapes[field],
            label=f"{label} primary/fixed {field}",
        )
        replay_difference = _validate_difference(
            primary_replay[field],
            expected_shape=expected_shapes[field],
            label=f"{label} primary/replay {field}",
        )
        _require(
            replay_difference.get("exact") is True,
            f"{label} primary/replay {field} is not exact",
        )
    _require(record.get("primary_replay_exact") is True, f"{label} replay failed")
    gate = _mapping(record.get("comparison_gate"), f"{label} comparison gate")
    _exact_keys(
        gate,
        {"criterion", "passed", "controls_candidate_advance"},
        f"{label} comparison gate",
    )
    _require(
        gate.get("criterion") == "fieldwise_bitwise_exact",
        f"{label} comparison criterion differs",
    )
    _require(type(gate.get("passed")) is bool, f"{label} comparison gate differs")
    _require(
        gate.get("controls_candidate_advance") is (panel == "coupled_s_k"),
        f"{label} candidate-control flag differs",
    )
    _require(record.get("validity_passed") is True, f"{label} validity failed")


def _validate_full_probe(
    value: Any,
    *,
    resolution: int,
    label: str,
) -> None:
    full = _mapping(value, label)
    _exact_keys(
        full,
        {
            "mode",
            "injected_geometry_exact",
            "injected_geometry_exact_passed",
            "authoritative_storage_identity",
            "authoritative_storage_identity_passed",
            "canonical_decode_contract",
            "canonical_decode_contract_passed",
            "query_panels",
            "fixed_id_prefix_matches_coupled_rows",
            "fixed_id_prefix_matches_coupled_rows_passed",
            "comparison_gate",
            "validity_passed",
        },
        label,
    )
    _require(full.get("mode") == "canonical_full_public_api", f"{label} mode differs")
    injection = _mapping(
        full.get("injected_geometry_exact"),
        f"{label} injected geometry",
    )
    storage = _mapping(
        full.get("authoritative_storage_identity"),
        f"{label} authoritative storage",
    )
    decode = _mapping(full.get("canonical_decode_contract"), f"{label} decode")
    _exact_keys(injection, set(PATHS), f"{label} injected geometry")
    _exact_keys(storage, set(PATHS), f"{label} authoritative storage")
    _exact_keys(decode, set(PATHS), f"{label} decode")
    for path in PATHS:
        _validate_true_bool_mapping(
            injection[path],
            GEOMETRY_FIELDS,
            f"{label} injected geometry {path}",
        )
        _validate_true_bool_mapping(
            storage[path],
            STORAGE_FIELDS,
            f"{label} authoritative storage {path}",
        )
        _validate_true_bool_mapping(
            decode[path],
            DECODE_CHECKS,
            f"{label} decode {path}",
        )
    for key in (
        "injected_geometry_exact_passed",
        "authoritative_storage_identity_passed",
        "canonical_decode_contract_passed",
    ):
        _require(full.get(key) is True, f"{label} {key} differs")

    panels = _mapping(full.get("query_panels"), f"{label} query panels")
    _exact_keys(panels, set(QUERY_PANELS), f"{label} query panels")
    for panel in QUERY_PANELS:
        _validate_panel_record(
            panels[panel],
            panel=panel,
            query_count=resolution if panel == "coupled_s_k" else FIXED_QUERY_K,
            label=f"{label} panel {panel}",
        )

    prefix = _mapping(
        full.get("fixed_id_prefix_matches_coupled_rows"),
        f"{label} prefix slicing",
    )
    _exact_keys(prefix, set(PATHS), f"{label} prefix slicing")
    for path in PATHS:
        _validate_true_bool_mapping(
            prefix[path],
            FIELDS,
            f"{label} prefix slicing {path}",
        )
    _require(
        full.get("fixed_id_prefix_matches_coupled_rows_passed") is True,
        f"{label} prefix slicing rollup differs",
    )
    comparison_gate = _mapping(
        full.get("comparison_gate"),
        f"{label} comparison gate",
    )
    _exact_keys(
        comparison_gate,
        {"criterion", "passed"},
        f"{label} comparison gate",
    )
    _require(
        comparison_gate.get("criterion") == "whole_trace_fieldwise_bitwise_exact",
        f"{label} comparison criterion differs",
    )
    _require(
        type(comparison_gate.get("passed")) is bool,
        f"{label} comparison result differs",
    )
    _require(full.get("validity_passed") is True, f"{label} validity failed")


def _validate_case_structure(
    summary: Mapping[str, Any],
    *,
    lane_ordinal: int,
) -> tuple[list[Mapping[str, Any]], bool]:
    cases = summary.get("cases")
    _require(isinstance(cases, list), "Lane cases are not a list")
    expected_cases = _expected_lane_cases(lane_ordinal)
    _require(len(cases) == len(expected_cases), "Lane case count differs")
    all_full_passed = True
    for case, (ordinal, case_id) in zip(cases, expected_cases, strict=True):
        case = _mapping(case, f"Case {case_id}")
        _exact_keys(
            case,
            {
                "case_id",
                "cohort_ordinal",
                "reader_index",
                "resolutions",
                "validity_passed",
                "decision_gates",
                "decision_outcome",
            },
            f"Case {case_id}",
        )
        _require(case.get("case_id") == case_id, f"Case order differs at {case_id}")
        _require(
            case.get("cohort_ordinal") == ordinal,
            f"Cohort ordinal differs for {case_id}",
        )
        _require(
            case.get("reader_index") == EXPECTED_CASE_METADATA[ordinal][0],
            f"Reader index differs for {case_id}",
        )
        resolutions = case.get("resolutions")
        _require(
            isinstance(resolutions, list) and len(resolutions) == len(RESOLUTIONS),
            f"Resolution order differs for {case_id}",
        )
        case_full_passed = True
        for row, expected_resolution in zip(resolutions, RESOLUTIONS, strict=True):
            row = _mapping(row, f"{case_id} resolution")
            _exact_keys(
                row,
                {
                    "case_id",
                    "cohort_ordinal",
                    "reader_index",
                    "resolution",
                    "canonical_frame",
                    "historical_centers",
                    "validity",
                    "precision_probes",
                    "validity_passed",
                    "decision_gates",
                    "decision_outcome",
                },
                f"{case_id} resolution {expected_resolution}",
            )
            _require(row.get("case_id") == case_id, "Resolution case ID differs")
            _require(
                row.get("cohort_ordinal") == ordinal,
                "Resolution cohort ordinal differs",
            )
            _require(
                row.get("reader_index") == EXPECTED_CASE_METADATA[ordinal][0],
                f"Resolution reader index differs for {case_id}",
            )
            _require(
                row.get("resolution") == expected_resolution,
                f"Resolution order differs for {case_id}",
            )
            canonical_frame = _mapping(
                row.get("canonical_frame"),
                f"{case_id} canonical frame",
            )
            _exact_keys(
                canonical_frame,
                {
                    "construction",
                    "physical_center_float64",
                    "physical_length",
                    "model_reference_length",
                    "effective_physical_length",
                    "queries",
                },
                f"{case_id} canonical frame",
            )
            _require(
                canonical_frame.get("construction")
                == (
                    "raw selected coordinates promoted to float64; physical "
                    "area-weighted center removed; coherent triangle geometry "
                    "divided by L_ref*model_reference_length; one float32 cast"
                ),
                f"Canonical construction description differs for {case_id}",
            )
            physical_center = canonical_frame.get("physical_center_float64")
            _require(
                isinstance(physical_center, list) and len(physical_center) == 3,
                f"Physical center contract differs for {case_id}",
            )
            for index, component in enumerate(physical_center):
                _finite_number(component, f"{case_id} physical center {index}")
            _require(
                canonical_frame.get("physical_length") == EXPECTED_PHYSICAL_LENGTH
                and canonical_frame.get("model_reference_length")
                == EXPECTED_MODEL_REFERENCE_LENGTH
                and canonical_frame.get("effective_physical_length")
                == EXPECTED_PHYSICAL_LENGTH * EXPECTED_MODEL_REFERENCE_LENGTH
                and canonical_frame.get("queries") == "canonical_trace_centroids",
                f"Canonical frame scale contract differs for {case_id}",
            )
            centers = _mapping(
                row.get("historical_centers"),
                f"{case_id} historical centers",
            )
            _exact_keys(
                centers,
                {
                    "primary_point_mean_float32",
                    "fixed_s10000_point_mean_float32",
                },
                f"{case_id} historical centers",
            )
            for center_name, center in centers.items():
                _require(
                    isinstance(center, list) and len(center) == 3,
                    f"{case_id} {center_name} contract differs",
                )
                for index, component in enumerate(center):
                    _finite_number(component, f"{case_id} {center_name} {index}")

            validity = _mapping(row.get("validity"), "Resolution validity")
            _exact_keys(
                validity,
                {
                    "canonical_bundle",
                    "canonical_construction_replay",
                    "canonical_construction_replay_passed",
                    "historical_path_topology",
                    "historical_path_topology_passed",
                    "fixed_q_is_exact_source_prefix",
                    "job305691_anchor_replay",
                    "model_local_data_stripped",
                    "model_probes_executed",
                },
                f"{case_id} resolution validity",
            )
            _validate_bundle_record(
                validity.get("canonical_bundle"),
                f"{case_id} K={expected_resolution} canonical bundle",
            )
            _validate_true_bool_mapping(
                validity.get("canonical_construction_replay"),
                CONSTRUCTION_CHECKS,
                f"{case_id} K={expected_resolution} construction replay",
            )
            for key in (
                "canonical_construction_replay_passed",
                "historical_path_topology_passed",
                "fixed_q_is_exact_source_prefix",
                "model_local_data_stripped",
                "model_probes_executed",
            ):
                _require(
                    validity.get(key) is True,
                    f"{case_id} K={expected_resolution} {key} differs",
                )
            _validate_true_bool_mapping(
                validity.get("historical_path_topology"),
                TOPOLOGY_CHECKS,
                f"{case_id} K={expected_resolution} historical topology",
            )
            _validate_anchor_replay(
                validity.get("job305691_anchor_replay"),
                ordinal=ordinal,
                resolution=expected_resolution,
                label=f"{case_id} K={expected_resolution} anchor replay",
            )

            probes = _mapping(row.get("precision_probes"), "Precision probes")
            _require(
                set(probes) == set(PRECISIONS),
                f"Precision order or set differs for {case_id}",
            )
            row_full_passed = True
            for precision in PRECISIONS:
                probe = _mapping(probes[precision], f"{precision} probe")
                _exact_keys(
                    probe,
                    {
                        "precision",
                        "canonical_full_public_api",
                        "validity_passed",
                        "decision_gates",
                    },
                    f"{case_id} K={expected_resolution} {precision} probe",
                )
                _require(
                    probe.get("precision") == precision,
                    f"Precision label differs for {case_id}",
                )
                full = _mapping(
                    probe.get("canonical_full_public_api"),
                    "Canonical-full probe",
                )
                _validate_full_probe(
                    full,
                    resolution=expected_resolution,
                    label=f"{case_id} K={expected_resolution} {precision}",
                )
                _require(
                    probe.get("validity_passed") is True,
                    f"{case_id} K={expected_resolution} {precision} validity failed",
                )
                precision_gates = _mapping(
                    probe.get("decision_gates"),
                    f"{case_id} K={expected_resolution} {precision} decision gates",
                )
                _exact_keys(
                    precision_gates,
                    {"full_passed"},
                    f"{case_id} K={expected_resolution} {precision} decision gates",
                )
                _require(
                    precision_gates.get("full_passed")
                    is full["comparison_gate"]["passed"],
                    f"{case_id} K={expected_resolution} precision rollup differs",
                )
                row_full_passed = (
                    bool(precision_gates["full_passed"]) and row_full_passed
                )
            row_gates = _mapping(
                row.get("decision_gates"),
                f"{case_id} K={expected_resolution} decision gates",
            )
            _exact_keys(
                row_gates,
                {"full_passed"},
                f"{case_id} K={expected_resolution} decision gates",
            )
            _require(
                row_gates.get("full_passed") is row_full_passed,
                f"{case_id} K={expected_resolution} full rollup differs",
            )
            _require(
                row.get("validity_passed") is True,
                f"{case_id} K={expected_resolution} validity failed",
            )
            expected_outcome = EXACT_OUTCOME if row_full_passed else REFUTED_OUTCOME
            _require(
                row.get("decision_outcome") == expected_outcome,
                f"{case_id} K={expected_resolution} outcome differs",
            )
            case_full_passed = row_full_passed and case_full_passed
        case_gates = _mapping(case.get("decision_gates"), f"{case_id} decision gates")
        _exact_keys(case_gates, {"full_passed"}, f"{case_id} decision gates")
        _require(
            case_gates.get("full_passed") is case_full_passed,
            f"{case_id} full rollup differs",
        )
        _require(case.get("validity_passed") is True, f"{case_id} validity failed")
        expected_outcome = EXACT_OUTCOME if case_full_passed else REFUTED_OUTCOME
        _require(
            case.get("decision_outcome") == expected_outcome,
            f"{case_id} outcome differs",
        )
        all_full_passed = case_full_passed and all_full_passed
    return cases, all_full_passed


def _validate_array_schema(
    arrays: Mapping[str, np.ndarray],
    *,
    lane_ordinal: int,
) -> None:
    for ordinal, case_id in _expected_lane_cases(lane_ordinal):
        for resolution in RESOLUTIONS:
            prefix = _unit_prefix(ordinal, case_id, resolution)
            ids = arrays[f"{prefix}__selected_cell_ids_int64"]
            cells = arrays[f"{prefix}__canonical_cells_int64"]
            points = arrays[f"{prefix}__canonical_points_float32"]
            centroids = arrays[f"{prefix}__canonical_centroids_float32"]
            areas = arrays[f"{prefix}__canonical_areas_float32"]
            normals = arrays[f"{prefix}__canonical_normals_float32"]
            _require(
                ids.dtype == np.dtype("<i8") and ids.shape == (resolution,),
                f"{prefix} selected-cell ID contract differs",
            )
            _, n_master_cells, historical_start = EXPECTED_CASE_METADATA[ordinal]
            expected_ids = (
                historical_start + np.arange(resolution, dtype="<i8")
            ) % n_master_cells
            _require(
                np.array_equal(ids, expected_ids),
                f"{prefix} selected-cell sequence differs",
            )
            _require(
                cells.dtype == np.dtype("<i8") and cells.shape == (resolution, 3),
                f"{prefix} cell contract differs",
            )
            _require(
                points.dtype == np.dtype("<f4")
                and points.ndim == 2
                and points.shape[0] > 0
                and points.shape[1] == 3,
                f"{prefix} point contract differs",
            )
            _require(
                centroids.dtype == np.dtype("<f4")
                and centroids.shape == (resolution, 3),
                f"{prefix} centroid contract differs",
            )
            _require(
                areas.dtype == np.dtype("<f4") and areas.shape == (resolution,),
                f"{prefix} area contract differs",
            )
            _require(
                normals.dtype == np.dtype("<f4") and normals.shape == (resolution, 3),
                f"{prefix} normal contract differs",
            )
            _require(
                all(
                    bool(np.isfinite(array).all())
                    for array in (points, centroids, areas, normals)
                )
                and bool((areas > 0).all()),
                f"{prefix} canonical geometry is non-finite or non-positive",
            )
            _require(
                bool((cells >= 0).all())
                and int(cells.max()) < points.shape[0]
                and np.array_equal(
                    np.unique(cells),
                    np.arange(points.shape[0], dtype="<i8"),
                ),
                f"{prefix} compact topology differs",
            )
            unit_deviation = float(
                np.max(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1) - 1.0))
            )
            area_center = np.einsum(
                "n,nd->d",
                areas.astype(np.float64),
                centroids.astype(np.float64),
            ) / np.sum(areas.astype(np.float64))
            center_deviation = float(np.max(np.abs(area_center)))
            _require(
                unit_deviation <= UNIT_ABS_TOLERANCE,
                f"{prefix} unit-normal gate failed independently",
            )
            _require(
                center_deviation <= CENTER_ABS_TOLERANCE,
                f"{prefix} area-center gate failed independently",
            )
            base_prefix = _unit_prefix(
                ordinal,
                case_id,
                RESOLUTIONS[0],
            )
            base_ids = arrays[f"{base_prefix}__selected_cell_ids_int64"]
            _require(
                np.array_equal(ids[:FIXED_QUERY_K], base_ids[:FIXED_QUERY_K]),
                f"{prefix} fixed query is not the exact selected-ID prefix",
            )
            for precision in PRECISIONS:
                for panel in QUERY_PANELS:
                    query_count = (
                        resolution if panel == "coupled_s_k" else FIXED_QUERY_K
                    )
                    for path in PATHS:
                        for field in FIELDS:
                            key = (
                                f"{prefix}__{precision}_canonical_full_"
                                f"{panel}_{path}_{field}"
                            )
                            expected_shape = (
                                (query_count,)
                                if field == "pressure"
                                else (query_count, 3)
                            )
                            array = arrays[key]
                            _require(
                                array.dtype == np.dtype("<f4")
                                and array.shape == expected_shape
                                and bool(np.isfinite(array).all()),
                                f"{key} dtype, shape, or finiteness differs",
                            )


def _recompute_comparisons(
    arrays: Mapping[str, np.ndarray],
    *,
    lane_ordinal: int,
    cases: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str], int, int, int]:
    rows = {
        (int(case["cohort_ordinal"]), int(row["resolution"])): row
        for case in cases
        for row in case["resolutions"]
    }
    replay_valid = True
    full_mismatches: list[str] = []
    full_comparisons = 0
    replay_and_prefix_comparisons = 0
    nonlicensing_fixed_panel_comparisons = 0
    for ordinal, case_id in _expected_lane_cases(lane_ordinal):
        for resolution in RESOLUTIONS:
            prefix = _unit_prefix(ordinal, case_id, resolution)
            row = rows[(ordinal, resolution)]
            for precision in PRECISIONS:
                full_summary = row["precision_probes"][precision][
                    "canonical_full_public_api"
                ]
                panels = full_summary["query_panels"]
                coupled_summary = panels["coupled_s_k"]
                for field in FIELDS:
                    coupled_prefix = f"{prefix}__{precision}_canonical_full_coupled_s_k"
                    primary = arrays[f"{coupled_prefix}_primary_{field}"]
                    fixed = arrays[f"{coupled_prefix}_fixed_{field}"]
                    full_comparisons += 1
                    observed_exact = _array_bitwise_equal(primary, fixed)
                    _require(
                        coupled_summary["primary_fixed_difference"][field]["exact"]
                        is observed_exact,
                        f"{prefix} {precision} {field} A/B summary differs",
                    )
                    if not observed_exact:
                        full_mismatches.append(
                            f"{prefix}:{precision}:coupled_s_k:{field}"
                        )
                for panel in QUERY_PANELS:
                    panel_prefix = f"{prefix}__{precision}_canonical_full_{panel}"
                    panel_summary = panels[panel]
                    for field in FIELDS:
                        primary = arrays[f"{panel_prefix}_primary_{field}"]
                        replay = arrays[f"{panel_prefix}_primary_replay_{field}"]
                        replay_and_prefix_comparisons += 1
                        observed_exact = _array_bitwise_equal(primary, replay)
                        _require(
                            panel_summary["primary_replay_difference"][field]["exact"]
                            is observed_exact,
                            f"{prefix} {precision} {panel} {field} replay summary differs",
                        )
                        if not observed_exact:
                            replay_valid = False
                    if panel == "fixed_id_prefix_s2500":
                        for field in FIELDS:
                            primary = arrays[f"{panel_prefix}_primary_{field}"]
                            fixed = arrays[f"{panel_prefix}_fixed_{field}"]
                            nonlicensing_fixed_panel_comparisons += 1
                            observed_exact = _array_bitwise_equal(primary, fixed)
                            _require(
                                panel_summary["primary_fixed_difference"][field][
                                    "exact"
                                ]
                                is observed_exact,
                                f"{prefix} {precision} fixed-panel {field} "
                                "A/B summary differs",
                            )
                for path in PATHS:
                    for field in FIELDS:
                        coupled = arrays[
                            f"{prefix}__{precision}_canonical_full_"
                            f"coupled_s_k_{path}_{field}"
                        ]
                        fixed_prefix = arrays[
                            f"{prefix}__{precision}_canonical_full_"
                            f"fixed_id_prefix_s2500_{path}_{field}"
                        ]
                        replay_and_prefix_comparisons += 1
                        observed_exact = _array_bitwise_equal(
                            coupled[:FIXED_QUERY_K],
                            fixed_prefix,
                        )
                        _require(
                            full_summary["fixed_id_prefix_matches_coupled_rows"][path][
                                field
                            ]
                            is observed_exact,
                            f"{prefix} {precision} {path} {field} prefix summary differs",
                        )
                        if not observed_exact:
                            replay_valid = False
                coupled_exact = all(
                    bool(coupled_summary["primary_fixed_difference"][field]["exact"])
                    for field in FIELDS
                )
                _require(
                    coupled_summary["comparison_gate"]["passed"] is coupled_exact
                    and full_summary["comparison_gate"]["passed"] is coupled_exact
                    and row["precision_probes"][precision]["decision_gates"][
                        "full_passed"
                    ]
                    is coupled_exact,
                    f"{prefix} {precision} A/B rollup differs",
                )
    expected_full = (
        len(_expected_lane_cases(lane_ordinal))
        * len(RESOLUTIONS)
        * len(PRECISIONS)
        * len(FIELDS)
    )
    expected_replay_and_prefix = (
        len(_expected_lane_cases(lane_ordinal))
        * len(RESOLUTIONS)
        * len(PRECISIONS)
        * (len(QUERY_PANELS) * len(FIELDS) + len(PATHS) * len(FIELDS))
    )
    _require(full_comparisons == expected_full, "Full comparison accounting differs")
    _require(
        replay_and_prefix_comparisons == expected_replay_and_prefix,
        "Replay/prefix comparison accounting differs",
    )
    return (
        replay_valid,
        full_mismatches,
        full_comparisons,
        replay_and_prefix_comparisons,
        nonlicensing_fixed_panel_comparisons,
    )


def _recompute_anchor_comparisons(
    arrays: Mapping[str, np.ndarray],
    anchor_arrays: Mapping[str, np.ndarray],
    *,
    lane_ordinal: int,
    cases: Sequence[Mapping[str, Any]],
) -> int:
    rows = {
        (int(case["cohort_ordinal"]), int(row["resolution"])): row
        for case in cases
        for row in case["resolutions"]
    }
    comparisons = 0
    for ordinal, case_id in _expected_lane_cases(lane_ordinal):
        if ordinal >= 4:
            continue
        resolution = RESOLUTIONS[0]
        prefix = _unit_prefix(ordinal, case_id, resolution)
        anchor_prefix = f"case_{ordinal:02d}_{case_id}"
        reported = rows[(ordinal, resolution)]["validity"]["job305691_anchor_replay"][
            "comparisons"
        ]
        for relative_name in _relative_unit_array_names():
            anchor_name = relative_name
            for panel in QUERY_PANELS:
                anchor_name = anchor_name.replace(f"_{panel}_", "_", 1)
            observed_exact = _array_bitwise_equal(
                arrays[f"{prefix}__{relative_name}"],
                anchor_arrays[f"{anchor_prefix}__{anchor_name}"],
            )
            comparisons += 1
            _require(
                observed_exact,
                f"{prefix} differs from frozen anchor for {relative_name}",
            )
            _require(
                reported[relative_name] is observed_exact,
                f"{prefix} anchor summary differs for {relative_name}",
            )
    _require(comparisons == 30, "Lane anchor-comparison accounting differs")
    return comparisons


def _load_and_audit_lane(
    json_path: Path,
    npz_path: Path,
    *,
    expected: ExpectedProvenance,
    anchor_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    json_path = Path(os.path.abspath(json_path))
    npz_path = Path(os.path.abspath(npz_path))
    json_bytes, json_sha256 = _load_verified_artifact_bytes(json_path, "Lane JSON")
    npz_bytes, npz_sha256 = _load_verified_artifact_bytes(npz_path, "Lane NPZ")
    summary = _load_json_bytes(json_bytes, json_path.name)
    forbidden_json = _forbidden_paths(summary)
    _require(not forbidden_json, f"Forbidden JSON keys: {forbidden_json}")
    _exact_keys(
        summary,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "decision_outcome",
            "generated_at_utc",
            "lane",
            "scientific_scope",
            "contract",
            "validity",
            "decision_gates",
            "cases",
            "npz_array_manifest",
            "provenance",
        },
        "Lane JSON",
    )
    _require(
        summary.get("schema_version") == LANE_SCHEMA_VERSION,
        "Lane schema version differs",
    )
    _require(
        summary.get("artifact_kind") == LANE_ARTIFACT_KIND,
        "Lane artifact kind differs",
    )
    lane = _mapping(summary.get("lane"), "Lane identity")
    _exact_keys(lane, {"ordinal", "count"}, "Lane identity")
    lane_ordinal = lane.get("ordinal")
    _require(
        type(lane_ordinal) is int and 0 <= lane_ordinal < LANE_COUNT,
        "Lane ordinal is outside the frozen range",
    )
    _require(lane.get("count") == LANE_COUNT, "Lane count differs")
    _validate_scope(summary, lane_ordinal=lane_ordinal)
    contract = _mapping(summary.get("contract"), "Lane contract")
    _require(dict(contract) == EXPECTED_CONTRACT, "Lane contract differs")
    cases, reported_full_passed = _validate_case_structure(
        summary,
        lane_ordinal=lane_ordinal,
    )
    lane_validity = _mapping(summary.get("validity"), "Lane validity")
    _exact_keys(
        lane_validity,
        {
            "all_cases_resolutions_and_precisions_passed",
            "geometry_input_manifest_lane_verification",
            "required_gates",
        },
        "Lane validity",
    )
    geometry_verification = _mapping(
        lane_validity.get("geometry_input_manifest_lane_verification"),
        "Geometry-manifest lane verification",
    )
    _exact_keys(
        geometry_verification,
        {"manifest_sha256", "lane_cases_verified", "lane_files_verified"},
        "Geometry-manifest lane verification",
    )
    _require(
        geometry_verification.get("manifest_sha256")
        == expected.geometry_input_manifest_sha256
        and geometry_verification.get("lane_cases_verified")
        == len(_expected_lane_cases(lane_ordinal))
        and geometry_verification.get("lane_files_verified")
        == 13 * len(_expected_lane_cases(lane_ordinal)),
        "Geometry-manifest lane verification differs",
    )
    _require(
        lane_validity.get("required_gates") == EXPECTED_REQUIRED_GATES,
        "Required validity-gate contract differs",
    )
    _require(
        lane_validity.get("all_cases_resolutions_and_precisions_passed") is True,
        "Lane validity rollup differs",
    )
    lane_decision_gates = _mapping(
        summary.get("decision_gates"),
        "Lane decision gates",
    )
    _exact_keys(lane_decision_gates, {"full"}, "Lane decision gates")
    full_gate = _mapping(lane_decision_gates.get("full"), "Lane full gate")
    _exact_keys(
        full_gate,
        {"criterion", "passed", "controls_candidate_advance"},
        "Lane full gate",
    )
    _require(
        full_gate.get("criterion") == FULL_GATE_CRITERION
        and full_gate.get("controls_candidate_advance") is True
        and full_gate.get("passed") is reported_full_passed,
        "Lane full decision gate differs",
    )
    status = summary.get("status")
    _require(
        status in {VALID_LANE_STATUS, INVALID_LANE_STATUS},
        "Lane status differs",
    )
    if status == VALID_LANE_STATUS:
        expected_outcome = EXACT_OUTCOME if reported_full_passed else REFUTED_OUTCOME
        _require(
            summary.get("decision_outcome") == expected_outcome,
            "Valid lane decision outcome differs",
        )
    else:
        _require(
            summary.get("decision_outcome") == INVALID_DIAGNOSTIC,
            "Invalid lane decision outcome differs",
        )
        raise LaneInvalid("Producer marked lane invalid")

    expected_names = _expected_array_names(lane_ordinal)
    arrays = _load_npz_bytes(npz_bytes, npz_path.name, expected_names)
    forbidden_arrays = [
        name
        for name in arrays
        if any(token in name.lower() for token in FORBIDDEN_KEY_TOKENS)
    ]
    _require(not forbidden_arrays, f"Forbidden NPZ keys: {forbidden_arrays}")
    _validate_manifest(summary, arrays)
    _validate_provenance(
        summary,
        npz_sha256=npz_sha256,
        expected=expected,
    )
    _validate_array_schema(arrays, lane_ordinal=lane_ordinal)
    (
        replay_valid,
        full_mismatches,
        full_comparisons,
        replay_and_prefix_comparisons,
        nonlicensing_fixed_panel_comparisons,
    ) = _recompute_comparisons(
        arrays,
        lane_ordinal=lane_ordinal,
        cases=cases,
    )
    observed_full_passed = not full_mismatches
    _require(
        reported_full_passed is observed_full_passed,
        "Reported full-lane gate differs from persisted tensors",
    )
    independently_valid = replay_valid
    anchor_comparisons = _recompute_anchor_comparisons(
        arrays,
        anchor_arrays,
        lane_ordinal=lane_ordinal,
        cases=cases,
    )
    return {
        "lane_ordinal": lane_ordinal,
        "case_ids": [case_id for _, case_id in _expected_lane_cases(lane_ordinal)],
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
        "producer_reported_validity_passed": True,
        "persisted_replay_and_prefix_exact": replay_valid,
        "independently_adjudicated_validity_passed": independently_valid,
        "full_field_tensor_comparisons": full_comparisons,
        "replay_and_prefix_tensor_comparisons": replay_and_prefix_comparisons,
        "nonlicensing_fixed_panel_tensor_comparisons": (
            nonlicensing_fixed_panel_comparisons
        ),
        "anchor_tensor_comparisons": anchor_comparisons,
        "full_mismatch_count": len(full_mismatches),
        "full_mismatch_keys": full_mismatches,
    }


def _base_result(
    *,
    status: str,
    outcome: str,
    lane_audits: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "artifact_kind": ADJUDICATION_ARTIFACT_KIND,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "outcome": outcome,
            "criterion": (
                "universal conjunction over every readable lane validity gate "
                "and every coupled S_K primary-versus-fixed pressure/WSS "
                "raw-byte comparison"
            ),
            "averaging_used": False,
        },
        "coverage": {
            "required_lane_ordinals": list(range(LANE_COUNT)),
            "adjudicated_lane_ordinals": sorted(
                audit["lane_ordinal"] for audit in lane_audits
            ),
            "required_case_ids": list(CASE_IDS),
            "resolutions": list(RESOLUTIONS),
            "precisions": list(PRECISIONS),
            "complete": len(lane_audits) == LANE_COUNT and not failures,
        },
        "lane_audits": sorted(lane_audits, key=lambda row: row["lane_ordinal"]),
        "failures": list(failures),
    }


def adjudicate(
    *,
    lane_artifacts: Sequence[tuple[Path, Path]],
    expected_provenance: ExpectedProvenance,
    anchor_npz_path: Path,
) -> dict[str, Any]:
    """Return an explicit cohort result without collapsing absence to refutation."""

    expected_provenance.validate()
    if len(lane_artifacts) < LANE_COUNT:
        return _base_result(
            status=INCOMPLETE_ADJUDICATION_STATUS,
            outcome=INCOMPLETE_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "missing_lane_artifacts",
                    "required_pairs": LANE_COUNT,
                    "received_pairs": len(lane_artifacts),
                }
            ],
        )
    if len(lane_artifacts) > LANE_COUNT:
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "wrong_lane_artifact_count",
                    "required_pairs": LANE_COUNT,
                    "received_pairs": len(lane_artifacts),
                }
            ],
        )

    try:
        anchor_bytes, anchor_sha256 = _load_verified_artifact_bytes(
            anchor_npz_path,
            "Frozen job305691 anchor NPZ",
        )
    except LaneUnavailable as failure:
        return _base_result(
            status=INCOMPLETE_ADJUDICATION_STATUS,
            outcome=INCOMPLETE_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "unavailable_anchor_artifact",
                    "message": str(failure),
                }
            ],
        )
    if anchor_sha256 != EXPECTED_STABLE_INPUT_HASHES["job305691_anchor_npz"]:
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "wrong_anchor_artifact",
                    "observed_sha256": anchor_sha256,
                }
            ],
        )
    try:
        anchor_arrays = _load_npz_bytes(
            anchor_bytes,
            Path(anchor_npz_path).name,
            _expected_anchor_array_names(),
        )
    except LaneUnavailable as failure:
        return _base_result(
            status=INCOMPLETE_ADJUDICATION_STATUS,
            outcome=INCOMPLETE_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "unavailable_anchor_artifact",
                    "message": str(failure),
                }
            ],
        )
    except LaneInvalid as failure:
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=[],
            failures=[
                {
                    "kind": "invalid_anchor_artifact",
                    "message": str(failure),
                }
            ],
        )

    lane_audits: list[Mapping[str, Any]] = []
    unavailable: list[Mapping[str, Any]] = []
    invalid: list[Mapping[str, Any]] = []
    for input_ordinal, (json_path, npz_path) in enumerate(lane_artifacts):
        try:
            lane_audits.append(
                _load_and_audit_lane(
                    json_path,
                    npz_path,
                    expected=expected_provenance,
                    anchor_arrays=anchor_arrays,
                )
            )
        except LaneUnavailable as failure:
            unavailable.append(
                {
                    "kind": "unavailable_lane_artifact",
                    "input_ordinal": input_ordinal,
                    "message": str(failure),
                }
            )
        except LaneInvalid as failure:
            invalid.append(
                {
                    "kind": "invalid_lane_artifact",
                    "input_ordinal": input_ordinal,
                    "message": str(failure),
                }
            )
    if unavailable:
        return _base_result(
            status=INCOMPLETE_ADJUDICATION_STATUS,
            outcome=INCOMPLETE_OUTCOME,
            lane_audits=lane_audits,
            failures=[*unavailable, *invalid],
        )
    if invalid:
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=lane_audits,
            failures=invalid,
        )

    observed_ordinals = [audit["lane_ordinal"] for audit in lane_audits]
    if sorted(observed_ordinals) != list(range(LANE_COUNT)):
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=lane_audits,
            failures=[
                {
                    "kind": "duplicate_or_missing_lane_ordinal",
                    "observed_lane_ordinals": observed_ordinals,
                }
            ],
        )
    observed_cases = [
        case_id
        for audit in sorted(lane_audits, key=lambda row: row["lane_ordinal"])
        for case_id in audit["case_ids"]
    ]
    expected_by_lane = [
        case_id
        for lane_ordinal in range(LANE_COUNT)
        for _, case_id in _expected_lane_cases(lane_ordinal)
    ]
    if observed_cases != expected_by_lane:
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=lane_audits,
            failures=[{"kind": "full_cohort_modulo_coverage_changed"}],
        )
    if not all(
        audit["independently_adjudicated_validity_passed"] for audit in lane_audits
    ):
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=lane_audits,
            failures=[{"kind": "lane_validity_failure"}],
        )
    any_full_mismatch = any(audit["full_mismatch_count"] > 0 for audit in lane_audits)
    result = _base_result(
        status=VALID_ADJUDICATION_STATUS,
        outcome=REFUTED_OUTCOME if any_full_mismatch else EXACT_OUTCOME,
        lane_audits=lane_audits,
        failures=[],
    )
    result["decision"]["full_passed"] = not any_full_mismatch
    result["decision"]["validity_passed"] = True
    result["decision"]["full_field_tensor_comparisons"] = sum(
        audit["full_field_tensor_comparisons"] for audit in lane_audits
    )
    result["decision"]["replay_and_prefix_tensor_comparisons"] = sum(
        audit["replay_and_prefix_tensor_comparisons"] for audit in lane_audits
    )
    result["decision"]["nonlicensing_fixed_panel_tensor_comparisons"] = sum(
        audit["nonlicensing_fixed_panel_tensor_comparisons"] for audit in lane_audits
    )
    result["decision"]["anchor_tensor_comparisons"] = sum(
        audit["anchor_tensor_comparisons"] for audit in lane_audits
    )
    result["decision"]["anchor_npz_sha256"] = anchor_sha256
    result["decision"]["full_mismatch_count"] = sum(
        audit["full_mismatch_count"] for audit in lane_audits
    )
    expected_full_comparisons = (
        len(CASE_IDS) * len(RESOLUTIONS) * len(PRECISIONS) * len(FIELDS)
    )
    expected_replay_and_prefix_comparisons = (
        len(CASE_IDS)
        * len(RESOLUTIONS)
        * len(PRECISIONS)
        * (len(QUERY_PANELS) * len(FIELDS) + len(PATHS) * len(FIELDS))
    )
    if (
        result["decision"]["full_field_tensor_comparisons"] != expected_full_comparisons
        or result["decision"]["replay_and_prefix_tensor_comparisons"]
        != expected_replay_and_prefix_comparisons
        or result["decision"]["anchor_tensor_comparisons"] != 120
    ):
        return _base_result(
            status=INVALID_ADJUDICATION_STATUS,
            outcome=INVALID_OUTCOME,
            lane_audits=lane_audits,
            failures=[{"kind": "comparison_accounting_failure"}],
        )
    return result


def _prepare_temporary(path: Path, payload: bytes) -> tuple[Path, tuple[int, int]]:
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
            metadata = os.fstat(stream.fileno())
            identity = (metadata.st_dev, metadata.st_ino)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, identity


def _link_temporary_no_clobber(temporary: Path, path: Path) -> None:
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error


def _verify_published_exact(
    path: Path,
    *,
    identity: tuple[int, int],
    payload: bytes,
) -> None:
    try:
        observed, observed_identity = _read_regular_file_bytes(
            path,
            f"Published output {path.name}",
        )
    except LaneUnavailable as error:
        raise OSError(f"Could not verify published output {path}") from error
    if observed_identity != identity or observed != payload:
        raise OSError(f"Published output changed during transaction: {path}")


def _cleanup_temporaries(temporaries: Sequence[Path]) -> None:
    for temporary in temporaries:
        temporary.unlink(missing_ok=True)


def _rollback_output_set(
    outputs: Sequence[tuple[Path, tuple[int, int]]],
) -> None:
    for path, identity in reversed(outputs):
        try:
            _unlink_if_same_inode(path, identity)
        except OSError:
            pass


def _unlink_if_same_inode(path: Path, identity: tuple[int, int]) -> None:
    """Roll back only the exact file linked by this process."""
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) == identity:
        path.unlink()


def write_adjudication(path: Path, result: Mapping[str, Any]) -> None:
    """Write canonical JSON and SHA-256 sidecar with atomic no-clobber links."""

    output = Path(os.path.abspath(path))
    sidecar = output.with_name(f"{output.name}.sha256")
    if (
        output.exists()
        or output.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise FileExistsError(f"Refusing to overwrite {output} or its sidecar")
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(payload).hexdigest()
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    output_temporary: Path | None = None
    sidecar_temporary: Path | None = None
    try:
        output_temporary, output_identity = _prepare_temporary(output, payload)
        sidecar_temporary, sidecar_identity = _prepare_temporary(
            sidecar,
            sidecar_payload,
        )
    except BaseException:
        _cleanup_temporaries(
            [
                temporary
                for temporary in (output_temporary, sidecar_temporary)
                if temporary is not None
            ]
        )
        raise

    published = (
        (output, output_identity),
        (sidecar, sidecar_identity),
    )
    temporaries = (output_temporary, sidecar_temporary)
    try:
        _link_temporary_no_clobber(output_temporary, output)
        _link_temporary_no_clobber(sidecar_temporary, sidecar)
        _verify_published_exact(output, identity=output_identity, payload=payload)
        _verify_published_exact(
            sidecar,
            identity=sidecar_identity,
            payload=sidecar_payload,
        )
    except BaseException:
        _rollback_output_set(published)
        try:
            _cleanup_temporaries(temporaries)
        except OSError:
            pass
        raise
    try:
        _cleanup_temporaries(temporaries)
    except BaseException:
        _rollback_output_set(published)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        action="append",
        nargs=2,
        default=[],
        metavar=("JSON", "NPZ"),
        help="one lane JSON/NPZ pair; repeat up to four times",
    )
    parser.add_argument("--expected-lane-producer-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-geometry-input-manifest-sha256", required=True)
    parser.add_argument(
        "--anchor-npz",
        type=Path,
        required=True,
        help="frozen job305691 NPZ; its canonical .sha256 sidecar is required",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    expected = ExpectedProvenance(
        lane_producer_sha256=args.expected_lane_producer_sha256,
        source_tree_sha256=args.expected_source_tree_sha256,
        geometry_input_manifest_sha256=(args.expected_geometry_input_manifest_sha256),
    )
    result = adjudicate(
        lane_artifacts=[
            (Path(json_path), Path(npz_path)) for json_path, npz_path in args.lane
        ],
        expected_provenance=expected,
        anchor_npz_path=args.anchor_npz,
    )
    write_adjudication(args.output_json, result)
    print(
        f"{result['status']} outcome={result['decision']['outcome']} "
        f"output={args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
