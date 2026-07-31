# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independently adjudicate the target-free canonical-geometry diagnostic.

This program does not import the diagnostic producer. It verifies artifact
sidecars and array manifests, recomputes every persisted prediction
comparison, replays the persisted job-304002 controls, and recomputes the
schema-v5 decision gates and outcome. Exactness is recomputed from contiguous
raw bytes, so equal-valued opposite signed zeros remain distinct. It cannot
independently replay runtime
facts that were not persisted, so those producer-only validity gates are
reported explicitly rather than silently treated as independently observed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import struct
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DIAGNOSTIC_SCHEMA_VERSION = 5
DIAGNOSTIC_ARTIFACT_KIND = "hqc_canonical_geometry_diagnostic"
AUDIT_SCHEMA_VERSION = 10
AUDIT_ARTIFACT_KIND = "hqc_canonical_geometry_independent_adjudication"
CASE_SPECS = (
    (0, "run_118", 21),
    (1, "run_129", 33),
    (2, "run_145", 51),
    (3, "run_149", 55),
)
CASE_IDS = tuple(spec[1] for spec in CASE_SPECS)
PRECISIONS = ("bfloat16", "float32")
MODES = ("canonical_derived", "canonical_full")
PATHS = ("primary", "fixed")
FIELDS = ("pressure", "wss")
GEOMETRY_FIELDS = ("points", "centroids", "areas", "normals")
DERIVED_TOLERANCE = 1.0e-3
EXPECTED_PHYSICAL_LENGTH = 5.0
EXPECTED_MODEL_REFERENCE_LENGTH = 8.0
EXPECTED_EFFECTIVE_LENGTH = 40.0
EXPECTED_CANONICAL_FRAME_CONSTRUCTION = (
    "raw selected coordinates promoted to float64; physical "
    "area-weighted center removed; coherent triangle geometry "
    "divided by L_ref*model_reference_length; one float32 cast"
)
FULL_AND_DERIVED_OUTCOME = "FULL_AND_DERIVED_PASS"
FULL_ONLY_OUTCOME = "FULL_ONLY_PASS"
CANONICAL_REPAIR_REFUTED = "CANONICAL_REPAIR_REFUTED"
INVALID_DIAGNOSTIC = "INVALID_DIAGNOSTIC"
VALID_STATUS = "VALID_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC"
INVALID_STATUS = "INVALID_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC"
EXPECTED_DIAGNOSTIC_SCRIPT_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)
EXPECTED_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fa6a7b683fa9aa02e4537ef69e8e977906df7c9fa6964cb759edfcee8d7b90cd"
)
EXPECTED_PRIOR_JSON_SHA256 = (
    "26aed78264e9fd66f329941ce000fc438cb26f6835f9f05ff128567d29444bf5"
)
EXPECTED_PRIOR_NPZ_SHA256 = (
    "d1e6a9fa1a39aa78a9cca26e52eb783a9e78aecbb961ce917164e25fac75a7ea"
)
EXPECTED_INPUT_HASHES = {
    "dataset_manifest": "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca",
    "dataset_config": "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3",
    "resolved_config": "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1",
    "model_checkpoint": "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88",
    "normalization_stats": "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94",
    "historical_metrics": "423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a",
    "prior_diagnostic_json": EXPECTED_PRIOR_JSON_SHA256,
    "prior_diagnostic_npz": EXPECTED_PRIOR_NPZ_SHA256,
}
EXPECTED_SELECTED_SOURCE_FILES = {
    "physicsnemo/datapipes/transforms/mesh/transforms.py": (
        "cde24a7d9ee1c0cecfd752a6d418c14ef72f1fe724e42b835e4f0e2ad6dd2dc3"
    ),
    "physicsnemo/experimental/nn/mesh_attention/kernel_decoder.py": (
        "0492569fb1ea802891a4aa4ceb57717f72d3b747349f811372b729518cc12f0d"
    ),
    "physicsnemo/experimental/nn/mesh_attention/model.py": (
        "ae336d795dfdd0952ffab730dff4fe88989b6c4c68c18c45477027847041c0ac"
    ),
    "physicsnemo/mesh/mesh.py": (
        "61619c352b24f56acfe0d8ecf08d637587ec28aff146f5e42feacdc44328d243"
    ),
}
EXPECTED_BUNDLE_MEMBER_SHA256_BY_RELATIVE_PATH = {
    "drivaerml_hqc_canonical_geometry_diagnostic_v5.py": (
        EXPECTED_DIAGNOSTIC_SCRIPT_SHA256
    ),
    "drivaerml_trace_fixed_query_audit.py": EXPECTED_PRODUCER_SHA256,
    "phase1_hqc_neutral_canonical_geometry_preregistration_v5_2026-07-28.json": (
        "8f13a32174a6b60f11bc45bc766f6ee7e59012a6530510fb703be2357bd558ec"
    ),
    "phase1_hqc_neutral_canonical_geometry_preregistration_v5_2026-07-28.json.sha256": (
        "7e7ce6fa2eaf2553937c7fee2365132bc46c9263af010d39bad265db10384df1"
    ),
    "drivaerml_hqc_neutral_canonical_geometry_v5_aga.sbatch": (
        "adc6bc6deabe3838113cef935fcc3de26b34e0e10bba077d12c755e2a01f01ce"
    ),
    "drivaerml_hqc_neutral_canonical_geometry_v5_aga_README.md": (
        "b095b07253c5582d946096c654318c9d40d1ac07608b41a5b2d3ea043f25cfd8"
    ),
    "phase1_hqc_neutral_canonical_geometry_launch_manifest_v5_2026-07-28.json": (
        "e6eb2a92fc01b100ab5e3b40b47d03263f0e6330b72e8f7a804bf4e65add17f2"
    ),
    "phase1_hqc_neutral_canonical_geometry_launch_manifest_v5_2026-07-28.json.sha256": (
        "47184ec2545b81773229aeac087b315cb24d3c706663e57b0e7290c342bc8004"
    ),
    "phase1_hqc_neutral_canonical_geometry_v4_abandoned_prelaunch_2026-07-28.json": (
        "2f4f7419f397b6a9310a543bceed26ddb7b8a717bf03d0875da87efda2e807b6"
    ),
    "phase1_hqc_neutral_canonical_geometry_v4_abandoned_prelaunch_2026-07-28.json.sha256": (
        "4948f33b0a727ffebab4359eeb4c653bd687b53571bc285cfc82769d1f31c4c9"
    ),
    "hqc_center_diagnostic_four_canaries_k2500_v2.json": (EXPECTED_PRIOR_JSON_SHA256),
    "hqc_center_diagnostic_four_canaries_k2500_v2.json.sha256": (
        "abc338206f2fd8db0224344f08498b2b7b0b54d05e67fc8976340c8682b05d9d"
    ),
    "hqc_center_diagnostic_four_canaries_k2500_v2.npz": EXPECTED_PRIOR_NPZ_SHA256,
    "hqc_center_diagnostic_four_canaries_k2500_v2.npz.sha256": (
        "3c3690bbbf41c762fe7d1222ed1d76011ced7ece2c6705c6a616075e10b93ef0"
    ),
}
EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME = (
    "hqc_neutral_canonical_geometry_four_canaries_k2500_v5"
)
EXPECTED_PRIOR_NPZ_BASENAME = "hqc_center_diagnostic_four_canaries_k2500_v2.npz"
EXPECTED_REMOTE_TASK_DIRECTORY = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/"
    "users/psharpe/agents/"
    "2026-07-28-mt-hqc-neutral-canonical-v5"
)
EXPECTED_REMOTE_REPO_ROOT = (
    "/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer"
)
EXPECTED_REMOTE_RECIPE_ROOT = (
    f"{EXPECTED_REMOTE_REPO_ROOT}/examples/cfd/external_aerodynamics/"
    "unified_external_aero_recipe"
)
EXPECTED_REMOTE_DATASET_ROOT = (
    "/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow"
)
EXPECTED_REMOTE_RUN_ROOT = (
    f"{EXPECTED_REMOTE_RECIPE_ROOT}/runs/t2_mesh_transformer_surface_flagship_seed42"
)
EXPECTED_REMOTE_HISTORICAL_RUN_ROOT = (
    "/home/psharpe/coreai_modulus_cae/users/psharpe/agents/"
    "2026-07-25-mt-coverage-sweep/artifacts/res10000/"
    "t2_mesh_transformer_surface_flagship_seed42"
)
EXPECTED_DIAGNOSTIC_SCRIPT_PATH = (
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/"
    "drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
)
EXPECTED_FROZEN_PRODUCER_PATH = (
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/drivaerml_trace_fixed_query_audit.py"
)
EXPECTED_OUTPUT_JSON_PATH = (
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/artifacts/"
    f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.json"
)
EXPECTED_OUTPUT_NPZ_PATH = (
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/artifacts/"
    f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.npz"
)
EXPECTED_COMMAND = (
    EXPECTED_DIAGNOSTIC_SCRIPT_PATH,
    "--producer",
    EXPECTED_FROZEN_PRODUCER_PATH,
    "--repo-root",
    EXPECTED_REMOTE_REPO_ROOT,
    "--dataset-root",
    EXPECTED_REMOTE_DATASET_ROOT,
    "--dataset-config",
    f"{EXPECTED_REMOTE_RECIPE_ROOT}/datasets/drivaer_ml_surface.yaml",
    "--resolved-config",
    f"{EXPECTED_REMOTE_RUN_ROOT}/resolved_config.yaml",
    "--checkpoint-dir",
    f"{EXPECTED_REMOTE_RUN_ROOT}/checkpoints",
    "--historical-metrics",
    f"{EXPECTED_REMOTE_HISTORICAL_RUN_ROOT}/metrics.jsonl",
    "--prior-diagnostic-json",
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/"
    "hqc_center_diagnostic_four_canaries_k2500_v2.json",
    "--prior-diagnostic-npz",
    f"{EXPECTED_REMOTE_TASK_DIRECTORY}/"
    "hqc_center_diagnostic_four_canaries_k2500_v2.npz",
    "--output-json",
    EXPECTED_OUTPUT_JSON_PATH,
    "--output-npz",
    EXPECTED_OUTPUT_NPZ_PATH,
)
EXPECTED_RUNTIME_STRINGS = {
    "python": "3.13.14",
    "platform": "Linux-6.8.0-1051-nvidia-64k-aarch64-with-glibc2.39",
    "numpy": "2.4.6",
    "torch": "2.12.0+cu130",
}
EXPECTED_HARDWARE = {
    "cuda_runtime": "13.0",
    "cuda_device_name": "NVIDIA GB300",
    "cuda_device_capability": [10, 3],
}
EXPECTED_GPU_MEMORY_TOTAL_MIB = 284208
EXPECTED_EXPERIMENTAL_WARNING = (
    f"{EXPECTED_FROZEN_PRODUCER_PATH}:765: ExperimentalFeatureWarning: "
    "You are importing from 'physicsnemo.experimental'. The APIs in this "
    "namespace are experimental, under active development, and may change "
    "without notice. Expect possible back-compatibility breaking changes and "
    "only partial test coverage."
)
EXPECTED_EXPERIMENTAL_WARNING_CONTINUATION = (
    "  from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import ("
)


def _expected_copied_bundle_root_basename(job_id: str) -> str:
    return f"hqc_neutral_canonical_geometry_four_canaries_v5_job{job_id}_2026-07-28"


def _expected_audit_output_basename(job_id: str) -> str:
    return (
        "phase1_hqc_neutral_canonical_geometry_adjudication_"
        f"v13_job{job_id}_2026-07-28.json"
    )


STATIC_ALLOWED_BUNDLE_RELATIVE_PATHS = frozenset(
    EXPECTED_BUNDLE_MEMBER_SHA256_BY_RELATIVE_PATH
)
FORBIDDEN_KEY_TOKENS = (
    "target",
    "truth",
    "error",
    "force",
    "area_weighted",
    "area_objective",
    "endpoint",
    "log_cliff",
    "reducer",
    "support",
    "futility",
    "mixed",
    "eligibility",
    "verdict",
)
FORBIDDEN_LOG_TOKENS = FORBIDDEN_KEY_TOKENS + (
    "pMeanTrim",
    "wallShearStressMeanTrim",
    "true_pressure",
    "true_wss",
)
ALLOWED_FORBIDDEN_TOKEN_PATHS = {
    "scientific_scope.may_not_be_used_as_hqc_verdict_output",
}
FLOAT_REL_TOLERANCE = 1.0e-10
FLOAT_ABS_TOLERANCE = 2.0e-12
CANONICAL_CENTROID_MAX_ABS_TOLERANCE = 5.0e-7
CANONICAL_CENTROID_RELATIVE_TOLERANCE = 5.0e-6
CANONICAL_AREA_RELATIVE_TOLERANCE = 5.0e-5
CANONICAL_NORMAL_RELATIVE_TOLERANCE = 5.0e-4
CANONICAL_NORMAL_MINIMUM_DOT = 0.999


class AdjudicationFailure(ValueError):
    """The persisted artifacts do not satisfy their frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjudicationFailure(message)


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _resolve_regular_input(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    _require(
        lexical.is_file() and not lexical.is_symlink(),
        f"{label} is missing or is a symlink",
    )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise AdjudicationFailure(f"{label} is missing") from error
    _require(lexical == resolved, f"{label} traverses a symlink")
    return resolved


def _array_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether shape, dtype, and contiguous raw bytes are identical."""

    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    _require(sidecar.is_file(), f"Missing sidecar for {path.name}")
    payload = sidecar.read_bytes()
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise AdjudicationFailure(f"Malformed sidecar for {path.name}") from error
    _require(len(lines) == 1, f"Malformed sidecar for {path.name}")
    parts = lines[0].split("  ")
    _require(len(parts) == 2, f"Malformed sidecar for {path.name}")
    expected, recorded_name = parts
    _require(recorded_name == path.name, f"Sidecar filename differs for {path.name}")
    _require(
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        f"Malformed digest for {path.name}",
    )
    actual = _sha256_file(path)
    _require(actual == expected, f"Sidecar digest differs for {path.name}")
    _require(
        payload == f"{actual}  {path.name}\n".encode("ascii"),
        f"Sidecar bytes are not canonical for {path.name}",
    )
    return actual


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    _require(set(value) == expected, f"{label} key set differs")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate object name: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise AdjudicationFailure(f"JSON contains non-finite numeric token: {value}")


def _parse_finite_json_float(value: str) -> float:
    decimal = Decimal(value)
    parsed = float(decimal)
    _require(math.isfinite(parsed), f"JSON float is out of range: {value}")
    _require(
        decimal == 0 or parsed != 0.0,
        f"JSON float underflows binary64: {value}",
    )
    return parsed


def _load_json_unique(path: Path) -> Any:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdjudicationFailure(f"Malformed JSON artifact: {path.name}") from error
    canonical = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(payload == canonical, f"JSON is not canonically serialized: {path.name}")
    return value


def _all_bool_values(value: Mapping[str, Any], label: str) -> bool:
    _require(
        all(type(item) is bool for item in value.values()),
        f"{label} contains a non-boolean report",
    )
    return all(value.values())


def _verify_bundle_manifest(path: Path) -> tuple[str, dict[Path, str]]:
    path = _resolve_regular_input(path, "Copied-bundle manifest")
    _require(
        path.name == "manifest.sha256",
        "Copied-bundle manifest filename differs",
    )
    base = path.parent
    payload = path.read_bytes()
    try:
        manifest_text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise AdjudicationFailure("Copied-bundle manifest is not ASCII") from error
    recorded: dict[Path, str] = {}
    for line_number, line in enumerate(
        manifest_text.splitlines(),
        start=1,
    ):
        parts = line.split("  ")
        _require(
            len(parts) == 2,
            f"Malformed copied-bundle manifest line {line_number}",
        )
        digest, recorded_name = parts
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"Malformed copied-bundle digest on line {line_number}",
        )
        relative = Path(recorded_name)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe copied-bundle path on line {line_number}",
        )
        member = base / relative
        _require(
            member != path and member.is_relative_to(base),
            f"Invalid copied-bundle member on line {line_number}",
        )
        _require(
            not member.is_symlink(),
            f"Copied-bundle member is a symlink on line {line_number}",
        )
        try:
            resolved_member = member.resolve(strict=True)
        except OSError as error:
            raise AdjudicationFailure("Bundle member is missing") from error
        _require(member not in recorded, "Duplicate copied-bundle manifest member")
        _require(
            resolved_member == member and member.is_file(),
            "Bundle member is missing or traverses a symlink",
        )
        _require(_sha256_file(member) == digest, "Copied-bundle member digest differs")
        recorded[member] = digest
    actual_entries = list(base.rglob("*"))
    for entry in (base, *actual_entries):
        _require(
            not entry.is_symlink(),
            f"Copied bundle contains a symlink: {entry.relative_to(base)}",
        )
        _require(
            entry.is_file() or entry.is_dir(),
            f"Copied bundle contains a special filesystem entry: "
            f"{entry.relative_to(base)}",
        )
        try:
            extended_attributes = os.listxattr(entry, follow_symlinks=False)
        except OSError as error:
            raise AdjudicationFailure(
                f"Cannot verify copied-bundle extended attributes: "
                f"{entry.relative_to(base)}"
            ) from error
        _require(
            not extended_attributes,
            f"Copied bundle contains extended attributes: {entry.relative_to(base)}",
        )
    # Empty real directories carry no artifact bytes and are allowed. Every
    # non-directory entry must be a regular, explicitly recorded file.
    actual_members = {
        member for member in actual_entries if member.is_file() and member != path
    }
    actual_directories = {entry for entry in actual_entries if entry.is_dir()}
    _require(
        actual_directories == {base / "artifacts", base / "sbatch_logs"},
        "Copied-bundle directory allowlist differs",
    )
    _require(
        actual_members == set(recorded),
        "Copied-bundle manifest is not complete",
    )
    canonical_manifest = "".join(
        f"{recorded[member]}  ./{member.relative_to(base).as_posix()}\n"
        for member in sorted(
            recorded,
            key=lambda candidate: candidate.relative_to(base).as_posix(),
        )
    ).encode("ascii")
    _require(
        payload == canonical_manifest,
        "Copied-bundle manifest bytes or order are not canonical",
    )
    for (
        relative_path,
        expected_digest,
    ) in EXPECTED_BUNDLE_MEMBER_SHA256_BY_RELATIVE_PATH.items():
        expected_member = base / relative_path
        _require(
            recorded.get(expected_member) == expected_digest,
            f"Frozen copied-bundle member differs: {relative_path}",
        )
    return _sha256_file(path), recorded


def _validate_completed_job_markers(
    bundle_members: Mapping[Path, str],
    bundle_root: Path,
    job_id: str,
) -> None:
    expected_done = f"DONE_{job_id}"
    expected_status = f"STATUS_{job_id}"
    done_members = [
        member for member in bundle_members if member.name.startswith("DONE_")
    ]
    status_members = [
        member for member in bundle_members if member.name.startswith("STATUS_")
    ]
    _require(
        done_members == [bundle_root / expected_done],
        "Copied bundle has no unique matching DONE marker",
    )
    _require(
        status_members == [bundle_root / expected_status],
        "Copied bundle has no unique matching STATUS marker",
    )
    _require(
        bundle_members[done_members[0]]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "DONE marker is not empty",
    )
    _require(
        status_members[0].read_bytes() == b"rc=0\ncompleted_units=1/1\n",
        "STATUS marker does not report one completed valid unit",
    )


def _expected_bundle_relative_paths(job_id: str) -> set[str]:
    output = EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME
    return set(STATIC_ALLOWED_BUNDLE_RELATIVE_PATHS) | {
        f"artifacts/{output}.json",
        f"artifacts/{output}.json.sha256",
        f"artifacts/{output}.npz",
        f"artifacts/{output}.npz.sha256",
        f"DONE_{job_id}",
        f"STATUS_{job_id}",
        f"sbatch_logs/mt-hqc-canon-v5_{job_id}.log",
    }


def _validate_bundle_allowlist(
    bundle_members: Mapping[Path, str],
    bundle_root: Path,
    job_id: str,
) -> None:
    observed = {member.relative_to(bundle_root).as_posix() for member in bundle_members}
    expected = _expected_bundle_relative_paths(job_id)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    _require(
        not missing and not unexpected,
        f"Copied bundle path allowlist differs: missing={missing}, "
        f"unexpected={unexpected}",
    )
    forbidden_paths = [
        relative_path
        for relative_path in observed
        if any(token in relative_path.lower() for token in FORBIDDEN_KEY_TOKENS)
    ]
    _require(
        not forbidden_paths,
        f"Copied bundle has forbidden relative paths: {sorted(forbidden_paths)}",
    )


def _validate_slurm_log_content(
    bundle_root: Path,
    job_id: str,
    summary: Mapping[str, Any],
) -> None:
    path = bundle_root / "sbatch_logs" / f"mt-hqc-canon-v5_{job_id}.log"
    payload = path.read_bytes()
    _require(
        0 < len(payload) <= 10 * 1024 * 1024,
        "Slurm log is empty or implausibly large",
    )
    _require(payload.endswith(b"\n"), "Slurm log is not LF-terminated")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise AdjudicationFailure("Slurm log is not ASCII text") from error
    _require(
        all(character == "\n" or 32 <= ord(character) <= 126 for character in text),
        "Slurm log contains a control or non-printable character",
    )
    _require(
        not any(character in text for character in "\\%&#"),
        "Slurm log contains an escape metacharacter",
    )
    lower = text.lower()
    normalized = "".join(character for character in lower if character.isalnum())
    found = sorted(
        token
        for token in FORBIDDEN_LOG_TOKENS
        if token.lower() in lower
        or "".join(character for character in token.lower() if character.isalnum())
        in normalized
    )
    _require(
        not found,
        f"Slurm log contains forbidden vocabulary: {found}",
    )
    lines = text.splitlines()
    _require(len(lines) >= 2, "Slurm log is missing wrapper preamble")

    sampler_lines: list[tuple[str, Any]] = []
    heartbeat_timestamps: list[tuple[int, datetime]] = []
    control_lines: list[str] = []
    gpu_pattern = re.compile(r"([0-3]), ([0-9]{1,3}), ([0-9]{1,6}), ([0-9]{1,6})")
    heartbeat_prefix = "GPU_HEARTBEAT "
    for line_index, line in enumerate(lines):
        if line.startswith(heartbeat_prefix):
            timestamp = line[len(heartbeat_prefix) :]
            _require(
                line_index >= 2,
                "Slurm GPU sampler precedes the wrapper preamble",
            )
            parsed = _require_log_timestamp(
                timestamp,
                "Slurm GPU-heartbeat timestamp",
            )
            heartbeat_timestamps.append((line_index, parsed))
            sampler_lines.append(("heartbeat", timestamp))
            continue
        match = gpu_pattern.fullmatch(line)
        if match is not None:
            _require(
                line_index >= 2,
                "Slurm GPU sampler precedes the wrapper preamble",
            )
            index, utilization, used, total = map(int, match.groups())
            _require(
                0 <= utilization <= 100
                and 0 <= used <= total
                and total == EXPECTED_GPU_MEMORY_TOTAL_MIB,
                "Slurm GPU sampler values are outside the frozen AGA contract",
            )
            sampler_lines.append(("gpu", index))
            continue
        control_lines.append(line)

    _require(sampler_lines, "Slurm log has no GPU sampler output")
    sampler_groups: list[list[int]] = []
    current_group: list[int] | None = None
    for kind, value in sampler_lines:
        if kind == "heartbeat":
            if current_group is not None:
                sampler_groups.append(current_group)
            current_group = []
        else:
            _require(
                current_group is not None,
                "Slurm GPU row precedes its heartbeat",
            )
            current_group.append(int(value))
    _require(current_group is not None, "Slurm GPU sampler grouping failed")
    sampler_groups.append(current_group)
    for group in sampler_groups[:-1]:
        _require(
            group == [0, 1, 2, 3],
            "Slurm GPU sampler has a malformed complete group",
        )
    _require(
        sampler_groups[-1] in ([0, 1, 2, 3], [], [0], [0, 1], [0, 1, 2]),
        "Slurm GPU sampler has a malformed final group",
    )
    _require(
        any(group == [0, 1, 2, 3] for group in sampler_groups),
        "Slurm log has no complete four-device GPU sample",
    )

    _require(
        len(control_lines) >= 7,
        "Slurm log is missing frozen control lines",
    )
    start_match = re.fullmatch(
        r"START ([^ ]+) host=(nvl72d[0-9]{3}-T[0-9]{2}) job=([0-9]+)",
        control_lines[0],
    )
    if start_match is None:
        raise AdjudicationFailure("Slurm START line differs")
    start_timestamp = _require_log_timestamp(
        start_match.group(1),
        "Slurm START timestamp",
    )
    _require(start_match.group(3) == job_id, "Slurm START job ID differs")
    _require(
        control_lines[1]
        == (
            f"VERSIONS {EXPECTED_RUNTIME_STRINGS['numpy']} "
            f"{EXPECTED_RUNTIME_STRINGS['torch']}"
        ),
        "Slurm VERSIONS line differs",
    )
    _require(
        control_lines[2:4]
        == [
            EXPECTED_EXPERIMENTAL_WARNING,
            EXPECTED_EXPERIMENTAL_WARNING_CONTINUATION,
        ],
        "Slurm experimental-warning lines differ",
    )
    control_lines = control_lines[:2] + control_lines[4:]

    cases = summary.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == len(CASE_IDS),
        "Slurm-log case source is missing or malformed",
    )
    expected_scientific_lines: list[str] = []
    for index, (case, case_id) in enumerate(zip(cases, CASE_IDS, strict=True), start=1):
        _require(isinstance(case, Mapping), "Slurm-log case source is malformed")
        validity_passed = case.get("validity_passed")
        outcome = case.get("decision_outcome")
        _require(
            validity_passed is True
            and outcome
            in {
                FULL_AND_DERIVED_OUTCOME,
                FULL_ONLY_OUTCOME,
                CANONICAL_REPAIR_REFUTED,
            },
            "Slurm-log case result is outside the successful diagnostic contract",
        )
        expected_scientific_lines.extend(
            (
                f"CANONICAL_CASE_START case={case_id}",
                f"CANONICAL_PRECISION_START case={case_id} precision=bfloat16",
                f"CANONICAL_PRECISION_START case={case_id} precision=float32",
                "CANONICAL_CASE_DONE "
                f"case={case_id} validity_passed=True outcome={outcome}",
                f"COMPLETED_UNITS={index}/4 case={case_id}",
            )
        )

    expected_scientific_lines.extend(
        (
            f"{summary.get('status')} json={EXPECTED_OUTPUT_JSON_PATH} "
            f"npz={EXPECTED_OUTPUT_NPZ_PATH}",
            f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.json: OK",
            f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.npz: OK",
        )
    )
    _require(
        control_lines[2 : 2 + len(expected_scientific_lines)]
        == expected_scientific_lines,
        "Slurm scientific/control line grammar differs",
    )
    tail = control_lines[2 + len(expected_scientific_lines) :]
    _require(len(tail) == 3, "Slurm success-marker lines differ")
    done_prefix = "HQC_NEUTRAL_CANONICAL_GEOMETRY_V5_DONE "
    _require(tail[0].startswith(done_prefix), "Slurm log done-marker line differs")
    timestamp = tail[0][len(done_prefix) :]
    done_timestamp = _require_log_timestamp(timestamp, "Slurm done timestamp")
    _require(
        tail[1:] == ["COMPLETED_UNITS=1/1 rc=0", "EXIT_CODE=0"],
        "Slurm completion/exit lines differ",
    )
    _require(
        done_timestamp >= start_timestamp
        and all(timestamp >= start_timestamp for _, timestamp in heartbeat_timestamps),
        "Slurm log timestamp precedes START",
    )


def _require_log_timestamp(value: str, label: str) -> datetime:
    _require(
        re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
            r"[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}",
            value,
        )
        is not None,
        f"{label} is not a second-resolution ISO-8601 timestamp",
    )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AdjudicationFailure(f"{label} is malformed") from error
    _require(
        parsed.tzinfo is not None and parsed.isoformat() == value,
        f"{label} is not canonical and timezone-aware",
    )
    return parsed


def _validate_canonical_npz_container(
    path: Path,
    *,
    expected_names: Sequence[str] | None = None,
) -> list[str]:
    payload = path.read_bytes()
    _require(
        len(payload) >= 22 and payload[:4] == b"PK\x03\x04",
        f"NPZ has a prefix or missing ZIP records: {path.name}",
    )
    with zipfile.ZipFile(path) as archive_zip:
        _require(not archive_zip.comment, f"NPZ has an archive comment: {path.name}")
        _require(
            payload[-22:-18] == b"PK\x05\x06",
            f"NPZ has trailing bytes or a missing end record: {path.name}",
        )
        infos = archive_zip.infolist()
        members = [info.filename for info in infos]
        _require(
            len(members) == len(set(members)),
            f"NPZ contains duplicate ZIP members: {path.name}",
        )
        _require(
            expected_names is None
            or members == [f"{name}.npy" for name in expected_names],
            f"NPZ member order differs: {path.name}",
        )
        previous_end = 0
        for info in infos:
            _require(
                info.header_offset == previous_end,
                f"NPZ has a prefix, gap, or orphan local record: {path.name}",
            )
            _require(
                not info.comment and not info.extra,
                f"NPZ member has comment or central extra metadata: {info.filename}",
            )
            _require(
                info.date_time == (1980, 1, 1, 0, 0, 0)
                and info.create_system == 3
                and info.create_version == 45
                and info.extract_version == 45
                and info.reserved == 0
                and info.flag_bits == 0
                and info.compress_type == zipfile.ZIP_STORED
                and info.volume == 0
                and info.internal_attr == 0
                and info.external_attr == 0x01800000,
                f"NPZ member metadata differs from canonical np.savez: {info.filename}",
            )
            (
                signature,
                extract_version,
                flags,
                compression,
                modified_time,
                modified_date,
                crc,
                compressed_size,
                file_size,
                filename_length,
                extra_length,
            ) = struct.unpack_from("<4s5H3I2H", payload, info.header_offset)
            filename_bytes = info.filename.encode("ascii")
            header_end = info.header_offset + 30
            local_filename = payload[header_end : header_end + filename_length]
            local_extra = payload[
                header_end + filename_length : header_end
                + filename_length
                + extra_length
            ]
            expected_extra = struct.pack(
                "<HHQQ",
                0x0001,
                16,
                info.file_size,
                info.compress_size,
            )
            _require(
                signature == b"PK\x03\x04"
                and extract_version == 45
                and flags == 0
                and compression == zipfile.ZIP_STORED
                and modified_time == 0
                and modified_date == 33
                and crc == info.CRC
                and compressed_size == 0xFFFFFFFF
                and file_size == 0xFFFFFFFF
                and local_filename == filename_bytes
                and local_extra == expected_extra,
                f"NPZ local header differs from canonical np.savez: {info.filename}",
            )
            previous_end = (
                header_end + filename_length + extra_length + info.compress_size
            )
        _require(
            previous_end == archive_zip.start_dir,
            f"NPZ has bytes outside canonical local members: {path.name}",
        )
        expected_central_size = sum(
            46 + len(info.filename.encode("ascii")) for info in infos
        )
        eocd_offset = len(payload) - 22
        _require(
            eocd_offset - archive_zip.start_dir == expected_central_size,
            f"NPZ central directory has extra bytes: {path.name}",
        )
        (
            eocd_signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack_from("<4s4H2IH", payload, eocd_offset)
        _require(
            eocd_signature == b"PK\x05\x06"
            and disk_number == 0
            and central_disk == 0
            and disk_entries == len(infos)
            and total_entries == len(infos)
            and central_size == expected_central_size
            and central_offset == archive_zip.start_dir
            and comment_length == 0,
            f"NPZ end record differs from canonical np.savez: {path.name}",
        )
    return members


def _load_npz_unique(
    path: Path,
    *,
    expected_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    members = _validate_canonical_npz_container(
        path,
        expected_names=expected_names,
    )
    _require(
        len(members) == len(set(members)),
        f"NPZ contains duplicate ZIP members: {path.name}",
    )
    _require(
        all(
            member.endswith(".npy") and "/" not in member and "\\" not in member
            for member in members
        ),
        f"NPZ contains a non-array ZIP member: {path.name}",
    )
    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        _require(
            len(names) == len(set(names)),
            f"NPZ contains duplicate members: {path.name}",
        )
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    _require(
        all(value.dtype != np.dtype("O") for value in arrays.values()),
        f"NPZ contains object arrays: {path.name}",
    )
    with zipfile.ZipFile(path) as archive_zip:
        for name, array in arrays.items():
            canonical = io.BytesIO()
            np.lib.format.write_array(
                canonical,
                array,
                allow_pickle=False,
            )
            _require(
                archive_zip.read(f"{name}.npy") == canonical.getvalue(),
                f"NPZ member bytes differ from canonical np.savez: {name}.npy",
            )
    return arrays


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                any(token in str(key).lower() for token in FORBIDDEN_KEY_TOKENS)
                and path not in ALLOWED_FORBIDDEN_TOKEN_PATHS
            ):
                found.append(path)
            found.extend(_forbidden_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_paths(nested, f"{prefix}[{index}]"))
    elif isinstance(value, str) and any(
        token in value.lower() for token in FORBIDDEN_KEY_TOKENS
    ):
        found.append(prefix)
    return found


def _require_runtime_string(value: Any, label: str) -> None:
    safe_ascii = False
    normalized = ""
    if type(value) is str:
        try:
            encoded = value.encode("ascii")
        except UnicodeEncodeError:
            encoded = b""
        safe_ascii = bool(
            encoded
            and all(32 <= byte <= 126 for byte in encoded)
            and not any(character in value for character in "\\%&#")
        )
        normalized = "".join(
            character for character in value.lower() if character.isalnum()
        )
    _require(
        type(value) is str
        and 0 < len(value) <= 512
        and safe_ascii
        and not any(
            token.lower() in value.lower()
            or "".join(character for character in token.lower() if character.isalnum())
            in normalized
            for token in FORBIDDEN_LOG_TOKENS
        ),
        f"{label} is missing, malformed, or contains forbidden vocabulary",
    )


def _require_utc_timestamp(value: Any, label: str) -> None:
    _require(type(value) is str, f"{label} is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AdjudicationFailure(f"{label} is not an ISO-8601 timestamp") from error
    _require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
        and parsed.isoformat() == value,
        f"{label} is not a canonical UTC timestamp",
    )


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    numerator = np.linalg.norm((left64 - right64).reshape(-1))
    denominator = max(
        float(np.linalg.norm(left64.reshape(-1))),
        float(np.linalg.norm(right64.reshape(-1))),
        1.0e-12,
    )
    return float(numerator / denominator)


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    _require(left.shape == right.shape, "Persisted comparison shapes differ")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "shape": list(left.shape),
        "left_dtype": f"torch.{left.dtype}",
        "right_dtype": f"torch.{right.dtype}",
        "exact": _array_bitwise_equal(left, right),
        "nonzero_count": int(np.count_nonzero(delta)),
        "maximum_absolute_difference": (
            float(np.max(np.abs(delta))) if delta.size else 0.0
        ),
        "relative_l2_difference": _relative_l2(left, right),
    }


def _require_float_match(observed: Any, reported: Any, label: str) -> None:
    _require(
        type(reported) is float,
        f"{label} has the wrong type",
    )
    observed_float = float(observed)
    reported_float = float(reported)
    _require(
        math.isfinite(observed_float) and math.isfinite(reported_float),
        f"{label} is non-finite",
    )
    _require(
        math.isclose(
            observed_float,
            reported_float,
            rel_tol=FLOAT_REL_TOLERANCE,
            abs_tol=FLOAT_ABS_TOLERANCE,
        ),
        f"{label} differs from independent recomputation",
    )


def _require_difference_match(
    observed: Mapping[str, Any],
    reported: Mapping[str, Any],
    label: str,
) -> None:
    _require_exact_keys(
        reported,
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
    _require(
        isinstance(reported["shape"], list)
        and all(type(value) is int for value in reported["shape"]),
        f"{label}.shape has the wrong type",
    )
    _require(
        type(reported["left_dtype"]) is str and type(reported["right_dtype"]) is str,
        f"{label} dtype labels have the wrong type",
    )
    _require(
        type(reported["exact"]) is bool,
        f"{label}.exact has the wrong type",
    )
    _require(
        type(reported["nonzero_count"]) is int and reported["nonzero_count"] >= 0,
        f"{label}.nonzero_count has the wrong type",
    )
    for key in ("shape", "left_dtype", "right_dtype", "exact", "nonzero_count"):
        _require(
            observed[key] == reported.get(key),
            f"{label}.{key} differs from independent recomputation",
        )
    for key in ("maximum_absolute_difference", "relative_l2_difference"):
        _require_float_match(observed[key], reported.get(key), f"{label}.{key}")


def _decision_outcome(
    *,
    validity_passed: bool,
    derived_passed: bool,
    full_passed: bool,
) -> str:
    if not validity_passed:
        return INVALID_DIAGNOSTIC
    if full_passed and derived_passed:
        return FULL_AND_DERIVED_OUTCOME
    if full_passed:
        return FULL_ONLY_OUTCOME
    return CANONICAL_REPAIR_REFUTED


def _case_prefix(case: Mapping[str, Any]) -> str:
    _require(
        type(case["cohort_ordinal"]) is int,
        "Case cohort ordinal has the wrong type",
    )
    _require(type(case["case_id"]) is str, "Case ID has the wrong type")
    return f"case_{case['cohort_ordinal']:02d}_{case['case_id']}"


def _expected_array_key_order(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    expected: list[str] = []
    for case in cases:
        prefix = _case_prefix(case)
        expected.extend(
            [
                f"{prefix}__selected_cell_ids_int64",
                f"{prefix}__canonical_cells_int64",
                f"{prefix}__canonical_points_float32",
                f"{prefix}__canonical_centroids_float32",
                f"{prefix}__canonical_areas_float32",
                f"{prefix}__canonical_normals_float32",
            ]
        )
        for precision in PRECISIONS:
            for mode in MODES:
                expected.extend(
                    f"{prefix}__{precision}_{mode}_{path}_{field}"
                    for path in ("primary", "fixed", "primary_replay")
                    for field in FIELDS
                )
            expected.extend(
                f"{prefix}__{precision}_historical_{path}_{field}"
                for path in PATHS
                for field in FIELDS
            )
            expected.extend(
                f"{prefix}__{precision}_historical_model_{path}_source_{name}"
                for path in PATHS
                for name in GEOMETRY_FIELDS
            )
    return expected


def _expected_array_keys(cases: Sequence[Mapping[str, Any]]) -> set[str]:
    return set(_expected_array_key_order(cases))


def _validate_array_manifest(
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    manifest = summary.get("npz_array_manifest")
    _require(isinstance(manifest, Mapping), "NPZ array manifest is missing")
    _require(set(manifest) == set(arrays), "NPZ array manifest key set differs")
    for key, value in arrays.items():
        record = manifest[key]
        _require(
            isinstance(record, Mapping),
            f"{key} array-manifest record is malformed",
        )
        _require_exact_keys(
            record,
            {"shape", "dtype", "sha256"},
            f"{key} array-manifest record",
        )
        _require(
            isinstance(record["shape"], list)
            and all(
                type(dimension) is int and dimension >= 0
                for dimension in record["shape"]
            ),
            f"{key} array-manifest shape has the wrong type",
        )
        _require(record.get("shape") == list(value.shape), f"{key} shape differs")
        _require(record.get("dtype") == str(value.dtype), f"{key} dtype differs")
        _require(record.get("sha256") == _sha256_array(value), f"{key} hash differs")


def _validate_array_schema(
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    prefix = _case_prefix(case)
    canonical_points = arrays[f"{prefix}__canonical_points_float32"]
    expected_field_shapes = {
        "pressure": (2500,),
        "wss": (2500, 3),
    }
    expected_geometry_shapes = {
        "points": canonical_points.shape,
        "centroids": (2500, 3),
        "areas": (2500,),
        "normals": (2500, 3),
    }
    for precision in PRECISIONS:
        for mode in MODES:
            for path in ("primary", "fixed", "primary_replay"):
                for field, expected_shape in expected_field_shapes.items():
                    key = f"{prefix}__{precision}_{mode}_{path}_{field}"
                    value = arrays[key]
                    _require(
                        value.dtype == np.dtype("<f4")
                        and value.shape == expected_shape,
                        f"{key} dtype or shape differs",
                    )
                    _require(bool(np.isfinite(value).all()), f"{key} is non-finite")
        for path in PATHS:
            for field, expected_shape in expected_field_shapes.items():
                key = f"{prefix}__{precision}_historical_{path}_{field}"
                value = arrays[key]
                _require(
                    value.dtype == np.dtype("<f4") and value.shape == expected_shape,
                    f"{key} dtype or shape differs",
                )
                _require(bool(np.isfinite(value).all()), f"{key} is non-finite")
            for name, expected_shape in expected_geometry_shapes.items():
                key = f"{prefix}__{precision}_historical_model_{path}_source_{name}"
                value = arrays[key]
                _require(
                    value.dtype == np.dtype("<f4") and value.shape == expected_shape,
                    f"{key} dtype or shape differs",
                )
                _require(bool(np.isfinite(value).all()), f"{key} is non-finite")


def _validate_canonical_arrays(
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, float]:
    prefix = _case_prefix(case)
    ids = arrays[f"{prefix}__selected_cell_ids_int64"]
    cells = arrays[f"{prefix}__canonical_cells_int64"]
    points = arrays[f"{prefix}__canonical_points_float32"]
    centroids = arrays[f"{prefix}__canonical_centroids_float32"]
    areas = arrays[f"{prefix}__canonical_areas_float32"]
    normals = arrays[f"{prefix}__canonical_normals_float32"]
    _require(
        ids.dtype == np.dtype("<i8") and ids.shape == (2500,), "ID contract differs"
    )
    _require(
        cells.dtype == np.dtype("<i8") and cells.shape == (2500, 3),
        "Canonical cell contract differs",
    )
    _require(
        points.dtype == np.dtype("<f4") and points.ndim == 2 and points.shape[1] == 3,
        "Canonical point contract differs",
    )
    _require(
        centroids.dtype == np.dtype("<f4") and centroids.shape == (2500, 3),
        "Canonical centroid contract differs",
    )
    _require(
        areas.dtype == np.dtype("<f4") and areas.shape == (2500,),
        "Canonical area contract differs",
    )
    _require(
        normals.dtype == np.dtype("<f4") and normals.shape == (2500, 3),
        "Canonical normal contract differs",
    )
    _require(
        cells.size == 0
        or (int(cells.min()) >= 0 and int(cells.max()) < points.shape[0]),
        "Canonical connectivity is out of range",
    )
    _require(
        int(cells.max()) + 1 == points.shape[0]
        and _array_bitwise_equal(
            np.unique(cells),
            np.arange(points.shape[0], dtype=np.int64),
        ),
        "Canonical compact point order/count differs",
    )
    for name, value in (
        ("points", points),
        ("centroids", centroids),
        ("areas", areas),
        ("normals", normals),
    ):
        _require(bool(np.isfinite(value).all()), f"Canonical {name} are non-finite")
    _require(bool(np.all(areas > 0.0)), "Canonical areas are not strictly positive")
    unit_deviation = float(
        np.max(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1) - 1.0))
    )
    weighted_center = np.einsum(
        "n,nd->d",
        areas.astype(np.float64),
        centroids.astype(np.float64),
    ) / float(np.sum(areas.astype(np.float64)))
    center_deviation = float(np.max(np.abs(weighted_center)))
    _require(unit_deviation <= 1.0e-6, "Canonical normals are not unit length")
    _require(center_deviation <= 1.0e-6, "Canonical centroids are not area-centred")
    triangles = points.astype(np.float64)[cells]
    edges_1 = triangles[:, 1] - triangles[:, 0]
    edges_2 = triangles[:, 2] - triangles[:, 0]
    crosses = np.cross(edges_1, edges_2)
    twice_areas = np.linalg.norm(crosses, axis=1)
    _require(
        bool(np.isfinite(twice_areas).all() and np.all(twice_areas > 0.0)),
        "Canonical point/connectivity geometry is degenerate",
    )
    recomputed_centroids = np.mean(triangles, axis=1)
    recomputed_areas = 0.5 * twice_areas
    recomputed_normals = crosses / twice_areas[:, None]
    centroid_max_abs = float(
        np.max(np.abs(recomputed_centroids - centroids.astype(np.float64)))
    )
    centroid_relative = _relative_l2(recomputed_centroids, centroids)
    area_relative = _relative_l2(recomputed_areas, areas)
    normal_relative = _relative_l2(recomputed_normals, normals)
    normal_dots = np.einsum(
        "nd,nd->n",
        recomputed_normals,
        normals.astype(np.float64),
    )
    minimum_normal_dot = float(np.min(normal_dots))
    _require(
        centroid_max_abs <= CANONICAL_CENTROID_MAX_ABS_TOLERANCE
        and centroid_relative <= CANONICAL_CENTROID_RELATIVE_TOLERANCE,
        "Canonical centroids are incoherent with points/connectivity",
    )
    _require(
        area_relative <= CANONICAL_AREA_RELATIVE_TOLERANCE,
        "Canonical areas are incoherent with points/connectivity",
    )
    _require(
        normal_relative <= CANONICAL_NORMAL_RELATIVE_TOLERANCE
        and minimum_normal_dot >= CANONICAL_NORMAL_MINIMUM_DOT,
        "Canonical normals are incoherent with points/connectivity",
    )
    frame = case.get("canonical_frame", {})
    _require(
        type(frame.get("physical_length")) is float
        and frame["physical_length"] == EXPECTED_PHYSICAL_LENGTH,
        "Physical length contract differs",
    )
    _require(
        type(frame.get("model_reference_length")) is float
        and frame["model_reference_length"] == EXPECTED_MODEL_REFERENCE_LENGTH,
        "Model reference length contract differs",
    )
    _require(
        type(frame.get("effective_physical_length")) is float
        and frame["effective_physical_length"] == EXPECTED_EFFECTIVE_LENGTH,
        "Effective length contract differs",
    )
    return {
        "maximum_unit_deviation": unit_deviation,
        "maximum_area_center_deviation": center_deviation,
        "point_connectivity_centroid_maximum_absolute_difference": (centroid_max_abs),
        "point_connectivity_centroid_relative_l2": centroid_relative,
        "point_connectivity_area_relative_l2": area_relative,
        "point_connectivity_normal_relative_l2": normal_relative,
        "point_connectivity_minimum_normal_dot": minimum_normal_dot,
    }


def _validate_summary_schema_and_provenance(
    summary: Mapping[str, Any],
    *,
    npz_sha256: str,
    prior_npz_sha256: str,
) -> None:
    _require_exact_keys(
        summary,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "decision_outcome",
            "generated_at_utc",
            "scientific_scope",
            "contract",
            "validity",
            "decision_gates",
            "cases",
            "npz_array_manifest",
            "provenance",
        },
        "diagnostic",
    )
    _require_utc_timestamp(summary["generated_at_utc"], "generated_at_utc")
    scope = summary["scientific_scope"]
    _require_exact_keys(
        scope,
        {
            "case_ids",
            "resolution",
            "precisions",
            "supervision_arrays_indexed",
            "synthetic_placeholders_stripped_before_model",
            "hqc_decision_statistics_computed",
            "may_not_be_used_as_hqc_verdict_output",
        },
        "scientific_scope",
    )
    _require(tuple(scope["case_ids"]) == CASE_IDS, "Scope cases differ")
    _require(
        type(scope["resolution"]) is int and scope["resolution"] == 2500,
        "Scope resolution differs",
    )
    _require(tuple(scope["precisions"]) == PRECISIONS, "Scope precisions differ")
    _require(scope["supervision_arrays_indexed"] is False, "Supervision scope differs")
    _require(
        scope["synthetic_placeholders_stripped_before_model"] is True,
        "Placeholder stripping scope differs",
    )
    _require(
        scope["hqc_decision_statistics_computed"] is False,
        "H-QC decision-statistic scope differs",
    )
    _require(
        scope["may_not_be_used_as_hqc_verdict_output"] is True,
        "H-QC verdict-use scope differs",
    )
    contract = summary["contract"]
    _require_exact_keys(
        contract,
        {
            "canonical_construction",
            "canonical_derived_fields",
            "canonical_full_fields",
            "query_frame",
            "derived_fieldwise_relative_tolerance",
            "full_comparison",
        },
        "contract",
    )
    _require(
        tuple(contract["canonical_derived_fields"])
        == ("centroids", "areas", "normals"),
        "Derived-field contract differs",
    )
    _require(
        contract["canonical_construction"]
        == (
            "float64 raw geometry -> physical area center -> divide by "
            "L_ref*model_reference_length -> one float32 cast"
        ),
        "Canonical construction contract differs",
    )
    _require(
        tuple(contract["canonical_full_fields"])
        == ("points", "centroids", "areas", "normals"),
        "Full-field contract differs",
    )
    _require(
        contract["query_frame"] == "canonical_trace_centroids",
        "Query-frame contract differs",
    )
    _require(
        type(contract["derived_fieldwise_relative_tolerance"]) is float
        and contract["derived_fieldwise_relative_tolerance"] == DERIVED_TOLERANCE,
        "Derived tolerance contract differs",
    )
    _require(
        contract["full_comparison"] == "fieldwise_bitwise_exact",
        "Full comparison contract differs",
    )
    provenance = summary["provenance"]
    _require_exact_keys(
        provenance,
        {
            "command",
            "diagnostic_script_path",
            "diagnostic_script_sha256",
            "frozen_producer_path",
            "frozen_producer_sha256",
            "source_tree_manifest_sha256",
            "selected_source_files",
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
        "provenance",
    )
    _require(
        isinstance(provenance["command"], list)
        and all(type(value) is str for value in provenance["command"])
        and tuple(provenance["command"]) == EXPECTED_COMMAND,
        "Frozen diagnostic command differs",
    )
    _require(
        provenance["diagnostic_script_path"] == EXPECTED_DIAGNOSTIC_SCRIPT_PATH,
        "Frozen diagnostic path differs",
    )
    _require(
        provenance["frozen_producer_path"] == EXPECTED_FROZEN_PRODUCER_PATH,
        "Frozen producer path differs",
    )
    _require(
        provenance["npz_path"] == EXPECTED_OUTPUT_NPZ_PATH,
        "Frozen output NPZ path differs",
    )
    _require(
        provenance["diagnostic_script_sha256"] == EXPECTED_DIAGNOSTIC_SCRIPT_SHA256,
        "Frozen diagnostic digest differs",
    )
    _require(
        provenance["frozen_producer_sha256"] == EXPECTED_PRODUCER_SHA256,
        "Frozen producer digest differs",
    )
    _require(
        provenance["source_tree_manifest_sha256"] == EXPECTED_SOURCE_TREE_SHA256,
        "Frozen source-tree digest differs",
    )
    _require(
        provenance["input_hashes"] == EXPECTED_INPUT_HASHES,
        "Frozen input-hash mapping differs",
    )
    _require(
        provenance["selected_source_files"] == EXPECTED_SELECTED_SOURCE_FILES,
        "Selected source-file hashes differ",
    )
    _require(provenance["npz_sha256"] == npz_sha256, "Provenance NPZ digest differs")
    _require(
        provenance["input_hashes"]["prior_diagnostic_npz"] == prior_npz_sha256,
        "Prior diagnostic NPZ digest differs",
    )
    for name, expected in EXPECTED_RUNTIME_STRINGS.items():
        _require_runtime_string(provenance[name], f"provenance.{name}")
        _require(
            provenance[name] == expected,
            f"provenance.{name} differs from the frozen AGA runtime",
        )
    hardware = provenance["hardware"]
    _require(
        isinstance(hardware, Mapping),
        "Hardware provenance is not an object",
    )
    _require_exact_keys(
        hardware,
        {
            "cuda_runtime",
            "cuda_device_name",
            "cuda_device_capability",
        },
        "provenance.hardware",
    )
    _require_runtime_string(
        hardware["cuda_runtime"],
        "provenance.hardware.cuda_runtime",
    )
    _require(
        hardware["cuda_runtime"] == EXPECTED_HARDWARE["cuda_runtime"],
        "CUDA runtime differs from the frozen AGA runtime",
    )
    _require_runtime_string(
        hardware["cuda_device_name"],
        "provenance.hardware.cuda_device_name",
    )
    _require(
        hardware["cuda_device_name"] == EXPECTED_HARDWARE["cuda_device_name"],
        "CUDA device name differs from the frozen AGA runtime",
    )
    capability = hardware["cuda_device_capability"]
    _require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(value) is int for value in capability)
        and capability == EXPECTED_HARDWARE["cuda_device_capability"],
        "CUDA device capability differs from the frozen AGA runtime",
    )
    slurm_job_id = provenance["slurm_job_id"]
    _require(
        type(slurm_job_id) is str and slurm_job_id.isdigit(),
        "Slurm job ID is missing or malformed",
    )


def _validate_reported_validity(
    case: Mapping[str, Any],
    *,
    canonical_checks: Mapping[str, float],
) -> bool:
    prefix = _case_prefix(case)
    validity = case["validity"]
    _require_exact_keys(
        validity,
        {
            "canonical_bundle",
            "canonical_construction_replay",
            "canonical_construction_replay_passed",
            "historical_path_topology",
            "historical_path_topology_passed",
            "job304002_geometry_replay",
            "job304002_geometry_replay_passed",
            "model_local_data_stripped",
            "model_probes_executed",
        },
        f"{prefix}.validity",
    )
    bundle = validity["canonical_bundle"]
    _require_exact_keys(
        bundle,
        {
            "passed",
            "checks",
            "shape_checks",
            "finite_checks",
            "maximum_unit_deviation",
            "maximum_area_center_deviation",
        },
        f"{prefix}.canonical_bundle",
    )
    _require_exact_keys(
        bundle["checks"],
        {
            "shapes",
            "topology",
            "finite",
            "positive_areas",
            "unit_normals",
            "area_centered",
        },
        f"{prefix}.canonical_bundle.checks",
    )
    _require_exact_keys(
        bundle["shape_checks"],
        {"points", "cells", "centroids", "areas", "normals"},
        f"{prefix}.canonical_bundle.shape_checks",
    )
    _require_exact_keys(
        bundle["finite_checks"],
        {"points", "centroids", "areas", "normals"},
        f"{prefix}.canonical_bundle.finite_checks",
    )
    _require_float_match(
        canonical_checks["maximum_unit_deviation"],
        bundle["maximum_unit_deviation"],
        f"{prefix}.canonical_bundle.maximum_unit_deviation",
    )
    _require_float_match(
        canonical_checks["maximum_area_center_deviation"],
        bundle["maximum_area_center_deviation"],
        f"{prefix}.canonical_bundle.maximum_area_center_deviation",
    )
    independent_bundle_checks = {
        "shapes": True,
        "finite": True,
        "positive_areas": True,
        "unit_normals": (canonical_checks["maximum_unit_deviation"] <= 1.0e-6),
        "area_centered": (canonical_checks["maximum_area_center_deviation"] <= 1.0e-6),
    }
    _require(
        _all_bool_values(
            bundle["shape_checks"],
            f"{prefix}.canonical_bundle.shape_checks",
        )
        and _all_bool_values(
            bundle["finite_checks"],
            f"{prefix}.canonical_bundle.finite_checks",
        ),
        f"{prefix} reports a failed canonical array subgate",
    )
    for key, observed in independent_bundle_checks.items():
        _require(
            bundle["checks"][key] is observed,
            f"{prefix}.canonical_bundle.checks.{key} differs",
        )
    _require(
        isinstance(bundle["checks"]["topology"], bool),
        f"{prefix}.canonical_bundle topology report is malformed",
    )
    bundle_passed = _all_bool_values(
        bundle["checks"],
        f"{prefix}.canonical_bundle.checks",
    )
    _require(
        bundle["passed"] is bundle_passed,
        f"{prefix}.canonical_bundle aggregate differs",
    )
    construction = validity["canonical_construction_replay"]
    _require_exact_keys(
        construction,
        {
            "cells",
            "points",
            "centroids",
            "areas",
            "normals",
            "physical_center",
            "physical_length",
            "model_reference_length",
        },
        f"{prefix}.canonical_construction_replay",
    )
    construction_passed = _all_bool_values(
        construction,
        f"{prefix}.canonical_construction_replay",
    )
    _require(
        validity["canonical_construction_replay_passed"] is construction_passed,
        f"{prefix} construction-replay aggregate differs",
    )
    topology = validity["historical_path_topology"]
    _require_exact_keys(
        topology,
        {
            "primary_matches_selected",
            "fixed_matches_selected",
            "primary_matches_fixed",
        },
        f"{prefix}.historical_path_topology",
    )
    topology_passed = _all_bool_values(
        topology,
        f"{prefix}.historical_path_topology",
    )
    _require(
        validity["historical_path_topology_passed"] is topology_passed,
        f"{prefix} topology aggregate differs",
    )
    prior_geometry = validity["job304002_geometry_replay"]
    _require_exact_keys(
        prior_geometry,
        {
            "cell_ids_int64",
            "pipeline_primary_points_float32",
            "pipeline_fixed_points_float32",
            "pipeline_primary_queries_float32",
            "pipeline_fixed_queries_float32",
        },
        f"{prefix}.job304002_geometry_replay",
    )
    prior_geometry_passed = _all_bool_values(
        prior_geometry,
        f"{prefix}.job304002_geometry_replay",
    )
    _require(
        validity["job304002_geometry_replay_passed"] is prior_geometry_passed,
        f"{prefix} prior-geometry aggregate differs",
    )
    _require(
        validity["model_local_data_stripped"] is True,
        f"{prefix} local-data stripping report failed",
    )
    safe_to_run_model = bundle_passed and topology_passed
    _require(
        validity["model_probes_executed"] is safe_to_run_model,
        f"{prefix} model-probe execution report differs",
    )
    probes = case["precision_probes"]
    _require(set(probes) == set(PRECISIONS), f"{prefix} precision probes differ")
    precision_validity: list[bool] = []
    for precision in PRECISIONS:
        precision_summary = probes[precision]
        _require_exact_keys(
            precision_summary,
            {
                "precision",
                "modes",
                "validity_passed",
                "decision_gates",
                "job304002_historical_replay",
            },
            f"{prefix}.{precision}",
        )
        _require(
            precision_summary["precision"] == precision,
            f"{prefix}.{precision} label differs",
        )
        _require(
            set(precision_summary["modes"]) == set(MODES),
            f"{prefix}.{precision} mode set differs",
        )
        mode_validity: list[bool] = []
        for mode in MODES:
            mode_summary = precision_summary["modes"][mode]
            _require_exact_keys(
                mode_summary,
                {
                    "mode",
                    "primary_fixed_difference",
                    "primary_replay_difference",
                    "primary_replay_exact",
                    "injected_geometry_exact",
                    "canonical_decode_contract",
                    "canonical_decode_contract_passed",
                    "comparison_gate",
                    "validity_passed",
                },
                f"{prefix}.{precision}.{mode}",
            )
            _require(
                mode_summary["mode"] == mode,
                f"{prefix}.{precision}.{mode} label differs",
            )
            expected_geometry_fields = (
                {"centroids", "areas", "normals"}
                if mode == "canonical_derived"
                else {"points", "centroids", "areas", "normals"}
            )
            injections = mode_summary["injected_geometry_exact"]
            _require_exact_keys(
                injections,
                {"primary", "fixed"},
                f"{prefix}.{precision}.{mode}.injections",
            )
            injection_passed = True
            for path in PATHS:
                _require_exact_keys(
                    injections[path],
                    expected_geometry_fields,
                    f"{prefix}.{precision}.{mode}.injections.{path}",
                )
                injection_passed = injection_passed and _all_bool_values(
                    injections[path],
                    f"{prefix}.{precision}.{mode}.injections.{path}",
                )
            decode = mode_summary["canonical_decode_contract"]
            _require_exact_keys(
                decode,
                {"primary", "fixed", "primary_replay"},
                f"{prefix}.{precision}.{mode}.decode",
            )
            decode_passed = True
            for path, checks in decode.items():
                _require_exact_keys(
                    checks,
                    {
                        "canonical_queries_exact",
                        "encoded_center_is_exact_zero",
                        "encoded_reference_length_is_exact_one",
                    },
                    f"{prefix}.{precision}.{mode}.decode.{path}",
                )
                decode_passed = decode_passed and _all_bool_values(
                    checks,
                    f"{prefix}.{precision}.{mode}.decode.{path}",
                )
            _require(
                mode_summary["canonical_decode_contract_passed"] is decode_passed,
                f"{prefix}.{precision}.{mode} decode aggregate differs",
            )
            replay_exact = mode_summary["primary_replay_exact"]
            _require(
                type(replay_exact) is bool,
                f"{prefix}.{precision}.{mode} replay report is non-boolean",
            )
            reported_mode_validity = replay_exact and injection_passed and decode_passed
            _require(
                mode_summary["validity_passed"] is reported_mode_validity,
                f"{prefix}.{precision}.{mode} validity aggregate differs",
            )
            mode_validity.append(reported_mode_validity)
        replay_value = precision_summary["job304002_historical_replay"]["passed"]
        _require(
            type(replay_value) is bool,
            f"{prefix}.{precision} prior replay report is non-boolean",
        )
        replay_passed = replay_value
        observed_precision_validity = all(mode_validity) and replay_passed
        _require(
            precision_summary["validity_passed"] is observed_precision_validity,
            f"{prefix}.{precision} validity aggregate differs",
        )
        precision_validity.append(observed_precision_validity)
    observed_case_validity = (
        bundle_passed
        and construction_passed
        and topology_passed
        and prior_geometry_passed
        and validity["model_local_data_stripped"] is True
        and safe_to_run_model
        and all(precision_validity)
    )
    _require(
        case["validity_passed"] is observed_case_validity,
        f"{prefix} validity aggregate differs",
    )
    return observed_case_validity


def _validate_prior_replay(
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    prior_arrays: Mapping[str, np.ndarray],
) -> None:
    prefix = _case_prefix(case)
    reported_geometry = case["validity"]["job304002_geometry_replay"]
    current_ids = arrays[f"{prefix}__selected_cell_ids_int64"]
    prior_ids = prior_arrays[f"{prefix}__cell_ids_int64"]
    ids_exact = _array_bitwise_equal(current_ids, prior_ids)
    _require(ids_exact, f"{prefix} selected IDs do not replay job 304002")
    _require(
        reported_geometry.get("cell_ids_int64") is ids_exact,
        f"{prefix} selected-ID replay report differs",
    )
    for precision in PRECISIONS:
        replay = case["precision_probes"][precision]["job304002_historical_replay"]
        _require_exact_keys(
            replay,
            {
                "job304002_primary_fixed_predictions",
                "job304002_model_source_geometry",
                "passed",
            },
            f"{prefix}.{precision}.job304002_historical_replay",
        )
        _require(
            set(replay["job304002_primary_fixed_predictions"]) == set(PATHS)
            and set(replay["job304002_model_source_geometry"]) == set(PATHS),
            f"{prefix}.{precision} prior-replay paths differ",
        )
        all_exact = True
        for path in PATHS:
            _require(
                set(replay["job304002_primary_fixed_predictions"][path]) == set(FIELDS),
                f"{prefix}.{precision}.{path} prior prediction fields differ",
            )
            _require(
                set(replay["job304002_model_source_geometry"][path])
                == set(GEOMETRY_FIELDS),
                f"{prefix}.{precision}.{path} prior geometry fields differ",
            )
            for field in FIELDS:
                current_key = f"{prefix}__{precision}_historical_{path}_{field}"
                prior_key = f"{prefix}__{precision}_{path}_{field}"
                observed = _difference(arrays[current_key], prior_arrays[prior_key])
                reported = replay["job304002_primary_fixed_predictions"][path][field]
                _require_difference_match(
                    observed,
                    reported,
                    f"{prefix}.{precision}.prior_prediction.{path}.{field}",
                )
                all_exact = all_exact and bool(observed["exact"])
            for name in GEOMETRY_FIELDS:
                current_key = (
                    f"{prefix}__{precision}_historical_model_{path}_source_{name}"
                )
                prior_key = f"{prefix}__{precision}_model_{path}_source_{name}"
                observed = _difference(arrays[current_key], prior_arrays[prior_key])
                reported = replay["job304002_model_source_geometry"][path][name]
                _require_difference_match(
                    observed,
                    reported,
                    f"{prefix}.{precision}.prior_geometry.{path}.{name}",
                )
                all_exact = all_exact and bool(observed["exact"])
        _require(all_exact, f"{prefix}.{precision} does not replay job 304002")
        _require(
            replay.get("passed") is all_exact,
            f"{prefix}.{precision} replay status differs",
        )


def _validate_prediction_comparisons(
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[bool, bool, dict[str, dict[str, float]]]:
    prefix = _case_prefix(case)
    _require(
        type(case["validity_passed"]) is bool,
        f"{prefix} validity report is non-boolean",
    )
    derived_passed = True
    full_passed = True
    maxima = {
        "canonical_derived": {"pressure": 0.0, "wss": 0.0},
        "canonical_full": {"pressure": 0.0, "wss": 0.0},
    }
    for precision in PRECISIONS:
        precision_summary = case["precision_probes"][precision]
        for mode in MODES:
            mode_summary = precision_summary["modes"][mode]
            _require(
                set(mode_summary["primary_fixed_difference"]) == set(FIELDS)
                and set(mode_summary["primary_replay_difference"]) == set(FIELDS),
                f"{prefix}.{precision}.{mode} comparison fields differ",
            )
            comparison_passed = True
            replay_exact = True
            for field in FIELDS:
                primary_key = f"{prefix}__{precision}_{mode}_primary_{field}"
                fixed_key = f"{prefix}__{precision}_{mode}_fixed_{field}"
                replay_key = f"{prefix}__{precision}_{mode}_primary_replay_{field}"
                primary_fixed = _difference(
                    arrays[primary_key],
                    arrays[fixed_key],
                )
                primary_replay = _difference(
                    arrays[primary_key],
                    arrays[replay_key],
                )
                _require_difference_match(
                    primary_fixed,
                    mode_summary["primary_fixed_difference"][field],
                    f"{prefix}.{precision}.{mode}.primary_fixed.{field}",
                )
                _require_difference_match(
                    primary_replay,
                    mode_summary["primary_replay_difference"][field],
                    f"{prefix}.{precision}.{mode}.primary_replay.{field}",
                )
                relative = float(primary_fixed["relative_l2_difference"])
                maxima[mode][field] = max(maxima[mode][field], relative)
                replay_exact = replay_exact and bool(primary_replay["exact"])
                if mode == "canonical_derived":
                    comparison_passed = comparison_passed and (
                        relative <= DERIVED_TOLERANCE
                    )
                else:
                    comparison_passed = comparison_passed and bool(
                        primary_fixed["exact"]
                    )
            _require(replay_exact, f"{prefix}.{precision}.{mode} replay is not exact")
            _require(
                mode_summary.get("primary_replay_exact") is replay_exact,
                f"{prefix}.{precision}.{mode} replay status differs",
            )
            comparison_gate = mode_summary["comparison_gate"]
            _require_exact_keys(
                comparison_gate,
                {"criterion", "passed"},
                f"{prefix}.{precision}.{mode}.comparison_gate",
            )
            expected_criterion = (
                "fieldwise_relative_l2_le_1e-3"
                if mode == "canonical_derived"
                else "fieldwise_bitwise_exact"
            )
            _require(
                comparison_gate["criterion"] == expected_criterion,
                f"{prefix}.{precision}.{mode} criterion differs",
            )
            _require(
                comparison_gate["passed"] is comparison_passed,
                f"{prefix}.{precision}.{mode} comparison gate differs",
            )
            if mode == "canonical_derived":
                derived_passed = derived_passed and comparison_passed
            else:
                full_passed = full_passed and comparison_passed
        reported = precision_summary["decision_gates"]
        _require_exact_keys(
            reported,
            {"derived_passed", "full_passed"},
            f"{prefix}.{precision}.decision_gates",
        )
        _require(
            reported.get("derived_passed")
            is all(
                float(
                    _difference(
                        arrays[
                            f"{prefix}__{precision}_canonical_derived_primary_{field}"
                        ],
                        arrays[
                            f"{prefix}__{precision}_canonical_derived_fixed_{field}"
                        ],
                    )["relative_l2_difference"]
                )
                <= DERIVED_TOLERANCE
                for field in FIELDS
            ),
            f"{prefix}.{precision} derived gate differs",
        )
        _require(
            reported.get("full_passed")
            is all(
                bool(
                    _difference(
                        arrays[f"{prefix}__{precision}_canonical_full_primary_{field}"],
                        arrays[f"{prefix}__{precision}_canonical_full_fixed_{field}"],
                    )["exact"]
                )
                for field in FIELDS
            ),
            f"{prefix}.{precision} full gate differs",
        )
    _require_exact_keys(
        case["decision_gates"],
        {"derived_passed", "full_passed"},
        f"{prefix}.decision_gates",
    )
    _require(
        case["decision_gates"]["derived_passed"] is derived_passed,
        f"{prefix} aggregate derived gate differs",
    )
    _require(
        case["decision_gates"]["full_passed"] is full_passed,
        f"{prefix} aggregate full gate differs",
    )
    expected_outcome = _decision_outcome(
        validity_passed=case["validity_passed"],
        derived_passed=derived_passed,
        full_passed=full_passed,
    )
    _require(
        case.get("decision_outcome") == expected_outcome,
        f"{prefix} outcome differs",
    )
    return derived_passed, full_passed, maxima


def adjudicate(
    *,
    bundle_manifest: Path,
    diagnostic_json: Path,
    diagnostic_npz: Path,
    prior_diagnostic_npz: Path,
) -> dict[str, Any]:
    diagnostic_json = _resolve_regular_input(diagnostic_json, "Diagnostic JSON")
    diagnostic_npz = _resolve_regular_input(diagnostic_npz, "Diagnostic NPZ")
    prior_diagnostic_npz = _resolve_regular_input(
        prior_diagnostic_npz,
        "Prior diagnostic NPZ",
    )
    bundle_manifest_sha256, bundle_members = _verify_bundle_manifest(bundle_manifest)
    bundle_root = _resolve_regular_input(
        bundle_manifest,
        "Copied-bundle manifest",
    ).parent
    _require(
        diagnostic_json
        == bundle_root / "artifacts" / f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.json",
        "Diagnostic JSON is outside the frozen wrapper layout",
    )
    _require(
        diagnostic_npz
        == bundle_root / "artifacts" / f"{EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.npz",
        "Diagnostic NPZ is outside the frozen wrapper layout",
    )
    _require(
        prior_diagnostic_npz == bundle_root / EXPECTED_PRIOR_NPZ_BASENAME,
        "Prior diagnostic NPZ is outside the frozen wrapper layout",
    )
    for required_path in (
        diagnostic_json,
        diagnostic_npz,
        prior_diagnostic_npz,
        diagnostic_json.with_name(f"{diagnostic_json.name}.sha256"),
        diagnostic_npz.with_name(f"{diagnostic_npz.name}.sha256"),
        prior_diagnostic_npz.with_name(f"{prior_diagnostic_npz.name}.sha256"),
    ):
        _require(
            required_path in bundle_members,
            f"Required input is absent from copied-bundle manifest: {required_path.name}",
        )
    json_sha256 = _verify_sidecar(diagnostic_json)
    npz_sha256 = _verify_sidecar(diagnostic_npz)
    prior_npz_sha256 = _verify_sidecar(prior_diagnostic_npz)
    _require(
        prior_npz_sha256 == EXPECTED_PRIOR_NPZ_SHA256,
        "Supplied prior NPZ is not frozen job 304002",
    )
    summary = _load_json_unique(diagnostic_json)
    _require(
        type(summary.get("schema_version")) is int
        and summary["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION,
        "Diagnostic schema version differs",
    )
    _require(
        summary.get("artifact_kind") == DIAGNOSTIC_ARTIFACT_KIND,
        "Diagnostic artifact kind differs",
    )
    _validate_summary_schema_and_provenance(
        summary,
        npz_sha256=npz_sha256,
        prior_npz_sha256=prior_npz_sha256,
    )
    job_id = str(summary["provenance"]["slurm_job_id"])
    _validate_completed_job_markers(
        bundle_members,
        bundle_root,
        job_id,
    )
    _validate_bundle_allowlist(
        bundle_members,
        bundle_root,
        job_id,
    )
    _require(
        bundle_root.name == _expected_copied_bundle_root_basename(job_id),
        "Copied-bundle root basename differs",
    )
    _validate_slurm_log_content(
        bundle_root,
        job_id,
        summary,
    )
    cases = summary.get("cases")
    _require(isinstance(cases, list), "Diagnostic cases are missing")
    for case in cases:
        _require(
            isinstance(case, Mapping)
            and type(case.get("cohort_ordinal")) is int
            and type(case.get("case_id")) is str
            and type(case.get("reader_index")) is int,
            "Diagnostic case identity has the wrong type",
        )
    _require(
        tuple(
            (
                case["cohort_ordinal"],
                case.get("case_id"),
                case["reader_index"],
            )
            for case in cases
        )
        == CASE_SPECS,
        "Diagnostic case identities or order differ",
    )
    for case in cases:
        prefix = _case_prefix(case)
        _require_exact_keys(
            case,
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
            prefix,
        )
        _require(
            type(case["resolution"]) is int and case["resolution"] == 2500,
            f"{prefix} resolution differs",
        )
        _require_exact_keys(
            case["canonical_frame"],
            {
                "construction",
                "physical_center_float64",
                "physical_length",
                "model_reference_length",
                "effective_physical_length",
                "queries",
            },
            f"{prefix}.canonical_frame",
        )
        _require(
            case["canonical_frame"]["queries"] == "canonical_trace_centroids",
            f"{prefix} canonical query label differs",
        )
        _require(
            case["canonical_frame"]["construction"]
            == EXPECTED_CANONICAL_FRAME_CONSTRUCTION,
            f"{prefix} canonical construction label differs",
        )
        _require(
            len(case["canonical_frame"]["physical_center_float64"]) == 3
            and all(
                type(value) is float and math.isfinite(value)
                for value in case["canonical_frame"]["physical_center_float64"]
            ),
            f"{prefix} physical center is malformed",
        )
        _require_exact_keys(
            case["historical_centers"],
            {
                "primary_point_mean_float32",
                "fixed_s10000_point_mean_float32",
            },
            f"{prefix}.historical_centers",
        )
        for name, center in case["historical_centers"].items():
            _require(
                isinstance(center, list)
                and len(center) == 3
                and all(
                    type(value) is float and math.isfinite(value) for value in center
                ),
                f"{prefix}.historical_centers.{name} is malformed",
            )
    disallowed_json_paths = _forbidden_paths(summary)
    _require(not disallowed_json_paths, "Diagnostic JSON has disallowed schema paths")
    arrays = _load_npz_unique(
        diagnostic_npz,
        expected_names=_expected_array_key_order(cases),
    )
    prior_arrays = _load_npz_unique(prior_diagnostic_npz)
    disallowed_array_keys = [
        key
        for key in arrays
        if any(token in key.lower() for token in FORBIDDEN_KEY_TOKENS)
    ]
    _require(not disallowed_array_keys, "Diagnostic NPZ has disallowed array keys")
    _require(
        set(arrays) == _expected_array_keys(cases),
        "Diagnostic NPZ key set differs from schema v5",
    )
    _validate_array_manifest(summary, arrays)

    all_derived_passed = True
    all_full_passed = True
    all_reported_validity_passed = True
    canonical_checks: dict[str, Any] = {}
    case_maxima: dict[str, Any] = {}
    for case in cases:
        case_id = str(case["case_id"])
        _validate_array_schema(case, arrays)
        canonical_checks[case_id] = _validate_canonical_arrays(case, arrays)
        _validate_prior_replay(case, arrays, prior_arrays)
        derived_passed, full_passed, maxima = _validate_prediction_comparisons(
            case,
            arrays,
        )
        reported_case_validity = _validate_reported_validity(
            case,
            canonical_checks=canonical_checks[case_id],
        )
        all_derived_passed = all_derived_passed and derived_passed
        all_full_passed = all_full_passed and full_passed
        all_reported_validity_passed = (
            all_reported_validity_passed and reported_case_validity
        )
        case_maxima[case_id] = maxima

    reported_validity = all_reported_validity_passed
    _require_exact_keys(
        summary["validity"],
        {"all_cases_and_precisions_passed", "required_gates"},
        "validity",
    )
    _require(
        summary["validity"]["required_gates"]
        == [
            "canonical_construction_replay",
            "shape",
            "topology",
            "finite_positive_unit",
            "job304002_geometry_and_primary_fixed_prediction_replay",
            "primary_replay_exact",
        ],
        "Top-level required validity gates differ",
    )
    _require(
        summary["validity"].get("all_cases_and_precisions_passed") is reported_validity,
        "Aggregate validity report differs from case reports",
    )
    _require_exact_keys(
        summary["decision_gates"],
        {"derived", "full"},
        "decision_gates",
    )
    for name in ("derived", "full"):
        _require_exact_keys(
            summary["decision_gates"][name],
            {"criterion", "passed"},
            f"decision_gates.{name}",
        )
    _require(
        summary["decision_gates"]["derived"]["criterion"]
        == (
            "primary-versus-fixed pressure and WSS relative L2 <= 1e-3 "
            "for every case and precision"
        ),
        "Aggregate derived criterion differs",
    )
    _require(
        summary["decision_gates"]["full"]["criterion"]
        == (
            "primary-versus-fixed pressure and WSS bitwise exact "
            "for every case and precision"
        ),
        "Aggregate full criterion differs",
    )
    _require(
        summary["decision_gates"]["derived"]["passed"] is all_derived_passed,
        "Aggregate derived gate differs",
    )
    _require(
        summary["decision_gates"]["full"]["passed"] is all_full_passed,
        "Aggregate full gate differs",
    )
    outcome = _decision_outcome(
        validity_passed=reported_validity,
        derived_passed=all_derived_passed,
        full_passed=all_full_passed,
    )
    _require(summary.get("decision_outcome") == outcome, "Aggregate outcome differs")
    expected_status = VALID_STATUS if reported_validity else INVALID_STATUS
    _require(summary.get("status") == expected_status, "Diagnostic status differs")
    audit_status = (
        "PASSED_PUBLICATION_GATE" if reported_validity else "AUDITED_INVALID_DIAGNOSTIC"
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_kind": AUDIT_ARTIFACT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": audit_status,
        "inputs": {
            "slurm_job_id": job_id,
            "copied_bundle_manifest": {
                "path": str(bundle_manifest),
                "sha256": bundle_manifest_sha256,
                "members_verified": len(bundle_members),
            },
            "diagnostic_json": {
                "path": str(diagnostic_json),
                "sha256": json_sha256,
            },
            "diagnostic_npz": {
                "path": str(diagnostic_npz),
                "sha256": npz_sha256,
            },
            "prior_diagnostic_npz": {
                "path": str(prior_diagnostic_npz),
                "sha256": prior_npz_sha256,
            },
        },
        "checks": {
            "all_sidecars_passed": True,
            "complete_copied_bundle_manifest_passed": True,
            "closed_root_relative_bundle_allowlist_passed": True,
            "slurm_log_content_passed": True,
            "matching_done_and_success_status_passed": True,
            "frozen_provenance_hashes_passed": True,
            "schema_and_scope_passed": True,
            "array_manifest_exact_key_shape_dtype_passed": True,
            "disallowed_json_paths": disallowed_json_paths,
            "disallowed_npz_keys": disallowed_array_keys,
            "canonical_array_validity_passed": True,
            "canonical_point_connectivity_coherence_passed": True,
            "job304002_persisted_replay_passed": True,
            "all_prediction_comparisons_recomputed": True,
            "all_same_input_replays_bitwise_exact": True,
            "decision_gates_recomputed": True,
            "outcome_recomputed": True,
            "producer_reported_validity": reported_validity,
        },
        "decision": {
            "derived_passed": all_derived_passed,
            "full_passed": all_full_passed,
            "outcome": outcome,
        },
        "canonical_array_checks": canonical_checks,
        "case_prediction_relative_l2_maxima": case_maxima,
        "producer_only_claims_not_independently_regenerated": [
            "raw-derived neutrality and the pre-CenterMesh single-cast construction history",
            "raw external primary/fixed point and query replay",
            "canonical construction repeat before persistence",
            "current primary/fixed path topology equality",
            "local-data stripping and supervision-array non-indexing",
            "same-input rerun and intervention-label fidelity",
            "injected internal geometry equality",
            "neutral decode center/reference/query runtime state",
            "the training-state checkpoint hash and fresh model inference",
        ],
        "interpretation_limit": (
            "The outcome label is recomputed from independent persisted decision "
            "gates plus a fully reconciled producer-reported validity reduction. "
            "The copied frozen code/input provenance makes the listed runtime claims "
            "auditable by implementation identity, but artifacts alone cannot "
            "independently regenerate them."
        ),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.with_name(f"{path.name}.sha256").exists():
        raise FileExistsError(f"Refusing to overwrite {path} or its sidecar")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = _sha256_file(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    with sidecar.open("xb") as stream:
        stream.write(f"{digest}  {path.name}\n".encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-json", type=Path, required=True)
    parser.add_argument("--diagnostic-npz", type=Path, required=True)
    parser.add_argument("--prior-diagnostic-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _validate_audit_output_path(
    output_json: Path,
    bundle_manifest: Path,
    job_id: str,
) -> Path:
    output = Path(os.path.abspath(output_json))
    bundle_root = bundle_manifest.resolve(strict=True).parent
    _require(
        not output.is_relative_to(bundle_root),
        "Adjudication output must be outside the immutable copied bundle",
    )
    _require(
        output.parent == bundle_root.parent
        and output.name == _expected_audit_output_basename(job_id),
        "Adjudication output path differs from the frozen sibling template",
    )
    sidecar = output.with_name(f"{output.name}.sha256")
    _require(
        not output.exists()
        and not output.is_symlink()
        and not sidecar.exists()
        and not sidecar.is_symlink(),
        "Adjudication output or sidecar already exists or is a symlink",
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = adjudicate(
        bundle_manifest=args.bundle_manifest,
        diagnostic_json=args.diagnostic_json,
        diagnostic_npz=args.diagnostic_npz,
        prior_diagnostic_npz=args.prior_diagnostic_npz,
    )
    output_json = _validate_audit_output_path(
        args.output_json,
        args.bundle_manifest,
        str(result["inputs"]["slurm_job_id"]),
    )
    _atomic_write_json(output_json, result)
    print(
        f"{result['status']} outcome={result['decision']['outcome']} "
        f"output={output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
