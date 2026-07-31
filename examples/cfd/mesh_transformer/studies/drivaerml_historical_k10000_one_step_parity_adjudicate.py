# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independently adjudicate the K=10k legacy/canonical one-step parity gate.

The producer persists raw flattened gradients and one-step parameter updates.
This reducer recomputes every deciding statistic from those vectors.  BF16 is
deciding; FP32 predictions, losses, gradients, and updates are diagnostic.

A valid pass retires a full same-cell legacy-versus-canonical retrain because
the paths are identical in exact arithmetic and operationally indistinguishable
under the preregistered gate.  A valid miss licenses only a fixed 16-step
microtrajectory.  Missing or malformed evidence is invalid or incomplete, not
scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_KIND = "drivaerml_historical_k10000_one_step_parity_adjudication"
PRODUCER_ARTIFACT_KIND = "drivaerml_historical_k10000_one_step_parity_producer"
PRODUCER_STATUS = "PASSED_HISTORICAL_K10000_ONE_STEP_PARITY_PRODUCER"

VALID_STATUS = "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION"
INVALID_STATUS = "INVALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION"
INCOMPLETE_STATUS = "INCOMPLETE_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION"

NEGLIGIBLE_OUTCOME = "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS"
FAIL_OUTCOME = "MATERIAL_PARITY_DIFFERENCE_FAIL"
INVALID_OUTCOME = "INVALID_ONE_STEP_PARITY_COMPARISON"
INCOMPLETE_OUTCOME = "INCOMPLETE_ONE_STEP_PARITY_COMPARISON"

CASE_SPECS = (
    (0, "run_118"),
    (12, "run_271"),
    (24, "run_429"),
    (35, "run_86"),
)
REGIMES = ("fresh_seed42", "checkpoint_epoch491")
PRECISIONS = ("bfloat16", "float32")
ARMS = ("legacy", "canonical")
DECIDING_PRECISION = "bfloat16"
RESOLUTION = 10_000
EXPECTED_PARAMETER_COUNT = 1_278_268

GRADIENT_COSINE_MIN = 0.999
UPDATE_COSINE_MIN = 0.9999
UPDATE_SYMMETRIC_RELATIVE_L2_MAX = 0.01
GRADIENT_NOISE_FRACTION_MAX = 0.1
ACTIVE_MODULE_ENERGY_FRACTION_MIN = 0.01
ACTIVE_MODULE_COSINE_MIN = 0.99

EXPECTED_PRODUCER_SHA256 = (
    "f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500"
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
EXPECTED_MODEL_CHECKPOINT_SHA256 = (
    "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
)
EXPECTED_TRAINING_STATE_SHA256 = (
    "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
)
EXPECTED_NORMALIZATION_STATE_SHA256 = (
    "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)

PARAMETER_LAYOUT_KEYS = (
    "parameter_slice_starts_int64",
    "parameter_slice_stops_int64",
    "parameter_slice_module_indices_int64",
)
VECTOR_FIELDS = (
    "prediction_pressure_float32",
    "prediction_wss_float32",
    "loss_float64",
    "gradient_float32",
    "parameter_update_float32",
    "learning_rate_float64",
    "selected_cell_ids_int64",
    "target_pressure_float32",
    "target_wss_float32",
    "target_measure_float32",
)
SHARED_ARRAY_FIELDS = (
    "selected_cell_ids_int64",
    "target_pressure_float32",
    "target_wss_float32",
    "target_measure_float32",
)

EXPECTED_CONTRACT = {
    "case_ordinals": [ordinal for ordinal, _ in CASE_SPECS],
    "case_ids": [case_id for _, case_id in CASE_SPECS],
    "regimes": list(REGIMES),
    "precisions": list(PRECISIONS),
    "arms": list(ARMS),
    "resolution": RESOLUTION,
    "fresh_seed": 42,
    "checkpoint_epoch": 491,
    "compile_enabled": False,
    "loss_type": "huber",
    "loss_delta": 1.0,
    "optimizer_class": "physicsnemo.optim.CombinedOptimizer(Muon,AdamW)",
    "stochastic_modules_present": False,
    "update_semantics": (
        "parameter_after_one_combined_muon_adamw_step_minus_parameter_before_step"
    ),
    "gradient_semantics": (
        "flattened_parameter_gradients_after_backward_before_optimizer_step"
    ),
    "no_source_measure_weights": True,
    "target_measure_preserved": True,
    "target_measure_used_by_loss": True,
}

EXPECTED_PROVENANCE = {
    "producer_sha256": EXPECTED_PRODUCER_SHA256,
    "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
    "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
    "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
    "resolved_config_sha256": EXPECTED_RESOLVED_CONFIG_SHA256,
    "model_checkpoint_sha256": EXPECTED_MODEL_CHECKPOINT_SHA256,
    "training_state_sha256": EXPECTED_TRAINING_STATE_SHA256,
    "normalization_state_sha256": EXPECTED_NORMALIZATION_STATE_SHA256,
}


class ArtifactUnavailable(RuntimeError):
    """A required artifact is absent."""


class ArtifactInvalid(RuntimeError):
    """A present artifact violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactInvalid(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} key set differs")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256_bytes(contiguous.view(np.uint8).tobytes())


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).view(np.uint8).tobytes()
        == np.ascontiguousarray(right).view(np.uint8).tobytes()
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactInvalid(f"{label} is not strict JSON: {error}") from error
    return _mapping(value, label)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def _verified_payload(path: Path, label: str) -> tuple[bytes, str]:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    if not path.exists():
        raise ArtifactUnavailable(f"{label} is missing: {path}")
    _require(path.is_file(), f"{label} must be a regular file")
    sidecar = _sidecar_path(path)
    _require(not sidecar.is_symlink(), f"{label} sidecar must not be a symlink")
    if not sidecar.exists():
        raise ArtifactUnavailable(f"{label} sidecar is missing: {sidecar}")
    _require(sidecar.is_file(), f"{label} sidecar must be a regular file")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    try:
        sidecar_payload = sidecar.read_bytes()
        expected = f"{digest}  {path.name}\n".encode("ascii")
    except UnicodeEncodeError as error:
        raise ArtifactInvalid(f"{label} filename is not ASCII") from error
    _require(sidecar_payload == expected, f"{label} sidecar is noncanonical or stale")
    return payload, digest


def _prefix(regime: str, precision: str, ordinal: int, case_id: str, arm: str) -> str:
    return f"{regime}__{precision}__case_{ordinal:02d}_{case_id}__{arm}__"


def _expected_array_keys() -> set[str]:
    keys = set(PARAMETER_LAYOUT_KEYS)
    for regime in REGIMES:
        for precision in PRECISIONS:
            for ordinal, case_id in CASE_SPECS:
                for arm in ARMS:
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    keys.update(f"{prefix}{field}" for field in VECTOR_FIELDS)
    return keys


def _load_npz(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            _require(
                len(archive.files) == len(set(archive.files)),
                "producer NPZ contains duplicate member names",
            )
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactInvalid(
            f"producer NPZ cannot be loaded safely: {error}"
        ) from error
    _require(set(arrays) == _expected_array_keys(), "producer NPZ key set differs")
    return arrays


def _validate_array_manifest(
    manifest_value: Any,
    arrays: Mapping[str, np.ndarray],
) -> None:
    manifest = _mapping(manifest_value, "array_manifest")
    _exact_keys(manifest, set(arrays), "array_manifest")
    for name, array in arrays.items():
        entry = _mapping(manifest[name], f"array_manifest.{name}")
        _exact_keys(entry, {"dtype", "shape", "sha256"}, f"array_manifest.{name}")
        _require(entry["dtype"] == array.dtype.str, f"{name} manifest dtype differs")
        _require(entry["shape"] == list(array.shape), f"{name} manifest shape differs")
        _require(
            entry["sha256"] == _array_sha256(array),
            f"{name} manifest SHA-256 differs",
        )


def _expected_shape_dtype(
    field: str,
    *,
    parameter_count: int,
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "prediction_pressure_float32": ((RESOLUTION,), np.dtype("<f4")),
        "prediction_wss_float32": ((RESOLUTION, 3), np.dtype("<f4")),
        "loss_float64": ((1,), np.dtype("<f8")),
        "gradient_float32": ((parameter_count,), np.dtype("<f4")),
        "parameter_update_float32": ((parameter_count,), np.dtype("<f4")),
        "learning_rate_float64": ((1,), np.dtype("<f8")),
        "selected_cell_ids_int64": ((RESOLUTION,), np.dtype("<i8")),
        "target_pressure_float32": ((RESOLUTION,), np.dtype("<f4")),
        "target_wss_float32": ((RESOLUTION, 3), np.dtype("<f4")),
        "target_measure_float32": ((RESOLUTION,), np.dtype("<f4")),
    }
    return schemas[field]


def _validate_parameter_layout(
    layout_value: Any,
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, tuple[str, ...], tuple[np.ndarray, ...], str]:
    layout = _mapping(layout_value, "parameter_layout")
    _exact_keys(
        layout,
        {
            "parameter_count",
            "parameter_names",
            "module_names",
            "ordered_parameter_names_sha256",
        },
        "parameter_layout",
    )
    parameter_count = layout["parameter_count"]
    _require(
        _is_int(parameter_count) and parameter_count == EXPECTED_PARAMETER_COUNT,
        "parameter_layout.parameter_count differs",
    )
    parameter_names = layout["parameter_names"]
    module_names = layout["module_names"]
    _require(
        isinstance(parameter_names, list)
        and parameter_names
        and all(isinstance(name, str) and name for name in parameter_names),
        "parameter_names must be a nonempty string list",
    )
    _require(
        isinstance(module_names, list)
        and module_names
        and all(isinstance(name, str) and name for name in module_names),
        "module_names must be a nonempty string list",
    )
    _require(
        len(parameter_names) == len(set(parameter_names)),
        "parameter_names are not unique",
    )
    _require(len(module_names) == len(set(module_names)), "module_names are not unique")
    names_payload = json.dumps(
        parameter_names, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    _require(
        layout["ordered_parameter_names_sha256"] == _sha256_bytes(names_payload),
        "ordered_parameter_names_sha256 differs",
    )
    parameter_order_sha256 = str(layout["ordered_parameter_names_sha256"])

    expected_module_names: list[str] = []
    expected_module_lookup: dict[str, int] = {}
    expected_module_indices: list[int] = []
    for parameter_name in parameter_names:
        module_name = parameter_name.rpartition(".")[0] or "<root>"
        if module_name not in expected_module_lookup:
            expected_module_lookup[module_name] = len(expected_module_names)
            expected_module_names.append(module_name)
        expected_module_indices.append(expected_module_lookup[module_name])
    _require(
        module_names == expected_module_names,
        "module_names do not match parameter-name parent modules",
    )

    starts = arrays["parameter_slice_starts_int64"]
    stops = arrays["parameter_slice_stops_int64"]
    module_indices = arrays["parameter_slice_module_indices_int64"]
    parameter_tensors = len(parameter_names)
    for name, value in (
        ("parameter_slice_starts_int64", starts),
        ("parameter_slice_stops_int64", stops),
        ("parameter_slice_module_indices_int64", module_indices),
    ):
        _require(value.dtype == np.dtype("<i8"), f"{name} dtype differs")
        _require(value.shape == (parameter_tensors,), f"{name} shape differs")
    _require(int(starts[0]) == 0, "parameter slices do not start at zero")
    _require(
        int(stops[-1]) == parameter_count,
        "parameter slices do not end at parameter_count",
    )
    _require(np.all(starts < stops), "parameter slices contain an empty range")
    _require(
        np.array_equal(starts[1:], stops[:-1]),
        "parameter slices are not a contiguous full partition",
    )
    _require(
        np.all((module_indices >= 0) & (module_indices < len(module_names))),
        "parameter module index is out of range",
    )
    _require(
        np.array_equal(
            module_indices,
            np.asarray(expected_module_indices, dtype="<i8"),
        ),
        "parameter module indices do not match parameter names",
    )

    module_positions: list[list[int]] = [[] for _ in module_names]
    for tensor_index, module_index in enumerate(module_indices.tolist()):
        module_positions[module_index].append(tensor_index)
    module_index_arrays = tuple(
        np.concatenate(
            [
                np.arange(int(starts[index]), int(stops[index]), dtype=np.int64)
                for index in positions
            ]
        )
        for positions in module_positions
    )
    return (
        parameter_count,
        tuple(module_names),
        module_index_arrays,
        parameter_order_sha256,
    )


def _validate_contract(contract_value: Any) -> None:
    contract = _mapping(contract_value, "contract")
    _exact_keys(contract, set(EXPECTED_CONTRACT), "contract")
    for key, expected in EXPECTED_CONTRACT.items():
        _require(type(contract[key]) is type(expected), f"contract.{key} type differs")
        _require(contract[key] == expected, f"contract.{key} differs")


def _validate_arm_control(value: Any, label: str) -> Mapping[str, Any]:
    control = _mapping(value, label)
    expected_keys = {
        "raw_source_geometry_sha256",
        "global_inputs_sha256",
        "initial_parameter_state_sha256",
        "initial_optimizer_state_sha256",
        "rng_state_sha256",
        "parameter_order_sha256",
        "batch_order_sha256",
        "source_measure_weights_present",
        "target_measure_present",
    }
    _exact_keys(control, expected_keys, label)
    for key in expected_keys - {
        "source_measure_weights_present",
        "target_measure_present",
    }:
        _require(_is_sha256(control[key]), f"{label}.{key} is not SHA-256")
    _require(
        control["source_measure_weights_present"] is False,
        f"{label} unexpectedly contains model-visible source measure weights",
    )
    _require(
        control["target_measure_present"] is True,
        f"{label} does not preserve the target measure",
    )
    return control


def _validate_records(
    records_value: Any,
    *,
    parameter_order_sha256: str,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    _require(isinstance(records_value, list), "records is not a list")
    expected_id_by_ordinal = dict(CASE_SPECS)
    expected_keys = {
        (regime, precision, ordinal)
        for regime in REGIMES
        for precision in PRECISIONS
        for ordinal, _ in CASE_SPECS
    }
    records: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, raw_record in enumerate(records_value):
        record = _mapping(raw_record, f"records[{index}]")
        _exact_keys(
            record,
            {"regime", "precision", "case_ordinal", "case_id", "arm_controls"},
            f"records[{index}]",
        )
        regime = record["regime"]
        precision = record["precision"]
        ordinal = record["case_ordinal"]
        _require(regime in REGIMES, f"records[{index}].regime differs")
        _require(precision in PRECISIONS, f"records[{index}].precision differs")
        _require(
            _is_int(ordinal) and ordinal in expected_id_by_ordinal,
            f"records[{index}].case_ordinal differs",
        )
        _require(
            record["case_id"] == expected_id_by_ordinal[ordinal],
            f"records[{index}].case_id differs",
        )
        key = (regime, precision, ordinal)
        _require(key not in records, f"duplicate record {key}")
        arm_controls = _mapping(
            record["arm_controls"], f"records[{index}].arm_controls"
        )
        _exact_keys(arm_controls, set(ARMS), f"records[{index}].arm_controls")
        legacy = _validate_arm_control(
            arm_controls["legacy"], f"records[{index}].arm_controls.legacy"
        )
        canonical = _validate_arm_control(
            arm_controls["canonical"], f"records[{index}].arm_controls.canonical"
        )
        _require(
            legacy == canonical,
            f"records[{index}] legacy/canonical arm controls differ",
        )
        _require(
            legacy["parameter_order_sha256"] == parameter_order_sha256,
            f"records[{index}] parameter order does not match parameter layout",
        )
        records[key] = legacy
    _require(set(records) == expected_keys, "records coverage differs")

    for regime in REGIMES:
        reference: tuple[str, str, str, str] | None = None
        for precision in PRECISIONS:
            for ordinal, _ in CASE_SPECS:
                control = records[regime, precision, ordinal]
                current = (
                    str(control["initial_parameter_state_sha256"]),
                    str(control["initial_optimizer_state_sha256"]),
                    str(control["rng_state_sha256"]),
                    str(control["parameter_order_sha256"]),
                )
                if reference is None:
                    reference = current
                _require(
                    current == reference,
                    f"{regime} initial state/RNG/order changes across cases or "
                    "precision",
                )
    for ordinal, case_id in CASE_SPECS:
        reference_case_controls: tuple[str, str, str] | None = None
        for regime in REGIMES:
            for precision in PRECISIONS:
                control = records[regime, precision, ordinal]
                current = (
                    str(control["raw_source_geometry_sha256"]),
                    str(control["global_inputs_sha256"]),
                    str(control["batch_order_sha256"]),
                )
                if reference_case_controls is None:
                    reference_case_controls = current
                _require(
                    current == reference_case_controls,
                    f"{case_id} source/global/batch controls change across "
                    "regimes or precision",
                )
    _require(
        len({str(control["parameter_order_sha256"]) for control in records.values()})
        == 1,
        "parameter order changes across regimes, precision, or cases",
    )
    _require(
        len({str(control["rng_state_sha256"]) for control in records.values()}) == 1,
        "RNG state changes across regimes, precision, or cases",
    )
    regime_states = {
        regime: (
            str(
                records[regime, PRECISIONS[0], CASE_SPECS[0][0]][
                    "initial_parameter_state_sha256"
                ]
            ),
            str(
                records[regime, PRECISIONS[0], CASE_SPECS[0][0]][
                    "initial_optimizer_state_sha256"
                ]
            ),
        )
        for regime in REGIMES
    }
    _require(
        regime_states[REGIMES[0]][0] != regime_states[REGIMES[1]][0],
        "fresh and checkpoint parameter states are identical",
    )
    _require(
        regime_states[REGIMES[0]][1] != regime_states[REGIMES[1]][1],
        "fresh and checkpoint optimizer states are identical",
    )
    return records


def _validate_provenance(
    provenance_value: Any,
    *,
    npz_sha256: str | None,
) -> Mapping[str, Any]:
    provenance = _mapping(provenance_value, "provenance")
    expected_keys = {
        "producer_sha256",
        "source_tree_sha256",
        "dataset_manifest_sha256",
        "dataset_config_sha256",
        "resolved_config_sha256",
        "model_checkpoint_sha256",
        "training_state_sha256",
        "normalization_state_sha256",
        "npz_sha256",
    }
    _exact_keys(provenance, expected_keys, "provenance")
    for key in expected_keys:
        _require(_is_sha256(provenance[key]), f"provenance.{key} is not SHA-256")
    for key, expected in EXPECTED_PROVENANCE.items():
        _require(provenance[key] == expected, f"provenance.{key} differs")
    if npz_sha256 is not None:
        _require(
            provenance["npz_sha256"] == npz_sha256,
            "provenance NPZ SHA differs",
        )
    return provenance


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    parameter_count: int,
) -> None:
    for regime in REGIMES:
        for precision in PRECISIONS:
            for ordinal, case_id in CASE_SPECS:
                for arm in ARMS:
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    for field in VECTOR_FIELDS:
                        name = f"{prefix}{field}"
                        array = arrays[name]
                        expected_shape, expected_dtype = _expected_shape_dtype(
                            field, parameter_count=parameter_count
                        )
                        _require(array.shape == expected_shape, f"{name} shape differs")
                        _require(array.dtype == expected_dtype, f"{name} dtype differs")
                        if array.dtype.kind == "f":
                            _require(np.isfinite(array).all(), f"{name} is non-finite")
                legacy_prefix = _prefix(regime, precision, ordinal, case_id, "legacy")
                canonical_prefix = _prefix(
                    regime, precision, ordinal, case_id, "canonical"
                )
                for field in SHARED_ARRAY_FIELDS:
                    _require(
                        _array_exact(
                            arrays[f"{legacy_prefix}{field}"],
                            arrays[f"{canonical_prefix}{field}"],
                        ),
                        f"{regime}/{precision}/{case_id} shared {field} differs by arm",
                    )
                for arm in ARMS:
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    _require(
                        np.unique(arrays[f"{prefix}selected_cell_ids_int64"]).size
                        == RESOLUTION,
                        f"{prefix} selected cell IDs are not unique",
                    )
                    _require(
                        np.all(arrays[f"{prefix}selected_cell_ids_int64"] >= 0),
                        f"{prefix} selected cell IDs are negative",
                    )
                    _require(
                        np.all(arrays[f"{prefix}target_measure_float32"] > 0.0),
                        f"{prefix} target measure is not positive",
                    )
                    _require(
                        float(arrays[f"{prefix}loss_float64"][0]) >= 0.0,
                        f"{prefix} loss is negative",
                    )
                    _require(
                        float(arrays[f"{prefix}learning_rate_float64"][0]) > 0.0,
                        f"{prefix} learning rate is not positive",
                    )

    # Data/target controls must be identical across regimes and precisions for a
    # given case.  This ensures the between-case gradient scale is calibrated
    # from case identity, not a changing sample or target reduction.
    for ordinal, case_id in CASE_SPECS:
        reference_prefix = _prefix(REGIMES[0], PRECISIONS[0], ordinal, case_id, ARMS[0])
        for regime in REGIMES:
            for precision in PRECISIONS:
                for arm in ARMS:
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    for field in SHARED_ARRAY_FIELDS:
                        _require(
                            _array_exact(
                                arrays[f"{reference_prefix}{field}"],
                                arrays[f"{prefix}{field}"],
                            ),
                            f"{case_id} shared {field} changes across execution cells",
                        )

    for regime in REGIMES:
        reference_lr: np.ndarray | None = None
        for precision in PRECISIONS:
            for ordinal, case_id in CASE_SPECS:
                for arm in ARMS:
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    learning_rate = arrays[f"{prefix}learning_rate_float64"]
                    if reference_lr is None:
                        reference_lr = learning_rate
                    _require(
                        _array_exact(learning_rate, reference_lr),
                        f"{regime} learning rate changes across cases/precision/arms",
                    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ArtifactInvalid("a deciding vector has zero norm")
    value = float(np.dot(left64, right64) / (left_norm * right_norm))
    return float(np.clip(value, -1.0, 1.0))


def _symmetric_relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left64) + np.linalg.norm(right64))
    if denominator == 0.0:
        raise ArtifactInvalid("a relative-distance vector pair has zero total norm")
    return float(2.0 * np.linalg.norm(left64 - right64) / denominator)


def _median_pairwise_legacy_gradient_distance(
    arrays: Mapping[str, np.ndarray],
    regime: str,
    precision: str,
) -> tuple[float, list[float]]:
    gradients = []
    for ordinal, case_id in CASE_SPECS:
        prefix = _prefix(regime, precision, ordinal, case_id, "legacy")
        gradients.append(arrays[f"{prefix}gradient_float32"])
    distances = [
        _symmetric_relative_l2(gradients[left], gradients[right])
        for left in range(len(gradients))
        for right in range(left + 1, len(gradients))
    ]
    return float(np.median(np.asarray(distances, dtype=np.float64))), distances


def _module_rows(
    legacy_gradient: np.ndarray,
    canonical_gradient: np.ndarray,
    module_names: Sequence[str],
    module_indices: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    legacy64 = np.asarray(legacy_gradient, dtype=np.float64)
    canonical64 = np.asarray(canonical_gradient, dtype=np.float64)
    total_energy = 0.5 * (
        float(np.dot(legacy64, legacy64)) + float(np.dot(canonical64, canonical64))
    )
    if total_energy == 0.0:
        raise ArtifactInvalid("mean two-arm gradient energy is zero")
    rows = []
    for name, indices in zip(module_names, module_indices, strict=True):
        legacy_module = legacy64[indices]
        canonical_module = canonical64[indices]
        energy = 0.5 * (
            float(np.dot(legacy_module, legacy_module))
            + float(np.dot(canonical_module, canonical_module))
        )
        fraction = energy / total_energy
        active = fraction >= ACTIVE_MODULE_ENERGY_FRACTION_MIN
        if energy == 0.0:
            cosine: float | None = None
        elif (
            np.linalg.norm(legacy_module) == 0.0
            or np.linalg.norm(canonical_module) == 0.0
        ):
            cosine = 0.0
        else:
            cosine = _cosine(legacy_module, canonical_module)
        passed = (not active) or (
            cosine is not None and cosine >= ACTIVE_MODULE_COSINE_MIN
        )
        rows.append(
            {
                "module": name,
                "mean_two_arm_gradient_energy_fraction": float(fraction),
                "active": active,
                "gradient_cosine": cosine,
                "passed": passed,
            }
        )
    return rows


def _descriptive_prediction_loss(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    prediction_pressure = arrays[f"{prefix}prediction_pressure_float32"].astype(
        np.float64
    )
    prediction_wss = arrays[f"{prefix}prediction_wss_float32"].astype(np.float64)
    target_pressure = arrays[f"{prefix}target_pressure_float32"].astype(np.float64)
    target_wss = arrays[f"{prefix}target_wss_float32"].astype(np.float64)
    measure = arrays[f"{prefix}target_measure_float32"].astype(np.float64)
    pressure_error = prediction_pressure - target_pressure
    wss_error = prediction_wss - target_wss
    pressure_relative_l2 = float(
        np.linalg.norm(pressure_error) / max(np.linalg.norm(target_pressure), 1.0e-30)
    )
    wss_relative_l2 = float(
        np.linalg.norm(wss_error) / max(np.linalg.norm(target_wss), 1.0e-30)
    )
    normalized_measure = measure / measure.sum()

    def huber(error: np.ndarray) -> np.ndarray:
        absolute = np.abs(error)
        return np.where(absolute <= 1.0, 0.5 * error**2, absolute - 0.5)

    pressure_huber = float(np.sum(normalized_measure * huber(pressure_error)))
    wss_huber = float(
        sum(
            np.sum(normalized_measure * huber(wss_error[:, component]))
            for component in range(3)
        )
    )
    reconstructed_loss = (pressure_huber + wss_huber) / 4.0
    reported_loss = float(arrays[f"{prefix}loss_float64"][0])
    return {
        "reported_loss": reported_loss,
        "reconstructed_float64_huber_loss": float(reconstructed_loss),
        "reported_minus_reconstructed_loss": float(reported_loss - reconstructed_loss),
        "pressure_relative_l2": pressure_relative_l2,
        "wss_frobenius_relative_l2": wss_relative_l2,
    }


def _compute_results(
    arrays: Mapping[str, np.ndarray],
    module_names: Sequence[str],
    module_indices: Sequence[np.ndarray],
) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    all_deciding_passed = True
    for precision in PRECISIONS:
        precision_rows: dict[str, Any] = {}
        for regime in REGIMES:
            reference_distance, pairwise_distances = (
                _median_pairwise_legacy_gradient_distance(arrays, regime, precision)
            )
            gradient_limit = GRADIENT_NOISE_FRACTION_MAX * reference_distance
            case_rows: dict[str, Any] = {}
            for ordinal, case_id in CASE_SPECS:
                legacy_prefix = _prefix(regime, precision, ordinal, case_id, "legacy")
                canonical_prefix = _prefix(
                    regime, precision, ordinal, case_id, "canonical"
                )
                legacy_gradient = arrays[f"{legacy_prefix}gradient_float32"]
                canonical_gradient = arrays[f"{canonical_prefix}gradient_float32"]
                legacy_update = arrays[f"{legacy_prefix}parameter_update_float32"]
                canonical_update = arrays[f"{canonical_prefix}parameter_update_float32"]
                gradient_cosine = _cosine(legacy_gradient, canonical_gradient)
                update_cosine = _cosine(legacy_update, canonical_update)
                gradient_relative_l2 = _symmetric_relative_l2(
                    legacy_gradient, canonical_gradient
                )
                update_relative_l2 = _symmetric_relative_l2(
                    legacy_update, canonical_update
                )
                modules = _module_rows(
                    legacy_gradient,
                    canonical_gradient,
                    module_names,
                    module_indices,
                )
                gates = {
                    "global_gradient_cosine": {
                        "value": gradient_cosine,
                        "inclusive_minimum": GRADIENT_COSINE_MIN,
                        "passed": gradient_cosine >= GRADIENT_COSINE_MIN,
                    },
                    "global_update_cosine": {
                        "value": update_cosine,
                        "inclusive_minimum": UPDATE_COSINE_MIN,
                        "passed": update_cosine >= UPDATE_COSINE_MIN,
                    },
                    "update_symmetric_relative_l2": {
                        "value": update_relative_l2,
                        "inclusive_maximum": UPDATE_SYMMETRIC_RELATIVE_L2_MAX,
                        "passed": (
                            update_relative_l2 <= UPDATE_SYMMETRIC_RELATIVE_L2_MAX
                        ),
                    },
                    "gradient_relative_to_between_case_distance": {
                        "path_symmetric_relative_l2": gradient_relative_l2,
                        "median_pairwise_between_case_legacy_gradient_distance": (
                            reference_distance
                        ),
                        "inclusive_maximum": gradient_limit,
                        "passed": gradient_relative_l2 <= gradient_limit,
                    },
                    "active_module_cosines": {
                        "active_energy_fraction_minimum": (
                            ACTIVE_MODULE_ENERGY_FRACTION_MIN
                        ),
                        "inclusive_cosine_minimum": ACTIVE_MODULE_COSINE_MIN,
                        "passed": all(row["passed"] for row in modules),
                    },
                }
                case_passed = all(gate["passed"] for gate in gates.values())
                if precision == DECIDING_PRECISION:
                    all_deciding_passed = all_deciding_passed and case_passed
                case_rows[case_id] = {
                    "case_ordinal": ordinal,
                    "passed": case_passed,
                    "deciding": precision == DECIDING_PRECISION,
                    "gates": gates,
                    "modules": modules,
                    "descriptive": {
                        "legacy": _descriptive_prediction_loss(arrays, legacy_prefix),
                        "canonical": _descriptive_prediction_loss(
                            arrays, canonical_prefix
                        ),
                    },
                }
            precision_rows[regime] = {
                "median_pairwise_between_case_legacy_gradient_distance": (
                    reference_distance
                ),
                "pairwise_between_case_legacy_gradient_distances": [
                    float(value) for value in pairwise_distances
                ],
                "gradient_path_distance_inclusive_maximum": gradient_limit,
                "cases": case_rows,
            }
        results[precision] = {
            "deciding": precision == DECIDING_PRECISION,
            "regimes": precision_rows,
        }
    return results, all_deciding_passed


def _validate_summary_envelope(summary: Mapping[str, Any]) -> None:
    _exact_keys(
        summary,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "contract",
            "parameter_layout",
            "records",
            "array_manifest",
            "provenance",
        },
        "producer JSON",
    )
    _require(
        _is_int(summary["schema_version"]) and summary["schema_version"] == 1,
        "producer schema version differs",
    )
    _require(
        summary["artifact_kind"] == PRODUCER_ARTIFACT_KIND,
        "producer artifact kind differs",
    )
    _require(summary["status"] == PRODUCER_STATUS, "producer status differs")
    _validate_contract(summary["contract"])


def _validate_and_compute(
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    json_sha256: str,
    npz_sha256: str,
) -> dict[str, Any]:
    _validate_summary_envelope(summary)
    (
        parameter_count,
        module_names,
        module_indices,
        parameter_order_sha256,
    ) = _validate_parameter_layout(summary["parameter_layout"], arrays)
    _validate_records(
        summary["records"],
        parameter_order_sha256=parameter_order_sha256,
    )
    _validate_arrays(arrays, parameter_count=parameter_count)
    _validate_array_manifest(summary["array_manifest"], arrays)
    provenance = _validate_provenance(summary["provenance"], npz_sha256=npz_sha256)
    results, passed = _compute_results(arrays, module_names, module_indices)
    outcome = NEGLIGIBLE_OUTCOME if passed else FAIL_OUTCOME
    next_step = (
        "Retire the full same-cell retrain and run the separately preregistered "
        "canonical fixed-Q source-support H-QC experiment."
        if passed
        else (
            "Run one fixed 16-step paired microtrajectory. Do not launch a full "
            "same-cell retrain from this one-step miss."
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": VALID_STATUS,
        "decision_outcome": outcome,
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "validity": {
            "producer_json_sha256": json_sha256,
            "producer_npz_sha256": npz_sha256,
            "producer_source_sha256": provenance["producer_sha256"],
            "raw_array_manifest_verified": True,
            "shared_controls_verified": True,
            "parameter_partition_verified": True,
            "all_values_finite": True,
        },
        "decision_contract": {
            "deciding_precision": DECIDING_PRECISION,
            "all_case_regime_gates_required": True,
            "gradient_cosine_inclusive_minimum": GRADIENT_COSINE_MIN,
            "update_cosine_inclusive_minimum": UPDATE_COSINE_MIN,
            "update_symmetric_relative_l2_inclusive_maximum": (
                UPDATE_SYMMETRIC_RELATIVE_L2_MAX
            ),
            "gradient_path_fraction_of_between_case_median_inclusive_maximum": (
                GRADIENT_NOISE_FRACTION_MAX
            ),
            "active_module_energy_fraction_inclusive_minimum": (
                ACTIVE_MODULE_ENERGY_FRACTION_MIN
            ),
            "active_module_cosine_inclusive_minimum": ACTIVE_MODULE_COSINE_MIN,
            "fp32_role": "diagnostic_only",
        },
        "results": results,
        "limited_claim": (
            "Operational one-step parity only for four fixed K=10000 cases, two "
            "states, the exact CombinedOptimizer(Muon,AdamW) update, and the "
            "source-unweighted/target-measure-preserving current recipe path. "
            "This is not training-trajectory, population, H-QC, architecture, "
            "or independent source/target evidence."
        ),
        "next_step": next_step,
    }


def adjudicate(*, producer_json: Path, producer_npz: Path) -> dict[str, Any]:
    unavailable_errors: list[str] = []
    invalid_errors: list[str] = []
    json_payload: bytes | None = None
    json_sha256: str | None = None
    npz_payload: bytes | None = None
    npz_sha256: str | None = None
    summary: Mapping[str, Any] | None = None
    arrays: dict[str, np.ndarray] | None = None

    for path, label, payload_name, digest_name in (
        (producer_json, "producer JSON", "json_payload", "json_sha256"),
        (producer_npz, "producer NPZ", "npz_payload", "npz_sha256"),
    ):
        try:
            payload, digest = _verified_payload(path, label)
        except ArtifactUnavailable as error:
            unavailable_errors.append(str(error))
        except ArtifactInvalid as error:
            invalid_errors.append(str(error))
        else:
            if payload_name == "json_payload":
                json_payload, json_sha256 = payload, digest
            else:
                npz_payload, npz_sha256 = payload, digest

    if json_payload is not None:
        try:
            summary = _load_json_bytes(json_payload, "producer JSON")
            _validate_summary_envelope(summary)
            _validate_provenance(
                summary["provenance"],
                npz_sha256=npz_sha256,
            )
        except ArtifactInvalid as error:
            invalid_errors.append(str(error))
    if npz_payload is not None:
        try:
            arrays = _load_npz(npz_payload)
        except ArtifactInvalid as error:
            invalid_errors.append(str(error))

    if invalid_errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INVALID_STATUS,
            "decision_outcome": INVALID_OUTCOME,
            "error": "; ".join(invalid_errors),
            "next_step": "Repair and rerun the instrument; no parity evidence exists.",
        }
    if unavailable_errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INCOMPLETE_STATUS,
            "decision_outcome": INCOMPLETE_OUTCOME,
            "error": "; ".join(unavailable_errors),
            "next_step": "Complete or rerun the instrument; no parity evidence exists.",
        }
    if summary is None or arrays is None or json_sha256 is None or npz_sha256 is None:
        raise RuntimeError("artifact classification left an unreachable empty value")
    try:
        return _validate_and_compute(
            summary,
            arrays,
            json_sha256=json_sha256,
            npz_sha256=npz_sha256,
        )
    except ArtifactInvalid as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INVALID_STATUS,
            "decision_outcome": INVALID_OUTCOME,
            "error": str(error),
            "next_step": "Repair and rerun the instrument; no parity evidence exists.",
        }


def _exclusive_publish(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite adjudication output: {path}")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"refusing to overwrite adjudication sidecar: {sidecar}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    sidecar_descriptor, temporary_sidecar_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_sidecar = Path(temporary_sidecar_name)
    digest = _sha256_bytes(payload)
    published_path = False
    published_sidecar = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(sidecar_descriptor, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published_path = True
        os.link(temporary_sidecar, sidecar)
        published_sidecar = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if published_sidecar:
            sidecar.unlink()
        if published_path:
            path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-json", type=Path, required=True)
    parser.add_argument("--producer-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = adjudicate(
        producer_json=args.producer_json,
        producer_npz=args.producer_npz,
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    digest = _exclusive_publish(args.output_json, payload)
    print(
        f"{result['status']} outcome={result['decision_outcome']} json_sha256={digest}",
        flush=True,
    )
    if result["decision_outcome"] == INVALID_OUTCOME:
        return 2
    if result["decision_outcome"] == INCOMPLETE_OUTCOME:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
