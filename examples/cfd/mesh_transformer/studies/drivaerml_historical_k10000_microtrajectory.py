# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce raw stateful evidence for the K=10k microtrajectory experiment.

The implementation currently contains the hardened complete-state codec,
persistent transition/replay/probe kernel, and streaming raw-state
serialization primitives.  It is not launchable until the remaining producer,
reducer, wrapper, and superseding implementation freeze are complete.
"""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import importlib.machinery
import io
import json
import math
import os
import random
import re
import stat
import struct
import sys
import tempfile
import types
import zipfile
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.modules.module as torch_module_runtime
import torch.optim.optimizer as torch_optimizer
from torch.utils._python_dispatch import TorchDispatchMode

STATE_SCHEMA_VERSION = 1
TRAJECTORY_RECORD_SCHEMA_VERSION = 1
FROZEN_V3_PRODUCER_FILENAME = "drivaerml_historical_k10000_one_step_parity.py"
EXPECTED_FROZEN_V3_PRODUCER_SHA256 = (
    "f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500"
)
PANEL_RESOLUTION = 10_000
FRESH_SEED = 42
CHECKPOINT_EPOCH = 491
EXPECTED_PARAMETER_COUNT = 1_278_268
FIXED_CASE_SPECS = (
    (0, 0, "run_118"),
    (1, 12, "run_271"),
    (2, 24, "run_429"),
    (3, 35, "run_86"),
)
PANEL_REPETITIONS = 4
STEP_CASE_INDICES = tuple(range(len(FIXED_CASE_SPECS))) * PANEL_REPETITIONS
TRAJECTORY_STEP_SCHEDULE = tuple(
    FIXED_CASE_SPECS[case_index][2] for case_index in STEP_CASE_INDICES
)
TRAJECTORY_STEP_COUNT = len(TRAJECTORY_STEP_SCHEDULE)
STATE_REGIMES = ("fresh_seed42", "checkpoint_epoch491")
GEOMETRY_PATHS = ("legacy", "canonical")
TRAJECTORY_CHECKPOINT_STEPS = (0, 1, 2, 4, 8, 16)
TRAJECTORY_REPLAY_STEPS = (0, 1, 2, 4, 8)
CROSSOVER_STEPS = (0, 4, 8, 16)
CROSSOVER_HISTORIES = ("legacy_updated_state", "canonical_updated_state")
CROSSOVER_EVALUATION_GEOMETRIES = ("legacy_geometry", "canonical_geometry")
HISTORY_TO_PATH_INDEX = (0, 1)
EVALUATION_GEOMETRY_TO_PATH_INDEX = (0, 1)
NPZ_IDENTITY_ARRAY_FIELDS = (
    "attempt_id_utf8",
    "launch_manifest_sha256_ascii",
)
PARAMETER_LAYOUT_ARRAY_FIELDS = (
    "parameter_slice_starts_int64",
    "parameter_slice_stops_int64",
    "parameter_slice_module_indices_int64",
)
CASE_CONTROL_ARRAY_FIELDS = (
    "selected_cell_ids_int64",
    "target_pressure_float32",
    "target_wss_float32",
    "target_measure_float32",
)
MAIN_ARRAY_FIELDS = (
    "prediction_pressure_float32",
    "prediction_wss_float32",
    "loss_float64",
    "gradient_float32",
    "parameter_update_float32",
    "learning_rates_pre_float64",
    "learning_rates_post_float64",
)
CHECKPOINT_ARRAY_FIELDS = ("parameter_vector_float32",)
REPLAY_ARRAY_FIELDS = (
    "prediction_pressure_float32",
    "prediction_wss_float32",
    "loss_float64",
    "gradient_float32",
    "parameter_update_float32",
)
CROSSOVER_ARRAY_FIELDS = (
    "prediction_pressure_float32",
    "prediction_wss_float32",
    "loss_float64",
    "gradient_float32",
    "proposed_parameter_update_float32",
)
MAIN_RECORD_COUNT = len(STATE_REGIMES) * len(GEOMETRY_PATHS) * TRAJECTORY_STEP_COUNT
CHECKPOINT_RECORD_COUNT = (
    len(STATE_REGIMES) * len(GEOMETRY_PATHS) * len(TRAJECTORY_CHECKPOINT_STEPS)
)
REPLAY_RECORD_COUNT = (
    len(STATE_REGIMES) * len(GEOMETRY_PATHS) * len(TRAJECTORY_REPLAY_STEPS)
)
CROSSOVER_RECORD_COUNT = (
    len(STATE_REGIMES)
    * len(CROSSOVER_STEPS)
    * len(FIXED_CASE_SPECS)
    * len(CROSSOVER_HISTORIES)
    * len(CROSSOVER_EVALUATION_GEOMETRIES)
)
EXECUTED_TRANSITION_COUNT = (
    MAIN_RECORD_COUNT + REPLAY_RECORD_COUNT + CROSSOVER_RECORD_COUNT
)
PACKAGE_RECORD_COUNT = len(STATE_REGIMES) * len(GEOMETRY_PATHS)
CASE_RECORD_COUNT = len(FIXED_CASE_SPECS)
MATCHED_MAIN_PAIR_COUNT = len(STATE_REGIMES) * TRAJECTORY_STEP_COUNT
T0_IDENTITY_COMPARISON_COUNT = (
    len(STATE_REGIMES) * len(FIXED_CASE_SPECS) * len(CROSSOVER_EVALUATION_GEOMETRIES)
)
STATE_TREE_IDENTITY_COUNT = (
    CHECKPOINT_RECORD_COUNT
    + 2 * MAIN_RECORD_COUNT
    + 2 * REPLAY_RECORD_COUNT
    + 2 * CROSSOVER_RECORD_COUNT
)
SCIENTIFIC_FIXED_ARRAY_COUNT = (
    len(PARAMETER_LAYOUT_ARRAY_FIELDS)
    + CASE_RECORD_COUNT * len(CASE_CONTROL_ARRAY_FIELDS)
    + MAIN_RECORD_COUNT * len(MAIN_ARRAY_FIELDS)
    + CHECKPOINT_RECORD_COUNT * len(CHECKPOINT_ARRAY_FIELDS)
    + REPLAY_RECORD_COUNT * len(REPLAY_ARRAY_FIELDS)
    + CROSSOVER_RECORD_COUNT * len(CROSSOVER_ARRAY_FIELDS)
)
FIXED_NON_STATE_ARRAY_COUNT = (
    len(NPZ_IDENTITY_ARRAY_FIELDS) + SCIENTIFIC_FIXED_ARRAY_COUNT
)


def _coordinate_index(value: Any, size: int, label: str) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise ValueError(f"{label} index is out of range")
    return value


def _case_identity(case_index: int) -> dict[str, Any]:
    case_index = _coordinate_index(
        case_index,
        len(FIXED_CASE_SPECS),
        "Case",
    )
    frozen_index, cohort_ordinal, case_id = FIXED_CASE_SPECS[case_index]
    if frozen_index != case_index:
        raise RuntimeError("Frozen case indices are not contiguous")
    return {
        "case_index": case_index,
        "cohort_ordinal": cohort_ordinal,
        "case_id": case_id,
        "prefix": (f"case_c{case_index:02d}_o{cohort_ordinal:02d}_{case_id}"),
    }


def _main_record_identity(
    regime_index: int,
    path_index: int,
    step_from: int,
) -> dict[str, Any]:
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    path_index = _coordinate_index(
        path_index,
        len(GEOMETRY_PATHS),
        "Geometry path",
    )
    step_from = _coordinate_index(
        step_from,
        TRAJECTORY_STEP_COUNT,
        "Main step",
    )
    case = _case_identity(STEP_CASE_INDICES[step_from])
    record_ordinal = (
        regime_index * len(GEOMETRY_PATHS) + path_index
    ) * TRAJECTORY_STEP_COUNT + step_from
    regime = STATE_REGIMES[regime_index]
    path = GEOMETRY_PATHS[path_index]
    prefix = (
        f"main_m{record_ordinal:03d}_r{regime_index:02d}_{regime}"
        f"_p{path_index:02d}_{path}_t{step_from:02d}_to_t{step_from + 1:02d}"
        f"_c{case['case_index']:02d}_o{case['cohort_ordinal']:02d}"
        f"_{case['case_id']}"
    )
    return {
        "record_ordinal": record_ordinal,
        "regime_index": regime_index,
        "regime": regime,
        "path_index": path_index,
        "path": path,
        "step_from": step_from,
        "step_to": step_from + 1,
        **{key: case[key] for key in ("case_index", "cohort_ordinal", "case_id")},
        "prefix": prefix,
    }


def _checkpoint_record_identity(
    regime_index: int,
    path_index: int,
    checkpoint_index: int,
) -> dict[str, Any]:
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    path_index = _coordinate_index(
        path_index,
        len(GEOMETRY_PATHS),
        "Geometry path",
    )
    checkpoint_index = _coordinate_index(
        checkpoint_index,
        len(TRAJECTORY_CHECKPOINT_STEPS),
        "Checkpoint",
    )
    state_step = TRAJECTORY_CHECKPOINT_STEPS[checkpoint_index]
    record_ordinal = (regime_index * len(GEOMETRY_PATHS) + path_index) * len(
        TRAJECTORY_CHECKPOINT_STEPS
    ) + checkpoint_index
    regime = STATE_REGIMES[regime_index]
    path = GEOMETRY_PATHS[path_index]
    return {
        "record_ordinal": record_ordinal,
        "regime_index": regime_index,
        "regime": regime,
        "path_index": path_index,
        "path": path,
        "state_step": state_step,
        "prefix": (
            f"checkpoint_k{record_ordinal:03d}_r{regime_index:02d}_{regime}"
            f"_p{path_index:02d}_{path}_t{state_step:02d}"
        ),
    }


def _replay_record_identity(
    regime_index: int,
    path_index: int,
    replay_index: int,
) -> dict[str, Any]:
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    path_index = _coordinate_index(
        path_index,
        len(GEOMETRY_PATHS),
        "Geometry path",
    )
    replay_index = _coordinate_index(
        replay_index,
        len(TRAJECTORY_REPLAY_STEPS),
        "Replay",
    )
    step_from = TRAJECTORY_REPLAY_STEPS[replay_index]
    case = _case_identity(STEP_CASE_INDICES[step_from])
    record_ordinal = (regime_index * len(GEOMETRY_PATHS) + path_index) * len(
        TRAJECTORY_REPLAY_STEPS
    ) + replay_index
    regime = STATE_REGIMES[regime_index]
    path = GEOMETRY_PATHS[path_index]
    return {
        "record_ordinal": record_ordinal,
        "regime_index": regime_index,
        "regime": regime,
        "path_index": path_index,
        "path": path,
        "step_from": step_from,
        "step_to": step_from + 1,
        **{key: case[key] for key in ("case_index", "cohort_ordinal", "case_id")},
        "prefix": (
            f"replay_y{record_ordinal:03d}_r{regime_index:02d}_{regime}"
            f"_p{path_index:02d}_{path}_t{step_from:02d}_to_t{step_from + 1:02d}"
            f"_c{case['case_index']:02d}_o{case['cohort_ordinal']:02d}"
            f"_{case['case_id']}"
        ),
    }


def _crossover_record_identity(
    regime_index: int,
    crossover_index: int,
    case_index: int,
    history_index: int,
    geometry_index: int,
) -> dict[str, Any]:
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    crossover_index = _coordinate_index(
        crossover_index,
        len(CROSSOVER_STEPS),
        "Crossover checkpoint",
    )
    history_index = _coordinate_index(
        history_index,
        len(CROSSOVER_HISTORIES),
        "Crossover history",
    )
    geometry_index = _coordinate_index(
        geometry_index,
        len(CROSSOVER_EVALUATION_GEOMETRIES),
        "Crossover geometry",
    )
    case = _case_identity(case_index)
    record_ordinal = (
        (
            (regime_index * len(CROSSOVER_STEPS) + crossover_index)
            * len(FIXED_CASE_SPECS)
            + case_index
        )
        * len(CROSSOVER_HISTORIES)
        + history_index
    ) * len(CROSSOVER_EVALUATION_GEOMETRIES) + geometry_index
    regime = STATE_REGIMES[regime_index]
    state_step = CROSSOVER_STEPS[crossover_index]
    history = CROSSOVER_HISTORIES[history_index]
    geometry = CROSSOVER_EVALUATION_GEOMETRIES[geometry_index]
    return {
        "record_ordinal": record_ordinal,
        "regime_index": regime_index,
        "regime": regime,
        "state_step": state_step,
        **{key: case[key] for key in ("case_index", "cohort_ordinal", "case_id")},
        "history_index": history_index,
        "history": history,
        "geometry_index": geometry_index,
        "evaluation_geometry": geometry,
        "prefix": (
            f"crossover_x{record_ordinal:03d}_r{regime_index:02d}_{regime}"
            f"_t{state_step:02d}_c{case_index:02d}_o{case['cohort_ordinal']:02d}"
            f"_{case['case_id']}_h{history_index:02d}_{history}"
            f"_g{geometry_index:02d}_{geometry}"
        ),
    }


_EXPECTED_COMBINED_OPTIMIZER = "physicsnemo.optim.combined_optimizer.CombinedOptimizer"
_EXPECTED_OPTIMIZER_MEMBERS = (
    "physicsnemo.optim.muon.Muon",
    "torch.optim.adamw.AdamW",
)
_OPTIMIZER_HOOK_ATTRIBUTES = (
    "_optimizer_step_pre_hooks",
    "_optimizer_step_post_hooks",
    "_optimizer_state_dict_pre_hooks",
    "_optimizer_state_dict_post_hooks",
    "_optimizer_load_state_dict_pre_hooks",
    "_optimizer_load_state_dict_post_hooks",
)
# State-dict hooks are intentionally excluded: PhysicsNeMo installs a legitimate
# device-buffer compatibility hook, and this codec never calls state_dict APIs.
_MODULE_HOOK_ATTRIBUTES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
_PARAMETER_HOOK_ATTRIBUTES = (
    "_backward_hooks",
    "_post_accumulate_grad_hooks",
)
_GLOBAL_MODULE_HOOK_ATTRIBUTES = (
    "_global_buffer_registration_hooks",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
)
_TENSOR_RECORD_KIND = "microtrajectory_torch_tensor_v1"
_TENSOR_RECORD_KEYS = {
    "kind",
    "device",
    "layout",
    "stride",
    "storage_offset",
    "requires_grad",
    "value",
}
STATE_TREE_SCHEMA_VERSION = 1
_STATE_TREE_KINDS = frozenset({"complete_state", "rng_state"})
_CANONICAL_NUMPY_DTYPES = frozenset(
    {
        "|b1",
        "|i1",
        "|u1",
        "<i2",
        "<u2",
        "<i4",
        "<u4",
        "<i8",
        "<u8",
        "<f2",
        "<f4",
        "<f8",
        "<c8",
        "<c16",
    }
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _canonical_torch_stride(shape: Sequence[int]) -> tuple[int, ...]:
    running = 1
    reversed_strides = []
    for size in reversed(shape):
        reversed_strides.append(running)
        running *= max(size, 1)
    return tuple(reversed(reversed_strides))


def _canonical_numpy_strides(
    shape: Sequence[int],
    itemsize: int,
) -> tuple[int, ...]:
    if any(size == 0 for size in shape):
        return (0,) * len(shape)
    running = itemsize
    reversed_strides = []
    for size in reversed(shape):
        reversed_strides.append(running)
        running *= size
    return tuple(reversed(reversed_strides))


def _require_canonical_numpy_array(value: np.ndarray, label: str) -> None:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact ndarray")
    if (
        value.dtype.str not in _CANONICAL_NUMPY_DTYPES
        or value.dtype.fields is not None
        or value.dtype.subdtype is not None
        or value.dtype.metadata is not None
    ):
        raise ValueError(f"{label} dtype is not a canonical primitive")
    expected_strides = _canonical_numpy_strides(value.shape, value.dtype.itemsize)
    if (
        not value.flags.c_contiguous
        or value.strides != expected_strides
        or any(stride < 0 for stride in value.strides)
    ):
        raise ValueError(f"{label} does not have canonical C-order strides")
    if value.dtype.str == "|b1":
        raw = np.frombuffer(value.tobytes(order="C"), dtype=np.uint8)
        if bool(((raw != 0) & (raw != 1)).any()):
            raise ValueError(f"{label} has noncanonical Boolean storage")


def _require_canonical_torch_bool(value: torch.Tensor, label: str) -> None:
    if value.dtype is not torch.bool:
        return
    raw = value.detach().view(torch.uint8)
    if bool(((raw != 0) & (raw != 1)).any()):
        raise ValueError(f"{label} has noncanonical Boolean storage")


def _qualified_class(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _hash_update(digest: Any, value: Any) -> None:
    """Hash a deterministic, type-exact representation of ``value``."""
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise TypeError("Tensor subclasses have no canonical state hash")
        _require_canonical_tensor(value, "Hashed Tensor")
        _require_canonical_torch_bool(value, "Hashed Tensor")
        tensor = value.detach().cpu()
        digest.update(b"torch\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        _require_canonical_numpy_array(value, "Hashed ndarray")
        digest.update(b"numpy\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
        digest.update(b"end-mapping\0")
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _hash_update(digest, item)
        digest.update(b"end-sequence\0")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        digest.update(f"int\0{value}\0".encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float\0")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"str\0")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        return
    raise TypeError(f"Unsupported state value type: {type(value).__name__}")


def stable_sha256(value: Any) -> str:
    """Return the canonical type-exact SHA-256 of ``value``."""
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _require_canonical_tensor(value: torch.Tensor, label: str) -> None:
    expected_stride = _canonical_torch_stride(value.shape)
    if (
        value.layout != torch.strided
        or not value.is_contiguous()
        or value.stride() != expected_stride
        or value.storage_offset() != 0
        or value.is_conj()
        or value.is_neg()
    ):
        raise ValueError(
            f"{label} must be a canonical contiguous strided tensor with "
            "storage offset zero and resolved conjugate/negative views"
        )
    _require_canonical_torch_bool(value, label)


def _require_inert_tensor(value: torch.Tensor, label: str) -> None:
    if value.requires_grad or value.grad is not None:
        raise ValueError(f"{label} must not require or carry gradients")


def _tensor_metadata(
    value: torch.Tensor,
    *,
    allow_requires_grad: bool = False,
) -> dict[str, Any]:
    _require_canonical_tensor(value, "Checkpoint tensor")
    if not allow_requires_grad:
        _require_inert_tensor(value, "Checkpoint tensor")
    return {
        "device": str(value.device),
        "layout": str(value.layout),
        "stride": tuple(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "requires_grad": bool(value.requires_grad),
    }


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise TypeError("Tensor subclasses cannot be cloned as canonical state")
        _require_canonical_tensor(value, "State tensor")
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise TypeError("ndarray subclasses cannot be cloned as canonical state")
        _require_canonical_numpy_array(value, "State ndarray")
        return value.copy(order="C")
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, bytes):
        return bytes(value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"Unsupported state value type: {type(value).__name__}")


def _capture_optimizer_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise ValueError("Optimizer state contains a Tensor subclass")
        _require_inert_tensor(value, "Optimizer tensor")
        return {
            "kind": _TENSOR_RECORD_KIND,
            **_tensor_metadata(value),
            "value": value.detach().cpu().contiguous().clone(),
        }
    if isinstance(value, Mapping):
        return {key: _capture_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_capture_optimizer_value(item) for item in value)
    if isinstance(value, list):
        return [_capture_optimizer_value(item) for item in value]
    return _clone_value(value)


def _restore_optimizer_value(value: Any) -> Any:
    if (
        isinstance(value, Mapping)
        and set(value) == _TENSOR_RECORD_KEYS
        and value.get("kind") == _TENSOR_RECORD_KIND
    ):
        tensor = value["value"]
        if type(tensor) is not torch.Tensor:
            raise ValueError("Encoded optimizer tensor has no tensor value")
        _require_canonical_tensor(tensor, "Encoded optimizer tensor")
        if (
            value["layout"] != str(torch.strided)
            or tuple(value["stride"]) != tuple(tensor.stride())
            or value["storage_offset"] != tensor.storage_offset()
            or value["requires_grad"] is not False
        ):
            raise ValueError("Encoded optimizer tensor layout metadata differs")
        return tensor.detach().clone()
    if isinstance(value, Mapping):
        return {key: _restore_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_optimizer_value(item) for item in value)
    if isinstance(value, list):
        return [_restore_optimizer_value(item) for item in value]
    return _clone_value(value)


def _tensor_leaves(value: Any, label: str) -> tuple[tuple[str, torch.Tensor], ...]:
    if isinstance(value, torch.Tensor):
        return ((label, value),)
    if isinstance(value, Mapping):
        return tuple(
            leaf
            for key, item in value.items()
            for leaf in _tensor_leaves(item, f"{label}.{key}")
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            leaf
            for index, item in enumerate(value)
            for leaf in _tensor_leaves(item, f"{label}[{index}]")
        )
    return ()


def _claim_unique_tensor_storages(
    occupied: list[tuple[str, int, int, int, str]],
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for label, tensor in tensors:
        _require_canonical_tensor(tensor, label)
        device = str(tensor.device)
        storage_identity = int(tensor.untyped_storage()._cdata)
        start = int(tensor.data_ptr())
        nbytes = int(tensor.numel() * tensor.element_size())
        stop = start + nbytes
        for (
            previous_device,
            previous_storage,
            previous_start,
            previous_stop,
            previous_label,
        ) in occupied:
            same_storage = (
                device == previous_device and storage_identity == previous_storage
            )
            address_overlap = (
                device == previous_device
                and nbytes > 0
                and previous_stop > previous_start
                and max(start, previous_start) < min(stop, previous_stop)
            )
            if same_storage or address_overlap:
                raise ValueError(
                    f"Tensor storage alias between {previous_label} and {label}"
                )
        occupied.append((device, storage_identity, start, stop, label))


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def _named_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    names = tuple(name for name, _parameter in parameters)
    identities = tuple(id(parameter) for _name, parameter in parameters)
    if not parameters or len(names) != len(set(names)):
        raise ValueError("Model parameter names are empty or nonunique")
    if len(identities) != len(set(identities)):
        raise ValueError("Aliased trainable parameters are unsupported")
    if any(
        type(parameter) is not torch.nn.Parameter for _name, parameter in parameters
    ):
        raise ValueError("Every trainable parameter must be an exact Parameter")
    if any(not parameter.requires_grad for _name, parameter in parameters):
        raise ValueError("Every named parameter must be trainable")
    return parameters


def assert_gradients_cleared(model: torch.nn.Module) -> tuple[str, ...]:
    """Require and return the frozen ordered names whose gradients are ``None``."""
    parameters = _named_trainable_parameters(model)
    uncleared = [name for name, parameter in parameters if parameter.grad is not None]
    if uncleared:
        raise ValueError(f"Trainable parameter gradients are not None: {uncleared}")
    return tuple(name for name, _parameter in parameters)


def _named_modules_exact(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Module], ...]:
    modules = tuple(model.named_modules(remove_duplicate=False))
    names = tuple(name for name, _module in modules)
    identities = tuple(id(module) for _name, module in modules)
    if len(names) != len(set(names)):
        raise ValueError("Module names are nonunique")
    if len(identities) != len(set(identities)):
        raise ValueError("Aliased module objects are unsupported")
    return modules


def _parameter_registry(model: torch.nn.Module) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for module_name, module in _named_modules_exact(model):
        for local_name, parameter in module._parameters.items():
            name = f"{module_name}.{local_name}" if module_name else local_name
            if name in names:
                raise ValueError(f"Registered parameter name is nonunique: {name}")
            names.add(name)
            if parameter is not None and type(parameter) is not torch.nn.Parameter:
                raise ValueError(f"Registered parameter {name} is a subclass")
            records.append({"name": name, "present": parameter is not None})
    return tuple(records)


def _module_registry(model: torch.nn.Module) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for module_name, module in _named_modules_exact(model):
        for local_name, child in module._modules.items():
            name = f"{module_name}.{local_name}" if module_name else local_name
            if name in names:
                raise ValueError(f"Registered module name is nonunique: {name}")
            names.add(name)
            records.append({"name": name, "present": child is not None})
    return tuple(records)


def _registered_buffers(
    model: torch.nn.Module,
) -> tuple[tuple[str, bool, torch.Tensor | None], ...]:
    records: list[tuple[str, bool, torch.Tensor | None]] = []
    names: set[str] = set()
    for module_name, module in _named_modules_exact(model):
        unknown_nonpersistent = module._non_persistent_buffers_set.difference(
            module._buffers
        )
        if unknown_nonpersistent:
            raise ValueError(
                f"{module_name or '<root>'} has unknown nonpersistent buffers: "
                f"{sorted(unknown_nonpersistent)}"
            )
        for local_name, buffer in module._buffers.items():
            name = f"{module_name}.{local_name}" if module_name else local_name
            if name in names:
                raise ValueError(f"Registered buffer name is nonunique: {name}")
            names.add(name)
            if buffer is not None and type(buffer) is not torch.Tensor:
                raise ValueError(f"Registered buffer {name} is a Tensor subclass")
            records.append(
                (
                    name,
                    local_name not in module._non_persistent_buffers_set,
                    buffer,
                )
            )
    return tuple(records)


def _validate_model_runtime(model: torch.nn.Module) -> dict[str, Any]:
    if any(
        getattr(torch_module_runtime, attribute)
        for attribute in _GLOBAL_MODULE_HOOK_ATTRIBUTES
    ):
        raise ValueError("Global module hooks are forbidden")
    for name, module in _named_modules_exact(model):
        for attribute in _MODULE_HOOK_ATTRIBUTES:
            if getattr(module, attribute, {}):
                raise ValueError(f"Module hooks are forbidden on {name or '<root>'}")
    for name, parameter in _named_trainable_parameters(model):
        for attribute in _PARAMETER_HOOK_ATTRIBUTES:
            if getattr(parameter, attribute, {}):
                raise ValueError(f"Parameter hooks are forbidden on {name}")
    return {
        "global_hooks_present": False,
        "module_hooks_present": False,
        "parameter_hooks_present": False,
    }


def capture_model_state(model: torch.nn.Module) -> dict[str, Any]:
    """Capture every parameter, registered buffer, and module mode."""
    runtime = _validate_model_runtime(model)
    gradients_none = assert_gradients_cleared(model)
    named_parameters = _named_trainable_parameters(model)
    registered_buffers = _registered_buffers(model)
    occupied: list[tuple[str, int, int, int, str]] = []
    _claim_unique_tensor_storages(
        occupied,
        tuple((f"parameter {name}", parameter) for name, parameter in named_parameters)
        + tuple(
            (f"buffer {name}", buffer)
            for name, _persistent, buffer in registered_buffers
            if buffer is not None
        ),
    )
    parameters = tuple(
        {
            "name": name,
            **_tensor_metadata(parameter, allow_requires_grad=True),
            "value": parameter.detach().cpu().contiguous().clone(),
        }
        for name, parameter in named_parameters
    )
    parameter_registry = _parameter_registry(model)
    buffers = tuple(
        {
            "name": name,
            "persistent": persistent,
            **(
                {
                    "device": None,
                    "layout": None,
                    "stride": None,
                    "storage_offset": None,
                    "requires_grad": None,
                }
                if buffer is None
                else _tensor_metadata(buffer)
            ),
            "value": (
                None if buffer is None else buffer.detach().cpu().contiguous().clone()
            ),
        }
        for name, persistent, buffer in registered_buffers
    )
    module_modes = tuple(
        {
            "name": name,
            "module_class": _qualified_class(module),
            "training": module.training,
        }
        for name, module in _named_modules_exact(model)
    )
    if any(type(record["training"]) is not bool for record in module_modes):
        raise ValueError("Every module training flag must be Boolean")
    return {
        "model_class": _qualified_class(model),
        "runtime": runtime,
        "parameter_registry": parameter_registry,
        "parameters": parameters,
        "buffers": buffers,
        "module_registry": _module_registry(model),
        "module_modes": module_modes,
        "gradients_none": gradients_none,
    }


def _split_registered_name(name: str) -> tuple[str, str]:
    module_name, separator, local_name = name.rpartition(".")
    if not separator:
        return "", name
    return module_name, local_name


def restore_model_state(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    """Restore a model snapshot and verify its canonical hash."""
    _exact_keys(
        state,
        {
            "model_class",
            "runtime",
            "parameter_registry",
            "parameters",
            "buffers",
            "module_registry",
            "module_modes",
            "gradients_none",
        },
        "model state",
    )
    if state["model_class"] != _qualified_class(model):
        raise ValueError("Model class differs from checkpoint")
    if state["runtime"] != _validate_model_runtime(model):
        raise ValueError("Model runtime differs from checkpoint")
    for label, observed, expected in (
        (
            "parameter",
            _parameter_registry(model),
            tuple(state["parameter_registry"]),
        ),
        (
            "module",
            _module_registry(model),
            tuple(state["module_registry"]),
        ),
    ):
        if observed != expected:
            raise ValueError(f"Registered {label} schema differs from checkpoint")

    current_parameters = _named_trainable_parameters(model)
    parameter_records = tuple(state["parameters"])
    if tuple(record["name"] for record in parameter_records) != tuple(
        name for name, _parameter in current_parameters
    ):
        raise ValueError("Model parameter order differs from checkpoint")
    with torch.no_grad():
        for (name, parameter), record in zip(
            current_parameters, parameter_records, strict=True
        ):
            _exact_keys(
                record,
                {
                    "name",
                    "device",
                    "layout",
                    "stride",
                    "storage_offset",
                    "requires_grad",
                    "value",
                },
                f"parameter {name}",
            )
            value = record["value"]
            metadata = _tensor_metadata(parameter, allow_requires_grad=True)
            if (
                type(value) is not torch.Tensor
                or value.dtype != parameter.dtype
                or value.shape != parameter.shape
                or any(record[key] != metadata[key] for key in metadata)
            ):
                raise ValueError(f"Parameter schema differs for {name}")
            _require_canonical_tensor(value, f"Checkpoint parameter {name}")
            _require_inert_tensor(value, f"Checkpoint parameter {name}")
            parameter.copy_(value.to(device=parameter.device))
            parameter.grad = None

        current_buffers = _registered_buffers(model)
        buffer_records = tuple(state["buffers"])
        current_buffer_schema = tuple(
            (name, persistent) for name, persistent, _buffer in current_buffers
        )
        checkpoint_buffer_schema = tuple(
            (record["name"], record["persistent"]) for record in buffer_records
        )
        if checkpoint_buffer_schema != current_buffer_schema:
            raise ValueError("Registered-buffer schema differs from checkpoint")
        for (name, _persistent, buffer), record in zip(
            current_buffers, buffer_records, strict=True
        ):
            _exact_keys(
                record,
                {
                    "name",
                    "persistent",
                    "device",
                    "layout",
                    "stride",
                    "storage_offset",
                    "requires_grad",
                    "value",
                },
                f"buffer {name}",
            )
            value = record["value"]
            if value is None:
                if buffer is not None or any(
                    record[key] is not None
                    for key in (
                        "device",
                        "layout",
                        "stride",
                        "storage_offset",
                        "requires_grad",
                    )
                ):
                    raise ValueError(f"Buffer {name} changed from None")
                continue
            metadata = _tensor_metadata(buffer) if buffer is not None else {}
            if (
                buffer is None
                or type(value) is not torch.Tensor
                or value.dtype != buffer.dtype
                or value.shape != buffer.shape
                or any(record[key] != metadata[key] for key in metadata)
            ):
                raise ValueError(f"Buffer schema differs for {name}")
            _require_canonical_tensor(value, f"Checkpoint buffer {name}")
            _require_inert_tensor(value, f"Checkpoint buffer {name}")
            buffer.copy_(value.to(device=buffer.device))

    current_modules = _named_modules_exact(model)
    mode_records = tuple(state["module_modes"])
    if tuple(record["name"] for record in mode_records) != tuple(
        name for name, _module in current_modules
    ):
        raise ValueError("Module-name order differs from checkpoint")
    for (name, module), record in zip(current_modules, mode_records, strict=True):
        _exact_keys(
            record,
            {"name", "module_class", "training"},
            f"module mode {name}",
        )
        if record["module_class"] != _qualified_class(module):
            raise ValueError(f"Module class differs for {name}")
        if type(record["training"]) is not bool:
            raise ValueError(f"Module mode is not Boolean for {name}")
        module.training = record["training"]

    if tuple(state["gradients_none"]) != assert_gradients_cleared(model):
        raise ValueError("Cleared-gradient manifest differs from checkpoint")
    if stable_sha256(capture_model_state(model)) != stable_sha256(state):
        raise ValueError("Restored model state differs from checkpoint bytes")


def _member_parameter_groups(
    member: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
) -> tuple[dict[str, Any], ...]:
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(member.param_groups):
        if "params" not in group:
            raise ValueError(f"Optimizer group {group_index} has no parameters")
        parameter_names: list[str] = []
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise ValueError("Optimizer contains a parameter outside the model")
            parameter_names.append(name)
        options = {
            key: _capture_optimizer_value(value)
            for key, value in sorted(group.items())
            if key != "params"
        }
        groups.append(
            {
                "parameters": tuple(parameter_names),
                "options": options,
            }
        )
    return tuple(groups)


def _ordinary_bound_method(value: object, method_name: str) -> bool:
    method = getattr(value, method_name, None)
    return getattr(method, "__self__", None) is value and getattr(
        method, "__func__", None
    ) is getattr(type(value), method_name, None)


def _validate_optimizer_runtime(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    if _qualified_class(optimizer) != _EXPECTED_COMBINED_OPTIMIZER:
        raise ValueError("Optimizer is not the exact CombinedOptimizer class")
    members = tuple(getattr(optimizer, "optimizers", ()))
    member_classes = tuple(_qualified_class(member) for member in members)
    if member_classes != _EXPECTED_OPTIMIZER_MEMBERS:
        raise ValueError(f"CombinedOptimizer member classes differ: {member_classes}")
    if getattr(optimizer, "_torch_compile_kwargs", object()) is not None:
        raise ValueError("Compiled optimizer steps are forbidden")
    if not (
        _ordinary_bound_method(optimizer, "step")
        and _ordinary_bound_method(optimizer, "zero_grad")
    ):
        raise ValueError("CombinedOptimizer step or zero_grad is overridden")
    step_functions = tuple(getattr(optimizer, "step_fns", ()))
    if len(step_functions) != len(members) or any(
        getattr(step_function, "__self__", None) is not member
        or getattr(step_function, "__func__", None)
        is not getattr(type(member), "step", None)
        for step_function, member in zip(step_functions, members, strict=True)
    ):
        raise ValueError(
            "CombinedOptimizer step functions are not ordinary member steps"
        )
    if any(
        not (
            _ordinary_bound_method(member, "step")
            and _ordinary_bound_method(member, "zero_grad")
        )
        for member in members
    ):
        raise ValueError("A contained optimizer method is overridden")
    for owner in (optimizer, *members):
        for attribute in _OPTIMIZER_HOOK_ATTRIBUTES:
            if getattr(owner, attribute, {}):
                raise ValueError("Optimizer hooks are forbidden")
    if (
        torch_optimizer._global_optimizer_pre_hooks
        or torch_optimizer._global_optimizer_post_hooks
    ):
        raise ValueError("Global optimizer hooks are forbidden")
    expected_groups = tuple(
        group for member in members for group in member.param_groups
    )
    if tuple(id(group) for group in optimizer.param_groups) != tuple(
        id(group) for group in expected_groups
    ):
        raise ValueError("CombinedOptimizer group aliases differ from its members")
    if optimizer.defaults:
        raise ValueError("CombinedOptimizer wrapper defaults must be empty")
    return {
        "combined_optimizer_class": _EXPECTED_COMBINED_OPTIMIZER,
        "member_classes": _EXPECTED_OPTIMIZER_MEMBERS,
        "compiled": False,
        "ordinary_step_functions": True,
        "hooks_present": False,
        "global_hooks_present": False,
    }


def capture_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Capture complete CombinedOptimizer state addressed by parameter name."""
    runtime = _validate_optimizer_runtime(optimizer)
    members = tuple(getattr(optimizer, "optimizers", ()))
    named_parameters = _named_trainable_parameters(model)
    name_by_id = {id(parameter): name for name, parameter in named_parameters}
    parameter_by_name = {name: parameter for name, parameter in named_parameters}
    occupied: list[tuple[str, int, int, int, str]] = []
    _claim_unique_tensor_storages(
        occupied,
        tuple(
            (f"model parameter {name}", parameter)
            for name, parameter in named_parameters
        )
        + tuple(
            (f"model buffer {name}", buffer)
            for name, _persistent, buffer in _registered_buffers(model)
            if buffer is not None
        ),
    )

    member_records: list[dict[str, Any]] = []
    all_member_names: list[str] = []
    for member_index, member in enumerate(members):
        groups = _member_parameter_groups(member, name_by_id)
        member_names = [name for group in groups for name in group["parameters"]]
        if len(member_names) != len(set(member_names)):
            raise ValueError("An optimizer member contains duplicate parameters")
        member_parameter_ids = {
            id(parameter)
            for group in member.param_groups
            for parameter in group["params"]
        }
        unknown_state = [
            key
            for key in member.state
            if not isinstance(key, torch.nn.Parameter)
            or id(key) not in member_parameter_ids
        ]
        if unknown_state:
            raise ValueError("Optimizer state is attached to the wrong member")
        for group_index, group in enumerate(member.param_groups):
            for key, value in group.items():
                if key != "params":
                    _claim_unique_tensor_storages(
                        occupied,
                        _tensor_leaves(
                            value,
                            f"optimizer {member_index} group {group_index}.{key}",
                        ),
                    )
        for name in member_names:
            parameter = parameter_by_name[name]
            if parameter in member.state:
                _claim_unique_tensor_storages(
                    occupied,
                    _tensor_leaves(
                        member.state[parameter],
                        f"optimizer {member_index} state {name}",
                    ),
                )
        states = tuple(
            {
                "parameter": name,
                "present": parameter_by_name[name] in member.state,
                "values": {
                    key: _capture_optimizer_value(value)
                    for key, value in sorted(
                        member.state.get(parameter_by_name[name], {}).items()
                    )
                },
            }
            for name in member_names
        )
        member_records.append(
            {
                "optimizer_class": _qualified_class(member),
                "parameter_groups": groups,
                "parameter_states": states,
            }
        )
        all_member_names.extend(member_names)

    expected_names = [name for name, _parameter in named_parameters]
    if len(all_member_names) != len(set(all_member_names)) or set(
        all_member_names
    ) != set(expected_names):
        raise ValueError("CombinedOptimizer partition is not exact and exhaustive")
    return {
        "optimizer_class": _qualified_class(optimizer),
        "runtime": runtime,
        "members": tuple(member_records),
    }


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    state: Mapping[str, Any],
) -> None:
    """Restore name-addressed CombinedOptimizer state and verify exactness."""
    _exact_keys(
        state,
        {"optimizer_class", "runtime", "members"},
        "optimizer state",
    )
    if state["optimizer_class"] != _qualified_class(optimizer):
        raise ValueError("CombinedOptimizer class differs from checkpoint")
    if state["runtime"] != _validate_optimizer_runtime(optimizer):
        raise ValueError("CombinedOptimizer runtime differs from checkpoint")
    members = tuple(getattr(optimizer, "optimizers", ()))
    member_records = tuple(state["members"])
    if len(members) != len(member_records):
        raise ValueError("CombinedOptimizer member count differs from checkpoint")

    named_parameters = _named_trainable_parameters(model)
    name_by_id = {id(parameter): name for name, parameter in named_parameters}
    for member, record in zip(members, member_records, strict=True):
        _exact_keys(
            record,
            {"optimizer_class", "parameter_groups", "parameter_states"},
            "optimizer member",
        )
        if record["optimizer_class"] != _qualified_class(member):
            raise ValueError("Contained optimizer class differs from checkpoint")
        current_groups = _member_parameter_groups(member, name_by_id)
        checkpoint_groups = tuple(record["parameter_groups"])
        if tuple(group["parameters"] for group in current_groups) != tuple(
            group["parameters"] for group in checkpoint_groups
        ):
            raise ValueError("Optimizer parameter groups differ from checkpoint")

        current_state_dict = member.state_dict()
        name_to_index: dict[str, int] = {}
        for group, state_group in zip(
            member.param_groups,
            current_state_dict["param_groups"],
            strict=True,
        ):
            for parameter, index in zip(
                group["params"], state_group["params"], strict=True
            ):
                name_to_index[name_by_id[id(parameter)]] = index

        parameter_states = tuple(record["parameter_states"])
        for parameter_state in parameter_states:
            _exact_keys(
                parameter_state,
                {"parameter", "present", "values"},
                "optimizer parameter state",
            )
            if type(parameter_state["present"]) is not bool:
                raise ValueError("Optimizer state membership is not Boolean")
        checkpoint_names = tuple(
            parameter_state["parameter"] for parameter_state in parameter_states
        )
        expected_names = tuple(
            name for group in checkpoint_groups for name in group["parameters"]
        )
        if checkpoint_names != expected_names:
            raise ValueError("Optimizer state parameter order differs from groups")
        conventional_state = {
            name_to_index[parameter_state["parameter"]]: _restore_optimizer_value(
                parameter_state["values"]
            )
            for parameter_state in parameter_states
            if parameter_state["present"]
        }
        conventional_groups = [
            {
                **_restore_optimizer_value(group["options"]),
                "params": [name_to_index[name] for name in group["parameters"]],
            }
            for group in checkpoint_groups
        ]
        member.load_state_dict(
            {
                "state": conventional_state,
                "param_groups": conventional_groups,
            }
        )
    optimizer.param_groups = [
        group for member in members for group in member.param_groups
    ]
    if state["runtime"] != _validate_optimizer_runtime(optimizer):
        raise ValueError("Restored CombinedOptimizer runtime differs")
    if stable_sha256(capture_optimizer_state(optimizer, model)) != stable_sha256(state):
        raise ValueError("Restored optimizer state differs from checkpoint bytes")


def _validate_generator_inventory(
    explicit_generators: Mapping[str, torch.Generator],
) -> tuple[str, ...]:
    raw_names = tuple(explicit_generators)
    if any(type(name) is not str or not name for name in raw_names):
        raise ValueError("Explicit generator names must be nonempty exact strings")
    names = tuple(sorted(raw_names))
    generators = tuple(explicit_generators[name] for name in names)
    if any(not isinstance(generator, torch.Generator) for generator in generators):
        raise ValueError("Explicit generator inventory contains a non-generator")
    if len({id(generator) for generator in generators}) != len(generators):
        raise ValueError("Explicit generator identities must be unique")
    default_generators = {id(torch.default_generator)}
    default_generators.update(
        id(generator) for generator in getattr(torch.cuda, "default_generators", ())
    )
    if any(id(generator) in default_generators for generator in generators):
        raise ValueError("A global default generator cannot be explicit")
    return names


def _reachable_torch_generators(
    value: Any,
    label: str,
    *,
    seen: set[int],
) -> tuple[dict[str, str], ...]:
    if isinstance(value, torch.Generator):
        return ({"path": label, "device": str(value.device)},)
    if isinstance(value, (torch.Tensor, np.ndarray, torch.nn.Module)):
        return ()
    if isinstance(value, torch.optim.Optimizer):
        return ()
    if isinstance(
        value,
        (
            ModuleType,
            Path,
            type,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.FunctionType,
            types.MethodType,
        ),
    ):
        return ()
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return ()
    identity = id(value)
    if identity in seen:
        return ()
    seen.add(identity)
    if isinstance(value, Mapping):
        return tuple(
            record
            for key, item in value.items()
            for record in (
                *_reachable_torch_generators(
                    key,
                    f"{label}.key[{key!r}]",
                    seen=seen,
                ),
                *_reachable_torch_generators(
                    item,
                    f"{label}[{key!r}]",
                    seen=seen,
                ),
            )
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(
            record
            for index, item in enumerate(value)
            for record in _reachable_torch_generators(
                item,
                f"{label}[{index}]",
                seen=seen,
            )
        )
    try:
        attributes = vars(value)
    except TypeError:
        return ()
    return _reachable_torch_generators(
        attributes,
        f"{label}.__dict__",
        seen=seen,
    )


def _assert_no_hidden_package_generators(package: TrajectoryPackage) -> None:
    """Reject generator objects reachable from model or optimizer state."""
    _require_trajectory_package(package)
    seen: set[int] = set()
    records = []
    for name, module in _named_modules_exact(package.model):
        records.extend(
            _reachable_torch_generators(
                vars(module),
                f"model.{name or '<root>'}",
                seen=seen,
            )
        )
    _validate_optimizer_runtime(package.optimizer)
    records.extend(
        _reachable_torch_generators(
            vars(package.optimizer),
            "optimizer.combined",
            seen=seen,
        )
    )
    for index, member in enumerate(package.optimizer.optimizers):
        records.extend(
            _reachable_torch_generators(
                vars(member),
                f"optimizer.member[{index}]",
                seen=seen,
            )
        )
    if records:
        raise ValueError(f"Package contains hidden torch.Generator objects: {records}")


def _require_raw_rng_tensor(value: Any, label: str) -> None:
    if type(value) is not torch.Tensor:
        raise ValueError(f"{label} must be an exact Tensor")
    _require_canonical_tensor(value, label)
    _require_inert_tensor(value, label)
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional CPU uint8 Tensor")


def capture_rng_state(
    explicit_generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    """Capture canonical raw global and explicitly used generator RNG states."""
    names = _validate_generator_inventory(explicit_generators)
    return {
        "python": _clone_value(random.getstate()),
        "numpy": _clone_value(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": tuple(
            value.cpu().clone() for value in torch.cuda.get_rng_state_all()
        ),
        "cuda_device_count": torch.cuda.device_count(),
        "explicit_generators": tuple(
            {
                "name": name,
                "device": str(explicit_generators[name].device),
                "state": explicit_generators[name].get_state().cpu().clone(),
            }
            for name in names
        ),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    explicit_generators: Mapping[str, torch.Generator],
) -> None:
    """Restore raw RNG states and verify their canonical hash."""
    _exact_keys(
        state,
        {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "cuda_device_count",
            "explicit_generators",
        },
        "RNG state",
    )
    if (
        type(state["cuda_device_count"]) is not int
        or state["cuda_device_count"] != torch.cuda.device_count()
    ):
        raise ValueError("Visible CUDA device count differs from checkpoint")
    _require_raw_rng_tensor(state["torch_cpu"], "Torch CPU RNG state")
    for device_index, value in enumerate(state["torch_cuda"]):
        _require_raw_rng_tensor(value, f"Torch CUDA RNG state {device_index}")
    expected_names = _validate_generator_inventory(explicit_generators)
    records = tuple(state["explicit_generators"])
    if tuple(record["name"] for record in records) != expected_names:
        raise ValueError("Explicit generator names differ from checkpoint")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(list(state["torch_cuda"]))
    for record in records:
        _exact_keys(record, {"name", "device", "state"}, "explicit generator")
        generator = explicit_generators[record["name"]]
        if str(generator.device) != record["device"]:
            raise ValueError(f"Explicit generator device differs for {record['name']}")
        _require_raw_rng_tensor(
            record["state"],
            f"Explicit generator {record['name']} RNG state",
        )
        generator.set_state(record["state"])
    if stable_sha256(capture_rng_state(explicit_generators)) != stable_sha256(state):
        raise ValueError("Restored RNG state differs from checkpoint bytes")


def capture_complete_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    explicit_generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    """Capture the complete replayable state package."""
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "model": capture_model_state(model),
        "optimizer": capture_optimizer_state(optimizer, model),
        "rng": capture_rng_state(explicit_generators),
    }


def restore_complete_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    *,
    explicit_generators: Mapping[str, torch.Generator],
) -> None:
    """Restore a complete package and require a byte-exact round trip."""
    _exact_keys(state, {"schema_version", "model", "optimizer", "rng"}, "state")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise ValueError("Complete-state schema version differs")
    restore_model_state(model, state["model"])
    restore_optimizer_state(optimizer, model, state["optimizer"])
    restore_rng_state(state["rng"], explicit_generators)
    restored = capture_complete_state(
        model,
        optimizer,
        explicit_generators=explicit_generators,
    )
    if stable_sha256(restored) != stable_sha256(state):
        raise ValueError("Complete-state round trip differs from checkpoint bytes")


class TrajectoryPackage:
    """One persistent model, optimizer, and explicit-RNG trajectory package."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        explicit_generators: Mapping[str, torch.Generator],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.explicit_generators = dict(explicit_generators)


ForwardLoss = Callable[
    [TrajectoryPackage],
    tuple[Mapping[str, torch.Tensor], torch.Tensor],
]
PackageFactory = Callable[[], TrajectoryPackage]
TransitionExecutor = Callable[
    ...,
    tuple[dict[str, Any], dict[str, Any]],
]


def _require_trajectory_package(package: TrajectoryPackage) -> None:
    if type(package) is not TrajectoryPackage:
        raise TypeError("Trajectory package must be an exact TrajectoryPackage")
    if not isinstance(package.model, torch.nn.Module):
        raise TypeError("Trajectory package model must be a Module")
    if not isinstance(package.optimizer, torch.optim.Optimizer):
        raise TypeError("Trajectory package optimizer must be an Optimizer")
    if not isinstance(package.explicit_generators, Mapping):
        raise TypeError("Trajectory package generators must be a mapping")


def _require_finite_state_value(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if (
            np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.complexfloating)
        ) and not bool(np.isfinite(value).all()):
            raise ValueError(f"{label} contains a non-finite array")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_state_value(item, f"{label}.{key!r}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_state_value(item, f"{label}[{index}]")


def _capture_valid_complete_state(
    package: TrajectoryPackage,
) -> dict[str, Any]:
    _require_trajectory_package(package)
    state = capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    _require_finite_state_value(state, "Complete state")
    return state


def _complete_state_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    if type(state) is not dict:
        raise TypeError("Complete state must be an exact dictionary")
    _exact_keys(state, {"schema_version", "model", "optimizer", "rng"}, "state")
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != STATE_SCHEMA_VERSION
    ):
        raise ValueError("Complete-state schema version differs")
    return {
        "complete": stable_sha256(state),
        "model": stable_sha256(state["model"]),
        "optimizer": stable_sha256(state["optimizer"]),
        "rng": stable_sha256(state["rng"]),
    }


def _flatten_parameters_float32(model: torch.nn.Module) -> torch.Tensor:
    vector = torch.cat(
        tuple(
            parameter.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            for _name, parameter in _named_trainable_parameters(model)
        )
    ).contiguous()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Flattened model parameters are non-finite")
    return vector.clone()


def _flatten_gradients_float32(model: torch.nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    missing: list[str] = []
    for name, parameter in _named_trainable_parameters(model):
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if type(gradient) is not torch.Tensor:
            raise ValueError(f"Gradient for {name} is a Tensor subclass")
        if gradient.is_sparse:
            raise ValueError(f"Sparse gradient is unsupported for {name}")
        if gradient.shape != parameter.shape:
            raise ValueError(f"Gradient shape differs for {name}")
        parts.append(
            gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        )
    if missing:
        raise ValueError(f"Parameters have absent gradients: {missing}")
    vector = torch.cat(tuple(parts)).contiguous()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Flattened model gradients are non-finite")
    return vector.clone()


def _ordered_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], ...]:
    _validate_optimizer_runtime(optimizer)
    records: list[dict[str, Any]] = []
    for member_index, member in enumerate(optimizer.optimizers):
        for group_index, group in enumerate(member.param_groups):
            value = group.get("lr")
            if type(value) not in {int, float} or isinstance(value, bool):
                raise ValueError(
                    f"Optimizer {member_index} group {group_index} LR is not scalar"
                )
            value_float64 = float(value)
            if not math.isfinite(value_float64) or value_float64 <= 0.0:
                raise ValueError(
                    f"Optimizer {member_index} group {group_index} LR is not positive"
                )
            records.append(
                {
                    "member_index": member_index,
                    "member_class": _qualified_class(member),
                    "group_index": group_index,
                    "value_float64": value_float64,
                }
            )
    if not records:
        raise ValueError("Optimizer has no learning rates")
    return tuple(records)


def _active_stochastic_modules(model: torch.nn.Module) -> tuple[str, ...]:
    stochastic_types = (
        torch.nn.modules.dropout._DropoutNd,
        torch.nn.RReLU,
    )
    return tuple(
        name or "<root>"
        for name, module in _named_modules_exact(model)
        if module.training and isinstance(module, stochastic_types)
    )


def _model_execution_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    buffer_schema = []
    for record in state["buffers"]:
        value = record["value"]
        buffer_schema.append(
            {
                key: record[key]
                for key in (
                    "name",
                    "persistent",
                    "device",
                    "layout",
                    "stride",
                    "storage_offset",
                    "requires_grad",
                )
            }
            | {
                "present": value is not None,
                "dtype": None if value is None else str(value.dtype),
                "shape": None if value is None else tuple(value.shape),
            }
        )
    return {
        "model_class": state["model_class"],
        "runtime": state["runtime"],
        "parameter_registry": state["parameter_registry"],
        "module_registry": state["module_registry"],
        "module_modes": state["module_modes"],
        "buffer_schema": tuple(buffer_schema),
    }


def _capture_outputs(
    outputs: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("Forward outputs must be a nonempty mapping")
    names = tuple(outputs)
    if any(type(name) is not str or not name for name in names):
        raise ValueError("Forward-output names must be nonempty exact strings")
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        value = outputs[name]
        if type(value) is not torch.Tensor or not torch.is_floating_point(value):
            raise ValueError(f"Forward output {name} must be a floating Tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Forward output {name} is non-finite")
        records.append(
            {
                "name": name,
                "value_float32": (
                    value.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .clone()
                ),
            }
        )
    return tuple(records)


def execute_stateful_transition(
    package: TrajectoryPackage,
    *,
    step_index: int,
    frozen_rng_state: Mapping[str, Any],
    forward_loss: ForwardLoss,
    require_rng_continuation: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Take exactly one authenticated forward/backward/optimizer transition."""
    if type(step_index) is not int or not 0 <= step_index <= TRAJECTORY_STEP_COUNT:
        raise ValueError("Transition step index must be an integer from 0 through 16")
    if not callable(forward_loss):
        raise TypeError("Forward/loss callback must be callable")
    if type(require_rng_continuation) is not bool:
        raise TypeError("RNG-continuation requirement must be Boolean")

    continuation_state = _capture_valid_complete_state(package)
    continuation_hashes = _complete_state_hashes(continuation_state)
    restore_rng_state(
        frozen_rng_state,
        package.explicit_generators,
    )
    package.optimizer.zero_grad(set_to_none=True)
    parameter_names = assert_gradients_cleared(package.model)
    active_stochastic = _active_stochastic_modules(package.model)
    if active_stochastic:
        raise ValueError(
            f"Active stochastic modules are forbidden: {active_stochastic}"
        )

    pre_state = _capture_valid_complete_state(package)
    pre_hashes = _complete_state_hashes(pre_state)
    if pre_hashes["model"] != continuation_hashes["model"]:
        raise ValueError("Per-step RNG restore mutated the model state")
    if pre_hashes["optimizer"] != continuation_hashes["optimizer"]:
        raise ValueError("Per-step RNG restore mutated the optimizer state")
    if require_rng_continuation and pre_hashes["rng"] != continuation_hashes["rng"]:
        raise ValueError("Frozen main-step RNG is not the preceding continuation RNG")
    before = _flatten_parameters_float32(package.model)
    learning_rates_pre = _ordered_learning_rates(package.optimizer)

    outputs, loss = forward_loss(package)
    output_records = _capture_outputs(outputs)
    if type(loss) is not torch.Tensor or loss.ndim != 0:
        raise ValueError("Forward/loss callback must return a scalar Tensor loss")
    if not torch.is_floating_point(loss) or not bool(torch.isfinite(loss)):
        raise ValueError("Transition loss is non-finite or non-floating")
    loss_float64 = np.asarray([float(loss.detach().item())], dtype="<f8")
    callback_model_state = capture_model_state(package.model)
    active_stochastic = _active_stochastic_modules(package.model)
    if active_stochastic:
        raise ValueError(
            f"Forward/loss callback activated stochastic modules: {active_stochastic}"
        )
    if _model_execution_contract(callback_model_state) != _model_execution_contract(
        pre_state["model"]
    ):
        raise ValueError("Forward/loss callback changed the model execution contract")
    if stable_sha256(callback_model_state["parameters"]) != stable_sha256(
        pre_state["model"]["parameters"]
    ):
        raise ValueError("Forward/loss callback mutated model parameters")
    if stable_sha256(
        capture_optimizer_state(package.optimizer, package.model)
    ) != stable_sha256(pre_state["optimizer"]):
        raise ValueError("Forward/loss callback mutated optimizer state")
    if assert_gradients_cleared(package.model) != parameter_names:
        raise ValueError("Forward/loss callback populated parameter gradients")

    loss.backward()
    gradient = _flatten_gradients_float32(package.model)
    package.optimizer.step()
    after = _flatten_parameters_float32(package.model)
    update = (after - before).contiguous()
    if not bool(torch.isfinite(update).all()):
        raise ValueError("Flattened parameter update is non-finite")
    learning_rates_post = _ordered_learning_rates(package.optimizer)

    package.optimizer.zero_grad(set_to_none=True)
    if assert_gradients_cleared(package.model) != parameter_names:
        raise ValueError("Trainable parameter order changed during transition")
    post_state = _capture_valid_complete_state(package)
    post_hashes = _complete_state_hashes(post_state)

    record = {
        "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
        "step_index": step_index,
        "parameter_names": parameter_names,
        "parameter_count": int(before.numel()),
        "parameter_order_sha256": stable_sha256(parameter_names),
        "outputs": output_records,
        "loss_float64": loss_float64,
        "gradient_float32": gradient,
        "parameter_update_float32": update.clone(),
        "learning_rates_pre": learning_rates_pre,
        "learning_rates_post": learning_rates_post,
        "state_sha256": {
            "continuation": continuation_hashes,
            "pre": pre_hashes,
            "post": post_hashes,
        },
        "rng_state": {
            "pre": _clone_value(pre_state["rng"]),
            "post": _clone_value(post_state["rng"]),
        },
    }
    return record, post_state


def _reachable_callback_generators(
    value: Any,
    label: str,
    *,
    seen: set[int],
) -> tuple[dict[str, str], ...]:
    """Find generator objects captured by an executable callback."""
    if isinstance(value, torch.Generator):
        return ({"path": label, "device": str(value.device)},)
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return ()
    if isinstance(value, (torch.Tensor, np.ndarray, ModuleType, Path, type)):
        return ()
    identity = id(value)
    if identity in seen:
        return ()
    seen.add(identity)

    if isinstance(value, types.MethodType):
        return (
            *_reachable_callback_generators(
                value.__func__,
                f"{label}.__func__",
                seen=seen,
            ),
            *_reachable_callback_generators(
                value.__self__,
                f"{label}.__self__",
                seen=seen,
            ),
        )
    if isinstance(value, types.BuiltinMethodType):
        return _reachable_callback_generators(
            value.__self__,
            f"{label}.__self__",
            seen=seen,
        )
    if isinstance(value, types.FunctionType):
        records = list(
            _reachable_callback_generators(
                value.__defaults__,
                f"{label}.__defaults__",
                seen=seen,
            )
        )
        records.extend(
            _reachable_callback_generators(
                value.__kwdefaults__,
                f"{label}.__kwdefaults__",
                seen=seen,
            )
        )
        for index, cell in enumerate(value.__closure__ or ()):
            try:
                cell_value = cell.cell_contents
            except ValueError:
                continue
            records.extend(
                _reachable_callback_generators(
                    cell_value,
                    f"{label}.__closure__[{index}]",
                    seen=seen,
                )
            )
        return tuple(records)
    if isinstance(value, functools.partial):
        return (
            *_reachable_callback_generators(
                value.func,
                f"{label}.func",
                seen=seen,
            ),
            *_reachable_callback_generators(
                value.args,
                f"{label}.args",
                seen=seen,
            ),
            *_reachable_callback_generators(
                value.keywords,
                f"{label}.keywords",
                seen=seen,
            ),
        )
    if isinstance(value, Mapping):
        return tuple(
            record
            for key, item in value.items()
            for record in (
                *_reachable_callback_generators(
                    key,
                    f"{label}.key[{key!r}]",
                    seen=seen,
                ),
                *_reachable_callback_generators(
                    item,
                    f"{label}[{key!r}]",
                    seen=seen,
                ),
            )
        )
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return tuple(
            record
            for index, item in enumerate(value)
            for record in _reachable_callback_generators(
                item,
                f"{label}[{index}]",
                seen=seen,
            )
        )
    try:
        attributes = vars(value)
    except TypeError:
        return ()
    return _reachable_callback_generators(
        attributes,
        f"{label}.__dict__",
        seen=seen,
    )


def _assert_no_callback_generators(forward_loss: ForwardLoss) -> None:
    records = _reachable_callback_generators(
        forward_loss,
        "forward_loss",
        seen=set(),
    )
    if records:
        raise ValueError(
            f"Forward/loss callback captures torch.Generator objects: {records}"
        )


class _RejectNondeterministicSeededOps(TorchDispatchMode):
    """Fail closed on every ATen operator tagged as seeded nondeterminism."""

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        if torch.Tag.nondeterministic_seeded in getattr(func, "tags", ()):
            raise RuntimeError(
                f"Seeded nondeterministic ATen operator is forbidden: {func}"
            )
        return func(*args, **({} if kwargs is None else kwargs))


def _synchronize_cuda_if_requested(synchronize_cuda: bool) -> None:
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _authenticate_strict_transition_rng(record: Mapping[str, Any]) -> None:
    try:
        rng_states = record["rng_state"]
        state_hashes = record["state_sha256"]
        pre_hash = stable_sha256(rng_states["pre"])
        post_hash = stable_sha256(rng_states["post"])
        if pre_hash != state_hashes["pre"]["rng"]:
            raise ValueError("Strict transition pre-RNG payload is not authenticated")
        if post_hash != state_hashes["post"]["rng"]:
            raise ValueError("Strict transition post-RNG payload is not authenticated")
    except (KeyError, TypeError) as error:
        raise ValueError("Strict transition RNG record is malformed") from error
    if pre_hash != post_hash:
        raise ValueError("Strict frozen-v3 transition consumed RNG")
    if stable_sha256(capture_rng_state({})) != post_hash:
        raise ValueError(
            "Strict transition live RNG differs from its recorded post-state"
        )


def execute_frozen_v3_transition(
    package: TrajectoryPackage,
    *,
    step_index: int,
    frozen_rng_state: Mapping[str, Any],
    forward_loss: ForwardLoss,
    execution_revalidator: Callable[[], Any],
    require_rng_continuation: bool = True,
    synchronize_cuda: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one production transition under the frozen-v3 trust boundary."""
    if not callable(execution_revalidator):
        raise TypeError("Execution revalidator must be callable")
    if type(synchronize_cuda) is not bool:
        raise TypeError("CUDA synchronization control must be Boolean")

    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    run_error: Exception | None = None
    try:
        _synchronize_cuda_if_requested(synchronize_cuda)
        execution_revalidator()
        _require_trajectory_package(package)
        if tuple(package.explicit_generators):
            raise ValueError(
                "Frozen-v3 transition explicit-generator inventory must be empty"
            )
        _assert_no_hidden_package_generators(package)
        _assert_no_callback_generators(forward_loss)
        with _RejectNondeterministicSeededOps():
            result = execute_stateful_transition(
                package,
                step_index=step_index,
                frozen_rng_state=frozen_rng_state,
                forward_loss=forward_loss,
                require_rng_continuation=require_rng_continuation,
            )
        _authenticate_strict_transition_rng(result[0])
    except Exception as error:
        run_error = error
    finally:
        synchronization_error: Exception | None = None
        revalidation_error: Exception | None = None
        try:
            _synchronize_cuda_if_requested(synchronize_cuda)
        except Exception as error:
            synchronization_error = error
        try:
            execution_revalidator()
        except Exception as error:
            revalidation_error = error

        if revalidation_error is not None:
            if synchronization_error is not None:
                if run_error is not None:
                    synchronization_error.__cause__ = run_error
                raise revalidation_error from synchronization_error
            if run_error is not None:
                raise revalidation_error from run_error
            raise revalidation_error
        if synchronization_error is not None:
            if run_error is not None:
                raise synchronization_error from run_error
            raise synchronization_error

    if run_error is not None:
        raise run_error
    if result is None:
        raise RuntimeError("Strict frozen-v3 transition produced no result")
    return result


def run_persistent_trajectory(
    package: TrajectoryPackage,
    *,
    frozen_rng_states: Sequence[Mapping[str, Any]],
    forward_losses: Sequence[ForwardLoss],
) -> dict[str, Any]:
    """Run the exact 16-step schedule without resetting persistent state."""
    rng_states = tuple(frozen_rng_states)
    callbacks = tuple(forward_losses)
    if len(rng_states) != TRAJECTORY_STEP_COUNT:
        raise ValueError("Persistent trajectory requires exactly 16 RNG states")
    if len(callbacks) != TRAJECTORY_STEP_COUNT:
        raise ValueError("Persistent trajectory requires exactly 16 callbacks")

    current_state = _capture_valid_complete_state(package)
    initial_checkpoint = _clone_value(current_state)
    checkpoint_records: list[dict[str, Any]] = [
        {
            "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
            "step": 0,
            "state": initial_checkpoint,
            "hashes": _complete_state_hashes(initial_checkpoint),
        }
    ]
    transitions: list[dict[str, Any]] = []
    for step_index, (rng_state, callback) in enumerate(
        zip(rng_states, callbacks, strict=True)
    ):
        expected_continuation_hash = stable_sha256(current_state)
        record, current_state = execute_stateful_transition(
            package,
            step_index=step_index,
            frozen_rng_state=rng_state,
            forward_loss=callback,
        )
        if (
            record["state_sha256"]["continuation"]["complete"]
            != expected_continuation_hash
        ):
            raise ValueError(f"Step {step_index} does not continue its preceding state")
        if record["state_sha256"]["post"]["complete"] != stable_sha256(current_state):
            raise ValueError(f"Step {step_index} post-state hash differs")
        transitions.append(record)
        completed_steps = step_index + 1
        if completed_steps in TRAJECTORY_CHECKPOINT_STEPS:
            checkpoint = _clone_value(current_state)
            checkpoint_records.append(
                {
                    "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
                    "step": completed_steps,
                    "state": checkpoint,
                    "hashes": _complete_state_hashes(checkpoint),
                }
            )

    observed_checkpoint_steps = tuple(record["step"] for record in checkpoint_records)
    if observed_checkpoint_steps != TRAJECTORY_CHECKPOINT_STEPS:
        raise RuntimeError("Required trajectory checkpoints were not all retained")
    return {
        "step_count": TRAJECTORY_STEP_COUNT,
        "transitions": tuple(transitions),
        "checkpoints": tuple(checkpoint_records),
    }


def _live_package_tensors(
    package: TrajectoryPackage,
    label: str,
) -> tuple[tuple[str, torch.Tensor], ...]:
    tensors: list[tuple[str, torch.Tensor]] = [
        (f"{label} parameter {name}", parameter)
        for name, parameter in _named_trainable_parameters(package.model)
    ]
    tensors.extend(
        (f"{label} buffer {name}", buffer)
        for name, _persistent, buffer in _registered_buffers(package.model)
        if buffer is not None
    )
    _validate_optimizer_runtime(package.optimizer)
    for member_index, member in enumerate(package.optimizer.optimizers):
        for group_index, group in enumerate(member.param_groups):
            for key, value in group.items():
                if key != "params":
                    tensors.extend(
                        _tensor_leaves(
                            value,
                            f"{label} optimizer {member_index} "
                            f"group {group_index}.{key}",
                        )
                    )
        for parameter, state in member.state.items():
            tensors.extend(
                _tensor_leaves(
                    state,
                    f"{label} optimizer {member_index} state {id(parameter)}",
                )
            )
    return tuple(tensors)


def _reachable_mutable_objects(
    value: Any,
    label: str,
    *,
    seen: set[int],
) -> tuple[tuple[int, str], ...]:
    if isinstance(value, (torch.Tensor, torch.nn.Module)):
        return ()
    if isinstance(value, torch.optim.Optimizer):
        return ()
    if isinstance(
        value,
        (
            ModuleType,
            Path,
            type,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.FunctionType,
            types.MethodType,
        ),
    ):
        return ()
    if value is None or type(value) in {bool, int, float, complex, str, bytes}:
        return ()
    identity = id(value)
    if identity in seen:
        return ()
    seen.add(identity)
    records: list[tuple[int, str]] = []
    if isinstance(value, np.ndarray):
        records.append((identity, label))
        if value.base is not None:
            records.extend(
                _reachable_mutable_objects(
                    value.base,
                    f"{label}.base",
                    seen=seen,
                )
            )
    elif isinstance(value, (bytearray, memoryview)):
        records.append((identity, label))
        if isinstance(value, memoryview):
            records.extend(
                _reachable_mutable_objects(
                    value.obj,
                    f"{label}.obj",
                    seen=seen,
                )
            )
    elif isinstance(value, Mapping):
        records.append((identity, label))
        for key, item in value.items():
            records.extend(
                _reachable_mutable_objects(
                    key,
                    f"{label}.key[{key!r}]",
                    seen=seen,
                )
            )
            records.extend(
                _reachable_mutable_objects(
                    item,
                    f"{label}[{key!r}]",
                    seen=seen,
                )
            )
    elif isinstance(value, (deque, list, set)):
        records.append((identity, label))
        for index, item in enumerate(value):
            records.extend(
                _reachable_mutable_objects(
                    item,
                    f"{label}[{index}]",
                    seen=seen,
                )
            )
    elif isinstance(value, (tuple, frozenset)):
        for index, item in enumerate(value):
            records.extend(
                _reachable_mutable_objects(
                    item,
                    f"{label}[{index}]",
                    seen=seen,
                )
            )
    else:
        try:
            attributes = vars(value)
        except TypeError:
            return tuple(records)
        records.append((identity, label))
        records.extend(
            _reachable_mutable_objects(
                attributes,
                f"{label}.__dict__",
                seen=seen,
            )
        )
    return tuple(records)


def _package_mutable_objects(
    package: TrajectoryPackage,
    label: str,
) -> dict[int, str]:
    seen: set[int] = set()
    records = []
    for name, module in _named_modules_exact(package.model):
        records.extend(
            _reachable_mutable_objects(
                vars(module),
                f"{label}.model.{name or '<root>'}",
                seen=seen,
            )
        )
    _validate_optimizer_runtime(package.optimizer)
    records.extend(
        _reachable_mutable_objects(
            vars(package.optimizer),
            f"{label}.optimizer.combined",
            seen=seen,
        )
    )
    for index, member in enumerate(package.optimizer.optimizers):
        records.extend(
            _reachable_mutable_objects(
                vars(member),
                f"{label}.optimizer.member[{index}]",
                seen=seen,
            )
        )
    return dict(records)


def _assert_disjoint_packages(
    persistent: TrajectoryPackage,
    clone: TrajectoryPackage,
) -> None:
    _require_trajectory_package(persistent)
    _require_trajectory_package(clone)
    persistent_modules = {
        id(module) for _name, module in _named_modules_exact(persistent.model)
    }
    clone_modules = {id(module) for _name, module in _named_modules_exact(clone.model)}
    if persistent_modules.intersection(clone_modules):
        raise ValueError("Clone and persistent packages share module objects")

    persistent_optimizers = {
        id(persistent.optimizer),
        *(id(member) for member in persistent.optimizer.optimizers),
    }
    clone_optimizers = {
        id(clone.optimizer),
        *(id(member) for member in clone.optimizer.optimizers),
    }
    if persistent_optimizers.intersection(clone_optimizers):
        raise ValueError("Clone and persistent packages share optimizer objects")
    if {
        id(generator) for generator in persistent.explicit_generators.values()
    }.intersection(id(generator) for generator in clone.explicit_generators.values()):
        raise ValueError("Clone and persistent packages share explicit generators")
    persistent_mutables = _package_mutable_objects(persistent, "persistent")
    clone_mutables = _package_mutable_objects(clone, "clone")
    shared_mutables = set(persistent_mutables).intersection(clone_mutables)
    if shared_mutables:
        details = tuple(
            (
                persistent_mutables[identity],
                clone_mutables[identity],
            )
            for identity in sorted(shared_mutables)
        )
        raise ValueError(
            f"Clone and persistent packages share mutable objects: {details}"
        )

    occupied: list[tuple[str, int, int, int, str]] = []
    _claim_unique_tensor_storages(
        occupied,
        _live_package_tensors(persistent, "persistent")
        + _live_package_tensors(clone, "clone"),
    )


def _assert_package_state_disjoint(
    package: TrajectoryPackage,
    state: Mapping[str, Any],
    *,
    package_label: str,
    state_label: str,
) -> None:
    occupied: list[tuple[str, int, int, int, str]] = []
    _claim_unique_tensor_storages(
        occupied,
        _live_package_tensors(package, package_label)
        + _tensor_leaves(state, state_label),
    )


def _non_global_package_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": state["model"],
        "optimizer": state["optimizer"],
        "explicit_generators": state["rng"]["explicit_generators"],
    }


def _capture_package_bindings(package: TrajectoryPackage) -> dict[str, Any]:
    model_bindings = tuple(
        {
            "module": module,
            "parameters_object": module._parameters,
            "parameters": tuple(module._parameters.items()),
            "buffers_object": module._buffers,
            "buffers": tuple(module._buffers.items()),
            "nonpersistent_object": module._non_persistent_buffers_set,
            "nonpersistent": frozenset(module._non_persistent_buffers_set),
            "modules_object": module._modules,
            "modules": tuple(module._modules.items()),
        }
        for _name, module in _named_modules_exact(package.model)
    )
    _validate_optimizer_runtime(package.optimizer)
    members = tuple(package.optimizer.optimizers)
    optimizer_bindings = {
        "members_object": package.optimizer.optimizers,
        "members": members,
        "step_fns_object": package.optimizer.step_fns,
        "step_fns": tuple(package.optimizer.step_fns),
        "groups_object": package.optimizer.param_groups,
        "groups": tuple(package.optimizer.param_groups),
        "member_records": tuple(
            {
                "member": member,
                "groups_object": member.param_groups,
                "groups": tuple(member.param_groups),
                "state_object": member.state,
            }
            for member in members
        ),
    }
    return {
        "model": package.model,
        "optimizer": package.optimizer,
        "generators_object": package.explicit_generators,
        "generators": tuple(
            (name, package.explicit_generators[name])
            for name in sorted(package.explicit_generators)
        ),
        "model_bindings": model_bindings,
        "optimizer_bindings": optimizer_bindings,
    }


def _same_object_sequence(
    observed: Sequence[object],
    expected: Sequence[object],
) -> bool:
    return len(observed) == len(expected) and all(
        left is right for left, right in zip(observed, expected, strict=True)
    )


def _same_named_object_mapping(
    observed: Mapping[str, object],
    expected: Sequence[tuple[str, object]],
) -> bool:
    return tuple(observed) == tuple(name for name, _value in expected) and all(
        observed[name] is value for name, value in expected
    )


def _sequence_binding_matches(value: Any, owner: object, attribute: str) -> bool:
    current = getattr(owner, attribute, None)
    return (
        current is value["object"]
        and isinstance(current, Sequence)
        and _same_object_sequence(current, value["items"])
    )


def _package_bindings_match(
    package: TrajectoryPackage,
    bindings: Mapping[str, Any],
) -> bool:
    if (
        package.model is not bindings["model"]
        or package.optimizer is not bindings["optimizer"]
        or package.explicit_generators is not bindings["generators_object"]
    ):
        return False
    if not _same_named_object_mapping(
        {
            name: package.explicit_generators[name]
            for name in sorted(package.explicit_generators)
        },
        bindings["generators"],
    ):
        return False
    for record in bindings["model_bindings"]:
        module = record["module"]
        if not (
            module._parameters is record["parameters_object"]
            and _same_named_object_mapping(
                module._parameters,
                record["parameters"],
            )
            and module._buffers is record["buffers_object"]
            and _same_named_object_mapping(module._buffers, record["buffers"])
            and module._non_persistent_buffers_set is record["nonpersistent_object"]
            and frozenset(module._non_persistent_buffers_set) == record["nonpersistent"]
            and module._modules is record["modules_object"]
            and _same_named_object_mapping(module._modules, record["modules"])
        ):
            return False
    optimizer = bindings["optimizer"]
    optimizer_bindings = bindings["optimizer_bindings"]
    for value, attribute in (
        (
            {
                "object": optimizer_bindings["members_object"],
                "items": optimizer_bindings["members"],
            },
            "optimizers",
        ),
        (
            {
                "object": optimizer_bindings["step_fns_object"],
                "items": optimizer_bindings["step_fns"],
            },
            "step_fns",
        ),
        (
            {
                "object": optimizer_bindings["groups_object"],
                "items": optimizer_bindings["groups"],
            },
            "param_groups",
        ),
    ):
        if not _sequence_binding_matches(value, optimizer, attribute):
            return False
    for record in optimizer_bindings["member_records"]:
        member = record["member"]
        if not (
            member.param_groups is record["groups_object"]
            and _same_object_sequence(member.param_groups, record["groups"])
            and member.state is record["state_object"]
        ):
            return False
    return True


def _restore_sequence_binding(
    owner: object,
    attribute: str,
    value: object,
    items: Sequence[object],
) -> None:
    if isinstance(value, list):
        value[:] = items
    setattr(owner, attribute, value)


def _restore_package_bindings(
    package: TrajectoryPackage,
    bindings: Mapping[str, Any],
) -> None:
    package.model = bindings["model"]
    package.optimizer = bindings["optimizer"]
    generator_mapping = bindings["generators_object"]
    generator_mapping.clear()
    generator_mapping.update(bindings["generators"])
    package.explicit_generators = generator_mapping

    for record in bindings["model_bindings"]:
        module = record["module"]
        for attribute, object_key, items_key in (
            ("_parameters", "parameters_object", "parameters"),
            ("_buffers", "buffers_object", "buffers"),
            ("_modules", "modules_object", "modules"),
        ):
            registry = record[object_key]
            registry.clear()
            registry.update(record[items_key])
            setattr(module, attribute, registry)
        nonpersistent = record["nonpersistent_object"]
        nonpersistent.clear()
        nonpersistent.update(record["nonpersistent"])
        module._non_persistent_buffers_set = nonpersistent

    optimizer = bindings["optimizer"]
    optimizer_bindings = bindings["optimizer_bindings"]
    _restore_sequence_binding(
        optimizer,
        "optimizers",
        optimizer_bindings["members_object"],
        optimizer_bindings["members"],
    )
    _restore_sequence_binding(
        optimizer,
        "step_fns",
        optimizer_bindings["step_fns_object"],
        optimizer_bindings["step_fns"],
    )
    for record in optimizer_bindings["member_records"]:
        member = record["member"]
        _restore_sequence_binding(
            member,
            "param_groups",
            record["groups_object"],
            record["groups"],
        )
        member.state = record["state_object"]
    _restore_sequence_binding(
        optimizer,
        "param_groups",
        optimizer_bindings["groups_object"],
        optimizer_bindings["groups"],
    )


def execute_isolated_transition(
    persistent_package: TrajectoryPackage,
    *,
    checkpoint: Mapping[str, Any],
    package_factory: PackageFactory,
    step_index: int,
    frozen_rng_state: Mapping[str, Any],
    forward_loss: ForwardLoss,
    require_rng_continuation: bool = False,
    transition_executor: TransitionExecutor = execute_stateful_transition,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a transition on a disjoint clone and restore the persistent package."""
    if not callable(package_factory):
        raise TypeError("Package factory must be callable")
    if type(checkpoint) is not dict:
        raise TypeError("Isolated transition checkpoint must be an exact dict")
    if not callable(transition_executor):
        raise TypeError("Transition executor must be callable")
    persistent_before = _capture_valid_complete_state(persistent_package)
    persistent_before_hash = stable_sha256(persistent_before)
    persistent_non_global_hash = stable_sha256(
        _non_global_package_state(persistent_before)
    )
    persistent_bindings = _capture_package_bindings(persistent_package)
    checkpoint_hash = stable_sha256(checkpoint)
    _assert_package_state_disjoint(
        persistent_package,
        checkpoint,
        package_label="persistent",
        state_label="input checkpoint",
    )
    protected_checkpoint = _clone_value(checkpoint)
    if stable_sha256(protected_checkpoint) != checkpoint_hash:
        raise RuntimeError("Protected checkpoint clone differs from its input")

    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    run_error: Exception | None = None
    mutation_error: Exception | None = None
    mutation_cause: Exception | None = None
    try:
        clone = package_factory()
        if stable_sha256(checkpoint) != checkpoint_hash:
            raise ValueError("Package factory mutated the input checkpoint")
        _assert_disjoint_packages(persistent_package, clone)
        _assert_package_state_disjoint(
            clone,
            checkpoint,
            package_label="clone",
            state_label="input checkpoint",
        )
        _assert_package_state_disjoint(
            clone,
            protected_checkpoint,
            package_label="clone",
            state_label="protected checkpoint",
        )
        restore_complete_state(
            clone.model,
            clone.optimizer,
            protected_checkpoint,
            explicit_generators=clone.explicit_generators,
        )
        result = transition_executor(
            clone,
            step_index=step_index,
            frozen_rng_state=frozen_rng_state,
            forward_loss=forward_loss,
            require_rng_continuation=require_rng_continuation,
        )
    except Exception as error:
        run_error = error
    finally:
        if not _package_bindings_match(persistent_package, persistent_bindings):
            mutation_error = ValueError(
                "Isolated transition changed persistent package object bindings"
            )
            _restore_package_bindings(persistent_package, persistent_bindings)
        try:
            persistent_during = _capture_valid_complete_state(persistent_package)
            if (
                stable_sha256(_non_global_package_state(persistent_during))
                != persistent_non_global_hash
            ):
                if mutation_error is None:
                    mutation_error = ValueError(
                        "Isolated transition mutated the persistent package"
                    )
            if stable_sha256(checkpoint) != checkpoint_hash:
                if mutation_error is None:
                    mutation_error = ValueError(
                        "Isolated transition mutated its input checkpoint"
                    )
                checkpoint.clear()
                checkpoint.update(_clone_value(protected_checkpoint))
                if stable_sha256(checkpoint) != checkpoint_hash:
                    raise RuntimeError(
                        "Could not restore the isolated input checkpoint"
                    )
        except Exception as error:
            mutation_error = ValueError(
                "Could not authenticate persistent state after isolated transition"
            )
            mutation_cause = error

        try:
            if mutation_error is None:
                restore_rng_state(
                    persistent_before["rng"],
                    persistent_package.explicit_generators,
                )
            else:
                restore_complete_state(
                    persistent_package.model,
                    persistent_package.optimizer,
                    persistent_before,
                    explicit_generators=persistent_package.explicit_generators,
                )
            persistent_after = _capture_valid_complete_state(persistent_package)
        except Exception as error:
            raise RuntimeError(
                "Could not restore persistent package after isolated transition"
            ) from error
        if stable_sha256(persistent_after) != persistent_before_hash:
            raise RuntimeError(
                "Persistent package differs after isolated transition rollback"
            )

    if mutation_error is not None:
        if mutation_cause is not None:
            raise mutation_error from mutation_cause
        if run_error is not None:
            raise mutation_error from run_error
        raise mutation_error
    if run_error is not None:
        raise run_error
    if result is None:
        raise RuntimeError("Isolated transition produced no result")
    return result


def verify_transition_replay(
    persistent_package: TrajectoryPackage,
    *,
    checkpoint: Mapping[str, Any],
    expected_transition: Mapping[str, Any],
    package_factory: PackageFactory,
    step_index: int,
    frozen_rng_state: Mapping[str, Any],
    forward_loss: ForwardLoss,
    transition_executor: TransitionExecutor = execute_stateful_transition,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-run one checkpoint transition and require a byte-exact record."""
    observed, post_state = execute_isolated_transition(
        persistent_package,
        checkpoint=checkpoint,
        package_factory=package_factory,
        step_index=step_index,
        frozen_rng_state=frozen_rng_state,
        forward_loss=forward_loss,
        require_rng_continuation=True,
        transition_executor=transition_executor,
    )
    if stable_sha256(observed) != stable_sha256(expected_transition):
        raise ValueError(f"Checkpoint replay differs at step {step_index}")
    return observed, post_state


def assert_matched_rng_transition(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    """Require two matched arms to have identical authenticated RNG transitions."""
    if left.get("step_index") != right.get("step_index"):
        raise ValueError("Matched RNG records have different step indices")
    for label, record in (("left", left), ("right", right)):
        try:
            state_hashes = record["state_sha256"]
            rng_states = record["rng_state"]
            for phase in ("pre", "post"):
                if stable_sha256(rng_states[phase]) != state_hashes[phase]["rng"]:
                    raise ValueError(
                        f"{label} {phase}-step RNG payload is not authenticated"
                    )
        except (KeyError, TypeError) as error:
            raise ValueError(f"{label} matched RNG record is malformed") from error
    for phase in ("pre", "post"):
        if stable_sha256(left["rng_state"][phase]) != stable_sha256(
            right["rng_state"][phase]
        ):
            raise ValueError(f"Matched {phase}-step RNG states differ")


def _checkpoint_state_at(
    checkpoints: Sequence[Mapping[str, Any]],
    step: int,
) -> dict[str, Any]:
    matches = tuple(record for record in checkpoints if record.get("step") == step)
    if len(matches) != 1 or type(matches[0].get("state")) is not dict:
        raise ValueError(
            f"Trajectory checkpoint is absent or duplicated at step {step}"
        )
    return matches[0]["state"]


def _run_paired_microtrajectory(
    packages: Mapping[str, TrajectoryPackage],
    *,
    package_factories: Mapping[str, PackageFactory],
    case_forward_losses: Mapping[str, Sequence[ForwardLoss]],
    transition_executor: TransitionExecutor = execute_stateful_transition,
) -> dict[str, Any]:
    """Run one regime/precision without computing any deciding statistic."""
    if type(packages) is not dict or tuple(packages) != GEOMETRY_PATHS:
        raise ValueError("Paired trajectory package order differs")
    if (
        type(package_factories) is not dict
        or tuple(package_factories) != GEOMETRY_PATHS
    ):
        raise ValueError("Paired trajectory factory order differs")
    if (
        type(case_forward_losses) is not dict
        or tuple(case_forward_losses) != GEOMETRY_PATHS
    ):
        raise ValueError("Paired trajectory callback order differs")
    callbacks = {path: tuple(case_forward_losses[path]) for path in GEOMETRY_PATHS}
    if any(
        len(callbacks[path]) != len(FIXED_CASE_SPECS)
        or any(not callable(callback) for callback in callbacks[path])
        for path in GEOMETRY_PATHS
    ):
        raise ValueError("Paired trajectory case callbacks differ")
    if any(not callable(package_factories[path]) for path in GEOMETRY_PATHS):
        raise ValueError("Paired trajectory package factory is not callable")
    if not callable(transition_executor):
        raise TypeError("Transition executor must be callable")

    legacy_package, canonical_package = (packages[path] for path in GEOMETRY_PATHS)
    _assert_disjoint_packages(legacy_package, canonical_package)
    package_bindings = {
        path: _capture_package_bindings(packages[path]) for path in GEOMETRY_PATHS
    }
    current_states = {
        path: _capture_valid_complete_state(packages[path]) for path in GEOMETRY_PATHS
    }
    if stable_sha256(current_states["legacy"]) != stable_sha256(
        current_states["canonical"]
    ):
        raise ValueError("Paired trajectory t0 states are not byte-exact")

    def assert_live_packages(label: str) -> None:
        for path in GEOMETRY_PATHS:
            if not _package_bindings_match(packages[path], package_bindings[path]):
                raise ValueError(f"{label} changed {path} package object bindings")
            observed = _capture_valid_complete_state(packages[path])
            if stable_sha256(observed) != stable_sha256(current_states[path]):
                raise ValueError(f"{label} mutated the {path} persistent package")

    assert_live_packages("Paired t0 validation")
    checkpoints: dict[str, list[dict[str, Any]]] = {
        path: [
            {
                "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
                "step": 0,
                "state": _clone_value(current_states[path]),
                "hashes": _complete_state_hashes(current_states[path]),
            }
        ]
        for path in GEOMETRY_PATHS
    }
    main: dict[str, list[dict[str, Any]]] = {path: [] for path in GEOMETRY_PATHS}
    for step_index, case_index in enumerate(STEP_CASE_INDICES):
        assert_live_packages(f"Pre-step {step_index} validation")
        if stable_sha256(current_states["legacy"]["rng"]) != stable_sha256(
            current_states["canonical"]["rng"]
        ):
            raise ValueError(f"Paired pre-step RNG states differ at step {step_index}")
        frozen_rng = _clone_value(current_states["legacy"]["rng"])
        matched_records = {}
        for path in GEOMETRY_PATHS:
            package = packages[path]
            restore_rng_state(
                current_states[path]["rng"],
                package.explicit_generators,
            )
            live_pre = _capture_valid_complete_state(package)
            if stable_sha256(live_pre) != stable_sha256(current_states[path]):
                raise ValueError(
                    f"{path} live state does not continue step {step_index}"
                )
            record, post_state = transition_executor(
                package,
                step_index=step_index,
                frozen_rng_state=frozen_rng,
                forward_loss=callbacks[path][case_index],
                require_rng_continuation=True,
            )
            matched_records[path] = record
            main[path].append(
                {
                    "step": step_index,
                    "case_index": case_index,
                    "transition": record,
                    "pre_state": _clone_value(live_pre),
                    "post_state": _clone_value(post_state),
                }
            )
            current_states[path] = post_state
        assert_matched_rng_transition(
            matched_records["legacy"],
            matched_records["canonical"],
        )
        assert_live_packages(f"Post-step {step_index} validation")
        completed_steps = step_index + 1
        if completed_steps in TRAJECTORY_CHECKPOINT_STEPS:
            for path in GEOMETRY_PATHS:
                state = _clone_value(current_states[path])
                checkpoints[path].append(
                    {
                        "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
                        "step": completed_steps,
                        "state": state,
                        "hashes": _complete_state_hashes(state),
                    }
                )

    expected_checkpoint_steps = TRAJECTORY_CHECKPOINT_STEPS
    if any(
        tuple(record["step"] for record in checkpoints[path])
        != expected_checkpoint_steps
        for path in GEOMETRY_PATHS
    ):
        raise RuntimeError("Paired trajectory checkpoint schedule differs")

    replays: dict[str, list[dict[str, Any]]] = {path: [] for path in GEOMETRY_PATHS}
    for path in GEOMETRY_PATHS:
        for step in TRAJECTORY_REPLAY_STEPS:
            assert_live_packages(f"Pre-replay {path} step {step} validation")
            checkpoint = _checkpoint_state_at(checkpoints[path], step)
            expected = main[path][step]
            observed, post_state = verify_transition_replay(
                packages[path],
                checkpoint=checkpoint,
                expected_transition=expected["transition"],
                package_factory=package_factories[path],
                step_index=step,
                frozen_rng_state=checkpoint["rng"],
                forward_loss=callbacks[path][expected["case_index"]],
                transition_executor=transition_executor,
            )
            if stable_sha256(post_state) != stable_sha256(expected["post_state"]):
                raise ValueError(f"{path} replay post-state differs at step {step}")
            replays[path].append(
                {
                    "step": step,
                    "case_index": expected["case_index"],
                    "transition": observed,
                    "pre_state": _clone_value(checkpoint),
                    "post_state": _clone_value(post_state),
                }
            )
            assert_live_packages(f"Post-replay {path} step {step} validation")

    crossovers = []
    t0_records: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for checkpoint_step in CROSSOVER_STEPS:
        for case_index in range(len(FIXED_CASE_SPECS)):
            history_records = []
            for history_path in GEOMETRY_PATHS:
                checkpoint = _checkpoint_state_at(
                    checkpoints[history_path],
                    checkpoint_step,
                )
                probe_rng = _clone_value(checkpoint["rng"])
                for evaluation_path in GEOMETRY_PATHS:
                    label = (
                        f"crossover {checkpoint_step}/{case_index}/"
                        f"{history_path}/{evaluation_path}"
                    )
                    assert_live_packages(f"Pre-{label} validation")
                    observed, post_state = execute_isolated_transition(
                        packages[history_path],
                        checkpoint=checkpoint,
                        package_factory=package_factories[history_path],
                        step_index=checkpoint_step,
                        frozen_rng_state=probe_rng,
                        forward_loss=callbacks[evaluation_path][case_index],
                        require_rng_continuation=False,
                        transition_executor=transition_executor,
                    )
                    history_records.append(observed)
                    crossovers.append(
                        {
                            "checkpoint_step": checkpoint_step,
                            "case_index": case_index,
                            "history_path": history_path,
                            "evaluation_path": evaluation_path,
                            "transition": observed,
                            "pre_state": _clone_value(checkpoint),
                            "post_state": _clone_value(post_state),
                        }
                    )
                    if checkpoint_step == 0:
                        key = (case_index, GEOMETRY_PATHS.index(evaluation_path))
                        t0_records.setdefault(key, {})[history_path] = observed
                    assert_live_packages(f"Post-{label} validation")
            reference = history_records[0]
            if any(
                stable_sha256(record["rng_state"])
                != stable_sha256(reference["rng_state"])
                for record in history_records[1:]
            ):
                raise ValueError(
                    "Matched crossover RNG transitions differ at "
                    f"step {checkpoint_step}, case {case_index}"
                )

    for key, records in t0_records.items():
        if tuple(records) != GEOMETRY_PATHS or stable_sha256(
            records["legacy"]
        ) != stable_sha256(records["canonical"]):
            raise ValueError(f"t0 crossover histories differ for cell {key}")

    if (
        sum(len(records) for records in main.values()) != 2 * TRAJECTORY_STEP_COUNT
        or sum(len(records) for records in replays.values())
        != 2 * len(TRAJECTORY_REPLAY_STEPS)
        or len(crossovers)
        != (
            len(CROSSOVER_STEPS)
            * len(FIXED_CASE_SPECS)
            * len(GEOMETRY_PATHS)
            * len(GEOMETRY_PATHS)
        )
    ):
        raise RuntimeError("Paired trajectory record cardinality differs")
    assert_live_packages("Final paired trajectory validation")
    return {
        "step_count": TRAJECTORY_STEP_COUNT,
        "main": {path: tuple(main[path]) for path in GEOMETRY_PATHS},
        "checkpoints": {path: tuple(checkpoints[path]) for path in GEOMETRY_PATHS},
        "replays": {path: tuple(replays[path]) for path in GEOMETRY_PATHS},
        "crossovers": tuple(crossovers),
    }


_RAW_NPZ_KEY = re.compile(r"[a-z][a-z0-9_]*")
_RAW_NPZ_DTYPES = frozenset({"|b1", "|u1", "<i8", "<f4", "<f8"})


def _raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


class _RawNpzWriter:
    """Stream canonical arrays into one exclusive, uncompressed ZIP64 NPZ."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Raw NPZ path must be an absolute Path")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            self._stream = os.fdopen(descriptor, "w+b")
        except Exception:
            os.close(descriptor)
            raise
        try:
            self._archive = zipfile.ZipFile(
                self._stream,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
                strict_timestamps=True,
            )
        except Exception:
            self._stream.close()
            raise
        self._manifest: dict[str, dict[str, Any]] = {}
        self._closed = False

    def add(self, key: str, value: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Raw NPZ writer is closed")
        if type(key) is not str or _RAW_NPZ_KEY.fullmatch(key) is None:
            raise ValueError(f"Raw NPZ key is not canonical: {key!r}")
        if key in self._manifest:
            raise ValueError(f"Raw NPZ key is duplicated: {key}")
        if type(value) is not np.ndarray:
            raise TypeError(f"Raw NPZ value for {key} must be an exact ndarray")
        if not value.flags.c_contiguous:
            raise ValueError(f"Raw NPZ value for {key} must be C-contiguous")
        if value.dtype.str not in _RAW_NPZ_DTYPES:
            raise ValueError(
                f"Raw NPZ value for {key} has forbidden dtype {value.dtype.str}"
            )
        _require_canonical_numpy_array(value, f"Raw NPZ value for {key}")
        if np.issubdtype(value.dtype, np.floating) and not bool(
            np.isfinite(value).all()
        ):
            raise ValueError(f"Raw NPZ value for {key} is non-finite")

        info = zipfile.ZipInfo(
            filename=f"{key}.npy",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        with self._archive.open(info, mode="w", force_zip64=True) as member:
            np.lib.format.write_array(
                member,
                value,
                version=(2, 0),
                allow_pickle=False,
            )
        self._manifest[key] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
            "sha256": _raw_array_sha256(value),
        }

    def manifest(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "dtype": record["dtype"],
                "shape": list(record["shape"]),
                "nbytes": record["nbytes"],
                "sha256": record["sha256"],
            }
            for key, record in sorted(self._manifest.items())
        }

    def member_order(self) -> tuple[str, ...]:
        return tuple(self._manifest)

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self._archive.close()
            finally:
                try:
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                finally:
                    self._stream.close()
        finally:
            self._closed = True

    def __enter__(self) -> _RawNpzWriter:
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_inode(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (observed.st_dev, observed.st_ino) == identity:
        path.unlink()


def _write_validated_raw_npz_no_clobber(
    output: Path,
    populate: Callable[[_RawNpzWriter], Any],
) -> dict[str, Any]:
    """Stage, validate, and atomically publish one complete raw NPZ."""
    if not isinstance(output, Path) or not output.is_absolute() or not output.name:
        raise ValueError("Published raw NPZ path must be an absolute Path")
    if not callable(populate):
        raise TypeError("Raw NPZ populate callback must be callable")
    parent = output.parent
    if parent.resolve(strict=True) != parent or not stat.S_ISDIR(
        parent.stat(follow_symlinks=False).st_mode
    ):
        raise ValueError("Published raw NPZ parent must be a canonical directory")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite raw NPZ: {output}")

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=parent,
        )
    )
    staging_path = staging_directory / "payload.npz"
    published_identity: tuple[int, int] | None = None
    try:
        with _RawNpzWriter(staging_path) as writer:
            populate_result = populate(writer)
            manifest = writer.manifest()
            member_order = writer.member_order()
        _fsync_directory(staging_directory)
        staging_stat = staging_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(staging_stat.st_mode) or staging_stat.st_nlink != 1:
            raise ValueError("Staged raw NPZ inode differs")
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        staged_validation = _validate_raw_npz(
            staging_path,
            manifest=manifest,
            expected_order=member_order,
        )

        os.link(staging_path, output, follow_symlinks=False)
        published_identity = staging_identity
        output_stat = output.stat(follow_symlinks=False)
        linked_staging_stat = staging_path.stat(follow_symlinks=False)
        if (
            (output_stat.st_dev, output_stat.st_ino) != staging_identity
            or (linked_staging_stat.st_dev, linked_staging_stat.st_ino)
            != staging_identity
            or output_stat.st_nlink != 2
            or linked_staging_stat.st_nlink != 2
        ):
            raise OSError("Published raw NPZ hard-link identity differs")
        staging_path.unlink()
        _fsync_directory(staging_directory)
        output_stat = output.stat(follow_symlinks=False)
        if (
            output_stat.st_dev,
            output_stat.st_ino,
        ) != staging_identity or output_stat.st_nlink != 1:
            raise OSError("Published raw NPZ final inode differs")
        final_validation = _validate_raw_npz(
            output,
            manifest=manifest,
            expected_order=member_order,
        )
        staging_directory.rmdir()
        _fsync_directory(parent)
        return {
            "populate_result": populate_result,
            "manifest": manifest,
            "member_order": member_order,
            "staged_validation": staged_validation,
            "validation": final_validation,
        }
    except BaseException:
        if published_identity is not None:
            _unlink_if_inode(output, published_identity)
        raise
    finally:
        staging_path.unlink(missing_ok=True)
        try:
            staging_directory.rmdir()
        except FileNotFoundError:
            pass
        _fsync_directory(parent)


def _write_npz_identity(
    writer: _RawNpzWriter,
    *,
    attempt_id: str,
    launch_manifest_sha256: str,
) -> tuple[str, str]:
    if type(writer) is not _RawNpzWriter:
        raise TypeError("Identity writer must be an exact _RawNpzWriter")
    if type(attempt_id) is not str or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("Attempt ID is not canonical")
    if (
        type(launch_manifest_sha256) is not str
        or _SHA256_HEX.fullmatch(launch_manifest_sha256) is None
    ):
        raise ValueError("Launch-manifest SHA-256 is not canonical")
    values = (
        np.frombuffer(attempt_id.encode("ascii"), dtype=np.uint8).copy(),
        np.frombuffer(
            launch_manifest_sha256.encode("ascii"),
            dtype=np.uint8,
        ).copy(),
    )
    for key, value in zip(NPZ_IDENTITY_ARRAY_FIELDS, values, strict=True):
        writer.add(key, value)
    return NPZ_IDENTITY_ARRAY_FIELDS


def _fixed_array_schema(
    family: str,
    *,
    parameter_tensor_count: int,
    learning_rate_count: int,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if type(parameter_tensor_count) is not int or parameter_tensor_count <= 0:
        raise ValueError("Parameter tensor count must be a positive integer")
    if type(learning_rate_count) is not int or learning_rate_count <= 0:
        raise ValueError("Learning-rate count must be a positive integer")
    if family == "parameter_layout":
        return tuple(
            (field, "<i8", (parameter_tensor_count,))
            for field in PARAMETER_LAYOUT_ARRAY_FIELDS
        )
    if family == "case_control":
        return (
            ("selected_cell_ids_int64", "<i8", (PANEL_RESOLUTION,)),
            ("target_pressure_float32", "<f4", (PANEL_RESOLUTION,)),
            ("target_wss_float32", "<f4", (PANEL_RESOLUTION, 3)),
            ("target_measure_float32", "<f4", (PANEL_RESOLUTION,)),
        )

    transition_fields = (
        ("prediction_pressure_float32", "<f4", (PANEL_RESOLUTION,)),
        ("prediction_wss_float32", "<f4", (PANEL_RESOLUTION, 3)),
        ("loss_float64", "<f8", (1,)),
        ("gradient_float32", "<f4", (EXPECTED_PARAMETER_COUNT,)),
    )
    if family == "main":
        return (
            *transition_fields,
            (
                "parameter_update_float32",
                "<f4",
                (EXPECTED_PARAMETER_COUNT,),
            ),
            ("learning_rates_pre_float64", "<f8", (learning_rate_count,)),
            ("learning_rates_post_float64", "<f8", (learning_rate_count,)),
        )
    if family == "checkpoint":
        return (
            (
                "parameter_vector_float32",
                "<f4",
                (EXPECTED_PARAMETER_COUNT,),
            ),
        )
    if family == "replay":
        return (
            *transition_fields,
            (
                "parameter_update_float32",
                "<f4",
                (EXPECTED_PARAMETER_COUNT,),
            ),
        )
    if family == "crossover":
        return (
            *transition_fields,
            (
                "proposed_parameter_update_float32",
                "<f4",
                (EXPECTED_PARAMETER_COUNT,),
            ),
        )
    raise ValueError(f"Unknown fixed-array family: {family!r}")


def _validate_array_group_values(
    values: Mapping[str, np.ndarray],
    schema: Sequence[tuple[str, str, tuple[int, ...]]],
) -> None:
    if type(values) is not dict:
        raise TypeError("Array-group values must be an exact dict")
    expected_fields = tuple(record[0] for record in schema)
    if (
        not expected_fields
        or len(expected_fields) != len(set(expected_fields))
        or set(values) != set(expected_fields)
    ):
        raise ValueError("Array-group fields differ from the exact schema")

    for field, dtype, shape in schema:
        value = values[field]
        if type(value) is not np.ndarray:
            raise TypeError(f"Array-group field {field} must be an exact ndarray")
        if value.dtype.str != dtype or value.shape != shape:
            raise ValueError(
                f"Array-group field {field} dtype or shape differs: "
                f"expected {dtype} {shape}, got {value.dtype.str} {value.shape}"
            )


def _write_array_group(
    writer: _RawNpzWriter,
    *,
    prefix: str,
    values: Mapping[str, np.ndarray],
    schema: Sequence[tuple[str, str, tuple[int, ...]]],
) -> tuple[str, ...]:
    if type(writer) is not _RawNpzWriter:
        raise TypeError("Array-group writer must be an exact _RawNpzWriter")
    if type(prefix) is not str or _RAW_NPZ_KEY.fullmatch(prefix) is None:
        raise ValueError("Array-group prefix is not canonical")
    _validate_array_group_values(values, schema)

    keys = []
    for field, _dtype, _shape in schema:
        value = values[field]
        key = f"{prefix}_{field}"
        writer.add(key, value)
        keys.append(key)
    return tuple(keys)


def _validated_parameter_evidence_contract(
    expected_parameter_names: Any,
    expected_parameter_numels: Any,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if (
        type(expected_parameter_names) is not tuple
        or not expected_parameter_names
        or any(type(name) is not str or not name for name in expected_parameter_names)
        or len(expected_parameter_names) != len(set(expected_parameter_names))
        or type(expected_parameter_numels) is not tuple
        or len(expected_parameter_numels) != len(expected_parameter_names)
        or any(
            type(numel) is not int or numel <= 0 for numel in expected_parameter_numels
        )
        or sum(expected_parameter_numels) != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("Frozen parameter evidence contract differs")
    return expected_parameter_names, expected_parameter_numels


def _validated_evidence_device(value: Any, label: str) -> torch.device:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty exact string")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{label} is not a Torch device") from error
    if str(device) != value:
        raise ValueError(f"{label} is not in canonical Torch form")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError(f"{label} has an invalid CPU index")
    elif device.type == "cuda":
        if (
            type(device.index) is not int
            or not 0 <= device.index < torch.cuda.device_count()
        ):
            raise ValueError(f"{label} is not a visible CUDA device")
    else:
        raise ValueError(f"{label} uses an unsupported device type")
    return device


def _model_evidence_structure_sha256(value: Mapping[str, Any]) -> str:
    """Hash immutable model structure while excluding parameter/buffer values."""
    if type(value) is not dict:
        raise ValueError("Model evidence structure must be an exact dictionary")
    parameter_schema = tuple(
        {
            key: record[key]
            for key in (
                "name",
                "device",
                "layout",
                "stride",
                "storage_offset",
                "requires_grad",
            )
        }
        | {
            "dtype": str(record["value"].dtype),
            "shape": tuple(record["value"].shape),
        }
        for record in value["parameters"]
    )
    buffer_schema = tuple(
        {
            key: record[key]
            for key in (
                "name",
                "persistent",
                "device",
                "layout",
                "stride",
                "storage_offset",
                "requires_grad",
            )
        }
        | {
            "present": record["value"] is not None,
            "dtype": (None if record["value"] is None else str(record["value"].dtype)),
            "shape": (
                None if record["value"] is None else tuple(record["value"].shape)
            ),
        }
        for record in value["buffers"]
    )
    return stable_sha256(
        {
            "model_class": value["model_class"],
            "runtime": value["runtime"],
            "parameter_registry": value["parameter_registry"],
            "parameter_schema": parameter_schema,
            "buffer_schema": buffer_schema,
            "module_registry": value["module_registry"],
            "module_modes": value["module_modes"],
            "gradients_none": value["gradients_none"],
        }
    )


def _validate_evidence_tensor_record(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise ValueError(f"{label} tensor record must be an exact dictionary")
    _exact_keys(value, _TENSOR_RECORD_KEYS, f"{label} tensor record")
    tensor = value["value"]
    if (
        value["kind"] != _TENSOR_RECORD_KIND
        or value["layout"] != str(torch.strided)
        or type(value["stride"]) is not tuple
        or any(type(item) is not int or item < 0 for item in value["stride"])
        or type(value["storage_offset"]) is not int
        or value["storage_offset"] != 0
        or value["requires_grad"] is not False
        or type(tensor) is not torch.Tensor
    ):
        raise ValueError(f"{label} tensor record metadata differs")
    _validated_evidence_device(value["device"], f"{label} tensor device")
    _require_canonical_tensor(tensor, f"{label} tensor value")
    _require_inert_tensor(tensor, f"{label} tensor value")
    if (
        tensor.device.type != "cpu"
        or tuple(value["stride"]) != tuple(tensor.stride())
        or (
            (torch.is_floating_point(tensor) or torch.is_complex(tensor))
            and not bool(torch.isfinite(tensor).all())
        )
    ):
        raise ValueError(f"{label} tensor record payload differs")


def _validate_evidence_optimizer_value(value: Any, label: str) -> None:
    if type(value) is dict:
        if set(value) == _TENSOR_RECORD_KEYS:
            _validate_evidence_tensor_record(value, label)
            return
        if any(type(key) is not str or not key for key in value):
            raise ValueError(f"{label} mapping keys differ")
        for key, item in value.items():
            _validate_evidence_optimizer_value(item, f"{label}.{key}")
        return
    if type(value) is tuple or type(value) is list:
        for index, item in enumerate(value):
            _validate_evidence_optimizer_value(item, f"{label}[{index}]")
        return
    if type(value) is np.ndarray:
        _require_canonical_numpy_array(value, label)
        if (
            np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.complexfloating)
        ) and not bool(np.isfinite(value).all()):
            raise ValueError(f"{label} contains a non-finite array")
        return
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite float")
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return
    raise ValueError(f"{label} contains a noncanonical value")


def _validate_replayable_generator_state(
    state: Any,
    *,
    device: torch.device,
    label: str,
) -> None:
    _require_raw_rng_tensor(state, label)
    try:
        validator = torch.Generator(device=device)
        validator.set_state(state)
        round_trip = validator.get_state()
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{label} is not replayable on {device}") from error
    if not torch.equal(round_trip, state):
        raise ValueError(f"{label} is not canonical for {device}")


def _validate_evidence_rng_state(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact dictionary")
    _require_state_tree_value(value, label)
    _exact_keys(
        value,
        {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "cuda_device_count",
            "explicit_generators",
        },
        label,
    )
    if (
        type(value["cuda_device_count"]) is not int
        or value["cuda_device_count"] < 0
        or type(value["torch_cuda"]) is not tuple
        or len(value["torch_cuda"]) != value["cuda_device_count"]
        or value["cuda_device_count"] != torch.cuda.device_count()
    ):
        raise ValueError(f"{label} CUDA inventory differs")
    _validate_replayable_generator_state(
        value["torch_cpu"],
        device=torch.device("cpu"),
        label=f"{label} Torch CPU state",
    )
    if value["torch_cpu"].numel() != torch.get_rng_state().numel():
        raise ValueError(f"{label} Torch CPU state size differs")
    current_cuda_states = tuple(torch.cuda.get_rng_state_all())
    if len(current_cuda_states) != value["cuda_device_count"]:
        raise ValueError(f"{label} CUDA state inventory differs")
    for device_index, (state, current_state) in enumerate(
        zip(value["torch_cuda"], current_cuda_states, strict=True)
    ):
        _validate_replayable_generator_state(
            state,
            device=torch.device("cuda", device_index),
            label=f"{label} Torch CUDA state {device_index}",
        )
        if state.numel() != current_state.numel():
            raise ValueError(f"{label} Torch CUDA state {device_index} size differs")

    python_validator = random.Random()
    try:
        python_validator.setstate(value["python"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} Python state differs") from error
    if stable_sha256(python_validator.getstate()) != stable_sha256(value["python"]):
        raise ValueError(f"{label} Python state is not canonical")

    numpy_validator = np.random.RandomState()
    try:
        numpy_validator.set_state(value["numpy"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} NumPy state differs") from error
    if stable_sha256(numpy_validator.get_state()) != stable_sha256(value["numpy"]):
        raise ValueError(f"{label} NumPy state is not canonical")

    records = value["explicit_generators"]
    if type(records) is not tuple:
        raise ValueError(f"{label} explicit-generator inventory differs")
    names = []
    for record in records:
        if type(record) is not dict:
            raise ValueError(f"{label} explicit-generator record differs")
        _exact_keys(
            record,
            {"name", "device", "state"},
            f"{label} explicit generator",
        )
        if (
            type(record["name"]) is not str
            or not record["name"]
            or type(record["device"]) is not str
            or not record["device"]
        ):
            raise ValueError(f"{label} explicit-generator identity differs")
        device = _validated_evidence_device(
            record["device"],
            f"{label} explicit generator {record['name']} device",
        )
        _validate_replayable_generator_state(
            record["state"],
            device=device,
            label=f"{label} explicit generator {record['name']}",
        )
        names.append(record["name"])
    if tuple(names) != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError(f"{label} explicit-generator order differs")


def _validate_evidence_model_state(
    value: Any,
    *,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    label: str,
) -> tuple[torch.Tensor, tuple[tuple[Any, ...], ...]]:
    if (
        type(expected_model_structure_sha256) is not str
        or _SHA256_HEX.fullmatch(expected_model_structure_sha256) is None
    ):
        raise ValueError("Expected model-structure SHA-256 differs")
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact dictionary")
    _exact_keys(
        value,
        {
            "model_class",
            "runtime",
            "parameter_registry",
            "parameters",
            "buffers",
            "module_registry",
            "module_modes",
            "gradients_none",
        },
        label,
    )
    if type(value["model_class"]) is not str or not value["model_class"]:
        raise ValueError(f"{label} model class differs")
    expected_runtime = {
        "global_hooks_present": False,
        "module_hooks_present": False,
        "parameter_hooks_present": False,
    }
    if value["runtime"] != expected_runtime:
        raise ValueError(f"{label} runtime differs")

    registry = value["parameter_registry"]
    if type(registry) is not tuple:
        raise ValueError(f"{label} parameter registry differs")
    present_registry_names = []
    registry_names = []
    for record in registry:
        if type(record) is not dict:
            raise ValueError(f"{label} parameter registry record differs")
        _exact_keys(record, {"name", "present"}, f"{label} parameter registry")
        if (
            type(record["name"]) is not str
            or not record["name"]
            or type(record["present"]) is not bool
        ):
            raise ValueError(f"{label} parameter registry identity differs")
        registry_names.append(record["name"])
        if record["present"]:
            present_registry_names.append(record["name"])
    if (
        len(registry_names) != len(set(registry_names))
        or tuple(present_registry_names) != expected_parameter_names
    ):
        raise ValueError(f"{label} parameter registry order differs")

    parameters = value["parameters"]
    if type(parameters) is not tuple or len(parameters) != len(
        expected_parameter_names
    ):
        raise ValueError(f"{label} parameter records differ")
    vector_parts = []
    schema = []
    for index, (record, expected_name, expected_numel) in enumerate(
        zip(
            parameters,
            expected_parameter_names,
            expected_parameter_numels,
            strict=True,
        )
    ):
        if type(record) is not dict:
            raise ValueError(f"{label} parameter {index} differs")
        _exact_keys(
            record,
            {
                "name",
                "device",
                "layout",
                "stride",
                "storage_offset",
                "requires_grad",
                "value",
            },
            f"{label} parameter {index}",
        )
        tensor = record["value"]
        if (
            record["name"] != expected_name
            or record["layout"] != str(torch.strided)
            or type(record["stride"]) is not tuple
            or any(type(item) is not int or item < 0 for item in record["stride"])
            or type(record["storage_offset"]) is not int
            or record["storage_offset"] != 0
            or record["requires_grad"] is not True
            or type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or not torch.is_floating_point(tensor)
            or tensor.numel() != expected_numel
        ):
            raise ValueError(f"{label} parameter mapping differs at {expected_name}")
        _validated_evidence_device(
            record["device"],
            f"{label} parameter {expected_name} device",
        )
        _require_canonical_tensor(tensor, f"{label} parameter {expected_name}")
        _require_inert_tensor(tensor, f"{label} parameter {expected_name}")
        if tuple(record["stride"]) != tuple(tensor.stride()) or not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{label} parameter payload differs at {expected_name}")
        vector_parts.append(tensor.reshape(-1).to(dtype=torch.float32))
        schema.append(
            (
                expected_name,
                record["device"],
                record["layout"],
                record["stride"],
                record["storage_offset"],
                record["requires_grad"],
                str(tensor.dtype),
                tuple(tensor.shape),
            )
        )
    if value["gradients_none"] != expected_parameter_names:
        raise ValueError(f"{label} cleared-gradient manifest differs")

    buffers = value["buffers"]
    if type(buffers) is not tuple:
        raise ValueError(f"{label} buffer records differ")
    buffer_names = []
    for record in buffers:
        if type(record) is not dict:
            raise ValueError(f"{label} buffer record differs")
        _exact_keys(
            record,
            {
                "name",
                "persistent",
                "device",
                "layout",
                "stride",
                "storage_offset",
                "requires_grad",
                "value",
            },
            f"{label} buffer",
        )
        if (
            type(record["name"]) is not str
            or not record["name"]
            or type(record["persistent"]) is not bool
        ):
            raise ValueError(f"{label} buffer identity differs")
        buffer_names.append(record["name"])
        tensor = record["value"]
        if tensor is None:
            if any(
                record[key] is not None
                for key in (
                    "device",
                    "layout",
                    "stride",
                    "storage_offset",
                    "requires_grad",
                )
            ):
                raise ValueError(f"{label} absent buffer metadata differs")
            continue
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or record["layout"] != str(torch.strided)
            or type(record["stride"]) is not tuple
            or any(type(item) is not int or item < 0 for item in record["stride"])
            or tuple(record["stride"]) != tuple(tensor.stride())
            or type(record["storage_offset"]) is not int
            or record["storage_offset"] != 0
            or record["requires_grad"] is not False
        ):
            raise ValueError(f"{label} buffer payload differs: {record['name']}")
        _validated_evidence_device(
            record["device"],
            f"{label} buffer {record['name']} device",
        )
        _require_canonical_tensor(tensor, f"{label} buffer {record['name']}")
        _require_inert_tensor(tensor, f"{label} buffer {record['name']}")
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{label} buffer is non-finite: {record['name']}")
    if len(buffer_names) != len(set(buffer_names)):
        raise ValueError(f"{label} buffer names are nonunique")

    for field in ("module_registry", "module_modes"):
        records = value[field]
        if type(records) is not tuple:
            raise ValueError(f"{label} {field} differs")
        names = []
        for record in records:
            if type(record) is not dict:
                raise ValueError(f"{label} {field} record differs")
            if field == "module_registry":
                _exact_keys(record, {"name", "present"}, f"{label} {field}")
                valid = type(record["present"]) is bool
            else:
                _exact_keys(
                    record,
                    {"name", "module_class", "training"},
                    f"{label} {field}",
                )
                valid = (
                    type(record["module_class"]) is str
                    and bool(record["module_class"])
                    and type(record["training"]) is bool
                )
            if type(record["name"]) is not str or not valid:
                raise ValueError(f"{label} {field} identity differs")
            names.append(record["name"])
        if len(names) != len(set(names)):
            raise ValueError(f"{label} {field} names are nonunique")
        if field == "module_modes" and (
            not records
            or records[0]["name"] != ""
            or records[0]["module_class"] != value["model_class"]
        ):
            raise ValueError(f"{label} root module mode differs")

    vector = torch.cat(tuple(vector_parts)).contiguous()
    if vector.numel() != EXPECTED_PARAMETER_COUNT or not bool(
        torch.isfinite(vector).all()
    ):
        raise ValueError(f"{label} parameter vector differs")
    observed_structure_sha256 = _model_evidence_structure_sha256(value)
    if observed_structure_sha256 != expected_model_structure_sha256:
        raise ValueError(f"{label} structure differs from the frozen model contract")
    return vector, tuple(schema)


def _validate_evidence_optimizer_state(
    value: Any,
    *,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_shapes: tuple[tuple[int, ...], ...],
    label: str,
) -> tuple[dict[str, Any], ...]:
    if (
        type(expected_parameter_shapes) is not tuple
        or len(expected_parameter_shapes) != len(expected_parameter_names)
        or any(
            type(shape) is not tuple
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            for shape in expected_parameter_shapes
        )
    ):
        raise ValueError(f"{label} expected parameter shapes differ")
    expected_member_partitions = (
        tuple(
            name
            for name, shape in zip(
                expected_parameter_names,
                expected_parameter_shapes,
                strict=True,
            )
            if len(shape) == 2
        ),
        tuple(
            name
            for name, shape in zip(
                expected_parameter_names,
                expected_parameter_shapes,
                strict=True,
            )
            if len(shape) != 2
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact dictionary")
    _exact_keys(
        value,
        {"optimizer_class", "runtime", "members"},
        label,
    )
    expected_runtime = {
        "combined_optimizer_class": _EXPECTED_COMBINED_OPTIMIZER,
        "member_classes": _EXPECTED_OPTIMIZER_MEMBERS,
        "compiled": False,
        "ordinary_step_functions": True,
        "hooks_present": False,
        "global_hooks_present": False,
    }
    if (
        value["optimizer_class"] != _EXPECTED_COMBINED_OPTIMIZER
        or value["runtime"] != expected_runtime
        or type(value["members"]) is not tuple
        or len(value["members"]) != len(_EXPECTED_OPTIMIZER_MEMBERS)
    ):
        raise ValueError(f"{label} identity differs")

    learning_rates = []
    all_parameter_names = []
    for member_index, (member, expected_class) in enumerate(
        zip(value["members"], _EXPECTED_OPTIMIZER_MEMBERS, strict=True)
    ):
        if type(member) is not dict:
            raise ValueError(f"{label} member {member_index} differs")
        _exact_keys(
            member,
            {"optimizer_class", "parameter_groups", "parameter_states"},
            f"{label} member {member_index}",
        )
        groups = member["parameter_groups"]
        states = member["parameter_states"]
        if (
            member["optimizer_class"] != expected_class
            or type(groups) is not tuple
            or not groups
            or type(states) is not tuple
        ):
            raise ValueError(f"{label} member {member_index} identity differs")
        member_names = []
        for group_index, group in enumerate(groups):
            if type(group) is not dict:
                raise ValueError(
                    f"{label} member {member_index} group {group_index} differs"
                )
            _exact_keys(
                group,
                {"parameters", "options"},
                f"{label} member {member_index} group {group_index}",
            )
            parameters = group["parameters"]
            options = group["options"]
            if (
                type(parameters) is not tuple
                or not parameters
                or any(type(name) is not str or not name for name in parameters)
                or type(options) is not dict
                or tuple(options) != tuple(sorted(options))
            ):
                raise ValueError(
                    f"{label} member {member_index} group {group_index} schema differs"
                )
            for key, option in options.items():
                _validate_evidence_optimizer_value(
                    option,
                    f"{label} member {member_index} group {group_index}.{key}",
                )
            lr = options.get("lr")
            if (
                type(lr) not in {int, float}
                or isinstance(lr, bool)
                or not math.isfinite(float(lr))
                or float(lr) <= 0.0
            ):
                raise ValueError(
                    f"{label} member {member_index} group {group_index} "
                    "learning rate differs"
                )
            learning_rates.append(
                {
                    "member_index": member_index,
                    "member_class": expected_class,
                    "group_index": group_index,
                    "value_float64": float(lr),
                }
            )
            member_names.extend(parameters)
        if len(member_names) != len(set(member_names)):
            raise ValueError(f"{label} member parameter partition is nonunique")
        if tuple(member_names) != expected_member_partitions[member_index]:
            raise ValueError(
                f"{label} member parameter partition differs from the frozen "
                "2D-to-Muon/non-2D-to-AdamW contract"
            )
        if len(states) != len(member_names):
            raise ValueError(f"{label} member parameter states differ")
        for state, expected_name in zip(states, member_names, strict=True):
            if type(state) is not dict:
                raise ValueError(f"{label} parameter state differs")
            _exact_keys(
                state,
                {"parameter", "present", "values"},
                f"{label} parameter state",
            )
            if (
                state["parameter"] != expected_name
                or type(state["present"]) is not bool
                or type(state["values"]) is not dict
                or tuple(state["values"]) != tuple(sorted(state["values"]))
                or (not state["present"] and state["values"])
            ):
                raise ValueError(f"{label} parameter-state identity differs")
            for key, item in state["values"].items():
                if type(key) is not str or not key:
                    raise ValueError(f"{label} parameter-state key differs")
                _validate_evidence_optimizer_value(
                    item,
                    f"{label} parameter state {expected_name}.{key}",
                )
        all_parameter_names.extend(member_names)
    if len(all_parameter_names) != len(set(all_parameter_names)) or set(
        all_parameter_names
    ) != set(expected_parameter_names):
        raise ValueError(f"{label} parameter partition is not exact")
    return tuple(learning_rates)


def _validate_evidence_complete_state(
    state: Any,
    *,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    expected_learning_rate_count: int,
    label: str,
) -> dict[str, Any]:
    if (
        type(expected_learning_rate_count) is not int
        or expected_learning_rate_count <= 0
    ):
        raise ValueError("Expected learning-rate count differs")
    if type(state) is not dict:
        raise ValueError(f"{label} must be an exact dictionary")
    _require_state_tree_value(state, label)
    hashes = _complete_state_hashes(state)
    parameter_vector, parameter_schema = _validate_evidence_model_state(
        state["model"],
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        label=f"{label} model",
    )
    learning_rates = _validate_evidence_optimizer_state(
        state["optimizer"],
        expected_parameter_names=expected_parameter_names,
        expected_parameter_shapes=tuple(record[-1] for record in parameter_schema),
        label=f"{label} optimizer",
    )
    if len(learning_rates) != expected_learning_rate_count:
        raise ValueError(f"{label} learning-rate count differs")
    _validate_evidence_rng_state(state["rng"], f"{label} RNG state")
    _require_finite_state_value(state, label)
    return {
        "hashes": hashes,
        "parameter_vector_float32": parameter_vector,
        "parameter_schema": parameter_schema,
        "learning_rates": learning_rates,
    }


def _validate_transition_learning_rates(
    observed: Any,
    expected: tuple[dict[str, Any], ...],
    label: str,
) -> None:
    if type(observed) is not tuple or len(observed) != len(expected):
        raise ValueError(f"{label} differ")
    fields = (
        "member_index",
        "member_class",
        "group_index",
        "value_float64",
    )
    for index, (record, reference) in enumerate(zip(observed, expected, strict=True)):
        if type(record) is not dict:
            raise ValueError(f"{label} differ at record {index}")
        _exact_keys(record, set(fields), f"{label} record {index}")
        if (
            type(record["member_index"]) is not int
            or type(record["member_class"]) is not str
            or not record["member_class"]
            or type(record["group_index"]) is not int
            or type(record["value_float64"]) is not float
            or not math.isfinite(record["value_float64"])
            or record["value_float64"] <= 0.0
            or any(record[field] != reference[field] for field in fields)
        ):
            raise ValueError(f"{label} differ at record {index}")


def _validate_transition_state_bindings(
    record: Any,
    *,
    continuation_state: Any,
    pre_state: Any,
    post_state: Any,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> None:
    if type(record) is not dict:
        raise TypeError("Transition record must be an exact dict")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != TRAJECTORY_RECORD_SCHEMA_VERSION
        or type(record.get("step_index")) is not int
        or not 0 <= record["step_index"] <= TRAJECTORY_STEP_COUNT
        or record.get("parameter_names") != expected_parameter_names
        or type(record.get("parameter_count")) is not int
        or record["parameter_count"] != EXPECTED_PARAMETER_COUNT
        or record.get("parameter_order_sha256")
        != stable_sha256(expected_parameter_names)
    ):
        raise ValueError("Transition parameter or coordinate identity differs")

    continuation = _validate_evidence_complete_state(
        continuation_state,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        expected_learning_rate_count=learning_rate_count,
        label="Transition continuation state",
    )
    pre = _validate_evidence_complete_state(
        pre_state,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        expected_learning_rate_count=learning_rate_count,
        label="Transition pre-state",
    )
    post = _validate_evidence_complete_state(
        post_state,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        expected_learning_rate_count=learning_rate_count,
        label="Transition post-state",
    )
    if not (
        continuation["parameter_schema"]
        == pre["parameter_schema"]
        == post["parameter_schema"]
    ):
        raise ValueError("Transition parameter schemas differ")
    if (
        continuation["hashes"]["model"] != pre["hashes"]["model"]
        or continuation["hashes"]["optimizer"] != pre["hashes"]["optimizer"]
    ):
        raise ValueError("Transition RNG restore changed model or optimizer state")

    state_hashes = record.get("state_sha256")
    rng_states = record.get("rng_state")
    if type(state_hashes) is not dict or type(rng_states) is not dict:
        raise ValueError("Transition state bindings are malformed")
    _exact_keys(
        state_hashes,
        {"continuation", "pre", "post"},
        "Transition state hashes",
    )
    _exact_keys(rng_states, {"pre", "post"}, "Transition RNG states")
    if (
        state_hashes["continuation"] != continuation["hashes"]
        or state_hashes["pre"] != pre["hashes"]
        or state_hashes["post"] != post["hashes"]
        or stable_sha256(rng_states["pre"]) != pre["hashes"]["rng"]
        or stable_sha256(rng_states["post"]) != post["hashes"]["rng"]
        or stable_sha256(rng_states["pre"]) != stable_sha256(pre_state["rng"])
        or stable_sha256(rng_states["post"]) != stable_sha256(post_state["rng"])
    ):
        raise ValueError("Transition complete-state bindings differ")

    update = record.get("parameter_update_float32")
    expected_update = (
        post["parameter_vector_float32"] - pre["parameter_vector_float32"]
    ).contiguous()
    if (
        type(update) is not torch.Tensor
        or update.device.type != "cpu"
        or update.dtype is not torch.float32
        or not update.is_contiguous()
        or update.shape != expected_update.shape
        or not torch.equal(update, expected_update)
    ):
        raise ValueError("Transition parameter update differs from its states")
    _validate_transition_learning_rates(
        record.get("learning_rates_pre"),
        pre["learning_rates"],
        "Transition pre learning rates",
    )
    _validate_transition_learning_rates(
        record.get("learning_rates_post"),
        post["learning_rates"],
        "Transition post learning rates",
    )


def _transition_array_values(
    record: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, np.ndarray]:
    if type(record) is not dict:
        raise TypeError("Transition record must be an exact dict")
    _exact_keys(
        record,
        {
            "schema_version",
            "step_index",
            "parameter_names",
            "parameter_count",
            "parameter_order_sha256",
            "outputs",
            "loss_float64",
            "gradient_float32",
            "parameter_update_float32",
            "learning_rates_pre",
            "learning_rates_post",
            "state_sha256",
            "rng_state",
        },
        "Transition record",
    )
    if family not in {"main", "replay", "crossover"}:
        raise ValueError(f"Transition array family differs: {family!r}")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != TRAJECTORY_RECORD_SCHEMA_VERSION
        or type(record["step_index"]) is not int
        or not 0 <= record["step_index"] <= TRAJECTORY_STEP_COUNT
        or type(record["parameter_count"]) is not int
        or record["parameter_count"] != EXPECTED_PARAMETER_COUNT
        or type(record["parameter_names"]) is not tuple
        or stable_sha256(record["parameter_names"]) != record["parameter_order_sha256"]
    ):
        raise ValueError("Transition parameter identity differs")

    output_records = record["outputs"]
    if type(output_records) is not tuple or len(output_records) != 2:
        raise ValueError("Transition output records differ")
    for output in output_records:
        if type(output) is not dict:
            raise ValueError("Transition output record is malformed")
        _exact_keys(
            output,
            {"name", "value_float32"},
            "Transition output record",
        )
    if tuple(output["name"] for output in output_records) != ("pressure", "wss"):
        raise ValueError("Transition output records differ")
    output_by_name = {
        output["name"]: output.get("value_float32") for output in output_records
    }

    def tensor_array(value: Any, label: str) -> np.ndarray:
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype is not torch.float32
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{label} tensor differs")
        result = value.detach().numpy().copy(order="C")
        _require_canonical_numpy_array(result, label)
        if result.dtype.str != "<f4":
            raise ValueError(f"{label} NumPy dtype differs")
        return result

    loss = record["loss_float64"]
    if (
        type(loss) is not np.ndarray
        or loss.dtype.str != "<f8"
        or loss.shape != (1,)
        or not loss.flags.c_contiguous
        or not bool(np.isfinite(loss).all())
    ):
        raise ValueError("Transition loss array differs")
    values = {
        "prediction_pressure_float32": tensor_array(
            output_by_name["pressure"],
            "Transition pressure",
        ),
        "prediction_wss_float32": tensor_array(
            output_by_name["wss"],
            "Transition WSS",
        ),
        "loss_float64": loss.copy(order="C"),
        "gradient_float32": tensor_array(
            record["gradient_float32"],
            "Transition gradient",
        ),
    }
    update_field = (
        "proposed_parameter_update_float32"
        if family == "crossover"
        else "parameter_update_float32"
    )
    values[update_field] = tensor_array(
        record["parameter_update_float32"],
        "Transition parameter update",
    )
    if family == "main":
        for phase in ("pre", "post"):
            learning_rates = record[f"learning_rates_{phase}"]
            if type(learning_rates) is not tuple or not learning_rates:
                raise ValueError("Transition learning-rate records differ")
            values[f"learning_rates_{phase}_float64"] = np.asarray(
                [entry["value_float64"] for entry in learning_rates],
                dtype="<f8",
            )
    return values


def _write_transition_evidence(
    writer: _RawNpzWriter,
    *,
    prefix: str,
    family: str,
    record: Mapping[str, Any],
    continuation_state: Mapping[str, Any],
    pre_state: Mapping[str, Any],
    post_state: Mapping[str, Any],
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> dict[str, Any]:
    """Stream one transition and its two complete states into raw evidence."""
    expected_parameter_names, expected_parameter_numels = (
        _validated_parameter_evidence_contract(
            expected_parameter_names,
            expected_parameter_numels,
        )
    )
    _validate_transition_state_bindings(
        record,
        continuation_state=continuation_state,
        pre_state=pre_state,
        post_state=post_state,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        learning_rate_count=learning_rate_count,
    )
    values = _transition_array_values(record, family=family)
    schema = _fixed_array_schema(
        family,
        parameter_tensor_count=len(expected_parameter_names),
        learning_rate_count=learning_rate_count,
    )
    array_keys = _write_array_group(
        writer,
        prefix=prefix,
        values=values,
        schema=schema,
    )
    pre_hashes = _complete_state_hashes(pre_state)
    post_hashes = _complete_state_hashes(post_state)
    if (
        record["state_sha256"]["continuation"]
        != _complete_state_hashes(continuation_state)
        or record["state_sha256"]["pre"] != pre_hashes
        or record["state_sha256"]["post"] != post_hashes
        or stable_sha256(record["rng_state"]["pre"]) != pre_hashes["rng"]
        or stable_sha256(record["rng_state"]["post"]) != post_hashes["rng"]
    ):
        raise ValueError("Transition complete-state bindings differ")
    pre_tree = _StateTreeEncoder(
        writer,
        prefix=f"{prefix}_pre_state",
        state_kind="complete_state",
    ).encode(pre_state)
    post_tree = _StateTreeEncoder(
        writer,
        prefix=f"{prefix}_post_state",
        state_kind="complete_state",
    ).encode(post_state)
    return {
        "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
        "step_index": record["step_index"],
        "parameter_names": list(record["parameter_names"]),
        "parameter_count": record["parameter_count"],
        "parameter_order_sha256": record["parameter_order_sha256"],
        "output_names": ["pressure", "wss"],
        "array_keys": {
            field: key for (field, _dtype, _shape), key in zip(schema, array_keys)
        },
        "learning_rates_pre": list(record["learning_rates_pre"]),
        "learning_rates_post": list(record["learning_rates_post"]),
        "state_sha256": record["state_sha256"],
        "rng_state_sha256": {
            phase: stable_sha256(record["rng_state"][phase])
            for phase in ("pre", "post")
        },
        "pre_state_tree": pre_tree,
        "post_state_tree": post_tree,
        "transition_stable_sha256": stable_sha256(record),
    }


def _checkpoint_parameter_vector(
    state: Mapping[str, Any],
    *,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> np.ndarray:
    expected_parameter_names, expected_parameter_numels = (
        _validated_parameter_evidence_contract(
            expected_parameter_names,
            expected_parameter_numels,
        )
    )
    validated = _validate_evidence_complete_state(
        state,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        expected_learning_rate_count=learning_rate_count,
        label="Checkpoint state",
    )
    result = validated["parameter_vector_float32"].detach().numpy().copy(order="C")
    _require_canonical_numpy_array(result, "Checkpoint parameter vector")
    return result


def _write_checkpoint_evidence(
    writer: _RawNpzWriter,
    *,
    prefix: str,
    state: Mapping[str, Any],
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> dict[str, Any]:
    expected_parameter_names, expected_parameter_numels = (
        _validated_parameter_evidence_contract(
            expected_parameter_names,
            expected_parameter_numels,
        )
    )
    values = {
        "parameter_vector_float32": _checkpoint_parameter_vector(
            state,
            expected_parameter_names=expected_parameter_names,
            expected_parameter_numels=expected_parameter_numels,
            expected_model_structure_sha256=expected_model_structure_sha256,
            learning_rate_count=learning_rate_count,
        ),
    }
    schema = _fixed_array_schema(
        "checkpoint",
        parameter_tensor_count=len(expected_parameter_names),
        learning_rate_count=learning_rate_count,
    )
    array_keys = _write_array_group(
        writer,
        prefix=prefix,
        values=values,
        schema=schema,
    )
    state_tree = _StateTreeEncoder(
        writer,
        prefix=f"{prefix}_state",
        state_kind="complete_state",
    ).encode(state)
    return {
        "schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
        "state_sha256": _complete_state_hashes(state),
        "parameter_vector_array_key": array_keys[0],
        "state_tree": state_tree,
    }


def _validate_regime_trace_for_evidence(
    trace: Any,
    *,
    regime_index: int,
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> None:
    """Authenticate every regime identity and lineage before the first write."""
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    expected_parameter_names, expected_parameter_numels = (
        _validated_parameter_evidence_contract(
            expected_parameter_names,
            expected_parameter_numels,
        )
    )
    if type(trace) is not dict:
        raise TypeError("Paired regime trace must be an exact dict")
    _exact_keys(
        trace,
        {"step_count", "main", "checkpoints", "replays", "crossovers"},
        "Paired regime trace",
    )
    if (
        type(trace["step_count"]) is not int
        or trace["step_count"] != TRAJECTORY_STEP_COUNT
    ):
        raise ValueError("Paired regime trace step count differs")
    for family in ("main", "checkpoints", "replays"):
        value = trace[family]
        if type(value) is not dict or tuple(value) != GEOMETRY_PATHS:
            raise ValueError(f"Paired regime {family} path order differs")

    checkpoint_by_path: dict[str, dict[int, Mapping[str, Any]]] = {}
    for path in GEOMETRY_PATHS:
        records = trace["checkpoints"][path]
        if type(records) is not tuple or len(records) != len(
            TRAJECTORY_CHECKPOINT_STEPS
        ):
            raise ValueError(f"Paired regime checkpoints differ for {path}")
        checkpoint_by_path[path] = {}
        for checkpoint_index, (record, expected_step) in enumerate(
            zip(records, TRAJECTORY_CHECKPOINT_STEPS, strict=True)
        ):
            if type(record) is not dict:
                raise ValueError(f"Paired regime checkpoint differs for {path}")
            _exact_keys(
                record,
                {"schema_version", "step", "state", "hashes"},
                f"Paired regime checkpoint {path}/{checkpoint_index}",
            )
            if (
                type(record["schema_version"]) is not int
                or record["schema_version"] != TRAJECTORY_RECORD_SCHEMA_VERSION
                or type(record["step"]) is not int
                or record["step"] != expected_step
            ):
                raise ValueError(
                    f"Paired regime checkpoint identity differs for {path}"
                )
            _validate_evidence_complete_state(
                record["state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                expected_learning_rate_count=learning_rate_count,
                label=f"Paired regime checkpoint {path}/{expected_step}",
            )
            if record["hashes"] != _complete_state_hashes(record["state"]):
                raise ValueError(f"Paired regime checkpoint hashes differ for {path}")
            checkpoint_by_path[path][expected_step] = record["state"]

    if stable_sha256(checkpoint_by_path["legacy"][0]) != stable_sha256(
        checkpoint_by_path["canonical"][0]
    ):
        raise ValueError("Paired regime t0 checkpoint states differ")

    for path_index, path in enumerate(GEOMETRY_PATHS):
        records = trace["main"][path]
        if type(records) is not tuple or len(records) != TRAJECTORY_STEP_COUNT:
            raise ValueError(f"Paired regime main records differ for {path}")
        for step_index, cell in enumerate(records):
            identity = _main_record_identity(regime_index, path_index, step_index)
            if type(cell) is not dict:
                raise ValueError(f"Paired regime main cell differs for {path}")
            _exact_keys(
                cell,
                {"step", "case_index", "transition", "pre_state", "post_state"},
                f"Paired regime main cell {path}/{step_index}",
            )
            if (
                type(cell["step"]) is not int
                or cell["step"] != step_index
                or type(cell["case_index"]) is not int
                or cell["case_index"] != identity["case_index"]
                or type(cell["transition"]) is not dict
                or type(cell["transition"].get("step_index")) is not int
                or cell["transition"]["step_index"] != step_index
            ):
                raise ValueError(f"Paired regime main identity differs for {path}")
            continuation = (
                checkpoint_by_path[path][0]
                if step_index == 0
                else records[step_index - 1]["post_state"]
            )
            _validate_transition_state_bindings(
                cell["transition"],
                continuation_state=continuation,
                pre_state=cell["pre_state"],
                post_state=cell["post_state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                learning_rate_count=learning_rate_count,
            )
            _validate_array_group_values(
                _transition_array_values(cell["transition"], family="main"),
                _fixed_array_schema(
                    "main",
                    parameter_tensor_count=len(expected_parameter_names),
                    learning_rate_count=learning_rate_count,
                ),
            )
            completed_step = step_index + 1
            if completed_step in TRAJECTORY_CHECKPOINT_STEPS and stable_sha256(
                checkpoint_by_path[path][completed_step]
            ) != stable_sha256(cell["post_state"]):
                raise ValueError(
                    f"Paired regime checkpoint lineage differs for {path} "
                    f"at step {completed_step}"
                )

    for step_index in range(TRAJECTORY_STEP_COUNT):
        left = trace["main"]["legacy"][step_index]["transition"]
        right = trace["main"]["canonical"][step_index]["transition"]
        for phase in ("pre", "post"):
            if stable_sha256(left["rng_state"][phase]) != stable_sha256(
                right["rng_state"][phase]
            ):
                raise ValueError(
                    f"Paired regime matched main RNG differs at step {step_index}"
                )

    for path_index, path in enumerate(GEOMETRY_PATHS):
        records = trace["replays"][path]
        if type(records) is not tuple or len(records) != len(TRAJECTORY_REPLAY_STEPS):
            raise ValueError(f"Paired regime replay records differ for {path}")
        for replay_index, (cell, replay_step) in enumerate(
            zip(records, TRAJECTORY_REPLAY_STEPS, strict=True)
        ):
            identity = _replay_record_identity(
                regime_index,
                path_index,
                replay_index,
            )
            if type(cell) is not dict:
                raise ValueError(f"Paired regime replay cell differs for {path}")
            _exact_keys(
                cell,
                {"step", "case_index", "transition", "pre_state", "post_state"},
                f"Paired regime replay cell {path}/{replay_index}",
            )
            if (
                type(cell["step"]) is not int
                or cell["step"] != replay_step
                or type(cell["case_index"]) is not int
                or cell["case_index"] != identity["case_index"]
                or type(cell["transition"]) is not dict
                or type(cell["transition"].get("step_index")) is not int
                or cell["transition"]["step_index"] != replay_step
            ):
                raise ValueError(f"Paired regime replay identity differs for {path}")
            main_cell = trace["main"][path][replay_step]
            continuation = checkpoint_by_path[path][replay_step]
            _validate_transition_state_bindings(
                cell["transition"],
                continuation_state=continuation,
                pre_state=cell["pre_state"],
                post_state=cell["post_state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                learning_rate_count=learning_rate_count,
            )
            _validate_array_group_values(
                _transition_array_values(cell["transition"], family="replay"),
                _fixed_array_schema(
                    "replay",
                    parameter_tensor_count=len(expected_parameter_names),
                    learning_rate_count=learning_rate_count,
                ),
            )
            if (
                stable_sha256(cell["pre_state"])
                != stable_sha256(main_cell["pre_state"])
                or stable_sha256(cell["post_state"])
                != stable_sha256(main_cell["post_state"])
                or stable_sha256(cell["transition"])
                != stable_sha256(main_cell["transition"])
            ):
                raise ValueError(
                    f"Paired regime replay lineage differs for {path} "
                    f"at step {replay_step}"
                )

    crossovers = trace["crossovers"]
    expected_crossover_count = (
        len(CROSSOVER_STEPS)
        * len(FIXED_CASE_SPECS)
        * len(GEOMETRY_PATHS)
        * len(GEOMETRY_PATHS)
    )
    if type(crossovers) is not tuple or len(crossovers) != expected_crossover_count:
        raise ValueError("Paired regime crossover records differ")
    local_ordinal = 0
    matched_rng_records: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    t0_records: dict[
        tuple[int, str],
        dict[str, Mapping[str, Any]],
    ] = {}
    for checkpoint_index, checkpoint_step in enumerate(CROSSOVER_STEPS):
        for case_index in range(len(FIXED_CASE_SPECS)):
            for history_index, history_path in enumerate(GEOMETRY_PATHS):
                for evaluation_index, evaluation_path in enumerate(GEOMETRY_PATHS):
                    cell = crossovers[local_ordinal]
                    local_ordinal += 1
                    _crossover_record_identity(
                        regime_index,
                        checkpoint_index,
                        case_index,
                        history_index,
                        evaluation_index,
                    )
                    if type(cell) is not dict:
                        raise ValueError("Paired regime crossover cell differs")
                    _exact_keys(
                        cell,
                        {
                            "checkpoint_step",
                            "case_index",
                            "history_path",
                            "evaluation_path",
                            "transition",
                            "pre_state",
                            "post_state",
                        },
                        f"Paired regime crossover cell {local_ordinal - 1}",
                    )
                    if (
                        type(cell["checkpoint_step"]) is not int
                        or cell["checkpoint_step"] != checkpoint_step
                        or type(cell["case_index"]) is not int
                        or cell["case_index"] != case_index
                        or cell["history_path"] != history_path
                        or cell["evaluation_path"] != evaluation_path
                        or type(cell["transition"]) is not dict
                        or type(cell["transition"].get("step_index")) is not int
                        or cell["transition"]["step_index"] != checkpoint_step
                    ):
                        raise ValueError("Paired regime crossover identity differs")
                    continuation = checkpoint_by_path[history_path][checkpoint_step]
                    _validate_transition_state_bindings(
                        cell["transition"],
                        continuation_state=continuation,
                        pre_state=cell["pre_state"],
                        post_state=cell["post_state"],
                        expected_parameter_names=expected_parameter_names,
                        expected_parameter_numels=expected_parameter_numels,
                        expected_model_structure_sha256=(
                            expected_model_structure_sha256
                        ),
                        learning_rate_count=learning_rate_count,
                    )
                    _validate_array_group_values(
                        _transition_array_values(
                            cell["transition"],
                            family="crossover",
                        ),
                        _fixed_array_schema(
                            "crossover",
                            parameter_tensor_count=len(expected_parameter_names),
                            learning_rate_count=learning_rate_count,
                        ),
                    )
                    key = (checkpoint_step, case_index)
                    matched_rng_records.setdefault(key, []).append(
                        cell["transition"]["rng_state"]
                    )
                    if checkpoint_step == 0:
                        t0_records.setdefault(
                            (case_index, evaluation_path),
                            {},
                        )[history_path] = cell

    for key, records in matched_rng_records.items():
        reference = stable_sha256(records[0])
        if any(stable_sha256(record) != reference for record in records[1:]):
            raise ValueError(f"Paired regime crossover RNG differs for cell {key}")
    for key, records in t0_records.items():
        if tuple(records) != GEOMETRY_PATHS or any(
            stable_sha256(records[path][field])
            != stable_sha256(records["legacy"][field])
            for path in GEOMETRY_PATHS[1:]
            for field in ("transition", "pre_state", "post_state")
        ):
            raise ValueError(f"Paired regime t0 crossover histories differ for {key}")


def _write_regime_evidence(
    writer: _RawNpzWriter,
    *,
    regime_index: int,
    trace: Mapping[str, Any],
    expected_parameter_names: tuple[str, ...],
    expected_parameter_numels: tuple[int, ...],
    expected_model_structure_sha256: str,
    learning_rate_count: int,
) -> dict[str, Any]:
    """Stream every raw record for one regime in the frozen global order."""
    regime_index = _coordinate_index(
        regime_index,
        len(STATE_REGIMES),
        "Regime",
    )
    expected_parameter_names, expected_parameter_numels = (
        _validated_parameter_evidence_contract(
            expected_parameter_names,
            expected_parameter_numels,
        )
    )
    _validate_regime_trace_for_evidence(
        trace,
        regime_index=regime_index,
        expected_parameter_names=expected_parameter_names,
        expected_parameter_numels=expected_parameter_numels,
        expected_model_structure_sha256=expected_model_structure_sha256,
        learning_rate_count=learning_rate_count,
    )

    main_records = []
    checkpoint_records = []
    replay_records = []
    crossover_records = []
    for path_index, path in enumerate(GEOMETRY_PATHS):
        path_main = trace["main"][path]
        if type(path_main) is not tuple or len(path_main) != TRAJECTORY_STEP_COUNT:
            raise ValueError(f"Paired regime main records differ for {path}")
        for step_index, cell in enumerate(path_main):
            identity = _main_record_identity(regime_index, path_index, step_index)
            if (
                type(cell) is not dict
                or cell.get("step") != step_index
                or cell.get("case_index") != identity["case_index"]
                or not isinstance(cell.get("transition"), Mapping)
                or cell["transition"].get("step_index") != step_index
            ):
                raise ValueError(f"Paired regime main identity differs for {path}")
            evidence = _write_transition_evidence(
                writer,
                prefix=identity["prefix"],
                family="main",
                record=cell["transition"],
                continuation_state=(
                    trace["checkpoints"][path][0]["state"]
                    if step_index == 0
                    else path_main[step_index - 1]["post_state"]
                ),
                pre_state=cell["pre_state"],
                post_state=cell["post_state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                learning_rate_count=learning_rate_count,
            )
            main_records.append({**identity, "evidence": evidence})

        path_checkpoints = trace["checkpoints"][path]
        if type(path_checkpoints) is not tuple or len(path_checkpoints) != len(
            TRAJECTORY_CHECKPOINT_STEPS
        ):
            raise ValueError(f"Paired regime checkpoints differ for {path}")
        for checkpoint_index, cell in enumerate(path_checkpoints):
            identity = _checkpoint_record_identity(
                regime_index,
                path_index,
                checkpoint_index,
            )
            if (
                type(cell) is not dict
                or cell.get("step") != identity["state_step"]
                or cell.get("hashes") != _complete_state_hashes(cell["state"])
            ):
                raise ValueError(
                    f"Paired regime checkpoint identity differs for {path}"
                )
            evidence = _write_checkpoint_evidence(
                writer,
                prefix=identity["prefix"],
                state=cell["state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                learning_rate_count=learning_rate_count,
            )
            checkpoint_records.append({**identity, "evidence": evidence})

        path_replays = trace["replays"][path]
        if type(path_replays) is not tuple or len(path_replays) != len(
            TRAJECTORY_REPLAY_STEPS
        ):
            raise ValueError(f"Paired regime replay records differ for {path}")
        for replay_index, cell in enumerate(path_replays):
            identity = _replay_record_identity(
                regime_index,
                path_index,
                replay_index,
            )
            if (
                type(cell) is not dict
                or cell.get("step") != identity["step_from"]
                or cell.get("case_index") != identity["case_index"]
                or not isinstance(cell.get("transition"), Mapping)
                or cell["transition"].get("step_index") != identity["step_from"]
            ):
                raise ValueError(f"Paired regime replay identity differs for {path}")
            evidence = _write_transition_evidence(
                writer,
                prefix=identity["prefix"],
                family="replay",
                record=cell["transition"],
                continuation_state=_checkpoint_state_at(
                    path_checkpoints,
                    identity["step_from"],
                ),
                pre_state=cell["pre_state"],
                post_state=cell["post_state"],
                expected_parameter_names=expected_parameter_names,
                expected_parameter_numels=expected_parameter_numels,
                expected_model_structure_sha256=expected_model_structure_sha256,
                learning_rate_count=learning_rate_count,
            )
            replay_records.append({**identity, "evidence": evidence})

    crossovers = trace["crossovers"]
    expected_crossover_count = (
        len(CROSSOVER_STEPS)
        * len(FIXED_CASE_SPECS)
        * len(GEOMETRY_PATHS)
        * len(GEOMETRY_PATHS)
    )
    if type(crossovers) is not tuple or len(crossovers) != expected_crossover_count:
        raise ValueError("Paired regime crossover records differ")
    local_ordinal = 0
    for checkpoint_index, checkpoint_step in enumerate(CROSSOVER_STEPS):
        for case_index in range(len(FIXED_CASE_SPECS)):
            for history_index, history_path in enumerate(GEOMETRY_PATHS):
                for evaluation_index, evaluation_path in enumerate(GEOMETRY_PATHS):
                    cell = crossovers[local_ordinal]
                    local_ordinal += 1
                    identity = _crossover_record_identity(
                        regime_index,
                        checkpoint_index,
                        case_index,
                        history_index,
                        evaluation_index,
                    )
                    if (
                        type(cell) is not dict
                        or cell.get("checkpoint_step") != checkpoint_step
                        or cell.get("case_index") != case_index
                        or cell.get("history_path") != history_path
                        or cell.get("evaluation_path") != evaluation_path
                        or not isinstance(cell.get("transition"), Mapping)
                        or cell["transition"].get("step_index") != checkpoint_step
                    ):
                        raise ValueError("Paired regime crossover identity differs")
                    evidence = _write_transition_evidence(
                        writer,
                        prefix=identity["prefix"],
                        family="crossover",
                        record=cell["transition"],
                        continuation_state=_checkpoint_state_at(
                            trace["checkpoints"][history_path],
                            checkpoint_step,
                        ),
                        pre_state=cell["pre_state"],
                        post_state=cell["post_state"],
                        expected_parameter_names=expected_parameter_names,
                        expected_parameter_numels=expected_parameter_numels,
                        expected_model_structure_sha256=(
                            expected_model_structure_sha256
                        ),
                        learning_rate_count=learning_rate_count,
                    )
                    crossover_records.append({**identity, "evidence": evidence})

    if (
        len(main_records) != 2 * TRAJECTORY_STEP_COUNT
        or len(checkpoint_records)
        != len(GEOMETRY_PATHS) * len(TRAJECTORY_CHECKPOINT_STEPS)
        or len(replay_records) != len(GEOMETRY_PATHS) * len(TRAJECTORY_REPLAY_STEPS)
        or len(crossover_records) != expected_crossover_count
    ):
        raise RuntimeError("Streamed regime evidence cardinality differs")
    return {
        "regime_index": regime_index,
        "regime": STATE_REGIMES[regime_index],
        "main_records": tuple(main_records),
        "checkpoint_records": tuple(checkpoint_records),
        "replay_records": tuple(replay_records),
        "crossover_records": tuple(crossover_records),
    }


def _validated_raw_manifest(
    manifest: Any,
    expected_order: Any,
) -> tuple[tuple[str, np.dtype[Any], tuple[int, ...], int, str], ...]:
    if type(manifest) is not dict:
        raise ValueError("Raw NPZ manifest must be an exact dict")
    if type(expected_order) is not tuple or any(
        type(key) is not str or _RAW_NPZ_KEY.fullmatch(key) is None
        for key in expected_order
    ):
        raise ValueError("Raw NPZ member order must be a canonical tuple")
    if (
        not expected_order
        or len(expected_order) != len(set(expected_order))
        or set(expected_order) != set(manifest)
    ):
        raise ValueError("Raw NPZ member order differs from its manifest")

    result = []
    for key in expected_order:
        record = manifest[key]
        if type(record) is not dict:
            raise ValueError(f"Raw NPZ manifest entry is malformed: {key}")
        _exact_keys(
            record,
            {"dtype", "shape", "nbytes", "sha256"},
            f"Raw NPZ manifest entry {key}",
        )
        dtype_text = record["dtype"]
        if type(dtype_text) is not str or dtype_text not in _RAW_NPZ_DTYPES:
            raise ValueError(f"Raw NPZ manifest dtype differs: {key}")
        shape_value = record["shape"]
        if type(shape_value) is not list or any(
            type(size) is not int or size < 0 for size in shape_value
        ):
            raise ValueError(f"Raw NPZ manifest shape differs: {key}")
        shape = tuple(shape_value)
        dtype = np.dtype(dtype_text)
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if type(record["nbytes"]) is not int or record["nbytes"] != expected_nbytes:
            raise ValueError(f"Raw NPZ manifest byte count differs: {key}")
        if (
            type(record["sha256"]) is not str
            or _SHA256_HEX.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Raw NPZ manifest SHA-256 differs: {key}")
        result.append(
            (
                key,
                dtype,
                shape,
                expected_nbytes,
                record["sha256"],
            )
        )
    return tuple(result)


def _pread_exact(
    descriptor: int,
    size: int,
    offset: int,
    context: str,
) -> bytes:
    chunks = []
    remaining = size
    position = offset
    while remaining:
        chunk = os.pread(descriptor, remaining, position)
        if not chunk:
            raise ValueError(f"Raw NPZ is truncated while reading {context}")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _zip_directory_contract(
    descriptor: int,
    file_size: int,
    member_count: int,
) -> tuple[int, int]:
    if file_size < 22:
        raise ValueError("Raw NPZ is shorter than a ZIP end record")
    end = _pread_exact(
        descriptor,
        22,
        file_size - 22,
        "ZIP end record",
    )
    (
        signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size_32,
        directory_offset_32,
        comment_size,
    ) = struct.unpack("<4s4H2IH", end)
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or directory_disk != 0
        or comment_size != 0
    ):
        raise ValueError("Raw NPZ ZIP end record is not canonical")

    locator_offset = file_size - 42
    locator = (
        _pread_exact(
            descriptor,
            20,
            locator_offset,
            "possible ZIP64 locator",
        )
        if locator_offset >= 0
        else b""
    )
    uses_zip64_end = locator.startswith(b"PK\x06\x07")
    if not uses_zip64_end:
        if (
            disk_entries == 0xFFFF
            or total_entries == 0xFFFF
            or directory_size_32 == 0xFFFFFFFF
            or directory_offset_32 == 0xFFFFFFFF
        ):
            raise ValueError("Raw NPZ ZIP64 end records are absent")
        if disk_entries != member_count or total_entries != member_count:
            raise ValueError("Raw NPZ ZIP member count differs")
        directory_size = directory_size_32
        directory_offset = directory_offset_32
        if (
            member_count > zipfile.ZIP_FILECOUNT_LIMIT
            or directory_offset > zipfile.ZIP64_LIMIT
            or directory_size > zipfile.ZIP64_LIMIT
        ):
            raise ValueError("Raw NPZ required ZIP64 end records are absent")
        if directory_offset + directory_size != file_size - 22:
            raise ValueError("Raw NPZ ZIP directory boundary differs")
        return directory_offset, directory_size

    if file_size < 98:
        raise ValueError("Raw NPZ ZIP64 end records are truncated")
    (
        locator_signature,
        zip64_disk,
        zip64_offset,
        disk_count,
    ) = struct.unpack("<4sIQI", locator)
    if locator_signature != b"PK\x06\x07" or zip64_disk != 0 or disk_count != 1:
        raise ValueError("Raw NPZ ZIP64 locator is not canonical")
    record = _pread_exact(
        descriptor,
        56,
        zip64_offset,
        "ZIP64 end record",
    )
    (
        zip64_signature,
        record_size,
        create_version,
        extract_version,
        zip64_disk_number,
        zip64_directory_disk,
        zip64_disk_entries,
        zip64_total_entries,
        directory_size,
        directory_offset,
    ) = struct.unpack("<4sQ2H2I4Q", record)
    if (
        zip64_signature != b"PK\x06\x06"
        or record_size != 44
        or create_version != 45
        or extract_version != 45
        or zip64_disk_number != 0
        or zip64_directory_disk != 0
        or zip64_disk_entries != member_count
        or zip64_total_entries != member_count
        or directory_offset + directory_size != zip64_offset
        or zip64_offset + 56 != locator_offset
        or not (
            member_count > zipfile.ZIP_FILECOUNT_LIMIT
            or directory_offset > zipfile.ZIP64_LIMIT
            or directory_size > zipfile.ZIP64_LIMIT
        )
    ):
        raise ValueError("Raw NPZ ZIP64 end record is not canonical")

    expected_disk_entries = min(member_count, 0xFFFF)
    expected_directory_size = min(directory_size, 0xFFFFFFFF)
    expected_directory_offset = min(directory_offset, 0xFFFFFFFF)
    if (
        disk_entries != expected_disk_entries
        or total_entries != expected_disk_entries
        or directory_size_32 != expected_directory_size
        or directory_offset_32 != expected_directory_offset
    ):
        raise ValueError("Raw NPZ ZIP64 fallback end record differs")
    return directory_offset, directory_size


def _zip64_central_values(
    *,
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
    disk_number: int,
    extra: bytes,
) -> tuple[int, int, int, int]:
    required_values = []
    if uncompressed_size == 0xFFFFFFFF:
        required_values.append("uncompressed_size")
    if compressed_size == 0xFFFFFFFF:
        required_values.append("compressed_size")
    if local_offset == 0xFFFFFFFF:
        required_values.append("local_offset")
    if disk_number == 0xFFFF:
        required_values.append("disk_number")
    if not required_values:
        if extra:
            raise ValueError("Raw NPZ central ZIP extra field is unexpected")
        return compressed_size, uncompressed_size, local_offset, disk_number

    if len(extra) < 4:
        raise ValueError("Raw NPZ central ZIP64 field is truncated")
    field_id, field_size = struct.unpack_from("<HH", extra)
    if field_id != 1 or field_size != len(extra) - 4:
        raise ValueError("Raw NPZ central ZIP64 field is not canonical")
    payload = memoryview(extra)[4:]
    position = 0
    values: dict[str, int] = {}
    for name in required_values:
        width = 4 if name == "disk_number" else 8
        if position + width > len(payload):
            raise ValueError("Raw NPZ central ZIP64 values are truncated")
        format_text = "<I" if width == 4 else "<Q"
        values[name] = struct.unpack_from(format_text, payload, position)[0]
        position += width
    if position != len(payload):
        raise ValueError("Raw NPZ central ZIP64 field has extra values")
    return (
        values.get("compressed_size", compressed_size),
        values.get("uncompressed_size", uncompressed_size),
        values.get("local_offset", local_offset),
        values.get("disk_number", disk_number),
    )


def _validate_zip_central_directory(
    descriptor: int,
    *,
    directory_offset: int,
    directory_size: int,
    infos: Sequence[zipfile.ZipInfo],
) -> None:
    position = directory_offset
    for info in infos:
        header = _pread_exact(
            descriptor,
            46,
            position,
            f"central header {info.filename}",
        )
        (
            signature,
            create_version,
            extract_version,
            flag_bits,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
            comment_size,
            disk_number,
            internal_attr,
            external_attr,
            local_offset,
        ) = struct.unpack("<4s6H3I5H2I", header)
        variable = _pread_exact(
            descriptor,
            filename_size + extra_size + comment_size,
            position + 46,
            f"central metadata {info.filename}",
        )
        filename = variable[:filename_size]
        extra = variable[filename_size : filename_size + extra_size]
        comment = variable[filename_size + extra_size :]
        if (
            (uncompressed_size == 0xFFFFFFFF) != (info.file_size > zipfile.ZIP64_LIMIT)
            or (compressed_size == 0xFFFFFFFF)
            != (info.compress_size > zipfile.ZIP64_LIMIT)
            or (local_offset == 0xFFFFFFFF)
            != (info.header_offset > zipfile.ZIP64_LIMIT)
            or disk_number == 0xFFFF
        ):
            raise ValueError(
                f"Raw NPZ central ZIP64 threshold differs: {info.filename}"
            )
        (
            compressed_size,
            uncompressed_size,
            local_offset,
            disk_number,
        ) = _zip64_central_values(
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_offset=local_offset,
            disk_number=disk_number,
            extra=extra,
        )
        if (
            signature != b"PK\x01\x02"
            or create_version != (3 << 8) | 45
            or extract_version != 45
            or flag_bits != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != 0
            or modified_date != 33
            or crc != info.CRC
            or compressed_size != info.compress_size
            or uncompressed_size != info.file_size
            or filename != info.filename.encode("ascii")
            or comment != b""
            or disk_number != 0
            or internal_attr != 0
            or external_attr != 0o600 << 16
            or local_offset != info.header_offset
        ):
            raise ValueError(
                f"Raw NPZ central directory entry differs: {info.filename}"
            )
        position += 46 + len(variable)
    if position != directory_offset + directory_size:
        raise ValueError("Raw NPZ central directory size differs")


def _validate_zip_local_headers(
    descriptor: int,
    *,
    infos: Sequence[zipfile.ZipInfo],
    directory_offset: int,
) -> None:
    position = 0
    for info in infos:
        if info.header_offset != position:
            raise ValueError(f"Raw NPZ local member boundary differs: {info.filename}")
        header = _pread_exact(
            descriptor,
            30,
            position,
            f"local header {info.filename}",
        )
        (
            signature,
            extract_version,
            flag_bits,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
        ) = struct.unpack("<4s5H3I2H", header)
        variable = _pread_exact(
            descriptor,
            filename_size + extra_size,
            position + 30,
            f"local metadata {info.filename}",
        )
        filename = variable[:filename_size]
        extra = variable[filename_size:]
        if len(extra) != 20:
            raise ValueError(f"Raw NPZ local ZIP64 field differs: {info.filename}")
        field_id, field_size, zip64_size, zip64_compressed_size = struct.unpack(
            "<HHQQ",
            extra,
        )
        if (
            signature != b"PK\x03\x04"
            or extract_version != 45
            or flag_bits != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != 0
            or modified_date != 33
            or crc != info.CRC
            or compressed_size != 0xFFFFFFFF
            or uncompressed_size != 0xFFFFFFFF
            or filename != info.filename.encode("ascii")
            or field_id != 1
            or field_size != 16
            or zip64_size != info.file_size
            or zip64_compressed_size != info.compress_size
        ):
            raise ValueError(f"Raw NPZ local header differs: {info.filename}")
        position += 30 + len(variable) + info.compress_size
    if position != directory_offset:
        raise ValueError("Raw NPZ local-member region size differs")


def _canonical_npy_header(
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return stream.getvalue()


def _expected_raw_npz_layout(
    records: Sequence[tuple[str, np.dtype[Any], tuple[int, ...], int, str]],
) -> dict[str, Any]:
    local_offsets = []
    member_sizes = []
    directory_offset = 0
    for key, dtype, shape, nbytes, _sha256 in records:
        local_offsets.append(directory_offset)
        member_size = len(_canonical_npy_header(dtype, shape)) + nbytes
        member_sizes.append(member_size)
        directory_offset += 30 + len(f"{key}.npy".encode("ascii")) + 20 + member_size

    directory_size = 0
    for (key, _dtype, _shape, _nbytes, _sha256), offset, member_size in zip(
        records,
        local_offsets,
        member_sizes,
        strict=True,
    ):
        zip64_value_count = 0
        if member_size > zipfile.ZIP64_LIMIT:
            zip64_value_count += 2
        if offset > zipfile.ZIP64_LIMIT:
            zip64_value_count += 1
        extra_size = 0 if zip64_value_count == 0 else 4 + 8 * zip64_value_count
        directory_size += 46 + len(f"{key}.npy".encode("ascii")) + extra_size
    uses_zip64_end = (
        len(records) > zipfile.ZIP_FILECOUNT_LIMIT
        or directory_offset > zipfile.ZIP64_LIMIT
        or directory_size > zipfile.ZIP64_LIMIT
    )
    file_size = directory_offset + directory_size + (76 if uses_zip64_end else 0) + 22
    return {
        "local_offsets": tuple(local_offsets),
        "member_sizes": tuple(member_sizes),
        "directory_offset": directory_offset,
        "directory_size": directory_size,
        "uses_zip64_end": uses_zip64_end,
        "file_size": file_size,
    }


def _stream_validate_npy_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    nbytes: int,
    expected_sha256: str,
) -> None:
    expected_header = _canonical_npy_header(dtype, shape)
    if info.file_size != len(expected_header) + nbytes:
        raise ValueError(f"Raw NPZ NPY member byte count differs: {info.filename}")
    digest = hashlib.sha256()
    with archive.open(info, "r") as member:
        if member.read(len(expected_header)) != expected_header:
            raise ValueError(f"Raw NPZ NPY header differs: {info.filename}")
        remaining = nbytes
        while remaining:
            chunk = member.read(min(8 << 20, remaining))
            if not chunk:
                raise ValueError(f"Raw NPZ NPY payload is truncated: {info.filename}")
            if len(chunk) % dtype.itemsize:
                raise ValueError(
                    f"Raw NPZ NPY payload alignment differs: {info.filename}"
                )
            digest.update(chunk)
            if np.issubdtype(dtype, np.floating):
                if not bool(np.isfinite(np.frombuffer(chunk, dtype=dtype)).all()):
                    raise ValueError(
                        f"Raw NPZ NPY payload is non-finite: {info.filename}"
                    )
            elif dtype.str == "|b1":
                raw = np.frombuffer(chunk, dtype=np.uint8)
                if bool(((raw != 0) & (raw != 1)).any()):
                    raise ValueError(
                        f"Raw NPZ NPY Boolean payload is noncanonical: {info.filename}"
                    )
            remaining -= len(chunk)
        if member.read(1) != b"":
            raise ValueError(f"Raw NPZ NPY payload has trailing bytes: {info.filename}")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Raw NPZ NPY payload SHA-256 differs: {info.filename}")


def _file_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_small_file(
    path: Path,
    *,
    maximum_bytes: int = 8 << 20,
) -> tuple[bytes, str]:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise ValueError("Stable small-file request is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open small file safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"Small-file contract differs: {path}")
        chunks = []
        total_bytes = 0
        while chunk := os.read(descriptor, min(1 << 20, maximum_bytes + 1)):
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > maximum_bytes:
                raise ValueError(f"Small file exceeds its byte limit: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = os.stat(path, follow_symlinks=False)
    if _file_identity(before) != _file_identity(after) or _file_identity(
        before
    ) != _file_identity(path_after):
        raise ValueError(f"Small file changed while being read: {path}")
    payload = b"".join(chunks)
    return payload, hashlib.sha256(payload).hexdigest()


_LINUX_MFD_CLOEXEC = 0x0001
_LINUX_MFD_ALLOW_SEALING = 0x0002
_LINUX_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_LINUX_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_LINUX_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_LINUX_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_LINUX_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_LINUX_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_SEALED_STATIC_INPUT_SEALS = (
    _LINUX_F_SEAL_SEAL | _LINUX_F_SEAL_SHRINK | _LINUX_F_SEAL_GROW | _LINUX_F_SEAL_WRITE
)


def _linux_memfd_create(name: str) -> int:
    """Create a sealable close-on-exec Linux memory file."""
    if (
        not sys.platform.startswith("linux")
        or type(name) is not str
        or not name
        or len(name.encode("utf-8")) > 249
    ):
        raise ValueError("Linux memfd request is malformed or unsupported")
    flags = _LINUX_MFD_CLOEXEC | _LINUX_MFD_ALLOW_SEALING
    creator = getattr(os, "memfd_create", None)
    if creator is not None:
        try:
            return int(creator(name, flags))
        except OSError as error:
            raise ValueError("Could not create a sealable Linux memfd") from error

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        creator = libc.memfd_create
    except AttributeError as error:
        raise ValueError("libc does not expose memfd_create") from error
    creator.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    creator.restype = ctypes.c_int
    descriptor = int(creator(name.encode("utf-8"), flags))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise ValueError("Could not create a sealable Linux memfd") from OSError(
            error_number,
            os.strerror(error_number),
        )
    return descriptor


def _descriptor_sha256(
    descriptor: int,
    *,
    size_bytes: int,
) -> str:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(size_bytes) is not int
        or size_bytes < 0
    ):
        raise ValueError("Descriptor SHA-256 request is malformed")
    digest = hashlib.sha256()
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(descriptor, min(8 << 20, size_bytes - offset), offset)
        if not chunk:
            raise ValueError("Descriptor is truncated while hashing")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size_bytes):
        raise ValueError("Descriptor grew beyond its authenticated size")
    return digest.hexdigest()


def _write_descriptor_exact(descriptor: int, payload: bytes, offset: int) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.pwrite(descriptor, view[written:], offset + written)
        if count <= 0:
            raise OSError("Could not complete memfd write")
        written += count


def _sealed_descriptor_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
    )


class _SealedStaticInput:
    """One authenticated static input retained in a sealed anonymous file."""

    __slots__ = (
        "_closed",
        "_descriptor",
        "_identity",
        "label",
        "sha256",
        "size_bytes",
    )

    def __init__(
        self,
        *,
        descriptor: int,
        label: str,
        sha256: str,
        size_bytes: int,
    ) -> None:
        self._descriptor = descriptor
        self.label = label
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self._closed = False
        self._identity = _sealed_descriptor_identity(os.fstat(descriptor))
        self.assert_sealed()

    @property
    def proc_path(self) -> str:
        if self._closed:
            raise ValueError(f"Sealed static input is closed: {self.label}")
        return f"/proc/self/fd/{self._descriptor}"

    def assert_sealed(self) -> None:
        if self._closed:
            raise ValueError(f"Sealed static input is closed: {self.label}")
        try:
            observed = os.fstat(self._descriptor)
            seals = int(
                fcntl.fcntl(
                    self._descriptor,
                    _LINUX_F_GET_SEALS,
                )
            )
            descriptor_flags = int(fcntl.fcntl(self._descriptor, fcntl.F_GETFD))
        except OSError as error:
            raise ValueError(
                f"Sealed static-input descriptor is unavailable: {self.label}"
            ) from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or _sealed_descriptor_identity(observed) != self._identity
            or observed.st_size != self.size_bytes
        ):
            raise ValueError(
                f"Sealed static-input descriptor identity differs: {self.label}"
            )
        if seals != _SEALED_STATIC_INPUT_SEALS:
            raise ValueError(f"Static-input seal set differs: {self.label}")
        if descriptor_flags & fcntl.FD_CLOEXEC == 0:
            raise ValueError(f"Static-input descriptor is inheritable: {self.label}")
        try:
            observed_sha256 = _descriptor_sha256(
                self._descriptor,
                size_bytes=self.size_bytes,
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not authenticate sealed static input: {self.label}"
            ) from error
        if observed_sha256 != self.sha256:
            raise ValueError(f"Sealed static-input payload differs: {self.label}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._descriptor)
        except OSError:
            pass


def _seal_static_input(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> _SealedStaticInput:
    """Authenticate one single-link regular file into an immutable memfd."""
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(label) is not str
        or re.fullmatch(r"[a-z][a-z0-9_]*", label) is None
        or type(expected_sha256) is not str
        or _SHA256_HEX.fullmatch(expected_sha256) is None
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or not all(
            hasattr(os, name)
            for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK", "pread", "pwrite")
        )
    ):
        raise ValueError("Sealed static-input request is malformed or unsupported")

    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        source_descriptor = os.open(path, source_flags)
    except OSError as error:
        raise ValueError(f"Could not open static input safely: {label}") from error

    snapshot_descriptor: int | None = None
    try:
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"Static-input file contract differs: {label}")
        snapshot_descriptor = _linux_memfd_create(
            f"microtrajectory-{label}",
        )
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                source_descriptor,
                min(8 << 20, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise ValueError(f"Static input is truncated: {label}")
            digest.update(chunk)
            _write_descriptor_exact(snapshot_descriptor, chunk, offset)
            offset += len(chunk)
        after = os.fstat(source_descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"Static-input path changed while reading: {label}"
            ) from error
        if _file_identity(before) != _file_identity(after) or _file_identity(
            before
        ) != _file_identity(path_after):
            raise ValueError(f"Static input changed while being read: {label}")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Static-input SHA-256 differs: {label}")
        snapshot_stat = os.fstat(snapshot_descriptor)
        if (
            not stat.S_ISREG(snapshot_stat.st_mode)
            or snapshot_stat.st_size != before.st_size
        ):
            raise ValueError(f"Static-input snapshot size differs: {label}")
        fcntl.fcntl(
            snapshot_descriptor,
            _LINUX_F_ADD_SEALS,
            _SEALED_STATIC_INPUT_SEALS,
        )
        if (
            int(fcntl.fcntl(snapshot_descriptor, _LINUX_F_GET_SEALS))
            != _SEALED_STATIC_INPUT_SEALS
        ):
            raise ValueError(f"Could not install exact static-input seals: {label}")
        result = _SealedStaticInput(
            descriptor=snapshot_descriptor,
            label=label,
            sha256=expected_sha256,
            size_bytes=before.st_size,
        )
        snapshot_descriptor = None
        return result
    except OSError as error:
        raise ValueError(f"Could not seal static input: {label}") from error
    finally:
        os.close(source_descriptor)
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)


class _SealedStaticInputBundle:
    """Own a fixed label-to-memfd inventory for the lifetime of one attempt."""

    __slots__ = ("_closed", "_inputs")

    def __init__(self, inputs: Mapping[str, _SealedStaticInput]) -> None:
        if (
            type(inputs) is not dict
            or not inputs
            or any(
                type(label) is not str
                or re.fullmatch(r"[a-z][a-z0-9_]*", label) is None
                or type(value) is not _SealedStaticInput
                or value.label != label
                for label, value in inputs.items()
            )
        ):
            raise ValueError("Sealed static-input bundle is malformed")
        self._inputs = dict(inputs)
        self._closed = False
        self.assert_sealed()

    @classmethod
    def from_paths(
        cls,
        specifications: Mapping[str, tuple[Path, str, int]],
    ) -> _SealedStaticInputBundle:
        if (
            type(specifications) is not dict
            or not specifications
            or any(
                type(label) is not str
                or type(specification) is not tuple
                or len(specification) != 3
                for label, specification in specifications.items()
            )
        ):
            raise ValueError("Static-input snapshot specifications are malformed")
        inputs: dict[str, _SealedStaticInput] = {}
        try:
            for label, (path, expected_sha256, maximum_bytes) in specifications.items():
                inputs[label] = _seal_static_input(
                    path,
                    label=label,
                    expected_sha256=expected_sha256,
                    maximum_bytes=maximum_bytes,
                )
            return cls(inputs)
        except BaseException:
            for value in inputs.values():
                value.close()
            raise

    @property
    def proc_paths(self) -> dict[str, str]:
        self.assert_sealed()
        return {label: value.proc_path for label, value in self._inputs.items()}

    @property
    def attestation(self) -> dict[str, dict[str, Any]]:
        self.assert_sealed()
        return {
            label: {
                "sha256": value.sha256,
                "size_bytes": value.size_bytes,
                "seal_mask": _SEALED_STATIC_INPUT_SEALS,
            }
            for label, value in self._inputs.items()
        }

    def assert_sealed(self) -> None:
        if self._closed:
            raise ValueError("Sealed static-input bundle is closed")
        for value in self._inputs.values():
            value.assert_sealed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for value in self._inputs.values():
            value.close()

    def __enter__(self) -> _SealedStaticInputBundle:
        self.assert_sealed()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


def _execute_authenticated_source_module(
    path: Path,
    *,
    payload: bytes,
    module_name: str,
) -> ModuleType:
    """Execute already-authenticated source bytes without reopening their path."""
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(payload) is not bytes
        or type(module_name) is not str
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module_name) is None
    ):
        raise ValueError("Authenticated source-module request is malformed")
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(  # noqa: S102 - the caller authenticated these exact source bytes.
            compile(payload, str(path), "exec"),
            module.__dict__,
        )
        if (
            sys.modules.get(module_name) is not module
            or module.__file__ != str(path)
            or module.__package__ != module_name.rpartition(".")[0]
        ):
            raise ValueError("Authenticated source module changed its identity")
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_frozen_v3_support(
    script_path: Path,
) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    """Execute the authenticated v3 producer bytes and load its frozen helpers."""
    if not isinstance(script_path, Path) or not script_path.is_absolute():
        raise ValueError("Microtrajectory script path must be absolute")
    producer_path = script_path.with_name(FROZEN_V3_PRODUCER_FILENAME)
    payload, digest = _stable_small_file(producer_path)
    if digest != EXPECTED_FROZEN_V3_PRODUCER_SHA256:
        raise ValueError(
            "Frozen v3 producer SHA-256 differs: "
            f"expected {EXPECTED_FROZEN_V3_PRODUCER_SHA256}, got {digest}"
        )

    producer = _execute_authenticated_source_module(
        producer_path,
        payload=payload,
        module_name="frozen_microtrajectory_v3_producer",
    )
    expected_constants = {
        "RESOLUTION": PANEL_RESOLUTION,
        "FRESH_SEED": FRESH_SEED,
        "CHECKPOINT_EPOCH": CHECKPOINT_EPOCH,
        "EXPECTED_PARAMETER_COUNT": EXPECTED_PARAMETER_COUNT,
        "TARGET_CONFIG": {"pressure": "scalar", "wss": "vector"},
        "CASE_SPECS": tuple(
            (cohort_ordinal, case_id)
            for _case_index, cohort_ordinal, case_id in FIXED_CASE_SPECS
        ),
        "REGIMES": STATE_REGIMES,
        "ARMS": GEOMETRY_PATHS,
    }
    for name, expected in expected_constants.items():
        if getattr(producer, name, None) != expected:
            raise ValueError(f"Frozen v3 producer constant differs: {name}")
    required_helpers = (
        "_load_support_modules",
        "_validate_single_rank_environment",
        "_canonical_geometry_for_model",
        "_tensor_raw_equal",
        "_parameter_order_sha256",
        "_parameter_layout",
        "_global_inputs_sha256",
        "_batch_order_sha256",
        "_new_model_optimizer",
        "_prepare_case",
    )
    if any(not callable(getattr(producer, name, None)) for name in required_helpers):
        raise ValueError("Frozen v3 producer helper inventory differs")

    support_contracts = (
        (
            producer.LEGACY_SUPPORT_FILENAME,
            producer.EXPECTED_LEGACY_SUPPORT_SHA256,
            "frozen_one_step_legacy_support",
        ),
        (
            producer.RUNTIME_HELPER_FILENAME,
            producer.EXPECTED_RUNTIME_HELPER_SHA256,
            "frozen_one_step_runtime",
        ),
        (
            producer.CANONICAL_HELPER_FILENAME,
            producer.EXPECTED_CANONICAL_HELPER_SHA256,
            "frozen_one_step_canonical_support",
        ),
    )
    supports = []
    for filename, expected_sha256, module_name in support_contracts:
        expected_path = producer_path.with_name(filename)
        support_payload, observed_sha256 = _stable_small_file(expected_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"Frozen support-module SHA-256 differs: {filename}")
        supports.append(
            _execute_authenticated_source_module(
                expected_path,
                payload=support_payload,
                module_name=module_name,
            )
        )
    legacy, runtime, canonical = supports
    return producer, legacy, runtime, canonical


def _verified_parameter_layout(
    producer: ModuleType,
    model: torch.nn.Module,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Cross-check the frozen v3 layout against the live exact model registry."""
    layout, arrays = producer._parameter_layout(model)
    if type(layout) is not dict or type(arrays) is not dict:
        raise ValueError("Frozen v3 parameter layout is malformed")
    _exact_keys(
        layout,
        {
            "parameter_count",
            "parameter_names",
            "module_names",
            "ordered_parameter_names_sha256",
        },
        "Frozen v3 parameter layout",
    )
    _exact_keys(
        arrays,
        set(PARAMETER_LAYOUT_ARRAY_FIELDS),
        "Frozen v3 parameter-layout arrays",
    )

    parameters = _named_trainable_parameters(model)
    names = tuple(name for name, _parameter in parameters)
    module_names = tuple(
        dict.fromkeys(name.rpartition(".")[0] or "<root>" for name in names)
    )
    expected_starts = []
    expected_stops = []
    expected_module_indices = []
    module_lookup = {name: index for index, name in enumerate(module_names)}
    offset = 0
    for name, parameter in parameters:
        expected_starts.append(offset)
        offset += parameter.numel()
        expected_stops.append(offset)
        expected_module_indices.append(
            module_lookup[name.rpartition(".")[0] or "<root>"]
        )
    expected_layout = {
        "parameter_count": offset,
        "parameter_names": list(names),
        "module_names": list(module_names),
        "ordered_parameter_names_sha256": producer._parameter_order_sha256(names),
    }
    if offset != EXPECTED_PARAMETER_COUNT or layout != expected_layout:
        raise ValueError("Frozen v3 and live parameter layouts differ")

    expected_arrays = {
        "parameter_slice_starts_int64": np.asarray(expected_starts, dtype="<i8"),
        "parameter_slice_stops_int64": np.asarray(expected_stops, dtype="<i8"),
        "parameter_slice_module_indices_int64": np.asarray(
            expected_module_indices,
            dtype="<i8",
        ),
    }
    for name in PARAMETER_LAYOUT_ARRAY_FIELDS:
        observed = arrays[name]
        expected = expected_arrays[name]
        if type(observed) is not np.ndarray:
            raise ValueError(f"Frozen v3 parameter-layout array is malformed: {name}")
        _require_canonical_numpy_array(
            observed,
            f"Frozen v3 parameter-layout array {name}",
        )
        if (
            observed.dtype.str != "<i8"
            or observed.shape != (len(parameters),)
            or not np.array_equal(observed, expected)
        ):
            raise ValueError(f"Frozen v3 parameter-layout array differs: {name}")
    return (
        {
            **layout,
            "parameter_names": names,
            "module_names": module_names,
            "microtrajectory_parameter_order_sha256": stable_sha256(names),
        },
        {name: arrays[name].copy(order="C") for name in PARAMETER_LAYOUT_ARRAY_FIELDS},
    )


def _new_verified_trajectory_package(
    producer: ModuleType,
    runtime: Any,
    *,
    regime: str,
    checkpoint_dir: Path,
    static_input_revalidator: Callable[[], Mapping[str, str]],
) -> tuple[TrajectoryPackage, dict[str, Any]]:
    """Construct and fully attest one exact frozen-v3 trajectory package."""
    if regime not in STATE_REGIMES:
        raise ValueError(f"Unknown trajectory state regime: {regime!r}")
    if not isinstance(checkpoint_dir, Path) or not checkpoint_dir.is_absolute():
        raise ValueError("Checkpoint directory must be an absolute Path")
    if not callable(static_input_revalidator):
        raise ValueError("Static-input revalidator must be callable")
    static_before = static_input_revalidator()
    model, optimizer, loaded_epoch = producer._new_model_optimizer(
        runtime,
        regime=regime,
        checkpoint_dir=checkpoint_dir,
    )
    static_after = static_input_revalidator()
    if static_before != static_after:
        raise ValueError("Frozen static inputs changed during package construction")
    expected_epoch = None if regime == "fresh_seed42" else CHECKPOINT_EPOCH
    if loaded_epoch != expected_epoch:
        raise ValueError(
            f"Loaded epoch differs for {regime}: "
            f"expected {expected_epoch}, got {loaded_epoch}"
        )
    package = TrajectoryPackage(model, optimizer, {})
    _require_trajectory_package(package)
    if package.explicit_generators:
        raise ValueError("Frozen v3 package explicit-generator inventory is not empty")
    _assert_no_hidden_package_generators(package)
    assert_gradients_cleared(model)
    layout, layout_arrays = _verified_parameter_layout(producer, model)
    learning_rates = _ordered_learning_rates(optimizer)
    state = _capture_valid_complete_state(package)
    return package, {
        "regime": regime,
        "loaded_epoch": loaded_epoch,
        "layout": layout,
        "layout_arrays": layout_arrays,
        "learning_rates_initial": learning_rates,
        "initial_state": state,
        "initial_state_sha256": _complete_state_hashes(state),
    }


def _new_verified_package_pair(
    producer: ModuleType,
    runtime: Any,
    *,
    regime: str,
    checkpoint_dir: Path,
    static_input_revalidator: Callable[[], Mapping[str, str]],
) -> dict[str, Any]:
    """Build disjoint legacy/canonical packages with the identical complete t0."""
    legacy, legacy_attestation = _new_verified_trajectory_package(
        producer,
        runtime,
        regime=regime,
        checkpoint_dir=checkpoint_dir,
        static_input_revalidator=static_input_revalidator,
    )
    canonical, canonical_attestation = _new_verified_trajectory_package(
        producer,
        runtime,
        regime=regime,
        checkpoint_dir=checkpoint_dir,
        static_input_revalidator=static_input_revalidator,
    )
    _assert_disjoint_packages(legacy, canonical)
    if (
        stable_sha256(legacy_attestation["layout"])
        != stable_sha256(canonical_attestation["layout"])
        or stable_sha256(legacy_attestation["layout_arrays"])
        != stable_sha256(canonical_attestation["layout_arrays"])
        or stable_sha256(legacy_attestation["learning_rates_initial"])
        != stable_sha256(canonical_attestation["learning_rates_initial"])
        or stable_sha256(legacy_attestation["initial_state"])
        != stable_sha256(canonical_attestation["initial_state"])
    ):
        raise ValueError("Repeated frozen v3 package construction is not byte-exact")

    initial_state = _clone_value(legacy_attestation["initial_state"])
    for package in (legacy, canonical):
        restore_complete_state(
            package.model,
            package.optimizer,
            initial_state,
            explicit_generators=package.explicit_generators,
        )
        observed = _capture_valid_complete_state(package)
        if stable_sha256(observed) != stable_sha256(initial_state):
            raise ValueError("Trajectory package differs after exact t0 restoration")
        _assert_package_state_disjoint(
            package,
            initial_state,
            package_label="live t0 package",
            state_label="protected t0 checkpoint",
        )
    return {
        "legacy": legacy,
        "canonical": canonical,
        "initial_state": initial_state,
        "attestation": {
            key: value
            for key, value in legacy_attestation.items()
            if key != "initial_state"
        },
    }


def _preflight_recipe_import_namespace(repo_root: Path) -> dict[str, Any]:
    """Bind clean flat-import names and exact source bytes before any import."""
    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ValueError("Repository root must be an absolute Path")
    recipe_source = (
        repo_root
        / "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
    ).resolve(strict=True)
    sources = {
        path.stem: path.resolve(strict=True)
        for path in sorted(recipe_source.glob("*.py"))
        if path.name != "__init__.py"
    }
    if not sources:
        raise ValueError("Pinned recipe source inventory is empty")
    collisions = sorted(set(sources).intersection(sys.modules))
    if collisions:
        raise ImportError(
            f"Flat recipe module names were loaded before authentication: {collisions}"
        )
    source_hashes = {}
    source_payloads = {}
    for name, path in sources.items():
        payload, digest = _stable_small_file(path)
        source_hashes[name] = digest
        source_payloads[name] = (path, payload, digest)
    if not sys.path or Path(sys.path[0]).resolve() != recipe_source:
        sys.path.insert(0, str(recipe_source))
    finder = _AuthenticatedRecipeFinder(source_payloads)
    sys.meta_path.insert(0, finder)
    return {
        "source_directory": str(recipe_source),
        "source_sha256": source_hashes,
        "finder": finder,
    }


class _AuthenticatedRecipeLoader:
    """Execute one captured recipe source payload without consulting bytecode."""

    def __init__(
        self,
        finder: _AuthenticatedRecipeFinder,
        *,
        fullname: str,
        path: Path,
        payload: bytes,
        sha256: str,
    ) -> None:
        self.finder = finder
        self.fullname = fullname
        self.path = str(path)
        self.payload = payload
        self.sha256 = sha256

    def create_module(self, _spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        spec = getattr(module, "__spec__", None)
        if (
            sys.modules.get(self.fullname) is not module
            or spec is None
            or spec.loader is not self
            or spec.name != self.fullname
            or spec.origin != self.path
        ):
            raise ImportError(
                f"Authenticated recipe import identity differs: {self.fullname}"
            )
        module.__file__ = self.path
        module.__cached__ = None
        module.__package__ = ""
        exec(  # noqa: S102 - preflight captured and authenticated these exact bytes.
            compile(self.payload, self.path, "exec"),
            module.__dict__,
        )
        self.finder._register(self.fullname, module, self)


class _AuthenticatedRecipeFinder:
    """First-position finder for the frozen recipe's historical flat imports."""

    def __init__(
        self,
        sources: Mapping[str, tuple[Path, bytes, str]],
    ) -> None:
        self._sources = dict(sources)
        self._loaded: dict[
            str,
            tuple[ModuleType, _AuthenticatedRecipeLoader],
        ] = {}

    def find_spec(
        self,
        fullname: str,
        _path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        source = self._sources.get(fullname)
        if source is None:
            return None
        if target is not None or fullname in self._loaded:
            raise ImportError(f"Authenticated recipe reload is forbidden: {fullname}")
        path, payload, digest = source
        loader = _AuthenticatedRecipeLoader(
            self,
            fullname=fullname,
            path=path,
            payload=payload,
            sha256=digest,
        )
        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=str(path),
            is_package=False,
        )
        spec.has_location = True
        spec.cached = None
        return spec

    def _register(
        self,
        fullname: str,
        module: ModuleType,
        loader: _AuthenticatedRecipeLoader,
    ) -> None:
        if fullname in self._loaded or loader.finder is not self:
            raise ImportError(
                f"Authenticated recipe import was registered twice: {fullname}"
            )
        self._loaded[fullname] = (module, loader)

    def validate(self) -> dict[str, dict[str, str]]:
        if not sys.meta_path or sys.meta_path[0] is not self:
            raise ImportError("Authenticated recipe finder is not first")
        result = {}
        for name, (module, loader) in sorted(self._loaded.items()):
            path, _payload, digest = self._sources[name]
            spec = getattr(module, "__spec__", None)
            _current_payload, current_digest = _stable_small_file(path)
            if (
                sys.modules.get(name) is not module
                or type(loader) is not _AuthenticatedRecipeLoader
                or loader.finder is not self
                or spec is None
                or spec.loader is not loader
                or module.__file__ != str(path)
                or module.__cached__ is not None
                or current_digest != digest
            ):
                raise ImportError(
                    f"Authenticated recipe module changed after import: {name}"
                )
            result[name] = {
                "path": str(path),
                "sha256": digest,
            }
        return result


def _callable_source_path(value: Any, label: str) -> Path:
    target = value.__init__ if isinstance(value, type) else value
    code = getattr(target, "__code__", None)
    if code is None or type(code.co_filename) is not str:
        raise ImportError(f"Pinned recipe callable has no source code: {label}")
    return Path(code.co_filename).resolve(strict=True)


def _validate_recipe_import_provenance(
    repo_root: Path,
    *,
    preflight: Mapping[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Authenticate loaded flat modules, source bytes, and runtime callables."""
    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ValueError("Repository root must be an absolute Path")
    if type(preflight) is not dict:
        raise ValueError("Recipe import preflight is malformed")
    _exact_keys(
        preflight,
        {"source_directory", "source_sha256", "finder"},
        "Recipe import preflight",
    )
    finder = preflight["finder"]
    recipe_source = (
        repo_root
        / "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
    ).resolve(strict=True)
    if (
        preflight["source_directory"] != str(recipe_source)
        or type(preflight["source_sha256"]) is not dict
        or type(finder) is not _AuthenticatedRecipeFinder
        or not sys.path
        or Path(sys.path[0]).resolve(strict=True) != recipe_source
        or not sys.meta_path
        or sys.meta_path[0] is not finder
    ):
        raise ImportError("Pinned recipe import boundary changed")
    required_exports = {
        "collate": ("build_collate_fn",),
        "datasets": ("build_dataset", "find_normalizer"),
        "forward_kwargs": ("resolve_forward_kwargs",),
        "loss": ("LossCalculator",),
        "output_normalize": ("normalize_output_to_tensordict",),
        "utils": (
            "build_muon_optimizer",
            "get_autocast_context",
            "set_seed",
        ),
    }
    missing_before_loss = sorted(
        set(required_exports).difference({"loss"}).difference(sys.modules)
    )
    if missing_before_loss:
        raise ImportError(
            f"Historical runtime omitted required flat imports: {missing_before_loss}"
        )
    if "loss" not in sys.modules:
        __import__("loss")

    source_hashes = preflight["source_sha256"]
    observed: dict[str, dict[str, str]] = {}
    for name in sorted(source_hashes):
        module = sys.modules.get(name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str:
            raise ImportError(f"Loaded recipe module has no exact file path: {name}")
        path = Path(module_file).resolve(strict=True)
        expected = (recipe_source / f"{name}.py").resolve(strict=True)
        spec = getattr(module, "__spec__", None)
        loader = None if spec is None else spec.loader
        loader_path = getattr(loader, "path", None)
        if (
            path != expected
            or spec is None
            or spec.name != name
            or type(spec.origin) is not str
            or Path(spec.origin).resolve(strict=True) != expected
            or type(loader) is not _AuthenticatedRecipeLoader
            or loader.finder is not finder
            or type(loader_path) is not str
            or Path(loader_path).resolve(strict=True) != expected
        ):
            raise ImportError(
                f"Flat recipe import provenance differs for {name}: "
                f"expected {expected}, got {path}"
            )
        _payload, digest = _stable_small_file(expected)
        if digest != source_hashes[name]:
            raise ImportError(f"Flat recipe source bytes changed for {name}")
        observed[name] = {
            "path": str(path),
            "sha256": digest,
        }
    if not set(required_exports).issubset(observed):
        raise ImportError(
            f"Required flat recipe imports are absent: "
            f"{sorted(set(required_exports).difference(observed))}"
        )

    for module_name, exports in required_exports.items():
        module = sys.modules[module_name]
        expected_path = (recipe_source / f"{module_name}.py").resolve(strict=True)
        for export in exports:
            value = getattr(module, export, None)
            if (
                not callable(value)
                or _callable_source_path(
                    value,
                    f"{module_name}.{export}",
                )
                != expected_path
            ):
                raise ImportError(
                    f"Pinned recipe callable provenance differs: {module_name}.{export}"
                )
    if (
        runtime.normalize_output
        is not sys.modules["output_normalize"].normalize_output_to_tensordict
        or runtime.autocast_context is not sys.modules["utils"].get_autocast_context
    ):
        raise ImportError("Historical runtime holds substituted recipe callables")
    finder_observed = finder.validate()
    if finder_observed != observed:
        raise ImportError("Authenticated recipe finder attestation differs")
    return observed


def _execution_backend_attestation(runtime: Any) -> dict[str, Any]:
    """Record the CUDA and PyTorch controls that can change exact replay."""
    device = getattr(runtime, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError(f"Historical runtime requires CUDA, got {device!r}")
    current_device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(current_device)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "current_device_index": current_device,
        "visible_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": tuple(torch.cuda.get_device_capability(current_device)),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
    }


def _validate_prepared_case(
    case: Mapping[str, Any],
    *,
    case_index: int,
    cohort_ordinal: int,
    case_id: str,
    historical_start: int,
) -> dict[str, Any]:
    """Validate one frozen-v3 prepared case and return immutable raw controls."""
    if type(case) is not dict:
        raise ValueError(f"Prepared case {case_id} must be an exact dict")
    _exact_keys(
        case,
        {
            "domain",
            "bundle",
            "targets",
            "target_measure",
            "selected_ids",
            "target_pressure",
            "target_wss",
            "target_measure_array",
            "raw_source_geometry_sha256",
            "global_inputs_sha256",
            "batch_order_sha256",
        },
        f"Prepared case {case_id}",
    )
    if case["domain"] is None or case["bundle"] is None:
        raise ValueError(f"Prepared case {case_id} has no model input")
    targets = case["targets"]
    if not isinstance(targets, Mapping) or set(targets.keys()) != {
        "pressure",
        "wss",
    }:
        raise ValueError(f"Prepared case {case_id} target keys differ")
    target_tensors = {
        "target_pressure": targets["pressure"],
        "target_wss": targets["wss"],
        "target_measure": case["target_measure"],
    }
    expected_tensor_shapes = {
        "target_pressure": (PANEL_RESOLUTION,),
        "target_wss": (PANEL_RESOLUTION, 3),
        "target_measure": (PANEL_RESOLUTION,),
    }
    for name, tensor in target_tensors.items():
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype is not torch.float32
            or tuple(tensor.shape) != expected_tensor_shapes[name]
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"Prepared case {case_id} tensor differs: {name}")
    if not bool((target_tensors["target_measure"] > 0.0).all()):
        raise ValueError(f"Prepared case {case_id} target measure is not positive")

    arrays = {
        "selected_ids": case["selected_ids"],
        "target_pressure": case["target_pressure"],
        "target_wss": case["target_wss"],
        "target_measure": case["target_measure_array"],
    }
    expected_array_contracts = {
        "selected_ids": ("<i8", (PANEL_RESOLUTION,)),
        "target_pressure": ("<f4", (PANEL_RESOLUTION,)),
        "target_wss": ("<f4", (PANEL_RESOLUTION, 3)),
        "target_measure": ("<f4", (PANEL_RESOLUTION,)),
    }
    for name, value in arrays.items():
        dtype, shape = expected_array_contracts[name]
        if type(value) is not np.ndarray:
            raise ValueError(f"Prepared case {case_id} array is malformed: {name}")
        _require_canonical_numpy_array(value, f"Prepared case {case_id} {name}")
        if value.dtype.str != dtype or value.shape != shape:
            raise ValueError(f"Prepared case {case_id} array differs: {name}")
        if np.issubdtype(value.dtype, np.floating) and not bool(
            np.isfinite(value).all()
        ):
            raise ValueError(f"Prepared case {case_id} array is non-finite: {name}")
    selected_ids = arrays["selected_ids"]
    if (
        type(historical_start) is not int
        or historical_start < 0
        or not np.array_equal(
            selected_ids,
            np.arange(
                historical_start,
                historical_start + PANEL_RESOLUTION,
                dtype="<i8",
            ),
        )
    ):
        raise ValueError(
            f"Prepared case {case_id} selected IDs or historical start differ"
        )
    if not bool((arrays["target_measure"] > 0.0).all()):
        raise ValueError(f"Prepared case {case_id} target array is not positive")

    tensor_array_pairs = (
        (target_tensors["target_pressure"], arrays["target_pressure"]),
        (target_tensors["target_wss"], arrays["target_wss"]),
        (target_tensors["target_measure"], arrays["target_measure"]),
    )
    for tensor, array in tensor_array_pairs:
        observed = tensor.detach().cpu().contiguous().numpy()
        if (
            observed.dtype.str != array.dtype.str
            or observed.shape != array.shape
            or observed.tobytes(order="C") != array.tobytes(order="C")
        ):
            raise ValueError(f"Prepared case {case_id} tensor/array controls differ")

    control_hashes = {}
    for name in (
        "raw_source_geometry_sha256",
        "global_inputs_sha256",
        "batch_order_sha256",
    ):
        value = case[name]
        if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"Prepared case {case_id} control hash differs: {name}")
        control_hashes[name] = value
    return {
        "case_index": case_index,
        "cohort_ordinal": cohort_ordinal,
        "case_id": case_id,
        "historical_start": historical_start,
        **control_hashes,
        "selected_ids_sha256": _raw_array_sha256(arrays["selected_ids"]),
        "target_pressure_sha256": _raw_array_sha256(arrays["target_pressure"]),
        "target_wss_sha256": _raw_array_sha256(arrays["target_wss"]),
        "target_measure_sha256": _raw_array_sha256(arrays["target_measure"]),
    }


def _snapshot_model_input(value: Any, label: str) -> Any:
    """Clone one model-visible input into a canonical CPU authentication tree."""
    if isinstance(value, torch.Tensor):
        if (
            type(value) is not torch.Tensor
            or value.requires_grad
            or value.grad is not None
            or value.is_conj()
            or value.is_neg()
        ):
            raise ValueError(f"{label} is not an inert exact Tensor")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{label} is non-finite")
        _require_canonical_torch_bool(value, label)
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "storage_offset": value.storage_offset(),
            "value": value.detach().cpu().contiguous().clone(),
        }
    if isinstance(value, np.ndarray):
        _require_canonical_numpy_array(value, label)
        if np.issubdtype(value.dtype, np.floating) and not bool(
            np.isfinite(value).all()
        ):
            raise ValueError(f"{label} is non-finite")
        return {
            "kind": "ndarray",
            "dtype": value.dtype.str,
            "shape": tuple(value.shape),
            "value": value.copy(order="C"),
        }
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if len(keys) != len(set(keys)) or any(type(key) is not str for key in keys):
            raise ValueError(f"{label} mapping keys are not unique exact strings")
        batch_size = getattr(value, "batch_size", None)
        device = getattr(value, "device", None)
        return {
            "kind": "mapping",
            "class": _qualified_class(value),
            "keys": keys,
            "batch_size": None if batch_size is None else tuple(batch_size),
            "device": None if device is None else str(device),
            "items": tuple(
                (key, _snapshot_model_input(value[key], f"{label}.{key}"))
                for key in keys
            ),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": tuple(
                _snapshot_model_input(item, f"{label}[{index}]")
                for index, item in enumerate(value)
            ),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": tuple(
                _snapshot_model_input(item, f"{label}[{index}]")
                for index, item in enumerate(value)
            ),
        }
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{label} has unsupported input type {type(value).__name__}")


def _mesh_input_snapshot(mesh: Any, label: str) -> dict[str, Any]:
    required = ("points", "cells", "point_data", "cell_data", "global_data")
    if any(not hasattr(mesh, name) for name in required):
        raise ValueError(f"{label} does not satisfy the Mesh input contract")
    return {
        "class": _qualified_class(mesh),
        "points": _snapshot_model_input(mesh.points, f"{label}.points"),
        "cells": _snapshot_model_input(mesh.cells, f"{label}.cells"),
        "point_data": _snapshot_model_input(
            mesh.point_data,
            f"{label}.point_data",
        ),
        "cell_data": _snapshot_model_input(mesh.cell_data, f"{label}.cell_data"),
        "global_data": _snapshot_model_input(
            mesh.global_data,
            f"{label}.global_data",
        ),
    }


def _case_execution_snapshot(case: Mapping[str, Any]) -> dict[str, Any]:
    """Capture all model-visible prepared-case bytes except benign mesh caches."""
    domain = case["domain"]
    if any(
        not hasattr(domain, name) for name in ("interior", "boundaries", "global_data")
    ):
        raise ValueError("Prepared case domain contract differs")
    boundary_names = tuple(domain.boundaries.keys())
    if (
        not boundary_names
        or len(boundary_names) != len(set(boundary_names))
        or any(type(name) is not str for name in boundary_names)
    ):
        raise ValueError("Prepared case boundary names differ")
    bundle = case["bundle"]
    bundle_fields = (
        "points",
        "cells",
        "centroids",
        "areas",
        "normals",
        "physical_center",
        "physical_length",
        "model_reference_length",
    )
    if any(not hasattr(bundle, name) for name in bundle_fields):
        raise ValueError("Prepared canonical bundle contract differs")
    return {
        "domain_class": _qualified_class(domain),
        "interior": _mesh_input_snapshot(domain.interior, "domain.interior"),
        "boundary_names": boundary_names,
        "boundaries": tuple(
            (
                name,
                _mesh_input_snapshot(
                    domain.boundaries[name],
                    f"domain.boundaries.{name}",
                ),
            )
            for name in boundary_names
        ),
        "domain_global_data": _snapshot_model_input(
            domain.global_data,
            "domain.global_data",
        ),
        "bundle_class": _qualified_class(bundle),
        "bundle": tuple(
            (
                name,
                _snapshot_model_input(
                    getattr(bundle, name),
                    f"canonical_bundle.{name}",
                ),
            )
            for name in bundle_fields
        ),
        "targets": _snapshot_model_input(case["targets"], "case.targets"),
        "target_measure": _snapshot_model_input(
            case["target_measure"],
            "case.target_measure",
        ),
        "selected_ids": _snapshot_model_input(
            case["selected_ids"],
            "case.selected_ids",
        ),
        "target_pressure": _snapshot_model_input(
            case["target_pressure"],
            "case.target_pressure",
        ),
        "target_wss": _snapshot_model_input(
            case["target_wss"],
            "case.target_wss",
        ),
        "target_measure_array": _snapshot_model_input(
            case["target_measure_array"],
            "case.target_measure_array",
        ),
        "raw_source_geometry_sha256": case["raw_source_geometry_sha256"],
        "global_inputs_sha256": case["global_inputs_sha256"],
        "batch_order_sha256": case["batch_order_sha256"],
    }


def _capture_case_bindings(case: Mapping[str, Any]) -> dict[str, Any]:
    domain = case["domain"]
    boundary_names = tuple(domain.boundaries.keys())
    return {
        "case": case,
        "items": tuple((key, case[key]) for key in case),
        "domain": domain,
        "interior": domain.interior,
        "boundaries": domain.boundaries,
        "boundary_items": tuple(
            (name, domain.boundaries[name]) for name in boundary_names
        ),
        "global_data": domain.global_data,
        "bundle": case["bundle"],
        "targets": case["targets"],
        "target_items": tuple(
            (name, case["targets"][name]) for name in case["targets"].keys()
        ),
    }


def _case_bindings_match(
    case: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> bool:
    if case is not bindings["case"]:
        return False
    if not _same_named_object_mapping(case, bindings["items"]):
        return False
    domain = case["domain"]
    if (
        domain is not bindings["domain"]
        or domain.interior is not bindings["interior"]
        or domain.boundaries is not bindings["boundaries"]
        or domain.global_data is not bindings["global_data"]
        or case["bundle"] is not bindings["bundle"]
        or case["targets"] is not bindings["targets"]
    ):
        return False
    if not _same_named_object_mapping(
        {name: domain.boundaries[name] for name in domain.boundaries.keys()},
        bindings["boundary_items"],
    ):
        return False
    return _same_named_object_mapping(
        {name: case["targets"][name] for name in case["targets"].keys()},
        bindings["target_items"],
    )


def _assert_case_unchanged(
    case: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    expected_sha256: str,
) -> None:
    if not _case_bindings_match(case, bindings):
        raise ValueError("Forward/loss callback changed prepared-case bindings")
    if stable_sha256(_case_execution_snapshot(case)) != expected_sha256:
        raise ValueError("Forward/loss callback mutated prepared-case input bytes")


def _validate_transition_prediction(
    prediction: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not isinstance(prediction, Mapping) or tuple(prediction.keys()) != (
        "pressure",
        "wss",
    ):
        raise ValueError("Transition prediction keys or order differ")
    expected_shapes = {
        "pressure": (PANEL_RESOLUTION,),
        "wss": (PANEL_RESOLUTION, 3),
    }
    result = {}
    for name, shape in expected_shapes.items():
        value = prediction[name]
        if (
            type(value) is not torch.Tensor
            or value.dtype is not torch.float32
            or value.device != device
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Transition prediction differs: {name}")
        result[name] = value
    return result


def _new_v3_forward_loss(
    producer: ModuleType,
    runtime: Any,
    case: dict[str, Any],
    *,
    case_index: int,
    cohort_ordinal: int,
    case_id: str,
    historical_start: int,
    precision: str,
    geometry_path: str,
) -> ForwardLoss:
    """Create the exact frozen-v3 forward/loss slice without update ownership."""
    if precision not in {"bfloat16", "float32"}:
        raise ValueError(f"Unknown transition precision: {precision!r}")
    if geometry_path not in GEOMETRY_PATHS:
        raise ValueError(f"Unknown transition geometry path: {geometry_path!r}")
    if not isinstance(getattr(runtime, "device", None), torch.device):
        raise ValueError("Historical runtime device is malformed")
    if producer.TARGET_CONFIG != {"pressure": "scalar", "wss": "vector"}:
        raise ValueError("Frozen v3 target configuration differs")
    target_config = {"pressure": "scalar", "wss": "vector"}
    device = runtime.device
    autocast_context = runtime.autocast_context
    normalize_output = runtime.normalize_output
    output_type = str(runtime.cfg.output_type)
    mesh_type = runtime.mesh_type
    canonical_geometry_for_model = getattr(
        producer,
        "_canonical_geometry_for_model",
        None,
    )
    tensor_raw_equal = getattr(producer, "_tensor_raw_equal", None)
    if (
        not callable(autocast_context)
        or not callable(normalize_output)
        or not callable(mesh_type)
        or (
            geometry_path == "canonical"
            and (
                not callable(canonical_geometry_for_model)
                or not callable(tensor_raw_equal)
            )
        )
    ):
        raise ValueError("Frozen v3 execution slice is malformed")
    _validate_prepared_case(
        case,
        case_index=case_index,
        cohort_ordinal=cohort_ordinal,
        case_id=case_id,
        historical_start=historical_start,
    )
    bindings = _capture_case_bindings(case)
    case_sha256 = stable_sha256(_case_execution_snapshot(case))

    loss_module = __import__("loss")
    loss_class = getattr(loss_module, "LossCalculator", None)
    if not isinstance(loss_class, type):
        raise ImportError("Pinned recipe LossCalculator is absent")
    loss_calculator = loss_class(
        target_config,
        loss_type="huber",
        n_spatial_dims=3,
        normalize_by_channels=True,
    )
    if (
        loss_calculator.target_config != target_config
        or loss_calculator.loss_type != "huber"
        or loss_calculator.n_spatial_dims != 3
        or loss_calculator.normalize_by_channels is not True
        or loss_calculator.delta != 1.0
        or loss_calculator.total_channels != 4
        or loss_calculator.field_weights != {"pressure": 1.0, "wss": 1.0}
    ):
        raise ValueError("Pinned Huber loss configuration differs")

    def forward_loss(
        package: TrajectoryPackage,
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        _require_trajectory_package(package)
        _assert_case_unchanged(
            case,
            bindings=bindings,
            expected_sha256=case_sha256,
        )
        try:
            domain = case["domain"]
            with autocast_context(precision):
                if geometry_path == "legacy":
                    raw_output = package.model(domain)
                else:
                    geometry = canonical_geometry_for_model(
                        package.model,
                        None,
                        domain,
                        case["bundle"],
                    )
                    encoded = package.model.encode(
                        domain,
                        canonical_source_geometry=geometry,
                    )
                    exact_pairs = (
                        (encoded.source_mesh.points, geometry.points),
                        (encoded.source_mesh.cells, geometry.cells),
                        (encoded.source_mesh.cell_centroids, geometry.centroids),
                        (encoded.source_mesh.cell_areas, geometry.areas),
                        (encoded.source_mesh.cell_normals, geometry.normals),
                        (encoded.center, geometry.center),
                        (encoded.reference_length, geometry.reference_length),
                    )
                    if not all(
                        tensor_raw_equal(left, right) for left, right in exact_pairs
                    ):
                        raise ValueError(
                            "Canonical geometry was not installed byte-exactly"
                        )
                    if "_measure_weights" in encoded.source_mesh.cell_data:
                        raise ValueError(
                            "Canonical encoded source unexpectedly carries weights"
                        )
                    query_mesh = mesh_type(points=geometry.centroids)
                    if not tensor_raw_equal(
                        query_mesh.points,
                        geometry.centroids,
                    ):
                        raise ValueError("Canonical query geometry differs")
                    raw_output = package.model.decode(encoded, query_mesh)

            normalized = normalize_output(
                raw_output,
                target_config,
                output_type,
            ).float()
            prediction = _validate_transition_prediction(
                normalized,
                device=device,
            )
            targets = case["targets"].float()
            target_measure = case["target_measure"].float()
            loss, loss_fields = loss_calculator(
                normalized,
                targets,
                target_measure,
            )
            expected_loss_keys = ("loss/pressure", "loss/wss", "loss/total")
            if (
                not isinstance(loss_fields, Mapping)
                or tuple(loss_fields.keys()) != expected_loss_keys
            ):
                raise ValueError("Pinned Huber loss-field keys or order differ")
            for name in expected_loss_keys:
                value = loss_fields[name]
                if (
                    type(value) is not torch.Tensor
                    or value.dtype is not torch.float32
                    or value.device != device
                    or value.ndim != 0
                    or not bool(torch.isfinite(value))
                ):
                    raise ValueError(f"Pinned Huber loss field differs: {name}")
            if (
                type(loss) is not torch.Tensor
                or loss is not loss_fields["loss/total"]
                or loss.dtype is not torch.float32
                or loss.device != device
                or loss.ndim != 0
                or not bool(torch.isfinite(loss))
            ):
                raise ValueError("Pinned Huber total loss differs")
            return prediction, loss
        finally:
            _assert_case_unchanged(
                case,
                bindings=bindings,
                expected_sha256=case_sha256,
            )

    return forward_loss


def _load_verified_v3_cases(
    producer: ModuleType,
    legacy: ModuleType,
    runtime_support: ModuleType,
    canonical: ModuleType,
    *,
    repo_root: Path,
    dataset_root: Path,
    dataset_config: Path,
    resolved_config: Path,
    checkpoint_dir: Path,
    geometry_manifest: Path,
    target_input_manifest: Path,
) -> dict[str, Any]:
    """Validate all frozen inputs, then prepare only the four sealed cases."""
    paths = {
        "repo_root": repo_root,
        "dataset_root": dataset_root,
        "dataset_config": dataset_config,
        "resolved_config": resolved_config,
        "checkpoint_dir": checkpoint_dir,
        "geometry_manifest": geometry_manifest,
        "target_input_manifest": target_input_manifest,
    }
    if any(
        not isinstance(path, Path) or not path.is_absolute() for path in paths.values()
    ):
        raise ValueError("Every frozen-v3 input path must be an absolute Path")

    recipe_preflight = _preflight_recipe_import_namespace(repo_root)
    producer._validate_single_rank_environment()
    legacy._validate_case_specs(runtime_support)
    hash_contracts = (
        (
            producer.EXPECTED_SOURCE_TREE_SHA256,
            legacy.EXPECTED_EXECUTION_SOURCE_TREE_SHA256,
        ),
        (
            producer.EXPECTED_DATASET_MANIFEST_SHA256,
            legacy.EXPECTED_DATASET_MANIFEST_SHA256,
        ),
        (
            producer.EXPECTED_DATASET_CONFIG_SHA256,
            legacy.EXPECTED_DATASET_CONFIG_SHA256,
        ),
        (
            producer.EXPECTED_RESOLVED_CONFIG_SHA256,
            legacy.EXPECTED_RESOLVED_CONFIG_SHA256,
        ),
        (
            producer.EXPECTED_MODEL_CHECKPOINT_SHA256,
            legacy.EXPECTED_MODEL_SHA256,
        ),
        (
            producer.EXPECTED_TRAINING_STATE_SHA256,
            legacy.EXPECTED_TRAINING_STATE_SHA256,
        ),
        (
            producer.EXPECTED_NORMALIZATION_STATE_SHA256,
            legacy.EXPECTED_NORMALIZATION_SHA256,
        ),
    )
    if any(left != right for left, right in hash_contracts):
        raise ValueError("Frozen v3 and support input hashes disagree")

    expected_static_inputs = {
        "Dataset manifest": producer.EXPECTED_DATASET_MANIFEST_SHA256,
        "Dataset config": producer.EXPECTED_DATASET_CONFIG_SHA256,
        "Resolved config": producer.EXPECTED_RESOLVED_CONFIG_SHA256,
        "Model checkpoint": producer.EXPECTED_MODEL_CHECKPOINT_SHA256,
        "Training state": producer.EXPECTED_TRAINING_STATE_SHA256,
        "Normalization state": producer.EXPECTED_NORMALIZATION_STATE_SHA256,
        "Current inference source": legacy.EXPECTED_CURRENT_INFER_SHA256,
        "Current MeshTransformer source": legacy.EXPECTED_CURRENT_MODEL_SOURCE_SHA256,
        "Current execution source tree": producer.EXPECTED_SOURCE_TREE_SHA256,
    }

    def revalidate_static_inputs() -> dict[str, str]:
        observed = legacy._validate_static_inputs(
            runtime_support,
            repo_root=repo_root,
            dataset_root=dataset_root,
            dataset_config=dataset_config,
            resolved_config=resolved_config,
            checkpoint_dir=checkpoint_dir,
        )
        if type(observed) is not dict or observed != expected_static_inputs:
            raise ValueError("Frozen static-input attestation differs")
        return dict(observed)

    static_inputs = revalidate_static_inputs()

    geometry = legacy._verify_geometry_manifest(
        runtime_support,
        geometry_manifest,
        dataset_root,
    )
    target = legacy._verify_target_input_manifest(
        target_input_manifest,
        dataset_root,
    )
    if type(geometry) is not dict or type(target) is not dict:
        raise ValueError("Frozen input-manifest attestations are malformed")
    _exact_keys(
        geometry,
        {"manifest_sha256", "cases_verified", "files_verified", "case_records"},
        "Geometry-manifest attestation",
    )
    _exact_keys(
        target,
        {
            "manifest_sha256",
            "cases_verified",
            "selected_ranges_verified",
            "case_records",
        },
        "Target-manifest attestation",
    )
    if (
        geometry["manifest_sha256"] != legacy.EXPECTED_GEOMETRY_MANIFEST_SHA256
        or geometry["cases_verified"] != 36
        or type(geometry["files_verified"]) is not int
        or geometry["files_verified"] <= 0
        or target["manifest_sha256"] != legacy.EXPECTED_TARGET_INPUT_MANIFEST_SHA256
        or target["cases_verified"] != 36
        or target["selected_ranges_verified"] != 72
    ):
        raise ValueError("Frozen input-manifest coverage differs")

    geometry_cases = geometry["case_records"]
    target_cases = target["case_records"]
    expected_case_ids = tuple(spec.case_id for spec in runtime_support.CASE_SPECS)
    if (
        type(geometry_cases) is not list
        or type(target_cases) is not list
        or tuple(record.get("case_id") for record in geometry_cases)
        != expected_case_ids
        or tuple(record.get("case_id") for record in target_cases) != expected_case_ids
    ):
        raise ValueError("Frozen 36-case input order differs")
    geometry_by_id = {record["case_id"]: record for record in geometry_cases}
    target_by_id = {record["case_id"]: record for record in target_cases}
    helper_by_id = {spec.case_id: spec for spec in runtime_support.CASE_SPECS}
    if (
        len(geometry_by_id) != 36
        or len(target_by_id) != 36
        or len(helper_by_id) != 36
        or set(geometry_by_id) != set(helper_by_id)
        or set(target_by_id) != set(helper_by_id)
    ):
        raise ValueError("Frozen 36-case input sets differ")

    revalidate_static_inputs()
    runtime = runtime_support._load_runtime(
        repo_root=repo_root,
        dataset_root=dataset_root,
        dataset_config_path=dataset_config,
        resolved_config_path=resolved_config,
        checkpoint_dir=checkpoint_dir,
    )
    revalidate_static_inputs()
    legacy_provenance = legacy._validate_import_provenance(repo_root)
    recipe_provenance = _validate_recipe_import_provenance(
        repo_root,
        preflight=recipe_preflight,
        runtime=runtime,
    )
    legacy._validate_reader(runtime)
    if (
        bool(runtime.cfg.compile) is not True
        or str(runtime.cfg.precision) != "bfloat16"
        or runtime.loaded_epoch != CHECKPOINT_EPOCH
    ):
        raise ValueError("Historical runtime configuration differs")
    backend = _execution_backend_attestation(runtime)

    prepared_cases: dict[str, dict[str, Any]] = {}
    case_controls = []
    for case_index, cohort_ordinal, case_id in FIXED_CASE_SPECS:
        spec = helper_by_id[case_id]
        if spec.cohort_ordinal != cohort_ordinal:
            raise ValueError(f"Frozen helper cohort ordinal differs for {case_id}")
        case = producer._prepare_case(
            legacy=legacy,
            runtime_support=runtime_support,
            canonical=canonical,
            runtime=runtime,
            dataset_root=dataset_root,
            spec=spec,
            geometry_case=geometry_by_id[case_id],
            target_case=target_by_id[case_id],
        )
        prepared_cases[case_id] = case
        control = _validate_prepared_case(
            case,
            case_index=case_index,
            cohort_ordinal=cohort_ordinal,
            case_id=case_id,
            historical_start=spec.historical_start,
        )
        if case["batch_order_sha256"] != producer._batch_order_sha256(
            case_id, case["selected_ids"]
        ) or case["global_inputs_sha256"] != producer._global_inputs_sha256(
            legacy, case["domain"]
        ):
            raise ValueError(f"Prepared case {case_id} identity controls differ")
        case_controls.append(control)
    return {
        "runtime": runtime,
        "cases": prepared_cases,
        "static_input_revalidator": revalidate_static_inputs,
        "attestation": {
            "static_inputs": static_inputs,
            "geometry_manifest": {
                "manifest_sha256": geometry["manifest_sha256"],
                "case_records_contract_checked": geometry["cases_verified"],
                "geometry_file_records_path_type_size_checked": geometry[
                    "files_verified"
                ],
                "prepared_case_count": len(prepared_cases),
                "prepared_case_geometry_memmap_files_sha256_verified": (
                    2 * len(prepared_cases)
                ),
            },
            "target_manifest": {
                "manifest_sha256": target["manifest_sha256"],
                "case_records_contract_checked": target["cases_verified"],
                "selected_target_range_records_contract_checked": target[
                    "selected_ranges_verified"
                ],
                "prepared_case_count": len(prepared_cases),
                "prepared_case_selected_target_ranges_sha256_verified": (
                    2 * len(prepared_cases)
                ),
            },
            "legacy_import_provenance": legacy_provenance,
            "recipe_import_provenance": recipe_provenance,
            "backend": backend,
            "case_controls": tuple(case_controls),
        },
    }


def _validate_raw_npz(
    path: Path,
    *,
    manifest: Any,
    expected_order: Any,
) -> dict[str, Any]:
    """Reread one raw NPZ with bounded memory and authenticate every byte."""
    records = _validated_raw_manifest(manifest, expected_order)
    expected_layout = _expected_raw_npz_layout(records)
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Validated raw NPZ path must be an absolute Path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Raw NPZ could not be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Validated raw NPZ is not a regular file")
        if before.st_nlink != 1:
            raise ValueError("Validated raw NPZ must have exactly one hard link")
        directory_offset, directory_size = _zip_directory_contract(
            descriptor,
            before.st_size,
            len(records),
        )
        if (
            before.st_size != expected_layout["file_size"]
            or directory_offset != expected_layout["directory_offset"]
            or directory_size != expected_layout["directory_size"]
        ):
            raise ValueError("Raw NPZ exact archive layout differs")
        stream = os.fdopen(os.dup(descriptor), "rb")
        try:
            with zipfile.ZipFile(stream, "r") as archive:
                infos = archive.infolist()
                expected_names = tuple(f"{key}.npy" for key in expected_order)
                if (
                    archive.comment != b""
                    or tuple(info.filename for info in infos) != expected_names
                    or len(infos) != len(set(expected_names))
                ):
                    raise ValueError("Raw NPZ ZIP member order or names differ")
                for info in infos:
                    if (
                        info.compress_type != zipfile.ZIP_STORED
                        or info.flag_bits != 0
                        or info.create_system != 3
                        or info.create_version != 45
                        or info.extract_version != 45
                        or info.date_time != (1980, 1, 1, 0, 0, 0)
                        or info.comment != b""
                        or info.volume != 0
                        or info.internal_attr != 0
                        or info.external_attr != 0o600 << 16
                        or info.file_size != info.compress_size
                    ):
                        raise ValueError(
                            f"Raw NPZ ZIP metadata differs: {info.filename}"
                        )
                if archive.start_dir != directory_offset:
                    raise ValueError("Raw NPZ ZIP start-directory offset differs")
                for info, expected_offset, expected_size in zip(
                    infos,
                    expected_layout["local_offsets"],
                    expected_layout["member_sizes"],
                    strict=True,
                ):
                    if (
                        info.header_offset != expected_offset
                        or info.file_size != expected_size
                        or info.compress_size != expected_size
                    ):
                        raise ValueError(
                            f"Raw NPZ exact member layout differs: {info.filename}"
                        )
                _validate_zip_local_headers(
                    descriptor,
                    infos=infos,
                    directory_offset=directory_offset,
                )
                _validate_zip_central_directory(
                    descriptor,
                    directory_offset=directory_offset,
                    directory_size=directory_size,
                    infos=infos,
                )
                for info, (
                    key,
                    dtype,
                    shape,
                    nbytes,
                    expected_sha256,
                ) in zip(infos, records, strict=True):
                    if info.filename != f"{key}.npy":
                        raise ValueError("Raw NPZ ZIP and manifest order differ")
                    _stream_validate_npy_member(
                        archive,
                        info,
                        dtype=dtype,
                        shape=shape,
                        nbytes=nbytes,
                        expected_sha256=expected_sha256,
                    )
        finally:
            stream.close()

        whole_digest = hashlib.sha256()
        position = 0
        while position < before.st_size:
            chunk = os.pread(
                descriptor,
                min(8 << 20, before.st_size - position),
                position,
            )
            if not chunk:
                raise ValueError("Raw NPZ changed or truncated during whole-file hash")
            whole_digest.update(chunk)
            position += len(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if _file_identity(before) != _file_identity(after) or _file_identity(
            before
        ) != _file_identity(path_after):
            raise ValueError("Raw NPZ changed while it was validated")
        return {
            "size_bytes": before.st_size,
            "sha256": whole_digest.hexdigest(),
            "member_count": len(records),
        }
    except (
        OSError,
        EOFError,
        RuntimeError,
        struct.error,
        zipfile.BadZipFile,
    ) as error:
        raise ValueError("Raw NPZ could not be validated safely") from error
    finally:
        os.close(descriptor)


_TORCH_STATE_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def _require_state_tree_value(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise TypeError(f"{label} contains a Tensor subclass")
        if str(value.dtype) not in _TORCH_STATE_DTYPES:
            raise ValueError(f"{label} contains an unsupported Torch dtype")
        _require_canonical_tensor(value, label)
        _require_inert_tensor(value, label)
        return
    if isinstance(value, np.ndarray):
        _require_canonical_numpy_array(value, label)
        return
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise TypeError(f"{label} mapping must be an exact dict")
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} mapping keys must be exact strings")
            _require_state_tree_value(item, f"{label}.{key}")
        return
    if isinstance(value, tuple):
        if type(value) is not tuple:
            raise TypeError(f"{label} tuple must be exact")
        for index, item in enumerate(value):
            _require_state_tree_value(item, f"{label}[{index}]")
        return
    if isinstance(value, list):
        if type(value) is not list:
            raise TypeError(f"{label} list must be exact")
        for index, item in enumerate(value):
            _require_state_tree_value(item, f"{label}[{index}]")
        return
    if isinstance(value, bytes):
        if type(value) is not bytes:
            raise TypeError(f"{label} bytes must be exact")
        return
    if value is None or type(value) in {bool, int, float, str}:
        return
    raise TypeError(f"Unsupported state-tree value: {type(value).__name__}")


class _StateTreeEncoder:
    """Encode a canonical state tree as strict JSON plus streamed byte arrays."""

    def __init__(
        self,
        writer: _RawNpzWriter,
        *,
        prefix: str,
        state_kind: str,
    ) -> None:
        if type(writer) is not _RawNpzWriter:
            raise TypeError("State-tree writer must be an exact _RawNpzWriter")
        if type(prefix) is not str or _RAW_NPZ_KEY.fullmatch(prefix) is None:
            raise ValueError("State-tree prefix is not canonical")
        if type(state_kind) is not str or state_kind not in _STATE_TREE_KINDS:
            raise ValueError("State-tree kind is unsupported")
        self._writer = writer
        self._prefix = prefix
        self._state_kind = state_kind
        self._array_index = 0
        self._encoded = False

    def _write_bytes(self, value: np.ndarray) -> str:
        raw = np.frombuffer(value.tobytes(order="C"), dtype=np.uint8).copy()
        key = f"{self._prefix}_leaf_{self._array_index:06d}_bytes"
        self._array_index += 1
        self._writer.add(key, raw)
        return key

    def _encode(self, value: Any) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return {
                "kind": "torch_tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "array_key": self._write_bytes(
                    tensor.reshape(-1).view(torch.uint8).numpy()
                ),
            }
        if isinstance(value, np.ndarray):
            return {
                "kind": "numpy_array",
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "array_key": self._write_bytes(value),
            }
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "items": [
                    {
                        "key": self._encode(key),
                        "value": self._encode(value[key]),
                    }
                    for key in sorted(value)
                ],
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [self._encode(item) for item in value],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [self._encode(item) for item in value],
            }
        if isinstance(value, bytes):
            return {
                "kind": "bytes",
                "array_key": self._write_bytes(np.frombuffer(value, dtype=np.uint8)),
            }
        if value is None:
            return {"kind": "none"}
        if type(value) is bool:
            return {"kind": "bool", "value": value}
        if type(value) is int:
            return {"kind": "int", "value_decimal": str(value)}
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("State tree contains a non-finite float")
            return {"kind": "float", "value_hex": value.hex()}
        if type(value) is str:
            return {"kind": "str", "value": value}
        raise TypeError(f"Unsupported state-tree value: {type(value).__name__}")

    def encode(self, value: Any) -> dict[str, Any]:
        if self._encoded:
            raise RuntimeError("State-tree encoder is single-use")
        _require_finite_state_value(value, "State tree")
        _require_state_tree_value(value, "State tree")
        state_sha256 = stable_sha256(value)
        root = self._encode(value)
        self._encoded = True
        return {
            "schema_version": STATE_TREE_SCHEMA_VERSION,
            "state_kind": self._state_kind,
            "stable_sha256": state_sha256,
            "root": root,
        }


def _state_tree_raw_bytes(
    arrays: Mapping[str, np.ndarray],
    key: Any,
    referenced_keys: set[str],
    expected_prefix: str,
) -> np.ndarray:
    if type(key) is not str or _RAW_NPZ_KEY.fullmatch(key) is None:
        raise ValueError("State-tree array key is not canonical")
    expected_key = f"{expected_prefix}_leaf_{len(referenced_keys):06d}_bytes"
    if key != expected_key:
        raise ValueError(f"State-tree array key is not the next contiguous leaf: {key}")
    if key in referenced_keys:
        raise ValueError(f"State-tree array key is referenced twice: {key}")
    try:
        value = arrays[key]
    except KeyError as error:
        raise ValueError(f"State-tree array is absent: {key}") from error
    if type(value) is not np.ndarray or value.dtype.str != "|u1" or value.ndim != 1:
        raise ValueError(f"State-tree array is not canonical bytes: {key}")
    try:
        _require_canonical_numpy_array(value, f"State-tree array {key}")
    except (TypeError, ValueError) as error:
        raise ValueError(f"State-tree array is not canonical bytes: {key}") from error
    referenced_keys.add(key)
    return value


def _state_tree_shape(value: Any) -> tuple[int, ...]:
    if type(value) is not list or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError("State-tree shape is malformed")
    return tuple(value)


def _decode_state_tree_node(
    node: Any,
    arrays: Mapping[str, np.ndarray],
    referenced_keys: set[str],
    expected_prefix: str,
) -> Any:
    if type(node) is not dict or type(node.get("kind")) is not str:
        raise ValueError("State-tree node is malformed")
    kind = node["kind"]
    if kind == "torch_tensor":
        _exact_keys(node, {"kind", "dtype", "shape", "array_key"}, "Tensor node")
        if type(node["dtype"]) is not str:
            raise ValueError("State-tree Torch dtype is malformed")
        dtype = _TORCH_STATE_DTYPES.get(node["dtype"])
        if dtype is None:
            raise ValueError("State-tree Torch dtype is unsupported")
        shape = _state_tree_shape(node["shape"])
        raw = _state_tree_raw_bytes(
            arrays,
            node["array_key"],
            referenced_keys,
            expected_prefix,
        )
        element_size = torch.empty((), dtype=dtype).element_size()
        expected_elements = math.prod(shape)
        if raw.nbytes != expected_elements * element_size:
            raise ValueError("State-tree Tensor byte count differs")
        if dtype is torch.bool and bool(((raw != 0) & (raw != 1)).any()):
            raise ValueError("State-tree Tensor has noncanonical Boolean storage")
        tensor = (
            torch.empty(expected_elements, dtype=dtype)
            if expected_elements == 0
            else torch.from_numpy(raw.copy()).view(dtype)
        )
        return tensor.reshape(shape).contiguous().clone()
    if kind == "numpy_array":
        _exact_keys(
            node,
            {"kind", "dtype", "shape", "array_key"},
            "ndarray node",
        )
        if type(node["dtype"]) is not str:
            raise ValueError("State-tree NumPy dtype is malformed")
        try:
            dtype = np.dtype(node["dtype"])
        except (TypeError, ValueError) as error:
            raise ValueError("State-tree NumPy dtype is invalid") from error
        if (
            dtype.str not in _CANONICAL_NUMPY_DTYPES
            or dtype.str != node["dtype"]
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.metadata is not None
        ):
            raise ValueError("State-tree NumPy dtype is not canonical")
        shape = _state_tree_shape(node["shape"])
        raw = _state_tree_raw_bytes(
            arrays,
            node["array_key"],
            referenced_keys,
            expected_prefix,
        )
        if raw.nbytes != math.prod(shape) * dtype.itemsize:
            raise ValueError("State-tree ndarray byte count differs")
        if dtype.str == "|b1" and bool(((raw != 0) & (raw != 1)).any()):
            raise ValueError("State-tree ndarray has noncanonical Boolean storage")
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape).copy()
    if kind == "mapping":
        _exact_keys(node, {"kind", "items"}, "mapping node")
        if type(node["items"]) is not list:
            raise ValueError("State-tree mapping items are malformed")
        result: dict[Any, Any] = {}
        previous_key: str | None = None
        for item in node["items"]:
            if type(item) is not dict:
                raise ValueError("State-tree mapping item is malformed")
            _exact_keys(item, {"key", "value"}, "mapping item")
            key = _decode_state_tree_node(
                item["key"],
                arrays,
                referenced_keys,
                expected_prefix,
            )
            if type(key) is not str:
                raise ValueError("State-tree mapping key is not an exact string")
            if key in result:
                raise ValueError("State-tree mapping key is duplicated")
            if previous_key is not None and key <= previous_key:
                raise ValueError("State-tree mapping keys are not strictly ordered")
            result[key] = _decode_state_tree_node(
                item["value"],
                arrays,
                referenced_keys,
                expected_prefix,
            )
            previous_key = key
        return result
    if kind in {"tuple", "list"}:
        _exact_keys(node, {"kind", "items"}, f"{kind} node")
        if type(node["items"]) is not list:
            raise ValueError(f"State-tree {kind} items are malformed")
        items = [
            _decode_state_tree_node(
                item,
                arrays,
                referenced_keys,
                expected_prefix,
            )
            for item in node["items"]
        ]
        return tuple(items) if kind == "tuple" else items
    if kind == "bytes":
        _exact_keys(node, {"kind", "array_key"}, "bytes node")
        return _state_tree_raw_bytes(
            arrays,
            node["array_key"],
            referenced_keys,
            expected_prefix,
        ).tobytes()
    if kind == "none":
        _exact_keys(node, {"kind"}, "none node")
        return None
    if kind == "bool":
        _exact_keys(node, {"kind", "value"}, "Boolean node")
        if type(node["value"]) is not bool:
            raise ValueError("State-tree Boolean is malformed")
        return node["value"]
    if kind == "int":
        _exact_keys(node, {"kind", "value_decimal"}, "integer node")
        text = node["value_decimal"]
        if type(text) is not str:
            raise ValueError("State-tree integer is malformed")
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError("State-tree integer is invalid") from error
        if str(value) != text:
            raise ValueError("State-tree integer is not canonical")
        return value
    if kind == "float":
        _exact_keys(node, {"kind", "value_hex"}, "float node")
        text = node["value_hex"]
        if type(text) is not str:
            raise ValueError("State-tree float is malformed")
        try:
            value = float.fromhex(text)
        except (ValueError, OverflowError) as error:
            raise ValueError("State-tree float is invalid") from error
        if not math.isfinite(value) or value.hex() != text:
            raise ValueError("State-tree float is not canonical")
        return value
    if kind == "str":
        _exact_keys(node, {"kind", "value"}, "string node")
        if type(node["value"]) is not str:
            raise ValueError("State-tree string is malformed")
        return node["value"]
    raise ValueError(f"Unknown state-tree node kind: {kind}")


def _decode_state_tree(
    envelope: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    expected_prefix: str,
    expected_state_kind: str,
    claimed_keys: set[str],
) -> tuple[Any, frozenset[str]]:
    if type(envelope) is not dict:
        raise ValueError("State-tree envelope is malformed")
    _exact_keys(
        envelope,
        {"schema_version", "state_kind", "stable_sha256", "root"},
        "State-tree envelope",
    )
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != STATE_TREE_SCHEMA_VERSION
    ):
        raise ValueError("State-tree schema version differs")
    if (
        type(expected_state_kind) is not str
        or expected_state_kind not in _STATE_TREE_KINDS
        or envelope["state_kind"] != expected_state_kind
    ):
        raise ValueError("State-tree kind differs")
    if (
        type(envelope["stable_sha256"]) is not str
        or _SHA256_HEX.fullmatch(envelope["stable_sha256"]) is None
    ):
        raise ValueError("State-tree stable SHA-256 is malformed")
    if (
        type(expected_prefix) is not str
        or _RAW_NPZ_KEY.fullmatch(expected_prefix) is None
    ):
        raise ValueError("Expected state-tree prefix is not canonical")
    if type(claimed_keys) is not set or any(
        type(key) is not str or _RAW_NPZ_KEY.fullmatch(key) is None
        for key in claimed_keys
    ):
        raise ValueError("Claimed state-tree key set is malformed")

    referenced_keys: set[str] = set()
    value = _decode_state_tree_node(
        envelope["root"],
        arrays,
        referenced_keys,
        expected_prefix,
    )
    _require_finite_state_value(value, "Decoded state tree")
    _require_state_tree_value(value, "Decoded state tree")
    if stable_sha256(value) != envelope["stable_sha256"]:
        raise ValueError("Decoded state-tree hash differs from its envelope")
    duplicate_claims = referenced_keys.intersection(claimed_keys)
    if duplicate_claims:
        raise ValueError(
            f"State-tree arrays are already claimed: {sorted(duplicate_claims)}"
        )
    claimed_keys.update(referenced_keys)
    return value, frozenset(referenced_keys)
