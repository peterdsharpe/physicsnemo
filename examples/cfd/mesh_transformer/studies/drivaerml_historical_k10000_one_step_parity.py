# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce raw evidence for the DrivAerML K=10k one-step parity gate.

The same four frozen cases are evaluated from a fresh seed-42 initialization
and from the exact epoch-491 model/optimizer checkpoint.  For each state and
precision, two independently restored arms run either the historical
``model(domain)`` path or the public canonical-source encode/decode path.
Both arms use an explicitly unweighted source, the identical target values,
the identical target quadrature measure, the current channel-normalized Huber
loss, and exactly one current CombinedOptimizer(Muon, AdamW) step.

This producer publishes raw predictions, gradients, parameter updates, and
control hashes only.  It computes no parity statistic and publishes no
categorical scientific outcome; the independent adjudicator owns that step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "drivaerml_historical_k10000_one_step_parity_producer"
STATUS = "PASSED_HISTORICAL_K10000_ONE_STEP_PARITY_PRODUCER"

RESOLUTION = 10_000
FRESH_SEED = 42
CHECKPOINT_EPOCH = 491
TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}
CASE_SPECS = (
    (0, "run_118"),
    (12, "run_271"),
    (24, "run_429"),
    (35, "run_86"),
)
REGIMES = ("fresh_seed42", "checkpoint_epoch491")
PRECISIONS = ("bfloat16", "float32")
ARMS = ("legacy", "canonical")

LEGACY_SUPPORT_FILENAME = "drivaerml_historical_k10000_replay.py"
RUNTIME_HELPER_FILENAME = "drivaerml_historical_k10000_replay_runtime.py"
CANONICAL_HELPER_FILENAME = "drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
EXPECTED_LEGACY_SUPPORT_SHA256 = (
    "bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5"
)
EXPECTED_RUNTIME_HELPER_SHA256 = (
    "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
)
EXPECTED_CANONICAL_HELPER_SHA256 = (
    "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
)

EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
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
EXPECTED_PARAMETER_COUNT = 1_278_268
EXPECTED_SINGLE_RANK_ENVIRONMENT = {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
}

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


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_module(
    path: Path,
    *,
    expected_sha256: str,
    module_name: str,
    label: str,
) -> ModuleType:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 differs: expected {expected_sha256}, got {observed}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {label} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_support_modules(
    script_path: Path,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    directory = script_path.parent
    legacy = _load_verified_module(
        directory / LEGACY_SUPPORT_FILENAME,
        expected_sha256=EXPECTED_LEGACY_SUPPORT_SHA256,
        module_name="frozen_one_step_legacy_support",
        label="Frozen historical replay support",
    )
    runtime = _load_verified_module(
        directory / RUNTIME_HELPER_FILENAME,
        expected_sha256=EXPECTED_RUNTIME_HELPER_SHA256,
        module_name="frozen_one_step_runtime",
        label="Frozen historical replay runtime",
    )
    canonical = _load_verified_module(
        directory / CANONICAL_HELPER_FILENAME,
        expected_sha256=EXPECTED_CANONICAL_HELPER_SHA256,
        module_name="frozen_one_step_canonical_support",
        label="Frozen canonical geometry helper",
    )
    return legacy, runtime, canonical


def _hash_update(digest: Any, value: Any) -> None:
    """Update ``digest`` with a deterministic typed representation."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.view(np.uint8).tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
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
        digest.update(f"float\0{value.hex()}\0".encode("ascii"))
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"str\0")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        return
    raise TypeError(f"Unsupported hash value type: {type(value).__name__}")


def _stable_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _parameter_order_sha256(names: Sequence[str]) -> str:
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _rng_state_sha256() -> str:
    return _stable_sha256(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }
    )


def _validate_single_rank_environment() -> None:
    observed = {name: os.environ.get(name) for name in EXPECTED_SINGLE_RANK_ENVIRONMENT}
    if observed != EXPECTED_SINGLE_RANK_ENVIRONMENT:
        raise ValueError(
            "One-step parity producer requires one torchrun rank: "
            f"expected={EXPECTED_SINGLE_RANK_ENVIRONMENT} observed={observed}"
        )


def _prefix(
    regime: str,
    precision: str,
    ordinal: int,
    case_id: str,
    arm: str,
) -> str:
    return f"{regime}__{precision}__case_{ordinal:02d}_{case_id}__{arm}__"


def _canonical_geometry_for_model(
    model: Any,
    runtime: Any,
    domain: Any,
    bundle: Any,
) -> Any:
    from physicsnemo.experimental.nn.mesh_attention import CanonicalSourceGeometry
    from physicsnemo.experimental.nn.mesh_attention import model as model_module

    if CanonicalSourceGeometry is not model_module.CanonicalSourceGeometry:
        raise ValueError("Public CanonicalSourceGeometry export has split identity")
    first_boundary = domain.boundaries[model.boundary_names[0]]
    device = first_boundary.points.device
    dtype = first_boundary.points.dtype
    return CanonicalSourceGeometry(
        points=bundle.points.to(device=device, dtype=dtype),
        cells=bundle.cells.to(
            device=first_boundary.cells.device,
            dtype=first_boundary.cells.dtype,
        ),
        centroids=bundle.centroids.to(device=device, dtype=dtype),
        areas=bundle.areas.to(device=device, dtype=dtype),
        normals=bundle.normals.to(device=device, dtype=dtype),
        center=torch.zeros(
            first_boundary.n_spatial_dims,
            device=device,
            dtype=dtype,
        ),
        reference_length=torch.ones((), device=device, dtype=dtype),
    )


def _tensor_raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _parameter_layout(
    model: torch.nn.Module,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    named_parameters = list(model.named_parameters())
    names = [name for name, _ in named_parameters]
    if not names or len(names) != len(set(names)):
        raise ValueError("Trainable parameter names are empty or nonunique")
    if any(not parameter.requires_grad for _, parameter in named_parameters):
        raise ValueError("named_parameters unexpectedly includes a frozen tensor")

    starts: list[int] = []
    stops: list[int] = []
    module_names: list[str] = []
    module_lookup: dict[str, int] = {}
    module_indices: list[int] = []
    offset = 0
    for name, parameter in named_parameters:
        starts.append(offset)
        offset += parameter.numel()
        stops.append(offset)
        module_name = name.rpartition(".")[0] or "<root>"
        if module_name not in module_lookup:
            module_lookup[module_name] = len(module_names)
            module_names.append(module_name)
        module_indices.append(module_lookup[module_name])
    if offset != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"Parameter count changed: expected {EXPECTED_PARAMETER_COUNT}, got {offset}"
        )
    layout = {
        "parameter_count": offset,
        "parameter_names": names,
        "module_names": module_names,
        "ordered_parameter_names_sha256": _parameter_order_sha256(names),
    }
    arrays = {
        "parameter_slice_starts_int64": np.asarray(starts, dtype="<i8"),
        "parameter_slice_stops_int64": np.asarray(stops, dtype="<i8"),
        "parameter_slice_module_indices_int64": np.asarray(module_indices, dtype="<i8"),
    }
    return layout, arrays


def _flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().reshape(-1).float().cpu()
            for _, parameter in model.named_parameters()
        ]
    )


def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        else:
            if parameter.grad.is_sparse:
                raise ValueError(f"Sparse gradient is unsupported for {name}")
            parts.append(parameter.grad.detach().reshape(-1).float().cpu())
    if missing:
        raise ValueError(f"Parameters have absent gradients: {missing}")
    return torch.cat(parts)


def _stochastic_modules(model: torch.nn.Module) -> list[str]:
    stochastic_types = (
        torch.nn.modules.dropout._DropoutNd,
        torch.nn.RReLU,
    )
    return [
        name or "<root>"
        for name, module in model.named_modules()
        if isinstance(module, stochastic_types)
    ]


def _new_model_optimizer(
    runtime: Any,
    *,
    regime: str,
    checkpoint_dir: Path,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int | None]:
    import hydra
    from utils import build_muon_optimizer, set_seed

    from physicsnemo.utils import load_checkpoint

    set_seed(FRESH_SEED, rank=0)
    model = hydra.utils.instantiate(runtime.cfg.model, _convert_="partial").to(
        runtime.device
    )
    optimizer = build_muon_optimizer(
        model,
        runtime.cfg,
        compile_optimizer=False,
    )
    loaded_epoch: int | None = None
    if regime == "checkpoint_epoch491":
        loaded_epoch = int(
            load_checkpoint(
                path=str(checkpoint_dir),
                models=model,
                optimizer=optimizer,
                device=runtime.device,
                epoch=CHECKPOINT_EPOCH,
            )
        )
        if loaded_epoch != CHECKPOINT_EPOCH:
            raise ValueError(
                f"Loaded checkpoint epoch {loaded_epoch}, expected {CHECKPOINT_EPOCH}"
            )
    elif regime != "fresh_seed42":
        raise ValueError(f"Unknown parameter regime {regime}")

    if optimizer.__class__.__name__ != "CombinedOptimizer":
        raise ValueError(
            f"Expected CombinedOptimizer, got {optimizer.__class__.__name__}"
        )
    contained = tuple(
        contained_optimizer.__class__.__name__
        for contained_optimizer in getattr(optimizer, "optimizers", ())
    )
    if contained != ("Muon", "AdamW"):
        raise ValueError(f"Combined optimizer members changed: {contained}")

    model_parameters = list(model.parameters())
    expected_parameter_ids = (
        [id(parameter) for parameter in model_parameters if parameter.ndim == 2],
        [id(parameter) for parameter in model_parameters if parameter.ndim != 2],
    )
    observed_parameter_ids = tuple(
        [
            id(parameter)
            for group in contained_optimizer.param_groups
            for parameter in group["params"]
        ]
        for contained_optimizer in optimizer.optimizers
    )
    if observed_parameter_ids != expected_parameter_ids:
        raise ValueError(
            "Combined optimizer parameter partition differs from the exact "
            "2D-to-Muon/non-2D-to-AdamW recipe contract"
        )
    if len(set(observed_parameter_ids[0]) | set(observed_parameter_ids[1])) != len(
        model_parameters
    ) or set(observed_parameter_ids[0]).intersection(observed_parameter_ids[1]):
        raise ValueError("Combined optimizer parameter partition is not exhaustive")

    state_entries = tuple(
        len(contained_optimizer.state) for contained_optimizer in optimizer.optimizers
    )
    if regime == "fresh_seed42" and state_entries != (0, 0):
        raise ValueError(
            f"Fresh optimizer states are unexpectedly nonempty: {state_entries}"
        )
    if regime == "checkpoint_epoch491" and any(
        entries == 0 for entries in state_entries
    ):
        raise ValueError(
            f"Checkpoint optimizer state is incomplete across members: {state_entries}"
        )
    model.train()
    stochastic = _stochastic_modules(model)
    if stochastic:
        raise ValueError(f"Active stochastic modules are forbidden: {stochastic}")
    return model, optimizer, loaded_epoch


def _learning_rate(optimizer: torch.optim.Optimizer) -> float:
    values = {float(group["lr"]) for group in optimizer.param_groups}
    if len(values) != 1:
        raise ValueError(f"Optimizer parameter groups have different LRs: {values}")
    value = values.pop()
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Optimizer learning rate is invalid: {value}")
    return value


def _raw_source_sha256(raw_mesh: Any, selected_ids: np.ndarray) -> str:
    return _stable_sha256(
        {
            "points": raw_mesh.points,
            "cells": raw_mesh.cells,
            "selected_ids": selected_ids,
        }
    )


def _global_inputs_sha256(legacy: ModuleType, domain: Any) -> str:
    return _stable_sha256(legacy._pipeline_globals_float32(domain))


def _batch_order_sha256(case_id: str, selected_ids: np.ndarray) -> str:
    return _stable_sha256({"case_id": case_id, "selected_ids": selected_ids})


def _prepare_case(
    *,
    legacy: ModuleType,
    runtime_support: ModuleType,
    canonical: ModuleType,
    runtime: Any,
    dataset_root: Path,
    spec: Any,
    geometry_case: Mapping[str, Any],
    target_case: Mapping[str, Any],
) -> dict[str, Any]:
    from physicsnemo.datapipes.transforms.mesh import (
        TARGET_QUADRATURE_MEASURE_KEY,
    )

    if TARGET_QUADRATURE_MEASURE_KEY != "_target_quadrature_measure":
        raise ValueError("Private target quadrature measure key changed")
    raw_mesh, input_arrays, _target_hashes = legacy._load_explicit_raw_subset(
        runtime,
        dataset_root,
        spec,
        geometry_case,
        target_case,
    )
    raw_canonical = canonical._build_canonical_raw_geometry(raw_mesh)
    physical_length = canonical._nested_tensor_value(raw_mesh.global_data, "L_ref")
    domain_with_targets, _pipeline_center = runtime_support._apply_pipeline(
        runtime,
        raw_mesh,
        fixed_center=None,
    )
    boundary = domain_with_targets.boundaries["vehicle"]
    if "_measure_weights" in boundary.cell_data.keys():
        raise ValueError(f"{spec.case_id} unexpectedly carries source weights")
    if tuple(boundary.cells.shape) != (RESOLUTION, 3):
        raise ValueError(f"{spec.case_id} boundary topology changed")

    batch = runtime.collate_fn([(domain_with_targets, {})])
    if set(batch["forward_kwargs"]) != {"domain"}:
        raise ValueError(f"{spec.case_id} forward kwargs changed")
    targets = batch["targets"].float()
    batch_target_measure = batch.get("target_measure")
    preserved_target_measure = domain_with_targets.interior.point_data.get(
        TARGET_QUADRATURE_MEASURE_KEY
    )
    if batch_target_measure is None or preserved_target_measure is None:
        raise ValueError(f"{spec.case_id} target measure is missing")
    if not _tensor_raw_equal(batch_target_measure, preserved_target_measure):
        raise ValueError(
            f"{spec.case_id} collated target measure differs bytewise from "
            "the preserved domain quadrature measure"
        )
    target_measure = batch_target_measure.float()
    if (
        tuple(targets["pressure"].shape) != (RESOLUTION,)
        or tuple(targets["wss"].shape) != (RESOLUTION, 3)
        or tuple(target_measure.shape) != (RESOLUTION,)
        or not bool(torch.isfinite(targets["pressure"]).all())
        or not bool(torch.isfinite(targets["wss"]).all())
        or not bool(torch.isfinite(target_measure).all())
        or not bool((target_measure > 0.0).all())
    ):
        raise ValueError(f"{spec.case_id} target tensors changed")

    domain = canonical._strip_local_data(domain_with_targets, runtime.mesh_type)
    if "_measure_weights" in domain.boundaries["vehicle"].cell_data.keys():
        raise ValueError(f"{spec.case_id} stripped model domain carries weights")
    reference_key = runtime.model.reference_length_key
    if reference_key is None:
        raise ValueError("One-step parity requires explicit reference_length_key")
    model_reference_length = canonical._nested_tensor_value(
        domain.global_data,
        reference_key,
    )
    bundle = canonical._finish_canonical_bundle(
        raw_canonical,
        physical_length=physical_length,
        model_reference_length=model_reference_length,
    )
    validity = canonical._bundle_validity(bundle, expected_cells=boundary.cells)
    if not validity["passed"]:
        raise ValueError(f"{spec.case_id} canonical geometry is invalid")

    selected_ids = np.asarray(input_arrays["selected_cell_ids_int64"], dtype="<i8")
    if (
        selected_ids.shape != (RESOLUTION,)
        or np.unique(selected_ids).size != RESOLUTION
        or int(selected_ids.min()) < 0
    ):
        raise ValueError(f"{spec.case_id} selected IDs changed")
    return {
        "domain": domain,
        "bundle": bundle,
        "targets": targets,
        "target_measure": target_measure,
        "selected_ids": selected_ids,
        "target_pressure": (
            targets["pressure"].detach().cpu().numpy().astype("<f4", copy=False)
        ),
        "target_wss": targets["wss"].detach().cpu().numpy().astype("<f4", copy=False),
        "target_measure_array": (
            target_measure.detach().cpu().numpy().astype("<f4", copy=False)
        ),
        "raw_source_geometry_sha256": _raw_source_sha256(raw_mesh, selected_ids),
        "global_inputs_sha256": _global_inputs_sha256(legacy, domain),
        "batch_order_sha256": _batch_order_sha256(spec.case_id, selected_ids),
    }


def _run_arm(
    *,
    runtime: Any,
    canonical_support: ModuleType,
    case: Mapping[str, Any],
    regime: str,
    precision: str,
    arm: str,
    checkpoint_dir: Path,
    expected_layout: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    from loss import LossCalculator

    model, optimizer, loaded_epoch = _new_model_optimizer(
        runtime,
        regime=regime,
        checkpoint_dir=checkpoint_dir,
    )
    layout, layout_arrays = _parameter_layout(model)
    if expected_layout is not None and layout != expected_layout:
        raise ValueError("Parameter layout changed across execution cells")

    parameter_names = list(layout["parameter_names"])
    initial_parameter_hash = _stable_sha256(model.state_dict())
    initial_optimizer_hash = _stable_sha256(optimizer.state_dict())
    rng_hash = _rng_state_sha256()
    learning_rate = _learning_rate(optimizer)
    before = _flatten_parameters(model)
    if before.numel() != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Initial flattened parameter count changed")

    loss_calculator = LossCalculator(
        TARGET_CONFIG,
        loss_type="huber",
        n_spatial_dims=3,
        normalize_by_channels=True,
    )
    optimizer.zero_grad(set_to_none=True)
    domain = case["domain"]
    with runtime.autocast_context(precision):
        if arm == "legacy":
            output = model(domain)
        elif arm == "canonical":
            geometry = _canonical_geometry_for_model(
                model,
                runtime,
                domain,
                case["bundle"],
            )
            encoded = model.encode(
                domain,
                canonical_source_geometry=geometry,
            )
            if not (
                _tensor_raw_equal(encoded.source_mesh.points, geometry.points)
                and _tensor_raw_equal(encoded.source_mesh.cells, geometry.cells)
                and _tensor_raw_equal(
                    encoded.source_mesh.cell_centroids, geometry.centroids
                )
                and _tensor_raw_equal(encoded.source_mesh.cell_areas, geometry.areas)
                and _tensor_raw_equal(
                    encoded.source_mesh.cell_normals, geometry.normals
                )
            ):
                raise ValueError("Canonical geometry was not installed byte-exactly")
            query_mesh = runtime.mesh_type(points=geometry.centroids)
            output = model.decode(encoded, query_mesh)
        else:
            raise ValueError(f"Unknown arm {arm}")

    prediction = runtime.normalize_output(
        output,
        TARGET_CONFIG,
        str(runtime.cfg.output_type),
    ).float()
    targets = case["targets"].float()
    target_measure = case["target_measure"].float()
    loss, _loss_fields = loss_calculator(
        prediction,
        targets,
        target_measure,
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError(f"{regime}/{precision}/{arm} loss is non-finite")
    loss.backward()
    gradient = _flatten_gradients(model)
    if gradient.numel() != EXPECTED_PARAMETER_COUNT or not bool(
        torch.isfinite(gradient).all()
    ):
        raise ValueError(f"{regime}/{precision}/{arm} gradient is invalid")
    optimizer.step()
    after = _flatten_parameters(model)
    update = after - before
    if not bool(torch.isfinite(update).all()):
        raise ValueError(f"{regime}/{precision}/{arm} update is non-finite")

    pressure = prediction["pressure"].detach().cpu()
    wss = prediction["wss"].detach().cpu()
    if tuple(pressure.shape) != (RESOLUTION,) or tuple(wss.shape) != (
        RESOLUTION,
        3,
    ):
        raise ValueError(f"{regime}/{precision}/{arm} prediction shape changed")
    if not bool(torch.isfinite(pressure).all()) or not bool(torch.isfinite(wss).all()):
        raise ValueError(f"{regime}/{precision}/{arm} prediction is non-finite")

    arrays = {
        "prediction_pressure_float32": pressure.numpy().astype("<f4", copy=False),
        "prediction_wss_float32": wss.numpy().astype("<f4", copy=False),
        "loss_float64": np.asarray([float(loss.detach().item())], dtype="<f8"),
        "gradient_float32": gradient.numpy().astype("<f4", copy=False),
        "parameter_update_float32": update.numpy().astype("<f4", copy=False),
        "learning_rate_float64": np.asarray([learning_rate], dtype="<f8"),
        "selected_cell_ids_int64": np.asarray(case["selected_ids"], dtype="<i8"),
        "target_pressure_float32": np.asarray(case["target_pressure"], dtype="<f4"),
        "target_wss_float32": np.asarray(case["target_wss"], dtype="<f4"),
        "target_measure_float32": np.asarray(case["target_measure_array"], dtype="<f4"),
    }
    if tuple(arrays) != VECTOR_FIELDS:
        raise RuntimeError("Producer vector-field schema changed")
    control = {
        "raw_source_geometry_sha256": case["raw_source_geometry_sha256"],
        "global_inputs_sha256": case["global_inputs_sha256"],
        "initial_parameter_state_sha256": initial_parameter_hash,
        "initial_optimizer_state_sha256": initial_optimizer_hash,
        "rng_state_sha256": rng_hash,
        "parameter_order_sha256": _parameter_order_sha256(parameter_names),
        "batch_order_sha256": case["batch_order_sha256"],
        "source_measure_weights_present": False,
        "target_measure_present": True,
    }
    runtime_record = {
        "layout": layout,
        "loaded_epoch": loaded_epoch,
        "optimizer_members": ["Muon", "AdamW"],
        "compile_enabled": False,
    }
    del model, optimizer, prediction, output, loss
    torch.cuda.empty_cache()
    return arrays, control, {"layout_arrays": layout_arrays, **runtime_record}


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _directory(path: Path, label: str, *, allow_symlink: bool = False) -> Path:
    if (path.is_symlink() and not allow_symlink) or not path.is_dir():
        raise ValueError(f"{label} must be a valid directory: {path}")
    return path.resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("One-step parity requires a little-endian host")
    args = _parse_args(argv)
    _validate_single_rank_environment()
    args.repo_root = _directory(args.repo_root, "Repository root")
    args.dataset_root = _directory(args.dataset_root, "Dataset root")
    args.checkpoint_dir = _directory(args.checkpoint_dir, "Checkpoint directory")
    for name in (
        "dataset_config",
        "resolved_config",
        "geometry_manifest",
        "target_input_manifest",
    ):
        setattr(args, name, _regular_file(getattr(args, name), name.replace("_", " ")))
    args.output_json = Path(os.path.abspath(args.output_json))
    args.output_npz = Path(os.path.abspath(args.output_npz))

    script_path = Path(__file__).resolve(strict=True)
    legacy, runtime_support, canonical = _load_support_modules(script_path)
    legacy._validate_case_specs(runtime_support)
    static_inputs = legacy._validate_static_inputs(
        runtime_support,
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config=args.dataset_config,
        resolved_config=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    if static_inputs["Current execution source tree"] != EXPECTED_SOURCE_TREE_SHA256:
        raise ValueError("Execution source tree differs")
    geometry_verification = legacy._verify_geometry_manifest(
        runtime_support,
        args.geometry_manifest,
        args.dataset_root,
    )
    geometry_cases = geometry_verification.pop("case_records")
    target_verification = legacy._verify_target_input_manifest(
        args.target_input_manifest,
        args.dataset_root,
    )
    target_cases = target_verification.pop("case_records")

    runtime = runtime_support._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    legacy._validate_import_provenance(args.repo_root)
    legacy._validate_reader(runtime)
    if bool(runtime.cfg.compile) is not True:
        raise ValueError("Historical resolved config no longer declares compile=true")
    if str(runtime.cfg.precision) != "bfloat16":
        raise ValueError("Historical resolved precision changed")

    geometry_by_id = {str(record["case_id"]): record for record in geometry_cases}
    target_by_id = {str(record["case_id"]): record for record in target_cases}
    helper_by_id = {spec.case_id: spec for spec in runtime_support.CASE_SPECS}
    if set(geometry_by_id) != set(helper_by_id) or set(target_by_id) != set(
        helper_by_id
    ):
        raise ValueError("Frozen input manifest case sets differ")

    prepared_cases = {
        case_id: _prepare_case(
            legacy=legacy,
            runtime_support=runtime_support,
            canonical=canonical,
            runtime=runtime,
            dataset_root=args.dataset_root,
            spec=helper_by_id[case_id],
            geometry_case=geometry_by_id[case_id],
            target_case=target_by_id[case_id],
        )
        for _ordinal, case_id in CASE_SPECS
    }

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    parameter_layout: dict[str, Any] | None = None
    runtime_attestations: list[dict[str, Any]] = []
    for regime in REGIMES:
        for precision in PRECISIONS:
            for ordinal, case_id in CASE_SPECS:
                arm_controls: dict[str, Any] = {}
                for arm in ARMS:
                    arm_arrays, control, attestation = _run_arm(
                        runtime=runtime,
                        canonical_support=canonical,
                        case=prepared_cases[case_id],
                        regime=regime,
                        precision=precision,
                        arm=arm,
                        checkpoint_dir=args.checkpoint_dir,
                        expected_layout=parameter_layout,
                    )
                    if parameter_layout is None:
                        parameter_layout = attestation["layout"]
                        arrays.update(attestation["layout_arrays"])
                    elif attestation["layout"] != parameter_layout:
                        raise ValueError("Parameter layout changes across arms")
                    if set(attestation["layout_arrays"]) != set(PARAMETER_LAYOUT_KEYS):
                        raise RuntimeError("Parameter-layout array schema changed")
                    for name in PARAMETER_LAYOUT_KEYS:
                        if not np.array_equal(
                            arrays[name], attestation["layout_arrays"][name]
                        ):
                            raise ValueError("Parameter layout changes across arms")
                    prefix = _prefix(regime, precision, ordinal, case_id, arm)
                    for field, value in arm_arrays.items():
                        arrays[f"{prefix}{field}"] = np.ascontiguousarray(value)
                    arm_controls[arm] = control
                    runtime_attestations.append(
                        {
                            "regime": regime,
                            "precision": precision,
                            "case_ordinal": ordinal,
                            "case_id": case_id,
                            "arm": arm,
                            "loaded_epoch": attestation["loaded_epoch"],
                            "optimizer_members": attestation["optimizer_members"],
                            "compile_enabled": attestation["compile_enabled"],
                        }
                    )
                if arm_controls["legacy"] != arm_controls["canonical"]:
                    raise ValueError(
                        f"{regime}/{precision}/{case_id} arm controls differ"
                    )
                records.append(
                    {
                        "regime": regime,
                        "precision": precision,
                        "case_ordinal": ordinal,
                        "case_id": case_id,
                        "arm_controls": arm_controls,
                    }
                )

    if parameter_layout is None:
        raise RuntimeError("No parameter layout was produced")
    expected_array_count = len(PARAMETER_LAYOUT_KEYS) + (
        len(REGIMES)
        * len(PRECISIONS)
        * len(CASE_SPECS)
        * len(ARMS)
        * len(VECTOR_FIELDS)
    )
    if len(records) != 16 or len(arrays) != expected_array_count:
        raise RuntimeError(
            f"Producer coverage changed: records={len(records)} "
            f"arrays={len(arrays)} expected_arrays={expected_array_count}"
        )

    npz_temporary, npz_sha256 = legacy._prepare_npz_temporary(
        args.output_npz,
        arrays,
    )
    try:
        provenance = {
            "producer_sha256": _sha256_file(script_path),
            "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "resolved_config_sha256": EXPECTED_RESOLVED_CONFIG_SHA256,
            "model_checkpoint_sha256": EXPECTED_MODEL_CHECKPOINT_SHA256,
            "training_state_sha256": EXPECTED_TRAINING_STATE_SHA256,
            "normalization_state_sha256": EXPECTED_NORMALIZATION_STATE_SHA256,
            "npz_sha256": npz_sha256,
        }
        contract = {
            "case_ordinals": [ordinal for ordinal, _ in CASE_SPECS],
            "case_ids": [case_id for _, case_id in CASE_SPECS],
            "regimes": list(REGIMES),
            "precisions": list(PRECISIONS),
            "arms": list(ARMS),
            "resolution": RESOLUTION,
            "fresh_seed": FRESH_SEED,
            "checkpoint_epoch": CHECKPOINT_EPOCH,
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
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": STATUS,
            "contract": contract,
            "parameter_layout": parameter_layout,
            "records": records,
            "array_manifest": _array_manifest(arrays),
            "provenance": provenance,
        }
        # Runtime attestations are deliberately checked locally rather than
        # extending the reducer-facing schema.
        if len(runtime_attestations) != 32 or any(
            row["optimizer_members"] != ["Muon", "AdamW"]
            or row["compile_enabled"] is not False
            or (
                row["loaded_epoch"] != CHECKPOINT_EPOCH
                if row["regime"] == "checkpoint_epoch491"
                else row["loaded_epoch"] is not None
            )
            for row in runtime_attestations
        ):
            raise ValueError("Runtime regime/optimizer attestation changed")
        payload = (
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        digest = legacy._publish_output_set(
            output_json=args.output_json,
            json_payload=payload,
            output_npz=args.output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )
    finally:
        npz_temporary.unlink(missing_ok=True)
    print(
        f"{STATUS} records={len(records)} arrays={len(arrays)} "
        f"json_sha256={digest} npz_sha256={npz_sha256} "
        f"generated_at={datetime.now(timezone.utc).isoformat()} "
        f"host={platform.node()} pid={os.getpid()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
