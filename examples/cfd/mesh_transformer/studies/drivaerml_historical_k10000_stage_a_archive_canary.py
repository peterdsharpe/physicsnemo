# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce one target-free Stage-A archive-domain replay lane.

The fixed ``run_118`` archive is used only as a post-pipeline input oracle.
This producer opens a staged input-only tree containing boundary/query geometry
and the seven required global fields, each bound to a preregistered SHA-256.
It never opens the oracle-bearing full manifest, input-freeze record, archived
prediction, truth, target, point-data, or cell-data payload.  The independent
reducer is the only process allowed to inspect archived predictions or publish
a categorical replay result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import stat
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_historical_k10000_stage_a_archive_domain_producer"
STATUS = "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_PRODUCER"

CASE_ID = "run_118"
READER_INDEX = 21
RESOLUTION = 10_000
BOUNDARY_POINT_COUNT = 29_949
CASE_DIRECTORY = "00021_run_118_domain_run_118.pdmsh"
PRECISION = "bfloat16"
TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}
FIELD_TYPES = {"pressure": "pressure", "wss": "stress"}

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

MODEL_FILENAME = "MeshTransformer.0.491.mdlus"
TRAINING_STATE_FILENAME = "checkpoint.0.491.pt"
NORM_STATS_FILENAME = "norm_stats.pt"
EPOCH = 491
PARAMETER_COUNT = 1_278_268

SOURCE_TREE_ROOTS = (
    Path("physicsnemo/experimental/nn/mesh_attention"),
    Path("physicsnemo/experimental/nn/symmetry"),
    Path("physicsnemo/mesh"),
    Path("physicsnemo/datapipes"),
)
RECIPE_SOURCE_ROOT = Path(
    "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
)

ARCHIVE_ARRAY_SPECS: Mapping[str, tuple[Path, tuple[int, ...], np.dtype[Any]]] = {
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
EXPECTED_ARCHIVE_INPUT_SHA256: Mapping[str, str] = {
    "archive_boundary_points_float32": (
        "01f7e493a9ae6ce5bf514d12a608855a6ce13ba559fd3b1859aca3635c420a46"
    ),
    "archive_boundary_cells_int64": (
        "a66df151ecc02c04d412bd20622a0c32115c7c87f9c899380ffea5a0cfd08e0d"
    ),
    "archive_query_points_float32": (
        "77bb36bb03e7f983d0763fa157ba2c61c234e4582475d3b220b0d44a572c082e"
    ),
    "archive_global_L_ref_float32": (
        "fca31f1667a6aa1bba12fca4e4ea1becd503379d80da3213af07f6cc5702828d"
    ),
    "archive_global_U_inf_float32": (
        "527665492391c17b8dd4486fd555390eadf800b9f9b60b7b4cd79e9e5ea9e9f0"
    ),
    "archive_global_U_inf_dir_float32": (
        "480376c6bf738a0227f2bbf2b3506b7cde209152c0ba9a9077e5527169eb292e"
    ),
    "archive_global_nu_float32": (
        "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c"
    ),
    "archive_global_p_inf_float32": (
        "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    ),
    "archive_global_reference_length_float32": (
        "03e3c2420f5066a5fa6e36735ed8cc4f6a251046263e1a6024f009deeee3b952"
    ),
    "archive_global_rho_inf_float32": (
        "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c"
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

DIRECT_ARCHIVE_ARRAY_NAMES = tuple(ARCHIVE_ARRAY_SPECS) + tuple(
    f"archive_global_{name}_float32" for name in GLOBAL_SHAPES
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

_FORBIDDEN_ARCHIVE_PARTS = frozenset({"point_data", "cell_data"})
_FORBIDDEN_ARCHIVE_TOKENS = ("pred_", "true_", "pressure", "wss", "target")


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    model: Any
    normalize_output: Callable[..., Any]
    redimensionalize: Callable[..., Any]
    normalizer: Any
    nondim: Any
    autocast_context: Callable[[str], AbstractContextManager[Any]]
    loaded_epoch: int
    provenance: Mapping[str, Any]


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


def _require_sha256(path: Path, expected: str, label: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 differs: expected={expected} observed={observed}"
        )


def _require_allowed_producer_path(relative: Path) -> None:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    if any(part in _FORBIDDEN_ARCHIVE_PARTS for part in lowered_parts):
        raise ValueError(
            f"Producer archive path enters a forbidden subtree: {relative}"
        )
    name = relative.name.lower()
    if any(token in name for token in _FORBIDDEN_ARCHIVE_TOKENS):
        raise ValueError(f"Producer archive path names a forbidden oracle: {relative}")


def _read_frozen_input_payload(
    archive_root: Path,
    relative: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, str]:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive-relative path: {relative}")
    _require_allowed_producer_path(relative)
    path = archive_root / relative
    payload = _safe_read_bytes(path)
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError(
            f"Archive payload changed: {relative}; "
            f"expected={expected_sha256} observed={observed}"
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
    array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array.copy(order="C")


def _load_archive_inputs(
    archive_root: Path,
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_ARCHIVE_INPUT_SHA256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if archive_root.is_symlink():
        raise ValueError(f"Historical archive root is a symlink: {archive_root}")
    if not archive_root.exists():
        raise FileNotFoundError(f"Historical archive root is missing: {archive_root}")
    if not archive_root.is_dir():
        raise ValueError(f"Historical archive root is not a directory: {archive_root}")
    if set(expected_hashes) != set(DIRECT_ARCHIVE_ARRAY_NAMES):
        raise ValueError("Frozen Stage-A archive-input hash inventory changed")
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for name, (relative, shape, dtype) in ARCHIVE_ARRAY_SPECS.items():
        payload, digest = _read_frozen_input_payload(
            archive_root,
            relative,
            expected_sha256=expected_hashes[name],
        )
        arrays[name] = _array_from_payload(
            payload,
            shape=shape,
            dtype=dtype,
            label=name,
        )
        records[name] = {
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(payload),
        }

    global_root = Path(f"{CASE_DIRECTORY}/_tensordict/global_data")
    for field, shape in GLOBAL_SHAPES.items():
        name = f"archive_global_{field}_float32"
        relative = global_root / f"{field}.memmap"
        payload, digest = _read_frozen_input_payload(
            archive_root,
            relative,
            expected_sha256=expected_hashes[name],
        )
        arrays[name] = _array_from_payload(
            payload,
            shape=shape,
            dtype=np.dtype("<f4"),
            label=name,
        )
        records[name] = {
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(payload),
        }

    cells = arrays["archive_boundary_cells_int64"]
    if int(cells.min()) < 0 or int(cells.max()) >= BOUNDARY_POINT_COUNT:
        raise ValueError("Archived boundary connectivity is out of range")
    if float(arrays["archive_global_reference_length_float32"]) != 8.0:
        raise ValueError("Archived model reference length changed")
    if float(arrays["archive_global_L_ref_float32"]) != 5.0:
        raise ValueError("Archived physical reference length changed")
    return arrays, {
        "historical_manifest_sha256": EXPECTED_HISTORICAL_MANIFEST_SHA256,
        "historical_manifest_opened": False,
        "input_freeze_record_sha256": EXPECTED_INPUT_FREEZE_SHA256,
        "input_freeze_record_opened": False,
        "input_hash_binding": "embedded_sha256_constants",
        "opened_payload_count": len(records),
        "opened_payloads": records,
    }


def _require_stripped_domain(domain: Any) -> None:
    associations = {
        "interior.point_data": domain.interior.point_data,
        "interior.cell_data": domain.interior.cell_data,
    }
    for boundary_name, boundary in domain.boundaries.items():
        associations[f"boundaries.{boundary_name}.point_data"] = boundary.point_data
        associations[f"boundaries.{boundary_name}.cell_data"] = boundary.cell_data
    nonempty = {
        name: sorted(str(key) for key in data.keys())
        for name, data in associations.items()
        if len(data.keys()) != 0
    }
    if nonempty:
        raise ValueError(f"Stage-A domain retains local fields: {nonempty}")
    global_keys = {str(key) for key in domain.global_data.keys()}
    if "_measure_weights" in global_keys:
        raise ValueError("Stage-A domain retains global measure weights")


def _build_stripped_domain(arrays: Mapping[str, np.ndarray]) -> Any:
    from physicsnemo.mesh import DomainMesh, Mesh

    boundary = Mesh(
        points=torch.from_numpy(
            np.array(arrays["archive_boundary_points_float32"], copy=True)
        ),
        cells=torch.from_numpy(
            np.array(arrays["archive_boundary_cells_int64"], copy=True)
        ),
        point_data={},
        cell_data={},
        global_data={},
    )
    interior = Mesh(
        points=torch.from_numpy(
            np.array(arrays["archive_query_points_float32"], copy=True)
        ),
        point_data={},
        cell_data={},
        global_data={},
    )
    globals_ = {
        field: torch.from_numpy(
            np.array(arrays[f"archive_global_{field}_float32"], copy=True)
        )
        for field in GLOBAL_SHAPES
    }
    domain = DomainMesh(
        interior=interior,
        boundaries={"vehicle": boundary},
        global_data=globals_,
    )
    _require_stripped_domain(domain)
    if domain.interior.n_points != RESOLUTION or boundary.n_cells != RESOLUTION:
        raise ValueError("Stage-A domain resolution changed")
    return domain


def _derive_encoded_geometry(domain: Any) -> dict[str, np.ndarray]:
    """Mirror the frozen model's default source-geometry construction."""
    from physicsnemo.mesh import Mesh

    _require_stripped_domain(domain)
    boundary = domain.boundaries["vehicle"]
    merged = Mesh.merge(
        [boundary.with_data(point_data={}, cell_data={}, global_data={})]
    )
    length = domain.global_data["reference_length"].reshape(())
    if length.dtype != merged.points.dtype or length.device != merged.points.device:
        raise ValueError("Reference length does not share source dtype/device")
    with torch.autocast(device_type=merged.points.device.type, enabled=False):
        areas = merged.cell_areas
        if not bool(torch.isfinite(areas).all()) or bool(torch.any(areas <= 0.0)):
            raise ValueError("Archive source contains a degenerate boundary cell")
        total_area = areas.sum()
        center = torch.einsum("n,nd->d", areas, merged.cell_centroids) / total_area
        source = Mesh(
            points=(merged.points - center) / length,
            cells=merged.cells,
        )
    return _encoded_geometry_arrays(
        source=source,
        center=center,
        reference_length=length,
    )


def _encoded_geometry_arrays(
    *,
    source: Any,
    center: torch.Tensor,
    reference_length: torch.Tensor,
) -> dict[str, np.ndarray]:
    if (
        len(source.point_data.keys()) != 0
        or len(source.cell_data.keys()) != 0
        or len(source.global_data.keys()) != 0
    ):
        raise ValueError("Encoded Stage-A source unexpectedly retains data fields")
    tensors = {
        "encoded_source_points_float32": source.points,
        "encoded_source_cells_int64": source.cells,
        "encoded_source_centroids_float32": source.cell_centroids,
        "encoded_source_areas_float32": source.cell_areas,
        "encoded_source_normals_float32": source.cell_normals,
        "encoded_center_float32": center,
        "encoded_reference_length_float32": reference_length,
    }
    arrays = {}
    for name, tensor in tensors.items():
        dtype = ENCODED_ARRAY_SCHEMAS[name][1]
        arrays[name] = np.array(
            tensor.detach().cpu().numpy(),
            dtype=dtype,
            copy=True,
            order="C",
        )
    for name, (shape, dtype) in ENCODED_ARRAY_SCHEMAS.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Derived encoded geometry schema changed for {name}")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Derived encoded geometry is non-finite for {name}")
    return arrays


def _model_forward_once(
    model: Any,
    domain: Any,
    autocast_context: Callable[[str], AbstractContextManager[Any]],
) -> Any:
    """Execute the sole model call; the call shape is intentionally positional."""
    with torch.no_grad(), autocast_context(PRECISION):
        return model(domain)


def _captured_model_forward_once(
    model: Any,
    domain: Any,
    autocast_context: Callable[[str], AbstractContextManager[Any]],
) -> tuple[Any, Any]:
    """Capture the actual EncodedBoundary created inside the sole forward."""
    if "encode" in model.__dict__:
        raise ValueError("Stage-A model unexpectedly shadows encode on the instance")
    original_encode = model.encode
    captured: list[Any] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        if len(args) != 1 or args[0] is not domain or kwargs:
            raise ValueError(
                "Stage-A forward did not call encode with one positional domain"
            )
        encoded = original_encode(*args)
        captured.append(encoded)
        return encoded

    model.encode = capture
    try:
        output = _model_forward_once(model, domain, autocast_context)
    finally:
        delattr(model, "encode")
    if len(captured) != 1:
        raise ValueError(
            f"Stage-A model encoded {len(captured)} domains, expected exactly one"
        )
    return output, captured[0]


def _predict_physical(
    runtime: Runtime,
    domain: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if domain.interior.points.device != runtime.device:
        raise ValueError("Stage-A domain is not on the frozen runtime device")
    _require_stripped_domain(domain)
    output, encoded = _captured_model_forward_once(
        runtime.model,
        domain,
        runtime.autocast_context,
    )
    prediction = runtime.normalize_output(output, TARGET_CONFIG, "mesh")
    physical = runtime.redimensionalize(
        prediction,
        normalizer=runtime.normalizer,
        nondim=runtime.nondim,
        field_types=FIELD_TYPES,
        global_data=domain.global_data,
    )
    arrays = {
        "prediction_pressure_physical_float32": np.ascontiguousarray(
            physical["pressure"].detach().float().cpu().numpy(), dtype="<f4"
        ),
        "prediction_wss_physical_float32": np.ascontiguousarray(
            physical["wss"].detach().float().cpu().numpy(), dtype="<f4"
        ),
    }
    for name, (shape, dtype) in PREDICTION_ARRAY_SCHEMAS.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Stage-A prediction {name} has {value.shape}/{value.dtype}, "
                f"expected {shape}/{dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Stage-A prediction is non-finite for {name}")
    encoded_arrays = _encoded_geometry_arrays(
        source=encoded.source_mesh,
        center=encoded.center,
        reference_length=encoded.reference_length,
    )
    return arrays, encoded_arrays


def _source_tree_manifest_sha256(repo_root: Path) -> str:
    files: list[Path] = []
    for root in SOURCE_TREE_ROOTS:
        files.extend(
            path for path in (repo_root / root).rglob("*.py") if path.is_file()
        )
    recipe_root = repo_root / RECIPE_SOURCE_ROOT
    files.extend(path for path in recipe_root.glob("*.py") if path.is_file())
    relative = sorted(path.relative_to(repo_root).as_posix() for path in files)
    if not relative:
        raise FileNotFoundError(f"No source files found below {repo_root}")
    digest = hashlib.sha256()
    for name in relative:
        digest.update(f"{_sha256_file(repo_root / name)}  {name}\n".encode())
    return digest.hexdigest()


def _load_frozen_checkpoint(
    load_checkpoint: Callable[..., int],
    *,
    checkpoint_dir: Path,
    model: Any,
    device: torch.device,
) -> int:
    """Load the exact preregistered epoch, independent of sibling checkpoints."""
    loaded_epoch = int(
        load_checkpoint(
            path=str(checkpoint_dir),
            models=model,
            device=device,
            epoch=EPOCH,
        )
    )
    if loaded_epoch != EPOCH:
        raise ValueError(f"Loaded epoch {loaded_epoch}, expected {EPOCH}")
    return loaded_epoch


def _load_runtime(
    *,
    repo_root: Path,
    resolved_config: Path,
    dataset_config: Path,
    checkpoint_dir: Path,
) -> Runtime:
    checks = {
        "resolved_config": (resolved_config, EXPECTED_RESOLVED_CONFIG_SHA256),
        "dataset_config": (dataset_config, EXPECTED_DATASET_CONFIG_SHA256),
        "model_checkpoint": (
            checkpoint_dir / MODEL_FILENAME,
            EXPECTED_MODEL_SHA256,
        ),
        "training_state": (
            checkpoint_dir / TRAINING_STATE_FILENAME,
            EXPECTED_TRAINING_STATE_SHA256,
        ),
        "normalization_state": (
            checkpoint_dir / NORM_STATS_FILENAME,
            EXPECTED_NORMALIZATION_SHA256,
        ),
        "current_infer_source": (
            repo_root / RECIPE_SOURCE_ROOT / "infer.py",
            EXPECTED_CURRENT_INFER_SHA256,
        ),
        "current_model_source": (
            repo_root / "physicsnemo/experimental/nn/mesh_attention/model.py",
            EXPECTED_CURRENT_MODEL_SOURCE_SHA256,
        ),
    }
    for label, (path, digest) in checks.items():
        _require_sha256(path, digest, label)
    source_tree_sha256 = _source_tree_manifest_sha256(repo_root)
    if source_tree_sha256 != EXPECTED_CURRENT_SOURCE_TREE_SHA256:
        raise ValueError(
            "Current execution source tree changed: "
            f"expected={EXPECTED_CURRENT_SOURCE_TREE_SHA256} "
            f"observed={source_tree_sha256}"
        )

    recipe_src = (repo_root / RECIPE_SOURCE_ROOT).resolve(strict=True)
    sys.path.insert(0, str(recipe_src))
    import hydra
    import infer as recipe_infer
    from nondim import NonDimensionalizeByMetadata
    from omegaconf import OmegaConf
    from output_normalize import normalize_output_to_tensordict
    from utils import get_autocast_context, set_seed

    import physicsnemo
    from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.experimental.nn.mesh_attention import model as model_module
    from physicsnemo.utils import load_checkpoint

    expected_imports = {
        "physicsnemo": (repo_root / "physicsnemo/__init__.py").resolve(strict=True),
        "mesh_transformer_model": (
            repo_root / "physicsnemo/experimental/nn/mesh_attention/model.py"
        ).resolve(strict=True),
        "recipe_infer": (recipe_src / "infer.py").resolve(strict=True),
    }
    observed_imports = {
        "physicsnemo": Path(physicsnemo.__file__).resolve(strict=True),
        "mesh_transformer_model": Path(model_module.__file__).resolve(strict=True),
        "recipe_infer": Path(recipe_infer.__file__).resolve(strict=True),
    }
    if observed_imports != expected_imports:
        raise ImportError(
            "Stage-A imports did not resolve to the frozen execution tree: "
            f"expected={expected_imports} observed={observed_imports}"
        )
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        raise ValueError(
            f"Stage-A requires one process per lane, got world_size={dist.world_size}"
        )
    if dist.device.type != "cuda":
        raise RuntimeError(f"Stage-A requires CUDA, got {dist.device}")

    cfg = OmegaConf.load(resolved_config)
    if (
        str(cfg.precision) != PRECISION
        or str(cfg.input_type) != "mesh"
        or str(cfg.output_type) != "mesh"
        or OmegaConf.to_container(cfg.forward_kwargs, resolve=True) != {"domain": ""}
        or cfg.model.get("trace_of", None) != "vehicle"
        or cfg.model.get("reference_length_key", None) != "reference_length"
    ):
        raise ValueError("Frozen current model/runtime configuration changed")

    set_seed(42, rank=0)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dist.device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("Stage-A model parameter count changed")
    if (
        getattr(model, "trace_of", None) != "vehicle"
        or getattr(model, "reference_length_key", None) != "reference_length"
        or getattr(model, "measure_normalization", None) is not False
        or getattr(model, "trace_self_correction", None) is not True
        or getattr(model, "trace_readouts", None) is not True
    ):
        raise ValueError("Instantiated Stage-A model contract changed")
    loaded_epoch = _load_frozen_checkpoint(
        load_checkpoint,
        checkpoint_dir=checkpoint_dir,
        model=model,
        device=dist.device,
    )
    model.eval()

    normalizer = NormalizeMeshFields(
        association="point_data",
        stats_file=str(checkpoint_dir / NORM_STATS_FILENAME),
    ).to(dist.device)
    if set(normalizer.stats) != {"wss"}:
        raise ValueError("Frozen output normalization fields changed")
    nondim = NonDimensionalizeByMetadata(fields=FIELD_TYPES)
    return Runtime(
        device=dist.device,
        model=model,
        normalize_output=normalize_output_to_tensordict,
        redimensionalize=recipe_infer.redimensionalize,
        normalizer=normalizer,
        nondim=nondim,
        autocast_context=get_autocast_context,
        loaded_epoch=loaded_epoch,
        provenance={
            **{label: digest for label, (_, digest) in checks.items()},
            "current_execution_source_tree": source_tree_sha256,
            "checkpoint_load_epoch": EPOCH,
            "parameter_count": PARAMETER_COUNT,
            "model_seed": 42,
            "import_provenance": {
                name: str(path) for name, path in observed_imports.items()
            },
        },
    )


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
        }
        for name, value in arrays.items()
    }


def _npz_payload(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _output_destinations(outputs: Sequence[Path]) -> list[Path]:
    destinations: list[Path] = []
    for output in outputs:
        normalized = Path(os.path.abspath(os.path.normpath(output)))
        destinations.extend(
            (normalized, normalized.with_name(f"{normalized.name}.sha256"))
        )
    if len(set(destinations)) != len(destinations):
        raise ValueError("Output paths and sidecars must be pairwise distinct")
    if len({path.resolve(strict=False) for path in destinations}) != len(destinations):
        raise ValueError("Resolved output paths and sidecars alias")
    return destinations


def _validate_output_targets(outputs: Sequence[Path]) -> None:
    for destination in _output_destinations(outputs):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")


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


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in sorted(set(paths)):
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


def _publish_with_sidecars(payloads: Mapping[Path, bytes]) -> dict[str, str]:
    outputs = tuple(Path(os.path.abspath(path)) for path in payloads)
    _validate_output_targets(outputs)
    expanded: dict[Path, bytes] = {}
    digests: dict[str, str] = {}
    for output, payload in zip(outputs, payloads.values(), strict=True):
        digest = _sha256_bytes(payload)
        digests[output.name] = digest
        expanded[output] = payload
        sidecar = output.with_name(f"{output.name}.sha256")
        expanded[sidecar] = f"{digest}  {output.name}\n".encode("ascii")

    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for destination, payload in expanded.items():
            temporaries[destination] = _write_fsynced_temporary(
                destination,
                payload,
            )
        for destination, temporary in temporaries.items():
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        _fsync_directories([path.parent for path in temporaries])
        for destination, expected in expanded.items():
            if _safe_read_bytes(destination) != expected:
                raise OSError(f"Published artifact changed: {destination}")
    except BaseException:
        for destination, temporary in reversed(published):
            _unlink_if_same_inode(destination, temporary)
        if temporaries:
            _fsync_directories([path.parent for path in temporaries])
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        if temporaries:
            _fsync_directories([path.parent for path in temporaries])
    return digests


def _require_noncategorical_document(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "outcome" in str(key).lower():
                raise ValueError(
                    f"Producer document contains categorical key at {path}"
                )
            _require_noncategorical_document(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_noncategorical_document(child, f"{path}[{index}]")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--archive-input-root", type=Path, required=True)
    parser.add_argument("--lane-label", choices=("A", "B"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.repo_root = _directory(args.repo_root, "Repository root")
    args.checkpoint_dir = _directory(args.checkpoint_dir, "Checkpoint directory")
    args.archive_input_root = _directory(
        args.archive_input_root,
        "Stage-A archive input root",
    )
    for name in (
        "resolved_config",
        "dataset_config",
    ):
        setattr(
            args,
            name,
            _regular_file(getattr(args, name), name.replace("_", " ")),
        )
    args.output_json = Path(os.path.abspath(args.output_json))
    args.output_npz = Path(os.path.abspath(args.output_npz))
    _validate_output_targets((args.output_json, args.output_npz))

    direct_arrays, archive_record = _load_archive_inputs(args.archive_input_root)
    runtime = _load_runtime(
        repo_root=args.repo_root,
        resolved_config=args.resolved_config,
        dataset_config=args.dataset_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    domain = _build_stripped_domain(direct_arrays).to(runtime.device)
    prediction_arrays, encoded_arrays = _predict_physical(runtime, domain)
    arrays = {
        **direct_arrays,
        **encoded_arrays,
        **prediction_arrays,
    }
    expected_names = (
        list(DIRECT_ARCHIVE_ARRAY_NAMES)
        + list(ENCODED_ARRAY_SCHEMAS)
        + list(PREDICTION_ARRAY_SCHEMAS)
    )
    if list(arrays) != expected_names:
        raise RuntimeError("Stage-A NPZ array inventory changed")

    npz_payload = _npz_payload(arrays)
    npz_sha256 = _sha256_bytes(npz_payload)
    producer_path = Path(__file__).resolve(strict=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lane_label": args.lane_label,
        "contract": {
            "case_id": CASE_ID,
            "reader_index": READER_INDEX,
            "resolution": RESOLUTION,
            "precision": PRECISION,
            "compiled_model": False,
            "archive_is_input_oracle_only": True,
            "archived_predictions_opened": False,
            "archived_truth_opened": False,
            "raw_targets_opened": False,
            "historical_manifest_opened": False,
            "input_freeze_record_opened": False,
            "dataset_reader_constructed": False,
            "model_call": "model(domain)",
            "model_call_count": 1,
            "model_call_keyword_arguments": [],
            "canonical_source_geometry_supplied": False,
            "encoded_geometry_captured_from_single_forward": True,
            "local_data_fields_present": False,
            "measure_weights_present": False,
            "categorical_decision_present": False,
            "process_isolated_lane": True,
            "checkpoint_load_epoch": EPOCH,
        },
        "archive_inputs": archive_record,
        "npz": {
            "filename": args.output_npz.name,
            "sha256": npz_sha256,
            "array_count": len(arrays),
            "array_manifest": _array_manifest(arrays),
        },
        "provenance": {
            "command": list(sys.argv),
            "producer_path": str(producer_path),
            "producer_sha256": _sha256_file(producer_path),
            "repo_root": str(args.repo_root),
            "archive_input_root": str(args.archive_input_root),
            "loaded_epoch": runtime.loaded_epoch,
            "frozen_inputs": dict(runtime.provenance),
            "historical_manifest_sha256": EXPECTED_HISTORICAL_MANIFEST_SHA256,
            "input_freeze_record_sha256": EXPECTED_INPUT_FREEZE_SHA256,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(runtime.device),
            "process": {
                "pid": os.getpid(),
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
    }
    _require_noncategorical_document(document)
    json_payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    digests = _publish_with_sidecars(
        {
            args.output_json: json_payload,
            args.output_npz: npz_payload,
        }
    )
    print(
        f"{STATUS} lane={args.lane_label} "
        f"json_sha256={digests[args.output_json.name]} "
        f"npz_sha256={digests[args.output_npz.name]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
