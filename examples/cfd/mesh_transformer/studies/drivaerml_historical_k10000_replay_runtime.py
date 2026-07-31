# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-only support for the frozen DrivAerML K=10,000 replay."""

import hashlib as _hashlib
import sys as _sys
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Any as _Any

import numpy as _np
import torch as _torch

__all__ = [
    "MODEL_FILENAME",
    "TRAINING_STATE_FILENAME",
    "NORM_STATS_FILENAME",
    "CaseSpec",
    "CASE_SPECS",
    "_validate_historical_starts",
    "_source_tree_manifest_sha256",
    "_load_runtime",
    "_native_geometry",
    "_apply_pipeline",
]

MODEL_FILENAME = "MeshTransformer.0.491.mdlus"
TRAINING_STATE_FILENAME = "checkpoint.0.491.pt"
NORM_STATS_FILENAME = "norm_stats.pt"

_BASELINE_K = 10_000
_EPOCH = 491
_READER_MASTER_SEED = 42
_READER_GENERATOR_SEED = 45
_RUNTIME_SAMPLING_RESOLUTION = 10_000
_TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}
_PARAMETER_COUNT = 1_278_268

_SOURCE_TREE_ROOTS = (
    _Path("physicsnemo/experimental/nn/mesh_attention"),
    _Path("physicsnemo/experimental/nn/symmetry"),
    _Path("physicsnemo/mesh"),
    _Path("physicsnemo/datapipes"),
)
_RECIPE_SOURCE_ROOT = _Path(
    "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
)


@_dataclass(frozen=True)
class CaseSpec:
    cohort_ordinal: int
    case_id: str
    reader_index: int
    n_master_cells: int
    historical_start: int


CASE_SPECS = (
    CaseSpec(0, "run_118", 21, 17_504_739, 14_045_027),
    CaseSpec(1, "run_129", 33, 16_380_547, 14_700_754),
    CaseSpec(2, "run_145", 51, 15_789_064, 9_195_926),
    CaseSpec(3, "run_149", 55, 18_007_064, 4_452_828),
    CaseSpec(4, "run_17", 77, 19_404_150, 6_369_582),
    CaseSpec(5, "run_171", 79, 18_792_923, 1_320_415),
    CaseSpec(6, "run_18", 88, 14_634_570, 10_215_595),
    CaseSpec(7, "run_183", 92, 14_932_664, 7_635_018),
    CaseSpec(8, "run_197", 107, 18_934_869, 16_494_923),
    CaseSpec(9, "run_202", 114, 17_796_743, 15_267_620),
    CaseSpec(10, "run_225", 136, 15_024_109, 3_789_927),
    CaseSpec(11, "run_270", 185, 18_857_430, 10_967_997),
    CaseSpec(12, "run_271", 186, 16_922_213, 5_453_831),
    CaseSpec(13, "run_298", 212, 15_063_884, 4_943_208),
    CaseSpec(14, "run_305", 221, 18_022_481, 16_998_850),
    CaseSpec(15, "run_320", 237, 16_199_351, 15_062_581),
    CaseSpec(16, "run_367", 285, 18_958_141, 5_352_845),
    CaseSpec(17, "run_380", 298, 19_519_305, 11_721_918),
    CaseSpec(18, "run_382", 300, 16_887_630, 11_083_431),
    CaseSpec(19, "run_399", 318, 16_222_090, 15_155_572),
    CaseSpec(20, "run_4", 319, 16_294_644, 13_228_777),
    CaseSpec(21, "run_409", 329, 16_591_548, 1_346_462),
    CaseSpec(22, "run_419", 340, 14_561_784, 12_777_694),
    CaseSpec(23, "run_424", 346, 16_588_938, 13_358_519),
    CaseSpec(24, "run_429", 351, 17_738_132, 365_298),
    CaseSpec(25, "run_431", 354, 15_747_949, 1_091_720),
    CaseSpec(26, "run_439", 362, 17_809_120, 8_840_407),
    CaseSpec(27, "run_465", 391, 16_443_085, 11_669_428),
    CaseSpec(28, "run_468", 394, 18_343_677, 15_504_945),
    CaseSpec(29, "run_469", 395, 19_780_049, 19_757_508),
    CaseSpec(30, "run_478", 404, 16_648_431, 16_079_300),
    CaseSpec(31, "run_489", 416, 16_063_459, 6_463_342),
    CaseSpec(32, "run_490", 418, 17_847_065, 191_824),
    CaseSpec(33, "run_495", 423, 15_715_663, 11_592_670),
    CaseSpec(34, "run_71", 453, 16_516_082, 2_240_523),
    CaseSpec(35, "run_86", 469, 17_188_261, 4_374_650),
)


@_dataclass
class _Runtime:
    repo_root: _Path
    recipe_root: _Path
    device: _torch.device
    cfg: _Any
    dataset: _Any
    collate_fn: _Any
    model: _Any
    normalize_output: _Any
    autocast_context: _Any
    mesh_type: _Any
    loaded_epoch: int


def _sha256_file(path: _Path, chunk_bytes: int = 8 << 20) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_manifest_sha256(repo_root: _Path) -> str:
    files: list[_Path] = []
    for root in _SOURCE_TREE_ROOTS:
        files.extend(
            path for path in (repo_root / root).rglob("*.py") if path.is_file()
        )
    recipe_root = repo_root / _RECIPE_SOURCE_ROOT
    files.extend(path for path in recipe_root.glob("*.py") if path.is_file())
    relative = sorted(path.relative_to(repo_root).as_posix() for path in files)
    if not relative:
        raise FileNotFoundError(f"No source files found below {repo_root}")
    digest = _hashlib.sha256()
    for name in relative:
        file_digest = _sha256_file(repo_root / name)
        digest.update(f"{file_digest}  {name}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_historical_starts() -> None:
    generator = _torch.Generator()
    generator.manual_seed(_READER_GENERATOR_SEED)
    for spec in CASE_SPECS:
        replayed = int(
            _torch.randint(
                0,
                spec.n_master_cells - _BASELINE_K + 1,
                (1,),
                generator=generator,
            ).item()
        )
        if replayed != spec.historical_start:
            raise ValueError(
                f"Historical RNG replay changed for {spec.case_id}: "
                f"expected {spec.historical_start}, got {replayed}"
            )


def _load_frozen_checkpoint(
    load_checkpoint: _Any,
    checkpoint_dir: _Path,
    model: _Any,
    device: _torch.device,
) -> int:
    loaded_epoch = int(
        load_checkpoint(
            path=str(checkpoint_dir),
            models=model,
            device=device,
            epoch=_EPOCH,
        )
    )
    if loaded_epoch != _EPOCH:
        raise ValueError(f"Loaded epoch {loaded_epoch}, expected {_EPOCH}")
    return loaded_epoch


def _load_runtime(
    *,
    repo_root: _Path,
    dataset_root: _Path,
    dataset_config_path: _Path,
    resolved_config_path: _Path,
    checkpoint_dir: _Path,
) -> _Runtime:
    """Instantiate the exact historical pipeline and epoch-491 model."""
    recipe_root = (
        repo_root / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
    )
    recipe_src = recipe_root / "src"
    if not recipe_src.is_dir():
        raise FileNotFoundError(f"Recipe source directory not found: {recipe_src}")
    _sys.path.insert(0, str(recipe_src))

    # Flat imports are the recipe's historical contract. Check their origin so
    # an installed package named ``datasets`` cannot be used accidentally.
    import datasets as recipe_datasets
    import hydra
    from collate import build_collate_fn
    from omegaconf import OmegaConf
    from output_normalize import normalize_output_to_tensordict
    from utils import get_autocast_context, resolve_dict, set_seed

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
        exterior_trace_self_entries,
    )
    from physicsnemo.mesh import Mesh
    from physicsnemo.utils import load_checkpoint

    datasets_origin = _Path(recipe_datasets.__file__).resolve().parent
    if datasets_origin != recipe_src.resolve():
        raise ImportError(
            f"Imported datasets from {datasets_origin}, expected {recipe_src.resolve()}"
        )

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        raise ValueError(
            f"H-QC producer requires one process per lane, got "
            f"world_size={dist.world_size}"
        )
    device = dist.device
    if device.type != "cuda":
        raise RuntimeError(f"Production H-QC inference requires CUDA, got {device}")

    cfg = OmegaConf.load(resolved_config_path)
    if str(cfg.precision) != "bfloat16":
        raise ValueError(f"Expected bfloat16 historical precision, got {cfg.precision}")
    if cfg.model.get("trace_of", None) != "vehicle":
        raise ValueError(f"Expected trace_of=vehicle, got {cfg.model.get('trace_of')}")
    if cfg.model.get("trace_self_correction", True) is not True:
        raise ValueError(
            "Expected effective trace_self_correction=True, got "
            f"{cfg.model.get('trace_self_correction')}"
        )
    if cfg.model.get("trace_readouts", True) is not True:
        raise ValueError(
            "Expected effective trace_readouts=True, got "
            f"{cfg.model.get('trace_readouts')}"
        )
    if float(cfg.model.get("reference_length_key") is not None) != 1.0:
        raise ValueError("Historical model is missing reference_length_key")

    ds_cfg = OmegaConf.load(dataset_config_path)
    OmegaConf.update(ds_cfg, "train_datadir", str(dataset_root), merge=False)
    # Reader subsampling is bypassed. The configured terminal SubsampleMesh sees
    # the already-reduced historical K=10k mesh and is a no-op.
    OmegaConf.update(
        ds_cfg,
        "sampling_resolution",
        _RUNTIME_SAMPLING_RESOLUTION,
        force_add=True,
    )
    dataset = recipe_datasets.build_dataset(
        ds_cfg,
        base_dir=recipe_root,
        augment=False,
        device=device,
        num_workers=1,
        pin_memory=False,
    )

    normalizer = recipe_datasets.find_normalizer([dataset])
    norm_stats_path = checkpoint_dir / NORM_STATS_FILENAME
    if normalizer is None:
        raise ValueError("Historical dataset pipeline has no NormalizeMeshFields")
    saved_stats = _torch.load(norm_stats_path, weights_only=True)
    normalizer.stats.clear()
    normalizer.stats.update(saved_stats)
    # The historical stats were saved from CUDA, but make the device contract
    # explicit in case a future Torch loader maps them to CPU.
    normalizer.to(device)

    forward_kwargs_spec = resolve_dict(cfg, "forward_kwargs")
    if forward_kwargs_spec != {"domain": ""}:
        raise ValueError(
            f"Unexpected historical forward_kwargs: {forward_kwargs_spec!r}"
        )
    collate_fn = build_collate_fn(
        input_type=str(cfg.input_type),
        forward_kwargs_spec=forward_kwargs_spec,
        target_config=_TARGET_CONFIG,
    )

    set_seed(_READER_MASTER_SEED, rank=0)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    if getattr(model, "trace_of", None) != "vehicle":
        raise ValueError(
            f"Instantiated model trace_of changed: got {getattr(model, 'trace_of', None)}"
        )
    if getattr(model, "trace_self_correction", None) is not True:
        raise ValueError(
            "Instantiated model must enable the exterior +1/2 trace self-correction"
        )
    if getattr(model, "trace_readouts", None) is not True:
        raise ValueError("Instantiated model must enable own-cell typed trace readouts")
    if (
        getattr(model, "trace_operator_read_out", None) is None
        or getattr(model, "trace_drive_read_out", None) is None
    ):
        raise ValueError("Instantiated model is missing own-cell typed trace readouts")
    correction_probe = exterior_trace_self_entries(
        _torch.zeros((1, 1), dtype=_torch.float32, device=device),
        _torch.zeros((1,), dtype=_torch.long, device=device),
    )
    if correction_probe.item() != 0.5:
        raise ValueError(
            "Exterior trace self-correction changed: expected +0.5, got "
            f"{correction_probe.item()}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != _PARAMETER_COUNT:
        raise ValueError(
            f"Historical model parameter count changed: got {parameter_count:,}"
        )
    loaded_epoch = _load_frozen_checkpoint(
        load_checkpoint,
        checkpoint_dir,
        model,
        device,
    )
    model.eval()

    return _Runtime(
        repo_root=repo_root,
        recipe_root=recipe_root,
        device=device,
        cfg=cfg,
        dataset=dataset,
        collate_fn=collate_fn,
        model=model,
        normalize_output=normalize_output_to_tensordict,
        autocast_context=get_autocast_context,
        mesh_type=Mesh,
        loaded_epoch=loaded_epoch,
    )


def _native_geometry(mesh: _Any) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Return physical cell centroids, unit normals, and float64 native areas."""
    vertices = mesh.points[mesh.cells].float()
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = _torch.linalg.cross(edge_1, edge_2)
    twice_area = _torch.linalg.vector_norm(cross, dim=-1)
    if bool(_torch.any(twice_area <= 0.0)):
        raise ValueError("Selected nested source contains a degenerate triangle")
    normals = cross / twice_area[:, None]
    centroids = vertices.mean(dim=1)
    return (
        centroids.cpu().numpy().astype("<f4", copy=False),
        normals.cpu().numpy().astype("<f4", copy=False),
        (0.5 * twice_area.double()).cpu().numpy().astype("<f8", copy=False),
    )


def _apply_pipeline(
    runtime: _Runtime,
    mesh: _Any,
    *,
    fixed_center: _torch.Tensor | None,
) -> tuple[_Any, _torch.Tensor]:
    """Apply the historical transforms with either native or explicit centering."""
    data = mesh.to(runtime.device)
    center_count = 0
    applied_center: _torch.Tensor | None = None
    for transform in runtime.dataset.transforms:
        if transform.__class__.__name__ == "CenterMesh":
            center_count += 1
            if fixed_center is None:
                applied_center = data.points.mean(dim=0)
            else:
                applied_center = fixed_center.to(runtime.device)
            data = data.translate(-applied_center)
        else:
            data = transform(data)
    if center_count != 1 or applied_center is None:
        raise ValueError(
            f"Expected exactly one CenterMesh transform, observed {center_count}"
        )
    if data.__class__.__name__ != "DomainMesh":
        raise TypeError(
            f"Historical transform chain did not produce DomainMesh: {type(data)}"
        )
    return data, applied_center
