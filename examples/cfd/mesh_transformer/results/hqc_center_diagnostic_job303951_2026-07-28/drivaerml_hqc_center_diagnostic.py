# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose H-QC's preprocessing-center equivalence failure without scoring truth.

This is an exploratory, non-deciding diagnostic for failed AGA job 303890.
It reuses the frozen H-QC producer to load the exact checkpoint, case, and
preprocessing chain, then localizes translation-frame sensitivity at K=2500.
No target-error or H-QC decision metric is computed or published.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

EXPECTED_PRODUCER_SHA256 = (
    "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
)
EXPECTED_FAILED_JOB_ID = 303890
CASE_ID = "run_118"
RESOLUTION = 2_500


def _load_producer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_hqc_producer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen producer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().double()
    right64 = right.detach().double()
    numerator = torch.linalg.vector_norm(left64 - right64)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(left64),
            torch.linalg.vector_norm(right64),
        ),
        left64.new_tensor(1.0e-12),
    )
    return float((numerator / denominator).item())


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(
            f"Cannot compare shapes {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.numel() == 0:
        return {
            "shape": list(left.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "exact": True,
            "nonzero_count": 0,
            "max_abs": 0.0,
            "relative_l2": 0.0,
        }
    delta = left.detach().double() - right.detach().double()
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "exact": bool(torch.equal(left, right)),
        "nonzero_count": int(torch.count_nonzero(delta).item()),
        "max_abs": float(torch.max(torch.abs(delta)).item()),
        "relative_l2": _relative_l2(left, right),
    }


def _translation_residual(primary: torch.Tensor, fixed: torch.Tensor) -> dict[str, Any]:
    if primary.shape != fixed.shape or primary.ndim != 2:
        raise ValueError(
            f"Translation comparison requires equal matrices, got "
            f"{tuple(primary.shape)} and {tuple(fixed.shape)}"
        )
    difference = fixed.detach().double() - primary.detach().double()
    translation = difference.mean(dim=0)
    residual = difference - translation
    return {
        "shape": list(primary.shape),
        "least_squares_translation_float64": [
            float(value) for value in translation.cpu().tolist()
        ],
        "max_abs_nonuniform_residual": float(torch.max(torch.abs(residual)).item()),
        "rms_nonuniform_residual": float(
            torch.sqrt(torch.mean(residual.square())).item()
        ),
        "nonzero_residual_count": int(torch.count_nonzero(residual).item()),
    }


def _normal_difference(primary: torch.Tensor, fixed: torch.Tensor) -> dict[str, Any]:
    result = _tensor_difference(primary, fixed)
    dots = torch.sum(primary.detach().double() * fixed.detach().double(), dim=-1)
    dots = dots.clamp(-1.0, 1.0)
    result.update(
        {
            "minimum_dot": float(torch.min(dots).item()),
            "maximum_angle_degrees": float(
                torch.rad2deg(torch.acos(torch.min(dots))).item()
            ),
        }
    )
    return result


def _prediction_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    return {
        name: _tensor_difference(left[name], right[name])
        for name in ("pressure", "wss")
    }


def _extract_prediction(output: Any) -> dict[str, torch.Tensor]:
    return {
        name: output.point_data[name].detach().float().clone()
        for name in ("pressure", "wss")
    }


def _encode_and_decode(
    runtime: Any, domain: Any
) -> tuple[Any, dict[str, torch.Tensor]]:
    encoded = runtime.model.encode(domain)
    return encoded, _extract_prediction(runtime.model.decode(encoded))


def _normalized_queries(encoded: Any) -> torch.Tensor:
    return (
        ((encoded.query_mesh.points - encoded.center) / encoded.reference_length)
        .detach()
        .clone()
    )


def _decode_at_normalized_queries(
    runtime: Any, encoded: Any, normalized_queries: torch.Tensor
) -> dict[str, torch.Tensor]:
    neutral = replace(
        encoded,
        center=torch.zeros_like(encoded.center),
        reference_length=torch.ones_like(encoded.reference_length),
    )
    query_mesh = runtime.mesh_type(points=normalized_queries)
    return _extract_prediction(runtime.model.decode(neutral, query_mesh))


def _source_geometry(encoded: Any) -> dict[str, torch.Tensor]:
    source = encoded.source_mesh
    return {
        "points": source.points.detach().clone(),
        "centroids": source.cell_centroids.detach().clone(),
        "areas": source.cell_areas.detach().clone(),
        "normals": source.cell_normals.detach().clone(),
    }


def _state_difference(primary: Any, fixed: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state_name in ("operator_state", "drive_state"):
        primary_state = getattr(primary, state_name)
        fixed_state = getattr(fixed, state_name)
        result[state_name] = {
            sector: _tensor_difference(
                getattr(primary_state, sector), getattr(fixed_state, sector)
            )
            for sector in ("scalars", "vectors", "pseudos")
        }
    if primary.kernel_cache is None or fixed.kernel_cache is None:
        raise ValueError("Expected kernel caches in both encoded boundaries")
    result["kernel_cache"] = {
        field: _tensor_difference(
            getattr(primary.kernel_cache, field),
            getattr(fixed.kernel_cache, field),
        )
        for field in (
            "panel_vertices",
            "centroids",
            "normals",
            "weights",
            "coefficients",
            "value_scalars",
            "value_vectors",
        )
    }
    return result


def _replace_stored_normals(domain: Any, mesh_type: Any) -> Any:
    from physicsnemo.mesh import DomainMesh

    boundary = domain.boundaries["vehicle"]
    cell_data = boundary.cell_data.clone()
    cell_data["normals"] = torch.zeros_like(cell_data["normals"])
    replacement = mesh_type(
        points=boundary.points,
        cells=boundary.cells,
        point_data=boundary.point_data,
        cell_data=cell_data,
        global_data=boundary.global_data,
    )
    return DomainMesh(
        interior=domain.interior,
        boundaries={"vehicle": replacement},
        global_data=domain.global_data,
    )


def _encode_with_source_override(
    runtime: Any,
    domain: Any,
    reference: Mapping[str, torch.Tensor],
    mode: str,
) -> Any:
    """Intercept the earliest source-mesh consumer and replace selected geometry."""
    if mode not in ("normals", "derived", "full"):
        raise ValueError(f"Unknown source override mode {mode!r}")
    model = runtime.model
    original = model._source_operator_input

    def intercepted(
        model_domain: Any,
        source_mesh: Any,
        boundary_operator: Any,
        global_operator: Any,
    ) -> Any:
        if mode == "full":
            source_mesh.points.copy_(reference["points"])
        keys = ("normals",) if mode == "normals" else ("centroids", "areas", "normals")
        for key in keys:
            source_mesh._cache["cell", key] = reference[key].to(
                device=source_mesh.points.device,
                dtype=source_mesh.points.dtype,
            )
        return original(
            model_domain,
            source_mesh,
            boundary_operator,
            global_operator,
        )

    object.__setattr__(model, "_source_operator_input", intercepted)
    try:
        return model.encode(domain)
    finally:
        object.__setattr__(model, "_source_operator_input", original)


def _provenance(
    *,
    hqc: Any,
    args: argparse.Namespace,
    producer_path: Path,
    npz_path: Path,
) -> dict[str, Any]:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    source_paths = (
        Path("physicsnemo/experimental/nn/mesh_attention/model.py"),
        Path("physicsnemo/experimental/nn/mesh_attention/kernel_decoder.py"),
        Path("physicsnemo/mesh/mesh.py"),
        Path("physicsnemo/mesh/transformations/geometric.py"),
        Path("physicsnemo/datapipes/transforms/mesh/transforms.py"),
    )
    return {
        "command": list(sys.argv),
        "diagnostic_script_path": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": hqc._sha256_file(Path(__file__).resolve()),
        "frozen_producer_path": str(producer_path),
        "frozen_producer_sha256": hqc._sha256_file(producer_path),
        "source_tree_manifest_sha256": hqc._source_tree_manifest_sha256(args.repo_root),
        "selected_source_files": {
            path.as_posix(): hqc._sha256_file(args.repo_root / path)
            for path in source_paths
        },
        "input_hashes": {
            "dataset_manifest": hqc._sha256_file(args.dataset_root / "manifest.json"),
            "dataset_config": hqc._sha256_file(args.dataset_config),
            "resolved_config": hqc._sha256_file(args.resolved_config),
            "model_checkpoint": hqc._sha256_file(
                args.checkpoint_dir / hqc.MODEL_FILENAME
            ),
            "normalization_stats": hqc._sha256_file(
                args.checkpoint_dir / hqc.NORM_STATS_FILENAME
            ),
            "historical_metrics": hqc._sha256_file(args.historical_metrics),
        },
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": hqc._sha256_file(npz_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hardware": {
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": [int(capability[0]), int(capability[1])],
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.producer = args.producer.resolve()
    args.repo_root = args.repo_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.dataset_config = args.dataset_config.resolve()
    args.resolved_config = args.resolved_config.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.historical_metrics = args.historical_metrics.resolve()
    args.output_json = args.output_json.resolve()
    args.output_npz = args.output_npz.resolve()
    for output in (args.output_json, args.output_npz):
        if output.exists() or output.with_name(f"{output.name}.sha256").exists():
            raise FileExistsError(f"Refusing to overwrite output or sidecar: {output}")

    hqc = _load_producer(args.producer)
    producer_sha = hqc._sha256_file(args.producer)
    if producer_sha != EXPECTED_PRODUCER_SHA256:
        raise ValueError(
            f"Frozen producer changed: expected {EXPECTED_PRODUCER_SHA256}, "
            f"got {producer_sha}"
        )
    hqc._validate_frozen_inputs(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
        historical_metrics_path=args.historical_metrics,
    )
    runtime = hqc._load_runtime(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        dataset_config_path=args.dataset_config,
        resolved_config_path=args.resolved_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    hqc._validate_reader(runtime)

    spec = next(spec for spec in hqc.CASE_SPECS if spec.case_id == CASE_ID)
    raw_mesh = runtime.dataset.reader._load_sample(spec.reader_index)
    ids_10k = hqc._cyclic_indices(
        spec.n_master_cells, spec.historical_start, hqc.BASELINE_K
    )
    subset_10k = hqc._compact_explicit_cell_subset(raw_mesh, ids_10k, runtime.mesh_type)
    fixed_center = hqc._pipeline_center_on_device(subset_10k, runtime.device)
    ids = hqc._cyclic_indices(spec.n_master_cells, spec.historical_start, RESOLUTION)
    subset = hqc._compact_explicit_cell_subset(raw_mesh, ids, runtime.mesh_type)
    primary_domain, primary_center = hqc._apply_pipeline(
        runtime, subset, fixed_center=None
    )
    fixed_domain, applied_fixed_center = hqc._apply_pipeline(
        runtime, subset, fixed_center=fixed_center
    )
    if not torch.equal(applied_fixed_center, fixed_center):
        raise ValueError("Fixed center changed while applying diagnostic pipeline")

    primary_boundary = primary_domain.boundaries["vehicle"]
    fixed_boundary = fixed_domain.boundaries["vehicle"]
    stored_normals_primary = primary_boundary.cell_data["normals"].detach().clone()
    stored_normals_fixed = fixed_boundary.cell_data["normals"].detach().clone()
    zero_normal_domain = _replace_stored_normals(primary_domain, runtime.mesh_type)

    with torch.no_grad(), runtime.autocast_context("bfloat16"):
        encoded_primary, prediction_primary = _encode_and_decode(
            runtime, primary_domain
        )
        encoded_fixed, prediction_fixed = _encode_and_decode(runtime, fixed_domain)
        _, prediction_repeat = _encode_and_decode(runtime, primary_domain)
        encoded_zero_normals, prediction_zero_normals = _encode_and_decode(
            runtime, zero_normal_domain
        )

        primary_source = _source_geometry(encoded_primary)
        encoded_fixed_primary_normals = _encode_with_source_override(
            runtime, fixed_domain, primary_source, "normals"
        )
        encoded_fixed_primary_derived = _encode_with_source_override(
            runtime, fixed_domain, primary_source, "derived"
        )
        encoded_fixed_primary_full = _encode_with_source_override(
            runtime, fixed_domain, primary_source, "full"
        )

        primary_queries = _normalized_queries(encoded_primary)
        fixed_queries = _normalized_queries(encoded_fixed)
        prediction_primary_canonical = _decode_at_normalized_queries(
            runtime, encoded_primary, primary_queries
        )
        prediction_fixed_canonical = _decode_at_normalized_queries(
            runtime, encoded_fixed, fixed_queries
        )
        prediction_fixed_at_primary_queries = _decode_at_normalized_queries(
            runtime, encoded_fixed, primary_queries
        )
        prediction_primary_at_fixed_queries = _decode_at_normalized_queries(
            runtime, encoded_primary, fixed_queries
        )
        prediction_primary_normals_at_primary_queries = _decode_at_normalized_queries(
            runtime, encoded_fixed_primary_normals, primary_queries
        )
        prediction_primary_derived_at_primary_queries = _decode_at_normalized_queries(
            runtime, encoded_fixed_primary_derived, primary_queries
        )
        prediction_primary_full_at_primary_queries = _decode_at_normalized_queries(
            runtime, encoded_fixed_primary_full, primary_queries
        )

    with torch.no_grad(), runtime.autocast_context("float32"):
        encoded_primary_fp32, prediction_primary_fp32 = _encode_and_decode(
            runtime, primary_domain
        )
        encoded_fixed_fp32, prediction_fixed_fp32 = _encode_and_decode(
            runtime, fixed_domain
        )

    fixed_source = _source_geometry(encoded_fixed)
    zero_normal_source = _source_geometry(encoded_zero_normals)
    arrays = {
        "cell_ids_int64": np.asarray(ids, dtype="<i8"),
        "pipeline_primary_points_float32": primary_boundary.points.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "pipeline_fixed_points_float32": fixed_boundary.points.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "pipeline_primary_queries_float32": primary_domain.interior.points.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "pipeline_fixed_queries_float32": fixed_domain.interior.points.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "stored_primary_normals_float32": stored_normals_primary.cpu()
        .numpy()
        .astype("<f4", copy=False),
        "stored_fixed_normals_float32": stored_normals_fixed.cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_primary_source_points_float32": primary_source["points"]
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_fixed_source_points_float32": fixed_source["points"]
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_primary_source_normals_float32": primary_source["normals"]
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_fixed_source_normals_float32": fixed_source["normals"]
        .cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_primary_normalized_queries_float32": primary_queries.cpu()
        .numpy()
        .astype("<f4", copy=False),
        "model_fixed_normalized_queries_float32": fixed_queries.cpu()
        .numpy()
        .astype("<f4", copy=False),
    }
    prediction_arrays = {
        "bf16_primary": prediction_primary,
        "bf16_fixed": prediction_fixed,
        "bf16_fixed_at_primary_queries": prediction_fixed_at_primary_queries,
        "bf16_primary_normals_at_primary_queries": (
            prediction_primary_normals_at_primary_queries
        ),
        "bf16_primary_derived_at_primary_queries": (
            prediction_primary_derived_at_primary_queries
        ),
        "bf16_primary_full_at_primary_queries": (
            prediction_primary_full_at_primary_queries
        ),
        "fp32_primary": prediction_primary_fp32,
        "fp32_fixed": prediction_fixed_fp32,
    }
    for label, prediction in prediction_arrays.items():
        for field, values in prediction.items():
            arrays[f"{label}_{field}_float32"] = (
                values.cpu().numpy().astype("<f4", copy=False)
            )
    hqc._atomic_write_npz(args.output_npz, arrays)
    hqc._write_sha256_sidecar(args.output_npz)

    source_geometry_differences = {
        "points": _tensor_difference(primary_source["points"], fixed_source["points"]),
        "centroids": _tensor_difference(
            primary_source["centroids"], fixed_source["centroids"]
        ),
        "areas": _tensor_difference(primary_source["areas"], fixed_source["areas"]),
        "normals": _normal_difference(
            primary_source["normals"], fixed_source["normals"]
        ),
        "normalized_queries": _tensor_difference(primary_queries, fixed_queries),
    }
    result = {
        "schema_version": 1,
        "artifact_kind": "hqc_center_cause_diagnostic",
        "status": "PASSED_NONDECIDING_DIAGNOSTIC",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "failed_job_id": EXPECTED_FAILED_JOB_ID,
            "case_id": CASE_ID,
            "resolution": RESOLUTION,
            "exploratory_post_failure": True,
            "hqc_decision_metrics_computed": False,
            "truth_fields_read_for_scoring": False,
            "may_not_be_used_as_hqc_verdict_output": True,
        },
        "model_contract": {
            "configured_boundary_operator_fields": dict(
                runtime.cfg.model.boundary_field_ranks.vehicle.operator
            ),
            "configured_boundary_drive_fields": dict(
                runtime.cfg.model.boundary_field_ranks.vehicle.drive
            ),
            "stored_pipeline_normals_declared_as_model_field": False,
            "model_geometry_dtype": str(primary_boundary.points.dtype),
            "historical_learned_precision": "bfloat16",
        },
        "external_pipeline": {
            "primary_center_float32": [
                float(value) for value in primary_center.detach().cpu().tolist()
            ],
            "fixed_s10000_center_float32": [
                float(value) for value in fixed_center.detach().cpu().tolist()
            ],
            "boundary_point_translation_residual": _translation_residual(
                primary_boundary.points, fixed_boundary.points
            ),
            "query_point_translation_residual": _translation_residual(
                primary_domain.interior.points, fixed_domain.interior.points
            ),
            "stored_pipeline_normal_difference": _normal_difference(
                stored_normals_primary, stored_normals_fixed
            ),
        },
        "internal_model_geometry": {
            "primary_internal_center_float32": [
                float(value) for value in encoded_primary.center.detach().cpu().tolist()
            ],
            "fixed_internal_center_float32": [
                float(value) for value in encoded_fixed.center.detach().cpu().tolist()
            ],
            "reference_length_primary": float(
                encoded_primary.reference_length.detach().cpu().item()
            ),
            "reference_length_fixed": float(
                encoded_fixed.reference_length.detach().cpu().item()
            ),
            "differences": source_geometry_differences,
            "encoded_state_differences": _state_difference(
                encoded_primary, encoded_fixed
            ),
        },
        "tests": {
            "historical_bfloat16_center_drift": _prediction_difference(
                prediction_primary, prediction_fixed
            ),
            "float32_center_drift": _prediction_difference(
                prediction_primary_fp32, prediction_fixed_fp32
            ),
            "bfloat16_repeat_determinism": _prediction_difference(
                prediction_primary, prediction_repeat
            ),
            "stored_pipeline_normals_zeroed": {
                "prediction_difference": _prediction_difference(
                    prediction_primary, prediction_zero_normals
                ),
                "internal_source_geometry_difference": {
                    key: (
                        _normal_difference(primary_source[key], zero_normal_source[key])
                        if key == "normals"
                        else _tensor_difference(
                            primary_source[key], zero_normal_source[key]
                        )
                    )
                    for key in ("points", "centroids", "areas", "normals")
                },
            },
            "canonical_query_decode_reproduction": {
                "primary": _prediction_difference(
                    prediction_primary, prediction_primary_canonical
                ),
                "fixed": _prediction_difference(
                    prediction_fixed, prediction_fixed_canonical
                ),
            },
            "query_coordinate_crossovers_bfloat16": {
                "primary_source_fixed_queries_vs_primary": _prediction_difference(
                    prediction_primary_at_fixed_queries, prediction_primary
                ),
                "fixed_source_primary_queries_vs_primary": _prediction_difference(
                    prediction_fixed_at_primary_queries, prediction_primary
                ),
                "fixed_source_primary_queries_vs_fixed": _prediction_difference(
                    prediction_fixed_at_primary_queries, prediction_fixed
                ),
            },
            "source_geometry_causal_ladder_at_primary_queries_bfloat16": {
                "fixed_source": _prediction_difference(
                    prediction_fixed_at_primary_queries, prediction_primary
                ),
                "fixed_source_with_primary_normals": _prediction_difference(
                    prediction_primary_normals_at_primary_queries,
                    prediction_primary,
                ),
                "fixed_source_with_primary_centroids_areas_normals": (
                    _prediction_difference(
                        prediction_primary_derived_at_primary_queries,
                        prediction_primary,
                    )
                ),
                "fixed_source_with_full_primary_internal_geometry": (
                    _prediction_difference(
                        prediction_primary_full_at_primary_queries,
                        prediction_primary,
                    )
                ),
            },
        },
        "provenance": _provenance(
            hqc=hqc,
            args=args,
            producer_path=args.producer,
            npz_path=args.output_npz,
        ),
    }
    hqc._atomic_write_bytes(
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )
    hqc._write_sha256_sidecar(args.output_json)
    print(
        f"PASSED_NONDECIDING_DIAGNOSTIC json={args.output_json} npz={args.output_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
