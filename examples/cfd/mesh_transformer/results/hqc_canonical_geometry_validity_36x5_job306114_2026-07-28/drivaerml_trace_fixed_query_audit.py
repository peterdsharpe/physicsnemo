# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce the preregistered H-QC aligned-trace DrivAerML audit lanes.

The historical H-CC resolution sweep changed its contiguous random block at
every resolution.  Consequently neither its sources nor its trace queries were
aligned across resolutions.  This producer reconstructs each ID-reference
case's historical 10k start, freezes an explicit cyclic order at that start,
and evaluates nested prefixes at 2.5k, 5k, 10k, 20k, and 40k cells.  Every
resolution is scored both on its coupled trace queries and on the first 2.5k
rows, Q=S_2500.

The primary preprocessing arm preserves the production CenterMesh behavior:
each resolution uses its own unweighted compacted-vertex centroid.  A second
pass applies the S_10000 centroid to every resolution and checks the model's
translation invariance.  This diagnostic does *not* freeze MeshTransformer's
internal area-weighted source center, which is intentionally recomputed from
S_k and is part of the source-count treatment.

Production execution is fail-closed.  It verifies the historical manifest,
configs, checkpoint files, source-code tree, cohort, raw sources, archived
10k artifacts, and all ordering/finite-value invariants before atomically
publishing an immutable JSON lane and its NPZ rescoring payload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = 2
ARTIFACT_KIND = "phase1_hqc_producer_lane"
PASSED_STATUS = "PASSED_HQC_PRODUCER_LANE"

RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
BASELINE_K = 10_000
FIXED_QUERY_K = 2_500
TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}

MANIFEST_SHA256 = "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
HISTORICAL_METRICS_SHA256 = (
    "423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a"
)
RESOLVED_CONFIG_SHA256 = (
    "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
)
DATASET_CONFIG_SHA256 = (
    "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
)
EXECUTION_SOURCE_TREE_MANIFEST_SHA256 = (
    "fa6a7b683fa9aa02e4537ef69e8e977906df7c9fa6964cb759edfcee8d7b90cd"
)
MODEL_FILENAME = "MeshTransformer.0.491.mdlus"
MODEL_SHA256 = "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
TRAINING_STATE_FILENAME = "checkpoint.0.491.pt"
TRAINING_STATE_SHA256 = (
    "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
)
NORM_STATS_FILENAME = "norm_stats.pt"
NORM_STATS_SHA256 = "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
RUN_ID = "t2_mesh_transformer_surface_flagship_seed42"
EPOCH = 491
READER_MASTER_SEED = 42
READER_GENERATOR_SEED = 45

ARCHIVED_UNIFORM_PRESSURE_MEAN = 0.16716310713026258
ARCHIVED_MEAN_ABS_TOLERANCE = 5.0e-6
ARCHIVED_CASE_ABS_TOLERANCE = 1.0e-3
RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE = 1.0e-6
CENTER_METRIC_RELATIVE_TOLERANCE = 1.0e-3
ARCHIVED_PRESSURE_FIELD_ABS_TOLERANCE = 1.0e-3
ARCHIVED_WSS_FIELD_ABS_TOLERANCE = 1.0e-5
ARCHIVED_PIPELINE_NORMAL_ABS_TOLERANCE = 2.0e-6
PIPELINE_NORMAL_GEOMETRY_ABS_TOLERANCE = 5.0e-7
PIPELINE_NORMAL_UNIT_ABS_TOLERANCE = 5.0e-6
SIGNATURE_QUANTIZATION = 1.0e-6

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

SIGNATURE_ALGORITHM = (
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


@dataclass(frozen=True)
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

COHORT_CASE_IDS = tuple(spec.case_id for spec in CASE_SPECS)

FROZEN_CONTRACT = {
    "hypothesis_id": "H-QC",
    "cohort_name": "id_reference",
    "cohort_case_ids": list(COHORT_CASE_IDS),
    "manifest_sha256": MANIFEST_SHA256,
    "historical_metrics_sha256": HISTORICAL_METRICS_SHA256,
    "resolved_config_sha256": RESOLVED_CONFIG_SHA256,
    "dataset_config_sha256": DATASET_CONFIG_SHA256,
    "execution_source_tree_manifest_sha256": EXECUTION_SOURCE_TREE_MANIFEST_SHA256,
    "run_id": RUN_ID,
    "epoch": EPOCH,
    "model_filename": MODEL_FILENAME,
    "model_sha256": MODEL_SHA256,
    "training_state_filename": TRAINING_STATE_FILENAME,
    "training_state_sha256": TRAINING_STATE_SHA256,
    "norm_stats_filename": NORM_STATS_FILENAME,
    "norm_stats_sha256": NORM_STATS_SHA256,
    "reader_seed": READER_MASTER_SEED,
    "reader_generator_seed": READER_GENERATOR_SEED,
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
    "historical_signature_algorithm": SIGNATURE_ALGORITHM,
    "metric_definitions": METRIC_DEFINITIONS,
}

SOURCE_ARRAY_RELATIVE_PATHS = {
    "points_sha256": Path("points.memmap"),
    "cells_sha256": Path("cells.memmap"),
    "pressure_sha256": Path("cell_data/pMeanTrim.memmap"),
    "wss_sha256": Path("cell_data/wallShearStressMeanTrim.memmap"),
}

SOURCE_TREE_ROOTS = (
    Path("physicsnemo/experimental/nn/mesh_attention"),
    Path("physicsnemo/experimental/nn/symmetry"),
    Path("physicsnemo/mesh"),
    Path("physicsnemo/datapipes"),
)
RECIPE_SOURCE_ROOT = Path(
    "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray, dtype: np.dtype[Any] | str | None = None) -> str:
    canonical = np.asarray(array, dtype=dtype)
    contiguous = np.ascontiguousarray(canonical)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _finite(value: float | np.floating[Any], label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {result}")
    return result


def _positive_norm(value: np.ndarray, label: str) -> float:
    result = float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must have a finite positive norm, got {result}")
    return result


def _cyclic_indices(n_cells: int, start: int, k: int) -> np.ndarray:
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    if not 0 <= start < n_cells:
        raise ValueError(f"start {start} is outside [0, {n_cells})")
    if not 0 < k <= n_cells:
        raise ValueError(f"k must be in [1, {n_cells}], got {k}")
    return (start + np.arange(k, dtype=np.int64)) % n_cells


def _relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    eps: float = 1.0e-8,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    numerator = np.sqrt(np.sum((prediction64 - truth64) ** 2, dtype=np.float64))
    denominator = np.sqrt(np.sum(truth64**2, dtype=np.float64)) + eps
    return _finite(numerator / denominator, "relative_l2")


def _centered_pattern_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    pred_centered = prediction64 - prediction64.mean(dtype=np.float64)
    truth_centered = truth64 - truth64.mean(dtype=np.float64)
    pred_norm = _positive_norm(pred_centered, "centered prediction")
    truth_norm = _positive_norm(truth_centered, "centered truth")
    cosine = float(np.dot(pred_centered.ravel(), truth_centered.ravel()))
    cosine /= pred_norm * truth_norm
    if not math.isfinite(cosine):
        raise ValueError(f"centered-pattern cosine is non-finite: {cosine}")
    signed_correlation = min(1.0, max(-1.0, cosine))
    positive_correlation = max(0.0, signed_correlation)
    pattern_error = math.sqrt(max(0.0, 1.0 - positive_correlation**2))
    return {
        "signed_centered_correlation": _finite(
            signed_correlation, "signed_centered_correlation"
        ),
        "positive_gain_pattern_error": _finite(
            pattern_error, "positive_gain_pattern_error"
        ),
    }


def _amplitude_ratio(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    truth_rms = math.sqrt(float(np.mean(truth64**2, dtype=np.float64)))
    if not math.isfinite(truth_rms) or truth_rms <= 0.0:
        raise ValueError(f"truth RMS must be finite and positive, got {truth_rms}")
    prediction_rms = math.sqrt(float(np.mean(prediction64**2, dtype=np.float64)))
    return _finite(prediction_rms / truth_rms, "amplitude_ratio")


def _area_weighted_pressure_relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    areas: np.ndarray,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    areas64 = np.asarray(areas, dtype=np.float64)
    area_sum = float(areas64.sum(dtype=np.float64))
    if not math.isfinite(area_sum) or area_sum <= 0.0:
        raise ValueError(f"native area sum must be finite and positive, got {area_sum}")
    weights = areas64 / area_sum
    numerator = math.sqrt(
        float(np.sum(weights * (prediction64 - truth64) ** 2, dtype=np.float64))
    )
    denominator = (
        math.sqrt(float(np.sum(weights * truth64**2, dtype=np.float64))) + 1.0e-8
    )
    return _finite(
        numerator / denominator,
        "area_weighted_pressure_relative_l2",
    )


def _full_uniform_metrics(
    prediction_pressure: np.ndarray,
    truth_pressure: np.ndarray,
    prediction_wss: np.ndarray,
    truth_wss: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    *,
    n_master_cells: int,
) -> dict[str, float]:
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
    actual = {
        "truth_pressure": true_p.shape,
        "prediction_wss": pred_wss.shape,
        "truth_wss": true_wss.shape,
        "normals": normals64.shape,
        "areas": areas64.shape,
    }
    for name, shape in expected_shapes.items():
        if actual[name] != shape:
            raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
    if not all(
        np.isfinite(value).all()
        for value in (pred_p, true_p, pred_wss, true_wss, normals64, areas64)
    ):
        raise ValueError("metric inputs must all be finite")

    wss_error = _relative_l2(pred_wss.ravel(), true_wss.ravel())
    normal_component = np.einsum(
        "ij,ij->i", pred_wss, normals64, dtype=np.float64, optimize=True
    )
    pred_wss_norm = float(np.linalg.norm(pred_wss))
    wss_normal_energy = float(np.linalg.norm(normal_component)) / (
        pred_wss_norm + 1.0e-8
    )

    subset_scale = float(n_master_cells) / float(m)
    pred_force = subset_scale * np.sum(
        areas64[:, None] * pred_p[:, None] * normals64,
        axis=0,
        dtype=np.float64,
    )
    true_force = subset_scale * np.sum(
        areas64[:, None] * true_p[:, None] * normals64,
        axis=0,
        dtype=np.float64,
    )
    force_error = float(np.linalg.norm(pred_force - true_force)) / (
        float(np.linalg.norm(true_force)) + 1.0e-12
    )

    result = {
        "pressure_relative_l2": _relative_l2(pred_p, true_p),
        **_centered_pattern_metrics(pred_p, true_p),
        "amplitude_ratio": _amplitude_ratio(pred_p, true_p),
        "wss_frobenius_relative_l2": wss_error,
        "wss_normal_energy": _finite(wss_normal_energy, "wss_normal_energy"),
        "scaled_subset_pressure_force_relative_error": _finite(
            force_error, "scaled_subset_pressure_force_relative_error"
        ),
    }
    if result["pressure_relative_l2"] <= 0.0:
        raise ValueError("pressure_relative_l2 must be strictly positive")
    if not -1.0 <= result["signed_centered_correlation"] <= 1.0:
        raise ValueError(f"signed correlation is outside [-1, 1]: {result}")
    if any(
        value < 0.0
        for key, value in result.items()
        if key != "signed_centered_correlation"
    ):
        raise ValueError(f"metric bundle contains a negative value: {result}")
    return result


def _score_metrics(
    prediction_pressure: np.ndarray,
    truth_pressure: np.ndarray,
    prediction_wss: np.ndarray,
    truth_wss: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    *,
    n_master_cells: int,
) -> dict[str, dict[str, float]]:
    return {
        "uniform": _full_uniform_metrics(
            prediction_pressure,
            truth_pressure,
            prediction_wss,
            truth_wss,
            normals,
            areas,
            n_master_cells=n_master_cells,
        ),
        "area_weighted": {
            "pressure_relative_l2": _area_weighted_pressure_relative_l2(
                prediction_pressure,
                truth_pressure,
                areas,
            )
        },
    }


def _metric_relative_change(primary: float, diagnostic: float) -> float:
    return _finite(
        abs(primary - diagnostic) / max(abs(primary), abs(diagnostic), 1.0e-12),
        "center metric relative change",
    )


def _native_area_reference(areas: np.ndarray, label: str) -> dict[str, float]:
    values = np.asarray(areas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{label} areas must be a nonempty vector, got {values.shape}")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{label} areas must be finite and strictly positive")
    total = _finite(values.sum(dtype=np.float64), f"{label} native area")
    return {
        "native_area": total,
        # Representativeness across unequal row counts must compare mean cell
        # area, not totals (Q has exactly one quarter as many rows as S10k).
        "mean_native_cell_area": _finite(
            total / len(values), f"{label} mean native cell area"
        ),
    }


def _prediction_relative_difference(
    primary: np.ndarray, diagnostic: np.ndarray
) -> float:
    primary64 = np.asarray(primary, dtype=np.float64)
    diagnostic64 = np.asarray(diagnostic, dtype=np.float64)
    numerator = float(np.linalg.norm(primary64 - diagnostic64))
    denominator = max(
        float(np.linalg.norm(primary64)),
        float(np.linalg.norm(diagnostic64)),
        1.0e-8,
    )
    return _finite(numerator / denominator, "center prediction relative difference")


def _translation_invariant_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"query points must have shape (N,3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("query points must be finite")
    return np.asarray(values - values[0], dtype="<f4")


def _translation_invariant_signature(
    *,
    query_points: np.ndarray,
    compacted_cells: np.ndarray,
    identity_components: Mapping[str, str],
) -> str:
    payload = {
        "algorithm": SIGNATURE_ALGORITHM,
        "canonical_raw_translation_invariant_points_sha256_float32": _sha256_array(
            _translation_invariant_points(query_points), "<f4"
        ),
        "compacted_connectivity_sha256_int64": _sha256_array(compacted_cells, "<i8"),
        "identity_components": dict(identity_components),
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


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
        file_digest = _sha256_file(repo_root / name)
        digest.update(f"{file_digest}  {name}\n".encode("utf-8"))
    return digest.hexdigest()


def _reader_seed_fork_chain_sha256() -> str:
    payload = {
        "dataloader_master_seed": READER_MASTER_SEED,
        "dataloader_sampler_child_seed": 43,
        "dataloader_dataset_child_seed": 44,
        "mesh_dataset_reader_child_seed": READER_GENERATOR_SEED,
        "manifest_sampler_shuffle": False,
        "world_size": 1,
        "cohort_order": list(COHORT_CASE_IDS),
        "reader_draw": "torch.randint(0,n_cells-k+1,(1,),generator=reader_generator)",
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _parse_historical_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] | None = None
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("phase") == "infer_step":
            sample_id = str(record["sample_id"])
            case_matches = [
                case_id
                for case_id in COHORT_CASE_IDS
                if f"_{case_id}_domain_{case_id}" in sample_id
            ]
            if len(case_matches) != 1:
                raise ValueError(
                    f"Cannot resolve one cohort case from sample_id={sample_id!r}"
                )
            case_id = case_matches[0]
            if case_id in rows:
                raise ValueError(f"Duplicate historical metrics row for {case_id}")
            rows[case_id] = record
        elif record.get("phase") == "infer_summary":
            summary = record
    if tuple(rows) != COHORT_CASE_IDS:
        raise ValueError(
            "Historical metrics rows are not in the frozen cohort order: "
            f"{tuple(rows)!r}"
        )
    mean = float(
        (summary or {})
        .get("metrics", {})
        .get(
            "pressure_l2",
            np.mean(
                [row["metrics"]["pressure_l2"] for row in rows.values()],
                dtype=np.float64,
            ),
        )
    )
    if abs(mean - ARCHIVED_UNIFORM_PRESSURE_MEAN) > ARCHIVED_MEAN_ABS_TOLERANCE:
        raise ValueError(
            f"Historical pressure mean {mean} differs from frozen "
            f"{ARCHIVED_UNIFORM_PRESSURE_MEAN}"
        )
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_name(f"{path.name}.sha256")
    line = f"{_sha256_file(path)}  {path.name}\n".encode("ascii")
    _atomic_write_bytes(sidecar, line)
    return sidecar


def _package_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback


@dataclass
class Runtime:
    repo_root: Path
    recipe_root: Path
    device: torch.device
    cfg: Any
    dataset: Any
    collate_fn: Any
    model: Any
    normalize_output: Any
    autocast_context: Any
    mesh_type: Any
    loaded_epoch: int


@dataclass
class ForwardResult:
    pressure: np.ndarray
    wss: np.ndarray
    truth_pressure: np.ndarray
    truth_wss: np.ndarray
    query_points: np.ndarray
    boundary_cells: np.ndarray
    boundary_normals: np.ndarray


def _score_forward_result(
    result: ForwardResult,
    native_normals: np.ndarray,
    native_areas: np.ndarray,
    *,
    n_master_cells: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Score one forward result on all trace rows and the frozen Q prefix."""
    if len(result.pressure) < FIXED_QUERY_K:
        raise ValueError(
            f"Forward result has {len(result.pressure)} rows, fewer than Q={FIXED_QUERY_K}"
        )
    coupled = _score_metrics(
        result.pressure,
        result.truth_pressure,
        result.wss,
        result.truth_wss,
        native_normals,
        native_areas,
        n_master_cells=n_master_cells,
    )
    fixed_q = _score_metrics(
        result.pressure[:FIXED_QUERY_K],
        result.truth_pressure[:FIXED_QUERY_K],
        result.wss[:FIXED_QUERY_K],
        result.truth_wss[:FIXED_QUERY_K],
        native_normals[:FIXED_QUERY_K],
        native_areas[:FIXED_QUERY_K],
        n_master_cells=n_master_cells,
    )
    return coupled, fixed_q


def _load_runtime(
    *,
    repo_root: Path,
    dataset_root: Path,
    dataset_config_path: Path,
    resolved_config_path: Path,
    checkpoint_dir: Path,
) -> Runtime:
    """Instantiate the exact historical pipeline and epoch-491 model."""
    recipe_root = (
        repo_root / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
    )
    recipe_src = recipe_root / "src"
    if not recipe_src.is_dir():
        raise FileNotFoundError(f"Recipe source directory not found: {recipe_src}")
    sys.path.insert(0, str(recipe_src))

    # Flat imports are the recipe's historical contract.  Check their origin so
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

    datasets_origin = Path(recipe_datasets.__file__).resolve().parent
    if datasets_origin != recipe_src.resolve():
        raise ImportError(
            f"Imported datasets from {datasets_origin}, expected {recipe_src.resolve()}"
        )

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        raise ValueError(
            f"H-QC producer requires one process per lane, got world_size={dist.world_size}"
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
    # Reader subsampling is bypassed through reader._load_sample.  The configured
    # terminal SubsampleMesh sees an already-reduced K<=40k mesh and is a no-op.
    OmegaConf.update(ds_cfg, "sampling_resolution", max(RESOLUTIONS), force_add=True)
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
    saved_stats = torch.load(norm_stats_path, weights_only=True)
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
        target_config=TARGET_CONFIG,
    )

    set_seed(READER_MASTER_SEED, rank=0)
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
        torch.zeros((1, 1), dtype=torch.float32, device=device),
        torch.zeros((1,), dtype=torch.long, device=device),
    )
    if correction_probe.item() != 0.5:
        raise ValueError(
            "Exterior trace self-correction changed: expected +0.5, got "
            f"{correction_probe.item()}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 1_278_268:
        raise ValueError(
            f"Historical model parameter count changed: got {parameter_count:,}"
        )
    loaded_epoch = int(
        load_checkpoint(path=str(checkpoint_dir), models=model, device=device)
    )
    if loaded_epoch != EPOCH:
        raise ValueError(f"Loaded epoch {loaded_epoch}, expected {EPOCH}")
    model.eval()

    return Runtime(
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


def _compact_explicit_cell_subset(
    mesh: Any,
    cell_ids: np.ndarray,
    mesh_type: Any,
) -> Any:
    """Select cells in caller order and compact points without reordering cells."""
    ids = torch.as_tensor(np.asarray(cell_ids, dtype=np.int64), dtype=torch.long)
    selected_cells = mesh.cells[ids]
    referenced, inverse = torch.unique(
        selected_cells,
        sorted=True,
        return_inverse=True,
    )
    compacted_cells = inverse.reshape_as(selected_cells)
    # Index even an empty TensorDict so its batch size tracks compacted points.
    point_data = mesh.point_data[referenced]
    return mesh_type(
        points=mesh.points[referenced],
        cells=compacted_cells,
        point_data=point_data,
        cell_data=mesh.cell_data[ids],
        global_data=mesh.global_data,
    )


def _native_geometry(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical cell centroids, unit normals, and float64 native areas."""
    vertices = mesh.points[mesh.cells].float()
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = torch.linalg.cross(edge_1, edge_2)
    twice_area = torch.linalg.vector_norm(cross, dim=-1)
    if bool(torch.any(twice_area <= 0.0)):
        raise ValueError("Selected nested source contains a degenerate triangle")
    normals = cross / twice_area[:, None]
    centroids = vertices.mean(dim=1)
    return (
        centroids.cpu().numpy().astype("<f4", copy=False),
        normals.cpu().numpy().astype("<f4", copy=False),
        (0.5 * twice_area.double()).cpu().numpy().astype("<f8", copy=False),
    )


def _pipeline_normal_diagnostics(
    native_normals: np.ndarray,
    boundary: Any,
    label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Validate normal geometry and winding in the frame seen by the model.

    The historical pipeline computes normals *after* float32 centering and
    non-dimensionalization.  Recomputing a triangle normal in that rounded
    frame is not componentwise identical to computing it from raw coordinates,
    even though a positive similarity preserves its orientation in exact
    arithmetic.  Validate the two contracts separately: the stored pipeline
    normal must match its transformed triangle, and its winding must agree with
    the raw native triangle.
    """
    observed = boundary.cell_data["normals"].detach().float()
    native = torch.as_tensor(
        np.asarray(native_normals, dtype=np.float32),
        device=observed.device,
        dtype=observed.dtype,
    )
    expected_shape = (boundary.n_cells, boundary.n_spatial_dims)
    if tuple(observed.shape) != expected_shape or tuple(native.shape) != expected_shape:
        raise ValueError(
            f"{label} normals have shapes observed={tuple(observed.shape)} "
            f"native={tuple(native.shape)}, expected={expected_shape}"
        )
    if not bool(torch.isfinite(observed).all()) or not bool(
        torch.isfinite(native).all()
    ):
        raise ValueError(f"{label} normals contain non-finite values")

    vertices = boundary.points[boundary.cells].float()
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    cross = torch.linalg.cross(edge_1, edge_2)
    cross_norm = torch.linalg.vector_norm(cross, dim=-1)
    if bool(torch.any(cross_norm <= 0.0)):
        raise ValueError(f"{label} transformed geometry contains a degenerate triangle")
    reconstructed = cross / cross_norm[:, None]

    geometry_error = float(torch.max(torch.abs(observed - reconstructed)).item())
    if geometry_error > PIPELINE_NORMAL_GEOMETRY_ABS_TOLERANCE:
        raise ValueError(
            f"{label} stored normals do not match transformed geometry: "
            f"max_abs={geometry_error}"
        )
    unit_error = float(
        torch.max(torch.abs(torch.linalg.vector_norm(observed, dim=-1) - 1.0)).item()
    )
    if unit_error > PIPELINE_NORMAL_UNIT_ABS_TOLERANCE:
        raise ValueError(f"{label} pipeline normals are not unit: max_abs={unit_error}")
    minimum_native_dot = float(torch.min(torch.sum(observed * native, dim=-1)).item())
    if minimum_native_dot <= 0.0:
        raise ValueError(
            f"{label} pipeline normal orientation disagrees with native winding: "
            f"min_dot={minimum_native_dot}"
        )

    return (
        observed.cpu().numpy(),
        {
            "max_unit_norm_abs_error": _finite(unit_error, f"{label} unit error"),
            "max_geometry_reconstruction_abs_error": _finite(
                geometry_error, f"{label} geometry reconstruction error"
            ),
            "min_native_dot": _finite(
                minimum_native_dot, f"{label} minimum native dot"
            ),
        },
    )


def _pipeline_center_on_device(mesh: Any, device: torch.device) -> torch.Tensor:
    return mesh.points.to(device).mean(dim=0)


def _apply_pipeline(
    runtime: Runtime,
    mesh: Any,
    *,
    fixed_center: torch.Tensor | None,
) -> tuple[Any, torch.Tensor]:
    """Apply the historical transforms with either native or explicit centering."""
    data = mesh.to(runtime.device)
    center_count = 0
    applied_center: torch.Tensor | None = None
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


def _run_forward(runtime: Runtime, domain: Any) -> ForwardResult:
    batch = runtime.collate_fn([(domain, {})])
    with torch.no_grad(), runtime.autocast_context(str(runtime.cfg.precision)):
        output = runtime.model(**batch["forward_kwargs"])
    prediction = runtime.normalize_output(
        output,
        TARGET_CONFIG,
        str(runtime.cfg.output_type),
    )
    targets = batch["targets"]
    pressure = prediction["pressure"].detach().float().cpu().numpy()
    wss = prediction["wss"].detach().float().cpu().numpy()
    truth_pressure = targets["pressure"].detach().float().cpu().numpy()
    truth_wss = targets["wss"].detach().float().cpu().numpy()
    query_points = domain.interior.points.detach().float().cpu().numpy()
    vehicle = domain.boundaries["vehicle"]
    boundary_cells = vehicle.cells.detach().cpu().numpy()
    boundary_normals = vehicle.cell_data["normals"].detach().float().cpu().numpy()
    n_queries = domain.interior.n_points
    expected_shapes = {
        "pressure": (n_queries,),
        "wss": (n_queries, 3),
        "truth_pressure": (n_queries,),
        "truth_wss": (n_queries, 3),
        "query_points": (n_queries, 3),
        "boundary_cells": (n_queries, 3),
        "boundary_normals": (n_queries, 3),
    }
    values = {
        "pressure": pressure,
        "wss": wss,
        "truth_pressure": truth_pressure,
        "truth_wss": truth_wss,
        "query_points": query_points,
        "boundary_cells": boundary_cells,
        "boundary_normals": boundary_normals,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"Trace output {name} has shape {values[name].shape}, expected {expected}"
            )
        if not np.isfinite(values[name]).all():
            raise ValueError(f"Trace output {name} contains non-finite values")
    if not np.allclose(
        query_points,
        vehicle.cell_centroids.detach().float().cpu().numpy(),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise ValueError("Trace query rows are not the vehicle cell centroids")
    return ForwardResult(**values)


def _vehicle_tensor_root(reader_path: Path) -> Path:
    root = reader_path / "_tensordict"
    required = (
        root / "meta.json",
        root / "points.memmap",
        root / "cells.memmap",
        root / "cell_data/pMeanTrim.memmap",
        root / "cell_data/wallShearStressMeanTrim.memmap",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Vehicle TensorDict files missing: {missing}")
    return root


def _source_identity(vehicle_root: Path) -> dict[str, str]:
    mesh_meta = vehicle_root / "meta.json"
    cell_meta = vehicle_root / "cell_data/meta.json"
    metadata_components = {
        "mesh_meta_sha256": _sha256_file(mesh_meta),
        "cell_data_meta_sha256": _sha256_file(cell_meta),
    }
    identity = {
        "metadata_sha256": _sha256_bytes(_canonical_json_bytes(metadata_components))
    }
    for key, relative in SOURCE_ARRAY_RELATIVE_PATHS.items():
        identity[key] = _sha256_file(vehicle_root / relative)
    return identity


def _archived_prediction_path(
    archive_root: Path,
    spec: CaseSpec,
) -> Path:
    return (
        archive_root
        / f"{spec.reader_index:05d}_{spec.case_id}_domain_{spec.case_id}.pdmsh"
    )


def _load_memmap_from_metadata(
    root: Path,
    relative_data: Path,
    metadata: Mapping[str, Any],
    field_name: str,
) -> np.ndarray:
    entry = metadata[field_name]
    dtypes = {
        "torch.float32": np.dtype("<f4"),
        "torch.int64": np.dtype("<i8"),
    }
    dtype = dtypes.get(entry["dtype"])
    if dtype is None:
        raise ValueError(f"Unsupported archived dtype {entry['dtype']!r}")
    shape = tuple(int(value) for value in entry["shape"])
    path = root / relative_data
    expected_bytes = math.prod(shape) * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes, expected {expected_bytes}"
        )
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def _load_archived_10k(archive_path: Path) -> dict[str, np.ndarray]:
    td_root = archive_path / "_tensordict"
    interior = td_root / "interior/_tensordict"
    boundary = td_root / "boundaries/vehicle/_tensordict"
    interior_meta = json.loads((interior / "meta.json").read_text())
    point_data_meta = json.loads((interior / "point_data/meta.json").read_text())
    boundary_meta = json.loads((boundary / "meta.json").read_text())
    boundary_cell_meta = json.loads((boundary / "cell_data/meta.json").read_text())
    return {
        "query_points": np.asarray(
            _load_memmap_from_metadata(
                interior, Path("points.memmap"), interior_meta, "points"
            )
        ),
        "true_pressure": np.asarray(
            _load_memmap_from_metadata(
                interior,
                Path("point_data/true_pressure.memmap"),
                point_data_meta,
                "true_pressure",
            )
        ),
        "true_wss": np.asarray(
            _load_memmap_from_metadata(
                interior,
                Path("point_data/true_wss.memmap"),
                point_data_meta,
                "true_wss",
            )
        ),
        "boundary_cells": np.asarray(
            _load_memmap_from_metadata(
                boundary, Path("cells.memmap"), boundary_meta, "cells"
            )
        ),
        "boundary_normals": np.asarray(
            _load_memmap_from_metadata(
                boundary,
                Path("cell_data/normals.memmap"),
                boundary_cell_meta,
                "normals",
            )
        ),
    }


def _raw_physical_fields(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        mesh.cell_data["pMeanTrim"].detach().float().cpu().numpy(),
        mesh.cell_data["wallShearStressMeanTrim"].detach().float().cpu().numpy(),
    )


def _l_ref(mesh: Any) -> float:
    value = float(mesh.global_data["L_ref"].detach().float().cpu().item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"L_ref must be finite and positive, got {value}")
    return value


def _archive_parity(
    *,
    archived: Mapping[str, np.ndarray],
    raw_mesh_10k: Any,
    raw_centroids_10k: np.ndarray,
    pipeline_normals_10k: np.ndarray,
    native_areas_10k: np.ndarray,
    cell_ids_10k: np.ndarray,
    fixed_center: np.ndarray,
    identity_components: Mapping[str, str],
) -> tuple[str, float, float, float]:
    raw_pressure, raw_wss = _raw_physical_fields(raw_mesh_10k)
    archived_pressure = np.asarray(archived["true_pressure"], dtype=np.float32)
    archived_wss = np.asarray(archived["true_wss"], dtype=np.float32)
    if raw_pressure.shape != (BASELINE_K,) or raw_wss.shape != (BASELINE_K, 3):
        raise ValueError("Raw historical 10k fields have unexpected shapes")
    pressure_max_abs = float(
        np.max(
            np.abs(
                raw_pressure.astype(np.float64) - archived_pressure.astype(np.float64)
            )
        )
    )
    wss_max_abs = float(
        np.max(np.abs(raw_wss.astype(np.float64) - archived_wss.astype(np.float64)))
    )
    if pressure_max_abs > ARCHIVED_PRESSURE_FIELD_ABS_TOLERANCE:
        raise ValueError(f"Archived pressure parity failed: max_abs={pressure_max_abs}")
    if wss_max_abs > ARCHIVED_WSS_FIELD_ABS_TOLERANCE:
        raise ValueError(f"Archived WSS parity failed: max_abs={wss_max_abs}")

    archived_cells = np.asarray(archived["boundary_cells"], dtype=np.int64)
    reconstructed_cells = raw_mesh_10k.cells.detach().cpu().numpy()
    if not np.array_equal(archived_cells, reconstructed_cells):
        raise ValueError("Archived 10k compacted connectivity does not match replay")
    archived_normals = np.asarray(archived["boundary_normals"], dtype=np.float32)
    pipeline_normal_max_abs = float(
        np.max(
            np.abs(
                archived_normals.astype(np.float64)
                - np.asarray(pipeline_normals_10k, dtype=np.float64)
            )
        )
    )
    if pipeline_normal_max_abs > ARCHIVED_PIPELINE_NORMAL_ABS_TOLERANCE:
        raise ValueError(
            "Archived 10k normals do not match current pipeline replay: "
            f"max_abs={pipeline_normal_max_abs}"
        )

    l_ref = _l_ref(raw_mesh_10k)
    archived_query = np.asarray(archived["query_points"], dtype=np.float32)
    reconstructed_query = (
        raw_centroids_10k - np.asarray(fixed_center, dtype=np.float32)
    ) / np.float32(l_ref)
    saved_coordinate_max_abs = float(
        np.max(
            np.abs(
                reconstructed_query.astype(np.float64)
                - archived_query.astype(np.float64)
            )
        )
    )
    if saved_coordinate_max_abs > RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE:
        raise ValueError(
            "Archived centered-coordinate reconstruction failed: "
            f"max_abs={saved_coordinate_max_abs}"
        )
    recovered_raw = archived_query * np.float32(l_ref) + np.asarray(
        fixed_center, dtype=np.float32
    )
    raw_frame_max_abs = float(
        np.max(
            np.abs(
                recovered_raw[:FIXED_QUERY_K].astype(np.float64)
                - raw_centroids_10k[:FIXED_QUERY_K].astype(np.float64)
            )
        )
    )
    if raw_frame_max_abs > RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE:
        raise ValueError(
            f"Archived raw-frame reconstruction failed: max_abs={raw_frame_max_abs}"
        )

    # Raw cell IDs are authoritative.  The archived composite repeats the
    # canonical reconstructed identity only after the independent numerical and
    # connectivity gates above; see SIGNATURE_ALGORITHM.
    canonical_signature = _translation_invariant_signature(
        query_points=raw_centroids_10k / np.float32(l_ref),
        compacted_cells=reconstructed_cells,
        identity_components=identity_components,
    )
    if cell_ids_10k.shape != (BASELINE_K,):
        raise ValueError(f"Historical cell IDs have shape {cell_ids_10k.shape}")
    del native_areas_10k  # Included through identity_components.
    return (
        canonical_signature,
        saved_coordinate_max_abs,
        raw_frame_max_abs,
        pipeline_normal_max_abs,
    )


def _validate_historical_starts() -> None:
    generator = torch.Generator()
    generator.manual_seed(READER_GENERATOR_SEED)
    for spec in CASE_SPECS:
        replayed = int(
            torch.randint(
                0,
                spec.n_master_cells - BASELINE_K + 1,
                (1,),
                generator=generator,
            ).item()
        )
        if replayed != spec.historical_start:
            raise ValueError(
                f"Historical RNG replay changed for {spec.case_id}: "
                f"expected {spec.historical_start}, got {replayed}"
            )


def _validate_frozen_inputs(
    *,
    repo_root: Path,
    dataset_root: Path,
    dataset_config_path: Path,
    resolved_config_path: Path,
    checkpoint_dir: Path,
    historical_metrics_path: Path,
) -> None:
    _require_sha256(
        dataset_root / "manifest.json", MANIFEST_SHA256, "DrivAerML manifest"
    )
    _require_sha256(
        dataset_config_path, DATASET_CONFIG_SHA256, "historical dataset config"
    )
    _require_sha256(
        resolved_config_path, RESOLVED_CONFIG_SHA256, "historical resolved config"
    )
    _require_sha256(
        historical_metrics_path,
        HISTORICAL_METRICS_SHA256,
        "historical 10k metrics",
    )
    _require_sha256(
        checkpoint_dir / MODEL_FILENAME, MODEL_SHA256, "deployed model checkpoint"
    )
    _require_sha256(
        checkpoint_dir / TRAINING_STATE_FILENAME,
        TRAINING_STATE_SHA256,
        "training-state checkpoint",
    )
    _require_sha256(
        checkpoint_dir / NORM_STATS_FILENAME,
        NORM_STATS_SHA256,
        "normalization stats",
    )
    source_tree_sha = _source_tree_manifest_sha256(repo_root)
    if source_tree_sha != EXECUTION_SOURCE_TREE_MANIFEST_SHA256:
        raise ValueError(
            "Execution source-tree manifest changed: expected "
            f"{EXECUTION_SOURCE_TREE_MANIFEST_SHA256}, got {source_tree_sha}"
        )

    manifest = json.loads((dataset_root / "manifest.json").read_text())
    cohort = tuple(sorted(str(value) for value in manifest["id_reference"]))
    if cohort != COHORT_CASE_IDS:
        raise ValueError(
            f"Manifest ID-reference cohort changed: expected {COHORT_CASE_IDS}, "
            f"got {cohort}"
        )
    _validate_historical_starts()


def _validate_reader(runtime: Runtime) -> None:
    paths = tuple(Path(path) for path in runtime.dataset.reader._paths)
    if len(paths) != 484:
        raise ValueError(f"Historical reader found {len(paths)} cases, expected 484")
    for spec in CASE_SPECS:
        path = paths[spec.reader_index]
        if spec.case_id not in path.parts:
            raise ValueError(
                f"Reader index {spec.reader_index} resolves to {path}, "
                f"not {spec.case_id}"
            )
        meta = json.loads((_vehicle_tensor_root(path) / "meta.json").read_text())
        n_cells = int(meta["cells"]["shape"][0])
        if n_cells != spec.n_master_cells:
            raise ValueError(
                f"{spec.case_id} has {n_cells} master cells, "
                f"expected {spec.n_master_cells}"
            )


def _center_diagnostic(
    primary_pressure: np.ndarray,
    fixed_pressure: np.ndarray,
    primary_metrics: Mapping[str, Any],
    fixed_metrics: Mapping[str, Any],
) -> dict[str, float]:
    result = {
        "pressure_prediction_relative_l2_difference": (
            _prediction_relative_difference(primary_pressure, fixed_pressure)
        ),
        "uniform_pressure_error_relative_change": _metric_relative_change(
            float(primary_metrics["uniform"]["pressure_relative_l2"]),
            float(fixed_metrics["uniform"]["pressure_relative_l2"]),
        ),
        "area_pressure_error_relative_change": _metric_relative_change(
            float(primary_metrics["area_weighted"]["pressure_relative_l2"]),
            float(fixed_metrics["area_weighted"]["pressure_relative_l2"]),
        ),
    }
    return result


def _center_diagnostics_pass(
    diagnostics_by_k: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> bool:
    return all(
        value <= CENTER_METRIC_RELATIVE_TOLERANCE
        for row in diagnostics_by_k.values()
        for diagnostic in row.values()
        for value in diagnostic.values()
    )


def _npz_key(spec: CaseSpec, k: int, field: str) -> str:
    return f"case_{spec.cohort_ordinal:02d}__k_{k:05d}__{field}"


def _run_case(
    *,
    runtime: Runtime,
    spec: CaseSpec,
    archived_metric_row: Mapping[str, Any],
    archive_root: Path,
    npz_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    reader_path = Path(runtime.dataset.reader._paths[spec.reader_index])
    vehicle_root = _vehicle_tensor_root(reader_path)
    _log(
        f"case={spec.case_id} ordinal={spec.cohort_ordinal} "
        f"reader_index={spec.reader_index} hashing_source"
    )
    source_identity = _source_identity(vehicle_root)
    raw_mesh = runtime.dataset.reader._load_sample(spec.reader_index)
    if raw_mesh.n_cells != spec.n_master_cells:
        raise ValueError(
            f"{spec.case_id} loaded n_cells={raw_mesh.n_cells}, "
            f"expected {spec.n_master_cells}"
        )

    max_ids = _cyclic_indices(
        spec.n_master_cells, spec.historical_start, max(RESOLUTIONS)
    )
    selection_by_k = {
        k: _cyclic_indices(spec.n_master_cells, spec.historical_start, k)
        for k in RESOLUTIONS
    }
    for k, ids in selection_by_k.items():
        if not np.array_equal(ids, max_ids[:k]):
            raise ValueError(f"{spec.case_id} k={k} is not a nested Kmax prefix")

    ids_10k = selection_by_k[BASELINE_K]
    subset_10k = _compact_explicit_cell_subset(raw_mesh, ids_10k, runtime.mesh_type)
    fixed_center_tensor = _pipeline_center_on_device(subset_10k, runtime.device)
    fixed_center = fixed_center_tensor.detach().float().cpu().numpy()

    archived_path = _archived_prediction_path(archive_root, spec)
    if not archived_path.is_dir():
        raise FileNotFoundError(f"Archived 10k prediction missing: {archived_path}")
    archived = _load_archived_10k(archived_path)

    fixed_q_reference: dict[str, np.ndarray] | None = None
    fixed_query_points_reference: np.ndarray | None = None
    primary_centers: dict[str, list[float]] = {}
    center_by_k: dict[str, dict[str, dict[str, float]]] = {}
    resolutions: list[dict[str, Any]] = []
    baseline_result: ForwardResult | None = None
    baseline_centroids: np.ndarray | None = None
    baseline_normals: np.ndarray | None = None
    baseline_pipeline_normals: np.ndarray | None = None
    baseline_areas: np.ndarray | None = None
    baseline_cells: np.ndarray | None = None

    for k in RESOLUTIONS:
        ids = selection_by_k[k]
        _log(f"case={spec.case_id} k={k} compacting")
        subset = _compact_explicit_cell_subset(raw_mesh, ids, runtime.mesh_type)
        compacted_cells = subset.cells.detach().cpu().numpy().astype("<i8", copy=False)
        raw_vertices = (
            subset.points[subset.cells]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
        )
        centroids, normals, areas = _native_geometry(subset)
        raw_pressure, raw_wss = _raw_physical_fields(subset)
        if subset.n_cells != k:
            raise ValueError(f"{spec.case_id} requested k={k}, got {subset.n_cells}")

        primary_domain, primary_center = _apply_pipeline(
            runtime, subset, fixed_center=None
        )
        primary_pipeline_normals, primary_normal_diagnostics = (
            _pipeline_normal_diagnostics(
                normals,
                primary_domain.boundaries["vehicle"],
                f"{spec.case_id} k={k} primary",
            )
        )
        _log(f"case={spec.case_id} k={k} forward=primary")
        primary = _run_forward(runtime, primary_domain)

        fixed_domain, applied_fixed_center = _apply_pipeline(
            runtime, subset, fixed_center=fixed_center_tensor
        )
        fixed_pipeline_normals, fixed_normal_diagnostics = _pipeline_normal_diagnostics(
            normals,
            fixed_domain.boundaries["vehicle"],
            f"{spec.case_id} k={k} fixed-center",
        )
        _log(f"case={spec.case_id} k={k} forward=fixed_center")
        fixed = _run_forward(runtime, fixed_domain)
        if not torch.equal(
            applied_fixed_center.detach().float().cpu(),
            fixed_center_tensor.detach().float().cpu(),
        ):
            raise ValueError("Fixed S10k center changed while applying pipeline")

        if not np.array_equal(primary.boundary_cells, compacted_cells):
            raise ValueError(f"{spec.case_id} k={k} primary cell order changed")
        if not np.array_equal(fixed.boundary_cells, compacted_cells):
            raise ValueError(f"{spec.case_id} k={k} fixed-center cell order changed")
        if not np.array_equal(primary.boundary_normals, primary_pipeline_normals):
            raise ValueError(
                f"{spec.case_id} k={k} primary pipeline normals changed during forward"
            )
        if not np.array_equal(fixed.boundary_normals, fixed_pipeline_normals):
            raise ValueError(
                f"{spec.case_id} k={k} fixed pipeline normals changed during forward"
            )
        if k == BASELINE_K and not np.array_equal(
            primary_pipeline_normals, fixed_pipeline_normals
        ):
            raise ValueError(
                f"{spec.case_id} k={k} identical primary/fixed centers changed normals"
            )
        for name in ("truth_pressure", "truth_wss"):
            if not np.array_equal(getattr(primary, name), getattr(fixed, name)):
                raise ValueError(
                    f"{spec.case_id} k={k} {name} changed with preprocessing center"
                )

        q_values = {
            "cell_ids": ids[:FIXED_QUERY_K],
            "truth_pressure": primary.truth_pressure[:FIXED_QUERY_K],
            "truth_wss": primary.truth_wss[:FIXED_QUERY_K],
            "normals": normals[:FIXED_QUERY_K],
            "areas": areas[:FIXED_QUERY_K],
            "fixed_center_pipeline_normals": fixed_pipeline_normals[:FIXED_QUERY_K],
            "raw_pressure": raw_pressure[:FIXED_QUERY_K],
            "raw_wss": raw_wss[:FIXED_QUERY_K],
            # Compacted-local vertex IDs may shift when later K prefixes add
            # lower global point IDs.  Ordered raw vertices, not local IDs, are
            # the cross-resolution physical-identity invariant.
            "raw_vertices": raw_vertices[:FIXED_QUERY_K],
        }
        if fixed_q_reference is None:
            fixed_q_reference = {
                name: np.array(value, copy=True) for name, value in q_values.items()
            }
            fixed_query_points_reference = np.array(
                fixed.query_points[:FIXED_QUERY_K], copy=True
            )
        else:
            for name, reference in fixed_q_reference.items():
                if not np.array_equal(q_values[name], reference):
                    raise ValueError(
                        f"{spec.case_id} k={k} fixed-Q identity changed for {name}"
                    )
            if not np.array_equal(
                fixed.query_points[:FIXED_QUERY_K], fixed_query_points_reference
            ):
                raise ValueError(
                    f"{spec.case_id} k={k} fixed-center Q coordinates changed"
                )

        primary_coupled, primary_fixed_q = _score_forward_result(
            primary,
            normals,
            areas,
            n_master_cells=spec.n_master_cells,
        )
        fixed_coupled, fixed_fixed_q = _score_forward_result(
            fixed,
            normals,
            areas,
            n_master_cells=spec.n_master_cells,
        )
        center_row = {
            "coupled": _center_diagnostic(
                primary.pressure,
                fixed.pressure,
                primary_coupled,
                fixed_coupled,
            ),
            "fixed_q": _center_diagnostic(
                primary.pressure[:FIXED_QUERY_K],
                fixed.pressure[:FIXED_QUERY_K],
                primary_fixed_q,
                fixed_fixed_q,
            ),
        }
        center_by_k[str(k)] = center_row
        center_max = max(
            value for diagnostic in center_row.values() for value in diagnostic.values()
        )
        if center_max > CENTER_METRIC_RELATIVE_TOLERANCE:
            raise ValueError(
                f"{spec.case_id} k={k} preprocessing-center equivalence failed: "
                f"{center_row}"
            )

        resolutions.append(
            {
                "k": k,
                "selection": {
                    "cell_ids_sha256_int64": _sha256_array(ids, "<i8"),
                    "q_prefix_sha256_int64": _sha256_array(ids[:FIXED_QUERY_K], "<i8"),
                    "nested_prefix_passed": True,
                },
                "metrics": {
                    "uniform": {
                        "coupled": primary_coupled["uniform"],
                        "fixed_q": primary_fixed_q["uniform"],
                    },
                    "area_weighted": {
                        "coupled": primary_coupled["area_weighted"],
                        "fixed_q": primary_fixed_q["area_weighted"],
                    },
                },
                "normal_diagnostics": {
                    "primary": primary_normal_diagnostics,
                    "fixed_center": fixed_normal_diagnostics,
                },
                "finite_checks_passed": True,
            }
        )
        primary_centers[str(k)] = [
            _finite(value, f"{spec.case_id} k={k} primary center")
            for value in primary_center.detach().float().cpu().tolist()
        ]

        array_values = {
            "raw_cell_ids_int64": ids.astype("<i8", copy=False),
            "compacted_cells_int64": compacted_cells,
            "raw_centroids_float32": centroids.astype("<f4", copy=False),
            "native_normals_float32": normals.astype("<f4", copy=False),
            "primary_pipeline_normals_float32": primary_pipeline_normals.astype(
                "<f4", copy=False
            ),
            "fixed_center_pipeline_normals_float32": fixed_pipeline_normals.astype(
                "<f4", copy=False
            ),
            "native_areas_float64": areas.astype("<f8", copy=False),
            "truth_pressure_float32": primary.truth_pressure.astype("<f4", copy=False),
            "truth_wss_float32": primary.truth_wss.astype("<f4", copy=False),
            "primary_query_points_float32": primary.query_points.astype(
                "<f4", copy=False
            ),
            "primary_pressure_float32": primary.pressure.astype("<f4", copy=False),
            "primary_wss_float32": primary.wss.astype("<f4", copy=False),
            "fixed_center_query_points_float32": fixed.query_points.astype(
                "<f4", copy=False
            ),
            "fixed_center_pressure_float32": fixed.pressure.astype("<f4", copy=False),
            "fixed_center_wss_float32": fixed.wss.astype("<f4", copy=False),
        }
        for field, value in array_values.items():
            npz_arrays[_npz_key(spec, k, field)] = np.ascontiguousarray(value)

        if k == BASELINE_K:
            baseline_result = primary
            baseline_centroids = centroids
            baseline_normals = normals
            baseline_pipeline_normals = primary_pipeline_normals
            baseline_areas = areas
            baseline_cells = compacted_cells

        del primary_domain, fixed_domain, primary, fixed, subset
        torch.cuda.empty_cache()

    if (
        fixed_q_reference is None
        or baseline_result is None
        or baseline_centroids is None
        or baseline_normals is None
        or baseline_pipeline_normals is None
        or baseline_areas is None
        or baseline_cells is None
    ):
        raise RuntimeError(f"{spec.case_id} did not produce all required references")

    q_pressure = np.asarray(fixed_q_reference["truth_pressure"], dtype=np.float32)
    q_normals = np.asarray(fixed_q_reference["normals"], dtype=np.float32)
    q_areas = np.asarray(fixed_q_reference["areas"], dtype=np.float64)
    identity_components = {
        "raw_cell_ids_sha256_int64": _sha256_array(ids_10k, "<i8"),
        "truth_pressure_sha256_float32": _sha256_array(
            baseline_result.truth_pressure, "<f4"
        ),
        "truth_wss_sha256_float32": _sha256_array(baseline_result.truth_wss, "<f4"),
        "normals_sha256_float32": _sha256_array(baseline_normals, "<f4"),
        "native_areas_sha256_float64": _sha256_array(baseline_areas, "<f8"),
    }
    (
        canonical_signature,
        saved_coordinate_max_abs,
        raw_frame_q_max_abs,
        saved_pipeline_normal_max_abs,
    ) = _archive_parity(
        archived=archived,
        raw_mesh_10k=subset_10k,
        raw_centroids_10k=baseline_centroids,
        pipeline_normals_10k=baseline_pipeline_normals,
        native_areas_10k=baseline_areas,
        cell_ids_10k=ids_10k,
        fixed_center=fixed_center,
        identity_components=identity_components,
    )

    archived_pressure_l2 = float(archived_metric_row["metrics"]["pressure_l2"])
    replay_pressure_l2 = next(
        row["metrics"]["uniform"]["coupled"]["pressure_relative_l2"]
        for row in resolutions
        if row["k"] == BASELINE_K
    )
    if abs(replay_pressure_l2 - archived_pressure_l2) > ARCHIVED_CASE_ABS_TOLERANCE:
        raise ValueError(
            f"{spec.case_id} 10k pressure replay {replay_pressure_l2} differs "
            f"from archived {archived_pressure_l2}"
        )
    expected_prefix = f"{spec.reader_index:05d}_{spec.case_id}_"
    if not str(archived_metric_row["sample_id"]).startswith(expected_prefix):
        raise ValueError(
            f"{spec.case_id} historical sample ID changed: "
            f"{archived_metric_row['sample_id']}"
        )

    truth_rms_q = math.sqrt(
        float(np.mean(q_pressure.astype(np.float64) ** 2, dtype=np.float64))
    )
    truth_rms_s10 = math.sqrt(
        float(
            np.mean(
                baseline_result.truth_pressure.astype(np.float64) ** 2,
                dtype=np.float64,
            )
        )
    )
    center_passed = (
        raw_frame_q_max_abs <= RAW_FRAME_RECONSTRUCTION_ABS_TOLERANCE
        and _center_diagnostics_pass(center_by_k)
    )
    if not center_passed:
        raise ValueError(f"{spec.case_id} center diagnostics did not pass")

    del raw_mesh, subset_10k
    return {
        "cohort_ordinal": spec.cohort_ordinal,
        "case_id": spec.case_id,
        "reader_index": spec.reader_index,
        "n_master_cells": spec.n_master_cells,
        "historical_start": spec.historical_start,
        "source_identity": source_identity,
        "historical_10k": {
            "reader_seed_fork_chain_sha256": _reader_seed_fork_chain_sha256(),
            "seed_fork_chain_replayed": True,
            "selection_sha256_int64": _sha256_array(ids_10k, "<i8"),
            "canonical_reconstructed_signature_sha256": canonical_signature,
            "saved_artifact_coordinate_max_abs_error": _finite(
                saved_coordinate_max_abs,
                "saved_artifact_coordinate_max_abs_error",
            ),
            "saved_artifact_pipeline_normals_max_abs_error": _finite(
                saved_pipeline_normal_max_abs,
                "saved_artifact_pipeline_normals_max_abs_error",
            ),
            "saved_artifact_parity_passed": True,
            "exact_archived_row_available": True,
            "archived_uniform_pressure_relative_l2": archived_pressure_l2,
        },
        "fixed_q": {
            "raw_cell_ids_sha256_int64": _sha256_array(
                fixed_q_reference["cell_ids"], "<i8"
            ),
            "truth_pressure_sha256_float32": _sha256_array(q_pressure, "<f4"),
            "normals_sha256_float32": _sha256_array(q_normals, "<f4"),
            "native_areas_sha256_float64": _sha256_array(q_areas, "<f8"),
            "truth_rms": _finite(truth_rms_q, "fixed-Q truth RMS"),
            **_native_area_reference(q_areas, "fixed-Q"),
            "identity_checks_passed": True,
        },
        "s10k_reference": {
            "truth_rms": _finite(truth_rms_s10, "S10k truth RMS"),
            **_native_area_reference(baseline_areas, "S10k"),
        },
        "centers": {
            "fixed_s10k": [
                _finite(value, f"{spec.case_id} fixed S10k center")
                for value in fixed_center.tolist()
            ],
            "primary_by_k": primary_centers,
            "raw_frame_q_reconstruction_max_abs": _finite(
                raw_frame_q_max_abs, "raw_frame_q_reconstruction_max_abs"
            ),
            "by_k": center_by_k,
            "passed": True,
        },
        "resolutions": resolutions,
    }


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _provenance(
    *,
    producer_path: Path,
    resolved_config_path: Path,
    npz_path: Path,
) -> dict[str, Any]:
    cuda_runtime = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    if cuda_runtime is None or cudnn_version is None:
        raise RuntimeError(
            "Production provenance requires concrete CUDA and cuDNN versions"
        )
    current_device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(current_device)
    return {
        "producer_path": str(producer_path.resolve()),
        "producer_sha256": _sha256_file(producer_path),
        "config_sha256": _sha256_file(resolved_config_path),
        "command": list(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "physicsnemo": _package_version(
                "nvidia-physicsnemo",
                fallback=_package_version("physicsnemo"),
            ),
        },
        "hardware": {
            "cuda_runtime": str(cuda_runtime),
            "visible_cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": str(torch.cuda.get_device_name(current_device)),
            "cuda_device_capability": [int(capability[0]), int(capability[1])],
            "cudnn_version": int(cudnn_version),
        },
        "rescoring_npz_path": str(npz_path.resolve()),
        "rescoring_npz_sha256": _sha256_file(npz_path),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument(
        "--historical-predictions",
        type=Path,
        required=True,
        help="Historical res10000 .../predictions directory.",
    )
    parser.add_argument("--lane-ordinal", type=int, required=True)
    parser.add_argument("--lane-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.lane_count <= 0:
        raise ValueError(f"lane-count must be positive, got {args.lane_count}")
    if not 0 <= args.lane_ordinal < args.lane_count:
        raise ValueError(
            f"lane-ordinal {args.lane_ordinal} outside [0,{args.lane_count})"
        )
    for output in (args.output_json, args.output_npz):
        if output.exists() or output.with_name(f"{output.name}.sha256").exists():
            raise FileExistsError(f"Refusing to overwrite output or sidecar: {output}")
    producer_path = Path(__file__).resolve()
    _validate_frozen_inputs(
        repo_root=args.repo_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        dataset_config_path=args.dataset_config.resolve(),
        resolved_config_path=args.resolved_config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        historical_metrics_path=args.historical_metrics.resolve(),
    )
    historical_rows = _parse_historical_metrics(args.historical_metrics.resolve())
    runtime = _load_runtime(
        repo_root=args.repo_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        dataset_config_path=args.dataset_config.resolve(),
        resolved_config_path=args.resolved_config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
    )
    _validate_reader(runtime)

    lane_specs = [
        spec
        for spec in CASE_SPECS
        if spec.cohort_ordinal % args.lane_count == args.lane_ordinal
    ]
    if not lane_specs:
        raise ValueError(
            f"Lane {args.lane_ordinal}/{args.lane_count} has no cohort cases"
        )
    _log(
        f"lane={args.lane_ordinal}/{args.lane_count} "
        f"cases={[spec.case_id for spec in lane_specs]}"
    )
    npz_arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for completed, spec in enumerate(lane_specs, start=1):
        cases.append(
            _run_case(
                runtime=runtime,
                spec=spec,
                archived_metric_row=historical_rows[spec.case_id],
                archive_root=args.historical_predictions.resolve(),
                npz_arrays=npz_arrays,
            )
        )
        _log(
            f"COMPLETED_UNITS={completed}/{len(lane_specs)} "
            f"case={spec.case_id} lane={args.lane_ordinal}"
        )

    if [case["cohort_ordinal"] for case in cases] != [
        spec.cohort_ordinal for spec in lane_specs
    ]:
        raise RuntimeError("Case output order changed within lane")
    _atomic_write_npz(args.output_npz, npz_arrays)
    _write_sha256_sidecar(args.output_npz)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": PASSED_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lane": {
            "ordinal": args.lane_ordinal,
            "count": args.lane_count,
        },
        "frozen": FROZEN_CONTRACT,
        "cases": cases,
        "provenance": _provenance(
            producer_path=producer_path,
            resolved_config_path=args.resolved_config.resolve(),
            npz_path=args.output_npz,
        ),
    }
    _atomic_write_bytes(
        args.output_json,
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    _write_sha256_sidecar(args.output_json)
    _log(
        f"PASSED lane={args.lane_ordinal}/{args.lane_count} "
        f"json={args.output_json} npz={args.output_npz}"
    )


if __name__ == "__main__":
    main()
