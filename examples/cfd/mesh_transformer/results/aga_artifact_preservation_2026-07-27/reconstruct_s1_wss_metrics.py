#!/usr/bin/env python3
"""Read-only reconstruction of historical S1 WSS metrics from AGA memmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EPS = 1.0e-8
SEEDS = tuple(range(42, 47))
SPLITS = ("s1_id_reference", "s1_extreme_test")
FAMILIES = ("mesh_transformer", "geotransolver")
METRIC_KEYS = (
    "wss_l2_historical_pointwise_mean",
    "wss_l2_current_whole_field_frobenius",
    "wss_magnitude_relative_l2",
    "wss_x_relative_l2",
    "wss_y_relative_l2",
    "wss_z_relative_l2",
)
COMPONENT_NAMES = ("x", "y", "z")
HASH_CHUNK_BYTES = 8 * 1024 * 1024
SUCCESSFUL_FINAL_JOB_IDS = {
    ("mesh_transformer", 42): 255921,
    ("mesh_transformer", 43): 255923,
    ("mesh_transformer", 44): 255925,
    ("mesh_transformer", 45): 259410,
    ("mesh_transformer", 46): 259411,
    ("geotransolver", 42): 255922,
    ("geotransolver", 43): 255924,
    ("geotransolver", 44): 255926,
    ("geotransolver", 45): 259341,
    ("geotransolver", 46): 259343,
}
HISTORICAL_FAILED_JOB_IDS = {
    ("mesh_transformer", 42): (254760,),
    ("mesh_transformer", 43): (254761,),
    ("mesh_transformer", 44): (254762,),
    ("mesh_transformer", 45): (259340,),
    ("mesh_transformer", 46): (259342,),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Return a content identity while checking that the file stayed stable."""
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest,
    }


def identity_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        identity = {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def run_id(family: str, seed: int) -> str:
    if family == "geotransolver":
        return f"oods1_geotransolver_surface_seed{seed}"
    if seed <= 44:
        return f"oods1_mesh_transformer_surface_flagship_s1_seed{seed}"
    return f"oods1_mesh_transformer_surface_flagship_seed{seed}"


def output_family(family: str) -> str:
    return "drivaer_ood_deep_geot" if family == "geotransolver" else "drivaer_ood_deep"


def launcher_name(family: str, seed: int) -> str:
    if family == "geotransolver":
        return "mt_ood_deepeval_geot.sbatch"
    return "mt_ood_deepeval.sbatch" if seed <= 44 else "mt_ood_deepeval_ext.sbatch"


def parse_jsonl(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected object, got {type(value).__name__}")
                records.append({"line_number": line_number, "value": value})
            except (json.JSONDecodeError, TypeError) as error:
                invalid.append(
                    {
                        "line_number": line_number,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    phases = Counter(item["value"].get("phase", "<missing>") for item in records)
    summary_positions = [
        i
        for i, item in enumerate(records)
        if item["value"].get("phase") == "infer_summary"
    ]
    if not summary_positions:
        raise ValueError(f"no infer_summary record in {path}")
    summary_position = summary_positions[-1]
    config_positions = [
        i
        for i, item in enumerate(records[:summary_position])
        if item["value"].get("phase") == "config"
    ]
    if not config_positions:
        raise ValueError(f"no config record before final infer_summary in {path}")
    config_position = config_positions[-1]
    segment = records[config_position : summary_position + 1]
    steps = {
        item["value"]["sample_id"]: item["value"]
        for item in segment
        if item["value"].get("phase") == "infer_step"
        and isinstance(item["value"].get("sample_id"), str)
    }
    summaries = [
        {
            "line_number": records[position]["line_number"],
            "num_samples": records[position]["value"].get("num_samples"),
            "metrics": records[position]["value"].get("metrics"),
            "timestamp": records[position]["value"].get("ts"),
        }
        for position in summary_positions
    ]
    return {
        "line_count": len(records) + len(invalid),
        "invalid_lines": invalid,
        "phase_counts": dict(sorted(phases.items())),
        "all_infer_summaries": summaries,
        "selected_segment": {
            "config_line_number": records[config_position]["line_number"],
            "summary_line_number": records[summary_position]["line_number"],
            "config": records[config_position]["value"],
            "summary": records[summary_position]["value"],
            "infer_steps_by_sample_id": steps,
            "infer_step_count": len(steps),
        },
    }


def log_evidence(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "file": file_record(path),
        "marker_counts": {
            "traceback": text.count("Traceback (most recent call last):"),
            "keyerror_wss_mag_l2": text.count("KeyError: 'wss_mag_l2'"),
            "checkpoint_lookup_failure": text.count("No valid checkpoint"),
            "inference_complete": text.count("Inference complete!"),
            "split_exit_zero": text.count("EXIT=0"),
            "all_done": text.count("ALL_DONE"),
        },
    }


def exact_job_log(repo: Path, job_id: int) -> Path:
    matches = sorted((repo / "sbatch_logs").glob(f"*_{job_id}.log"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one sbatch log for job {job_id}, found {len(matches)}"
        )
    return matches[0]


def point_data_paths(case_dir: Path) -> tuple[Path, Path, Path]:
    point_data = case_dir / "_tensordict" / "interior" / "_tensordict" / "point_data"
    return (
        point_data / "meta.json",
        point_data / "pred_wss.memmap",
        point_data / "true_wss.memmap",
    )


def load_wss_memmap(
    path: Path, meta: dict[str, Any], field: str
) -> tuple[np.memmap, tuple[int, ...], str]:
    spec = meta.get(field)
    if not isinstance(spec, dict):
        raise ValueError(f"{field!r} metadata absent from {path.parent / 'meta.json'}")
    shape = tuple(int(value) for value in spec["shape"])
    dtype_name = spec["dtype"]
    dtype_map = {
        "torch.float32": np.dtype("<f4"),
        "torch.float64": np.dtype("<f8"),
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype_name!r} for {path}")
    dtype = dtype_map[dtype_name]
    expected_size = math.prod(shape) * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"size mismatch for {path}: {actual_size} != {expected_size}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape), shape, dtype_name


def relative_l2(numerator_sq: float, denominator_sq: float) -> float:
    return math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + EPS)


def compute_case(case_dir: Path) -> dict[str, Any]:
    meta_path, pred_path, true_path = point_data_paths(case_dir)
    input_records = [
        file_record(meta_path),
        file_record(pred_path),
        file_record(true_path),
    ]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pred_map, pred_shape, pred_dtype = load_wss_memmap(pred_path, meta, "pred_wss")
    true_map, true_shape, true_dtype = load_wss_memmap(true_path, meta, "true_wss")
    if pred_shape != true_shape or len(pred_shape) != 2 or pred_shape[-1] != 3:
        raise ValueError(
            f"expected matching (N, 3) WSS arrays, got {pred_shape}, {true_shape}"
        )
    if pred_dtype != true_dtype:
        raise ValueError(f"dtype mismatch: {pred_dtype} != {true_dtype}")

    pred = np.asarray(pred_map, dtype=np.float64)
    true = np.asarray(true_map, dtype=np.float64)
    del pred_map, true_map
    pred_nonfinite = int(pred.size - np.count_nonzero(np.isfinite(pred)))
    true_nonfinite = int(true.size - np.count_nonzero(np.isfinite(true)))
    if pred_nonfinite or true_nonfinite:
        raise ValueError(
            f"non-finite WSS entries: pred={pred_nonfinite}, true={true_nonfinite}"
        )

    error = pred - true
    point_error_norm = np.sqrt(np.sum(error * error, axis=-1))
    point_target_norm = np.sqrt(np.sum(true * true, axis=-1))
    pointwise_ratio = point_error_norm / (point_target_norm + EPS)
    pred_magnitude = np.sqrt(np.sum(pred * pred, axis=-1))
    true_magnitude = point_target_norm
    magnitude_error = pred_magnitude - true_magnitude

    error_sq_sum = float(np.sum(error * error))
    target_sq_sum = float(np.sum(true * true))
    magnitude_error_sq_sum = float(np.sum(magnitude_error * magnitude_error))
    magnitude_target_sq_sum = float(np.sum(true_magnitude * true_magnitude))
    component_error_sq_sum = np.sum(error * error, axis=0)
    component_target_sq_sum = np.sum(true * true, axis=0)
    metrics = {
        "wss_l2_historical_pointwise_mean": float(np.mean(pointwise_ratio)),
        "wss_l2_current_whole_field_frobenius": relative_l2(
            error_sq_sum, target_sq_sum
        ),
        "wss_magnitude_relative_l2": relative_l2(
            magnitude_error_sq_sum, magnitude_target_sq_sum
        ),
    }
    for index, component in enumerate(COMPONENT_NAMES):
        metrics[f"wss_{component}_relative_l2"] = relative_l2(
            float(component_error_sq_sum[index]),
            float(component_target_sq_sum[index]),
        )

    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (meta_path, pred_path, true_path)
    }
    for record in input_records:
        path = Path(record["path"])
        if after[path] != (record["size_bytes"], record["mtime_ns"]):
            raise RuntimeError(f"file changed while computing metrics: {path}")

    return {
        "sample_id": case_dir.stem,
        "case_path": str(case_dir),
        "shape": list(pred_shape),
        "dtype": pred_dtype,
        "zero_target_point_norm_count": int(np.count_nonzero(point_target_norm == 0)),
        "metrics": metrics,
        "sufficient_sums": {
            "pointwise_ratio_sum": float(np.sum(pointwise_ratio)),
            "n_points": int(pred_shape[0]),
            "error_sq_sum": error_sq_sum,
            "target_sq_sum": target_sq_sum,
            "magnitude_error_sq_sum": magnitude_error_sq_sum,
            "magnitude_target_sq_sum": magnitude_target_sq_sum,
            "component_error_sq_sum": [
                float(value) for value in component_error_sq_sum
            ],
            "component_target_sq_sum": [
                float(value) for value in component_target_sq_sum
            ],
        },
        "input_files": {
            "point_data_meta": input_records[0],
            "pred_wss": input_records[1],
            "true_wss": input_records[2],
        },
    }


def arithmetic_mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(collected) / len(collected)


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_mean = {
        key: arithmetic_mean(case["metrics"][key] for case in cases)
        for key in METRIC_KEYS
    }
    pointwise_sum = math.fsum(
        case["sufficient_sums"]["pointwise_ratio_sum"] for case in cases
    )
    n_points = sum(case["sufficient_sums"]["n_points"] for case in cases)
    error_sq_sum = math.fsum(case["sufficient_sums"]["error_sq_sum"] for case in cases)
    target_sq_sum = math.fsum(
        case["sufficient_sums"]["target_sq_sum"] for case in cases
    )
    magnitude_error_sq_sum = math.fsum(
        case["sufficient_sums"]["magnitude_error_sq_sum"] for case in cases
    )
    magnitude_target_sq_sum = math.fsum(
        case["sufficient_sums"]["magnitude_target_sq_sum"] for case in cases
    )
    component_error_sq_sum = [
        math.fsum(
            case["sufficient_sums"]["component_error_sq_sum"][index] for case in cases
        )
        for index in range(3)
    ]
    component_target_sq_sum = [
        math.fsum(
            case["sufficient_sums"]["component_target_sq_sum"][index] for case in cases
        )
        for index in range(3)
    ]
    pooled = {
        "wss_l2_historical_pointwise_mean": pointwise_sum / n_points,
        "wss_l2_current_whole_field_frobenius": relative_l2(
            error_sq_sum, target_sq_sum
        ),
        "wss_magnitude_relative_l2": relative_l2(
            magnitude_error_sq_sum, magnitude_target_sq_sum
        ),
    }
    for index, component in enumerate(COMPONENT_NAMES):
        pooled[f"wss_{component}_relative_l2"] = relative_l2(
            component_error_sq_sum[index], component_target_sq_sum[index]
        )
    return {
        "successful_case_count": len(cases),
        "point_count": n_points,
        "case_mean": case_mean,
        "pooled_over_all_points": pooled,
    }


def process_group(repo: Path, family: str, split: str, seed: int) -> dict[str, Any]:
    identifier = run_id(family, seed)
    group_dir = repo / "output" / output_family(family) / split / identifier
    metrics_path = group_dir / "metrics.jsonl"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"missing group directory: {group_dir}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")

    metrics_record = file_record(metrics_path)
    parsed = parse_jsonl(metrics_path)
    selected = parsed["selected_segment"]
    logged_summary = selected["summary"]
    expected_cases = int(logged_summary["num_samples"])
    prediction_dir = group_dir / "predictions"
    case_dirs = sorted(path for path in prediction_dir.glob("*.pdmsh") if path.is_dir())
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for completed, case_dir in enumerate(case_dirs, 1):
        try:
            cases.append(compute_case(case_dir))
        except Exception as error:  # preserve every per-case failure and continue
            failures.append(
                {
                    "case_path": str(case_dir),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if completed % 25 == 0 or completed == len(case_dirs):
            print(
                f"COMPLETED_UNITS={completed}/{len(case_dirs)} "
                f"group={family}/{split}/seed{seed}",
                flush=True,
            )

    if not cases:
        raise RuntimeError(f"no readable WSS cases in {prediction_dir}")
    summary = summarize_cases(cases)
    steps = selected.pop("infer_steps_by_sample_id")
    comparisons: list[dict[str, Any]] = []
    missing_step_sample_ids: list[str] = []
    for case in cases:
        step = steps.get(case["sample_id"])
        if step is None:
            missing_step_sample_ids.append(case["sample_id"])
            continue
        logged = step.get("metrics", {}).get("wss_l2")
        if not isinstance(logged, (int, float)):
            missing_step_sample_ids.append(case["sample_id"])
            continue
        recomputed = case["metrics"]["wss_l2_historical_pointwise_mean"]
        comparisons.append(
            {
                "sample_id": case["sample_id"],
                "logged_wss_l2": float(logged),
                "recomputed_historical_pointwise_mean": recomputed,
                "absolute_delta": abs(recomputed - float(logged)),
            }
        )
    comparison_deltas = [item["absolute_delta"] for item in comparisons]
    logged_metrics = logged_summary.get("metrics", {})
    logged_group_wss = logged_metrics.get("wss_l2")
    recomputed_group_wss = summary["case_mean"]["wss_l2_historical_pointwise_mean"]
    group_comparison = {
        "logged_infer_summary_wss_l2": logged_group_wss,
        "recomputed_historical_pointwise_case_mean": recomputed_group_wss,
        "absolute_delta": (
            abs(recomputed_group_wss - float(logged_group_wss))
            if isinstance(logged_group_wss, (int, float))
            else None
        ),
        "matched_infer_steps": len(comparisons),
        "missing_infer_step_sample_ids": missing_step_sample_ids,
        "per_case_max_absolute_delta": (
            max(comparison_deltas) if comparison_deltas else None
        ),
        "per_case_mean_absolute_delta": (
            arithmetic_mean(comparison_deltas) if comparison_deltas else None
        ),
    }
    raw_records = [record for case in cases for record in case["input_files"].values()]
    output_log = (
        repo / "output" / output_family(family) / f"eval_{split}_seed{seed}.log"
    )
    output_log_evidence = log_evidence(output_log) if output_log.is_file() else None
    return {
        "family": family,
        "split": split,
        "seed": seed,
        "run_id": identifier,
        "group_path": str(group_dir),
        "prediction_path": str(prediction_dir),
        "metrics_file": metrics_record,
        "evaluation_log": (
            output_log_evidence["file"] if output_log_evidence is not None else None
        ),
        "evaluation_log_history": output_log_evidence,
        "metrics_jsonl": {
            key: value for key, value in parsed.items() if key != "selected_segment"
        },
        "selected_metrics_segment": selected,
        "logged_vs_recomputed": group_comparison,
        "expected_case_count_from_selected_summary": expected_cases,
        "raw_case_directory_count": len(case_dirs),
        "missing_case_count": max(expected_cases - len(case_dirs), 0),
        "unexpected_case_count": max(len(case_dirs) - expected_cases, 0),
        "case_processing_failures": failures,
        "summary": summary,
        "raw_wss_input_tree_sha256": identity_digest(raw_records),
        "cases": cases,
    }


def find_checkpoint_records(
    repo: Path,
    family: str,
    seed: int,
    selected_config: dict[str, Any],
) -> dict[str, Any]:
    identifier = run_id(family, seed)
    run_dir = (
        repo
        / "examples"
        / "cfd"
        / "external_aerodynamics"
        / "unified_external_aero_recipe"
        / "runs"
        / identifier
    )
    epoch = int(selected_config["epoch"])
    checkpoint_dir = run_dir / "checkpoints"
    mdlus_matches = sorted(checkpoint_dir.glob(f"*.0.{epoch}.mdlus"))
    pt_matches = sorted(checkpoint_dir.glob(f"checkpoint.0.{epoch}.pt"))
    if len(mdlus_matches) != 1:
        raise ValueError(
            f"expected one epoch-{epoch} mdlus for {identifier}, "
            f"found {len(mdlus_matches)}"
        )
    if len(pt_matches) != 1:
        raise ValueError(
            f"expected one epoch-{epoch} pt for {identifier}, found {len(pt_matches)}"
        )
    config_checkpoint = selected_config.get("checkpoint")
    if not isinstance(config_checkpoint, str) or Path(config_checkpoint).name != (
        "checkpoints"
    ):
        raise ValueError(
            f"unexpected logged checkpoint path for {identifier}: {config_checkpoint!r}"
        )
    successful_job_id = SUCCESSFUL_FINAL_JOB_IDS[(family, seed)]
    failed_job_ids = HISTORICAL_FAILED_JOB_IDS.get((family, seed), ())
    return {
        "family": family,
        "seed": seed,
        "run_id": identifier,
        "selected_logged_config": selected_config,
        "resolved_config": file_record(run_dir / "resolved_config.yaml"),
        "model_checkpoint": file_record(mdlus_matches[0]),
        "training_checkpoint": file_record(pt_matches[0]),
        "normalization_stats": file_record(checkpoint_dir / "norm_stats.pt"),
        "successful_final_job": {
            "job_id": successful_job_id,
            **log_evidence(exact_job_log(repo, successful_job_id)),
        },
        "historical_failed_jobs": [
            {
                "job_id": job_id,
                **log_evidence(exact_job_log(repo, job_id)),
            }
            for job_id in failed_job_ids
        ],
    }


def static_provenance(repo: Path, wrapper: Path) -> dict[str, Any]:
    recipe = (
        repo
        / "examples"
        / "cfd"
        / "external_aerodynamics"
        / "unified_external_aero_recipe"
    )
    paths = [
        recipe / "src" / "metrics.py",
        recipe / "src" / "infer.py",
        recipe / "src" / "datasets.py",
        recipe / "src" / "nondim.py",
        recipe / "src" / "output_normalize.py",
        recipe / "conf" / "infer.yaml",
        recipe / "conf" / "base.yaml",
        recipe / "datasets" / "drivaer_ml_surface.yaml",
        recipe / "conf" / "model" / "mesh_transformer_surface_flagship.yaml",
        recipe / "conf" / "model" / "geotransolver_surface.yaml",
        repo / "mt_ood_deepeval.sbatch",
        repo / "mt_ood_deepeval_ext.sbatch",
        repo / "mt_ood_deepeval_geot.sbatch",
        (repo.parent / "mt_datasets" / "drivaerml_ood_s1train" / "manifest.json"),
        Path(__file__).resolve(),
        wrapper.resolve(),
    ]
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if path.is_file():
            records.append(file_record(path))
        else:
            missing.append(str(path))
    return {
        "repo_path": str(repo),
        "git_metadata_present": (repo / ".git").exists(),
        "files": records,
        "missing_files": missing,
    }


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        raise ZeroDivisionError("zero ID metric in degradation ratio")
    return numerator / denominator


def build_differential(groups: list[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (group["family"], group["split"], group["seed"]): group for group in groups
    }
    metrics: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        per_seed: dict[str, dict[str, float]] = {}
        for family in FAMILIES:
            family_values: dict[str, float] = {}
            for seed in SEEDS:
                id_value = index[(family, "s1_id_reference", seed)]["summary"][
                    "case_mean"
                ][metric]
                extreme_value = index[(family, "s1_extreme_test", seed)]["summary"][
                    "case_mean"
                ][metric]
                family_values[str(seed)] = ratio(extreme_value, id_value)
            per_seed[family] = family_values
        mt_values = list(per_seed["mesh_transformer"].values())
        geot_values = list(per_seed["geotransolver"].values())
        paired_differences = {
            str(seed): (
                per_seed["geotransolver"][str(seed)]
                - per_seed["mesh_transformer"][str(seed)]
            )
            for seed in SEEDS
        }
        metrics[metric] = {
            "per_seed_extreme_over_id": per_seed,
            "family_summary": {
                family: {
                    "mean": arithmetic_mean(per_seed[family].values()),
                    "min": min(per_seed[family].values()),
                    "max": max(per_seed[family].values()),
                }
                for family in FAMILIES
            },
            "lower_mesh_transformer_range_disjoint_from_geotransolver": (
                max(mt_values) < min(geot_values)
            ),
            "disjoint_separation_gap_geot_min_minus_mt_max": (
                min(geot_values) - max(mt_values)
            ),
            "same_seed_label_geot_minus_mesh_transformer": paired_differences,
            "same_seed_label_positive_difference_count": sum(
                value > 0.0 for value in paired_differences.values()
            ),
        }
    return {
        "ordering": "lower extreme/ID degradation ratio is better",
        "statistical_scope": (
            "Descriptive 5-vs-5 range separation only. Shared seed labels and "
            "identical evaluation targets do not by themselves declare a "
            "paired or independent inferential sampling model."
        ),
        "metrics": metrics,
    }


def target_identity_checks(groups: list[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (group["family"], group["split"], group["seed"]): group for group in groups
    }
    checks: list[dict[str, Any]] = []
    for split in SPLITS:
        for seed in SEEDS:
            mt = {
                case["sample_id"]: case["input_files"]["true_wss"]["sha256"]
                for case in index[("mesh_transformer", split, seed)]["cases"]
            }
            geot = {
                case["sample_id"]: case["input_files"]["true_wss"]["sha256"]
                for case in index[("geotransolver", split, seed)]["cases"]
            }
            common = sorted(set(mt) & set(geot))
            identical = sum(mt[sample_id] == geot[sample_id] for sample_id in common)
            checks.append(
                {
                    "split": split,
                    "seed": seed,
                    "mesh_transformer_case_count": len(mt),
                    "geotransolver_case_count": len(geot),
                    "common_sample_id_count": len(common),
                    "identical_true_wss_sha256_count": identical,
                    "all_case_ids_and_true_wss_hashes_identical": (
                        set(mt) == set(geot) and identical == len(common)
                    ),
                }
            )
    return {
        "same_seed_cross_family": checks,
        "all_same_seed_cross_family_targets_identical": all(
            check["all_case_ids_and_true_wss_hashes_identical"] for check in checks
        ),
    }


def unexpected_group_dirs(repo: Path) -> list[str]:
    expected = {
        str(repo / "output" / output_family(family) / split / run_id(family, seed))
        for family in FAMILIES
        for split in SPLITS
        for seed in SEEDS
    }
    discovered: set[str] = set()
    for family in FAMILIES:
        for split in SPLITS:
            root = repo / "output" / output_family(family) / split
            if not root.is_dir():
                continue
            for child in root.iterdir():
                if child.is_dir() and (child / "predictions").is_dir():
                    discovered.add(str(child))
    return sorted(discovered - expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    wrapper = args.wrapper.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    static = static_provenance(repo, wrapper)

    groups: list[dict[str, Any]] = []
    group_failures: list[dict[str, Any]] = []
    for family in FAMILIES:
        for split in SPLITS:
            for seed in SEEDS:
                print(f"START group={family}/{split}/seed{seed}", flush=True)
                try:
                    groups.append(process_group(repo, family, split, seed))
                except Exception as error:
                    group_failures.append(
                        {
                            "family": family,
                            "split": split,
                            "seed": seed,
                            "expected_path": str(
                                repo
                                / "output"
                                / output_family(family)
                                / split
                                / run_id(family, seed)
                            ),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    run_provenance: list[dict[str, Any]] = []
    run_provenance_failures: list[dict[str, Any]] = []
    group_index = {
        (group["family"], group["split"], group["seed"]): group for group in groups
    }
    for family in FAMILIES:
        for seed in SEEDS:
            reference_group = group_index.get(
                (family, "s1_id_reference", seed)
            ) or group_index.get((family, "s1_extreme_test", seed))
            if reference_group is None:
                run_provenance_failures.append(
                    {
                        "family": family,
                        "seed": seed,
                        "error": "no processed group available for logged config",
                    }
                )
                continue
            selected_config = reference_group["selected_metrics_segment"]["config"]
            try:
                record = find_checkpoint_records(repo, family, seed, selected_config)
                record["launcher"] = str(repo / launcher_name(family, seed))
                run_provenance.append(record)
            except Exception as error:
                run_provenance_failures.append(
                    {
                        "family": family,
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    all_provenance_records = [
        *static["files"],
        *[
            record[key]
            for record in run_provenance
            for key in (
                "resolved_config",
                "model_checkpoint",
                "training_checkpoint",
                "normalization_stats",
            )
        ],
        *[record["successful_final_job"]["file"] for record in run_provenance],
        *[
            failure["file"]
            for record in run_provenance
            for failure in record["historical_failed_jobs"]
        ],
    ]
    raw_records = [
        file_record
        for group in groups
        for case in group["cases"]
        for file_record in case["input_files"].values()
    ]
    metrics_and_logs = [
        record
        for group in groups
        for record in (group["metrics_file"], group["evaluation_log"])
        if record is not None
    ]
    expected_group_count = len(FAMILIES) * len(SPLITS) * len(SEEDS)
    complete_group_set = len(groups) == expected_group_count and not group_failures
    differential = build_differential(groups) if complete_group_set else None
    target_checks = target_identity_checks(groups) if complete_group_set else None
    case_failures = sum(len(group["case_processing_failures"]) for group in groups)
    invalid_json_lines = sum(
        len(group["metrics_jsonl"]["invalid_lines"]) for group in groups
    )
    historical_evaluation_tracebacks = sum(
        (
            group["evaluation_log_history"]["marker_counts"]["traceback"]
            if group["evaluation_log_history"] is not None
            else 0
        )
        for group in groups
    )
    historical_failed_jobs = sum(
        len(record["historical_failed_jobs"]) for record in run_provenance
    )
    missing_cases = sum(group["missing_case_count"] for group in groups)
    unexpected_cases = sum(group["unexpected_case_count"] for group in groups)
    result = {
        "schema_version": 1,
        "audit": "historical_s1_wss_metric_reconstruction",
        "started_utc": started,
        "finished_utc": utc_now(),
        "host": socket.gethostname(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "command": {
            "argv": [sys.executable, *sys.argv],
            "shell_escaped": shlex.join([sys.executable, *sys.argv]),
            "python_version": sys.version,
            "numpy_version": np.__version__,
        },
        "read_only_contract": {
            "historical_roots_read": [
                str(repo / "output" / "drivaer_ood_deep"),
                str(repo / "output" / "drivaer_ood_deep_geot"),
                str(
                    repo
                    / "examples"
                    / "cfd"
                    / "external_aerodynamics"
                    / "unified_external_aero_recipe"
                ),
            ],
            "historical_artifacts_modified": False,
            "only_write_root": str(output_dir),
        },
        "definitions": {
            "epsilon": EPS,
            "arithmetic": "float64 NumPy over saved physical float32 WSS arrays",
            "historical_wss_l2": (
                "mean_i(||pred_i - true_i||_2 / (||true_i||_2 + eps)); "
                "the producing evaluator applied its last-axis reduction to "
                "an (N,3) tensor"
            ),
            "current_whole_field_frobenius_wss_l2": (
                "||pred - true||_F / (||true||_F + eps), per case"
            ),
            "vector_magnitude_relative_l2": (
                "|| ||pred_i||_2 - ||true_i||_2 ||_2 / "
                "(|| ||true_i||_2 ||_2 + eps), per case"
            ),
            "per_component_relative_l2": (
                "||pred[:,j] - true[:,j]||_2 / (||true[:,j]||_2 + eps), per case"
            ),
            "group_summary": (
                "arithmetic mean over case metrics, matching infer.py summary "
                "aggregation; a separately labeled pooled-over-points value "
                "is also retained"
            ),
            "degradation_ratio": "S1 extreme-test case mean / S1 ID-reference case mean",
            "weighting": "unweighted; no cell-area or target-measure weighting",
        },
        "historical_evaluator_semantics": {
            "metrics_source_path": str(
                repo
                / "examples"
                / "cfd"
                / "external_aerodynamics"
                / "unified_external_aero_recipe"
                / "src"
                / "metrics.py"
            ),
            "function": "_relative_l2",
            "reduction": "spatial_axis = -1, followed by mean",
            "input_shape_for_bare_wss": "(N, 3)",
            "consequence": "bare wss_l2 was a pointwise-mean vector-relative error",
            "comment_code_mismatch": (
                "the historical MetricCalculator comment called the aggregate "
                "flattened/Frobenius, but passed p and t without flattening"
            ),
        },
        "provenance": {
            **static,
            "runs": run_provenance,
            "run_failures": run_provenance_failures,
            "scoped_source_config_checkpoint_sha256": identity_digest(
                all_provenance_records
            ),
        },
        "output_identity": {
            "raw_wss_and_metadata_file_count": len(raw_records),
            "raw_wss_and_metadata_sha256": identity_digest(raw_records),
            "metrics_and_evaluation_log_file_count": len(metrics_and_logs),
            "metrics_and_evaluation_logs_sha256": identity_digest(metrics_and_logs),
        },
        "counts": {
            "expected_groups": expected_group_count,
            "processed_groups": len(groups),
            "group_failures": len(group_failures),
            "expected_cases_from_selected_summaries": sum(
                group["expected_case_count_from_selected_summary"] for group in groups
            ),
            "raw_case_directories": sum(
                group["raw_case_directory_count"] for group in groups
            ),
            "successful_cases": sum(
                group["summary"]["successful_case_count"] for group in groups
            ),
            "case_processing_failures": case_failures,
            "missing_cases": missing_cases,
            "unexpected_cases": unexpected_cases,
            "invalid_metrics_jsonl_lines": invalid_json_lines,
            "historical_append_log_tracebacks": historical_evaluation_tracebacks,
            "historical_failed_slurm_jobs_preserved": historical_failed_jobs,
            "missing_static_provenance_files": len(static["missing_files"]),
            "run_provenance_failures": len(run_provenance_failures),
        },
        "unexpected_group_directories": unexpected_group_dirs(repo),
        "group_failures": group_failures,
        "target_identity_checks": target_checks,
        "degradation_differential": differential,
        "groups": groups,
    }
    result["complete"] = (
        complete_group_set
        and case_failures == 0
        and missing_cases == 0
        and unexpected_cases == 0
        and invalid_json_lines == 0
        and not static["missing_files"]
        and not run_provenance_failures
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    output_name = f"s1_wss_metric_reconstruction_2026-07-27_{digest[:16]}.json"
    output_path = output_dir / output_name
    output_path.write_bytes(payload)
    sidecar_path = output_dir / f"{output_name}.sha256"
    sidecar_path.write_text(f"{digest}  {output_name}\n", encoding="utf-8")
    print(f"RESULT={output_path}", flush=True)
    print(f"RESULT_SHA256={digest}", flush=True)
    print(f"RESULT_SIDECAR={sidecar_path}", flush=True)
    print(f"COMPLETE={str(result['complete']).lower()}", flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
