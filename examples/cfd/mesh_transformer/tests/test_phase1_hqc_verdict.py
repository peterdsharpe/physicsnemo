# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the pre-registered Phase-1 H-QC reduction."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import phase1_hqc_verdict as hqc
import pytest

FAKE_PRODUCER_SHA256 = "f" * 64
FAKE_CONFIG_SHA256 = hqc.FROZEN_CONTRACT["resolved_config_sha256"]
ORIGINAL_VALIDATE_RESCORING_NPZ = hqc._validate_rescoring_npz
ORIGINAL_EXPECTED_PRODUCER_SHA256 = hqc.EXPECTED_PRODUCER_SHA256


def _digest(index: int) -> str:
    return f"{index:064x}"


def _full_metrics(pressure: float) -> dict[str, float]:
    return {
        "pressure_relative_l2": pressure,
        "signed_centered_correlation": 0.98,
        "positive_gain_pattern_error": 0.2,
        "amplitude_ratio": 1.0,
        "wss_frobenius_relative_l2": 0.3,
        "wss_normal_energy": 0.1,
        "scaled_subset_pressure_force_relative_error": 0.4,
    }


def _normal_diagnostics(
    pipeline_normals: np.ndarray,
    native_normals: np.ndarray,
    *,
    geometry_error: float = 0.0,
) -> dict[str, float]:
    diagnostics = hqc._recompute_pipeline_normal_diagnostics(
        pipeline_normals,
        native_normals,
        "test fixture",
    )
    return {
        "max_unit_norm_abs_error": diagnostics["max_unit_norm_abs_error"],
        "max_geometry_reconstruction_abs_error": geometry_error,
        "min_native_dot": diagnostics["min_native_dot"],
    }


def _case(
    ordinal: int,
    *,
    uniform_coupled_ratio: float = 4.0,
    uniform_fixed_ratio: float = 1.1,
    area_coupled_ratio: float = 4.0,
    area_fixed_ratio: float = 1.1,
) -> dict:
    case_id = hqc.EXPECTED_CASES[ordinal]
    n_cells = hqc.EXPECTED_MASTER_CELLS[case_id]
    start = hqc.EXPECTED_HISTORICAL_STARTS[case_id]
    archived = hqc.ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case_id]
    q_hash = hqc._cyclic_indices_sha256(n_cells, start, hqc.FIXED_QUERY_K)
    selection_hashes = {
        k: hqc._cyclic_indices_sha256(n_cells, start, k) for k in hqc.RESOLUTIONS
    }
    resolutions = []
    for k in hqc.RESOLUTIONS:
        is_endpoint = k in hqc.ENDPOINTS
        uniform_coupled = archived * (uniform_coupled_ratio if is_endpoint else 1.0)
        uniform_fixed = archived * (uniform_fixed_ratio if is_endpoint else 1.0)
        area_baseline = archived * 1.2
        area_coupled = area_baseline * (area_coupled_ratio if is_endpoint else 1.0)
        area_fixed = area_baseline * (area_fixed_ratio if is_endpoint else 1.0)
        resolutions.append(
            {
                "k": k,
                "selection": {
                    "cell_ids_sha256_int64": selection_hashes[k],
                    "q_prefix_sha256_int64": q_hash,
                    "nested_prefix_passed": True,
                },
                "normal_diagnostics": {
                    arm: {
                        "max_unit_norm_abs_error": 0.0,
                        "max_geometry_reconstruction_abs_error": 0.0,
                        "min_native_dot": 1.0,
                    }
                    for arm in ("primary", "fixed_center")
                },
                "metrics": {
                    "uniform": {
                        "coupled": _full_metrics(uniform_coupled),
                        "fixed_q": _full_metrics(uniform_fixed),
                    },
                    "area_weighted": {
                        "coupled": {"pressure_relative_l2": area_coupled},
                        "fixed_q": {"pressure_relative_l2": area_fixed},
                    },
                },
                "finite_checks_passed": True,
            }
        )
    return {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": hqc.EXPECTED_READER_INDICES[case_id],
        "n_master_cells": n_cells,
        "historical_start": start,
        "source_identity": {
            "metadata_sha256": _digest(10_000 + 10 * ordinal),
            "points_sha256": _digest(10_001 + 10 * ordinal),
            "cells_sha256": _digest(10_002 + 10 * ordinal),
            "pressure_sha256": _digest(10_003 + 10 * ordinal),
            "wss_sha256": _digest(10_004 + 10 * ordinal),
        },
        "historical_10k": {
            "reader_seed_fork_chain_sha256": (
                hqc.EXPECTED_READER_SEED_FORK_CHAIN_SHA256
            ),
            "seed_fork_chain_replayed": True,
            "selection_sha256_int64": selection_hashes[hqc.BASELINE_K],
            "canonical_reconstructed_signature_sha256": _digest(21_000 + ordinal),
            "saved_artifact_coordinate_max_abs_error": 1.0e-8,
            "saved_artifact_pipeline_normals_max_abs_error": 1.0e-8,
            "saved_artifact_parity_passed": True,
            "exact_archived_row_available": True,
            "archived_uniform_pressure_relative_l2": archived,
        },
        "fixed_q": {
            "raw_cell_ids_sha256_int64": q_hash,
            "truth_pressure_sha256_float32": _digest(30_000 + 4 * ordinal),
            "normals_sha256_float32": _digest(30_001 + 4 * ordinal),
            "native_areas_sha256_float64": _digest(30_002 + 4 * ordinal),
            "truth_rms": 1.0,
            "native_area": 1.0,
            "mean_native_cell_area": 1.0 / hqc.FIXED_QUERY_K,
            "identity_checks_passed": True,
        },
        "s10k_reference": {
            "truth_rms": 1.5,
            "native_area": 4.0,
            "mean_native_cell_area": 4.0 / hqc.BASELINE_K,
        },
        "centers": {
            "fixed_s10k": [0.0, 0.0, 0.0],
            "primary_by_k": {str(k): [float(k), 0.0, 0.0] for k in hqc.RESOLUTIONS},
            "raw_frame_q_reconstruction_max_abs": 1.0e-8,
            "by_k": {
                str(k): {
                    arm: {
                        "pressure_prediction_relative_l2_difference": 1.0e-8,
                        "uniform_pressure_error_relative_change": 1.0e-8,
                        "area_pressure_error_relative_change": 1.0e-8,
                    }
                    for arm in ("coupled", "fixed_q")
                }
                for k in hqc.RESOLUTIONS
            },
            "passed": True,
        },
        "resolutions": resolutions,
    }


def _artifact(
    cases: list[dict],
    *,
    lane_ordinal: int = 0,
    lane_count: int = 1,
) -> dict:
    return {
        "schema_version": 2,
        "artifact_kind": "phase1_hqc_producer_lane",
        "status": "PASSED_HQC_PRODUCER_LANE",
        "generated_at_utc": "2026-07-27T12:00:00+00:00",
        "lane": {"ordinal": lane_ordinal, "count": lane_count},
        "frozen": deepcopy(hqc.FROZEN_CONTRACT),
        "cases": cases,
        "provenance": {
            "producer_path": "/frozen/drivaerml_trace_fixed_query_audit.py",
            "producer_sha256": FAKE_PRODUCER_SHA256,
            "config_sha256": FAKE_CONFIG_SHA256,
            "command": ["python3", "producer.py", f"--lane={lane_ordinal}"],
            "python": "3.13.5",
            "platform": "Linux-aarch64",
            "versions": {
                "numpy": "2.3.1",
                "torch": "2.7.1",
                "physicsnemo": "2.0.0",
            },
            "hardware": {
                "cuda_runtime": "12.9",
                "visible_cuda_device_count": 1,
                "cuda_device_name": "NVIDIA GB200",
                "cuda_device_capability": [10, 0],
                "cudnn_version": 90501,
            },
            "rescoring_npz_path": f"/frozen/lane_{lane_ordinal}.npz",
            "rescoring_npz_sha256": _digest(40_000 + lane_ordinal),
        },
    }


def _write_artifacts(tmp_path: Path, artifacts: list[dict]) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, artifact in enumerate(artifacts):
        path = tmp_path / f"lane_{index}.json"
        path.write_text(json.dumps(artifact))
        path.with_suffix(".npz").write_bytes(b"mock NPZ replaced in rescore tests")
        paths.append(path)
    return paths


@pytest.fixture(autouse=True)
def _freeze_test_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hqc, "EXPECTED_PRODUCER_SHA256", FAKE_PRODUCER_SHA256)
    monkeypatch.setattr(
        hqc,
        "_validate_rescoring_npz",
        lambda json_path, cases, provenance, context: {
            "path": str(json_path.with_suffix(".npz").resolve()),
            "sha256": provenance["rescoring_npz_sha256"],
            "key_count": len(cases) * len(hqc.RESOLUTIONS) * len(hqc.NPZ_FIELDS),
            "case_count": len(cases),
            "all_arrays_finite_with_frozen_shapes_and_dtypes": True,
            "all_json_metrics_recomputed": True,
            "all_fixed_center_diagnostics_recomputed": True,
            "all_pipeline_normal_npz_diagnostics_recomputed": True,
            "all_reported_pipeline_geometry_reconstruction_checks_passed": True,
            "fixed_center_q_pipeline_normals_exact_across_k": True,
            "baseline_pipeline_normals_exact_between_center_arms": True,
            "all_q_and_s10k_references_recomputed": True,
            "max_metric_absolute_difference": 0.0,
            "max_metric_relative_difference": 0.0,
            "max_center_absolute_difference": 0.0,
            "max_pipeline_normal_unit_abs_error": 0.0,
            "min_pipeline_native_dot": 1.0,
            "max_reported_pipeline_normal_geometry_abs_error": 0.0,
            "max_coordinate_reconstruction_residual": 0.0,
            "recovered_coordinate_scale_range": [5.0, 5.0],
        },
    )


def _supported_cases() -> list[dict]:
    return [_case(ordinal) for ordinal in range(len(hqc.EXPECTED_CASES))]


def _real_npz_evidence(tmp_path: Path) -> tuple[Path, dict, list[dict], dict]:
    """Construct one full-sized, internally consistent lane for NPZ rescore tests."""

    case = _case(0)
    arrays = {}
    fixed_center = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    case["centers"]["fixed_s10k"] = fixed_center.tolist()
    q_reference = None
    baseline_reference = None
    for k, row in zip(hqc.RESOLUTIONS, case["resolutions"], strict=True):
        index = np.arange(k, dtype=np.float64)
        raw_centroids = np.column_stack(
            (
                1.0 + 1.0e-4 * index,
                2.0 + 1.0e-3 * np.remainder(index, 97.0),
                3.0 + 2.0e-3 * np.remainder(index, 31.0),
            )
        ).astype("<f4")
        normals = np.zeros((k, 3), dtype="<f4")
        normals[:, 2] = 1.0
        pipeline_x = 1.0e-4 * np.sin(index)
        pipeline_normals = np.column_stack(
            (
                pipeline_x,
                np.zeros(k, dtype=np.float64),
                np.sqrt(1.0 - pipeline_x**2),
            )
        ).astype("<f4")
        areas = np.ones(k, dtype="<f8")
        truth_pressure = (1.0 + 1.0e-5 * index).astype("<f4")
        truth_wss = np.column_stack(
            (truth_pressure, 0.5 * truth_pressure, 0.25 * truth_pressure)
        ).astype("<f4")
        primary_pressure = (
            truth_pressure.astype(np.float64) * (1.05 + k / 1.0e6)
            + 1.0e-4 * np.sin(index)
        ).astype("<f4")
        primary_wss = (1.1 * truth_wss).astype("<f4")
        fixed_pressure = (primary_pressure.astype(np.float64) * 1.000001).astype("<f4")
        fixed_wss = (primary_wss.astype(np.float64) * 1.000001).astype("<f4")
        primary_center = np.array(
            [0.1 + k * 1.0e-8, 0.2, 0.3],
            dtype=np.float32,
        )
        primary_query = ((raw_centroids - primary_center) / 5.0).astype("<f4")
        fixed_query = ((raw_centroids - fixed_center) / 5.0).astype("<f4")
        raw_ids = (case["historical_start"] + np.arange(k, dtype=np.int64)) % case[
            "n_master_cells"
        ]
        compacted_cells = np.arange(3 * k, dtype="<i8").reshape(k, 3)
        values = {
            "raw_cell_ids_int64": raw_ids.astype("<i8", copy=False),
            "compacted_cells_int64": compacted_cells,
            "raw_centroids_float32": raw_centroids,
            "native_normals_float32": normals,
            "native_areas_float64": areas,
            "truth_pressure_float32": truth_pressure,
            "truth_wss_float32": truth_wss,
            "primary_query_points_float32": primary_query,
            "primary_pipeline_normals_float32": pipeline_normals,
            "primary_pressure_float32": primary_pressure,
            "primary_wss_float32": primary_wss,
            "fixed_center_query_points_float32": fixed_query,
            "fixed_center_pipeline_normals_float32": pipeline_normals.copy(),
            "fixed_center_pressure_float32": fixed_pressure,
            "fixed_center_wss_float32": fixed_wss,
        }
        for field, value in values.items():
            arrays[hqc._npz_key(0, k, field)] = np.ascontiguousarray(value)

        primary = hqc._recompute_metric_bundle(
            prediction_pressure=primary_pressure,
            truth_pressure=truth_pressure,
            prediction_wss=primary_wss,
            truth_wss=truth_wss,
            normals=normals,
            areas=areas,
            n_master_cells=case["n_master_cells"],
            context=f"k{k}.primary",
        )
        primary_q = hqc._recompute_metric_bundle(
            prediction_pressure=primary_pressure[: hqc.FIXED_QUERY_K],
            truth_pressure=truth_pressure[: hqc.FIXED_QUERY_K],
            prediction_wss=primary_wss[: hqc.FIXED_QUERY_K],
            truth_wss=truth_wss[: hqc.FIXED_QUERY_K],
            normals=normals[: hqc.FIXED_QUERY_K],
            areas=areas[: hqc.FIXED_QUERY_K],
            n_master_cells=case["n_master_cells"],
            context=f"k{k}.primary_q",
        )
        fixed = hqc._recompute_metric_bundle(
            prediction_pressure=fixed_pressure,
            truth_pressure=truth_pressure,
            prediction_wss=fixed_wss,
            truth_wss=truth_wss,
            normals=normals,
            areas=areas,
            n_master_cells=case["n_master_cells"],
            context=f"k{k}.fixed",
        )
        fixed_q = hqc._recompute_metric_bundle(
            prediction_pressure=fixed_pressure[: hqc.FIXED_QUERY_K],
            truth_pressure=truth_pressure[: hqc.FIXED_QUERY_K],
            prediction_wss=fixed_wss[: hqc.FIXED_QUERY_K],
            truth_wss=truth_wss[: hqc.FIXED_QUERY_K],
            normals=normals[: hqc.FIXED_QUERY_K],
            areas=areas[: hqc.FIXED_QUERY_K],
            n_master_cells=case["n_master_cells"],
            context=f"k{k}.fixed_q",
        )
        row["metrics"] = {
            "uniform": {
                "coupled": primary["uniform"],
                "fixed_q": primary_q["uniform"],
            },
            "area_weighted": {
                "coupled": primary["area_weighted"],
                "fixed_q": primary_q["area_weighted"],
            },
        }
        row["normal_diagnostics"] = {
            arm: _normal_diagnostics(pipeline_normals, normals)
            for arm in ("primary", "fixed_center")
        }
        case["centers"]["primary_by_k"][str(k)] = primary_center.tolist()
        case["centers"]["by_k"][str(k)] = {
            "coupled": hqc._recompute_center_diagnostic(
                primary_pressure,
                fixed_pressure,
                primary,
                fixed,
            ),
            "fixed_q": hqc._recompute_center_diagnostic(
                primary_pressure[: hqc.FIXED_QUERY_K],
                fixed_pressure[: hqc.FIXED_QUERY_K],
                primary_q,
                fixed_q,
            ),
        }
        if k == hqc.FIXED_QUERY_K:
            q_reference = (raw_ids, truth_pressure, normals, areas)
        if k == hqc.BASELINE_K:
            baseline_reference = (truth_pressure, areas)

    assert q_reference is not None and baseline_reference is not None
    q_ids, q_pressure, q_normals, q_areas = q_reference
    case["fixed_q"].update(
        {
            "raw_cell_ids_sha256_int64": hqc._sha256_array(q_ids, "<i8"),
            "truth_pressure_sha256_float32": hqc._sha256_array(q_pressure, "<f4"),
            "normals_sha256_float32": hqc._sha256_array(q_normals, "<f4"),
            "native_areas_sha256_float64": hqc._sha256_array(q_areas, "<f8"),
            "truth_rms": math.sqrt(float(np.mean(q_pressure.astype(np.float64) ** 2))),
            "native_area": float(q_areas.sum(dtype=np.float64)),
            "mean_native_cell_area": float(q_areas.mean(dtype=np.float64)),
        }
    )
    baseline_pressure, baseline_areas = baseline_reference
    case["s10k_reference"] = {
        "truth_rms": math.sqrt(
            float(np.mean(baseline_pressure.astype(np.float64) ** 2))
        ),
        "native_area": float(baseline_areas.sum(dtype=np.float64)),
        "mean_native_cell_area": float(baseline_areas.mean(dtype=np.float64)),
    }
    validated_cases = [hqc._validate_case(case, "real_npz.case")]
    artifact = _artifact([case])
    json_path = tmp_path / "lane_0.json"
    json_path.write_text(json.dumps(artifact))
    npz_path = json_path.with_suffix(".npz")
    np.savez(npz_path, **arrays)
    provenance = validated_cases and hqc._validate_provenance(
        artifact["provenance"],
        "real_npz.provenance",
    )
    provenance["rescoring_npz_sha256"] = hqc._sha256_file(npz_path)
    return json_path, provenance, validated_cases, arrays


def test_dual_weighting_support_uses_complete_disjoint_lanes(tmp_path: Path):
    cases = _supported_cases()
    artifacts = [
        _artifact(cases[1::2], lane_ordinal=1, lane_count=2),
        _artifact(cases[0::2], lane_ordinal=0, lane_count=2),
    ]

    result = hqc.build_verdict(_write_artifacts(tmp_path, artifacts))

    assert result["status"] == "SUPPORTED_HQC_DUAL_WEIGHTING"
    assert result["uniform_hqc_supported"] is True
    assert result["dual_weighting_supported"] is True
    assert result["licensed_wording"] == (
        "query co-sampling dominates the physically scored cliff"
    )
    assert [case["case_id"] for case in result["cases"]] == list(hqc.EXPECTED_CASES)
    uniform = result["metric_panels"]["uniform"]
    assert uniform["outcome"] == "SUPPORTED"
    assert uniform["endpoints"]["2500"]["favorable_paired_reduction_count"] == 36
    assert uniform["endpoints"]["40000"]["support"]["passed"] is True


def test_futility_and_mixed_are_distinct_eligible_outcomes(tmp_path: Path):
    futile = [
        _case(ordinal, uniform_fixed_ratio=3.0)
        for ordinal in range(len(hqc.EXPECTED_CASES))
    ]
    futile_result = hqc.build_verdict(
        _write_artifacts(tmp_path / "futile", [_artifact(futile)])
    )
    assert futile_result["status"] == "FUTILE_HQC_DOMINANT_MECHANISM"
    assert futile_result["metric_panels"]["uniform"]["outcome"] == "FUTILE"
    assert (
        futile_result["metric_panels"]["uniform"]["endpoints"]["40000"]["futility"][
            "fixed_40k_over_10k_median_at_least_2x"
        ]
        is True
    )

    mixed = [
        _case(ordinal, uniform_fixed_ratio=1.5)
        for ordinal in range(len(hqc.EXPECTED_CASES))
    ]
    mixed_result = hqc.build_verdict(
        _write_artifacts(tmp_path / "mixed", [_artifact(mixed)])
    )
    assert mixed_result["status"] == "MIXED_HQC"
    assert mixed_result["metric_panels"]["uniform"]["outcome"] == "MIXED"


def test_area_nearly_flat_is_not_mislabeled_as_dual_support(tmp_path: Path):
    cases = [
        _case(ordinal, area_coupled_ratio=1.1)
        for ordinal in range(len(hqc.EXPECTED_CASES))
    ]

    result = hqc.build_verdict(_write_artifacts(tmp_path, [_artifact(cases)]))

    assert result["status"] == "SUPPORTED_HQC_UNIFORM_AREA_NEARLY_FLAT"
    assert result["metric_panels"]["uniform"]["outcome"] == "SUPPORTED"
    assert result["metric_panels"]["area_weighted"]["outcome"] == "INELIGIBLE"
    assert result["metric_panels"]["area_weighted"][
        "area_coupled_nearly_flat_endpoints"
    ] == [2500, 40000]
    assert result["dual_weighting_supported"] is False
    assert result["licensed_wording"] is None


def test_one_area_endpoint_at_flat_boundary_stays_metric_specific(tmp_path: Path):
    cases = _supported_cases()
    for case in cases:
        baseline = 1.2 * hqc.ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case["case_id"]]
        endpoint = next(row for row in case["resolutions"] if row["k"] == 2_500)
        endpoint["metrics"]["area_weighted"]["coupled"]["pressure_relative_l2"] = (
            1.25 * baseline
        )

    result = hqc.build_verdict(_write_artifacts(tmp_path, [_artifact(cases)]))

    area = result["metric_panels"]["area_weighted"]
    assert area["area_coupled_nearly_flat_endpoints"] == [2_500]
    assert area["outcome"] == "INELIGIBLE"
    assert result["status"] == "SUPPORTED_HQC_UNIFORM_ONLY_METRIC_SPECIFIC"
    assert result["dual_weighting_supported"] is False


def test_eligibility_uses_mean_cell_area_not_total_area(tmp_path: Path):
    result = hqc.build_verdict(
        _write_artifacts(tmp_path, [_artifact(_supported_cases())])
    )

    ratio = result["common_stage0_eligibility"]["q_support_ratios"]["run_118"]
    assert ratio["q_over_s10k_total_native_area_diagnostic"] == pytest.approx(0.25)
    assert ratio["q_over_s10k_mean_native_cell_area"] == pytest.approx(1.0)
    assert ratio["mean_native_cell_area_within_factor_two"] is True
    assert result["common_stage0_eligibility"]["passed"] is True


def test_exact_support_and_futility_boundaries(tmp_path: Path):
    cases = _supported_cases()
    support_ratio = 2.0**0.25
    for ordinal, case in enumerate(cases):
        fixed_ratio = support_ratio if ordinal < 27 else 2.0
        for row in case["resolutions"]:
            if row["k"] in hqc.ENDPOINTS:
                baseline = hqc.ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case["case_id"]]
                row["metrics"]["uniform"]["coupled"]["pressure_relative_l2"] = (
                    2.0 * baseline
                )
                row["metrics"]["uniform"]["fixed_q"]["pressure_relative_l2"] = (
                    fixed_ratio * baseline
                )
    supported = hqc.build_verdict(
        _write_artifacts(tmp_path / "support", [_artifact(cases)])
    )
    uniform = supported["metric_panels"]["uniform"]
    assert uniform["outcome"] == "SUPPORTED"
    assert uniform["endpoints"]["2500"]["favorable_paired_reduction_count"] == 27
    assert uniform["endpoints"]["2500"]["support"]["passed"] is True

    boundary_futile = _supported_cases()
    for case in boundary_futile:
        baseline = hqc.ARCHIVED_UNIFORM_PRESSURE_BY_CASE[case["case_id"]]
        for row in case["resolutions"]:
            if row["k"] in hqc.ENDPOINTS:
                row["metrics"]["uniform"]["coupled"]["pressure_relative_l2"] = (
                    2.0 * baseline
                )
                row["metrics"]["uniform"]["fixed_q"]["pressure_relative_l2"] = (
                    math.sqrt(2.0) * baseline
                )
    futile = hqc.build_verdict(
        _write_artifacts(tmp_path / "futile", [_artifact(boundary_futile)])
    )
    assert futile["metric_panels"]["uniform"]["outcome"] == "FUTILE"


def test_failed_replay_is_ineligible_not_a_scientific_failure(tmp_path: Path):
    cases = _supported_cases()
    baseline = next(
        row for row in cases[0]["resolutions"] if row["k"] == hqc.BASELINE_K
    )
    baseline["metrics"]["uniform"]["coupled"]["pressure_relative_l2"] += 0.01

    result = hqc.build_verdict(_write_artifacts(tmp_path, [_artifact(cases)]))

    assert result["status"] == "INELIGIBLE_HQC_PANEL"
    assert result["common_stage0_eligibility"]["passed"] is False
    assert (
        "archived_10k_case_replay_all_cases"
        in result["common_stage0_eligibility"]["failed_checks"]
    )
    assert result["metric_panels"]["uniform"]["outcome"] == "INELIGIBLE"


def test_center_disagreement_blocks_without_selecting_an_arm(tmp_path: Path):
    cases = _supported_cases()
    centers = cases[5]["centers"]
    centers["by_k"]["40000"]["fixed_q"][
        "pressure_prediction_relative_l2_difference"
    ] = 2.0e-3
    centers["passed"] = False

    result = hqc.build_verdict(_write_artifacts(tmp_path, [_artifact(cases)]))

    assert result["status"] == "BLOCKED_HQC_CENTERING"
    assert result["centering"]["passed"] is False
    assert result["centering"]["failed_case_ids"] == [cases[5]["case_id"]]
    assert result["metric_panels"]["uniform"]["outcome"] == "SUPPORTED"
    assert result["dual_weighting_supported"] is False


def test_verdict_rejects_frozen_contract_and_selection_drift(tmp_path: Path):
    artifact = _artifact(_supported_cases())
    artifact["frozen"]["epoch"] += 1
    with pytest.raises(hqc.VerdictInputError, match="frozen differs"):
        hqc.build_verdict(_write_artifacts(tmp_path / "contract", [artifact]))

    artifact = _artifact(_supported_cases())
    artifact["cases"][3]["resolutions"][2]["selection"]["cell_ids_sha256_int64"] = (
        _digest(999)
    )
    with pytest.raises(hqc.VerdictInputError, match="does not match start/N/k"):
        hqc.build_verdict(_write_artifacts(tmp_path / "selection", [artifact]))


def test_verdict_rejects_changed_start_and_historical_row(tmp_path: Path):
    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["historical_start"] += 1
    with pytest.raises(hqc.VerdictInputError, match="historical_start is not frozen"):
        hqc.build_verdict(_write_artifacts(tmp_path / "start", [artifact]))

    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["historical_10k"]["archived_uniform_pressure_relative_l2"] += (
        1.0e-5
    )
    with pytest.raises(hqc.VerdictInputError, match="frozen historical row"):
        hqc.build_verdict(_write_artifacts(tmp_path / "history", [artifact]))


def test_verdict_rejects_missing_lane_and_duplicate_case(tmp_path: Path):
    cases = _supported_cases()
    incomplete = _artifact(cases[0::2], lane_ordinal=0, lane_count=2)
    with pytest.raises(hqc.VerdictInputError, match="lane ordinals must be complete"):
        hqc.build_verdict(_write_artifacts(tmp_path / "lane", [incomplete]))

    duplicate = _artifact(cases + [deepcopy(cases[0])])
    with pytest.raises(hqc.VerdictInputError, match="duplicate H-QC cohort ordinal"):
        hqc.build_verdict(_write_artifacts(tmp_path / "duplicate", [duplicate]))


def test_verdict_rejects_inconsistent_center_pass_flag(tmp_path: Path):
    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["centers"]["raw_frame_q_reconstruction_max_abs"] = 2.0e-6

    with pytest.raises(hqc.VerdictInputError, match="passed disagrees"):
        hqc.build_verdict(_write_artifacts(tmp_path, [artifact]))


def test_final_producer_hash_must_be_frozen(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hqc, "EXPECTED_PRODUCER_SHA256", "0" * 64)

    with pytest.raises(hqc.VerdictInputError, match="not frozen"):
        hqc.build_verdict(_write_artifacts(tmp_path, [_artifact(_supported_cases())]))


def test_output_cannot_alias_input_and_is_publish_once(tmp_path: Path):
    path = _write_artifacts(tmp_path, [_artifact(_supported_cases())])[0]
    with pytest.raises(hqc.VerdictInputError, match="must not alias"):
        hqc._validate_output_path(path, [path])

    output = tmp_path / "verdict.json"
    output.write_text("first evidence\n")
    with pytest.raises(FileExistsError):
        hqc._write_json_once(output, {"replacement": True})
    assert output.read_text() == "first evidence\n"


def test_signed_correlation_does_not_treat_anticorrelation_as_a_good_pattern():
    truth = np.array([-1.0, 0.0, 1.0])
    wss = np.column_stack((truth, truth + 2.0, truth + 3.0))
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))
    metrics = hqc._recompute_metric_bundle(
        prediction_pressure=-truth,
        truth_pressure=truth,
        prediction_wss=wss,
        truth_wss=wss,
        normals=normals,
        areas=np.ones(3),
        n_master_cells=10,
        context="anticorrelation",
    )

    assert metrics["uniform"]["signed_centered_correlation"] == pytest.approx(-1.0)
    assert metrics["uniform"]["positive_gain_pattern_error"] == pytest.approx(1.0)


def test_corrected_producer_sha_must_be_frozen_after_source_stabilizes():
    assert (
        ORIGINAL_EXPECTED_PRODUCER_SHA256
        == "8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f"
    )


def test_npz_rescore_recomputes_every_metric_and_center_arm(tmp_path: Path):
    json_path, provenance, cases, _ = _real_npz_evidence(tmp_path)

    result = ORIGINAL_VALIDATE_RESCORING_NPZ(
        json_path,
        cases,
        provenance,
        "real_npz",
    )

    assert result["key_count"] == len(hqc.RESOLUTIONS) * len(hqc.NPZ_FIELDS)
    assert result["all_json_metrics_recomputed"] is True
    assert result["all_fixed_center_diagnostics_recomputed"] is True
    assert result["all_pipeline_normal_npz_diagnostics_recomputed"] is True
    assert result["fixed_center_q_pipeline_normals_exact_across_k"] is True
    assert result["baseline_pipeline_normals_exact_between_center_arms"] is True
    assert result["min_pipeline_native_dot"] > 0.0
    assert result["max_metric_absolute_difference"] == pytest.approx(0.0)
    assert result["max_center_absolute_difference"] < 1.0e-6


def test_metric_recomputation_error_does_not_disclose_deciding_values():
    with pytest.raises(hqc.VerdictInputError) as error:
        hqc._compare_recomputed_number(
            1234.56789,
            -9876.54321,
            "secret_endpoint_metric",
        )

    assert str(error.value) == "secret_endpoint_metric differs from NPZ recomputation"


def test_npz_rescore_rejects_hash_key_and_metric_tampering(tmp_path: Path):
    json_path, provenance, cases, arrays = _real_npz_evidence(tmp_path)
    npz_path = json_path.with_suffix(".npz")

    tampered = dict(arrays)
    prediction_key = hqc._npz_key(0, 40_000, "primary_pressure_float32")
    tampered[prediction_key] = np.array(tampered[prediction_key], copy=True)
    tampered[prediction_key][0] += 0.5
    np.savez(npz_path, **tampered)
    with pytest.raises(hqc.VerdictInputError, match="SHA-256 does not match"):
        ORIGINAL_VALIDATE_RESCORING_NPZ(
            json_path,
            cases,
            provenance,
            "tampered_hash",
        )

    provenance["rescoring_npz_sha256"] = hqc._sha256_file(npz_path)
    with pytest.raises(hqc.VerdictInputError, match="differs from NPZ recomputation"):
        ORIGINAL_VALIDATE_RESCORING_NPZ(
            json_path,
            cases,
            provenance,
            "tampered_metric",
        )

    missing_key = dict(arrays)
    missing_key.pop(hqc._npz_key(0, 5_000, "native_areas_float64"))
    np.savez(npz_path, **missing_key)
    provenance["rescoring_npz_sha256"] = hqc._sha256_file(npz_path)
    with pytest.raises(hqc.VerdictInputError, match="NPZ has wrong keys"):
        ORIGINAL_VALIDATE_RESCORING_NPZ(
            json_path,
            cases,
            provenance,
            "tampered_keys",
        )


def test_npz_rescore_rejects_pipeline_normal_tampering(tmp_path: Path):
    json_path, provenance, cases, arrays = _real_npz_evidence(tmp_path)
    npz_path = json_path.with_suffix(".npz")

    def validate(tampered: dict[str, np.ndarray], context: str) -> None:
        np.savez(npz_path, **tampered)
        provenance["rescoring_npz_sha256"] = hqc._sha256_file(npz_path)
        ORIGINAL_VALIDATE_RESCORING_NPZ(
            json_path,
            cases,
            provenance,
            context,
        )

    nonunit = dict(arrays)
    key = hqc._npz_key(0, 5_000, "primary_pipeline_normals_float32")
    nonunit[key] = np.array(nonunit[key], copy=True)
    nonunit[key][0] *= 2.0
    with pytest.raises(hqc.VerdictInputError, match="not unit length"):
        validate(nonunit, "nonunit_pipeline_normal")

    flipped = dict(arrays)
    flipped[key] = np.array(flipped[key], copy=True)
    flipped[key][0] *= -1.0
    with pytest.raises(hqc.VerdictInputError, match="preserve native orientation"):
        validate(flipped, "flipped_pipeline_normal")

    fixed_q_drift = dict(arrays)
    fixed_key = hqc._npz_key(
        0,
        5_000,
        "fixed_center_pipeline_normals_float32",
    )
    fixed_q_drift[fixed_key] = np.array(fixed_q_drift[fixed_key], copy=True)
    fixed_q_drift[fixed_key][[0, 1]] = fixed_q_drift[fixed_key][[1, 0]]
    with pytest.raises(
        hqc.VerdictInputError,
        match="fixed-Q changed for fixed_center_pipeline_normals_float32",
    ):
        validate(fixed_q_drift, "fixed_q_pipeline_normal_drift")

    baseline_drift = dict(arrays)
    baseline_key = hqc._npz_key(
        0,
        hqc.BASELINE_K,
        "primary_pipeline_normals_float32",
    )
    baseline_drift[baseline_key] = np.array(baseline_drift[baseline_key], copy=True)
    baseline_drift[baseline_key][[0, 1]] = baseline_drift[baseline_key][[1, 0]]
    with pytest.raises(
        hqc.VerdictInputError,
        match="primary/fixed-center pipeline normals differ",
    ):
        validate(baseline_drift, "baseline_pipeline_normal_drift")


def test_json_normal_diagnostics_and_archive_parity_are_fail_closed(
    tmp_path: Path,
):
    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["resolutions"][0]["normal_diagnostics"]["primary"][
        "min_native_dot"
    ] = 0.0
    with pytest.raises(hqc.VerdictInputError, match="preserve native orientation"):
        hqc.build_verdict(_write_artifacts(tmp_path / "orientation", [artifact]))

    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["resolutions"][0]["normal_diagnostics"]["fixed_center"][
        "max_geometry_reconstruction_abs_error"
    ] = 1.0e-6
    with pytest.raises(hqc.VerdictInputError, match="geometry_reconstruction"):
        hqc.build_verdict(_write_artifacts(tmp_path / "geometry", [artifact]))

    artifact = _artifact(_supported_cases())
    artifact["cases"][0]["historical_10k"][
        "saved_artifact_pipeline_normals_max_abs_error"
    ] = 3.0e-6
    with pytest.raises(hqc.VerdictInputError, match="parity_passed disagrees"):
        hqc.build_verdict(_write_artifacts(tmp_path / "archive", [artifact]))
