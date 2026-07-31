# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the pre-registered Phase-1 H4 reduction."""

from __future__ import annotations

import hashlib
import json
import statistics
from copy import deepcopy
from pathlib import Path

import pytest
from phase1_h4_verdict import (
    ALGEBRA_ERROR_KEYS,
    EXPECTED_AUDIT_SCRIPT_SHA256,
    EXPECTED_CASES,
    EXPECTED_PRODUCTION_VERSIONS,
    FROZEN_SEED,
    FROZEN_SOURCE_IDENTITIES,
    FROZEN_STARTS,
    VerdictInputError,
    _cyclic_indices_sha256,
    _validate_output_path,
    _write_json_once,
    build_verdict,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _metrics(
    pressure: float,
    raw_wss: float,
    fill_q95: float,
) -> dict:
    return {
        "p0_projection_floor": {
            "CpMeanTrim": {"relative_l2_floor": pressure},
            "wallShearStressMeanTrim": {
                "raw_p0_relative_l2_floor": raw_wss,
            },
        },
        "fill_distance": {
            "normalized_by_master_bbox_diagonal": {
                "area_weighted_quantiles": {"q95": fill_q95},
            },
        },
    }


def _representation(
    *,
    metrics: dict,
    definition: dict,
    digest_offset: int,
) -> dict:
    return {
        "k": 40_000,
        "definition": definition,
        "support_points_sha256_float32": _digest(digest_offset),
        "support_normals_sha256_float32": _digest(digest_offset + 1),
        "master_assignment_sha256_int64": _digest(digest_offset + 2),
        "empty_representation_cell_count": 0,
        **metrics,
    }


def _algebra(ordinal: int) -> dict:
    errors = {key: 1.0e-15 for key in ALGEBRA_ERROR_KEYS}
    return {
        **errors,
        "max_load_bearing_error": 1.0e-15,
        "tolerance": 1.0e-12,
        "passed": True,
        "operator_sha256_streaming_assignment_a_b_area_float64": _digest(100 + ordinal),
    }


def _artifact(
    case_id: str,
    cyclic_pressure: tuple[float, float, float, float],
    *,
    cover_pressure: float,
) -> dict:
    starts = FROZEN_STARTS[case_id]
    cover_construction = {
        "candidate_multiplier": 16,
        "lloyd_iterations": 2,
        "lloyd_history": [
            {
                "empty_before_repair": 0,
                "repaired_support_indices": [],
                "repair_master_cells": [],
                "shift": {
                    "min": 0.0,
                    "mean": 0.1,
                    "q50": 0.1,
                    "q90": 0.2,
                    "q95": 0.2,
                    "q99": 0.2,
                    "max": 0.3,
                },
                "iteration": 1,
                "represented_measure": {},
                "neighbor_backend": "scipy.spatial.cKDTree",
            },
            {
                "empty_before_repair": 1,
                "repaired_support_indices": [7],
                "repair_master_cells": [11],
                "shift": {
                    "min": 0.0,
                    "mean": 0.1,
                    "q50": 0.1,
                    "q90": 0.2,
                    "q95": 0.2,
                    "q99": 0.2,
                    "max": 0.3,
                },
                "iteration": 2,
                "represented_measure": {},
                "neighbor_backend": "scipy.spatial.cKDTree",
            },
        ],
        "restriction_empty_cell_repairs": [
            {
                "empty_before_repair": 1,
                "repaired_support_indices": [5],
                "repair_master_cells": [13],
                "shift": {
                    "min": 0.0,
                    "mean": 0.001,
                    "q50": 0.0,
                    "q90": 0.0,
                    "q95": 0.0,
                    "q99": 0.0,
                    "max": 1.0,
                },
                "restriction_attempt": 1,
                "repair_scope": "empty_supports_only",
            }
        ],
    }
    cover_metrics = _metrics(cover_pressure, 0.5, 0.4)
    cover = _representation(
        metrics=cover_metrics,
        definition=cover_construction,
        digest_offset=10,
    )
    replicates = []
    for ordinal, (start, pressure) in enumerate(zip(starts, cyclic_pressure)):
        cyclic_metrics = _metrics(pressure, 1.0 + ordinal, 0.8 + ordinal)
        cyclic_definition = {
            "start": start,
            "frozen_seed": FROZEN_SEED,
            "replicate_ordinal": ordinal,
            "indices_sha256_int64": _cyclic_indices_sha256(
                FROZEN_SOURCE_IDENTITIES[case_id]["n_cells"],
                start,
                40_000,
            ),
            "first_indices": [start + offset for offset in range(10)],
            "wraps": False,
        }
        cyclic = _representation(
            metrics=cyclic_metrics,
            definition=cyclic_definition,
            digest_offset=30 + 3 * ordinal,
        )
        replicates.append(
            {
                "ordinal": ordinal,
                "frozen_cyclic_start": start,
                "frozen_seed": FROZEN_SEED,
                "representations": {
                    "cyclic_sparse": cyclic,
                    "normal_aware_centroidal_cover": deepcopy(cover),
                },
                "common_master_algebra": _algebra(ordinal),
                "execution": {
                    "cyclic_restriction_neighbor_backend": ("scipy.spatial.cKDTree"),
                    "final_neighbor_backends": [
                        "scipy.spatial.cKDTree",
                        "scipy.spatial.cKDTree",
                    ],
                },
                "comparison": {
                    "ratio_definition": (
                        "normal_aware_centroidal_cover / cyclic_sparse"
                    ),
                    "CpMeanTrim_relative_l2_floor_ratio": (cover_pressure / pressure),
                    "wallShearStressMeanTrim_relative_l2_floor_ratio": (
                        0.5 / (1.0 + ordinal)
                    ),
                    "normalized_area_weighted_fill_q95_ratio": (0.4 / (0.8 + ordinal)),
                },
            }
        )

    source_identity = FROZEN_SOURCE_IDENTITIES[case_id]
    source_arrays = {
        key: {"sha256": digest}
        for key, digest in source_identity["array_sha256"].items()
    }
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-07-27T12:00:00+00:00",
        "status": "PASSED_PRODUCTION_COMMON_MASTER_AUDIT",
        "design": {
            "requested_k": 40_000,
            "synthetic_k": None,
            "hash_inputs": True,
            "candidate_multiplier": 16,
            "lloyd_iterations": 2,
            "normal_aware_assignment": {
                "squared_cost": "||x_i-s_j||^2 + lambda^2 ||n_i-m_j||^2",
                "lambda": "sqrt(per_case_master_area / k)",
            },
            "geometry_chunk_cells": 250_000,
            "point_chunk": 1_000_000,
            "repair_pool_size": 2_048,
            "workers": 16,
            "explicit_cyclic_replicates": {
                case_id: [{"start": start, "seed": FROZEN_SEED} for start in starts]
            },
        },
        "cases": [
            {
                "case_id": case_id,
                "effective_k": 40_000,
                "source": {
                    "kind": "curated_drivaerml_vehicle_tensordict_memmaps",
                    "n_cells": source_identity["n_cells"],
                    "arrays": source_arrays,
                    "metadata": {
                        key: {"sha256": digest}
                        for key, digest in source_identity["metadata_sha256"].items()
                    },
                },
                "full_master": {
                    "n_cells": source_identity["n_cells"],
                    "degenerate_triangle_count": 0,
                },
                "cover_construction": cover_construction,
                "cyclic_replicates": replicates,
                "execution": {
                    "cover_restriction_neighbor_backend": ("scipy.spatial.cKDTree")
                },
            }
        ],
        "provenance": {
            "script": {
                "sha256": EXPECTED_AUDIT_SCRIPT_SHA256,
            },
            "python": EXPECTED_PRODUCTION_VERSIONS["python"],
            "platform": "Linux-aarch64-with-glibc2.39",
            "versions": {
                "numpy": EXPECTED_PRODUCTION_VERSIONS["numpy"],
                "scipy": EXPECTED_PRODUCTION_VERSIONS["scipy"],
            },
        },
    }


def _write_artifacts(
    tmp_path: Path,
    *,
    run_1_pressure: tuple[float, float, float, float] = (1.0, 2.0, 100.0, 200.0),
    run_1_cover: float = 1.5,
) -> list[Path]:
    inputs = {
        "run_1": _artifact(
            "run_1",
            run_1_pressure,
            cover_pressure=run_1_cover,
        ),
        "run_118": _artifact(
            "run_118",
            (1.0, 2.0, 3.0, 4.0),
            cover_pressure=0.7,
        ),
    }
    paths = []
    for case_id in reversed(EXPECTED_CASES):
        path = tmp_path / f"{case_id}.json"
        path.write_text(json.dumps(inputs[case_id]))
        paths.append(path)
    return paths


def _build_verdict(paths: list[Path]) -> dict:
    return build_verdict(paths)


def test_verdict_uses_cover_over_median_absolute_cyclic_metric(tmp_path: Path):
    result = _build_verdict(_write_artifacts(tmp_path))

    assert [case["case_id"] for case in result["cases"]] == list(EXPECTED_CASES)
    run_1 = result["cases"][0]
    aggregation = run_1["aggregation"]
    assert (
        aggregation["cyclic_sparse_median_absolute"]["pressure_relative_l2_floor"]
        == 51.0
    )
    assert aggregation["cover_over_median_cyclic"][
        "pressure_relative_l2_floor"
    ] == pytest.approx(1.5 / 51.0)
    # This is deliberately unlike the median of the four start-level ratios.
    start_ratios = [
        row["cover_over_cyclic_ratio_diagnostic_only"]["pressure_relative_l2_floor"]
        for row in run_1["start_level_rows"]
    ]
    assert statistics.median(start_ratios) != pytest.approx(1.5 / 51.0)
    assert len(run_1["start_level_rows"]) == 4
    assert result["h4_passed"] is True


def test_scientific_gate_failure_is_a_recorded_verdict(tmp_path: Path):
    paths = _write_artifacts(
        tmp_path,
        run_1_pressure=(1.0, 1.0, 1.0, 1.0),
        run_1_cover=0.81,
    )

    result = _build_verdict(paths)

    assert result["status"] == "FAILED_H4_COMMON_MASTER_GATE"
    assert result["h4_passed"] is False
    assert result["cases"][0]["gates"]["pressure_relative_l2_floor"] == {
        "ratio": 0.81,
        "threshold": 0.8,
        "comparison": "less_than_or_equal",
        "passed": False,
    }


def test_threshold_boundaries_keep_fill_strict(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    for path in paths:
        artifact = json.loads(path.read_text())
        for replicate in artifact["cases"][0]["cyclic_replicates"]:
            cyclic = replicate["representations"]["cyclic_sparse"]
            cover = replicate["representations"]["normal_aware_centroidal_cover"]
            cyclic["p0_projection_floor"]["CpMeanTrim"]["relative_l2_floor"] = 1.0
            cover["p0_projection_floor"]["CpMeanTrim"]["relative_l2_floor"] = 0.8
            cyclic["p0_projection_floor"]["wallShearStressMeanTrim"][
                "raw_p0_relative_l2_floor"
            ] = 1.0
            cover["p0_projection_floor"]["wallShearStressMeanTrim"][
                "raw_p0_relative_l2_floor"
            ] = 1.0
            cyclic["fill_distance"]["normalized_by_master_bbox_diagonal"][
                "area_weighted_quantiles"
            ]["q95"] = 1.0
            cover["fill_distance"]["normalized_by_master_bbox_diagonal"][
                "area_weighted_quantiles"
            ]["q95"] = 1.0
            comparison = replicate["comparison"]
            comparison["CpMeanTrim_relative_l2_floor_ratio"] = 0.8
            comparison["wallShearStressMeanTrim_relative_l2_floor_ratio"] = 1.0
            comparison["normalized_area_weighted_fill_q95_ratio"] = 1.0
        path.write_text(json.dumps(artifact))

    result = _build_verdict(paths)

    for case in result["cases"]:
        assert case["gates"]["pressure_relative_l2_floor"]["passed"] is True
        assert case["gates"]["raw_wss_relative_l2_floor"]["passed"] is True
        assert case["gates"]["normalized_area_weighted_fill_q95"]["passed"] is False
    assert result["h4_passed"] is False


def test_verdict_rejects_a_changed_frozen_start(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    artifact["cases"][0]["cyclic_replicates"][0]["frozen_cyclic_start"] += 1
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="wrong frozen start"):
        _build_verdict(paths)


def test_verdict_rejects_a_noninvariant_cover(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    replicate = artifact["cases"][0]["cyclic_replicates"][1]
    cover_floor = replicate["representations"]["normal_aware_centroidal_cover"][
        "p0_projection_floor"
    ]["CpMeanTrim"]
    cover_floor["relative_l2_floor"] *= 1.01
    replicate["comparison"]["CpMeanTrim_relative_l2_floor_ratio"] *= 1.01
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="cover changed across cyclic starts"):
        _build_verdict(paths)


def test_verdict_reconstructs_the_cyclic_index_hash(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    definition = artifact["cases"][0]["cyclic_replicates"][2]["representations"][
        "cyclic_sparse"
    ]["definition"]
    definition["indices_sha256_int64"] = _digest(999)
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="does not match start/k"):
        _build_verdict(paths)


def test_verdict_requires_the_frozen_source_identity(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    artifact["cases"][0]["source"]["arrays"]["CpMeanTrim"]["sha256"] = _digest(999)
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="source identity differs"):
        _build_verdict(paths)


def test_verdict_rejects_a_negative_algebra_error(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    algebra = artifact["cases"][0]["cyclic_replicates"][0]["common_master_algebra"]
    algebra["constant_max_abs_error"] = -1.0e-15
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="must all lie in"):
        _build_verdict(paths)


def test_verdict_requires_restriction_repair_scope(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    repair = artifact["cases"][0]["cover_construction"][
        "restriction_empty_cell_repairs"
    ][0]
    del repair["repair_scope"]
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="repair_scope"):
        _build_verdict(paths)


def test_verdict_rejects_out_of_range_repair_indices(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    repair = artifact["cases"][0]["cover_construction"][
        "restriction_empty_cell_repairs"
    ][0]
    repair["repaired_support_indices"] = [-999]
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match=r"indices in \[0, 40000\)"):
        _build_verdict(paths)


def test_verdict_requires_the_hard_frozen_audit_sha(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    artifact = json.loads(paths[1].read_text())
    artifact["provenance"]["script"]["sha256"] = _digest(999)
    paths[1].write_text(json.dumps(artifact))

    with pytest.raises(VerdictInputError, match="frozen production audit script"):
        _build_verdict(paths)


def test_frozen_audit_sha_matches_current_producer_source():
    producer = (
        Path(__file__).resolve().parents[1]
        / "studies"
        / "drivaerml_common_master_audit.py"
    )

    assert hashlib.sha256(producer.read_bytes()).hexdigest() == (
        EXPECTED_AUDIT_SCRIPT_SHA256
    )


def test_verdict_output_cannot_alias_an_input(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    with pytest.raises(VerdictInputError, match="must differ"):
        _validate_output_path(paths[0], paths)


def test_verdict_output_is_publish_once(tmp_path: Path):
    output = tmp_path / "verdict.json"
    output.write_text("first evidence\n")

    with pytest.raises(FileExistsError):
        _write_json_once(output, {"replacement": True})

    assert output.read_text() == "first evidence\n"
