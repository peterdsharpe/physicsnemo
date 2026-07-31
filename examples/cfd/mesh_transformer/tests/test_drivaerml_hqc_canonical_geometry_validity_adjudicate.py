# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the four-lane canonical-geometry adjudicator."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

STUDIES = Path(__file__).resolve().parents[1] / "studies"
sys.path.insert(0, str(STUDIES))

import drivaerml_hqc_canonical_geometry_validity_adjudicate as audit  # noqa: E402

PRODUCER_SHA256 = "a" * 64
SOURCE_TREE_SHA256 = "b" * 64
GEOMETRY_MANIFEST_SHA256 = "c" * 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _write_sidecar(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n",
        encoding="ascii",
    )


def _array_manifest(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256_array(value),
        }
        for name, value in sorted(arrays.items())
    }


def _arrays_for_lane(lane_ordinal: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for ordinal, case_id in audit._expected_lane_cases(lane_ordinal):
        _, n_master_cells, historical_start = audit.EXPECTED_CASE_METADATA[ordinal]
        for resolution in audit.RESOLUTIONS:
            prefix = audit._unit_prefix(ordinal, case_id, resolution)
            arrays[f"{prefix}__selected_cell_ids_int64"] = (
                historical_start + np.arange(resolution, dtype="<i8")
            ) % n_master_cells
            arrays[f"{prefix}__canonical_cells_int64"] = np.column_stack(
                (
                    np.arange(resolution, dtype="<i8"),
                    np.arange(1, resolution + 1, dtype="<i8"),
                    np.arange(2, resolution + 2, dtype="<i8"),
                )
            )
            points = np.zeros((resolution + 2, 3), dtype="<f4")
            points[:, 0] = np.arange(resolution + 2, dtype="<f4")
            arrays[f"{prefix}__canonical_points_float32"] = points
            arrays[f"{prefix}__canonical_centroids_float32"] = np.zeros(
                (resolution, 3),
                dtype="<f4",
            )
            arrays[f"{prefix}__canonical_areas_float32"] = np.ones(
                resolution,
                dtype="<f4",
            )
            normals = np.zeros((resolution, 3), dtype="<f4")
            normals[:, 0] = 1.0
            arrays[f"{prefix}__canonical_normals_float32"] = normals
            for precision in audit.PRECISIONS:
                for panel in audit.QUERY_PANELS:
                    count = (
                        resolution if panel == "coupled_s_k" else audit.FIXED_QUERY_K
                    )
                    for path in audit.PATHS:
                        arrays[
                            f"{prefix}__{precision}_canonical_full_"
                            f"{panel}_{path}_pressure"
                        ] = np.zeros(count, dtype="<f4")
                        arrays[
                            f"{prefix}__{precision}_canonical_full_{panel}_{path}_wss"
                        ] = np.zeros((count, 3), dtype="<f4")
    return arrays


def _anchor_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    resolution = audit.RESOLUTIONS[0]
    for ordinal, case_id in enumerate(audit.CASE_IDS[:4]):
        source = _arrays_for_lane(ordinal % audit.LANE_COUNT)
        lane_prefix = audit._unit_prefix(ordinal, case_id, resolution)
        anchor_prefix = f"case_{ordinal:02d}_{case_id}"
        for relative_name in audit._relative_unit_array_names()[:6]:
            arrays[f"{anchor_prefix}__{relative_name}"] = source[
                f"{lane_prefix}__{relative_name}"
            ]
        for precision in audit.PRECISIONS:
            for mode in ("derived", "full"):
                for path in audit.PATHS:
                    for field in audit.FIELDS:
                        source_name = (
                            f"{lane_prefix}__{precision}_canonical_full_"
                            f"coupled_s_k_{path}_{field}"
                        )
                        arrays[
                            f"{anchor_prefix}__{precision}_canonical_"
                            f"{mode}_{path}_{field}"
                        ] = source[source_name]
            for path in ("primary", "fixed"):
                for field in audit.FIELDS:
                    source_name = (
                        f"{lane_prefix}__{precision}_canonical_full_"
                        f"coupled_s_k_{path}_{field}"
                    )
                    arrays[
                        f"{anchor_prefix}__{precision}_historical_{path}_{field}"
                    ] = source[source_name]
                for field in audit.GEOMETRY_FIELDS:
                    source_name = {
                        "points": "canonical_points_float32",
                        "centroids": "canonical_centroids_float32",
                        "areas": "canonical_areas_float32",
                        "normals": "canonical_normals_float32",
                    }[field]
                    arrays[
                        f"{anchor_prefix}__{precision}_historical_model_"
                        f"{path}_source_{field}"
                    ] = source[f"{lane_prefix}__{source_name}"]
    assert set(arrays) == set(audit._expected_anchor_array_names())
    return arrays


def _difference(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = max(
        float(np.linalg.norm(left.astype(np.float64))),
        float(np.linalg.norm(right.astype(np.float64))),
        1.0e-12,
    )
    return {
        "shape": list(left.shape),
        "left_dtype": "torch.float32",
        "right_dtype": "torch.float32",
        "exact": audit._array_bitwise_equal(left, right),
        "nonzero_count": int(np.count_nonzero(delta)),
        "maximum_absolute_difference": (
            float(np.max(np.abs(delta))) if delta.size else 0.0
        ),
        "relative_l2_difference": float(np.linalg.norm(delta) / denominator),
    }


def _full_probe_summary(
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
    precision: str,
    resolution: int,
) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel in audit.QUERY_PANELS:
        panel_prefix = f"{prefix}__{precision}_canonical_full_{panel}"
        primary_fixed = {
            field: _difference(
                arrays[f"{panel_prefix}_primary_{field}"],
                arrays[f"{panel_prefix}_fixed_{field}"],
            )
            for field in audit.FIELDS
        }
        primary_replay = {
            field: _difference(
                arrays[f"{panel_prefix}_primary_{field}"],
                arrays[f"{panel_prefix}_primary_replay_{field}"],
            )
            for field in audit.FIELDS
        }
        replay_exact = all(row["exact"] for row in primary_replay.values())
        comparison_passed = all(row["exact"] for row in primary_fixed.values())
        panels[panel] = {
            "query_count": (
                resolution if panel == "coupled_s_k" else audit.FIXED_QUERY_K
            ),
            "source": (
                "single_public_decode"
                if panel == "coupled_s_k"
                else "first_2500_rows_of_single_public_decode"
            ),
            "primary_fixed_difference": primary_fixed,
            "primary_replay_difference": primary_replay,
            "primary_replay_exact": replay_exact,
            "comparison_gate": {
                "criterion": "fieldwise_bitwise_exact",
                "passed": comparison_passed,
                "controls_candidate_advance": panel == "coupled_s_k",
            },
            "validity_passed": replay_exact,
        }
    prefix_checks = {
        path: {
            field: audit._array_bitwise_equal(
                arrays[
                    f"{prefix}__{precision}_canonical_full_coupled_s_k_{path}_{field}"
                ][: audit.FIXED_QUERY_K],
                arrays[
                    f"{prefix}__{precision}_canonical_full_"
                    f"fixed_id_prefix_s2500_{path}_{field}"
                ],
            )
            for field in audit.FIELDS
        }
        for path in audit.PATHS
    }
    prefix_passed = all(
        check
        for path_checks in prefix_checks.values()
        for check in path_checks.values()
    )
    full_passed = panels["coupled_s_k"]["comparison_gate"]["passed"]
    validity_passed = (
        all(panel["validity_passed"] for panel in panels.values()) and prefix_passed
    )
    all_true_injection = {
        path: {field: True for field in audit.GEOMETRY_FIELDS} for path in audit.PATHS
    }
    all_true_storage = {
        path: {field: True for field in audit.STORAGE_FIELDS} for path in audit.PATHS
    }
    all_true_decode = {
        path: {field: True for field in audit.DECODE_CHECKS} for path in audit.PATHS
    }
    return {
        "mode": "canonical_full_public_api",
        "injected_geometry_exact": all_true_injection,
        "injected_geometry_exact_passed": True,
        "authoritative_storage_identity": all_true_storage,
        "authoritative_storage_identity_passed": True,
        "canonical_decode_contract": all_true_decode,
        "canonical_decode_contract_passed": True,
        "query_panels": panels,
        "fixed_id_prefix_matches_coupled_rows": prefix_checks,
        "fixed_id_prefix_matches_coupled_rows_passed": prefix_passed,
        "comparison_gate": {
            "criterion": "whole_trace_fieldwise_bitwise_exact",
            "passed": full_passed,
        },
        "validity_passed": validity_passed,
    }


def _resolution_summary(
    arrays: dict[str, np.ndarray],
    *,
    ordinal: int,
    case_id: str,
    resolution: int,
) -> dict[str, Any]:
    prefix = audit._unit_prefix(ordinal, case_id, resolution)
    probes: dict[str, Any] = {}
    for precision in audit.PRECISIONS:
        full = _full_probe_summary(
            arrays,
            prefix=prefix,
            precision=precision,
            resolution=resolution,
        )
        probes[precision] = {
            "precision": precision,
            "canonical_full_public_api": full,
            "validity_passed": full["validity_passed"],
            "decision_gates": {
                "full_passed": full["comparison_gate"]["passed"],
            },
        }
    anchor_required = ordinal < 4 and resolution == audit.RESOLUTIONS[0]
    anchor_replay: dict[str, Any] = {
        "required": anchor_required,
        "passed": True,
        "compared_arrays": (
            len(audit._relative_unit_array_names()) if anchor_required else 0
        ),
    }
    if anchor_required:
        anchor_replay["comparisons"] = {
            name: True for name in audit._relative_unit_array_names()
        }
    validity_passed = all(probe["validity_passed"] for probe in probes.values())
    full_passed = all(
        probe["decision_gates"]["full_passed"] for probe in probes.values()
    )
    return {
        "case_id": case_id,
        "cohort_ordinal": ordinal,
        "reader_index": audit.EXPECTED_CASE_METADATA[ordinal][0],
        "resolution": resolution,
        "canonical_frame": {
            "construction": (
                "raw selected coordinates promoted to float64; physical "
                "area-weighted center removed; coherent triangle geometry "
                "divided by L_ref*model_reference_length; one float32 cast"
            ),
            "physical_center_float64": [0.0, 0.0, 0.0],
            "physical_length": audit.EXPECTED_PHYSICAL_LENGTH,
            "model_reference_length": audit.EXPECTED_MODEL_REFERENCE_LENGTH,
            "effective_physical_length": (
                audit.EXPECTED_PHYSICAL_LENGTH * audit.EXPECTED_MODEL_REFERENCE_LENGTH
            ),
            "queries": "canonical_trace_centroids",
        },
        "historical_centers": {
            "primary_point_mean_float32": [0.0, 0.0, 0.0],
            "fixed_s10000_point_mean_float32": [0.0, 0.0, 0.0],
        },
        "validity": {
            "canonical_bundle": {
                "passed": True,
                "checks": {
                    "shapes": True,
                    "topology": True,
                    "finite": True,
                    "positive_areas": True,
                    "unit_normals": True,
                    "area_centered": True,
                },
                "shape_checks": {
                    "points": True,
                    "cells": True,
                    "centroids": True,
                    "areas": True,
                    "normals": True,
                },
                "finite_checks": {field: True for field in audit.GEOMETRY_FIELDS},
                "maximum_unit_deviation": 0.0,
                "maximum_area_center_deviation": 0.0,
            },
            "canonical_construction_replay": {
                check: True for check in audit.CONSTRUCTION_CHECKS
            },
            "canonical_construction_replay_passed": True,
            "historical_path_topology": {
                check: True for check in audit.TOPOLOGY_CHECKS
            },
            "historical_path_topology_passed": True,
            "fixed_q_is_exact_source_prefix": True,
            "job305691_anchor_replay": anchor_replay,
            "model_local_data_stripped": True,
            "model_probes_executed": True,
        },
        "precision_probes": probes,
        "validity_passed": validity_passed,
        "decision_gates": {"full_passed": full_passed},
        "decision_outcome": (
            audit.EXACT_OUTCOME if full_passed else audit.REFUTED_OUTCOME
        ),
    }


def _lane_summary(
    lane_ordinal: int,
    arrays: dict[str, np.ndarray],
    npz_sha256: str,
) -> dict[str, Any]:
    expected_cases = audit._expected_lane_cases(lane_ordinal)
    comparisons_per_lane = (
        len(expected_cases)
        * len(audit.RESOLUTIONS)
        * len(audit.PRECISIONS)
        * len(audit.FIELDS)
    )
    deduplicated_per_lane = (
        len(expected_cases)
        * len(audit.PRECISIONS)
        * len(audit.FIELDS)
        * (1 + 2 * (len(audit.RESOLUTIONS) - 1))
    )
    emitted_per_lane = (
        len(expected_cases)
        * len(audit.RESOLUTIONS)
        * len(audit.PRECISIONS)
        * len(audit.QUERY_PANELS)
        * len(audit.FIELDS)
    )
    cases = [
        {
            "case_id": case_id,
            "cohort_ordinal": ordinal,
            "reader_index": audit.EXPECTED_CASE_METADATA[ordinal][0],
            "resolutions": [
                _resolution_summary(
                    arrays,
                    ordinal=ordinal,
                    case_id=case_id,
                    resolution=resolution,
                )
                for resolution in audit.RESOLUTIONS
            ],
            "validity_passed": True,
            "decision_gates": {"full_passed": True},
            "decision_outcome": audit.EXACT_OUTCOME,
        }
        for ordinal, case_id in expected_cases
    ]
    for case in cases:
        case["validity_passed"] = all(
            row["validity_passed"] for row in case["resolutions"]
        )
        case["decision_gates"]["full_passed"] = all(
            row["decision_gates"]["full_passed"] for row in case["resolutions"]
        )
        case["decision_outcome"] = (
            audit.EXACT_OUTCOME
            if case["decision_gates"]["full_passed"]
            else audit.REFUTED_OUTCOME
        )
    all_validity_passed = all(case["validity_passed"] for case in cases)
    all_full_passed = all(case["decision_gates"]["full_passed"] for case in cases)
    return {
        "schema_version": audit.LANE_SCHEMA_VERSION,
        "artifact_kind": audit.LANE_ARTIFACT_KIND,
        "status": (
            audit.VALID_LANE_STATUS
            if all_validity_passed
            else audit.INVALID_LANE_STATUS
        ),
        "decision_outcome": (
            audit.INVALID_DIAGNOSTIC
            if not all_validity_passed
            else (audit.EXACT_OUTCOME if all_full_passed else audit.REFUTED_OUTCOME)
        ),
        "generated_at_utc": "2026-07-28T12:00:00+00:00",
        "lane": {"ordinal": lane_ordinal, "count": audit.LANE_COUNT},
        "scientific_scope": {
            "case_ids": [case_id for _, case_id in expected_cases],
            "resolutions": list(audit.RESOLUTIONS),
            "precisions": list(audit.PRECISIONS),
            "licensing_field_tensor_comparisons_per_lane": comparisons_per_lane,
            "licensing_field_tensor_comparisons_full_cohort": (
                comparisons_per_lane * audit.LANE_COUNT
            ),
            "deduplicated_panel_field_summaries_per_lane": (deduplicated_per_lane),
            "deduplicated_panel_field_summaries_full_cohort": (
                deduplicated_per_lane * audit.LANE_COUNT
            ),
            "emitted_panel_field_records_per_lane": emitted_per_lane,
            "emitted_panel_field_records_full_cohort": (
                emitted_per_lane * audit.LANE_COUNT
            ),
            "prefix_summaries_are_independent_decisions": False,
            "supervision_arrays_indexed": False,
            "supervision_files_opened_by_model_producer": False,
            "raw_dataset_sample_loader_called": False,
            "geometry_only_memmap_allowlist_applied": True,
            "synthetic_placeholders_stripped_before_model": True,
            "hqc_decision_statistics_computed": False,
            "may_not_be_used_as_hqc_verdict_output": True,
        },
        "contract": audit.EXPECTED_CONTRACT,
        "validity": {
            "all_cases_resolutions_and_precisions_passed": all_validity_passed,
            "geometry_input_manifest_lane_verification": {
                "manifest_sha256": GEOMETRY_MANIFEST_SHA256,
                "lane_cases_verified": len(expected_cases),
                "lane_files_verified": 13 * len(expected_cases),
            },
            "required_gates": audit.EXPECTED_REQUIRED_GATES,
        },
        "decision_gates": {
            "full": {
                "criterion": audit.FULL_GATE_CRITERION,
                "passed": all_full_passed,
                "controls_candidate_advance": True,
            },
        },
        "cases": cases,
        "npz_array_manifest": _array_manifest(arrays),
        "provenance": {
            "command": ["producer.py"],
            "diagnostic_script_path": "/task/producer.py",
            "diagnostic_script_sha256": PRODUCER_SHA256,
            "canonical_helper_path": "/task/helper.py",
            "canonical_helper_sha256": audit.EXPECTED_CANONICAL_HELPER_SHA256,
            "frozen_producer_path": "/task/hqc.py",
            "frozen_producer_sha256": (audit.EXPECTED_FROZEN_HQC_PRODUCER_SHA256),
            "import_provenance": {},
            "source_tree_manifest_sha256": SOURCE_TREE_SHA256,
            "input_hashes": {
                **audit.EXPECTED_STABLE_INPUT_HASHES,
                "geometry_input_manifest": GEOMETRY_MANIFEST_SHA256,
            },
            "npz_path": f"/task/lane{lane_ordinal}.npz",
            "npz_sha256": npz_sha256,
            "slurm_job_id": str(100 + lane_ordinal),
            "python": "3.13.0",
            "platform": "Linux",
            "numpy": np.__version__,
            "torch": "test",
            "hardware": {},
        },
    }


def _write_lane(tmp_path: Path, lane_ordinal: int) -> tuple[Path, Path]:
    arrays = _arrays_for_lane(lane_ordinal)
    npz_path = tmp_path / f"lane{lane_ordinal}.npz"
    np.savez(npz_path, **arrays)
    _write_sidecar(npz_path)
    summary = _lane_summary(lane_ordinal, arrays, _sha256_file(npz_path))
    json_path = tmp_path / f"lane{lane_ordinal}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(json_path)
    return json_path, npz_path


def _rewrite_lane(
    lane: tuple[Path, Path],
    mutation: Callable[[dict[str, Any], dict[str, np.ndarray]], None],
    *,
    rebuild_summary: bool = False,
) -> None:
    json_path, npz_path = lane
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutation(summary, arrays)
    np.savez(npz_path, **arrays)
    _write_sidecar(npz_path)
    if rebuild_summary:
        summary = _lane_summary(
            int(summary["lane"]["ordinal"]),
            arrays,
            _sha256_file(npz_path),
        )
    else:
        summary["npz_array_manifest"] = _array_manifest(arrays)
        summary["provenance"]["npz_sha256"] = _sha256_file(npz_path)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(json_path)


@pytest.fixture
def expected_provenance() -> audit.ExpectedProvenance:
    return audit.ExpectedProvenance(
        lane_producer_sha256=PRODUCER_SHA256,
        source_tree_sha256=SOURCE_TREE_SHA256,
        geometry_input_manifest_sha256=GEOMETRY_MANIFEST_SHA256,
    )


@pytest.fixture
def anchor_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(audit, "RESOLUTIONS", (2, 3, 4, 5, 6))
    monkeypatch.setattr(audit, "FIXED_QUERY_K", 2)
    path = tmp_path / "anchor.npz"
    np.savez(path, **_anchor_arrays())
    _write_sidecar(path)
    monkeypatch.setitem(
        audit.EXPECTED_STABLE_INPUT_HASHES,
        "job305691_anchor_npz",
        _sha256_file(path),
    )
    return path


@pytest.fixture
def lanes(
    tmp_path: Path,
    anchor_npz: Path,
) -> list[tuple[Path, Path]]:
    del anchor_npz
    return [_write_lane(tmp_path, lane) for lane in range(audit.LANE_COUNT)]


def test_all_four_lanes_pass(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.VALID_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.EXACT_OUTCOME
    assert result["decision"]["validity_passed"] is True
    assert result["decision"]["full_passed"] is True
    assert result["decision"]["full_mismatch_count"] == 0
    assert result["decision"]["full_field_tensor_comparisons"] == 720
    assert result["decision"]["replay_and_prefix_tensor_comparisons"] == 3_600
    assert result["decision"]["anchor_tensor_comparisons"] == 120


def test_missing_lane_is_incomplete_not_refuted(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    result = audit.adjudicate(
        lane_artifacts=lanes[:3],
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INCOMPLETE_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.INCOMPLETE_OUTCOME
    assert "REFUTED" not in result["decision"]["outcome"]


def test_primary_replay_mismatch_invalidates_experiment(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    def mutate(
        summary: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        del summary
        ordinal, case_id = audit._expected_lane_cases(0)[0]
        prefix = audit._unit_prefix(ordinal, case_id, audit.RESOLUTIONS[1])
        key = f"{prefix}__bfloat16_canonical_full_coupled_s_k_primary_replay_pressure"
        arrays[key][audit.FIXED_QUERY_K] = 1.0

    _rewrite_lane(lanes[0], mutate, rebuild_summary=True)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.INVALID_OUTCOME


def test_one_full_ab_mismatch_is_valid_refutation(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    def mutate(
        summary: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        del summary
        ordinal, case_id = audit._expected_lane_cases(0)[0]
        prefix = audit._unit_prefix(ordinal, case_id, audit.RESOLUTIONS[1])
        key = f"{prefix}__bfloat16_canonical_full_coupled_s_k_fixed_pressure"
        arrays[key][audit.FIXED_QUERY_K] = 1.0

    _rewrite_lane(lanes[0], mutate, rebuild_summary=True)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.VALID_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.REFUTED_OUTCOME
    assert result["decision"]["validity_passed"] is True
    assert result["decision"]["full_passed"] is False
    assert result["decision"]["full_mismatch_count"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_count"])
def test_duplicate_or_wrong_lane_is_invalid(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
    mutation: str,
) -> None:
    if mutation == "duplicate":
        supplied = [lanes[0], lanes[1], lanes[2], lanes[0]]
    else:
        _rewrite_lane(
            lanes[3],
            lambda summary, arrays: summary["lane"].__setitem__("count", 5),
        )
        supplied = lanes

    result = audit.adjudicate(
        lane_artifacts=supplied,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.INVALID_OUTCOME


@pytest.mark.parametrize("corruption", ["npz", "sidecar"])
def test_corrupt_npz_or_sidecar_is_incomplete(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
    corruption: str,
) -> None:
    _, npz_path = lanes[2]
    if corruption == "npz":
        npz_path.write_bytes(b"not an npz")
        _write_sidecar(npz_path)
    else:
        npz_path.with_name(f"{npz_path.name}.sha256").write_text(
            f"{'0' * 64}  {npz_path.name}\n",
            encoding="ascii",
        )

    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INCOMPLETE_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.INCOMPLETE_OUTCOME


def test_forbidden_json_key_is_rejected(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    def mutate(
        summary: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        del arrays
        summary["contract"]["target_pressure"] = "forbidden"

    _rewrite_lane(lanes[1], mutate)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert "Forbidden JSON keys" in result["failures"][0]["message"]


@pytest.mark.parametrize(
    "constituent",
    [
        "contract",
        "top_decision_gate",
        "construction_replay",
        "bundle_area_centered",
        "historical_topology",
        "source_prefix",
        "model_probe",
        "public_injection",
        "authoritative_storage",
        "decode_contract",
        "prefix_slice",
        "primary_replay",
        "anchor_replay",
    ],
)
def test_false_constituent_with_true_rollups_is_invalid(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
    constituent: str,
) -> None:
    def mutate(
        summary: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        del arrays
        row = summary["cases"][0]["resolutions"][0]
        full = row["precision_probes"]["bfloat16"]["canonical_full_public_api"]
        mutations: dict[str, Callable[[], None]] = {
            "contract": lambda: summary.__setitem__("contract", {}),
            "top_decision_gate": lambda: summary.__setitem__("decision_gates", {}),
            "construction_replay": lambda: row["validity"][
                "canonical_construction_replay"
            ].__setitem__("points", False),
            "bundle_area_centered": lambda: row["validity"]["canonical_bundle"][
                "checks"
            ].__setitem__("area_centered", False),
            "historical_topology": lambda: row["validity"][
                "historical_path_topology"
            ].__setitem__("primary_matches_selected", False),
            "source_prefix": lambda: row["validity"].__setitem__(
                "fixed_q_is_exact_source_prefix",
                False,
            ),
            "model_probe": lambda: row["validity"].__setitem__(
                "model_probes_executed",
                False,
            ),
            "public_injection": lambda: full["injected_geometry_exact"][
                "primary"
            ].__setitem__("points", False),
            "authoritative_storage": lambda: full["authoritative_storage_identity"][
                "primary"
            ].__setitem__("points", False),
            "decode_contract": lambda: full["canonical_decode_contract"][
                "primary"
            ].__setitem__("canonical_queries_exact", False),
            "prefix_slice": lambda: full["fixed_id_prefix_matches_coupled_rows"][
                "primary"
            ].__setitem__("pressure", False),
            "primary_replay": lambda: full["query_panels"]["coupled_s_k"][
                "primary_replay_difference"
            ]["pressure"].__setitem__("exact", False),
            "anchor_replay": lambda: row["validity"]["job305691_anchor_replay"][
                "comparisons"
            ].__setitem__("canonical_points_float32", False),
        }
        mutations[constituent]()

    _rewrite_lane(lanes[0], mutate)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert result["decision"]["outcome"] == audit.INVALID_OUTCOME


def test_required_anchor_is_recomputed_from_frozen_npz(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    def mutate(
        summary: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        del summary
        ordinal, case_id = audit._expected_lane_cases(0)[0]
        prefix = audit._unit_prefix(ordinal, case_id, audit.RESOLUTIONS[0])
        arrays[f"{prefix}__canonical_points_float32"][0, 1] = 1.0

    _rewrite_lane(lanes[0], mutate, rebuild_summary=True)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert "differs from frozen anchor" in result["failures"][0]["message"]


def test_changed_anchor_with_self_consistent_sidecar_is_invalid(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    with np.load(anchor_npz, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    first_name = sorted(arrays)[0]
    arrays[first_name].reshape(-1)[0] += 1
    np.savez(anchor_npz, **arrays)
    _write_sidecar(anchor_npz)

    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INVALID_ADJUDICATION_STATUS
    assert result["failures"][0]["kind"] == "wrong_anchor_artifact"


def test_json_is_parsed_from_the_same_bytes_that_were_verified(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = audit._load_json_bytes
    replaced = False

    def replace_path_after_load(payload: bytes, name: str) -> dict[str, Any]:
        nonlocal replaced
        if name == lanes[0][0].name and not replaced:
            replaced = True
            lanes[0][0].write_bytes(b"{}\n")
            _write_sidecar(lanes[0][0])
        return dict(real_load(payload, name))

    monkeypatch.setattr(audit, "_load_json_bytes", replace_path_after_load)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert replaced
    assert result["status"] == audit.VALID_ADJUDICATION_STATUS


def test_npz_is_parsed_from_the_same_bytes_that_were_verified(
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = audit._load_npz_bytes
    replaced = False

    def replace_path_after_load(
        payload: bytes,
        name: str,
        expected_names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        nonlocal replaced
        if name == lanes[0][1].name and not replaced:
            replaced = True
            lanes[0][1].write_bytes(b"not the verified NPZ")
            _write_sidecar(lanes[0][1])
        return real_load(payload, name, expected_names)

    monkeypatch.setattr(audit, "_load_npz_bytes", replace_path_after_load)
    result = audit.adjudicate(
        lane_artifacts=lanes,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert replaced
    assert result["status"] == audit.VALID_ADJUDICATION_STATUS


def test_symlinked_parent_directory_is_unavailable(
    tmp_path: Path,
    lanes: list[tuple[Path, Path]],
    expected_provenance: audit.ExpectedProvenance,
    anchor_npz: Path,
) -> None:
    real_directory = tmp_path / "real_lane"
    real_directory.mkdir()
    copied_pair: list[Path] = []
    for source in lanes[0]:
        copied = real_directory / source.name
        copied.write_bytes(source.read_bytes())
        copied.with_name(f"{copied.name}.sha256").write_bytes(
            source.with_name(f"{source.name}.sha256").read_bytes()
        )
        copied_pair.append(copied)
    alias = tmp_path / "lane_alias"
    alias.symlink_to(real_directory, target_is_directory=True)
    supplied = [
        (alias / copied_pair[0].name, alias / copied_pair[1].name),
        *lanes[1:],
    ]

    result = audit.adjudicate(
        lane_artifacts=supplied,
        expected_provenance=expected_provenance,
        anchor_npz_path=anchor_npz,
    )

    assert result["status"] == audit.INCOMPLETE_ADJUDICATION_STATUS
    assert "traverses a symlink" in result["failures"][0]["message"]


def test_atomic_writer_refuses_output_or_sidecar_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "adjudication.json"
    result = {"status": "test"}
    audit.write_adjudication(output, result)

    assert output.is_file()
    assert output.with_name(f"{output.name}.sha256").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.write_adjudication(output, result)


def test_atomic_writer_rolls_back_json_if_sidecar_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adjudication.json"
    real_publish = audit._link_temporary_no_clobber
    calls = 0

    def fail_second_publication(temporary: Path, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError("simulated sidecar race")
        real_publish(temporary, path)

    monkeypatch.setattr(
        audit,
        "_link_temporary_no_clobber",
        fail_second_publication,
    )
    with pytest.raises(FileExistsError, match="simulated sidecar race"):
        audit.write_adjudication(output, {"status": "test"})

    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_atomic_writer_rolls_back_if_sidecar_link_succeeds_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adjudication.json"
    real_publish = audit._link_temporary_no_clobber
    calls = 0

    def fail_after_second_publication(temporary: Path, path: Path) -> None:
        nonlocal calls
        calls += 1
        real_publish(temporary, path)
        if calls == 2:
            raise OSError("simulated post-link failure")

    monkeypatch.setattr(
        audit,
        "_link_temporary_no_clobber",
        fail_after_second_publication,
    )
    with pytest.raises(OSError, match="simulated post-link failure"):
        audit.write_adjudication(output, {"status": "test"})

    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_atomic_writer_detects_and_does_not_bless_concurrent_json_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adjudication.json"
    real_publish = audit._link_temporary_no_clobber
    calls = 0

    def mutate_after_json_publication(temporary: Path, path: Path) -> None:
        nonlocal calls
        calls += 1
        real_publish(temporary, path)
        if calls == 1:
            path.write_bytes(b"concurrent mutation\n")

    monkeypatch.setattr(
        audit,
        "_link_temporary_no_clobber",
        mutate_after_json_publication,
    )
    with pytest.raises(OSError, match="changed during transaction"):
        audit.write_adjudication(output, {"status": "test"})

    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()
