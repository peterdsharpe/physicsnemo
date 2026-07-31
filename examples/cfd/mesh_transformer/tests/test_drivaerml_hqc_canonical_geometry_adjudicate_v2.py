# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for independent canonical-geometry adjudication."""

import hashlib
import json
import zipfile
from pathlib import Path

import drivaerml_hqc_canonical_geometry_adjudicate_v2 as audit
import numpy as np
import pytest


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n",
        encoding="ascii",
    )


def _write_bundle_manifest(root: Path) -> Path:
    manifest = root / "manifest.sha256"
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path != manifest
    )
    manifest.write_text(
        "".join(
            f"{_sha256_file(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in members
        ),
        encoding="ascii",
    )
    return manifest


def _prediction(offset: float, field: str) -> np.ndarray:
    base = np.linspace(1.0 + offset, 2.0 + offset, 2500, dtype=np.float32)
    if field == "pressure":
        return base
    return np.stack((base, 0.5 * base, -0.25 * base), axis=1).astype(
        np.float32,
    )


def _difference_maps(primary, fixed, replay):
    return (
        {
            field: audit._difference(primary[field], fixed[field])
            for field in audit.FIELDS
        },
        {
            field: audit._difference(primary[field], replay[field])
            for field in audit.FIELDS
        },
    )


def _synthetic_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    prior: dict[str, np.ndarray] = {}
    cases: list[dict] = []
    for case_index, case_id, reader_index in audit.CASE_SPECS:
        prefix = f"case_{case_index:02d}_{case_id}"
        ids = np.arange(2500, dtype="<i8")
        cells = np.tile(np.array([[0, 1, 2]], dtype="<i8"), (2500, 1))
        points = np.array(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 2.0, 0.0]],
            dtype="<f4",
        )
        centroids = np.zeros((2500, 3), dtype="<f4")
        areas = np.full(2500, 3.0, dtype="<f4")
        normals = np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype="<f4"),
            (2500, 1),
        )
        arrays.update(
            {
                f"{prefix}__selected_cell_ids_int64": ids,
                f"{prefix}__canonical_cells_int64": cells,
                f"{prefix}__canonical_points_float32": points,
                f"{prefix}__canonical_centroids_float32": centroids,
                f"{prefix}__canonical_areas_float32": areas,
                f"{prefix}__canonical_normals_float32": normals,
            }
        )
        prior[f"{prefix}__cell_ids_int64"] = ids.copy()
        precision_probes: dict[str, dict] = {}
        for precision_index, precision in enumerate(audit.PRECISIONS):
            modes: dict[str, dict] = {}
            for mode in audit.MODES:
                primary = {
                    field: _prediction(
                        case_index + 0.1 * precision_index,
                        field,
                    )
                    for field in audit.FIELDS
                }
                fixed = {field: value.copy() for field, value in primary.items()}
                if mode == "canonical_derived":
                    fixed = {
                        field: value + np.float32(1.0e-5)
                        for field, value in fixed.items()
                    }
                replay = {field: value.copy() for field, value in primary.items()}
                for field in audit.FIELDS:
                    arrays[f"{prefix}__{precision}_{mode}_primary_{field}"] = primary[
                        field
                    ]
                    arrays[f"{prefix}__{precision}_{mode}_fixed_{field}"] = fixed[field]
                    arrays[f"{prefix}__{precision}_{mode}_primary_replay_{field}"] = (
                        replay[field]
                    )
                primary_fixed, primary_replay = _difference_maps(
                    primary,
                    fixed,
                    replay,
                )
                geometry_fields = (
                    ("centroids", "areas", "normals")
                    if mode == "canonical_derived"
                    else ("points", "centroids", "areas", "normals")
                )
                decode_checks = {
                    "canonical_queries_exact": True,
                    "encoded_center_is_exact_zero": True,
                    "encoded_reference_length_is_exact_one": True,
                }
                modes[mode] = {
                    "mode": mode,
                    "primary_fixed_difference": primary_fixed,
                    "primary_replay_difference": primary_replay,
                    "primary_replay_exact": True,
                    "injected_geometry_exact": {
                        path: {name: True for name in geometry_fields}
                        for path in audit.PATHS
                    },
                    "canonical_decode_contract": {
                        path: dict(decode_checks)
                        for path in ("primary", "fixed", "primary_replay")
                    },
                    "canonical_decode_contract_passed": True,
                    "comparison_gate": {
                        "criterion": (
                            "fieldwise_relative_l2_le_1e-3"
                            if mode == "canonical_derived"
                            else "fieldwise_bitwise_exact"
                        ),
                        "passed": True,
                    },
                    "validity_passed": True,
                }
            prior_prediction_report: dict[str, dict] = {}
            prior_geometry_report: dict[str, dict] = {}
            for path_index, path in enumerate(audit.PATHS):
                prior_prediction_report[path] = {}
                prior_geometry_report[path] = {}
                for field in audit.FIELDS:
                    value = _prediction(
                        case_index + precision_index + path_index,
                        field,
                    )
                    current_key = f"{prefix}__{precision}_historical_{path}_{field}"
                    prior_key = f"{prefix}__{precision}_{path}_{field}"
                    arrays[current_key] = value
                    prior[prior_key] = value.copy()
                    prior_prediction_report[path][field] = audit._difference(
                        value,
                        prior[prior_key],
                    )
                geometry = {
                    "points": points,
                    "centroids": centroids,
                    "areas": areas,
                    "normals": normals,
                }
                for name, value in geometry.items():
                    current_key = (
                        f"{prefix}__{precision}_historical_model_{path}_source_{name}"
                    )
                    prior_key = f"{prefix}__{precision}_model_{path}_source_{name}"
                    arrays[current_key] = value.copy()
                    prior[prior_key] = value.copy()
                    prior_geometry_report[path][name] = audit._difference(
                        value,
                        prior[prior_key],
                    )
            precision_probes[precision] = {
                "precision": precision,
                "modes": modes,
                "validity_passed": True,
                "decision_gates": {
                    "derived_passed": True,
                    "full_passed": True,
                },
                "job304002_historical_replay": {
                    "job304002_primary_fixed_predictions": (prior_prediction_report),
                    "job304002_model_source_geometry": prior_geometry_report,
                    "passed": True,
                },
            }
        cases.append(
            {
                "case_id": case_id,
                "cohort_ordinal": case_index,
                "reader_index": reader_index,
                "resolution": 2500,
                "canonical_frame": {
                    "construction": (
                        "raw selected coordinates promoted to float64; "
                        "physical area-weighted center removed"
                    ),
                    "physical_center_float64": [0.0, 0.0, 0.0],
                    "physical_length": 5.0,
                    "model_reference_length": 8.0,
                    "effective_physical_length": 40.0,
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
                        "finite_checks": {
                            "points": True,
                            "centroids": True,
                            "areas": True,
                            "normals": True,
                        },
                        "maximum_unit_deviation": 0.0,
                        "maximum_area_center_deviation": 0.0,
                    },
                    "canonical_construction_replay": {
                        "cells": True,
                        "points": True,
                        "centroids": True,
                        "areas": True,
                        "normals": True,
                        "physical_center": True,
                        "physical_length": True,
                        "model_reference_length": True,
                    },
                    "canonical_construction_replay_passed": True,
                    "historical_path_topology": {
                        "primary_matches_selected": True,
                        "fixed_matches_selected": True,
                        "primary_matches_fixed": True,
                    },
                    "historical_path_topology_passed": True,
                    "job304002_geometry_replay": {
                        "cell_ids_int64": True,
                        "pipeline_primary_points_float32": True,
                        "pipeline_fixed_points_float32": True,
                        "pipeline_primary_queries_float32": True,
                        "pipeline_fixed_queries_float32": True,
                    },
                    "job304002_geometry_replay_passed": True,
                    "model_local_data_stripped": True,
                    "model_probes_executed": True,
                },
                "precision_probes": precision_probes,
                "validity_passed": True,
                "decision_gates": {
                    "derived_passed": True,
                    "full_passed": True,
                },
                "decision_outcome": audit.FULL_AND_DERIVED_OUTCOME,
            }
        )

    diagnostic_npz = tmp_path / "diagnostic.npz"
    prior_npz = tmp_path / "prior.npz"
    np.savez(diagnostic_npz, **arrays)
    np.savez(prior_npz, **prior)
    _write_sidecar(diagnostic_npz)
    _write_sidecar(prior_npz)
    prior_sha256 = _sha256_file(prior_npz)
    expected_inputs = dict(audit.EXPECTED_INPUT_HASHES)
    expected_inputs["prior_diagnostic_npz"] = prior_sha256
    monkeypatch.setattr(audit, "EXPECTED_PRIOR_NPZ_SHA256", prior_sha256)
    monkeypatch.setattr(audit, "EXPECTED_INPUT_HASHES", expected_inputs)
    monkeypatch.setattr(
        audit,
        "EXPECTED_BUNDLE_MEMBER_SHA256_BY_BASENAME",
        {},
    )
    summary = {
        "schema_version": 2,
        "artifact_kind": audit.DIAGNOSTIC_ARTIFACT_KIND,
        "status": audit.VALID_STATUS,
        "decision_outcome": audit.FULL_AND_DERIVED_OUTCOME,
        "generated_at_utc": "2026-07-28T00:00:00+00:00",
        "scientific_scope": {
            "case_ids": list(audit.CASE_IDS),
            "resolution": 2500,
            "precisions": list(audit.PRECISIONS),
            "supervision_arrays_indexed": False,
            "synthetic_placeholders_stripped_before_model": True,
            "hqc_decision_statistics_computed": False,
            "may_not_be_used_as_hqc_verdict_output": True,
        },
        "contract": {
            "canonical_construction": (
                "float64 raw geometry -> physical area center -> divide by "
                "L_ref*model_reference_length -> one float32 cast"
            ),
            "canonical_derived_fields": ["centroids", "areas", "normals"],
            "canonical_full_fields": [
                "points",
                "centroids",
                "areas",
                "normals",
            ],
            "query_frame": "canonical_trace_centroids",
            "derived_fieldwise_relative_tolerance": 0.001,
            "full_comparison": "fieldwise_bitwise_exact",
        },
        "validity": {
            "all_cases_and_precisions_passed": True,
            "required_gates": [
                "canonical_construction_replay",
                "shape",
                "topology",
                "finite_positive_unit",
                "job304002_geometry_and_primary_fixed_prediction_replay",
                "primary_replay_exact",
            ],
        },
        "decision_gates": {
            "derived": {
                "criterion": (
                    "primary-versus-fixed pressure and WSS relative L2 <= 1e-3 "
                    "for every case and precision"
                ),
                "passed": True,
            },
            "full": {
                "criterion": (
                    "primary-versus-fixed pressure and WSS bitwise exact "
                    "for every case and precision"
                ),
                "passed": True,
            },
        },
        "cases": cases,
        "npz_array_manifest": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": audit._sha256_array(value),
            }
            for key, value in sorted(arrays.items())
        },
        "provenance": {
            "command": [],
            "diagnostic_script_path": "/frozen/diagnostic.py",
            "diagnostic_script_sha256": (audit.EXPECTED_DIAGNOSTIC_SCRIPT_SHA256),
            "frozen_producer_path": "/frozen/producer.py",
            "frozen_producer_sha256": audit.EXPECTED_PRODUCER_SHA256,
            "source_tree_manifest_sha256": audit.EXPECTED_SOURCE_TREE_SHA256,
            "selected_source_files": dict(audit.EXPECTED_SELECTED_SOURCE_FILES),
            "input_hashes": expected_inputs,
            "npz_path": str(diagnostic_npz),
            "npz_sha256": _sha256_file(diagnostic_npz),
            "slurm_job_id": "123",
            "python": "3.13",
            "platform": "test",
            "numpy": np.__version__,
            "torch": "test",
            "hardware": {},
        },
    }
    diagnostic_json = tmp_path / "diagnostic.json"
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(diagnostic_json)
    (tmp_path / "DONE_123").touch()
    (tmp_path / "STATUS_123").write_text(
        "rc=0\ncompleted_units=1/1\n",
        encoding="ascii",
    )
    manifest = _write_bundle_manifest(tmp_path)
    return manifest, diagnostic_json, diagnostic_npz, prior_npz


def _refresh_json_and_manifest(
    manifest: Path,
    diagnostic_json: Path,
    diagnostic_npz: Path,
) -> None:
    summary = json.loads(diagnostic_json.read_text())
    summary["provenance"]["npz_sha256"] = _sha256_file(diagnostic_npz)
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(diagnostic_npz)
    _write_sidecar(diagnostic_json)
    _write_bundle_manifest(manifest.parent)


def _adjudicate(paths):
    manifest, diagnostic_json, diagnostic_npz, prior_npz = paths
    return audit.adjudicate(
        bundle_manifest=manifest,
        diagnostic_json=diagnostic_json,
        diagnostic_npz=diagnostic_npz,
        prior_diagnostic_npz=prior_npz,
    )


@pytest.mark.parametrize(
    ("validity", "derived", "full", "expected"),
    [
        (False, True, True, audit.INVALID_DIAGNOSTIC),
        (True, True, True, audit.FULL_AND_DERIVED_OUTCOME),
        (True, False, True, audit.FULL_ONLY_OUTCOME),
        (True, True, False, audit.CANONICAL_REPAIR_REFUTED),
        (True, False, False, audit.CANONICAL_REPAIR_REFUTED),
    ],
)
def test_decision_table(validity, derived, full, expected):
    assert (
        audit._decision_outcome(
            validity_passed=validity,
            derived_passed=derived,
            full_passed=full,
        )
        == expected
    )


def test_full_synthetic_publication_gate(tmp_path, monkeypatch):
    result = _adjudicate(_synthetic_artifacts(tmp_path, monkeypatch))

    assert result["status"] == "PASSED_PUBLICATION_GATE"
    assert result["decision"] == {
        "derived_passed": True,
        "full_passed": True,
        "outcome": audit.FULL_AND_DERIVED_OUTCOME,
    }
    assert result["checks"]["all_prediction_comparisons_recomputed"] is True
    assert len(result["producer_only_claims_not_independently_regenerated"]) == 9


def test_fabricated_prior_identity_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(
        audit,
        "EXPECTED_PRIOR_NPZ_SHA256",
        "d1e6a9fa1a39aa78a9cca26e52eb783a9e78aecbb961ce917164e25fac75a7ea",
    )

    with pytest.raises(
        audit.AdjudicationFailure,
        match="not frozen job 304002",
    ):
        _adjudicate(paths)


def test_prediction_dtype_or_shape_tampering_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with np.load(diagnostic_npz, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    key = "case_00_run_118__float32_canonical_full_fixed_wss"
    arrays[key] = arrays[key].reshape(-1).astype(np.float64)
    np.savez(diagnostic_npz, **arrays)
    summary = json.loads(diagnostic_json.read_text())
    summary["npz_array_manifest"][key] = {
        "shape": list(arrays[key].shape),
        "dtype": str(arrays[key].dtype),
        "sha256": audit._sha256_array(arrays[key]),
    }
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="dtype or shape differs",
    ):
        _adjudicate(paths)


def test_array_manifest_record_extra_key_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    key = "case_00_run_118__canonical_points_float32"
    summary["npz_array_manifest"][key]["unexpected_schema_member"] = "arbitrary"
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="array-manifest record key set differs",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("cohort_ordinal", "0"),
        ("cohort_ordinal", False),
        ("reader_index", "21"),
    ],
)
def test_case_identity_type_coercion_fails_closed(
    tmp_path,
    monkeypatch,
    field,
    malformed,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    summary["cases"][0][field] = malformed
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="identity has the wrong type",
    ):
        _adjudicate(paths)


def test_duplicate_npz_member_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with zipfile.ZipFile(diagnostic_npz) as archive:
        name = archive.namelist()[0]
        payload = archive.read(name)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(diagnostic_npz, mode="a") as archive:
            archive.writestr(name, payload)
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="duplicate ZIP members",
    ):
        _adjudicate(paths)


def test_reported_validity_reduction_tampering_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    summary["cases"][0]["precision_probes"]["float32"]["modes"]["canonical_full"][
        "injected_geometry_exact"
    ]["primary"]["points"] = False
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="validity aggregate differs",
    ):
        _adjudicate(paths)


def test_truthy_string_validity_leaf_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    summary["cases"][0]["validity"]["canonical_construction_replay"]["points"] = "false"
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="non-boolean report",
    ):
        _adjudicate(paths)


def test_geometrically_incoherent_bundle_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with np.load(diagnostic_npz, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    key = "case_00_run_118__canonical_points_float32"
    arrays[key] = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype="<f4",
    )
    np.savez(diagnostic_npz, **arrays)
    summary = json.loads(diagnostic_json.read_text())
    summary["npz_array_manifest"][key]["sha256"] = audit._sha256_array(arrays[key])
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="degenerate",
    ):
        _adjudicate(paths)


def test_matching_done_and_success_status_are_required(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, _, _, _ = paths
    (tmp_path / "DONE_123").unlink()
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="DONE marker",
    ):
        _adjudicate(paths)


def test_bundle_manifest_must_be_complete(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    (tmp_path / "unrecorded").write_text("extra")

    with pytest.raises(
        audit.AdjudicationFailure,
        match="not complete",
    ):
        _adjudicate(paths)


def test_forbidden_schema_vocabulary_is_complete():
    found = audit._forbidden_paths(
        {
            "area_objective": {},
            "log_cliff": {},
            "reducer_output": {},
            "hqc_verdict": {},
            "scientific_scope": {
                "may_not_be_used_as_hqc_verdict_output": True,
            },
        }
    )

    assert found == [
        "area_objective",
        "log_cliff",
        "reducer_output",
        "hqc_verdict",
    ]


def test_exactness_requires_matching_dtype():
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = left.astype(np.float64)

    assert audit._difference(left, right)["exact"] is False


def test_comparison_tampering_is_detected():
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = np.array([1.0, 2.25], dtype=np.float32)
    reported = audit._difference(left, right)
    reported["relative_l2_difference"] = 0.0

    with pytest.raises(
        audit.AdjudicationFailure,
        match="differs from independent recomputation",
    ):
        audit._require_difference_match(
            audit._difference(left, right),
            reported,
            "probe",
        )


def test_comparison_boolean_type_tampering_is_detected():
    value = np.array([1.0, 2.0], dtype=np.float32)
    reported = audit._difference(value, value)
    reported["exact"] = 1

    with pytest.raises(
        audit.AdjudicationFailure,
        match="wrong type",
    ):
        audit._require_difference_match(
            audit._difference(value, value),
            reported,
            "probe",
        )


@pytest.mark.parametrize("malformed", [False, "0"])
def test_comparison_float_type_tampering_is_detected(malformed):
    value = np.array([1.0, 2.0], dtype=np.float32)
    reported = audit._difference(value, value)
    reported["maximum_absolute_difference"] = malformed

    with pytest.raises(
        audit.AdjudicationFailure,
        match="wrong type",
    ):
        audit._require_difference_match(
            audit._difference(value, value),
            reported,
            "probe",
        )
