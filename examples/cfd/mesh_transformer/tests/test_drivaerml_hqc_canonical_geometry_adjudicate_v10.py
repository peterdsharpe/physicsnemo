# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for independent canonical-geometry adjudication."""

import hashlib
import json
import os
import struct
import zipfile
from pathlib import Path

import drivaerml_hqc_canonical_geometry_adjudicate_v10 as audit
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
                    "construction": audit.EXPECTED_CANONICAL_FRAME_CONSTRUCTION,
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

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    diagnostic_npz = artifacts / f"{audit.EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.npz"
    prior_npz = tmp_path / audit.EXPECTED_PRIOR_NPZ_BASENAME
    arrays = {name: arrays[name] for name in audit._expected_array_key_order(cases)}
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
        "EXPECTED_BUNDLE_MEMBER_SHA256_BY_RELATIVE_PATH",
        {},
    )
    summary = {
        "schema_version": audit.DIAGNOSTIC_SCHEMA_VERSION,
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
            "command": list(audit.EXPECTED_COMMAND),
            "diagnostic_script_path": audit.EXPECTED_DIAGNOSTIC_SCRIPT_PATH,
            "diagnostic_script_sha256": (audit.EXPECTED_DIAGNOSTIC_SCRIPT_SHA256),
            "frozen_producer_path": audit.EXPECTED_FROZEN_PRODUCER_PATH,
            "frozen_producer_sha256": audit.EXPECTED_PRODUCER_SHA256,
            "source_tree_manifest_sha256": audit.EXPECTED_SOURCE_TREE_SHA256,
            "selected_source_files": dict(audit.EXPECTED_SELECTED_SOURCE_FILES),
            "input_hashes": expected_inputs,
            "npz_path": audit.EXPECTED_OUTPUT_NPZ_PATH,
            "npz_sha256": _sha256_file(diagnostic_npz),
            "slurm_job_id": "123",
            "python": "3.13",
            "platform": "test",
            "numpy": np.__version__,
            "torch": "test",
            "hardware": {
                "cuda_runtime": "test",
                "cuda_device_name": "test device",
                "cuda_device_capability": [9, 0],
            },
        },
    }
    diagnostic_json = artifacts / f"{audit.EXPECTED_DIAGNOSTIC_OUTPUT_BASENAME}.json"
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
    log = tmp_path / "sbatch_logs" / "mt-hqc-canon-v5_123.log"
    log.parent.mkdir()
    log.write_text(
        "synthetic successful task\n"
        "COMPLETED_UNITS=1/1 rc=0\n"
        "EXIT_CODE=0\n"
        "HQC_NEUTRAL_CANONICAL_GEOMETRY_V5_DONE\n",
        encoding="ascii",
    )
    dynamic_paths = audit._expected_bundle_relative_paths("123") - set(
        audit.STATIC_ALLOWED_BUNDLE_RELATIVE_PATHS
    )
    observed_paths = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        audit,
        "STATIC_ALLOWED_BUNDLE_RELATIVE_PATHS",
        frozenset(observed_paths - dynamic_paths),
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


def _rewrite_first_zipinfo_metadata(
    path: Path,
    *,
    comment: bytes = b"",
    extra: bytes = b"",
) -> None:
    rewritten = path.with_name(f"{path.name}.rewrite")
    with (
        zipfile.ZipFile(path, mode="r") as source,
        zipfile.ZipFile(
            rewritten,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as destination,
    ):
        for index, info in enumerate(source.infolist()):
            copied = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            copied.compress_type = info.compress_type
            copied.create_system = info.create_system
            copied.external_attr = info.external_attr
            copied.internal_attr = info.internal_attr
            copied.comment = comment if index == 0 else info.comment
            copied.extra = extra if index == 0 else info.extra
            destination.writestr(copied, source.read(info))
    rewritten.replace(path)


def _rewrite_first_npy_payload(path: Path, transform) -> None:
    rewritten = path.with_name(f"{path.name}.rewrite")
    with (
        zipfile.ZipFile(path, mode="r") as source,
        zipfile.ZipFile(
            rewritten,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as destination,
    ):
        for index, info in enumerate(source.infolist()):
            payload = source.read(info)
            if index == 0:
                payload = transform(payload)
            with destination.open(
                info.filename,
                mode="w",
                force_zip64=True,
            ) as member:
                member.write(payload)
    rewritten.replace(path)


def _insert_npy_header_comment(payload: bytes) -> bytes:
    major = payload[6]
    if major == 1:
        header_length = struct.unpack_from("<H", payload, 8)[0]
        header_start = 10
    else:
        header_length = struct.unpack_from("<I", payload, 8)[0]
        header_start = 12
    header_end = header_start + header_length
    closing_brace = payload.rfind(b"}", header_start, header_end)
    hidden = b" # target_pressure=[1,2,3]"
    assert closing_brace >= header_start
    assert payload[closing_brace + 1 : header_end - 1].isspace()
    assert len(hidden) <= header_end - 1 - (closing_brace + 1)
    modified = bytearray(payload)
    modified[closing_brace + 1 : closing_brace + 1 + len(hidden)] = hidden
    return bytes(modified)


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


def test_array_manifest_shape_type_coercion_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    key = "case_00_run_118__selected_cell_ids_int64"
    summary["npz_array_manifest"][key]["shape"] = [2500.0]
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="array-manifest shape has the wrong type",
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


def test_npz_archive_comment_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with zipfile.ZipFile(diagnostic_npz, mode="a") as archive:
        archive.comment = b'{"target_pressure":[1,2,3]}'
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="NPZ has an archive comment",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("comment", "extra"),
    [
        (b'{"target_pressure":[1,2,3]}', b""),
        (b"", struct.pack("<HH3s", 0xCAFE, 3, b"xyz")),
    ],
)
def test_npz_member_metadata_fails_closed(
    tmp_path,
    monkeypatch,
    comment,
    extra,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    _rewrite_first_zipinfo_metadata(
        diagnostic_npz,
        comment=comment,
        extra=extra,
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="NPZ member has comment or central extra metadata",
    ):
        _adjudicate(paths)


def test_npz_trailing_payload_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with diagnostic_npz.open("ab") as stream:
        stream.write(b'{"target_pressure":[1,2,3]}')
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="NPZ has trailing bytes",
    ):
        _adjudicate(paths)


def test_npz_member_order_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with np.load(diagnostic_npz, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True) for name in reversed(archive.files)
        }
    np.savez(diagnostic_npz, **arrays)
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="NPZ member order differs",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    "transform",
    [
        _insert_npy_header_comment,
        lambda payload: payload + b'{"target_trailing_member":[1,2,3]}',
    ],
    ids=["header-comment", "member-tail"],
)
def test_npz_inner_member_payload_fails_closed(
    tmp_path,
    monkeypatch,
    transform,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    _rewrite_first_npy_payload(diagnostic_npz, transform)
    expected_names = audit._expected_array_key_order(
        json.loads(diagnostic_json.read_text())["cases"]
    )
    assert audit._validate_canonical_npz_container(
        diagnostic_npz,
        expected_names=expected_names,
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="NPZ member bytes differ from canonical np.savez",
    ):
        _adjudicate(paths)


def test_duplicate_json_object_name_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, _, _ = paths
    payload = diagnostic_json.read_text(encoding="utf-8")
    diagnostic_json.write_text(
        '{"scientific_scope":{"target_payload":[1,2,3]},' + payload[1:],
        encoding="utf-8",
    )
    _write_sidecar(diagnostic_json)
    _write_bundle_manifest(manifest.parent)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="duplicate object name: scientific_scope",
    ):
        _adjudicate(paths)


def test_nonfinite_json_token_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, _, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    summary["generated_at_utc"] = float("nan")
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(diagnostic_json)
    _write_bundle_manifest(manifest.parent)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="non-finite numeric token: NaN",
    ):
        _adjudicate(paths)


def test_underflowing_nonzero_json_float_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, _, _ = paths
    payload = diagnostic_json.read_text(encoding="utf-8")
    payload = payload.replace(
        '"maximum_unit_deviation": 0.0',
        '"maximum_unit_deviation": 1e-9999',
        1,
    )
    diagnostic_json.write_text(payload, encoding="utf-8")
    _write_sidecar(diagnostic_json)
    _write_bundle_manifest(manifest.parent)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="JSON float underflows binary64",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("generated_at_utc", "not an ISO-8601 timestamp"),
        ("command", "Frozen diagnostic command differs"),
        ("hardware", "Hardware provenance is not an object"),
        ("canonical_construction", "canonical construction label differs"),
    ],
)
def test_unbound_json_provenance_value_fails_closed(
    tmp_path,
    monkeypatch,
    location,
    message,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    if location == "generated_at_utc":
        summary["generated_at_utc"] = "target_pressure=[1,2,3]"
    elif location == "command":
        summary["provenance"]["command"].append("target_pressure=[1,2,3]")
    elif location == "hardware":
        summary["provenance"]["hardware"] = [1.0, 2.0, 3.0]
    else:
        summary["cases"][0]["canonical_frame"]["construction"] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(audit.AdjudicationFailure, match=message):
        _adjudicate(paths)


def test_noncanonical_json_bytes_fail_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, _, _ = paths
    diagnostic_json.write_text(
        diagnostic_json.read_text(encoding="utf-8") + " \n",
        encoding="utf-8",
    )
    _write_sidecar(diagnostic_json)
    _write_bundle_manifest(manifest.parent)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="JSON is not canonically serialized",
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


def test_completed_markers_must_remain_at_bundle_root(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (tmp_path / "DONE_123").rename(decoy / "DONE_123")
    (tmp_path / "STATUS_123").rename(decoy / "STATUS_123")
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="DONE marker",
    ):
        _adjudicate(paths)


def test_frozen_member_is_bound_to_root_relative_path(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    frozen = tmp_path / "frozen_member"
    frozen.write_text("frozen", encoding="ascii")
    expected_digest = _sha256_file(frozen)
    relocated = tmp_path / "decoy" / frozen.name
    relocated.parent.mkdir()
    frozen.rename(relocated)
    _write_bundle_manifest(tmp_path)
    monkeypatch.setattr(
        audit,
        "EXPECTED_BUNDLE_MEMBER_SHA256_BY_RELATIVE_PATH",
        {frozen.name: expected_digest},
    )

    with pytest.raises(
        audit.AdjudicationFailure,
        match="Frozen copied-bundle member differs",
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


def test_recorded_extra_target_payload_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    (tmp_path / "unexpected_target_payload.json").write_text(
        '{"target_pressure":[1,2,3]}\n',
        encoding="utf-8",
    )
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="bundle path allowlist differs",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    "payload",
    [
        "target_pressure=[1.0,2.0,3.0]\n",
        "HQC_verdict=eligible\n",
    ],
)
def test_allowlisted_slurm_log_forbidden_payload_fails_closed(
    tmp_path,
    monkeypatch,
    payload,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    log = tmp_path / "sbatch_logs" / "mt-hqc-canon-v5_123.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(payload)
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="Slurm log contains forbidden vocabulary",
    ):
        _adjudicate(paths)


def test_allowlisted_slurm_log_completion_marker_is_required(
    tmp_path,
    monkeypatch,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    log = tmp_path / "sbatch_logs" / "mt-hqc-canon-v5_123.log"
    log.write_text("synthetic successful task\n", encoding="utf-8")
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="Slurm log is missing completion marker",
    ):
        _adjudicate(paths)


def test_unrecorded_bundle_symlink_alias_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    (tmp_path / "unrecorded_alias").symlink_to(tmp_path / "DONE_123")

    with pytest.raises(
        audit.AdjudicationFailure,
        match="contains a symlink",
    ):
        _adjudicate(paths)


def test_unrecorded_broken_symlink_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    (tmp_path / "broken_alias").symlink_to(tmp_path / "missing")

    with pytest.raises(
        audit.AdjudicationFailure,
        match="contains a symlink",
    ):
        _adjudicate(paths)


def test_unrecorded_fifo_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    os.mkfifo(tmp_path / "unrecorded_fifo")

    with pytest.raises(
        audit.AdjudicationFailure,
        match="special filesystem entry",
    ):
        _adjudicate(paths)


def test_manifest_symlink_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, prior_npz = paths
    alias = tmp_path / "manifest_alias.sha256"
    alias.symlink_to(manifest)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="manifest is missing or is a symlink",
    ):
        _adjudicate((alias, diagnostic_json, diagnostic_npz, prior_npz))


def test_manifest_symlinked_ancestor_fails_closed(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    paths = _synthetic_artifacts(real, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, prior_npz = paths
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    alias_manifest = alias / manifest.name

    with pytest.raises(
        audit.AdjudicationFailure,
        match="manifest traverses a symlink",
    ):
        _adjudicate(
            (alias_manifest, diagnostic_json, diagnostic_npz, prior_npz),
        )


def test_recorded_symlink_member_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    (tmp_path / "recorded_alias").symlink_to(tmp_path / "DONE_123")
    _write_bundle_manifest(tmp_path)

    with pytest.raises(
        audit.AdjudicationFailure,
        match="member is a symlink",
    ):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("location", "malformed", "message"),
    [
        ("schema", 4.0, "schema version differs"),
        ("scope_resolution", 2500.0, "Scope resolution differs"),
        ("case_resolution", 2500.0, "resolution differs"),
        ("slurm_job_id", 123, "Slurm job ID is missing or malformed"),
    ],
)
def test_numeric_identity_type_coercion_fails_closed(
    tmp_path,
    monkeypatch,
    location,
    malformed,
    message,
):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    summary = json.loads(diagnostic_json.read_text())
    if location == "schema":
        summary["schema_version"] = malformed
    elif location == "scope_resolution":
        summary["scientific_scope"]["resolution"] = malformed
    elif location == "case_resolution":
        summary["cases"][0]["resolution"] = malformed
    else:
        summary["provenance"]["slurm_job_id"] = malformed
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(audit.AdjudicationFailure, match=message):
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
    assert audit._forbidden_paths({"safe_key": "target_pressure=[1,2,3]"}) == [
        "safe_key"
    ]


def test_exactness_requires_matching_dtype():
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = left.astype(np.float64)

    assert audit._difference(left, right)["exact"] is False


def test_exactness_distinguishes_signed_zero():
    positive_zero = np.array([0.0], dtype=np.float32)
    negative_zero = np.array([-0.0], dtype=np.float32)

    difference = audit._difference(positive_zero, negative_zero)

    assert np.array_equal(positive_zero, negative_zero)
    assert difference["nonzero_count"] == 0
    assert difference["relative_l2_difference"] == 0.0
    assert difference["exact"] is False


def test_signed_zero_full_artifact_fails_closed(tmp_path, monkeypatch):
    paths = _synthetic_artifacts(tmp_path, monkeypatch)
    manifest, diagnostic_json, diagnostic_npz, _ = paths
    with np.load(diagnostic_npz, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    keys = {
        "primary": "case_00_run_118__float32_canonical_full_primary_pressure",
        "fixed": "case_00_run_118__float32_canonical_full_fixed_pressure",
        "replay": "case_00_run_118__float32_canonical_full_primary_replay_pressure",
    }
    arrays[keys["primary"]][0] = np.float32(0.0)
    arrays[keys["fixed"]][0] = np.float32(-0.0)
    arrays[keys["replay"]][0] = np.float32(0.0)
    np.savez(diagnostic_npz, **arrays)
    summary = json.loads(diagnostic_json.read_text())
    for key in keys.values():
        summary["npz_array_manifest"][key]["sha256"] = audit._sha256_array(arrays[key])
    diagnostic_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_json_and_manifest(manifest, diagnostic_json, diagnostic_npz)

    with pytest.raises(
        audit.AdjudicationFailure,
        match=r"primary_fixed\.pressure\.exact differs",
    ):
        _adjudicate(paths)


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
