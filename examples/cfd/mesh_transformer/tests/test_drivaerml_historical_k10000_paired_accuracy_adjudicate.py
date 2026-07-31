# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused adversarial tests for the paired canonical accuracy reducer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import drivaerml_historical_k10000_canonical_arm as producer
import drivaerml_historical_k10000_paired_accuracy_adjudicate as audit
import numpy as np
import pytest


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def test_reducer_contract_matches_frozen_canonical_producer() -> None:
    producer_path = Path(producer.__file__)
    assert (
        hashlib.sha256(producer_path.read_bytes()).hexdigest()
        == audit.EXPECTED_CANONICAL_PRODUCER_SHA256
    )
    assert tuple(producer.PAIRING_CONTROL_SUFFIXES) == audit.PAIRING_CONTROL_SUFFIXES
    assert (
        tuple(producer.CANONICAL_GEOMETRY_SUFFIXES) == audit.CANONICAL_GEOMETRY_SUFFIXES
    )
    assert tuple(producer.PREDICTION_SUFFIXES) == audit.PREDICTION_SUFFIXES
    assert tuple(producer.CASE_ARRAY_SUFFIXES) == audit.CANONICAL_ARRAY_SUFFIXES
    assert len(audit.CANONICAL_ARRAY_SUFFIXES) == audit.ARRAYS_PER_CANONICAL_CASE == 22
    assert audit.CASE_COUNT * audit.ARRAYS_PER_CANONICAL_CASE == 792
    assert audit.CASE_COUNT * audit.ARRAYS_PER_LEGACY_CASE == 720


def test_valid_canonical_contract_audit_key_is_not_a_conclusion() -> None:
    audit._reject_canonical_producer_conclusions(
        {
            "contract": {
                "producer_reads_supervision_archive_metrics_or_ceilings": False,
            }
        }
    )
    with pytest.raises(audit.ArtifactInvalid, match="forbidden conclusion key"):
        audit._reject_canonical_producer_conclusions(
            {"contract": {"candidate_metrics": {"pressure": 0.1}}}
        )


def test_inclusive_ceiling_equality_passes_and_one_ulp_above_refutes() -> None:
    equal = dict(audit.FROZEN_CEILINGS)
    outcome, gates = audit._classify_canonical_means(equal)
    assert outcome == audit.NONINFERIORITY_SUCCESS_OUTCOME
    assert all(record["passed"] for record in gates.values())

    name = "uniform_pressure_relative_l2"
    above = dict(equal)
    above[name] = float(np.nextafter(equal[name], np.inf))
    outcome, gates = audit._classify_canonical_means(above)
    assert outcome == audit.VALID_REFUTATION
    assert not gates[name]["passed"]
    assert all(
        record["passed"] for other_name, record in gates.items() if other_name != name
    )


def test_frobenius_wss_is_not_bugged_pointwise_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "RESOLUTION", 2)
    truth = np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype="<f4")
    prediction = np.array([[2.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype="<f4")
    frobenius = audit._relative_l2(prediction, truth)
    pointwise = audit._historical_pointwise_wss(prediction, truth)
    weighted = audit._weighted_relative_l2(
        prediction,
        truth,
        np.array([0.5, 0.5], dtype="<f8"),
    )

    assert frobenius == pytest.approx(1.0 / np.sqrt(10_001.0))
    assert weighted == pytest.approx(frobenius)
    assert pointwise == pytest.approx(0.5)
    assert pointwise > 40.0 * frobenius


def test_expected_digest_rejects_payload_tampering_with_fresh_sidecar(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"value": 1}\n')
    _write_sidecar(artifact)
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert (
        audit._verified_payload(
            artifact,
            "test artifact",
            expected_sha256=expected,
        )[1]
        == expected
    )

    artifact.write_bytes(b'{"value": 2}\n')
    _write_sidecar(artifact)
    with pytest.raises(audit.ArtifactInvalid, match="SHA-256 differs"):
        audit._verified_payload(
            artifact,
            "test artifact",
            expected_sha256=expected,
        )


def test_stale_sidecar_is_incomplete_not_a_valid_refutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"value": 1}\n')
    _write_sidecar(artifact)
    artifact.write_bytes(b'{"value": 2}\n')
    with pytest.raises(audit.ArtifactUnavailable, match="sidecar"):
        audit._verified_payload(artifact, "test artifact")


def test_replica_comparison_is_raw_byte_sensitive_to_signed_zero() -> None:
    left = {"prediction": np.array([0.0, 1.0], dtype="<f4")}
    right = {"prediction": np.array([-0.0, 1.0], dtype="<f4")}
    assert audit._mismatch_names(left, right, ("prediction",)) == ["prediction"]
    right["prediction"][0] = 0.0
    assert audit._mismatch_names(left, right, ("prediction",)) == []


@pytest.fixture
def small_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    tuple[int, str, int, int, int],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    monkeypatch.setattr(audit, "RESOLUTION", 2)
    spec = (0, "run_test", 7, 2, 10)
    monkeypatch.setattr(audit, "CASE_SPECS", (spec,))

    raw_points = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 20.0, 30.0],
            [10.0, 21.0, 30.0],
            [11.0, 21.0, 30.0],
        ],
        dtype="<f4",
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype="<i8")
    raw_centroids = raw_points[cells].mean(axis=1, dtype=np.float32).astype("<f4")
    pipeline_center = np.array([10.5, 20.5, 30.0], dtype="<f4")
    pipeline_points = ((raw_points - pipeline_center) / 5.0).astype("<f4")
    pipeline_queries = ((raw_centroids - pipeline_center) / 5.0).astype("<f4")
    globals_array = np.array(
        [
            2.0,
            0.0,
            0.0,
            0.0,
            2.0,
            1.0e-5,
            5.0,
            1.0,
            0.0,
            0.0,
            8.0,
        ],
        dtype="<f4",
    )
    truth_pressure = np.array([1.0, 2.0], dtype="<f4")
    truth_wss = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype="<f4",
    )
    prediction_pressure = truth_pressure.copy()
    prediction_wss = truth_wss.copy()
    q_inf = 4.0
    wss_scale = audit.WSS_NORMALIZATION_STD + audit.NORMALIZATION_EPSILON

    legacy_suffixes: dict[str, np.ndarray] = {
        "selected_cell_ids_int64": np.array([10, 11], dtype="<i8"),
        "compacted_cells_int64": cells.copy(),
        "raw_centroids_float32": raw_centroids.copy(),
        "native_normals_float32": np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype="<f4"),
            (2, 1),
        ),
        "native_areas_float64": np.array([0.5, 0.5], dtype="<f8"),
        "raw_target_pressure_float32": np.array([4.0, 8.0], dtype="<f4"),
        "raw_target_wss_float32": np.asarray(
            truth_wss * wss_scale * q_inf,
            dtype="<f4",
        ),
        "pipeline_boundary_points_float32": pipeline_points.copy(),
        "pipeline_queries_float32": pipeline_queries.copy(),
        "pipeline_normals_float32": np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype="<f4"),
            (2, 1),
        ),
        "pipeline_globals_float32": globals_array,
        "prediction_pressure_training_float32": prediction_pressure.copy(),
        "prediction_wss_training_float32": prediction_wss.copy(),
        "truth_pressure_training_float32": truth_pressure.copy(),
        "truth_wss_training_float32": truth_wss.copy(),
        "prediction_pressure_physical_float32": np.asarray(
            prediction_pressure * q_inf,
            dtype="<f4",
        ),
        "prediction_wss_physical_float32": np.asarray(
            prediction_wss * wss_scale * q_inf,
            dtype="<f4",
        ),
        "truth_pressure_physical_float32": np.asarray(
            truth_pressure * q_inf,
            dtype="<f4",
        ),
        "truth_wss_physical_float32": np.asarray(
            truth_wss * wss_scale * q_inf,
            dtype="<f4",
        ),
        "pipeline_center_float32": pipeline_center.copy(),
    }
    assert set(legacy_suffixes) == set(audit.stage_b.ARRAY_SCHEMAS)

    reconstructed = audit._reconstruct_canonical_geometry(raw_points, cells)
    canonical_suffixes: dict[str, np.ndarray] = {
        suffix: legacy_suffixes[suffix].copy()
        for suffix in audit.LEGACY_SHARED_CONTROL_SUFFIXES
    }
    canonical_suffixes.update(
        {
            "raw_points_float32": raw_points.copy(),
            **{name: value.copy() for name, value in reconstructed.items()},
            "canonical_queries_float32": reconstructed[
                "canonical_centroids_float32"
            ].copy(),
            "prediction_pressure_training_float32": prediction_pressure.copy(),
            "prediction_wss_training_float32": prediction_wss.copy(),
            "prediction_pressure_physical_float32": np.asarray(
                prediction_pressure * q_inf,
                dtype="<f4",
            ),
            "prediction_wss_physical_float32": np.asarray(
                prediction_wss * wss_scale * q_inf,
                dtype="<f4",
            ),
        }
    )
    assert set(canonical_suffixes) == set(audit.CANONICAL_ARRAY_SUFFIXES)
    prefix = audit._case_prefix(spec[0], spec[1])
    legacy = {
        f"{prefix}__{name}": np.ascontiguousarray(value)
        for name, value in legacy_suffixes.items()
    }
    canonical = {
        f"{prefix}__{name}": np.ascontiguousarray(value)
        for name, value in canonical_suffixes.items()
    }
    return spec, canonical, legacy


def test_exact_pairing_and_independent_geometry_reconstruction_pass(
    small_case: tuple[
        tuple[int, str, int, int, int],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    spec, canonical, legacy = small_case
    result = audit._adjudicate_case(
        spec=spec,
        canonical_arrays=canonical,
        stage_b_arrays=legacy,
    )
    controls = result["validity_controls"]
    assert controls["passed"]
    assert controls["shared_precanonical_inputs_raw_byte_exact"]
    assert controls["canonical_geometry_independently_reconstructed_raw_byte_exact"]
    assert controls["canonical_queries_equal_centroids_raw_byte_exact"]
    assert (
        controls["canonical_query_inverse_maximum_absolute_difference"]
        <= audit.CANONICAL_PHYSICAL_INVERSE_ABS_TOLERANCE
    )
    assert result["canonical_metrics"] == result["legacy_metrics"]
    assert result["canonical_minus_legacy_metric_deltas"] == {
        name: 0.0 for name in audit.DECIDING_METRICS
    }
    assert result["per_case_metrics_deciding"] is False
    assert result["casewise_metric_deltas_deciding"] is False


def test_precanonical_input_drift_is_invalidity_not_refutation(
    small_case: tuple[
        tuple[int, str, int, int, int],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    spec, canonical, legacy = small_case
    prefix = audit._case_prefix(spec[0], spec[1])
    canonical[f"{prefix}__pipeline_queries_float32"] = canonical[
        f"{prefix}__pipeline_queries_float32"
    ].copy()
    canonical[f"{prefix}__pipeline_queries_float32"][0, 0] = np.nextafter(
        canonical[f"{prefix}__pipeline_queries_float32"][0, 0],
        np.float32(np.inf),
        dtype=np.float32,
    )
    result = audit._adjudicate_case(
        spec=spec,
        canonical_arrays=canonical,
        stage_b_arrays=legacy,
    )
    controls = result["validity_controls"]
    assert not controls["passed"]
    assert controls["shared_precanonical_input_mismatches"] == [
        "pipeline_queries_float32"
    ]


def test_canonical_query_replica_or_frame_drift_fails_controls(
    small_case: tuple[
        tuple[int, str, int, int, int],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    spec, canonical, legacy = small_case
    prefix = audit._case_prefix(spec[0], spec[1])
    canonical[f"{prefix}__canonical_queries_float32"] = canonical[
        f"{prefix}__canonical_queries_float32"
    ].copy()
    canonical[f"{prefix}__canonical_queries_float32"][0, 0] += np.float32(1.0)
    result = audit._adjudicate_case(
        spec=spec,
        canonical_arrays=canonical,
        stage_b_arrays=legacy,
    )
    assert not result["validity_controls"]["passed"]
    assert not result["validity_controls"][
        "canonical_queries_equal_centroids_raw_byte_exact"
    ]


def test_atomic_publish_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    digest = audit._atomic_publish(output, b'{"outcome": "first"}\n')
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit._atomic_publish(output, b'{"outcome": "second"}\n')
