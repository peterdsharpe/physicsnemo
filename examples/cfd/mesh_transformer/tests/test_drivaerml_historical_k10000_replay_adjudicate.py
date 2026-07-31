# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused adversarial tests for the historical K=10,000 replay reducer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

import drivaerml_historical_k10000_replay as producer
import drivaerml_historical_k10000_replay_adjudicate as audit
import numpy as np
import pytest
import torch


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def _small_schema(
    resolution: int,
) -> dict[str, tuple[tuple[int | None, ...], np.dtype]]:
    return {
        "selected_cell_ids_int64": ((resolution,), np.dtype("<i8")),
        "compacted_cells_int64": ((resolution, 3), np.dtype("<i8")),
        "raw_centroids_float32": ((resolution, 3), np.dtype("<f4")),
        "native_normals_float32": ((resolution, 3), np.dtype("<f4")),
        "native_areas_float64": ((resolution,), np.dtype("<f8")),
        "raw_target_pressure_float32": ((resolution,), np.dtype("<f4")),
        "raw_target_wss_float32": ((resolution, 3), np.dtype("<f4")),
        "pipeline_boundary_points_float32": ((None, 3), np.dtype("<f4")),
        "pipeline_queries_float32": ((resolution, 3), np.dtype("<f4")),
        "pipeline_normals_float32": ((resolution, 3), np.dtype("<f4")),
        "pipeline_globals_float32": (
            (len(audit.GLOBAL_FIELD_ORDER),),
            np.dtype("<f4"),
        ),
        "prediction_pressure_training_float32": (
            (resolution,),
            np.dtype("<f4"),
        ),
        "prediction_wss_training_float32": (
            (resolution, 3),
            np.dtype("<f4"),
        ),
        "truth_pressure_training_float32": ((resolution,), np.dtype("<f4")),
        "truth_wss_training_float32": ((resolution, 3), np.dtype("<f4")),
        "prediction_pressure_physical_float32": (
            (resolution,),
            np.dtype("<f4"),
        ),
        "prediction_wss_physical_float32": (
            (resolution, 3),
            np.dtype("<f4"),
        ),
        "truth_pressure_physical_float32": ((resolution,), np.dtype("<f4")),
        "truth_wss_physical_float32": ((resolution, 3), np.dtype("<f4")),
        "pipeline_center_float32": ((3,), np.dtype("<f4")),
    }


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch) -> tuple[int, str, int, int, int]:
    resolution = 2
    spec = (0, "run_test", 7, 100, 10)
    monkeypatch.setattr(audit, "RESOLUTION", resolution)
    monkeypatch.setattr(audit, "CASE_COUNT", 1)
    monkeypatch.setattr(audit, "CASE_SPECS", (spec,))
    monkeypatch.setattr(audit, "ARRAY_SCHEMAS", _small_schema(resolution))
    return spec


def _case_arrays(
    spec: tuple[int, str, int, int, int],
) -> dict[str, np.ndarray]:
    ordinal, case_id, _, _, start = spec
    prefix = audit._case_prefix(ordinal, case_id)
    resolution = audit.RESOLUTION
    cells = np.array([[0, 1, 2], [0, 2, 1]], dtype="<i8")
    truth_pressure = np.array([1.0, 2.0], dtype="<f4")
    truth_wss = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype="<f4",
    )
    normals = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype="<f4",
    )
    globals_array = np.array(
        [
            1.0,
            2.0,
            0.0,
            3.0,
            4.0,
            1.0e-5,
            5.0,
            1.0 / np.sqrt(5.0),
            2.0 / np.sqrt(5.0),
            0.0,
            8.0,
        ],
        dtype="<f4",
    )
    q_inf = np.float32(10.0)
    p_inf = np.float32(3.0)
    wss_scale = np.float32(np.float32(0.00313) + np.float32(1.0e-8))
    pressure_physical = np.asarray(truth_pressure * q_inf + p_inf, dtype="<f4")
    wss_physical = np.asarray(truth_wss * wss_scale * q_inf, dtype="<f4")
    suffixes = {
        "selected_cell_ids_int64": np.arange(
            start,
            start + resolution,
            dtype="<i8",
        ),
        "compacted_cells_int64": cells,
        "raw_centroids_float32": np.zeros((resolution, 3), dtype="<f4"),
        "native_normals_float32": normals.copy(),
        "native_areas_float64": np.ones(resolution, dtype="<f8"),
        "raw_target_pressure_float32": pressure_physical.copy(),
        "raw_target_wss_float32": wss_physical.copy(),
        "pipeline_boundary_points_float32": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype="<f4",
        ),
        "pipeline_queries_float32": np.zeros((resolution, 3), dtype="<f4"),
        "pipeline_normals_float32": normals.copy(),
        "pipeline_globals_float32": globals_array,
        "prediction_pressure_training_float32": truth_pressure.copy(),
        "prediction_wss_training_float32": truth_wss.copy(),
        "truth_pressure_training_float32": truth_pressure.copy(),
        "truth_wss_training_float32": truth_wss.copy(),
        "prediction_pressure_physical_float32": pressure_physical.copy(),
        "prediction_wss_physical_float32": wss_physical.copy(),
        "truth_pressure_physical_float32": pressure_physical.copy(),
        "truth_wss_physical_float32": wss_physical.copy(),
        "pipeline_center_float32": np.zeros(3, dtype="<f4"),
    }
    assert set(suffixes) == set(audit.ARRAY_SCHEMAS)
    return {
        f"{prefix}__{suffix}": np.ascontiguousarray(value)
        for suffix, value in suffixes.items()
    }


def _normalization_state() -> dict[str, np.ndarray | float]:
    return {
        "wss_mean": np.zeros(3, dtype="<f8"),
        "wss_std": float(np.float32(0.00313)),
    }


def _archived_case(
    arrays: Mapping[str, np.ndarray],
    spec: tuple[int, str, int, int, int],
) -> dict[str, np.ndarray]:
    prefix = audit._case_prefix(spec[0], spec[1])
    return {
        "boundary_points": arrays[f"{prefix}__pipeline_boundary_points_float32"].copy(),
        "cells": arrays[f"{prefix}__compacted_cells_int64"].copy(),
        "normals": arrays[f"{prefix}__pipeline_normals_float32"].copy(),
        "query_points": arrays[f"{prefix}__pipeline_queries_float32"].copy(),
        "pred_pressure": arrays[
            f"{prefix}__prediction_pressure_physical_float32"
        ].copy(),
        "pred_wss": arrays[f"{prefix}__prediction_wss_physical_float32"].copy(),
        "true_pressure": arrays[f"{prefix}__truth_pressure_physical_float32"].copy(),
        "true_wss": arrays[f"{prefix}__truth_wss_physical_float32"].copy(),
        "globals": arrays[f"{prefix}__pipeline_globals_float32"].copy(),
    }


def _target_record(
    arrays: Mapping[str, np.ndarray],
    spec: tuple[int, str, int, int, int],
) -> dict[str, dict[str, str]]:
    prefix = audit._case_prefix(spec[0], spec[1])
    return {
        "pressure": {
            "selected_sha256": audit._array_sha256(
                arrays[f"{prefix}__raw_target_pressure_float32"]
            )
        },
        "wss": {
            "selected_sha256": audit._array_sha256(
                arrays[f"{prefix}__raw_target_wss_float32"]
            )
        },
    }


def _adjudicate_small_case(
    arrays_a: Mapping[str, np.ndarray],
    arrays_b: Mapping[str, np.ndarray],
    archived: Mapping[str, np.ndarray],
    spec: tuple[int, str, int, int, int],
) -> dict[str, Any]:
    return audit._adjudicate_case(
        spec=spec,
        arrays_a=arrays_a,
        arrays_b=arrays_b,
        archived=archived,
        metric_row={"metrics": {"pressure_l2": 0.0}},
        target_record=_target_record(arrays_a, spec),
        normalization_state=_normalization_state(),
    )


def _producer_document(
    arrays: Mapping[str, np.ndarray],
    spec: tuple[int, str, int, int, int],
    *,
    label: str,
    npz_path: Path,
    npz_sha256: str,
) -> dict[str, Any]:
    prefix = audit._case_prefix(spec[0], spec[1])
    case_hashes = {
        suffix: audit._array_sha256(arrays[f"{prefix}__{suffix}"])
        for suffix in audit.ARRAY_SCHEMAS
    }
    manifest = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": audit._array_sha256(value),
        }
        for name, value in arrays.items()
    }
    target = _target_record(arrays, spec)
    return {
        "schema_version": 1,
        "artifact_kind": "phase1_historical_k10000_replay_producer",
        "status": "COMPLETED_HISTORICAL_K10000_REPLAY_PRODUCER",
        "generated_at_utc": "2026-07-28T12:00:00+00:00",
        "replay_label": label,
        "contract": {
            "legacy_call": "model(domain)",
            "canonical_source_geometry_present": False,
            "producer_reads_archive_or_metrics": False,
            "candidate_or_canonical_arm_present": False,
            "resolution": audit.RESOLUTION,
            "precision": "bfloat16",
            "case_count": audit.CASE_COUNT,
            "global_field_order": list(audit.GLOBAL_FIELD_ORDER),
            "measure_weights_required_absent": True,
        },
        "summary": {
            "case_count": audit.CASE_COUNT,
            "array_count": audit.CASE_COUNT * audit.ARRAYS_PER_CASE,
            "measure_weights_absent_case_count": audit.CASE_COUNT,
        },
        "cases": [
            {
                "cohort_ordinal": spec[0],
                "case_id": spec[1],
                "reader_index": spec[2],
                "n_master_cells": spec[3],
                "historical_start": spec[4],
                "resolution": audit.RESOLUTION,
                "n_compacted_points": 3,
                "measure_weights_absent": True,
                "target_input_verification": {
                    "pressure_selected_sha256": target["pressure"]["selected_sha256"],
                    "wss_selected_sha256": target["wss"]["selected_sha256"],
                },
                "array_sha256": case_hashes,
            }
        ],
        "npz": {
            "filename": npz_path.name,
            "sha256": npz_sha256,
            "array_count": audit.CASE_COUNT * audit.ARRAYS_PER_CASE,
            "array_manifest": manifest,
        },
        "provenance": {
            "command": ["producer.py"],
            "producer_path": "/task/producer.py",
            "producer_sha256": audit.EXPECTED_PRODUCER_SHA256,
            "helper_path": "/task/helper.py",
            "helper_sha256": audit.EXPECTED_HELPER_SHA256,
            "repo_root": "/task/source",
            "dataset_root": "/scratch/dataset",
            "static_inputs": {
                "source_tree": audit.EXPECTED_CURRENT_SOURCE_TREE_SHA256,
                "model_source": audit.EXPECTED_CURRENT_MODEL_SOURCE_SHA256,
                "checkpoint": audit.EXPECTED_MODEL_CHECKPOINT_SHA256,
            },
            "geometry_verification": {
                "manifest_sha256": audit.EXPECTED_GEOMETRY_MANIFEST_SHA256,
            },
            "target_input_verification": {
                "manifest_sha256": audit.EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
            },
            "historical_input_freeze_sha256": (
                audit.EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256
            ),
            "import_provenance": {},
            "python": "3.13",
            "numpy": np.__version__,
            "torch": "test",
            "cuda_runtime": "test",
            "device_name": "test",
        },
    }


def test_array_schema_has_exactly_twenty_required_suffixes() -> None:
    assert len(audit.ARRAY_SCHEMAS) == audit.ARRAYS_PER_CASE == 20
    assert "raw_target_pressure_float32" in audit.ARRAY_SCHEMAS
    assert "raw_target_wss_float32" in audit.ARRAY_SCHEMAS
    assert "pipeline_boundary_points_float32" in audit.ARRAY_SCHEMAS
    assert "pipeline_globals_float32" in audit.ARRAY_SCHEMAS


def test_reducer_contract_constants_match_current_producer() -> None:
    assert (
        hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        == audit.EXPECTED_PRODUCER_SHA256
    )
    assert tuple(producer.EXPECTED_CASE_SPECS) == audit.CASE_SPECS
    assert tuple(producer.GLOBAL_FIELD_ORDER) == audit.GLOBAL_FIELD_ORDER
    assert (
        producer.EXPECTED_TARGET_INPUT_MANIFEST_SHA256
        == audit.EXPECTED_TARGET_INPUT_MANIFEST_SHA256
    )
    assert (
        producer.EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256
        == audit.EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256
    )
    assert (
        producer.EXPECTED_GEOMETRY_MANIFEST_SHA256
        == audit.EXPECTED_GEOMETRY_MANIFEST_SHA256
    )
    assert producer.EXPECTED_HELPER_SHA256 == audit.EXPECTED_HELPER_SHA256
    assert (
        producer.EXPECTED_EXECUTION_SOURCE_TREE_SHA256
        == audit.EXPECTED_CURRENT_SOURCE_TREE_SHA256
    )
    assert (
        producer.EXPECTED_CURRENT_MODEL_SOURCE_SHA256
        == audit.EXPECTED_CURRENT_MODEL_SOURCE_SHA256
    )
    assert producer.EXPECTED_MODEL_SHA256 == audit.EXPECTED_MODEL_CHECKPOINT_SHA256
    assert (
        producer.EXPECTED_NORMALIZATION_SHA256
        == audit.EXPECTED_NORMALIZATION_STATE_SHA256
    )
    assert audit.TRAINING_PHYSICAL_ABS_TOLERANCE == 2.0e-6
    assert audit.TRAINING_PHYSICAL_REL_TOLERANCE == 0.0
    assert audit.PIPELINE_NORMAL_ABS_TOLERANCE == 2.0e-6


def test_decision_gate_names_only_model_consumed_archive_fields() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert '"model_visible_archive_inputs_exact"' not in source
    assert '"model_consumed_archive_fields_parity_passed"' in source
    assert '"archive_derived_target_measure_weights_excluded": True' in source


def test_cli_and_adjudicate_signature_are_aligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Path] = {}

    def fake_adjudicate(**kwargs: Path) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "status": audit.INCOMPLETE_STATUS,
            "decision_outcome": audit.INCOMPLETE_REPLAY,
        }

    monkeypatch.setattr(audit, "adjudicate", fake_adjudicate)
    monkeypatch.setattr(audit, "_atomic_publish", lambda path, payload: "0" * 64)
    values = {
        "producer": tmp_path / "producer.py",
        "producer-a-json": tmp_path / "a.json",
        "producer-a-npz": tmp_path / "a.npz",
        "producer-b-json": tmp_path / "b.json",
        "producer-b-npz": tmp_path / "b.npz",
        "target-input-manifest": tmp_path / "targets.json",
        "historical-predictions": tmp_path / "historical",
        "historical-manifest": tmp_path / "manifest.sha256",
        "historical-metrics": tmp_path / "metrics.jsonl",
        "normalization-state": tmp_path / "norm_stats.pt",
        "output-json": tmp_path / "adjudication.json",
    }
    argv = [
        item for name, value in values.items() for item in (f"--{name}", str(value))
    ]

    audit.main(argv)

    assert set(observed) == {
        "producer_path",
        "producer_a_json",
        "producer_a_npz",
        "producer_b_json",
        "producer_b_npz",
        "target_input_manifest",
        "historical_predictions",
        "historical_manifest",
        "historical_metrics",
        "normalization_state",
    }


def test_array_exact_and_difference_are_signed_zero_sensitive() -> None:
    positive = np.array([0.0, 1.0], dtype="<f4")
    negative = np.array([-0.0, 1.0], dtype="<f4")
    assert audit._array_exact(positive, negative) is False
    difference = audit._byte_difference(positive, negative)
    assert difference["exact"] is False
    assert difference["differing_elements_including_signed_zero"] == 1
    assert difference["maximum_absolute_difference"] == 0.0


def test_verified_loader_never_reopens_payload_after_sidecar_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lane.json"
    original = b'{"version":1}\n'
    changed = b'{"version":2}\n'
    path.write_bytes(original)
    _write_sidecar(path)
    real_read = audit._read_regular_file_bytes
    calls: list[str] = []

    def mutate_after_payload_read(
        candidate: Path,
        label: str,
    ) -> tuple[bytes, tuple[int, int]]:
        payload, identity = real_read(candidate, label)
        calls.append(Path(candidate).name)
        if str(label).endswith("sidecar"):
            path.write_bytes(changed)
        return payload, identity

    monkeypatch.setattr(audit, "_read_regular_file_bytes", mutate_after_payload_read)
    payload, digest = audit._load_verified_artifact_bytes(path, "lane")

    assert payload == original
    assert digest == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == changed
    assert calls == ["lane.json", "lane.json.sha256"]


def test_archived_array_uses_only_manifest_bound_tree_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.array([0.0, -0.0, 2.0], dtype="<f4")
    root = "./case/tensor"
    tree = {f"{root}/values.memmap": values.tobytes()}
    metadata = {
        "values": {
            "shape": [3],
            "dtype": "torch.float32",
            "device": "cpu",
        }
    }
    monkeypatch.setattr(
        audit,
        "_read_regular_file_bytes",
        lambda *args, **kwargs: pytest.fail("archive bytes were reopened"),
    )

    loaded = audit._archived_array(
        tree,
        tensor_root=root,
        metadata=metadata,
        field="values",
        relative="values.memmap",
        shape=(3,),
        dtype_name="torch.float32",
    )

    assert loaded.tobytes() == values.tobytes()


def test_normalization_state_is_hash_bound_and_loaded_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "norm_stats.pt"
    torch.save(
        {
            "wss": {
                "type": "vector",
                "mean": torch.zeros(3, dtype=torch.float32),
                "std": torch.tensor(0.00313, dtype=torch.float32),
            }
        },
        path,
    )
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        audit,
        "EXPECTED_NORMALIZATION_STATE_SHA256",
        expected_digest,
    )
    real_load = torch.load
    observed: dict[str, Any] = {}

    def checked_load(stream: io.BytesIO, **kwargs: Any) -> Any:
        observed.update(kwargs)
        assert isinstance(stream, io.BytesIO)
        return real_load(stream, **kwargs)

    monkeypatch.setattr(audit.torch, "load", checked_load)

    state, digest = audit._load_normalization_state(path)

    assert observed == {"map_location": "cpu", "weights_only": True}
    assert digest == expected_digest
    assert np.array_equal(state["wss_mean"], np.zeros(3))
    assert state["wss_std"] == float(np.float32(0.00313))


def test_case_schema_rejects_sparse_compaction(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays = _case_arrays(small_contract)
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    arrays[f"{prefix}__compacted_cells_int64"][:] = np.array(
        [[0, 2, 2], [0, 2, 2]],
        dtype="<i8",
    )

    with pytest.raises(audit.ArtifactInvalid, match="not dense"):
        audit._validate_case_array_schema(
            arrays,
            spec=small_contract,
            n_compacted_points=3,
        )


def test_producer_a_b_arrays_must_be_bitwise_exact(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays_a = _case_arrays(small_contract)
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    archived = _archived_case(arrays_a, small_contract)
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    arrays_b[f"{prefix}__pipeline_center_float32"][0] = -0.0

    case = _adjudicate_small_case(
        arrays_a,
        arrays_b,
        archived,
        small_contract,
    )
    status, outcome = audit._classify_complete([case])

    assert case["replicas_exact"] is False
    assert case["replica_mismatch_arrays"] == ["pipeline_center_float32"]
    assert status == audit.INVALID_STATUS
    assert outcome == audit.INVALID_REPLAY


def test_truth_mismatch_is_invalid_not_refutation(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays_a = _case_arrays(small_contract)
    archived = _archived_case(arrays_a, small_contract)
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    arrays_a[f"{prefix}__truth_pressure_physical_float32"][0] += 1.0
    arrays_b[f"{prefix}__truth_pressure_physical_float32"][0] += 1.0

    case = _adjudicate_small_case(
        arrays_a,
        arrays_b,
        archived,
        small_contract,
    )
    status, outcome = audit._classify_complete([case])

    assert case["input_parity"]["truth_chain_control_exact"] is False
    assert case["historical_predictions_exact"] is True
    assert status == audit.INVALID_STATUS
    assert outcome == audit.INVALID_REPLAY


def test_archive_derived_weights_ignore_wrong_producer_areas(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays = _case_arrays(small_contract)
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    arrays[f"{prefix}__prediction_pressure_training_float32"][0] = 2.0
    arrays[f"{prefix}__prediction_pressure_physical_float32"][0] = 23.0
    archived = _archived_case(arrays, small_contract)
    control = _adjudicate_small_case(arrays, arrays, archived, small_contract)
    wrong_a = {name: value.copy() for name, value in arrays.items()}
    wrong_b = {name: value.copy() for name, value in arrays.items()}
    for replica in (wrong_a, wrong_b):
        replica[f"{prefix}__native_areas_float64"][:] = [1.0, 9.0]
        replica[f"{prefix}__native_normals_float32"][0] = [1.0, 0.0, 0.0]

    changed = _adjudicate_small_case(
        wrong_a,
        wrong_b,
        archived,
        small_contract,
    )

    assert changed["input_parity"]["passed"] is True
    diagnostic = changed["input_parity"]["producer_geometry_diagnostics"]
    assert diagnostic["native_normalized_weights_vs_archive_derived"][
        "maximum_absolute_difference"
    ] == pytest.approx(0.4)
    assert diagnostic["native_normals_vs_archive_derived"]["passed"] is False
    assert diagnostic["deciding"] is False
    assert (
        changed["corrected_baseline_metrics"] == control["corrected_baseline_metrics"]
    )
    corrected_name = "archive_normalized_area_weighted_pressure_relative_l2"
    producer_weighted = audit._weighted_relative_l2(
        arrays[f"{prefix}__prediction_pressure_training_float32"],
        arrays[f"{prefix}__truth_pressure_training_float32"],
        np.array([0.1, 0.9]),
    )
    assert changed["corrected_baseline_metrics"][corrected_name] != pytest.approx(
        producer_weighted
    )
    zero_a = {name: value.copy() for name, value in arrays.items()}
    zero_b = {name: value.copy() for name, value in arrays.items()}
    for replica in (zero_a, zero_b):
        replica[f"{prefix}__native_areas_float64"][:] = 0.0
    audit._validate_case_array_schema(
        zero_a,
        spec=small_contract,
        n_compacted_points=3,
    )
    zero_case = _adjudicate_small_case(
        zero_a,
        zero_b,
        archived,
        small_contract,
    )
    zero_diagnostic = zero_case["input_parity"]["producer_geometry_diagnostics"][
        "native_normalized_weights_vs_archive_derived"
    ]
    assert zero_diagnostic["computable"] is False
    assert zero_case["input_parity"]["passed"] is True
    assert (
        zero_case["corrected_baseline_metrics"] == control["corrected_baseline_metrics"]
    )


def test_degenerate_archived_triangle_is_invalid(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays = _case_arrays(small_contract)
    archived = _archived_case(arrays, small_contract)
    archived["boundary_points"][2] = [2.0, 0.0, 0.0]

    with pytest.raises(audit.ArtifactInvalid, match="degenerate triangle"):
        _adjudicate_small_case(arrays, arrays, archived, small_contract)


def test_archived_stored_normal_mismatch_is_invalid(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays = _case_arrays(small_contract)
    archived = _archived_case(arrays, small_contract)
    archived["normals"][0] *= -1.0

    case = _adjudicate_small_case(arrays, arrays, archived, small_contract)
    status, outcome = audit._classify_complete([case])

    geometry = case["input_parity"]["archive_geometry_control"]
    assert geometry["stored_normals_vs_derived"]["passed"] is False
    assert case["input_parity"]["passed"] is False
    assert status == audit.INVALID_STATUS
    assert outcome == audit.INVALID_REPLAY


def test_serialized_training_physical_mixup_is_invalid(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays_a = _case_arrays(small_contract)
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    prediction_name = f"{prefix}__prediction_pressure_training_float32"
    truth_name = f"{prefix}__truth_pressure_training_float32"
    prediction_physical = f"{prefix}__prediction_pressure_physical_float32"
    arrays_a[prediction_name][:] = [3.0, 4.0]
    arrays_a[prediction_physical][:] = arrays_a[prediction_name] * np.float32(
        10.0
    ) + np.float32(3.0)
    archived = _archived_case(arrays_a, small_contract)
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    for replica in (arrays_a, arrays_b):
        serialized_prediction = replica[prediction_name].copy()
        replica[prediction_name][:] = replica[truth_name]
        replica[truth_name][:] = serialized_prediction

    case = _adjudicate_small_case(
        arrays_a,
        arrays_b,
        archived,
        small_contract,
    )
    status, outcome = audit._classify_complete([case])

    chain = case["input_parity"]["training_physical_chain_control"]
    assert chain["comparisons"]["prediction_pressure"]["passed"] is False
    assert chain["comparisons"]["truth_pressure"]["passed"] is False
    assert chain["passed"] is False
    assert case["corrected_baseline_metrics"] is None
    assert status == audit.INVALID_STATUS
    assert outcome == audit.INVALID_REPLAY


def test_prediction_mismatch_is_valid_refutation(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays_a = _case_arrays(small_contract)
    archived = _archived_case(arrays_a, small_contract)
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    for arrays in (arrays_a, arrays_b):
        training = arrays[f"{prefix}__prediction_wss_training_float32"]
        physical = arrays[f"{prefix}__prediction_wss_physical_float32"]
        training[0, 0] += 1.0
        physical[0, 0] = np.float32(
            training[0, 0]
            * np.float32(np.float32(0.00313) + np.float32(1.0e-8))
            * np.float32(10.0)
        )

    case = _adjudicate_small_case(
        arrays_a,
        arrays_b,
        archived,
        small_contract,
    )
    status, outcome = audit._classify_complete([case])

    assert case["input_parity"]["passed"] is True
    assert case["historical_predictions_exact"] is False
    assert status == audit.VALID_STATUS
    assert outcome == audit.VALID_REFUTATION


def test_exact_case_and_metric_pass_truth_table(
    small_contract: tuple[int, str, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _case_arrays(small_contract)
    archived = _archived_case(arrays, small_contract)
    case = _adjudicate_small_case(arrays, arrays, archived, small_contract)
    monkeypatch.setattr(audit, "ARCHIVED_PRESSURE_MEAN", 0.0)

    status, outcome = audit._classify_complete([case])

    assert status == audit.VALID_STATUS
    assert outcome == audit.EXACT_OUTCOME


def test_signed_zero_query_mismatch_is_invalid(
    small_contract: tuple[int, str, int, int, int],
) -> None:
    arrays_a = _case_arrays(small_contract)
    archived = _archived_case(arrays_a, small_contract)
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    prefix = audit._case_prefix(small_contract[0], small_contract[1])
    for arrays in (arrays_a, arrays_b):
        arrays[f"{prefix}__pipeline_queries_float32"][0, 0] = -0.0
    case = _adjudicate_small_case(
        arrays_a,
        arrays_b,
        archived,
        small_contract,
    )
    status, outcome = audit._classify_complete([case])

    assert case["input_parity"]["pipeline_queries"]["exact"] is False
    assert case["input_parity"]["query_coordinate_max_abs"] == 0.0
    assert status == audit.INVALID_STATUS
    assert outcome == audit.INVALID_REPLAY


def test_load_producer_enforces_schema_hashes_and_target_binding(
    tmp_path: Path,
    small_contract: tuple[int, str, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_PRODUCER_SHA256", "a" * 64)
    arrays = _case_arrays(small_contract)
    npz_path = tmp_path / "lane-a.npz"
    np.savez(npz_path, **arrays)
    _write_sidecar(npz_path)
    document = _producer_document(
        arrays,
        small_contract,
        label="A",
        npz_path=npz_path,
        npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    )
    json_path = tmp_path / "lane-a.json"
    json_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(json_path)

    provenance, loaded = audit._load_producer(
        json_path,
        npz_path,
        "A",
        {small_contract[1]: _target_record(arrays, small_contract)},
    )

    assert set(loaded) == set(arrays)
    assert provenance["npz_sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()


def test_producer_cannot_smuggle_outcome(
    tmp_path: Path,
    small_contract: tuple[int, str, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_PRODUCER_SHA256", "a" * 64)
    arrays = _case_arrays(small_contract)
    npz_path = tmp_path / "lane-a.npz"
    np.savez(npz_path, **arrays)
    document = _producer_document(
        arrays,
        small_contract,
        label="A",
        npz_path=npz_path,
        npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    )
    document["summary"]["preliminary_outcome"] = "PASS"

    with pytest.raises(audit.ArtifactInvalid, match="forbidden conclusion"):
        audit._validate_producer_document(
            document,
            expected_label="A",
            npz_path=npz_path,
            npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            arrays=arrays,
            target_records={small_contract[1]: _target_record(arrays, small_contract)},
        )


def test_duplicate_npz_member_is_invalid() -> None:
    stream = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("x.npy", b"first")
            archive.writestr("x.npy", b"second")

    with pytest.raises(audit.ArtifactInvalid, match="duplicate ZIP"):
        audit._load_npz_bytes(stream.getvalue(), "duplicate", ("x",))


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_outcome"),
    [
        (
            audit.ArtifactUnavailable("missing lane"),
            audit.INCOMPLETE_STATUS,
            audit.INCOMPLETE_REPLAY,
        ),
        (
            audit.ArtifactInvalid("bad lane schema"),
            audit.INVALID_STATUS,
            audit.INVALID_REPLAY,
        ),
    ],
)
def test_missing_or_corrupt_lane_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_status: str,
    expected_outcome: str,
) -> None:
    producer = tmp_path / "producer.py"
    producer.write_bytes(b"producer")
    monkeypatch.setattr(
        audit,
        "EXPECTED_PRODUCER_SHA256",
        hashlib.sha256(b"producer").hexdigest(),
    )
    monkeypatch.setattr(audit, "_load_target_manifest", lambda path: ({}, "b" * 64))
    monkeypatch.setattr(
        audit,
        "_load_normalization_state",
        lambda path: (_normalization_state(), "c" * 64),
    )
    monkeypatch.setattr(
        audit,
        "_load_producer",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    result = audit.adjudicate(
        producer_path=producer,
        producer_a_json=tmp_path / "a.json",
        producer_a_npz=tmp_path / "a.npz",
        producer_b_json=tmp_path / "b.json",
        producer_b_npz=tmp_path / "b.npz",
        target_input_manifest=tmp_path / "targets.json",
        historical_predictions=tmp_path / "historical",
        historical_manifest=tmp_path / "manifest",
        historical_metrics=tmp_path / "metrics",
        normalization_state=tmp_path / "norm_stats.pt",
    )

    assert result["status"] == expected_status
    assert result["decision_outcome"] == expected_outcome


def test_historical_metrics_are_parsed_from_one_frozen_payload(
    small_contract: tuple[int, str, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = small_contract[1]
    payload = (
        json.dumps(
            {
                "phase": "infer_step",
                "sample_id": f"00007_{case_id}_domain_{case_id}",
                "metrics": {"pressure_l2": 0.25},
            }
        )
        + "\n"
        + json.dumps(
            {
                "phase": "infer_summary",
                "metrics": {"pressure_l2": 0.25},
            }
        )
        + "\n"
    ).encode()
    monkeypatch.setattr(
        audit,
        "EXPECTED_HISTORICAL_METRICS_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(audit, "ARCHIVED_PRESSURE_MEAN", 0.25)
    monkeypatch.setattr(
        audit,
        "_read_regular_file_bytes",
        lambda *args, **kwargs: pytest.fail("metrics payload was reopened"),
    )

    rows = audit._historical_metric_rows(payload)

    assert list(rows) == [case_id]
    assert rows[case_id]["metrics"]["pressure_l2"] == 0.25


def test_atomic_publish_rolls_back_and_removes_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adjudication.json"
    real_link = os.link
    calls = 0

    def fail_second_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated sidecar publication failure")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(audit.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated sidecar"):
        audit._atomic_publish(output, b'{"status":"test"}\n')

    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_publish_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "adjudication.json"
    output.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit._atomic_publish(output, b"replacement")

    assert output.read_text(encoding="utf-8") == "sentinel"
