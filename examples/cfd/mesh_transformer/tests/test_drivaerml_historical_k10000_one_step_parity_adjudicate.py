# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused adversarial tests for the one-step parity adjudicator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import drivaerml_historical_k10000_one_step_parity_adjudicate as audit
import numpy as np
import pytest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": audit._array_sha256(value),
        }
        for name, value in arrays.items()
    }


@pytest.fixture
def valid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    resolution = 4
    monkeypatch.setattr(audit, "RESOLUTION", resolution)
    expected_contract = dict(audit.EXPECTED_CONTRACT)
    expected_contract["resolution"] = resolution
    monkeypatch.setattr(audit, "EXPECTED_CONTRACT", expected_contract)
    monkeypatch.setattr(audit, "EXPECTED_PARAMETER_COUNT", 8)

    parameter_names = ["operator.weight", "output.bias"]
    module_names = ["operator", "output"]
    names_payload = json.dumps(
        parameter_names, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    parameter_order_sha256 = hashlib.sha256(names_payload).hexdigest()
    arrays: dict[str, np.ndarray] = {
        "parameter_slice_starts_int64": np.array([0, 4], dtype="<i8"),
        "parameter_slice_stops_int64": np.array([4, 8], dtype="<i8"),
        "parameter_slice_module_indices_int64": np.array([0, 1], dtype="<i8"),
    }

    records: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(audit.REGIMES):
        state_hash = _digest(f"state-{regime}")
        optimizer_hash = _digest(f"optimizer-{regime}")
        learning_rate = 3.0e-3 if regime_index == 0 else 3.0e-7
        for precision_index, precision in enumerate(audit.PRECISIONS):
            for case_index, (ordinal, case_id) in enumerate(audit.CASE_SPECS):
                common_control = {
                    "raw_source_geometry_sha256": _digest(f"geometry-{case_id}"),
                    "global_inputs_sha256": _digest(f"globals-{case_id}"),
                    "initial_parameter_state_sha256": state_hash,
                    "initial_optimizer_state_sha256": optimizer_hash,
                    "rng_state_sha256": _digest("rng-seed42-after-model-restore"),
                    "parameter_order_sha256": parameter_order_sha256,
                    "batch_order_sha256": _digest(f"batch-{case_id}"),
                    "source_measure_weights_present": False,
                    "target_measure_present": True,
                }
                records.append(
                    {
                        "regime": regime,
                        "precision": precision,
                        "case_ordinal": ordinal,
                        "case_id": case_id,
                        "arm_controls": {
                            arm: dict(common_control) for arm in audit.ARMS
                        },
                    }
                )
                selected_ids = np.arange(
                    100 * (case_index + 1),
                    100 * (case_index + 1) + resolution,
                    dtype="<i8",
                )
                target_pressure = np.linspace(
                    0.25 + case_index,
                    1.0 + case_index,
                    resolution,
                    dtype="<f4",
                )
                target_wss = np.stack(
                    (
                        target_pressure,
                        0.5 * target_pressure,
                        -0.25 * target_pressure,
                    ),
                    axis=-1,
                ).astype("<f4")
                target_measure = np.linspace(1.0, 2.0, resolution, dtype="<f4")
                # Four deliberately distinct ordinary case gradients give a
                # nonzero between-case calibration scale.
                legacy_gradient = (
                    np.arange(1, 9, dtype=np.float32)
                    + np.float32(case_index * 0.75)
                    + np.float32(regime_index * 0.125)
                    + np.float32(precision_index * 0.0625)
                ).astype("<f4")
                legacy_update = (-learning_rate * legacy_gradient).astype("<f4")
                for arm in audit.ARMS:
                    prefix = audit._prefix(regime, precision, ordinal, case_id, arm)
                    arrays[f"{prefix}prediction_pressure_float32"] = (
                        target_pressure + np.float32(0.01)
                    ).astype("<f4")
                    arrays[f"{prefix}prediction_wss_float32"] = (
                        target_wss + np.float32(0.01)
                    ).astype("<f4")
                    arrays[f"{prefix}loss_float64"] = np.array([5.0e-5], dtype="<f8")
                    arrays[f"{prefix}gradient_float32"] = legacy_gradient.copy()
                    arrays[f"{prefix}parameter_update_float32"] = legacy_update.copy()
                    arrays[f"{prefix}learning_rate_float64"] = np.array(
                        [learning_rate], dtype="<f8"
                    )
                    arrays[f"{prefix}selected_cell_ids_int64"] = selected_ids.copy()
                    arrays[f"{prefix}target_pressure_float32"] = target_pressure.copy()
                    arrays[f"{prefix}target_wss_float32"] = target_wss.copy()
                    arrays[f"{prefix}target_measure_float32"] = target_measure.copy()

    npz_sha256 = _digest("synthetic-npz")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": audit.PRODUCER_ARTIFACT_KIND,
        "status": audit.PRODUCER_STATUS,
        "contract": dict(expected_contract),
        "parameter_layout": {
            "parameter_count": 8,
            "parameter_names": parameter_names,
            "module_names": module_names,
            "ordered_parameter_names_sha256": hashlib.sha256(names_payload).hexdigest(),
        },
        "records": records,
        "array_manifest": _array_manifest(arrays),
        "provenance": {
            **audit.EXPECTED_PROVENANCE,
            "npz_sha256": npz_sha256,
        },
    }
    return summary, arrays, npz_sha256


def _refresh_manifest(summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    summary["array_manifest"] = _array_manifest(arrays)


def _validate(
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    npz_sha256: str,
) -> dict[str, Any]:
    return audit._validate_and_compute(
        summary,
        arrays,
        json_sha256=_digest("synthetic-json"),
        npz_sha256=npz_sha256,
    )


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def _write_artifacts(
    directory: Path,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    npz_path = directory / "producer.npz"
    np.savez(npz_path, **arrays)
    _write_sidecar(npz_path)
    summary["provenance"]["npz_sha256"] = hashlib.sha256(
        npz_path.read_bytes()
    ).hexdigest()
    _refresh_manifest(summary, arrays)
    json_path = directory / "producer.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(json_path)
    return json_path, npz_path


def test_valid_identical_vectors_pass_and_fp32_is_diagnostic(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    result = _validate(summary, arrays, npz_sha256)
    assert result["status"] == audit.VALID_STATUS
    assert result["decision_outcome"] == audit.NEGLIGIBLE_OUTCOME
    assert result["results"]["bfloat16"]["deciding"] is True
    assert result["results"]["float32"]["deciding"] is False
    assert all(
        case["passed"]
        for regime in result["results"]["bfloat16"]["regimes"].values()
        for case in regime["cases"].values()
    )


def test_fp32_failure_cannot_change_bf16_pass(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    prefix = audit._prefix(audit.REGIMES[0], "float32", ordinal, case_id, "canonical")
    arrays[f"{prefix}parameter_update_float32"] *= -1.0
    _refresh_manifest(summary, arrays)

    result = _validate(summary, arrays, npz_sha256)
    assert result["decision_outcome"] == audit.NEGLIGIBLE_OUTCOME
    assert (
        result["results"]["float32"]["regimes"][audit.REGIMES[0]]["cases"][case_id][
            "passed"
        ]
        is False
    )


def test_bf16_update_miss_is_valid_material_difference(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "canonical"
    )
    arrays[f"{prefix}parameter_update_float32"] *= -1.0
    _refresh_manifest(summary, arrays)

    result = _validate(summary, arrays, npz_sha256)
    assert result["status"] == audit.VALID_STATUS
    assert result["decision_outcome"] == audit.FAIL_OUTCOME
    assert "16-step" in result["next_step"]


def test_active_module_gate_uses_mean_two_arm_energy(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "canonical"
    )
    gradient = arrays[f"{prefix}gradient_float32"].copy()
    gradient[:4] *= -1.0
    arrays[f"{prefix}gradient_float32"] = gradient
    _refresh_manifest(summary, arrays)

    result = _validate(summary, arrays, npz_sha256)
    modules = result["results"]["bfloat16"]["regimes"][audit.REGIMES[0]]["cases"][
        case_id
    ]["modules"]
    assert modules[0]["active"] is True
    assert modules[0]["passed"] is False
    assert result["decision_outcome"] == audit.FAIL_OUTCOME


def test_symmetric_relative_l2_and_inclusive_thresholds() -> None:
    vector = np.array([1.0, 0.0], dtype="<f4")
    scaled = np.array([0.99, 0.0], dtype="<f4")
    assert audit._symmetric_relative_l2(vector, scaled) == pytest.approx(
        audit._symmetric_relative_l2(scaled, vector)
    )
    assert audit._cosine(vector, scaled) == pytest.approx(1.0)
    gate = 2.0 * 0.01 / 1.99
    assert audit._symmetric_relative_l2(vector, scaled) == pytest.approx(gate)


def test_nonfinite_raw_gradient_is_invalid_not_a_scientific_fail(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "canonical"
    )
    arrays[f"{prefix}gradient_float32"][0] = np.nan
    _refresh_manifest(summary, arrays)
    with pytest.raises(audit.ArtifactInvalid, match="non-finite"):
        _validate(summary, arrays, npz_sha256)


def test_parameter_slice_gap_is_invalid(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    arrays["parameter_slice_starts_int64"][1] = 5
    _refresh_manifest(summary, arrays)
    with pytest.raises(audit.ArtifactInvalid, match="contiguous full partition"):
        _validate(summary, arrays, npz_sha256)


def test_module_mapping_must_be_derived_from_parameter_names(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    summary["parameter_layout"]["module_names"] = ["output", "operator"]
    with pytest.raises(audit.ArtifactInvalid, match="parent modules"):
        _validate(summary, arrays, npz_sha256)


def test_control_order_and_rng_must_match_the_frozen_restore(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    record = summary["records"][0]
    for arm in audit.ARMS:
        record["arm_controls"][arm]["parameter_order_sha256"] = _digest("wrong-order")
    with pytest.raises(audit.ArtifactInvalid, match="parameter order"):
        _validate(summary, arrays, npz_sha256)

    expected_order = summary["parameter_layout"]["ordered_parameter_names_sha256"]
    for arm in audit.ARMS:
        record["arm_controls"][arm]["parameter_order_sha256"] = expected_order
        record["arm_controls"][arm]["rng_state_sha256"] = _digest("wrong-rng")
    with pytest.raises(audit.ArtifactInvalid, match="RNG"):
        _validate(summary, arrays, npz_sha256)


def test_fresh_and_checkpoint_states_must_be_distinct(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    fresh = summary["records"][0]["arm_controls"]["legacy"]
    for record in summary["records"]:
        if record["regime"] != "checkpoint_epoch491":
            continue
        for arm in audit.ARMS:
            record["arm_controls"][arm]["initial_parameter_state_sha256"] = fresh[
                "initial_parameter_state_sha256"
            ]
    with pytest.raises(audit.ArtifactInvalid, match="parameter states are identical"):
        _validate(summary, arrays, npz_sha256)


def test_signed_zero_shared_target_difference_is_invalid(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    legacy_prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "legacy"
    )
    canonical_prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "canonical"
    )
    arrays[f"{legacy_prefix}target_pressure_float32"][0] = 0.0
    arrays[f"{canonical_prefix}target_pressure_float32"][0] = -0.0
    _refresh_manifest(summary, arrays)
    with pytest.raises(audit.ArtifactInvalid, match="shared target_pressure"):
        _validate(summary, arrays, npz_sha256)


def test_source_measure_weight_presence_is_invalid(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    record = summary["records"][0]
    for arm in audit.ARMS:
        record["arm_controls"][arm]["source_measure_weights_present"] = True
    with pytest.raises(audit.ArtifactInvalid, match="source measure weights"):
        _validate(summary, arrays, npz_sha256)


def test_provenance_mismatch_is_invalid(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    summary["provenance"]["training_state_sha256"] = _digest("wrong")
    with pytest.raises(audit.ArtifactInvalid, match="training_state"):
        _validate(summary, arrays, npz_sha256)


def test_producer_source_binding_mismatch_is_invalid(
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, npz_sha256 = valid_payload
    summary["provenance"]["producer_sha256"] = _digest("different-producer")
    with pytest.raises(audit.ArtifactInvalid, match="producer_sha256"):
        _validate(summary, arrays, npz_sha256)


def test_end_to_end_valid_fail_exits_zero_and_publishes(
    tmp_path: Path,
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, _npz_sha256 = valid_payload
    ordinal, case_id = audit.CASE_SPECS[0]
    prefix = audit._prefix(
        audit.REGIMES[0], audit.DECIDING_PRECISION, ordinal, case_id, "canonical"
    )
    arrays[f"{prefix}parameter_update_float32"] *= -1.0
    producer_json, producer_npz = _write_artifacts(tmp_path, summary, arrays)
    output = tmp_path / "adjudication.json"
    exit_code = audit.main(
        [
            "--producer-json",
            str(producer_json),
            "--producer-npz",
            str(producer_npz),
            "--output-json",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    assert output.with_name(f"{output.name}.sha256").is_file()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_outcome"] == audit.FAIL_OUTCOME


def test_missing_sidecar_is_incomplete(
    tmp_path: Path,
    valid_payload: tuple[dict[str, Any], dict[str, np.ndarray], str],
) -> None:
    summary, arrays, _npz_sha256 = valid_payload
    producer_json, producer_npz = _write_artifacts(tmp_path, summary, arrays)
    producer_npz.with_name(f"{producer_npz.name}.sha256").unlink()
    result = audit.adjudicate(
        producer_json=producer_json,
        producer_npz=producer_npz,
    )
    assert result["status"] == audit.INCOMPLETE_STATUS
    assert result["decision_outcome"] == audit.INCOMPLETE_OUTCOME


def test_present_invalid_artifact_takes_precedence_over_missing_artifact(
    tmp_path: Path,
) -> None:
    producer_json = tmp_path / "producer.json"
    producer_json.write_bytes(b"{not strict JSON}\n")
    _write_sidecar(producer_json)
    result = audit.adjudicate(
        producer_json=producer_json,
        producer_npz=tmp_path / "missing.npz",
    )
    assert result["status"] == audit.INVALID_STATUS
    assert result["decision_outcome"] == audit.INVALID_OUTCOME
    assert "strict JSON" in result["error"]
    assert "missing" not in result["error"]


def test_dangling_artifact_symlink_is_invalid(tmp_path: Path) -> None:
    producer_json = tmp_path / "producer.json"
    producer_json.symlink_to(tmp_path / "absent-target.json")
    result = audit.adjudicate(
        producer_json=producer_json,
        producer_npz=tmp_path / "missing.npz",
    )
    assert result["status"] == audit.INVALID_STATUS
    assert result["decision_outcome"] == audit.INVALID_OUTCOME
    assert "symlink" in result["error"]


def test_publication_refuses_existing_output_or_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "adjudication.json"
    output.write_bytes(b"owned by another run\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit._exclusive_publish(output, b'{"new": true}\n')
    assert output.read_bytes() == b"owned by another run\n"

    output.unlink()
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.write_bytes(b"owned sidecar\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit._exclusive_publish(output, b'{"new": true}\n')
    assert not output.exists()
    assert sidecar.read_bytes() == b"owned sidecar\n"


def test_preregistration_names_exact_optimizer_and_limited_claims() -> None:
    prereg = (
        Path(__file__).resolve().parents[1]
        / "studies"
        / "phase1_historical_k10000_one_step_parity_prereg_v3_2026-07-29.json"
    )
    payload = json.loads(prereg.read_text(encoding="utf-8"))
    rendered = prereg.read_text(encoding="utf-8")
    assert "CombinedOptimizer(Muon,AdamW)" in rendered
    assert "AdamW-only surrogate is forbidden" in rendered
    assert "target quadrature/HT measure" in rendered
    assert "Source measure weights" in rendered
    assert payload["truth_table"][0]["outcome"] == audit.NEGLIGIBLE_OUTCOME
    assert payload["truth_table"][1]["outcome"] == audit.FAIL_OUTCOME
