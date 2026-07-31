# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the blind canonical fixed-prefix H-QC reducer."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import drivaerml_hqc_canonical_fixed_prefix_reduce as reduce
import numpy as np
import pytest


def _errors() -> dict[str, dict[int, list[float]]]:
    result = {
        arm: {resolution: [1.0] * 36 for resolution in reduce.RESOLUTIONS}
        for arm in ("coupled", "fixed")
    }
    for endpoint in reduce.ENDPOINTS:
        result["coupled"][endpoint] = [4.0] * 36
        result["fixed"][endpoint] = [1.1] * 36
    return result


def _constant_errors(
    *,
    coupled_baseline: float = 1.0,
    fixed_baseline: float = 1.0,
    coupled_endpoint: float = 4.0,
    fixed_endpoint: float = 1.1,
) -> dict[str, dict[int, list[float]]]:
    result = {
        arm: {resolution: [1.0] * 36 for resolution in reduce.RESOLUTIONS}
        for arm in ("coupled", "fixed")
    }
    result["coupled"][reduce.BASELINE_K] = [coupled_baseline] * 36
    result["fixed"][reduce.BASELINE_K] = [fixed_baseline] * 36
    for endpoint in reduce.ENDPOINTS:
        result["coupled"][endpoint] = [coupled_endpoint] * 36
        result["fixed"][endpoint] = [fixed_endpoint] * 36
    return result


def _encoded_compute_inputs(
    uniform: dict[str, dict[int, float]],
    area: dict[str, dict[int, float]],
    *,
    diagnostic_value: float = 7.0,
) -> tuple[dict, dict]:
    predictions = {}
    targets = {}
    for ordinal, *_ in reduce.CASE_SPECS:
        case_predictions = {}
        for precision in reduce.PRECISIONS:
            for resolution in reduce.RESOLUTIONS:
                for arm, panel_name in (
                    ("coupled", "coupled_s_k"),
                    ("fixed", "fixed_id_prefix_s2500"),
                ):
                    for field in ("pressure", "wss"):
                        if precision == "bfloat16" and field == "pressure":
                            values = [
                                uniform[arm][resolution],
                                area[arm][resolution],
                            ]
                        else:
                            values = [diagnostic_value, diagnostic_value * 2.0]
                        case_predictions[(precision, panel_name, resolution, field)] = (
                            np.asarray(values, dtype=np.float64)
                        )
        predictions[ordinal] = {
            "predictions": case_predictions,
            "areas": {
                resolution: np.ones(resolution, dtype="<f4")
                for resolution in reduce.RESOLUTIONS
            },
        }
        targets[ordinal] = {
            "truth_pressure_float32": np.ones(40_000, dtype="<f4"),
            "truth_wss_float32": np.ones((40_000, 3), dtype="<f4"),
        }
    return predictions, targets


def _supported_scalar_panel() -> dict[str, dict[int, float]]:
    return {
        "coupled": {
            2_500: 2.0,
            5_000: 1.0,
            10_000: 1.0,
            20_000: 1.0,
            40_000: 2.0,
        },
        "fixed": {
            2_500: 2.0,
            5_000: 2.0,
            10_000: 2.0,
            20_000: 2.0,
            40_000: 2.0,
        },
    }


def _sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def _main_argv(
    tmp_path: Path,
    *,
    preregistration: Path | None = None,
    prediction_lane_json: tuple[Path, ...] = (),
    prediction_lane_npz: tuple[Path, ...] = (),
) -> tuple[list[str], Path]:
    missing = tmp_path / "missing"
    output = tmp_path / "adjudication.json"
    argv = [
        "--preregistration",
        str(preregistration or missing.with_name("prereg.json")),
        "--activation",
        str(missing.with_name("activation.json")),
        "--launch-manifest",
        str(missing.with_name("launch.json")),
        "--target-wrapper",
        str(missing.with_name("wrapper.sbatch")),
        "--target-producer-test",
        str(missing.with_name("producer_test.py")),
        "--target-wrapper-test",
        str(missing.with_name("wrapper_test.py")),
        "--reducer-test",
        str(missing.with_name("reducer_test.py")),
        "--target-json",
        str(missing.with_name("target.json")),
        "--target-npz",
        str(missing.with_name("target.npz")),
        "--target-done",
        str(missing.with_name("DONE_1.json")),
        "--canonical-k10000-json",
        str(missing.with_name("canonical.json")),
        "--canonical-k10000-npz",
        str(missing.with_name("canonical.npz")),
        "--stage-b-k10000-json",
        str(missing.with_name("stage_b.json")),
        "--stage-b-k10000-npz",
        str(missing.with_name("stage_b.npz")),
        "--accepted-adjudication-json",
        str(missing.with_name("accepted.json")),
        "--output-json",
        str(output),
    ]
    for path in prediction_lane_json:
        argv.extend(("--prediction-lane-json", str(path)))
    for path in prediction_lane_npz:
        argv.extend(("--prediction-lane-npz", str(path)))
    return argv, output


def _static_target_document(npz_path: Path) -> dict:
    digest = "a" * 64
    manifest = {}
    cases = []
    for (
        ordinal,
        case_id,
        reader_index,
        n_master_cells,
        historical_start,
    ) in reduce.CASE_SPECS:
        prefix = f"case_{ordinal:02d}_{case_id}__"
        for key, (dtype, shape) in reduce._target_array_contract().items():
            if key.startswith(prefix):
                manifest[key] = {
                    "shape": list(shape),
                    "dtype": dtype.str,
                    "nbytes": int(np.prod(shape)) * dtype.itemsize,
                    "sha256": digest,
                }
        target_records = {}
        for field, raw_field_name, components, shape in (
            ("pressure", "pMeanTrim", 1, [40_000]),
            ("wss", "wallShearStressMeanTrim", 3, [40_000, 3]),
        ):
            row_bytes = components * 4
            tail_rows = min(40_000, n_master_cells - historical_start)
            head_rows = 40_000 - tail_rows
            spans = [
                {
                    "offset": historical_start * row_bytes,
                    "count": tail_rows * row_bytes,
                }
            ]
            if head_rows:
                spans.append({"offset": 0, "count": head_rows * row_bytes})
            target_records[field] = {
                "raw_field_name": raw_field_name,
                "source_relative_path": (
                    f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                    f"_tensordict/cell_data/{raw_field_name}.memmap"
                ),
                "source_size_bytes": n_master_cells * row_bytes,
                "source_spans_bytes": spans,
                "selected_shape": shape,
                "selected_dtype": "float32_little_endian",
                "selected_sha256": digest,
                "prefix_sha256_by_resolution": {
                    str(value): digest for value in reduce.RESOLUTIONS
                },
                "historical_k10000_prefix_authenticated": True,
            }
        cases.append(
            {
                "cohort_ordinal": ordinal,
                "case_id": case_id,
                "reader_index": reader_index,
                "n_master_cells": n_master_cells,
                "historical_start": historical_start,
                "max_resolution": 40_000,
                "selection": {
                    "kind": "ordered_cyclic_prefix_from_historical_k10000_start",
                    "wraps": historical_start + 40_000 > n_master_cells,
                    "selected_cell_ids_sha256_by_resolution": {
                        str(value): digest for value in reduce.RESOLUTIONS
                    },
                },
                "logical_case_symlink": f"/dataset/{case_id}",
                "symlink_target": f"/source/{case_id}",
                "resolved_case_root": f"/source/{case_id}",
                "cell_data_metadata": {
                    "relative_path": (
                        f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                        "_tensordict/cell_data/meta.json"
                    ),
                    "size_bytes": reduce.EXPECTED_METADATA_SIZE_BYTES[ordinal],
                    "sha256": reduce.EXPECTED_METADATA_SHA256[ordinal],
                },
                "targets": target_records,
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "drivaerml_hqc_nested_raw_target_bundle",
        "status": "PASSED_HQC_NESTED_RAW_TARGET_FREEZE",
        "generated_at_utc": "2026-07-29T00:00:00Z",
        "dataset_root_input": "/dataset",
        "dataset_root_resolved": "/dataset",
        "dataset_manifest_sha256": reduce.EXPECTED_DATASET_SHA256,
        "geometry_manifest": {
            "path": "/stage/drivaerml_geometry_input_manifest_36cases_v1.json",
            "sha256": reduce.EXPECTED_GEOMETRY_SHA256,
        },
        "historical_k10000_target_manifest": {
            "path": ("/stage/historical_k10000_selected_target_input_manifest_v1.json"),
            "sha256": reduce.EXPECTED_HISTORICAL_TARGET_SHA256,
            "prefix_hashes_authenticated": 72,
        },
        "case_count": 36,
        "resolutions": list(reduce.RESOLUTIONS),
        "max_resolution": 40_000,
        "fixed_query_resolution": 2_500,
        "physical_globals": {
            "array_suffix": "physical_globals_float32",
            "field_order": [
                "U_inf_x",
                "U_inf_y",
                "U_inf_z",
                "p_inf",
                "rho_inf",
                "nu",
                "L_ref",
            ],
            "dtype": "float32_little_endian",
            "source": "frozen target-free geometry manifest",
            "transformed_by_target_freezer": False,
        },
        "selection": (
            "one Kmax ordered cyclic panel; every smaller S_k and fixed Q are "
            "exact array prefixes"
        ),
        "read_allowlist": [
            "dataset manifest.json",
            "frozen target-free geometry manifest",
            "frozen historical K=10k target manifest",
            "vehicle cell_data/meta.json",
            "vehicle cell_data/pMeanTrim.memmap selected byte spans only",
            (
                "vehicle cell_data/wallShearStressMeanTrim.memmap selected "
                "byte spans only"
            ),
        ],
        "read_exclusions": {
            "model_opened": False,
            "prediction_opened": False,
            "metric_opened": False,
            "decision_threshold_opened": False,
            "other_cell_data_opened": False,
            "point_data_opened": False,
            "interior_opened": False,
        },
        "publication_contract": {
            "json_manifest_linked_last": True,
            "producer_outputs_are_not_a_commit_marker": True,
            "valid_only_after_external_sidecar_checks_and_done_marker": True,
            "interrupted_partial_bundle_must_not_be_overwritten": True,
        },
        "cases": cases,
        "cohort_sha256": reduce.EXPECTED_COHORT_SHA256,
        "npz": {
            "path": str(npz_path),
            "sha256": digest,
            "array_count": 144,
        },
        "array_manifest": manifest,
        "provenance": {
            "command": ["producer"],
            "script_path": "/producer.py",
            "script_sha256": reduce.EXPECTED_TARGET_PRODUCER_SHA256,
            "numpy": np.__version__,
        },
    }


def _static_launch_manifest(task_root: Path) -> dict:
    digest = "a" * 64
    bindings = {
        key: digest
        for key in (
            "preregistration_sha256",
            "activation_adjudication_sha256",
            "wrapper_sha256",
            "target_wrapper_test_sha256",
            "reducer_sha256",
            "reducer_test_sha256",
        )
    }
    bindings.update(
        {
            "target_producer_sha256": reduce.EXPECTED_TARGET_PRODUCER_SHA256,
            "target_producer_test_sha256": (
                reduce.EXPECTED_TARGET_PRODUCER_TEST_SHA256
            ),
            "geometry_manifest_sha256": reduce.EXPECTED_GEOMETRY_SHA256,
            "historical_target_manifest_sha256": (
                reduce.EXPECTED_HISTORICAL_TARGET_SHA256
            ),
            "dataset_manifest_sha256": reduce.EXPECTED_DATASET_SHA256,
            "cohort_sha256": reduce.EXPECTED_COHORT_SHA256,
            "one_step_producer_sha256": reduce.EXPECTED_ONE_STEP_PRODUCER_SHA256,
            "prediction_lane_json_sha256": list(reduce.EXPECTED_LANE_JSON_SHA256),
            "prediction_lane_npz_sha256": list(reduce.EXPECTED_LANE_NPZ_SHA256),
            "canonical_k10000_json_sha256": (reduce.EXPECTED_CANONICAL_K10_JSON_SHA256),
            "canonical_k10000_npz_sha256": (reduce.EXPECTED_CANONICAL_K10_NPZ_SHA256),
            "stage_b_k10000_json_sha256": reduce.EXPECTED_STAGE_B_K10_JSON_SHA256,
            "stage_b_k10000_npz_sha256": reduce.EXPECTED_STAGE_B_K10_NPZ_SHA256,
            "accepted_adjudication_sha256": (
                reduce.EXPECTED_ACCEPTED_ADJUDICATION_SHA256
            ),
        }
    )
    return {
        "schema_version": 1,
        "artifact_kind": reduce.LAUNCH_MANIFEST_KIND,
        "status": reduce.LAUNCH_MANIFEST_STATUS,
        "attempt_id": task_root.name,
        "task_logical": str(task_root),
        "task_physical": str(task_root),
        "artifacts": {
            "target_json_relative_path": (
                "artifacts/hqc_nested_raw_target_bundle_v1.json"
            ),
            "target_npz_relative_path": (
                "artifacts/hqc_nested_raw_target_bundle_v1.npz"
            ),
            "target_done_pattern": "DONE_<slurm_job_id>.json",
            "reducer_output_relative_path": (
                "artifacts/hqc_canonical_fixed_prefix_adjudication_v1.json"
            ),
        },
        "bindings": bindings,
    }


def _static_activation_document() -> dict:
    cases = {
        case_id: {"deciding": True, "passed": True}
        for case_id in ("run_118", "run_271", "run_429", "run_86")
    }
    return {
        "schema_version": 1,
        "artifact_kind": ("drivaerml_historical_k10000_one_step_parity_adjudication"),
        "status": "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        "decision_outcome": "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS",
        "created_at_utc": "2026-07-29T00:00:00Z",
        "validity": {
            "producer_source_sha256": reduce.EXPECTED_ONE_STEP_PRODUCER_SHA256,
            "producer_json_sha256": "a" * 64,
            "producer_npz_sha256": "b" * 64,
            "raw_array_manifest_verified": True,
            "shared_controls_verified": True,
            "parameter_partition_verified": True,
            "all_values_finite": True,
        },
        "decision_contract": {
            "deciding_precision": "bfloat16",
            "all_case_regime_gates_required": True,
            "gradient_cosine_inclusive_minimum": 0.999,
            "update_cosine_inclusive_minimum": 0.9999,
            "update_symmetric_relative_l2_inclusive_maximum": 0.01,
            "gradient_path_fraction_of_between_case_median_inclusive_maximum": 0.1,
            "active_module_energy_fraction_inclusive_minimum": 0.01,
            "active_module_cosine_inclusive_minimum": 0.99,
            "fp32_role": "diagnostic_only",
        },
        "results": {
            precision: {
                "deciding": precision == "bfloat16",
                "regimes": {
                    regime: {"cases": cases}
                    for regime in ("fresh_seed42", "checkpoint_epoch491")
                },
            }
            for precision in reduce.PRECISIONS
        },
        "limited_claim": "synthetic test fixture",
        "next_step": "synthetic test fixture",
    }


def test_json_exact_rejects_bool_integer_coercion_recursively() -> None:
    assert not reduce._json_exact(True, 1)
    assert not reduce._json_exact(False, 0)
    assert not reduce._json_exact(
        {"schema_version": True, "required": 1},
        {"schema_version": 1, "required": True},
    )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("schema_version",), True, "schema_version differs"),
        (
            ("decision_contract", "all_case_regime_gates_required"),
            1,
            "decision contract differs",
        ),
    ],
)
def test_activation_rejects_bool_integer_substitutions(
    path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    document = _static_activation_document()
    destination = document
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = replacement
    with pytest.raises(reduce.InvalidEvidence, match=message):
        reduce._validate_activation(document)


def test_truth_transform_is_exact_float32() -> None:
    raw_pressure = np.array([101.0, 103.0], dtype="<f4")
    raw_wss = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype="<f4")
    physical_globals = np.array(
        [2.0, 0.0, 0.0, 100.0, 1.25, 1.0e-5, 5.0],
        dtype="<f4",
    )
    pressure, wss = reduce._float32_truth(
        raw_pressure,
        raw_wss,
        physical_globals,
    )
    q_inf = (
        np.float32(0.5)
        * physical_globals[4]
        * np.sum(
            physical_globals[:3] * physical_globals[:3],
            dtype=np.float32,
        )
    )
    expected_pressure = np.asarray(
        (raw_pressure - physical_globals[3]) / q_inf,
        dtype="<f4",
    )
    expected_wss = np.asarray(
        (raw_wss / q_inf) / (np.float32(0.00313) + np.float32(1.0e-8)),
        dtype="<f4",
    )
    assert pressure.tobytes() == expected_pressure.tobytes()
    assert wss.tobytes() == expected_wss.tobytes()


def test_truth_transform_rejects_nonpositive_dynamic_pressure() -> None:
    with pytest.raises(reduce.InvalidEvidence, match="dynamic pressure"):
        reduce._float32_truth(
            np.ones(1, dtype="<f4"),
            np.ones((1, 3), dtype="<f4"),
            np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0e-5, 1.0], dtype="<f4"),
        )


def test_metric_panel_supports_both_endpoints() -> None:
    result = reduce._metric_panel(_errors(), common_eligible=True)
    assert result["classification"] == "SUPPORTED"
    assert result["eligible"] is True
    assert result["both_endpoint_support_passed"] is True
    assert result["any_futility_triggered"] is False


def test_futility_precedes_separately_passing_support() -> None:
    errors = _errors()
    # The ordinary even-N median ratio is (0.1 + 3.9) / 2 = 2, while
    # the median of logs is log(sqrt(0.39)) < 0. This intentionally
    # exercises the frozen K=40k futility precedence in an overlap.
    errors["fixed"][40_000] = [0.1] * 18 + [3.9] * 18
    result = reduce._metric_panel(errors, common_eligible=True)
    endpoint = result["endpoints"]["40000"]
    assert endpoint["support_passed"] is True
    assert endpoint["futility_k40000_ratio_triggered"] is True
    assert result["classification"] == reduce.FUTILE_OUTCOME


def test_k2500_wiring_requires_exact_metric_identity() -> None:
    coupled = {
        "uniform_pressure_relative_l2": 0.25,
        "uniform_wss_frobenius_relative_l2": 0.5,
        "canonical_area_pressure_relative_l2": 0.2,
        "canonical_area_wss_frobenius_relative_l2": 0.4,
    }
    reduce._validate_k2500_metric_identity(coupled, dict(coupled))
    fixed = dict(coupled)
    fixed["uniform_pressure_relative_l2"] += 1.0e-12
    with pytest.raises(reduce.InvalidEvidence, match="not exactly identical"):
        reduce._validate_k2500_metric_identity(coupled, fixed)


def test_absent_coupled_cliff_is_ineligible() -> None:
    errors = _errors()
    for endpoint in reduce.ENDPOINTS:
        errors["coupled"][endpoint] = [1.0] * 36
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["classification"] == reduce.INELIGIBLE_OUTCOME
    assert result["both_endpoint_coupled_cliffs_passed"] is False


def test_zero_baseline_is_ineligible_without_nan_or_inf() -> None:
    errors = _errors()
    errors["coupled"][reduce.BASELINE_K][0] = 0.0
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["classification"] == reduce.INELIGIBLE_OUTCOME
    assert result["baseline_fixed_over_coupled_median_error_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_eligible_but_unsupported_and_nonfutile_is_mixed() -> None:
    errors = _constant_errors(coupled_endpoint=2.0, fixed_endpoint=1.3)
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["eligible"] is True
    assert result["both_endpoint_support_passed"] is False
    assert result["any_futility_triggered"] is False
    assert result["classification"] == reduce.MIXED_OUTCOME


@pytest.mark.parametrize("baseline", reduce.BASELINE_BOUNDS)
def test_baseline_comparability_bounds_are_inclusive(baseline: float) -> None:
    errors = _constant_errors(
        fixed_baseline=baseline,
        fixed_endpoint=1.1 * baseline,
    )
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["baseline_fixed_over_coupled_median_error_ratio"] == baseline
    assert result["baseline_comparability_passed"] is True
    assert result["classification"] == "SUPPORTED"


@pytest.mark.parametrize(
    "baseline",
    (
        math.nextafter(reduce.BASELINE_BOUNDS[0], 0.0),
        math.nextafter(reduce.BASELINE_BOUNDS[1], math.inf),
    ),
)
def test_baseline_comparability_just_outside_is_ineligible(
    baseline: float,
) -> None:
    errors = _constant_errors(
        fixed_baseline=baseline,
        fixed_endpoint=1.1 * baseline,
    )
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["baseline_comparability_passed"] is False
    assert result["classification"] == reduce.INELIGIBLE_OUTCOME


def test_coupled_cliff_log_and_count_boundaries_are_inclusive() -> None:
    errors = _constant_errors(coupled_endpoint=2.0, fixed_endpoint=1.0)
    for endpoint in reduce.ENDPOINTS:
        errors["coupled"][endpoint] = [1.5] * 12 + [2.0] * 24
    result = reduce._metric_panel(errors, common_eligible=True)
    for endpoint in reduce.ENDPOINTS:
        row = result["endpoints"][str(endpoint)]
        assert row["coupled_median_log_error_ratio"] == reduce.CLIFF_LOG_MIN
        assert row["coupled_error_ratio_at_least_2_case_count"] == 24
        assert row["eligibility_coupled_cliff_passed"] is True
    assert result["classification"] == "SUPPORTED"

    for endpoint in reduce.ENDPOINTS:
        errors["coupled"][endpoint] = [1.5] * 13 + [2.0] * 23
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["coupled_median_log_error_ratio"]
        == reduce.CLIFF_LOG_MIN
    )
    assert (
        result["endpoints"]["2500"]["coupled_error_ratio_at_least_2_case_count"] == 23
    )
    assert result["classification"] == reduce.INELIGIBLE_OUTCOME


def test_support_fraction_boundary_is_inclusive() -> None:
    boundary = math.exp(reduce.SUPPORT_FRACTION_MAX * reduce.CLIFF_LOG_MIN)
    errors = _constant_errors(coupled_endpoint=2.0, fixed_endpoint=boundary)
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["classification"] == "SUPPORTED"
    assert (
        result["endpoints"]["2500"]["fixed_positive_log_fraction_of_coupled"]
        <= reduce.SUPPORT_FRACTION_MAX
    )

    above = math.nextafter(boundary, math.inf)
    errors = _constant_errors(coupled_endpoint=2.0, fixed_endpoint=above)
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["fixed_positive_log_fraction_of_coupled"]
        > reduce.SUPPORT_FRACTION_MAX
    )
    assert result["classification"] == reduce.MIXED_OUTCOME


def test_support_fixed_log_boundary_is_inclusive() -> None:
    errors = _constant_errors(coupled_endpoint=3.0, fixed_endpoint=1.25)
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["fixed_median_log_error_ratio"]
        == reduce.SUPPORT_FIXED_LOG_MAX
    )
    assert result["classification"] == "SUPPORTED"

    errors = _constant_errors(
        coupled_endpoint=3.0,
        fixed_endpoint=math.nextafter(1.25, math.inf),
    )
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["classification"] == reduce.MIXED_OUTCOME


def test_support_favorable_count_boundary_is_inclusive() -> None:
    errors = _constant_errors(coupled_endpoint=2.0, fixed_endpoint=1.1)
    for endpoint in reduce.ENDPOINTS:
        errors["fixed"][endpoint] = [1.1] * 27 + [2.0] * 9
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["paired_fixed_log_less_than_coupled_case_count"]
        == 27
    )
    assert result["classification"] == "SUPPORTED"

    for endpoint in reduce.ENDPOINTS:
        errors["fixed"][endpoint] = [1.1] * 26 + [2.0] * 10
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["paired_fixed_log_less_than_coupled_case_count"]
        == 26
    )
    assert result["classification"] == reduce.MIXED_OUTCOME


def test_futility_fraction_boundary_is_inclusive() -> None:
    errors = _constant_errors()
    errors["coupled"][2_500] = [4.0] * 36
    errors["fixed"][2_500] = [2.0] * 36
    result = reduce._metric_panel(errors, common_eligible=True)
    assert (
        result["endpoints"]["2500"]["fixed_positive_log_fraction_of_coupled"]
        == reduce.FUTILITY_FRACTION_MIN
    )
    assert result["endpoints"]["2500"]["futility_fraction_triggered"] is True
    assert result["classification"] == reduce.FUTILE_OUTCOME

    errors["fixed"][2_500] = [math.nextafter(2.0, 0.0)] * 36
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["endpoints"]["2500"]["futility_fraction_triggered"] is False
    assert result["classification"] == reduce.MIXED_OUTCOME


def test_k40000_futility_ratio_boundary_is_inclusive() -> None:
    errors = _errors()
    errors["fixed"][40_000] = [0.1] * 18 + [3.9] * 18
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["endpoints"]["40000"]["fixed_median_error_ratio"] == 2.0
    assert result["endpoints"]["40000"]["futility_k40000_ratio_triggered"] is True
    assert result["classification"] == reduce.FUTILE_OUTCOME

    errors["fixed"][40_000] = [0.1] * 18 + [math.nextafter(3.9, 0.0)] * 18
    result = reduce._metric_panel(errors, common_eligible=True)
    assert result["endpoints"]["40000"]["fixed_median_error_ratio"] < 2.0
    assert result["endpoints"]["40000"]["futility_k40000_ratio_triggered"] is False
    assert result["classification"] == "SUPPORTED"


@pytest.mark.parametrize(
    ("area_k40000", "expected"),
    (
        (1.25, reduce.AREA_FLAT_OUTCOME),
        (math.nextafter(1.25, math.inf), reduce.UNIFORM_ONLY_OUTCOME),
    ),
)
def test_compute_panel_area_subclassification(
    monkeypatch: pytest.MonkeyPatch,
    area_k40000: float,
    expected: str,
) -> None:
    uniform = _supported_scalar_panel()
    area = {
        "coupled": {
            2_500: 1.25,
            5_000: 1.0,
            10_000: 1.0,
            20_000: 1.0,
            40_000: area_k40000,
        },
        "fixed": {
            2_500: 1.25,
            5_000: 1.0,
            10_000: 1.0,
            20_000: 1.0,
            40_000: 1.25,
        },
    }
    predictions, targets = _encoded_compute_inputs(uniform, area)
    monkeypatch.setattr(
        reduce,
        "_relative_l2",
        lambda prediction, truth: float(prediction[0]),
    )
    monkeypatch.setattr(
        reduce,
        "_weighted_relative_l2",
        lambda prediction, truth, weights: float(prediction[1]),
    )
    result = reduce._compute_panel(predictions=predictions, targets=targets)
    assert result["decision_outcome"] == expected
    assert (
        result["canonical_area_bfloat16_pressure_panel"]["endpoints"]["2500"][
            "coupled_median_error_ratio"
        ]
        == 1.25
    )


def test_compute_panel_dual_weighting_and_diagnostic_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported = _supported_scalar_panel()
    monkeypatch.setattr(
        reduce,
        "_relative_l2",
        lambda prediction, truth: float(prediction[0]),
    )
    monkeypatch.setattr(
        reduce,
        "_weighted_relative_l2",
        lambda prediction, truth, weights: float(prediction[1]),
    )
    predictions_low, targets = _encoded_compute_inputs(
        supported,
        supported,
        diagnostic_value=1.0e-20,
    )
    result_low = reduce._compute_panel(
        predictions=predictions_low,
        targets=targets,
    )
    predictions_high, _ = _encoded_compute_inputs(
        supported,
        supported,
        diagnostic_value=1.0e20,
    )
    result_high = reduce._compute_panel(
        predictions=predictions_high,
        targets=targets,
    )
    assert result_low["decision_outcome"] == reduce.DUAL_OUTCOME
    assert result_high["decision_outcome"] == reduce.DUAL_OUTCOME
    assert (
        result_low["uniform_bfloat16_pressure_panel"]
        == result_high["uniform_bfloat16_pressure_panel"]
    )
    assert (
        result_low["canonical_area_bfloat16_pressure_panel"]
        == result_high["canonical_area_bfloat16_pressure_panel"]
    )
    assert result_low["ordered_diagnostics"] != result_high["ordered_diagnostics"]


def test_present_corrupt_sidecar_wins_over_missing_input(tmp_path: Path) -> None:
    present = tmp_path / "present.json"
    present.write_text("{}\n", encoding="utf-8")
    present.with_name(f"{present.name}.sha256").write_text(
        f"{'0' * 64}  {present.name}\n",
        encoding="ascii",
    )
    missing = tmp_path / "missing.json"
    with pytest.raises(reduce.InvalidEvidence, match="sidecar differs"):
        reduce._preflight_inputs(
            (
                (missing, None, True),
                (present, None, True),
            )
        )


def test_missing_input_is_reported_as_incomplete(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    errors = reduce._preflight_inputs(((missing, None, True),))
    assert errors == [f"required artifact is absent: {missing}"]


def test_malformed_orphan_sidecar_wins_over_missing_primary(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    missing.with_name(f"{missing.name}.sha256").write_text(
        "not-a-sidecar\n",
        encoding="ascii",
    )
    with pytest.raises(reduce.InvalidEvidence, match="orphan.*malformed"):
        reduce._preflight_inputs(((missing, None, True),))


def test_well_formed_orphan_sidecar_remains_incomplete(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    missing.with_name(f"{missing.name}.sha256").write_text(
        f"{'a' * 64}  {missing.name}\n",
        encoding="ascii",
    )
    errors = reduce._preflight_inputs(((missing, None, True),))
    assert errors == [f"required artifact is absent: {missing}"]


def test_malformed_present_json_is_invalid_even_without_sidecar(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"broken":', encoding="utf-8")
    with pytest.raises(reduce.InvalidEvidence, match="strict JSON"):
        reduce._audit_present_target_envelope(
            target,
            npz_path=tmp_path / "target.npz",
        )


def test_strict_json_translates_integer_digit_limit() -> None:
    payload = b'{"value":' + b"9" * 5_000 + b"}"
    with pytest.raises(reduce.InvalidEvidence, match="not strict JSON"):
        reduce._strict_json(payload, context="oversized integer")


def test_strict_json_translates_recursion_limit() -> None:
    payload = b'{"value":' + b"[" * 20_000 + b"]" * 20_000 + b"}"
    with pytest.raises(reduce.InvalidEvidence, match="not strict JSON"):
        reduce._strict_json(payload, context="deep nesting")


def test_deep_invalid_target_json_wins_before_missing_npz(tmp_path: Path) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    reduce._validate_target_document_static(document, npz_path=npz_path)
    document["read_exclusions"]["model_opened"] = True
    target = tmp_path / "target.json"
    target.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(reduce.InvalidEvidence, match="exclusions differ"):
        reduce._audit_present_target_envelope(target, npz_path=npz_path)


def test_target_document_rejects_false_source_provenance(tmp_path: Path) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    document["cases"][0]["targets"]["pressure"]["raw_field_name"] = "pressure"
    with pytest.raises(reduce.InvalidEvidence, match="pressure contract differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


def test_target_document_rejects_false_cyclic_wrap_flag(tmp_path: Path) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    document["cases"][29]["selection"]["wraps"] = False
    with pytest.raises(reduce.InvalidEvidence, match="selection contract differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


def test_target_document_cross_binds_selected_and_array_hashes(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    document["cases"][0]["targets"]["pressure"]["selected_sha256"] = "b" * 64
    with pytest.raises(reduce.InvalidEvidence, match="manifest binding differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


def test_target_document_rejects_extra_npz_record_field(tmp_path: Path) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    document["npz"]["unregistered"] = "value"
    with pytest.raises(reduce.InvalidEvidence, match="NPZ record differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


def test_target_document_rejects_boolean_schema_version(tmp_path: Path) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    document["schema_version"] = True
    with pytest.raises(reduce.InvalidEvidence, match="schema_version differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


@pytest.mark.parametrize(
    "mutation",
    ("array_count", "manifest_shape", "manifest_nbytes", "selected_shape"),
)
def test_target_document_rejects_numeric_type_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    npz_path = tmp_path / "target.npz"
    document = _static_target_document(npz_path)
    if mutation == "array_count":
        document["npz"]["array_count"] = 144.0
    elif mutation == "manifest_shape":
        first = next(iter(document["array_manifest"].values()))
        first["shape"][0] = float(first["shape"][0])
    elif mutation == "manifest_nbytes":
        first = next(iter(document["array_manifest"].values()))
        first["nbytes"] = float(first["nbytes"])
    else:
        document["cases"][0]["targets"]["pressure"]["selected_shape"] = [40_000.0]
    with pytest.raises(reduce.InvalidEvidence, match="differs"):
        reduce._validate_target_document_static(document, npz_path=npz_path)


def test_malformed_present_npz_is_invalid_even_if_json_is_missing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.npz"
    target.write_bytes(b"not-an-npz")
    with pytest.raises(reduce.InvalidEvidence, match="valid no-pickle NPZ"):
        reduce._audit_present_target_npz(target)


def test_parseable_wrong_member_npz_is_invalid_before_missing_json(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.npz"
    np.savez(target, wrong=np.zeros(1, dtype="<f4"))
    with pytest.raises(reduce.InvalidEvidence, match="member names differ"):
        reduce._audit_present_target_npz(target)


def test_npz_translates_unsupported_zip_compression() -> None:
    npy = io.BytesIO()
    np.save(npy, np.zeros(1, dtype="<f4"), allow_pickle=False)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        stream.writestr("value.npy", npy.getvalue())
    payload = bytearray(archive.getvalue())
    local = payload.find(b"PK\x03\x04")
    central = payload.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    payload[local + 8 : local + 10] = (99).to_bytes(2, "little")
    payload[central + 10 : central + 12] = (99).to_bytes(2, "little")
    with pytest.raises(reduce.InvalidEvidence, match="valid no-pickle NPZ"):
        reduce._npz_arrays(bytes(payload), context="unsupported compression")


def test_malformed_present_done_is_invalid_even_if_npz_is_missing(
    tmp_path: Path,
) -> None:
    done = tmp_path / "DONE.json"
    done.write_text("{}\n", encoding="utf-8")
    with pytest.raises(reduce.InvalidEvidence, match="fields differ"):
        reduce._audit_present_target_done(done)


def test_present_done_rejects_boolean_schema_version(tmp_path: Path) -> None:
    done = tmp_path / "DONE_123.json"
    done.write_text(
        json.dumps(
            {
                "artifact_kind": "drivaerml_hqc_nested_target_bundle_commit",
                "activation_adjudication_sha256": "a" * 64,
                "attempt_id": "attempt-1",
                "job_id": "123",
                "json_sha256": "b" * 64,
                "launch_manifest_sha256": "c" * 64,
                "npz_sha256": "d" * 64,
                "preregistration_sha256": "e" * 64,
                "producer_sha256": reduce.EXPECTED_TARGET_PRODUCER_SHA256,
                "reducer_schema_validation_performed": False,
                "schema_version": True,
                "status": ("CONTENT_COMMITTED_UNVALIDATED_HQC_NESTED_TARGET_BUNDLE"),
                "wrapper_sha256": "f" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(reduce.InvalidEvidence, match="DONE envelope differs"):
        reduce._audit_present_target_done(done)


def test_unauthorized_target_bytes_are_not_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_npz = tmp_path / "unseen-target.npz"
    target_npz.write_bytes(b"must-not-open")
    original_stable_read = reduce._stable_read

    def guarded_read(path: Path) -> bytes:
        if path == target_npz:
            pytest.fail("unauthorized target bytes were opened")
        return original_stable_read(path)

    monkeypatch.setattr(reduce, "_stable_read", guarded_read)
    missing = tmp_path / "missing"
    args = SimpleNamespace(
        preregistration=missing.with_name("prereg.json"),
        activation=missing.with_name("activation.json"),
        launch_manifest=missing.with_name("launch.json"),
        target_wrapper=missing.with_name("wrapper.sbatch"),
        target_producer_test=missing.with_name("producer_test.py"),
        target_wrapper_test=missing.with_name("wrapper_test.py"),
        reducer_test=missing.with_name("reducer_test.py"),
        target_json=missing.with_name("target.json"),
        target_npz=target_npz,
        target_done=missing.with_name("DONE_1.json"),
        canonical_k10000_json=missing.with_name("canonical.json"),
        canonical_k10000_npz=missing.with_name("canonical.npz"),
        stage_b_k10000_json=missing.with_name("stage_b.json"),
        stage_b_k10000_npz=missing.with_name("stage_b.npz"),
        accepted_adjudication_json=missing.with_name("accepted.json"),
        prediction_lane_json=[
            missing.with_name(f"lane_{index}.json") for index in range(4)
        ],
        prediction_lane_npz=[
            missing.with_name(f"lane_{index}.npz") for index in range(4)
        ],
        output_json=missing.with_name("output.json"),
    )
    result = reduce.adjudicate(args)
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "without a valid authenticated" in result["errors"][0]


def test_target_appearing_after_unauthorized_inventory_is_not_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_npz = tmp_path / "late-target.npz"
    missing = tmp_path / "missing"
    args = SimpleNamespace(
        preregistration=missing.with_name("prereg.json"),
        activation=missing.with_name("activation.json"),
        launch_manifest=missing.with_name("launch.json"),
        target_wrapper=missing.with_name("wrapper.sbatch"),
        target_producer_test=missing.with_name("producer_test.py"),
        target_wrapper_test=missing.with_name("wrapper_test.py"),
        reducer_test=missing.with_name("reducer_test.py"),
        target_json=missing.with_name("target.json"),
        target_npz=target_npz,
        target_done=missing.with_name("DONE_1.json"),
        canonical_k10000_json=missing.with_name("canonical.json"),
        canonical_k10000_npz=missing.with_name("canonical.npz"),
        stage_b_k10000_json=missing.with_name("stage_b.json"),
        stage_b_k10000_npz=missing.with_name("stage_b.npz"),
        accepted_adjudication_json=missing.with_name("accepted.json"),
        prediction_lane_json=[
            missing.with_name(f"late_lane_{index}.json") for index in range(4)
        ],
        prediction_lane_npz=[
            missing.with_name(f"late_lane_{index}.npz") for index in range(4)
        ],
        output_json=missing.with_name("output.json"),
    )
    original_entry_exists = reduce._entry_exists
    original_stable_read = reduce._stable_read
    calls = 0

    def racing_entry_exists(path: Path) -> bool:
        nonlocal calls
        calls += 1
        if calls == 5:
            target_npz.write_bytes(b"late-unseen-target")
        return original_entry_exists(path)

    def guarded_read(path: Path) -> bytes:
        if path == target_npz:
            pytest.fail("late unauthorized target bytes were opened")
        return original_stable_read(path)

    monkeypatch.setattr(reduce, "_entry_exists", racing_entry_exists)
    monkeypatch.setattr(reduce, "_stable_read", guarded_read)
    result = reduce.adjudicate(args)
    assert result["decision_outcome"] == reduce.INCOMPLETE_OUTCOME
    assert target_npz.exists()


def test_missing_prediction_lanes_publish_incomplete_result(
    tmp_path: Path,
) -> None:
    argv, output = _main_argv(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        reduce.main(argv)
    assert exit_info.value.code == 3
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_outcome"] == reduce.INCOMPLETE_OUTCOME
    assert any(
        "exactly four JSON and four NPZ prediction lanes" in error
        for error in result["errors"]
    )


def test_present_bad_partial_lane_wins_over_missing_lane_cardinality(
    tmp_path: Path,
) -> None:
    bad_lane = tmp_path / "bad-lane.json"
    bad_lane.write_text('{"malformed":true}\n', encoding="utf-8")
    _sidecar(bad_lane)
    argv, output = _main_argv(
        tmp_path,
        prediction_lane_json=(bad_lane,),
    )
    with pytest.raises(SystemExit) as exit_info:
        reduce.main(argv)
    assert exit_info.value.code == 4
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "frozen SHA-256 differs" in result["errors"][0]


def test_extra_prediction_lane_publishes_invalid_result(tmp_path: Path) -> None:
    missing_lanes = tuple(tmp_path / f"lane-{index}.json" for index in range(5))
    argv, output = _main_argv(
        tmp_path,
        prediction_lane_json=missing_lanes,
    )
    with pytest.raises(SystemExit) as exit_info:
        reduce.main(argv)
    assert exit_info.value.code == 4
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "more than four" in result["errors"][0]


def test_overlong_input_path_publishes_invalid_result(tmp_path: Path) -> None:
    overlong = tmp_path / ("x" * 5_000)
    argv, output = _main_argv(tmp_path, preregistration=overlong)
    with pytest.raises(SystemExit) as exit_info:
        reduce.main(argv)
    assert exit_info.value.code == 4
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "safely" in result["errors"][0]


def test_raced_missing_preregistration_does_not_mask_invalid_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    activation = tmp_path / "activation.json"
    activation.write_text('{"broken":', encoding="utf-8")
    missing = tmp_path / "missing"
    args = SimpleNamespace(
        preregistration=preregistration,
        activation=activation,
        launch_manifest=missing.with_name("launch.json"),
        target_wrapper=missing.with_name("wrapper.sbatch"),
        target_producer_test=missing.with_name("producer_test.py"),
        target_wrapper_test=missing.with_name("wrapper_test.py"),
        reducer_test=missing.with_name("reducer_test.py"),
        target_json=missing.with_name("target.json"),
        target_npz=missing.with_name("target.npz"),
        target_done=missing.with_name("DONE_1.json"),
        canonical_k10000_json=missing.with_name("canonical.json"),
        canonical_k10000_npz=missing.with_name("canonical.npz"),
        stage_b_k10000_json=missing.with_name("stage_b.json"),
        stage_b_k10000_npz=missing.with_name("stage_b.npz"),
        accepted_adjudication_json=missing.with_name("accepted.json"),
        prediction_lane_json=[],
        prediction_lane_npz=[],
        output_json=missing.with_name("output.json"),
    )
    original_stable_read = reduce._stable_read
    raced = False

    def disappearing_read(path: Path) -> bytes:
        nonlocal raced
        if path == preregistration and not raced:
            raced = True
            raise reduce.IncompleteEvidence("raced preregistration disappearance")
        return original_stable_read(path)

    monkeypatch.setattr(reduce, "_stable_read", disappearing_read)
    result = reduce.adjudicate(args)
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "not strict JSON" in result["errors"][0]


def test_target_binding_audit_does_not_restat_flapping_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_json = tmp_path / "target.json"
    target_json.write_text("{}\n", encoding="utf-8")
    target_npz = tmp_path / "target.npz"
    target_npz.write_bytes(b"snapshot")
    missing = tmp_path / "missing"
    original_regular_entry = reduce._regular_entry
    json_checks = 0

    def flapping_regular_entry(path: Path) -> bool:
        nonlocal json_checks
        if path == target_json:
            json_checks += 1
            return json_checks > 1
        return original_regular_entry(path)

    monkeypatch.setattr(reduce, "_regular_entry", flapping_regular_entry)
    reduce._audit_present_target_bindings(
        json_path=target_json,
        npz_path=target_npz,
        done_path=missing.with_name("DONE_1.json"),
        activation_path=missing.with_name("activation.json"),
        preregistration_path=missing.with_name("prereg.json"),
        launch_manifest_path=missing.with_name("launch.json"),
        wrapper_path=missing.with_name("wrapper.sbatch"),
    )
    assert json_checks == 1


def test_present_bad_static_manifest_binding_wins_over_missing_companions(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "attempt-1"
    (task_root / "artifacts").mkdir(parents=True)
    manifest = _static_launch_manifest(task_root)
    manifest["bindings"]["dataset_manifest_sha256"] = "b" * 64
    manifest_path = task_root / "hqc_nested_target_freeze_launch_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    _sidecar(manifest_path)
    missing = task_root / "missing"
    args = SimpleNamespace(
        preregistration=missing.with_name("prereg.json"),
        activation=missing.with_name("activation.json"),
        launch_manifest=manifest_path,
        target_wrapper=missing.with_name("wrapper.sbatch"),
        target_producer_test=missing.with_name("producer_test.py"),
        target_wrapper_test=missing.with_name("wrapper_test.py"),
        reducer_test=missing.with_name("reducer_test.py"),
        target_json=task_root / "artifacts/hqc_nested_raw_target_bundle_v1.json",
        target_npz=task_root / "artifacts/hqc_nested_raw_target_bundle_v1.npz",
        target_done=task_root / "DONE_123.json",
        canonical_k10000_json=missing.with_name("canonical.json"),
        canonical_k10000_npz=missing.with_name("canonical.npz"),
        stage_b_k10000_json=missing.with_name("stage_b.json"),
        stage_b_k10000_npz=missing.with_name("stage_b.npz"),
        accepted_adjudication_json=missing.with_name("accepted.json"),
        prediction_lane_json=[],
        prediction_lane_npz=[],
        output_json=(
            task_root / "artifacts/hqc_canonical_fixed_prefix_adjudication_v1.json"
        ),
    )
    result = reduce.adjudicate(args)
    assert result["decision_outcome"] == reduce.INVALID_OUTCOME
    assert "dataset_manifest_sha256 differs" in result["errors"][0]


def test_launch_manifest_must_occupy_exact_attempt_root_path(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "attempt-1"
    artifacts = task_root / "artifacts"
    artifacts.mkdir(parents=True)
    manifest = _static_launch_manifest(task_root)
    with pytest.raises(reduce.InvalidEvidence, match="exact attempt-root"):
        reduce._validate_launch_manifest_static(
            manifest,
            launch_manifest_path=task_root / "renamed-launch.json",
            target_json_path=(artifacts / "hqc_nested_raw_target_bundle_v1.json"),
            target_npz_path=artifacts / "hqc_nested_raw_target_bundle_v1.npz",
            target_done_path=task_root / "DONE_123.json",
            output_json_path=(
                artifacts / "hqc_canonical_fixed_prefix_adjudication_v1.json"
            ),
        )


def test_launch_manifest_rejects_boolean_schema_version(tmp_path: Path) -> None:
    task_root = tmp_path / "attempt-1"
    artifacts = task_root / "artifacts"
    artifacts.mkdir(parents=True)
    manifest = _static_launch_manifest(task_root)
    manifest["schema_version"] = True
    with pytest.raises(reduce.InvalidEvidence, match="envelope differs"):
        reduce._validate_launch_manifest_static(
            manifest,
            launch_manifest_path=(
                task_root / "hqc_nested_target_freeze_launch_manifest_v1.json"
            ),
            target_json_path=(artifacts / "hqc_nested_raw_target_bundle_v1.json"),
            target_npz_path=artifacts / "hqc_nested_raw_target_bundle_v1.npz",
            target_done_path=task_root / "DONE_123.json",
            output_json_path=(
                artifacts / "hqc_canonical_fixed_prefix_adjudication_v1.json"
            ),
        )


def test_launch_manifest_rejects_embedded_null_in_internal_root(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "attempt-1"
    (task_root / "artifacts").mkdir(parents=True)
    manifest = _static_launch_manifest(task_root)
    manifest["task_physical"] = f"{tmp_path}/bad\u0000component/attempt-1"
    with pytest.raises(reduce.InvalidEvidence, match="namespace is unavailable"):
        reduce._validate_launch_manifest_static(
            manifest,
            launch_manifest_path=(
                task_root / "hqc_nested_target_freeze_launch_manifest_v1.json"
            ),
            target_json_path=(
                task_root / "artifacts/hqc_nested_raw_target_bundle_v1.json"
            ),
            target_npz_path=(
                task_root / "artifacts/hqc_nested_raw_target_bundle_v1.npz"
            ),
            target_done_path=task_root / "DONE_123.json",
            output_json_path=(
                task_root / "artifacts/hqc_canonical_fixed_prefix_adjudication_v1.json"
            ),
        )


def test_preregistration_projection_rejects_scientific_mutation() -> None:
    prereg_path = (
        Path(__file__).resolve().parents[1]
        / "studies"
        / "phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json"
    )
    document = json.loads(prereg_path.read_text(encoding="utf-8"))
    reducer_sha256 = "b" * 64
    document["implementation_freeze"] = {
        "reducer_path": ("studies/drivaerml_hqc_canonical_fixed_prefix_reduce.py"),
        "reducer_sha256": reducer_sha256,
        "reducer_test_path": (
            "tests/test_drivaerml_hqc_canonical_fixed_prefix_reduce.py"
        ),
        "reducer_test_sha256": "c" * 64,
        "target_producer_path": ("studies/drivaerml_hqc_nested_target_input_freeze.py"),
        "target_producer_sha256": reduce.EXPECTED_TARGET_PRODUCER_SHA256,
        "target_producer_test_path": (
            "tests/test_drivaerml_hqc_nested_target_input_freeze.py"
        ),
        "target_producer_test_sha256": (reduce.EXPECTED_TARGET_PRODUCER_TEST_SHA256),
        "exit_codes": {"valid": 0, "incomplete": 3, "invalid": 4},
    }
    projection = dict(document)
    projection.pop("implementation_freeze")
    digest = hashlib.sha256(
        json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == reduce.EXPECTED_PREREGISTRATION_CONTRACT_SHA256
    reduce._validate_preregistration(
        document,
        reducer_sha256=reducer_sha256,
    )

    document["decision_gates"]["support_per_endpoint"][
        "fixed_positive_median_log_cliff_fraction_of_coupled_inclusive_maximum"
    ] = 0.2501
    with pytest.raises(
        reduce.InvalidEvidence,
        match="scientific contract projection differs",
    ):
        reduce._validate_preregistration(
            document,
            reducer_sha256=reducer_sha256,
        )
    document["decision_gates"]["support_per_endpoint"][
        "fixed_positive_median_log_cliff_fraction_of_coupled_inclusive_maximum"
    ] = 0.25
    document["implementation_freeze"]["exit_codes"]["valid"] = False
    with pytest.raises(
        reduce.InvalidEvidence,
        match="exit-code contract differs",
    ):
        reduce._validate_preregistration(
            document,
            reducer_sha256=reducer_sha256,
        )


def test_current_preregistration_binds_current_reducer_and_test() -> None:
    prereg_path = (
        Path(__file__).resolve().parents[1]
        / "studies"
        / "phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json"
    )
    document = json.loads(prereg_path.read_text(encoding="utf-8"))
    implementation = document.get("implementation_freeze")
    if implementation is None:
        pytest.skip("implementation freeze has not yet been published")
    reducer_sha256 = hashlib.sha256(
        Path(reduce.__file__).resolve().read_bytes()
    ).hexdigest()
    reducer_test_sha256 = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()
    assert implementation["reducer_sha256"] == reducer_sha256
    assert implementation["reducer_test_sha256"] == reducer_test_sha256
    reduce._validate_preregistration(
        document,
        reducer_sha256=reducer_sha256,
    )


def test_publish_is_exactly_once_and_json_is_canonical(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    document = {"schema_version": 1, "value": 2}
    digest = reduce._publish_json_once(output, document)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert output.with_name(f"{output.name}.sha256").read_text(encoding="ascii") == (
        f"{digest}  {output.name}\n"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reduce._publish_json_once(output, document)


def test_publish_rolls_back_owned_sidecar_before_json_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    original_link = os.link
    link_calls = 0

    def fail_json_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("injected pre-commit link failure")
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(reduce.os, "link", fail_json_link)
    with pytest.raises(OSError, match="injected pre-commit"):
        reduce._publish_json_once(output, {"schema_version": 1})
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_preserves_commit_if_json_link_succeeds_then_reports_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    document = {"schema_version": 1, "value": 2}
    payload = reduce._canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    original_link = os.link
    link_calls = 0

    def link_then_fail(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal link_calls
        link_calls += 1
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )
        if link_calls == 2:
            raise OSError("injected post-commit link report")

    monkeypatch.setattr(reduce.os, "link", link_then_fail)
    with pytest.raises(OSError, match="injected post-commit"):
        reduce._publish_json_once(output, document)
    assert output.read_bytes() == payload
    assert output.with_name(f"{output.name}.sha256").read_bytes() == (
        f"{digest}  {output.name}\n".encode("ascii")
    )
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reduce._publish_json_once(output, document)


def test_publish_preserves_commit_after_post_link_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    document = {"schema_version": 1, "value": 2}
    payload = reduce._canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    original_fsync_directory = reduce._fsync_directory
    fsync_calls = 0

    def fail_second_directory_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected post-commit fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        reduce,
        "_fsync_directory",
        fail_second_directory_fsync,
    )
    with pytest.raises(OSError, match="injected post-commit"):
        reduce._publish_json_once(output, document)
    assert output.read_bytes() == payload
    assert output.with_name(f"{output.name}.sha256").read_bytes() == (
        f"{digest}  {output.name}\n".encode("ascii")
    )
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reduce._publish_json_once(output, document)


def test_publish_preserves_commit_after_post_link_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    sidecar = output.with_name(f"{output.name}.sha256")
    document = {"schema_version": 1, "value": 2}
    payload = reduce._canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    original_stable_read = reduce._stable_read
    sidecar_reads = 0

    def fail_second_sidecar_read(path: Path) -> bytes:
        nonlocal sidecar_reads
        if path == sidecar:
            sidecar_reads += 1
            if sidecar_reads == 2:
                raise reduce.InvalidEvidence(
                    "injected post-commit verification failure"
                )
        return original_stable_read(path)

    monkeypatch.setattr(reduce, "_stable_read", fail_second_sidecar_read)
    with pytest.raises(reduce.InvalidEvidence, match="injected post-commit"):
        reduce._publish_json_once(output, document)
    assert output.read_bytes() == payload
    assert sidecar.read_bytes() == (f"{digest}  {output.name}\n".encode("ascii"))
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reduce._publish_json_once(output, document)


def test_output_may_not_alias_a_missing_input(tmp_path: Path) -> None:
    missing_target = tmp_path / "target.json"
    records = ((missing_target, None, True),)
    with pytest.raises(ValueError, match="alias required inputs"):
        reduce._validate_output_paths(missing_target, records)


def test_output_may_not_alias_input_through_symlinked_parent(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="alias required inputs"):
        reduce._validate_output_paths(
            real / "same.json",
            ((alias / "same.json", None, True),),
        )


def test_manifest_rejects_array_digest_drift() -> None:
    arrays = {"value": np.arange(4, dtype="<f4")}
    manifest = {
        "value": {
            "shape": [4],
            "dtype": "float32",
            "sha256": reduce._array_sha256(arrays["value"]),
        }
    }
    reduce._validate_array_manifest(manifest, arrays, context="test")
    arrays["value"][0] = 9.0
    with pytest.raises(reduce.InvalidEvidence, match="digest differs"):
        reduce._validate_array_manifest(manifest, arrays, context="test")


def test_manifest_rejects_unhashable_dtype_categorically() -> None:
    arrays = {"value": np.arange(4, dtype="<f4")}
    manifest = {
        "value": {
            "shape": [4],
            "dtype": [],
            "sha256": reduce._array_sha256(arrays["value"]),
        }
    }
    with pytest.raises(reduce.InvalidEvidence, match="dtype differs"):
        reduce._validate_array_manifest(manifest, arrays, context="test")


def test_sealed_k10000_anchor_rehearsal() -> None:
    root = Path(__file__).resolve().parents[1]
    prediction_root = (
        root
        / "results"
        / "hqc_canonical_geometry_validity_36x5_job306114_2026-07-28"
        / "artifacts"
    )
    accepted_root = (
        root / "results" / "historical_k10000_paired_accuracy_job307260_2026-07-28"
    )
    required = (
        prediction_root / "canonical_geometry_validity_lane0.npz",
        accepted_root / "artifacts/historical_k10000_canonical_lane_A.npz",
        accepted_root
        / "stage_b_v2_license/artifacts/historical_k10000_replay_lane_A.npz",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("sealed local K=10000 rehearsal artifacts are unavailable")

    predictions, _ = reduce._validate_prediction_lanes(
        [
            prediction_root / f"canonical_geometry_validity_lane{lane}.json"
            for lane in range(4)
        ],
        [
            prediction_root / f"canonical_geometry_validity_lane{lane}.npz"
            for lane in range(4)
        ],
    )
    stage_b_npz = (
        accepted_root
        / "stage_b_v2_license/artifacts/historical_k10000_replay_lane_A.npz"
    )
    targets: dict[int, dict[str, np.ndarray]] = {}
    with np.load(stage_b_npz, allow_pickle=False) as archive:
        for ordinal, case_id, *_ in reduce.CASE_SPECS:
            prefix = f"case_{ordinal:02d}_{case_id}__"
            targets[ordinal] = {
                "selected_cell_ids_int64": archive[
                    f"{prefix}selected_cell_ids_int64"
                ].copy(),
                "physical_globals_float32": archive[
                    f"{prefix}pipeline_globals_float32"
                ][:7].copy(),
                "raw_target_pressure_float32": archive[
                    f"{prefix}raw_target_pressure_float32"
                ].copy(),
                "raw_target_wss_float32": archive[
                    f"{prefix}raw_target_wss_float32"
                ].copy(),
                "truth_pressure_float32": archive[
                    f"{prefix}truth_pressure_training_float32"
                ].copy(),
                "truth_wss_float32": archive[
                    f"{prefix}truth_wss_training_float32"
                ].copy(),
            }
    report = reduce._validate_k10000_anchor(
        canonical_json_path=(
            accepted_root / "artifacts/historical_k10000_canonical_lane_A.json"
        ),
        canonical_npz_path=(
            accepted_root / "artifacts/historical_k10000_canonical_lane_A.npz"
        ),
        stage_b_json_path=(
            accepted_root
            / "stage_b_v2_license/artifacts/historical_k10000_replay_lane_A.json"
        ),
        stage_b_npz_path=stage_b_npz,
        adjudication_path=(
            accepted_root
            / "artifacts/historical_k10000_paired_accuracy_adjudication.json"
        ),
        predictions=predictions,
        targets=targets,
    )
    assert report["exact_target_free_to_canonical_arrays"] == 288
    assert report["exact_canonical_to_stage_b_shared_control_arrays"] == 360
    assert report["exact_target_to_stage_b_id_and_global_arrays"] == 72
    assert report["exact_raw_target_arrays"] == 72
    assert report["exact_reconstructed_training_truth_arrays"] == 72
    assert report["uniform_metrics_compared"] == 72
    assert (
        report["canonical_area_pressure_cohort_mean"]
        == reduce.CANONICAL_AREA_K10_PRESSURE_MEAN
    )
    assert (
        abs(
            report["canonical_area_wss_cohort_mean"]
            - reduce.CANONICAL_AREA_K10_WSS_MEAN
        )
        <= reduce.CANONICAL_AREA_K10_MEAN_ATOL
    )
