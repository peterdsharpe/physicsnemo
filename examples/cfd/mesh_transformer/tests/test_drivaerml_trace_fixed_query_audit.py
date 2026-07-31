# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the H-QC nested-source/fixed-query producer."""

import math

import numpy as np
import pytest
import torch
from drivaerml_trace_fixed_query_audit import (
    BASELINE_K,
    CASE_SPECS,
    FIXED_QUERY_K,
    FROZEN_CONTRACT,
    RESOLUTIONS,
    SCHEMA_VERSION,
    ForwardResult,
    _amplitude_ratio,
    _area_weighted_pressure_relative_l2,
    _center_diagnostic,
    _center_diagnostics_pass,
    _centered_pattern_metrics,
    _compact_explicit_cell_subset,
    _cyclic_indices,
    _full_uniform_metrics,
    _metric_relative_change,
    _native_area_reference,
    _pipeline_normal_diagnostics,
    _prediction_relative_difference,
    _relative_l2,
    _score_forward_result,
    _translation_invariant_signature,
    _validate_historical_starts,
)

from physicsnemo.mesh import Mesh


def test_historical_starts_replay_exact_seed_fork_chain():
    _validate_historical_starts()
    assert len(CASE_SPECS) == 36
    assert CASE_SPECS[0].historical_start == 14_045_027
    assert CASE_SPECS[-1].historical_start == 4_374_650


def test_nested_cyclic_prefixes_preserve_order_across_wrap():
    n_cells = 11
    start = 8
    maximum = _cyclic_indices(n_cells, start, 10)
    np.testing.assert_array_equal(
        maximum,
        np.array([8, 9, 10, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )
    for k in (2, 4, 7, 10):
        np.testing.assert_array_equal(
            _cyclic_indices(n_cells, start, k),
            maximum[:k],
        )


def test_explicit_cell_compaction_does_not_reorder_selected_cells():
    mesh = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cells=torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 4],
                [1, 4, 2],
            ]
        ),
        cell_data={
            "pMeanTrim": torch.tensor([10.0, 20.0, 30.0]),
            "wallShearStressMeanTrim": torch.arange(9.0).reshape(3, 3),
        },
    )
    selected = _compact_explicit_cell_subset(
        mesh,
        np.array([2, 0], dtype=np.int64),
        Mesh,
    )

    # Point IDs are compacted in torch.unique's sorted order, but cell rows and
    # cell_data retain the caller's [2, 0] order.
    torch.testing.assert_close(
        selected.cells,
        torch.tensor([[1, 3, 2], [0, 1, 2]]),
    )
    torch.testing.assert_close(
        selected.cell_data["pMeanTrim"],
        torch.tensor([30.0, 10.0]),
    )


def test_added_lower_point_ids_shift_local_ids_but_not_q_vertices():
    mesh = Mesh(
        points=torch.tensor(
            [
                [-3.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-3.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        ),
        # The nested first cell uses high global point IDs. The added second
        # cell introduces lower IDs, so sorted compaction renumbers the first.
        cells=torch.tensor([[3, 4, 5], [0, 1, 2]]),
        cell_data={
            "pMeanTrim": torch.tensor([10.0, 20.0]),
            "wallShearStressMeanTrim": torch.zeros(2, 3),
        },
    )
    q_only = _compact_explicit_cell_subset(mesh, np.array([0], dtype=np.int64), Mesh)
    expanded = _compact_explicit_cell_subset(
        mesh, np.array([0, 1], dtype=np.int64), Mesh
    )

    assert not torch.equal(q_only.cells[0], expanded.cells[0])
    torch.testing.assert_close(
        q_only.points[q_only.cells[0]],
        expanded.points[expanded.cells[0]],
        rtol=0.0,
        atol=0.0,
    )


def test_pipeline_normal_check_accepts_thin_offset_roundoff_without_a_flip():
    # A real run_118 triangle from the failed v1 canary. Recomputing its normal
    # after float32 centering and x/5 scaling changes a component by ~1.1e-3,
    # despite preserving the triangle's orientation.
    raw_vertices = torch.tensor(
        [
            [3.1082122, -0.7398441, 0.18303624],
            [3.1080654, -0.73968834, 0.18319201],
            [3.1084510, -0.7399537, 0.18273494],
        ],
        dtype=torch.float32,
    )
    center = torch.tensor(
        [1.5264689922332764, -0.001357387751340866, 0.11935365200042725],
        dtype=torch.float32,
    )

    def normal(vertices):
        cross = torch.linalg.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        return cross / torch.linalg.vector_norm(cross)

    native = normal(raw_vertices)
    transformed_vertices = (raw_vertices - center) / 5.0
    pipeline = normal(transformed_vertices)
    mesh = Mesh(
        points=transformed_vertices,
        cells=torch.tensor([[0, 1, 2]]),
        cell_data={"normals": pipeline[None]},
    )

    observed, diagnostics = _pipeline_normal_diagnostics(
        native[None].numpy(), mesh, "thin-offset"
    )

    assert torch.max(torch.abs(native - pipeline)).item() > 1.0e-3
    np.testing.assert_array_equal(observed, pipeline[None].numpy())
    assert diagnostics["max_geometry_reconstruction_abs_error"] == 0.0
    assert diagnostics["min_native_dot"] > 0.999


def test_pipeline_normal_check_rejects_reversed_transformed_winding():
    raw_vertices = torch.tensor(
        [
            [3.1082122, -0.7398441, 0.18303624],
            [3.1080654, -0.73968834, 0.18319201],
            [3.1084510, -0.7399537, 0.18273494],
        ],
        dtype=torch.float32,
    )
    center = torch.tensor(
        [1.5264689922332764, -0.001357387751340866, 0.11935365200042725],
        dtype=torch.float32,
    )
    transformed_vertices = (raw_vertices - center) / 5.0
    native_cross = torch.linalg.cross(
        raw_vertices[1] - raw_vertices[0], raw_vertices[2] - raw_vertices[0]
    )
    native = native_cross / torch.linalg.vector_norm(native_cross)
    cells = torch.tensor([[0, 2, 1]])
    reversed_cross = torch.linalg.cross(
        transformed_vertices[2] - transformed_vertices[0],
        transformed_vertices[1] - transformed_vertices[0],
    )
    reversed_normal = reversed_cross / torch.linalg.vector_norm(reversed_cross)
    mesh = Mesh(
        points=transformed_vertices,
        cells=cells,
        cell_data={"normals": reversed_normal[None]},
    )

    with pytest.raises(ValueError, match="orientation disagrees with native winding"):
        _pipeline_normal_diagnostics(native[None].numpy(), mesh, "reversed")


def test_pressure_metric_matches_frozen_legacy_formula():
    prediction = np.array([2.0, 0.0, 3.0])
    truth = np.array([1.0, 2.0, 2.0])
    expected = math.sqrt(6.0) / (3.0 + 1.0e-8)
    assert _relative_l2(prediction, truth) == pytest.approx(expected)


def test_centered_pattern_and_amplitude_have_distinct_jobs():
    truth = np.array([-1.0, 0.0, 1.0])
    prediction = 4.0 * truth + 7.0
    metrics = _centered_pattern_metrics(prediction, truth)
    assert metrics["signed_centered_correlation"] == pytest.approx(1.0)
    assert metrics["positive_gain_pattern_error"] < 3.0e-8
    assert _amplitude_ratio(4.0 * truth, truth) == pytest.approx(4.0)


def test_centered_pattern_metrics_penalize_sign_inversion():
    truth = np.array([-1.0, 0.0, 1.0])
    metrics = _centered_pattern_metrics(-truth, truth)
    assert metrics["signed_centered_correlation"] == pytest.approx(-1.0)
    assert metrics["positive_gain_pattern_error"] == pytest.approx(1.0)


def test_area_weighting_uses_native_triangle_measure():
    truth = np.ones(2)
    prediction = np.array([2.0, 1.0])
    uniform = _relative_l2(prediction, truth)
    weighted = _area_weighted_pressure_relative_l2(
        prediction,
        truth,
        np.array([100.0, 1.0]),
    )
    assert weighted > uniform
    assert weighted == pytest.approx(math.sqrt(100.0 / 101.0) / (1.0 + 1.0e-8))


def test_unequal_supports_compare_mean_cell_area_not_total_area():
    fixed_q = _native_area_reference(np.full(2_500, 0.9), "Q")
    s10k = _native_area_reference(np.ones(10_000), "S10k")

    assert fixed_q["native_area"] / s10k["native_area"] == pytest.approx(0.225)
    assert fixed_q["mean_native_cell_area"] / s10k[
        "mean_native_cell_area"
    ] == pytest.approx(0.9)


def test_full_bundle_uses_frobenius_wss_and_predicted_normal_energy():
    pressure = np.array([1.0, -1.0])
    truth_pressure = np.array([0.5, -0.5])
    prediction_wss = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    truth_wss = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    areas = np.array([1.0, 2.0])

    metrics = _full_uniform_metrics(
        pressure,
        truth_pressure,
        prediction_wss,
        truth_wss,
        normals,
        areas,
        n_master_cells=10,
    )

    expected_wss = math.sqrt(2.0) / (math.sqrt(2.0) + 1.0e-8)
    expected_normal = 1.0 / (math.sqrt(6.0) + 1.0e-8)
    assert metrics["wss_frobenius_relative_l2"] == pytest.approx(expected_wss)
    assert metrics["wss_normal_energy"] == pytest.approx(expected_normal)
    assert "scaled_subset_pressure_force_relative_error" in metrics
    assert "ht_pressure_force_relative_error" not in metrics
    assert all(math.isfinite(value) and value >= 0.0 for value in metrics.values())


def test_forward_result_scoring_wires_predicted_and_true_wss_once():
    truth_pressure = np.linspace(1.0, 2.0, FIXED_QUERY_K, dtype=np.float32)
    pressure = truth_pressure + np.linspace(-0.1, 0.1, FIXED_QUERY_K, dtype=np.float32)
    truth_wss = np.ones((FIXED_QUERY_K, 3), dtype=np.float32)
    wss = 0.5 * truth_wss
    normals = np.zeros((FIXED_QUERY_K, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    areas = np.ones(FIXED_QUERY_K, dtype=np.float64)
    result = ForwardResult(
        pressure=pressure,
        wss=wss,
        truth_pressure=truth_pressure,
        truth_wss=truth_wss,
        query_points=np.zeros((FIXED_QUERY_K, 3), dtype=np.float32),
        boundary_cells=np.zeros((FIXED_QUERY_K, 3), dtype=np.int64),
        boundary_normals=normals,
    )

    coupled, fixed_q = _score_forward_result(
        result,
        normals,
        areas,
        n_master_cells=FIXED_QUERY_K,
    )

    assert coupled == fixed_q
    assert coupled["uniform"]["wss_frobenius_relative_l2"] == pytest.approx(0.5)


def test_center_diagnostic_is_zero_for_translation_invariant_predictions():
    pressure = np.array([1.0, 2.0, 4.0])
    metrics = {
        "uniform": {"pressure_relative_l2": 0.25},
        "area_weighted": {"pressure_relative_l2": 0.5},
    }
    row = _center_diagnostic(pressure, pressure.copy(), metrics, metrics)
    assert row == {
        "pressure_prediction_relative_l2_difference": 0.0,
        "uniform_pressure_error_relative_change": 0.0,
        "area_pressure_error_relative_change": 0.0,
    }
    assert _prediction_relative_difference(pressure, pressure) == 0.0
    assert _metric_relative_change(0.5, 0.5) == 0.0


def test_center_gate_checks_fixed_q_separately_from_coupled():
    passing = {
        "pressure_prediction_relative_l2_difference": 0.0,
        "uniform_pressure_error_relative_change": 0.0,
        "area_pressure_error_relative_change": 0.0,
    }
    fixed_q_failure = {**passing, "uniform_pressure_error_relative_change": 2.0e-3}
    assert not _center_diagnostics_pass(
        {"40000": {"coupled": passing, "fixed_q": fixed_q_failure}}
    )


def test_canonical_signature_is_exactly_translation_invariant():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    components = {"raw_cell_ids_sha256_int64": "a" * 64}
    signature = _translation_invariant_signature(
        query_points=points,
        compacted_cells=cells,
        identity_components=components,
    )
    translated = _translation_invariant_signature(
        query_points=points + np.array([7.0, -2.0, 4.0], dtype=np.float32),
        compacted_cells=cells,
        identity_components=components,
    )
    assert signature == translated


def test_frozen_contract_exposes_fixed_q_and_all_resolution_arms():
    assert SCHEMA_VERSION == 2
    assert FROZEN_CONTRACT["baseline_k"] == BASELINE_K
    assert FROZEN_CONTRACT["fixed_query_k"] == FIXED_QUERY_K
    assert FROZEN_CONTRACT["resolutions"] == list(RESOLUTIONS)
    assert FROZEN_CONTRACT["center_use_area_weighting"] is False
    assert FROZEN_CONTRACT["precision"] == "bfloat16"
    assert FROZEN_CONTRACT["inference_compile"] is False
    assert FROZEN_CONTRACT["norm_stats_filename"] == "norm_stats.pt"
    assert FROZEN_CONTRACT["archived_pipeline_normal_abs_tolerance"] == 2.0e-6
    assert FROZEN_CONTRACT["pipeline_normal_geometry_abs_tolerance"] == 5.0e-7
    assert FROZEN_CONTRACT["pipeline_normal_unit_abs_tolerance"] == 5.0e-6
    assert (
        FROZEN_CONTRACT["norm_stats_sha256"]
        == "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
    )
