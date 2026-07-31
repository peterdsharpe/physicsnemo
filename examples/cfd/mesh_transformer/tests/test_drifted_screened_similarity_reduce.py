# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered similarity-coordinate reducer."""

from __future__ import annotations

import copy

import drifted_screened_principal_part as base
import drifted_screened_similarity as study
import drifted_screened_similarity_reduce as reducer
import pytest


def _report(
    arm: str,
    seed: int,
    *,
    guard_scale: float,
    operator_scales: dict[str, float],
    oracle: float = 0.001,
    quadrature: float = 1.0e-5,
) -> dict:
    feature_system, loss = study.ARM_SPECS[arm]
    pde = {}
    for split in base.PDE_SPLIT_ORDER:
        value = operator_scales.get(split, guard_scale)
        pde[split] = {
            "metrics": {
                "learned_field_relative_l2": {"mean": value},
                "exact_trace_residual_relative_l2": {"mean": value},
                "oracle_field_relative_l2": {"mean": oracle},
            },
            "cases": [{"quadrature_relative_frobenius": quadrature}],
        }
    resolution = {
        split: {
            "resolutions": {
                str(resolution): {
                    "learned_field_relative_l2": guard_scale,
                    "exact_trace_residual_relative_l2": guard_scale,
                    "oracle_field_relative_l2": oracle,
                }
                for resolution in study.RESOLUTIONS
            }
        }
        for split in base.RESOLUTION_SPLITS
    }
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "feature_system": feature_system,
        "loss": loss,
        "parameters": 8_834,
        "learned_singular_coefficient": 0.159,
        "protocol": {
            "train_steps": study.TRAIN_STEPS,
            "train_boundary_points": study.TRAIN_BOUNDARY_POINTS,
            "train_query_points": study.TRAIN_QUERY_POINTS,
            "train_quadrature_order_per_half_panel": study.TRAIN_QUADRATURE_ORDER,
            "train_solution_modes": list(study.TRAIN_SOLUTION_MODES),
            "held_out_solution_modes": list(study.HELD_OUT_SOLUTION_MODES),
            "evaluation_cases_per_split": study.EVALUATION_CASES,
            "evaluation_boundary_points": study.EVALUATION_BOUNDARY_POINTS,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "resolution_cases_per_split": study.RESOLUTION_CASES,
            "resolutions": list(study.RESOLUTIONS),
            "quadrature_order_per_half_panel": study.QUADRATURE_ORDER,
            "check_quadrature_order_per_half_panel": study.CHECK_QUADRATURE_ORDER,
            "kernel_evaluation_pairs_per_split": study.KERNEL_EVALUATION_PAIRS,
        },
        "pde_evaluation": pde,
        "resolution_evaluation": resolution,
        "boundary_spectrum_evaluation": {
            "solution_modes": list(study.HELD_OUT_SOLUTION_MODES),
            "metrics": {
                "learned_field_relative_l2": {"mean": guard_scale},
                "exact_trace_residual_relative_l2": {"mean": guard_scale},
                "oracle_field_relative_l2": {"mean": oracle},
            },
            "cases": [{"quadrature_relative_frobenius": quadrature}],
        },
        "kernel_evaluation": {
            "near_singular": {"scaled_kernel_relative_l2": guard_scale}
        },
        "source": {"relevant_source_sha256": "same"},
    }


def _index(
    *,
    pointwise_guard: float = 1.0,
    pointwise_operator_scales: dict[str, float] | None = None,
    hybrid_guard: float = 1.0,
    hybrid_operator_scales: dict[str, float] | None = None,
    oracle: float = 0.001,
) -> dict:
    baseline_scales = {split: 1.0 for split in reducer.OPERATOR_SPLITS}
    candidate_values = {
        "similarity_pointwise": (
            pointwise_guard,
            pointwise_operator_scales or baseline_scales,
        ),
        "similarity_hybrid": (
            hybrid_guard,
            hybrid_operator_scales or baseline_scales,
        ),
    }
    return {
        (arm, seed): _report(
            arm,
            seed,
            guard_scale=(candidate_values[arm][0] if arm in candidate_values else 1.0),
            operator_scales=(
                candidate_values[arm][1] if arm in candidate_values else baseline_scales
            ),
            oracle=oracle,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def _uniform_operator_scale(value: float) -> dict[str, float]:
    return {split: value for split in reducer.OPERATOR_SPLITS}


def test_pass_under_both_losses_identifies_raw_parameterization() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            pointwise_operator_scales=_uniform_operator_scale(0.5),
            hybrid_operator_scales=_uniform_operator_scale(0.5),
        )
    )
    assert decision["verdict"] == "raw_parameterization_is_principal_cause"


def test_hybrid_only_pass_identifies_complementarity() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            pointwise_operator_scales=_uniform_operator_scale(0.9),
            hybrid_operator_scales=_uniform_operator_scale(0.5),
        )
    )
    assert decision["verdict"] == "coordinates_and_task_alignment_are_complementary"


def test_pointwise_only_pass_prefers_pointwise_identification() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            pointwise_operator_scales=_uniform_operator_scale(0.5),
            hybrid_operator_scales=_uniform_operator_scale(0.9),
        )
    )
    assert (
        decision["verdict"]
        == "similarity_coordinates_help_pointwise_identification_only"
    )


def test_one_extreme_does_not_establish_operator_transfer() -> None:
    one_split = _uniform_operator_scale(0.9)
    one_split["ood_high_screening"] = 0.5
    decision = reducer.apply_registered_decision(
        _index(pointwise_operator_scales=one_split)
    )
    assert decision["verdict"] == "similarity_coordinates_help_one_operator_extreme"
    assert decision["comparisons"]["pointwise"]["operator_splits_passed"] == 1


def test_null_result_rejects_coordinate_mismatch_as_principal() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "similarity_coordinates_not_principal"


def test_numerical_sanity_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            pointwise_operator_scales=_uniform_operator_scale(0.5),
            hybrid_operator_scales=_uniform_operator_scale(0.5),
            oracle=0.03,
        )
    )
    assert decision["verdict"] == "numerically_unresolved"


def test_validation_requires_complete_unique_factorial() -> None:
    reports = list(_index().values())
    validated = reducer.validate_reports(reports)
    assert set(validated) == {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}

    duplicate = copy.deepcopy(reports)
    duplicate.append(copy.deepcopy(reports[0]))
    with pytest.raises(ValueError, match="duplicate"):
        reducer.validate_reports(duplicate)
