# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered supervision reducer."""

from __future__ import annotations

import copy

import drifted_screened_principal_part as base
import drifted_screened_supervision as study
import drifted_screened_supervision_reduce as reducer
import pytest


def _report(
    arm: str,
    seed: int,
    *,
    guard_scale: float,
    operator_scale: float,
    boundary_scale: float | None = None,
    oracle: float = 0.001,
    quadrature: float = 1.0e-5,
) -> dict:
    pde = {}
    for split in base.PDE_SPLIT_ORDER:
        value = operator_scale if split in reducer.OPERATOR_SPLITS else guard_scale
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
    spectrum_value = boundary_scale if boundary_scale is not None else guard_scale
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
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
                "learned_field_relative_l2": {"mean": spectrum_value},
                "exact_trace_residual_relative_l2": {"mean": spectrum_value},
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
    solution_guard: float = 1.0,
    solution_operator: float = 1.0,
    solution_boundary: float | None = None,
    hybrid_guard: float = 1.0,
    hybrid_operator: float = 1.0,
    hybrid_boundary: float | None = None,
    oracle: float = 0.001,
) -> dict:
    scales = {
        "pointwise": (1.0, 1.0, 1.0),
        "solution": (
            solution_guard,
            solution_operator,
            solution_boundary,
        ),
        "hybrid": (
            hybrid_guard,
            hybrid_operator,
            hybrid_boundary,
        ),
    }
    return {
        (arm, seed): _report(
            arm,
            seed,
            guard_scale=scales[arm][0],
            operator_scale=scales[arm][1],
            boundary_scale=scales[arm][2],
            oracle=oracle,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_hybrid_only_pass_means_losses_are_complementary() -> None:
    decision = reducer.apply_registered_decision(
        _index(solution_operator=0.9, hybrid_operator=0.5)
    )
    assert (
        decision["verdict"]
        == "kernel_identification_and_solution_alignment_are_complementary"
    )
    assert (
        decision["comparisons_to_pointwise"]["hybrid"]["supervision_claim_earned"]
        is True
    )


def test_solution_pass_means_alignment_improves_transfer() -> None:
    decision = reducer.apply_registered_decision(
        _index(solution_operator=0.5, hybrid_operator=0.9)
    )
    assert decision["verdict"] == "solution_alignment_improves_operator_transfer"
    assert (
        decision["comparisons_to_pointwise"]["solution"]["operator_splits_passed"] == 3
    )


def test_solution_operator_gain_with_boundary_loss_is_overfit() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            solution_operator=0.5,
            solution_boundary=1.6,
            hybrid_operator=0.9,
        )
    )
    assert decision["verdict"] == "solution_only_overfits_boundary_distribution"
    assert (
        decision["comparisons_to_pointwise"]["solution"][
            "boundary_distribution_overfit"
        ]
        is True
    )


def test_null_result_rejects_supervision_mismatch_as_principal() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "supervision_mismatch_not_principal"


def test_numerical_sanity_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(
        _index(solution_operator=0.5, oracle=0.03)
    )
    assert decision["verdict"] == "numerically_unresolved"
    assert decision["numerical_sanity"] is False


def test_validation_requires_complete_unique_factorial() -> None:
    reports = list(_index().values())
    validated = reducer.validate_reports(reports)
    assert set(validated) == {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}

    duplicate = copy.deepcopy(reports)
    duplicate.append(copy.deepcopy(reports[0]))
    with pytest.raises(ValueError, match="duplicate"):
        reducer.validate_reports(duplicate)
