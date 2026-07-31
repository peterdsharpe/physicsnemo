# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered two-limit asymptotic-carrier reducer."""

from __future__ import annotations

import copy

import drifted_screened_asymptotic as study
import drifted_screened_asymptotic_reduce as reducer
import drifted_screened_principal_part as base
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
    trained = arm != "fixed_carrier"
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "parameters": 0 if not trained else 8_834,
        "learned_singular_coefficient": base.DOUBLE_COEFFICIENT,
        "training_history": [],
        "protocol": {
            "training_applied": trained,
            "train_steps": study.TRAIN_STEPS if trained else 0,
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


def _uniform(value: float) -> dict[str, float]:
    return {split: value for split in reducer.OPERATOR_SPLITS}


def _index(
    *,
    fixed_guard: float = 1.0,
    fixed_operator: dict[str, float] | None = None,
    learned_guard: float = 1.0,
    learned_operator: dict[str, float] | None = None,
    oracle: float = 0.001,
) -> dict:
    baseline = _uniform(1.0)
    values = {
        "raw_hybrid": (1.0, baseline),
        "fixed_carrier": (fixed_guard, fixed_operator or baseline),
        "learned_carrier": (learned_guard, learned_operator or baseline),
    }
    return {
        (arm, seed): _report(
            arm,
            seed,
            guard_scale=values[arm][0],
            operator_scales=values[arm][1],
            oracle=oracle,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_learned_transition_earns_complexity() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            fixed_guard=0.9,
            fixed_operator=_uniform(0.9),
            learned_guard=0.5,
            learned_operator=_uniform(0.5),
        )
    )
    assert decision["verdict"] == "learned_transition_earned"
    assert decision["learned_vs_fixed_carrier"]["learning_complexity_earned"]


def test_passing_fixed_carrier_is_preferred_when_learning_adds_little() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            fixed_operator=_uniform(0.5),
            learned_operator=_uniform(0.5),
        )
    )
    assert decision["verdict"] == "analytic_two_limit_carrier_sufficient"


def test_passing_scaffold_without_complexity_gain_is_reported_honestly() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            fixed_operator=_uniform(0.9),
            learned_operator=_uniform(0.5),
        )
    )
    assert decision["verdict"] == "scaffold_passes_but_learning_complexity_not_earned"


def test_high_screening_only_is_a_specialist() -> None:
    one_split = _uniform(0.9)
    one_split["ood_high_screening"] = 0.5
    decision = reducer.apply_registered_decision(_index(learned_operator=one_split))
    assert decision["verdict"] == "high_screening_specialist_only"


def test_null_result_rejects_two_limit_scaffold() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "two_limit_scaffold_not_sufficient"


def test_numerical_sanity_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            fixed_operator=_uniform(0.5),
            learned_operator=_uniform(0.5),
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
