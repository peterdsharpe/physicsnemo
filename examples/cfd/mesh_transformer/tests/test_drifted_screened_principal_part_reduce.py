# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered principal-part reducer."""

from __future__ import annotations

import copy

import drifted_screened_principal_part as study
import drifted_screened_principal_part_reduce as reducer
import pytest


def _report(
    arm: str,
    seed: int,
    *,
    fixed_scale: float,
    flexible_scale: float,
    operator_fixed_scale: float | None = None,
    operator_flexible_scale: float | None = None,
    oracle: float = 0.001,
    quadrature: float = 1.0e-5,
) -> dict:
    is_fixed = arm == "fixed_principal"
    near_value = fixed_scale if is_fixed else flexible_scale
    operator_value = (
        (operator_fixed_scale if operator_fixed_scale is not None else fixed_scale)
        if is_fixed
        else (
            operator_flexible_scale
            if operator_flexible_scale is not None
            else flexible_scale
        )
    )
    pde = {}
    for split in study.PDE_SPLIT_ORDER:
        value = operator_value if split in reducer.OPERATOR_SPLITS else near_value
        if split == "in_distribution":
            value = fixed_scale if is_fixed else flexible_scale
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
                    "learned_field_relative_l2": near_value,
                    "exact_trace_residual_relative_l2": near_value,
                    "oracle_field_relative_l2": oracle,
                }
                for resolution in study.RESOLUTIONS
            }
        }
        for split in study.RESOLUTION_SPLITS
    }
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "parameters": 1,
        "learned_singular_coefficient": None,
        "protocol": {
            "train_steps": study.TRAIN_STEPS,
            "train_batch_size": study.TRAIN_BATCH_SIZE,
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
        "kernel_evaluation": {
            "near_singular": {"scaled_kernel_relative_l2": near_value}
        },
        "source": {"relevant_source_sha256": "same"},
    }


def _index(
    *,
    fixed_scale: float = 0.001,
    flexible_scale: float = 0.003,
    operator_fixed_scale: float | None = None,
    operator_flexible_scale: float | None = None,
    oracle: float = 0.001,
    quadrature: float = 1.0e-5,
) -> dict:
    return {
        (arm, seed): _report(
            arm,
            seed,
            fixed_scale=fixed_scale,
            flexible_scale=flexible_scale,
            operator_fixed_scale=operator_fixed_scale,
            operator_flexible_scale=operator_flexible_scale,
            oracle=oracle,
            quadrature=quadrature,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_operator_transfer_pass_branch() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "fixed_principal_improves_operator_transfer"
    assert decision["principal_part_earned"] is True
    assert decision["operator_transfer_earned"] is True


def test_near_trace_only_branch() -> None:
    decision = reducer.apply_registered_decision(
        _index(operator_fixed_scale=0.001, operator_flexible_scale=0.0012)
    )
    assert decision["verdict"] == "fixed_principal_improves_near_trace_only"
    assert decision["principal_part_earned"] is True
    assert decision["operator_transfer_earned"] is False


def test_hard_constraint_not_earned_branch() -> None:
    decision = reducer.apply_registered_decision(
        _index(fixed_scale=0.001, flexible_scale=0.0015)
    )
    assert decision["verdict"] == "hard_fixed_coefficient_not_earned"
    assert decision["id_noninferior"] is True
    assert decision["near_boundary_pass"] is False


def test_interpolation_harm_branch() -> None:
    decision = reducer.apply_registered_decision(
        _index(fixed_scale=0.002, flexible_scale=0.001)
    )
    assert decision["verdict"] == "fixed_principal_harms_interpolation"
    assert decision["id_noninferior"] is False


def test_numerical_sanity_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(_index(oracle=0.03))
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
