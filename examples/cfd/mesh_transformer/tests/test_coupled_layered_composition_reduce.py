# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered coupled-composition reducer."""

from __future__ import annotations

import copy

import coupled_layered_composition as study
import coupled_layered_composition_reduce as reducer
import pytest

PARAMETERS = {
    "ordered_pool": 4_152,
    "sorted_generator": 4_417,
    "path_generator": 4_417,
    "analytic_path": 0,
}


def _report(
    arm: str,
    seed: int,
    *,
    operator_values: dict[str, float],
    cross_values: dict[str, float],
    pair_field: float,
    contrast_error: float,
    predicted_contrast: float,
    certification: float = 1.0e-12,
) -> dict:
    trained = arm != "analytic_path"
    return {
        "study": study.STUDY,
        "arm": arm,
        "seed": seed,
        "parameters": PARAMETERS[arm],
        "training_history": [],
        "protocol": {
            "training_applied": trained,
            "train_steps": study.TRAIN_STEPS if trained else 0,
            "train_batch_size": study.TRAIN_BATCH_SIZE,
            "train_layers": study.TRAIN_LAYERS,
            "doubled_layers": study.DOUBLED_LAYERS,
            "train_twists": list(study.TRAIN_TWISTS),
            "held_out_twists": list(study.HELD_OUT_TWISTS),
            "evaluation_profiles_per_split": study.EVALUATION_PROFILES,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "order_pairs": study.ORDER_PAIRS,
            "token_width": study.TOKEN_WIDTH,
            "head_width": study.HEAD_WIDTH,
            "generator_width": study.GENERATOR_WIDTH,
        },
        "split_evaluation": {
            split: {
                "metrics": {
                    "operator_relative_l2": {
                        "mean": operator_values[split],
                        "median": operator_values[split],
                        "maximum": operator_values[split],
                    },
                    "cross_channel_relative_l2": {
                        "mean": cross_values[split],
                        "median": cross_values[split],
                        "maximum": cross_values[split],
                    },
                    "boundary_max_abs_error": certification,
                }
            }
            for split in study.SPLITS
        },
        "order_challenge": {
            "paired_field_relative_l2": pair_field,
            "contrast_relative_l2": contrast_error,
            "contrast_recovery_fraction": 1.0 - contrast_error,
            "predicted_contrast_relative_l2": predicted_contrast,
            "true_contrast_relative_l2": 0.05,
            "multiset_max_abs_difference": 0.0,
        },
        "local_map_evaluation": (
            {"relative_frobenius": 0.1, "maximum_abs_error": 0.1}
            if arm in study.GENERATOR_ARMS
            else None
        ),
        "reference_certification": {
            "boundary_max_abs_error": certification,
            "transfer_determinant_max_abs_error": certification,
            "rotation_covariance_max_abs_error": certification,
        },
        "source": {"relevant_source_sha256": "same"},
    }


def _uniform(value: float) -> dict[str, float]:
    return {split: value for split in study.SPLITS}


def _index(
    *,
    path_operator: dict[str, float] | None = None,
    path_cross: dict[str, float] | None = None,
    path_pair: float = 0.4,
    path_contrast: float = 0.1,
    sorted_predicted_contrast: float = 0.0,
    certification: float = 1.0e-12,
    oracle_error: float = 0.0,
) -> dict:
    values = {
        "ordered_pool": (_uniform(1.0), _uniform(1.0), 1.0, 0.6, 0.02),
        "sorted_generator": (_uniform(0.8), _uniform(0.8), 0.8, 1.0, 0.0),
        "path_generator": (
            path_operator or _uniform(0.8),
            path_cross or _uniform(0.4),
            path_pair,
            path_contrast,
            0.05,
        ),
        "analytic_path": (
            _uniform(oracle_error),
            _uniform(oracle_error),
            oracle_error,
            oracle_error,
            0.05,
        ),
    }
    values["sorted_generator"] = (
        values["sorted_generator"][0],
        values["sorted_generator"][1],
        values["sorted_generator"][2],
        values["sorted_generator"][3],
        sorted_predicted_contrast,
    )
    return {
        (arm, seed): _report(
            arm,
            seed,
            operator_values=values[arm][0],
            cross_values=values[arm][1],
            pair_field=values[arm][2],
            contrast_error=values[arm][3],
            predicted_contrast=values[arm][4],
            certification=certification,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_coupled_path_composition_can_earn_all_claims() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "coupled_path_composition_earned"
    assert decision["composition"]["composition_claim_earned"]
    assert decision["breadth_transfer"]["breadth_transfer_earned"]


def test_composition_can_pass_without_breadth_transfer() -> None:
    path_cross = _uniform(0.4)
    path_cross["doubled_layers"] = 0.8
    decision = reducer.apply_registered_decision(_index(path_cross=path_cross))
    assert decision["verdict"] == "order_only_not_breadth"
    assert decision["composition"]["composition_claim_earned"]
    assert not decision["breadth_transfer"]["breadth_transfer_earned"]


def test_failed_order_recovery_rejects_composition() -> None:
    decision = reducer.apply_registered_decision(_index(path_contrast=0.4))
    assert decision["verdict"] == "coupled_composition_not_earned"
    assert not decision["composition"]["composition_claim_earned"]


def test_sorted_contrast_invalidates_order_instrument() -> None:
    decision = reducer.apply_registered_decision(
        _index(sorted_predicted_contrast=1.0e-3)
    )
    assert decision["verdict"] == "order_challenge_invalid"


def test_oracle_error_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(_index(oracle_error=1.0e-4))
    assert decision["verdict"] == "numerically_unresolved"


def test_validation_requires_complete_unique_factorial() -> None:
    reports = list(_index().values())
    validated = reducer.validate_reports(reports)
    assert set(validated) == {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}

    duplicate = copy.deepcopy(reports)
    duplicate.append(copy.deepcopy(reports[0]))
    with pytest.raises(ValueError, match="duplicate"):
        reducer.validate_reports(duplicate)


def test_registered_seed_count_is_enforced() -> None:
    index = _index()
    for seed in study.SEEDS[:2]:
        index[("path_generator", seed)]["order_challenge"]["contrast_relative_l2"] = 0.3
    decision = reducer.apply_registered_decision(index)
    assert decision["composition"]["contrast_pass_seed_count"] == 3
    assert decision["verdict"] == "coupled_composition_not_earned"
