# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered path-ordered generator reducer."""

from __future__ import annotations

import copy

import layered_screened_context as base
import layered_screened_generator as study
import layered_screened_generator_reduce as reducer
import pytest

PARAMETERS = {
    "scalar_correction": 11_041,
    "ordered_raw": 11_073,
    "sorted_generator": 4_417,
    "path_generator": 4_417,
    "analytic_path": 0,
}


def _report(
    arm: str,
    seed: int,
    *,
    split_values: dict[str, float],
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
            "train_modes": list(base.TRAIN_MODES),
            "held_out_modes": list(base.HELD_OUT_MODES),
            "n_layers": base.N_LAYERS,
            "train_kappa_range": list(base.TRAIN_KAPPA_RANGE),
            "low_kappa_range": list(base.LOW_KAPPA_RANGE),
            "high_kappa_range": list(base.HIGH_KAPPA_RANGE),
            "evaluation_profiles_per_split": study.EVALUATION_PROFILES,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "order_pairs": study.ORDER_PAIRS,
            "generator_width": study.GENERATOR_WIDTH,
        },
        "split_evaluation": {
            split: {
                "metrics": {
                    "field_relative_l2": {
                        "mean": split_values[split],
                        "median": split_values[split],
                        "maximum": split_values[split],
                    },
                    "log_amplitude_rmse": split_values[split],
                }
            }
            for split in base.SPLITS
        },
        "order_challenge": {
            "paired_field_relative_l2": pair_field,
            "contrast_relative_l2": contrast_error,
            "contrast_recovery_fraction": 1.0 - contrast_error,
            "true_contrast_relative_l2": 0.05,
            "predicted_contrast_relative_l2": predicted_contrast,
            "scalar_input_max_abs_difference": 0.0,
        },
        "reference_certification": {
            "constant_profile_max_abs_error": certification,
            "transfer_determinant_max_abs_error": certification,
            "boundary_max_abs_error": certification,
        },
        "source": {"relevant_source_sha256": "same"},
    }


def _uniform(value: float) -> dict[str, float]:
    return {split: value for split in base.SPLITS}


def _index(
    *,
    path_splits: dict[str, float] | None = None,
    path_pair: float = 0.05,
    path_contrast: float = 0.1,
    sorted_predicted_contrast: float = 0.0,
    certification: float = 1.0e-12,
    oracle_error: float = 0.0,
) -> dict:
    values = {
        "scalar_correction": (_uniform(1.0), 1.0, 1.0, 0.0),
        "ordered_raw": (_uniform(1.0), 0.1, 0.5, 0.02),
        "sorted_generator": (_uniform(0.8), 0.5, 1.0, sorted_predicted_contrast),
        "path_generator": (
            path_splits or _uniform(0.5),
            path_pair,
            path_contrast,
            0.05,
        ),
        "analytic_path": (
            _uniform(oracle_error),
            oracle_error,
            oracle_error,
            0.05,
        ),
    }
    return {
        (arm, seed): _report(
            arm,
            seed,
            split_values=values[arm][0],
            pair_field=values[arm][1],
            contrast_error=values[arm][2],
            predicted_contrast=values[arm][3],
            certification=certification,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_path_ordered_generator_can_earn_both_claims() -> None:
    decision = reducer.apply_registered_decision(_index())
    assert decision["verdict"] == "path_ordered_generator_earned"
    assert decision["composition"]["composition_claim_earned"]
    assert decision["local_law_transfer"]["local_law_transfer_earned"]


def test_composition_can_pass_without_local_law_transfer() -> None:
    path = _uniform(0.8)
    decision = reducer.apply_registered_decision(_index(path_splits=path))
    assert decision["verdict"] == "composition_earned_local_law_not_transferable"
    assert decision["composition"]["composition_claim_earned"]
    assert not decision["local_law_transfer"]["local_law_transfer_earned"]


def test_failed_composition_is_attributed_to_field_supervision() -> None:
    decision = reducer.apply_registered_decision(
        _index(path_pair=0.1, path_contrast=0.5)
    )
    assert decision["verdict"] == "field_supervision_does_not_identify_generator"
    assert not decision["composition"]["composition_claim_earned"]


def test_sorted_generator_contrast_invalidates_pair_instrument() -> None:
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
