# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered layered variable-coefficient reducer."""

from __future__ import annotations

import copy

import layered_screened_context as study
import layered_screened_context_reduce as reducer
import pytest

PARAMETERS = {
    "fixed_optical": 0,
    "scalar_correction": 11_041,
    "ordered_carrier": 11_073,
    "ordered_raw": 11_073,
}


def _report(
    arm: str,
    seed: int,
    *,
    split_values: dict[str, float],
    pair_field: float,
    contrast_error: float,
    blind_contrast: float = 0.0,
    certification: float = 1.0e-12,
) -> dict:
    trained = arm != "fixed_optical"
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
            "train_modes": list(study.TRAIN_MODES),
            "held_out_modes": list(study.HELD_OUT_MODES),
            "n_layers": study.N_LAYERS,
            "train_kappa_range": list(study.TRAIN_KAPPA_RANGE),
            "low_kappa_range": list(study.LOW_KAPPA_RANGE),
            "high_kappa_range": list(study.HIGH_KAPPA_RANGE),
            "evaluation_profiles_per_split": study.EVALUATION_PROFILES,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "order_pairs": study.ORDER_PAIRS,
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
            for split in study.SPLITS
        },
        "order_challenge": {
            "paired_field_relative_l2": pair_field,
            "contrast_relative_l2": contrast_error,
            "contrast_recovery_fraction": 1.0 - contrast_error,
            "true_contrast_relative_l2": 0.05,
            "predicted_contrast_relative_l2": blind_contrast,
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
    return {split: value for split in study.SPLITS}


def _index(
    *,
    carrier_splits: dict[str, float] | None = None,
    raw_splits: dict[str, float] | None = None,
    carrier_pair: float = 0.1,
    raw_pair: float = 0.1,
    carrier_contrast: float = 0.1,
    raw_contrast: float = 0.1,
    certification: float = 1.0e-12,
    scalar_blind_contrast: float = 0.0,
) -> dict:
    fixed = _uniform(1.0)
    scalar = _uniform(1.0)
    carrier = carrier_splits or _uniform(1.0)
    raw = raw_splits or _uniform(1.0)
    values = {
        "fixed_optical": (fixed, 1.0, 1.0, 0.0),
        "scalar_correction": (scalar, 1.0, 1.0, scalar_blind_contrast),
        "ordered_carrier": (
            carrier,
            carrier_pair,
            carrier_contrast,
            0.05,
        ),
        "ordered_raw": (raw, raw_pair, raw_contrast, 0.05),
    }
    return {
        (arm, seed): _report(
            arm,
            seed,
            split_values=values[arm][0],
            pair_field=values[arm][1],
            contrast_error=values[arm][2],
            blind_contrast=values[arm][3],
            certification=certification,
        )
        for arm in study.ARMS
        for seed in study.SEEDS
    }


def test_ordered_context_and_carrier_can_both_earn_their_claims() -> None:
    carrier = _uniform(0.9)
    carrier["ood_low_coefficient"] = 0.5
    carrier["ood_high_coefficient"] = 0.5
    decision = reducer.apply_registered_decision(
        _index(
            carrier_splits=carrier,
            raw_splits=_uniform(1.0),
            carrier_pair=0.05,
            raw_pair=0.1,
        )
    )
    assert decision["verdict"] == "ordered_context_and_carrier_earned"
    assert decision["ordered_context_earned"]
    assert decision["carrier_representation"]["representation_claim_earned"]


def test_ordered_context_only_is_reported_separately() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            carrier_splits=_uniform(1.0),
            raw_splits=_uniform(1.0),
            carrier_pair=0.1,
            raw_pair=0.1,
        )
    )
    assert decision["verdict"] == "ordered_context_only"
    assert decision["ordered_context_earned"]
    assert not decision["carrier_representation"]["representation_claim_earned"]


def test_failed_order_recovery_rejects_both_feedforward_claims() -> None:
    decision = reducer.apply_registered_decision(
        _index(
            carrier_pair=0.5,
            raw_pair=0.5,
            carrier_contrast=0.5,
            raw_contrast=0.5,
        )
    )
    assert decision["verdict"] == "feedforward_context_not_sufficient"
    assert not decision["ordered_context_earned"]


def test_numerical_sanity_preempts_scientific_verdict() -> None:
    decision = reducer.apply_registered_decision(_index(certification=1.0e-4))
    assert decision["verdict"] == "numerically_unresolved"


def test_blind_arm_contrast_marks_order_challenge_invalid() -> None:
    decision = reducer.apply_registered_decision(_index(scalar_blind_contrast=1.0e-3))
    assert decision["verdict"] == "order_challenge_invalid"


def test_validation_requires_complete_unique_factorial() -> None:
    reports = list(_index().values())
    validated = reducer.validate_reports(reports)
    assert set(validated) == {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}

    duplicate = copy.deepcopy(reports)
    duplicate.append(copy.deepcopy(reports[0]))
    with pytest.raises(ValueError, match="duplicate"):
        reducer.validate_reports(duplicate)
