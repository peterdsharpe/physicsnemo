# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preregistered solved-density decision rule."""

from __future__ import annotations

import copy

import laplace_solved_density_ceiling as study
import laplace_solved_density_ceiling_reduce as reduce
import pytest


def _baseline(error: float = 0.04) -> dict:
    return {
        "study": reduce.BASELINE_STUDY,
        "registered_seeds": list(reduce.SEEDS),
        "registered_arms": {
            "pure": {},
            "gate_only": {},
            "contraction_only": {},
            "full": {},
        },
        "protocol_fingerprint": reduce.BASELINE_PROTOCOL_FINGERPRINT,
        "source_fingerprint": reduce.BASELINE_SOURCE_FINGERPRINT,
        "splits": {
            split: {
                "arm_seed_relative_l2_means": {
                    "pure": {str(seed): error for seed in reduce.SEEDS}
                }
            }
            for split in reduce.SPLITS
        },
    }


def _oracle(error: float = 0.01, trace: float = 1.0e-14) -> dict:
    return {
        "study": study.STUDY,
        "evaluation_protocol": study.evaluation_protocol(study.evaluation_config()),
        "registered_parameters": 0,
        "trainable_parameters": 0,
        "environment": {"device": "cuda"},
        "evaluation": {
            "accuracy_dtype": "float32",
            "splits": {split: {"relative_l2_mean": error} for split in reduce.SPLITS},
            "boundary_trace": {
                "dtype": "float64",
                "splits": {
                    split: {"relative_l2_mean": trace} for split in reduce.SPLITS
                },
            },
            "resolution": {"64": {}, "128": {}, "256": {}},
        },
    }


def test_density_inference_verdict() -> None:
    result = reduce.reduce_reports(_oracle(error=0.01), _baseline(error=0.04))

    assert result["verdict"] == "density_inference_is_principal_bottleneck"
    assert result["decision"]["density_inference"]["passed"] is True
    assert result["decision"]["boundary_discretization"]["passed"] is False
    assert result["ratios"]["mixed_geometry_modes"]["role"] == "exploratory"
    assert result["ratios"]["interpolation"][
        "oracle_over_pure_geometric_mean"
    ] == pytest.approx(0.25)


def test_boundary_discretization_verdict() -> None:
    result = reduce.reduce_reports(_oracle(error=0.036), _baseline(error=0.04))

    assert result["verdict"] == "finite_boundary_discretization_is_principal_bottleneck"
    assert result["decision"]["density_inference"]["passed"] is False
    assert result["decision"]["boundary_discretization"]["passed"] is True


def test_intermediate_or_failed_trace_result_is_split_dependent() -> None:
    oracle = _oracle(error=0.01)
    oracle["evaluation"]["splits"]["unseen_boundary_frequencies"][
        "relative_l2_mean"
    ] = 0.03
    oracle["evaluation"]["boundary_trace"]["splits"]["unseen_geometry_modes"][
        "relative_l2_mean"
    ] = 1.0e-8

    result = reduce.reduce_reports(oracle, _baseline(error=0.04))

    assert result["verdict"] == "split_dependent"
    assert result["decision"]["density_inference"]["passed"] is False
    assert result["decision"]["boundary_discretization"]["passed"] is False


def test_rejects_protocol_or_baseline_drift() -> None:
    oracle = _oracle()
    oracle["evaluation_protocol"]["cases_per_split"] = 63
    with pytest.raises(ValueError, match="protocol mismatch"):
        reduce.reduce_reports(oracle, _baseline())

    baseline = _baseline()
    baseline["source_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        reduce.reduce_reports(_oracle(), baseline)


def test_rejects_non_cuda_or_nonfinite_metrics() -> None:
    oracle = _oracle()
    oracle["environment"]["device"] = "cpu"
    with pytest.raises(ValueError, match="generated on CUDA"):
        reduce.reduce_reports(oracle, _baseline())

    oracle = copy.deepcopy(_oracle())
    oracle["evaluation"]["splits"]["interpolation"]["relative_l2_mean"] = float("nan")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reduce.reduce_reports(oracle, _baseline())
