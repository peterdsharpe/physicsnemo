# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the pre-registered heads-versus-rank probe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import heads_rank_study  # noqa: E402


def test_every_arm_holds_the_fixed_total_score_capacity() -> None:
    """The design's invariant: H * (R0 + 2*R1) identical across arms."""

    capacities = {
        arm.key: heads_rank_study.score_capacity(arm) for arm in heads_rank_study.ARMS
    }
    assert set(capacities.values()) == {heads_rank_study.TOTAL_SCORE_CAPACITY}


def test_arm_parameter_counts_match_within_one_percent() -> None:
    """The probe is capacity-matched: parameters within 1% of the reference."""

    table = heads_rank_study.parameter_table()
    assert table["h4_reference"]["parameters"] == 104_537
    for row in table.values():
        assert abs(row["parameters_vs_reference"]) < 0.01


def test_registry_arms_reject_non_reference_capacity() -> None:
    """The absolute rank overrides are meaningless off the reference config."""

    from train import make_model

    with pytest.raises(ValueError, match="reference capacity"):
        make_model("mesh_transformer_kernel_singpair_h1", "large")


def test_registry_arms_carry_the_declared_head_counts() -> None:
    """Each built arm exposes the pre-registered head/rank split."""

    for arm in heads_rank_study.ARMS:
        model = heads_rank_study.build_arm(arm)
        from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
            KernelBasisCrossDecoder,
        )

        decoder_heads = {
            module.heads
            for module in model.modules()
            if isinstance(module, KernelBasisCrossDecoder)
        }
        assert decoder_heads == {arm.heads}, arm.key


def _synthetic_reports(means: dict[str, list[float]]) -> dict[str, list[dict]]:
    reports: dict[str, list[dict]] = {}
    for arm in heads_rank_study.ARMS:
        arm_reports = []
        for index, seed in enumerate(heads_rank_study.SEEDS):
            splits = {
                split: {"relative_l2_mean": means[arm.key][index]}
                for split in heads_rank_study.SPLITS
            }
            arm_reports.append(
                {
                    "parameters": 100_000,
                    "run_config": {
                        "model": arm.model,
                        "seed": seed,
                        "steps": heads_rank_study.STEPS,
                    },
                    "evaluation": {"splits": splits},
                }
            )
        reports[arm.key] = arm_reports
    return reports


def test_verdict_rule_is_two_sided() -> None:
    """The pre-registered rule must be able to return either verdict."""

    tie = heads_rank_study.aggregate(
        _synthetic_reports(
            {
                "h4_reference": [0.050, 0.052, 0.048],
                "h1": [0.051, 0.049, 0.052],
                "h8": [0.049, 0.053, 0.050],
            }
        )
    )
    assert tie["verdict"] == "heads_are_bookkeeping"

    separated = heads_rank_study.aggregate(
        _synthetic_reports(
            {
                "h4_reference": [0.050, 0.052, 0.048],
                "h1": [0.150, 0.148, 0.152],
                "h8": [0.049, 0.053, 0.050],
            }
        )
    )
    assert (
        separated["verdict"]
        == "difference_localizes_in_per_head_value_output_structure"
    )
    outside = [
        entry for entry in separated["comparisons"] if not entry["within_tolerance"]
    ]
    assert {entry["arm"] for entry in outside} == {"h1"}
