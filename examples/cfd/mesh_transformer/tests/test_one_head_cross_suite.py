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

"""Tests for the pre-registered one-head cross-suite confirmation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import one_head_cross_suite  # noqa: E402


def test_one_head_settings_preserve_total_score_capacity_in_any_dimension() -> None:
    """4*(12 + D*4) = 1*(48 + D*16) must hold for both suite dimensions."""

    settings = one_head_cross_suite.ONE_HEAD_SETTINGS
    for spatial_dims in (2, 3):
        reference = 4 * (12 + spatial_dims * 4)
        probe = settings["heads"] * (
            settings["scalar_rank"] + spatial_dims * settings["vector_rank"]
        )
        assert probe == reference


def test_h1_arm_parameter_counts_match_h4_within_one_percent() -> None:
    """Each h1 arm stays parameter-matched to its archived h4 comparator."""

    h4_parameters = {"screened": 104_293, "pf_velocity": 132_641, "laplace3d": 104_537}
    table = one_head_cross_suite.parameter_table()
    for key, row in table.items():
        assert abs(row["parameters"] - h4_parameters[key]) / h4_parameters[key] < 0.01
        assert row["parameters"] < h4_parameters[key]  # fewer params, every suite


def test_built_arms_carry_one_full_width_head() -> None:
    """Every suite's h1 arm builds with a single decoder head."""

    from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
        KernelBasisCrossDecoder,
    )

    for suite in one_head_cross_suite.SUITES:
        model = one_head_cross_suite.build_arm(suite)
        decoder_heads = {
            module.heads
            for module in model.modules()
            if isinstance(module, KernelBasisCrossDecoder)
        }
        assert decoder_heads == {1}, suite.key


def test_pseudo_sector_composes_with_one_head() -> None:
    """The pf arm keeps drive_pseudo_dim=8 alongside the one-head trade."""

    import torch
    from potential_flow import SAMPLE_BUILDERS, SPLITS

    torch.manual_seed(0)
    suite = next(s for s in one_head_cross_suite.SUITES if s.key == "pf_velocity")
    model = one_head_cross_suite.build_arm(suite)
    sample = SAMPLE_BUILDERS["potential_flow_velocity"](
        123,
        device="cpu",
        dtype=torch.float32,
        **SPLITS["potential_flow_velocity"]["in_distribution"],
    )
    velocity = model(sample.domain).point_data["velocity"]
    assert velocity.shape[-1] == 2
    assert bool(torch.isfinite(velocity).all())


def test_laplace3d_one_head_rejects_scalar_controls() -> None:
    """The 3D builder refuses the meaningless one-head + scalar combination."""

    from laplace3d_study import build_mesh_transformer_3d

    with pytest.raises(ValueError, match="scalar controls"):
        build_mesh_transformer_3d(one_head=True, scalar_only=True)


def test_comparator_paths_resolve_to_archived_seed_runs() -> None:
    """Every suite's h4 comparator resolves and covers the h1 seeds."""

    comparators = one_head_cross_suite.load_comparators()
    for suite in one_head_cross_suite.SUITES:
        runs = comparators[suite.key]
        seeds = {run["seed"] for run in runs}
        assert set(one_head_cross_suite.SEEDS) <= seeds, suite.key
        for run in runs:
            record = run.get("splits", run)
            for split in suite.splits:
                assert split in record, (suite.key, split)


def _synthetic_reports(
    means: dict[str, list[float]],
) -> dict[str, list[dict]]:
    reports: dict[str, list[dict]] = {}
    for suite in one_head_cross_suite.SUITES:
        suite_reports = []
        for index, seed in enumerate(one_head_cross_suite.SEEDS):
            suite_reports.append(
                {
                    "model": suite.model,
                    "seed": seed,
                    "steps": suite.steps,
                    "parameters": 100_000,
                    "splits": {
                        split: means[suite.key][index] for split in suite.splits
                    },
                }
            )
        reports[suite.key] = suite_reports
    return reports


def _synthetic_comparators(value: float) -> dict[str, list[dict]]:
    return {
        suite.key: [
            {
                "seed": seed,
                **{split: value + 0.001 * index for split in suite.splits},
            }
            for index, seed in enumerate(one_head_cross_suite.SEEDS)
        ]
        for suite in one_head_cross_suite.SUITES
    }


def test_flip_rule_is_two_sided() -> None:
    """The pre-registered rule must be able to return either verdict."""

    comparators = _synthetic_comparators(0.050)
    matched = one_head_cross_suite.aggregate(
        _synthetic_reports(
            {
                "screened": [0.050, 0.051, 0.049],
                "pf_velocity": [0.048, 0.050, 0.052],
                "laplace3d": [0.049, 0.050, 0.051],
            }
        ),
        comparators,
    )
    assert matched["verdict"] == one_head_cross_suite.VERDICT_FLIP

    one_decisive_loss = one_head_cross_suite.aggregate(
        _synthetic_reports(
            {
                "screened": [0.050, 0.051, 0.049],
                "pf_velocity": [0.048, 0.050, 0.052],
                "laplace3d": [0.150, 0.148, 0.152],
            }
        ),
        comparators,
    )
    assert one_decisive_loss["verdict"] == one_head_cross_suite.VERDICT_KEEP
    assert one_decisive_loss["any_decisive_loss"]
    losses = [
        entry
        for entry in one_decisive_loss["comparisons"]
        if entry["decisive_loss"]
    ]
    assert {entry["suite"] for entry in losses} == {"laplace3d"}


def test_registry_names_are_wired_into_each_driver() -> None:
    """The h1 arm names are accepted by their drivers' registries."""

    from potential_flow import FAMILY_MODEL_NAMES

    assert (
        "mesh_transformer_kernel_singpair_pseudo_h1"
        in FAMILY_MODEL_NAMES["potential_flow_velocity"]
    )
    from screened_laplace import _build_model as build_screened

    assert build_screened("mesh_transformer_kernel_singonly_h1") is not None
    from laplace3d_study import _build_model as build_3d

    assert build_3d("mesh_transformer_kernel_singpair_h1") is not None
