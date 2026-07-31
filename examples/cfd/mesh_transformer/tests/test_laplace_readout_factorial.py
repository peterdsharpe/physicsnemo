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

"""Focused tests for the Laplace scalar-readout factorial."""

from __future__ import annotations

import math
from dataclasses import replace

import laplace_readout_factorial as study
import pytest
import torch
from train import TRAIN_SPLIT, make_model, make_training_case, train_model

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _assert_same_state(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    except_names: frozenset[str] = frozenset(),
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name in actual_state:
        if name not in except_names:
            torch.testing.assert_close(
                actual_state[name], expected_state[name], rtol=0.0, atol=0.0
            )


def test_full_arm_is_the_unmodified_registered_model() -> None:
    """The positive control must remain byte-for-byte model-equivalent."""

    seed = 17
    torch.manual_seed(seed)
    expected = make_model(study.MODEL, study.CAPACITY, study.GAUGE)
    actual = study.build_arm_model("full", seed)

    _assert_same_state(actual, expected)
    assert all(parameter.requires_grad for parameter in actual.parameters())


@pytest.mark.parametrize("arm_key", tuple(study.ARMS))
def test_disabled_paths_are_exactly_neutral_or_zero(arm_key: str) -> None:
    """A disabled gate is one and a disabled contraction is zero."""

    arm = study.ARMS[arm_key]
    model = study.build_arm_model(arm_key, 17)
    scalar_gate, vector_dots = study._readout_modules(model)

    gate_input = torch.randn(3, scalar_gate.in_features)
    gate = 2.0 * torch.sigmoid(scalar_gate(gate_input))
    if arm.scalar_gate:
        assert all(parameter.requires_grad for parameter in scalar_gate.parameters())
    else:
        torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0.0, atol=0.0)
        assert all(
            torch.count_nonzero(parameter) == 0 and not parameter.requires_grad
            for parameter in scalar_gate.parameters()
        )

    dot_input = torch.randn(3, vector_dots.in_features)
    contraction = vector_dots(dot_input)
    if arm.scalar_from_vector_dots:
        assert all(parameter.requires_grad for parameter in vector_dots.parameters())
    else:
        torch.testing.assert_close(
            contraction, torch.zeros_like(contraction), rtol=0.0, atol=0.0
        )
        assert all(
            torch.count_nonzero(parameter) == 0 and not parameter.requires_grad
            for parameter in vector_dots.parameters()
        )


def test_every_arm_shares_all_nonintervened_initial_parameters() -> None:
    """The factorial changes no parameter outside its declared readout paths."""

    reference = study.build_arm_model("full", 29)
    reference_rng_state = torch.random.get_rng_state()
    reference_names = tuple(name for name, _ in reference.named_parameters())
    for arm_key in study.ARMS:
        candidate = study.build_arm_model(arm_key, 29)
        torch.testing.assert_close(
            torch.random.get_rng_state(), reference_rng_state, rtol=0.0, atol=0.0
        )
        assert (
            tuple(name for name, _ in candidate.named_parameters()) == reference_names
        )
        _assert_same_state(
            candidate,
            reference,
            except_names=study.INTERVENED_PARAMETERS,
        )


def test_step_zero_predictions_obey_the_neutral_gate_algebra() -> None:
    """At initialization, learning permission alone cannot change a value."""

    sample = make_training_case(
        TRAIN_SPLIT,
        distribution="boundary_balanced_mixture",
        seed=901,
        case_index=0,
        n_boundary=16,
        n_query=16,
        device=DEVICE,
        dtype=DTYPE,
        problem="dirichlet",
    )

    def prediction(arm_key: str) -> torch.Tensor:
        model = study.build_arm_model(arm_key, 59).to(device=DEVICE, dtype=DTYPE)
        model.eval()
        with torch.no_grad():
            return model(sample.domain).point_data["potential"]

    torch.testing.assert_close(
        prediction("full"),
        prediction("contraction_only"),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prediction("gate_only"),
        prediction("pure"),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("arm_key", tuple(study.ARMS))
def test_every_arm_can_train_and_run_the_registered_evaluator(arm_key: str) -> None:
    """A one-update CPU smoke test exercises each arm end to end."""

    config = replace(
        study.experiment_config(43),
        steps=1,
        train_boundary_points=8,
        train_query_points=4,
        report_every=1,
        validation_every=1,
        validation_cases=1,
        evaluation_cases=1,
        evaluation_boundary_points=8,
        # The benchmark's near-boundary metric needs enough fixed random
        # queries to contain at least one point with preimage radius >= 0.8.
        evaluation_query_points=32,
        harmonic_cases=1,
    )
    model = study.build_arm_model(arm_key, config.seed).to(device=DEVICE, dtype=DTYPE)
    history, selected = train_model(model, config, device=DEVICE, dtype=DTYPE)
    arm = study.ARMS[arm_key]
    for name, parameter in model.named_parameters():
        if (
            not arm.scalar_gate and name.startswith("output_projection.scalar_gate.")
        ) or (
            not arm.scalar_from_vector_dots
            and name.startswith("output_projection.scalar_from_vector_dots.")
        ):
            assert parameter.grad is None
    evaluation = study.evaluate_readout_factorial(
        model,
        config,
        device=DEVICE,
        dtype=DTYPE,
        resolutions=(8, 16),
    )

    assert history[-1]["step"] == 1
    assert selected is not None
    assert set(evaluation["splits"]) == set(study.SPLITS)
    assert all(len(evaluation["split_cases"][split]) == 1 for split in study.SPLITS)
    assert set(evaluation["resolution"]) == {"8", "16"}
    assert "harmonic_residual" in evaluation
    assert evaluation["accuracy_dtype"] == "float32"
    assert evaluation["harmonic_residual"]["dtype"] == "float64"
    assert {parameter.dtype for parameter in model.parameters()} == {DTYPE}

    def numeric_leaves(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from numeric_leaves(child)
        elif isinstance(value, (float, int)):
            yield float(value)

    assert all(math.isfinite(value) for value in numeric_leaves(evaluation))
