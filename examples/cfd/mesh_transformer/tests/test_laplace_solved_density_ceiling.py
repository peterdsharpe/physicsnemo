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

"""Focused tests for the fixed-carrier solved-density ceiling."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import laplace_readout_factorial as factorial
import laplace_solved_density_ceiling as study
import pytest
import torch
from layer_potential import SolvedDoubleLayerPotential

DEVICE = torch.device("cpu")


def test_protocol_is_the_frozen_factorial_evaluation() -> None:
    """The ceiling must remain paired to the completed factorial banks."""

    config = study.evaluation_config()
    assert config == factorial.experiment_config(factorial.SEEDS[0])
    assert study.SPLITS == factorial.SPLITS
    assert study.RESOLUTIONS == factorial.RESOLUTIONS
    assert study._SPLIT_SEED_OFFSETS == factorial._SPLIT_SEED_OFFSETS

    protocol = study.evaluation_protocol(config)
    assert protocol == {
        "source_study": "laplace_readout_factorial_v1",
        "problem": "dirichlet",
        "evaluation_seed": 97_000_037,
        "splits": list(factorial.SPLITS),
        "split_seed_offsets": {
            name: factorial._SPLIT_SEED_OFFSETS[name] for name in factorial.SPLITS
        },
        "cases_per_split": 64,
        "fixed_boundary_points": 128,
        "query_points_per_case": 512,
        "accuracy_dtype": "float32",
        "boundary_trace_dtype": "float64",
        "boundary_trace_cases_per_split": 64,
        "resolution_seed": 100_000_037,
        "resolution_cases": 8,
        "resolution_query_points": 512,
        "resolutions": [64, 128, 256],
    }


def test_model_is_the_parameter_free_solved_density_control() -> None:
    model = study.build_model()
    assert isinstance(model, SolvedDoubleLayerPotential)
    assert sum(parameter.numel() for parameter in model.parameters()) == 0


def test_small_evaluation_preserves_schema_and_numerical_contracts() -> None:
    """A small real run exercises every split, trace solve, and resolution."""

    config = replace(
        study.evaluation_config(),
        evaluation_cases=1,
        evaluation_boundary_points=8,
        evaluation_query_points=32,
    )
    evaluation = study.evaluate_solved_density_ceiling(
        study.build_model(),
        config,
        device=DEVICE,
        dtype=torch.float32,
        resolutions=(8, 16),
    )

    assert set(evaluation["splits"]) == set(study.SPLITS)
    assert set(evaluation["split_cases"]) == set(study.SPLITS)
    assert all(len(evaluation["split_cases"][name]) == 1 for name in study.SPLITS)
    assert evaluation["accuracy_dtype"] == "float32"
    assert evaluation["boundary_trace"]["dtype"] == "float64"
    assert set(evaluation["boundary_trace"]["splits"]) == set(study.SPLITS)
    assert set(evaluation["resolution"]) == {"8", "16"}

    for name in study.SPLITS:
        case = evaluation["split_cases"][name][0]
        aggregate = evaluation["splits"][name]
        for metric, value in case.items():
            assert aggregate[f"{metric}_mean"] == pytest.approx(value)
        assert evaluation["boundary_trace"]["splits"][name]["relative_l2_max"] < 1.0e-12

    def numeric_leaves(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from numeric_leaves(child)
        elif isinstance(value, list):
            for child in value:
                yield from numeric_leaves(child)
        elif isinstance(value, (float, int)):
            yield float(value)

    assert all(math.isfinite(value) for value in numeric_leaves(evaluation))


def test_evaluator_rejects_unpaired_dtype() -> None:
    with pytest.raises(ValueError, match="must use float32"):
        study.evaluate_solved_density_ceiling(
            study.build_model(),
            study.evaluation_config(),
            device=DEVICE,
            dtype=torch.float64,
        )


def test_run_study_writes_one_complete_atomic_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The CLI seam publishes one self-describing JSON artifact."""

    evaluation = {
        "accuracy_dtype": "float32",
        "splits": {},
        "split_cases": {},
        "boundary_trace": {"dtype": "float64", "splits": {}},
        "resolution": {},
    }
    monkeypatch.setattr(
        study,
        "evaluate_solved_density_ceiling",
        lambda model, config, *, device, dtype: evaluation,
    )
    monkeypatch.setattr(study, "runtime_environment", lambda device: {"device": "cpu"})
    monkeypatch.setattr(
        study,
        "source_provenance",
        lambda: {"relevant_source_sha256": "source-digest"},
    )
    output = tmp_path / "report.json"

    report = study.run_study(device=DEVICE, output=output)

    assert json.loads(output.read_text()) == report
    assert report["study"] == study.STUDY
    assert report["evaluation"] == evaluation
    assert report["registered_parameters"] == 0
    assert report["trainable_parameters"] == 0
    assert report["source"]["relevant_source_sha256"] == "source-digest"
    assert not list(tmp_path.glob("*.tmp"))
