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

"""Focused tests for the Laplace readout-factorial reducer."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict

import laplace_readout_factorial as experiment
import laplace_readout_factorial_reduce as reducer
import pytest

SOURCE_FINGERPRINT = reducer.EXPECTED_SOURCE_FINGERPRINT


def _evaluation_protocol(seed: int) -> dict[str, object]:
    config = experiment.experiment_config(seed)
    return {
        "splits": list(experiment.SPLITS),
        "split_seed_offsets": {
            name: experiment._SPLIT_SEED_OFFSETS[name]  # noqa: SLF001
            for name in experiment.SPLITS
        },
        "resolutions": list(experiment.RESOLUTIONS),
        "cases_per_split": config.evaluation_cases,
        "harmonic_cases": config.harmonic_cases,
    }


def _reports(
    *,
    errors: dict[str, dict[str, float]] | None = None,
    residuals: dict[str, float] | None = None,
    resolutions: dict[str, tuple[float, float, float]] | None = None,
) -> list[tuple[str, dict[str, object]]]:
    errors = errors or {
        split: {
            "pure": 1.10 if split == reducer.ID_SPLIT else 1.20,
            "gate_only": 0.50,
            "contraction_only": 2.00,
            "full": 1.00,
        }
        for split in experiment.SPLITS
    }
    residuals = residuals or {
        "pure": 1.0e-3,
        "gate_only": 1.0e-3,
        "contraction_only": 1.0e-2,
        "full": 1.0e-2,
    }
    resolutions = resolutions or {arm: (0.30, 0.20, 0.10) for arm in experiment.ARMS}

    reports: list[tuple[str, dict[str, object]]] = []
    for arm_key, arm in experiment.ARMS.items():
        for seed in experiment.SEEDS:
            resolution = {
                str(value): {"relative_l2_mean": resolutions[arm_key][index]}
                for index, value in enumerate(experiment.RESOLUTIONS)
            }
            report: dict[str, object] = {
                "study": reducer.STUDY,
                "arm": asdict(arm),
                "run_config": asdict(experiment.experiment_config(seed)),
                "evaluation_protocol": _evaluation_protocol(seed),
                "accuracy_dtype": "float32",
                "residual_dtype": "float64",
                "registered_parameters": reducer.REGISTERED_PARAMETERS,
                "trainable_parameters": reducer.TRAINABLE_PARAMETERS[arm_key],
                "history": [{"step": 0}, {"step": experiment.STEPS}],
                "selected_validation": {"step": 0},
                "source": {"relevant_source_sha256": SOURCE_FINGERPRINT},
                "environment": deepcopy(reducer.EXPECTED_ENVIRONMENT),
                "evaluation": {
                    "accuracy_dtype": "float32",
                    "splits": {
                        split: {"relative_l2_mean": errors[split][arm_key]}
                        for split in experiment.SPLITS
                    },
                    "split_cases": {
                        split: [
                            {
                                "relative_l2": errors[split][arm_key],
                                "relative_linf": errors[split][arm_key],
                                "near_boundary_relative_l2": errors[split][arm_key],
                                "sampled_boundary_range_violation": 0.0,
                                "certified_maximum_principle_violation": 0.0,
                            }
                            for _ in range(
                                experiment.experiment_config(seed).evaluation_cases
                            )
                        ]
                        for split in experiment.SPLITS
                    },
                    "resolution": resolution,
                    "harmonic_residual": {
                        "dtype": "float64",
                        "normalized_laplacian_l2_mean": residuals[arm_key],
                    },
                },
            }
            reports.append((f"{arm_key}-{seed}.json", report))
    return reports


def test_reducer_computes_registered_factorial_effects_and_exact_bars() -> None:
    """Ratios, effects, interactions, and inclusive thresholds stay paired."""

    summary = reducer.reduce_reports(_reports())
    interpolation = summary["splits"]["interpolation"]

    assert interpolation["arm_arithmetic_means"]["pure"] == pytest.approx(1.10)
    pure_ratio = interpolation["paired_arm_over_full"]["pure"]
    assert set(pure_ratio["per_seed"]) == {str(seed) for seed in experiment.SEEDS}
    assert pure_ratio["geometric_mean"] == pytest.approx(1.10)

    effects = interpolation["factorial_effects"]
    expected_gate_marginal = math.sqrt(0.50 / (1.10 * 2.00))
    assert effects["gate"]["marginal_on_over_off"]["geometric_mean"] == pytest.approx(
        expected_gate_marginal
    )
    assert effects["gate"]["simple_on_over_off"]["contraction_off"][
        "geometric_mean"
    ] == pytest.approx(0.50 / 1.10)
    assert effects["gate"]["simple_on_over_off"]["contraction_on"][
        "geometric_mean"
    ] == pytest.approx(0.50)
    interaction = effects["log_scale_interaction"]
    assert interaction["mean"] == pytest.approx(math.log(1.10))
    assert interaction["multiplicative_ratio"] == pytest.approx(1.10)

    pure = summary["decisions"]["pure_core"]
    assert pure["status"] == "sufficient"
    assert all(pure["accuracy_checks"].values())
    assert all(pure["residual_checks"].values())

    factors = summary["decisions"]["factors"]
    assert factors["gate"]["status"] == "earned"
    assert factors["gate"]["qualifying_splits"] == list(experiment.SPLITS)
    assert factors["gate"]["no_seed_has_tenfold_residual_penalty"] is True
    assert factors["gate"]["resolution"]["passed"] is True
    assert factors["contraction"]["status"] == "inconclusive"


def test_accuracy_win_with_tenfold_residual_penalty_is_a_tradeoff() -> None:
    """The reducer must not relabel an accuracy/fidelity exchange as support."""

    summary = reducer.reduce_reports(
        _reports(
            residuals={
                "pure": 1.0e-4,
                "contraction_only": 1.0e-4,
                "gate_only": 1.0e-2,
                "full": 1.0e-2,
            }
        )
    )

    gate = summary["decisions"]["factors"]["gate"]
    assert gate["at_least_two_qualifying_splits"] is True
    assert gate["marginal_residual_on_over_off"] == pytest.approx(100.0)
    assert gate["no_seed_has_tenfold_residual_penalty"] is False
    assert gate["status"] == "tradeoff"
    assert gate["earned"] is False


def test_nonmonotone_enabled_arm_makes_accuracy_win_a_tradeoff() -> None:
    """Averaging two enabled cells must not hide broken resolution transfer."""

    resolutions = {
        "pure": (0.30, 0.20, 0.10),
        "contraction_only": (0.30, 0.20, 0.10),
        "gate_only": (0.30, 0.20, 0.25),
        "full": (0.30, 0.20, 0.10),
    }
    summary = reducer.reduce_reports(_reports(resolutions=resolutions))

    gate = summary["decisions"]["factors"]["gate"]
    assert gate["resolution"]["enabled_arm_monotone"] == {
        "gate_only": False,
        "full": True,
    }
    assert gate["status"] == "tradeoff"


def test_reducer_rejects_missing_duplicate_and_mixed_fingerprint_reports() -> None:
    """Only one exact, source-matched 4 x 5 matrix is admissible."""

    reports = _reports()
    with pytest.raises(ValueError, match="exactly 20"):
        reducer.reduce_reports(reports[:-1])

    duplicate = deepcopy(reports)
    duplicate[-1] = ("duplicate.json", deepcopy(duplicate[0][1]))
    with pytest.raises(ValueError, match="duplicate report"):
        reducer.reduce_reports(duplicate)

    wrong_source = deepcopy(reports)
    wrong_source[-1][1]["source"]["relevant_source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="registered source"):
        reducer.reduce_reports(wrong_source)


def test_reducer_rejects_registered_protocol_drift() -> None:
    """A matching arm/seed label cannot bless a changed evaluation bank."""

    reports = deepcopy(_reports())
    reports[-1][1]["evaluation_protocol"]["cases_per_split"] = 32

    with pytest.raises(ValueError, match="evaluation_protocol differs"):
        reducer.reduce_reports(reports)


def test_reducer_uses_paired_seed_ratios() -> None:
    """Heterogeneous seeds distinguish paired ratios from aggregate ratios."""

    reports = _reports()
    full_by_seed = dict(zip(experiment.SEEDS, (1.0, 10.0, 1.0, 10.0, 1.0), strict=True))
    pure_by_seed = dict(zip(experiment.SEEDS, (2.0, 5.0, 2.0, 5.0, 2.0), strict=True))
    for _, report in reports:
        arm = report["arm"]["key"]
        seed = report["run_config"]["seed"]
        if arm not in {"pure", "full"}:
            continue
        value = (pure_by_seed if arm == "pure" else full_by_seed)[seed]
        split = report["evaluation"]["splits"][reducer.ID_SPLIT]
        split["relative_l2_mean"] = value
        for case in report["evaluation"]["split_cases"][reducer.ID_SPLIT]:
            case["relative_l2"] = value

    summary = reducer.reduce_reports(reports)
    paired = summary["splits"][reducer.ID_SPLIT]["paired_arm_over_full"]["pure"]
    expected = math.prod(
        pure_by_seed[seed] / full_by_seed[seed] for seed in experiment.SEEDS
    ) ** (1.0 / len(experiment.SEEDS))

    assert paired["geometric_mean"] == pytest.approx(expected)
    assert paired["geometric_mean"] != pytest.approx(
        sum(pure_by_seed.values()) / sum(full_by_seed.values())
    )


def test_one_residual_outlier_cannot_be_hidden_by_other_seeds() -> None:
    """A tenfold fidelity failure in one seed cannot average into 'earned'."""

    reports = _reports()
    for _, report in reports:
        arm = report["arm"]["key"]
        seed = report["run_config"]["seed"]
        if arm in {"gate_only", "full"}:
            value = 100.0 if seed == experiment.SEEDS[0] else 0.01
        else:
            value = 1.0
        report["evaluation"]["harmonic_residual"]["normalized_laplacian_l2_mean"] = (
            value
        )

    summary = reducer.reduce_reports(reports)
    gate = summary["decisions"]["factors"]["gate"]

    assert gate["marginal_residual_on_over_off"] < 10.0
    assert gate["marginal_residual_per_seed_on_over_off"][
        str(experiment.SEEDS[0])
    ] == pytest.approx(100.0)
    assert gate["no_seed_has_tenfold_residual_penalty"] is False
    assert gate["status"] == "tradeoff"


def test_pure_residual_bars_apply_to_every_seed() -> None:
    """A low average residual cannot hide one seed outside the literal bar."""

    reports = _reports()
    for _, report in reports:
        arm = report["arm"]["key"]
        seed = report["run_config"]["seed"]
        if arm == "pure":
            value = 2.0e-3 if seed == experiment.SEEDS[0] else 1.0e-12
        elif arm == "full":
            value = 1.0
        else:
            continue
        report["evaluation"]["harmonic_residual"]["normalized_laplacian_l2_mean"] = (
            value
        )

    summary = reducer.reduce_reports(reports)
    pure = summary["decisions"]["pure_core"]

    assert pure["pure_residual_arithmetic_mean"] < 1.0e-3
    assert pure["residual_checks"]["pure_at_most_1e-3_in_every_seed"] is False
    assert pure["status"] == "not_sufficient"


def test_reducer_rejects_environment_drift() -> None:
    """All reports must come from the registered execution environment."""

    reports = deepcopy(_reports())
    reports[-1][1]["environment"]["device_name"] = "different GPU"

    with pytest.raises(ValueError, match="environment differs"):
        reducer.reduce_reports(reports)


def test_reduce_directory_atomically_replaces_one_summary(tmp_path) -> None:
    """The summary may live beside the 20 inputs and be safely regenerated."""

    input_directory = tmp_path / "runs"
    input_directory.mkdir()
    for label, report in _reports():
        (input_directory / label).write_text(json.dumps(report, allow_nan=False) + "\n")
    output = input_directory / "summary.json"

    first = reducer.reduce_directory(input_directory, output)
    assert json.loads(output.read_text()) == first
    output.write_text("{broken")
    second = reducer.reduce_directory(input_directory, output)

    assert json.loads(output.read_text()) == second
    assert second == first
