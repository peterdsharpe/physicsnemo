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

"""Tests for the executable MeshTransformer research-study policy."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from study import (  # noqa: E402
    CURRENT_CONTROL,
    DENSE_CONTROL,
    FP32_CONTRACT_TOLERANCE,
    RUN_BY_KEY,
    STUDY_RUNS,
    aggregate_study,
    command_manifest,
    discover_reports,
    make_command_specs,
    make_spectral_report,
    select_runs,
    validate_declared_report_set,
)


def _report(
    *,
    seed: int,
    model: str,
    capacity: str = "reference",
    identifier: float = 0.15,
    geometry: float = 0.17,
    trace: float = 0.12,
    laplacian: float = 0.25,
    mode_3: float = 0.11,
    mode_4: float = 0.13,
    similarity: float = 2.0e-6,
    superposition: float = 3.0e-6,
    zero_drive: float = 0.0,
    refinement: tuple[float, ...] = (0.02, 0.007, 0.001, 0.0),
    successive_refinement: tuple[float, ...] = (0.015, 0.006, 0.001),
) -> dict[str, object]:
    resolutions = (32, 64, 128, 256)
    return {
        "run_config": {
            "model": model,
            "capacity": capacity,
            "steps": 1_000,
            "seed": seed,
            "training_drive_distribution": "boundary_balanced_mixture",
            "training_objective": "auto",
            "evaluation_seed": 83_000_019,
            "evaluation_cases": 32,
            "evaluation_boundary_points": 128,
            "evaluation_query_points": 512,
            "harmonic_cases": 2,
        },
        "parameters": 1234,
        "elapsed_seconds": 12.5,
        "evaluation": {
            "splits": {
                "interpolation": {
                    "relative_l2_mean": identifier,
                    "certified_maximum_principle_violation_mean": 0.0,
                },
                "unseen_geometry_modes": {"relative_l2_mean": geometry},
                "stronger_deformation": {"relative_l2_mean": geometry},
                "unseen_boundary_frequencies": {"relative_l2_mean": 0.3},
            },
            "boundary_trace": {
                "interpolation": {"relative_l2_mean": trace},
            },
            "harmonic_residual": {
                "normalized_laplacian_l2_mean": laplacian,
            },
            "mode_response": {
                "3": {"relative_l2_mean": mode_3},
                "4": {"relative_l2_mean": mode_4},
            },
            "similarity": {"paired_covariance_error_max": similarity},
            "drive_linearity": {
                "superposition_relative_l2_max": superposition,
                "zero_drive_rms_max": zero_drive,
            },
            "resolution": {
                str(resolution): {
                    "change_from_finest_mean": refinement[index],
                    **(
                        {}
                        if index == 0
                        else {
                            "change_from_previous_mean": successive_refinement[
                                index - 1
                            ]
                        }
                    ),
                }
                for index, resolution in enumerate(resolutions)
            },
        },
    }


def test_declared_matrix_covers_every_approved_experiment() -> None:
    """The executable manifest must not silently omit a planned comparison."""

    assert len({run.key for run in STUDY_RUNS}) == len(STUDY_RUNS)
    factorial = select_runs(groups=("factorial",))
    assert [run.key for run in factorial] == [
        "factorial_minimal_moment",
        "factorial_encoded_moment",
        "factorial_minimal_pair",
        "factorial_encoded_pair",
    ]
    minimal_pair = RUN_BY_KEY["factorial_minimal_pair"]
    assert (minimal_pair.model, minimal_pair.capacity) == (
        "encoded_pair_kernel",
        "shallow",
    )
    assert [run.model for run in select_runs(groups=("simple_dense_control",))] == [
        "pair_kernel"
    ]
    assert {
        run.drive_distribution for run in select_runs(groups=("objective_control",))
    } == {
        "boundary_balanced_mixture",
        "disk_interior_balanced_mixture",
        "uniform_pure_mode",
    }
    assert {run.model for run in select_runs(groups=("stf",))} == {
        "stf_multipole_l1",
        "stf_multipole_l2",
        "stf_multipole_l4",
    }
    matched = select_runs(groups=("stf_control",))
    assert [(run.model, run.capacity) for run in matched] == [
        ("lifted_mesh_transformer", "stf_matched")
    ]
    assert {run.model for run in select_runs(groups=("layer_potential",))} == {
        "double_layer_direct",
        "double_layer_solved",
        "double_layer_richardson",
        "double_layer_encoded",
    }
    assert {run.model for run in select_runs(groups=("external",))} == {
        "globe_exact",
        "globe_hierarchical",
        "geotransolver_matched",
        "geotransolver_published_scale",
    }
    assert all(
        run.role == "external" and not run.eligible_for_advancement
        for run in select_runs(groups=("external",))
    )


def test_commands_use_one_early_seed_three_finalist_seeds_and_unique_paths(
    tmp_path: Path,
) -> None:
    """Protocol expansion must be deterministic and collision-free."""

    runs = (RUN_BY_KEY[CURRENT_CONTROL], RUN_BY_KEY[DENSE_CONTROL])
    early = make_command_specs(
        runs,
        phase="early",
        output_root=tmp_path,
        python="python-test",
        early_seed=101,
    )
    finalists = make_command_specs(
        runs,
        phase="finalists",
        output_root=tmp_path,
        python="python-test",
        finalist_seeds=(101, 103, 107),
        device="cuda:0",
    )
    assert len(early) == 2
    assert {spec.seed for spec in early} == {101}
    assert len(finalists) == 6
    assert {spec.seed for spec in finalists} == {101, 103, 107}
    assert len({spec.report_path for spec in early + finalists}) == 8
    assert all("--steps" in spec.argv and "1000" in spec.argv for spec in finalists)
    assert all("--device" in spec.argv and "cuda:0" in spec.argv for spec in finalists)

    manifest = command_manifest(finalists)
    assert manifest["schema_version"] == 1
    assert len(manifest["commands"]) == 6
    assert all("train.py" in item["shell"] for item in manifest["commands"])


def test_declared_report_validation_rejects_missing_finalist_seed() -> None:
    """Reject finalist groups that omit any required training seed."""

    reports = {
        CURRENT_CONTROL: [
            _report(seed=17, model="lifted_mesh_transformer"),
            _report(seed=29, model="lifted_mesh_transformer"),
        ]
    }
    with pytest.raises(ValueError, match="training seeds"):
        validate_declared_report_set(
            reports,
            phase="finalists",
            expected_seeds=(17, 29, 43),
        )


def test_declared_report_validation_rejects_mislabeled_model() -> None:
    """Reject reports whose model metadata contradicts the declared run."""

    reports = {
        CURRENT_CONTROL: [_report(seed=17, model="mesh_transformer")],
    }
    with pytest.raises(ValueError, match="expected 'lifted_mesh_transformer'"):
        validate_declared_report_set(
            reports,
            phase="early",
            expected_seeds=(17,),
        )


def test_selection_gates_apply_seed_means_worst_contract_and_exact_gap_closure() -> (
    None
):
    """A nominal candidate passes only when every stated gate passes."""

    reports = {
        CURRENT_CONTROL: [_report(seed=17, model="mesh_transformer", identifier=0.50)],
        DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.10)],
        "candidate": [
            _report(seed=17, model="candidate", identifier=0.19),
            _report(
                seed=29,
                model="candidate",
                identifier=0.19,
                similarity=FP32_CONTRACT_TOLERANCE * 0.9,
            ),
            _report(seed=43, model="candidate", identifier=0.19),
        ],
    }
    aggregate = aggregate_study(reports)
    gates = aggregate["candidates"]["candidate"]["selection_gates"]
    assert gates["passed"]
    # (0.50 - 0.19) / (0.50 - 0.10) = 0.775.
    assert gates["derived"]["candidate_gap_closure"] == pytest.approx(0.775)
    assert gates["checks"]["geometry_ood_ratio"]["value"] == pytest.approx(0.17 / 0.19)
    assert (
        aggregate["candidates"]["candidate"]["metrics"]["similarity_covariance_max"][
            "maximum"
        ]
        == FP32_CONTRACT_TOLERANCE * 0.9
    )


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    [
        ({"mode_3": 0.201}, "mode_3"),
        ({"mode_4": 0.201}, "mode_4"),
        ({"identifier": 0.201}, "id_accuracy"),
        ({"geometry": 0.181}, "geometry_ood_ratio"),
        ({"trace": 0.201}, "boundary_trace"),
        ({"laplacian": 0.401}, "laplacian_residual"),
        (
            {"similarity": FP32_CONTRACT_TOLERANCE * 1.01},
            "similarity_covariance",
        ),
        (
            {"superposition": FP32_CONTRACT_TOLERANCE * 1.01},
            "drive_superposition",
        ),
        (
            {"zero_drive": FP32_CONTRACT_TOLERANCE * 1.01},
            "zero_drive_preservation",
        ),
        (
            {"successive_refinement": (0.015, 0.006, 0.007)},
            "boundary_refinement",
        ),
    ],
)
def test_each_selection_gate_can_reject_independently(
    mutation: dict[str, object], failed_gate: str
) -> None:
    """Every gate is executable policy, rather than report-only metadata."""

    reports = {
        CURRENT_CONTROL: [_report(seed=17, model="mesh_transformer", identifier=0.50)],
        DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.10)],
        "candidate": [_report(seed=17, model="candidate", **mutation)],
    }
    aggregate = aggregate_study(reports)
    gates = aggregate["candidates"]["candidate"]["selection_gates"]
    assert not gates["passed"]
    assert not gates["checks"][failed_gate]["passed"]


def test_gap_closure_rejects_candidate_that_closes_less_than_75_percent() -> None:
    """The ID threshold does not subsume the relative-control requirement."""

    reports = {
        CURRENT_CONTROL: [_report(seed=17, model="mesh_transformer", identifier=0.22)],
        DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.10)],
        # ID passes 0.20, but this closes only 1/6 of the 0.12 control gap.
        "candidate": [_report(seed=17, model="candidate", identifier=0.20)],
    }
    gates = aggregate_study(reports)["candidates"]["candidate"]["selection_gates"]
    assert gates["checks"]["id_accuracy"]["passed"]
    assert not gates["checks"]["dense_gap_closure"]["passed"]


def test_factorial_effects_separate_encoder_and_decoder_contrasts() -> None:
    """Report boundary-processor and decoder effects as separate contrasts."""

    reports = {
        "factorial_minimal_moment": [
            _report(seed=17, model="lifted_mesh_transformer", identifier=0.60)
        ],
        CURRENT_CONTROL: [
            _report(seed=17, model="lifted_mesh_transformer", identifier=0.50)
        ],
        DENSE_CONTROL: [
            _report(seed=17, model="pair_kernel", identifier=0.10, geometry=0.12)
        ],
        "factorial_encoded_pair": [
            _report(
                seed=17,
                model="encoded_pair_kernel",
                identifier=0.08,
                geometry=0.10,
            )
        ],
    }

    effects = aggregate_study(reports)["factorial_effects"]
    identifier = effects["metrics"]["id_relative_l2"]
    assert identifier["pair_decoder_gain_with_minimal_boundary"] == pytest.approx(0.50)
    assert identifier["boundary_encoder_gain_with_pair_decoder"] == pytest.approx(0.02)
    assert effects["meets_declared_pair_encoder_practical_effect_threshold"] is True


def test_aggregate_rejects_mixed_source_fingerprints() -> None:
    """Never aggregate results produced by different source revisions."""

    current = _report(seed=17, model="lifted_mesh_transformer", identifier=0.50)
    dense = _report(seed=17, model="pair_kernel", identifier=0.10)
    current["source"] = {"relevant_source_sha256": "a" * 64}
    dense["source"] = {"relevant_source_sha256": "b" * 64}
    with pytest.raises(ValueError, match="source fingerprint"):
        aggregate_study({CURRENT_CONTROL: [current], DENSE_CONTROL: [dense]})


def test_aggregate_rejects_duplicate_seeds_and_non_improving_dense_control() -> None:
    """Reject duplicate replicates and an invalid dense-control baseline."""

    with pytest.raises(ValueError, match="duplicate training seeds"):
        aggregate_study(
            {
                CURRENT_CONTROL: [
                    _report(seed=17, model="mesh_transformer", identifier=0.5),
                    _report(seed=17, model="mesh_transformer", identifier=0.5),
                ],
                DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.1)],
            }
        )

    with pytest.raises(ValueError, match="dense control improves"):
        aggregate_study(
            {
                CURRENT_CONTROL: [
                    _report(seed=17, model="mesh_transformer", identifier=0.1)
                ],
                DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.2)],
            }
        )


def test_report_discovery_requires_exactly_one_json_per_seed(tmp_path: Path) -> None:
    """Require one unambiguous report artifact for each run and seed."""

    phase = tmp_path / "early" / "candidate" / "seed-17"
    phase.mkdir(parents=True)
    report = _report(seed=17, model="candidate")
    (phase / "candidate.json").write_text(json.dumps(report))
    assert discover_reports(tmp_path, "early") == {
        "candidate": [phase / "candidate.json"]
    }
    (phase / "accidental-copy.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exactly one JSON report"):
        discover_reports(tmp_path, "early")


def test_spectral_report_recovers_rank_one_boundary_mean(tmp_path: Path) -> None:
    """The checkpoint hook must compose the complete extraction pipeline."""

    checkpoint = tmp_path / "boundary_mean.pt"
    torch.save(
        {
            "model": "boundary_mean",
            "capacity": "reference",
            "state_dict": {},
            "run_config": {"seed": 17},
        },
        checkpoint,
    )
    report = make_spectral_report(
        checkpoint,
        dtype_name="float64",
        n_boundary=8,
        n_query_angles=16,
        radii=(0.25, 0.6),
        maximum_mode=3,
        requested_ranks=(0, 1, 8),
    )
    assert report["model"] == "boundary_mean"
    assert report["spectrum"]["learned"]["numerical_rank"] == 1
    assert report["spectrum"]["analytic"]["numerical_rank"] == 8
    assert report["spectrum"]["numerical_rank_relative_tolerance"] > 0.0
    assert report["dtype"] == "float64"
    assert report["source_matches_evaluator"] is None
    assert report["source_moment_family_norms"] is None
    assert (
        len(report["operator_matrix"]["learned"]),
        len(report["operator_matrix"]["learned"][0]),
        len(report["operator_matrix"]["learned"][0][0]),
    ) == (2, 16, 8)
    transfer = report["fourier_transfer"]["learned"]["magnitude"]
    # Two radii, seven output modes, and four nonnegative input modes.
    assert (len(transfer), len(transfer[0]), len(transfer[0][0])) == (2, 7, 4)


def test_replicates_must_share_configuration() -> None:
    """Require replicates to differ only in their training seed."""

    first = _report(seed=17, model="candidate")
    second = deepcopy(_report(seed=29, model="candidate"))
    second["run_config"]["evaluation_seed"] += 1
    with pytest.raises(ValueError, match="one evaluation configuration"):
        aggregate_study(
            {
                CURRENT_CONTROL: [
                    _report(seed=17, model="mesh_transformer", identifier=0.5)
                ],
                DENSE_CONTROL: [_report(seed=17, model="pair_kernel", identifier=0.1)],
                "candidate": [first, second],
            }
        )


def test_candidates_must_share_one_evaluation_bank_and_dtype() -> None:
    """Compare candidates only on a shared case bank and numeric dtype."""

    current = _report(seed=17, model="mesh_transformer", identifier=0.5)
    dense = _report(seed=17, model="pair_kernel", identifier=0.1)
    current["dtype"] = "float32"
    dense["dtype"] = "float64"
    with pytest.raises(ValueError, match="one evaluation bank"):
        aggregate_study(
            {
                CURRENT_CONTROL: [current],
                DENSE_CONTROL: [dense],
            }
        )
