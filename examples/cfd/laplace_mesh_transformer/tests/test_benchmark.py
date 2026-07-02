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

"""Contracts for benchmark metrics, baselines, and model wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import benchmark as performance_benchmark  # noqa: E402
import train as training_benchmark  # noqa: E402
from conformal_laplace import (  # noqa: E402
    HarmonicDrive,
    build_domain_sample,
    evaluate_potential,
    sample_drive,
    sample_geometry,
    sample_similarity,
    transform_sample,
    unit_circle,
)
from metrics import (  # noqa: E402
    aggregate_metrics,
    case_metrics,
    certified_maximum_principle_violation,
    paired_case_bootstrap,
    sampled_boundary_range_violation,
    weighted_relative_l2,
)
from models import (  # noqa: E402
    BoundaryMean,
    InvariantPairKernel,
    build_lifted_mesh_transformer,
    build_mesh_transformer,
)
from train import (  # noqa: E402
    EVALUATION_SPLITS,
    TINY_CONFIG,
    TRAIN_SPLIT,
    RunConfig,
    _certified_boundary_range,
    _pointwise_laplacian,
    _predict,
    evaluate_boundary_trace,
    evaluate_drive_linearity_contract,
    evaluate_harmonic_residual,
    make_case,
    make_training_case,
    relative_mse,
    train_model,
)


def _sample(*, constant: bool = False):
    geometry = sample_geometry(
        21,
        modes=(2, 3),
        deformation_range=(0.2, 0.2),
        dtype=torch.float64,
    )
    if constant:
        drive = HarmonicDrive(
            constant=torch.tensor(1.7, dtype=torch.float64),
            modes=(),
            coefficients=torch.empty(0, dtype=torch.complex128),
        )
    else:
        drive = sample_drive(
            22,
            modes=(1, 2, 3),
            regularity=1.0,
            dtype=torch.float64,
        )
    return build_domain_sample(
        geometry,
        drive,
        n_boundary=24,
        n_query=32,
        query_seed=23,
    )


def test_weighted_metrics_have_declared_physical_semantics() -> None:
    """Apply physical quadrature weights and the declared normalizations."""

    prediction = torch.tensor([2.0, 0.0], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    weights = torch.tensor([3.0, 1.0], dtype=torch.float64)
    expected = torch.sqrt(torch.tensor(7.0 / 7.0, dtype=torch.float64))
    torch.testing.assert_close(
        weighted_relative_l2(prediction, target, weights), expected
    )
    torch.testing.assert_close(
        relative_mse(prediction, target, weights), expected.square()
    )

    boundary = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    interior = torch.tensor([-1.5, 2.25], dtype=torch.float64)
    expected_violation = (
        torch.tensor(0.5, dtype=torch.float64) / boundary.square().mean().sqrt()
    )
    torch.testing.assert_close(
        sampled_boundary_range_violation(interior, boundary), expected_violation
    )


def test_case_aggregation_weights_domains_equally() -> None:
    """Average complete PDE cases rather than pooling their query points."""

    first = {
        "relative_l2": 1.0,
        "relative_linf": 2.0,
        "near_boundary_relative_l2": 3.0,
        "sampled_boundary_range_violation": 4.0,
    }
    second = {name: 3.0 * value for name, value in first.items()}
    aggregate = aggregate_metrics((first, second))
    assert aggregate["relative_l2_mean"] == 2.0
    assert aggregate["relative_l2_median"] == 2.0
    assert aggregate["relative_linf_mean"] == 4.0


def test_paired_case_bootstrap_is_deterministic_and_case_paired() -> None:
    """Bootstrap aligned continuous cases reproducibly as paired samples."""

    left = [1.0, 4.0, 9.0, 16.0]
    right = [0.5, 2.0, 4.5, 8.0]

    first = paired_case_bootstrap(left, right, seed=31, resamples=1_000)
    second = paired_case_bootstrap(left, right, seed=31, resamples=1_000)

    assert first == second
    assert first["mean"] == pytest.approx(3.75)
    lower, upper = first["case_bootstrap_interval"]
    assert lower < first["mean"] < upper


@pytest.mark.parametrize("model", [BoundaryMean(), InvariantPairKernel(hidden_dim=8)])
def test_baselines_exactly_reproduce_constant_boundary_data(model) -> None:
    """Require both baseline operators to preserve constant solutions."""

    sample = _sample(constant=True)
    model = model.to(dtype=torch.float64)
    prediction = model(sample.domain).point_data["potential"]
    torch.testing.assert_close(prediction, sample.target, rtol=2.0e-14, atol=2.0e-14)


def test_dense_pair_kernel_is_similarity_invariant_by_construction() -> None:
    """Keep dense relative-kernel predictions invariant under O(2) similarities."""

    torch.manual_seed(5)
    model = InvariantPairKernel(hidden_dim=12, hidden_layers=2).to(dtype=torch.float64)
    sample = _sample()
    transformed = transform_sample(
        sample,
        sample_similarity(
            26,
            scale_range=(3.2, 3.2),
            translation_extent=2.0,
            reflection=True,
            dtype=torch.float64,
        ),
    )
    original = model(sample.domain).point_data["potential"]
    actual = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(actual, original, rtol=2.0e-13, atol=2.0e-13)


def test_drive_linearity_contract_measures_complete_model() -> None:
    """Measure superposition and zero response on the complete model path."""

    torch.manual_seed(29)
    model = InvariantPairKernel(hidden_dim=8, hidden_layers=1).double()
    result = evaluate_drive_linearity_contract(
        model,
        seed=30,
        n_cases=1,
        n_boundary=12,
        n_query=8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert result["superposition_relative_l2_max"] < 2.0e-14
    assert result["zero_drive_rms_max"] < 2.0e-14


def test_laplace_mean_lift_exactly_reproduces_constants() -> None:
    """Anchor the learned moment correction to preserve constant solutions."""

    torch.manual_seed(6)
    model = build_lifted_mesh_transformer(TINY_CONFIG).to(dtype=torch.float64)
    sample = _sample(constant=True)
    prediction = model(sample.domain).point_data["potential"]
    torch.testing.assert_close(prediction, sample.target, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("lifted", [False, True])
def test_mesh_transformer_is_o2_similarity_invariant(lifted: bool) -> None:
    """Preserve scalar output under translations, scaling, and O(2) actions."""

    torch.manual_seed(27)
    builder = build_lifted_mesh_transformer if lifted else build_mesh_transformer
    model = builder(TINY_CONFIG).to(dtype=torch.float64)
    sample = _sample()
    transformed = transform_sample(
        sample,
        sample_similarity(
            28,
            scale_range=(2.7, 2.7),
            translation_extent=1.5,
            reflection=True,
            dtype=torch.float64,
        ),
    )
    original = model(sample.domain).point_data["potential"]
    actual = model(transformed.domain).point_data["potential"]
    torch.testing.assert_close(actual, original, rtol=3.0e-10, atol=3.0e-11)


def test_mesh_transformer_benchmark_wiring_supports_training() -> None:
    """Connect mesh fields to predictions and gradients through the benchmark."""

    torch.manual_seed(7)
    sample = _sample()
    model = build_mesh_transformer(TINY_CONFIG).to(dtype=torch.float64)
    prediction = model(sample.domain).point_data["potential"]
    assert prediction.shape == sample.target.shape
    assert torch.isfinite(prediction).all()

    loss = relative_mse(prediction, sample.target, sample.area_jacobian)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_case_metrics_include_sampled_boundary_range_diagnostic() -> None:
    """Report every declared per-case error and boundary-range diagnostic."""

    sample = _sample()
    metrics = case_metrics(
        sample.target,
        sample.target,
        sample.area_jacobian,
        sample.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        sample.query_preimages.abs(),
    )
    assert metrics["relative_l2"] == 0.0
    assert metrics["relative_linf"] == 0.0
    assert metrics["near_boundary_relative_l2"] == 0.0
    # Discrete midpoint samples approximate the continuous boundary extrema.
    assert metrics["sampled_boundary_range_violation"] < 2.0e-2


def test_continuous_boundary_enclosure_gives_certified_maximum_principle_check() -> (
    None
):
    """Use a certified continuous trace enclosure for maximum-principle checks."""

    sample = _sample()
    lower, upper = _certified_boundary_range(sample.drive, n_samples=257)

    angles = torch.linspace(
        0.0,
        2.0 * torch.pi,
        10_001,
        dtype=torch.float64,
    )[:-1]
    dense_trace = evaluate_potential(sample.drive, unit_circle(angles))
    assert dense_trace.min() >= lower
    assert dense_trace.max() <= upper
    assert (
        certified_maximum_principle_violation(
            sample.target,
            lower,
            upper,
            sample.drive.boundary_rms,
        )
        == 0.0
    )

    violating = sample.target.clone()
    violating[0] = upper + sample.drive.boundary_rms
    torch.testing.assert_close(
        certified_maximum_principle_violation(
            violating,
            lower,
            upper,
            sample.drive.boundary_rms,
        ),
        torch.ones((), dtype=torch.float64),
    )


def test_generalization_splits_isolate_their_declared_axes() -> None:
    """Vary only the declared geometry or frequency axis in each OOD split."""

    mixed = EVALUATION_SPLITS["mixed_geometry_modes"]
    frequency = EVALUATION_SPLITS["unseen_boundary_frequencies"]

    assert mixed.deformation_range == TRAIN_SPLIT.deformation_range
    assert mixed.drive_modes == TRAIN_SPLIT.drive_modes
    assert mixed.drive_include_constant
    assert set(mixed.geometry_modes) > set(TRAIN_SPLIT.geometry_modes)

    assert frequency.geometry_modes == TRAIN_SPLIT.geometry_modes
    assert frequency.deformation_range == TRAIN_SPLIT.deformation_range
    assert set(frequency.drive_modes).isdisjoint(TRAIN_SPLIT.drive_modes)
    assert not frequency.drive_include_constant

    train_sample = make_case(
        TRAIN_SPLIT,
        seed=41,
        case_index=0,
        n_boundary=16,
        n_query=8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert train_sample.drive.modes == TRAIN_SPLIT.drive_modes
    assert torch.count_nonzero(train_sample.drive.coefficients) == len(
        TRAIN_SPLIT.drive_modes
    )


def test_disk_interior_balancing_equalizes_each_component_exactly() -> None:
    """Give every sampled disk harmonic exactly equal interior energy."""

    arguments = {
        "spec": TRAIN_SPLIT,
        "seed": 43,
        "case_index": 2,
        "n_boundary": 16,
        "n_query": 8,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    balanced = make_training_case(
        distribution="disk_interior_balanced_mixture", **arguments
    )

    modal_area_energies = balanced.drive.coefficients.abs().square() / (
        2.0
        * balanced.drive.coefficients.real.new_tensor(
            [mode + 1.0 for mode in TRAIN_SPLIT.drive_modes]
        )
    )
    torch.testing.assert_close(
        modal_area_energies,
        balanced.drive.constant.square().expand_as(modal_area_energies),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        balanced.drive.boundary_rms,
        torch.ones((), dtype=torch.float64),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_uniform_pure_mode_training_stream_is_deterministic() -> None:
    """Sample one boundary mode uniformly and reproducibly per training case."""

    arguments = {
        "spec": TRAIN_SPLIT,
        "distribution": "uniform_pure_mode",
        "seed": 47,
        "case_index": 5,
        "n_boundary": 16,
        "n_query": 8,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    first = make_training_case(**arguments)
    second = make_training_case(**arguments)

    assert first.drive.modes == second.drive.modes
    assert len(first.drive.modes) == 1
    assert first.drive.modes[0] in TRAIN_SPLIT.drive_modes
    assert first.drive.constant == 0.0
    torch.testing.assert_close(first.drive.coefficients, second.drive.coefficients)


def test_prediction_boundary_strips_private_interior_data() -> None:
    """Prevent analytic targets and metadata from leaking into prediction."""

    sample = _sample()

    class InspectingModel(torch.nn.Module):
        """Assert that a model receives only public inference-time fields."""

        def forward(self, domain):
            assert list(domain.interior.point_data.keys()) == []
            assert list(domain.interior.cell_data.keys()) == []
            assert list(domain.interior.global_data.keys()) == []
            return domain.interior.with_data(
                point_data={
                    "potential": domain.interior.points.new_zeros(
                        domain.interior.n_points
                    )
                }
            )

    prediction = _predict(InspectingModel(), sample)
    assert prediction.shape == sample.target.shape
    assert "potential" in sample.domain.interior.point_data


def test_performance_harness_checks_dense_oracle_and_emits_json(
    monkeypatch, capsys
) -> None:
    """Emit auditable timings only after checking the exact dense oracle."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--component",
            "attention",
            "--phase",
            "inference",
            "--device",
            "cpu",
            "--n-source",
            "8",
            "--n-query",
            "7",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--check",
        ],
    )

    performance_benchmark.main()

    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == 2
    assert len(record["source"]["relevant_source_sha256"]) == 64
    assert record["correctness"]["oracle"] == "dense_all_pairs_attention"
    assert record["correctness"]["passed"] is True
    assert record["timings"]["build_moments"]["median_ms"] >= 0.0


@pytest.mark.parametrize("kind", ["constant", "affine"])
def test_harmonic_residual_accepts_query_independent_or_affine_models(kind) -> None:
    """Return zero residual for constant and affine harmonic functions."""

    class AnalyticModel(torch.nn.Module):
        """Produce an exactly harmonic constant or affine potential."""

        def __init__(self) -> None:
            super().__init__()
            self.constant = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

        def forward(self, domain):
            if kind == "constant":
                potential = self.constant.expand(domain.interior.n_points)
            else:
                potential = (
                    domain.interior.points[:, 0]
                    - 0.3 * domain.interior.points[:, 1]
                    + self.constant
                )
            return domain.interior.with_data(point_data={"potential": potential})

    result = evaluate_harmonic_residual(
        AnalyticModel(),
        seed=53,
        n_cases=1,
        n_boundary=16,
        n_query=8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert result["normalized_laplacian_l2_mean"] == 0.0


def test_harmonic_residual_detects_nonharmonic_quadratic() -> None:
    """Detect the nonzero Laplacian of a radial quadratic potential."""

    class QuadraticModel(torch.nn.Module):
        """Produce a quadratic potential with a known nonzero Laplacian."""

        def forward(self, domain):
            potential = domain.interior.points.square().sum(dim=-1)
            return domain.interior.with_data(point_data={"potential": potential})

    result = evaluate_harmonic_residual(
        QuadraticModel(),
        seed=59,
        n_cases=1,
        n_boundary=16,
        n_query=8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert result["normalized_laplacian_l2_mean"] > 0.0


def test_pointwise_laplacian_uses_each_outputs_own_coordinate() -> None:
    """Exclude cross-query derivatives from each pointwise Laplacian."""

    coordinates = torch.tensor(
        [[0.2, -0.3], [0.4, 0.5]], dtype=torch.float64, requires_grad=True
    )
    first = coordinates[0].square().sum() + 3.0 * coordinates[1].square().sum()
    second = 5.0 * coordinates[0].square().sum() + 7.0 * coordinates[1].square().sum()

    actual = _pointwise_laplacian(torch.stack((first, second)), coordinates)

    # Cross-query terms must not contaminate the physical Laplacian of output i
    # with respect to query coordinate i.
    torch.testing.assert_close(actual, actual.new_tensor([4.0, 28.0]))


def test_training_restores_step_zero_when_updates_degrade_validation(
    monkeypatch,
) -> None:
    """Restore the initial checkpoint when every update harms validation."""

    class ScalarModel(torch.nn.Module):
        """Expose one scalar parameter for checkpoint-selection testing."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    model = ScalarModel()
    sample = SimpleNamespace(
        target=torch.ones(1),
        area_jacobian=torch.ones(1),
    )
    monkeypatch.setattr(training_benchmark, "make_case", lambda *args, **kwargs: sample)
    monkeypatch.setattr(
        training_benchmark,
        "_predict",
        lambda candidate, ignored: candidate.weight.expand(1),
    )
    monkeypatch.setattr(
        training_benchmark,
        "evaluate_split",
        lambda candidate, *args, **kwargs: {
            "relative_l2_mean": float(candidate.weight.detach().abs())
        },
    )
    config = RunConfig(
        steps=2,
        train_boundary_points=1,
        train_query_points=1,
        evaluation_boundary_points=1,
        evaluation_query_points=1,
        validation_cases=1,
        validation_every=1,
        report_every=10,
        learning_rate=0.1,
        weight_decay=0.0,
    )

    history, selected = train_model(
        model,
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert history[0]["step"] == 0
    assert selected == {"step": 0, "validation_relative_l2": 0.0}
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight))


def test_collocation_training_selects_without_interior_validation(
    monkeypatch,
) -> None:
    """Select collocation checkpoints without consulting interior labels."""

    class CollocationModel(torch.nn.Module):
        """Expose a boundary-only objective for selection-path testing."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def collocation_loss(self, domain) -> torch.Tensor:
            """Return a simple loss independent of private interior targets."""

            del domain
            return (self.weight - 1.0).square()

    sample = SimpleNamespace(domain=object())
    monkeypatch.setattr(
        training_benchmark,
        "make_training_case",
        lambda *args, **kwargs: sample,
    )
    monkeypatch.setattr(
        training_benchmark,
        "evaluate_split",
        lambda *args, **kwargs: pytest.fail("interior validation leaked"),
    )
    model = CollocationModel()
    config = RunConfig(
        steps=1,
        train_boundary_points=1,
        train_query_points=1,
        evaluation_boundary_points=1,
        evaluation_query_points=1,
        validation_cases=1,
        validation_every=1,
        report_every=1,
        learning_rate=0.1,
        weight_decay=0.0,
        training_objective="boundary_collocation",
    )

    history, selected = train_model(
        model,
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert "validation_boundary_collocation_mse" in history[0]
    assert selected is not None
    assert set(selected) == {"step", "validation_boundary_collocation_mse"}
    assert selected["step"] == 1


def test_boundary_trace_queries_common_polygon_panel_centroids() -> None:
    """Evaluate every candidate on the same discrete boundary trace points."""

    class ExactDiscreteTrace(torch.nn.Module):
        """Return the stored panel trace after verifying query placement."""

        def forward(self, domain):
            boundary = domain.boundaries["dirichlet"]
            torch.testing.assert_close(
                domain.interior.points,
                boundary.cell_centroids,
                rtol=0.0,
                atol=0.0,
            )
            return domain.interior.with_data(
                point_data={"potential": boundary.cell_data["boundary_value"]}
            )

    result = evaluate_boundary_trace(
        ExactDiscreteTrace(),
        TRAIN_SPLIT,
        seed=67,
        n_cases=1,
        n_boundary=12,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert result["relative_l2_mean"] == 0.0


def test_evaluate_only_report_preserves_checkpoint_provenance(
    monkeypatch, tmp_path
) -> None:
    """Retain loaded-checkpoint provenance in evaluate-only reports."""

    model = torch.nn.Linear(1, 1)
    checkpoint_path = tmp_path / "input.pt"
    torch.save(
        {
            "model": "mesh_transformer",
            "capacity": "tiny",
            "state_dict": model.state_dict(),
            "run_config": {"seed": 7},
            "source": {"relevant_source_sha256": "old-source"},
        },
        checkpoint_path,
    )
    args = SimpleNamespace(
        model="mesh_transformer",
        capacity="tiny",
        steps=0,
        cases_per_step=1,
        train_boundary_points=4,
        train_query_points=4,
        learning_rate=3.0e-4,
        weight_decay=1.0e-6,
        seed=17,
        validation_seed=19,
        evaluation_seed=23,
        evaluation_cases=1,
        evaluation_boundary_points=4,
        evaluation_query_points=4,
        harmonic_cases=0,
        device="cpu",
        dtype="float32",
        matmul_precision="highest",
        output_dir=tmp_path / "output",
        evaluate_only=True,
        checkpoint=checkpoint_path,
    )
    monkeypatch.setattr(training_benchmark, "_parse_args", lambda: args)
    monkeypatch.setattr(training_benchmark, "make_model", lambda *ignored: model)
    monkeypatch.setattr(
        training_benchmark, "evaluate_model", lambda *ignored, **kwargs: {}
    )
    monkeypatch.setattr(
        training_benchmark,
        "source_provenance",
        lambda: {"relevant_source_sha256": "current-source"},
    )
    monkeypatch.setattr(training_benchmark, "runtime_environment", lambda *ignored: {})

    training_benchmark.main()

    report = json.loads((args.output_dir / "mesh_transformer_tiny.json").read_text())
    metadata = report["input_checkpoint_metadata"]
    assert metadata["run_config"] == {"seed": 7}
    assert metadata["source"] == {"relevant_source_sha256": "old-source"}
    assert metadata["source_matches_evaluator"] is False
    assert report["source"] == {"relevant_source_sha256": "current-source"}
    assert report["checkpoint"] is None
