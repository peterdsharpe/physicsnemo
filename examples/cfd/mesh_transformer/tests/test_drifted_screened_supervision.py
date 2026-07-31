# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the drifted-screened supervision study."""

from __future__ import annotations

import drifted_screened_principal_part as base
import drifted_screened_supervision as study
import torch

DEVICE = torch.device("cpu")


def test_differentiable_layer_matches_validated_exact_evaluator() -> None:
    sample = base.build_pde_sample(
        123,
        split="in_distribution",
        n_boundary=16,
        n_query=12,
        device=DEVICE,
    )
    pairs = study.layer_pairs(
        sample.query_points,
        sample.boundary,
        drift=sample.drift,
        quadrature_order=4,
    )
    values = study.scaled_kernel(
        None,
        pairs,
        kappa=sample.kappa,
        drift_magnitude=float(torch.linalg.vector_norm(sample.drift)),
    )
    actual = study.integrate_scaled_kernel(values, pairs)
    expected = base.double_layer_influence(
        sample.query_points,
        sample.boundary,
        kappa=sample.kappa,
        drift=sample.drift,
        quadrature_order=4,
        model=None,
    )
    assert torch.allclose(actual, expected, atol=1.0e-15, rtol=1.0e-15)


def test_solution_loss_differentiates_through_trace_solve() -> None:
    torch.manual_seed(5)
    model = base.ScaledKernelModel("free_principal").double()
    sample = base.build_pde_sample(
        456,
        split="in_distribution",
        n_boundary=16,
        n_query=16,
        device=DEVICE,
    )
    kernel_loss, solution_loss = study.training_losses(
        model,
        sample,
        quadrature_order=4,
    )
    solution_loss.backward()
    gradient_norm = sum(
        float(parameter.grad.square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(kernel_loss)
    assert torch.isfinite(solution_loss)
    assert gradient_norm > 0.0
    assert model.singular_coefficient.grad is not None


def test_held_out_boundary_spectrum_changes_only_the_solution() -> None:
    training = base.build_pde_sample(
        789,
        split="in_distribution",
        n_boundary=24,
        n_query=24,
        device=DEVICE,
        solution_modes=study.TRAIN_SOLUTION_MODES,
    )
    held_out = base.build_pde_sample(
        789,
        split="in_distribution",
        n_boundary=24,
        n_query=24,
        device=DEVICE,
        solution_modes=study.HELD_OUT_SOLUTION_MODES,
    )
    assert set(study.TRAIN_SOLUTION_MODES).isdisjoint(study.HELD_OUT_SOLUTION_MODES)
    assert training.kappa == held_out.kappa
    assert torch.equal(training.drift, held_out.drift)
    assert torch.equal(training.boundary.points, held_out.boundary.points)
    assert torch.equal(training.query_points, held_out.query_points)
    assert not torch.equal(training.boundary_values, held_out.boundary_values)
    assert not torch.equal(training.target, held_out.target)


def test_every_arm_uses_the_same_free_principal_architecture() -> None:
    counts = []
    for _ in study.ARMS:
        model = base.ScaledKernelModel("free_principal")
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1
    assert counts[0] == 8_834


def test_small_run_preserves_scientific_schema() -> None:
    report = study.run_arm(
        arm="hybrid",
        seed=study.SEEDS[0],
        device=DEVICE,
        train_steps=2,
        train_boundary_points=12,
        train_query_points=12,
        train_quadrature_order=4,
        evaluation_cases=1,
        evaluation_boundary_points=16,
        evaluation_query_points=16,
        resolution_cases=1,
        resolutions=(16, 24),
        quadrature_order=4,
        check_quadrature_order=4,
        kernel_evaluation_pairs=64,
    )
    assert report["study"] == study.STUDY
    assert report["arm"] == "hybrid"
    assert set(report["pde_evaluation"]) == set(base.PDE_SPLIT_ORDER)
    assert set(report["resolution_evaluation"]) == set(base.RESOLUTION_SPLITS)
    assert report["boundary_spectrum_evaluation"]["solution_modes"] == list(
        study.HELD_OUT_SOLUTION_MODES
    )


def test_registered_protocol_constants() -> None:
    assert study.ARMS == ("pointwise", "solution", "hybrid")
    assert study.SEEDS == (17, 29, 43, 59, 71)
    assert study.TRAIN_STEPS == 4_000
    assert study.RESOLUTIONS == (64, 128, 256)
    assert study.TRAIN_SOLUTION_MODES == (0, 1, 2, 3)
    assert study.HELD_OUT_SOLUTION_MODES == (5, 6, 7, 8)
