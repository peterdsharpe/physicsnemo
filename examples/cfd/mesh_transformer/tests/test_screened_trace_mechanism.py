# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the screened trace-mechanism discriminator."""

from __future__ import annotations

import math

import pytest
import screened_trace_mechanism as study
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def test_stored_polygon_normals_and_canonical_double_jump_agree() -> None:
    boundary = study.unit_polygon(64, device=DEVICE, dtype=DTYPE)
    assert torch.all(
        torch.sum(boundary.cell_normals * boundary.cell_centroids, dim=-1) < 0
    )

    matrix = study.trace_matrix(
        boundary,
        layer="double",
        kappa=0.001,
        quadrature_order=64,
        zero_diagonal=False,
    )
    center = study.field_matrix(
        torch.zeros((1, 2), dtype=DTYPE),
        boundary,
        layer="double",
        kappa=0.001,
        quadrature_order=64,
    )
    ones = torch.ones(64, dtype=DTYPE)

    assert float((matrix @ ones).mean()) == pytest.approx(1.0, abs=5.0e-6)
    assert float((center @ ones).item()) == pytest.approx(1.0, abs=5.0e-6)


def test_zeroing_the_diagonal_changes_only_the_single_layer() -> None:
    boundary = study.unit_polygon(16, device=DEVICE, dtype=DTYPE)
    single_full = study.trace_matrix(
        boundary,
        layer="single",
        kappa=0.1,
        quadrature_order=32,
        zero_diagonal=False,
    )
    single_zero = study.trace_matrix(
        boundary,
        layer="single",
        kappa=0.1,
        quadrature_order=32,
        zero_diagonal=True,
    )
    double_full = study.trace_matrix(
        boundary,
        layer="double",
        kappa=0.1,
        quadrature_order=32,
        zero_diagonal=False,
    )
    double_zero = study.trace_matrix(
        boundary,
        layer="double",
        kappa=0.1,
        quadrature_order=32,
        zero_diagonal=True,
    )

    assert torch.all(single_full.diagonal() > 0)
    assert torch.count_nonzero(single_zero.diagonal()) == 0
    assert torch.equal(single_full - torch.diag(single_full.diagonal()), single_zero)
    assert torch.equal(double_full, double_zero)
    assert torch.all(double_full.diagonal() == 0.5)


def test_small_manufactured_cell_closes_under_the_full_dense_solve() -> None:
    boundary = study.unit_polygon(16, device=DEVICE, dtype=DTYPE)
    queries = study.interior_queries(
        device=DEVICE,
        dtype=DTYPE,
        angles_per_ring=8,
    )
    density = study.manufactured_densities(boundary)["zero_charge"]
    cell = study.evaluate_cell(
        boundary,
        queries,
        layer="single",
        density_name="zero_charge",
        true_density=density,
        kappa=0.05,
        quadrature_order=32,
        check_quadrature_order=16,
    )

    assert set(cell["methods"]) == set(study.METHOD_NAMES)
    assert cell["methods"]["dense_full"]["density_relative_l2"] < 1.0e-12
    assert cell["methods"]["dense_full"]["true_trace_relative_l2"] < 1.0e-12
    assert cell["methods"]["dense_full"]["field_relative_l2"] < 1.0e-12
    assert math.isfinite(cell["full_operator"]["condition_number"])


def test_registered_constants_match_the_preregistration() -> None:
    assert study.N_PANELS == 64
    assert study.KAPPAS == (0.3, 0.1, 0.05, 0.02, 0.01, 0.001)
    assert study.RICHARDSON_STEPS == 8
    assert study.SINGLE_COEFFICIENT == pytest.approx(1.0 / (2.0 * math.pi))
    assert study.DOUBLE_COEFFICIENT == pytest.approx(1.0 / (2.0 * math.pi))
    assert study.QUADRATURE_ORDER == 256
    assert study.CHECK_QUADRATURE_ORDER == 128
