# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Phase-0 cluster/ancestry projection prototype."""

from phase0_common_surface import _partition_diagnostics, _sphere


def test_cluster_ancestry_map_preserves_linear_surface_totals():
    coarse = _sphere(1)
    result = _partition_diagnostics(_sphere(2), coarse.cell_centroids)

    assert result["n_empty_clusters"] == 0
    assert result["area_relative_error"] < 1.0e-14
    assert result["constant_max_abs_error"] < 1.0e-14
    assert result["scalar_integral_relative_error"] < 1.0e-14
    assert result["coarse_restrict_prolong_roundtrip_max_abs"] < 1.0e-14
    assert result["force_relative_error"] < 1.0e-14
    assert result["wss_tangency_max_abs"] < 1.0e-14
    # Moment is not a linear total of piecewise-constant traction at the
    # cluster centroid; this is a measured projection floor, not an identity.
    assert result["moment_relative_error_from_piecewise_constant_traction"] > 0.0
