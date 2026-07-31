# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the production common-master audit."""

import drivaerml_common_master_audit as audit
import numpy as np
from drivaerml_common_master_audit import Support, _repair_empty_supports


def test_restriction_repair_moves_only_empty_supports():
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    original_normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    support = Support(
        name="normal_aware_centroidal_cover",
        points=original_points.copy(),
        normals=original_normals.copy(),
        normal_length_scale=0.5,
        definition={},
    )
    accumulation = {
        "measures": np.array([1.0, 0.0, 2.0], dtype=np.float64),
        # Deliberately different nonempty centroids/normals: a restriction
        # repair must ignore these rather than perform an extra Lloyd update.
        "centroid_sums": np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "normal_sums": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "farthest": {
            "indices": np.array([17], dtype=np.int64),
            "points": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
            "normals": np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        },
    }

    repaired_indices, repaired_master_cells = _repair_empty_supports(
        support,
        accumulation,
    )

    assert repaired_indices == [1]
    assert repaired_master_cells == [17]
    np.testing.assert_array_equal(support.points[[0, 2]], original_points[[0, 2]])
    np.testing.assert_array_equal(support.normals[[0, 2]], original_normals[[0, 2]])
    np.testing.assert_array_equal(
        support.points[1],
        np.array([7.0, 8.0, 9.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        support.normals[1],
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


def test_field_restriction_does_not_add_an_undeclared_lloyd_update(monkeypatch):
    original_points = np.eye(3, dtype=np.float32)
    original_normals = np.eye(3, dtype=np.float32)
    support = Support(
        name="normal_aware_centroidal_cover",
        points=original_points.copy(),
        normals=original_normals.copy(),
        normal_length_scale=0.5,
        definition={},
    )
    first = {
        "measures": np.array([1.0, 0.0, 2.0], dtype=np.float64),
        "centroid_sums": np.full((3, 3), 100.0, dtype=np.float64),
        "normal_sums": np.full((3, 3), 100.0, dtype=np.float64),
        "cp_sums": np.zeros(3, dtype=np.float64),
        "wss_sums": np.zeros((3, 3), dtype=np.float64),
        "farthest": {
            "indices": np.array([17], dtype=np.int64),
            "points": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
            "normals": np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        },
        "neighbor_backend": "test",
    }
    second = {
        "measures": np.ones(3, dtype=np.float64),
        "centroid_sums": np.zeros((3, 3), dtype=np.float64),
        "normal_sums": np.zeros((3, 3), dtype=np.float64),
        "cp_sums": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "wss_sums": np.ones((3, 3), dtype=np.float64),
        "neighbor_backend": "test",
    }
    calls = iter((first, second))
    monkeypatch.setattr(
        audit, "_accumulate_assignment", lambda *args, **kwargs: next(calls)
    )

    fields, history, backend = audit._restrict_fields(
        case=object(),
        support=support,
        chunk_cells=1,
        workers=1,
        repair_pool_size=1,
        allow_repair=True,
    )

    np.testing.assert_array_equal(support.points[[0, 2]], original_points[[0, 2]])
    np.testing.assert_array_equal(support.normals[[0, 2]], original_normals[[0, 2]])
    assert history[0]["repair_scope"] == "empty_supports_only"
    assert history[0]["repaired_support_indices"] == [1]
    np.testing.assert_array_equal(fields.cp, np.array([1.0, 2.0, 3.0]))
    assert backend == "test"
