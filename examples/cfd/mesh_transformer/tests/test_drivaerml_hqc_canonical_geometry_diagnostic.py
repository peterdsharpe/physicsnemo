# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the target-free H-QC canonical-geometry diagnostic."""

import numpy as np
import pytest
import torch
from drivaerml_hqc_canonical_geometry_diagnostic import (
    CENTER_ABS_TOLERANCE,
    CanonicalSourceBundle,
    _apply_canonical_geometry,
    _build_canonical_raw_geometry,
    _bundle_validity,
    _difference_is_exact,
    _difference_within,
    _finish_canonical_bundle,
    _forbidden_artifact_keys,
    _prediction_difference,
    _require_no_local_data,
    _target_free_subset,
)

from physicsnemo.mesh import DomainMesh, Mesh


def _two_triangle_mesh(translation: torch.Tensor | None = None) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    if translation is not None:
        points = points + translation
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [0, 3, 1]]),
        cell_data={
            "pMeanTrim": torch.tensor([101.0, 202.0]),
            "wallShearStressMeanTrim": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "unrelated": torch.tensor([7.0, 8.0]),
        },
        global_data={"L_ref": torch.tensor(2.0)},
    )


def test_target_free_subset_preserves_requested_topology_without_raw_values():
    mesh = _two_triangle_mesh()
    subset = _target_free_subset(mesh, np.array([1, 0]), Mesh)

    torch.testing.assert_close(
        subset.points[subset.cells],
        mesh.points[mesh.cells[torch.tensor([1, 0])]],
        rtol=0.0,
        atol=0.0,
    )
    assert set(subset.cell_data.keys()) == {
        "pMeanTrim",
        "wallShearStressMeanTrim",
    }
    assert torch.count_nonzero(subset.cell_data["pMeanTrim"]).item() == 0
    assert torch.count_nonzero(subset.cell_data["wallShearStressMeanTrim"]).item() == 0
    assert "unrelated" not in subset.cell_data


def test_canonical_bundle_is_translation_invariant_and_area_centered():
    mesh = _two_triangle_mesh()
    translated = _two_triangle_mesh(torch.tensor([8.0, -4.0, 2.0]))
    bundle = _finish_canonical_bundle(
        _build_canonical_raw_geometry(mesh),
        physical_length=2.0,
        model_reference_length=4.0,
    )
    translated_bundle = _finish_canonical_bundle(
        _build_canonical_raw_geometry(translated),
        physical_length=2.0,
        model_reference_length=4.0,
    )

    for field in ("points", "centroids", "areas", "normals"):
        torch.testing.assert_close(
            getattr(bundle, field),
            getattr(translated_bundle, field),
            rtol=0.0,
            atol=1.0e-15,
        )
    torch.testing.assert_close(
        translated_bundle.physical_center - bundle.physical_center,
        torch.tensor([8.0, -4.0, 2.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )
    validity = _bundle_validity(bundle, expected_cells=mesh.cells)
    assert validity["passed"]
    assert validity["maximum_area_center_deviation"] <= CENTER_ABS_TOLERANCE
    assert bundle.points.dtype == torch.float32
    assert bundle.centroids.dtype == torch.float32
    assert bundle.areas.dtype == torch.float32
    assert bundle.normals.dtype == torch.float32


def test_canonical_geometry_rejects_degenerate_or_nontriangular_topology():
    degenerate = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    with pytest.raises(ValueError, match="degenerate"):
        _build_canonical_raw_geometry(degenerate)

    segment = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        cells=torch.tensor([[0, 1]]),
    )
    with pytest.raises(ValueError, match="points must have shape"):
        _build_canonical_raw_geometry(segment)


def test_source_override_distinguishes_derived_from_full():
    source = Mesh(
        points=torch.tensor([[9.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.0, 1.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    original_points = source.points.clone()
    bundle = CanonicalSourceBundle(
        points=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        centroids=torch.tensor([[0.0, 1.0 / 3.0, 0.0]]),
        areas=torch.tensor([1.0]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        physical_center=torch.zeros(3, dtype=torch.float64),
        physical_length=1.0,
        model_reference_length=1.0,
    )

    _apply_canonical_geometry(source, bundle, "canonical_derived")
    assert torch.equal(source.points, original_points)
    assert torch.equal(source.cell_centroids, bundle.centroids)
    assert torch.equal(source.cell_areas, bundle.areas)
    assert torch.equal(source.cell_normals, bundle.normals)

    _apply_canonical_geometry(source, bundle, "canonical_full")
    assert torch.equal(source.points, bundle.points)


def test_prediction_gates_are_fieldwise_and_exactness_is_stricter():
    primary = {
        "pressure": torch.tensor([1.0, 2.0]),
        "wss": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    nearby = {
        "pressure": torch.tensor([1.0, 2.001]),
        "wss": primary["wss"].clone(),
    }
    difference = _prediction_difference(primary, nearby)

    assert _difference_within(difference, 1.0e-3)
    assert not _difference_is_exact(difference)
    assert _difference_is_exact(_prediction_difference(primary, primary))


def test_local_data_and_artifact_vocabulary_guards_fail_closed():
    clean = DomainMesh(
        interior=Mesh(points=torch.zeros(1, 3)),
        boundaries={
            "vehicle": Mesh(
                points=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                cells=torch.tensor([[0, 1, 2]]),
            )
        },
    )
    _require_no_local_data(clean)

    dirty = DomainMesh(
        interior=Mesh(
            points=torch.zeros(1, 3),
            point_data={"pressure": torch.ones(1)},
        ),
        boundaries={"vehicle": clean.boundaries["vehicle"]},
    )
    with pytest.raises(ValueError, match="retains local data"):
        _require_no_local_data(dirty)

    assert _forbidden_artifact_keys({"metric": {"relative_l2": 0.0}}) == []
    assert _forbidden_artifact_keys(
        {
            "nested": {"target_error": 0.0},
            "truth_available": False,
            "force_metric": 0.0,
        }
    ) == ["nested.target_error", "truth_available", "force_metric"]
