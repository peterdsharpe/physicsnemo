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

"""Regression tests for synchronization-free mesh linear algebra."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.transformations.geometric import transform

_CUDA = torch.cuda.is_available()


def _curve_mesh(device: torch.device | str) -> Mesh:
    """Return a non-axis-aligned 1-manifold with all normal caches populated."""
    mesh = Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.0, 0.25], [1.5, 1.5]],
            dtype=torch.float64,
            device=device,
        ),
        cells=torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device),
    )
    _ = mesh.cell_areas
    _ = mesh.cell_normals
    _ = mesh.point_normals
    return mesh


def _triangle_mesh(device: torch.device | str) -> Mesh:
    return Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.25, 0.1], [0.2, 1.1]],
            dtype=torch.float64,
            device=device,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.long, device=device),
    )


def test_cotan_inv_ex_matches_previous_inverse(device, monkeypatch):
    """The non-checking inverse must preserve the previous FEM weights."""
    mesh = _triangle_mesh(device)
    actual_weights, actual_edges = compute_cotan_weights_fem(mesh)

    def previous_inverse(matrix, *, check_errors=False):
        del check_errors
        return SimpleNamespace(inverse=torch.linalg.inv(matrix))

    monkeypatch.setattr(torch.linalg, "inv_ex", previous_inverse)
    expected_weights, expected_edges = compute_cotan_weights_fem(mesh)

    torch.testing.assert_close(actual_edges, expected_edges)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_transformed_normal_caches_match_previous_solve(device):
    """Cached normal propagation must match the previous checked solve."""
    mesh = _curve_mesh(device)
    matrix = torch.tensor(
        [[1.75, 0.3], [-0.2, 1.25]], dtype=mesh.points.dtype, device=device
    )

    original_cell_normals = mesh.cell_normals
    original_point_normals = mesh.point_normals
    original_areas = mesh.cell_areas
    transformed = transform(mesh, matrix, assume_invertible=True)

    cell_raw = torch.linalg.solve(matrix.T, original_cell_normals.T).T
    point_raw = torch.linalg.solve(matrix.T, original_point_normals.T).T
    det = matrix.det()

    torch.testing.assert_close(
        transformed._cache["cell", "normals"],
        det.sign() * F.normalize(cell_raw, dim=-1),
    )
    torch.testing.assert_close(
        transformed._cache["point", "normals"],
        det.sign() * F.normalize(point_raw, dim=-1),
    )
    torch.testing.assert_close(
        transformed._cache["cell", "areas"],
        original_areas * det.abs() * cell_raw.norm(dim=-1),
    )


@pytest.mark.skipif(not _CUDA, reason="CUDA required to detect host synchronizations")
def test_cotan_weight_inverse_is_sync_free(monkeypatch):
    """FEM Gram-matrix inversion must not synchronize CUDA for error checks."""
    device = torch.device("cuda")
    mesh = _triangle_mesh(device)
    _ = mesh.cell_areas

    # Topology extraction has independently data-dependent output. Replace it
    # with the known topology so this test isolates the fixed-size inversion.
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]], device=device)
    inverse = torch.arange(3, device=device)
    from physicsnemo.mesh.utilities import _topology

    monkeypatch.setattr(_topology, "extract_unique_edges", lambda _: (edges, inverse))

    # Warm lazy CUDA-library initialization and allocator growth outside the guard.
    compute_cotan_weights_fem(mesh)
    torch.cuda.synchronize()

    # The surrounding topology path contains separate data-dependent indexing,
    # so guard the inversion call itself. The called assertion also makes this a
    # regression test against reverting to the synchronizing ``linalg.inv``.
    original_inv_ex = torch.linalg.inv_ex
    called = False

    def checked_inv_ex(*args, **kwargs):
        nonlocal called
        called = True
        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return original_inv_ex(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)

    monkeypatch.setattr(torch.linalg, "inv_ex", checked_inv_ex)
    compute_cotan_weights_fem(mesh)
    torch.cuda.synchronize()
    assert called


@pytest.mark.skipif(not _CUDA, reason="CUDA required to detect host synchronizations")
def test_cached_normal_transformation_is_sync_free():
    """Inverse-transpose cache propagation must not synchronize CUDA."""
    device = torch.device("cuda")
    mesh = _curve_mesh(device)
    matrix = torch.tensor(
        [[1.75, 0.3], [-0.2, 1.25]], dtype=mesh.points.dtype, device=device
    )

    transform(mesh, matrix, assume_invertible=True)
    torch.cuda.synchronize()

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        transform(mesh, matrix, assume_invertible=True)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
