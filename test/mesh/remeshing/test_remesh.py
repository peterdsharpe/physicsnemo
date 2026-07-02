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

"""Tests for the ACVD-based ``remesh`` entry point."""

import pytest
import torch

from physicsnemo.mesh import Mesh

pytest.importorskip("pyacvd")

from physicsnemo.mesh.primitives.planar import unit_square  # noqa: E402
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral  # noqa: E402
from physicsnemo.mesh.remeshing import remesh  # noqa: E402


def test_remesh_basic_sphere():
    """remesh round-trips a triangle surface and yields a valid 2D-in-3D mesh."""
    mesh = sphere_icosahedral.load(subdivisions=3)
    out = remesh(mesh, n_clusters=100)

    assert isinstance(out, Mesh)
    assert out.n_cells > 0
    assert out.n_manifold_dims == 2 and out.n_spatial_dims == 3
    assert out.points.device == mesh.points.device
    assert not torch.is_floating_point(out.cells)  # cells stay integer


def test_remesh_preserves_dtype():
    """remesh restores the input floating dtype even though pyvista round-trips
    through float32."""
    base = sphere_icosahedral.load(subdivisions=3)
    mesh = Mesh(points=base.points.double(), cells=base.cells)  # float64
    out = remesh(mesh, n_clusters=80)
    assert out.points.dtype == torch.float64
    assert not torch.is_floating_point(out.cells)


def test_remesh_rejects_non_surface():
    """remesh guards against non-2D-in-3D inputs with a clear NotImplementedError
    instead of a confusing downstream pyacvd failure."""
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2]])  # 1D curve in 3D
    mesh = Mesh(points=pts, cells=cells)
    with pytest.raises(NotImplementedError, match="2D triangle surface"):
        remesh(mesh, n_clusters=2)


def test_remesh_planar_2d_mesh():
    """PyVista's zero-padded third coordinate supports 2D embeddings."""
    mesh = unit_square.load(subdivisions=8)

    out = remesh(mesh, n_clusters=20)

    assert out.n_cells > 0
    assert out.n_manifold_dims == 2
    assert out.n_spatial_dims == 2


def test_remesh_preserves_micro_scale_geometry():
    """Absolute repair tolerances must not erase a valid small mesh."""
    reference_mesh = sphere_icosahedral.load(subdivisions=2).to(torch.float64)
    mesh = Mesh(points=reference_mesh.points * 1e-5, cells=reference_mesh.cells)

    reference = remesh(reference_mesh, n_clusters=40)
    out = remesh(mesh, n_clusters=40)

    assert torch.equal(out.cells, reference.cells)
    torch.testing.assert_close(out.points / 1e-5, reference.points)


def test_remesh_preserves_large_float64_translation():
    """Centering before float32 interop keeps nearby translated points distinct."""
    base = sphere_icosahedral.load(subdivisions=2).to(torch.float64)
    offset = base.points.new_tensor([1e8, 1e8, 1e8])
    mesh = Mesh(points=base.points + offset, cells=base.cells)

    reference = remesh(base, n_clusters=40)
    out = remesh(mesh, n_clusters=40)

    assert out.points.dtype == torch.float64
    assert torch.equal(out.cells, reference.cells)
    torch.testing.assert_close(out.points - offset, reference.points, atol=2e-8, rtol=0)


def test_remesh_finite_large_coordinates_do_not_overflow_midpoint():
    base = unit_square.load(subdivisions=4).to(torch.float64)
    points = base.points * 2e307
    points[:, 0] += 8e307
    mesh = Mesh(
        points=points,
        cells=base.cells,
    )

    out = remesh(mesh, n_clusters=10)

    assert out.points.isfinite().all()
    assert out.n_cells > 0


def test_remesh_preserves_global_data():
    base = sphere_icosahedral.load(subdivisions=2)
    mesh = Mesh(
        points=base.points,
        cells=base.cells,
        global_data={"case_id": torch.tensor(7)},
    )

    out = remesh(mesh, n_clusters=40)

    assert torch.equal(out.global_data["case_id"], torch.tensor(7))


def test_remesh_rejects_mesh_without_cells():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.empty((0, 3), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="empty mesh"):
        remesh(mesh, n_clusters=3)


@pytest.mark.parametrize(
    "points",
    [
        torch.zeros((3, 2)),
        torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [float("inf"), 0.0], [0.0, 1.0]]),
    ],
)
def test_remesh_rejects_invalid_extent(points):
    mesh = Mesh(points=points, cells=torch.tensor([[0, 1, 2]]))

    with pytest.raises(ValueError, match="zero or non-finite extent"):
        remesh(mesh, n_clusters=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_remesh_restores_cuda_reduced_precision():
    mesh = sphere_icosahedral.load(subdivisions=2).to(
        device="cuda", dtype=torch.float16
    )

    out = remesh(mesh, n_clusters=40)

    assert out.points.device.type == "cuda"
    assert out.points.dtype == torch.float16
    assert out.cells.device.type == "cuda"
    assert out.points.isfinite().all()


@pytest.mark.parametrize("n_clusters", [-1, 0, 1, 2])
def test_remesh_rejects_too_few_clusters(n_clusters):
    mesh = sphere_icosahedral.load(subdivisions=1)

    with pytest.raises(ValueError, match="n_clusters must be at least 3"):
        remesh(mesh, n_clusters=n_clusters)
