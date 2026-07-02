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

"""Tests for Mesh serialization round-trips (memmap and pickle).

TensorDict records zero-element tensors in memmap metadata but does not write a
backing file for them. These tests verify that mesh serialization reconstructs
those tensors with their original shape and dtype.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d

### Memmap Round-Trip Tests ###


class TestMemmapRoundTrip:
    """Tests for Mesh.save() / Mesh.load() (memmap) round-trip."""

    def test_regular_mesh(self, tmp_path):
        """Regular mesh with non-empty cells survives memmap round-trip."""
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.cells.shape == mesh.cells.shape
        assert torch.equal(loaded.cells, mesh.cells)
        assert torch.allclose(loaded.points, mesh.points)

    def test_point_cloud_direct(self, tmp_path):
        """Point cloud created via constructor (cells=None) survives memmap round-trip."""
        pc = Mesh(points=torch.randn(10, 3))

        assert pc.cells is not None
        assert pc.cells.shape == (0, 1)

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.cells is not None, "cells should not be None after memmap load"
        assert loaded.cells.shape == (0, 1)
        assert loaded.cells.dtype == torch.long

    def test_point_cloud_via_to_point_cloud(self, tmp_path):
        """Point cloud created via to_point_cloud() survives memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()

        assert pc.cells.shape == (0, 1)

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.cells is not None, "cells should not be None after memmap load"
        assert loaded.cells.shape == (0, 1)
        assert loaded.cells.dtype == torch.long

    def test_n_manifold_dims_after_load(self, tmp_path):
        """n_manifold_dims works correctly after memmap round-trip of point cloud."""
        pc = two_triangles_2d.load().to_point_cloud()
        assert pc.n_manifold_dims == 0

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.n_manifold_dims == 0

    def test_n_cells_after_load(self, tmp_path):
        """n_cells returns 0 for a point cloud after memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        assert pc.n_cells == 0

        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert loaded.n_cells == 0

    def test_repr_after_load(self, tmp_path):
        """repr does not raise for a point cloud after memmap round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        result = repr(loaded)
        assert "n_manifold_dims=0, n_spatial_dims=2" in result
        assert "n_cells=0" in result

    def test_points_preserved(self, tmp_path):
        """Point coordinates survive memmap round-trip exactly."""
        pc = two_triangles_2d.load().to_point_cloud()
        pc.save(tmp_path / "pc.pt")
        loaded = Mesh.load(tmp_path / "pc.pt")

        assert torch.allclose(loaded.points, pc.points)

    def test_with_point_data(self, tmp_path):
        """point_data survives memmap round-trip alongside empty cells."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            point_data={"velocity": torch.randn(5, 3), "pressure": torch.randn(5)},
        )
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.cells.shape == (0, 1)
        assert "velocity" in loaded.point_data.keys()
        assert "pressure" in loaded.point_data.keys()
        assert torch.allclose(
            loaded.point_data["velocity"], mesh.point_data["velocity"]
        )

    def test_with_cell_and_global_data(self, tmp_path):
        """cell_data and global_data survive memmap round-trip on a real mesh."""
        # Use a 2-triangle mesh so n_cells > 0 (cell_data is non-trivial).
        mesh = two_triangles_2d.load()
        mesh.point_data["p_scalar"] = torch.randn(mesh.n_points)
        mesh.point_data["p_vector"] = torch.randn(mesh.n_points, 3)
        mesh.cell_data["c_scalar"] = torch.randn(mesh.n_cells)
        mesh.cell_data["c_vector"] = torch.randn(mesh.n_cells, 3)
        mesh.global_data["g_scalar"] = torch.tensor(1.5)
        mesh.global_data["g_vector"] = torch.tensor([1.0, 2.0, 3.0])

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert set(loaded.point_data.keys()) == {"p_scalar", "p_vector"}
        assert set(loaded.cell_data.keys()) == {"c_scalar", "c_vector"}
        assert set(loaded.global_data.keys()) == {"g_scalar", "g_vector"}

        for field in ("point_data", "cell_data", "global_data"):
            for key in getattr(mesh, field).keys():
                assert torch.allclose(
                    getattr(loaded, field)[key], getattr(mesh, field)[key]
                ), f"{field}[{key!r}] mismatch after round-trip"

    def test_all_dimension_configs(self, tmp_path, dims_all):
        """All (n_spatial_dims, n_manifold_dims) configurations survive memmap round-trip."""
        from test.mesh.conftest import create_simple_mesh

        n_spatial_dims, n_manifold_dims = dims_all
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims)
        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells is not None
        assert loaded.n_manifold_dims == mesh.n_manifold_dims
        assert loaded.n_spatial_dims == mesh.n_spatial_dims
        assert loaded.n_points == mesh.n_points
        assert loaded.n_cells == mesh.n_cells

    @pytest.mark.parametrize("vertices_per_cell", [1, 2, 3, 4])
    def test_empty_cells_preserve_shape_and_dtype(self, tmp_path, vertices_per_cell):
        """Empty connectivity retains its manifold dimension and dtype."""
        mesh = Mesh(
            points=torch.randn(5, 4),
            cells=torch.empty((0, vertices_per_cell), dtype=torch.int32),
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.cells.shape == (0, vertices_per_cell)
        assert loaded.cells.dtype == torch.int32
        assert loaded.n_manifold_dims == vertices_per_cell - 1

    def test_fully_empty_geometry_preserves_shape_and_dtype(self, tmp_path):
        """A mesh with no points or cells can be loaded without losing metadata."""
        mesh = Mesh(
            points=torch.empty((0, 4), dtype=torch.float64),
            cells=torch.empty((0, 3), dtype=torch.int32),
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.points.shape == (0, 4)
        assert loaded.points.dtype == torch.float64
        assert loaded.cells.shape == (0, 3)
        assert loaded.cells.dtype == torch.int32
        assert loaded.n_spatial_dims == 4
        assert loaded.n_manifold_dims == 2

    def test_zero_element_data_preserves_nested_shapes_and_dtypes(self, tmp_path):
        """Zero-width user fields survive at every association and nesting level."""
        point_data = TensorDict(
            {
                "embedding": torch.empty((3, 0), dtype=torch.float16),
                "nested": TensorDict(
                    {"features": torch.empty((3, 2, 0), dtype=torch.float64)},
                    batch_size=[3],
                ),
            },
            batch_size=[3],
        )
        mesh = Mesh(
            points=torch.randn(3, 3),
            cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            point_data=point_data,
            cell_data={"embedding": torch.empty((1, 0), dtype=torch.int16)},
            global_data={"embedding": torch.empty((2, 0), dtype=torch.float64)},
        )

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")

        assert loaded.point_data["embedding"].shape == (3, 0)
        assert loaded.point_data["embedding"].dtype == torch.float16
        assert loaded.point_data["nested", "features"].shape == (3, 2, 0)
        assert loaded.point_data["nested", "features"].dtype == torch.float64
        assert loaded.cell_data["embedding"].shape == (1, 0)
        assert loaded.cell_data["embedding"].dtype == torch.int16
        assert loaded.global_data["embedding"].shape == (2, 0)
        assert loaded.global_data["embedding"].dtype == torch.float64

    def test_empty_cached_adjacency_round_trip(self, tmp_path):
        """A cached adjacency with no indices remains loadable and empty."""
        mesh = Mesh(points=torch.randn(4, 3))
        expected = mesh.get_point_to_points_adjacency().to_list()

        mesh.save(tmp_path / "mesh.pt")
        loaded = Mesh.load(tmp_path / "mesh.pt")
        adjacency = loaded.get_point_to_points_adjacency()

        assert adjacency.to_list() == expected
        assert adjacency.indices.shape == (0,)
        assert adjacency.indices.dtype == torch.int64

    def test_domain_mesh_preserves_nested_empty_tensors(self, tmp_path):
        """Domain, interior, and boundary empties all survive one round-trip."""
        interior = Mesh(
            points=torch.randn(4, 3),
            cells=torch.empty((0, 4), dtype=torch.int32),
        )
        boundary = Mesh(
            points=torch.empty((0, 3), dtype=torch.float64),
            cells=torch.empty((0, 3), dtype=torch.int32),
            point_data={"embedding": torch.empty((0, 2), dtype=torch.float16)},
        )
        domain = DomainMesh(
            interior=interior,
            boundaries={"empty": boundary},
            global_data={"embedding": torch.empty((2, 0), dtype=torch.float64)},
        )

        domain.save(tmp_path / "domain.pt")
        loaded = DomainMesh.load(tmp_path / "domain.pt")

        assert loaded.interior.cells.shape == (0, 4)
        assert loaded.interior.cells.dtype == torch.int32
        assert loaded.boundaries["empty"].points.shape == (0, 3)
        assert loaded.boundaries["empty"].points.dtype == torch.float64
        assert loaded.boundaries["empty"].cells.shape == (0, 3)
        assert loaded.boundaries["empty"].cells.dtype == torch.int32
        assert loaded.boundaries["empty"].point_data["embedding"].shape == (0, 2)
        assert loaded.boundaries["empty"].point_data["embedding"].dtype == torch.float16
        assert loaded.global_data["embedding"].shape == (2, 0)
        assert loaded.global_data["embedding"].dtype == torch.float64


### Pickle (torch.save / torch.load) Round-Trip Tests ###


class TestTorchSaveLoadRoundTrip:
    """Tests for torch.save() / torch.load() (pickle) round-trip.

    These should already work without the Mesh.load() fix, since pickle
    preserves empty tensors. Included as regression tests.
    """

    def test_regular_mesh(self, tmp_path):
        """Regular mesh survives torch.save/load round-trip."""
        mesh = two_triangles_2d.load()
        path = tmp_path / "mesh.pt"
        torch.save(mesh, path)
        loaded = torch.load(path, weights_only=False)

        assert loaded.cells is not None
        assert torch.equal(loaded.cells, mesh.cells)

    def test_point_cloud(self, tmp_path):
        """Point cloud survives torch.save/load round-trip."""
        pc = two_triangles_2d.load().to_point_cloud()
        path = tmp_path / "pc.pt"
        torch.save(pc, path)
        loaded = torch.load(path, weights_only=False)

        assert loaded.cells is not None
        assert loaded.cells.shape == (0, 1)
        assert loaded.n_manifold_dims == 0
        assert repr(loaded)  # should not raise
