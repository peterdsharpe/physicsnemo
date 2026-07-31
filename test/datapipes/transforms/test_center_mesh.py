# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the mesh centering transform."""

import torch

from physicsnemo.datapipes.transforms.mesh import CenterMesh
from physicsnemo.mesh import DomainMesh, Mesh


def test_center_mesh_optionally_stores_subtracted_center():
    mesh = Mesh(
        points=torch.tensor(
            [
                [2.0, 1.0, -1.0],
                [4.0, 3.0, 1.0],
            ]
        ),
        global_data={"case_id": torch.tensor(7)},
    )

    centered = CenterMesh(
        use_area_weighting=False,
        store_center_as="center",
    )(mesh)

    torch.testing.assert_close(centered.points.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(
        centered.global_data["center"],
        torch.tensor([3.0, 2.0, 0.0]),
    )
    assert "center" not in mesh.global_data


def test_center_mesh_stores_domain_center_at_domain_level():
    domain = DomainMesh(
        interior=Mesh(
            points=torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            )
        ),
        boundaries={
            "wall": Mesh(
                points=torch.tensor(
                    [
                        [10.0, 0.0, 0.0],
                        [12.0, 0.0, 0.0],
                    ]
                )
            )
        },
    )

    centered = CenterMesh(
        use_area_weighting=False,
        store_center_as="center",
    ).apply_to_domain(domain)

    torch.testing.assert_close(
        centered.global_data["center"], torch.tensor([3.0, 0.0, 0.0])
    )
    torch.testing.assert_close(centered.interior.points.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(
        centered.boundaries["wall"].points,
        torch.tensor(
            [
                [7.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]
        ),
    )
    assert "center" not in centered.interior.global_data


def test_center_mesh_default_does_not_add_metadata():
    mesh = Mesh(points=torch.tensor([[2.0, 0.0], [4.0, 0.0]]))

    centered = CenterMesh(use_area_weighting=False)(mesh)

    assert not centered.global_data.keys()
