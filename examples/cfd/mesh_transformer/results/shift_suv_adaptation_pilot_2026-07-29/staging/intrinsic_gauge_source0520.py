# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-era intrinsic reference-length transform.

This task-local copy deliberately uses bare geometric cell areas.  That is
the path taken by the source-training snapshot: the snapshot predated
``physicsnemo.mesh.calculus.measure``, so its compatibility fallback resolved
to ``mesh.cell_areas``.  Keeping that behavior here makes the source-domain
canary a test of the saved checkpoint rather than a test of a newer measure
contract.
"""

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import Mesh

__all__ = ["ComputeIntrinsicReferenceLength", "measure_weighted_rms_radius"]


def measure_weighted_rms_radius(mesh: Mesh) -> torch.Tensor:
    """Return the geometric-area-weighted RMS radius of cell centroids."""
    weights = mesh.cell_areas
    centroids = mesh.cell_centroids
    total = weights.sum()
    center = (weights[:, None] * centroids).sum(dim=0) / total
    radius_squared = (
        weights * (centroids - center).square().sum(dim=-1)
    ).sum() / total
    return radius_squared.sqrt()


@register()
class ComputeIntrinsicReferenceLength(MeshTransform):
    """Write a checkpoint-era intrinsic reference length to ``global_data``."""

    def __init__(
        self,
        scale_constant: float,
        field_name: str = "reference_length",
    ) -> None:
        super().__init__()
        if not (
            isinstance(scale_constant, (int, float))
            and not isinstance(scale_constant, bool)
            and float(scale_constant) > 0.0
            and float(scale_constant) == float(scale_constant)
        ):
            raise ValueError(
                "scale_constant must be a finite positive number, got "
                f"{scale_constant!r}"
            )
        if not isinstance(field_name, str) or not field_name:
            raise ValueError(f"field_name must be a non-empty str, got {field_name!r}")
        self.scale_constant = float(scale_constant)
        self.field_name = field_name

    def __call__(self, mesh: Mesh) -> Mesh:
        reference = self.scale_constant * measure_weighted_rms_radius(mesh)
        new_global = mesh.global_data.clone()
        new_global[self.field_name] = reference
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=new_global,
        )

    def extra_repr(self) -> str:
        return f"scale_constant={self.scale_constant}, field_name={self.field_name!r}"
