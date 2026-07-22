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

"""Intrinsic reference-length gauge (cross-family transfer fix, 2026-07-22).

Recipe-local module registered into the global datapipe component registry.
Lives in its own module (not ``domain_transforms.py``) deliberately: the
running all-boundary training chains consume ``TopologyAwareDomainMeshReader``
from that file, and syncing it mid-chain would change their subsample RNG —
this module plus one import line in ``datasets.py`` is inert for them.

Why this transform exists: the DrivAerML surface arms declared a FIXED
``reference_length: 8.0`` (the tunnel scale in L_ref units — the
measure-weighted RMS radius of the full DrivAerML boundary set).  That
constant silently encodes DrivAerML's geometry statistics: on SHIFT-SUV the
same constant put the norm-free drive stream outside its stable operating
range and every MeshTransformer checkpoint diverged to non-physical output
(notebook @sec-nb-crossfam-verdict), with a clean gauge dose-response
(1.3e15 -> 5.1 -> 1.7 relative-L2 at gauge 8 -> 16 -> 48) localizing the
failure to the gauge.  The exact flaw was called in the external review:
"physical scaling equivariance is not presently guaranteed".

The repair: make the gauge intrinsic but keep the OPERATING POINT.  The
stable regime is gauge >> source spread (measured dose-response on the
all-boundary arm: NaN at 1, ~1e24 at 2, healthy >= 6, where the vehicle
spread is ~1), so a naive "gauge = my own RMS radius" would land in the
overflow zone.  Instead::

    reference_length = scale_constant * r_RMS(sample)

where ``r_RMS`` is the sample's measure-weighted RMS radius about its
measure-weighted centroid, and ``scale_constant`` is calibrated ONCE on the
DrivAerML TRAIN split as ``8.0 / mean(r_RMS)`` — DrivAerML samples then
reproduce ~8.0 (existing checkpoints see approximately unchanged inputs)
and any other geometry family lands at the same *relative* operating point
instead of DrivAerML's absolute one.

Invariances (tested): translation-invariant (centroid-centered),
rotation-invariant (radius norms), and similarity-COVARIANT — scaling the
geometry by ``s`` scales the gauge by exactly ``s`` (the effective cell
measures scale by ``s^m`` but cancel in the weighted mean; no
measure-divided-by-distance anywhere, per the dimension-generic measure
discipline).  Subsample-robust: ``cell_measures`` composes Horvitz-Thompson
measure weights, so the statistic computed after ``SubsampleMesh`` is an
unbiased estimate of the full-resolution value.
"""

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures

__all__ = ["ComputeIntrinsicReferenceLength", "measure_weighted_rms_radius"]


def measure_weighted_rms_radius(mesh: Mesh) -> torch.Tensor:
    r"""Measure-weighted RMS radius of *mesh* about its measure-weighted centroid.

    .. math::

       \bar c = \frac{\sum_i w_i c_i}{\sum_i w_i},\qquad
       r_\mathrm{RMS} = \sqrt{\frac{\sum_i w_i \lVert c_i-\bar c\rVert^2}
                                    {\sum_i w_i}}

    with :math:`c_i` the cell centroids and :math:`w_i` the effective cell
    measures (geometric measure times any composed Horvitz-Thompson
    subsampling weights).

    Returns
    -------
    torch.Tensor
        0-dim tensor in the mesh's coordinate units and dtype.
    """
    weights = cell_measures(mesh)
    centroids = mesh.cell_centroids
    total = weights.sum()
    center = (weights[:, None] * centroids).sum(dim=0) / total
    r_squared = (weights * (centroids - center).square().sum(dim=-1)).sum() / total
    return r_squared.sqrt()


@register()
class ComputeIntrinsicReferenceLength(MeshTransform):
    r"""Write ``global_data[field_name] = scale_constant * r_RMS(mesh)``.

    Drop-in replacement for the fixed ``SetGlobalField {reference_length: X}``
    gauge (see the module docstring for the physics rationale and the
    calibration convention for ``scale_constant``).  Inert for models that
    do not declare ``reference_length_key``.

    Parameters
    ----------
    scale_constant : float
        Multiplier applied to the sample's measure-weighted RMS radius.
        Calibrate once on the reference family's TRAIN split so that family
        reproduces its previously-fixed gauge value; never re-fit per
        evaluation set.
    field_name : str
        The ``global_data`` key to write.  Default ``"reference_length"``.
    """

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
            and float(scale_constant) == float(scale_constant)  # not NaN
        ):
            raise ValueError(
                f"scale_constant must be a finite positive number, got "
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
