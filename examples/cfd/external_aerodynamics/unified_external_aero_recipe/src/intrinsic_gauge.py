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
try:
    # Post-#1770 layout: Horvitz-Thompson-aware effective measures
    # (geometric measure x composed subsampling weights).
    from physicsnemo.mesh.calculus.measure import cell_measures
except (ModuleNotFoundError, ImportError):  # pragma: no cover - pre-merge trees

    def cell_measures(mesh):
        """Pre-#1770 fallback: bare geometric measure. Under uniform
        subsampling the unweighted statistic has the same expectation as
        the HT-weighted one, so calibration and evaluation remain
        consistent across tree versions."""
        return mesh.cell_areas

__all__ = ["ComputeIntrinsicReferenceLength", "measure_weighted_rms_radius"]


def measure_weighted_rms_radius(mesh: Mesh) -> torch.Tensor:
    r"""Measure-weighted RMS radius of *mesh* about its measure-weighted centroid.

    Thin wrapper over the model's own gauge reduction
    (:func:`physicsnemo.experimental.nn.mesh_attention.measure_weighted_rms_radius`)
    so this transform and :class:`MeshTransformer`'s built-in intrinsic gauge
    can never drift apart in formula, float64 accumulation, or degenerate-
    geometry validation.  ``test_matches_model_intrinsic_gauge`` pins the
    agreement.

    The one intentional difference is the **weights**: this transform runs
    after ``SubsampleMesh`` and is estimating the statistic of the
    full-resolution geometry, so it weights by ``cell_measures`` (geometric
    measure times composed Horvitz-Thompson inclusion weights).  The model
    weights by bare ``cell_areas``, its own quadrature measure, so its gauge
    stays consistent with the operator it normalizes.  On a mesh carrying no
    measure weights the two coincide bitwise.

    Returns
    -------
    torch.Tensor
        0-dim tensor in the mesh's coordinate units and dtype.
    """
    ### Imported lazily: this module is loaded by the dataset pipeline for
    ### every model on an intrinsic-gauge dataset, including ones that ignore
    ### the gauge entirely, and there is no reason to drag the mesh_attention
    ### stack (or its ExperimentalFeatureWarning) into those builds.
    from physicsnemo.experimental.nn.mesh_attention import (
        measure_weighted_rms_radius as model_gauge,
    )

    return model_gauge(cell_measures(mesh), mesh.cell_centroids)


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
