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

r"""The intrinsic scale gauge: measure-weighted RMS radius (radius of gyration).

This module holds the *one* implementation of the statistic the
:class:`MeshTransformer` nondimensionalizes lengths by.  It exists because
the statistic is computed in two places that must never drift apart:

1. :meth:`MeshTransformer._intrinsic_reference_length`, the default gauge
   used whenever ``reference_length_key`` is ``None``;
2. dataset-side transforms that compute a per-sample gauge and feed it
   through the *explicit* ``reference_length_key`` override — the shape the
   DrivAerML cross-family repair takes (``ComputeIntrinsicReferenceLength``
   in the unified external-aero recipe).

Those two paths deliberately supply **different weights**, and the
difference is not a bug:

- the model weights by ``mesh.cell_areas``, the bare geometric measure it
  uses for *all* of its quadrature, so the gauge is consistent with the
  operator it normalizes;
- a dataset transform running after subsampling weights by
  ``cell_measures(mesh)`` — geometric measure times composed
  Horvitz--Thompson inclusion weights — because its job is to estimate the
  statistic of the *full-resolution* geometry from a subsample.

Sharing the reduction (and its float64 discipline and validation) while
letting each caller pass its own weights is the point of this module.  Do
not "unify" the two weight choices without a measurement: they answer
different questions.

.. note::
   Placement is deliberately in the *experimental* namespace.  The
   statistic is a pure mesh geometric quantity and would sit naturally in
   ``physicsnemo.mesh.calculus.measure`` beside ``cell_measures`` — but
   that module is released public API, and promoting a symbol there is a
   support commitment rather than a refactor.  The promotion is a
   standing, deliberately-deferred decision.
"""

import torch
from jaxtyping import Float

__all__ = ["measure_weighted_rms_radius"]


def measure_weighted_rms_radius(
    weights: Float[torch.Tensor, " n_cells"],
    centroids: Float[torch.Tensor, "n_cells n_dims"],
    dtype: torch.dtype | None = None,
    *,
    validate: bool = True,
) -> Float[torch.Tensor, ""]:
    r"""Measure-weighted RMS radius of *centroids* about their weighted centroid.

    .. math::

       \bar c = \frac{\sum_i w_i c_i}{\sum_i w_i},
       \qquad
       L = \sqrt{\frac{\sum_i w_i \lVert c_i - \bar c\rVert^2}{\sum_i w_i}}

    Degree-1 positive homogeneity in the geometry is what makes the model's
    scale equivariance unconditional: scaling the geometry by :math:`s`
    scales :math:`L` by exactly :math:`s`, because the measures scale by
    :math:`s^m` and cancel in both weighted means.  The statistic is also
    translation- and rotation-invariant, refinement-convergent, smooth in
    the boundary shape, and differentiable through the mesh geometry.

    The reduction is accumulated in float64 regardless of input dtype and
    cast back at the end.  Measured honestly, this is **defensive, not a bug
    fix**: on realistic boundary statistics (10k--50k cells with measures
    spanning ~2e-9 to ~7e-5, the DrivAerML vehicle's range) a float32
    accumulation already agrees with the float64 answer to ~1e-8--4e-8
    relative, because torch's ``sum`` reduces pairwise rather than naively.
    What the promotion buys is that the gauge is a function of the geometry
    alone and not of the dtype it arrived in — so the model's built-in gauge
    and a dataset-side transform computing the same statistic agree
    regardless of pipeline precision, and no future change to input dtype
    can quietly move every nondimensionalized length downstream.

    Parameters
    ----------
    weights
        Per-cell integration measure.  Callers choose which measure is
        appropriate; see the module docstring on why the model and the
        dataset transforms legitimately differ here.
    centroids
        Per-cell centroids, shape ``(n_cells, n_dims)``.
    dtype
        Result dtype.  Defaults to ``centroids.dtype``.
    validate
        Check that the result is finite and strictly positive.  Skipped
        automatically under :func:`torch.compile` tracing, since the check
        requires a host synchronization.

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) gauge, in the coordinate units of *centroids*.

    Raises
    ------
    ValueError
        If *validate* and the statistic is non-finite or non-positive.  It
        vanishes exactly when every cell centroid coincides with the
        weighted centroid, which is a degenerate boundary rather than a
        numerical accident.
    """
    if dtype is None:
        dtype = centroids.dtype

    weights64 = weights.double()
    centroids64 = centroids.double()
    total = weights64.sum()
    center = torch.einsum("n,nd->d", weights64, centroids64) / total
    radius_squared = (centroids64 - center).square().sum(dim=-1)
    length = torch.sqrt(torch.einsum("n,n->", weights64, radius_squared) / total)
    length = length.to(dtype)

    if validate and not torch.compiler.is_compiling():
        if not torch.isfinite(length).item() or length.item() <= 0.0:
            raise ValueError(
                "Intrinsic reference length (measure-weighted RMS boundary "
                "radius) must be finite and positive; it vanishes when every "
                "boundary cell centroid coincides with the boundary "
                "centroid.  Supply reference_length_key to override the "
                "intrinsic scale gauge for such degenerate geometries"
            )
    return length
