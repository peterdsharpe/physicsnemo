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

"""Tests for the intrinsic reference-length gauge transform."""

import intrinsic_gauge
import pytest
import torch
from intrinsic_gauge import (
    ComputeIntrinsicReferenceLength,
    measure_weighted_rms_radius,
)

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import compose_measure_weights


def _plate_mesh(
    a: float = 2.0,
    b: float = 1.0,
    n: int = 50,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """Regular right-triangle mesh of the plate [0,a]x[0,b] in 3D."""
    xs = torch.linspace(0.0, a, n + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, b, n + 1, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack(
        (gx.reshape(-1), gy.reshape(-1), torch.zeros_like(gx).reshape(-1)), dim=-1
    ) + torch.tensor(offset, dtype=torch.float64)
    idx = torch.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    v00 = idx[:-1, :-1].reshape(-1)
    v10 = idx[1:, :-1].reshape(-1)
    v01 = idx[:-1, 1:].reshape(-1)
    v11 = idx[1:, 1:].reshape(-1)
    cells = torch.cat(
        (
            torch.stack((v00, v10, v11), dim=-1),
            torch.stack((v00, v11, v01), dim=-1),
        ),
        dim=0,
    )
    return Mesh(points=points, cells=cells)


def test_matches_model_intrinsic_gauge():
    """The transform and the model's built-in gauge are ONE implementation.

    The cross-family repair works by computing a per-sample gauge dataset-side
    and feeding it through the model's *explicit* ``reference_length_key``
    override.  That means the same statistic is reachable by two routes, and a
    silent divergence between them would corrupt the comparison the whole
    transfer experiment rests on.  Both now call
    ``mesh_attention.measure_weighted_rms_radius``; this pins it.

    On a mesh with no recorded measure weights the two weightings
    (``cell_measures`` vs the model's bare ``cell_areas``) coincide bitwise,
    so equality here is exact rather than approximate.
    """
    from physicsnemo.experimental.nn.mesh_attention import (
        measure_weighted_rms_radius as model_gauge,
    )

    mesh = _plate_mesh(a=2.0, b=1.0, n=20, offset=(3.0, -1.0, 0.5))

    transform_value = measure_weighted_rms_radius(mesh)
    model_value = model_gauge(mesh.cell_areas, mesh.cell_centroids)
    assert torch.equal(transform_value, model_value)

    ### And through the transform itself, with the calibration constant.
    scale_constant = 26.476592786355283
    written = ComputeIntrinsicReferenceLength(scale_constant=scale_constant)(mesh)
    assert torch.allclose(
        written.global_data["reference_length"],
        scale_constant * model_value,
        rtol=0.0,
        atol=0.0,
    )


def test_gauge_is_dtype_independent_on_realistic_measures():
    """The gauge must be a function of the geometry, not of its dtype.

    The DrivAerML vehicle carries cell areas from ~2e-9 to ~7e-5 in a single
    sample, and the gauge scales every length the model sees.  The shared
    reduction promotes to float64 internally so a float32 pipeline and a
    float64 one cannot disagree about the operating point.

    Measured scope, stated so nobody over-reads this test: float32 input is
    NOT badly wrong without the promotion (torch sums pairwise, so it lands
    within ~1e-8..4e-8 relative on these statistics).  The promotion removes
    dtype as a variable rather than repairing a large error -- which is why
    the calibration constant fitted before it remains valid.
    """
    from physicsnemo.experimental.nn.mesh_attention import (
        measure_weighted_rms_radius as model_gauge,
    )

    generator = torch.Generator().manual_seed(11)
    n = 20_000
    centroids64 = torch.rand(n, 3, dtype=torch.float64, generator=generator) * 5.0
    ### Log-uniform measures spanning the real vehicle's decades.
    exponents = torch.rand(n, dtype=torch.float64, generator=generator)
    weights64 = 10.0 ** (-9.0 + 5.0 * exponents)

    reference = model_gauge(weights64, centroids64)
    from_float32 = model_gauge(
        weights64.float(), centroids64.float(), torch.float64
    )
    assert from_float32 == pytest.approx(float(reference), rel=1e-6)


def test_plate_rms_matches_closed_form():
    # Uniform plate [0,a]x[0,b]: r_RMS = sqrt((a^2 + b^2)/12).
    a, b = 2.0, 1.0
    mesh = _plate_mesh(a=a, b=b, n=50)
    expected = ((a**2 + b**2) / 12.0) ** 0.5
    got = float(measure_weighted_rms_radius(mesh))
    # Cell-centroid (midpoint) quadrature of a quadratic on a regular grid:
    # small, refinement-vanishing bias.
    assert got == pytest.approx(expected, rel=1e-2)


def test_two_cell_exact_value():
    # Two right triangles with hand-computed centroids, areas, and RMS.
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(points=points, cells=cells)
    # Triangle A: verts 0,1,2 -> centroid (1/3, 1/3), area 1/2.
    # Triangle B: verts 1,3,2 -> centroid (4/3, 1/3), area | (2,0)x(-1,1) |/2 = 1.
    c = torch.tensor(
        [[1.0 / 3.0, 1.0 / 3.0], [4.0 / 3.0, 1.0 / 3.0]], dtype=torch.float64
    )
    w = torch.tensor([0.5, 1.0], dtype=torch.float64)
    center = (w[:, None] * c).sum(0) / w.sum()
    expected = float(
        (((w * ((c - center) ** 2).sum(-1)).sum() / w.sum()) ** 0.5)
    )
    got = float(measure_weighted_rms_radius(mesh))
    assert got == pytest.approx(expected, rel=1e-12)


def test_translation_invariance_and_similarity_covariance():
    mesh = _plate_mesh(n=20)
    base = measure_weighted_rms_radius(mesh)

    shifted = _plate_mesh(n=20, offset=(5.0, -3.0, 11.0))
    assert float(measure_weighted_rms_radius(shifted)) == pytest.approx(
        float(base), rel=1e-12
    )

    scale = 3.7
    scaled = Mesh(points=mesh.points * scale, cells=mesh.cells)
    assert float(measure_weighted_rms_radius(scaled)) == pytest.approx(
        scale * float(base), rel=1e-12
    )


def test_horvitz_thompson_weight_robustness():
    # Keeping every other cell with composed inverse-inclusion weights is a
    # deterministic stand-in for SubsampleMesh: the weighted statistic must
    # track the full-mesh value closely (regular grid -> tiny residual).
    mesh = _plate_mesh(n=30)
    full = float(measure_weighted_rms_radius(mesh))
    keep = torch.arange(0, mesh.n_cells, 2)
    sub = mesh.select_cells(keep) if hasattr(mesh, "select_cells") else None
    if sub is None:
        pytest.skip("Mesh.select_cells unavailable; covered by pipeline tests")
    compose_measure_weights(sub, 2.0)
    got = float(measure_weighted_rms_radius(sub))
    assert got == pytest.approx(full, rel=5e-3)


def test_transform_writes_scaled_gauge_and_is_deterministic():
    mesh = _plate_mesh(n=10)
    transform = ComputeIntrinsicReferenceLength(scale_constant=11.0)
    out1 = transform(mesh)
    out2 = transform(mesh)
    r = float(measure_weighted_rms_radius(mesh))
    assert float(out1.global_data["reference_length"]) == pytest.approx(
        11.0 * r, rel=1e-12
    )
    assert float(out1.global_data["reference_length"]) == float(
        out2.global_data["reference_length"]
    )
    # Input mesh untouched; other global_data preserved.
    assert "reference_length" not in mesh.global_data.keys()


def test_validation():
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError):
            ComputeIntrinsicReferenceLength(scale_constant=bad)
    with pytest.raises(ValueError):
        ComputeIntrinsicReferenceLength(scale_constant=8.0, field_name="")


def test_registered_in_datapipe_registry():
    from physicsnemo.datapipes.registry import _resolve_component

    target = _resolve_component("ComputeIntrinsicReferenceLength")
    assert target == "intrinsic_gauge.ComputeIntrinsicReferenceLength"
    assert intrinsic_gauge.ComputeIntrinsicReferenceLength is not None
