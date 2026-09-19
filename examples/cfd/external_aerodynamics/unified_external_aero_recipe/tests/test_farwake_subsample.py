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

"""Tests for SdfBiasedSubsampleInteriorPoints (far-wake oversampling arm, 2026-09-10)."""

import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain_transforms import SdfBiasedSubsampleInteriorPoints  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _domain(n=40000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) * 2 - 1
    # signed distance: 77% near the wall, 11% mid, 10% far (7.8% at 0.05-0.4, 2.2% beyond 0.4)
    u = torch.rand(n, generator=g)
    sdf = torch.where(
        u < 0.77,
        torch.rand(n, generator=g) * 0.01,
        torch.where(
            u < 0.88,
            0.01 + torch.rand(n, generator=g) * 0.04,
            torch.where(
                u < 0.978,
                0.05 + torch.rand(n, generator=g) * 0.35,
                0.4 + torch.rand(n, generator=g) * 0.6,
            ),
        ),
    )
    interior = Mesh(
        points=pts,
        point_data=TensorDict(
            {"sdf": sdf[:, None], "nut": torch.rand(n, 1, generator=g)}, batch_size=[n]
        ),
    )
    wall = Mesh(
        points=torch.rand(30, 3, generator=g),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": wall},
        global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0])},
    )


def test_expected_count_and_far_share():
    """Check the sampling count, far-field coverage, and inclusion correction."""
    dm = _domain()
    t = SdfBiasedSubsampleInteriorPoints(n_points_expected=10000)
    t.set_generator(torch.Generator().manual_seed(1))
    out = t.apply_to_domain(dm)
    n = out.interior.points.shape[0]
    assert abs(n - 10000) < 400, n  # Poisson count noise ~ sqrt(10000)
    pi = out.interior.point_data["inclusion_pi"].reshape(-1)
    assert torch.all(pi > 0) and torch.all(pi <= 1.0)
    sdf_in = dm.interior.point_data["sdf"].reshape(-1)
    sdf_out = out.interior.point_data["sdf"].reshape(-1)
    far_in = (sdf_in >= 0.05).float().mean().item()
    far_out = (sdf_out >= 0.05).float().mean().item()
    assert far_in < 0.12 and far_out > 0.25, (far_in, far_out)  # far band ~10% -> ~32%
    # boundaries and global data untouched
    assert out.boundaries["vehicle"].n_cells == 2
    assert torch.equal(out.global_data["U_inf_dir"], dm.global_data["U_inf_dir"])
    # the Horvitz-Thompson estimate of the far-band share from the kept sample is unbiased
    w = 1.0 / pi
    ht_far = (w * (sdf_out >= 0.05).float()).sum().item() / w.sum().item()
    assert abs(ht_far - far_in) < 0.02, (ht_far, far_in)


def test_reproducible_and_identity_cases():
    """Reuse seeded draws and preserve domains that need no thinning."""
    dm = _domain()
    t1 = SdfBiasedSubsampleInteriorPoints(n_points_expected=10000)
    t1.set_generator(torch.Generator().manual_seed(7))
    t2 = SdfBiasedSubsampleInteriorPoints(n_points_expected=10000)
    t2.set_generator(torch.Generator().manual_seed(7))
    a = t1.apply_to_domain(dm)
    b = t2.apply_to_domain(dm)
    assert torch.equal(a.interior.points, b.interior.points)
    # small interior: passthrough
    small = _domain(n=500)
    assert t1.apply_to_domain(small) is small
    # bare Mesh: identity
    assert t1(small.interior) is small.interior


def test_requires_sdf_and_validates_args():
    """Reject absent SDF fields and invalid sampling bands."""
    dm = _domain(n=20000)
    dm2 = DomainMesh(
        interior=Mesh(points=dm.interior.points),
        boundaries=dm.boundaries,
        global_data=dm.global_data,
    )
    with pytest.raises(KeyError):
        SdfBiasedSubsampleInteriorPoints(n_points_expected=1000).apply_to_domain(dm2)
    with pytest.raises(ValueError):
        SdfBiasedSubsampleInteriorPoints(n_points_expected=1000, band_edges=(0.4, 0.05))
    with pytest.raises(ValueError):
        SdfBiasedSubsampleInteriorPoints(
            n_points_expected=1000, band_weights=(1.0, 2.0)
        )


def test_explicit_point_measure_gets_the_inclusion_correction():
    """SDF-biased thinning preserves the represented point measure in expectation."""
    from physicsnemo.mesh.calculus.measure import point_measures, set_point_measures

    domain = _domain(n=20000)
    set_point_measures(domain.interior, torch.full((20000,), 2.0), dimension=3)
    transform = SdfBiasedSubsampleInteriorPoints(n_points_expected=5000)
    transform.set_generator(torch.Generator().manual_seed(1))
    sampled = transform.apply_to_domain(domain).interior
    pi = sampled.point_data["inclusion_pi"].flatten()
    torch.testing.assert_close(point_measures(sampled), 2.0 / pi)
    torch.testing.assert_close(
        point_measures(domain.interior), torch.full((20000,), 2.0)
    )
