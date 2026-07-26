# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for ``field_mode="homogeneous"`` (declared drive degree one).

Parametrized over ``kernel_mlp_members`` from the start: member-free and
member-carrying dictionaries exercise different decoder paths, and a
member-blind contract test has already certified a property this core did not
have (see the measure-normalization entry in the lab notebook).
"""
import math

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
SPHERE = Mesh(points=_S.points.double(), cells=_S.cells)
MEMBERS = [0, 8]


def build(mode, members, seed=0, measure_normalization=False):
    torch.manual_seed(seed)
    return MeshTransformer(
        n_spatial_dims=3, output_field_ranks={"u": 0, "w": 1},
        boundary_field_ranks={"b": {"operator": {}, "drive": {"g": 0}}},
        global_field_ranks={"operator": {}, "drive": {"h": 0}},
        field_mode=mode, query_decoder="kernel",
        measure_normalization=measure_normalization,
        kernel_mlp_members=members, kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        operator_scalar_dim=16, operator_vector_dim=4,
        drive_scalar_dim=16, drive_vector_dim=4,
        operator_layers=1, drive_layers=1, query_layers=1,
        heads=1, scalar_rank=16, vector_rank=4,
    ).eval().to(torch.float64)


def domain(drive=1.0, queries=None):
    n = SPHERE.n_cells
    q = queries if queries is not None else Mesh(
        points=torch.tensor([[0., 0., 0.], [.3, .1, -.2], [.5, .5, .1]],
                            dtype=torch.float64))
    return DomainMesh(
        interior=q,
        boundaries={"b": SPHERE.with_data(cell_data={
            "g": drive * torch.linspace(-1, 1, n, dtype=torch.float64)})},
        global_data={"h": torch.tensor(drive, dtype=torch.float64)})


@pytest.mark.parametrize("members", MEMBERS)
def test_zero_drive_is_exactly_zero(members):
    """Zero preservation must be EXACT despite the read-in's biased MLP."""
    m = build("homogeneous", members)
    with torch.no_grad():
        out = m(domain(drive=0.0))
    assert out.point_data["u"].abs().max().item() == 0.0
    assert out.point_data["w"].abs().max().item() == 0.0


@pytest.mark.parametrize("members", MEMBERS)
def test_degree_is_exactly_one(members):
    """THE declared law: u(k*g) == k*u(g) to roundoff, for any k > 0."""
    m = build("homogeneous", members)
    with torch.no_grad():
        base = m(domain(drive=1.0)).point_data["u"]
        for k in (0.25, 3.0, 50.0):
            scaled = m(domain(drive=k)).point_data["u"]
            rel = float((scaled - k * base).abs().max() / (k * base).abs().max())
            assert rel < 1e-11, f"degree-1 violated at k={k}: {rel}"


@pytest.mark.parametrize("members", MEMBERS)
def test_amplification_is_bounded_unlike_the_nonlinear_mode(members):
    """The point of the mode: a k-fold drive shift gives a k-fold output shift.

    The unbounded mode is measured alongside so the comparison is in the test
    rather than only in the notebook.
    """
    k = 20.0
    ratios = {}
    for mode in ("homogeneous", "zero_preserving_nonlinear"):
        m = build(mode, members)
        with torch.no_grad():
            a = float(m(domain(drive=1.0)).point_data["u"].abs().max())
            b = float(m(domain(drive=k)).point_data["u"].abs().max())
        ratios[mode] = b / a
    assert ratios["homogeneous"] == pytest.approx(k, rel=1e-9)
    assert ratios["zero_preserving_nonlinear"] > 10.0 * k


@pytest.mark.parametrize("members", MEMBERS)
def test_not_secretly_linear(members):
    """Degree one must not collapse to additivity, or the mode buys nothing.

    The read-in is zero-initialized on purpose -- like ``QuadraticFieldReadIn``'s
    ``layer_scale`` default, the mode starts indistinguishable from the
    drive-linear machinery and the nonlinearity is *learned*.  So additivity at
    initialization is the design, and the property under test is the
    CAPABILITY: once the read-in carries any nonzero weight, the map must stop
    being additive while remaining exactly degree one.
    """
    m = build("homogeneous", members)
    with torch.no_grad():
        torch.manual_seed(3)
        final = m.quadratic_read_in.mlp[-1]
        final.weight.normal_(0.0, 0.5)
        final.bias.normal_(0.0, 0.5)
    n = SPHERE.n_cells
    ga = torch.linspace(-1, 1, n, dtype=torch.float64)
    gb = torch.cos(4 * torch.linspace(0, 3.14, n, dtype=torch.float64))

    def run(vec, h):
        dom = DomainMesh(
            interior=Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2]],
                                              dtype=torch.float64)),
            boundaries={"b": SPHERE.with_data(cell_data={"g": vec})},
            global_data={"h": torch.tensor(h, dtype=torch.float64)})
        with torch.no_grad():
            return m(dom).point_data["u"]

    lhs = run(ga + gb, 2.0)
    rhs = run(ga, 1.0) + run(gb, 1.0)
    gap = float((lhs - rhs).abs().max() / lhs.abs().max())
    assert gap > 1e-3, f"map is additive (gap {gap}); it is merely linear"

    ### ...and the declared degree still holds exactly with those weights.
    base = run(ga, 1.0)
    scaled = run(4.0 * ga, 4.0)
    rel = float((scaled - 4.0 * base).abs().max() / (4.0 * base).abs().max())
    assert rel < 1e-11, f"degree-1 broken once the nonlinearity is active: {rel}"


@pytest.mark.parametrize("members", MEMBERS)
def test_similarity_equivariance(members):
    m = build("homogeneous", members)
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0], [0, 0, 1]],
                     dtype=torch.float64)
    s, t = 2.5, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    q = Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2]], dtype=torch.float64))
    n = SPHERE.n_cells
    g = torch.linspace(-1, 1, n, dtype=torch.float64)
    rot = Mesh(points=(SPHERE.points @ R.T) * s + t, cells=SPHERE.cells)
    with torch.no_grad():
        a = m(DomainMesh(interior=q,
                         boundaries={"b": SPHERE.with_data(cell_data={"g": g})},
                         global_data={"h": torch.tensor(1.0, dtype=torch.float64)})
              ).point_data["u"]
        b = m(DomainMesh(interior=Mesh(points=(q.points @ R.T) * s + t),
                         boundaries={"b": rot.with_data(cell_data={"g": g})},
                         global_data={"h": torch.tensor(1.0, dtype=torch.float64)})
              ).point_data["u"]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("members", MEMBERS)
def test_composes_with_measure_normalization(members):
    """The two declared fixes must hold *simultaneously*.

    ``measure_normalization`` rescales the boundary quadrature weights to close
    the measure-dimension hole in units invariance; ``field_mode="homogeneous"``
    pins the drive degree at one.  They touch the same forward pass, so the
    combined configuration is the one that has to be certified -- shipping two
    separately-tested flags and assuming the conjunction works is precisely the
    member-blind mistake the module header warns about.
    """
    m = build("homogeneous", members, measure_normalization=True)
    with torch.no_grad():
        assert m(domain(drive=0.0)).point_data["u"].abs().max().item() == 0.0
        base = m(domain(drive=1.0)).point_data["u"]
        for k in (0.25, 3.0, 50.0):
            scaled = m(domain(drive=k)).point_data["u"]
            rel = float((scaled - k * base).abs().max() / (k * base).abs().max())
            assert rel < 1e-11, f"degree-1 lost under normalization at k={k}: {rel}"


@pytest.mark.parametrize("members", MEMBERS)
def test_query_independence_bitwise(members):
    m = build("homogeneous", members)
    q = Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2], [.5, .5, .1]],
                                 dtype=torch.float64))
    with torch.no_grad():
        full = m(domain(queries=q)).point_data["u"]
        sub = m(domain(queries=Mesh(points=q.points[1:2]))).point_data["u"]
    assert torch.equal(full[1:2], sub)
