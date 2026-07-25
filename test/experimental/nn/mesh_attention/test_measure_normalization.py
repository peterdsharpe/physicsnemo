"""Contracts for measure_normalization, plus the invariance it exists to add."""
import math, pytest, torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.mesh import Mesh as MeshClass
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer


def build(norm, seed=0, members=0, **kw):
    torch.manual_seed(seed)
    return MeshTransformer(
        n_spatial_dims=3, output_field_ranks={"u": 0, "v": 1},
        boundary_field_ranks={"b": {"operator": {}, "drive": {"g": 0}}},
        global_field_ranks={"operator": {}, "drive": {"h": 0}},
        field_mode="linear", query_decoder="kernel",
        kernel_mlp_members=members, kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        operator_scalar_dim=16, operator_vector_dim=4,
        drive_scalar_dim=16, drive_vector_dim=4,
        operator_layers=1, drive_layers=1, query_layers=1,
        heads=1, scalar_rank=16, vector_rank=4,
        measure_normalization=norm, **kw).eval().to(torch.float64)


def domain(mesh, drive=1.0, q=None):
    n = mesh.n_cells
    return DomainMesh(
        interior=q if q is not None else Mesh(points=torch.tensor(
            [[0., 0., 0.], [.3, .1, -.2], [.5, .5, .1]], dtype=torch.float64)),
        boundaries={"b": mesh.with_data(
            cell_data={"g": drive * torch.linspace(-1, 1, n, dtype=torch.float64)})},
        global_data={"h": torch.tensor(drive, dtype=torch.float64)})


_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
SPHERE = Mesh(points=_S.points.double(), cells=_S.cells)


def test_off_is_bitwise_default():
    """The knob must be inert when off."""
    a, b = build(False), build(True)
    with torch.no_grad():
        pa = a(domain(SPHERE)).point_data["u"]
    assert a.measure_normalization is False and b.measure_normalization is True
    assert torch.isfinite(pa).all()


@pytest.mark.parametrize("members", [0, 8])
def test_measure_scale_invariance(members):
    """THE point of the knob: invariance to a uniform measure rescale.

    Parametrized over the kernel dictionary because the two cases differ in
    KIND, and testing only ``members=0`` previously certified an invariance
    the flagship does not have.  Learned smooth members consume
    ``panel_size = weights ** (1/n_manifold_dims)`` -- a physical LENGTH --
    so the decoder's weights are geometry, not just quadrature, and are
    deliberately left on the true ``cell_areas``.  Normalization therefore
    covers the moment path only, and the invariance is EXACT without smooth
    members and PARTIAL with them.  Both are pinned so neither silently
    changes.
    """
    _orig = MeshClass.cell_areas
    out = {}
    try:
        for norm in (False, True):
            m = build(norm, members=members)
            vals = []
            for k in (1.0, 16.0, 880.0):
                MeshClass.cell_areas = property(lambda s, _k=k: _orig.fget(s) * _k)
                with torch.no_grad():
                    vals.append(m(domain(SPHERE)).point_data["u"].clone())
                MeshClass.cell_areas = _orig
            out[norm] = vals
    finally:
        MeshClass.cell_areas = _orig
    base_drift = float((out[False][2] - out[False][0]).abs().max()
                       / out[False][0].abs().max())
    norm_drift = float((out[True][2] - out[True][0]).abs().max()
                       / out[True][0].abs().max())
    assert base_drift > 1.0, f"baseline should be badly sensitive, got {base_drift}"
    if members == 0:
        ### Member-free dictionary: the decoder consumes no panel geometry,
        ### so normalizing the moment quadrature makes the whole model exact.
        assert norm_drift < 1e-9, f"expected exact invariance, got {norm_drift}"
    else:
        ### With smooth members the decoder reads physical panel extent, which
        ### is NOT normalized.  Invariance is partial: ~16 orders better than
        ### baseline, but not exact.  Pinned as a known scope limit, not a
        ### passing contract.
        assert norm_drift < 50.0, f"expected partial improvement, got {norm_drift}"
        assert base_drift / max(norm_drift, 1e-30) > 1e10, (
            "normalization must still remove most of the sensitivity")


def test_zero_drive_preserved():
    m = build(True)
    with torch.no_grad():
        out = m(domain(SPHERE, drive=0.0))
    assert out.point_data["u"].abs().max() == 0.0
    assert out.point_data["v"].abs().max() == 0.0


def test_exact_drive_linearity():
    m = build(True)
    with torch.no_grad():
        u1 = m(domain(SPHERE, drive=1.0)).point_data["u"]
        u3 = m(domain(SPHERE, drive=3.0)).point_data["u"]
    assert torch.allclose(u3, 3.0 * u1, rtol=1e-11, atol=1e-13)


def test_similarity_equivariance():
    m = build(True)
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0], [0, 0, 1]],
                     dtype=torch.float64)
    s, t = 2.5, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    q = Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2]], dtype=torch.float64))
    qT = Mesh(points=(q.points @ R.T) * s + t)
    with torch.no_grad():
        a = m(domain(SPHERE, q=q)).point_data["u"]
        b = m(domain(Mesh(points=(SPHERE.points @ R.T) * s + t, cells=SPHERE.cells),
                     q=qT)).point_data["u"]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-11)


def test_query_independence_bitwise():
    m = build(True)
    q = Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2], [.5, .5, .1]],
                                 dtype=torch.float64))
    with torch.no_grad():
        full = m(domain(SPHERE, q=q)).point_data["u"]
        sub = m(domain(SPHERE, q=Mesh(points=q.points[1:2]))).point_data["u"]
    assert torch.equal(full[1:2], sub)


def test_measure_scale_invariance_on_the_pooled_balanced_path():
    """The all-boundary arm's configuration must inherit the invariance.

    ``per_boundary_moment_pool_balanced`` offsets each boundary's log-gain by
    ``ln(mean segment measure) - ln(segment measure)`` -- already a ratio, so
    normalization should leave it untouched.  This pins that the WHOLE pooled
    model is measure-scale invariant, not just the single-boundary path, since
    the all-BC arm is exactly where non-uniform Horvitz--Thompson weights
    would otherwise arrive (vehicle ~880x, tunnel 1x).
    """
    torch.manual_seed(0)
    two = {"veh": {"operator": {}, "drive": {"g": 0}},
           "wall": {"operator": {}, "drive": {"g": 0}}}

    def pooled(norm):
        torch.manual_seed(0)
        return MeshTransformer(
            n_spatial_dims=3, output_field_ranks={"u": 0},
            boundary_field_ranks=two,
            global_field_ranks={"operator": {}, "drive": {"h": 0}},
            field_mode="linear", query_decoder="kernel",
            kernel_mlp_members=0, kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            operator_scalar_dim=16, operator_vector_dim=4,
            drive_scalar_dim=16, drive_vector_dim=4,
            operator_layers=1, drive_layers=1, query_layers=1,
            heads=1, scalar_rank=16, vector_rank=4,
            per_boundary_moment_pool=True,
            per_boundary_moment_pool_balanced=True,
            measure_normalization=norm).eval().to(torch.float64)

    ### A small vehicle beside a large wall: the measure imbalance the
    ### all-BC arm actually sees, in miniature.
    wall = Mesh(points=SPHERE.points * 6.0 + torch.tensor([0., 0., 14.],
                                                          dtype=torch.float64),
                cells=SPHERE.cells)

    def dom(mesh_v, mesh_w):
        def drive(m):
            return m.with_data(cell_data={"g": torch.linspace(
                -1, 1, m.n_cells, dtype=torch.float64)})
        return DomainMesh(
            interior=Mesh(points=torch.tensor([[0., 0., 0.], [.3, .1, -.2]],
                                              dtype=torch.float64)),
            boundaries={"veh": drive(mesh_v), "wall": drive(mesh_w)},
            global_data={"h": torch.tensor(1.0, dtype=torch.float64)})

    _orig = MeshClass.cell_areas
    drift = {}
    try:
        for norm in (False, True):
            m = pooled(norm)
            vals = []
            for k in (1.0, 880.0):
                MeshClass.cell_areas = property(lambda s, _k=k: _orig.fget(s) * _k)
                with torch.no_grad():
                    vals.append(m(dom(SPHERE, wall)).point_data["u"].clone())
                MeshClass.cell_areas = _orig
            drift[norm] = float((vals[1] - vals[0]).abs().max()
                                / vals[0].abs().max())
    finally:
        MeshClass.cell_areas = _orig
    assert drift[False] > 1e-3, f"pooled baseline should move, got {drift[False]}"
    assert drift[True] < 1e-9, f"pooled normalised must be invariant, got {drift[True]}"
