"""Contracts for measure_normalization, plus the invariance it exists to add."""

import math

import pytest
import torch

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    exact_single_layer_member,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import (
    MEASURE_WEIGHTS_KEY,
    cell_measure_weights,
)
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral


def build(norm=None, seed=0, members=0, **kw):
    torch.manual_seed(seed)
    if norm is not None:
        kw["measure_normalization"] = norm
    return (
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"u": 0, "v": 1},
            boundary_field_ranks={"b": {"operator": {}, "drive": {"g": 0}}},
            global_field_ranks={"operator": {}, "drive": {"h": 0}},
            field_mode="linear",
            query_decoder="kernel",
            kernel_mlp_members=members,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            operator_scalar_dim=16,
            operator_vector_dim=4,
            drive_scalar_dim=16,
            drive_vector_dim=4,
            operator_layers=1,
            drive_layers=1,
            query_layers=1,
            heads=1,
            scalar_rank=16,
            vector_rank=4,
            **kw,
        )
        .eval()
        .to(torch.float64)
    )


def domain(mesh, drive=1.0, q=None):
    n = mesh.n_cells
    cell_data = {
        "g": drive * torch.linspace(-1, 1, n, dtype=torch.float64),
    }
    if MEASURE_WEIGHTS_KEY in mesh.cell_data:
        cell_data[MEASURE_WEIGHTS_KEY] = mesh.cell_data[MEASURE_WEIGHTS_KEY]
    return DomainMesh(
        interior=q
        if q is not None
        else Mesh(
            points=torch.tensor(
                [[0.0, 0.0, 0.0], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]],
                dtype=torch.float64,
            )
        ),
        boundaries={"b": mesh.with_data(cell_data=cell_data)},
        global_data={"h": torch.tensor(drive, dtype=torch.float64)},
    )


_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
SPHERE = Mesh(points=_S.points.double(), cells=_S.cells)


def with_measure_weights(mesh, weights):
    return mesh.with_data(cell_data={MEASURE_WEIGHTS_KEY: weights})


def test_off_is_bitwise_default():
    """Explicitly disabling the knob retains the historical default path."""
    default, disabled = build(), build(False)
    with torch.no_grad():
        expected = default(domain(SPHERE)).point_data
        actual = disabled(domain(SPHERE)).point_data
        encoded = disabled.encode(domain(SPHERE))
    assert default.measure_normalization is False
    assert disabled.measure_normalization is False
    assert MEASURE_WEIGHTS_KEY not in encoded.source_mesh.cell_data
    assert torch.equal(actual["u"], expected["u"])
    assert torch.equal(actual["v"], expected["v"])


def test_public_nonuniform_weights_survive_encode_and_change_output():
    """The public model path must consume, not strip, effective measure."""
    weights = torch.linspace(0.2, 5.0, SPHERE.n_cells, dtype=torch.float64)
    weighted = with_measure_weights(SPHERE, weights)
    model = build(False, members=8)
    with torch.no_grad():
        encoded = model.encode(domain(weighted))
        plain_output = model(domain(SPHERE)).point_data["u"]
        weighted_output = model(domain(weighted)).point_data["u"]
    torch.testing.assert_close(
        cell_measure_weights(encoded.source_mesh), weights, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        encoded.kernel_cache.panel_areas,
        encoded.source_mesh.cell_areas,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        encoded.kernel_cache.quadrature_measures,
        encoded.source_mesh.cell_areas * weights,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(weighted_output, plain_output)
    assert (weighted_output - plain_output).abs().max() > 0.0


def test_public_measure_weights_are_cast_to_geometry_dtype():
    """Dimensionless metadata may arrive wider than the model geometry."""
    mesh = Mesh(points=SPHERE.points.float(), cells=SPHERE.cells)
    weights = torch.linspace(0.2, 5.0, mesh.n_cells, dtype=torch.float64)
    boundary = mesh.with_data(
        cell_data={
            "g": torch.linspace(-1.0, 1.0, mesh.n_cells),
            MEASURE_WEIGHTS_KEY: weights,
        }
    )
    sample = DomainMesh(
        interior=Mesh(points=torch.tensor([[0.1, 0.2, 0.3]])),
        boundaries={"b": boundary},
        global_data={"h": torch.tensor(1.0)},
    )
    model = build(False).float()

    with torch.no_grad():
        encoded = model.encode(sample)

    retained = cell_measure_weights(encoded.source_mesh)
    assert retained.dtype == mesh.points.dtype
    torch.testing.assert_close(retained, weights.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "bad_value",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_public_measure_weights_must_be_finite_and_positive(bad_value):
    weights = torch.ones(SPHERE.n_cells, dtype=torch.float64)
    weights[0] = bad_value
    model = build(False)
    with pytest.raises(ValueError, match="effective quadrature measure"):
        model.encode(domain(with_measure_weights(SPHERE, weights)))


def test_cyclic_ht_factor_is_exact_for_a_linear_panel_total():
    """Enumerating every production start recovers the full linear total."""
    mesh_data = sphere_icosahedral.load(radius=1.0, subdivisions=0)
    mesh = Mesh(points=mesh_data.points.double(), cells=mesh_data.cells)
    query = torch.tensor([[3.0, 0.2, -0.4]], dtype=torch.float64)
    per_panel = exact_single_layer_member(query, mesh.points[mesh.cells]).squeeze(0)
    full_total = per_panel.sum()
    n_cells = mesh.n_cells
    n_kept = 7

    bare_totals = []
    ht_totals = []
    for start in range(n_cells):
        indices = _cyclic_block_indices(n_cells, n_kept, start)
        bare = per_panel[indices].sum()
        bare_totals.append(bare)
        ht_totals.append(bare * (n_cells / n_kept))

    bare_mean = torch.stack(bare_totals).mean()
    ht_mean = torch.stack(ht_totals).mean()
    torch.testing.assert_close(ht_mean, full_total, rtol=2.0e-15, atol=2.0e-15)
    torch.testing.assert_close(
        bare_mean / full_total,
        torch.tensor(n_kept / n_cells, dtype=torch.float64),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("members", [0, 8])
def test_measure_scale_invariance(members):
    """Normalization removes nuisance scale, but retains measure shape."""
    shape = torch.linspace(0.2, 5.0, SPHERE.n_cells, dtype=torch.float64)
    out = {}
    for norm in (False, True):
        model = build(norm, members=members)
        vals = []
        for scale in (1.0, 16.0, 880.0):
            weighted = with_measure_weights(SPHERE, shape * scale)
            with torch.no_grad():
                vals.append(model(domain(weighted)).point_data["u"].clone())
        out[norm] = vals
    base_drift = float(
        (out[False][2] - out[False][0]).abs().max() / out[False][0].abs().max()
    )
    norm_drift = float(
        (out[True][2] - out[True][0]).abs().max() / out[True][0].abs().max()
    )
    assert base_drift > 1.0, f"baseline should be badly sensitive, got {base_drift}"
    assert norm_drift < 1e-9, f"expected scale invariance, got {norm_drift}"

    uniform = torch.ones(SPHERE.n_cells, dtype=torch.float64)
    with torch.no_grad():
        uniform_output = build(True, members=members)(
            domain(with_measure_weights(SPHERE, uniform))
        ).point_data["u"]
    assert not torch.equal(out[True][0], uniform_output)


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
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float64,
    )
    s, t = 2.5, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    q = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.1, -0.2]], dtype=torch.float64)
    )
    qT = Mesh(points=(q.points @ R.T) * s + t)
    with torch.no_grad():
        a = m(domain(SPHERE, q=q)).point_data["u"]
        b = m(
            domain(Mesh(points=(SPHERE.points @ R.T) * s + t, cells=SPHERE.cells), q=qT)
        ).point_data["u"]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-11)


def test_query_independence_bitwise():
    m = build(True)
    q = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]], dtype=torch.float64
        )
    )
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
    two = {
        "veh": {"operator": {}, "drive": {"g": 0}},
        "wall": {"operator": {}, "drive": {"g": 0}},
    }

    def pooled(norm):
        torch.manual_seed(0)
        return (
            MeshTransformer(
                n_spatial_dims=3,
                output_field_ranks={"u": 0},
                boundary_field_ranks=two,
                global_field_ranks={"operator": {}, "drive": {"h": 0}},
                field_mode="linear",
                query_decoder="kernel",
                kernel_mlp_members=0,
                kernel_include_polynomial_members=False,
                kernel_include_single_layer_member=True,
                operator_scalar_dim=16,
                operator_vector_dim=4,
                drive_scalar_dim=16,
                drive_vector_dim=4,
                operator_layers=1,
                drive_layers=1,
                query_layers=1,
                heads=1,
                scalar_rank=16,
                vector_rank=4,
                per_boundary_moment_pool=True,
                per_boundary_moment_pool_balanced=True,
                measure_normalization=norm,
            )
            .eval()
            .to(torch.float64)
        )

    ### A small vehicle beside a large wall: the measure imbalance the
    ### all-BC arm actually sees, in miniature.
    wall = Mesh(
        points=SPHERE.points * 6.0
        + torch.tensor([0.0, 0.0, 14.0], dtype=torch.float64),
        cells=SPHERE.cells,
    )

    def dom(mesh_v, mesh_w):
        def drive(m):
            data = {
                "g": torch.linspace(-1, 1, m.n_cells, dtype=torch.float64),
            }
            if MEASURE_WEIGHTS_KEY in m.cell_data:
                data[MEASURE_WEIGHTS_KEY] = m.cell_data[MEASURE_WEIGHTS_KEY]
            return m.with_data(cell_data=data)

        return DomainMesh(
            interior=Mesh(
                points=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.3, 0.1, -0.2]], dtype=torch.float64
                )
            ),
            boundaries={"veh": drive(mesh_v), "wall": drive(mesh_w)},
            global_data={"h": torch.tensor(1.0, dtype=torch.float64)},
        )

    drift = {}
    vehicle_shape = torch.linspace(0.5, 2.0, SPHERE.n_cells, dtype=torch.float64)
    wall_shape = torch.linspace(2.0, 0.5, wall.n_cells, dtype=torch.float64)
    for norm in (False, True):
        m = pooled(norm)
        vals = []
        for scale in (1.0, 880.0):
            vehicle = with_measure_weights(SPHERE, vehicle_shape * scale)
            weighted_wall = with_measure_weights(wall, wall_shape * scale)
            with torch.no_grad():
                vals.append(m(dom(vehicle, weighted_wall)).point_data["u"].clone())
        drift[norm] = float((vals[1] - vals[0]).abs().max() / vals[0].abs().max())
    assert drift[False] > 1e-3, f"pooled baseline should move, got {drift[False]}"
    assert drift[True] < 1e-9, f"pooled normalised must be invariant, got {drift[True]}"

    # Mesh.merge requires common cell-data keys. An explicitly weighted
    # boundary must compose with an unweighted boundary as unit factors,
    # rather than either dropping the public key or rejecting the domain.
    mixed_model = pooled(False)
    mixed_domain = dom(with_measure_weights(SPHERE, vehicle_shape), wall)
    with torch.no_grad():
        mixed_weights = cell_measure_weights(
            mixed_model.encode(mixed_domain).source_mesh
        )
    torch.testing.assert_close(
        mixed_weights[: SPHERE.n_cells], vehicle_shape, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        mixed_weights[SPHERE.n_cells :],
        torch.ones(wall.n_cells, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
