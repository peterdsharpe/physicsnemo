"""Does the intrinsic gauge ADD or REMOVE cross-sample input-scale variance?

H-GC framed the +/-3.5% per-sample gauge spread as ADDED variance. But the
gauge spread exists precisely BECAUSE the geometries differ in size -- so it
may be cancelling the geometry's own variation rather than adding to it.
Measure both, on a population of geometries with DrivAerML's measured size
statistics (r_RMS = 0.30215 +/- 0.00890, CV = 2.95%).
"""
import math, statistics, torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.experimental.nn.mesh_attention import (
    MeshTransformer, measure_weighted_rms_radius)

exec(open("/tmp/hgc_gate.py").read().split("# Real DrivAerML vehicle")[0].split('"""', 2)[2])

C = 26.476592786355283
base = scattered(3000, 2.3e-9, 6.6e-5, (5.0, 2.0, 1.6), seed=7)

model = MeshTransformer(
    n_spatial_dims=3, output_field_ranks={"pressure": 0, "wss": 1},
    boundary_field_ranks={"vehicle": {"operator": {}, "drive": {}}},
    global_field_ranks={"operator": {}, "drive": {"U_inf_dir": 1}},
    reference_length_key="reference_length",
    field_mode="zero_preserving_nonlinear", query_decoder="kernel",
    trace_of="vehicle", kernel_mlp_members=8,
    kernel_include_polynomial_members=False,
    kernel_include_single_layer_member=True,
    kernel_monopole_free_single_layer=False,
    kernel_checkpoint_query_chunks=True,
    operator_scalar_dim=64, operator_vector_dim=16,
    drive_scalar_dim=96, drive_vector_dim=24,
    operator_layers=2, drive_layers=1, query_layers=1,
    heads=1, scalar_rank=96, vector_rank=32,
    query_chunk_size=65536, attention_chunk_size=65536,
).eval()

### A population of vehicles whose SIZES match DrivAerML's measured spread.
g = torch.Generator().manual_seed(3)
size_factors = 1.0 + 0.0295 * torch.randn(12, generator=g)   # CV = 2.95%

def state_norm(mesh, L):
    dom = DomainMesh(
        interior=Mesh(points=mesh.cell_centroids.clone()),
        boundaries={"vehicle": mesh},
        global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
                     "reference_length": torch.tensor(float(L))})
    with torch.no_grad():
        e = model.encode(dom)
    return math.sqrt(sum(float(t.square().sum())
                         for t in (e.drive_state.scalars, e.drive_state.vectors)))

fixed, intrinsic, radii = [], [], []
for f in size_factors:
    m = Mesh(points=base.points * float(f), cells=base.cells)
    r = float(measure_weighted_rms_radius(m.cell_areas, m.cell_centroids))
    radii.append(r)
    fixed.append(state_norm(m, 8.0))              # the historical fixed gauge
    intrinsic.append(state_norm(m, C * r))        # the intrinsic gauge

def cv(xs):
    return 100.0 * statistics.stdev(xs) / statistics.mean(xs)

print(f"population: {len(radii)} vehicles, r_RMS CV = {cv(radii):.2f}% "
      f"(DrivAerML train measured 2.95%)\n")
print(f"{'gauge':>12} {'mean ||drive||':>16} {'CV across samples':>20}")
print(f"{'fixed 8.0':>12} {statistics.mean(fixed):>16.6g} {cv(fixed):>19.4f}%")
print(f"{'intrinsic':>12} {statistics.mean(intrinsic):>16.6g} {cv(intrinsic):>19.4f}%")
print(f"\nratio of cross-sample spreads (intrinsic / fixed): "
      f"{cv(intrinsic)/cv(fixed):.4f}")
