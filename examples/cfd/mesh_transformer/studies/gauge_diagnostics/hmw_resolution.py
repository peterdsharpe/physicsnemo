"""Does stripping HT weights break DISCRETIZATION INVARIANCE?

A boundary-integral operator must not care how finely the SAME geometry is
sampled: sum_c |sigma_c| f_c is a quadrature and must converge, not scale.
compose_measure_weights exists to keep that true under subsampling.  The
model strips those weights.  Measure what the model's quadrature does as
the SAME surface is sampled at increasing source counts.
"""
import math, torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, compose_measure_weights
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

torch.manual_seed(0)

### One fixed sphere-like closed surface, sampled at several resolutions.
def sphere(n_sub):
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    return sphere_icosahedral.load(radius=1.0, subdivisions=n_sub)

full = sphere(5)                       # the "true" surface, ~20k cells
print(f"reference surface: {full.n_cells} cells, area = {float(full.cell_areas.sum()):.6f}"
      f"  (exact sphere = {4*math.pi:.6f})\n")

def subsample(mesh, n, seed=0):
    """What the recipe's SubsampleMesh does: keep n cells, record the HT weight."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(mesh.n_cells, generator=g)[:n]
    verts = mesh.points[mesh.cells[idx]]                    # (n,3,3)
    pts = verts.reshape(-1, 3)
    cells = torch.arange(3 * n).reshape(n, 3)
    out = Mesh(points=pts, cells=cells)
    compose_measure_weights(out, mesh.n_cells / n)
    return out

print(f"{'n_sources':>10} {'sum(cell_areas)':>17} {'sum(cell_measures)':>20}")
print(f"{'':>10} {'(what model uses)':>17} {'(HT-corrected)':>20}")
for n in (500, 1000, 2000, 4000, 8000):
    s = subsample(full, n)
    print(f"{n:>10} {float(s.cell_areas.sum()):>17.6g} {float(cell_measures(s).sum()):>20.6g}")

print("\n-> bare areas scale LINEARLY with source count; HT-corrected is invariant.\n")

### Now the model itself: does its output move with source count?
model = MeshTransformer(
    n_spatial_dims=3, output_field_ranks={"u": 0},
    boundary_field_ranks={"b": {"operator": {}, "drive": {}}},
    global_field_ranks={"operator": {}, "drive": {"g": 0}},
    field_mode="linear", query_decoder="kernel",
    kernel_mlp_members=0, kernel_include_polynomial_members=False,
    kernel_include_single_layer_member=True,
    operator_scalar_dim=32, operator_vector_dim=8,
    drive_scalar_dim=32, drive_vector_dim=8,
    operator_layers=1, drive_layers=1, query_layers=1,
    heads=1, scalar_rank=32, vector_rank=8,
).eval()

queries = Mesh(points=torch.tensor([[0., 0., 0.], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]]))
print(f"{'n_sources':>10} {'prediction at 3 interior query points':>44}")
for n in (500, 1000, 2000, 4000, 8000):
    s = subsample(full, n)
    dom = DomainMesh(interior=queries, boundaries={"b": s},
                     global_data={"g": torch.tensor(1.0)})
    with torch.no_grad():
        out = model(dom)
    v = out.point_data["u"].flatten()
    print(f"{n:>10}    {v[0]:>13.6g} {v[1]:>13.6g} {v[2]:>13.6g}")
print("\nA discretization-invariant operator returns the SAME values as the")
print("SAME surface is sampled more finely.")
