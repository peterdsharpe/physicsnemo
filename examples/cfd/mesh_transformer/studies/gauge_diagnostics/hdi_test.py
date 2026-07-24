"""H-DI: is the resolution drift COVERAGE, or genuine quadrature non-convergence?

Two ways to give the model "more sources" for the SAME sphere:

  Arm A (subsample)  -- keep n panels of a fixed fine mesh.  Coverage = n/N:
                        the panels have gaps between them.  This is what the
                        recipe's SubsampleMesh does.
  Arm B (refine)     -- an icosphere at increasing subdivision level.  Coverage
                        is always 100%; the panels tile the sphere exactly.
                        This is the TEXTBOOK discretization-invariance test.

A boundary-integral operator must be invariant under Arm B.  Arm A is not a
resolution change at all -- it is a different, gappy surface.
"""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

torch.manual_seed(0)

def build():
    torch.manual_seed(0)
    return MeshTransformer(
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

model = build()
queries = Mesh(points=torch.tensor([[0., 0., 0.], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]]))

def predict(mesh):
    dom = DomainMesh(interior=queries, boundaries={"b": mesh},
                     global_data={"g": torch.tensor(1.0)})
    with torch.no_grad():
        return model(dom).point_data["u"].flatten().clone()

full = sphere_icosahedral.load(radius=1.0, subdivisions=5)   # 20480 cells

def subsample(mesh, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(mesh.n_cells, generator=g)[:n]
    v = mesh.points[mesh.cells[idx]]
    return Mesh(points=v.reshape(-1, 3), cells=torch.arange(3 * n).reshape(n, 3))

print("ARM B -- REFINEMENT (coverage 100%, the true discretization test)")
print(f"{'subdiv':>7} {'cells':>7} {'area':>10} {'prediction at 3 queries':>40}")
refB = None
for k in (2, 3, 4, 5):
    m = sphere_icosahedral.load(radius=1.0, subdivisions=k)
    p = predict(m)
    if refB is None: refB = p
    d = float((p - refB).abs().max() / refB.abs().max()) * 100
    print(f"{k:>7} {m.n_cells:>7} {float(m.cell_areas.sum()):>10.5f} "
          f"{p[0]:>13.6f}{p[1]:>13.6f}{p[2]:>13.6f}   (drift {d:6.2f}%)")

print("\nARM A -- SUBSAMPLING a fixed 20480-cell mesh (coverage = n/20480)")
print(f"{'n':>7} {'coverage':>9} {'area':>10} {'prediction at 3 queries':>40}")
refA = None
for n in (320, 1280, 5120, 20480):
    m = subsample(full, n)
    p = predict(m)
    if refA is None: refA = p
    d = float((p - refA).abs().max() / refA.abs().max()) * 100
    print(f"{n:>7} {100*n/full.n_cells:>8.1f}% {float(m.cell_areas.sum()):>10.5f} "
          f"{p[0]:>13.6f}{p[1]:>13.6f}{p[2]:>13.6f}   (drift {d:6.2f}%)")
