"""Why does subsampling bias the operator? Test the NEAR-FIELD mechanism.

A layer potential's value near the boundary is dominated by the closest
panels.  Random subsampling deletes ~99% of them, and no REWEIGHTING can
restore a singular integral's near field from a sparse sample -- which is
why the uniform HT factor (measured inert) cannot fix this.

Prediction: subsampling bias grows sharply as the query approaches the
boundary, and is small deep in the interior.
"""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

torch.manual_seed(0)
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

full = sphere_icosahedral.load(radius=1.0, subdivisions=5)

def subsample(mesh, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(mesh.n_cells, generator=g)[:n]
    v = mesh.points[mesh.cells[idx]]
    return Mesh(points=v.reshape(-1, 3), cells=torch.arange(3 * n).reshape(n, 3))

### Queries along a radius: deep interior -> just under the surface.
radii = [0.0, 0.25, 0.50, 0.75, 0.90, 0.97, 0.995]
d = torch.tensor([0.577, 0.577, 0.577])
queries = Mesh(points=torch.stack([r * d for r in radii]))

def predict(mesh):
    dom = DomainMesh(interior=queries, boundaries={"b": mesh},
                     global_data={"g": torch.tensor(1.0)})
    with torch.no_grad():
        return model(dom).point_data["u"].flatten().clone()

truth = predict(full)                      # 100% coverage = the reference
print("relative error vs the FULL-coverage operator, by query depth\n")
print(f"{'wall gap':>9} {'radius':>7} " + "".join(f"{f'{n} src':>12}" for n in (320, 1280, 5120)))
print(f"{'':>9} {'':>7} " + "".join(f"{f'({100*n/20480:.1f}%)':>12}" for n in (320, 1280, 5120)))
preds = {n: predict(subsample(full, n)) for n in (320, 1280, 5120)}
for i, r in enumerate(radii):
    row = "".join(f"{100*abs(float(preds[n][i]-truth[i])/float(truth[i])):>11.1f}%"
                  for n in (320, 1280, 5120))
    print(f"{1.0-r:>9.3f} {r:>7.3f} {row}")
print("\nDrift concentrating toward the wall => near-field loss, not a scale error.")
