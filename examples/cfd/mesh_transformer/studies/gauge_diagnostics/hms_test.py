"""H-MS: do NORMALISED quadrature measures fix measure-scale sensitivity (P1)
and reduce the subsampling bias (P2)?

Prototype: patch Mesh.cell_areas to return omega / sum(omega).  That is the
proposed 'shape' half of the split; the 'magnitude' half (one dimensionless
global scalar) is not needed for these two predictions.
"""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.mesh import Mesh as MeshClass
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
queries = Mesh(points=torch.tensor([[0., 0., 0.], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]]))
_orig = MeshClass.cell_areas

def predict(mesh):
    dom = DomainMesh(interior=queries, boundaries={"b": mesh},
                     global_data={"g": torch.tensor(1.0)})
    with torch.no_grad():
        return model(dom).point_data["u"].flatten().clone()

def subsample(mesh, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(mesh.n_cells, generator=g)[:n]
    v = mesh.points[mesh.cells[idx]]
    return Mesh(points=v.reshape(-1, 3), cells=torch.arange(3 * n).reshape(n, 3))

NORM = property(lambda self: _orig.fget(self) / _orig.fget(self).sum())

### ---- P1: invariance to uniform measure rescale --------------------------
print("P1 -- uniform measure rescale (geometry fixed)")
print(f"{'k':>8} {'baseline drift':>16} {'normalised drift':>18}")
mesh4 = sphere_icosahedral.load(radius=1.0, subdivisions=4)
b0 = n0 = None
for k in (1.0, 16.0, 880.0):
    MeshClass.cell_areas = property(lambda self, _k=k: _orig.fget(self) * _k)
    pb = predict(mesh4)
    MeshClass.cell_areas = property(
        lambda self, _k=k: (lambda a: a * _k / (a * _k).sum())(_orig.fget(self)))
    pn = predict(mesh4)
    MeshClass.cell_areas = _orig
    if b0 is None: b0, n0 = pb.clone(), pn.clone()
    db = float((pb - b0).abs().max() / b0.abs().max()) * 100
    dn = float((pn - n0).abs().max() / n0.abs().max()) * 100
    print(f"{k:>8.0f} {db:>15.3f}% {dn:>17.3e}%")

### ---- P2: subsampling bias at fixed coverage -----------------------------
print("\nP2 -- subsampling bias vs the FULL-coverage operator (same weighting)")
print(f"{'coverage':>9} {'baseline err':>14} {'normalised err':>16}")
for scheme, label in ((_orig, "baseline"), (NORM, "normalised")):
    pass
res = {}
for label, prop in (("baseline", _orig), ("normalised", NORM)):
    MeshClass.cell_areas = prop
    try:
        truth = predict(full)
        errs = []
        for n in (320, 1280, 5120):
            p = predict(subsample(full, n))
            errs.append(float((p - truth).abs().max() / truth.abs().max()) * 100)
    finally:
        MeshClass.cell_areas = _orig
    res[label] = errs
for i, n in enumerate((320, 1280, 5120)):
    print(f"{100*n/full.n_cells:>8.1f}% {res['baseline'][i]:>13.1f}% "
          f"{res['normalised'][i]:>15.1f}%")
b, nn = res["baseline"][0], res["normalised"][0]
print(f"\nP2 bar: >=2x reduction at 1.6% coverage. "
      f"measured {b:.1f}% -> {nn:.1f}%  = {b/nn:.2f}x  "
      f"{'PASS' if b/nn >= 2 else 'FAIL'}")
