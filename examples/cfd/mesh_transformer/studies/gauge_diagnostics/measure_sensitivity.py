"""CORRECTED test: is the model sensitive to a uniform MEASURE rescale at FIXED geometry?

The earlier test (hmw_causation.py) scaled the geometry by s AND the declared
gauge by s, leaving the NONDIMENSIONAL geometry -- and hence the nondimensional
cell measures -- identical.  That measured scale equivariance, not measure
sensitivity.  It cannot support the claim that restoring HT weights is inert.

Correct test: hold geometry fixed, multiply only the cell MEASURES by k
(exactly what a Horvitz-Thompson weight does), and look at the output.
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

mesh = sphere_icosahedral.load(radius=1.0, subdivisions=4)
queries = Mesh(points=torch.tensor([[0., 0., 0.], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]]))

def predict():
    dom = DomainMesh(interior=queries, boundaries={"b": mesh},
                     global_data={"g": torch.tensor(1.0)})
    with torch.no_grad():
        return model(dom).point_data["u"].flatten().clone()

_orig = MeshClass.cell_areas
base = None
print("geometry FIXED; only the quadrature measure is multiplied by k")
print(f"{'k':>8} {'prediction at 3 interior queries':>44} {'drift':>10}")
try:
    for k in (1.0, 2.0, 4.0, 16.0, 64.0, 880.0):
        MeshClass.cell_areas = property(lambda self, _k=k: _orig.fget(self) * _k)
        p = predict()
        if base is None:
            base = p.clone()
        d = float((p - base).abs().max() / base.abs().max()) * 100
        print(f"{k:>8.1f} {p[0]:>14.6f}{p[1]:>14.6f}{p[2]:>14.6f} {d:>9.3f}%")
finally:
    MeshClass.cell_areas = _orig

print("\nk = 880 is the real DrivAerML HT factor (8.8M cells -> 10k subsample).")
print("If drift ~ 0 the model truly cannot use an HT correction;")
print("if drift is large, restoring HT weights is a REAL intervention.")
