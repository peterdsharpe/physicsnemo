"""Is the model even SENSITIVE to a uniform measure rescale?

For a SINGLE boundary the HT correction is exactly a uniform factor
(n_before/n_after) on every cell measure.  If the model is invariant to a
uniform measure rescale, then restoring HT weights CANNOT change anything
for single-boundary resolution transfer, and the observed drift has another
cause entirely (coverage, not weighting).
"""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

torch.manual_seed(0)
full = sphere_icosahedral.load(radius=1.0, subdivisions=4)

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
queries = Mesh(points=torch.tensor([[0., 0., 0.], [0.3, 0.1, -0.2]]))

### Uniform measure rescale, applied GEOMETRICALLY is impossible without
### changing shape -- so instead scale the whole geometry by s, which scales
### every cell measure by s^2 while the model's intrinsic/declared gauge
### absorbs the length change.  With a DECLARED gauge scaled by s too, the
### nondimensional geometry is identical and ONLY the measure magnitude moves.
print("Same nondimensional shape; only the absolute measure magnitude changes.")
print(f"{'geometry scale':>15} {'cell-measure factor':>21} {'prediction':>26}")
base = None
for s in (0.5, 1.0, 2.0, 4.0):
    scaled = Mesh(points=full.points * s, cells=full.cells)
    dom = DomainMesh(interior=Mesh(points=queries.points * s),
                     boundaries={"b": scaled},
                     global_data={"g": torch.tensor(1.0),
                                  "L": torch.tensor(float(s))})
    with torch.no_grad():
        out = model(dom).point_data["u"].flatten()
    if base is None:
        base = out.clone()
    rel = float((out - base).abs().max() / base.abs().max()) * 100
    print(f"{s:>15.2f} {s**2:>21.2f} {out[0]:>13.6f} {out[1]:>11.6f}   "
          f"(drift {rel:.3f}%)")

print("\nIf drift ~ 0: the model is invariant to a uniform measure rescale,")
print("so restoring HT weights cannot affect SINGLE-boundary resolution transfer.")
