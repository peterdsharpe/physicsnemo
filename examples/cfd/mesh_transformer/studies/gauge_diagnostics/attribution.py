"""Two registered follow-ups at once:
  (iii) is the member MLP the amplifier?  -> members in {0, 8}
   (i)  does the picture hold at LARGER shifts? -> shift ladder

End-to-end output ratio under a family-statistics shift, same random-init
flagship weights within each column.
"""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer
torch.set_default_dtype(torch.float64)

_S = sphere_icosahedral.load(radius=1.0, subdivisions=2)
BASE = Mesh(points=_S.points.double(), cells=_S.cells)

def flagship(mode, members):
    torch.manual_seed(0)
    return MeshTransformer(
        n_spatial_dims=3, output_field_ranks={"pressure": 0, "wss": 1},
        boundary_field_ranks={"vehicle": {"operator": {}, "drive": {}}},
        global_field_ranks={"operator": {}, "drive": {"U_inf_dir": 1}},
        reference_length_key="reference_length",
        field_mode=mode, query_decoder="kernel", trace_of="vehicle",
        kernel_mlp_members=members, kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        kernel_monopole_free_single_layer=False,
        operator_scalar_dim=24, operator_vector_dim=6,
        drive_scalar_dim=32, drive_vector_dim=8,
        operator_layers=2, drive_layers=1, query_layers=1,
        heads=1, scalar_rank=32, vector_rank=8,
        query_chunk_size=8192, attention_chunk_size=8192,
    ).eval().to(torch.float64)

def peak(m, mesh):
    d = DomainMesh(interior=Mesh(points=mesh.cell_centroids.clone()),
                   boundaries={"vehicle": mesh},
                   global_data={"U_inf_dir": torch.tensor([1.,0.,0.], dtype=torch.float64),
                                "reference_length": torch.tensor(8.0, dtype=torch.float64)})
    with torch.no_grad():
        o = m(d)
    return float(max(o.point_data["pressure"].abs().max(),
                     o.point_data["wss"].abs().max()))

### Shift ladder: increasingly different family statistics (anisotropic stretch).
SHIFTS = [(1.7, 0.75, 0.9), (3.0, 0.5, 0.8), (6.0, 0.3, 0.7)]
print(f"{'shift (aspect)':>22} " + "".join(
    f"{c:>22}" for c in ("nonlin m=8", "nonlin m=0", "homog m=8", "homog m=0")))
for sh in SHIFTS:
    fam = Mesh(points=BASE.points * torch.tensor(sh, dtype=torch.float64), cells=BASE.cells)
    row = f"{str(sh):>22} "
    for mode in ("zero_preserving_nonlinear", "homogeneous"):
        for mem in (8, 0):
            m = flagship(mode, mem)
            r = peak(m, fam) / peak(m, BASE)
            row += f"{r:>22,.4g}"
    print(row)
print("\nm=8 vs m=0 isolates the learned smooth members' contribution.")
