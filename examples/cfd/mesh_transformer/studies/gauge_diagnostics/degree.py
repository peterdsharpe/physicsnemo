"""H-DEG, structural half: what is the effective drive degree of the mode the
FLAGSHIP actually uses, and how much does it amplify an input shift?

Degree is a property of the architecture, not the weights, so a random-init
model measures it exactly.  Scale the drive by lambda; if the output scales as
lambda^d, d is the effective degree.  A degree-d map turns an input-statistic
ratio r into r^d at the output -- which is the proposed cross-family amplifier.
"""
import math, torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer
torch.set_default_dtype(torch.float64)

_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
MESH = Mesh(points=_S.points.double(), cells=_S.cells)
Q = Mesh(points=torch.tensor([[0.,0.,0.],[.3,.1,-.2],[.5,.5,.1]], dtype=torch.float64))

def build(mode):
    torch.manual_seed(0)
    return MeshTransformer(
        n_spatial_dims=3, output_field_ranks={"u": 0},
        boundary_field_ranks={"b": {"operator": {}, "drive": {"g": 0}}},
        global_field_ranks={"operator": {}, "drive": {"h": 0}},
        field_mode=mode, query_decoder="kernel",
        kernel_mlp_members=8, kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        operator_scalar_dim=32, operator_vector_dim=8,
        drive_scalar_dim=32, drive_vector_dim=8,
        operator_layers=2, drive_layers=1, query_layers=1,
        heads=1, scalar_rank=32, vector_rank=8,
    ).eval().to(torch.float64)

def out_norm(m, lam):
    n = MESH.n_cells
    dom = DomainMesh(
        interior=Q,
        boundaries={"b": MESH.with_data(cell_data={
            "g": lam * torch.linspace(-1, 1, n, dtype=torch.float64)})},
        global_data={"h": torch.tensor(lam, dtype=torch.float64)})
    with torch.no_grad():
        return float(m(dom).point_data["u"].abs().max())

print(f"{'mode':>28} {'lambda':>8} {'|output|':>14} {'local degree':>14}")
for mode in ("zero_preserving_nonlinear", "quadratic", "linear"):
    m = build(mode)
    lams = [0.5, 1.0, 2.0, 4.0]
    vals = [out_norm(m, l) for l in lams]
    for i, (l, v) in enumerate(zip(lams, vals)):
        d = ""
        if i > 0 and vals[i-1] > 0:
            d = f"{math.log(v/vals[i-1])/math.log(l/lams[i-1]):.2f}"
        print(f"{mode if i==0 else '':>28} {l:>8.2f} {v:>14.6g} {d:>14}")
    if vals[0] > 0:
        deg = math.log(vals[-1]/vals[0])/math.log(lams[-1]/lams[0])
        print(f"{'':>28} {'':>8} {'overall degree:':>14} {deg:>13.2f}")
        for r in (1.5, 2.0, 4.0):
            print(f"{'':>28} {'':>8} amplification of a {r}x input shift: {r**deg:.3g}")
    print()
