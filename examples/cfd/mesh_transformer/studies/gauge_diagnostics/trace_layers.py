"""H-DEG trace prediction: WHERE does the magnitude ratio grow?

Pre-registered (@sec-nb-hdeg): under a family shift the ratio must COMPOUND
through the drive blocks if the amplifier is the multiplicative read-in; a
jump at the decode instead means a kernel/geometry singularity and H-DEG is
wrong.

Structural test: same random-init flagship, two boundaries whose statistics
differ the way two vehicle families do.  Degree is a property of the
architecture, so random init exercises the amplifier exactly; the trained
checkpoint would add weight-specific detail, not change which stage grows.
"""
import math, torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer
torch.set_default_dtype(torch.float64)

def flagship(mode):
    torch.manual_seed(0)
    return MeshTransformer(
        n_spatial_dims=3, output_field_ranks={"pressure": 0, "wss": 1},
        boundary_field_ranks={"vehicle": {"operator": {}, "drive": {}}},
        global_field_ranks={"operator": {}, "drive": {"U_inf_dir": 1}},
        reference_length_key="reference_length",
        field_mode=mode, query_decoder="kernel", trace_of="vehicle",
        kernel_mlp_members=8, kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        kernel_monopole_free_single_layer=False,
        operator_scalar_dim=64, operator_vector_dim=16,
        drive_scalar_dim=96, drive_vector_dim=24,
        operator_layers=2, drive_layers=1, query_layers=1,
        heads=1, scalar_rank=96, vector_rank=32,
        query_chunk_size=8192, attention_chunk_size=8192,
    ).eval().to(torch.float64)

_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
BASE = Mesh(points=_S.points.double(), cells=_S.cells)

def dom(mesh, gauge=8.0):
    return DomainMesh(
        interior=Mesh(points=mesh.cell_centroids.clone()),
        boundaries={"vehicle": mesh},
        global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                     "reference_length": torch.tensor(gauge, dtype=torch.float64)})

### "family B": same topology, different aspect/size statistics -- the kind of
### shift that separates two vehicle families under a FIXED declared gauge.
FAM_B = Mesh(points=BASE.points * torch.tensor([1.7, 0.75, 0.9], dtype=torch.float64),
             cells=BASE.cells)

def trace(model, domain):
    rec = {}
    hooks = []
    def mk(name):
        def hook(mod, inp, out):
            vals = []
            for o in (out if isinstance(out, (tuple, list)) else [out]):
                for t in (getattr(o, "scalars", None), getattr(o, "vectors", None),
                          o if torch.is_tensor(o) else None):
                    if torch.is_tensor(t) and t.numel():
                        vals.append(float(t.double().abs().max()))
            if vals:
                rec.setdefault(name, []).append(max(vals))
        return hook
    for n, m in model.named_modules():
        if n and not list(m.children()):
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        out = model(domain)
    for h in hooks: h.remove()
    peak = float(max(out.point_data["pressure"].abs().max(),
                     out.point_data["wss"].abs().max()))
    return rec, peak

for mode in ("zero_preserving_nonlinear", "homogeneous"):
    m = flagship(mode)
    ra, pa = trace(m, dom(BASE))
    rb, pb = trace(m, dom(FAM_B))
    print(f"\n=== {mode} ===   output ratio B/A = {pb/pa:,.4g}")
    print(f"{'stage (first module of each group)':>44} {'ratio B/A':>12}")
    seen = set()
    for name in ra:
        if name not in rb: continue
        group = name.split(".")[0] + "." + (name.split(".")[1] if "." in name[len(name.split('.')[0]):] else "")
        group = ".".join(name.split(".")[:2])
        if group in seen: continue
        seen.add(group)
        a, b = max(ra[name]), max(rb[name])
        if a > 0:
            print(f"{group:>44} {b/a:>12,.4g}")
