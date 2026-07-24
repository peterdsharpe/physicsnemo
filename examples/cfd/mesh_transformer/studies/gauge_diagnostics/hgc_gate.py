"""H-GC cheap gate: drive-stream activation magnitude vs the declared gauge.

Builds a DrivAerML-like surface (tiny scattered triangles, car-sized box,
real area decades), runs the flagship-shaped MeshTransformer's encode at a
range of declared reference lengths, and reports the local log-log
sensitivity s = dlog||a|| / dlog L near L = 8.
"""
import math
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

torch.manual_seed(0)


def scattered(n, area_lo, area_hi, box, seed):
    g = torch.Generator().manual_seed(seed)
    hi = torch.tensor(box)
    centroids = hi * torch.rand(n, 3, generator=g)
    areas = torch.exp(math.log(area_lo)
                      + (math.log(area_hi) - math.log(area_lo))
                      * torch.rand(n, generator=g))
    a = torch.randn(n, 3, generator=g)
    b = torch.randn(n, 3, generator=g)
    u = a / a.norm(dim=-1, keepdim=True)
    b = b - (b * u).sum(-1, keepdim=True) * u
    v = b / b.norm(dim=-1, keepdim=True)
    side = (4.0 * areas / math.sqrt(3.0)).sqrt()
    p0 = centroids - side[:, None] * (u * 0.5 + v * math.sqrt(3.0) / 6.0)
    p1 = p0 + side[:, None] * u
    p2 = p0 + side[:, None] * (u * 0.5 + v * math.sqrt(3.0) / 2.0)
    i = torch.arange(n)
    return Mesh(points=torch.cat([p0, p1, p2]), cells=torch.stack([i, i + n, i + 2 * n], 1))


# Real DrivAerML vehicle statistics (from the recipe's own regression fixture).
vehicle = scattered(3000, 2.3e-9, 6.6e-5, (5.0, 2.0, 1.6), seed=7)

# Exact flagship configuration (conf/model/mesh_transformer_surface_flagship.yaml).
model = MeshTransformer(
    n_spatial_dims=3,
    output_field_ranks={"pressure": 0, "wss": 1},
    boundary_field_ranks={"vehicle": {"operator": {}, "drive": {}}},
    global_field_ranks={"operator": {}, "drive": {"U_inf_dir": 1}},
    reference_length_key="reference_length",
    field_mode="zero_preserving_nonlinear",
    query_decoder="kernel",
    trace_of="vehicle",
    kernel_mlp_members=8,
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

gauges = [4.0, 6.0, 7.41, 8.0, 8.64, 12.0, 16.0, 24.0, 48.0]
print(f"{'L_ref':>8} {'||drive state||':>18} {'||operator state||':>20}")
mags = {}
for L in gauges:
    domain = DomainMesh(
        interior=Mesh(points=vehicle.cell_centroids.clone()),
        boundaries={"vehicle": vehicle},
        global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
                     "reference_length": torch.tensor(float(L))},
    )
    with torch.no_grad():
        enc = model.encode(domain)
    drive = enc.drive_state
    op = enc.operator_state
    d = math.sqrt(sum(float(t.square().sum()) for t in (drive.scalars, drive.vectors)))
    o = math.sqrt(sum(float(t.square().sum()) for t in (op.scalars, op.vectors)))
    mags[L] = d
    print(f"{L:>8.2f} {d:>18.6g} {o:>20.6g}")

# Local log-log slope across the actual DrivAerML gauge span [7.41, 8.64].
s_band = (math.log(mags[8.64]) - math.log(mags[7.41])) / (math.log(8.64) - math.log(7.41))
s_wide = (math.log(mags[16.0]) - math.log(mags[6.0])) / (math.log(16.0) - math.log(6.0))
print(f"\nlocal slope s over the REAL gauge span [7.41, 8.64] : {s_band:+.3f}")
print(f"slope over a wide span [6, 16]                      : {s_wide:+.3f}")
spread = abs(s_band) * 3.5
print(f"\n=> a +/-3.5% gauge spread implies ~{spread:.1f}% drive-magnitude spread")
print(f"   PRE-REGISTERED: |s| in 1..4 => refute variance; |s| > 8 => variance live")
