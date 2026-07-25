"""Smoke test: does the FLAGSHIP configuration run with field_mode=quadratic?
Forward + backward, exactly the flagship's other settings.  Run BEFORE
queueing a 14-hour arm behind it (the covaug lesson)."""
import torch
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer

_S = sphere_icosahedral.load(radius=1.0, subdivisions=3)
veh = Mesh(points=_S.points, cells=_S.cells)
n = veh.n_cells
for mode in ("zero_preserving_nonlinear", "quadratic"):
    torch.manual_seed(0)
    try:
        m = MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"pressure": 0, "wss": 1},
            boundary_field_ranks={"vehicle": {"operator": {}, "drive": {}}},
            global_field_ranks={"operator": {}, "drive": {"U_inf_dir": 1}},
            reference_length_key="reference_length",
            field_mode=mode, query_decoder="kernel", trace_of="vehicle",
            kernel_mlp_members=8, kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            kernel_monopole_free_single_layer=False,
            kernel_checkpoint_query_chunks=True,
            operator_scalar_dim=64, operator_vector_dim=16,
            drive_scalar_dim=96, drive_vector_dim=24,
            operator_layers=2, drive_layers=1, query_layers=1,
            heads=1, scalar_rank=96, vector_rank=32,
            query_chunk_size=4096, attention_chunk_size=4096)
        dom = DomainMesh(
            interior=Mesh(points=veh.cell_centroids.clone()),
            boundaries={"vehicle": veh},
            global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
                         "reference_length": torch.tensor(8.0)})
        out = m(dom)
        loss = out.point_data["pressure"].square().mean() + out.point_data["wss"].square().mean()
        loss.backward()
        gn = sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)
        params = sum(p.numel() for p in m.parameters())
        print(f"{mode:>28}: OK  params={params:,}  loss={float(loss):.4g}  gradnorm2={gn:.4g}")
    except Exception as e:
        print(f"{mode:>28}: FAILED  {type(e).__name__}: {e}")
