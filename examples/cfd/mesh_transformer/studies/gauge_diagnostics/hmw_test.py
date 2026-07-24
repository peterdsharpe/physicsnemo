"""H-MW-2: are the discarded Horvitz-Thompson weights UNIFORM in production?

Runs the recipe's real all-boundary transform chain on the synthetic tunnel
built to DrivAerML's measured shape statistics, then reports, per boundary:
the subsample ratio (= the recorded HT weight), the geometric measure the
model actually integrates with, and the HT-corrected measure it should.
"""
import sys, torch
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
from physicsnemo.mesh.calculus.measure import cell_measures, cell_measure_weights
from test_mesh_transformer_configs import (
    _tunnel_boundaries, _raw_vehicle_cell_data, _global_data,
    _instantiate_pipeline_transforms, _run_pipeline)
from physicsnemo.mesh import DomainMesh, Mesh

torch.manual_seed(3)
RES = 3000
boundaries = _tunnel_boundaries(n_vehicle_cells=60_000)   # vehicle >> cap
boundaries["vehicle"] = boundaries["vehicle"].with_data(
    cell_data=_raw_vehicle_cell_data(boundaries["vehicle"].n_cells))
raw = DomainMesh(
    interior=Mesh(points=torch.rand(2000, 3) * torch.tensor([120., 30.2, 40.])
                  + torch.tensor([-40., -15.1, 0.]),
                  point_data={"UMeanTrim": torch.randn(2000, 3) * 30.0}),
    boundaries=boundaries, global_data=_global_data(post_pipeline=False))

dom = _run_pipeline(raw, _instantiate_pipeline_transforms(
    "drivaer_ml_surface_allbc", sampling_resolution=RES))

print(f"all-boundary pipeline at sampling_resolution={RES}\n")
print(f"{'boundary':>10} {'cells':>7} {'HT weight':>11} "
      f"{'sum(cell_areas)':>17} {'sum(cell_measures)':>20}")
tot_bare = tot_ht = 0.0
rows = []
for name in sorted(dom.boundary_names):
    m = dom.boundaries[name]
    w = cell_measure_weights(m)
    bare = float(m.cell_areas.sum()); ht = float(cell_measures(m).sum())
    tot_bare += bare; tot_ht += ht
    rows.append((name, bare, ht))
    print(f"{name:>10} {m.n_cells:>7} {float(w[0]):>11.4g} "
          f"{bare:>17.6g} {ht:>20.6g}")

print(f"\n{'':>10} {'':>7} {'':>11} {tot_bare:>17.6g} {tot_ht:>20.6g}  <- totals")
print("\nshare of total quadrature measure:")
print(f"{'boundary':>10} {'model sees (bare)':>19} {'should be (HT)':>16} {'error':>10}")
for name, bare, ht in rows:
    a, b = 100 * bare / tot_bare, 100 * ht / tot_ht
    print(f"{name:>10} {a:>18.4f}% {b:>15.4f}% {b/a if a else float('nan'):>9.2f}x")
