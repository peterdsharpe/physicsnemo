# H-QC neutral canonical-geometry diagnostic

This AGA task runs the frozen, target-free four-canary engineering
diagnostic registered in
`phase1_hqc_neutral_canonical_geometry_preregistration_v1_2026-07-28.json`.
It is not an H-QC rerun and computes no truth-relative error, force,
area-objective, endpoint, eligibility, support, futility, or mixed statistic.

For each frozen `K=2500` case, the script builds one neutral source bundle
from the raw selected topology before either historical `CenterMesh`. Raw
float32 coordinates are promoted to float64; the physical area-weighted
center and coherent points/centroids/areas/normals are computed; the bundle
is divided by the frozen `L_ref * reference_length = 40`; and each field is
cast once to float32. Identical bundle tensors and canonical trace centroids
are then supplied symmetrically to the historical primary and fixed paths in
both bfloat16 and float32 contexts.

The run first requires bitwise replay of job 304002's selected IDs, external
geometry, internal source geometry, and primary/fixed predictions. The
reduced intervention must keep pressure and WSS within relative L2 `1e-3`
for every case and precision. The coherent full bundle must make both fields
bitwise identical. Safe gate failures are serialized before the script exits
nonzero.

The wrapper binds exact hashes for the diagnostic, frozen H-QC loader, new
preregistration, prior job 304002 JSON/NPZ, execution source tree, checkpoint,
normalization state, dataset manifest/config, and historical metrics. It
refuses to overwrite outputs. A successful task creates atomic JSON/NPZ
artifacts with SHA-256 sidecars plus `DONE_<jobid>`; every exit creates
`STATUS_<jobid>`.
