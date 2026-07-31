# H-QC neutral canonical-geometry diagnostic

This AGA task runs the frozen, target-free four-canary engineering
diagnostic registered in
`phase1_hqc_neutral_canonical_geometry_preregistration_v3_2026-07-28.json`.
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

The run first requires raw-byte bitwise replay of job 304002's selected IDs,
external geometry, internal source geometry, and primary/fixed predictions.
The reduced intervention is tested against relative L2 `1e-3` for pressure
and WSS in every case and precision. The coherent full bundle is separately
tested for raw-byte bitwise identity: equal-valued `+0.0` and `-0.0` fail
that gate. These two decision gates produce one of four registered outcomes:
`FULL_AND_DERIVED_PASS`, `FULL_ONLY_PASS`,
`CANONICAL_REPAIR_REFUTED`, or `INVALID_DIAGNOSTIC`. A valid scientific
refutation is preserved and exits zero; only failed diagnostic validity
exits nonzero. Safe validity failures are serialized before exit.

The wrapper binds exact hashes for the diagnostic, frozen H-QC loader,
schema-v3 preregistration, prior job 304002 JSON/NPZ, execution source tree,
checkpoint, normalization state, dataset manifest/config, and historical
metrics. The diagnostic also asserts the frozen physical `L_ref=5`, model
reference length `=8`, and effective scale `=40` exactly. It refuses to
overwrite outputs. A valid task creates atomic JSON/NPZ artifacts with
SHA-256 sidecars plus `DONE_<jobid>`; every exit creates
`STATUS_<jobid>`.
