# H-QC preprocessing-center failure diagnostic

Exploratory, non-deciding follow-up to failed H-QC v2 job 303890. The corrected
v2 diagnostic covers all four failed `K=2500` canaries (`run_118`, `run_129`,
`run_145`, and `run_149`) with the exact frozen checkpoint and source tree. It
does not use or persist target values, score truth, or emit an H-QC verdict.

This supersedes diagnostic job 303951 only for its normal-angle summaries:
that first artifact took dot products of nominally unit float32 vectors without
renormalizing them in float64. Its componentwise, relative, and prediction
differences remain valid and its complete bundle is preserved.

The probe distinguishes:

- zeroing or copying the stored pipeline-normal field from normals recomputed
  internally by `MeshTransformer`;
- query-coordinate drift from source-geometry drift;
- internal normal, centroid, area, pairwise-combination, and full-panel pins;
- historical bfloat16 inference from float32 inference.

It saves an immutable JSON summary and an NPZ containing geometry and prediction
arrays without truth fields. Both receive SHA-256 sidecars.

The validity-only preregistration is
`phase1_hqc_center_cause_diagnostic_preregistration_v2_2026-07-28.json`,
SHA-256
`6e86a16a73869d382b8b98bd4cd3c0059ffb1c65334542c5eb25715267db8594`.
The wrapper verifies it together with the corrected diagnostic, frozen H-QC
producer, execution source tree, dataset, config, and checkpoint inputs.

Remote task directory:

`/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-hqc-center-diagnostic-v2`

Launch:

```bash
sbatch drivaerml_hqc_center_diagnostic_aga.sbatch
```

Initial status: frozen locally and not yet launched.
