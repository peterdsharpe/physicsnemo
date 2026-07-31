# H-QC preprocessing-center failure diagnostic

Exploratory, non-deciding follow-up to failed H-QC v2 job 303890. The job
reproduces `run_118` at `K=2500` with the exact frozen checkpoint and source
tree, but it does not score truth or emit an H-QC verdict.

The probe distinguishes:

- the stored pipeline-normal field from normals recomputed internally by
  `MeshTransformer`;
- query-coordinate drift from source-geometry drift;
- internal normal drift from centroids/areas and full panel geometry; and
- historical bfloat16 inference from float32 inference.

It saves an immutable JSON summary and an NPZ containing geometry and prediction
arrays without truth fields. Both receive SHA-256 sidecars.

Remote task directory:

`/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-hqc-center-diagnostic`

Launch:

```bash
sbatch drivaerml_hqc_center_diagnostic_aga.sbatch
```

Initial status: staged locally, not yet launched.
