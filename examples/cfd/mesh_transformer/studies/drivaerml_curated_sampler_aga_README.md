# Curated DrivAerML production-order sampler audit

This CPU-only AGA task reads the `run_1` and `run_2` vehicle-boundary memmaps
in place. It does not modify the dataset. At 40,000 cells it compares four
production-form cyclic blocks against four seeded simple-random subsets using
geometric-area, Cp, WSS, RMS-radius, and frozen-query fill-distance metrics.

The two cases are diagnostic examples, not a population estimate for the
435-case training split. The raw-VTP sampler artifact is kept separate because
the curated cell order is permuted.

Launch from this directory with:

```bash
sbatch drivaerml_curated_sampler_aga.sbatch
```

The job writes its JSON and SHA-256 sidecar under `artifacts/`, a merged log
under `sbatch_logs/`, and `STATUS_<job-id>`.
