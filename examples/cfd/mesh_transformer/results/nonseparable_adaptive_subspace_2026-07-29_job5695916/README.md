# Formal moving-subspace realizability census

## Scientific verdict

The natural connection-aware moving rank-four subspace earned none of four
registered profile batches:

`adaptive_subspace_not_earned`

Geometric-mean cross-channel relative errors were:

| Propagator | Low frequency | Mid frequency | High, 16 layers | High, 32 layers |
|---|---:|---:|---:|---:|
| Fixed rank four | **0.313** | **0.388** | **0.353** | **0.357** |
| Global eigenspace | 0.313 | 0.389 | 0.354 | 0.358 |
| Local basis, no connection | 0.381 | 0.791 | 0.939 | 0.915 |
| Local basis, exact connection | 0.314 | 0.390 | 0.356 | 0.360 |
| Static per-query oracle | 0.123 | 0.124 | 0.085 | 0.091 |

Ignoring coordinate connections made the moving basis catastrophically wrong.
Applying the exact overlap between adjacent bases reduced error by factors of
2.0--2.6 on the shifted spectra. Once corrected, however, the moving basis was
within roughly one percent of fixed and global truncation and slightly worse
on every aggregate split. It supplied necessary coordinate bookkeeping, not
an adaptive accuracy gain.

The static oracle remains excellent because it chooses its factors separately
after each complete solution is known. This experiment found no corresponding
local moving-subspace construction. Under the registered decision, the
controlled rank-four branch ends with fixed reduced propagation as the robust
baseline.

## Provenance

The four reports and frozen `summary.json` are the authoritative outputs of
HSG Slurm job `5695916`, which completed successfully in 44 seconds. The
summary SHA-256 is
`c5261a400b43e5a095f2b86ff2e8dbe6fda26cba8f6cf1f2b61c6b4b9ec25214`.
All reports record the same relevant-source fingerprint,
`7cae61c627f7b65cad08e0f4941e75611bf1f9df38535e4f044f96197636ce15`.
An independent local reduction was byte-identical.

