# Nonseparable latent composition — reduced-budget pilot

## Exploratory result

At 750 updates, the physical-order rank-four model reduced in-distribution
cross-channel error to 42%, versus 60% for global rank-four compression and
100% for the sorted control. It recovered 44% of the heterogeneous
layer-swap contrast, compared with 25% for the global model and zero for the
sorted model.

The result is not sufficient to advance the claim. Fixed rank-four spectral
truncation reached 29% in-distribution error, and the ordered learned model
retained 91%--94% error under faster coefficient variation. The per-case
rank-four oracle was near 11%--12%.

This run is an optimization and effect-size pilot only. It used one seed, 750
updates, 1,024 training samples per heterogeneity level, 32 evaluation
profiles, and 12 query positions. A full-budget single-seed check follows
before formal gates are frozen.

## Provenance

- HSG Slurm job: `5694534`
- Terminal state: `COMPLETED`, exit code `0:0`, elapsed time `00:00:55`
- Study source SHA-256 at staging:
  `562d28f3a03cf357a1129a0732bbf0972f9f741bc769213452cf9718aceb740f`

The `pilot/` directory contains all seven reports and their full stdout logs.
