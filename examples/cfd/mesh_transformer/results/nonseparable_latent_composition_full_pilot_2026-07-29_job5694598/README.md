# Nonseparable latent composition — full-budget pilot

## Exploratory result

At the planned training budget, physical-order rank four reached 27.0%
cross-channel error in distribution and 30.1% at high heterogeneity. This
improved the global rank-four model by 28% and 36%, respectively, and matched
fixed rank-four truncation within 11% and 4%.

The layer-swap result moved in the same direction: physical order recovered
58% of the heterogeneous response, versus 43% for global compression, zero
for the sorted control, 69% for fixed truncation, and 83% for the per-case
oracle.

Breadth remained weak. Physical-order error was 67.8% on faster spatial
variation and 65.5% on the combined fast/high shift, compared with 39.1% and
37.6% for fixed truncation. This pilot therefore motivates separate formal
claims for ordered compression and breadth transfer.

## Provenance

- HSG Slurm job: `5694598`
- Terminal state: `COMPLETED`, exit code `0:0`, elapsed time `00:01:01`
- Study source SHA-256 used by this pilot:
  `562d28f3a03cf357a1129a0732bbf0972f9f741bc769213452cf9718aceb740f`

This is a single-seed exploratory artifact. Its evaluation profiles are not
used in the formal comparison.
