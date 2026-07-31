# Coupled physical-order composition formal result

This directory contains the clean five-seed result for the registered test of
physical-order composition with coupled local modes.

## Scientific result

The registered verdict is `coupled_path_composition_earned`.

The physical-order generator recovered 98.3%--99.5% of the true
layer-permutation response contrast in all five seeds. The position-aware
pooled model recovered 72.1%--74.1%, while the sorted generator recovered
exactly zero. Physical-order composition reduced paired field error by a
factor of 17.7 relative to pooling and passed the interpolation guard in every
seed.

The breadth gates also passed in all five seeds. Relative to pooling, the
physical-order model reduced cross-channel error by a factor of 44 on unseen
twist rates and 38 when the number of layers doubled. Its full-operator error
remained about 0.06% on both shifts, versus roughly 1% for pooling.

The full-operator metric alone understates the distinction: direct-channel
response dominates its norm, while the pooled model's held-out cross-channel
error was 25%. The registered cross-channel metric exposed that failure. The
physical-order model learned the local coefficient map to 0.50% relative
error; the order-blind sorted model remained at 14.5%, showing that correct
composition also improved identification of the local map from global field
data.

The analytic path oracle was exact, the sorted arm was exactly blind to the
matched permutation, and all boundary, determinant, and rotation-covariance
certifications passed. The result supports physical-order composition as a
coupled propagation primitive. It does not yet establish a surrogate for a
full nonseparable two-dimensional field, and the exact local state-space form
still carries substantial solver structure.

## Contents and provenance

- `formal/` contains the twenty arm-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

An independent local reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`96e3f46071468f874bd81004f7bb090d720d1bb5b433c4ed1c62d25954746117`.
All reports share source digest
`799538179682a2d47bf8bd012a7ea5e74594db7a8271b62814fccca66a2fba56`.
