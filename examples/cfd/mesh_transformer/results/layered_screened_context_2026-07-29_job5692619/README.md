# Layered variable-coefficient context formal result

This directory contains the clean five-seed result for the registered
comparison of optical summaries and position-aware coefficient context.

## Scientific result

The registered verdict is `feedforward_context_not_sufficient`.

The matched layer permutations produced a 4.0% true response contrast while
remaining exactly identical to every scalar-summary input. As required, the
fixed and scalar arms predicted zero contrast. Neither ordered arm met the 80%
recovery gate: the raw ordered model recovered 8%--52% across seeds, while the
ordered carrier recovered essentially none. Their paired field errors were
2.8 and 2.5 times lower than the scalar model, short of the registered
fivefold threshold.

The optical carrier did not earn the separate representation claim. Relative
to the capacity-matched raw ordered model, the ordered carrier was 72% worse
in distribution and 89% worse at low coefficients. It improved held-out-mode
error by a factor of 2.4, but no registered transfer split passed. The
parameter-free carrier itself was the best arm on held-out modes and both
coefficient extremes, while remaining the worst interpolation and layer-order
model.

The result distinguishes information from computation. Scalar optical
summaries are provably insufficient because layer transfer matrices do not
commute. Supplying ordered samples carries some of the missing signal, but a
mean-pooled feed-forward encoder trained on field error did not learn the
ordered composition reliably. The next test should build path ordering or a
residual-controlled propagation solve into the architecture.

## Contents and provenance

- `formal/` contains the twenty arm-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

The local independent reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`682b5662181439892eb95e46f5d44dcc297a8ef30ce6cb3c29b1ac8c4c1b1449`.
All reports share source digest
`374cdedbcc070c429019a402671c03761ed8fdf7e4f91bab8b6abd770124b206`.

