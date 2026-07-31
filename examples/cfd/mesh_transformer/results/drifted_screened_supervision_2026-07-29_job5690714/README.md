# Supervision-alignment formal result

This directory contains the clean five-seed result for the registered
pointwise-versus-solution-supervision comparison.

## Scientific result

The registered verdict is `supervision_mismatch_not_principal`.

Solution-only and hybrid training both improved in-distribution field error,
held-out boundary-spectrum transfer, and near-boundary accuracy. Neither
improved both field error and exact-trace residual on any of the three
operator-parameter extrapolations under the registered criterion. The
solution-only arm also produced much larger exact-trace residuals, showing
that accurate fields on the training distribution do not identify the
boundary factorization uniquely. Hybrid supervision retained trace fidelity
and is useful for interpolation, geometry transfer, and boundary-spectrum
transfer, but it did not solve operator extrapolation.

The next registered test therefore concerns operator-normalized coordinates,
not another loss function.

## Contents and provenance

- `formal/` contains the fifteen arm-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

The local independent reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`77c1060278928a9d3e9a7bbf1e31bded10027cb61738e949332ab123c24cb5ff`.
All reports share source digest
`62a9029193de452b8684d17685d07a3e46163d00797cab05c610fdf1dd7f5e4a`.

This bundle comes only from the isolated clean formal run. Two earlier
submissions accidentally targeted one output directory concurrently; both
were cancelled and their directory was excluded before any scientific
analysis.
