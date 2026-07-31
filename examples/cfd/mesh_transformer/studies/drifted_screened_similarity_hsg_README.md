# Similarity-coordinate test for a drifted screened kernel

## Scientific question

Does operator extrapolation fail because the learned remainder receives raw
parameters instead of the dimensionless groups that govern the Green kernel?

For displacement `x-y = r rhat`, the exact scaled kernel depends on drift and
screening through

`z = r sqrt(kappa^2 + |b|^2/4)` and `eta = r b.rhat`.

The similarity representation supplies `z` and `eta`, but not the exponential
or Bessel response. It retains the same six inputs, learned singular form,
network capacity, initialization, training supports, and budget as the raw
representation.

## Registered factorial

The four arms cross:

- raw versus similarity coordinates; and
- pointwise versus hybrid supervision.

The representation claim requires noninferiority on interpolation, held-out
boundary spectra, near-boundary fields, and refinement, plus at least a 30%
reduction in both field and exact-trace error on two of the three
operator-parameter extrapolations in at least four of five paired seeds.

Passing under both losses identifies raw coordinates as a principal cause.
Passing only under hybrid supervision identifies a coordinate-by-objective
interaction. A split-specific gain is useful but does not establish general
operator transfer.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: four-arm signal check
- `artifacts/formal/`: twenty registered arm-by-seed reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Each mode refuses to overwrite prior reports.

## Status

The four-arm pilot completed cleanly and reproduced the preceding raw-arm
pilot metrics exactly. It is a signal check, not a verdict.

The registered five-seed formal factorial also completed cleanly. Its verdict
is `similarity_coordinates_not_principal`. Neither similarity arm passed the
interpolation guards or any joint field-and-trace operator-transfer split.
Both improved high-screening behavior, especially the trace residual, but
lost substantially on interpolation and low screening. All ten raw formal
controls exactly reproduce the preceding supervision study.
