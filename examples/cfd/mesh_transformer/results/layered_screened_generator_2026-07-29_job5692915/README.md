# Path-ordered local-generator formal result

This directory contains the clean five-seed result for the registered test of
physical-order composition in a layered variable-coefficient operator.

## Scientific result

The registered verdict is
`composition_earned_local_law_not_transferable`.

The physical-order generator recovered 97.9%--99.2% of the true layer-order
contrast in all five seeds. Its geometric-mean paired field error was 0.00145,
26 times lower than the mean-pooled ordered model and 74 times lower than the
scalar-summary model. It also reduced interpolation error by a factor of 30
and held-out-mode error by a factor of 310 relative to the pooled model.
Every registered composition gate passed.

The capacity-matched generator that sorted layers before composition predicted
exactly zero contrast, as required. This isolates physical ordering as the
cause of the gain rather than parameter count or the local state-space form
alone.

The separate local-law transfer claim failed. The physical-order generator
reduced low-coefficient error by a factor of 4.6 in all five seeds, but its
high-coefficient error was 15% worse in geometric mean and improved in only
two seeds. Ordered composition is therefore supported; extrapolation of the
field-supervised local constitutive map is not.

The analytic path oracle was exact on every split and the matched-pair test,
and all numerical-certification checks passed. The next test should preserve
physical-order composition while constraining or directly supervising the
local coefficient law outside the training range.

## Contents and provenance

- `formal/` contains the twenty-five arm-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

An independent local reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`33645837048b3b70e135ef575dd7041d4e84cef29d8bf8502f841b08d230bc82`.
All reports share source digest
`a6200d22bb38581bdd01b1c1baeb87fd5d238e9e99f10a3f65a8192ed2bf3f74`.
The ten replayed scalar and pooled control reports exactly match the
corresponding scientific payloads from the preceding formal experiment.
