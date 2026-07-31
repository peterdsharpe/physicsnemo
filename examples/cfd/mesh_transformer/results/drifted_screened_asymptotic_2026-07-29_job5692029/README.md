# Two-limit asymptotic-carrier formal result

This directory contains the clean five-seed result for the registered
comparison of a raw learned kernel, a fixed two-limit carrier, and that carrier
with a learned transition correction.

## Scientific result

The registered verdict is `learned_transition_earned`.

The learned carrier passed every interpolation guard and all three joint
field-and-trace operator-transfer criteria in all five seeds. Relative to the
capacity-matched raw model, its geometric-mean field error fell from 0.01238
to 0.00142 in distribution, from 0.03547 to 0.00240 at low screening, from
0.25857 to 0.00819 at high screening, and from 0.05179 to 0.00154 at stronger
drift. Held-out-boundary field error fell from 0.01686 to 0.00475.

The parameter-free carrier isolated why learning remains necessary. It
improved both screening extremes and near-boundary evaluation, but was 4.16
times worse than the raw model in distribution and failed the interpolation
guard. The learned transition was 36.2 times more accurate than the fixed
carrier in distribution and improved every registered operator metric, so its
complexity was earned.

The result supports a specific mechanism: enforcing the correct small- and
large-screening forms turns the remaining transition into a learnable smooth
response. It does not yet establish transfer to spatially varying
coefficients, where no single global similarity coordinate gives the exact
kernel.

## Contents and provenance

- `formal/` contains the fifteen arm-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

The local independent reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`81eef13636e0fd645f448331b89ef6b75e69ae0c916e224a43a535a238e88ec3`.
All reports share source digest
`4b6c45d7678333fe7ef633b32d574e985c15d7d1619c8838b65295a4aeb4e8a1`.

All five raw controls exactly match the corresponding controls from the
preceding similarity-coordinate study across training history and every
scientific metric.
