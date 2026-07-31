# Nonseparable diagonal-carrier gate

## Scientific question

Does exact uncoupled propagation provide the prior needed to learn a
nonseparable mode-conversion correction progressively from partial target
coverage?

## Registered comparison

The fixed carrier solves the diagonal part of the known local spectral
operator and uses no coupled target labels. A boundary-linear coordinate
network learns only the residual response. Three capacity controls and twelve
target-only arms use the same nested orthogonal drives as the corrected
saturation study.

The instrument advances to source transfer only if every capacity seed stays
below 0.75% (geometric mean below 0.5%), eight target fields reach 1.5%, one
field remains above 2.5%, four and eight fields materially improve on one,
and the geometric-mean curve is strictly monotone.

## Execution layout

- `code/`: staged source
- `artifacts/formal/`: three capacity and twelve target-only reports
- `artifacts/summary.json`: frozen reduction
- `sbatch_logs/`: merged scheduler log
- `STATUS_*` / `DONE_*`: terminal state

## Status

Completed as HSG job 5700594. The capacity control passed at 0.099%
geometric-mean error. Target-only error decreased monotonically from 5.55%,
5.21%, and 4.67% at one, two, and four fields to 0.104% at eight fields.
However, the four-field result was 84.1% of the one-field error rather than
the registered maximum of 80%. The formal verdict is therefore
`reject_diagonal_carrier_instrument`; no source-transfer comparison is
authorized.
