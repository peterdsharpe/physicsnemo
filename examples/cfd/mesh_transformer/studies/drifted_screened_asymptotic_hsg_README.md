# Two-limit asymptotic carrier for a drifted screened kernel

## Scientific question

Does a learned boundary kernel fail under operator extrapolation because its
additive output representation enforces the singular limit but not the
exponentially screened limit?

The fixed carrier blends the leading small-screening response with the
leading large-screening decay and includes the known angular and drift-gauge
factors. It deliberately omits the exact Bessel response. The learned carrier
adds a bounded, capacity-matched transition correction that vanishes in both
limits.

## Registered comparison

The three arms are:

- the current raw-coordinate kernel under hybrid supervision;
- the fixed, parameter-free two-limit carrier; and
- the carrier with an 8,834-parameter learned correction under the same
  hybrid supervision.

The learned scaffold must remain within 20% of the raw model on interpolation,
held-out boundary modes, near-boundary fields, and refinement while improving
both field and exact-trace error by at least 30% on two of three
operator-parameter extrapolations in at least four of five seeds.

Learning earns its complexity over the fixed carrier only if it improves
interpolation and held-out-boundary fields by at least 30% without worsening
any operator split by more than 20%.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: fifteen registered arm-by-seed reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Each mode refuses to overwrite prior reports.

## Status

The four-report pilot completed cleanly. It is a signal check, not a verdict.
The fixed and learned carriers strongly improved the screened and drifted
extrapolation cases at 500 steps, while the learned correction also reduced
the fixed carrier's interpolation error.

The registered five-seed formal comparison completed successfully. Its
verdict is `learned_transition_earned`: the learned carrier passed all four
interpolation guards, all three operator-transfer criteria, and the
complexity-over-fixed-carrier gate. The frozen result is in
`results/drifted_screened_asymptotic_2026-07-29_job5692029/`.
