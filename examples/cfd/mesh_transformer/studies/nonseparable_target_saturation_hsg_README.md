# Nonseparable target-field saturation

## Scientific question

Does the admitted nonseparable elliptic task retain a genuine multi-field
target-learning regime for a capable learner that is exactly linear in the
boundary drive?

## Registered comparison

The capacity control sees the exact response matrix. Target-only arms see
one, two, four, or eight nested orthogonal boundary drives. The orthogonal
design is a prospective amendment replacing an invalid Gaussian design whose
eight-field condition number varied from 15.6 to 2647.

The instrument advances to source-transfer measurement only if it represents
the full operator within 0.5%, reaches 1.5% with eight target fields, remains
above 2.5% with one field, improves materially by four and eight fields, and
improves monotonically across all four budgets.

## Execution layout

- `code/`: staged source
- `artifacts/formal/`: one capacity report and twelve target-only reports
- `artifacts/summary.json`: frozen reduction
- `sbatch_logs/`: merged scheduler log
- `STATUS_*` / `DONE_*`: terminal state

## Status

The formal matrix completed cleanly as HSG job `5700316`. The frozen verdict
is `reject_learning_instrument`. Geometric-mean held-out error was 118.2%,
110.5%, 110.7%, and 0.602% at one, two, four, and eight fields. The eight-field
threshold passed, but the curve was neither materially improving nor
monotone before full coverage. The formal capacity control also missed its
0.5% gate at 1.06%, despite a 0.424% local signal check.

All thirteen reports passed the finite-artifact audit, and an independent
local reduction reproduced `artifacts/summary.json` byte for byte. No
source-transfer result was run or inspected.
