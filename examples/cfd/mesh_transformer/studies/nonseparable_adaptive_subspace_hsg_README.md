# Dynamic realizability of the rank-four oracle

## Scientific question

The static per-query rank-four oracle is accurate, but its factors are chosen
after seeing the full solution. Does a natural rank-four subspace remain
accurate when it must be propagated through the coefficient field?

## Registered comparison

At every layer, the four lowest eigenvectors of the local coefficient matrix
define a moving reduced basis. Two propagators isolate the coordinate-change
mechanism:

- local naive: reuse reduced coordinates when the basis rotates;
- local connected: project coordinates through the exact overlap of
  consecutive bases.

Both solve the reduced two-point boundary problem and predict only the coupled
correction above the exact diagonal carrier. Fixed first-mode truncation, one
global eigenspace per profile, and the static per-query oracle are controls.

The connected propagator must stay within 20% of fixed truncation on
low-frequency fields, beat fixed, global, and naive propagation by 25% on all
faster splits, and remain stable from sixteen to thirty-two layers at the
highest frequencies. Every gate must pass in four independent profile
batches; the oracle must remain below 15%.

## Pilot role

The reduced pilot checks the moving-basis boundary solve, numerical stability,
and whether the comparison has enough separation to justify the four formal
batches. It is not evidence for the registered claim.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: reduced pilot report
- `artifacts/formal/`: four formal reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

## Status

The implementation passed seven focused local tests. A reduced pilot completed
cleanly as HSG job `5695856`; it showed that exact basis connections repair a
large naive moving-basis error but return almost exactly to fixed/global
truncation.

The registered four-batch census completed cleanly as HSG job `5695916`.
Adaptive-subspace transfer passed in zero batches. Connection-aware error was
0.314, 0.390, 0.356, and 0.360 across the four splits, versus 0.313, 0.388,
0.353, and 0.357 for fixed truncation. All oracle controls passed, and an
independent local reduction reproduced the formal summary byte for byte.

The natural moving rank-four subspace is not earned; the controlled low-rank
branch is closed under the registered decision.
