# Coefficient-frequency coverage versus local-law transfer

## Scientific question

Does the continuum flow fail on fast coefficient fields because those spectra
were absent from training, or because the learned low-rank dynamics is not a
transferable local law?

## Registered comparison

Two identical physical continuum flows train at sixteen layers with the same
data count and optimization budget:

- narrow support: coefficient frequencies one and two;
- broad support: coefficient frequencies one through four.

They are evaluated on smooth frequencies one and two, covered fast
frequencies three and four, and unseen frequencies five and six. The unseen
test is repeated at thirty-two layers. Fixed rank-four truncation and the
per-case rank-four oracle are unchanged references.

Coverage requires the broad arm to remain within 20% of fixed truncation on
smooth and covered-fast fields, improve on the narrow arm by 25% on
covered-fast fields, and retain smooth-field performance within 20%.
Transferable local-law evidence additionally requires the same fixed and
narrow-arm gates on unseen frequencies at both resolutions, with no more than
20% resolution drift. Each claim requires four of five formal seeds, and the
oracle must remain below 15%.

## Pilot role

The reduced-budget single-seed pilot checks only that sixteen-layer training
is numerically healthy and that broader support moves the covered-fast metric
enough to make the formal comparison informative.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: reduced-budget reports
- `artifacts/pilot_full/`: full-budget single-seed reports
- `artifacts/formal/`: formal reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

## Status

The implementation passed seven focused local tests. A reduced-budget pilot
completed cleanly as HSG job `5695362`; both learned arms were underfit, but
broader support moved unseen-frequency error in the predicted direction.

A full-budget single-seed pilot then completed as HSG job `5695470`. Broad
training improved covered-fast error from 0.678 to 0.517, but increased
smooth-field error from 0.262 to 0.431 and remained at 0.606 and 0.532 on
unseen frequencies, versus 0.349 and 0.352 for fixed truncation. The
comparison is therefore discriminating.

The registered formal comparison completed cleanly as HSG job `5695543`.
Coverage and transferable-local-law claims each passed in zero of five seeds.
Broad training reduced covered-fast error from 0.696 to 0.503, but raised
smooth-field error from 0.278 to 0.403 and remained above the fixed baseline
on every shifted split.

All controls passed, and an independent local reduction reproduced the formal
summary byte for byte. The rank-four learned local-flow branch is closed under
the registered decision rule.
