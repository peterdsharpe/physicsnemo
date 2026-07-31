# Continuum-consistent rank-four latent flow

## Scientific question

Did the preceding recurrent surrogate fail on faster coefficient fields
because its update had no continuum meaning? The proposed replacement learns
a shared affine latent differential equation and integrates it with the
physical layer width.

## Registered comparison

All learned arms retain rank four, the exact uncoupled carrier, the same
training data, and nearly matched parameter counts. The comparison is:

- the previous physical-order gated recurrence;
- the path-length-scaled flow after sorting layers;
- the same flow in physical order;
- fixed rank-four truncation and the per-case rank-four oracle.

Training uses eight coefficient layers. Smooth fields are evaluated with four,
eight, sixteen, and thirty-two layers using matched underlying continuous
profiles. Fast fields are evaluated at eight and sixteen layers.

The continuum claim requires physical flow to stay within 20% of fixed
truncation at all three shifted resolutions and beat the old recurrence by
25% on both refinements. It must also beat sorted flow by 25% on every shifted
resolution, so resolution stability without physical order cannot pass. The
frequency claim separately requires 20% fixed noninferiority and 25% gains
over both recurrence and sorted flow on both fast splits. Each formal claim
will require four of five seeds; interpolation and oracle guards apply to
both.

## Pilot role

The reduced-budget, one-seed pilot checks that the flow is trainable and that
the comparison is discriminating. It will not be treated as evidence for
either registered claim.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: six pilot reports
- `artifacts/pilot_full/`: six full-budget pilot reports
- `artifacts/formal/`: eighteen formal reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

## Status

The reduced-budget pilot completed cleanly as HSG job `5694906`. The previous
recurrence's cross-channel error grew from 41% at eight layers to 106% at four,
129% at sixteen, and 170% at thirty-two. The physical flow stayed between 48%
and 50% across all four resolutions, the predicted continuum effect.

At 750 updates the flow remained underfit in distribution and retained
91%--95% error on fast fields, where the sorted control was slightly better.

The full-budget pilot then completed cleanly as HSG job `5694948`. The
physical flow matched fixed truncation on smooth fields and stayed at
27%--31% error from four through thirty-two layers; the old recurrence rose
from 26% at eight layers to 145% at thirty-two. On faster fields the physical
flow instead retained 66%--85% error, well above the 39%--40% fixed baseline.
This clean separation makes the formal comparison informative: it can confirm
continuum consistency without conflating it with coefficient-frequency
transfer.

The registered formal comparison completed cleanly as HSG job `5695143`.
Continuum transfer passed in all five seeds; frequency transfer passed in
none. The physical flow stayed near 27%--33% cross-channel error from four
through thirty-two layers while the old recurrence rose to 170% at
thirty-two. On fast fields, however, the physical flow remained at 84% and
68% error, versus 40% and 39% for fixed truncation.

All controls passed, and an independent local reduction reproduced the formal
summary byte for byte. A post-run rerun fix makes the reducer ignore its own
existing summary; it does not change the formal reports, gates, or verdict.
