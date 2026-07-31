# Rank-four latent composition on a nonseparable field

## Scientific question

Can one consistent rank-four latent state approach the per-case low-rank
ceiling for heterogeneous mode coupling, and does updating that state in
physical order outperform a profile-global compression at the same rank?

## Exploratory pilot

The pilot compares:

- a profile-global full-rank correction;
- the same global encoder with a rank-four output;
- a rank-four recurrent state after sorting the coefficient layers;
- the same state updated in physical order;
- the exact uncoupled carrier, a fixed rank-four spectral truncation, and the
  unattainable per-case rank-four oracle.

All learned arms predict only the heterogeneous correction on top of exact
uncoupled modal propagation. The correction vanishes exactly for zero coupling
and at both boundaries. The primary outcomes are residual-relative and
cross-channel errors; aggregate operator error is diagnostic only.

The pilot uses one seed and a reduced training/evaluation budget. It estimates
optimization viability and effect size; it cannot confirm the architecture
claim. Formal gates will be frozen only after the pilot is reduced.

## Registered formal comparison

The formal comparison uses model seeds 29, 43, 59, 71, and 83 plus coefficient
and layer-swap samples not used in either pilot.

The ordered-compression claim passes only if at least four seeds satisfy all
of the following:

- path rank four is at least 25% better than global rank four on both
  in-distribution and high-heterogeneity cross-channel error;
- it is within 20% of fixed rank-four truncation on both splits; and
- it recovers at least 50% of the heterogeneous layer-swap response.

The breadth claim is adjudicated separately. On both faster spatial variation
and the combined fast/high shift, path rank four must close at least half of
the global-to-oracle gap and remain within 20% of fixed truncation, again in
at least four seeds. The per-case oracle must stay below 15% error on every
split, and every sorted model must predict zero layer-swap contrast.

The frozen reducer reports `ordered_compression_earned_breadth_refuted` when
the first claim passes and the second does not; a local ordering win is not
allowed to masquerade as transferable path dynamics.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: seven pilot reports
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

The job refuses to overwrite prior pilot reports.

## Status

The reduced-budget exploratory pilot completed cleanly as HSG job `5694534`.
Physical ordering improved in-distribution cross-channel error from 60% for
the global rank-four model to 42%, and recovered 44% of the layer-swap
contrast versus 25% for global compression and zero for the sorted control.
It nevertheless trailed the fixed rank-four truncation (29%) and retained
91%--94% error under the two faster-variation shifts.

Because the pilot used only 750 of the planned 4,000 updates and one eighth of
the planned training data, those failures may be optimization-limited. A
single-seed full-budget check completed as job `5694598`. Physical-order
cross-channel error was 27% in distribution and 30% at high heterogeneity,
matching fixed truncation and improving global rank four by 28% and 36%.
Under faster variation it remained 65%--68%, about 70% worse than fixed
truncation. The ordering signal is real enough for the registered five-seed
test, while the breadth criterion is intentionally still in jeopardy.

The formal comparison completed cleanly as HSG job `5694690`. The frozen
verdict is `ordered_compression_earned_breadth_refuted`: ordered compression
passed in four of five seeds and breadth passed in zero. The copied result is
`results/nonseparable_latent_composition_2026-07-29_job5694690/`; its summary
SHA-256 is
`3ba76ef9dcc1bc74a51f3492575d22a2495a8a4e70d1a414be367177b7fa52ea`.
An independent local reduction reproduced it byte for byte.
