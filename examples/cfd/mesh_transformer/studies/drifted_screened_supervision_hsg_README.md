# Supervision alignment for a reusable boundary kernel

## Scientific question

Does pointwise Green-kernel regression optimize the wrong error for a kernel
whose actual role is the composed boundary solution operator
`D_K A_K^{-1}`?

Prior work has trained Green-function models with PDE, boundary-integral, or
end-to-end solution losses. This study isolates the supervision choice while
holding the kernel representation, inputs, capacity, initialization, sampled
operators, geometries, pair supports, and update count fixed.

## Registered comparison

All arms use the learned-coefficient singular representation selected by the
preceding principal-part experiment:

- `pointwise`: normalized exact scaled-kernel error;
- `solution`: relative field error after differentiating through the learned
  trace solve;
- `hybrid`: the sum of the individually normalized losses.

Training uses boundary modes 0--3. Modes 5--8 are never used for optimization
and form a separate boundary-spectrum test. The common evaluation also covers
interpolation, unseen geometry, near-boundary queries, boundary refinement,
low/high screening, and stronger drift.

An aligned arm supports the claim only if it remains within `1.2x` of pointwise
training on interpolation, held-out boundary spectra, near-boundary fields,
and the 256-panel field endpoint, while reducing both field and exact-trace
error to at most `0.7x` on at least two of the three operator-parameter
extrapolations, in at least four of five paired seeds.

If solution-only gains on operator transfer but exceeds `1.5x` on held-out
boundary modes, the registered interpretation is boundary-distribution
overfit. If only hybrid passes, kernel identification and solution alignment
are complementary. If neither passes, the next experiment moves to
operator-normalized coordinates.

## Execution record

The formal protocol uses five seeds, 4,000 online training geometries per arm,
48 boundary panels, 64 field queries, and split-panel Gauss quadrature. Pilot
mode shortens training and evaluation without changing the model or training
supports.

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: fifteen registered reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Launch:

```bash
MODE=pilot sbatch --export=ALL,MODE=pilot drifted_screened_supervision_hsg.sbatch
MODE=formal sbatch --export=ALL,MODE=formal drifted_screened_supervision_hsg.sbatch
```

Each mode refuses to overwrite prior reports.

## Status

The registered five-seed comparison completed in an isolated output directory
as HSG job `5690714`. The independent reduction is byte-identical to the
frozen remote summary. Its verdict is
`supervision_mismatch_not_principal`: both aligned-loss arms improved the
interpolation guards, but neither passed even one of the three joint
field-and-trace operator-transfer criteria.

An underpowered 500-step pilot completed as HSG job `5690606`; it is not a
verdict.

The pilot showed that the arms separate in the intended way. At seed 17,
in-distribution field error was `0.239` for pointwise, `0.078` for
solution-only, and `0.070` for hybrid training. However, solution-only's exact
trace residual was `0.427` and its high-screening field error was `77.5`,
compared with `0.057` and `0.421` for the undertrained pointwise arm. The
hybrid learned coefficient `0.15920`, close to `1/(2*pi)`, while the
pointwise coefficient had reached only `0.10559` at this short budget.

Thus the pilot neither confirms aligned supervision nor licenses stopping:
solution-only can fit fields through a distorted effective trace, while the
hybrid may retain kernel identification. The registered 4,000-step,
five-seed factorial is required.

Two earlier formal submissions (`5690653` and `5690659`) were invalidated
without analysis after a duplicate submission created concurrent writers to
one output directory. Both were cancelled, and that directory is preserved
but excluded. No report from that directory contributes to the formal result.
