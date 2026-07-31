# Path-ordered learned local generator

## Scientific question

Did the preceding variable-coefficient model fail because it pooled ordered
coefficient samples before prediction, rather than composing their local
propagation maps in physical order?

## Registered comparison

The five arms are:

- the preceding scalar optical correction;
- the preceding mean-pooled ordered raw model;
- a learned local generator composed after sorting layers by coefficient;
- the same learned generator composed in physical order; and
- the exact analytic path product as a parameter-free oracle.

The learned generator infers a positive local potential from each coefficient
sample. It is given the known transverse Fourier contribution, but not the
exact local screening law. The sorted and physical-order arms have identical
parameters and differ only in composition order.

The physical-order model earns the composition claim only if it recovers at
least 80% of the matched-pair contrast, improves paired field error at least
fivefold over the scalar model and twofold over the pooled model, and
preserves interpolation and held-mode accuracy. A separate local-law transfer
claim requires at least 30% improvement over the pooled model at both
coefficient extremes.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: twenty-five registered arm-by-seed reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Each mode refuses to overwrite prior reports.

## Status

The registered 4,000-step, five-seed comparison completed cleanly. The frozen
verdict is `composition_earned_local_law_not_transferable`: the physical-order
generator passed every composition gate in all five seeds, while the separate
high-coefficient transfer gate failed. The analytic oracle was exact and the
sorted control predicted exactly zero layer-order contrast. An independent
local reduction reproduced the formal summary byte for byte, and the replayed
scalar and pooled controls exactly matched the preceding experiment.

The earlier six-report, 500-step pilot remains a signal check only and was not
used to update the registered claims.
