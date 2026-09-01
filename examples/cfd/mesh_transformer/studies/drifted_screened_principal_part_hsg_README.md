# Principal-part transfer under lower-order operator change

## Scientific question

When the principal elliptic operator is known but drift and screening are
learned from data, does fixing the universal double-layer singular limit
improve boundary-trace and field transfer over learning that limit?

The three arms share data, invariant inputs, network capacity, and training
budget. They differ only in whether the scaled kernel's coincident-point limit
is fixed, learned as one scalar, or learned with the complete kernel.

## Evidence and decisions

Five paired seeds are evaluated separately on interpolation, low/high
screening, high drift, unseen geometry, near-boundary queries, and boundary
refinement. The exact gauge-transformed kernel supplies both an oracle field
and an independent trace residual.

The fixed arm supports the principal-part claim only if:

- its in-distribution field error is no more than 1.2 times the better
  flexible arm;
- against each flexible arm, the geometric-mean field and exact-trace errors
  are at most 0.5 on both the near-boundary bank and the 256-panel
  in-distribution bank, with the fixed arm better in at least four of five
  paired seeds.

An operator-transfer claim additionally requires field and trace ratios at
most 0.7, with the same four-of-five direction, on at least two of low
screening, high screening, and high drift. Oracle mean field error must remain
below 0.02 and the exact trace quadrature check below 0.001 on every split.

## Task layout

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: fifteen registered arm/seed reports
- `sbatch_logs/`: merged training, evaluation, and GPU-utilization log
- `STATUS_*` / `DONE_*`: terminal state

The four GPU workers each run a disjoint sequential task list. The validated
HSG environment is
`agents/2026-07-17-paper-training-ab/.venv`.

## Launch

```bash
MODE=pilot sbatch --export=ALL,MODE=pilot drifted_screened_principal_part_hsg.sbatch
MODE=formal sbatch --export=ALL,MODE=formal drifted_screened_principal_part_hsg.sbatch
```

Each mode refuses to overwrite existing reports.

## Status

The underpowered pilot completed as HSG job `5689881`. It is not a verdict,
but it showed that the comparison is discriminating:

- near-singular scaled-kernel relative error was `0.0034` for the fixed arm,
  `0.0038` for the free-coefficient arm, and `0.149` for the fully learned
  arm at seed 17;
- the free arm learned coefficient `0.159240`, close to the analytic
  `1/(2*pi) = 0.159155`, and beat the fixed arm on several whole-solution
  pilot endpoints.

Thus the full study can both reject an unstructured kernel and refute the
stronger claim that the coefficient itself must be hard-fixed.

The formal five-seed factorial completed as HSG job `5689933`. All fifteen
registered reports passed schema and numerical-sanity checks, and a fresh
local reduction reproduced the archived summary byte for byte. The registered
verdict is `hard_fixed_coefficient_not_earned`.

The scientific result is two-part:

- the unstructured scaled-kernel arm is much worse in interpolation,
  near-boundary evaluation, and four of five transfer splits, supporting
  explicit singular-form encoding;
- the free-coefficient arm learns a coefficient within `0.41%` of
  `1 / (2*pi)` in every seed and remains too close to the fixed arm for hard
  fixation to pass either the near-boundary, fine-resolution, or
  operator-transfer criteria.

The high-screening endpoint remains poor for every arm and is not evidence
that the unstructured model transfers: it has the best field error there but a
worse exact-trace residual. The next experiment should test supervision of the
smooth remainder through the downstream trace solve rather than further
hard-coding the principal coefficient.
