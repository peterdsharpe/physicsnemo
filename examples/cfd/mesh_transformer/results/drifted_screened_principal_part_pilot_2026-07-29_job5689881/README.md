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

The fixed arm earns the principal-part claim only if:

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

Preregistered. The underpowered pilot is ready to launch; the five-seed
experiment remains gated on a discriminating pilot and a frozen reducer.
