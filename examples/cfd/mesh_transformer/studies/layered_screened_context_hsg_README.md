# Ordered coefficient context in a layered elliptic operator

## Scientific question

Does a variable-coefficient boundary-to-interior map require the coefficient
field in spatial order, or can optical-depth summaries carry the relevant
operator variation?

The periodic-strip equation separates into exact one-dimensional Fourier-mode
problems. Constant-layer transfer matrices do not commute, so a registered
pair challenge holds optical summaries fixed while changing only layer order.

## Registered comparison

The four arms are:

- a parameter-free optical carrier;
- a capacity-matched scalar correction that sees only order-blind summaries;
- the carrier plus a position-aware coefficient encoder; and
- the same ordered encoder on a raw boundary-preserving base.

Five seeds train on modes 0--3 and layered coefficients in `[0.25, 3.0]`.
Modes 4--7, low and high coefficient ranges, and matched optical-depth layer
permutations are held out.

Ordered context supports the nonlocality claim only if it recovers at least 80%
of the paired contrast, reduces paired field error by at least fivefold
relative to the scalar arm, and preserves interpolation and held-mode
accuracy. The carrier supports a separate representation claim only if it
improves at least two of the low-coefficient, high-coefficient, and layer-order
tests by at least 30% without losing the guards.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: twenty registered arm-by-seed reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Each mode refuses to overwrite prior reports.

## Status

The reference and reducer pass their focused local tests. The five-report,
500-step pilot completed cleanly and confirmed a 4.0% matched-pair response
contrast with exactly identical scalar inputs. The ordered models had not
learned that contrast at the pilot budget, so the registered 4,000-step formal
comparison remains necessary. No pilot metric has been used to update the
registered scientific claims.

The twenty-report formal comparison then completed successfully. Its verdict
is `feedforward_context_not_sufficient`: neither ordered arm recovered 80% of
the layer-order contrast or achieved the required fivefold paired-error
reduction, and the ordered optical carrier failed its interpolation guard.
The frozen result is in
`results/layered_screened_context_2026-07-29_job5692619/`.
