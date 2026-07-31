# Fixed canonical screened double layer

## Scientific question

Can the screened parameter live entirely inside the known double-layer kernel,
while one shared eight-step density processor spans interpolation, both
screening extrapolation directions, and unseen boundary modes?

The study compares a dense discrete trace solve with eight untrained
unit-relaxation residual steps. Both use the same correctly normalized
double-layer kernel, fixed jump, exact Bessel targets, and frozen evaluation
and resolution banks.

## Task layout

- `code/`: exact staged source
- `artifacts/report.json`: atomic result and preregistered decision
- `sbatch_logs/`: merged Slurm log with split-level heartbeats
- `STATUS_<job>` / `DONE_<job>`: terminal status and success marker

The run uses the validated HSG environment at
`agents/2026-07-17-paper-training-ab/.venv`.

## Relaunch

From the task directory:

```bash
sbatch screened_canonical_double_layer_hsg.sbatch
```

The launcher refuses to overwrite an existing result.

## Status

Completed on HSG on 2026-07-29 as job `5689394`. The fixed eight-step
processor passed every field, trace, dense-agreement, and quadrature bar, but
failed the preregistered monotone-refinement requirement on three splits: its
roughly `1e-3` truncation floor becomes visible after boundary refinement. The
accepted local bundle is
`results/screened_canonical_double_layer_2026-07-29_job5689394/`.
