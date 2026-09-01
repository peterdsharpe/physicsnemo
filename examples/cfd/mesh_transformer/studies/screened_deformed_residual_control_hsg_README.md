# Residual-controlled screened solver on deformed geometries

## Scientific question

Does one unconditioned, residual-controlled second-kind density processor span
screening variation and smooth geometry deformation, or does geometry require a
conditioned preconditioner?

Exact modified-Bessel fields provide labels on Fourier-deformed domains. The
candidate applies unit residual updates to the canonical double-layer trace
until `1e-6` relative residual; the oracle solves the same trace densely.

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
sbatch screened_deformed_residual_control_hsg.sbatch
```

The launcher refuses to overwrite an existing result.

## Status

Preregistered and ready to launch.
