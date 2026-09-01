# Laplace readout factorial

## Scientific question

Does the exact double-layer propagation core already carry the
boundary-to-interior solution on the mature two-dimensional Laplace suite, or
do the learned output gate and query-geometry contraction carry essential
accuracy?

The four matched cells are `pure`, `gate_only`, `contraction_only`, and
`full`. Five paired seeds use the same 3,000-update training protocol and
frozen 64-case evaluation bank. The preregistration and decision bars are in
the dated lab notebook.

## Task layout

- `code/`: exact staged source used by the run
- `laplace_readout_factorial_hsg*.sbatch`: smoke and full launchers
- `run_logs/`: one training log per arm and seed
- `artifacts/`: one atomic JSON report per arm and seed
- `sbatch_logs/`: merged Slurm logs
- `STATUS_<job>` / `DONE_<job>`: terminal status and success markers

The staged code is imported through `PYTHONPATH`. The Python environment is
the persistent, previously validated HSG aarch64 environment at
`agents/2026-07-17-paper-training-ab/.venv`; every result records a hash of
the staged source.

## Relaunch

From this directory:

```bash
sbatch laplace_readout_factorial_hsg_smoke.sbatch
sbatch laplace_readout_factorial_hsg.sbatch
```

The full job runs one arm per GPU and advances through five paired seed waves.
Do not combine its cells with archived full-arm runs; the paired initialization
and training stream are part of the experiment.

## Status

Completed on HSG on 2026-07-29. Smoke job `5688102` and full job `5688113`
both exited successfully; all 20 reports are present and the logs contain no
unexplained errors. The strict paired reducer classifies the pure readout as
`sufficient`; neither the gate nor the contraction is warranted. The
accepted local result bundle is
`results/laplace_readout_factorial_2026-07-29_job5688113/`.
