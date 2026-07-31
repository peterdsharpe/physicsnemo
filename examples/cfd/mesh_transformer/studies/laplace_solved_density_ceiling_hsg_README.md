# Laplace solved-density ceiling

## Scientific question

After removing the unnecessary learned query corrections, is the remaining
Laplace error caused mainly by inference of the boundary density or by the
finite polygonal boundary representation?

The parameter-free control solves the dense second-kind boundary equation on
the same 128-panel polygons and evaluates it with the same analytic panel
operator and frozen CUDA case banks as the completed five-seed factorial. The
preregistered decision rule is in the dated lab notebook.

## Task layout

- `code/`: the exact shared factorial source plus the registered control
- `artifacts/solved_density_ceiling.json`: the atomic result
- `sbatch_logs/`: merged Slurm log
- `STATUS_<job>` / `DONE_<job>`: terminal status and success marker

The run uses the previously validated HSG environment at
`agents/2026-07-17-paper-training-ab/.venv`.

## Relaunch

From the task directory:

```bash
sbatch laplace_solved_density_ceiling_hsg.sbatch
```

The launcher refuses to overwrite an existing result.

## Status

Completed on HSG on 2026-07-29 as job `5689064`. The preregistered verdict is
`split-dependent`: density inference dominates after refinement and on the
geometry stress tests, while finite boundary resolution dominates the
near-boundary and unseen-frequency errors. The accepted local bundle is
`results/laplace_solved_density_ceiling_2026-07-29_job5689064/`.
