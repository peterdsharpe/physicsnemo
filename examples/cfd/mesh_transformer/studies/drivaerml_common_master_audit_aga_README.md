# DrivAerML full-master reconstruction audit

This CPU-only AGA study compares two 10,000-cell P0 representations on
`run_1` (training) and `run_118` (validation):

- the production-form cyclic sparse support with ambient nearest-centroid
  reconstruction; and
- a deterministic normal-aware centroidal cover at the same budget, using the
  preregistered cost
  `||x-s||² + lambda² ||n-m||²` with
  `lambda = sqrt(full-master area / k)`.

Each case's complete curated CFD vehicle boundary is its own frozen integration
master. The study never transfers between cars and does not claim that the
sparse triangles form a remesh. It reads the four TensorDict memmaps directly,
without importing PhysicsNeMo or copying the dataset.

For each representation, the JSON reports exact all-cell fill-distance
statistics (including the deciding area-weighted q95), negative-normal
assignment area, area-weighted Cp/WSS P0 projection floors, pressure
force/moment error, and raw-versus-tangent-projected WSS diagnostics. It also
constructs the implicit common-master cross measure and gates constants,
`RP=I`, the Pythagorean projection identity, vector-integral preservation, and
mass adjointness below `1e-12`. Input arrays, metadata, support selections,
assignments, the operator stream, script, command, runtime, and software
versions are hashed or recorded for provenance.

Run the dependency-bounded synthetic smoke test locally with:

```bash
python3 drivaerml_common_master_audit.py \
  --synthetic-smoke \
  --output /tmp/drivaerml_common_master_smoke.json
```

The local smoke path uses SciPy when available and a bounded NumPy neighbor
search otherwise. The production job uses the task-local `.venv`, whose SciPy
cKDTree handles the roughly 17 million master cells per case.

The 10k engineering screen freezes one production-form start per case. Both
were drawn before execution with a Torch generator seed of `20270727`
(`20260727 + k`) and are passed as explicit CLI values:

| Case | Start | Seed |
|---|---:|---:|
| `run_1` | 13499736 | 20270727 |
| `run_118` | 14705232 | 20270727 |

After the first 10k artifact exposed that restriction-time empty-cell repair
was performing an undeclared full Lloyd update, the corrected `run_118`
integration sanity is launched with:

```bash
sbatch drivaerml_common_master_repair_sanity_aga.sbatch
```

It reuses the frozen 10k start and source, writes a distinct `repairfix`
artifact, and must show that any restriction-time repair moves only empty
supports. It has no effect-size gate and cannot decide H4.

The CLI accepts repeated `--cyclic-replicate CASE START SEED` arguments. The
deciding 40k run uses all four frozen draws on each case without changing the
implementation:

```text
run_1:   11914080, 11878329, 14546186, 2529054
run_118:  2517171,   651819,  4220546, 6617334
seed: 20300727 (20260727 + 40000)
```

Launch the production audit from this directory with:

```bash
sbatch drivaerml_common_master_audit_aga.sbatch
```

The wrapper requests 16 CPU cores, 96 GiB, and two hours in `cpu-short`. It
writes one JSON and SHA-256 sidecar per completed case under `artifacts/`, so
a later-case failure cannot erase an earlier result. It also writes a merged
log under `sbatch_logs/` and `STATUS_<job-id>`. It does not modify the dataset.
On resubmission it reuses only outputs whose sidecars verify and refuses to
overwrite an unverified artifact. The job is intentionally not submitted by
adding these files.

Launch the preregistered deciding audit as a two-task array with:

```bash
sbatch drivaerml_common_master_audit_aga_k40000.sbatch
```

Array task 0 runs `run_1`; task 1 runs `run_118`. Each task constructs the
cover once, evaluates the four explicit cyclic starts above, writes an
independent JSON and SHA-256 sidecar, and records `COMPLETED_UNITS=1/1`,
`STATUS_<array-job>_<task>`, and `DONE_<array-job>_<task>`. The wrapper
redirects `srun` stdin from `/dev/null`, preventing the input-consumption bug
found in the first 10k wrapper. A deciding output is never reused or
overwritten: after a partial array failure, verify the completed artifact and
resubmit only the missing array index. An atomic per-output claim prevents
duplicate submissions from racing into the same path, and a task cannot
report exit zero unless its completed-unit count is exactly one.

## Run history

- Job 303310 completed only the first 10k case because `srun` consumed the
  loop's stdin; its `run_1` output is valid, but the job is not a two-case
  completion.
- Job 303327 reused that verified `run_1` artifact and completed the original
  `run_118` 10k engineering output.
- Array job 303349 was cancelled after 1 minute 42 seconds, before any 40k
  artifact was written. An audit found that restriction-time empty-cell
  repair could perform an undeclared full Lloyd update. The old 10k
  `run_118` artifact exercised that bug and is superseded as a controlled
  two-iteration cover comparison.
- Corrected audit source SHA-256, frozen before the repair sanity run:
  `44a95610f844daba0b3367809fc061e1edbb2c334f37d46fc4ca32b21258aa77`.
- Job 303463 completed the corrected `run_118` 10k sanity in 3 minutes
  26 seconds. It exercised one restriction repair with
  `repair_scope=empty_supports_only`, retained zero final empty cells, and
  passed the algebra gate at `2.92e-15`. Artifact SHA-256:
  `e9073da9735421fe30f66d2307e1c88ec3e9585c5539cc46eb0248ad862866a9`.
