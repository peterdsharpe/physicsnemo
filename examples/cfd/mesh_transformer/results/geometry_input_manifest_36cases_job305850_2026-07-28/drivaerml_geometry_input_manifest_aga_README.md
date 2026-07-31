# DrivAerML target-free geometry input freeze

This CPU-only AGA task freezes the exact raw geometry inputs needed by the
36-case, five-resolution canonical-geometry validity study before any of its
32 new cases produce model output.

The producer follows a closed path allowlist. For each frozen case it hashes
only the vehicle `points.memmap`, `cells.memmap`, structural TensorDict
metadata, and the five case-global physical inputs `U_inf`, `p_inf`, `rho_inf`,
`nu`, and `L_ref`. It records the case symlink target and validates the frozen
reader index, cell count, cyclic start, point/cell metadata, file sizes, and
exact `L_ref=5`. It never traverses or opens `point_data`, `cell_data`,
`interior`, pressure, or wall-shear-stress files.

The task is an input-provenance measurement, not model inference. It publishes
one JSON artifact plus SHA-256 sidecar with atomic no-clobber hard links and
rolls back its half of the pair if either destination collides. It refuses
existing files or dangling links. `DONE_<jobid>` is written only after the
sidecar verifies; `STATUS_<jobid>` is written on every normal shell exit after
Slurm starts the wrapper.

The task must be staged at
`/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1`
with real `artifacts/` and `sbatch_logs/` directories created before
submission. Submit from that exact directory:

```bash
cd /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1
env -i \
  HOME=/home/psharpe \
  USER=psharpe \
  LOGNAME=psharpe \
  PATH=/usr/bin:/bin:/cm/local/apps/slurm/current/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  SLURM_CONF=/cm/shared/apps/slurm/etc/oci-aga-slurm-1/slurm.conf \
  /cm/local/apps/slurm/current/bin/sbatch \
  --export=ALL drivaerml_geometry_input_manifest_aga.sbatch
```

The wrapper rejects any other submission directory. The task uses AGA's
CPU-only `cpu` partition and `cpu-short` QoS. Slurm records this Lustre
directory using its physical `/scratch/fsw/...` spelling; the wrapper
canonicalizes `SLURM_SUBMIT_DIR` and accepts that exact physical path while
retaining the logical `/home/...` spelling for artifact paths.
