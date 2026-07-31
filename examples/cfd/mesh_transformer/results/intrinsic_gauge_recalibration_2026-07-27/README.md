# Train-only intrinsic-gauge recalibration — 2026-07-27

This task reruns `studies/calibrate_intrinsic_gauge.py` on the exact AGA
DrivAerML manifest after repairing the script to iterate the train sampler
rather than all dataset entries. It is CPU-only and reads the dataset without
modifying it.

The isolated remote task directory is:

`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-27-mt-intrinsic-gauge-calibration`

The task snapshot contains the local `physicsnemo/` package and unified recipe
at the current dirty research state. `aga_dataset_paths.yaml` is copied over
the snapshot's placeholder `datasets/dataset_paths.yaml`; the canonical local
and remote repositories are not changed.

Job `302536` completed on `cpu/cpu-short` with exit code `0:0` in 7m35s
(maximum resident set 32,979,212 KiB). It consumed exactly 435 unique training
IDs and matched the frozen manifest set hash.

- Corrected \(C\): `26.48435695783964`
- Historical contaminated \(C\): `26.476592786355283`
- Relative change: `+0.0293247%`
- Result SHA-256:
  `cf3ebb2c34f291642693f595676370a755406b05723075b2d5f58a1c831f5c1d`
- Log SHA-256:
  `8ee4bce39ad82f56218dc0139fc0c18bf49e94ea93915b90efb68bb046d12180`
- Frozen manifest SHA-256:
  `60775de5708dc53276b5d437306787339afe16b32b9c9bccdf6668a2814ecc52`

No Slurm job from this task remains active.
