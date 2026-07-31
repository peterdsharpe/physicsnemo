# Phase-1 fixed-carrier convergence on AGA

This CPU-only continuation runs one more uniform in-plane refinement of the
frozen subdivision-3 sphere carrier. Level 4 produced only one successive
represented-measure shape change below the predeclared 0.5% screen; level 5
is needed to decide the required second successive transition. The field,
force, and moment drift threshold remains 2%.

The task directory contains a byte-for-byte copy of the local `physicsnemo`
package and the four study modules used by the run. It also contains the exact
156-face x86-generated representation as a portable NPZ. Freezing that object
is necessary because PyACVD produced a different remesh on AGA's ARM stack;
the first level-5 attempt correctly failed the level-0 map-identity gate. The
wrapper verifies the NPZ SHA-256 before execution, records code hashes and the
base Git SHA, and uses the existing AGA Python environment only for
third-party dependencies.

Expected layout:

```text
code/
  physicsnemo/
  examples/cfd/mesh_transformer/
    provenance.py
    studies/{common_surface_transfer,phase1_common_surface,
             phase1_fixed_carrier_convergence}.py
artifacts/
sbatch_logs/
phase1_representation_156faces_2026-07-27.npz
phase1_fixed_carrier_convergence_aga.sbatch
```

Submit from the self-contained task directory:

```bash
sbatch phase1_fixed_carrier_convergence_aga.sbatch
```

The job requests 16 CPU cores, 96 GiB, and two hours on AGA `cpu-short`. It
writes a JSON artifact and SHA-256 sidecar, a merged Slurm log, and an
exit-status marker even when the payload fails.
