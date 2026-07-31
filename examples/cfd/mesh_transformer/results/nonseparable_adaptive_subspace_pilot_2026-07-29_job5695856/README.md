# Moving-subspace realizability pilot

This reduced analytic pilot checked the moving-basis boundary solve before the
registered four-batch census. Cross-channel relative errors were:

| Propagator | Low frequency | Mid frequency | High, 16 layers | High, 32 layers |
|---|---:|---:|---:|---:|
| Fixed rank four | 0.305 | 0.374 | 0.341 | 0.345 |
| Global eigenspace | 0.305 | 0.374 | 0.341 | 0.346 |
| Local basis, no connection | 0.379 | 0.779 | 0.950 | 0.964 |
| Local basis, exact connection | 0.305 | 0.376 | 0.343 | 0.348 |
| Static per-query oracle | 0.121 | 0.123 | 0.085 | 0.090 |

The connection term was essential for numerical correctness, but the corrected
moving basis reproduced rather than improved on fixed/global truncation. The
run completed successfully on HSG as Slurm job `5695856`; it was exploratory
and did not count toward the formal claim.

