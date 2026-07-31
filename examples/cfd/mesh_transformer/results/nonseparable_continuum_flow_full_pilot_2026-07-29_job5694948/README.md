# Full-budget continuum-flow pilot

This exploratory single-seed run checked whether the frozen training budget
made the continuum-flow comparison discriminating before spending five formal
seeds. It was not used to adjudicate either registered claim.

Cross-channel relative errors were:

| Arm | 8 layers | 4 layers | 16 layers | 32 layers | Fast, 8 | Fast, 16 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed rank four | 0.287 | 0.307 | 0.287 | 0.287 | 0.397 | 0.387 |
| Previous recurrence | 0.257 | 0.782 | 0.926 | 1.447 | 0.725 | 1.048 |
| Sorted continuum flow | 0.987 | 0.981 | 0.988 | 0.988 | 0.771 | 0.780 |
| Physical continuum flow | 0.275 | 0.310 | 0.269 | 0.269 | 0.853 | 0.661 |
| Per-case rank-four oracle | 0.119 | 0.121 | 0.119 | 0.119 | 0.121 | 0.126 |

The physical flow stayed essentially invariant when the same smooth profile
was sampled with 4--32 layers, while the previous recurrence deteriorated
monotonically under refinement. The flow did not transfer to faster
coefficient variation. The pilot therefore predicts a clean separation
between continuum consistency and frequency breadth, which justifies the
registered five-seed experiment.

The run completed on one four-GPU HSG node as Slurm job `5694948`. The six JSON
reports are the authoritative outputs; matching stdout logs are retained
beside them. The staged source fingerprint recorded by the reports is
`fb4fa62c0f89d7592dd5adf474d14a29b696d43a0e9fe2c9f486e09f879d8854`.

