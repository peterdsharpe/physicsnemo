# Reduced-budget frequency-coverage pilot

This single-seed pilot tested numerical health only. After 750 updates,
cross-channel relative errors were:

| Arm | Smooth | Covered fast | Unseen, 16 layers | Unseen, 32 layers |
|---|---:|---:|---:|---:|
| Narrow training | 0.503 | 0.844 | 0.827 | 0.796 |
| Broad training | 0.763 | 0.732 | 0.608 | 0.613 |
| Fixed rank four | 0.298 | 0.395 | 0.357 | 0.358 |
| Per-case oracle | 0.125 | 0.120 | 0.082 | 0.087 |

Both learned arms were underfit, so these values were not used to assess the
registered claims. The run completed successfully on HSG as Slurm job
`5695362`.

