# Full-budget frequency-coverage pilot

This exploratory single-seed run checked whether the frozen spectrum
intervention made the formal comparison discriminating. Cross-channel
relative errors were:

| Arm | Smooth | Covered fast | Unseen, 16 layers | Unseen, 32 layers |
|---|---:|---:|---:|---:|
| Narrow training | **0.262** | 0.678 | 0.748 | 0.566 |
| Broad training | 0.431 | 0.517 | 0.606 | 0.532 |
| Fixed rank four | 0.307 | **0.384** | **0.349** | **0.352** |
| Per-case oracle | 0.122 | 0.126 | 0.082 | 0.089 |

Broader support improved the covered-fast metric by 23.7%, close to but below
the registered 25% threshold. It simultaneously increased smooth-field error
by 64.7% and remained 1.5--1.7 times the fixed baseline on unseen
frequencies. The full five-seed test is therefore capable of refuting the
coverage claim rather than merely confirming that more spectral support
helps.

The run completed successfully on HSG as Slurm job `5695470`. It was
exploratory and is not counted toward either formal claim.

