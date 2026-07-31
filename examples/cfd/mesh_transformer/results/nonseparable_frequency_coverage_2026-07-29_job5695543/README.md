# Formal coefficient-frequency coverage comparison

## Scientific verdict

Broader spectral support improved fast-field accuracy but did not earn the
coverage claim in any of five seeds:

`coverage_not_earned`

Geometric-mean cross-channel relative errors were:

| Arm | Smooth | Covered fast | Unseen, 16 layers | Unseen, 32 layers |
|---|---:|---:|---:|---:|
| Narrow training | **0.278** | 0.696 | 0.769 | 0.614 |
| Broad training | 0.403 | 0.503 | 0.590 | 0.489 |
| Fixed rank four | 0.313 | **0.379** | **0.355** | **0.360** |
| Per-case oracle | 0.122 | 0.128 | 0.085 | 0.091 |

Broad training reduced covered-fast error by 25.5%--33.1% in every seed, so
the intervention had its intended effect. It nevertheless remained 33% worse
than fixed truncation there and increased smooth-field error by 30%--74% in
every seed. On unseen frequencies it remained 36%--66% worse than fixed
truncation. Coverage and local-law transfer therefore each passed in zero of
five seeds.

The result is a trade-off, not a null intervention: the fixed-capacity flow
reallocated accuracy toward the newly represented spectra without learning a
single law that spans them. The registered decision is to retire this
rank-four learned local-flow branch rather than enlarge or retune it after the
failure.

## Provenance

The thirteen reports and frozen `summary.json` are the authoritative outputs
of HSG Slurm job `5695543`, which completed successfully in 6 minutes 25
seconds. Formal learned seeds were 29, 43, 59, 71, and 83; the formal
evaluation profiles were disjoint from both pilots.

The summary SHA-256 is
`1288212c09169b03598c3361e300c49162e628e5c6276daf18a6386adcd5d074`.
All reports record the same relevant-source fingerprint,
`6f7f7ff5ed8167b907db14e55d0ebcf073177e1ed2362a339fb75d0522352133`.
An independent local reduction was byte-identical.

