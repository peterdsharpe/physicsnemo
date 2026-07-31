# Nonseparable low-rank ceiling

## Scientific question

After exact propagation of the uncoupled transverse modes, is the response
caused by a nonseparable coefficient field low rank enough to justify a
rank-four path-composed surrogate?

The aggregate operator norm is not the deciding metric: the uncoupled carrier
already dominates it. The experiment therefore measures the heterogeneous
cross-channel response directly.

## Registered decision

Four independent samples each contain 256 coefficient profiles, 32 interior
query locations, six transverse channels, and three heterogeneity levels.
Rank four advances to a learned comparison only if every sample satisfies:

- best-case rank-four cross-channel error is at most 15% at every level;
- best-case rank-three error remains above 20% at every level;
- fixed spectral and global-average rank-four subspaces are each at least
  twice as inaccurate as the rank-four oracle at the strongest level; and
- cross-channel response is at least 0.4% of the full operator norm there.

The first two gates establish that rank four is both sufficient and necessary
at the chosen fidelity. The baseline gap establishes room for an adaptive
latent state. The final gate prevents a numerically impressive ratio on a
negligible physical effect.

## Scope

The oracle chooses its rank-four factors separately for every profile and
query, so passage proves possibility, not learnability or consistent dynamics.
The next experiment must compare a path-composed state with a profile-global
learned compression at equal rank.

## Execution layout

- `code/`: staged source
- `artifacts/formal/`: four independent census reports
- `artifacts/summary.json`: frozen registered reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

The job refuses to overwrite any existing report.

## Status

The formal comparison completed cleanly as HSG job `5694242`. The frozen
verdict is `rank_four_ceiling_earned`: every gate passed in all four
independent samples. Rank-four oracle cross-channel error was 12.1%--12.4%;
rank three remained at 23.5%--24.5%; and both registered rank-four baselines
were about 2.6 times worse at the strongest heterogeneity.

The copied result is
`results/nonseparable_rank_census_2026-07-29_job5694242/`. Its summary
SHA-256 is
`7ae875f2d95fd2bc3b6ea72dea679b237bc35d32cb5484dc765f7b920af2c373`.
An independent local reduction reproduced it byte for byte.
