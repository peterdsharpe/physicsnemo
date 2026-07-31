# Nonseparable rank census — formal result

## Scientific verdict

The frozen verdict is `rank_four_ceiling_earned`. All registered gates passed
in all four independent profile samples.

After exact uncoupled modal propagation, a per-case best rank-four correction
left 12.1%--12.4% cross-channel error across the three heterogeneity levels.
Rank three remained at 23.5%--24.5%. At the strongest heterogeneity, fixed
spectral and global-average rank-four subspaces both left about 31.2% error,
2.56--2.59 times the rank-four oracle error in every replicate.

The coupling response was only 0.50%--0.54% of the aggregate operator norm at
the strongest level. This confirms why aggregate operator error is not a
valid fidelity metric for the next experiment: the diagonal carrier can look
nearly exact while missing all heterogeneous coupling.

This result qualifies a rank-four learned discriminator. It does not show that
a consistent learned state can attain the per-profile, per-query oracle.

## Provenance

- HSG Slurm job: `5694242`
- Terminal state: `COMPLETED`, exit code `0:0`, elapsed time `00:00:48`
- Formal reports: four seeds, 256 profiles × 32 query locations × three
  heterogeneity levels each
- Source fingerprint recorded by every report:
  `4ff303b01cf6ff05f29d3c17f9f38dfdf098be067d152ca2d383a9b08cf2dcd5`
- Summary SHA-256:
  `7ae875f2d95fd2bc3b6ea72dea679b237bc35d32cb5484dc765f7b920af2c373`

An independent local invocation of the frozen reducer reproduced
`summary.json` byte for byte.

## Contents

- `formal/seed*.json`: independent census reports
- `formal/seed*.stdout.log`: full per-report console output
- `summary.json`: frozen registered reduction
- `nonseparable-rank.log`: merged batch log
- `STATUS_5694242` / `DONE_5694242`: terminal markers
