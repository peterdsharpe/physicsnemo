# MT3 preregistration skeleton (branched on the HiLift ladder readout)

*Status: PRE-RESULT. Written 2026-09-03 while the HiLift generalization ladder
(prereg 499bcfe8) is still training, so that the design decision is committed
to a decision rule before the numbers exist. Thresholds are in the units of the
readout they must explain (feedback memory `prereg-threshold-calibration`).*

## What this week fixed about the premise

After the critic review and its follow-ups, the MeshTransformer2 (MT2) claim
set that survives measurement is: exact SE(3) equivariance as a *contract*
(not an accuracy edge on vehicle data: canonicalized GeoTransolver matches it
at 1.01x, P1''); learning-rate robustness (R1); biased-density robustness at
1.20x with the similarity gauge (S1); in-family parity with GeoTransolver on
HiLift (0.98) and a 1.085x deficit on DrivAerML. Retired: similarity
equivariance (constant gauge, now fixed), measure completeness (fixed), the
drive-degree contract (vacuous), query-independence at acceptable cost (2.7-3.9x
in every construction, incl. resource-normalized), exact O(3) covariance of the
vector head (six constructions, all >= 12% wall shear or untrainable), and the
exact-kernel line on separated 3D aerodynamics (K0: oracle 4.8 vs 0.77).

Therefore MT3 is **not** "MT2 plus more contracts". The only open question that
justifies a new architecture is generalization, and the ladder is its test.

## Decision rule (commit before the ladder reads out)

Let r(n) = MT2/GT pressure rel-L2 on the shared HiLift val set at n training
cases, n in {35, 210, 1260}; let d_axis = (OOD-test error)/(in-dist val error)
for each architecture on the aoa and stall splits at the single sealed test
evaluation. Two seeds each; a difference smaller than the larger of the two
seed spreads is "not resolvable".

**Branch A — MT2 wins small data (r(35) <= 0.85, resolvable).** Structure buys
data-efficiency. MT3 = MT2 + similarity gauge as the frozen base, with the
kernel-decoder MeshTransformer's *interior* head grafted on for the
boundary->interior task (K-alt), targeted at data-efficiency claims. First
arm: MT3 vs GeoTransolver on DrivAerML volume at n in {14, 54, 218}.
Prediction: r(14) <= 0.8 on interior pressure. Falsifier: r(14) >= 0.95.

**Branch B — GeoTransolver degrades less on an OOD axis (d_GT <= 0.8 d_MT2 on
aoa or stall, resolvable).** The transfer-carrier hypothesis (GeoTransolver's
pointwise raw coordinate features) becomes the object of study. MT3's first
arm is a *mechanism discriminator*, not a product: MT2-gauge with a
deliberately non-equivariant raw-coordinate side channel (breaking the SE(3)
contract on purpose), 2 seeds, same split. Prediction: if the hypothesis is
right, d_MT2+raw moves at least halfway toward d_GT. Falsifier: no movement
(< 20% of the gap) -> the carrier is elsewhere (attention topology, loss
weighting), and the next discriminator is a GeoTransolver ablation removing
its raw coordinates.

**Branch C — flat (every r(n) within seed spread of 1.0 and no resolvable d
difference).** Architecture does not buy generalization on this dataset at
this power. MT3 is *not preregistered*; the program moves its generalization
question to the G5 corpus (single-factor twins, results/g5_*), and the MT2
deliverable is written up as: a robust SE(3)-equivariant soft-slice surrogate
at GeoTransolver parity, with measured density and learning-rate robustness
and a measured map of which structure buys which property at what cost.

**Branch D — mixed (A and B both true).** Run Branch B's discriminator first
(cheapest, most informative), then decide A.

## Fixed regardless of branch

- Statistical power: any new arm uses >= 3 seeds and reports a paired
  per-case statistic, not just means (critic C-power).
- The book's hero framing is rewritten from the ladder verdict, not before.
- The HiLift test split is evaluated exactly once, after all ladder decisions,
  and the evaluation configuration is committed before it runs.
