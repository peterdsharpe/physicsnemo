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

## Addendum 2026-09-05 — Branch V (boundary→interior), written before the full V0 readout

The V0 pilot (notebook `#sec-nb-v0-pilot`) puts MT2's passive interior decode
2.9–4.0x behind GeoTransolver's token-level query interaction at 50 epochs,
and the exact-kernel decoder 6.2x behind. If the 500-epoch runs hold within
those bands, the useful-architecture question for the restated goal is:

> How can interior queries receive interaction (the thing GeoTransolver's
> queries-as-tokens gives) while remaining exactly query-independent (the
> thing a deployable surrogate needs: predictions at one point cannot depend
> on which other points were asked)?

**Candidate mechanism — equivariant latent volume tokens (LVT).** Add a fixed
set of K latent tokens whose *positions* are constructed equivariantly from
the boundary alone (so they are query-independent by construction): for each
slice anchor (position z_s, mean normal m_s, RMS radius rho_s of its assigned
boundary points) place tokens at z_s + c_j rho_s m_s for a few fixed offsets
c_j ∈ {0.5, 1, 2} (in gauge units, along the anchor normal into the fluid),
plus one token at the domain centroid. These tokens join the encoder as
ordinary interacting tokens (they read and write, like boundary tokens), so
the slice states after the encoder carry *off-surface* context. Interior
queries then decode passively from slices + boundary tokens + LVT states
(the existing `_ReadBlock` path with the LVT set appended to the source),
still never writing. Contracts preserved exactly: SE(3) covariance (token
positions are covariant constructions), measure completeness (LVT carry
zero measure weight in boundary sums; a separate learned weight for the
latent set), query independence (LVT depend only on the source).

**Decision rule for launching Branch V** (after full V0 lands):
- If full V0 gives MT2-interior/GT ≥ 2.0 (interaction gap confirmed) → launch
  LVT with K = 3 × n_slices + 1, 2 seeds, same protocol, pilot-gated.
  Prediction in readout units: MT2-interior-LVT / GT ≤ 1.5 on interior
  pressure (halving the gap) with exact query independence retained
  (allclose 1e-12 at shared queries). Falsifier: ratio ≥ 2.0 → interaction
  *at the query itself* is required; the honest product is then
  "GeoTransolver for interior accuracy, MT2 for contracts", and the next
  candidate is a hybrid decode (a small query-side self-attention over a
  fixed lattice of probe points that is re-computed per request but never
  depends on the request's other queries — i.e. interaction with a canonical
  probe set, not with the batch).
- If full V0 gives MT2-interior/GT ≤ 1.5 (the pilot gap closed with training)
  → Branch V is unnecessary; report the 500-epoch parity and proceed to the
  interior G-ledger axes (query-resolution transfer, SDF-band error,
  surface-trace consistency).
- The kernel-decoder arm's fate is decided by the far-band readout
  (SDF ≥ 0.5 L_ref) as preregistered in `#sec-nb-v0-prereg`.
