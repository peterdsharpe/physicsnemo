# Preregistration draft: G5 — single-factor generalization on the cfd-datagen corpus

*Draft written 2026-09-03 by the mesh-transformer program fork, for import into `examples/cfd/mesh_transformer/results/` (as JSON) and the lab notebook once the cfd-datagen launch plan is fixed. Status: PRE-DATA. Every threshold below is stated in the units of the readout it must explain, per the N1/L1 lesson (feedback memory `prereg-threshold-calibration`).*

## Question

Does any architectural structure buy generalization on 3D RANS labels along a *single controlled factor*, when the factor is either (a) a symmetry the architecture carries exactly (pose, scale) or (b) a coverage axis it does not (Reynolds number, geometry family, boundary-condition layout and values)? The 2D conformal-Laplace result (G3, kernel rerun) says structure pays exactly on (a)-type axes and not at all on (b)-type axes; the high-lift and P1'' results say that on production datasets nothing separates the architectures in-family and canonicalization matches equivariance. G5 asks the same question on a 3D corpus built to isolate factors.

## Dependencies on the dataset (must exist for the arm to be valid)

| G5 arm | requires cfd-datagen recommendation |
|---|---|
| A1 pose twins, A2 scale twins | rec. 1 twins on `geometry_pose` and scale |
| A3 Re twins | rec. 1 twins on `physics_parameters` |
| A4 mesh twins (noise floor) | rec. 1 or rec. 3 refinement twins |
| B1 bank-held-out geometry | rec. 2 val/test geometries generated |
| B2 profile-held-out BCs | none (profiles are in the base corpus) |
| all surface metrics | rec. 4 `body_id` on `no_slip` |
| uncertainty-weighted metrics | rec. 5 block-difference fields |

If rec. 1 is not adopted, A1–A3 degrade to post-hoc marginal splits and their predictions below are void; only B1/B2 and the in-family control remain.

## Architectures and protocol

- GeoTransolver (lr 3e-3 and 1e-3, best per val), GeoTransolver + canonicalization where a single freestream exists (`open_opposed`, `ground_effect`, `confined_opposed`; NOT `dual_inlet`/`crossflow`), MeshTransformer2 with `similarity_gauge=true` (lr 1e-3), and the kernel-decoder MeshTransformer flagship config as the GLOBE-lineage arm. Optional: AB-UPT if the baseline exists by then.
- Surface task first (pressure and wall shear on body faces, measure-weighted rel-L2), volume task second (interior velocity/pressure on cell centroids with `_target_quadrature_measure`), since boundary→interior is the program's stated objective and MT2 has never run on it.
- Training set: the 3,360-case train corpus minus any held-out bank (B1). Two seeds minimum per architecture; report seed spread with every number. Sampling resolution 10k surface / 40k interior points per case; frozen otherwise.
- Every OOD number is reported next to the per-stratum label floor from A4 (medium→fine discretization difference) and the block-difference field (rec. 5). A difference between architectures smaller than the floor is reported as "not resolvable", not as a result.

## Arms, predictions, falsifiers (rel-L2 on body pressure unless stated)

**A1 — pose twins (exact SE(3) axis).** Evaluate each model on the pose twin of a training case (same geometry, physics, BCs, mesh stream; different rotation and translation).
- Prediction: MT2 twin-vs-original prediction difference equals its own mesh-realization floor (A4) to within 10% of that floor; raw GeoTransolver's difference exceeds the floor by ≥ 5× on `dual_inlet`/`crossflow` cases (no canonical frame available); canonicalized GeoTransolver on single-freestream profiles matches MT2 within 1.05× (P1'' replicated on real labels).
- Falsifier for the equivariance contract's practical value: canonicalized GT within 1.05× on *all* profiles including the two without a freestream (some other frame estimate suffices) → equivariance is a convenience here too.

**A2 — scale twins (exact similarity axis, first real test).** Same case at a different `reference_length_m` with viscosity adjusted so Re is fixed.
- Prediction: MT2 (gauge) twin difference ≤ 1.1× its A4 floor across the full 0.02–20 m range; GeoTransolver degrades monotonically with |log scale ratio|, exceeding 2× in-family error at a 10× scale change (it has no scale gauge and cannot be canonicalized for scale without knowing the reference length is arbitrary).
- Falsifier: GeoTransolver within 1.2× at 10× scale change → the network learns scale from context and the gauge buys nothing measurable.

**A3 — Reynolds twins (coverage axis).** Same case at a different Re, one decade away, within the training Re range.
- Prediction (from G3 and G2): no architecture generalizes across a decade of Re better than another; all degrade by > 1.5× versus in-family, with the ordering by in-family accuracy preserved. Structure does not buy Re transfer.
- Falsifier: any architecture degrades < 1.2× while another degrades > 2× on the same twins → a real Re-transfer mechanism exists and becomes the program's next object of study.

**A4 — mesh twins (the noise floor, not an architecture test).** Same case, different `mesh_resolution` draw (and, where rec. 3 exists, the refined mesh).
- Readout: per-stratum (profile × Re decade × bank) label difference in the recipe's metric. This defines "resolvable" for every other arm. Prediction: the floor exceeds 5% rel-L2 at Re ≥ 1e7 for most strata and is below 3% at Re ≤ 1e5.

**B1 — bank-held-out geometry (coverage axis, strong version of G2).** Train on four banks, test on the fifth's val/test geometries; rotate the held-out bank.
- Prediction: degradation ratio ordering follows shape-space MMD (Chapter 1b): holding out `zero-to-cad` costs the most, `streamlined` the least. Between architectures, prediction from G2 is that GeoTransolver's transfer improves more with training size than MT2's; run at two training sizes (1,000 and full) to test the slope. Falsifier for the "pointwise coordinate features carry transfer" hypothesis: MT2 (gauge) slope ≥ GeoTransolver's on the held-out bank.

**B2 — profile-held-out boundary conditions.** Train without `crossflow` (or `dual_inlet`), test on it.
- Prediction: architectures with explicit BC operators/values as inputs (kernel-decoder MT, MT2 with boundary scalars) degrade less than coordinate-feature architectures; threshold: ≥ 1.3× separation in degradation ratio. Falsifier: all within 1.1× → BC ingestion structure does not help on realized 3D flows either.

## Cost

Training: 4 architectures × 2 seeds × (1 in-family + up to 5 held-out configurations) on 3k cases of 1.5–5M cells — dominated by the volume task; budget ~2–4k GPU-hours on GB300, in chain-ahead links. Evaluation: twins and held-out sets are eval-only once trained. Preregistration and dataset decisions must be frozen together, which is why this draft exists before generation.

## What would change the program

- A1/A2 confirmed and A3/B negative: the thesis becomes "exact symmetries buy exact generalization on their axes, coverage axes are bought by data" — publishable with the 2D result as mechanism and 3D as confirmation.
- Any B-arm showing a ≥ 1.3× architecture separation with the A4 floor respected: the first measured architecture-driven generalization in 3D, and MT3's design target.
- Everything within the A4 floor: the honest conclusion that at 1.5–5M cells with transition SST the labels do not resolve architecture differences on OOD axes, which itself bounds what any surrogate benchmark on this operator can claim.
