# cfd-datagen: feedback from the mesh-transformer program

*Written 2026-09-03 after reading the research book (`docs/book/*.qmd`), the README, the geometry-bank contract, the schema-v3 chapter, the frozen production-corpus notebook entries, and the synthesizer's sampling code (`config.py`, `manifest.py`, `randomness.py`, `aga/production_plan.py`). Numbers below are quoted from those sources; nothing was run.*

The context on our side, in one paragraph: over the last three weeks the MeshTransformer program measured that most of its architectural claims do not separate architectures on DrivAerML/HiLift. GeoTransolver and MeshTransformer2 (MT2) are at parity in-family on HiLift (0.98x) and within 1.085x on DrivAerML at best-vs-best learning rates; MT2's exact rotation equivariance is matched by a canonicalized GeoTransolver at 1.01x cost; our zero-shot cross-family transfer result (DrivAerML → SHIFT-SUV) has a mechanism nobody has been able to move; and the only place any structural ingredient bought decisive generalization was a 2D exact-Laplace suite where the kernel decoder generalizes across unseen boundary-condition frequency content (rel-L2 0.07 vs ~1.0 for every soft-slice architecture). An independent critic's verdict was that the program has been climbing a single 48-case DrivAerML split ratio. **The thing we lack is exactly what this generator is designed to produce: a 3D corpus whose factors of variation are explicit, independent, and wide enough that architecture-level generalization differences become measurable at all.** So the short answer to (a) is yes, strongly — with the caveats and changes below, several of which matter more than corpus size.

---

## a) Is this valuable for teasing out generalization differences between architectures?

### Yes, for four specific reasons

1. **It varies the factors the production datasets hold fixed.** DrivAerML and SHIFT-SUV have Vendi ≈ 11–13 effective shapes (your own Chapter 1b), one freestream, one scale, one pose, one BC layout. Every "generalization" experiment we ran on them was therefore either in-distribution (where architectures tie) or a cross-family jump so large (car → SUV) that nothing separates cleanly. Your generator samples geometry family, dimensional scale (0.02–20 m, log-uniform), pose (whole-case SO(3) plus body rotation/translation), five BC profiles, three body layouts, and Re over five decades (1e3–1e8, log-uniform) as *independent named Philox streams*. That is the first 3D dataset where one can hold everything fixed and move one factor.

2. **It has the two axes where our program found structure actually pays.** In 2D, exactness on the PDE's own symmetry bought exact generalization on that axis (amplitude via linearity, rotation via equivariance) and nothing bought generalization on the coverage axis (BC frequency content). Your corpus gives us a *3D* test of exactly that: scale/pose/rotation are symmetries the architectures either carry exactly (MT2 with the new gauge) or not (GeoTransolver, unless canonicalized — which needs a known freestream, and your `dual_inlet`/`crossflow` profiles have no single freestream direction: that is the first dataset where canonicalization is *not* available for free and equivariance becomes a real, testable advantage). Meanwhile Re, profile, layout, and geometry family are coverage axes. A corpus that separates the two classes of axis is the instrument we have been missing.

3. **It ships the label-uncertainty metadata our noise-floor work could not get.** Our N1 probe found that ~70% of residual energy is shared across all trained models on both DrivAerML and HiLift and could not attribute it (label noise vs shared inductive bias) because those datasets carry no per-sample uncertainty. Your `mean_window` block (velocity/pressure/force block-RMS, `stability_pass`) is a model-free per-sample label-uncertainty estimate. Retaining `mean_unstable` samples rather than filtering them is also the right call for us: the hard strata (crossflow, massive separation) are where architectures might actually differ.

4. **It is DomainMesh-native with measures.** Interior cell volumes as `_target_quadrature_measure`, boundaries grouped by operator with per-face prescribed values, and the same key tree on every sample means both our surface recipe and the volume recipe (which the critic correctly noted MT2 has never run on) read it with the existing datapipe. Boundary→interior on this corpus is the program's stated north star and has never had a dataset.

### Caveats that limit the value as currently planned

- **One draw per geometry, train split only.** The frozen corpus (`assignment-5189ddd1900c1369`) is 3,360 cases = one seed per train geometry across five banks. Every factor is sampled jointly at random. OOD splits then have to be carved *post hoc* by thresholding a joint distribution, which confounds factors (a high-Re case is also a random geometry, random pose, random profile). For architecture comparisons, marginal-distribution shifts on a jointly random corpus are a weak instrument; **paired, single-factor counterfactuals are a strong one** (see recommendation 1). The generator's stream design makes that nearly free; the current plan does not use it.
- **No val/test geometry is generated.** The banks' 420 held-out geometries (200 blobs, 50+50 CAD/airframes, 60+60 streamlined/cars) are not in the assignment. Without them there is no leakage-clean geometry-OOD evaluation at all; downstream users will end up splitting the train geometries, which is exactly the "define splits after the fact" the book warns against.
- **The body is not identifiable in the artifact.** Schema v3 merges body walls and enclosure walls into one `no_slip` boundary and defers "which faces belong to the body" to the retained posed STL. Every surface-surrogate metric we have (pressure and wall-shear rel-L2 on the vehicle, force coefficients) needs the body faces; asking every consumer to re-match against an STL is fragile and will be done wrong. For multi-body samples it is essential.
- **Label error is not just stationarity.** Your accuracy ladder shows medium→fine grid changes of a few percent in wake velocity/pressure on four cases and a *non-monotone* sequence. For a dataset meant to separate architectures whose differences are 2–20%, the per-sample discretization error is as important as the mean-window error, and today it exists for four cases only (recommendation 3).
- **Transition SST at 1e3–1e8 with wall functions and a fixed layer rule** means realized y+ regimes vary by decades and the transition model is being used far outside anything validated. That is fine for a "one documented operator" stance, but downstream anyone comparing architectures on "laminar vs turbulent" strata should know the regime label is nominal-Re, not realized-flow. Your open direction 1 (characterize realized regimes) is the fix; it should be cheap to add a few realized diagnostics per sample now (recommendation 6).

Net: as planned, the corpus is an excellent *pretraining* set and a weak *discrimination* instrument. Three cheap additions turn it into both.

---

## b) Recommended changes to the run, ranked by (value to this research) / (cost)

### 1. Add counterfactual twin draws — the single highest-value change

**What.** For a subset of geometries, emit paired cases that share every named stream except one. The streams already exist and are independent (`geometry_selection`, `geometry_pose`, `domain`, `boundary_layout`, `boundary_values`, `physics_parameters`, `mesh_resolution`). A twin is "same root seed for all namespaces, different key for one." Implementation is a per-namespace seed override in the manifest (`stream_seed_overrides: {namespace: seed}`), recorded like any other manifest field, so it stays reproducible and auditable. The planner's `draws_per_geometry` argument is the hook; a twin is a second draw with a declared varied namespace.

**Which twins, and what each measures.**

| twin (vary only…) | what it isolates | which of our claims it tests |
|---|---|---|
| `physics_parameters` (Re / speed / viscosity) | Re generalization at fixed geometry, pose, BCs, mesh topology | over-symmetrization / Reynolds conditioning (our M1 never actually tested it) |
| `geometry_pose` (rotation + translation) | exact SE(3) covariance on *real* solver labels, incl. mesh-realization noise | MT2's equivariance vs canonicalized GeoTransolver; also gives a **model-free label-noise floor** (two solves of the same physics in different poses differ only by discretization) |
| scale only (reference length, with Re held fixed via viscosity) | exact scale covariance | our new similarity gauge; GeoTransolver cannot be canonicalized for scale |
| `boundary_layout` (slip/no-slip bands) | BC-ingestion at fixed geometry | GLOBE-style BC channels vs coordinate features |
| `boundary_values` (inlet speeds, outlet pressure, second inlet) | BC-value extrapolation, incl. linearity checks at low Re | the amplitude axis where 2D linearity was exact |
| `mesh_resolution` | discretization error per case; density robustness with real remeshing | P3 density probe on real labels; per-sample numerical-error estimate |

**Size.** Twins are the expensive kind of sample (a full solve each), so keep it targeted: 200 base cases stratified across banks and profiles × 3 twins each (pose, physics, mesh) = 600 extra solves, plus scale/layout/value twins on a 60-case subset (180 more). That is ~25% on top of 3,360 and it converts the corpus from "pretraining set" to "instrument." If budget forces a choice, do **pose + mesh twins first**: together they give the label-noise floor that every downstream comparison needs to be honest about, and they cost nothing in new physics.

**Why now.** This is a planner/manifest change. Once the corpus is frozen and launched, adding twins later means a separate campaign with a different code revision and a requalification argument; doing it inside the same assignment keeps one operator identity.

### 2. Generate the val and test geometries too, with more draws per geometry

**What.** Extend the assignment to the banks' val/test splits (420 geometries). Give the *test* geometries 2–3 draws each (different `physics_parameters`/`geometry_pose` seeds) so geometry-OOD evaluation has enough samples per geometry to be seed-robust; today's 60-geometry SHIFT-SUV zero-shot numbers swing 0.9–1.1 across training seeds and we cannot tell signal from noise.

**Why.** Bank-level geometry OOD (train on four banks, test on the fifth) is the cleanest 3D test of "does the architecture learn flow or learn the shape family," and Chapter 1b's MMD analysis already shows the banks are well separated in shape space. Without generated test geometries there is no leakage-clean version of that experiment. Cost: ~420–900 solves. This can be a separate shard family with its own seed range so it never mixes with training seeds.

### 3. Add a mesh-refinement twin for a stratified subset (and record medium→fine error per case)

**What.** For ~100 cases spanning profiles, banks, and Re decades, solve a second mesh at ~2.5× the background budget (the accuracy chapter's medium→fine step) and store the medium-vs-fine field differences as per-case scalars in `diagnostics/solve/discretization/` (velocity RMS over inlet speed, pressure RMS over dynamic head, force fraction — the same normalizations you already use for the mean window). Also keep both `.pdmsh` artifacts.

**Why.** Your Chapter 5c shows scheme and grid effects that "can exceed the last refinement step and vary strongly by case." A downstream comparison claiming a 10% architecture difference on strata whose per-case discretization error is 5–15% is not a result. This subset makes the numerical error a *measured* per-stratum quantity, lets us report architecture differences only where they exceed it, and doubles as the mesh-realization twin in recommendation 1. It is also the honest answer to "how good are the labels" that DrivAerML never gave us.

### 4. Put body identity into the artifact

**What.** On `no_slip`, add `cell_data/body_id [F] int64` (0 = enclosure wall, 1..k = body components as posed) and `global_data/geometry/body_count`. Optionally `cell_data/source_patch_id` for the band layout so BC-layout experiments can recover the three bands without the STL. Both are known at export time (the exporter already matches patch names and face counts to the manifest).

**Why.** Every surface metric we compute is on the body. `ground_effect` and `confined_opposed` put enclosure walls and body walls in one merged boundary with identical operators, and "derive it from the posed STL" will be implemented five different ways by five consumers. This is a few integers per face and prevents a whole class of silent evaluation bugs. It keeps the operator-typed grouping you chose; it just adds a label, not a boundary.

### 5. Export the half-window disagreement *field*, not only its scalars

**What.** You already compute the two 40 T_c block means to test stationarity. Store their per-point difference for velocity and pressure (`solution_uncertainty/velocity_block_difference [N,3]`, `.../gauge_pressure_block_difference [N]`, and the same on `no_slip` for wall shear). One or two extra fields, ~15% artifact growth.

**Why.** The scalar `mean_window` block says *whether* a sample is stationary; the field says *where* it is not. That enables (i) uncertainty-weighted losses and metrics, so an architecture is not penalized for failing to fit an unconverged wake, (ii) a direct test of the N1 question — does the shared residual across architectures co-locate with label non-stationarity, and (iii) an honest per-point noise floor. Without the field, downstream users will either drop `mean_unstable` samples (undoing your deliberate retention) or treat unconverged regions as truth. This is the highest-value *schema* change for us and the one that is impossible to add after the natives are cleaned up — which argues for deciding it before launch.

### 6. Record realized-regime diagnostics per sample

**What.** A handful of scalars per sample from fields you already have: area-weighted mean and quantiles of y+ on `no_slip` (you store y+ per face — just summarize it), fraction of body faces with reverse flow / separated (from wall shear direction vs local freestream), volume fraction of intermittency below 0.5 (laminar), wake reverse-flow fraction on the retained interior points. Put them in `diagnostics/flow/`.

**Why.** Stratifying architecture comparisons by *realized* regime (attached vs separated, resolved vs wall-function-dominated y+) is the difference between "MT2 is worse at high Re" and "MT2 is worse where y+ > 300 and the layer coverage collapsed." Chapter 9 already flags that the fixed layer rule cannot hold one y+ regime across five Re decades; the corpus should make that visible per sample so it can be a covariate, not a confound. Cheap, and it also feeds your own open direction 1 (active learning on sparse regimes).

### 7. Rebalance the Re prior toward where architectures can be distinguished

**What.** Log-uniform 1e3–1e8 puts 20% of the corpus in each decade. Consider a mild tilt (e.g., 30/25/20/15/10 from low to high) or, better, hold the marginal but ensure the twin subsets (rec. 1) and the test geometries (rec. 2) are decade-balanced.

**Why.** At Re ≥ 1e7 with a 1.5–5M-cell mesh and wall functions, the labels are the most model-dependent (transition SST far from its calibration, y+ in the hundreds) and the least reproducible across mesh realizations — your own crossflow stationarity failures cluster in the hard strata. Architecture differences measured there will be dominated by label error (rec. 3 will show this). The low and middle decades are where the same architectures can be told apart cleanly. This is a soft suggestion; recs. 3 and 6 make the corpus self-diagnosing either way.

### 8. Two smaller items

- **Freestream/drive reference for the non-freestream profiles.** `reference/velocity` is stored per case, but for `dual_inlet` and `crossflow` there is no single drive vector. State in the schema which inlet defines `reference/velocity` (largest area-weighted speed? the manifest's "primary inlet"?) and store the second inlet's direction and speed as global scalars too. Our equivariant models take a drive vector; canonicalization-style baselines need to know it is ill-defined for these profiles — which is precisely what makes those profiles valuable.
- **Keep the native cases for the twin and refinement subsets.** Your retention policy already keeps a stratified subset; make the twin/refinement subsets part of it explicitly, since those are the samples most likely to need re-export (e.g., if rec. 5 is adopted after launch).

### What I would not change

- The fixed numerical operator, no-replacement rule, and retention of `mean_unstable` samples. These are the properties that make the corpus usable as an instrument; every downstream temptation will be to filter the hard cases, and the frozen denominators are what stop that.
- The operator-typed boundary grouping. It is the right abstraction for BC-ingestion experiments; rec. 4 adds a label to it rather than reverting to patch-keyed boundaries.
- Interior connectivity omission. Point clouds with volumes are what every architecture in our comparison consumes; connectivity would triple the artifact for no benefit to the experiments we would run.

---

## Priority summary for the launch decision

If only one change is possible before launch: **rec. 1 with pose + mesh twins** (label-noise floor plus exact-symmetry tests, ~400 solves). If two: **rec. 5** (block-difference fields; cannot be recovered after cleanup). If three: **rec. 4** (body identity; cheap, prevents silent bugs). Recs. 2 and 3 can be separate shard families launched after the main corpus without breaking the operator identity, as long as they are planned now so the seed ranges and geometry splits are frozen together. Recs. 6–8 are schema/metadata additions worth doing before the exporter is frozen but not blocking.

What we would run on it first, to be concrete: train GeoTransolver and MT2 (gauge config) on four banks, evaluate on the fifth (rec. 2); read the pose and scale twins for exact-covariance-on-real-labels (rec. 1); read the physics twins for Re generalization at fixed geometry; and report every difference against the per-stratum discretization and stationarity floors (recs. 3, 5). That experiment does not exist today on any 3D dataset, and it is the one whose outcome would actually change what we build next.
