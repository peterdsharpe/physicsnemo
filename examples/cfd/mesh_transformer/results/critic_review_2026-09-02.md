# Independent critic's review of the mesh-transformer research program

Date: 2026-09-02. Reviewer: external, no prior contact with the program.
Sources: the research book (`examples/cfd/mesh_transformer/book/*.qmd`,
branch `psharpe/mesh-transformer`), the checked-in artifacts under
`examples/cfd/mesh_transformer/results/`, the model under test
(`physicsnemo/experimental/nn/mt2/model.py`), its contract tests
(`test/experimental/nn/test_mt2.py`), and the training recipe configs under
`examples/cfd/external_aerodynamics/unified_external_aero_recipe/`. Every
number below was recomputed from the artifact named next to it; every
criticism cites a file and anchor. Nothing was committed.

## Glossary (project terms used below)

- **MT1 / MeshTransformer**: the original GLOBE-lineage architecture (typed
  scalar/vector streams, signed boundary self-attention, exact single/double
  layer panel-integral decoder). **GLOBE**: the equivariant boundary-integral
  network (arXiv 2511.15856) MT1 generalizes.
- **MT2 / MeshTransformer2**: the current product: a soft-slice
  ("Transolver-style") global-routing backbone fed only rotation/translation
  invariants, with a GLOBE-style multi-vector output head.
- **GT / GeoTransolver**: the mainstream baseline, a soft-slice global-attention
  point transformer consuming raw coordinates, normals and the freestream
  vector.
- **DrivAerML**: 435-car automotive surface-pressure/wall-shear dataset
  (time-averaged transient RANS-class labels). **SHIFT-SUV**: a second vehicle
  dataset with two body styles (estate, fastback) used as held-out families.
  **HiLiftAeroML**: 1,800-case high-lift aircraft dataset (compressible).
- **rel-L2**: relative L2 error on dataset-normalized fields; 1.0 is the
  mean-predictor level.
- **P1–P4**: adversarial probes: P1 random-yaw rotation orbit, P2 drive
  amplitude scaling, P3 10:1 spatially biased sampling with Horvitz–Thompson
  (**HT**) inverse-inclusion weights, P4 companion-query-set stability.
- **G1–G4**: the "generalization north star" ledger: G1 error vs training-set
  size, G2 zero-shot held-out-family error vs training-set size, G3 boundary
  condition (**BC**) axes on an exact-label 2D Laplace suite, G4 the group orbit
  (= P1).
- **M1–M4**: the "literature wave": M1 scale-equivariance relaxation, M2
  anchor-conditioned passive decode (**AB-UPT**-style), M3 invariant
  completeness audit, M4 measure-weighted normalization audit (**NEO**-style).
- **HL / HLGEN**: the HiLiftAeroML in-family wave and its planned
  generalization ladder. **N1, R1, V6, W1, W1', L1, L1b, L2, W2**: the
  2026-09 "instrument wave" and the vector-head parity line (see
  `18-notebook.qmd` from `#sec-nb-campaign-review` onward).
- **OOD**: out-of-distribution. **BIE**: boundary integral equation.
  **ZPN**: zero-preserving-nonlinear drive read-in mode of MT1.

---

## 1. What the program claims, and what the evidence supports

The program's own authoritative summary (`17-program-review.qmd`
`#sec-program-review-2026-08`, and the callout at the top of `index.qmd`)
makes, in my words, the following claims. Each is graded against the
artifacts.

| # | Claim (my paraphrase) | Where claimed | Evidence and my recomputation | Grade |
|---|---|---|---|---|
| C1 | MT2 matches GT in-family on DrivAerML to within 10–20%. | `index.qmd` callout; `17-program-review.qmd` §verdict, `#sec-review-frontier` | `mt2_v3c_reduction.json` + `mega_reduction.json`: five MT2-v3c seeds at 10k cells {0.0636, 0.0627, 0.0634, 0.0631, 0.0671}, mean 0.0640; GT five seeds mean 0.0531 → **1.20×** at the frozen lr (the review reports 1.18× from three seeds). Best-vs-best (R1, `instrument_wave_reduction_2026-09-01.json`): MT2 lr 1e-3 mean 0.0577 / GT 0.0532 = **1.085×** (2 seeds). | Well-supported. |
| C2 | MT2 is exactly rotation-robust; GT degrades 26–32× off-pose. | `index.qmd`; `17-program-review.qmd` §3 | `probe_reduction.json` P1: GT means 1.71/1.60/1.40 vs in-pose 0.053 → 32×/30×/26×; MT arms ratio ≈1.003. MT2 P1 ratios 0.996–1.000 (`mt2_v3c_reduction.json`). Note the orbit is **yaw only** (z-rotations), and only proper rotations. | Well-supported for yaw. |
| C3 | Recovering pose robustness by augmentation costs GT 1.8–1.9× accuracy; hence exact equivariance is worth ~1.6×. | `17-program-review.qmd` `#sec-review-frontier`; `18-notebook.qmd #sec-nb-mega-synthesis` | `mega_reduction.json` gt_rotaug in10k {0.1015, 0.0977} / 0.0532 = 1.87×. Two seeds, fixed 500 epochs, no re-tuning for the harder augmented task, and **no canonicalization counterfactual** (see §4 item 1: on a yaw orbit with a known freestream direction, aligning the input frame to U_inf is exact and costs nothing). | Plausible but under-powered, and compared against the weakest remedy. |
| C4 | MT2 enjoys *exact similarity equivariance* (translation, rotation, **scale**) by construction, enforced by fp64 tests. | `02-architecture.qmd #sec-mt2-architecture` ("All four contracts — similarity equivariance …"); `index.qmd` M1 ledger row; `17-program-review.qmd` M1 bullet; `model.py` module docstring | `model.py` line 428: `r = (points - center) / self.reference_length` with `reference_length=8.0` a **constant** (also in `conf/model/mt2_surface.yaml`: "Constant scale gauge"). `r_mag` and `log r_mag` are then fed to the backbone. I ran the default model in fp64: rescaling the cloud ×2 changes outputs by **20.0%**, ×0.5 by **18.9%**. `test_mt2.py` contains no scale-equivariance test. | **Contradicted by the code.** |
| C5 | M1: relaxing scale equivariance was tested and refuted (1.3–2.0% gain). | `18-notebook.qmd #sec-nb-m1-verdict-wave-close`; `index.qmd` ledger | Because of C4 the backbone already sees absolute size through `r_mag`; the M1 feature `log(mean |points − center|)` is a deterministic function of quantities the slice aggregation already computes. A null was guaranteed by construction. `m1_reduction.json` vs `ladder_reduction.json`: n=218 0.0740 vs 0.0750 — inside noise, as expected for a redundant feature. The "transfer penalty" (0.95 vs 0.83) is inside MT2's zero-shot seed spread at those rungs (0.80–1.08). | **Unsupported** (invalid test of the hypothesis). |
| C6 | M4: MT2 is "measure-complete"; the P3 density floor is information-limited, not a bug. | `18-notebook.qmd #sec-nb-lit-wave-prereg` (M4), `#sec-nb-m1-verdict-wave-close`; `17-program-review.qmd` | `model.py` line 427: `center = points.mean(dim=1)` — an **unweighted** mean over the sample; likewise `scale_conditioning` (line 458). Under a 10:1 spatial bias with exact HT weights, I measured the unweighted centroid shifting **0.52** cloud-RMS-radii versus **0.027** for the HT-weighted centroid (19× larger). Every seed invariant (`r_mag`, `log r_mag`, `r̂·d`, `r̂·n`) and the anchor invariants `z_mag`, `ẑ·d` are functions of that center. The recipe's `CenterMesh` also uses `use_area_weighting: false` (`datasets/drivaer_ml_surface.yaml` line 58). MT2-v3c's P3 ratio is 13–14× (`mt2_v3c_reduction.json`), 3.7× *worse* than GT's 3.6× with no weights at all. | **Contradicted by the code.** |
| C7 | The declared drive-degree-1 contract is exact and is a merit for aerodynamics; G3 "measured its value at 4.5–12×". | `17-program-review.qmd` §3; `18-notebook.qmd #sec-nb-g3-verdict` last paragraph; `02-architecture.qmd` forward pass step 4 | On the recipe the drive is the **unit** freestream direction (`U_inf_dir`, `datasets/drivaer_ml_surface.yaml` lines 59–68), so P2's "d = 1.0000" is the identity `out × |d|` measured on a rescaled unit vector — a tautology the notebook itself calls "weak support only" (`#sec-nb-contract-probes-verdict`). Physically, pressure *coefficient* is degree 0 in freestream speed and dimensional pressure is degree 2; degree 1 is the wrong law for both. The G3 4.5–12× number is for exact linearity on a *linear* PDE (Laplace) and does not transfer to Navier–Stokes, where the PDE is not linear in the drive. | **Unsupported**, and the physics is mis-stated. |
| C8 | G2: GT's zero-shot transfer improves monotonically with training data (0.784 → 0.657, "five points, no reversal"); MT2's is flat. | `18-notebook.qmd #sec-nb-g1g2-verdict`; `index.qmd #sec-tech-tree`; `17-program-review.qmd #sec-review-refutation-wave` | `ladder_reduction.json` (2 seeds/rung). GT estate per seed 42: 0.730, 0.715, 0.669, 0.647, **0.681** (n=218 worse than n=109 and n=54) — the per-seed curve *does* reverse; means 0.784, 0.740, 0.685, 0.669, 0.657 with within-rung spreads 0.107, 0.051, 0.032, 0.045, 0.049. Beyond n=54 the mean steps (0.016, 0.012) are smaller than the seed spread. The robust part is the endpoints: GT at n=435 (5 seeds) 0.614; MT2 at n=435 (5 seeds) **0.997**, i.e. mean-predictor level. MT2 at n=109 has seeds 1.08 and 0.80 (spread 0.29). | Endpoints well-supported; the *slope* story is under-powered. |
| C9 | Locality is refuted as the transfer carrier. | `#sec-nb-locality-verdict` | `locladder_reduction.json`: 2 seeds, estate at n=218 {0.97, 1.09}; the refutation rests on the local arm not beating a plain-MT2 curve whose own spread is 0.1–0.3. | Plausible but under-powered. |
| C10 | Multi-family training transfers ~7× for both architectures; MT2 transfers "at parity". | `17-program-review.qmd` §verdict and `#sec-review-frontier` | `mega_reduction.json`: fastback zero-shot MT2-multi 0.1267 vs GT-multi 0.1101 → **1.15× worse**; in-family MT2-multi 1.52× GT (the review says so in the notebook but "parity" in the chapter). | The 7× coverage effect is well-supported (3 seeds); "parity" is generous. |
| C11 | Query independence costs 2.7–3.9×; anchor decode loses 3.1× at matched VRAM. | `#sec-nb-v5-line-verdict`, `#sec-nb-instrument-wave-verdict` | `instrument_wave_reduction_2026-09-01.json`: v5a4 {0.209, 0.221} vs v3c@6500 {0.0697, 0.0671} → 3.14×. Consistent across three passive designs. | Well-supported within this design family. |
| C12 | HL: on HiLiftAeroML the DrivAerML gap closes (MT2/GT = 0.98). | `#sec-nb-hl-verdict` | `highlift_wave_reduction.json`: GT lr1e-3 {0.0408, 0.0435} mean 0.0422; MT2 {0.0401, 0.0425} mean 0.0413; ratio 0.979. Difference of means 0.0009 versus within-arm spreads 0.0027/0.0024. Two seeds, two-point lr grid. Also, the committed `datasets/highlift_surface.yaml` has **no** `ComputeFreestreamDirection`/`U_inf_dir` step and no `reference_length` leaf, yet `mt2_surface.yaml` consumes `global_data.U_inf_dir`; the MT2 HiLift dataset config that actually ran is not in the record (last commit to that yaml: 2026-05-20). | Plausible (parity), under-powered, and **not reproducible from the checked-in record**. |
| C13 | Learning-rate robustness is a reproducible MT2 architecture property. | `#sec-nb-hl-verdict`, `#sec-nb-instrument-wave-verdict` (R1) | R1 artifact: GT at lr 1e-2 {0.739, 0.221}, MT2 {0.0874, 0.0914}. MT2 missed its own <1.5× bound (1.55×). N1 (`n1_noise_floor_2026-09-01.json`) shows MT2 has *larger* seed-to-seed prediction disagreement than GT (same-arch D 0.83 vs 0.66). | Plausible; two seeds; robustness to lr and robustness to seed diverge. |
| C14 | The DrivAerML 1.1–1.2× gap was label noise. | `#sec-nb-campaign-review`; still framed as HL's purpose in `17-program-review.qmd #sec-review-honesty` | N1: shared-residual fraction 0.713 (DrivAerML) vs 0.703 (HiLift) — no discrimination; the notebook itself retracts (`#sec-nb-n1-verdict`). L1 finds label mirror-antisymmetry <1% (`l1_label_covariance_2026-09-02.json`). | **Unsupported** (retracted in the notebook, still present in the chapter). |
| C15 | G3: linearity buys exact amplitude extrapolation; *no* tested mechanism generalizes across BC frequency content. | `#sec-nb-g3-verdict`; `17-program-review.qmd` | `g3/g3_reduction.json`: mt1_linear T2 ratios 0.92/0.87 — but rel-L2 of an exactly linear model is *invariant* to drive scaling, so this is a tautology, not a measurement; the informative numbers are the 4.5–12× penalties of the non-linear arms. The T1 claim is **confounded**: `studies/g3_bc_generalization.py` line 246–249 builds the MT1 arm with the **moment** query decoder (`make_model("mesh_transformer","reference")`), the configuration with the *proved* m ≤ 2 angular ceiling (`index.qmd` qualification 2), not the kernel-decoder reference configuration that `06-benchmarks.qmd #sec-transolver` reports at freq-OOD **0.0748** on the *same* `unseen_boundary_frequencies` split (modes 5–8 vs 1–4). | Amplitude part: trivially true. Frequency part: **contradicted by the program's own chapter 6**. |
| C16 | MT2 is at "equal-or-lower cost" than GT. | `17-program-review.qmd` §verdict; `index.qmd` "at equal cost" | Frontier table lists MT2-v3c at 5.9 GB; the notebook reports **9.8 GB** at 10k tokens (`#sec-nb-v5a4-gate`, `#sec-nb-instrument-wave-verdict` V6: "v3c at 10,000 tokens 9.81 GB") vs GT 4.6 GB → 2.1× the memory. | **Contradicted** (and internally inconsistent). |
| C17 | The MT1 synthetic-suite results (3-parameter BIE matching the analytic constant; kernel-decoder repair 11.3×; Transolver beaten 20×). | `index.qmd`, `04-learned-bie.qmd`, `06-benchmarks.qmd` | Frozen banks, 3 seeds, checked-in JSON. The Transolver "20×" is at matched *steps* (3k, lr 3e-4) while Transolver's own step curve is still descending at 30k — a data-efficiency claim, not a convergence claim (the chapter says so). | Well-supported, but not about the product (MT2 has none of these ingredients). |

Summary of grades: 6 well-supported (C1, C2, C10-coverage, C11, C17,
C8-endpoints), 5 plausible-but-under-powered (C3, C9, C12, C13, C8-slope),
3 unsupported (C5, C7, C14), 4 contradicted by the program's own code or data
(C4, C6, C15-frequency, C16).

---

## 2. Internal inconsistencies

Ordered by consequence.

1. **Scale equivariance is claimed and not implemented.**
   `02-architecture.qmd #sec-mt2-architecture` states all four contracts
   including similarity equivariance are "exact by construction and enforced
   by fp64 unit tests". `model.py` divides by a constant 8.0 and feeds `r_mag`
   to the network; there is no scale test. The same chapter's callout
   (`02-architecture.qmd` step 1, "Using the override safely") warns: "If you
   must override, override with a similarity-covariant functional of the input
   geometry — never a constant … A number pinned in a config file is not a
   length." `conf/model/mt2_surface.yaml` pins exactly such a number, with the
   comment "Constant scale gauge, identical to the MeshTransformer flagship's".
   The book's own MT1 history (`17-future.qmd` "The scale gauge is intrinsic
   by default") records that the constant gauge was the first suspect for MT1's
   cross-family transfer failure. MT2 inherited the constant and nobody
   re-asked the question; instead M1 "tested" relaxing a symmetry MT2 never
   had (C5).

2. **The M4 "closed by derivation" audit missed the two unweighted
   population statistics** (`center`, `raw_scale`; C6). The consequence is not
   cosmetic: the whole P3 story for MT2 ("information-limited estimator
   variance, no bug", `17-program-review.qmd #sec-review-refutation-wave`) rests
   on this audit, and MT2's 13–14× is presented alongside a sentence written
   for MT1 ("the measure contract halves the damage", `17-program-review.qmd`
   §3) although for MT2 the measure-aware model is 3.7× *more* density
   sensitive than the weight-blind baseline.

3. **Frequency generalization: chapter 6 vs G3.** `06-benchmarks.qmd`
   (`#tbl-splits` and the paragraph after it) calls the
   `unseen_boundary_frequencies` split "the program's sharpest discriminator"
   and states "everything with a PDE-conforming operator has freq-OOD ≈ ID"
   (MT1 singular-only 0.0748). `18-notebook.qmd #sec-nb-g3-verdict` concludes on
   the same split that "no arm generalizes across frequency content … even the
   exactly-linear mt1 fails … a data/coverage limit no candidate mechanism
   addresses", and `17-program-review.qmd` propagates that into the north
   star. The G3 MT1 arm is the moment decoder (§1 C15). Neither text notices
   the other.

4. **Program-review chapter is nine days stale and says so nowhere in its
   verdict.** `17-program-review.qmd` and `index.qmd #sec-tech-tree` (both
   "as of 2026-08-24") list HL as RUNNING, describe the "irreducible-error
   question" as what HL "exists to answer", and give 10–20%/1.10×/1.18× as the
   in-family gap. The notebook since then: HL complete (parity), the
   noise-floor attribution retracted (N1), best-vs-best 1.085× (R1), the
   v5a4 arm re-opened and re-closed at 3.1× (V6), and six vector-head arms
   (W1, W1', L1, L1b, L2, W2). `index.qmd` says the review chapter is "the
   authoritative, current synthesis".

5. **VRAM: 5.9 GB vs 9.8 GB** for the same MT2-v3c configuration
   (`17-program-review.qmd` `tbl-frontier` rows vs `18-notebook.qmd`
   `#sec-nb-v5a4-gate` and V6). One of them is wrong; the "equal-or-lower
   cost" sentence depends on which.

6. **"Full O(3)" vs the default head.** `02-architecture.qmd
   #sec-mt2-architecture` lists "full O(3)" among MT2's exact contracts;
   the same section then documents that the default head violates reflection
   covariance and that the fix is default-off. `test_mt2.py::test_parity_fix_reflection_equivariance`
   asserts that the default model is *not* reflection covariant. W1/W1'/L2
   then show every covariant head is 6–26% worse (`#sec-nb-l2-verdict`).
   The honest statement is "SO(3)-equivariant; reflection covariance costs
   accuracy on this dataset and is off".

7. **Degree-1 self-contradiction.** `02-architecture.qmd` step 4 says the
   homogeneous mode is "the right declaration when the target depends …
   roughly linearly on its amplitude, as nondimensional surface loads under a
   unit freestream do". Nondimensional loads are degree 0 in freestream speed
   by definition. `18-notebook.qmd #sec-nb-contract-probes-verdict` concedes
   the drive axis "is not a live axis" on the recipe; `17-program-review.qmd`
   §3 still lists "measures exactly 1.0000" as a result against GT.

8. **G2 "five points, no reversal"** (`#sec-nb-g1g2-verdict`) vs the artifact:
   seed 42 reverses at n=218 (§1 C8).

9. **"Transferring at parity when trained on two families"**
   (`17-program-review.qmd` §verdict) vs 1.15× worse zero-shot and 1.52× worse
   in-family in `mega_reduction.json`, both of which the notebook reports
   (`#sec-nb-mega-synthesis`).

10. **Missing artifacts.** The HiLift generalization-ladder preregistration
    "499bcfe8" cited at `#sec-nb-hl-verdict` has no file under `results/`
    (grep of the repo finds only the notebook line). The MT2 HiLift dataset
    configuration is not in the repo (§1 C12). The notebook's own lesson
    ("job-local fixes are loans, not payments", `#sec-nb-mt2-stage1-diversity`)
    applies.

11. **Stale test comment.** `test_mt2.py` lines 30–32 explain the
    anisotropic fixture by "the principal-axis frame is exactly covariant only
    where the covariance spectrum is non-degenerate"; the principal-axis frame
    was deleted at v2 (`#sec-nb-mt2-v2`). Harmless, but it signals that the
    test file was extended rather than re-read.

12. **The book's "thesis in one paragraph"** (`index.qmd`) describes an
    architecture — declared contracts, two-member exact singular dictionary,
    one signed attention layer, ~82k parameters — that the product no longer
    is. MT2 has no kernel dictionary, softmax (unsigned) routing, no
    superposition contract, no query independence, and 8.6M parameters. The
    index's opening thesis and its opening callout describe two different
    programs.

---

## 3. Hill-climbing diagnosis

**Stated objective** (`index.qmd` "Problem statement and scope"; the north
star at `#sec-nb-generalization-northstar`): a surrogate for boundary-driven
steady PDEs mapping {boundary geometry, BC types, BC values} → interior field
at bare query points, for *unknown* operators, that generalizes beyond the
training distribution along geometry, boundary data, PDE parameters and
resolution, with generalization measured as curves.

**Revealed objective** (what the runs actually optimized): DrivAerML surface
pressure rel-L2 at 10,000 cells on a fixed 48-case validation split, expressed
as a ratio to GeoTransolver at a frozen protocol. Evidence:

- Every MT2 iteration gate from v0 to v3c is that ratio (`#sec-nb-mt2-charter`
  S0_1; `#sec-nb-mt2-v3a`; `#sec-nb-mt2-v3b-verdict`; `#sec-nb-mt2-stage0-synthesis`).
  Every accepted step was an ingredient copied from GT (routing at v0,
  separating features at v0b, relational features at v3b); the program says so
  itself ("every gap-close came from adopting GeoTransolver's ingredients",
  `#sec-nb-checkin-2026-08-20`).
- Along the way the axes the stated objective cares about got *worse* and were
  scoped away rather than optimized: zero-shot transfer 0.70–0.80 (v0b) →
  0.95–1.03 (v3c) → 0.997 five-seed; density sensitivity 5.7× (v0) → 13–14×
  (v3c). v3c was made canonical despite failing its own zero-shot gate (< 0.85)
  by the pre-declared rule (`#sec-nb-mt2-stage0-synthesis`).
- The problem class in the stated objective is boundary → *interior*. The
  entire industrial campaign is surface → surface (targets on the same cells
  that are the inputs; `mt2_surface.yaml` `points: interior.points` are the
  surface centroids). The unified recipe ships `drivaer_ml_volume.yaml` and
  `highlift_volume.yaml`; neither has been touched by MT2.
- The last ten days (W1, W1', L1, L1b, L2, W2: six arms, ~14 lanes) climb a
  vector-head parity detail whose entire stake is ≤26% of *wall-shear* error
  on one dataset while pressure is the headline metric and the AB-UPT baseline
  declared "the live threat" on 2026-08-20 (`#sec-nb-lit-synthesis` item 4)
  remains unmeasured.

**Where the two objectives diverge, concretely.** The stated objective
rewards structure that buys extrapolation. The revealed objective rewards
matching GT's in-distribution number while retaining a yaw-invariance the
benchmark does not test. The local optimum this converges to is exactly what
MT2 is: "GeoTransolver over rotation-invariant features." That is a
respectable engineering artifact, but the program's own G-ledger says it does
not generalize better on any axis except the declared group orbit (P1), and
§1 C4/C6 show the "declared structure" is looser than advertised (scale not
equivariant; centering density-sensitive). The mechanism hunt for the G2 gap
(`#sec-nb-m1-verdict-wave-close` "what remains") ends with "GeoTransolver's
raw coordinates and non-equivariant encoder learn what exact similarity
equivariance forbids" — a hypothesis that cannot be right as stated, because
MT2 is *not* scale-equivariant and both models see absolute size in the same
L_ref = 5 m units.

One more sign of local-optimum behaviour: the *strongest* structural result
the program ever had — frequency-content extrapolation by an exact singular
kernel dictionary (`06-benchmarks.qmd #sec-transolver`, freq-OOD 0.075 vs
Transolver 0.46 at 10× compute) — was dropped from the product because it
was "evidence-orphaned at industrial scale" (`02-architecture.qmd` last
paragraph), and then G3 re-asked the frequency question *without* it and
concluded structure cannot help (§2 item 3).

---

## 4. Missed directions and unturned stones

Ranked by (information gained)/(cost). Costs are in "lanes" (one 500-epoch
4-GPU DrivAerML run ≈ 3–6 h per the notebook's telemetry) or eval-only.

1. **Canonicalization counterfactual for GT (eval-only).** The headline
   differentiator (C3: "augmentation costs GT 1.87×; equivariance is free")
   compares against the *most expensive* way to obtain yaw robustness. On this
   instrument the orbit is 1-parameter (z-rotations) and the freestream
   direction is an input: rotate the input so U_inf ∥ +x, run GT, rotate the
   vector outputs back. Because GT was trained in canonical pose this is
   *exactly* invariant at zero training cost. The notebook's correction that
   "no single canonical frame exists for problems with several non-orthogonal
   vector inputs" (`#sec-nb-contract-probes-prereg`) is true in general and
   irrelevant to a yaw orbit with one free vector. If GT-canonicalized posts
   P1 ≈ 1.00 at in-family 1.00×, the 1.6× "structured model is the more
   accurate one at equal pose-robustness" sentence in `17-program-review.qmd`
   is void. Cost: a few GPU-minutes on existing checkpoints.

2. **Covariant gauge and weighted centroid in MT2 (code + 3 lanes).** Replace
   the constant 8.0 with the measure-weighted RMS radius and the unweighted
   mean with the measure-weighted mean; add the missing scale-equivariance
   test; retrain v3c at 2–3 seeds; read in-family, estate/fastback zero-shot,
   P3. This is the program's own MT1 lesson (`02-architecture.qmd` step 1),
   the only untested candidate for the G2 gap that lives inside MT2, and a
   direct test of the M4 closure. Note the risk the MT1 history flags: the
   intrinsic gauge did *not* rescue MT1's transfer (`17-future.qmd`); MT2 may
   behave the same, which would itself be informative about where family
   binding lives.

3. **G3 with the kernel-decoder MT1 and a kernel readout for MT2
   (< 3 GPU-hours).** `mt1_linear` took 35 min per seed on G3. Run
   `mesh_transformer_kernel_nomlp` (the singular-only reference configuration
   of `02-architecture.qmd #sec-reference-config`) on T0–T3. If T1 ≈ 0.07 as
   chapter 6 predicts, frequency-content generalization *is* purchasable by
   structure and the north star's "spectrum coverage is a data axis"
   conclusion is wrong; if T1 ≈ 1.0, chapter 6 needs a retraction. Either way
   a contradiction closes for almost nothing.

4. **The interior problem (4 lanes).** Run MT2 and GT on `drivaer_ml_volume`
   (or `highlift_volume`) with bare interior query points. This is the stated
   problem class. It also decides whether MT2 *can* address it: MT2's
   accuracy was shown to depend on the query-side token stream
   (`#sec-nb-v5-line-verdict`), which volume queries at 10⁶ points cannot
   afford; a 3× penalty at query-independence would then be the product's
   real operating point.

5. **Seeds at the decision rungs (≈12 lanes).** Five seeds for GT and MT2 at
   G2 rungs n=54 and n=218 and for both HL lr-1e-3 arms. Every "curve",
   "slope", "parity" and "robustness" verdict since 2026-08-20 is two-seed;
   the program's own H-SEED lesson (`#sec-nb-hseed-summary`, "the seed
   outranked the architecture") says three seeds hide heavy tails.

6. **AB-UPT as a measured baseline (4 lanes).** Declared the live threat on
   2026-08-20; still absent from every table. Also the natural comparator for
   the query-independence price (C11), which is currently a comparison of
   MT2 against itself.

7. **A model-free label-noise handle for DrivAerML.** N1 could not separate
   label noise from shared inductive bias. The mirror-antisymmetry statistic
   (L1) gives a lower bound (<1%). A time-window split of the transient
   averages or mesh-refinement pairs would give an upper bound; neither is in
   the shipped dataset, but asking the dataset owners costs an email and
   would settle whether the 1.085× gap is even resolvable.

8. **Declare the right drive degree, or drop the contract (writing + 2
   lanes).** For coefficient targets, degree 0; for dimensional targets,
   degree 2. If HiLiftAeroML varies freestream speed (Mach/Re) across cases,
   MT2 currently discards that information (`model.py` line 423–424 keeps
   only the direction) while GT receives the full `U_inf` — an input
   asymmetry that could hide a real MT2 disadvantage or advantage on the one
   dataset with a live amplitude axis. Check the metadata; if it varies, add
   `|U_inf|` (or Re) as a declared scalar and re-read HL.

9. **Theory the program invokes but does not use.** The literature wave
   (`#sec-nb-lit-synthesis` item 1) leaned on results (Elesedy–Zaidi,
   Tahmasebi–Jegelka) that predict a decaying equivariance advantage and
   called the G1 crossing an "anomaly forbidding a sign flip". Those results
   presuppose a correctly declared symmetry group. MT2 declares scale
   equivariance it does not have and omits reflection it should have
   (for an inviscid/RANS car, reflection covariance holds; L1 measured the
   labels 99% covariant). The "anomaly" is not an anomaly under the model's
   actual symmetry group; the theory was applied to the wrong object.

10. **Reflection augmentation for GT and mirrored-pose P1.** The P1 orbit
    uses proper rotations only. A mirror probe on GT (free, eval-only) and
    on MT2 default (which is not covariant) would quantify how much the
    parity line is worth *before* running W2.

---

## 5. Methodological critique

**Preregistration practice.** The discipline is real and unusually
transparent: preregistrations exist as dated JSON with hashes, failed
predictions are recorded, and the notebook discloses its own errors (N1 and
L1 threshold miscalibration; the true7 derivation error). Three weaknesses
survive peer review poorly:

- *Decision rules that do not partition the outcome space.* The HL H1 rule
  (`highlift_wave_preregistration_2026-08-17.json`) covers "larger gap" and
  "similar gap with higher absolute errors"; the observed outcome (similar
  gap, *lower* absolute errors) is in neither branch, and the campaign review
  then chose the noise-floor interpretation anyway (`#sec-nb-campaign-review`),
  which N1 had to retract a day later. The mt2_stage0 preregistration left the
  [1.30, 1.50] band unassigned (`#sec-nb-mt2-stage0b-verdict`).
- *Mid-flight amendments that add a selection step.* The HL lr amendment
  (`#sec-nb-hl-lr-amendment`) is defensible — it was written before any
  1e-3 result — but "each architecture reports its best over {3e-3, 1e-3}"
  with two seeds is a max over noisy draws; with two lr points per
  architecture the winner is decided by which architecture happened to have
  its optimum closer to a grid point.
- *Thresholds in the wrong units*, acknowledged twice
  (`#sec-nb-n1-verdict`, `#sec-nb-l1-verdict`). The fix stated there
  ("state thresholds in the readout's units and derive them") is the right
  one; it should also apply retroactively to M1's ">5% at n=218", which was
  a threshold on an arm that could not move by construction (§1 C5).

**Statistical power.** Two seeds per cell for G1/G2, LOC, M1, HL, R1, V6,
W1, W1', L2 and (planned) W2. Measured two-seed spreads: MT2 zero-shot 0.29
at n=109; GT estate 0.107 at n=14; GT HiLift at lr 3e-3 {0.148, 0.397}. No
paired test, confidence interval or effect-size statement appears anywhere in
the book for the 2026-08 campaign; `17-program-review.qmd` uses the word
"seed" only in the LOC bullet. The campaign review lists "two-seed statistical
power stated plainly in the book" as still open (`#sec-nb-campaign-review`);
it should be a blocker for every "curve" verdict, not an item.

**Instrument validity.**

- The frozen instrument is *one* 48-case validation split of one dataset at
  one sampling resolution, with a mean-predictor at 1.0 by construction of
  the normalization. Every architectural decision since 2026-08-04 was made
  on it. The program calls this a strength ("never widened retroactively",
  `17-program-review.qmd #sec-review-instrument`); it is also a single-split
  overfitting risk that eleven MT2 iterations, fourteen "review-hardening"
  lanes, and the instrument wave all report against.
- P2 measures the identity `out × |d|` (C7). P3's "exact HT weights" enter a
  model whose centering ignores them (C6). P4 was declared an instrument
  artifact for two of four models (`#sec-nb-v5a3-pilot-p4`) and never
  re-run.
- The HiLift normalization statistics are marked in the dataset yaml as
  "TODO: recompute stats for cell-centered data (these are vertex-projected
  estimates)" (`datasets/highlift_surface.yaml`); rel-L2 on mis-normalized
  fields is still a ratio, but the mean-predictor level is then not 1.0.

**Fairness of the baseline comparison.** The steel-manning is mostly
genuine (community config, own lr preference, retuned grid). Three
asymmetries are not discussed:

- GT receives `U_inf` (with magnitude); MT2 receives only its direction
  (`mt2_surface.yaml` vs `geotransolver_surface.yaml`). Inert on DrivAerML,
  potentially live on HiLift.
- GT receives no measure weights and cannot; MT2's P3 advantage/disadvantage
  therefore compares a weight-consuming model to a weight-blind one, and the
  weight-blind one wins.
- The rotation-augmentation arm is the only remedy tested for GT's pose
  hole (item 1 of §4); a reviewer will ask for the canonicalization control
  first.

**Reproducibility of the record.** The HL MT2 dataset config and the HL
ladder preregistration are not in the repo (§2 item 10). The frontier
table's VRAM numbers are inconsistent with the notebook (§2 item 5). The
book's own standing caveats (`17-future.qmd` "Standing caveats, restated")
end at the 2026-07 campaign; nothing equivalent exists for the 2026-08
campaign's two-seed cells and single-split instrument.

**What would not survive peer review as written:** the "exact similarity
equivariance" claim (C4), the M1 and M4 closures (C5, C6), the degree-1
merit (C7), the frequency-coverage conclusion of G3 (C15), the
"equal-or-lower cost" sentence (C16), the "1.6× more accurate at equal pose
robustness" sentence without a canonicalization control (C3), and any
"curve" statement on two seeds (C8, C9, C12, C13).

---

## 6. Recommendations (prioritized)

1. **Reframe the product claim (framing change, no compute).**
   *Question:* what is MT2, stated so every sentence survives §1? *Proposed
   statement:* "MT2 is a soft-slice point transformer over
   rotation/translation-invariant features with a measure-weighted assignment
   and a GLOBE-style vector head. It is exactly yaw/rotation-equivariant
   (SO(3)); it is **not** scale-equivariant or reflection-covariant by
   default; its drive-degree contract is inert on unit-direction inputs. On
   DrivAerML it is within 1.09–1.20× of GeoTransolver in-family at ~2× the
   training memory, has no single-family zero-shot transfer, is 3.7× more
   density-sensitive than GT, and is markedly more learning-rate robust. On
   HiLiftAeroML it is at parity (2 seeds)." Retire: "similarity
   equivariance", "measure-complete", "exact drive degree" as differentiators,
   "equal-or-lower cost", "transferring at parity", and the label-noise
   attribution. Update `02-architecture.qmd`, `17-program-review.qmd`,
   `index.qmd` in one commit (the architecture chapter's own rule).
   *Falsifier:* none — this is bookkeeping the artifacts already demand.

2. **Run the GT canonicalization control (eval-only, hours).**
   *Question:* does GeoTransolver recover P1 for free by aligning the frame to
   U_inf? *Falsifier for the program's headline:* GT-canonicalized P1 ratio
   ≤ 1.02 at in-family 1.00× → the "augmentation price" and "1.6× at equal
   pose robustness" claims are withdrawn and the yaw-orbit result becomes
   "MT2 does natively what GT does with a two-line wrapper".

3. **Make MT2 actually similarity-covariant and re-measure (code + 3–4
   lanes).** Weighted centroid, measure-weighted RMS-radius gauge (with the
   per-dataset override retained as an explicit, declared *per-case* length),
   scale test added to `test_mt2.py`. Retrain v3c (3 seeds), read in-family,
   zero-shot (estate/fastback), P3. *Question:* does the constant gauge and
   unweighted centering explain part of MT2's zero-shot flatness and P3 floor?
   *Falsifier:* zero-shot ≥ 0.90 and P3 ≥ 12× with in-family unchanged → the
   gauge is not the family-binding mechanism (as with MT1) and M4's
   information-limited reading is reinstated on proper evidence.

4. **Re-run G3 with the kernel-decoder MT1 and a kernel readout for MT2
   (< 3 GPU-hours).** *Question:* is frequency-content generalization
   purchasable by exact singular structure, as chapter 6 claims and G3 denies?
   *Falsifier for chapter 6:* kernel MT1 T1 > 0.5. *Falsifier for G3:* kernel
   MT1 T1 < 0.15. Whichever fires, one chapter changes; if chapter 6 holds,
   the exact dictionary re-enters the MT3 design brief as the only mechanism
   the program has ever shown to extrapolate a BC axis.

5. **Take the stated problem seriously once: volume targets (4 lanes).**
   *Question:* can MT2 (and GT) map surface → interior field at bare query
   points at 10⁵–10⁶ queries, and at what accuracy relative to the surface
   task? *Falsifier for the product framing:* MT2 requires query-side
   interaction to reach GT-class accuracy (as v5 showed) and the volume
   query count forces the passive decoder → the interior product runs at the
   measured 3× penalty. If so, the book's problem statement should be
   rewritten to the surface-trace task the program actually addresses.

6. **Power before curves (≈12 lanes).** Five seeds at G2 n=54 and n=218 for
   both architectures and at HL lr 1e-3 for both; pre-declared paired
   statistic (sign test or bootstrap CI on the seed-mean difference).
   *Question:* which of "GT transfer compounds", "MT2 flat", "HL parity", "MT2
   lr-robust" survive at n=5? *Falsifier:* any verdict whose CI includes the
   null is downgraded to "not measured" in the review chapter.

7. **Fix the drive contract or retire it (writing + at most 2 lanes).**
   Declare degree 0 for coefficient targets (or 2 for dimensional ones);
   check whether HiLiftAeroML varies |U_inf| and, if so, give MT2 the scalar
   GT already gets and re-read HL. *Falsifier:* if |U_inf| is constant on
   both datasets, remove the "exact drive degree" claim from every industrial
   statement; it is unfalsifiable there.

8. **Cap the parity line at W2 and measure AB-UPT (4 lanes).** W2 answers
   whether an odd-coefficient head closes the wall-shear penalty; whatever it
   says, six arms on a ≤26% wall-shear effect is enough. AB-UPT has been the
   declared strongest competitor since 2026-08-20 and is the natural
   comparator for the query-independence price. *Falsifier for the program:*
   AB-UPT at matched VRAM is within 1.1× of GT and query-independent by
   construction → the "accuracy and query interaction are entangled in this
   design family" finding is a property of this design family only, and the
   family should change.

---

## 7. Verdict

The program's discipline — preregistration, dated verdicts, disclosed
errors, frozen artifacts — is well above the field's norm, and several of its
negative results (query-independence price, coverage-vs-generalization
reframing, the futility of frame construction) are genuinely useful. But the
positive claims that carry the product framing do not hold up against the
program's own code and artifacts: MT2 is not scale-equivariant, is not
measure-complete, is not at equal cost, has no single-family transfer, and
its remaining differentiators (yaw robustness, lr robustness, a tie on
HiLift) rest on two seeds and lack the cheap controls (canonicalization,
five seeds) a referee would demand first. The program has hill-climbed an
in-distribution ratio on one 48-case split for a month and, in doing so,
walked away from the one mechanism it ever showed to extrapolate (the exact
singular dictionary on the frequency axis) and from the problem class it
says it is solving (boundary → interior). The single biggest risk is not any
one wrong number; it is that the book's "source of truth" chapters now assert
contracts the model does not have, so every downstream inference built on
"exact similarity equivariance" — the M1 refutation, the G1 "anomaly", the
"what remains" hypothesis for the G2 gap, the MT3 design brief — is reasoning
from a false premise. Fix the premise (recommendations 1–3) before running
another lane on MT3.
