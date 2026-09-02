# Scoping a 3D exact-kernel arm for DrivAerML surface loads

Date: 2026-09-02. Status: design document, read-only analysis; nothing launched,
nothing committed. Companion to `critic_review_2026-09-02.md` (§3 "walked away
from the one mechanism it ever showed to extrapolate") and to the notebook
entry `book/18-notebook.qmd#sec-nb-g3-kernel-rerun` (kernel-decoder
MeshTransformer T1 = 0.070 on unseen drive modes vs 1.00 for the moment
decoder, MeshTransformer2 and the soft-slice baseline).

Every number below is quoted from a checked-in artifact or from code read in
this session; the file:line is given next to it. Where a statement is my
inference or a prior, it says so.

---

## 0. Recommendation in five lines

1. **Do not re-train the MeshTransformer kernel decoder on DrivAerML surface
   pressure as "the 3D kernel arm".** That arm exists
   (`conf/model/mesh_transformer_surface_flagship.yaml`), was trained at
   scale, and its exact members were measured at 0–8% in-family and zero
   on zero-shot OOD (`book/16-drivaerml.qmd#sec-drivaerml-factorial`). Its
   failure is explained below and is structural to *surface-to-surface on a
   0.1%-coverage subsample*, not to 3D.
2. **Stage K0 (eval-only, zero parameters, ~3–4 engineer-days, <10 GPU-h):**
   a potential-flow oracle built from the repo's existing exact 3D
   single/double-layer triangle integrals on a *complete, closed, coarse
   remesh* of each vehicle; Neumann data `U_inf . n`; Morino second-kind
   solve; Bernoulli `Cp`. Read out on the frozen instrument and every
   G-ledger axis. Gate: DrivAerML val pressure rel-L2 < 0.90, else the
   premise "the exact kernel is the Laplace double layer, viscous effects are
   corrections" is refuted for this dataset and no learned kernel arm runs.
3. **Stage K1 (smallest learned arm, ~4–5 engineer-days, ~100 GPU-h):**
   the K0 field (`Cp_pf` scalar, tangential velocity `u_t` vector) supplied
   as a declared physical input to *both* MeshTransformer2-v3c and
   GeoTransolver at their best learning rates, 2 seeds each (4 lanes). The
   test is not in-family accuracy; it is SHIFT-SUV estate zero-shot
   (MT2-v3c 0.997, GeoTransolver 0.614 today) and the P3 / 5-resolution
   axes. Falsifier: neither architecture moves ≥ 0.10 on estate.
4. **Stage K2 (conditional on K1):** the exact kernel as a fixed,
   HT-weighted communication operator inside MeshTransformer2's slice
   blocks — the mechanism arm proper (formulation (b) below). Not before K1.
5. **Alternate smallest arm if the interior problem is admitted:** the
   existing kernel-decoder MeshTransformer, unchanged, on
   `drivaer_ml_volume.yaml` — the boundary-to-interior map is where the 2D
   mechanism actually lived and where its spectral leverage is largest.

Document path: `examples/cfd/mesh_transformer/results/kernel3d_arm_scoping_2026-09-02.md`.

---

## 1. What exists

### 1.1 Inventory

| Piece | Path | Dimension status | Evidence / notes |
|---|---|---|---|
| Exact 3D double layer over flat triangles (van Oosterom–Strackee signed solid angle) | `physicsnemo/experimental/nn/mesh_attention/kernel_decoder.py:619` `_triangle_double_layer_member` | **3D, done** | Row sums = −1 at interior points of a closed outward surface; `test/experimental/nn/mesh_attention/test_kernel_decoder.py:199,260` |
| Exact 3D single layer `∫ dS/(4π r)` over flat triangles (Hess–Smith / Newman per-edge `asinh` form) | `kernel_decoder.py:756` `_triangle_single_layer_member` | **3D, done** | Finite on-panel; `test_kernel_decoder.py:384,533` (adaptive quadrature; constant density on a sphere vs analytic) |
| Dimension dispatchers | `kernel_decoder.py:663` `exact_double_layer_member`, `:851` `exact_single_layer_member` | 2D+3D | `_EXACT_MEMBER_VERTICES = {2: 2, 3: 3}` (line 283) |
| Exterior-trace jump correction (+1/2 own-panel) | `kernel_decoder.py:896` `exterior_trace_self_entries` | dimension-independent | Constant +1/2 for any flat panel, either dimension |
| Dimension-generic Barnes–Hut opening angle `h/d = μ^{1/m}/d` | `kernel_decoder.py:546` `subtended_angle` | 2D+3D | |
| Source cache (panel vertices, normals, areas, HT `measure_factors`, coefficients, typed values) | `kernel_decoder.py:971` `KernelDecoderCache` | 2D+3D | `quadrature_measures = weights * measure_factors` (line 1020) |
| Dense operator-conditioned pair-kernel decoder, chunked, checkpointable, `self_indices` trace declaration | `kernel_decoder.py:1040` `KernelBasisCrossDecoder`, `forward:2371`; `LinearKernelBasisCrossDecoder:2509`, `NonlinearZeroKernelBasisCrossDecoder:2555` | 2D+3D | `test_kernel_decoder.py:1138` (singpair 3D trains, row-stable), `:3411` (checkpointing bitwise in 3D) |
| Barnes–Hut single-tree backend | `physicsnemo/experimental/nn/mesh_attention/barnes_hut.py` (`pair_triangle_double_layer:91`, `pair_triangle_single_layer:117`, `single_tree_partition:184`, `build_node_aggregates:346`, `segment_sum_by_query:445`), uses `physicsnemo/mesh/spatial/cluster_tree.py` | **3D only** (2D raises) | Fidelity ~θ^2.3, ≤1e-3 at θ ≤ 0.125; **6.2× slower than dense at 50k sources, no crossover through 100k** — a memory-bounded inference fallback, not a training speed-up (`results/notebook_full_chronology_through_2026-07-29.qmd#sec-nb-nochunk`) |
| The full model with `query_decoder="kernel"`, `trace_of`, `measure_normalization`, `kernel_decode_backend` | `physicsnemo/experimental/nn/mesh_attention/model.py:429` `MeshTransformer`; `encode` at `:2281` | 2D+3D | Center = measure-weighted centroid (`model.py:2334`); gauge = `reference_length_key` or intrinsic RMS radius; `MEASURE_WEIGHTS_KEY` on boundary `cell_data` survives into the source mesh (`model.py:2286-2307, 2337-2344`) |
| Recipe wiring of that model for DrivAerML surface | `examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf/model/mesh_transformer_surface{,_trace,_flagship,_allbc,_allbc_pooled}.yaml` | 3D, **trained at scale** | Flagship: `trace_of: vehicle`, singpair exact dictionary + 8 MLP members, `field_mode: homogeneous`, gauge 8.0, 2 operator layers, dims 64/16/96/24, ranks 96/32 |
| 3D synthetic exact-kernel BIE (2 parameters: kernel coefficient + Richardson relaxation, 8 steps) | `examples/cfd/mesh_transformer/problems/laplace3d_study.py:72` `SolidAngleBIE`, `SolvedSolidAngleOracle`; `problems/laplace3d.py:375` `solid_angle_influence` | 3D, all-Dirichlet interior only | Matches the dense oracle to 3e-4 on sphere/star (`book/09-laplace3d.qmd#tbl-3d`) |
| Kernel-decoder MeshTransformer on the 3D Laplace suite | `results/learned_bie_2026-07-02.json` keys `iteration_10_laplace3d_meshtransformer`, `iteration_12_dictionary_ablation_3d` | 3D, synthetic | Singular-only dictionary best: T1 0.0072, T2 0.0121, shell median 0.377 vs oracle 0.753 (`book/06-benchmarks.qmd#tbl-dictionary-3d`) |
| Learned 2D harmonic-panel BIE (13 params; the `learned_bie` line) | `examples/cfd/mesh_transformer/models/self_consistent_kernel.py:530` `HarmonicPanelBIE`, `:806` `NeumannHarmonicPanelBIE` | **2D only** | Laurent basis `Re(ζ^k)`, `ζ = n·r + i n×r` — complex-plane construction; artifact caveat[0]: "the harmonic Laurent basis is Laplace-specific (3D analogue: solid harmonics × Legendre in n·r̂ with exact triangle quadrature)"; caveat[1]: "the c1 term is the classical double layer so standard FMM/H-matrix acceleration applies verbatim" |
| Analytic 2D layer-potential controls | `models/layer_potential.py` (`_validate_boundary` requires `n_spatial_dims == 2`) | **2D only** | |
| 2D conformal-Laplace and 2D potential-flow suites | `problems/conformal_laplace.py`, `problems/potential_flow.py` (exterior conformal map `G(z) = z + Σ b_m z^{-m}`, circle theorem, pseudoscalar streamfunction), `studies/g3_bc_generalization.py` | **2D only** | The G3 result that motivates this document is on this suite |
| Pseudoscalar `0o` sector | `MeshTransformer(drive_pseudo_dim>0)` requires `n_spatial_dims == 2` (`model.py:1051` docstring, `:1423` construction-time error) | **2D only** | Irrelevant to 3D scalars/vectors; in 3D the parity issue is MT2's `e_phi` pseudovector basis (`physicsnemo/experimental/nn/mt2/model.py:260,304-327`) |
| Screened-Laplace panel BIE (Bessel `K0/K1`, Gauss-16) | `problems/screened_laplace.py:351` `ScreenedPanelBIE` | 2D only | |
| GLOBE | `physicsnemo/experimental/models/globe/model.py:66`; `conf/model/globe_surface.yaml` | 3D, in recipe | *Learnable* "Green's-function-like" MLP kernels (`MultiscaleKernel`), not exact — not an exact-kernel arm |
| Mesh tooling a kernel arm needs | `physicsnemo/mesh/remeshing/_remeshing.py:212` `remesh(mesh, n_clusters)` (Warp mass-weighted centroidal clustering, BVH projection to the source surface, compact triangle reconnection); `physicsnemo/mesh/repair/` (`orientation.py:146` `fix_orientation`, `degenerate_removal.py:30` `remove_degenerate_cells`, `hole_filling.py:142` `fill_holes`, `pipeline.py:28` `repair_mesh`); `physicsnemo/mesh/spatial/sdf.py:185` `signed_distance_field` + `physicsnemo/mesh/generate/marching_cubes.py:29` `marching_cubes`; `physicsnemo/mesh/spatial/bvh.py:260` `BVH` (`find_candidate_cells:609`; no closest-point routine — assemble one from the SDF's point-to-triangle distance path or Warp's mesh query) | 3D | **No watertightness check exists anywhere in the aero recipe** (grep "watertight" over `physicsnemo/` and the recipe: nothing) |

Summary of the dimension split: **everything that evaluates or propagates an
exact Laplace kernel is already 3D-capable and tested**; everything that
*solves* an exact BIE with a learned basis (`HarmonicPanelBIE`, the analytic
controls, the conformal generators, the G3 suite itself) is 2D-only and
tied to the complex plane. The 3D solve exists only in the 2-parameter
`SolidAngleBIE` (interior Dirichlet, synthetic).

### 1.2 What was already run in 3D on DrivAerML, and what it measured

The MeshTransformer kernel decoder (formulation (c) of §3 — model emits a
per-panel density, the exact operator propagates it) ran on DrivAerML surface
pressure/wall shear through the unified recipe in 2026-07. The record:

- **Kernel factorial at flagship rung, 3 seeds, one mechanism toggled**
  (`book/16-drivaerml.qmd#sec-drivaerml-factorial`): both members 0.0145,
  exact-only 0.0146, MLP-only 0.0157 (2026-07 val metric, *not* rel-L2;
  see caveat below). Pre-registered falsifier fired (MLP-only < 1.25× the
  flagship). Zero-shot S1/S2 rider: exact-only 1.11–1.13× vs MLP-only
  1.11–1.14× — no robustness difference. Chapter's own words: "the
  exact-kernel demotion at this rung is total".
- **Trace factorial** (`#sec-drivaerml-trace-factorial`): the dominant
  mechanism was boundary *self-identification* (the +1/2 jump constant or a
  learned own-cell readout, near-perfect substitutes), 3.8× at flagship
  capacity.
- **Absolute accuracy vs GeoTransolver:** flagship s42 pressure rel-L2
  0.167 / wss 0.705 vs GeoTransolver 0.061 / 0.280 on `id_reference`
  (`#tbl-crossfamily`); the 9.6M ×2L8 rung 0.107 / 0.479. On the frozen
  2026-08 instrument the MeshTransformer homogeneous arm reads 0.133–0.144
  (`results/mixedres_training_2026-07-31/sweep/p3v2_reduction.json` `p3v2_unif/homog_fixed_seed*`) vs
  GeoTransolver 0.053.
- **Resolution sweep on one unchanged checkpoint**
  (`#tbl-coverage-cliff`): 2.5k 0.897, 5k 0.800, **10k 0.167**, 20k 0.411,
  **40k 24,811** — a sharp minimum at the training resolution, catastrophic
  above it. The chapter's reading: the trained weights fit the systematic
  part of the *sampling* error at one rate ("coverage-specific regressor").
- **Cross-family zero-shot (SHIFT-SUV estate)** (`#tbl-crossfamily`): every
  MeshTransformer checkpoint diverged (8×10^4 to 1.3×10^15, one NaN) while
  GeoTransolver read 0.677 through the identical pipeline; the constant gauge
  8.0 was the localized culprit; the intrinsic-gauge repair did *not*
  restore transfer (`#sec-program-review`).
- **P3 biased density (exact HT, v3):** MeshTransformer homogeneous
  0.235–0.298 vs uniform 0.133–0.144 → **1.7–2.1×**; GeoTransolver
  0.192 vs 0.053 → 3.6×; MeshTransformer2-v3c 0.81–0.93 vs 0.063 →
  **13–14×** (`results/mixedres_training_2026-07-31/sweep/{p3v3_reduction,mt2_v3c_reduction}.json`).
  This is the one G-ledger axis on which the kernel-decoder arm is already
  the best of the three.
- **Cost:** flagship 784k params, 35.8 GB peak, ~12.8 h for 500 epochs on
  4× GB300 at 10k (`book/18-notebook.qmd:2553`) vs GeoTransolver 4.6 GB,
  ~3.6 h. The 2026-07-11 decode profile (`chronology#sec-nb-decode-profile`)
  attributed 7.4 s of an 8.3 s step to an encoder dispatch storm, since
  fixed; the ×2 rung without chunking ran 0.59–0.66 s/step at 58 GiB
  (`#sec-nb-nochunk`).

Caveat on units: the chapter-16 "val" ledger (0.1019 → 0.0145 → 0.00583,
GeoTransolver control 0.00249) is on a different validation metric from the
frozen instrument's rel-L2 (GeoTransolver 0.053). The two are not
inter-convertible from the record; I use rel-L2 wherever both exist.

### 1.3 The 2D result this document is trying to transfer

`results/g3/g3_reduction_with_kernel.json`, 3 seeds, 2D conformal Laplace,
train drive modes 1–4, test T1 = modes 5–8 (`book/18-notebook.qmd#sec-nb-g3-kernel-rerun`):

| arm | T0 | T1 unseen modes | T2b ×4 amplitude | T3 geometry |
|---|--:|--:|--:|--:|
| mt1_kernel (singular-only exact double layer, linear mode) | 0.0158 | **0.0701** | 0.0154 | 0.0232 |
| mt1_linear (moment decoder) | 0.0723 | 1.0043 | 0.0632 | 0.0898 |
| mt2_bscalar (MeshTransformer2) | 0.0624 | 1.1485 | 0.7289 | 0.0669 |
| softslice_2d (Transolver-class) | 0.1114 | 1.2090 | 0.7760 | 0.1221 |

Same axis, same suite, different protocol: `book/06-benchmarks.qmd#tbl-transolver`
singular-only 0.0748 freq-OOD vs Transolver-native 0.46 at 10× steps.
Also `results/learned_bie_2026-07-02.json` `resolution_transfer_seed17`:
trained at 64 panels, error falls monotonically 0.183 → 0.127 → 0.068 →
0.038 at 32/64/128/256 panels — quadrature convergence under *refinement*.

Three properties of that setting matter for what transfers:

1. **Boundary-to-interior.** The query is off the boundary. For Dirichlet
   data of angular order `m` on a disk of radius `R`, the interior response
   at radius `r` is `(r/R)^m`: the kernel carries a mode-dependent
   attenuation that a learned router must have seen. That is the "spectral
   response for free".
2. **Complete boundary.** Every panel is present; the exact panel integrals
   sum to the exact operator; refining the boundary converges.
3. **Linear PDE, linear drive.** The encoder's job is the second-kind solve
   `(1/2 I + K)^{-1}`, a smooth compact perturbation of the identity.

None of the three holds on the DrivAerML surface task as currently posed.

---

## 2. Why the exact kernel was null on DrivAerML, and what that predicts

**(i) Surface-to-surface removes most of the spectral leverage of the double
layer.** With queries on the source surface, the double-layer operator's
trace is `±1/2 I + K` with `K` compact (weakly singular, `~κ/r`, on smooth
patches). The `±1/2` is a constant and the trace factorial found exactly that
constant (or a learned substitute) was worth 3.8× while the rest of the
dictionary was worth 0–8% — the measurement is what the operator theory
predicts. The mode-dependent response that generalized frequency content in
2D is in the *propagation* `boundary → interior`, and in the single layer's
Neumann-to-Dirichlet action (on a sphere, `Y_lm → −R/(l+1) Y_lm` for the
exterior problem) — neither of which the MeshTransformer surface arm
exercised: its single layer received a learned density, not the known
Neumann data.

**(ii) A 10k-cell subsample of a 17.7M-triangle vehicle is not a boundary;
it is a Monte Carlo estimate of one.** Coverage 0.06–0.11%. An *exactly*
evaluated boundary integral on that sample carries ≈34% field-level error
(`book/06-benchmarks.qmd#sec-four-properties`, `book/16-drivaerml.qmd#fig-coverage-cliff`
grey curve: 0.337 / 0.248 / 0.174 at 10k / 20k / 40k); the trained model
read 0.167 at 10k, i.e. *better than its own quadrature*, which is only
possible by fitting the sampling noise at one rate — hence the 148,000×
cliff at 40k. The H-DI probe (`chronology#sec-nb-hdi-verdict`) showed the
same operator converging at second order under refinement (2.51% → 0.59% →
0.12%) and not converging at all under subsampling; the bias is a
*completeness* effect (222% at the sphere centre, 0.3% at the wall). The
exact kernel never had an exact operator to be exact about.

**(iii) The target is not linear in the drive and not harmonic.** Time-averaged
RANS-class pressure on a bluff body: massive base separation, underbody and
wheel flows, A-pillar vortices. The Laplace kernel is the *inviscid, attached*
part. Whatever a learned model adds on top is the part that dominates the
rel-L2 on a car.

**What this predicts for a new arm.** (a) In-family DrivAerML surface pressure
is the axis where an exact kernel is *least* likely to help; predictions
should not be written there. (b) Any arm that puts the exact operator on the
*subsample* inherits the 34% quadrature floor and the coverage-cliff risk;
the kernel's source set must be a **complete closed surface**, decoupled from
the 10k target sample. (c) The axes where exact structure can win by
construction are resolution transfer (quadrature converges; a fixed complete
source set is query-resolution-independent), density bias (an HT-weighted
kernel sum is design-unbiased; a fixed-source kernel field is
density-independent), and family transfer (the kernel has no family), which
are exactly the G-ledger axes on which MeshTransformer2-v3c is weakest
(13–14× P3; 0.997 estate).

---

## 3. The physics, and three candidate formulations

### 3.1 The exact kernel for incompressible external aerodynamics

Perturbation potential `φ`, total velocity `u = U_inf + ∇φ`, `Δφ = 0`
outside the body, `∂φ/∂n = −U_inf · n` on the body `S`, `∇φ → 0` at
infinity. With `G = 1/(4π r)`, the direct (Morino) formulation on the
exterior side of `S` is the second-kind equation

```
(1/2) φ(x) − ∫_S ∂G/∂n_y (x,y) φ(y) dS_y  =  ∫_S G(x,y) (U_inf · n_y) dS_y ,    x ∈ S,
```

i.e. `(1/2 I − K) φ_S = S [U_inf · n]`. Both operators are the repo's exact
members (`_triangle_double_layer_member`, `_triangle_single_layer_member`),
the own-panel entry of `K` is the exact principal value (zero for a flat
panel; `exterior_trace_self_entries` handles the trace side), and the
right-hand side is **known data**, not a learned density. Surface velocity is
`u_S = U_inf + ∇_S φ_S` (tangential gradient of the surface potential),
`Cp = 1 − |u_S|²/|U_inf|²`. Ground plane: image system (evaluate every
member at the query and at its mirror image across the ground plane — a
coordinate flip; no helper exists in `physicsnemo/mesh/transformations`,
none is needed). Pressure is nonlinear in velocity (Bernoulli) and the
whole map is exactly degree 0 in `|U_inf|` for `Cp` — which is the correct
declaration for coefficient targets (the critic's C7; MT2's degree-1
bypass is the wrong law here).

What potential flow cannot represent: separation and base pressure (it
predicts full pressure recovery at the rear stagnation point), wake
asymmetry, viscous drag, wall shear (`τ ≡ 0`), rotating wheels, engine-bay
flow. For DrivAerML rel-L2 these are not corrections; they are most of the
signal in the rear third of the car. For HiLiftAeroML (compressible;
dataset yaml declares `temperature`/`density` fields; freestream in
slug-inch-second-Rankine; Mach not recorded in the checked-in yaml) the
potential model additionally needs a **Kutta condition and wake doublet
sheet** from the trailing edges, or it predicts zero lift; at landing Mach
a Prandtl–Glauert factor is a small correction but the wake sheet is not
optional. HiLift is therefore a stage-2 target, not a stage-0 one.

Units and normalization: the recipe's `NonDimensionalizeByMetadata`
(`datasets/drivaer_ml_surface.yaml`) produces the pressure field the metric
sees; the oracle must emit exactly that quantity (check the transform's
formula — `(p − p_inf)/(ρ U²)` vs `Cp` differ by a factor 2 and the rel-L2 is
scale-sensitive against a fixed target). Wall shear is normalized by a fixed
`std 0.00313`.

### 3.2 Candidate formulations

| | (a) Potential-flow kernel, learned correction | (b) Exact kernel as fixed communication operator in MT2 | (c) Learned density, exact propagation (= MeshTransformer kernel decoder) |
|---|---|---|---|
| What the exact operator acts on | Known Neumann data `U_inf·n` on a complete closed remesh | Learned per-token states, HT-weighted, on the 10k token set (or pooled to the remesh) | Learned per-panel density (typed values) from boundary attention |
| Learned part | Correction to `Cp_pf`, `u_t` (MT2 or GT, unchanged architecture) | Everything except the operator's kernel | Encoder (attention) and coefficients |
| Buys by construction | Family-agnostic, query-resolution-flat, density-independent inviscid field; exact similarity + O(3) invariance of the field; degree 0 in `|U|` | Design-unbiased global exchange under biased sampling (HT weights); resolution-convergent aggregation; SE(3) exact; a global-context channel MT2's centering lacks | Query independence (bitwise); exact linearity if declared; interior queries at no extra structure |
| Cannot capture | Separation, shear magnitude, wake; only what a correction learns | Nothing physical is imposed; it is a routing prior | Same as MT1: measured 0–8% on in-family surface pressure; coverage cliff on subsamples |
| Cost at 10k / 40k | Precompute per case: assemble `K`,`S` on the remesh `N_c ≈ 10–20k` (`N_c²` = 1–4×10^8 pairs, sub-second on GPU), one dense or GMRES solve (`N_c³/3 ≈ 3×10^12` flops fp64, seconds), evaluate at any number of queries `O(N_q N_c)`. 483 + 100 + 1804 cases → hours once, cached to disk as a cell field | Per forward: `S`,`D` on `N_tok × N_tok` = 10^8 pairs (fp32 400 MB each) computed once per sample and shared across layers; at 40k tokens 6.4 GB per matrix — needs chunked matvec without materializing, or `barnes_hut.py` (memory-bounded, slow) | Measured: 35.8 GB / 12.8 h at 10k for 784k params; 20.6 s/step dense at 50k; chunking mandatory |
| Wall-shear vector head | `τ = α û_t + β (n × û_t)` with invariant `α`, parity-odd-gated `β` (same construction as MT2 `parity_fix`); `û_t` is a *flow-aligned* polar vector basis replacing the freestream-based `e_φ` complements; optional magnitude prior `|τ| ∝ |u_t|^{1.8}` (turbulent flat plate) | Vector channels propagate componentwise through the scalar kernel (equivariant); a polar-vector channel `∇_x S σ` is available as `torch.autograd` of the exact single-layer closed form w.r.t. `query_points` — exact panel velocity influence with no new closed form | Typed vector values already exist (`KernelDecoderCache.value_vectors`); measured wss 0.705 vs GeoTransolver 0.280 |
| Status | Not built | Not built | Built, wired, trained; demoted at this task |

Notes on (b): the natural insertion point is `_SliceBlock.forward`
(`physicsnemo/experimental/nn/mt2/model.py:76-125`): alongside the softmax
slice aggregation `z_states = einsum(a, h)`, add `h ← h + W [S_w ρ(h)]`
with `ρ(h)` a linear map of the token states and `S_w` the HT-weighted
exact single-layer matrix over tokens (`w_j = A_j / π_j`, from
`interior.point_data._target_quadrature_measure`). It is linear in `h`, so
autograd through it is one extra matmul; the matrix is geometry-only and
rotation-invariant, so it is computed once per sample in the forward. The
double layer adds a second, orientation-odd channel. The 34% quadrature
floor still applies to this operator on the 10k sample; that is the
reason to run (a) first — (a) sidesteps it by construction.

Notes on (c) for the interior problem: `drivaer_ml_volume.yaml` and
`highlift_volume.yaml` exist; the kernel decoder needs no change; volume
queries at 10^5–10^6 points make dense decode `O(N_q N_s)` — at 10^6 × 20k
= 2×10^10 pairs, minutes per sample forward, which is the regime where
`barnes_hut.py` was measured to pay (memory-bounded, linear in queries,
~20 min/sample at 8.8M queries × 50k sources).

---

## 4. Recommendation and preregistration skeleton

### 4.1 Hypothesis under test

**H:** the exact-kernel mechanism that generalized boundary-condition
frequency content in 2D transfers to 3D external aerodynamics, measured as
improvement on the G-ledger axes (resolution, density, family transfer) at
non-inferior in-family accuracy — *not* as in-family accuracy.

**Premise P (tested first, for free):** the inviscid Laplace field explains a
non-trivial fraction of DrivAerML surface pressure, so that "viscous effects
are corrections" is a defensible description of the residual.

### 4.2 Stage K0 — potential-flow oracle (eval-only)

**Build.** For each case: (1) `remesh(vehicle, n_clusters≈8k)` → ~16k
triangles; `fix_orientation`; `remove_degenerate_cells`; `fill_holes`;
fallback for non-closed results: `signed_distance_field` → `marching_cubes`
at ~2 cm → `remesh` to the same count. (2) **Gauss gate per case**: with unit
density the double layer sums to −1 at three interior probe points and to 0
at the exterior trace of every query; reject/repair the case if
`|Σ_j D_ij + 1| > 1e-3` (this is `test_exact_member_interior_gauss_identity`
promoted to a data-quality gate). (3) Assemble `K`, `S` with the existing
members in fp64; solve `(1/2 I − K) φ = S[U_inf·n]` (dense LU or GMRES);
ground plane by images. (4) `∇_S φ` by per-face least-squares over the
1-ring; `Cp`, `u_t` at coarse-face centroids; map to the 10k query
centroids by nearest face (closest-point query: `BVH.find_candidate_cells`
plus exact point-to-triangle distance, or the SDF path) with barycentric
interpolation. (5) Emit
predictions through the recipe's `metrics.py::_relative_l2` (area/HT
weighted) so the readout is the instrument's.

**Validation before any DrivAerML number:** sphere in uniform flow
(`Cp = 1 − (9/4) sin²θ`), prolate ellipsoid (analytic), one DrivAerML case
against a coarse/fine remesh pair (resolution convergence of the oracle
itself).

**Readouts and predictions (readout units: rel-L2 as `metrics.py`).**

| axis | comparators today | K0 prediction | falsifier / decision |
|---|---|---|---|
| DrivAerML val pressure (48 cases, 10k) | GT 0.0532 (best lr 3e-3), MT2-v3c 0.0577 (best lr 1e-3); frozen-lr 5-seed 0.0531 / 0.0640 | **0.55–0.80** (prior: front/hood/windshield/roof captured, base/underbody/wheels not) | **≥ 0.90 → P refuted**; stop the kernel line on DrivAerML surface, write the negative. ≤ 0.50 → stronger than expected; K1 prediction bands tighten |
| 5-resolution sweep 2.5k/5k/10k/20k/40k | GT 0.0572/0.0538/0.0530/0.0529/0.0530; MT2-v3c s42 0.0892/0.0702/0.0636/0.0619/0.0611; MT1 flagship 0.897/0.800/0.167/0.411/24,811 | **flat: max−min ≤ 0.01** (only the query set changes) | a slope indicates a mapping/near-field defect in the pipeline, not physics; fix before K1 |
| SHIFT-SUV estate / fastback zero-shot (mean-predictor = 1.0) | GT estate 0.614 (5 seeds); MT2-v3c 0.997 estate / 0.99 fastback; MT1 diverged | **within ±0.10 of the DrivAerML value** (the kernel has no family) | if the SUV number is ≥ 0.15 worse than DrivAerML, the inviscid fraction is body-style dependent; record, does not block K1 |
| P3 (10:1 biased query set, exact HT) | GT 3.6×; MT2-v3c 13–14×; MT1-homog 1.7–2.1× | **1.00 ± 0.02** (sources fixed; only the weighted metric changes) | trivially true; it is the calibration of the probe against a sample-independent predictor |
| HiLift full split pressure | GT 0.0422, MT2 0.0413 (lr 1e-3) | **≥ 0.9 without a Kutta/wake model** (documented control) | not a test of H; establishes the wake-sheet requirement |
| wall shear | GT 0.0825, MT2 0.0844 | no magnitude prediction (inviscid). Diagnostic: area-weighted mean cosine between `û_t` and the label direction on the top-50%-`|τ|` cells; prior ≥ 0.7 | if < 0.5, the flow-aligned basis of K1's vector head is dropped |

**Cost.** Remesh + repair + Gauss gate ~8 h; solve + gradient + Bernoulli +
mapping ~12 h; analytic validation ~3 h; readout harness reusing the N1
per-point-prediction path ~6 h. Runs: ~2,400 cases × 5–30 s → < 10 GPU-h
(one node, no training). **~3–4 engineer-days, < 10 GPU-h.**

### 4.3 Stage K1 — the smallest learned arm

**Arms (4 lanes, 500 epochs, 10k, bf16, no augmentation, the frozen protocol):**

- `mt2_v3c_pf`: MeshTransformer2-v3c at lr 1e-3, `boundary_scalars = [Cp_pf]`
  (the `n_boundary_scalars` path already exists — the G3 `mt2_bscalar` arm),
  vector head basis extended with `û_t` and `n × û_t` (parity-odd gated),
  seeds 42/43.
- `gt_pf`: GeoTransolver at lr 3e-3, `local_embedding` extended with
  `[Cp_pf, u_t]` (6 → 10 features), seeds 42/43.
- Comparators: the existing best-lr checkpoints `iw_mt2_lr1e3_seed4{2,3}`
  and `ref_gt_fixed_seed4{2,3}` (`results/instrument_wave_reduction_2026-09-01.json`).
- Should the S1 `similarity_gauge=True` arm land first, K1 runs on top of it
  (otherwise the P3 prediction below is confounded by the unweighted centroid).

**Predictions in readout units.**

| axis | MT2-v3c today | GT today | K1 prediction | decision rule (partitioned) |
|---|---|---|---|---|
| DrivAerML val pressure | 0.0577 | 0.0532 | **|Δ| ≤ 0.004** for both (within 2-seed spread; structure buys OOD, not ID — the 2D lesson) | Δ ≤ −0.006 for either: report as a bonus, not the test. Δ ≥ +0.006: the field hurts in-family; investigate normalization before reading OOD |
| SHIFT-SUV estate zero-shot | 0.997 | 0.614 | **MT2+pf ≤ 0.85; GT+pf ≤ 0.55** | (i) both move ≥ 0.10 → the inviscid field is family-portable → K2 registered. (ii) only MT2 moves → the field substitutes for global context MT2's invariants lack; MT2-specific, K2 registered with that reading. (iii) only GT moves → GT can exploit the field, MT2 cannot ingest it → an MT2 ingestion defect, not a physics verdict; fix and rerun before K2. (iv) neither moves ≥ 0.10 → **H refuted at the feature level; K2 not launched** |
| SHIFT-SUV fastback | 0.99 | — (GT single-family not in record; multi-family 0.110) | MT2+pf ≤ 0.85 | same rule as estate; both families must agree in sign |
| P3 exact-HT ratio | 13–14× | 3.6× | MT2+pf ≤ 8× (on v3c); ≤ 2× if on `similarity_gauge=True` | ≥ 12× on v3c → the field does not offset the centering defect; no reading on H |
| 5-resolution sweep | 0.089 → 0.061 | 0.057 → 0.053 | MT2+pf at 2.5k ≤ 0.080; ratio 2.5k/40k ≤ 1.30 (from 1.46) | flat curve (≤ 1.10) → the field carried the resolution-dependent part |
| wall shear | 0.0844 | 0.0825 | MT2+pf wss ≤ 0.080 if the flow-aligned basis is admitted; falsifier: ≥ 0.084 | this is the only head change; the scalar path is untouched |
| HiLift | 0.0413 | 0.0422 | not run in K1 (no wake model) | — |

**Statistical discipline.** Two seeds per cell is the program's habit and
the critic's standing objection. The decision rules above are written so
that a 0.10 move on estate is 3× the largest observed MT2 two-seed spread
at n=435 (0.946–1.026) and 2× GT's (0.556–0.641); anything under 0.05 is
"not measured" by rule.

**Cost.** Pipeline transform attaching `Cp_pf`, `u_t` from the K0 cache as
`cell_data` on `boundaries.vehicle` and `interior.point_data` (~8 h); MT2
head extension + contract tests (rotation, translation, reflection with the
gated pseudovector) (~8 h); GT feature width change (~3 h); configs and
sbatch (~3 h). Lanes: 4 × (3.6–5 h on 4× GB300) ≈ 60–80 GPU-h; G-axis evals
~10 GPU-h. **~4–5 engineer-days, ~100 GPU-h.**

### 4.4 Stage K2 — the mechanism arm (conditional)

Only if K1 rule (i) or (ii) fires. Insert the HT-weighted exact single- and
double-layer operators as a fixed exchange channel in each `_SliceBlock`
(§3.2 (b)), on the token set, one matrix pair per sample per forward. Arms:
`mt2_v3c_kop` (operator channel on), ablation with a random fixed
symmetric kernel of the same spectrum (controls for "any global linear
exchange helps"), 3 seeds each, best lr. Predictions written after K1's
numbers exist. Rough cost: ~2 weeks engineering (chunked matvec at 40k,
memory at 10k, tests), 6 lanes ≈ 120–150 GPU-h.

### 4.5 Why not other "smallest" arms

- *Re-run `mesh_transformer_surface_flagship` on the frozen instrument:*
  already on it as `homog_fixed_seed*` (0.133–0.144 in-family, P3 1.7–2.1×,
  cross-family seed-unstable over 2491×). Nothing new would be learned.
- *MeshTransformer kernel decoder on the complete remesh as source set,
  10k targets as queries (trace via nearest-face `self_indices`):* this is
  the cleanest test of formulation (c) without the subsample confound, and
  it needs only a data-pipeline change plus a small `trace_of` relaxation
  (currently requires query count = cell count, index-aligned;
  `model.py:trace_of` docstring). It is a reasonable K1-alternate for the
  interior problem (§0 line 5) but on the surface task it re-tests an
  architecture 2.5× behind GeoTransolver in absolute terms; K1 tests the
  kernel's *output* inside the two architectures the instrument compares.

---

## 5. Risks

### 5.1 Singular integrals and the jump branch

The closed forms are finite everywhere including on-panel
(`_triangle_single_layer_member` docstring; `_triangle_double_layer_member`),
but the double layer evaluated *exactly on* a panel returns an accidental
±1/2 branch of `atan2` (`exterior_trace_self_entries` docstring: measured on
DrivAerML as the *interior* branch). With a remeshed source surface, the
10k query centroids lie within remesh error (mm–cm) of `S_c`, on either side
at random: a query a hair inside gets the interior limit — a local error of
exactly 1 in the trace. **Mitigation (mandatory):** project each query onto
`S_c` (closest point on the nearest face), evaluate at the projected point with
`self_indices = nearest face`, so `exterior_trace_self_entries` serves the
exact +1/2. **Gate:** with unit density, `D[1]` must vanish at every
projected query to 1e-3 (exterior trace of a closed surface); this catches
side errors, orientation flips and leaks in one number per case.

### 5.2 Near field of the coarse surface

Queries are `O(h)` from panels of size `h`: the near field is exact by the
closed forms, but the *represented geometry* is the coarse remesh, not the
true surface; `Cp` from the coarse potential is smooth at scale `h`. Sharp
features (A-pillar, spoiler edges, mirror stalks) are lost below `h`. This
is a modeling error of the oracle, not a numerical one; it is why K0's
prediction band is wide and why the remesh count should be swept (8k, 16k,
32k triangles) on a handful of cases before fixing it.

### 5.3 Non-watertight and degenerate input

DrivAerML surfaces carry slivers to 1e-11 area (`src/domain_transforms.py:315-339`
`DropDegenerateCells`, wired last in every surface yaml); ~17.7M triangles per
vehicle; wheel contact patches and internal parts may leave open edges. No
watertightness check exists in the recipe. The remesher's output is not
guaranteed closed; the Gauss gate (§5.1) is the arbiter, and marching cubes
on the SDF is the always-closed fallback (at the price of thin features).
Expect a per-case repair loop; budget it (§4.2 cost). Orientation: the
double layer is odd in `n` and `_triangle_double_layer_member` reconciles
winding with supplied normals by sign, so a globally consistent orientation
(`fix_orientation`) is required but per-face flips are tolerated.

### 5.4 Measure weighting

In K0/K1 the kernel integrates over complete panels: geometric area is the
measure, no HT factor, no `measure_normalization` question. In K2 (operator
on the sampled tokens) the HT inverse-inclusion weights are both licensed and
necessary; the plumbing exists (`MEASURE_WEIGHTS_KEY`, `cell_measure_weights`,
`compose_measure_weights`, `KernelDecoderCache.measure_factors`,
`_target_quadrature_measure`). But the program has already shown that a
*learned* model on a Monte Carlo operator learns the rate-specific noise
(coverage cliff) and that "no reweighting reconstructs a missing global
integral" (`book/06-benchmarks.qmd#sec-four-properties`, with the
counter-example that correct HT weights *do* repair a decimated integral in
mean: 93.8% → 0.4%). K2 therefore either trains across resolutions or pools
token densities onto the complete remesh before applying the operator; this
is the design question K2's preregistration must answer, and one more reason
(a) precedes (b).

### 5.5 Contracts

- **SE(3):** exact for every formulation; the members are functions of
  relative vectors and normals only (module docstring, `kernel_decoder.py:202-208`).
- **Reflection:** the double layer flips with `n`, the single layer does
  not; `Cp` is a true scalar, `u_t` a polar vector, `n × u_t` a
  pseudovector — the K1 head's `β` coefficient must be parity-odd (MT2's
  `parity_fix` gate `r̂ · (n̂ × d̂)` is the existing pattern). Note the L1
  finding: DrivAerML labels are mirror-covariant only to 0.67–0.93% energy
  (`results/l1_label_covariance_2026-09-02.json`), and every exact
  reflection repair so far cost 6–19% on wall shear — the flow-aligned
  basis is a bet that a *physical* odd vector does better than a geometric
  one; K0's cosine diagnostic decides whether to place it.
- **Scale:** the double layer is dimensionless (solid angle); the single
  layer carries a length and is evaluated in the normalized frame. For
  potential flow the gauge is irrelevant: `Cp` and `u_t / |U_inf|` are
  exactly similarity-invariant with no reference length at all — the
  kernel field is the one input in the pipeline that the constant
  `reference_length: 8.0` (the localized cross-family culprit) cannot
  contaminate. The consumer architectures keep their own gauge problems
  (critic C4/C6; `similarity_gauge` S1 arm, `mt2/model.py:435-449`).
- **Drive degree:** `Cp_pf` is degree 0 in `|U_inf|`; MT2's degree-1 output
  bypass multiplies the correction by `|d|`, which is 1 on the recipe
  (unit `U_inf_dir`) — inert on DrivAerML, wrong in principle; declare
  degree 0 for the correction if the HiLift stage ever varies `|U_inf|`.
- **Query independence:** K0/K1 fields are per-point functions of the
  fixed remesh — bitwise query-independent; MT2 remains
  query-interacting (the measured 2.7–3.9× price of removing that is
  unchanged by K1).

### 5.6 Instrument and record hazards inherited

- The HiLift MT2 dataset config that ran is not in the repo (critic C12;
  `datasets/highlift_surface.yaml` lacks `U_inf_dir` and `reference_length`);
  a HiLift kernel stage must first recover it.
- The frozen instrument is one 48-case split; K1's in-family Δ is written
  as a *no-change* prediction precisely so it is not another hill-climb
  readout.
- The 2026-07 MeshTransformer ledger and the 2026-08 rel-L2 instrument are
  on different metrics; do not mix them in any K-stage table.

---

## 6. Pre-launch checklist

1. Confirm the exact formula in `NonDimensionalizeByMetadata` for
   `pMeanTrim` and reproduce it in the oracle's `Cp` output.
2. Sphere and ellipsoid analytic checks of the Morino solve to 1e-3 rel-L2.
3. Remesh count sweep (8k/16k/32k) on 5 DrivAerML cases; Gauss gate pass
   rate over the 483 cases; fallback rate to marching cubes.
4. Ground-plane image treatment on vs off (one case) — expected to matter
   on the underbody.
5. Write the K0 preregistration JSON in the `instrument_wave_preregistration_2026-09-01.json`
   format (`arms`, `predictions`, `falsifier`, `cost`, `declared_before_results`)
   with the bands of §4.2 *before* the first DrivAerML readout.
6. Decide K1's dependency on the S1 `similarity_gauge` arm; if S1 has not
   landed, write the P3 row of §4.3 as "not measured".
