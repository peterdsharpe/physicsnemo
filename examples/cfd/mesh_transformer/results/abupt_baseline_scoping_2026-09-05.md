# Scoping an AB-UPT / Transolver-class baseline for the frozen instruments

Date: 2026-09-05. Status: design document, read-only analysis; nothing launched,
nothing committed. Companion to `critic_review_2026-09-02.md` (§4 item 6 and
§5 item 8, "AB-UPT as a measured baseline, 4 lanes") and to the notebook entries
`book/18-notebook.qmd#sec-nb-lit-synthesis` (item 4: "AB-UPT joins the
measured-baseline set") and `#sec-nb-hl-ladder-dataeff` (the data-efficiency
headline this baseline is meant to stress).

Every number below is quoted from a checked-in artifact or from code read in
this session; the file:line or anchor is given next to it. Where a statement is
my inference or a prior, it says so.

---

## 0. Recommendation in five lines

1. **PhysicsNeMo has no AB-UPT.** The only "anchored" code is MeshTransformer2's
   own v5a4 anchor-conditioned decode (`physicsnemo/experimental/nn/mt2/model.py:268`),
   and AeroJEPA's prototype-anchor tokenizer (`physicsnemo/experimental/nn/point_tokenizer.py`),
   neither of which is a UPT. A **plain Transolver** is fully wired into the
   recipe today (`conf/model/transolver_surface.yaml` ->
   `physicsnemo.models.transolver.Transolver`), untested on HiLift.
2. **Run the plain-Transolver control first, at the 35-case HiLift rung.** It
   is the same attention family as GeoTransolver with no geometry encoder, no
   ball-query local features and no contracts, costs ~0.5 engineer-day plus
   4 small lanes, and is the cheapest arm that can refute the current
   interpretation ("the data-efficiency gap is MT2's geometric contracts"):
   if Transolver lands near MT2's 0.138 rather than GeoTransolver's 0.392, the
   gap belongs to GeoTransolver's geometry encoder, not to MT2's contracts.
3. **Gate the AB-UPT implementation on that readout.** A faithful surface
   AB-UPT (anchor subset runs all token-token interaction; every query decodes
   passively by cross-attending to the anchors) is ~5-7 engineer-days in
   `physicsnemo/experimental/models/abupt/` with the recipe contract, because
   the repo already has FPS, RoPE, transformer encoder/decoder blocks and
   conditioning embedders. The repo's own artifacts describe the mechanism
   only at the level of the notebook (#sec-nb-lit-wave-prereg) and the v5a4
   code comment; the lit-review file records CarBench numbers, not the
   architecture, so the primary paper must be re-read before implementing.
4. **Frozen-instrument budget for the full baseline set** (both models, all
   three instruments): ~41 lanes, ~700-1,500 GPU-hours on GB300 at the
   program's measured per-lane costs (§4); the Transolver half alone is
   ~20 lanes, ~350-750 GPU-hours. The refuting arm alone (Transolver at
   35 cases, 2 seeds x lr {1e-3, 3e-3}) is ~30-130 GPU-hours.
5. **Preregistered expectation (§3):** a Transolver-class model at 35 cases
   lands near GeoTransolver's 0.392 (band 0.30-0.50), not MT2's 0.138.
   Falsifier: AB-UPT (or Transolver) <= 1.3 x 0.138 = 0.18 at 35 cases ->
   the advantage is not contract-specific and belongs to a design feature
   shared with that model, to be identified by ablation before the
   data-efficiency headline is published.

---

## 1. What exists in the repo

### 1.1 Search results

Grep of `physicsnemo/` for `upt`, `ab_upt`, `anchored`, `universal physics
transformer`, `transolver` (case-insensitive, `*.py`):

| Hit | What it is | Usable as an AB-UPT baseline? |
|---|---|---|
| `physicsnemo/experimental/nn/mt2/model.py:268-273` | MeshTransformer2 flag `n_anchors` + `query_independent`: "AB-UPT-style anchor-conditioned decode -- only a fixed-size anchor subset runs the interacting encoder; all points decode through the read-only path." | No. This is MT2's soft-slice trunk with an anchor/query split (the V6 arm). It measures MT2's price for query independence, not AB-UPT. |
| `physicsnemo/experimental/nn/point_tokenizer.py` | `PointCloudTokenizer`: identity / random / FPS / voxel-pooled FPS / prototype-anchored clustering, kNN mean-pooled token features (for AeroJEPA). | Reusable component for anchor selection and supernode pooling; not a model. |
| `physicsnemo/experimental/models/aerojepa/` | `AeroJEPA`: context encoder + target encoder + query decoder, JEPA head; forward takes `context_pos, context_feat, gen_params, query_pos[, query_sdf]` unbatched (`aerojepa.py:417-426`). | No. Different training objective (JEPA), unbatched contract, its own tokenizer/anchor files on disk (`layers/prototype_anchors.py`). Closest UPT-*shaped* encoder/decoder in the repo, but not a drop-in baseline. |
| `physicsnemo/models/transolver/transolver.py:335` | `Transolver` (plain, Wu et al.), `PhysicsAttentionIrregularMesh` backbone. | **Yes** -- and a recipe yaml already exists (§1.3). |
| `physicsnemo/nn/module/physics_attention.py:209,561,663,786` | `PhysicsAttentionBase`, `PhysicsAttentionIrregularMesh`, structured 2D/3D variants. | Components of Transolver and GeoTransolver. |
| `physicsnemo/models/geotransolver/`, `physicsnemo/experimental/models/geotransolver/` | GeoTransolver (the program's incumbent baseline). | Already measured. |
| `physicsnemo/models/flare/`, `physicsnemo/nn/module/flare_attention.py` | FLARE (another attention-family surrogate; yaml `flare_surface.yaml` exists). | Possible third same-family control; out of scope here. |
| `physicsnemo/domain_parallel/shard_utils/attention_patches.py:1108`, `physicsnemo/mesh/generate/implicit_domain.py` | "anchored" used in unrelated senses (ring-attention masks; mesh generation). | No. |

Grep of the whole tree for `ab-upt|abupt|ab_upt` outside
`examples/cfd/mesh_transformer/{results,book}` returns only the
`mt2/model.py:268` comment. **There is no AB-UPT implementation in
PhysicsNeMo.**

### 1.2 The recipe's model contract (what any new model yaml must satisfy)

Read from `conf/model/geotransolver_surface.yaml`, `conf/model/mt2_surface.yaml`,
`src/train.py:995-1062`, `src/datasets.py:977-986`, `src/forward_kwargs.py`
(module docstring), `src/output_normalize.py:68-116`, `README.md:238-305,509-528`.

- A model yaml is `# @package _global_` and declares four things:
  `input_type` (`tensors` | `mesh`), `output_type`, a `forward_kwargs:` block,
  and a `model:` block with `_target_`. No registry edits are needed
  (`README.md:303-305`).
- `forward_kwargs` maps `model.forward()` kwarg names to dotted DomainMesh
  paths. For surface datasets the available inputs are `interior.points`
  (subsampled cell centroids, `(1, N, 3)`), `boundaries.vehicle.cell_data.normals`
  (`(1, N, 3)`), `global_data.U_inf` (`(3,)`, padded to `(1, 1, 3)` for
  tensor models), `global_data.U_inf_dir` (unit direction, added by
  `ComputeFreestreamDirection` in both surface dataset yamls), and
  `interior.point_data._target_quadrature_measure` (per-point measure
  weights, used by MT2). Lists concatenate along the last dim; the
  `{source, expand_like}` modifier broadcasts a per-sample vector across
  the per-point axis (this is how `transolver_surface.yaml` feeds `U_inf`).
- Targets always live at `interior.point_data.<name>` and are declared in
  the dataset yaml's `targets:` block. `build_dataloaders` sums
  `field_dim()` over the targets (scalar = 1, vector = 3) and injects it as
  top-level `cfg.out_dim` so a template's `out_dim: ${out_dim}` resolves
  (`datasets.py:977-986`). DrivAerML surface: `pressure: scalar, wss: vector`
  -> `out_dim = 4` (`datasets/drivaer_ml_surface.yaml:116-118`). HiLift
  surface: `pressure, temperature, density: scalar; velocity, tau_wall: vector`
  -> `out_dim = 9` (`datasets/highlift_surface.yaml`, `targets:` block).
- The model must return a `(B, N, out_dim)` tensor for `output_type: tensors`;
  `split_concat_by_target` slices channels in `targets:` order and squeezes
  scalars (`output_normalize.py:68-116`). A model with structured heads
  (MT2's `out_scalars` / `out_vectors`) must lay its output out in the same
  order -- `mt2/model.py:648` concatenates `[scalars, vectors]`, which is why
  the HiLift target order (three scalars, then two vectors) is compatible.
  Note that `mt2_surface.yaml` ships `out_scalars: 1, out_vectors: 1`
  (DrivAerML); the HiLift lanes must have overridden these to `3` / `2` on
  the command line -- the override is not in the repo and I did not locate
  the launch script, so it should be verified against the run configs
  before any new model copies the HiLift lane setup.
- Training is `hydra.utils.instantiate(cfg.model)` -> DDP -> Muon optimizer
  (`build_muon_optimizer`, `train.py:1042-1054`), StepLR (step 100, gamma
  0.1), 500 epochs, lr 3e-3 default (`conf/train.yaml`). The frozen protocol
  used by every program lane: `sampling_resolution=10000`, bf16, no
  augmentation, 500 epochs (`hilift_generalization_ladder_preregistration_2026-08-24.json`
  "protocol"; `instrument_wave_preregistration_2026-09-01.json` R1 "runs").
- A new tensor-input template should be added to
  `tests/test_synthetic_configs.py` `_RecipeSpec` list (lines 204-260) so
  the synthetic end-to-end smoke test covers it; `transolver_surface` is
  **not** currently listed there with `highlift_surface`, so the Transolver
  arm needs that one-line addition plus a smoke run before launch.

### 1.3 The plain Transolver is already recipe-ready

`conf/model/transolver_surface.yaml` (read in full):

- `_target_: physicsnemo.models.transolver.Transolver`, `functional_dim: 3`
  (`fx` = `U_inf` broadcast per cell via `expand_like: embedding`),
  `embedding_dim: 6` (points + normals), `n_layers: 8, n_hidden: 256,
  n_head: 8, slice_num: 256, mlp_ratio: 4, unified_pos: false, use_te: false,
  plus: false`, `out_dim: ${out_dim}`.
- `Transolver.forward(fx, embedding, time=None)` concatenates `[embedding, fx]`
  and runs `PhysicsAttentionIrregularMesh` blocks (`transolver.py:691-763`).
  It has no geometry encoder, no local ball-query features, no
  global-token path other than the per-point `U_inf` channel -- exactly the
  "same family, no geometry encoder" control.
- Differences from `geotransolver_surface.yaml` worth stating in the prereg:
  GeoTransolver runs 12 layers (Transolver template 8), has a separate
  `global_embedding` path for `U_inf`, and a `geometry` path; its
  `include_local_features: false` on surfaces means its local ball-query
  channel is already off. Parameter counts should be logged from the
  recipe's `Parameters:` line and reported, but resource axes (peak VRAM,
  s/step) are the program's declared matching frame
  (#sec-nb-resource-accounting).
- Nothing in the repo has run this template on `highlift_surface`;
  `merge_global_data_from: "../../global_data"` in the HiLift reader puts
  `U_inf` where `expand_like` expects it, so it should compose, but a
  synthetic smoke test is the honest first step.

---

## 2. Cost of the two candidate baselines

### 2.1 What the program's own artifacts say AB-UPT is

The lit-review file (`lit_review_sota_scaling_2026-08-20.md`) does **not**
describe the AB-UPT mechanism. It records: CarBench (arXiv:2512.07847,
third-party) AB-UPT 0.136 rel-L2 > Transolver 0.150 > Transolver++ 0.157 on
DrivAerML surface; and the AB-UPT + LoRA transfer paper (arXiv:2605.27968)
as "the current generalization-per-sample story" (zero-shot 4.4x degradation
between related families; LoRA-20 > scratch-103). The notebook adds one
mechanism-level sentence: AB-UPT's anchor-conditioned decoding is one "where
a fixed anchor subset carries all token--token interaction and every query
decodes passively from it" (#sec-nb-lit-wave-prereg, M2 arm), and the lit
synthesis credits the transfer paper with "the GLOBAL geometry encoder as
the non-portable part" (#sec-nb-lit-synthesis item 2). The v5a4 code
comment (`mt2/model.py:268-273`) restates the same mechanism and adds the
program's own contract lesson: the anchor count must be absolute, not a
fraction of the query count, or the anchor set depends on the query set.

That is the full extent of what can be cited from checked-in artifacts.
Anything more specific -- anchor counts, positional encodings, the
surface/volume branching, supernode pooling radii -- must come from
re-reading the primary paper (Alkin et al., "AB-UPT"; the arXiv id is not
recorded in any repo artifact and I did not verify it in this session).
The implementation estimate below therefore carries a
"paper re-read + design note" line item, and the design note should be
committed before code so the reproduction is auditable.

### 2.2 Faithful surface AB-UPT in `physicsnemo/experimental/models/abupt/`

Scope for a surface-only variant with the recipe contract (`forward(points,
normals, u_inf) -> (B, N, out_dim)`): the "branched" (surface/volume) part of
AB-UPT collapses to a single branch, so what gets implemented is anchored
UPT on the surface; the document and yaml should say so rather than claim a
full AB-UPT reproduction.

Components already in the repo (inference from the module listing in
`physicsnemo/nn/module/` and `physicsnemo/nn/functional/`):
`farthest_point_sampling` (used by `point_tokenizer.py`), kNN
(`physicsnemo.nn.functional.neighbors.knn`), `rope.py`, `embedding_layers.py`
/ `conditioning_embedders.py` (for `U_inf` conditioning), `transformer_layers.py`,
`transformer_decoder.py`, `attention_layers.py`. What must be written:

| Item | Engineer-hours (my estimate) |
|---|---|
| Paper re-read, design note in `results/` (anchor selection, encoder depth/width, decoder cross-attention, conditioning, positional encoding, anchor count as absolute), reconcile with the query-independence contract | 4-6 |
| Model: anchor selection (deterministic prefix at eval, random at train), supernode/anchor pooling, anchor encoder (self-attention), query decoder (cross-attention to anchors, no query-query interaction), `U_inf` conditioning, output head; `physicsnemo.core.module.Module` + `MetaData` boilerplate to match `models/transolver/transolver.py` | 12-16 |
| Unit tests: shape contract for both datasets' `out_dim`, query-independence test in the style of MT2's contract tests (prediction at a query is unchanged when other queries are removed, anchors fixed), bf16 + `torch.compile` smoke | 6-8 |
| Recipe: `conf/model/abupt_surface.yaml` (`input_type: tensors`, `forward_kwargs` by analogy with `transolver_surface.yaml`), `_RecipeSpec` entries for both surface datasets, synthetic e2e pass | 3-4 |
| Resource calibration on one DrivAerML lane: pick anchor count / width so peak VRAM sits in the GeoTransolver-MT2 envelope (GT 4.6 GB, MT2-v3c 9.8 GB at 10k tokens; #sec-nb-resource-accounting, #sec-nb-instrument-wave-verdict V6), record s/step | 6-10 |
| Total | **~31-44 h (5-7 engineer-days incl. review)** |

Risks to record now: (i) no reference checkpoint or matching metric exists --
CarBench's 0.136 is rel-L2 under a different protocol, so there is no
calibration anchor for "did we reproduce it"; (ii) the program's only
anchor-decode datum (V6: MT2-v5a4 with 2,560 anchors at 10k tokens lost 3.1x
to the VRAM-matched MT2 control, `instrument_wave_reduction_2026-09-01.json`,
#sec-nb-instrument-wave-verdict) is about MT2's trunk, not AB-UPT, and must
not be read as a prediction for AB-UPT; but it does say that anchor count is
the sensitive knob and should be swept at pilot scale before seeds.

### 2.3 Plain Transolver as the same-family control

| Item | Engineer-hours |
|---|---|
| Add `("transolver_surface", "transolver_surface", "highlift_surface", "surface", [])` and the DrivAerML twin to `tests/test_synthetic_configs.py`; run the synthetic e2e | 1-2 |
| One-epoch smoke on each dataset on the cluster (compose check, VRAM, s/step) | 1-2 |
| Launch + reduction script entries (reuse the ladder reduction pattern that produced `hilift_ladder_reduction_2026-09-03.json`) | 1-2 |
| Total | **~3-6 h (about half an engineer-day)** |

---

## 3. Preregistration skeleton (to be frozen as a JSON artifact before launch)

Readout units throughout: **mean relative-L2 of pressure over the frozen val
instrument**, as in every program table (DrivAerML: frozen val at
`sampling_resolution=10000`; HiLift: the shared 180-case val set). Secondary
fields (wall shear / `tau_wall`, `velocity`) reported, not gated.

Arms: **T** = plain Transolver (`transolver_surface.yaml`, unmodified except
`out_dim` auto-set); **A** = AB-UPT surface (`abupt_surface.yaml`, §2.2),
conditional on the T readout at rung 35 (see §3.5). Both use the frozen
protocol: `sampling_resolution=10000`, 500 epochs, bf16, `augment=false`,
Muon + StepLR as shipped, `save_interval=5`, singleton 3:59 chains,
`training.val_interval` left at its default unless AGA I/O forces the
`val_interval=25` note already recorded for the B1 lanes
(#sec-nb-hl-ladder-ood-indist addendum 2026-09-05).

### 3.1 DrivAerML in-family (frozen instrument)

- Runs per arm: 5 seeds x lr {1e-3, 3e-3} = 10 lanes (the symmetric grid the
  R1 verdict established; #sec-nb-instrument-wave-verdict). The headline is
  each model's best lr by val, both reported.
- Reference values (same instrument): GeoTransolver **0.0532** (5-seed, 3e-3);
  MT2-v3c **0.0577** (2-seed, 1e-3), best-vs-best ratio 1.085
  (#sec-nb-instrument-wave-verdict, R1 table).
- Prediction for T: 1.0-1.3x GeoTransolver (0.053-0.069). Prior: CarBench's
  external ordering puts Transolver ~10% behind AB-UPT and the program's
  N1 reading says DrivAerML in-family is floor-limited; a Transolver far
  outside this band would say the GeoTransolver geometry path matters
  in-family after all.
- Prediction for A: within 1.1x of GeoTransolver (<= 0.059). Falsifier per
  the critic (§5 item 8): A within 1.1x of GT **and** query-independent by
  construction -> the "accuracy and query interaction are entangled" finding
  is a property of MT2's design family only.
- Also report: peak VRAM and train/val s/step, so A and T land on the
  accuracy-vs-VRAM frontier already drawn for GT / MT2 / gt_big.

### 3.2 HiLift data-efficiency rungs (the headline instrument)

- Rungs: `super_scarce` (35), `scarce` (210), `full` (1,260); 2 seeds each;
  shared 180-case val. lr chosen per model by val on the `full` split over
  {3e-3, 1e-3} (the ladder's declared gate rule;
  `hilift_generalization_ladder_preregistration_2026-08-24.json` "gate"),
  then inherited by the 35/210 rungs. Full-split lanes therefore run first
  (2 seeds x 2 lr = 4 lanes per arm), then 4 lanes per arm for the two
  scarce rungs: 8 lanes per arm.
- All five HiLift targets trained (as the GT/MT2 lanes did; the interim table
  at #sec-nb-hl-interim lists temperature and density), pressure gated.
- Reference values (`hilift_ladder_reduction_2026-09-03.json`,
  #sec-nb-hl-ladder-dataeff):

  | n_train | GeoTransolver | MT2 | MT2/GT |
  |---|---|---|---|
  | 35 | 0.392 (0.375-0.409) | 0.138 (0.135-0.141) | 0.35 |
  | 210 | 0.159 (0.157-0.160) | 0.064 (0.062-0.067) | 0.41 |
  | 1,260 | 0.042 | 0.041 | 0.98 |

  plus the steel-man control already in: GeoTransolver at 35 cases, lr 3e-3,
  0.392 / 0.481 (#sec-nb-hl-ladder-ood-indist addendum 2026-09-04); the
  1,000-epoch and 210-case controls are still running.
- **Program's honest prediction (P1):** a Transolver-class model at 35 cases
  lands near GeoTransolver's 0.392 -- band **0.30-0.50** -- not near MT2's
  0.138, *if* the data-efficiency advantage is MT2's geometric contracts
  (SE(3) equivariance by construction, similarity gauge, measure-weighted
  aggregation). At 210, band 0.12-0.20; at 1,260, parity within seed spread
  (0.040-0.045).
- **Falsifier (F1):** A (or T) at 35 cases **<= 1.3 x 0.138 = 0.18** -> the
  advantage is not contract-specific. Interpretation is then forced: the
  small-data collapse is GeoTransolver's, and the design feature MT2 shares
  with the baseline that beat it must be named and ablated (candidates:
  fixed-budget geometry summary vs. GeoTransolver's separate geometry/global
  paths; per-point raw coordinates vs. invariants; depth/width at 35 cases).
  The data-efficiency headline is not published until that ablation exists.
- Intermediate band (0.18-0.30): partial; report as "the contracts buy part
  of the gap" with the ratio, no stronger claim.
- Resolvability rule as in the ladder: a difference smaller than the larger
  of the two seed spreads is "not resolvable"
  (`mt3_preregistration_skeleton_2026-09-03.md`, "Decision rule").
- The HiLift `test` split stays sealed. Adding A/T to the sealed-test
  comparison would require a new one-shot prereg; this skeleton does not
  authorize it.

### 3.3 P3 biased-density probe (eval-only)

- Instrument: `drivaer_probe_biased.yaml`, 10:1 front/back cell-density bias
  at res 10,000, uniform-10k baseline on the identical instrument
  (`contract_axis_probes_preregistration_2026-08-09.json` P3; verdict at
  #sec-nb-contract-probes-synthesis). Readout: pressure rel-L2 biased /
  uniform, per checkpoint, on the best-lr DrivAerML seeds from §3.1.
- Reference: GeoTransolver degrades 1.9-3.6x; MeshTransformer 1.65-2.24x
  with exact inclusion probabilities; MT2 with the similarity gauge 1.20x
  (S1, quoted in `mt3_preregistration_skeleton_2026-09-03.md`).
- Prediction for T: **1.9-3.6x**, the GeoTransolver band -- it never sees
  weights and has the same per-point attention. Prediction for A: 1.3-2.5x
  (prior: FPS-style anchor selection is partly density-normalizing; this is
  a guess, not a measured property).
- Falsifier for "measure-aware aggregation is what buys density robustness":
  A <= 1.3x. Then the robustness comes with anchor pooling and the S1
  claim needs to be rewritten as "measure weighting *or* anchor pooling".

### 3.4 Fixed regardless of readout

- Statistical power for any published ratio: >= 3 seeds (critic C-power;
  `mt3_preregistration_skeleton_2026-09-03.md` "Fixed regardless of branch").
  The 2-seed HiLift rungs above match the existing ladder for comparability;
  a third seed is added at 35 cases for whichever arm decides F1.
- Prereg JSON committed with sha before the first lane; the 5-line
  recommendation in §0 is the hypothesis, not the conclusion.

### 3.5 Ordering and gates

1. T on HiLift `super_scarce` at lr {1e-3, 3e-3} x 2 seeds (4 small lanes),
   in parallel with T on `full` (4 lanes) -- the lr for the 35-case headline
   is still chosen from `full` per the gate rule, but the 3e-3 super_scarce
   lanes are the steel-man control the GT ladder already ran.
2. If T reads out in the P1 band, AB-UPT is worth its 5-7 days as the
   *strongest* Transolver-class competitor; if T already fires F1, the
   ablation program (§3.2) starts immediately and AB-UPT becomes secondary.
3. A on both instruments only after its DrivAerML resource calibration lane.

---

## 4. Resource estimate

Per-lane wall-clock from the program's telemetry (all on 4x GB300, 500
epochs, res 10,000):

- DrivAerML: GeoTransolver **~3.6 h** (#sec-nb-resource-accounting); MT2
  variants **3-6 h** (#sec-nb-mt2-stage0-synthesis, "1.8-9.8 GB / 3-6 h
  training (GT-class)");
  the critic uses "3-6 h per lane" (`critic_review_2026-09-02.md` §4
  preamble). Call it **15-25 GPU-h per DrivAerML lane**; a Transolver at 8
  layers should sit at the low end, an AB-UPT at the low-to-middle depending
  on anchor count.
- HiLift `full`: 315 optimizer steps/epoch, normal per-step 0.27 s and normal
  epoch 60-140 s including per-epoch validation
  (#sec-nb-hl-lr-amendment; 2026-09-05 operational addendum) -> 8-20 h wall,
  **35-80 GPU-h per lane**. Under the current AGA I/O degradation (400-560
  s/epoch) multiply by 4-7, or resume with `val_interval=25`.
- HiLift `super_scarce` / `scarce`: ~9 and ~53 optimizer steps/epoch; the
  epoch cost is then dominated by validating 180 cases, so I estimate
  **2-8 h wall, 8-30 GPU-h per lane** (inference from the step counts and the
  validation footprint; not directly measured in any artifact -- the ladder
  logs on AGA have the true number and should be read before the prereg
  is sealed).
- P3: eval-only, well under 5 GPU-h per checkpoint set (the K0/N1 eval-only
  arms ran at < 10 GPU-h; #sec-nb-k0-verdict).

| Block | Lanes | GPU-h (range) | Engineer-hours |
|---|---|---|---|
| T, DrivAerML in-family (5 seeds x 2 lr) | 10 | 150-250 | 2 (launch + reduce) |
| T, HiLift full (2 seeds x 2 lr) | 4 | 140-320 | 1 |
| T, HiLift 35 + 210 (2 seeds each, plus 3e-3 control at 35) | 6 | 50-180 | 1 |
| T, P3 eval | 0 | < 5 | 1 |
| A, implementation + calibration (§2.2) | 1 | 15-25 | 31-44 |
| A, DrivAerML in-family | 10 | 150-250 | 2 |
| A, HiLift full + 35 + 210 | 10 | 190-500 | 2 |
| A, P3 eval | 0 | < 5 | 1 |
| Reductions, notebook entries, prereg JSON | -- | -- | 6-8 |
| **Total, both arms** | **~41** | **~700-1,540** | **~47-62** |
| **Transolver only** | **20** | **~345-755** | **~6-8 + 3-6 (§2.3)** |

**Smallest arm that could refute the program's interpretation:** plain
Transolver on HiLift `super_scarce`, 2 seeds x lr {1e-3, 3e-3} -- 4 lanes,
~30-130 GPU-h, ~half an engineer-day. It cannot confirm the interpretation
(a Transolver near 0.39 leaves AB-UPT untested), but a Transolver near 0.14
ends the "geometric contracts" reading on its own, before any AB-UPT code is
written. Because the lr is nominally chosen on `full`, running the 4 `full`
lanes concurrently (fan-out, not sequential) keeps the protocol intact at
~170-450 GPU-h total for the decisive Transolver block.
