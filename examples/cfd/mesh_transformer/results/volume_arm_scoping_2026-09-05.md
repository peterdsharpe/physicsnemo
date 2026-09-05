# Scoping the boundary-to-interior (volume) arm: V0 on DrivAerML volume

Date: 2026-09-05. Status: design document, read-only analysis; nothing
launched, nothing committed. Companion to `critic_review_2026-09-02.md`
(§3 "the problem class in the stated objective is boundary → interior",
§4 item 4, §6 item 5), `kernel3d_arm_scoping_2026-09-02.md` (§0 line 5
"K-alt", §3.2 notes on (c)), `mt3_preregistration_skeleton_2026-09-03.md`
(Branch A names "DrivAerML volume" as MT3's first arm), and the notebook
scheduling entry `book/18-notebook.qmd#sec-nb-sched-2026-09-05`.

Every number below is quoted from a checked-in artifact or from code read
in this session, with the file:line or anchor next to it. Where a statement
is an inference or a prior, it says so. Two facts could not be established
from the repository and are marked UNVERIFIED where they appear: the
per-case interior point count of the curated DrivAerML volume files, and
the location of the K0 oracle code (only its JSON artifact is in the tree).

---

## 0. Recommendation in five lines

1. **The DrivAerML volume recipe is a complete boundary → interior data
   contract today** (`datasets/drivaer_ml_volume.yaml`): interior points
   with `velocity`/`pressure`/`nut` targets in `interior.point_data`, the
   subsampled `boundaries.vehicle` surface (with composed HT measure weights)
   kept alongside by default, and per-point `sdf`/`sdf_normals` from the
   full-resolution STL. HiLift volume is **not** runnable as committed
   (nondimensionalization and normalization commented out,
   `highlift_volume.yaml:58-66,83-86`). V0 is DrivAerML-only.
2. **GeoTransolver-volume is wired and smoke-tested but has never produced a
   number in this program**; the kernel-decoder MeshTransformer needs only a
   yaml (flagship minus `trace_of`) because `MeshTransformer.forward` already
   decodes at `domain.interior.points`; **MT2 needs an architecture step**:
   its `query_independent` read path consumes query normals in five places
   (`mt2/model.py:157-165, 361-370, 544-556, 583-596`). The minimal covariant
   fix is an `interior_queries` mode (§2.3) that drops every n-dependent
   query invariant and replaces the head's `n_hat` by the soft-anchor mean
   normal. A zero-code companion arm feeds `interior.point_data.sdf_normals`
   as `query_normals` (the same feature GeoTransolver-volume consumes).
3. **V0 = four 50-epoch pilots, then 2 seeds x 500 epochs x 3 arms** on
   `drivaer_ml_volume` at the frozen protocol (`sampling_resolution=10000`,
   bf16, no augmentation): GeoTransolver-volume (lr 3e-3), MeshTransformer
   kernel-volume (flagship dims), MT2-interior (lr 1e-3, sub-arm chosen by
   pilot). Three recipe-side prerequisites: a dataset variant with
   `ComputeFreestreamDirection`, `SetDomainGlobalField reference_length`,
   `DropDegenerateCells`, and a scalars-first `targets:` order (§2.4).
   ~3 engineer-days, ~190 GPU-h.
4. **Predictions in readout units (mean interior-pressure rel-L2 over the
   val split, unweighted point mean):** MT2-interior / GeoTransolver-volume
   in 1.3-2.5x (falsifier for the v5 "entanglement" finding: <= 1.2x; for
   the interior product: >= 3x); MeshTransformer-kernel / GeoTransolver
   1.5-3x on pressure, 1.0-2.0x on velocity, and **at or below GeoTransolver
   in the far band (sdf >= 0.5 L_ref)** (falsifier for K-alt: MT1 >= 1.5x
   GT in that band). No absolute number for any arm exists in the record;
   absolute bands in §3.3 are priors and say so.
5. **What the volume task buys the G-ledger that surface never could:** query
   independence as a first-class, exactly testable property (interior
   queries are never tokens for MT1 and MT2-interior; they *are* tokens for
   GeoTransolver), query-resolution transfer at fixed source, SDF-band
   stratified error, and a surface-trace consistency readout — all eval-only
   on the V0 checkpoints (§3.4).

Document path: `examples/cfd/mesh_transformer/results/volume_arm_scoping_2026-09-05.md`.

---

## 1. The volume data contract

### 1.1 What `drivaer_ml_volume.yaml` delivers to a model

Read from `datasets/drivaer_ml_volume.yaml`, `physicsnemo/datapipes/readers/mesh.py:399-662`,
`src/forward_kwargs.py`, `src/collate.py`, `src/sdf.py:100-190`, `src/nondim.py:144-270`,
`physicsnemo/datapipes/transforms/mesh/transforms.py:198-262, 959-1139`.

| Item | Where it lives in the sample | How it gets there | Notes |
|---|---|---|---|
| Interior query points | `interior.points`, `(N, 3)`, nondimensional (`x / L_ref`, `L_ref` = 5 m per case) and centered | `DomainMeshReader(subsample_n_points=${sampling_resolution}, drop_interior_cells=true)` → `_subsample_mesh_points` (cyclic contiguous block, uniform inclusion); `NonDimensionalizeByMetadata` divides points by `L_ref` (`nondim.py:159-160, 245-249`); `CenterMesh(use_area_weighting=false)` translates by the **interior sample's point mean** and applies the same shift to boundaries (`transforms.py:240-262`) | Interior is a point cloud (tet topology dropped at load). **No inclusion weights are recorded for the point subsample** (`mesh.py:62-65`, "this does NOT maintain measure weights"). |
| Targets | `interior.point_data.{velocity (N,3), pressure (N,), nut (N,)}` | `pMeanTrim → (p − p_inf)/q_inf` (Cp), `UMeanTrim → U/|U_inf|`, `nutMeanTrim` identity then z-scored `mean 4.8e-4, std 9.4e-4` (`drivaer_ml_volume.yaml:62-67, 83-86`) | `targets:` order is `velocity, pressure, nut` (`:94-97`) → `out_dim = 5` for tensor models (`datasets.py:983-986`). |
| `_target_quadrature_measure` | **absent** | Only `MeshToDomainMesh(interior_points='cell_centroids')` materializes it (`transforms.py:1139`, `TARGET_QUADRATURE_MEASURE_KEY = "_target_quadrature_measure"`, `:45`); the volume yaml has no terminal `MeshToDomainMesh` (`:87-92`) | `extract_target_measure` returns `None` (`forward_kwargs.py:300-308`) → loss and metrics reduce by **unweighted point mean** (`loss.py:129-156`, `metrics.py:96-128`). See §4.3. |
| Geometry features at queries | `interior.point_data.sdf (N,1)`, `interior.point_data.sdf_normals (N,3)` unit vectors | `ComputeSDFFromBoundary(boundary_name=stl_geometry, use_winding_number=false)` on the **full-resolution** `*_single_solid.stl.pmsh` (`extra_boundaries` are never subsampled, `mesh.py:457-464`); `sdf_normals = normalize(query − closest_point)`, with the oriented hit-face normal substituted inside a 128-eps near-wall band (`sdf.py:137-182`) | These are what `geotransolver_volume.yaml` and `transolver_volume.yaml` consume. `sdf_normals` is a true (polar) unit vector field defined at every query; it is discontinuous across the medial surface (two equidistant faces). |
| Boundary tokens | `boundaries.vehicle`: triangle Mesh, `points`, `cells`, `cell_data.{_measure_weights, pMeanTrim, wallShearStressMeanTrim, ...}` | `drop_in_file_boundaries: false` is the yaml default (`:43`); the reader subsamples the boundary to `subsample_n_cells=${sampling_resolution}` cells via `_subsample_mesh_cells`, which **composes the HT weight `N/k` into `cell_data._measure_weights`** (`mesh.py:86-129`, `MEASURE_WEIGHTS_KEY = "_measure_weights"`, `measure.py:80`) | So at `sampling_resolution=10000` a sample carries **10k boundary cells + 10k interior points**. `cell_centroids`, `cell_normals`, `cell_areas` are Mesh properties (`mesh.py:779, 800, 833`) reachable by `forward_kwargs` paths (getattr walk, `forward_kwargs.py:85-108`). The boundary `cell_data` also carries the **surface labels** (`pMeanTrim`, `wallShearStressMeanTrim`; `globe_volume.yaml:25-28`) — a model must not be pointed at them. `geotransolver_volume.yaml:36` and `transolver_volume.yaml:26` opt out via `dataset_reader_overrides: drop_in_file_boundaries: true` (I/O only; `datasets.py:98-107`). |
| Global data | `global_data.{U_inf (3,), rho_inf, p_inf, nu, L_ref}` (physical) | Embedded in the `.pdmsh`; `NonDimensionalizeByMetadata` never touches `global_data` (`domain_transforms.py:194-206`) | **No `U_inf_dir`** (no `ComputeFreestreamDirection`) and **no `reference_length`** leaf (no `SetDomainGlobalField`) — the surface yaml has both (`drivaer_ml_surface.yaml:54-56, 100-102`). Both MeshTransformer templates need them (§2.4). |
| Augmentation | z-rotation + xy-translation, applied to point_data and global_data | `drivaer_ml_volume.yaml:47-56`; inserted after `CenterMesh` by the builder | Off under the frozen protocol (`augment=false`, `base.yaml:28`). |
| Split | manifest mode (`train_split=train`, `val_split=val`, `conf/train.yaml:55-56`) | sibling `manifest.json` under `${dataset_paths.drivaer_ml}` | Whether the volume manifest is the **same 435/48 case list** as the surface instrument must be verified on the cluster (paths are `???` in `datasets/dataset_paths.yaml`); the notebook's "epoch = 54 optimizer steps" (`#sec-nb-hl-lr-amendment`) is the surface split. |

Collate (tensor mode) pads every `forward_kwargs` tensor to `ndim >= 2` and
prepends a batch dim (`collate.py:71-84`): `(N,3) → (1,N,3)`, `(N,) → (1,1,N)`,
`(3,) → (1,1,3)`. MT2 already tolerates all three layouts (`mt2/model.py:441-443,
472`). B = 1 is enforced (`collate.py:141-148`).

### 1.2 `highlift_volume.yaml`

Interior is a point cloud (no normals), `ComputeSDFFromBoundary` runs on the
STL, but `CenterMesh`, `NonDimensionalizeByMetadata` and `NormalizeMeshFields`
are **commented out** (`highlift_volume.yaml:58-66, 83-86`), `targets:` are
raw `avg(P)`-derived fields in slug-inch-second-Rankine, and there is no
`U_inf_dir`. The HiLift *surface* MT2 dataset config is itself not in the
repo (critic C12; `kernel3d_arm_scoping` §5.6). HiLift volume is a stage-2
target after a dataset-yaml pass; V0 does not touch it.

### 1.3 Interior points per case, and per-case cost

**Points per case.** UNVERIFIED: no checked-in artifact records the interior
point count of the curated `domain_*.pdmsh` files, and the dataset is not
mounted here. My prior from the public DrivAerML description is O(10^8)
volume cells per case, i.e. the interior subsample at 10k is a coverage of
order 10^-4 (the surface instrument's 10k-of-17.7M is 0.06-0.11%,
`kernel3d_arm_scoping` §2 (ii)). The pre-launch checklist (§5) includes the
one-liner that settles it. What *is* determined by the yaml: the model sees
exactly `min(sampling_resolution, n_points)` interior points and
`min(sampling_resolution, n_cells)` vehicle cells per sample; the recipe
default `sampling_resolution` is **200,000** for volume (`conf/train.yaml:62`),
the program's frozen protocol is **10,000**.

**Per-case cost at the frozen protocol (10k tokens + 10k queries), against
the surface reference.** Surface references, all from the notebook: DrivAerML
epoch = 54 optimizer steps (`#sec-nb-hl-lr-amendment`); MT2-v3c at 10,000
tokens 9.81 GB peak (`#sec-nb-instrument-wave-verdict` V6), validation step
0.21 s (same entry; v5a4 with the read path 0.30 s); normal per-step 0.27 s
(2026-09-05 operational addendum, `#sec-nb-hl-ladder-ood-indist`);
GeoTransolver 4.6 GB / ~3.6 h per 500-epoch lane on 4x GB300
(`#sec-nb-resource-accounting`); MeshTransformer flagship 35.8 GB / ~12.8 h
(same entry).

| Arm | What changes vs its surface run | Estimated peak VRAM (train) | Estimated s/step | Estimated lane (500 ep, 4x GB300) |
|---|---|---|---|---|
| GeoTransolver-volume | tokens = 10k interior points; `include_local_features: true` with 6 radii (`geotransolver_volume.yaml:65-67`) — an unmeasured path in this program | 5-8 GB (4.6 GB + ball-query features) | 0.13-0.2 | ~3.5-4.5 h |
| MeshTransformer kernel-volume | pair count `N_q x N_s` = 10k x 10k, identical to the surface flagship; trace branches (`exterior_trace_self_entries`) not exercised | ~36 GB (flagship datum; no mechanism for it to be cheaper) | as flagship | ~13 h |
| MT2-interior | encoder identical to surface (10k boundary tokens); plus 4 `_ReadBlock`s at 10k queries: per block `geo (N_q,S,8)` and `geo_feat (N_q,S,96)` (~0.5 GB bf16) and the `_kernel_readout` tiles `(4096, N_s)` fp32 = 164 MB per chunk, 3 chunks, all saved for backward | ~14 GB (9.8 + ~4) | 0.30-0.35 (V6 datum: read path adds ~0.09 s at 10k queries) | ~4-5 h |

At 40k interior queries the MT2 read path alone adds ~16 GB (linear in
`N_q`; §4.1); at 200k it does not fit without recompute-in-backward. V0
trains at 10k queries and evaluates at 10k and 40k (no autograd; linear).

---

## 2. Readiness per architecture

### 2.1 GeoTransolver-volume

- **Config exists**: `conf/model/geotransolver_volume.yaml` (`physicsnemo.models.geotransolver.GeoTransolver`,
  `functional_dim: 7` = coords + sdf + sdf_normals, `global_dim: 3` = `U_inf`,
  12 layers, hidden 256, 256 slices, `include_local_features: true`, radii
  `[0.01 .. 5.0]`, neighbors `[8 .. 128]`). It is the recipe's *default*
  composition (`conf/train.yaml:40, 44`; README "Recipe Gallery" line 539).
- **Has it run in this program?** No. Grep of `results/`, `studies/`, and the
  book for `drivaer_ml_volume` / `geotransolver_volume` finds only the critic
  review, the kernel scoping, the recipe README and the synthetic tests
  (`tests/test_synthetic_configs.py:112, 229-235`). Chapter 16 planned
  "volume task second" (`16-drivaerml.qmd:104`) and never got there;
  `17-program-review.qmd:615-617` and `18-notebook.qmd:4449` say the volume
  recipes exist and were never touched. There is **no GeoTransolver-volume
  number anywhere in the tree**, so V0's GT arm is a first measurement, not a
  comparison against a known baseline.
- **Information diet**: raw centered coordinates, SDF, SDF normals, physical
  `U_inf` (with magnitude); no boundary tokens (`drop_in_file_boundaries:
  true`). Its interior queries *are* its tokens — the `GALEBlock` slice
  attention mixes them (`physicsnemo/models/geotransolver/geotransolver.py:43,
  240-241`), so a prediction at one interior point depends
  on which other interior points were sampled (the surface P4 companion
  instability was 2% for GT, `#sec-nb-v5a3-pilot-p4`).
- **Fairness note**: `CenterMesh` centers on the interior sample's mean
  (density- and sample-dependent) and GT consumes absolute coordinates, so
  GT's inputs carry a small per-sample jitter that the equivariant arms
  ignore. Not worth a fix for V0; record it.

### 2.2 MeshTransformer (MT1) kernel-decoder flagship

- **Decodes at interior queries natively.** `MeshTransformer.forward(domain)`
  = `decode(encode(domain))`, "Encode `domain.boundaries` and predict at
  `domain.interior`" (`mesh_attention/model.py:2802-2804`); the class
  docstring: "source tokens are codimension-one boundary cells with
  geometric measure. The query tokens are `domain.interior.points`"
  (`:432-433`). `decode` chunks queries by `query_chunk_size` (`:2603-2610`)
  and the kernel decoder's `kernel_checkpoint_query_chunks` is the memory
  lever (`mesh_transformer_surface_flagship.yaml:64-65`). The 3D exact
  members are dimension-dispatched and tested (`kernel3d_arm_scoping` §1.1).
- **No volume yaml exists.** `conf/model/mesh_transformer_surface{,_trace,_flagship,_allbc,_allbc_pooled}.yaml`
  only. The flagship's single interior-incompatible line is `trace_of:
  vehicle` (`:56`): `decode` then *requires* `query_mesh.n_points ==
  n_trace` (`model.py:2589-2598`), which a 10k interior sample would satisfy
  by accident at equal counts and be **wrong** (queries would be treated as
  index-aligned boundary centroids). `mesh_transformer_volume.yaml` =
  flagship with `trace_of` removed, `output_field_ranks: {pressure: 0,
  nut: 0, velocity: 1}`, everything else unchanged (`field_mode:
  homogeneous`, `query_decoder: kernel`, 8 MLP members + single layer,
  dims 64/16/96/24, ranks 96/32, `kernel_checkpoint_query_chunks: true`).
  `output_type: mesh` returns a `Mesh` with `point_data` keyed by name, so
  the `targets:` order is irrelevant for this arm.
- **Schema fit**: `_validate_domain` requires the boundary-name set to equal
  the schema exactly (`:1862-1869`); after `DropBoundary [stl_geometry]` the
  volume DomainMesh has exactly `{vehicle}` — matches
  `boundary_field_ranks: {vehicle: ...}`. The reader's composed
  `_measure_weights` on the subsampled boundary enter through
  `MEASURE_WEIGHTS_KEY` at encode (`:2286-2307, 2337-2344`) exactly as on the
  surface.
- **Two dataset-side prerequisites** (both absent from the volume yaml):
  `reference_length_key: reference_length` reads a `global_data` leaf
  (`:2062-2077`) that only the surface yaml injects (`SetGlobalField
  reference_length: 8.0`, `drivaer_ml_surface.yaml:100-102`); and
  `_validate_boundary_measures` rejects zero-area cells, which the surface
  yaml guards with `DropDegenerateCells` last (`:83`,
  `domain_transforms.py:315-339`). For a DomainMesh the domain-level
  injection is `SetDomainGlobalField` (`domain_transforms.py:155-180`).
  Alternative for the gauge: `reference_length_key: null` (intrinsic
  measure-weighted RMS radius of the boundary, `:463-479`) — but V0 should
  keep the flagship's constant 8.0 so the arm is the flagship and nothing
  else; the intrinsic gauge did not rescue MT1 transfer (`17-future.qmd`,
  quoted in `kernel3d_arm_scoping` §1.2) and would be a second delta.
- **Near-wall queries**: the tet mesh has boundary-layer points at
  "sub-micron wall distances" (`sdf.py:141-143`). The closed-form members are
  finite everywhere and the accidental `atan2` branch bites only *exactly on*
  a panel (`kernel3d_arm_scoping` §5.1); with a 0.06%-coverage boundary
  sample, a near-wall query is almost never on a *sampled* panel. Low risk;
  the pilot's non-finite-loss barrier (`base.yaml:44`) catches the rest.

### 2.3 MT2 — exactly what is missing, and the minimal covariant change

**Where query normals enter today** (`physicsnemo/experimental/nn/mt2/model.py`):

1. Query seed invariants: `(q_rhat · q_nhat)`, `(q_nhat · q_d)` — two of the
   five (`:547-556`); the same `self.embed` (width `n_seed`) is reused for
   queries (`:563`), so the seed width is shared with tokens.
2. `_local_invariants_at(q_r, q_n, q_d, ...)`: `(nbar · n_i)`,
   `(delta · n_i)/rho` — two of seven per radius (`:361-370`).
3. `_ReadBlock.forward(q_h, q_r, q_n, ...)`: `(rel_hat · n_exp)`,
   `(n_exp · m_s)` — two of `N_GEO = 8` (`:157-165`).
4. The vector head basis `{d, n, r, e_th_n, e_ph_n, e_th_d, e_ph_d}` uses
   `n_hat` at the query (`:571, 583-596`); `parity_fix` and `odd_head` also
   read it (`:617-644`).
5. `forward` substitutes `query_normals = normals` when none are given
   (`:540`) — which is *shape-wrong* for `N_q != N_tok` and *semantically
   wrong* for interior queries. `query_normals` is therefore effectively
   required.

Nothing else in the read path depends on the query being a token: the
routing (`final_assign`, `z_states`, `z_pos`, `m_s`) is built from the
source only (`:528-535`), and `_kernel_readout` is a pure function of
`(q_r, src_r, src_h, src_w)` (`:187-198`). The contract test
`test_query_set_independence` (`test/experimental/nn/test_mt2.py:161-174`)
already asserts companion-set invariance to 1e-12.

**Two ways to run interior queries, both covariant:**

*V0-a (zero model code).* `query_points: interior.points`,
`query_normals: interior.point_data.sdf_normals`. The closest-point
direction is a unit polar vector, rotation-covariant, defined at every
interior point (`sdf.py:137-182`), so every existing contract test carries
over unchanged, and it is *exactly the geometric feature GeoTransolver-volume
consumes* — the fair-diet arm. Its defects: (i) it is a preprocessing input
computed from the full 17.7M-face STL, not from the 10k boundary tokens the
model sees, so the "bare query" claim is weaker; (ii) it is discontinuous on
the medial surface (§4.2); (iii) `r_hat · n_hat` at an interior point encodes
"which side of the nearest panel", a sensible but non-smooth invariant.

*V0-b (the minimal covariant change — an `interior_queries` mode).* Flag
`interior_queries: bool = False` (requires `query_independent=True`);
`query_normals` may then be `None`. Changes, each a few lines:

- Query seed invariants become `{|q|, log|q|, q_hat · d_hat}` (3), embedded
  by a **separate** `self.query_embed` (`Linear(n_seed_q, hidden) → GELU →
  Linear`), with `n_seed_q = 3 + 5·len(local_radii)·[use_local_features]
  + n_query_scalars`. The optional `n_query_scalars` channel admits `sdf` as
  a per-query invariant scalar (the query-side twin of the existing
  `n_boundary_scalars` token channel, `:247-250, 490-492`); default 0 for
  the bare-query arm.
- `_local_invariants_at` without `q_n`: per radius `{nbar · d, |nbar|,
  (delta · d)/rho, |delta|/rho, log mass}` (5). All are integrals of source
  vectors projected on `d` or normed; exactly SE(3)-covariant.
- `_ReadBlock(n_geo=6)`: geo = `{dist, log dist, rel_hat · d, rel_hat · m_s,
  |z_s|, z_hat_s · d}` — the v3b relational set minus the two `n` terms.
  `N_GEO` becomes a constructor argument so the token-side `_SliceBlock`
  (`N_GEO = 8`) is untouched and frozen checkpoints load.
- Head basis at interior queries: replace `n_hat` by the **soft-anchor mean
  normal** `n_q = normalize(Σ_s mix_qs · m_s)` from the last read block's
  slice mixture (`mix`, `m_s` at `:172, 534-535`; the same construction
  `odd_head` already uses for `m_q`, `:607-612`). `m_s` are measure-weighted
  means of true vectors, so `n_q` is a true vector, rotation-covariant,
  translation-invariant, and defined everywhere (the mixture is a softmax;
  degeneracy only if all anchor normals cancel, which a `clamp_min(eps)`
  handles as today). The basis is then `{d_hat, n_q, q_hat, e_th(q,n_q),
  e_ph(q,n_q), e_th(q,d), e_ph(q,d)}` — the globe7 layout with one
  substitution; `vector_basis` variants and `parity_fix` compose as before.
  For velocity this basis contains the two vectors that matter physically:
  `d_hat` (freestream) and `q_hat`/`n_q` (body-relative direction).
- `local_readout_rho` should accept a tuple (concatenate one `_kernel_readout`
  per radius before `local_read`). Reason: the readout is in gauge units and
  with `similarity_gauge=true` the gauge is the boundary RMS radius
  (~0.25-0.3 L_ref for a car, i.e. ~1.3 m), so `rho = 0.02` is ~2.5 cm
  physical; a query 20 cm from the body sees `exp(-64)` mass — the local
  channel is dead over most of the volume. Pilot knob: `(0.05, 0.25, 1.0)`.
- Tests to add in `test_mt2.py`, mirroring `setup_qi`: rotation equivariance
  with `query_normals=None`, translation invariance, companion-set
  independence, and a shape test at `N_q != N_tok` (the current fixture uses
  `N_q < N_tok` slices of the token set, never disjoint interior points).

Contracts under V0-b: SE(3) exact by construction (every query feature is a
function of `{q − c, d, source geometry}` through norms, dots and
measure-weighted sums); measure-aware routing unchanged; query independence
exact (queries never write); reflection status unchanged from the default
head (documented negative, `#sec-nb-parity-thread-closed`). Estimated
effort: 8-12 engineer-hours including tests.

### 2.4 Recipe wiring: one forward with boundary tokens and interior queries

`conf/model/mt2_volume.yaml` (template `mt2_surface.yaml`):

```yaml
input_type: tensors
output_type: tensors
forward_kwargs:
  points: boundaries.vehicle.cell_centroids      # encoder set = boundary tokens (Mesh property)
  normals: boundaries.vehicle.cell_normals
  measure_weights: boundaries.vehicle.cell_areas # see note (a)
  drive: global_data.U_inf_dir                   # see note (b)
  query_points: interior.points
  # V0-a only:
  # query_normals: interior.point_data.sdf_normals
model:
  _target_: physicsnemo.experimental.nn.MeshTransformer2
  out_scalars: 2        # pressure, nut   -- see note (c)
  out_vectors: 1        # velocity
  hidden: 192
  n_layers: 12
  n_slices: 256
  mlp_ratio: 4
  similarity_gauge: true          # reference MT2 configuration since S1
  query_independent: true
  interior_queries: true          # V0-b; omit for V0-a
  n_decoder_layers: 4
  local_readout_rho: [0.05, 0.25, 1.0]
```

(a) The reader composes the HT factor `N/k` into `cell_data._measure_weights`
but the *effective* measure `area x weight` is not materialized as a field
for boundaries (only `MeshToDomainMesh` does that for the surface task).
`cell_areas` alone is exact for MT2 because the HT factor is a per-sample
constant: it cancels in `softmax(logits + log_w)` (`:88, 115, 529`) and in
the gauge's normalized `w_n` (`:456`). `walk_path` resolves Mesh properties
via `getattr` (`forward_kwargs.py:85-108`); collate turns `(N,)` into
`(1,1,N)`, which `measure_weights.reshape(b, n)` accepts (`:472`).

(b) `global_data.U_inf_dir` does not exist in the volume pipeline. Add
`ComputeFreestreamDirection` before `CenterMesh` (it is domain-aware,
`domain_transforms.py:250-257`) — inert for GeoTransolver. This is the same
gap the critic found on `highlift_surface.yaml` (C12).

(c) MT2 emits `[scalars..., vectors...]` (`:647-649`) and
`split_concat_by_target` slices channels in `targets:` order
(`abupt_baseline_scoping` §1.2, `output_normalize.py:68-116`). The volume
yaml's order `velocity, pressure, nut` would assign MT2's `pressure` channel
to `velocity[0]`. Reorder to `pressure, nut, velocity` in the dataset
variant (GeoTransolver only sees `out_dim = 5`; no existing volume
checkpoint depends on the order).

**Dataset variant** `datasets/drivaer_ml_volume_mt.yaml` (used by all three
arms so the data is identical): the committed `drivaer_ml_volume.yaml` plus
`ComputeFreestreamDirection` (before `CenterMesh`),
`SetDomainGlobalField {reference_length: 8.0}` (for MT1; ignored by MT2 with
`similarity_gauge` and by GT), `DropDegenerateCells` last (default
`apply_to_domain` broadcasts to interior and boundaries,
`transforms/mesh/base.py:65-86`; verify it is a no-op on the point-cloud
interior), and the scalars-first `targets:` block. Register
`("mt2_volume", "drivaer_ml_volume_mt", ...)` and the MT1 twin in
`tests/test_synthetic_configs.py` `_RecipeSpec` (lines 213-265) and add the
boundary-policy row (`:108-132`) so the synthetic end-to-end covers the
composition before any cluster time.

---

## 3. Proposed arm V0

### 3.1 Arms, protocol, gates

Frozen protocol as every program lane: `sampling_resolution=10000`, bf16,
`augment=false`, 500 epochs, Muon + StepLR(100, 0.1), `save_interval` per
the ladder practice, seeds 42/43, one node of 4x GB300, `val_interval=25`
while AGA I/O is degraded (`#sec-nb-hl-ladder-ood-indist` addendum). Readout
= `MetricCalculator` rel-L2 per field, mean over the val split, in training
space (`infer.py` header), plus per-point predictions saved for the
stratified readouts of §3.4.

| Arm | Config | lr | Pilot (1 seed, 50 epochs) | Full |
|---|---|---|---|---|
| `gt_vol` | `geotransolver_volume.yaml` + `drivaer_ml_volume_mt` | 3e-3 (its DrivAerML best, R1) | yes | 2 seeds |
| `mt1_kernel_vol` | `mesh_transformer_volume.yaml` (§2.2) | the flagship's frozen-protocol lr (verify in the flagship launch record; the recipe default is 3e-3) | yes | 2 seeds |
| `mt2_int_a` | `mt2_volume.yaml` with `query_normals: sdf_normals` | 1e-3 (its DrivAerML best, R1) | yes | one of a/b, by pilot |
| `mt2_int_b` | `mt2_volume.yaml`, `interior_queries: true`, bare queries | 1e-3 | yes | one of a/b, by pilot |

**Pilot gates (declared before launch, in readout units):**

- G1 *not collapsed*: val interior-pressure rel-L2 at epoch 50 **<= 0.6**
  (mean predictor = 1.0 by construction of rel-L2) **and** lower than at
  epoch 25 (still descending). Reference for the collapse signature: the v5
  v1/v2 passive decoders sat at a 0.064 *loss* wall while v5a3 left it at
  epoch 49 (`#sec-nb-v5a3-pilot-p4`).
- G2 *memory fits*: peak train VRAM <= 60 GB per GPU at 10k + 10k, leaving
  room for the 40k-query validation.
- G3 *throughput*: <= 1.0 s/step (else a 500-epoch lane exceeds 8 h and the
  arm is re-scoped, not launched).
- MT2 sub-arm rule: carry the sub-arm with the lower epoch-50 pressure
  rel-L2 to full; if within 5% of each other, carry `mt2_int_b` (the arm the
  program's claim is about).

Full arms then run 2 seeds each (the program's habit and the critic's
standing objection, `critic_review` §5); a third seed is added for whichever
pair decides a rule in §3.3 when its two-seed spread overlaps the threshold
(`abupt_baseline_scoping` §3.4 practice).

### 3.2 What the readout measures, stated before the numbers

- **Unweighted point mean over the subsampled interior** (no
  `_target_quadrature_measure`, §1.1). The interior sample follows the CFD
  mesh's point density, which is highest in the boundary layer and wake
  refinement regions. In-family volume rel-L2 is therefore a
  *refinement-weighted* metric, not a volume integral, and is **not
  comparable** to any surface number.
- **Velocity rel-L2 includes the freestream mean.** `velocity = U/|U_inf|`
  has a mean of order the unit freestream, so full-vector rel-L2 will be
  small for every arm and mostly reflects the wake deficit. The
  discriminating quantity is the **perturbation-velocity rel-L2**,
  `||u_pred − u_label|| / ||u_label − U_inf_dir||`, computed from the saved
  per-point predictions (the N1 per-point path, `instrument_wave_reduction_2026-09-01.json`).
  Both are reported; the perturbation form is the one predictions below are
  written in for velocity.
- `nut` is reported, not gated.

### 3.3 Predictions in readout units, and falsifiers

There is no volume number for any architecture in the program's record, so
the *absolute* bands below are priors and the *ratios* are the tests. The
three inputs the caller named:

(i) **The 2D kernel result** (`#sec-nb-g3-kernel-rerun`,
`results/g3/g3_reduction_with_kernel.json`): the singular-only exact
double-layer decoder was the best arm on every axis (T0 0.0158 vs 0.0624 for
mt2_bscalar; T1 unseen modes 0.070 vs 1.15), on a **boundary → interior**
Laplace problem where the operator is exactly the PDE's. That is the K-alt
premise: the spectral leverage lives in propagation off the boundary.

(ii) **K0's failure** (`#sec-nb-k0-verdict`,
`results/k0_potential_flow_oracle_2026-09-02.json`): the exact potential-flow
oracle scores 4.79 on surface Cp against 0.774 for the case mean, because
69% of the surface has label Cp in [−0.5, 0] (separated base) where inviscid
flow predicts ~0. **Does this carry to interior velocity?** *Partly, and the
part that carries is decided by the sample measure, not the physics.*
Pointwise, the RANS mean velocity outside the boundary layer and the wake is
close to irrotational — the body's displacement effect on the outer flow
*is* the potential-flow part, and a 1/r-type kernel is the right operator
there. But (a) the interior sample is refinement-weighted (§3.2), so most
sampled points sit in the boundary layer and the wake, exactly where the
inviscid operator fails; (b) the perturbation-velocity energy is dominated by
the wake deficit (`u − U_inf ~ −U_inf` in the wake versus ~0.1 elsewhere); and
(c) interior *pressure* in the near wake is the same base-pressure deficit
K0 could not represent. So in the *metric-weighted* sense K0's verdict
carries, and no V0 arm is predicted to win on the headline number by having
the right operator. The place the argument does **not** carry is the far
band, which is why §3.4 stratifies the metric by `sdf`: a kernel decoder that
cannot beat GeoTransolver *where the operator is right* has no interior
leverage at all.

(iii) **MT2's surface parity with GeoTransolver** (1.085x best-vs-best on
DrivAerML, 0.98 on HiLift; `#sec-nb-instrument-wave-verdict`,
`#sec-nb-hl-verdict`) was obtained with an *interacting* query stream; the
passive read path cost 2.7-3.9x on the surface (`#sec-nb-v5-line-verdict`,
V6 3.1x at matched VRAM) where the query *is* a token and per-point surface
detail is the signal. In the interior the query never had a token identity,
the field away from the wall is smoother, and the boundary tokens still
interact among themselves; the price should shrink but is not predicted to
vanish.

| Readout (val mean, 10k) | GeoTransolver-volume (prior, absolute) | Prediction | Decision rule (partitioned) |
|---|---|---|---|
| Interior pressure rel-L2 | 0.08-0.20 (prior: harder than the 0.053 surface; sample is wake-heavy; GT gets sdf) | **MT2-interior / GT in [1.3, 2.5]** | <= 1.2: the v5 "accuracy and query interaction are entangled" finding was surface-specific (query = token identity) and MT2 has a query-independent interior product at near-parity → the book's problem statement stands and the passive decoder re-opens. In [1.2, 3.0): report the price as the interior operating point. **>= 3.0**: the passive price persists off the surface → critic §6.5's outcome: MT2 has no interior product without query-side interaction; the framing is rewritten to the surface-trace task and MT3 must own the interior decode (or the design family changes, cf. AB-UPT). |
| Interior pressure rel-L2 | as above | **MT1-kernel / GT in [1.5, 3.0]** (surface standing was 2.5-2.7x: 0.133-0.144 vs 0.053, `kernel3d_arm_scoping` §1.2) | <= 1.2: the kernel decoder's boundary → interior leverage is real on RANS data → K-alt succeeds and re-enters the MT3 brief (Branch A). >= 2.5 on *both* pressure and velocity: no better than its surface standing → **K-alt closes**; the exact-kernel line is closed on this dataset class entirely (K0 closed the surface, this closes the interior). |
| Perturbation-velocity rel-L2 | 0.15-0.40 (prior) | **MT1-kernel / GT in [1.0, 2.0]**, better standing than on pressure (the velocity field has a large irrotational component the 1/r members carry exactly; pressure in the wake does not) | MT1 velocity ratio > pressure ratio → the "right kind of operator" reading is false even directionally. |
| Perturbation-velocity rel-L2 | as above | MT2-interior / GT in [1.2, 2.2] | same partition as the pressure row |
| Far band (sdf >= 0.5 L_ref), pressure and perturbation velocity | — | **MT1-kernel <= 1.0x GT** in the far band | **MT1 >= 1.5x GT in the far band** → the kernel decoder buys nothing even where the operator is right → K-alt closes regardless of the headline. |
| Near-wall band (sdf < 0.01 L_ref) | — | MT2-interior's ratio to GT is worst here (the local-detail channel is what the read path lacks; `#sec-nb-v5a3-pilot-p4`) | if MT2's near-wall ratio is *not* its worst band, the passive price is not a local-detail deficit and the v5a3 mechanism story is wrong |
| `mt2_int_a` vs `mt2_int_b` (pilot and, if both are carried, full) | — | a within 10% of b (the SDF normal is a convenience, not information the boundary tokens lack) | b better than a by > 10% → `sdf_normals` is a harmful discontinuous feature; drop it from every MT2 volume config. a better by > 25% → the bare-query encoder cannot recover near-wall orientation from 10k tokens; `n_query_scalars` (sdf) becomes the next arm. |

**Statistical discipline.** Two seeds per cell; thresholds are placed so
that the decisive ratios (1.2, 2.5, 3.0) are separated by more than the
largest two-seed ratio spread the program has measured on DrivAerML at
n=435 (MT2 pressure 0.0627-0.0671 at lr 3e-3, i.e. ~7%). Anything inside
[1.2, 1.3] or [2.5, 3.0] is "not resolvable at two seeds" and triggers the
third seed before a verdict.

### 3.4 G-ledger axes the volume task enables (eval-only on the V0 checkpoints)

The surface instrument could never separate "query" from "token": every
query was a token. The volume task makes **query independence a first-class,
exactly testable property** — interior queries are never tokens for
MT1 and MT2-interior, and *are* tokens for GeoTransolver-volume.

| Axis | Instrument | Prediction | What it decides |
|---|---|---|---|
| **G-QI: companion-set independence (P4 made first-class)** | Same checkpoint, same 5,000 interior points, two different companion sets of 5,000 (the surface P4 design, `contract_axis_probes_preregistration_2026-08-09.json`) | MT1 and MT2-interior: identical to 1e-12 (`test_query_set_independence`); GeoTransolver-volume: 1-5% (surface GT was 2%) | Whether the interacting design's query sensitivity grows in the interior, where a deployment would evaluate at points the training sample never contained |
| **G-QRES: query-resolution transfer at fixed source** | Train 10k tokens / 10k queries; evaluate at 40k and 200k queries with 10k tokens | MT1, MT2-interior: predictions unchanged by construction (only the metric's sample changes); GT: `|Δ| <= 0.01` (surface GT was flat 2.5k-40k, `kernel3d_arm_scoping` §4.2 table) | A null prediction for GT; the axis discriminates only if GT is not flat in the volume |
| **G-SRES: source-resolution transfer at fixed queries** | Tokens 10k → 40k, queries fixed at 10k | MT1: <= 2x degradation (the surface coverage cliff was 0.167 → 24,811 at 40k, `16-drivaerml.qmd#tbl-coverage-cliff`); MT2: <= 1.1x (surface v3c 0.089 → 0.061 across 2.5k-40k) | Whether the MT1 cliff was source-side (survives here) or trace-side (does not) |
| **G-STRAT: SDF-band stratification** | Per-point predictions binned by `interior.point_data.sdf` (free, already in the sample): `< 0.01`, `[0.01, 0.1)`, `[0.1, 0.5)`, `>= 0.5` L_ref | §3.3 far-band and near-wall rows | Localizes *where* each architecture's error lives; the only readout that can test the K-alt premise independently of the sample measure |
| **G-TRACE: surface consistency of a volume-trained model** | Evaluate the volume checkpoints at the vehicle cell centroids as queries (MT1, MT2-interior natively; GT with `sdf = 0`, `sdf_normals = cell_normals` as tokens) against the surface labels the boundary `cell_data` already carries (`pMeanTrim`, `globe_volume.yaml:25-28`) | every arm >= 2x its surface-trained counterpart (the trace is the hardest region; diagnostic, not gated) | A zero-shot surface readout from an interior model; the first surface/volume consistency number in the program |
| **P3 on the query set** | 10:1 spatially biased *query* sample at fixed tokens | MT1, MT2-interior: ratio 1.00 exactly (queries do not interact; only the metric's sample changes); GT-volume: 1.5-3.6x (its tokens change) | P3 becomes a pure test of the interacting design; no HT weights are possible on the interior (no inclusion weights recorded, §1.1), which is itself a recipe finding |
| **Physics consistency (optional)** | `∇·u` of the predicted velocity by autograd w.r.t. `query_points` (MT1 and MT2-interior are functions of `q` alone; GT is not) | not gated | Only an interior model can be asked this; recorded as a capability, not a claim |

---

## 4. Risks

### 4.1 Memory of MT2's read path at 40k+ queries

`_kernel_readout` tiles the pairwise kernel as `(4096, N_src)` fp32 chunks
(`mt2/model.py:187-198`), and `_local_invariants_at` does the same per radius
(`:349-378`), so no `(N_q, N_src)` matrix is materialized *at once*. But
under autograd every chunk's `k` and `d2` are saved: per `_ReadBlock`, `N_q x
N_src x 4 B` ≈ 1.6 GB at 40k x 10k, times `n_decoder_layers = 4` ≈ 6.4 GB,
plus `geo`/`geo_feat` `(N_q, S, 8+96)` ≈ 2 GB bf16 per block → ~+16 GB at
40k queries over the 10k figure; at 200k queries (the recipe default
resolution) ~+80 GB — does not fit. Mitigations, in order: train at 10k
queries (V0), evaluate at 40k/200k with `torch.no_grad()` (linear, chunked),
and if training at 40k is ever needed, wrap each `_ReadBlock` in
`torch.utils.checkpoint` (the flagship's `kernel_checkpoint_query_chunks` is
the same pattern on the MT1 side). The multi-radius readout (§2.3) multiplies
the readout term by the number of radii; at 10k queries that is ~1.5 GB.

### 4.2 Degenerate interior normals

- V0-a: `sdf_normals` is unit-normalized always (`sdf.py:180-181`), so no
  zero vectors; but it flips across the medial surface (equidistant to two
  faces — the plane of symmetry above the roof, the gap between wheel and
  arch, the underbody-to-ground midplane if the ground were present, which
  it is not in the STL). MT2's `r_hat · n_hat` and the head's `e_th_n,
  e_ph_n` are then discontinuous there. Expect a visible error ridge in
  G-STRAT; it is the reason V0-b exists.
- V0-b: no query normal is consumed; `n_q` is a softmax mixture of anchor
  normals and is defined everywhere. The kernel-mean normal `nbar` inside
  `_local_invariants_at` can vanish by cancellation (a query midway between
  two anti-parallel patches); it enters only through `|nbar|` and `nbar · d`
  (both bounded), not as a basis vector, so it is harmless.
- Boundary orientation: the DrivAerML `vehicle` patch is oriented **inward**
  (signed volume −6.3 to −8.0 m³, `#sec-nb-k0-verdict`). A global sign flip
  is consistent across cases and learned; MT1's double layer flips sign with
  `n`, also consistently. Only *mixed* orientation within a mesh would hurt;
  K0 found the orientation consistent.
- MT1 needs no query normals at all (vector outputs are expanded on relative
  vectors and typed values, `kernel3d_arm_scoping` §3.2 (c)).

### 4.3 Loss and metric weighting (`_target_quadrature_measure`)

Absent for volume (§1.1), so the loss (`huber`, delta 1.0, `base.yaml:36`;
`normalize_by_channels` divides the 5-channel total by 5, `loss.py:391-392`)
and every metric are unweighted point means. Consequences: (i) in-family
numbers are refinement-weighted, not volume-integrated (§3.2); (ii) the
velocity vector contributes 3 of 5 channels; (iii) no HT reweighting is
possible for a biased query sample because `_subsample_mesh_points` records
no inclusion weights and `drop_interior_cells: true` discards the tet
topology a dual-volume measure would need — a design tension worth an
upstream note, not a V0 blocker. All three arms see the identical loss, so
the *ratios* in §3.3 are unaffected; the absolute bands are.

### 4.4 Evaluation cost

- At 10k and 40k queries, 48 val cases x 3 arms is minutes.
- At 200k queries: MT2 read path ~20x the 10k cost, no autograd, fine; MT1
  dense decode `200k x 10k` pairs x 10 members per sample — tens of seconds
  (the K-alt note's "minutes per sample" is at 10^6 x 20k); GT at 200k tokens
  runs its ball queries over 200k points with 6 radii and up to 128
  neighbors — feasible, unmeasured.
- Full-mesh (O(10^7-10^8) points, UNVERIFIED count) is infeasible for all
  three without chunked inference, which `infer.py` declares out of scope
  ("Very large volume meshes may need a smaller cap", `infer.py:60-62`). Cap
  the eval at 200k with a fixed sampling seed and say so.
- Per-point prediction dumps for G-STRAT/G-TRACE at 40k x 48 cases x 5
  channels are ~40 MB per arm.

### 4.5 Record and protocol hazards

- No volume run exists in the program; V0's GeoTransolver number is a first
  measurement, and its lr (3e-3) is inherited from the surface, not tuned
  for the volume. A 1e-3 sibling for GT at pilot scale is cheap insurance
  against the same lr confound the HiLift wave hit
  (`#sec-nb-hl-lr-amendment`).
- The DrivAerML volume manifest split must be confirmed identical to the
  surface instrument's (§1.1); if it is not, the val set is a new instrument
  and must be frozen by hash in the preregistration.
- `compile: true` (`conf/train.yaml:77`): MT2's Python chunk loops are
  shape-static at fixed `N_q`; validation at a different `N_q` recompiles
  once. Not a correctness risk.
- The MT1 flagship's frozen-protocol lr and launch configuration should be
  quoted from the launch record, not the recipe default, before the yaml is
  frozen.
- The K0 oracle code is not in the tree (only
  `results/k0_potential_flow_oracle_2026-09-02.json`); an optional
  "K0-volume" rider (oracle velocity `U_inf + ∇φ` at the interior points, to
  put a zero-parameter potential-flow number on the far band) would first
  need that code recovered and committed.

---

## 5. Pre-launch checklist

1. Inventory three DrivAerML volume cases on the cluster:
   `DomainMesh.load(p).interior.n_points`, `boundaries["vehicle"].n_cells`,
   and the manifest's train/val case lists versus the surface manifest.
   Record the counts in the preregistration.
2. Write `datasets/drivaer_ml_volume_mt.yaml` (§2.4), `conf/model/mt2_volume.yaml`,
   `conf/model/mesh_transformer_volume.yaml`; add both to
   `tests/test_synthetic_configs.py`; pass the synthetic end-to-end.
3. Implement `interior_queries` in `physicsnemo/experimental/nn/mt2/model.py`
   with the four contract tests of §2.3; confirm the existing 19 test
   functions in `test/experimental/nn/test_mt2.py` still pass and frozen
   checkpoints still load (`N_GEO` of `_SliceBlock` untouched).
4. One-epoch smoke of all four pilot arms on one GPU: peak VRAM, s/step,
   finite loss; confirm the `targets:` channel order reaches MT2's head in
   the intended order (assert on `split_concat_by_target` keys).
5. Freeze the preregistration JSON in the `instrument_wave_preregistration_2026-09-01.json`
   format (`title, date, motivation, arms, operational, declared_before_results`)
   with the §3.1 gates, §3.3 bands and §3.4 axes, before the first pilot
   reads out.
6. Launch the four pilots concurrently (fan-out), then the six full lanes.

## 6. Cost

| Block | Lanes | GPU-h (4x GB300) | Engineer-hours |
|---|---|---|---|
| Dataset variant + two model yamls + synthetic tests + smoke | — | ~1 | 4-6 |
| MT2 `interior_queries` mode + tests | — | — | 8-12 |
| Pilots (GT ~0.4 h, MT2 x2 ~0.5 h each, MT1 ~1.3 h; x4 GPUs) | 4 | ~11 | 2 |
| Full: GT 2 x 3.6 h, MT2 2 x 4.5 h, MT1 2 x 13 h; x4 GPUs | 6 | ~170 | 2 |
| Evals: 40k/200k queries, G-QI, G-SRES, G-STRAT, G-TRACE, P3-query | — | ~10 | 4-6 (stratified-readout script reusing the N1 per-point path) |
| Preregistration JSON, reduction, notebook entry | — | — | 4 |
| **Total** | **10** | **~190** | **~24-32 (3-4 engineer-days)** |

The MT1 lanes are 55% of the compute. If budget binds, run MT1 at one seed
in the first wave and add the second only if its ratio to GT lands in a
"not resolvable" band of §3.3 — the K-alt closing rule (>= 2.5x on both
fields) is far enough from any spread that one seed can fire it.
