# Hierarchical decode backend: design study (task #41)

Status: **design registered, implementation deferred** until the kernel
feature set stabilizes (the velocity-front sweep is live). Dense decode
stays the oracle. Sibling thread: FLARE-style separable smooth members
(task #43) — see *Interface split* below.

Date: 2026-07-07. Author: research session (fork). Sources: GLOBE
`code_to_submit/globe/cluster_tree.py` + `field_kernel.py::BarnesHutKernel`,
GLOBE paper `main.tex` (hierarchical-acceleration appendix + ablations),
`physicsnemo.mesh.spatial.cluster_tree` (upstreamed), our
`kernel_decoder.py` and the measured scale study
(`results/scale_checkpoint_paired_2026-07-05.json`).

## 1. What GLOBE does (extracted, with the measured frontier)

- **Tree**: morton-code Linear BVH (`ClusterTree.from_points`), leaf_size
  default 1, built over BOTH source and target points; per-node AABBs,
  squared diameters, contiguous morton-sorted subtree ranges, area-weighted
  aggregates. Construction is level-batched torch (O(log N) python
  iterations of vectorized ops), `torch.compile`-aware.
- **Dual-tree traversal** (`find_dual_interaction_pairs`): combined opening
  criterion `(D_T + D_S)/r < θ` on AABB gap distance; four interaction
  categories — (near,near) exact pairs, (far,far) node–node centroid
  evaluations broadcast to the target node, (near,far) point–node, and
  (far,near) node–point with a survivor-broadcast mapping. Two-stage
  per-point filtering inside leaf pairs.
- **Far-field approximation is monopole-only through the learned kernel**:
  a cluster becomes one "virtual source" (area-weighted centroid, averaged
  normals/features, summed strengths) fed to the SAME neural kernel. No
  true multipole expansion — the paper names "higher-order multipole
  expansions and a CUDA tree traversal" as future work.
- **`expand_far_targets`** converts (far,far) → (near,far), killing
  target-side blockiness. Measured (paper ablation appendix): θ=0.5
  WITHOUT expansion ≈ 8× validation-loss inflation (up to ~12× at θ=1);
  WITH expansion 2–3× of dense across θ ∈ {0.5, 1, 2}.
- **Scale demonstrated**: DrivAerML training step at S=T=80,000 faces in
  ~7.8 s on one B200 per DDP rank; the dense 80k² evaluation is infeasible
  on the same hardware. Gradient checkpointing passes INDICES into the
  checkpoint boundary (not gathered floats) — ~37× checkpoint-memory
  reduction on near-field chunks.
- **Already upstreamed**: `physicsnemo.mesh.spatial.cluster_tree` carries
  the tree, the dual traversal, the interaction plans, and the segmented
  reductions, Apache-2.0, with tests (`test/mesh/spatial/test_cluster_tree.py`).
  The infrastructure cost of this design is therefore mostly paid.

## 2. Why our problem is better-posed than GLOBE's

GLOBE accelerates a *learned* kernel, so its far field can only be a
heuristic (evaluate the same net at a virtual source). Our hierarchical
path serves the **two exact singular members** — classical Laplace layer
potentials with *known* multipole expansions and *rigorous* error bounds.
The learned content (conditioned coefficients `C_jh`, values `V_jhf`)
enters **linearly** as panel densities:

```
u_hf(x) = Σ_j [ φ_DL,j(x)·C_j,DL,h + φ_SL,j(x)·C_j,SL,h ] · V_jhf
        = Σ_j φ_DL,j(x)·ρ^DL_jhf + φ_SL,j(x)·ρ^SL_jhf ,   ρ = C·V.
```

This is textbook panel-method potential evaluation with (learned) panel
densities — the exact setting the FMM/Barnes-Hut literature solved. The
learned weights ride *inside* the cluster moments exactly; the only
approximation is the truncation order `p`, with the classical
`(a/r)^{p+1}` bound. Under the opening criterion `r ≥ D_S/θ` the
per-cluster relative truncation error is bounded by
`θ^{p+1}/(1−θ)·const`, tunable independently of θ.

## 3. Multipole math for our two exact members

### 3.1 Two dimensions (complex variable)

Write points as complex numbers; cluster center `z_c`, target `ζ = z−z_c`,
panel j = segment `[y_a, y_b]`, length `ℓ_j`, constant unit normal with
complex form `n_j`. Both members expand in ONE basis
`{ln ζ, ζ^{-1}, …, ζ^{-p}}`:

- **Single layer** `φ_SL,j(x) = −(1/2π)∫_Pj ln|z−y| ds`. Using
  `ln|z−y| = Re[ln ζ − Σ_{k≥1} ((y−z_c)/ζ)^k / k]`:

  ```
  Σ_j ρ_j φ_SL,j(x) ≈ −(1/2π) Re[ A_0 ln ζ + Σ_{k=1..p} A_k ζ^{−k} ]
  A_0 = Σ_j ρ_j ℓ_j                    (net monopole)
  A_k += Σ_j ρ_j · (−1/k) · m_{j,k},   m_{j,k} = ∫_Pj (y−z_c)^k ds
  ```

- **Double layer** `φ_DL,j(x) = −(σ_j/2π)·(signed subtended angle)`, with
  `∂_n ln|z−y| = Re[n_j/(y−z)]` and
  `1/(y−z) = −ζ^{−1} Σ_{k≥0} ((y−z_c)/ζ)^k`:

  ```
  Σ_j ρ_j φ_DL,j(x) ≈ (1/2π) Re[ Σ_{k=0..p−1} B_k ζ^{−(k+1)} ]
  B_k = −Σ_j ρ_j σ_j n_j · m_{j,k}
  ```

  (σ_j is the same orientation factor the dense member carries; it rides
  into the moment.)

- **Closed-form panel moments**: with `y(t) = y_a + t·(y_b−y_a)/ℓ_j`,

  ```
  m_{j,k} = ∫_Pj (y−z_c)^k ds
          = ℓ_j · [ (y_b−z_c)^{k+1} − (y_a−z_c)^{k+1} ]
                  / [ (k+1) · (y_b−y_a) ]
  ```

  — a complex polynomial antiderivative, exact, vectorized over panels,
  computed once per tree/geometry (independent of h, f, and weights).

- The two members **fold into one coefficient vector per cluster**:
  `{A_0, (A_k + B_{k−1})_{k=1..p}}` per (h, f) channel. Evaluation per
  (query, cluster) is `O(p)` complex ops shared across all F channels'
  coefficient dot products.

- **Monopole-free deflation compatibility**: the deflated SL member has
  zero net charge by construction ⇒ its `A_0` vanishes identically; the
  deflation is a rank-one correction of the density and commutes with the
  (linear) moment map. No special handling needed beyond applying the
  deflation to ρ before moment assembly.

### 3.2 Three dimensions (real solid harmonics, p ≤ 3)

`1/|x−y| = Σ_{l=0..p} Σ_{m=−l..l} R_lm(y−c)·S_lm(x−c) + O((a/r)^{p+1})`
with regular/irregular solid harmonics R/S.

- **Single layer**: cluster moments
  `M_lm = Σ_j ρ_j ∫_Tj R_lm(y−c) dS`. `R_lm` is a degree-l polynomial;
  integrals over flat triangles of degree ≤ 3 polynomials are **exact**
  under standard symmetric Gauss rules (degree-2: 3-point; degree-3:
  4-point) or the classical monomial formulas — "closed form" either way,
  vectorized per panel, computed once per geometry as `m_{j,lm}`.
- **Double layer**: `∂_n(1/|x−y|)` ⇒ moments
  `M'_lm = Σ_j ρ_j σ_j ∫_Tj n_j·∇R_lm(y−c) dS`; `∇R_lm` is degree l−1,
  `n_j` constant per flat triangle — same exact quadrature. Both members
  again share the irregular-harmonic evaluation `S_lm(x−c)`.
- Coefficient count `(p+1)²` per channel: 9 at p=2, 16 at p=3.

### 3.3 Moment assembly with learned densities

`ρ_jhf = C_jh·V_jhf` (reference config: H=1; F ≈ F_s + F_v·D + F_p).
Per-node moments `M^{hf}_k = Σ_{j∈node} ρ_jhf·m_{j,k}` assemble by
segment-reduction over the node's **contiguous morton-sorted panel range**
(`node_range_start/count` — already stored by ClusterTree). At p ≤ 3 and
tree depth ~17, direct per-node assembly (each panel contributes to its
~log S ancestors) costs `O(S·log S·p_coeffs·F)` scatter-adds — simple and
sufficient; M2M translation operators are a later optimization, not
needed for acceptance. Geometric moments `m_{j,k}` are per-geometry
constants; per-encode work is one einsum + scatter chain, linear in S.
Vector value channels enter component-wise (the moment map is linear, so
O(D)-equivariance is inherited exactly from the dense operator);
pseudoscalar values ride as extra F channels unchanged.

## 4. Traversal: single-tree, and the contract it preserves

**Decision: single-tree (source-side only) query descent, NOT GLOBE's
dual-tree, for the decode.** Rationale — the program's query-set
independence contract:

- Dual-tree opening decisions read *target-node* AABBs, which depend on
  the query SET; even with `expand_far_targets=True`, a query's
  interaction list changes when other queries move. Bitwise query-set
  independence is unrecoverable.
- Single-tree descent classifies (query point, source node) pairs with
  the criterion `dist(x, node AABB)·θ ≥ D_S` — each query's interaction
  list is a function of **its own position and the source tree alone**.
  With per-query reductions in tree order, **bitwise query-set
  independence survives exactly**. This is a structural win over GLOBE's
  scheme and keeps the program's oldest decode contract intact under
  approximation.
- Cost: O(Q·log S) classification instead of dual-tree O(Q+S); at
  Q = 10⁶, S = 82k, depth ≈ 17 this is ~1.7×10⁷ frontier tests of batched
  AABB arithmetic — negligible next to the near-field evaluation.
  Implementation is level-batched (a (query, active-node) frontier per
  level, `torch.cat`/mask advance — same idiom as the upstream dual
  traversal, ~150 lines). The upstream ClusterTree structure (AABBs,
  diameters, ranges, leaf bookkeeping) is reused as-is; only the
  traversal routine is new.
- GLOBE's dual-tree remains available (upstreamed) as an opt-in
  throughput mode for a future boundary→boundary self-interaction use;
  the decode default is single-tree.

**Contract statement (replaces "bitwise dense equivalence"):**

1. Deterministic given `(source mesh, θ, p, leaf_size)` — tree built by
   stable morton argsort in the model's normalized frame.
2. **Bitwise query-set independent** (single-tree; per-query fixed
   reduction order).
3. Similarity-equivariant exactly: the tree lives in the gauge-normalized
   frame, so similar inputs produce identical trees and identical
   arithmetic.
4. Zero-drive ⇒ zero output exactly (moments linear in V).
5. Linear-mode superposition: the BH operator is a fixed linear map of ρ
   given geometry — superposition holds to floating point, same as dense.
6. Deviation from the dense oracle is APPROXIMATE and measured; bounded
   by the pre-registered acceptance below, tunable via (θ, p) with
   `θ→0` or `p→∞` recovering dense.

## 5. Backend design (decoder integration)

- Knob: `decode_backend: Literal["dense", "barnes_hut"] = "dense"` on
  `KernelBasisCrossDecoder` + `kernel_decode_backend` passthrough on
  `MeshTransformer`. Default bitwise-preserving (the BH path is never
  entered). BH parameters: `bh_theta=0.5`, `bh_order=3` (2D) / `2` (3D,
  escalate to 3 if acceptance fails), `bh_leaf_size=32` (tune; GLOBE used
  1 — our near-field pair evaluation is cheap closed forms, so larger
  leaves amortize traversal).
- Cache: `KernelDecoderCache` gains optional tree + geometric-moment
  fields (geometry-only, computed at encode alongside panel_vertices;
  reusable across decodes). Per-encode: density contraction ρ = C·V and
  node moment tensors.
- Decode per query chunk: (a) single-tree descent → per-query far-node
  list + near-panel list; (b) far field: gather node coefficients,
  evaluate `S_lm(x−c)` / complex inverse powers, contract; (c) near
  field: pair-list variant of the existing exact member evaluation (the
  closed forms vectorize over a flat (query, panel) pair list; smooth
  members likewise if present — see interface split); (d) sum. All
  chunk-local ⇒ composes with `checkpoint_query_chunks` unchanged
  (GLOBE's indices-into-checkpoint trick adopted for the near list).
- Gradients: moments/gathers/contractions are autograd-clean; the tree
  (argsort) is not differentiated — geometry is not a training variable.
  Phase 1 rollout may still be inference-first in *usage*, but nothing in
  the design blocks training.

### Interface split with the FLARE thread (task #43)

BH serves the **exact singular members only**. The smooth learned members
(now load-bearing on nonlinear problems) are O(Q·S) through their pair
MLP and are NOT multipole-expandable (learned, non-harmonic). The
combined-backend picture:

| dictionary | dense | BH only | BH + FLARE |
|---|---|---|---|
| singpair (linear suites) | O(QS) | **O(Q log S)** | — |
| + smooth members (RANS arms) | O(QS) | O(QS) (members dominate) | **O(Q log S + (Q+S)r)** |

A members-carrying arm gets sublinear only when BOTH land. If FLARE's
separable members fail their accuracy bar, the fallback is
near-field-restricted smooth members (compactly supported by
construction), which BH's near list serves naturally — noted here as the
contingency, not designed.

## 6. Pre-registered acceptance criteria

- **A1 (accuracy)**: max per-case relative L2 deviation of the decoded
  field, BH vs dense oracle, `< 1e-3` at (θ=0.5, p=3 in 2D / p=2 in 3D)
  on (i) the 3D scale-study sphere/star geometries (subdiv 4–6, 10⁴-query
  sample) and (ii) the far-field ladder annulus banks including the
  r∈[8,12] far bands (the sensitive probe). Escalate p before touching θ
  if failed; report the (θ, p) frontier.
- **A2 (speed)**: decode wall-clock at (S, Q) = (81,920, 10⁶) **< 60 s on
  one GB200** (dense measured: 978 s), i.e. ≥16×; report the full grid.
- **A3 (contracts)**: bitwise query-subset independence under BH;
  determinism across two processes; zero-drive exact; linear-mode
  superposition at the dense test's tolerance.
- **A4 (science regression)**: far-field ladder decay-exponent
  diagnostics under BH within seed-noise of dense; one AirFRANS case
  decode deviation < 1e-3.
- **A5 (training, phase 2)**: checkpointed BH training step ==
  retained-graph BH gradients bitwise; loss curve on one AirFRANS scarce
  seed statistically indistinguishable from dense (same seed, deviation
  attributable to the measured field deviation).

## 7. Implementation plan and effort

1. `physicsnemo/experimental/nn/mesh_attention/multipole.py`: 2D complex
   panel moments + far evaluation; 3D solid-harmonic triangle moments +
   far evaluation; unit tests vs brute-force panel sums and vs dense
   members at increasing r (~350 lines + tests).
2. Single-tree level-batched descent over the upstream ClusterTree
   (~150 lines; lives beside the decoder or upstream in mesh.spatial).
3. `kernel_decoder.py`: backend knob, cache extension, near-field
   pair-list evaluator, far-field contraction (~300 lines).
4. Contract + acceptance tests (~350 lines) and the A1/A2 measurement
   script (extend `scale_study.py` with `--decode-backend`).

Estimate: one focused implementation campaign comparable to the
moment-pool build (2–4 agent-days including verification), AFTER the
sweep verdicts freeze the dictionary. Risks: near-field pair volume on
real clustered meshes (mitigate: leaf_size tuning; measure near-fraction
on AirFRANS/DrivAerML geometries first); 3D p=2 may miss A1 (escalate to
p=3, 16 coeffs, still cheap); double-layer orientation signs in moments
(caught by the dense-oracle unit tests); morton quantization makes
cross-machine bitwise reproduction out of scope (documented).

## 8. Validation protocol (which suites certify the backend)

1. Unit: moment closed forms vs adaptive quadrature; expansion vs dense
   member sums, error ∝ (a/r)^{p+1} slope verified.
2. Deviation grid: A1 geometries × (θ ∈ {0.25, 0.5, 1}, p ∈ {1..4}) —
   the measured accuracy/speed frontier, booked as a figure.
3. Suites under BH: far-field ladder (exponent diagnostics), 3D
   scale-study transfer bank, one AirFRANS scarce case.
4. Perf: scale_study cost grid with `--decode-backend barnes_hut`,
   paired against the dense/checkpointed archive
   (`scale_checkpoint_paired_2026-07-05.json`).
