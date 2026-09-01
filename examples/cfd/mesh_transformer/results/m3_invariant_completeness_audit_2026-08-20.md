# M3: Invariant-completeness audit of MeshTransformer2 (2026-08-20)

Preregistered as arm `M3_invariant_completeness_audit` of
`results/lit_wave_preregistration_2026-08-20.json` (sha256 prefix `69a81e2a`).
Analytic/CPU probe in torch fp64 with randomly initialized weights in eval
mode; no training. Demo script: `results/m3_demo.py` (run with the repo
`.venv`; output reproduced in full below).

## Setting

MeshTransformer2 (MT2, `physicsnemo/experimental/nn/mt2/model.py`) is a
surrogate for surface aerodynamics that is exactly equivariant to rotation,
translation, and scale by construction: its backbone consumes only scalar
invariants of the per-point geometry, and vectors re-enter only at the output
head. The inputs are per-point positions (centered by the cloud mean and
scaled by a reference length, giving `r_i`), unit surface normals `n_i`, and a
single global drive direction `d` (the freestream). The invariants actually
computed by the code are:

- **Seed invariants** (5 per point): `|r|`, `log|r|`, `r̂·d`, `r̂·n`, `n·d`.
- **Slice-relational invariants** (8 per point per slice, every layer): each
  soft slice `s` owns two data-adaptive equivariant anchors — a weighted mean
  position `z_s = Σ_i a_is r_i` and a normalized weighted mean normal
  `m_s = normalize(Σ_i a_is n_i)`, with softmax weights `a_is` that are
  themselves functions of invariants. From `rel = r_i − z_s` the code builds
  `|rel|`, `log|rel|`, `rel̂·d`, `rel̂·n_i`, `rel̂·m_s`, `n_i·m_s`, `|z_s|`,
  `ẑ_s·d`.
- **Optional local features** (flag-gated, default off): Gaussian-kernel patch
  integrals at fixed radii, projected as `n̄·n_i`, `n̄·d`, `|n̄|`, `δ·n_i`,
  `δ·d`, `|δ|`, `log(mass)`, where the kernel weights depend on inter-point
  distances `|r_i − r_j|`.
- **Vector output head**: invariant coefficients over a 7-vector basis
  `{d, n, r̂, e_θ(r̂,n), e_φ(r̂,n), e_θ(r̂,d), e_φ(r̂,d)}`, where
  `e_θ` is a projection (linear in its arguments) and `e_φ = r̂ × e_θ` is a
  **cross product** (`physicsnemo/nn/functional/equivariant_ops.py::spherical_basis`).

The question (M3): is this invariant set expressive enough to separate
geometry pairs that differ aerodynamically, or is there an expressivity
ceiling of the kind Joshi et al.'s Geometric Weisfeiler–Leman (GWL) analysis
predicts for scalarization (invariant-only) architectures — with
chirality/handedness as the conjectured blind spot, since the invariant list
contains no pseudoscalar (no triple-product term like `r̂·(n×d)`)?

**Preregistered falsifier**: "No pair construction with matched invariants and
materially different geometry → ceiling mechanism refuted at this scale."

## Part 1 — Derivation from the code

Throughout, `M` is a reflection (an orthogonal matrix with determinant −1,
e.g. `diag(1,−1,1)`), and "full-problem mirror" means reflecting everything:
`r → Mr`, `n → Mn`, `d → Md`.

### 1a. Every backbone feature is parity-even

Every quantity in the three feature families above is either a norm or a dot
product of two vectors that co-reflect. Norms and dot products are invariant
under *any* orthogonal map, reflections included: `(Ma)·(Mb) = a·b`. The only
subtlety is the data-adaptive anchors, which follow by induction over layers:
the seed invariants are even, so the first-layer hidden states and softmax
weights `a_is` are identical for the mirrored problem; therefore the anchors
co-reflect (`z_s → Mz_s`, `m_s → Mm_s`), so the relational invariants are
even, so the next layer's hidden states are identical, and so on. The kernel
readout in the optional decoder uses only distances. Conclusion, verified to
bitwise equality below:

> **Scalar outputs of MT2 are exactly invariant under the full-problem
> mirror**: `s(MG, Mn, Md) = s(G, n, d)` at corresponding points.

### 1b. For scalar fields this is correct physics, not a ceiling

Incompressible Navier–Stokes is parity-covariant: mirroring the geometry
*and* the freestream mirrors the whole flow, and pressure — a true scalar —
satisfies `p_{MG,Md}(Mx) = p_{G,d}(x)`. MT2's exact scalar mirror invariance
is precisely this identity. The conjectured chirality trap dissolves on
inspection, in both of its versions:

- **Same drive, mirror plane containing `d` (`Md = d`)**: a left-handed vs a
  right-handed strake under a symmetric drive. MT2 provably predicts identical
  pressure at mirrored points — and so does the physics, by the identity
  above with `Md = d`. Even the resulting *asymmetric* loads come out right:
  integrated force `∫ p n dA` mirrors automatically because the normals
  mirror, so left- and right-handed strakes correctly receive opposite side
  forces from identical scalar fields.
- **Same drive, generic mirror (`Md ≠ d`)** — the yawed-drive case where real
  left/right strakes shed different vortices: MT2 *separates* this pair (the
  `·d` invariants change), and it does so under exactly the physical
  identification `MT2(MG, Mn, d) = MT2(G, n, Md)` — the mirrored geometry at
  drive `d` is treated as the original geometry at the mirrored drive, which
  is what Navier–Stokes says too. Measured below: separation 0.13–0.37
  relative, identification residual exactly 0.

The only scalar targets outside the function class are
**pseudoscalar fields** (odd under the full-problem mirror), e.g. wall
helicity density or a signed streamwise-vorticity field. Neither DrivAerML
target (pressure, wall shear stress) is a pseudoscalar.

### 1c. The vector head has a parity anomaly — the opposite defect

The two `e_φ` basis vectors are cross products, hence axial (pseudo-)vectors:
under the full-problem mirror `e_φ → −M e_φ`, while the other five basis
vectors are polar (`b → Mb`) and all coefficients are parity-even. Wall shear
stress is a polar vector, so correct physics demands `v(MX) = M v(X)`; MT2
instead produces `M(v − 2v_φ)` where `v_φ` is the part carried by the two
`e_φ` channels. So MT2's vector head **violates the reflection equivariance
that the physics has** — a trained MT2 cannot be simultaneously correct on a
problem and its mirror image unless its `e_φ` coefficients vanish. This is
not the conjectured ceiling (an inability to express chirality); it is an
uncontrolled *breaking* of parity symmetry: the model can express
chirality-dependent vector fields, but with the wrong (even) transformation
law, so whatever it learns on one handedness transfers with the wrong sign to
the other. Measured below: relative deviation of the mirrored-problem vector
output from the correct mirror is 0.56–1.37 at random initialization, and
drops to exactly 0 when the two `e_φ` head channels are zeroed.

### 1d. A real GWL-style ceiling exists — symmetric-orbit scrambling

The seed invariants have an exact continuous degeneracy: rotating each point
*and its normal* by an arbitrary per-point angle about the drive axis
preserves all five seeds (each is built only from `|r|`, the angle to `d`,
and the point's own normal). Generically the adaptive anchors break this
degeneracy — `z_s` moves when points move. But if the cloud has a discrete
rotational symmetry `C_k` (k ≥ 2) about an axis parallel to `d`, then all
points in a symmetry orbit carry identical hidden states at every layer
(the whole forward pass commutes with the rotation), each orbit's weighted
position and normal sums lie **on the symmetry axis**, and therefore every
anchor `z_s` and `m_s` lies on the axis. Distances and all eight relational
invariants to on-axis anchors are invariant under per-orbit azimuthal
rotation. By induction over layers the entire network output is exactly
unchanged under **independent per-orbit rotations about the drive axis** —
an enormous family of materially different geometries mapped to identical
predictions.

Aerodynamic instance: a slender body with two opposite straight fins vs the
same body with the fins **helically twisted** (90°/unit length), both at zero
incidence (drive along the axis). MT2 provably assigns them identical
pressure and identical shear at corresponding points. Real flows differ
drastically — helical strakes are a standard vortex-suppression device, and a
twisted fin set produces a roll moment a straight one cannot. Note the
construction also manufactures chiral-from-achiral pairs (left- and
right-handed twists both match the straight body), so this is where the
chirality intuition was pointing, but the operative mechanism is invariant
*incompleteness under symmetric orbits*, not parity.

## Part 2 — Numerical demonstration

Script: `results/m3_demo.py`. MeshTransformer2 with `hidden=128`,
`n_layers=6`, `n_slices=64`, fp64, eval mode, random weights, seeds 0/1/2;
default flags (no local features, no decoder split) except where stated.
Geometry: a 400-point chiral helical strake (chirality signature
`χ = Σ_i (r_i × n_i)·d̂ = +93.44` for the original, −93.44 for the mirror —
`χ` is invariant under proper rotations of the configuration, so the sign
flip with `χ ≠ 0` certifies that no rotation relates the pair). All numbers
are relative L2 differences `‖Δ‖/‖ref‖` over the point cloud.

| Test | seed 0 | seed 1 | seed 2 | reading |
|---|---|---|---|---|
| T1 full mirror, scalar residual | 0.0 | 0.0 | 0.0 | exact parity invariance (bitwise: coordinate-plane mirror is exact in fp) |
| T1 full mirror, vector vs correct mirror `Mv` | 1.37 | 0.75 | 0.56 | parity anomaly of the `e_φ` channels |
| T1b same, `e_φ` head channels zeroed | 0.0 | 0.0 | 0.0 | anomaly causally isolated to the two axial channels |
| T2 same drive, `Md=d`: scalar separation | 0.0 | 0.0 | 0.0 | blind — and physics says the true scalars are equal too |
| T3 same drive, `Md≠d`: scalar separation | 0.14 | 0.13 | 0.37 | chirality under yaw IS separated |
| T3 identity `MT2(MG,d)` vs `MT2(G,Md)` | 0.0 | 0.0 | 0.0 | separation happens via exactly the physical identification |
| T4 per-pair azimuthal scramble of a C2 cloud (127–138% geometry change) | 5.2e−16 | 7.7e−16 | 2.0e−15 | **matched-invariant pair: model provably blind** |
| T5 straight vs helically twisted fins, axial drive | 5.1e−16 | 8.7e−16 | 1.3e−15 | **aerodynamically meaningful blind pair** |
| T5 at 1° yaw | 3.5e−3 | 1.2e−2 | 1.5e−2 | blindness decays ≈linearly with yaw |
| T5 at 5° yaw | 1.7e−2 | 5.8e−2 | 7.4e−2 | |
| T5 at 10° yaw | 3.3e−2 | 1.2e−1 | 1.5e−1 | |
| T6 same fin pair, `use_local_features=True` | 8.6e−2 | 7.1e−2 | 6.0e−2 | v4 local channel (inter-point distances) breaks the ceiling instance |

Test-power control: the T4/T5 clouds change by 127–138% in relative L2 while
producing roundoff-level output differences, whereas every non-degenerate
comparison in the same battery (T3, T6, T5-with-yaw) separates at the 1e−2 to
4e−1 level. The instrument can see differences; these pairs have
none visible to the model.

**Instrument note (eps artifact, worth fixing independently).** With exactly
in-plane fin normals (`n_z = 0`), each slice's mean-normal numerator sums to
exactly zero over antipodal pairs, and `m_s = Σaᵢnᵢ / clamp_min(‖Σaᵢnᵢ‖, 1e−12)`
divides roundoff (~1e−16) by `eps = 1e−12`, injecting O(1e−4) noise vectors
that amplified to a *fake* O(1e−2) "separation" in a first version of T5.
Tilting the normals (`n_z = sin 0.3`) removes the degeneracy and the
separation collapses to 1e−15, confirming both the blindness and the
artifact. On any geometry where a slice's mean normal nearly cancels, this
clamp feeds noise into the backbone.

## Part 3 — Does the blindness matter at DrivAerML scale?

Honestly: marginally, and not through the mechanism the literature synthesis
conjectured.

- **The symmetric-orbit ceiling (T4/T5) cannot fire exactly on DrivAerML.**
  It requires a discrete rotational symmetry about an axis parallel to the
  drive. Cars have a mirror plane, not a rotation axis, and the drive is
  longitudinal; no car in the dataset satisfies the condition. Near-misses
  (locally near-axisymmetric features such as wheels or mirrors) do not
  trigger it either, because the anchor collapse is a *global* condition. The
  yaw sweep shows the blindness is a knife-edge in exact form but leaves a
  weak-gradient neighborhood: at 1° drive misalignment the separation is only
  3.5e−3 to 1.5e−2 relative.
- **The vector-head parity anomaly (T1) is the DrivAerML-relevant finding.**
  Cars are near-mirror-symmetric and the drive is in the symmetry plane, so
  the true wall-shear field is near-mirror-symmetric; MT2's `e_φ` channels
  transform with the wrong sign under that mirror, so the model cannot
  inherit left/right consistency for free and any `e_φ` content it learns on
  one side transfers with inverted sign to the other. Effect size at random
  initialization is O(1); the trained magnitude is unmeasured here (would
  require reading `e_φ` coefficient norms from a trained checkpoint — cheap
  follow-up). Pressure, the headline metric, is unaffected.
- **For general aerodynamics the ceiling is real, not contrived.** Bodies of
  revolution and finned axisymmetric bodies at zero incidence — missiles,
  projectiles, chimneys/risers with helical strakes — sit exactly in the
  blind family, and the blind directions (fin twist, strake handedness and
  pitch) are aerodynamically first-order there.

## Verdict against the preregistered falsifier

The falsifier — "no pair construction with matched invariants and materially
different geometry → ceiling mechanism refuted at this scale" — **does not
fire**: T4/T5 are exact constructive counterexamples (invariants matched to
machine precision through every layer, geometry different by 127–138%,
difference aerodynamically first-order). The GWL-style ceiling mechanism is
therefore **confirmed to exist** in MT2's invariant set. However, the
preregistered scale qualifier cuts the other way: the construction requires a
rotational symmetry axis parallel to the drive, which no DrivAerML geometry
has, so within DrivAerML-scale geometry variation the invariant set is
**adequate**, and this mechanism is *disfavored* as the operative explanation
of the G1 scaling anomaly (MT2's data-scaling curve crossing below
GeoTransolver's). Net: **PARTIAL** — ceiling confirmed in general, refuted as
a DrivAerML-scale limitation; and the audit surfaced a distinct, benchmark-
relevant defect (the vector-head parity anomaly) that is a symmetry
*violation* rather than a symmetry-induced ceiling. The originally
conjectured mechanism — parity-even invariants making the model
chirality-blind for scalar fields — is refuted outright: that invariance is
exactly the parity covariance of the physics, and same-drive mirror pairs are
separated whenever the physics separates them.

## Implications for MT3

1. **Fix the vector-head parity anomaly (cheap, do first).** Either drop the
   two axial `e_φ` channels (the five polar vectors span 3-space except where
   `r̂, n, d` are coplanar — and at such points the out-of-plane polar
   component must vanish by symmetry anyway, though conditioning degrades
   nearby), or better, multiply each `e_φ` coefficient by a parity-odd
   invariant gate such as the triple product `r̂·(n̂×d̂)`. That gate flips
   sign under the mirror exactly as needed to make `c_φ e_φ` polar, restoring
   full O(3) equivariance, and it vanishes precisely where the odd component
   must vanish. This is a pseudoscalar *channel* in the smallest possible
   dose — no feature fields needed.
2. **Break the symmetric-orbit ceiling with inter-point distances.** The
   flag-gated v4 local-feature channel already does it (T6: separation 6e−2
   to 8.6e−2 where the default model is blind at 1e−15), because Gaussian
   patch integrals see `|r_i − r_j|`, which per-orbit scrambling changes. Any
   channel with pairwise-distance content suffices; full feature-field
   architectures (Brehmer/GATr direction) are sufficient but not necessary
   for this failure mode. If MT3 keeps an invariant-only backbone, it should
   keep (a cheapened form of) the local channel on by default.
3. **Pseudoscalar targets remain out of class** for the scalar head (exact
   parity-evenness). Irrelevant for pressure/WSS; relevant if the program
   ever predicts helicity-, circulation-, or vorticity-sign-derived surface
   quantities. The same `r̂·(n̂×d̂)`-gated channels would fix this too.
4. **Repair the `m_s` eps clamp** (`model.py`, `_SliceBlock.forward` and the
   decoder pooling): when a slice's mean normal nearly cancels, `0/eps`
   injects O(1e−4) noise into the relational features. A smooth gate (e.g.
   scaling by `‖Σaᵢnᵢ‖/(‖Σaᵢnᵢ‖+eps)`) removes the amplification.

## Reproduction

`/home/psharpe/gh/physicsnemo/.venv/bin/python results/m3_demo.py` from
`examples/cfd/mesh_transformer/`; fp64 on CPU, runtime ≈40 s. Full output:

```text
chi(G, d_in)  = +93.438744
chi(MG, d_in) = -93.438744  (sign flip, nonzero)
seed 0:
  T1 scalar mirror residual        0.000e+00  (expect ~1e-15)
  T1 vector vs correct mirror      1.373e+00  (expect O(0.1-1))
  T1b same, e_phi channels zeroed  0.000e+00  (expect ~1e-15)
  T2 same-drive (Md=d) scalar sep  0.000e+00  (expect ~1e-15)
  T3 same-drive (Md!=d) scalar sep 1.446e-01  (expect O(0.01-1))
  T3 identity vs (G, Md) residual  0.000e+00  (expect ~1e-15)
  T4 scramble geom change 1.274, scalar sep 5.150e-16  (BLIND: matched-invariant pair)
  T5 straight-vs-twisted fins, axial drive: sep 5.098e-16  (BLIND)
     yaw  1.0 deg: sep 3.533e-03
     yaw  5.0 deg: sep 1.657e-02
     yaw 10.0 deg: sep 3.331e-02
  T6 same pair, use_local_features=True: sep 8.576e-02  (degeneracy broken)
seed 1:
  T1 scalar mirror residual        0.000e+00  (expect ~1e-15)
  T1 vector vs correct mirror      7.483e-01  (expect O(0.1-1))
  T1b same, e_phi channels zeroed  0.000e+00  (expect ~1e-15)
  T2 same-drive (Md=d) scalar sep  0.000e+00  (expect ~1e-15)
  T3 same-drive (Md!=d) scalar sep 1.314e-01  (expect O(0.01-1))
  T3 identity vs (G, Md) residual  0.000e+00  (expect ~1e-15)
  T4 scramble geom change 1.290, scalar sep 7.709e-16  (BLIND: matched-invariant pair)
  T5 straight-vs-twisted fins, axial drive: sep 8.650e-16  (BLIND)
     yaw  1.0 deg: sep 1.150e-02
     yaw  5.0 deg: sep 5.756e-02
     yaw 10.0 deg: sep 1.150e-01
  T6 same pair, use_local_features=True: sep 7.149e-02  (degeneracy broken)
seed 2:
  T1 scalar mirror residual        0.000e+00  (expect ~1e-15)
  T1 vector vs correct mirror      5.639e-01  (expect O(0.1-1))
  T1b same, e_phi channels zeroed  0.000e+00  (expect ~1e-15)
  T2 same-drive (Md=d) scalar sep  0.000e+00  (expect ~1e-15)
  T3 same-drive (Md!=d) scalar sep 3.712e-01  (expect O(0.01-1))
  T3 identity vs (G, Md) residual  0.000e+00  (expect ~1e-15)
  T4 scramble geom change 1.382, scalar sep 2.041e-15  (BLIND: matched-invariant pair)
  T5 straight-vs-twisted fins, axial drive: sep 1.324e-15  (BLIND)
     yaw  1.0 deg: sep 1.539e-02
     yaw  5.0 deg: sep 7.406e-02
     yaw 10.0 deg: sep 1.516e-01
  T6 same pair, use_local_features=True: sep 5.981e-02  (degeneracy broken)
```
