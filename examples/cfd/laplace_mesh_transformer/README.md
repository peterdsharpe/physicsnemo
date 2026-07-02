# MeshTransformer on exact variable-geometry Laplace problems

This example asks a deliberately narrow first question: which mesh-native,
global operator structures can learn a boundary-to-interior PDE solution map
and retain their behavior under new geometries, coordinate frames, physical
scales, and boundary discretizations?

It is a diagnostic research benchmark, not a claim that Laplace's equation is
representative of every PDE. The labels are analytic, the train/test geometry
shift is explicit, and every model receives only physical mesh data. This
keeps a failure attributable to the surrogate rather than a numerical PDE
solver, hidden coordinate encoding, or ambiguous split.

The current conclusion is intentionally split. `MeshTransformer` is a useful,
rigorous mesh-native global **moment primitive**: it is quadrature-aware,
exactly linear in the drive when requested, similarity-covariant, reusable at
new queries, and linear in source-plus-query entity count at fixed feature
order. Its present scalar/vector query decoder is not, however, a general
elliptic surrogate. Under the benchmark's assumptions it has a proved angular
nullspace above order two. This study tests three principled ways out:
higher-order typed moments, a nonseparable relative kernel, and a
PDE-conforming boundary-density solve followed by analytic propagation.

## Why elliptic information flow is global

For the connected interior Dirichlet problem

\[
\Delta u=0\quad\text{in }\Omega,
\qquad u=g\quad\text{on }\partial\Omega,
\]

harmonic measure gives

\[
u(x)=\int_{\partial\Omega}g(y)\,d\omega_x(y).
\]

On a sufficiently regular domain, \(d\omega_x\) has a Poisson-kernel density
with respect to boundary measure. Every boundary portion can affect every
interior query, and the kernel depends on the domain as a whole. Thus a
universal function of only \(x-y\) and \(n_y\) is not sufficient on arbitrary
geometries unless a global geometry representation also conditions it.

An equivalent layer-potential view splits the work differently: first solve a
global boundary integral equation for a density, then propagate that density
with an analytic source-query kernel. This decomposition is central to the
controls below. Here "homogeneous" means zero volume forcing, not zero
boundary data. The global-information statement is specific to connected
elliptic problems; it is not asserted for every PDE class.

## Exact PDE family

Let

\[
F_a(z)=z+\sum_{m\in S}a_m z^m,
\qquad z\in\mathbb D,
\]

with coefficients constrained by

\[
\kappa=\sum_{m\in S}m|a_m|<1.
\]

Then \(|F'_a(z)-1|\leq\kappa\), so (F'_a) cannot vanish. More strongly,

\[
(1-\kappa)|z_1-z_2|
\leq |F_a(z_1)-F_a(z_2)|
\leq (1+\kappa)|z_1-z_2|,
\]

which makes the map injective and quantitatively nondegenerate on the disk.
The physical domain may also undergo any similarity

\[
x=t+L R F_a(z),\qquad L>0,\quad R\in O(2).
\]

For a holomorphic polynomial (H_c),

\[
u(x)=\operatorname{Re}H_c(z)
\]

solves

\[
\Delta_x u=0\quad\text{in }\Omega_a,
\qquad
u=g_c\quad\text{on }\partial\Omega_a
\]

exactly. The target therefore has no PDE-discretization error. Query points are
sampled uniformly in the reference disk, and physical-area losses use the
change-of-variables weight

\[
J(z)=L^2|F'_a(z)|^2.
\]

The polygonal boundary is only the input quadrature. Boundary values are
sampled at parameter-space panel midpoints, and panel winding is corrected
under reflections so mesh normals remain outward.

## What the model sees

The generated `DomainMesh` contains only:

- physical boundary vertices and cells;
- the scalar Dirichlet value in `dirichlet.cell_data["boundary_value"]`;
- physical query coordinates; and
- the explicit physical `reference_length` used for nondimensionalization.

Targets, area weights, and reference-disk radii live in interior `point_data`.
`MeshTransformer` intentionally strips all interior data before encoding and
decoding, so these quantities cannot become model inputs. Conformal
coefficients and preimages remain outside the `DomainMesh` entirely.

The harmonic modes are analytic label generators and evaluation probes, not
positional encodings. Their indices, phases, and reference coordinates are
never passed to the network; learned geometry still uses only typed physical
mesh coordinates, normals, measures, and declared fields.

The reference model uses `field_mode="linear"`. At fixed geometry its
drive-to-solution map therefore obeys zero preservation and superposition by
construction, matching the homogeneous linear PDE.

`MeanLiftedDirichletModel` is an optional, PDE-specific consistency wrapper:

\[
u_\theta[g]=\bar g+N_\theta[g-\bar g],
\qquad
\bar g=\frac{\int_{\partial\Omega}g\,dS}
{\int_{\partial\Omega}dS}.
\]

It exactly maps constant Dirichlet data to the same constant, remains linear,
and adds no coordinate or interaction-length heuristic. This is a valid
Laplace prior but not a universal property of boundary-driven PDEs, so both
the raw and lifted models are benchmarked rather than silently building it
into the generic layer.

For this benchmark, the physical input is sufficient. The tuple

\[
(\partial\Omega,\,dS,\,n,\,g,\,x_{query})
\]

determines the Dirichlet solution. Relative displacement \(x-y\) is derivable
from the supplied boundary and query coordinates; no additional latent
physical variable is missing. The failure diagnosed below is that the current
decoder never forms sufficiently expressive joint source-query geometry. The
reference length is a nondimensionalization datum, not extra PDE information.

## Structural diagnosis: an exact angular ceiling

Fix an encoded boundary, use `field_mode="linear"`, and request a scalar
output. Assume the centered query position \(x\) is the only query-side polar
vector: the problem supplies no physical global operator vector or other
preferred direction. With only invariant-scalar and polar-vector states, the
current decoder can produce only

\[
u(x)=c(|x|^2)+b(|x|^2)\mathbin{\cdot}x
     +x^\mathsf{T}A(|x|^2)x.
\]

The coefficients depend on the encoded boundary and are linear in the drive.
On every circle centered at the model origin, this contains only angular
orders \(m=0,1,2\). On one query ring the boundary-to-ring matrix therefore has
rank at most five: one monopole, two dipoles, and two quadrupoles. Multiple
radii may have distinct radial profiles, so this is not a rank-five claim for
an arbitrary multi-radius matrix.

This ceiling is about physical tensor order, not ordinary feature rank.
`scalar_rank`, `vector_rank`, width, heads, source depth, and query depth add
multiplicity and radial/source-side complexity but do not introduce an
\(m\geq3\) irrep. A real physical preferred vector changes the assumptions;
inventing a Cartesian axis to evade them would violate coordinate-frame
independence. Nonlinear field mode can form higher products, but then exact
drive superposition is lost and angular order is not controlled systematically.

The proof is complemented by randomized-weight tests spanning width, heads,
feature rank, and query depth. They scan all resolvable disk modes and find
only floating-point noise above order two. The stored rank-two moment family
is reducible—it can contain trace, symmetric-trace-free, and antisymmetric
pieces—so a nonzero moment norm alone does not imply a pure order-two channel
or successful transmission to the output.

## Operator and Fourier diagnostics

`spectral_diagnostics.py` applies every discrete boundary basis vector to a
fixed geometry and extracts the complete linear boundary-to-query matrix. On
the unit disk it compares that matrix with the midpoint-trapezoid Poisson
operator, reports its singular values and Eckart--Young best-rank errors, and
computes

\[
T_{\ell k}(r):\quad
e^{ik\theta}\;\longmapsto\;
\text{coefficient of }e^{i\ell\phi}
\text{ on radius }r.
\]

Signed output orders are retained so conjugate-frequency leakage is visible.
SVD here measures compressibility of the composed discrete operator; it is not
a claim that an internal learned pair kernel was separately truncated and
retrained. `study.py spectral-report` records the complete matrices, spectra,
Fourier transfers, grid, dtype, checkpoint metadata, and typed source-moment
norms when the model exposes them.

## Generalization protocol

Training cases are generated online rather than drawn repeatedly from a small
stored geometry set. The default training distribution uses shape modes
\(\{2,3\}\), deformation bound \(\kappa\in[0.05,0.35]\), and boundary modes
\(\{1,2,3,4\}\). Every case is an independent random mixture of the constant
response and all four nonconstant modes. Before exact boundary-RMS
normalization, the constant and each real harmonic mode have equal expected
*boundary* energy. This removes any additional decay imposed by the sampled
boundary spectrum, but it does not undo the PDE's interior attenuation: on the
unit disk, mode \(k\) has area energy proportional to \(1/(k+1)\). The mixed
physical-area objective therefore still emphasizes low modes. Validation and
evaluation use disjoint random mixtures. Pure modes are reserved for the
diagnostic response curve below and are never a positional encoding or
training-only hint in the reference run.

Two explicit optimization controls test whether that weighting hides a
represented mode:

- `disk_interior_balanced_mixture` fixes modal magnitudes and randomizes only
  phases/signs. Since disk mode \(k\) has area energy
  \(\pi|c_k|^2/[2(k+1)]\), choosing
  \(|c_k|=\sqrt{2(k+1)}\) and constant magnitude one gives every included
  component exactly equal disk-area energy in every sampled case before the
  common boundary-RMS normalization.
- `uniform_pure_mode` draws one nonconstant mode uniformly per case. With a
  per-case relative loss, it is an exact equal-mode optimization control.

Equal disk energy is only a declared diagnostic on deformed domains; the
physical Jacobian changes the exact modal inner products there. Neither
control can repair a proved representational nullspace.

Evaluation isolates distinct questions:

| Split | Change from training | What it probes |
| --- | --- | --- |
| `interpolation` | independent seeds, same parameter family | ordinary generalization |
| `unseen_geometry_modes` | shape modes \(\{4,5\}\) | new geometric structure |
| `stronger_deformation` | \(\kappa\in[0.45,0.65]\) | extrapolation in distortion |
| `mixed_geometry_modes` | shape modes \(\{2,3,4,5\}\), unchanged deformation range | new mode combinations |
| `unseen_boundary_frequencies` | zero-mean drive modes \(\{5,6,7,8\}\) | whether a learned PDE operator transfers beyond trained right-hand sides without a seen constant masking their interior error |
| paired similarities | rotations, reflections, translations, scales in \([0.2,5]\) | architectural covariance rather than augmentation |
| boundary resolution | 32, 64, 128, and 256 panels for the same continuous cases | quadrature/remeshing stability |

Geometry OOD and boundary-frequency OOD are reported separately. Combining
them into one score would make it impossible to tell whether failure came from
geometry conditioning or from the learned boundary operator.

## Metrics

The primary metric is per-domain physical-area relative (L^2), followed by
an equal-weight mean and median over domains. A large mesh therefore cannot
dominate a split merely by containing more query points.

The report also includes:

- relative (L^\infty);
- relative (L^2) in the reference annulus \(|z|\geq0.8\), where boundary
  effects are strongest;
- excursion beyond the sampled boundary-value range, normalized by boundary
  RMS (a discretization-aware proxy, not an exact maximum-principle test);
- certified violation of the continuous harmonic maximum principle, using a
  dense trace grid enlarged by the analytic derivative bound
  \(\sum_k k|c_k|\), so sampling cannot create a false positive;
- boundary-quadrature-weighted predicted boundary-trace error;
- paired similarity-covariance error;
- zero-drive and two-drive superposition error for the complete model;
- prediction change relative to the finest boundary quadrature and between
  successive refinements;
- a pure-mode response curve from the constant mode through harmonic mode 12,
  separating trained modes 1--4 from spectral extrapolation; and
- an optional autograd Laplacian diagnostic
  \(\|L^2\Delta u(\theta)\|/\|u\|\), with both norms taken in \(L^2\).

The Laplacian is a diagnostic, not a training penalty. The architecture does
not hard-code harmonicity or the maximum principle, so these measurements show
whether supervised operator learning recovers them approximately.

## Controlled architectural ablations

The first experiment is a two-by-two factorial. It changes boundary
processing and boundary-to-query decoding independently:

| Boundary processor | Moment decoder | Dense invariant pair decoder |
| --- | --- | --- |
| Minimal pointwise lift | `lifted_mesh_transformer/shallow` | `encoded_pair_kernel/shallow` |
| Current global encoder | `lifted_mesh_transformer/reference` | `encoded_pair_kernel` |

Both pair cells use the same signed relative decoder. It sees dimensionless
relative invariants and operator/geometry latents, while a separate
geometry-conditioned linear projection maps the drive state to source-density
channels. In the shallow row those latents come only from the shared pointwise
lift; in the encoded row they have undergone global boundary processing. The
kernel never sees the drive, so superposition remains exact. There is no
softmax, absolute position, Fourier feature, cutoff, or fitted radius. Both
dense decoders pay \(O(N_sN_q)\) query cost. If they dominate both moment rows,
source-query factorization—not the boundary encoder—is the main failure.
All four cells use the same exact boundary-mean lift, so this comparison is not
confounded by giving only the pair decoders the constant-solution prior. The
raw, unlifted MeshTransformer remains a historical diagnostic row.

The still simpler `pair_kernel` row—identity residual drive plus a learned
function of only \((|x-y|^2,n_y\cdot(x-y))\)—is retained as a useful dense
control, but not mislabeled as a strict factorial cell because it does not
share the shallow MeshTransformer's learned pointwise source lift.

`stf_multipole.py` provides systematic planar symmetric-trace-free orders
\(\ell_{max}=1,2,4\). It forms quadrature-weighted source multipoles and
matching query contractions with exact constant lifting and fixed-order
\(O(N_s+N_q)\) entity scaling. On a centered disk, mode \(k\) is present only
when the required order is present; on deformed domains, physical-centroid STF
order is not identical to conformal-reference mode. Learned invariant radial
gates preserve angular type but do not enforce harmonicity. A 9,934-parameter
ordinary scalar/vector control is matched within 0.55% of the 9,880-parameter
\(\ell_{max}=4\) model, so tensor order is not confounded with parameter count.

`layer_potential.py` separates density processing from PDE propagation using
the exact constant-panel 2D double-layer influence:

- direct density lifts the exact constant and uses \(g-\bar g\) itself as the
  residual density, testing whether analytic propagation alone is sufficient;
- solved density evaluates the dense collocation BEM oracle
  \((\tfrac12I+K)\mu=g\);
- Richardson density learns a fixed number of exactly drive-linear residual
  iterations from boundary collocation loss; and
- encoded density uses the current MeshTransformer boundary encoder to predict
  residual density, then evaluates the analytic kernel.

Analytic panel evaluation is harmonic away from panels, but the finite chordal
polygon is not the smooth generating boundary. Smooth-domain query points can
lie outside that polygon near the boundary, where a double layer has a jump.
This discrepancy must vanish under refinement; the solved-density row is a
discrete diagnostic oracle, not the exact smooth-domain solution at finite
panel count. Assembly/storage cost \(O(N_s^2)\), a direct solve costs
\(O(N_s^3)\), a \(p\)-step Richardson processor costs \(O(pN_s^2)\), and query
evaluation costs \(O(N_sN_q)\).

## External controls

[GLOBE](https://arxiv.org/abs/2511.15856) is the closest architectural peer:
it retains nonlinear source-query geometry and boundary communication. The
study runs `theta=0` exact interactions and the same weights/configuration with
its hierarchical backend. GLOBE is area-weighted and its learned exact kernel
is Euclidean-equivariant, but stock processing does not guarantee linearity in
boundary values and includes a prescribed far-field envelope. The current
hierarchy is a Barnes--Hut-style dual-tree aggregate, not a claim of a
classical multipole-to-local FMM or a certified opening-angle error tolerance.

[GeoTransolver](https://arxiv.org/abs/2512.20399v3) is an empirical capacity
control rather than a theory peer. It receives normalized query tokens and a
separate boundary geometry/context cloud containing position, normal, measure,
BC identity, and value. The manually chosen multiscale ball-query path is
disabled. Consequently these are radius-free GALE/context ablations, not
published GeoTransolver reproductions. The matched 72-wide row has 133,083
parameters, 2.1% below the 135,945-parameter reference moment model. The
20-layer, 360-wide row has 29,144,481 parameters and is only a
published-parameter-scale control; it does not reproduce the paper's optimizer,
precision, or local-radius path. Centering/scaling normalize translations
and units, but Cartesian components are ordinary channels, so O(2) behavior is
empirical rather than constructive. The
[Transolver paper](https://proceedings.mlr.press/v235/wu24r.html) supplies the
learned-slice baseline context.

## Prespecified selection gates

`study.py` declares the entire matrix, emits reproducible commands, runs one
seed for early elimination and three seeds for selected finalists, validates a
shared evaluation bank, and applies these necessary—not sufficient—gates:

- trained mode-3 and mode-4 relative errors are each strictly below 0.20;
- ID relative \(L^2\leq0.20\) and at least 75% of the current-to-dense ID gap
  is closed;
- unseen-geometry error is at most 1.2 times ID error;
- boundary-trace error is at most 0.20 and normalized Laplacian residual at
  most 0.40;
- similarity covariance, superposition, and zero-drive preservation hold to
  \(128\epsilon_{fp32}\); and
- the two finest successive boundary refinements provide finite Cauchy
  evidence: the final change contracts and is at most 0.05.

The last test is finite refinement evidence, not a proof of asymptotic
convergence. Diagnostic oracles and external controls are reported through the
same metrics but do not become production candidates merely by passing them.

## Canonical architectural-ablation results (2026-07-01)

The checked-in
[machine-readable ablation artifact](results/architectural_ablation_2026-07-01.json)
records every compact metric, gate outcome, source fingerprint, spectral
summary, and GLOBE backend measurement used below. All canonical reports share
source fingerprint
`7432cb7911d2fcf829ce873f69a0d4d26b5959e0691e7daf9ef490d20b0be065`.
The early-elimination rows use one seed and eight cases per split. The finalist
rows use seeds 17, 29, and 43 and 32 cases per split; means below are followed
by sample standard deviations across training seeds.

### Finalist decision

Only the learned Richardson-density candidate passed every early gate. The
three-seed finalist protocol compares it with the current moment model and the
minimal dense pair control on the larger common evaluation bank:

| Model | Parameters | ID L2 | Unseen geometry | Strong deformation | Unseen frequencies | Mode 3 | Mode 4 | Trace | Laplacian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current encoded moments | 135,945 | 0.4251 ± 0.0006 | 0.4524 ± 0.0003 | 0.3810 ± 0.0074 | 0.9998 ± 0.0000 | 0.9742 ± 0.0045 | 0.9708 ± 0.0007 | 0.5385 ± 0.0006 | 0.8654 ± 0.5736 |
| Minimal dense pair | 48,668 | 0.1836 ± 0.0329 | 0.1786 ± 0.0342 | 0.2356 ± 0.0278 | 0.8946 ± 0.0152 | 0.1534 ± 0.0248 | 0.4362 ± 0.0446 | 0.2396 ± 0.0468 | 3.6782 ± 1.9693 |
| Learned 8-step Richardson density | 8 | 0.023453 ± 0.000001 | 0.018954 ± 0.000001 | 0.028627 ± 0.000002 | 0.071077 ± 0.000001 | 0.079219 ± 0.000001 | 0.102102 ± 0.000000 | 0.000423 ± 0.000005 | 0.000045 ± 0.000015 |

The Richardson model closes 166% of the current-to-dense ID gap, has an
unseen-geometry/ID ratio of 0.81, preserves superposition to `1.39e-7`, and
passes the similarity, zero-drive, trace, Laplacian, modal, and refinement
gates. Its small nonzero continuous maximum-principle metric
(0.00742) remains report-only. The conclusion is not that eight learned
numbers solve every elliptic PDE: for this fixed Laplace operator they learn a
short, resolution-independent polynomial iteration for the boundary integral
equation, while the analytic kernel supplies the correct query-space physics.

### What caused the original failure

The controlled early two-by-two factorial changes boundary processing and
source-query decoding independently:

| Boundary processor | Decoder | Parameters | ID L2 | Unseen geometry | Strong deformation | Mode 3 | Mode 4 | Trace | Laplacian |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimal lift | Moments | 56,341 | 0.5305 | 0.5462 | 0.3925 | 0.9908 | 0.9851 | 0.5471 | 1.7479 |
| Global encoder | Moments | 135,945 | 0.5303 | 0.5462 | 0.3898 | 0.9904 | 0.9850 | 0.5470 | 1.4896 |
| Minimal lift | Dense relative pair | 48,668 | 0.1892 | 0.2023 | 0.2839 | 0.1706 | 0.4835 | 0.2694 | 2.5298 |
| Global encoder | Dense relative pair | 130,736 | 0.1443 | 0.1563 | 0.2179 | 0.1418 | 0.4488 | 0.1920 | 3.5163 |

Adding global boundary encoding changes moment-decoder ID error by only
0.00026. Replacing moments with the relative pair decoder improves ID error by
0.34135 with minimal processing and 0.38595 with global processing. Once joint
source-query geometry exists, the global encoder becomes useful: it improves
pair-model ID, unseen-geometry, and stronger-deformation errors by 23.71%,
22.74%, and 23.24%, respectively. The decoder factorization is therefore the
primary original bottleneck; global boundary communication is a secondary
capacity that becomes useful after that bottleneck is removed.

The objective controls cannot change the nullspace. Equal disk-interior-energy
training leaves mode-3/mode-4 errors at 0.9870/0.9853, and uniform pure-mode
training leaves them at 0.9853/0.9845. Their ID errors remain 0.5300 and
0.5283. A simpler 19,008-parameter pair kernel reaches 0.0910 ID error, but its
mode-4 error is 0.2083 and its Laplacian residual is 7.9980. Better supervised
values do not imply a PDE-conforming operator.

### Typed order and PDE-conforming propagation

The STF order study separates physical tensor order from ordinary width:

| Model | Parameters | ID L2 | Unseen geometry | Mode 3 | Mode 4 | Trace | Laplacian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar/vector matched control | 9,934 | 0.5312 | 0.5480 | 0.9915 | 0.9851 | 0.5472 | 0.3415 |
| STF through order 1 | 2,470 | 0.6379 | 0.6420 | 0.9960 | 1.0002 | 0.7227 | 0.0003 |
| STF through order 2 | 4,940 | 0.5332 | 0.5476 | 0.9915 | 0.9848 | 0.5474 | 0.0000 |
| STF through order 4 | 9,880 | 0.1322 | 0.0953 | 0.1666 | 0.2848 | 0.1318 | 0.2139 |

Order one transmits the dipole, order two adds the quadrupole, and order four
makes modes three and four learnable. At nearly identical parameter count,
order four reduces ID error from 0.5312 to 0.1322. Ordinary scalar/vector
widening is not a substitute for physical irrep order. The order-four model
still misses the mode-4 gate and has 0.4308 stronger-deformation and 0.9737
frequency-OOD error, so a fixed low order remains a real approximation choice,
not a complete generic solver.

The layer-potential controls then isolate density processing from propagation:

| Density processor | Parameters | ID L2 | Unseen geometry | Strong deformation | Unseen frequencies | Mode 3 | Mode 4 | Trace | Laplacian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Use boundary value directly | 0 | 0.3739 | 0.4174 | 0.4425 | 0.5069 | 0.5049 | 0.5021 | 0.4498 | 0.000026 |
| Dense solved-density oracle | 0 | 0.003369 | 0.002686 | 0.02581 | 0.05988 | 0.06742 | 0.01584 | 0.0000003 | 0.000057 |
| Learned 8-step Richardson | 8 | 0.003390 | 0.002748 | 0.02596 | 0.05992 | 0.06742 | 0.01592 | 0.000499 | 0.000028 |
| Mesh encoder to density | 99,029 | 0.01974 | 0.05500 | 0.09088 | 0.06612 | 0.07703 | 0.02441 | 0.02165 | 0.000027 |

An analytic kernel alone is insufficient: the density must solve the boundary
integral equation. Once it does, the eight-parameter iteration is almost
indistinguishable from the dense discrete oracle. The learned mesh encoder is
accurate in absolute terms but its unseen-geometry/ID ratio is 2.79, so it
fails the prespecified geometry gate. For this operator, a small
PDE-structured iteration generalizes better than a large learned geometry
encoder.

### Spectral and external-control evidence

On the centered-disk extraction, the current moment operator has numerical
rank five and relative operator error 0.367785. The exact discrete Poisson
operator has rank 64 and best possible rank-five error 0.366495. The trained
moment map is therefore within 0.00129 absolute error of the best rank-five
approximation: optimization has nearly exhausted its angular image.

The minimal and encoded pair operators have numerical ranks 51 and 55. Their
relative errors to the Poisson operator are 0.208158 and 0.214148. A rank-five
truncation loses 0.250364 and 0.270399 of those learned operators,
respectively; rank eight loses about 0.083--0.086 and rank 16 less than
0.0003. Their rank-five truncations have errors 0.37235 and 0.37613 against
the Poisson operator, erasing the dense models' advantage. Moderate-rank
compression may work on this disk, but five physical channels are
insufficient. These are composed-operator spectra at one
geometry, not a claim of universal kernel rank across geometries.

The common-budget external rows are deliberately controls, not published
recipe reproductions:

| Control | Parameters | ID L2 | Unseen geometry | Mode 3 | Mode 4 | Trace | Laplacian | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLOBE exact interactions | 60,179 | 0.6219 | 0.6398 | 0.9023 | 0.9806 | 0.6904 | 6.0985 | 262 s |
| Radius-free GeoTransolver, matched | 133,083 | 0.7631 | 0.8758 | 1.0580 | 1.0552 | 0.9437 | 0.3355 | 42 s |
| Radius-free GeoTransolver, 29M | 29,144,481 | 0.7432 | 0.8492 | 0.9959 | 1.0029 | 0.8914 | 0.0162 | 299 s |

At 1,000 online updates these controls are under-optimized and cannot support
a ranking of the published methods. They do confirm that neither generic
capacity nor exact pair interactions automatically supply drive linearity or
PDE fidelity. In the same-checkpoint GLOBE backend sweep, increasing `theta`
from 0 to 1 lowers incremental peak CUDA allocation from 99.6 MB to 8.8 MB,
changes the ID prediction by 0.182 in target-normalized L2, and changes sweep
time only modestly at this small mesh (7.1 s versus 8.0 s). The opening angle
is an approximation control, not a certified tolerance; this single timing is
not an asymptotic performance claim.

The resulting architecture decision is narrow and explicit: for Laplace-type
elliptic problems, prefer an exactly drive-linear boundary-density processor
plus a PDE-conforming relative kernel, and add controlled hierarchy only for
scale. Retain the generic `MeshTransformer` as a fast low-order mesh
functional. For PDE families without an analytic kernel, typed irreps are the
principled linear-time path; if practical order is insufficient, use a signed
nonseparable pair-kernel interface with a dense oracle and controlled
hierarchical backend.

## Pre-ablation reference findings (2026-07-01)

The checked-in [machine-readable summary](results/reference_2026-07-01.json)
records the exact source fingerprint, environment, per-seed accuracy values and
aggregates, paired case intervals, mode responses, selected physics
diagnostics, and representative performance sweep points.
These are exploratory reference measurements, not a claim of benchmark
saturation. Learned models used 1,000 online updates with seeds 17, 29, and 43;
all frozen checkpoints were then evaluated on the same fresh 64-case-per-split
bank (`evaluation_seed=97000037`). Values below are mean ± sample standard
deviation across training seeds; the parameter-free mean has one replicate.

| Model | Parameters | ID relative L2 | Unseen geometry | Strong deformation | Mixed geometry | Unseen frequencies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary mean | 0 | 0.800 | 0.825 | 0.790 | 0.823 | 1.000 |
| MeshTransformer | 135,945 | 0.458 ± 0.002 | 0.487 ± 0.001 | 0.481 ± 0.009 | 0.468 ± 0.002 | 1.000 ± 0.000 |
| Constant-lifted MeshTransformer | 135,945 | 0.452 ± 0.001 | 0.483 ± 0.000 | 0.464 ± 0.006 | 0.465 ± 0.001 | 1.000 ± 0.000 |
| Dense invariant pair kernel | 19,008 | 0.107 ± 0.025 | 0.110 ± 0.026 | 0.139 ± 0.018 | 0.100 ± 0.026 | 0.642 ± 0.025 |

The raw MeshTransformer improves ID error over the boundary mean by 0.342;
the paired 64-case bootstrap interval is [0.301, 0.383]. Its unseen-geometry
error is close to its ID error, so geometry shift itself is not the dominant
failure. The dense pair kernel improves over MeshTransformer by 0.351 ID
error, with paired interval [0.314, 0.388], but pays quadratic
source-query cost.

The pure-mode response localizes this trained configuration's limitation:

| Model | Mode 0 | Mode 1 | Mode 2 | Mode 3 | Mode 4 | Mode 5 | Mode 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MeshTransformer | 0.062 | 0.043 | 0.129 | 0.977 | 0.989 | 1.000 | 1.000 |
| Constant-lifted | 0.000 | 0.039 | 0.113 | 0.975 | 0.988 | 1.000 | 1.000 |
| Dense pair kernel | 0.000 | 0.102 | 0.094 | 0.101 | 0.224 | 0.389 | 0.878 |

Modes 1--4 all occur in training, yet this trained reference configuration
learns only modes 1--2 under the mixed physical-area objective. Interior
attenuation explains why optimization emphasizes lower modes, but it cannot
explain the exact nullspace proved above. Increased query depth did not help
because it did not introduce a new physical irrep. The implemented
equal-interior-energy and pure-mode controls quantify optimization effects;
the higher-order and nonseparable prototypes test representation directly.

Physics diagnostics prevent value accuracy from being over-interpreted:

The following pre-ablation trace values used the earlier smooth-boundary query
diagnostic. New study reports use the common polygon-panel interior-trace
contract described above, so trace columns must not be compared across those
protocol versions without reevaluation.

| Model | Boundary-trace L2 | Certified max-principle violation | Normalized Laplacian L2 | Mean per-seed maximum similarity error | 32-to-256-panel change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boundary mean | 0.875 | 0.000 | 0.000 | 1.3e-7 | 0.0004 |
| MeshTransformer | 0.610 | 0.012 | 0.401 | 6.9e-7 | 0.0043 |
| Constant-lifted | 0.606 | 0.013 | 0.427 | 5.3e-7 | 0.0037 |
| Dense pair kernel | 0.162 | 0.001 | 9.756 | 1.2e-6 | 0.0121 |

The dense kernel has the best supervised values but by far the worst autograd
Laplacian residual. Conversely, a constant prediction is harmonic but
uninformative. PDE fidelity and supervised solution error are complementary
requirements, not interchangeable metrics. The architectural similarity
contract holds to ordinary fp32 roundoff, and all learned models are stable
under boundary-quadrature refinement.

## Run

The executable study matrix is the preferred entry point. Early elimination
uses one seed and a smaller, shared evaluation bank:

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/study.py \
  commands --phase early --group factorial --group objective_control \
  --group simple_dense_control --group stf_control --group stf \
  --group layer_potential \
  --device cuda --execute
```

Run external controls separately because their costs differ substantially:

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/study.py \
  commands --phase early --group external --device cuda --execute
```

To isolate hierarchy error from training, sweep the opening angle on the
single exact-trained GLOBE checkpoint rather than comparing independently
trained weights:

```bash
checkpoint=outputs/laplace_mesh_transformer/study/early/globe_exact/seed-17
checkpoint="$checkpoint/globe_exact_reference.pt"
uv run --no-sync python \
  examples/cfd/laplace_mesh_transformer/globe_backend_study.py \
  --checkpoint "$checkpoint" \
  --device cuda \
  --output outputs/laplace_mesh_transformer/study/globe_theta_sweep.json
```

After naming finalists, the driver expands them to seeds 17, 29, and 43 and
automatically includes three-seed current-moment and dense-pair controls:

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/study.py \
  commands --phase finalists \
  --finalist layer_richardson_density \
  --device cuda --execute

uv run --no-sync python examples/cfd/laplace_mesh_transformer/study.py \
  aggregate --phase finalists \
  --output outputs/laplace_mesh_transformer/study/finalists.json
```

The manifest written by `commands --manifest ...` records every exact argv.
`spectral-report` loads a checkpoint and records its disk operator, SVD,
best-rank errors, signed Fourier transfer, and source-moment families where
available.

For an individual run, call `train.py` directly. For example:

From the repository root, the following reproduces the learning protocol used
for each learned reference replicate (the output directory must be distinct
for each model and seed):

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/train.py \
  --model mesh_transformer \
  --capacity reference \
  --steps 1000 \
  --seed 17 \
  --validation-seed 71000011 \
  --evaluation-seed 97000037 \
  --evaluation-cases 64 \
  --evaluation-boundary-points 128 \
  --evaluation-query-points 512 \
  --harmonic-cases 4 \
  --device cuda \
  --output-dir outputs/laplace_mesh_transformer/mesh_transformer_seed17
```

Repeat with seeds 29 and 43. Replace the model with
`lifted_mesh_transformer` and `pair_kernel` for the other learned rows. The
parameter-free `boundary_mean` row uses the same evaluation flags with
`--steps 0`; its training seed does not affect the shared evaluation bank.

After producing the reports, `aggregate.py` reproduces a paired interval. For
example, pass the three MeshTransformer reports to `--left` and the single
boundary-mean report to `--right`; the checked summary records bootstrap seed
20260701 and 100,000 resamples. The script first averages corresponding cases
across training seeds and then resamples the 64 aligned continuous PDE cases,
so seeds are not incorrectly counted as independent problems.

Train the dense control with the identical synthetic distributions:

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/train.py \
  --model pair_kernel \
  --steps 1000 \
  --device cuda \
  --output-dir outputs/laplace_mesh_transformer/pair_kernel
```

Use `--model lifted_mesh_transformer` to measure the physically constrained
constant-mode ablation with the same `tiny`, `shallow`, `reference`, or `large`
core.

Evaluate the parameter-free floor with `--model boundary_mean --steps 0`.
Each training run writes a checkpoint and a JSON report containing the full
run configuration, learning history, aggregate and per-case split metrics,
source fingerprint, software/hardware metadata, parameter count, and wall
time. Use `--checkpoint ... --evaluate-only` to rerun an evaluation without
overwriting the input checkpoint. Supplying a checkpoint without
`--evaluate-only` is a weights-only warm start, not an optimizer-state resume.
Training restores the checkpoint with the lowest error on a fixed validation
stream; the reported evaluation streams use disjoint seeds, and the selected
step/error are recorded explicitly. Validation uses the declared deployment
discretization (128 boundary panels and 512 query points by default), rather
than the coarser 64/128 training discretization, so checkpoint selection tests
resolution transfer without inspecting the final evaluation bank.
`--seed` controls model initialization and online training cases. Validation
and final evaluation use separate fixed seeds shared by every model and
replicate, making comparisons paired; override them only with the explicit
`--validation-seed` and `--evaluation-seed` flags.
Float32 matmul precision defaults to `highest`; selecting `high` or `medium`
requires the explicit `--matmul-precision` flag and is recorded in that report.

Five MeshTransformer capacities are exposed for honest rank/depth ablations:
`tiny`, `stf_matched`, `shallow`, `reference`, and `large`. `shallow` removes boundary
self-propagation while retaining one high-rank global boundary-to-query
operator; it isolates the simplest separable kernel. These settings change
finite-rank capacity or depth, not physical symmetry or interaction length
(there is no interaction cutoff).

Run the analytic-generator and benchmark-contract tests with:

```bash
uv run --no-sync pytest -q examples/cfd/laplace_mesh_transformer/tests
```

## Measure execution cost

`benchmark.py` keeps geometry generation and cache population outside timed
regions and reports synchronized timing samples, throughput, incremental
memory, architecture settings, software/hardware metadata, and the current Git
SHA as one JSON record. For example:

```bash
uv run --no-sync python examples/cfd/laplace_mesh_transformer/benchmark.py \
  --component attention --phase inference \
  --n-source 65536 --n-query 65536 --device cuda

uv run --no-sync python examples/cfd/laplace_mesh_transformer/benchmark.py \
  --component model --phase training \
  --n-source 4096 --n-query 4096 --device cuda
```

Use `--check` at 256 or fewer entities to compare attention values and
gradients with the dense all-pairs oracle, or to compare chunked and unchunked
full models. Performance profiling is intentionally separate from correctness
tests so profiler overhead and hardware variance do not become pass/fail
criteria.

On the reference RTX 4090 Laptop GPU, the packed moment optimization was
measured in a one-off controlled ablation against the four-reduction
implementation at Git commit `02656d0225d6341ef13a78fa4d6283738a8a3f6a`,
with identical weights and 65,536 source entities:

| Mode | Four reductions | One packed reduction | Speedup | Peak workspace, old → packed |
| --- | ---: | ---: | ---: | ---: |
| Linear | 50.38 ms | 12.98 ms | 3.88× | 24.5 → 38.5 MiB |
| Zero-preserving nonlinear | 42.81 ms | 12.57 ms | 3.41× | 56.0 → 56.0 MiB |

The optimization is exact for the checked inference input and changes only
ordinary reduction order in general. The linear-mode speedup trades 14 MiB of
temporary workspace for fewer matrix multiplications; nonlinear workspace is
unchanged. Only the aggregate ablation values were retained, so treat this
speedup as directional; the post-change scaling rows below come from retained
benchmark records with complete timing samples.

Representative post-change scaling medians are shown below. These use the
microbenchmark architecture in `benchmark.py`, not the 135,945-parameter
accuracy model: operator scalar/vector widths 16/4, drive widths 24/8, one
operator, drive, and query block, two heads, and scalar/vector ranks 4/2. The
complete linear model has 18,779 parameters and the nonlinear model 37,275.

| Component | Mode | Sources / queries | Forward or build | Backward or evaluate | Peak workspace |
| --- | --- | ---: | ---: | ---: | ---: |
| Attention inference | Linear | 65,536 / 65,536 | 8.16 ms | 0.94 ms | 38.0 MiB source build |
| Attention inference | Linear | 262,144 / 262,144 | 32.99 ms | 13.84 ms | 38.0 MiB source build |
| Attention inference | Nonlinear | 65,536 / 65,536 | 16.66 ms | 3.61 ms | 56.0 MiB source build |
| Attention inference | Nonlinear | 262,144 / 262,144 | 52.40 ms | 14.08 ms | 56.0 MiB source build |
| Attention training | Linear | 65,536 / 65,536 | 9.74 ms | 33.15 ms | 89.0 MiB |
| Attention training | Nonlinear | 65,536 / 65,536 | 13.11 ms | 34.59 ms | 160.0 MiB |
| Full-model inference | Linear | 65,536 / 65,536 | 56.75 ms encode | 9.68 ms decode | 109.5 / 85.0 MiB |
| Full-model inference | Nonlinear | 65,536 / 65,536 | 69.79 ms encode | 16.04 ms decode | 151.0 / 138.5 MiB |
| Full-model forward/backward | Linear | 16,384 / 16,384 | 149.83 ms forward | 237.37 ms backward | 168.1 MiB |
| Full-model forward/backward | Nonlinear | 16,384 / 16,384 | 242.30 ms forward | 302.79 ms backward | 274.3 MiB |

From 65,536 to 262,144 sources, linear moment-build time grows 4.04× while
entity count grows 4×. Source inference workspace remains bounded by the
65,536-entity chunk. Training memory is not chunk-bounded because autograd
retains saved activations: linear attention uses 5.57, 22.63, and 89.00 MiB at
4,096, 16,384, and 65,536 entities, respectively. These are hardware-specific
measurements, not performance thresholds; the JSON summary retains
representative sweep points and environment metadata. Full-model
forward/backward timings exclude an optimizer update and use three warmups and
seven repeats.

## Interpretation and limitations

This benchmark can support the following claims if the measurements do:

- learning a nontrivial boundary-to-interior elliptic operator;
- generalizing across controlled smooth geometry shifts;
- preserving similarity covariance by construction;
- remaining stable as boundary quadrature is refined; and
- trading finite-rank linear cost against a dense invariant kernel.

It cannot establish performance on corners, holes, topology changes, mixed or
Neumann boundary conditions, variable coefficients, nonlinear PDEs, three
dimensions, or long-time dynamics. Smooth simply connected 2D domains are the
right first falsifiable case, not the final benchmark suite.

Relevant variable-geometry operator-learning comparisons include
[GINO](https://arxiv.org/abs/2309.00583),
[Geo-FNO](https://arxiv.org/abs/2207.05209),
[BENO](https://openreview.net/forum?id=xZqEdxTqd2), and
[Boundary-Augmented Neural Operators](https://openreview.net/forum?id=DqZoWaDwfN).
Their inclusion here motivates geometry-OOD and resolution tests; it does not
imply an apples-to-apples numerical comparison with this small analytic task.
