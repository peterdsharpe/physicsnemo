# Dataset-chapters thread: plan of record

Commissioned by Peter 2026-07-07 (fork thread). Scope settled by Q&A:
**8 chapters, multi-field split; 06 slims to the cross-cutting core;
datasets sit before the industrial chapters; one atomic renumber at the
end.** New experiments approved: N-S wide-Reynolds sweep,
unseen-geometry showcases, resolution-transfer galleries (new-BC
generalization NOT commissioned).

## Chapter set and temporary filenames

Working filenames `ds1..ds8-*.qmd`, slotted after 06 in `_quarto.yml`;
the atomic renumber (07..14 datasets, 15 AirFRANS, 16 DrivAerML,
17 future, 18 notebook) happens LAST, in one pass, coordinated with the
parent session (which appends to the notebook file).

| file | suite | ladder rung | primary sources |
|---|---|---|---|
| ds1-laplace2d.qmd | 2D Laplace conformal star (+ multi-body & deep-cavity stress families as its hard-geometry section) | linear scalar, exact labels — the foundation | 06 suite intro, @sec-encoder-stress, problems/conformal_laplace.py, encoder_stress.py |
| ds2-screened.qmd | screened Laplace (κ̃ family) | first PDE-parameter axis (parametric OOD) | 06 @sec-screened, 05-parametric, problems/screened_laplace.py |
| ds3-laplace3d.qmd | 3D Laplace (sphere/star/shell tiers) | dimension jump + topology (shell) | 06 3D sections, problems/laplace3d.py |
| ds4-liouville.qmd | Liouville | first nonlinearity; the superposition wall | 06 @sec-liouville, problems/liouville.py |
| ds5-potential-flow.qmd | exterior potential flow | vector drive, circulation/parity, exterior domain | 06 @sec-fluid-suites, problems/potential_flow.py |
| ds6-euler-bernoulli.qmd | Euler–Bernoulli beam | first multi-field; parity-of-degree wall | 06 @sec-euler-bernoulli, problems/euler_bernoulli.py |
| ds7-euler-rotational.qmd | rotational Euler | exact vorticity, momentum certification | 06 @sec-euler-rotational, problems/euler_rotational.py |
| ds8-navier-stokes.qmd | N-S star cavity (FEM catalogs) | solver labels, Reynolds grading | 06 Part VI, datasets/fem_navier_stokes.py, catalogs v1/v1-lowre (+ new v2 wide-Re bands) |

## The standard chapter format (every chapter, same order)

1. **The problem** — intuitive statement first (words + BC-annotated
   geometry diagram), then governing equations + BCs + declared
   contracts (which field_mode is licensed and why).
2. **Why this rung** — what the suite adds to the difficulty ladder;
   what it can falsify that the previous rung cannot.
3. **The gallery (centerpiece)** — solution-field small multiples
   across the geometry/drive distribution; split-by-split panels
   (train vs each eval split, dial differences visualized); BC diagrams
   with type/value annotations. Exact-label suites: fields generated in
   live cells (cheap, CPU, memory-light). FEM suites: dated npz
   precomputed on cluster into book/data/.
4. **Benchmarks** — table + figure from the checked-in archive: naive
   all-mean floor, MeshTransformer reference arm, the
   informative ablations for THIS suite (linear-vs-nonlinear on
   linear/nonlinear problems; pseudo on parity suites; members on
   nonlinear; singonly-vs-singpair on hard geometry), plus external
   baselines where they exist (Transolver rows).
5. **Variations** — per-suite: resolution-transfer gallery (all);
   unseen-geometry showcase (ds1, ds3, ds5: creative dials/topology);
   Reynolds response (ds8: wide-Re bands); parameter response (ds2:
   κ̃ extrapolation, already archived).

## New cluster experiments (job prefix mt_ds_*, own workdir/venv)

- **E1 wide-Re catalogs** (ds8): four bands, log-uniform, same
  geometry/h protocol as v1: v2-re1 [1,10], v2-re10 [10,100],
  v2-re100 [100,1000], v2-re1000 [1000,5000]; ~320 train + eval splits
  each; the generator's convergence/noise gates decide feasibility at
  the top band — a mostly-rejected band is itself a reportable
  steady-solver limit. Then reference + members arms per band → the
  accuracy-vs-Re curve.
- **E2 showcase checkpoints**: retrain the per-suite reference arms
  (cheap, 1.5–3k steps) saving checkpoints; eval on showcase
  geometries (encoder-stress dials pushed past training, fused stars,
  topology changes) + qualitative field plots (predicted vs exact) →
  dated npz for galleries.
- **E3 resolution transfer**: same checkpoints, re-eval across 4× up-
  and down-resolution per suite → per-suite transfer curves.

## Coordination rules

- No edits to 09-future/10-notebook filenames until the final atomic
  pass; notebook appends stay single-write atomic.
- 06 slimming happens AFTER all 8 chapters exist and render (move,
  don't duplicate; anchors preserved; 06 keeps: Part I evaluation
  methodology, Part II walls arc + hero figure, Part III cross-suite
  component evidence, Part V Transolver; per-suite setup/galleries/
  results tables move out).
- Cluster jobs never touch the parent's workdir; local compute stays
  memory-light (no FEM solves, no training locally).
