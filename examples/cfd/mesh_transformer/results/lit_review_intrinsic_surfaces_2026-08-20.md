# Lit review: intrinsic & gauge-equivariant surface networks (agent report, 2026-08-20)

Full text in session transcript; key content:

RANKED IMPORT MECHANISMS:
1. **Mass-matrix-weighted continuous-operator communication (DiffusionNet,
   arXiv:2012.00888)**: learned per-channel heat-diffusion times on a robust
   Laplacian; all aggregation area-weighted. Best-evidenced density-robustness
   mechanism (FAUST: ~2x degradation under remesh vs ~100x baselines). Targets
   our P3 13x directly. Caveats: "agnostic" = 2-8x not zero; fails across
   disconnected components (car wheels/mirrors!); cross-density evidence is
   within-family only.
2. **Multiscale pointwise-intrinsic features (HKS-class) + extrinsic locals,
   globals isolated to a small adaptable pathway.** Externally corroborated
   diagnosis x2: AB-UPT transfer paper (arXiv:2605.27968) — zero-shot to
   held-out family R^2 = -5.27, frozen GLOBAL geometry encoder is the
   non-portable component, LoRA-20-samples recovers R^2 0.85; CarBench
   (arXiv:2512.07847) — "dataset size beats geometric diversity" for
   Transolver-family transfer. Matches our G2/anchor-binding verdict exactly.
3. **Gauge-equivariant local frames (GEM-CNN arXiv:2003.05425, HSN, Hermes
   arXiv:2310.19589)**: log-map + irrep-constrained kernels + parallel
   transport — exact rotation equivariance with PURELY LOCAL frames (no
   family content). Proven on surface physics (Suk et al. arXiv:2212.05023:
   7.6% WSS error hemodynamics). NO published remeshing/density robustness
   for any gauge method — import only on top of mechanism 1. Bilateral-
   symmetry trap on symmetric car bodies (GEM-CNN needed explicit breaking).

NOTABLE CALIBRATION: "Symmetry in the Wild" (arXiv:2605.18816): on
canonically-aligned aero, equivariance COSTS in-distribution accuracy
(AB-GATr 2.19% vs AB-UPT 1.16%); our measured equivariance-for-free at
1.18x with augmentation costing GT 1.87x is STRONGER than published — a
headline claim for the program.
