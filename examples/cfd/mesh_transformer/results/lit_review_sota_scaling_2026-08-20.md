# Lit review: transformer-operator SOTA + scaling methodology (agent report, 2026-08-20)

Full text in session transcript; key content:

SANITY ANCHORS: GeoTransolver paper (arXiv:2512.20399): DrivAerML 2.86%
rel-L1 surface pressure, drag R^2 0.996 at ~400 train, 29M params (NOTE
rel-L1 vs our normalized L2; OOD-drag-extremes test split convention).
CarBench (arXiv:2512.07847, third-party): AB-UPT 0.136 rel-L2 > Transolver
0.150 > Transolver++ 0.157 — lineage self-reported deltas shrink externally.

THE METHODOLOGY VACUUM (opportunity #1): across ~25 papers, ONE work
compares architectures by fitted data-scaling exponent (CNO vs FNO,
arXiv:2302.01178: r=0.37 vs 0.28, toy PDEs); ZERO on vehicle aero; CarBench's
"size beats diversity" headline is n_train-confounded (exactly what our
curve protocol removes). Our nested-subset + seeds + per-field-exponent
protocol is first-of-kind and publishable independent of any architecture
win. Standard to meet: >=4 seeds w/ error bars (arXiv:2511.01830), expect
WSS to scale worse than pressure.

THREATS: (1) AB-UPT + LoRA owns the current generalization-per-sample story
(arXiv:2605.27968: zero-shot 4.4x degradation between related families =
the published bar; LoRA-20 > scratch-103). MITIGATION: add AB-UPT as a
measured baseline on the flagship curves. (2) The pose-axis referee trap:
"vehicle aero is canonically aligned" — counter with breaking the published
robustness/accuracy trade-off (Symmetry in the Wild numbers).

DIRECTLY ACTIONABLE: HiLiftAeroML (arXiv:2605.19565) SHIPS deterministic
curated splits designed for data-efficiency/OOD measurement — our high-lift
wave should use those curated splits, not ad-hoc ones. Also: "Transolver is
a linear transformer" (arXiv:2511.06294) — slicing is efficient mixing, not
physics; theory ammunition.
