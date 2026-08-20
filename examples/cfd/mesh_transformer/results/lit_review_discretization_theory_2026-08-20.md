# Lit review: neural-operator discretization theory (agent report, 2026-08-20)

Full text in session transcript; key content:

NOVELTY CONFIRMATIONS: our P4 query-set-dependence statistic appears to be a
NOVEL measurement (no paper names or measures it); our 2.7x passive-decoder
cost is not established anywhere; our P3 biased-sampling instrument tests a
regime absent from both theory (invariance theorems assume a FIXED sampling
measure) and practice (all published subsampling tests are uniform).

MECHANISM FORMALIZED: Transolver-class slice tokens are self-normalized
importance-sampling (SNIS) estimates of measure-conditional expectations
(Continuum Attention, arXiv:2406.06486 gives the formal license). SNIS keeps
O(1/N_eff) bias + variance inflation under skewed weights (BR-SNIS); a 10:1
bias destroys effective sample size where the physics lives -> the residual
density floor is INFORMATION-limited, matching our "no weighting removed it".

RANKED IMPORTS:
1. **Anchor-conditioned decoding (AB-UPT, arXiv:2502.09692)**: a small anchor
   set self-attends; all queries cross-attend to anchors only. Keeps
   interaction (upstream of queries) -> expected ~0 accuracy cost (their
   Table 3, flagged as progression-not-ablation) with query-independence
   at fixed anchors. Reopens our shelved v5 line cheaply: v5a3 was one step
   short (all-token encoder instead of a designated anchor core).
2. **Fixed-reference-measure normalization (LOCA arXiv:2201.01032) +
   Galerkin LINEAR aggregation (Cao arXiv:2105.14995)**: linear attention
   takes quadrature weights UNBIASEDLY (no SNIS); LinearNO (arXiv:2511.06294)
   says slicing's gains are the projections, not inter-slice attention.
3. **NEO-style log-weight bias in EVERY softmax (arXiv:2605.24390)**: the
   only published biased-sampling win. AUDIT ITEM: our v2+ weighted only the
   assignment softmax, not every normalization -- reconciliation candidate
   for why our weighting under-delivered.
