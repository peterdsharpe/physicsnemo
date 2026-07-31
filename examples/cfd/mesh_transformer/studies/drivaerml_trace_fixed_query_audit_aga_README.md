# H-QC aligned-trace fixed-query audit, producer schema v2

This is the deciding correction to the historical H-CC resolution sweep. The
old sweep did not hold its sampled source or trace queries fixed: its stale AGA
reader drew a non-wrapping contiguous block with
`torch.randint(0, N-K+1, ...)`, so changing `K` also changed the random start.
The saved arms therefore cannot be repaired by slicing their first rows.

The producer reconstructs the exact historical 10k start for every one of the
36 `id_reference` cases from the seed-42 loader chain (reader generator seed
45), then defines one explicit cyclic order per case. The source arms
`K={2500,5000,10000,20000,40000}` are strict prefixes of that order. The model
still decodes its required trace query at every source cell, but every arm is
also scored on the identical first 2,500 raw cells, `Q=S_2500`.

The primary arm retains the production preprocessing frame: `CenterMesh`
recomputes the unweighted compacted-vertex centroid at every `K`. A diagnostic
arm applies the `S_10000` preprocessing center at every `K` and must agree with
the primary pressure prediction and uniform/area-weighted pressure errors
within `1e-3`, separately on both the coupled `S_K` output and the fixed
`Q=S_2500` slice.

That diagnostic checks translation invariance only. MeshTransformer itself
recomputes an area-weighted source center from `S_K`; freezing the external
`CenterMesh` translation does not freeze this internal, source-dependent
frame. H-QC therefore tests the effect of changing source context—including
that internal normalization—while holding the scored raw cell IDs fixed.

The first schema-v1 canary (AGA job 303767) stopped before publishing any
model-output artifact. Its raw-frame normal check compared normals recomputed
before and after float32 centering and scale conversion with an invalid
componentwise `2e-6` tolerance. A frozen 36-case × 5-resolution × 2-center
diagnostic found a maximum vector difference of `0.003404` but a minimum dot
product of `0.9999942` and no winding reversals. Schema v2 instead checks each
pipeline normal against its triangle in the actual transformed frame, checks
unit length and native/pipeline winding separately, and preserves both
pipeline-normal tensors for independent reduction. Raw native normals remain
the authority for physical metrics and fixed-cell identity.

## Frozen execution inputs

The wrapper and producer independently fail closed on:

- audit producer SHA-256
  `8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f`;
- historical 36-case manifest SHA-256
  `51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca`;
- historical dataset config SHA-256
  `a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3`;
- resolved training config SHA-256
  `a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1`;
- deployed epoch-491 model
  `MeshTransformer.0.491.mdlus`, SHA-256
  `4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88`;
- companion training state SHA-256
  `3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7`;
- normalization statistics SHA-256
  `31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94`;
- archived 10k metrics SHA-256
  `423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a`;
- the exact historical execution source-tree manifest SHA-256
  `fa6a7b683fa9aa02e4537ef69e8e977906df7c9fa6964cb759edfcee8d7b90cd`.

Each case additionally hashes its full raw points, cells, pressure, WSS, and
metadata sources. The producer checks its replayed 10k selection and compacted
connectivity against the saved artifact, checks saved physical pressure within
`1e-3` and WSS within `1e-5`, reconstructs saved coordinates within `1e-6`,
checks saved normals against the current 10k pipeline normals within `2e-6`,
and reproduces each archived pressure metric within `1e-3`. The reducer checks
the 36-case archived pressure mean within `5e-6`.

The fixed-Q and S10k references record both total native area and mean native
cell area. Their representativeness gate compares the means: comparing totals
would mechanically insert the fourfold row-count difference
(`2500/10000`) and make a representative Q fail by construction.

## Metrics

All model metrics use the training-space predictions and targets before
re-dimensionalization. Each resolution reports a coupled `S_K` score and a
fixed-`Q` score:

- uniform pressure relative L2, using the exact historical pressure formula;
- signed centered pressure correlation, positive-gain pattern error, and
  amplitude ratio;
- a newly recomputed whole-field WSS Frobenius relative L2;
- predicted WSS normal-energy fraction;
- the native-triangle scaled-subset pressure-force relative error; and
- native-area-weighted pressure relative L2.

The whole-field WSS Frobenius metric is a new secondary diagnostic, not a
replay of the historical `wss_l2`. The stale historical implementation reduced
vector components per cell and then averaged cells; its pressure metric was
unaffected.

## Launch

Create the fresh persistent AGA task directory
`/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-27-mt-hqc-fixed-query-v2`.
Do not reuse the failed schema-v1 directory. Copy these four files into it:

```text
drivaerml_trace_fixed_query_audit.py
drivaerml_trace_fixed_query_audit_aga.sbatch
drivaerml_trace_fixed_query_audit_aga_README.md
phase1_hqc_preregistration_v2_2026-07-27.json
```

Then launch from that directory:

```bash
mkdir -p sbatch_logs
sbatch drivaerml_trace_fixed_query_audit_aga.sbatch
```

The wrapper verifies the frozen producer and preregistration SHA-256 digests
before requesting any inference output. The schema-v2 preregistration is frozen
at SHA-256
`88560f29c8ba970f6f17654d99f13309c0d8831e28d90139aea8ea41314f53db`.

The job requests one four-GPU GB300 node for one hour. It runs four independent
single-process lanes concurrently, one per GPU, assigning cohort ordinal
`i` to lane `i mod 4`. It records a five-minute GPU heartbeat, per-lane logs,
monotone completed-unit lines, a merged Slurm log, `STATUS_<job>`, and a
`DONE_<job>` marker only after all four lanes and checksums pass.

No output is reused or overwritten. The wrapper refuses to launch if any lane
target or checksum sidecar already exists. After a partial failure, preserve
the completed lanes and launch only the missing lane manually with distinct
output names (or copy the task inputs to a new task directory); never delete or
overwrite a valid lane to make a rerun fit.

## Artifacts

Each lane atomically publishes:

```text
artifacts/phase1_hqc_producer_lane<L>.json
artifacts/phase1_hqc_producer_lane<L>.json.sha256
artifacts/phase1_hqc_producer_lane<L>.npz
artifacts/phase1_hqc_producer_lane<L>.npz.sha256
```

The JSON is the reducer contract. The NPZ preserves ordered raw cell IDs,
compacted connectivity, raw centroids, native normals/areas, primary and
fixed-center pipeline normals, training-space truth, primary predictions,
fixed-center predictions, and both query tensors for independent rescoring.
The JSON records the NPZ hash, producer hash, resolved-config hash, exact
command, and software versions.

This audit supports a statement only about the named epoch-491 checkpoint and
the frozen 36-case `id_reference` cohort. It is not an architecture-wide or
dataset-wide convergence claim.
