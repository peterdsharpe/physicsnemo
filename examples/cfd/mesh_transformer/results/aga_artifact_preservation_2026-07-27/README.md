# AGA artifact preservation manifest — 2026-07-27

This is a read-only identity pass for the completed MeshTransformer
measure-normalization and homogeneous-plus-measure-normalization runs on AGA.
It does not copy, delete, rename, or modify any source artifact.

The exact remote task directory is:

`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-27-mt-reorientation-artifact-manifest`

The CPU-only Slurm job hashes:

- the five named historical task directories referenced by the notebook and
  their relevant configs, metrics, logs, Slurm scripts, and status markers;
- three bounded source snapshots (the base remote repo and the two named
  isolated trees), excluding virtual environments and output/run trees;
- all direct checkpoint files, resolved config, `metrics.jsonl`, and training
  log for the five H-BOTH and five same-tree measure-normalization seeds
  42–46, plus the two explicitly named earlier measure-normalization runs;
- the exact top-level dataset layout and known manifest/units files for the
  training DrivAerML tree, ID-reference shadow, and SHIFT-SUV pilot.

Raw dataset case files and TensorBoard event files are deliberately excluded.
The resulting manifest is an identity record, not a byte-for-byte archive;
the original files remain at their existing AGA paths.

## Completed jobs

- Job `302461` completed on `cpu/cpu-short` with exit code 0 in 1m18s.
  Its 42 MB manifest retained every discovered task-directory JSON identity,
  including tens of thousands of prediction-tree metadata files. It remains
  on AGA but was superseded for local retrieval because that inventory was
  broader than the requested compact pass.
- Job `302492` completed on `cpu/cpu-short` with exit code 0 in 43s.
  The compact manifest is complete, records zero collection errors, and
  covers 5 task directories, 3 source snapshots, 3 dataset roots, 12 named
  runs, 67 grouped evaluation metric files, and 1,212 checkpoint files.

The locally retrieved compact manifest SHA-256 is
`156754b0fe0e90a8dc11799ac2675b5b4faf784f90b71c725a090b9c0a49f2c3`.
The job log SHA-256 is
`c0649146316b0561f8f5048413f78db210c86b6a8a5b8200b7451feb4da5892e`.

Raw predictions do exist in the historical harvest tree. A targeted check
found TensorDict `.memmap` arrays for predicted/true pressure and WSS plus
geometry under
`2026-07-25-mt-wave-harvest/artifacts/homog_s42/res10000/.../predictions/`.
Those large arrays were not recursively hashed by the compact job.

## Historical S1 WSS metric reconstruction

CPU-only job `302553` completed on `cpu/cpu-short` with exit code 0 in
21 seconds. It read all 20 historical S1 evaluation groups: MeshTransformer
and GeoTransolver, seeds 42–46, each on the 24-case ID-reference and 98-case
extreme-test splits. All 1,220 saved cases were processed, with zero missing
cases, malformed JSONL rows, diagnostic failures, or provenance failures.
The source artifacts were not modified.

The producing evaluator is identified by exact content:

- `src/metrics.py` SHA-256
  `b3857182f6e0f9bb7b41d4cd685530835f831e310784668e1873420acd1f8110`;
- `src/infer.py` SHA-256
  `92ccda928a5b8c5ebfe3f0add99a8d023662153b4a6e10708ffcc0ce12ec16eb`;
- all ten resolved configs, epoch-491 checkpoints, normalization states,
  launchers, final Slurm logs, and five failed-attempt logs are individually
  hashed in the result.

The historical `_relative_l2` reduced an `(N, 3)` WSS tensor over its last
axis and then averaged over points. Thus logged bare `wss_l2` was
`mean_i(||e_i|| / (||y_i|| + 1e-8))`, despite a nearby code comment calling
the aggregate flattened/Frobenius. Recomputed values match the final logged
summaries to at most `2.71e-7` by group and `1.95e-6` by case.

Extreme/ID degradation ratios from the saved arrays are:

| definition | MeshTransformer, range (mean) | GeoTransolver, range (mean) | lower-MT 5v5 separation |
|---|---:|---:|---|
| historical pointwise mean | 1.071806–1.112527 (1.091755) | 1.125986–1.136442 (1.131503) | yes; gap 0.013459 |
| corrected whole-field Frobenius | 1.026218–1.048026 (1.037132) | 1.021310–1.027552 (1.024810) | **no**; ranges overlap |
| vector-magnitude relative L2 | 1.026367–1.047291 (1.038155) | 1.018218–1.026704 (1.021191) | no |
| x-component relative L2 | 1.039648–1.053857 (1.046882) | 1.028910–1.037749 (1.032087) | no |
| y-component relative L2 | 1.016637–1.044324 (1.031966) | 0.986471–0.998335 (0.990525) | no |
| z-component relative L2 | 1.019714–1.045360 (1.031411) | 1.050079–1.064664 (1.056515) | yes; gap 0.004719 |

The reported pointwise 5v5 separation is reproducible, but the headline
differential does not survive the corrected whole-field Frobenius definition:
the ranges overlap, GeoTransolver has the lower degradation ratio at all five
same seed labels, and its mean is lower. This is a descriptive reconstruction;
matching seed labels and identical evaluation-target hashes do not alone
declare a paired or independent inferential sampling model.

The content-addressed local result is
`s1_wss_metric_reconstruction_2026-07-27_0ba01541f67d34f6.json`, SHA-256
`0ba01541f67d34f64a1588cd09c3ee3af94c9e84b3cbb30ce9de1b2f43444c27`.
Its scoped identities are:

- raw WSS arrays plus TensorDict metadata:
  `a144cb52b06d265ccdd00bcd3269655b599e9779d2bdc7cc8e89508ddae10152`;
- evaluator logs plus `metrics.jsonl`:
  `b61b67ea91b0d60f8c46f76266ed271de3d184dc52397b8766a95a3a6d4b5d25`;
- producing source, configs, checkpoints, and job evidence:
  `7d9f72e087bd047a9331df9068bedf51670c7efa0a5a928b1978d2ebb32bffd7`.

The job log SHA-256 is
`0cb2f4e95ca9503440a439c25fff7dc09aa3cfcd42bdae00e3e421cd411e59c2`.
The base AGA source snapshot has no `.git` metadata; these are exact content
identities, not an unavailable commit-SHA claim.
