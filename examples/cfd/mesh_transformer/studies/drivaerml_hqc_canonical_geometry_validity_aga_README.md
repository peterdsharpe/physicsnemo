# Full-cohort canonical-geometry validity gate on AGA

This package runs the preregistered, target-free 36-case validity experiment
defined by
`phase1_hqc_canonical_geometry_validity_preregistration_v1_2026-07-28.json`.
It asks one narrow question: when the public cast-once canonical source bundle
is supplied symmetrically, are the frozen primary-center and fixed-center
paths raw-byte identical?

The experiment is not an H-QC rerun and is not an accuracy experiment. The
model producer may open only vehicle geometry, structural metadata, and the
five allowed global physical inputs. It may not open or index pressure, WSS,
`cell_data`, `point_data`, interior, truth, force, or other supervision. The
four lanes cover the frozen 36 cases modulo four at
`K={2500,5000,10000,20000,40000}` in bfloat16 and float32.

Each encoded path performs one coupled `S_K` trace decode. `Q` is the first
2,500 cell-identity rows sliced from that decode; it is not a standalone
fixed-coordinate query. There are exactly 720 licensing full-field tensor
comparisons. The 1,296 deduplicated panel summaries and 1,440 emitted records
are deterministic views of those tensors, not independent observations.

## Frozen launch bytes

The producer, reducer, full execution-source manifest, and preregistration
were frozen in that order after their focused and independent audits:

- producer:
  `f96e272dd5d34d0a7e7d708b312f5b2856218715f22e5a489a6acbf5192baab5`;
- reducer:
  `ea5769b927f5a3ada3030ca8849706f391b4cb0aa0c61384098e698a1538d9c9`;
- execution-source manifest:
  `e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4`;
- preregistration:
  `99bc8521c98170b1569d3471739dda13fbf23677d2c55248ddedb3969ca79501`;
- wrapper:
  `72583a100ef31ed989b2e410271b7d4eae730db9afce402140b21818e383511a`.

Do not edit any frozen file afterward. Any executable or contract edit
requires new hashes and a new prelaunch audit. Before staging, require a
placeholder scan over the producer, reducer, preregistration, wrapper, and
this README to return no matches.

The wrapper also pins:

| Input | SHA-256 |
|---|---|
| full execution-source manifest | `e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4` |
| execution source tree | `fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e` |
| target-free geometry manifest | `3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0` |
| frozen canonical helper | `694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac` |
| frozen H-QC loader | `8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f` |
| job-305691 anchor JSON | `09e336442881f0641c14c91c17dd80ac440d474f996925cf79f2174bc4cacd88` |
| job-305691 anchor NPZ | `edee836e0cc5c66690276e6787496cbd6b81fb08decb7d54ecf1f36b333ddc9f` |
| job-305691 anchor NPZ sidecar | `ecb00237458f08432976329c2ac28582a3719be49943f1d59ced30ec1d7a7b73` |
| dataset manifest | `51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca` |
| dataset config | `a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3` |
| resolved config | `a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1` |
| epoch-491 model | `4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88` |
| normalization state | `31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94` |
| training state | `3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7` |

The producer-scoped execution-tree digest is the canonical manifest hash over
Python files
below `physicsnemo/experimental/nn/mesh_attention`,
`physicsnemo/experimental/nn/symmetry`, `physicsnemo/mesh`,
`physicsnemo/datapipes`, and the top-level Python files in the recipe `src/`
directory. Both the wrapper and producer recompute it independently. It is
required in addition to the broader staged-source manifest, not instead of it.

The broader `execution_source_manifest.sha256` is a canonical, path-sorted
`sha256sum` stream over:

- every regular non-`.pyc` file recursively below `physicsnemo/`, pruning
  every `__pycache__/`;
- every regular non-`.pyc` file directly below the recipe `src/`; and
- the single recipe file `datasets/drivaer_ml_surface.yaml`.

Its paths are relative to `execution_source/`, use the standard two-space
`sha256sum` format, and end in one newline. The working-tree dataset YAML has
an unrelated uncommitted edit, so the launch bundle preserves the historical
checkpoint-compatible YAML separately at
`frozen_inputs/drivaer_ml_surface.yaml`; its required digest is
`a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3`.
Generate the manifest from the exact local code state plus that exact frozen
YAML:

```bash
canonical_manifest_path="$(mktemp)"
frozen_dataset_config=/home/psharpe/gh/physicsnemo/examples/cfd/mesh_transformer/results/phase1_hqc_canonical_geometry_validity_launch_v1_2026-07-28/frozen_inputs/drivaer_ml_surface.yaml
dataset_path=examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/drivaer_ml_surface.yaml
(
  cd /home/psharpe/gh/physicsnemo
  {
    find physicsnemo \
      -type d -name __pycache__ -prune -o \
      -type f ! -name '*.pyc' -print
    find \
      examples/cfd/external_aerodynamics/unified_external_aero_recipe/src \
      -maxdepth 1 -type f ! -name '*.pyc' -print
    printf '%s\n' "$dataset_path"
  } |
    LC_ALL=C sort |
    while IFS= read -r file; do
      if [[ "$file" == "$dataset_path" ]]; then
        digest="$(sha256sum "$frozen_dataset_config" | awk '{print $1}')"
        printf '%s  %s\n' "$digest" "$file"
      else
        sha256sum "$file"
      fi
    done
) > "$canonical_manifest_path"
sha256sum "$canonical_manifest_path"
```

The manifest stays outside `execution_source/`. The wrapper pins the
manifest's own bytes, reconstructs the same sorted stream from the staged
inventory, byte-compares it with the manifest to reject missing or extra
files, and then runs `sha256sum --check`. Copy
`$canonical_manifest_path` into the task as
`execution_source_manifest.sha256`, then remove the temporary local file
after the frozen task bundle has been verified. It is not a repository
deliverable.

## Exact task layout

Stage only after the hashes above are final. Do not run these commands while
audits are still open. The logical task path is

```text
/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1
```

and its required compute-node physical path is

```text
/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1
```

The task must be new and contain this layout before submission:

```text
2026-07-28-mt-canonical-validity-36x5-v1/
├── artifacts/
├── execution_source/
│   ├── physicsnemo/
│   └── examples/cfd/external_aerodynamics/unified_external_aero_recipe/
│       ├── src/
│       └── datasets/
│           └── drivaer_ml_surface.yaml
├── lane_logs/
├── sbatch_logs/
├── execution_source_manifest.sha256
├── launch_manifest.json
├── launch_manifest.json.sha256
├── drivaerml_hqc_canonical_geometry_validity.py
├── drivaerml_hqc_canonical_geometry_validity_adjudicate.py
├── drivaerml_hqc_canonical_geometry_diagnostic_v5.py
├── drivaerml_trace_fixed_query_audit.py
├── drivaerml_geometry_input_manifest_36cases_v1.json
├── hqc_neutral_canonical_geometry_four_canaries_k2500_v5.json
├── hqc_neutral_canonical_geometry_four_canaries_k2500_v5.npz
├── hqc_neutral_canonical_geometry_four_canaries_k2500_v5.npz.sha256
├── phase1_hqc_canonical_geometry_validity_preregistration_v1_2026-07-28.json
├── drivaerml_hqc_canonical_geometry_validity_aga.sbatch
└── drivaerml_hqc_canonical_geometry_validity_aga_README.md
```

`execution_source` must be a real task-local snapshot, not a symlink. Copy the
complete local `physicsnemo/` package, only the direct regular files from the
unified recipe's `src/`, and the frozen checkpoint-compatible dataset YAML
from this launch bundle—not the dirty working-tree YAML. Exclude `__pycache__`
and bytecode. Do not stage recipe studies, tests, tools, runs, README,
requirements, shared checkout, or venv. The wrapper rejects every symlink,
every `__pycache__` directory, and every `*.pyc` anywhere in this source
snapshot before Python import. It also rejects unregistered recipe files and
verifies the imported package location.

The geometry manifest comes from
`results/geometry_input_manifest_36cases_job305850_2026-07-28/artifacts/`.
The three anchor files (JSON, NPZ, and the canonical NPZ sidecar) come from
`results/hqc_neutral_canonical_geometry_four_canaries_v5_job305691_2026-07-28/artifacts/`.
The canonical helper and frozen loader are the exact versions preserved in
the latter job bundle. The anchor NPZ sidecar is required because the
independent reducer consumes and same-byte verifies it; other staged
scientific inputs are checked directly against frozen digests. Lane and
reducer outputs create and verify their own sidecars.

The interpreter and checkpoint remain shared, but are addressed by their
physical `/scratch/fsw/...` paths. No shared source code is imported. The
shared venv is

```text
/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
```

## Launch behavior

The wrapper requests one whole AGA node on `batch/short` for two hours:
4 GB300 GPUs, 32 CPU cores, and 768 GiB. Four concurrent
`/scratch/.../.venv-recipe/bin/python -m torch.distributed.run --standalone
--nproc_per_node=1` processes bind to GPU IDs 0–3 via
`CUDA_VISIBLE_DEVICES`, with eight CPU threads per lane. Invoking the module
through the physical interpreter is required here: the venv's `torchrun` console
script has a logical `/home/...` shebang, which is not safe for AGA native
extensions. Each lane has a 95-minute hard timeout and its own immutable log.

The wrapper fails closed before inference if the physical submit directory,
directory topology, full source manifest, producer-scoped source hash,
symlink, venv, or no-clobber check differs. A nonblocking task lock prevents
concurrent runs. A five-minute heartbeat records GPU use, completed cases,
and completed lanes, but is informational. A `K=40000` unit may legitimately
be quiet: the per-lane 95-minute timeout and two-hour Slurm wall time are the
only preregistered liveness bounds.

After it proves that the logical task directory resolves exactly to the frozen
physical task directory, the wrapper switches its runtime `TASK_DIR` to the
physical `/scratch` spelling before deriving any task-local path. This is
required for the reducer's component-wise `O_NOFOLLOW` reads and writes:
passing the logical `/home/.../coreai_modulus_cae/...` spelling would encounter
that parent symlink and correctly fail closed.

After all lane processes have ended, the reducer always receives all four
expected JSON/NPZ pairs, even when a lane failed or timed out. It independently
checks schemas, provenance, sidecars, exact NPZ array sets, primary replay,
prefix slicing, 720 licensing comparisons, 3,600 primary-replay/prefix
comparisons, 720 nonlicensing fixed-panel consistency comparisons, and 120
direct comparisons with the same-byte-verified frozen anchor NPZ. For the
terminal decision,
the wrapper reads the reducer JSON bytes once, validates the canonical sidecar
against the hash of those same bytes, and parses those same bytes; it does not
verify and then reopen the result. The wrapper then distinguishes:

| Reducer result | Meaning | Wrapper |
|---|---|---|
| valid + `CANONICAL_FULL_VALIDITY_PASS` | all 720 tensors exact | `DONE_<jobid>`, exit 0 |
| valid + `CANONICAL_FULL_VALIDITY_REFUTED` | at least one valid mismatch | `DONE_<jobid>`, exit 0 |
| `INCOMPLETE_COHORT` | required lane unavailable | `BLOCKED_<jobid>`, nonzero |
| `INVALID_COHORT` | readable instrument contract failed | `BLOCKED_<jobid>`, nonzero |

`STATUS_<jobid>` records the shell exit and completed-lane count on every
normal wrapper exit. Existing lane outputs, sidecars, lane logs, reducer
output, or terminal markers are never overwritten.

## Preflight and submit

Before any remote copy, validate locally:

```bash
bash -n examples/cfd/mesh_transformer/studies/drivaerml_hqc_canonical_geometry_validity_aga.sbatch
.venv/bin/python -m json.tool \
  examples/cfd/mesh_transformer/results/phase1_hqc_canonical_geometry_validity_preregistration_v1_2026-07-28.json \
  >/dev/null
```

After staging, independently compare the remote hashes to the frozen local
hash list, require `find execution_source -type l` to print nothing, run
cache scans that find no `__pycache__` directory or `*.pyc`, run
`sha256sum --check execution_source_manifest.sha256` from
`execution_source/`, and recompute the producer-scoped execution-tree digest
before asking Slurm for resources. Run a scheduler-only preflight first:

```bash
cd /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1
env -i \
  HOME=/home/psharpe \
  USER=psharpe \
  LOGNAME=psharpe \
  PATH=/usr/bin:/bin:/cm/local/apps/slurm/current/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  SLURM_CONF=/cm/shared/apps/slurm/etc/oci-aga-slurm-1/slurm.conf \
  /cm/local/apps/slurm/current/bin/sbatch \
  --test-only \
  --export=ALL \
  drivaerml_hqc_canonical_geometry_validity_aga.sbatch
```

Only after that command succeeds and an independent launch audit is `GO`,
repeat the same eight-variable `env -i` command without `--test-only`.
Submission is a one-shot action: do not submit a replacement merely because
the first job is pending.

Accept a live launch only after all three checks hold:

1. the absolute sbatch log exists and advances;
2. error scans over the sbatch and lane logs have no unexplained traceback,
   CUDA OOM, rendezvous, or `srun` failure;
3. at least one real `COMPLETED_UNITS=1/9` case line appears in a lane log.

The result can license only the next validity step. A pass authorizes a
separately frozen exact historical `K=10000` truth replay, followed by a new
target-using accuracy/noninferiority preregistration. It does not itself
license H-QC, accuracy, training changes, or an architectural claim.
