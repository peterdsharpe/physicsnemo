# SHIFT-SUV adaptation pilot

## Scientific purpose

Test whether a DrivAerML-trained initialization reduces the target examples
needed to learn SHIFT-SUV estate and fastback fields, conditional on one source
checkpoint. Compare the within-architecture pretraining benefit for
MeshTransformer and GeoTransolver. This is not an untouched-family test and
does not identify physics-specific transfer.

The pilot uses only source `train` and `val` cases. No test case appears in the
task manifests.

## Current stage

The preregistered validation-only pilot is collecting training checkpoints.
No cross-arm validation ranking has been inspected. Three pretrained
MeshTransformer arms at the largest registered learning rate reached the
prospective non-finite guard; their fully completed epoch-1 checkpoints remain
eligible under `terminal_events.json`, and the other registered arms continue.
Formal selection will occur only after every eligible checkpoint has been
re-evaluated on the two fixed validation samples.

Before outcomes were inspected, the interpretation gained one further gate:
if either selected checkpoint in a half-label comparison is the final stored
epoch, that comparison is evidence about fixed-exposure learning speed rather
than target-label efficiency. A longer prospective horizon is then required
before claiming that fewer simulations suffice.

The same pre-outcome audit now flags optimizer-grid censoring. If either
selected learning rate lies at an edge of the registered grid, the result
remains valid within that grid, but confirmation must add a stable flanking or
intermediate rate before supporting an optimized label-efficiency claim.

An outcome-independent geometry study has also completed using only the
declared training meshes. DrivAerML was 1.209 times farther from estate than
fastback under the frozen morphology descriptor; the bootstrap interval for
the estate-minus-fastback distance stayed above zero. Geometry-scaffold reuse
therefore predicts a materially larger pretraining gain on fastbacks for both
architectures. Adaptation outcomes remain uninspected.

Before those outcomes were inspected, the formal reducer was extended to
evaluate that joint prediction directly: it reports each architecture's
geometric-mean fastback/estate pretraining-gain ratio, resampling the two
families independently while retaining all within-family pairings.

The earlier canary history is retained below because it explains the frozen
execution source and the preregistration amendments.

The staged source snapshot hashes are:

- `physicsnemo/`: `17a84aa9d0427e50b2dfb6a5d32ff4720a9df2a470b79bd5007653875da63258`
- recipe `src/`, `conf/`, and `datasets/`:
  `bf17c217d8934a609890030a31937b829d1b2c9b2898bfae5d39ed739a4f234b`

The model-only warm-start path was checked independently: two unequal random
initializations became byte-identical after loading each source checkpoint,
while the absent optimizer/scheduler state correctly left the returned epoch
at zero.

The first accepted-path canary exposed a non-finite first loss for the
pretrained MeshTransformer on a fastback training case. Before changing the
adaptation protocol, the follow-up diagnostic distinguishes:

- reduced-precision overflow: fastback becomes finite in float32;
- family-specific source failure: estate is finite but fastback remains
  non-finite in float32; and
- a broken target pipeline: a random MeshTransformer is also non-finite on
  fastback.

The diagnostic uses only the same training-origin canary cases. It does not
inspect the scientific validation split.

Both float32 target lanes were also non-finite, while a random fastback model
completed. This does not yet establish architectural brittleness because the
checkpoint predates substantial model-code changes. The next compatibility
control uses the exact MeshTransformer implementation at commit `0520f294`
(its `model.py` and `kernel_decoder.py` hashes match the AGA source snapshot
that produced the checkpoint). It asks, in this order:

1. Is the checkpoint finite on two DrivAerML source cases?
2. If so, is it finite on the estate and fastback training-origin canaries?
3. Does a random model under the same historical implementation remain finite?

Target failure is interpretable only if the source-domain control passes.
The historical-code lane also uses the checkpoint-era dataset-side intrinsic
gauge: geometric cell areas only, matching the fallback taken before the
effective-measure module existed. This prevents a newer sampling-weight
contract from entering the compatibility control.

If that source-domain gate passes, `validation_preregistration.json` freezes
the 48-run pilot before validation outcomes are inspected: two architectures,
scratch versus DrivAerML initialization, two body styles, nested 64/128-case
target budgets, and a symmetric three-learning-rate grid. The primary
scientific check is whether 64 pretrained cases materially outperform 128
scratch cases on both body styles. The test splits remain sealed.

The gate passed in job 5701700. Under the checkpoint-era implementation, the
source checkpoint was finite on DrivAerML and on both target styles; the
earlier current-code NaNs were therefore a checkpoint/code compatibility
artifact. Job 5701874 then completed eleven eager fine-tuning steps at the
10,000-cell source resolution for both target styles and both ends of the
registered learning-rate grid without a non-finite loss. Only training-origin
canary cases were used.

Before any registered validation case was read, the preregistration was
amended from 200 compiled epochs to 70 eager epochs. Eager execution took
about 1.0--1.1 seconds per MeshTransformer training case after startup,
whereas shape-specialized compilation was dominated by repeated compilation.
The 70-epoch design preserves equal target exposure across label budgets and
fits the 128-case arms inside HSG's four-hour limit.

## Relaunch

From this directory:

```bash
sbatch canary_hsg.sbatch
```

After the historical source-domain gate passes:

```bash
sbatch pilot_hsg.sbatch
```

The pilot uses restart-safe one-hour short-queue segments because the normal
queue wait exceeded the run time. Re-submit the unchanged array until every
task has a `DONE_PILOT_*` marker; stable run directories restore the latest
completed epoch, and the registered comparison still uses only completed
epochs 1, 11, 21, 31, 41, 51, 61, and 70.

After the first terminal learning-rate arm caused its four-run array task to
trip the job-level stalled-log guard, execution moved to
`pilot_hsg_pool.sbatch`. Its eight fixed lanes resume the same 47 nonterminal
registered runs from their existing checkpoints and omit only the arm already
recorded in `terminal_events.json`. The scientific design and selection rule
are unchanged.

A learning-rate arm that triggers the prospective non-finite-loss guard is
terminal rather than missing. Its fully completed registered checkpoints
remain eligible for validation selection, and unavailable later checkpoints
have infinite selection error. This clarification was recorded before any
cross-arm validation ranking was inspected and applies symmetrically.
Because segmented restarts truncate each run's text log, the first such event
is also written once to `terminal_events.json`. The formal reducer treats that
ledger as authoritative and discards any later records produced by an
automatic restart of an already-terminal arm.

A later code-path audit, also before any cross-arm ranking, found that the
training loop's random surface subsample was not common across epochs and
label budgets. Those online validation values are therefore integrity
diagnostics only. Formal selection re-evaluates every eligible stored
checkpoint on two declared validation-only point samples, common to every
arm, and averages them within case.

The fixed evaluator completed 99/99 cases in its end-to-end canary. Repeating
the same checkpoint and evaluation seed while changing the unused training
split from 64 to 128 cases produced an identical hash of all 99
pressure/WSS metric pairs, confirming that formal validation no longer
depends on the label budget's training iterator.

The first full fixed-validation launch failed before producing a result:
Python's default CSV line terminator left a carriage return on the final TSV
field, which Hydra rejected as part of the integer evaluation seed. The
manifest writer now requests `lineterminator="\n"`. The regenerated manifest
is byte-identical to the failed manifest after newline normalization and
contains zero carriage returns. Failed-launch markers and logs are preserved
under `failed_launch_5712721/`; retry 5712792 was accepted only after new
validation result records appeared. This changes no evaluation unit,
checkpoint, seed, split, or selection rule.

Retry 5712792 completed all 726 registered validation evaluations with zero
invalid records. Two independent formal reductions were byte-identical
(`f57eb93618ff2ec90089b5e21528a0a9ce5f74c1435309ff2adb124fea033bd1`);
the complete result is `analysis/formal_validation_summary.json`. The sealed
test splits remain unopened.

One restart landed immediately on a node used by the preceding segment and
reused its fixed rendezvous port. Seven lanes progressed, while the affected
lane failed before training; the watchdog stopped the segment after 30
minutes. The valid checkpoints are retained, and subsequent pool segments
derive a disjoint four-port block from each Slurm job ID. This changes no
training or selection rule.

The task reuses the immutable dependency environment from
`2026-07-21-crossfam-hsg/repo/.venv-recipe` while placing the current source
snapshot first on `PYTHONPATH`. Job-local checkpoints, compiler caches, logs,
status markers, and artifacts remain inside this task directory.
