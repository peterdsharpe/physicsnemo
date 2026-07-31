# SHIFT-SUV matched-update convergence extension

## Scientific question

The parent adaptation pilot found same-budget benefits from DrivAerML
initialization, but no half-label success. Every selected model was still best
at the final epoch, and most selected learning rates were on the search-grid
boundary. This follow-up asks the narrower question: after matching target
optimization, can a pretrained model using 64 target vehicles replace a
scratch model's additional 64 vehicles?

The prospective protocol is `preregistration.json`. It compares only the four
half-label questions: two architectures and two target families, each with a
pretrained model using 64 target cases and a scratch model using 128.
Because it has no matched-update scratch-64 arm, it cannot decide whether a
smaller same-budget pretraining gain persists after convergence.

## Design

- Start from the exact epoch-70 checkpoints selected by the complete parent
  validation reduction.
- Load model weights only and reset optimizer/scheduler state symmetrically.
- Sweep continuation learning rates `1e-4`, `3e-4`, `1e-3`, and `3e-3`.
- Continue pretrained-64 for 210 epochs and scratch-128 for 70 epochs, so both
  end at exactly 17,920 total optimizer updates.
- Save five-epoch checkpoints. The early and middle checkpoint pairs differ by
  only 64 updates; the final pair is exact.
- Select and compare on the same two fixed validation-only surface samples
  used by the parent pilot. Test splits remain sealed.

The broad half-label branch advances only if pressure
`pretrained_64 / scratch_128 <= 0.95`, its paired-case 95% upper bound is below
one, WSS degradation is at most 10%, and the final learning curve passes the
predeclared plateau gate on both body families.

Before any continuation validation result existed, a logical audit froze two
analysis clarifications in `analysis_contract_clarification.json`: each stage
selects learning rate only among checkpoints at that stage's update budget,
and the plateau gate uses absolute relative change. The contemporaneous
`prospective_forecast.json` records the parent-evidence forecast: half-label
success is unlikely for MeshTransformer, while GeoTransolver remains a
genuinely discriminating outcome.

Before any continuation result was inspected,
`prospective_interpretation_amendment.json` withdrew the broader
*optimization-only* interpretation of a negative half-label result.
`prospective_persistent_gain_preregistration.json` freezes the missing
scratch-64 comparison that becomes active if the half-label gate closes or is
inconclusive. If that comparison finds a material persistent gain in any
architecture-family cell,
`prospective_label_equivalent_bracketing.json` freezes an adaptive,
threshold-directed scratch learning curve to measure how many target vehicles
the gain is worth.

## Provenance

Parent task:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
```

This task reads the parent's immutable code snapshots, manifests, selected
checkpoints, and fixed evaluator. It writes all new checkpoints, logs, and
results under:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-30-shift-suv-convergence-extension
```

The training manifest is generated from the parent formal result and refuses
any parent-result hash mismatch, non-epoch-70 selection, incomplete selected
run, missing checkpoint, or unsealed parent test policy.

## Relaunch

From the task directory, submit the unchanged restart-safe pool until every
lane is complete:

```bash
sbatch convergence_hsg_pool.sbatch
```

Each short-queue segment resumes from the latest five-epoch checkpoint. A
prospective non-finite-loss guard makes only that continuation-rate arm
terminal; earlier registered checkpoints remain available for validation.

## Current status

Pre-registration and training manifest are frozen. Job `5714604` was accepted
after all eight lanes wrote a real continuation checkpoint with no error,
blocker, or numerical-terminal marker. Four restart-safe continuation segments
(`5714746`, `5714765`, `5714766`, and `5714767`) are dependency-chained behind
it. Formal validation has not started, no continuation metric has been used for
a scientific decision, and the test splits remain sealed. The fixed-validation
manifest guard, launcher, and formal reducer are staged but cannot create a
manifest until every training arm is either complete or prospectively
terminal.
