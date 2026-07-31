# Coupled physical-order composition

## Scientific question

Does physical-order composition remain useful when local heterogeneous
coefficients couple two field channels, or was the preceding success confined
to independently propagated scalar modes?

## Why this is the next ledge

The layered scalar study established that ordered samples are not enough:
their local propagators must be composed. It did not test changing local
eigenspaces or cross-channel response. A two-component positive-definite
coefficient field is the smallest system that introduces those effects while
retaining an exact reference.

This problem matters because nonseparable two-dimensional coefficients mix
transverse modes. A failure here would stop the program before a much more
expensive full-field experiment. A success would justify that next step.

## Registered comparison

The four arms are:

- a position-aware, mean-pooled predictor of the full boundary response;
- a rotation-equivariant learned local generator composed after sorting
  layers;
- the same generator composed in physical order; and
- the exact block path product as a numerical oracle.

All learned arms use about 4.2k--4.4k parameters, the same training profiles,
full boundary-response targets, update count, and five seeds. Training uses
slowly rotating local eigenspaces. Evaluation holds out faster twists, doubles
the number of layers, and swaps two layers while preserving the coefficient
multiset exactly.

Physical-order composition earns the claim only if it recovers at least 80%
of the matched-permutation effect in four seeds, halves paired error relative
to pooling, and remains within 20% of pooling in distribution. Breadth
transfer separately requires at least twofold lower cross-channel error on
both faster twists and doubled layer count, again in four seeds. The sorted
control must predict zero permutation contrast and the analytic oracle must be
exact.

## Risks and salvage

The learned spectral map may fail even within the training coefficient range,
which would confound composition with local identification. Its local-map
error is therefore measured separately. The paired effect may also be too
small to identify; the reducer rejects the study if the true contrast is below
1%. Either failure still identifies what must be fixed before attempting a
nonseparable field.

## Cost and exam

The pilot uses five short reports. The deciding comparison uses twenty
arm-by-seed reports on one four-GPU node and should complete well inside a
two-hour short-QOS allocation. The frozen reducer encodes every acceptance
gate above.

## Execution layout

- `code/`: staged source
- `artifacts/pilot/`: underpowered signal check
- `artifacts/formal/`: twenty registered arm-by-seed reports
- `artifacts/summary.json`: frozen formal reduction
- `sbatch_logs/`: execution log
- `STATUS_*` / `DONE_*`: terminal state

Each mode refuses to overwrite prior reports.

## Status

The registered 4,000-step, five-seed comparison completed cleanly. The frozen
verdict is `coupled_path_composition_earned`: all composition and breadth
gates passed in all five seeds. The physical-order model recovered
98.3%--99.5% of the matched-permutation contrast and reduced cross-channel
error relative to pooling by factors of 44 on unseen twists and 38 with twice
as many layers. The analytic oracle was exact and the sorted control predicted
exactly zero permutation contrast.

An independent local reduction reproduced the formal summary byte for byte.
The earlier five-report, 500-step pilot remains a signal check only.
