# Similarity-coordinate formal result

This directory contains the clean five-seed result for the registered
raw-versus-similarity coordinate factorial.

## Scientific result

The registered verdict is `similarity_coordinates_not_principal`.

Supplying the exact dimensionless groups of the drifted screened kernel did
not improve general operator transfer. Under both pointwise and hybrid
supervision, similarity coordinates failed every interpolation guard and all
three joint field-and-trace operator-transfer criteria. They did improve
high-screening trace error substantially and high-screening field error
modestly, but worsened interpolation, held-out boundary modes, refinement,
low-screening fields, and stronger-drift fields.

The result rejects the hypothesis that the main extrapolation problem was
having to discover the products and square root that form the similarity
variables. It does not reject the value of those variables for a
high-screening specialist. The next scientific test should encode limiting
response forms explicitly rather than replace one generic coordinate chart
with another.

## Contents and provenance

- `formal/` contains the twenty coordinate-by-loss-by-seed reports.
- `summary.json` is the frozen formal reduction.
- `STATUS` records successful terminal completion.

The local independent reduction is byte-identical to `summary.json`, whose
SHA-256 digest is
`6790c10bbf8973dc5fb3f4b6a9968b923eb0b8e7639fa3c829e8e623ac25afb8`.
All reports share source digest
`f1fa1e768f7022a4f762c7cff4e4e964deffeb7668c34e8e36fbf3ca78f98de5`.

All ten raw-coordinate controls exactly match the corresponding pointwise and
hybrid reports from the preceding supervision study across training history
and every scientific metric.
