# Drifted screened principal-part comparison

This directory preserves the formal five-seed result for the registered
comparison of a fixed singular coefficient, a learned global singular
coefficient, and an unstructured distance-scaled kernel.

The registered verdict is `hard_fixed_coefficient_not_earned`. Singular-form
encoding is strongly supported, but hard-fixing the coefficient did not meet
the near-boundary, resolution, or operator-transfer thresholds. The learned
coefficient averaged `0.1589679`, compared with the analytic
`1 / (2*pi) = 0.1591549`.

- `formal/` contains the fifteen arm/seed reports.
- `summary.json` is the frozen reduction.
- `STATUS` records terminal execution state.

The archived summary was independently regenerated from the reports and
matched byte for byte. Its SHA-256 digest is
`b892bf306f23a6041fe46470cd44b37b4722bb754379697cbace2259c06cd270`.
The execution environment and protocol are recorded in the reports; the
research interpretation is kept in the lab notebook.
