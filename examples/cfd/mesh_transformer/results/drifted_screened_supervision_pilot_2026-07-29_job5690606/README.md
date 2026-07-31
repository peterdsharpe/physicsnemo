# Supervision-alignment pilot

This directory preserves the underpowered 500-step signal check for the
registered comparison of pointwise, solution-only, and hybrid supervision of
one structured boundary kernel.

The pilot is not a scientific verdict. It established that all arms train
stably and that the comparison can reveal the intended distinction:
solution-only training fit in-distribution fields rapidly while producing a
large exact-trace residual and catastrophic high-screening extrapolation; the
hybrid retained a near-analytic singular coefficient and a much smaller
near-singular kernel error.

- `pilot/` contains four arm/seed reports.
- `STATUS` records successful terminal state.

The formal five-seed study uses the frozen reducer and longer registered
training budget.
