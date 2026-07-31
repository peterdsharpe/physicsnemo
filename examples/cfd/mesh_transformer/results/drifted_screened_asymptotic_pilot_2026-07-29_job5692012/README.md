# Two-limit asymptotic-carrier pilot

This directory preserves the underpowered 500-step signal check for the
registered comparison of a raw learned kernel, a fixed matched-asymptotic
carrier, and that carrier with a learned transition correction.

The pilot is not a scientific verdict. It established that the learned carrier
trains stably and that the comparison is strongly discriminating. The fixed
carrier sharply improved strong-screening behavior but remained worse than the
mature raw model is expected to be in interpolation. At this short budget, the
learned correction improved both the carrier's interpolation error and its
operator-parameter transfer.

- `pilot/` contains the four registered signal-check reports.
- `STATUS` records successful terminal completion.

The five-seed, 4,000-step formal comparison is required for any belief update.
