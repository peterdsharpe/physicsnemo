# Formal continuum-flow comparison

## Scientific verdict

The physical continuum flow earned discretization transfer in all five formal
seeds and earned coefficient-frequency transfer in none:

`continuum_earned_frequency_transfer_refuted`

Geometric-mean cross-channel relative errors were:

| Arm | 8 layers | 4 layers | 16 layers | 32 layers | Fast, 8 | Fast, 16 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed rank four | 0.307 | 0.328 | 0.306 | 0.306 | **0.398** | **0.389** |
| Previous recurrence | 0.274 | 0.956 | 1.026 | 1.697 | 0.869 | 1.267 |
| Sorted continuum flow | 0.974 | 0.968 | 0.976 | 0.977 | 0.769 | 0.777 |
| Physical continuum flow | **0.280** | **0.331** | **0.274** | **0.275** | 0.837 | 0.677 |
| Per-case rank-four oracle | 0.118 | 0.122 | 0.119 | 0.119 | 0.120 | 0.128 |

The physical flow's 32-layer error was 0.97--1.00 times its eight-layer error
in every seed. The previous recurrence instead reached 1.70 aggregate error at
32 layers. Thus scaling the update by physical path length cured the
discretization pathology.

That change did not yield a frequency-general local law. At eight layers the
physical flow was 2.10 times the fixed baseline and was worse than sorted flow
in every seed. At sixteen layers it was still 1.74 times the fixed baseline;
it beat sorted flow by only 8--18%, below the registered 25% threshold.

All controls were valid. The per-case oracle remained below 15% error on every
split, and sorting erased the order contrast to numerical precision.

## Provenance

The eighteen reports and frozen `summary.json` are the authoritative outputs
of HSG Slurm job `5695143`, which completed successfully in 4 minutes 39
seconds. Formal learned seeds were 29, 43, 59, 71, and 83, with evaluation and
order challenges disjoint from the pilots.

The summary SHA-256 is
`e0312e7efdffbf2f6d62f1b7d628fab930205e34aa25f328f464d9bc9dbb908a`.
All reports record the same relevant-source fingerprint,
`6c3a147e3c92b4193b35efa9b7bdffebc1fe23e7e515bd46051ce285feb183bd`.

An independent local reduction was byte-identical. That rerun exposed a
reproducibility-only bug: the reducer initially tried to parse an already
existing `summary.json` as an arm report. It now selects only per-seed reports.
The formal reduction ran before its summary existed, so the archived metrics,
gates, and verdict were unaffected.

