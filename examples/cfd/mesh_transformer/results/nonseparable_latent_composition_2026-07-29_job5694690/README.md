# Nonseparable latent composition — formal result

## Scientific verdict

The frozen verdict is `ordered_compression_earned_breadth_refuted`.

Physical-order rank four passed every ordered-compression gate in four of five
new model seeds. Its geometric-mean cross-channel error was 28.2% in
distribution and 30.4% at high heterogeneity, compared with 40.7% and 47.8%
for global rank four and 31.3% and 31.4% for fixed truncation. It recovered
55.9%--65.1% of the heterogeneous layer-swap response.

Breadth passed in zero seeds. On faster coefficient variation, path rank four
left 88.8% error versus 39.0% for fixed truncation; on the combined shift it
left 71.1% versus 37.8%. Every seed failed all four registered breadth gates.
The sorted control unexpectedly beat the physical-order model on both fast
splits, consistent with the learned recurrence extrapolating its smooth
training sequences in the wrong way.

The per-case rank-four oracle remained between 9.5% and 12.5%, and every
sorted model predicted layer-swap contrast at numerical zero. The result
therefore supports ordered compression but rejects the generic recurrent
update as a transferable path law.

## Provenance

- HSG Slurm job: `5694690`
- Terminal state: `COMPLETED`, exit code `0:0`, elapsed time `00:03:12`
- Learned reports: four arms × five new seeds
- Analytic reports: diagonal carrier, fixed rank four, and rank-four oracle
- Formal evaluation and order-challenge samples were distinct from both
  exploratory pilots
- Source fingerprint recorded by every report:
  `0a48b2b228ebda80100032a5426f7f0f1be28892c07ee3565277c0107409e8d1`
- Summary SHA-256:
  `3ba76ef9dcc1bc74a51f3492575d22a2495a8a4e70d1a414be367177b7fa52ea`

An independent local invocation of the frozen reducer reproduced
`summary.json` byte for byte.

## Contents

- `formal/*.json`: 23 registered reports
- `formal/*.stdout.log`: full per-report console output
- `summary.json`: frozen registered reduction
- `nonsep-latent.log`: merged batch log
- `STATUS_formal_5694690` / `DONE_formal_5694690`: terminal markers
