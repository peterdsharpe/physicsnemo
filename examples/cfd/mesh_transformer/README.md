# ISLA research program

This directory holds the research record of ISLA (Invariant Slice
Attention), the equivariant slice-attention surrogate for steady
boundary-value PDE problems that lives in
`physicsnemo/experimental/nn/isla/`. The trainable recipe is the unified
external-aerodynamics recipe under
`examples/cfd/external_aerodynamics/unified_external_aero_recipe/`
(model configurations `isla_*.yaml`).

- `book/` — the Quarto book: the architecture (chapter 2), the benchmarks
  and their protocol, the evidence chapters, the design-decision ledger
  (appendix), the program status and the dated lab notebook. Render with
  `quarto render` inside `book/`.
- `results/` — dated result artifacts (JSON reductions, preregistrations,
  archived run scripts) that the book's figures and tables are generated
  from. Measurements and provenance manifests are retained unchanged;
  identical source snapshots and binary payloads share copies through
  relative links. See [ARCHIVE.md](ARCHIVE.md) for the layout and the
  retrieval instructions for obsolete artifacts retained in Git history.
- `research/` — the transfer-program and cross-dataset campaign artifacts
  the book cites by path.

The program's first architecture, the exact-kernel MeshTransformer (MT1,
2026-07), and its Laplace benchmark suite were removed from the tree on
2026-09-14 as obsolete. The last commit carrying that code is tagged
`mt1-final`; the book keeps its results as history and states where it was
superseded.
