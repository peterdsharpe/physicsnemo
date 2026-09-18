# Book style guide

The book is an instructive guide to the ISLA program in the manner of a
Drela user guide (TASOPT, XFOIL): it teaches the reader what the method is,
why each part is there, how to use it, and what the evidence says, with a
clean throughline and optional depth folded away. It is written for an
*uninitiated expert*: fluent in PDE numerics and machine learning, new to
this program. This file fixes the conventions every chapter follows.
`CLAIMS.md` remains the source of truth for numbers; this file governs
presentation.

## The arc

| Part | Chapters | The reader leaves knowing |
|---|---|---|
| Front matter | `index.qmd` | the thesis in one paragraph, the trade in one picture, the names and colors, how to read the book |
| I The problem | `01-problem.qmd` | what a boundary-to-field surrogate is, the datasets, the metric, the protocol |
| II The architecture | `02-isla.qmd`, `04-baselines-and-fair-comparison.qmd` | how ISLA computes, one idea at a time; what the baselines are and what "fair" means |
| III Using ISLA | `09-using-isla.qmd` | the configurations, the inputs, the recipe, the learning-rate rule, the inference sampler, checkpoints, cost |
| IV Evidence | `03-benchmarks.qmd`, `05-…`, `06-…`, `08-…` | the standing on one page; then the studies behind each row |
| V Limits and status | `12-…`, `13-program-status.qmd` | what each claim does not say; what is running |
| Appendices | `17-…`, `18-notebook.qmd` | the design-decision ledger; the dated record |

Chapters state current knowledge. History (what was believed before, what
was retracted) lives in the notebook and the status chapter only.

## Teaching pattern for every section

1. **Lead with the message** in one bold sentence a reader can quote.
2. **Words, then picture, then symbols.** Describe the mechanism physically,
   point to a figure, then write the equation. Never open with an equation.
3. **Concrete before abstract.** Use the running example (the Joukowski
   airfoil of Part I, `palette.joukowski()`) or one car/wing before the
   general statement.
4. **Define at first use, once per chapter.** A term is defined the first
   time a chapter uses it, even if an earlier chapter defined it.
5. **One idea per figure; a "How to read this" clause in every caption**
   stating what the figure does *not* show.
6. **Optional depth goes in a dropdown**, never inline:
   ```
   ::: {.callout-note collapse="true" title="Why the softmax runs over points, not slices"}
   …
   :::
   ```
   Use `callout-note` for derivations and detail, `callout-tip` for practical
   advice in Part III, `callout-warning` for a misreading to avoid. The
   throughline must read completely with every dropdown closed.
7. **Numbers live in tables or on their own line**, computed from the
   artifact (`#| output: asis`), with the artifact path in the caption.
   Prose carries magnitudes only when they change what the reader does.
8. **Every cross-architecture claim carries its scope** (dataset, rung,
   instrument) in the same sentence, and every ratio is *baseline ÷ ISLA*.

## Nomenclature (canonical names; use these and no synonyms)

| Say | Never say | Meaning |
|---|---|---|
| ISLA | "ISLA, reference configuration", "the reference configuration", MT2, MeshTransformer2 | the definitive model as shipped (`frame_mode="relative"`, `scale_mode="total_measure"`, the class defaults); its configuration is described once in 02 and 09 and never used as a name. A variant is named by its variant name |
| the relative frame | "the reference frame", "the frame-free frame" ("reference frame" reads as a coordinate frame to a PDE reader) | the reference configuration's construction of position: `frame_mode="relative"` |
| the legacy centered variant (diagnostic) | "the constant-gauge variant" in a main result; "prior reference"; "centered ISLA" | `frame_mode="centered"`, `scale_mode="reference_length"`; cut from every main figure and table on 2026-09-18; appears only in collapsed diagnostic notes, labeled |
| form; one-car floor | | a structural choice of the architecture (as opposed to capacity or inputs); the converged error on a single training car (a capacity measure) |
| the sample-frame probe | "the probe instrument" alone | the 10,000-cell uniform and 10:1-biased DrivAerML evaluation draws, each model's frame taken from the drawn sample |
| $M$ slices, $S$ global scalars | $S$ for the slice count | symbol discipline: $M = 256$ slices; $S$ is reserved for global scalars |
| the similarity-gauge variant | | `frame_mode="centered"`, `similarity_gauge=True` |
| the weights-off ablation | "uniform ISLA" | `use_measure_weights=False`; diagnostic, never adopted |
| the raw-coordinate diagnostic | | `raw_coord_channel`; breaks the contracts by design |
| GeoTransolver, Transolver, GeoTransolver-volume | "GeoTransolver, released"; "the released baselines"; "the baselines" without naming them; "GT"/"T" outside code | the definitive public models, never modified; no qualifier |
| GeoTransolver + ISLA's <feature> (ablation) | "research variant", "GeoTransolver, measure-weighted pooling" | a baseline with one or more of ISLA's features added to isolate an effect (e.g. "GeoTransolver + ISLA's measure-weighted pooling + HT-centered datapipe (ablation)"); never a variant of the public model |
| unit-direction protocol | "udrv" | every model receives the freestream as a unit vector |
| global vector inputs, global scalar inputs | "drive", "freestream feature" | the problem's global conditions; $K$ vectors, $S$ scalars |
| boundary cell / point / token | | one sampled surface element and its token |
| measure weight $w_i$ | "area", "quadrature weight" alone | the surface area a sampled cell stands for |
| slice, anchor | "cluster", "centroid" for slices | a soft group of tokens; its measure-weighted mean position and normal |
| relational invariants | "geo features" | the scalars describing a token relative to an anchor |
| routing / read-back | | softmax over points into slices / softmax over slices back to points |
| length scale $L$ | "gauge" in prose | the square root of the total measure |
| DrivAerML (cars), HiLiftAeroML (wings) | | the two datasets; "HiLift" allowed after first use in a chapter |
| the $n$-case rung | | a training-set size on the data-efficiency ladder |
| area-weighted relative L2, float32 instrument | "rel-L2" in prose | the metric and its reporting precision |
| the deficit anatomy | | the decomposition of the in-distribution gap into named parts |

Symbols: $x_i, n_i, w_i$ inputs; $\hat g_k$ global unit vectors; $\sigma$
global scalars; $L$ length scale; $r_i = x_i / L$; $h_i$ token state;
$a_{is}$ routing weight; $z_s, m_s$ anchors; $\xi_{is}$ relational
invariants; $Z_s$ slice state; $\pi_{is}$ read-back weight; $\hat u_i$
radial basis vector. Definitions use $\equiv$.

## Color and mark semantics (from `palette.py`)

| Concept | Color | Marker / line | Notes |
|---|---|---|---|
| ISLA reference | `BLUE` | filled square, solid | |
| ISLA variants | `BLUE_LIGHT` | open diamond (similarity gauge, dashed); open square dotted is reserved for the legacy centered variant inside diagnostic notes | same hue, lighter |
| ISLA ablations | `BLUE_PALE`, hatched | | never adopted |
| GeoTransolver | `RED` | filled circle, solid | also GeoTransolver-volume |
| Transolver | `AMBER` | filled triangle, dashed | |
| baseline research variants | `RED_LIGHT` / `AMBER_LIGHT` | open marker, dotted | |
| slices, anchors | `VIOLET` | | concept figures only |
| queries, interior points | `AQUA` | | concept figures only |
| contract satisfied | `GREEN` | | |
| context (grids, reference lines, non-data) | grays | | never encodes a data distinction |
| signed fields | `DIVERGING` (blue–paper–red) | | unsigned fields: viridis |

In Mermaid diagrams: data boxes `fill:#f4f3ee,stroke:#898781`; fixed
geometric maps `fill:#ffffff,stroke:#2a78d6,stroke-dasharray:4 3`; learned
maps `fill:#e6f0fb,stroke:#2a78d6`; contracts `stroke:#008300`.

Use `STYLE[key]` / `series_kwargs(key)` / `bar_kwargs(key)` for every data
series so labels and colors never drift. Prefer direct labels (`label_line`)
to legends; when a legend is needed, its text is the canonical name.

## Figures and tables

- Every axis: quantity and unit in brackets, `[–]` for dimensionless.
- Bars start at zero; log axes only where the range spans a decade, with
  1–2–5 ticks written out.
- Multi-panel figures stack vertically and share the x-axis; no dual axes.
- Seeds are dots on bars or bands on lines; the seed mean is the mark.
- Table captions name the artifact; the `{#tbl-…}` label is on the caption line.
- A "How to read this" clause appears in every results figure caption.

## What the chapters do not do

- **Cut, never retract.** A claim now known wrong does not appear at all:
  not as "we previously believed", "retired", "reframed" or "superseded". A
  reader must never watch the book argue with itself. History is the
  notebook's job.
- **No dates, no chronology, no project slang.** No "as of 2026-09-05", no
  "wave", "round", "campaign", "addendum", "critic review", "before/after
  the fix". No node codes (A35b, W2-C, EQ-PRICE) or version tags in prose;
  say what the experiment measures ("the single-factor ablation at 35
  training cases"). Exceptions: the status chapter's tech tree and ledger,
  and the notebook.
- **No run identifiers** (`rf_dr_mt2_meas_seed42`) in prose; they appear
  in code cells and in captions' artifact paths only.
- **No numbers from memory.** Every number is computed from a file in
  `results/` (or `research/`) or quoted from `CLAIMS.md` with its source.
  Define the quantity before the number: what is measured, on which split,
  under what protocol, and which way a ratio points.
- **Resource-matched comparisons only.** A cross-architecture comparison
  states parameters, tokens per case and memory of both sides, or says in
  the same sentence where they differ and whether the difference could buy
  accuracy.
- **No jitter or dodge on categorical axes**; seeds and architectures are
  distinguished by marker, fill or line style.
- **No hedged verdicts** and no result made to look better than it is. If a
  comparison is unfair, fix it or drop it; do not caption around it.
- **Scope and limits sit beside the claim**, not only in the limits chapter.
- **Chapter rewrites do not edit `18-notebook.qmd` or `results/`.**
