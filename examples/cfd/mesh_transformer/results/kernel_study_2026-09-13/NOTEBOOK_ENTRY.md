## 2026-09-13 -- Kernel study: where ISLA's training step goes, and the options that shorten it without changing the model

**Setting.** ISLA (Invariant Slice Attention) is a soft-slice transformer for surface
fields: 10,000 surface points are routed to 256 slices by a softmax over points, slice
states are quadrature-weighted means, and each point reads the slices back weighted by
a softmax over slices. What makes it different from Transolver-style slicing is the
geometry region in every layer: for every point-slice pair it evaluates six relational
invariants of the point and the slice anchor (distance, its log, and four dot products
of unit vectors), feeds them to a 6->1 Linear that biases the routing, and pools them
back to the point with the routing weights. That is a (batch x points x slices x 6)
tensor per layer -- 15 M numbers at 10k tokens, 61 M at 40k -- and the book's cost
section says the step is 3.7x slower than GeoTransolver's (`matched_memory_isla_gt_2026-09-08.json`).

**Question.** Where does the time and memory of one training step (forward, backward,
AdamW) actually go, and which implementation options -- fusion, recompute, compile
modes, a custom kernel -- shorten it while computing exactly the same function?
Options were admitted only if they reproduce the eager reference forward to 1e-6
relative and the parameter gradients to 1e-5 in fp32 and agree in float64 (same
arithmetic, different order).

**Instruments.** One NVIDIA GB300 on AGA (driver 580.167.08, torch 2.12.0+cu130), the
GPU the recipe trains on and the book's number was measured on; the reference
configuration (`isla_surface_reference.yaml`, hidden 192, 12 layers, 256 slices,
`geo_checkpoint: true`) on synthetic inputs; step time = median of 10 cuda-synchronized
steps after 3 warm-ups; peak = `max_memory_allocated`. An analytic roofline (FLOPs and
bytes per stage against measured peaks: 1925 TFLOP/s bf16, 7135 GB/s), torch.profiler
traces with every kernel attributed to its launching operator and phase, allocation
snapshots at the peak, and a numerics gate against the eager reference in fp32 and
float64. Artifacts in `results/kernel_study_2026-09-13/` (jobs 736338, 736352, 738400);
tables in `TABLES.md`, report in `REPORT.md`. An RTX 4090 Laptop was used for the first
pass; its numbers are secondary and labelled.

**What was seen.**

1. *The 3.7x reproduces and is a kernel, not the architecture.* Eager, with the
   middle-dimension point softmax that was replaced on 2026-09-10, ISLA takes 244.7 ms
   against GeoTransolver's 67.8 ms at 10k tokens (3.61x). With the row-parallel softmax
   the same eager step is 86.7 ms (1.28x), or 67.0 ms (0.99x) without `geo_checkpoint`.
   The recipe compiles both models (`compile: true`): compiled, ISLA is 47.3 ms with
   `geo_checkpoint` (1.05x GeoTransolver's 45.2 ms) or 39.0 ms without (0.86x), at 1.30
   or 1.94 GiB peak against GeoTransolver's 2.70. At 40k tokens the original-softmax
   eager step is 1284 ms; the reference eager 135 ms; compiled 108-119 ms; GeoTransolver
   compiled 53 ms.

2. *The step is far from any roofline, for two different reasons at two sizes.* The
   analytic step at 10k tokens is 404 GFLOP and 62 GB of traffic as eager PyTorch
   executes it (8.8 ms at the GB300's bandwidth), or 15 GB if every stage were fused
   (2.1 ms). The eager reference runs 33.8 ms of kernels inside a 91 ms step -- 3,224
   kernel launches, so 63% of the wall time is host-side launch and autograd overhead
   on the Grace CPU. At 40k the kernels fill the step (130.6 of 135 ms) and the
   invariants stage alone takes 71 ms against a 21 ms roofline for its eager traffic:
   it carries 61% of all bytes, and its fused-traffic roofline is 11x smaller. Every
   stage but the point MLP is memory-bound; the MLP reaches 10.7% of the bf16 FLOP peak.

3. *`torch.compile` fuses the elementwise chain but keeps the tensor, and its default
   backward configuration is poor.* Inductor still materializes the (B,N,S,6)
   invariants twice per layer (fp32, and a bf16 copy for the GEMM); one generated
   reduction kernel for the geometry backward is 13.4 of the 20.0 ms of backward
   kernels at 10k and 56 of 82 ms at 40k. `mode="max-autotune-no-cudagraphs"` fixes
   the configuration: 10k 39.0 -> 33.3 ms; 40k 108 -> 41 ms; 40k batch 2 177 -> 67 ms.
   `mode="reduce-overhead"` (CUDA graphs) removes the launch-bound regime: 26.0 ms at
   10k, 0.58x of default-compiled GeoTransolver, at the cost of static shapes and
   activation memory that `max_memory_allocated` cannot see (2.02 GiB reserved).

4. *A fused Triton kernel for the geometry region is exact and removes the tensor.*
   `geo_kernel="fused"` (opt-in; `physicsnemo/experimental/nn/isla/geo_kernel.py`)
   evaluates invariants, routing bias, slice softmax and pooling on register tiles of
   8 points x 256 slices and never writes a (B,N,S,.) intermediate; the backward
   recomputes the tile. Region-level, one layer at 10k tokens: eager 0.70 / 1.20 ms
   (forward / backward), Inductor 0.43 / 2.09, fused 0.23 / 0.55; at 40k: 2.87 / 3.73,
   0.73 / 11.67, 0.20 / 0.77; region memory 0.265 / 0.125 / 0.034 GiB. Whole step,
   compiled: 45.3 ms at 10k (kernel time 8.8 ms in a 41.5 ms wall, so launch-bound; no
   wall-time gain over 39.0) and 52.1 ms at 40k (vs 108), at 1.27 / 4.50 GiB (vs 1.94 /
   7.13); at 40k batch 2, 54.7 ms and 8.82 GiB against GeoTransolver's 51.6 ms and
   19.85 GiB. Numerics: output 5.4e-7 relative to the eager reference (the reference's
   own fp32-vs-fp64 floor is 6.2e-7), gradient 1.1e-7, float64 agreement 1e-15; 12 new
   tests, 99 existing ISLA tests pass; compiles with zero graph breaks.

5. *Memory map.* At the eager peak (1.35 GiB at 10k) the largest live blocks are the
   MLP activations (0.48 GiB), the transposed copies the fast point softmax makes
   (0.22), and the assign logits (0.19); with `geo_checkpoint` off under compile the
   kept (B,N,S,6) invariants and (B,N,S,3) relative vectors are 0.71 of 1.76 GiB
   (2.63 of 6.96 at 40k). `geo_checkpoint` under compile buys 0.64 GiB for +21% time at
   10k; the fused kernel makes it moot.

**Why it matters.** The cost section's premise -- that ISLA's geometry makes it
several times more expensive than GeoTransolver -- does not hold on the path the
recipe runs: at 10k tokens the two are within 5-14% of each other in step time, and
ISLA uses half the memory. The remaining cost is not arithmetic (the step is 400
GFLOP on a 2 PFLOP/s part) but memory traffic of one stage and host launch overhead,
both of which are implementation choices with exact-arithmetic remedies. At the token
counts where the traffic dominates (40k and above) the fused kernel and Inductor's
autotuned mode each halve the step; combined they should do better, and none of them
changes a single output beyond fp32 roundoff.

**What is next.** (a) Measure fused + `max-autotune` and fused + `reduce-overhead`
(the profiles predict the gains add: 8.8 ms of kernels in a 41.5 ms wall at 10k).
(b) Expose `compile_mode` in the recipe and measure GeoTransolver under the same modes
so the anchor column is like-for-like. (c) One training run with `geo_kernel="fused"`
for neutrality (its bf16 backward accumulates in fp32 where eager's rounds to bf16 --
more precise, not identical) before any default changes. (d) Correct the book's cost
section from the reconciliation table in `REPORT.md`, on the GB300 numbers.
