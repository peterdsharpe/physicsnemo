# FLARE and the kernel decoder: what low-rank routing can and cannot provide

**Thread**: task #43 (Peter's suggestion, 2026-07-07). Status: analysis
complete; measured cost split below; recommendation at the end.
Companion threads: #41 (hierarchical/FMM — owns the exact singular
members), #42/V4 (log-radial features — orthogonal, changes the member
MLP's *inputs*, not its pairwise cost).

## 1. What FLARE actually is

Read from `physicsnemo/experimental/nn/flare_attention.py` (used by
`experimental/models/flare/flare.py` to replace Transolver's physics
attention): a two-pass low-rank attention through `S = n_global_queries`
**learned global query slots** `G`:

```
z = SDPA(G, K, V)   # aggregate:  S slots gather from N tokens   O(N·S·D)
y = SDPA(K, G, z)   # distribute: N tokens read from S slots     O(N·S·D)
```

The effective token-token attention matrix is
`softmax(K·Gᵀ)·softmax(G·Kᵀ)` — rank ≤ S, row-stochastic both passes —
so all-to-all coupling remains at O(N·S·D) instead of O(N²·D). The
slots are data-independent parameters; everything learned is in the
projections.

## 2. Mapping onto the MeshTransformer — where FLARE is and isn't news

**The encoder is already a FLARE.** The separable signed-moment
attention aggregates measure-weighted key⊗value moments
(`M = Σⱼ ωⱼ kⱼ⊗vⱼ` — the "aggregate" pass, rank = scalar_rank +
vector_rank typed slots) and every cell reads them back (the
"distribute" pass). Ours differs in the principled directions: signed
and softmax-free (quadrature license — boundary-integral kernels change
sign), measure-weighted (the Mesh owns quadrature), and typed
(scalar/vector slots transform equivariantly, which FLARE's generic
slots cannot). FLARE applied to the encoder would *add* a softmax and
*remove* the type system for the same asymptotics we already have.
**Verdict: N/A for the encoder — we already run the stronger version.**

**The decode is where the money is.** The dense kernel decode is

```
u(x) = Σⱼ [ Σₘ Cₘ(opⱼ) · φₘ(x, yⱼ) ] · Vⱼ        O(Q·S) pairs
```

with two member families of very different character:

- **Exact singular members** (double/single layer): closed-form panel
  integrals (atan2 / log / asinh of pair geometry). Dense and nonseparable;
  the *hierarchical* thread (#41) owns their acceleration (far-field
  expansion is the classical answer — see GLOBE's Barnes–Hut with
  `expand_far_targets`, which is the same two-sided expansion idea).
- **Smooth MLP members**: `member_mlp(stacked pair invariants)` — the
  H4 verdict made these **essential** (6× velocity on AirFRANS),
  and they cost O(Q·S·hidden) with hidden=48 and 2-layer SiLU — per
  pair, far more FLOPs than the exact members. This is the
  FLARE-shaped target: replace the pairwise evaluation with per-point
  features whose products reconstruct the member values.

## 3. The contract analysis: separability vs exactness

A separable smooth member is `φ(x)ᵀψ(y)` with per-query and per-source
features only. Our house standard is *exact* contracts. What remains?

**Exactly separable AND exactly contract-preserving: polynomial
members.** Every pair invariant expands separably and exactly:

- `a = |r|² = |x|² + |y|² − 2·x·y` — the cross term `x·y = Σ_d x_d y_d`
  is rank-D separable; so any polynomial in `a` (degree p) is exactly
  separable at finite rank (multinomial expansion).
- `b = n_y·(x−y) = n_y·x − n_y·y` — rank D+1 with source features
  `(n_y, n_y·y)` and query features `(x, 1)`. Same for `v_c·r`.
- All contractions are joint O(D) invariants ⇒ **exact equivariance**;
  all displacement-borne ⇒ **exact translation invariance**; evaluated
  in the gauge frame ⇒ **exact similarity invariance**.

The catch is expressivity: the H4/banded evidence says the smooth
members justify their cost on *sharp near-wall radial structure*
(boundary layer at d/c ~ 5×10⁻⁴). Polynomials in `|r|²` are the wrong
basis for boundary layers — degree needed to localize at 10⁻⁴ scales
is absurd. **Exact separability is available but expressively wrong
for exactly the structure the members exist to carry.**

**Sharp radial + separable: only approximately equivariant.** The
classical separable route for stationary kernels is Fourier/random
features: `cos(w·r) = cos(w·x)cos(w·y) + sin(w·x)sin(w·y)` — exactly
translation-invariant, exactly separable, arbitrarily sharp in |r| with
enough frequencies. But a *finite learned* frequency set picks
directions: exact O(D) equivariance holds only if the frequency set is
group-closed AND the learned coefficients are tied by the group action
(a steerable basis — a substantial implementation), or in expectation over random
frequencies (not exact). **Exact equivariance + separability + sharp
radial expressivity do not coexist at finite rank with free
coefficients.** A licensed-approximation arm (measured equivariance
error, dense as oracle) is possible under the program's Barnes-Hut-style
precedent but is a *weaker* offer than the alternative below.

**The in-house resolution: typed-moment routing for the smooth
stream.** FLARE's real lesson is architectural, not spectral: route
smooth content through a rank bottleneck of *slots*. We already own the
exactly equivariant version of that construction — the typed moment
cross-attention (the moment decoder path). A **hybrid decoder** keeps
the exact singular members dense (they carry the singular physics and
the hierarchical thread will price their acceleration) and moves the
*smooth* content from the pairwise MLP to a typed-moment query stream:

```
u(x) = Σⱼ Σₘ∈exact Cₘ(opⱼ)·φₘ(x,yⱼ)·Vⱼ      dense, O(Q·S), unchanged
     + MomentCross(x; typed moments of the boundary)   O(Q + S)
```

- Exact contracts by construction (the moment operators are already
  typed, signed, measure-weighted, gauge-framed).
- Rank knob = moment channels — the direct analogue of FLARE's
  `n_global_queries`.
- Known measured limitation, stated up front: the moment stream carries
  the **proved m ≤ 2 angular ceiling** (README §6.2) — the ceiling that
  motivated the kernel decoder. The hybrid bets that the *smooth
  residual* (near-wall corrections on top of exact singular members)
  needs less angular order than the full field did. That is a
  falsifiable bet, not a guarantee.

## 4. Measured cost split (GPU, GB200, chunked decode)

Job `mt_decode_split` (slurm 4257002), `studies/decode_cost_split.py`,
median of 5, `query_chunk_size=2048`. Ratios are the durable quantity
(both paths are O(Q·S)).

| scale (S×Q) | singpair decode | members8 decode | MLP path (diff) | MLP share | peak mem singpair → members8 |
|---|--:|--:|--:|--:|--:|
| 1,000×4,096 (AirFRANS step) | 3.66 ms | 119.3 ms | 115.6 ms | **96.9%** | 0.64 → 1.99 GB |
| 10,000×10,000 (DrivAerML step) | 76.2 ms | 2,909.8 ms | 2,833.6 ms | **97.4%** | 6.4 → 19.6 GB |

**The smooth-member MLP is ~97% of the members-arm decode cost at both
scales** — the members arm decodes 33–38× slower than singpair, and
essentially all of it is the pairwise MLP, not the exact panel
integrals. Consequences: (i) the FLARE-shaped target owns the cost of
the current best-arm class almost entirely; a successful hybrid would improve
up to ~30× decode speedup and ~3× decode memory; (ii) for
*members-class* configurations, the hierarchical thread (#41)
accelerates only the residual ~3% until the smooth stream is
restructured — the two threads are complementary but THIS one is the
binding constraint today (for singpair configs at product scope the
priorities reverse: there #41 owns 100% of the 978 s forward).

Operational note (2026-07-07): the first attempt ran the 10k×10k point
on the dev machine with an unchunked decoder — ~10⁸ pairs × ~14
features materialized at once, >46 GB RSS, near-OOM. The script now
defaults to chunked decode + the small scale, and heavy benchmarks run
on cluster nodes only (see the hpc-resources conventions).

## 5. Recommendation and pre-registered experiment

1. **Do not FLARE the encoder** — it is already the stronger low-rank
   router (typed, signed, quadrature-licensed).
2. **Do not build separable smooth members inside the kernel decoder**:
   the exactly-contract-preserving version (polynomials) is expressively
   wrong for the near-wall structure the members exist to carry, and the
   expressive version (learned Fourier features) breaks exact
   equivariance. Neither meets the house standard; a prototype was
   NOT built.
3. **The supported direction is the hybrid decoder** (exact members
   dense + typed-moment smooth stream), a model-level change reusing
   existing implementation. Pre-registered experiment (run only after the
   V1–V5 sweep verdicts fix the smooth-member requirements):
   `mt_nl_hybrid` vs `mt_nl_members` on AirFRANS scarce, 3→5 seeds,
   protocol-v1. *Acceptance: ≥80% of the members arm's velocity gain
   over singpair at the measured decode-cost reduction; falsifier: the
   angular ceiling bites — hybrid ≪ members — which prices the pairwise
   MLP as irreplaceable and hands the speed problem entirely to
   thread #41.*
4. **Cost reality check**: whether any of this matters is set by the
   measured split above — if the exact members dominate decode cost,
   FLARE-shaped work is premature optimization and #41 is the only
   speed thread that pays.
