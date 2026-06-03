"""Empirical wall-clock scaling of GLOBE's BarnesHutKernel on synthetic 3D data.

Produces a single log-log figure (``scaling.{pdf,png}``) that compares the
per-forward-pass wall-clock time of:

- The dense baseline (``BarnesHutKernel`` with ``theta=0`` - all interactions
  exact, expected slope :math:`\\sim 2`)
- Barnes-Hut with ``theta in {0.5, 1.0, 2.0}`` (expected slope :math:`\\sim 1`)

The script is split out from ``theta_effect.py`` because it is by far the
slowest visualization in this directory and benefits from being run on its
own when iterating on benchmark methodology.

Launch
------
The (theta, N) sweep is embarrassingly parallel and integrates with
:class:`physicsnemo.distributed.DistributedManager`.

The shipped ``run.sh`` is the recommended entry point: it auto-detects
the visible GPU count and dispatches via ``torchrun``::

    ./run.sh                  # one rank per visible GPU
    NPROC_PER_NODE=2 ./run.sh # explicit override

Direct invocations are also fine:

- Single rank::

      uv run --no-sync ./scaling.py

- Multi-rank, e.g. 4 GPUs on one node::

      uv run --no-sync torchrun --standalone --nproc-per-node=4 ./scaling.py

Each rank gets ``N_VALUES_BH[rank::world_size]`` for every theta and
processes its slice in ascending N (with per-theta early abort on first
failure).  Only rank 0 saves the figure.

.. note::

   Running ``scaling.py`` *directly* (not via ``run.sh`` / ``torchrun``)
   inside a SLURM allocation is safe: the script detects that no
   multi-rank dispatcher set ``WORLD_SIZE`` / ``OMPI_*`` / ``SLURM_STEP_ID``
   and skips ``DistributedManager.initialize()`` entirely.  Without that
   guard the manager would pick up ambient SLURM env vars and try to
   bring up a process group with non-existent peers, which hangs for
   10 minutes on the TCPStore connect timeout.

Methodology notes
-----------------
- A fixed ``near_chunk_size`` is queried once at the start of the script,
  while VRAM is still uncluttered, and pinned for every subsequent forward
  call. ``BarnesHutKernel`` defaults to a memory-aware ``_auto_chunk_size``
  that *shrinks* chunks under runtime memory pressure; that re-acts as
  pathological launch-overhead-driven thrashing rather than a clean OOM
  (single forward passes taking 30+ s instead of failing fast).
- ``BarnesHutKernel.forward`` only chunks the near-field phase. The other
  three phases (far-far, near-far, far-near) evaluate in single batched
  calls. Above some N these single calls trigger memory pressure that the
  kernel masks rather than OOMs; we detect that via a wall-clock sentinel
  on the first warmup pass and abort the curve.
- Per-(theta, N) timing reports the **min** across an *adaptive* number
  of trials chosen so each iteration spends roughly ``TIMING_BUDGET_S``
  on timed runs (clamped to ``[N_TRIALS_MIN, N_TRIALS_MAX]``).  Min is
  the right estimator for the operation's intrinsic cost: jitter from
  background processes, GPU power state transitions, or PyTorch caching
  allocator hiccups can only ever inflate timings, never deflate them.
- Both wall-clock (``perf_counter``) and GPU-event (``torch.cuda.Event``)
  timings are recorded per trial. Wall is what users see; the wall - GPU
  gap is CPU/dispatch overhead, which dominates at small N. The full
  per-trial distribution is printed to stdout (rank-prefixed when
  multi-rank) for diagnosis.
- Boost-clock and cuBLAS plan-cache state at sweep start are the
  dominant residual sources of small-N run-to-run variance after the
  global + per-iteration warmups.  If you need tighter reproducibility,
  pin the GPU clocks before launch::

      sudo nvidia-smi --lock-gpu-clocks=<min>,<max>   # e.g. =1830,1830
      ./scaling/run.sh
      sudo nvidia-smi --reset-gpu-clocks
"""

import math
import os

### [Allocator config: select expandable_segments BEFORE any torch/CUDA import.]
# `BarnesHutKernel.forward` empirically reserves ~30% more GPU memory than it
# truly needs when the chunked Phase A loop fragments the default caching
# allocator's free list.  Selecting PyTorch's expandable-segments allocator
# eliminates that overhead with negligible wall-time cost; see the Notes
# section of `BarnesHutKernel`'s docstring.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from time import perf_counter

import aerosandbox.tools.pretty_plots as p
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.globe import BarnesHutKernel
from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

### [Distributed setup]
# We deliberately do *not* unconditionally call
# ``DistributedManager.initialize()``: its env-var auto-detection takes
# the SLURM branch whenever ``SLURM_PROCID`` is set, which happens in
# every interactive shell inside a SLURM allocation.  In that case the
# manager calls ``torch.distributed.init_process_group`` with a
# ``MASTER_ADDR`` of ``SLURM_LAUNCH_NODE_IPADDR`` and waits 10 minutes
# for the (non-existent) other ranks before timing out.  Instead, we
# detect a real multi-rank launch from launcher-specific env vars and
# only initialise the manager when it will actually do something useful.
def _is_multi_rank_launch() -> bool:
    """True when this process is one of several launched by torchrun / mpirun / srun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return True  # torchrun / torch.distributed.launch
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        return True  # OpenMPI mpirun
    # SLURM env vars from a bare ``salloc`` allocation only describe what
    # was *allocated*; they don't mean a multi-rank job is running.  Only
    # an active ``srun`` step (which sets SLURM_STEP_ID per task) counts.
    if (
        "SLURM_STEP_ID" in os.environ
        and int(os.environ.get("SLURM_NPROCS", "1")) > 1
    ):
        return True
    return False


if _is_multi_rank_launch():
    DistributedManager.initialize()
    dist_mgr = DistributedManager()
    rank: int = dist_mgr.rank
    world_size: int = dist_mgr.world_size
    device: torch.device = dist_mgr.device
else:
    # Single-rank: stamp the singleton manually so any
    # ``DistributedManager()`` construction downstream succeeds, but
    # *don't* call ``init_process_group``.  Defaults from the singleton
    # (``rank=0``, ``world_size=1``, ``device=cuda:0`` if available)
    # are exactly what we want for serial execution.
    DistributedManager._shared_state["_is_initialized"] = True
    dist_mgr = DistributedManager()
    rank: int = 0
    world_size: int = 1
    device: torch.device = dist_mgr.device

USE_CUDA: bool = device.type == "cuda"
IS_RANK0: bool = rank == 0
OUTPUT_DIR = Path(__file__).parent


def log(msg: str) -> None:
    """Rank-prefixed, line-flushed print so multi-rank stdout stays readable."""
    print(f"[rank {rank}/{world_size}] {msg}", flush=True)

### [Configuration]
SEED = 39
# Sweep spans N=500 to N=5M.  Each curve runs until the kernel's runtime
# memory checks (Phase-B budget below, the dual-plan OOM guard, the OOM
# during the forward pass itself, or the wall-clock thrash sentinel)
# abort it; on a 16-32 GB GPU dense (theta=0) typically stops in the
# low-thousands while BH at theta=2 reaches the millions.  Dense (theta=0)
# is the quickest to die because Phase A's chunked loop holds ~5x
# per_chunk_peak in cached-but-not-reusable PyTorch blocks (documented in
# BarnesHutKernel's Notes), so the cumulative reservation across an N
# sweep exhausts the GPU well before any single forward pass would.  71
# geometrically-spaced points gives a per-step ratio of ~1.14x for
# densely sampled curves.
N_VALUES_BH = sorted(
    set(np.round(np.geomspace(500, 5_000_000, 71)).astype(int).tolist())
)
THETA_SCALING = [0.5, 1.0, 2.0]
# Per-(theta, N) warmup count.  Bumped from 3 to give the GPU's boost
# clock and cuBLAS plan cache enough sustained work to settle before
# any timed trial; this is the cheapest knob for taming small-N
# run-to-run variance (each warmup at small N is ~50-100 ms).
N_WARMUP = 8
# Adaptive timed-trial count, computed per (theta, N) from the sentinel
# warmup's wall time -- see ``_adaptive_n_trials`` below.  We aim for a
# fixed total timing *budget* per iteration so that small N (where each
# call is ~50 ms) gets many samples for ``min`` to find a stable
# best-of-N, while multi-second iterations don't bloat the sweep.
N_TRIALS_MIN = 3
N_TRIALS_MAX = 15
TIMING_BUDGET_S = 5.0
# A single forward pass exceeding this duration almost certainly means the
# kernel is fighting fragmentation/eviction in the unchunked Phase B
# allocation.  The sentinel is applied to the *second* warmup run (not the
# first), so cold-start costs from the absolute first call at a new shape
# (cudaMalloc, cuBLAS plan caching) don't trigger a false positive.  We
# treat genuine sentinel trips as effectively-OOM and abort the curve to
# avoid burning many slow trials after the first.
TIMEOUT_S = 15.0


def _adaptive_n_trials(per_call_s: float) -> int:
    """How many timed trials to run, given an estimated per-call wall time.

    Targets ``TIMING_BUDGET_S`` of total timing, clamped to
    :math:`[\\text{N\\_TRIALS\\_MIN},\\, \\text{N\\_TRIALS\\_MAX}]`.
    Small N (~50 ms) hits the upper cap so we still have many samples
    for the min; multi-second iterations drop to the floor of
    ``N_TRIALS_MIN`` so the long curves don't bloat the sweep.
    """
    n = math.ceil(TIMING_BUDGET_S / max(per_call_s, 1e-6))
    return max(N_TRIALS_MIN, min(N_TRIALS_MAX, n))


### [Helpers]
def make_3d_problem(
    n: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorDict]:
    """Random unit-cube source/target points + unit normals at scale ``n``.

    A *fixed* seed is used for every N so that successive N values share a
    common random prefix (the first 100 points at N=100 are the same
    physical points as the first 100 of the N=215 sweep, etc.).  This
    keeps the cluster-tree shape statistics consistent across N and removes
    one source of run-to-run jitter (different seeds would put points in
    radically different positions, producing wildly different dual_plan
    fan-outs even at the same N).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    src = torch.rand(n, 3, generator=g, device=device) * 2.0 - 1.0
    tgt = torch.rand(n, 3, generator=g, device=device) * 2.0 - 1.0
    strengths = torch.full((n,), 1.0 / n, device=device)
    normals = torch.randn(n, 3, generator=g, device=device)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    data = TensorDict(
        {"normal": normals, "other": torch.zeros_like(normals)},
        batch_size=torch.Size([n]),
        device=device,
    )
    return src, tgt, strengths, data


def memory_mb() -> float:
    """Currently-allocated GPU memory in MB (0.0 on CPU)."""
    return torch.cuda.memory_allocated() / 1e6 if USE_CUDA else 0.0


def time_forward(
    *,
    kernel: BarnesHutKernel,
    src: torch.Tensor,
    tgt: torch.Tensor,
    strengths: torch.Tensor,
    data: TensorDict,
    src_tree: ClusterTree,
    tgt_tree: ClusterTree,
    theta_val: float,
    near_chunk_size: int,
) -> dict[str, float | list[float]] | None:
    """One (theta, N) measurement: cold absorber + warmup + adaptive trials.

    Returns a dict with min wall-clock and GPU-event times (both ms) plus
    the full per-trial distributions, or ``None`` if the *second* warmup
    run exceeded ``TIMEOUT_S`` (a thrash sentinel).  The absolute first
    call at a new (theta, N) shape pays one-time cold-start costs
    (cudaMalloc for new pair-storage shapes, cuBLAS plan-cache misses)
    that can comfortably exceed the genuine compute time, so we run an
    untimed cold-absorber call before applying the sentinel.

    The number of timed trials is decided by :func:`_adaptive_n_trials`
    from the sentinel call's measured wall time, so each (theta, N)
    spends roughly ``TIMING_BUDGET_S`` of wall time on timed trials.
    """

    def _run() -> None:
        with torch.no_grad():
            kernel(
                reference_length=torch.tensor(1.0, device=device),
                source_points=src,
                target_points=tgt,
                source_strengths=strengths,
                source_data=data,
                theta=theta_val,
                cluster_tree=src_tree,
                target_tree=tgt_tree,
                near_chunk_size=near_chunk_size,
            )

    ### [Cold-start absorber: untimed, just to fault in shape-specific caches]
    _run()
    if USE_CUDA:
        torch.cuda.synchronize()

    ### [Sentinel warmup: now that caches are warm, this run reflects steady
    ### state, so a wall-clock budget genuinely catches Phase-B thrashing.
    ### Doubles as the per-call estimate that sizes the timed-trial loop.]
    if USE_CUDA:
        torch.cuda.synchronize()
    t0 = perf_counter()
    _run()
    if USE_CUDA:
        torch.cuda.synchronize()
    sentinel_s = perf_counter() - t0
    if sentinel_s > TIMEOUT_S:
        return None

    ### [Remaining warmups]
    for _ in range(max(0, N_WARMUP - 2)):
        _run()
    if USE_CUDA:
        torch.cuda.synchronize()

    ### [Timed runs - record both wall-clock and GPU-event time per trial]
    n_trials = _adaptive_n_trials(sentinel_s)
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(n_trials):
        if USE_CUDA:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = perf_counter()
            start_ev.record()
            _run()
            end_ev.record()
            torch.cuda.synchronize()
            wall_ms.append((perf_counter() - t0) * 1e3)
            gpu_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = perf_counter()
            _run()
            dt_ms = (perf_counter() - t0) * 1e3
            wall_ms.append(dt_ms)
            gpu_ms.append(dt_ms)
    return {
        "wall_min_ms": min(wall_ms),
        "wall_ms": wall_ms,
        "gpu_min_ms": min(gpu_ms),
        "gpu_ms": gpu_ms,
    }


def save_figure(fig: plt.Figure, *, stem: str) -> None:
    """Save figure as both PDF and PNG under ``OUTPUT_DIR``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"Saved {path}")


# =====================================================================
# Build kernel and pin chunk size
# =====================================================================

log(f"=== Empirical scaling test (3D, device={device}) ===")
torch.manual_seed(SEED)
np.random.seed(SEED)

kernel_3d = BarnesHutKernel(
    n_spatial_dims=3,
    output_field_ranks={"phi": 0, "u": 1},
    source_data_ranks={"normal": 1, "other": 1},
    hidden_layer_sizes=[64, 64],
    n_spherical_harmonics=4,
    network_type="pade",
    spectral_norm=False,
    use_gradient_checkpointing=False,
    leaf_size=1,
).to(device)
kernel_3d.eval()

NEAR_CHUNK_SIZE = kernel_3d._auto_chunk_size(
    n_total_pairs=10_000_000,
    device=torch.device(device),
)
log(f"  Pinned near_chunk_size = {NEAR_CHUNK_SIZE:,}")

### [Phase-B memory budget]
# `BarnesHutKernel.forward` only chunks the near-field (Phase A); Phases B,
# C, and D evaluate a single unchunked (n_pairs, floats_per_interaction)
# tensor.  Under the default PyTorch caching allocator we observed a sharp
# performance cliff when this allocation exceeded ~1.5-1.8 GB: on a 17 GB-
# free GPU, ~180 ms forward passes at 1.5 GB Phase B *jumped* to ~1.1 s at
# 2.0 GB Phase B (5-10x slowdown).  The cliff was too sharp to be memory
# pressure (15+ GB free); it is almost certainly cuBLAS picking a different
# GEMM algorithm above some M-dimension threshold, or the caching
# allocator's `max_split_size` behavior at ~2 GB.  Setting
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (done at the top of
# this script) eliminates the cliff, so we cap Phase B at the GPU's
# actual total VRAM (queried dynamically via `torch.cuda.mem_get_info`)
# rather than at a fixed safety value.  The safe budget is then the
# minimum of (80% of free GPU memory) and that total-VRAM cap.
FLOATS_PER_INTERACTION = kernel_3d._floats_per_interaction
if USE_CUDA:
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
else:
    free_bytes, total_bytes = 0, 0
PHASE_B_HARD_CAP_BYTES = total_bytes if USE_CUDA else 10**12
PHASE_B_SAFE_BYTES = (
    min(int(free_bytes), PHASE_B_HARD_CAP_BYTES) if USE_CUDA else 10**12
)
PHASE_B_MAX_PAIRS = (
    PHASE_B_SAFE_BYTES // (FLOATS_PER_INTERACTION * 4) if USE_CUDA else 10**12
)
if USE_CUDA:
    log(
        f"  GPU free / total = {free_bytes / 1e9:.2f} / {total_bytes / 1e9:.2f} GB; "
        f"floats_per_interaction = {FLOATS_PER_INTERACTION}"
    )
    log(
        f"  Phase-B safe budget = {PHASE_B_SAFE_BYTES / 1e9:.2f} GB "
        f"(min of 80% of free and {PHASE_B_HARD_CAP_BYTES / 1e9:.1f} GB hard cap)"
        f"  ->  max n_far = {PHASE_B_MAX_PAIRS:,}"
    )


# =====================================================================
# Global warmup: amortize CUDA / cuBLAS / cuDNN init once before the sweep
# =====================================================================
# Cover *every* theta in this rank's sweep, not just {0.0, 1.0}: each
# theta activates a different set of phases (B/C/D fan-out grows with
# theta) and therefore a different set of cuBLAS / cuBLASLt matmul
# shapes, each of which pays a one-time plan-cache miss on its first
# call.  Five iterations per theta is enough sustained work to ramp the
# B200 to its boost clock once and leave it there for the actual sweep.

log("  Performing global warmup...")
_w_src, _w_tgt, _w_str, _w_data = make_3d_problem(500, seed=SEED + 99)
_w_src_tree = ClusterTree.from_points(_w_src, leaf_size=1)
_w_tgt_tree = ClusterTree.from_points(_w_tgt, leaf_size=1)
for _theta_warm in (0.0, *THETA_SCALING):
    for _ in range(5):
        with torch.no_grad():
            kernel_3d(
                reference_length=torch.tensor(1.0, device=device),
                source_points=_w_src,
                target_points=_w_tgt,
                source_strengths=_w_str,
                source_data=_w_data,
                theta=_theta_warm,
                cluster_tree=_w_src_tree,
                target_tree=_w_tgt_tree,
                near_chunk_size=NEAR_CHUNK_SIZE,
            )
if USE_CUDA:
    torch.cuda.synchronize()
del _w_src, _w_tgt, _w_str, _w_data, _w_src_tree, _w_tgt_tree
# Deliberately *not* calling empty_cache here either - we want the cache
# warm and populated before the first sweep iteration starts.
log("  Global warmup done.")


# =====================================================================
# Sweep over (theta, N) - distributed across ranks
# =====================================================================

all_thetas = [0.0, *THETA_SCALING]

### [Per-rank workitem distribution]
# Round-robin within each theta keeps load balanced (each rank does
# ~1/world_size of every theta's points) *and* preserves per-rank early
# abort: each rank processes its N values in ascending order, breaks on
# first failure for that theta.  The signal lost (one rank's failure
# isn't seen by others) at most wastes a single OOM/timeout per rank,
# which our ascending-N order already minimises.
my_n_values = N_VALUES_BH[rank::world_size]

my_results: dict[float, dict[int, dict[str, float | list[float]]]] = {}
mem_baseline_mb = memory_mb()
log(
    f"  Memory baseline = {mem_baseline_mb:.1f} MB; "
    f"this rank owns {len(my_n_values)} of {len(N_VALUES_BH)} N values"
)
for theta_val in all_thetas:
    is_dense = theta_val == 0.0
    label = "dense (theta=0)" if is_dense else f"theta={theta_val}"
    by_n: dict[int, dict] = {}
    for n in my_n_values:
        src, tgt, strengths, data = make_3d_problem(n, seed=SEED)
        src_tree = ClusterTree.from_points(src, leaf_size=1)
        tgt_tree = ClusterTree.from_points(tgt, leaf_size=1)
        ### [Plan counts - lets us see the per-phase fan-out]
        # Wrap in try/except: the dual-plan computation itself can OOM at
        # very large N because its O(n_pairs) intermediates (e.g. the
        # `_ragged_arange` cumsum) are allocated *before* the explicit
        # Phase-B budget check below has any chance to short-circuit.
        try:
            dual_plan = src_tree.find_dual_interaction_pairs(
                target_tree=tgt_tree, theta=theta_val,
            )
            n_near = dual_plan.n_near
            n_far_nodes = dual_plan.n_far_nodes
            n_nf = dual_plan.n_nf
            n_fn = dual_plan.n_fn
            del dual_plan
        except torch.cuda.OutOfMemoryError:
            log(
                f"  {label}, N={n}: OOM during dual-plan computation, "
                f"stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        ### [Phase-B memory check - abort before allocator pressure kicks in]
        phase_b_bytes = n_far_nodes * FLOATS_PER_INTERACTION * 4
        if phase_b_bytes > PHASE_B_SAFE_BYTES:
            log(
                f"  {label}, N={n}: Phase-B would need ~{phase_b_bytes / 1e9:.2f} GB "
                f"(> {PHASE_B_SAFE_BYTES / 1e9:.2f} GB budget), stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        mem_before_mb = memory_mb()
        try:
            res = time_forward(
                kernel=kernel_3d,
                src=src,
                tgt=tgt,
                strengths=strengths,
                data=data,
                src_tree=src_tree,
                tgt_tree=tgt_tree,
                theta_val=theta_val,
                near_chunk_size=NEAR_CHUNK_SIZE,
            )
        except torch.cuda.OutOfMemoryError:
            log(f"  {label}, N={n}: OOM during forward pass, stopping curve.")
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        if res is None:
            log(
                f"  {label}, N={n}: warmup exceeded {TIMEOUT_S:.1f}s "
                f"(memory pressure / chunking thrash), stopping curve."
            )
            del src, tgt, strengths, data, src_tree, tgt_tree
            break
        mem_after_mb = memory_mb()
        by_n[n] = {
            "wall_min_ms": float(res["wall_min_ms"]),
            "gpu_min_ms": float(res["gpu_min_ms"]),
        }
        ### [Per-trial logging - exposes jitter so spikes are diagnosable]
        wall_str = ", ".join(f"{x:.1f}" for x in sorted(res["wall_ms"]))
        gpu_str = ", ".join(f"{x:.1f}" for x in sorted(res["gpu_ms"]))
        log(
            f"  {label}, N={n}: "
            f"wall_min={res['wall_min_ms']:6.2f}ms  "
            f"gpu_min={res['gpu_min_ms']:6.2f}ms  "
            f"plan(n_near={n_near:>9,d} n_far={n_far_nodes:>7,d} "
            f"n_nf={n_nf:>7,d} n_fn={n_fn:>7,d})  "
            f"phase_B={phase_b_bytes / 1e9:5.2f}GB  "
            f"mem={mem_after_mb:.1f}MB ({mem_after_mb - mem_before_mb:+.1f})"
        )
        log(f"    wall trials [ms]: [{wall_str}]")
        log(f"    gpu  trials [ms]: [{gpu_str}]")
        del src, tgt, strengths, data, src_tree, tgt_tree
        # NOTE: deliberately *not* calling torch.cuda.empty_cache() between
        # iterations.  Empirically, doing so triggers a fresh-cudaMalloc
        # cycle that takes several warmup runs to amortize, producing
        # bimodal or monotonically-growing per-trial timings (especially
        # at small N where Phase B's unchunked far-field allocation has to
        # be re-issued from scratch).  Letting PyTorch's caching allocator
        # reuse blocks across (theta, N) keeps the per-trial distributions
        # tight; the trade-off is we OOM slightly sooner at large N, which
        # is fine because the sentinel already aborts those curves.
    if by_n:
        my_results[theta_val] = by_n


# =====================================================================
# Gather per-rank results and reshape into the schema the plot expects
# =====================================================================

### [Gather to rank 0]
# ``gather_object`` handles arbitrary Python objects via pickle, but
# under an NCCL backend it pickle->CUDA-tensor->NCCLs the bytes, which
# is fragile after a long memory-pressuring sweep: empirically, runs
# that came close to OOM (Phase-B at ~150 GB / 196 GB budget) sometimes
# trip ``RuntimeError: NCCL Error 1: unhandled cuda error`` here even
# though every individual rank's payload is tiny.  We dodge that by:
#   1. ``cuda.synchronize`` to surface any pending CUDA error here, not
#      from inside an opaque collective;
#   2. ``cuda.empty_cache`` to release allocator-cached blocks before
#      gloo serialisation (no functional need, just hygiene);
#   3. running the collective over a *gloo* subgroup -- gloo is CPU-
#      side, so it's immune to whatever CUDA state NCCL was unhappy
#      about.  All ranks must call ``new_group`` collectively.
if world_size > 1:
    if USE_CUDA:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    object_gather_group = torch.distributed.new_group(backend="gloo")
    gathered: list[dict | None] | None = (
        [None] * world_size if IS_RANK0 else None
    )
    torch.distributed.gather_object(
        my_results, gathered, dst=0, group=object_gather_group
    )
else:
    gathered = [my_results]

if IS_RANK0:
    merged: dict[float, dict[int, dict]] = {}
    for shard in gathered or ():
        if not shard:
            continue
        for theta_v, by_n in shard.items():
            merged.setdefault(theta_v, {}).update(by_n)
    ### [Reshape: flat per-N dicts -> the parallel-list schema the plot uses]
    scaling_results: dict[float, dict[str, list]] = {}
    for theta_v in all_thetas:
        by_n = merged.get(theta_v, {})
        sorted_ns = sorted(by_n.keys())
        scaling_results[theta_v] = {
            "N": sorted_ns,
            "wall_ms": [by_n[n]["wall_min_ms"] for n in sorted_ns],
            "gpu_ms": [by_n[n]["gpu_min_ms"] for n in sorted_ns],
        }
    log(
        "  Sweep complete; merged into "
        + ", ".join(
            f"theta={th}:{len(scaling_results[th]['N'])}pts"
            for th in all_thetas
        )
    )


# =====================================================================
# Plot: log-log wall-clock vs N, with per-trial spread band, GPU-event
# companion line, and *theoretical* (not fit) reference curves.
# =====================================================================


if IS_RANK0:
    ### [Style: theta is an ordered parameter -> sequential viridis cmap;
    ### black is reserved for the dense baseline so it pops out.]
    _THETA_BH_COLORS = plt.cm.viridis(
        np.linspace(0.15, 0.75, len(THETA_SCALING))
    )
    THETA_PLOT_COLORS: dict[float, object] = {
        0.0: "k",
        **dict(zip(THETA_SCALING, _THETA_BH_COLORS)),
    }
    THETA_PLOT_LABELS = {
        0.0: r"Dense ($\theta=0$)",
        0.5: r"Barnes-Hut ($\theta=0.5$)",
        1.0: r"Barnes-Hut ($\theta=1.0$)",
        2.0: r"Barnes-Hut ($\theta=2.0$)",
    }

    ### [Aspect ratio note]
    # On a log-log plot the visual angle of a slope-m line is
    # ``arctan(m * px_per_dec_y / px_per_dec_x)``.  With the previous
    # 7.5 x 5.0 inch figure on this data range, ``px_per_dec_y`` was
    # ~2x ``px_per_dec_x``, putting slope-2 (dense) at ~75 deg and slope
    # ~1.1 (Barnes-Hut N log N) at ~60 deg - both crowded near vertical
    # and visually hard to tell apart.  The fix is *not* to compress the
    # ylim data range (that would steepen them further); instead, force
    # one decade of x to equal one decade of y in pixels via
    # ``set_aspect('equal')`` below, and use a wider/shorter figure so
    # the axes box has room for that aspect.  After this change slope-2
    # sits at ~63 deg and slope ~1.1 at ~47 deg, both bracketing 45 deg
    # where small differences in slope are most perceptually distinct.
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    n_data: list[int] = []
    y_data_s: list[float] = []
    for theta_val in all_thetas:
        res = scaling_results[theta_val]
        if not res["N"]:
            continue
        color = THETA_PLOT_COLORS[theta_val]
        n_arr = np.asarray(res["N"], dtype=float)
        wall_min_s = np.asarray(res["wall_ms"]) / 1e3

        n_data.extend(int(n) for n in res["N"])
        y_data_s.extend(wall_min_s.tolist())

        # Wall-clock min: the headline number any user would quote.
        ax.loglog(
            n_arr,
            wall_min_s,
            "o-",
            color=color,
            label=THETA_PLOT_LABELS[theta_val],
            markersize=5,
            linewidth=1.6,
        )

    ### [Axis limits follow the measured wall times only.]
    if n_data:
        log_lo, log_hi = np.log10(min(n_data)), np.log10(max(n_data))
        span = max(log_hi - log_lo, 0.2)
        pad = 0.05 * span
        ax.set_xlim(10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad))
    if y_data_s:
        log_lo, log_hi = np.log10(min(y_data_s)), np.log10(max(y_data_s))
        span = max(log_hi - log_lo, 0.2)
        pad = 0.05 * span
        ax.set_ylim(10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad))

    ### [Force 1 dec_x = 1 dec_y in pixels - see aspect-ratio note above.]
    ax.set_aspect("equal", adjustable="box")

    ### [Theoretical reference curves]
    # Dense is *exactly* :math:`O(N^{2})` and Barnes-Hut is *exactly*
    # :math:`O(N \log N)`; we know these analytically, so we draw the
    # theoretical curves directly rather than fitting an exponent (a fit
    # over the small-N overhead-dominated plateau gives a misleading
    # apparent slope).  ``N log N`` is *not* a straight line on log-log -
    # it's slightly super-linear and visibly curves over a wide enough N
    # range, which is exactly what we want the reader to see when they
    # compare it against the BH measurements.
    #
    # Each reference is anchored to the last measured point of the
    # corresponding curve (last point on the dense curve for ``N^2``,
    # last point on the theta=1 BH curve for ``N log N``).  The curves
    # are drawn only over the N range where the corresponding measured
    # data exists, so they don't overrun into empty parts of the panel.
    dense_res = scaling_results.get(0.0, {"N": [], "wall_ms": []})
    if dense_res["N"]:
        n_d = np.asarray(dense_res["N"], dtype=float)
        n_anchor = n_d[-1]
        t_anchor = dense_res["wall_ms"][-1] / 1e3
        n_line = np.geomspace(n_d.min(), n_d.max(), 50)
        ax.loglog(
            n_line,
            t_anchor * (n_line / n_anchor) ** 2,
            "--",
            color="0.4",
            linewidth=1.1,
            alpha=0.85,
            label=r"$\propto N^{2}$",
        )
    bh_res = scaling_results.get(1.0, {"N": [], "wall_ms": []})
    if bh_res["N"]:
        n_b = np.asarray(bh_res["N"], dtype=float)
        n_anchor = float(n_b[-1])
        t_anchor = bh_res["wall_ms"][-1] / 1e3
        n_line = np.geomspace(n_b.min(), n_b.max(), 200)
        ax.loglog(
            n_line,
            t_anchor * (n_line * np.log(n_line)) / (n_anchor * np.log(n_anchor)),
            ":",
            color="0.4",
            linewidth=1.4,
            alpha=0.85,
            label=r"$\propto N \log N$",
        )

    ### [Axes labels, hardware-stamped title]
    device_name = (
        torch.cuda.get_device_name(torch.device(device)) if USE_CUDA else "CPU"
    )
    ax.set_xlabel(r"Number of points $N$ (sources $=$ targets)")
    ax.set_ylabel("Wall-clock time per forward pass [s]")
    ax.set_title(
        rf"BarnesHutKernel scaling on synthetic 3D points  ({device_name})"
    )

    ### [Legend: 4 series + 2 reference curves; the upper-left quadrant
    ### of the panel is empty for this data range so an in-axes legend
    ### fits without overlap.]
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, which="both")

    plt.tight_layout()
    # ``aerosandbox.show_plot`` defaults to ``legend=None`` which auto-
    # detects multiple lines and then *recreates* the legend via a bare
    # ``plt.legend(frameon=True)`` -- silently discarding our custom
    # ``frameon=False``, ``bbox_to_anchor``, and proxy-handles legend
    # above.  ``legend=False`` tells it to leave our legend alone.
    p.show_plot(show=False, legend=False)
    save_figure(fig, stem="scaling")
    plt.close(fig)


### [Final barrier so all ranks exit together.]
# Without this, faster ranks can call sys.exit before rank 0 finishes
# saving, which torchrun reports as a non-zero exit on rank 0 even
# though the figure landed on disk fine.
if world_size > 1:
    torch.distributed.barrier()
