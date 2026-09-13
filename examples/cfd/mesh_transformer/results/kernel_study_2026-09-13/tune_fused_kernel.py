"""Launch-configuration sweep for the fused geometry region (geo_kernel.py): points per
program x warps, forward and backward kernels timed separately on the reference
shapes (256 slices; 10k and 40k tokens; batch 1), fp32 geometry with bf16 pre-logits
(the autocast case) and pure fp32. Also the eager and torch.compile'd region for the
same call, so the kernel's own speedup is on record separately from the whole-step
numbers of bench_pareto.py.

Usage: python tune_fused_kernel.py <out.json>
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_common as bc  # noqa: E402
import physicsnemo.experimental.nn.isla.geo_kernel as gk  # noqa: E402
import physicsnemo.experimental.nn.isla.model as im  # noqa: E402
from physicsnemo.experimental.nn.isla.geo_kernel import fused_geo_region  # noqa: E402

S = 256


def inputs(n, autocast, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = (torch.randn(1, n, 3, generator=g) * 0.03).cuda().requires_grad_()
    nh = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1).cuda().requires_grad_()
    d = torch.nn.functional.normalize(torch.randn(1, 1, 3, generator=g), dim=-1).cuda().expand(1, n, 3)
    z = (torch.randn(1, S, 3, generator=g) * 0.03).cuda().requires_grad_()
    ms = torch.nn.functional.normalize(torch.randn(1, S, 3, generator=g), dim=-1).cuda().requires_grad_()
    logits = torch.randn(1, n, S, generator=g).cuda()
    logits = (logits.bfloat16() if autocast else logits).requires_grad_()
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    return lin, logits, r, nh, d, z, ms


def timeit(fn, n_warm=3, n_meas=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_meas):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def fwd_bwd_times(region, args, autocast, extra):
    lin, logits, r, nh, d, z, ms = args

    def fwd():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return region(lin, logits, r, nh, d, z, ms, 1e-12, *extra)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        outs = region(lin, logits, r, nh, d, z, ms, 1e-12, *extra)
    gouts = [torch.randn_like(o) for o in outs]

    def bwd():
        torch.autograd.backward(outs, gouts, retain_graph=True)

    t_f = timeit(fwd)
    t_b = timeit(bwd)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        o2 = region(lin, logits, r, nh, d, z, ms, 1e-12, *extra)
    torch.autograd.backward(o2, gouts)
    peak = (torch.cuda.max_memory_allocated() - base) / bc.GIB
    return dict(fwd_ms=t_f, bwd_ms=t_b, fwd_bwd_incremental_peak_gib=peak)


def main():
    out = sys.argv[1]
    torch.backends.cuda.matmul.allow_tf32 = True
    res = dict(env=bc.env_info(), load=bc.load_snapshot(), slices=S, sweep=[], baselines=[])
    compiled_region = torch.compile(im._geo_region, dynamic=False)
    for n in (10_000, 40_000):
        for autocast in (True, False):
            args = inputs(n, autocast)
            for name, region, extra in (("eager", im._geo_region, (None, True)), ("compiled", compiled_region, (None, True))):
                try:
                    rec = dict(region=name, tokens=n, autocast=autocast, **fwd_bwd_times(region, args, autocast, extra))
                except torch.OutOfMemoryError:
                    rec = dict(region=name, tokens=n, autocast=autocast, status="oom")
                torch.cuda.empty_cache()
                res["baselines"].append(rec); print(rec, flush=True)
            for block_n in (2, 4, 8, 16):
                for warps in (2, 4, 8):
                    gk.FWD_BLOCK_N = gk.BWD_BLOCK_N = block_n
                    gk.NUM_WARPS = warps
                    try:
                        rec = dict(tokens=n, autocast=autocast, block_n=block_n, num_warps=warps, **fwd_bwd_times(fused_geo_region, args, autocast, (True,)))
                    except Exception as e:  # noqa: BLE001
                        rec = dict(tokens=n, autocast=autocast, block_n=block_n, num_warps=warps, status="error", error=str(e)[:200])
                    res["sweep"].append(rec); print(rec, flush=True)
            gk.FWD_BLOCK_N = gk.BWD_BLOCK_N = None; gk.NUM_WARPS = 4
            bc.dump(res, out)
    best = {}
    for rec in res["sweep"]:
        if "fwd_ms" not in rec:
            continue
        key = (rec["tokens"], rec["autocast"])
        for k in ("fwd_ms", "bwd_ms"):
            if k not in best.setdefault(str(key), {}) or rec[k] < best[str(key)][k][1]:
                best[str(key)][k] = ((rec["block_n"], rec["num_warps"]), rec[k])
    res["best"] = best
    bc.dump(res, out)
    print("best", best)


if __name__ == "__main__":
    main()
