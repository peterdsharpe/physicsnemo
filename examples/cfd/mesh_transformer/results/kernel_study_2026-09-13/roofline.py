"""Roofline of the ISLA reference configuration (hidden 192, 12 layers, 256 slices).

Three parts, all written to one JSON:

1. Device peaks, measured on this GPU rather than quoted: bf16 tensor-core GEMM,
   fp32 GEMM (TF32 off), and streaming bandwidth (a 1 GiB fp32 add: read 2, write 1).

2. Analytic FLOPs and bytes per stage of one training step (forward + backward,
   with the geo_checkpoint recompute), for tokens in {10k, 40k}, batch in {1, 2},
   bf16 autocast and fp32. Bytes are the traffic of the EAGER implementation (every
   intermediate tensor written and read back once) and, alongside, the traffic of an
   ideally fused kernel for the same stage (inputs read once, outputs written once).
   The ratio of the two is the fusion headroom; the roofline time of a stage is
   max(flops / peak_flops, bytes / peak_bw).

3. Achieved time per stage, from a torch.profiler trace of the eager reference
   (record_shapes=True): every CUDA kernel is attributed to the innermost CPU
   operator that launched it (by correlation id and timestamp), and the operator is
   assigned a stage from its name and input shapes. The achieved bandwidth or FLOP
   rate of each stage against the measured peaks says how far each stage sits from
   its roofline.

Stages: assign (LayerNorm + hidden->slices Linear), point_softmax (routing softmax
over points, both per layer), anchors (a^T r, a^T n), invariants (the (B,N,S,.)
relational geometry), geo_linear (6->1 routing bias and the pooled 6->hidden/2
projection), slice_softmax (point->slice mix), geo_pool (mix-weighted pooling of the
invariants), slice_states_readback (a^T h and mix z, plus the slice MLP),
mlp (broadcast Linear + point MLP), layernorm_gelu, autocast_casts, head_embed,
optimizer, other.

Usage: python roofline.py <out.json>
"""
import bisect
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_common as bc  # noqa: E402

H, S, L, MLP = 192, 256, 12, 4
GEO = 6  # relative frame: six relational invariants


# ---------------------------------------------------------------- 1. device peaks
def measure_peaks():
    dev = "cuda"
    out = {}
    for name, dt, tf32 in (("bf16_gemm_tflops", torch.bfloat16, False), ("fp32_gemm_tflops", torch.float32, False), ("tf32_gemm_tflops", torch.float32, True)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        n = 8192 if dt == torch.bfloat16 else 6144
        a = torch.randn(n, n, device=dev, dtype=dt); b = torch.randn(n, n, device=dev, dtype=dt)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            a @ b
        torch.cuda.synchronize(); dt_s = (time.perf_counter() - t0) / 10
        out[name] = 2 * n**3 / dt_s / 1e12
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(2**28, device=dev); y = torch.randn(2**28, device=dev)  # 1 GiB each
    for _ in range(3):
        z = x + y
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        z = x + y
    torch.cuda.synchronize(); dt_s = (time.perf_counter() - t0) / 10
    out["stream_bw_gbs"] = 3 * x.numel() * 4 / dt_s / 1e9
    del x, y, z
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- 2. analytic model
def analytic(N, B, bf16):
    """Per-stage FLOPs and bytes for one training step (fwd + bwd + geo recompute)."""
    f = 4  # geometry and routing tensors stay fp32 under autocast (r, n, z, logits are fp32)
    m = 2 if bf16 else 4  # matmul operand dtype
    NS = B * N * S
    st = {}

    def add(name, flops_f, bytes_eager_f, bytes_fused_f, bwd_flops_mult=2.0, bwd_bytes_mult=2.0, recompute=False):
        # backward of a GEMM is two GEMMs (2x flops); backward of memory-bound elementwise
        # roughly re-reads inputs and writes grads (taken as 2x forward traffic).
        rec = 1.0 if recompute else 0.0
        st[name] = dict(flops=flops_f * (1 + bwd_flops_mult + rec), bytes_eager=bytes_eager_f * (1 + bwd_bytes_mult + rec),
                        bytes_fused=bytes_fused_f * (1 + bwd_bytes_mult + rec), flops_fwd=flops_f, bytes_eager_fwd=bytes_eager_f)

    # assign: LN (3 passes over B N H) + Linear H->S
    add("assign", L * 2 * B * N * H * S, L * (B * N * H * 4 * 3 + B * N * H * m + NS * m + NS * f), L * (B * N * H * 4 + NS * f))
    # point softmax: two per layer; eager: add(log_w) r+w, transpose copy r+w, softmax r+w  (6 passes of NS fp32)
    add("point_softmax", L * 2 * 5 * NS, L * 2 * 6 * NS * f, L * 2 * 2 * NS * f)
    # anchors: two bmm (S x N)(N x 3), operands cast to bf16 under autocast (cast: read fp32 write bf16)
    add("anchors", L * 2 * 2 * NS * 3, L * (NS * f + NS * m + 2 * NS * m), L * NS * m)
    # invariants (relative frame): rel, dist, clamp, rel_hat, 4 dots, log, cat -> (B,N,S,6); ~40 flops per (n,s)
    inv_eager = L * (3 + 4 + 2 + 7 + 4 * 10 + 1 + 12) * NS * f  # passes over NS-sized fp32 tensors
    inv_fused = L * (GEO * NS * f)  # a fused kernel writes the (B,N,S,6) invariants once (or nothing, if fused into the routing)
    add("invariants", L * 40 * NS, inv_eager, inv_fused, bwd_flops_mult=2.0, bwd_bytes_mult=2.5, recompute=True)
    # geo_linear: cast geo to bf16 (r 6f, w 6m), GEMM (NS x 6)(6 x 1) r 6m w m; pooled 6->H/2 projection is B N-sized (small)
    add("geo_linear", L * (2 * NS * GEO + 2 * B * N * GEO * (H // 2)), L * (GEO * NS * f + GEO * NS * m + GEO * NS * m + NS * m), L * (GEO * NS * m + NS * m), recompute=True)
    # slice softmax: add bias (r 2, w 1), softmax (r 1, w 1) over (B,N,S)
    add("slice_softmax", L * 6 * NS, L * 5 * NS * f, L * 2 * NS * f, recompute=True)
    # geo_pool: bmm (B N)x(1,S)x(S,6): read mix (cast to bf16) and geo bf16
    add("geo_pool", L * 2 * NS * GEO, L * (NS * f + NS * m + GEO * NS * m + NS * m), L * (NS * m + GEO * NS * m), recompute=True)
    # slice states + readback: (S x N)(N x H) and (N x S)(S x H) + slice MLP
    add("slice_states_readback", L * (2 * 2 * NS * H + 2 * 2 * B * S * H * MLP * H), L * (2 * (NS * f + NS * m) + 4 * B * N * H * m), L * (2 * NS * m + 4 * B * N * H * m))
    # broadcast Linear (2.5H -> H), point MLP (H->4H->H), geo_feat (6 -> H/2)
    add("mlp", L * (2 * B * N * (2 * H + H // 2) * H + 2 * 2 * B * N * H * MLP * H), L * (B * N * (2 * H + H // 2 + H + MLP * H * 2 + H) * m), L * (B * N * (2 * H + H // 2 + H + MLP * H * 2 + H) * m))
    add("layernorm_gelu", L * (2 * 5 * B * N * H + 8 * B * N * MLP * H), L * (2 * 3 * B * N * H * 4 + 2 * B * N * MLP * H * m), L * (2 * 2 * B * N * H * 4 + 2 * B * N * MLP * H * m))
    # AdamW over 8.7M params: read p, g, m, v; write p, m, v (fp32)
    n_params = 8_697_000
    st["optimizer"] = dict(flops=12 * n_params, bytes_eager=7 * 4 * n_params, bytes_fused=7 * 4 * n_params, flops_fwd=0, bytes_eager_fwd=0)
    return st


def roofline_times(stages, peaks, bf16):
    pf = (peaks["bf16_gemm_tflops"] if bf16 else peaks["fp32_gemm_tflops"]) * 1e12
    bw = peaks["stream_bw_gbs"] * 1e9
    out = {}
    for k, v in stages.items():
        t_c = v["flops"] / pf
        t_m_e = v["bytes_eager"] / bw
        t_m_f = v["bytes_fused"] / bw
        out[k] = dict(v, roofline_ms_eager=1e3 * max(t_c, t_m_e), roofline_ms_fused=1e3 * max(t_c, t_m_f),
                      bound="memory" if t_m_e > t_c else "compute")
    return out


# ---------------------------------------------------------------- 3. achieved per stage
def classify(name, dims, N, B):
    l = name.lower()
    flat = []
    for d in dims:
        if d and isinstance(d[0], list):  # aten::cat lists its inputs' shapes one level down
            flat += [tuple(x) for x in d if x]
        elif d:
            flat.append(tuple(d))
    dims = flat
    def has(shape):
        return any(d == shape for d in dims)
    def has4d_ns():
        return any(len(d) == 4 and d[1] == N and d[2] == S for d in dims) or any(len(d) == 4 and d[2] == S and d[1] in (N, 1) for d in dims)
    if re.search(r"adam|_foreach|lerp|addcmul|addcdiv|zero_", l):
        return "optimizer"
    if "layer_norm" in l or "gelu" in l:
        return "layernorm_gelu"
    if "softmax" in l:
        return "point_softmax" if has((B, S, N)) else "slice_softmax"
    if l in ("aten::bmm", "aten::baddbmm"):
        if any(d[-1] == 3 or (len(d) == 3 and d[1] == 3) for d in dims):
            return "anchors"
        if any(d[0] == B * N for d in dims):
            return "geo_pool"
        return "slice_states_readback"
    if l in ("aten::mm", "aten::addmm"):
        if any(d[-1] == GEO or (len(d) == 2 and d[0] == GEO) for d in dims):
            return "geo_linear"
        if any(len(d) == 2 and d[0] == B * N * S for d in dims) or any(len(d) == 2 and d[1] == B * N * S for d in dims):
            return "geo_linear"
        if any(len(d) == 2 and (d[0] == B * S or d[1] == B * S) for d in dims):
            return "slice_states_readback"
        if any(len(d) == 2 and S in d and H in d for d in dims) or any(len(d) == 2 and d == (B * N, S) for d in dims):
            return "assign"
        if any(len(d) == 2 and (MLP * H in d or 2 * H + H // 2 in d or (d[0] == H and d[1] == H)) for d in dims):
            return "mlp"
        return "head_embed"
    if has4d_ns():
        return "invariants"
    if l in ("aten::copy_", "aten::_to_copy", "aten::to", "aten::contiguous", "aten::clone"):
        if has((B, S, N)) or has((B, N, S)):
            return "point_softmax" if has((B, S, N)) else "autocast_casts"
        return "other"
    if has((B, N, S)) or has((B, S, N)):
        if l in ("aten::add", "aten::add_") and any(d == (B, N, 1) for d in dims):
            return "point_softmax"
        if has((B, S, N)):
            return "point_softmax"
        return "slice_softmax"
    if any(len(d) == 3 and d[1] == N and d[2] in (7, 3, 1) for d in dims) or "cross" in l or "stack" in l:
        return "head_embed"
    if any(d[-1] in (H, MLP * H, 2 * H + H // 2) and (len(d) == 2 or (len(d) == 3 and d[1] == N)) for d in dims):
        return "mlp"  # residual adds, bias-gradient sums, casts of the (B,N,H)/(B,N,4H) activations
    return "other"


def attribute_profile(N, B, autocast, n_prof=3):
    torch.backends.cuda.matmul.allow_tf32 = True
    model, _ = bc.make_model()
    inputs, target = bc.make_inputs(N, B, out_dim=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step = bc.train_step_fn(model, inputs, target, opt, autocast)
    timing = bc.measure(step, 3, 10)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(n_prof):
            step()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    tr = json.load(open(path))["traceEvents"]
    os.unlink(path)
    ops = defaultdict(list)
    for e in tr:
        if e.get("cat") == "cpu_op" and e.get("name", "").startswith("aten::"):
            ops[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"], e.get("args", {}).get("Input Dims", [])))
    for tid in ops:
        ops[tid].sort()
    starts = {tid: [o[0] for o in v] for tid, v in ops.items()}
    corr_op = {}
    for e in tr:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            tid, ts = e["tid"], e["ts"]
            if tid not in ops:
                continue
            i = bisect.bisect_right(starts[tid], ts) - 1
            best = None
            while i >= 0 and ts - ops[tid][i][0] < 5e6:
                a, b, name, dims = ops[tid][i]
                if a <= ts <= b and (best is None or (b - a) < (best[1] - best[0])):
                    best = (a, b, name, dims)
                i -= 1
            if best is not None:
                corr_op[e["args"]["correlation"]] = (best[2], best[3])
    by_stage, by_stage_op = defaultdict(float), defaultdict(lambda: defaultdict(float))
    unattributed = 0.0
    for e in tr:
        if e.get("cat") == "kernel":
            ms = e["dur"] / 1e3 / n_prof
            op = corr_op.get(e.get("args", {}).get("correlation"))
            if op is None:
                unattributed += ms; by_stage["unattributed"] += ms; continue
            stg = classify(op[0], op[1], N, B)
            by_stage[stg] += ms
            by_stage_op[stg][f"{op[0]} {op[1][:3]}"] += ms
    detail = {s: sorted(((k, v) for k, v in d.items()), key=lambda kv: -kv[1])[:6] for s, d in by_stage_op.items()}
    del model, opt, inputs, target
    torch.cuda.empty_cache()
    return dict(timing_ms=timing["step_ms_median"], peak_gib=timing["peak_allocated_gib"], cuda_ms_by_stage=dict(by_stage),
                total_cuda_ms=sum(by_stage.values()), detail={s: [dict(op=k, ms=v) for k, v in d] for s, d in detail.items()})


def main():
    out_path = sys.argv[1]
    res = dict(env=bc.env_info(), load=bc.load_snapshot(), reference=dict(H=H, S=S, L=L, mlp_ratio=MLP, geo=GEO))
    res["peaks"] = measure_peaks()
    print("peaks", res["peaks"], flush=True)
    res["analytic"] = {}
    for N in (10_000, 40_000):
        for B in (1, 2):
            for prec in ("bf16", "fp32"):
                st = roofline_times(analytic(N, B, prec == "bf16"), res["peaks"], prec == "bf16")
                tot = dict(flops=sum(v["flops"] for v in st.values()), bytes_eager=sum(v["bytes_eager"] for v in st.values()),
                           bytes_fused=sum(v["bytes_fused"] for v in st.values()), roofline_ms_eager=sum(v["roofline_ms_eager"] for v in st.values()),
                           roofline_ms_fused=sum(v["roofline_ms_fused"] for v in st.values()))
                res["analytic"][f"N{N}_B{B}_{prec}"] = dict(stages=st, total=tot)
                print(f"N={N} B={B} {prec}: {tot['flops']/1e9:.0f} GFLOP, eager {tot['bytes_eager']/1e9:.0f} GB -> roofline {tot['roofline_ms_eager']:.1f} ms; fused {tot['bytes_fused']/1e9:.0f} GB -> {tot['roofline_ms_fused']:.1f} ms", flush=True)
    res["achieved"] = {}
    for N, B, prec in ((10_000, 1, "bf16"), (10_000, 1, "fp32"), (10_000, 2, "bf16"), (40_000, 1, "bf16"), (40_000, 1, "fp32"), (40_000, 2, "bf16"), (10_000, 2, "fp32"), (40_000, 2, "fp32")):
        key = f"N{N}_B{B}_{prec}"
        try:
            a = attribute_profile(N, B, prec == "bf16")
            st = res["analytic"][key]["stages"]
            bw = res["peaks"]["stream_bw_gbs"] * 1e9
            pf = (res["peaks"]["bf16_gemm_tflops"] if prec == "bf16" else res["peaks"]["fp32_gemm_tflops"]) * 1e12
            for s, ms in a["cuda_ms_by_stage"].items():
                if s in st and ms > 0:
                    st[s]["achieved_ms"] = ms
                    st[s]["achieved_gbs_eager_traffic"] = st[s]["bytes_eager"] / (ms / 1e3) / 1e9
                    st[s]["achieved_tflops"] = st[s]["flops"] / (ms / 1e3) / 1e12
                    st[s]["pct_of_bw_peak"] = 100 * st[s]["achieved_gbs_eager_traffic"] / (bw / 1e9)
                    st[s]["pct_of_flops_peak"] = 100 * st[s]["achieved_tflops"] / (pf / 1e12)
            res["achieved"][key] = a
            print(key, f"step {a['timing_ms']:.1f} ms, cuda {a['total_cuda_ms']:.1f} ms", {k: round(v, 1) for k, v in sorted(a["cuda_ms_by_stage"].items(), key=lambda kv: -kv[1])}, flush=True)
        except torch.OutOfMemoryError as e:
            res["achieved"][key] = dict(status="oom", error=str(e)[:200]); torch.cuda.empty_cache(); print(key, "OOM", flush=True)
        bc.dump(res, out_path)
    bc.dump(res, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
