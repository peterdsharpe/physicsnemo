"""torch.profiler tables for the ISLA kernel study: top CUDA kernels by time, split into
forward / backward / optimizer phases, and the top operators by device memory.

Phase attribution: the training step wraps forward, backward and optimizer in
record_function ranges on the CPU side (outside any compiled region). Every CUDA
kernel in the trace carries the correlation id of the runtime launch that issued
it; the launch's CPU timestamp falls inside exactly one range, which gives the
kernel its phase even though it runs asynchronously later.

Memory: (a) key_averages self_device_memory_usage per operator (eager only: under
torch.compile the allocations belong to the generated code), (b) a categorized
snapshot of every live block at the moment of peak allocation, from
torch.cuda.memory._record_memory_history, bucketed by the ISLA source line that
allocated it (eager) or by the compiled frame (compiled).

Usage: python profile_kernels.py <out.json> <option> [tokens=10000] [batch=1] [precision=bf16|fp32]
"""
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_common as bc  # noqa: E402
from kernel_options import OPTIONS, apply_option  # noqa: E402

N_PROF = 3


def phase_kernel_table(prof):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    tr = json.load(open(path))["traceEvents"]
    os.unlink(path)
    ranges = [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in tr
              if e.get("cat") in ("user_annotation", "cpu_op") and e.get("name") in ("phase:forward", "phase:backward", "phase:optimizer")]
    ranges.sort()
    corr_phase = {}
    for e in tr:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            ts = e["ts"]
            for a, b, name in ranges:
                if a <= ts <= b:
                    corr_phase[e["args"]["correlation"]] = name.split(":")[1]
                    break
    by = defaultdict(lambda: defaultdict(float))
    tot = defaultdict(float)
    for e in tr:
        if e.get("cat") == "kernel":
            ph = corr_phase.get(e.get("args", {}).get("correlation"), "unattributed")
            by[ph][e["name"]] += e["dur"] / 1e3 / N_PROF
            tot[ph] += e["dur"] / 1e3 / N_PROF
    table = {}
    for ph, d in by.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:25]
        table[ph] = dict(total_ms=tot[ph], n_distinct_kernels=len(d), n_launches=sum(1 for e in tr if e.get("cat") == "kernel" and corr_phase.get(e.get("args", {}).get("correlation"), "unattributed") == ph) / N_PROF,
                         top_kernels=[dict(kernel=k[:200], ms=v, pct=100 * v / tot[ph]) for k, v in top])
    return table


def classify_kernel(name):
    l = name.lower()
    if "triton_" in l:
        kind = "inductor_" + ("red" if "_red_" in l else "poi" if "_poi_" in l else "per" if "_per_" in l else "tem" if "_tem_" in l else "other")
        return kind
    if "softmax" in l:
        return "softmax"
    if re.search(r"gemm|cutlass|cublas|xmma|sm80|sm89|sm90", l):
        return "gemm"
    if re.search(r"layer_norm|layernorm|gammabeta", l):
        return "layernorm"
    if re.search(r"adam|foreach|multi_tensor", l):
        return "optimizer"
    if re.search(r"reduce_kernel|reduce", l):
        return "reduce"
    if re.search(r"elementwise|vectorized|unrolled|copy|fill|catarray|index|gather|scatter", l):
        return "elementwise_copy"
    return "other"


def memory_snapshot_at_peak(step, source_marker="isla/model.py"):
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    step()
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    # replay the device trace to find the live set at peak
    live, cur, peak, peak_live = {}, 0, 0, {}
    for tr in snap.get("device_traces", []):
        for ev in tr:
            act = ev.get("action")
            if act == "alloc":
                live[ev["addr"]] = ev
                cur += ev["size"]
                if cur > peak:
                    peak, peak_live = cur, dict(live)
            elif act in ("free_completed",):
                e = live.pop(ev["addr"], None)
                if e is not None:
                    cur -= e["size"]
    buckets = defaultdict(lambda: [0, 0])
    for ev in peak_live.values():
        frames = ev.get("frames", [])
        key = "unknown"
        for fr in frames:
            fn = fr.get("filename", "")
            if source_marker in fn:
                key = f"{os.path.basename(fn)}:{fr['line']} {fr['name']}"
                break
        else:
            for fr in frames:
                fn = fr.get("filename", "")
                if "torch/_inductor" in fn or "torch/_functorch" in fn or "torch/_dynamo" in fn:
                    key = "compiled_graph:" + fr["name"]
                    break
        buckets[key][0] += ev["size"]
        buckets[key][1] += 1
    top = sorted(buckets.items(), key=lambda kv: -kv[1][0])[:25]
    return dict(peak_live_bytes=peak, peak_live_gib=peak / bc.GIB, n_live_blocks=len(peak_live),
                buckets=[dict(site=k, bytes=v[0], gib=v[0] / bc.GIB, pct=100 * v[0] / peak, n_blocks=v[1]) for k, v in top])


def main():
    out_path, opt_name = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    autocast = (sys.argv[5] if len(sys.argv) > 5 else "bf16") == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = True
    opt = OPTIONS[opt_name]
    model, kw = bc.make_model(**opt.get("model_kw", {}))
    model = apply_option(model, opt)
    inputs, target = bc.make_inputs(n, batch, out_dim=4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optim.zero_grad(set_to_none=True)
        with record_function("phase:forward"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                out = model(**inputs)
            loss = torch.nn.functional.mse_loss(out.float(), target)
        with record_function("phase:backward"):
            loss.backward()
        with record_function("phase:optimizer"):
            optim.step()

    res = dict(env=bc.env_info(), option=opt_name, description=opt["description"], model_kw=kw, tokens=n, batch=batch,
               precision="bf16_autocast" if autocast else "fp32", load=bc.load_snapshot())
    res["timing"] = bc.measure(step, n_warm=3, n_meas=10)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False, profile_memory=True) as prof:
        for _ in range(N_PROF):
            step()
        torch.cuda.synchronize()
    res["kernels_by_phase"] = phase_kernel_table(prof)
    # kernel-kind buckets over all phases
    kinds = defaultdict(float)
    for ph, t in res["kernels_by_phase"].items():
        for k in t["top_kernels"]:
            pass
    ka = prof.key_averages()
    for ev in ka:
        if ev.self_device_time_total > 0 and ev.device_type.name == "CUDA":
            kinds[classify_kernel(ev.key)] += ev.self_device_time_total / 1e3 / N_PROF
    res["cuda_ms_by_kernel_kind"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))
    res["total_cuda_ms_per_step"] = sum(kinds.values())
    mem_ops = [(ev.key, ev.self_device_memory_usage / bc.GIB / N_PROF, ev.count / N_PROF) for ev in ka if ev.self_device_memory_usage > 0]
    res["top_ops_by_device_memory_allocated"] = [dict(op=k[:160], gib_per_step=v, calls_per_step=c) for k, v, c in sorted(mem_ops, key=lambda x: -x[1])[:25]]
    res["memory_at_peak"] = memory_snapshot_at_peak(step)
    bc.dump(res, out_path)
    print(json.dumps({k: res[k] for k in ("option", "tokens", "batch", "precision", "total_cuda_ms_per_step", "cuda_ms_by_kernel_kind")}, indent=1))
    for ph, t in res["kernels_by_phase"].items():
        print(f"== {ph}: {t['total_ms']:.1f} ms, {t['n_launches']:.0f} launches/step")
        for k in t["top_kernels"][:8]:
            print(f"   {k['ms']:7.2f} ms {k['pct']:5.1f}%  {k['kernel'][:100]}")
    print("== memory at peak", res["memory_at_peak"]["peak_live_gib"], "GiB")
    for b in res["memory_at_peak"]["buckets"][:10]:
        print(f"   {b['gib']:6.3f} GiB {b['pct']:5.1f}%  {b['site'][:100]}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
