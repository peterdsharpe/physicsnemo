"""Render the study's markdown tables (TABLES.md) from the artifact JSONs in this
directory: roofline (analytic vs achieved per stage), profile (top kernels by phase,
memory at peak), Pareto (local GPU and GB300), the step-time reconciliation table,
and the numerics gate. Every number in the report and the notebook entry is produced
here from a committed artifact.

Usage: python make_tables.py   (reads and writes in its own directory)
"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def env_line(env, load=None):
    s = f"{env['gpu']}, driver {env.get('driver', '?')}, torch {env['torch']} (CUDA {env['cuda']}), host {env.get('hostname', '?')}"
    if load:
        s += f"; load snapshot: GPU util {load.get('gpu_util_pct', '?')}%, {load.get('gpu_mem_used_mib', '?')} MiB used before the run, loadavg {load.get('loadavg', '?')}"
    return s


def pareto_section(name, d, title):
    if d is None:
        return f"## {title}\n\n(artifact {name} missing)\n"
    lines = [f"## {title}", "", f"Artifact: `{name}`. {env_line(d['env'], d['runs'][0].get('load'))}.", "",
             "Step = zero_grad, forward, MSE, backward, AdamW; median of 10 cuda-synchronized steps after 3 warm-ups; "
             "peak = max_memory_allocated over the measured steps (total, including the resident model + optimizer state).", ""]
    configs = sorted({(r["tokens"], r["batch"], r["precision"]) for r in d["runs"]})
    for n, b, prec in configs:
        rows = []
        gt = {r["option"]: r for r in d["runs"] if (r["tokens"], r["batch"], r["precision"]) == (n, b, prec)}
        anchor = gt.get("geotransolver_compiled", {}).get("step_ms_median")
        anchor_e = gt.get("geotransolver_eager", {}).get("step_ms_median")
        for r in d["runs"]:
            if (r["tokens"], r["batch"], r["precision"]) != (n, b, prec):
                continue
            if r["status"] != "ok":
                rows.append([r["option"], r["status"].upper(), "-", "-", "-", "-"]); continue
            ms = r["step_ms_median"]
            rows.append([r["option"], fmt(ms), fmt(r["peak_allocated_gib"], 2), fmt(r["incremental_peak_gib"], 2),
                         fmt(ms / anchor_e, 2) if anchor_e else "-", fmt(ms / anchor, 2) if anchor else "-"])
        lines += [f"### {n:,} tokens, batch {b}, {prec}", "",
                  table(rows, ["option", "step ms", "peak GiB", "incremental GiB", "x GT eager", "x GT compiled"]), ""]
    return "\n".join(lines)


def reconciliation(local, gb300):
    lines = ["## Step-time reconciliation (ISLA / GeoTransolver, 10,000 tokens, batch 1, bf16 autocast)", "",
             "The book's cost section quotes 3.7x from `matched_memory_isla_gt_2026-09-08.json` (GB300, eager, "
             "pre-2026-09-10 middle-dimension point softmax, geo_checkpoint on: 245.7 vs 64.9 ms). Every ratio the "
             "study can reproduce, on both instruments; ISLA rows are the reference configuration unless stated.", ""]
    rows = []
    pairs = [("eager_ckpt_slow_softmax", "geotransolver_eager", "eager, original (middle-dim) point softmax, geo_checkpoint on -- the book's configuration"),
             ("eager_ckpt", "geotransolver_eager", "eager, fast point softmax, geo_checkpoint on (isla_surface_reference.yaml, run eager)"),
             ("eager_nockpt", "geotransolver_eager", "eager, fast point softmax, geo_checkpoint off"),
             ("fused_geo_eager", "geotransolver_eager", "eager + fused Triton geometry region (geo_kernel='fused')"),
             ("compile_model_ckpt", "geotransolver_compiled", "torch.compile(model) both (the recipe's compile: true), geo_checkpoint on"),
             ("compile_model_nockpt", "geotransolver_compiled", "torch.compile(model) both, geo_checkpoint off"),
             ("fused_geo_compile_model", "geotransolver_compiled", "torch.compile(model) both + fused geometry region")]
    for iso, gto, desc in pairs:
        row = [desc]
        for d in (local, gb300):
            if d is None:
                row += ["-", "-"]; continue
            recs = {r["option"]: r for r in d["runs"] if (r["tokens"], r["batch"], r["precision"]) == (10_000, 1, "bf16_autocast") and r["status"] == "ok"}
            a, g = recs.get(iso), recs.get(gto)
            if a and g:
                row += [f"{a['step_ms_median']:.1f} / {g['step_ms_median']:.1f}", f"{a['step_ms_median'] / g['step_ms_median']:.2f}x"]
            else:
                row += ["-", "-"]
        rows.append(row)
    lg = local["env"]["gpu"] if local else "local GPU"
    gg = gb300["env"]["gpu"] if gb300 else "GB300"
    lines += [table(rows, ["configuration (ISLA vs GeoTransolver)", f"{lg}: ISLA / GT ms", "ratio", f"{gg}: ISLA / GT ms", "ratio"]), ""]
    if gb300:
        recs40 = {r["option"]: r for r in gb300["runs"] if (r["tokens"], r["batch"], r["precision"]) == (40_000, 1, "bf16_autocast") and r["status"] == "ok"}
        if recs40:
            rows = [[k, fmt(v["step_ms_median"]), fmt(v["peak_allocated_gib"], 2)] for k, v in recs40.items()]
            lines += ["GB300 at 40,000 tokens, batch 1, bf16 (same artifact):", "", table(rows, ["option", "step ms", "peak GiB"]), ""]
    return "\n".join(lines)


def roofline_section(r):
    if r is None:
        return "## Roofline\n\n(artifact roofline.json missing)\n"
    p = r["peaks"]
    lines = ["## Roofline of the reference configuration", "", f"Artifact: `roofline.json`. {env_line(r['env'], r['load'])}.", "",
             f"Measured peaks on this GPU: bf16 GEMM {p['bf16_gemm_tflops']:.1f} TFLOP/s, fp32 GEMM {p['fp32_gemm_tflops']:.1f} TFLOP/s "
             f"(TF32 {p['tf32_gemm_tflops']:.1f}), streaming bandwidth {p['stream_bw_gbs']:.0f} GB/s (1 GiB fp32 add).", ""]
    for key, a in r["analytic"].items():
        ach = r["achieved"].get(key, {})
        st = a["stages"]
        rows = []
        for s, v in sorted(st.items(), key=lambda kv: -kv[1]["bytes_eager"]):
            rows.append([s, f"{v['flops'] / 1e9:.1f}", f"{v['bytes_eager'] / 1e9:.2f}", f"{v['bytes_fused'] / 1e9:.2f}", v["bound"],
                         f"{v['roofline_ms_eager']:.2f}", f"{v['roofline_ms_fused']:.2f}",
                         fmt(v.get("achieved_ms"), 2), fmt(v.get("pct_of_bw_peak"), 0), fmt(v.get("pct_of_flops_peak"), 1)])
        tot = a["total"]
        hdr = f"### {key.replace('_', ' ')}"
        if "timing_ms" in ach:
            hdr += f" -- measured eager step {ach['timing_ms']:.1f} ms ({ach['total_cuda_ms']:.1f} ms of CUDA kernels), peak {ach['peak_gib']:.2f} GiB"
        elif ach.get("status") == "oom":
            hdr += " -- eager reference OOM on this GPU (analytic only)"
        lines += [hdr, "",
                  f"Totals: {tot['flops'] / 1e9:.0f} GFLOP per step; eager traffic {tot['bytes_eager'] / 1e9:.0f} GB -> roofline {tot['roofline_ms_eager']:.1f} ms; "
                  f"fused traffic {tot['bytes_fused'] / 1e9:.0f} GB -> roofline {tot['roofline_ms_fused']:.1f} ms.", "",
                  table(rows, ["stage", "GFLOP", "eager GB", "fused GB", "bound", "roofline ms (eager)", "roofline ms (fused)", "achieved ms", "% of BW peak (eager traffic)", "% of FLOP peak"]), ""]
    return "\n".join(lines)


def profile_section(name, d):
    if d is None:
        return f"## Profile {name}\n\n(missing)\n"
    lines = [f"## Profile: {d['option']} ({d['tokens']:,} tokens, batch {d['batch']}, {d['precision']})", "",
             f"Artifact: `{name}`. {env_line(d['env'], d['load'])}. Step {d['timing']['step_ms_median']:.1f} ms, "
             f"peak {d['timing']['peak_allocated_gib']:.2f} GiB; CUDA kernel time {d['total_cuda_ms_per_step']:.1f} ms per step.", "",
             "CUDA ms per step by kernel kind: " + ", ".join(f"{k} {v:.1f}" for k, v in d["cuda_ms_by_kernel_kind"].items()), ""]
    for ph in ("forward", "backward", "optimizer"):
        t = d["kernels_by_phase"].get(ph)
        if not t:
            continue
        rows = [[k["kernel"][:90], f"{k['ms']:.2f}", f"{k['pct']:.0f}"] for k in t["top_kernels"][:10]]
        lines += [f"### {ph}: {t['total_ms']:.1f} ms, {t['n_launches']:.0f} kernel launches per step", "", table(rows, ["kernel", "ms/step", "% of phase"]), ""]
    m = d["memory_at_peak"]
    rows = [[b["site"][:90], f"{b['gib']:.3f}", f"{b['pct']:.0f}", b["n_blocks"]] for b in m["buckets"][:12]]
    lines += [f"### Live allocations at the peak ({m['peak_live_gib']:.2f} GiB, {m['n_live_blocks']} blocks), by allocating source line", "",
              table(rows, ["allocation site", "GiB", "% of peak", "blocks"]), ""]
    if d.get("top_ops_by_device_memory_allocated"):
        rows = [[o["op"][:70], f"{o['gib_per_step']:.2f}", f"{o['calls_per_step']:.0f}"] for o in d["top_ops_by_device_memory_allocated"][:10]]
        lines += ["### Operators by device memory allocated per step (profiler self_device_memory_usage)", "", table(rows, ["operator", "GiB allocated/step", "calls/step"]), ""]
    return "\n".join(lines)


def numerics_section(name, d):
    if d is None:
        return f"## Numerics {name}\n\n(missing)\n"
    f = d["fp32_floor"]
    lines = [f"## Numerics gate ({d['tokens']:,} tokens, batch {d['batch']}, fp32, TF32 off)", "",
             f"Artifact: `{name}`. {env_line(d['env'])}. Reference: eager, geo_checkpoint=True, fast_point_softmax=True.", "",
             f"fp32 roundoff floor of the reference itself (eager fp32 vs eager fp64, same weights): output rel L2 {f['out_rel_l2']:.2e}, "
             f"gradient rel L2 {f['grad_rel_l2']:.2e}.", ""]
    rows = []
    for k, v in d["options"].items():
        if v.get("status") != "ok":
            rows.append([k, "ERROR", "-", "-", "-", "-", "-"]); continue
        ok = v["out_rel_l2"] < 1e-6 and v["grad_rel_l2"] < 1e-5 and v["grad_max_param_rel_l2"] < 1e-5
        rows.append([k, f"{v['out_rel_l2']:.1e}", f"{v['grad_rel_l2']:.1e}", f"{v['grad_max_param_rel_l2']:.1e}",
                     fmt(v.get("fp64_out_rel_l2"), 1) if v.get("fp64_out_rel_l2") is None else f"{v['fp64_out_rel_l2']:.1e}",
                     fmt(v.get("fp64_grad_rel_l2"), 1) if v.get("fp64_grad_rel_l2") is None else f"{v['fp64_grad_rel_l2']:.1e}",
                     "pass" if ok else "FAIL"])
    lines += [table(rows, ["option", "fp32 out rel L2 (bar 1e-6)", "fp32 grad rel L2 (bar 1e-5)", "worst param grad rel L2", "fp64 out rel L2", "fp64 grad rel L2", "bars"]), ""]
    return "\n".join(lines)


def tune_section(t):
    if t is None:
        return "## Fused kernel launch sweep\n\n(missing)\n"
    lines = ["## Fused geometry region: kernel-level timing and launch sweep", "", f"Artifact: `tune_fused_kernel.json`. {env_line(t['env'], t['load'])}.", ""]
    rows = []
    for b in t["baselines"]:
        rows.append([b["region"], b["tokens"], "bf16 autocast" if b["autocast"] else "fp32", fmt(b.get("fwd_ms"), 2), fmt(b.get("bwd_ms"), 2), fmt(b.get("fwd_bwd_incremental_peak_gib"), 3)])
    for key, best in t.get("best", {}).items():
        pass
    for s in t["sweep"]:
        if "fwd_ms" in s:
            rows.append([f"fused (block_n {s['block_n']}, warps {s['num_warps']})", s["tokens"], "bf16 autocast" if s["autocast"] else "fp32", f"{s['fwd_ms']:.2f}", f"{s['bwd_ms']:.2f}", f"{s['fwd_bwd_incremental_peak_gib']:.3f}"])
    lines += [table(rows, ["region implementation", "tokens", "precision", "forward ms", "backward ms", "fwd+bwd incremental peak GiB"]), ""]
    lines += ["Best launch configurations: " + json.dumps(t.get("best", {})), ""]
    return "\n".join(lines)


def merged(*names):
    """Concatenate the runs of several Pareto artifacts from the same instrument."""
    parts = [(n, load(n)) for n in names]
    parts = [(n, d) for n, d in parts if d]
    if not parts:
        return None
    out = dict(parts[0][1])
    out["runs"] = [r for _, d in parts for r in d["runs"]]
    out["sources"] = [n for n, _ in parts]
    return out


def main():
    local = load("pareto_rtx4090_partial.json")
    gb300_full = merged("pareto_gb300_full_bf16.json", "pareto_gb300_full_fp32.json")
    gb300_first = merged("pareto_gb300.json", "pareto_gb300_ext.json")
    out = ["# ISLA kernel study 2026-09-13: tables", "",
           "Generated by `make_tables.py` from the artifact JSONs in this directory. The GB300 (aga2, the "
           "cluster the recipe trains on) is the primary instrument; the RTX 4090 Laptop numbers are the "
           "development-machine measurements taken before the study moved to the cluster.", ""]
    out.append(reconciliation(local, gb300_full or gb300_first))
    for name in ("roofline_gb300.json", "roofline.json"):
        out.append(roofline_section(load(name)).replace("## Roofline of the reference configuration", f"## Roofline of the reference configuration (`{name}`)"))
    for name in sorted(os.listdir(D)):
        if name.startswith("profile_") and name.endswith(".json"):
            out.append(profile_section(name, load(name)))
    if gb300_full:
        out.append(pareto_section(" + ".join(gb300_full["sources"]), gb300_full, "Pareto: GB300, all options (aga2 job 738400)"))
    if gb300_first:
        out.append(pareto_section(" + ".join(gb300_first["sources"]), gb300_first, "Pareto: GB300, reconciliation set (aga2 jobs 736338, 736352)"))
    if local:
        out.append(pareto_section("pareto_rtx4090_partial.json", local, "Pareto: RTX 4090 Laptop (partial; stopped when the study moved to the cluster)"))
    for name in ("numerics_10k.json", "numerics_fused_10k.json", "numerics_fused_10k_gb300.json"):
        out.append(numerics_section(name, load(name)))
    for name in ("tune_fused_kernel_gb300.json", "tune_fused_kernel.json"):
        out.append(tune_section(load(name)).replace("Artifact: `tune_fused_kernel.json`", f"Artifact: `{name}`"))
    open(os.path.join(D, "TABLES.md"), "w").write("\n".join(out))
    print("wrote TABLES.md")


if __name__ == "__main__":
    main()
