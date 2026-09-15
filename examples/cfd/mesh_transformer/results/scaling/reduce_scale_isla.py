"""Reduce the ISLA arm of the SCALE study (notebook #sec-nb-scale-prereg, verdict #sec-nb-scale-isla-verdict).

Per run: validation means of pressure / velocity / wall-shear relative L2 from the FLOAT32 evaluation
(hl_evals_fp32/<run>/**/metrics.jsonl, the program's reporting instrument since #sec-nb-snapshot-ladder-verdict) with the
bf16 evaluation (hl_evals/<run>/) alongside and the per-run fp32-to-bf16 shift; the two artifacts' sampled points are
asserted bit-identical per case. Reference arms use their fp32 re-evaluations when present (hl_evals_fp32 / iw_evals_fp32),
otherwise their bf16 numbers, labelled. Also
parameter count, peak logged memory and median step time from runs/<run>/train.log, and GPU-hours as the sum of
logged step times times four GPUs. Per arm: two-seed means. Runs on the cluster login node with the recipe venv.
Usage: python reduce_scale_isla.py <out.json>
"""
import glob, json, re, statistics as st, sys

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = {
    "hl": {"ref": ["mt2_hl_lr1_seed42", "mt2_hl_lr1_seed43"],
           "w384": ["scale_isla_hl_w384_seed42", "scale_isla_hl_w384_seed43"],
           "w512": ["scale_isla_hl_w512_seed42", "scale_isla_hl_w512_seed43"],
           "t20k": ["floor_hl_mt2_full_20k_seed42", "floor_hl_mt2_full_20k_seed43"],
           "t40k": ["scale_isla_hl_t40k_seed42", "scale_isla_hl_t40k_seed43"],
           "t80k": ["scale_isla_hl_t80k_seed42", "scale_isla_hl_t80k_seed43"],
           "c384x40k": ["scale_isla_hl_c384x40k_seed42", "scale_isla_hl_c384x40k_seed43"],
           "w384nw": ["scale_isla_hl_w384nw_seed42", "scale_isla_hl_w384nw_seed43"],
           "c512x80k": ["scale_isla_hl_c512x80k_seed42", "scale_isla_hl_c512x80k_seed43"]},
    "dr": {"ref": ["iw_mt2_lr1e3_seed42", "iw_mt2_lr1e3_seed43"],
           "w384": ["scale_isla_dr_w384_seed42", "scale_isla_dr_w384_seed43"],
           "w512": ["scale_isla_dr_w512_seed42", "scale_isla_dr_w512_seed43"],
           "t40k": ["scale_isla_dr_t40k_seed42", "scale_isla_dr_t40k_seed43"],
           "t80k": ["scale_isla_dr_t80k_seed42", "scale_isla_dr_t80k_seed43"],
           "c384x40k": ["scale_isla_dr_c384x40k_seed42", "scale_isla_dr_c384x40k_seed43"],
           "w384nw": ["scale_isla_dr_w384nw_seed42", "scale_isla_dr_w384nw_seed43"],
           "c512x80k": ["scale_isla_dr_c512x80k_seed42", "scale_isla_dr_c512x80k_seed43"],
           "kernelctrl": ["scale_isla_dr_kernelctrl_seed42", "scale_isla_dr_kernelctrl_seed43"],
           "g512x80k": ["scale_isla_dr_g512x80k_seed42", "scale_isla_dr_g512x80k_seed43"],
           # RELFRAME corner (rows 32-33, preregistered 2026-09-14): isla_surface_reference (frame_mode=relative,
           # scale_mode=total_measure) at width 512 x 80,000 cells, trained and evaluated under code_globin + recipe_globin
           "r512x80k": ["scale_isla_dr_r512x80k_seed42", "scale_isla_dr_r512x80k_seed43"]},
}
# Numerics families (coordinator amendment): t80k, c512x80k and kernelctrl were trained from code_perf (fast
# point-softmax kernel; bf16-roundoff-level differences from the reference checkpoints); everything else from code_isla5.
FAST_KERNEL = {"t80k", "c512x80k", "kernelctrl", "g512x80k"}
GLOBIN = {"r512x80k"}  # code_globin (9f94e0df): global-inputs API, eager geometry kernel; own reference below
# "÷ own reference" for arms whose reference is not the constant-gauge lane: the relative-frame reference
# rf_dr_mt2_meas_seed{42,43} (frame_mode=relative, scale_mode=total_measure, width 192, 10,000 cells, lr 1e-3, 500 epochs,
# from code_frame/recipe_frame), float32 on the 48 validation cars, evaluated under code_eval_frame + recipe_frame by the
# frame program (book artifact results/frame_reduction_2026-09-11.json, key drivaer_val). code_globin evaluates the
# old-signature arithmetic bitwise at K=1, so the corner under code_globin is like for like (coordinator, 2026-09-14).
OWN_REF = {"dr_r512x80k": {"runs": ["rf_dr_mt2_meas_seed42", "rf_dr_mt2_meas_seed43"], "pressure_l2": 0.05522,
                           "seed_pressure": [0.054718, 0.055721], "wss_l2": [0.080220, 0.080929],
                           "snapshot": "code_eval_frame", "source": "results/frame_reduction_2026-09-11.json#drivaer_val"}}
STEP = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: ([0-9.eE+-]+|nan) Step: ([0-9.]+)s Mem: ([0-9.]+)GB")


# Snapshot provenance. Launchers write <OUTDIR>/SNAPSHOT_USED (the first PYTHONPATH entry) since 2026-09-10; evaluations
# made before that are named here from the launcher versions in force when they ran. A number without a nameable snapshot
# is refused. Pre-sidecar manifest (root, run-prefix or dir) -> snapshot:
PRE_SIDECAR_SNAPSHOTS = {
    # float32/bf16 headline evaluations of the SCALE arms (code_isla5 launcher for the reference-kernel arms, code_perf for
    # the fast-kernel arms; both loaded their own checkpoints, 0 skipped loads)
    ("hl_evals", "scale_isla_dr_w384_"): "code_isla5", ("hl_evals_fp32", "scale_isla_dr_w384_"): "code_isla5",
    ("hl_evals", "scale_isla_dr_w512_"): "code_isla5", ("hl_evals_fp32", "scale_isla_dr_w512_"): "code_isla5",
    ("hl_evals", "scale_isla_dr_w384nw_"): "code_isla5", ("hl_evals_fp32", "scale_isla_dr_w384nw_"): "code_isla5",
    ("hl_evals", "scale_isla_dr_t80k_"): "code_perf", ("hl_evals_fp32", "scale_isla_dr_t80k_"): "code_perf",
    ("hl_evals", "scale_isla_dr_c512x80k_"): "code_perf", ("hl_evals_fp32", "scale_isla_dr_c512x80k_"): "code_perf",
    ("hl_evals", "scale_isla_dr_kernelctrl_"): "code_perf", ("hl_evals_fp32", "scale_isla_dr_kernelctrl_"): "code_perf",
    # reference arms: fp32 re-evaluation campaign (main session) and legacy bf16 evaluations, both under the snapshot that wrote them
    ("iw_evals_fp32", "iw_mt2_lr1e3_"): "code", ("iw_evals", "iw_mt2_lr1e3_"): "code",
    ("hl_evals_fp32", "mt2_hl_lr1_"): "code", ("hl_evals", "mt2_hl_lr1_"): "code",
    ("hl_evals", "floor_hl_mt2_full_20k_"): "code", ("hl_evals_fp32", "floor_hl_mt2_full_20k_"): "code",
    # density probes: 10k probes of the SCALE arms under code_isla5 / code_perf; campaign E (transfer session) under code;
    # the first true-80k probe (job 700105) under code_perf_eval (corner) and code (references) -> the reference rows are STRUCK
    ("scale_probe_fp32", "scale_isla_dr_w512_"): "code_isla5", ("scale_probe_fp32", "scale_isla_dr_c512x80k_"): "code_perf",
    ("transfer/campaign_e_fp32", "iw_mt2_"): "code",
    ("scale_probe_fp32_80k", "scale_isla_dr_c512x80k_"): "code_perf_eval", ("scale_probe_fp32_80k", "iw_mt2_"): "code (STRUCK: legacy snapshot at 80,000 cells)",
    ("scale_probe_fp32_40k", "scale_isla_dr_c512x80k_"): "code_perf_eval", ("scale_probe_fp32_40k", "iw_mt2_"): "code_eval",
    ("scale_probe_fp32_20k", "iw_mt2_"): "code_eval", ("scale_probe_fp32_40kv80", "iw_mt2_"): "code_eval",
    ("scale_probe_fp32_60k", "iw_mt2_"): "code_eval", ("scale_probe_fp32_65k", "iw_mt2_"): "code_eval", ("scale_probe_fp32_70k", "iw_mt2_"): "code_eval",
    ("scale_probe_fp32_80kx", "iw_mt2_"): "code_eval", ("scale_probe_fp32_80kx", "scale_isla_dr_c512x80k_"): "code_eval",
}


def snapshot_of(root, run, outdir=None):
    """Name the evaluation snapshot of a run: sidecar first, then the pre-sidecar manifest; None if unknown."""
    import os
    for d in ([outdir] if outdir else []) + [f"{T}/{root}/{run}"]:
        sc = f"{d}/SNAPSHOT_USED"
        if os.path.exists(sc):
            return os.path.basename(open(sc).read().strip().rstrip("/"))
    for (r, prefix), snap in PRE_SIDECAR_SNAPSHOTS.items():
        if r == root and run.startswith(prefix):
            return snap
    return None


def _log_clean(root, run):
    """Refuse a run whose evaluation log records a skipped checkpoint load (the evaluation would be of the seeded init)."""
    for lg in glob.glob(f"{T}/{root}/{run}.log") + glob.glob(f"{T}/{root}/{run}/*.log"):
        txt = open(lg, errors="ignore").read()
        if "skipping load" in txt or "Could not find valid model file" in txt:
            raise AssertionError(f"{root}/{run}: evaluation log records a skipped checkpoint load ({lg}); number struck")
    return True


def _metrics_in(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/*/metrics.jsonl") or glob.glob(f"{T}/{root}/{run}/metrics.jsonl")
    if not ps:
        return None
    _log_clean(root, run)
    snap = snapshot_of(root, run)
    if snap is None:
        raise AssertionError(f"{root}/{run}: evaluation snapshot cannot be named (no SNAPSHOT_USED sidecar, not in the manifest); number refused")
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return {f: st.mean(r[f] for r in rows) for f in F if f in rows[0]} | {"n_cases": len(rows), "snapshot": snap}


def _points_identical(run):
    """Assert the fp32 and bf16 artifacts sampled the same points (bit-identical) per case."""
    import numpy as np, os
    a = glob.glob(f"{T}/hl_evals/{run}/*/predictions") + glob.glob(f"{T}/iw_evals/{run}/*/predictions")
    b = glob.glob(f"{T}/hl_evals_fp32/{run}/*/predictions") + glob.glob(f"{T}/iw_evals_fp32/{run}/*/predictions")
    if not a or not b:
        return None
    cases = sorted(set(os.listdir(a[0])) & set(os.listdir(b[0])))
    for cid in cases:
        pa = np.memmap(f"{a[0]}/{cid}/_tensordict/interior/_tensordict/points.memmap", dtype=np.float32, mode="r")
        pb = np.memmap(f"{b[0]}/{cid}/_tensordict/interior/_tensordict/points.memmap", dtype=np.float32, mode="r")
        if pa.shape != pb.shape or not np.array_equal(pa, pb):
            raise AssertionError(f"{run} {cid}: fp32 and bf16 artifacts sampled different points")
    return len(cases)


def metrics(run):
    """fp32 headline with bf16 alongside. Returns None if neither evaluation exists."""
    fp32 = _metrics_in("hl_evals_fp32", run) or _metrics_in("iw_evals_fp32", run)
    bf16 = _metrics_in("hl_evals", run) or _metrics_in("iw_evals", run)
    if fp32 is None and bf16 is None:
        return None
    head = dict(fp32 if fp32 else bf16)
    head["instrument"] = "float32" if fp32 else "bf16 (fp32 re-evaluation not yet available)"
    if fp32 and bf16:
        head["bf16"] = {f: bf16[f] for f in F if f in bf16} | {"snapshot": bf16["snapshot"]}  # carry the bf16 side's snapshot name (was dropped -> shown as "?")
        head["fp32_over_bf16_pressure"] = fp32["pressure_l2"] / bf16["pressure_l2"]
        head["points_identical_cases"] = _points_identical(run)
    return head


def training(run):
    try:
        txt = open(f"{T}/runs/{run}/train.log").read()
    except FileNotFoundError:
        return None
    params = re.findall(r"Parameters: ([\d,]+)", txt)
    steps = STEP.findall(txt)
    if not steps:
        return {"params": int(params[-1].replace(",", "")) if params else None}
    times = [float(s[4]) for s in steps]; mems = [float(s[5]) for s in steps]
    last_epoch = max(int(s[0]) for s in steps)
    nan_steps = sum(1 for s in steps if s[3] == "nan")
    return {"params": int(params[-1].replace(",", "")) if params else None, "peak_mem_gb": max(mems),
            "median_step_s": st.median(times), "gpu_hours": sum(times) * 4 / 3600, "logged_steps": len(steps),
            "last_epoch": last_epoch, "nan_steps": nan_steps, "completed": "Training completed" in txt}


out = {"runs": {}, "arms": {}}
for ds, arms in ARMS.items():
    for arm, runs in arms.items():
        per = []
        for r in runs:
            m, t = metrics(r), training(r)
            out["runs"][r] = {"metrics": m, "training": t}
            if m:
                per.append((m, t))
        if per:
            a = {f: st.mean(m[f] for m, _ in per) for f in F if all(f in m for m, _ in per)}
            a["n_seeds"] = len(per); a["seed_pressure"] = [m["pressure_l2"] for m, _ in per]
            a["instrument"] = sorted({m["instrument"] for m, _ in per})
            a["snapshot"] = sorted({m.get("snapshot", "?") for m, _ in per} | {m["bf16"].get("snapshot", "?") for m, _ in per if "bf16" in m})
            if all("bf16" in m for m, _ in per):
                a["bf16_pressure_l2"] = st.mean(m["bf16"]["pressure_l2"] for m, _ in per)
                a["fp32_over_bf16_pressure"] = st.mean(m["fp32_over_bf16_pressure"] for m, _ in per)
            for k in ("params", "peak_mem_gb", "median_step_s", "gpu_hours"):
                v = [t[k] for _, t in per if t and t.get(k) is not None]
                a[k] = st.mean(v) if v else None
            a["numerics"] = ("relative frame, eager geometry kernel (code_globin)" if arm in GLOBIN else
                             "fast point-softmax (code_perf)" if arm in FAST_KERNEL else "reference kernel (code_isla5 / reference lanes)")
            if arm == "w384nw":
                a["label"] = "ablation (discretization-dependent; never a reference configuration)"
            if arm == "r512x80k":
                a["label"] = ("DIVERGED at the protocol learning rate (both seeds; step loss blew up before epoch 25 and "
                              "plateaued at 0.056-0.060, the mean-field level, for the remaining epochs); the evaluated "
                              "checkpoint is a mean-field predictor; bars not graded (notebook 2026-09-15)")
            out["arms"][f"{ds}_{arm}"] = a
for ds in ARMS:
    ref = out["arms"].get(f"{ds}_ref")
    if ref:
        for arm in ARMS[ds]:
            a = out["arms"].get(f"{ds}_{arm}")
            if a:
                a["pressure_over_ref"] = a["pressure_l2"] / ref["pressure_l2"]
                own = OWN_REF.get(f"{ds}_{arm}")
                if own:
                    a["own_reference"] = own
                    a["pressure_over_own_ref"] = a["pressure_l2"] / own["pressure_l2"]
# Density-bias probe (float32; 10:1 biased sampling vs the uniform control, both at 10,000 cells): biased / uniform
# pressure error per arm. The reference arm's probe comes from the transfer session's campaign E; the SCALE arms from
# highlift/hl_scale_isla_probe_fp32_aga.sbatch.
PROBE = {"dr_ref": [f"{T}/transfer/campaign_e_fp32/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_c512x80k": [f"{T}/scale_probe_fp32/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)],
         "dr_w512": [f"{T}/scale_probe_fp32/scale_isla_dr_w512_seed{s}" for s in (42, 43)],
         # resolution-generalization test. "@40k": the first mirror run, which asked for 80,000 cells from probe datasets
         # whose reader pools only 40,000 (both samplers then keep the whole pool, so biased == uniform and the count is
         # 40,000; only the uniform column is meaningful). "@80k": the true 80,000-cell probe on the *_80k dataset variants
         # (160,000-cell pool). The similarity-gauge reference (iw_mt2_gauge, campaign E at 10k) is the principled control.
         "dr_ref@40k": [f"{T}/scale_probe_fp32_40k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_c512x80k@40k": [f"{T}/scale_probe_fp32_40k/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)],
         "dr_ref@80k": [f"{T}/scale_probe_fp32_80k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_c512x80k@80k": [f"{T}/scale_probe_fp32_80k/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)],
         "dr_gauge_ref": [f"{T}/transfer/campaign_e_fp32/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@40k": [f"{T}/scale_probe_fp32_40k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@80k": [f"{T}/scale_probe_fp32_80k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         # count-curve points for the two 10k-trained references: 20k (base 40k pool), 40k and 60k on the 320k-pool variant
         "dr_ref@20k": [f"{T}/scale_probe_fp32_20k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_ref@40kv80": [f"{T}/scale_probe_fp32_40kv80/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_ref@60k": [f"{T}/scale_probe_fp32_60k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@20k": [f"{T}/scale_probe_fp32_20k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@40kv80": [f"{T}/scale_probe_fp32_40kv80/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@60k": [f"{T}/scale_probe_fp32_60k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         # bracket of the 80k cliff (65,536 hypothesis) and the kernel swap at 80k
         "dr_ref@65k": [f"{T}/scale_probe_fp32_65k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_ref@70k": [f"{T}/scale_probe_fp32_70k/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@65k": [f"{T}/scale_probe_fp32_65k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@70k": [f"{T}/scale_probe_fp32_70k/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_ref@80k_fastkernel": [f"{T}/scale_probe_fp32_80kx/iw_mt2_lr1e3_seed{s}" for s in (42, 43)],
         "dr_c512x80k@80k_refkernel": [f"{T}/scale_probe_fp32_80kx/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)],
         "dr_gauge_ref@80k_codeeval": [f"{T}/scale_probe_fp32_80kx/iw_mt2_gauge_seed{s}" for s in (42, 43)],
         "dr_g512x80k": [f"{T}/scale_probe_fp32/scale_isla_dr_g512x80k_seed{s}" for s in (42, 43)],
         "dr_g512x80k@80k": [f"{T}/scale_probe_fp32_80k/scale_isla_dr_g512x80k_seed{s}" for s in (42, 43)]}
# HiLift probes: SAMPLE FRAME ONLY (program instruction 2026-09-10), highlift_probe_{biased3,unif2}_sf.yaml at 10,000
# cells for every HiLift arm and the reference, and the _sf_80k pair at the native count of the 40k/80k arms.
# Launcher highlift/hl_scale_isla_probe_hl_sf_fp32_aga.sbatch (code_eval, float32, sidecars).
PROBE["hl_ref"] = [f"{T}/scale_probe_hl_fp32_sf/mt2_hl_lr1_seed{s}" for s in (42, 43)]
for _arm in ("w384", "w512", "w384nw", "t40k", "t80k", "c384x40k", "c512x80k"):
    PROBE[f"hl_{_arm}"] = [f"{T}/scale_probe_hl_fp32_sf/scale_isla_hl_{_arm}_seed{s}" for s in (42, 43)]
for _arm, _cells in (("t40k", "40k"), ("c384x40k", "40k"), ("t80k", "80k"), ("c512x80k", "80k")):
    PROBE[f"hl_{_arm}@{_cells}"] = [f"{T}/scale_probe_hl_fp32_sf_native/scale_isla_hl_{_arm}_seed{s}" for s in (42, 43)]
# RELFRAME corner and its own reference: SAMPLE FRAME ONLY, under code_globin + recipe_globin
# (highlift/hl_scale_isla_globin_probe_sf_fp32_aga.sbatch). The 10k key doubles as the arm's density column.
PROBE["dr_r512x80k"] = [f"{T}/scale_probe_fp32_sf/scale_isla_dr_r512x80k_seed{s}" for s in (42, 43)]
PROBE["dr_r512x80k@80k_sf"] = [f"{T}/scale_probe_fp32_sf_80k/scale_isla_dr_r512x80k_seed{s}" for s in (42, 43)]
PROBE["dr_rf_ref@10k_sf"] = [f"{T}/scale_probe_fp32_sf/rf_dr_mt2_meas_seed{s}" for s in (42, 43)]


def _probe_metric(d):
    ps = glob.glob(f"{d}/*/metrics.jsonl") + glob.glob(f"{d}/*/*/metrics.jsonl")
    if not ps:
        return None
    import os
    root = os.path.relpath(os.path.dirname(os.path.dirname(d)), T); run = os.path.basename(os.path.dirname(d))
    snap = snapshot_of(root, run, outdir=d)
    if snap is None:
        raise AssertionError(f"{d}: probe snapshot cannot be named; number refused")
    _probe_metric.last_snapshot = snap
    for lg in glob.glob(f"{d}.log"):
        txt = open(lg, errors="ignore").read()
        if "skipping load" in txt or "Could not find valid model file" in txt:
            return None  # skipped checkpoint load: the probe evaluated the seeded initialization; struck
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"]["pressure_l2"] for r in rows if r.get("phase") == "infer_step"]
    return st.mean(rows) if rows else None


out["density_probe"] = {}
for arm, dirs in PROBE.items():
    per = []
    for d in dirs:
        u, b = _probe_metric(f"{d}/unif"), _probe_metric(f"{d}/biased")
        if u and b:
            per.append({"uniform": u, "biased": b, "biased_over_uniform": b / u, "snapshot": _probe_metric.last_snapshot})
    if per:
        out["density_probe"][arm] = {"n_seeds": len(per), "uniform": st.mean(x["uniform"] for x in per), "biased": st.mean(x["biased"] for x in per),
                                     "biased_over_uniform": st.mean(x["biased_over_uniform"] for x in per), "per_seed": per,
                                     "snapshot": sorted({x["snapshot"] for x in per})}
        if arm in out["arms"]:
            out["arms"][arm]["density_biased_over_uniform"] = out["density_probe"][arm]["biased_over_uniform"]
# Convergence curves (coordinator addition): for the 10k-trained reference and the 80k-trained models, the float32
# uniform-sampling error and the biased/uniform ratio at 10,000 and 80,000 evaluation cells. Error falling with count is
# fine; a sampling-distribution dependence that does not vanish with refinement is the failure the redirect names.
out["convergence"] = {}
# STRUCK: the 80,000-cell evaluations of the two legacy references run under the legacy code snapshot (job 700105) collapsed
# (0.814 / 0.892) while the same checkpoints under code_eval are healthy at 65k, 70k and 80k; the cause is that snapshot's
# evaluation path at 80,000 cells (not the softmax kernel: micro-test, CPU control and kernel swap all clean). Kept in the
# probe block with this label; excluded from the convergence curves, whose 80k point is the code_eval evaluation.
STRUCK = {"dr_ref@80k": "legacy code snapshot at 80,000 cells; struck (see notebook)",
          "dr_gauge_ref@80k": "legacy code snapshot at 80,000 cells; struck (see notebook)"}
for k, why in STRUCK.items():
    if k in out["density_probe"]:
        out["density_probe"][k]["struck"] = why
CELLS = (10000, 20000, 40000, "40000v80", 60000, 65000, 70000, 80000, "80000_kernelswap")
for model, keys in (("reference_10k_trained", ("dr_ref", "dr_ref@20k", "dr_ref@40k", "dr_ref@40kv80", "dr_ref@60k", "dr_ref@65k", "dr_ref@70k", "dr_ref@80k_fastkernel", None)),
                    ("c512x80k_80k_trained", ("dr_c512x80k", None, "dr_c512x80k@40k", None, None, None, None, "dr_c512x80k@80k", "dr_c512x80k@80k_refkernel")),
                    ("gauge_reference_10k_trained", ("dr_gauge_ref", "dr_gauge_ref@20k", "dr_gauge_ref@40k", "dr_gauge_ref@40kv80", "dr_gauge_ref@60k", "dr_gauge_ref@65k", "dr_gauge_ref@70k", "dr_gauge_ref@80k_codeeval", None)),
                    ("g512x80k_80k_trained_similarity_gauge", ("dr_g512x80k", None, None, None, None, None, None, "dr_g512x80k@80k", None))):
    curve = {}
    for cells, key in zip(CELLS, keys):
        if key is None:
            continue
        pr = out["density_probe"].get(key)
        if pr:
            curve[str(cells)] = {"uniform": pr["uniform"], "biased": pr["biased"], "biased_over_uniform": pr["biased_over_uniform"]}
            if cells == 40000:
                curve[str(cells)]["note"] = "40,000-cell pool exhausted: biased == uniform; only the uniform column is meaningful"
            if cells == "40000v80":
                curve[str(cells)]["note"] = "40,000 cells drawn from the 320,000-cell pool variant (dataset-variant control for the 40k point)"
    if curve:
        out["convergence"][model] = curve
# Sample-frame density probe (coordinator task, 2026-09-10): the canonical sample-frame datasets
# drivaer_probe_{biased3,unif2}_sf(_80k).yaml move CenterMesh to directly after the sampler, so a model that consumes
# absolute centred coordinates sees the drawn sample's mean rather than the pool's. ISLA centres internally, so its ratio is
# expected to equal the pool-frame ratio; each entry reports "sample frame (pool frame)". Launcher:
# highlift/hl_scale_isla_probe_sf_fp32_aga.sbatch (code_eval, float32, sidecars). Keys: <arm>@<cells>_sf -> pool-frame key.
PROBE_SF = {"dr_c512x80k@80k_sf": ([f"{T}/scale_probe_fp32_sf_80k/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)], "dr_c512x80k@80k"),
            "dr_g512x80k@80k_sf": ([f"{T}/scale_probe_fp32_sf_80k/scale_isla_dr_g512x80k_seed{s}" for s in (42, 43)], "dr_g512x80k@80k"),
            "dr_c512x80k@10k_sf": ([f"{T}/scale_probe_fp32_sf/scale_isla_dr_c512x80k_seed{s}" for s in (42, 43)], "dr_c512x80k"),
            "dr_g512x80k@10k_sf": ([f"{T}/scale_probe_fp32_sf/scale_isla_dr_g512x80k_seed{s}" for s in (42, 43)], "dr_g512x80k"),
            "dr_ref@10k_sf": ([f"{T}/scale_probe_fp32_sf/iw_mt2_lr1e3_seed{s}" for s in (42, 43)], "dr_ref"),
            "dr_gauge_ref@10k_sf": ([f"{T}/scale_probe_fp32_sf/iw_mt2_gauge_seed{s}" for s in (42, 43)], "dr_gauge_ref")}
out["density_probe_sample_frame"] = {}
for key, (dirs, pool_key) in PROBE_SF.items():
    per = []
    for d in dirs:
        u, b = _probe_metric(f"{d}/unif"), _probe_metric(f"{d}/biased")
        if u and b:
            per.append({"uniform": u, "biased": b, "biased_over_uniform": b / u, "snapshot": _probe_metric.last_snapshot})
    if per:
        pool = out["density_probe"].get(pool_key)
        e = {"n_seeds": len(per), "uniform": st.mean(x["uniform"] for x in per), "biased": st.mean(x["biased"] for x in per),
             "biased_over_uniform": st.mean(x["biased_over_uniform"] for x in per), "per_seed": per,
             "snapshot": sorted({x["snapshot"] for x in per}), "pool_frame_key": pool_key}
        if pool:
            e["pool_frame"] = {k: pool[k] for k in ("uniform", "biased", "biased_over_uniform")}
            e["sample_frame_over_pool_frame_ratio"] = e["biased_over_uniform"] / pool["biased_over_uniform"]
            e["report"] = f"{e['biased_over_uniform']:.2f}x ({pool['biased_over_uniform']:.2f}x)"
        out["density_probe_sample_frame"][key] = e
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out["density_probe_sample_frame"].items():
    print("probe_sf", k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk not in ("per_seed", "pool_frame")})
for k, v in out["convergence"].items():
    print("convergence", k, {c: {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in d.items()} for c, d in v.items()})
for k, v in out["density_probe"].items():
    print("probe", k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "per_seed"})
for k, v in out["arms"].items():
    print(k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "seed_pressure"})
