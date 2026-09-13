"""Shared harness for the ISLA kernel study (2026-09-13).

Reference configuration = isla_surface_reference.yaml: hidden 192, 12 layers, 256
slices, 1 scalar + 1 vector output, frame_mode=relative, scale_mode=total_measure,
geo_checkpoint=True, fast_point_softmax=True. Synthetic inputs as in the earlier cost
scripts (positions ~ N(0,1) * (3,2,1), unit normals, measure weights U(0.5,1.5), unit
drive). One training step = zero_grad, forward (optionally under bf16 autocast), MSE
against a random target, backward, AdamW step.

Instruments: cuda-synchronized wall time per step (median over N_MEAS after N_WARM
warm-ups), peak allocated memory over the measured steps (total and incremental over
the resident model + optimizer state), memory_reserved after the step, and a
load snapshot (nvidia-smi utilization/memory, /proc/loadavg) taken right before.
"""
import gc
import json
import os
import statistics
import subprocess
import time

import torch

from physicsnemo.experimental.nn.isla import ISLA

GIB = 2**30
REF_KW = dict(out_scalars=1, out_vectors=1, hidden=192, n_layers=12, n_slices=256, mlp_ratio=4,
              frame_mode="relative", scale_mode="total_measure", geo_checkpoint=True, fast_point_softmax=True)


def load_snapshot():
    snap = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        u, mu, mt, cs, cm, t, p = [x.strip() for x in q.split(",")]
        snap.update(gpu_util_pct=float(u), gpu_mem_used_mib=float(mu), gpu_mem_total_mib=float(mt), sm_clock_mhz=float(cs),
                    mem_clock_mhz=float(cm), gpu_temp_c=float(t), power_w=float(p))
    except Exception as e:  # noqa: BLE001
        snap["nvidia_smi_error"] = repr(e)
    try:
        snap["loadavg"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except Exception:  # noqa: BLE001
        pass
    return snap


def make_inputs(n, batch=1, seed=0, device="cuda", dtype=torch.float32, out_dim=4):
    g = torch.Generator().manual_seed(seed)
    pts = (torch.randn(batch, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])).to(dtype)
    nrm = torch.nn.functional.normalize(torch.randn(batch, n, 3, generator=g), dim=-1).to(dtype)
    w = (torch.rand(batch, n, generator=g) + 0.5).to(dtype)
    drv = torch.nn.functional.normalize(torch.randn(batch, 3, generator=g), dim=-1).to(dtype)
    target = torch.randn(batch, n, out_dim, generator=g).to(dtype)
    return dict(points=pts.to(device), normals=nrm.to(device), drive=drv.to(device), measure_weights=w.to(device)), target.to(device)


def make_model(seed=0, device="cuda", **overrides):
    torch.manual_seed(seed)
    kw = dict(REF_KW, **overrides)
    return ISLA(**kw).to(device), kw


def train_step_fn(model, inputs, target, opt, autocast):
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model(**inputs)
        loss = torch.nn.functional.mse_loss(out.float(), target)
        loss.backward()
        opt.step()
        return loss
    return step


def measure(step, n_warm=3, n_meas=10):
    for _ in range(n_warm):
        step()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_meas):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated()
    return dict(step_ms=times, step_ms_median=statistics.median(times), step_ms_min=min(times),
                resident_bytes=resident, peak_allocated_bytes=peak, incremental_peak_bytes=peak - resident,
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_gib=peak / GIB, incremental_peak_gib=(peak - resident) / GIB)


def run_config(build, n, batch, autocast, n_warm=3, n_meas=10, lr=1e-4, seed=0):
    """build() -> model. Returns a record; OOM is recorded, not raised."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    rec = dict(tokens=n, batch=batch, precision="bf16_autocast" if autocast else "fp32", load=load_snapshot())
    try:
        model = build()
        inputs, target = make_inputs(n, batch, seed=seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        rec["n_params"] = sum(p.numel() for p in model.parameters())
        rec.update(measure(train_step_fn(model, inputs, target, opt, autocast), n_warm, n_meas))
        rec["status"] = "ok"
        del model, opt, inputs, target
    except torch.OutOfMemoryError as e:
        rec["status"] = "oom"; rec["error"] = str(e)[:300]
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["status"] = "error"; rec["error"] = traceback.format_exc()[-2000:]
    gc.collect(); torch.cuda.empty_cache()
    return rec


def env_info():
    return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                device_props={k: getattr(torch.cuda.get_device_properties(0), k) for k in ("multi_processor_count", "total_memory", "major", "minor")})


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
