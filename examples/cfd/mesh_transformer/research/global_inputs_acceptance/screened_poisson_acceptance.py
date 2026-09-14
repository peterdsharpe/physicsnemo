# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLOBIN-ACCEPT: ISLA on a steady boundary-value problem with NO global
direction and ONE global scalar parameter (preregistration: the lab notebook,
#sec-nb-globin-accept-prereg, 2026-09-14).

Problem family. The screened Poisson (modified Helmholtz) equation
``lap(u) - kappa^2 u = 0`` inside a random star-shaped closed surface in 3D,
with Dirichlet data u on the boundary and the parameter kappa in [0, 3]
(kappa = 0 is Laplace's equation: steady conduction with pinned boundary
temperatures). Exact solutions are superpositions of the fundamental solution
``exp(-kappa |x - y|) / |x - y|`` with sources y placed OUTSIDE the domain, so
u is smooth inside and the interior labels are exact. Inputs to the model are
the boundary sample (positions, unit normals, Horvitz-Thompson area weights),
the Dirichlet trace as the per-cell boundary scalar, the scalar kappa as the
one global scalar input, and NO global vector input (``n_global_vectors=0``).
Output: u at interior query points (query tokens; the query normal is the
level-set gradient of the surface function, defined everywhere but the origin).

Arms. (A) ISLA with the scalar input; (B) the same model with the scalar
withheld (``n_global_scalars=0``: it must infer the screening from the trace,
which is not identifiable in general); trivial predictors (boundary mean of
the trace; nearest boundary value). Two seeds each. Test sets: T0 fresh
in-range cases; T1 kappa in [3, 4] (parameter extrapolation, reported, not
part of the acceptance bar).

Run:  python screened_poisson_acceptance.py --steps 3000 --out <json>

Second family (``--family sphere_bessel``, the conditioning DISCRIMINATOR of
#sec-nb-globin-discrim-prereg): the same equation inside a sphere of radius
R in [0.8, 1.2] with the Dirichlet trace g = sum_k a_k Y_k(d) prescribed
directly (real harmonics up to order three). The exact interior solution is
u = sum_k a_k [i_l(kappa r) / i_l(kappa R)] Y_k(d) with i_l the modified
spherical Bessel function of the first kind, so the trace carries NO
information about kappa while the interior depends on it strongly (kappa in
[0, 6]); a model without the scalar input can at best predict the
kappa-averaged interior.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import spherical_in

from physicsnemo.experimental.nn.isla import ISLA

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LMAX = 3  # real spherical-harmonic perturbation order of the surface


# ----------------------------------------------------------------------------- geometry
def _real_sh_basis(d: torch.Tensor) -> torch.Tensor:
    """Low-order real spherical harmonics (l = 1..3, unnormalized polynomial
    forms) of unit directions d (..., 3) -> (..., 15)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    l1 = [x, y, z]
    l2 = [x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1]
    l3 = [x * (x * x - 3 * y * y), y * (3 * x * x - y * y), z * (x * x - y * y), x * y * z,
          x * (5 * z * z - 1), y * (5 * z * z - 1), z * (5 * z * z - 3)]
    return torch.stack(l1 + l2 + l3, dim=-1)


def surface_radius(coef: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """rho(d) = 1 + sum_k coef_k Y_k(d); coef (B, 15), d (B, N, 3) -> (B, N)."""
    return 1.0 + (_real_sh_basis(d) * coef[:, None, :]).sum(-1)


def level_set_gradient(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Unit gradient of f(x) = |x| - rho(x/|x|) at x (B, N, 3): the outward unit
    normal on the surface and a smooth normal field off it."""
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        r = xx.norm(dim=-1)
        f = r - surface_radius(coef, xx / r[..., None])
        (g,) = torch.autograd.grad(f.sum(), xx)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sample_directions(b: int, n: int, gen: torch.Generator) -> torch.Tensor:
    d = torch.randn(b, n, 3, generator=gen, device=DEV)
    return d / d.norm(dim=-1, keepdim=True)


def make_cases(b: int, n_boundary: int, n_query: int, kappa_range: tuple[float, float],
               gen: torch.Generator, n_sources: int = 6):
    """One batch of exact cases. Returns a dict of tensors on DEV."""
    coef = (torch.rand(b, 15, generator=gen, device=DEV) * 2 - 1) * 0.12
    coef[:, 3:8] *= 0.7
    coef[:, 8:] *= 0.4
    ### boundary sample: directions uniform on the sphere; x = rho d; the area
    ### measure relative to solid angle is rho^2 / (d . n), so the Horvitz-
    ### Thompson weight of each of the N draws is 4 pi rho^2 / (d . n) / N.
    d = sample_directions(b, n_boundary, gen)
    rho = surface_radius(coef, d)
    x = rho[..., None] * d
    nrm = level_set_gradient(coef, x)
    cos = (d * nrm).sum(-1).clamp_min(0.2)
    w = 4 * math.pi * rho.square() / cos / n_boundary
    ### interior queries: x = s rho(d) d with s ~ U^(1/3) (volume-uniform in the
    ### star-shaped body), s <= 0.95 to stay off the boundary
    dq = sample_directions(b, n_query, gen)
    s = torch.rand(b, n_query, generator=gen, device=DEV).pow(1 / 3) * 0.95
    xq = (s * surface_radius(coef, dq))[..., None] * dq
    nq = level_set_gradient(coef, xq)
    ### sources outside the body (rho <= ~1.4): radius in [1.8, 2.8]
    ds = sample_directions(b, n_sources, gen)
    rs = 1.8 + torch.rand(b, n_sources, 1, generator=gen, device=DEV)
    ys = ds * rs
    cs = torch.randn(b, n_sources, generator=gen, device=DEV)
    kappa = kappa_range[0] + (kappa_range[1] - kappa_range[0]) * torch.rand(b, generator=gen, device=DEV)

    def u_at(p):  # p (B, M, 3) -> (B, M)
        dist = (p[:, :, None, :] - ys[:, None, :, :]).norm(dim=-1)  # (B, M, S)
        return (cs[:, None, :] * torch.exp(-kappa[:, None, None] * dist) / dist).sum(-1)

    ub, uq = u_at(x), u_at(xq)
    ### the PDE is linear: normalize each case by the boundary RMS of the trace
    scale = ub.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return dict(points=x, normals=nrm, weights=w, trace=(ub / scale)[..., None], kappa=kappa[:, None],
                query_points=xq, query_normals=nq, target=(uq / scale)[..., None])


_L_OF_BASIS = [0] + [1] * 3 + [2] * 5 + [3] * 7  # harmonic degree of [1, Y_1.., Y_2.., Y_3..]


def _bessel_ratio(l: int, kappa: torch.Tensor, r: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """i_l(kappa r) / i_l(kappa R) for kappa (B,), r (B, M), R (B,); (r/R)^l at kappa = 0."""
    k = kappa[:, None].expand_as(r)
    kr = (k * r).double().cpu().numpy(); kR = (k * R[:, None].expand_as(r)).double().cpu().numpy()
    with np.errstate(all="ignore"):
        num = spherical_in(l, kr); den = spherical_in(l, kR)
        ratio = np.where(kR > 1e-6, num / np.where(den == 0, 1.0, den), (r / R[:, None]).double().cpu().numpy() ** l)
    return torch.as_tensor(ratio, dtype=r.dtype, device=r.device)


def make_cases_sphere(b: int, n_boundary: int, n_query: int, kappa_range: tuple[float, float], gen: torch.Generator):
    """The discriminator family: sphere, prescribed trace, exact Bessel interior."""
    R = 0.8 + 0.4 * torch.rand(b, generator=gen, device=DEV)
    d = sample_directions(b, n_boundary, gen)
    x = R[:, None, None] * d
    w = (4 * math.pi * R.square() / n_boundary)[:, None].expand(b, n_boundary)
    dq = sample_directions(b, n_query, gen)
    sq = torch.rand(b, n_query, generator=gen, device=DEV).pow(1 / 3) * 0.95
    xq = (sq * R[:, None])[..., None] * dq
    ### trace coefficients, decaying with degree; the constant term included
    a = torch.randn(b, 16, generator=gen, device=DEV) / torch.tensor([1.0 + l for l in _L_OF_BASIS], device=DEV)
    kappa = kappa_range[0] + (kappa_range[1] - kappa_range[0]) * torch.rand(b, generator=gen, device=DEV)
    Yb = torch.cat([torch.ones(b, n_boundary, 1, device=DEV), _real_sh_basis(d)], dim=-1)  # (B, N, 16)
    Yq = torch.cat([torch.ones(b, n_query, 1, device=DEV), _real_sh_basis(dq)], dim=-1)
    ub = (a[:, None, :] * Yb).sum(-1)
    rq = xq.norm(dim=-1)
    ratios = torch.stack([_bessel_ratio(l, kappa, rq, R) for l in range(4)], dim=-1)  # (B, M, 4)
    uq = (a[:, None, :] * Yq * ratios[..., _L_OF_BASIS]).sum(-1)
    scale = ub.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return dict(points=x, normals=d, weights=w, trace=(ub / scale)[..., None], kappa=kappa[:, None],
                query_points=xq, query_normals=dq, target=(uq / scale)[..., None])


# ----------------------------------------------------------------------------- model / arms
def build(arm: str, seed: int) -> ISLA:
    torch.manual_seed(seed)
    return ISLA(
        out_scalars=1, out_vectors=0, hidden=128, n_layers=6, n_slices=64, mlp_ratio=4,
        n_global_vectors=0, n_global_scalars=1 if arm == "isla_scalar" else 0, n_boundary_scalars=1,
        query_tokens=True, query_mass="source_total",
    ).to(DEV)


def forward(m: ISLA, c: dict, arm: str) -> torch.Tensor:
    kw = dict(points=c["points"], normals=c["normals"], measure_weights=c["weights"], boundary_scalars=c["trace"],
              query_points=c["query_points"], query_normals=c["query_normals"])
    if arm == "isla_scalar":
        kw["global_scalars"] = c["kappa"]
    return m(**kw)


def rel_l2(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Per-case relative L2 over the queries -> (B,)."""
    return ((pred - tgt).square().sum((1, 2)) / tgt.square().sum((1, 2)).clamp_min(1e-12)).sqrt()


def trivial_predictors(c: dict) -> dict:
    tr = c["trace"][..., 0]
    mean = (c["weights"] * tr).sum(-1, keepdim=True) / c["weights"].sum(-1, keepdim=True)
    mean_pred = mean[..., None].expand_as(c["target"])
    d2 = torch.cdist(c["query_points"], c["points"])
    idx = d2.argmin(-1)
    nearest = torch.gather(tr, 1, idx)[..., None]
    return {"boundary_mean": rel_l2(mean_pred, c["target"]), "nearest_boundary_value": rel_l2(nearest, c["target"])}


# ----------------------------------------------------------------------------- run
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-boundary", type=int, default=1024)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=["isla_scalar", "isla_noscalar"])
    ap.add_argument("--eval-cases", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", choices=["sources", "sphere_bessel"], default="sources")
    ap.add_argument("--kappa-max", type=float, default=None, help="training/T0 kappa range upper end (default 3, or 6 for sphere_bessel)")
    args = ap.parse_args()
    kmax = args.kappa_max if args.kappa_max is not None else (6.0 if args.family == "sphere_bessel" else 3.0)
    cases = make_cases_sphere if args.family == "sphere_bessel" else make_cases

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gen_eval = torch.Generator(device=DEV).manual_seed(12345)
    tests = {
        "T0_in_range": cases(args.eval_cases, args.n_boundary, args.n_query, (0.0, kmax), gen_eval),
        "T1_kappa_3_4": cases(args.eval_cases, args.n_boundary, args.n_query, (kmax, kmax + 1.0), gen_eval),
    }
    out = {"commit": commit, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
           "config": vars(args) | {"out": str(args.out), "kappa_max": kmax}, "trivial": {}, "arms": {}}
    for name, c in tests.items():
        out["trivial"][name] = {k: {"mean": v.mean().item(), "median": v.median().item()} for k, v in trivial_predictors(c).items()}
    print("trivial:", json.dumps(out["trivial"]), flush=True)

    for arm in args.arms:
        out["arms"][arm] = {}
        for seed in args.seeds:
            m = build(arm, seed)
            n_params = sum(p.numel() for p in m.parameters())
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.steps, pct_start=0.05)
            gen = torch.Generator(device=DEV).manual_seed(1000 + seed)
            t0 = time.time(); losses = []
            for step in range(args.steps):
                c = cases(args.batch, args.n_boundary, args.n_query, (0.0, kmax), gen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                    pred = forward(m, c, arm)
                loss = rel_l2(pred.float(), c["target"]).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); losses.append(loss.item())
                if step % 200 == 0 or step == args.steps - 1:
                    print(f"[{arm} seed {seed}] step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                          f"({time.time() - t0:.0f} s)", flush=True)
            m.eval(); res = {"params": n_params, "train_seconds": time.time() - t0, "final_train_loss": sum(losses[-100:]) / 100}
            with torch.no_grad():
                for name, c in tests.items():
                    r = rel_l2(forward(m, c, arm).float(), c["target"])  # float32 evaluation
                    res[name] = {"mean": r.mean().item(), "median": r.median().item(), "p90": r.quantile(0.9).item()}
            out["arms"][arm][f"seed{seed}"] = res
            print(f"[{arm} seed {seed}] {json.dumps({k: v for k, v in res.items() if k.startswith('T')})}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=2))
            del m, opt; torch.cuda.empty_cache()
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
