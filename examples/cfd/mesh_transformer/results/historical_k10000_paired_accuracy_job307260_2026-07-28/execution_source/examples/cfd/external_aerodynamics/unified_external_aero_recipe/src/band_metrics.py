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

"""Spatial-band (per-region) surface metrics over saved infer.py predictions.

The all-boundary (H-ground) verdict needs more than a global aggregate: the
hypothesized effects concentrate where ground/boundary physics lives
(external review item 8; pre-registration in the lab notebook's allbc-launch
entry).  This tool partitions each case's vehicle surface into
physics-motivated regions and reports the corrected (direction-sensitive)
relative-L2 metrics per region, plus a per-region pseudo-force decomposition.

VECTOR-METRIC SEMANTICS (discovered validating this tool): the recipe's
``_relative_l2`` on an (N, 3) vector reduces over ``dim=-1`` -- the fleet
tables' ``wss_l2`` is therefore the POINTWISE-MEAN relative error
(mean over points of per-point ||d_tau||/||tau||), not the Frobenius joint
L2.  Both are direction-sensitive; they weight points differently.  This
tool reports BOTH: ``wss_l2_pw`` (comparable to every fleet/infer table)
and ``wss_l2_frob`` (whose squared sums decompose exactly across regions).
Each reconciles with its own global exactly: pointwise via cell-count
weighting, Frobenius via summed squares.

Inputs are the native prediction artifacts infer.py writes
(``<output_dir>/<run_id>/predictions/*.pdmsh``): DomainMeshes whose
``interior`` carries ``pred_/true_{pressure,wss}`` at the 10k query
centroids (physical units) and whose ``vehicle`` boundary is the matching
subsampled surface (cells align 1:1 with queries; verified ~3e-8 here).

REGION DEFINITIONS (deterministic; per-case TRUE geometry only, so every
model is scored on the identical partition — fairness by construction).
Coordinates are the artifact's centered, L_ref-nondimensionalized frame.
The streamwise coordinate is the projection onto ``U_inf_dir`` (robust to
the pipeline's z-rotations); the vertical axis is z (DrivAerML/SHIFT are
z-up; a sanity assert below verifies the vertical extent is the smallest
of the two cross-flow extents).  With per-case fractions
``x_frac`` (0 = nose = upstream extreme, 1 = base = downstream extreme)
and ``z_frac`` (0 = ground side), the disjoint, exhaustive regions are,
in priority order:

  underbody_front : z_frac < 0.18 and x_frac < 1/3
  underbody_mid   : z_frac < 0.18 and 1/3 <= x_frac < 2/3
  underbody_rear  : z_frac < 0.18 and x_frac >= 2/3
  rear_face       : z_frac >= 0.18 and x_frac >= 0.88
  upper_body      : remainder

Thresholds (documented, chosen from DrivAerML geometry before any
three-way numbers existed): 0.18 of body height ~= the bottom ~24 cm of a
~1.33 m body — underbody, rockers, wheel bottoms (the ground-effect
region); 0.88 of body length ~= the last ~60 cm — the base region that
anchors the wake.  The same constants apply to every case and model.

EXACTNESS CONTRACT: per case, the region-wise squared-error and
squared-target sums must reconcile with the global sums to roundoff
(asserted), so the global relative L2 is exactly recoverable from the
per-region pieces and nothing is dropped or double-counted.

Pseudo-force decomposition (same caveat as the book's pseudo-coefficients:
integrated on the 10k-cell subsample): per region, the streamwise
pressure force  F_p = sum p * (-n_x) * A  and friction force
F_f = sum tau_x * A, for pred and true, in (physical Pa) x (nondim area)
units — valid for pred-vs-true comparison, not as absolute CD.

Usage:
  python band_metrics.py PRED_DIR [PRED_DIR ...] [--out results.json]
where each PRED_DIR is a run's ``predictions/`` directory (or its parent).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from physicsnemo.mesh import DomainMesh

Z_FRAC_UNDERBODY = 0.18
X_FRAC_THIRDS = (1.0 / 3.0, 2.0 / 3.0)
X_FRAC_REAR_FACE = 0.88
REGIONS = (
    "underbody_front",
    "underbody_mid",
    "underbody_rear",
    "rear_face",
    "upper_body",
)
RECONCILE_RTOL = 1e-5


def _region_ids(
    queries: torch.Tensor, u_dir: torch.Tensor
) -> torch.Tensor:
    """Assign each query point a region id (index into REGIONS)."""
    s = queries @ (u_dir / u_dir.norm())
    x_frac = (s - s.min()) / (s.max() - s.min())
    z = queries[:, 2]
    z_frac = (z - z.min()) / (z.max() - z.min())

    rid = torch.full((queries.shape[0],), REGIONS.index("upper_body"))
    under = z_frac < Z_FRAC_UNDERBODY
    rid[under & (x_frac < X_FRAC_THIRDS[0])] = REGIONS.index("underbody_front")
    rid[
        under & (x_frac >= X_FRAC_THIRDS[0]) & (x_frac < X_FRAC_THIRDS[1])
    ] = REGIONS.index("underbody_mid")
    rid[under & (x_frac >= X_FRAC_THIRDS[1])] = REGIONS.index("underbody_rear")
    rid[~under & (x_frac >= X_FRAC_REAR_FACE)] = REGIONS.index("rear_face")
    return rid


def _case_metrics(path: Path) -> dict:
    m = DomainMesh.load(str(path))
    inter = m.interior
    veh = m.boundaries["vehicle"]
    q = inter.points
    g = veh.global_data
    u_dir = g["U_inf_dir"].to(q.dtype)

    # Sanity: z-up assumption -- vertical extent is the smaller cross-flow
    # extent for a car-like body.
    ext = q.max(0).values - q.min(0).values
    cross = [i for i in range(3) if abs(float(u_dir[i])) < 0.9]
    if len(cross) == 2 and not float(ext[2]) <= float(ext[cross[0]]) + 1e-6:
        raise AssertionError(
            f"z-up sanity failed for {path.name}: extents {ext.tolist()}, "
            f"U_inf_dir {u_dir.tolist()}"
        )

    # Geometry for forces: verify centroid<->query alignment, then areas
    # and normals per cell (== per query).
    verts = veh.points[veh.cells]
    centroids = verts.mean(dim=1)
    align = (centroids - q).abs().max()
    if float(align) > 1e-4:
        raise AssertionError(
            f"centroid/query misalignment {float(align):.3e} in {path.name}"
        )
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    area = 0.5 * torch.linalg.cross(e1, e2).norm(dim=-1)
    normals = veh.cell_data["normals"]

    rid = _region_ids(q, u_dir)
    pd = inter.point_data
    dp = pd["pred_pressure"] - pd["true_pressure"]
    dw = pd["pred_wss"] - pd["true_wss"]
    u_hat = u_dir / u_dir.norm()
    # Pointwise per-point relative errors (the fleet-table statistic).
    eps = 1e-8
    pw = dw.norm(dim=-1) / (pd["true_wss"].norm(dim=-1) + eps)

    out: dict = {"case": path.name, "regions": {}}
    tot = {
        "p_num2": float((dp**2).sum()),
        "p_den2": float((pd["true_pressure"] ** 2).sum()),
        "w_num2": float((dw**2).sum()),
        "w_den2": float((pd["true_wss"] ** 2).sum()),
        "pw_sum": float(pw.sum()),
    }
    acc = {k: 0.0 for k in tot}
    for i, name in enumerate(REGIONS):
        sel = rid == i
        n = int(sel.sum())
        p_num2 = float((dp[sel] ** 2).sum())
        p_den2 = float((pd["true_pressure"][sel] ** 2).sum())
        w_num2 = float((dw[sel] ** 2).sum())
        w_den2 = float((pd["true_wss"][sel] ** 2).sum())
        pw_sum = float(pw[sel].sum())
        for k, v in zip(
            ("p_num2", "p_den2", "w_num2", "w_den2", "pw_sum"),
            (p_num2, p_den2, w_num2, w_den2, pw_sum),
        ):
            acc[k] += v
        # Streamwise pseudo-forces on the subsample (see module docstring).
        nx = normals[sel] @ u_hat
        a = area[sel]
        f = {
            "fp_pred": float((pd["pred_pressure"][sel] * -nx * a).sum()),
            "fp_true": float((pd["true_pressure"][sel] * -nx * a).sum()),
            "ff_pred": float(((pd["pred_wss"][sel] @ u_hat) * a).sum()),
            "ff_true": float(((pd["true_wss"][sel] @ u_hat) * a).sum()),
        }
        out["regions"][name] = {
            "n_cells": n,
            "pressure_l2": math.sqrt(p_num2 / p_den2) if p_den2 > 0 else None,
            "wss_l2_pw": pw_sum / n if n else None,
            "wss_l2_frob": math.sqrt(w_num2 / w_den2) if w_den2 > 0 else None,
            "p_num2": p_num2,
            "p_den2": p_den2,
            "w_num2": w_num2,
            "w_den2": w_den2,
            **f,
        }
    # Exactness contract: regions reconcile with global to roundoff.
    for k in tot:
        if not math.isclose(acc[k], tot[k], rel_tol=RECONCILE_RTOL):
            raise AssertionError(
                f"reconciliation failed for {k} in {path.name}: "
                f"sum(regions)={acc[k]!r} vs global={tot[k]!r}"
            )
    out["global"] = {
        "pressure_l2": math.sqrt(tot["p_num2"] / tot["p_den2"]),
        "wss_l2_pw": tot["pw_sum"] / q.shape[0],
        "wss_l2_frob": math.sqrt(tot["w_num2"] / tot["w_den2"]),
    }
    return out


def run_dir(pred_dir: Path) -> dict:
    if (pred_dir / "predictions").is_dir():
        pred_dir = pred_dir / "predictions"
    cases = sorted(pred_dir.glob("*.pdmsh"))
    if not cases:
        raise FileNotFoundError(f"no .pdmsh predictions under {pred_dir}")
    per_case = [_case_metrics(p) for p in cases]

    summary: dict = {"n_cases": len(per_case), "regions": {}, "global": {}}
    for field in ("pressure_l2", "wss_l2_pw", "wss_l2_frob"):
        summary["global"][field] = sum(
            c["global"][field] for c in per_case
        ) / len(per_case)
    for name in REGIONS:
        vals = {
            f: [c["regions"][name][f] for c in per_case
                if c["regions"][name][f] is not None]
            for f in ("pressure_l2", "wss_l2_pw", "wss_l2_frob")
        }
        forces = {
            f: sum(c["regions"][name][f] for c in per_case)
            for f in ("fp_pred", "fp_true", "ff_pred", "ff_true")
        }
        summary["regions"][name] = {
            "mean_n_cells": sum(c["regions"][name]["n_cells"] for c in per_case)
            / len(per_case),
            **{
                f: (sum(v) / len(v) if v else None)
                for f, v in vals.items()
            },
            **forces,
        }
    return {"summary": summary, "per_case": per_case}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pred_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    for d in args.pred_dirs:
        label = d.name if d.name != "predictions" else d.parent.name
        r = run_dir(d)
        results[label] = r
        s = r["summary"]
        print(f"== {label}  ({s['n_cases']} cases)")
        print(
            f"   global: pressure_l2={s['global']['pressure_l2']:.6f} "
            f"wss_l2_pw={s['global']['wss_l2_pw']:.6f} "
            f"wss_l2_frob={s['global']['wss_l2_frob']:.6f}"
        )
        for name in REGIONS:
            rr = s["regions"][name]
            print(
                f"   {name:16s} n~{rr['mean_n_cells']:7.1f} "
                f"p_l2={rr['pressure_l2']:.4f} "
                f"wss_pw={rr['wss_l2_pw']:.4f} "
                f"wss_frob={rr['wss_l2_frob']:.4f} "
                f"Fp(pred/true)={rr['fp_pred']:.4g}/{rr['fp_true']:.4g} "
                f"Ff={rr['ff_pred']:.4g}/{rr['ff_true']:.4g}"
            )
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
