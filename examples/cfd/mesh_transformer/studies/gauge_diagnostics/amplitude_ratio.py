# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The signed replacement for the retired Q2 ratio.

Q2 was ``pressure_l2(40k) / pressure_l2(10k)``, and it was gated once
before anyone checked what it measures.  It measures two *opposite*
pathologies on one axis: a prediction that collapses to zero scores
:math:`\\approx 1` on a relative :math:`L^2` (the numerator becomes
:math:`\\lVert t \\rVert`), while a prediction that diverges scores
arbitrarily large.  A single ratio cannot tell "the model stopped
predicting" from "the model exploded", which is why its seed spread was
44.8x and why gating it was a mistake (@sec-nb-homog-seeds-verdict).

This computes the *signed* diagnostic instead:

    amplitude ratio  r = RMS(pred) / RMS(true)

    r -> 0   collapse   (no signal; relative L2 saturates near 1)
    r ~ 1    healthy amplitude
    r >> 1   divergence

Reads the memmaps the harvest already writes, so it applies retroactively
to every arm ever harvested -- no instrument change and no re-run.

Usage::

    python amplitude_ratio.py [ARTIFACT_ROOT] [TAG ...]
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import numpy as np

DEFAULT_ROOT = os.path.expanduser(
    "~/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-wave-harvest/artifacts"
)
SPLITS = ("res2500", "res5000", "res10000", "res20000", "res40000", "shift")


def _sample_dirs(root: str, tag: str, split: str) -> list[str]:
    return sorted(
        glob.glob(
            f"{root}/{tag}/{split}/*/predictions/*/_tensordict/interior"
            "/_tensordict/point_data/"
        )
    )


def amplitude_ratios(root: str, tag: str, split: str) -> list[float]:
    """RMS(pred)/RMS(true) per sample, float64, NaN-safe."""
    out = []
    for d in _sample_dirs(root, tag, split):
        try:
            t = np.memmap(d + "true_pressure.memmap", dtype=np.float32, mode="r")
            p = np.memmap(d + "pred_pressure.memmap", dtype=np.float32, mode="r")
        except FileNotFoundError:
            continue
        t64, p64 = t.astype(np.float64), p.astype(np.float64)
        ### The shift family stores ABSOLUTE pressure; the amplitude that
        ### matters is the fluctuation about p_inf, matching the denominator
        ### `pressure_l2` itself uses (@sec-nb-metric-audit).
        p_inf = _read_p_inf(d)
        t64, p64 = t64 - p_inf, p64 - p_inf
        den = np.sqrt((t64**2).mean())
        if not np.isfinite(den) or den == 0.0:
            continue
        out.append(float(np.sqrt((p64**2).mean()) / den))
    return out


def shape_errors(root: str, tag: str, split: str) -> list[float]:
    """Scale-invariant relative error: the error a perfect rescale cannot fix.

    ``min_a ||a*p - t|| / ||t||`` has the closed form :math:`|\\sin\\theta|
    = \\sqrt{1 - \\cos^2\\theta}` with :math:`\\cos\\theta` the cosine
    similarity of the two fluctuation fields.  This is the number that
    separates the two candidate stories for cross-family failure:

    * ~0    -> the predicted FIELD is right and only its SCALE is wrong,
              so one calibration constant would fix transfer;
    * ~1    -> the prediction is unrelated to the target and the large
              ``pressure_l2`` is not merely a gain error.

    It is needed because on a diverged arm ``pressure_l2`` collapses onto
    the amplitude ratio automatically (``||p - t|| ~ ||p||`` once
    ``||p|| >> ||t||``) and therefore carries no shape information at all.
    """
    out = []
    for d in _sample_dirs(root, tag, split):
        try:
            t = np.memmap(d + "true_pressure.memmap", dtype=np.float32, mode="r")
            p = np.memmap(d + "pred_pressure.memmap", dtype=np.float32, mode="r")
        except FileNotFoundError:
            continue
        p_inf = _read_p_inf(d)
        t64 = t.astype(np.float64) - p_inf
        p64 = p.astype(np.float64) - p_inf
        nt, npd = np.linalg.norm(t64), np.linalg.norm(p64)
        if not (np.isfinite(nt) and np.isfinite(npd)) or nt == 0.0 or npd == 0.0:
            continue
        cos = float(np.dot(p64, t64) / (nt * npd))
        out.append(float(np.sqrt(max(0.0, 1.0 - cos * cos))))
    return out


def pattern_errors(root: str, tag: str, split: str) -> list[float]:
    """Shape error after removing each field's OWN mean: the affine-invariant error.

    ``shape_errors`` can be driven to ~1 by a constant offset alone, because
    a near-constant vector is almost orthogonal to a zero-mean fluctuation.
    The predictions on the shift family do carry a large spurious offset, so
    that confound is live and has to be divided out before any claim about
    the spatial *pattern*.

    This is ``min_{a,b} ||a*p + b - t|| / ||t - mean(t)||``: the error that
    survives an arbitrary gain AND an arbitrary offset.  Near 0 means the
    model has the right spatial pattern and only its affine calibration is
    wrong -- a two-parameter fix.  Near 1 means the pattern itself is wrong.
    """
    out = []
    for d in _sample_dirs(root, tag, split):
        try:
            t = np.memmap(d + "true_pressure.memmap", dtype=np.float32, mode="r")
            p = np.memmap(d + "pred_pressure.memmap", dtype=np.float32, mode="r")
        except FileNotFoundError:
            continue
        t64 = t.astype(np.float64)
        p64 = p.astype(np.float64)
        t64 = t64 - t64.mean()
        p64 = p64 - p64.mean()
        nt, npd = np.linalg.norm(t64), np.linalg.norm(p64)
        if not (np.isfinite(nt) and np.isfinite(npd)) or nt == 0.0 or npd == 0.0:
            continue
        cos = float(np.dot(p64, t64) / (nt * npd))
        out.append(float(np.sqrt(max(0.0, 1.0 - cos * cos))))
    return out


def _read_p_inf(point_data_dir: str) -> float:
    """Per-sample freestream pressure, or 0.0 when the dataset omits it."""
    sample = point_data_dir.split("/_tensordict/interior/")[0]
    for g in glob.glob(
        f"{sample}/_tensordict/boundaries/*/_tensordict/global_data/p_inf.memmap"
    ):
        return float(np.memmap(g, dtype=np.float32, mode="r")[0])
    return 0.0


def logged_l2(root: str, tag: str, split: str) -> list[float]:
    fs = glob.glob(f"{root}/{tag}/{split}/*/metrics.jsonl")
    if not fs:
        return []
    rows = [json.loads(line) for line in open(fs[0]) if line.strip()]
    return [
        r["metrics"]["pressure_l2"] for r in rows if r.get("phase") == "infer_step"
    ]


def _verdict(r: float) -> str:
    if r < 0.5:
        return "COLLAPSE"
    if r > 2.0:
        return "DIVERGE"
    return "healthy"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    tags = sys.argv[2:] or sorted(
        os.path.basename(p.rstrip("/")) for p in glob.glob(f"{root}/*/")
    )
    print(
        f"{'arm':<12s}{'split':<10s}{'med L2':>11s}{'med r':>10s}"
        f"{'shapeErr':>10s}{'pattErr':>10s}  verdict"
    )
    for tag in tags:
        for split in SPLITS:
            r = amplitude_ratios(root, tag, split)
            l2 = logged_l2(root, tag, split)
            sh = shape_errors(root, tag, split)
            pa = pattern_errors(root, tag, split)
            if not r:
                continue
            mr = statistics.median(r)
            ml2 = statistics.median(l2) if l2 else float("nan")
            msh = statistics.median(sh) if sh else float("nan")
            mpa = statistics.median(pa) if pa else float("nan")
            print(
                f"{tag:<12s}{split:<10s}{ml2:11.4g}{mr:10.4g}"
                f"{msh:10.4g}{mpa:10.4g}  {_verdict(mr)}"
            )
        print()


if __name__ == "__main__":
    main()
