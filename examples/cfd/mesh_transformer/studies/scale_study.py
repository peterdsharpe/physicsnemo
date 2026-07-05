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

r"""Scale-ladder study: can the kernel-decoder MeshTransformer reach product scope?

The product scope for the mesh-attention program is roughly
:math:`N_s = 10^5` boundary cells and :math:`N_q = 10^6` interior query
points per case.  The architecture is now at its simplest measured form --
the ``singpair`` dictionary (exactly two exact singular members: double layer
plus single layer, no polynomial and no MLP smooth members) and **one**
encoder layer -- so the cost curves measured here are the ones that matter.

Three chapters:

1. **Forward/backward cost sweep** (measurement only, no training) over a
   grid of icosphere resolutions (``subdivisions`` :math:`s` gives
   :math:`20\cdot4^s` triangles: 320 / 1280 / 5120 / 20480 for
   :math:`s=2..5`) crossed with query counts :math:`\{10^3, 10^4, 10^5\}` on
   realistic ``build_laplace3d_sample`` sphere cases: forward wall-clock,
   forward+backward wall-clock, and peak GPU memory, with per-point OOM
   guards (an OOM is a *result*, recorded honestly, never a crash).  The
   public ``MeshTransformer.encode``/``decode`` split separates the encoder
   cost (boundary self-attention + drive blocks + kernel source cache) from
   the decoder cost (query lift + dense pair-kernel decode + output
   projection).

2. **Accuracy-versus-resolution transfer** (light training): train the
   singpair arm at the benchmark default resolution (subdivisions=2, 320
   triangles) exactly as in ``laplace3d_study.run_experiment``, then
   evaluate zero-shot at subdivisions {2, 3, 4} on the sphere and star
   tiers.  Exact cell-integrated quadrature members suggest accuracy should
   *hold* as resolution grows -- this is the architecture's
   resolution-transfer claim at scale.

3. **Extrapolated wall**: fit power laws
   :math:`t \approx c\,N_q^{\alpha} N_s^{\beta}` to the measured grid and
   extrapolate time and memory to the product scope, with an honest
   statement of what breaks first.

Cost-model notes (pre-registered, so the fits have something to falsify):

- The *decoder* is the documented dense :math:`O(N_qN_s)` pair-kernel
  evaluation (``kernel_decoder.py``); expect
  :math:`\alpha \approx \beta \approx 1` for decode time at large sizes.
- The *encoder* is separable-moment attention (``attention.py``): source
  moments are formed once and evaluated independently per cell, so encoder
  cost is :math:`O(N_s)` per layer, **not** :math:`O(N_s^2)` -- global
  boundary self-attention is *not* the wall in this architecture.
- Inference memory is bounded by decoder chunking: the kernel decoder
  evaluates its dense chunk at an internal ``query_chunk_size`` (2048 by
  default -- note this is *not* the ``MeshTransformer(query_chunk_size=...)``
  constructor argument, which only bounds the outer decode loop; the dense
  chunk is currently not constructor-exposed), so forward peak memory scales
  like :math:`O(\text{chunk}\times N_s)` plus :math:`O(N_q)` outputs.
- *Training* memory is not chunk-bounded: autograd saves the per-pair
  ``members`` and ``kernel`` tensors of every chunk until ``backward``, so
  training peak memory scales like :math:`O(N_qN_s)`.  This is the expected
  first structural break at product scope.

This is a benchmark-local research asset, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from laplace3d import build_laplace3d_sample
from laplace3d_study import (
    build_mesh_transformer_3d,
    merge_dirichlet_boundaries,
)

#: Cost-sweep grid defaults.  Icosphere subdivisions s -> 20 * 4**s triangles.
DEFAULT_SUBDIVISIONS = (2, 3, 4, 5)  # 320, 1280, 5120, 20480 triangles
DEFAULT_QUERIES = (1_000, 10_000, 100_000)

#: Product scope the program must reach (100k boundary cells, 1M queries).
PRODUCT_N_SOURCES = 100_000
PRODUCT_N_QUERIES = 1_000_000

#: Reference accelerator for the "what breaks first" statement (A100 80GB).
REFERENCE_GPU_BYTES = 80 * 1024**3

TRANSFER_TIERS = ("sphere", "star")
TRANSFER_SUBDIVISIONS = (2, 3, 4)
TRANSFER_N_QUERY = 2_000


def build_scale_model():
    """The singpair arm at its simplest measured depth: one encoder layer.

    ``operator_layers=1`` cites the iteration-19 mixed-BC discriminator
    result (book/05-benchmarks.qmd, ``@sec-mixed-bc``, checked-in key
    ``iteration_19_mixed_bc_discriminator``): on the genuinely nonlocal
    mixed-BC shell tier, enc1 matches enc2 (mean relative L2 0.051 vs 0.049
    over seeds 17/18/19) while enc0 is unreliable (seed-18 blowup 0.112) --
    exactly one encoder layer suffices, and the Dirichlet stress families
    had already shown depth beyond one to be dead weight (iteration 18).
    The dictionary is the two-member "singpair" (exact double layer + exact
    single layer, no polynomial, no MLP members).
    """

    return build_mesh_transformer_3d(
        kernel_mlp_members=0,
        kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        operator_layers=1,
    )


# ---------------------------------------------------------------------------
# Chapter 1: forward/backward cost sweep
# ---------------------------------------------------------------------------


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _timed(function, *, device: torch.device, warmup: int, repeats: int) -> dict:
    """Synchronized wall-clock samples of ``function`` in milliseconds."""

    for _ in range(warmup):
        function()
    _synchronize(device)
    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        _synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1.0e3)
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def _reset_peak(device: torch.device) -> int | None:
    """Reset the allocator peak and return the current baseline, CUDA only."""

    if device.type != "cuda":
        return None
    _synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return int(torch.cuda.memory_allocated(device))


def _read_peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    _synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def measure_grid_point(
    model,
    *,
    subdivisions: int,
    n_query: int,
    device: torch.device,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict:
    """Measure one (resolution, query-count) point, recording OOMs as data.

    Stages run cheapest-first; a training-stage OOM does not discard the
    inference measurements, and an inference OOM skips the strictly larger
    training stages rather than burning time on guaranteed failures.
    """

    sample = build_laplace3d_sample(
        seed,
        tier="sphere",
        bc_regime="dirichlet",
        subdivisions=subdivisions,
        n_query=n_query,
        device=device,
    )
    domain = sample.domain
    merged = merge_dirichlet_boundaries(domain)
    inner = model.model  # the wrapped MeshTransformer: public encode/decode
    record: dict = {
        "subdivisions": subdivisions,
        "n_boundary_cells": int(
            sum(mesh.n_cells for mesh in domain.boundaries.values())
        ),
        "n_query": n_query,
        "sample_seed": seed,
        "oom_stages": [],
        "skipped_stages": [],
    }

    def guarded(stage: str, function):
        try:
            return function()
        except Exception as error:  # noqa: BLE001 - re-raised unless OOM
            if not _is_oom(error):
                raise
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record["oom_stages"].append(stage)
            return None

    # --- inference: full forward, then the public encode/decode split ------
    def forward_inference():
        baseline = _reset_peak(device)
        with torch.no_grad():
            timing = _timed(
                lambda: model(domain), device=device, warmup=warmup, repeats=repeats
            )
        record["forward_ms"] = timing
        record["forward_baseline_bytes"] = baseline
        record["forward_peak_bytes"] = _read_peak(device)
        return True

    def encode_decode_split():
        with torch.no_grad():
            record["encode_ms"] = _timed(
                lambda: inner.encode(merged),
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
            encoded = inner.encode(merged)
            record["decode_ms"] = _timed(
                lambda: inner.decode(encoded),
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        return True

    # --- training: forward + backward through the dense decode -------------
    def train_step():
        model.zero_grad(set_to_none=True)
        prediction = model(domain).point_data["potential"]
        prediction.square().sum().backward()

    def forward_backward():
        baseline = _reset_peak(device)
        record["forward_backward_ms"] = _timed(
            train_step, device=device, warmup=warmup, repeats=repeats
        )
        record["train_baseline_bytes"] = baseline
        record["train_peak_bytes"] = _read_peak(device)
        model.zero_grad(set_to_none=True)
        return True

    inference_ok = guarded("forward", forward_inference) is not None
    if inference_ok:
        guarded("encode_decode", encode_decode_split)
        guarded("forward_backward", forward_backward)
    else:
        # Training strictly dominates inference memory; do not re-OOM.
        record["skipped_stages"] = ["encode_decode", "forward_backward"]

    record["status"] = "ok" if not record["oom_stages"] else "oom"
    return record


def run_cost_sweep(
    model,
    *,
    subdivisions: tuple[int, ...],
    queries: tuple[int, ...],
    device: torch.device,
    seed: int,
    warmup: int,
    repeats: int,
) -> list[dict]:
    """Sweep the (resolution x query-count) grid with per-point OOM guards."""

    grid = []
    for point_index, s in enumerate(sorted(subdivisions)):
        for q in sorted(queries):
            grid.append(
                measure_grid_point(
                    model,
                    subdivisions=s,
                    n_query=q,
                    device=device,
                    seed=seed + 7919 * point_index + q,
                    warmup=warmup,
                    repeats=repeats,
                )
            )
            print(
                json.dumps(
                    {
                        k: grid[-1].get(k)
                        for k in (
                            "subdivisions",
                            "n_boundary_cells",
                            "n_query",
                            "status",
                            "oom_stages",
                        )
                    }
                ),
                flush=True,
            )
    return grid


# ---------------------------------------------------------------------------
# Chapter 3: scaling fits and the extrapolated wall
# ---------------------------------------------------------------------------


def fit_power_law(points: list[tuple[float, float, float]]) -> dict | None:
    r"""Least-squares fit of :math:`v \approx c\,q^{\alpha} s^{\beta}` in log space.

    ``points`` are ``(n_query, n_sources, value)`` triples; requires at least
    two distinct values along each axis (otherwise the exponent is not
    identifiable and ``None`` is returned -- degenerate smoke grids must not
    produce fake exponents).
    """

    clean = [(q, s, v) for q, s, v in points if v is not None and v > 0.0]
    if len(clean) < 3:
        return None
    if len({q for q, _, _ in clean}) < 2 or len({s for _, s, _ in clean}) < 2:
        return None
    design = torch.tensor(
        [[1.0, math.log(q), math.log(s)] for q, s, _ in clean], dtype=torch.float64
    )
    target = torch.tensor([math.log(v) for _, _, v in clean], dtype=torch.float64)
    solution = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution.squeeze(-1)
    residual = design @ solution - target
    total = target - target.mean()
    r_squared = float(
        1.0
        - residual.square().sum()
        / total.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
    )
    return {
        "coefficient": math.exp(float(solution[0])),
        "query_exponent": float(solution[1]),
        "source_exponent": float(solution[2]),
        "r_squared": r_squared,
        "n_points": len(clean),
    }


def fit_power_law_sources(points: list[tuple[float, float]]) -> dict | None:
    r"""Single-variable fit :math:`v \approx c\,s^{\beta}` (encoder cost)."""

    clean = [(s, v) for s, v in points if v is not None and v > 0.0]
    if len(clean) < 2 or len({s for s, _ in clean}) < 2:
        return None
    design = torch.tensor([[1.0, math.log(s)] for s, _ in clean], dtype=torch.float64)
    target = torch.tensor([math.log(v) for _, v in clean], dtype=torch.float64)
    solution = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution.squeeze(-1)
    residual = design @ solution - target
    total = target - target.mean()
    r_squared = float(
        1.0
        - residual.square().sum()
        / total.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
    )
    return {
        "coefficient": math.exp(float(solution[0])),
        "source_exponent": float(solution[1]),
        "r_squared": r_squared,
        "n_points": len(clean),
    }


#: metric -> (timing?, peak key that validates its wall-clock, ideal scaling).
#: The ideal scaling is the pre-registered cost model: dense decode work and
#: retained training activations are O(Nq*Ns); the separable-moment encoder
#: and the chunk-bounded forward peak are O(Ns).
_METRIC_SPECS: dict[str, tuple[bool, str | None, tuple[bool, bool]]] = {
    "forward_ms": (True, "forward_peak_bytes", (True, True)),
    "encode_ms": (True, "forward_peak_bytes", (False, True)),
    "decode_ms": (True, "forward_peak_bytes", (True, True)),
    "forward_backward_ms": (True, "train_peak_bytes", (True, True)),
    "forward_peak_bytes": (False, None, (False, True)),
    "train_peak_bytes": (False, None, (True, True)),
}


#: Backward of this graph replays a bounded multiple of the forward work
#: (~2.5-3.5x measured at clean points), so a forward+backward exceeding this
#: generous ratio indicates allocator paging/thrash, not architecture cost.
_FWDBWD_OUTLIER_RATIO = 6.0


def _paging_suspected(
    record: dict, peak_key: str | None, device_total_memory_bytes: int | None
) -> bool:
    """True when the stage's peak allocation exceeded physical device memory.

    A CUDA allocation larger than the device (possible under WSL2 / unified
    shared-memory drivers, which page to host RAM instead of raising OOM)
    keeps the run alive, but its wall-clock measures the page migrations,
    not the architecture -- such timings are excluded from the fits and the
    exclusion is reported, never silently absorbed.
    """

    if peak_key is None or device_total_memory_bytes is None:
        return False
    peak = record.get(peak_key)
    return peak is not None and peak > device_total_memory_bytes


def _timing_exclusion_reason(
    record: dict, name: str, device_total_memory_bytes: int | None
) -> str | None:
    """Why this point's wall-clock for ``name`` is not architecture cost."""

    timing, peak_key, _ = _METRIC_SPECS[name]
    if not timing or record.get(name) is None:
        return None
    if _paging_suspected(record, peak_key, device_total_memory_bytes):
        return (
            f"{name}: {peak_key} exceeded physical device memory (the driver "
            f"paged to host RAM instead of raising OOM)"
        )
    if name == "forward_backward_ms" and record.get("forward_ms") is not None:
        ratio = record[name]["median_ms"] / record["forward_ms"]["median_ms"]
        if ratio > _FWDBWD_OUTLIER_RATIO:
            return (
                f"{name}: {ratio:.0f}x the forward pass (backward replays a "
                f"bounded multiple of forward work; near-capacity allocator "
                f"paging suspected)"
            )
    return None


def _grid_metric(
    grid: list[dict],
    key: str,
    *,
    device_total_memory_bytes: int | None = None,
) -> list[tuple]:
    timing, _, _ = _METRIC_SPECS[key]
    points = []
    for record in grid:
        value = record.get(key)
        if timing and value is not None:
            value = value["median_ms"]
        if value is None:
            continue
        if _timing_exclusion_reason(record, key, device_total_memory_bytes):
            continue
        points.append((record["n_query"], record["n_boundary_cells"], value))
    return points


def analyze_cost_grid(
    grid: list[dict],
    *,
    device_total_memory_bytes: int | None = None,
) -> dict:
    """Fit measured scaling exponents and extrapolate to the product scope.

    Two extrapolations per metric, both reported: the *fitted* power law
    evaluated at the product scope, and an *ideal-exponent* proportional
    scaling from the largest cleanly measured point (decode/train assume
    exact :math:`O(N_qN_s)`; encode and the chunk-bounded forward peak
    assume :math:`O(N_s)` -- the encoder is separable-moment attention per
    ``attention.py``, so there is no dense :math:`O(N_s^2)` term to
    extrapolate).  Points whose peak allocation exceeded physical device
    memory (shared-memory paging) are excluded from wall-clock fits and
    listed in ``notes``; their memory numbers remain valid allocator counts
    and stay in the memory fits.
    """

    notes: list[str] = []
    for record in grid:
        reasons = sorted(
            {
                reason
                for name in _METRIC_SPECS
                if (
                    reason := _timing_exclusion_reason(
                        record, name, device_total_memory_bytes
                    )
                )
            }
        )
        notes.extend(
            f"point (n_query={record['n_query']}, "
            f"n_boundary_cells={record['n_boundary_cells']}) excluded "
            f"from the timing fits -- {reason}"
            for reason in reasons
        )

    fits: dict[str, dict | None] = {}
    for name in _METRIC_SPECS:
        points = _grid_metric(
            grid, name, device_total_memory_bytes=device_total_memory_bytes
        )
        if name == "encode_ms":
            # Encoder cost is query independent; collapse to per-resolution
            # medians before the single-variable fit.
            encode_by_s: dict[float, list[float]] = {}
            for _, s, v in points:
                encode_by_s.setdefault(s, []).append(v)
            fits[name] = fit_power_law_sources(
                [(s, statistics.median(vs)) for s, vs in sorted(encode_by_s.items())]
            )
        else:
            fits[name] = fit_power_law(points)
    encode_fit = fits.get("encode_ms")
    if encode_fit is not None and encode_fit["source_exponent"] < 0.5:
        notes.append(
            "encode_ms shows no growth with boundary size over the measured "
            "range (separable-moment encoder, O(Ns) per layer): the "
            "measurements are kernel-launch-overhead dominated and the "
            "fitted exponent is a floor artifact, not architecture cost; "
            "the encoder is not the wall"
        )

    extrapolation: dict = {
        "n_boundary_cells": PRODUCT_N_SOURCES,
        "n_query": PRODUCT_N_QUERIES,
        "fitted": {},
        "ideal_from_largest_measured": {},
    }
    for name, fit in fits.items():
        if fit is None:
            extrapolation["fitted"][name] = None
        elif name == "encode_ms":
            # A floor-dominated (non-growing) encode fit extrapolates to
            # nonsense at product scope; the ideal O(Ns) entry and the note
            # above carry the honest number instead.
            extrapolation["fitted"][name] = (
                fit["coefficient"] * PRODUCT_N_SOURCES ** fit["source_exponent"]
                if fit["source_exponent"] >= 0.5
                else None
            )
        else:
            extrapolation["fitted"][name] = (
                fit["coefficient"]
                * PRODUCT_N_QUERIES ** fit["query_exponent"]
                * PRODUCT_N_SOURCES ** fit["source_exponent"]
            )
    fitted = extrapolation["fitted"]
    if (
        fitted.get("forward_backward_ms") is not None
        and fitted.get("forward_ms") is not None
        and fitted["forward_backward_ms"] < fitted["forward_ms"]
    ):
        # Hard invariant: forward+backward can never cost less than forward.
        # A violation means the clean training points that survived the
        # paging exclusions are too small (overhead-dominated) to identify
        # the exponents; refuse the number rather than report it.
        notes.append(
            "the fitted forward_backward_ms extrapolation fell below the "
            "forward-only extrapolation (the forward+backward points that "
            "survive the paging exclusions are small and overhead-dominated "
            "on this device); it is suppressed in favor of the ideal "
            "O(Nq*Ns) entry"
        )
        fitted["forward_backward_ms"] = None

    for name, (timing, _, (scales_q, scales_s)) in _METRIC_SPECS.items():
        # Largest clean point *for this metric*: paging-affected wall-clock
        # must not seed the proportional extrapolation either.
        largest = None
        for record in grid:
            value = record.get(name)
            if value is None:
                continue
            if _timing_exclusion_reason(record, name, device_total_memory_bytes):
                continue
            size = (record["n_query"] if scales_q else 1) * record["n_boundary_cells"]
            if largest is None or size > largest[0]:
                largest = (size, record)
        if largest is None:
            extrapolation["ideal_from_largest_measured"][name] = None
            continue
        record = largest[1]
        value = record[name]["median_ms"] if timing else record[name]
        factor = (PRODUCT_N_QUERIES / record["n_query"] if scales_q else 1.0) * (
            PRODUCT_N_SOURCES / record["n_boundary_cells"] if scales_s else 1.0
        )
        extrapolation["ideal_from_largest_measured"][name] = {
            "value": value * factor,
            "from_point": {
                "n_query": record["n_query"],
                "n_boundary_cells": record["n_boundary_cells"],
            },
        }

    extrapolation["what_breaks_first"] = _what_breaks_first(extrapolation)
    return {"scaling_fits": fits, "extrapolation": extrapolation, "notes": notes}


def _what_breaks_first(extrapolation: dict) -> str:
    """Compose the honest product-scope statement from the extrapolated numbers."""

    fitted = extrapolation["fitted"]
    ideal = extrapolation.get("ideal_from_largest_measured") or {}

    def best(name):
        # Prefer the fitted power law; fall back to the ideal-exponent
        # extrapolation when the grid was too degenerate to fit.
        if fitted.get(name) is not None:
            return fitted[name]
        entry = ideal.get(name)
        return None if entry is None else entry["value"]

    def gib(value):
        return None if value is None else value / 1024**3

    train_gib = gib(best("train_peak_bytes"))
    forward_gib = gib(best("forward_peak_bytes"))
    fwdbwd_s = (best("forward_backward_ms") or 0.0) / 1.0e3
    forward_s = (best("forward_ms") or 0.0) / 1.0e3
    if train_gib is None:
        return "insufficient measured points to extrapolate"

    reference_gib = REFERENCE_GPU_BYTES / 1024**3
    parts = []
    if train_gib > reference_gib:
        parts.append(
            f"training memory breaks first: one forward+backward at product "
            f"scope extrapolates to ~{train_gib:.0f} GiB "
            f"({train_gib / reference_gib:.0f}x an 80 GiB A100) because "
            f"autograd retains the dense O(Nq*Ns) pair activations of the "
            f"kernel decode; gradient checkpointing over query chunks (or a "
            f"hierarchical decode backend) is required before product-scale "
            f"training is possible"
        )
    else:
        parts.append(
            f"training memory extrapolates to ~{train_gib:.1f} GiB and fits "
            f"an 80 GiB A100"
        )
    if forward_gib is not None:
        if forward_gib > reference_gib:
            parts.append(
                f"inference forward peak (~{forward_gib:.0f} GiB at the "
                f"current fixed dense chunk of 2048) also exceeds 80 GiB and "
                f"needs a smaller kernel-decoder chunk (not currently "
                f"constructor-exposed)"
            )
        else:
            parts.append(
                f"inference stays chunk-bounded at ~{forward_gib:.1f} GiB "
                f"and is limited by wall-clock instead: ~{forward_s:.0f} s "
                f"per forward (~{fwdbwd_s:.0f} s forward+backward), the "
                f"dense O(Nq*Ns) decode"
            )
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Chapter 2: accuracy-versus-resolution transfer
# ---------------------------------------------------------------------------


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


@torch.no_grad()
def _validation_score(model, *, device, dtype) -> float:
    """Mean sphere+star relative L2 (3 cases each), matching laplace3d_study."""

    model.eval()
    scores = []
    for index, tier in enumerate(TRANSFER_TIERS):
        errors = []
        for case in range(3):
            sample = build_laplace3d_sample(
                71_000_011 + 7919 * case + 1_000_003 * index,
                tier=tier,
                bc_regime="dirichlet",
                device=device,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        scores.append(sum(errors) / len(errors))
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_transfer(
    model,
    *,
    device: torch.device,
    n_cases: int,
    dtype=torch.float32,
    subdivision_levels: tuple[int, ...] = TRANSFER_SUBDIVISIONS,
    n_query: int = TRANSFER_N_QUERY,
) -> dict:
    """Zero-shot resolution transfer: same weights, finer boundary meshes."""

    model.eval()
    results = {}
    cells = [(tier, s) for tier in sorted(TRANSFER_TIERS) for s in subdivision_levels]
    for index, (tier, s) in enumerate(cells):
        errors = []
        for case in range(n_cases):
            sample = build_laplace3d_sample(
                97_000_037 + 7919 * case + 1_000_003 * index,
                tier=tier,
                bc_regime="dirichlet",
                subdivisions=s,
                n_query=n_query,
                device=device,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        results[f"{tier}_subdiv{s}"] = sum(errors) / len(errors)
    return results


def run_transfer(
    *,
    steps: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    eval_cases: int,
) -> dict:
    """Train once at subdivisions=2 (benchmark default), evaluate zero-shot.

    Training replicates ``laplace3d_study.run_experiment`` (alternating
    sphere/star all-Dirichlet cases at the benchmark default resolution,
    AdamW, best-validation state selection).  ``laplace3d_study`` saves only
    a JSON report, never weights, so the study trains once and caches its
    own checkpoint; a rerun with the same seed reuses it.
    """

    torch.manual_seed(seed)
    dtype = torch.float32
    model = build_scale_model().to(device)
    checkpoint = output_dir / f"singpair_enc1_seed{seed}.pt"
    history: list[dict] = []
    reused = checkpoint.exists()
    if reused:
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3.0e-4, weight_decay=1.0e-6
        )
        train_tiers = ("sphere", "star")
        best_val, best_state = float("inf"), None
        for step in range(1, steps + 1):
            model.train()
            sample = build_laplace3d_sample(
                seed + 104_729 * step,
                tier=train_tiers[step % 2],
                bc_regime="dirichlet",
                device=device,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
            loss = torch.sum((prediction - sample.target).square()) / torch.sum(
                sample.target.square()
            ).clamp_min(1.0e-30)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == steps:
                score = _validation_score(model, device=device, dtype=dtype)
                history.append({"step": step, "validation": score})
                if score < best_val:
                    best_val = score
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), checkpoint)

    return {
        "steps": steps,
        "seed": seed,
        "train_subdivisions": 2,
        "checkpoint": str(checkpoint),
        "reused_checkpoint": reused,
        "history": history,
        "zero_shot": evaluate_transfer(
            model, device=device, n_cases=eval_cases, dtype=dtype
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_study(
    *,
    mode: str = "all",
    output_dir: str,
    device: str = "cuda",
    steps: int = 3_000,
    seed: int = 17,
    subdivisions: tuple[int, ...] = DEFAULT_SUBDIVISIONS,
    queries: tuple[int, ...] = DEFAULT_QUERIES,
    warmup: int = 1,
    repeats: int = 3,
    eval_cases: int = 6,
) -> dict:
    if mode not in ("cost", "transfer", "all"):
        raise ValueError("mode must be 'cost', 'transfer', or 'all'")
    device_obj = torch.device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    torch.manual_seed(seed)
    model = build_scale_model().to(device_obj)
    report: dict = {
        "study": "scale_study",
        "model": "mesh_transformer_kernel_singpair_enc1",
        "mode": mode,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device_obj),
        "device_name": (
            torch.cuda.get_device_name(device_obj)
            if device_obj.type == "cuda"
            else "cpu"
        ),
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(device_obj).total_memory
            if device_obj.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }

    if mode in ("cost", "all"):
        grid = run_cost_sweep(
            model,
            subdivisions=tuple(subdivisions),
            queries=tuple(queries),
            device=device_obj,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )
        report["cost"] = {
            "timing": {"warmup": warmup, "repeats": repeats},
            "grid": grid,
            **analyze_cost_grid(
                grid,
                device_total_memory_bytes=report["device_total_memory_bytes"],
            ),
        }

    if mode in ("transfer", "all"):
        report["transfer"] = run_transfer(
            steps=steps,
            seed=seed,
            device=device_obj,
            output_dir=out,
            eval_cases=eval_cases,
        )

    report["elapsed_seconds"] = time.time() - start
    suffix = "" if mode == "all" else f"_{mode}"
    (out / f"scale_study{suffix}_seed{seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cost", "transfer", "all"), default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--subdivisions", type=int, nargs="+", default=list(DEFAULT_SUBDIVISIONS)
    )
    parser.add_argument("--queries", type=int, nargs="+", default=list(DEFAULT_QUERIES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eval-cases", type=int, default=6)
    arguments = parser.parse_args()
    result = run_study(
        mode=arguments.mode,
        output_dir=arguments.output_dir,
        device=arguments.device,
        steps=arguments.steps,
        seed=arguments.seed,
        subdivisions=tuple(arguments.subdivisions),
        queries=tuple(arguments.queries),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        eval_cases=arguments.eval_cases,
    )
    summary: dict = {"study": result["study"], "mode": result["mode"]}
    if "cost" in result:
        summary["scaling_fits"] = result["cost"]["scaling_fits"]
        summary["what_breaks_first"] = result["cost"]["extrapolation"][
            "what_breaks_first"
        ]
    if "transfer" in result:
        summary["zero_shot"] = result["transfer"]["zero_shot"]
    print(json.dumps(summary, indent=2))
