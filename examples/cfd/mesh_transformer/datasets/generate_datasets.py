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

r"""Generate solver-verified boundary-to-interior benchmark datasets.

Two families are implemented.  ``star_random_trace`` (Laplace, scalar) is
documented below; ``ns_cavity_star`` is the program's first suite whose
labels come from a numerical solver on a genuinely *nonlinear* PDE — steady
incompressible Navier-Stokes at moderate Reynolds number
(:mod:`fem_navier_stokes`, Taylor-Hood P2-P1 with Newton), on the same
certified star-deformed disks, driven by a smooth band-limited *tangential*
boundary velocity localized on a boundary arc (driven-lid-like).  A
tangential-only drive has zero normal flux pointwise, so the global
compatibility condition :math:`\oint u\cdot n\,ds = 0` holds exactly at the
continuous level by construction; the tiny discrete defect of the
interpolated trace is absorbed by the mean-pressure Lagrange multiplier and
reported per case.  The Reynolds number ``Re`` (peak drive speed 1, unit
reference length, so ``Re = 1/nu``) is the family's global operator scalar
axis; cases whose Newton solve fails (after backtracking and viscosity
continuation) are rejected and resampled deterministically, with rejection
counts reported in the manifest.  Splits: ``train``/``eval_id`` (shared
distribution), ``eval_unseen_Re`` (a higher Reynolds band),
``eval_unseen_geometry`` (the bank's stronger-deformation range), and
``eval_unseen_drive_profile`` (a disjoint, higher modulation band).
Per-case verification records the Newton residual, divergence norm, mass
and momentum balance checks, and a coarse self-consistency re-solve; a
deterministic subset additionally re-solves at ``target_h / 1.5`` (a finer
mesh) to bound the label noise floor, reported in the manifest.
Supplementary catalog *slices* that move only the Reynolds band (e.g. the
iteration-37 low-Re slice ``v1-lowre``, Re log-uniform [0.5, 5] with the
unseen band at (5, 10]) are generated with ``--reynolds-range`` /
``--unseen-reynolds-range``; every other distribution parameter and the
verification discipline stay the production family's.

The first family, ``star_random_trace``, decouples boundary conditions from
manufactured-solution reachability:

- **Geometry** reuses the 2D bank's certified star-deformed disks
  (:func:`conformal_laplace.sample_geometry`: random Fourier deformation
  ``F(z) = z + sum a_m z^m`` of the unit disk with ``sum m |a_m| < 1``),
  with the bank's training ranges (modes ``(2, 3)``, deformation
  ``0.05..0.35``) and its stronger-deformation OOD range (``0.45..0.65``).
  The physical similarity is the identity, so ``reference_length == 1``.
- **Dirichlet traces** are random band-limited Fourier series in the
  *normalized boundary arc length* ``s`` — NOT in the conformal preimage
  angle.  A trace band-limited in the preimage angle would be exactly the
  boundary restriction of a low-degree harmonic polynomial (the manufactured
  family); a band in arc length is generically outside every finite
  manufactured construction, which is the point of this dataset.  Mode ``k``
  carries weight ``k**(-decay)`` (bank convention: ``regularity = 2``) with
  independent standard-normal coefficients, and the whole trace is
  normalized to a prescribed boundary RMS.  The band edge is controllable:
  the train split uses modes ``1..4`` and the
  ``eval_unseen_trace_frequencies`` split uses modes ``5..8`` (higher band
  edge, disjoint from training — mirroring the bank's unseen-drive-modes
  split).
- **Ground truth** comes from :func:`fem_reference.solve_dirichlet` (P2
  triangles on a 2048-gon sampling of the smooth boundary, ``target_h =
  0.02``), verified per case by (a) the maximum principle against the trace
  range and (b) a self-consistency re-solve at 1.5x the mesh size and half
  the boundary resolution; aggregate statistics land in the manifest.

Each case's on-disk layout matches ``dataset_catalog.REQUIRED_ARRAYS``: the
benchmark-facing boundary discretization is a 64-panel polygon with
``boundary_value`` sampled at panel *parameter midpoints*, and interior
targets sit at area-uniform query points — both conventions inherited from
``conformal_laplace.build_domain_sample`` so cataloged cases drop into the
existing training/eval loops through ``dataset_catalog.load_domain_sample``.

Example::

    python generate_datasets.py --family star_random_trace \
        --n-cases 8 --version v0-demo --workers 1

This is a benchmark-local research utility, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from conformal_laplace import (
    conformal_derivative,
    conformal_map,
    sample_disk_preimages,
    sample_geometry,
    unit_circle,
)
from dataset_catalog import (
    SCHEMA_VERSION,
    case_filename,
    catalog_dir,
    save_case,
    sha256_of_file,
    validate_catalog,
    write_manifest,
)
from fem_navier_stokes import NewtonError, solve_navier_stokes
from fem_reference import solve_dirichlet

from physicsnemo.mesh import Mesh

FAMILIES = ("star_random_trace", "ns_cavity_star")

SPLIT_ORDER = (
    "train",
    "eval_id",
    "eval_unseen_trace_frequencies",
    "eval_stronger_deformation",
)

NS_SPLIT_ORDER = (
    "train",
    "eval_id",
    "eval_unseen_Re",
    "eval_unseen_geometry",
    "eval_unseen_drive_profile",
)

FAMILY_SPLIT_ORDER: dict[str, tuple[str, ...]] = {
    "star_random_trace": SPLIT_ORDER,
    "ns_cavity_star": NS_SPLIT_ORDER,
}


@dataclass(frozen=True)
class SplitSpec:
    """Distribution parameters of one split of the star_random_trace family."""

    geometry_modes: tuple[int, ...]
    deformation_range: tuple[float, float]
    trace_band: tuple[int, int]  # inclusive Fourier mode range in arc length


SPLIT_SPECS: dict[str, SplitSpec] = {
    "train": SplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        trace_band=(1, 4),
    ),
    "eval_id": SplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        trace_band=(1, 4),
    ),
    "eval_unseen_trace_frequencies": SplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        trace_band=(5, 8),
    ),
    "eval_stronger_deformation": SplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.45, 0.65),
        trace_band=(1, 4),
    ),
}


@dataclass(frozen=True)
class GeneratorSettings:
    """Everything one worker needs to build a case (picklable primitives)."""

    family: str = "star_random_trace"
    n_boundary: int = 64
    n_query: int = 128
    n_fem_boundary: int = 2048
    target_h: float = 0.02
    trace_decay: float = 2.0
    boundary_rms: float = 1.0
    equation: str = "laplace"
    kappa: float = 0.0
    base_seed: int = 0
    verify: bool = True
    self_check_h_factor: float = 1.5


@dataclass(frozen=True)
class NSSplitSpec:
    """Distribution parameters of one split of the ns_cavity_star family."""

    geometry_modes: tuple[int, ...]
    deformation_range: tuple[float, float]
    reynolds_range: tuple[float, float]  # log-uniform sampling band
    drive_band: tuple[int, int]  # inclusive modulation mode range
    drive_width_range: tuple[float, float]  # arc width as fraction of perimeter


NS_SPLIT_SPECS: dict[str, NSSplitSpec] = {
    "train": NSSplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        reynolds_range=(10.0, 200.0),
        drive_band=(0, 2),
        drive_width_range=(0.25, 0.5),
    ),
    "eval_id": NSSplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        reynolds_range=(10.0, 200.0),
        drive_band=(0, 2),
        drive_width_range=(0.25, 0.5),
    ),
    "eval_unseen_Re": NSSplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        reynolds_range=(200.0, 300.0),
        drive_band=(0, 2),
        drive_width_range=(0.25, 0.5),
    ),
    "eval_unseen_geometry": NSSplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.45, 0.65),
        reynolds_range=(10.0, 200.0),
        drive_band=(0, 2),
        drive_width_range=(0.25, 0.5),
    ),
    "eval_unseen_drive_profile": NSSplitSpec(
        geometry_modes=(2, 3),
        deformation_range=(0.05, 0.35),
        reynolds_range=(10.0, 200.0),
        drive_band=(3, 5),
        drive_width_range=(0.25, 0.5),
    ),
}


@dataclass(frozen=True)
class NSGeneratorSettings:
    """Everything one worker needs to build an N-S case (picklable).

    ``reynolds_range`` / ``unseen_reynolds_range`` (both optional) override
    the split specs' log-uniform Reynolds bands -- the supplementary-slice
    mechanism (see :func:`_ns_split_spec`).  Geometry, drive, and query
    distributions always stay the production family's.
    """

    family: str = "ns_cavity_star"
    n_boundary: int = 64
    n_query_interior: int = 128
    n_query_near: int = 32
    interior_margin: float = 0.12
    near_band: tuple[float, float] = (0.02, 0.08)
    n_fem_boundary: int = 1024
    target_h: float = 0.0105
    drive_decay: float = 1.0
    base_seed: int = 0
    verify: bool = True
    self_check_h_factor: float = 1.5
    noise_floor_stride: int = 0  # 0 disables the finer-mesh subset re-solve
    max_attempts: int = 8
    max_newton_iterations: int = 25
    newton_tolerance: float = 1.0e-10
    reynolds_range: tuple[float, float] | None = None
    unseen_reynolds_range: tuple[float, float] | None = None


def _ns_split_spec(split_name: str, settings: NSGeneratorSettings) -> NSSplitSpec:
    """One split's distribution spec, with the settings' Reynolds override.

    ``settings.reynolds_range`` (when set) replaces the log-uniform Reynolds
    band of every split *except* ``eval_unseen_Re``, whose band comes from
    ``settings.unseen_reynolds_range`` -- the supplementary-catalog-slice
    mechanism (e.g. the iteration-37 low-Re slice ``v1-lowre`` trains and
    evaluates in-distribution at Re [0.5, 5] with the unseen band directly
    above at (5, 10]).  Only the Reynolds band is overridable: geometry,
    drive, and query distributions are the family's by construction, so a
    slice differs from the production catalog in exactly one declared axis.
    """

    spec = NS_SPLIT_SPECS[split_name]
    override = (
        settings.unseen_reynolds_range
        if split_name == "eval_unseen_Re"
        else settings.reynolds_range
    )
    if override is None:
        return spec
    low, high = float(override[0]), float(override[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(
            f"Reynolds band override must satisfy 0 < low < high, got {override!r}"
        )
    return NSSplitSpec(**{**asdict(spec), "reynolds_range": (low, high)})


def _substream(seed: int, stream: int) -> int:
    """Derive independent deterministic seeds (mirrors liouville._substream)."""

    return seed + 15_485_863 * stream


def split_sizes(n_cases: int, family: str = "star_random_trace") -> dict[str, int]:
    """Deterministic split allocation per family.

    ``star_random_trace`` gives each of its three eval splits ``n // 6``
    cases; ``ns_cavity_star`` gives each of its four eval splits ``n // 10``
    (so a 1,500-case catalog carries 900 train and 150 per eval split).
    Datasets with fewer than four cases (smoke tests) are all-train; from
    four cases up, every evaluation split is nonempty.
    """

    if not isinstance(n_cases, int) or isinstance(n_cases, bool) or n_cases < 1:
        raise ValueError("n_cases must be a positive integer")
    order = FAMILY_SPLIT_ORDER[family]
    n_eval_splits = len(order) - 1
    divisor = 6 if family == "star_random_trace" else 10
    n_eval = max(1, n_cases // divisor) if n_cases >= 4 else 0
    sizes = {name: n_eval for name in order if name != "train"}
    sizes["train"] = n_cases - n_eval_splits * n_eval
    return sizes


def split_ranges(n_cases: int, family: str = "star_random_trace") -> dict[str, dict]:
    """Contiguous case-index ranges per nonempty split, in family order."""

    sizes = split_sizes(n_cases, family)
    ranges: dict[str, dict] = {}
    cursor = 0
    for name in FAMILY_SPLIT_ORDER[family]:
        if sizes[name] == 0:
            continue
        ranges[name] = {"start": cursor, "stop": cursor + sizes[name]}
        cursor += sizes[name]
    return ranges


def _split_of(index: int, ranges: dict[str, dict]) -> str:
    for name, span in ranges.items():
        if span["start"] <= index < span["stop"]:
            return name
    raise ValueError(f"case index {index} is outside every split range")


def _sample_arc_length_trace(
    seed: int,
    arc_lengths: np.ndarray,
    total_length: float,
    *,
    band: tuple[int, int],
    decay: float,
    boundary_rms: float,
) -> tuple[np.ndarray, dict]:
    """Random band-limited Fourier trace in normalized arc length.

    Returns the trace evaluated at ``arc_lengths`` plus the exact
    coefficients so a case is fully reproducible from its parameters.  The
    normalization uses the exact arc-length RMS of the series
    (``a0**2 + 0.5 * sum(w_k**2 * (a_k**2 + b_k**2))``).
    """

    k_lo, k_hi = band
    if k_lo < 1 or k_hi < k_lo:
        raise ValueError("trace band must satisfy 1 <= k_lo <= k_hi")
    rng = np.random.default_rng(seed)
    modes = np.arange(k_lo, k_hi + 1)
    weights = modes.astype(np.float64) ** (-decay)
    constant = float(rng.standard_normal())
    cosine = rng.standard_normal(modes.shape[0])
    sine = rng.standard_normal(modes.shape[0])
    energy = constant**2 + 0.5 * np.sum(weights**2 * (cosine**2 + sine**2))
    scale = boundary_rms / math.sqrt(energy)

    phases = 2.0 * math.pi * np.outer(arc_lengths / total_length, modes)
    values = scale * (
        constant
        + np.cos(phases) @ (weights * cosine)
        + np.sin(phases) @ (weights * sine)
    )
    parameters = {
        "band": [int(k_lo), int(k_hi)],
        "decay": float(decay),
        "boundary_rms": float(boundary_rms),
        "normalization_scale": float(scale),
        "constant": constant,
        "cosine_coefficients": cosine.tolist(),
        "sine_coefficients": sine.tolist(),
    }
    return values, parameters


def _star_boundary(geometry, angles: torch.Tensor) -> np.ndarray:
    """Physical boundary points ``F(exp(i * angles))`` (identity similarity)."""

    z = unit_circle(angles)
    mapped = conformal_map(geometry, z)
    return torch.stack((mapped.real, mapped.imag), dim=-1).numpy()


def _cumulative_arc_length(loop: np.ndarray) -> tuple[np.ndarray, float]:
    """Arc length at each closed-loop vertex plus the total loop length."""

    chords = np.linalg.norm(np.diff(loop, axis=0, append=loop[:1]), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(chords[:-1])))
    return arc, float(chords.sum())


def _star_tangents(geometry, angles: torch.Tensor) -> np.ndarray:
    """Unit tangents of the smooth star curve at preimage angles.

    The curve is ``theta -> F(exp(i theta))`` with derivative
    ``i exp(i theta) F'(exp(i theta))``; normalizing gives the exact unit
    tangent of the *smooth* boundary (counterclockwise orientation), which
    is where the tangential drive lives.
    """

    z = unit_circle(angles)
    derivative = 1j * z * conformal_derivative(geometry, z)
    tangent = torch.stack((derivative.real, derivative.imag), dim=-1)
    return (tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)).numpy()


class _DegenerateDriveError(RuntimeError):
    """The sampled drive profile is numerically zero (resample the case)."""


def _sample_arc_drive_profile(
    seed: int,
    *,
    band: tuple[int, int],
    decay: float,
    width_range: tuple[float, float],
    total_length: float,
):
    r"""Random smooth band-limited tangential-drive profile on a boundary arc.

    The profile is ``g(s) = w(tau) m(tau)`` in the arc-local coordinate
    ``tau = (s - s_c) / (width/2)`` (wrapped): ``w`` is the :math:`C^\infty`
    bump ``exp(1 - 1/(1 - tau^2))`` supported on ``|tau| < 1`` and ``m`` is
    a band-limited Fourier modulation ``sum_k (1+k)^{-decay} (a_k
    cos(pi k tau) + b_k sin(pi k tau))`` over the split's inclusive mode
    band.  Returns ``(evaluate, parameters)`` where ``evaluate`` maps arc
    lengths to *unnormalized* profile values; the caller normalizes to unit
    peak speed (the ``U = 1`` convention that makes ``Re = 1/nu``).
    """

    k_lo, k_hi = band
    if k_lo < 0 or k_hi < k_lo:
        raise ValueError("drive band must satisfy 0 <= k_lo <= k_hi")
    rng = np.random.default_rng(seed)
    center_fraction = float(rng.uniform(0.0, 1.0))
    width_fraction = float(rng.uniform(*width_range))
    modes = np.arange(k_lo, k_hi + 1)
    weights = (1.0 + modes.astype(np.float64)) ** (-decay)
    cosine = rng.standard_normal(modes.shape[0])
    sine = rng.standard_normal(modes.shape[0])

    def evaluate(arc_lengths: np.ndarray) -> np.ndarray:
        offset = (
            np.asarray(arc_lengths, dtype=np.float64) / total_length
            - center_fraction
            + 0.5
        ) % 1.0 - 0.5
        tau = offset / (0.5 * width_fraction)
        window = np.zeros_like(tau)
        inside = np.abs(tau) < 1.0
        t = tau[inside]
        window[inside] = np.exp(1.0 - 1.0 / (1.0 - t * t))
        phases = np.pi * np.outer(tau, modes)
        modulation = np.cos(phases) @ (weights * cosine) + np.sin(phases) @ (
            weights * sine
        )
        return window * modulation

    parameters = {
        "band": [int(k_lo), int(k_hi)],
        "decay": float(decay),
        "center_fraction": center_fraction,
        "width_fraction": width_fraction,
        "cosine_coefficients": cosine.tolist(),
        "sine_coefficients": sine.tolist(),
    }
    return evaluate, parameters


def _polygon_wall_distance(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polygonal boundary."""

    starts = loop
    ends = np.roll(loop, -1, axis=0)
    vectors = ends - starts
    lengths2 = np.maximum((vectors**2).sum(axis=1), 1.0e-300)
    offsets = points[:, None, :] - starts[None]
    t = np.clip((offsets * vectors[None]).sum(axis=-1) / lengths2[None], 0.0, 1.0)
    projections = starts[None] + t[..., None] * vectors[None]
    return np.linalg.norm(points[:, None, :] - projections, axis=-1).min(axis=1)


def _sample_bucketed_queries(
    seed: int,
    loop: np.ndarray,
    *,
    n_interior: int,
    n_near: int,
    interior_margin: float,
    near_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform interior queries plus a near-wall bucket.

    Returns ``(points, wall_distances)`` with the ``n_interior``
    interior-bucket points (distance >= ``interior_margin``) first and the
    ``n_near`` near-wall points (distance in ``near_band``) after them.
    Within each bucket the points are area-uniform over the bucket region
    (rejection sampling in the bounding box).
    """

    from matplotlib.path import Path

    rng = np.random.default_rng(seed)
    path = Path(loop)
    low = loop.min(axis=0)
    high = loop.max(axis=0)
    interior: list[np.ndarray] = []
    near: list[np.ndarray] = []
    interior_distance: list[np.ndarray] = []
    near_distance: list[np.ndarray] = []
    n_interior_found = n_near_found = 0
    for _ in range(4000):
        if n_interior_found >= n_interior and n_near_found >= n_near:
            break
        candidates = rng.uniform(low, high, size=(1024, 2))
        candidates = candidates[path.contains_points(candidates)]
        if candidates.shape[0] == 0:
            continue
        distance = _polygon_wall_distance(candidates, loop)
        mask = distance >= interior_margin
        if n_interior_found < n_interior and mask.any():
            interior.append(candidates[mask])
            interior_distance.append(distance[mask])
            n_interior_found += int(mask.sum())
        mask = (distance >= near_band[0]) & (distance <= near_band[1])
        if n_near_found < n_near and mask.any():
            near.append(candidates[mask])
            near_distance.append(distance[mask])
            n_near_found += int(mask.sum())
    if n_interior_found < n_interior or n_near_found < n_near:
        raise RuntimeError(
            "query rejection sampling failed to fill both distance buckets"
        )
    points = np.concatenate(
        (
            np.concatenate(interior, axis=0)[:n_interior],
            np.concatenate(near, axis=0)[:n_near],
        ),
        axis=0,
    )
    distances = np.concatenate(
        (
            np.concatenate(interior_distance)[:n_interior],
            np.concatenate(near_distance)[:n_near],
        )
    )
    return points, distances


def _generate_case(job: tuple[int, str, GeneratorSettings]) -> dict:
    """Build one star_random_trace case (runs in a worker process)."""

    index, split_name, settings = job
    spec = SPLIT_SPECS[split_name]
    case_seed = settings.base_seed + 104_729 * index
    start = time.perf_counter()

    geometry = sample_geometry(
        _substream(case_seed, 0),
        modes=spec.geometry_modes,
        deformation_range=spec.deformation_range,
        dtype=torch.float64,
    )

    # Dense boundary sampling drives the FEM ground truth; the benchmark
    # discretization below is a coarse subsampling of the same curve.
    n_dense = settings.n_fem_boundary
    dense_angles = 2.0 * math.pi * torch.arange(n_dense, dtype=torch.float64) / n_dense
    fem_loop = _star_boundary(geometry, dense_angles)
    dense_arc, total_length = _cumulative_arc_length(fem_loop)
    trace_seed = _substream(case_seed, 2)
    fem_trace, trace_parameters = _sample_arc_length_trace(
        trace_seed,
        dense_arc,
        total_length,
        band=spec.trace_band,
        decay=settings.trace_decay,
        boundary_rms=settings.boundary_rms,
    )

    # Benchmark boundary: n_boundary panels with parameter-midpoint values
    # (conventions of conformal_laplace.build_domain_sample, identity
    # similarity, hence the (successor, index) cell winding).
    n_boundary = settings.n_boundary
    panel_angles = (
        2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    )
    boundary_points = _star_boundary(geometry, panel_angles)
    indices = np.arange(n_boundary, dtype=np.int64)
    boundary_cells = np.stack((np.roll(indices, -1), indices), axis=-1)
    midpoint_angles = panel_angles + math.pi / n_boundary
    dense_theta = dense_angles.numpy()
    midpoint_arc = np.interp(
        midpoint_angles.numpy(),
        np.concatenate((dense_theta, [2.0 * math.pi])),
        np.concatenate((dense_arc, [total_length])),
    )
    boundary_value, _ = _sample_arc_length_trace(
        trace_seed,
        midpoint_arc,
        total_length,
        band=spec.trace_band,
        decay=settings.trace_decay,
        boundary_rms=settings.boundary_rms,
    )

    mesh = Mesh(
        points=torch.from_numpy(boundary_points),
        cells=torch.from_numpy(boundary_cells),
    )
    centroids = mesh.cell_centroids.numpy()
    normals = mesh.cell_normals.numpy()
    measures = mesh.cell_areas.numpy()

    query_preimages = sample_disk_preimages(
        _substream(case_seed, 1), settings.n_query, dtype=torch.float64
    )
    mapped = conformal_map(geometry, query_preimages)
    query_points = torch.stack((mapped.real, mapped.imag), dim=-1).numpy()

    solution = solve_dirichlet(
        [fem_loop],
        [fem_trace],
        query_points,
        equation=settings.equation,
        kappa=settings.kappa,
        target_h=settings.target_h,
    )

    verification = {
        "max_principle_violation": float(
            max(
                0.0,
                solution.u_query.max() - fem_trace.max(),
                fem_trace.min() - solution.u_query.min(),
            )
        ),
        "queries_snapped": solution.diagnostics.n_queries_snapped,
        "max_snap_distance": solution.diagnostics.max_snap_distance,
        "linear_residual": solution.diagnostics.linear_residual,
    }
    if settings.verify:
        coarse = solve_dirichlet(
            [fem_loop[::2]],
            [fem_trace[::2]],
            query_points,
            equation=settings.equation,
            kappa=settings.kappa,
            target_h=settings.target_h * settings.self_check_h_factor,
        )
        verification["self_consistency_rel_l2"] = float(
            np.linalg.norm(solution.u_query - coarse.u_query)
            / max(np.linalg.norm(solution.u_query), 1.0e-300)
        )

    elapsed = time.perf_counter() - start
    params = {
        "family": settings.family,
        "split": split_name,
        "case_index": index,
        "case_seed": case_seed,
        "reference_length": 1.0,
        "geometry": {
            "modes": list(spec.geometry_modes),
            "coefficients_real": geometry.coefficients.real.tolist(),
            "coefficients_imag": geometry.coefficients.imag.tolist(),
            "deformation_bound": float(geometry.deformation_bound.item()),
            "deformation_range": list(spec.deformation_range),
        },
        "trace": {"seed": trace_seed, "parametrization": "arc_length"}
        | trace_parameters,
        "solver": {
            "equation": settings.equation,
            "kappa": settings.kappa,
            "target_h": settings.target_h,
            "degree": 2,
            "n_fem_boundary": settings.n_fem_boundary,
        },
        "verification": verification,
        "generation_seconds": elapsed,
    }
    arrays = {
        "boundary_points": boundary_points,
        "boundary_cells": boundary_cells,
        "boundary_loop_offsets": np.array([0, n_boundary], dtype=np.int64),
        "boundary_cell_centroids": centroids,
        "boundary_cell_normals": normals,
        "boundary_cell_measures": measures,
        "boundary_value": boundary_value,
        "query_points": query_points,
        "u_query": solution.u_query,
    }
    return {
        "index": index,
        "split": split_name,
        "arrays": arrays,
        "params": params,
        "verification": verification | {"generation_seconds": elapsed},
    }


def _generate_ns_case(job: tuple[int, str, "NSGeneratorSettings"]) -> dict:
    """Build one ns_cavity_star case (runs in a worker process).

    A case attempt samples geometry, Reynolds number, drive profile, and
    queries from deterministic substreams; attempts whose Newton solve (or
    whose verification re-solves) fail are recorded and resampled with the
    next attempt substream, up to ``settings.max_attempts`` before the whole
    generation aborts.  Rejections are part of the dataset's reported
    statistics, not silent.
    """

    index, split_name, settings = job
    spec = _ns_split_spec(split_name, settings)
    case_seed = settings.base_seed + 104_729 * index
    start = time.perf_counter()
    rejected: list[dict] = []

    for attempt in range(settings.max_attempts):
        attempt_seed = _substream(case_seed, 100 + attempt)
        geometry = sample_geometry(
            _substream(attempt_seed, 0),
            modes=spec.geometry_modes,
            deformation_range=spec.deformation_range,
            dtype=torch.float64,
        )
        n_dense = settings.n_fem_boundary
        dense_angles = (
            2.0 * math.pi * torch.arange(n_dense, dtype=torch.float64) / n_dense
        )
        fem_loop = _star_boundary(geometry, dense_angles)
        dense_arc, total_length = _cumulative_arc_length(fem_loop)

        reynolds_rng = np.random.default_rng(_substream(attempt_seed, 3))
        log_lo, log_hi = np.log(spec.reynolds_range)
        reynolds = float(np.exp(reynolds_rng.uniform(log_lo, log_hi)))

        drive_seed = _substream(attempt_seed, 2)
        evaluate_drive, drive_parameters = _sample_arc_drive_profile(
            drive_seed,
            band=spec.drive_band,
            decay=settings.drive_decay,
            width_range=spec.drive_width_range,
            total_length=total_length,
        )
        raw = evaluate_drive(dense_arc)
        peak = float(np.abs(raw).max())
        if peak < 1.0e-8:
            rejected.append(
                {"attempt": attempt, "reason": "degenerate_drive", "peak": peak}
            )
            continue
        scale = 1.0 / peak
        tangents_dense = _star_tangents(geometry, dense_angles)
        fem_velocity = (scale * raw)[:, None] * tangents_dense

        # Benchmark boundary: n_boundary panels, drive at parameter midpoints
        # (the star_random_trace conventions, identity similarity).
        n_boundary = settings.n_boundary
        panel_angles = (
            2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
        )
        boundary_points = _star_boundary(geometry, panel_angles)
        indices = np.arange(n_boundary, dtype=np.int64)
        boundary_cells = np.stack((np.roll(indices, -1), indices), axis=-1)
        midpoint_angles = panel_angles + math.pi / n_boundary
        dense_theta = dense_angles.numpy()
        midpoint_arc = np.interp(
            midpoint_angles.numpy(),
            np.concatenate((dense_theta, [2.0 * math.pi])),
            np.concatenate((dense_arc, [total_length])),
        )
        midpoint_tangents = _star_tangents(geometry, midpoint_angles)
        boundary_velocity = (scale * evaluate_drive(midpoint_arc))[
            :, None
        ] * midpoint_tangents

        mesh = Mesh(
            points=torch.from_numpy(boundary_points),
            cells=torch.from_numpy(boundary_cells),
        )
        centroids = mesh.cell_centroids.numpy()
        normals = mesh.cell_normals.numpy()
        measures = mesh.cell_areas.numpy()

        query_points, wall_distance = _sample_bucketed_queries(
            _substream(attempt_seed, 1),
            fem_loop,
            n_interior=settings.n_query_interior,
            n_near=settings.n_query_near,
            interior_margin=settings.interior_margin,
            near_band=settings.near_band,
        )

        viscosity = 1.0 / reynolds
        try:
            solution = solve_navier_stokes(
                [fem_loop],
                [fem_velocity],
                query_points,
                viscosity=viscosity,
                target_h=settings.target_h,
                max_newton_iterations=settings.max_newton_iterations,
                newton_tolerance=settings.newton_tolerance,
                continuation=True,
            )
        except NewtonError as error:
            rejected.append(
                {
                    "attempt": attempt,
                    "reason": "newton",
                    "reynolds": reynolds,
                    "detail": str(error),
                }
            )
            continue

        diagnostics = solution.diagnostics
        verification = {
            "newton_iterations": diagnostics.newton_iterations,
            "continuation_solves": diagnostics.continuation_solves,
            "backtracking_steps": diagnostics.backtracking_steps,
            "relative_residual": diagnostics.relative_residual,
            "boundary_flux": diagnostics.boundary_flux,
            "lagrange_multiplier": diagnostics.lagrange_multiplier,
            "divergence_l2_normalized": diagnostics.divergence_l2_normalized,
            "momentum_balance_error": diagnostics.momentum_balance_error,
            "queries_snapped": diagnostics.n_queries_snapped,
            "max_snap_distance": diagnostics.max_snap_distance,
            "rejected_attempts": len(rejected),
        }

        try:
            if settings.verify:
                coarse = solve_navier_stokes(
                    [fem_loop[::2]],
                    [fem_velocity[::2]],
                    query_points,
                    viscosity=viscosity,
                    target_h=settings.target_h * settings.self_check_h_factor,
                    max_newton_iterations=settings.max_newton_iterations,
                    newton_tolerance=settings.newton_tolerance,
                    continuation=True,
                )
                verification["self_consistency_rel_l2_velocity"] = float(
                    np.linalg.norm(solution.velocity_query - coarse.velocity_query)
                    / max(np.linalg.norm(solution.velocity_query), 1.0e-300)
                )
                verification["self_consistency_rel_l2_pressure"] = float(
                    np.linalg.norm(solution.pressure_query - coarse.pressure_query)
                    / max(np.linalg.norm(solution.pressure_query), 1.0e-300)
                )
            if settings.noise_floor_stride and (
                index % settings.noise_floor_stride == 0
            ):
                fine = solve_navier_stokes(
                    [fem_loop],
                    [fem_velocity],
                    query_points,
                    viscosity=viscosity,
                    target_h=settings.target_h / settings.self_check_h_factor,
                    max_newton_iterations=settings.max_newton_iterations,
                    newton_tolerance=settings.newton_tolerance,
                    continuation=True,
                )
                verification["label_noise_floor_velocity"] = float(
                    np.linalg.norm(solution.velocity_query - fine.velocity_query)
                    / max(np.linalg.norm(fine.velocity_query), 1.0e-300)
                )
                verification["label_noise_floor_pressure"] = float(
                    np.linalg.norm(solution.pressure_query - fine.pressure_query)
                    / max(np.linalg.norm(fine.pressure_query), 1.0e-300)
                )
        except NewtonError as error:
            rejected.append(
                {
                    "attempt": attempt,
                    "reason": "verification_newton",
                    "reynolds": reynolds,
                    "detail": str(error),
                }
            )
            continue

        elapsed = time.perf_counter() - start
        params = {
            "family": settings.family,
            "split": split_name,
            "case_index": index,
            "case_seed": case_seed,
            "attempt": attempt,
            "rejected_attempts": rejected,
            "reference_length": 1.0,
            "reynolds": reynolds,
            "viscosity": viscosity,
            "geometry": {
                "modes": list(spec.geometry_modes),
                "coefficients_real": geometry.coefficients.real.tolist(),
                "coefficients_imag": geometry.coefficients.imag.tolist(),
                "deformation_bound": float(geometry.deformation_bound.item()),
                "deformation_range": list(spec.deformation_range),
            },
            "drive": {
                "seed": drive_seed,
                "parametrization": "arc_length_tangential",
                "normalization_scale": scale,
                "peak_speed": 1.0,
            }
            | drive_parameters,
            "queries": {
                "n_interior": settings.n_query_interior,
                "n_near_wall": settings.n_query_near,
                "interior_margin": settings.interior_margin,
                "near_band": list(settings.near_band),
            },
            "solver": {
                "equation": "navier_stokes",
                "elements": "Taylor-Hood P2-P1",
                "target_h": settings.target_h,
                "n_fem_boundary": settings.n_fem_boundary,
                "pressure_gauge": "zero mean (Lagrange multiplier)",
            },
            "verification": verification,
            "generation_seconds": elapsed,
        }
        arrays = {
            "boundary_points": boundary_points,
            "boundary_cells": boundary_cells,
            "boundary_loop_offsets": np.array([0, n_boundary], dtype=np.int64),
            "boundary_cell_centroids": centroids,
            "boundary_cell_normals": normals,
            "boundary_cell_measures": measures,
            "boundary_velocity": boundary_velocity,
            "query_points": query_points,
            "query_wall_distance": wall_distance,
            "velocity_query": solution.velocity_query,
            "pressure_query": solution.pressure_query,
        }
        return {
            "index": index,
            "split": split_name,
            "arrays": arrays,
            "params": params,
            "verification": verification | {"generation_seconds": elapsed},
        }

    raise RuntimeError(
        f"case {index} ({split_name}) failed every one of "
        f"{settings.max_attempts} attempts: {rejected}"
    )


def _aggregate_verification(results: list[dict]) -> dict:
    """Per-split summary statistics of the per-case verification numbers."""

    summary: dict[str, dict] = {}
    for split_name in SPLIT_ORDER:
        rows = [r["verification"] for r in results if r["split"] == split_name]
        if not rows:
            continue
        entry = {
            "n_cases": len(rows),
            "max_principle_violation_max": max(
                r["max_principle_violation"] for r in rows
            ),
            "queries_snapped_total": int(sum(r["queries_snapped"] for r in rows)),
            "max_snap_distance_max": max(r["max_snap_distance"] for r in rows),
            "linear_residual_max": max(r["linear_residual"] for r in rows),
            "generation_seconds_mean": sum(r["generation_seconds"] for r in rows)
            / len(rows),
        }
        if all("self_consistency_rel_l2" in r for r in rows):
            values = [r["self_consistency_rel_l2"] for r in rows]
            entry["self_consistency_rel_l2_max"] = max(values)
            entry["self_consistency_rel_l2_mean"] = sum(values) / len(values)
        summary[split_name] = entry
    return summary


def _aggregate_ns_verification(results: list[dict]) -> dict:
    """Per-split summaries of the N-S per-case verification numbers.

    Alongside the extrema of the solver diagnostics, the label-noise-floor
    block aggregates the finer-mesh (``target_h / 1.5``) re-solves of the
    deterministic case subset -- the manifest's headline bound on how far
    the stored labels sit from the mesh-converged solution.
    """

    summary: dict[str, dict] = {}
    for split_name in NS_SPLIT_ORDER:
        rows = [r["verification"] for r in results if r["split"] == split_name]
        if not rows:
            continue

        def stat(key: str, reducer) -> float:
            return reducer(r[key] for r in rows)

        entry = {
            "n_cases": len(rows),
            "newton_iterations_max": stat("newton_iterations", max),
            "continuation_solves_total": int(
                sum(r["continuation_solves"] for r in rows)
            ),
            "relative_residual_max": stat("relative_residual", max),
            "boundary_flux_max_abs": max(abs(r["boundary_flux"]) for r in rows),
            "divergence_l2_normalized_max": stat("divergence_l2_normalized", max),
            "momentum_balance_error_max": stat("momentum_balance_error", max),
            "queries_snapped_total": int(sum(r["queries_snapped"] for r in rows)),
            "max_snap_distance_max": stat("max_snap_distance", max),
            "rejected_attempts_total": int(sum(r["rejected_attempts"] for r in rows)),
            "generation_seconds_mean": sum(r["generation_seconds"] for r in rows)
            / len(rows),
        }
        for field in ("velocity", "pressure"):
            key = f"self_consistency_rel_l2_{field}"
            values = [r[key] for r in rows if key in r]
            if values:
                entry[f"{key}_max"] = max(values)
                entry[f"{key}_mean"] = sum(values) / len(values)
            key = f"label_noise_floor_{field}"
            values = [r[key] for r in rows if key in r]
            if values:
                entry[f"{key}_max"] = max(values)
                entry[f"{key}_mean"] = sum(values) / len(values)
                entry[f"{key}_cases"] = len(values)
        summary[split_name] = entry
    return summary


def generate_dataset(
    *,
    family: str,
    n_cases: int,
    version: str,
    workers: int = 1,
    settings: GeneratorSettings | None = None,
    root: Path | str | None = None,
    created: str = "unspecified",
) -> Path:
    """Generate one cataloged dataset version and return its directory.

    Cases are generated in parallel (``spawn`` multiprocessing) when
    ``workers > 1``, then written and checksummed serially, and the finished
    catalog is validated before returning.  ``created`` is stamped verbatim
    into the manifest (the CLI passes the current date; library callers and
    tests pass an explicit string so outputs stay reproducible).

    ``workers > 1`` requires the calling process to be spawn-reimportable
    (a real script or module, as with the CLI); interactive or stdin-driven
    callers should use ``workers=1``.
    """

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    settings_type = (
        NSGeneratorSettings if family == "ns_cavity_star" else GeneratorSettings
    )
    settings = settings_type() if settings is None else settings
    if not isinstance(settings, settings_type):
        raise TypeError(
            f"family {family!r} requires {settings_type.__name__}, "
            f"got {type(settings).__name__}"
        )
    if settings.family != family:
        settings = settings_type(**{**asdict(settings), "family": family})
    worker_function = (
        _generate_ns_case if family == "ns_cavity_star" else _generate_case
    )
    ranges = split_ranges(n_cases, family)
    jobs = [(index, _split_of(index, ranges), settings) for index in range(n_cases)]

    directory = catalog_dir(family, version, root)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing dataset version at {directory}"
        )

    # Cases are written and checksummed INCREMENTALLY as workers return
    # them (imap_unordered), so progress is observable on disk and a
    # crashed run leaves its completed cases behind; only the manifest is
    # withheld until every case exists, so an interrupted directory can
    # never validate.  The on-disk result is identical to a batch write.
    checksums: dict[str, str] = {}
    results: list[dict] = []

    def record(result: dict) -> None:
        path = save_case(directory, result["index"], result["arrays"], result["params"])
        checksums[case_filename(result["index"])] = sha256_of_file(path)
        results.append(
            {
                "index": result["index"],
                "split": result["split"],
                "verification": result["verification"],
            }
        )

    start = time.perf_counter()
    if workers == 1:
        for job in jobs:
            record(worker_function(job))
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(worker_function, jobs):
                record(result)
    wall_seconds = time.perf_counter() - start
    results.sort(key=lambda r: r["index"])

    if family == "ns_cavity_star":
        from fem_navier_stokes import LINEAR_SOLVER

        split_params = {
            name: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(_ns_split_spec(name, settings)).items()
            }
            for name in ranges
        }
        solver_settings = {
            "solver": "fem_navier_stokes.solve_navier_stokes",
            "equation": (
                "steady incompressible Navier-Stokes: "
                "-nu lap(u) + (u.grad)u + grad p = 0, div u = 0"
            ),
            "element": "Taylor-Hood P2-P1 triangles",
            "mesher": "triangle (constrained Delaunay, flags pq30a<area>)",
            "pressure_gauge": "zero mean pressure via Lagrange multiplier",
            "nonlinear_solver": (
                "Newton with analytic Jacobian, Stokes initial guess, "
                "backtracking, viscosity continuation ladder 8-4-2-1"
            ),
            "linear_solver": LINEAR_SOLVER,
            "target_h": settings.target_h,
            "n_fem_boundary": settings.n_fem_boundary,
            "self_check_h_factor": settings.self_check_h_factor,
            "noise_floor_stride": settings.noise_floor_stride,
            "newton_tolerance": settings.newton_tolerance,
            "reynolds_convention": (
                "Re = U L / nu with peak drive speed U = 1 (normalized) and "
                "reference length L = 1, so nu = 1/Re; the loader exposes "
                "viscosity = 1/Re as the global operator scalar"
            ),
        }
        verification = _aggregate_ns_verification(results)
    else:
        split_params = {
            name: {
                "geometry_modes": list(SPLIT_SPECS[name].geometry_modes),
                "deformation_range": list(SPLIT_SPECS[name].deformation_range),
                "trace_band": list(SPLIT_SPECS[name].trace_band),
            }
            for name in ranges
        }
        solver_settings = {
            "solver": "fem_reference.solve_dirichlet",
            "element": "P2 triangles",
            "mesher": "triangle (constrained Delaunay, flags pq30a<area>)",
            "target_h": settings.target_h,
            "n_fem_boundary": settings.n_fem_boundary,
            "equation": settings.equation,
            "kappa": settings.kappa,
            "self_check_h_factor": settings.self_check_h_factor,
        }
        verification = _aggregate_verification(results)

    splits = {
        name: {
            "start": span["start"],
            "stop": span["stop"],
            "params": split_params[name],
        }
        for name, span in ranges.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "version": version,
        "created": created,
        "n_cases": n_cases,
        "seeds": {"base_seed": settings.base_seed, "case_seed_stride": 104_729},
        "generator_settings": asdict(settings),
        "solver_settings": solver_settings,
        "splits": splits,
        "verification": verification
        | {
            "wall_seconds_total": wall_seconds,
            "wall_seconds_per_case": wall_seconds / n_cases,
            "workers": workers,
        },
        "checksums": checksums,
    }
    write_manifest(directory, manifest)
    validate_catalog(directory)
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--n-cases", type=int, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=None, help="catalog root directory")
    parser.add_argument("--n-boundary", type=int, default=64)
    parser.add_argument("--n-query", type=int, default=128)
    parser.add_argument(
        "--n-fem-boundary",
        type=int,
        default=None,
        help="dense FEM boundary polygon size (default: 2048 Laplace, 1024 N-S)",
    )
    parser.add_argument(
        "--target-h",
        type=float,
        default=None,
        help="FEM mesh target edge length (default: family production value)",
    )
    parser.add_argument(
        "--noise-floor-stride",
        type=int,
        default=0,
        help="N-S only: every k-th case re-solves at target_h/1.5 to bound "
        "the label noise floor (0 disables)",
    )
    parser.add_argument(
        "--reynolds-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="N-S only: override the log-uniform Reynolds band of every "
        "split except eval_unseen_Re (the supplementary-slice mechanism; "
        "default: the production family bands)",
    )
    parser.add_argument(
        "--unseen-reynolds-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="N-S only: override the eval_unseen_Re Reynolds band",
    )
    parser.add_argument("--no-verify", action="store_true")
    arguments = parser.parse_args(argv)

    import datetime

    if arguments.family == "ns_cavity_star":
        defaults = NSGeneratorSettings()
        settings: GeneratorSettings | NSGeneratorSettings = NSGeneratorSettings(
            family=arguments.family,
            n_boundary=arguments.n_boundary,
            n_fem_boundary=arguments.n_fem_boundary or defaults.n_fem_boundary,
            target_h=arguments.target_h or defaults.target_h,
            base_seed=arguments.seed,
            verify=not arguments.no_verify,
            noise_floor_stride=arguments.noise_floor_stride,
            reynolds_range=(
                tuple(arguments.reynolds_range)
                if arguments.reynolds_range is not None
                else None
            ),
            unseen_reynolds_range=(
                tuple(arguments.unseen_reynolds_range)
                if arguments.unseen_reynolds_range is not None
                else None
            ),
        )
    else:
        if (
            arguments.reynolds_range is not None
            or arguments.unseen_reynolds_range is not None
        ):
            parser.error("--reynolds-range applies to the ns_cavity_star family only")
        settings = GeneratorSettings(
            family=arguments.family,
            n_boundary=arguments.n_boundary,
            n_query=arguments.n_query,
            n_fem_boundary=arguments.n_fem_boundary or 2048,
            target_h=arguments.target_h or 0.02,
            base_seed=arguments.seed,
            verify=not arguments.no_verify,
        )
    directory = generate_dataset(
        family=arguments.family,
        n_cases=arguments.n_cases,
        version=arguments.version,
        workers=arguments.workers,
        settings=settings,
        root=arguments.root,
        created=datetime.date.today().isoformat(),
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "directory": str(directory),
                "n_cases": manifest["n_cases"],
                "splits": {
                    name: spec["stop"] - spec["start"]
                    for name, spec in manifest["splits"].items()
                },
                "verification": manifest["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
