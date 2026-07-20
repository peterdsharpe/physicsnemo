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

r"""Torch-native loader for the preprocessed AirFRANS catalog.

Reads the ``.npz`` cases written by :mod:`airfrans_preprocess` (the one-time
VTK converter, the only module of this example allowed to import PyVista;
this loader depends on numpy/torch only) and rebuilds the program's
:class:`~physicsnemo.mesh.DomainMesh` sample interface:

- one ``"airfoil"`` boundary (segment mesh, geometry only -- the no-slip
  condition is homogeneous, so the boundary carries **no** drive data),
- an interior query mesh whose point data are the three AirFRANS targets
  :math:`\Delta U/|U_\infty|` (rank 1), :math:`C_p` (rank 0), and
  :math:`\ln(1 + \nu_t/\nu)` (rank 0), and
- global data ``freestream_direction`` (the unit far-field velocity, the
  global rank-1 drive), ``log_reynolds``
  (:math:`\ln(|U_\infty| c / \nu)`, the single dimensionless global operator
  scalar of the pre-registration -- replacing GLOBE's two-reference-length
  construction with the model's intrinsic gauge plus one scalar), and
  ``viscous_scale`` (:math:`\mathrm{Re}_c^{-1/2}`, from the same Reynolds
  number: the declared auxiliary boundary-layer scale of the H4 arm, a
  rank-0 global operator field only where an arm declares it -- carrying
  the value in ``global_data`` is free for the arms that do not).

Catalog layout (written by the preprocessor)::

    <catalog>/
        manifest.json             # the AirFRANS distribution's, unchanged
        preprocess_manifest.json  # converter provenance (constants, masks)
        {case_name}.npz           # one file per case

Split resolution follows GLOBE (and the official AirFRANS manifest): the
manifest keys are ``"{task}_train"`` / ``"{task}_test"`` for the four tasks
``full`` / ``scarce`` / ``reynolds`` / ``aoa``, except that the ``scarce``
task defines no test list of its own and evaluates against ``full_test``.

Label pathologies arrive as NaN rows (all target fields NaN at a masked
point, GLOBE's convention); consumers exclude NaN target rows per field.

This is a benchmark-local research utility, not a proposed public API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from physicsnemo.mesh import DomainMesh, Mesh

# --- AirFRANS working constants -------------------------------------------
#
# NOTE: RHO = 1 is correct.  The AirFRANS authors report 1.204 kg/m^3 in
# places, but the OpenFOAM case files use 1, and the data itself confirms it:
# RHO = 1 yields the physically required constant far-field total pressure,
# RHO = 1.204 does not.  (Same verification as GLOBE's dataset adapter.)
RHO = 1.0  # kg/m^3
NU = 1.56e-5  # m^2/s
CHORD = 1.0  # m (every AirFRANS airfoil is unit chord)

#: The two published label-pathology masks, applied at preprocessing time by
#: setting ALL target fields at an offending point to NaN (whole-row
#: exclusion, identically for training and evaluation of every arm):
#: energy-conservation violations (:math:`C_{pt} > 1.02`) and near-wall
#: pressure-gradient artifacts (:math:`|\nabla C_p \cdot c| > 20`).
CPT_MASK_THRESHOLD = 1.02
GRAD_CP_MASK_THRESHOLD = 20.0

TASKS = ("full", "scarce", "reynolds", "aoa")

TARGET_FIELDS = ("delta_velocity", "pressure_coefficient", "log_nut_ratio")

#: Per-case array schema (name: (ndim, numpy dtype kind)), mirroring the
#: validation discipline of ``dataset_catalog``.
CASE_ARRAYS: dict[str, tuple[int, str]] = {
    "boundary_points": (2, "f"),
    "boundary_cells": (2, "i"),
    "query_points": (2, "f"),
    "delta_velocity": (2, "f"),
    "pressure_coefficient": (1, "f"),
    "log_nut_ratio": (1, "f"),
    "cpt": (1, "f"),
    "is_surface": (1, "b"),
    "u_inf": (1, "f"),
    "nu": (0, "f"),
    "chord": (0, "f"),
}


class AirFRANSCatalogError(RuntimeError):
    """A preprocessed AirFRANS catalog failed an integrity or schema check."""


@dataclass(frozen=True)
class AirFRANSCase:
    """One reloaded case: device/dtype-resolved tensors plus its constants.

    ``targets`` holds the three learning targets (NaN rows mark the points
    excluded by the published pathology masks); ``cpt`` is the diagnostic
    total-pressure coefficient (NaN at the same rows); ``u_inf`` stays
    float64 so the physical reconstructions used by the evaluation metrics
    (raw velocity, dynamic pressure) lose nothing to the storage cast.
    """

    name: str
    boundary_points: torch.Tensor  # (n_boundary, 2)
    boundary_cells: torch.Tensor  # (n_boundary, 2) int64
    query_points: torch.Tensor  # (n_query, 2)
    targets: dict[str, torch.Tensor]
    cpt: torch.Tensor  # (n_query,)
    is_surface: torch.Tensor  # (n_query,) bool
    u_inf: torch.Tensor  # (2,) float64
    nu: float
    chord: float

    @property
    def n_query(self) -> int:
        return self.query_points.shape[0]

    @property
    def u_inf_magnitude(self) -> float:
        return float(torch.linalg.vector_norm(self.u_inf))

    @property
    def dynamic_pressure(self) -> float:
        """Freestream dynamic pressure ``0.5 * RHO * |U_inf|^2`` (RHO = 1)."""

        return 0.5 * RHO * self.u_inf_magnitude**2

    @property
    def log_reynolds(self) -> float:
        """The global operator scalar ``ln(|U_inf| * chord / nu)``."""

        return math.log(self.u_inf_magnitude * self.chord / self.nu)

    @property
    def viscous_scale(self) -> float:
        """The declared auxiliary boundary-layer scale ``Re_c^(-1/2)``.

        The dimensionless lambda of the H4 contract (book/07-airfrans.qmd):
        at Re ~ 4e6 the boundary layer lives at ``delta/c ~ Re^(-1/2) ~
        5e-4``, and the aux-scale arm hands the kernel decoder exactly this
        ratio, computed from the same Reynolds number as ``log_reynolds``.
        """

        return math.exp(-0.5 * self.log_reynolds)


def load_manifest(directory: Path | str) -> dict:
    """Read the (pass-through) AirFRANS ``manifest.json`` from a catalog."""

    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise AirFRANSCatalogError(f"manifest not found: {path}")
    return json.loads(path.read_text())


def split_case_names(manifest: dict, task: str, split: str) -> list[str]:
    """Case names of one task split, with GLOBE's ``scarce`` resolution.

    ``split`` is ``"train"`` or ``"test"``.  The official manifest defines
    ``scarce_train`` but no ``scarce_test``; the scarce task trains on 200
    cases and is validated/evaluated against ``full_test`` (GLOBE's
    convention, mirrored here).
    """

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; available: {TASKS}")
    if split == "train":
        key = f"{task}_train"
    elif split == "test":
        key = f"{'full' if task == 'scarce' else task}_test"
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if key not in manifest:
        raise AirFRANSCatalogError(f"manifest defines no {key!r} list")
    return list(manifest[key])


def _validate_case_arrays(arrays: dict[str, np.ndarray]) -> list[str]:
    """Return a list of schema problems (empty when the case is well-formed)."""

    problems: list[str] = []
    for name, (ndim, kind) in CASE_ARRAYS.items():
        if name not in arrays:
            problems.append(f"missing array {name!r}")
            continue
        value = arrays[name]
        if value.ndim != ndim or value.dtype.kind != kind:
            problems.append(
                f"array {name!r} must have ndim={ndim} and dtype kind {kind!r}, "
                f"got ndim={value.ndim}, dtype={value.dtype}"
            )
    if problems:
        return problems

    n_boundary = arrays["boundary_points"].shape[0]
    if arrays["boundary_points"].shape[1] != 2:
        problems.append("boundary_points must have two columns")
    if arrays["boundary_cells"].shape != (n_boundary, 2):
        problems.append(
            "boundary_cells must have shape (n_boundary, 2) "
            "(closed loops: one segment per point)"
        )
    cells = arrays["boundary_cells"]
    if cells.size and (cells.min() < 0 or cells.max() >= n_boundary):
        problems.append("boundary_cells reference out-of-range points")
    n_query = arrays["query_points"].shape[0]
    if arrays["query_points"].shape[1] != 2:
        problems.append("query_points must have two columns")
    if arrays["delta_velocity"].shape != (n_query, 2):
        problems.append("delta_velocity must have shape (n_query, 2)")
    problems.extend(
        f"array {name!r} must have one row per query point"
        for name in ("pressure_coefficient", "log_nut_ratio", "cpt", "is_surface")
        if arrays[name].shape != (n_query,)
    )
    if arrays["u_inf"].shape != (2,):
        problems.append("u_inf must have shape (2,)")
    problems.extend(
        f"array {name!r} contains non-finite values"
        for name in ("boundary_points", "query_points", "u_inf", "nu", "chord")
        if not np.isfinite(arrays[name]).all()
    )
    return problems


def load_case(
    directory: Path | str,
    name: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> AirFRANSCase:
    """Load one preprocessed case ``<directory>/<name>.npz``.

    Geometry and targets are cast to ``dtype`` (training precision);
    ``u_inf`` is kept float64 for exact physical reconstructions in the
    metrics.  NaN target rows (the published pathology masks) pass through
    untouched.
    """

    path = Path(directory) / f"{name}.npz"
    if not path.is_file():
        raise AirFRANSCatalogError(f"case file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    problems = _validate_case_arrays(arrays)
    if problems:
        raise AirFRANSCatalogError(
            f"case file {path} is malformed: " + "; ".join(problems)
        )
    device = torch.device("cpu") if device is None else torch.device(device)

    def tensor(key: str, target_dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arrays[key])).to(
            device=device, dtype=target_dtype
        )

    return AirFRANSCase(
        name=name,
        boundary_points=tensor("boundary_points", dtype),
        boundary_cells=tensor("boundary_cells", torch.int64),
        query_points=tensor("query_points", dtype),
        targets={key: tensor(key, dtype) for key in TARGET_FIELDS},
        cpt=tensor("cpt", dtype),
        is_surface=tensor("is_surface", torch.bool),
        u_inf=tensor("u_inf", torch.float64),
        nu=float(arrays["nu"]),
        chord=float(arrays["chord"]),
    )


def case_domain(
    case: AirFRANSCase,
    *,
    n_queries: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[DomainMesh, torch.Tensor]:
    """Build the model-facing :class:`DomainMesh` for one case.

    Returns ``(domain, query_indices)``.  With ``n_queries=None`` (the
    evaluation convention) every query point is used in stored order; with
    ``n_queries=n`` a fresh uniform subsample of ``min(n, n_query)`` points
    is drawn per call from ``generator`` (a CPU :class:`torch.Generator`,
    the training convention: 4,096 volume queries per iteration,
    GLOBE-matched).  The full airfoil boundary is used every time -- the
    protocol declares no boundary downsampling.

    The domain carries the three targets as interior point data (aligned
    with ``query_indices``); trainers strip them before the forward pass.
    Global data are the unit ``freestream_direction`` (rank-1 drive; never
    zero, so the zero-drive contract is vacuous here), ``log_reynolds``
    (operator scalar), and ``viscous_scale`` (``Re_c^(-1/2)``, the declared
    auxiliary scale consumed only by arms that declare it as an operator
    field).  No ``reference_length`` is declared: the arms run on the
    model's intrinsic gauge per the pre-registration.
    """

    n_total = case.n_query
    if n_queries is None or n_queries >= n_total:
        indices = torch.arange(n_total, device=case.query_points.device)
    else:
        if n_queries < 1:
            raise ValueError("n_queries must be a positive integer or None")
        permutation = torch.randperm(n_total, generator=generator)
        indices = permutation[:n_queries].to(case.query_points.device)

    boundary = Mesh(points=case.boundary_points, cells=case.boundary_cells)
    interior = Mesh(
        points=case.query_points[indices],
        point_data={key: value[indices] for key, value in case.targets.items()},
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={"airfoil": boundary},
        global_data=_case_global_data(case),
    )
    return domain, indices


def _case_global_data(case: AirFRANSCase) -> dict[str, torch.Tensor]:
    """The case's global drive/operator data (shared by both query builds)."""

    dtype = case.query_points.dtype
    device = case.query_points.device
    direction = case.u_inf / torch.linalg.vector_norm(case.u_inf)
    return {
        "freestream_direction": direction.to(device=device, dtype=dtype),
        "log_reynolds": torch.tensor(case.log_reynolds, device=device, dtype=dtype),
        "viscous_scale": torch.tensor(case.viscous_scale, device=device, dtype=dtype),
    }


def surface_vertex_query_indices(case: AirFRANSCase) -> torch.Tensor:
    """Query-row index of each boundary vertex (exact-coincidence bijection).

    Real AirFRANS catalogs carry every airfoil-surface volume-mesh node as a
    query row, and those nodes ARE the boundary polyline's vertices (measured
    on the v1 catalog: ``n_surface == n_boundary_points`` with bitwise
    point coincidence, max distance exactly 0).  Returns a ``(n_boundary,)``
    ``torch.long`` tensor whose element ``j`` is the query row whose point
    equals ``boundary_points[j]`` bitwise.  The match is exact by
    construction -- both arrays are cast from the same float64 sources --
    so no nearest-neighbor tolerance is involved; any failure of the
    bijection (a missing vertex, a duplicate match, or a count mismatch)
    raises :class:`AirFRANSCatalogError` loudly rather than guessing.
    """

    surface_rows = torch.nonzero(case.is_surface, as_tuple=False).squeeze(1)
    boundary = case.boundary_points.detach().cpu().numpy()
    queries = case.query_points[surface_rows].detach().cpu().numpy()
    if surface_rows.numel() != boundary.shape[0]:
        raise AirFRANSCatalogError(
            f"case {case.name!r}: the surface task requires the vertex<->query "
            f"bijection, but {int(surface_rows.numel())} surface query rows != "
            f"{boundary.shape[0]} boundary vertices"
        )
    by_bytes = {queries[i].tobytes(): int(surface_rows[i]) for i in range(len(queries))}
    if len(by_bytes) != len(queries):
        raise AirFRANSCatalogError(
            f"case {case.name!r}: duplicate surface query points break the "
            "vertex<->query bijection"
        )
    rows = []
    for j in range(boundary.shape[0]):
        row = by_bytes.get(boundary[j].tobytes())
        if row is None:
            raise AirFRANSCatalogError(
                f"case {case.name!r}: boundary vertex {j} has no bitwise-"
                "coincident surface query row; the catalog does not satisfy "
                "the surface task's exact vertex<->query contract"
            )
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64, device=case.query_points.device)


def surface_case_domain(case: AirFRANSCase) -> DomainMesh:
    r"""Panel-centroid surface companion domain (the trace-mode task).

    The query mesh is the airfoil panels' **centroids**, one per boundary
    cell in cell order -- exactly the whole-mesh identity map that
    ``trace_of="airfoil"`` declares (query ``i`` is cell ``i``), so both the
    trace arm and its control decode the same query set.  The single target
    is the surface pressure coefficient interpolated vertex-to-centroid as
    the :math:`\tfrac12`-average of the panel's two endpoint values (via the
    exact vertex<->query bijection of
    :func:`surface_vertex_query_indices`); the midpoint rule is
    second-order in the panel length (~1.2e-3 chord on the v1 catalog), and
    a NaN-masked vertex propagates NaN to both adjacent centroids (the
    consumers' NaN-row exclusion then drops them, unchanged).  Full panel
    set every call -- the protocol declares no boundary downsampling, and
    the trace contract forbids subsampling by construction.
    """

    vertex_rows = surface_vertex_query_indices(case)
    cells = case.boundary_cells
    centroids = 0.5 * (
        case.boundary_points[cells[:, 0]] + case.boundary_points[cells[:, 1]]
    )
    vertex_cp = case.targets["pressure_coefficient"][vertex_rows]
    centroid_cp = 0.5 * (vertex_cp[cells[:, 0]] + vertex_cp[cells[:, 1]])
    boundary = Mesh(points=case.boundary_points, cells=cells)
    interior = Mesh(
        points=centroids,
        point_data={"pressure_coefficient": centroid_cp},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"airfoil": boundary},
        global_data=_case_global_data(case),
    )


def point_segment_distances(
    points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Minimum distance from each point to a set of 2D segments.

    Exact point-to-segment distance (projection parameter clamped to
    ``[0, 1]``), vectorized over a ``(points, segments)`` grid and chunked
    over points so the ~180k-query x ~1k-segment AirFRANS case stays within
    a few hundred MB of workspace on GPU.  Returns ``(n_points,)`` in the
    input dtype.
    """

    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if segment_starts.shape != segment_ends.shape or segment_starts.shape[-1] != 2:
        raise ValueError("segments must have matching (s, 2) endpoints")
    direction = segment_ends - segment_starts  # (s, 2)
    length_squared = (direction * direction).sum(dim=-1).clamp_min(1.0e-30)  # (s,)
    minima: list[torch.Tensor] = []
    for chunk in points.split(chunk_size):
        offset = chunk[:, None, :] - segment_starts[None, :, :]  # (n, s, 2)
        parameter = (offset * direction[None, :, :]).sum(dim=-1) / length_squared
        parameter = parameter.clamp(0.0, 1.0)
        nearest = segment_starts[None, :, :] + parameter[..., None] * direction
        distances = torch.linalg.vector_norm(chunk[:, None, :] - nearest, dim=-1)
        minima.append(distances.min(dim=1).values)
    return torch.cat(minima)


def boundary_distances(case: AirFRANSCase, *, chunk_size: int = 8192) -> torch.Tensor:
    """Distance from every query point to the airfoil boundary polyline."""

    starts = case.boundary_points[case.boundary_cells[:, 0]]
    ends = case.boundary_points[case.boundary_cells[:, 1]]
    return point_segment_distances(
        case.query_points, starts, ends, chunk_size=chunk_size
    )


__all__ = [
    "AirFRANSCase",
    "AirFRANSCatalogError",
    "CASE_ARRAYS",
    "CHORD",
    "CPT_MASK_THRESHOLD",
    "GRAD_CP_MASK_THRESHOLD",
    "NU",
    "RHO",
    "TARGET_FIELDS",
    "TASKS",
    "boundary_distances",
    "case_domain",
    "load_case",
    "load_manifest",
    "point_segment_distances",
    "split_case_names",
    "surface_case_domain",
    "surface_vertex_query_indices",
]
