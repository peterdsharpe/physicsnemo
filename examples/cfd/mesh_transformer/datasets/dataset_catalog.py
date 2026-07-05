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

r"""Save, load, and validate cataloged boundary-to-interior benchmark datasets.

Layout
------

::

    datasets/<family>/<version>/
        manifest.json
        case_00000.npz
        case_00001.npz
        ...

Each ``case_%05d.npz`` stores, losslessly, the arrays in
:data:`REQUIRED_ARRAYS` (benchmark boundary discretization with per-cell
geometry, per-cell Dirichlet values, interior query points, and the
solver-verified interior solution at the queries) plus a ``__params__`` JSON
string holding the case's family parameters (seeds, geometry coefficients,
trace coefficients, solver settings, per-case verification numbers).

``manifest.json`` records the family name, version, generator settings,
split definitions as case-index ranges, solver settings, aggregated
verification statistics, seeds, creation date, and a SHA-256 checksum per
case file.  :func:`validate_catalog` re-derives everything checkable.

The loader (:func:`load_domain_sample`) reconstructs the benchmark's
:class:`~physicsnemo.mesh.DomainMesh` sample interface — one ``"dirichlet"``
boundary carrying ``boundary_value`` cell data, an interior query mesh
carrying the ``potential`` target as point data, and ``reference_length``
global data — mirroring ``liouville.build_liouville_sample`` so cataloged
cases can drive the existing training and evaluation loops unmodified.

This is a benchmark-local research utility, not a proposed public API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA_VERSION = 1

CATALOG_ROOT = Path(__file__).resolve().parent
"""Default on-disk location of cataloged datasets."""

REQUIRED_ARRAYS: dict[str, tuple[int, str]] = {
    # name: (ndim, numpy dtype kind)
    "boundary_points": (2, "f"),
    "boundary_cells": (2, "i"),
    "boundary_loop_offsets": (1, "i"),
    "boundary_cell_centroids": (2, "f"),
    "boundary_cell_normals": (2, "f"),
    "boundary_cell_measures": (1, "f"),
    "boundary_value": (1, "f"),
    "query_points": (2, "f"),
    "u_query": (1, "f"),
}

#: Multi-field Navier-Stokes schema: the scalar Dirichlet trace becomes a
#: rank-1 ``boundary_velocity`` drive, the scalar target becomes velocity
#: (rank 1) plus pressure (rank 0), and every query carries its distance to
#: the boundary so evaluations can be bucketed into interior vs near-wall.
NS_REQUIRED_ARRAYS: dict[str, tuple[int, str]] = {
    "boundary_points": (2, "f"),
    "boundary_cells": (2, "i"),
    "boundary_loop_offsets": (1, "i"),
    "boundary_cell_centroids": (2, "f"),
    "boundary_cell_normals": (2, "f"),
    "boundary_cell_measures": (1, "f"),
    "boundary_velocity": (2, "f"),
    "query_points": (2, "f"),
    "query_wall_distance": (1, "f"),
    "velocity_query": (2, "f"),
    "pressure_query": (1, "f"),
}

#: Per-family array schema; families not listed use :data:`REQUIRED_ARRAYS`.
FAMILY_ARRAY_SCHEMAS: dict[str, dict[str, tuple[int, str]]] = {
    "ns_cavity_star": NS_REQUIRED_ARRAYS,
}

_PARAMS_KEY = "__params__"

MANIFEST_REQUIRED_KEYS = (
    "schema_version",
    "family",
    "version",
    "created",
    "n_cases",
    "seeds",
    "generator_settings",
    "solver_settings",
    "splits",
    "verification",
    "checksums",
)


class CatalogError(RuntimeError):
    """A cataloged dataset failed an integrity or schema check."""


@dataclass(frozen=True)
class CatalogCase:
    """One reloaded case: raw arrays plus its JSON family parameters."""

    index: int
    arrays: dict[str, np.ndarray]
    params: dict


def catalog_dir(family: str, version: str, root: Path | str | None = None) -> Path:
    """Return ``<root>/<family>/<version>`` (root defaults to the catalog)."""

    base = CATALOG_ROOT if root is None else Path(root)
    return base / family / version


def case_filename(index: int) -> str:
    """Canonical case file name for a case index."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("case index must be a non-negative integer")
    return f"case_{index:05d}.npz"


def _array_schema(family: str | None) -> dict[str, tuple[int, str]]:
    """Array schema for one family (default: the scalar Laplace schema)."""

    if family is None:
        return REQUIRED_ARRAYS
    return FAMILY_ARRAY_SCHEMAS.get(family, REQUIRED_ARRAYS)


def _validate_arrays(
    arrays: dict[str, np.ndarray], family: str | None = None
) -> list[str]:
    """Return a list of schema problems (empty when the case is well-formed)."""

    schema = _array_schema(family)
    problems: list[str] = []
    for name, (ndim, kind) in schema.items():
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

    per_boundary = ["boundary_cells"] + [
        name
        for name in (
            "boundary_cell_centroids",
            "boundary_cell_normals",
            "boundary_cell_measures",
            "boundary_value",
            "boundary_velocity",
        )
        if name in schema
    ]
    per_query = [
        name
        for name in (
            "u_query",
            "velocity_query",
            "pressure_query",
            "query_wall_distance",
        )
        if name in schema
    ]
    two_column = [
        name
        for name in (
            "boundary_points",
            "boundary_cell_centroids",
            "boundary_cell_normals",
            "boundary_velocity",
            "velocity_query",
        )
        if name in schema
    ]

    n_boundary = arrays["boundary_points"].shape[0]
    problems.extend(
        f"array {name!r} must have {n_boundary} rows to match "
        "boundary_points (closed loops: one cell per point)"
        for name in per_boundary
        if arrays[name].shape[0] != n_boundary
    )
    problems.extend(
        f"array {name!r} must have two columns"
        for name in two_column
        if arrays[name].shape[1] != 2
    )
    if arrays["boundary_cells"].shape[1] != 2:
        problems.append("boundary_cells must have two columns (line cells)")
    offsets = arrays["boundary_loop_offsets"]
    if (
        offsets.shape[0] < 2
        or offsets[0] != 0
        or offsets[-1] != n_boundary
        or np.any(np.diff(offsets) < 3)
    ):
        problems.append(
            "boundary_loop_offsets must start at 0, end at n_boundary, and "
            "delimit loops of at least three points"
        )
    if arrays["query_points"].shape[1] != 2:
        problems.append("query_points must have two columns")
    problems.extend(
        f"array {name!r} must have one row per query point"
        for name in per_query
        if arrays[name].shape[0] != arrays["query_points"].shape[0]
    )
    cells = arrays["boundary_cells"]
    if cells.size and (cells.min() < 0 or cells.max() >= n_boundary):
        problems.append("boundary_cells reference out-of-range points")
    problems.extend(
        f"array {name!r} contains non-finite values"
        for name in schema
        if arrays[name].dtype.kind == "f" and not np.isfinite(arrays[name]).all()
    )
    return problems


def save_case(
    directory: Path | str,
    index: int,
    arrays: dict[str, np.ndarray],
    params: dict,
) -> Path:
    """Write one schema-validated case file and return its path."""

    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    problems = _validate_arrays(arrays, params.get("family"))
    if problems:
        raise CatalogError(
            f"refusing to save malformed case {index}: " + "; ".join(problems)
        )
    encoded = json.dumps(params, sort_keys=True)  # raises on non-serializable
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / case_filename(index)
    np.savez(path, **arrays, **{_PARAMS_KEY: np.str_(encoded)})
    return path


def load_case(directory: Path | str, index: int) -> CatalogCase:
    """Reload one case file (arrays bit-identical to what was saved)."""

    path = Path(directory) / case_filename(index)
    if not path.is_file():
        raise CatalogError(f"case file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != _PARAMS_KEY}
        if _PARAMS_KEY not in data.files:
            raise CatalogError(f"case file {path} is missing its parameters")
        params = json.loads(str(data[_PARAMS_KEY]))
    problems = _validate_arrays(arrays, params.get("family"))
    if problems:
        raise CatalogError(f"case file {path} is malformed: " + "; ".join(problems))
    return CatalogCase(index=index, arrays=arrays, params=params)


def sha256_of_file(path: Path | str) -> str:
    """SHA-256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Path | str, manifest: dict) -> Path:
    """Validate manifest keys and write ``manifest.json``."""

    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing:
        raise CatalogError(f"manifest is missing required keys: {missing}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(directory: Path | str) -> dict:
    """Read ``manifest.json`` from a catalog directory."""

    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise CatalogError(f"manifest not found: {path}")
    return json.loads(path.read_text())


def validate_catalog(directory: Path | str) -> dict:
    """Re-derive every checkable property of one cataloged dataset version.

    Checks the manifest schema, the split index ranges (must partition
    ``[0, n_cases)`` without gaps or overlaps), the presence and SHA-256
    checksum of every case file, and the per-case array schema, including
    that each case's recorded split matches the manifest ranges.

    Returns a small summary dict; raises :class:`CatalogError` on any
    failure, listing every detected problem.
    """

    directory = Path(directory)
    manifest = load_manifest(directory)
    problems: list[str] = []
    missing_keys = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing_keys:
        raise CatalogError(f"manifest is missing required keys: {missing_keys}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest['schema_version']} != {SCHEMA_VERSION}"
        )

    n_cases = manifest["n_cases"]
    splits = manifest["splits"]
    ranges = sorted(
        (spec["start"], spec["stop"], name) for name, spec in splits.items()
    )
    cursor = 0
    for start, stop, name in ranges:
        if start != cursor or stop <= start:
            problems.append(
                f"split {name!r} range [{start}, {stop}) does not tile [0, n_cases)"
            )
        cursor = stop
    if cursor != n_cases:
        problems.append(f"split ranges cover [0, {cursor}), expected [0, {n_cases})")

    def split_of(index: int) -> str:
        for name, spec in splits.items():
            if spec["start"] <= index < spec["stop"]:
                return name
        return ""

    checksums = manifest["checksums"]
    for index in range(n_cases):
        filename = case_filename(index)
        path = directory / filename
        if filename not in checksums:
            problems.append(f"manifest has no checksum for {filename}")
        if not path.is_file():
            problems.append(f"case file missing: {filename}")
            continue
        if filename in checksums and sha256_of_file(path) != checksums[filename]:
            problems.append(f"checksum mismatch for {filename}")
            continue
        try:
            case = load_case(directory, index)
        except CatalogError as error:
            problems.append(str(error))
            continue
        recorded_split = case.params.get("split")
        if recorded_split != split_of(index):
            problems.append(
                f"case {index} records split {recorded_split!r} but the manifest "
                f"places it in {split_of(index)!r}"
            )
    if len(checksums) != n_cases:
        problems.append(
            f"manifest lists {len(checksums)} checksums for {n_cases} cases"
        )

    if problems:
        raise CatalogError(
            f"catalog validation failed for {directory}:\n- " + "\n- ".join(problems)
        )
    return {
        "family": manifest["family"],
        "version": manifest["version"],
        "n_cases": n_cases,
        "splits": {name: spec["stop"] - spec["start"] for name, spec in splits.items()},
    }


def split_indices(manifest: dict, split: str) -> range:
    """Case-index range of one named split."""

    if split not in manifest["splits"]:
        raise KeyError(
            f"unknown split {split!r}; available: {sorted(manifest['splits'])}"
        )
    spec = manifest["splits"][split]
    return range(spec["start"], spec["stop"])


def iter_split(directory: Path | str, split: str) -> Iterator[CatalogCase]:
    """Yield every case of one split, in index order."""

    manifest = load_manifest(directory)
    for index in split_indices(manifest, split):
        yield load_case(directory, index)


def load_domain_sample(
    case: CatalogCase,
    *,
    device=None,
    dtype=None,
):
    """Rebuild the benchmark ``DomainMesh`` sample interface from one case.

    Mirrors ``liouville.build_liouville_sample``: the returned domain has a
    single ``"dirichlet"`` boundary (all loops concatenated; per-loop extents
    are in ``boundary_loop_offsets``) with ``boundary_value`` cell data, an
    interior mesh whose points are the query points and whose ``potential``
    point data is the solver-verified target, and ``reference_length``
    global data.  Returns ``(domain, target)`` with ``target`` the interior
    potential tensor, so cataloged cases can drive the existing training and
    evaluation loops unmodified.
    """

    import torch

    from physicsnemo.mesh import DomainMesh, Mesh

    device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.float32 if dtype is None else dtype
    arrays = case.arrays

    boundary = Mesh(
        points=torch.from_numpy(np.ascontiguousarray(arrays["boundary_points"])).to(
            device=device, dtype=dtype
        ),
        cells=torch.from_numpy(np.ascontiguousarray(arrays["boundary_cells"])).to(
            device=device, dtype=torch.int64
        ),
        cell_data={
            "boundary_value": torch.from_numpy(
                np.ascontiguousarray(arrays["boundary_value"])
            ).to(device=device, dtype=dtype)
        },
    )
    target = torch.from_numpy(np.ascontiguousarray(arrays["u_query"])).to(
        device=device, dtype=dtype
    )
    interior = Mesh(
        points=torch.from_numpy(np.ascontiguousarray(arrays["query_points"])).to(
            device=device, dtype=dtype
        ),
        point_data={"potential": target},
    )
    reference_length = float(case.params.get("reference_length", 1.0))
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": torch.tensor(
                reference_length, device=device, dtype=dtype
            )
        },
    )
    return domain, target


def load_ns_domain_sample(
    case: CatalogCase,
    *,
    device=None,
    dtype=None,
):
    """Rebuild the multi-field N-S ``DomainMesh`` sample from one case.

    The returned domain has a single ``"dirichlet"`` boundary carrying the
    rank-1 ``boundary_velocity`` drive as cell data, an interior mesh whose
    points are the query points and whose ``velocity``/``pressure`` point
    data are the solver-verified targets, and global data
    ``reference_length`` (1.0 for this family), ``viscosity`` (the
    dimensionless PDE coefficient :math:`\\tilde\\nu = 1/\\mathrm{Re}`, the
    declared global operator scalar), and ``reynolds`` (recorded for
    reporting; bijective with viscosity).  Returns ``(domain, targets)``
    with ``targets = {"velocity": (n_q, 2), "pressure": (n_q,)}``.
    """

    import torch

    from physicsnemo.mesh import DomainMesh, Mesh

    device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.float32 if dtype is None else dtype
    arrays = case.arrays

    def tensor(name: str, kind=None):
        target_dtype = torch.int64 if kind == "i" else dtype
        return torch.from_numpy(np.ascontiguousarray(arrays[name])).to(
            device=device, dtype=target_dtype
        )

    boundary = Mesh(
        points=tensor("boundary_points"),
        cells=tensor("boundary_cells", "i"),
        cell_data={"boundary_velocity": tensor("boundary_velocity")},
    )
    targets = {
        "velocity": tensor("velocity_query"),
        "pressure": tensor("pressure_query"),
    }
    interior = Mesh(points=tensor("query_points"), point_data=dict(targets))
    reynolds = float(case.params["reynolds"])
    reference_length = float(case.params.get("reference_length", 1.0))
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": torch.tensor(
                reference_length, device=device, dtype=dtype
            ),
            "viscosity": torch.tensor(1.0 / reynolds, device=device, dtype=dtype),
            "reynolds": torch.tensor(reynolds, device=device, dtype=dtype),
        },
    )
    return domain, targets


__all__ = [
    "CATALOG_ROOT",
    "CatalogCase",
    "CatalogError",
    "FAMILY_ARRAY_SCHEMAS",
    "MANIFEST_REQUIRED_KEYS",
    "NS_REQUIRED_ARRAYS",
    "REQUIRED_ARRAYS",
    "SCHEMA_VERSION",
    "case_filename",
    "catalog_dir",
    "iter_split",
    "load_case",
    "load_domain_sample",
    "load_ns_domain_sample",
    "load_manifest",
    "save_case",
    "sha256_of_file",
    "split_indices",
    "validate_catalog",
    "write_manifest",
]
