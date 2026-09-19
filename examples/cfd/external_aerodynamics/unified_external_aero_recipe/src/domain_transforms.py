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

"""
Recipe-local DomainMesh-aware datapipe components for boundary-typed
surface tasks.

The recipe's stock surface pipeline reads a *single* boundary Mesh out of
each ``.pdmsh`` (``pattern: .../boundaries/vehicle``) and terminates in
``MeshToDomainMesh``, so every other typed boundary in the file (the
curated DrivAerML layout carries ``vehicle``, ``no_slip``, ``slip``,
``inlet``, ``outlet``) is dropped before the model ever sees it.  Models
that consume the full boundary-value problem instead read the whole
``DomainMesh`` -- which leaves these gaps this module fills:

- The stock ``DomainMeshReader`` applies its cell AND point subsample caps
  uniformly to every mesh; the point cap destroys most of a cell-carrying
  boundary's cells after the cell cap has already compacted it (the
  measured 378-cell vehicle starvation).
  :class:`TopologyAwareDomainMeshReader` applies the cell cap to
  cell-carrying meshes and the point cap to point clouds only.
- No stock transform converts a loaded ``DomainMesh`` (volume interior +
  many boundaries) into the recipe's *surface-task* contract (interior =
  one boundary's cell centroids carrying the targets) while keeping every
  boundary: ``MeshToDomainMesh.apply_to_domain`` is an identity
  passthrough.  :class:`BoundaryMeshToDomainMesh` adds exactly that
  domain-aware path.
- ``SetGlobalField`` only defines the ``Mesh`` path; its default
  ``apply_to_domain`` broadcast writes each sub-mesh's own ``global_data``
  and never the *domain-level* ``global_data`` that the recipe contract
  (and DomainMesh-native models) read.  :class:`SetDomainGlobalField`
  overrides the domain path to inject at the domain level.
- ``NonDimensionalizeByMetadata`` never touches ``global_data``, so a
  declared freestream drive would be the RAW physical vector.
  :class:`ComputeFreestreamDirection` derives the unit direction as a new
  global leaf (leaving ``U_inf`` intact for inference-side
  re-dimensionalization).

Recipe-local module registered into the global datapipe component registry
so the classes can be referenced via ``${dp:...}`` short names in Hydra
YAML configs.  Import this module before Hydra instantiation
(``src/datasets.py`` does this at import time, like :mod:`nondim` and
:mod:`sdf`).
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from warnings import warn

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.readers.mesh import DomainMeshReader, _subsample_mesh
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import (
    MeshToDomainMesh,
    RandomRotateMesh,
    SetGlobalField,
    SubsampleMesh,
)
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.mesh.transforms import (
    _compact_points,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.geometry import compute_cell_areas
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY, scale_measures


@register()
class TopologyAwareDomainMeshReader(DomainMeshReader):
    r"""``DomainMeshReader`` whose subsample caps respect mesh topology.

    The stock reader applies BOTH ``subsample_n_cells`` and
    ``subsample_n_points`` to EVERY mesh.  For a cell-carrying boundary
    that combination is destructive: the cell cap keeps ``n_cells`` cells
    and compacts unreferenced points (~3 unique points per cell when the
    on-disk cell order is shuffled, as in the curated DrivAerML files),
    and the subsequent point cap slices a contiguous point block that
    keeps only cells with ALL vertices inside it -- roughly
    ``fraction**3`` of the cells.  Measured on BC-labeled DrivAerML run_1
    at ``sampling_resolution=10000``: the 17.7M-cell vehicle boundary
    collapsed to 378 cells (~``(1/3)**3 * 10k``), silently starving the
    all-boundary arm's source set while the vehicle-only arm (cell cap
    only) fed the full 10k cells.

    This reader instead applies the CELL cap to meshes that have cells
    (the boundaries) and the POINT cap only to point clouds (the volume
    interior) -- the only combination that both bounds the 165.8M-point
    interior at read time and leaves boundary cell counts at the requested
    resolution.  Meshes below their cap pass through complete.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ### Take ownership of the caps: the parent's uniform subsample pass
        ### in __getitem__ is disabled (both attributes None) and the split
        ### rule below runs inside _load_sample instead.
        self._cell_mesh_n_cells = self.subsample_n_cells
        self._point_cloud_n_points = self.subsample_n_points
        self.subsample_n_cells = None
        self.subsample_n_points = None

    def _cap(self, mesh: Mesh, generator: torch.Generator | None) -> Mesh:
        if mesh.n_cells > 0:
            return _subsample_mesh(
                mesh,
                n_cells=self._cell_mesh_n_cells,
                n_points=None,
                generator=generator,
            )
        return _subsample_mesh(
            mesh,
            n_cells=None,
            n_points=self._point_cloud_n_points,
            generator=generator,
        )

    def _load_sample(self, index: int) -> DomainMesh:
        dm = super()._load_sample(index)
        ### Per-(seed, epoch, index) generator, matching the parent reader's
        ### post-#1742 derivation (the shared mutable `_subsample_generator`
        ### is gone; per-index spawning is what makes subsampling reproducible
        ### independent of read order and epoch -- the set_epoch drift fix).
        generator = (
            None
            if self._seed_base is None
            else spawn_generator(self._seed_base, self._epoch, index)
        )
        return DomainMesh(
            interior=self._cap(dm.interior, generator),
            boundaries={
                name: self._cap(dm.boundaries[name], generator)
                for name in dm.boundary_names
            },
            global_data=dm.global_data,
        )


@register()
class SetDomainGlobalField(SetGlobalField):
    r"""``SetGlobalField`` that injects into *domain-level* ``global_data``.

    Identical to :class:`~physicsnemo.datapipes.transforms.mesh.SetGlobalField`
    on a bare ``Mesh``.  On a ``DomainMesh`` the base class's default
    ``apply_to_domain`` broadcast would set every sub-mesh's own
    ``global_data`` while leaving ``DomainMesh.global_data`` -- the level
    the recipe contract reads freestream conditions from -- untouched.
    This override writes the constant fields at the domain level instead
    (sub-mesh ``global_data`` is left unchanged).

    Typical use: inject the constant scale gauge ``reference_length: 1.0``
    into an already-nondimensionalized DomainMesh pipeline (positions are
    scaled by ``L_ref`` in ``NonDimensionalizeByMetadata``, so the gauge in
    that space is exactly 1.0).
    """

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Set the configured domain-level global fields."""
        reference = domain.interior.points
        new_gd = domain.global_data.clone()
        new_gd.update(self._fields.to(device=reference.device, dtype=reference.dtype))
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=new_gd,
        )


@register()
class ComputeFreestreamDirection(MeshTransform):
    r"""Write the unit freestream direction into ``global_data``.

    Computes ``global_data[output_field] = U / |U|`` from the physical
    freestream vector ``global_data[velocity_field]`` and stores it as a NEW
    leaf; the physical vector itself is left untouched, so inference-side
    re-dimensionalization (which derives ``q_inf`` from ``U_inf``) and the
    force-coefficient integration keep reading the correct physical
    freestream.

    Why this exists: ``NonDimensionalizeByMetadata`` non-dimensionalizes
    fields and geometry but never ``global_data``, so models that declare a
    freestream as a global vector input would otherwise consume the RAW
    physical vector (~39 m/s for DrivAerML). A surrogate whose inputs are
    declared nondimensional must not see it: an architecture of this program
    that consumed the raw vector multiplied it straight through its
    score/value products (initial predictions O(1e9), float32 overflow on
    the all-boundary arm; measured on the DrivAerML tunnel probe, 2026-07).
    The unit direction plus the Cp/Cf-nondimensionalized targets is the
    AirFRANS-campaign convention (``freestream_direction``); for DrivAerML
    the discarded magnitude is a per-dataset constant, so no per-case
    information is lost.

    Place it before the augmentation insertion point so that
    ``RandomRotateMesh(transform_global_data=true)`` rotates the direction
    together with ``U_inf`` and the geometry (it rotates every ``(3,)``
    global leaf).

    Domain-aware: on a ``DomainMesh`` the direction is computed from and
    written to the *domain-level* ``global_data`` (the level the recipe
    contract and DomainMesh-native models read).
    """

    def __init__(
        self,
        velocity_field: str = "U_inf",
        output_field: str = "U_inf_dir",
    ) -> None:
        super().__init__()
        self._velocity_field = velocity_field
        self._output_field = output_field

    def _direction(self, global_data: TensorDict) -> torch.Tensor:
        if self._velocity_field not in global_data.keys():
            raise KeyError(
                f"ComputeFreestreamDirection: {self._velocity_field!r} not "
                f"found in global_data (available: "
                f"{sorted(global_data.keys())!r})."
            )
        velocity = global_data[self._velocity_field].float()
        norm = torch.linalg.vector_norm(velocity)
        if not torch.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"ComputeFreestreamDirection: |{self._velocity_field}| must "
                f"be finite and positive, got {norm.item()!r}."
            )
        return velocity / norm

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        new_gd[self._output_field] = self._direction(mesh.global_data)
        new_mesh = mesh.copy()  # ty: ignore[unresolved-attribute]
        new_mesh.global_data = new_gd
        return new_mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Derive the flow direction from domain-level velocity metadata."""
        new_gd = domain.global_data.clone()
        new_gd[self._output_field] = self._direction(domain.global_data)
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return (
            f"{self._output_field} = {self._velocity_field} / |{self._velocity_field}|"
        )


@register()
class SplitInteriorSupport(MeshTransform):
    r"""Move a deterministic prefix of the interior points into a ``support``
    boundary (transfer program D1, 2026-09-09).

    ISLA's ``support_tokens`` mode needs an interior *support* set that is a
    function of the case, not of the requested output points. The volume
    reader already draws the interior sample deterministically per case
    (``subsample_n_points`` with a per-index generator), so the first
    ``n_support`` interior points are a reproducible per-case set. This
    transform moves them, with the listed ``point_data`` fields (e.g. the
    signed distance and its gradient), into ``boundaries[boundary_name]`` as
    a points-only ``Mesh``; the remaining interior points, which keep every
    field including the targets, are the passive queries the loss scores.
    Run it *after* the SDF transform (so the support carries the SDF) and
    before target injection; the ``targets:`` block then attaches to the
    reduced interior only.

    At inference the same transform yields the same support for a case, so a
    sentinel-query study can hold the support fixed while the requested
    query set varies (see the eval skeleton in
    ``examples/cfd/mesh_transformer/research/transfer_program/studies/computational_support``).
    """

    def __init__(
        self,
        n_support: int,
        boundary_name: str = "support",
        point_data_fields: tuple[str, ...] = ("sdf", "sdf_normals"),
    ) -> None:
        super().__init__()
        if n_support <= 0:
            raise ValueError("n_support must be positive")
        self.n_support = int(n_support)
        self.boundary_name = boundary_name
        self.point_data_fields = tuple(point_data_fields)

    def __call__(
        self, mesh: Mesh
    ) -> Mesh:  # bare Mesh: identity (support needs a DomainMesh)
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Split interior samples into support inputs and prediction queries."""
        interior = domain.interior
        n = interior.points.shape[0]
        if n <= self.n_support:
            raise ValueError(
                f"SplitInteriorSupport: interior has {n} points, need more than n_support={self.n_support}"
            )
        idx = torch.arange(n, device=interior.points.device)
        support = interior.slice_points(idx[: self.n_support])
        queries = interior.slice_points(idx[self.n_support :])
        keep = {
            k: support.point_data[k]
            for k in self.point_data_fields
            if k in support.point_data.keys()
        }
        missing = [k for k in self.point_data_fields if k not in keep]
        if missing:
            raise KeyError(
                f"SplitInteriorSupport: interior point_data lacks {missing!r}"
            )
        support_mesh = Mesh(
            points=support.points,
            cells=support.cells,
            point_data=TensorDict(keep, batch_size=[support.points.shape[0]]),
            global_data=interior.global_data,
        )
        boundaries = (
            dict(domain.boundaries.items())
            if hasattr(domain.boundaries, "items")
            else {name: domain.boundaries[name] for name in domain.boundary_names}
        )
        if self.boundary_name in boundaries:
            raise KeyError(
                f"SplitInteriorSupport: boundary {self.boundary_name!r} already exists"
            )
        boundaries[self.boundary_name] = support_mesh
        return DomainMesh(
            interior=queries, boundaries=boundaries, global_data=domain.global_data
        )

    def extra_repr(self) -> str:
        return f"n_support={self.n_support}, boundary_name={self.boundary_name!r}, fields={self.point_data_fields!r}"


@register()
class BoundaryMeshToDomainMesh(MeshToDomainMesh):
    r"""``MeshToDomainMesh`` whose ``DomainMesh`` path re-targets one boundary.

    On a bare ``Mesh`` this behaves exactly like the base transform.  On a
    ``DomainMesh`` (where the base class is an identity passthrough) it
    rebuilds the recipe's surface-task contract *without dropping the other
    boundaries*:

    - ``interior`` becomes a point cloud at ``boundaries[boundary_name]``'s
      cell centroids (or vertices, per ``interior_points``), with the
      declared target fields moved into ``interior.point_data``;
    - ``boundaries[boundary_name]`` is kept with its target fields stripped
      (so consumers cannot read targets through the boundary);
    - every *other* boundary passes through unchanged -- this is what
      exposes the full typed-boundary set (``vehicle``, ``no_slip``,
      ``slip``, ``inlet``, ``outlet``) to boundary-typed models;
    - the previous (volume) interior is discarded;
    - domain-level ``global_data`` passes through unchanged.

    The class name deliberately ends in ``MeshToDomainMesh`` so the dataset
    builder's target auto-injection (``datasets._maybe_inject_targets``)
    fills ``cell_data_targets`` / ``point_data_targets`` from the dataset
    YAML's ``targets:`` block exactly as for the base transform.
    """

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:  # type: ignore[override]
        """Convert the selected boundary into a surface prediction domain."""
        available = list(domain.boundary_names)
        if self._boundary_name not in available:
            raise KeyError(
                f"BoundaryMeshToDomainMesh: boundary {self._boundary_name!r} "
                f"not found in DomainMesh (available: {available!r})."
            )
        ### Reuse the base single-Mesh conversion (including its diagonal
        ### validation and error messages) to split the named boundary into
        ### (interior-with-targets, boundary-without-targets)...
        converted = super().__call__(domain.boundaries[self._boundary_name])
        ### ...then keep every other boundary and the domain-level
        ### global_data (the converted DomainMesh's global_data is the
        ### boundary's own, which is not the case-level record).
        new_boundaries = {name: domain.boundaries[name] for name in available}
        new_boundaries[self._boundary_name] = converted.boundaries[self._boundary_name]
        return DomainMesh(
            interior=converted.interior,
            boundaries=new_boundaries,
            global_data=domain.global_data,
        )


@register()
class DropDegenerateCells(MeshTransform):
    r"""Drop cells with non-finite or degenerate current geometry.

    Recompute geometric measures from the current coordinates, in the mesh's
    dtype, using the same area routine as ``Mesh.cell_areas``. Its direct
    triangle area calculation preserves thin valid faces without Gram
    cancellation. Cached areas are ignored: centering, rotation, and scaling
    can collapse a face through rounding. Cells whose area is zero or
    non-finite in the mesh's dtype cannot supply usable quadrature weights.

    Place this last in the transform chain so it sees the same coordinates
    the model will. Meshes without rejected cells pass through unchanged.
    Only cells and their associated data are sliced; vertices are retained.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        cell_points = mesh.points[mesh.cells]
        edges = cell_points[:, 1:] - cell_points[:, :1]
        finite_points = torch.isfinite(cell_points).all(dim=(-2, -1))
        areas = compute_cell_areas(edges)
        keep = finite_points & torch.isfinite(areas) & (areas > 0)
        n_bad = int((~keep).sum())
        if n_bad == 0:
            return mesh
        warn(
            f"DropDegenerateCells: dropping {n_bad} cell(s) with "
            "non-finite or degenerate geometry"
        )
        return mesh.slice_cells(keep)


@register()
class PrefixPlusRandomSubsampleMesh(SubsampleMesh):
    r"""Deterministic-prefix + random-complement subsampling (probe P4).

    Keeps the first ``n_prefix`` cells in stable cell-index order (the same
    PHYSICAL cells for every draw of a given case) plus a seeded random
    sample of the remainder up to ``n_cells``. Two evaluations with
    different ``n_cells`` (or seeds) then share the prefix exactly, so
    prediction differences AT the prefix measure companion-set sensitivity
    -- the query-independence probe. HT measure bookkeeping is inherited
    for the random stage and set to 1 for the prefix (deterministic
    inclusion).
    """

    def __init__(self, n_prefix: int, n_cells: int, compact: bool = True):
        super().__init__(n_cells=n_cells, compact=compact)
        if n_prefix > n_cells:
            raise ValueError("n_prefix must be <= n_cells")
        self.n_prefix = int(n_prefix)

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n <= self.n_cells:
            return mesh
        device = mesh.cells.device
        prefix = torch.arange(self.n_prefix, device=device)
        n_rest = self.n_cells - self.n_prefix
        if n_rest > 0:
            pool = n - self.n_prefix
            generator = self._generator
            if generator is not None and generator.device != device:
                generator = None
            perm = torch.randperm(pool, device=device, generator=generator)
            rest = self.n_prefix + perm[:n_rest]
            indices = torch.cat([prefix, rest])
        else:
            indices = prefix
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        ### Prefix cells have inclusion probability 1; the random stage
        ### carries the usual inverse inclusion probability.
        w = torch.ones(len(indices), device=device)
        if n_rest > 0:
            w[self.n_prefix :] = (n - self.n_prefix) / n_rest
        scale_measures(mesh, w)
        return mesh


### ---------------------------------------------------------------------
### Contract-axis probe transforms (prereg: contract_axis_probes_2026-08-09)
### Job-local additions for the 2026-08-06-defect4-incumbent task dir.
### ---------------------------------------------------------------------


@register()
class ScaleGlobalField(MeshTransform):
    r"""Multiply one global_data field by a constant factor (probe P2).

    Applied to the global vector input only: ``U_inf_dir`` for ISLA (the
    recipe's unit direction, scaled to amplitude ``factor``) or ``U_inf`` for
    GeoTransolver (its physical-velocity global embedding).
    """

    def __init__(self, field_name: str, factor: float) -> None:
        super().__init__()
        self.field_name = field_name
        self.factor = float(factor)

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        new_gd[self.field_name] = new_gd[self.field_name] * self.factor
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return f"field_name={self.field_name!r}, factor={self.factor}"


@register()
class BiasedSubsampleMesh(SubsampleMesh):
    r"""Spatially biased cell subsampling with per-cell HT weights (probe P3).

    Cells with centroid x below the per-sample median draw with
    ``bias``:1 relative probability. Inclusion probabilities are composed
    into the measure weights per cell (pi_i ~= k * w_i / sum(w), the
    with-replacement approximation, adequate at k << N and recorded as an
    approximation in the preregistration), so a measure-consistent
    consumer sees an asymptotically unbiased quadrature while a
    measure-blind consumer sees a front-loaded point cloud.
    """

    def __init__(self, n_cells: int, bias: float = 10.0, compact: bool = True):
        super().__init__(n_cells=n_cells, compact=compact)
        self.bias = float(bias)

    def __call__(self, mesh: Mesh) -> Mesh:
        n_before = mesh.n_cells
        if n_before <= self.n_cells:
            return mesh
        x = mesh.cell_centroids[:, 0]
        w = torch.where(
            x < x.median(),
            torch.full_like(x, self.bias),
            torch.ones_like(x),
        )
        generator = self._generator
        if generator is not None and generator.device != w.device:
            generator = None
        indices = torch.multinomial(
            w, self.n_cells, replacement=False, generator=generator
        )
        pi = (self.n_cells * w[indices] / w.sum()).clamp(max=1.0)
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        scale_measures(mesh, 1.0 / pi)
        return mesh


@register()
class PoissonBiasedSubsampleMesh(MeshTransform):
    r"""Biased cell subsampling with EXACT per-cell HT weights (probe P3 v3).

    Independent (Poisson) sampling: cell i is kept with probability
    pi_i = min(1, c * w_i) where w_i is the sampling weight and c is set so
    the expected kept count is ``n_cells_expected``. Inclusion
    probabilities are exact by construction (no with-replacement
    approximation), at the cost of a variable per-sample cell count.

    ``weight_mode`` selects the weight:

    - ``"front_back"`` (default): the 10:1 (``bias``:1) front/back split
      on centroid x, the P3 density-bias probe.
    - ``"area"``: w_i = cell area (``mesh.cell_areas``), so before clamping
      pi_i is proportional to area -- the inclusion law a uniform
      point-cloud generator over the *surface* would produce, as opposed
      to the uniform-over-cells law that reads the mesher's refinement
      pattern (consistency benchmark, 2026-09-10). ``bias`` is unused.
    """

    _WEIGHT_MODES = ("front_back", "area")

    def __init__(
        self,
        n_cells_expected: int,
        bias: float = 10.0,
        compact: bool = True,
        weight_mode: str = "front_back",
    ) -> None:
        super().__init__()
        self.n_cells_expected = int(n_cells_expected)
        self.bias = float(bias)
        self.compact = compact
        if weight_mode not in self._WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {self._WEIGHT_MODES}, got {weight_mode!r}"
            )
        self.weight_mode = weight_mode
        self._generator: torch.Generator | None = None

    def _weights(self, mesh: Mesh) -> torch.Tensor:
        if self.weight_mode == "area":
            ### Degenerate (zero / non-finite) areas get weight 0: never
            ### drawn, so no 1/pi is ever formed for them.
            areas = mesh.cell_areas
            return torch.where(
                torch.isfinite(areas), areas, torch.zeros_like(areas)
            ).clamp_min(0.0)
        x = mesh.cell_centroids[:, 0]
        return torch.where(
            x < x.median(), torch.full_like(x, self.bias), torch.ones_like(x)
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n <= self.n_cells_expected:
            return mesh
        w = self._weights(mesh)
        c = self.n_cells_expected / w.sum()
        pi = (c * w).clamp(max=1.0)
        ### One renormalization pass restores the expected count lost to
        ### clamping (exactness of pi is what matters, not the count).
        deficit = self.n_cells_expected - pi.sum()
        if deficit > 0:
            free = pi < 1.0
            pi[free] = (pi[free] * (1 + deficit / pi[free].sum())).clamp(max=1.0)
        generator = self._generator
        if generator is not None and generator.device != pi.device:
            generator = None
        keep = torch.rand(n, device=pi.device, generator=generator) < pi
        indices = keep.nonzero(as_tuple=True)[0]
        kept_pi = pi[indices]
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        scale_measures(mesh, 1.0 / kept_pi)
        return mesh


@register()
class StratifiedSubsampleMesh(MeshTransform):
    r"""Uniform-over-cells subsampling with spatial stratification (QUAD-VAR, 2026-09-17).

    Same marginal inclusion law as :class:`SubsampleMesh` -- every cell is kept
    with probability ``n / N`` and carries the measure weight ``N / n`` -- but
    the ``n`` kept cells are a *systematic* sample along the Morton (Z-order)
    ordering of the cell centroids: one cell per consecutive run of ``N / n``
    cells, with a single random offset per draw. A systematic sample has the
    same expectation as an independent draw for every cell-weighted integral
    and a lower variance for integrands that are smooth over the surface, so
    it lowers the finite-sample noise of an aggregate at a fixed token budget
    without changing the sampling convention the model was trained under.
    The offset is drawn from the transform's generator when one is attached
    (the pipeline seeds it), so the draw is reproducible; no cell is ever
    drawn twice.
    """

    def __init__(self, n_cells: int, compact: bool = True, bits: int = 16) -> None:
        super().__init__()
        self.n_cells = int(n_cells)
        self.compact = compact
        if not 1 <= int(bits) <= 21:
            raise ValueError(f"bits must be in [1, 21], got {bits!r}")
        self.bits = int(bits)
        self._generator: torch.Generator | None = None

    def _morton_order(self, centroids: torch.Tensor) -> torch.Tensor:
        ### Quantize each axis to ``bits`` levels over the centroid bounding
        ### box and interleave the bits (x in bit 3b, y in 3b+1, z in 3b+2):
        ### a Z-order curve, so consecutive codes are spatial neighbours.
        lo = centroids.min(dim=0).values
        span = (centroids.max(dim=0).values - lo).clamp_min(1e-12)
        levels = 2**self.bits
        q = ((centroids - lo) / span * (levels - 1)).round().long().clamp(0, levels - 1)
        code = torch.zeros(
            centroids.shape[0], dtype=torch.int64, device=centroids.device
        )
        for b in range(self.bits):
            for axis in range(3):
                code |= ((q[:, axis] >> b) & 1) << (3 * b + axis)
        return torch.argsort(code)

    def __call__(self, mesh: Mesh) -> Mesh:
        n_before = mesh.n_cells
        if n_before <= self.n_cells:
            return mesh
        order = self._morton_order(mesh.cell_centroids)
        step = n_before / self.n_cells
        generator = self._generator
        if generator is not None and generator.device != order.device:
            generator = None
        offset = torch.rand((), device=order.device, generator=generator) * step
        ### floor(offset + k * step), k = 0..n-1: n distinct positions in [0, N)
        ### because step >= 1, each cell included with probability exactly n / N.
        positions = (
            torch.floor(
                offset
                + step
                * torch.arange(self.n_cells, device=order.device, dtype=torch.float64)
            )
            .long()
            .clamp(max=n_before - 1)
        )
        indices = order[positions].sort().values
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        scale_measures(mesh, n_before / self.n_cells)
        return mesh

    def extra_repr(self) -> str:
        return f"n_cells={self.n_cells}, bits={self.bits}"


def _pose_rotation_for_key(key: int, salt: int) -> torch.Tensor:
    """Uniform SO(3) rotation matrix drawn from a generator seeded by (key, salt).

    Same construction as ``RandomRotateMesh(mode="uniform")``: an isotropic
    Gaussian 4-vector normalized to a unit quaternion. Deterministic in
    ``key`` and ``salt``, independent of process, worker, epoch or shuffle
    order.
    """
    g = torch.Generator().manual_seed((int(key) * 1_000_003 + int(salt)) % (2**63 - 1))
    q = torch.randn(4, generator=g, dtype=torch.float64)
    q = q / q.norm()
    return RandomRotateMesh._quaternion_to_rotation_matrix(q)


@register()
class FixedRandomPose(MeshTransform):
    """Rotate a case by a uniform SO(3) rotation that is a deterministic function of the case.

    POSE-BENCH (transfer program, 2026-09-09). Every training and validation
    case receives an independent random pose, but the SAME pose every time
    the case is loaded, so that (a) the validation set is a fixed posed
    benchmark identical for every architecture, and (b) training on posed
    data is a data property, not an augmentation stream. The rotation is
    seeded by the reader's ``case_key`` global field
    (``MeshReaderWithGlobalData(store_case_key=True)``) and ``salt``.
    Positions, every vector field in point/cell/global data (e.g. wall shear,
    normals, ``U_inf``) rotate together; scalars are invariant. Place it
    BEFORE ``ComputeFreestreamDirection`` so the unit direction is computed
    from the rotated freestream. Deterministic: no generator, never reseeded.
    """

    def __init__(self, salt: int = 0, key_field: str = "case_key") -> None:
        super().__init__()
        self.salt = int(salt)
        self.key_field = key_field

    def _matrix(self, global_data: TensorDict, like: torch.Tensor) -> torch.Tensor:
        if self.key_field not in global_data.keys():
            raise KeyError(
                f"FixedRandomPose needs global_data[{self.key_field!r}]; enable "
                "store_case_key on MeshReaderWithGlobalData"
            )
        # The draw is made on a CPU generator (device-independent stream); the matrix
        # moves to the mesh's device, since the pipeline may run its transforms on GPU.
        return _pose_rotation_for_key(int(global_data[self.key_field]), self.salt).to(
            device=like.device, dtype=like.dtype
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        R = self._matrix(mesh.global_data, mesh.points)
        return mesh.transform(
            R,
            transform_point_data=True,
            transform_cell_data=True,
            transform_global_data=True,
            assume_invertible=True,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply the sample-specific fixed pose to geometry and vector fields."""
        gd = (
            domain.global_data
            if self.key_field in domain.global_data.keys()
            else domain.interior.global_data
        )
        R = self._matrix(gd, domain.interior.points)
        return domain.transform(
            R,
            transform_point_data=True,
            transform_cell_data=True,
            transform_global_data=True,
            assume_invertible=True,
        )

    def extra_repr(self) -> str:
        return f"salt={self.salt}, key_field={self.key_field!r}"


@register()
class SetConstantCellField(MeshTransform):
    r"""Write a constant per-cell scalar field (a per-case CONDITION input).

    ``cell_data[field_name] = value * ones(n_cells, 1)``.  The value is a
    property of the dataset the case comes from, so it is set per dataset
    YAML (e.g. ``0.0`` in the DrivAerML pipeline, ``1.0`` in the SHIFT-SUV
    pipelines) and rides along into ``boundaries.<name>.cell_data`` like
    ``normals``.  Transfer-program campaign C (T1, complete conditioning):
    lets ISLA read it as a boundary scalar (``n_boundary_scalars=1`` with
    ``forward_kwargs.boundary_scalars`` pointing at the field) and
    GeoTransolver / Transolver as an extra per-point functional channel
    (append the field to ``forward_kwargs.local_embedding`` and raise
    ``functional_dim`` by one).  A scalar is an invariant, so every ISLA
    covariance contract is untouched.  Place before ``MeshToDomainMesh``.
    """

    def __init__(self, field_name: str = "cond", value: float = 0.0) -> None:
        super().__init__()
        self._field_name = str(field_name)
        self._value = float(value)

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n == 0:
            raise ValueError(
                "SetConstantCellField: the mesh has no cells; place the "
                "transform before MeshToDomainMesh on a cell-carrying surface."
            )
        new_cd = mesh.cell_data.clone()
        new_cd[self._field_name] = torch.full(
            (n, 1), self._value, dtype=mesh.points.dtype, device=mesh.points.device
        )
        return mesh.with_data(cell_data=new_cd)

    def extra_repr(self) -> str:
        return f"cell_data[{self._field_name!r}] = {self._value}"


@register()
class SetGlobalFieldsFromTable(MeshTransform):
    r"""Per-case global fields from a JSON table keyed by case name.

    FRAME-FULL (2026-09-10): the frame of a surface (area-weighted centroid,
    RMS radius) is a property of the geometry, so it is computed once per
    case from the FULL mesh and passed in as ``global_data`` instead of being
    estimated from the sampled points: no sampling dependence, no estimator
    variance. The table is ``{case_name: {field: value, ...}, ...}`` (keys
    starting with ``_`` are provenance and ignored); the case is identified
    through ``global_data[key_field]``, the CRC-32 case key
    :class:`~merge_global_data.MeshReaderWithGlobalData` writes with
    ``store_case_key: true``, so every listed name is hashed the same way at
    construction (a hash collision among the listed names is an error).
    Fields are written in the mesh's dtype; place after
    ``NonDimensionalizeByMetadata`` and give the table's values in the same
    coordinates (the frame datasets omit ``CenterMesh``, whose shift is a
    sample statistic).
    """

    def __init__(
        self, table: str, fields: Sequence[str], key_field: str = "case_key"
    ) -> None:
        super().__init__()
        self._table_path = str(table)
        self._fields = tuple(fields)
        self._key_field = key_field
        with open(self._table_path) as f:
            raw = json.load(f)
        self._rows: dict[int, TensorDict] = {}
        for name, row in raw.items():
            if name.startswith("_"):
                continue
            key = zlib.crc32(name.encode()) & 0x7FFFFFFF
            if key in self._rows:
                raise ValueError(
                    f"SetGlobalFieldsFromTable: case-key collision for {name!r} in {table}"
                )
            self._rows[key] = TensorDict(
                {k: torch.as_tensor(row[k], dtype=torch.float64) for k in self._fields},
                batch_size=[],
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        if self._key_field not in mesh.global_data.keys():
            raise KeyError(
                f"SetGlobalFieldsFromTable: global_data[{self._key_field!r}] missing; "
                f"set store_case_key: true on the reader."
            )
        key = int(mesh.global_data[self._key_field])
        if key not in self._rows:
            raise KeyError(
                f"SetGlobalFieldsFromTable: case key {key} not in {self._table_path}"
            )
        new_gd = mesh.global_data.clone()
        new_gd.update(
            self._rows[key].to(device=mesh.points.device, dtype=mesh.points.dtype)
        )
        return mesh.with_data(global_data=new_gd)

    def extra_repr(self) -> str:
        return f"{list(self._fields)} from {self._table_path} ({len(self._rows)} cases) by {self._key_field}"


@register()
class SdfBiasedSubsampleInteriorPoints(MeshTransform):
    r"""Poisson-subsample the *interior point cloud* with an inclusion
    probability that rises with the signed distance to the wall (far-wake
    oversampling arm, 2026-09-10).

    Motivation: on DrivAerML the interior query-token model's one remaining
    eddy-viscosity deficit against GeoTransolver-volume sits in the far wake
    beyond 0.4 body lengths, which holds about 2% of a uniform interior
    sample, and slice anchors follow the query mass. This transform draws
    the training queries from a larger uniform pool with per-point inclusion
    probability ``pi_i = min(1, c * w(sdf_i))``, where ``w`` is a step
    function of the signed distance over ``band_edges`` with values
    ``band_weights`` and ``c`` sets the expected kept count to
    ``n_points_expected`` (one renormalization pass restores the count lost
    to clamping, as in :class:`PoissonBiasedSubsampleMesh`). Inclusion
    probabilities are exact and stored per kept point as
    ``point_data[pi_field]`` for provenance, so an importance-weighted loss
    can be built from them; this transform does not itself reweight the loss
    (the recipe loss is a plain mean over queries), so the objective's
    spatial weighting shifts toward the far field by construction. Run it
    after the SDF transform. Bare ``Mesh`` inputs pass through unchanged.
    """

    def __init__(
        self,
        n_points_expected: int,
        band_edges: Sequence[float] = (0.05, 0.4),
        band_weights: Sequence[float] = (1.0, 3.0, 8.0),
        sdf_field: str = "sdf",
        pi_field: str = "inclusion_pi",
    ) -> None:
        super().__init__()
        if n_points_expected <= 0:
            raise ValueError("n_points_expected must be positive")
        edges = [float(e) for e in band_edges]
        weights = [float(w) for w in band_weights]
        if len(weights) != len(edges) + 1:
            raise ValueError("band_weights must have one more entry than band_edges")
        if any(e2 <= e1 for e1, e2 in zip(edges, edges[1:])) or any(
            w <= 0 for w in weights
        ):
            raise ValueError(
                "band_edges must increase and band_weights must be positive"
            )
        self.n_points_expected = int(n_points_expected)
        self.band_edges = tuple(edges)
        self.band_weights = tuple(weights)
        self.sdf_field = sdf_field
        self.pi_field = pi_field
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator) -> None:
        """Set the random generator used for inclusion draws."""
        self._generator = generator

    def _weights(self, sdf: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.band_edges, dtype=sdf.dtype, device=sdf.device)
        band = torch.bucketize(sdf.abs(), edges)  # 0 .. len(edges)
        w = torch.tensor(self.band_weights, dtype=sdf.dtype, device=sdf.device)[band]
        return torch.where(torch.isfinite(sdf), w, torch.zeros_like(w))

    def _inclusion(self, sdf: torch.Tensor) -> torch.Tensor:
        w = self._weights(sdf)
        c = self.n_points_expected / w.sum()
        pi = (c * w).clamp(max=1.0)
        deficit = self.n_points_expected - pi.sum()
        if deficit > 0:
            free = pi < 1.0
            if bool(free.any()):
                pi[free] = (pi[free] * (1 + deficit / pi[free].sum())).clamp(max=1.0)
        return pi

    def __call__(
        self, mesh: Mesh
    ) -> Mesh:  # bare Mesh: identity (needs the DomainMesh interior)
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Subsample interior points and correct explicit quadrature measures."""
        interior = domain.interior
        n = interior.points.shape[0]
        if n <= self.n_points_expected:
            return domain
        if self.sdf_field not in interior.point_data.keys():
            raise KeyError(
                f"SdfBiasedSubsampleInteriorPoints: interior point_data lacks {self.sdf_field!r}; "
                "run the SDF transform first"
            )
        sdf = interior.point_data[self.sdf_field].reshape(n).to(torch.float32)
        pi = self._inclusion(sdf)
        generator = self._generator
        if generator is not None and generator.device != pi.device:
            generator = None
        keep = torch.rand(n, device=pi.device, generator=generator) < pi
        idx = keep.nonzero(as_tuple=True)[0]
        kept = interior.slice_points(idx)
        pd = kept.point_data.clone()
        pd[self.pi_field] = pi[idx].to(kept.points.dtype)[:, None]
        new_interior = Mesh(
            points=kept.points,
            cells=kept.cells,
            point_data=pd,
            global_data=interior.global_data,
        )
        if EFFECTIVE_MEASURE_KEY in new_interior.point_data:
            scale_measures(new_interior, 1.0 / pi[idx], association="points")
        return DomainMesh(
            interior=new_interior,
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"n_points_expected={self.n_points_expected}, band_edges={self.band_edges}, "
            f"band_weights={self.band_weights}, sdf_field={self.sdf_field!r}"
        )


@register()
class ComposeQuadratureMeasure(MeshTransform):
    """Materialize the shared effective cell measure for model field lookup.

    Readers and samplers have already applied their corrections through the
    mesh measure API. This adapter only ensures the complete measure is stored
    under the shared key; it does not normalize it or apply another factor.
    Meshes without cells are unchanged, including point-cloud interiors.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures

        set_cell_measures(mesh, cell_measures(mesh).to(mesh.points.dtype))
        return mesh
