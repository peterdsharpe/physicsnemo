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
that consume the full boundary-value problem (e.g. the ``MeshTransformer``
all-boundary arm) instead read the whole ``DomainMesh`` -- which leaves
these gaps this module fills:

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

from warnings import warn

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.readers.mesh import DomainMeshReader, _subsample_mesh
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import (
    MeshToDomainMesh,
    SetGlobalField,
    SubsampleMesh,
)
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.datapipes.transforms.mesh.transforms import (
    SubsampleMesh,
    _compact_points,
)
from physicsnemo.mesh.calculus.measure import compose_measure_weights


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
    freestream drive would otherwise consume the RAW physical vector
    (~39 m/s for DrivAerML).  The MeshTransformer's contract requires
    declared fields to be nondimensional, and its zero-preserving drive
    stream is norm-free by construction -- a 39x oversized drive multiplies
    straight through the score/value products (measured on the DrivAerML
    tunnel probe: raw-drive predictions at initialization are O(1e9) on the
    vehicle-only arm and overflow float32 to NaN on the all-boundary arm).
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
    r"""Drop cells whose fp32 area is non-finite or non-positive.

    DrivAerML surfaces contain sliver cells with areas down to ~1e-11 --
    deep enough in cross-product cancellation territory that any fp32
    coordinate perturbation (centering, rotation, device-specific
    evaluation order) can round the recomputed area to exact zero, which
    downstream measure validation rightly rejects. Place this LAST in the
    transform chain so it sees exactly the points the model will: the
    same tensor, device, and kernel produce the same areas at encode.
    Dropped cells carry ~1e-12 of the total measure, so the effective
    quadrature is unchanged to fp32 precision.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        areas = mesh.cell_areas
        keep = torch.isfinite(areas) & (areas > 0)
        n_bad = int((~keep).sum())
        if n_bad == 0:
            return mesh
        warn(
            f"DropDegenerateCells: dropping {n_bad} cell(s) with "
            "non-finite or non-positive fp32 area"
        )
        return mesh.slice_cells(keep.nonzero(as_tuple=True)[0])


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
        compose_measure_weights(mesh, w)
        return mesh
