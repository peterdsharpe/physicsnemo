# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

r"""A mesh-native global transformer for boundary-driven PDE surrogates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from tensordict import TensorDict

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.mesh import (
    DomainMesh,
    FieldLayout,
    Mesh,
    RankSpecDict,
    ScalarVectorFields,
    flatten_rank_spec,
    validate_rank_spec,
)

from .attention import AttentionMoments, ScalarVectorState, TypedProjection
from .block import (
    GeometryConditionedLinear,
    LinearMeshFieldBlock,
    MeshOperatorBlock,
    NonlinearZeroMeshFieldBlock,
    PointwiseGeometryBlock,
)

FieldMode = Literal["linear", "zero_preserving_nonlinear"]
FieldRoleRanks: TypeAlias = dict[str, RankSpecDict]
_FIELD_ROLES = ("operator", "drive")


@dataclass
class MetaData(ModelMetaData):
    """Runtime capabilities of :class:`MeshTransformer`."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = False


@dataclass(frozen=True)
class EncodedBoundary:
    r"""Reusable boundary state returned by :meth:`MeshTransformer.encode`.

    This is the sole public cache object.  It binds the dimensionless source
    quadrature, encoded operator and drive states, source frame, domain-level
    operator/drive data, query-block source moments, and the default query
    mesh.  It intentionally contains no tree, neighbourhood, or
    query-dependent state.
    """

    source_mesh: Mesh
    operator_state: ScalarVectorState
    drive_state: ScalarVectorState
    center: torch.Tensor
    reference_length: torch.Tensor
    global_operator_state: ScalarVectorState
    global_drive_state: ScalarVectorState
    query_moments: tuple[AttentionMoments, ...]
    query_mesh: Mesh
    global_data: TensorDict


def _role_spec(spec: FieldRoleRanks, role: str, *, label: str) -> RankSpecDict:
    if not isinstance(spec, dict):
        raise TypeError(f"{label} must be a dict, got {type(spec).__name__}")
    unexpected = set(spec) - set(_FIELD_ROLES)
    if unexpected:
        raise ValueError(
            f"{label} contains unknown field roles {sorted(unexpected)}; "
            f"expected only {_FIELD_ROLES}"
        )
    value = spec.get(role, {})
    validate_rank_spec(
        value,
        allowed_ranks=(0, 1),
        source_label=f"{label}[{role!r}]",
    )
    return value


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        relation = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}, got {value}")


def _rank_entries(
    rank_spec: RankSpecDict,
    path: tuple[str, ...] = (),
) -> list[tuple[str, tuple[str, ...], int]]:
    entries: list[tuple[str, tuple[str, ...], int]] = []
    for key, value in rank_spec.items():
        leaf_path = (*path, key)
        if isinstance(value, dict):
            entries.extend(_rank_entries(value, leaf_path))
        else:
            entries.append((".".join(leaf_path), leaf_path, value))
    return entries


def _td_get(data: TensorDict, path: tuple[str, ...]) -> torch.Tensor:
    key: str | tuple[str, ...] = path[0] if len(path) == 1 else path
    try:
        return data[key]
    except KeyError:
        raise ValueError(f"Missing declared field {'.'.join(path)!r}") from None


class MeshTransformer(Module):
    r"""Global similarity-covariant transformer on a :class:`DomainMesh`.

    The source tokens are codimension-one boundary cells with geometric
    measure.  The query tokens are ``domain.interior.points``.  All learned
    interactions are signed, separable Galerkin integrals; no graph edge,
    neighbour radius, Fourier coordinate, or absolute Cartesian component is
    part of the model.

    Parameters
    ----------
    n_spatial_dims : int
        Ambient dimension ``D``.
    output_field_ranks : RankSpecDict
        Named point predictions, with rank 0 for invariant scalars and rank 1
        for polar vectors.
    boundary_field_ranks : dict[str, FieldRoleRanks]
        Schema for each boundary-condition name.  Each value may contain an
        ``"operator"`` mapping (geometry/material conditioning) and a
        ``"drive"`` mapping (boundary data whose zero defines the homogeneous
        zero-input test).  Declared boundary fields are read from cell data.
    global_field_ranks : FieldRoleRanks, optional
        Domain-level operator and drive fields read from
        ``DomainMesh.global_data``.
    reference_length_key : str or None, optional
        Global-data leaf containing the positive physical reference length.
        Coordinates and codimension-one measures are divided by ``L`` and
        ``L**(D-1)``.  ``None`` means coordinates are already dimensionless;
        it does not estimate a data-dependent length.
    field_mode : {"linear", "zero_preserving_nonlinear"}
        ``linear`` guarantees fixed-geometry superposition.  The nonlinear
        mode guarantees zero drive produces zero output but does not claim
        superposition.
    query_chunk_size : int, default=65536
        Maximum number of independent query points decoded together.
    attention_chunk_size : int or None, default=65536
        Maximum entities passed through a typed attention projection at once.
        Chunking changes temporary memory, not the moment operator. ``None``
        disables projection chunking.

    Notes
    -----
    All declared physical fields are expected to be nondimensional.  Rank-1
    leaves are polar vectors and must be transformed together with the mesh
    under rotations or reflections.  Axial vectors and higher tensor types
    require a future representation extension and are rejected rather than
    silently treated as channels.
    """

    def __init__(
        self,
        n_spatial_dims: int,
        output_field_ranks: RankSpecDict,
        boundary_field_ranks: dict[str, FieldRoleRanks],
        global_field_ranks: FieldRoleRanks | None = None,
        reference_length_key: str | None = None,
        field_mode: FieldMode = "linear",
        operator_scalar_dim: int = 32,
        operator_vector_dim: int = 8,
        drive_scalar_dim: int = 64,
        drive_vector_dim: int = 16,
        operator_layers: int = 3,
        drive_layers: int = 2,
        query_layers: int | None = None,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        query_chunk_size: int = 65536,
        attention_chunk_size: int | None = 65536,
    ) -> None:
        _require_int("n_spatial_dims", n_spatial_dims, minimum=2)
        if not isinstance(boundary_field_ranks, dict):
            raise TypeError("boundary_field_ranks must be a dict")
        if not boundary_field_ranks:
            raise ValueError("boundary_field_ranks must declare at least one boundary")
        if global_field_ranks is not None and not isinstance(global_field_ranks, dict):
            raise TypeError("global_field_ranks must be a dict or None")
        if reference_length_key is not None and (
            not isinstance(reference_length_key, str) or not reference_length_key
        ):
            raise TypeError("reference_length_key must be a non-empty string or None")
        if field_mode not in ("linear", "zero_preserving_nonlinear"):
            raise ValueError(
                "field_mode must be 'linear' or 'zero_preserving_nonlinear'"
            )
        for name, value, minimum in (
            ("operator_scalar_dim", operator_scalar_dim, 1),
            ("operator_vector_dim", operator_vector_dim, 1),
            ("drive_scalar_dim", drive_scalar_dim, 1),
            ("drive_vector_dim", drive_vector_dim, 1),
            ("operator_layers", operator_layers, 0),
            ("drive_layers", drive_layers, 0),
            ("heads", heads, 1),
            ("scalar_rank", scalar_rank, 0),
            ("vector_rank", vector_rank, 0),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            _require_int(name, value, minimum=minimum)
        if attention_chunk_size is not None:
            _require_int("attention_chunk_size", attention_chunk_size, minimum=1)
        if query_layers is None:
            query_layers = 1 if field_mode == "linear" else 2
        _require_int("query_layers", query_layers, minimum=1)
        if scalar_rank + vector_rank == 0:
            raise ValueError("at least one attention rank must be positive")

        validate_rank_spec(
            output_field_ranks,
            allowed_ranks=(0, 1),
            source_label="output_field_ranks",
        )
        if not flatten_rank_spec(output_field_ranks):
            raise ValueError("output_field_ranks must contain at least one field")

        global_field_ranks = {} if global_field_ranks is None else global_field_ranks
        boundary_names = sorted(boundary_field_ranks)
        for name in boundary_names:
            if not isinstance(name, str) or not name:
                raise ValueError("boundary condition names must be non-empty strings")
            for role in _FIELD_ROLES:
                _role_spec(
                    boundary_field_ranks[name],
                    role,
                    label=f"boundary_field_ranks[{name!r}]",
                )
            operator_names = set(
                flatten_rank_spec(boundary_field_ranks[name].get("operator", {}))
            )
            drive_names = set(
                flatten_rank_spec(boundary_field_ranks[name].get("drive", {}))
            )
            if overlap := operator_names & drive_names:
                raise ValueError(
                    f"Boundary {name!r} fields cannot have both operator and "
                    f"drive roles: {sorted(overlap)}"
                )
        for role in _FIELD_ROLES:
            _role_spec(global_field_ranks, role, label="global_field_ranks")
        global_operator_names = set(
            flatten_rank_spec(global_field_ranks.get("operator", {}))
        )
        global_drive_names = set(flatten_rank_spec(global_field_ranks.get("drive", {})))
        if overlap := global_operator_names & global_drive_names:
            raise ValueError(
                "Global fields cannot have both operator and drive roles: "
                f"{sorted(overlap)}"
            )
        if reference_length_key is not None:
            declared_global_names = global_operator_names | global_drive_names
            if reference_length_key in declared_global_names:
                raise ValueError(
                    "reference_length_key is used only for geometric "
                    "nondimensionalization and must not also be a learned field"
                )

        # Freeze caller-owned mutable schemas before constructing layouts or
        # checkpoint metadata. Public configuration must not drift away from
        # the modules if a caller later edits their original dictionaries.
        output_field_ranks = deepcopy(output_field_ranks)
        boundary_field_ranks = deepcopy(boundary_field_ranks)
        global_field_ranks = deepcopy(global_field_ranks)

        super().__init__(meta=MetaData())
        self._args["output_field_ranks"] = deepcopy(output_field_ranks)
        self._args["boundary_field_ranks"] = deepcopy(boundary_field_ranks)
        self._args["global_field_ranks"] = deepcopy(global_field_ranks)
        self.n_spatial_dims = n_spatial_dims
        self.output_field_ranks = output_field_ranks
        self.boundary_field_ranks = boundary_field_ranks
        self.global_field_ranks = global_field_ranks
        self.reference_length_key = reference_length_key
        self.field_mode = field_mode
        self.operator_scalar_dim = operator_scalar_dim
        self.operator_vector_dim = operator_vector_dim
        self.drive_scalar_dim = drive_scalar_dim
        self.drive_vector_dim = drive_vector_dim
        self.operator_layers = operator_layers
        self.drive_layers = drive_layers
        self.query_layers = query_layers
        self.heads = heads
        self.scalar_rank = scalar_rank
        self.vector_rank = vector_rank
        self.query_chunk_size = query_chunk_size
        self.attention_chunk_size = attention_chunk_size
        self.boundary_names = tuple(boundary_names)

        self._boundary_layouts: dict[str, dict[str, FieldLayout | None]] = {
            role: {} for role in _FIELD_ROLES
        }
        self._boundary_names_by_rank: dict[str, dict[int, tuple[str, ...]]] = {}
        for role in _FIELD_ROLES:
            union: dict[str, int] = {}
            for name in boundary_names:
                rank_spec = _role_spec(
                    boundary_field_ranks[name],
                    role,
                    label=f"boundary_field_ranks[{name!r}]",
                )
                flat = flatten_rank_spec(rank_spec)
                for field_name, rank in flat.items():
                    previous = union.get(field_name)
                    if previous is not None and previous != rank:
                        raise ValueError(
                            f"Boundary field {field_name!r} has conflicting ranks "
                            f"{previous} and {rank}"
                        )
                    union[field_name] = rank
                self._boundary_layouts[role][name] = (
                    FieldLayout(rank_spec, n_spatial_dims) if flat else None
                )
            self._boundary_names_by_rank[role] = {
                rank: tuple(
                    sorted(name for name, value in union.items() if value == rank)
                )
                for rank in (0, 1)
            }

        self._global_entries = {
            role: tuple(
                sorted(
                    _rank_entries(
                        _role_spec(
                            global_field_ranks,
                            role,
                            label="global_field_ranks",
                        )
                    ),
                    key=lambda item: item[0],
                )
            )
            for role in _FIELD_ROLES
        }

        boundary_operator_scalars = len(self._boundary_names_by_rank["operator"][0])
        boundary_operator_vectors = len(self._boundary_names_by_rank["operator"][1])
        global_operator_scalars = sum(
            rank == 0 for _, _, rank in self._global_entries["operator"]
        )
        global_operator_vectors = sum(
            rank == 1 for _, _, rank in self._global_entries["operator"]
        )
        # BC one-hot + (source, query) association indicators.
        raw_operator_scalars = (
            boundary_operator_scalars
            + global_operator_scalars
            + len(boundary_names)
            + 2
        )
        # Boundary/global vectors + normalized position + source normal.
        raw_operator_vectors = boundary_operator_vectors + global_operator_vectors + 2
        self.operator_lift = TypedProjection(
            raw_operator_scalars,
            raw_operator_vectors,
            operator_scalar_dim,
            operator_vector_dim,
            scalar_bias=True,
        )
        # A shared nonlinear typed feature map gives source and query
        # coordinates a rich finite-rank basis before global interaction.
        # It is pointwise (not a neighbourhood heuristic); boundary-wide
        # information still enters only through the Galerkin moments below.
        self.operator_input_block = PointwiseGeometryBlock(
            operator_scalar_dim, operator_vector_dim
        )
        self.operator_blocks = nn.ModuleList(
            [
                MeshOperatorBlock(
                    operator_scalar_dim,
                    operator_vector_dim,
                    heads=heads,
                    scalar_rank=scalar_rank,
                    vector_rank=vector_rank,
                    entity_chunk_size=attention_chunk_size,
                )
                for _ in range(operator_layers)
            ]
        )

        boundary_drive_scalars = len(self._boundary_names_by_rank["drive"][0])
        boundary_drive_vectors = len(self._boundary_names_by_rank["drive"][1])
        self._boundary_drive_scalars = boundary_drive_scalars
        self._boundary_drive_vectors = boundary_drive_vectors
        global_drive_scalars = sum(
            rank == 0 for _, _, rank in self._global_entries["drive"]
        )
        global_drive_vectors = sum(
            rank == 1 for _, _, rank in self._global_entries["drive"]
        )
        raw_drive_scalars = boundary_drive_scalars + global_drive_scalars
        raw_drive_vectors = boundary_drive_vectors + global_drive_vectors
        if raw_drive_scalars + raw_drive_vectors == 0:
            raise ValueError(
                "At least one boundary or global field must have the 'drive' role"
            )
        self.drive_lift = GeometryConditionedLinear(
            operator_scalar_dim,
            operator_vector_dim,
            raw_drive_scalars,
            raw_drive_vectors,
            drive_scalar_dim,
            drive_vector_dim,
        )

        block_type = (
            LinearMeshFieldBlock
            if field_mode == "linear"
            else NonlinearZeroMeshFieldBlock
        )
        block_kwargs = dict(
            geometry_scalar_dim=operator_scalar_dim,
            geometry_vector_dim=operator_vector_dim,
            field_scalar_dim=drive_scalar_dim,
            field_vector_dim=drive_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank,
            entity_chunk_size=attention_chunk_size,
        )
        self.drive_blocks = nn.ModuleList(
            [block_type(**block_kwargs) for _ in range(drive_layers)]
        )
        self.query_blocks = nn.ModuleList(
            [block_type(**block_kwargs) for _ in range(query_layers)]
        )

        self.output_layout = FieldLayout(output_field_ranks, n_spatial_dims)
        self.output_projection = GeometryConditionedLinear(
            operator_scalar_dim,
            operator_vector_dim,
            drive_scalar_dim,
            drive_vector_dim,
            self.output_layout.n_scalars,
            self.output_layout.n_vectors,
        )

    def _validate_domain(self, domain: DomainMesh) -> None:
        if not isinstance(domain, DomainMesh):
            raise TypeError(f"domain must be a DomainMesh, got {type(domain).__name__}")
        actual_names = set(domain.boundaries.keys())
        expected_names = set(self.boundary_names)
        if actual_names != expected_names:
            raise ValueError(
                "Domain boundary names must exactly match the model schema; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        if domain.interior.n_spatial_dims != self.n_spatial_dims:
            raise ValueError(
                f"Expected {self.n_spatial_dims} spatial dimensions, got "
                f"{domain.interior.n_spatial_dims}"
            )
        if domain.interior.points.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Mesh geometry must use float32 or float64; mixed-precision "
                "execution should use autocast rather than reduced-precision points"
            )
        for name in self.boundary_names:
            mesh = domain.boundaries[name]
            if mesh.codimension != 1:
                raise ValueError(
                    f"Boundary {name!r} must be codimension one, got "
                    f"codimension={mesh.codimension}"
                )
            if mesh.n_cells == 0:
                raise ValueError(f"Boundary {name!r} must contain at least one cell")
            if mesh.points.device != domain.interior.points.device:
                raise ValueError("All boundary and query meshes must share a device")
            if mesh.points.dtype != domain.interior.points.dtype:
                raise ValueError("All boundary and query meshes must share a dtype")

    def _pack_boundary_role(
        self,
        domain: DomainMesh,
        role: str,
    ) -> ScalarVectorState:
        scalar_names = self._boundary_names_by_rank[role][0]
        vector_names = self._boundary_names_by_rank[role][1]
        scalar_index = {name: index for index, name in enumerate(scalar_names)}
        vector_index = {name: index for index, name in enumerate(vector_names)}
        scalar_parts: list[torch.Tensor] = []
        vector_parts: list[torch.Tensor] = []

        for boundary_name in self.boundary_names:
            mesh = domain.boundaries[boundary_name]
            scalars = mesh.points.new_zeros(mesh.n_cells, len(scalar_names))
            vectors = mesh.points.new_zeros(
                mesh.n_cells, len(vector_names), self.n_spatial_dims
            )
            layout = self._boundary_layouts[role][boundary_name]
            if layout is not None:
                packed = layout.pack(mesh.cell_data)
                if packed.scalars.dtype != mesh.points.dtype:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must have "
                        f"dtype {mesh.points.dtype}"
                    )
                if packed.scalars.device != mesh.points.device:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must be on "
                        f"{mesh.points.device}"
                    )
                for local, field_name in enumerate(layout.scalar_names):
                    scalars[:, scalar_index[field_name]] = packed.scalars[:, local]
                for local, field_name in enumerate(layout.vector_names):
                    vectors[:, vector_index[field_name], :] = packed.vectors[
                        :, local, :
                    ]
            scalar_parts.append(scalars)
            vector_parts.append(vectors)
        return ScalarVectorState(
            torch.cat(scalar_parts, dim=0), torch.cat(vector_parts, dim=0)
        )

    def _pack_global_role(
        self,
        global_data: TensorDict,
        role: str,
        n_entities: int,
        reference: torch.Tensor,
    ) -> ScalarVectorState:
        scalars: list[torch.Tensor] = []
        vectors: list[torch.Tensor] = []
        for name, path, rank in self._global_entries[role]:
            value = _td_get(global_data, path)
            expected = () if rank == 0 else (self.n_spatial_dims,)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Global {role} field {name!r} is rank {rank} and must "
                    f"have shape {expected}, got {tuple(value.shape)}"
                )
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(
                    f"Global {role} field {name!r} must share mesh device and dtype"
                )
            if rank == 0:
                scalars.append(value.expand(n_entities))
            else:
                vectors.append(value.expand(n_entities, -1))
        return ScalarVectorState(
            torch.stack(scalars, dim=-1)
            if scalars
            else reference.new_empty(n_entities, 0),
            torch.stack(vectors, dim=1)
            if vectors
            else reference.new_empty(n_entities, 0, self.n_spatial_dims),
        )

    def _reference_length(
        self,
        global_data: TensorDict,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.reference_length_key is None:
            return reference.new_ones(())
        path = tuple(self.reference_length_key.split("."))
        length = _td_get(global_data, path)
        if length.numel() != 1:
            raise ValueError(
                f"Reference length {self.reference_length_key!r} must be scalar"
            )
        if length.device != reference.device or length.dtype != reference.dtype:
            raise ValueError("Reference length must share mesh device and dtype")
        length = length.reshape(())
        if not torch.compiler.is_compiling() and (
            not torch.isfinite(length).item() or length.item() <= 0.0
        ):
            raise ValueError("Reference length must be finite and positive")
        return length

    def _source_operator_input(
        self,
        domain: DomainMesh,
        source_mesh: Mesh,
        boundary_operator: ScalarVectorState,
        global_operator: ScalarVectorState,
    ) -> ScalarVectorState:
        n = source_mesh.n_cells
        bc_one_hot = source_mesh.points.new_zeros(n, len(self.boundary_names))
        offset = 0
        for index, name in enumerate(self.boundary_names):
            count = domain.boundaries[name].n_cells
            bc_one_hot[offset : offset + count, index] = 1.0
            offset += count
        association = source_mesh.points.new_zeros(n, 2)
        association[:, 0] = 1.0
        return ScalarVectorState(
            torch.cat(
                (
                    boundary_operator.scalars,
                    global_operator.scalars,
                    bc_one_hot,
                    association,
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    boundary_operator.vectors,
                    global_operator.vectors,
                    source_mesh.cell_centroids[:, None, :],
                    source_mesh.cell_normals[:, None, :],
                ),
                dim=1,
            ),
        )

    def _query_operator_input(
        self,
        points: torch.Tensor,
        global_operator: ScalarVectorState,
    ) -> ScalarVectorState:
        n = points.shape[0]
        boundary_scalars = points.new_zeros(
            n, len(self._boundary_names_by_rank["operator"][0])
        )
        boundary_vectors = points.new_zeros(
            n,
            len(self._boundary_names_by_rank["operator"][1]),
            self.n_spatial_dims,
        )
        bc_one_hot = points.new_zeros(n, len(self.boundary_names))
        association = points.new_zeros(n, 2)
        association[:, 1] = 1.0
        query_normal = points.new_zeros(n, 1, self.n_spatial_dims)
        return ScalarVectorState(
            torch.cat(
                (
                    boundary_scalars,
                    global_operator.scalars.expand(n, -1),
                    bc_one_hot,
                    association,
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    boundary_vectors,
                    global_operator.vectors.expand(n, -1, -1),
                    points[:, None, :],
                    query_normal,
                ),
                dim=1,
            ),
        )

    def encode(self, domain: DomainMesh) -> EncodedBoundary:
        r"""Encode a boundary once for reuse at one or more query meshes."""
        self._validate_domain(domain)
        geometry_meshes = [
            domain.boundaries[name].with_data(
                point_data={}, cell_data={}, global_data={}
            )
            for name in self.boundary_names
        ]
        merged = Mesh.merge(geometry_meshes)
        length = self._reference_length(domain.global_data, merged.points)
        # Geometry and quadrature construction stay outside ambient AMP. The
        # learned projections may autocast, but centering, normals, and source
        # measure are numerical mesh operations and retain the input geometry
        # precision.
        with torch.autocast(device_type=merged.points.device.type, enabled=False):
            weights = merged.cell_areas
            total_measure = weights.sum()
        if not torch.compiler.is_compiling() and (
            not torch.isfinite(total_measure).item() or total_measure.item() <= 0.0
        ):
            raise ValueError("Boundary measure must be finite and positive")
        with torch.autocast(device_type=merged.points.device.type, enabled=False):
            center = torch.einsum("n,nd->d", weights, merged.cell_centroids)
            center = center / total_measure
            source_mesh = Mesh(
                points=(merged.points - center) / length,
                cells=merged.cells,
            )
            # Populate the immutable geometric cache at full geometry precision
            # before learned layers are entered under any outer autocast scope.
            _ = source_mesh.cell_centroids
            _ = source_mesh.cell_areas
            _ = source_mesh.cell_normals

        boundary_operator = self._pack_boundary_role(domain, "operator")
        global_operator = self._pack_global_role(
            domain.global_data,
            "operator",
            source_mesh.n_cells,
            source_mesh.points,
        )
        operator = self.operator_input_block(
            self.operator_lift(
                self._source_operator_input(
                    domain,
                    source_mesh,
                    boundary_operator,
                    global_operator,
                )
            )
        )
        for block in self.operator_blocks:
            operator = block(source_mesh, operator)

        boundary_drive = self._pack_boundary_role(domain, "drive")
        global_drive = self._pack_global_role(
            domain.global_data,
            "drive",
            source_mesh.n_cells,
            source_mesh.points,
        )
        raw_drive = ScalarVectorState(
            torch.cat((boundary_drive.scalars, global_drive.scalars), dim=-1),
            torch.cat((boundary_drive.vectors, global_drive.vectors), dim=1),
        )
        drive = self.drive_lift(operator, raw_drive)
        for block in self.drive_blocks:
            drive = block(source_mesh, operator, drive)

        global_operator_single = self._pack_global_role(
            domain.global_data,
            "operator",
            1,
            source_mesh.points,
        )
        global_drive_single = self._pack_global_role(
            domain.global_data,
            "drive",
            1,
            source_mesh.points,
        )
        query_moments = tuple(
            block.build_source_moments(source_mesh, operator, drive)
            for block in self.query_blocks
        )
        query_mesh = domain.interior.with_data(
            point_data={},
            cell_data={},
            global_data=domain.global_data,
        )
        return EncodedBoundary(
            source_mesh=source_mesh,
            operator_state=operator,
            drive_state=drive,
            center=center,
            reference_length=length,
            global_operator_state=global_operator_single,
            global_drive_state=global_drive_single,
            query_moments=query_moments,
            query_mesh=query_mesh,
            global_data=domain.global_data.copy(),
        )

    def decode(
        self,
        encoded: EncodedBoundary,
        query_mesh: Mesh | None = None,
    ) -> Mesh:
        r"""Evaluate an encoded boundary at arbitrary query mesh points."""
        if not isinstance(encoded, EncodedBoundary):
            raise TypeError("encoded must be an EncodedBoundary returned by encode")
        query_mesh = encoded.query_mesh if query_mesh is None else query_mesh
        if not isinstance(query_mesh, Mesh):
            raise TypeError(
                f"query_mesh must be a Mesh, got {type(query_mesh).__name__}"
            )
        if query_mesh.n_spatial_dims != self.n_spatial_dims:
            raise ValueError("query_mesh has the wrong spatial dimension")
        if (
            query_mesh.points.device != encoded.center.device
            or query_mesh.points.dtype != encoded.center.dtype
        ):
            raise ValueError("query_mesh must share encoded boundary device and dtype")

        if len(encoded.query_moments) != len(self.query_blocks):
            raise ValueError(
                "EncodedBoundary query moments do not match this decoder depth"
            )
        scalar_outputs: list[torch.Tensor] = []
        vector_outputs: list[torch.Tensor] = []
        n_queries = query_mesh.n_points
        starts = range(0, n_queries, self.query_chunk_size)
        slices = [
            slice(start, min(start + self.query_chunk_size, n_queries))
            for start in starts
        ]
        if not slices:
            slices = [slice(0, 0)]

        for chunk in slices:
            normalized_points = (
                query_mesh.points[chunk] - encoded.center
            ) / encoded.reference_length
            query_operator = self.operator_input_block(
                self.operator_lift(
                    self._query_operator_input(
                        normalized_points, encoded.global_operator_state
                    )
                )
            )
            n_chunk = normalized_points.shape[0]
            raw_query_drive = ScalarVectorState(
                torch.cat(
                    (
                        normalized_points.new_zeros(
                            n_chunk, self._boundary_drive_scalars
                        ),
                        encoded.global_drive_state.scalars.expand(n_chunk, -1),
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        normalized_points.new_zeros(
                            n_chunk,
                            self._boundary_drive_vectors,
                            self.n_spatial_dims,
                        ),
                        encoded.global_drive_state.vectors.expand(n_chunk, -1, -1),
                    ),
                    dim=1,
                ),
            )
            # Global drive quantities (for example a prescribed far field)
            # are legitimate pointwise query inputs as well as boundary
            # inputs.  Boundary-only drive channels remain exactly zero here.
            query_drive: ScalarVectorState | None = self.drive_lift(
                query_operator, raw_query_drive
            )
            for block, source_moments in zip(
                self.query_blocks, encoded.query_moments, strict=True
            ):
                query_drive = block.evaluate_cross(
                    query_operator, source_moments, query_drive
                )
            if query_drive is None:  # guarded by query_layers >= 1 at construction
                raise RuntimeError("query decoder produced no field state")
            output = self.output_projection(query_operator, query_drive)
            scalar_outputs.append(output.scalars)
            vector_outputs.append(output.vectors)

        packed_output = ScalarVectorState(
            torch.cat(scalar_outputs, dim=0),
            torch.cat(vector_outputs, dim=0),
        )
        point_data = self.output_layout.unpack(
            ScalarVectorFields(packed_output.scalars, packed_output.vectors)
        )
        return query_mesh.with_data(
            point_data=point_data,
            cell_data={},
            global_data=encoded.global_data,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        r"""Encode ``domain.boundaries`` and predict at ``domain.interior``."""
        return self.decode(self.encode(domain))


__all__ = [
    "EncodedBoundary",
    "FieldMode",
    "FieldRoleRanks",
    "MeshTransformer",
]
