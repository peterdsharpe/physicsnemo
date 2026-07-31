# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reference-surface operators for nonmatching cell-centered surface fields.

The physical measure lives on one frozen, sufficiently fine reference mesh.
A full-cover representation mesh is mapped to that reference by assigning
each reference cell to a representation face.  The resulting piecewise-
constant prolongation and area-adjoint restriction never construct a dense
source-by-target matrix.

This module deliberately distinguishes two geometries:

``build_reference_surface_map``
    A discrete common-refinement approximation for a *full-cover* surface.
    Reference centroids are projected to the closest representation triangle.
    Distance, orientation, and nonempty-face gates fail closed.

``build_voronoi_reconstruction``
    A centroid Voronoi reconstruction, optionally normal-aware.  It can fill
    holes in a sparse panel set and is therefore an extrapolation prior, not a
    conservative remesh.  It exists only for explicit sensitivity diagnostics.

The implementation remains study-local until production-scale geometry and
refinement-convergence audits justify a public mesh API.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from physicsnemo.mesh.spatial import signed_distance_field
from physicsnemo.nn.functional.neighbors import knn


def _expanded_leading_weights(
    weights: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    return weights.reshape(weights.shape + (1,) * (values.ndim - 1))


def _validate_values(
    values: torch.Tensor,
    expected: int,
    device: torch.device,
    name: str,
) -> None:
    if values.ndim == 0 or values.shape[0] != expected:
        raise ValueError(
            f"{name} must have leading shape ({expected},), got {tuple(values.shape)}"
        )
    if values.device != device:
        raise ValueError(
            f"{name} and the reference-surface map must share a device, "
            f"got {values.device} and {device}"
        )


@dataclass(frozen=True)
class ReferenceSurfaceMap:
    r"""Mass-adjoint P0 map between a representation and a reference surface.

    ``reference_to_representation[i] = j`` defines the binary incidence
    matrix :math:`S_{ij}`.  With diagonal reference and representation
    measures :math:`D_r` and :math:`D_c`, prolongation is ``S`` and
    restriction is :math:`D_c^{-1}S^T D_r`.

    Every representation cell must receive positive reference measure.  A
    sparse panel subset generally fails the geometry gate before this object is
    constructed; forcing it through a Voronoi assignment is explicitly a
    reconstruction sensitivity, not a physical overlap map.
    """

    reference_to_representation: torch.Tensor
    reference_measures: torch.Tensor
    representation_measures: torch.Tensor

    def __post_init__(self) -> None:
        assignment = self.reference_to_representation
        reference = self.reference_measures
        representation = self.representation_measures

        if assignment.ndim != 1 or assignment.dtype != torch.long:
            raise ValueError(
                "reference_to_representation must be a rank-1 int64 tensor"
            )
        if reference.shape != assignment.shape:
            raise ValueError(
                "reference_measures must have the same shape as "
                "reference_to_representation"
            )
        if representation.ndim != 1 or representation.numel() == 0:
            raise ValueError("representation_measures must be a nonempty rank-1 tensor")
        if (
            assignment.device != reference.device
            or assignment.device != representation.device
        ):
            raise ValueError("all map tensors must share a device")
        if not torch.is_floating_point(reference) or not torch.is_floating_point(
            representation
        ):
            raise ValueError("surface measures must be floating-point tensors")
        if not torch.isfinite(reference).all() or not (reference > 0).all():
            raise ValueError("reference_measures must be finite and strictly positive")
        if not torch.isfinite(representation).all() or not (representation > 0).all():
            raise ValueError(
                "every representation cell must receive finite positive "
                "reference measure"
            )
        if assignment.numel() == 0:
            raise ValueError("the reference surface must contain at least one cell")
        if assignment.min() < 0 or assignment.max() >= representation.numel():
            raise ValueError("reference_to_representation contains an invalid index")

        accumulated = reference.new_zeros(representation.shape)
        accumulated.index_add_(0, assignment, reference)
        eps = torch.finfo(reference.dtype).eps
        if not torch.allclose(
            accumulated,
            representation.to(reference.dtype),
            rtol=32 * eps,
            atol=32 * eps * float(reference.sum()),
        ):
            raise ValueError(
                "representation_measures must equal the reference measures "
                "accumulated by reference_to_representation"
            )

    @classmethod
    def from_assignment(
        cls,
        reference_to_representation: torch.Tensor,
        reference_measures: torch.Tensor,
        n_representation_cells: int,
    ) -> "ReferenceSurfaceMap":
        """Construct a fail-closed map and accumulate represented measures."""
        if n_representation_cells <= 0:
            raise ValueError("n_representation_cells must be positive")
        if reference_to_representation.ndim != 1:
            raise ValueError("reference_to_representation must be rank 1")
        if reference_measures.shape != reference_to_representation.shape:
            raise ValueError(
                "reference_measures must match reference_to_representation"
            )
        if reference_to_representation.dtype != torch.long:
            raise ValueError("reference_to_representation must have dtype int64")
        if reference_to_representation.device != reference_measures.device:
            raise ValueError(
                "reference_to_representation and reference_measures must share a device"
            )
        if reference_to_representation.numel() == 0:
            raise ValueError("the reference surface must contain at least one cell")
        if (
            reference_to_representation.min() < 0
            or reference_to_representation.max() >= n_representation_cells
        ):
            raise ValueError("reference_to_representation contains an invalid index")

        represented = reference_measures.new_zeros(n_representation_cells)
        represented.index_add_(
            0,
            reference_to_representation,
            reference_measures,
        )
        empty = torch.nonzero(represented <= 0, as_tuple=False).flatten()
        if empty.numel() > 0:
            preview = empty[:8].tolist()
            raise ValueError(
                f"{empty.numel()} representation cells receive no positive "
                f"reference measure (first indices: {preview})"
            )
        return cls(
            reference_to_representation=reference_to_representation,
            reference_measures=reference_measures,
            representation_measures=represented,
        )

    @property
    def n_reference_cells(self) -> int:
        return self.reference_to_representation.numel()

    @property
    def n_representation_cells(self) -> int:
        return self.representation_measures.numel()

    def prolong_to_reference(self, representation_values: torch.Tensor) -> torch.Tensor:
        """Piecewise-constant prolongation onto the frozen reference cells."""
        _validate_values(
            representation_values,
            self.n_representation_cells,
            self.reference_measures.device,
            "representation_values",
        )
        return representation_values[self.reference_to_representation]

    def restrict_reference(self, reference_values: torch.Tensor) -> torch.Tensor:
        """Area-adjoint restriction from the reference to representation cells."""
        _validate_values(
            reference_values,
            self.n_reference_cells,
            self.reference_measures.device,
            "reference_values",
        )
        compute_dtype = torch.promote_types(
            reference_values.dtype, self.reference_measures.dtype
        )
        values = reference_values.to(compute_dtype)
        weights = self.reference_measures.to(compute_dtype)
        weighted = values * _expanded_leading_weights(weights, values)
        result = torch.zeros(
            (self.n_representation_cells, *values.shape[1:]),
            dtype=compute_dtype,
            device=values.device,
        )
        indices = self.reference_to_representation.reshape(
            (-1,) + (1,) * (values.ndim - 1)
        ).expand_as(weighted)
        result.scatter_add_(0, indices, weighted)
        denominator = _expanded_leading_weights(
            self.representation_measures.to(compute_dtype),
            result,
        )
        return result / denominator

    def project_reference(self, reference_values: torch.Tensor) -> torch.Tensor:
        """Orthogonally project reference data into the represented P0 space."""
        return self.prolong_to_reference(self.restrict_reference(reference_values))

    def sha256(self) -> str:
        """Content digest of the exact discrete operator."""
        digest = hashlib.sha256()
        for tensor in (
            self.reference_to_representation,
            self.reference_measures,
            self.representation_measures,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class SurfaceMapDiagnostics:
    """Geometry diagnostics that determine whether a map is admissible."""

    reference_distance: torch.Tensor
    reference_normal_alignment: torch.Tensor
    representation_distance: torch.Tensor
    representation_normal_alignment: torch.Tensor
    represented_to_native_measure: torch.Tensor
    method: str

    def summary(self) -> dict[str, Any]:
        """Return JSON-ready load-bearing diagnostics."""

        def distribution(values: torch.Tensor) -> dict[str, float]:
            values = values.detach().double().cpu()
            quantiles = torch.quantile(
                values,
                torch.tensor([0.5, 0.95, 0.99], dtype=values.dtype),
            )
            return {
                "min": float(values.min()),
                "q50": float(quantiles[0]),
                "q95": float(quantiles[1]),
                "q99": float(quantiles[2]),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }

        return {
            "method": self.method,
            "reference_to_representation_distance": distribution(
                self.reference_distance
            ),
            "reference_to_representation_normal_dot": distribution(
                self.reference_normal_alignment
            ),
            "representation_to_reference_distance": distribution(
                self.representation_distance
            ),
            "representation_to_reference_normal_dot": distribution(
                self.representation_normal_alignment
            ),
            "represented_to_native_measure_ratio": distribution(
                self.represented_to_native_measure
            ),
            "negative_reference_normal_fraction": float(
                (self.reference_normal_alignment < 0).double().mean()
            ),
            "negative_representation_normal_fraction": float(
                (self.representation_normal_alignment < 0).double().mean()
            ),
        }


def _gate_geometry(
    diagnostics: SurfaceMapDiagnostics,
    *,
    max_distance: float,
    min_normal_alignment: float,
) -> None:
    if not math.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    if (
        not math.isfinite(min_normal_alignment)
        or min_normal_alignment < -1
        or min_normal_alignment > 1
    ):
        raise ValueError("min_normal_alignment must lie in [-1, 1]")

    largest_distance = max(
        float(diagnostics.reference_distance.max()),
        float(diagnostics.representation_distance.max()),
    )
    if largest_distance > max_distance:
        raise ValueError(
            "surface coverage gate failed: symmetric maximum distance "
            f"{largest_distance:.9g} exceeds {max_distance:.9g}"
        )
    smallest_alignment = min(
        float(diagnostics.reference_normal_alignment.min()),
        float(diagnostics.representation_normal_alignment.min()),
    )
    if smallest_alignment < min_normal_alignment:
        raise ValueError(
            "surface orientation gate failed: minimum matched normal dot "
            f"{smallest_alignment:.9g} is below {min_normal_alignment:.9g}"
        )


def build_reference_surface_map(
    reference: Mesh,
    representation: Mesh,
    *,
    max_distance: float,
    min_normal_alignment: float,
) -> tuple[ReferenceSurfaceMap, SurfaceMapDiagnostics]:
    """Build a discrete full-cover map using closest representation triangles.

    The fine reference cells are the common-refinement quadrature.  Each is
    assigned wholly to the representation triangle whose surface realizes the
    closest point.  This is a convergent whole-cell approximation to overlap,
    not exact polygon clipping; callers must perform a reference-refinement
    study before promoting a quantitative claim.
    """
    forward = signed_distance_field(representation, reference.cell_centroids)
    reverse = signed_distance_field(reference, representation.cell_centroids)
    if (forward.hit_faces < 0).any() or (reverse.hit_faces < 0).any():
        raise ValueError("nearest-surface query left at least one cell unmatched")

    reference_normals = reference.cell_normals
    representation_normals = representation.cell_normals
    forward_alignment = (
        reference_normals * representation_normals[forward.hit_faces]
    ).sum(dim=-1)
    reverse_alignment = (
        representation_normals * reference_normals[reverse.hit_faces]
    ).sum(dim=-1)

    transfer = ReferenceSurfaceMap.from_assignment(
        forward.hit_faces,
        cell_measures(reference),
        representation.n_cells,
    )
    native_measures = cell_measures(representation).to(
        dtype=transfer.representation_measures.dtype,
        device=transfer.representation_measures.device,
    )
    diagnostics = SurfaceMapDiagnostics(
        reference_distance=forward.sdf.abs(),
        reference_normal_alignment=forward_alignment,
        representation_distance=reverse.sdf.abs(),
        representation_normal_alignment=reverse_alignment,
        represented_to_native_measure=transfer.representation_measures
        / native_measures,
        method="closest_triangle_discrete_common_refinement",
    )
    _gate_geometry(
        diagnostics,
        max_distance=max_distance,
        min_normal_alignment=min_normal_alignment,
    )
    return transfer, diagnostics


def build_voronoi_reconstruction(
    reference: Mesh,
    representation: Mesh,
    *,
    normal_weight: float | None = None,
) -> tuple[ReferenceSurfaceMap, SurfaceMapDiagnostics]:
    """Build an explicitly nonphysical centroid-Voronoi reconstruction.

    This map fills the entire reference from any nonempty set of representation
    cells.  For an incomplete/sparse representation it is an extrapolation
    prior and must never be reported as remesh overlap.
    """
    if normal_weight is not None and (
        not math.isfinite(normal_weight) or normal_weight <= 0
    ):
        raise ValueError("normal_weight must be finite and positive")

    reference_centroids = reference.cell_centroids
    representation_centroids = representation.cell_centroids
    reference_normals = reference.cell_normals
    representation_normals = representation.cell_normals
    if normal_weight is None:
        search_points = representation_centroids
        queries = reference_centroids
        method = "ambient_centroid_voronoi_reconstruction"
    else:
        search_points = torch.cat(
            (
                representation_centroids,
                normal_weight * representation_normals,
            ),
            dim=-1,
        )
        queries = torch.cat(
            (
                reference_centroids,
                normal_weight * reference_normals,
            ),
            dim=-1,
        )
        method = "normal_aware_centroid_voronoi_reconstruction"

    assignment, _ = knn(search_points, queries, k=1)
    assignment = assignment[:, 0]
    transfer = ReferenceSurfaceMap.from_assignment(
        assignment,
        cell_measures(reference),
        representation.n_cells,
    )

    forward_distance = (
        reference_centroids - representation_centroids[assignment]
    ).norm(dim=-1)
    forward_alignment = (reference_normals * representation_normals[assignment]).sum(
        dim=-1
    )
    reverse_assignment, _ = knn(reference_centroids, representation_centroids, k=1)
    reverse_assignment = reverse_assignment[:, 0]
    reverse_distance = (
        representation_centroids - reference_centroids[reverse_assignment]
    ).norm(dim=-1)
    reverse_alignment = (
        representation_normals * reference_normals[reverse_assignment]
    ).sum(dim=-1)
    native_measures = cell_measures(representation).to(
        dtype=transfer.representation_measures.dtype,
        device=transfer.representation_measures.device,
    )
    diagnostics = SurfaceMapDiagnostics(
        reference_distance=forward_distance,
        reference_normal_alignment=forward_alignment,
        representation_distance=reverse_distance,
        representation_normal_alignment=reverse_alignment,
        represented_to_native_measure=transfer.representation_measures
        / native_measures,
        method=method,
    )
    return transfer, diagnostics


__all__ = [
    "ReferenceSurfaceMap",
    "SurfaceMapDiagnostics",
    "build_reference_surface_map",
    "build_voronoi_reconstruction",
]
