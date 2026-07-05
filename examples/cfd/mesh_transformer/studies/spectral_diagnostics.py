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

r"""Spectral diagnostics for boundary-to-interior Laplace operators.

These utilities deliberately live beside the benchmark rather than in the
generic mesh-attention package.  They use the unit disk as a representation-
theoretic microscope: on a concentric query ring, a linear O(2)-equivariant
scalar operator is diagonal in angular Fourier order.  This makes missing
angular orders visible without conflating them with geometry-distribution
shift.

The extracted matrix is the discrete map from cell-centered boundary samples
to scalar query values at one fixed geometry.  Extraction requires one model
evaluation per source degree of freedom and is therefore a diagnostic, not a
production execution path.  By default, an additional dense probe verifies
that the model really acts linearly on the selected boundary field.

The analytic reference is the periodic trapezoidal discretization of the
unit-disk Poisson integral.  It is not a claim that polygon panels are an exact
circle: its columns correspond to samples at uniformly spaced panel-midpoint
angles, matching the benchmark's continuous unit-disk problem.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from physicsnemo.experimental.nn.mesh_attention import EncodedBoundary
from physicsnemo.mesh import DomainMesh, Mesh


@dataclass(frozen=True)
class OperatorSpectrum:
    """Singular spectrum and Eckart--Young best-rank errors for one operator."""

    singular_values: torch.Tensor
    numerical_rank: int
    relative_best_rank_errors: dict[int, float]


@dataclass(frozen=True)
class OperatorSpectrumComparison:
    """Spectral comparison between a learned and an analytic discrete map."""

    learned: OperatorSpectrum
    analytic: OperatorSpectrum
    relative_operator_error: float


@dataclass(frozen=True)
class MomentFamilyNorms:
    r"""Frobenius norms of the four typed source-moment families.

    The two mixed families each carry one Cartesian index.  The final family
    carries two Cartesian indices and is a *reducible* rank-two tensor: its
    norm includes both trace and symmetric-trace-free content (and any
    antisymmetric content).  These diagnostics report what the implementation
    stores; they do not relabel that tensor as a pure :math:`\ell=2` irrep.
    """

    scalar_key_scalar_value: float
    vector_key_scalar_value: float
    scalar_key_vector_value: float
    vector_key_vector_value: float


def _positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _integer_modes(name: str, modes: Sequence[int]) -> tuple[int, ...]:
    result = tuple(modes)
    if any(isinstance(mode, bool) or not isinstance(mode, int) for mode in result):
        raise TypeError(f"{name} must contain only integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate modes")
    return result


def _nonnegative_modes(name: str, modes: Sequence[int]) -> tuple[int, ...]:
    result = _integer_modes(name, modes)
    if any(mode < 0 for mode in result):
        raise ValueError(f"{name} must contain only non-negative modes")
    return result


def uniform_angles(
    n_angles: int,
    *,
    offset: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a uniform angular grid on ``[offset, offset + 2 pi)``."""

    _positive_int("n_angles", n_angles)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    return (
        offset
        + 2.0 * math.pi * torch.arange(n_angles, device=device, dtype=dtype) / n_angles
    )


def concentric_ring_preimages(
    radii: torch.Tensor,
    n_angles: int,
    *,
    angular_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return complex disk points and their shared angular grid.

    The complex points have shape ``(n_rings, n_angles)``.  Radii must lie in
    ``[0, 1)``; boundary queries are excluded because the Poisson kernel is
    singular when a query coincides with a source sample.
    """

    _positive_int("n_angles", n_angles)
    if radii.ndim != 1 or radii.dtype not in (torch.float32, torch.float64):
        raise ValueError("radii must be a one-dimensional float32/float64 tensor")
    if radii.numel() == 0:
        raise ValueError("radii must be nonempty")
    if (
        not torch.isfinite(radii).all().item()
        or torch.any((radii < 0.0) | (radii >= 1.0)).item()
    ):
        raise ValueError("radii must be finite and lie in [0, 1)")
    angles = uniform_angles(
        n_angles,
        offset=angular_offset,
        device=radii.device,
        dtype=radii.dtype,
    )
    unit = torch.polar(torch.ones_like(angles), angles)
    return radii[:, None] * unit[None, :], angles


def _validate_uniform_angles(
    angles: torch.Tensor, *, label: str = "boundary_angles"
) -> None:
    if angles.ndim != 1 or angles.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{label} must be a one-dimensional real tensor")
    if angles.numel() < 3:
        raise ValueError(f"{label} must contain at least three samples")
    if not torch.isfinite(angles).all().item():
        raise ValueError(f"{label} must be finite")
    period = angles.new_tensor(2.0 * math.pi)
    ordered = torch.sort(torch.remainder(angles, period)).values
    gaps = torch.cat((ordered[1:] - ordered[:-1], ordered[:1] + period - ordered[-1:]))
    expected = period / angles.numel()
    tolerance = 2.0e-5 if angles.dtype == torch.float32 else 2.0e-12
    if not torch.allclose(
        gaps, expected.expand_as(gaps), rtol=tolerance, atol=tolerance
    ):
        raise ValueError(f"{label} must be uniformly spaced modulo 2 pi")


def unit_disk_poisson_matrix(
    boundary_angles: torch.Tensor,
    query_preimages: torch.Tensor,
) -> torch.Tensor:
    r"""Return the midpoint-trapezoid matrix for the unit-disk Poisson map.

    For boundary samples :math:`g(\theta_j)` at ``n`` uniform angles, the
    returned matrix implements

    .. math::

       u(z_i) \approx \frac{1}{n}\sum_j
       \frac{1-|z_i|^2}{|e^{i\theta_j}-z_i|^2}g(\theta_j).

    ``query_preimages`` may have any shape; it is flattened into matrix rows.
    The result has shape ``(query_preimages.numel(), n_boundary)``.
    """

    _validate_uniform_angles(boundary_angles)
    if not torch.is_complex(query_preimages):
        raise TypeError("query_preimages must be a complex tensor")
    if (
        query_preimages.real.dtype != boundary_angles.dtype
        or query_preimages.device != boundary_angles.device
    ):
        raise ValueError("query_preimages and boundary_angles must share dtype/device")
    if not torch.isfinite(query_preimages).all().item():
        raise ValueError("query_preimages must be finite")
    queries = query_preimages.reshape(-1)
    radii = queries.abs()
    if torch.any(radii >= 1.0).item():
        raise ValueError("Poisson-matrix queries must lie strictly inside the disk")
    phase = torch.angle(queries)[:, None] - boundary_angles[None, :]
    radius = radii[:, None]
    denominator = 1.0 - 2.0 * radius * torch.cos(phase) + radius.square()
    return (1.0 - radius.square()) / (boundary_angles.numel() * denominator)


def _replace_boundary_values(
    domain: DomainMesh,
    values: torch.Tensor,
    *,
    boundary_name: str,
    input_field: str,
) -> DomainMesh:
    boundary = domain.boundaries[boundary_name]
    cell_data = {key: boundary.cell_data[key] for key in boundary.cell_data.keys()}
    cell_data[input_field] = values
    boundaries = {key: domain.boundaries[key] for key in domain.boundaries.keys()}
    boundaries[boundary_name] = boundary.with_data(cell_data=cell_data)
    return DomainMesh(
        # Strip every query field so the diagnostic cannot accidentally expose
        # labels or evaluation metadata to a candidate model.
        interior=domain.interior.with_data(point_data={}, cell_data={}),
        boundaries=boundaries,
        global_data=domain.global_data.copy(),
    )


def _scalar_prediction(
    model: Callable[[DomainMesh], Mesh],
    domain: DomainMesh,
    output_field: str,
) -> torch.Tensor:
    prediction = model(domain)
    if not isinstance(prediction, Mesh):
        raise TypeError("model must return a Mesh")
    if output_field not in prediction.point_data:
        raise ValueError(f"prediction does not contain point field {output_field!r}")
    values = prediction.point_data[output_field]
    if values.ndim != 1 or values.shape[0] != domain.interior.n_points:
        raise ValueError(
            f"prediction field {output_field!r} must have shape "
            f"({domain.interior.n_points},)"
        )
    return values


@torch.no_grad()
def extract_discrete_operator(
    model: Callable[[DomainMesh], Mesh],
    domain: DomainMesh,
    *,
    boundary_name: str = "dirichlet",
    input_field: str = "boundary_value",
    output_field: str = "potential",
    verify_linearity: bool = True,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-7,
) -> torch.Tensor:
    r"""Extract a fixed-geometry scalar boundary-to-query matrix.

    Column ``j`` is the response to a unit value in boundary cell ``j`` and
    zero in all other cells of ``boundary_name``.  Other boundary fields and
    other named boundaries are held fixed.  Consequently, callers should set
    every other drive field to zero when diagnosing a multi-drive model.

    With ``verify_linearity=True``, the zero input must map to zero and the
    extracted matrix must reproduce one deterministic dense probe.  This is a
    useful regression check, not a mathematical proof of global linearity.
    """

    if boundary_name not in domain.boundaries:
        raise ValueError(f"domain has no boundary named {boundary_name!r}")
    boundary = domain.boundaries[boundary_name]
    if input_field not in boundary.cell_data:
        raise ValueError(f"boundary does not contain cell field {input_field!r}")
    template = boundary.cell_data[input_field]
    if template.ndim != 1 or template.shape[0] != boundary.n_cells:
        raise ValueError(
            f"boundary field {input_field!r} must have shape ({boundary.n_cells},)"
        )
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("rtol and atol must be finite and non-negative")

    module = model if isinstance(model, nn.Module) else None
    was_training = module.training if module is not None else None
    if module is not None:
        module.eval()
    try:
        zero = torch.zeros_like(template)
        zero_domain = _replace_boundary_values(
            domain,
            zero,
            boundary_name=boundary_name,
            input_field=input_field,
        )
        zero_response = _scalar_prediction(model, zero_domain, output_field)
        if verify_linearity and not torch.allclose(
            zero_response, torch.zeros_like(zero_response), rtol=rtol, atol=atol
        ):
            maximum = zero_response.abs().max().item()
            raise ValueError(
                "selected map is not zero-preserving; "
                f"maximum zero-input response is {maximum:.6g}"
            )

        columns: list[torch.Tensor] = []
        for source in range(boundary.n_cells):
            basis = torch.zeros_like(template)
            basis[source] = 1.0
            basis_domain = _replace_boundary_values(
                domain,
                basis,
                boundary_name=boundary_name,
                input_field=input_field,
            )
            columns.append(_scalar_prediction(model, basis_domain, output_field))
        matrix = torch.stack(columns, dim=-1)

        if verify_linearity:
            probe = torch.linspace(
                -0.731,
                0.917,
                boundary.n_cells,
                device=template.device,
                dtype=template.dtype,
            )
            probe_domain = _replace_boundary_values(
                domain,
                probe,
                boundary_name=boundary_name,
                input_field=input_field,
            )
            actual = _scalar_prediction(model, probe_domain, output_field)
            expected = matrix @ probe
            if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
                error = (actual - expected).abs().max().item()
                raise ValueError(
                    "selected map failed the dense linear reconstruction probe; "
                    f"maximum absolute error is {error:.6g}"
                )
        return matrix
    finally:
        if module is not None and was_training is not None:
            module.train(was_training)


def operator_spectrum(
    matrix: torch.Tensor,
    ranks: Sequence[int],
    *,
    rank_rtol: float | None = None,
) -> OperatorSpectrum:
    r"""Compute singular values and relative best-rank Frobenius errors."""

    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("matrix must be a two-dimensional floating-point tensor")
    if matrix.numel() == 0 or not torch.isfinite(matrix).all().item():
        raise ValueError("matrix must be nonempty and finite")
    requested = tuple(ranks)
    maximum_rank = min(matrix.shape)
    if any(
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank > maximum_rank
        for rank in requested
    ):
        raise ValueError(f"ranks must be integers in [0, {maximum_rank}]")
    if len(set(requested)) != len(requested):
        raise ValueError("ranks must not contain duplicates")

    singular_values = torch.linalg.svdvals(matrix)
    squared = singular_values.square()
    total = squared.sum()
    if total.item() == 0.0:
        raise ValueError("relative best-rank errors are undefined for a zero operator")
    errors = {
        rank: torch.sqrt(squared[rank:].sum() / total).item() for rank in requested
    }
    if rank_rtol is None:
        rank_rtol = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    if not math.isfinite(rank_rtol) or rank_rtol < 0.0:
        raise ValueError("rank_rtol must be finite and non-negative")
    numerical_rank = int(
        torch.count_nonzero(singular_values > singular_values[0] * rank_rtol).item()
    )
    return OperatorSpectrum(singular_values, numerical_rank, errors)


def compare_operator_spectra(
    learned_matrix: torch.Tensor,
    analytic_matrix: torch.Tensor,
    ranks: Sequence[int],
    *,
    rank_rtol: float | None = None,
) -> OperatorSpectrumComparison:
    """Compare learned and analytic maps on exactly the same discrete grids."""

    if learned_matrix.shape != analytic_matrix.shape:
        raise ValueError("learned and analytic matrices must have identical shapes")
    if (
        learned_matrix.device != analytic_matrix.device
        or learned_matrix.dtype != analytic_matrix.dtype
    ):
        raise ValueError("learned and analytic matrices must share dtype/device")
    denominator = torch.linalg.matrix_norm(analytic_matrix)
    if denominator.item() == 0.0:
        raise ValueError("analytic operator must have nonzero Frobenius norm")
    relative_error = (
        torch.linalg.matrix_norm(learned_matrix - analytic_matrix) / denominator
    ).item()
    return OperatorSpectrumComparison(
        learned=operator_spectrum(learned_matrix, ranks, rank_rtol=rank_rtol),
        analytic=operator_spectrum(analytic_matrix, ranks, rank_rtol=rank_rtol),
        relative_operator_error=relative_error,
    )


def fourier_transfer_matrix(
    operator_matrix: torch.Tensor,
    boundary_angles: torch.Tensor,
    query_angles: torch.Tensor,
    *,
    input_modes: Sequence[int],
    output_modes: Sequence[int],
) -> torch.Tensor:
    r"""Return complex Fourier transfer coefficients on concentric rings.

    ``operator_matrix`` must have shape ``(n_rings, n_query_angles,
    n_boundary)``.  Entry ``[r, ell_index, k_index]`` is the coefficient of
    :math:`e^{i\ell\phi}` in the response to boundary samples
    :math:`e^{ik\theta}`, where ``ell`` and ``k`` are the corresponding
    entries of ``output_modes`` and ``input_modes``.  Signed modes are retained
    because an incorrect real operator can map a positive complex input mode
    to a negative output mode.  A real learned matrix is extended
    complex-linearly for this diagnostic.  No normalization is applied on the
    source side; output coefficients use the usual discrete angular mean.
    """

    if operator_matrix.ndim != 3 or not operator_matrix.is_floating_point():
        raise ValueError("operator_matrix must be a three-dimensional real tensor")
    _validate_uniform_angles(boundary_angles)
    if operator_matrix.shape[-1] != boundary_angles.numel():
        raise ValueError("operator source dimension must match boundary_angles")
    n_rings, n_query_angles, _ = operator_matrix.shape
    if query_angles.ndim == 1:
        if query_angles.shape[0] != n_query_angles:
            raise ValueError("query angle count must match operator rows per ring")
        query_angles = query_angles.expand(n_rings, -1)
    elif query_angles.shape != (n_rings, n_query_angles):
        raise ValueError(
            "query_angles must have shape (n_query_angles,) or "
            "(n_rings, n_query_angles)"
        )
    if (
        query_angles.dtype != operator_matrix.dtype
        or query_angles.device != operator_matrix.device
        or boundary_angles.dtype != operator_matrix.dtype
        or boundary_angles.device != operator_matrix.device
    ):
        raise ValueError("operator and angle tensors must share dtype/device")
    if (
        not torch.isfinite(operator_matrix).all().item()
        or not torch.isfinite(query_angles).all().item()
    ):
        raise ValueError("operator and query angles must be finite")
    for ring in range(n_rings):
        _validate_uniform_angles(query_angles[ring], label="query_angles on each ring")
    input_modes = _integer_modes("input_modes", input_modes)
    output_modes = _integer_modes("output_modes", output_modes)
    if not input_modes or not output_modes:
        raise ValueError("input_modes and output_modes must be nonempty")
    if 2 * max(map(abs, input_modes)) >= boundary_angles.numel():
        raise ValueError("input modes must lie strictly below the source Nyquist order")
    if 2 * max(map(abs, output_modes)) >= n_query_angles:
        raise ValueError("output modes must lie strictly below the query Nyquist order")

    real_dtype = operator_matrix.dtype
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    input_orders = boundary_angles.new_tensor(input_modes)
    output_orders = query_angles.new_tensor(output_modes)
    source_basis = torch.exp(
        1j * boundary_angles[:, None].to(complex_dtype) * input_orders[None, :]
    )
    responses = torch.einsum(
        "ras,sk->rak", operator_matrix.to(complex_dtype), source_basis
    )
    analysis_basis = torch.exp(
        -1j * query_angles[:, :, None].to(complex_dtype) * output_orders[None, None, :]
    )
    return torch.einsum("ral,rak->rlk", analysis_basis, responses) / n_query_angles


def _moment_norms(
    real: EncodedBoundary,
    imaginary: EncodedBoundary | None,
) -> tuple[MomentFamilyNorms, ...]:
    if imaginary is not None and len(real.query_moments) != len(
        imaginary.query_moments
    ):
        raise ValueError("real and imaginary encodings have different decoder depths")

    result: list[MomentFamilyNorms] = []
    for layer, real_moments in enumerate(real.query_moments):
        imaginary_moments = (
            None if imaginary is None else imaginary.query_moments[layer]
        )

        def norm(name: str) -> float:
            real_tensor = getattr(real_moments, name)
            squared = real_tensor.double().square().sum()
            if imaginary_moments is not None:
                imaginary_tensor = getattr(imaginary_moments, name)
                squared = squared + imaginary_tensor.double().square().sum()
            return torch.sqrt(squared).item()

        result.append(
            MomentFamilyNorms(
                scalar_key_scalar_value=norm("scalar_key_scalar_value"),
                vector_key_scalar_value=norm("vector_key_scalar_value"),
                scalar_key_vector_value=norm("scalar_key_vector_value"),
                vector_key_vector_value=norm("vector_key_vector_value"),
            )
        )
    return tuple(result)


@torch.no_grad()
def fourier_moment_family_norms(
    model: Any,
    domain: DomainMesh,
    boundary_angles: torch.Tensor,
    *,
    modes: Sequence[int] = tuple(range(9)),
    boundary_name: str = "dirichlet",
    input_field: str = "boundary_value",
) -> dict[int, tuple[MomentFamilyNorms, ...]]:
    r"""Report typed source-moment norms for complex boundary Fourier modes.

    For mode zero, the report encodes the constant trace.  For positive mode
    ``k``, it combines separate real-model encodings of ``cos(k theta)`` and
    ``sin(k theta)`` as the Frobenius norm of the complex response to
    ``exp(i k theta)``.  The combination is phase-invariant and keeps the
    underlying model strictly real.
    """

    if not hasattr(model, "encode"):
        raise TypeError("model must expose MeshTransformer-compatible encode(domain)")
    _validate_uniform_angles(boundary_angles)
    modes = _nonnegative_modes("modes", modes)
    if not modes:
        raise ValueError("modes must be nonempty")
    if boundary_name not in domain.boundaries:
        raise ValueError(f"domain has no boundary named {boundary_name!r}")
    boundary = domain.boundaries[boundary_name]
    if boundary.n_cells != boundary_angles.numel():
        raise ValueError("boundary angle count must match boundary cell count")
    if 2 * max(modes) >= boundary.n_cells:
        raise ValueError("modes must lie strictly below the boundary Nyquist order")
    if (
        boundary_angles.device != boundary.points.device
        or boundary_angles.dtype != boundary.points.dtype
    ):
        raise ValueError("boundary angles and mesh must share dtype/device")

    module = model if isinstance(model, nn.Module) else None
    was_training = module.training if module is not None else None
    if module is not None:
        module.eval()
    try:
        report: dict[int, tuple[MomentFamilyNorms, ...]] = {}
        for mode in modes:
            cosine = torch.cos(mode * boundary_angles)
            cosine_domain = _replace_boundary_values(
                domain,
                cosine,
                boundary_name=boundary_name,
                input_field=input_field,
            )
            real_encoding = model.encode(cosine_domain)
            if not isinstance(real_encoding, EncodedBoundary):
                raise TypeError("model.encode must return EncodedBoundary")
            imaginary_encoding = None
            if mode > 0:
                sine = torch.sin(mode * boundary_angles)
                sine_domain = _replace_boundary_values(
                    domain,
                    sine,
                    boundary_name=boundary_name,
                    input_field=input_field,
                )
                imaginary_encoding = model.encode(sine_domain)
                if not isinstance(imaginary_encoding, EncodedBoundary):
                    raise TypeError("model.encode must return EncodedBoundary")
            report[mode] = _moment_norms(real_encoding, imaginary_encoding)
        return report
    finally:
        if module is not None and was_training is not None:
            module.train(was_training)


__all__ = [
    "MomentFamilyNorms",
    "OperatorSpectrum",
    "OperatorSpectrumComparison",
    "compare_operator_spectra",
    "concentric_ring_preimages",
    "extract_discrete_operator",
    "fourier_moment_family_norms",
    "fourier_transfer_matrix",
    "operator_spectrum",
    "uniform_angles",
    "unit_disk_poisson_matrix",
]
