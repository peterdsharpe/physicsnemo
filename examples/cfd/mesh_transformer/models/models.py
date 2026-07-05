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

"""Models and physically controlled baselines for the Laplace benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh


@dataclass(frozen=True)
class MeshTransformerConfig:
    """Finite-capacity settings used by the reference benchmark."""

    operator_scalar_dim: int = 32
    operator_vector_dim: int = 8
    drive_scalar_dim: int = 48
    drive_vector_dim: int = 12
    operator_layers: int = 2
    drive_layers: int = 1
    query_layers: int = 1
    heads: int = 4
    scalar_rank: int = 12
    vector_rank: int = 4
    query_chunk_size: int = 65536
    attention_chunk_size: int | None = 65536


def build_mesh_transformer(
    config: MeshTransformerConfig,
    *,
    query_decoder: str = "moment",
    kernel_mlp_members: int = 8,
    kernel_include_polynomial_members: bool = True,
    kernel_include_single_layer_member: bool = False,
    kernel_monopole_free_single_layer: bool = False,
    bounded_output_gate_invariants: bool = False,
    bounded_query_geometry: bool = False,
    decaying_direct_drive: bool = False,
    field_mode: str = "linear",
    boundary_field_ranks: dict | None = None,
    global_field_ranks: dict | None = None,
    reference_length_key: str | None = "reference_length",
) -> MeshTransformer:
    """Construct the scalar, linear Dirichlet-to-interior benchmark model.

    ``query_decoder="kernel"`` selects the dense kernel-basis query decoder
    (exact panel quadrature plus learned smooth pair members) in place of the
    separable moment decoder; every other benchmark convention is unchanged.
    ``kernel_include_polynomial_members=False`` drops the fixed polynomial
    smooth members for ablation; combined with ``kernel_mlp_members=0`` the
    kernel dictionary is the exact singular member alone.
    ``kernel_include_single_layer_member=True`` adds the exact single-layer
    member alongside the double layer (needed for completeness on multiply
    connected domains); with the smooth families off this is the two-member
    "singpair" dictionary.  ``bounded_output_gate_invariants=True`` feeds the
    output projection's sigmoid gates compactified (bounded) query invariants
    -- the far-field gate-collapse fix, which alone proved insufficient (it
    unmasked polynomially growing direct-drive branches; see the model
    docstring).  ``bounded_query_geometry=True`` is the source-side
    completion: the query operator state is built from the compactified
    position x/sqrt(1+|x|^2), bounding every learned query-radius dependence
    at once while the kernel dictionary's exact members keep the raw
    coordinates.  ``decaying_direct_drive=True`` multiplies the query-side
    direct-drive contribution by the fixed analytic envelope 1/(1+|x|^2) of
    the raw query radius (bounded is not decaying: iteration 29 measured the
    direct drive converging to a direction-dependent constant while the
    exact exterior velocity decays like r^-2), and
    ``kernel_monopole_free_single_layer=True`` deflates the exact
    single-layer member to zero net charge, structurally killing its log-r
    monopole tail (licensed only for zero-net-flux exteriors; see the model
    and decoder docstrings).  All four default ``False``: bitwise identical
    to the historical arms.

    ``boundary_field_ranks`` and ``global_field_ranks`` optionally replace the
    fixed scalar-Dirichlet schema; ``None`` (the default) reproduces the
    historical declaration exactly.  Benchmarks whose drive is not a scalar
    boundary trace (for example exterior potential flow, whose drive is a
    global rank-1 far-field velocity) pass their schemas through here rather
    than duplicating the model wiring.

    ``reference_length_key`` selects the scale gauge.  The benchmark default
    ``"reference_length"`` consumes the dataset's declared similarity scale
    (the historical, explicit gauge, preserved bitwise).  ``None`` selects
    the model's intrinsic gauge: the measure-weighted RMS boundary radius
    (radius of gyration), which is degree-1 homogeneous in the geometry and
    therefore unconditionally scale equivariant with no dataset convention
    to drift.
    """

    if boundary_field_ranks is None:
        boundary_field_ranks = {
            "dirichlet": {
                "operator": {},
                "drive": {"boundary_value": 0},
            }
        }
    if global_field_ranks is None:
        global_field_ranks = {"operator": {}, "drive": {}}
    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0},
        boundary_field_ranks=boundary_field_ranks,
        global_field_ranks=global_field_ranks,
        reference_length_key=reference_length_key,
        field_mode=field_mode,
        query_decoder=query_decoder,
        kernel_mlp_members=kernel_mlp_members,
        kernel_include_polynomial_members=kernel_include_polynomial_members,
        kernel_include_single_layer_member=kernel_include_single_layer_member,
        kernel_monopole_free_single_layer=kernel_monopole_free_single_layer,
        bounded_output_gate_invariants=bounded_output_gate_invariants,
        bounded_query_geometry=bounded_query_geometry,
        decaying_direct_drive=decaying_direct_drive,
        **asdict(config),
    )


class MeanLiftedDirichletModel(nn.Module):
    r"""Enforce the exact constant solution and learn only the residual map.

    If :math:`\bar g` is the boundary-measure mean, this wrapper evaluates

    .. math::

        u_\theta[g] = \bar g + N_\theta[g-\bar g].

    The construction is linear in the Dirichlet data, exactly reproduces every
    constant boundary condition, and introduces no coordinate frame, length
    scale, or locality heuristic.  It is Laplace-specific physics structure,
    so it remains an example wrapper rather than a generic MeshTransformer
    contract.
    """

    def __init__(self, residual_model: MeshTransformer) -> None:
        super().__init__()
        self.residual_model = residual_model

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        weights = boundary.cell_areas
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual_boundary = boundary.with_data(
            cell_data={"boundary_value": values - mean}
        )
        residual_domain = DomainMesh(
            interior=domain.interior,
            boundaries={"dirichlet": residual_boundary},
            global_data=domain.global_data,
        )
        residual = self.residual_model(residual_domain)
        return residual.with_data(
            point_data={"potential": residual.point_data["potential"] + mean},
            cell_data={},
            global_data=domain.global_data,
        )


def build_lifted_mesh_transformer(
    config: MeshTransformerConfig,
    *,
    reference_length_key: str | None = "reference_length",
) -> MeanLiftedDirichletModel:
    """Construct the constant-consistent Laplace specialization."""

    return MeanLiftedDirichletModel(
        build_mesh_transformer(config, reference_length_key=reference_length_key)
    )


class SingleStreamFusionControl(nn.Module):
    r"""MEASUREMENT-ONLY control: boundary values fused into the operator stream.

    **This control must never be offered as a real modeling mode.**  It exists
    to price the benchmark's last unmeasured architectural factorization: the
    two-stream operator/drive separation.  The production ``MeshTransformer``
    routes geometry through a nonlinear, biased operator stream and boundary
    data through a zero-preserving (linear or structurally multiplicative)
    drive stream -- the learned analogue of the boundary-integral-equation
    split into a density solve (drive) composed with propagation conditioned
    on geometry (operator).  That separation is what makes the exact
    zero-drive and fixed-geometry superposition contracts structural.

    Pre-registered design (iteration 26).  The control re-declares the scalar
    Dirichlet ``boundary_value`` as an *operator* field, so the boundary data
    rides with geometry through the full nonlinear encoder (biases and
    products allowed), and feeds a constant ``unit_drive = 1`` per boundary
    cell as the sole drive field.  Because every drive-path projection,
    kernel coefficient, and the output head is conditioned on the operator
    state, propagating the constant makes the prediction an *unrestricted*
    learned function of the fused (geometry, boundary-value) encoding -- the
    single-stream architecture -- while reusing the identical blocks, kernel
    dictionary, parameter shapes (one extra operator-lift input column, +32
    parameters at reference capacity), and training protocol of the paired
    two-stream arm.  No core model code is modified.

    Structural consequences, stated as falsifiable hypotheses against the
    paired two-stream ``singpair`` arm at matched parameters:

    - **H1** (cost of the contract): on the linear Laplace bank the
      two-stream model matches or beats the control in-distribution --
      the exact-superposition contract costs nothing.  Falsifier: the
      control wins ID by more than twice the across-seed sample sd.
    - **H2** (value of the contract): the two-stream model is decisively
      better on drive-OOD splits (``unseen_boundary_frequencies``), because
      linearity in the drive is structural rather than learned.  Falsifier:
      the control matches two-stream on that split.
    - **H3** (measured violation): the control has a nonzero zero-drive
      response and nonzero superposition residual.  This is trivially true
      structurally (the constant drive is never zero); what is measured is
      the *magnitude* after training, via the benchmark's first-class
      ``drive_linearity`` metrics.  On the *nonlinear* Liouville problem a
      nonzero zero-drive response is physically correct (the exact
      zero-Dirichlet Liouville solution on the unit disk is
      ``u = log 4 - 2 log(1 + |w|**2)``, ``u(0) = log 4``), so there the same
      measurement prices the two-stream contract as a *misspecification*.

    The wrapper accepts the benchmark's standard single-``dirichlet`` scalar
    Dirichlet domains and injects the constant drive on the fly; callers and
    evaluation harnesses treat it exactly like every other candidate.
    """

    #: Name of the injected constant drive field (value 1 per boundary cell).
    UNIT_DRIVE_KEY = "unit_drive"

    def __init__(self, fused_model: MeshTransformer) -> None:
        super().__init__()
        drive_ranks = fused_model.boundary_field_ranks["dirichlet"]["drive"]
        if set(drive_ranks) != {self.UNIT_DRIVE_KEY}:
            raise ValueError(
                "fused_model must declare exactly the constant "
                f"{self.UNIT_DRIVE_KEY!r} drive field, got {sorted(drive_ranks)}"
            )
        self.fused_model = fused_model

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        values = boundary.cell_data["boundary_value"]
        fused_boundary = boundary.with_data(
            cell_data={
                "boundary_value": values,
                self.UNIT_DRIVE_KEY: torch.ones_like(values),
            }
        )
        fused_domain = DomainMesh(
            interior=domain.interior,
            boundaries={"dirichlet": fused_boundary},
            global_data=domain.global_data,
        )
        return self.fused_model(fused_domain)


def build_single_stream_control(
    config: MeshTransformerConfig,
    *,
    field_mode: str = "linear",
    reference_length_key: str | None = "reference_length",
) -> SingleStreamFusionControl:
    """Construct the single-stream control paired with the ``singpair`` arm.

    Everything except the stream assignment of ``boundary_value`` matches the
    two-stream ``mesh_transformer_kernel_singpair`` arm bitwise: the same
    capacity ``config``, the same two-member exact singular kernel dictionary
    (double layer plus single layer; no polynomial and no MLP smooth
    members), and the same ``field_mode``.  ``field_mode`` here governs only
    how the constant unit drive propagates; the map from physical boundary
    values to the prediction is unrestricted (nonlinear, biased) in *both*
    settings, which is the point of the control.  See
    :class:`SingleStreamFusionControl` for the pre-registered hypotheses and
    the measurement-only status of this arm.
    """

    return SingleStreamFusionControl(
        build_mesh_transformer(
            config,
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            field_mode=field_mode,
            boundary_field_ranks={
                "dirichlet": {
                    "operator": {"boundary_value": 0},
                    "drive": {SingleStreamFusionControl.UNIT_DRIVE_KEY: 0},
                }
            },
            reference_length_key=reference_length_key,
        )
    )


def _benchmark_boundary(domain: DomainMesh) -> Mesh:
    if set(domain.boundaries.keys()) != {"dirichlet"}:
        raise ValueError("benchmark domains must contain only a 'dirichlet' boundary")
    boundary = domain.boundaries["dirichlet"]
    if "boundary_value" not in boundary.cell_data:
        raise ValueError("dirichlet.cell_data must contain 'boundary_value'")
    return boundary


def _reference_length(domain: DomainMesh) -> torch.Tensor:
    try:
        length = domain.global_data["reference_length"].reshape(())
    except KeyError:
        raise ValueError("domain.global_data must contain 'reference_length'") from None
    if not torch.compiler.is_compiling() and (
        not torch.isfinite(length).item() or length.item() <= 0.0
    ):
        raise ValueError("reference_length must be finite and positive")
    return length


def _constant_exact_boundary_mean(
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Return a linear quadrature mean that preserves constants bit-for-bit."""

    anchor = values[0]
    return anchor + torch.sum(weights * (values - anchor)) / weights.sum()


def _prediction_mesh(domain: DomainMesh, potential: torch.Tensor) -> Mesh:
    return domain.interior.with_data(
        point_data={"potential": potential},
        cell_data={},
        global_data=domain.global_data,
    )


class BoundaryMean(nn.Module):
    r"""Parameter-free, quadrature-weighted constant baseline.

    This baseline exactly reproduces constant Dirichlet data.  It deliberately
    has no spatial capacity and therefore quantifies how much a learned model
    improves over predicting one domain-wide value.
    """

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        weights = boundary.cell_areas
        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        return _prediction_mesh(domain, mean.expand(domain.interior.n_points))


class InvariantPairKernel(nn.Module):
    r"""Dense linear pair-kernel baseline with the same physical symmetries.

    For normalized query position :math:`x`, source centroid :math:`y`, and
    outward normal :math:`n`, the learned kernel sees only

    .. math::

        (\lVert x-y\rVert^2,\; n\cdot(x-y)).

    These are joint O(2) invariants and contain no absolute position or fitted
    interaction radius.  The output is linear in the boundary data and uses
    the boundary measure.  Unlike ``MeshTransformer``, it materializes dense
    query-source pairs and has no global geometry encoder.  It is therefore a
    useful control for the expressiveness/cost tradeoff, not a proposed
    production architecture.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or hidden_layers < 1 or query_chunk_size < 1:
            raise ValueError(
                "hidden_dim, hidden_layers, and chunk size must be positive"
            )

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        final = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(final.weight, std=1.0e-2 / hidden_dim**0.5)
        layers.append(final)
        self.kernel = nn.Sequential(*layers)
        self.query_chunk_size = query_chunk_size

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = _benchmark_boundary(domain)
        length = _reference_length(domain)
        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            total_measure = weights.sum()
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / total_measure
            source_points = (boundary.cell_centroids - center) / length
            query_points = (domain.interior.points - center) / length
            normals = boundary.cell_normals

        values = boundary.cell_data["boundary_value"]
        mean = _constant_exact_boundary_mean(weights, values)
        residual = values - mean

        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            query = query_points[start : start + self.query_chunk_size]
            displacement = query[:, None, :] - source_points[None, :, :]
            features = torch.stack(
                (
                    displacement.square().sum(dim=-1),
                    torch.einsum("qsd,sd->qs", displacement, normals),
                ),
                dim=-1,
            )
            pair_kernel = self.kernel(features).squeeze(-1)
            chunks.append(
                mean + torch.einsum("qs,s,s->q", pair_kernel, weights, residual)
            )

        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return _prediction_mesh(domain, potential)


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


__all__ = [
    "BoundaryMean",
    "InvariantPairKernel",
    "MeanLiftedDirichletModel",
    "MeshTransformerConfig",
    "SingleStreamFusionControl",
    "build_lifted_mesh_transformer",
    "build_mesh_transformer",
    "build_single_stream_control",
    "parameter_count",
]
