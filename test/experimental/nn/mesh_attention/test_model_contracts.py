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

"""Integration contracts for the boundary-driven mesh transformer."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention import (
    CanonicalSourceGeometry,
    MeshTransformer,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import (
    MEASURE_WEIGHTS_KEY,
    cell_measure_weights,
    cell_measures,
)

_BOUNDARY_RANKS = {
    "wall": {
        "operator": {"material": 0},
        "drive": {"forcing": 0, "traction": 1},
    }
}
_GLOBAL_RANKS = {
    "operator": {"reynolds": 0, "flow_direction": 1},
    "drive": {"source_strength": 0},
}
_OUTPUT_RANKS = {"pressure": 0, "velocity": 1}


def _model(
    device: torch.device | str,
    *,
    field_mode: str = "linear",
    query_decoder: str = "moment",
    dtype: torch.dtype = torch.float64,
    query_chunk_size: int = 2,
    reference_length_key: str | None = "reference.length",
    bounded_query_geometry: bool = False,
    measure_normalization: bool = False,
    boundary_field_ranks: dict | None = None,
) -> MeshTransformer:
    torch.manual_seed(732)
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=(
            _BOUNDARY_RANKS if boundary_field_ranks is None else boundary_field_ranks
        ),
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key=reference_length_key,
        measure_normalization=measure_normalization,
        field_mode=field_mode,
        query_decoder=query_decoder,
        bounded_query_geometry=bounded_query_geometry,
        operator_scalar_dim=5,
        operator_vector_dim=3,
        drive_scalar_dim=6,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=query_chunk_size,
    ).to(device=device, dtype=dtype)
    # Property tests should not silently accumulate float64 inputs in float32.
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = dtype
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.2, 0.2)
    model.eval()
    return model


def _orthogonal(
    device: torch.device | str,
    *,
    reflection: bool,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(219)
    transform, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    if (torch.linalg.det(transform) < 0).item() != reflection:
        transform[:, 0] *= -1
    return transform.to(device)


def _domain(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float64,
    forcing: torch.Tensor | None = None,
    traction: torch.Tensor | None = None,
    source_strength: torch.Tensor | float = 0.35,
    transform: torch.Tensor | None = None,
    reverse_boundary_orientation: bool = False,
    scale: float = 1.0,
    translation: torch.Tensor | None = None,
    source_permutation: torch.Tensor | None = None,
    query_permutation: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> DomainMesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.1],
            [0.1, 1.1, -0.1],
            [-0.1, 0.2, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    # Consistently oriented faces of the tetrahedron.
    cells = torch.tensor(
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        device=device,
        dtype=torch.long,
    )
    query_points = torch.tensor(
        [
            [0.18, 0.21, 0.17],
            [0.42, 0.11, 0.23],
            [0.08, 0.56, 0.19],
            [0.25, 0.22, 0.51],
            [0.91, 0.77, 0.62],
        ],
        device=device,
        dtype=dtype,
    )
    query_cells = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.long)
    material = torch.tensor([0.2, 0.7, -0.3, 1.1], device=device, dtype=dtype)
    if forcing is None:
        forcing = torch.tensor([0.8, -0.4, 1.2, 0.3], device=device, dtype=dtype)
    else:
        forcing = forcing.to(device=device, dtype=dtype)
    if traction is None:
        traction = torch.tensor(
            [
                [0.3, -0.2, 0.1],
                [0.7, 0.4, -0.3],
                [-0.1, 0.8, 0.2],
                [0.5, -0.6, 0.9],
            ],
            device=device,
            dtype=dtype,
        )
    else:
        traction = traction.to(device=device, dtype=dtype)
    flow_direction = torch.tensor([0.9, 0.3, -0.2], device=device, dtype=dtype)

    if transform is not None:
        points = torch.einsum("nd,ed->ne", points, transform)
        query_points = torch.einsum("nd,ed->ne", query_points, transform)
        traction = torch.einsum("nd,ed->ne", traction, transform)
        flow_direction = torch.einsum("d,ed->e", flow_direction, transform)
    if reverse_boundary_orientation:
        # An orientation-reversing coordinate change must also reverse simplex
        # winding for oriented cell normals to represent the same polar normal.
        cells = cells[:, [0, 2, 1]]

    if translation is None:
        translation = points.new_zeros(3)
    else:
        translation = translation.to(device=device, dtype=dtype)
    points = scale * points + translation
    query_points = scale * query_points + translation

    if source_permutation is not None:
        source_permutation = source_permutation.to(device=device)
        cells = cells[source_permutation]
        material = material[source_permutation]
        forcing = forcing[source_permutation]
        traction = traction[source_permutation]

    target = target_scale * torch.arange(
        query_points.shape[0], device=device, dtype=dtype
    )
    if query_permutation is not None:
        query_permutation = query_permutation.to(device=device)
        query_points = query_points[query_permutation]
        target = target[query_permutation]
        inverse = torch.empty_like(query_permutation)
        inverse[query_permutation] = torch.arange(
            query_permutation.numel(), device=device
        )
        query_cells = inverse[query_cells]

    boundary = Mesh(
        points=points,
        cells=cells,
        cell_data={
            "material": material,
            "forcing": forcing,
            "traction": traction,
        },
    )
    interior = Mesh(
        points=query_points,
        cells=query_cells,
        point_data={
            "pressure": target,
            "velocity": target[:, None].expand(-1, 3),
            "unrelated_target": -target,
        },
        cell_data={"target_cell_value": target.new_tensor([target_scale])},
    )
    strength = torch.as_tensor(source_strength, device=device, dtype=dtype)
    return DomainMesh(
        interior=interior,
        boundaries={"wall": boundary},
        global_data={
            "reference": {"length": points.new_tensor(2.3 * scale)},
            "reynolds": points.new_tensor(0.61),
            "flow_direction": flow_direction,
            "source_strength": strength,
            "unused_target": points.new_tensor(1000.0 * target_scale),
        },
    )


def _assert_output_close(
    actual: Mesh,
    expected: Mesh,
    *,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-11,
) -> None:
    torch.testing.assert_close(
        actual.point_data["pressure"],
        expected.point_data["pressure"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        actual.point_data["velocity"],
        expected.point_data["velocity"],
        rtol=rtol,
        atol=atol,
    )


def test_domain_schema_failures_are_explicit(device):
    model = _model(device)
    domain = _domain(device)

    with pytest.raises(TypeError, match="DomainMesh"):
        model(torch.zeros(2, 3, device=device, dtype=torch.float64))

    with pytest.raises(ValueError, match="boundary names"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={},
                global_data=domain.global_data,
            )
        )

    with pytest.raises(ValueError, match="unexpected=.*inlet"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={
                    "wall": domain.boundaries["wall"],
                    "inlet": domain.boundaries["wall"],
                },
                global_data=domain.global_data,
            )
        )

    volumetric = Mesh(
        points=domain.boundaries["wall"].points,
        cells=torch.tensor([[0, 1, 2, 3]], device=device),
        cell_data={
            "material": torch.ones(1, device=device, dtype=torch.float64),
            "forcing": torch.ones(1, device=device, dtype=torch.float64),
            "traction": torch.ones(1, 3, device=device, dtype=torch.float64),
        },
    )
    with pytest.raises(ValueError, match="codimension one"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={"wall": volumetric},
                global_data=domain.global_data,
            )
        )

    boundary = domain.boundaries["wall"]
    missing_drive = boundary.with_data(
        cell_data={
            "material": boundary.cell_data["material"],
            "traction": boundary.cell_data["traction"],
        }
    )
    with pytest.raises(ValueError, match="missing leaf 'forcing'"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={"wall": missing_drive},
                global_data=domain.global_data,
            )
        )

    invalid_global = domain.global_data.clone()
    invalid_global["reference", "length"] = torch.tensor(
        -1.0, device=device, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="finite and positive"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={"wall": boundary},
                global_data=invalid_global,
            )
        )

    float_boundary = boundary.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="share a dtype"):
        model(
            DomainMesh(
                interior=domain.interior,
                boundaries={"wall": float_boundary},
                global_data=domain.global_data,
            )
        )

    with pytest.raises(ValueError, match="conflicting ranks"):
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"out": 0},
            boundary_field_ranks={
                "a": {"drive": {"same_name": 0}},
                "b": {"drive": {"same_name": 1}},
            },
            operator_scalar_dim=2,
            operator_vector_dim=1,
            drive_scalar_dim=2,
            drive_vector_dim=1,
            operator_layers=0,
            drive_layers=0,
            query_layers=1,
            heads=1,
            scalar_rank=1,
            vector_rank=0,
        )

    with pytest.raises(ValueError, match="both operator and drive roles"):
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"out": 0},
            boundary_field_ranks={
                "wall": {
                    "operator": {"ambiguous": 0},
                    "drive": {"ambiguous": 0},
                }
            },
        )

    with pytest.raises(ValueError, match="must not also be a learned field"):
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"out": 0},
            boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
            global_field_ranks={"operator": {"reference": {"length": 0}}},
            reference_length_key="reference.length",
        )


def test_constructor_schemas_are_frozen_from_caller_mutation():
    output_ranks = {"pressure": 0}
    boundary_ranks = {"wall": {"drive": {"forcing": 0}}}
    global_ranks = {"operator": {"coefficient": 0}}
    model = MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks=output_ranks,
        boundary_field_ranks=boundary_ranks,
        global_field_ranks=global_ranks,
        operator_scalar_dim=2,
        operator_vector_dim=1,
        drive_scalar_dim=2,
        drive_vector_dim=1,
        operator_layers=0,
        drive_layers=0,
        query_layers=1,
        heads=1,
        scalar_rank=1,
        vector_rank=0,
    )

    output_ranks["pressure"] = 1
    boundary_ranks["wall"]["drive"]["forcing"] = 1
    global_ranks["operator"]["coefficient"] = 1

    assert model.output_field_ranks == {"pressure": 0}
    assert model.boundary_field_ranks == {"wall": {"drive": {"forcing": 0}}}
    assert model.global_field_ranks == {"operator": {"coefficient": 0}}
    assert model._args["output_field_ranks"] == {"pressure": 0}
    assert model.boundary_names == ("wall",)


def _intrinsic_gauge_model(device, **overrides):
    kwargs = dict(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0},
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key=None,
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=0,
        drive_layers=0,
        query_layers=1,
        heads=1,
        scalar_rank=1,
        vector_rank=1,
    )
    kwargs.update(overrides)
    return MeshTransformer(**kwargs).to(device=device, dtype=torch.float64)


def test_none_reference_length_uses_intrinsic_radius_of_gyration(device):
    r"""The default gauge is the measure-weighted RMS boundary radius.

    ``reference_length_key=None`` derives :math:`L` intrinsically as the
    radius of gyration of the boundary quadrature about its measure-weighted
    centroid, accumulated in float64.  Degree-1 positive homogeneity makes
    the normalized source frame exactly scale invariant.
    """
    model = _intrinsic_gauge_model(device)

    domain = _domain(device)
    encoded = model.encode(domain)
    scale = 3.1
    scaled = model.encode(_domain(device, scale=scale))

    boundary = domain.boundaries["wall"]
    weights = boundary.cell_areas.double()
    centroids = boundary.cell_centroids.double()
    center = torch.einsum("n,nd->d", weights, centroids) / weights.sum()
    expected = torch.sqrt(
        torch.einsum("n,n->", weights, (centroids - center).square().sum(-1))
        / weights.sum()
    )

    torch.testing.assert_close(
        encoded.reference_length, expected, rtol=2.0e-15, atol=0.0
    )
    torch.testing.assert_close(
        scaled.reference_length,
        scale * encoded.reference_length,
        rtol=2.0e-14,
        atol=0.0,
    )
    # The intrinsic gauge cancels the physical scale: the normalized source
    # frame and its quadrature are invariants of the similarity class.
    torch.testing.assert_close(
        scaled.center,
        scale * encoded.center,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        scaled.source_mesh.points,
        encoded.source_mesh.points,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        scaled.source_mesh.cell_areas,
        encoded.source_mesh.cell_areas,
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_explicit_reference_length_key_bypasses_intrinsic_gauge(device, monkeypatch):
    """The explicit-key override is bitwise the pre-intrinsic behavior.

    Mirrors the knob-discipline regressions (polynomial/single-layer/
    scalar-only/pseudo): with a key supplied the model must consume exactly
    the declared scalar and never touch the intrinsic estimator, so its
    numerics are unchanged from before the intrinsic default existed.
    """
    model = _model(device)
    domain = _domain(device)

    def _forbidden(self, *args, **kwargs):
        raise AssertionError(
            "intrinsic gauge must not run when reference_length_key is set"
        )

    monkeypatch.setattr(MeshTransformer, "_intrinsic_reference_length", _forbidden)
    encoded = model.encode(domain)
    torch.testing.assert_close(
        encoded.reference_length,
        domain.global_data["reference", "length"].reshape(()),
        rtol=0.0,
        atol=0.0,
    )
    prediction = model.decode(encoded)
    assert torch.isfinite(prediction.point_data["pressure"]).all()


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
@pytest.mark.parametrize("scale", [0.2, 5.0])
def test_intrinsic_gauge_similarity_contract_across_scales(
    device, scale, query_decoder
):
    """Scale equivariance without any declared reference length.

    The intrinsic gauge is degree-1 homogeneous, so the similarity contract
    holds unconditionally -- no global-data length has to be kept consistent
    by the caller across the 0.2x-5x range.
    """
    intrinsic = _model(device, query_decoder=query_decoder, reference_length_key=None)

    original = intrinsic(_domain(device))
    transformed = intrinsic(
        _domain(
            device,
            scale=scale,
            translation=torch.tensor([8.2, -4.3, 1.7]),
        )
    )
    _assert_output_close(transformed, original)


def test_intrinsic_gauge_is_immune_to_reference_length_corruption(device):
    """Convention-drift immunity: no scale input exists to corrupt.

    An explicit-key model changes its predictions when the declared
    reference length drifts (here scaled 3x at evaluation time); the
    intrinsic-gauge model with identical weights has no such input, so the
    identical corruption is a bitwise no-op.  This asserts the structural
    contrast only; the magnitude of the accuracy degradation on a trained
    model is demonstrated in the mesh_transformer example's gauge tests.
    """
    explicit = _model(device)
    intrinsic = _model(device, reference_length_key=None)

    def corrupted_domain():
        domain = _domain(device)
        domain.global_data["reference", "length"] = (
            3.0 * domain.global_data["reference", "length"]
        )
        return domain

    with torch.no_grad():
        explicit_clean = explicit(_domain(device)).point_data["pressure"]
        explicit_drift = explicit(corrupted_domain()).point_data["pressure"]
        intrinsic_clean = intrinsic(_domain(device)).point_data["pressure"]
        intrinsic_drift = intrinsic(corrupted_domain()).point_data["pressure"]

    torch.testing.assert_close(intrinsic_drift, intrinsic_clean, rtol=0.0, atol=0.0)
    relative_drift = torch.linalg.norm(
        explicit_drift - explicit_clean
    ) / torch.linalg.norm(explicit_clean)
    # An untrained model shows a genuine (not roundoff) sensitivity; trained
    # models degrade by orders of magnitude more (see the example demo).
    assert float(relative_drift) > 1.0e-6


@pytest.mark.parametrize("length", [0.0, float("nan"), float("inf"), -float("inf")])
def test_reference_length_rejects_nonpositive_or_nonfinite_values(device, length):
    model = _model(device)
    domain = _domain(device)
    domain.global_data["reference", "length"] = torch.tensor(
        length, device=device, dtype=torch.float64
    )

    with pytest.raises(ValueError, match="finite and positive"):
        model.encode(domain)


def test_reference_length_rejects_nonscalar_or_mismatched_dtype(device):
    model = _model(device)
    domain = _domain(device)
    domain.global_data["reference", "length"] = torch.ones(
        2, device=device, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="must be scalar"):
        model.encode(domain)

    domain = _domain(device)
    domain.global_data["reference", "length"] = torch.tensor(
        2.3, device=device, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="share mesh device and dtype"):
        model.encode(domain)


@pytest.mark.parametrize("failure", ["zero_measure", "nonfinite_measure"])
def test_boundary_cells_require_finite_positive_measure(device, failure):
    model = _model(device)
    domain = _domain(device)
    boundary = domain.boundaries["wall"]
    points = boundary.points.clone()
    cells = boundary.cells.clone()
    if failure == "zero_measure":
        cells[0] = cells[0, 0]
    else:
        points[0, 0] = float("nan")
    invalid_boundary = Mesh(
        points=points,
        cells=cells,
        cell_data=boundary.cell_data,
    )
    invalid_domain = DomainMesh(
        interior=domain.interior,
        boundaries={"wall": invalid_boundary},
        global_data=domain.global_data,
    )

    with pytest.raises(ValueError, match="finite positive measure"):
        model.encode(invalid_domain)


def test_vector_only_output_schema(device):
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks={"velocity": 1},
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key="reference.length",
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=0,
        drive_layers=0,
        query_layers=1,
        heads=1,
        scalar_rank=1,
        vector_rank=1,
    ).to(device=device, dtype=torch.float64)

    output = model(_domain(device))

    assert set(output.point_data.keys()) == {"velocity"}
    assert output.point_data["velocity"].shape == (5, 3)


def test_heterogeneous_boundary_schemas_share_one_canonical_source(device):
    dtype = torch.float64
    wall = Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            device=device,
            dtype=dtype,
        ),
        cells=torch.tensor([[0, 1], [1, 2]], device=device),
        cell_data={
            "roughness": torch.tensor([0.1, 0.2], device=device, dtype=dtype),
            "temperature": torch.tensor([1.0, 2.0], device=device, dtype=dtype),
        },
    )
    inlet = Mesh(
        points=torch.tensor(
            [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        ),
        cells=torch.tensor([[0, 1], [1, 2]], device=device),
        cell_data={
            "velocity": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]], device=device, dtype=dtype
            )
        },
    )
    query = Mesh(
        points=torch.tensor([[0.5, 0.5], [0.25, 0.75]], device=device, dtype=dtype)
    )
    model = MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"pressure": 0, "velocity": 1},
        boundary_field_ranks={
            "wall": {
                "operator": {"roughness": 0},
                "drive": {"temperature": 0},
            },
            "inlet": {"drive": {"velocity": 1}},
        },
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=1,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
    ).to(device=device, dtype=dtype)

    first = model(DomainMesh(query, {"wall": wall, "inlet": inlet}))
    second = model(DomainMesh(query, {"inlet": inlet, "wall": wall}))

    _assert_output_close(first, second, rtol=0, atol=0)
    assert first.point_data["pressure"].shape == (2,)
    assert first.point_data["velocity"].shape == (2, 2)


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_forward_mesh_contract_and_target_data_nonleakage(device, query_decoder):
    model = _model(device, query_decoder=query_decoder)
    first_domain = _domain(device, target_scale=1.0)
    second_domain = _domain(device, target_scale=913.0)

    first = model(first_domain)
    second = model(second_domain)
    assert isinstance(first, Mesh)
    torch.testing.assert_close(first.points, first_domain.interior.points)
    torch.testing.assert_close(first.cells, first_domain.interior.cells)
    assert set(first.point_data.keys()) == {"pressure", "velocity"}
    assert first.point_data.batch_size == torch.Size([first.n_points])
    assert not first.cell_data.keys(include_nested=True, leaves_only=True)
    assert set(first.global_data.keys(include_nested=True, leaves_only=True)) == set(
        first_domain.global_data.keys(include_nested=True, leaves_only=True)
    )
    # Point/cell targets and undeclared global data are not model inputs.
    _assert_output_close(first, second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_linear_mode_has_zero_and_superposition_laws(device, query_decoder):
    model = _model(device, field_mode="linear", query_decoder=query_decoder)
    dtype = torch.float64
    first_forcing = torch.tensor([0.2, -0.7, 1.1, 0.4], dtype=dtype)
    second_forcing = torch.tensor([-0.5, 0.8, 0.3, -1.2], dtype=dtype)
    first_traction = torch.arange(12, dtype=dtype).reshape(4, 3) / 9.0
    second_traction = torch.arange(12, dtype=dtype).reshape(4, 3).flip(0) / 13.0
    alpha, beta = 1.37, -0.41

    first = model(
        _domain(
            device,
            forcing=first_forcing,
            traction=first_traction,
            source_strength=0.23,
        )
    )
    second = model(
        _domain(
            device,
            forcing=second_forcing,
            traction=second_traction,
            source_strength=-0.61,
        )
    )
    combined = model(
        _domain(
            device,
            forcing=alpha * first_forcing + beta * second_forcing,
            traction=alpha * first_traction + beta * second_traction,
            source_strength=alpha * 0.23 + beta * -0.61,
        )
    )
    for name in ("pressure", "velocity"):
        torch.testing.assert_close(
            combined.point_data[name],
            alpha * first.point_data[name] + beta * second.point_data[name],
            rtol=3.0e-11,
            atol=3.0e-12,
        )

    zero = model(
        _domain(
            device,
            forcing=torch.zeros(4),
            traction=torch.zeros(4, 3),
            source_strength=0.0,
        )
    )
    assert torch.count_nonzero(zero.point_data["pressure"]).item() == 0
    assert torch.count_nonzero(zero.point_data["velocity"]).item() == 0


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_nonlinear_mode_is_exactly_zero_preserving(device, query_decoder):
    model = _model(
        device, field_mode="zero_preserving_nonlinear", query_decoder=query_decoder
    )
    zero = model(
        _domain(
            device,
            forcing=torch.zeros(4),
            traction=torch.zeros(4, 3),
            source_strength=0.0,
        )
    )
    assert torch.count_nonzero(zero.point_data["pressure"]).item() == 0
    assert torch.count_nonzero(zero.point_data["velocity"]).item() == 0


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_full_model_backward_reaches_geometry_fields_and_parameters(
    device, query_decoder
):
    """Training gradients reach every physical input path and model parameter."""
    model = _model(
        device,
        field_mode="linear",
        query_decoder=query_decoder,
        query_chunk_size=2,
    )
    domain = _domain(device)
    boundary = domain.boundaries["wall"]
    differentiable_inputs = {
        "boundary_points": boundary.points,
        "query_points": domain.interior.points,
        "material": boundary.cell_data["material"],
        "forcing": boundary.cell_data["forcing"],
        "traction": boundary.cell_data["traction"],
        "reynolds": domain.global_data["reynolds"],
        "flow_direction": domain.global_data["flow_direction"],
        "source_strength": domain.global_data["source_strength"],
        "reference_length": domain.global_data["reference", "length"],
    }
    for tensor in differentiable_inputs.values():
        tensor.requires_grad_()

    output = model(domain)
    pressure_cotangent = torch.linspace(
        0.7,
        1.3,
        output.n_points,
        device=device,
        dtype=torch.float64,
    )
    velocity_cotangent = torch.linspace(
        -0.8,
        1.1,
        output.n_points * model.n_spatial_dims,
        device=device,
        dtype=torch.float64,
    ).reshape(output.n_points, model.n_spatial_dims)
    loss = (output.point_data["pressure"] * pressure_cotangent).sum() + (
        output.point_data["velocity"] * velocity_cotangent
    ).sum()
    loss.backward()

    for name, tensor in differentiable_inputs.items():
        assert tensor.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(tensor.grad).all(), f"nonfinite gradient for {name}"
        assert torch.count_nonzero(tensor.grad).item(), f"zero gradient for {name}"

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient reached parameter {name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"nonfinite gradient for parameter {name}"
        )


def test_query_read_in_and_residual_scales_have_distinct_initialization(device):
    """The decoder begins with an order-one read, then small residual updates."""
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0},
        boundary_field_ranks={"wall": {"drive": {"forcing": 0}}},
        field_mode="linear",
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
    ).to(device=device, dtype=torch.float64)

    for scale in (
        model.query_blocks[0].message_scale.scalar_scale,
        model.query_blocks[0].message_scale.vector_scale,
    ):
        torch.testing.assert_close(scale, torch.ones_like(scale))
    for scale in (
        model.query_blocks[1].message_scale.scalar_scale,
        model.query_blocks[1].message_scale.vector_scale,
        model.query_blocks[0].pointwise_scale.scalar_scale,
        model.query_blocks[0].pointwise_scale.vector_scale,
        model.drive_blocks[0].message_scale.scalar_scale,
        model.drive_blocks[0].message_scale.vector_scale,
        model.operator_input_block.scale.scalar_scale,
        model.operator_input_block.scale.vector_scale,
    ):
        torch.testing.assert_close(scale, torch.full_like(scale, 1.0e-2))


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_global_only_drive_uses_every_decoder_parameter(device, query_decoder):
    """A pointwise global seed must not bypass the order-one query read path."""
    torch.manual_seed(733)
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0},
        boundary_field_ranks={"wall": {"operator": {"material": 0}}},
        global_field_ranks={"drive": {"source_strength": 0}},
        reference_length_key="reference.length",
        field_mode="linear",
        query_decoder=query_decoder,
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=3,
    ).to(device=device, dtype=torch.float64)
    domain = _domain(device)
    domain.global_data["source_strength"].requires_grad_()

    output = model(domain)
    output.point_data["pressure"].square().sum().backward()

    assert domain.global_data["source_strength"].grad is not None
    assert torch.count_nonzero(domain.global_data["source_strength"].grad)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient reached parameter {name}"
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_amp_preserves_mesh_geometry_precision(device, query_decoder):
    model = _model(device, query_decoder=query_decoder, dtype=torch.float32)
    domain = _domain(device, dtype=torch.float32)
    device_type = torch.device(device).type
    autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        encoded = model.encode(domain)
        output = model.decode(encoded)

    assert encoded.source_mesh.points.dtype == torch.float32
    assert encoded.source_mesh.cell_areas.dtype == torch.float32
    assert torch.isfinite(output.point_data["pressure"]).all()
    assert torch.isfinite(output.point_data["velocity"]).all()


@pytest.mark.parametrize("bounded_query_geometry", [False, True])
@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
@pytest.mark.parametrize("reflection", [False, True])
def test_model_is_o3_equivariant_with_consistent_orientation(
    device, reflection, query_decoder, bounded_query_geometry
):
    # The compactified query-position injection rescales each position by a
    # function of its own invariant norm, so O(3) equivariance must be
    # exactly preserved with the knob on.
    model = _model(
        device,
        query_decoder=query_decoder,
        bounded_query_geometry=bounded_query_geometry,
    )
    original = model(_domain(device))
    transform = _orthogonal(device, reflection=reflection)
    transformed = model(
        _domain(
            device,
            transform=transform,
            reverse_boundary_orientation=reflection,
        )
    )

    torch.testing.assert_close(
        transformed.point_data["pressure"],
        original.point_data["pressure"],
        rtol=3.0e-10,
        atol=3.0e-11,
    )
    torch.testing.assert_close(
        transformed.point_data["velocity"],
        torch.einsum("nd,ed->ne", original.point_data["velocity"], transform),
        rtol=3.0e-10,
        atol=3.0e-11,
    )


def test_encoded_frame_uses_boundary_measure_and_explicit_length(device):
    r"""Centering and normalized quadrature follow the declared measure.

    An unweighted average of cell centroids would retain translation and
    similarity covariance, yet vary under nonuniform remeshing.  Assert the
    stronger quadrature contract directly rather than inferring it from model
    outputs.
    """
    model = _model(device)
    domain = _domain(device)
    boundary = domain.boundaries["wall"]
    length = domain.global_data["reference", "length"].reshape(())
    physical_areas = boundary.cell_areas
    expected_center = (
        torch.einsum("n,nd->d", physical_areas, boundary.cell_centroids)
        / physical_areas.sum()
    )

    encoded = model.encode(domain)

    torch.testing.assert_close(
        encoded.center, expected_center, rtol=2.0e-14, atol=2.0e-14
    )
    torch.testing.assert_close(
        encoded.source_mesh.points,
        (boundary.points - expected_center) / length,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        encoded.source_mesh.cell_areas,
        physical_areas / length ** (model.n_spatial_dims - 1),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    torch.testing.assert_close(
        torch.einsum(
            "n,nd->d",
            encoded.source_mesh.cell_areas,
            encoded.source_mesh.cell_centroids,
        ),
        torch.zeros(model.n_spatial_dims, device=device, dtype=torch.float64),
        rtol=0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("bounded_query_geometry", [False, True])
@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_similarity_and_source_query_permutation_contracts(
    device, query_decoder, bounded_query_geometry
):
    # The compactification acts on the nondimensionalized coordinate, after
    # the reference-length division, so similarity covariance must be exactly
    # preserved with the knob on.
    model = _model(
        device,
        query_decoder=query_decoder,
        bounded_query_geometry=bounded_query_geometry,
    )
    original = model(_domain(device))

    transformed = model(
        _domain(
            device,
            scale=3.7,
            translation=torch.tensor([8.2, -4.3, 1.7]),
        )
    )
    _assert_output_close(transformed, original)

    source_permutation = torch.tensor([2, 0, 3, 1])
    source_permuted = model(_domain(device, source_permutation=source_permutation))
    _assert_output_close(source_permuted, original)

    query_permutation = torch.tensor([3, 0, 4, 1, 2])
    query_permuted = model(_domain(device, query_permutation=query_permutation))
    torch.testing.assert_close(
        query_permuted.point_data["pressure"],
        original.point_data["pressure"][query_permutation.to(device)],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        query_permuted.point_data["velocity"],
        original.point_data["velocity"][query_permutation.to(device)],
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def _smooth_circle_domain(
    n_cells: int,
    device: torch.device | str,
    *,
    graded: bool = False,
) -> DomainMesh:
    """Unit-circle boundary with smooth, resolution-independent field data."""
    dtype = torch.float64
    parameter = torch.arange(n_cells, device=device, dtype=dtype) / n_cells
    angles = 2.0 * torch.pi * parameter
    if graded:
        # A fixed smooth reparameterization with derivative
        # 2*pi*(1 + 0.35*cos(2*pi*s)) > 0 gives nonuniform nested panel
        # families without changing the represented circle.
        angles = angles + 0.35 * torch.sin(2.0 * torch.pi * parameter)
    points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    indices = torch.arange(n_cells, device=device)
    # Clockwise edge orientation makes the 2D cell normals point outward.
    cells = torch.stack((torch.roll(indices, -1), indices), dim=-1)
    centroids = 0.5 * (points + torch.roll(points, -1, dims=0))
    phase = torch.atan2(centroids[:, 1], centroids[:, 0])
    boundary = Mesh(
        points=points,
        cells=cells,
        cell_data={
            "coefficient": 0.3 + centroids[:, 0] - 0.2 * centroids[:, 1],
            "boundary_value": torch.cos(2.0 * phase) + 0.25 * torch.sin(3.0 * phase),
        },
    )
    query = Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [0.2, -0.1], [0.65, 0.15], [0.85, 0.1]],
            device=device,
            dtype=dtype,
        )
    )
    return DomainMesh(interior=query, boundaries={"wall": boundary})


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
@pytest.mark.parametrize("field_mode", ["linear", "zero_preserving_nonlinear"])
@pytest.mark.parametrize("graded", [False, True])
def test_mesh_transformer_converges_under_smooth_boundary_refinement(
    device, field_mode, graded, query_decoder
):
    r"""The learned operator has a stable continuum quadrature limit.

    This is not a PDE-accuracy assertion: one frozen randomized operator is
    evaluated on increasingly fine discretizations of the same smooth
    boundary data.  Its discretization error should contract at the expected
    midpoint-panel rate instead of depending on entity count or tessellation.
    """
    torch.manual_seed(918)
    model = MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"potential": 0, "flux": 1},
        boundary_field_ranks={
            "wall": {
                "operator": {"coefficient": 0},
                "drive": {"boundary_value": 0},
            }
        },
        reference_length_key=None,
        field_mode=field_mode,
        query_decoder=query_decoder,
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=1,
        heads=1,
        scalar_rank=2,
        vector_rank=1,
        query_chunk_size=8,
        attention_chunk_size=None,
    ).to(device=device, dtype=torch.float64)
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.3, 0.3)
    model.eval()

    def packed_output(n_cells: int) -> torch.Tensor:
        output = model(_smooth_circle_domain(n_cells, device, graded=graded))
        return torch.cat(
            (
                output.point_data["potential"][:, None],
                output.point_data["flux"],
            ),
            dim=-1,
        )

    with torch.no_grad():
        reference = packed_output(256)
        errors = [
            (packed_output(n_cells) - reference).abs().max() for n_cells in (16, 32, 64)
        ]

    roundoff = 20.0 * torch.finfo(torch.float64).eps
    assert errors[1] <= torch.maximum(0.45 * errors[0], errors[0].new_tensor(roundoff))
    assert errors[2] <= torch.maximum(0.45 * errors[1], errors[1].new_tensor(roundoff))


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
@pytest.mark.parametrize("field_mode", ["linear", "zero_preserving_nonlinear"])
def test_encode_decode_reuse_and_query_chunk_invariance(
    device, field_mode, query_decoder
):
    model = _model(
        device,
        field_mode=field_mode,
        query_decoder=query_decoder,
        query_chunk_size=1,
    )
    domain = _domain(device)
    encoded = model.encode(domain)
    assert not encoded.query_mesh.point_data.keys(include_nested=True, leaves_only=True)
    with pytest.raises(TypeError, match="query_mesh must be a Mesh"):
        model.decode(encoded, torch.zeros(2, 3, device=device))

    one_at_a_time = model.decode(encoded)
    model.query_chunk_size = 64
    all_at_once = model.decode(encoded)
    direct = model(domain)
    _assert_output_close(all_at_once, one_at_a_time)
    _assert_output_close(direct, one_at_a_time)

    subset_indices = torch.tensor([4, 1, 3], device=device)
    alternate_query = Mesh(
        points=domain.interior.points[subset_indices],
        point_data={
            "pressure": torch.full((3,), 1.0e9, device=device, dtype=torch.float64)
        },
    )
    subset = model.decode(encoded, alternate_query)
    torch.testing.assert_close(
        subset.point_data["pressure"],
        one_at_a_time.point_data["pressure"][subset_indices],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        subset.point_data["velocity"],
        one_at_a_time.point_data["velocity"][subset_indices],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert set(subset.point_data.keys()) == {"pressure", "velocity"}

    empty = model.decode(
        encoded,
        Mesh(points=domain.interior.points[:0]),
    )
    assert empty.point_data["pressure"].shape == (0,)
    assert empty.point_data["velocity"].shape == (0, 3)


def test_decode_reuses_cached_query_moments(device, monkeypatch):
    model = _model(device)
    encoded = model.encode(_domain(device))
    expected = model.decode(encoded)
    assert len(encoded.query_moments) == len(model.query_blocks)

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("decode rebuilt source moments")

    for block in model.query_blocks:
        monkeypatch.setattr(block, "build_source_moments", fail_if_rebuilt)

    actual = model.decode(encoded)
    _assert_output_close(actual, expected)


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_mdlus_checkpoint_roundtrip(device, tmp_path, query_decoder):
    model = _model(device, query_decoder=query_decoder, dtype=torch.float32)
    domain = _domain(device, dtype=torch.float32)
    expected = model(domain)
    checkpoint = tmp_path / "mesh_transformer.mdlus"
    model.save(checkpoint)

    restored = MeshTransformer.from_checkpoint(checkpoint).to(device=device)
    restored.eval()
    actual = restored(domain)
    _assert_output_close(actual, expected, rtol=2.0e-6, atol=2.0e-7)


def _prescribed_canonical_geometry(
    model: MeshTransformer,
    domain: DomainMesh,
) -> CanonicalSourceGeometry:
    """Build a valid bundle whose cached fields intentionally are not derived."""
    with torch.no_grad():
        historical = model.encode(domain)
    source = historical.source_mesh
    offsets = source.points.new_tensor([0.17, -0.11, 0.23])
    area_factors = torch.linspace(
        1.1,
        1.4,
        source.n_cells,
        device=source.points.device,
        dtype=source.points.dtype,
    )
    return CanonicalSourceGeometry(
        points=(source.points + offsets).clone(),
        cells=source.cells.clone(),
        centroids=(source.cell_centroids.flip(0) - offsets).clone(),
        areas=(source.cell_areas.flip(0) * area_factors).clone(),
        normals=(-source.cell_normals.roll(1, dims=0)).clone(),
        center=torch.zeros_like(historical.center),
        reference_length=torch.ones_like(historical.reference_length),
    )


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_canonical_source_geometry_is_public_and_authoritative_on_cpu(query_decoder):
    model = _model("cpu", query_decoder=query_decoder)
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)

    # These fields deliberately disagree with geometry re-derived from the
    # prescribed points. The public encode must therefore consume the bundle,
    # not merely accept and then ignore it.
    derived = Mesh(points=prescribed.points, cells=prescribed.cells)
    assert not torch.equal(derived.cell_centroids, prescribed.centroids)
    assert not torch.equal(derived.cell_areas, prescribed.areas)
    assert not torch.equal(derived.cell_normals, prescribed.normals)

    with torch.no_grad():
        encoded = model.encode(domain, canonical_source_geometry=prescribed)
        prediction = model.decode(encoded)

    torch.testing.assert_close(
        encoded.source_mesh.points, prescribed.points, rtol=0.0, atol=0.0
    )
    assert torch.equal(encoded.source_mesh.cells, prescribed.cells)
    for name, expected in (
        ("cell_centroids", prescribed.centroids),
        ("cell_areas", prescribed.areas),
        ("cell_normals", prescribed.normals),
    ):
        actual = getattr(encoded.source_mesh, name)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert actual.data_ptr() == expected.data_ptr()
    torch.testing.assert_close(encoded.center, prescribed.center, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        encoded.reference_length,
        prescribed.reference_length,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(prediction.point_data["pressure"]).all()
    assert torch.isfinite(prediction.point_data["velocity"]).all()


def test_canonical_source_geometry_default_off_is_bitwise_on_cpu():
    model = _model("cpu")
    with torch.no_grad():
        omitted = model.encode(_domain("cpu"))
        explicit_none = model.encode(_domain("cpu"), canonical_source_geometry=None)
        omitted_prediction = model.decode(omitted)
        explicit_none_prediction = model.decode(explicit_none)

    for omitted_value, explicit_value in (
        (omitted.source_mesh.points, explicit_none.source_mesh.points),
        (omitted.source_mesh.cells, explicit_none.source_mesh.cells),
        (omitted.source_mesh.cell_centroids, explicit_none.source_mesh.cell_centroids),
        (omitted.source_mesh.cell_areas, explicit_none.source_mesh.cell_areas),
        (omitted.source_mesh.cell_normals, explicit_none.source_mesh.cell_normals),
        (omitted.center, explicit_none.center),
        (omitted.reference_length, explicit_none.reference_length),
        (
            omitted_prediction.point_data["pressure"],
            explicit_none_prediction.point_data["pressure"],
        ),
        (
            omitted_prediction.point_data["velocity"],
            explicit_none_prediction.point_data["velocity"],
        ),
    ):
        assert torch.equal(omitted_value, explicit_value)


def test_canonical_source_geometry_measure_normalization_preserves_areas_on_cpu():
    model = _model("cpu", measure_normalization=True)
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)

    with torch.no_grad():
        encoded = model.encode(domain, canonical_source_geometry=prescribed)

    assert encoded.source_mesh.cell_areas.data_ptr() == prescribed.areas.data_ptr()
    assert torch.equal(encoded.source_mesh.cell_areas, prescribed.areas)
    expected_factor = (1.0 / prescribed.areas.double().sum()).to(prescribed.areas.dtype)
    torch.testing.assert_close(
        cell_measure_weights(encoded.source_mesh),
        torch.ones_like(prescribed.areas) * expected_factor,
        rtol=0.0,
        atol=0.0,
    )
    tolerance = 8.0 * torch.finfo(prescribed.areas.dtype).eps
    torch.testing.assert_close(
        cell_measures(encoded.source_mesh).sum(),
        prescribed.areas.new_tensor(1.0),
        rtol=tolerance,
        atol=tolerance,
    )


def test_canonical_source_geometry_validates_merged_boundary_order_on_cpu():
    boundary_ranks = {
        "wall_a": _BOUNDARY_RANKS["wall"],
        "wall_b": _BOUNDARY_RANKS["wall"],
    }
    model = _model("cpu", boundary_field_ranks=boundary_ranks)
    original = _domain("cpu")
    boundary = original.boundaries["wall"]
    domain = DomainMesh(
        interior=original.interior,
        boundaries={"wall_a": boundary, "wall_b": boundary},
        global_data=original.global_data,
    )
    prescribed = _prescribed_canonical_geometry(model, domain)

    expected_cells = torch.cat(
        (boundary.cells, boundary.cells + boundary.n_points),
        dim=0,
    )
    assert model.boundary_names == ("wall_a", "wall_b")
    assert torch.equal(prescribed.cells, expected_cells)
    with torch.no_grad():
        encoded = model.encode(domain, canonical_source_geometry=prescribed)
    assert torch.equal(encoded.source_mesh.cells, expected_cells)

    n_boundary_cells = boundary.n_cells
    swapped_boundaries = replace(
        prescribed,
        cells=torch.cat(
            (
                prescribed.cells[n_boundary_cells:],
                prescribed.cells[:n_boundary_cells],
            ),
            dim=0,
        ),
    )
    with pytest.raises(
        ValueError,
        match="merged boundary topology and cell ordering",
    ):
        model.encode(domain, canonical_source_geometry=swapped_boundaries)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("points_shape", r"points must have shape"),
        ("cells_dtype", r"cells must have dtype"),
        ("cells_topology", r"merged boundary topology and cell ordering"),
        ("cells_order", r"merged boundary topology and cell ordering"),
        ("areas_device", r"areas must be on cpu"),
        ("areas_dtype", r"areas must have dtype"),
        ("centroids_nonfinite", r"centroids must contain only finite"),
        ("areas_nonpositive", r"areas must contain only positive"),
        ("areas_nonfinite_total", r"areas must have a finite positive total"),
        ("normals_nonunit", r"normals must be unit vectors"),
        ("negative_zero_center", r"center must contain raw positive zeros"),
        ("nonneutral_center", r"center must contain raw positive zeros"),
        ("nonneutral_length", r"reference_length must be exactly \+1"),
    ],
)
def test_canonical_source_geometry_rejects_malformed_cpu_input(failure, message):
    model = _model("cpu")
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)

    if failure == "points_shape":
        malformed = replace(prescribed, points=prescribed.points[:-1])
    elif failure == "cells_dtype":
        malformed = replace(prescribed, cells=prescribed.cells.to(torch.int32))
    elif failure == "cells_topology":
        cells = prescribed.cells.clone()
        cells[0] = cells[0].roll(1)
        malformed = replace(prescribed, cells=cells)
    elif failure == "cells_order":
        malformed = replace(prescribed, cells=prescribed.cells.flip(0))
    elif failure == "areas_device":
        malformed = replace(
            prescribed,
            areas=torch.empty_like(prescribed.areas, device="meta"),
        )
    elif failure == "areas_dtype":
        malformed = replace(prescribed, areas=prescribed.areas.float())
    elif failure == "centroids_nonfinite":
        centroids = prescribed.centroids.clone()
        centroids[0, 0] = float("nan")
        malformed = replace(prescribed, centroids=centroids)
    elif failure == "areas_nonpositive":
        areas = prescribed.areas.clone()
        areas[0] = 0.0
        malformed = replace(prescribed, areas=areas)
    elif failure == "areas_nonfinite_total":
        malformed = replace(
            prescribed,
            areas=torch.full_like(
                prescribed.areas, torch.finfo(prescribed.areas.dtype).max
            ),
        )
    elif failure == "normals_nonunit":
        malformed = replace(prescribed, normals=2.0 * prescribed.normals)
    elif failure == "negative_zero_center":
        malformed = replace(prescribed, center=torch.full_like(prescribed.center, -0.0))
    elif failure == "nonneutral_center":
        malformed = replace(prescribed, center=torch.ones_like(prescribed.center))
    else:
        malformed = replace(
            prescribed,
            reference_length=prescribed.reference_length.new_tensor(2.0),
        )

    with pytest.raises(ValueError, match=message):
        model.encode(domain, canonical_source_geometry=malformed)


@pytest.mark.parametrize("query_decoder", ["moment", "kernel"])
def test_canonical_source_geometry_preserves_measure_weights_on_cpu(query_decoder):
    model = _model("cpu", query_decoder=query_decoder)
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)
    boundary = domain.boundaries["wall"]
    cell_data = dict(boundary.cell_data.items())
    weights = torch.linspace(
        0.25,
        2.0,
        boundary.n_cells,
        dtype=boundary.points.dtype,
        device=boundary.points.device,
    )
    cell_data[MEASURE_WEIGHTS_KEY] = weights
    weighted = DomainMesh(
        interior=domain.interior,
        boundaries={"wall": boundary.with_data(cell_data=cell_data)},
        global_data=domain.global_data,
    )

    with torch.no_grad():
        encoded = model.encode(weighted, canonical_source_geometry=prescribed)
        prediction = model.decode(encoded)

    assert encoded.source_mesh.cell_areas.data_ptr() == prescribed.areas.data_ptr()
    torch.testing.assert_close(
        encoded.source_mesh.cell_areas,
        prescribed.areas,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cell_measure_weights(encoded.source_mesh),
        weights,
        rtol=0.0,
        atol=0.0,
    )
    if query_decoder == "kernel":
        torch.testing.assert_close(
            encoded.kernel_cache.panel_areas,
            prescribed.areas,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            encoded.kernel_cache.quadrature_measures,
            prescribed.areas * weights,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            encoded.kernel_cache.representation_measure_factors,
            weights,
            rtol=0.0,
            atol=0.0,
        )
    assert torch.isfinite(prediction.point_data["pressure"]).all()
    assert torch.isfinite(prediction.point_data["velocity"]).all()


def test_canonical_source_geometry_fills_unit_weights_across_boundaries_on_cpu():
    boundary_ranks = {
        "wall_a": _BOUNDARY_RANKS["wall"],
        "wall_b": _BOUNDARY_RANKS["wall"],
    }
    model = _model("cpu", boundary_field_ranks=boundary_ranks)
    original = _domain("cpu")
    boundary = original.boundaries["wall"]
    explicit_weights = torch.linspace(
        0.25,
        2.0,
        boundary.n_cells,
        dtype=boundary.points.dtype,
        device=boundary.points.device,
    )
    weighted_boundary = boundary.with_data(
        cell_data={
            **dict(boundary.cell_data.items()),
            MEASURE_WEIGHTS_KEY: explicit_weights,
        }
    )
    domain = DomainMesh(
        interior=original.interior,
        boundaries={"wall_a": weighted_boundary, "wall_b": boundary},
        global_data=original.global_data,
    )
    prescribed = _prescribed_canonical_geometry(model, domain)

    with torch.no_grad():
        encoded = model.encode(domain, canonical_source_geometry=prescribed)

    expected_weights = torch.cat(
        (explicit_weights, torch.ones_like(explicit_weights)),
    )
    torch.testing.assert_close(
        cell_measure_weights(encoded.source_mesh),
        expected_weights,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cell_measures(encoded.source_mesh),
        prescribed.areas * expected_weights,
        rtol=0.0,
        atol=0.0,
    )


def test_canonical_source_geometry_composes_measure_normalization_on_cpu():
    model = _model("cpu", measure_normalization=True)
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)
    boundary = domain.boundaries["wall"]
    weights = torch.linspace(
        0.25,
        2.0,
        boundary.n_cells,
        dtype=boundary.points.dtype,
        device=boundary.points.device,
    )
    weighted = DomainMesh(
        interior=domain.interior,
        boundaries={
            "wall": boundary.with_data(
                cell_data={
                    **dict(boundary.cell_data.items()),
                    MEASURE_WEIGHTS_KEY: weights,
                }
            )
        },
        global_data=domain.global_data,
    )

    with torch.no_grad():
        encoded = model.encode(weighted, canonical_source_geometry=prescribed)

    normalization = (1.0 / (prescribed.areas * weights).double().sum()).to(
        prescribed.areas.dtype
    )
    torch.testing.assert_close(
        cell_measure_weights(encoded.source_mesh),
        weights * normalization,
        rtol=0.0,
        atol=0.0,
    )
    tolerance = 8.0 * torch.finfo(prescribed.areas.dtype).eps
    torch.testing.assert_close(
        cell_measures(encoded.source_mesh).sum(),
        prescribed.areas.new_tensor(1.0),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_canonical_source_geometry_rejects_invalid_measure_weights_on_cpu(bad_value):
    model = _model("cpu")
    domain = _domain("cpu")
    prescribed = _prescribed_canonical_geometry(model, domain)
    boundary = domain.boundaries["wall"]
    weights = torch.ones_like(boundary.cell_areas)
    weights[0] = bad_value
    weighted = DomainMesh(
        interior=domain.interior,
        boundaries={
            "wall": boundary.with_data(
                cell_data={
                    **dict(boundary.cell_data.items()),
                    MEASURE_WEIGHTS_KEY: weights,
                }
            )
        },
        global_data=domain.global_data,
    )

    with pytest.raises(ValueError, match="effective quadrature measure"):
        model.encode(weighted, canonical_source_geometry=prescribed)
