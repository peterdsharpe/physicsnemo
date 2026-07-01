# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Integration contracts for the boundary-driven mesh transformer."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

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
    dtype: torch.dtype = torch.float64,
    query_chunk_size: int = 2,
) -> MeshTransformer:
    torch.manual_seed(732)
    model = MeshTransformer(
        n_spatial_dims=3,
        output_field_ranks=_OUTPUT_RANKS,
        boundary_field_ranks=_BOUNDARY_RANKS,
        global_field_ranks=_GLOBAL_RANKS,
        reference_length_key="reference.length",
        field_mode=field_mode,
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


def test_none_reference_length_means_already_dimensionless(device):
    model = MeshTransformer(
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
    ).to(device=device, dtype=torch.float64)

    encoded = model.encode(_domain(device))

    torch.testing.assert_close(
        encoded.reference_length,
        torch.ones((), device=device, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


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


def test_forward_mesh_contract_and_target_data_nonleakage(device):
    model = _model(device)
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


def test_linear_mode_has_zero_and_superposition_laws(device):
    model = _model(device, field_mode="linear")
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


def test_nonlinear_mode_is_exactly_zero_preserving(device):
    model = _model(device, field_mode="zero_preserving_nonlinear")
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


def test_amp_preserves_mesh_geometry_precision(device):
    model = _model(device, dtype=torch.float32)
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


@pytest.mark.parametrize("reflection", [False, True])
def test_model_is_o3_equivariant_with_consistent_orientation(device, reflection):
    model = _model(device)
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


def test_similarity_and_source_query_permutation_contracts(device):
    model = _model(device)
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


@pytest.mark.parametrize("field_mode", ["linear", "zero_preserving_nonlinear"])
def test_encode_decode_reuse_and_query_chunk_invariance(device, field_mode):
    model = _model(device, field_mode=field_mode, query_chunk_size=1)
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


def test_mdlus_checkpoint_roundtrip(device, tmp_path):
    model = _model(device, dtype=torch.float32)
    domain = _domain(device, dtype=torch.float32)
    expected = model(domain)
    checkpoint = tmp_path / "mesh_transformer.mdlus"
    model.save(checkpoint)

    restored = MeshTransformer.from_checkpoint(checkpoint).to(device=device)
    restored.eval()
    actual = restored(domain)
    _assert_output_close(actual, expected, rtol=2.0e-6, atol=2.0e-7)
