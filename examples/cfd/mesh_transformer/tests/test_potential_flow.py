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

"""Contracts for the exterior potential-flow, velocity, and boundary-layer
families, including the pseudoscalar-wall reformulations (velocity output and
trace-driven transformer arms)."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import SimilarityTransform, points_to_complex  # noqa: E402
from encoder_stress import (  # noqa: E402
    mesh_cell_winding,
    polyline_is_simple,
    winding_number,
)
from models import MeshTransformerConfig, build_mesh_transformer  # noqa: E402
from potential_flow import (  # noqa: E402
    _TARGET_KEYS,
    FAMILY_MODEL_NAMES,
    MODEL_NAMES,
    SPLITS,
    ExteriorBody,
    _annulus_preimages,
    _build_model,
    _sample_freestream,
    body_to_physical,
    boundary_layer_term,
    build_boundary_layer_sample,
    build_potential_flow_sample,
    build_potential_flow_velocity_sample,
    disturbance_complex_potential,
    disturbance_complex_velocity,
    disturbance_streamfunction,
    exterior_map,
    invert_body_map,
    run_experiment,
    sample_exterior_body,
    streamfunction_at_physical,
    total_streamfunction,
    velocity_at_physical,
    widened_training_spec,
)

_BUILDERS = {
    "potential_flow": build_potential_flow_sample,
    "potential_flow_velocity": build_potential_flow_velocity_sample,
    "boundary_layer": build_boundary_layer_sample,
}


def _every_split(seeds=(0, 1)):
    for family, builder in _BUILDERS.items():
        for split_name, spec in SPLITS[family].items():
            for seed in seeds:
                yield (
                    family,
                    split_name,
                    builder(seed * 7919 + 11, dtype=torch.float64, **spec),
                )


def _min_distance_to_loop(points: torch.Tensor, loop: torch.Tensor) -> torch.Tensor:
    """Minimum euclidean distance from each point to a closed vertex loop."""

    start = loop
    edge = torch.roll(loop, -1, dims=0) - loop
    edge_sq = edge.square().sum(dim=-1).clamp_min(1.0e-30)
    offset = points[:, None, :] - start[None, :, :]
    t = ((offset * edge[None, :, :]).sum(dim=-1) / edge_sq[None, :]).clamp(0.0, 1.0)
    nearest = start[None, :, :] + t[..., None] * edge[None, :, :]
    return (points[:, None, :] - nearest).norm(dim=-1).min(dim=-1).values


def test_exterior_bodies_are_certified_across_the_wilder_range() -> None:
    """Injectivity certificate, simple panel loops, and bi-Lipschitz margin.

    The corner cases push beyond every split (deformation 0.75 > the 0.70
    ceiling of ``wilder_shapes``; modes beyond the unseen-mode split).
    """

    corner_cases = [
        ((1, 2, 3), (0.68, 0.70)),
        ((1, 2, 3), (0.73, 0.75)),  # beyond the wilder_shapes split
        ((4, 5, 6), (0.30, 0.35)),
        ((2, 5, 8), (0.55, 0.60)),
    ]
    for modes, deformation_range in corner_cases:
        for seed in (0, 1, 2):
            body = sample_exterior_body(
                seed,
                seed + 101,
                modes=modes,
                deformation_range=deformation_range,
            )
            kappa = float(body.deformation_bound)
            assert deformation_range[0] <= kappa <= deformation_range[1] < 1.0
            # Dense and panel-resolution loops are simple.
            for n_points in (192, 2048):
                angles = (
                    2.0
                    * math.pi
                    * torch.arange(n_points, dtype=torch.float64)
                    / n_points
                )
                z = torch.polar(torch.ones_like(angles), angles)
                loop = torch.stack(
                    (body_to_physical(body, z).real, body_to_physical(body, z).imag),
                    dim=-1,
                )
                assert polyline_is_simple(loop), (modes, deformation_range, seed)
            # Quantitative bi-Lipschitz lower bound on random exterior pairs.
            generator = torch.Generator().manual_seed(seed)
            radii = 1.0 + 3.0 * torch.rand(128, generator=generator).double()
            pair_angles = 2.0 * math.pi * torch.rand(128, generator=generator).double()
            z = torch.polar(radii, pair_angles)
            z1, z2 = z[:64], z[64:]
            lhs = (exterior_map(body, z1) - exterior_map(body, z2)).abs()
            rhs = (1.0 - kappa) * (z1 - z2).abs()
            assert bool((lhs >= rhs - 1.0e-12).all())


def test_disturbance_streamfunction_is_harmonic_in_physical_coordinates() -> None:
    """FD Laplacian of the potential part is <= 1e-4 relative, on every split.

    The stencil lives in *physical* coordinates and re-inverts the exterior
    map at each stencil point, so this certifies the pushed-forward field,
    not just the canonical one.  Checked away from the body (|zeta| >= 1.2),
    where the stencil provably stays in the fluid.
    """

    for family, split_name, sample in _every_split(seeds=(0,)):
        scale = float(sample.body.similarity.scale)
        away = sample.query_preimages.abs() >= 1.2
        points = sample.domain.interior.points[away]
        if points.shape[0] == 0:
            continue
        step = 1.0e-3 * scale

        def psi(p: torch.Tensor) -> torch.Tensor:
            return streamfunction_at_physical(
                sample.body, sample.canonical_freestream, sample.circulation, p
            )

        laplacian = -4.0 * psi(points)
        for offset in ([step, 0.0], [-step, 0.0], [0.0, step], [0.0, -step]):
            laplacian = laplacian + psi(
                points + torch.tensor(offset, dtype=torch.float64)
            )
        laplacian = laplacian / step**2
        field_scale = float(sample.target.abs().mean()) + 1.0
        relative = laplacian.abs() * scale**2 / field_scale
        assert float(relative.max()) < 1.0e-4, (family, split_name)


def test_body_is_a_streamline_and_trace_matches_cells() -> None:
    """Impermeability and boundary-data alignment, exactly.

    The total streamfunction vanishes on the body (the exact statement of
    n . grad phi = 0 for potential flow); ``boundary_value`` equals the
    disturbance trace at the stored parameter midpoints, cell-aligned.
    """

    for family, split_name, sample in _every_split(seeds=(0,)):
        on_body = total_streamfunction(
            sample.body,
            sample.canonical_freestream,
            sample.circulation,
            sample.boundary_midpoint_preimages,
        )
        assert float(on_body.abs().max()) < 1.0e-12, (family, split_name)
        trace = disturbance_streamfunction(
            sample.body,
            sample.canonical_freestream,
            sample.circulation,
            sample.boundary_midpoint_preimages,
        )
        if family == "boundary_layer":
            boundary = sample.domain.boundaries["dirichlet"]
            trace = trace + boundary.cell_data["layer_amplitude"]
        boundary = sample.domain.boundaries["dirichlet"]
        torch.testing.assert_close(
            boundary.cell_data["boundary_value"], trace, rtol=0.0, atol=1.0e-12
        )
        length = float(sample.domain.global_data["reference_length"])
        drift = (boundary.cell_centroids - sample.boundary_midpoints).norm(dim=-1)
        assert float(drift.max()) < 0.05 * length, (family, split_name)


def test_circulation_changes_the_target_but_not_the_scalarized_trace() -> None:
    """The vortex term vanishes on the body: trace-driven baselines are blind.

    Same seed, circulation on versus off: identical geometry, freestream,
    queries, and ``boundary_value``; different targets.  This is the
    documented structural gap between the scalarized-drive controls and the
    vector-drive transformer arms.
    """

    with_gamma = build_potential_flow_sample(
        11, dtype=torch.float64, circulation_magnitude_range=(1.0, 1.5)
    )
    without_gamma = build_potential_flow_sample(
        11, dtype=torch.float64, circulation_magnitude_range=(0.0, 0.0)
    )
    assert torch.equal(
        with_gamma.domain.interior.points, without_gamma.domain.interior.points
    )
    torch.testing.assert_close(
        with_gamma.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        without_gamma.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert float((with_gamma.target - without_gamma.target).abs().max()) > 1.0e-3


def test_far_field_disturbance_decays() -> None:
    """Without circulation the disturbance is O(1/r); the log term is Gamma's."""

    sample = build_potential_flow_sample(
        5, dtype=torch.float64, circulation_magnitude_range=(0.0, 0.0)
    )
    generator = torch.Generator().manual_seed(0)
    angles = 2.0 * math.pi * torch.rand(64, generator=generator).double()
    for radius in (10.0, 50.0):
        z = torch.polar(torch.full((64,), radius, dtype=torch.float64), angles)
        psi = disturbance_streamfunction(
            sample.body, sample.canonical_freestream, 0.0, z
        )
        bound = (1.0 + float(sample.body.deformation_bound)) / radius
        assert float(psi.abs().max()) <= bound + 1.0e-12


def test_queries_are_exterior_and_preimages_respect_margins() -> None:
    """Winding-number exteriority for far queries; exact margins for all."""

    for family, split_name, sample in _every_split(seeds=(0,)):
        radii = sample.query_preimages.abs()
        assert float(radii.min()) > 1.0, (family, split_name)
        boundary = sample.domain.boundaries["dirichlet"]
        far = radii >= 1.2
        if bool(far.any()):
            winding = mesh_cell_winding(boundary, sample.domain.interior.points[far])
            assert bool((winding == 0).all()), (family, split_name)
        # The body anchor is enclosed by the CCW loop.
        anchor = sample.body.similarity.translation[None, :]
        assert int(winding_number(sample.boundary_loop, anchor)[0]) == 1


def test_boundary_layer_target_matches_the_closed_form() -> None:
    """Recompute the target from independently Newton-inverted preimages.

    This exercises the full physical-coordinates round trip (push-forward,
    inversion, smooth part, layer term) and must agree to 1e-6; in float64 it
    agrees to ~1e-13.
    """

    for spec in SPLITS["boundary_layer"].values():
        sample = build_boundary_layer_sample(9, dtype=torch.float64, **spec)
        z = invert_body_map(
            sample.body, points_to_complex(sample.domain.interior.points)
        )
        recomputed = disturbance_streamfunction(
            sample.body, sample.canonical_freestream, 0.0, z
        ) + boundary_layer_term(sample.layer_profile, sample.layer_thickness, z)
        torch.testing.assert_close(recomputed, sample.target, rtol=0.0, atol=1.0e-6)


def test_near_and_far_buckets_are_geometrically_separated() -> None:
    """The mask matches the conformal wall coordinate and physical distance.

    Physical wall distance is pinned to ``L (1 +- kappa) (r - 1)`` by the
    bi-Lipschitz certificate, so near-bucket points must sit within
    ``~3 delta (1 + kappa) L`` of the panel loop and far-bucket points beyond
    ``~3 delta (1 - kappa) L`` (up to panel chord sag).
    """

    for split_name, spec in SPLITS["boundary_layer"].items():
        sample = build_boundary_layer_sample(4, dtype=torch.float64, **spec)
        delta = sample.layer_thickness
        radii = sample.query_preimages.abs()
        expected = radii - 1.0 < 3.0 * delta
        assert torch.equal(sample.near_wall_mask, expected), split_name
        n_near = int(sample.near_wall_mask.sum())
        assert n_near == sample.near_wall_mask.numel() // 2, split_name
        kappa = float(sample.body.deformation_bound)
        scale = float(sample.body.similarity.scale)
        distances = _min_distance_to_loop(
            sample.domain.interior.points, sample.boundary_loop
        )
        sag = 2.0 * scale * (math.pi / sample.boundary_loop.shape[0]) ** 2
        near = distances[sample.near_wall_mask]
        far = distances[~sample.near_wall_mask]
        assert float(near.max()) <= scale * (1.0 + kappa) * 3.0 * delta + sag
        assert float(far.min()) >= scale * (1.0 - kappa) * 3.0 * delta - sag


def test_sample_layout_contract() -> None:
    """Boundary key, counts, dtypes, global fields, and drive normalization."""

    a = build_potential_flow_sample(13)
    assert a.domain.boundaries.keys() == {"dirichlet"}
    boundary = a.domain.boundaries["dirichlet"]
    assert boundary.n_cells == 160
    assert boundary.points.dtype == torch.float32
    assert a.domain.interior.n_points == 256
    assert set(a.domain.global_data.keys()) == {
        "reference_length",
        "freestream_velocity",
        "circulation",
    }
    freestream = a.domain.global_data["freestream_velocity"]
    assert freestream.shape == (2,)
    assert abs(float(freestream.norm()) - 1.0) < 1.0e-6
    assert float(a.domain.global_data["reference_length"]) > 0.0
    assert torch.isfinite(a.target).all()
    assert a.near_wall_mask is None and a.layer_profile is None

    b = build_boundary_layer_sample(13)
    assert b.domain.boundaries.keys() == {"dirichlet"}
    boundary = b.domain.boundaries["dirichlet"]
    assert boundary.n_cells == 192
    assert set(boundary.cell_data.keys()) == {"boundary_value", "layer_amplitude"}
    assert set(b.domain.global_data.keys()) == {
        "reference_length",
        "freestream_velocity",
        "layer_thickness",
    }
    assert 0.0 < float(b.domain.global_data["layer_thickness"]) < 1.0
    assert b.circulation == 0.0
    # Unit-RMS band-limited amplitude (exact circle RMS).
    assert abs(b.layer_profile.circle_rms - 1.0) < 1.0e-12
    assert torch.isfinite(b.target).all()


def test_generator_determinism() -> None:
    """The same seed reproduces every tensor of the sample bit-for-bit."""

    for builder in _BUILDERS.values():
        first = builder(321)
        second = builder(321)
        assert torch.equal(first.target, second.target)
        assert torch.equal(first.domain.interior.points, second.domain.interior.points)
        assert torch.equal(
            first.domain.boundaries["dirichlet"].points,
            second.domain.boundaries["dirichlet"].points,
        )
        assert torch.equal(
            first.domain.boundaries["dirichlet"].cell_data["boundary_value"],
            second.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        )
        assert torch.equal(
            first.domain.global_data["freestream_velocity"],
            second.domain.global_data["freestream_velocity"],
        )


def test_model_arms_consume_the_vector_drive() -> None:
    """Every registered arm runs forward on its families and returns finite
    fields; transformer arms differentiate through the rank-1 global drive."""

    torch.manual_seed(11)
    samples = {
        "potential_flow": build_potential_flow_sample(5),
        "potential_flow_velocity": build_potential_flow_velocity_sample(5),
        "boundary_layer": build_boundary_layer_sample(5),
    }
    for family, sample in samples.items():
        for name in FAMILY_MODEL_NAMES[family]:
            model = _build_model(name, family)
            with torch.no_grad():
                prediction = model(sample.domain).point_data[_TARGET_KEYS[family]]
            assert prediction.shape == sample.target.shape, (family, name)
            assert torch.isfinite(prediction).all(), (family, name)
        # One backward pass through the vector-drive path.
        model = _build_model("mesh_transformer_kernel", family)
        prediction = model(sample.domain).point_data[_TARGET_KEYS[family]]
        loss = (prediction - sample.target).square().sum()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_new_arm_construction_forward_backward() -> None:
    """Construction, forward, and backward for every pseudoscalar-wall arm.

    Covers the three rank-1-output velocity arms (Family A') and the two
    trace-driven arms (Family A).  Each must build, produce a finite
    prediction of the target's shape, and backpropagate finite gradients
    into every parameter that receives one.
    """

    samples = {
        "potential_flow": build_potential_flow_sample(5),
        "potential_flow_velocity": build_potential_flow_velocity_sample(5),
    }
    cases = [
        ("potential_flow_velocity", name)
        for name in FAMILY_MODEL_NAMES["potential_flow_velocity"]
    ] + [
        ("potential_flow", "mesh_transformer_kernel_trace"),
        ("potential_flow", "mesh_transformer_kernel_singpair_trace"),
    ]
    for family, name in cases:
        torch.manual_seed(3)
        sample = samples[family]
        model = _build_model(name, family)
        prediction = model(sample.domain).point_data[_TARGET_KEYS[family]]
        assert prediction.shape == sample.target.shape, (family, name)
        assert torch.isfinite(prediction).all(), (family, name)
        loss = (prediction - sample.target).square().sum()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients, (family, name)
        assert all(torch.isfinite(g).all() for g in gradients), (family, name)


def test_new_arm_schemas_and_registry_boundaries() -> None:
    """Schema contracts of the two pseudoscalar-wall reformulations.

    The velocity arms keep the Family A drive schema but declare the rank-1
    output; the trace arms keep the scalar output but swap the drive for the
    certified trace plus circulation, withholding the raw far-field vector.
    Unregistered (model, family) pairs are rejected.
    """

    velocity_model = _build_model("mesh_transformer_kernel", "potential_flow_velocity")
    assert velocity_model.output_field_ranks == {"velocity": 1}
    assert velocity_model.global_field_ranks == {
        "operator": {},
        "drive": {"circulation": 0, "freestream_velocity": 1},
    }
    assert velocity_model.boundary_field_ranks == {
        "dirichlet": {"operator": {}, "drive": {}}
    }

    trace_model = _build_model("mesh_transformer_kernel_trace", "potential_flow")
    assert trace_model.output_field_ranks == {"potential": 0}
    assert trace_model.boundary_field_ranks == {
        "dirichlet": {"operator": {}, "drive": {"boundary_value": 0}}
    }
    assert trace_model.global_field_ranks == {
        "operator": {},
        "drive": {"circulation": 0},
    }

    pseudo_model = _build_model(
        "mesh_transformer_kernel_pseudo", "potential_flow_velocity"
    )
    assert pseudo_model.output_field_ranks == {"velocity": 1}
    assert pseudo_model.boundary_field_ranks == {
        "dirichlet": {"operator": {}, "drive": {}}
    }
    assert pseudo_model.global_field_ranks == {
        "operator": {},
        "drive": {"circulation": "0o", "freestream_velocity": 1},
    }
    assert pseudo_model.drive_pseudo_dim > 0
    singpair_pseudo = _build_model(
        "mesh_transformer_kernel_singpair_pseudo", "potential_flow_velocity"
    )
    assert singpair_pseudo.kernel_decoder.include_single_layer_member is True
    assert singpair_pseudo.kernel_decoder.n_members == 2
    assert singpair_pseudo.drive_pseudo_dim == pseudo_model.drive_pseudo_dim
    # The untyped velocity arm is untouched: circulation stays a true scalar
    # and the pseudo sector stays off.
    assert velocity_model.drive_pseudo_dim == 0

    for name, family in (
        ("boundary_mean", "potential_flow_velocity"),
        ("pair_kernel", "potential_flow_velocity"),
        ("mesh_transformer_kernel_trace", "potential_flow_velocity"),
        ("mesh_transformer_kernel_trace", "boundary_layer"),
        ("mesh_transformer_kernel_pseudo", "potential_flow"),
        ("mesh_transformer_kernel_pseudo", "boundary_layer"),
        ("mesh_transformer_kernel_singpair_pseudo", "potential_flow"),
    ):
        with pytest.raises(ValueError, match="not registered"):
            _build_model(name, family)

    # The CLI surface is the deduplicated union of the family registries.
    assert MODEL_NAMES == (
        "boundary_mean",
        "pair_kernel",
        "mesh_transformer_kernel",
        "mesh_transformer_kernel_singonly",
        "mesh_transformer_kernel_singpair",
        "mesh_transformer_kernel_trace",
        "mesh_transformer_kernel_singpair_trace",
        "mesh_transformer_kernel_pseudo",
        "mesh_transformer_kernel_singpair_pseudo",
        "mesh_transformer_kernel_singpair_pseudo_h1",
    )
    assert set(MODEL_NAMES) == {
        name for names in FAMILY_MODEL_NAMES.values() for name in names
    }


def test_pseudo_arm_circulation_is_live_and_row_stable() -> None:
    """The typed-circulation arm differentiates through Gamma and keeps the
    bitwise query-subset independence contract with pseudo channels on.

    Background: with circulation typed as a true scalar the exact
    circulation velocity Gamma * x_perp / (2 pi |x|^2) is axial and
    unrepresentable, which pinned circulation_ood at 0.647; the "0o"
    declaration gives Gamma an equivariant read-out path, so its input
    gradient must be nonzero.  The row-stability half runs in float64,
    matching the discipline of the in-tree kernel-decoder bitwise tests
    (float32 GEMMs are not row-order stable in the surrounding lifts), and
    asserts to one-ulp tolerance rather than bitwise: at this benchmark's
    boundary size (160 cells) multithreaded CPU reductions repartition with
    the query-batch shape, an effect measured at the same magnitude on the
    pre-existing untyped ``mesh_transformer_kernel`` arm (the strict bitwise
    contract is asserted at single-grain sizes in
    ``test/experimental/nn/mesh_attention/``).
    """

    from physicsnemo.mesh import Mesh

    torch.manual_seed(7)
    sample = build_potential_flow_velocity_sample(5)
    model = _build_model("mesh_transformer_kernel_pseudo", "potential_flow_velocity")

    domain = sample.domain
    domain.global_data["circulation"].requires_grad_()
    prediction = model(domain).point_data["velocity"]
    assert prediction.shape == sample.target.shape
    assert torch.isfinite(prediction).all()
    (prediction - sample.target).square().sum().backward()
    assert domain.global_data["circulation"].grad is not None
    assert torch.count_nonzero(domain.global_data["circulation"].grad)
    pseudo_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "pseudo" in name and parameter.numel()
    ]
    assert pseudo_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in pseudo_gradients
    )

    sample64 = build_potential_flow_velocity_sample(5, dtype=torch.float64)
    model64 = model.double().eval()
    for module in model64.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    queries = sample64.domain.interior.points
    subset = torch.tensor([5, 2, 11])
    with torch.no_grad():
        encoded = model64.encode(sample64.domain)
        message_full = model64.kernel_decoder(queries, encoded.kernel_cache)
        message_subset = model64.kernel_decoder(queries[subset], encoded.kernel_cache)
        full = model64.decode(encoded).point_data["velocity"]
        partial = model64.decode(encoded, Mesh(points=queries[subset]))
    assert encoded.kernel_cache.value_pseudos is not None
    assert encoded.kernel_cache.value_pseudos.shape[-1] > 0
    for sector in ("scalars", "vectors", "pseudos"):
        torch.testing.assert_close(
            getattr(message_subset, sector),
            getattr(message_full, sector)[subset],
            rtol=1.0e-13,
            atol=1.0e-14,
        )
    torch.testing.assert_close(
        partial.point_data["velocity"], full[subset], rtol=1.0e-13, atol=1.0e-14
    )


def test_velocity_family_reuses_the_flow_and_splits() -> None:
    """Family A' shares splits and, seed for seed, the exact same flow.

    Same body, panels, queries, freestream, circulation, and certified trace
    as Family A; only the target changes -- to the ``(n_query, 2)``
    polar-vector disturbance velocity.
    """

    assert SPLITS["potential_flow_velocity"] == SPLITS["potential_flow"]
    a = build_potential_flow_sample(21, dtype=torch.float64)
    b = build_potential_flow_velocity_sample(21, dtype=torch.float64)
    assert b.family == "potential_flow_velocity"
    assert torch.equal(a.domain.interior.points, b.domain.interior.points)
    assert torch.equal(
        a.domain.boundaries["dirichlet"].points,
        b.domain.boundaries["dirichlet"].points,
    )
    assert torch.equal(
        a.domain.boundaries["dirichlet"].cell_data["boundary_value"],
        b.domain.boundaries["dirichlet"].cell_data["boundary_value"],
    )
    assert torch.equal(
        a.domain.global_data["freestream_velocity"],
        b.domain.global_data["freestream_velocity"],
    )
    assert a.circulation == b.circulation
    assert set(b.domain.global_data.keys()) == {
        "reference_length",
        "freestream_velocity",
        "circulation",
    }
    assert b.target.shape == (256, 2)
    assert torch.isfinite(b.target).all()


def test_velocity_target_matches_fd_gradients_of_the_streamfunction() -> None:
    """The dW/dz push-forward is exact: FD certification at <= 1e-5 relative.

    Central differences of the (independently Newton-inverted) disturbance
    streamfunction in *physical* coordinates give the velocity through
    ``u = L dpsi*/dy, v = -L dpsi*/dx``; the closed-form target must agree to
    relative L2 <= 1e-5 on every split, including with circulation (the
    velocity is single-valued; only the potential is multivalued).
    """

    for split_name, spec in SPLITS["potential_flow_velocity"].items():
        sample = build_potential_flow_velocity_sample(7, dtype=torch.float64, **spec)
        away = sample.query_preimages.abs() >= 1.2  # FD stencil stays exterior
        points = sample.domain.interior.points[away]
        assert points.shape[0] > 0, split_name
        scale = float(sample.body.similarity.scale)
        step = 1.0e-4 * scale

        def psi(p: torch.Tensor) -> torch.Tensor:
            return streamfunction_at_physical(
                sample.body, sample.canonical_freestream, sample.circulation, p
            )

        dx = torch.tensor([step, 0.0], dtype=torch.float64)
        dy = torch.tensor([0.0, step], dtype=torch.float64)
        fd = torch.stack(
            (
                scale * (psi(points + dy) - psi(points - dy)) / (2.0 * step),
                -scale * (psi(points + dx) - psi(points - dx)) / (2.0 * step),
            ),
            dim=-1,
        )
        exact = sample.target[away]
        relative = float(
            torch.linalg.vector_norm(fd - exact)
            / torch.linalg.vector_norm(exact).clamp_min(1.0e-30)
        )
        assert relative <= 1.0e-5, (split_name, relative)


def test_velocity_target_matches_fd_gradients_of_the_potential() -> None:
    """Cross-check against the velocity *potential* at zero circulation.

    ``u = L dphi*/dx, v = L dphi*/dy`` from ``Re`` of the disturbance
    complex potential; restricted to circulation-free flows, where the
    potential is single valued (no branch cut crosses the stencil).
    """

    sample = build_potential_flow_velocity_sample(
        3, dtype=torch.float64, circulation_magnitude_range=(0.0, 0.0)
    )
    away = sample.query_preimages.abs() >= 1.2
    points = sample.domain.interior.points[away]
    scale = float(sample.body.similarity.scale)
    step = 1.0e-4 * scale

    def phi(p: torch.Tensor) -> torch.Tensor:
        z = invert_body_map(sample.body, points_to_complex(p))
        return disturbance_complex_potential(
            sample.body, sample.canonical_freestream, 0.0, z
        ).real

    dx = torch.tensor([step, 0.0], dtype=torch.float64)
    dy = torch.tensor([0.0, step], dtype=torch.float64)
    fd = torch.stack(
        (
            scale * (phi(points + dx) - phi(points - dx)) / (2.0 * step),
            scale * (phi(points + dy) - phi(points - dy)) / (2.0 * step),
        ),
        dim=-1,
    )
    exact = sample.target[away]
    relative = float(
        torch.linalg.vector_norm(fd - exact)
        / torch.linalg.vector_norm(exact).clamp_min(1.0e-30)
    )
    assert relative <= 1.0e-5, relative


def test_velocity_target_matches_independent_inversion() -> None:
    """Recompute the target from independently Newton-inverted preimages."""

    for spec in SPLITS["potential_flow_velocity"].values():
        sample = build_potential_flow_velocity_sample(9, dtype=torch.float64, **spec)
        recomputed = velocity_at_physical(
            sample.body,
            sample.canonical_freestream,
            sample.circulation,
            sample.domain.interior.points,
        )
        torch.testing.assert_close(recomputed, sample.target, rtol=0.0, atol=1.0e-10)


def test_mirror_flips_the_streamfunction_and_rotates_the_velocity() -> None:
    """The pseudoscalar wall, certified on the exact fields.

    Mirror the whole problem across the x-axis (conjugated map coefficients,
    inverse rotation, conjugated translation and far field, negated
    circulation).  The disturbance streamfunction at mirrored points flips
    sign -- it is a 2D pseudoscalar, so an O(2)-equivariant scalar built
    from dot products of polar vectors (mirror-even by construction) can
    only fit it with the zero function; that is the one-line proof of the
    trained relative-L2 ~ 1 no-response.  The disturbance velocity instead
    transforms as a polar vector (its mirror image is the mirrored vector
    field), which is exactly why Family A' is representable.
    """

    body = sample_exterior_body(3, 104, modes=(1, 2, 3), deformation_range=(0.05, 0.35))
    _, canonical, circulation = _sample_freestream(11, body, (0.5, 1.5))
    mirrored = ExteriorBody(
        modes=body.modes,
        coefficients=body.coefficients.conj().resolve_conj(),
        similarity=SimilarityTransform(
            scale=body.similarity.scale,
            rotation=body.similarity.rotation.T.contiguous(),
            translation=body.similarity.translation
            * torch.tensor([1.0, -1.0], dtype=torch.float64),
        ),
    )
    z = _annulus_preimages(5, 64, (1.05, 4.0))
    # The mirrored problem's physical points are the mirrored physical points.
    torch.testing.assert_close(
        body_to_physical(mirrored, z.conj()),
        body_to_physical(body, z).conj(),
        rtol=0.0,
        atol=1.0e-13,
    )
    psi = disturbance_streamfunction(body, canonical, circulation, z)
    psi_mirrored = disturbance_streamfunction(
        mirrored, canonical.conj(), -circulation, z.conj()
    )
    torch.testing.assert_close(psi_mirrored, -psi, rtol=0.0, atol=1.0e-13)
    velocity = disturbance_complex_velocity(body, canonical, circulation, z)
    velocity_mirrored = disturbance_complex_velocity(
        mirrored, canonical.conj(), -circulation, z.conj()
    )
    torch.testing.assert_close(
        velocity_mirrored, velocity.conj(), rtol=0.0, atol=1.0e-13
    )


def test_scalar_only_arms_reject_the_rank_1_drive() -> None:
    """The structural claim of the benchmark: scalar-only ablations cannot
    consume a global rank-1 drive, so the vector channel is load-bearing."""

    scalar_only = replace(
        MeshTransformerConfig(),
        operator_vector_dim=0,
        drive_vector_dim=0,
        vector_rank=0,
    )
    for family in ("potential_flow", "boundary_layer"):
        from potential_flow import _FAMILY_SCHEMAS

        with pytest.raises(ValueError, match="rank-1"):
            build_mesh_transformer(
                scalar_only, query_decoder="kernel", **_FAMILY_SCHEMAS[family]
            )


def test_default_build_mesh_transformer_schema_is_unchanged() -> None:
    """The models.py passthrough keeps the historical defaults bit-for-bit."""

    model = build_mesh_transformer(MeshTransformerConfig())
    assert model.boundary_field_ranks == {
        "dirichlet": {"operator": {}, "drive": {"boundary_value": 0}}
    }
    assert model.global_field_ranks == {"operator": {}, "drive": {}}


@pytest.mark.parametrize("family", ("potential_flow", "boundary_layer"))
def test_driver_smoke_produces_finite_json(tmp_path: Path, family: str) -> None:
    """A zero-step CPU run writes a finite report with the benchmark shape."""

    report = run_experiment(
        model_name="boundary_mean",
        family=family,
        steps=0,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    on_disk = json.loads((tmp_path / f"{family}_boundary_mean_seed3.json").read_text())
    expected_keys = set(SPLITS[family])
    if family == "boundary_layer":
        for split in SPLITS[family]:
            expected_keys |= {f"{split}/near_wall", f"{split}/far_field"}
    for payload in (report, on_disk):
        assert payload["parameters"] == 0
        assert payload["family"] == family
        assert set(payload["splits"]) == expected_keys
        for value in payload["splits"].values():
            assert math.isfinite(value)
        assert math.isfinite(payload["best_validation_relative_l2"])
        if family == "potential_flow":
            # BoundaryMean is constant, hence exactly harmonic.
            assert abs(payload["pde_residual"]) < 1.0e-9
        else:
            assert payload["pde_residual"] is None
        # Operator-fidelity block: per-split residual for the harmonic
        # family; None (with a note) for the deliberately non-harmonic
        # boundary layer.  No maximum principle is licensed on either.
        fidelity = payload["fidelity"]
        if family == "potential_flow":
            assert set(fidelity["pde_residual"]) == set(SPLITS[family])
            assert all(
                abs(v) < 1.0e-9 for v in fidelity["pde_residual"].values()
            )
        else:
            assert fidelity["pde_residual"] is None
        assert fidelity["max_principle_violation"] is None


def test_widened_training_spec_touches_the_training_annulus_only() -> None:
    """The M1 knob widens training queries and nothing else.

    ``widened_training_spec`` must change exactly the radial query knob of
    the in-distribution TRAINING spec (``query_radius_range`` outer radius
    for the exterior-flow families, ``outer_query_radius`` for the boundary
    layer), leave every other spec entry bitwise identical, never mutate
    :data:`SPLITS` (evaluation stays on the frozen banks), and reject radii
    that do not exceed the inner margin.
    """

    for family in ("potential_flow", "potential_flow_velocity"):
        frozen = {name: dict(spec) for name, spec in SPLITS[family].items()}
        widened = widened_training_spec(family, 8.0)
        assert widened["query_radius_range"] == (1.05, 8.0)
        baseline = dict(SPLITS[family]["in_distribution"])
        for key, value in widened.items():
            if key != "query_radius_range":
                assert baseline[key] == value
        assert set(widened) == set(baseline)
        assert {name: dict(spec) for name, spec in SPLITS[family].items()} == frozen
        with pytest.raises(ValueError, match="inner query radius"):
            widened_training_spec(family, 1.05)
    widened_layer = widened_training_spec("boundary_layer", 6.0)
    assert widened_layer["outer_query_radius"] == 6.0
    assert "query_radius_range" not in widened_layer
    with pytest.raises(ValueError, match="unknown family"):
        widened_training_spec("no_such_family", 8.0)


def test_driver_train_query_outer_and_checkpoint_knobs(tmp_path: Path) -> None:
    """The additive driver knobs thread through and default to off.

    A one-step ``pair_kernel`` run with ``train_query_outer`` records the
    widened training spec in the report and writes a loadable checkpoint of
    the reported state; the default path records ``None``, writes no
    checkpoint, and trains on the unmodified in-distribution spec.
    """

    report = run_experiment(
        model_name="pair_kernel",
        family="potential_flow",
        steps=1,
        seed=5,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
        train_query_outer=6.0,
        save_checkpoint=True,
    )
    assert report["train_query_outer"] == 6.0
    assert report["train_spec"]["query_radius_range"] == [1.05, 6.0]
    assert report["checkpoint"] == "potential_flow_pair_kernel_seed5.pt"
    checkpoint = torch.load(
        tmp_path / report["checkpoint"], weights_only=True, map_location="cpu"
    )
    assert checkpoint["model"] == "pair_kernel"
    assert checkpoint["train_query_outer"] == 6.0
    reloaded = _build_model("pair_kernel", "potential_flow")
    reloaded.load_state_dict(checkpoint["state_dict"])

    default = run_experiment(
        model_name="boundary_mean",
        family="potential_flow",
        steps=0,
        seed=5,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    assert default["train_query_outer"] is None
    assert default["checkpoint"] is None
    assert default["train_spec"]["query_radius_range"] == [1.05, 4.0]
    assert not (tmp_path / "potential_flow_boundary_mean_seed5.pt").exists()


def test_bounded_gates_knob_threads_through_the_arm_registry() -> None:
    """The far-field gate-fix knob reaches the output projection, default off.

    ``bounded_gates=True`` must flip ``bounded_gate_invariants`` on the
    transformer arm's output projection while adding no parameters (the
    state-dict keys are unchanged, so gate-fixed checkpoints load into
    either parameterization and the flag alone selects the architecture);
    the default must stay off on every registered arm, and the
    radius-blind baselines must reject the flag rather than silently
    ignore it.
    """

    family = "potential_flow_velocity"
    arm = "mesh_transformer_kernel_singpair_pseudo"
    torch.manual_seed(3)
    default = _build_model(arm, family)
    torch.manual_seed(3)
    fixed = _build_model(arm, family, bounded_gates=True)
    assert default.output_projection.bounded_gate_invariants is False
    assert fixed.output_projection.bounded_gate_invariants is True
    assert list(default.state_dict()) == list(fixed.state_dict())
    fixed.load_state_dict(default.state_dict())

    # The scalar-family path (build_mesh_transformer passthrough) too.
    scalar_fixed = _build_model(
        "mesh_transformer_kernel_singpair", "potential_flow", bounded_gates=True
    )
    assert scalar_fixed.output_projection.bounded_gate_invariants is True

    for baseline in ("boundary_mean", "pair_kernel"):
        with pytest.raises(ValueError, match="transformer arms only"):
            _build_model(baseline, "potential_flow", bounded_gates=True)


def test_bounded_query_knob_threads_through_the_arm_registry() -> None:
    """The source-side far-field fix knob reaches the model, default off.

    ``bounded_query=True`` must flip ``bounded_query_geometry`` on the
    transformer arm (the compactified query-position injection -- the
    completion of the gate fix, whose falsifier showed the gate collapse had
    been suppressing polynomially growing direct-drive branches) while
    adding no parameters, so checkpoints load into either parameterization
    and the flag alone selects the architecture.  The two knobs compose:
    they bound at different places (query injection vs gate inputs).  The
    radius-blind baselines must reject the flag rather than silently ignore
    it.
    """

    family = "potential_flow_velocity"
    arm = "mesh_transformer_kernel_singpair_pseudo"
    torch.manual_seed(3)
    default = _build_model(arm, family)
    torch.manual_seed(3)
    fixed = _build_model(arm, family, bounded_query=True)
    torch.manual_seed(3)
    composed = _build_model(arm, family, bounded_gates=True, bounded_query=True)
    assert default.bounded_query_geometry is False
    assert fixed.bounded_query_geometry is True
    assert fixed.output_projection.bounded_gate_invariants is False
    assert composed.bounded_query_geometry is True
    assert composed.output_projection.bounded_gate_invariants is True
    assert list(default.state_dict()) == list(composed.state_dict())
    composed.load_state_dict(default.state_dict())

    # The scalar-family path (build_mesh_transformer passthrough) too.
    scalar_fixed = _build_model(
        "mesh_transformer_kernel_singpair", "potential_flow", bounded_query=True
    )
    assert scalar_fixed.bounded_query_geometry is True

    for baseline in ("boundary_mean", "pair_kernel"):
        with pytest.raises(ValueError, match="transformer arms only"):
            _build_model(baseline, "potential_flow", bounded_query=True)


def test_decay_structure_knobs_thread_through_the_arm_registry() -> None:
    """Iteration 30's decay-structure knobs reach the model, default off.

    ``decaying_drive=True`` must flip ``decaying_direct_drive`` on the
    transformer arm (the analytic 1/(1+|x|^2) direct-drive envelope: bounded
    is not decaying -- iteration 29 measured the direct drive converging to
    a direction-dependent constant while the exact exterior velocity decays
    like r^-2) and ``monopole_free_sl=True`` must flip
    ``monopole_free_single_layer`` on the kernel decoder (zero-net-charge
    single-layer deflation, structurally killing the log-r tail that was
    being fit by near-cancellation).  Neither adds parameters, so
    checkpoints load into any parameterization and the flags alone select
    the architecture; all four far-field knobs compose.  ``monopole_free_sl``
    is rejected on arms lacking the single-layer member, and the
    radius-blind baselines reject both flags.
    """

    family = "potential_flow_velocity"
    arm = "mesh_transformer_kernel_singpair_pseudo"
    torch.manual_seed(3)
    default = _build_model(arm, family)
    torch.manual_seed(3)
    decaying = _build_model(arm, family, decaying_drive=True)
    torch.manual_seed(3)
    monopole_free = _build_model(arm, family, monopole_free_sl=True)
    torch.manual_seed(3)
    composed = _build_model(
        arm,
        family,
        bounded_gates=True,
        bounded_query=True,
        decaying_drive=True,
        monopole_free_sl=True,
    )
    assert default.decaying_direct_drive is False
    assert default.kernel_decoder.monopole_free_single_layer is False
    assert decaying.decaying_direct_drive is True
    assert decaying.kernel_decoder.monopole_free_single_layer is False
    assert monopole_free.decaying_direct_drive is False
    assert monopole_free.kernel_decoder.monopole_free_single_layer is True
    assert composed.decaying_direct_drive is True
    assert composed.kernel_decoder.monopole_free_single_layer is True
    assert composed.bounded_query_geometry is True
    assert composed.output_projection.bounded_gate_invariants is True
    assert list(default.state_dict()) == list(composed.state_dict())
    composed.load_state_dict(default.state_dict())

    # The scalar-family path (build_mesh_transformer passthrough) too.
    scalar_fixed = _build_model(
        "mesh_transformer_kernel_singpair",
        "potential_flow",
        decaying_drive=True,
        monopole_free_sl=True,
    )
    assert scalar_fixed.decaying_direct_drive is True
    assert scalar_fixed.kernel_decoder.monopole_free_single_layer is True

    # No single-layer member -> no monopole to control.
    with pytest.raises(ValueError, match="monopole_free_single_layer"):
        _build_model("mesh_transformer_kernel", family, monopole_free_sl=True)

    for baseline in ("boundary_mean", "pair_kernel"):
        for flag in ("decaying_drive", "monopole_free_sl"):
            with pytest.raises(ValueError, match="transformer arms only"):
                _build_model(baseline, "potential_flow", **{flag: True})


def test_velocity_family_pde_residual_is_finite() -> None:
    """The strong-form residual extends to the rank-1 velocity family.

    The disturbance velocity is harmonic componentwise (dW/dz is
    holomorphic), so the driver's fidelity block licenses the same
    ``||lap u|| L^2 / ||u||`` convention there; at random initialization it
    must simply be finite (harmonicity is diagnosed, not enforced).
    """

    from potential_flow import pde_residual

    torch.manual_seed(0)
    model = _build_model(
        "mesh_transformer_kernel_singonly", "potential_flow_velocity"
    )
    value = pde_residual(
        model,
        family="potential_flow_velocity",
        seed=83_000_019,
        device=torch.device("cpu"),
    )
    assert math.isfinite(value)
