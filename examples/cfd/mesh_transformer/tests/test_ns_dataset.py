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

"""Catalog round-trip, loader, and arm tests for the ns_cavity_star suite.

Generation runs at deliberately cheap solver settings (coarse mesh, small
boundary polygon) so the whole file stays CI-fast; the production-accuracy
questions live in ``test_fem_navier_stokes.py`` and the generation
manifest's verification block.
"""

import dataset_catalog
import dataset_train_ns
import generate_datasets
import numpy as np
import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh

CHEAP_SETTINGS = generate_datasets.NSGeneratorSettings(
    target_h=0.09,
    n_fem_boundary=192,
    n_query_interior=24,
    n_query_near=8,
    noise_floor_stride=5,
)


@pytest.fixture(scope="module")
def ns_catalog(tmp_path_factory):
    """A tiny generated ns_cavity_star catalog shared by this module."""

    root = tmp_path_factory.mktemp("ns_catalog")
    directory = generate_datasets.generate_dataset(
        family="ns_cavity_star",
        n_cases=7,
        version="v-test",
        workers=1,
        settings=CHEAP_SETTINGS,
        root=root,
        created="test",
    )
    return directory


def test_split_allocation_covers_every_ns_split():
    """From four cases up every ns eval split is nonempty; sizes add up."""

    sizes = generate_datasets.split_sizes(1500, "ns_cavity_star")
    assert sizes["train"] == 1500 - 4 * 150
    assert all(
        sizes[name] == 150
        for name in generate_datasets.NS_SPLIT_ORDER
        if name != "train"
    )
    tiny = generate_datasets.split_sizes(7, "ns_cavity_star")
    assert tiny["train"] == 3
    assert sum(tiny.values()) == 7
    ranges = generate_datasets.split_ranges(7, "ns_cavity_star")
    assert [span["start"] for span in ranges.values()] == [0, 3, 4, 5, 6]


def test_reynolds_band_override_is_split_aware():
    """The slice mechanism moves ONLY the Reynolds band, per split role."""

    settings = generate_datasets.NSGeneratorSettings(
        reynolds_range=(0.5, 5.0),
        unseen_reynolds_range=(5.0, 10.0),
    )
    for split in generate_datasets.NS_SPLIT_ORDER:
        spec = generate_datasets._ns_split_spec(split, settings)
        default = generate_datasets.NS_SPLIT_SPECS[split]
        expected = (5.0, 10.0) if split == "eval_unseen_Re" else (0.5, 5.0)
        assert spec.reynolds_range == expected, split
        assert spec.geometry_modes == default.geometry_modes
        assert spec.deformation_range == default.deformation_range
        assert spec.drive_band == default.drive_band
        assert spec.drive_width_range == default.drive_width_range
    # No override: the production specs come back untouched.
    plain = generate_datasets.NSGeneratorSettings()
    for split in generate_datasets.NS_SPLIT_ORDER:
        assert (
            generate_datasets._ns_split_spec(split, plain)
            is generate_datasets.NS_SPLIT_SPECS[split]
        )
    with pytest.raises(ValueError, match="0 < low < high"):
        generate_datasets._ns_split_spec(
            "train",
            generate_datasets.NSGeneratorSettings(reynolds_range=(5.0, 0.5)),
        )


class _RigidRotationField(torch.nn.Module):
    """u = (y, -x), p = 0: zero Laplacian and divergence, unit-normalized
    momentum residual (the residual IS the advection term)."""

    def forward(self, domain):
        points = domain.interior.points
        velocity = torch.stack((points[:, 1], -points[:, 0]), dim=-1)
        return domain.interior.with_data(
            point_data={
                "velocity": velocity,
                "pressure": points.new_zeros(points.shape[0]),
            },
            cell_data={},
            global_data=domain.global_data,
        )


class _KovasznayField(torch.nn.Module):
    """The exact Kovasznay steady N-S solution at the CASE viscosity."""

    def forward(self, domain):
        import math as _math

        nu = domain.global_data["viscosity"].reshape(())
        lam = 0.5 / nu - torch.sqrt(0.25 / nu**2 + 4.0 * _math.pi**2)
        x, y = domain.interior.points[:, 0], domain.interior.points[:, 1]
        two_pi = 2.0 * _math.pi
        exponential = torch.exp(lam * x)
        velocity = torch.stack(
            (
                1.0 - exponential * torch.cos(two_pi * y),
                (lam / two_pi) * exponential * torch.sin(two_pi * y),
            ),
            dim=-1,
        )
        pressure = 0.5 * (1.0 - torch.exp(2.0 * lam * x))
        return domain.interior.with_data(
            point_data={"velocity": velocity, "pressure": pressure},
            cell_data={},
            global_data=domain.global_data,
        )


def test_fidelity_residuals_certify_the_instrument(ns_catalog):
    """Exact analytic N-S fields score ~0; a dropped pressure scores 1.

    The Kovasznay leg only vanishes when the metric reads nu~ from the
    case (the lambda in the exact solution depends on it), which pins the
    'the model doesn't know the PDE, the metric does' plumbing.
    """

    case = dataset_catalog.load_case(ns_catalog, 0)
    device = torch.device("cpu")
    momentum, divergence = dataset_train_ns._case_fidelity_residuals(
        _KovasznayField().double(), case, device=device
    )
    assert momentum < 1.0e-8
    assert divergence < 1.0e-10
    momentum, divergence = dataset_train_ns._case_fidelity_residuals(
        _RigidRotationField().double(), case, device=device
    )
    assert momentum == pytest.approx(1.0, abs=1.0e-12)
    assert divergence < 1.0e-12


def test_catalog_round_trip_and_manifest(ns_catalog):
    """validate_catalog passes; the manifest carries the N-S verification."""

    summary = dataset_catalog.validate_catalog(ns_catalog)
    assert summary["family"] == "ns_cavity_star"
    assert summary["n_cases"] == 7
    manifest = dataset_catalog.load_manifest(ns_catalog)
    assert set(manifest["splits"]) == set(generate_datasets.NS_SPLIT_ORDER)
    verification = manifest["verification"]
    for split in generate_datasets.NS_SPLIT_ORDER:
        entry = verification[split]
        assert entry["relative_residual_max"] <= 1.0e-10
        assert entry["momentum_balance_error_max"] <= 1.0e-10
        assert entry["boundary_flux_max_abs"] <= 1.0e-3
        assert "self_consistency_rel_l2_velocity_max" in entry
        assert "self_consistency_rel_l2_pressure_max" in entry
    # The stride-5 noise floor touches case indices 0 and 5.
    assert verification["train"]["label_noise_floor_velocity_cases"] == 1
    solver = manifest["solver_settings"]
    assert solver["element"] == "Taylor-Hood P2-P1 triangles"
    assert "Lagrange multiplier" in solver["pressure_gauge"]


def test_case_arrays_follow_ns_schema(ns_catalog):
    """Reloaded cases carry the multi-field schema and sane physics."""

    case = dataset_catalog.load_case(ns_catalog, 0)
    arrays = case.arrays
    n_boundary = arrays["boundary_points"].shape[0]
    assert arrays["boundary_velocity"].shape == (n_boundary, 2)
    n_query = arrays["query_points"].shape[0]
    assert arrays["velocity_query"].shape == (n_query, 2)
    assert arrays["pressure_query"].shape == (n_query,)
    assert arrays["query_wall_distance"].shape == (n_query,)
    # Interior bucket first, near-wall bucket after (the generator layout).
    settings = CHEAP_SETTINGS
    interior = arrays["query_wall_distance"][: settings.n_query_interior]
    near = arrays["query_wall_distance"][settings.n_query_interior :]
    assert interior.min() >= settings.interior_margin
    assert near.max() <= settings.near_band[1]
    assert near.min() >= settings.near_band[0]
    # Peak drive speed is normalized to <= 1 (panel midpoints subsample the
    # dense normalization grid, so the max sits at or below one).
    speeds = np.linalg.norm(arrays["boundary_velocity"], axis=1)
    assert speeds.max() <= 1.0 + 1.0e-12
    assert case.params["reynolds"] > 0
    assert case.params["viscosity"] == pytest.approx(1.0 / case.params["reynolds"])


def test_loader_builds_multifield_domain(ns_catalog):
    """The N-S loader exposes the drive, targets, and operator scalars."""

    case = dataset_catalog.load_case(ns_catalog, 1)
    domain, targets = dataset_catalog.load_ns_domain_sample(case)
    boundary = domain.boundaries["dirichlet"]
    assert boundary.cell_data["boundary_velocity"].shape[1] == 2
    assert set(targets) == {"velocity", "pressure"}
    assert targets["velocity"].shape == domain.interior.points.shape
    assert float(domain.global_data["viscosity"]) == pytest.approx(
        1.0 / float(domain.global_data["reynolds"]), rel=1.0e-5
    )
    assert float(domain.global_data["reference_length"]) == 1.0


def test_every_arm_builds_and_predicts(ns_catalog):
    """All six declared arms construct and emit both output fields."""

    bank = dataset_train_ns.NSCaseBank(ns_catalog, device=torch.device("cpu"))
    domain, targets, wall_distance, reynolds = bank.sample(0)
    for name in dataset_train_ns.MODEL_NAMES:
        model = dataset_train_ns.make_ns_model(name)
        predictions = dataset_train_ns._predictions(model, domain)
        assert predictions["velocity"].shape == targets["velocity"].shape, name
        assert predictions["pressure"].shape == targets["pressure"].shape, name
        assert all(
            bool(torch.isfinite(value).all()) for value in predictions.values()
        ), name


def test_linear_pseudo_confound_arm_is_drive_linear_with_pseudo_sector(
    ns_catalog,
):
    """The iteration-38 confound control has the advertised structure.

    ``mt_singpair_linear_pseudo`` must (1) carry the same pseudo sector as
    the nonlinear arm (``drive_pseudo_dim=8``) while staying
    ``field_mode='linear'``, (2) sit strictly between the plain linear arm
    and nl_pseudo in parameter count -- the residual difference to
    nl_pseudo IS the nonlinear read-in stack, the treatment variable of
    the confound check -- and (3) remain exactly odd in the drive in
    float64 (negating the boundary velocity negates every prediction to
    roundoff): the pseudo sector adds parity machinery and capacity
    without breaking the declared drive-linearity.
    """

    linear = dataset_train_ns.make_ns_model("mt_singpair_linear")
    confound = dataset_train_ns.make_ns_model("mt_singpair_linear_pseudo")
    nonlinear = dataset_train_ns.make_ns_model("mt_singpair_nl_pseudo")
    assert confound.field_mode == "linear"
    assert confound.drive_pseudo_dim == nonlinear.drive_pseudo_dim == 8
    counts = [
        sum(p.numel() for p in model.parameters())
        for model in (linear, confound, nonlinear)
    ]
    assert counts[0] < counts[1] < counts[2]

    bank = dataset_train_ns.NSCaseBank(
        ns_catalog, device=torch.device("cpu"), dtype=torch.float64
    )
    domain, _, _, _ = bank.sample(0)
    boundary = domain.boundaries["dirichlet"]
    negated = DomainMesh(
        interior=Mesh(points=domain.interior.points),
        boundaries={
            "dirichlet": boundary.with_data(
                cell_data={
                    "boundary_velocity": -boundary.cell_data["boundary_velocity"]
                }
            )
        },
        global_data=domain.global_data,
    )
    torch.manual_seed(0)
    model = confound.double().eval()
    with torch.no_grad():
        plus = dataset_train_ns._predictions(model, domain)
        minus = dataset_train_ns._predictions(model, negated)
    for field, prediction in plus.items():
        odd_violation = float((prediction + minus[field]).abs().max())
        scale = float(prediction.abs().max())
        assert odd_violation <= 1.0e-10 * max(scale, 1.0), (field, odd_violation)


def test_boundary_mean_pressure_floor_is_exactly_one(ns_catalog):
    """The floor predicts zero pressure, so its relative L2 is exactly 1."""

    bank = dataset_train_ns.NSCaseBank(ns_catalog, device=torch.device("cpu"))
    model = dataset_train_ns.make_ns_model("boundary_mean")
    aggregate, per_case = dataset_train_ns._evaluate_cases(
        model, bank, [0, 1, 2], near_threshold=CHEAP_SETTINGS.interior_margin
    )
    assert aggregate["pressure"] == pytest.approx(1.0, abs=1.0e-6)
    assert all(record["reynolds"] > 0 for record in per_case)


def test_training_epoch_smoke(ns_catalog, tmp_path):
    """One CPU epoch of the linear arm runs and writes its report."""

    report = dataset_train_ns.run_experiment(
        dataset_dir=ns_catalog,
        model_name="mt_singpair_linear",
        epochs=1,
        seed=0,
        device="cpu",
        output_dir=tmp_path,
    )
    assert np.isfinite(report["best_validation_combined_relative_l2"])
    assert set(report["splits"]) == {
        "eval_id",
        "eval_unseen_Re",
        "eval_unseen_geometry",
        "eval_unseen_drive_profile",
    }
    for record in report["per_case"]["eval_id"]:
        assert {"reynolds", "velocity", "pressure", "combined"} <= set(record)
    fidelity = report["fidelity"]
    for split in report["splits"]:
        assert np.isfinite(fidelity["momentum_residual"][split])
        assert np.isfinite(fidelity["divergence_residual"][split])
    assert "nu~ = 1/Re comes from the CASE" in fidelity["momentum_residual_note"]
    assert (tmp_path / "mt_singpair_linear_seed0.json").is_file()
