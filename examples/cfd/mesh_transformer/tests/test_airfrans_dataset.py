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

"""Loader, arm-wiring, and loss contracts for the AirFRANS adapter.

CI-safe by construction: no AirFRANS download and no PyVista -- a tiny
synthetic case npz (16-segment ellipse boundary, 40 query points including
NaN-masked and on-surface rows, a plausible 5-degree-AoA freestream) is
written directly in the loader's schema.  The arm tests import the trainer's
own :func:`airfrans_train.build_arm`, so the wiring exercised here is the
wiring the campaign trains.  The zero-drive contract is deliberately NOT
tested: the freestream drive is never zero on this benchmark.
"""

from __future__ import annotations

import json
import math

import airfrans_dataset
import airfrans_train
import numpy as np
import pytest
import torch
from torch.nn import functional as F  # noqa: N812 - torch convention

CASE_NAME = "airFoil2D_synth_0"
N_BOUNDARY = 16
N_QUERY = 40
N_SURFACE = 6
MASKED_ROWS = (5, 17)  # row 5 is also on-surface
U_INF = (44.63, 3.905)  # |U| ~ 44.8 m/s at ~5 degrees AoA
NU = 1.56e-5


def _ellipse(t: np.ndarray) -> np.ndarray:
    """A unit-chord thin-ellipse 'airfoil' in the z = 0 plane."""

    return np.stack((0.5 + 0.5 * np.cos(t), 0.08 * np.sin(t)), axis=-1)


def _write_synthetic_case(directory) -> None:
    rng = np.random.default_rng(20260705)
    t_boundary = 2.0 * np.pi * np.arange(N_BOUNDARY) / N_BOUNDARY
    boundary_points = _ellipse(t_boundary)  # CCW for increasing t
    index = np.arange(N_BOUNDARY, dtype=np.int64)
    boundary_cells = np.stack((index, np.roll(index, -1)), axis=-1)

    # First N_SURFACE queries sit on the ellipse; the rest fill an annulus.
    t_surface = 2.0 * np.pi * rng.random(N_SURFACE)
    radii = 0.7 + 2.3 * rng.random(N_QUERY - N_SURFACE)
    angles = 2.0 * np.pi * rng.random(N_QUERY - N_SURFACE)
    volume = np.stack((0.5 + radii * np.cos(angles), radii * np.sin(angles)), axis=-1)
    query_points = np.concatenate((_ellipse(t_surface), volume))

    delta_velocity = 0.2 * rng.standard_normal((N_QUERY, 2))
    pressure_coefficient = 0.4 * rng.standard_normal(N_QUERY)
    log_nut_ratio = np.abs(2.0 * rng.standard_normal(N_QUERY))
    cpt = pressure_coefficient + rng.random(N_QUERY)
    masked = np.zeros(N_QUERY, dtype=bool)
    masked[list(MASKED_ROWS)] = True
    delta_velocity[masked] = np.nan
    pressure_coefficient[masked] = np.nan
    log_nut_ratio[masked] = np.nan
    cpt[masked] = np.nan
    is_surface = np.zeros(N_QUERY, dtype=bool)
    is_surface[:N_SURFACE] = True

    np.savez(
        directory / f"{CASE_NAME}.npz",
        boundary_points=boundary_points.astype(np.float64),
        boundary_cells=boundary_cells,
        query_points=query_points.astype(np.float64),
        delta_velocity=delta_velocity.astype(np.float32),
        pressure_coefficient=pressure_coefficient.astype(np.float32),
        log_nut_ratio=log_nut_ratio.astype(np.float32),
        cpt=cpt.astype(np.float32),
        is_surface=is_surface,
        u_inf=np.asarray(U_INF, dtype=np.float64),
        nu=np.float64(NU),
        chord=np.float64(1.0),
    )


BAND_CASE = "airFoil2D_synth_bands"
#: One query point per wall-distance band: (1 + d, 0) sits at distance
#: exactly d from the convex boundary polygon's vertex (1, 0).
BAND_DISTANCES = (5.0e-5, 5.0e-4, 5.0e-3, 5.0e-2, 0.5, 2.0)


def _write_band_case(directory) -> None:
    """A case whose six query points sit at one known distance per band."""

    rng = np.random.default_rng(11)
    t_boundary = 2.0 * np.pi * np.arange(N_BOUNDARY) / N_BOUNDARY
    boundary_points = _ellipse(t_boundary)  # vertex 0 is exactly (1, 0)
    index = np.arange(N_BOUNDARY, dtype=np.int64)
    boundary_cells = np.stack((index, np.roll(index, -1)), axis=-1)
    n = len(BAND_DISTANCES)
    query_points = np.stack((1.0 + np.asarray(BAND_DISTANCES), np.zeros(n)), axis=-1)
    np.savez(
        directory / f"{BAND_CASE}.npz",
        boundary_points=boundary_points.astype(np.float64),
        boundary_cells=boundary_cells,
        query_points=query_points.astype(np.float64),
        delta_velocity=(0.2 * rng.standard_normal((n, 2))).astype(np.float32),
        pressure_coefficient=(0.4 * rng.standard_normal(n)).astype(np.float32),
        log_nut_ratio=np.abs(rng.standard_normal(n)).astype(np.float32),
        cpt=(0.4 * rng.standard_normal(n)).astype(np.float32),
        is_surface=np.zeros(n, dtype=bool),
        u_inf=np.asarray(U_INF, dtype=np.float64),
        nu=np.float64(NU),
        chord=np.float64(1.0),
    )


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """A one-case synthetic catalog in the preprocessor's on-disk layout."""

    directory = tmp_path_factory.mktemp("airfrans_catalog")
    _write_synthetic_case(directory)
    _write_band_case(directory)
    manifest = {
        "full_train": [CASE_NAME],
        "full_test": [CASE_NAME],
        "scarce_train": [CASE_NAME],
        "reynolds_train": [CASE_NAME],
        "reynolds_test": [CASE_NAME],
        "aoa_train": [CASE_NAME],
        "aoa_test": [CASE_NAME],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def case(catalog):
    return airfrans_dataset.load_case(catalog, CASE_NAME)


def test_loader_round_trips_shapes_and_dtypes(case):
    assert case.boundary_points.shape == (N_BOUNDARY, 2)
    assert case.boundary_points.dtype == torch.float32
    assert case.boundary_cells.shape == (N_BOUNDARY, 2)
    assert case.boundary_cells.dtype == torch.int64
    assert case.query_points.shape == (N_QUERY, 2)
    assert set(case.targets) == set(airfrans_dataset.TARGET_FIELDS)
    assert case.targets["delta_velocity"].shape == (N_QUERY, 2)
    assert case.targets["pressure_coefficient"].shape == (N_QUERY,)
    assert case.targets["log_nut_ratio"].shape == (N_QUERY,)
    assert all(value.dtype == torch.float32 for value in case.targets.values())
    assert case.is_surface.dtype == torch.bool
    assert int(case.is_surface.sum()) == N_SURFACE
    assert case.u_inf.dtype == torch.float64
    assert case.nu == pytest.approx(NU)
    assert case.chord == pytest.approx(1.0)
    # The pathology masks arrive as whole-NaN rows, preserved by the loader.
    for value in case.targets.values():
        finite = torch.isfinite(value)
        row_valid = finite.all(dim=-1) if value.ndim > 1 else finite
        assert int((~row_valid).sum()) == len(MASKED_ROWS)
        assert not bool(row_valid[list(MASKED_ROWS)].any())


def test_split_resolution_follows_globe(catalog):
    manifest = airfrans_dataset.load_manifest(catalog)
    assert airfrans_dataset.split_case_names(manifest, "scarce", "train") == [CASE_NAME]
    # GLOBE's convention: scarce has no test list; it resolves to full_test.
    assert (
        airfrans_dataset.split_case_names(manifest, "scarce", "test")
        == manifest["full_test"]
    )
    assert airfrans_dataset.split_case_names(manifest, "reynolds", "test") == [
        CASE_NAME
    ]
    with pytest.raises(ValueError, match="unknown task"):
        airfrans_dataset.split_case_names(manifest, "hypersonic", "train")
    with pytest.raises(ValueError, match="train.*or.*test"):
        airfrans_dataset.split_case_names(manifest, "full", "validation")


def test_domain_mesh_construction(case):
    domain, indices = airfrans_dataset.case_domain(case)
    assert indices.shape == (N_QUERY,)
    assert set(domain.boundaries.keys()) == {"airfoil"}
    assert domain.boundaries["airfoil"].n_cells == N_BOUNDARY
    assert domain.interior.n_points == N_QUERY
    assert set(domain.interior.point_data.keys()) == set(airfrans_dataset.TARGET_FIELDS)
    direction = domain.global_data["freestream_direction"]
    assert direction.shape == (2,)
    assert float(torch.linalg.vector_norm(direction)) == pytest.approx(1.0, rel=1e-6)
    magnitude = math.hypot(*U_INF)
    assert float(domain.global_data["log_reynolds"]) == pytest.approx(
        math.log(magnitude * 1.0 / NU), rel=1e-6
    )
    # The declared auxiliary boundary-layer scale: Re^(-1/2) from the same
    # Reynolds number the loader already computes.
    assert float(domain.global_data["viscous_scale"]) == pytest.approx(
        (magnitude * 1.0 / NU) ** -0.5, rel=1e-6
    )


def test_query_subsampling_is_seeded_and_full_for_eval(case):
    first = airfrans_dataset.case_domain(
        case, n_queries=16, generator=torch.Generator().manual_seed(7)
    )
    second = airfrans_dataset.case_domain(
        case, n_queries=16, generator=torch.Generator().manual_seed(7)
    )
    assert first[1].shape == (16,)
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[0].interior.points, case.query_points[first[1]])
    # Consecutive draws from one generator differ (fresh subsample per step).
    generator = torch.Generator().manual_seed(7)
    draw_a = airfrans_dataset.case_domain(case, n_queries=16, generator=generator)
    draw_b = airfrans_dataset.case_domain(case, n_queries=16, generator=generator)
    assert not torch.equal(draw_a[1], draw_b[1])
    # n_queries >= n_query (or None) means the full set, in stored order.
    full = airfrans_dataset.case_domain(case, n_queries=10_000)
    assert torch.equal(full[1], torch.arange(N_QUERY))


@pytest.mark.parametrize("arm", airfrans_train.ARM_NAMES)
def test_each_arm_trains_one_step_on_the_synthetic_case(case, arm):
    """Forward + backward through the trainer's own arm wiring.

    ``build_arm`` is the trainer's constructor, so this covers the exact
    models the campaign trains: reference singpair configuration, the
    pseudo sector for ``mt_nl_pseudo``, and the far-field decay recipe for
    ``mt_nl_decay`` (monopole deflation off).
    """

    torch.manual_seed(0)
    model = airfrans_train.build_arm(arm)
    assert not getattr(model.kernel_decoder, "monopole_free_single_layer", False)
    domain, _ = airfrans_dataset.case_domain(case)
    targets = {key: domain.interior.point_data[key] for key in case.targets}
    predictions = airfrans_train._predictions(model, domain)
    for name, prediction in predictions.items():
        assert prediction.shape == targets[name].shape, (arm, name)
        assert bool(torch.isfinite(prediction).all()), (arm, name)
    loss = airfrans_train.masked_huber_loss(predictions, targets)
    assert bool(torch.isfinite(loss)), arm
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        assert parameter.grad is not None, (arm, name)
        assert bool(torch.isfinite(parameter.grad).all()), (arm, name)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(bool(g.abs().sum() > 0) for g in grads), arm


def test_masked_huber_matches_manual_row_exclusion(case):
    torch.manual_seed(1)
    targets = case.targets
    predictions = {
        key: torch.randn_like(value).requires_grad_(True)
        for key, value in targets.items()
    }
    loss = airfrans_train.masked_huber_loss(predictions, targets)
    assert bool(torch.isfinite(loss))

    expected = 0.0
    for key, target in targets.items():
        finite = torch.isfinite(target)
        valid = finite.all(dim=-1) if target.ndim > 1 else finite
        expected = expected + F.huber_loss(
            predictions[key][valid], target[valid], delta=1.0
        )
    torch.testing.assert_close(loss, expected)

    loss.backward()
    for key, prediction in predictions.items():
        assert prediction.grad is not None
        assert bool(torch.isfinite(prediction.grad).all()), key
        # Masked rows contribute nothing: their gradients are exactly zero.
        grad = prediction.grad[list(MASKED_ROWS)]
        assert float(grad.abs().sum()) == 0.0, key


def test_pseudo_sector_is_live_at_nonzero_aoa(case):
    """The open design point of the pre-registration, pinned empirically.

    ``mt_nl_pseudo`` declares no ``"0o"`` fields; ``drive_pseudo_dim=8``
    alone must (a) construct and (b) produce nonzero pseudo-sector
    activations from the internally generated wedge products when the
    freestream is at nonzero angle of attack -- otherwise H2's arm carries
    a dead sector and needs rethinking.
    """

    torch.manual_seed(2)
    model = airfrans_train.build_arm("mt_nl_pseudo")
    assert model.drive_pseudo_dim == 8
    domain, _ = airfrans_dataset.case_domain(case)
    with torch.no_grad():
        encoded = model.encode(domain)
    cache = encoded.kernel_cache
    assert cache is not None and cache.value_pseudos is not None
    assert cache.value_pseudos.shape[-1] > 0
    assert float(cache.value_pseudos.abs().sum()) > 0.0


def test_aux_scale_arm_responds_to_the_declared_viscous_scale(case):
    """H4 wiring: only the scale arm consumes the declared viscous scale.

    ``mt_nl_scale`` declares ``viscous_scale`` as a rank-0 global operator
    field and hands it to the kernel decoder, so changing the declared value
    (same weights, same geometry, same drive) must change its prediction.
    ``mt_nl_members`` -- the capacity control -- does NOT declare the field,
    so the identical perturbation is a bitwise no-op for it, pinning that
    the two H4 arms differ by the declaration alone.
    """

    torch.manual_seed(4)
    scale_arm = airfrans_train.build_arm("mt_nl_scale")
    assert scale_arm.kernel_auxiliary_scale_key == "viscous_scale"
    assert scale_arm.kernel_decoder.auxiliary_scale is True
    assert scale_arm.kernel_decoder.mlp_members == 8

    torch.manual_seed(4)
    members_arm = airfrans_train.build_arm("mt_nl_members")
    assert members_arm.kernel_auxiliary_scale_key is None
    assert members_arm.kernel_decoder.auxiliary_scale is False
    assert members_arm.kernel_decoder.mlp_members == 8

    # The baseline singpair arm has no learned pair-radial pathway at all;
    # the H4 arms add exactly the MLP members (plus, scale arm, the contract).
    torch.manual_seed(4)
    assert airfrans_train.build_arm("mt_nl").kernel_decoder.mlp_members == 0

    domain, _ = airfrans_dataset.case_domain(case)
    perturbed, _ = airfrans_dataset.case_domain(case)
    perturbed.global_data["viscous_scale"] = (
        4.0 * perturbed.global_data["viscous_scale"]
    )

    with torch.no_grad():
        for arm, model, should_respond in (
            ("mt_nl_scale", scale_arm, True),
            ("mt_nl_members", members_arm, False),
        ):
            baseline = airfrans_train._predictions(model, domain)
            modified = airfrans_train._predictions(model, perturbed)
            responded = any(
                not torch.equal(modified[name], baseline[name])
                for name in airfrans_train.OUTPUT_FIELD_RANKS
            )
            assert responded is should_respond, arm


def test_log_radial_arm_differs_from_members_only_by_the_feature_knob():
    """H4-L / V4 wiring: the log arm is the members arm plus one knob.

    ``mt_nl_members_log`` must differ from ``mt_nl_members`` -- its
    pre-registered comparison arm -- by ``kernel_log_radial_features=True``
    alone (kwargs diff), so the experiment isolates the feature map.
    Capacity check: the built models carry identical parameter NAMES and
    differ in size by exactly the member MLP's widened first-layer input
    (nearly equal totals; the campaign records both in the reports).
    """

    members_kwargs = airfrans_train._ARM_KWARGS["mt_nl_members"]
    log_kwargs = airfrans_train._ARM_KWARGS["mt_nl_members_log"]
    assert {
        key: value
        for key, value in log_kwargs.items()
        if members_kwargs.get(key) != value
    } == {"kernel_log_radial_features": True}
    assert set(members_kwargs) - set(log_kwargs) == set()

    torch.manual_seed(6)
    log_arm = airfrans_train.build_arm("mt_nl_members_log")
    torch.manual_seed(6)
    members_arm = airfrans_train.build_arm("mt_nl_members")
    assert log_arm.kernel_decoder.log_radial_features is True
    assert members_arm.kernel_decoder.log_radial_features is False
    assert log_arm.kernel_decoder.mlp_members == 8
    assert members_arm.kernel_decoder.mlp_members == 8
    assert log_arm.kernel_auxiliary_scale_key is None

    # Same parameter set; the only shape change is the appended log-radial
    # input block of the member MLP's first layer.
    assert list(log_arm.state_dict()) == list(members_arm.state_dict())
    members_first = members_arm.kernel_decoder.member_mlp[0].weight
    log_first = log_arm.kernel_decoder.member_mlp[0].weight
    assert log_first.shape[-1] == 2 * members_first.shape[-1]
    log_params = sum(p.numel() for p in log_arm.parameters())
    members_params = sum(p.numel() for p in members_arm.parameters())
    assert log_params - members_params == (
        members_first.shape[0] * members_first.shape[-1]
    )
    assert (log_params - members_params) / members_params < 0.05


def test_normalization_and_metrics_are_finite(catalog):
    device = torch.device("cpu")
    bank = airfrans_train.AirFRANSCaseBank(catalog, device=device)
    normalization = airfrans_train.compute_normalization(bank, [CASE_NAME])
    for field, stats in normalization["fields"].items():
        assert math.isfinite(stats["mean"]) and stats["std"] > 0, field
        assert stats["n_points"] == N_QUERY - len(MASKED_ROWS)

    torch.manual_seed(3)
    model = airfrans_train.build_arm("mt_nl")
    aggregate, per_case = airfrans_train._evaluate_cases(
        model, bank, [CASE_NAME], normalization
    )
    for key in airfrans_train.HEADLINE_METRICS:
        assert math.isfinite(aggregate[key]), key
    # The reference per-sample block rides along in every evaluation.
    for key in airfrans_train.GLOBE_HEADLINE_METRICS:
        assert math.isfinite(aggregate[key]), key
    assert math.isfinite(airfrans_train._headline_score(aggregate))
    record = per_case[0]
    assert record["case"] == CASE_NAME
    assert record["n_valid"] == N_QUERY - len(MASKED_ROWS)
    # Row 5 is on-surface AND masked, so one surface point drops out.
    assert record["n_surface_valid"] == N_SURFACE - 1
    assert math.isfinite(record["mae/delta_velocity"])
    assert math.isfinite(record["mae/pressure_coefficient"])


def test_point_segment_distances_analytic():
    """Exact point-to-segment distances on a hand-checkable configuration."""

    starts = torch.tensor([[0.0, 0.0]])
    ends = torch.tensor([[1.0, 0.0]])
    points = torch.tensor([[0.5, 0.3], [2.0, 0.0], [-1.0, -1.0], [0.25, 0.0]])
    distances = airfrans_dataset.point_segment_distances(points, starts, ends)
    expected = torch.tensor([0.3, 1.0, math.sqrt(2.0), 0.0])
    torch.testing.assert_close(distances, expected)
    # Chunking must not change anything.
    chunked = airfrans_dataset.point_segment_distances(
        points, starts, ends, chunk_size=1
    )
    torch.testing.assert_close(chunked, distances, rtol=0.0, atol=0.0)


def _unit_normalization(n_points: int) -> dict:
    """std = 1 for every raw field, so band errors read off directly."""

    return {
        "convention": "unit-std test constants",
        "n_cases": 1,
        "fields": {
            field: {"mean": 0.0, "std": 1.0, "n_points": n_points}
            for field in ("u_x", "u_y", "p", "nut")
        },
    }


def test_distance_bands_localize_known_errors(catalog):
    """Known errors at known wall distances land in exactly their bands.

    The band case puts one query point per band at distance exactly d from
    the boundary polygon's (1, 0) vertex; a pressure error injected at the
    band-2 point and a velocity error at the band-4 point must show up in
    those bands' metrics and nowhere else.
    """

    case = airfrans_dataset.load_case(catalog, BAND_CASE)
    distances = airfrans_dataset.boundary_distances(case)
    torch.testing.assert_close(
        distances,
        torch.tensor(BAND_DISTANCES, dtype=distances.dtype),
        rtol=5.0e-3,
        atol=0.0,
    )

    predictions = {key: value.clone() for key, value in case.targets.items()}
    predictions["pressure_coefficient"][2] += 1.0 / case.dynamic_pressure
    predictions["delta_velocity"][4] += torch.tensor([0.3, 0.4])
    record = airfrans_train._case_metrics(
        case, predictions, _unit_normalization(len(BAND_DISTANCES)), distances
    )

    n_bands = airfrans_train.N_DISTANCE_BANDS
    assert [record[f"n_valid@band{k}"] for k in range(n_bands)] == [1] * n_bands
    # Pressure: raw error of 1 Pa at the band-2 point only (unit std).
    for band in range(n_bands):
        expected = 1.0 if band == 2 else 0.0
        assert record[f"zscore_mse/p@band{band}"] == pytest.approx(
            expected, abs=1.0e-3
        ), band
    # Velocity: |dU| error 0.5 (nondimensional MAE) at the band-4 point;
    # raw component errors 0.3 |U_inf| and 0.4 |U_inf| in the z-score MSE.
    magnitude = case.u_inf_magnitude
    for band in range(n_bands):
        assert record[f"mae/delta_velocity@band{band}"] == pytest.approx(
            0.5 if band == 4 else 0.0, abs=1.0e-6
        ), band
        assert record[f"zscore_mse/u_x@band{band}"] == pytest.approx(
            (0.3 * magnitude) ** 2 if band == 4 else 0.0, rel=1.0e-4, abs=1.0e-6
        ), band
        assert record[f"zscore_mse/u_y@band{band}"] == pytest.approx(
            (0.4 * magnitude) ** 2 if band == 4 else 0.0, rel=1.0e-4, abs=1.0e-6
        ), band
        assert record[f"zscore_mse/nut@band{band}"] == pytest.approx(0.0, abs=1.0e-9)
    # The whole-domain metrics see the same errors, averaged over all points.
    assert record["zscore_mse/p"] == pytest.approx(1.0 / len(BAND_DISTANCES), rel=1e-3)
    # Band counts partition the valid points.
    assert sum(record[f"n_valid@band{k}"] for k in range(n_bands)) == record["n_valid"]


def test_globe_schedule_formula():
    """The protocol-v1 LR law: scaled peak, cosine to peak/64, no warmup."""

    spec = airfrans_train.resolve_schedule("globe", None, 4096)
    assert spec["base_lr"] == pytest.approx(1.0e-3)
    # 1e-3 * sqrt(1 * 4096 / 2048) = 1.41421e-3 (single GPU, 4096 queries).
    assert spec["peak_lr"] == pytest.approx(1.0e-3 * math.sqrt(2.0))
    assert spec["floor_lr"] == pytest.approx(spec["peak_lr"] / 64.0)
    assert spec["warmup"] is None  # GLOBE has no warmup
    lrs = [
        airfrans_train.schedule_learning_rate(spec, epoch, 1000)
        for epoch in range(1, 1001)
    ]
    assert lrs[0] == pytest.approx(spec["peak_lr"])
    assert lrs[-1] == pytest.approx(spec["floor_lr"])
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    # An explicit --lr overrides the base in either protocol.
    assert airfrans_train.resolve_schedule("globe", 2.0e-3, 2048)[
        "peak_lr"
    ] == pytest.approx(2.0e-3)
    flat = airfrans_train.resolve_schedule("flat", None, 4096)
    assert flat["base_lr"] == flat["peak_lr"] == flat["floor_lr"]
    assert flat["base_lr"] == pytest.approx(3.0e-4)
    assert airfrans_train.schedule_learning_rate(flat, 7, 100) == pytest.approx(3.0e-4)
    with pytest.raises(ValueError, match="unknown schedule"):
        airfrans_train.resolve_schedule("cyclic", None, 4096)


def test_globe_schedule_smoke_run(catalog, tmp_path):
    """A 2-epoch --schedule globe run: LR endpoints, recorded spec, bands."""

    report = airfrans_train.run_experiment(
        catalog_dir=catalog,
        task="scarce",
        arm="mt_nl",
        epochs=2,
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        queries_per_step=16,
        schedule="globe",
        validate_every=1,
    )
    peak = 1.0e-3 * math.sqrt(16 / 2048)
    assert report["schedule"]["name"] == "globe"
    assert report["schedule"]["peak_lr"] == pytest.approx(peak)
    assert report["history"][0]["lr"] == pytest.approx(peak)
    assert report["history"][-1]["lr"] == pytest.approx(peak / 64.0)
    # Validation epochs log both conventions' headline metrics on one line.
    validation_log = report["history"][-1]
    pooled = validation_log["validation_pooled_train"]
    persample = validation_log["validation_reference_persample"]
    assert set(pooled) == {"u_x", "u_y", "p", "p_surface"}
    assert set(persample) == {"u_x", "u_y", "c_p", "c_p_surface_only"}
    assert all(math.isfinite(value) for value in pooled.values())
    assert all(math.isfinite(value) for value in persample.values())
    # Band diagnostics ride along in every evaluation (eval-only included).
    assert report["band_edges"][0] == 0.0 and report["band_edges"][-1] is None
    test_metrics = report["splits"]["test"]
    band_counts = [
        test_metrics[f"n_valid@band{k}"] for k in range(airfrans_train.N_DISTANCE_BANDS)
    ]
    assert sum(band_counts) == N_QUERY - len(MASKED_ROWS)


def test_checkpoint_resume_continues_bitwise(catalog, tmp_path):
    """Kill-and-resume reproduces the uninterrupted run exactly.

    Epoch 1 with --checkpoint-every 1 stands in for a walltime-killed run;
    resuming to epoch 2 must reproduce the fresh 2-epoch run's second-epoch
    loss bit-for-bit (generators, optimizer state, and best-so-far state
    all restored; nothing in an epoch consumes global RNG).
    """

    common = dict(
        catalog_dir=catalog,
        task="scarce",
        arm="mt_nl",
        seed=1,
        device="cpu",
        queries_per_step=8,
        validate_every=1,
    )
    fresh = airfrans_train.run_experiment(
        **common, epochs=2, output_dir=tmp_path / "fresh"
    )
    interrupted = airfrans_train.run_experiment(
        **common, epochs=1, output_dir=tmp_path / "resumed", checkpoint_every=1
    )
    assert airfrans_train.resume_path(
        tmp_path / "resumed", "mt_nl", "scarce", 1
    ).is_file()
    resumed = airfrans_train.run_experiment(
        **common,
        epochs=2,
        output_dir=tmp_path / "resumed",
        checkpoint_every=1,
        resume=True,
    )
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]
    assert (
        resumed["history"][0]["train_loss"] == interrupted["history"][0]["train_loss"]
    )
    assert resumed["history"][1]["train_loss"] == fresh["history"][1]["train_loss"]
    assert (
        resumed["best_validation_zscore_mse_mean"]
        == fresh["best_validation_zscore_mse_mean"]
    )


def _square_boundary(n_query: int) -> dict:
    """Minimal geometry for a hand-constructed AirFRANSCase."""

    return {
        "boundary_points": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        "boundary_cells": torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64
        ),
        "query_points": torch.zeros(n_query, 2),
    }


def _reference_persample_zscore(true_vals, pred_vals):
    """Independent numpy reimplementation of GLOBE's per-variant reduction.

    NaN filter, then the SAMPLE's own unbiased std of the true values
    (``ddof=1``, torch's ``.std()`` default), std <= 0 skipped, point-mean
    of the squared normalized error.
    """

    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    valid = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    true_vals, pred_vals = true_vals[valid], pred_vals[valid]
    if true_vals.size < 2:
        return None
    std = true_vals.std(ddof=1)
    if not std > 0:
        return None
    return float((((pred_vals - true_vals) / std) ** 2).mean())


def test_globe_zscore_convention_hand_built():
    """The reference per-sample convention on a fully hand-checked sample.

    Five points: one all-NaN row (masked-label semantics), two on-surface
    rows, a constant true log-nut field (the std > 0 guard), and known
    errors -- checked against literal hand computations AND an independent
    numpy reimplementation of the benchmark reduction for every variant.
    """

    # float64 throughout so the hand literals hold to 1e-12 (real catalogs
    # store float32; the convention itself is precision-agnostic).
    nan, f64 = math.nan, torch.float64
    true_delta = torch.tensor(
        [[0.1, 0.0], [-0.1, 0.2], [0.2, -0.2], [-0.2, 0.4], [nan, nan]], dtype=f64
    )
    pred_delta = torch.nan_to_num(true_delta) + torch.tensor(
        [[0.03, 0.04], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=f64
    )
    true_cp = torch.tensor([1.0, 2.0, 3.0, 4.0, nan], dtype=f64)
    pred_cp = torch.tensor([1.5, 2.0, 3.0, 4.0, 0.0], dtype=f64)
    # Constant true field: the per-sample std is zero (guard teeth).
    true_log_nut = torch.tensor([1.0, 1.0, 1.0, 1.0, nan], dtype=f64)
    pred_log_nut = torch.tensor([1.2, 1.0, 1.0, 1.0, 0.0], dtype=f64)
    true_cpt = torch.tensor([2.0, 3.0, 4.0, 5.0, nan], dtype=f64)
    case = airfrans_dataset.AirFRANSCase(
        name="hand_built",
        **_square_boundary(5),
        targets={
            "delta_velocity": true_delta,
            "pressure_coefficient": true_cp,
            "log_nut_ratio": true_log_nut,
        },
        cpt=true_cpt,
        is_surface=torch.tensor([True, True, False, False, True]),
        u_inf=torch.tensor([50.0, 0.0], dtype=torch.float64),
        nu=NU,
        chord=1.0,
    )
    predictions = {
        "delta_velocity": pred_delta,
        "pressure_coefficient": pred_cp,
        "log_nut_ratio": pred_log_nut,
    }
    record = airfrans_train._globe_zscore_metrics(case, predictions)
    assert set(record) == {
        f"globe_zscore_mse/{key}"
        for key in (
            "u_x",
            "u_y",
            "u_mag",
            "c_p",
            "c_p_surface_only",
            "c_pt",
            "c_pt_surface_only",
            "log_nut_ratio",
        )
    }

    # Hand-computed literals (unbiased stds):
    # c_p: err [0.5,0,0,0], std([1,2,3,4]) = sqrt(5/3) -> (0.25/4)*(3/5).
    assert record["globe_zscore_mse/c_p"] == pytest.approx(0.0375, rel=1e-12)
    # c_p surface: rows {0,1} (the NaN surface row drops AFTER the surface
    # mask), std([1,2]) = sqrt(1/2) -> (0.25/2)*2 = 0.25.
    assert record["globe_zscore_mse/c_p_surface_only"] == pytest.approx(0.25, rel=1e-12)
    # u_x: err 0.03 at one of four rows, var([.1,-.1,.2,-.2], ddof=1) = 0.1/3.
    assert record["globe_zscore_mse/u_x"] == pytest.approx(0.00675, rel=1e-12)
    # u_y: err 0.04 at one of four rows, var([0,.2,-.2,.4], ddof=1) = 0.2/3.
    assert record["globe_zscore_mse/u_y"] == pytest.approx(0.006, rel=1e-12)
    # Constant true field: the sample's std is zero, the variant is skipped.
    assert record["globe_zscore_mse/log_nut_ratio"] is None

    # Every variant against the independent reimplementation (including the
    # engineered magnitudes and total pressure).
    direction = np.array([1.0, 0.0])
    true_u = true_delta.numpy() + direction
    pred_u = pred_delta.numpy() + direction
    pred_cpt = pred_cp.numpy() + (pred_u**2).sum(axis=1)
    surface = case.is_surface.numpy()

    def surfaced(values):
        return np.where(surface, values, np.nan)

    expected = {
        "u_x": _reference_persample_zscore(true_u[:, 0], pred_u[:, 0]),
        "u_y": _reference_persample_zscore(true_u[:, 1], pred_u[:, 1]),
        "u_mag": _reference_persample_zscore(
            np.linalg.norm(true_u, axis=1), np.linalg.norm(pred_u, axis=1)
        ),
        "c_p": _reference_persample_zscore(true_cp.numpy(), pred_cp.numpy()),
        "c_p_surface_only": _reference_persample_zscore(
            surfaced(true_cp.numpy()), surfaced(pred_cp.numpy())
        ),
        "c_pt": _reference_persample_zscore(true_cpt.numpy(), pred_cpt),
        "c_pt_surface_only": _reference_persample_zscore(
            surfaced(true_cpt.numpy()), surfaced(pred_cpt)
        ),
        "log_nut_ratio": _reference_persample_zscore(
            true_log_nut.numpy(), pred_log_nut.numpy()
        ),
    }
    for key, value in expected.items():
        actual = record[f"globe_zscore_mse/{key}"]
        if value is None:
            assert actual is None, key
        else:
            assert actual == pytest.approx(value, rel=1e-9), key


def test_globe_zscore_mean_predictor_floor(catalog):
    """The convention check with teeth: the per-sample mean predictor floor.

    Under the benchmark's torch-default UNBIASED std, predicting each
    sample's own true mean scores exactly (N_valid - 1) / N_valid per
    variant -- the biased/unbiased variance ratio -- which is the 'floor of
    1.0' identity up to Bessel's correction (5.5e-6 at 180k points, exact
    values asserted here).
    """

    # (a) Volume variants on the band case (6 points, no NaN, no surface).
    case = airfrans_dataset.load_case(catalog, BAND_CASE)
    n = len(BAND_DISTANCES)
    predictions = {
        "delta_velocity": case.targets["delta_velocity"]
        .double()
        .mean(dim=0)
        .expand(n, 2),
        "pressure_coefficient": torch.full(
            (n,), float(case.targets["pressure_coefficient"].double().mean())
        ).double(),
        "log_nut_ratio": torch.full(
            (n,), float(case.targets["log_nut_ratio"].double().mean())
        ).double(),
    }
    record = airfrans_train._globe_zscore_metrics(case, predictions)
    floor = (n - 1) / n
    for key in ("u_x", "u_y", "c_p", "log_nut_ratio"):
        assert record[f"globe_zscore_mse/{key}"] == pytest.approx(floor, abs=1e-12), key
    # (mag and c_pt are engineered nonlinearly from the mean fields, so the
    # mean-field predictor is not their per-variant mean: no floor claim.)
    assert record["globe_zscore_mse/c_p_surface_only"] is None  # no surface

    # (b) The surface variant floors at (N_surface_valid - 1) / N_surface_valid
    # when the prediction is the surface mean -- and the NaN-masked surface
    # row must already be excluded (masking order teeth).
    main = airfrans_dataset.load_case(catalog, CASE_NAME)
    cp = main.targets["pressure_coefficient"].double()
    surface_valid = main.is_surface & torch.isfinite(cp)
    n_surface = int(surface_valid.sum())
    assert n_surface == N_SURFACE - 1  # row 5 is surface AND masked
    predictions = {
        "delta_velocity": torch.nan_to_num(main.targets["delta_velocity"].double()),
        "pressure_coefficient": torch.full(
            (main.n_query,), float(cp[surface_valid].mean())
        ).double(),
        "log_nut_ratio": torch.nan_to_num(main.targets["log_nut_ratio"].double()),
    }
    record = airfrans_train._globe_zscore_metrics(main, predictions)
    assert record["globe_zscore_mse/c_p_surface_only"] == pytest.approx(
        (n_surface - 1) / n_surface, abs=1e-12
    )
    # Exact predictions elsewhere score exactly zero.
    assert record["globe_zscore_mse/u_x"] == pytest.approx(0.0, abs=1e-15)
    assert record["globe_zscore_mse/log_nut_ratio"] == pytest.approx(0.0, abs=1e-15)
