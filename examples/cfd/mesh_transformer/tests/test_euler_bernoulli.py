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

"""Contracts for the multi-field euler_bernoulli family: exact Bernoulli
pressure labels, reuse of the certified potential-flow machinery, the
drive-parity identities behind the pre-registered linear pressure wall, and
the first multi-field (velocity + pressure) model arms."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from euler_bernoulli import (  # noqa: E402
    MODEL_NAMES,
    OUTPUT_FIELD_RANKS,
    SPLITS,
    FarFieldBoundaryMean,
    _build_model,
    _combined_relative_l2,
    _multi_field_loss,
    _predictions,
    bernoulli_pressure,
    body_tangency_residual,
    build_euler_bernoulli_sample,
    pressure_at_physical,
    run_experiment,
)
from potential_flow import (  # noqa: E402
    SPLITS as POTENTIAL_FLOW_SPLITS,
)
from potential_flow import (  # noqa: E402
    build_potential_flow_velocity_sample,
    disturbance_velocity,
)


def _every_split(seeds=(0,)):
    for split_name, spec in SPLITS.items():
        for seed in seeds:
            yield (
                split_name,
                build_euler_bernoulli_sample(
                    seed * 7919 + 11, dtype=torch.float64, **spec
                ),
            )


def test_pressure_satisfies_the_bernoulli_identity_independently() -> None:
    """p* = (|U|^2 - |u_total|^2)/2 against independently recomputed |u|.

    The disturbance velocity is recomputed from independently Newton-inverted
    preimages of the sample's *physical* query points, the Bernoulli identity
    is applied to it, and the result must match the stored pressure label to
    1e-12 in float64, on every split.
    """

    for split_name, sample in _every_split():
        recomputed = pressure_at_physical(
            sample.body,
            sample.canonical_freestream,
            sample.freestream,
            sample.circulation,
            sample.domain.interior.points,
        )
        error = float((recomputed - sample.targets["pressure"]).abs().max())
        assert error < 1.0e-12, (split_name, error)
        # The expanded drive-quadratic form is the same identity:
        # p* = -U . u_d - |u_d|^2 / 2 for a unit far field.
        u_d = disturbance_velocity(
            sample.body,
            sample.canonical_freestream,
            sample.circulation,
            sample.query_preimages,
        )
        expanded = -(sample.freestream[None, :] * u_d).sum(dim=-1) - 0.5 * (
            u_d.square().sum(dim=-1)
        )
        torch.testing.assert_close(
            expanded, sample.targets["pressure"], rtol=0.0, atol=1.0e-12
        )


def test_family_reuses_the_certified_flow_and_splits() -> None:
    """Same splits, and seed for seed bitwise the same flow as Family A'.

    The velocity target and every model-facing tensor come unchanged from
    the certified ``potential_flow_velocity`` builder, so its FD, mirror,
    and independent-inversion certifications transfer bitwise.
    """

    assert SPLITS == POTENTIAL_FLOW_SPLITS["potential_flow"]
    assert set(SPLITS) == {
        "in_distribution",
        "unseen_geometry_modes",
        "wilder_shapes",
        "circulation_ood",
        "farfield_queries",
    }
    a = build_potential_flow_velocity_sample(21, dtype=torch.float64)
    b = build_euler_bernoulli_sample(21, dtype=torch.float64)
    assert torch.equal(b.targets["velocity"], a.target)
    assert torch.equal(a.domain.interior.points, b.domain.interior.points)
    assert torch.equal(
        a.domain.boundaries["dirichlet"].points,
        b.domain.boundaries["dirichlet"].points,
    )
    assert torch.equal(
        a.domain.global_data["freestream_velocity"],
        b.domain.global_data["freestream_velocity"],
    )
    assert a.circulation == b.circulation
    assert b.targets["pressure"].shape == (256,)
    assert torch.isfinite(b.targets["pressure"]).all()


def test_on_body_impermeability_through_the_velocity() -> None:
    """The exact total velocity is tangent to the smooth body curve.

    Im[conj(t) u_total] = 0 with t the exact-map tangent i zeta e^{i alpha}
    G'(zeta) -- the velocity-level streamline statement -- at the stored
    panel-midpoint preimages and on a dense on-circle grid, to 1e-12.
    """

    for split_name, sample in _every_split():
        residual = body_tangency_residual(
            sample.body,
            sample.canonical_freestream,
            sample.circulation,
            sample.flow.boundary_midpoint_preimages,
        )
        assert float(residual.abs().max()) < 1.0e-12, split_name
    dense = torch.polar(
        torch.ones(4096, dtype=torch.float64),
        2.0 * math.pi * torch.arange(4096, dtype=torch.float64) / 4096.0,
    )
    sample = build_euler_bernoulli_sample(3, dtype=torch.float64)
    residual = body_tangency_residual(
        sample.body, sample.canonical_freestream, sample.circulation, dense
    )
    assert float(residual.abs().max()) < 1.0e-12


def test_velocity_is_drive_odd_and_pressure_is_drive_even() -> None:
    """The exact parity identities behind the pre-registered linear wall.

    Negating the joint drive (U, Gamma) exactly negates the disturbance
    velocity and exactly preserves the Bernoulli pressure; the training
    drive distribution is symmetric under this negation, so an exactly
    drive-linear (odd) model can fit the velocity but only the zero
    function for the pressure.
    """

    sample = build_euler_bernoulli_sample(7, dtype=torch.float64)
    u = disturbance_velocity(
        sample.body,
        sample.canonical_freestream,
        sample.circulation,
        sample.query_preimages,
    )
    u_negated = disturbance_velocity(
        sample.body,
        -sample.canonical_freestream,
        -sample.circulation,
        sample.query_preimages,
    )
    torch.testing.assert_close(u_negated, -u, rtol=0.0, atol=1.0e-13)
    pressure_negated = bernoulli_pressure(-sample.freestream, u_negated)
    torch.testing.assert_close(
        pressure_negated, sample.targets["pressure"], rtol=0.0, atol=1.0e-13
    )


def test_linear_arm_is_exactly_drive_odd_the_wall_is_structural() -> None:
    """The pre-registered wall control really is exactly odd in the drive.

    In float64, negating both global drive fields negates every prediction
    of the ``field_mode='linear'`` arm bit-for-bit-tight (its pressure
    prediction is therefore an odd fit of an even target: the wall).  The
    nonlinear arm must NOT be odd -- its even (drive-quadratic) response
    component is exactly what makes the pressure representable.
    """

    sample = build_euler_bernoulli_sample(5, dtype=torch.float64)

    def negated_predictions(model):
        with torch.no_grad():
            plus = _predictions(model, sample.domain)
            gd = sample.domain.global_data
            gd["freestream_velocity"] = -gd["freestream_velocity"]
            gd["circulation"] = -gd["circulation"]
            minus = _predictions(model, sample.domain)
            gd["freestream_velocity"] = -gd["freestream_velocity"]
            gd["circulation"] = -gd["circulation"]
        return plus, minus

    torch.manual_seed(0)
    linear = _build_model("mt_singpair_linear").double().eval()
    plus, minus = negated_predictions(linear)
    for field in OUTPUT_FIELD_RANKS:
        odd_violation = float((plus[field] + minus[field]).abs().max())
        scale = float(plus[field].abs().max())
        assert odd_violation <= 1.0e-10 * max(scale, 1.0), (field, odd_violation)

    torch.manual_seed(0)
    nonlinear = _build_model("mt_singpair_nl").double().eval()
    plus, minus = negated_predictions(nonlinear)
    even_component = float((plus["pressure"] + minus["pressure"]).abs().max())
    assert even_component > 1.0e-8


def test_model_arms_multi_field_construction_forward_backward() -> None:
    """Every arm builds, emits both fields, and backpropagates finitely.

    This is the first multi-field output: the prediction mesh must unpack
    into a named rank-1 ``velocity`` and rank-0 ``pressure`` of the target
    shapes, and the multi-field loss must produce finite gradients in every
    transformer arm (including the pseudo-sector parameters of the typed
    arm).
    """

    sample = build_euler_bernoulli_sample(5)
    expected_shapes = {"velocity": (256, 2), "pressure": (256,)}
    parameter_counts = {}
    for name in MODEL_NAMES:
        torch.manual_seed(3)
        model = _build_model(name)
        predictions = _predictions(model, sample.domain)
        assert set(predictions) == set(OUTPUT_FIELD_RANKS), name
        for field, prediction in predictions.items():
            assert prediction.shape == expected_shapes[field], (name, field)
            assert prediction.shape == sample.targets[field].shape, (name, field)
            assert torch.isfinite(prediction).all(), (name, field)
        parameter_counts[name] = sum(p.numel() for p in model.parameters())
        if parameter_counts[name]:
            loss = _multi_field_loss(predictions, sample.targets)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients, name
            assert all(torch.isfinite(g).all() for g in gradients), name
        if name == "mt_singpair_nl_pseudo":
            pseudo_gradients = [
                parameter.grad
                for parameter_name, parameter in model.named_parameters()
                if "pseudo" in parameter_name and parameter.grad is not None
            ]
            assert pseudo_gradients
    assert parameter_counts["boundary_mean"] == 0
    # The nonlinear read-in adds capacity over linear; the pseudo sector
    # adds capacity over plain nonlinear.  The quadratic (declared-degree)
    # arm sits between linear and nonlinear: it is the linear machinery
    # plus one bilinear composition.
    assert (
        parameter_counts["mt_singpair_linear"]
        < parameter_counts["mt_singpair_q2"]
        < parameter_counts["mt_singpair_nl"]
        < parameter_counts["mt_singpair_nl_pseudo"]
    )
    assert (
        parameter_counts["mt_singpair_q2"]
        < parameter_counts["mt_singpair_q2_pseudo"]
    )


def test_registry_schemas_and_modes() -> None:
    """Arm registry contracts: modes, dictionary, schemas, and rejection."""

    linear = _build_model("mt_singpair_linear")
    nonlinear = _build_model("mt_singpair_nl")
    pseudo = _build_model("mt_singpair_nl_pseudo")
    quadratic = _build_model("mt_singpair_q2")
    quadratic_pseudo = _build_model("mt_singpair_q2_pseudo")
    for model in (linear, nonlinear, pseudo, quadratic, quadratic_pseudo):
        assert model.output_field_ranks == {"velocity": 1, "pressure": 0}
        assert model.kernel_decoder.include_single_layer_member is True
        assert model.kernel_decoder.n_members == 2
        assert model.boundary_field_ranks == {
            "dirichlet": {"operator": {}, "drive": {}}
        }
    assert linear.field_mode == "linear"
    assert nonlinear.field_mode == "zero_preserving_nonlinear"
    assert pseudo.field_mode == "zero_preserving_nonlinear"
    assert quadratic.field_mode == "quadratic"
    assert quadratic_pseudo.field_mode == "quadratic"
    assert quadratic.quadratic_read_in is not None
    assert quadratic_pseudo.quadratic_read_in is not None
    assert linear.quadratic_read_in is None
    assert nonlinear.quadratic_read_in is None
    for model in (linear, nonlinear, quadratic):
        assert model.global_field_ranks == {
            "operator": {},
            "drive": {"circulation": 0, "freestream_velocity": 1},
        }
        assert model.drive_pseudo_dim == 0
    for model in (pseudo, quadratic_pseudo):
        assert model.global_field_ranks == {
            "operator": {},
            "drive": {"circulation": "0o", "freestream_velocity": 1},
        }
        assert model.drive_pseudo_dim == 8

    assert isinstance(_build_model("boundary_mean"), FarFieldBoundaryMean)
    with pytest.raises(ValueError, match="unknown model"):
        _build_model("pair_kernel")
    assert MODEL_NAMES == (
        "boundary_mean",
        "mt_singpair_linear",
        "mt_singpair_nl",
        "mt_singpair_nl_pseudo",
        "mt_singpair_q2",
        "mt_singpair_q2_pseudo",
    )


def test_quadratic_arm_is_exactly_degree_two_in_the_drive() -> None:
    """The iteration-35 declared-degree contract on the real benchmark arm.

    Scaling the joint global drive (U, Gamma) by alpha at fixed geometry
    must make every q2 prediction entry an exact degree-<=2 polynomial in
    alpha (velocity degree 1, pressure degree 2 are both representable and
    nothing higher exists), at float64 machine precision, with random
    weights -- the structural fix for the implicit-degree blowup that
    iteration 34 measured on the nonlinear arms (effective degree ~21).
    """

    sample = build_euler_bernoulli_sample(5, dtype=torch.float64)
    torch.manual_seed(1)
    model = _build_model("mt_singpair_q2").double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.3, 0.3)

    gd = sample.domain.global_data
    base = {k: gd[k].clone() for k in ("freestream_velocity", "circulation")}

    @torch.no_grad()
    def output(alpha: float) -> torch.Tensor:
        for key, value in base.items():
            gd[key] = value * alpha
        predictions = _predictions(model, sample.domain)
        for key, value in base.items():
            gd[key] = value
        return torch.cat(
            (predictions["velocity"].flatten(), predictions["pressure"].flatten())
        )

    at_one, at_two = output(1.0), output(2.0)
    quadratic = (at_two - 2.0 * at_one) / 2.0
    linear = at_one - quadratic
    assert float(output(0.0).abs().max()) == 0.0
    for alpha in (0.37, 3.1, 11.0, -1.6):
        actual = output(alpha)
        predicted = linear * alpha + quadratic * alpha**2
        residual = float((actual - predicted).abs().max() / actual.abs().max())
        assert residual < 1.0e-12, (alpha, residual)


def test_far_field_boundary_mean_floor_is_exactly_one() -> None:
    """The parameter-free control predicts the far-field constants exactly.

    Zero disturbance velocity and zero Bernoulli pressure give per-field and
    combined relative L2 of exactly 1.0 -- the declared no-response floor.
    """

    sample = build_euler_bernoulli_sample(9, dtype=torch.float64)
    model = FarFieldBoundaryMean()
    predictions = _predictions(model, sample.domain)
    for field in OUTPUT_FIELD_RANKS:
        assert float(predictions[field].abs().max()) == 0.0
    assert abs(_combined_relative_l2(predictions, sample.targets) - 1.0) < 1.0e-12


def test_generator_determinism() -> None:
    """The same seed reproduces both targets bit-for-bit."""

    first = build_euler_bernoulli_sample(321)
    second = build_euler_bernoulli_sample(321)
    for field in OUTPUT_FIELD_RANKS:
        assert torch.equal(first.targets[field], second.targets[field])
    assert torch.equal(first.domain.interior.points, second.domain.interior.points)


def test_driver_smoke_produces_finite_multi_field_json(tmp_path: Path) -> None:
    """A zero-step CPU control run writes the per-field report shape.

    Split keys must include ``<split>``, ``<split>/velocity``, and
    ``<split>/pressure`` for all five reused exterior-flow splits, and the
    parameter-free floor must report exactly 1.0 everywhere.
    """

    report = run_experiment(
        model_name="boundary_mean",
        steps=0,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    on_disk = json.loads(
        (tmp_path / "euler_bernoulli_boundary_mean_seed3.json").read_text()
    )
    expected_keys = set()
    for split in SPLITS:
        expected_keys |= {split, f"{split}/velocity", f"{split}/pressure"}
    for payload in (report, on_disk):
        assert payload["parameters"] == 0
        assert payload["family"] == "euler_bernoulli"
        assert set(payload["splits"]) == expected_keys
        for value in payload["splits"].values():
            assert math.isfinite(value)
            assert abs(value - 1.0) < 1.0e-6
        assert "pre_registered_prediction" in payload["design_notes"]
        # Operator-fidelity block: the zero-velocity/zero-pressure floor is
        # exactly harmonic and exactly Bernoulli-consistent (p_bern(0) = 0),
        # and no maximum principle is licensed on this multi-field family.
        fidelity = payload["fidelity"]
        assert set(fidelity["pde_residual"]) == set(SPLITS)
        assert all(abs(v) < 1.0e-9 for v in fidelity["pde_residual"].values())
        assert set(fidelity["bernoulli_consistency"]) == set(SPLITS)
        assert all(
            abs(v) < 1.0e-9 for v in fidelity["bernoulli_consistency"].values()
        )
        assert fidelity["max_principle_violation"] is None


def test_driver_smoke_trains_one_transformer_step(tmp_path: Path) -> None:
    """One CPU optimizer step of the primary arm exercises the full loop:
    multi-field loss, backward, validation bookkeeping, and the JSON report."""

    report = run_experiment(
        model_name="mt_singpair_nl",
        steps=1,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    assert report["parameters"] > 0
    assert len(report["history"]) == 1
    assert math.isfinite(report["history"][0]["validation_mean_per_field_relative_l2"])
    for value in report["splits"].values():
        assert math.isfinite(value)
