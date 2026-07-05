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

"""Contracts for the rotational-Euler multi-field family: the steady-Euler
momentum certification of the exact labels (the family's headline
credential), the Bessel-J series machinery, the drive-parity identities
behind the pre-registered linear pressure wall, the eigenvalue guard, and
the one-head multi-field model arms."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from euler_rotational import (  # noqa: E402
    FIRST_DIRICHLET_EIGENVALUE,
    MODEL_NAMES,
    OUTPUT_FIELD_RANKS,
    SPLITS,
    DriveBoundaryMean,
    _build_model,
    _combined_relative_l2,
    _multi_field_loss,
    _predictions,
    bessel_j,
    build_euler_rotational_sample,
    divergence_residual,
    euler_momentum_residual,
    helmholtz_residual,
    pressure,
    run_experiment,
    streamfunction,
    velocity,
    vorticity,
)

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402


def _every_split(seeds=(0, 1)):
    for split_name, spec in SPLITS.items():
        for seed in seeds:
            yield (
                split_name,
                build_euler_rotational_sample(
                    seed * 7919 + 11, dtype=torch.float64, **spec
                ),
            )


def test_bessel_series_is_exact() -> None:
    """The J_m power series against exact identities and references.

    The three-term recurrence J_{m-1}(x) + J_{m+1}(x) = (2m/x) J_m(x) and
    Neumann's identity J_0(x)^2 + 2 sum_k J_k(x)^2 = 1 hold to float64
    roundoff.  torch.special.bessel_j0/j1 are only single-precision-grade
    kernels (measured ~3e-7 against scipy), so the comparison against them
    is deliberately loose; scipy, where available, pins the series at
    1e-13.
    """

    x = torch.linspace(0.05, 3.0, 30, dtype=torch.float64)
    for order in range(1, 7):
        lhs = bessel_j(order - 1, x) + bessel_j(order + 1, x)
        rhs = (2.0 * order / x) * bessel_j(order, x)
        torch.testing.assert_close(lhs, rhs, rtol=1.0e-12, atol=1.0e-14)
    total = bessel_j(0, x).square()
    for order in range(1, 21):
        total = total + 2.0 * bessel_j(order, x).square()
    torch.testing.assert_close(total, torch.ones_like(x), rtol=0.0, atol=1.0e-13)
    torch.testing.assert_close(
        bessel_j(0, x), torch.special.bessel_j0(x), rtol=0.0, atol=5.0e-7
    )
    torch.testing.assert_close(
        bessel_j(1, x), torch.special.bessel_j1(x), rtol=0.0, atol=5.0e-7
    )
    scipy_special = pytest.importorskip("scipy.special")
    for order in (0, 1, 4):
        reference = torch.from_numpy(scipy_special.jv(order, x.numpy()))
        torch.testing.assert_close(
            bessel_j(order, x), reference, rtol=0.0, atol=1.0e-13
        )


def test_exact_labels_satisfy_steady_euler_across_splits() -> None:
    """THE HEADLINE CERTIFICATION: momentum, incompressibility, Helmholtz.

    At the sample's random interior queries, in float64 autograd through
    the closed-form fields, on two seeds of every split (including the
    near-eigenvalue tier -- its labels must stay certified or the tier is
    dropped):

    - steady Euler momentum ||(u.grad)u + grad p|| / ||(u.grad)u|| <= 1e-10
      (measured ~1e-15/1e-16; this verifies the rotational Bernoulli
      derivation H(psi) = -c^2 psi^2 / 2, not just the code);
    - incompressibility ||div u|| L / ||u|| <= 1e-12 (exact by
      construction);
    - streamfunction Helmholtz ||(lap + c^2) psi|| L^2 / ||psi|| <= 1e-10.
    """

    for split_name, sample in _every_split():
        points = sample.domain.interior.points
        momentum = euler_momentum_residual(sample.flow, points)
        assert momentum < 1.0e-10, (split_name, momentum)
        incompressibility = divergence_residual(sample.flow, points)
        assert incompressibility < 1.0e-12, (split_name, incompressibility)
        helmholtz = helmholtz_residual(sample.flow, points)
        assert helmholtz < 1.0e-10, (split_name, helmholtz)


def test_labels_and_derivative_machinery_are_consistent() -> None:
    """Closed-form fields against independent derivative and trace paths.

    The closed-form velocity (manual S_m' chain rule) must match float64
    autograd of the streamfunction; the stored targets must equal the
    closed-form fields at the stored queries; the streamfunction's boundary
    trace must equal the direct mode sum sum_m Re[c_m e^{im theta}] (the
    radial normalization is exactly 1 on the boundary); and the vorticity
    is genuinely nonzero -- this family is rotational.
    """

    for split_name, sample in _every_split(seeds=(0,)):
        queries = sample.domain.interior.points
        pts = queries.clone().requires_grad_(True)
        psi = streamfunction(sample.flow, pts)
        (grad,) = torch.autograd.grad(psi.sum(), pts)
        u_autograd = torch.stack((grad[:, 1], -grad[:, 0]), dim=-1)
        torch.testing.assert_close(
            u_autograd, velocity(sample.flow, queries), rtol=0.0, atol=1.0e-12
        )
        torch.testing.assert_close(
            sample.targets["velocity"],
            velocity(sample.flow, queries),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            sample.targets["pressure"],
            pressure(sample.flow, queries),
            rtol=0.0,
            atol=0.0,
        )
        boundary_points = sample.domain.boundaries["dirichlet"].points
        local = boundary_points - sample.flow.center[None, :]
        theta = torch.atan2(local[:, 1], local[:, 0])
        trace = torch.zeros_like(theta)
        for order, coefficient in sample.flow.coefficients.items():
            trace = trace + (
                coefficient * torch.polar(torch.ones_like(theta), order * theta)
            ).real
        torch.testing.assert_close(
            streamfunction(sample.flow, boundary_points),
            trace,
            rtol=0.0,
            atol=1.0e-10,
        )
        omega = vorticity(sample.flow, queries)
        assert float(omega.abs().max()) > 1.0e-2, split_name


def test_velocity_is_drive_odd_and_pressure_is_drive_even() -> None:
    """The exact parity identities behind the pre-registered linear wall.

    Negating every mode coefficient negates the boundary-velocity drive and
    the interior velocity exactly, and preserves the pressure exactly; the
    phase distribution is negation-symmetric, so an exactly drive-linear
    (odd) model can fit the velocity but only the zero function for the
    pressure.
    """

    sample = build_euler_rotational_sample(
        7, dtype=torch.float64, **SPLITS["in_distribution"]
    )
    negated = dataclasses.replace(
        sample.flow,
        coefficients={m: -c for m, c in sample.flow.coefficients.items()},
    )
    queries = sample.domain.interior.points
    torch.testing.assert_close(
        velocity(negated, queries),
        -velocity(sample.flow, queries),
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        pressure(negated, queries),
        pressure(sample.flow, queries),
        rtol=0.0,
        atol=1.0e-14,
    )
    drive = sample.domain.boundaries["dirichlet"].cell_data["boundary_velocity"]
    assert torch.isfinite(drive).all()


def test_eigenvalue_guard_and_series_validity() -> None:
    """The safe band is enforced, with the physics documented in the error.

    Any coupling at or above j_{0,1} = 2.404826 must be rejected (the
    interior Helmholtz solution is not determined by boundary data at
    Dirichlet eigenvalues), and mode orders beyond the documented series
    validity range must be rejected too.
    """

    assert abs(FIRST_DIRICHLET_EIGENVALUE - 2.4048255576957727) < 1.0e-12
    for name, spec in SPLITS.items():
        low, high = spec["coupling_range"]
        assert 0.0 < low < high < FIRST_DIRICHLET_EIGENVALUE, name
    with pytest.raises(ValueError, match="eigenvalue"):
        build_euler_rotational_sample(
            0, coupling_range=(2.41, 2.5), modes=(0, 1)
        )
    with pytest.raises(ValueError, match="validity"):
        build_euler_rotational_sample(
            0, coupling_range=(0.5, 1.0), modes=(11, 12)
        )


def _scaled_drive_domain(domain: DomainMesh, alpha: float) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"]
    scaled = Mesh(
        points=boundary.points,
        cells=boundary.cells,
        cell_data={
            "boundary_velocity": alpha * boundary.cell_data["boundary_velocity"]
        },
    )
    return DomainMesh(
        interior=Mesh(points=domain.interior.points),
        boundaries={"dirichlet": scaled},
        global_data=domain.global_data,
    )


def _negated_drive_domain(domain: DomainMesh) -> DomainMesh:
    return _scaled_drive_domain(domain, -1.0)


def test_linear_arm_is_exactly_drive_odd_the_wall_is_structural() -> None:
    """The pre-registered wall control really is exactly odd in the drive.

    In float64, negating the boundary-velocity drive negates every
    prediction of the ``field_mode='linear'`` arm to roundoff (its pressure
    prediction is therefore an odd fit of an even target: the wall).  The
    nonlinear arm must NOT be odd -- its even (drive-quadratic) response
    component is exactly what makes the pressure representable.
    """

    sample = build_euler_rotational_sample(
        5, dtype=torch.float64, **SPLITS["in_distribution"]
    )
    negated_domain = _negated_drive_domain(sample.domain)

    torch.manual_seed(0)
    linear = _build_model("mt_singpair_linear").double().eval()
    with torch.no_grad():
        plus = _predictions(linear, sample.domain)
        minus = _predictions(linear, negated_domain)
    for field in OUTPUT_FIELD_RANKS:
        odd_violation = float((plus[field] + minus[field]).abs().max())
        scale = float(plus[field].abs().max())
        assert odd_violation <= 1.0e-10 * max(scale, 1.0), (field, odd_violation)

    torch.manual_seed(0)
    nonlinear = _build_model("mt_singpair_nl").double().eval()
    with torch.no_grad():
        plus = _predictions(nonlinear, sample.domain)
        minus = _predictions(nonlinear, negated_domain)
    even_component = float((plus["pressure"] + minus["pressure"]).abs().max())
    assert even_component > 1.0e-8


def test_model_arms_multi_field_construction_forward_backward() -> None:
    """Every arm builds, emits both fields, and backpropagates finitely.

    All transformer arms carry the flipped one-head reference configuration
    of iteration 32 (heads 1, ranks 48/16); the pseudo arm's pseudo-sector
    parameters must receive gradients through the multi-field loss.
    """

    sample = build_euler_rotational_sample(5, **SPLITS["in_distribution"])
    expected_shapes = {"velocity": (128, 2), "pressure": (128,)}
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
            assert model.heads == 1, name
            assert model.scalar_rank == 48 and model.vector_rank == 16, name
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
    # The quadratic (declared-degree) arm is the linear machinery plus one
    # bilinear composition: strictly between linear and nonlinear.
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
            "dirichlet": {"operator": {}, "drive": {"boundary_velocity": 1}}
        }
        assert model.global_field_ranks == {
            "operator": {"vorticity_coupling": 0},
            "drive": {},
        }
    assert linear.field_mode == "linear"
    assert nonlinear.field_mode == "zero_preserving_nonlinear"
    assert pseudo.field_mode == "zero_preserving_nonlinear"
    assert quadratic.field_mode == "quadratic"
    assert quadratic_pseudo.field_mode == "quadratic"
    assert quadratic.quadratic_read_in is not None
    assert linear.quadratic_read_in is None
    assert nonlinear.quadratic_read_in is None
    assert linear.drive_pseudo_dim == 0
    assert nonlinear.drive_pseudo_dim == 0
    assert pseudo.drive_pseudo_dim == 8
    assert quadratic.drive_pseudo_dim == 0
    assert quadratic_pseudo.drive_pseudo_dim == 8

    assert isinstance(_build_model("boundary_mean"), DriveBoundaryMean)
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

    Scaling the boundary-velocity drive by alpha at fixed geometry and
    fixed coupling must make every q2 prediction entry an exact degree-<=2
    polynomial in alpha at float64 machine precision, with random weights.
    This is the structural negation of the iteration-34 diagnosis: the
    near-eigenvalue detonation rode drive monomials of implicit degree ~21,
    which this mode removes by construction rather than by training.
    """

    sample = build_euler_rotational_sample(
        5, dtype=torch.float64, **SPLITS["in_distribution"]
    )
    torch.manual_seed(1)
    model = _build_model("mt_singpair_q2").double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.3, 0.3)

    @torch.no_grad()
    def output(alpha: float) -> torch.Tensor:
        predictions = _predictions(model, _scaled_drive_domain(sample.domain, alpha))
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


def test_boundary_mean_floor_is_the_drive_mean_and_pressure_pins() -> None:
    """The parameter-free floor predicts the boundary-measure drive mean.

    Velocity: the measure-weighted mean of the cell drive at every query
    (the only O(2)-equivariant constant the data supplies).  Pressure: no
    boundary drive exists, so the floor is zero and its relative L2 is
    exactly 1.0 -- the declared no-response floor for the wall column.
    """

    sample = build_euler_rotational_sample(
        9, dtype=torch.float64, **SPLITS["in_distribution"]
    )
    model = DriveBoundaryMean()
    predictions = _predictions(model, sample.domain)
    boundary = sample.domain.boundaries["dirichlet"]
    weights = boundary.cell_areas
    expected = (
        weights[:, None] * boundary.cell_data["boundary_velocity"]
    ).sum(dim=0) / weights.sum()
    torch.testing.assert_close(
        predictions["velocity"],
        expected[None, :].repeat(sample.domain.interior.n_points, 1),
        rtol=0.0,
        atol=0.0,
    )
    assert float(predictions["pressure"].abs().max()) == 0.0
    error = float(
        torch.linalg.vector_norm(
            predictions["pressure"] - sample.targets["pressure"]
        )
        / torch.linalg.vector_norm(sample.targets["pressure"])
    )
    assert abs(error - 1.0) < 1.0e-12
    assert math.isfinite(
        _combined_relative_l2(predictions, sample.targets)
    )


def test_generator_determinism() -> None:
    """The same seed reproduces both targets and the drive bit-for-bit."""

    first = build_euler_rotational_sample(321, **SPLITS["in_distribution"])
    second = build_euler_rotational_sample(321, **SPLITS["in_distribution"])
    for field in OUTPUT_FIELD_RANKS:
        assert torch.equal(first.targets[field], second.targets[field])
    assert torch.equal(first.domain.interior.points, second.domain.interior.points)
    assert torch.equal(
        first.domain.boundaries["dirichlet"].cell_data["boundary_velocity"],
        second.domain.boundaries["dirichlet"].cell_data["boundary_velocity"],
    )


def test_driver_smoke_produces_finite_multi_field_json(tmp_path: Path) -> None:
    """A zero-step CPU floor run writes the per-field report shape.

    Split keys must include ``<split>``, ``<split>/velocity``, and
    ``<split>/pressure`` for all four splits; the floor's pressure column
    must be exactly 1.0; its momentum and divergence residuals must be
    exactly zero (a constant velocity with zero pressure is an exact steady
    Euler solution); and its Helmholtz ``pde_residual`` must equal the
    calibration value: the mean of the two fidelity cases' coupling^2
    (``lap u = 0`` for a constant, leaving exactly ``c~^2 u``).
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
        (tmp_path / "euler_rotational_boundary_mean_seed3.json").read_text()
    )
    expected_keys = set()
    for split in SPLITS:
        expected_keys |= {split, f"{split}/velocity", f"{split}/pressure"}
    for payload in (report, on_disk):
        assert payload["parameters"] == 0
        assert payload["family"] == "euler_rotational"
        assert set(payload["splits"]) == expected_keys
        for key, value in payload["splits"].items():
            assert math.isfinite(value), key
            if key.endswith("/pressure"):
                assert abs(value - 1.0) < 1.0e-6, key
        assert "pre_registered_prediction" in payload["design_notes"]
        assert "well_posedness_note" in payload["design_notes"]
        fidelity = payload["fidelity"]
        assert set(fidelity["pde_residual"]) == set(SPLITS)
        assert set(fidelity["momentum_residual"]) == set(SPLITS)
        assert set(fidelity["divergence_residual"]) == set(SPLITS)
        assert all(
            abs(v) < 1.0e-9 for v in fidelity["momentum_residual"].values()
        )
        assert all(
            abs(v) < 1.0e-9 for v in fidelity["divergence_residual"].values()
        )
        assert fidelity["max_principle_violation"] is None
    # Helmholtz calibration of the constant-velocity floor: exactly the
    # mean coupling^2 of the two fidelity cases per split.
    for split_index, name in enumerate(sorted(SPLITS)):
        couplings = [
            build_euler_rotational_sample(
                83_000_019 + case, n_query=4, **SPLITS[name]
            ).flow.coupling
            for case in range(2)
        ]
        expected = sum(c**2 for c in couplings) / 2.0
        measured = report["fidelity"]["pde_residual"][name]
        assert abs(measured - expected) < 1.0e-6 * expected, (name, measured)


@pytest.mark.parametrize(
    "model_name",
    [
        "mt_singpair_linear",
        "mt_singpair_nl",
        "mt_singpair_nl_pseudo",
        "mt_singpair_q2",
    ],
)
def test_twenty_step_smoke_per_arm(model_name: str, tmp_path: Path) -> None:
    """Twenty CPU optimizer steps of each transformer arm, end to end.

    Exercises the full loop -- multi-field loss, backward, clipping,
    validation bookkeeping, best-state restore, evaluation, the fidelity
    block, and the JSON report -- and requires every reported number to be
    finite.  Twenty steps makes no accuracy claim (the nonlinear arms carry
    a known early-training drive-quadratic transient, as in iteration 25).
    """

    report = run_experiment(
        model_name=model_name,
        steps=20,
        seed=3,
        device="cpu",
        output_dir=str(tmp_path),
        eval_cases=1,
    )
    assert report["parameters"] > 0
    assert len(report["history"]) == 1
    assert report["history"][0]["step"] == 20
    assert math.isfinite(
        report["history"][0]["validation_mean_per_field_relative_l2"]
    )
    for value in report["splits"].values():
        assert math.isfinite(value)
    for block in ("pde_residual", "momentum_residual", "divergence_residual"):
        for value in report["fidelity"][block].values():
            assert math.isfinite(value)
