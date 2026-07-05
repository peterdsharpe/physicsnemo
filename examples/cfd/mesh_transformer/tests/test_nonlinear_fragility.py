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

"""Contracts for the nonlinear-fragility strong-inference analysis helpers.

The verdicts of ``studies/nonlinear_fragility.py`` follow pre-registered
decision rules; these tests pin the rule implementations (the effective-degree
power-law fit, the amplification factor, the three verdict functions, the
exact-homogeneity claims behind the drive-scaling probes, and the checkpoint
guard) on inputs whose correct classification is known by construction, so a
silent change to a rule cannot masquerade as a change in scientific
conclusion.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from euler_bernoulli import (  # noqa: E402
    bernoulli_pressure,
    build_euler_bernoulli_sample,
)
from euler_rotational import (  # noqa: E402
    SPLITS as ER_SPLITS,
)
from euler_rotational import (  # noqa: E402
    RotationalFlow,
    build_euler_rotational_sample,
    pressure,
    velocity,
)
from euler_rotational import _build_model as _build_er_model  # noqa: E402
from nonlinear_fragility import (  # noqa: E402
    ER_ARM,
    ER_FAMILY,
    N2_CONDITIONED_TRACES,
    N2_OUTPUT_TRACES,
    RULES,
    _renormalized,
    _scaled_drive_domain,
    amplification_factor,
    fit_power_law,
    load_er_checkpoint,
    verdict_n1,
    verdict_n2,
    verdict_n3,
)
from potential_flow import disturbance_velocity  # noqa: E402

_ALPHAS = [0.1 * (4.0 / 0.1) ** (i / 24) for i in range(25)]


def test_fit_power_law_recovers_exact_degrees() -> None:
    """The fit returns the exact degree of a homogeneous response."""

    for degree in (1.0, 2.0, 5.0):
        magnitudes = [0.37 * a**degree for a in _ALPHAS]
        for band in ((0.5, 1.5), (2.0, 4.0)):
            fitted = fit_power_law(_ALPHAS, magnitudes, band)
            assert fitted is not None
            assert abs(fitted - degree) < 1.0e-9
    # A mixed polynomial's slope approaches its top degree off-range: the
    # exact effect N1 posits (small high-degree component dominating).
    wide = [0.1 * (8.0 / 0.1) ** (i / 39) for i in range(40)]
    mixed = [a + 0.01 * a**5 for a in wide]
    low = fit_power_law(wide, mixed, (0.1, 0.3))
    high = fit_power_law(wide, mixed, (4.0, 8.0))
    assert abs(low - 1.0) < 0.05
    assert high > 3.5
    assert fit_power_law(_ALPHAS[:2], [1.0, 2.0], (0.0, 10.0)) is None


def test_amplification_factor_matches_the_bessel_zero() -> None:
    """A(c) grows monotonically toward j_{0,1} and hits ~77 at 2.38."""

    values = [amplification_factor(c) for c in (1.0, 1.8, 2.25, 2.30, 2.38)]
    assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))
    assert abs(values[0] - 1.0 / 0.7651976865579666) < 1.0e-9  # 1/J0(1)
    assert 70.0 < values[-1] < 85.0  # the documented ~77x at the band edge


def _slopes(velocity_pair: tuple, pressure_pair: tuple) -> dict:
    return {
        "velocity": {
            "target_degree": 1.0,
            "on_range": velocity_pair[0],
            "extrapolated": velocity_pair[1],
        },
        "pressure": {
            "target_degree": 2.0,
            "on_range": pressure_pair[0],
            "extrapolated": pressure_pair[1],
        },
    }


def test_verdict_n1_rules() -> None:
    """Degree >= target + 1 supports; degree == target everywhere excludes."""

    assert verdict_n1(_slopes((1.0, 1.1), (2.0, 2.1)))["verdict"] == "excluded"
    # One field a full degree above its target supports, regardless of the
    # other field's fidelity.
    assert verdict_n1(_slopes((1.0, 1.0), (2.0, 3.2)))["verdict"] == "supported"
    assert verdict_n1(_slopes((1.0, 2.4), (2.0, 2.0)))["verdict"] == "supported"
    # A drift beyond the tolerance but short of a full degree is ambiguous.
    assert verdict_n1(_slopes((1.0, 1.7), (2.0, 2.0)))["verdict"] == "ambiguous"
    # An on-range misfit blocks exclusion even with a matched off-range slope.
    assert verdict_n1(_slopes((0.2, 1.0), (2.0, 2.0)))["verdict"] == "ambiguous"


def test_verdict_n2_rules() -> None:
    """Output blowup at pinned drive supports; all-smooth traces exclude."""

    smooth = {name: 1.2 for name in N2_CONDITIONED_TRACES + N2_OUTPUT_TRACES}
    assert verdict_n2([smooth])["verdict"] == "excluded"

    detonating = dict(smooth)
    detonating["output_velocity_rms"] = 1.0e6
    detonating["kernel_coefficients_max"] = 4.0e3
    supported = verdict_n2([smooth, detonating])
    assert supported["verdict"] == "supported"
    assert supported["worst_output_trace"] == "case1:output_velocity_rms"
    assert supported["worst_conditioned_trace"] == "case1:kernel_coefficients_max"

    # An internal trace above the exclusion bar without an output blowup is
    # not smooth continuation, but not support either.
    internal_only = dict(smooth)
    internal_only["query_operator_scalar_rms"] = 10.0
    assert verdict_n2([internal_only])["verdict"] == "ambiguous"


def test_verdict_n3_rules() -> None:
    """Ordinary renormalized error supports; persistent blowup excludes."""

    assert verdict_n3(1.0e6, 0.3, 2.0)["verdict"] == "supported"
    assert verdict_n3(1.0e6, 1.0e4, 2.0)["verdict"] == "excluded"
    assert verdict_n3(1.0e6, 10.0, 2.0)["verdict"] == "ambiguous"
    # Support additionally requires the unrenormalized blowup to be present.
    assert verdict_n3(50.0, 0.3, 2.0)["verdict"] == "ambiguous"
    assert verdict_n3(1.0e6, 0.3, None)["verdict"] == "supported"
    assert RULES["n3_renormalized_ordinary"] < RULES["n3_renormalized_exclude"]


def test_scaled_drive_domain_matches_exact_homogeneity() -> None:
    """The N1 probe's exact-response claim: velocity degree 1, pressure 2.

    Scaling the joint drive ``(U, Gamma)`` by alpha scales the exact
    disturbance velocity by alpha (joint linearity) and the exact Bernoulli
    pressure by alpha**2 -- verified through the certified label chain, not
    asserted.
    """

    sample = build_euler_bernoulli_sample(21, dtype=torch.float64)
    alpha = 3.0
    scaled = _scaled_drive_domain(sample.domain, alpha)
    for key in ("freestream_velocity", "circulation"):
        assert torch.allclose(
            scaled.global_data[key], alpha * sample.domain.global_data[key]
        )
    assert scaled.global_data["reference_length"] is not None
    assert torch.equal(
        scaled.boundaries["dirichlet"].points,
        sample.domain.boundaries["dirichlet"].points,
    )

    base = disturbance_velocity(
        sample.body,
        sample.canonical_freestream,
        sample.circulation,
        sample.query_preimages,
    )
    scaled_disturbance = disturbance_velocity(
        sample.body,
        alpha * sample.canonical_freestream,
        alpha * sample.circulation,
        sample.query_preimages,
    )
    assert torch.allclose(scaled_disturbance, alpha * base, rtol=1e-12, atol=1e-12)
    base_pressure = bernoulli_pressure(sample.freestream, base)
    scaled_pressure = bernoulli_pressure(
        alpha * sample.freestream, scaled_disturbance
    )
    assert torch.allclose(
        scaled_pressure, (alpha**2) * base_pressure, rtol=1e-12, atol=1e-12
    )


def test_renormalized_rotational_sample_is_exact() -> None:
    """The N3 renormalization is a bitwise-exact solution rescaling.

    Scaling the mode coefficients by the factor scales the exact velocity by
    the factor and the exact pressure by its square (both fields are
    homogeneous in the coefficients), so ``_renormalized`` -- which scales
    the drive and targets directly -- must agree with the exact machinery
    run on the rescaled flow.
    """

    coupling = 2.34
    sample = build_euler_rotational_sample(
        13,
        coupling_range=(coupling, coupling),
        modes=ER_SPLITS["near_eigenvalue"]["modes"],
        n_boundary=16,
        n_query=8,
        dtype=torch.float64,
    )
    factor = 1.0 / amplification_factor(sample.flow.coupling)
    assert factor < 0.05  # the band genuinely amplifies (~77x at 2.38)
    domain, targets = _renormalized(sample, factor)
    rescaled_flow = RotationalFlow(
        center=sample.flow.center,
        radius=sample.flow.radius,
        coupling=sample.flow.coupling,
        coefficients={
            m: c * factor for m, c in sample.flow.coefficients.items()
        },
    )
    points = sample.domain.interior.points
    assert torch.allclose(
        targets["velocity"], velocity(rescaled_flow, points), rtol=1e-12, atol=1e-14
    )
    assert torch.allclose(
        targets["pressure"], pressure(rescaled_flow, points), rtol=1e-12, atol=1e-14
    )
    assert torch.allclose(
        domain.global_data["vorticity_coupling"],
        sample.domain.global_data["vorticity_coupling"],
    )


def test_load_er_checkpoint_guards_arm_and_family(tmp_path: Path) -> None:
    """The loader refuses a payload from the wrong arm or family."""

    torch.manual_seed(11)
    source = _build_er_model(ER_ARM)
    good = tmp_path / "good.pt"
    torch.save(
        {
            "model": ER_ARM,
            "family": ER_FAMILY,
            "seed": 17,
            "steps": 3000,
            "state_dict": source.state_dict(),
        },
        good,
    )
    model, metadata = load_er_checkpoint(good, torch.device("cpu"))
    assert metadata["seed"] == 17
    assert not model.training

    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "model": ER_ARM,
            "family": "euler_bernoulli",
            "state_dict": source.state_dict(),
        },
        bad,
    )
    try:
        load_er_checkpoint(bad, torch.device("cpu"))
    except ValueError as error:
        assert "euler_bernoulli" in str(error)
    else:
        raise AssertionError("wrong-family checkpoint must be refused")
