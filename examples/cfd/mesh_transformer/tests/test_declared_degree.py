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

"""Contracts for the iteration-35 declared-degree acceptance study.

The study's verdicts are only as good as its instruments, so the
instruments are tested without any trained checkpoint: the structural
degree test must PASS on a random-weight q2 arm (the contract is
weight-independent) and FAIL on a random-weight nonlinear arm (the test
discriminates), the scaled-domain builders must scale the whole drive and
nothing else, and the archived nonlinear comparators must resolve to the
checked-in per-seed numbers the pre-registration quotes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from declared_degree import (  # noqa: E402
    RULES,
    eb_structural_domains,
    er_structural_domains,
    id_degradation_verdict,
    load_comparators,
    structural_degree_test,
)
from euler_bernoulli import _build_model as _build_eb_model  # noqa: E402
from euler_rotational import _build_model as _build_er_model  # noqa: E402


def _randomized(model: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.3, 0.3)
    return model.double().eval()


def test_structural_degree_test_passes_on_q2_for_any_weights() -> None:
    """The acceptance instrument passes the q2 arms with random weights.

    The declared-degree contract is structural, so no training may be
    needed: both suites' q2 arms must fit the exact degree-<=2 polynomial
    at the pre-registered float64 residual, including the extrapolated and
    sign-flipped probe alphas, with exactly zero output at zero drive.
    """

    device = torch.device("cpu")
    torch.manual_seed(11)
    eb = structural_degree_test(
        _randomized(_build_eb_model("mt_singpair_q2")),
        eb_structural_domains(424_243, device),
    )
    torch.manual_seed(11)
    er = structural_degree_test(
        _randomized(_build_er_model("mt_singpair_q2")),
        er_structural_domains(424_243, device),
    )
    for result in (eb, er):
        assert result["passes"], result
        assert result["worst_relative_residual"] < RULES["structural_fit_residual_max"]
        assert result["relative_residuals"]["zero_drive_max_output"] == 0.0


def test_structural_degree_test_rejects_the_nonlinear_mode() -> None:
    """The instrument discriminates: the nl arm fails the identical probe."""

    device = torch.device("cpu")
    torch.manual_seed(11)
    result = structural_degree_test(
        _randomized(_build_er_model("mt_singpair_nl")),
        er_structural_domains(424_243, device),
    )
    assert not result["passes"], result
    assert result["worst_relative_residual"] > 1.0e-3


def test_comparators_resolve_to_the_archived_preregistration_numbers() -> None:
    """The rule-4 comparators are the checked-in archive values.

    The pre-registration quotes euler_rotational nl pressure ID 0.181 and
    nl+0o 0.103; the loader must reproduce those (3-seed means of the
    iteration-33 archive) and expose per-seed values for the sd tolerance.
    """

    comparators = load_comparators(EXAMPLE_DIR)
    er = comparators["euler_rotational"]
    assert abs(er["mt_singpair_nl"]["in_distribution/pressure"]["mean"] - 0.181) < 5e-4
    assert (
        abs(er["mt_singpair_nl_pseudo"]["in_distribution/pressure"]["mean"] - 0.103)
        < 5e-4
    )
    assert len(er["mt_singpair_nl"]["in_distribution/pressure"]["per_seed"]) == 3
    eb = comparators["euler_bernoulli"]
    assert len(eb["mt_singpair_nl"]["in_distribution/pressure"]["per_seed"]) == 5

    # The rule keys to BOTH nonlinear arms, so a hypothetical q2 result must
    # sit near the better (pseudo) arm's pressure to pass outright.
    verdict = id_degradation_verdict(
        {
            "in_distribution/velocity": [0.05, 0.05, 0.05],
            "in_distribution/pressure": [0.10, 0.11, 0.10],
        },
        er,
    )
    assert verdict["passes"]
    verdict = id_degradation_verdict(
        {
            "in_distribution/velocity": [0.05, 0.05, 0.05],
            "in_distribution/pressure": [5.0, 5.0, 5.0],
        },
        er,
    )
    assert not verdict["passes"]
