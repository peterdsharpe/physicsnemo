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

"""Scale-gauge knob contracts and the convention-drift demonstration.

The MeshTransformer's scale gauge is intrinsic by default (measure-weighted
RMS boundary radius); the benchmark keeps the explicit ``reference_length``
gauge as its historical default via ``make_model(..., gauge="explicit")``.

The demonstration here is the payoff of the intrinsic gauge: a briefly
trained explicit-gauge model evaluated under a drifted reference-length
convention (the declared ``reference_length`` scaled 3x at evaluation time
only) degrades severely, while the intrinsic-gauge model has no such input
to corrupt -- the identical corruption is a bitwise no-op.  This is a
*demonstration* of the failure mode and its structural elimination, not a
benchmark; accuracy claims live in the 3000-step study protocol.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import train as training_benchmark  # noqa: E402
from metrics import weighted_relative_l2  # noqa: E402
from train import (  # noqa: E402
    TRAIN_SPLIT,
    RunConfig,
    make_case,
    make_model,
    train_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
FLAGSHIP = "mesh_transformer_kernel_singonly"
DEMO_STEPS = 200


# ---------------------------------------------------------------------------
# Knob wiring (no training).
# ---------------------------------------------------------------------------


def test_gauge_knob_wires_reference_length_key():
    """The gauge argument maps onto MeshTransformer's reference_length_key."""
    explicit = make_model(FLAGSHIP, "tiny")
    assert explicit.reference_length_key == "reference_length"
    spelled_out = make_model(FLAGSHIP, "tiny", "explicit")
    assert spelled_out.reference_length_key == "reference_length"
    intrinsic = make_model(FLAGSHIP, "tiny", "intrinsic")
    assert intrinsic.reference_length_key is None
    lifted = make_model("lifted_mesh_transformer", "tiny", "intrinsic")
    assert lifted.residual_model.reference_length_key is None


def test_gauge_default_is_bitwise_noop():
    """Explicitly passing the default gauge must not change anything.

    Mirrors the model-side knob regressions: same seed gives the same
    parameter tensors in the same order under ``gauge="explicit"`` spelled
    out versus omitted.
    """
    torch.manual_seed(11)
    reference = make_model(FLAGSHIP, "tiny")
    torch.manual_seed(11)
    explicit = make_model(FLAGSHIP, "tiny", "explicit")
    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name, expected in reference_state.items():
        torch.testing.assert_close(explicit_state[name], expected, rtol=0.0, atol=0.0)


def test_gauge_knob_rejects_misuse():
    """Baselines reject a non-default gauge; unknown gauges are errors."""
    with pytest.raises(ValueError, match="MeshTransformer-family"):
        make_model("boundary_mean", "tiny", "intrinsic")
    with pytest.raises(ValueError, match="unknown gauge"):
        make_model(FLAGSHIP, "tiny", "implicit")


def test_checkpoint_gauge_mismatch_is_rejected(tmp_path, monkeypatch):
    """A checkpoint trained under one gauge cannot silently evaluate under
    another: the gauge adds no parameters, so nothing in the state dict
    would catch the reinterpretation."""
    torch.manual_seed(5)
    model = make_model("mesh_transformer", "tiny")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": "mesh_transformer",
            "capacity": "tiny",
            "state_dict": model.state_dict(),
            "run_config": {"gauge": "explicit"},
            "source": {"relevant_source_sha256": "irrelevant"},
        },
        checkpoint_path,
    )
    args = SimpleNamespace(
        model="mesh_transformer",
        capacity="tiny",
        gauge="intrinsic",
        steps=0,
        cases_per_step=1,
        train_boundary_points=4,
        train_query_points=4,
        learning_rate=3.0e-4,
        weight_decay=1.0e-6,
        seed=17,
        validation_seed=19,
        evaluation_seed=23,
        evaluation_cases=1,
        evaluation_boundary_points=4,
        evaluation_query_points=4,
        harmonic_cases=0,
        device="cpu",
        dtype="float32",
        matmul_precision="highest",
        output_dir=tmp_path / "output",
        evaluate_only=True,
        checkpoint=checkpoint_path,
    )
    monkeypatch.setattr(training_benchmark, "_parse_args", lambda: args)
    with pytest.raises(ValueError, match="checkpoint gauge"):
        training_benchmark.main()


# ---------------------------------------------------------------------------
# Convention-drift demonstration (brief training; not a benchmark).
# ---------------------------------------------------------------------------


def _train_flagship(gauge: str) -> torch.nn.Module:
    torch.manual_seed(17)
    model = make_model(FLAGSHIP, "tiny", gauge).to(device=DEVICE, dtype=DTYPE)
    config = RunConfig(
        model=FLAGSHIP,
        capacity="tiny",
        gauge=gauge,
        steps=DEMO_STEPS,
        cases_per_step=1,
        train_boundary_points=48,
        train_query_points=64,
        learning_rate=1.0e-3,
        seed=17,
        report_every=DEMO_STEPS,
        validation_every=DEMO_STEPS,
        validation_cases=2,
        evaluation_cases=4,
        evaluation_boundary_points=48,
        evaluation_query_points=64,
        harmonic_cases=0,
    )
    train_model(model, config, device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


@torch.no_grad()
def _evaluate_in_distribution(
    model: torch.nn.Module,
    *,
    reference_length_drift: float | None = None,
    n_cases: int = 12,
    seed: int = 90_001,
) -> tuple[float, list[torch.Tensor]]:
    """Mean ID relative L2, optionally under a drifted length convention.

    ``reference_length_drift`` rescales only the *declared*
    ``reference_length`` global (the evaluation-time convention), never the
    geometry or the targets: this is exactly the convention-drift failure
    mode, not a physical rescaling.
    """
    errors: list[float] = []
    predictions: list[torch.Tensor] = []
    for index in range(n_cases):
        sample = make_case(
            TRAIN_SPLIT,
            seed=seed,
            case_index=index,
            n_boundary=64,
            n_query=128,
            device=DEVICE,
            dtype=DTYPE,
        )
        domain = sample.domain
        if reference_length_drift is not None:
            domain.global_data["reference_length"] = (
                reference_length_drift * domain.global_data["reference_length"]
            )
        prediction = model(domain).point_data["potential"]
        predictions.append(prediction)
        errors.append(
            float(weighted_relative_l2(prediction, sample.target, sample.area_jacobian))
        )
    return sum(errors) / len(errors), predictions


@pytest.fixture(scope="module")
def trained_gauge_pair():
    """One briefly trained flagship-arm model per gauge (shared per module)."""
    return {gauge: _train_flagship(gauge) for gauge in ("explicit", "intrinsic")}


def test_convention_drift_demonstration(trained_gauge_pair):
    """Drifting L_ref 3x at eval wrecks the explicit-gauge model only.

    Measured at this demo scale (200 steps, tiny capacity, seed 17, CUDA
    fp32): explicit clean ~0.092 -> drifted ~0.43 (4.6x degradation);
    intrinsic is bitwise unchanged because no reference-length input exists.
    Thresholds are set loosely so the demonstration is robust across
    devices, while remaining orders of magnitude away from the intrinsic
    model's exact immunity.
    """
    explicit = trained_gauge_pair["explicit"]
    intrinsic = trained_gauge_pair["intrinsic"]

    explicit_clean, _ = _evaluate_in_distribution(explicit)
    explicit_drift, _ = _evaluate_in_distribution(explicit, reference_length_drift=3.0)
    intrinsic_clean, clean_predictions = _evaluate_in_distribution(intrinsic)
    intrinsic_drift, drift_predictions = _evaluate_in_distribution(
        intrinsic, reference_length_drift=3.0
    )
    print(
        json.dumps(
            {
                "explicit_clean": explicit_clean,
                "explicit_drift": explicit_drift,
                "explicit_degradation": explicit_drift / explicit_clean,
                "intrinsic_clean": intrinsic_clean,
                "intrinsic_drift": intrinsic_drift,
            }
        )
    )

    # The explicit model trained (well below the boundary-mean ~1.0 level)
    # and the drifted convention costs it a large factor.
    assert explicit_clean < 0.25
    assert explicit_drift > 2.0 * explicit_clean

    # The intrinsic model is immune by construction: bitwise, per case.
    for clean, drifted in zip(clean_predictions, drift_predictions, strict=True):
        torch.testing.assert_close(drifted, clean, rtol=0.0, atol=0.0)


def test_gauge_neutrality_smoke(trained_gauge_pair):
    """Intrinsic-gauge accuracy tracks the explicit gauge at demo scale.

    A one-seed, 200-step smoke check (measured 0.0921 vs 0.0919 on CUDA
    fp32); the 3000-step neutrality claim runs under the study protocol.
    """
    explicit_clean, _ = _evaluate_in_distribution(trained_gauge_pair["explicit"])
    intrinsic_clean, _ = _evaluate_in_distribution(trained_gauge_pair["intrinsic"])
    assert intrinsic_clean < 1.5 * explicit_clean
