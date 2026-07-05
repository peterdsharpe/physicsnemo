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

"""Contracts for the far-field strong-inference analysis helpers.

The verdicts of ``studies/farfield_inference.py`` follow pre-registered
decision rules; these tests pin the rule implementations (exponent fits,
trace metrics, and the three verdict functions) on synthetic inputs whose
correct classification is known by construction, so a silent change to a
rule cannot masquerade as a change in scientific conclusion.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from farfield_inference import (  # noqa: E402
    ARM,
    BANDS,
    FAMILY,
    fit_band_exponent,
    load_checkpoint,
    trace_metrics,
    verdict_m1,
    verdict_m2,
    verdict_m3,
)
from potential_flow import _build_model  # noqa: E402

_RADII = [1.05 * (12.0 / 1.05) ** (i / 63) for i in range(64)]


def test_fit_band_exponent_recovers_power_laws() -> None:
    """The band fit returns the exact slope of a pure power law."""

    for exponent in (-2.0, -1.0, 0.0, 1.0):
        magnitudes = [3.7 * r**exponent for r in _RADII]
        for band in BANDS:
            fitted = fit_band_exponent(_RADII, magnitudes, band)
            assert fitted is not None
            assert abs(fitted - exponent) < 1.0e-9
    # log r is not a power law: its local exponent 1/log(r) must land between
    # the band endpoints' values (a flattening tail, never a growth law).
    logs = [math.log(r) for r in _RADII]
    fitted = fit_band_exponent(_RADII, logs, (8.0, 12.0))
    assert 1.0 / math.log(12.0) < fitted < 1.0 / math.log(8.0)
    assert fit_band_exponent(_RADII[:4], [1.0] * 4, (8.0, 12.0)) is None


def test_trace_metrics_classify_smooth_divergent_and_oscillating() -> None:
    """The M3 metrics separate smooth continuation from divergence/oscillation."""

    smooth = trace_metrics(_RADII, [1.0 + 0.01 * math.log(r) for r in _RADII])
    assert smooth["far_near_ratio"] < 1.1
    assert smooth["oscillations"] == 0

    divergent = trace_metrics(
        _RADII, [1.0 if r <= 4.0 else (r / 4.0) ** 3 for r in _RADII]
    )
    assert divergent["far_near_ratio"] > 3.0

    oscillating = trace_metrics(
        _RADII,
        [
            1.0 if r <= 4.0 else 1.0 + math.sin(6.0 * math.pi * math.log(r / 4.0))
            for r in _RADII
        ],
    )
    assert oscillating["oscillations"] >= 2
    assert oscillating["far_near_ratio"] <= 3.0


def test_verdict_m1_rules() -> None:
    """Sufficiency, exclusion, and pending branches of the M1 rule."""

    baseline = {"splits": {"farfield_queries": 0.70, "in_distribution": 0.03}}
    rescued = {"splits": {"farfield_queries": 0.05, "in_distribution": 0.03}}
    unmoved = {"splits": {"farfield_queries": 0.65, "in_distribution": 0.03}}
    partial = {"splits": {"farfield_queries": 0.20, "in_distribution": 0.03}}
    assert verdict_m1(baseline, rescued)["verdict"].startswith("supported")
    assert verdict_m1(baseline, unmoved)["verdict"].startswith("excluded")
    assert verdict_m1(baseline, partial)["verdict"] == "ambiguous"
    assert verdict_m1(None, rescued)["verdict"] == "pending"


def test_verdict_m2_rules() -> None:
    """Support requires a far-only departure; exclusion requires far fidelity."""

    labels = [f"[{low}, {high})" for low, high in BANDS]

    def deltas(near: float, far: float) -> dict:
        return {
            label: (near if index < 3 else far) for index, label in enumerate(labels)
        }

    assert verdict_m2(deltas(0.1, 1.5))["verdict"] == "supported"
    assert verdict_m2(deltas(0.1, 0.1))["verdict"] == "excluded"
    # A large departure AWAY from the member tails (super-decay) excludes:
    # the member basis cannot out-decay the exact law, so the pathology
    # must live in the coefficients, not the members.
    collapse = verdict_m2(deltas(0.1, -24.0))
    assert collapse["verdict"] == "excluded"
    assert "opposite the member tails" in collapse["reason"]
    # A near-band misfit invalidates the probe: ambiguous, not supported.
    assert verdict_m2(deltas(1.0, 1.5))["verdict"] == "ambiguous"
    assert verdict_m2(deltas(0.1, 0.4))["verdict"] == "ambiguous"


def test_verdict_m3_rules() -> None:
    """Any divergent or oscillating trace supports; all-smooth excludes."""

    smooth = {"a": {"far_near_ratio": 1.1, "oscillations": 0}}
    divergent = {"a": {"far_near_ratio": 5.0, "oscillations": 0}}
    oscillating = {"a": {"far_near_ratio": 1.0, "oscillations": 3}}
    borderline = {"a": {"far_near_ratio": 2.0, "oscillations": 0}}
    assert verdict_m3([smooth])["verdict"] == "excluded"
    assert verdict_m3([smooth, divergent])["verdict"] == "supported"
    assert verdict_m3([oscillating])["verdict"] == "supported"
    assert verdict_m3([borderline])["verdict"] == "ambiguous"


def test_load_checkpoint_honors_the_bounded_gates_flag(tmp_path: Path) -> None:
    """The analysis rebuilds the architecture the checkpoint was trained as.

    Gate-fixed checkpoints record ``bounded_gates``; ``load_checkpoint``
    must thread it into the rebuild (the state dicts are interchangeable --
    the knob adds no parameters -- so only this flag selects the gate
    parameterization).  Pre-fix checkpoints carry no flag and must rebuild
    the historical raw-gate arm.
    """

    for flag in (False, True):
        torch.manual_seed(11)
        source = _build_model(ARM, FAMILY, bounded_gates=flag)
        path = tmp_path / f"checkpoint_{flag}.pt"
        torch.save(
            {
                "model": ARM,
                "family": FAMILY,
                "bounded_gates": flag,
                "state_dict": source.state_dict(),
            },
            path,
        )
        model, metadata = load_checkpoint(path, torch.device("cpu"))
        assert model.output_projection.bounded_gate_invariants is flag
        assert metadata["bounded_gates"] is flag

    torch.manual_seed(11)
    legacy = _build_model(ARM, FAMILY)
    path = tmp_path / "legacy.pt"
    torch.save(
        {"model": ARM, "family": FAMILY, "state_dict": legacy.state_dict()}, path
    )
    model, metadata = load_checkpoint(path, torch.device("cpu"))
    assert model.output_projection.bounded_gate_invariants is False
    assert "bounded_gates" not in metadata


def test_load_checkpoint_honors_the_decay_structure_flags(tmp_path: Path) -> None:
    """Decay-structured checkpoints rebuild the iteration-30 arm.

    ``decaying_drive`` records ``MeshTransformer(decaying_direct_drive=...)``
    (the analytic 1/(1+|x|^2) direct-drive envelope) and ``monopole_free_sl``
    records ``kernel_monopole_free_single_layer`` (the zero-net-charge
    single-layer deflation).  Like the bounding knobs they add no
    parameters, so only the flags select the parameterization; all four
    flags compose, and pre-fix checkpoints (no flags) rebuild the historical
    arm (covered by the legacy case above).
    """

    for flag in (False, True):
        torch.manual_seed(19)
        source = _build_model(
            ARM,
            FAMILY,
            bounded_gates=flag,
            bounded_query=flag,
            decaying_drive=flag,
            monopole_free_sl=flag,
        )
        path = tmp_path / f"checkpoint_decay_{flag}.pt"
        torch.save(
            {
                "model": ARM,
                "family": FAMILY,
                "bounded_gates": flag,
                "bounded_query": flag,
                "decaying_drive": flag,
                "monopole_free_sl": flag,
                "state_dict": source.state_dict(),
            },
            path,
        )
        model, metadata = load_checkpoint(path, torch.device("cpu"))
        assert model.decaying_direct_drive is flag
        assert model.kernel_decoder.monopole_free_single_layer is flag
        assert model.bounded_query_geometry is flag
        assert metadata["decaying_drive"] is flag
        assert metadata["monopole_free_sl"] is flag


def test_load_checkpoint_honors_the_bounded_query_flag(tmp_path: Path) -> None:
    """Source-bounded checkpoints rebuild the compactified-injection arm.

    ``bounded_query`` records ``MeshTransformer(bounded_query_geometry=...)``
    -- the completion of the gate fix.  Like the gate knob it adds no
    parameters, so only the flag selects the parameterization; the two flags
    compose, and pre-fix checkpoints (no flag) rebuild the historical arm
    (covered by the legacy case of the bounded-gates test above).
    """

    for flag in (False, True):
        torch.manual_seed(13)
        source = _build_model(ARM, FAMILY, bounded_gates=flag, bounded_query=flag)
        path = tmp_path / f"checkpoint_query_{flag}.pt"
        torch.save(
            {
                "model": ARM,
                "family": FAMILY,
                "bounded_gates": flag,
                "bounded_query": flag,
                "state_dict": source.state_dict(),
            },
            path,
        )
        model, metadata = load_checkpoint(path, torch.device("cpu"))
        assert model.bounded_query_geometry is flag
        assert model.output_projection.bounded_gate_invariants is flag
        assert metadata["bounded_query"] is flag
