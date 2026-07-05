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

"""Contracts for the single-stream operator/drive fusion CONTROL (iter. 26).

Pre-registered study design, duplicated from ``models.SingleStreamFusionControl``
so the measurement cannot drift from its declaration:

- The control re-declares the Dirichlet ``boundary_value`` as an *operator*
  field (boundary data rides the full nonlinear, biased geometry encoder)
  and feeds a constant ``unit_drive = 1`` as the sole drive, making the
  prediction an unrestricted function of the fused encoding.  It is a
  measurement-only arm and must never be promoted to a real mode.
- **H1**: on the linear Laplace bank, two-stream matches or beats the
  control in-distribution (falsifier: control wins ID by > 2x seed sd).
- **H2**: two-stream is decisively better on drive-OOD
  (``unseen_boundary_frequencies``) because drive linearity is structural
  (falsifier: control matches that split).
- **H3**: the control has *measurably* nonzero zero-drive response and
  superposition residual; the two-stream arm holds both to float roundoff.
  Structural truth is asserted here; the trained magnitude is measured by
  the benchmark's first-class ``drive_linearity`` metrics.

Arms are parameter-matched: 104,537 (two-stream ``singpair``) versus 104,569
(control) at reference capacity on Laplace, and 171,449 versus 171,481 in the
zero-preserving nonlinear mode used on Liouville — both far inside the
pre-registered +-5% budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _sub in ("", "models", "problems"):
    _entry = str(EXAMPLE_DIR / _sub) if _sub else str(EXAMPLE_DIR)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from conformal_laplace import (  # noqa: E402
    build_domain_sample,
    sample_drive,
    sample_geometry,
)
from models import (  # noqa: E402
    MeshTransformerConfig,
    SingleStreamFusionControl,
    build_mesh_transformer,
    build_single_stream_control,
    parameter_count,
)

from physicsnemo.mesh import DomainMesh  # noqa: E402

_TINY = MeshTransformerConfig(
    operator_scalar_dim=8,
    operator_vector_dim=2,
    drive_scalar_dim=8,
    drive_vector_dim=2,
    operator_layers=1,
    drive_layers=1,
    query_layers=1,
    heads=2,
    scalar_rank=4,
    vector_rank=2,
)


def _build_two_stream(config: MeshTransformerConfig, field_mode: str = "linear"):
    """The paired two-stream ``singpair`` arm with the identical dictionary."""

    return build_mesh_transformer(
        config,
        query_decoder="kernel",
        kernel_mlp_members=0,
        kernel_include_polynomial_members=False,
        kernel_include_single_layer_member=True,
        field_mode=field_mode,
    )


def _sample(seed: int = 3, n_boundary: int = 24, n_query: int = 12):
    geometry = sample_geometry(
        seed, modes=(2, 3), deformation_range=(0.05, 0.35), dtype=torch.float32
    )
    drive = sample_drive(
        seed + 1,
        modes=(1, 2, 3, 4),
        regularity=0.0,
        boundary_rms=1.0,
        include_constant=True,
        dtype=torch.float32,
    )
    return build_domain_sample(
        geometry, drive, n_boundary=n_boundary, n_query=n_query, query_seed=seed + 2
    )


def _domain_with_values(domain: DomainMesh, values: torch.Tensor) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"]
    return DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}, global_data={}),
        boundaries={
            "dirichlet": boundary.with_data(cell_data={"boundary_value": values})
        },
        global_data=domain.global_data,
    )


def test_parameter_match_within_declared_budget() -> None:
    """Both arm pairs sit far inside the pre-registered +-5% budget."""

    reference = MeshTransformerConfig()
    for field_mode, expected_two, expected_control in (
        ("linear", 104_537, 104_569),
        ("zero_preserving_nonlinear", 171_449, 171_481),
    ):
        two_stream = parameter_count(_build_two_stream(reference, field_mode))
        control = parameter_count(
            build_single_stream_control(reference, field_mode=field_mode)
        )
        assert two_stream == expected_two
        assert control == expected_control
        assert abs(control - two_stream) / two_stream < 0.05


def test_construction_forward_backward() -> None:
    """The control constructs, predicts finite scalars, and trains end to end."""

    torch.manual_seed(0)
    model = build_single_stream_control(_TINY)
    assert isinstance(model, SingleStreamFusionControl)
    sample = _sample()
    values = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    prediction = model(_domain_with_values(sample.domain, values)).point_data[
        "potential"
    ]
    assert prediction.shape == (sample.domain.interior.n_points,)
    assert torch.isfinite(prediction).all()

    prediction.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    # The re-declared boundary_value column of the operator lift is live:
    # boundary data genuinely enters through the nonlinear geometry encoder.
    lift_grad = model.fused_model.operator_lift.scalar.weight.grad
    assert lift_grad is not None and float(lift_grad.abs().sum()) > 0.0


def test_boundary_values_enter_nonlinearly() -> None:
    """Scaling the drive does not scale the control's output (no linearity)."""

    torch.manual_seed(1)
    model = build_single_stream_control(_TINY).eval()
    sample = _sample(seed=11)
    values = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    with torch.no_grad():
        full = model(_domain_with_values(sample.domain, values)).point_data["potential"]
        half = model(_domain_with_values(sample.domain, 0.5 * values)).point_data[
            "potential"
        ]
    deviation = float((half - 0.5 * full).abs().max())
    assert deviation > 1.0e-4


def test_measured_contract_violation_h3() -> None:
    """H3 (structural part): control violates zero-drive/superposition; two-stream holds both to roundoff."""

    torch.manual_seed(2)
    control = build_single_stream_control(_TINY).eval()
    two_stream = _build_two_stream(_TINY).eval()
    sample = _sample(seed=23)
    values = sample.domain.boundaries["dirichlet"].cell_data["boundary_value"]
    other = torch.roll(values, shifts=5)
    zeros = torch.zeros_like(values)
    tolerance = 128.0 * torch.finfo(torch.float32).eps

    with torch.no_grad():
        for model, is_control in ((control, True), (two_stream, False)):
            zero_response = model(_domain_with_values(sample.domain, zeros)).point_data[
                "potential"
            ]
            zero_rms = float(zero_response.square().mean().sqrt())
            first = model(_domain_with_values(sample.domain, values)).point_data[
                "potential"
            ]
            second = model(_domain_with_values(sample.domain, other)).point_data[
                "potential"
            ]
            combined = model(
                _domain_with_values(sample.domain, 0.731 * values - 1.217 * other)
            ).point_data["potential"]
            expected = 0.731 * first - 1.217 * second
            superposition = float(
                (combined - expected).norm()
                / expected.norm().clamp_min(torch.finfo(torch.float32).eps)
            )
            if is_control:
                # Trivially true structurally; the trained magnitude is the
                # study's measured quantity (drive_linearity metrics).
                assert zero_rms > 1.0e-5
                assert superposition > 1.0e-4
            else:
                assert zero_rms <= tolerance
                assert superposition <= tolerance


def test_registered_in_laplace_and_liouville_registries() -> None:
    """Both problem registries build the declared arm pairs."""

    import liouville
    import train

    assert "mesh_transformer_single_stream" in train.MODEL_NAMES
    laplace_control = train.make_model("mesh_transformer_single_stream", "reference")
    assert isinstance(laplace_control, SingleStreamFusionControl)
    assert laplace_control.fused_model.field_mode == "linear"
    assert laplace_control.fused_model.reference_length_key == "reference_length"
    # The gauge knob applies to the control like every MeshTransformer arm.
    intrinsic = train.make_model(
        "mesh_transformer_single_stream", "reference", gauge="intrinsic"
    )
    assert intrinsic.fused_model.reference_length_key is None

    nl_two_stream = liouville._build_model("mesh_transformer_kernel_nl_singpair")
    assert nl_two_stream.field_mode == "zero_preserving_nonlinear"
    assert nl_two_stream.kernel_decoder.include_single_layer_member is True
    assert nl_two_stream.kernel_decoder.n_members == 2
    nl_control = liouville._build_model("mesh_transformer_single_stream_nl")
    assert isinstance(nl_control, SingleStreamFusionControl)
    assert nl_control.fused_model.field_mode == "zero_preserving_nonlinear"
    assert (
        parameter_count(nl_control) - parameter_count(nl_two_stream)
        == 32  # one extra operator-lift input column at reference capacity
    )


def test_control_is_documented_as_measurement_only() -> None:
    """The non-production status and hypotheses stay pre-registered in code."""

    docstring = SingleStreamFusionControl.__doc__
    assert "never be offered as a real modeling mode" in docstring
    for hypothesis in ("H1", "H2", "H3", "Falsifier"):
        assert hypothesis in docstring
