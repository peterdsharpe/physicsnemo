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

r"""Structural contracts of the DECLARED-degree quadratic field mode.

The quadratic mode's whole claim is one sentence: the prediction is exactly
a polynomial of degree at most two in the drive at fixed geometry, for ANY
weights.  The acceptance test is therefore algebraic, not statistical: scale
every drive input by :math:`\alpha` and the output must be exactly
:math:`c_1\alpha + c_2\alpha^2` per output entry -- two evaluations determine
the coefficients and every further :math:`\alpha` (including large,
extrapolated ones: the failure regime of the implicit-degree nonlinear mode
diagnosed in iteration 34) must match at float64 machine precision.  The
same probe run on the ``zero_preserving_nonlinear`` mode must FAIL, so the
test discriminates rather than passing vacuously.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import ScalarVectorState
from physicsnemo.experimental.nn.mesh_attention.block import (
    LinearMeshFieldBlock,
    QuadraticFieldReadIn,
)
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    LinearKernelBasisCrossDecoder,
)
from physicsnemo.experimental.nn.mesh_attention.model import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

#: Alphas beyond the fit pair, deliberately including sign flips and the
#: >2x extrapolation band where iteration 34 measured the nonlinear mode's
#: implicit degree ~21.
_PROBE_ALPHAS = (0.37, 3.1, 11.0, -1.6, 40.0)


def _circle_domain(
    *,
    alpha: float = 1.0,
    pseudo: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> DomainMesh:
    """A 2D segment boundary with boundary and global drives scaled jointly."""

    generator = torch.Generator(device="cpu").manual_seed(1234)
    n_boundary = 14
    angles = torch.linspace(0.0, 2.0 * torch.pi, n_boundary + 1, dtype=dtype)[:-1]
    points = torch.stack((angles.cos(), 0.8 * angles.sin()), dim=-1)
    index = torch.arange(n_boundary)
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    cell_data = {
        "bvel": alpha * torch.randn(n_boundary, 2, generator=generator, dtype=dtype),
        "bscalar": alpha * torch.randn(n_boundary, generator=generator, dtype=dtype),
    }
    global_data = {
        "gdrive": alpha * torch.tensor([0.3, -0.7], dtype=dtype),
        "material": torch.tensor(1.3, dtype=dtype),
    }
    if pseudo:
        global_data["gpseudo"] = alpha * torch.tensor(0.9, dtype=dtype)
    queries = 0.5 * torch.randn(9, 2, generator=generator, dtype=dtype)
    return DomainMesh(
        interior=Mesh(points=queries.to(device)),
        boundaries={
            "wall": Mesh(
                points=points.to(device),
                cells=cells.to(device),
                cell_data={k: v.to(device) for k, v in cell_data.items()},
            )
        },
        global_data={k: v.to(device) for k, v in global_data.items()},
    )


def _model(
    *,
    field_mode: str,
    query_decoder: str = "kernel",
    pseudo: bool = False,
    device: torch.device | str = "cpu",
) -> MeshTransformer:
    torch.manual_seed(88)
    global_drive: dict[str, int | str] = {"gdrive": 1}
    if pseudo:
        global_drive["gpseudo"] = "0o"
    model = MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks={"velocity": 1, "pressure": 0},
        boundary_field_ranks={
            "wall": {"operator": {}, "drive": {"bvel": 1, "bscalar": 0}}
        },
        global_field_ranks={"operator": {"material": 0}, "drive": global_drive},
        field_mode=field_mode,
        query_decoder=query_decoder,
        kernel_mlp_members=2,
        operator_scalar_dim=8,
        operator_vector_dim=4,
        drive_scalar_dim=12,
        drive_vector_dim=6,
        drive_pseudo_dim=4 if pseudo else 0,
        operator_layers=1,
        drive_layers=1,
        heads=2,
        scalar_rank=4,
        vector_rank=2,
    ).to(device=device, dtype=torch.float64)
    for module in model.modules():
        if hasattr(module, "accumulation_dtype"):
            module.accumulation_dtype = torch.float64
    # The declared-degree claim is FOR ANY WEIGHTS; the zero-initialized
    # gates and small layer scales of a fresh model would test a nearly
    # trivial configuration, so every parameter is randomized.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.uniform_(-0.4, 0.4)
    model.eval()
    return model


@torch.no_grad()
def _flat_output(model: MeshTransformer, domain: DomainMesh) -> torch.Tensor:
    point_data = model(domain).point_data
    return torch.cat(
        (point_data["velocity"].flatten(), point_data["pressure"].flatten())
    )


def _degree_two_residual(
    model: MeshTransformer, *, pseudo: bool = False
) -> tuple[float, float]:
    r"""Worst relative residual of the exact degree-<=2 polynomial fit.

    ``output(alpha)`` at :math:`\alpha \in \{1, 2\}` determines
    :math:`c_1, c_2` of :math:`c_1\alpha + c_2\alpha^2` exactly (the
    constant term is pinned by zero preservation, asserted separately);
    every probe alpha must then reproduce the evaluation to roundoff.
    Returns ``(worst_relative_residual, zero_drive_magnitude)``.
    """

    at_one = _flat_output(model, _circle_domain(alpha=1.0, pseudo=pseudo))
    at_two = _flat_output(model, _circle_domain(alpha=2.0, pseudo=pseudo))
    quadratic = (at_two - 2.0 * at_one) / 2.0
    linear = at_one - quadratic
    worst = 0.0
    for alpha in _PROBE_ALPHAS:
        actual = _flat_output(model, _circle_domain(alpha=alpha, pseudo=pseudo))
        predicted = linear * alpha + quadratic * alpha**2
        residual = float(
            (actual - predicted).abs().max() / actual.abs().max().clamp_min(1e-300)
        )
        worst = max(worst, residual)
    zero = float(
        _flat_output(model, _circle_domain(alpha=0.0, pseudo=pseudo)).abs().max()
    )
    return worst, zero


@pytest.mark.parametrize("query_decoder", ["kernel", "moment"])
def test_quadratic_mode_is_exactly_degree_two_in_the_drive(query_decoder) -> None:
    """The declared-degree contract at machine precision, any weights.

    This is the pre-registered iteration-35 acceptance test: the
    iteration-34 degree probe (joint drive scaling at fixed geometry)
    becomes exact -- output(alpha * drive) is a degree-<=2 polynomial in
    alpha with fit residual at float64 roundoff, on both query decoders,
    including alphas far beyond any training range.
    """

    model = _model(field_mode="quadratic", query_decoder=query_decoder)
    residual, zero = _degree_two_residual(model)
    assert residual < 1.0e-12, residual
    assert zero == 0.0, zero


def test_quadratic_mode_pseudo_sector_keeps_the_declared_degree() -> None:
    """The 0o-typed drive rides the same degree-2 contract unchanged."""

    model = _model(field_mode="quadratic", pseudo=True)
    residual, zero = _degree_two_residual(model, pseudo=True)
    assert residual < 1.0e-12, residual
    assert zero == 0.0, zero


def test_nonlinear_mode_fails_the_degree_probe_the_test_discriminates() -> None:
    """The same probe rejects the implicit-degree nonlinear mode.

    Guards against the acceptance test passing vacuously: the
    zero_preserving_nonlinear mode (measured implicit drive degree ~21 in
    iteration 34) must produce a LARGE polynomial-fit residual under the
    identical protocol.
    """

    model = _model(field_mode="zero_preserving_nonlinear")
    residual, _ = _degree_two_residual(model)
    assert residual > 1.0e-3, residual


def test_quadratic_mode_has_a_genuine_quadratic_component() -> None:
    """The even (drive-quadratic) response is nonzero: no silent linearity.

    A quadratic arm that degenerated to the linear machinery would satisfy
    the degree contract trivially and rebuild the euler pressure wall; the
    even component output(+d) + output(-d) must be structurally available
    with generic weights.
    """

    model = _model(field_mode="quadratic")
    plus = _flat_output(model, _circle_domain(alpha=1.0))
    minus = _flat_output(model, _circle_domain(alpha=-1.0))
    assert float((plus + minus).abs().max()) > 1.0e-8


def test_read_in_residual_is_exactly_bilinear() -> None:
    """B(L2 u, L3 u) scales exactly quadratically at the module level.

    ``QuadraticFieldReadIn(geometry, alpha * field) - alpha * field`` must
    equal ``alpha**2 * (QuadraticFieldReadIn(geometry, field) - field)``:
    the residual is a homogeneous degree-2 form of the field for any
    weights, including the parity-odd branches.
    """

    torch.manual_seed(5)
    module = QuadraticFieldReadIn(5, 3, 6, 4, field_pseudo_dim=2).double()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.5, 0.5)
    geometry = ScalarVectorState(
        torch.randn(7, 5, dtype=torch.float64),
        torch.randn(7, 3, 2, dtype=torch.float64),
    )
    field = ScalarVectorState(
        torch.randn(7, 6, dtype=torch.float64),
        torch.randn(7, 4, 2, dtype=torch.float64),
        torch.randn(7, 2, dtype=torch.float64),
    )
    with torch.no_grad():
        base = module(geometry, field)
        for alpha in (2.0, -3.0, 0.25):
            scaled_field = ScalarVectorState(
                alpha * field.scalars, alpha * field.vectors, alpha * field.pseudos
            )
            scaled = module(geometry, scaled_field)
            for sector in ("scalars", "vectors", "pseudos"):
                actual = getattr(scaled, sector) - alpha * getattr(field, sector)
                expected = alpha**2 * (getattr(base, sector) - getattr(field, sector))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_quadratic_mode_reuses_the_linear_machinery_and_knob_discipline() -> None:
    """Mode selection contracts: classes, defaults, and rejection.

    The quadratic mode must route through the LINEAR field blocks and
    kernel decoder (that reuse is what makes the degree provable), carry
    the read-in only in quadratic mode (state dicts of the other modes stay
    bitwise pre-extension), default query depth to the linear machinery's,
    and reject unknown modes with the full menu.
    """

    quadratic = _model(field_mode="quadratic")
    assert isinstance(quadratic.quadratic_read_in, QuadraticFieldReadIn)
    assert isinstance(quadratic.kernel_decoder, LinearKernelBasisCrossDecoder)
    assert all(
        isinstance(block, LinearMeshFieldBlock) for block in quadratic.drive_blocks
    )
    assert quadratic.query_layers == 1

    linear = _model(field_mode="linear")
    assert linear.quadratic_read_in is None
    nonlinear = _model(field_mode="zero_preserving_nonlinear")
    assert nonlinear.quadratic_read_in is None
    assert not any("quadratic_read_in" in name for name, _ in linear.named_parameters())
    assert any("quadratic_read_in" in name for name, _ in quadratic.named_parameters())

    moment = _model(field_mode="quadratic", query_decoder="moment")
    assert all(isinstance(block, LinearMeshFieldBlock) for block in moment.query_blocks)

    with pytest.raises(ValueError, match="field_mode must be"):
        MeshTransformer(
            n_spatial_dims=2,
            output_field_ranks={"pressure": 0},
            boundary_field_ranks={"wall": {"operator": {}, "drive": {"bscalar": 0}}},
            field_mode="degree_2",
        )
