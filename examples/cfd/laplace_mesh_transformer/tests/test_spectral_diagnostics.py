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

"""Operator and representation-theoretic checks for the Laplace benchmark."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from conformal_laplace import (  # noqa: E402
    ConformalGeometry,
    HarmonicDrive,
    build_domain_sample,
)
from models import (  # noqa: E402
    BoundaryMean,
    MeshTransformerConfig,
    build_mesh_transformer,
)
from spectral_diagnostics import (  # noqa: E402
    compare_operator_spectra,
    concentric_ring_preimages,
    extract_discrete_operator,
    fourier_moment_family_norms,
    fourier_transfer_matrix,
    operator_spectrum,
    uniform_angles,
    unit_disk_poisson_matrix,
)

from physicsnemo.mesh import DomainMesh  # noqa: E402


def _unit_disk_domain(
    *,
    n_boundary: int,
    query_preimages: torch.Tensor,
) -> DomainMesh:
    dtype = query_preimages.real.dtype
    geometry = ConformalGeometry(
        modes=(),
        coefficients=torch.empty(0, dtype=query_preimages.dtype),
    )
    drive = HarmonicDrive(
        constant=torch.zeros((), dtype=dtype),
        modes=(),
        coefficients=torch.empty(0, dtype=query_preimages.dtype),
    )
    return build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        query_preimages=query_preimages.reshape(-1),
    ).domain


def _with_boundary_values(domain: DomainMesh, values: torch.Tensor) -> DomainMesh:
    boundary = domain.boundaries["dirichlet"]
    return DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}),
        boundaries={
            "dirichlet": boundary.with_data(cell_data={"boundary_value": values})
        },
        global_data=domain.global_data.copy(),
    )


def test_poisson_matrix_has_exact_fourier_transfer_up_to_quadrature_aliasing() -> None:
    """The analytic reference must recover r**k on each concentric ring."""

    n_angles = 128
    radii = torch.tensor([0.0, 0.27, 0.73], dtype=torch.float64)
    queries, query_angles = concentric_ring_preimages(
        radii, n_angles, angular_offset=0.137
    )
    boundary_angles = uniform_angles(
        n_angles, offset=math.pi / n_angles, dtype=torch.float64
    )
    poisson = unit_disk_poisson_matrix(boundary_angles, queries).reshape(
        radii.numel(), n_angles, n_angles
    )
    input_modes = tuple(range(9))
    output_modes = tuple(range(-8, 9))
    transfer = fourier_transfer_matrix(
        poisson,
        boundary_angles,
        query_angles,
        input_modes=input_modes,
        output_modes=output_modes,
    )

    expected = torch.zeros_like(transfer)
    for input_index, mode in enumerate(input_modes):
        # A source mode exp(i k theta) gives r**k exp(i k phi).
        output_index = output_modes.index(mode)
        expected[:, output_index, input_index] = radii.to(torch.complex128) ** mode
    torch.testing.assert_close(transfer, expected, rtol=2.0e-12, atol=2.0e-12)


def test_spectrum_reports_eckart_young_errors_and_analytic_comparison() -> None:
    """Reported tails must equal the optimal truncated-SVD errors."""

    generator = torch.Generator().manual_seed(1207)
    analytic = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    left, singular, right_h = torch.linalg.svd(analytic, full_matrices=False)
    learned = (left[:, :3] * singular[:3]) @ right_h[:3]

    analytic_report = operator_spectrum(analytic, (0, 1, 3, 7))
    expected_rank_three = torch.sqrt(
        singular[3:].square().sum() / singular.square().sum()
    ).item()
    assert analytic_report.numerical_rank == 7
    assert analytic_report.relative_best_rank_errors[0] == pytest.approx(1.0)
    assert analytic_report.relative_best_rank_errors[3] == pytest.approx(
        expected_rank_three
    )
    assert analytic_report.relative_best_rank_errors[7] == pytest.approx(0.0)

    comparison = compare_operator_spectra(learned, analytic, (0, 3, 7))
    assert comparison.learned.numerical_rank == 3
    assert comparison.learned.relative_best_rank_errors[3] < 2.0e-15
    assert comparison.relative_operator_error == pytest.approx(expected_rank_three)


def test_operator_extraction_recovers_boundary_mean_and_restores_mode() -> None:
    """Basis probing must recover a known map without changing train/eval state."""

    query_preimages = torch.tensor(
        [0.0 + 0.0j, 0.2 + 0.1j, -0.4 + 0.3j], dtype=torch.complex128
    )
    domain = _unit_disk_domain(n_boundary=12, query_preimages=query_preimages)
    model = BoundaryMean().train()
    operator = extract_discrete_operator(model, domain)

    torch.testing.assert_close(
        operator,
        torch.full((3, 12), 1.0 / 12.0, dtype=torch.float64),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert model.training


class _AffineBoundaryMean(BoundaryMean):
    """Intentionally invalid affine map used to audit the extractor guard."""

    def forward(self, domain: DomainMesh):  # type: ignore[no-untyped-def]
        prediction = super().forward(domain)
        return prediction.with_data(
            point_data={"potential": prediction.point_data["potential"] + 1.0}
        )


def test_operator_extraction_rejects_nonzero_affine_offset() -> None:
    """An affine offset must not be silently represented as a linear matrix."""

    domain = _unit_disk_domain(
        n_boundary=8,
        query_preimages=torch.tensor([0.0 + 0.0j], dtype=torch.complex128),
    )
    with pytest.raises(ValueError, match="not zero-preserving"):
        extract_discrete_operator(_AffineBoundaryMean(), domain)


def test_mesh_transformer_operator_pipeline_reconstructs_and_analyzes_map() -> None:
    """The full extraction, spectrum, and signed-transfer pipeline must compose."""

    n_boundary = 8
    n_query_angles = 16
    queries, query_angles = concentric_ring_preimages(
        torch.tensor([0.43], dtype=torch.float64),
        n_query_angles,
        angular_offset=0.217,
    )
    domain = _unit_disk_domain(
        n_boundary=n_boundary, query_preimages=queries.reshape(-1)
    )
    config = MeshTransformerConfig(
        operator_scalar_dim=4,
        operator_vector_dim=2,
        drive_scalar_dim=5,
        drive_vector_dim=2,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=1,
        scalar_rank=3,
        vector_rank=2,
        query_chunk_size=64,
        attention_chunk_size=None,
    )
    torch.manual_seed(1801)
    model = build_mesh_transformer(config).double().eval()
    operator = extract_discrete_operator(model, domain)

    probe = torch.linspace(-0.8, 1.1, n_boundary, dtype=torch.float64)
    direct = model(_with_boundary_values(domain, probe)).point_data["potential"]
    torch.testing.assert_close(operator @ probe, direct, rtol=2.0e-11, atol=2.0e-11)

    spectrum = operator_spectrum(operator, (0, 3, n_boundary))
    assert spectrum.numerical_rank <= 5
    transfer = fourier_transfer_matrix(
        operator.reshape(1, n_query_angles, n_boundary),
        uniform_angles(n_boundary, offset=math.pi / n_boundary, dtype=torch.float64),
        query_angles,
        input_modes=(0, 1, 2, 3),
        output_modes=tuple(range(-3, 4)),
    )
    assert transfer.shape == (1, 7, 4)
    assert torch.isfinite(transfer).all()


def test_typed_moment_report_covers_modes_zero_through_eight() -> None:
    """Every requested Fourier mode and decoder layer must be reported."""

    n_boundary = 20
    domain = _unit_disk_domain(
        n_boundary=n_boundary,
        query_preimages=torch.tensor([0.0 + 0.0j], dtype=torch.complex128),
    )
    angles = uniform_angles(
        n_boundary, offset=math.pi / n_boundary, dtype=torch.float64
    )
    torch.manual_seed(1901)
    model = build_mesh_transformer(
        MeshTransformerConfig(
            operator_scalar_dim=5,
            operator_vector_dim=2,
            drive_scalar_dim=6,
            drive_vector_dim=2,
            operator_layers=0,
            drive_layers=0,
            query_layers=2,
            heads=1,
            scalar_rank=3,
            vector_rank=2,
            query_chunk_size=64,
            attention_chunk_size=None,
        )
    ).double()
    model.train()
    report = fourier_moment_family_norms(model, domain, angles)

    assert tuple(report) == tuple(range(9))
    assert all(len(layers) == 2 for layers in report.values())
    values = [
        value
        for layers in report.values()
        for layer in layers
        for value in (
            layer.scalar_key_scalar_value,
            layer.vector_key_scalar_value,
            layer.scalar_key_vector_value,
            layer.vector_key_vector_value,
        )
    ]
    assert all(math.isfinite(value) and value >= 0.0 for value in values)
    assert max(values) > 0.0

    # With no drive-processing layers, a circular source has one radial vector
    # basis. Scalar moments can carry only k=0, the mixed one-index moments
    # only k=1, and the reducible two-index moment only k=0 or k=2.
    tolerance = 2.0e-11 * (1.0 + max(values))
    for mode, layers in report.items():
        for layer in layers:
            if mode != 0:
                assert layer.scalar_key_scalar_value <= tolerance
            if mode != 1:
                assert layer.vector_key_scalar_value <= tolerance
                assert layer.scalar_key_vector_value <= tolerance
            if mode not in (0, 2):
                assert layer.vector_key_vector_value <= tolerance
    assert model.training


_CEILING_CONFIGS = (
    MeshTransformerConfig(
        operator_scalar_dim=4,
        operator_vector_dim=1,
        drive_scalar_dim=4,
        drive_vector_dim=1,
        operator_layers=0,
        drive_layers=0,
        query_layers=1,
        heads=1,
        scalar_rank=1,
        vector_rank=1,
        query_chunk_size=128,
        attention_chunk_size=None,
    ),
    MeshTransformerConfig(
        operator_scalar_dim=7,
        operator_vector_dim=3,
        drive_scalar_dim=9,
        drive_vector_dim=3,
        operator_layers=1,
        drive_layers=1,
        query_layers=2,
        heads=2,
        scalar_rank=4,
        vector_rank=2,
        query_chunk_size=128,
        attention_chunk_size=None,
    ),
    MeshTransformerConfig(
        operator_scalar_dim=11,
        operator_vector_dim=4,
        drive_scalar_dim=13,
        drive_vector_dim=5,
        operator_layers=2,
        drive_layers=2,
        query_layers=4,
        heads=3,
        scalar_rank=7,
        vector_rank=5,
        query_chunk_size=128,
        attention_chunk_size=None,
    ),
)


@pytest.mark.parametrize("seed", [2111, 2113])
@pytest.mark.parametrize(
    "config",
    _CEILING_CONFIGS,
    ids=("narrow-shallow", "medium", "wide-deep"),
)
def test_random_scalar_vector_decoder_has_no_angular_order_above_two(
    config: MeshTransformerConfig,
    seed: int,
) -> None:
    r"""Width, head count, feature rank, and depth cannot create new irreps.

    This numerical regression complements the algebraic argument.  On a
    centered disk and one fixed-radius query ring, every query scalar is a
    radial coefficient multiplying contractions of source moments with zero,
    one, or two copies of the query vector.  It therefore contains only
    angular orders 0, 1, and 2.  Configured attention ``rank`` counts feature
    multiplicity; it is not physical tensor order.
    """

    n_boundary = 24
    n_query_angles = 64
    radii = torch.tensor([0.61], dtype=torch.float64)
    queries, query_angles = concentric_ring_preimages(
        radii, n_query_angles, angular_offset=0.173
    )
    domain = _unit_disk_domain(
        n_boundary=n_boundary, query_preimages=queries.reshape(-1)
    )
    generator = torch.Generator().manual_seed(seed + 10_000)
    boundary_values = torch.randn(n_boundary, generator=generator, dtype=torch.float64)
    domain = _with_boundary_values(domain, boundary_values)

    torch.manual_seed(seed)
    model = build_mesh_transformer(config).double().eval()
    with torch.no_grad():
        values = model(domain).point_data["potential"]

    # Scan every independent Fourier order of this real angular grid, including
    # the Nyquist coefficient. Checking only low modes could miss leakage tied
    # to the 24-panel source discretization.
    modes = torch.arange(0, n_query_angles // 2 + 1, dtype=torch.float64)
    analysis = torch.exp(
        -1j * query_angles[:, None].to(torch.complex128) * modes[None, :]
    )
    coefficients = (
        torch.einsum("am,a->m", analysis, values.to(torch.complex128)) / n_query_angles
    )
    high = coefficients[3:].abs().max()
    scale = torch.maximum(values.abs().max(), coefficients[:3].abs().max())
    assert high.item() <= 2.0e-11 * (1.0 + scale.item())
