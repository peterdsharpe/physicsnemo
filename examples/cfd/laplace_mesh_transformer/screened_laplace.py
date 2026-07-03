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

r"""Parametric-PDE testbed: screened Laplace (modified Helmholtz) on disks.

Iteration-3 research question: for a steady homogeneous PDE with a global
operator parameter -- here :math:`\Delta u - \kappa^2 u = 0` with
dimensionless screening :math:`\tilde\kappa = \kappa L_{\mathrm{ref}}` -- should
the parameter enter a learned boundary-integral surrogate through the
**kernel basis** (the parameter-dependent fundamental solutions
:math:`K_0, K_1`) or through **learned coefficient conditioning** (a network
mapping :math:`\tilde\kappa` to coefficients of the fixed Laplace harmonic
basis)?  Measured by in-distribution versus :math:`\tilde\kappa`-OOD
generalization, this decides the division of labor between PDE-conforming
decoders and learned encoders acting as coefficient conditioners.

Design notes (numerical knobs, not physical scales):

- Domains are circles (exact Bessel-series solutions), radius and center
  randomized; the reference length is the radius, so the screening parameter
  supplied in ``global_data`` is exactly the dimensionless
  :math:`\tilde\kappa = \kappa R`.
- The Yukawa kernel has no elementary panel antiderivative, so panels use
  fixed-order Gauss--Legendre quadrature (default 16 nodes/panel).  The
  self-panel entry of the trace operator is zeroed and the analytic
  ``1/2`` jump retained, matching the Dirichlet-family gauge; the omitted
  ``O(h log h)`` self-contribution is absorbed by the learned coefficients at
  fixed panel count.  This breaks exact resolution transfer and is documented
  rather than hidden; the :math:`\tilde\kappa`-OOD comparison is the point of
  this experiment.
- Both models share the drive-linear Richardson solve through
  ``T = 1/2 I + K`` (zero-kernel initialization inside the convergent regime)
  and are exactly linear in the boundary data, O(2)/similarity invariant, and
  query-set independent.  There is no constant lift: constants do not solve
  the screened equation.

This is a benchmark-local research prototype, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)


class _BesselK0(torch.autograd.Function):
    """K0 with the analytic derivative (torch.special lacks autograd here)."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(z)
        return torch.special.modified_bessel_k0(z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (z,) = ctx.saved_tensors
        return -grad_output * _bessel_k1(z)


class _BesselK1(torch.autograd.Function):
    """K1 with the analytic derivative K1' = -K0 - K1/z."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(z)
        return torch.special.modified_bessel_k1(z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (z,) = ctx.saved_tensors
        return grad_output * (-_bessel_k0(z) - _bessel_k1(z) / z)


def _bessel_k0(z: torch.Tensor) -> torch.Tensor:
    return _BesselK0.apply(z)


def _bessel_k1(z: torch.Tensor) -> torch.Tensor:
    return _BesselK1.apply(z)


def modified_bessel_i(order: int, x: torch.Tensor, *, terms: int = 40) -> torch.Tensor:
    r"""Series :math:`I_m(x)=\sum_k (x/2)^{2k+m}/(k!\,(k+m)!)`.

    Accurate to machine precision in float64 for ``x <= 25`` and
    ``order <= 10`` (documented validity range; asserted by callers).
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    half = 0.5 * x
    term = half.pow(order) / math.factorial(order)
    total = term
    for k in range(1, terms):
        term = term * half * half / (k * (k + order))
        total = total + term
    return total


@dataclass(frozen=True)
class ScreenedSample:
    """One exact screened-Laplace Dirichlet problem on a disk."""

    domain: DomainMesh
    target: torch.Tensor
    kappa_tilde: float


def build_screened_sample(
    seed: int,
    *,
    kappa_range: tuple[float, float],
    modes: tuple[int, ...],
    n_boundary: int = 64,
    n_query: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ScreenedSample:
    """Sample a disk, screening parameter, and balanced boundary spectrum."""

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(
            torch.empty((), dtype=torch.float64).uniform_(
                low, high, generator=generator
            )
        )

    radius = uniform(0.5, 2.0)
    center = torch.tensor([uniform(-2.0, 2.0), uniform(-2.0, 2.0)], dtype=torch.float64)
    offset = uniform(0.0, 2.0 * math.pi)
    kappa_tilde = uniform(*kappa_range)
    if kappa_tilde > 25.0:
        raise ValueError("kappa_tilde exceeds the Bessel-series validity range")
    kappa = kappa_tilde / radius

    coefficients = {
        m: torch.polar(
            torch.tensor(1.0 / math.sqrt(len(modes)), dtype=torch.float64),
            torch.tensor(uniform(0.0, 2.0 * math.pi), dtype=torch.float64),
        )
        for m in modes
    }

    def exact(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(r)
        for m, c in coefficients.items():
            radial = modified_bessel_i(m, kappa * r) / modified_bessel_i(
                m, torch.tensor(kappa * radius, dtype=torch.float64)
            )
            phase = torch.polar(torch.ones_like(theta), m * theta)
            total = total + (c * phase).real * radial
        return total

    vertex_angles = (
        offset
        + 2.0 * math.pi * torch.arange(n_boundary, dtype=torch.float64) / n_boundary
    )
    points = center + radius * torch.stack(
        (vertex_angles.cos(), vertex_angles.sin()), dim=-1
    )
    index = torch.arange(n_boundary)
    cells = torch.stack((index, torch.roll(index, -1)), dim=-1)
    midpoint_angles = vertex_angles + math.pi / n_boundary
    boundary_values = exact(
        torch.full((n_boundary,), radius, dtype=torch.float64), midpoint_angles
    )

    query_r = (
        radius
        * 0.95
        * torch.sqrt(torch.rand(n_query, dtype=torch.float64, generator=generator))
    )
    query_theta = (
        2.0 * math.pi * torch.rand(n_query, dtype=torch.float64, generator=generator)
    )
    query_points = center + torch.stack(
        (query_r * query_theta.cos(), query_r * query_theta.sin()), dim=-1
    )
    target = exact(query_r, query_theta)

    boundary = Mesh(
        points=points.to(device=device, dtype=dtype),
        cells=cells.to(device=device),
        cell_data={"boundary_value": boundary_values.to(device=device, dtype=dtype)},
    )
    interior = Mesh(points=query_points.to(device=device, dtype=dtype))
    domain = DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={
            "reference_length": torch.tensor(radius, device=device, dtype=dtype),
            "screening": torch.tensor(kappa_tilde, device=device, dtype=dtype),
        },
    )
    return ScreenedSample(
        domain=domain,
        target=target.to(device=device, dtype=dtype),
        kappa_tilde=kappa_tilde,
    )


def _panel_frame(
    query_points: torch.Tensor,
    panel_start: torch.Tensor,
    panel_end: torch.Tensor,
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return start/end vectors (y - x), sigma = n x tau, and panel lengths."""

    start_vector = panel_start.unsqueeze(0) - query_points.unsqueeze(1)
    end_vector = panel_end.unsqueeze(0) - query_points.unsqueeze(1)
    tangent = panel_end - panel_start
    lengths = tangent.norm(dim=-1)
    tangent = tangent / lengths[:, None]
    sigma = normals[:, 0] * tangent[:, 1] - normals[:, 1] * tangent[:, 0]
    return start_vector, end_vector, sigma, lengths


def harmonic_panel_influence(
    query_points: torch.Tensor,
    panel_start: torch.Tensor,
    panel_end: torch.Tensor,
    normals: torch.Tensor,
    singular_coefficient: torch.Tensor,
    regular_coefficients: torch.Tensor,
    *,
    zero_singular_diagonal: bool = False,
) -> torch.Tensor:
    """Exact panel-integrated Laplace-harmonic influences (tensor coefficients).

    Functional twin of ``HarmonicPanelBIE._influence`` accepting coefficients
    as tensors so a conditioning network can supply them per sample.
    """

    start_vector, end_vector, sigma, _ = _panel_frame(
        query_points, panel_start, panel_end, normals
    )
    cross = (
        start_vector[..., 0] * end_vector[..., 1]
        - start_vector[..., 1] * end_vector[..., 0]
    )
    dot = torch.sum(start_vector * end_vector, dim=-1)
    subtended = -sigma[None, :] * torch.atan2(cross, dot)
    if zero_singular_diagonal:
        subtended = subtended * (
            1.0
            - torch.eye(
                subtended.shape[0], device=subtended.device, dtype=subtended.dtype
            )
        )
    result = singular_coefficient * subtended

    def zeta_of(displacement_to_query: torch.Tensor) -> torch.Tensor:
        parallel = torch.einsum("qsd,sd->qs", -displacement_to_query, normals)
        perpendicular = -(
            displacement_to_query[..., 0] * normals[None, :, 1]
            - displacement_to_query[..., 1] * normals[None, :, 0]
        )
        return torch.complex(parallel, perpendicular)

    zeta_start = zeta_of(start_vector)
    zeta_end = zeta_of(end_vector)
    power_start = torch.ones_like(zeta_start)
    power_end = torch.ones_like(zeta_end)
    for order in range(regular_coefficients.shape[-1]):
        power_start = power_start * zeta_start
        power_end = power_end * zeta_end
        antiderivative = (power_end - power_start) / (order + 1)
        result = result + regular_coefficients[..., order] * (
            sigma[None, :] * antiderivative.imag
        )
    return result


def yukawa_panel_influence(
    query_points: torch.Tensor,
    panel_start: torch.Tensor,
    panel_end: torch.Tensor,
    normals: torch.Tensor,
    single_coefficient: torch.Tensor,
    double_coefficient: torch.Tensor,
    kappa_tilde: torch.Tensor,
    *,
    zero_diagonal: bool = False,
) -> torch.Tensor:
    r"""Gauss--Legendre panel integrals of the screened fundamental solutions.

    Kernel: ``c_s K_0(\tilde\kappa \rho) + c_d \tilde\kappa K_1(\tilde\kappa
    \rho) (n\cdot r)/\rho`` with ``r = x - y`` and ``\rho = |r|`` in the
    normalized frame.  Sixteen nodes per panel is a numerical accuracy knob.
    """

    nodes = torch.tensor(
        _GAUSS_NODES, device=query_points.device, dtype=query_points.dtype
    )
    weights = torch.tensor(
        _GAUSS_WEIGHTS, device=query_points.device, dtype=query_points.dtype
    )
    midpoint = 0.5 * (panel_start + panel_end)
    half_edge = 0.5 * (panel_end - panel_start)
    lengths = 2.0 * half_edge.norm(dim=-1)
    # (s, g, 2) quadrature points on each panel.
    points = midpoint[:, None, :] + nodes[None, :, None] * half_edge[:, None, :]
    displacement = query_points[:, None, None, :] - points[None, :, :, :]
    rho = displacement.norm(dim=-1).clamp_min(1.0e-12)
    scaled = (kappa_tilde * rho).clamp(1.0e-8, 80.0)
    k0 = _bessel_k0(scaled)
    k1 = _bessel_k1(scaled)
    normal_dot = torch.einsum("qsgd,sd->qsg", displacement, normals)
    kernel = single_coefficient * k0 + double_coefficient * (
        kappa_tilde * k1 * normal_dot / rho
    )
    influence = (kernel * weights[None, None, :]).sum(dim=-1) * (0.5 * lengths)[None, :]
    if zero_diagonal:
        influence = influence * (
            1.0
            - torch.eye(
                influence.shape[0], device=influence.device, dtype=influence.dtype
            )
        )
    return influence


class ScreenedPanelBIE(nn.Module):
    r"""Learned screened-Laplace layer potential with a Richardson solve.

    ``kernel_form="yukawa"`` places the screening parameter in the basis
    (:math:`K_0/K_1` fundamental solutions); ``kernel_form="harmonic"`` keeps
    the Laplace harmonic basis.  ``conditioned=True`` produces the kernel
    coefficients from a small MLP of :math:`\tilde\kappa` (the
    encoder-as-conditioner pattern); otherwise they are direct parameters.
    """

    def __init__(
        self,
        *,
        kernel_form: str = "yukawa",
        conditioned: bool = False,
        regular_orders: int = 3,
        n_iterations: int = 8,
        hidden_dim: int = 32,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if kernel_form not in ("yukawa", "harmonic", "union"):
            raise ValueError("kernel_form must be 'yukawa', 'harmonic', or 'union'")
        self.kernel_form = kernel_form
        self.conditioned = conditioned
        self.regular_orders = regular_orders
        self.n_iterations = n_iterations
        self.query_chunk_size = query_chunk_size
        if kernel_form == "yukawa":
            n_coefficients = 2
        elif kernel_form == "harmonic":
            n_coefficients = 2 + regular_orders
        else:
            # Union basis: Yukawa (2) plus harmonic (2 + regular_orders); the
            # families overlap in the mid-range but each carries its own
            # asymptotic regime (kappa -> 0 harmonic, kappa -> inf screened).
            n_coefficients = 4 + regular_orders
        if conditioned:
            self.conditioner = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_coefficients),
            )
            with torch.no_grad():
                self.conditioner[-1].weight.mul_(1.0e-2)
                self.conditioner[-1].bias.normal_(std=1.0e-2)
        else:
            self.coefficients = nn.Parameter(1.0e-2 * torch.randn(n_coefficients))
        if n_iterations:
            self.relaxation = nn.Parameter(torch.ones(n_iterations))
        else:
            self.register_parameter("relaxation", None)

    def _kernel_coefficients(self, kappa_tilde: torch.Tensor) -> torch.Tensor:
        if self.conditioned:
            return self.conditioner(kappa_tilde.reshape(1, 1)).reshape(-1)
        return self.coefficients

    def _influence(
        self,
        query_points: torch.Tensor,
        panel_start: torch.Tensor,
        panel_end: torch.Tensor,
        normals: torch.Tensor,
        coefficients: torch.Tensor,
        kappa_tilde: torch.Tensor,
        *,
        zero_diagonal: bool = False,
    ) -> torch.Tensor:
        if self.kernel_form in ("yukawa", "union"):
            screened = yukawa_panel_influence(
                query_points,
                panel_start,
                panel_end,
                normals,
                coefficients[0],
                coefficients[1],
                kappa_tilde,
                zero_diagonal=zero_diagonal,
            )
            if self.kernel_form == "yukawa":
                return screened
            return screened + harmonic_panel_influence(
                query_points,
                panel_start,
                panel_end,
                normals,
                coefficients[2],
                coefficients[3:],
                zero_singular_diagonal=zero_diagonal,
            )
        return harmonic_panel_influence(
            query_points,
            panel_start,
            panel_end,
            normals,
            coefficients[0],
            coefficients[1:],
            zero_singular_diagonal=zero_diagonal,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = domain.boundaries["dirichlet"]
        values = boundary.cell_data["boundary_value"]
        length = domain.global_data["reference_length"].reshape(())
        kappa_tilde = domain.global_data["screening"].reshape(())

        with torch.autocast(device_type=boundary.points.device.type, enabled=False):
            weights = boundary.cell_areas / length
            center = torch.einsum("n,nd->d", weights, boundary.cell_centroids)
            center = center / weights.sum()
            vertices = boundary.points[boundary.cells]
            panel_start = (vertices[:, 0] - center) / length
            panel_end = (vertices[:, 1] - center) / length
            midpoints = (boundary.cell_centroids - center) / length
        normals = boundary.cell_normals
        coefficients = self._kernel_coefficients(kappa_tilde.to(values.dtype))

        density = values
        if self.n_iterations:
            influence = self._influence(
                midpoints,
                panel_start,
                panel_end,
                normals,
                coefficients,
                kappa_tilde,
                zero_diagonal=True,
            )
            relaxation = self.relaxation.to(dtype=values.dtype)
            for step in relaxation.unbind():
                trace = 0.5 * density + influence @ density
                density = density + step * (values - trace)

        query_points = (domain.interior.points - center) / length
        chunks: list[torch.Tensor] = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            influence = self._influence(
                query_points[start : start + self.query_chunk_size],
                panel_start,
                panel_end,
                normals,
                coefficients,
                kappa_tilde,
            )
            chunks.append((influence * density[None, :]).sum(dim=-1))
        potential = (
            torch.cat(chunks) if chunks else domain.interior.points.new_empty((0,))
        )
        return domain.interior.with_data(
            point_data={"potential": potential},
            cell_data={},
            global_data=domain.global_data,
        )


SPLITS: dict[str, dict] = {
    "in_distribution": {"kappa_range": (0.5, 2.0), "modes": (0, 1, 2, 3)},
    "ood_high_screening": {"kappa_range": (3.0, 5.0), "modes": (0, 1, 2, 3)},
    "ood_low_screening": {"kappa_range": (0.05, 0.3), "modes": (0, 1, 2, 3)},
    "unseen_modes": {"kappa_range": (0.5, 2.0), "modes": (4, 5, 6)},
}


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    *,
    eval_seed: int,
    n_cases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    report: dict[str, float] = {}
    for split_index, (name, spec) in enumerate(sorted(SPLITS.items())):
        errors = []
        for case in range(n_cases):
            sample = build_screened_sample(
                eval_seed + 7919 * case + 1_000_003 * split_index,
                kappa_range=spec["kappa_range"],
                modes=spec["modes"],
                device=device,
                dtype=dtype,
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        report[name] = sum(errors) / len(errors)
    return report


def pde_residual(
    model: nn.Module,
    *,
    seed: int,
    device: torch.device,
) -> float:
    """Return ``||(lap - kappa^2) u_pred|| * L^2 / ||u_pred||`` via autograd."""

    model.eval()
    residuals = []
    for case in range(2):
        sample = build_screened_sample(
            seed + case,
            kappa_range=SPLITS["in_distribution"]["kappa_range"],
            modes=SPLITS["in_distribution"]["modes"],
            n_query=32,
            device=device,
            dtype=torch.float64,
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        u = model_fp64(domain).point_data["potential"]
        (grad,) = torch.autograd.grad(u.sum(), points, create_graph=True)
        laplacian = torch.zeros_like(u)
        for component in range(2):
            (second,) = torch.autograd.grad(
                grad[:, component].sum(), points, create_graph=True
            )
            laplacian = laplacian + second[:, component]
        length = sample.domain.global_data["reference_length"].double()
        kappa = sample.domain.global_data["screening"].double() / length
        residual = (laplacian - kappa**2 * u) * length**2
        residuals.append(
            float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(u).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


def run_experiment(
    *,
    model_name: str,
    steps: int,
    seed: int,
    device: str,
    output_dir: str,
    eval_cases: int = 16,
) -> dict:
    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    builders = {
        "yukawa": dict(kernel_form="yukawa", conditioned=False),
        "conditioned_harmonic": dict(kernel_form="harmonic", conditioned=True),
        "conditioned_yukawa": dict(kernel_form="yukawa", conditioned=True),
        "conditioned_union": dict(kernel_form="union", conditioned=True),
    }
    model = ScreenedPanelBIE(**builders[model_name]).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-6)
    train_spec = SPLITS["in_distribution"]

    best_state, best_val, history = None, float("inf"), []
    start_time = time.time()
    for step in range(1, steps + 1):
        model.train()
        sample = build_screened_sample(
            seed + 104_729 * step,
            kappa_range=train_spec["kappa_range"],
            modes=train_spec["modes"],
            device=device_t,
            dtype=dtype,
        )
        prediction = model(sample.domain).point_data["potential"]
        loss = torch.sum((prediction - sample.target).square()) / torch.sum(
            sample.target.square()
        ).clamp_min(1.0e-30)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == steps:
            validation = evaluate_splits(
                model,
                eval_seed=71_000_011,
                n_cases=4,
                device=device_t,
                dtype=dtype,
            )["in_distribution"]
            history.append({"step": step, "validation_relative_l2": validation})
            if validation < best_val:
                best_val = validation
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_validation_relative_l2": best_val,
        "splits": evaluate_splits(
            model,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            device=device_t,
            dtype=dtype,
        ),
        "pde_residual": pde_residual(model, seed=83_000_019, device=device_t),
        "state": {
            k: v.tolist() for k, v in model.state_dict().items() if v.numel() <= 16
        },
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "yukawa",
            "conditioned_harmonic",
            "conditioned_yukawa",
            "conditioned_union",
        ),
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
    )
    print(json.dumps({k: result[k] for k in ("model", "splits", "pde_residual")}))
