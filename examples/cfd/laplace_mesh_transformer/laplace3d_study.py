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

r"""3D learned boundary-integral model and benchmark driver.

``SolidAngleBIE`` is the three-dimensional analogue of the pruned 2D flagship:
a learned multiple of the double-layer kernel, integrated *exactly* over
triangles by the van Oosterom--Strackee signed solid angle, with a drive-linear
shared-relaxation Richardson solve through ``T = 1/2 I + K`` (zeroed
diagonal).  Because solid angles are dimensionless and invariant under
similarity transformations, the model needs no reference frame or length: it
has exactly two parameters (kernel coefficient and relaxation) and is exactly
linear in the Dirichlet data, similarity invariant, and query-set independent.

Scope (v1, documented): all-Dirichlet problems.  The mixed/Robin regimes of
the 3D suite require a Green-representation solve with unknown Cauchy data
(iteration 6); the shell tier additionally probes the classical completeness
deficiency of a pure double-layer representation on multiply connected
domains, so it is evaluated as a declared hard tier rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from laplace3d import build_laplace3d_sample, solid_angle_influence
from torch import nn

from physicsnemo.mesh import DomainMesh, Mesh


def _merged_panels(domain: DomainMesh):
    """Concatenate triangle vertices and Dirichlet data over all boundaries."""

    triangles, values = [], []
    for name in sorted(domain.boundaries.keys()):
        mesh = domain.boundaries[name]
        if "boundary_value" not in mesh.cell_data:
            raise ValueError(
                f"boundary {name!r} lacks Dirichlet data; SolidAngleBIE v1 "
                "supports the all-Dirichlet regime only"
            )
        triangles.append(mesh.points[mesh.cells])
        values.append(mesh.cell_data["boundary_value"])
    return torch.cat(triangles, dim=0), torch.cat(values, dim=0)


class SolidAngleBIE(nn.Module):
    """Two-parameter learned double-layer model with exact triangle quadrature."""

    def __init__(self, *, n_iterations: int = 8, query_chunk_size: int = 4096) -> None:
        super().__init__()
        if (
            isinstance(n_iterations, bool)
            or not isinstance(n_iterations, int)
            or n_iterations < 0
        ):
            raise ValueError("n_iterations must be a non-negative integer")
        self.n_iterations = n_iterations
        self.kernel_coefficient = nn.Parameter(1.0e-2 * torch.randn(()))
        if n_iterations:
            self.relaxation = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("relaxation", None)
        self.query_chunk_size = query_chunk_size

    def _density(self, triangles: torch.Tensor, values: torch.Tensor):
        centroids = triangles.mean(dim=1)
        density = values
        if self.n_iterations:
            influence = solid_angle_influence(centroids, triangles)
            influence = influence * (
                1.0
                - torch.eye(
                    influence.shape[0],
                    device=influence.device,
                    dtype=influence.dtype,
                )
            )
            influence = self.kernel_coefficient * influence
            for _ in range(self.n_iterations):
                trace = 0.5 * density + influence @ density
                density = density + self.relaxation * (values - trace)
        return density

    def forward(self, domain: DomainMesh) -> Mesh:
        triangles, values = _merged_panels(domain)
        density = self._density(triangles, values)
        query_points = domain.interior.points
        chunks = []
        for start in range(0, query_points.shape[0], self.query_chunk_size):
            influence = self.kernel_coefficient * solid_angle_influence(
                query_points[start : start + self.query_chunk_size], triangles
            )
            chunks.append((influence * density[None, :]).sum(dim=-1))
        potential = torch.cat(chunks) if chunks else query_points.new_empty((0,))
        return domain.interior.with_data(
            point_data={"potential": potential},
            cell_data={},
            global_data=domain.global_data,
        )


class SolvedSolidAngleOracle(nn.Module):
    """Parameter-free control: dense second-kind solve with the exact kernel.

    The double-layer representation ``u = D[mu]`` with outward normals has
    interior trace ``(-1/2 I + K) mu`` in the sign convention of
    ``solid_angle_influence`` (interior row sums are exactly -1).  The dense
    solve therefore uses ``-1/2 I + K``; on simply connected domains this is
    the classical interior Dirichlet BIE, and on shells it exposes the
    completeness deficiency of the pure double layer.
    """

    def forward(self, domain: DomainMesh) -> Mesh:
        triangles, values = _merged_panels(domain)
        centroids = triangles.mean(dim=1)
        influence = solid_angle_influence(centroids, triangles)
        influence = influence * (
            1.0
            - torch.eye(
                influence.shape[0],
                device=influence.device,
                dtype=influence.dtype,
            )
        )
        system = (
            -0.5
            * torch.eye(
                influence.shape[0],
                device=influence.device,
                dtype=influence.dtype,
            )
            + influence
        )
        density = torch.linalg.solve(system, values)
        query_points = domain.interior.points
        potential = (
            solid_angle_influence(query_points, triangles) * density[None, :]
        ).sum(dim=-1)
        return domain.interior.with_data(
            point_data={"potential": potential},
            cell_data={},
            global_data=domain.global_data,
        )


SPLITS_3D = {
    "sphere": {"tier": "sphere", "bc_regime": "dirichlet"},
    "star": {"tier": "star", "bc_regime": "dirichlet"},
    "star_unseen_modes": {
        "tier": "star",
        "bc_regime": "dirichlet",
        "star_modes": "ood",
    },
    "shell_topology": {"tier": "shell", "bc_regime": "dirichlet"},
}


def _make_sample(seed: int, spec: dict, dtype: torch.dtype, device="cpu"):
    from laplace3d import _STAR_MODES_OOD

    kwargs = dict(spec)
    if kwargs.pop("star_modes", None) == "ood":
        kwargs["star_modes"] = _STAR_MODES_OOD
    return build_laplace3d_sample(seed, dtype=dtype, device=device, **kwargs)


def _relative_l2(prediction, target):
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


@torch.no_grad()
def evaluate_splits(model, *, eval_seed: int, n_cases: int, dtype, device="cpu"):
    model.eval()
    report = {}
    for index, (name, spec) in enumerate(sorted(SPLITS_3D.items())):
        errors = []
        for case in range(n_cases):
            sample = _make_sample(
                eval_seed + 7919 * case + 1_000_003 * index, spec, dtype, device
            )
            prediction = model(sample.domain).point_data["potential"]
            errors.append(_relative_l2(prediction, sample.target))
        report[name] = sum(errors) / len(errors)
    return report


def run_experiment(
    *,
    model_name: str,
    steps: int,
    seed: int,
    output_dir: str,
    device: str = "cuda",
    eval_cases: int = 12,
):
    torch.manual_seed(seed)
    dtype = torch.float32
    model = SolvedSolidAngleOracle() if model_name == "oracle" else SolidAngleBIE()
    model = model.to(torch.device(device))

    history = []
    start = time.time()
    if model_name != "oracle":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3.0e-4, weight_decay=1.0e-6
        )
        best_val, best_state = float("inf"), None
        train_specs = [SPLITS_3D["sphere"], SPLITS_3D["star"]]
        for step in range(1, steps + 1):
            model.train()
            spec = train_specs[step % 2]
            sample = _make_sample(seed + 104_729 * step, spec, dtype, device)
            prediction = model(sample.domain).point_data["potential"]
            loss = torch.sum((prediction - sample.target).square()) / torch.sum(
                sample.target.square()
            ).clamp_min(1.0e-30)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == steps:
                validation = evaluate_splits(
                    model, eval_seed=71_000_011, n_cases=3, dtype=dtype, device=device
                )
                score = 0.5 * (validation["sphere"] + validation["star"])
                history.append({"step": step, "validation": score})
                if score < best_val:
                    best_val = score
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)

    report = {
        "model": model_name,
        "seed": seed,
        "steps": steps if model_name != "oracle" else 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.time() - start,
        "history": history,
        "splits": evaluate_splits(
            model, eval_seed=97_000_037, n_cases=eval_cases, dtype=dtype, device=device
        ),
        "state": {k: v.tolist() for k, v in model.state_dict().items()},
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("solid_angle_bie", "oracle"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    result = run_experiment(
        model_name=arguments.model,
        steps=arguments.steps,
        seed=arguments.seed,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    print(
        json.dumps({k: result[k] for k in ("model", "parameters", "splits", "state")})
    )
