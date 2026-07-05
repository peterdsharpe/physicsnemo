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

The ``mesh_transformer_kernel`` and ``mesh_transformer_kernel_nomlp`` arms
run the 3D :class:`~physicsnemo.experimental.nn.MeshTransformer` with the
dense kernel-basis query decoder (exact van Oosterom--Strackee triangle
quadrature member, plus 8 or 0 learned smooth MLP members) in the same
drive-linear, all-Dirichlet convention as the 2D benchmark in ``models.py``.
"""

from __future__ import annotations

import argparse
import json
import time

import _paths  # noqa: F401
import torch
from laplace3d import build_laplace3d_sample, solid_angle_influence
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
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


def merge_dirichlet_boundaries(domain: DomainMesh) -> DomainMesh:
    """Concatenate all-Dirichlet boundaries into one ``dirichlet`` boundary.

    The re-presentation used by :class:`MergedBoundaryDirichletModel`,
    exposed as a function so studies that drive the wrapped model's
    ``encode``/``decode`` split directly (for example the scale study's
    encoder-versus-decoder cost separation) apply the identical merge.
    Concatenation preserves per-cell geometry and data exactly, so it
    introduces no approximation.
    """

    parts = []
    for name in sorted(domain.boundaries.keys()):
        mesh = domain.boundaries[name]
        if "boundary_value" not in mesh.cell_data:
            raise ValueError(
                f"boundary {name!r} lacks Dirichlet data; the "
                "MeshTransformer arms support the all-Dirichlet regime only"
            )
        parts.append(
            Mesh(
                points=mesh.points,
                cells=mesh.cells,
                cell_data={"boundary_value": mesh.cell_data["boundary_value"]},
            )
        )
    return DomainMesh(
        interior=domain.interior,
        boundaries={"dirichlet": Mesh.merge(parts)},
        global_data=domain.global_data,
    )


class MergedBoundaryDirichletModel(nn.Module):
    r"""Present multi-boundary all-Dirichlet domains as one named boundary.

    ``MeshTransformer`` requires domain boundary names to match its declared
    schema exactly.  The 3D suite names boundaries by geometric role
    (``outer``, plus ``inner`` on shells) while the benchmark model schema
    declares a single ``dirichlet`` boundary; this wrapper concatenates all
    boundaries -- points, triangles, and ``boundary_value`` cell data -- into
    one mesh (:func:`merge_dirichlet_boundaries`) before invoking the wrapped
    model.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, domain: DomainMesh) -> Mesh:
        return self.model(merge_dirichlet_boundaries(domain))


def build_mesh_transformer_3d(
    *,
    kernel_mlp_members: int = 8,
    kernel_include_polynomial_members: bool = True,
    kernel_include_single_layer_member: bool = False,
    scalar_only: bool = False,
    scalar_wide: bool = False,
    one_head: bool = False,
    operator_layers: int = 2,
) -> MergedBoundaryDirichletModel:
    """Construct the 3D kernel-decoder analog of the 2D benchmark model.

    Same schema and conventions as ``models.build_mesh_transformer`` with
    ``query_decoder="kernel"`` (drive-linear, scalar Dirichlet data, global
    ``reference_length`` nondimensionalization), lifted to
    ``n_spatial_dims=3`` where the exact decoder member is the van
    Oosterom--Strackee triangle solid angle.  Capacity settings mirror the
    2D ``MeshTransformerConfig``; ``operator_layers`` (encoder depth) is a
    knob because the encoder-depth studies showed one layer suffices (the
    scale study measures cost curves at that depth).  ``one_head`` (default
    ``False``: bitwise-identical arms) applies the iteration-31 one-head
    capacity trade -- one full-width attention head at quadrupled per-head
    ranks (heads 4 -> 1, scalar_rank 12 -> 48, vector_rank 4 -> 16), which
    holds the total score capacity heads * (scalar_rank + D * vector_rank)
    exactly fixed in every D (4*(12 + D*4) = 1*(48 + D*16)); it is mutually
    exclusive with the scalar-control knobs, whose rank settings it would
    silently override.
    """

    if one_head and (scalar_only or scalar_wide):
        raise ValueError("one_head cannot be combined with the scalar controls")

    return MergedBoundaryDirichletModel(
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={
                "dirichlet": {
                    "operator": {},
                    "drive": {"boundary_value": 0},
                }
            },
            global_field_ranks={"operator": {}, "drive": {}},
            reference_length_key="reference_length",
            field_mode="linear",
            query_decoder="kernel",
            kernel_mlp_members=kernel_mlp_members,
            kernel_include_polynomial_members=kernel_include_polynomial_members,
            kernel_include_single_layer_member=kernel_include_single_layer_member,
            operator_scalar_dim=52 if scalar_wide else 32,
            operator_vector_dim=0 if scalar_only else 8,
            drive_scalar_dim=76 if scalar_wide else 48,
            drive_vector_dim=0 if scalar_only else 12,
            operator_layers=operator_layers,
            drive_layers=1,
            query_layers=1,
            heads=1 if one_head else 4,
            scalar_rank=48 if one_head else (18 if scalar_wide else 12),
            vector_rank=16 if one_head else (0 if scalar_only else 4),
            query_chunk_size=65536,
            attention_chunk_size=65536,
        )
    )


def _build_model(model_name: str) -> nn.Module:
    if model_name == "oracle":
        return SolvedSolidAngleOracle()
    if model_name == "solid_angle_bie":
        return SolidAngleBIE()
    if model_name in (
        "mesh_transformer_kernel",
        "mesh_transformer_kernel_nomlp",
        "mesh_transformer_kernel_nopoly",
        "mesh_transformer_kernel_singonly",
        "mesh_transformer_kernel_singpair",
        "mesh_transformer_kernel_singpair_h1",
        "mesh_transformer_kernel_singonly_scalar",
        "mesh_transformer_kernel_singonly_scalar_wide",
    ):
        # "_singpair" is the singular-only dictionary plus the exact
        # single-layer member: exactly two exact members (double layer and
        # single layer), no polynomial and no MLP smooth members.  The
        # single layer carries the net-flux (winding) component that a pure
        # double-layer representation cannot express on multiply connected
        # domains -- the declared deficiency of the shell-topology tier.
        # The "_h1" suffix applies the iteration-31 one-head capacity trade
        # on top (iteration-32 cross-suite confirmation; pre-registered in
        # studies/one_head_cross_suite.py).
        base_name = model_name.removesuffix("_h1")
        return build_mesh_transformer_3d(
            kernel_mlp_members=(
                0
                if (
                    "_nomlp" in base_name
                    or "_singonly" in base_name
                    or "_singpair" in base_name
                )
                else 8
            ),
            kernel_include_polynomial_members="_singonly" not in base_name
            and "_singpair" not in base_name
            and not base_name.endswith("_nopoly"),
            kernel_include_single_layer_member=base_name.endswith("_singpair"),
            scalar_only="_scalar" in base_name,
            scalar_wide=base_name.endswith("_scalar_wide"),
            one_head=model_name.endswith("_h1"),
        )
    if model_name in ("transolver_intree_matched", "transolver_intree_native"):
        from transolver_intree import build_transolver_intree

        # Mainstream-baseline arm: the in-tree Transolver on the shared
        # boundary-cell + query token sequence of the 2D bank
        # (transolver.build_token_sequence) lifted verbatim to 3D --
        # triangle centroids as token positions (3D coordinates), and outward
        # unit normal (3), dimensionless triangle area (area / L_ref^2),
        # Dirichlet value, and the boundary/query indicator as the 6 function
        # channels.  Multi-boundary shells reuse the exact merge the
        # MeshTransformer arms use.
        return MergedBoundaryDirichletModel(
            build_transolver_intree(
                model_name.removeprefix("transolver_"), space_dim=3
            )
        )
    raise ValueError(f"unknown model {model_name!r}")


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


def pde_residual(
    model: nn.Module,
    *,
    spec: dict,
    seed: int,
    device: torch.device | str = "cpu",
) -> float:
    r"""Return ``||lap u_pred|| * L^2 / ||u_pred||`` via autograd (float64).

    The 2D drivers' strong-form convention lifted to 3D: two cases of the
    requested split spec, 32 interior queries, float64.  The exact
    point-charge solution and any harmonic model (the exact solid-angle
    member, the dense oracle) score float-noise zero, so this diagnoses
    *harmonicity* of the prediction, not accuracy.  The fused
    scaled-dot-product-attention backends (used by the Transolver arms) do
    not implement double backward, so the forward is recorded with the
    numerically equivalent math backend -- a no-op for the other arms.
    """

    from torch.nn.attention import SDPBackend, sdpa_kernel

    model.eval()
    residuals = []
    for case in range(2):
        sample = _make_sample(
            seed + case, {**spec, "n_query": 32}, torch.float64, device
        )
        model_fp64 = model.double()
        points = sample.domain.interior.points.clone().requires_grad_(True)
        domain = DomainMesh(
            interior=Mesh(points=points),
            boundaries=dict(sample.domain.boundaries.items()),
            global_data=sample.domain.global_data,
        )
        with sdpa_kernel([SDPBackend.MATH]):
            u = model_fp64(domain).point_data["potential"]
        laplacian = torch.zeros_like(u)
        if u.grad_fn is not None:
            (gradient,) = torch.autograd.grad(
                u.sum(), points, create_graph=True, allow_unused=True
            )
            if gradient is not None:
                for component in range(3):
                    (second,) = torch.autograd.grad(
                        gradient[:, component].sum(),
                        points,
                        create_graph=True,
                        allow_unused=True,
                    )
                    if second is not None:
                        laplacian = laplacian + second[:, component]
        length = sample.domain.global_data["reference_length"].reshape(())
        residual = laplacian.detach() * length**2
        residuals.append(
            float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(u.detach()).clamp_min(1.0e-30)
            )
        )
    return sum(residuals) / len(residuals)


@torch.no_grad()
def max_principle_violation(
    model: nn.Module,
    *,
    spec: dict,
    seed: int,
    device: torch.device | str = "cpu",
) -> float:
    """Sampled boundary-range violation of the harmonic maximum principle.

    Uses the shared :func:`metrics.sampled_boundary_range_violation`
    convention -- the prediction's excursion beyond the sampled Dirichlet
    range over *all* boundaries (on shells the principle bounds the interior
    by the range over both boundaries together), normalized by sampled
    boundary RMS.  Licensed on the all-Dirichlet suite; a discretization-
    aware proxy rather than a certified continuous enclosure.  Two cases per
    split spec, float64.
    """

    from metrics import sampled_boundary_range_violation

    model_fp64 = model.double()
    model_fp64.eval()
    violations = []
    for case in range(2):
        sample = _make_sample(seed + case, spec, torch.float64, device)
        prediction = model_fp64(sample.domain).point_data["potential"]
        boundary_values = torch.cat(
            [
                sample.domain.boundaries[name].cell_data["boundary_value"]
                for name in sorted(sample.domain.boundaries.keys())
            ]
        )
        violations.append(
            float(sampled_boundary_range_violation(prediction, boundary_values))
        )
    return sum(violations) / len(violations)


def fidelity_metrics(
    model: nn.Module,
    *,
    splits: dict,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split, the strong-form residual (:func:`pde_residual`) and the
    sampled maximum-principle violation (:func:`max_principle_violation`),
    both deliberately subsampled (two cases per split; 32 queries for the
    autograd residual) so the block stays cheap relative to training.
    """

    return {
        "pde_residual": {
            name: pde_residual(model, spec=spec, seed=seed, device=device)
            for name, spec in sorted(splits.items())
        },
        "pde_residual_note": (
            "||lap u|| L^2 / ||u|| via float64 autograd at 32 interior "
            "points on two cases per split; harmonicity of the prediction, "
            "not accuracy -- the exact point-charge solution and any "
            "harmonic model score ~0"
        ),
        "max_principle_violation": {
            name: max_principle_violation(model, spec=spec, seed=seed, device=device)
            for name, spec in sorted(splits.items())
        },
        "max_principle_note": (
            "sampled boundary-range violation normalized by boundary RMS "
            "(metrics.sampled_boundary_range_violation) over all Dirichlet "
            "boundaries, two cases per split; a discretization-aware proxy, "
            "not a certified continuous enclosure"
        ),
    }


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
    model = _build_model(model_name).to(torch.device(device))

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

    n_parameters = sum(p.numel() for p in model.parameters())
    report = {
        "model": model_name,
        "seed": seed,
        "steps": steps if model_name != "oracle" else 0,
        "parameters": n_parameters,
        "elapsed_seconds": time.time() - start,
        "history": history,
        "splits": evaluate_splits(
            model, eval_seed=97_000_037, n_cases=eval_cases, dtype=dtype, device=device
        ),
        "fidelity": fidelity_metrics(
            model, splits=SPLITS_3D, seed=83_000_019, device=device
        ),
    }
    # Full state is human-readable only for the few-parameter physics arms;
    # the MeshTransformer arms would dump ~1e5 floats into the JSON report.
    if n_parameters <= 64:
        report["state"] = {k: v.tolist() for k, v in model.state_dict().items()}
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
            "solid_angle_bie",
            "oracle",
            "mesh_transformer_kernel",
            "mesh_transformer_kernel_nomlp",
            "mesh_transformer_kernel_nopoly",
            "mesh_transformer_kernel_singonly",
            "mesh_transformer_kernel_singpair",
            "mesh_transformer_kernel_singpair_h1",
            "mesh_transformer_kernel_singonly_scalar",
            "mesh_transformer_kernel_singonly_scalar_wide",
            "transolver_intree_matched",
            "transolver_intree_native",
        ),
    )
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
    summary_keys = ("model", "parameters", "splits", "state")
    print(json.dumps({k: result[k] for k in summary_keys if k in result}))
