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

r"""Mixed Dirichlet/Neumann boundary conditions through the MeshTransformer.

Pre-registered hypothesis (recorded before any mixed-BC training run)
=====================================================================

For all-Dirichlet interior Laplace problems the double-layer density solves
a :math:`\tfrac12 I`-dominated second-kind Fredholm system, so boundary
self-attention should contribute little: a 0-layer encoder is expected to
match a 2-layer one (this is what the all-Dirichlet study measured).  For
MIXED Dirichlet/Neumann problems the unknown Cauchy data on each patch
depends *globally* on the prescribed data on the other patches -- the
Calderon solve is genuinely nonlocal -- so encoder depth should matter:
``operator_layers=2`` is expected to beat ``operator_layers=0``.  If depth
still does not matter on mixed BCs, the encoder is dead weight across the
problem class and the architecture should be collapsed.

"Encoder depth" here means ``operator_layers`` (the geometric boundary
self-attention stack), swept 0/1/2 across otherwise-identical arms; the
drive self-attention depth is held fixed at ``drive_layers=1`` (the
all-Dirichlet study's setting), so boundary-to-boundary *data* coupling has
the same single-layer budget in every arm.

What the 3D suite's mixed regime actually provides
==================================================

``laplace3d.build_laplace3d_sample(bc_regime="mixed")`` assigns BC types
**per boundary**, not per cell: the sorted boundary names get Dirichlet at
index 0 and Neumann elsewhere.  Consequences, verified against the
generator:

- ``sphere`` and ``star`` tiers have a single boundary (``outer``) and fall
  back to **all-Dirichlet** (documented in ``laplace3d.py``: "single-boundary
  tiers fall back to all-Dirichlet ... a per-cell hemisphere split is future
  work").  These tiers carry *no Neumann data at all* under
  ``bc_regime="mixed"``, so the originally requested ``sphere``/``star``/
  ``star_unseen_modes`` mixed splits would be all-Dirichlet and are omitted.
- ``shell`` is the only genuinely mixed cell of the suite: the ``inner``
  boundary carries ``cell_data["boundary_value"]`` (Dirichlet trace) and the
  ``outer`` boundary carries ``cell_data["boundary_flux"]``
  (:math:`L\,\partial_n u`, dimensionless).  Every boundary is homogeneous
  in BC type, so the "split cells by BC type" wrapper reduces to routing
  whole boundaries by their cell-data key; no approximation is introduced.

Splits are therefore shell-based: ``shell_mixed`` (the suite's default
resolution) for train/validation/eval, plus ``shell_mixed_fine`` (one more
icosphere subdivision) as a resolution-OOD eval split.  Geometry varies per
seed through random radius, center, orientation, and cavity radius.

All-Dirichlet samples and the sanity control
============================================

:class:`~physicsnemo.experimental.nn.MeshTransformer` requires domain
boundary names to match its declared schema exactly and rejects empty
declared boundaries ("must contain at least one cell").  An all-Dirichlet
sample therefore *cannot* be presented to the two-boundary
``{"dirichlet", "neumann"}`` schema with a single model/parameter set, and
this study rejects such samples with a clear error instead of silently
re-schematizing (documented choice: "skip such samples").  The all-Dirichlet
sanity control lives in ``laplace3d_study.py`` (same dictionary, same dims,
single ``dirichlet`` boundary schema).

Robin BCs are OUT OF SCOPE for this iteration and are rejected explicitly.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from laplace3d import build_laplace3d_sample
from laplace3d_study import pde_residual
from torch import nn

from physicsnemo.experimental.nn import MeshTransformer
from physicsnemo.mesh import DomainMesh, Mesh

#: Generator cell-data key carrying the Dirichlet trace (dimensionless u).
DIRICHLET_KEY = "boundary_value"
#: Generator cell-data key carrying the Neumann flux (dimensionless L du/dn).
NEUMANN_KEY = "boundary_flux"
#: Generator cell-data keys of the (out-of-scope) Robin regime.
ROBIN_KEYS = ("robin_value", "robin_beta")


def classify_boundary_bc(mesh: Mesh) -> str:
    """Classify one generator boundary as ``dirichlet`` or ``neumann``.

    The 3D suite attaches exactly one BC data key per boundary; every cell
    of a boundary shares that BC type.  Robin data and ambiguous or missing
    data are rejected rather than guessed.
    """

    keys = set(mesh.cell_data.keys())
    if robin := keys.intersection(ROBIN_KEYS):
        raise NotImplementedError(
            f"Robin boundary data {sorted(robin)} is out of scope for the "
            "mixed Dirichlet/Neumann study; generate samples with "
            "bc_regime='mixed'"
        )
    has_dirichlet = DIRICHLET_KEY in keys
    has_neumann = NEUMANN_KEY in keys
    if has_dirichlet and has_neumann:
        raise ValueError(
            f"Boundary carries both {DIRICHLET_KEY!r} and {NEUMANN_KEY!r}; "
            "the 3D suite attaches exactly one BC type per boundary"
        )
    if has_dirichlet:
        return "dirichlet"
    if has_neumann:
        return "neumann"
    raise ValueError(
        f"Boundary carries neither {DIRICHLET_KEY!r} nor {NEUMANN_KEY!r} "
        f"cell data (found {sorted(keys)})"
    )


def split_boundaries_by_bc_type(domain: DomainMesh) -> DomainMesh:
    """Re-present a mixed-regime domain under the named-BC-boundary schema.

    Routes each generator boundary (named by geometric role: ``outer``,
    ``inner``) into the declared ``dirichlet`` / ``neumann`` boundaries by
    its BC cell-data key, keeping only that key.  Boundaries of the same BC
    type are concatenated with :meth:`Mesh.merge` in sorted-name order,
    which preserves per-cell geometry (hence areas and normals) and data
    exactly.

    Raises
    ------
    ValueError
        If the sample has no Dirichlet cells (ill-posed for the interior
        problem; the generator guarantees at least one Dirichlet boundary)
        or no Neumann cells (all-Dirichlet fallback samples cannot be
        presented to the two-boundary schema because ``MeshTransformer``
        rejects empty declared boundaries; see the module docstring).
    NotImplementedError
        If any boundary carries Robin data (out of scope).
    """

    parts: dict[str, list[Mesh]] = {"dirichlet": [], "neumann": []}
    for name in sorted(domain.boundaries.keys()):
        mesh = domain.boundaries[name]
        bc_type = classify_boundary_bc(mesh)
        data_key = DIRICHLET_KEY if bc_type == "dirichlet" else NEUMANN_KEY
        parts[bc_type].append(
            Mesh(
                points=mesh.points,
                cells=mesh.cells,
                cell_data={data_key: mesh.cell_data[data_key]},
            )
        )
    if not parts["dirichlet"]:
        raise ValueError(
            "Mixed sample has no Dirichlet cells; the interior problem would "
            "be nonunique and the generator is expected to guarantee at "
            "least one Dirichlet boundary"
        )
    if not parts["neumann"]:
        raise ValueError(
            "Sample is all-Dirichlet (the generator's documented fallback "
            "for single-boundary tiers under bc_regime='mixed'); it cannot "
            "be presented to the two-boundary {'dirichlet', 'neumann'} "
            "schema because MeshTransformer rejects empty declared "
            "boundaries.  Use the shell tier for genuinely mixed problems "
            "and laplace3d_study.py for the all-Dirichlet control."
        )
    return DomainMesh(
        interior=domain.interior,
        boundaries={
            "dirichlet": Mesh.merge(parts["dirichlet"]),
            "neumann": Mesh.merge(parts["neumann"]),
        },
        global_data=domain.global_data,
    )


class MixedBoundaryModel(nn.Module):
    """Route generator boundaries by BC type into the named-BC schema.

    The wrapped :class:`MeshTransformer` declares ``dirichlet`` and
    ``neumann`` boundaries; BC type is conveyed to the model by boundary
    identity (each declared boundary gets its own one-hot channel and its
    own drive field: ``boundary_value`` on Dirichlet cells and
    ``boundary_flux`` on Neumann cells, the undeclared channel packed as
    zero).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, domain: DomainMesh) -> Mesh:
        return self.model(split_boundaries_by_bc_type(domain))


def build_mesh_transformer_mixed(*, operator_layers: int = 2) -> MixedBoundaryModel:
    """Singpair-dictionary MeshTransformer with named Dirichlet/Neumann BCs.

    Identical dimension settings to ``laplace3d_study.build_mesh_transformer_3d``
    (default capacity: not scalar-only, not wide) and the "singpair" kernel
    dictionary of the all-Dirichlet study -- exactly two exact members
    (double layer plus single layer), no polynomial and no MLP smooth
    members.  Only ``operator_layers`` (encoder depth) varies across arms.
    The single-layer member matters here: shells are multiply connected, and
    a pure double-layer dictionary cannot carry the net-flux component.
    """

    return MixedBoundaryModel(
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"potential": 0},
            boundary_field_ranks={
                "dirichlet": {
                    "operator": {},
                    "drive": {DIRICHLET_KEY: 0},
                },
                "neumann": {
                    "operator": {},
                    "drive": {NEUMANN_KEY: 0},
                },
            },
            global_field_ranks={"operator": {}, "drive": {}},
            reference_length_key="reference_length",
            field_mode="linear",
            query_decoder="kernel",
            kernel_mlp_members=0,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            operator_scalar_dim=32,
            operator_vector_dim=8,
            drive_scalar_dim=48,
            drive_vector_dim=12,
            operator_layers=operator_layers,
            drive_layers=1,
            query_layers=1,
            heads=4,
            scalar_rank=12,
            vector_rank=4,
            query_chunk_size=65536,
            attention_chunk_size=65536,
        )
    )


#: Registry of study arms: encoder-depth sweep at fixed dictionary/capacity.
MODEL_ARMS = {
    "mt_singpair_mixed": 2,
    "mt_singpair_mixed_enc1": 1,
    "mt_singpair_mixed_enc0": 0,
}


def _build_model(model_name: str) -> nn.Module:
    if model_name in MODEL_ARMS:
        return build_mesh_transformer_mixed(operator_layers=MODEL_ARMS[model_name])
    raise ValueError(f"unknown model {model_name!r}")


# Only the shell tier of the 3D suite is genuinely mixed (see module
# docstring); sphere/star under bc_regime="mixed" are all-Dirichlet by the
# generator's documented fallback and are deliberately absent.
MIXED_SPLITS = {
    "shell_mixed": {"tier": "shell", "bc_regime": "mixed"},
    "shell_mixed_fine": {
        "tier": "shell",
        "bc_regime": "mixed",
        "subdivisions": 3,
    },
}


def _make_sample(seed: int, spec: dict, dtype: torch.dtype, device="cpu"):
    return build_laplace3d_sample(seed, dtype=dtype, device=device, **spec)


def fidelity_metrics(
    model: nn.Module,
    *,
    splits: dict,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict:
    """Operator-fidelity block appended (additively) to the report JSON.

    Per split, the strong-form residual under the all-Dirichlet study's
    convention (:func:`laplace3d_study.pde_residual`: float64 autograd, two
    cases, 32 interior queries -- the mixed samples are the same generator's
    shells, so the residual convention transfers verbatim).  No
    maximum-principle violation is licensed here: the outer boundary carries
    Neumann flux data, so the prescribed Dirichlet trace alone does not
    bound the interior solution and no data-licensed range enclosure exists.
    """

    return {
        "pde_residual": {
            name: pde_residual(model, spec=spec, seed=seed, device=device)
            for name, spec in sorted(splits.items())
        },
        "pde_residual_note": (
            "||lap u|| L^2 / ||u|| via float64 autograd at 32 interior "
            "points on two cases per split; harmonicity of the prediction, "
            "not accuracy -- the exact point-charge solution scores ~0"
        ),
        "max_principle_violation": None,
        "max_principle_note": (
            "n/a: the mixed regime prescribes Neumann flux on the outer "
            "boundary, so the known Dirichlet trace does not bound the "
            "interior solution and no data-licensed violation metric exists"
        ),
    }


def _relative_l2(prediction, target):
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


@torch.no_grad()
def evaluate_splits(
    model,
    *,
    eval_seed: int,
    n_cases: int,
    dtype,
    device="cpu",
    splits: dict | None = None,
):
    splits = MIXED_SPLITS if splits is None else splits
    model.eval()
    report = {}
    for index, (name, spec) in enumerate(sorted(splits.items())):
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
    splits: dict | None = None,
):
    splits = MIXED_SPLITS if splits is None else splits
    if "shell_mixed" not in splits:
        raise ValueError("splits must contain the 'shell_mixed' training split")
    torch.manual_seed(seed)
    dtype = torch.float32
    model = _build_model(model_name).to(torch.device(device))

    history = []
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-6)
    best_val, best_state = float("inf"), None
    train_spec = splits["shell_mixed"]
    for step in range(1, steps + 1):
        model.train()
        sample = _make_sample(seed + 104_729 * step, train_spec, dtype, device)
        prediction = model(sample.domain).point_data["potential"]
        loss = torch.sum((prediction - sample.target).square()) / torch.sum(
            sample.target.square()
        ).clamp_min(1.0e-30)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == steps:
            validation = evaluate_splits(
                model,
                eval_seed=71_000_011,
                n_cases=3,
                dtype=dtype,
                device=device,
                splits=splits,
            )
            score = validation["shell_mixed"]
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
        "steps": steps,
        "parameters": n_parameters,
        "elapsed_seconds": time.time() - start,
        "history": history,
        "splits": evaluate_splits(
            model,
            eval_seed=97_000_037,
            n_cases=eval_cases,
            dtype=dtype,
            device=device,
            splits=splits,
        ),
        "fidelity": fidelity_metrics(
            model, splits=splits, seed=83_000_019, device=device
        ),
    }
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ARMS))
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
    summary_keys = ("model", "parameters", "splits")
    print(json.dumps({k: result[k] for k in summary_keys if k in result}))
