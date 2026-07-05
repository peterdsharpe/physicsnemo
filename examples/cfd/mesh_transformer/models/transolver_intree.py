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

r"""In-tree Transolver baseline for the Laplace boundary-to-interior benchmark.

:class:`physicsnemo.models.transolver.Transolver` (with
``structured_shape=None`` and ``unified_pos=False``) is PhysicsNeMo's own port
of the official irregular-mesh Transolver.  This module wires it onto the
*identical* adaptation contract as the hand-rolled port in :mod:`transolver`:
both arms consume the same shared boundary+query token sequence produced by
:func:`transolver.build_token_sequence` (coordinates as the ``space_dim=2``
embedding channels, the 5 function channels described in that module's
docstring) and read the scalar prediction at the query tokens.  The in-tree
``forward`` concatenates ``(embedding, fx)`` -- coordinates first, then
function features -- which is exactly the ported model's preprocess layout.

Residual differences vs the ported arm (and the official repository) at
identical hyperparameters, all inside
``physicsnemo/nn/module/physics_attention.py``:

* **Slice-temperature clamp.**  The in-tree implementation clamps the
  learnable per-head slice temperature to ``[0.5, 5]`` before the softmax
  (``_compute_slices_from_projections``); the official code and the ported
  arm leave it unclamped.  Both initialize the temperature at 0.5, so the
  arms start identically, but during training the in-tree temperature cannot
  move below 0.5 (sharper-than-init slice assignments are unreachable).
* **Slice-pooling epsilon.**  In-tree adds ``1e-2`` to the per-slice weight
  sum and normalizes the weights *before* pooling; the official code (and the
  ported arm) pool first and divide by ``sum + 1e-5``.  Same math up to the
  epsilon; the larger value slightly damps sparsely populated slices.
* **Fused QKV.**  One ``Linear(dim_head, 3 * dim_head, bias=False)`` replaces
  the three separate no-bias projections -- identical parameter count and
  math, different initialization draws.
* **SDPA.**  Slice-token attention runs through
  ``torch.nn.functional.scaled_dot_product_attention`` instead of explicit
  softmax attention; numerically equivalent, except that the SDPA path omits
  attention-weight dropout -- inert here because every preset uses
  ``dropout=0``.
* **No placeholder parameter.**  The official model allocates a
  ``placeholder`` parameter of size ``hidden_dim`` that is unused whenever
  function features are supplied (``fun_dim > 0``, always true on this
  benchmark); the in-tree model omits it, so each preset is exactly
  ``hidden_dim`` parameters smaller than the corresponding ported preset.

Transformer Engine is disabled (``use_te=False``) so the layer stack (plain
``nn.LayerNorm`` / ``nn.Linear``) matches the ported arm and the official
repository.

Capacity presets reuse :data:`transolver.TRANSOLVER_PRESETS` verbatim:
``"intree_matched"`` (hidden 64, 4 heads, 3 blocks, ``M=64``, ``mlp_ratio=2``)
has 102,989 parameters -- the ported arm's 103,053 minus the 64-parameter
inert placeholder -- and ``"intree_native"`` (hidden 256, 8 heads, 8 blocks,
``M=32``, ``mlp_ratio=1``) has 2,809,153 parameters (2,809,409 minus 256).
"""

from __future__ import annotations

from dataclasses import asdict

from models import _prediction_mesh
from torch import nn
from transolver import TRANSOLVER_PRESETS, TransolverPreset, build_token_sequence

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.models.transolver import Transolver


class InTreeTransolverLaplaceAdapter(nn.Module):
    """Run the in-tree Transolver on the exact conformal-Laplace protocol.

    Boundary cells and interior queries form one shared token sequence built
    by :func:`transolver.build_token_sequence` (the same adaptation contract
    as the ported arm); the prediction is read at the query tokens.

    ``space_dim`` lifts the identical contract to other spatial dimensions
    (the 3D Laplace suite uses ``space_dim=3``: triangle centroids as token
    positions; unit normal, dimensionless triangle area, Dirichlet value, and
    boundary/query indicator as the 6 function channels).
    ``global_feature_keys`` appends the named ``domain.global_data`` scalars
    as constant extra function channels on every token (the screened-Laplace
    suite passes ``("screening",)`` so the dimensionless :math:`\\tilde\\kappa`
    reaches the model); see :func:`transolver.build_token_sequence` for why
    this is the steel-man injection.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float = 0.0,
        space_dim: int = 2,
        global_feature_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.global_feature_keys = tuple(global_feature_keys)
        self.model = Transolver(
            # Normal (space_dim), measure, value, indicator, plus one constant
            # channel per declared global scalar.
            functional_dim=space_dim + 3 + len(self.global_feature_keys),
            embedding_dim=space_dim,
            out_dim=1,
            n_layers=layers,
            n_hidden=hidden_dim,
            n_head=heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            structured_shape=None,
            unified_pos=False,
            use_te=False,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        parameter = next(self.model.parameters())
        coordinates, features, n_boundary = build_token_sequence(
            domain,
            device=parameter.device,
            dtype=parameter.dtype,
            global_feature_keys=self.global_feature_keys,
        )
        # The in-tree forward concatenates (embedding, fx): coordinates first,
        # then function features -- the ported model's exact preprocess layout.
        output = self.model(features, embedding=coordinates)  # (1, S + Q, 1)
        potential = output[0, n_boundary:, 0]
        return _prediction_mesh(domain, potential)


INTREE_TRANSOLVER_PRESETS: dict[str, TransolverPreset] = {
    # 102,989 parameters: the ported "matched" preset minus its inert
    # 64-parameter placeholder; still within 1.3% of the 104,261-parameter
    # mesh_transformer_kernel_singonly reference arm.
    "intree_matched": TRANSOLVER_PRESETS["matched"],
    # 2,809,153 parameters: the ported "native" preset minus its inert
    # 256-parameter placeholder.
    "intree_native": TRANSOLVER_PRESETS["native"],
}


def build_transolver_intree(
    preset: str,
    *,
    space_dim: int = 2,
    global_feature_keys: tuple[str, ...] = (),
) -> InTreeTransolverLaplaceAdapter:
    """Construct the benchmark-adapted in-tree Transolver at one capacity.

    ``space_dim`` and ``global_feature_keys`` adapt the token contract to the
    sibling suites (3D Laplace, screened Laplace) without touching the model:
    each extra input channel only widens the first preprocess linear, adding
    ``2 * hidden_dim`` parameters (e.g. the screened ``intree_matched`` arm
    has 103,117 parameters and the 3D arm 103,245, vs 102,989 for the 2D
    bank).
    """

    try:
        settings = INTREE_TRANSOLVER_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown in-tree Transolver preset {preset!r}; "
            f"expected one of {tuple(INTREE_TRANSOLVER_PRESETS)}"
        ) from None
    return InTreeTransolverLaplaceAdapter(
        **asdict(settings),
        space_dim=space_dim,
        global_feature_keys=global_feature_keys,
    )


__all__ = [
    "INTREE_TRANSOLVER_PRESETS",
    "InTreeTransolverLaplaceAdapter",
    "build_transolver_intree",
]
