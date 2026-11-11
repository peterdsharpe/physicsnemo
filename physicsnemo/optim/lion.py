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

"""Lion optimizer with decoupled weight decay."""

from __future__ import annotations

import math

import torch


class Lion(torch.optim.Optimizer):
    """Sign-momentum optimizer with decoupled weight decay.

    Implements Chen et al., *Symbolic Discovery of Optimization Algorithms*,
    Algorithm 2 (https://arxiv.org/abs/2302.06675).

    Parameters
    ----------
    params : iterable of torch.Tensor or dict
        Parameters to optimize, or dictionaries defining parameter groups.
    lr : float, optional, default=5e-5
        Nonnegative learning rate.
    betas : tuple[float, float], optional, default=(0.9, 0.99)
        Update and momentum averaging coefficients, each in [0, 1).
    weight_decay : float, optional, default=0.05
        Nonnegative decoupled weight decay.

    Notes
    -----
    Parameters are normally maintained in float32 under autocast. Sparse
    gradients are unsupported.
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.99), weight_decay=0.05):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one Lion update and return the optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta_update, beta_momentum = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                direction = (
                    momentum.mul(beta_update)
                    .add(gradient, alpha=1 - beta_update)
                    .sign_()
                )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                # Keep scheduled LR in tensor arithmetic: Inductor can capture
                # a stale scalar when a changing float is passed as add_'s alpha.
                parameter.add_(direction * -group["lr"])
                momentum.mul_(beta_momentum).add_(gradient, alpha=1 - beta_momentum)
        return loss
