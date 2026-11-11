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

"""Learning-rate schedules stepped once per optimizer update."""

from __future__ import annotations

import math

import torch


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Per-update linear warmup followed by cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose initial group learning rates define the peak rates.
    total_steps : int
        Number of optimizer updates, including warmup; must be at least two.
    warmup_steps : int
        Number of warmup updates, from zero to ``total_steps - 1``.
    min_lr : float, optional, default=1e-6
        Nonnegative final learning rate, at most every group's peak rate.
    last_epoch : int, optional, default=-1
        Index of the last scheduler step, following PyTorch's scheduler API.

    Notes
    -----
    Call ``step()`` after each successful optimizer update. With warmup, the
    first update uses peak_lr / warmup_steps and the last warmup update uses
    peak_lr. Without warmup, the first update uses peak_lr. The final planned
    update uses min_lr, and later updates remain at min_lr. Resume by restoring
    scheduler state instead of recomputing a shorter schedule.
    """

    def __init__(
        self, optimizer, total_steps, warmup_steps, min_lr=1e-6, last_epoch=-1
    ):
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < 2
        ):
            raise ValueError("total_steps must be an integer >= 2")
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or not 0 <= warmup_steps < total_steps
        ):
            raise ValueError("warmup_steps must be an integer in [0, total_steps)")
        if not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError("min_lr must be finite and nonnegative")
        if any(min_lr > group["lr"] for group in optimizer.param_groups):
            raise ValueError("min_lr must not exceed any group's peak learning rate")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """Return each group learning rate for the next optimizer update."""
        step = self.last_epoch
        if step < self.warmup_steps:
            return [lr * (step + 1) / self.warmup_steps for lr in self.base_lrs]
        if self.warmup_steps:
            progress = (step + 1 - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
        else:
            progress = step / (self.total_steps - 1)
        factor = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return [self.min_lr + (lr - self.min_lr) * factor for lr in self.base_lrs]
