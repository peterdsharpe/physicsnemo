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

"""Checkpointable exponential moving averages of model parameters."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


class ModelEMA:
    """Average parameters after successful updates; copy the latest buffers.

    Parameters
    ----------
    model : torch.nn.Module
        Module to average. Pass the original module before DDP or compilation
        so parameter names are independent of wrappers. Construct the average
        after moving the model to its training device.
    decay : float
        Exponential decay coefficient in [0, 1).

    Notes
    -----
    Call ``update()`` after successful optimizer updates. Parameters accumulate
    in at least float32; buffers are copied from the latest model state.
    ``average_parameters()`` temporarily copies averages into existing storage,
    preserving optimizer references, then restores raw weights even on error.
    Save and restore ``state_dict()`` alongside the model and optimizer state.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        if isinstance(decay, bool) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.model = model
        self.decay = float(decay)
        self.updates = 0
        self.parameters = {
            name: p.detach()
            .to(
                dtype=torch.float32
                if p.dtype in (torch.float16, torch.bfloat16)
                else p.dtype
            )
            .clone()
            for name, p in model.named_parameters()
        }
        self.buffers = {name: b.detach().clone() for name, b in model.named_buffers()}

    @torch.no_grad()
    def update(self) -> None:
        """Accumulate current parameters and copy buffers after an optimizer step."""
        for name, p in self.model.named_parameters():
            self.parameters[name].lerp_(
                p.detach().to(self.parameters[name]), 1 - self.decay
            )
        for name, b in self.model.named_buffers():
            self.buffers[name].copy_(b)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return detached CPU copies of the EMA accumulators and update count."""
        return {
            "decay": self.decay,
            "updates": self.updates,
            "parameters": {
                k: v.detach().cpu().clone() for k, v in self.parameters.items()
            },
            "buffers": {k: v.detach().cpu().clone() for k, v in self.buffers.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate and restore EMA state without replacing accumulator storage."""
        if state.get("decay") != self.decay:
            raise ValueError("EMA decay differs from the checkpoint")
        updates = state.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("Invalid EMA update count")
        for group in ("parameters", "buffers"):
            expected = getattr(self, group)
            saved = state.get(group, {})
            if expected.keys() != saved.keys():
                raise ValueError(f"EMA {group} keys differ from the checkpoint")
            for name, value in expected.items():
                if value.shape != saved[name].shape:
                    raise ValueError(f"EMA shape mismatch for {name}")
        with torch.no_grad():
            for group in ("parameters", "buffers"):
                for name, value in getattr(self, group).items():
                    value.copy_(state[group][name])
        self.updates = updates

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily expose EMA weights for validation or model export."""
        live = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        backup = {name: value.detach().clone() for name, value in live.items()}
        averaged = self.parameters | self.buffers
        try:
            with torch.no_grad():
                for name, value in live.items():
                    value.copy_(averaged[name])
            yield
        finally:
            with torch.no_grad():
                for name, value in live.items():
                    value.copy_(backup[name])
