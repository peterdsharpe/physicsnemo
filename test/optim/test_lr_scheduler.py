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

"""Schedule endpoint and interrupted-training checks."""

import copy

import pytest
import torch
from timm.optim import Lion

from physicsnemo.optim.lr_scheduler import WarmupCosine


@pytest.mark.parametrize("warmup", [0, 4])
def test_schedule_endpoints_and_resume(warmup, device):
    """Preserve scheduled learning rates and updates across checkpoint resume."""
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0], device=device))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.05, foreach=True)
    scheduler = WarmupCosine(opt, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    rates = []
    for step in range(12):
        rates.append(opt.param_groups[0]["lr"])
        p.grad = torch.tensor([(-1.0) ** step, 0.3], device=device)
        opt.step()
        scheduler.step()
        if step == 5:
            checkpoint = (
                p.detach().clone(),
                copy.deepcopy(opt.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
    assert rates[0] == pytest.approx(0.1 / warmup if warmup else 0.1)
    assert rates[warmup - 1 if warmup else 0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.001)
    resumed = torch.nn.Parameter(checkpoint[0])
    opt2 = Lion([resumed], lr=0.1, betas=(0.9, 0.99), weight_decay=0.05, foreach=True)
    scheduler2 = WarmupCosine(opt2, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    opt2.load_state_dict(checkpoint[1])
    scheduler2.load_state_dict(checkpoint[2])
    for step in range(6, 12):
        assert opt2.param_groups[0]["lr"] == rates[step]
        resumed.grad = torch.tensor([(-1.0) ** step, 0.3], device=device)
        opt2.step()
        scheduler2.step()
    torch.testing.assert_close(p, resumed, rtol=0, atol=0)
    torch.testing.assert_close(
        opt.state[p]["exp_avg"], opt2.state[resumed]["exp_avg"], rtol=0, atol=0
    )
