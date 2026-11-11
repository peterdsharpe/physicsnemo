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

"""Recipe optimizer construction and compiled scheduling checks."""

import copy
from functools import partial

import pytest
import torch
from timm.optim import Lion

from physicsnemo.optim.lr_scheduler import WarmupCosine


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "target,shape,with_bias,backend",
    [
        ("torch.optim.AdamW", (2,), False, "eager"),
        (None, (2,), False, "eager"),
        (None, (2, 2), False, "eager"),
        (None, (2, 2), True, "eager"),
        ("timm.optim.Lion", (2,), False, "inductor"),
    ],
)
def test_compiled_optimizer_scheduler_updates_match_eager(
    monkeypatch, target, shape, with_bias, backend, device
):
    """Construct a real scheduler around compiled steps and preserve its updates."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from utils import build_optimizer

    # Eager covers scheduler wrapping cheaply; real Inductor catches stale
    # scalar learning rates that Dynamo's eager backend cannot expose.
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend=backend))
    config = {"lr": 0.1, "weight_decay": 0.01}
    if target is not None:
        config["_target_"] = target
    if target == "timm.optim.Lion":
        config["betas"] = [0.9, 0.99]
        # The single-tensor CPU path can retain a stale LR under Inductor.
        config["foreach"] = True
    cfg = OmegaConf.create({"training": {"optimizer": config}})
    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.ones(shape, device=device))
    )
    if with_bias:
        model.register_parameter(
            "bias", torch.nn.Parameter(torch.ones(2, device=device))
        )
    reference = copy.deepcopy(model)
    optimizer = build_optimizer(model, cfg, compile_optimizer=True)
    if target is not None:
        expected_type = Lion if target == "timm.optim.Lion" else torch.optim.AdamW
        assert isinstance(optimizer, expected_type)
    reference_optimizer = build_optimizer(reference, cfg)
    scheduler = instantiate(
        {
            "_target_": "physicsnemo.optim.lr_scheduler.WarmupCosine",
            "total_steps": 10,
            "warmup_steps": 2,
        },
        optimizer=optimizer,
    )
    reference_scheduler = WarmupCosine(
        reference_optimizer, total_steps=10, warmup_steps=2
    )
    for _ in range(4):
        for current, opt, schedule in (
            (model, optimizer, scheduler),
            (reference, reference_optimizer, reference_scheduler),
        ):
            opt.zero_grad()
            sum(p.square().sum() for p in current.parameters()).backward()
            opt.step()
            schedule.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
    assert scheduler.last_epoch == 4
    assert not torch.equal(model.weight, torch.ones_like(model.weight))
