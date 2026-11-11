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

"""EMA arithmetic, storage preservation, and checkpoint validation."""

import pytest
import torch

from physicsnemo.optim import ModelEMA
from physicsnemo.utils import load_checkpoint, save_checkpoint


def test_ema_values_buffers_and_exception_restore(device):
    """Average weights, copy buffers, and restore raw storage after errors."""
    model = torch.nn.Linear(1, 1, bias=False, device=device)
    model.register_buffer("counter", torch.tensor(1, device=device))
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ModelEMA(model, 0.75)
    with torch.no_grad():
        model.weight.fill_(10)
        model.counter.fill_(7)
    ema.update()
    assert ema.updates == 1
    torch.testing.assert_close(
        ema.parameters["weight"], torch.tensor([[4.0]], device=device)
    )
    original_pointer = model.weight.data_ptr()
    with pytest.raises(RuntimeError, match="validation failed"):
        with ema.average_parameters():
            torch.testing.assert_close(
                model.weight, torch.tensor([[4.0]], device=device)
            )
            assert model.counter.item() == 7
            assert model.weight.data_ptr() == original_pointer
            raise RuntimeError("validation failed")
    torch.testing.assert_close(model.weight, torch.tensor([[10.0]], device=device))
    assert model.weight.data_ptr() == original_pointer


def test_ema_accumulates_reduced_precision_parameters_in_float32(device):
    """Retain small EMA increments when model weights use reduced precision."""
    model = torch.nn.Linear(1, 1, bias=False, device=device).bfloat16()
    with torch.no_grad():
        model.weight.fill_(1)
    ema = ModelEMA(model, 0.9999)
    with torch.no_grad():
        model.weight.fill_(2)
    ema.update()
    assert ema.parameters["weight"].dtype == torch.float32
    assert 1.0 < ema.parameters["weight"].item() < 1.001


def test_ema_rejects_changed_decay_and_shape(device):
    """Reject incompatible EMA checkpoint decay or parameter dimensions."""
    ema = ModelEMA(torch.nn.Linear(2, 1, device=device), 0.9)
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(torch.nn.Linear(2, 1, device=device), 0.99).load_state_dict(
            ema.state_dict()
        )
    with pytest.raises(ValueError, match="shape"):
        ModelEMA(torch.nn.Linear(3, 1, device=device), 0.9).load_state_dict(
            ema.state_dict()
        )


def test_ema_export_loads_as_inference_checkpoint_without_mutating_raw(
    tmp_path, device
):
    """Export averaged inference weights while preserving raw training weights."""
    model = torch.nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ModelEMA(model, 0.5)
    with torch.no_grad():
        model.weight.fill_(4)
    ema.update()
    with ema.average_parameters():
        save_checkpoint(tmp_path, models=model, epoch=5)
    restored = torch.nn.Linear(1, 1, bias=False, device=device)
    assert load_checkpoint(tmp_path, models=restored, device=device) == 5
    torch.testing.assert_close(restored.weight, torch.tensor([[3.0]], device=device))
    torch.testing.assert_close(model.weight, torch.tensor([[4.0]], device=device))
