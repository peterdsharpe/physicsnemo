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

"""Hand-derived Lion updates."""

import torch

from physicsnemo.optim import Lion


def test_lion_hand_computed_updates(device):
    """Match two hand-calculated Lion parameter and momentum updates."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64, device=device))
    opt = Lion([p], lr=0.1, weight_decay=0.2)
    p.grad = torch.tensor([2.0, -3.0], dtype=p.dtype, device=device)
    opt.step()
    torch.testing.assert_close(
        p, torch.tensor([0.88, 2.06], dtype=p.dtype, device=device)
    )
    p.grad = torch.tensor([-0.01, 0.01], dtype=p.dtype, device=device)
    opt.step()
    torch.testing.assert_close(
        p, torch.tensor([0.7624, 2.1188], dtype=p.dtype, device=device)
    )
    torch.testing.assert_close(
        opt.state[p]["momentum"],
        torch.tensor([0.0197, -0.0296], dtype=p.dtype, device=device),
    )
