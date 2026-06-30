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

r"""Mesh attention building blocks.

This subpackage provides a hierarchical, equivariant, distance-decaying
attention mechanism whose tokens are mesh cells:

- :class:`~physicsnemo.experimental.nn.mesh_attention.attention.MeshAttention` -
  the attention layer (and :class:`RadialDecay`, its learnable radial envelope).
- :class:`~physicsnemo.experimental.nn.mesh_attention.block.MeshTransformerBlock`
  - a pre-norm transformer block built on it.

See ``README.md`` in this directory for the full design write-up: how it works,
the motivation, and the engineering/theoretical tradeoffs.
"""

from .attention import MeshAttention, RadialDecay
from .block import MeshTransformerBlock

__all__ = ["RadialDecay", "MeshAttention", "MeshTransformerBlock"]
