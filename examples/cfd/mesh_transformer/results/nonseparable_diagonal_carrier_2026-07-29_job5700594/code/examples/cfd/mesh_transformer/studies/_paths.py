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

"""Put every source directory of this example on ``sys.path``.

The example's modules import each other by flat module name (``from models
import ...``, ``import train``) while living in sibling subdirectories
(``models/``, ``problems/``, ``studies/``, ``datasets/``) plus the example
root (``metrics.py``, ``provenance.py``).  Scripts that import across
directories start with ``import _paths`` so they stay runnable as plain CLIs
(``python problems/train.py ...``) from any working directory.  An identical
copy of this shim lives in each source subdirectory; importing any copy is
idempotent.  ``tests/conftest.py`` performs the same setup for pytest.
"""

import sys
from pathlib import Path

_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]

for _subdirectory in ("", "models", "problems", "studies", "datasets"):
    _entry = str(_EXAMPLE_ROOT / _subdirectory) if _subdirectory else str(_EXAMPLE_ROOT)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
