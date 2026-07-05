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

"""Make the example's flat-module imports resolve for the test suite.

The example's source lives in sibling subdirectories of the example root
(``models/``, ``problems/``, ``studies/``, ``datasets/``) plus shared modules
at the root itself (``metrics.py``, ``provenance.py``), all imported by flat
module name.  This conftest runs before test collection and puts each source
directory on ``sys.path`` (mirroring the ``_paths.py`` shim used by the CLI
scripts).
"""

import sys
from pathlib import Path

_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]

for _subdirectory in ("", "models", "problems", "studies", "datasets"):
    _entry = str(_EXAMPLE_ROOT / _subdirectory) if _subdirectory else str(_EXAMPLE_ROOT)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
