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

"""Per-case global fields injected from a JSON table, keyed by the case's CRC
(``SetGlobalFieldsFromTable``)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh

from domain_transforms import SetGlobalFieldsFromTable

_TABLE = {
    "run_1": {"frame_center": [0.1, -0.2, 0.3], "frame_scale": 0.55, "n_cells": 17},
    "run_10": {"frame_center": [1.0, 2.0, 3.0], "frame_scale": 0.7, "n_cells": 19},
    "_provenance": {"script": "compute_frame_table.py"},
}


def _crc(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def _mesh(case_key: int | None, dtype=torch.float32) -> Mesh:
    gd = {"U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=dtype)}
    if case_key is not None:
        gd["case_key"] = torch.tensor(case_key, dtype=torch.int64)
    return Mesh(
        points=torch.randn(12, 3, dtype=dtype),
        cells=torch.stack([torch.arange(0, 10), torch.arange(1, 11), torch.arange(2, 12)], dim=1),
        global_data=TensorDict(gd, batch_size=[]),
    )


def test_table_fields_are_injected_by_case_key(tmp_path: Path) -> None:
    table = tmp_path / "frame_table.json"
    table.write_text(json.dumps(_TABLE))
    t = SetGlobalFieldsFromTable(table=str(table), fields=["frame_center", "frame_scale"])
    for name, row in _TABLE.items():
        if name.startswith("_"):
            continue
        out = t(_mesh(_crc(name)))
        assert torch.allclose(out.global_data["frame_center"], torch.tensor(row["frame_center"]))
        assert out.global_data["frame_center"].dtype == torch.float32
        assert out.global_data["frame_scale"].shape == () and float(out.global_data["frame_scale"]) == pytest.approx(row["frame_scale"])
        assert "n_cells" not in out.global_data.keys()  # only the requested fields
        assert "U_inf" in out.global_data.keys()  # existing fields kept
    with pytest.raises(KeyError):
        t(_mesh(_crc("run_999")))
    with pytest.raises(KeyError):
        t(_mesh(None))
    assert "2 cases" in t.extra_repr()
