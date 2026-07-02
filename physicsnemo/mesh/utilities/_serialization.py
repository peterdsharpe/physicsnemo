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

"""Serialization helpers for mesh tensorclasses."""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


def _restore_empty_tensors(tensordict: TensorDict, prefix: Path) -> None:
    """Restore zero-element tensors omitted by TensorDict's memmap writer.

    TensorDict records the shape and dtype of an empty tensor in ``meta.json``
    but does not create a corresponding ``.memmap`` file. Its loader therefore
    omits the key. Reconstructing those storage-free tensors from the metadata
    preserves shapes without changing the on-disk format.
    """
    with (prefix / "meta.json").open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    empty_tensors: dict[str, torch.Tensor] = {}
    for key, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        if entry.get("type") is not None:
            child = tensordict.get(key, None)
            child_prefix = prefix / key
            if isinstance(child, TensorDict) and child_prefix.is_dir():
                _restore_empty_tensors(child, child_prefix)
            continue

        shape = entry.get("shape")
        dtype_name = entry.get("dtype")
        if (
            shape is None
            or dtype_name is None
            or entry.get("is_nested", False)
            or math.prod(shape) != 0
            or key in tensordict
        ):
            continue

        dtype = getattr(torch, dtype_name.removeprefix("torch."))
        empty_tensors[key] = torch.empty(
            shape,
            dtype=dtype,
            device=tensordict.device,
        )

    if empty_tensors:
        with tensordict.unlock_():
            tensordict.update(empty_tensors)


def _load_memmap_with_empty_tensors(
    cls,
    prefix: Path,
    metadata: dict[str, Any],
    *,
    robust_key: bool | None,
    **kwargs: Any,
):
    """Load a tensorclass memmap while retaining metadata-declared empties."""
    non_tensordict = dict(metadata)
    non_tensordict.pop("_type", None)

    other_metadata_path = prefix / "other.pickle"
    if other_metadata_path.exists():
        with other_metadata_path.open("rb") as other_metadata_file:
            non_tensordict.update(
                pickle.load(other_metadata_file)  # noqa: S301
            )

    tensordict_prefix = prefix / "_tensordict"
    tensordict = TensorDict.load_memmap(
        tensordict_prefix,
        **kwargs,
        non_blocking=False,
        robust_key=robust_key,
    )
    _restore_empty_tensors(tensordict, tensordict_prefix)
    return cls._from_tensordict(tensordict, non_tensordict)


__all__ = ["_load_memmap_with_empty_tensors"]
