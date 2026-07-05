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

"""Reproducibility metadata shared by the learning and timing benchmarks."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parent


def _source_files() -> tuple[Path, ...]:
    """Return the repository sources that define this benchmark's semantics."""

    files = set(EXAMPLE_ROOT.glob("*.py"))
    for subdirectory in ("models", "problems", "studies", "datasets"):
        files.update((EXAMPLE_ROOT / subdirectory).glob("*.py"))
    files.update(
        (REPOSITORY_ROOT / "physicsnemo/experimental/nn/mesh_attention").glob("*.py")
    )
    files.update(
        {
            REPOSITORY_ROOT / "physicsnemo/mesh/calculus/integration.py",
            REPOSITORY_ROOT / "physicsnemo/mesh/domain_mesh.py",
            REPOSITORY_ROOT / "physicsnemo/mesh/fields.py",
            REPOSITORY_ROOT / "physicsnemo/mesh/mesh.py",
        }
    )
    return tuple(sorted(path for path in files if path.is_file()))


def _git_output(arguments: tuple[str, ...]) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.check_output(  # noqa: S603 - fixed read-only git query
            (git, *arguments),
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_provenance() -> dict[str, Any]:
    """Fingerprint exact relevant file contents, including uncommitted edits."""

    files = _source_files()
    relative_paths = tuple(str(path.relative_to(REPOSITORY_ROOT)) for path in files)
    digest = hashlib.sha256()
    for relative_path, path in zip(relative_paths, files, strict=True):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    status = _git_output(("status", "--short", "--", *relative_paths))
    status_lines = [] if not status else status.splitlines()
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "relevant_worktree_dirty": bool(status_lines),
        "relevant_status": status_lines,
        "relevant_source_sha256": digest.hexdigest(),
        "source_files": list(relative_paths),
    }


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """Describe the software and hardware relevant to numerical results."""

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "device": str(device),
        "torch_intraop_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
            }
        )
    else:
        result["device_name"] = platform.processor()
    return result


__all__ = ["runtime_environment", "source_provenance"]
