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

"""Focused tests for the paired, same-checkpoint GLOBE backend study."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import globe_backend_study as backend_study  # noqa: E402
from external_baselines import GlobeLaplaceAdapter  # noqa: E402


def _tiny_globe() -> GlobeLaplaceAdapter:
    return GlobeLaplaceAdapter(
        communication_layers=0,
        theta=0.0,
        hidden_dim=8,
        hidden_layers=1,
        latent_scalars=2,
        latent_vectors=1,
        n_spherical_harmonics=1,
        network_type="mlp",
    )


def _save_checkpoint(
    path: Path,
    *,
    model_name: str = "globe_exact",
    source_digest: str = "matching-source",
) -> None:
    torch.manual_seed(601)
    model = _tiny_globe()
    torch.save(
        {
            "model": model_name,
            "capacity": "reference",
            "state_dict": model.state_dict(),
            "run_config": {"seed": 601, "steps": 3},
            "source": {"relevant_source_sha256": source_digest},
        },
        path,
    )


def _patch_tiny_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backend_study,
        "make_model",
        lambda model_name, capacity: _tiny_globe(),
    )


def test_same_checkpoint_theta_sweep_reports_paired_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare every opening angle on one checkpoint and paired case bank."""

    checkpoint = tmp_path / "tiny_globe_exact.pt"
    _save_checkpoint(checkpoint)
    _patch_tiny_builder(monkeypatch)
    monkeypatch.setattr(
        backend_study,
        "source_provenance",
        lambda: {"relevant_source_sha256": "matching-source"},
    )

    report = backend_study.run_globe_backend_study(
        checkpoint,
        device_name="cpu",
        dtype_name="float32",
        thetas=(1.0, 0.0),
        splits=("interpolation",),
        evaluation_seed=607,
        evaluation_cases=1,
        n_boundary=6,
        n_query=8,
        warmup_cases=0,
    )

    assert report["checkpoint"]["source_matches_evaluator"] is True
    assert report["evaluation"]["thetas"] == [0.0, 1.0]
    assert report["interpretation"]["parameter_state"].startswith("one exact")
    assert "not a certified error tolerance" in report["interpretation"]["theta"]
    assert len(report["model_state_sha256"]) == 64
    assert set(report["theta_results"]) == {"0", "1"}

    exact = report["theta_results"]["0"]
    approximate = report["theta_results"]["1"]
    exact_split = exact["splits"]["interpolation"]
    approximate_split = approximate["splits"]["interpolation"]
    assert exact_split["cases"] == 1
    assert (
        exact_split["prediction_delta_to_theta_zero"][
            "target_normalized_weighted_l2_mean"
        ]
        == 0.0
    )
    assert (
        approximate_split["prediction_delta_to_theta_zero"][
            "target_normalized_weighted_l2_mean"
        ]
        >= 0.0
    )
    assert exact["forward_elapsed_synchronized_seconds"] >= 0.0
    assert approximate["forward_elapsed_synchronized_seconds"] >= 0.0
    assert exact["cuda_peak_allocation"]["method"] == "not_measured_on_cpu"


def test_source_mismatch_requires_explicit_historical_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject stale checkpoints unless historical evaluation is explicit."""

    checkpoint = tmp_path / "tiny_globe_exact.pt"
    _save_checkpoint(checkpoint, source_digest="old-source")
    _patch_tiny_builder(monkeypatch)
    monkeypatch.setattr(
        backend_study,
        "source_provenance",
        lambda: {"relevant_source_sha256": "current-source"},
    )

    with pytest.raises(ValueError, match="source fingerprint differs"):
        backend_study.load_exact_globe_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    _, metadata = backend_study.load_exact_globe_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
        allow_source_mismatch=True,
    )
    assert metadata["source_matches_evaluator"] is False
    assert metadata["source_mismatch_allowed"] is True


def test_backend_sweep_rejects_nonexact_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start hierarchy sweeps only from an exact-interaction checkpoint."""

    checkpoint = tmp_path / "hierarchical.pt"
    _save_checkpoint(checkpoint, model_name="globe_hierarchical")
    monkeypatch.setattr(
        backend_study,
        "source_provenance",
        lambda: {"relevant_source_sha256": "matching-source"},
    )

    with pytest.raises(ValueError, match="exact-trained"):
        backend_study.load_exact_globe_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_backend_sweep_requires_theta_zero(tmp_path: Path) -> None:
    """Require an exact theta-zero reference in every backend sweep."""

    with pytest.raises(ValueError, match="include the exact theta=0"):
        backend_study.run_globe_backend_study(
            tmp_path / "unused.pt",
            thetas=(0.25, 1.0),
        )


def test_split_seed_does_not_depend_on_subset_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each split's case bank fixed when other splits are omitted."""

    seeds: list[int] = []

    def record_case(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return object()

    monkeypatch.setattr(backend_study, "make_case", record_case)
    bank = backend_study.build_evaluation_bank(
        splits=("unseen_boundary_frequencies",),
        evaluation_seed=700,
        evaluation_cases=2,
        n_boundary=6,
        n_query=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert tuple(bank) == ("unseen_boundary_frequencies",)
    assert seeds == [400_700, 400_700]
