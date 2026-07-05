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

"""Tests for the mixed Dirichlet/Neumann MeshTransformer study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import mixed_bc_study  # noqa: E402
from laplace3d import build_laplace3d_sample  # noqa: E402

from physicsnemo.mesh import DomainMesh  # noqa: E402

COARSE = {"subdivisions": 1, "n_query": 16}


def _shell_mixed_sample(seed: int = 3):
    return build_laplace3d_sample(seed, tier="shell", bc_regime="mixed", **COARSE)


def test_split_preserves_cells_values_and_normals_exactly() -> None:
    """Routing by BC type is a re-presentation, not an approximation."""

    for seed in (3, 11, 42):
        sample = _shell_mixed_sample(seed)
        split = mixed_bc_study.split_boundaries_by_bc_type(sample.domain)

        assert set(split.boundaries.keys()) == {"dirichlet", "neumann"}
        dirichlet = split.boundaries["dirichlet"]
        neumann = split.boundaries["neumann"]

        # Dirichlet + Neumann cell counts sum to the original total.
        n_original = sum(m.n_cells for m in sample.domain.boundaries.values())
        assert dirichlet.n_cells + neumann.n_cells == n_original
        # Well-posedness: the generator guarantees at least one Dirichlet cell.
        assert dirichlet.n_cells >= 1

        # In the 3D suite, shell-mixed is inner=Dirichlet, outer=Neumann;
        # each routed boundary must reproduce its source boundary exactly:
        # triangle vertex coordinates, cell data values, and cell normals.
        pairs = {
            "inner": (dirichlet, mixed_bc_study.DIRICHLET_KEY),
            "outer": (neumann, mixed_bc_study.NEUMANN_KEY),
        }
        for source_name, (routed, key) in pairs.items():
            source = sample.domain.boundaries[source_name]
            assert routed.n_cells == source.n_cells
            assert set(routed.cell_data.keys()) == {key}
            assert torch.equal(routed.cell_data[key], source.cell_data[key])
            assert torch.equal(routed.points[routed.cells], source.points[source.cells])
            assert torch.allclose(
                routed.cell_normals, source.cell_normals, rtol=0.0, atol=0.0
            )
            assert torch.allclose(
                routed.cell_areas, source.cell_areas, rtol=0.0, atol=0.0
            )

        # Interior queries and global data pass through untouched.
        assert torch.equal(split.interior.points, sample.domain.interior.points)
        assert torch.equal(
            split.global_data["reference_length"],
            sample.domain.global_data["reference_length"],
        )


def test_split_rejects_all_dirichlet_fallback_samples() -> None:
    """Sphere/star under bc_regime='mixed' are all-Dirichlet by generator
    fallback; the two-boundary schema cannot represent them (MeshTransformer
    rejects empty declared boundaries), so the wrapper must refuse loudly."""

    for tier in ("sphere", "star"):
        sample = build_laplace3d_sample(5, tier=tier, bc_regime="mixed", **COARSE)
        assert sample.bc_types == {"outer": "dirichlet"}
        with pytest.raises(ValueError, match="all-Dirichlet"):
            mixed_bc_study.split_boundaries_by_bc_type(sample.domain)


def test_split_rejects_no_dirichlet_and_robin_data() -> None:
    sample = _shell_mixed_sample()
    # Drop the (Dirichlet) inner boundary: no Dirichlet cell remains.
    neumann_only = DomainMesh(
        interior=sample.domain.interior,
        boundaries={"outer": sample.domain.boundaries["outer"]},
        global_data=sample.domain.global_data,
    )
    with pytest.raises(ValueError, match="no Dirichlet"):
        mixed_bc_study.split_boundaries_by_bc_type(neumann_only)

    robin = build_laplace3d_sample(5, tier="sphere", bc_regime="robin", **COARSE)
    with pytest.raises(NotImplementedError, match="Robin"):
        mixed_bc_study.split_boundaries_by_bc_type(robin.domain)


def test_all_arms_train_step_and_eval_are_finite() -> None:
    """One forward+backward per arm on a coarse genuinely-mixed shell sample
    (mixed *sphere* samples are all-Dirichlet and unusable here), then a
    coarse-split evaluation; every loss, gradient, and error must be finite."""

    coarse_splits = {
        "shell_mixed": {"tier": "shell", "bc_regime": "mixed", **COARSE},
        "shell_mixed_fine": {
            "tier": "shell",
            "bc_regime": "mixed",
            "subdivisions": 2,
            "n_query": 16,
        },
    }
    parameter_counts = {}
    for name, operator_layers in mixed_bc_study.MODEL_ARMS.items():
        torch.manual_seed(7)
        model = mixed_bc_study._build_model(name)
        assert len(model.model.operator_blocks) == operator_layers
        sample = _shell_mixed_sample(9)
        prediction = model(sample.domain).point_data["potential"]
        assert prediction.shape == sample.target.shape
        loss = torch.sum((prediction - sample.target).square())
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        parameter_counts[name] = sum(p.numel() for p in model.parameters())

        report = mixed_bc_study.evaluate_splits(
            model,
            eval_seed=123,
            n_cases=1,
            dtype=torch.float32,
            splits=coarse_splits,
        )
        assert set(report.keys()) == set(coarse_splits.keys())
        assert all(torch.isfinite(torch.tensor(v)) for v in report.values())

    # Encoder depth is the only difference between arms.
    assert (
        parameter_counts["mt_singpair_mixed_enc0"]
        < parameter_counts["mt_singpair_mixed_enc1"]
        < parameter_counts["mt_singpair_mixed"]
    )


def test_registry_and_splits_schema() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        mixed_bc_study._build_model("nope")
    # Only genuinely mixed cells of the suite: shell tier, mixed regime.
    for spec in mixed_bc_study.MIXED_SPLITS.values():
        assert spec["tier"] == "shell"
        assert spec["bc_regime"] == "mixed"


def test_run_experiment_writes_finite_report(tmp_path: Path) -> None:
    coarse_splits = {
        "shell_mixed": {"tier": "shell", "bc_regime": "mixed", **COARSE},
    }
    report = mixed_bc_study.run_experiment(
        model_name="mt_singpair_mixed_enc0",
        steps=2,
        seed=5,
        output_dir=str(tmp_path),
        device="cpu",
        eval_cases=1,
        splits=coarse_splits,
    )
    written = json.loads((tmp_path / "mt_singpair_mixed_enc0_seed5.json").read_text())
    assert written["splits"] == report["splits"]
    assert written["parameters"] == report["parameters"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in written["splits"].values())
    assert written["history"]
    # Operator-fidelity block: per-split strong-form residual (finite at
    # this training budget; harmonicity is diagnosed, not enforced) and an
    # explicit n/a marker for the maximum principle (the Neumann outer
    # boundary means the Dirichlet trace alone bounds nothing).
    fidelity = written["fidelity"]
    assert set(fidelity["pde_residual"]) == set(coarse_splits)
    assert all(
        torch.isfinite(torch.tensor(v)) for v in fidelity["pde_residual"].values()
    )
    assert fidelity["max_principle_violation"] is None
