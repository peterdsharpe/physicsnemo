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

"""Contracts for the dataset-backed (fixed-dataset, epoch-based) trainer.

Runs against the checked-in 8-case ``star_random_trace/v0-demo`` catalog on
CPU, so the whole file stays well under a minute.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import dataset_catalog  # noqa: E402
import dataset_train  # noqa: E402

DEMO_DIR = EXAMPLE_DIR / "datasets" / "star_random_trace" / "v0-demo"
MODEL = "mesh_transformer_kernel_singonly"
SEED = 3


@pytest.fixture(scope="module")
def two_epoch_run(tmp_path_factory):
    """One shared 2-epoch CPU training run on the demo catalog."""

    output_dir = tmp_path_factory.mktemp("dataset_train_report")
    report = dataset_train.run_experiment(
        dataset_dir=DEMO_DIR,
        model_name=MODEL,
        epochs=2,
        seed=SEED,
        device="cpu",
        output_dir=output_dir,
    )
    return report, output_dir


def test_losses_and_validation_are_finite(two_epoch_run):
    report, _ = two_epoch_run
    assert len(report["history"]) == 2
    for record in report["history"]:
        assert math.isfinite(record["train_relative_mse"])
        # epochs // 12 == 0 -> validation every epoch.
        assert math.isfinite(record["validation_relative_l2"])
    assert math.isfinite(report["best_validation_relative_l2"])
    assert report["best_epoch"] in (1, 2)


def test_report_written_and_round_trips(two_epoch_run):
    report, output_dir = two_epoch_run
    path = output_dir / f"{MODEL}_seed{SEED}.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == report


def test_all_manifest_splits_evaluated(two_epoch_run):
    report, _ = two_epoch_run
    manifest = dataset_catalog.load_manifest(DEMO_DIR)
    eval_names = {name for name in manifest["splits"] if name != "train"}
    assert set(report["splits"]) == eval_names
    assert all(math.isfinite(value) for value in report["splits"].values())
    assert report["split_sizes"] == {
        name: spec["stop"] - spec["start"] for name, spec in manifest["splits"].items()
    }
    assert report["n_train_cases"] == report["split_sizes"]["train"]
    assert report["verification"] == manifest["verification"]
    assert report["dataset"]["family"] == manifest["family"]
    assert report["dataset"]["version"] == manifest["version"]


def test_same_seed_reproduces_first_epoch_loss(two_epoch_run, tmp_path):
    """The seeded shuffle and init make the first epoch bit-reproducible."""

    reference, _ = two_epoch_run
    repeat = dataset_train.run_experiment(
        dataset_dir=DEMO_DIR,
        model_name=MODEL,
        epochs=1,
        seed=SEED,
        device="cpu",
        output_dir=tmp_path,
    )
    assert (
        repeat["history"][0]["train_relative_mse"]
        == reference["history"][0]["train_relative_mse"]
    )
