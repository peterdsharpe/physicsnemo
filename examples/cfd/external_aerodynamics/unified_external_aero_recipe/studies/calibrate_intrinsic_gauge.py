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

"""Calibrate the intrinsic-gauge scale constant on the DrivAerML TRAIN split.

Computes the measure-weighted RMS radius of every train sample's vehicle
boundary AS THE PIPELINE DELIVERS IT (same transforms, same 10k-cell
subsample with Horvitz-Thompson weights), then reports

    C = 8.0 / mean(r_RMS)

so DrivAerML samples reproduce the historical fixed gauge of 8.0 and other
geometry families land at the same relative drive-stream operating point.
TRAIN split only — calibration must never see evaluation data.

Run from the recipe root:
    python studies/calibrate_intrinsic_gauge.py
"""

import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RECIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RECIPE_ROOT / "src"))

import torch  # noqa: E402
from datasets import build_dataloaders, load_dataset_config, load_manifest  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from intrinsic_gauge import measure_weighted_rms_radius  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402


def _sha256_lines(values: Iterable[str | int]) -> str:
    """Hash an ordered sequence in a simple, inspectable canonical form."""
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _case_id_from_source_path(source_path: str) -> str:
    """Extract the unique DrivAerML ``run_*`` ID from a reader source path."""
    matches = [part for part in Path(source_path).parts if part.startswith("run_")]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one DrivAerML run_* component in reader "
            f"source_path, got {matches!r} from {source_path!r}."
        )
    return matches[0]


def _collect_calibration_samples(
    dataset: Any,
    sampler: Iterable[int],
) -> tuple[list[int], list[str], list[float]]:
    """Measure exactly the dataset indices selected by ``sampler``."""
    sample_indices = [int(index) for index in sampler]
    if not sample_indices:
        raise RuntimeError("Calibration sampler selected no training samples.")

    case_ids: list[str] = []
    radii: list[float] = []
    for position, index in enumerate(sample_indices, start=1):
        domain, metadata = dataset[index]
        if not isinstance(metadata, dict) or "source_path" not in metadata:
            raise RuntimeError(
                "Calibration requires reader metadata['source_path'] to "
                f"record exact case identity; index {index} returned {metadata!r}."
            )
        case_ids.append(_case_id_from_source_path(str(metadata["source_path"])))
        boundary = domain.boundaries["vehicle"]
        radii.append(float(measure_weighted_rms_radius(boundary)))
        if position % 50 == 0:
            print(f"PROGRESS {position}/{len(sample_indices)}", flush=True)

    return sample_indices, case_ids, radii


def _validate_manifest_case_ids(
    consumed_case_ids: list[str],
    expected_case_ids: list[str],
) -> None:
    """Require one-to-one equality with the frozen train-manifest case set."""
    if len(set(consumed_case_ids)) != len(consumed_case_ids):
        raise RuntimeError("Calibration sampler consumed duplicate case IDs.")
    if len(set(expected_case_ids)) != len(expected_case_ids):
        raise RuntimeError("Frozen training manifest contains duplicate case IDs.")
    if sorted(consumed_case_ids) != sorted(expected_case_ids):
        consumed = set(consumed_case_ids)
        expected = set(expected_case_ids)
        raise RuntimeError(
            "Calibration sampler does not match the frozen training manifest: "
            f"missing={sorted(expected - consumed)[:10]!r}, "
            f"unexpected={sorted(consumed - expected)[:10]!r}, "
            f"consumed={len(consumed_case_ids)}, expected={len(expected_case_ids)}."
        )


def _load_frozen_training_manifest(cfg: Any) -> tuple[Path, list[str]]:
    """Load the frozen training manifest, failing closed when it is absent."""
    dataset_cfg = load_dataset_config(RECIPE_ROOT / "datasets" / f"{cfg.dataset}.yaml")
    train_manifest = dataset_cfg.get("train_manifest", None)
    manifest = dataset_cfg.get("manifest", None)
    if train_manifest is not None:
        manifest_path = Path(str(train_manifest))
        split = None
    elif manifest is not None:
        manifest_path = Path(str(manifest))
        split = str(cfg.train_split)
    else:
        manifest_path = Path(str(dataset_cfg.train_datadir)) / "manifest.json"
        split = str(cfg.train_split)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Train-only calibration requires a frozen training manifest; "
            f"resolved path does not exist: {manifest_path}"
        )
    return manifest_path, load_manifest(manifest_path, split=split)


def main() -> None:
    # Single-process job: SLURM launcher vars would steer DistributedManager
    # into multi-process initialization on a 1-task CPU allocation.
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    for key in [k for k in os.environ if k.startswith(("SLURM_", "PMI_", "PMIX_"))]:
        del os.environ[key]
    DistributedManager.initialize()
    with initialize_config_dir(config_dir=str(RECIPE_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="infer",
            overrides=[
                "model=mesh_transformer_surface_flagship",
                "dataset=drivaer_ml_surface",
                "run_id=intrinsic_gauge_calibration",
                # The val loader iterates infer_split; point it at TRAIN.
                "infer_split=train",
                "sampling_resolution=10000",
                # Runs on CPU-only nodes; pinned host memory needs CUDA.
                "dataloader.pin_memory=false",
            ],
        )
    _train_loader, val_loader, _norm, _info = build_dataloaders(cfg)
    dataset = val_loader.dataset

    sample_indices, case_ids, radii = _collect_calibration_samples(
        dataset, val_loader.sampler
    )
    manifest_path, manifest_case_ids = _load_frozen_training_manifest(cfg)
    _validate_manifest_case_ids(case_ids, manifest_case_ids)
    manifest_record = {
        "available": True,
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_ids_sha256": _sha256_lines(sorted(manifest_case_ids)),
    }

    values = torch.tensor(radii, dtype=torch.float64)
    mean = float(values.mean())
    result = {
        "EDIT_2026-07-27": (
            "Corrected train-only replacement. The historical script "
            "mislabeled all 484 cases (435 train + 49 held out) as train and "
            "produced C=26.476592786355283. This script now iterates the "
            "frozen training sampler and refuses to write unless its exact "
            "case-ID set matches the frozen manifest."
        ),
        "status": "corrected_train_only_calibration",
        "historical_invalid_calibration": {
            "status": "invalid_split_leak_historical_reproduction_only",
            "n_samples": 484,
            "r_rms_mean": 0.3021536821053048,
            "scale_constant_C": 26.476592786355283,
            "split": "all_484_cases_mislabeled_as_train",
        },
        "execution_provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "slurm_job_id": slurm_job_id,
        },
        "n_samples": len(radii),
        "r_rms_mean": mean,
        "r_rms_std": float(values.std()),
        "r_rms_min": float(values.min()),
        "r_rms_max": float(values.max()),
        "fixed_gauge_reproduced": 8.0,
        "scale_constant_C": 8.0 / mean,
        "split": "train",
        "sampling_resolution": 10000,
        "sample_indices": sample_indices,
        "sample_indices_sha256": _sha256_lines(sample_indices),
        "case_ids": case_ids,
        "case_ids_sha256": _sha256_lines(case_ids),
        "case_id_set_sha256": _sha256_lines(sorted(case_ids)),
        "frozen_training_manifest": manifest_record,
    }
    out = RECIPE_ROOT / "studies" / "intrinsic_gauge_calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
