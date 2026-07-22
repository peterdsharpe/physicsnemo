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

import json
import sys
from pathlib import Path

RECIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RECIPE_ROOT / "src"))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from datasets import build_dataloaders  # noqa: E402
from intrinsic_gauge import measure_weighted_rms_radius  # noqa: E402


def main() -> None:
    with initialize_config_dir(
        config_dir=str(RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="infer",
            overrides=[
                "model=mesh_transformer_surface_flagship",
                "dataset=drivaer_ml_surface",
                "run_id=intrinsic_gauge_calibration",
                # The val loader iterates infer_split; point it at TRAIN.
                "infer_split=train",
                "sampling_resolution=10000",
            ],
        )
    _train_loader, val_loader, _norm, _info = build_dataloaders(cfg)
    dataset = val_loader.dataset

    radii: list[float] = []
    for index in range(len(dataset)):
        domain, _metadata = dataset[index]
        boundary = domain.boundaries["vehicle"]
        radii.append(float(measure_weighted_rms_radius(boundary)))
        if (index + 1) % 50 == 0:
            print(f"PROGRESS {index + 1}/{len(dataset)}", flush=True)

    values = torch.tensor(radii, dtype=torch.float64)
    mean = float(values.mean())
    result = {
        "n_samples": len(radii),
        "r_rms_mean": mean,
        "r_rms_std": float(values.std()),
        "r_rms_min": float(values.min()),
        "r_rms_max": float(values.max()),
        "fixed_gauge_reproduced": 8.0,
        "scale_constant_C": 8.0 / mean,
        "split": "train",
        "sampling_resolution": 10000,
    }
    out = RECIPE_ROOT / "studies" / "intrinsic_gauge_calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
