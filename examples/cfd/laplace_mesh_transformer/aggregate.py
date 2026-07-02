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

"""Reproduce paired case intervals from conformal-Laplace run reports.

Each side may contain one parameter-free report or several training-seed
replicates. Corresponding per-case metrics are averaged across replicates
before cases are resampled, preserving the paired evaluation-bank design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from metrics import paired_case_bootstrap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, nargs="+", required=True)
    parser.add_argument("--right", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", default="relative_l2")
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_701)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def _replicate_mean_cases(
    reports: list[dict[str, Any]], split: str, metric: str
) -> list[float]:
    values = torch.tensor(
        [
            [case[metric] for case in report["evaluation"]["split_cases"][split]]
            for report in reports
        ],
        dtype=torch.float64,
    )
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("reports must contain a nonempty aligned case bank")
    return values.mean(dim=0).tolist()


def main() -> None:
    """Aggregate aligned per-case reports and bootstrap their difference."""

    args = _parse_args()
    left = _load(args.left)
    right = _load(args.right)
    left_splits = tuple(left[0]["evaluation"]["split_cases"])
    if any(
        tuple(report["evaluation"]["split_cases"]) != left_splits for report in left
    ):
        raise ValueError("left reports do not contain the same ordered splits")
    if any(
        tuple(report["evaluation"]["split_cases"]) != left_splits for report in right
    ):
        raise ValueError("left and right reports do not contain aligned splits")

    result = {
        "metric": args.metric,
        "left_reports": [str(path.resolve()) for path in args.left],
        "right_reports": [str(path.resolve()) for path in args.right],
        "replicate_reduction": "mean corresponding cases across training seeds",
        "sampling_unit": "paired continuous PDE case",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "confidence": args.confidence,
        "splits": {},
    }
    for split in left_splits:
        left_cases = _replicate_mean_cases(left, split, args.metric)
        right_cases = _replicate_mean_cases(right, split, args.metric)
        result["splits"][split] = paired_case_bootstrap(
            left_cases,
            right_cases,
            seed=args.bootstrap_seed,
            resamples=args.bootstrap_resamples,
            confidence=args.confidence,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
