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

r"""Heads-versus-rank probe: is the head count pure score-capacity bookkeeping?

PRE-REGISTERED DESIGN (written 2026-07-04, before any run of this study).

**Question.** Per attention head the effective score matrix has rank at most
:math:`R_0 + D R_1` (mesh_attention README section 7), so one layer's
pair-coefficient class is determined by the TOTAL score capacity
:math:`H (R_0 + D R_1)` -- the early design review proved heads and
separable rank enter the score identically in a single layer, suggesting the
head count is redundant with the ranks.  But heads are NOT inert after the
scores: the per-head value spaces (``*_value_dim = dim // heads``) and the
typed output maps contract head-by-head, so a head is an aligned
moment-group whose values cannot mix with another head's before read-out.
At fixed total score capacity, does the split between heads :math:`H` and
per-head ranks :math:`(R_0, R_1)` matter on the full current architecture?
If a difference appears, it can only live in the per-head value/output
structure, because the score-coefficient class is matched by construction.

**Arms** (2D Laplace reference bank, ``problems/train.py``, singpair
dictionary, reference capacity, 3000 steps, seeds 17/29/43, float32, and the
iteration-26 finalist evaluation bank: validation seed 71000011, evaluation
seed 97000037, 32 evaluation cases, 128 boundary points, 512 query points,
4 harmonic cases).  All three arms hold :math:`H (R_0 + 2 R_1) = 80`
(:math:`D = 2`) exactly; the ranks divide evenly, so no rounding was needed:

======================================  ===  =====  =====  ==============
arm (train.py ``--model``)               H    R_0    R_1   parameters
======================================  ===  =====  =====  ==============
mesh_transformer_kernel_singpair          4     12      4  104,537 (ref)
mesh_transformer_kernel_singpair_h1       1     48     16  104,123 (-0.40%)
mesh_transformer_kernel_singpair_h8       8      6      2  104,897 (+0.34%)
======================================  ===  =====  =====  ==============

The residual parameter spread (0.74% end to end) comes from the per-head
structures that are exactly the object under test: the kernel decoder's
member-coefficient head (``n_members * H`` outputs) and the head-blocked
typed value/output maps (e.g. ``drive_vector_dim=12`` floors to value dim 1
at :math:`H=8`).  Matching them away would require changing the channel
widths, which would un-match everything else; 0.74% is recorded, not hidden.

**Falsifiable rule** (declared before the runs).  For each probe arm
(``_h1``, ``_h8``) and each of the four evaluation splits (interpolation,
unseen_geometry_modes, stronger_deformation, unseen_boundary_frequencies),
compare 3-seed mean relative-L2 against the reference arm with tolerance
twice the larger of the two arms' seed standard deviations on that split:

- every cell within tolerance -> **heads are bookkeeping** for capacity on
  this architecture; the one-layer redundancy claim extends to the full
  model and the component-eval loose end closes;
- any cell decisively outside -> the difference **localizes in the per-head
  value/output structure** (the only unmatched machinery), and the parsimony
  question (fewer heads?) opens with a measured direction.

This is a two-sided probe: either outcome is a result.  Three seeds bound
the claim strength -- a within-2-sd tie is reported as such, not as proof of
exact equivalence.

Typical use from this directory::

    python heads_rank_study.py params
    python heads_rank_study.py commands --device cuda --output-root out
    python heads_rank_study.py aggregate --output-root out

This is a benchmark-local research asset, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = EXAMPLE_DIR / "problems" / "train.py"

SEEDS = (17, 29, 43)
STEPS = 3_000
#: Iteration-26 finalist evaluation bank, reused verbatim so the reference
#: arm is directly comparable with the stream-factorization measurement.
EVALUATION_ARGS = (
    "--validation-seed",
    "71000011",
    "--evaluation-seed",
    "97000037",
    "--evaluation-cases",
    "32",
    "--evaluation-boundary-points",
    "128",
    "--evaluation-query-points",
    "512",
    "--harmonic-cases",
    "4",
)
SPLITS = (
    "interpolation",
    "unseen_geometry_modes",
    "stronger_deformation",
    "unseen_boundary_frequencies",
)
#: Pre-registered tolerance multiplier: |mean difference| vs 2 x max seed sd.
TOLERANCE_SEED_SD_MULTIPLIER = 2.0


@dataclass(frozen=True)
class Arm:
    """One point of the fixed-total-score-capacity trade."""

    key: str
    model: str
    heads: int
    scalar_rank: int
    vector_rank: int


ARMS: tuple[Arm, ...] = (
    Arm("h4_reference", "mesh_transformer_kernel_singpair", 4, 12, 4),
    Arm("h1", "mesh_transformer_kernel_singpair_h1", 1, 48, 16),
    Arm("h8", "mesh_transformer_kernel_singpair_h8", 8, 6, 2),
)
REFERENCE_KEY = "h4_reference"
#: Every arm must hold H * (R_0 + D * R_1) with D = 2 at exactly this value.
TOTAL_SCORE_CAPACITY = 80


def score_capacity(arm: Arm) -> int:
    """Total separable score capacity H * (R_0 + D * R_1) in D = 2."""

    return arm.heads * (arm.scalar_rank + 2 * arm.vector_rank)


def build_arm(arm: Arm):
    """Build the exact module the training runs use (via train.make_model)."""

    from train import make_model

    return make_model(arm.model, "reference")


def parameter_table() -> dict[str, Any]:
    """Construct every arm and report exact parameter counts and capacity."""

    from models import parameter_count

    rows = {}
    for arm in ARMS:
        capacity = score_capacity(arm)
        if capacity != TOTAL_SCORE_CAPACITY:
            raise AssertionError(
                f"{arm.key} breaks the fixed-capacity contract: {capacity}"
            )
        rows[arm.key] = {
            "model": arm.model,
            "heads": arm.heads,
            "scalar_rank": arm.scalar_rank,
            "vector_rank": arm.vector_rank,
            "total_score_capacity": capacity,
            "parameters": parameter_count(build_arm(arm)),
        }
    reference = rows[REFERENCE_KEY]["parameters"]
    for row in rows.values():
        row["parameters_vs_reference"] = (row["parameters"] - reference) / reference
    return rows


def make_commands(
    *,
    output_root: Path,
    python: str = sys.executable,
    device: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Resolve the nine deterministic train.py invocations."""

    commands = []
    for arm in ARMS:
        for seed in SEEDS:
            output_dir = output_root / arm.key / f"seed-{seed}"
            argv = [
                python,
                str(TRAIN_SCRIPT),
                "--model",
                arm.model,
                "--capacity",
                "reference",
                "--steps",
                str(STEPS),
                "--seed",
                str(seed),
                "--dtype",
                "float32",
                *EVALUATION_ARGS,
                "--output-dir",
                str(output_dir),
            ]
            if device is not None:
                argv.extend(("--device", device))
            commands.append(tuple(argv))
    return tuple(commands)


def load_reports(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load one report per arm/seed, rejecting mismatched configurations."""

    reports: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        arm_reports = []
        for seed in SEEDS:
            path = (
                output_root / arm.key / f"seed-{seed}" / f"{arm.model}_reference.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing report {path}")
            report = json.loads(path.read_text())
            config = report["run_config"]
            if config["model"] != arm.model or config["seed"] != seed:
                raise ValueError(f"report {path} does not match its directory")
            if config["steps"] != STEPS:
                raise ValueError(f"report {path} ran {config['steps']} steps")
            arm_reports.append(report)
        reports[arm.key] = arm_reports
    return reports


def _split_values(reports: Sequence[Mapping[str, Any]], split: str) -> list[float]:
    values = [
        float(report["evaluation"]["splits"][split]["relative_l2_mean"])
        for report in reports
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite relative L2 on split {split}")
    return values


def aggregate(reports: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Per-arm, per-split 3-seed summary plus the pre-registered verdict."""

    summary: dict[str, Any] = {}
    for arm in ARMS:
        arm_reports = reports[arm.key]
        summary[arm.key] = {
            "parameters": int(arm_reports[0]["parameters"]),
            "splits": {},
        }
        for split in SPLITS:
            values = _split_values(arm_reports, split)
            summary[arm.key]["splits"][split] = {
                "per_seed": values,
                "mean": statistics.fmean(values),
                "seed_sd": statistics.stdev(values),
            }
    comparisons = []
    for arm in ARMS:
        if arm.key == REFERENCE_KEY:
            continue
        for split in SPLITS:
            reference = summary[REFERENCE_KEY]["splits"][split]
            probe = summary[arm.key]["splits"][split]
            tolerance = TOLERANCE_SEED_SD_MULTIPLIER * max(
                reference["seed_sd"], probe["seed_sd"]
            )
            difference = probe["mean"] - reference["mean"]
            comparisons.append(
                {
                    "arm": arm.key,
                    "split": split,
                    "mean_difference": difference,
                    "tolerance": tolerance,
                    "within_tolerance": abs(difference) <= tolerance,
                }
            )
    all_within = all(entry["within_tolerance"] for entry in comparisons)
    return {
        "schema_version": 1,
        "protocol": {
            "steps": STEPS,
            "seeds": list(SEEDS),
            "splits": list(SPLITS),
            "tolerance_rule": (
                "per arm and split, |3-seed mean difference vs reference| <= "
                f"{TOLERANCE_SEED_SD_MULTIPLIER} * max(seed sd of the two arms)"
            ),
        },
        "arms": summary,
        "comparisons": comparisons,
        "verdict": (
            "heads_are_bookkeeping"
            if all_within
            else "difference_localizes_in_per_head_value_output_structure"
        ),
    }


def _format_table(result: Mapping[str, Any]) -> str:
    lines = [
        f"{'arm':<14}{'split':<30}{'mean':>10}{'seed sd':>10}  per-seed",
    ]
    for arm in ARMS:
        for split in SPLITS:
            cell = result["arms"][arm.key]["splits"][split]
            seeds = ", ".join(f"{value:.4f}" for value in cell["per_seed"])
            lines.append(
                f"{arm.key:<14}{split:<30}{cell['mean']:>10.4f}"
                f"{cell['seed_sd']:>10.4f}  [{seeds}]"
            )
    lines.append("")
    for entry in result["comparisons"]:
        status = "within" if entry["within_tolerance"] else "OUTSIDE"
        lines.append(
            f"{entry['arm']:<6} vs reference on {entry['split']:<30}"
            f"delta {entry['mean_difference']:+.4f} vs tolerance "
            f"{entry['tolerance']:.4f} -> {status}"
        )
    lines.append("")
    lines.append(f"verdict: {result['verdict']}")
    return "\n".join(lines)


def main() -> None:
    """Dispatch parameter audit, command emission, or aggregation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("params", help="print exact arm parameter counts")
    commands = subparsers.add_parser("commands", help="emit or run the 9 runs")
    commands.add_argument("--output-root", type=Path, required=True)
    commands.add_argument("--python", default=sys.executable)
    commands.add_argument("--device")
    commands.add_argument("--execute", action="store_true")
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="aggregate the 9 reports and apply the verdict rule"
    )
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "params":
        print(json.dumps(parameter_table(), indent=2))
        return
    if args.command == "commands":
        argvs = make_commands(
            output_root=args.output_root, python=args.python, device=args.device
        )
        for argv in argvs:
            print(shlex.join(argv))
        if args.execute:
            for argv in argvs:
                subprocess.run(argv, check=True)  # noqa: S603
        return
    if args.command == "aggregate":
        result = aggregate(load_reports(args.output_root))
        print(_format_table(result))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "REFERENCE_KEY",
    "SEEDS",
    "SPLITS",
    "STEPS",
    "TOTAL_SCORE_CAPACITY",
    "Arm",
    "aggregate",
    "build_arm",
    "load_reports",
    "make_commands",
    "parameter_table",
    "score_capacity",
]
