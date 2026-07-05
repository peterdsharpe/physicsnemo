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

r"""One-head cross-suite confirmation: does the iteration-31 finding travel?

PRE-REGISTERED DESIGN (written 2026-07-04, before any run of this study).

**Question.**  Iteration 31 (archive key ``iteration_31_heads_vs_rank``;
``studies/heads_rank_study.py``) found that at matched total score capacity
:math:`H (R_0 + D R_1) = 80` a SINGLE full-width attention head beats the
4-head reference on 2D Laplace in-distribution by 15% (2.9x the larger seed
sd, decisively outside the pre-registered tolerance), with better seed
stability and the fewest parameters -- the gain prices the head-blocking of
the value/output maps.  Scope was one PDE, one capacity point, three seeds.
Does the one-head advantage travel to other suites, or is it
Laplace-specific?

**Arms.**  One capacity-preserving ``*_h1`` variant per suite (heads
:math:`4 \to 1`, scalar_rank :math:`12 \to 48`, vector_rank :math:`4 \to
16`; the identity :math:`4(12 + D\,4) = 1(48 + D\,16)` holds in every
spatial dimension), three seeds each, against the ALREADY-ARCHIVED 4-head
comparator runs -- protocols matched run-for-run, nothing re-trained on the
h4 side:

=====================  ==========================================  ======================================================
suite                  h1 arm (driver ``--model``)                 archived h4 comparator (key / entry)
=====================  ==========================================  ======================================================
screened               mesh_transformer_kernel_singonly_h1        ``iteration_13_singonly_universality`` /
(2000 steps, s17-19)   (``problems/screened_laplace.py``)          ``screened_singonly_2000steps.runs`` (seeds 17/18/19)
pf_velocity            mesh_transformer_kernel_singpair_pseudo_h1 ``iteration_23_pseudoscalar_extension`` /
(3000 steps, s17-19)   (``problems/potential_flow.py``,            ``runs_3000steps.mesh_transformer_kernel_
                       family potential_flow_velocity)             singpair_pseudo`` (seeds 17-21, all five used)
laplace3d              mesh_transformer_kernel_singpair_h1        ``iteration_16_single_layer_member`` /
(3000 steps, s17-19)   (``problems/laplace3d_study.py``)           ``runs_3d_singpair`` (seeds 17/18/19)
=====================  ==========================================  ======================================================

The pf_velocity arm composes the one-head trade with the pseudoscalar
sector (``drive_pseudo_dim=8``); the pseudo channels ride the scalar moment
machinery, so their per-head split widens from 8/4 = 2 to 8 exactly as the
scalar channels do -- verified by construction test before any run.

**Falsifiable rule for a DEFAULT change** (declared before the runs).  Per
suite and evaluation split, with 3-seed h1 statistics against the archived
h4 seed statistics, tolerance :math:`2 \max(\mathrm{sd}_{h1},
\mathrm{sd}_{h4})`:

- h1 *matches-or-beats* h4 on a split iff
  :math:`\overline{h1} - \overline{h4} \le` tolerance;
- h1 *loses decisively* on a split iff
  :math:`\overline{h1} - \overline{h4} >` tolerance;
- **flip the reference configuration to one head** iff h1
  matches-or-beats h4 on EVERY split of at least two of these suites AND
  no split anywhere shows a decisive loss -- another parsimony rung: fewer
  heads, simpler story, same parameters;
- **any decisive loss** -> heads stay at 4 and the iteration-31 result is
  recorded as Laplace-specific.

This is a one-sided confirmation rule on purpose: iteration 31 already
established the 2D-Laplace win two-sidedly; here h1 only has to not lose.
Three seeds per suite bound the claim strength -- a within-tolerance match
is reported as such, not as proof of equivalence.

Typical use from this directory::

    python one_head_cross_suite.py params
    python one_head_cross_suite.py commands --device cuda --output-root out
    python one_head_cross_suite.py aggregate --output-root out

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
PROBLEM_DIR = EXAMPLE_DIR / "problems"
#: The research archive carrying every archived h4 comparator run.
ARCHIVE_PATH = EXAMPLE_DIR / "results" / "learned_bie_2026-07-02.json"

SEEDS = (17, 18, 19)
#: Pre-registered tolerance multiplier: mean difference vs 2 x max seed sd.
TOLERANCE_SEED_SD_MULTIPLIER = 2.0
#: The one-head capacity trade under test (identical to iteration 31).
ONE_HEAD_SETTINGS = {"heads": 1, "scalar_rank": 48, "vector_rank": 16}


@dataclass(frozen=True)
class Suite:
    """One suite of the cross-suite confirmation matrix."""

    key: str
    script: str
    model: str
    steps: int
    splits: tuple[str, ...]
    #: Key path into the archive JSON resolving to the h4 per-seed run list.
    comparator_path: tuple[str, ...]
    comparator_label: str
    report_name: str
    extra_args: tuple[str, ...] = ()


SUITES: tuple[Suite, ...] = (
    Suite(
        key="screened",
        script="screened_laplace.py",
        model="mesh_transformer_kernel_singonly_h1",
        steps=2_000,
        splits=(
            "in_distribution",
            "ood_low_screening",
            "ood_high_screening",
            "unseen_modes",
        ),
        comparator_path=(
            "iteration_13_singonly_universality",
            "screened_singonly_2000steps",
            "runs",
        ),
        comparator_label=(
            "mesh_transformer_kernel_singonly, 2000 steps, seeds 17/18/19 "
            "(iteration_13_singonly_universality)"
        ),
        report_name="mesh_transformer_kernel_singonly_h1_seed{seed}.json",
    ),
    Suite(
        key="pf_velocity",
        script="potential_flow.py",
        model="mesh_transformer_kernel_singpair_pseudo_h1",
        steps=3_000,
        splits=(
            "in_distribution",
            "unseen_geometry_modes",
            "wilder_shapes",
            "circulation_ood",
            "farfield_queries",
        ),
        comparator_path=(
            "iteration_23_pseudoscalar_extension",
            "runs_3000steps",
            "mesh_transformer_kernel_singpair_pseudo",
        ),
        comparator_label=(
            "mesh_transformer_kernel_singpair_pseudo, potential_flow_velocity, "
            "3000 steps, seeds 17-21 (iteration_23_pseudoscalar_extension)"
        ),
        report_name=(
            "potential_flow_velocity_"
            "mesh_transformer_kernel_singpair_pseudo_h1_seed{seed}.json"
        ),
        extra_args=("--family", "potential_flow_velocity"),
    ),
    Suite(
        key="laplace3d",
        script="laplace3d_study.py",
        model="mesh_transformer_kernel_singpair_h1",
        steps=3_000,
        splits=("sphere", "star", "star_unseen_modes", "shell_topology"),
        comparator_path=("iteration_16_single_layer_member", "runs_3d_singpair"),
        comparator_label=(
            "mesh_transformer_kernel_singpair (3D), 3000 steps, seeds 17/18/19 "
            "(iteration_16_single_layer_member)"
        ),
        report_name="mesh_transformer_kernel_singpair_h1_seed{seed}.json",
    ),
)
#: Minimum number of suites that must pass on every split for the flip.
REQUIRED_PASSING_SUITES = 2

VERDICT_FLIP = "flip_default_to_one_head"
VERDICT_KEEP = "laplace_specific_heads_stay_at_4"


def build_arm(suite: Suite):
    """Build the exact module the training runs use (via each driver)."""

    if suite.key == "screened":
        from screened_laplace import _build_model

        return _build_model(suite.model)
    if suite.key == "pf_velocity":
        from potential_flow import _build_model

        return _build_model(suite.model, "potential_flow_velocity")
    if suite.key == "laplace3d":
        from laplace3d_study import _build_model

        return _build_model(suite.model)
    raise AssertionError(f"unhandled suite {suite.key!r}")


def parameter_table() -> dict[str, Any]:
    """Construct every h1 arm and report exact parameter counts."""

    from models import parameter_count

    return {
        suite.key: {
            "model": suite.model,
            **ONE_HEAD_SETTINGS,
            "parameters": parameter_count(build_arm(suite)),
        }
        for suite in SUITES
    }


def make_commands(
    *,
    output_root: Path,
    python: str = sys.executable,
    device: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Resolve the nine deterministic driver invocations."""

    commands = []
    for suite in SUITES:
        for seed in SEEDS:
            argv = [
                python,
                str(PROBLEM_DIR / suite.script),
                "--model",
                suite.model,
                *suite.extra_args,
                "--steps",
                str(suite.steps),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_root / suite.key / f"seed-{seed}"),
            ]
            if device is not None:
                argv.extend(("--device", device))
            commands.append(tuple(argv))
    return tuple(commands)


def load_reports(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load one report per suite/seed, rejecting mismatched configurations."""

    reports: dict[str, list[dict[str, Any]]] = {}
    for suite in SUITES:
        suite_reports = []
        for seed in SEEDS:
            path = (
                output_root
                / suite.key
                / f"seed-{seed}"
                / suite.report_name.format(seed=seed)
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing report {path}")
            report = json.loads(path.read_text())
            if report["model"] != suite.model or report["seed"] != seed:
                raise ValueError(f"report {path} does not match its directory")
            if report["steps"] != suite.steps:
                raise ValueError(f"report {path} ran {report['steps']} steps")
            suite_reports.append(report)
        reports[suite.key] = suite_reports
    return reports


def load_comparators(
    archive_path: Path = ARCHIVE_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve each suite's archived h4 per-seed runs from the archive."""

    archive = json.loads(archive_path.read_text())
    comparators: dict[str, list[dict[str, Any]]] = {}
    for suite in SUITES:
        node: Any = archive
        for key in suite.comparator_path:
            node = node[key]
        if not isinstance(node, list) or not node:
            raise ValueError(
                f"comparator path {suite.comparator_path} for {suite.key} "
                "does not resolve to a non-empty run list"
            )
        comparators[suite.key] = node
    return comparators


def _split_values(
    runs: Sequence[Mapping[str, Any]], split: str, source: str
) -> list[float]:
    values = []
    for run in runs:
        record = run.get("splits", run)
        values.append(float(record[split]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite relative L2 on {source} split {split}")
    return values


def aggregate(
    reports: Mapping[str, Sequence[Mapping[str, Any]]],
    comparators: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Per-suite, per-split summary plus the pre-registered flip verdict."""

    if comparators is None:
        comparators = load_comparators()
    suites: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    for suite in SUITES:
        h1_runs = reports[suite.key]
        h4_runs = comparators[suite.key]
        entry: dict[str, Any] = {
            "model": suite.model,
            "comparator": suite.comparator_label,
            "h1_parameters": int(h1_runs[0]["parameters"]),
            "h1_seeds": [int(run["seed"]) for run in h1_runs],
            "h4_seeds": [int(run["seed"]) for run in h4_runs],
            "splits": {},
        }
        suite_pass = True
        for split in suite.splits:
            h1_values = _split_values(h1_runs, split, f"{suite.key} h1")
            h4_values = _split_values(h4_runs, split, f"{suite.key} h4")
            h1_mean, h1_sd = statistics.fmean(h1_values), statistics.stdev(h1_values)
            h4_mean, h4_sd = statistics.fmean(h4_values), statistics.stdev(h4_values)
            tolerance = TOLERANCE_SEED_SD_MULTIPLIER * max(h1_sd, h4_sd)
            difference = h1_mean - h4_mean
            matches_or_beats = difference <= tolerance
            suite_pass = suite_pass and matches_or_beats
            entry["splits"][split] = {
                "h1_per_seed": h1_values,
                "h1_mean": h1_mean,
                "h1_seed_sd": h1_sd,
                "h4_per_seed": h4_values,
                "h4_mean": h4_mean,
                "h4_seed_sd": h4_sd,
            }
            comparisons.append(
                {
                    "suite": suite.key,
                    "split": split,
                    "mean_difference": difference,
                    "tolerance": tolerance,
                    "matches_or_beats": matches_or_beats,
                    "decisive_loss": not matches_or_beats,
                }
            )
        entry["all_splits_match_or_beat"] = suite_pass
        suites[suite.key] = entry
    passing = sum(1 for entry in suites.values() if entry["all_splits_match_or_beat"])
    any_decisive_loss = any(entry["decisive_loss"] for entry in comparisons)
    flip = passing >= REQUIRED_PASSING_SUITES and not any_decisive_loss
    return {
        "schema_version": 1,
        "protocol": {
            "seeds": list(SEEDS),
            "one_head_settings": dict(ONE_HEAD_SETTINGS),
            "tolerance_rule": (
                "per suite and split, h1 matches-or-beats iff "
                "mean(h1) - mean(h4) <= "
                f"{TOLERANCE_SEED_SD_MULTIPLIER} * max(seed sd of the two arms); "
                "flip iff every split of >= "
                f"{REQUIRED_PASSING_SUITES} suites matches-or-beats AND no "
                "split anywhere is a decisive loss"
            ),
        },
        "suites": suites,
        "comparisons": comparisons,
        "passing_suites": passing,
        "any_decisive_loss": any_decisive_loss,
        "verdict": VERDICT_FLIP if flip else VERDICT_KEEP,
    }


def _format_table(result: Mapping[str, Any]) -> str:
    lines = [
        f"{'suite':<12}{'split':<24}{'h1 mean':>9}{'h1 sd':>8}"
        f"{'h4 mean':>9}{'h4 sd':>8}{'delta':>9}{'tol':>8}  status",
    ]
    by_suite = result["suites"]
    for entry in result["comparisons"]:
        cell = by_suite[entry["suite"]]["splits"][entry["split"]]
        status = "match/beat" if entry["matches_or_beats"] else "DECISIVE LOSS"
        lines.append(
            f"{entry['suite']:<12}{entry['split']:<24}"
            f"{cell['h1_mean']:>9.4f}{cell['h1_seed_sd']:>8.4f}"
            f"{cell['h4_mean']:>9.4f}{cell['h4_seed_sd']:>8.4f}"
            f"{entry['mean_difference']:>+9.4f}{entry['tolerance']:>8.4f}"
            f"  {status}"
        )
    lines.append("")
    lines.append(
        f"passing suites: {result['passing_suites']}/{len(by_suite)} "
        f"(need >= {REQUIRED_PASSING_SUITES}); "
        f"any decisive loss: {result['any_decisive_loss']}"
    )
    lines.append(f"verdict: {result['verdict']}")
    return "\n".join(lines)


def main() -> None:
    """Dispatch parameter audit, command emission, or aggregation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("params", help="print exact h1 arm parameter counts")
    commands = subparsers.add_parser("commands", help="emit or run the 9 runs")
    commands.add_argument("--output-root", type=Path, required=True)
    commands.add_argument("--python", default=sys.executable)
    commands.add_argument("--device")
    commands.add_argument("--execute", action="store_true")
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="aggregate the 9 reports and apply the flip rule"
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
    "ARCHIVE_PATH",
    "ONE_HEAD_SETTINGS",
    "REQUIRED_PASSING_SUITES",
    "SEEDS",
    "SUITES",
    "VERDICT_FLIP",
    "VERDICT_KEEP",
    "Suite",
    "aggregate",
    "build_arm",
    "load_comparators",
    "load_reports",
    "make_commands",
    "parameter_table",
]
