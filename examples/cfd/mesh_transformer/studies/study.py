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

"""Execute and adjudicate the PDE-theoretic MeshTransformer study.

This driver keeps research policy out of :mod:`train`: it declares the
approved ablations, emits the exact commands used for every replicate, and
applies the advancement gates to the resulting JSON reports.  Every command
uses the same fixed validation/evaluation banks supplied by ``train.py``;
only the training seed changes between finalist replicates.

Typical use from the repository root::

    python examples/cfd/mesh_transformer/studies/study.py commands \
        --phase early --execute
    python examples/cfd/mesh_transformer/studies/study.py commands \
        --phase finalists --finalist factorial_encoded_pair --finalist stf_l4 \
        --execute
    python examples/cfd/mesh_transformer/studies/study.py aggregate \
        --phase finalists

``spectral-report`` extracts a complete disk boundary-to-query operator from
one checkpoint, compares its singular spectrum with the discrete Poisson
operator, and reports its signed Fourier transfer matrix.  This is deliberately
a benchmark diagnostic, not a production inference path.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import _paths  # noqa: F401

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = EXAMPLE_DIR / "problems" / "train.py"
DEFAULT_OUTPUT_ROOT = Path("outputs/mesh_transformer/study")
DEFAULT_EARLY_SEED = 17
DEFAULT_FINALIST_SEEDS = (17, 29, 43)
DEFAULT_STEPS = 1_000
EARLY_TRAIN_ARGS = (
    "--validation-cases",
    "4",
    "--evaluation-cases",
    "8",
    "--evaluation-query-points",
    "256",
)

# These are numerical-contract tolerances, not accuracy hyperparameters.
# 128 ulps leaves room for reductions through a moderately deep fp32 network
# while remaining orders of magnitude below a learned-surrogate error.
FP32_CONTRACT_TOLERANCE = 128.0 * 2.0**-23
REFINEMENT_ABSOLUTE_TOLERANCE = FP32_CONTRACT_TOLERANCE
REFINEMENT_FINE_CAUCHY_TOLERANCE = 0.05


@dataclass(frozen=True)
class StudyRun:
    """One unique training configuration in the approved experimental matrix."""

    key: str
    model: str
    capacity: str = "reference"
    drive_distribution: str = "boundary_balanced_mixture"
    training_objective: str = "auto"
    groups: tuple[str, ...] = ()
    role: Literal["candidate", "control", "oracle", "external"] = "candidate"
    eligible_for_advancement: bool = True
    description: str = ""


# The first four entries are the controlled two-by-two factorial.  The
# ``minimal`` boundary processor is MeshTransformer with zero source-processing
# blocks (the shallow capacity) or the pointwise pair lift.  The encoded rows
# use the current MeshTransformer source encoder.  ``factorial_encoded_moment``
# is also the baseline objective-control run and is therefore not duplicated.
STUDY_RUNS: tuple[StudyRun, ...] = (
    StudyRun(
        key="factorial_minimal_moment",
        model="lifted_mesh_transformer",
        capacity="shallow",
        groups=("factorial",),
        role="control",
        eligible_for_advancement=False,
        description="Minimal pointwise boundary lift + current moment decoder",
    ),
    StudyRun(
        key="factorial_encoded_moment",
        model="lifted_mesh_transformer",
        capacity="reference",
        groups=("factorial", "objective_control", "current_control"),
        role="control",
        eligible_for_advancement=False,
        description="Current boundary encoder + current moment decoder",
    ),
    StudyRun(
        key="factorial_minimal_pair",
        model="encoded_pair_kernel",
        capacity="shallow",
        groups=("factorial", "dense_control"),
        description=(
            "The same shallow pointwise boundary lift + dense invariant pair decoder"
        ),
    ),
    StudyRun(
        key="factorial_encoded_pair",
        model="encoded_pair_kernel",
        groups=("factorial", "prototype"),
        description="Current boundary encoder + dense invariant pair decoder",
    ),
    StudyRun(
        key="simple_invariant_pair",
        model="pair_kernel",
        groups=("simple_dense_control", "prototype"),
        description=(
            "Identity residual drive + two-invariant dense pair kernel; simpler "
            "but not a strict factorial cell"
        ),
    ),
    StudyRun(
        key="objective_interior_balanced",
        model="lifted_mesh_transformer",
        drive_distribution="disk_interior_balanced_mixture",
        groups=("objective_control",),
        role="control",
        eligible_for_advancement=False,
        description="Current model with equal disk-interior-energy mode weighting",
    ),
    StudyRun(
        key="objective_uniform_pure_mode",
        model="lifted_mesh_transformer",
        drive_distribution="uniform_pure_mode",
        groups=("objective_control",),
        role="control",
        eligible_for_advancement=False,
        description="Current model trained on one uniformly sampled mode per case",
    ),
    StudyRun(
        key="stf_scalar_vector_matched",
        model="lifted_mesh_transformer",
        capacity="stf_matched",
        groups=("stf_control",),
        role="control",
        eligible_for_advancement=False,
        description=(
            "Ordinary scalar/vector MeshTransformer matched within 0.6% of "
            "the ell=4 STF parameter count"
        ),
    ),
    StudyRun(
        key="stf_l1",
        model="stf_multipole_l1",
        groups=("stf", "prototype"),
        description="Typed symmetric-trace-free multipoles through ell=1",
    ),
    StudyRun(
        key="stf_l2",
        model="stf_multipole_l2",
        groups=("stf", "prototype"),
        description="Typed symmetric-trace-free multipoles through ell=2",
    ),
    StudyRun(
        key="stf_l4",
        model="stf_multipole_l4",
        groups=("stf", "prototype"),
        description="Typed symmetric-trace-free multipoles through ell=4",
    ),
    StudyRun(
        key="layer_direct_density",
        model="double_layer_direct",
        groups=("layer_potential", "prototype"),
        role="control",
        eligible_for_advancement=False,
        description="Boundary values used directly as double-layer density",
    ),
    StudyRun(
        key="layer_solved_density",
        model="double_layer_solved",
        groups=("layer_potential", "prototype"),
        role="oracle",
        eligible_for_advancement=False,
        description="Dense analytic boundary-density solve + double-layer evaluation",
    ),
    StudyRun(
        key="layer_richardson_density",
        model="double_layer_richardson",
        training_objective="boundary_collocation",
        groups=("layer_potential", "prototype"),
        description="Learned linear density iteration + analytic double-layer kernel",
    ),
    StudyRun(
        key="layer_encoded_density",
        model="double_layer_encoded",
        training_objective="boundary_collocation",
        groups=("layer_potential", "prototype"),
        description="Current linear boundary encoder + analytic double-layer kernel",
    ),
    StudyRun(
        key="globe_exact",
        model="globe_exact",
        groups=("external",),
        role="external",
        eligible_for_advancement=False,
        description="Stock GLOBE with exact all-pairs communication",
    ),
    StudyRun(
        key="globe_hierarchical",
        model="globe_hierarchical",
        groups=("external",),
        role="external",
        eligible_for_advancement=False,
        description="Stock GLOBE with its hierarchical communication backend",
    ),
    StudyRun(
        key="geotransolver_matched",
        model="geotransolver_matched",
        groups=("external",),
        role="external",
        eligible_for_advancement=False,
        description=(
            "133,083-parameter GeoTransolver control (2.1% below the reference "
            "moment model) without ball-query features"
        ),
    ),
    StudyRun(
        key="geotransolver_published_scale",
        model="geotransolver_published_scale",
        groups=("external",),
        role="external",
        eligible_for_advancement=False,
        description=(
            "29,144,481-parameter GeoTransolver control at published parameter "
            "scale, without ball-query features or the published training recipe"
        ),
    ),
)

RUN_BY_KEY = {run.key: run for run in STUDY_RUNS}
CURRENT_CONTROL = "factorial_encoded_moment"
DENSE_CONTROL = "factorial_minimal_pair"


@dataclass(frozen=True)
class CommandSpec:
    """A fully resolved invocation and the report it is expected to create."""

    phase: Literal["early", "finalists"]
    run: StudyRun
    seed: int
    argv: tuple[str, ...]
    output_dir: Path
    report_path: Path


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def select_runs(
    *,
    groups: Sequence[str] = (),
    keys: Sequence[str] = (),
) -> tuple[StudyRun, ...]:
    """Select runs in declaration order, rejecting typos and empty selections."""

    unknown_keys = sorted(set(keys) - RUN_BY_KEY.keys())
    if unknown_keys:
        raise ValueError(f"unknown study run(s): {', '.join(unknown_keys)}")
    known_groups = {group for run in STUDY_RUNS for group in run.groups}
    unknown_groups = sorted(set(groups) - known_groups)
    if unknown_groups:
        raise ValueError(f"unknown study group(s): {', '.join(unknown_groups)}")
    if not groups and not keys:
        return STUDY_RUNS
    key_set = set(keys)
    group_set = set(groups)
    selected = tuple(
        run
        for run in STUDY_RUNS
        if run.key in key_set or group_set.intersection(run.groups)
    )
    if not selected:
        raise ValueError("run selection is empty")
    return selected


def make_command_specs(
    runs: Sequence[StudyRun],
    *,
    phase: Literal["early", "finalists"],
    output_root: Path,
    python: str,
    steps: int = DEFAULT_STEPS,
    early_seed: int = DEFAULT_EARLY_SEED,
    finalist_seeds: Sequence[int] = DEFAULT_FINALIST_SEEDS,
    device: str | None = None,
    dtype: str = "float32",
    extra_train_args: Sequence[str] = (),
) -> tuple[CommandSpec, ...]:
    """Resolve deterministic, non-overwriting ``train.py`` invocations."""

    _positive_integer("steps", steps)
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if phase == "early":
        seeds = (early_seed,)
    elif phase == "finalists":
        if not finalist_seeds:
            raise ValueError("finalist_seeds must be nonempty")
        seeds = tuple(finalist_seeds)
    else:
        raise ValueError(f"unknown phase {phase!r}")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not runs:
        raise ValueError("runs must be nonempty")

    specs: list[CommandSpec] = []
    for run in runs:
        for seed in seeds:
            output_dir = output_root / phase / run.key / f"seed-{seed}"
            argv = [
                python,
                str(TRAIN_SCRIPT),
                "--model",
                run.model,
                "--capacity",
                run.capacity,
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--training-drive-distribution",
                run.drive_distribution,
                "--training-objective",
                run.training_objective,
                "--dtype",
                dtype,
                "--output-dir",
                str(output_dir),
            ]
            if device is not None:
                argv.extend(("--device", device))
            argv.extend(extra_train_args)
            specs.append(
                CommandSpec(
                    phase=phase,
                    run=run,
                    seed=seed,
                    argv=tuple(argv),
                    output_dir=output_dir,
                    report_path=output_dir / f"{run.model}_{run.capacity}.json",
                )
            )
    report_paths = [spec.report_path for spec in specs]
    if len(set(report_paths)) != len(report_paths):
        raise AssertionError("study commands would overwrite a report")
    return tuple(specs)


def command_manifest(specs: Sequence[CommandSpec]) -> dict[str, Any]:
    """Return a machine-readable record that can be stored before execution."""

    return {
        "schema_version": 1,
        "train_script": str(TRAIN_SCRIPT),
        "commands": [
            {
                "phase": spec.phase,
                "run": asdict(spec.run),
                "seed": spec.seed,
                "argv": list(spec.argv),
                "shell": shlex.join(spec.argv),
                "output_dir": str(spec.output_dir),
                "expected_report": str(spec.report_path),
            }
            for spec in specs
        ],
    }


def execute_commands(specs: Sequence[CommandSpec]) -> None:
    """Execute commands sequentially and fail at the first unsuccessful run."""

    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec.run.key}, seed {spec.seed}: "
            f"{shlex.join(spec.argv)}",
            flush=True,
        )
        # Every executable argument is resolved from this module's declarative
        # run matrix or an explicit CLI value; no shell is involved.
        subprocess.run(spec.argv, check=True, cwd=Path.cwd())  # noqa: S603
        if not spec.report_path.is_file():
            raise RuntimeError(
                f"training command completed without expected report {spec.report_path}"
            )


def discover_reports(
    output_root: Path, phase: Literal["early", "finalists"]
) -> dict[str, list[Path]]:
    """Discover one ``train.py`` report beneath each run/seed directory."""

    phase_root = output_root / phase
    result: dict[str, list[Path]] = {}
    if not phase_root.exists():
        raise FileNotFoundError(f"study phase directory does not exist: {phase_root}")
    for run_dir in sorted(path for path in phase_root.iterdir() if path.is_dir()):
        reports: list[Path] = []
        for seed_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            candidates = sorted(seed_dir.glob("*.json"))
            if len(candidates) != 1:
                raise ValueError(
                    f"expected exactly one JSON report in {seed_dir}, found "
                    f"{len(candidates)}"
                )
            reports.append(candidates[0])
        if reports:
            result[run_dir.name] = reports
    if not result:
        raise ValueError(f"no reports found beneath {phase_root}")
    return result


def validate_declared_report_set(
    reports_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    phase: Literal["early", "finalists"],
    expected_seeds: Sequence[int],
) -> None:
    """Reject missing replicates and reports filed under the wrong study key."""

    expected = tuple(expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be nonempty and unique")
    required_count = 1 if phase == "early" else 3
    if len(expected) != required_count:
        raise ValueError(f"{phase} aggregation requires {required_count} seed(s)")
    for key, reports in reports_by_candidate.items():
        if key not in RUN_BY_KEY:
            raise ValueError(f"report directory has undeclared study key {key!r}")
        run = RUN_BY_KEY[key]
        actual_seeds: list[int] = []
        for report in reports:
            try:
                config = report["run_config"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"{key} report has no run_config") from error
            declared = {
                "model": run.model,
                "capacity": run.capacity,
                "training_drive_distribution": run.drive_distribution,
                "training_objective": run.training_objective,
            }
            for field, expected_value in declared.items():
                if config.get(field) != expected_value:
                    raise ValueError(
                        f"{key} report declares {field}={config.get(field)!r}, "
                        f"expected {expected_value!r}"
                    )
            seed = config.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"{key} report seed must be an integer")
            actual_seeds.append(seed)
        if sorted(actual_seeds) != sorted(expected):
            raise ValueError(
                f"{key} has training seeds {sorted(actual_seeds)}, expected "
                f"{sorted(expected)}"
            )


def _nested_float(report: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"report is missing {'.'.join(path)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"report metric {'.'.join(path)} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"report metric {'.'.join(path)} is not finite")
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric values must be nonempty and finite")
    return {
        "mean": statistics.fmean(values),
        "seed_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "replicates": len(values),
    }


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "id_relative_l2": ("evaluation", "splits", "interpolation", "relative_l2_mean"),
    "unseen_geometry_relative_l2": (
        "evaluation",
        "splits",
        "unseen_geometry_modes",
        "relative_l2_mean",
    ),
    "stronger_deformation_relative_l2": (
        "evaluation",
        "splits",
        "stronger_deformation",
        "relative_l2_mean",
    ),
    "unseen_frequency_relative_l2": (
        "evaluation",
        "splits",
        "unseen_boundary_frequencies",
        "relative_l2_mean",
    ),
    "maximum_principle_violation": (
        "evaluation",
        "splits",
        "interpolation",
        "certified_maximum_principle_violation_mean",
    ),
    "boundary_trace_relative_l2": (
        "evaluation",
        "boundary_trace",
        "interpolation",
        "relative_l2_mean",
    ),
    "normalized_laplacian_l2": (
        "evaluation",
        "harmonic_residual",
        "normalized_laplacian_l2_mean",
    ),
    "mode_3_relative_l2": (
        "evaluation",
        "mode_response",
        "3",
        "relative_l2_mean",
    ),
    "mode_4_relative_l2": (
        "evaluation",
        "mode_response",
        "4",
        "relative_l2_mean",
    ),
    "similarity_covariance_max": (
        "evaluation",
        "similarity",
        "paired_covariance_error_max",
    ),
    "superposition_relative_l2_max": (
        "evaluation",
        "drive_linearity",
        "superposition_relative_l2_max",
    ),
    "zero_drive_rms_max": (
        "evaluation",
        "drive_linearity",
        "zero_drive_rms_max",
    ),
}


def _refinement_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    try:
        resolution = report["evaluation"]["resolution"]
    except (KeyError, TypeError) as error:
        raise ValueError("report is missing evaluation.resolution") from error
    if not isinstance(resolution, Mapping) or len(resolution) < 3:
        raise ValueError("resolution study must contain at least three resolutions")
    try:
        ordered = sorted((int(key), key) for key in resolution)
    except (TypeError, ValueError) as error:
        raise ValueError("resolution-study keys must be integer strings") from error
    if len({integer for integer, _ in ordered}) != len(ordered):
        raise ValueError("resolution-study resolutions must be unique")
    deltas_to_finest = [
        _nested_float(
            report,
            ("evaluation", "resolution", key, "change_from_finest_mean"),
        )
        for _, key in ordered
    ]
    successive = [
        _nested_float(
            report,
            ("evaluation", "resolution", key, "change_from_previous_mean"),
        )
        for _, key in ordered[1:]
    ]
    nonnegative = all(delta >= 0.0 for delta in (*deltas_to_finest, *successive))
    finest_reference_zero = deltas_to_finest[-1] <= REFINEMENT_ABSOLUTE_TOLERANCE
    fine_pair_contracts = (
        successive[-1] <= successive[-2] + REFINEMENT_ABSOLUTE_TOLERANCE
    )
    fine_pair_is_cauchy = successive[-1] <= REFINEMENT_FINE_CAUCHY_TOLERANCE
    return {
        "passed": (
            nonnegative
            and finest_reference_zero
            and fine_pair_contracts
            and fine_pair_is_cauchy
        ),
        "resolutions": [integer for integer, _ in ordered],
        "change_from_finest_mean": deltas_to_finest,
        "change_from_previous_mean": successive,
        "criterion": (
            "finite Cauchy evidence: the finest successive refinement change "
            "must not exceed the preceding change and must be <= "
            f"{REFINEMENT_FINE_CAUCHY_TOLERANCE:.6g}; this is not a proof of "
            "asymptotic convergence"
        ),
    }


def _validate_replicates(reports: Sequence[Mapping[str, Any]]) -> None:
    if not reports:
        raise ValueError("each candidate must have at least one report")
    signatures: set[str] = set()
    seeds: list[int] = []
    for report in reports:
        try:
            config = report["run_config"]
            signature = json.dumps(
                {key: value for key, value in config.items() if key != "seed"},
                sort_keys=True,
            )
            seed = config["seed"]
        except (KeyError, TypeError) as error:
            raise ValueError("report has an incomplete run_config") from error
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("run_config.seed must be an integer")
        signatures.add(signature)
        seeds.append(seed)
    if len(signatures) != 1:
        raise ValueError("candidate reports do not share one evaluation configuration")
    if len(set(seeds)) != len(seeds):
        raise ValueError("candidate reports contain duplicate training seeds")


def aggregate_candidate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce seed replicates while retaining worst-case contract diagnostics."""

    _validate_replicates(reports)
    metrics = {
        name: _summary([_nested_float(report, path) for report in reports])
        for name, path in METRIC_PATHS.items()
    }
    refinements = [_refinement_gate(report) for report in reports]
    return {
        "replicates": len(reports),
        "training_seeds": [int(report["run_config"]["seed"]) for report in reports],
        "run_config_invariants": {
            key: reports[0]["run_config"][key]
            for key in (
                "model",
                "capacity",
                "steps",
                "training_drive_distribution",
                "training_objective",
                "evaluation_seed",
                "evaluation_cases",
                "evaluation_boundary_points",
                "evaluation_query_points",
                "harmonic_cases",
            )
        },
        "parameters": _summary([float(report["parameters"]) for report in reports]),
        "elapsed_seconds": _summary(
            [float(report["elapsed_seconds"]) for report in reports]
        ),
        "metrics": metrics,
        "refinement_replicates": refinements,
    }


def _check(value: float, operator: str, threshold: float) -> dict[str, Any]:
    if operator == "<":
        passed = value < threshold
    elif operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"unknown comparison operator {operator!r}")
    return {
        "passed": passed,
        "value": value,
        "operator": operator,
        "threshold": threshold,
    }


def evaluate_selection_gates(
    candidate: Mapping[str, Any],
    *,
    current_control_id: float,
    dense_control_id: float,
) -> dict[str, Any]:
    """Apply every numerical and structural gate in the approved plan.

    Accuracy and PDE metrics use means over training seeds.  Exact-contract
    metrics use the worst reported seed.  This prevents replicate averaging
    from concealing a broken equivariance or superposition contract.
    """

    metrics = candidate["metrics"]
    candidate_id = float(metrics["id_relative_l2"]["mean"])
    gap = current_control_id - dense_control_id
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError(
            "gap closure is undefined unless the dense control improves on "
            "the current control"
        )
    gap_closure = (current_control_id - candidate_id) / gap
    geometry = float(metrics["unseen_geometry_relative_l2"]["mean"])
    geometry_ratio = math.inf if candidate_id == 0.0 else geometry / candidate_id

    checks = {
        "mode_3": _check(float(metrics["mode_3_relative_l2"]["mean"]), "<", 0.20),
        "mode_4": _check(float(metrics["mode_4_relative_l2"]["mean"]), "<", 0.20),
        "id_accuracy": _check(candidate_id, "<=", 0.20),
        "dense_gap_closure": _check(gap_closure, ">=", 0.75),
        "geometry_ood_ratio": _check(geometry_ratio, "<=", 1.20),
        "boundary_trace": _check(
            float(metrics["boundary_trace_relative_l2"]["mean"]), "<=", 0.20
        ),
        "laplacian_residual": _check(
            float(metrics["normalized_laplacian_l2"]["mean"]), "<=", 0.40
        ),
        "similarity_covariance": _check(
            float(metrics["similarity_covariance_max"]["maximum"]),
            "<=",
            FP32_CONTRACT_TOLERANCE,
        ),
        "drive_superposition": _check(
            float(metrics["superposition_relative_l2_max"]["maximum"]),
            "<=",
            FP32_CONTRACT_TOLERANCE,
        ),
        "zero_drive_preservation": _check(
            float(metrics["zero_drive_rms_max"]["maximum"]),
            "<=",
            FP32_CONTRACT_TOLERANCE,
        ),
        "boundary_refinement": {
            "passed": all(
                bool(item["passed"]) for item in candidate["refinement_replicates"]
            ),
            "criterion": "every training-seed replicate passes refinement convergence",
        },
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "derived": {
            "current_control_id": current_control_id,
            "dense_control_id": dense_control_id,
            "current_to_dense_gap": gap,
            "candidate_gap_closure": gap_closure,
            "unseen_geometry_to_id_ratio": geometry_ratio,
            "fp32_contract_tolerance": FP32_CONTRACT_TOLERANCE,
        },
    }


def aggregate_study(
    reports_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    current_control: str = CURRENT_CONTROL,
    dense_control: str = DENSE_CONTROL,
) -> dict[str, Any]:
    """Aggregate candidates and attach advancement decisions."""

    if current_control not in reports_by_candidate:
        raise ValueError(f"missing current control {current_control!r}")
    if dense_control not in reports_by_candidate:
        raise ValueError(f"missing dense control {dense_control!r}")
    candidates = {
        key: aggregate_candidate(reports)
        for key, reports in reports_by_candidate.items()
    }
    evaluation_signatures = {
        (
            report["run_config"]["evaluation_seed"],
            report["run_config"]["evaluation_cases"],
            report["run_config"]["evaluation_boundary_points"],
            report["run_config"]["evaluation_query_points"],
            report["run_config"]["harmonic_cases"],
            report.get("dtype"),
        )
        for reports in reports_by_candidate.values()
        for report in reports
    }
    if len(evaluation_signatures) != 1:
        raise ValueError(
            "all candidates must share one evaluation bank, resolution, and dtype"
        )
    source_digests = {
        report.get("source", {}).get("relevant_source_sha256")
        for reports in reports_by_candidate.values()
        for report in reports
        if report.get("source", {}).get("relevant_source_sha256") is not None
    }
    reports_with_source = sum(
        report.get("source", {}).get("relevant_source_sha256") is not None
        for reports in reports_by_candidate.values()
        for report in reports
    )
    report_count = sum(len(reports) for reports in reports_by_candidate.values())
    if reports_with_source not in (0, report_count):
        raise ValueError("source fingerprints must be present on every report or none")
    if len(source_digests) > 1:
        raise ValueError("all candidate reports must share one source fingerprint")
    current_id = float(candidates[current_control]["metrics"]["id_relative_l2"]["mean"])
    dense_id = float(candidates[dense_control]["metrics"]["id_relative_l2"]["mean"])
    for candidate in candidates.values():
        candidate["selection_gates"] = evaluate_selection_gates(
            candidate,
            current_control_id=current_id,
            dense_control_id=dense_id,
        )
    for key, candidate in candidates.items():
        run = RUN_BY_KEY.get(key)
        role = "candidate" if run is None else run.role
        eligible = True if run is None else run.eligible_for_advancement
        candidate["study_role"] = role
        candidate["eligible_for_advancement"] = eligible
        candidate["advance"] = bool(eligible and candidate["selection_gates"]["passed"])
    result = {
        "schema_version": 1,
        "replicate_reduction": (
            "mean across training seeds for accuracy/PDE metrics; global maximum "
            "for fp32 contracts; every seed must converge under refinement"
        ),
        "controls": {
            "current_moment": current_control,
            "dense_pair": dense_control,
        },
        "gate_policy": {
            "mode_3_relative_l2_strict_upper_bound": 0.20,
            "mode_4_relative_l2_strict_upper_bound": 0.20,
            "id_relative_l2_maximum": 0.20,
            "current_to_dense_gap_closure_minimum": 0.75,
            "unseen_geometry_to_id_ratio_maximum": 1.20,
            "boundary_trace_relative_l2_maximum": 0.20,
            "normalized_laplacian_l2_maximum": 0.40,
            "fp32_contract_tolerance": FP32_CONTRACT_TOLERANCE,
            "refinement": (
                "the two finest successive changes provide finite Cauchy evidence: "
                "the final change contracts and is at most 0.05"
            ),
        },
        "candidates": candidates,
    }
    if source_digests:
        result["relevant_source_sha256"] = next(iter(source_digests))
    factorial_keys = (
        "factorial_minimal_moment",
        "factorial_encoded_moment",
        "factorial_minimal_pair",
        "factorial_encoded_pair",
    )
    if all(key in candidates for key in factorial_keys):
        result["factorial_effects"] = _factorial_effects(candidates)
    return result


def _factorial_effects(candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Report decoder and encoder contrasts without conflating their effects."""

    metric_names = (
        "id_relative_l2",
        "unseen_geometry_relative_l2",
        "stronger_deformation_relative_l2",
        "unseen_frequency_relative_l2",
    )
    minimal_moment = candidates["factorial_minimal_moment"]["metrics"]
    encoded_moment = candidates["factorial_encoded_moment"]["metrics"]
    minimal_pair = candidates["factorial_minimal_pair"]["metrics"]
    encoded_pair = candidates["factorial_encoded_pair"]["metrics"]
    effects: dict[str, Any] = {}
    for name in metric_names:
        mm = float(minimal_moment[name]["mean"])
        em = float(encoded_moment[name]["mean"])
        mp = float(minimal_pair[name]["mean"])
        ep = float(encoded_pair[name]["mean"])
        effects[name] = {
            "pair_decoder_gain_with_minimal_boundary": mm - mp,
            "pair_decoder_gain_with_encoded_boundary": em - ep,
            "boundary_encoder_gain_with_moment_decoder": mm - em,
            "boundary_encoder_gain_with_pair_decoder": mp - ep,
            "boundary_encoder_relative_gain_with_pair_decoder": (
                (mp - ep) / mp if mp > 0.0 else math.nan
            ),
        }
    geometry_gain = effects["unseen_geometry_relative_l2"][
        "boundary_encoder_relative_gain_with_pair_decoder"
    ]
    deformation_gain = effects["stronger_deformation_relative_l2"][
        "boundary_encoder_relative_gain_with_pair_decoder"
    ]
    return {
        "error_reduction_sign_convention": (
            "positive means the named replacement reduced relative-L2 error"
        ),
        "metrics": effects,
        "boundary_encoder_materiality_threshold": 0.05,
        "meets_declared_pair_encoder_practical_effect_threshold": (
            geometry_gain >= 0.05 and deformation_gain >= 0.05
        ),
        "practical_effect_rule": (
            "the encoded pair must improve both unseen-geometry and "
            "stronger-deformation mean errors by at least 5% relative to minimal "
            "pair; this effect-size flag is not a paired uncertainty interval"
        ),
    }


def _load_reports(paths_by_candidate: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, paths in paths_by_candidate.items():
        reports = []
        for path in paths:
            try:
                reports.append(json.loads(path.read_text()))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON report {path}: {error}") from error
        result[key] = reports
    return result


def _tensor_json(tensor: Any) -> Any:
    """Convert a detached real or complex torch tensor to JSON-native values."""

    tensor = tensor.detach().cpu()
    if tensor.is_complex():
        return {
            "real": tensor.real.tolist(),
            "imaginary": tensor.imag.tolist(),
            "magnitude": tensor.abs().tolist(),
        }
    return tensor.tolist()


def make_spectral_report(
    checkpoint_path: Path,
    *,
    device_name: str = "cpu",
    dtype_name: str = "float32",
    n_boundary: int = 64,
    n_query_angles: int = 64,
    radii: Sequence[float] = (0.25, 0.5, 0.75),
    maximum_mode: int = 8,
    requested_ranks: Sequence[int] = (0, 1, 2, 3, 5, 8, 16, 32, 64),
    rank_rtol: float | None = None,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Extract SVD, Fourier transfer, and typed moments from one checkpoint."""

    import torch
    from conformal_laplace import ConformalGeometry, HarmonicDrive, build_domain_sample
    from provenance import source_provenance
    from spectral_diagnostics import (
        compare_operator_spectra,
        concentric_ring_preimages,
        extract_discrete_operator,
        fourier_moment_family_norms,
        fourier_transfer_matrix,
        uniform_angles,
        unit_disk_poisson_matrix,
    )
    from train import make_model

    from physicsnemo.mesh import DomainMesh

    _positive_integer("n_boundary", n_boundary)
    _positive_integer("n_query_angles", n_query_angles)
    if dtype_name == "float32":
        dtype = torch.float32
        complex_dtype = torch.complex64
    elif dtype_name == "float64":
        dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        raise ValueError("dtype_name must be float32 or float64")
    if not radii or any(
        not math.isfinite(radius) or not 0.0 <= radius < 1.0 for radius in radii
    ):
        raise ValueError("radii must be finite, nonempty, and lie in [0, 1)")
    if maximum_mode < 0:
        raise ValueError("maximum_mode must be nonnegative")
    if 2 * maximum_mode >= min(n_boundary, n_query_angles):
        raise ValueError("maximum_mode must lie below source and query Nyquist orders")

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    try:
        model_name = checkpoint["model"]
        capacity = checkpoint["capacity"]
        state_dict = checkpoint["state_dict"]
    except KeyError as error:
        raise ValueError(
            f"checkpoint is missing required key {error.args[0]!r}"
        ) from error
    torch.manual_seed(int(checkpoint.get("run_config", {}).get("seed", 0)))
    evaluator_source = source_provenance()
    checkpoint_source = checkpoint.get("source")
    checkpoint_digest = (
        checkpoint_source.get("relevant_source_sha256")
        if isinstance(checkpoint_source, Mapping)
        else None
    )
    evaluator_digest = evaluator_source["relevant_source_sha256"]
    source_matches = (
        None if checkpoint_digest is None else checkpoint_digest == evaluator_digest
    )
    if source_matches is False and not allow_source_mismatch:
        raise ValueError(
            "checkpoint source fingerprint differs from the spectral evaluator; "
            "pass allow_source_mismatch=True only for an explicitly historical audit"
        )
    model = make_model(model_name, capacity).to(device=device, dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()

    radius_tensor = torch.tensor(radii, device=device, dtype=dtype)
    queries, query_angles = concentric_ring_preimages(
        radius_tensor, n_query_angles, angular_offset=0.173
    )
    geometry = ConformalGeometry(
        modes=(), coefficients=torch.empty(0, device=device, dtype=complex_dtype)
    )
    drive = HarmonicDrive(
        constant=torch.zeros((), device=device, dtype=dtype),
        modes=(),
        coefficients=torch.empty(0, device=device, dtype=complex_dtype),
    )
    domain = build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        query_preimages=queries.reshape(-1),
    ).domain
    boundary_angles = uniform_angles(
        n_boundary,
        offset=math.pi / n_boundary,
        device=device,
        dtype=dtype,
    )
    learned = extract_discrete_operator(model, domain).reshape(
        len(radii), n_query_angles, n_boundary
    )
    analytic = unit_disk_poisson_matrix(boundary_angles, queries).reshape_as(learned)
    maximum_rank = min(learned.numel() // n_boundary, n_boundary)
    ranks = tuple(
        sorted(set(rank for rank in requested_ranks if 0 <= rank <= maximum_rank))
    )
    if not ranks:
        raise ValueError("no requested rank is valid for the extracted operator")
    if rank_rtol is None:
        effective_rank_rtol = max(learned.shape[0] * learned.shape[1], n_boundary)
        effective_rank_rtol *= torch.finfo(dtype).eps
    else:
        if not math.isfinite(rank_rtol) or rank_rtol < 0.0:
            raise ValueError("rank_rtol must be finite and nonnegative")
        effective_rank_rtol = rank_rtol
    comparison = compare_operator_spectra(
        learned.reshape(-1, n_boundary),
        analytic.reshape(-1, n_boundary),
        ranks,
        rank_rtol=effective_rank_rtol,
    )
    input_modes = tuple(range(maximum_mode + 1))
    output_modes = tuple(range(-maximum_mode, maximum_mode + 1))
    learned_transfer = fourier_transfer_matrix(
        learned,
        boundary_angles,
        query_angles,
        input_modes=input_modes,
        output_modes=output_modes,
    )
    analytic_transfer = fourier_transfer_matrix(
        analytic,
        boundary_angles,
        query_angles,
        input_modes=input_modes,
        output_modes=output_modes,
    )

    moment_families: dict[str, Any] | None = None
    moment_model: Any = model
    moment_drive = "as supplied"
    if hasattr(model, "residual_model"):
        residual_model = model.residual_model

        class _MeanFreeEncoder:
            def encode(self, incoming: DomainMesh) -> Any:
                boundary = incoming.boundaries["dirichlet"]
                values = boundary.cell_data["boundary_value"]
                weights = boundary.cell_areas
                anchor = values[0]
                mean = anchor + torch.sum(weights * (values - anchor)) / weights.sum()
                residual_boundary = boundary.with_data(
                    cell_data={"boundary_value": values - mean}
                )
                return residual_model.encode(
                    DomainMesh(
                        interior=incoming.interior,
                        boundaries={"dirichlet": residual_boundary},
                        global_data=incoming.global_data,
                    )
                )

        moment_model = _MeanFreeEncoder()
        moment_drive = "mean-free residual used by constant-lifted model"
    if hasattr(moment_model, "encode"):
        moments = fourier_moment_family_norms(
            moment_model,
            domain,
            boundary_angles,
            modes=input_modes,
        )
        moment_families = {
            str(mode): [asdict(layer) for layer in layers]
            for mode, layers in moments.items()
        }

    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "model": model_name,
        "capacity": capacity,
        "device": str(device),
        "dtype": dtype_name,
        "checkpoint_run_config": checkpoint.get("run_config"),
        "checkpoint_source": checkpoint_source,
        "evaluator_source": evaluator_source,
        "source_matches_evaluator": source_matches,
        "grid": {
            "n_boundary": n_boundary,
            "n_query_angles": n_query_angles,
            "radii": list(radii),
            "boundary_angle_offset": math.pi / n_boundary,
            "query_angle_offset": 0.173,
            "input_modes": list(input_modes),
            "output_modes": list(output_modes),
        },
        "spectrum": {
            "numerical_rank_relative_tolerance": effective_rank_rtol,
            "relative_operator_frobenius_error": comparison.relative_operator_error,
            "learned": {
                "singular_values": _tensor_json(comparison.learned.singular_values),
                "numerical_rank": comparison.learned.numerical_rank,
                "relative_best_rank_errors": comparison.learned.relative_best_rank_errors,
            },
            "analytic": {
                "singular_values": _tensor_json(comparison.analytic.singular_values),
                "numerical_rank": comparison.analytic.numerical_rank,
                "relative_best_rank_errors": comparison.analytic.relative_best_rank_errors,
            },
        },
        "operator_matrix": {
            "learned": _tensor_json(learned),
            "analytic": _tensor_json(analytic),
        },
        "fourier_transfer": {
            "learned": _tensor_json(learned_transfer),
            "analytic": _tensor_json(analytic_transfer),
        },
        "source_moment_family_norms": moment_families,
        "source_moment_drive": moment_drive if moment_families is not None else None,
    }


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result:
        raise argparse.ArgumentTypeError("list must be nonempty")
    return result


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("list must be nonempty")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = subparsers.add_parser(
        "commands", help="emit or execute the declared study commands"
    )
    commands.add_argument("--phase", choices=("early", "finalists"), required=True)
    commands.add_argument("--run", action="append", default=[], dest="runs")
    commands.add_argument("--group", action="append", default=[], dest="groups")
    commands.add_argument(
        "--finalist",
        action="append",
        default=[],
        help="finalist run key; alias for --run in the finalists phase",
    )
    commands.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    commands.add_argument("--early-seed", type=int, default=DEFAULT_EARLY_SEED)
    commands.add_argument(
        "--finalist-seeds", type=_parse_csv_ints, default=DEFAULT_FINALIST_SEEDS
    )
    commands.add_argument("--python", default=sys.executable)
    commands.add_argument("--device")
    commands.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    commands.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands.add_argument("--manifest", type=Path)
    commands.add_argument("--execute", action="store_true")
    commands.add_argument(
        "--full-evaluation",
        action="store_true",
        help="use train.py's full evaluation defaults during early elimination",
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate reports and apply all selection gates"
    )
    aggregate.add_argument("--phase", choices=("early", "finalists"), required=True)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    aggregate.add_argument("--current-control", default=CURRENT_CONTROL)
    aggregate.add_argument("--dense-control", default=DENSE_CONTROL)
    aggregate.add_argument(
        "--expected-seeds",
        type=_parse_csv_ints,
        help="override 17 for early or 17,29,43 for finalists",
    )
    aggregate.add_argument("--output", type=Path)

    spectral = subparsers.add_parser(
        "spectral-report", help="extract the disk operator and spectral diagnostics"
    )
    spectral.add_argument("--checkpoint", type=Path, required=True)
    spectral.add_argument("--device", default="cpu")
    spectral.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    spectral.add_argument("--n-boundary", type=int, default=64)
    spectral.add_argument("--n-query-angles", type=int, default=64)
    spectral.add_argument("--radii", type=_parse_csv_floats, default=(0.25, 0.5, 0.75))
    spectral.add_argument("--maximum-mode", type=int, default=8)
    spectral.add_argument(
        "--ranks", type=_parse_csv_ints, default=(0, 1, 2, 3, 5, 8, 16, 32, 64)
    )
    spectral.add_argument("--rank-rtol", type=float)
    spectral.add_argument("--allow-source-mismatch", action="store_true")
    spectral.add_argument("--output", type=Path)
    return parser


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(serialized, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
        print(output)


def main() -> None:
    """Dispatch study execution, aggregation, or spectral diagnostics."""

    args = _build_parser().parse_args()
    if args.command == "commands":
        selected_keys = tuple(args.runs) + tuple(args.finalist)
        if args.phase == "early" and args.finalist:
            raise ValueError("--finalist is only valid with --phase finalists")
        if args.phase == "finalists" and not selected_keys and not args.groups:
            raise ValueError("finalists phase requires --finalist, --run, or --group")
        # Finalist advancement is defined relative to these controls.  Include
        # their three-seed estimates automatically so the emitted phase is
        # self-contained and can be aggregated without borrowing one-seed
        # early-elimination measurements.
        if args.phase == "finalists":
            selected_keys += (CURRENT_CONTROL, DENSE_CONTROL)
        runs = select_runs(groups=args.groups, keys=selected_keys)
        specs = make_command_specs(
            runs,
            phase=args.phase,
            output_root=args.output_root,
            python=args.python,
            steps=args.steps,
            early_seed=args.early_seed,
            finalist_seeds=args.finalist_seeds,
            device=args.device,
            dtype=args.dtype,
            extra_train_args=(
                ()
                if args.phase == "finalists" or args.full_evaluation
                else EARLY_TRAIN_ARGS
            ),
        )
        manifest = command_manifest(specs)
        _write_or_print(manifest, args.manifest)
        if args.execute:
            execute_commands(specs)
        return
    if args.command == "aggregate":
        paths = discover_reports(args.output_root, args.phase)
        loaded = _load_reports(paths)
        expected_seeds = args.expected_seeds
        if expected_seeds is None:
            expected_seeds = (
                (DEFAULT_EARLY_SEED,)
                if args.phase == "early"
                else DEFAULT_FINALIST_SEEDS
            )
        validate_declared_report_set(
            loaded,
            phase=args.phase,
            expected_seeds=expected_seeds,
        )
        result = aggregate_study(
            loaded,
            current_control=args.current_control,
            dense_control=args.dense_control,
        )
        result["report_paths"] = {
            key: [str(path.resolve()) for path in values]
            for key, values in paths.items()
        }
        _write_or_print(result, args.output)
        return
    if args.command == "spectral-report":
        result = make_spectral_report(
            args.checkpoint,
            device_name=args.device,
            dtype_name=args.dtype,
            n_boundary=args.n_boundary,
            n_query_angles=args.n_query_angles,
            radii=args.radii,
            maximum_mode=args.maximum_mode,
            requested_ranks=args.ranks,
            rank_rtol=args.rank_rtol,
            allow_source_mismatch=args.allow_source_mismatch,
        )
        _write_or_print(result, args.output)
        return
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    main()


__all__ = [
    "CURRENT_CONTROL",
    "DENSE_CONTROL",
    "FP32_CONTRACT_TOLERANCE",
    "STUDY_RUNS",
    "CommandSpec",
    "StudyRun",
    "aggregate_candidate",
    "aggregate_study",
    "command_manifest",
    "discover_reports",
    "evaluate_selection_gates",
    "execute_commands",
    "make_command_specs",
    "make_spectral_report",
    "select_runs",
    "validate_declared_report_set",
]
