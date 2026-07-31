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

"""Validate and reduce the registered Laplace scalar-readout factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import laplace_readout_factorial as experiment

STUDY = "laplace_readout_factorial_v1"
FULL_ARM = "full"
PURE_ARM = "pure"
ID_SPLIT = "interpolation"
REGISTERED_PARAMETERS = 104_261
EXPECTED_SOURCE_FINGERPRINT = (
    "36a0e7406a9d7e432a458cdff61a17a8fcd34d81f4b6819041795e406e4cab2f"
)
EXPECTED_ENVIRONMENT = {
    "cuda_runtime": "13.0",
    "device": "cuda",
    "device_name": "NVIDIA GB200",
    "device_total_memory_bytes": 197_897_617_408,
    "float32_matmul_precision": "highest",
    "platform": "Linux-6.8.0-1046-nvidia-64k-aarch64-with-glibc2.39",
    "python": "3.12.13",
    "torch": "2.13.0+cu130",
    "torch_intraop_threads": 8,
}
TRAINABLE_PARAMETERS = {
    "pure": 104_096,
    "gate_only": 104_165,
    "contraction_only": 104_192,
    "full": 104_261,
}

_FACTOR_ARMS = {
    "gate": {
        "off": ("pure", "contraction_only"),
        "on": ("gate_only", "full"),
        "simple_effects": {
            "contraction_off": ("gate_only", "pure"),
            "contraction_on": ("full", "contraction_only"),
        },
    },
    "contraction": {
        "off": ("pure", "gate_only"),
        "on": ("contraction_only", "full"),
        "simple_effects": {
            "gate_off": ("contraction_only", "pure"),
            "gate_on": ("full", "gate_only"),
        },
    },
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _json_exact(value: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _json_exact(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _json_exact(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("a geometric mean requires at least one value")
    if any(value < 0.0 for value in values):
        raise ValueError("a geometric mean requires nonnegative values")
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _seed_map(values: Mapping[int, Any]) -> dict[str, Any]:
    return {str(seed): values[seed] for seed in experiment.SEEDS}


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator > 0.0:
        return numerator / denominator
    return 1.0 if numerator == 0.0 else None


def _ratio_summary(
    numerator: Mapping[int, float],
    denominator: Mapping[int, float],
) -> dict[str, Any]:
    ratios = {
        seed: _ratio(numerator[seed], denominator[seed]) for seed in experiment.SEEDS
    }
    finite_ratios = tuple(value for value in ratios.values() if value is not None)
    has_infinite_ratio = len(finite_ratios) != len(ratios)
    return {
        "per_seed": _seed_map(ratios),
        "geometric_mean": (
            None if has_infinite_ratio else _geometric_mean(finite_ratios)
        ),
        "has_infinite_ratio": has_infinite_ratio,
        "seeds_below_one": sum(
            value is not None and value < 1.0 for value in ratios.values()
        ),
    }


def _marginal_ratio(
    values: Mapping[str, Mapping[int, float]],
    factor: str,
) -> dict[str, Any]:
    arms = _FACTOR_ARMS[factor]
    ratios: dict[int, float | None] = {}
    for seed in experiment.SEEDS:
        on = math.sqrt(values[arms["on"][0]][seed] * values[arms["on"][1]][seed])
        off = math.sqrt(values[arms["off"][0]][seed] * values[arms["off"][1]][seed])
        ratios[seed] = _ratio(on, off)
    finite_ratios = tuple(value for value in ratios.values() if value is not None)
    has_infinite_ratio = len(finite_ratios) != len(ratios)
    return {
        "per_seed": _seed_map(ratios),
        "geometric_mean": (
            None if has_infinite_ratio else _geometric_mean(finite_ratios)
        ),
        "has_infinite_ratio": has_infinite_ratio,
        "seeds_below_one": sum(
            value is not None and value < 1.0 for value in ratios.values()
        ),
    }


def _protocol_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    run_config = dict(_mapping(report.get("run_config"), "run_config"))
    run_config.pop("seed", None)
    return {
        "study": report.get("study"),
        "run_config_without_seed": run_config,
        "evaluation_protocol": report.get("evaluation_protocol"),
        "accuracy_dtype": report.get("accuracy_dtype"),
        "residual_dtype": report.get("residual_dtype"),
    }


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_evaluation_protocol(seed: int) -> dict[str, Any]:
    config = experiment.experiment_config(seed)
    return {
        "splits": list(experiment.SPLITS),
        "split_seed_offsets": {
            name: experiment._SPLIT_SEED_OFFSETS[name]  # noqa: SLF001
            for name in experiment.SPLITS
        },
        "resolutions": list(experiment.RESOLUTIONS),
        "cases_per_split": config.evaluation_cases,
        "harmonic_cases": config.harmonic_cases,
    }


def _validate_report(
    report: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int, str, str]:
    if report.get("study") != STUDY:
        raise ValueError(f"{label} has the wrong study identifier")

    arm_record = _mapping(report.get("arm"), f"{label} arm")
    arm_key = arm_record.get("key")
    if arm_key not in experiment.ARMS:
        raise ValueError(f"{label} has an unregistered arm {arm_key!r}")
    if not _json_exact(dict(arm_record), asdict(experiment.ARMS[arm_key])):
        raise ValueError(f"{label} arm declaration differs from the registration")

    run_config = _mapping(report.get("run_config"), f"{label} run_config")
    seed = run_config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{label} seed must be an integer")
    if seed not in experiment.SEEDS:
        raise ValueError(f"{label} has an unregistered seed {seed!r}")
    if not _json_exact(dict(run_config), asdict(experiment.experiment_config(seed))):
        raise ValueError(f"{label} run_config differs from the registered protocol")
    if not _json_exact(
        report.get("evaluation_protocol"), _expected_evaluation_protocol(seed)
    ):
        raise ValueError(
            f"{label} evaluation_protocol differs from the registered protocol"
        )
    if report.get("accuracy_dtype") != "float32":
        raise ValueError(f"{label} accuracy dtype differs from the registration")
    if report.get("residual_dtype") != "float64":
        raise ValueError(f"{label} residual dtype differs from the registration")
    if (
        type(report.get("registered_parameters")) is not int
        or report["registered_parameters"] != REGISTERED_PARAMETERS
    ):
        raise ValueError(f"{label} registered parameter count differs")
    if (
        type(report.get("trainable_parameters")) is not int
        or report["trainable_parameters"] != TRAINABLE_PARAMETERS[arm_key]
    ):
        raise ValueError(f"{label} trainable parameter count differs")
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{label} has no training history")
    first_history = _mapping(history[0], f"{label} first history record")
    final_history = _mapping(history[-1], f"{label} final history record")
    if first_history.get("step") != 0 or final_history.get("step") != experiment.STEPS:
        raise ValueError(f"{label} has incomplete training history")
    selected = _mapping(
        report.get("selected_validation"),
        f"{label} selected_validation",
    )
    selected_step = selected.get("step")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or not 0 <= selected_step <= experiment.STEPS
    ):
        raise ValueError(f"{label} has an invalid selected validation step")

    source = _mapping(report.get("source"), f"{label} source")
    source_fingerprint = source.get("relevant_source_sha256")
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in source_fingerprint)
    ):
        raise ValueError(f"{label} has no valid source fingerprint")
    if source_fingerprint != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError(f"{label} was not produced by the registered source")
    if not _json_exact(report.get("environment"), EXPECTED_ENVIRONMENT):
        raise ValueError(f"{label} environment differs from the registered run")

    evaluation = _mapping(report.get("evaluation"), f"{label} evaluation")
    if evaluation.get("accuracy_dtype") != "float32":
        raise ValueError(f"{label} evaluation accuracy dtype differs")
    residual = _mapping(
        evaluation.get("harmonic_residual"),
        f"{label} harmonic_residual",
    )
    if residual.get("dtype") != "float64":
        raise ValueError(f"{label} harmonic residual dtype differs")

    splits = _mapping(evaluation.get("splits"), f"{label} splits")
    if set(splits) != set(experiment.SPLITS):
        raise ValueError(f"{label} split set differs from the registration")
    split_cases = _mapping(evaluation.get("split_cases"), f"{label} split_cases")
    if set(split_cases) != set(experiment.SPLITS):
        raise ValueError(f"{label} split-case set differs from the registration")
    expected_case_metrics = {
        "relative_l2",
        "relative_linf",
        "near_boundary_relative_l2",
        "sampled_boundary_range_violation",
        "certified_maximum_principle_violation",
    }
    for split in experiment.SPLITS:
        split_metrics = _mapping(splits[split], f"{label} split {split}")
        reported_mean = _positive_finite(
            split_metrics.get("relative_l2_mean"),
            f"{label} {split} relative_l2_mean",
        )
        cases = split_cases[split]
        if not isinstance(cases, list) or len(cases) != 64:
            raise ValueError(f"{label} {split} must contain exactly 64 cases")
        case_errors: list[float] = []
        for index, value in enumerate(cases):
            case = _mapping(value, f"{label} {split} case {index}")
            if set(case) != expected_case_metrics:
                raise ValueError(f"{label} {split} case {index} metric set differs")
            for name in expected_case_metrics:
                metric = _nonnegative_finite(
                    case[name],
                    f"{label} {split} case {index} {name}",
                )
                if name == "relative_l2":
                    case_errors.append(metric)
        recomputed_mean = statistics.fmean(case_errors)
        if not math.isclose(
            reported_mean,
            recomputed_mean,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"{label} {split} relative_l2_mean is not its case mean")

    resolution = _mapping(evaluation.get("resolution"), f"{label} resolution")
    expected_resolutions = {str(value) for value in experiment.RESOLUTIONS}
    if set(resolution) != expected_resolutions:
        raise ValueError(f"{label} resolution set differs from the registration")
    for value in experiment.RESOLUTIONS:
        metrics = _mapping(resolution[str(value)], f"{label} resolution {value}")
        _nonnegative_finite(
            metrics.get("relative_l2_mean"),
            f"{label} resolution {value} relative_l2_mean",
        )
    _nonnegative_finite(
        residual.get("normalized_laplacian_l2_mean"),
        f"{label} normalized_laplacian_l2_mean",
    )

    protocol_fingerprint = _fingerprint(_protocol_projection(report))
    return arm_key, seed, source_fingerprint, protocol_fingerprint


def validate_reports(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Require the exact registered 4 x 5 matrix and one source/protocol."""

    expected_count = len(experiment.ARMS) * len(experiment.SEEDS)
    if len(reports) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} per-run reports, got {len(reports)}"
        )

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    source_fingerprints: set[str] = set()
    protocol_fingerprints: set[str] = set()
    for label, report in reports:
        arm, seed, source_fingerprint, protocol_fingerprint = _validate_report(
            report, label=label
        )
        key = (arm, seed)
        if key in indexed:
            raise ValueError(f"duplicate report for arm {arm!r}, seed {seed}")
        indexed[key] = report
        source_fingerprints.add(source_fingerprint)
        protocol_fingerprints.add(protocol_fingerprint)

    expected = {(arm, seed) for arm in experiment.ARMS for seed in experiment.SEEDS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"report matrix differs from the registration; missing={missing}, "
            f"extra={extra}"
        )
    if len(source_fingerprints) != 1:
        raise ValueError("all reports must share one source fingerprint")
    if len(protocol_fingerprints) != 1:
        raise ValueError("all reports must share one protocol fingerprint")
    return indexed


def _split_values(
    reports: Mapping[tuple[str, int], Mapping[str, Any]],
    split: str,
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for arm in experiment.ARMS:
        result[arm] = {}
        for seed in experiment.SEEDS:
            evaluation = reports[(arm, seed)]["evaluation"]
            result[arm][seed] = statistics.fmean(
                case["relative_l2"] for case in evaluation["split_cases"][split]
            )
    return result


def _factorial_effects(
    values: Mapping[str, Mapping[int, float]],
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for factor, arms in _FACTOR_ARMS.items():
        effects[factor] = {
            "marginal_on_over_off": _marginal_ratio(values, factor),
            "simple_on_over_off": {
                background: _ratio_summary(
                    values[numerator],
                    values[denominator],
                )
                for background, (numerator, denominator) in arms[
                    "simple_effects"
                ].items()
            },
        }

    interactions: dict[int, float] = {}
    for seed in experiment.SEEDS:
        interactions[seed] = math.log(
            values["full"][seed]
            * values["pure"][seed]
            / (values["gate_only"][seed] * values["contraction_only"][seed])
        )
    mean_interaction = statistics.fmean(interactions.values())
    effects["log_scale_interaction"] = {
        "definition": "log(full * pure / (gate_only * contraction_only))",
        "per_seed": _seed_map(interactions),
        "mean": mean_interaction,
        "multiplicative_ratio": math.exp(mean_interaction),
    }
    return effects


def _reduce_splits(
    reports: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in experiment.SPLITS:
        values = _split_values(reports, split)
        result[split] = {
            "arm_seed_relative_l2_means": {
                arm: _seed_map(seed_values) for arm, seed_values in values.items()
            },
            "arm_arithmetic_means": {
                arm: statistics.fmean(seed_values.values())
                for arm, seed_values in values.items()
            },
            "paired_arm_over_full": {
                arm: _ratio_summary(seed_values, values[FULL_ARM])
                for arm, seed_values in values.items()
            },
            "factorial_effects": _factorial_effects(values),
        }
    return result


def _reduce_residuals(
    reports: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, dict[int, float]] = {}
    for arm in experiment.ARMS:
        values[arm] = {}
        for seed in experiment.SEEDS:
            residual = reports[(arm, seed)]["evaluation"]["harmonic_residual"]
            values[arm][seed] = float(residual["normalized_laplacian_l2_mean"])
    means = {
        arm: statistics.fmean(seed_values.values())
        for arm, seed_values in values.items()
    }
    return {
        "arm_seed_normalized_laplacian_l2_means": {
            arm: _seed_map(seed_values) for arm, seed_values in values.items()
        },
        "arm_arithmetic_means": means,
        "pure_over_full_arithmetic_mean_ratio": _ratio(
            means[PURE_ARM], means[FULL_ARM]
        ),
        "factorial_marginal_on_over_off": {
            factor: _marginal_ratio(values, factor) for factor in _FACTOR_ARMS
        },
    }


def _is_monotone(errors: Mapping[str, float]) -> bool:
    ordered = [errors[str(value)] for value in experiment.RESOLUTIONS]
    return all(left >= right for left, right in zip(ordered, ordered[1:]))


def _reduce_resolution(
    reports: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    seed_values: dict[str, dict[int, dict[str, float]]] = {}
    arm_means: dict[str, dict[str, float]] = {}
    for arm in experiment.ARMS:
        seed_values[arm] = {}
        for seed in experiment.SEEDS:
            resolution = reports[(arm, seed)]["evaluation"]["resolution"]
            seed_values[arm][seed] = {
                str(value): float(resolution[str(value)]["relative_l2_mean"])
                for value in experiment.RESOLUTIONS
            }
        arm_means[arm] = {
            str(value): statistics.fmean(
                seed_values[arm][seed][str(value)] for seed in experiment.SEEDS
            )
            for value in experiment.RESOLUTIONS
        }

    factor_checks: dict[str, Any] = {}
    for factor, arms in _FACTOR_ARMS.items():
        enabled_arms = tuple(arms["on"])
        enabled_arm_seed_monotone = {
            arm: {
                str(seed): _is_monotone(seed_values[arm][seed])
                for seed in experiment.SEEDS
            }
            for arm in enabled_arms
        }
        factor_checks[factor] = {
            "enabled_arms": list(enabled_arms),
            "enabled_arm_seed_monotone": enabled_arm_seed_monotone,
            "enabled_arm_monotone": {
                arm: _is_monotone(arm_means[arm]) for arm in enabled_arms
            },
        }
        factor_checks[factor]["passed"] = all(
            passed
            for arm in enabled_arms
            for passed in enabled_arm_seed_monotone[arm].values()
        )
    return {
        "arm_seed_relative_l2_means": {
            arm: {str(seed): seed_values[arm][seed] for seed in experiment.SEEDS}
            for arm in experiment.ARMS
        },
        "arm_arithmetic_means": arm_means,
        "arm_monotone_64_to_128_to_256": {
            arm: _is_monotone(errors) for arm, errors in arm_means.items()
        },
        "factor_enabled_arm_checks": factor_checks,
    }


def _pure_decision(
    splits: Mapping[str, Any],
    residuals: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = {
        split: splits[split]["paired_arm_over_full"][PURE_ARM]["geometric_mean"]
        for split in experiment.SPLITS
    }
    accuracy_checks = {
        split: ratio <= (1.10 if split == ID_SPLIT else 1.20)
        for split, ratio in ratios.items()
    }
    pure_seed_residuals = residuals["arm_seed_normalized_laplacian_l2_means"][PURE_ARM]
    full_seed_residuals = residuals["arm_seed_normalized_laplacian_l2_means"][FULL_ARM]
    per_seed_residual_checks = {
        str(seed): {
            "pure_at_most_1e-3": pure_seed_residuals[str(seed)] <= 1.0e-3,
            "pure_at_least_tenfold_lower_than_full": (
                pure_seed_residuals[str(seed)] <= full_seed_residuals[str(seed)] / 10.0
            ),
        }
        for seed in experiment.SEEDS
    }
    residual_checks = {
        "pure_at_most_1e-3_in_every_seed": all(
            checks["pure_at_most_1e-3"] for checks in per_seed_residual_checks.values()
        ),
        "pure_at_least_tenfold_lower_than_full_in_every_seed": all(
            checks["pure_at_least_tenfold_lower_than_full"]
            for checks in per_seed_residual_checks.values()
        ),
    }
    passed = all(accuracy_checks.values()) and all(residual_checks.values())
    return {
        "status": "sufficient" if passed else "not_sufficient",
        "passed": passed,
        "paired_geometric_mean_pure_over_full": ratios,
        "accuracy_checks": accuracy_checks,
        "residual_checks": residual_checks,
        "per_seed_residual_checks": per_seed_residual_checks,
        "pure_residual_arithmetic_mean": residuals["arm_arithmetic_means"][PURE_ARM],
        "full_residual_arithmetic_mean": residuals["arm_arithmetic_means"][FULL_ARM],
    }


def _factor_decision(
    factor: str,
    splits: Mapping[str, Any],
    residuals: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    split_checks: dict[str, Any] = {}
    qualifying_splits: list[str] = []
    for split in experiment.SPLITS:
        marginal = splits[split]["factorial_effects"][factor]["marginal_on_over_off"]
        improvement_passed = marginal["geometric_mean"] <= 0.80
        direction_passed = marginal["seeds_below_one"] >= 4
        qualifies = improvement_passed and direction_passed
        if qualifies:
            qualifying_splits.append(split)
        split_checks[split] = {
            "marginal_geometric_mean_on_over_off": marginal["geometric_mean"],
            "improvement_at_least_20_percent": improvement_passed,
            "seeds_in_improving_direction": marginal["seeds_below_one"],
            "same_direction_at_least_4_of_5": direction_passed,
            "qualifies": qualifies,
        }

    accuracy_passed = len(qualifying_splits) >= 2
    residual_summary = residuals["factorial_marginal_on_over_off"][factor]
    residual_ratio = residual_summary["geometric_mean"]
    residual_passed = all(
        ratio is not None and ratio < 10.0
        for ratio in residual_summary["per_seed"].values()
    )
    resolution_check = resolution["factor_enabled_arm_checks"][factor]
    resolution_passed = bool(resolution_check["passed"])
    earned = accuracy_passed and residual_passed and resolution_passed
    if earned:
        status = "earned"
    elif accuracy_passed and (not residual_passed or not resolution_passed):
        status = "tradeoff"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "earned": earned,
        "qualifying_splits": qualifying_splits,
        "at_least_two_qualifying_splits": accuracy_passed,
        "split_checks": split_checks,
        "marginal_residual_on_over_off": residual_ratio,
        "marginal_residual_per_seed_on_over_off": residual_summary["per_seed"],
        "no_seed_has_tenfold_residual_penalty": residual_passed,
        "resolution": resolution_check,
    }


def reduce_reports(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate all reports, compute paired effects, and apply the bars."""

    indexed = validate_reports(reports)
    first = next(iter(indexed.values()))
    source_fingerprint = first["source"]["relevant_source_sha256"]
    protocol_fingerprint = _fingerprint(_protocol_projection(first))
    splits = _reduce_splits(indexed)
    residuals = _reduce_residuals(indexed)
    resolution = _reduce_resolution(indexed)
    return {
        "schema_version": 1,
        "study": STUDY,
        "source_fingerprint": source_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "registered_arms": {key: asdict(arm) for key, arm in experiment.ARMS.items()},
        "registered_seeds": list(experiment.SEEDS),
        "ratio_convention": (
            "arm/full and factor-on/factor-off; values below one mean lower error"
        ),
        "splits": splits,
        "harmonic_residual": residuals,
        "resolution_transfer": resolution,
        "decisions": {
            "pure_core": _pure_decision(splits, residuals),
            "factors": {
                factor: _factor_decision(
                    factor,
                    splits,
                    residuals,
                    resolution,
                )
                for factor in _FACTOR_ARMS
            },
        },
    }


def load_reports(
    input_directory: Path,
    *,
    exclude: Path | None = None,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Read every per-run JSON beneath one directory in stable path order."""

    if not input_directory.is_dir():
        raise ValueError(f"input directory does not exist: {input_directory}")
    excluded = None if exclude is None else exclude.resolve()
    paths = [
        path
        for path in sorted(input_directory.rglob("*.json"))
        if excluded is None or path.resolve() != excluded
    ]
    reports: list[tuple[str, Mapping[str, Any]]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON report {path}: {error}") from error
        reports.append((str(path), _mapping(value, str(path))))
    return reports


def reduce_directory(input_directory: Path, output: Path) -> dict[str, Any]:
    """Reduce one directory and atomically publish exactly one summary."""

    reports = load_reports(input_directory, exclude=output)
    summary = reduce_reports(reports)
    experiment.atomic_write_json(output, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate, reduce, and publish one factorial summary."""

    args = _parse_args()
    input_directory = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = reduce_directory(input_directory, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "pure_core": summary["decisions"]["pure_core"]["status"],
                "gate": summary["decisions"]["factors"]["gate"]["status"],
                "contraction": summary["decisions"]["factors"]["contraction"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
