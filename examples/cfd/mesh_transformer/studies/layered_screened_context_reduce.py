# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the registered layered variable-coefficient context study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import layered_screened_context as study

GUARD_SPLITS = ("in_distribution", "held_out_modes")
COEFFICIENT_OOD_SPLITS = ("ood_low_coefficient", "ood_high_coefficient")


def _numeric_leaves(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from _numeric_leaves(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _split_value(report: dict[str, Any], split: str) -> float:
    return float(
        report["split_evaluation"][split]["metrics"]["field_relative_l2"]["mean"]
    )


def _order_value(report: dict[str, Any], metric: str) -> float:
    return float(report["order_challenge"][metric])


def _paired_summary(
    candidate: list[float],
    baseline: list[float],
) -> dict[str, Any]:
    if len(candidate) != len(study.SEEDS) or len(baseline) != len(study.SEEDS):
        raise ValueError("paired summaries require every registered seed")
    ratios = [
        candidate_value / baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    return {
        "geometric_mean_ratio": _geometric_mean(ratios),
        "candidate_better_seed_count": sum(
            candidate_value < baseline_value
            for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
        ),
        "ratios_by_seed": {
            str(seed): ratio for seed, ratio in zip(study.SEEDS, ratios, strict=True)
        },
    }


def validate_reports(
    reports: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(arm, seed) for arm in study.ARMS for seed in study.SEEDS}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        if report.get("study") != study.STUDY:
            raise ValueError("input contains a report from another study")
        key = (str(report["arm"]), int(report["seed"]))
        if key in index:
            raise ValueError(f"duplicate report {key}")
        if key not in expected:
            raise ValueError(f"unregistered report {key}")
        if not all(math.isfinite(value) for value in _numeric_leaves(report)):
            raise ValueError(f"nonfinite value in report {key}")
        trained = key[0] != "fixed_optical"
        protocol = report["protocol"]
        expected_protocol = {
            "training_applied": trained,
            "train_steps": study.TRAIN_STEPS if trained else 0,
            "train_batch_size": study.TRAIN_BATCH_SIZE,
            "train_modes": list(study.TRAIN_MODES),
            "held_out_modes": list(study.HELD_OUT_MODES),
            "n_layers": study.N_LAYERS,
            "train_kappa_range": list(study.TRAIN_KAPPA_RANGE),
            "low_kappa_range": list(study.LOW_KAPPA_RANGE),
            "high_kappa_range": list(study.HIGH_KAPPA_RANGE),
            "evaluation_profiles_per_split": study.EVALUATION_PROFILES,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "order_pairs": study.ORDER_PAIRS,
        }
        for name, value in expected_protocol.items():
            if protocol.get(name) != value:
                raise ValueError(
                    f"report {key} has nonregistered {name}: "
                    f"{protocol.get(name)!r} != {value!r}"
                )
        if set(report["split_evaluation"]) != set(study.SPLITS):
            raise ValueError(f"report {key} has the wrong evaluation splits")
        index[key] = report
    missing = expected - set(index)
    if missing:
        raise ValueError(f"incomplete factorial: missing={missing}")

    source_hashes = {
        report["source"]["relevant_source_sha256"] for report in index.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("arm reports do not share one source fingerprint")
    if any(index[("fixed_optical", seed)]["parameters"] != 0 for seed in study.SEEDS):
        raise ValueError("fixed optical carrier unexpectedly has learned parameters")
    ordered_counts = {
        index[(arm, seed)]["parameters"]
        for arm in study.ORDERED_ARMS
        for seed in study.SEEDS
    }
    if len(ordered_counts) != 1:
        raise ValueError("ordered learned arms do not have matched capacity")
    scalar_count = index[("scalar_correction", study.SEEDS[0])]["parameters"]
    ordered_count = next(iter(ordered_counts))
    if abs(scalar_count / ordered_count - 1.0) > 0.01:
        raise ValueError("scalar and ordered learned capacities differ by more than 1%")

    deterministic_keys = (
        "parameters",
        "training_history",
        "split_evaluation",
        "order_challenge",
        "reference_certification",
    )
    reference = index[("fixed_optical", study.SEEDS[0])]
    for seed in study.SEEDS[1:]:
        report = index[("fixed_optical", seed)]
        if any(report[name] != reference[name] for name in deterministic_keys):
            raise ValueError("deterministic optical carrier differs across seed labels")
    return index


def _split_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    split: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_split_value(index[(candidate, seed)], split) for seed in study.SEEDS],
        [_split_value(index[(baseline, seed)], split) for seed in study.SEEDS],
    )


def _order_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_order_value(index[(candidate, seed)], metric) for seed in study.SEEDS],
        [_order_value(index[(baseline, seed)], metric) for seed in study.SEEDS],
    )


def _best_ordered_guard(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
    split: str,
) -> dict[str, Any]:
    candidate_values = [
        _split_value(index[(candidate, seed)], split) for seed in study.SEEDS
    ]
    best_values = [
        min(_split_value(index[(arm, seed)], split) for arm in study.ORDERED_ARMS)
        for seed in study.SEEDS
    ]
    return _paired_summary(candidate_values, best_values)


def _nonlocality_candidate(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    candidate: str,
) -> dict[str, Any]:
    guards = {
        split: _best_ordered_guard(index, candidate=candidate, split=split)
        for split in GUARD_SPLITS
    }
    guard_pass = all(
        summary["geometric_mean_ratio"] <= 1.2 for summary in guards.values()
    )
    contrast_by_seed = {
        str(seed): _order_value(
            index[(candidate, seed)],
            "contrast_relative_l2",
        )
        for seed in study.SEEDS
    }
    contrast_pass_seed_count = sum(value <= 0.2 for value in contrast_by_seed.values())
    paired_field = _order_comparison(
        index,
        candidate=candidate,
        baseline="scalar_correction",
        metric="paired_field_relative_l2",
    )
    paired_field_pass = (
        paired_field["geometric_mean_ratio"] <= 0.2
        and paired_field["candidate_better_seed_count"] >= 4
    )
    return {
        "candidate": candidate,
        "guards": guards,
        "guard_pass": guard_pass,
        "contrast_relative_l2_by_seed": contrast_by_seed,
        "contrast_pass_seed_count": contrast_pass_seed_count,
        "paired_field_vs_scalar": paired_field,
        "paired_field_pass": paired_field_pass,
        "nonlocality_claim_earned": (
            guard_pass and contrast_pass_seed_count >= 4 and paired_field_pass
        ),
    }


def _carrier_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    guards = {
        split: _split_comparison(
            index,
            candidate="ordered_carrier",
            baseline="ordered_raw",
            split=split,
        )
        for split in GUARD_SPLITS
    }
    guard_pass = all(
        summary["geometric_mean_ratio"] <= 1.2 for summary in guards.values()
    )
    improvements = {
        split: _split_comparison(
            index,
            candidate="ordered_carrier",
            baseline="ordered_raw",
            split=split,
        )
        for split in COEFFICIENT_OOD_SPLITS
    }
    improvements["layer_order"] = _order_comparison(
        index,
        candidate="ordered_carrier",
        baseline="ordered_raw",
        metric="paired_field_relative_l2",
    )
    improvement_passes = {
        name: (
            summary["geometric_mean_ratio"] <= 0.7
            and summary["candidate_better_seed_count"] >= 4
        )
        for name, summary in improvements.items()
    }
    improved_split_count = sum(improvement_passes.values())
    return {
        "baseline": "ordered_raw",
        "candidate": "ordered_carrier",
        "guards": guards,
        "guard_pass": guard_pass,
        "improvements": improvements,
        "improvement_passes": improvement_passes,
        "improved_split_count": improved_split_count,
        "representation_claim_earned": guard_pass and improved_split_count >= 2,
    }


def apply_registered_decision(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    nonlocality = {
        arm: _nonlocality_candidate(index, candidate=arm) for arm in study.ORDERED_ARMS
    }
    ordered_context_earned = any(
        result["nonlocality_claim_earned"] for result in nonlocality.values()
    )
    carrier = _carrier_comparison(index)

    certification_maxima = {
        name: max(
            float(report["reference_certification"][name]) for report in index.values()
        )
        for name in (
            "constant_profile_max_abs_error",
            "transfer_determinant_max_abs_error",
            "boundary_max_abs_error",
        )
    }
    scalar_input_maximum = max(
        _order_value(report, "scalar_input_max_abs_difference")
        for report in index.values()
    )
    true_contrast_minimum = min(
        _order_value(report, "true_contrast_relative_l2") for report in index.values()
    )
    blind_predicted_contrast_maximum = max(
        _order_value(index[(arm, seed)], "predicted_contrast_relative_l2")
        for arm in ("fixed_optical", "scalar_correction")
        for seed in study.SEEDS
    )
    numerical_sanity = (
        certification_maxima["constant_profile_max_abs_error"] <= 1.0e-10
        and certification_maxima["boundary_max_abs_error"] <= 1.0e-10
        and certification_maxima["transfer_determinant_max_abs_error"] <= 1.0e-8
        and scalar_input_maximum <= 1.0e-12
        and true_contrast_minimum >= 0.02
    )
    order_challenge_valid = blind_predicted_contrast_maximum <= 1.0e-12

    if not numerical_sanity:
        verdict = "numerically_unresolved"
    elif not order_challenge_valid:
        verdict = "order_challenge_invalid"
    elif ordered_context_earned and carrier["representation_claim_earned"]:
        verdict = "ordered_context_and_carrier_earned"
    elif ordered_context_earned:
        verdict = "ordered_context_only"
    elif carrier["representation_claim_earned"]:
        verdict = "carrier_without_nonlocality_unresolved"
    else:
        verdict = "feedforward_context_not_sufficient"
    return {
        "verdict": verdict,
        "numerical_sanity": numerical_sanity,
        "order_challenge_valid": order_challenge_valid,
        "certification_maxima": certification_maxima,
        "scalar_input_maximum": scalar_input_maximum,
        "true_contrast_minimum": true_contrast_minimum,
        "blind_predicted_contrast_maximum": blind_predicted_contrast_maximum,
        "nonlocality": nonlocality,
        "ordered_context_earned": ordered_context_earned,
        "carrier_representation": carrier,
    }


def summarize(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in study.ARMS:
        reports = [index[(arm, seed)] for seed in study.SEEDS]
        arms[arm] = {
            "parameters": reports[0]["parameters"],
            "field_relative_l2_geometric_means": {
                split: _geometric_mean(
                    [_split_value(report, split) for report in reports]
                )
                for split in study.SPLITS
            },
            "order_challenge": {
                metric: _geometric_mean(
                    [_order_value(report, metric) for report in reports]
                )
                for metric in (
                    "paired_field_relative_l2",
                    "contrast_relative_l2",
                    "true_contrast_relative_l2",
                )
            },
        }
    return {
        "study": study.STUDY,
        "arms": arms,
        "decision": apply_registered_decision(index),
        "source_sha256": next(
            iter(
                {
                    report["source"]["relevant_source_sha256"]
                    for report in index.values()
                }
            )
        ),
    }


def reduce_directory(input_dir: Path) -> dict[str, Any]:
    reports = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open() as stream:
            reports.append(json.load(stream))
    return summarize(validate_reports(reports))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = reduce_directory(args.input_dir)
    study.atomic_write_json(args.output, summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
