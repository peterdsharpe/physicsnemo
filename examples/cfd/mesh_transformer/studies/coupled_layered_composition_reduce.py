# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reduce the registered coupled path-composition study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import coupled_layered_composition as study
import layered_screened_context as shared

BREADTH_SPLITS = ("held_out_twist", "doubled_layers")


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
    if not values or any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require nonnegative finite values")
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _split_value(
    report: dict[str, Any],
    split: str,
    metric: str,
) -> float:
    return float(report["split_evaluation"][split]["metrics"][metric]["mean"])


def _order_value(report: dict[str, Any], metric: str) -> float:
    return float(report["order_challenge"][metric])


def _paired_summary(
    candidate: list[float],
    baseline: list[float],
) -> dict[str, Any]:
    if len(candidate) != len(study.SEEDS) or len(baseline) != len(study.SEEDS):
        raise ValueError("paired summaries require every registered seed")
    if any(value <= 0.0 for value in baseline):
        raise ValueError("paired baselines must be positive")
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
        trained = key[0] != "analytic_path"
        protocol = report["protocol"]
        expected_protocol = {
            "training_applied": trained,
            "train_steps": study.TRAIN_STEPS if trained else 0,
            "train_batch_size": study.TRAIN_BATCH_SIZE,
            "train_layers": study.TRAIN_LAYERS,
            "doubled_layers": study.DOUBLED_LAYERS,
            "train_twists": list(study.TRAIN_TWISTS),
            "held_out_twists": list(study.HELD_OUT_TWISTS),
            "evaluation_profiles_per_split": study.EVALUATION_PROFILES,
            "evaluation_query_points": study.EVALUATION_QUERY_POINTS,
            "order_pairs": study.ORDER_PAIRS,
            "token_width": study.TOKEN_WIDTH,
            "head_width": study.HEAD_WIDTH,
            "generator_width": study.GENERATOR_WIDTH,
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
    generator_counts = {
        index[(arm, seed)]["parameters"]
        for arm in study.GENERATOR_ARMS
        for seed in study.SEEDS
    }
    if len(generator_counts) != 1:
        raise ValueError("learned generator arms do not have matched capacity")
    if any(index[("analytic_path", seed)]["parameters"] != 0 for seed in study.SEEDS):
        raise ValueError("analytic oracle unexpectedly has learned parameters")

    deterministic_keys = (
        "parameters",
        "training_history",
        "split_evaluation",
        "order_challenge",
        "local_map_evaluation",
        "reference_certification",
    )
    reference = index[("analytic_path", study.SEEDS[0])]
    for seed in study.SEEDS[1:]:
        report = index[("analytic_path", seed)]
        if any(report[name] != reference[name] for name in deterministic_keys):
            raise ValueError("analytic oracle differs across seed labels")
    return index


def _split_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    split: str,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [
            _split_value(index[("path_generator", seed)], split, metric)
            for seed in study.SEEDS
        ],
        [
            _split_value(index[("ordered_pool", seed)], split, metric)
            for seed in study.SEEDS
        ],
    )


def _order_comparison(
    index: dict[tuple[str, int], dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    return _paired_summary(
        [_order_value(index[("path_generator", seed)], metric) for seed in study.SEEDS],
        [_order_value(index[("ordered_pool", seed)], metric) for seed in study.SEEDS],
    )


def apply_registered_decision(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    interpolation = _split_comparison(
        index,
        split="in_distribution",
        metric="operator_relative_l2",
    )
    interpolation_pass = interpolation["geometric_mean_ratio"] <= 1.2
    contrast_by_seed = {
        str(seed): _order_value(
            index[("path_generator", seed)],
            "contrast_relative_l2",
        )
        for seed in study.SEEDS
    }
    contrast_pass_seed_count = sum(value <= 0.2 for value in contrast_by_seed.values())
    paired_field = _order_comparison(index, metric="paired_field_relative_l2")
    paired_field_pass = (
        paired_field["geometric_mean_ratio"] <= 0.5
        and paired_field["candidate_better_seed_count"] >= 4
    )
    composition_claim_earned = (
        interpolation_pass and contrast_pass_seed_count >= 4 and paired_field_pass
    )

    breadth = {
        split: _split_comparison(
            index,
            split=split,
            metric="cross_channel_relative_l2",
        )
        for split in BREADTH_SPLITS
    }
    breadth_passes = {
        split: (
            summary["geometric_mean_ratio"] <= 0.5
            and summary["candidate_better_seed_count"] >= 4
        )
        for split, summary in breadth.items()
    }
    breadth_transfer_earned = all(breadth_passes.values())

    certification_maxima = {
        name: max(
            float(report["reference_certification"][name]) for report in index.values()
        )
        for name in (
            "boundary_max_abs_error",
            "transfer_determinant_max_abs_error",
            "rotation_covariance_max_abs_error",
        )
    }
    multiset_maximum = max(
        _order_value(report, "multiset_max_abs_difference") for report in index.values()
    )
    true_contrast_minimum = min(
        _order_value(report, "true_contrast_relative_l2") for report in index.values()
    )
    sorted_predicted_contrast_maximum = max(
        _order_value(
            index[("sorted_generator", seed)],
            "predicted_contrast_relative_l2",
        )
        for seed in study.SEEDS
    )
    oracle_split_maximum = max(
        _split_value(index[("analytic_path", seed)], split, metric)
        for seed in study.SEEDS
        for split in study.SPLITS
        for metric in ("operator_relative_l2", "cross_channel_relative_l2")
    )
    oracle_pair_maximum = max(
        _order_value(index[("analytic_path", seed)], "paired_field_relative_l2")
        for seed in study.SEEDS
    )
    oracle_contrast_maximum = max(
        _order_value(index[("analytic_path", seed)], "contrast_relative_l2")
        for seed in study.SEEDS
    )
    numerical_sanity = (
        certification_maxima["boundary_max_abs_error"] <= 1.0e-10
        and certification_maxima["transfer_determinant_max_abs_error"] <= 1.0e-8
        and certification_maxima["rotation_covariance_max_abs_error"] <= 1.0e-10
        and multiset_maximum <= 1.0e-12
        and true_contrast_minimum >= 0.01
        and oracle_split_maximum <= 1.0e-12
        and oracle_pair_maximum <= 1.0e-12
        and oracle_contrast_maximum <= 1.0e-12
    )
    order_challenge_valid = sorted_predicted_contrast_maximum <= 1.0e-12

    if not numerical_sanity:
        verdict = "numerically_unresolved"
    elif not order_challenge_valid:
        verdict = "order_challenge_invalid"
    elif composition_claim_earned and breadth_transfer_earned:
        verdict = "coupled_path_composition_earned"
    elif composition_claim_earned:
        verdict = "order_only_not_breadth"
    else:
        verdict = "coupled_composition_not_earned"
    return {
        "verdict": verdict,
        "numerical_sanity": numerical_sanity,
        "order_challenge_valid": order_challenge_valid,
        "certification_maxima": certification_maxima,
        "multiset_maximum": multiset_maximum,
        "true_contrast_minimum": true_contrast_minimum,
        "sorted_predicted_contrast_maximum": sorted_predicted_contrast_maximum,
        "oracle_split_maximum": oracle_split_maximum,
        "oracle_pair_maximum": oracle_pair_maximum,
        "oracle_contrast_maximum": oracle_contrast_maximum,
        "composition": {
            "interpolation": interpolation,
            "interpolation_pass": interpolation_pass,
            "contrast_relative_l2_by_seed": contrast_by_seed,
            "contrast_pass_seed_count": contrast_pass_seed_count,
            "paired_field": paired_field,
            "paired_field_pass": paired_field_pass,
            "composition_claim_earned": composition_claim_earned,
        },
        "breadth_transfer": {
            "comparisons": breadth,
            "passes": breadth_passes,
            "breadth_transfer_earned": breadth_transfer_earned,
        },
    }


def summarize(
    index: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in study.ARMS:
        reports = [index[(arm, seed)] for seed in study.SEEDS]
        local_values = [
            report["local_map_evaluation"]["relative_frobenius"]
            for report in reports
            if report["local_map_evaluation"] is not None
        ]
        arms[arm] = {
            "parameters": reports[0]["parameters"],
            "operator_relative_l2_geometric_means": {
                split: _geometric_mean(
                    [
                        _split_value(report, split, "operator_relative_l2")
                        for report in reports
                    ]
                )
                for split in study.SPLITS
            },
            "cross_channel_relative_l2_geometric_means": {
                split: _geometric_mean(
                    [
                        _split_value(report, split, "cross_channel_relative_l2")
                        for report in reports
                    ]
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
            "local_map_relative_frobenius_geometric_mean": (
                _geometric_mean(local_values) if local_values else None
            ),
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
    shared.atomic_write_json(args.output, summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
