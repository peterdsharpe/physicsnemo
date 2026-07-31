# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Apply the pre-registered 40k common-master H4 decision rule.

This script is intentionally a pure, fail-closed reduction of the production
audit artifacts.  It does not rerun geometry or choose thresholds.  For each
case it divides the one invariant cover metric by the median *absolute* cyclic
metric across the four frozen starts, then applies the rule recorded in
``book/18-notebook.qmd`` before the deciding output existed.

Run from the repository root after both production artifacts are present::

    python3 examples/cfd/mesh_transformer/studies/phase1_h4_verdict.py \
      examples/cfd/mesh_transformer/results/drivaerml_common_master_run_1_k40000_2026-07-27.json \
      examples/cfd/mesh_transformer/results/drivaerml_common_master_run_118_k40000_2026-07-27.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_DATE = "2026-07-27"
EXPECTED_K = 40_000
EXPECTED_CASES = ("run_1", "run_118")
FROZEN_SEED = 20_300_727
FROZEN_STARTS = {
    "run_1": (11_914_080, 11_878_329, 14_546_186, 2_529_054),
    "run_118": (2_517_171, 651_819, 4_220_546, 6_617_334),
}
EXPECTED_ALGEBRA_TOLERANCE = 1.0e-12
EXPECTED_AUDIT_STATUS = "PASSED_PRODUCTION_COMMON_MASTER_AUDIT"
EXPECTED_AUDIT_SCRIPT_SHA256 = (
    "44a95610f844daba0b3367809fc061e1edbb2c334f37d46fc4ca32b21258aa77"
)
EXPECTED_SOURCE_KIND = "curated_drivaerml_vehicle_tensordict_memmaps"
EXPECTED_NEIGHBOR_BACKEND = "scipy.spatial.cKDTree"
EXPECTED_PRODUCTION_VERSIONS = {
    "python": "3.13.14",
    "numpy": "2.5.1",
    "scipy": "1.16.1",
}
FROZEN_SOURCE_IDENTITIES = {
    "run_1": {
        "n_cells": 17_684_913,
        "array_sha256": {
            "points": (
                "c4b17fc985f84bfe046e948c637c4dff32001b440593620f6f6d6b493f5361a4"
            ),
            "cells": (
                "bb7df92b6535208b4cb81fbe85ea009694d851c351be0351b413b49f02c27def"
            ),
            "CpMeanTrim": (
                "7a41b16506e68981a546e1e00458f22a38e0642685a4863a310802b45dcce095"
            ),
            "wallShearStressMeanTrim": (
                "a651f8d484553891d9f0452a7b3a92c4701e4545a749756fb199ce5ef467a601"
            ),
        },
        "metadata_sha256": {
            "mesh": (
                "65743a44e4907d7f97a690da24f2bcf591849e0f81cc825390fe22725a02204e"
            ),
            "cell_data": (
                "cfce5811c1843a1b9b7c294fdd93d2cb47ddea6b77c28806379c7d46e5a82c6f"
            ),
        },
    },
    "run_118": {
        "n_cells": 17_504_739,
        "array_sha256": {
            "points": (
                "cbbebfd0698cee3bc0125a237ca635bc490810a8d687e39234c3a9d5ebc54fed"
            ),
            "cells": (
                "63977cdca62fbf0f317caefa998eea9522b166b23a51cfb850e8b9fc00a258f7"
            ),
            "CpMeanTrim": (
                "945f515ba8dad418687efa306418cc9daeff0dec31877fc82ce8ead022462cf2"
            ),
            "wallShearStressMeanTrim": (
                "7851f9ada0e40178229ab564d93fb9d8604730555f4ea57088600cc9117bffff"
            ),
        },
        "metadata_sha256": {
            "mesh": (
                "52dc4499048f6acf851b1f6f19b143284fdc97bc0956feee814404bb1fe141fa"
            ),
            "cell_data": (
                "a9ca75da58e64210b6dcfc07ca769fed473add2f0404b04e5b3a119d1f59487d"
            ),
        },
    },
}

METRICS = {
    "pressure_relative_l2_floor": {
        "path": ("p0_projection_floor", "CpMeanTrim", "relative_l2_floor"),
        "threshold": 0.80,
        "comparison": "less_than_or_equal",
    },
    "raw_wss_relative_l2_floor": {
        "path": (
            "p0_projection_floor",
            "wallShearStressMeanTrim",
            "raw_p0_relative_l2_floor",
        ),
        "threshold": 1.00,
        "comparison": "less_than_or_equal",
    },
    "normalized_area_weighted_fill_q95": {
        "path": (
            "fill_distance",
            "normalized_by_master_bbox_diagonal",
            "area_weighted_quantiles",
            "q95",
        ),
        "threshold": 1.00,
        "comparison": "less_than",
    },
}

ALGEBRA_ERROR_KEYS = (
    "measure_total_relative_error",
    "restriction_measure_replay_max_relative_error",
    "constant_max_abs_error",
    "representation_roundtrip_max_abs_error",
    "pythagorean_relative_error",
    "a_to_b_vector_integral_relative_error",
    "b_to_a_vector_integral_relative_error",
    "mass_adjoint_relative_error",
)
SOURCE_ARRAY_KEYS = (
    "points",
    "cells",
    "CpMeanTrim",
    "wallShearStressMeanTrim",
)
SOURCE_METADATA_KEYS = ("mesh", "cell_data")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class VerdictInputError(ValueError):
    """An input artifact violates the frozen H4 protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one JSON artifact without replacing prior evidence."""

    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_output_path(output: Path, inputs: list[Path]) -> None:
    output_resolved = output.resolve()
    if output_resolved in {path.resolve() for path in inputs}:
        raise VerdictInputError("H4 output path must differ from both input artifacts")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cyclic_indices_sha256(n_cells: int, start: int, k: int) -> str:
    """Reconstruct the little-endian int64 digest emitted on AGA."""

    digest = hashlib.sha256()
    chunk_size = 4096
    for offset in range(0, k, chunk_size):
        stop = min(offset + chunk_size, k)
        values = tuple((start + index) % n_cells for index in range(offset, stop))
        digest.update(struct.pack(f"<{len(values)}q", *values))
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerdictInputError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerdictInputError(f"{context} must be a JSON array")
    return value


def _at(value: Any, path: tuple[str, ...], context: str) -> Any:
    current = value
    for key in path:
        current = _mapping(current, context)
        if key not in current:
            raise VerdictInputError(f"{context} is missing {'.'.join(path)}")
        current = current[key]
    return current


def _finite_number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerdictInputError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VerdictInputError(f"{context} must be finite")
    if positive and result <= 0.0:
        raise VerdictInputError(f"{context} must be positive")
    return result


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise VerdictInputError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _metric_values(representation: dict[str, Any], context: str) -> dict[str, float]:
    return {
        name: _finite_number(
            _at(representation, specification["path"], context),
            f"{context}.{name}",
            positive=True,
        )
        for name, specification in METRICS.items()
    }


def _validate_representation_hashes(
    representation: dict[str, Any],
    context: str,
) -> dict[str, str]:
    return {
        key: _sha256(representation.get(key), f"{context}.{key}")
        for key in (
            "support_points_sha256_float32",
            "support_normals_sha256_float32",
            "master_assignment_sha256_int64",
        )
    }


def _validate_repair_events(
    value: Any,
    context: str,
    *,
    event_kind: str,
    n_support: int,
    n_master: int,
) -> dict[str, int]:
    events = _sequence(value, context)
    if event_kind == "lloyd":
        if len(events) != 2:
            raise VerdictInputError(f"{context} must contain exactly two Lloyd events")
    elif event_kind == "restriction":
        if len(events) > 2:
            raise VerdictInputError(f"{context} exceeds the two-repair limit")
    else:
        raise AssertionError(f"unknown repair event kind {event_kind!r}")

    total_empty = 0
    total_repaired = 0
    for index, raw_event in enumerate(events):
        event_context = f"{context}[{index}]"
        event = _mapping(raw_event, event_context)
        empty = event.get("empty_before_repair")
        if isinstance(empty, bool) or not isinstance(empty, int) or empty < 0:
            raise VerdictInputError(
                f"{event_context}.empty_before_repair must be a nonnegative integer"
            )
        support_indices = _sequence(
            event.get("repaired_support_indices"),
            f"{event_context}.repaired_support_indices",
        )
        master_cells = _sequence(
            event.get("repair_master_cells"),
            f"{event_context}.repair_master_cells",
        )
        if len(support_indices) != empty or len(master_cells) != empty:
            raise VerdictInputError(
                f"{event_context} does not account for every empty cluster"
            )
        for values, upper, label in (
            (support_indices, n_support, "repaired_support_indices"),
            (master_cells, n_master, "repair_master_cells"),
        ):
            if any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < upper
                for item in values
            ):
                raise VerdictInputError(
                    f"{event_context}.{label} must contain indices in [0, {upper})"
                )
            if len(set(values)) != len(values):
                raise VerdictInputError(
                    f"{event_context}.{label} must not contain duplicates"
                )

        shift = _mapping(event.get("shift"), f"{event_context}.shift")
        shift_values = {
            key: _finite_number(
                shift.get(key),
                f"{event_context}.shift.{key}",
            )
            for key in ("min", "mean", "q50", "q90", "q95", "q99", "max")
        }
        if any(value < 0.0 for value in shift_values.values()):
            raise VerdictInputError(f"{event_context}.shift must be nonnegative")
        ordered_shift = [
            shift_values[key] for key in ("min", "q50", "q90", "q95", "q99", "max")
        ]
        if ordered_shift != sorted(ordered_shift) or not (
            shift_values["min"] <= shift_values["mean"] <= shift_values["max"]
        ):
            raise VerdictInputError(
                f"{event_context}.shift distribution is not ordered"
            )

        if event_kind == "lloyd":
            if event.get("iteration") != index + 1:
                raise VerdictInputError(
                    f"{event_context}.iteration must equal {index + 1}"
                )
            if event.get("neighbor_backend") != EXPECTED_NEIGHBOR_BACKEND:
                raise VerdictInputError(
                    f"{event_context}.neighbor_backend is not frozen"
                )
            _mapping(
                event.get("represented_measure"),
                f"{event_context}.represented_measure",
            )
        else:
            if empty == 0:
                raise VerdictInputError(
                    f"{event_context} records a restriction repair with no empty cell"
                )
            if event.get("restriction_attempt") != index + 1:
                raise VerdictInputError(
                    f"{event_context}.restriction_attempt must equal {index + 1}"
                )
            if event.get("repair_scope") != "empty_supports_only":
                raise VerdictInputError(
                    f"{event_context}.repair_scope must be 'empty_supports_only'"
                )
        total_empty += empty
        total_repaired += len(support_indices)
    return {
        "events": len(events),
        "empty_before_repair_total": total_empty,
        "repaired_cluster_total": total_repaired,
    }


def _validate_source(
    case: dict[str, Any],
    case_id: str,
    context: str,
) -> dict[str, Any]:
    source = _mapping(case.get("source"), f"{context}.source")
    if source.get("kind") != EXPECTED_SOURCE_KIND:
        raise VerdictInputError(
            f"{context}.source.kind must be {EXPECTED_SOURCE_KIND!r}"
        )
    n_cells = source.get("n_cells")
    if (
        isinstance(n_cells, bool)
        or not isinstance(n_cells, int)
        or n_cells <= EXPECTED_K
    ):
        raise VerdictInputError(
            f"{context}.source.n_cells must exceed the representation size"
        )
    full_master = _mapping(case.get("full_master"), f"{context}.full_master")
    if full_master.get("n_cells") != n_cells:
        raise VerdictInputError(
            f"{context} source and full-master cell counts disagree"
        )
    if full_master.get("degenerate_triangle_count") != 0:
        raise VerdictInputError(f"{context} full master contains degenerate triangles")

    arrays = _mapping(source.get("arrays"), f"{context}.source.arrays")
    array_hashes = {}
    for key in SOURCE_ARRAY_KEYS:
        entry = _mapping(arrays.get(key), f"{context}.source.arrays.{key}")
        array_hashes[key] = _sha256(
            entry.get("sha256"),
            f"{context}.source.arrays.{key}.sha256",
        )

    metadata = _mapping(source.get("metadata"), f"{context}.source.metadata")
    metadata_hashes = {}
    for key in SOURCE_METADATA_KEYS:
        entry = _mapping(metadata.get(key), f"{context}.source.metadata.{key}")
        metadata_hashes[key] = _sha256(
            entry.get("sha256"),
            f"{context}.source.metadata.{key}.sha256",
        )
    observed = {
        "kind": source["kind"],
        "n_cells": n_cells,
        "array_sha256": array_hashes,
        "metadata_sha256": metadata_hashes,
    }
    expected = {
        "kind": EXPECTED_SOURCE_KIND,
        **FROZEN_SOURCE_IDENTITIES[case_id],
    }
    if observed != expected:
        raise VerdictInputError(
            f"{context} source identity differs from the frozen 10k audit"
        )
    return observed


def _validate_algebra(value: Any, context: str) -> dict[str, Any]:
    algebra = _mapping(value, context)
    tolerance = _finite_number(algebra.get("tolerance"), f"{context}.tolerance")
    if tolerance != EXPECTED_ALGEBRA_TOLERANCE:
        raise VerdictInputError(
            f"{context}.tolerance must be {EXPECTED_ALGEBRA_TOLERANCE:.1e}"
        )
    if algebra.get("passed") is not True:
        raise VerdictInputError(f"{context}.passed must be true")
    errors = {
        key: _finite_number(algebra.get(key), f"{context}.{key}")
        for key in ALGEBRA_ERROR_KEYS
    }
    invalid_errors = {
        key: error for key, error in errors.items() if error < 0.0 or error > tolerance
    }
    if invalid_errors:
        raise VerdictInputError(
            f"{context} algebra errors must all lie in [0, {tolerance:.1e}]"
        )
    observed_max = _finite_number(
        algebra.get("max_load_bearing_error"),
        f"{context}.max_load_bearing_error",
    )
    recomputed_max = max(errors.values())
    if observed_max != recomputed_max:
        raise VerdictInputError(
            f"{context}.max_load_bearing_error does not match its components"
        )
    if observed_max > tolerance:
        raise VerdictInputError(
            f"{context} exceeds its {tolerance:.1e} algebra tolerance"
        )
    return {
        "passed": True,
        "tolerance": tolerance,
        "max_load_bearing_error": observed_max,
        "operator_sha256": _sha256(
            algebra.get("operator_sha256_streaming_assignment_a_b_area_float64"),
            f"{context}.operator_sha256_streaming_assignment_a_b_area_float64",
        ),
    }


def _validate_case(
    artifact_path: Path,
    artifact: dict[str, Any],
    artifact_sha256: str,
) -> tuple[str, dict[str, Any]]:
    context = str(artifact_path)
    if artifact.get("schema_version") != 1:
        raise VerdictInputError(f"{context} must use production audit schema 1")
    if artifact.get("status") != EXPECTED_AUDIT_STATUS:
        raise VerdictInputError(f"{context}.status must be {EXPECTED_AUDIT_STATUS!r}")

    design = _mapping(artifact.get("design"), f"{context}.design")
    if design.get("requested_k") != EXPECTED_K:
        raise VerdictInputError(f"{context}.design.requested_k must be {EXPECTED_K}")
    if design.get("synthetic_k") is not None:
        raise VerdictInputError(f"{context} must be a production, not synthetic, audit")
    if design.get("hash_inputs") is not True:
        raise VerdictInputError(f"{context} must hash all production inputs")
    if design.get("candidate_multiplier") != 16:
        raise VerdictInputError(f"{context} changed the frozen cover candidate count")
    if design.get("lloyd_iterations") != 2:
        raise VerdictInputError(f"{context} changed the frozen Lloyd iteration count")
    expected_design = {
        "normal_aware_assignment": {
            "squared_cost": "||x_i-s_j||^2 + lambda^2 ||n_i-m_j||^2",
            "lambda": "sqrt(per_case_master_area / k)",
        },
        "geometry_chunk_cells": 250_000,
        "point_chunk": 1_000_000,
        "repair_pool_size": 2_048,
        "workers": 16,
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise VerdictInputError(f"{context}.design.{key} is not frozen")

    cases = _sequence(artifact.get("cases"), f"{context}.cases")
    if len(cases) != 1:
        raise VerdictInputError(f"{context} must contain exactly one case")
    case = _mapping(cases[0], f"{context}.cases[0]")
    case_id = case.get("case_id")
    if case_id not in EXPECTED_CASES:
        raise VerdictInputError(f"{context} has unexpected case {case_id!r}")
    case_context = f"{context}:{case_id}"
    if case.get("effective_k") != EXPECTED_K:
        raise VerdictInputError(f"{case_context}.effective_k must be {EXPECTED_K}")

    expected_starts = FROZEN_STARTS[case_id]
    expected_design_replicates = {
        case_id: [{"start": start, "seed": FROZEN_SEED} for start in expected_starts]
    }
    if design.get("explicit_cyclic_replicates") != expected_design_replicates:
        raise VerdictInputError(
            f"{case_context} does not declare the four frozen cyclic replicates"
        )

    source_provenance = _validate_source(case, case_id, case_context)
    cover_construction = _mapping(
        case.get("cover_construction"),
        f"{case_context}.cover_construction",
    )
    if cover_construction.get("candidate_multiplier") != 16:
        raise VerdictInputError(
            f"{case_context}.cover_construction changed candidate_multiplier"
        )
    if cover_construction.get("lloyd_iterations") != 2:
        raise VerdictInputError(
            f"{case_context}.cover_construction changed lloyd_iterations"
        )
    lloyd_history = _sequence(
        cover_construction.get("lloyd_history"),
        f"{case_context}.cover_construction.lloyd_history",
    )
    if len(lloyd_history) != 2:
        raise VerdictInputError(
            f"{case_context}.cover_construction must contain two Lloyd updates"
        )
    lloyd_repairs = _validate_repair_events(
        lloyd_history,
        f"{case_context}.cover_construction.lloyd_history",
        event_kind="lloyd",
        n_support=EXPECTED_K,
        n_master=source_provenance["n_cells"],
    )
    restriction_repairs = _validate_repair_events(
        cover_construction.get("restriction_empty_cell_repairs"),
        f"{case_context}.cover_construction.restriction_empty_cell_repairs",
        event_kind="restriction",
        n_support=EXPECTED_K,
        n_master=source_provenance["n_cells"],
    )
    case_execution = _mapping(
        case.get("execution"),
        f"{case_context}.execution",
    )
    if (
        case_execution.get("cover_restriction_neighbor_backend")
        != EXPECTED_NEIGHBOR_BACKEND
    ):
        raise VerdictInputError(
            f"{case_context} did not use the frozen cover neighbor backend"
        )

    replicates = _sequence(
        case.get("cyclic_replicates"),
        f"{case_context}.cyclic_replicates",
    )
    if len(replicates) != 4:
        raise VerdictInputError(f"{case_context} must contain exactly four replicates")

    cover_baseline: dict[str, Any] | None = None
    rows = []
    cyclic_values = {name: [] for name in METRICS}
    cyclic_index_hashes = set()
    for ordinal, (raw_replicate, expected_start) in enumerate(
        zip(replicates, expected_starts)
    ):
        replicate_context = f"{case_context}.cyclic_replicates[{ordinal}]"
        replicate = _mapping(raw_replicate, replicate_context)
        if replicate.get("ordinal") != ordinal:
            raise VerdictInputError(f"{replicate_context}.ordinal is not frozen")
        if replicate.get("frozen_cyclic_start") != expected_start:
            raise VerdictInputError(f"{replicate_context} has the wrong frozen start")
        if replicate.get("frozen_seed") != FROZEN_SEED:
            raise VerdictInputError(f"{replicate_context} has the wrong frozen seed")

        representations = _mapping(
            replicate.get("representations"),
            f"{replicate_context}.representations",
        )
        cyclic = _mapping(
            representations.get("cyclic_sparse"),
            f"{replicate_context}.representations.cyclic_sparse",
        )
        cover = _mapping(
            representations.get("normal_aware_centroidal_cover"),
            f"{replicate_context}.representations.normal_aware_centroidal_cover",
        )
        if cyclic.get("k") != EXPECTED_K or cover.get("k") != EXPECTED_K:
            raise VerdictInputError(f"{replicate_context} changed representation k")
        if cyclic.get("empty_representation_cell_count") != 0:
            raise VerdictInputError(f"{replicate_context} cyclic map has empty cells")
        if cover.get("empty_representation_cell_count") != 0:
            raise VerdictInputError(f"{replicate_context} cover map has empty cells")

        cyclic_definition = _mapping(
            cyclic.get("definition"),
            f"{replicate_context}.representations.cyclic_sparse.definition",
        )
        if cyclic_definition.get("start") != expected_start:
            raise VerdictInputError(
                f"{replicate_context} cyclic definition has the wrong start"
            )
        if cyclic_definition.get("frozen_seed") != FROZEN_SEED:
            raise VerdictInputError(
                f"{replicate_context} cyclic definition has the wrong seed"
            )
        if cyclic_definition.get("replicate_ordinal") != ordinal:
            raise VerdictInputError(
                f"{replicate_context} cyclic definition has the wrong ordinal"
            )
        cyclic_index_hash = _sha256(
            cyclic_definition.get("indices_sha256_int64"),
            f"{replicate_context}.cyclic.indices_sha256_int64",
        )
        expected_cyclic_index_hash = _cyclic_indices_sha256(
            source_provenance["n_cells"],
            expected_start,
            EXPECTED_K,
        )
        if cyclic_index_hash != expected_cyclic_index_hash:
            raise VerdictInputError(
                f"{replicate_context} cyclic index hash does not match start/k"
            )
        expected_first_indices = [
            (expected_start + index) % source_provenance["n_cells"]
            for index in range(10)
        ]
        if cyclic_definition.get("first_indices") != expected_first_indices:
            raise VerdictInputError(
                f"{replicate_context} cyclic first indices do not match start/k"
            )
        expected_wraps = expected_start + EXPECTED_K > source_provenance["n_cells"]
        if cyclic_definition.get("wraps") is not expected_wraps:
            raise VerdictInputError(
                f"{replicate_context} cyclic wrap flag does not match start/k"
            )
        cyclic_index_hashes.add(cyclic_index_hash)

        if cover.get("definition") != cover_construction:
            raise VerdictInputError(
                f"{replicate_context} cover definition differs from case construction"
            )
        if cover_baseline is None:
            cover_baseline = cover
        elif cover != cover_baseline:
            raise VerdictInputError(
                f"{case_context} deterministic cover changed across cyclic starts"
            )

        algebra = _validate_algebra(
            replicate.get("common_master_algebra"),
            f"{replicate_context}.common_master_algebra",
        )
        replicate_execution = _mapping(
            replicate.get("execution"),
            f"{replicate_context}.execution",
        )
        if replicate_execution.get(
            "cyclic_restriction_neighbor_backend"
        ) != EXPECTED_NEIGHBOR_BACKEND or replicate_execution.get(
            "final_neighbor_backends"
        ) != [EXPECTED_NEIGHBOR_BACKEND, EXPECTED_NEIGHBOR_BACKEND]:
            raise VerdictInputError(
                f"{replicate_context} did not use the frozen neighbor backend"
            )
        cyclic_hashes = _validate_representation_hashes(
            cyclic,
            f"{replicate_context}.cyclic",
        )
        cover_hashes = _validate_representation_hashes(
            cover,
            f"{replicate_context}.cover",
        )
        cyclic_metrics = _metric_values(cyclic, f"{replicate_context}.cyclic")
        cover_metrics = _metric_values(cover, f"{replicate_context}.cover")

        comparison = _mapping(
            replicate.get("comparison"),
            f"{replicate_context}.comparison",
        )
        if (
            comparison.get("ratio_definition")
            != "normal_aware_centroidal_cover / cyclic_sparse"
        ):
            raise VerdictInputError(
                f"{replicate_context} has an unexpected comparison definition"
            )
        reported_ratio_keys = {
            "pressure_relative_l2_floor": "CpMeanTrim_relative_l2_floor_ratio",
            "raw_wss_relative_l2_floor": (
                "wallShearStressMeanTrim_relative_l2_floor_ratio"
            ),
            "normalized_area_weighted_fill_q95": (
                "normalized_area_weighted_fill_q95_ratio"
            ),
        }
        diagnostic_ratios = {}
        for name, reported_key in reported_ratio_keys.items():
            recomputed = cover_metrics[name] / cyclic_metrics[name]
            reported = _finite_number(
                comparison.get(reported_key),
                f"{replicate_context}.comparison.{reported_key}",
            )
            if not math.isclose(recomputed, reported, rel_tol=1.0e-14, abs_tol=0.0):
                raise VerdictInputError(
                    f"{replicate_context}.{reported_key} disagrees with "
                    "the absolute metrics"
                )
            diagnostic_ratios[name] = recomputed
            cyclic_values[name].append(cyclic_metrics[name])

        rows.append(
            {
                "ordinal": ordinal,
                "frozen_cyclic_start": expected_start,
                "frozen_seed": FROZEN_SEED,
                "absolute_metrics": {
                    "cyclic_sparse": cyclic_metrics,
                    "normal_aware_centroidal_cover": cover_metrics,
                },
                "cover_over_cyclic_ratio_diagnostic_only": diagnostic_ratios,
                "integrity": {
                    "algebra": algebra,
                    "cyclic_indices_sha256_int64": cyclic_index_hash,
                    "cyclic_representation_sha256": cyclic_hashes,
                    "cover_representation_sha256": cover_hashes,
                    "cyclic_empty_representation_cell_count": 0,
                    "cover_empty_representation_cell_count": 0,
                },
            }
        )

    if len(cyclic_index_hashes) != 4:
        raise VerdictInputError(f"{case_context} cyclic index blocks are not distinct")
    if cover_baseline is None:
        raise AssertionError("four validated replicates must define a cover")

    cover_metrics = _metric_values(cover_baseline, f"{case_context}.cover")
    median_cyclic = {
        name: statistics.median(values) for name, values in cyclic_values.items()
    }
    ratios = {name: cover_metrics[name] / median_cyclic[name] for name in METRICS}
    gates = {}
    for name, specification in METRICS.items():
        ratio = ratios[name]
        threshold = specification["threshold"]
        comparison = specification["comparison"]
        if comparison == "less_than_or_equal":
            passed = ratio <= threshold
        elif comparison == "less_than":
            passed = ratio < threshold
        else:
            raise AssertionError(f"unknown frozen comparison {comparison!r}")
        gates[name] = {
            "ratio": ratio,
            "threshold": threshold,
            "comparison": comparison,
            "passed": passed,
        }
    case_passed = all(gate["passed"] for gate in gates.values())

    audit_provenance = _mapping(
        artifact.get("provenance"),
        f"{context}.provenance",
    )
    audit_script = _mapping(
        audit_provenance.get("script"),
        f"{context}.provenance.script",
    )
    audit_script_sha256 = _sha256(
        audit_script.get("sha256"),
        f"{context}.provenance.script.sha256",
    )
    if audit_script_sha256 != EXPECTED_AUDIT_SCRIPT_SHA256:
        raise VerdictInputError(
            f"{context} was not produced by the frozen production audit script"
        )
    versions = _mapping(
        audit_provenance.get("versions"),
        f"{context}.provenance.versions",
    )
    observed_versions = {
        "python": audit_provenance.get("python"),
        "numpy": versions.get("numpy"),
        "scipy": versions.get("scipy"),
    }
    if observed_versions != EXPECTED_PRODUCTION_VERSIONS:
        raise VerdictInputError(
            f"{context} did not use the frozen production package versions"
        )
    if "aarch64" not in str(audit_provenance.get("platform")):
        raise VerdictInputError(f"{context} was not produced on the AGA ARM platform")
    return case_id, {
        "case_id": case_id,
        "input_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": artifact_sha256,
            "generated_at_utc": artifact.get("generated_at_utc"),
            "audit_script_sha256": audit_script_sha256,
        },
        "source_provenance": source_provenance,
        "effective_k": EXPECTED_K,
        "frozen_seed": FROZEN_SEED,
        "frozen_cyclic_starts": list(expected_starts),
        "start_level_rows": rows,
        "aggregation": {
            "definition": (
                "one invariant cover absolute metric divided by the median "
                "absolute cyclic metric; start-level ratios are not aggregated"
            ),
            "normal_aware_centroidal_cover_absolute": cover_metrics,
            "cyclic_sparse_absolute_by_metric": cyclic_values,
            "cyclic_sparse_median_absolute": median_cyclic,
            "cover_over_median_cyclic": ratios,
        },
        "integrity": {
            "cover_invariant_across_replicates": True,
            "cover_canonical_json_sha256": _canonical_sha256(cover_baseline),
            "four_distinct_cyclic_index_blocks": True,
            "lloyd_repair_accounting": lloyd_repairs,
            "restriction_repair_accounting": restriction_repairs,
            "final_empty_representation_cell_count": {
                "cyclic_sparse": 0,
                "normal_aware_centroidal_cover": 0,
            },
        },
        "gates": gates,
        "all_h4_gates_pass": case_passed,
    }


def build_verdict(
    artifact_paths: list[Path],
) -> dict[str, Any]:
    """Load, validate, and aggregate exactly the two deciding artifacts."""

    if len(artifact_paths) != 2:
        raise VerdictInputError("H4 requires exactly two production artifacts")
    resolved = [path.resolve() for path in artifact_paths]
    if len(set(resolved)) != 2:
        raise VerdictInputError("H4 input artifact paths must be distinct")

    cases_by_id = {}
    for path in artifact_paths:
        try:
            artifact_bytes = path.read_bytes()
            artifact = json.loads(artifact_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerdictInputError(f"cannot read {path}: {error}") from error
        case_id, case_result = _validate_case(
            path,
            _mapping(artifact, str(path)),
            hashlib.sha256(artifact_bytes).hexdigest(),
        )
        if case_id in cases_by_id:
            raise VerdictInputError(f"duplicate H4 case {case_id!r}")
        cases_by_id[case_id] = case_result
    if set(cases_by_id) != set(EXPECTED_CASES):
        raise VerdictInputError(f"H4 cases must be exactly {', '.join(EXPECTED_CASES)}")

    cases = [cases_by_id[case_id] for case_id in EXPECTED_CASES]
    audit_script_hashes = {
        case["input_artifact"]["audit_script_sha256"] for case in cases
    }
    if len(audit_script_hashes) != 1:
        raise VerdictInputError(
            "the two deciding artifacts used different production audit scripts"
        )
    passed = all(case["all_h4_gates_pass"] for case in cases)
    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED_H4_COMMON_MASTER_GATE"
        if passed
        else "FAILED_H4_COMMON_MASTER_GATE",
        "hypothesis": {
            "id": "H4",
            "scope": (
                "Whether deterministic normal-aware coverage reduces the "
                "40k induced-reconstruction substrate floor enough to remain "
                "causally live on both frozen DrivAerML cases."
            ),
        },
        "decision_rule": {
            "frozen_before_deciding_output_on": PROTOCOL_DATE,
            "protocol_history": (
                "The 20%/no-worse/lower-fill thresholds were pre-registered "
                "before production execution. The run_1/run_118 cases and "
                "median-of-four reduction were amended after the explicitly "
                "non-deciding 10k engineering screen and before any 40k output."
            ),
            "cases": list(EXPECTED_CASES),
            "k": EXPECTED_K,
            "cyclic_replicates_per_case": 4,
            "frozen_seed": FROZEN_SEED,
            "aggregation": (
                "cover absolute metric / median of four absolute cyclic metrics"
            ),
            "thresholds": {
                name: {
                    "value": specification["threshold"],
                    "comparison": specification["comparison"],
                }
                for name, specification in METRICS.items()
            },
            "joint_gate": "every metric must pass on both cases",
        },
        "cases": cases,
        "h4_passed": passed,
        "pre_registered_next_action": (
            "advance_to_frozen_checkpoint_compatibility_screen"
            if passed
            else "deprioritize_support_coverage_and_do_not_start_covering_training"
        ),
        "limitations": [
            (
                "The two cases are diagnostic replicates, not a vehicle-population "
                "estimate."
            ),
            (
                "Two frozen run_1 cyclic blocks overlap by 4,249 of 40,000 "
                "triangles; they were retained to avoid a post-hoc redraw."
            ),
            (
                "A passing reconstruction-floor gate motivates a compatibility "
                "screen but is not a causal trained-model result."
            ),
        ],
        "integrity": {
            "all_inputs_hashed": True,
            "production_audit_script_sha256": audit_script_hashes.pop(),
            "all_algebra_gates_passed_at_tolerance": EXPECTED_ALGEBRA_TOLERANCE,
            "all_final_representation_maps_nonempty": True,
        },
        "provenance": {
            "script": {
                "path": str(script_path),
                "sha256": _sha256_file(script_path),
            },
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _parse_args() -> argparse.Namespace:
    example_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs=2,
        type=Path,
        metavar=("RUN_1_JSON", "RUN_118_JSON"),
        help="The two per-case 40k production common-master audit artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(example_root / "results" / f"phase1_h4_verdict_{PROTOCOL_DATE}.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_output_path(args.output, args.artifacts)
    result = build_verdict(args.artifacts)
    _write_json_once(args.output, result)
    print(f"{result['status']} h4_passed={result['h4_passed']} artifact={args.output}")


if __name__ == "__main__":
    main()
