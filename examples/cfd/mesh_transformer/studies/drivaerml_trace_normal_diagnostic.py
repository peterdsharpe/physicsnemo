# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose raw-versus-pipeline triangle-normal reconstruction for H-QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import drivaerml_trace_fixed_query_audit as hqc
import numpy as np
import torch


def _difference_summary(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    delta = np.linalg.norm(first64 - second64, axis=1)
    cosine = np.einsum("ij,ij->i", first64, second64)
    return {
        "delta_quantiles": {
            str(q): float(np.quantile(delta, q))
            for q in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "max_one_minus_cosine": float(np.max(1.0 - cosine)),
        "negative_cosine_count": int(np.count_nonzero(cosine < 0.0)),
        "over_2e-6_count": int(np.count_nonzero(delta > 2.0e-6)),
        "row_count": int(len(delta)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--case-id", default="run_118")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = hqc._load_runtime(
        repo_root=args.repo_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        dataset_config_path=args.dataset_config.resolve(),
        resolved_config_path=args.resolved_config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
    )
    hqc._validate_reader(runtime)
    specs = [spec for spec in hqc.CASE_SPECS if spec.case_id == args.case_id]
    if len(specs) != 1:
        raise ValueError(f"Expected one frozen case named {args.case_id}")
    spec = specs[0]
    archived = hqc._load_archived_10k(
        hqc._archived_prediction_path(
            args.historical_predictions.resolve(),
            spec,
        )
    )
    raw_mesh = runtime.dataset.reader._load_sample(spec.reader_index)
    selections = {
        k: hqc._cyclic_indices(spec.n_master_cells, spec.historical_start, k)
        for k in hqc.RESOLUTIONS
    }
    subset_10k = hqc._compact_explicit_cell_subset(
        raw_mesh,
        selections[hqc.BASELINE_K],
        runtime.mesh_type,
    )
    fixed_center = hqc._pipeline_center_on_device(subset_10k, runtime.device)

    rows: dict[str, object] = {}
    primary_q_reference: np.ndarray | None = None
    fixed_q_reference: np.ndarray | None = None
    for k in hqc.RESOLUTIONS:
        subset = hqc._compact_explicit_cell_subset(
            raw_mesh,
            selections[k],
            runtime.mesh_type,
        )
        _, native_normals, _ = hqc._native_geometry(subset)
        primary_domain, _ = hqc._apply_pipeline(
            runtime,
            subset,
            fixed_center=None,
        )
        fixed_domain, _ = hqc._apply_pipeline(
            runtime,
            subset,
            fixed_center=fixed_center,
        )
        primary_normals = (
            primary_domain.boundaries["vehicle"]
            .cell_data["normals"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        fixed_normals = (
            fixed_domain.boundaries["vehicle"]
            .cell_data["normals"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        primary_q = primary_normals[: hqc.FIXED_QUERY_K]
        fixed_q = fixed_normals[: hqc.FIXED_QUERY_K]
        if primary_q_reference is None:
            primary_q_reference = np.array(primary_q, copy=True)
            fixed_q_reference = np.array(fixed_q, copy=True)
        row = {
            "native_vs_primary": _difference_summary(
                native_normals,
                primary_normals,
            ),
            "native_vs_fixed": _difference_summary(native_normals, fixed_normals),
            "primary_vs_fixed": _difference_summary(
                primary_normals,
                fixed_normals,
            ),
            "q_primary_vs_k2500": _difference_summary(
                primary_q_reference,
                primary_q,
            ),
            "q_fixed_vs_k2500": _difference_summary(
                fixed_q_reference,
                fixed_q,
            ),
        }
        if k == hqc.BASELINE_K:
            row["archived_vs_primary"] = _difference_summary(
                archived["boundary_normals"],
                primary_normals,
            )
        rows[str(k)] = row
        del subset, primary_domain, fixed_domain
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "case_id": spec.case_id,
                "diagnostic_only": True,
                "resolutions": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
