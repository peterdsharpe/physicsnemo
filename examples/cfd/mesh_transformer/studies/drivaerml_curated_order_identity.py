# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check whether raw-derived DrivAerML arrays equal curated ``.pdmsh`` arrays.

The raw surface is transformed exactly as in
``drivaerml_sampler_prototype.py``: discard unrelated arrays, then call
``PolyData.triangulate(pass_verts=False, pass_lines=False)``.  Curated arrays
remain remote and read-only; a standard-library Python helper hashes their
memmap files over SSH.

Example
-------
Run from the repository root::

    uv run --no-sync python \
      examples/cfd/mesh_transformer/studies/drivaerml_curated_order_identity.py \
      --remote-host aga1 \
      --case run_1 /data/run_1/boundary_1.vtp \
        /scratch/.../run_1/domain_run_1.pdmsh \
      --case run_2 /data/run_2/boundary_2.vtp \
        /scratch/.../run_2/domain_run_2.pdmsh \
      --sampler-result examples/cfd/mesh_transformer/results/\
drivaerml_raw_vtp_sampler_prototype_2026-07-27.json \
      --output examples/cfd/mesh_transformer/results/\
drivaerml_curated_order_identity_2026-07-27.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ARRAY_FILES = {
    "points": "points.memmap",
    "cells": "cells.memmap",
    "CpMeanTrim": "cell_data/CpMeanTrim.memmap",
    "wallShearStressMeanTrim": "cell_data/wallShearStressMeanTrim.memmap",
}
TORCH_TO_NUMPY_DTYPE = {
    "torch.float32": np.dtype("<f4"),
    "torch.int64": np.dtype("<i8"),
}
REMOTE_HASH_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

pdmsh = Path(sys.argv[1])
root = pdmsh / "_tensordict" / "boundaries" / "vehicle" / "_tensordict"
with (root / "meta.json").open() as stream:
    mesh_meta = json.load(stream)
with (root / "cell_data" / "meta.json").open() as stream:
    cell_meta = json.load(stream)

specs = {
    "points": ("points.memmap", mesh_meta["points"]),
    "cells": ("cells.memmap", mesh_meta["cells"]),
    "CpMeanTrim": (
        "cell_data/CpMeanTrim.memmap",
        cell_meta["CpMeanTrim"],
    ),
    "wallShearStressMeanTrim": (
        "cell_data/wallShearStressMeanTrim.memmap",
        cell_meta["wallShearStressMeanTrim"],
    ),
}
result = {}
for name, (relative_path, metadata) in specs.items():
    path = root / relative_path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    result[name] = {
        "path": str(path),
        "shape": metadata["shape"],
        "dtype": metadata["dtype"],
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
print(json.dumps(result, sort_keys=True))
"""


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dtype_str": array.dtype.str,
        "bytes": int(array.size * array.dtype.itemsize),
    }


def _array_record(array: np.ndarray) -> dict[str, Any]:
    return {
        **_array_metadata(array),
        "sha256": _sha256_array(array),
    }


def _hash_indexed_array(
    array: np.ndarray,
    indices: np.ndarray,
    *,
    index_remap: np.ndarray | None = None,
    chunk_rows: int = 1_000_000,
) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(indices), chunk_rows):
        selected = array[indices[start : start + chunk_rows]]
        if index_remap is not None:
            selected = index_remap[selected]
        canonical = np.ascontiguousarray(selected)
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _hash_fan_connectivity(
    connectivity: np.ndarray,
    offsets: np.ndarray,
    triangle_permutation: np.ndarray,
    inverse_point_permutation: np.ndarray,
    chunk_rows: int = 1_000_000,
) -> str:
    """Hash curator-style fan triangles in permuted/remapped storage order."""
    triangles_per_polygon = np.diff(offsets).astype(np.int64) - 2
    cumulative = np.empty(len(triangles_per_polygon) + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(triangles_per_polygon, out=cumulative[1:])
    if cumulative[-1] != len(triangle_permutation):
        raise ValueError(
            "Fan triangle count disagrees with the triangulated cell count: "
            f"{cumulative[-1]} != {len(triangle_permutation)}"
        )

    digest = hashlib.sha256()
    for start in range(0, len(triangle_permutation), chunk_rows):
        source_triangle = triangle_permutation[start : start + chunk_rows]
        polygon = np.searchsorted(cumulative, source_triangle, side="right") - 1
        local_triangle = source_triangle - cumulative[polygon]
        polygon_start = offsets[polygon].astype(np.int64, copy=False)
        triangles = np.empty((len(source_triangle), 3), dtype=np.int64)
        triangles[:, 0] = connectivity[polygon_start]
        triangles[:, 1] = connectivity[polygon_start + local_triangle + 1]
        triangles[:, 2] = connectivity[polygon_start + local_triangle + 2]
        triangles = inverse_point_permutation[triangles]
        digest.update(memoryview(np.ascontiguousarray(triangles)).cast("B"))
    return digest.hexdigest()


def _permutation_summary(permutation: np.ndarray) -> dict[str, Any]:
    count = len(permutation)
    fixed_count = 0
    displacement_sum = 0.0
    for start in range(0, count, 1_000_000):
        stop = min(start + 1_000_000, count)
        expected = np.arange(start, stop, dtype=np.int64)
        chunk = permutation[start:stop]
        fixed_count += int(np.count_nonzero(chunk == expected))
        displacement_sum += float(
            np.abs(chunk.astype(np.float64) - expected).sum(dtype=np.float64)
        )
    mismatches = np.flatnonzero(
        permutation[: min(count, 1_000_000)]
        != np.arange(min(count, 1_000_000), dtype=np.int64)
    )
    return {
        "definition": "curated_index -> raw_triangulated_index",
        "sha256_int64": _sha256_array(permutation),
        "first_16_raw_indices_in_curated_order": [
            int(value) for value in permutation[:16]
        ],
        "first_nonidentity_curated_index": (
            int(mismatches[0]) if len(mismatches) else None
        ),
        "fixed_index_count": fixed_count,
        "fixed_index_fraction": fixed_count / count,
        "mean_absolute_index_displacement": displacement_sum / count,
        "mean_absolute_index_displacement_over_count": (
            displacement_sum / count / count
        ),
    }


def _hash_raw_triangulation(
    path: Path,
    point_permutation_seed: int,
    cell_permutation_seed: int,
) -> dict[str, Any]:
    source_stat = path.stat()
    source_hash = _sha256_file(path)
    surface = pv.read(path)
    if not isinstance(surface, pv.PolyData):
        raise TypeError(f"Expected PolyData in {path}, got {type(surface).__name__}")

    required_fields = ("CpMeanTrim", "wallShearStressMeanTrim")
    missing = [name for name in required_fields if name not in surface.cell_data]
    if missing:
        raise KeyError(f"{path} is missing required cell fields: {missing}")

    raw_n_points = surface.n_points
    raw_n_cells = surface.n_cells
    raw_is_all_triangles = surface.is_all_triangles
    for name in list(surface.cell_data.keys()):
        if name not in required_fields:
            del surface.cell_data[name]
    surface.point_data.clear()
    surface.field_data.clear()

    point_generator = torch.Generator().manual_seed(point_permutation_seed)
    point_permutation = torch.randperm(
        surface.n_points, generator=point_generator
    ).numpy()
    inverse_point_permutation = np.empty_like(point_permutation)
    inverse_point_permutation[point_permutation] = np.arange(
        len(point_permutation), dtype=np.int64
    )

    polygons = surface.GetPolys()
    polygon_offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    polygon_connectivity = vtk_to_numpy(polygons.GetConnectivityArray())
    fan_triangle_count = int(np.sum(np.diff(polygon_offsets).astype(np.int64) - 2))
    cell_generator = torch.Generator().manual_seed(cell_permutation_seed)
    cell_permutation = torch.randperm(
        fan_triangle_count, generator=cell_generator
    ).numpy()
    fan_connectivity_hash = _hash_fan_connectivity(
        polygon_connectivity,
        polygon_offsets,
        cell_permutation,
        inverse_point_permutation,
    )

    if raw_is_all_triangles:
        triangles = surface
    else:
        triangles = surface.triangulate(
            pass_verts=False,
            pass_lines=False,
            inplace=False,
            progress_bar=False,
        )
        del surface
        gc.collect()

    arrays = {
        "points": np.asarray(triangles.points),
        "cells": np.asarray(triangles.regular_faces),
        "CpMeanTrim": np.asarray(triangles.cell_data["CpMeanTrim"], dtype=np.float32),
        "wallShearStressMeanTrim": np.asarray(
            triangles.cell_data["wallShearStressMeanTrim"], dtype=np.float32
        ),
    }
    if fan_triangle_count != len(arrays["cells"]):
        raise ValueError(
            "Curator fan and PyVista triangle counts differ: "
            f"{fan_triangle_count} != {len(arrays['cells'])}"
        )
    reconstructed_arrays = {
        "points": {
            **_array_metadata(arrays["points"]),
            "sha256": _hash_indexed_array(arrays["points"], point_permutation),
        },
        "cells": {
            **_array_metadata(arrays["cells"]),
            "sha256": fan_connectivity_hash,
        },
        "CpMeanTrim": {
            **_array_metadata(arrays["CpMeanTrim"]),
            "sha256": _hash_indexed_array(arrays["CpMeanTrim"], cell_permutation),
        },
        "wallShearStressMeanTrim": {
            **_array_metadata(arrays["wallShearStressMeanTrim"]),
            "sha256": _hash_indexed_array(
                arrays["wallShearStressMeanTrim"], cell_permutation
            ),
        },
    }
    result = {
        "source": {
            "path": str(path.resolve()),
            "bytes": source_stat.st_size,
            "sha256": source_hash,
            "raw_n_points": raw_n_points,
            "raw_n_cells": raw_n_cells,
            "raw_is_all_triangles": raw_is_all_triangles,
        },
        "transformation": (
            "identity"
            if raw_is_all_triangles
            else "pyvista.PolyData.triangulate("
            "pass_verts=False, pass_lines=False, inplace=False, "
            "progress_bar=False)"
        ),
        "arrays": {name: _array_record(array) for name, array in arrays.items()},
        "curator_permutation_reconstruction": {
            "point_permutation_seed": point_permutation_seed,
            "cell_permutation_seed": cell_permutation_seed,
            "algorithm": (
                "Independent torch.Generators produce the point and cell "
                "permutations. Points are indexed by the first permutation. "
                "Raw polygons are fan-tessellated from their first vertex, "
                "matching PhysicsNeMo-Curator's Rust-source conversion. Cell "
                "point indices are remapped by the inverse point permutation "
                "before cell rows and cell_data are indexed by the second "
                "permutation."
            ),
            "point_mapping": _permutation_summary(point_permutation),
            "cell_mapping": _permutation_summary(cell_permutation),
            "arrays": reconstructed_arrays,
        },
    }
    del (
        arrays,
        cell_permutation,
        fan_connectivity_hash,
        inverse_point_permutation,
        polygon_connectivity,
        polygon_offsets,
        point_permutation,
        reconstructed_arrays,
        triangles,
    )
    gc.collect()
    return result


def _hash_remote_curated(
    remote_host: str, curated_path: Path
) -> tuple[dict[str, Any], list[str]]:
    ssh = shutil.which("ssh")
    if ssh is None:
        raise RuntimeError("ssh is required to hash remote curated arrays")
    command = [ssh, remote_host, "python3", "-", str(curated_path)]
    completed = subprocess.run(  # noqa: S603
        command,
        input=REMOTE_HASH_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote hash command failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout), command


def _compare_arrays(
    raw: dict[str, Any], curated: dict[str, Any]
) -> tuple[dict[str, Any], bool, bool]:
    comparisons = {}
    all_raw_identical = True
    all_reconstructed_identical = True
    for name in ARRAY_FILES:
        raw_array = raw["arrays"][name]
        reconstructed_array = raw["curator_permutation_reconstruction"]["arrays"][name]
        curated_array = curated[name]
        curated_dtype = TORCH_TO_NUMPY_DTYPE[curated_array["dtype"]]
        raw_checks = {
            "shape": raw_array["shape"] == curated_array["shape"],
            "dtype": np.dtype(raw_array["dtype_str"]) == curated_dtype,
            "bytes": raw_array["bytes"] == curated_array["bytes"],
            "sha256": raw_array["sha256"] == curated_array["sha256"],
        }
        reconstructed_checks = {
            "shape": reconstructed_array["shape"] == curated_array["shape"],
            "dtype": np.dtype(reconstructed_array["dtype_str"]) == curated_dtype,
            "bytes": reconstructed_array["bytes"] == curated_array["bytes"],
            "sha256": reconstructed_array["sha256"] == curated_array["sha256"],
        }
        raw_bitwise_identical = all(raw_checks.values())
        reconstructed_bitwise_identical = all(reconstructed_checks.values())
        all_raw_identical &= raw_bitwise_identical
        all_reconstructed_identical &= reconstructed_bitwise_identical
        comparisons[name] = {
            "raw_triangulated": raw_array,
            "curator_permutation_reconstruction": reconstructed_array,
            "curated_pdmsh": curated_array,
            "raw_vs_curated_checks": raw_checks,
            "raw_vs_curated_bitwise_identical": raw_bitwise_identical,
            "reconstruction_vs_curated_checks": reconstructed_checks,
            "reconstruction_vs_curated_bitwise_identical": (
                reconstructed_bitwise_identical
            ),
        }
    return comparisons, all_raw_identical, all_reconstructed_identical


def _match_sampler_case(
    raw: dict[str, Any], sampler_result: dict[str, Any]
) -> dict[str, Any]:
    raw_hash = raw["source"]["sha256"]
    matches = [
        case["source"]
        for case in sampler_result["cases"]
        if case["source"]["sha256"] == raw_hash
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one sampler case with source hash {raw_hash}, "
            f"found {len(matches)}"
        )
    source = matches[0]
    checks = {
        "source_sha256": source["sha256"] == raw_hash,
        "raw_n_points": source["raw_n_points"] == raw["source"]["raw_n_points"],
        "raw_n_cells": source["raw_n_cells"] == raw["source"]["raw_n_cells"],
        "triangulated_n_cells": source["triangulated_n_cells"]
        == raw["arrays"]["cells"]["shape"][0],
    }
    return {
        "sampler_source": source,
        "checks": checks,
        "exact_case_match": all(checks.values()),
    }


def _git_output(repo_root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    completed = subprocess.run(  # noqa: S603
        [git, *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("CASE_ID", "RAW_VTP", "CURATED_PDMSH"),
    )
    parser.add_argument("--remote-host", required=True)
    parser.add_argument(
        "--point-permutation-seed",
        type=int,
        required=True,
        help="Effective seed reproducing the curated vehicle point order.",
    )
    parser.add_argument(
        "--cell-permutation-seed",
        type=int,
        required=True,
        help="Effective seed reproducing the curated vehicle cell order.",
    )
    parser.add_argument(
        "--curator-checkout",
        type=Path,
        required=True,
        help="PhysicsNeMo-Curator checkout containing the audited pipeline.",
    )
    parser.add_argument("--sampler-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    sampler_path = args.sampler_result.resolve()
    with sampler_path.open() as stream:
        sampler_result = json.load(stream)

    cases = []
    all_cases_raw_identical = True
    all_cases_reconstructed = True
    remote_commands = []
    for case_id, raw_path_arg, curated_path_arg in args.case:
        raw = _hash_raw_triangulation(
            Path(raw_path_arg),
            args.point_permutation_seed,
            args.cell_permutation_seed,
        )
        curated, remote_command = _hash_remote_curated(
            args.remote_host, Path(curated_path_arg)
        )
        (
            comparisons,
            raw_arrays_identical,
            reconstructed_arrays_identical,
        ) = _compare_arrays(raw, curated)
        sampler_match = _match_sampler_case(raw, sampler_result)
        case_reconstructed = (
            reconstructed_arrays_identical and sampler_match["exact_case_match"]
        )
        all_cases_raw_identical &= (
            raw_arrays_identical and sampler_match["exact_case_match"]
        )
        all_cases_reconstructed &= case_reconstructed
        remote_commands.append(remote_command)
        cases.append(
            {
                "case_id": case_id,
                "raw_source": raw["source"],
                "curated_source": {
                    "host": args.remote_host,
                    "path": curated_path_arg,
                    "access": "read-only hashing over SSH",
                },
                "transformation": raw["transformation"],
                "arrays": comparisons,
                "curator_permutation_mapping": raw[
                    "curator_permutation_reconstruction"
                ],
                "sampler_artifact_case_match": sampler_match,
                "raw_order_all_four_arrays_bitwise_identical": (raw_arrays_identical),
                "curator_permutation_all_four_arrays_bitwise_identical": (
                    reconstructed_arrays_identical
                ),
                "diagnostic_passed": case_reconstructed,
            }
        )
        del raw, curated
        gc.collect()

    script_path = Path(__file__).resolve()
    curator_checkout = args.curator_checkout.resolve()
    permutation_source = (
        curator_checkout
        / "src/physicsnemo_curator/domains/mesh/filters/random_permutation.py"
    )
    conversion_source = (
        curator_checkout
        / "src/physicsnemo_curator/domains/mesh/sources/_vtk_convert.py"
    )
    pipeline_source = curator_checkout / "examples/cae/drivaerml/main.py"
    if all_cases_raw_identical:
        status = "PASSED_EXACT_CURATED_IDENTITY_FOR_AUDITED_CASES"
        conclusion = (
            "The existing raw-VTP sampler diagnostic is exact curated-order "
            "evidence for run_1 and run_2: points, triangle connectivity, "
            "CpMeanTrim, and wallShearStressMeanTrim are byte-for-byte "
            "identical after the diagnostic's PyVista triangulation."
        )
    elif all_cases_reconstructed:
        status = "ORDER_MISMATCH_EXACT_CURATOR_PERMUTATION_RECONSTRUCTED"
        conclusion = (
            "The raw-derived and curated arrays are not in the same order. "
            "For both audited cases, however, every curated byte is exactly "
            "reproduced by the curator's fan tessellation, seeded point "
            "permutation, inverse connectivity remap, and cell permutation. "
            "The sampler artifact uses PyVista/VTK triangulation and raw cell "
            "order, so it must not be interpreted as a curated-order "
            "diagnostic."
        )
    else:
        status = "FAILED_CURATED_IDENTITY_AND_PERMUTATION_RECONSTRUCTION"
        conclusion = (
            "At least one raw-derived array differs from both its curated "
            "counterpart and the attempted curator-permutation "
            "reconstruction; the raw sampler diagnostic is not promoted."
        )
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "conclusion": conclusion,
        "scope_caveat": (
            "This identity result covers exactly run_1 and run_2. It is not "
            "evidence that the same conversion identity holds for all 435 "
            "training cases, nor is a two-case sampler audit a population "
            "estimate."
        ),
        "hash_definition": (
            "SHA256 over contiguous row-major array bytes. Curated .memmap "
            "files store the same contiguous tensor bytes; shape and dtype "
            "metadata are checked independently."
        ),
        "sampler_artifact": {
            "path": str(sampler_path),
            "sha256": _sha256_file(sampler_path),
            "schema_version": sampler_result.get("schema_version"),
            "original_status": sampler_result.get("status"),
            "interpretation": (
                "Promoted to exact curated-order evidence only for the two "
                "case hashes matched below."
                if all_cases_raw_identical
                else "Not promoted: its raw cell order differs from the "
                "curated on-disk order."
            ),
        },
        "cases": cases,
        "provenance": {
            "repository": str(repo_root),
            "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
            "git_branch": _git_output(repo_root, "branch", "--show-current"),
            "script": {
                "path": str(script_path.relative_to(repo_root)),
                "sha256": _sha256_file(script_path),
            },
            "curator": {
                "checkout": str(curator_checkout),
                "git_commit": _git_output(curator_checkout, "rev-parse", "HEAD"),
                "permutation_source": {
                    "path": str(permutation_source.relative_to(curator_checkout)),
                    "sha256": _sha256_file(permutation_source),
                },
                "polygon_conversion_source": {
                    "path": str(conversion_source.relative_to(curator_checkout)),
                    "sha256": _sha256_file(conversion_source),
                    "relevant_operation": (
                        "fan-tessellate every polygon from its first vertex"
                    ),
                },
                "drivaerml_pipeline_source": {
                    "path": str(pipeline_source.relative_to(curator_checkout)),
                    "sha256": _sha256_file(pipeline_source),
                },
                "configured_base_seed": 42,
                "verified_effective_point_seed": (args.point_permutation_seed),
                "verified_effective_cell_seed": args.cell_permutation_seed,
                "lineage_discrepancy": (
                    "The current checkout's RandomPermutationFilter uses one "
                    "generator sequentially for points and cells. The stored "
                    "curated bytes instead require independent point/cell "
                    "generators with seeds 44/45. This checkout documents the "
                    "intended permutation stage but is not exact source "
                    "identity for the producing cell-order implementation."
                ),
            },
            "command": sys.argv,
            "remote_commands": remote_commands,
            "remote_hash_script_sha256": hashlib.sha256(
                REMOTE_HASH_SCRIPT.encode()
            ).hexdigest(),
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "versions": {
                "numpy": np.__version__,
                "pyvista": pv.__version__,
                "torch": torch.__version__,
                "vtk": vtk.vtkVersion.GetVTKVersion(),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    if not (all_cases_raw_identical or all_cases_reconstructed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
