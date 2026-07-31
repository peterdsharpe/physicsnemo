# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the target-free geometry inputs for the 36-case validity study.

Only the vehicle point/connectivity tensors, structural TensorDict metadata,
and the five explicitly required case-global physical inputs are opened.  No
point-data, cell-data, interior, pressure, or wall-shear-stress file is
traversed, opened, or hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_KIND = "drivaerml_target_free_geometry_input_manifest"
STATUS = "PASSED_TARGET_FREE_GEOMETRY_INPUT_FREEZE"

CASE_SPECS = (
    (0, "run_118", 21, 17_504_739, 14_045_027),
    (1, "run_129", 33, 16_380_547, 14_700_754),
    (2, "run_145", 51, 15_789_064, 9_195_926),
    (3, "run_149", 55, 18_007_064, 4_452_828),
    (4, "run_17", 77, 19_404_150, 6_369_582),
    (5, "run_171", 79, 18_792_923, 1_320_415),
    (6, "run_18", 88, 14_634_570, 10_215_595),
    (7, "run_183", 92, 14_932_664, 7_635_018),
    (8, "run_197", 107, 18_934_869, 16_494_923),
    (9, "run_202", 114, 17_796_743, 15_267_620),
    (10, "run_225", 136, 15_024_109, 3_789_927),
    (11, "run_270", 185, 18_857_430, 10_967_997),
    (12, "run_271", 186, 16_922_213, 5_453_831),
    (13, "run_298", 212, 15_063_884, 4_943_208),
    (14, "run_305", 221, 18_022_481, 16_998_850),
    (15, "run_320", 237, 16_199_351, 15_062_581),
    (16, "run_367", 285, 18_958_141, 5_352_845),
    (17, "run_380", 298, 19_519_305, 11_721_918),
    (18, "run_382", 300, 16_887_630, 11_083_431),
    (19, "run_399", 318, 16_222_090, 15_155_572),
    (20, "run_4", 319, 16_294_644, 13_228_777),
    (21, "run_409", 329, 16_591_548, 1_346_462),
    (22, "run_419", 340, 14_561_784, 12_777_694),
    (23, "run_424", 346, 16_588_938, 13_358_519),
    (24, "run_429", 351, 17_738_132, 365_298),
    (25, "run_431", 354, 15_747_949, 1_091_720),
    (26, "run_439", 362, 17_809_120, 8_840_407),
    (27, "run_465", 391, 16_443_085, 11_669_428),
    (28, "run_468", 394, 18_343_677, 15_504_945),
    (29, "run_469", 395, 19_780_049, 19_757_508),
    (30, "run_478", 404, 16_648_431, 16_079_300),
    (31, "run_489", 416, 16_063_459, 6_463_342),
    (32, "run_490", 418, 17_847_065, 191_824),
    (33, "run_495", 423, 15_715_663, 11_592_670),
    (34, "run_71", 453, 16_516_082, 2_240_523),
    (35, "run_86", 469, 17_188_261, 4_374_650),
)
GLOBAL_INPUTS = ("U_inf", "p_inf", "rho_inf", "nu", "L_ref")
FORBIDDEN_PATH_PARTS = {
    "cell_data",
    "point_data",
    "interior",
    "pmeantrim",
    "wallshearstressmeantrim",
    "pressure",
    "wss",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {path}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )


def _relative_paths(case_id: str) -> tuple[Path, ...]:
    domain = Path(f"domain_{case_id}.pdmsh")
    root = domain / "_tensordict"
    vehicle = root / "boundaries" / "vehicle"
    vehicle_td = vehicle / "_tensordict"
    global_data = root / "global_data"
    paths = (
        domain / "meta.json",
        root / "meta.json",
        root / "boundaries" / "meta.json",
        vehicle / "meta.json",
        vehicle_td / "meta.json",
        vehicle_td / "points.memmap",
        vehicle_td / "cells.memmap",
        global_data / "meta.json",
        *(global_data / f"{name}.memmap" for name in GLOBAL_INPUTS),
    )
    for path in paths:
        lowered = {part.lower() for part in path.parts}
        forbidden = lowered.intersection(FORBIDDEN_PATH_PARTS)
        if forbidden:
            raise ValueError(
                f"Geometry allowlist contains forbidden path parts: {path}"
            )
    return paths


def _validate_metadata(
    case_root: Path,
    case_id: str,
    expected_n_cells: int,
) -> tuple[int, int]:
    domain = case_root / f"domain_{case_id}.pdmsh"
    vehicle_meta_path = (
        domain / "_tensordict" / "boundaries" / "vehicle" / "_tensordict" / "meta.json"
    )
    global_meta_path = domain / "_tensordict" / "global_data" / "meta.json"
    vehicle_meta = _strict_json(vehicle_meta_path)
    global_meta = _strict_json(global_meta_path)

    point_meta = vehicle_meta.get("points")
    cell_meta = vehicle_meta.get("cells")
    if not isinstance(point_meta, Mapping) or not isinstance(cell_meta, Mapping):
        raise ValueError(f"{case_id} vehicle metadata lacks points or cells")
    point_shape = point_meta.get("shape")
    cell_shape = cell_meta.get("shape")
    if (
        not isinstance(point_shape, list)
        or len(point_shape) != 2
        or type(point_shape[0]) is not int
        or point_shape[1] != 3
        or point_meta.get("dtype") != "torch.float32"
        or point_meta.get("device") != "cpu"
    ):
        raise ValueError(f"{case_id} point metadata changed: {point_meta}")
    if (
        cell_shape != [expected_n_cells, 3]
        or cell_meta.get("dtype") != "torch.int64"
        or cell_meta.get("device") != "cpu"
    ):
        raise ValueError(f"{case_id} cell metadata changed: {cell_meta}")
    n_points = point_shape[0]

    observed_global_names = {
        key for key, value in global_meta.items() if isinstance(value, Mapping)
    }
    if observed_global_names != set(GLOBAL_INPUTS):
        raise ValueError(
            f"{case_id} global input names changed: {sorted(observed_global_names)}"
        )
    for name in GLOBAL_INPUTS:
        metadata = global_meta[name]
        expected_shape = [3] if name == "U_inf" else []
        if (
            metadata.get("shape") != expected_shape
            or metadata.get("dtype") != "torch.float32"
            or metadata.get("device") != "cpu"
        ):
            raise ValueError(f"{case_id} global metadata changed for {name}")

    points = vehicle_meta_path.parent / "points.memmap"
    cells = vehicle_meta_path.parent / "cells.memmap"
    if points.stat().st_size != n_points * 3 * 4:
        raise ValueError(f"{case_id} point file size differs from metadata")
    if cells.stat().st_size != expected_n_cells * 3 * 8:
        raise ValueError(f"{case_id} cell file size differs from metadata")
    return n_points, expected_n_cells


def _read_float32(path: Path, count: int) -> list[float]:
    payload = path.read_bytes()
    if len(payload) != count * 4:
        raise ValueError(f"Unexpected float32 payload size for {path}")
    return [float(value) for value in struct.unpack(f"<{count}f", payload)]


def _inspect_case(
    dataset_root: Path,
    spec: tuple[int, str, int, int, int],
    *,
    workers: int,
) -> dict[str, Any]:
    cohort_ordinal, case_id, reader_index, n_master_cells, historical_start = spec
    case_link = dataset_root / case_id
    link_stat = case_link.lstat()
    if not stat.S_ISLNK(link_stat.st_mode):
        raise ValueError(f"Case entry is not a symlink: {case_link}")
    symlink_target = os.readlink(case_link)
    case_root = case_link.resolve(strict=True)
    if not case_root.is_dir():
        raise ValueError(f"Case target is not a directory: {case_root}")

    relative_paths = _relative_paths(case_id)
    paths = [case_root / relative for relative in relative_paths]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Allowed input is not a regular non-symlink file: {path}")
    n_points, n_cells = _validate_metadata(
        case_root,
        case_id,
        n_master_cells,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        digests = list(executor.map(_sha256_file, paths))
    files = {
        relative.as_posix(): {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        for relative, path, digest in zip(relative_paths, paths, digests, strict=True)
    }

    global_root = case_root / f"domain_{case_id}.pdmsh" / "_tensordict" / "global_data"
    global_values = {
        name: _read_float32(
            global_root / f"{name}.memmap",
            3 if name == "U_inf" else 1,
        )
        for name in GLOBAL_INPUTS
    }
    if global_values["L_ref"] != [5.0]:
        raise ValueError(f"{case_id} L_ref changed: {global_values['L_ref']}")
    return {
        "cohort_ordinal": cohort_ordinal,
        "case_id": case_id,
        "reader_index": reader_index,
        "n_master_cells": n_master_cells,
        "n_master_points": n_points,
        "historical_start": historical_start,
        "logical_symlink": str(case_link),
        "symlink_target": symlink_target,
        "resolved_case_root": str(case_root),
        "global_input_values_float32": global_values,
        "files": files,
    }


def _validate_output_target(output: Path) -> None:
    sidecar = output.with_name(f"{output.name}.sha256")
    if (
        output.exists()
        or output.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise FileExistsError(f"Refusing to overwrite output or sidecar: {output}")


def _write_fsynced_temporary(path: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _unlink_if_same_inode(path: Path, reference: Path) -> None:
    """Remove ``path`` only if it is still our published hard link."""
    try:
        path_stat = path.stat(follow_symlinks=False)
        reference_stat = reference.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (path_stat.st_dev, path_stat.st_ino) == (
        reference_stat.st_dev,
        reference_stat.st_ino,
    ):
        path.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_manifest(output: Path, payload: bytes) -> str:
    """Publish a JSON/sidecar pair atomically without replacing either path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(f"{output.name}.sha256")
    digest = _sha256_bytes(payload)
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    output_temporary: Path | None = None
    sidecar_temporary: Path | None = None
    output_published = False
    sidecar_published = False
    try:
        output_temporary = _write_fsynced_temporary(output, payload)
        sidecar_temporary = _write_fsynced_temporary(sidecar, sidecar_payload)
        # POSIX hard-link creation is an atomic no-clobber publication
        # primitive: an existing file, directory, or symlink yields EEXIST.
        os.link(output_temporary, output, follow_symlinks=False)
        output_published = True
        os.link(sidecar_temporary, sidecar, follow_symlinks=False)
        sidecar_published = True
        _fsync_directory(output.parent)
        if _sha256_file(output) != digest:
            raise OSError("Published geometry manifest failed SHA-256 verification")
        if sidecar.read_bytes() != sidecar_payload:
            raise OSError("Published geometry manifest sidecar changed")
    except BaseException:
        if sidecar_published and sidecar_temporary is not None:
            _unlink_if_same_inode(sidecar, sidecar_temporary)
        if output_published and output_temporary is not None:
            _unlink_if_same_inode(output, output_temporary)
        _fsync_directory(output.parent)
        raise
    finally:
        if output_temporary is not None:
            output_temporary.unlink(missing_ok=True)
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)
        _fsync_directory(output.parent)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if type(args.workers) is not int or not 1 <= args.workers <= 16:
        raise ValueError("--workers must be an integer in [1,16]")
    dataset_root_input = Path(os.path.abspath(args.dataset_root))
    if dataset_root_input.is_symlink() or not dataset_root_input.is_dir():
        raise ValueError("Dataset root must be a regular directory, not a symlink")
    dataset_root = dataset_root_input.resolve(strict=True)
    output = Path(os.path.abspath(args.output_json))
    _validate_output_target(output)

    dataset_manifest = dataset_root / "manifest.json"
    if dataset_manifest.is_symlink() or not dataset_manifest.is_file():
        raise ValueError("Dataset manifest must be a regular non-symlink file")
    cohort_payload = [
        {
            "cohort_ordinal": ordinal,
            "case_id": case_id,
            "reader_index": reader_index,
            "n_master_cells": n_cells,
            "historical_start": start,
        }
        for ordinal, case_id, reader_index, n_cells, start in CASE_SPECS
    ]
    cases = [
        _inspect_case(
            dataset_root,
            spec,
            workers=args.workers,
        )
        for spec in CASE_SPECS
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_input": str(dataset_root_input),
        "dataset_root_resolved": str(dataset_root),
        "dataset_manifest": {
            "path": str(dataset_manifest),
            "sha256": _sha256_file(dataset_manifest),
            "size_bytes": dataset_manifest.stat().st_size,
        },
        "cohort_sha256": _sha256_bytes(_canonical_json_bytes(cohort_payload)),
        "case_count": len(cases),
        "cases": cases,
        "target_exclusion": {
            "opened_associations": [
                "vehicle geometry points",
                "vehicle geometry cells",
                "case global physical inputs",
                "structural metadata",
            ],
            "forbidden_path_parts": sorted(FORBIDDEN_PATH_PARTS),
            "point_data_opened": False,
            "cell_data_opened": False,
            "interior_opened": False,
            "supervision_values_opened": False,
            "supervision_values_hashed": False,
            "supervision_values_serialized": False,
        },
        "provenance": {
            "command": list(os.sys.argv),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "workers": args.workers,
        },
    }
    if len(cases) != 36 or [case["case_id"] for case in cases] != [
        spec[1] for spec in CASE_SPECS
    ]:
        raise RuntimeError("Geometry manifest cohort coverage changed")
    payload = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_publish_manifest(output, payload)
    print(f"{STATUS} cases={len(cases)} output={output}", flush=True)


if __name__ == "__main__":
    main()
