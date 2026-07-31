# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Freeze only the selected raw targets used by the historical K=10k replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_KIND = "drivaerml_historical_k10000_selected_target_input_manifest"
STATUS = "PASSED_HISTORICAL_K10000_SELECTED_TARGET_INPUT_FREEZE"
RESOLUTION = 10_000
EXPECTED_CASE_COUNT = 36
DATASET_MANIFEST_SHA256 = (
    "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
)
EXPECTED_COHORT_SHA256 = (
    "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
)

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

TARGETS = {
    "pressure": {
        "field": "pMeanTrim",
        "shape_suffix": (),
        "components": 1,
    },
    "wss": {
        "field": "wallShearStressMeanTrim",
        "shape_suffix": (3,),
        "components": 3,
    },
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


def _safe_read_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise ValueError(f"Input changed while being read: {path}")
    return b"".join(chunks)


def _safe_pread(
    path: Path,
    *,
    offset: int,
    count: int,
    expected_file_size: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_file_size:
            raise ValueError(f"Target source size changed: {path}")
        payload = os.pread(descriptor, count, offset)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != count:
        raise ValueError(f"Short target read from {path}: {len(payload)} != {count}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Target source changed while being read: {path}")
    return payload


def _strict_json_bytes(payload: bytes, *, source: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {source}")

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _strict_json(path: Path) -> Any:
    return _strict_json_bytes(_safe_read_bytes(path), source=str(path))


def _inspect_case(
    dataset_root: Path,
    spec: tuple[int, str, int, int, int],
) -> dict[str, Any]:
    ordinal, case_id, reader_index, n_cells, start = spec
    case_link = dataset_root / case_id
    if not case_link.is_symlink():
        raise ValueError(f"Dataset case is not a symlink: {case_link}")
    symlink_target = os.readlink(case_link)
    case_root = case_link.resolve(strict=True)
    tensor_root = (
        case_root
        / f"domain_{case_id}.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
    )
    metadata_path = tensor_root / "cell_data" / "meta.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"Target metadata is not a regular file: {metadata_path}")
    metadata_payload = _safe_read_bytes(metadata_path)
    metadata = _strict_json_bytes(metadata_payload, source=str(metadata_path))
    selected: dict[str, Any] = {}
    for name, contract in TARGETS.items():
        field = contract["field"]
        entry = metadata.get(field)
        expected_shape = [n_cells, *contract["shape_suffix"]]
        if (
            not isinstance(entry, Mapping)
            or entry.get("shape") != expected_shape
            or entry.get("dtype") != "torch.float32"
            or entry.get("device") != "cpu"
        ):
            raise ValueError(f"{case_id} metadata changed for {field}: {entry!r}")
        path = tensor_root / "cell_data" / f"{field}.memmap"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Target source is not a regular file: {path}")
        components = int(contract["components"])
        file_size = n_cells * components * 4
        offset = start * components * 4
        count = RESOLUTION * components * 4
        if offset < 0 or offset + count > file_size:
            raise ValueError(f"{case_id} selected target range is invalid")
        payload = _safe_pread(
            path,
            offset=offset,
            count=count,
            expected_file_size=file_size,
        )
        selected[name] = {
            "raw_field_name": field,
            "source_relative_path": (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                f"_tensordict/cell_data/{field}.memmap"
            ),
            "source_size_bytes": file_size,
            "source_offset_bytes": offset,
            "selected_size_bytes": count,
            "selected_shape": [RESOLUTION, *contract["shape_suffix"]],
            "selected_dtype": "float32_little_endian",
            "selected_sha256": _sha256_bytes(payload),
        }
    return {
        "cohort_ordinal": ordinal,
        "case_id": case_id,
        "reader_index": reader_index,
        "n_master_cells": n_cells,
        "historical_start": start,
        "resolution": RESOLUTION,
        "logical_case_symlink": str(case_link),
        "symlink_target": symlink_target,
        "resolved_case_root": str(case_root),
        "cell_data_metadata": {
            "relative_path": (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                "_tensordict/cell_data/meta.json"
            ),
            "size_bytes": len(metadata_payload),
            "sha256": _sha256_bytes(metadata_payload),
        },
        "selected_targets": selected,
    }


def _atomic_publish(path: Path, payload: bytes) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    for destination in (path, sidecar):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for destination, content in ((path, payload), (sidecar, sidecar_payload)):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(name)
            temporaries[destination] = temporary
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for destination in (path, sidecar):
            temporary = temporaries[destination]
            os.link(temporary, destination, follow_symlinks=False)
            published.append((destination, temporary))
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _safe_read_bytes(path) != payload:
            raise RuntimeError(f"Published payload verification failed: {path}")
        if _safe_read_bytes(sidecar) != sidecar_payload:
            raise RuntimeError(f"Published sidecar verification failed: {sidecar}")
    except BaseException:
        for destination, temporary in reversed(published):
            try:
                destination_stat = destination.stat(follow_symlinks=False)
                temporary_stat = temporary.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (destination_stat.st_dev, destination_stat.st_ino) == (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            ):
                destination.unlink()
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset_root_input = Path(os.path.abspath(args.dataset_root))
    if dataset_root_input.is_symlink() or not dataset_root_input.is_dir():
        raise ValueError("Dataset root must be a regular directory")
    dataset_root = dataset_root_input.resolve(strict=True)
    output = Path(os.path.abspath(args.output_json))
    dataset_manifest = dataset_root / "manifest.json"
    dataset_manifest_payload = _safe_read_bytes(dataset_manifest)
    if _sha256_bytes(dataset_manifest_payload) != DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest changed")
    dataset_manifest_document = _strict_json_bytes(
        dataset_manifest_payload,
        source=str(dataset_manifest),
    )
    expected_case_ids = [spec[1] for spec in CASE_SPECS]
    cohort_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "cohort_ordinal": row[0],
                    "case_id": row[1],
                    "reader_index": row[2],
                    "n_master_cells": row[3],
                    "historical_start": row[4],
                }
                for row in CASE_SPECS
            ]
        )
    )
    if len(CASE_SPECS) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Frozen selected-target cohort size changed")
    if cohort_sha256 != EXPECTED_COHORT_SHA256:
        raise RuntimeError("Frozen selected-target cohort identity changed")
    if (
        not isinstance(dataset_manifest_document, Mapping)
        or dataset_manifest_document.get("id_reference") != expected_case_ids
    ):
        raise RuntimeError("Dataset reference cohort order changed")
    cases = [_inspect_case(dataset_root, spec) for spec in CASE_SPECS]
    if (
        len(cases) != EXPECTED_CASE_COUNT
        or [case["case_id"] for case in cases] != expected_case_ids
    ):
        raise RuntimeError("Selected-target cohort coverage changed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_input": str(dataset_root_input),
        "dataset_root_resolved": str(dataset_root),
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "case_count": len(cases),
        "resolution": RESOLUTION,
        "selection": "nonwrapping contiguous historical start through start+9999",
        "read_allowlist": [
            "cell_data/meta.json",
            "cell_data/pMeanTrim.memmap selected byte range only",
            "cell_data/wallShearStressMeanTrim.memmap selected byte range only",
        ],
        "read_exclusions": {
            "other_cell_data_opened": False,
            "point_data_opened": False,
            "interior_opened": False,
            "model_output_generated": False,
        },
        "cases": cases,
        "cohort_sha256": cohort_sha256,
        "provenance": {
            "command": list(os.sys.argv),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_bytes(_safe_read_bytes(Path(__file__))),
        },
    }
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    digest = _atomic_publish(output, payload)
    print(f"{STATUS} cases={len(cases)} json_sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
