# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reduce the sealed canonical fixed-prefix H-QC panel.

This is the sole categorical publisher for the conditional H-QC successor.
It runs no model and opens no dataset.  BF16 pressure is deciding; FP32 and
WSS are ordered diagnostics.  Missing evidence is incomplete, present bad
evidence is invalid, and every output is published exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import os
import re
import stat
import statistics
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_KIND = "phase1_hqc_canonical_fixed_prefix_adjudication"
VALID_STATUS = "VALID_HQC_CANONICAL_FIXED_PREFIX_ADJUDICATION"
INVALID_STATUS = "INVALID_HQC_CANONICAL_FIXED_PREFIX_ADJUDICATION"
INCOMPLETE_STATUS = "INCOMPLETE_HQC_CANONICAL_FIXED_PREFIX_ADJUDICATION"

INCOMPLETE_OUTCOME = "INCOMPLETE_HQC_CANONICAL_FIXED_PREFIX"
INVALID_OUTCOME = "INVALID_HQC_CANONICAL_FIXED_PREFIX"
INELIGIBLE_OUTCOME = "INELIGIBLE_HQC_CANONICAL_FIXED_PREFIX"
FUTILE_OUTCOME = "FUTILE_HQC_CANONICAL_FIXED_PREFIX"
MIXED_OUTCOME = "MIXED_HQC_CANONICAL_FIXED_PREFIX"
DUAL_OUTCOME = "SUPPORTED_HQC_ALIGNED_FIXED_PREFIX_DUAL_WEIGHTING"
AREA_FLAT_OUTCOME = "SUPPORTED_HQC_UNIFORM_CANONICAL_AREA_NEARLY_FLAT"
UNIFORM_ONLY_OUTCOME = "SUPPORTED_HQC_UNIFORM_ONLY_METRIC_SPECIFIC"

RESOLUTIONS = (2_500, 5_000, 10_000, 20_000, 40_000)
BASELINE_K = 10_000
FIXED_Q = 2_500
ENDPOINTS = (2_500, 40_000)
PRECISIONS = ("bfloat16", "float32")

BASELINE_BOUNDS = (0.5, 2.0)
CLIFF_LOG_MIN = math.log(2.0)
CLIFF_RATIO_MIN = 2.0
CLIFF_COUNT_MIN = 24
SUPPORT_FRACTION_MAX = 0.25
SUPPORT_FIXED_LOG_MAX = math.log(1.25)
SUPPORT_FAVORABLE_MIN = 27
FUTILITY_FRACTION_MIN = 0.5
FUTILITY_K40_RATIO_MIN = 2.0
AREA_FLAT_RATIO_MAX = 1.25
K10_METRIC_ATOL = 1.0e-12
CANONICAL_AREA_K10_PRESSURE_MEAN = 0.18187839912525805
CANONICAL_AREA_K10_WSS_MEAN = 0.21075915210639293
CANONICAL_AREA_K10_MEAN_ATOL = 1.0e-14

EXPECTED_COHORT_SHA256 = (
    "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
)
EXPECTED_DATASET_SHA256 = (
    "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
)
EXPECTED_HISTORICAL_TARGET_SHA256 = (
    "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49"
)
EXPECTED_TARGET_PRODUCER_SHA256 = (
    "8cbd315e0e91b4a08adf695eb7342d34030a1135f86982554bc3a09e7bb63440"
)
EXPECTED_TARGET_PRODUCER_TEST_SHA256 = (
    "adb04fed7ec7429002e3c1956a5e05818777cef39f54b785a20e643ffcef6ec5"
)
EXPECTED_ONE_STEP_PRODUCER_SHA256 = (
    "f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500"
)

EXPECTED_LANE_JSON_SHA256 = (
    "d8ddad6078e789e43f5f1160ebd894e65fc7c42f26d3b10e9cf5530702d42447",
    "4588a8e10df4e621fc4e8c0d693d8a5f4d1cc51487d42c5a6e211346502e8e9b",
    "f89bf6fd08c7996ba1e37b1806cb900ec26c3191e4779567b3390c1c0b58c3bb",
    "a0aad20fadf01fadfca1ede36dc1e91ec100d8b1bc50b031bb1cdb157b6ddb2b",
)
EXPECTED_LANE_NPZ_SHA256 = (
    "451521ae74ffa1964ddcfab50a135694507a41e13f074e3b2ec87c38b6914e2f",
    "cf133c5ea47e74d32ebc3c9feeec73485b4a9b43c2b6e3f3715830b76ab3fe19",
    "d503ffbc5a5042ed887b9988b7f3205d7fed246c3f1da998fb14f68369d6b18a",
    "f0f34ec5c5b5197cdf9c9ba34cacb8141d45ac19052a5a7f8c0a0974d5e3685d",
)
EXPECTED_CANONICAL_K10_JSON_SHA256 = (
    "5f22307064457240e7e7a3955e07e6ac6d61d7591f5f5af1edd83a91d6478ac4"
)
EXPECTED_CANONICAL_K10_NPZ_SHA256 = (
    "6225609a551c7b1ef7474c974a5c135f494bb47d483073bdb1b454c07733d84d"
)
EXPECTED_STAGE_B_K10_JSON_SHA256 = (
    "83f0d53326a128def18d538d427ac48b11a0d789417366cfd7c1dc020b193d67"
)
EXPECTED_STAGE_B_K10_NPZ_SHA256 = (
    "38353473af9eba1e606e170c70c5d6ed4d7fd736d631c6388dd2244dc1e203c4"
)
EXPECTED_ACCEPTED_ADJUDICATION_SHA256 = (
    "a6698cfe79c6651ba686f175235d089286d79a14437bf3b190ea267b0e9e1bce"
)
EXPECTED_PREREGISTRATION_CONTRACT_SHA256 = (
    "41a9f641023ff1b9baf6e38a1f6e0ba4dc360a5cefbb8e7503d16cb4cdb9d30c"
)
LAUNCH_MANIFEST_KIND = "drivaerml_hqc_nested_target_freeze_launch_manifest"
LAUNCH_MANIFEST_STATUS = "FROZEN_HQC_NESTED_TARGET_FREEZE_LAUNCH"
ATTEMPT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

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
EXPECTED_METADATA_SHA256 = (
    "a9ca75da58e64210b6dcfc07ca769fed473add2f0404b04e5b3a119d1f59487d",
    "104fe0464220affbf80c21462ae545e17745cfab422285ffd03b68f4a132b6e4",
    "b61585f06a4b94332318425dea29fef6dd60b28c9346c6e416a5f14d39cb460b",
    "9b5dc694f200c3ce2bffc049b20fbf6ccbe47af7ff420c11f5e2c557d3dfe284",
    "805f9e1fb3799f0d6d122fc5044262daabac8f4f23ad10e4540426fc82249ff3",
    "626d1b70df42c745eb1318227f90487e1ad80d67733eaed7566201b10a43065b",
    "eda98c3decbe0ac018d801d4ca9d0622b3be9190a35636d681574913abf2a35b",
    "9404aba883b937f1e4c9420afdb309342e518fb59e05cf5fa4777f030ee4e9b9",
    "75ae597b5fe99f4281b8431d263bdba18c383b02dc024353e8011121661e16d3",
    "a419234ae622e69bf365941bd765f5dcd21aed55f5ebd9bc847ce6e9d54cbf68",
    "dd4b052ef6a4b34bb169061dace17429503560408cbc78582633419c064e93ff",
    "4ccff9192304e85e0505391c1ce5842e64f85392a73d6ead9990296b747f8836",
    "61e0e5159f1b53fde094da82d2660f50ffacddd1f574347946d9a9e205e8f532",
    "e8052583b15e1f0d0a16d4787f9444caf0906b804e3ba435602aae920a915894",
    "ec92859fa946c6ff29c818059070006676957f13ed202669e03fb14adfb35360",
    "68aea1d5e17526190f3160957c966820fba516e40695fe784bf236f234e54709",
    "2fc958f822745675fea248ffff84300c98ed4f7f1855d33153d150bab39dbdc1",
    "756b0bab2ec80b875924c879440bef71f14e70e0953e672e5ff3fac24f37055b",
    "f2adcbc9fd8f55d96c1f9ce3ec9bb969392ff4a73ba7b3ca3b3c03f29534d23a",
    "33ddf239ed1a1193021430d260e74bb64c3252b9eb1b425b52a3f65fa589bc09",
    "60bf75dead21a4f8b45831685e7e39ff10994fcbc0f69c87ef28dbd71667e14e",
    "420bafee9b08ae3045953f1eb211a3b92a89b995f7bad9571c47d2bded552408",
    "7cd00b2d286962fdc34048898cf96b1ff46dd92bd0461ac57ebf098403b79e34",
    "85fe6be437516f4c6cf9a4900a445d27bd76403d16a2740112e65e5bb40df590",
    "5d545fa71ca9a6ac866ff089272cc476449d60bd8415e3d6a5bda9da1d28c29f",
    "8f0949c39cba60d84d274498a855b9262b8f3f467edb82d2e626ea6322c5bdcb",
    "d1436ca9663e77fc89d00c710743ecde632dc2c798531a1511a5e4d2209b7e9e",
    "5e93241b562cc097c63d88ceb55e95559b4024c658e0c4a70ed4d1ff05c9eeb1",
    "6f07c903b41e2eddb8afc45dae8b35721c2b25ac523e3847498490a31465fc51",
    "3293ef9b84883a5e8f7ee6d12bb291834ebe62adb91193f0db24dc3da3193043",
    "2b733dd88f8b6d3e6664a334838cc6cf10de0d65e0fa60dfe7ac686aff461b3e",
    "276f12f284e08e6f6e5059609f6e8e89ae25c11d7c77d28c127914fa0185b8d8",
    "077ddccde81a48d3ee39bf6bc6e6da8e1a66b4a0611d4a046cb6cbc94ad29a7d",
    "be745734de16b4d0d5a3ed051fd574c3dd04d56dfc6e7f6659fb6631d5e63b4d",
    "f645c7dc22636e54e04c2f9b35b15b1707b4d53a0070a2e2f33157c7ebe1420d",
    "259b8477ba038b88815fb6c3fbe9dc74c4cbd524894968294d3046ddcba2498f",
)
EXPECTED_METADATA_SIZE_BYTES = (506,) * 13 + (464,) * 23

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class IncompleteEvidence(RuntimeError):
    """A required artifact is not present yet."""


class InvalidEvidence(RuntimeError):
    """Present evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def _json_exact(value: Any, expected: Any) -> bool:
    """Compare strict-JSON values without Python bool/int coercion."""
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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise IncompleteEvidence(f"required artifact is absent: {path}") from error
    except OSError as error:
        raise InvalidEvidence(f"could not open artifact safely: {path}") from error
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise InvalidEvidence(f"artifact is not a regular file: {path}")
            chunks = []
            while chunk := os.read(descriptor, 8 << 20):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        except OSError as error:
            raise InvalidEvidence(f"could not read artifact safely: {path}") from error
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise InvalidEvidence(f"artifact changed while being read: {path}")
    return b"".join(chunks)


def _verified_payload(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[bytes, str]:
    payload = _stable_read(path)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        sidecar_payload = _stable_read(sidecar)
    except IncompleteEvidence as error:
        raise IncompleteEvidence(f"canonical sidecar is absent: {sidecar}") from error
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar_payload != expected_sidecar:
        raise InvalidEvidence(f"canonical sidecar differs: {sidecar}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise InvalidEvidence(f"frozen SHA-256 differs: {path}")
    return payload, digest


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidEvidence(f"{context} has duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise InvalidEvidence(f"{context} has non-finite token {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as error:
        raise InvalidEvidence(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise InvalidEvidence(f"{context} must be a JSON object")
    return value


def _npz_arrays(payload: bytes, *, context: str) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise InvalidEvidence(f"{context} contains duplicate ZIP members")
            if any(
                not name.endswith(".npy")
                or Path(name).name != name
                or name.startswith(".")
                for name in names
            ):
                raise InvalidEvidence(f"{context} contains invalid ZIP members")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (
        InvalidEvidence,
        ValueError,
        TypeError,
        OSError,
        EOFError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
        lzma.LZMAError,
    ) as error:
        if isinstance(error, InvalidEvidence):
            raise
        raise InvalidEvidence(f"{context} is not a valid no-pickle NPZ") from error
    return arrays


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _typed_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in arrays:
        raise InvalidEvidence(f"required NPZ array is absent: {key}")
    value = arrays[key]
    if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
        raise InvalidEvidence(f"NPZ array contract differs: {key}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise InvalidEvidence(f"NPZ array is non-finite: {key}")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidEvidence(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidEvidence(f"{context} must be an array")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise InvalidEvidence("adjudication contains a non-JSON value") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_once(path: Path, document: Mapping[str, Any]) -> str:
    """Publish a sidecar and JSON commit marker monotonically, exactly once."""
    payload = _canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    destinations = ((sidecar, sidecar_payload), (path, payload))
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"output directory is invalid: {path.parent}")
    for destination, _ in destinations:
        try:
            os.lstat(destination)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"refusing to overwrite {destination}")

    temporaries: list[tuple[Path, Path]] = []
    sidecar_temporary: Path | None = None
    try:
        for destination, content in destinations:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            temporaries.append((destination, temporary))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        sidecar_temporary = temporaries[0][1]
        json_temporary = temporaries[1][1]
        os.link(sidecar_temporary, sidecar, follow_symlinks=False)
        _fsync_directory(path.parent)
        if _stable_read(sidecar) != sidecar_payload:
            raise RuntimeError(f"published payload differs: {sidecar}")
        # This link is the irreversible commit point. Once any JSON entry is
        # externally visible, never make the namespace non-monotonic by
        # removing either destination, even if this syscall reports an error.
        os.link(json_temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        for destination, content in destinations:
            if _stable_read(destination) != content:
                raise RuntimeError(f"published payload differs: {destination}")
    except BaseException as error:
        try:
            os.lstat(path)
        except FileNotFoundError:
            json_visible = False
        except OSError:
            # An uninspectable commit-marker pathname is conservatively
            # treated as visible: rollback must never make it disappear.
            json_visible = True
        else:
            json_visible = True
        if not json_visible and sidecar_temporary is not None:
            try:
                sidecar_stat = sidecar.stat(follow_symlinks=False)
                temporary_stat = sidecar_temporary.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (sidecar_stat.st_dev, sidecar_stat.st_ino) == (
                    temporary_stat.st_dev,
                    temporary_stat.st_ino,
                ):
                    sidecar.unlink()
            try:
                _fsync_directory(path.parent)
            except OSError as cleanup_error:
                error.add_note(
                    f"sidecar rollback directory fsync also failed: {cleanup_error}"
                )
        raise
    finally:
        for _, temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return digest


def _preflight_inputs(
    records: Sequence[tuple[Path, str | None, bool]],
) -> list[str]:
    """Classify every path before reduction so present bad evidence wins."""
    incomplete: list[str] = []
    invalid: list[str] = []
    for path, expected_sha256, requires_sidecar in records:
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            incomplete.append(f"required artifact is absent: {path}")
            if requires_sidecar:
                orphan_sidecar = path.with_name(f"{path.name}.sha256")
                try:
                    orphan_metadata = os.lstat(orphan_sidecar)
                except FileNotFoundError:
                    continue
                except OSError:
                    invalid.append(
                        f"could not inspect orphan canonical sidecar safely: "
                        f"{orphan_sidecar}"
                    )
                    continue
                if stat.S_ISLNK(orphan_metadata.st_mode) or not stat.S_ISREG(
                    orphan_metadata.st_mode
                ):
                    invalid.append(
                        "orphan canonical sidecar is not a regular "
                        f"non-symlink file: {orphan_sidecar}"
                    )
                    continue
                try:
                    orphan_payload = _stable_read(orphan_sidecar)
                except IncompleteEvidence as error:
                    incomplete.append(str(error))
                    continue
                except InvalidEvidence as error:
                    invalid.append(str(error))
                    continue
                suffix = f"  {path.name}\n".encode("ascii")
                if (
                    len(orphan_payload) != 64 + len(suffix)
                    or orphan_payload[64:] != suffix
                    or re.fullmatch(rb"[0-9a-f]{64}", orphan_payload[:64]) is None
                ):
                    invalid.append(
                        f"orphan canonical sidecar is malformed: {orphan_sidecar}"
                    )
            continue
        except OSError:
            invalid.append(f"could not inspect artifact safely: {path}")
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            invalid.append(f"artifact is not a regular non-symlink file: {path}")
            continue
        try:
            payload = _stable_read(path)
        except IncompleteEvidence as error:
            incomplete.append(str(error))
            continue
        except InvalidEvidence as error:
            invalid.append(str(error))
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            invalid.append(f"frozen SHA-256 differs: {path}")
        if not requires_sidecar:
            continue
        sidecar = path.with_name(f"{path.name}.sha256")
        try:
            sidecar_metadata = os.lstat(sidecar)
        except FileNotFoundError:
            incomplete.append(f"canonical sidecar is absent: {sidecar}")
            continue
        except OSError:
            invalid.append(f"could not inspect canonical sidecar safely: {sidecar}")
            continue
        if stat.S_ISLNK(sidecar_metadata.st_mode) or not stat.S_ISREG(
            sidecar_metadata.st_mode
        ):
            invalid.append(
                f"canonical sidecar is not a regular non-symlink file: {sidecar}"
            )
            continue
        try:
            observed = _stable_read(sidecar)
        except IncompleteEvidence as error:
            incomplete.append(str(error))
            continue
        except InvalidEvidence as error:
            invalid.append(str(error))
            continue
        expected = f"{digest}  {path.name}\n".encode("ascii")
        if observed != expected:
            invalid.append(f"canonical sidecar differs: {sidecar}")
    if invalid:
        raise InvalidEvidence("; ".join(invalid))
    return incomplete


def _load_json_once(
    path: Path,
    *,
    context: str,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any] | None, str | None, bool, list[str]]:
    """Read and parse one JSON artifact once, then authenticate its sidecar."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        sidecar = path.with_name(f"{path.name}.sha256")
        try:
            sidecar_metadata = os.lstat(sidecar)
        except FileNotFoundError:
            return None, None, False, [f"required artifact is absent: {path}"]
        except OSError as error:
            raise InvalidEvidence(
                f"could not inspect orphan canonical sidecar safely: {sidecar}"
            ) from error
        if stat.S_ISLNK(sidecar_metadata.st_mode) or not stat.S_ISREG(
            sidecar_metadata.st_mode
        ):
            raise InvalidEvidence(
                f"orphan canonical sidecar is not a regular non-symlink file: {sidecar}"
            )
        try:
            sidecar_payload = _stable_read(sidecar)
        except IncompleteEvidence:
            return None, None, False, [f"required artifact is absent: {path}"]
        suffix = f"  {path.name}\n".encode("ascii")
        if (
            len(sidecar_payload) != 64 + len(suffix)
            or sidecar_payload[64:] != suffix
            or re.fullmatch(rb"[0-9a-f]{64}", sidecar_payload[:64]) is None
        ):
            raise InvalidEvidence(f"orphan canonical sidecar is malformed: {sidecar}")
        return None, None, False, [f"required artifact is absent: {path}"]
    except OSError as error:
        raise InvalidEvidence(f"could not inspect artifact safely: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise InvalidEvidence(f"artifact is not a regular non-symlink file: {path}")
    try:
        payload = _stable_read(path)
    except IncompleteEvidence:
        return None, None, False, [f"required artifact is absent: {path}"]
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise InvalidEvidence(f"frozen SHA-256 differs: {path}")
    document = _strict_json(payload, context=context)
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        sidecar_metadata = os.lstat(sidecar)
    except FileNotFoundError:
        return (
            document,
            digest,
            False,
            [f"canonical sidecar is absent: {sidecar}"],
        )
    except OSError as error:
        raise InvalidEvidence(
            f"could not inspect canonical sidecar safely: {sidecar}"
        ) from error
    if stat.S_ISLNK(sidecar_metadata.st_mode) or not stat.S_ISREG(
        sidecar_metadata.st_mode
    ):
        raise InvalidEvidence(
            f"canonical sidecar is not a regular non-symlink file: {sidecar}"
        )
    try:
        observed_sidecar = _stable_read(sidecar)
    except IncompleteEvidence:
        return (
            document,
            digest,
            False,
            [f"canonical sidecar is absent: {sidecar}"],
        )
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if observed_sidecar != expected_sidecar:
        raise InvalidEvidence(f"canonical sidecar differs: {sidecar}")
    return document, digest, True, []


def _validate_array_manifest(
    manifest: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    context: str,
    require_nbytes: bool = False,
) -> None:
    records = _mapping(manifest, f"{context} array manifest")
    _require(set(records) == set(arrays), f"{context} array names differ")
    for key, value in arrays.items():
        record = _mapping(records[key], f"{context} manifest entry {key}")
        expected_keys = {"shape", "dtype", "sha256"}
        if require_nbytes:
            expected_keys.add("nbytes")
        _require(
            set(record) == expected_keys,
            f"{context} manifest entry fields differ: {key}",
        )
        _require(
            _json_exact(record.get("shape"), list(value.shape)),
            f"{context} manifest shape differs: {key}",
        )
        dtype_text = record.get("dtype")
        _require(
            isinstance(dtype_text, str)
            and dtype_text in (str(value.dtype), value.dtype.str),
            f"{context} manifest dtype differs: {key}",
        )
        if require_nbytes:
            _require(
                _json_exact(record.get("nbytes"), value.nbytes),
                f"{context} manifest byte count differs: {key}",
            )
        _require(
            record.get("sha256") == _array_sha256(value),
            f"{context} manifest digest differs: {key}",
        )


def _basename_matches(value: Any, path: Path) -> bool:
    return isinstance(value, str) and Path(value).name == path.name


def _validate_preregistration(
    document: Mapping[str, Any],
    *,
    reducer_sha256: str,
) -> None:
    required = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "phase1_hqc_canonical_fixed_prefix_preregistration",
        "status": "PREREGISTERED_CONDITIONAL_PREOUTPUT",
    }
    for key, value in required.items():
        _require(
            _json_exact(document.get(key), value),
            f"preregistration {key} differs",
        )
    contract_projection = dict(document)
    contract_projection.pop("implementation_freeze", None)
    projection_payload = json.dumps(
        contract_projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _require(
        hashlib.sha256(projection_payload).hexdigest()
        == EXPECTED_PREREGISTRATION_CONTRACT_SHA256,
        "preregistration scientific contract projection differs",
    )
    blindness = _mapping(document.get("blindness"), "preregistration blindness")
    for key in (
        "frozen_before_new_target_bundle_exists",
        "target_producer_may_not_read_predictions_metrics_or_thresholds",
        "reducer_is_sole_categorical_publisher",
        "reducer_output_is_publish_once",
        "all_valid_finite_outcomes_are_reported",
        "no_threshold_may_be_changed_after_endpoint exposure",
    ):
        _require(
            blindness.get(key) is True, f"preregistration blindness differs: {key}"
        )
    _require(
        blindness.get("new_k2500_or_k40000_truth_opened_or_scored_at_registration")
        is False,
        "preregistration endpoint-exposure declaration differs",
    )
    implementation = _mapping(
        document.get("implementation_freeze"),
        "preregistration implementation freeze",
    )
    _require(
        set(implementation)
        == {
            "reducer_path",
            "reducer_sha256",
            "reducer_test_path",
            "reducer_test_sha256",
            "target_producer_path",
            "target_producer_sha256",
            "target_producer_test_path",
            "target_producer_test_sha256",
            "exit_codes",
        },
        "preregistration implementation-freeze fields differ",
    )
    _require(
        implementation.get("reducer_path")
        == "studies/drivaerml_hqc_canonical_fixed_prefix_reduce.py"
        and implementation.get("reducer_sha256") == reducer_sha256
        and implementation.get("reducer_test_path")
        == "tests/test_drivaerml_hqc_canonical_fixed_prefix_reduce.py"
        and isinstance(implementation.get("reducer_test_sha256"), str)
        and SHA256_RE.fullmatch(implementation["reducer_test_sha256"]) is not None,
        "preregistered reducer identity differs",
    )
    _require(
        implementation.get("target_producer_path")
        == "studies/drivaerml_hqc_nested_target_input_freeze.py"
        and implementation.get("target_producer_sha256")
        == EXPECTED_TARGET_PRODUCER_SHA256
        and implementation.get("target_producer_test_path")
        == "tests/test_drivaerml_hqc_nested_target_input_freeze.py"
        and implementation.get("target_producer_test_sha256")
        == EXPECTED_TARGET_PRODUCER_TEST_SHA256,
        "preregistered target producer differs",
    )
    _require(
        _json_exact(
            implementation.get("exit_codes"),
            {"valid": 0, "incomplete": 3, "invalid": 4},
        ),
        "preregistered reducer exit-code contract differs",
    )


def _validate_activation(document: Mapping[str, Any]) -> None:
    required = {
        "schema_version": 1,
        "artifact_kind": ("drivaerml_historical_k10000_one_step_parity_adjudication"),
        "status": "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        "decision_outcome": "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS",
    }
    _require(
        set(document)
        == {
            "schema_version",
            "artifact_kind",
            "status",
            "decision_outcome",
            "created_at_utc",
            "validity",
            "decision_contract",
            "results",
            "limited_claim",
            "next_step",
        },
        "activation top-level fields differ",
    )
    for key, value in required.items():
        _require(_json_exact(document.get(key), value), f"activation {key} differs")
    validity = _mapping(document.get("validity"), "activation validity")
    _require(
        validity.get("producer_source_sha256") == EXPECTED_ONE_STEP_PRODUCER_SHA256,
        "activation producer identity differs",
    )
    for key in (
        "raw_array_manifest_verified",
        "shared_controls_verified",
        "parameter_partition_verified",
        "all_values_finite",
    ):
        _require(validity.get(key) is True, f"activation validity differs: {key}")
    for key in ("producer_json_sha256", "producer_npz_sha256"):
        value = validity.get(key)
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"activation {key} is malformed",
        )
    _require(
        _json_exact(
            document.get("decision_contract"),
            {
                "deciding_precision": "bfloat16",
                "all_case_regime_gates_required": True,
                "gradient_cosine_inclusive_minimum": 0.999,
                "update_cosine_inclusive_minimum": 0.9999,
                "update_symmetric_relative_l2_inclusive_maximum": 0.01,
                "gradient_path_fraction_of_between_case_median_inclusive_maximum": (
                    0.1
                ),
                "active_module_energy_fraction_inclusive_minimum": 0.01,
                "active_module_cosine_inclusive_minimum": 0.99,
                "fp32_role": "diagnostic_only",
            },
        ),
        "activation decision contract differs",
    )
    results = _mapping(document.get("results"), "activation results")
    _require(set(results) == set(PRECISIONS), "activation precision coverage differs")
    expected_cases = {"run_118", "run_271", "run_429", "run_86"}
    for precision, deciding in (("bfloat16", True), ("float32", False)):
        precision_result = _mapping(
            results[precision],
            f"activation {precision} result",
        )
        _require(
            precision_result.get("deciding") is deciding,
            f"activation {precision} role differs",
        )
        regimes = _mapping(
            precision_result.get("regimes"),
            f"activation {precision} regimes",
        )
        _require(
            set(regimes) == {"fresh_seed42", "checkpoint_epoch491"},
            f"activation {precision} regime coverage differs",
        )
        for regime_name, raw_regime in regimes.items():
            regime = _mapping(
                raw_regime,
                f"activation {precision}/{regime_name}",
            )
            cases = _mapping(
                regime.get("cases"),
                f"activation {precision}/{regime_name} cases",
            )
            _require(
                set(cases) == expected_cases,
                f"activation {precision}/{regime_name} case coverage differs",
            )
            if precision == "bfloat16":
                _require(
                    all(
                        isinstance(case, Mapping)
                        and case.get("deciding") is True
                        and case.get("passed") is True
                        for case in cases.values()
                    ),
                    f"activation {regime_name} contains a nonpassing BF16 case",
                )


def _validate_launch_manifest_static(
    document: Mapping[str, Any],
    *,
    launch_manifest_path: Path,
    target_json_path: Path,
    target_npz_path: Path,
    target_done_path: Path,
    output_json_path: Path,
) -> tuple[str, Path]:
    _require(
        set(document)
        == {
            "schema_version",
            "artifact_kind",
            "status",
            "attempt_id",
            "task_logical",
            "task_physical",
            "artifacts",
            "bindings",
        },
        "launch manifest top-level fields differ",
    )
    _require(
        _json_exact(document.get("schema_version"), 1)
        and document.get("artifact_kind") == LAUNCH_MANIFEST_KIND
        and document.get("status") == LAUNCH_MANIFEST_STATUS,
        "launch manifest envelope differs",
    )
    attempt_id = document.get("attempt_id")
    _require(
        isinstance(attempt_id, str) and ATTEMPT_ID_RE.fullmatch(attempt_id) is not None,
        "launch manifest attempt ID is malformed",
    )
    for key in ("task_logical", "task_physical"):
        value = document.get(key)
        _require(
            isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).name == attempt_id,
            f"launch manifest {key} differs from attempt ID",
        )
    logical_root = Path(str(document["task_logical"]))
    physical_root = Path(str(document["task_physical"]))
    try:
        logical_metadata = os.lstat(logical_root)
        physical_metadata = os.lstat(physical_root)
        logical_resolved = logical_root.resolve(strict=True)
        physical_resolved = physical_root.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise InvalidEvidence(
            "launch manifest task namespace is unavailable"
        ) from error
    _require(
        stat.S_ISDIR(logical_metadata.st_mode)
        and stat.S_ISDIR(physical_metadata.st_mode)
        and not stat.S_ISLNK(logical_metadata.st_mode)
        and not stat.S_ISLNK(physical_metadata.st_mode)
        and logical_resolved == physical_resolved
        and physical_resolved == physical_root,
        "launch manifest logical/physical namespace mapping differs",
    )
    _require(
        launch_manifest_path.name == "hqc_nested_target_freeze_launch_manifest_v1.json"
        and launch_manifest_path.parent.resolve(strict=False) == physical_root,
        "launch manifest file is outside its exact attempt-root location",
    )
    artifacts_root = physical_root / "artifacts"
    try:
        artifacts_metadata = os.lstat(artifacts_root)
    except (FileNotFoundError, OSError) as error:
        raise InvalidEvidence(
            "launch manifest artifacts directory is unavailable"
        ) from error
    _require(
        stat.S_ISDIR(artifacts_metadata.st_mode)
        and not stat.S_ISLNK(artifacts_metadata.st_mode)
        and artifacts_root.resolve(strict=True) == artifacts_root,
        "launch manifest artifacts directory is not a physical directory",
    )
    artifacts = _mapping(document.get("artifacts"), "launch manifest artifacts")
    expected_artifacts = {
        "target_json_relative_path": ("artifacts/hqc_nested_raw_target_bundle_v1.json"),
        "target_npz_relative_path": ("artifacts/hqc_nested_raw_target_bundle_v1.npz"),
        "target_done_pattern": "DONE_<slurm_job_id>.json",
        "reducer_output_relative_path": (
            "artifacts/hqc_canonical_fixed_prefix_adjudication_v1.json"
        ),
    }
    _require(
        _json_exact(artifacts, expected_artifacts),
        "launch manifest artifacts differ",
    )
    expected_paths = {
        "target JSON": physical_root / expected_artifacts["target_json_relative_path"],
        "target NPZ": physical_root / expected_artifacts["target_npz_relative_path"],
        "reducer output": (
            physical_root / expected_artifacts["reducer_output_relative_path"]
        ),
    }
    supplied_paths = {
        "target JSON": target_json_path,
        "target NPZ": target_npz_path,
        "reducer output": output_json_path,
    }
    for label, expected_path in expected_paths.items():
        _require(
            supplied_paths[label].resolve(strict=False)
            == expected_path.resolve(strict=False),
            f"launch manifest {label} path differs from attempt namespace",
        )
    allowed_artifact_names = {
        target_json_path.name,
        f"{target_json_path.name}.sha256",
        target_npz_path.name,
        f"{target_npz_path.name}.sha256",
        output_json_path.name,
        f"{output_json_path.name}.sha256",
    }
    try:
        observed_artifact_names = {entry.name for entry in os.scandir(artifacts_root)}
    except OSError as error:
        raise InvalidEvidence(
            "launch manifest artifacts directory cannot be inventoried"
        ) from error
    _require(
        observed_artifact_names <= allowed_artifact_names,
        "attempt artifacts directory contains an unrecognized entry",
    )
    _require(
        target_done_path.parent.resolve(strict=False) == physical_root
        and re.fullmatch(r"DONE_[0-9]+(?:_[0-9]+)?\.json", target_done_path.name)
        is not None,
        "launch manifest target DONE path differs from attempt namespace",
    )
    try:
        done_entries = [
            entry
            for entry in os.scandir(physical_root)
            if re.fullmatch(r"DONE_[0-9]+(?:_[0-9]+)?\.json", entry.name) is not None
        ]
    except OSError as error:
        raise InvalidEvidence("attempt root cannot be inventoried") from error
    _require(
        len(done_entries) <= 1,
        "attempt root contains multiple target DONE markers",
    )
    if done_entries:
        _require(
            done_entries[0].name == target_done_path.name
            and done_entries[0].is_file(follow_symlinks=False)
            and not done_entries[0].is_symlink(),
            "attempt root target DONE marker differs from supplied evidence",
        )
    bindings = _mapping(document.get("bindings"), "launch manifest bindings")
    expected_binding_keys = {
        "preregistration_sha256",
        "activation_adjudication_sha256",
        "wrapper_sha256",
        "target_producer_sha256",
        "target_producer_test_sha256",
        "target_wrapper_test_sha256",
        "reducer_sha256",
        "reducer_test_sha256",
        "geometry_manifest_sha256",
        "historical_target_manifest_sha256",
        "dataset_manifest_sha256",
        "cohort_sha256",
        "one_step_producer_sha256",
        "prediction_lane_json_sha256",
        "prediction_lane_npz_sha256",
        "canonical_k10000_json_sha256",
        "canonical_k10000_npz_sha256",
        "stage_b_k10000_json_sha256",
        "stage_b_k10000_npz_sha256",
        "accepted_adjudication_sha256",
    }
    _require(
        set(bindings) == expected_binding_keys,
        "launch manifest binding fields differ",
    )
    static_bindings: dict[str, Any] = {
        "target_producer_sha256": EXPECTED_TARGET_PRODUCER_SHA256,
        "target_producer_test_sha256": EXPECTED_TARGET_PRODUCER_TEST_SHA256,
        "geometry_manifest_sha256": EXPECTED_GEOMETRY_SHA256,
        "historical_target_manifest_sha256": EXPECTED_HISTORICAL_TARGET_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "one_step_producer_sha256": EXPECTED_ONE_STEP_PRODUCER_SHA256,
        "prediction_lane_json_sha256": list(EXPECTED_LANE_JSON_SHA256),
        "prediction_lane_npz_sha256": list(EXPECTED_LANE_NPZ_SHA256),
        "canonical_k10000_json_sha256": EXPECTED_CANONICAL_K10_JSON_SHA256,
        "canonical_k10000_npz_sha256": EXPECTED_CANONICAL_K10_NPZ_SHA256,
        "stage_b_k10000_json_sha256": EXPECTED_STAGE_B_K10_JSON_SHA256,
        "stage_b_k10000_npz_sha256": EXPECTED_STAGE_B_K10_NPZ_SHA256,
        "accepted_adjudication_sha256": EXPECTED_ACCEPTED_ADJUDICATION_SHA256,
    }
    for key, value in static_bindings.items():
        _require(bindings.get(key) == value, f"launch manifest {key} differs")
    for key in expected_binding_keys - set(static_bindings):
        value = bindings.get(key)
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"launch manifest {key} is malformed",
        )
    return attempt_id, physical_root


def _validate_launch_manifest(
    document: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    activation_sha256: str,
    wrapper_path: Path,
    target_producer_test_path: Path,
    target_wrapper_test_path: Path,
    reducer_test_path: Path,
    launch_manifest_path: Path,
    target_json_path: Path,
    target_npz_path: Path,
    target_done_path: Path,
    output_json_path: Path,
    reducer_sha256: str,
) -> tuple[str, str]:
    attempt_id, _ = _validate_launch_manifest_static(
        document,
        launch_manifest_path=launch_manifest_path,
        target_json_path=target_json_path,
        target_npz_path=target_npz_path,
        target_done_path=target_done_path,
        output_json_path=output_json_path,
    )
    implementation = _mapping(
        preregistration.get("implementation_freeze"),
        "preregistration implementation freeze",
    )
    bindings = _mapping(document.get("bindings"), "launch manifest bindings")
    expected_bindings = {
        "preregistration_sha256": preregistration_sha256,
        "activation_adjudication_sha256": activation_sha256,
        "wrapper_sha256": hashlib.sha256(_stable_read(wrapper_path)).hexdigest(),
        "target_producer_sha256": EXPECTED_TARGET_PRODUCER_SHA256,
        "target_producer_test_sha256": hashlib.sha256(
            _stable_read(target_producer_test_path)
        ).hexdigest(),
        "target_wrapper_test_sha256": hashlib.sha256(
            _stable_read(target_wrapper_test_path)
        ).hexdigest(),
        "reducer_sha256": reducer_sha256,
        "reducer_test_sha256": hashlib.sha256(
            _stable_read(reducer_test_path)
        ).hexdigest(),
        "geometry_manifest_sha256": EXPECTED_GEOMETRY_SHA256,
        "historical_target_manifest_sha256": EXPECTED_HISTORICAL_TARGET_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "one_step_producer_sha256": EXPECTED_ONE_STEP_PRODUCER_SHA256,
        "prediction_lane_json_sha256": list(EXPECTED_LANE_JSON_SHA256),
        "prediction_lane_npz_sha256": list(EXPECTED_LANE_NPZ_SHA256),
        "canonical_k10000_json_sha256": EXPECTED_CANONICAL_K10_JSON_SHA256,
        "canonical_k10000_npz_sha256": EXPECTED_CANONICAL_K10_NPZ_SHA256,
        "stage_b_k10000_json_sha256": EXPECTED_STAGE_B_K10_JSON_SHA256,
        "stage_b_k10000_npz_sha256": EXPECTED_STAGE_B_K10_NPZ_SHA256,
        "accepted_adjudication_sha256": EXPECTED_ACCEPTED_ADJUDICATION_SHA256,
    }
    _require(
        _json_exact(bindings, expected_bindings),
        "launch manifest bindings differ",
    )
    _require(
        implementation.get("reducer_sha256") == reducer_sha256
        and implementation.get("reducer_test_sha256")
        == expected_bindings["reducer_test_sha256"],
        "launch manifest reducer implementation differs from preregistration",
    )
    _require(
        implementation.get("target_producer_sha256") == EXPECTED_TARGET_PRODUCER_SHA256
        and implementation.get("target_producer_test_sha256")
        == expected_bindings["target_producer_test_sha256"],
        "launch manifest target-producer implementation differs from preregistration",
    )
    return attempt_id, str(expected_bindings["wrapper_sha256"])


def _audit_launch_manifest_available_bindings(
    document: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any] | None,
    preregistration_sha256: str | None,
    activation_sha256: str | None,
    reducer_sha256: str,
    wrapper_path: Path,
    target_producer_test_path: Path,
    target_wrapper_test_path: Path,
    reducer_test_path: Path,
) -> None:
    """Check every manifest binding whose companion is presently available."""
    bindings = _mapping(document.get("bindings"), "launch manifest bindings")
    if preregistration is not None:
        implementation = _mapping(
            preregistration.get("implementation_freeze"),
            "preregistration implementation freeze",
        )
        for key in (
            "reducer_sha256",
            "reducer_test_sha256",
            "target_producer_sha256",
            "target_producer_test_sha256",
        ):
            _require(
                bindings.get(key) == implementation.get(key),
                f"launch manifest/preregistration {key} binding differs",
            )
    available: dict[str, str] = {"reducer_sha256": reducer_sha256}
    if preregistration_sha256 is not None:
        available["preregistration_sha256"] = preregistration_sha256
    if activation_sha256 is not None:
        available["activation_adjudication_sha256"] = activation_sha256
    for key, path in (
        ("wrapper_sha256", wrapper_path),
        ("target_producer_test_sha256", target_producer_test_path),
        ("target_wrapper_test_sha256", target_wrapper_test_path),
        ("reducer_test_sha256", reducer_test_path),
    ):
        if _regular_entry(path):
            try:
                payload = _stable_read(path)
            except IncompleteEvidence:
                continue
            available[key] = hashlib.sha256(payload).hexdigest()
    for key, value in available.items():
        _require(bindings.get(key) == value, f"launch manifest {key} differs")


def _lane_array_key(
    ordinal: int,
    case_id: str,
    resolution: int,
    suffix: str,
) -> str:
    return f"case_{ordinal:02d}_{case_id}__k{resolution:05d}__{suffix}"


def _case_array_key(ordinal: int, case_id: str, suffix: str) -> str:
    return f"case_{ordinal:02d}_{case_id}__{suffix}"


def _validate_prediction_lanes(
    json_paths: Sequence[Path],
    npz_paths: Sequence[Path],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, str]]]:
    _require(
        len(json_paths) == len(npz_paths) == 4,
        "exactly four prediction lanes are required",
    )
    retained: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, str]] = []
    for lane in range(4):
        json_payload, json_sha256 = _verified_payload(
            json_paths[lane],
            expected_sha256=EXPECTED_LANE_JSON_SHA256[lane],
        )
        npz_payload, npz_sha256 = _verified_payload(
            npz_paths[lane],
            expected_sha256=EXPECTED_LANE_NPZ_SHA256[lane],
        )
        document = _strict_json(json_payload, context=f"prediction lane {lane}")
        required = {
            "schema_version": 1,
            "artifact_kind": "hqc_canonical_geometry_full_cohort_validity_lane",
            "status": "VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_LANE",
            "decision_outcome": "CANONICAL_FULL_VALIDITY_PASS",
        }
        for key, value in required.items():
            _require(
                _json_exact(document.get(key), value),
                f"prediction lane {lane} {key} differs",
            )
        _require(
            _json_exact(document.get("lane"), {"ordinal": lane, "count": 4}),
            f"prediction lane {lane} assignment differs",
        )
        validity = _mapping(
            document.get("validity"),
            f"prediction lane {lane} validity",
        )
        _require(
            validity.get("all_cases_resolutions_and_precisions_passed") is True,
            f"prediction lane {lane} is not wholly valid",
        )
        geometry_verification = _mapping(
            validity.get("geometry_input_manifest_lane_verification"),
            f"prediction lane {lane} geometry verification",
        )
        _require(
            geometry_verification.get("manifest_sha256") == EXPECTED_GEOMETRY_SHA256,
            f"prediction lane {lane} geometry identity differs",
        )
        arrays = _npz_arrays(npz_payload, context=f"prediction lane {lane} NPZ")
        _validate_array_manifest(
            document.get("npz_array_manifest"),
            arrays,
            context=f"prediction lane {lane}",
        )
        expected_specs = [spec for spec in CASE_SPECS if spec[0] % 4 == lane]
        cases = _sequence(document.get("cases"), f"prediction lane {lane} cases")
        _require(
            len(cases) == len(expected_specs),
            f"prediction lane {lane} case count differs",
        )
        expected_array_names: set[str] = set()
        for raw_case, spec in zip(cases, expected_specs, strict=True):
            ordinal, case_id, reader_index, _, _ = spec
            case = _mapping(raw_case, f"prediction lane {lane} case {ordinal}")
            _require(
                _json_exact(case.get("cohort_ordinal"), ordinal)
                and case.get("case_id") == case_id
                and _json_exact(case.get("reader_index"), reader_index)
                and case.get("validity_passed") is True
                and case.get("decision_outcome") == "CANONICAL_FULL_VALIDITY_PASS",
                f"prediction lane {lane} case {ordinal} envelope differs",
            )
            resolution_records = _sequence(
                case.get("resolutions"),
                f"prediction lane {lane} case {ordinal} resolutions",
            )
            _require(
                [record.get("resolution") for record in resolution_records]
                == list(RESOLUTIONS)
                and all(
                    isinstance(record, Mapping)
                    and _json_exact(record.get("cohort_ordinal"), ordinal)
                    and record.get("case_id") == case_id
                    and record.get("validity_passed") is True
                    and record.get("decision_outcome") == "CANONICAL_FULL_VALIDITY_PASS"
                    for record in resolution_records
                ),
                f"prediction lane {lane} case {ordinal} resolution envelope differs",
            )
            retained_case: dict[str, Any] = {
                "ids": {},
                "areas": {},
                "predictions": {},
                "geometry_k10000": {},
            }
            max_ids: np.ndarray | None = None
            max_areas: np.ndarray | None = None
            for resolution in RESOLUTIONS:
                geometry_contract = {
                    "selected_cell_ids_int64": (np.dtype("<i8"), (resolution,)),
                    "canonical_cells_int64": (
                        np.dtype("<i8"),
                        (resolution, 3),
                    ),
                    "canonical_centroids_float32": (
                        np.dtype("<f4"),
                        (resolution, 3),
                    ),
                    "canonical_areas_float32": (
                        np.dtype("<f4"),
                        (resolution,),
                    ),
                    "canonical_normals_float32": (
                        np.dtype("<f4"),
                        (resolution, 3),
                    ),
                }
                current_geometry: dict[str, np.ndarray] = {}
                for suffix, (dtype, shape) in geometry_contract.items():
                    key = _lane_array_key(ordinal, case_id, resolution, suffix)
                    expected_array_names.add(key)
                    current_geometry[suffix] = _typed_array(
                        arrays,
                        key,
                        dtype=dtype,
                        shape=shape,
                    )
                point_key = _lane_array_key(
                    ordinal,
                    case_id,
                    resolution,
                    "canonical_points_float32",
                )
                expected_array_names.add(point_key)
                points = arrays.get(point_key)
                _require(
                    isinstance(points, np.ndarray)
                    and points.dtype == np.dtype("<f4")
                    and points.ndim == 2
                    and points.shape[1:] == (3,)
                    and points.flags.c_contiguous
                    and np.isfinite(points).all(),
                    f"prediction lane point contract differs: {point_key}",
                )
                cells = current_geometry["canonical_cells_int64"]
                _require(
                    bool(np.all(cells >= 0))
                    and (cells.size == 0 or int(cells.max()) < len(points)),
                    f"prediction lane topology differs: {point_key}",
                )
                areas = current_geometry["canonical_areas_float32"]
                _require(
                    bool(np.all(areas > 0.0)),
                    f"prediction lane areas are nonpositive: {point_key}",
                )
                ids = current_geometry["selected_cell_ids_int64"]
                _require(
                    bool(np.all(ids >= 0)) and len(np.unique(ids)) == len(ids),
                    (
                        "prediction lane selected IDs are negative or duplicate: "
                        f"{ordinal}/{resolution}"
                    ),
                )
                if resolution == max(RESOLUTIONS):
                    max_ids = ids
                    max_areas = areas
                retained_case["ids"][resolution] = ids.copy()
                retained_case["areas"][resolution] = areas.copy()
                for precision in PRECISIONS:
                    for panel, query_count in (
                        ("coupled_s_k", resolution),
                        ("fixed_id_prefix_s2500", FIXED_Q),
                    ):
                        panel_values: dict[str, dict[str, np.ndarray]] = {}
                        for copy_name in ("primary", "fixed", "primary_replay"):
                            copy_values: dict[str, np.ndarray] = {}
                            for field, shape in (
                                ("pressure", (query_count,)),
                                ("wss", (query_count, 3)),
                            ):
                                suffix = (
                                    f"{precision}_canonical_full_{panel}_"
                                    f"{copy_name}_{field}"
                                )
                                key = _lane_array_key(
                                    ordinal,
                                    case_id,
                                    resolution,
                                    suffix,
                                )
                                expected_array_names.add(key)
                                copy_values[field] = _typed_array(
                                    arrays,
                                    key,
                                    dtype=np.dtype("<f4"),
                                    shape=shape,
                                )
                            panel_values[copy_name] = copy_values
                        for field in ("pressure", "wss"):
                            primary = panel_values["primary"][field]
                            _require(
                                _same_bytes(primary, panel_values["fixed"][field])
                                and _same_bytes(
                                    primary,
                                    panel_values["primary_replay"][field],
                                ),
                                (
                                    "prediction lane copy exactness differs: "
                                    f"{ordinal}/{resolution}/{precision}/{panel}/{field}"
                                ),
                            )
                            retained_case["predictions"][
                                (precision, panel, resolution, field)
                            ] = primary.copy()
                    coupled = panel_values  # keep mypy from widening below
                    del coupled
                    for field in ("pressure", "wss"):
                        coupled_primary = retained_case["predictions"][
                            (precision, "coupled_s_k", resolution, field)
                        ]
                        fixed_primary = retained_case["predictions"][
                            (
                                precision,
                                "fixed_id_prefix_s2500",
                                resolution,
                                field,
                            )
                        ]
                        _require(
                            _same_bytes(coupled_primary[:FIXED_Q], fixed_primary),
                            (
                                "prediction lane fixed-prefix contract differs: "
                                f"{ordinal}/{resolution}/{precision}/{field}"
                            ),
                        )
                if resolution == BASELINE_K:
                    retained_case["geometry_k10000"] = {
                        **{
                            suffix: value.copy()
                            for suffix, value in current_geometry.items()
                        },
                        "canonical_points_float32": points.copy(),
                    }
            if max_ids is None:
                raise InvalidEvidence(
                    f"prediction lane maximum-resolution IDs are absent: {ordinal}"
                )
            if max_areas is None:
                raise InvalidEvidence(
                    f"prediction lane maximum-resolution areas are absent: {ordinal}"
                )
            for resolution in RESOLUTIONS:
                _require(
                    _same_bytes(
                        retained_case["ids"][resolution],
                        max_ids[:resolution],
                    ),
                    f"prediction lane selected IDs are not nested: {ordinal}/{resolution}",
                )
                _require(
                    _same_bytes(
                        retained_case["areas"][resolution],
                        max_areas[:resolution],
                    ),
                    (
                        "prediction lane canonical areas are not nested: "
                        f"{ordinal}/{resolution}"
                    ),
                )
            retained[ordinal] = retained_case
        _require(
            set(arrays) == expected_array_names,
            f"prediction lane {lane} NPZ member coverage differs",
        )
        provenance.append(
            {
                "json_sha256": json_sha256,
                "npz_sha256": npz_sha256,
            }
        )
    _require(set(retained) == set(range(36)), "prediction cohort coverage differs")
    return retained, provenance


def _float32_truth(
    raw_pressure: np.ndarray,
    raw_wss: np.ndarray,
    physical_globals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = physical_globals[:3]
    p_inf = physical_globals[3]
    rho_inf = physical_globals[4]
    dynamic_pressure = (
        np.float32(0.5)
        * rho_inf
        * np.sum(
            velocity * velocity,
            dtype=np.float32,
        )
    )
    if not np.isfinite(dynamic_pressure) or dynamic_pressure <= 0.0:
        raise InvalidEvidence("dynamic pressure is nonpositive or non-finite")
    pressure = np.asarray(
        (raw_pressure - p_inf) / dynamic_pressure,
        dtype="<f4",
    )
    wss_scale = np.float32(0.00313) + np.float32(1.0e-8)
    wss = np.asarray((raw_wss / dynamic_pressure) / wss_scale, dtype="<f4")
    if not np.isfinite(pressure).all() or not np.isfinite(wss).all():
        raise InvalidEvidence("reconstructed training truth is non-finite")
    return pressure, wss


def _validate_target_done(
    document: Mapping[str, Any],
    *,
    done_path: Path,
    activation_sha256: str,
    target_json_sha256: str,
    target_npz_sha256: str,
    preregistration_sha256: str,
    attempt_id: str,
    launch_manifest_sha256: str,
    wrapper_sha256: str,
) -> None:
    _require(
        set(document)
        == {
            "artifact_kind",
            "activation_adjudication_sha256",
            "attempt_id",
            "job_id",
            "json_sha256",
            "launch_manifest_sha256",
            "npz_sha256",
            "preregistration_sha256",
            "producer_sha256",
            "reducer_schema_validation_performed",
            "schema_version",
            "status",
            "wrapper_sha256",
        },
        "target DONE fields differ",
    )
    expected = {
        "schema_version": 1,
        "artifact_kind": "drivaerml_hqc_nested_target_bundle_commit",
        "reducer_schema_validation_performed": False,
        "status": "CONTENT_COMMITTED_UNVALIDATED_HQC_NESTED_TARGET_BUNDLE",
        "activation_adjudication_sha256": activation_sha256,
        "attempt_id": attempt_id,
        "json_sha256": target_json_sha256,
        "launch_manifest_sha256": launch_manifest_sha256,
        "npz_sha256": target_npz_sha256,
        "preregistration_sha256": preregistration_sha256,
        "producer_sha256": EXPECTED_TARGET_PRODUCER_SHA256,
        "wrapper_sha256": wrapper_sha256,
    }
    for key, value in expected.items():
        _require(
            _json_exact(document.get(key), value),
            f"target DONE {key} differs",
        )
    job_id = document.get("job_id")
    _require(
        isinstance(job_id, str)
        and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is not None
        and done_path.name == f"DONE_{job_id}.json",
        "target DONE job ID or filename is malformed",
    )


def _target_array_contract() -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    result: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    for ordinal, case_id, *_ in CASE_SPECS:
        prefix = f"case_{ordinal:02d}_{case_id}__"
        result.update(
            {
                f"{prefix}selected_cell_ids_int64": (
                    np.dtype("<i8"),
                    (40_000,),
                ),
                f"{prefix}physical_globals_float32": (
                    np.dtype("<f4"),
                    (7,),
                ),
                f"{prefix}raw_target_pressure_float32": (
                    np.dtype("<f4"),
                    (40_000,),
                ),
                f"{prefix}raw_target_wss_float32": (
                    np.dtype("<f4"),
                    (40_000, 3),
                ),
            }
        )
    return result


def _validate_target_arrays_static(arrays: Mapping[str, np.ndarray]) -> None:
    contract = _target_array_contract()
    _require(set(arrays) == set(contract), "target bundle NPZ member names differ")
    for key, (dtype, shape) in contract.items():
        value = _typed_array(arrays, key, dtype=dtype, shape=shape)
        if key.endswith("selected_cell_ids_int64"):
            ordinal = int(key.split("_", 2)[1])
            n_master_cells = CASE_SPECS[ordinal][3]
            _require(
                bool(np.all(value >= 0))
                and bool(np.all(value < n_master_cells))
                and len(np.unique(value)) == len(value),
                f"target bundle selected IDs are invalid: {key}",
            )
    for ordinal, case_id, *_ in CASE_SPECS:
        prefix = f"case_{ordinal:02d}_{case_id}__"
        _, _, _, n_master_cells, historical_start = CASE_SPECS[ordinal]
        expected_ids = (
            historical_start + np.arange(max(RESOLUTIONS), dtype="<i8")
        ) % n_master_cells
        _require(
            _same_bytes(
                arrays[f"{prefix}selected_cell_ids_int64"],
                expected_ids,
            ),
            f"target bundle cyclic selected IDs differ: {case_id}",
        )
        _float32_truth(
            arrays[f"{prefix}raw_target_pressure_float32"],
            arrays[f"{prefix}raw_target_wss_float32"],
            arrays[f"{prefix}physical_globals_float32"],
        )


def _validate_target_document_static(
    document: Mapping[str, Any],
    *,
    npz_path: Path,
) -> None:
    _require(
        set(document)
        == {
            "schema_version",
            "artifact_kind",
            "status",
            "generated_at_utc",
            "dataset_root_input",
            "dataset_root_resolved",
            "dataset_manifest_sha256",
            "geometry_manifest",
            "historical_k10000_target_manifest",
            "case_count",
            "resolutions",
            "max_resolution",
            "fixed_query_resolution",
            "physical_globals",
            "selection",
            "read_allowlist",
            "read_exclusions",
            "publication_contract",
            "cases",
            "cohort_sha256",
            "npz",
            "array_manifest",
            "provenance",
        },
        "target bundle top-level fields differ",
    )
    required = {
        "schema_version": 1,
        "artifact_kind": "drivaerml_hqc_nested_raw_target_bundle",
        "status": "PASSED_HQC_NESTED_RAW_TARGET_FREEZE",
        "case_count": 36,
        "resolutions": list(RESOLUTIONS),
        "max_resolution": max(RESOLUTIONS),
        "fixed_query_resolution": FIXED_Q,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
        "selection": (
            "one Kmax ordered cyclic panel; every smaller S_k and fixed Q are "
            "exact array prefixes"
        ),
    }
    for key, value in required.items():
        _require(
            _json_exact(document.get(key), value),
            f"target bundle {key} differs",
        )
    _require(
        isinstance(document.get("generated_at_utc"), str)
        and bool(document["generated_at_utc"])
        and isinstance(document.get("dataset_root_input"), str)
        and isinstance(document.get("dataset_root_resolved"), str),
        "target bundle path or timestamp fields are malformed",
    )
    geometry = _mapping(
        document.get("geometry_manifest"),
        "target bundle geometry manifest",
    )
    historical = _mapping(
        document.get("historical_k10000_target_manifest"),
        "target bundle historical target manifest",
    )
    _require(
        set(geometry) == {"path", "sha256"}
        and _basename_matches(
            geometry.get("path"),
            Path("drivaerml_geometry_input_manifest_36cases_v1.json"),
        )
        and Path(str(geometry.get("path"))).is_absolute()
        and geometry.get("sha256") == EXPECTED_GEOMETRY_SHA256,
        "target bundle geometry identity differs",
    )
    _require(
        set(historical) == {"path", "sha256", "prefix_hashes_authenticated"}
        and _basename_matches(
            historical.get("path"),
            Path("historical_k10000_selected_target_input_manifest_v1.json"),
        )
        and Path(str(historical.get("path"))).is_absolute()
        and historical.get("sha256") == EXPECTED_HISTORICAL_TARGET_SHA256
        and type(historical.get("prefix_hashes_authenticated")) is int
        and historical.get("prefix_hashes_authenticated") == 72,
        "target bundle historical target identity differs",
    )
    _require(
        _json_exact(
            document.get("physical_globals"),
            {
                "array_suffix": "physical_globals_float32",
                "field_order": [
                    "U_inf_x",
                    "U_inf_y",
                    "U_inf_z",
                    "p_inf",
                    "rho_inf",
                    "nu",
                    "L_ref",
                ],
                "dtype": "float32_little_endian",
                "source": "frozen target-free geometry manifest",
                "transformed_by_target_freezer": False,
            },
        ),
        "target bundle physical-global contract differs",
    )
    _require(
        _json_exact(
            document.get("read_allowlist"),
            [
                "dataset manifest.json",
                "frozen target-free geometry manifest",
                "frozen historical K=10k target manifest",
                "vehicle cell_data/meta.json",
                "vehicle cell_data/pMeanTrim.memmap selected byte spans only",
                (
                    "vehicle cell_data/wallShearStressMeanTrim.memmap selected "
                    "byte spans only"
                ),
            ],
        ),
        "target bundle read allowlist differs",
    )
    _require(
        _json_exact(
            document.get("read_exclusions"),
            {
                "model_opened": False,
                "prediction_opened": False,
                "metric_opened": False,
                "decision_threshold_opened": False,
                "other_cell_data_opened": False,
                "point_data_opened": False,
                "interior_opened": False,
            },
        ),
        "target bundle blind-read exclusions differ",
    )
    _require(
        _json_exact(
            document.get("publication_contract"),
            {
                "json_manifest_linked_last": True,
                "producer_outputs_are_not_a_commit_marker": True,
                "valid_only_after_external_sidecar_checks_and_done_marker": True,
                "interrupted_partial_bundle_must_not_be_overwritten": True,
            },
        ),
        "target bundle publication contract differs",
    )
    npz_record = _mapping(document.get("npz"), "target bundle NPZ record")
    _require(
        set(npz_record) == {"path", "sha256", "array_count"}
        and _basename_matches(npz_record.get("path"), npz_path)
        and _json_exact(npz_record.get("array_count"), 144)
        and isinstance(npz_record.get("sha256"), str)
        and SHA256_RE.fullmatch(npz_record["sha256"]) is not None,
        "target bundle internal NPZ record differs",
    )
    provenance = _mapping(document.get("provenance"), "target bundle provenance")
    _require(
        set(provenance) == {"command", "script_path", "script_sha256", "numpy"}
        and provenance.get("script_sha256") == EXPECTED_TARGET_PRODUCER_SHA256
        and isinstance(provenance.get("command"), list)
        and isinstance(provenance.get("script_path"), str)
        and isinstance(provenance.get("numpy"), str),
        "target bundle producer provenance differs",
    )
    manifest = _mapping(document.get("array_manifest"), "target array manifest")
    contract = _target_array_contract()
    _require(set(manifest) == set(contract), "target array manifest names differ")
    for key, (dtype, shape) in contract.items():
        record = _mapping(manifest[key], f"target array manifest {key}")
        _require(
            set(record) == {"shape", "dtype", "nbytes", "sha256"}
            and _json_exact(record.get("shape"), list(shape))
            and record.get("dtype") == dtype.str
            and _json_exact(
                record.get("nbytes"),
                int(np.prod(shape)) * dtype.itemsize,
            )
            and isinstance(record.get("sha256"), str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None,
            f"target array manifest contract differs: {key}",
        )
    cases = _sequence(document.get("cases"), "target bundle cases")
    _require(len(cases) == 36, "target bundle case count differs")
    expected_hash_keys = {str(value) for value in RESOLUTIONS}
    for raw_case, spec in zip(cases, CASE_SPECS, strict=True):
        ordinal, case_id, reader_index, n_master_cells, historical_start = spec
        case = _mapping(raw_case, f"target bundle case {ordinal}")
        _require(
            set(case)
            == {
                "cohort_ordinal",
                "case_id",
                "reader_index",
                "n_master_cells",
                "historical_start",
                "max_resolution",
                "selection",
                "logical_case_symlink",
                "symlink_target",
                "resolved_case_root",
                "cell_data_metadata",
                "targets",
            }
            and _json_exact(case.get("cohort_ordinal"), ordinal)
            and case.get("case_id") == case_id
            and _json_exact(case.get("reader_index"), reader_index)
            and _json_exact(case.get("n_master_cells"), n_master_cells)
            and _json_exact(case.get("historical_start"), historical_start)
            and _json_exact(case.get("max_resolution"), max(RESOLUTIONS)),
            f"target bundle case {ordinal} identity or fields differ",
        )
        selection = _mapping(
            case.get("selection"),
            f"target bundle case {ordinal} selection",
        )
        selection_hashes = _mapping(
            selection.get("selected_cell_ids_sha256_by_resolution"),
            f"target bundle case {ordinal} selected-ID hashes",
        )
        _require(
            selection.get("kind")
            == "ordered_cyclic_prefix_from_historical_k10000_start"
            and set(selection)
            == {
                "kind",
                "wraps",
                "selected_cell_ids_sha256_by_resolution",
            }
            and selection.get("wraps")
            is (historical_start + max(RESOLUTIONS) > n_master_cells)
            and set(selection_hashes) == expected_hash_keys
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in selection_hashes.values()
            ),
            f"target bundle case {ordinal} selection contract differs",
        )
        ids_key = f"case_{ordinal:02d}_{case_id}__selected_cell_ids_int64"
        _require(
            selection_hashes[str(max(RESOLUTIONS))] == manifest[ids_key]["sha256"],
            f"target bundle case {ordinal} selected-ID manifest binding differs",
        )
        metadata = _mapping(
            case.get("cell_data_metadata"),
            f"target bundle case {ordinal} metadata",
        )
        _require(
            set(metadata) == {"relative_path", "size_bytes", "sha256"}
            and metadata.get("relative_path")
            == (
                f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                "_tensordict/cell_data/meta.json"
            )
            and type(metadata.get("size_bytes")) is int
            and metadata["size_bytes"] == EXPECTED_METADATA_SIZE_BYTES[ordinal]
            and metadata.get("sha256") == EXPECTED_METADATA_SHA256[ordinal]
            and all(
                isinstance(case.get(key), str)
                for key in (
                    "logical_case_symlink",
                    "symlink_target",
                    "resolved_case_root",
                )
            ),
            f"target bundle case {ordinal} path/metadata contract differs",
        )
        targets = _mapping(
            case.get("targets"),
            f"target bundle case {ordinal} targets",
        )
        _require(
            set(targets) == {"pressure", "wss"},
            f"target bundle case {ordinal} target coverage differs",
        )
        for field, raw_field_name, components, shape in (
            ("pressure", "pMeanTrim", 1, [40_000]),
            ("wss", "wallShearStressMeanTrim", 3, [40_000, 3]),
        ):
            record = _mapping(
                targets[field],
                f"target bundle case {ordinal} {field}",
            )
            hashes = _mapping(
                record.get("prefix_sha256_by_resolution"),
                f"target bundle case {ordinal} {field} hashes",
            )
            row_bytes = components * np.dtype("<f4").itemsize
            tail_rows = min(
                max(RESOLUTIONS),
                n_master_cells - historical_start,
            )
            head_rows = max(RESOLUTIONS) - tail_rows
            expected_spans = [
                {
                    "offset": historical_start * row_bytes,
                    "count": tail_rows * row_bytes,
                }
            ]
            if head_rows:
                expected_spans.append(
                    {
                        "offset": 0,
                        "count": head_rows * row_bytes,
                    }
                )
            _require(
                set(record)
                == {
                    "raw_field_name",
                    "source_relative_path",
                    "source_size_bytes",
                    "source_spans_bytes",
                    "selected_shape",
                    "selected_dtype",
                    "selected_sha256",
                    "prefix_sha256_by_resolution",
                    "historical_k10000_prefix_authenticated",
                }
                and record.get("raw_field_name") == raw_field_name
                and record.get("source_relative_path")
                == (
                    f"domain_{case_id}.pdmsh/_tensordict/boundaries/vehicle/"
                    f"_tensordict/cell_data/{raw_field_name}.memmap"
                )
                and _json_exact(
                    record.get("source_size_bytes"),
                    n_master_cells * row_bytes,
                )
                and _json_exact(record.get("source_spans_bytes"), expected_spans)
                and sum(span["count"] for span in expected_spans)
                == max(RESOLUTIONS) * row_bytes
                and _json_exact(record.get("selected_shape"), shape)
                and record.get("selected_dtype") == "float32_little_endian"
                and record.get("historical_k10000_prefix_authenticated") is True
                and isinstance(record.get("selected_sha256"), str)
                and SHA256_RE.fullmatch(record["selected_sha256"]) is not None
                and set(hashes) == expected_hash_keys
                and all(
                    isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                    for value in hashes.values()
                ),
                f"target bundle case {ordinal} {field} contract differs",
            )
            target_key = f"case_{ordinal:02d}_{case_id}__raw_target_{field}_float32"
            _require(
                record["selected_sha256"]
                == hashes[str(max(RESOLUTIONS))]
                == manifest[target_key]["sha256"],
                f"target bundle case {ordinal} {field} manifest binding differs",
            )


def _validate_target_bundle(
    *,
    json_path: Path,
    npz_path: Path,
    done_path: Path,
    activation_sha256: str,
    preregistration_sha256: str,
    attempt_id: str,
    launch_manifest_sha256: str,
    wrapper_sha256: str,
    predictions: Mapping[int, Mapping[str, Any]] | None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str]]:
    json_payload, json_sha256 = _verified_payload(json_path)
    npz_payload, npz_sha256 = _verified_payload(npz_path)
    done_payload = _stable_read(done_path)
    document = _strict_json(json_payload, context="blind target bundle")
    done = _strict_json(done_payload, context="blind target DONE")
    _validate_target_document_static(document, npz_path=npz_path)
    required = {
        "schema_version": 1,
        "artifact_kind": "drivaerml_hqc_nested_raw_target_bundle",
        "status": "PASSED_HQC_NESTED_RAW_TARGET_FREEZE",
        "case_count": 36,
        "resolutions": list(RESOLUTIONS),
        "max_resolution": max(RESOLUTIONS),
        "fixed_query_resolution": FIXED_Q,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
    }
    for key, value in required.items():
        _require(
            _json_exact(document.get(key), value),
            f"target bundle {key} differs",
        )
    geometry = _mapping(
        document.get("geometry_manifest"),
        "target bundle geometry manifest",
    )
    historical = _mapping(
        document.get("historical_k10000_target_manifest"),
        "target bundle historical target manifest",
    )
    _require(
        geometry.get("sha256") == EXPECTED_GEOMETRY_SHA256,
        "target bundle geometry identity differs",
    )
    _require(
        historical.get("sha256") == EXPECTED_HISTORICAL_TARGET_SHA256
        and historical.get("prefix_hashes_authenticated") == 72,
        "target bundle historical target identity differs",
    )
    npz_record = _mapping(document.get("npz"), "target bundle NPZ record")
    _require(
        _basename_matches(npz_record.get("path"), npz_path)
        and npz_record.get("sha256") == npz_sha256
        and _json_exact(npz_record.get("array_count"), 144),
        "target bundle internal NPZ identity differs",
    )
    provenance = _mapping(document.get("provenance"), "target bundle provenance")
    _require(
        provenance.get("script_sha256") == EXPECTED_TARGET_PRODUCER_SHA256,
        "target bundle producer identity differs",
    )
    _require(
        _json_exact(
            document.get("publication_contract"),
            {
                "json_manifest_linked_last": True,
                "producer_outputs_are_not_a_commit_marker": True,
                "valid_only_after_external_sidecar_checks_and_done_marker": True,
                "interrupted_partial_bundle_must_not_be_overwritten": True,
            },
        ),
        "target bundle publication contract differs",
    )
    _require(
        _json_exact(
            document.get("read_exclusions"),
            {
                "model_opened": False,
                "prediction_opened": False,
                "metric_opened": False,
                "decision_threshold_opened": False,
                "other_cell_data_opened": False,
                "point_data_opened": False,
                "interior_opened": False,
            },
        ),
        "target bundle blind-read exclusions differ",
    )
    globals_record = _mapping(
        document.get("physical_globals"),
        "target bundle physical globals",
    )
    _require(
        globals_record.get("field_order")
        == [
            "U_inf_x",
            "U_inf_y",
            "U_inf_z",
            "p_inf",
            "rho_inf",
            "nu",
            "L_ref",
        ]
        and globals_record.get("array_suffix") == "physical_globals_float32"
        and globals_record.get("dtype") == "float32_little_endian"
        and globals_record.get("transformed_by_target_freezer") is False,
        "target bundle physical-global contract differs",
    )
    arrays = _npz_arrays(npz_payload, context="blind target bundle NPZ")
    _validate_target_arrays_static(arrays)
    _validate_array_manifest(
        document.get("array_manifest"),
        arrays,
        context="blind target bundle",
        require_nbytes=True,
    )
    cases = _sequence(document.get("cases"), "blind target bundle cases")
    _require(len(cases) == 36, "blind target bundle case count differs")
    retained: dict[int, dict[str, np.ndarray]] = {}
    expected_array_names: set[str] = set()
    for raw_case, spec in zip(cases, CASE_SPECS, strict=True):
        ordinal, case_id, reader_index, n_master_cells, historical_start = spec
        case = _mapping(raw_case, f"target bundle case {ordinal}")
        _require(
            _json_exact(case.get("cohort_ordinal"), ordinal)
            and case.get("case_id") == case_id
            and _json_exact(case.get("reader_index"), reader_index)
            and _json_exact(case.get("n_master_cells"), n_master_cells)
            and _json_exact(case.get("historical_start"), historical_start)
            and _json_exact(case.get("max_resolution"), max(RESOLUTIONS)),
            f"target bundle case {ordinal} identity differs",
        )
        prefix = f"case_{ordinal:02d}_{case_id}__"
        array_contract = {
            "selected_cell_ids_int64": (np.dtype("<i8"), (40_000,)),
            "physical_globals_float32": (np.dtype("<f4"), (7,)),
            "raw_target_pressure_float32": (np.dtype("<f4"), (40_000,)),
            "raw_target_wss_float32": (np.dtype("<f4"), (40_000, 3)),
        }
        case_arrays: dict[str, np.ndarray] = {}
        for suffix, (dtype, shape) in array_contract.items():
            key = f"{prefix}{suffix}"
            expected_array_names.add(key)
            case_arrays[suffix] = _typed_array(
                arrays,
                key,
                dtype=dtype,
                shape=shape,
            )
        ids = case_arrays["selected_cell_ids_int64"]
        _require(
            bool(np.all(ids >= 0))
            and len(np.unique(ids)) == len(ids)
            and bool(np.all(ids < n_master_cells)),
            f"target bundle selected IDs are invalid: {case_id}",
        )
        selection = _mapping(
            case.get("selection"),
            f"target bundle case {ordinal} selection",
        )
        selection_hashes = _mapping(
            selection.get("selected_cell_ids_sha256_by_resolution"),
            f"target bundle case {ordinal} selection hashes",
        )
        _require(
            set(selection_hashes) == {str(value) for value in RESOLUTIONS},
            f"target bundle case {ordinal} selection hash coverage differs",
        )
        targets = _mapping(
            case.get("targets"),
            f"target bundle case {ordinal} targets",
        )
        _require(
            set(targets) == {"pressure", "wss"},
            f"target bundle case {ordinal} target coverage differs",
        )
        for resolution in RESOLUTIONS:
            _require(
                selection_hashes[str(resolution)] == _array_sha256(ids[:resolution]),
                (
                    "target bundle selected-ID prefix hash differs: "
                    f"{case_id}/{resolution}"
                ),
            )
            if predictions is not None:
                _require(
                    _same_bytes(
                        ids[:resolution],
                        predictions[ordinal]["ids"][resolution],
                    ),
                    (
                        "target/prediction selected-ID join differs: "
                        f"{case_id}/{resolution}"
                    ),
                )
        for field, array_suffix in (
            ("pressure", "raw_target_pressure_float32"),
            ("wss", "raw_target_wss_float32"),
        ):
            target_record = _mapping(
                targets[field],
                f"target bundle case {ordinal} {field}",
            )
            hashes = _mapping(
                target_record.get("prefix_sha256_by_resolution"),
                f"target bundle case {ordinal} {field} hashes",
            )
            _require(
                target_record.get("historical_k10000_prefix_authenticated") is True
                and set(hashes) == {str(value) for value in RESOLUTIONS},
                f"target bundle case {ordinal} {field} prefix contract differs",
            )
            value = case_arrays[array_suffix]
            for resolution in RESOLUTIONS:
                _require(
                    hashes[str(resolution)] == _array_sha256(value[:resolution]),
                    (
                        f"target bundle {field} prefix hash differs: "
                        f"{case_id}/{resolution}"
                    ),
                )
        pressure, wss = _float32_truth(
            case_arrays["raw_target_pressure_float32"],
            case_arrays["raw_target_wss_float32"],
            case_arrays["physical_globals_float32"],
        )
        retained[ordinal] = {
            **{name: value.copy() for name, value in case_arrays.items()},
            "truth_pressure_float32": pressure,
            "truth_wss_float32": wss,
        }
    _require(
        set(arrays) == expected_array_names,
        "target bundle NPZ member coverage differs",
    )
    _validate_target_done(
        done,
        done_path=done_path,
        activation_sha256=activation_sha256,
        target_json_sha256=json_sha256,
        target_npz_sha256=npz_sha256,
        preregistration_sha256=preregistration_sha256,
        attempt_id=attempt_id,
        launch_manifest_sha256=launch_manifest_sha256,
        wrapper_sha256=wrapper_sha256,
    )
    return retained, {
        "json_sha256": json_sha256,
        "npz_sha256": npz_sha256,
        "done_sha256": hashlib.sha256(done_payload).hexdigest(),
    }


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    if (
        prediction64.shape != truth64.shape
        or not np.isfinite(prediction64).all()
        or not np.isfinite(truth64).all()
    ):
        raise InvalidEvidence("uniform relative-L2 inputs are invalid")
    value = float(np.linalg.norm(prediction64 - truth64)) / (
        float(np.linalg.norm(truth64)) + 1.0e-8
    )
    if not math.isfinite(value):
        raise InvalidEvidence("uniform relative L2 is non-finite")
    return value


def _weighted_relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    areas: np.ndarray,
) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    area64 = np.asarray(areas, dtype=np.float64)
    if (
        prediction64.shape != truth64.shape
        or area64.shape != prediction64.shape[:1]
        or not np.isfinite(prediction64).all()
        or not np.isfinite(truth64).all()
        or not np.isfinite(area64).all()
        or not np.all(area64 > 0.0)
    ):
        raise InvalidEvidence("canonical-area relative-L2 inputs are invalid")
    weights = area64 / np.sum(area64, dtype=np.float64)
    expanded = weights[:, None] if prediction64.ndim == 2 else weights
    numerator = math.sqrt(
        float(
            np.sum(
                expanded * (prediction64 - truth64) ** 2,
                dtype=np.float64,
            )
        )
    )
    denominator = (
        math.sqrt(float(np.sum(expanded * truth64**2, dtype=np.float64))) + 1.0e-8
    )
    value = numerator / denominator
    if not math.isfinite(value):
        raise InvalidEvidence("canonical-area relative L2 is non-finite")
    return value


def _validate_anchor_lane(
    *,
    document: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    npz_path: Path,
    npz_sha256: str,
    canonical: bool,
) -> None:
    if canonical:
        expected_kind = "phase1_historical_k10000_canonical_arm_producer"
        expected_status = "COMPLETED_HISTORICAL_K10000_CANONICAL_ARM_PRODUCER"
        label_key = "lane_label"
        expected_array_count = 792
    else:
        expected_kind = "phase1_historical_k10000_replay_producer"
        expected_status = "COMPLETED_HISTORICAL_K10000_REPLAY_PRODUCER"
        label_key = "replay_label"
        expected_array_count = 720
    _require(
        document.get("schema_version") == 1
        and document.get("artifact_kind") == expected_kind
        and document.get("status") == expected_status
        and document.get(label_key) == "A",
        f"K=10000 {'canonical' if canonical else 'Stage-B'} envelope differs",
    )
    summary = _mapping(
        document.get("summary"),
        f"K=10000 {'canonical' if canonical else 'Stage-B'} summary",
    )
    _require(
        summary.get("case_count") == 36
        and summary.get("array_count") == expected_array_count,
        f"K=10000 {'canonical' if canonical else 'Stage-B'} summary differs",
    )
    npz_record = _mapping(
        document.get("npz"),
        f"K=10000 {'canonical' if canonical else 'Stage-B'} NPZ record",
    )
    _require(
        _basename_matches(npz_record.get("filename"), npz_path)
        and npz_record.get("sha256") == npz_sha256
        and npz_record.get("array_count") == expected_array_count,
        f"K=10000 {'canonical' if canonical else 'Stage-B'} NPZ identity differs",
    )
    _validate_array_manifest(
        npz_record.get("array_manifest"),
        arrays,
        context=f"K=10000 {'canonical' if canonical else 'Stage-B'}",
    )
    cases = _sequence(
        document.get("cases"),
        f"K=10000 {'canonical' if canonical else 'Stage-B'} cases",
    )
    _require(len(cases) == 36, "K=10000 anchor case count differs")
    for raw_case, spec in zip(cases, CASE_SPECS, strict=True):
        ordinal, case_id, reader_index, _, historical_start = spec
        case = _mapping(raw_case, f"K=10000 anchor case {ordinal}")
        _require(
            case.get("cohort_ordinal") == ordinal
            and case.get("case_id") == case_id
            and case.get("reader_index") == reader_index
            and case.get("historical_start") == historical_start
            and case.get("resolution") == BASELINE_K,
            f"K=10000 anchor case {ordinal} identity differs",
        )


def _validate_k10000_anchor(
    *,
    canonical_json_path: Path,
    canonical_npz_path: Path,
    stage_b_json_path: Path,
    stage_b_npz_path: Path,
    adjudication_path: Path,
    predictions: Mapping[int, Mapping[str, Any]],
    targets: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    canonical_json_payload, canonical_json_sha256 = _verified_payload(
        canonical_json_path,
        expected_sha256=EXPECTED_CANONICAL_K10_JSON_SHA256,
    )
    canonical_npz_payload, canonical_npz_sha256 = _verified_payload(
        canonical_npz_path,
        expected_sha256=EXPECTED_CANONICAL_K10_NPZ_SHA256,
    )
    stage_b_json_payload, stage_b_json_sha256 = _verified_payload(
        stage_b_json_path,
        expected_sha256=EXPECTED_STAGE_B_K10_JSON_SHA256,
    )
    stage_b_npz_payload, stage_b_npz_sha256 = _verified_payload(
        stage_b_npz_path,
        expected_sha256=EXPECTED_STAGE_B_K10_NPZ_SHA256,
    )
    adjudication_payload, adjudication_sha256 = _verified_payload(
        adjudication_path,
        expected_sha256=EXPECTED_ACCEPTED_ADJUDICATION_SHA256,
    )
    canonical_document = _strict_json(
        canonical_json_payload,
        context="accepted K=10000 canonical lane",
    )
    stage_b_document = _strict_json(
        stage_b_json_payload,
        context="accepted K=10000 Stage-B lane",
    )
    adjudication = _strict_json(
        adjudication_payload,
        context="accepted K=10000 adjudication",
    )
    canonical_arrays = _npz_arrays(
        canonical_npz_payload,
        context="accepted K=10000 canonical NPZ",
    )
    stage_b_arrays = _npz_arrays(
        stage_b_npz_payload,
        context="accepted K=10000 Stage-B NPZ",
    )
    _validate_anchor_lane(
        document=canonical_document,
        arrays=canonical_arrays,
        npz_path=canonical_npz_path,
        npz_sha256=canonical_npz_sha256,
        canonical=True,
    )
    _validate_anchor_lane(
        document=stage_b_document,
        arrays=stage_b_arrays,
        npz_path=stage_b_npz_path,
        npz_sha256=stage_b_npz_sha256,
        canonical=False,
    )
    _require(
        adjudication.get("schema_version") == 1
        and adjudication.get("artifact_kind")
        == "phase1_historical_k10000_paired_accuracy_adjudication"
        and adjudication.get("status") == "VALID_CANONICAL_NONINFERIORITY_ADJUDICATION"
        and adjudication.get("decision_outcome") == "CANONICAL_NONINFERIORITY_PASS"
        and adjudication.get("failures") == [],
        "accepted K=10000 adjudication envelope differs",
    )
    accepted_provenance = _mapping(
        adjudication.get("provenance"),
        "accepted K=10000 adjudication provenance",
    )
    _require(
        accepted_provenance.get("canonical_a_json_sha256") == canonical_json_sha256
        and accepted_provenance.get("canonical_a_npz_sha256") == canonical_npz_sha256
        and accepted_provenance.get("sealed_stage_b_a_json_sha256")
        == stage_b_json_sha256
        and accepted_provenance.get("sealed_stage_b_a_npz_sha256")
        == stage_b_npz_sha256,
        "accepted K=10000 adjudication input bindings differ",
    )
    accepted_cases = _sequence(
        adjudication.get("cases"),
        "accepted K=10000 adjudication cases",
    )
    _require(len(accepted_cases) == 36, "accepted K=10000 case count differs")
    shared_suffixes = (
        "selected_cell_ids_int64",
        "compacted_cells_int64",
        "raw_centroids_float32",
        "native_normals_float32",
        "native_areas_float64",
        "pipeline_boundary_points_float32",
        "pipeline_queries_float32",
        "pipeline_normals_float32",
        "pipeline_globals_float32",
        "pipeline_center_float32",
    )
    lane_to_canonical_suffixes = {
        "selected_cell_ids_int64": "selected_cell_ids_int64",
        "canonical_cells_int64": "canonical_cells_int64",
        "canonical_points_float32": "canonical_points_float32",
        "canonical_centroids_float32": "canonical_centroids_float32",
        "canonical_areas_float32": "canonical_areas_float32",
        "canonical_normals_float32": "canonical_normals_float32",
    }
    uniform_pressure: list[float] = []
    uniform_wss: list[float] = []
    area_pressure: list[float] = []
    area_wss: list[float] = []
    for raw_accepted_case, spec in zip(
        accepted_cases,
        CASE_SPECS,
        strict=True,
    ):
        ordinal, case_id, _, _, _ = spec
        prefix = f"case_{ordinal:02d}_{case_id}__"
        accepted_case = _mapping(
            raw_accepted_case,
            f"accepted K=10000 adjudication case {ordinal}",
        )
        _require(
            accepted_case.get("cohort_ordinal") == ordinal
            and accepted_case.get("case_id") == case_id,
            f"accepted K=10000 adjudication case {ordinal} differs",
        )
        for suffix in shared_suffixes:
            _require(
                _same_bytes(
                    canonical_arrays[f"{prefix}{suffix}"],
                    stage_b_arrays[f"{prefix}{suffix}"],
                ),
                f"K=10000 canonical/Stage-B shared control differs: {case_id}/{suffix}",
            )
        lane_geometry = predictions[ordinal]["geometry_k10000"]
        for lane_suffix, canonical_suffix in lane_to_canonical_suffixes.items():
            _require(
                _same_bytes(
                    lane_geometry[lane_suffix],
                    canonical_arrays[f"{prefix}{canonical_suffix}"],
                ),
                f"K=10000 target-free/canonical anchor differs: {case_id}/{lane_suffix}",
            )
        for field, canonical_suffix in (
            ("pressure", "prediction_pressure_training_float32"),
            ("wss", "prediction_wss_training_float32"),
        ):
            lane_prediction = predictions[ordinal]["predictions"][
                ("bfloat16", "coupled_s_k", BASELINE_K, field)
            ]
            _require(
                _same_bytes(
                    lane_prediction,
                    canonical_arrays[f"{prefix}{canonical_suffix}"],
                ),
                f"K=10000 target-free/canonical prediction differs: {case_id}/{field}",
            )
        target = targets[ordinal]
        _require(
            _same_bytes(
                target["selected_cell_ids_int64"][:BASELINE_K],
                stage_b_arrays[f"{prefix}selected_cell_ids_int64"],
            )
            and _same_bytes(
                target["physical_globals_float32"],
                stage_b_arrays[f"{prefix}pipeline_globals_float32"][:7],
            ),
            f"K=10000 target/Stage-B join differs: {case_id}",
        )
        for field in ("pressure", "wss"):
            raw_suffix = f"raw_target_{field}_float32"
            truth_suffix = f"truth_{field}_training_float32"
            _require(
                _same_bytes(
                    target[raw_suffix][:BASELINE_K],
                    stage_b_arrays[f"{prefix}{raw_suffix}"],
                ),
                f"K=10000 raw target differs: {case_id}/{field}",
            )
            _require(
                _same_bytes(
                    target[f"truth_{field}_float32"][:BASELINE_K],
                    stage_b_arrays[f"{prefix}{truth_suffix}"],
                ),
                f"K=10000 reconstructed truth differs: {case_id}/{field}",
            )
        prediction_pressure = predictions[ordinal]["predictions"][
            ("bfloat16", "coupled_s_k", BASELINE_K, "pressure")
        ]
        prediction_wss = predictions[ordinal]["predictions"][
            ("bfloat16", "coupled_s_k", BASELINE_K, "wss")
        ]
        truth_pressure = target["truth_pressure_float32"][:BASELINE_K]
        truth_wss = target["truth_wss_float32"][:BASELINE_K]
        areas = predictions[ordinal]["areas"][BASELINE_K]
        pressure_value = _relative_l2(prediction_pressure, truth_pressure)
        wss_value = _relative_l2(prediction_wss, truth_wss)
        area_pressure_value = _weighted_relative_l2(
            prediction_pressure,
            truth_pressure,
            areas,
        )
        area_wss_value = _weighted_relative_l2(
            prediction_wss,
            truth_wss,
            areas,
        )
        accepted_metrics = _mapping(
            accepted_case.get("canonical_metrics"),
            f"accepted K=10000 metrics {case_id}",
        )
        _require(
            abs(
                pressure_value - float(accepted_metrics["uniform_pressure_relative_l2"])
            )
            <= K10_METRIC_ATOL
            and abs(
                wss_value - float(accepted_metrics["uniform_wss_frobenius_relative_l2"])
            )
            <= K10_METRIC_ATOL,
            f"K=10000 accepted uniform metric differs: {case_id}",
        )
        uniform_pressure.append(pressure_value)
        uniform_wss.append(wss_value)
        area_pressure.append(area_pressure_value)
        area_wss.append(area_wss_value)
    canonical_arm = _mapping(
        adjudication.get("canonical_arm"),
        "accepted K=10000 canonical arm",
    )
    accepted_means = _mapping(
        canonical_arm.get("cohort_means"),
        "accepted K=10000 cohort means",
    )
    uniform_pressure_mean = float(np.mean(uniform_pressure, dtype=np.float64))
    uniform_wss_mean = float(np.mean(uniform_wss, dtype=np.float64))
    area_pressure_mean = float(np.mean(area_pressure, dtype=np.float64))
    area_wss_mean = float(np.mean(area_wss, dtype=np.float64))
    _require(
        abs(
            uniform_pressure_mean
            - float(accepted_means["uniform_pressure_relative_l2"])
        )
        <= K10_METRIC_ATOL
        and abs(
            uniform_wss_mean
            - float(accepted_means["uniform_wss_frobenius_relative_l2"])
        )
        <= K10_METRIC_ATOL,
        "K=10000 accepted uniform cohort mean differs",
    )
    _require(
        abs(area_pressure_mean - CANONICAL_AREA_K10_PRESSURE_MEAN)
        <= CANONICAL_AREA_K10_MEAN_ATOL
        and abs(area_wss_mean - CANONICAL_AREA_K10_WSS_MEAN)
        <= CANONICAL_AREA_K10_MEAN_ATOL,
        "K=10000 canonical-area rehearsal mean differs",
    )
    return {
        "exact_target_free_to_canonical_arrays": 36 * 8,
        "exact_canonical_to_stage_b_shared_control_arrays": 36 * 10,
        "exact_target_to_stage_b_id_and_global_arrays": 36 * 2,
        "exact_raw_target_arrays": 36 * 2,
        "exact_reconstructed_training_truth_arrays": 36 * 2,
        "uniform_metrics_compared": 36 * 2,
        "uniform_pressure_cohort_mean": uniform_pressure_mean,
        "uniform_wss_cohort_mean": uniform_wss_mean,
        "canonical_area_pressure_cohort_mean": area_pressure_mean,
        "canonical_area_wss_cohort_mean": area_wss_mean,
        "canonical_json_sha256": canonical_json_sha256,
        "canonical_npz_sha256": canonical_npz_sha256,
        "stage_b_json_sha256": stage_b_json_sha256,
        "stage_b_npz_sha256": stage_b_npz_sha256,
        "accepted_adjudication_sha256": adjudication_sha256,
    }


def _median(values: Sequence[float]) -> float:
    if len(values) != 36:
        raise InvalidEvidence(f"cohort statistic has {len(values)} values, not 36")
    value = float(statistics.median(values))
    if not math.isfinite(value):
        raise InvalidEvidence("cohort statistic is non-finite")
    return value


def _metric_panel(
    errors: Mapping[str, Mapping[int, Sequence[float]]],
    *,
    common_eligible: bool,
    common_checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen pressure gates to one weighting panel."""
    _require(set(errors) == {"coupled", "fixed"}, "metric panel arms differ")
    for arm in ("coupled", "fixed"):
        _require(
            set(errors[arm]) == set(RESOLUTIONS),
            f"metric panel {arm} resolution coverage differs",
        )
        for resolution in RESOLUTIONS:
            values = list(errors[arm][resolution])
            _require(
                len(values) == 36
                and all(math.isfinite(value) and value >= 0.0 for value in values),
                f"metric panel {arm}/{resolution} values are invalid",
            )

    nonpositive_locations = [
        f"{arm}/{resolution}/{index}"
        for arm in ("coupled", "fixed")
        for resolution in RESOLUTIONS
        for index, value in enumerate(errors[arm][resolution])
        if value <= 0.0
    ]
    baseline_ratio_median: float | None = None
    endpoint_rows: dict[str, Any] = {}
    baseline_comparable = False
    coupled_cliff_passed = False
    if not nonpositive_locations:
        baseline_ratios = [
            fixed / coupled
            for fixed, coupled in zip(
                errors["fixed"][BASELINE_K],
                errors["coupled"][BASELINE_K],
                strict=True,
            )
        ]
        baseline_ratio_median = _median(baseline_ratios)
        baseline_comparable = (
            BASELINE_BOUNDS[0] <= baseline_ratio_median <= BASELINE_BOUNDS[1]
        )
        for endpoint in ENDPOINTS:
            coupled_ratios = [
                endpoint_error / baseline_error
                for endpoint_error, baseline_error in zip(
                    errors["coupled"][endpoint],
                    errors["coupled"][BASELINE_K],
                    strict=True,
                )
            ]
            fixed_ratios = [
                endpoint_error / baseline_error
                for endpoint_error, baseline_error in zip(
                    errors["fixed"][endpoint],
                    errors["fixed"][BASELINE_K],
                    strict=True,
                )
            ]
            coupled_logs = [math.log(value) for value in coupled_ratios]
            fixed_logs = [math.log(value) for value in fixed_ratios]
            coupled_median_log = _median(coupled_logs)
            fixed_median_log = _median(fixed_logs)
            coupled_positive = max(0.0, coupled_median_log)
            fixed_positive = max(0.0, fixed_median_log)
            fraction = (
                fixed_positive / coupled_positive if coupled_positive > 0.0 else None
            )
            coupled_count = sum(value >= CLIFF_RATIO_MIN for value in coupled_ratios)
            favorable_count = sum(
                fixed < coupled
                for fixed, coupled in zip(
                    fixed_logs,
                    coupled_logs,
                    strict=True,
                )
            )
            cliff = (
                coupled_median_log >= CLIFF_LOG_MIN and coupled_count >= CLIFF_COUNT_MIN
            )
            support = (
                fraction is not None
                and fraction <= SUPPORT_FRACTION_MAX
                and fixed_median_log <= SUPPORT_FIXED_LOG_MAX
                and favorable_count >= SUPPORT_FAVORABLE_MIN
            )
            fraction_futility = (
                fraction is not None and fraction >= FUTILITY_FRACTION_MIN
            )
            fixed_median_ratio = _median(fixed_ratios)
            k40_futility = (
                endpoint == 40_000 and fixed_median_ratio >= FUTILITY_K40_RATIO_MIN
            )
            endpoint_rows[str(endpoint)] = {
                "coupled_median_log_error_ratio": coupled_median_log,
                "coupled_median_error_ratio": _median(coupled_ratios),
                "coupled_error_ratio_at_least_2_case_count": coupled_count,
                "fixed_median_log_error_ratio": fixed_median_log,
                "fixed_median_error_ratio": fixed_median_ratio,
                "fixed_positive_log_fraction_of_coupled": fraction,
                "paired_fixed_log_less_than_coupled_case_count": favorable_count,
                "eligibility_coupled_cliff_passed": cliff,
                "support_passed": support,
                "futility_fraction_triggered": fraction_futility,
                "futility_k40000_ratio_triggered": k40_futility,
                "futility_triggered": fraction_futility or k40_futility,
            }
        coupled_cliff_passed = all(
            endpoint_rows[str(endpoint)]["eligibility_coupled_cliff_passed"]
            for endpoint in ENDPOINTS
        )

    eligible = (
        common_eligible
        and not nonpositive_locations
        and baseline_comparable
        and coupled_cliff_passed
    )
    any_futility = eligible and any(
        endpoint_rows[str(endpoint)]["futility_triggered"] for endpoint in ENDPOINTS
    )
    all_support = eligible and all(
        endpoint_rows[str(endpoint)]["support_passed"] for endpoint in ENDPOINTS
    )
    if not eligible:
        classification = INELIGIBLE_OUTCOME
    elif any_futility:
        classification = FUTILE_OUTCOME
    elif all_support:
        classification = "SUPPORTED"
    else:
        classification = MIXED_OUTCOME
    return {
        "common_checks": dict(common_checks or {}),
        "nonpositive_metric_locations": nonpositive_locations,
        "baseline_fixed_over_coupled_median_error_ratio": baseline_ratio_median,
        "baseline_comparability_passed": baseline_comparable,
        "both_endpoint_coupled_cliffs_passed": coupled_cliff_passed,
        "eligible": eligible,
        "endpoints": endpoint_rows,
        "any_futility_triggered": bool(any_futility),
        "both_endpoint_support_passed": bool(all_support),
        "classification": classification,
    }


def _validate_k2500_metric_identity(
    coupled: Mapping[str, float],
    fixed: Mapping[str, float],
) -> None:
    _require(
        coupled == fixed,
        "K=2500 coupled and fixed-prefix metrics are not exactly identical",
    )


def _compute_panel(
    *,
    predictions: Mapping[int, Mapping[str, Any]],
    targets: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    errors: dict[str, dict[str, dict[int, list[float]]]] = {
        "uniform": {
            "coupled": {resolution: [] for resolution in RESOLUTIONS},
            "fixed": {resolution: [] for resolution in RESOLUTIONS},
        },
        "canonical_area": {
            "coupled": {resolution: [] for resolution in RESOLUTIONS},
            "fixed": {resolution: [] for resolution in RESOLUTIONS},
        },
    }
    diagnostic_errors: dict[str, Any] = {
        precision: {
            weighting: {
                field: {
                    arm: {resolution: [] for resolution in RESOLUTIONS}
                    for arm in ("coupled", "fixed")
                }
                for field in ("pressure", "wss")
            }
            for weighting in ("uniform", "canonical_area")
        }
        for precision in PRECISIONS
    }
    case_rows: list[dict[str, Any]] = []
    truth_rms_ratios: list[float | None] = []
    mean_area_ratios: list[float] = []
    for spec in CASE_SPECS:
        ordinal, case_id, _, _, _ = spec
        target = targets[ordinal]
        truth_pressure_max = target["truth_pressure_float32"]
        truth_wss_max = target["truth_wss_float32"]
        truth_q = np.asarray(truth_pressure_max[:FIXED_Q], dtype=np.float64)
        truth_k10 = np.asarray(
            truth_pressure_max[:BASELINE_K],
            dtype=np.float64,
        )
        truth_q_rms = math.sqrt(float(np.mean(truth_q**2, dtype=np.float64)))
        truth_k10_rms = math.sqrt(float(np.mean(truth_k10**2, dtype=np.float64)))
        truth_rms_ratio = truth_q_rms / truth_k10_rms if truth_k10_rms > 0.0 else None
        areas_k10 = np.asarray(
            predictions[ordinal]["areas"][BASELINE_K],
            dtype=np.float64,
        )
        mean_area_ratio = float(
            np.mean(areas_k10[:FIXED_Q], dtype=np.float64)
            / np.mean(areas_k10, dtype=np.float64)
        )
        truth_rms_ratios.append(truth_rms_ratio)
        mean_area_ratios.append(mean_area_ratio)
        case_metrics: dict[str, Any] = {}
        for precision in PRECISIONS:
            precision_rows: dict[str, Any] = {}
            for resolution in RESOLUTIONS:
                resolution_rows: dict[str, Any] = {}
                for arm, panel_name, query_count in (
                    ("coupled", "coupled_s_k", resolution),
                    ("fixed", "fixed_id_prefix_s2500", FIXED_Q),
                ):
                    truth_pressure = truth_pressure_max[:query_count]
                    truth_wss = truth_wss_max[:query_count]
                    areas = predictions[ordinal]["areas"][resolution][:query_count]
                    prediction_pressure = predictions[ordinal]["predictions"][
                        (precision, panel_name, resolution, "pressure")
                    ]
                    prediction_wss = predictions[ordinal]["predictions"][
                        (precision, panel_name, resolution, "wss")
                    ]
                    values = {
                        "uniform_pressure_relative_l2": _relative_l2(
                            prediction_pressure,
                            truth_pressure,
                        ),
                        "uniform_wss_frobenius_relative_l2": _relative_l2(
                            prediction_wss,
                            truth_wss,
                        ),
                        "canonical_area_pressure_relative_l2": (
                            _weighted_relative_l2(
                                prediction_pressure,
                                truth_pressure,
                                areas,
                            )
                        ),
                        "canonical_area_wss_frobenius_relative_l2": (
                            _weighted_relative_l2(
                                prediction_wss,
                                truth_wss,
                                areas,
                            )
                        ),
                    }
                    resolution_rows[arm] = values
                    for weighting, field, metric_key in (
                        (
                            "uniform",
                            "pressure",
                            "uniform_pressure_relative_l2",
                        ),
                        (
                            "uniform",
                            "wss",
                            "uniform_wss_frobenius_relative_l2",
                        ),
                        (
                            "canonical_area",
                            "pressure",
                            "canonical_area_pressure_relative_l2",
                        ),
                        (
                            "canonical_area",
                            "wss",
                            "canonical_area_wss_frobenius_relative_l2",
                        ),
                    ):
                        diagnostic_errors[precision][weighting][field][arm][
                            resolution
                        ].append(values[metric_key])
                    if precision == "bfloat16":
                        errors["uniform"][arm][resolution].append(
                            values["uniform_pressure_relative_l2"]
                        )
                        errors["canonical_area"][arm][resolution].append(
                            values["canonical_area_pressure_relative_l2"]
                        )
                precision_rows[str(resolution)] = resolution_rows
                if resolution == FIXED_Q:
                    _validate_k2500_metric_identity(
                        resolution_rows["coupled"],
                        resolution_rows["fixed"],
                    )
            case_metrics[precision] = precision_rows
        case_rows.append(
            {
                "cohort_ordinal": ordinal,
                "case_id": case_id,
                "truth_q2500_over_s10000_rms_ratio": truth_rms_ratio,
                "mean_canonical_area_q2500_over_s10000_ratio": mean_area_ratio,
                "metrics": case_metrics,
            }
        )
    truth_rms_passed = all(
        value is not None and math.isfinite(value) and 0.5 <= value <= 2.0
        for value in truth_rms_ratios
    )
    mean_area_passed = all(
        math.isfinite(value) and 0.5 <= value <= 2.0 for value in mean_area_ratios
    )
    common_checks = {
        "truth_q2500_over_s10000_rms_all_inclusive_0p5_to_2": (truth_rms_passed),
        "mean_canonical_area_q2500_over_s10000_all_inclusive_0p5_to_2": (
            mean_area_passed
        ),
        "truth_rms_ratio_minimum": (
            min(value for value in truth_rms_ratios if value is not None)
            if any(value is not None for value in truth_rms_ratios)
            else None
        ),
        "truth_rms_ratio_maximum": (
            max(value for value in truth_rms_ratios if value is not None)
            if any(value is not None for value in truth_rms_ratios)
            else None
        ),
        "mean_area_ratio_minimum": (
            min(mean_area_ratios) if mean_area_ratios else None
        ),
        "mean_area_ratio_maximum": (
            max(mean_area_ratios) if mean_area_ratios else None
        ),
    }
    common_eligible = truth_rms_passed and mean_area_passed
    uniform_panel = _metric_panel(
        errors["uniform"],
        common_eligible=common_eligible,
        common_checks=common_checks,
    )
    area_panel = _metric_panel(
        errors["canonical_area"],
        common_eligible=common_eligible,
        common_checks=common_checks,
    )
    if uniform_panel["classification"] == "SUPPORTED":
        if area_panel["classification"] == "SUPPORTED":
            outcome = DUAL_OUTCOME
        else:
            area_nearly_flat = all(
                area_panel["endpoints"]
                .get(str(endpoint), {})
                .get(
                    "coupled_median_error_ratio",
                    math.inf,
                )
                <= AREA_FLAT_RATIO_MAX
                for endpoint in ENDPOINTS
            )
            outcome = AREA_FLAT_OUTCOME if area_nearly_flat else UNIFORM_ONLY_OUTCOME
    else:
        outcome = str(uniform_panel["classification"])
    return {
        "decision_outcome": outcome,
        "common_eligibility": common_checks,
        "uniform_bfloat16_pressure_panel": uniform_panel,
        "canonical_area_bfloat16_pressure_panel": area_panel,
        "case_metrics": case_rows,
        "ordered_diagnostics": {
            "float32_and_wss_are_nondeciding": True,
            "errors": diagnostic_errors,
        },
    }


def _artifact_ready(path: Path, *, sidecar: bool = True) -> bool:
    paths = [path]
    if sidecar:
        paths.append(path.with_name(f"{path.name}.sha256"))
    for candidate in paths:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise InvalidEvidence(
                f"could not inspect artifact readiness safely: {candidate}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise InvalidEvidence(
                f"artifact is not a regular non-symlink file: {candidate}"
            )
    return True


def _entry_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise InvalidEvidence(f"could not inspect artifact safely: {path}") from error
    return True


def _regular_entry(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise InvalidEvidence(f"could not inspect artifact safely: {path}") from error
    return not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(metadata.st_mode)


def _audit_present_target_envelope(path: Path, *, npz_path: Path) -> None:
    if not _regular_entry(path):
        return
    if _artifact_ready(path):
        payload, _ = _verified_payload(path)
    else:
        payload = _stable_read(path)
    document = _strict_json(payload, context="present blind target bundle")
    _validate_target_document_static(document, npz_path=npz_path)


def _audit_present_target_npz(path: Path) -> None:
    if not _regular_entry(path):
        return
    if _artifact_ready(path):
        payload, _ = _verified_payload(path)
    else:
        payload = _stable_read(path)
    arrays = _npz_arrays(payload, context="present blind target NPZ")
    _validate_target_arrays_static(arrays)


def _audit_present_target_done(path: Path) -> None:
    if not _regular_entry(path):
        return
    document = _strict_json(_stable_read(path), context="present blind target DONE")
    _require(
        set(document)
        == {
            "artifact_kind",
            "activation_adjudication_sha256",
            "attempt_id",
            "job_id",
            "json_sha256",
            "launch_manifest_sha256",
            "npz_sha256",
            "preregistration_sha256",
            "producer_sha256",
            "reducer_schema_validation_performed",
            "schema_version",
            "status",
            "wrapper_sha256",
        },
        "present target DONE fields differ",
    )
    _require(
        _json_exact(document.get("schema_version"), 1)
        and document.get("artifact_kind") == "drivaerml_hqc_nested_target_bundle_commit"
        and document.get("status")
        == "CONTENT_COMMITTED_UNVALIDATED_HQC_NESTED_TARGET_BUNDLE"
        and document.get("producer_sha256") == EXPECTED_TARGET_PRODUCER_SHA256
        and document.get("reducer_schema_validation_performed") is False,
        "present target DONE envelope differs",
    )
    for key in (
        "activation_adjudication_sha256",
        "json_sha256",
        "launch_manifest_sha256",
        "npz_sha256",
        "preregistration_sha256",
        "wrapper_sha256",
    ):
        value = document.get(key)
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"present target DONE {key} is malformed",
        )
    job_id = document.get("job_id")
    _require(
        isinstance(job_id, str)
        and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is not None
        and path.name == f"DONE_{job_id}.json",
        "present target DONE job ID or filename is malformed",
    )
    attempt_id = document.get("attempt_id")
    _require(
        isinstance(attempt_id, str) and ATTEMPT_ID_RE.fullmatch(attempt_id) is not None,
        "present target DONE attempt ID is malformed",
    )


def _audit_present_target_bindings(
    *,
    json_path: Path,
    npz_path: Path,
    done_path: Path,
    activation_path: Path,
    preregistration_path: Path,
    launch_manifest_path: Path,
    wrapper_path: Path,
) -> None:
    observed: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    incomplete: list[str] = []
    for key, path in (
        ("json_sha256", json_path),
        ("npz_sha256", npz_path),
        ("activation_adjudication_sha256", activation_path),
        ("preregistration_sha256", preregistration_path),
        ("launch_manifest_sha256", launch_manifest_path),
        ("wrapper_sha256", wrapper_path),
    ):
        if _regular_entry(path):
            try:
                payload = _stable_read(path)
            except IncompleteEvidence as error:
                incomplete.append(str(error))
                continue
            payloads[key] = payload
            observed[key] = hashlib.sha256(payload).hexdigest()
    if "json_sha256" in payloads and "npz_sha256" in observed:
        document = _strict_json(
            payloads["json_sha256"],
            context="present blind target bundle binding",
        )
        npz_record = _mapping(
            document.get("npz"),
            "present blind target NPZ binding",
        )
        _require(
            npz_record.get("sha256") == observed["npz_sha256"],
            "present target JSON/NPZ binding differs",
        )
    if _regular_entry(done_path):
        try:
            done_payload = _stable_read(done_path)
        except IncompleteEvidence as error:
            incomplete.append(str(error))
        else:
            done = _strict_json(
                done_payload,
                context="present blind target DONE binding",
            )
            for key, digest in observed.items():
                _require(
                    done.get(key) == digest,
                    f"present target DONE {key} binding differs",
                )
    if incomplete:
        raise IncompleteEvidence("; ".join(incomplete))


def _target_records(
    args: argparse.Namespace,
) -> list[tuple[Path, str | None, bool]]:
    return [
        (args.target_json, None, True),
        (args.target_npz, None, True),
        (args.target_done, None, False),
    ]


def _implementation_records(
    args: argparse.Namespace,
) -> list[tuple[Path, str | None, bool]]:
    return [
        (args.target_wrapper, None, False),
        (
            args.target_producer_test,
            EXPECTED_TARGET_PRODUCER_TEST_SHA256,
            False,
        ),
        (args.target_wrapper_test, None, False),
        (args.reducer_test, None, False),
    ]


def _prediction_anchor_records(
    args: argparse.Namespace,
) -> list[tuple[Path, str | None, bool]]:
    records: list[tuple[Path, str | None, bool]] = [
        (
            args.canonical_k10000_json,
            EXPECTED_CANONICAL_K10_JSON_SHA256,
            True,
        ),
        (
            args.canonical_k10000_npz,
            EXPECTED_CANONICAL_K10_NPZ_SHA256,
            True,
        ),
        (
            args.stage_b_k10000_json,
            EXPECTED_STAGE_B_K10_JSON_SHA256,
            True,
        ),
        (
            args.stage_b_k10000_npz,
            EXPECTED_STAGE_B_K10_NPZ_SHA256,
            True,
        ),
        (
            args.accepted_adjudication_json,
            EXPECTED_ACCEPTED_ADJUDICATION_SHA256,
            True,
        ),
    ]
    for lane, path in enumerate(args.prediction_lane_json):
        records.append(
            (
                path,
                EXPECTED_LANE_JSON_SHA256[lane] if lane < 4 else None,
                True,
            )
        )
    for lane, path in enumerate(args.prediction_lane_npz):
        records.append(
            (
                path,
                EXPECTED_LANE_NPZ_SHA256[lane] if lane < 4 else None,
                True,
            )
        )
    return records


def _input_records(
    args: argparse.Namespace,
) -> list[tuple[Path, str | None, bool]]:
    return [
        (args.preregistration, None, True),
        (args.activation, None, True),
        (args.launch_manifest, None, True),
        *_target_records(args),
        *_implementation_records(args),
        *_prediction_anchor_records(args),
    ]


def _validate_output_paths(
    output: Path,
    records: Sequence[tuple[Path, str | None, bool]],
) -> None:
    output_sidecar = output.with_name(f"{output.name}.sha256")
    try:
        normalized_output = output.resolve(strict=False)
        normalized_output_sidecar = output_sidecar.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"output path cannot be normalized: {output}") from error
    inputs: set[Path] = set()
    for path, _, requires_sidecar in records:
        try:
            inputs.add(path.resolve(strict=False))
        except (OSError, RuntimeError) as error:
            raise InvalidEvidence(
                f"input path cannot be normalized safely: {path}"
            ) from error
        if requires_sidecar:
            sidecar = path.with_name(f"{path.name}.sha256")
            try:
                inputs.add(sidecar.resolve(strict=False))
            except (OSError, RuntimeError) as error:
                raise InvalidEvidence(
                    f"input sidecar path cannot be normalized safely: {sidecar}"
                ) from error
    overlap = {normalized_output, normalized_output_sidecar} & inputs
    if overlap:
        raise ValueError(
            f"output paths alias required inputs: {sorted(map(str, overlap))}"
        )
    for path in (output, output_sidecar):
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"refusing to overwrite {path}")


def _created_at_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _next_step(outcome: str) -> str:
    return {
        INCOMPLETE_OUTCOME: (
            "Complete or rerun the missing instrument in a new attempt namespace; "
            "no H-QC evidence exists."
        ),
        INVALID_OUTCOME: (
            "Repair the invalid instrument and rerun in a new attempt namespace; "
            "no H-QC evidence exists."
        ),
        INELIGIBLE_OUTCOME: (
            "Do not reinterpret an absent baseline cliff as evidence for or "
            "against scored-ID/truth-set expansion."
        ),
        FUTILE_OUTCOME: (
            "Reject scored-ID/truth-set expansion as the dominant within-panel "
            "explanation and derive a source-representation diagnostic."
        ),
        MIXED_OUTCOME: (
            "Resolve the mixed mechanism with a smaller targeted diagnostic; do "
            "not launch a broad architecture sweep."
        ),
        DUAL_OUTCOME: (
            "Design a genuinely disjoint-query decoder experiment; this aligned "
            "prefix result does not establish independent-query generalization."
        ),
        AREA_FLAT_OUTCOME: (
            "Treat the cliff as sensitive to uniform evaluation weighting within "
            "this panel, then test measure-aware evaluation and training separately."
        ),
        UNIFORM_ONLY_OUTCOME: (
            "Run a targeted physical-objective diagnostic before architecture work."
        ),
    }[outcome]


def _adjudicate_valid(args: argparse.Namespace) -> dict[str, Any]:
    incomplete: list[str] = []
    reducer_sha256 = hashlib.sha256(_stable_read(Path(__file__).resolve())).hexdigest()

    (
        observed_preregistration,
        observed_preregistration_sha256,
        preregistration_authenticated,
        preregistration_incomplete,
    ) = _load_json_once(
        args.preregistration,
        context="H-QC preregistration",
    )
    incomplete.extend(preregistration_incomplete)
    if observed_preregistration is not None:
        _validate_preregistration(
            observed_preregistration,
            reducer_sha256=reducer_sha256,
        )
    preregistration = (
        observed_preregistration if preregistration_authenticated else None
    )
    preregistration_sha256 = (
        observed_preregistration_sha256 if preregistration_authenticated else None
    )

    (
        observed_activation,
        observed_activation_sha256,
        activation_authenticated,
        activation_incomplete,
    ) = _load_json_once(
        args.activation,
        context="one-step activation",
    )
    incomplete.extend(activation_incomplete)
    if observed_activation is not None:
        _validate_activation(observed_activation)
    activation = observed_activation if activation_authenticated else None
    activation_sha256 = observed_activation_sha256 if activation_authenticated else None

    (
        observed_launch_manifest,
        observed_launch_manifest_sha256,
        launch_manifest_authenticated,
        launch_manifest_incomplete,
    ) = _load_json_once(
        args.launch_manifest,
        context="target-freeze launch manifest",
    )
    incomplete.extend(launch_manifest_incomplete)
    if observed_launch_manifest is not None:
        _validate_launch_manifest_static(
            observed_launch_manifest,
            launch_manifest_path=args.launch_manifest,
            target_json_path=args.target_json,
            target_npz_path=args.target_npz,
            target_done_path=args.target_done,
            output_json_path=args.output_json,
        )
        _audit_launch_manifest_available_bindings(
            observed_launch_manifest,
            preregistration=observed_preregistration,
            preregistration_sha256=observed_preregistration_sha256,
            activation_sha256=observed_activation_sha256,
            reducer_sha256=reducer_sha256,
            wrapper_path=args.target_wrapper,
            target_producer_test_path=args.target_producer_test,
            target_wrapper_test_path=args.target_wrapper_test,
            reducer_test_path=args.reducer_test,
        )
    implementation_incomplete = _preflight_inputs(_implementation_records(args))
    incomplete.extend(implementation_incomplete)
    implementation_ready = not implementation_incomplete

    attempt_id: str | None = None
    wrapper_sha256: str | None = None
    if (
        observed_launch_manifest is not None
        and observed_preregistration is not None
        and observed_preregistration_sha256 is not None
        and observed_activation_sha256 is not None
        and implementation_ready
    ):
        try:
            observed_attempt_id, observed_wrapper_sha256 = _validate_launch_manifest(
                observed_launch_manifest,
                preregistration=observed_preregistration,
                preregistration_sha256=observed_preregistration_sha256,
                activation_sha256=observed_activation_sha256,
                wrapper_path=args.target_wrapper,
                target_producer_test_path=args.target_producer_test,
                target_wrapper_test_path=args.target_wrapper_test,
                reducer_test_path=args.reducer_test,
                launch_manifest_path=args.launch_manifest,
                target_json_path=args.target_json,
                target_npz_path=args.target_npz,
                target_done_path=args.target_done,
                output_json_path=args.output_json,
                reducer_sha256=reducer_sha256,
            )
        except IncompleteEvidence as error:
            incomplete.append(str(error))
        else:
            if (
                launch_manifest_authenticated
                and preregistration_authenticated
                and activation_authenticated
            ):
                attempt_id = observed_attempt_id
                wrapper_sha256 = observed_wrapper_sha256
    launch_manifest_sha256 = (
        observed_launch_manifest_sha256
        if attempt_id is not None and launch_manifest_authenticated
        else None
    )
    authorization_ready = all(
        value is not None
        for value in (
            activation_sha256,
            preregistration_sha256,
            launch_manifest_sha256,
            attempt_id,
            wrapper_sha256,
        )
    )
    target_paths = (
        args.target_json,
        args.target_json.with_name(f"{args.target_json.name}.sha256"),
        args.target_npz,
        args.target_npz.with_name(f"{args.target_npz.name}.sha256"),
        args.target_done,
    )
    target_evidence_present = any(_entry_exists(path) for path in target_paths)
    if not authorization_ready:
        if target_evidence_present:
            raise InvalidEvidence(
                "target evidence exists without a valid authenticated activation, "
                "preregistration, and launch manifest"
            )
        incomplete.append(
            "target audit withheld until activation, preregistration, and launch "
            "manifest authorization are complete"
        )
    if len(args.prediction_lane_json) > 4 or len(args.prediction_lane_npz) > 4:
        raise InvalidEvidence(
            "more than four JSON or NPZ prediction lanes were supplied"
        )
    incomplete.extend(_preflight_inputs(_prediction_anchor_records(args)))
    if len(args.prediction_lane_json) < 4 or len(args.prediction_lane_npz) < 4:
        incomplete.append(
            "exactly four JSON and four NPZ prediction lanes are required"
        )
    if authorization_ready:
        incomplete.extend(_preflight_inputs(_target_records(args)))
        target_audits = (
            lambda: _audit_present_target_envelope(
                args.target_json,
                npz_path=args.target_npz,
            ),
            lambda: _audit_present_target_npz(args.target_npz),
            lambda: _audit_present_target_done(args.target_done),
            lambda: _audit_present_target_bindings(
                json_path=args.target_json,
                npz_path=args.target_npz,
                done_path=args.target_done,
                activation_path=args.activation,
                preregistration_path=args.preregistration,
                launch_manifest_path=args.launch_manifest,
                wrapper_path=args.target_wrapper,
            ),
        )
        for audit in target_audits:
            try:
                audit()
            except IncompleteEvidence as error:
                incomplete.append(str(error))

    lanes_ready = (
        len(args.prediction_lane_json) == 4
        and len(args.prediction_lane_npz) == 4
        and all(
            _artifact_ready(path)
            for path in (
                *args.prediction_lane_json,
                *args.prediction_lane_npz,
            )
        )
    )
    if (
        len(args.prediction_lane_json) == 4
        and len(args.prediction_lane_npz) == 4
        and not lanes_ready
    ):
        incomplete.append(
            "a prediction-lane artifact became unavailable after preflight"
        )
    predictions: dict[int, dict[str, Any]] | None = None
    lane_provenance: list[dict[str, str]] | None = None
    if lanes_ready:
        predictions, lane_provenance = _validate_prediction_lanes(
            args.prediction_lane_json,
            args.prediction_lane_npz,
        )

    target_ready = all(
        (
            _artifact_ready(args.target_json),
            _artifact_ready(args.target_npz),
            _artifact_ready(args.target_done, sidecar=False),
        )
    )
    if authorization_ready and not target_ready:
        incomplete.append("a target-bundle artifact became unavailable after preflight")
    targets: dict[int, dict[str, np.ndarray]] | None = None
    target_provenance: dict[str, str] | None = None
    if (
        target_ready
        and activation_sha256 is not None
        and preregistration_sha256 is not None
        and launch_manifest_sha256 is not None
        and attempt_id is not None
        and wrapper_sha256 is not None
    ):
        targets, target_provenance = _validate_target_bundle(
            json_path=args.target_json,
            npz_path=args.target_npz,
            done_path=args.target_done,
            activation_sha256=activation_sha256,
            preregistration_sha256=preregistration_sha256,
            attempt_id=attempt_id,
            launch_manifest_sha256=launch_manifest_sha256,
            wrapper_sha256=wrapper_sha256,
            predictions=predictions,
        )

    anchor_ready = all(
        _artifact_ready(path)
        for path in (
            args.canonical_k10000_json,
            args.canonical_k10000_npz,
            args.stage_b_k10000_json,
            args.stage_b_k10000_npz,
            args.accepted_adjudication_json,
        )
    )
    if not anchor_ready:
        incomplete.append(
            "a K=10000 anchor artifact became unavailable after preflight"
        )
    anchor: dict[str, Any] | None = None
    if anchor_ready and predictions is not None and targets is not None:
        anchor = _validate_k10000_anchor(
            canonical_json_path=args.canonical_k10000_json,
            canonical_npz_path=args.canonical_k10000_npz,
            stage_b_json_path=args.stage_b_k10000_json,
            stage_b_npz_path=args.stage_b_k10000_npz,
            adjudication_path=args.accepted_adjudication_json,
            predictions=predictions,
            targets=targets,
        )
    if incomplete:
        raise IncompleteEvidence("; ".join(incomplete))
    if (
        preregistration is None
        or preregistration_sha256 is None
        or activation is None
        or activation_sha256 is None
        or launch_manifest_sha256 is None
        or attempt_id is None
        or wrapper_sha256 is None
        or predictions is None
        or lane_provenance is None
        or targets is None
        or target_provenance is None
        or anchor is None
    ):
        raise InvalidEvidence(
            "complete input audit did not produce every required value"
        )

    panel = _compute_panel(predictions=predictions, targets=targets)
    outcome = str(panel.pop("decision_outcome"))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": VALID_STATUS,
        "decision_outcome": outcome,
        "created_at_utc": _created_at_utc(),
        "validity": {
            "conditional_activation_verified": True,
            "attempt_namespace_and_launch_manifest_verified": True,
            "prediction_lane_sidecars_and_manifests_verified": True,
            "blind_target_commit_verified": True,
            "all_selected_id_joins_verified": True,
            "all_required_values_finite": True,
            "k10000_anchor_verified": True,
            "pressure_is_sole_deciding_field": True,
            "bfloat16_is_sole_deciding_precision": True,
        },
        "provenance": {
            "reducer_sha256": reducer_sha256,
            "preregistration_sha256": preregistration_sha256,
            "activation_sha256": activation_sha256,
            "attempt_id": attempt_id,
            "launch_manifest_sha256": launch_manifest_sha256,
            "target_wrapper_sha256": wrapper_sha256,
            "target_bundle": target_provenance,
            "prediction_lanes": lane_provenance,
        },
        "k10000_anchor": anchor,
        **panel,
        "categorical_precedence": [
            "incomplete_or_invalid_before_scientific_reduction",
            "ineligible_before_futility_or_support",
            "futility_before_support",
            "support_only_if_both_endpoints_and_no_futility",
            "otherwise_mixed",
        ],
        "limited_claim": (
            "This is an operational result for one sealed checkpoint, cohort, "
            "aligned nested panel, and metric contract. The fixed panel holds "
            "cell IDs and truths fixed, not the source-dependent query-coordinate "
            "tensor. It is not evidence of independent-query generalization, "
            "training benefit, population generalization, force accuracy, or "
            "architecture superiority."
        ),
        "next_step": _next_step(outcome),
    }


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _adjudicate_valid(args)
    except IncompleteEvidence as error:
        outcome = INCOMPLETE_OUTCOME
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INCOMPLETE_STATUS,
            "decision_outcome": outcome,
            "created_at_utc": _created_at_utc(),
            "errors": [str(error)],
            "scientific_evidence_exists": False,
            "next_step": _next_step(outcome),
        }
    except InvalidEvidence as error:
        outcome = INVALID_OUTCOME
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INVALID_STATUS,
            "decision_outcome": outcome,
            "created_at_utc": _created_at_utc(),
            "errors": [str(error)],
            "scientific_evidence_exists": False,
            "next_step": _next_step(outcome),
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--target-wrapper", type=Path, required=True)
    parser.add_argument("--target-producer-test", type=Path, required=True)
    parser.add_argument("--target-wrapper-test", type=Path, required=True)
    parser.add_argument("--reducer-test", type=Path, required=True)
    parser.add_argument(
        "--prediction-lane-json",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--prediction-lane-npz",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--target-json", type=Path, required=True)
    parser.add_argument("--target-npz", type=Path, required=True)
    parser.add_argument("--target-done", type=Path, required=True)
    parser.add_argument("--canonical-k10000-json", type=Path, required=True)
    parser.add_argument("--canonical-k10000-npz", type=Path, required=True)
    parser.add_argument("--stage-b-k10000-json", type=Path, required=True)
    parser.add_argument("--stage-b-k10000-npz", type=Path, required=True)
    parser.add_argument(
        "--accepted-adjudication-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.output_json = Path(os.path.abspath(args.output_json))
    for field in (
        "preregistration",
        "activation",
        "launch_manifest",
        "target_wrapper",
        "target_producer_test",
        "target_wrapper_test",
        "reducer_test",
        "target_json",
        "target_npz",
        "target_done",
        "canonical_k10000_json",
        "canonical_k10000_npz",
        "stage_b_k10000_json",
        "stage_b_k10000_npz",
        "accepted_adjudication_json",
    ):
        setattr(args, field, Path(os.path.abspath(getattr(args, field))))
    args.prediction_lane_json = [
        Path(os.path.abspath(path)) for path in args.prediction_lane_json
    ]
    args.prediction_lane_npz = [
        Path(os.path.abspath(path)) for path in args.prediction_lane_npz
    ]
    try:
        _validate_output_paths(args.output_json, _input_records(args))
    except InvalidEvidence as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "status": INVALID_STATUS,
            "decision_outcome": INVALID_OUTCOME,
            "created_at_utc": _created_at_utc(),
            "errors": [str(error)],
            "scientific_evidence_exists": False,
            "next_step": _next_step(INVALID_OUTCOME),
        }
    else:
        result = adjudicate(args)
    digest = _publish_json_once(args.output_json, result)
    print(
        f"{result['status']} outcome={result['decision_outcome']} json_sha256={digest}",
        flush=True,
    )
    if result["decision_outcome"] == INVALID_OUTCOME:
        raise SystemExit(4)
    if result["decision_outcome"] == INCOMPLETE_OUTCOME:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
