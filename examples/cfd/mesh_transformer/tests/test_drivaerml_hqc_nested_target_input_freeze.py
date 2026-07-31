# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the blind nested-resolution raw-target freezer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import drivaerml_hqc_nested_target_input_freeze as freeze
import numpy as np
import pytest


def _synthetic_inputs(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    dataset_manifest = b'{"id_reference":["run_x"]}\n'
    (dataset / "manifest.json").write_bytes(dataset_manifest)

    source = tmp_path / "source" / "run_x"
    target_root = (
        source
        / "domain_run_x.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
        / "cell_data"
    )
    target_root.mkdir(parents=True)
    metadata = {
        "pMeanTrim": {
            "shape": [8],
            "dtype": "torch.float32",
            "device": "cpu",
        },
        "wallShearStressMeanTrim": {
            "shape": [8, 3],
            "dtype": "torch.float32",
            "device": "cpu",
        },
        "ignored": {
            "shape": [8],
            "dtype": "torch.float32",
            "device": "cpu",
        },
    }
    metadata_path = target_root / "meta.json"
    metadata_path.write_text(json.dumps(metadata))
    pressure = np.arange(8, dtype="<f4")
    wss = np.arange(24, dtype="<f4").reshape(8, 3)
    (target_root / "pMeanTrim.memmap").write_bytes(pressure.tobytes())
    (target_root / "wallShearStressMeanTrim.memmap").write_bytes(wss.tobytes())
    (target_root / "ignored.memmap").write_bytes(b"must-not-open")
    (dataset / "run_x").symlink_to(source)
    symlink_target = os.readlink(dataset / "run_x")

    cohort = [
        {
            "cohort_ordinal": 0,
            "case_id": "run_x",
            "reader_index": 0,
            "n_master_cells": 8,
            "historical_start": 6,
        }
    ]
    cohort_sha256 = freeze._sha256_bytes(freeze._canonical_json_bytes(cohort))
    geometry = {
        "schema_version": 1,
        "artifact_kind": freeze.GEOMETRY_ARTIFACT_KIND,
        "status": freeze.GEOMETRY_STATUS,
        "case_count": 1,
        "cohort_sha256": cohort_sha256,
        "dataset_manifest": {
            "sha256": hashlib.sha256(dataset_manifest).hexdigest(),
        },
        "cases": [
            {
                **cohort[0],
                "resolved_case_root": str(source.resolve()),
                "symlink_target": symlink_target,
                "global_input_values_float32": {
                    "U_inf": [2.0, 0.0, 0.0],
                    "p_inf": [1.0],
                    "rho_inf": [1.25],
                    "nu": [1.0e-5],
                    "L_ref": [5.0],
                },
            }
        ],
    }
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry))
    historical_target = {
        "schema_version": 1,
        "artifact_kind": freeze.HISTORICAL_TARGET_ARTIFACT_KIND,
        "status": freeze.HISTORICAL_TARGET_STATUS,
        "case_count": 1,
        "resolution": 2,
        "cohort_sha256": cohort_sha256,
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest).hexdigest(),
        "cases": [
            {
                **cohort[0],
                "resolution": 2,
                "resolved_case_root": str(source.resolve()),
                "symlink_target": symlink_target,
                "cell_data_metadata": {
                    "size_bytes": len(metadata_path.read_bytes()),
                    "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                },
                "selected_targets": {
                    "pressure": {
                        "raw_field_name": "pMeanTrim",
                        "source_relative_path": (
                            "domain_run_x.pdmsh/_tensordict/boundaries/vehicle/"
                            "_tensordict/cell_data/pMeanTrim.memmap"
                        ),
                        "source_size_bytes": 8 * 4,
                        "source_offset_bytes": 6 * 4,
                        "selected_size_bytes": 2 * 4,
                        "selected_shape": [2],
                        "selected_dtype": "float32_little_endian",
                        "selected_sha256": hashlib.sha256(
                            pressure[6:8].tobytes()
                        ).hexdigest(),
                    },
                    "wss": {
                        "raw_field_name": "wallShearStressMeanTrim",
                        "source_relative_path": (
                            "domain_run_x.pdmsh/_tensordict/boundaries/vehicle/"
                            "_tensordict/cell_data/wallShearStressMeanTrim.memmap"
                        ),
                        "source_size_bytes": 8 * 3 * 4,
                        "source_offset_bytes": 6 * 3 * 4,
                        "selected_size_bytes": 2 * 3 * 4,
                        "selected_shape": [2, 3],
                        "selected_dtype": "float32_little_endian",
                        "selected_sha256": hashlib.sha256(
                            wss[6:8].tobytes()
                        ).hexdigest(),
                    },
                },
            }
        ],
    }
    historical_target_path = tmp_path / "historical_targets.json"
    historical_target_path.write_text(json.dumps(historical_target))

    monkeypatch.setattr(freeze, "RESOLUTIONS", (2, 3, 5))
    monkeypatch.setattr(freeze, "MAX_RESOLUTION", 5)
    monkeypatch.setattr(freeze, "FIXED_QUERY_RESOLUTION", 2)
    monkeypatch.setattr(freeze, "HISTORICAL_ANCHOR_RESOLUTION", 2)
    monkeypatch.setattr(freeze, "CASE_SPECS", ((0, "run_x", 0, 8, 6),))
    monkeypatch.setattr(freeze, "EXPECTED_CASE_COUNT", 1)
    monkeypatch.setattr(freeze, "EXPECTED_COHORT_SHA256", cohort_sha256)
    monkeypatch.setattr(
        freeze,
        "DATASET_MANIFEST_SHA256",
        hashlib.sha256(dataset_manifest).hexdigest(),
    )
    monkeypatch.setattr(
        freeze,
        "GEOMETRY_MANIFEST_SHA256",
        hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freeze,
        "HISTORICAL_TARGET_MANIFEST_SHA256",
        hashlib.sha256(historical_target_path.read_bytes()).hexdigest(),
    )
    return dataset, geometry_path, historical_target_path, pressure, wss


def test_run469_cyclic_span_arithmetic_is_exact():
    pressure = freeze._cyclic_f32_byte_spans(
        n_rows=19_780_049,
        start_row=19_757_508,
        row_count=40_000,
        components=1,
    )
    wss = freeze._cyclic_f32_byte_spans(
        n_rows=19_780_049,
        start_row=19_757_508,
        row_count=40_000,
        components=3,
    )
    assert pressure == ((79_030_032, 90_164), (0, 69_836))
    assert wss == ((237_090_096, 270_492), (0, 209_508))

    ids = freeze._selected_ids(19_780_049, 19_757_508)
    assert ids.shape == (40_000,)
    assert (ids[0], ids[22_540], ids[22_541], ids[-1]) == (
        19_757_508,
        19_780_048,
        0,
        17_458,
    )
    assert hashlib.sha256(ids.tobytes()).hexdigest() == (
        "60f55b582fd8d21859962853a08039188b532078c4e9ae0496f7fff16704d342"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rows": 0, "start_row": 0, "row_count": 1, "components": 1},
        {"n_rows": 8, "start_row": -1, "row_count": 1, "components": 1},
        {"n_rows": 8, "start_row": 8, "row_count": 1, "components": 1},
        {"n_rows": 8, "start_row": 0, "row_count": 0, "components": 1},
        {"n_rows": 8, "start_row": 0, "row_count": 9, "components": 1},
        {"n_rows": 8, "start_row": 0, "row_count": 1, "components": 0},
    ],
)
def test_cyclic_span_arithmetic_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        freeze._cyclic_f32_byte_spans(**kwargs)


def test_cyclic_span_arithmetic_rejects_bool_and_noninteger():
    with pytest.raises(TypeError):
        freeze._cyclic_f32_byte_spans(
            n_rows=8,
            start_row=False,
            row_count=2,
            components=1,
        )
    with pytest.raises(TypeError):
        freeze._cyclic_f32_byte_spans(
            n_rows=8,
            start_row=1,
            row_count=2.0,
            components=1,
        )


def test_safe_cyclic_rows_preserves_float32_bits_across_wrap(tmp_path):
    bits = np.asarray(
        [
            0x00000000,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x7FC01234,
            0x00000001,
        ],
        dtype="<u4",
    )
    path = tmp_path / "bits.memmap"
    path.write_bytes(bits.tobytes())
    payload, spans = freeze._safe_cyclic_f32_rows(
        path,
        n_rows=6,
        start_row=4,
        row_count=4,
        components=1,
    )
    expected = bits[[4, 5, 0, 1]].tobytes()
    assert payload == expected
    assert spans == ((16, 8), (0, 8))


def test_safe_cyclic_vector_rows_use_one_descriptor_in_order(
    tmp_path,
    monkeypatch,
):
    values = np.arange(24, dtype="<f4").reshape(8, 3)
    path = tmp_path / "vectors.memmap"
    path.write_bytes(values.tobytes())
    original_open = os.open
    original_close = os.close
    original_pread = os.pread
    opened = []
    closed = []
    reads = []

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    def tracked_pread(descriptor, count, offset):
        reads.append((descriptor, count, offset))
        return original_pread(descriptor, count, offset)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    monkeypatch.setattr(os, "pread", tracked_pread)
    payload, spans = freeze._safe_cyclic_f32_rows(
        path,
        n_rows=8,
        start_row=6,
        row_count=5,
        components=3,
    )

    assert payload == values[[6, 7, 0, 1, 2]].tobytes()
    assert spans == ((72, 24), (0, 36))
    assert len(opened) == 1
    assert closed == opened
    assert reads == [(opened[0], 24, 72), (opened[0], 36, 0)]


def test_exact_end_and_full_cycle_have_expected_spans(tmp_path):
    values = np.arange(8, dtype="<f4")
    path = tmp_path / "values.memmap"
    path.write_bytes(values.tobytes())
    exact_end, exact_spans = freeze._safe_cyclic_f32_rows(
        path,
        n_rows=8,
        start_row=5,
        row_count=3,
        components=1,
    )
    full_cycle, full_spans = freeze._safe_cyclic_f32_rows(
        path,
        n_rows=8,
        start_row=5,
        row_count=8,
        components=1,
    )
    assert exact_end == values[5:].tobytes()
    assert exact_spans == ((20, 12),)
    assert full_cycle == values[[5, 6, 7, 0, 1, 2, 3, 4]].tobytes()
    assert full_spans == ((20, 12), (0, 20))


def test_safe_cyclic_rows_rejects_size_directory_and_symlink(tmp_path):
    path = tmp_path / "values.memmap"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="size changed"):
        freeze._safe_cyclic_f32_rows(
            path,
            n_rows=2,
            start_row=0,
            row_count=1,
            components=1,
        )
    with pytest.raises(ValueError, match="not a regular file"):
        freeze._safe_cyclic_f32_rows(
            tmp_path,
            n_rows=1,
            start_row=0,
            row_count=1,
            components=1,
        )
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        freeze._safe_cyclic_f32_rows(
            link,
            n_rows=1,
            start_row=0,
            row_count=1,
            components=1,
        )


def test_safe_cyclic_rows_rejects_short_second_read(tmp_path, monkeypatch):
    values = np.arange(8, dtype="<f4")
    path = tmp_path / "values.memmap"
    path.write_bytes(values.tobytes())
    original = os.pread
    calls = 0

    def short_second(descriptor, count, offset):
        nonlocal calls
        calls += 1
        payload = original(descriptor, count, offset)
        return payload[:-1] if calls == 2 else payload

    monkeypatch.setattr(os, "pread", short_second)
    with pytest.raises(ValueError, match="Short target read"):
        freeze._safe_cyclic_f32_rows(
            path,
            n_rows=8,
            start_row=6,
            row_count=5,
            components=1,
        )


def test_safe_cyclic_rows_rejects_mutation_between_spans(tmp_path, monkeypatch):
    values = np.arange(8, dtype="<f4")
    path = tmp_path / "values.memmap"
    path.write_bytes(values.tobytes())
    original = os.pread
    calls = 0

    def mutate_after_tail(descriptor, count, offset):
        nonlocal calls
        payload = original(descriptor, count, offset)
        calls += 1
        if calls == 1:
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"\xff")
                stream.flush()
                os.fsync(stream.fileno())
        return payload

    monkeypatch.setattr(os, "pread", mutate_after_tail)
    with pytest.raises(ValueError, match="changed while being read"):
        freeze._safe_cyclic_f32_rows(
            path,
            n_rows=8,
            start_row=6,
            row_count=5,
            components=1,
        )


def test_main_publishes_cyclic_raw_bundle(tmp_path, monkeypatch):
    dataset, geometry, historical, pressure, wss = _synthetic_inputs(
        tmp_path,
        monkeypatch,
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_json = output_dir / "targets.json"
    output_npz = output_dir / "targets.npz"
    freeze.main(
        [
            "--dataset-root",
            str(dataset),
            "--geometry-manifest",
            str(geometry),
            "--historical-k10000-target-manifest",
            str(historical),
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ]
    )

    document = json.loads(output_json.read_text())
    assert document["status"] == freeze.STATUS
    assert document["case_count"] == 1
    assert document["npz"]["array_count"] == 4
    assert (
        document["historical_k10000_target_manifest"]["prefix_hashes_authenticated"]
        == 2
    )
    assert document["physical_globals"]["field_order"] == [
        "U_inf_x",
        "U_inf_y",
        "U_inf_z",
        "p_inf",
        "rho_inf",
        "nu",
        "L_ref",
    ]
    assert document["physical_globals"]["transformed_by_target_freezer"] is False
    assert document["cases"][0]["selection"]["wraps"] is True
    assert document["cases"][0]["targets"]["pressure"]["source_spans_bytes"] == [
        {"offset": 24, "count": 8},
        {"offset": 0, "count": 12},
    ]
    assert document["read_exclusions"]["model_opened"] is False
    assert document["read_exclusions"]["prediction_opened"] is False
    assert document["publication_contract"]["json_manifest_linked_last"] is True
    assert (
        document["publication_contract"][
            "valid_only_after_external_sidecar_checks_and_done_marker"
        ]
        is True
    )

    with np.load(output_npz, allow_pickle=False) as archive:
        prefix = "case_00_run_x__"
        expected_ids = np.asarray([6, 7, 0, 1, 2], dtype="<i8")
        assert np.array_equal(
            archive[f"{prefix}selected_cell_ids_int64"],
            expected_ids,
        )
        assert np.array_equal(
            archive[f"{prefix}raw_target_pressure_float32"],
            pressure[expected_ids],
        )
        assert np.array_equal(
            archive[f"{prefix}raw_target_wss_float32"],
            wss[expected_ids],
        )
        assert np.array_equal(
            archive[f"{prefix}physical_globals_float32"],
            np.asarray([2.0, 0.0, 0.0, 1.0, 1.25, 1.0e-5, 5.0], dtype="<f4"),
        )

    for path in (output_json, output_npz):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.with_name(f"{path.name}.sha256").read_text() == (
            f"{digest}  {path.name}\n"
        )


def test_main_refuses_any_existing_bundle_output(tmp_path, monkeypatch):
    dataset, geometry, historical, _, _ = _synthetic_inputs(tmp_path, monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_json = output_dir / "targets.json"
    output_npz = output_dir / "targets.npz"
    output_npz.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError):
        freeze.main(
            [
                "--dataset-root",
                str(dataset),
                "--geometry-manifest",
                str(geometry),
                "--historical-k10000-target-manifest",
                str(historical),
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
            ]
        )
    assert output_npz.read_bytes() == b"sentinel"
    assert not output_json.exists()


def test_main_rejects_k10000_target_prefix_drift(tmp_path, monkeypatch):
    dataset, geometry, historical, pressure, _ = _synthetic_inputs(
        tmp_path,
        monkeypatch,
    )
    pressure[6] += np.float32(1.0)
    next(tmp_path.rglob("pMeanTrim.memmap")).write_bytes(pressure.tobytes())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="prefix differs from sealed target manifest"):
        freeze.main(
            [
                "--dataset-root",
                str(dataset),
                "--geometry-manifest",
                str(geometry),
                "--historical-k10000-target-manifest",
                str(historical),
                "--output-json",
                str(output_dir / "targets.json"),
                "--output-npz",
                str(output_dir / "targets.npz"),
            ]
        )


def test_main_rejects_symlink_text_drift_with_same_destination(tmp_path, monkeypatch):
    dataset, geometry, historical, _, _ = _synthetic_inputs(tmp_path, monkeypatch)
    case_link = dataset / "run_x"
    resolved = case_link.resolve()
    case_link.unlink()
    case_link.symlink_to(os.path.relpath(resolved, start=case_link.parent))
    assert case_link.resolve() == resolved
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="symlink target changed"):
        freeze.main(
            [
                "--dataset-root",
                str(dataset),
                "--geometry-manifest",
                str(geometry),
                "--historical-k10000-target-manifest",
                str(historical),
                "--output-json",
                str(output_dir / "targets.json"),
                "--output-npz",
                str(output_dir / "targets.npz"),
            ]
        )


def test_target_freezer_has_no_model_or_physicsnemo_dependency():
    source = Path(freeze.__file__).read_text()
    assert "import torch" not in source
    assert "physicsnemo" not in source
