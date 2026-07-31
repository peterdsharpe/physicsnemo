# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the selected-target input freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import drivaerml_historical_k10000_target_input_freeze as freeze
import numpy as np
import pytest


def _synthetic_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = b'{"id_reference":["run_x"]}\n'
    (dataset / "manifest.json").write_bytes(manifest)
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
        "shape": [8],
        "device": "cpu",
    }
    (target_root / "meta.json").write_text(json.dumps(metadata))
    pressure = np.arange(8, dtype="<f4")
    wss = np.arange(24, dtype="<f4").reshape(8, 3)
    (target_root / "pMeanTrim.memmap").write_bytes(pressure.tobytes())
    (target_root / "wallShearStressMeanTrim.memmap").write_bytes(wss.tobytes())
    (target_root / "ignored.memmap").write_bytes(b"never-open")
    (dataset / "run_x").symlink_to(source)
    monkeypatch.setattr(freeze, "RESOLUTION", 3)
    monkeypatch.setattr(freeze, "CASE_SPECS", ((0, "run_x", 0, 8, 2),))
    monkeypatch.setattr(freeze, "EXPECTED_CASE_COUNT", 1)
    monkeypatch.setattr(
        freeze,
        "EXPECTED_COHORT_SHA256",
        freeze._sha256_bytes(
            freeze._canonical_json_bytes(
                [
                    {
                        "cohort_ordinal": 0,
                        "case_id": "run_x",
                        "reader_index": 0,
                        "n_master_cells": 8,
                        "historical_start": 2,
                    }
                ]
            )
        ),
    )
    monkeypatch.setattr(
        freeze,
        "DATASET_MANIFEST_SHA256",
        hashlib.sha256(manifest).hexdigest(),
    )
    return dataset, pressure, wss


def test_safe_pread_reads_only_requested_range(tmp_path):
    values = np.arange(10, dtype="<f4")
    path = tmp_path / "values.memmap"
    path.write_bytes(values.tobytes())
    payload = freeze._safe_pread(
        path,
        offset=2 * 4,
        count=3 * 4,
        expected_file_size=10 * 4,
    )
    assert payload == values[2:5].tobytes()


def test_safe_pread_rejects_size_and_symlink(tmp_path):
    path = tmp_path / "values.memmap"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="size changed"):
        freeze._safe_pread(path, offset=0, count=4, expected_file_size=8)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        freeze._safe_pread(link, offset=0, count=4, expected_file_size=4)


def test_inspect_case_hashes_exact_selected_target_bytes(tmp_path, monkeypatch):
    dataset, pressure, wss = _synthetic_dataset(tmp_path, monkeypatch)
    case = freeze._inspect_case(dataset, freeze.CASE_SPECS[0])
    assert (
        case["selected_targets"]["pressure"]["selected_sha256"]
        == hashlib.sha256(pressure[2:5].tobytes()).hexdigest()
    )
    assert (
        case["selected_targets"]["wss"]["selected_sha256"]
        == hashlib.sha256(wss[2:5].tobytes()).hexdigest()
    )
    assert case["selected_targets"]["pressure"]["source_offset_bytes"] == 8
    assert case["selected_targets"]["wss"]["source_offset_bytes"] == 24


def test_inspect_case_rejects_target_metadata_drift(tmp_path, monkeypatch):
    dataset, _, _ = _synthetic_dataset(tmp_path, monkeypatch)
    metadata_path = next(tmp_path.rglob("cell_data/meta.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["pMeanTrim"]["shape"] = [7]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata changed"):
        freeze._inspect_case(dataset, freeze.CASE_SPECS[0])


def test_inspect_case_hashes_and_parses_the_same_metadata_read(tmp_path, monkeypatch):
    dataset, _, _ = _synthetic_dataset(tmp_path, monkeypatch)
    metadata_path = next(tmp_path.rglob("cell_data/meta.json"))
    original = freeze._safe_read_bytes
    metadata_reads = 0

    def counted(path):
        nonlocal metadata_reads
        if Path(path) == metadata_path:
            metadata_reads += 1
        return original(path)

    monkeypatch.setattr(freeze, "_safe_read_bytes", counted)
    freeze._inspect_case(dataset, freeze.CASE_SPECS[0])
    assert metadata_reads == 1


def test_main_publishes_manifest_and_sidecar(tmp_path, monkeypatch):
    dataset, pressure, wss = _synthetic_dataset(tmp_path, monkeypatch)
    output = tmp_path / "result" / "targets.json"
    freeze.main(
        [
            "--dataset-root",
            str(dataset),
            "--output-json",
            str(output),
        ]
    )
    document = json.loads(output.read_text())
    assert document["status"] == freeze.STATUS
    assert document["case_count"] == 1
    assert document["read_exclusions"]["other_cell_data_opened"] is False
    assert (
        document["cases"][0]["selected_targets"]["pressure"]["selected_sha256"]
        == hashlib.sha256(pressure[2:5].tobytes()).hexdigest()
    )
    assert (
        document["cases"][0]["selected_targets"]["wss"]["selected_sha256"]
        == hashlib.sha256(wss[2:5].tobytes()).hexdigest()
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_name(f"{output.name}.sha256").read_text() == (
        f"{digest}  {output.name}\n"
    )


def test_main_refuses_to_overwrite(tmp_path, monkeypatch):
    dataset, _, _ = _synthetic_dataset(tmp_path, monkeypatch)
    output = tmp_path / "sentinel.json"
    output.write_text("sentinel")
    with pytest.raises(FileExistsError):
        freeze.main(
            [
                "--dataset-root",
                str(dataset),
                "--output-json",
                str(output),
            ]
        )
    assert output.read_text() == "sentinel"


def test_main_rejects_dataset_reference_order_drift(tmp_path, monkeypatch):
    dataset, _, _ = _synthetic_dataset(tmp_path, monkeypatch)
    manifest = b'{"id_reference":["run_y"]}\n'
    (dataset / "manifest.json").write_bytes(manifest)
    monkeypatch.setattr(
        freeze,
        "DATASET_MANIFEST_SHA256",
        hashlib.sha256(manifest).hexdigest(),
    )
    with pytest.raises(RuntimeError, match="reference cohort order changed"):
        freeze.main(
            [
                "--dataset-root",
                str(dataset),
                "--output-json",
                str(tmp_path / "result.json"),
            ]
        )


def test_atomic_publish_rolls_back_when_postpublication_read_differs(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.json"
    sidecar = output.with_name(f"{output.name}.sha256")
    original = freeze._safe_read_bytes

    def altered(path):
        if Path(path) == output:
            return b"altered-after-link"
        return original(path)

    monkeypatch.setattr(freeze, "_safe_read_bytes", altered)
    with pytest.raises(RuntimeError, match="payload verification failed"):
        freeze._atomic_publish(output, b'{"status":"ok"}\n')
    assert not output.exists()
    assert not sidecar.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_aga_wrapper_embeds_current_producer_sha():
    producer = Path(freeze.__file__).resolve()
    wrapper = producer.with_name(
        "drivaerml_historical_k10000_target_input_freeze_aga.sbatch"
    )
    digest = hashlib.sha256(producer.read_bytes()).hexdigest()
    assert f"readonly SCRIPT_SHA256={digest}" in wrapper.read_text()


def test_selected_target_freeze_has_no_model_dependency():
    source = Path(freeze.__file__).read_text()
    assert "import torch" not in source
    assert "physicsnemo" not in source
    assert "model" not in freeze.TARGETS
