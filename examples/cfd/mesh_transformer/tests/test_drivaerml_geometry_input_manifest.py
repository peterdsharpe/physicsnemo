# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the target-free DrivAerML geometry input manifest."""

import hashlib
import json
import struct
from pathlib import Path

import drivaerml_geometry_input_manifest as manifest_module
import pytest
from drivaerml_geometry_input_manifest import (
    FORBIDDEN_PATH_PARTS,
    GLOBAL_INPUTS,
    _atomic_publish_manifest,
    _inspect_case,
    _relative_paths,
    _validate_output_target,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_case(tmp_path: Path, *, l_ref: float = 5.0):
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    source = tmp_path / "source" / "run_x"
    vehicle_td = (
        source
        / "domain_run_x.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
    )
    global_root = source / "domain_run_x.pdmsh" / "_tensordict" / "global_data"
    _write_json(source / "domain_run_x.pdmsh" / "meta.json", {"type": "domain"})
    _write_json(
        source / "domain_run_x.pdmsh" / "_tensordict" / "meta.json",
        {"type": "tensordict"},
    )
    _write_json(
        source / "domain_run_x.pdmsh" / "_tensordict" / "boundaries" / "meta.json",
        {"vehicle": {"type": "Mesh"}},
    )
    _write_json(vehicle_td.parent / "meta.json", {"type": "mesh"})
    _write_json(
        vehicle_td / "meta.json",
        {
            "points": {
                "device": "cpu",
                "shape": [2, 3],
                "dtype": "torch.float32",
            },
            "cells": {
                "device": "cpu",
                "shape": [1, 3],
                "dtype": "torch.int64",
            },
            "point_data": {"type": "TensorDict"},
            "cell_data": {"type": "TensorDict"},
        },
    )
    vehicle_td.joinpath("points.memmap").write_bytes(struct.pack("<6f", *range(6)))
    vehicle_td.joinpath("cells.memmap").write_bytes(struct.pack("<3q", 0, 1, 0))
    _write_json(
        global_root / "meta.json",
        {
            name: {
                "device": "cpu",
                "shape": [3] if name == "U_inf" else [],
                "dtype": "torch.float32",
            }
            for name in GLOBAL_INPUTS
        },
    )
    values = {
        "U_inf": (1.0, 2.0, 3.0),
        "p_inf": (4.0,),
        "rho_inf": (5.0,),
        "nu": (6.0,),
        "L_ref": (l_ref,),
    }
    for name, entries in values.items():
        global_root.joinpath(f"{name}.memmap").write_bytes(
            struct.pack(f"<{len(entries)}f", *entries)
        )

    # If the implementation traverses local target associations, this invalid
    # metadata is guaranteed to fail JSON parsing.
    target_root = vehicle_td / "cell_data"
    target_root.mkdir()
    target_root.joinpath("meta.json").write_bytes(b"{not-json")
    target_root.joinpath("pMeanTrim.memmap").write_bytes(b"do-not-open")
    target_root.joinpath("wallShearStressMeanTrim.memmap").write_bytes(b"do-not-open")
    dataset.joinpath("run_x").symlink_to(source)
    return dataset


def test_allowlist_contains_only_geometry_global_and_structural_paths():
    paths = _relative_paths("run_x")
    assert len(paths) == 13
    assert any(path.name == "points.memmap" for path in paths)
    assert any(path.name == "cells.memmap" for path in paths)
    assert {
        path.stem for path in paths if path.parent.name == "global_data"
    }.issuperset(GLOBAL_INPUTS)
    for path in paths:
        assert not {part.lower() for part in path.parts}.intersection(
            FORBIDDEN_PATH_PARTS
        )


def test_case_inspection_hashes_geometry_without_opening_target_files(tmp_path):
    dataset = _synthetic_case(tmp_path)
    result = _inspect_case(
        dataset,
        (0, "run_x", 7, 1, 0),
        workers=2,
    )

    assert result["case_id"] == "run_x"
    assert result["reader_index"] == 7
    assert result["n_master_cells"] == 1
    assert result["n_master_points"] == 2
    assert result["global_input_values_float32"]["L_ref"] == [5.0]
    assert len(result["files"]) == 13
    assert all("cell_data" not in path for path in result["files"])
    assert all("point_data" not in path for path in result["files"])
    assert all("interior" not in path for path in result["files"])


def test_case_inspection_rejects_scale_and_metadata_drift(tmp_path):
    dataset = _synthetic_case(tmp_path, l_ref=4.0)
    with pytest.raises(ValueError, match="L_ref changed"):
        _inspect_case(dataset, (0, "run_x", 7, 1, 0), workers=1)

    dataset = _synthetic_case(tmp_path / "second")
    meta = (
        dataset.resolve()
        / "run_x"
        / "domain_run_x.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
        / "meta.json"
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["cells"]["shape"] = [2, 3]
    meta.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cell metadata changed"):
        _inspect_case(dataset, (0, "run_x", 7, 1, 0), workers=1)


def test_output_target_rejects_existing_sidecar_and_dangling_link(tmp_path):
    clean = tmp_path / "manifest.json"
    _validate_output_target(clean)

    clean.with_name("manifest.json.sha256").write_text(
        "digest  manifest.json\n",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate_output_target(clean)

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate_output_target(dangling)


def test_manifest_pair_is_published_with_matching_sidecar(tmp_path):
    output = tmp_path / "manifest.json"
    payload = b'{"status":"ok"}\n'

    digest = _atomic_publish_manifest(output, payload)

    assert output.read_bytes() == payload
    assert output.with_name("manifest.json.sha256").read_text(encoding="ascii") == (
        f"{digest}  manifest.json\n"
    )
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("collision", ["output", "sidecar"])
def test_manifest_pair_does_not_clobber_post_validation_collision(
    tmp_path,
    collision,
):
    output = tmp_path / "manifest.json"
    sidecar = output.with_name("manifest.json.sha256")
    _validate_output_target(output)
    collided_path = output if collision == "output" else sidecar
    collided_path.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        _atomic_publish_manifest(output, b'{"status":"new"}\n')

    assert collided_path.read_bytes() == b"sentinel"
    if collision == "output":
        assert not sidecar.exists()
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_manifest_pair_cleans_first_temporary_if_second_preparation_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "manifest.json"
    original = manifest_module._write_fsynced_temporary
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected sidecar preparation failure")
        return original(path, payload)

    monkeypatch.setattr(
        manifest_module,
        "_write_fsynced_temporary",
        fail_second,
    )
    with pytest.raises(OSError, match="injected"):
        _atomic_publish_manifest(output, b'{"status":"new"}\n')

    assert not output.exists()
    assert not output.with_name("manifest.json.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_aga_wrapper_embeds_current_producer_sha():
    producer = Path(manifest_module.__file__).resolve()
    wrapper = producer.with_name("drivaerml_geometry_input_manifest_aga.sbatch")
    digest = hashlib.sha256(producer.read_bytes()).hexdigest()

    assert f"check_sha {digest} \\" in wrapper.read_text(encoding="utf-8")
