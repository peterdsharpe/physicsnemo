# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the target-blind historical K=10k canonical arm."""

from __future__ import annotations

import ast
import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

import drivaerml_historical_k10000_canonical_arm as arm
import drivaerml_historical_k10000_replay as legacy
import drivaerml_hqc_canonical_geometry_diagnostic_v5 as canonical
import numpy as np
import pytest
import torch

from physicsnemo.mesh import Mesh


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_hashes_are_exactly_bound():
    study_dir = Path(arm.__file__).resolve().parent
    assert (
        _sha(study_dir / arm.LEGACY_SUPPORT_FILENAME)
        == arm.EXPECTED_LEGACY_SUPPORT_SHA256
    )
    assert (
        _sha(study_dir / arm.RUNTIME_HELPER_FILENAME)
        == arm.EXPECTED_RUNTIME_HELPER_SHA256
    )
    assert (
        _sha(study_dir / arm.CANONICAL_HELPER_FILENAME)
        == arm.EXPECTED_CANONICAL_HELPER_SHA256
    )


def test_module_load_fails_before_import_when_hash_differs(tmp_path, monkeypatch):
    path = tmp_path / "helper.py"
    path.write_text("RAISED = True\n")
    imported = False

    def forbidden(*args, **kwargs):
        nonlocal imported
        del args, kwargs
        imported = True
        raise AssertionError("import attempted")

    monkeypatch.setattr(arm.importlib.util, "spec_from_file_location", forbidden)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        arm._load_verified_module(
            path,
            expected_sha256="0" * 64,
            module_name="must_not_load",
            label="test helper",
        )
    assert not imported


def _synthetic_geometry_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(arm, "RESOLUTION", 2)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    case_root = tmp_path / "case"
    case_root.mkdir()
    (dataset / "run_x").symlink_to(case_root)
    tensor_root = (
        case_root
        / "domain_run_x.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
    )
    tensor_root.mkdir(parents=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype="<f4",
    )
    cells = np.array(
        [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]],
        dtype="<i8",
    )
    (tensor_root / "points.memmap").write_bytes(points.tobytes())
    (tensor_root / "cells.memmap").write_bytes(cells.tobytes())
    base = "domain_run_x.pdmsh/_tensordict/boundaries/vehicle/_tensordict"
    geometry_case = {
        "case_id": "run_x",
        "resolved_case_root": str(case_root),
        "n_master_points": len(points),
        "files": {
            f"{base}/points.memmap": {
                "size_bytes": len(points.tobytes()),
                "sha256": _sha(tensor_root / "points.memmap"),
            },
            f"{base}/cells.memmap": {
                "size_bytes": len(cells.tobytes()),
                "sha256": _sha(tensor_root / "cells.memmap"),
            },
        },
        "global_input_values_float32": {
            "U_inf": [38.889, 0.0, 0.0],
            "p_inf": [0.0],
            "rho_inf": [1.0],
            "nu": [1.0],
            "L_ref": [5.0],
        },
    }
    spec = SimpleNamespace(
        case_id="run_x",
        historical_start=1,
        n_master_cells=4,
    )
    return dataset, geometry_case, spec


def test_geometry_loader_opens_only_cells_and_points(tmp_path, monkeypatch):
    dataset, geometry_case, spec = _synthetic_geometry_subset(tmp_path, monkeypatch)
    opened: list[str] = []
    original = legacy._open_regular

    def tracked(path, *, expected_size):
        opened.append(Path(path).name)
        return original(path, expected_size=expected_size)

    monkeypatch.setattr(legacy, "_open_regular", tracked)
    mesh, arrays = arm._load_geometry_only_subset(
        legacy,
        SimpleNamespace(mesh_type=Mesh),
        dataset,
        spec,
        geometry_case,
    )
    assert opened == ["cells.memmap", "points.memmap"]
    assert mesh.n_cells == 2
    assert mesh.n_points == 4
    assert arrays["selected_cell_ids_int64"].tolist() == [1, 2]
    assert arrays["compacted_cells_int64"].tolist() == [[0, 2, 1], [0, 3, 2]]
    assert arrays["raw_points_float32"].shape == (4, 3)
    assert arrays["raw_points_float32"].tobytes() == (
        mesh.points.numpy().astype("<f4", copy=False).tobytes()
    )
    assert set(mesh.cell_data.keys()) == {
        "pMeanTrim",
        "wallShearStressMeanTrim",
    }
    assert not torch.count_nonzero(mesh.cell_data["pMeanTrim"])
    assert not torch.count_nonzero(mesh.cell_data["wallShearStressMeanTrim"])


def test_canonical_bundle_uses_float64_area_center_scale_then_one_float32_cast():
    mesh = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.long),
    )
    raw = canonical._build_canonical_raw_geometry(mesh)
    bundle = canonical._finish_canonical_bundle(
        raw,
        physical_length=torch.tensor(5.0),
        model_reference_length=torch.tensor(8.0),
    )
    assert raw.points.dtype == torch.float64
    assert raw.centroids.dtype == torch.float64
    assert raw.areas.dtype == torch.float64
    assert raw.normals.dtype == torch.float64
    assert bundle.points.dtype == torch.float32
    assert bundle.centroids.dtype == torch.float32
    assert bundle.areas.dtype == torch.float32
    assert bundle.normals.dtype == torch.float32
    assert bundle.physical_center.tolist() == pytest.approx([2 / 3, 2 / 3, 0.0])
    assert bundle.centroids.tolist() == [[0.0, 0.0, 0.0]]
    assert bundle.areas.item() == pytest.approx(2.0 / (40.0**2))


class _CanonicalModel:
    boundary_names = ("vehicle",)
    reference_length_key = "reference_length"

    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, domain, *, canonical_source_geometry):
        del domain
        self.encode_calls += 1
        geometry = canonical_source_geometry
        source_mesh = Mesh(points=geometry.points, cells=geometry.cells)
        source_mesh._cache["cell", "centroids"] = geometry.centroids
        source_mesh._cache["cell", "areas"] = geometry.areas
        source_mesh._cache["cell", "normals"] = geometry.normals
        return SimpleNamespace(
            source_mesh=source_mesh,
            center=geometry.center,
            reference_length=geometry.reference_length,
            trace_slice=slice(0, 2),
        )

    def decode(self, encoded, query_mesh):
        del encoded
        self.decode_calls += 1
        return Mesh(
            points=query_mesh.points,
            point_data={
                "pressure": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "wss": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.float32,
                ),
            },
        )


def test_public_canonical_forward_encodes_and_decodes_once(tmp_path, monkeypatch):
    del tmp_path
    monkeypatch.setattr(arm, "RESOLUTION", 2)
    model = _CanonicalModel()
    bundle = SimpleNamespace(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        centroids=torch.tensor(
            [[1 / 3, 1 / 3, 0.0], [1 / 3, 1 / 3, 0.0]],
            dtype=torch.float32,
        ),
        areas=torch.tensor([0.5, 0.5], dtype=torch.float32),
        normals=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )
    boundary = Mesh(
        points=bundle.points.clone(),
        cells=bundle.cells.clone(),
    )
    domain = SimpleNamespace(
        boundaries={"vehicle": boundary},
        global_data={},
    )
    precision_calls: list[str] = []

    def autocast(precision):
        precision_calls.append(precision)
        return contextlib.nullcontext()

    runtime = SimpleNamespace(
        model=model,
        mesh_type=Mesh,
        autocast_context=autocast,
        normalize_output=lambda output, target_config, output_type: output.point_data,
        cfg=SimpleNamespace(output_type="mesh"),
    )

    def identity_redim(value, **kwargs):
        del kwargs
        return value

    arrays, validity, canonical_queries = arm._run_canonical_forward(
        canonical,
        runtime,
        domain,
        bundle,
        {
            "function": identity_redim,
            "normalizer": object(),
            "nondim": object(),
            "field_types": arm.TARGET_CONFIG,
        },
    )
    assert model.encode_calls == 1
    assert model.decode_calls == 1
    assert precision_calls == ["bfloat16"]
    assert validity["authoritative_cache"]["passed"]
    assert set(validity["authoritative_cache"]["cache_values_bitwise_exact"]) == {
        "points",
        "cells",
        "centroids",
        "areas",
        "normals",
    }
    assert validity["decode_contract_passed"]
    assert tuple(arrays) == arm.PREDICTION_SUFFIXES
    assert canonical_queries.tobytes() == bundle.centroids.numpy().tobytes()
    assert arrays["prediction_pressure_training_float32"].tolist() == [1.0, 2.0]
    assert (
        arrays["prediction_wss_physical_float32"].tobytes()
        == arrays["prediction_wss_training_float32"].tobytes()
    )


def test_case_array_contract_separates_controls_geometry_and_predictions():
    assert len(arm.PAIRING_CONTROL_SUFFIXES) == 11
    assert len(arm.CANONICAL_GEOMETRY_SUFFIXES) == 7
    assert len(arm.PREDICTION_SUFFIXES) == 4
    assert len(arm.CASE_ARRAY_SUFFIXES) == 22
    assert len(set(arm.CASE_ARRAY_SUFFIXES)) == 22
    assert all(
        "truth" not in name and "target" not in name for name in arm.CASE_ARRAY_SUFFIXES
    )


def test_single_rank_torchrun_environment_is_required(monkeypatch):
    for name, value in arm.EXPECTED_SINGLE_RANK_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert (
        arm._validate_single_rank_environment() == arm.EXPECTED_SINGLE_RANK_ENVIRONMENT
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="requires one torchrun rank"):
        arm._validate_single_rank_environment()


def test_cli_and_source_are_oracle_free_and_use_one_public_encode_decode():
    source = Path(arm.__file__).read_text()
    tree = ast.parse(source)
    cli_flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert cli_flags == {
        "--repo-root",
        "--dataset-root",
        "--dataset-config",
        "--resolved-config",
        "--checkpoint-dir",
        "--geometry-manifest",
        "--lane-label",
        "--output-json",
        "--output-npz",
    }
    forbidden_cli_tokens = ("target", "truth", "baseline", "metric", "ceiling")
    assert not any(
        token in flag.lower() for token in forbidden_cli_tokens for flag in cli_flags
    )
    assert "_verify_target_input_manifest" not in source
    assert "_safe_selected_target" not in source
    assert "_load_explicit_raw_subset" not in source

    encode_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "encode"
        and any(keyword.arg == "canonical_source_geometry" for keyword in node.keywords)
    ]
    decode_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode"
    ]
    assert len(encode_calls) == 1
    assert [keyword.arg for keyword in encode_calls[0].keywords] == [
        "canonical_source_geometry"
    ]
    assert len(decode_calls) == 1
    assert len(decode_calls[0].args) == 2
    assert '"hostname": platform.node()' in source
    assert '"rank_environment": dict(rank_environment)' in source


def test_atomic_output_set_is_non_overwriting(tmp_path):
    output_json = tmp_path / "canonical.json"
    output_npz = tmp_path / "canonical.npz"
    arrays = {"x": np.array([0.0, -0.0, 1.0], dtype="<f4")}
    temporary, npz_sha256 = legacy._prepare_npz_temporary(output_npz, arrays)
    payload = b'{"status":"ok"}\n'
    legacy._publish_output_set(
        output_json=output_json,
        json_payload=payload,
        output_npz=output_npz,
        npz_temporary=temporary,
        npz_sha256=npz_sha256,
    )
    assert output_json.read_bytes() == payload
    assert (tmp_path / "canonical.json.sha256").is_file()
    assert (tmp_path / "canonical.npz.sha256").is_file()

    next_temporary, next_sha256 = legacy._prepare_npz_temporary(
        tmp_path / "next.npz",
        arrays,
    )
    with pytest.raises(FileExistsError):
        legacy._publish_output_set(
            output_json=output_json,
            json_payload=payload,
            output_npz=tmp_path / "next.npz",
            npz_temporary=next_temporary,
            npz_sha256=next_sha256,
        )
    next_temporary.unlink(missing_ok=True)
