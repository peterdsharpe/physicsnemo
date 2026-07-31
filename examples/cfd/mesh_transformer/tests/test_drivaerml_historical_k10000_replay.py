# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the archive-blind historical K=10k replay producer."""

from __future__ import annotations

import ast
import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

import drivaerml_historical_k10000_replay as replay
import drivaerml_historical_k10000_replay_runtime as replay_runtime
import numpy as np
import pytest
import torch

from physicsnemo.mesh import Mesh


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_case_table_is_complete_ordered_and_nonwrapping():
    assert len(replay.EXPECTED_CASE_SPECS) == 36
    assert [row[0] for row in replay.EXPECTED_CASE_SPECS] == list(range(36))
    assert len({row[1] for row in replay.EXPECTED_CASE_SPECS}) == 36
    assert len({row[2] for row in replay.EXPECTED_CASE_SPECS}) == 36
    for _, _, _, n_cells, start in replay.EXPECTED_CASE_SPECS:
        assert 0 <= start
        assert start + replay.RESOLUTION <= n_cells


def test_frozen_input_records_match_bound_hashes():
    results = Path(__file__).parents[1] / "results"
    target = (
        results
        / "historical_k10000_selected_target_input_freeze_job306302_2026-07-28"
        / "artifacts"
        / "historical_k10000_selected_target_input_manifest_v1.json"
    )
    historical = (
        results
        / "phase1_historical_k10000_input_freeze_v1_2026-07-28"
        / "historical_k10000_input_freeze_v1.json"
    )
    assert _sha(target) == replay.EXPECTED_TARGET_INPUT_MANIFEST_SHA256
    assert _sha(historical) == replay.EXPECTED_HISTORICAL_INPUT_FREEZE_SHA256


def test_producer_binds_only_sibling_outcome_free_runtime_helper():
    producer_path = Path(replay.__file__).resolve()
    helper_path = producer_path.with_name(replay.HELPER_FILENAME)
    assert helper_path == Path(replay_runtime.__file__).resolve()
    assert _sha(helper_path) == replay.EXPECTED_HELPER_SHA256
    source = producer_path.read_text()
    assert "--helper" not in source
    assert "drivaerml_trace_fixed_query_audit" not in source


def test_safe_read_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        replay._safe_read_bytes(link)


def test_strict_json_rejects_duplicate_and_nonfinite_tokens():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        replay._strict_json_bytes(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(ValueError, match="Non-finite JSON token"):
        replay._strict_json_bytes(b'{"x":NaN}', "nonfinite")


def test_safe_hashed_pread_binds_full_file_and_selected_bytes(tmp_path):
    values = np.arange(12, dtype="<i8")
    path = tmp_path / "cells.memmap"
    path.write_bytes(values.tobytes())
    payload = replay._safe_hashed_pread(
        path,
        expected_file_size=len(values.tobytes()),
        expected_file_sha256=_sha(path),
        offset=3 * 8,
        count=4 * 8,
    )
    assert payload == values[3:7].tobytes()
    with pytest.raises(ValueError, match="SHA-256 changed"):
        replay._safe_hashed_pread(
            path,
            expected_file_size=len(values.tobytes()),
            expected_file_sha256="0" * 64,
            offset=0,
            count=8,
        )


def test_safe_hashed_rows_reads_sorted_unique_rows(tmp_path):
    points = np.arange(30, dtype="<f4").reshape(10, 3)
    path = tmp_path / "points.memmap"
    path.write_bytes(points.tobytes())
    selected = replay._safe_hashed_rows(
        path,
        expected_file_size=len(points.tobytes()),
        expected_file_sha256=_sha(path),
        n_rows=10,
        row_indices=np.array([1, 4, 9], dtype=np.int64),
    )
    assert selected.tobytes() == points[[1, 4, 9]].tobytes()
    with pytest.raises(ValueError, match="sorted, and unique"):
        replay._safe_hashed_rows(
            path,
            expected_file_size=len(points.tobytes()),
            expected_file_sha256=_sha(path),
            n_rows=10,
            row_indices=np.array([4, 1], dtype=np.int64),
        )


def test_safe_selected_target_binds_range_hash_and_signed_zero(tmp_path):
    values = np.array([1.0, 0.0, -0.0, 2.0], dtype="<f4")
    path = tmp_path / "target.memmap"
    path.write_bytes(values.tobytes())
    selected = values[1:3].tobytes()
    assert (
        replay._safe_selected_target(
            path,
            expected_file_size=len(values.tobytes()),
            offset=4,
            count=8,
            expected_selected_sha256=hashlib.sha256(selected).hexdigest(),
        )
        == selected
    )
    with pytest.raises(ValueError, match="Selected target bytes changed"):
        replay._safe_selected_target(
            path,
            expected_file_size=len(values.tobytes()),
            offset=4,
            count=8,
            expected_selected_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )


def _synthetic_explicit_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "RESOLUTION", 2)
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
    cell_data = tensor_root / "cell_data"
    cell_data.mkdir(parents=True)
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
    pressure = np.arange(4, dtype="<f4")
    wss = np.arange(12, dtype="<f4").reshape(4, 3)
    (tensor_root / "points.memmap").write_bytes(points.tobytes())
    (tensor_root / "cells.memmap").write_bytes(cells.tobytes())
    (cell_data / "pMeanTrim.memmap").write_bytes(pressure.tobytes())
    (cell_data / "wallShearStressMeanTrim.memmap").write_bytes(wss.tobytes())
    base = "domain_run_x.pdmsh/_tensordict/boundaries/vehicle/_tensordict"
    geometry = {
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
    target = {
        "case_id": "run_x",
        "resolved_case_root": str(case_root),
        "selected_targets": {},
    }
    for name, field, values_, components in (
        ("pressure", "pMeanTrim", pressure, 1),
        ("wss", "wallShearStressMeanTrim", wss, 3),
    ):
        selected = values_[1:3].tobytes()
        target["selected_targets"][name] = {
            "source_relative_path": f"{base}/cell_data/{field}.memmap",
            "source_size_bytes": len(values_.tobytes()),
            "source_offset_bytes": components * 4,
            "selected_size_bytes": 2 * components * 4,
            "selected_sha256": hashlib.sha256(selected).hexdigest(),
        }
    spec = SimpleNamespace(
        case_id="run_x",
        historical_start=1,
        n_master_cells=4,
    )
    return dataset, geometry, target, spec, pressure, wss


def test_explicit_loader_opens_only_geometry_and_selected_targets(
    tmp_path, monkeypatch
):
    dataset, geometry, target, spec, pressure, wss = _synthetic_explicit_subset(
        tmp_path, monkeypatch
    )
    opened: list[str] = []
    original = replay._open_regular

    def tracked(path, *, expected_size):
        opened.append(Path(path).name)
        return original(path, expected_size=expected_size)

    monkeypatch.setattr(replay, "_open_regular", tracked)
    mesh, arrays, hashes = replay._load_explicit_raw_subset(
        SimpleNamespace(mesh_type=Mesh),
        dataset,
        spec,
        geometry,
        target,
    )
    assert opened == [
        "cells.memmap",
        "points.memmap",
        "pMeanTrim.memmap",
        "wallShearStressMeanTrim.memmap",
    ]
    assert mesh.n_cells == 2
    assert mesh.n_points == 4
    assert arrays["selected_cell_ids_int64"].tolist() == [1, 2]
    assert arrays["raw_target_pressure_float32"].tobytes() == pressure[1:3].tobytes()
    assert arrays["raw_target_wss_float32"].tobytes() == wss[1:3].tobytes()
    assert hashes == {
        "pressure_selected_sha256": hashlib.sha256(pressure[1:3].tobytes()).hexdigest(),
        "wss_selected_sha256": hashlib.sha256(wss[1:3].tobytes()).hexdigest(),
    }
    assert "_measure_weights" not in mesh.cell_data.keys()
    assert mesh.global_data["U_inf"].shape == (3,)
    for name in ("p_inf", "rho_inf", "nu", "L_ref"):
        assert mesh.global_data[name].shape == ()


def test_pipeline_globals_have_fixed_order():
    global_data = {
        "U_inf": torch.tensor([1.0, 2.0, 3.0]),
        "p_inf": torch.tensor(4.0),
        "rho_inf": torch.tensor(5.0),
        "nu": torch.tensor(6.0),
        "L_ref": torch.tensor(7.0),
        "U_inf_dir": torch.tensor([8.0, 9.0, 10.0]),
        "reference_length": torch.tensor(11.0),
    }
    observed = replay._pipeline_globals_float32(
        SimpleNamespace(global_data=global_data)
    )
    assert observed.tolist() == list(range(1, 12))
    assert len(replay.GLOBAL_FIELD_ORDER) == 11


def test_pipeline_globals_reject_length_one_scalar_regression():
    global_data = {
        "U_inf": torch.tensor([1.0, 0.0, 0.0]),
        "p_inf": torch.tensor([0.0]),
        "rho_inf": torch.tensor(1.0),
        "nu": torch.tensor(1.0),
        "L_ref": torch.tensor(5.0),
        "U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
        "reference_length": torch.tensor(8.0),
    }
    with pytest.raises(ValueError, match="p_inf.*expected=\\(\\)"):
        replay._pipeline_globals_float32(SimpleNamespace(global_data=global_data))


def test_full_current_transform_and_collate_preserve_scalar_model_contract(
    tmp_path,
    monkeypatch,
):
    dataset, geometry, target, spec, _, _ = _synthetic_explicit_subset(
        tmp_path, monkeypatch
    )
    raw_mesh, _, _ = replay._load_explicit_raw_subset(
        SimpleNamespace(mesh_type=Mesh),
        dataset,
        spec,
        geometry,
        target,
    )

    recipe_src = (
        Path(__file__).parents[2]
        / "external_aerodynamics"
        / "unified_external_aero_recipe"
        / "src"
    )
    monkeypatch.syspath_prepend(str(recipe_src))
    from collate import build_collate_fn
    from domain_transforms import ComputeFreestreamDirection
    from nondim import NonDimensionalizeByMetadata

    from physicsnemo.datapipes.transforms.mesh import (
        CenterMesh,
        ComputeSurfaceNormals,
        DropMeshFields,
        MeshToDomainMesh,
        NormalizeMeshFields,
        RenameMeshFields,
        SetGlobalField,
        SubsampleMesh,
    )

    transforms = [
        DropMeshFields(global_data=["TimeValue"]),
        ComputeFreestreamDirection(),
        CenterMesh(use_area_weighting=False),
        NonDimensionalizeByMetadata(
            fields={
                "pMeanTrim": "pressure",
                "wallShearStressMeanTrim": "stress",
            },
            association="cell_data",
        ),
        RenameMeshFields(
            cell_data={
                "pMeanTrim": "pressure",
                "wallShearStressMeanTrim": "wss",
            }
        ),
        NormalizeMeshFields(
            association="cell_data",
            fields={
                "wss": {
                    "type": "vector",
                    "mean": [0.0, 0.0, 0.0],
                    "std": 0.00313,
                }
            },
        ),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        SubsampleMesh(n_cells=2),
        SetGlobalField(fields={"reference_length": 8.0}),
        MeshToDomainMesh(
            cell_data_targets=["pressure", "wss"],
            interior_points="cell_centroids",
            boundary_name="vehicle",
        ),
    ]
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dataset=SimpleNamespace(transforms=transforms),
    )
    domain, _ = replay_runtime._apply_pipeline(
        runtime,
        raw_mesh,
        fixed_center=None,
    )
    observed = replay._pipeline_globals_float32(domain)
    assert observed.shape == (11,)
    assert domain.global_data["U_inf"].shape == (3,)
    assert domain.global_data["U_inf_dir"].shape == (3,)
    for name in ("p_inf", "rho_inf", "nu", "L_ref", "reference_length"):
        assert domain.global_data[name].shape == ()
    assert "_measure_weights" not in domain.boundaries["vehicle"].cell_data.keys()

    collate = build_collate_fn(
        input_type="mesh",
        forward_kwargs_spec={"domain": ""},
        target_config=replay.TARGET_CONFIG,
    )
    batch = collate([(domain, {})])
    assert set(batch["forward_kwargs"]) == {"domain"}
    assert batch["forward_kwargs"]["domain"] is domain
    assert "target_measure" in batch
    assert "_target_quadrature_measure" in domain.interior.point_data.keys()


class _DomainOnlyModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, domain):
        del domain
        self.calls += 1
        return {
            "pressure": torch.tensor([2.0, 3.0], dtype=torch.float32),
            "wss": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.float32,
            ),
        }


def test_forward_calls_domain_only_once(monkeypatch):
    monkeypatch.setattr(replay, "RESOLUTION", 2)
    model = _DomainOnlyModel()
    targets = {
        "pressure": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "wss": torch.zeros((2, 3), dtype=torch.float32),
    }
    runtime = SimpleNamespace(
        collate_fn=lambda samples: {
            "forward_kwargs": {"domain": samples[0][0]},
            "targets": targets,
        },
        autocast_context=lambda precision: contextlib.nullcontext(),
        cfg=SimpleNamespace(precision="bfloat16", output_type="mesh"),
        model=model,
        normalize_output=lambda output, target_config, output_type: output,
    )
    domain = SimpleNamespace(
        global_data={
            "U_inf": torch.tensor([1.0, 0.0, 0.0]),
            "p_inf": torch.tensor(0.0),
            "rho_inf": torch.tensor(1.0),
            "nu": torch.tensor(1.0),
            "L_ref": torch.tensor(5.0),
            "U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
            "reference_length": torch.tensor(8.0),
        },
        boundaries={"vehicle": SimpleNamespace(cell_data={})},
    )

    def identity_redim(td, **kwargs):
        del kwargs
        return td

    result = replay._run_forward(
        runtime,
        domain,
        {
            "function": identity_redim,
            "normalizer": object(),
            "nondim": object(),
            "field_types": replay.TARGET_CONFIG,
        },
    )
    assert model.calls == 1
    assert result["prediction_pressure_physical"].tobytes() == (
        result["prediction_pressure_training"].tobytes()
    )


def test_forward_rejects_canonical_geometry_in_collated_kwargs(monkeypatch):
    monkeypatch.setattr(replay, "RESOLUTION", 2)
    runtime = SimpleNamespace(
        collate_fn=lambda samples: {
            "forward_kwargs": {
                "domain": samples[0][0],
                "canonical_source_geometry": object(),
            },
            "targets": {},
        }
    )
    with pytest.raises(ValueError, match="forward kwargs changed"):
        replay._run_forward(runtime, object(), {})


def test_output_target_validation_rejects_alias_and_existing(tmp_path):
    output = tmp_path / "artifact.json"
    with pytest.raises(ValueError, match="pairwise distinct"):
        replay._validate_output_targets(output, output)
    output.write_text("sentinel")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        replay._validate_output_targets(output)
    assert output.read_text() == "sentinel"


def test_atomic_output_set_publishes_and_refuses_overwrite(tmp_path):
    output_json = tmp_path / "artifact.json"
    output_npz = tmp_path / "artifact.npz"
    arrays = {"x": np.array([0.0, -0.0, 1.0], dtype="<f4")}
    temporary, npz_sha = replay._prepare_npz_temporary(output_npz, arrays)
    payload = b'{"status":"ok"}\n'
    digest = replay._publish_output_set(
        output_json=output_json,
        json_payload=payload,
        output_npz=output_npz,
        npz_temporary=temporary,
        npz_sha256=npz_sha,
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert output_json.read_bytes() == payload
    assert replay._sha256_file(output_npz) == npz_sha
    assert (tmp_path / "artifact.json.sha256").is_file()
    assert (tmp_path / "artifact.npz.sha256").is_file()

    next_temporary, next_sha = replay._prepare_npz_temporary(
        tmp_path / "next.npz",
        arrays,
    )
    with pytest.raises(FileExistsError):
        replay._publish_output_set(
            output_json=output_json,
            json_payload=payload,
            output_npz=tmp_path / "next.npz",
            npz_temporary=next_temporary,
            npz_sha256=next_sha,
        )
    next_temporary.unlink(missing_ok=True)


def test_source_has_one_domain_only_model_call_and_no_oracle_cli():
    source = Path(replay.__file__).read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model"
    ]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == [None]
    assert "canonical_source_geometry=None" not in source
    assert "--historical-predictions" not in source
    assert "--historical-metrics" not in source
    assert "_load_sample" not in source
    assert "preliminary_outcome" not in source
    assert "decision_outcome" not in source
    assert '"producer_reads_archive_or_metrics": False' in source


def test_array_manifest_hashes_signed_zero_words():
    positive = np.array([0.0], dtype="<f4")
    negative = np.array([-0.0], dtype="<f4")
    manifest = replay._array_manifest({"positive": positive, "negative": negative})
    assert manifest["positive"]["sha256"] != manifest["negative"]["sha256"]
