# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the full-cohort canonical-geometry validity producer."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import drivaerml_hqc_canonical_geometry_validity as validity_module
import numpy as np
import pytest
import torch
from drivaerml_hqc_canonical_geometry_validity import (
    ALLOWED_RAW_GLOBAL_FIELDS,
    ANCHOR_CASE_IDS,
    ANCHOR_FULL_AND_DERIVED_OUTCOME,
    CANONICAL_FULL_OUTCOME,
    CANONICAL_REPAIR_REFUTED,
    CASE_IDS,
    EXPECTED_DATASET_MANIFEST_SHA256,
    INVALID_DIAGNOSTIC,
    INVALID_STATUS,
    LANE_COUNT,
    PRECISIONS,
    QUERY_PANELS,
    RESOLUTIONS,
    VALID_STATUS,
    _anchor_replay,
    _decision_outcome,
    _expected_geometry_manifest_paths,
    _expected_unit_array_names,
    _forbidden_artifact_keys,
    _lane_specs,
    _prepare_npz_temporary,
    _publish_output_set_no_clobber,
    _run_full_mode,
    _run_resolution,
    _strict_json,
    _target_free_file_subset,
    _target_free_subset,
    _unit_array_prefix,
    _validate_anchor_summary,
    _validate_geometry_manifest,
    _validate_import_provenance,
    _validate_output_targets,
    _validate_reader_target_free,
)
from drivaerml_trace_fixed_query_audit import _cyclic_indices


def _fake_hqc(case_ids=CASE_IDS):
    return SimpleNamespace(
        CASE_SPECS=tuple(
            SimpleNamespace(case_id=case_id, cohort_ordinal=ordinal)
            for ordinal, case_id in enumerate(case_ids)
        )
    )


class _ArrayHelper:
    @staticmethod
    def _array_bitwise_equal(left, right):
        left = np.asarray(left)
        right = np.asarray(right)
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.ascontiguousarray(left).tobytes()
            == np.ascontiguousarray(right).tobytes()
        )


class _TrapRawMesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = torch.tensor([[0, 1, 2], [0, 3, 1]])
    n_cells = 2
    n_spatial_dims = 3
    global_data = {
        "U_inf": torch.tensor([1.0, 0.0, 0.0]),
        "p_inf": torch.tensor(1.0),
        "rho_inf": torch.tensor(1.0),
        "nu": torch.tensor(1.0),
        "L_ref": torch.tensor(5.0),
        "forbidden_extra": torch.tensor(999.0),
    }

    @property
    def cell_data(self):
        raise AssertionError("raw local supervision association was accessed")

    @property
    def point_data(self):
        raise AssertionError("raw local supervision association was accessed")


class _CapturedMesh:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FullHelper:
    @staticmethod
    def _tensor_bitwise_equal(left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )

    @classmethod
    def _injected_geometry_exact(cls, encoded, bundle, mode):
        assert mode == "canonical_full"
        return {
            "points": cls._tensor_bitwise_equal(
                encoded.source_mesh.points.cpu(),
                bundle.points,
            ),
            "cells": cls._tensor_bitwise_equal(
                encoded.source_mesh.cells.cpu(),
                bundle.cells,
            ),
            "centroids": cls._tensor_bitwise_equal(
                encoded.source_mesh.cell_centroids.cpu(),
                bundle.centroids,
            ),
            "areas": cls._tensor_bitwise_equal(
                encoded.source_mesh.cell_areas.cpu(),
                bundle.areas,
            ),
            "normals": cls._tensor_bitwise_equal(
                encoded.source_mesh.cell_normals.cpu(),
                bundle.normals,
            ),
        }

    @staticmethod
    def _extract_prediction(output, n_queries):
        assert output.point_data["pressure"].shape == (n_queries,)
        assert output.point_data["wss"].shape == (n_queries, 3)
        return {
            name: value.detach().float().clone()
            for name, value in output.point_data.items()
        }

    @classmethod
    def _prediction_difference(cls, left, right):
        return {
            field: {
                "exact": cls._tensor_bitwise_equal(left[field], right[field]),
            }
            for field in ("pressure", "wss")
        }

    @staticmethod
    def _difference_is_exact(difference):
        return all(value["exact"] for value in difference.values())


class _FakeCanonicalModel:
    boundary_names = ("vehicle",)

    def __init__(
        self,
        *,
        fixed_delta=0.0,
        replay_delta=0.0,
        center_value=0.0,
        negative_zero_center=False,
    ):
        self.fixed_delta = fixed_delta
        self.replay_delta = replay_delta
        self.center_value = center_value
        self.negative_zero_center = negative_zero_center
        self.encode_calls = []
        self.decode_query_counts = []

    def encode(self, domain, *, canonical_source_geometry):
        self.encode_calls.append(canonical_source_geometry)
        call_ordinal = len(self.encode_calls)
        delta = self.fixed_delta if domain.label == "fixed" else 0.0
        if call_ordinal == 3:
            delta = self.replay_delta
        center = canonical_source_geometry.center
        if self.center_value != 0.0:
            center = torch.full_like(center, self.center_value)
        elif self.negative_zero_center:
            center = -torch.zeros_like(center)
        source_mesh = SimpleNamespace(
            points=canonical_source_geometry.points,
            cells=canonical_source_geometry.cells,
            cell_centroids=canonical_source_geometry.centroids,
            cell_areas=canonical_source_geometry.areas,
            cell_normals=canonical_source_geometry.normals,
        )
        return SimpleNamespace(
            source_mesh=source_mesh,
            center=center,
            reference_length=canonical_source_geometry.reference_length,
            trace_slice=slice(0, canonical_source_geometry.cells.shape[0]),
            output_delta=delta,
        )

    def decode(self, encoded, query_mesh):
        count = query_mesh.points.shape[0]
        self.decode_query_counts.append(count)
        assert count == encoded.trace_slice.stop - encoded.trace_slice.start
        base = torch.arange(count, dtype=query_mesh.points.dtype)
        pressure = base + encoded.output_delta
        wss = torch.stack((base, base + 1.0, base + 2.0), dim=-1)
        wss = wss + encoded.output_delta
        return SimpleNamespace(
            point_data={
                "pressure": pressure,
                "wss": wss,
            }
        )


def _fake_full_runtime(*, model, resolution=5_000):
    cells = torch.zeros((resolution, 3), dtype=torch.long)
    points = torch.zeros((resolution + 2, 3), dtype=torch.float32)
    boundary = SimpleNamespace(
        points=points,
        cells=cells,
        n_spatial_dims=3,
    )
    primary = SimpleNamespace(
        label="primary",
        boundaries={"vehicle": boundary},
    )
    fixed = SimpleNamespace(
        label="fixed",
        boundaries={"vehicle": boundary},
    )
    normals = torch.zeros((resolution, 3), dtype=torch.float32)
    normals[:, 0] = 1.0
    bundle = SimpleNamespace(
        points=points.clone(),
        cells=cells.clone(),
        centroids=torch.arange(resolution * 3, dtype=torch.float32).reshape(
            resolution,
            3,
        ),
        areas=torch.ones(resolution, dtype=torch.float32),
        normals=normals,
    )
    runtime = SimpleNamespace(model=model, mesh_type=_CapturedMesh)
    return runtime, primary, fixed, bundle


class _ShaHelper:
    @staticmethod
    def _sha256_file(file_path):
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    @classmethod
    def _require_sha256(cls, file_path, expected, _label):
        observed = cls._sha256_file(file_path)
        if observed != expected:
            raise ValueError(f"SHA mismatch: expected {expected}, got {observed}")


def _synthetic_geometry_manifest(tmp_path, mutation=None):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    case_source = (tmp_path / "source" / "run_118").resolve()
    case_source.mkdir(parents=True)
    case_link = dataset_root / "run_118"
    case_link.symlink_to(case_source, target_is_directory=True)

    files = {}
    for relative in sorted(_expected_geometry_manifest_paths("run_118")):
        file_path = case_source / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(relative.encode("utf-8"))
        files[relative] = {
            "size_bytes": file_path.stat().st_size,
            "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        }
    if mutation is not None:
        removed = sorted(files)[0]
        if mutation in {"missing", "substituted"}:
            files.pop(removed)
        if mutation in {"extra", "substituted"}:
            replacement = "domain_run_118.pdmsh/_tensordict/global_data/extra.memmap"
            replacement_path = case_source / replacement
            replacement_path.write_bytes(b"extra")
            files[replacement] = {
                "size_bytes": replacement_path.stat().st_size,
                "sha256": hashlib.sha256(replacement_path.read_bytes()).hexdigest(),
            }

    spec = SimpleNamespace(
        case_id="run_118",
        cohort_ordinal=0,
        reader_index=21,
        n_master_cells=17_504_739,
        historical_start=14_045_027,
    )
    case_record = {
        "case_id": spec.case_id,
        "cohort_ordinal": spec.cohort_ordinal,
        "reader_index": spec.reader_index,
        "n_master_cells": spec.n_master_cells,
        "historical_start": spec.historical_start,
        "symlink_target": str(case_source),
        "resolved_case_root": str(case_source),
        "files": files,
    }
    cases = [{"case_id": case_id} for case_id in CASE_IDS]
    cases[0] = case_record
    manifest = {
        "schema_version": 1,
        "artifact_kind": "drivaerml_target_free_geometry_input_manifest",
        "status": "PASSED_TARGET_FREE_GEOMETRY_INPUT_FREEZE",
        "case_count": len(CASE_IDS),
        "cases": cases,
        "dataset_root_resolved": str(dataset_root.resolve()),
        "dataset_manifest": {"sha256": EXPECTED_DATASET_MANIFEST_SHA256},
        "target_exclusion": {
            "point_data_opened": False,
            "cell_data_opened": False,
            "interior_opened": False,
            "supervision_values_opened": False,
            "supervision_values_hashed": False,
            "supervision_values_serialized": False,
        },
    }
    manifest_path = tmp_path / "geometry_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_root, manifest_path, spec


def _synthetic_geometry_files(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    case_source = (tmp_path / "source" / "run_x").resolve()
    vehicle_root = (
        case_source
        / "domain_run_x.pdmsh"
        / "_tensordict"
        / "boundaries"
        / "vehicle"
        / "_tensordict"
    )
    global_root = case_source / "domain_run_x.pdmsh" / "_tensordict" / "global_data"
    vehicle_root.mkdir(parents=True)
    global_root.mkdir(parents=True)
    vehicle_root.joinpath("meta.json").write_text(
        json.dumps(
            {
                "points": {"shape": [4, 3], "dtype": "torch.float32"},
                "cells": {"shape": [2, 3], "dtype": "torch.int64"},
            }
        ),
        encoding="utf-8",
    )
    np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="<f4",
    ).tofile(vehicle_root / "points.memmap")
    np.asarray([[0, 1, 2], [0, 3, 1]], dtype="<i8").tofile(
        vehicle_root / "cells.memmap"
    )
    globals_by_name = {
        "U_inf": np.asarray([1.0, 2.0, 3.0], dtype="<f4"),
        "p_inf": np.asarray([4.0], dtype="<f4"),
        "rho_inf": np.asarray([5.0], dtype="<f4"),
        "nu": np.asarray([6.0], dtype="<f4"),
        "L_ref": np.asarray([5.0], dtype="<f4"),
    }
    for name, values in globals_by_name.items():
        values.tofile(global_root / f"{name}.memmap")
    forbidden = vehicle_root / "cell_data"
    forbidden.mkdir()
    forbidden.joinpath("pMeanTrim.memmap").write_bytes(b"must-not-open")
    forbidden.joinpath("wallShearStressMeanTrim.memmap").write_bytes(b"must-not-open")
    dataset_root.joinpath("run_x").symlink_to(
        case_source,
        target_is_directory=True,
    )
    spec = SimpleNamespace(case_id="run_x", n_master_cells=2)
    return dataset_root, spec


def _anchor_arrays(spec):
    prefix = f"case_{spec.cohort_ordinal:02d}_{spec.case_id}"
    values = {}
    anchor = {}
    for name in sorted(_expected_unit_array_names()):
        anchor_name = name
        for panel in QUERY_PANELS:
            anchor_name = anchor_name.replace(f"_{panel}_", "_", 1)
        key = f"{prefix}__{anchor_name}"
        if key not in anchor:
            anchor[key] = np.array([float(len(anchor))], dtype="<f4")
        values[name] = anchor[key].copy()
    return values, anchor


def test_exact_grid_and_four_lanes_cover_every_case_once():
    assert len(CASE_IDS) == 36
    assert RESOLUTIONS == (2_500, 5_000, 10_000, 20_000, 40_000)
    assert PRECISIONS == ("bfloat16", "float32")

    hqc = _fake_hqc()
    lanes = [_lane_specs(hqc, lane, LANE_COUNT) for lane in range(LANE_COUNT)]
    assert all(len(lane) == 9 for lane in lanes)
    observed = {spec.case_id for lane in lanes for spec in lane}
    assert observed == set(CASE_IDS)
    assert sum(len(lane) for lane in lanes) == len(CASE_IDS)
    for ordinal, lane in enumerate(lanes):
        assert all(spec.cohort_ordinal % LANE_COUNT == ordinal for spec in lane)


def test_subset_never_reads_local_values_and_allowlists_global_inputs():
    subset = _target_free_subset(
        _TrapRawMesh(),
        np.array([1, 0]),
        _CapturedMesh,
    )
    assert set(subset.global_data) == set(ALLOWED_RAW_GLOBAL_FIELDS)
    assert "forbidden_extra" not in subset.global_data
    assert set(subset.cell_data) == {
        "pMeanTrim",
        "wallShearStressMeanTrim",
    }
    assert torch.count_nonzero(subset.cell_data["pMeanTrim"]).item() == 0
    assert torch.count_nonzero(subset.cell_data["wallShearStressMeanTrim"]).item() == 0


def test_file_subset_opens_only_geometry_and_allowed_globals(
    tmp_path,
    monkeypatch,
):
    dataset_root, spec = _synthetic_geometry_files(tmp_path)
    opened = []
    original_memmap = validity_module.np.memmap
    original_fromfile = validity_module.np.fromfile

    def traced_memmap(filename, *args, **kwargs):
        opened.append(Path(filename))
        return original_memmap(filename, *args, **kwargs)

    def traced_fromfile(file, *args, **kwargs):
        opened.append(Path(file))
        return original_fromfile(file, *args, **kwargs)

    monkeypatch.setattr(validity_module.np, "memmap", traced_memmap)
    monkeypatch.setattr(validity_module.np, "fromfile", traced_fromfile)
    subset = _target_free_file_subset(
        dataset_root,
        spec,
        np.asarray([1, 0], dtype=np.int64),
        _CapturedMesh,
    )

    assert len(opened) == 2 + len(ALLOWED_RAW_GLOBAL_FIELDS)
    assert {path.name for path in opened} == {
        "points.memmap",
        "cells.memmap",
        *(f"{name}.memmap" for name in ALLOWED_RAW_GLOBAL_FIELDS),
    }
    assert all("cell_data" not in path.parts for path in opened)
    assert all("point_data" not in path.parts for path in opened)
    assert set(subset.global_data) == set(ALLOWED_RAW_GLOBAL_FIELDS)
    assert subset.global_data["L_ref"].shape == ()
    assert subset.points.shape == (4, 3)
    assert subset.cells.shape == (2, 3)
    assert torch.count_nonzero(subset.cell_data["pMeanTrim"]).item() == 0


def test_file_subset_is_exactly_equivalent_to_in_memory_geometry_subset(tmp_path):
    dataset_root, spec = _synthetic_geometry_files(tmp_path)
    ids = np.asarray([1, 0], dtype=np.int64)
    raw = SimpleNamespace(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.int64),
        n_cells=2,
        n_spatial_dims=3,
        global_data={
            "U_inf": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "p_inf": torch.tensor(4.0, dtype=torch.float32),
            "rho_inf": torch.tensor(5.0, dtype=torch.float32),
            "nu": torch.tensor(6.0, dtype=torch.float32),
            "L_ref": torch.tensor(5.0, dtype=torch.float32),
        },
    )

    direct = _target_free_file_subset(dataset_root, spec, ids, _CapturedMesh)
    in_memory = _target_free_subset(raw, ids, _CapturedMesh)

    for name in ("points", "cells"):
        assert torch.equal(getattr(direct, name), getattr(in_memory, name))
    assert set(direct.global_data) == set(in_memory.global_data)
    for name in direct.global_data:
        assert torch.equal(direct.global_data[name], in_memory.global_data[name])
    assert set(direct.cell_data) == set(in_memory.cell_data)
    for name in direct.cell_data:
        assert torch.equal(direct.cell_data[name], in_memory.cell_data[name])


def test_public_full_mode_decodes_only_full_trace_and_slices_prefix():
    model = _FakeCanonicalModel()
    runtime, primary, fixed, bundle = _fake_full_runtime(model=model)

    summary, arrays = _run_full_mode(
        _FullHelper,
        runtime,
        primary_domain=primary,
        fixed_domain=fixed,
        bundle=bundle,
    )

    assert len(model.encode_calls) == 3
    assert model.decode_query_counts == [5_000, 5_000, 5_000]
    assert summary["validity_passed"]
    assert summary["comparison_gate"]["passed"]
    assert summary["mode"] == "canonical_full_public_api"
    assert len(arrays) == 12
    for path in ("primary", "fixed", "primary_replay"):
        for field in ("pressure", "wss"):
            coupled = arrays[f"canonical_full_coupled_s_k_{path}_{field}"]
            prefix = arrays[f"canonical_full_fixed_id_prefix_s2500_{path}_{field}"]
            assert _FullHelper._tensor_bitwise_equal(prefix, coupled[:2_500])


def test_valid_ab_mismatch_refutes_but_replay_mismatch_invalidates():
    model = _FakeCanonicalModel(fixed_delta=1.0)
    runtime, primary, fixed, bundle = _fake_full_runtime(model=model)
    summary, _ = _run_full_mode(
        _FullHelper,
        runtime,
        primary_domain=primary,
        fixed_domain=fixed,
        bundle=bundle,
    )
    assert summary["validity_passed"]
    assert not summary["comparison_gate"]["passed"]
    assert (
        _decision_outcome(
            validity_passed=summary["validity_passed"],
            full_passed=summary["comparison_gate"]["passed"],
        )
        == CANONICAL_REPAIR_REFUTED
    )

    model = _FakeCanonicalModel(replay_delta=1.0)
    runtime, primary, fixed, bundle = _fake_full_runtime(model=model)
    summary, _ = _run_full_mode(
        _FullHelper,
        runtime,
        primary_domain=primary,
        fixed_domain=fixed,
        bundle=bundle,
    )
    assert not summary["validity_passed"]
    assert summary["comparison_gate"]["passed"]
    assert (
        _decision_outcome(
            validity_passed=summary["validity_passed"],
            full_passed=summary["comparison_gate"]["passed"],
        )
        == INVALID_DIAGNOSTIC
    )


@pytest.mark.parametrize(
    ("center_value", "negative_zero"),
    [(1.0, False), (0.0, True)],
)
def test_public_full_mode_does_not_repair_invalid_encoded_center(
    center_value,
    negative_zero,
):
    model = _FakeCanonicalModel(
        center_value=center_value,
        negative_zero_center=negative_zero,
    )
    runtime, primary, fixed, bundle = _fake_full_runtime(model=model)

    summary, _ = _run_full_mode(
        _FullHelper,
        runtime,
        primary_domain=primary,
        fixed_domain=fixed,
        bundle=bundle,
    )

    assert not summary["validity_passed"]
    assert not summary["canonical_decode_contract_passed"]


def test_panel_counts_are_labeled_without_inflating_licensing_gate():
    licensing = len(CASE_IDS) * len(RESOLUTIONS) * len(PRECISIONS) * 2
    deduplicated_panel_summaries = (
        len(CASE_IDS) * len(PRECISIONS) * 2 * (1 + (len(RESOLUTIONS) - 1) * 2)
    )
    emitted_panel_records = len(CASE_IDS) * len(RESOLUTIONS) * len(PRECISIONS) * 2 * 2
    assert licensing == 720
    assert deduplicated_panel_summaries == 1_296
    assert emitted_panel_records == 1_440
    assert licensing // LANE_COUNT == 180
    assert deduplicated_panel_summaries // LANE_COUNT == 324


def test_run469_k40000_wrap_preserves_frozen_prefix():
    n_cells = 19_780_049
    start = 19_757_508
    full = _cyclic_indices(n_cells, start, 40_000)
    frozen_prefix = _cyclic_indices(n_cells, start, 2_500)

    assert _ArrayHelper._array_bitwise_equal(full[:2_500], frozen_prefix)
    assert full[0] == start
    assert full[n_cells - start] == 0
    assert full[-1] == 17_458


def test_import_provenance_is_pinned_to_requested_checkout():
    repo_root = Path(validity_module.__file__).resolve().parents[4]
    provenance = _validate_import_provenance(repo_root)

    assert Path(provenance["physicsnemo_init"]) == (
        repo_root / "physicsnemo" / "__init__.py"
    )
    assert Path(provenance["mesh_attention_model"]) == (
        repo_root
        / "physicsnemo"
        / "experimental"
        / "nn"
        / "mesh_attention"
        / "model.py"
    )


def test_target_free_reader_validation_uses_vehicle_reader_root(tmp_path):
    spec = SimpleNamespace(
        case_id="run_118",
        reader_index=21,
        n_master_cells=17_504_739,
    )
    vehicle_root = tmp_path / "run_118" / "boundaries" / "vehicle"
    tensor_root = vehicle_root / "_tensordict"
    tensor_root.mkdir(parents=True)
    (tensor_root / "meta.json").write_text(
        json.dumps({"cells": {"shape": [spec.n_master_cells, 3]}}),
        encoding="utf-8",
    )
    paths = [tmp_path / f"unused_{index}" for index in range(484)]
    paths[spec.reader_index] = vehicle_root
    runtime = SimpleNamespace(
        dataset=SimpleNamespace(reader=SimpleNamespace(_paths=paths))
    )
    hqc = SimpleNamespace(CASE_SPECS=(spec,))

    _validate_reader_target_free(runtime, hqc)


def test_main_rejects_unverified_executable_before_import(
    tmp_path,
    monkeypatch,
):
    producer = tmp_path / "producer.py"
    helper = tmp_path / "helper.py"
    producer.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    helper.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    dataset_root = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoint"
    for directory in (repo_root, dataset_root, checkpoint_dir):
        directory.mkdir()
    input_files = {
        "dataset_config": tmp_path / "dataset.yaml",
        "resolved_config": tmp_path / "resolved.yaml",
        "geometry_manifest": tmp_path / "geometry.json",
        "anchor_json": tmp_path / "anchor.json",
        "anchor_npz": tmp_path / "anchor.npz",
    }
    for path in input_files.values():
        path.write_bytes(b"placeholder")
    imported: list[Path] = []

    def record_import(path, name):
        del name
        imported.append(path)
        raise AssertionError("Unverified executable was imported")

    monkeypatch.setattr(validity_module, "_load_module", record_import)
    with pytest.raises(ValueError, match="Frozen H-QC producer SHA-256 differs"):
        validity_module.main(
            [
                "--producer",
                str(producer),
                "--canonical-helper",
                str(helper),
                "--repo-root",
                str(repo_root),
                "--dataset-root",
                str(dataset_root),
                "--dataset-config",
                str(input_files["dataset_config"]),
                "--resolved-config",
                str(input_files["resolved_config"]),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--geometry-input-manifest",
                str(input_files["geometry_manifest"]),
                "--anchor-json",
                str(input_files["anchor_json"]),
                "--anchor-npz",
                str(input_files["anchor_npz"]),
                "--lane-ordinal",
                "0",
                "--lane-count",
                str(LANE_COUNT),
                "--output-json",
                str(tmp_path / "lane.json"),
                "--output-npz",
                str(tmp_path / "lane.npz"),
            ]
        )

    assert imported == []


def test_main_publishes_explicit_invalid_lane_before_nonzero_exit(
    tmp_path,
    monkeypatch,
):
    producer = tmp_path / "producer.py"
    helper_path = tmp_path / "helper.py"
    producer.write_text("# frozen producer fixture\n", encoding="utf-8")
    helper_path.write_text("# frozen helper fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        validity_module,
        "EXPECTED_PRODUCER_SHA256",
        hashlib.sha256(producer.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        validity_module,
        "EXPECTED_HELPER_SHA256",
        hashlib.sha256(helper_path.read_bytes()).hexdigest(),
    )

    repo_root = tmp_path / "repo"
    dataset_root = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoint"
    for directory in (repo_root, dataset_root, checkpoint_dir):
        directory.mkdir()
    input_files = {
        "dataset_config": tmp_path / "dataset.yaml",
        "resolved_config": tmp_path / "resolved.yaml",
        "geometry_manifest": tmp_path / "geometry.json",
        "anchor_json": tmp_path / "anchor.json",
        "anchor_npz": tmp_path / "anchor.npz",
    }
    for name, path in input_files.items():
        if name != "anchor_npz":
            path.write_text("{}\n", encoding="utf-8")
    np.savez(input_files["anchor_npz"])

    spec = SimpleNamespace(
        case_id=ANCHOR_CASE_IDS[0],
        cohort_ordinal=0,
        reader_index=21,
    )
    replay = {
        "required": True,
        "passed": False,
        "compared_arrays": 0,
        "not_executed_reason": "model_preflight_validity_failed",
    }
    unit = {
        "case_id": spec.case_id,
        "cohort_ordinal": spec.cohort_ordinal,
        "reader_index": spec.reader_index,
        "resolution": RESOLUTIONS[0],
        "validity": {
            "job305691_anchor_replay": replay,
            "model_probes_executed": False,
        },
        "precision_probes": {},
        "validity_passed": False,
        "decision_gates": {"full_passed": False},
        "decision_outcome": INVALID_DIAGNOSTIC,
    }
    case = {
        "case_id": spec.case_id,
        "cohort_ordinal": spec.cohort_ordinal,
        "reader_index": spec.reader_index,
        "resolutions": [unit],
        "validity_passed": False,
        "decision_gates": {"full_passed": False},
        "decision_outcome": INVALID_DIAGNOSTIC,
    }
    prefix = _unit_array_prefix(spec, RESOLUTIONS[0])
    lane_arrays = {
        f"{prefix}__selected_cell_ids_int64": np.arange(3, dtype="<i8"),
        f"{prefix}__canonical_cells_int64": np.zeros((3, 3), dtype="<i8"),
        f"{prefix}__canonical_points_float32": np.zeros((3, 3), dtype="<f4"),
        f"{prefix}__canonical_centroids_float32": np.zeros((3, 3), dtype="<f4"),
        f"{prefix}__canonical_areas_float32": np.ones(3, dtype="<f4"),
        f"{prefix}__canonical_normals_float32": np.zeros((3, 3), dtype="<f4"),
    }
    hqc = SimpleNamespace(
        _require_sha256=lambda *_args: None,
        _load_runtime=lambda **_kwargs: SimpleNamespace(),
    )
    helper = SimpleNamespace(_array_manifest=lambda *_args: {})

    def load_fixture(_path, name):
        return hqc if name == "frozen_hqc_producer" else helper

    monkeypatch.setattr(validity_module, "_load_module", load_fixture)
    monkeypatch.setattr(validity_module, "_lane_specs", lambda *_args: (spec,))
    monkeypatch.setattr(validity_module, "_validate_static_inputs", lambda *_args: None)
    monkeypatch.setattr(
        validity_module,
        "_validate_geometry_manifest",
        lambda *_args, **_kwargs: {"fixture": True},
    )
    monkeypatch.setattr(
        validity_module, "_validate_anchor_summary", lambda *_args: None
    )
    monkeypatch.setattr(
        validity_module,
        "_validate_reader_target_free",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        validity_module,
        "_run_case",
        lambda *_args, **_kwargs: (case, lane_arrays),
    )
    monkeypatch.setattr(
        validity_module,
        "_provenance",
        lambda **_kwargs: {"fixture": True},
    )

    output_json = tmp_path / "lane.json"
    output_npz = tmp_path / "lane.npz"
    with pytest.raises(RuntimeError, match="failed a validity gate"):
        validity_module.main(
            [
                "--producer",
                str(producer),
                "--canonical-helper",
                str(helper_path),
                "--repo-root",
                str(repo_root),
                "--dataset-root",
                str(dataset_root),
                "--dataset-config",
                str(input_files["dataset_config"]),
                "--resolved-config",
                str(input_files["resolved_config"]),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--geometry-input-manifest",
                str(input_files["geometry_manifest"]),
                "--anchor-json",
                str(input_files["anchor_json"]),
                "--anchor-npz",
                str(input_files["anchor_npz"]),
                "--lane-ordinal",
                "0",
                "--lane-count",
                str(LANE_COUNT),
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
            ]
        )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["status"] == INVALID_STATUS
    assert summary["decision_outcome"] == INVALID_DIAGNOSTIC
    assert (
        summary["cases"][0]["resolutions"][0]["validity"]["job305691_anchor_replay"]
        == replay
    )
    with np.load(output_npz, allow_pickle=False) as archive:
        assert set(archive.files) == set(lane_arrays)
    for artifact in (
        output_json,
        output_json.with_name("lane.json.sha256"),
        output_npz,
        output_npz.with_name("lane.npz.sha256"),
    ):
        assert artifact.is_file()
    assert not list(tmp_path.glob(".*.tmp"))


def test_geometry_manifest_requires_exact_case_file_allowlist(
    tmp_path,
    monkeypatch,
):
    dataset_root, manifest_path, spec = _synthetic_geometry_manifest(tmp_path)
    monkeypatch.setattr(
        validity_module,
        "EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    result = _validate_geometry_manifest(
        _ShaHelper,
        path=manifest_path,
        dataset_root=dataset_root.resolve(),
        lane_specs=(spec,),
    )

    assert result["lane_cases_verified"] == 1
    assert result["lane_files_verified"] == 13


@pytest.mark.parametrize("mutation", ["missing", "extra", "substituted"])
def test_geometry_manifest_rejects_rehashed_file_inventory_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    dataset_root, manifest_path, spec = _synthetic_geometry_manifest(
        tmp_path,
        mutation=mutation,
    )
    monkeypatch.setattr(
        validity_module,
        "EXPECTED_GEOMETRY_INPUT_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="file inventory changed"):
        _validate_geometry_manifest(
            _ShaHelper,
            path=manifest_path,
            dataset_root=dataset_root.resolve(),
            lane_specs=(spec,),
        )


def test_lane_contract_rejects_count_bounds_and_cohort_drift():
    hqc = _fake_hqc()
    with pytest.raises(ValueError, match="frozen value"):
        _lane_specs(hqc, 0, LANE_COUNT + 1)
    with pytest.raises(ValueError, match="outside"):
        _lane_specs(hqc, LANE_COUNT, LANE_COUNT)
    with pytest.raises(ValueError, match="cohort or ordering"):
        _lane_specs(_fake_hqc(tuple(reversed(CASE_IDS))), 0, LANE_COUNT)


def test_unit_array_schema_is_exact_and_anchor_replay_is_raw_byte_strict():
    assert len(_expected_unit_array_names()) == 30
    spec = SimpleNamespace(case_id=ANCHOR_CASE_IDS[0], cohort_ordinal=0)
    arrays, anchor = _anchor_arrays(spec)

    replay = _anchor_replay(
        _ArrayHelper,
        spec=spec,
        resolution=RESOLUTIONS[0],
        arrays=arrays,
        anchor_arrays=anchor,
    )
    assert replay["required"]
    assert replay["passed"]
    assert replay["compared_arrays"] == 30

    signed_zero_name = sorted(arrays)[0]
    arrays[signed_zero_name] = np.array([-0.0], dtype="<f4")
    signed_zero_anchor_name = signed_zero_name
    for panel in QUERY_PANELS:
        signed_zero_anchor_name = signed_zero_anchor_name.replace(
            f"_{panel}_",
            "_",
            1,
        )
    anchor[f"case_00_{spec.case_id}__{signed_zero_anchor_name}"] = np.array(
        [0.0],
        dtype="<f4",
    )
    replay = _anchor_replay(
        _ArrayHelper,
        spec=spec,
        resolution=RESOLUTIONS[0],
        arrays=arrays,
        anchor_arrays=anchor,
    )
    assert not replay["passed"]
    assert replay["comparisons"][signed_zero_name] is False


def test_anchor_replay_is_required_only_for_the_four_overlap_units():
    nonanchor = SimpleNamespace(case_id=CASE_IDS[4], cohort_ordinal=4)
    replay = _anchor_replay(
        _ArrayHelper,
        spec=nonanchor,
        resolution=RESOLUTIONS[0],
        arrays={},
        anchor_arrays={},
    )
    assert replay == {"required": False, "passed": True, "compared_arrays": 0}

    anchor = SimpleNamespace(case_id=ANCHOR_CASE_IDS[0], cohort_ordinal=0)
    replay = _anchor_replay(
        _ArrayHelper,
        spec=anchor,
        resolution=RESOLUTIONS[1],
        arrays={},
        anchor_arrays={},
    )
    assert replay == {"required": False, "passed": True, "compared_arrays": 0}

    with pytest.raises(ValueError, match="exact unit array schema"):
        _anchor_replay(
            _ArrayHelper,
            spec=anchor,
            resolution=RESOLUTIONS[0],
            arrays={},
            anchor_arrays={},
            model_probes_executed=True,
        )


def test_required_anchor_preflight_failure_returns_explicit_invalid_unit(
    monkeypatch,
    tmp_path,
):
    resolution = RESOLUTIONS[0]
    ids = np.arange(resolution, dtype=np.int64)
    subset = SimpleNamespace(
        cells=torch.zeros((resolution, 3), dtype=torch.int64),
        global_data={"L_ref": torch.tensor(1.0)},
    )
    bundle = SimpleNamespace(
        cells=subset.cells.clone(),
        points=torch.zeros((resolution + 2, 3), dtype=torch.float32),
        centroids=torch.zeros((resolution, 3), dtype=torch.float32),
        areas=torch.ones(resolution, dtype=torch.float32),
        normals=torch.zeros((resolution, 3), dtype=torch.float32),
        physical_center=torch.zeros(3, dtype=torch.float64),
        physical_length=1.0,
        model_reference_length=1.0,
    )
    spec = SimpleNamespace(
        case_id=ANCHOR_CASE_IDS[0],
        cohort_ordinal=0,
        reader_index=21,
        n_master_cells=resolution,
        historical_start=0,
    )

    def apply_pipeline(_runtime, _subset, fixed_center):
        center = torch.zeros(3) if fixed_center is None else fixed_center
        return SimpleNamespace(global_data={"L_ref": torch.tensor(1.0)}), center

    helper = SimpleNamespace(
        _array_bitwise_equal=_ArrayHelper._array_bitwise_equal,
        _build_canonical_raw_geometry=lambda _subset: bundle,
        _nested_tensor_value=lambda values, key: values[key],
        _tensor_bitwise_equal=torch.equal,
        _strip_local_data=lambda domain, _mesh_type: domain,
        _finish_canonical_bundle=lambda *_args, **_kwargs: bundle,
        _bundle_difference=lambda _left, _right: {"construction": True},
        _bundle_validity=lambda _bundle, expected_cells: {
            "passed": False,
            "cell_order_preserved": torch.equal(
                _bundle.cells,
                expected_cells,
            ),
        },
        _path_topology_checks=lambda *_args: {"topology": True},
    )
    hqc = SimpleNamespace(
        _cyclic_indices=lambda *_args: ids.copy(),
        _apply_pipeline=apply_pipeline,
    )
    runtime = SimpleNamespace(
        mesh_type=_CapturedMesh,
        model=SimpleNamespace(reference_length_key="L_ref"),
    )
    monkeypatch.setattr(
        validity_module,
        "_target_free_file_subset",
        lambda *_args: subset,
    )

    result, arrays = _run_resolution(
        helper,
        hqc,
        runtime,
        spec=spec,
        dataset_root=tmp_path,
        fixed_center=torch.zeros(3),
        resolution=resolution,
        anchor_arrays={},
    )

    replay = result["validity"]["job305691_anchor_replay"]
    assert replay == {
        "required": True,
        "passed": False,
        "compared_arrays": 0,
        "not_executed_reason": "model_preflight_validity_failed",
    }
    assert result["validity"]["model_probes_executed"] is False
    assert result["precision_probes"] == {}
    assert result["validity_passed"] is False
    assert result["decision_outcome"] == INVALID_DIAGNOSTIC
    assert set(arrays) == {
        "selected_cell_ids_int64",
        "canonical_cells_int64",
        "canonical_points_float32",
        "canonical_centroids_float32",
        "canonical_areas_float32",
        "canonical_normals_float32",
    }


def test_anchor_summary_contract_is_narrow():
    summary = {
        "schema_version": 5,
        "artifact_kind": "hqc_canonical_geometry_diagnostic",
        "status": "VALID_NONDECIDING_CANONICAL_GEOMETRY_DIAGNOSTIC",
        "decision_outcome": ANCHOR_FULL_AND_DERIVED_OUTCOME,
        "scientific_scope": {
            "case_ids": list(ANCHOR_CASE_IDS),
            "resolution": RESOLUTIONS[0],
            "precisions": list(PRECISIONS),
        },
        "validity": {"all_cases_and_precisions_passed": True},
        "decision_gates": {"full": {"passed": True}},
    }
    _validate_anchor_summary(summary)

    invalid = json.loads(json.dumps(summary))
    invalid["decision_outcome"] = CANONICAL_FULL_OUTCOME
    with pytest.raises(ValueError, match="anchor contract changed"):
        _validate_anchor_summary(invalid)


@pytest.mark.parametrize(
    ("validity", "full", "expected"),
    [
        (False, True, INVALID_DIAGNOSTIC),
        (True, True, CANONICAL_FULL_OUTCOME),
        (True, False, CANONICAL_REPAIR_REFUTED),
    ],
)
def test_full_arm_controls_candidate_outcome(
    validity,
    full,
    expected,
):
    assert (
        _decision_outcome(
            validity_passed=validity,
            full_passed=full,
        )
        == expected
    )


def test_strict_json_rejects_duplicates_and_nonfinite_tokens(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite JSON token"):
        _strict_json(nonfinite)


def test_output_targets_reject_files_and_dangling_symlinks(tmp_path):
    clean = tmp_path / "clean.json"
    _validate_output_targets(clean)

    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate_output_targets(existing)

    sidecar_target = tmp_path / "sidecar.json"
    sidecar_target.with_name("sidecar.json.sha256").write_text(
        "digest  sidecar.json\n",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate_output_targets(sidecar_target)

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate_output_targets(dangling)


@pytest.mark.parametrize(
    ("json_relative", "npz_relative"),
    [
        ("lane", "lane"),
        ("nested/../lane", "lane"),
        ("lane.json", "lane.json.sha256"),
        ("lane.npz.sha256", "lane.npz"),
    ],
)
def test_output_targets_reject_identical_and_cross_sidecar_aliases(
    tmp_path,
    json_relative,
    npz_relative,
):
    with pytest.raises(ValueError, match="pairwise distinct"):
        _validate_output_targets(
            tmp_path / json_relative,
            tmp_path / npz_relative,
        )


@pytest.mark.parametrize(
    ("json_relative", "npz_relative"),
    [
        ("lane", "lane"),
        ("lane.json", "lane.json.sha256"),
        ("lane.npz.sha256", "lane.npz"),
    ],
)
def test_output_publication_alias_rejection_does_not_leak_temporaries(
    tmp_path,
    json_relative,
    npz_relative,
):
    output_json = tmp_path / json_relative
    output_npz = tmp_path / npz_relative
    npz_temporary, npz_sha256 = _prepare_npz_temporary(
        output_npz,
        {"probe": np.arange(3, dtype="<i8")},
    )

    with pytest.raises(ValueError, match="pairwise distinct"):
        _publish_output_set_no_clobber(
            output_json=output_json,
            json_payload=b'{"status":"invalid"}\n',
            output_npz=output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )

    assert not npz_temporary.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_output_set_publishes_npz_json_and_matching_sidecars(tmp_path):
    output_json = tmp_path / "lane.json"
    output_npz = tmp_path / "lane.npz"
    npz_temporary, npz_sha256 = _prepare_npz_temporary(
        output_npz,
        {"probe": np.arange(3, dtype="<i8")},
    )
    payload = b'{"status":"valid"}\n'

    json_sha256 = _publish_output_set_no_clobber(
        output_json=output_json,
        json_payload=payload,
        output_npz=output_npz,
        npz_temporary=npz_temporary,
        npz_sha256=npz_sha256,
    )

    assert output_json.read_bytes() == payload
    with np.load(output_npz, allow_pickle=False) as archive:
        assert np.array_equal(archive["probe"], np.arange(3, dtype="<i8"))
    assert (
        output_json.with_name("lane.json.sha256").read_text(encoding="ascii")
        == f"{json_sha256}  lane.json\n"
    )
    assert (
        output_npz.with_name("lane.npz.sha256").read_text(encoding="ascii")
        == f"{npz_sha256}  lane.npz\n"
    )
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("collision_name", ["lane.json", "lane.npz.sha256"])
def test_output_set_rolls_back_without_clobbering_race_collision(
    tmp_path,
    collision_name,
):
    output_json = tmp_path / "lane.json"
    output_npz = tmp_path / "lane.npz"
    npz_temporary, npz_sha256 = _prepare_npz_temporary(
        output_npz,
        {"probe": np.arange(3, dtype="<i8")},
    )
    collision = tmp_path / collision_name
    collision.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        _publish_output_set_no_clobber(
            output_json=output_json,
            json_payload=b'{"status":"valid"}\n',
            output_npz=output_npz,
            npz_temporary=npz_temporary,
            npz_sha256=npz_sha256,
        )

    assert collision.read_bytes() == b"sentinel"
    for candidate in (
        output_json,
        output_json.with_name("lane.json.sha256"),
        output_npz,
        output_npz.with_name("lane.npz.sha256"),
    ):
        if candidate != collision:
            assert not candidate.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_vocabulary_and_prefixes_are_fail_closed():
    assert (
        _forbidden_artifact_keys(
            {
                "status": VALID_STATUS,
                "comparison": {"relative_l2_difference": 0.0},
            }
        )
        == []
    )
    assert _forbidden_artifact_keys(
        {
            "nested": {"target_error": 0.0},
            "truth_available": False,
            "force_metric": 0.0,
        }
    ) == ["nested.target_error", "truth_available", "force_metric"]

    spec = SimpleNamespace(case_id="run_118", cohort_ordinal=0)
    assert _unit_array_prefix(spec, 2_500) == "case_00_run_118__k02500"
    assert _unit_array_prefix(spec, 40_000) == "case_00_run_118__k40000"
