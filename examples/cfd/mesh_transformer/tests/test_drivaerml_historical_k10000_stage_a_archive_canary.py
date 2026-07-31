# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for the target-free Stage-A archive-domain canary."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import drivaerml_historical_k10000_stage_a_archive_canary as producer
import drivaerml_historical_k10000_stage_a_archive_canary_adjudicate as reducer
import numpy as np
import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh


def _write_manifest(
    path: Path,
    payloads: dict[Path, bytes],
) -> tuple[str, dict[str, str]]:
    entries = {
        f"./{relative.as_posix()}": hashlib.sha256(payload).hexdigest()
        for relative, payload in payloads.items()
    }
    lines = [f"{digest}  {name}\n".encode() for name, digest in sorted(entries.items())]
    payload = b"".join(lines)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), entries


def _producer_archive(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    points = np.zeros((producer.BOUNDARY_POINT_COUNT, 3), dtype="<f4")
    cells = np.tile(
        np.array([[0, 1, 2]], dtype="<i8"),
        (producer.RESOLUTION, 1),
    )
    queries = points[cells].mean(axis=1)
    # The real archive differs from a fresh NumPy centroid reduction by up to
    # one float32 ULP. Query points are independently manifest-bound inputs.
    queries[0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    arrays = {
        "archive_boundary_points_float32": points,
        "archive_boundary_cells_int64": cells,
        "archive_query_points_float32": queries,
        "archive_global_L_ref_float32": np.array(5.0, dtype="<f4"),
        "archive_global_U_inf_float32": np.array([3.0, 0.0, 0.0], dtype="<f4"),
        "archive_global_U_inf_dir_float32": np.array([1.0, 0.0, 0.0], dtype="<f4"),
        "archive_global_nu_float32": np.array(1.0, dtype="<f4"),
        "archive_global_p_inf_float32": np.array(0.0, dtype="<f4"),
        "archive_global_reference_length_float32": np.array(8.0, dtype="<f4"),
        "archive_global_rho_inf_float32": np.array(1.0, dtype="<f4"),
    }
    payloads: dict[Path, bytes] = {}
    for name, (relative, _, _) in producer.ARCHIVE_ARRAY_SPECS.items():
        payloads[relative] = arrays[name].tobytes()
    global_root = Path(f"{producer.CASE_DIRECTORY}/_tensordict/global_data")
    for field in producer.GLOBAL_SHAPES:
        name = f"archive_global_{field}_float32"
        payloads[global_root / f"{field}.memmap"] = arrays[name].tobytes()
    forbidden = {
        Path(
            f"{producer.CASE_DIRECTORY}/_tensordict/interior/_tensordict/"
            "point_data/pred_pressure.memmap"
        ): b"producer-must-not-read-prediction",
        Path(
            f"{producer.CASE_DIRECTORY}/_tensordict/interior/_tensordict/"
            "point_data/true_pressure.memmap"
        ): b"producer-must-not-read-truth",
        Path(
            f"{producer.CASE_DIRECTORY}/_tensordict/boundaries/vehicle/"
            "_tensordict/cell_data/pressure.memmap"
        ): b"producer-must-not-read-target",
    }
    payloads.update(forbidden)
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    expected_hashes = {
        name: hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in arrays
    }
    return root, expected_hashes, arrays, set(forbidden)


def _domain_arrays() -> dict[str, np.ndarray]:
    points = np.zeros((producer.BOUNDARY_POINT_COUNT, 3), dtype="<f4")
    points[1] = [1.0, 0.0, 0.0]
    points[2] = [0.0, 1.0, 0.0]
    cells = np.tile(
        np.array([[0, 1, 2]], dtype="<i8"),
        (producer.RESOLUTION, 1),
    )
    return {
        "archive_boundary_points_float32": points,
        "archive_boundary_cells_int64": cells,
        "archive_query_points_float32": points[cells].mean(axis=1),
        "archive_global_L_ref_float32": np.array(5.0, dtype="<f4"),
        "archive_global_U_inf_float32": np.array([3.0, 0.0, 0.0], dtype="<f4"),
        "archive_global_U_inf_dir_float32": np.array([1.0, 0.0, 0.0], dtype="<f4"),
        "archive_global_nu_float32": np.array(1.0, dtype="<f4"),
        "archive_global_p_inf_float32": np.array(0.0, dtype="<f4"),
        "archive_global_reference_length_float32": np.array(8.0, dtype="<f4"),
        "archive_global_rho_inf_float32": np.array(1.0, dtype="<f4"),
    }


def test_producer_loader_opens_only_embedded_hash_bound_input_oracles(
    tmp_path,
    monkeypatch,
):
    root, expected_hashes, expected, forbidden = _producer_archive(tmp_path)
    opened: list[Path] = []
    original = producer._safe_read_bytes

    def recording_read(path, **kwargs):
        opened.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(producer, "_safe_read_bytes", recording_read)
    arrays, record = producer._load_archive_inputs(
        root,
        expected_hashes=expected_hashes,
    )

    assert list(arrays) == list(expected)
    assert record["opened_payload_count"] == 10
    assert record["historical_manifest_opened"] is False
    assert record["input_freeze_record_opened"] is False
    assert all(np.array_equal(arrays[name], value) for name, value in expected.items())
    opened_relative = {
        path.relative_to(root) for path in opened if path.is_relative_to(root)
    }
    assert opened_relative.isdisjoint(forbidden)
    assert opened_relative == {
        spec[0] for spec in producer.ARCHIVE_ARRAY_SPECS.values()
    } | {
        Path(f"{producer.CASE_DIRECTORY}/_tensordict/global_data/{field}.memmap")
        for field in producer.GLOBAL_SHAPES
    }


def test_producer_loader_rejects_unbound_or_changed_payload(tmp_path):
    root, expected_hashes, _, _ = _producer_archive(tmp_path)
    victim = root / next(iter(producer.ARCHIVE_ARRAY_SPECS.values()))[0]
    victim.write_bytes(victim.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive payload changed"):
        producer._load_archive_inputs(
            root,
            expected_hashes=expected_hashes,
        )


def test_producer_loader_rejects_archive_root_symlink(tmp_path):
    root, expected_hashes, _, _ = _producer_archive(tmp_path)
    alias = tmp_path / "archive-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root is a symlink"):
        producer._load_archive_inputs(
            alias,
            expected_hashes=expected_hashes,
        )


def test_producer_loader_rejects_incomplete_embedded_hash_inventory(tmp_path):
    root, expected_hashes, _, _ = _producer_archive(tmp_path)
    expected_hashes.pop(next(iter(expected_hashes)))
    with pytest.raises(ValueError, match="hash inventory changed"):
        producer._load_archive_inputs(root, expected_hashes=expected_hashes)


def test_producer_cli_excludes_oracle_bearing_manifest_and_freeze_inputs():
    args = producer._parse_args(
        [
            "--repo-root",
            "repo",
            "--resolved-config",
            "resolved.yaml",
            "--dataset-config",
            "dataset.yaml",
            "--checkpoint-dir",
            "checkpoint",
            "--archive-input-root",
            "input-only",
            "--lane-label",
            "A",
            "--output-json",
            "lane.json",
            "--output-npz",
            "lane.npz",
        ]
    )
    assert args.archive_input_root == Path("input-only")
    assert not hasattr(args, "historical_predictions_manifest")
    assert not hasattr(args, "input_freeze_record")
    source = Path(producer.__file__).read_text()
    assert "--historical-predictions-manifest" not in source
    assert "--input-freeze-record" not in source


def test_stage_a_domain_is_stripped_and_encoded_geometry_is_finite():
    arrays = _domain_arrays()
    domain = producer._build_stripped_domain(arrays)
    producer._require_stripped_domain(domain)
    assert not domain.interior.point_data.keys()
    assert not domain.interior.cell_data.keys()
    assert not domain.boundaries["vehicle"].point_data.keys()
    assert not domain.boundaries["vehicle"].cell_data.keys()

    encoded = producer._derive_encoded_geometry(domain)
    assert list(encoded) == list(producer.ENCODED_ARRAY_SCHEMAS)
    assert np.isfinite(encoded["encoded_source_areas_float32"]).all()
    assert np.all(encoded["encoded_source_areas_float32"] > 0.0)


def test_stage_a_domain_rejects_any_measure_or_local_field():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    boundary = Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2]]),
        cell_data={"_measure_weights": torch.ones(1)},
    )
    domain = DomainMesh(
        interior=Mesh(points=torch.zeros(1, 3)),
        boundaries={"vehicle": boundary},
        global_data={},
    )
    with pytest.raises(ValueError, match="retains local fields"):
        producer._require_stripped_domain(domain)


def test_model_forward_is_exactly_one_positional_domain_only_call():
    sentinel = object()

    class Spy:
        def __init__(self):
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "prediction"

    spy = Spy()
    output = producer._model_forward_once(
        spy,
        sentinel,
        lambda precision: contextlib.nullcontext(),
    )
    assert output == "prediction"
    assert spy.calls == [((sentinel,), {})]


def test_checkpoint_load_is_epoch_pinned_against_asymmetric_newer_model_file(
    tmp_path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / producer.MODEL_FILENAME).write_bytes(b"frozen-model")
    (checkpoint_dir / producer.TRAINING_STATE_FILENAME).write_bytes(
        b"frozen-training-state"
    )
    # This is the v1 failure mode: a newer model file exists without the
    # correspondingly indexed training-state file. An unpinned loader can
    # select the two sides independently and still report epoch 491.
    (checkpoint_dir / "MeshTransformer.0.999.mdlus").write_bytes(
        b"asymmetric-newer-model"
    )
    calls = []
    selections = []

    def asymmetric_loader(**kwargs):
        calls.append(kwargs)
        requested = kwargs.get("epoch")
        model_epochs = {
            int(path.name.split(".")[-2])
            for path in checkpoint_dir.glob("MeshTransformer.0.*.mdlus")
        }
        state_epochs = {
            int(path.name.split(".")[-2])
            for path in checkpoint_dir.glob("checkpoint.0.*.pt")
        }
        selected_model = requested if requested is not None else max(model_epochs)
        selected_state = requested if requested is not None else max(state_epochs)
        selections.append((selected_model, selected_state))
        return selected_state

    model = object()
    device = torch.device("cpu")
    # Reproduce v1's false reassurance: independent latest-file discovery
    # selects model 999 and state 491, yet the API return value is still 491.
    assert (
        asymmetric_loader(
            path=str(checkpoint_dir),
            models=model,
            device=device,
        )
        == 491
    )
    assert selections == [(999, 491)]
    calls.clear()
    selections.clear()

    loaded_epoch = producer._load_frozen_checkpoint(
        asymmetric_loader,
        checkpoint_dir=checkpoint_dir,
        model=model,
        device=device,
    )

    assert loaded_epoch == producer.EPOCH == 491
    assert selections == [(491, 491)]
    assert calls == [
        {
            "path": str(checkpoint_dir),
            "models": model,
            "device": device,
            "epoch": 491,
        }
    ]


def test_checkpoint_load_rejects_wrong_returned_epoch():
    with pytest.raises(ValueError, match="Loaded epoch 490, expected 491"):
        producer._load_frozen_checkpoint(
            lambda **kwargs: 490,
            checkpoint_dir=Path("checkpoints"),
            model=object(),
            device=torch.device("cpu"),
        )


def test_single_forward_captures_exact_encoded_object_without_second_encode():
    domain = object()
    encoded = SimpleNamespace(name="encoded")

    class Model:
        def __init__(self):
            self.encode_calls = 0
            self.forward_calls = 0

        def encode(self, value):
            self.encode_calls += 1
            assert value is domain
            return encoded

        def __call__(self, value):
            self.forward_calls += 1
            return ("prediction", self.encode(value))

    model = Model()
    output, captured = producer._captured_model_forward_once(
        model,
        domain,
        lambda precision: contextlib.nullcontext(),
    )
    assert output == ("prediction", encoded)
    assert captured is encoded
    assert model.forward_calls == 1
    assert model.encode_calls == 1
    assert "encode" not in model.__dict__


def test_producer_document_guard_rejects_categorical_keys():
    producer._require_noncategorical_document({"status": "complete"})
    with pytest.raises(ValueError, match="categorical key"):
        producer._require_noncategorical_document(
            {"nested": {"preliminary_outcome": "pass"}}
        )


def test_producer_transaction_publishes_both_artifacts_and_sidecars(tmp_path):
    json_path = tmp_path / "lane.json"
    npz_path = tmp_path / "lane.npz"
    payloads = {json_path: b'{"lane":"A"}\n', npz_path: b"npz-bytes"}
    digests = producer._publish_with_sidecars(payloads)
    for path, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        assert path.read_bytes() == payload
        assert (path.parent / f"{path.name}.sha256").read_text() == (
            f"{digest}  {path.name}\n"
        )
        assert digests[path.name] == digest
    assert not list(tmp_path.glob(".*.tmp"))


def test_producer_transaction_rolls_back_and_leaves_no_temporary(
    tmp_path,
    monkeypatch,
):
    json_path = tmp_path / "lane.json"
    npz_path = tmp_path / "lane.npz"
    original_link = producer.os.link
    link_calls = 0

    def fail_second_link(*args, **kwargs):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("injected link failure")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(producer.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected"):
        producer._publish_with_sidecars({json_path: b"json", npz_path: b"npz"})
    assert not json_path.exists()
    assert not npz_path.exists()
    assert not (tmp_path / "lane.json.sha256").exists()
    assert not (tmp_path / "lane.npz.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_producer_transaction_refuses_overwrite(tmp_path):
    output = tmp_path / "lane.json"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        producer._publish_with_sidecars({output: b"replacement"})
    assert output.read_bytes() == b"existing"


def _blank_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in reducer.ARRAY_SCHEMAS.items()
    }


def _lane(label: str, arrays: dict[str, np.ndarray]) -> reducer.Lane:
    return reducer.Lane(
        label=label,
        document={
            "provenance": {
                "process": {
                    "pid": 100 if label == "A" else 200,
                    "hostname": "canary-node",
                    "slurm_job_id": "12345",
                    "cuda_visible_devices": "0" if label == "A" else "1",
                }
            }
        },
        arrays=arrays,
        json_sha256=label.lower() * 64,
        npz_sha256=label.lower() * 64,
    )


def test_signed_zero_is_a_deciding_raw_byte_difference():
    positive = np.array([0.0], dtype="<f4")
    negative = np.array([-0.0], dtype="<f4")
    assert not reducer._arrays_exact(positive, negative)
    difference = reducer._byte_difference(positive, negative)
    assert not difference["exact"]
    assert difference["differing_elements_including_signed_zero"] == 1
    assert difference["maximum_absolute_difference"] == 0.0


def test_truth_table_exact_pass_and_valid_refutation():
    arrays_a = _blank_arrays()
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    archive = {
        "pressure": arrays_a["prediction_pressure_physical_float32"].copy(),
        "wss": arrays_a["prediction_wss_physical_float32"].copy(),
    }
    outcome, comparisons = reducer._decide_complete(
        _lane("A", arrays_a),
        _lane("B", arrays_b),
        archive,
    )
    assert outcome == reducer.EXACT_PASS
    assert all(
        record["exact"]
        for record in comparisons["archive_prediction_comparisons"].values()
    )

    arrays_a["prediction_pressure_physical_float32"][0] = 1.0
    arrays_b["prediction_pressure_physical_float32"][0] = 1.0
    outcome, comparisons = reducer._decide_complete(
        _lane("A", arrays_a),
        _lane("B", arrays_b),
        archive,
    )
    assert outcome == reducer.VALID_REFUTATION
    assert not comparisons["archive_prediction_comparisons"]["pressure"]["exact"]


def test_truth_table_nondeterministic_lanes_are_invalid():
    arrays_a = _blank_arrays()
    arrays_b = {name: value.copy() for name, value in arrays_a.items()}
    arrays_b["prediction_wss_physical_float32"][0, 0] = -0.0
    archive = {
        "pressure": arrays_a["prediction_pressure_physical_float32"].copy(),
        "wss": arrays_a["prediction_wss_physical_float32"].copy(),
    }
    outcome, comparisons = reducer._decide_complete(
        _lane("A", arrays_a),
        _lane("B", arrays_b),
        archive,
    )
    assert outcome == reducer.INVALID
    assert "not raw-byte deterministic" in comparisons["reason"]


def test_truth_table_requires_distinct_exact_gpu_tokens_and_processes():
    arrays = _blank_arrays()
    archive = {
        "pressure": arrays["prediction_pressure_physical_float32"].copy(),
        "wss": arrays["prediction_wss_physical_float32"].copy(),
    }
    lane_a = _lane("A", arrays)
    lane_b = _lane("B", {name: value.copy() for name, value in arrays.items()})
    lane_b.document["provenance"]["process"]["cuda_visible_devices"] = "0,1"
    outcome, comparisons = reducer._decide_complete(lane_a, lane_b, archive)
    assert outcome == reducer.INVALID
    assert not comparisons["process_isolation"]["lane_gpu_tokens_exact"]


def test_lane_artifact_separation_rejects_hardlink_alias(tmp_path):
    paths = [tmp_path / name for name in ("a.json", "a.npz", "b.json", "b.npz")]
    for path in paths:
        path.write_bytes(path.name.encode())
        path.with_name(f"{path.name}.sha256").write_bytes(b"sidecar")
    paths[3].unlink()
    os.link(paths[1], paths[3])
    with pytest.raises(ValueError, match="share an inode"):
        reducer._validate_distinct_lane_artifacts(paths)


def _reducer_archive(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    pressure = np.arange(reducer.RESOLUTION, dtype="<f4")
    wss = np.arange(reducer.RESOLUTION * 3, dtype="<f4").reshape(reducer.RESOLUTION, 3)
    payloads = {
        reducer.ARCHIVED_PREDICTION_SPECS["pressure"][0]: pressure.tobytes(),
        reducer.ARCHIVED_PREDICTION_SPECS["wss"][0]: wss.tobytes(),
        Path(
            f"{reducer.CASE_DIRECTORY}/_tensordict/interior/_tensordict/"
            "point_data/true_pressure.memmap"
        ): b"reducer-must-not-read-truth",
        Path(
            f"{reducer.CASE_DIRECTORY}/_tensordict/boundaries/vehicle/"
            "_tensordict/points.memmap"
        ): b"reducer-does-not-need-geometry",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = tmp_path / "manifest.sha256"
    digest, entries = _write_manifest(manifest, payloads)
    return root, manifest, digest, entries, pressure, wss


def test_reducer_opens_only_manifest_bound_archived_predictions(
    tmp_path,
    monkeypatch,
):
    root, manifest, digest, _, pressure, wss = _reducer_archive(tmp_path)
    opened: list[Path] = []
    original = reducer._safe_read_bytes

    def recording_read(path, **kwargs):
        opened.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(reducer, "_safe_read_bytes", recording_read)
    archived, record, _ = reducer._load_archived_predictions(
        root,
        manifest,
        expected_manifest_sha256=digest,
        expected_manifest_entries=None,
    )
    assert np.array_equal(archived["pressure"], pressure)
    assert np.array_equal(archived["wss"], wss)
    assert record["opened_payload_count"] == 2
    opened_relative = {
        path.relative_to(root)
        for path in opened
        if path != manifest and path.is_relative_to(root)
    }
    assert opened_relative == {
        spec[0] for spec in reducer.ARCHIVED_PREDICTION_SPECS.values()
    }


def _write_attested(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")


def _valid_lane_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path, dict[str, str]]:
    arrays = _blank_arrays()
    manifest_entries: dict[str, str] = {}
    opened: dict[str, dict[str, object]] = {}
    for name, (relative, _, _) in reducer.DIRECT_ARRAY_SPECS.items():
        digest = reducer._array_sha256(arrays[name])
        manifest_entries[f"./{relative.as_posix()}"] = digest
        opened[name] = {
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": arrays[name].nbytes,
        }
    npz_stream = io.BytesIO()
    np.savez(npz_stream, **arrays)
    npz_payload = npz_stream.getvalue()
    npz_path = tmp_path / "lane.npz"
    _write_attested(npz_path, npz_payload)

    producer_sha = "a" * 64
    monkeypatch.setattr(reducer, "EXPECTED_PRODUCER_SHA256", producer_sha)
    document = {
        "schema_version": 1,
        "artifact_kind": reducer.PRODUCER_ARTIFACT_KIND,
        "status": reducer.PRODUCER_STATUS,
        "lane_label": "A",
        "contract": {
            "case_id": reducer.CASE_ID,
            "reader_index": reducer.READER_INDEX,
            "resolution": reducer.RESOLUTION,
            "precision": reducer.PRECISION,
            "compiled_model": False,
            "archive_is_input_oracle_only": True,
            "archived_predictions_opened": False,
            "archived_truth_opened": False,
            "raw_targets_opened": False,
            "historical_manifest_opened": False,
            "input_freeze_record_opened": False,
            "dataset_reader_constructed": False,
            "model_call": "model(domain)",
            "model_call_count": 1,
            "model_call_keyword_arguments": [],
            "canonical_source_geometry_supplied": False,
            "encoded_geometry_captured_from_single_forward": True,
            "local_data_fields_present": False,
            "measure_weights_present": False,
            "categorical_decision_present": False,
            "process_isolated_lane": True,
            "checkpoint_load_epoch": 491,
        },
        "archive_inputs": {
            "historical_manifest_sha256": (reducer.EXPECTED_HISTORICAL_MANIFEST_SHA256),
            "historical_manifest_opened": False,
            "input_freeze_record_sha256": reducer.EXPECTED_INPUT_FREEZE_SHA256,
            "input_freeze_record_opened": False,
            "input_hash_binding": "embedded_sha256_constants",
            "opened_payload_count": len(opened),
            "opened_payloads": opened,
        },
        "npz": {
            "filename": npz_path.name,
            "sha256": hashlib.sha256(npz_payload).hexdigest(),
            "array_count": len(arrays),
            "array_manifest": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": reducer._array_sha256(value),
                }
                for name, value in arrays.items()
            },
        },
        "provenance": {
            "producer_sha256": producer_sha,
            "repo_root": "/frozen/repo",
            "loaded_epoch": 491,
            "historical_manifest_sha256": (reducer.EXPECTED_HISTORICAL_MANIFEST_SHA256),
            "input_freeze_record_sha256": reducer.EXPECTED_INPUT_FREEZE_SHA256,
            "process": {
                "pid": 100,
                "hostname": "canary-node",
                "slurm_job_id": "12345",
                "cuda_visible_devices": "0",
            },
            "frozen_inputs": {
                "resolved_config": reducer.EXPECTED_RESOLVED_CONFIG_SHA256,
                "dataset_config": reducer.EXPECTED_DATASET_CONFIG_SHA256,
                "model_checkpoint": reducer.EXPECTED_MODEL_SHA256,
                "training_state": reducer.EXPECTED_TRAINING_STATE_SHA256,
                "normalization_state": reducer.EXPECTED_NORMALIZATION_SHA256,
                "current_infer_source": reducer.EXPECTED_CURRENT_INFER_SHA256,
                "current_model_source": (reducer.EXPECTED_CURRENT_MODEL_SOURCE_SHA256),
                "current_execution_source_tree": (
                    reducer.EXPECTED_CURRENT_SOURCE_TREE_SHA256
                ),
                "checkpoint_load_epoch": 491,
                "parameter_count": 1_278_268,
                "model_seed": 42,
                "import_provenance": {
                    "physicsnemo": "/frozen/repo/physicsnemo/__init__.py",
                    "mesh_transformer_model": (
                        "/frozen/repo/physicsnemo/experimental/nn/"
                        "mesh_attention/model.py"
                    ),
                    "recipe_infer": (
                        "/frozen/repo/examples/cfd/external_aerodynamics/"
                        "unified_external_aero_recipe/src/infer.py"
                    ),
                },
            },
        },
    }
    json_path = tmp_path / "lane.json"
    _write_attested(
        json_path,
        json.dumps(document, sort_keys=True).encode() + b"\n",
    )
    return json_path, npz_path, manifest_entries


def test_lane_loader_parses_exact_attested_npz_bytes_once(tmp_path, monkeypatch):
    json_path, npz_path, manifest_entries = _valid_lane_artifacts(
        tmp_path,
        monkeypatch,
    )
    reads: dict[Path, int] = {}
    original = reducer._safe_read_bytes

    def counted(path, **kwargs):
        path = Path(path)
        reads[path] = reads.get(path, 0) + 1
        return original(path, **kwargs)

    monkeypatch.setattr(reducer, "_safe_read_bytes", counted)
    lane = reducer._load_lane(
        json_path,
        npz_path,
        "A",
        manifest_entries,
    )
    assert lane.label == "A"
    assert set(lane.arrays) == set(reducer.ARRAY_SCHEMAS)
    assert reads[json_path] == 1
    assert reads[npz_path] == 1
    assert reads[json_path.with_name(f"{json_path.name}.sha256")] == 1
    assert reads[npz_path.with_name(f"{npz_path.name}.sha256")] == 1


def test_lane_loader_rejects_producer_categorical_leakage(tmp_path, monkeypatch):
    json_path, npz_path, manifest_entries = _valid_lane_artifacts(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json_path.read_text())
    document["preliminary_outcome"] = "pass"
    _write_attested(
        json_path,
        json.dumps(document, sort_keys=True).encode() + b"\n",
    )
    with pytest.raises(ValueError, match="categorical key"):
        reducer._load_lane(json_path, npz_path, "A", manifest_entries)


@pytest.mark.parametrize(
    "path",
    [
        ("contract", "checkpoint_load_epoch"),
        ("provenance", "loaded_epoch"),
        ("provenance", "frozen_inputs", "checkpoint_load_epoch"),
    ],
)
def test_lane_loader_requires_epoch_pin_attestation(
    tmp_path,
    monkeypatch,
    path,
):
    json_path, npz_path, manifest_entries = _valid_lane_artifacts(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json_path.read_text())
    cursor = document
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = 999
    _write_attested(
        json_path,
        json.dumps(document, sort_keys=True).encode() + b"\n",
    )

    with pytest.raises(ValueError, match="contract changed"):
        reducer._load_lane(json_path, npz_path, "A", manifest_entries)


def test_reducer_main_publishes_incomplete_for_missing_inputs(tmp_path):
    output = tmp_path / "result.json"
    reducer.main(
        [
            "--lane-a-json",
            str(tmp_path / "missing-a.json"),
            "--lane-a-npz",
            str(tmp_path / "missing-a.npz"),
            "--lane-b-json",
            str(tmp_path / "missing-b.json"),
            "--lane-b-npz",
            str(tmp_path / "missing-b.npz"),
            "--historical-predictions",
            str(tmp_path / "missing-archive"),
            "--historical-predictions-manifest",
            str(tmp_path / "missing-manifest"),
            "--output-json",
            str(output),
        ]
    )
    document = json.loads(output.read_text())
    assert document["outcome"] == reducer.INCOMPLETE
    assert output.with_name(f"{output.name}.sha256").is_file()


def test_reducer_main_publishes_invalid_for_corrupt_present_manifest(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.write_bytes(b"corrupt")
    output = tmp_path / "result.json"
    reducer.main(
        [
            "--lane-a-json",
            str(tmp_path / "missing-a.json"),
            "--lane-a-npz",
            str(tmp_path / "missing-a.npz"),
            "--lane-b-json",
            str(tmp_path / "missing-b.json"),
            "--lane-b-npz",
            str(tmp_path / "missing-b.npz"),
            "--historical-predictions",
            str(tmp_path / "archive"),
            "--historical-predictions-manifest",
            str(manifest),
            "--output-json",
            str(output),
        ]
    )
    document = json.loads(output.read_text())
    assert document["outcome"] == reducer.INVALID


def test_reducer_transaction_rolls_back_on_sidecar_link_failure(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result.json"
    original_link = reducer.os.link
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(reducer.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        reducer._publish_json(output, b"payload")
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_stage_a_fixed_input_contract_is_run_118_only():
    assert producer.CASE_ID == reducer.CASE_ID == "run_118"
    assert producer.READER_INDEX == reducer.READER_INDEX == 21
    assert producer.RESOLUTION == reducer.RESOLUTION == 10_000
    assert len(producer.ARCHIVE_ARRAY_SPECS) == 3
    assert set(producer.GLOBAL_SHAPES) == {
        "L_ref",
        "U_inf",
        "U_inf_dir",
        "nu",
        "p_inf",
        "reference_length",
        "rho_inf",
    }
    assert not hasattr(SimpleNamespace(**producer.ARCHIVE_ARRAY_SPECS), "pred_pressure")


def test_reducer_binds_exact_producer_source():
    producer_path = (
        Path(__file__).parents[1]
        / "studies"
        / "drivaerml_historical_k10000_stage_a_archive_canary.py"
    )
    assert hashlib.sha256(producer_path.read_bytes()).hexdigest() == (
        reducer.EXPECTED_PRODUCER_SHA256
    )
