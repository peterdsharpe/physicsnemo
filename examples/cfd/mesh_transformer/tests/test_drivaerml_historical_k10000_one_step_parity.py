# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the frozen K=10k one-step parity producer."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

STUDY = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "drivaerml_historical_k10000_one_step_parity.py"
)


def _load_study():
    spec = importlib.util.spec_from_file_location("one_step_parity_study", STUDY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def study():
    return _load_study()


def test_frozen_panel_and_optimizer_contract(study) -> None:
    assert study.CASE_SPECS == (
        (0, "run_118"),
        (12, "run_271"),
        (24, "run_429"),
        (35, "run_86"),
    )
    assert study.REGIMES == ("fresh_seed42", "checkpoint_epoch491")
    assert study.PRECISIONS == ("bfloat16", "float32")
    assert study.CHECKPOINT_EPOCH == 491
    source = STUDY.read_text()
    assert "build_muon_optimizer(" in source
    assert "compile_optimizer=False" in source
    assert 'contained != ("Muon", "AdamW")' in source
    assert "observed_parameter_ids != expected_parameter_ids" in source
    assert "state_entries != (0, 0)" in source
    assert "Checkpoint optimizer state is incomplete" in source
    assert "optimizer.step()" in source
    assert "loss.backward()" in source


def test_support_hashes_are_frozen(study) -> None:
    assert study.EXPECTED_LEGACY_SUPPORT_SHA256 == (
        "bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5"
    )
    assert study.EXPECTED_RUNTIME_HELPER_SHA256 == (
        "dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896"
    )
    assert study.EXPECTED_CANONICAL_HELPER_SHA256 == (
        "694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac"
    )
    assert study.EXPECTED_SOURCE_TREE_SHA256 == (
        "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
    )


def test_frozen_manifest_helper_signatures_and_call_binding(study) -> None:
    legacy = study._load_verified_module(
        STUDY.with_name(study.LEGACY_SUPPORT_FILENAME),
        expected_sha256=study.EXPECTED_LEGACY_SUPPORT_SHA256,
        module_name="one_step_signature_legacy_support",
        label="Frozen historical replay support",
    )
    runtime = study._load_verified_module(
        STUDY.with_name(study.RUNTIME_HELPER_FILENAME),
        expected_sha256=study.EXPECTED_RUNTIME_HELPER_SHA256,
        module_name="one_step_signature_runtime_support",
        label="Frozen historical replay runtime",
    )
    canonical = study._load_verified_module(
        STUDY.with_name(study.CANONICAL_HELPER_FILENAME),
        expected_sha256=study.EXPECTED_CANONICAL_HELPER_SHA256,
        module_name="one_step_signature_canonical_support",
        label="Frozen canonical geometry helper",
    )
    assert tuple(inspect.signature(legacy._verify_geometry_manifest).parameters) == (
        "helper",
        "manifest_path",
        "dataset_root",
    )
    assert tuple(
        inspect.signature(legacy._verify_target_input_manifest).parameters
    ) == (
        "manifest_path",
        "dataset_root",
    )

    support_modules = {
        "legacy": legacy,
        "runtime_support": runtime,
        "canonical": canonical,
        "canonical_support": canonical,
    }
    visited: set[str] = set()
    tree = ast.parse(STUDY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id not in support_modules
        ):
            continue
        owner = node.func.value.id
        function_name = node.func.attr
        function = getattr(support_modules[owner], function_name)
        assert all(keyword.arg is not None for keyword in node.keywords)
        inspect.signature(function).bind(
            *([object()] * len(node.args)),
            **{str(keyword.arg): object() for keyword in node.keywords},
        )
        visited.add(f"{owner}.{function_name}")
    assert "legacy._verify_geometry_manifest" in visited
    assert "legacy._verify_target_input_manifest" in visited
    assert "runtime_support._load_runtime" in visited
    assert "canonical._finish_canonical_bundle" in visited


def test_prefix_matches_reducer_schema(study) -> None:
    assert (
        study._prefix("fresh_seed42", "bfloat16", 0, "run_118", "legacy")
        == "fresh_seed42__bfloat16__case_00_run_118__legacy__"
    )


def test_parameter_layout_is_contiguous_and_module_mapped(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Sequential(torch.nn.Linear(4, 2)),
    )
    expected = sum(parameter.numel() for parameter in model.parameters())
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", expected)
    layout, arrays = study._parameter_layout(model)
    starts = arrays["parameter_slice_starts_int64"]
    stops = arrays["parameter_slice_stops_int64"]
    module_indices = arrays["parameter_slice_module_indices_int64"]
    assert starts.dtype.str == "<i8"
    assert stops.dtype.str == "<i8"
    assert module_indices.dtype.str == "<i8"
    assert starts[0] == 0
    assert stops[-1] == expected
    np.testing.assert_array_equal(starts[1:], stops[:-1])
    assert len(layout["parameter_names"]) == len(starts)
    assert set(module_indices.tolist()) == set(range(len(layout["module_names"])))


def test_raw_tensor_equality_distinguishes_signed_zero(study) -> None:
    positive = torch.tensor([0.0, 1.0], dtype=torch.float32)
    negative = torch.tensor([-0.0, 1.0], dtype=torch.float32)
    assert not study._tensor_raw_equal(positive, negative)
    assert study._tensor_raw_equal(positive, positive.clone())


def test_stable_hash_is_mapping_order_independent_and_value_sensitive(study) -> None:
    first = {"b": torch.tensor([1.0]), "a": [1, 2, 3]}
    second = {"a": [1, 2, 3], "b": torch.tensor([1.0])}
    changed = {"a": [1, 2, 4], "b": torch.tensor([1.0])}
    assert study._stable_sha256(first) == study._stable_sha256(second)
    assert study._stable_sha256(first) != study._stable_sha256(changed)


def test_array_manifest_uses_raw_dtype_and_bytes(study) -> None:
    arrays = {
        "float": np.asarray([0.0, -0.0], dtype="<f4"),
        "integer": np.asarray([1, 2], dtype="<i8"),
    }
    manifest = study._array_manifest(arrays)
    assert manifest["float"]["dtype"] == "<f4"
    assert manifest["integer"]["dtype"] == "<i8"
    assert manifest["float"]["shape"] == [2]
    changed = {"float": np.asarray([0.0, 0.0], dtype="<f4")}
    assert (
        manifest["float"]["sha256"] != study._array_manifest(changed)["float"]["sha256"]
    )


def test_stochastic_module_audit_finds_training_randomness(study) -> None:
    deterministic = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    stochastic = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
    assert study._stochastic_modules(deterministic) == []
    assert study._stochastic_modules(stochastic) == ["1"]


def test_single_rank_environment_is_exact(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in study.EXPECTED_SINGLE_RANK_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    study._validate_single_rank_environment()
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one torchrun rank"):
        study._validate_single_rank_environment()


def test_source_contains_required_measure_and_canonical_calls() -> None:
    source = STUDY.read_text()
    assert 'batch.get("target_measure")' in source
    assert "domain_with_targets.interior.point_data.get(" in source
    assert "_tensor_raw_equal(batch_target_measure, preserved_target_measure)" in source
    assert "loss_calculator(" in source
    assert "target_measure," in source
    assert '"_measure_weights" in boundary.cell_data.keys()' in source
    assert "canonical_source_geometry=geometry" in source
    assert "runtime.mesh_type(points=geometry.centroids)" in source
    assert "model(domain)" in source


def test_cli_help_precedes_runtime_environment_gate(study) -> None:
    with pytest.raises(SystemExit) as caught:
        study.main(["--help"])
    assert caught.value.code == 0
