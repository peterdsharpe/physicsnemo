# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the minimal historical K=10,000 runtime helper."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import drivaerml_historical_k10000_replay as producer
import drivaerml_historical_k10000_replay_runtime as runtime
import drivaerml_trace_fixed_query_audit as legacy
import numpy as np
import pytest
import torch

EXPECTED_PUBLIC_API = [
    "MODEL_FILENAME",
    "TRAINING_STATE_FILENAME",
    "NORM_STATS_FILENAME",
    "CaseSpec",
    "CASE_SPECS",
    "_validate_historical_starts",
    "_source_tree_manifest_sha256",
    "_load_runtime",
    "_native_geometry",
    "_apply_pipeline",
]
EXPECTED_COHORT_SHA256 = (
    "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
)


def _case_rows() -> list[dict[str, int | str]]:
    return [
        {
            "cohort_ordinal": spec.cohort_ordinal,
            "case_id": spec.case_id,
            "reader_index": spec.reader_index,
            "n_master_cells": spec.n_master_cells,
            "historical_start": spec.historical_start,
        }
        for spec in runtime.CASE_SPECS
    ]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_public_api_is_exactly_the_producer_runtime_surface() -> None:
    assert runtime.__all__ == EXPECTED_PUBLIC_API
    assert {name for name in vars(runtime) if not name.startswith("_")} == {
        "MODEL_FILENAME",
        "TRAINING_STATE_FILENAME",
        "NORM_STATS_FILENAME",
        "CaseSpec",
        "CASE_SPECS",
    }
    assert runtime.MODEL_FILENAME == "MeshTransformer.0.491.mdlus"
    assert runtime.TRAINING_STATE_FILENAME == "checkpoint.0.491.pt"
    assert runtime.NORM_STATS_FILENAME == "norm_stats.pt"
    assert runtime._RUNTIME_SAMPLING_RESOLUTION == 10_000


def test_case_table_matches_producer_and_frozen_cohort_identity() -> None:
    observed = tuple(
        (
            spec.cohort_ordinal,
            spec.case_id,
            spec.reader_index,
            spec.n_master_cells,
            spec.historical_start,
        )
        for spec in runtime.CASE_SPECS
    )
    assert observed == tuple(producer.EXPECTED_CASE_SPECS)
    assert len(observed) == 36
    assert [row[0] for row in observed] == list(range(36))
    assert hashlib.sha256(_canonical_json_bytes(_case_rows())).hexdigest() == (
        EXPECTED_COHORT_SHA256
    )


def test_case_spec_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        runtime.CASE_SPECS[0].historical_start = 0  # type: ignore[misc]


def test_historical_starts_replay_exact_rng_chain() -> None:
    runtime._validate_historical_starts()
    generator = torch.Generator().manual_seed(45)
    replayed = tuple(
        int(
            torch.randint(
                0,
                spec.n_master_cells - 10_000 + 1,
                (1,),
                generator=generator,
            ).item()
        )
        for spec in runtime.CASE_SPECS
    )
    assert replayed == tuple(spec.historical_start for spec in runtime.CASE_SPECS)


def test_historical_start_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime.CASE_SPECS[0]
    changed = runtime.CaseSpec(
        original.cohort_ordinal,
        original.case_id,
        original.reader_index,
        original.n_master_cells,
        original.historical_start + 1,
    )
    monkeypatch.setattr(runtime, "CASE_SPECS", (changed, *runtime.CASE_SPECS[1:]))

    with pytest.raises(ValueError, match="Historical RNG replay changed"):
        runtime._validate_historical_starts()


def test_checkpoint_load_is_epoch_pinned_when_newer_model_exists(
    tmp_path: Path,
) -> None:
    for name in (
        "MeshTransformer.0.491.mdlus",
        "MeshTransformer.0.999.mdlus",
        "checkpoint.0.491.pt",
    ):
        (tmp_path / name).touch()
    loaded_paths: list[tuple[Path, Path]] = []

    def fake_load_checkpoint(
        *,
        path: str,
        models: object,
        device: torch.device,
        epoch: int | None = None,
    ) -> int:
        del models, device
        model_path = Path(path) / f"MeshTransformer.0.{epoch}.mdlus"
        training_state_path = Path(path) / f"checkpoint.0.{epoch}.pt"
        assert model_path.is_file()
        assert training_state_path.is_file()
        loaded_paths.append((model_path, training_state_path))
        return int(epoch) if epoch is not None else 0

    loaded_epoch = runtime._load_frozen_checkpoint(
        fake_load_checkpoint,
        tmp_path,
        object(),
        torch.device("cpu"),
    )

    assert loaded_epoch == 491
    assert loaded_paths == [
        (
            tmp_path / "MeshTransformer.0.491.mdlus",
            tmp_path / "checkpoint.0.491.pt",
        )
    ]
    assert (tmp_path / "MeshTransformer.0.999.mdlus").is_file()


def test_checkpoint_load_rejects_wrong_reported_epoch(tmp_path: Path) -> None:
    def fake_load_checkpoint(**_: object) -> int:
        return 490

    with pytest.raises(ValueError, match="Loaded epoch 490, expected 491"):
        runtime._load_frozen_checkpoint(
            fake_load_checkpoint,
            tmp_path,
            object(),
            torch.device("cpu"),
        )


def test_source_tree_manifest_behavior_matches_original_helper() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    assert runtime._source_tree_manifest_sha256(
        repo_root
    ) == legacy._source_tree_manifest_sha256(repo_root)


def test_native_geometry_is_bitwise_identical_to_original_helper() -> None:
    mesh = SimpleNamespace(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        cells=torch.tensor(
            [[0, 1, 2], [0, 2, 3]],
            dtype=torch.int64,
        ),
    )

    extracted = runtime._native_geometry(mesh)
    original = legacy._native_geometry(mesh)

    for observed, expected in zip(extracted, original, strict=True):
        assert observed.dtype == expected.dtype
        assert observed.shape == expected.shape
        assert observed.tobytes() == expected.tobytes()
    assert extracted[0].dtype == np.dtype("<f4")
    assert extracted[1].dtype == np.dtype("<f4")
    assert extracted[2].dtype == np.dtype("<f8")
    np.testing.assert_array_equal(
        extracted[0],
        np.array(
            [[2.0 / 3.0, 1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0, 1.0]],
            dtype="<f4",
        ),
    )
    np.testing.assert_array_equal(extracted[2], np.array([1.0, 1.5], dtype="<f8"))


def test_native_geometry_rejects_degenerate_triangle() -> None:
    mesh = SimpleNamespace(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ),
        cells=torch.tensor([[0, 1, 2]]),
    )

    with pytest.raises(ValueError, match="degenerate triangle"):
        runtime._native_geometry(mesh)


class _Mesh:
    def __init__(self, points: torch.Tensor):
        self.points = points

    def to(self, device: torch.device) -> _Mesh:
        return _Mesh(self.points.to(device))

    def translate(self, offset: torch.Tensor) -> _Mesh:
        return _Mesh(self.points + offset)


class CenterMesh:
    def __call__(self, mesh: _Mesh) -> _Mesh:
        raise AssertionError("The helper must intercept CenterMesh")


class DomainMesh:
    def __init__(self, points: torch.Tensor):
        self.points = points


class _ToDomain:
    def __call__(self, mesh: _Mesh) -> DomainMesh:
        return DomainMesh(mesh.points)


@pytest.mark.parametrize(
    "fixed_center",
    [None, torch.tensor([2.0, 3.0, 4.0])],
)
def test_apply_pipeline_is_identical_to_original_helper(
    fixed_center: torch.Tensor | None,
) -> None:
    mesh = _Mesh(
        torch.tensor(
            [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
            dtype=torch.float32,
        )
    )
    fake_runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dataset=SimpleNamespace(transforms=[CenterMesh(), _ToDomain()]),
    )

    observed_domain, observed_center = runtime._apply_pipeline(
        fake_runtime,
        mesh,
        fixed_center=fixed_center,
    )
    expected_domain, expected_center = legacy._apply_pipeline(
        fake_runtime,
        mesh,
        fixed_center=fixed_center,
    )

    torch.testing.assert_close(observed_center, expected_center, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        observed_domain.points,
        expected_domain.points,
        rtol=0.0,
        atol=0.0,
    )


def test_apply_pipeline_requires_exactly_one_center_transform() -> None:
    mesh = _Mesh(torch.zeros((2, 3)))
    fake_runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dataset=SimpleNamespace(transforms=[_ToDomain()]),
    )

    with pytest.raises(ValueError, match="exactly one CenterMesh"):
        runtime._apply_pipeline(fake_runtime, mesh, fixed_center=None)


def test_source_contains_no_scientific_result_or_reference_reader_surface() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for token in (
        "outcome",
        "oracle",
        "archive",
        "metric",
        "score",
        "tolerance",
        "verdict",
        "prediction",
    ):
        assert token not in lowered
    assert re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", lowered) is None

    tree = ast.parse(source)
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert not any(
        forbidden in name.lower()
        for name in assigned_names
        for forbidden in ("outcome", "oracle", "metric", "score", "tolerance")
    )
