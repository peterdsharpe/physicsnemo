# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native-domain, training, independent-query, and physical-export contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import datasets as datasets_module
import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import (
    _apply_dataset_reader_overrides,
    build_dataloaders,
    build_dataset,
)
from forces import ForceContext
from forward_kwargs import extract_targets, walk_path
from hydra import compose, initialize_config_dir
from infer import (
    _to_pointwise,
    attach_and_save,
    build_redim_field_types,
    redimensionalize,
)
from loss import LossCalculator
from metrics import MetricCalculator
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict

from physicsnemo import Module
from physicsnemo.experimental.models.ab_upt import ABUPT
from physicsnemo.datapipes.readers import DomainMeshReader
from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(
    hidden_dim=24,
    num_heads=3,
    num_interaction_blocks=1,
    num_shared_layers=1,
    num_decoder_layers=1,
    pooling_radius=2.0,
    pooling_chunk_size=31,
    radius_search_backend="torch",
    coordinate_scale=1000.0,
)
COUNTS = dict(
    geometry_points=17,
    num_supernodes=5,
    num_surface_anchors=7,
    num_volume_anchors=9,
    surface_queries=11,
    volume_queries=13,
)


COUNT_KEYS = {
    "geometry": "geometry_points",
    "supernodes": "num_supernodes",
    "surface_anchors": "num_surface_anchors",
    "volume_anchors": "num_volume_anchors",
    "surface_queries": "surface_queries",
    "interior": "volume_queries",
}


def make_model(mode="joint", **kwargs):
    """Construct the library model with the dataset channel layout."""
    return ABUPT(
        surface_channels=4 if mode != "volume" else 0,
        volume_channels=5 if mode != "surface" else 0,
        parameter_channels=3,
        **kwargs,
    )


def make_reader(root, mode="joint", **counts):
    """Use the actual dataset selection schema with small test counts."""
    cfg = OmegaConf.load(ROOT / "datasets" / f"drivaer_ml_ab_upt_{mode}.yaml")
    cfg.train_datadir = str(root)
    cfg.sampling_resolution = 13
    specs = OmegaConf.to_container(cfg.pipeline.reader.point_sets, resolve=True)
    for name, spec in specs.items():
        spec["count"] = counts[COUNT_KEYS[name]]
    return DomainMeshReader(root, pattern="run_*/domain_*.pdmsh", point_sets=specs)


def anchor_queries(specs, enabled):
    """Select shared training indices or independent validation queries."""
    for name, anchors in (
        ("surface_queries", "surface_anchors"),
        ("interior", "volume_anchors"),
    ):
        if name in specs:
            if OmegaConf.is_config(specs):
                OmegaConf.update(
                    specs,
                    f"{name}.indices_from",
                    anchors if enabled else None,
                    force_add=True,
                )
            else:
                specs[name]["indices_from"] = anchors if enabled else None


def predict(model, **kwargs):
    """Normalize native tensor outputs through the shared recipe path."""
    groups = []
    targets = {}
    if model.surface_channels:
        targets.update({"surface.pressure": "scalar", "surface.wss": "vector"})
        groups.append(["surface.pressure", "surface.wss"])
    if model.volume_channels:
        targets.update(
            {
                "volume.velocity": "vector",
                "volume.pressure": "scalar",
                "volume.nut": "scalar",
            }
        )
        groups.append(["volume.velocity", "volume.pressure", "volume.nut"])
    return normalize_output_to_tensordict(
        model(**kwargs),
        targets,
        "tensors",
        output_groups=groups if len(groups) == 2 else None,
    )


@pytest.fixture
def native_case(tmp_path):
    """Write a real memory-mapped domain with distinct surface/volume labels."""
    torch.manual_seed(41)
    # Separate, consistently wound nondegenerate triangles.
    origins = torch.rand(40, 3)
    corners = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    surface = Mesh(
        points=(origins[:, None] + corners).reshape(-1, 3),
        cells=torch.arange(120).reshape(-1, 3),
        cell_data={
            "pMeanTrim": torch.arange(40).float() + 10,
            "wallShearStressMeanTrim": torch.randn(40, 3) * 0.001,
        },
    )
    interior = Mesh(
        points=torch.rand(61, 3),
        point_data={
            "pMeanTrim": torch.arange(61).float() + 100,
            "UMeanTrim": torch.randn(61, 3) + 30,
            "nutMeanTrim": torch.rand(61) * 0.001,
        },
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={"vehicle": surface},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(1.2),
            "p_inf": torch.tensor(10.0),
            "L_ref": torch.tensor(4.0),
        },
    )
    path = tmp_path / "run_1" / "domain_1.pdmsh"
    path.parent.mkdir()
    domain.save(str(path))
    return tmp_path, domain


def pipeline(root, mode):
    """Compose actual user-facing model and dataset YAMLs at small sizes."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model=ab_upt_{mode}"])
    ds = OmegaConf.load(ROOT / "datasets" / f"drivaer_ml_ab_upt_{mode}.yaml")
    ds.train_datadir = str(root)
    ds.sampling_resolution = 13
    for name, spec in cfg.dataset_reader_overrides.point_sets.items():
        spec.count = COUNTS[COUNT_KEYS[name]]
    cfg.model.update(SMALL)
    ds = _apply_dataset_reader_overrides(ds, cfg)
    dataset = build_dataset(ds, device="cpu", num_workers=1)
    dataset.reader.set_generator(torch.Generator().manual_seed(123))
    return cfg, ds, dataset


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_native_pipeline_train_checkpoint_export(native_case, mode, tmp_path):
    """All modes optimize and round-trip with unequal query sizes."""
    root, original = native_case
    if mode == "surface":
        # Exercise the actual recipe with no volume coordinates or labels on disk.
        root = tmp_path / "surface_only"
        path = root / "run_1" / "domain_1.pdmsh"
        path.parent.mkdir(parents=True)
        DomainMesh(
            interior=Mesh(points=torch.empty(0, 3)),
            boundaries={"vehicle": original.boundaries["vehicle"]},
            global_data=original.global_data,
        ).save(str(path))
    cfg, ds, dataset = pipeline(root, mode)
    domain, meta = dataset[0]
    assert ("surface_anchors" in domain.boundary_names) == (mode != "volume")
    assert ("volume_anchors" in domain.boundary_names) == (mode != "surface")
    if mode == "surface":
        assert domain.interior.n_points == 0
        assert "volume_anchors" not in cfg.forward_kwargs
    if mode == "volume":
        assert "surface_anchors" not in cfg.forward_kwargs
    targets = OmegaConf.to_container(ds.targets)
    sources = OmegaConf.to_container(ds.target_sources)
    collate = build_collate_fn(
        "tensors", OmegaConf.to_container(cfg.forward_kwargs), targets, sources
    )
    batch = collate([(domain, meta)])
    cfg.model.pooling_message_fp32 = mode == "joint"
    model = hydra.utils.instantiate(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = LossCalculator(targets, loss_type="mse")
    pred = normalize_output_to_tensordict(
        model(**batch["forward_kwargs"]),
        targets,
        "tensors",
        output_groups=cfg.get("output_groups"),
    )
    loss, _ = loss_fn(pred, batch["targets"])
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt.step()
    metrics = MetricCalculator(targets, metrics=["l2"])(pred.detach(), batch["targets"])
    assert all(
        torch.isfinite(v).all()
        for v in metrics.values(include_nested=True, leaves_only=True)
    )
    path = tmp_path / f"{mode}.mdlus"
    model.save(str(path))
    restored = Module.from_checkpoint(str(path))
    assert restored.pooling_message_fp32 == cfg.model.pooling_message_fp32
    assert restored.pooling_chunk_size == cfg.model.pooling_chunk_size
    expected, actual = (
        predict(model, **batch["forward_kwargs"]),
        predict(restored, **batch["forward_kwargs"]),
    )
    for name in targets:
        key = as_nested_key(name)
        torch.testing.assert_close(expected[key], actual[key])
        assert expected[key].shape[:2] == (1, 11 if name.startswith("surface") else 13)
    normalizer = next(
        t
        for t in dataset.transforms
        if hasattr(t, "inverse_td") and hasattr(t, "_stats")
    )
    nondim = next(t for t in dataset.transforms if hasattr(t, "_fields"))
    field_types = build_redim_field_types(ds)
    true_pts = _to_pointwise(batch["targets"], "tensors")
    physical = redimensionalize(
        true_pts,
        normalizer=normalizer,
        nondim=nondim,
        field_types=field_types,
        global_data=domain.global_data,
    )
    # Check inversion against the selected original values, independent of transforms.
    raw, _ = dataset.reader[0]
    for name, source in sources.items():
        torch.testing.assert_close(
            physical[as_nested_key(name)], walk_path(raw, source), rtol=1e-5, atol=2e-5
        )
    output = tmp_path / f"export_{mode}.pdmsh"
    attach_and_save(
        domain,
        physical,
        physical,
        targets,
        output,
        rescale_geometry=True,
        target_sources=sources,
    )
    saved = DomainMesh.load(str(output))
    for name, source in sources.items():
        mesh_path, leaf = source.split(".point_data.")
        data = walk_path(saved, mesh_path).point_data
        parts = list(as_nested_key(leaf))
        parts[-1] = "pred_" + parts[-1]
        torch.testing.assert_close(data[tuple(parts)], physical[as_nested_key(name)])
        assert as_nested_key(leaf) not in data
    torch.testing.assert_close(saved.interior.points, raw.interior.points)
    if mode != "volume":
        context = ForceContext.from_config(
            OmegaConf.create({"enabled": True}), field_types, "cpu"
        )
        forces = context.coefficients(domain, true_pts, true_pts, normalizer, sources)
        assert forces is not None and forces[0] == forces[1]
    dataset.close()


def test_sampling_stream_independence(native_case):
    """Epoch changes train samples; query counts and order cannot change anchors."""
    root, _ = native_case
    reader = make_reader(root, **COUNTS)
    reader.set_generator(torch.Generator().manual_seed(37))
    first, _ = reader[0]
    again, _ = reader[0]
    reader.point_sets["surface_queries"]["count"] = 19
    reader.point_sets["interior"]["count"] = 23
    changed_queries, _ = reader[0]
    for name in (
        "geometry",
        "supernodes",
        "surface_anchors",
        "volume_anchors",
    ):
        torch.testing.assert_close(
            first.boundaries[name].points, again.boundaries[name].points
        )
        torch.testing.assert_close(
            first.boundaries[name].points, changed_queries.boundaries[name].points
        )
    reader.set_epoch(1)
    changed_epoch, _ = reader[0]
    assert not torch.equal(
        first.boundaries["geometry"].points,
        changed_epoch.boundaries["geometry"].points,
    )


@pytest.mark.parametrize(
    "mode,split",
    [
        ("surface", "manifest"),
        ("volume", "manifest"),
        ("joint", "manifest"),
        ("joint", "directory"),
        ("joint", "fallback"),
    ],
)
def test_anchor_training_loaders_keep_independent_validation_queries(
    native_case, mode, split, monkeypatch
):
    """Production train/inference loaders honor query counts during evaluation."""
    root, original = native_case
    cfg, ds, initial_dataset = pipeline(root, mode)
    initial_dataset.close()
    cfg.dataset = f"drivaer_ml_ab_upt_{mode}"
    anchor_queries(cfg.dataset_reader_overrides.point_sets, True)
    cfg.dataloader.prefetch_factor = 0
    cfg.dataloader.num_workers = 1
    cfg.dataloader.use_streams = False
    cfg.dataloader.pin_memory = False
    if split == "manifest":
        path = root / "run_2" / "domain_2.pdmsh"
        path.parent.mkdir()
        original.save(str(path))
        (root / "manifest.json").write_text(
            json.dumps({"train": ["run_1"], "val": ["run_2"]})
        )
    else:
        cfg.train_split = cfg.val_split = None
        if split == "directory":
            val_root = root / "validation"
            path = val_root / "run_2" / "domain_2.pdmsh"
            path.parent.mkdir(parents=True)
            original.save(str(path))
            ds.val_datadir = str(val_root)
    monkeypatch.setattr(datasets_module, "load_dataset_config", lambda _: ds)
    monkeypatch.setattr(
        datasets_module,
        "DistributedManager",
        lambda: SimpleNamespace(world_size=1, rank=0),
    )
    train_loader, val_loader, _, _ = build_dataloaders(cfg)
    try:
        train = next(iter(train_loader))["forward_kwargs"]
        val = next(iter(val_loader))["forward_kwargs"]
        branches = ["surface", "volume"] if mode == "joint" else [mode]
        for branch in branches:
            anchors, queries = f"{branch}_anchors", f"{branch}_queries"
            torch.testing.assert_close(train[queries], train[anchors])
            assert val[queries].shape[1] == COUNTS[queries]
            assert val[anchors].shape[1] == COUNTS[f"num_{branch}_anchors"]
            assert val[queries].shape[1] != val[anchors].shape[1]
        train_loader.set_epoch(4)
        later_val = next(iter(val_loader))["forward_kwargs"]
        for key in val:
            torch.testing.assert_close(later_val[key], val[key])
    finally:
        train_loader.dataset.close()
        val_loader.dataset.close()


def test_query_chunk_values_and_gradients(native_case):
    """The model reuses one differentiable context across all chunks."""
    root, _ = native_case
    cfg, ds, dataset = pipeline(root, "joint")
    collate = build_collate_fn(
        "tensors",
        OmegaConf.to_container(cfg.forward_kwargs),
        OmegaConf.to_container(ds.targets),
        OmegaConf.to_container(ds.target_sources),
    )
    batch = collate([dataset[0]])
    model = make_model(**SMALL, query_chunk_size=100)
    whole = predict(model, **batch["forward_kwargs"])
    sum(
        v.square().mean() for v in whole.values(include_nested=True, leaves_only=True)
    ).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.query_chunk_size = 3
    chunks = predict(model, **batch["forward_kwargs"])
    for key, value in whole.items(include_nested=True, leaves_only=True):
        torch.testing.assert_close(value, chunks[key], atol=2e-6, rtol=2e-5)
    sum(
        v.square().mean() for v in chunks.values(include_nested=True, leaves_only=True)
    ).backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], atol=2e-5, rtol=2e-4)
    dataset.close()


def test_surface_integration_measures(native_case):
    """Equal-area sampling recovers the complete surface integration measure."""
    root, original = native_case
    reader = make_reader(root, **COUNTS)
    sampled, _ = reader[0]
    torch.testing.assert_close(
        cell_measures(sampled.boundaries["vehicle"]).sum(),
        cell_measures(original.boundaries["vehicle"]).sum(),
    )


def test_target_contract_rejects_wrong_mode(native_case):
    """A mismatched model/dataset cannot silently drop one supervised head."""
    root, _ = native_case
    cfg, ds, dataset = pipeline(root, "joint")
    batch = build_collate_fn(
        "tensors",
        OmegaConf.to_container(cfg.forward_kwargs),
        OmegaConf.to_container(ds.targets),
        OmegaConf.to_container(ds.target_sources),
    )([dataset[0]])
    model = make_model(mode="surface", **SMALL)
    with pytest.raises(ValueError, match="disabled output head"):
        predict(model, **batch["forward_kwargs"])
    with pytest.raises(ValueError, match="exactly match"):
        extract_targets(dataset[0][0], {"surface.pressure": "scalar"}, {})
    dataset.close()


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_reader_keeps_labels_aligned_with_original_coordinates(native_case, mode):
    """Match sampled coordinates to the original mesh, independently of the reader."""
    root, original = native_case
    reader = make_reader(root, mode=mode, **COUNTS)
    reader.set_generator(torch.Generator().manual_seed(654))
    sampled, _ = reader[0]
    surface = sampled.boundaries.get("surface_queries")
    if mode != "volume":
        torch.testing.assert_close(
            surface.points, sampled.boundaries["vehicle"].cell_centroids
        )
    else:
        assert surface is None
    for enabled, query_mesh, source_positions, source_data, fields in (
        (
            mode != "volume",
            surface,
            original.boundaries["vehicle"].cell_centroids,
            original.boundaries["vehicle"].cell_data,
            (
                ("surface.pressure", "pMeanTrim"),
                ("surface.wss", "wallShearStressMeanTrim"),
            ),
        ),
        (
            mode != "surface",
            sampled.interior,
            original.interior.points,
            original.interior.point_data,
            (
                ("volume.velocity", "UMeanTrim"),
                ("volume.pressure", "pMeanTrim"),
                ("volume.nut", "nutMeanTrim"),
            ),
        ),
    ):
        if not enabled:
            assert query_mesh is None or query_mesh.n_points == 0
            continue
        distances = torch.cdist(query_mesh.points.double(), source_positions.double())
        nearest = distances.argmin(dim=1)
        torch.testing.assert_close(query_mesh.points, source_positions[nearest])
        assert nearest.unique().numel() == query_mesh.n_points
        for target, source in fields:
            torch.testing.assert_close(
                query_mesh.point_data[as_nested_key(target)],
                source_data[source][nearest],
            )


@pytest.mark.parametrize(
    "coordinate_dtypes,metadata_dtype,model_dtype",
    [
        ((torch.float32, torch.float64), torch.float64, torch.float32),
        ((torch.float32,), torch.float32, torch.float64),
    ],
)
def test_model_accepts_mixed_native_dtypes_without_changing_inputs(
    coordinate_dtypes, metadata_dtype, model_dtype
):
    """Native dtypes match model-precision execution and retain input gradients."""
    torch.manual_seed(81)
    model = make_model(**SMALL, query_chunk_size=4).to(dtype=model_dtype)
    coordinates = {
        name: torch.rand(1, count, 3)
        for name, count in (
            ("geometry", 19),
            ("supernodes", 5),
            ("surface_anchors", 7),
            ("volume_anchors", 9),
            ("surface_queries", 11),
            ("volume_queries", 13),
        )
    }
    baseline = {
        name: value.clone().requires_grad_() for name, value in coordinates.items()
    }
    baseline["parameters"] = torch.tensor([[[30.0, 1.0, 0.0]]], requires_grad=True)
    native = {
        name: value.to(coordinate_dtypes[i % len(coordinate_dtypes)])
        .clone()
        .requires_grad_()
        for i, (name, value) in enumerate(coordinates.items())
    }
    native["parameters"] = (
        baseline["parameters"].detach().to(metadata_dtype).clone().requires_grad_()
    )
    original = {name: value.detach().clone() for name, value in native.items()}
    expected = predict(model, **baseline)
    sum(
        v.square().mean()
        for v in expected.values(include_nested=True, leaves_only=True)
    ).backward()
    parameter_gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual = predict(model, **native)
    sum(
        v.square().mean() for v in actual.values(include_nested=True, leaves_only=True)
    ).backward()
    for key, value in actual.items(include_nested=True, leaves_only=True):
        assert value.dtype == model_dtype
        torch.testing.assert_close(value, expected[key])
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, parameter_gradients[name])
    for name, value in native.items():
        # Both the dtype and values used by export stay intact.
        torch.testing.assert_close(value, original[name])
        assert value.grad is not None and torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad.float(), baseline[name].grad)


@pytest.mark.parametrize(
    "override",
    [
        {"coordinate_scale": 1.0},
        {"pooling_radius": 0.5},
        {"num_heads": 4},
        {"num_interaction_blocks": 0, "num_shared_layers": 3},
        {"train_on_anchors": True},
        {"parameter_scale": 40.0},
    ],
)
def test_production_checkpoint_load_rejects_changed_numerics(tmp_path, override):
    """Weight-compatible settings must not silently change the restored field."""
    path = tmp_path / "semantics.mdlus"
    source = make_model(**SMALL)
    source.save(str(path))
    destination = make_model(**(SMALL | override))
    with pytest.raises(ValueError, match="checkpoint settings"):
        destination.load(str(path))


def test_production_checkpoint_load_allows_decode_tuning(tmp_path):
    """Operational chunk sizes/backend/precision can change on continuation."""
    path = tmp_path / "decode-tuning.mdlus"
    source = make_model(**SMALL, query_chunk_size=17)
    source.save(str(path))
    destination = make_model(
        **(
            SMALL
            | {
                "pooling_chunk_size": 97,
                "pooling_message_fp32": True,
                "radius_search_backend": None,
            }
        ),
        query_chunk_size=3,
    )
    destination.load(str(path))
    assert destination.pooling_message_fp32
    assert not source.pooling_message_fp32
    args = {
        name: torch.rand(1, count, 3)
        for name, count in (
            ("geometry", 19),
            ("supernodes", 5),
            ("surface_anchors", 7),
            ("volume_anchors", 9),
            ("surface_queries", 11),
            ("volume_queries", 13),
        )
    }
    args["parameters"] = torch.tensor([[30.0, 0.0, 0.0]])
    expected, actual = predict(source, **args), predict(destination, **args)
    for key, value in expected.items(include_nested=True, leaves_only=True):
        torch.testing.assert_close(actual[key], value, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_anchor_supervision_aligns_labels_and_keeps_independent_evaluation(
    native_case, mode, monkeypatch
):
    """Anchor training gathers matching labels; eval accepts arbitrary queries."""
    root, original = native_case
    cfg, ds, dataset = pipeline(root, mode)
    anchor_queries(dataset.reader.point_sets, True)
    raw, _ = dataset.reader[0]
    for field, enabled, query_mesh, original_mesh in (
        (
            "surface",
            mode != "volume",
            raw.boundaries.get("surface_queries"),
            original.boundaries["vehicle"],
        ),
        ("volume", mode != "surface", raw.interior, original.interior),
    ):
        if not enabled:
            continue
        torch.testing.assert_close(
            query_mesh.points, raw.boundaries[f"{field}_anchors"].points
        )
        source_points = (
            original_mesh.cell_centroids if field == "surface" else original_mesh.points
        )
        source_data = (
            original_mesh.cell_data if field == "surface" else original_mesh.point_data
        )
        nearest = torch.cdist(query_mesh.points, source_points).argmin(dim=1)
        torch.testing.assert_close(query_mesh.points, source_points[nearest])
        torch.testing.assert_close(
            query_mesh.point_data[field, "pressure"], source_data["pMeanTrim"][nearest]
        )
    collate = build_collate_fn(
        "tensors",
        OmegaConf.to_container(cfg.forward_kwargs),
        OmegaConf.to_container(ds.targets),
        OmegaConf.to_container(ds.target_sources),
    )
    batch = collate([dataset[0]])
    model = make_model(mode=mode, train_on_anchors=True, **SMALL)
    decode_queries = model.decode_queries

    def unexpected_query_decode(*args, **kwargs):
        pytest.fail("Anchor training must not evaluate a separate query stream")

    monkeypatch.setattr(model, "decode_queries", unexpected_query_decode)
    prediction = predict(model, **batch["forward_kwargs"])
    loss, _ = LossCalculator(OmegaConf.to_container(ds.targets), loss_type="mse")(
        prediction, batch["targets"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # Equal-size permutations must fail too: shape checks alone miss label swaps.
    field = "surface" if mode != "volume" else "volume"
    bad = dict(batch["forward_kwargs"])
    bad[f"{field}_queries"] = bad[f"{field}_queries"].flip(1)
    with pytest.raises(ValueError, match="coordinates, and order"):
        predict(model, **bad)
    bad[f"{field}_queries"] = bad[f"{field}_queries"][:, :-1]
    with pytest.raises(ValueError, match="coordinates, and order"):
        predict(model, **bad)
    monkeypatch.setattr(model, "decode_queries", decode_queries)
    model.eval()
    anchor_queries(dataset.reader.point_sets, False)
    independent = collate([dataset[0]])
    with torch.no_grad():
        actual = predict(model, **independent["forward_kwargs"])
    for key, value in actual.items(include_nested=True, leaves_only=True):
        assert value.shape == independent["targets"][key].shape
        assert value.shape[1] == (11 if key[0] == "surface" else 13)
    # Full-surface evaluation is not constrained by the training anchor count.
    if mode != "volume":
        dataset.reader.point_sets["surface_queries"]["count"] = None
        full = collate([dataset[0]])
        with torch.no_grad():
            actual = predict(model, **full["forward_kwargs"])
        assert (
            actual["surface", "pressure"].shape[1]
            == original.boundaries["vehicle"].n_cells
        )
    dataset.close()


def test_parameter_scaling_matches_explicit_unit_conversion(tmp_path):
    """Only operator conditioning changes; metadata and query outputs stay intact."""
    torch.manual_seed(701)
    scaled = make_model(**SMALL, parameter_scale=40.0, train_on_anchors=True).eval()
    unit = make_model(**SMALL).eval()
    unit.load_state_dict(
        {**scaled.state_dict(), "_extra_state": unit.get_extra_state()}
    )
    args = {
        name: torch.rand(1, count, 3)
        for name, count in (
            ("geometry", 19),
            ("supernodes", 5),
            ("surface_anchors", 7),
            ("volume_anchors", 9),
            ("surface_queries", 11),
            ("volume_queries", 13),
        )
    }
    parameters = torch.tensor([[[40.0, 2.0, 0.0]]], requires_grad=True)
    original = parameters.detach().clone()
    actual = predict(scaled, **args, parameters=parameters)
    expected = predict(unit, **args, parameters=parameters / 40.0)
    for key, value in actual.items(include_nested=True, leaves_only=True):
        torch.testing.assert_close(value, expected[key])
    sum(v.square().mean() for v in actual.values(True, True)).backward()
    assert parameters.grad is not None and torch.isfinite(parameters.grad).all()
    torch.testing.assert_close(parameters.detach(), original)
    path = tmp_path / "scaled-anchor-training.mdlus"
    scaled.save(str(path))
    restored = Module.from_checkpoint(str(path)).eval()
    assert restored.parameter_scale == 40.0 and restored.train_on_anchors
    result = predict(restored, **args, parameters=parameters.detach())
    for key, value in actual.items(include_nested=True, leaves_only=True):
        torch.testing.assert_close(result[key], value)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_anchor_training_compiles_without_validation_graph_breaks(monkeypatch, device):
    """Compile the model and attention blocks; compare both branches' gradients."""
    torch.manual_seed(42)
    model = make_model(**(SMALL | {"coordinate_scale": 1}), train_on_anchors=True).to(
        device
    )
    args = {
        name: torch.rand(1, count, 3, device=device)
        for name, count in (
            ("geometry", 19),
            ("supernodes", 5),
            ("surface_anchors", 7),
            ("volume_anchors", 9),
        )
    }
    args.update(
        parameters=torch.ones(1, 3, device=device),
        surface_queries=args["surface_anchors"].clone(),
        volume_queries=args["volume_anchors"].clone(),
    )
    # Radius search has separate dynamic-shape graph breaks. Freeze only its
    # pooled output so fullgraph detects validation breaks in the real model.
    pooled = model._pool(args["geometry"], args["supernodes"]).detach()
    monkeypatch.setattr(model, "_pool", lambda geometry, supernodes: pooled)
    expected = predict(model, **args)
    actual = predict(torch.compile(model, backend="aot_eager", fullgraph=True), **args)
    for key, value in expected.items(include_nested=True, leaves_only=True):
        torch.testing.assert_close(actual[key], value)
    weights = (model.surface_head.weight, model.volume_head.weight)
    expected_loss = sum(t.square().mean() for t in expected.values(True, True))
    actual_loss = sum(t.square().mean() for t in actual.values(True, True))
    expected_grads = torch.autograd.grad(expected_loss, weights)
    actual_grads = torch.autograd.grad(actual_loss, weights)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
