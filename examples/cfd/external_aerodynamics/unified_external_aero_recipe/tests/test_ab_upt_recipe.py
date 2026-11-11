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

"""AB-UPT surface and volume recipes end to end on a native domain mesh."""

from pathlib import Path

import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import (
    _apply_dataset_reader_overrides,
    build_dataset,
    find_normalizer,
)
from forces import ForceContext
from hydra import compose, initialize_config_dir
from infer import (
    _to_pointwise,
    attach_and_save,
    build_redim_field_types,
    redimensionalize,
)
from loss import LossCalculator
from nondim import NonDimensionalizeByMetadata
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict
from utils import field_dim

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {
    "geometry": 17,
    "supernodes": 5,
    "surface_anchors": 7,
    "volume_anchors": 9,
    "interior": 13,
}
SMALL_MODEL = {
    "hidden_dim": 24,
    "num_heads": 3,
    "num_interaction_blocks": 1,
    "num_shared_layers": 1,
    "num_decoder_layers": 1,
    "pooling_radius": 2.0,
    "query_chunk_size": 4,
    "radius_search_backend": "torch",
}
RAW_NAMES = {
    "surface": {"pressure": "pMeanTrim", "wss": "wallShearStressMeanTrim"},
    "volume": {"velocity": "UMeanTrim", "pressure": "pMeanTrim", "nut": "nutMeanTrim"},
}


@pytest.fixture
def native_case(tmp_path):
    """Write a real memory-mapped domain with surface and volume labels."""
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


@pytest.mark.parametrize("mode", ["surface", "volume"])
def test_train_step_forces_and_export(native_case, mode, tmp_path):
    """The shipped YAMLs train, integrate forces and export in physical units."""
    root, original = native_case
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model=ab_upt_{mode}"])
    for name, spec in cfg.dataset_reader_overrides.point_sets.items():
        spec.count = COUNTS[name]
    ds = OmegaConf.load(ROOT / "datasets" / f"drivaer_ml_ab_upt_{mode}.yaml")
    ds.train_datadir = str(root)
    ds.sampling_resolution = COUNTS["interior"]
    ds = _apply_dataset_reader_overrides(ds, cfg)
    dataset = build_dataset(ds, device="cpu", num_workers=1)
    dataset.reader.set_generator(torch.Generator().manual_seed(123))
    targets = OmegaConf.to_container(ds.targets)

    domain, meta = dataset[0]
    assert domain.interior.n_points == COUNTS["interior"]
    collate = build_collate_fn(
        "tensors", OmegaConf.to_container(cfg.forward_kwargs), targets
    )
    batch = collate([(domain, meta)])

    # Production derives out_dim from the dataset's targets in build_dataloaders.
    OmegaConf.update(
        cfg, "out_dim", sum(field_dim(t) for t in targets.values()), force_add=True
    )
    model = hydra.utils.instantiate(
        OmegaConf.merge(OmegaConf.to_container(cfg.model, resolve=True), SMALL_MODEL)
    )
    output = model(**batch["forward_kwargs"])
    pred = normalize_output_to_tensordict(output, targets, "tensors")
    loss, _ = LossCalculator(targets, loss_type="mse")(pred, batch["targets"])
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

    normalizer = find_normalizer([dataset])
    nondim = next(
        t for t in dataset.transforms if isinstance(t, NonDimensionalizeByMetadata)
    )
    field_types = build_redim_field_types(ds)
    true_pts = _to_pointwise(batch["targets"], "tensors")
    raw, _ = dataset.reader[0]
    if mode == "surface":
        # Equal-area sampling recovers the complete surface integration measure.
        torch.testing.assert_close(
            cell_measures(raw.boundaries["vehicle"]).sum(),
            cell_measures(original.boundaries["vehicle"]).sum(),
        )
        context = ForceContext.from_config(
            OmegaConf.create({"enabled": True}), field_types, "cpu"
        )
        forces = context.coefficients(domain, true_pts, true_pts, normalizer)
        assert forces is not None and forces[0] == forces[1]

    # Inversion recovers the selected raw values, independent of the transforms.
    physical = redimensionalize(
        true_pts,
        normalizer=normalizer,
        nondim=nondim,
        field_types=field_types,
        global_data=domain.global_data,
    )
    for name in targets:
        torch.testing.assert_close(
            physical[name], raw.interior.point_data[name], rtol=1e-5, atol=2e-5
        )
    if mode == "surface":
        source = original.boundaries["vehicle"]
        positions, source_data = source.cell_centroids, source.cell_data
    else:
        positions, source_data = original.interior.points, original.interior.point_data
    nearest = torch.cdist(raw.interior.points, positions).argmin(dim=1)
    for name, raw_name in RAW_NAMES[mode].items():
        torch.testing.assert_close(
            raw.interior.point_data[name], source_data[raw_name][nearest]
        )

    output_path = tmp_path / f"export_{mode}.pdmsh"
    attach_and_save(
        domain, physical, physical, targets, output_path, rescale_geometry=True
    )
    saved = DomainMesh.load(str(output_path))
    for name in targets:
        torch.testing.assert_close(
            saved.interior.point_data[f"pred_{name}"], physical[name]
        )
    torch.testing.assert_close(saved.interior.points, raw.interior.points)
    dataset.close()
