# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ISLA recipe configurations compose to the intended architecture (2026-09-15).

The plain names carry the reference configuration (relative frame, total-measure scale):
``model=isla_surface`` and ``model=isla_volume``. ``isla_surface_constant_gauge`` pins the centered
frame with a fixed reference length (the ablation). The deprecated ``_reference`` names resolve to
the plain files. The interior reference pairs the Horvitz-Thompson-corrected boundary measure with
the total-measure scale (2026-09-14)."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from domain_transforms import ComposeQuadratureMeasure  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import compose_measure_weights  # noqa: E402

ISLA_TARGET = "physicsnemo.experimental.nn.ISLA"


def _compose(model, dataset):
    """The recipe's model configs are `@package _global_`: the architecture block lands at cfg.model and
    the data-to-model mapping at cfg.forward_kwargs."""
    with initialize_config_dir(config_dir=str(_RECIPE_ROOT / "conf"), version_base=None):
        return compose(config_name="train", overrides=[f"model={model}", f"dataset={dataset}", "+out_dim=5"])


def test_plain_surface_name_is_the_reference_configuration():
    cfg = _compose("isla_surface", "drivaer_ml_surface")
    m = cfg.model
    assert m._target_ == ISLA_TARGET
    assert m.frame_mode == "relative" and m.scale_mode == "total_measure"
    assert "reference_length" not in m
    assert m.geo_checkpoint is True
    assert cfg.forward_kwargs.global_vectors == "global_data.U_inf_dir"
    assert cfg.forward_kwargs.measure_weights == "interior.point_data._target_quadrature_measure"


def test_constant_gauge_variant_pins_the_centered_frame():
    m = _compose("isla_surface_constant_gauge", "drivaer_ml_surface").model
    assert m._target_ == ISLA_TARGET
    assert m.frame_mode == "centered" and m.scale_mode == "reference_length"
    assert m.reference_length == 8.0


@pytest.mark.parametrize(
    "deprecated, plain, dataset",
    [
        ("isla_surface_reference", "isla_surface", "drivaer_ml_surface"),
        ("isla_volume_reference", "isla_volume", "drivaer_ml_volume_reference"),
    ],
)
def test_deprecated_reference_names_resolve_to_the_plain_configs(deprecated, plain, dataset):
    old, new = _compose(deprecated, dataset), _compose(plain, dataset)
    for key in ("model", "forward_kwargs"):
        assert OmegaConf.to_container(old[key], resolve=False) == OmegaConf.to_container(new[key], resolve=False), key


def test_isla_volume_composes_with_the_corrected_measure_dataset():
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "drivaer_ml_volume_reference.yaml")
    cfg = _compose("isla_volume", "drivaer_ml_volume_reference")
    m = cfg.model
    assert m.frame_mode == "relative" and m.scale_mode == "total_measure"
    assert m.query_tokens is True and m.n_query_scalars == 1
    assert cfg.forward_kwargs.measure_weights == "boundaries.vehicle.cell_data.quadrature_measure"
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)  # ${dp:...} nodes kept as strings
    targets = [t["_target_"] for t in raw]
    assert "ComposeQuadratureMeasure" in targets[-1], targets
    assert "DropDegenerateCells" in targets[-2], targets
    assert ds.pipeline.reader.boundary_subsample == "cells"


def test_total_measure_scale_sees_the_surface_area_through_the_corrected_field():
    """End to end on a toy surface: raw areas of a 1-in-5 subsample sum to a fifth of the
    body; the corrected field sums to the body, so the total-measure length unit is the
    body's and not the subsample's."""
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(3000, 3, generator=g)
    mesh = Mesh(points=pts, cells=torch.arange(3000).reshape(1000, 3))
    full = float(mesh.cell_areas.sum())
    sub = mesh.slice_cells(torch.arange(0, 1000, 5))
    compose_measure_weights(sub, 5.0)
    sub = ComposeQuadratureMeasure()(sub)
    raw_L = float(sub.cell_areas.sum().sqrt()); corrected_L = float(sub.cell_data["quadrature_measure"].sum().sqrt())
    assert abs(corrected_L**2 - full) / full < 0.15
    assert raw_L**2 / full < 0.3
