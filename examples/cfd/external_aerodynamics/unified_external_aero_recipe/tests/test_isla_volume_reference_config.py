# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interior reference configuration composes and pairs the corrected boundary measure with the total-measure scale (2026-09-14)."""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from domain_transforms import ComposeQuadratureMeasure  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import compose_measure_weights  # noqa: E402


def test_isla_volume_reference_composes_with_the_corrected_measure_dataset():
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "drivaer_ml_volume_reference.yaml")
    with initialize_config_dir(config_dir=str(_RECIPE_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=["model=isla_volume_reference", "dataset=drivaer_ml_volume_reference", "+out_dim=5"])
    m = cfg.model.model
    assert m.frame_mode == "relative" and m.scale_mode == "total_measure"
    assert m.query_tokens is True and m.n_query_scalars == 1
    assert cfg.model.forward_kwargs.measure_weights == "boundaries.vehicle.cell_data.quadrature_measure"
    targets = [t["_target_"] for t in ds.pipeline.transforms]
    assert targets[-1].endswith("ComposeQuadratureMeasure"), targets
    assert targets[-2].endswith("DropDegenerateCells")
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
