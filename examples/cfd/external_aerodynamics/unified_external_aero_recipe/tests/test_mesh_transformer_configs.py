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

"""Synthetic smoke tests for the MeshTransformer templates (both DrivAerML arms).

The MeshTransformer enters the recipe as a DomainMesh-native model with a
zero-glue contract: ``forward_kwargs: {domain: ""}`` hands the collated
``DomainMesh`` sample straight to ``forward(domain) -> Mesh``.  Two arms are
pre-registered for the DrivAerML campaign:

- ``mesh_transformer_surface`` -- vehicle-only (the fair-comparison arm,
  same information diet as every other recipe model);
- ``mesh_transformer_surface_allbc`` -- all five typed boundaries
  (``vehicle``, ``no_slip``, ``slip``, ``inlet``, ``outlet``; the H-ground
  arm), fed by the ``drivaer_ml_surface_allbc`` dataset variant;
- ``mesh_transformer_surface_trace`` -- the declared boundary-trace arm
  (``trace_of: vehicle``), the pre-registered acceptance experiment of the
  GeoTransolver-gap verdict: identical to the vehicle-only arm except the
  one declaration, and dependent on the pipeline's alignment guarantee
  (interior.points IS boundaries.vehicle.cell_centroids, in cell order),
  which ``test_surface_pipeline_pins_declared_trace_alignment`` pins
  bitwise.

Each template test composes ``conf/train.yaml`` exactly like the other
synthetic config tests, builds a tiny synthetic post-pipeline ``DomainMesh``
matching the recipe contract (real triangulations -- the MeshTransformer
rejects zero-area cells, so the random-connectivity conftest factories are
not usable here), runs the collate + ``forward``, checks output shapes and
finiteness, and runs a full backward with
``kernel_checkpoint_query_chunks=true`` (pinned on in both templates -- the
training-memory requirement at recipe scale) through multiple decode chunks.

The dataset-side tests cover the recipe-local transforms that expose the
auxiliary boundaries (``BoundaryMeshToDomainMesh``), the domain-level
scale-gauge injection (``SetDomainGlobalField``), and the unit-freestream
derivation (``ComputeFreestreamDirection``), plus end-to-end runs of both
dataset YAMLs' transform chains (reader excluded) from raw field names
through to a model forward.

The all-BC chain test doubles as the regression test for the cluster NaN
(2026-07): it rebuilds the REAL DrivAerML shape statistics -- five typed
boundaries at tunnel proportions (x in [-40, 80] m), auxiliary cell areas
O(1-7) m^2 against vehicle cell areas O(1e-9..1e-4) m^2, L_ref = 5 -- and
requires finite forward AND backward.  Root cause of the original NaN: the
model's zero-preserving drive stream is norm-free, so its magnitude is set
entirely by the inputs; the raw physical U_inf (~39 m/s) drive times
tunnel-scale geometry invariants (sources at |x| ~ 16 gauge units under
the old vehicle-scale gauge) overflowed float32 through the score/value
products.  Fixed by the unit-direction drive + the tunnel-scale gauge
(reference_length: 8.0), both now pinned by these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import _maybe_inject_targets
from domain_transforms import (
    BoundaryMeshToDomainMesh,
    ComputeFreestreamDirection,
    SetDomainGlobalField,
    TopologyAwareDomainMeshReader,
)
from hydra import compose, initialize_config_dir
from loss import LossCalculator
from omegaconf import DictConfig, OmegaConf
from output_normalize import normalize_output_to_tensordict

from physicsnemo.datapipes.readers.mesh import DomainMeshReader
from physicsnemo.mesh import DomainMesh, Mesh

_RECIPE_ROOT = Path(__file__).resolve().parent.parent

### The recipe's diagnostic tools are importable modules (tools/ is not a
### package; mirror conftest's src insertion).
if str(_RECIPE_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT / "tools"))

import probe_boundary_contributions as probe  # noqa: E402

_ALL_BOUNDARIES = ("vehicle", "no_slip", "slip", "inlet", "outlet")
_TARGETS = {"pressure": "scalar", "wss": "vector"}

### Small enough for CPU seconds, large enough that the decode runs several
### query chunks under the shrunk `query_chunk_size` below.
_QUERY_CHUNK_SIZE = 8


def _compose_train_cfg(model: str, dataset: str) -> DictConfig:
    """Compose ``conf/train.yaml`` with the given model + dataset selection.

    The MeshTransformer templates (like GLOBE's) declare per-target
    ``output_field_ranks`` and never reference ``${out_dim}``, so no
    ``out_dim`` injection is needed here.
    """
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"),
        version_base=None,
    ):
        return compose(
            config_name="train",
            overrides=[f"model={model}", f"dataset={dataset}"],
        )


### ---------------------------------------------------------------------------
### Synthetic geometry: real triangulations (positive cell areas)
### ---------------------------------------------------------------------------


def _patch(
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    n_u: int,
    n_v: int,
) -> Mesh:
    """Planar rectangular patch, triangulated into non-degenerate triangles.

    ``origin + s*u + t*v`` for ``s, t`` on a regular grid; every triangle has
    strictly positive area (``u``, ``v`` linearly independent), which the
    MeshTransformer requires of boundary cells.
    """
    o = torch.tensor(origin, dtype=torch.float32)
    uu = torch.tensor(u, dtype=torch.float32)
    vv = torch.tensor(v, dtype=torch.float32)
    s = torch.linspace(0.0, 1.0, n_u)
    t = torch.linspace(0.0, 1.0, n_v)
    points = (
        o[None, None, :]
        + s[:, None, None] * uu[None, None, :]
        + t[None, :, None] * vv[None, None, :]
    ).reshape(-1, 3)

    cells = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i * n_v + j
            b = (i + 1) * n_v + j
            cells.append([a, b, a + 1])
            cells.append([b, b + 1, a + 1])
    return Mesh(points=points, cells=torch.tensor(cells, dtype=torch.int64))


def _boundary_meshes(names: tuple[str, ...]) -> dict[str, Mesh]:
    """A tiny wind-tunnel-like layout: vehicle + optional far boundaries."""
    layouts = {
        ### The "car": a patch above the ground, finer than the others.
        "vehicle": ((-0.5, -0.3, 0.2), (1.0, 0.0, 0.0), (0.0, 0.6, 0.0), 6, 5),
        ### Ground plane under the vehicle.
        "no_slip": ((-2.0, -1.0, 0.0), (4.0, 0.0, 0.0), (0.0, 2.0, 0.0), 4, 3),
        ### Ceiling.
        "slip": ((-2.0, -1.0, 2.0), (4.0, 0.0, 0.0), (0.0, 2.0, 0.0), 3, 3),
        ### Upstream / downstream faces.
        "inlet": ((-2.0, -1.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0), 3, 3),
        "outlet": ((2.0, -1.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0), 3, 3),
    }
    return {name: _patch(*layouts[name]) for name in names}


def _scattered_tiny_triangles(
    n: int,
    area_lo: float,
    area_hi: float,
    box_lo: tuple[float, float, float],
    box_hi: tuple[float, float, float],
    *,
    seed: int,
) -> Mesh:
    """``n`` disconnected triangles with log-uniform areas, random orientation.

    Mimics the real DrivAerML vehicle after the reader subsample: cells with
    areas spanning several decades (2.3e-9..6.6e-5 m^2 on run_1), scattered
    over a car-sized box. Disconnected triangles are a legal codimension-one
    Mesh and every cell has exactly the sampled positive area.
    """
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor(box_lo)
    hi = torch.tensor(box_hi)
    centroids = lo + (hi - lo) * torch.rand(n, 3, generator=g)
    log_lo = torch.log(torch.tensor(area_lo))
    log_hi = torch.log(torch.tensor(area_hi))
    areas = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(n, generator=g))
    ### Random orthonormal in-plane frame per triangle.
    a = torch.randn(n, 3, generator=g)
    b = torch.randn(n, 3, generator=g)
    u = a / a.norm(dim=-1, keepdim=True)
    b = b - (b * u).sum(-1, keepdim=True) * u
    v = b / b.norm(dim=-1, keepdim=True)
    side = (4.0 * areas / (3.0**0.5)).sqrt()  # equilateral side for the area
    p0 = centroids - side[:, None] * (u * 0.5 + v * (3.0**0.5) / 6.0)
    p1 = p0 + side[:, None] * u
    p2 = p0 + side[:, None] * (u * 0.5 + v * (3.0**0.5) / 2.0)
    idx = torch.arange(n)
    return Mesh(
        points=torch.cat([p0, p1, p2], dim=0),
        cells=torch.stack([idx, idx + n, idx + 2 * n], dim=1),
    )


def _tunnel_boundaries(n_vehicle_cells: int) -> dict[str, Mesh]:
    """The REAL DrivAerML shape statistics (run_1 probe), in meters.

    Tunnel proportions x in [-40, 80]; auxiliary boundaries at their real
    cell counts and per-cell area scales (outlet/inlet 722 cells at 1.2,
    no_slip 722 at 5.0, slip 2,888 at 2.3-7.3); the vehicle as
    ``n_vehicle_cells`` tiny scattered triangles (areas 2.3e-9..6.6e-5)
    over a car-sized box near the origin. This is the geometry that
    reproduced the cluster NaN under the original vehicle-scale gauge.
    """
    return {
        "vehicle": _scattered_tiny_triangles(
            n_vehicle_cells,
            2.3e-9,
            6.6e-5,
            (0.0, -1.0, 0.0),
            (5.0, 1.0, 1.6),
            seed=7,
        ),
        ### 22 m x 40 m faces, 722 triangles each -> area 1.219 per cell.
        "outlet": _patch(
            (80.0, -11.0, 0.0), (0.0, 22.0, 0.0), (0.0, 0.0, 40.0), 20, 20
        ),
        "inlet": _patch(
            (-40.0, -11.0, 0.0), (0.0, 22.0, 0.0), (0.0, 0.0, 40.0), 20, 20
        ),
        ### Ground: 120 m x 30.2 m, 722 triangles -> area 5.019 per cell.
        "no_slip": _patch(
            (-40.0, -15.1, 0.0), (120.0, 0.0, 0.0), (0.0, 30.2, 0.0), 20, 20
        ),
        ### Two side walls: 120 m x 40 m, 1,444 triangles each -> area 3.3.
        "slip": Mesh.merge(
            [
                _patch(
                    (-40.0, -15.1, 0.0), (120.0, 0.0, 0.0), (0.0, 0.0, 40.0), 39, 20
                ),
                _patch((-40.0, 15.1, 0.0), (120.0, 0.0, 0.0), (0.0, 0.0, 40.0), 39, 20),
            ]
        ),
    }


### The constant scale gauge both dataset pipelines inject (8.0 in L_ref
### units = 40 m physical, the tunnel scale). See the dataset YAML comments.
_REFERENCE_LENGTH = 8.0


def _global_data(*, post_pipeline: bool = True) -> dict[str, torch.Tensor]:
    """Case-level freestream record, mirroring the curated DrivAerML layout.

    With ``post_pipeline=True`` the record additionally carries the two
    leaves the dataset pipelines derive: the constant ``reference_length``
    gauge (``SetGlobalField`` / ``SetDomainGlobalField``) and the unit
    freestream direction ``U_inf_dir`` (``ComputeFreestreamDirection``).
    """
    gd = {
        "U_inf": torch.tensor([38.889, 0.0, 0.0]),
        "p_inf": torch.tensor(0.0),
        "rho_inf": torch.tensor(1.225),
        "nu": torch.tensor(1.5e-5),
        "L_ref": torch.tensor(5.0),
    }
    if post_pipeline:
        gd["reference_length"] = torch.tensor(_REFERENCE_LENGTH)
        gd["U_inf_dir"] = torch.tensor([1.0, 0.0, 0.0])
    return gd


def _post_pipeline_domain(boundary_names: tuple[str, ...]) -> DomainMesh:
    """Synthetic post-pipeline surface DomainMesh per the recipe contract.

    interior = vehicle cell centroids carrying the targets; boundaries =
    the requested typed set (geometry-only; the model ignores undeclared
    cell fields such as the pipelines' precomputed ``normals``).
    """
    boundaries = _boundary_meshes(boundary_names)
    n = boundaries["vehicle"].n_cells
    interior = Mesh(
        points=boundaries["vehicle"].cell_centroids.clone(),
        point_data={
            "pressure": torch.randn(n),
            "wss": torch.randn(n, 3) * 0.1,
        },
    )
    return DomainMesh(
        interior=interior,
        boundaries=boundaries,
        global_data=_global_data(),
    )


### ---------------------------------------------------------------------------
### Template smoke tests: compose -> collate -> forward -> backward
### ---------------------------------------------------------------------------


_ARM_SPECS = [
    ("mesh_transformer_surface", "drivaer_ml_surface", ("vehicle",)),
    ("mesh_transformer_surface_trace", "drivaer_ml_surface", ("vehicle",)),
    (
        "mesh_transformer_surface_allbc",
        "drivaer_ml_surface_allbc",
        _ALL_BOUNDARIES,
    ),
    (
        "mesh_transformer_surface_allbc_pooled",
        "drivaer_ml_surface_allbc",
        _ALL_BOUNDARIES,
    ),
]


@pytest.mark.parametrize(
    "model,dataset,boundary_names",
    _ARM_SPECS,
    ids=[
        "vehicle_only",
        "vehicle_trace",
        "all_typed_boundaries",
        "all_typed_boundaries_pooled",
    ],
)
def test_mesh_transformer_template_forward_backward(
    model: str, dataset: str, boundary_names: tuple[str, ...]
) -> None:
    """Hydra-composed template runs forward + backward on a synthetic domain."""
    torch.manual_seed(0)
    train_cfg = _compose_train_cfg(model, dataset)

    assert OmegaConf.select(train_cfg, "input_type") == "mesh"
    assert OmegaConf.select(train_cfg, "output_type") == "mesh"
    ### The training-memory knob is a hard requirement at recipe scale
    ### (50k-200k surface points); pin it on in the template.
    assert train_cfg.model.kernel_checkpoint_query_chunks is True
    ### Both arms must run on the same explicit scale gauge (the intrinsic
    ### gauge would differ wildly between the vehicle and the full tunnel).
    assert train_cfg.model.reference_length_key == "reference_length"

    domain = _post_pipeline_domain(boundary_names)
    forward_kwargs_spec = OmegaConf.to_container(train_cfg.forward_kwargs, resolve=True)
    collate = build_collate_fn(
        input_type="mesh",
        forward_kwargs_spec=forward_kwargs_spec,
        target_config=_TARGETS,
    )
    batch = collate([(domain, {})])
    ### The zero-glue contract: the collate hands the DomainMesh itself in.
    assert batch["forward_kwargs"]["domain"] is domain

    ### Shrink only the decode chunk so the checkpointed chunk loop runs
    ### several times; every capacity knob stays at template values.
    small_model_cfg = OmegaConf.merge(
        train_cfg.model, OmegaConf.create({"query_chunk_size": _QUERY_CHUNK_SIZE})
    )
    model_inst = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    assert domain.interior.n_points > _QUERY_CHUNK_SIZE  # multiple chunks
    ### The pooled arm must actually carry its per-boundary gain parameters.
    if OmegaConf.select(train_cfg, "model.per_boundary_moment_pool", default=False):
        assert any(
            name.endswith("moment_segment_log_gain")
            for name, _ in model_inst.named_parameters()
        )

    output = model_inst(**batch["forward_kwargs"])
    assert isinstance(output, Mesh)
    pred_td = normalize_output_to_tensordict(output, _TARGETS, "mesh")

    for name in _TARGETS:
        pred_t = pred_td[name]
        target_t = batch["targets"][name]
        assert pred_t.shape == target_t.shape, (
            f"shape mismatch for {name}: pred={tuple(pred_t.shape)} "
            f"vs target={tuple(target_t.shape)}"
        )
        assert torch.isfinite(pred_t).all(), f"{name} prediction not finite"

    ### Backward through the checkpointed kernel decode: finite loss and
    ### finite, non-trivial gradients.
    lc = LossCalculator(
        target_config=_TARGETS,
        loss_type=train_cfg.training.loss_type,
    )
    loss, _ = lc(pred_td.float(), batch["targets"].float())
    assert torch.isfinite(loss), f"loss not finite: {float(loss)}"
    loss.backward()

    grad_norm_sq = 0.0
    for name, p in model_inst.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
        grad_norm_sq += float(p.grad.square().sum())
    assert grad_norm_sq > 0.0, "backward produced an all-zero gradient"


def test_arm_templates_differ_only_in_declared_deltas() -> None:
    """The four arms' model blocks differ only by their declared deltas.

    This pins the pre-registered comparisons: capacity, kernel dictionary,
    field mode, and (critically) the scale gauge must not drift between the
    arms.  vehicle-only vs all-BC differ only in ``boundary_field_ranks``;
    the pooled arm differs from all-BC only by
    ``per_boundary_moment_pool: true``; the trace arm differs from
    vehicle-only only by ``trace_of: vehicle`` (the matched-protocol
    requirement of the boundary-trace acceptance experiment).
    """
    cfg_vehicle = _compose_train_cfg("mesh_transformer_surface", "drivaer_ml_surface")
    cfg_trace = _compose_train_cfg(
        "mesh_transformer_surface_trace", "drivaer_ml_surface"
    )
    cfg_allbc = _compose_train_cfg(
        "mesh_transformer_surface_allbc", "drivaer_ml_surface_allbc"
    )
    cfg_pooled = _compose_train_cfg(
        "mesh_transformer_surface_allbc_pooled", "drivaer_ml_surface_allbc"
    )
    vehicle_model = OmegaConf.to_container(cfg_vehicle.model, resolve=True)
    trace_model = OmegaConf.to_container(cfg_trace.model, resolve=True)
    allbc_model = OmegaConf.to_container(cfg_allbc.model, resolve=True)
    pooled_model = OmegaConf.to_container(cfg_pooled.model, resolve=True)

    assert pooled_model.pop("per_boundary_moment_pool") is True
    assert pooled_model == allbc_model

    ### The trace acceptance experiment is matched-protocol by construction:
    ### exactly one declared delta against the plain singpair arm.
    assert trace_model.pop("trace_of") == "vehicle"
    assert trace_model == vehicle_model

    assert set(vehicle_model.pop("boundary_field_ranks")) == {"vehicle"}
    assert set(allbc_model.pop("boundary_field_ranks")) == set(_ALL_BOUNDARIES)
    assert vehicle_model == allbc_model


### ---------------------------------------------------------------------------
### Recipe-local transform unit tests
### ---------------------------------------------------------------------------


def test_set_domain_global_field_injects_at_domain_level() -> None:
    """SetDomainGlobalField writes DomainMesh.global_data (the level models read)."""
    domain = DomainMesh(
        interior=Mesh(points=torch.randn(10, 3)),
        boundaries=_boundary_meshes(("vehicle",)),
        global_data=_global_data(post_pipeline=False),
    )
    transform = SetDomainGlobalField(fields={"reference_length": 1.0})
    out = transform.apply_to_domain(domain)

    assert float(out.global_data["reference_length"]) == 1.0
    assert out.global_data["reference_length"].dtype == torch.float32
    ### Original untouched; existing keys preserved.
    assert "reference_length" not in domain.global_data.keys()
    assert torch.equal(out.global_data["U_inf"], domain.global_data["U_inf"])


def test_compute_freestream_direction_mesh_and_domain() -> None:
    """ComputeFreestreamDirection: unit U_inf_dir, U_inf left physical."""
    transform = ComputeFreestreamDirection(
        velocity_field="U_inf", output_field="U_inf_dir"
    )

    ### Mesh path (the stock surface pipeline).
    mesh = _boundary_meshes(("vehicle",))["vehicle"].with_data(
        global_data=_global_data(post_pipeline=False)
    )
    out_mesh = transform(mesh)
    direction = out_mesh.global_data["U_inf_dir"]
    assert torch.allclose(torch.linalg.vector_norm(direction), torch.tensor(1.0))
    assert torch.allclose(direction, torch.tensor([1.0, 0.0, 0.0]))
    ### The physical freestream is untouched (inference re-dimensionalization
    ### derives q_inf from it).
    assert torch.equal(out_mesh.global_data["U_inf"], mesh.global_data["U_inf"])

    ### Domain path (the all-BC pipeline): written at the DOMAIN level.
    domain = DomainMesh(
        interior=Mesh(points=torch.randn(10, 3)),
        boundaries=_boundary_meshes(("vehicle",)),
        global_data=_global_data(post_pipeline=False),
    )
    out_domain = transform.apply_to_domain(domain)
    assert torch.allclose(
        out_domain.global_data["U_inf_dir"], torch.tensor([1.0, 0.0, 0.0])
    )
    assert "U_inf_dir" not in domain.global_data.keys()  # original untouched

    ### Missing velocity field is a loud config error.
    with pytest.raises(KeyError, match="U_inf"):
        ComputeFreestreamDirection(velocity_field="U_inf")(
            mesh.with_data(global_data={})
        )


def test_boundary_mesh_to_domain_mesh_retargets_and_keeps_boundaries() -> None:
    """BoundaryMeshToDomainMesh: vehicle -> interior, all boundaries kept."""
    torch.manual_seed(1)
    boundaries = _boundary_meshes(_ALL_BOUNDARIES)
    n = boundaries["vehicle"].n_cells
    boundaries["vehicle"] = boundaries["vehicle"].with_data(
        cell_data={
            "pressure": torch.randn(n),
            "wss": torch.randn(n, 3),
            "normals": torch.randn(n, 3),
        },
    )
    domain = DomainMesh(
        ### A stand-in volume interior; the transform must discard it.
        interior=Mesh(points=torch.randn(50, 3)),
        boundaries=boundaries,
        global_data=_global_data(),
    )
    transform = BoundaryMeshToDomainMesh(
        cell_data_targets=["pressure", "wss"],
        interior_points="cell_centroids",
        boundary_name="vehicle",
    )
    out = transform.apply_to_domain(domain)

    ### Interior re-targeted onto the vehicle's cell centroids with targets.
    assert torch.equal(out.interior.points, boundaries["vehicle"].cell_centroids)
    assert torch.equal(
        out.interior.point_data["pressure"],
        boundaries["vehicle"].cell_data["pressure"],
    )
    assert torch.equal(
        out.interior.point_data["wss"], boundaries["vehicle"].cell_data["wss"]
    )
    ### Every typed boundary survives; targets stripped from the vehicle,
    ### non-target features kept; other boundaries pass through untouched.
    assert sorted(out.boundary_names) == sorted(_ALL_BOUNDARIES)
    vehicle_out = out.boundaries["vehicle"]
    assert "pressure" not in vehicle_out.cell_data.keys()
    assert "wss" not in vehicle_out.cell_data.keys()
    assert "normals" in vehicle_out.cell_data.keys()
    assert torch.equal(out.boundaries["inlet"].points, boundaries["inlet"].points)
    ### Domain-level global_data passes through.
    assert torch.equal(out.global_data["U_inf"], domain.global_data["U_inf"])

    ### A missing boundary is a loud config error.
    bad = BoundaryMeshToDomainMesh(cell_data_targets=["pressure"], boundary_name="wing")
    with pytest.raises(KeyError, match="wing"):
        bad.apply_to_domain(domain)


### ---------------------------------------------------------------------------
### Dataset YAML transform chains, end-to-end from raw field names
### ---------------------------------------------------------------------------


def _instantiate_pipeline_transforms(dataset: str, sampling_resolution: int) -> list:
    """Instantiate a dataset YAML's transforms exactly as build_dataset does.

    The reader is excluded (no data on disk in CI); target auto-injection
    into the ``*MeshToDomainMesh`` terminal mirrors production.
    """
    ds_yaml = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    ds_yaml = OmegaConf.merge(ds_yaml, {"sampling_resolution": sampling_resolution})
    target_names = list(OmegaConf.to_container(ds_yaml.targets, resolve=True))
    transforms = []
    for t_cfg in ds_yaml.pipeline.transforms:
        t_cfg = _maybe_inject_targets(t_cfg, target_names)
        transforms.append(hydra.utils.instantiate(t_cfg))
    return transforms


def _run_pipeline(sample, transforms):
    """Apply a transform chain with MeshDataset's DomainMesh dispatch."""
    for t in transforms:
        if isinstance(sample, DomainMesh):
            sample = t.apply_to_domain(sample)
        else:
            sample = t(sample)
    return sample


def _forward_through_template(
    model: str,
    dataset: str,
    domain: DomainMesh,
    *,
    query_chunk_size: int = _QUERY_CHUNK_SIZE,
    with_backward: bool = False,
) -> None:
    """Collate a post-pipeline domain and run the composed template's model."""
    train_cfg = _compose_train_cfg(model, dataset)
    collate = build_collate_fn(
        input_type="mesh",
        forward_kwargs_spec=OmegaConf.to_container(
            train_cfg.forward_kwargs, resolve=True
        ),
        target_config=_TARGETS,
    )
    batch = collate([(domain, {})])
    small_model_cfg = OmegaConf.merge(
        train_cfg.model, OmegaConf.create({"query_chunk_size": query_chunk_size})
    )
    model_inst = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    with torch.no_grad() if not with_backward else torch.enable_grad():
        output = model_inst(**batch["forward_kwargs"])
    pred_td = normalize_output_to_tensordict(output, _TARGETS, "mesh")
    for name in _TARGETS:
        assert pred_td[name].shape == batch["targets"][name].shape
        assert torch.isfinite(pred_td[name]).all(), f"{name} prediction not finite"

    if with_backward:
        lc = LossCalculator(
            target_config=_TARGETS, loss_type=train_cfg.training.loss_type
        )
        loss, _ = lc(pred_td.float(), batch["targets"].float())
        assert torch.isfinite(loss), f"loss not finite: {float(loss)}"
        loss.backward()
        for pname, p in model_inst.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {pname}"


def _raw_vehicle_cell_data(n_cells: int) -> dict[str, torch.Tensor]:
    """Raw (pre-pipeline) DrivAerML surface field names on the vehicle."""
    return {
        "pMeanTrim": torch.randn(n_cells) * 50.0,
        "wallShearStressMeanTrim": torch.randn(n_cells, 3) * 0.003,
    }


def test_surface_yaml_chain_injects_gauge_and_feeds_model() -> None:
    """drivaer_ml_surface.yaml transforms: raw vehicle Mesh -> DomainMesh -> model.

    Pins the documented SetGlobalField addition: the produced DomainMesh
    carries the constant ``reference_length`` gauge at the domain level.
    """
    torch.manual_seed(2)
    transforms = _instantiate_pipeline_transforms(
        "drivaer_ml_surface", sampling_resolution=16
    )

    ### What MeshReaderWithGlobalData produces: the vehicle boundary Mesh
    ### with the parent DomainMesh's global_data merged in (plus the
    ### boundary-local TimeValue that DropMeshFields removes).
    vehicle = _boundary_meshes(("vehicle",))["vehicle"]
    raw = vehicle.with_data(
        cell_data=_raw_vehicle_cell_data(vehicle.n_cells),
        global_data={
            **_global_data(post_pipeline=False),
            "TimeValue": torch.tensor(1.0),
        },
    )
    domain = _run_pipeline(raw, transforms)

    assert isinstance(domain, DomainMesh)
    assert domain.boundary_names == ["vehicle"]
    assert float(domain.global_data["reference_length"]) == _REFERENCE_LENGTH
    ### The derived unit drive; U_inf itself stays physical.
    assert torch.allclose(
        domain.global_data["U_inf_dir"], torch.tensor([1.0, 0.0, 0.0])
    )
    assert torch.allclose(
        torch.linalg.vector_norm(domain.global_data["U_inf"]),
        torch.tensor(38.889),
    )
    assert "TimeValue" not in domain.global_data.keys()
    assert set(_TARGETS) <= set(domain.interior.point_data.keys())
    assert domain.interior.n_points == 16  # SubsampleMesh cap

    _forward_through_template("mesh_transformer_surface", "drivaer_ml_surface", domain)


def test_surface_pipeline_pins_declared_trace_alignment() -> None:
    """The pipeline + collate deliver EXACTLY the trace arm's declared map.

    ``trace_of: vehicle`` declares that query ``i`` IS vehicle cell ``i``
    (its centroid, in cell order).  The model can validate the COUNT but
    not the ORDER, so the order is this pipeline's pinned responsibility:
    the terminal ``MeshToDomainMesh(interior_points: cell_centroids)``
    builds ``interior.points`` from the SAME subsampled vehicle mesh whose
    stripped copy becomes ``boundaries.vehicle`` (SubsampleMesh runs
    earlier, keeping cells and centroids paired), and the mesh-input
    collate hands the DomainMesh through unmodified.  Pinned BITWISE here,
    end to end, plus the loud count-mismatch rejection.
    """
    torch.manual_seed(6)
    transforms = _instantiate_pipeline_transforms(
        "drivaer_ml_surface", sampling_resolution=16
    )
    vehicle = _boundary_meshes(("vehicle",))["vehicle"]
    raw = vehicle.with_data(
        cell_data=_raw_vehicle_cell_data(vehicle.n_cells),
        global_data={
            **_global_data(post_pipeline=False),
            "TimeValue": torch.tensor(1.0),
        },
    )
    domain = _run_pipeline(raw, transforms)

    ### The declared identity map, bitwise: same count, same order, same
    ### values -- interior.points IS the vehicle's cell centroids.
    assert torch.equal(
        domain.interior.points, domain.boundaries["vehicle"].cell_centroids
    )
    assert domain.interior.n_points == domain.boundaries["vehicle"].n_cells

    ### The collate preserves the alignment trivially: the DomainMesh
    ### object itself is handed to the model (no reindexing, no copy).
    train_cfg = _compose_train_cfg(
        "mesh_transformer_surface_trace", "drivaer_ml_surface"
    )
    collate = build_collate_fn(
        input_type="mesh",
        forward_kwargs_spec=OmegaConf.to_container(
            train_cfg.forward_kwargs, resolve=True
        ),
        target_config=_TARGETS,
    )
    batch = collate([(domain, {})])
    assert batch["forward_kwargs"]["domain"] is domain

    ### The trace template consumes the aligned domain: forward + backward.
    _forward_through_template(
        "mesh_transformer_surface_trace",
        "drivaer_ml_surface",
        domain,
        with_backward=True,
    )

    ### A broken alignment is a loud declaration error, not silent skew.
    misaligned = DomainMesh(
        interior=Mesh(
            points=domain.interior.points[:-1].clone(),
            point_data=domain.interior.point_data[:-1],
        ),
        boundaries=dict(domain.boundaries.items()),
        global_data=domain.global_data,
    )
    small_model_cfg = OmegaConf.merge(
        train_cfg.model, OmegaConf.create({"query_chunk_size": _QUERY_CHUNK_SIZE})
    )
    model_inst = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    with pytest.raises(ValueError, match="cell centroids"):
        with torch.no_grad():
            model_inst(misaligned)


def test_allbc_yaml_chain_real_shape_stats_forward_backward() -> None:
    """drivaer_ml_surface_allbc.yaml chain on REAL shape statistics -> model.

    The acceptance path for the H-ground arm AND the regression test for
    the cluster NaN (loss non-finite from step 1, 2026-07): the synthetic
    raw DomainMesh reproduces the run_1 probe's shape statistics -- tunnel
    proportions (x in [-40, 80] m), auxiliary boundaries at their real cell
    counts and O(1-7) m^2 areas, a vehicle of tiny scattered triangles with
    areas spanning 2.3e-9..6.6e-5 m^2, physical U_inf = 38.889 m/s -- and
    the whole chain + template forward + backward must stay finite. Under
    the original config (raw-U_inf drive, vehicle-scale gauge 1.0) this
    exact geometry overflows float32 to NaN at initialization.
    """
    torch.manual_seed(3)
    ### Above the 722-cell auxiliary boundaries (they must pass through
    ### complete, as at the pilot resolutions) and below the vehicle and
    ### slip counts (they must be randomly capped).
    resolution = 1000
    transforms = _instantiate_pipeline_transforms(
        "drivaer_ml_surface_allbc", sampling_resolution=resolution
    )

    ### What DomainMeshReader produces (post reader caps): a volume-interior
    ### point-cloud sliver + all five typed boundaries + case-level
    ### global_data with raw field names. The vehicle arrives above the
    ### random-subsample cap to exercise SubsampleMesh; the small auxiliary
    ### boundaries must pass through complete.
    boundaries = _tunnel_boundaries(n_vehicle_cells=1500)
    boundaries["vehicle"] = boundaries["vehicle"].with_data(
        cell_data=_raw_vehicle_cell_data(boundaries["vehicle"].n_cells)
    )
    raw = DomainMesh(
        interior=Mesh(
            points=torch.rand(2000, 3) * torch.tensor([120.0, 30.2, 40.0])
            + torch.tensor([-40.0, -15.1, 0.0]),
            point_data={"UMeanTrim": torch.randn(2000, 3) * 30.0},
        ),
        boundaries=boundaries,
        global_data=_global_data(post_pipeline=False),
    )
    domain = _run_pipeline(raw, transforms)

    assert isinstance(domain, DomainMesh)
    assert sorted(domain.boundary_names) == sorted(_ALL_BOUNDARIES)
    assert float(domain.global_data["reference_length"]) == _REFERENCE_LENGTH
    assert torch.allclose(
        torch.linalg.vector_norm(domain.global_data["U_inf_dir"]),
        torch.tensor(1.0),
    )
    assert set(_TARGETS) <= set(domain.interior.point_data.keys())
    ### SubsampleMesh semantics: only meshes exceeding the cap shrink; the
    ### 722-cell auxiliary boundaries pass through complete.
    assert domain.boundaries["vehicle"].n_cells == resolution
    assert domain.interior.n_points == resolution
    assert domain.boundaries["outlet"].n_cells == 722
    assert domain.boundaries["no_slip"].n_cells == 722
    assert domain.boundaries["slip"].n_cells == resolution  # 2,888 -> capped
    ### The surface task: queries sit at the vehicle's cell centroids (the
    ### re-target ran before centering/nondim, which are affine and applied
    ### domain-consistently).
    assert torch.allclose(
        domain.interior.points,
        domain.boundaries["vehicle"].cell_centroids,
        atol=1e-5,
    )
    ### The regression must keep its teeth: sources genuinely span the
    ### tunnel (~15 L_ref units, ~2 gauge units), the far-outside-the-car
    ### regime that overflowed before the tunnel-scale gauge + unit-drive
    ### fix (at gauge 1.0 these coordinates sit at |x| ~ 15).
    span = max(
        float(domain.boundaries[b].points.abs().max()) for b in domain.boundary_names
    )
    assert span > 10.0  # nondim (L_ref) units

    _forward_through_template(
        "mesh_transformer_surface_allbc",
        "drivaer_ml_surface_allbc",
        domain,
        query_chunk_size=256,
        with_backward=True,
    )


### ---------------------------------------------------------------------------
### Reader-level regression: the 378-cell vehicle starvation (2026-07)
### ---------------------------------------------------------------------------


def test_topology_aware_reader_feeds_full_vehicle(tmp_path: Path) -> None:
    """The allbc READER + chain feed the vehicle the full sampling cap.

    Regression for the real-data probe finding: with the stock
    ``DomainMeshReader`` the allbc arm's vehicle boundary collapsed to 378
    of 10,000 cells, because the reader applies BOTH caps to EVERY mesh --
    the cell cap compacts the (shuffled-cell-order) vehicle to ~3 unique
    points per cell, then the point cap keeps only cells with all three
    vertices inside a contiguous point block, ~(1/3)^3 of them.  This test
    exercises the ACTUAL reader on a saved synthetic ``.pdmsh`` with real
    DrivAerML shape statistics (the transform-only chain tests structurally
    could not see a reader-level bug) and pins the post-pipeline
    per-boundary cell counts: vehicle == sampling cap, auxiliaries == full.
    """
    torch.manual_seed(4)
    resolution = 3000

    boundaries = _tunnel_boundaries(n_vehicle_cells=6000)
    boundaries["vehicle"] = boundaries["vehicle"].with_data(
        cell_data=_raw_vehicle_cell_data(boundaries["vehicle"].n_cells)
    )
    raw = DomainMesh(
        ### The interior is a large POINT CLOUD (no cells), like the curated
        ### 165.8M-point volume interior: only the point cap can bound it.
        interior=Mesh(
            points=torch.rand(30_000, 3) * torch.tensor([120.0, 30.2, 40.0])
            + torch.tensor([-40.0, -15.1, 0.0]),
            point_data={"UMeanTrim": torch.randn(30_000, 3) * 30.0},
        ),
        boundaries=boundaries,
        global_data=_global_data(post_pipeline=False),
    )
    case_dir = tmp_path / "run_1"
    case_dir.mkdir()
    raw.save(str(case_dir / "case.pdmsh"))

    reader_kwargs = dict(
        path=tmp_path,
        pattern="run_*/*.pdmsh",  # the production glob
        subsample_n_cells=resolution,
        subsample_n_points=resolution,
    )

    ### The recipe reader: cell cap on cell-carrying meshes, point cap on
    ### point clouds only.
    read, _meta = TopologyAwareDomainMeshReader(**reader_kwargs)[0]
    assert read.boundaries["vehicle"].n_cells == resolution
    assert read.interior.n_points == resolution
    assert read.boundaries["outlet"].n_cells == 722
    assert read.boundaries["inlet"].n_cells == 722
    assert read.boundaries["no_slip"].n_cells == 722
    assert read.boundaries["slip"].n_cells == 2888
    ### Cell fields ride along with the subsampled cells.
    assert read.boundaries["vehicle"].cell_data["pMeanTrim"].shape == (resolution,)

    ### Contrast: the stock reader's uniform caps starve the vehicle (the
    ### 378-class failure). Pinning the contrast guards against silently
    ### reverting the yaml to ${dp:DomainMeshReader}.
    stock, _meta = DomainMeshReader(**reader_kwargs)[0]
    assert stock.boundaries["vehicle"].n_cells < resolution

    ### End-to-end: reader output through the allbc transform chain gives
    ### the information-matched source set (vehicle == cap, aux == full).
    transforms = _instantiate_pipeline_transforms(
        "drivaer_ml_surface_allbc", sampling_resolution=resolution
    )
    domain = _run_pipeline(read, transforms)
    counts = {name: domain.boundaries[name].n_cells for name in domain.boundary_names}
    assert counts == {
        "vehicle": resolution,
        "inlet": 722,
        "no_slip": 722,
        "outlet": 722,
        "slip": 2888,
    }
    assert domain.interior.n_points == resolution
    assert torch.allclose(
        domain.interior.points,
        domain.boundaries["vehicle"].cell_centroids,
        atol=1e-5,
    )


def test_topology_aware_reader_seeded_subsampling_is_reproducible(
    tmp_path: Path,
) -> None:
    """The recipe reader's seeded path is executable and deterministic."""
    vehicle = _patch(
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        8,
        7,
    )
    vehicle = vehicle.with_data(cell_data={"cell_id": torch.arange(vehicle.n_cells)})
    raw = DomainMesh(
        interior=Mesh(points=torch.arange(90, dtype=torch.float32).reshape(30, 3)),
        boundaries={"vehicle": vehicle},
    )
    case_dir = tmp_path / "run_1"
    case_dir.mkdir()
    raw.save(str(case_dir / "case.pdmsh"))

    reader = TopologyAwareDomainMeshReader(
        path=tmp_path,
        pattern="run_*/*.pdmsh",
        subsample_n_cells=9,
        subsample_n_points=8,
    )
    reader.set_generator(torch.Generator().manual_seed(1234))
    first, _ = reader[0]
    reader.set_generator(torch.Generator().manual_seed(1234))
    second, _ = reader[0]

    assert torch.equal(first.interior.points, second.interior.points)
    assert torch.equal(
        first.boundaries["vehicle"].cell_data["cell_id"],
        second.boundaries["vehicle"].cell_data["cell_id"],
    )


### ---------------------------------------------------------------------------
### Boundary-contribution probe (tools/probe_boundary_contributions.py)
### ---------------------------------------------------------------------------


def test_probe_boundary_contributions_partition_and_area_domination() -> None:
    """The moment-partition probe is exact and shows tunnel domination at init.

    On the real-stats synthetic tunnel with an UNTRAINED all-BC model:

    - the per-boundary moment parts sum to the full moments (linearity of
      the quadrature; the residual is float roundoff),
    - moment fractions sum to 1 and cover all five boundaries,
    - the tunnel dominates the moment mass at init roughly like its share
      of the source measure (the vehicle's cells carry ~1e-11 of the total
      area here), which is the initialization-imbalance premise the
      cluster probe quantifies at trained weights.
    """
    torch.manual_seed(5)
    boundaries = _tunnel_boundaries(n_vehicle_cells=400)
    n = boundaries["vehicle"].n_cells
    domain = DomainMesh(
        interior=Mesh(
            points=boundaries["vehicle"].cell_centroids.clone(),
            point_data={
                "pressure": torch.randn(n),
                "wss": torch.randn(n, 3) * 0.1,
            },
        ),
        boundaries=boundaries,
        global_data=_global_data(),
    )
    cfg = _compose_train_cfg(
        "mesh_transformer_surface_allbc", "drivaer_ml_surface_allbc"
    )
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    records = probe.capture_boundary_contributions(model, domain)

    ### One record per encoder attention layer (operator + drive blocks).
    assert len(records) >= 2
    for record in records:
        assert set(record.boundaries) == set(_ALL_BOUNDARIES)
        fractions = [c.moment_fraction for c in record.boundaries.values()]
        assert abs(sum(fractions) - 1.0) < 1e-6
        assert record.partition_residual < 1e-3, record.layer
        assert record.total_moment_norm > 0.0
        vehicle = record.boundaries["vehicle"]
        assert vehicle.n_cells == n
        ### Tunnel domination at init, tracking the measure imbalance.
        assert vehicle.area_fraction < 1e-5
        assert vehicle.moment_fraction < 1e-2
        assert probe.tunnel_vehicle_ratio(record) > 1e2
        for contribution in record.boundaries.values():
            assert contribution.mean_value_norm > 0.0

    ### Instrumentation restores the modules: a second capture reproduces
    ### the same layer set, and the un-instrumented encode still runs.
    records_again = probe.capture_boundary_contributions(model, domain)
    assert [r.layer for r in records_again] == [r.layer for r in records]
    with torch.no_grad():
        model.encode(domain)

    ### The report renders with and without a trained-side column.
    text = probe.format_report(records, records_again)
    assert "tunnel/vehicle moment ratio" in text
    assert "vehicle" in text

    ### The probe also handles POOLED models (segmented build_moments calls
    ### with per-boundary gains): the partition identity must still hold,
    ### and suppressing a boundary's gain must show up in its contribution.
    cfg_pooled = _compose_train_cfg(
        "mesh_transformer_surface_allbc_pooled", "drivaer_ml_surface_allbc"
    )
    pooled = hydra.utils.instantiate(cfg_pooled.model, _convert_="partial")
    pooled_records = probe.capture_boundary_contributions(pooled, domain)
    assert len(pooled_records) >= 2
    for record in pooled_records:
        assert record.partition_residual < 1e-3, record.layer
        fractions = [c.moment_fraction for c in record.boundaries.values()]
        assert abs(sum(fractions) - 1.0) < 1e-6
    with torch.no_grad():
        ### Suppress the slip walls (the largest measure) by e^-6 in every
        ### block's gains; slip's moment share must drop in every layer.
        slip_index = list(pooled.boundary_names).index("slip")
        for name, parameter in pooled.named_parameters():
            if name.endswith("moment_segment_log_gain"):
                parameter[slip_index] = -6.0
    suppressed_records = probe.capture_boundary_contributions(pooled, domain)
    for before, after in zip(pooled_records, suppressed_records):
        assert (
            after.boundaries["slip"].moment_fraction
            < before.boundaries["slip"].moment_fraction
        ), before.layer
