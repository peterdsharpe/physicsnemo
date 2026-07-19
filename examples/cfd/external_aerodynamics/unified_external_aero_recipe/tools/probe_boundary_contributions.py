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

"""Per-boundary attention-moment contribution probe for the MeshTransformer.

DIAGNOSTIC QUESTION (DrivAerML H-ground negative, 2026-07): the all-BC arm
underfits relative to the vehicle-only arm.  The model's encoder attention
moments are quadrature-weighted with RAW cell areas
(``MeshAttention.build_moments``: :math:`M=\\sum_n \\omega_n\\,k_n\\otimes
v_n`), and the tunnel panels carry ~1e6x the area of vehicle cells -- so
"ignore the tunnel" requires the learned, BC-one-hot-conditioned key/value
features to suppress a ~1e6 measure ratio starting from an O(1) init.
Hypothesis: optimization pathology, partially won within the budget.

This probe partitions every encoder attention layer's moment integrals by
source-cell boundary membership (the moments are linear over source cells,
so per-boundary slices sum exactly to the full moments) and reports, per
boundary, at INIT and at TRAINED weights:

- ``n_cells`` and the boundary's fraction of the total source measure;
- the fraction of the moment-tensor norm contributed by the boundary;
- the mean per-cell norm of the projected (lifted) VALUE features -- the
  learned suppression mechanism itself.

VERDICT GUIDE: if the trained tunnel-vs-vehicle contribution ratio dropped
from ~area-ratio (>=1e2 at init on subsampled cases; up to ~1e6 at full
measure ratio) toward 1e3-1e4 but not to <=1e0, the optimization story is
confirmed; if it reached vehicle dominance, the coarse-quadrature story
takes over.

Instrumentation is hooks-only: each ``MeshAttention.build_moments`` is
wrapped per-instance for the duration of one ``encode()`` and restored
afterwards; no core model code is touched.  Pooled models
(``per_boundary_moment_pool``) are handled: their per-boundary gains are
applied to the partitioned parts, so the report shows what the pooled
model actually integrates.  The domain is built through the recipe's own
``TopologyAwareDomainMeshReader`` (the production I/O path) plus the
all-BC dataset YAML's transform chain exactly as validation does
(augmentations excluded, as in the recipe's val split).  The INIT model is
a fresh seeded construction of the composed template -- representative
init statistics, not bitwise the training run's init.

Usage::

    python tools/probe_boundary_contributions.py \\
        --checkpoint runs/<run_id>/checkpoints \\
        --case /path/to/run_123/domain_123.pdmsh \\
        [--sampling-resolution 10000] [--epoch N] [--seed 42] \\
        [--model mesh_transformer_surface_allbc] \\
        [--dataset drivaer_ml_surface_allbc] [--device cuda]

``--checkpoint`` accepts either the recipe's checkpoint DIRECTORY
(``runs/<run_id>/checkpoints``; loads the latest or ``--epoch``) or a
single ``.mdlus`` / ``.pt`` file.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
_SRC = _RECIPE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

### Recipe-local side-effect imports (same set as src/datasets.py).
import hydra  # noqa: E402
import nondim  # noqa: E402, F401
import sdf  # noqa: E402, F401
from datasets import _maybe_inject_targets  # noqa: E402
from domain_transforms import TopologyAwareDomainMeshReader  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tabulate import tabulate  # noqa: E402

import physicsnemo.datapipes  # noqa: E402, F401  (registers ${dp:...})
from physicsnemo.experimental.nn.mesh_attention.attention import (  # noqa: E402
    MeshAttention,
)
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402
from physicsnemo.utils.checkpoint import load_model_weights  # noqa: E402

### ---------------------------------------------------------------------------
### Domain construction: reproduce the training pipeline for one case
### ---------------------------------------------------------------------------


def compose_train_cfg(model: str, dataset: str) -> DictConfig:
    """Compose ``conf/train.yaml`` with the given model + dataset selection."""
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"),
        version_base=None,
    ):
        return compose(
            config_name="train",
            overrides=[f"model={model}", f"dataset={dataset}"],
        )


def instantiate_pipeline_transforms(dataset: str, sampling_resolution: int) -> list:
    """Instantiate a dataset YAML's transforms exactly as ``build_dataset`` does.

    The reader is excluded (the caller reproduces its subsampling); target
    auto-injection into the ``*MeshToDomainMesh`` transform mirrors
    production.
    """
    ds_yaml = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    ds_yaml = OmegaConf.merge(ds_yaml, {"sampling_resolution": sampling_resolution})
    target_names = list(OmegaConf.to_container(ds_yaml.targets, resolve=True))
    transforms = []
    for t_cfg in ds_yaml.pipeline.transforms:
        t_cfg = _maybe_inject_targets(t_cfg, target_names)
        transforms.append(hydra.utils.instantiate(t_cfg))
    return transforms


def build_domain_from_case(
    case_path: str | Path,
    dataset: str,
    sampling_resolution: int,
    *,
    generator: torch.Generator | None = None,
) -> DomainMesh:
    """Load one ``.pdmsh`` case and run it through the dataset's chain.

    Reads through the recipe's own :class:`TopologyAwareDomainMeshReader`
    (the exact production I/O path, including its topology-aware
    contiguous-block subsampling) followed by the YAML transform chain with
    ``MeshDataset``'s DomainMesh dispatch.  Augmentations are excluded (the
    recipe's validation path).
    """
    case = Path(case_path)
    reader = TopologyAwareDomainMeshReader(
        path=case.parent,
        pattern=case.name,
        subsample_n_cells=sampling_resolution,
        subsample_n_points=sampling_resolution,
    )
    if generator is not None:
        reader.set_generator(generator)
    dm, _metadata = reader[0]
    for transform in instantiate_pipeline_transforms(dataset, sampling_resolution):
        if isinstance(dm, DomainMesh):
            dm = transform.apply_to_domain(dm)
        else:  # pragma: no cover -- the all-BC chain is DomainMesh throughout
            dm = transform(dm)
    return dm


### ---------------------------------------------------------------------------
### Capture: partition every attention layer's moments by boundary
### ---------------------------------------------------------------------------


@dataclass
class BoundaryContribution:
    """One boundary's share of one attention layer's moment integrals."""

    n_cells: int
    area: float
    area_fraction: float
    moment_norm: float
    moment_fraction: float
    mean_value_norm: float


@dataclass
class LayerRecord:
    """Per-boundary partition of one ``build_moments`` call."""

    layer: str
    total_moment_norm: float
    ### Relative norm of (sum of per-boundary moments) - (full moments):
    ### the partition-correctness certificate (float-roundoff-sized).
    partition_residual: float
    boundaries: dict[str, BoundaryContribution] = field(default_factory=dict)


def _moments_norm_sq(moments) -> torch.Tensor:
    return sum(
        tensor.square().sum()
        for tensor in (
            moments.scalar_key_scalar_value,
            moments.vector_key_scalar_value,
            moments.scalar_key_vector_value,
            moments.vector_key_vector_value,
        )
    )


def boundary_cell_ranges(model, domain: DomainMesh) -> dict[str, slice]:
    """Source-cell index range per boundary in the model's merge order.

    ``encode()`` merges ``domain.boundaries[name]`` in ``model.boundary_names``
    order (sorted), so cumulative cell counts give each boundary's slice of
    the merged source mesh.
    """
    ranges: dict[str, slice] = {}
    offset = 0
    for name in model.boundary_names:
        count = domain.boundaries[name].n_cells
        ranges[name] = slice(offset, offset + count)
        offset += count
    return ranges


def capture_boundary_contributions(
    model,
    domain: DomainMesh,
) -> list[LayerRecord]:
    """Run ``model.encode(domain)`` with per-boundary moment partitioning.

    Every :class:`MeshAttention` submodule's ``build_moments`` is wrapped
    (instance attribute; restored in a ``finally``) to additionally compute
    the moments of each boundary's source-cell slice.  The moments are
    linear over source cells, so the per-boundary parts sum to the full
    moments up to accumulation roundoff (reported as
    ``partition_residual``).

    Pooled models (``per_boundary_moment_pool=true``) pass ``segments`` +
    ``segment_log_gain`` into ``build_moments``; each boundary's part is
    then scaled by its ``exp(log_gain)`` so the reported contributions are
    exactly what the pooled model integrates, and the partition identity
    still holds.
    """
    ranges = boundary_cell_ranges(model, domain)
    total_cells = sum(r.stop - r.start for r in ranges.values())
    records: list[LayerRecord] = []
    wrapped_modules: list = []

    def make_wrapper(layer_name: str, module: MeshAttention, original):
        def wrapped(
            source_mesh: Mesh,
            key_state,
            value_state,
            segments=None,
            segment_log_gain=None,
            **build_kwargs,
        ):
            ### Signature-transparent passthrough (e.g. the balanced-pool
            ### flag): the probe decomposes the same call it forwards.
            full = original(
                source_mesh,
                key_state,
                value_state,
                segments=segments,
                segment_log_gain=segment_log_gain,
                **build_kwargs,
            )
            ### Partition only calls over the merged source (defensive; every
            ### encode-time call matches).
            if key_state.n_entities != total_cells:
                return full
            record = LayerRecord(
                layer=layer_name,
                total_moment_norm=float(_moments_norm_sq(full).sqrt()),
                partition_residual=0.0,
            )
            areas = source_mesh.cell_areas
            total_area = float(areas.sum())
            part_sum = None
            norms: dict[str, float] = {}
            for index, (bname, rng) in enumerate(ranges.items()):
                part = original(
                    source_mesh.slice_cells(rng),
                    key_state.slice(rng),
                    value_state.slice(rng),
                )
                if segment_log_gain is not None:
                    ### The pooled model's segments follow model.boundary_names
                    ### order -- the same order as `ranges` -- so gain row
                    ### `index` belongs to this boundary.
                    gain = torch.exp(segment_log_gain[index]).to(
                        part.scalar_key_scalar_value.dtype
                    )
                    part = type(part)(
                        part.scalar_key_scalar_value * gain[:, None, None],
                        part.vector_key_scalar_value * gain[:, None, None, None],
                        part.scalar_key_vector_value * gain[:, None, None, None],
                        part.vector_key_vector_value * gain[:, None, None, None, None],
                    )
                norms[bname] = float(_moments_norm_sq(part).sqrt())
                if part_sum is None:
                    part_sum = list(
                        (
                            part.scalar_key_scalar_value,
                            part.vector_key_scalar_value,
                            part.scalar_key_vector_value,
                            part.vector_key_vector_value,
                        )
                    )
                else:
                    part_sum[0] = part_sum[0] + part.scalar_key_scalar_value
                    part_sum[1] = part_sum[1] + part.vector_key_scalar_value
                    part_sum[2] = part_sum[2] + part.scalar_key_vector_value
                    part_sum[3] = part_sum[3] + part.vector_key_vector_value
                ### The learned suppression mechanism: per-cell norm of the
                ### projected value features on this boundary's cells.
                values = module.project_values(value_state.slice(rng))
                per_cell = (
                    values.scalars.square().sum(dim=(1, 2))
                    + values.vectors.square().sum(dim=(1, 2, 3))
                ).sqrt()
                boundary_area = float(areas[rng].sum())
                record.boundaries[bname] = BoundaryContribution(
                    n_cells=rng.stop - rng.start,
                    area=boundary_area,
                    area_fraction=boundary_area / total_area,
                    moment_norm=norms[bname],
                    moment_fraction=0.0,  # filled below
                    mean_value_norm=float(per_cell.mean()),
                )
            full_tensors = (
                full.scalar_key_scalar_value,
                full.vector_key_scalar_value,
                full.scalar_key_vector_value,
                full.vector_key_vector_value,
            )
            residual_sq = sum(
                (ps - ft).square().sum() for ps, ft in zip(part_sum, full_tensors)
            )
            record.partition_residual = float(
                residual_sq.sqrt() / max(record.total_moment_norm, 1e-30)
            )
            norm_sum = max(sum(norms.values()), 1e-30)
            for bname, contribution in record.boundaries.items():
                contribution.moment_fraction = norms[bname] / norm_sum
            records.append(record)
            return full

        return wrapped

    for name, module in model.named_modules():
        if isinstance(module, MeshAttention):
            original = module.build_moments
            module.build_moments = make_wrapper(name, module, original)
            wrapped_modules.append(module)
    try:
        with torch.no_grad():
            model.encode(domain)
    finally:
        for module in wrapped_modules:
            del module.build_moments  # restore the class method
    return records


### ---------------------------------------------------------------------------
### Reporting
### ---------------------------------------------------------------------------


def tunnel_vehicle_ratio(record: LayerRecord, vehicle: str = "vehicle") -> float:
    """Non-vehicle (tunnel) over vehicle moment-contribution ratio."""
    tunnel = sum(
        contribution.moment_norm
        for name, contribution in record.boundaries.items()
        if name != vehicle
    )
    return tunnel / max(record.boundaries[vehicle].moment_norm, 1e-30)


def format_report(
    init_records: list[LayerRecord],
    trained_records: list[LayerRecord] | None,
) -> str:
    """Render the per-layer, per-boundary comparison tables."""
    lines: list[str] = []
    trained_by_layer = (
        {record.layer: record for record in trained_records}
        if trained_records is not None
        else {}
    )
    for init_record in init_records:
        trained_record = trained_by_layer.get(init_record.layer)
        rows = []
        for bname, ic in init_record.boundaries.items():
            tc = trained_record.boundaries[bname] if trained_record else None
            rows.append(
                [
                    bname,
                    ic.n_cells,
                    f"{ic.area_fraction:.3e}",
                    f"{ic.moment_fraction:.3e}",
                    f"{tc.moment_fraction:.3e}" if tc else "--",
                    f"{ic.mean_value_norm:.3e}",
                    f"{tc.mean_value_norm:.3e}" if tc else "--",
                ]
            )
        lines.append(f"\n=== {init_record.layer} ===")
        lines.append(
            f"partition residual: init {init_record.partition_residual:.2e}"
            + (
                f", trained {trained_record.partition_residual:.2e}"
                if trained_record
                else ""
            )
        )
        lines.append(
            tabulate(
                rows,
                headers=[
                    "boundary",
                    "n_cells",
                    "area_frac",
                    "moment_frac_init",
                    "moment_frac_trained",
                    "mean|value|_init",
                    "mean|value|_trained",
                ],
                tablefmt="simple",
            )
        )
        headline = (
            f"tunnel/vehicle moment ratio: init {tunnel_vehicle_ratio(init_record):.3e}"
        )
        if trained_record:
            headline += f"  ->  trained {tunnel_vehicle_ratio(trained_record):.3e}"
        lines.append(headline)
    return "\n".join(lines)


### ---------------------------------------------------------------------------
### Entry point
### ---------------------------------------------------------------------------


def _load_trained(model, checkpoint: str, epoch: int | None, device: str) -> None:
    path = Path(checkpoint)
    if path.is_dir():
        loaded = load_checkpoint(
            path=str(path), models=model, epoch=epoch, device=device
        )
        print(f"loaded checkpoint epoch {loaded} from directory {path}")
    else:
        load_model_weights(model, str(path), device=device)
        print(f"loaded weights file {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory (runs/<run_id>/checkpoints) or .mdlus/.pt file",
    )
    parser.add_argument("--case", required=True, help="path to one .pdmsh case")
    parser.add_argument("--sampling-resolution", type=int, default=10_000)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="mesh_transformer_surface_allbc")
    parser.add_argument("--dataset", default="drivaer_ml_surface_allbc")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cfg = compose_train_cfg(args.model, args.dataset)
    generator = torch.Generator().manual_seed(args.seed)
    domain = build_domain_from_case(
        args.case, args.dataset, args.sampling_resolution, generator=generator
    ).to(args.device)
    counts = {name: domain.boundaries[name].n_cells for name in domain.boundary_names}
    print(f"case: {args.case}")
    print(f"post-pipeline boundary cells: {counts}")

    torch.manual_seed(args.seed)
    init_model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(args.device)
    init_model.eval()
    init_records = capture_boundary_contributions(init_model, domain)

    trained_model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    _load_trained(trained_model, args.checkpoint, args.epoch, args.device)
    trained_model = trained_model.to(args.device)
    trained_model.eval()
    trained_records = capture_boundary_contributions(trained_model, domain)

    print(format_report(init_records, trained_records))


if __name__ == "__main__":
    main()
