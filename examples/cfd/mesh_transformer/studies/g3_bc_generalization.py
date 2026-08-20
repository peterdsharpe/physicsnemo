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

"""G3: out-of-distribution generalization over boundary-condition factors.

Preregistration: results/g3_2d_bc_generalization_preregistration_2026-08-20.json

Three architectures are trained identically on the exact 2D conformal-Laplace
generator (drive modes 1-4, boundary_rms 1.0, deformation kappa <= 0.3,
randomized similarity pose) and evaluated on one-factor-moved test sets:

- T0: fresh in-range cases (control)
- T1: drive modes 5-8 only (unseen boundary frequencies)
- T2a/T2b: boundary_rms 2.0 / 4.0 (amplitude extrapolation; linearity anchor)
- T3: deformation kappa in (0.3, 0.6] (unseen geometry strength)

Arms:

- ``mt2_bscalar``: MeshTransformer2 with boundary scalars, 2D embedded in the
  z = 0 plane; boundary panels and interior queries form one token set and
  the loss reads the interior tokens (adapter below).
- ``softslice_2d``: the suite's in-tree Transolver soft-slice baseline at
  matched parameter count (its native DomainMesh adapter).
- ``mt1_linear``: the native MeshTransformer at the suite's reference
  capacity in linear field mode -- the drive-linearity exactness anchor.

Usage::

    python studies/g3_bc_generalization.py            # full preregistered run
    python studies/g3_bc_generalization.py --smoke    # 2 epochs / 8 cases
    python studies/g3_bc_generalization.py --arms mt2_bscalar --seeds 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from torch import nn

from conformal_laplace import (
    ConformalLaplaceSample,
    build_domain_sample,
    sample_drive,
    sample_geometry,
    sample_similarity,
)
from models import parameter_count
from train import make_model, relative_mse
from transolver_intree import build_transolver_intree

from physicsnemo.experimental.nn import MeshTransformer2
from physicsnemo.mesh import DomainMesh, Mesh

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "g3_2d_2026-08-20"

# Deterministic, split-disjoint seed streams (same scheme as train._case_seed).
SPLIT_BASE_SEEDS = {
    "train": 10_000_000,
    "T0": 20_000_000,
    "T1": 30_000_000,
    "T2a": 40_000_000,
    "T2b": 50_000_000,
    "T3": 60_000_000,
}
STREAM_GEOMETRY, STREAM_DRIVE, STREAM_QUERY, STREAM_SIMILARITY = 0, 1, 2, 3

GEOMETRY_MODES = (2, 3)  # the suite's training geometry family
TRAIN_DEFORMATION = (0.05, 0.3)
T3_DEFORMATION = (0.3, 0.6)
TRAIN_DRIVE_MODES = (1, 2, 3, 4)
T1_DRIVE_MODES = (5, 6, 7, 8)
DRIVE_REGULARITY = 2.0  # generator default: k^-2 trace regularity (prereg)

MT2_CONFIG = dict(
    out_scalars=1,
    out_vectors=0,
    hidden=56,
    n_layers=2,
    n_slices=32,
    mlp_ratio=4,
    reference_length=1.0,
    n_boundary_scalars=2,  # [dirichlet value, is_boundary flag]
    query_independent=False,
)


def _case_seed(base: int, case_index: int, stream: int) -> int:
    return base + 1_000_003 * case_index + 104_729 * stream


def make_g3_case(
    split: str,
    case_index: int,
    *,
    n_boundary: int,
    n_query: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ConformalLaplaceSample:
    """Build one deterministic case of the named split (one factor moved)."""

    base = SPLIT_BASE_SEEDS[split]
    deformation = TRAIN_DEFORMATION
    drive_modes = TRAIN_DRIVE_MODES
    boundary_rms = 1.0
    include_constant = True
    if split == "T1":
        drive_modes = T1_DRIVE_MODES
        include_constant = False  # modes 5-8 ONLY (constant is in-range)
    elif split == "T2a":
        boundary_rms = 2.0
    elif split == "T2b":
        boundary_rms = 4.0
    elif split == "T3":
        deformation = T3_DEFORMATION

    geometry = sample_geometry(
        _case_seed(base, case_index, STREAM_GEOMETRY),
        modes=GEOMETRY_MODES,
        deformation_range=deformation,
        device=device,
        dtype=dtype,
    )
    drive = sample_drive(
        _case_seed(base, case_index, STREAM_DRIVE),
        modes=drive_modes,
        regularity=DRIVE_REGULARITY,
        boundary_rms=boundary_rms,
        include_constant=include_constant,
        device=device,
        dtype=dtype,
    )
    similarity = sample_similarity(
        _case_seed(base, case_index, STREAM_SIMILARITY),
        device=device,
        dtype=dtype,
    )
    return build_domain_sample(
        geometry,
        drive,
        n_boundary=n_boundary,
        n_query=n_query,
        query_seed=_case_seed(base, case_index, STREAM_QUERY),
        similarity=similarity,
    )


class MT2LaplaceAdapter(nn.Module):
    """Run MeshTransformer2 on the 2D Dirichlet-to-interior protocol.

    The 2D problem is embedded in the z = 0 plane.  Boundary panel midpoints
    and interior query points form one token set: boundary tokens carry the
    Dirichlet trace plus an is-boundary flag as boundary scalars, in-plane
    outward normals, and panel-length measure weights; interior tokens carry
    zero scalars, the constant out-of-plane normal (0, 0, 1) (pose-invariant
    in this embedding; a zero normal would be garbage after the model's
    normal normalization), and per-point area quadrature weights (enclosed
    polygon area / n_query).  The drive is the constant unit vector (1, 0, 0);
    with MT2's drive-degree-one bypass this leaves outputs untouched.  The
    prediction is read at the interior tokens' scalar output.
    """

    def __init__(self, **mt2_kwargs) -> None:
        super().__init__()
        self.model = MeshTransformer2(**mt2_kwargs)

    def forward(self, domain: DomainMesh) -> Mesh:
        boundary = domain.boundaries["dirichlet"]
        source = boundary.cell_centroids  # (S, 2) panel midpoints
        queries = domain.interior.points  # (Q, 2)
        n_source = source.shape[0]
        n_query = queries.shape[0]

        def lift(points_2d: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [points_2d, points_2d.new_zeros(points_2d.shape[0], 1)], dim=-1
            )

        points = torch.cat([lift(source), lift(queries)], dim=0)

        normals_boundary = lift(boundary.cell_normals)
        normals_interior = queries.new_zeros(n_query, 3)
        normals_interior[:, 2] = 1.0
        normals = torch.cat([normals_boundary, normals_interior], dim=0)

        values = boundary.cell_data["boundary_value"]
        scalars_boundary = torch.stack([values, torch.ones_like(values)], dim=-1)
        scalars = torch.cat(
            [scalars_boundary, values.new_zeros(n_query, 2)], dim=0
        )

        # Measure weights: exact panel lengths for boundary tokens; enclosed
        # polygon area (shoelace) split uniformly over queries for interior
        # tokens -- each token weighted by its quadrature measure.
        panel_lengths = boundary.cell_areas
        vertices = boundary.points
        rolled = torch.roll(vertices, -1, dims=0)
        area = 0.5 * torch.abs(
            torch.sum(
                vertices[:, 0] * rolled[:, 1] - rolled[:, 0] * vertices[:, 1]
            )
        )
        interior_weights = (area / n_query).expand(n_query)
        weights = torch.cat([panel_lengths, interior_weights], dim=0)

        drive = points.new_tensor([1.0, 0.0, 0.0])
        out = self.model(
            points[None],
            normals[None],
            drive,
            measure_weights=weights[None],
            boundary_scalars=scalars[None],
        )  # (1, S + Q, 1)
        potential = out[0, n_source:, 0]
        return domain.interior.with_data(
            point_data={"potential": potential},
            cell_data={},
            global_data=domain.global_data,
        )


def build_arm(arm: str) -> nn.Module:
    if arm == "mt2_bscalar":
        return MT2LaplaceAdapter(**MT2_CONFIG)
    if arm == "softslice_2d":
        return build_transolver_intree("intree_matched")
    if arm == "mt1_linear":
        # The suite's home model at reference capacity, linear field mode
        # (build_mesh_transformer default), moment query decoder.
        return make_model("mesh_transformer", "reference")
    raise ValueError(f"unknown arm {arm!r}")


def _stripped(sample: ConformalLaplaceSample) -> DomainMesh:
    """Model-facing domain: targets and eval-only metadata removed."""

    return DomainMesh(
        interior=sample.domain.interior.with_data(
            point_data={}, cell_data={}, global_data={}
        ),
        boundaries=sample.domain.boundaries,
        global_data=sample.domain.global_data,
    )


def predict(model: nn.Module, sample: ConformalLaplaceSample) -> torch.Tensor:
    return model(_stripped(sample)).point_data["potential"]


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target)
    ).item()


def evaluate(model: nn.Module, cases: list[ConformalLaplaceSample]) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for sample in cases:
            errors.append(relative_l2(predict(model, sample), sample.target))
    model.train()
    return sum(errors) / len(errors)


def train_arm_seed(
    model: nn.Module,
    train_cases: list[ConformalLaplaceSample],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    log_prefix: str,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
) -> list[float]:
    """Identical protocol across arms: Adam, batch = 1 case, fp32.

    ``checkpoint_path`` enables resumable training (model + optimizer +
    epoch counter) so cluster wall-time limits lose minutes, not runs.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_losses: list[float] = []
    start_epoch = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        epoch_losses = list(state["epoch_losses"])
        print(f"[{log_prefix}] resumed at epoch {start_epoch}", flush=True)
    shuffler = torch.Generator(device="cpu")
    for epoch in range(start_epoch, epochs):
        shuffler.manual_seed(1_000_000 * seed + epoch)
        order = torch.randperm(len(train_cases), generator=shuffler).tolist()
        total = 0.0
        for index in order:
            sample = train_cases[index]
            optimizer.zero_grad(set_to_none=True)
            loss = relative_mse(
                predict(model, sample), sample.target, sample.area_jacobian
            )
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_losses.append(total / len(train_cases))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[{log_prefix}] epoch {epoch:4d}  "
                f"train rel-MSE {epoch_losses[-1]:.6f}",
                flush=True,
            )
        if checkpoint_path is not None and (
            epoch % checkpoint_every == 0 or epoch == epochs - 1
        ):
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "epoch_losses": epoch_losses,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["mt2_bscalar", "softslice_2d", "mt1_linear"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--n-boundary", type=int, default=128)
    parser.add_argument("--n-query", type=int, default=512)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "g3_results.json")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Enable resumable training checkpoints (for wall-time-limited jobs)",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        help="Merge partial result JSONs into --output instead of training",
    )
    args = parser.parse_args()

    if args.merge is not None:
        merged: dict[str, dict] = {}
        config = None
        for path in args.merge:
            payload = json.loads(path.read_text())
            config = payload.pop("config", config)
            for arm, seeds in payload.items():
                merged.setdefault(arm, {}).update(seeds)
        # Union the recorded per-run bookkeeping across partials.
        for path in args.merge:
            partial_config = json.loads(path.read_text()).get("config", {})
            for key in ("parameter_counts", "wall_time_seconds", "final_train_rel_mse"):
                merged_map = config.setdefault(key, {})
                for name, value in partial_config.get(key, {}).items():
                    if isinstance(value, dict):
                        merged_map.setdefault(name, {}).update(value)
                    else:
                        merged_map[name] = value
        merged["config"] = config
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"Merged {len(args.merge)} partials -> {args.output}")
        return

    if args.smoke:
        args.epochs, args.n_train, args.n_test = 2, 8, 8
        args.output = RESULTS_DIR / "g3_results_smoke.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        # Preregistered CPU fallback (recorded in the config block).
        args.n_boundary, args.n_query, args.n_train = 96, 384, 128
    torch.set_float32_matmul_precision("highest")

    start = time.time()
    print(f"Generating cases on {device} ...", flush=True)
    train_cases = [
        make_g3_case(
            "train", i, n_boundary=args.n_boundary, n_query=args.n_query, device=device
        )
        for i in range(args.n_train)
    ]
    test_sets = {
        split: [
            make_g3_case(
                split, i, n_boundary=args.n_boundary, n_query=args.n_query, device=device
            )
            for i in range(args.n_test)
        ]
        for split in ("T0", "T1", "T2a", "T2b", "T3")
    }
    print(f"Generated in {time.time() - start:.1f}s", flush=True)

    results: dict[str, dict] = {}
    param_counts: dict[str, int] = {}
    wall_times: dict[str, dict[str, float]] = {}
    final_losses: dict[str, dict[str, float]] = {}
    if args.output.exists():
        # Idempotent restart (chained cluster jobs): keep finished runs.
        previous = json.loads(args.output.read_text())
        previous_config = previous.pop("config", {})
        results = previous
        param_counts = previous_config.get("parameter_counts", {})
        wall_times = previous_config.get("wall_time_seconds", {})
        final_losses = previous_config.get("final_train_rel_mse", {})
    for arm in args.arms:
        results.setdefault(arm, {})
        wall_times.setdefault(arm, {})
        final_losses.setdefault(arm, {})
        for seed in args.seeds:
            if str(seed) in results[arm]:
                print(f"[{arm}/s{seed}] already complete; skipping", flush=True)
                continue
            run_start = time.time()
            torch.manual_seed(seed)
            model = build_arm(arm).to(device=device, dtype=torch.float32)
            param_counts[arm] = parameter_count(model)
            checkpoint_path = None
            if args.checkpoint_dir is not None:
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = args.checkpoint_dir / f"{arm}_seed{seed}.pt"
            losses = train_arm_seed(
                model,
                train_cases,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=seed,
                log_prefix=f"{arm}/s{seed}",
                checkpoint_path=checkpoint_path,
            )
            metrics = {
                split: evaluate(model, cases) for split, cases in test_sets.items()
            }
            results[arm][str(seed)] = metrics
            wall_times[arm][str(seed)] = time.time() - run_start
            final_losses[arm][str(seed)] = losses[-1]
            print(f"[{arm}/s{seed}] {metrics}", flush=True)
            # Incremental write so partial progress is never lost.
            _write(args, device, results, param_counts, wall_times, final_losses)

    _write(args, device, results, param_counts, wall_times, final_losses)
    print(f"Done in {time.time() - start:.1f}s -> {args.output}", flush=True)


def _write(args, device, results, param_counts, wall_times, final_losses) -> None:
    config = {
        "preregistration": "g3_2d_bc_generalization_preregistration_2026-08-20.json",
        "device": str(device),
        "dtype": "float32",
        "matmul_precision": "highest",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch": "1 case",
        "training_loss": "relative_mse (area-jacobian-weighted relative L2^2, the suite's standard)",
        "eval_metric": "unweighted rel-L2 = ||pred-target||_2 / ||target||_2 per case, mean over cases",
        "n_train_cases": args.n_train,
        "n_test_cases_per_split": args.n_test,
        "n_boundary": args.n_boundary,
        "n_query": args.n_query,
        "smoke": args.smoke,
        "geometry_modes": list(GEOMETRY_MODES),
        "train_deformation_range": list(TRAIN_DEFORMATION),
        "t3_deformation_range": list(T3_DEFORMATION),
        "train_drive_modes": list(TRAIN_DRIVE_MODES),
        "t1_drive_modes": list(T1_DRIVE_MODES),
        "t1_include_constant": False,
        "drive_regularity": DRIVE_REGULARITY,
        "boundary_rms": {"train/T0/T1/T3": 1.0, "T2a": 2.0, "T2b": 4.0},
        "similarity": "sample_similarity defaults (scale 0.5-2 log-uniform, random O(2) incl. reflections, translation extent 2)",
        "split_base_seeds": SPLIT_BASE_SEEDS,
        "seed_streams": {"geometry": 0, "drive": 1, "query": 2, "similarity": 3},
        "arms": {
            "mt2_bscalar": {
                **MT2_CONFIG,
                "adapter": "boundary panels + interior queries as one token set; "
                "z=0 embedding; interior normals (0,0,1); constant drive (1,0,0); "
                "boundary scalars [value, is_boundary]; measure weights = panel "
                "lengths / shoelace-area per query",
            },
            "softslice_2d": "transolver_intree preset intree_matched (native suite adapter)",
            "mt1_linear": "make_model('mesh_transformer', 'reference'): moment decoder, field_mode=linear, explicit gauge",
        },
        "deviations": [
            "MT2 uses n_boundary_scalars=2 ([value, is_boundary flag]) instead of the "
            "prereg's 1: interior query tokens need a zero-scalar channel plus a flag "
            "to be distinguishable from zero-Dirichlet boundary panels.",
            "T1 drops the constant drive term (include_constant=False) so the test "
            "set contains modes 5-8 ONLY, matching the suite's "
            "unseen_boundary_frequencies convention.",
        ],
        "parameter_counts": param_counts,
        "wall_time_seconds": wall_times,
        "final_train_rel_mse": final_losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    payload["config"] = config
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
