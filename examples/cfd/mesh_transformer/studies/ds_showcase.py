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

"""E2/E3 runner for the dataset chapters: showcases and resolution ladders.

The dataset-chapters plan (``studies/notes/dataset_chapters_plan.md``)
commissions, per suite: **E2**, an unseen-geometry *showcase* — train the
suite's reference arm cheaply, then evaluate it on geometry dials pushed
past training with qualitative predicted-vs-exact fields ("does it at
least give a reasonable prediction? most architectures give total
garbage" — Peter's framing); and **E3**, a *resolution ladder* — the same
checkpoint re-evaluated across a 4x span of boundary resolution.

One suite = one adapter implementing the three-phase protocol below;
the runner is deliberately dumb about suites and smart about phases:

- ``train``   -> run the suite's OWN training driver as a subprocess
                 (reference-arm flags pinned here), producing the
                 driver's native checkpoint in ``--work-dir``.
- ``showcase``-> load the checkpoint, build the pre-registered showcase
                 ladder (in-distribution anchor plus escalating dials),
                 record per-case relative L2 AND export the fields
                 (queries, exact, predicted, boundary loops) to a dated
                 npz for the chapter galleries.
- ``ladder``  -> re-evaluate a fixed eval bank across boundary-resolution
                 multipliers, exporting the transfer curve.

Adapters registered (all five complete): ``laplace2d``, ``screened``,
``laplace3d``, ``liouville``, ``potential_flow``.  Every showcase ladder
is PRE-REGISTERED in its ``_*_SHOWCASE`` constant below — an
in-distribution anchor followed by dials escalated past each suite's
training band (bands restated per rung from the suites' own split
constants), so "past band" is checkable against the suite source, and
each suite's E3 ladder spans a 4-8x range about its training
resolution.

DEVICE SPLIT: ``--device`` applies to the TRAIN phase only.  The
showcase and ladder phases always evaluate on CPU in float64 — the
exact-label builders produce CPU samples, evaluation is minutes of
work, and mixing a CUDA model with CPU samples was measured to crash
(job 4263092).  The train phase is where the GPU earns its allocation.

READING CAVEAT for far-query rungs (potential_flow "far queries r in
4-8" and any future spatial-extrapolation rung): the exact disturbance
field decays with radius, so the relative-L2 denominator is small and
the ratio is harsh — an untrained model scores in the thousands there,
and even a good model's far-rung ratio overstates its absolute error.
Chapter cells reading these records should either show absolute error
alongside, or normalize by the near-field target norm; the exported
fields support both.

CPU smoke (run before ANY cluster submission)::

    python studies/ds_showcase.py --suite laplace2d --phase all \
        --work-dir /tmp/ds_smoke --steps 2 --smoke

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for entry in (_HERE.parent, _HERE.parent / "problems", _HERE.parent / "datasets"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteAdapter:
    """Everything the runner needs to know about one suite."""

    name: str
    train_command: Callable[[Path, int, int], list[str]]
    checkpoint_path: Callable[[Path], Path]
    showcase: Callable[[Path, Path, torch.device, bool], dict]
    ladder: Callable[[Path, Path, torch.device, bool], dict]


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    denominator = float(target.double().square().sum().sqrt())
    if denominator == 0.0:
        return float("nan")
    return float(
        (prediction.double() - target.double()).square().sum().sqrt() / denominator
    )


# ---------------------------------------------------------------------------
# laplace2d adapter (complete)
# ---------------------------------------------------------------------------

_L2D_ARM = "mesh_transformer_kernel_singpair"
_L2D_CAPACITY = "reference"


def _l2d_train_command(
    work_dir: Path, steps: int, seed: int, device: str = "cpu"
) -> list[str]:
    return [
        sys.executable,
        str(_HERE.parent / "problems" / "train.py"),
        "--model", _L2D_ARM,
        "--capacity", _L2D_CAPACITY,
        "--steps", str(steps),
        "--seed", str(seed),
        "--device", device,
        "--output-dir", str(work_dir / "train"),
    ]


def _l2d_checkpoint(work_dir: Path) -> Path:
    return work_dir / "train" / f"{_L2D_ARM}_{_L2D_CAPACITY}.pt"


def _l2d_load(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from train import make_model

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    gauge = payload.get("run_config", {}).get("gauge", "explicit")
    model = make_model(payload["model"], payload["capacity"], gauge).to(device)
    model.load_state_dict(payload["state_dict"])
    # The showcase samples are built in float64 (exact labels); run the
    # trained float32 weights in float64 for the evaluation.
    model.double()
    model.eval()
    return model


#: The pre-registered laplace2d showcase ladder: an in-distribution anchor,
#: then each dial escalated PAST its training range (training ranges per
#: the suite constants: deformation <= 0.35, multi-body gap_ratio >= 0.5,
#: cavity depth <= 0.55).  Every rung reuses the suite's own exact-label
#: builders, so "exact" always means exact.
_L2D_SHOWCASE: list[dict] = [
    {"kind": "train_family", "label": "in-distribution anchor",
     "kwargs": {}},
    {"kind": "train_family", "label": "deformation 0.45-0.65 (past band)",
     "kwargs": {"deformation_range": (0.45, 0.65)}},
    {"kind": "train_family", "label": "deformation 0.70-0.90 (far past)",
     "kwargs": {"deformation_range": (0.70, 0.90)}},
    {"kind": "multi_body", "label": "two bodies, training gaps",
     "kwargs": {"gap_ratio_range": (0.5, 1.5)}},
    {"kind": "multi_body", "label": "two bodies, narrow gap 0.15-0.30",
     "kwargs": {"gap_ratio_range": (0.15, 0.30)}},
    {"kind": "multi_body", "label": "two bodies, extreme gap 0.08-0.15",
     "kwargs": {"gap_ratio_range": (0.08, 0.15)}},
    {"kind": "deep_cavity", "label": "cavity, training depths",
     "kwargs": {"depth_range": (0.35, 0.55)}},
    {"kind": "deep_cavity", "label": "cavity, depth 0.65-0.80 (past band)",
     "kwargs": {"depth_range": (0.65, 0.80)}},
]

_L2D_SEEDS = (5, 9, 23)


def _l2d_train_family_sample(seed: int, n_query: int, *, n_boundary: int = 64,
                             deformation_range=(0.05, 0.35)):
    from conformal_laplace import build_domain_sample, sample_drive, sample_geometry

    geometry = sample_geometry(
        seed, deformation_range=deformation_range, dtype=torch.float64)
    drive = sample_drive(seed + 1_000_003)
    return build_domain_sample(
        geometry, drive, n_boundary=n_boundary, n_query=n_query,
        query_seed=seed + 7)


def _l2d_build(kind: str, seed: int, n_query: int, **kwargs):
    if kind == "train_family":
        return _l2d_train_family_sample(seed, n_query, **kwargs)
    from encoder_stress import build_deep_cavity_sample, build_multi_body_sample

    builder = {
        "multi_body": build_multi_body_sample,
        "deep_cavity": build_deep_cavity_sample,
    }[kind]
    return builder(seed, n_query=n_query, dtype=torch.float64, **kwargs)


def _l2d_predict(model: torch.nn.Module, sample) -> torch.Tensor:
    from physicsnemo.mesh import DomainMesh

    # Strip interior metadata at the benchmark boundary (targets and
    # evaluation-only coordinates live on the generated interior mesh),
    # mirroring problems/train.py's _predict.
    domain = DomainMesh(
        interior=sample.domain.interior.with_data(
            point_data={}, cell_data={}, global_data={}),
        boundaries=sample.domain.boundaries,
        global_data=sample.domain.global_data,
    )
    with torch.no_grad():
        predicted = model(domain)
    key = next(iter(predicted.point_data.keys()))
    return predicted.point_data[key]


def _l2d_showcase(checkpoint: Path, out: Path, device: torch.device,
                  smoke: bool) -> dict:
    model = _l2d_load(checkpoint, device)
    n_query = 512 if smoke else 16384
    seeds = _L2D_SEEDS[:1] if smoke else _L2D_SEEDS
    payload: dict[str, np.ndarray] = {}
    records = []
    k = 0
    for rung, spec in enumerate(_L2D_SHOWCASE):
        for seed in seeds:
            sample = _l2d_build(spec["kind"], seed, n_query, **spec["kwargs"])
            prediction = _l2d_predict(model, sample)
            error = _relative_l2(prediction.cpu(), sample.target.cpu())
            records.append({
                "rung": rung, "kind": spec["kind"], "label": spec["label"],
                "seed": seed, "relative_l2": error,
                "n_query": int(sample.target.shape[0]),
            })
            payload[f"case{k}_queries"] = (
                sample.domain.interior.points.cpu().numpy().astype(np.float32))
            payload[f"case{k}_exact"] = (
                sample.target.cpu().numpy().astype(np.float32))
            payload[f"case{k}_pred"] = (
                prediction.cpu().numpy().astype(np.float32))
            loops = getattr(sample, "boundary_loops", None)
            if loops is None:
                loops = tuple(
                    mesh.points.cpu()
                    for mesh in sample.domain.boundaries.values())
            for j, loop in enumerate(loops):
                payload[f"case{k}_loop{j}"] = (
                    np.asarray(loop, dtype=np.float32))
            payload[f"case{k}_n_loops"] = np.int64(len(loops))
            print(json.dumps(records[-1]), flush=True)
            k += 1
    payload["n_cases"] = np.int64(k)
    payload["__records__"] = np.frombuffer(
        json.dumps(records).encode(), dtype=np.uint8)
    np.savez_compressed(out, **payload)
    return {"cases": k, "records": records}


#: E3 resolution multipliers about each family's training resolution.
_L2D_LADDER_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)


def _l2d_ladder(checkpoint: Path, out: Path, device: torch.device,
                smoke: bool) -> dict:
    model = _l2d_load(checkpoint, device)
    n_query = 512 if smoke else 4096
    seeds = _L2D_SEEDS[:1] if smoke else _L2D_SEEDS
    multipliers = _L2D_LADDER_MULTIPLIERS[:2] if smoke \
        else _L2D_LADDER_MULTIPLIERS
    records = []
    for multiplier in multipliers:
        n_boundary = max(16, int(round(64 * multiplier)))
        for seed in seeds:
            sample = _l2d_train_family_sample(
                seed, n_query, n_boundary=n_boundary)
            prediction = _l2d_predict(model, sample)
            records.append({
                "multiplier": multiplier, "n_boundary": n_boundary,
                "seed": seed,
                "relative_l2": _relative_l2(
                    prediction.cpu(), sample.target.cpu()),
            })
            print(json.dumps(records[-1]), flush=True)
    out.write_text(json.dumps({"suite": "laplace2d", "ladder": records},
                              indent=1))
    return {"points": len(records)}


# ---------------------------------------------------------------------------
# Generic driver-checkpoint machinery shared by the four driver-based suites.
# Each suite driver saves (via --save-checkpoint, the potential_flow.py
# convention) a payload {"model": name, ..., "state_dict": ...} whose model
# is rebuilt by that module's own _build_model.
# ---------------------------------------------------------------------------


def _driver_checkpoint(work_dir: Path) -> Path:
    matches = sorted((work_dir / "train").glob("*_seed*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"no driver checkpoint under {work_dir / 'train'} — did the "
            "train phase run with --save-checkpoint?")
    return matches[-1]


def _driver_load(
    module: str, checkpoint: Path, device: torch.device
) -> torch.nn.Module:
    import importlib

    builder = importlib.import_module(module)._build_model
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if "family" in payload:
        # potential_flow.py's _build_model(model_name, family, **flags)
        # signature; the checkpoint records the far-field knob flags.
        model = builder(
            payload["model"],
            payload["family"],
            bounded_gates=bool(payload.get("bounded_gates", False)),
            bounded_query=bool(payload.get("bounded_query", False)),
            decaying_drive=bool(payload.get("decaying_drive", False)),
            monopole_free_sl=bool(payload.get("monopole_free_sl", False)),
        ).to(device)
    else:
        model = builder(payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.double()
    model.eval()
    return model


def _run_showcase(
    model: torch.nn.Module,
    specs: list[dict],
    build: Callable[[dict, int, int], object],
    out: Path,
    *,
    n_query: int,
    seeds: tuple[int, ...],
) -> dict:
    """Shared showcase loop: evaluate every (rung, seed), export fields."""

    payload: dict[str, np.ndarray] = {}
    records = []
    k = 0
    for rung, spec in enumerate(specs):
        for seed in seeds:
            sample = build(spec, seed, n_query)
            prediction = _l2d_predict(model, sample)
            records.append({
                "rung": rung, "label": spec["label"], "seed": seed,
                "relative_l2": _relative_l2(
                    prediction.cpu(), sample.target.cpu()),
                "n_query": int(sample.target.shape[0]),
            })
            payload[f"case{k}_queries"] = (
                sample.domain.interior.points.cpu().numpy().astype(np.float32))
            payload[f"case{k}_exact"] = (
                sample.target.cpu().numpy().astype(np.float32))
            payload[f"case{k}_pred"] = (
                prediction.cpu().numpy().astype(np.float32))
            loops = getattr(sample, "boundary_loops", None)
            if loops is None:
                loops = tuple(
                    mesh.points.cpu()
                    for mesh in sample.domain.boundaries.values())
            for j, loop in enumerate(loops):
                payload[f"case{k}_loop{j}"] = np.asarray(loop, dtype=np.float32)
            payload[f"case{k}_n_loops"] = np.int64(len(loops))
            print(json.dumps(records[-1]), flush=True)
            k += 1
    payload["n_cases"] = np.int64(k)
    payload["__records__"] = np.frombuffer(
        json.dumps(records).encode(), dtype=np.uint8)
    np.savez_compressed(out, **payload)
    return {"cases": k, "records": records}


def _run_ladder(
    model: torch.nn.Module,
    suite: str,
    resolutions: list[dict],
    build: Callable[[dict, int, int], object],
    out: Path,
    *,
    n_query: int,
    seeds: tuple[int, ...],
) -> dict:
    records = []
    for spec in resolutions:
        for seed in seeds:
            sample = build(spec, seed, n_query)
            prediction = _l2d_predict(model, sample)
            records.append({
                **{key: value for key, value in spec.items()
                   if isinstance(value, (int, float, str))},
                "seed": seed,
                "relative_l2": _relative_l2(
                    prediction.cpu(), sample.target.cpu()),
            })
            print(json.dumps(records[-1]), flush=True)
    out.write_text(json.dumps({"suite": suite, "ladder": records}, indent=1))
    return {"points": len(records)}


def _driver_train_command(script: str, arm: str, extra: list[str] | None = None):
    def command(work_dir: Path, steps: int, seed: int,
                device: str = "cpu") -> list[str]:
        return [
            sys.executable, str(_HERE.parent / "problems" / script),
            "--model", arm,
            "--steps", str(steps),
            "--seed", str(seed),
            "--device", device,
            "--output-dir", str(work_dir / "train"),
            "--save-checkpoint",
            *(extra or []),
        ]

    return command


_SHOWCASE_SEEDS = (5, 9, 23)


def _suite_showcase(module: str, specs: list[dict], build, *,
                    n_query_full: int = 16384):
    def showcase(checkpoint: Path, out: Path, device: torch.device,
                 smoke: bool) -> dict:
        model = _driver_load(module, checkpoint, device)
        return _run_showcase(
            model, specs, build, out,
            n_query=512 if smoke else n_query_full,
            seeds=_SHOWCASE_SEEDS[:1] if smoke else _SHOWCASE_SEEDS)

    return showcase


def _suite_ladder(module: str, suite: str, resolutions: list[dict], build, *,
                  n_query_full: int = 4096):
    def ladder(checkpoint: Path, out: Path, device: torch.device,
               smoke: bool) -> dict:
        model = _driver_load(module, checkpoint, device)
        return _run_ladder(
            model, suite,
            resolutions[:2] if smoke else resolutions, build, out,
            n_query=512 if smoke else n_query_full,
            seeds=_SHOWCASE_SEEDS[:1] if smoke else _SHOWCASE_SEEDS)

    return ladder


# --- screened: training band kappa (0.5, 2.0), modes (0-3) (SPLITS) --------

_SCREENED_SHOWCASE: list[dict] = [
    {"label": "in-distribution anchor",
     "kwargs": {"kappa_range": (0.5, 2.0), "modes": (0, 1, 2, 3)}},
    {"label": "high screening 3-5 (past band)",
     "kwargs": {"kappa_range": (3.0, 5.0), "modes": (0, 1, 2, 3)}},
    {"label": "extreme screening 8-15 (far past)",
     "kwargs": {"kappa_range": (8.0, 15.0), "modes": (0, 1, 2, 3)}},
    {"label": "low screening 0.05-0.3 (Laplace limit)",
     "kwargs": {"kappa_range": (0.05, 0.3), "modes": (0, 1, 2, 3)}},
    {"label": "unseen modes 4-6",
     "kwargs": {"kappa_range": (0.5, 2.0), "modes": (4, 5, 6)}},
    {"label": "unseen modes + high screening (composed)",
     "kwargs": {"kappa_range": (3.0, 5.0), "modes": (4, 5, 6)}},
]

_SCREENED_LADDER = [
    {"multiplier": m, "n_boundary": max(16, int(round(64 * m)))}
    for m in (0.5, 1.0, 2.0, 4.0)
]


def _screened_build(spec: dict, seed: int, n_query: int):
    from screened_laplace import build_screened_sample

    kwargs = dict(spec.get("kwargs", {}))
    kwargs.setdefault("kappa_range", (0.5, 2.0))
    kwargs.setdefault("modes", (0, 1, 2, 3))
    return build_screened_sample(
        seed, n_query=n_query, dtype=torch.float64,
        n_boundary=spec.get("n_boundary", 64), **kwargs)


# --- laplace3d: trained on sphere+star dirichlet at subdivisions 2 ---------

_L3D_SHOWCASE: list[dict] = [
    {"label": "sphere, in-distribution anchor",
     "kwargs": {"tier": "sphere"}},
    {"label": "star, in-distribution",
     "kwargs": {"tier": "star"}},
    {"label": "star, unseen modes (ood set)",
     "kwargs": {"tier": "star", "star_modes": "ood"}},
    {"label": "shell (unseen topology)",
     "kwargs": {"tier": "shell"}},
    {"label": "shell at subdivisions 3 (topology + resolution)",
     "kwargs": {"tier": "shell", "subdivisions": 3}},
]

_L3D_LADDER = [
    {"multiplier": 4.0 ** (s - 2), "subdivisions": s} for s in (1, 2, 3, 4)
]


def _l3d_build(spec: dict, seed: int, n_query: int):
    from laplace3d import _STAR_MODES_OOD, build_laplace3d_sample

    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("star_modes") == "ood":
        kwargs["star_modes"] = _STAR_MODES_OOD
    kwargs.setdefault("subdivisions", spec.get("subdivisions", 2))
    return build_laplace3d_sample(
        seed, bc_regime="dirichlet", n_query=n_query,
        dtype=torch.float64, **kwargs)


# --- liouville: training band modes (2,3), deformation (0.05, 0.35) --------

_LIOUVILLE_SHOWCASE: list[dict] = [
    {"label": "in-distribution anchor",
     "kwargs": {"geometry_modes": (2, 3), "deformation_range": (0.05, 0.35)}},
    {"label": "deformation 0.45-0.65 (past band)",
     "kwargs": {"geometry_modes": (2, 3), "deformation_range": (0.45, 0.65)}},
    {"label": "deformation 0.70-0.90 (far past)",
     "kwargs": {"geometry_modes": (2, 3), "deformation_range": (0.70, 0.90)}},
    {"label": "unseen modes 4-5",
     "kwargs": {"geometry_modes": (4, 5), "deformation_range": (0.05, 0.35)}},
    {"label": "unseen modes + strong deformation (composed)",
     "kwargs": {"geometry_modes": (4, 5), "deformation_range": (0.45, 0.65)}},
]

_LIOUVILLE_LADDER = [
    {"multiplier": m, "n_boundary": max(16, int(round(64 * m)))}
    for m in (0.5, 1.0, 2.0, 4.0)
]


def _liouville_build(spec: dict, seed: int, n_query: int):
    from liouville import build_liouville_sample

    kwargs = dict(spec.get("kwargs", {}))
    return build_liouville_sample(
        seed, n_query=n_query, dtype=torch.float64,
        n_boundary=spec.get("n_boundary", 64), **kwargs)


# --- potential_flow (velocity family A'): circulation band (0, 1.5),
#     deformation (0.05, 0.35), modes (1-3), query radius (1.05, 4.0) -------

_PF_ARM = "mesh_transformer_kernel_singpair_pseudo"

_PF_SHOWCASE: list[dict] = [
    {"label": "in-distribution anchor", "kwargs": {}},
    {"label": "circulation 1.5-3.0 (past band)",
     "kwargs": {"circulation_magnitude_range": (1.5, 3.0)}},
    {"label": "circulation 3.0-5.0 (far past)",
     "kwargs": {"circulation_magnitude_range": (3.0, 5.0)}},
    {"label": "deformation 0.45-0.65 (past band)",
     "kwargs": {"deformation_range": (0.45, 0.65)}},
    {"label": "unseen modes 4-5",
     "kwargs": {"modes": (4, 5)}},
    {"label": "far queries r in 4-8 (spatial extrapolation)",
     "kwargs": {"query_radius_range": (4.0, 8.0)}},
]

_PF_LADDER = [
    {"multiplier": m, "n_boundary": max(40, int(round(160 * m)))}
    for m in (0.5, 1.0, 2.0, 4.0)
]


def _pf_build(spec: dict, seed: int, n_query: int):
    from potential_flow import build_potential_flow_velocity_sample

    kwargs = dict(spec.get("kwargs", {}))
    return build_potential_flow_velocity_sample(
        seed, n_query=n_query, dtype=torch.float64,
        n_boundary=spec.get("n_boundary", 160), **kwargs)


def _pf_train_command(work_dir: Path, steps: int, seed: int,
                      device: str = "cpu") -> list[str]:
    return [
        sys.executable, str(_HERE.parent / "problems" / "potential_flow.py"),
        "--model", _PF_ARM,
        "--family", "potential_flow_velocity",
        "--steps", str(steps),
        "--seed", str(seed),
        "--device", device,
        "--output-dir", str(work_dir / "train"),
        "--save-checkpoint",
    ]


SUITES: dict[str, SuiteAdapter] = {
    "laplace2d": SuiteAdapter(
        name="laplace2d",
        train_command=_l2d_train_command,
        checkpoint_path=_l2d_checkpoint,
        showcase=_l2d_showcase,
        ladder=_l2d_ladder,
    ),
    "screened": SuiteAdapter(
        name="screened",
        train_command=_driver_train_command(
            "screened_laplace.py", "mesh_transformer_kernel_singonly"),
        checkpoint_path=_driver_checkpoint,
        showcase=_suite_showcase(
            "screened_laplace", _SCREENED_SHOWCASE, _screened_build),
        ladder=_suite_ladder(
            "screened_laplace", "screened", _SCREENED_LADDER, _screened_build),
    ),
    "laplace3d": SuiteAdapter(
        name="laplace3d",
        train_command=_driver_train_command(
            "laplace3d_study.py", "mesh_transformer_kernel_singpair"),
        checkpoint_path=_driver_checkpoint,
        showcase=_suite_showcase(
            "laplace3d_study", _L3D_SHOWCASE, _l3d_build, n_query_full=4096),
        ladder=_suite_ladder(
            "laplace3d_study", "laplace3d", _L3D_LADDER, _l3d_build,
            n_query_full=2048),
    ),
    "liouville": SuiteAdapter(
        name="liouville",
        train_command=_driver_train_command(
            "liouville.py", "mesh_transformer_kernel_nl_singpair"),
        checkpoint_path=_driver_checkpoint,
        showcase=_suite_showcase(
            "liouville", _LIOUVILLE_SHOWCASE, _liouville_build),
        ladder=_suite_ladder(
            "liouville", "liouville", _LIOUVILLE_LADDER, _liouville_build),
    ),
    "potential_flow": SuiteAdapter(
        name="potential_flow",
        train_command=_pf_train_command,
        checkpoint_path=_driver_checkpoint,
        showcase=_suite_showcase(
            "potential_flow", _PF_SHOWCASE, _pf_build),
        ladder=_suite_ladder(
            "potential_flow", "potential_flow", _PF_LADDER, _pf_build),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument(
        "--phase", choices=("train", "showcase", "ladder", "all"),
        default="all")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny query counts / truncated ladders for the CPU smoke")
    arguments = parser.parse_args()

    adapter = SUITES[arguments.suite]
    # --device drives TRAINING only; evaluation is CPU float64 (see the
    # module docstring's DEVICE SPLIT note).
    device = torch.device("cpu")
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")

    if arguments.phase in ("train", "all"):
        try:
            existing = adapter.checkpoint_path(arguments.work_dir)
        except FileNotFoundError:
            existing = None
        if existing is not None and Path(existing).exists():
            # Resume-chain guard: training already produced its
            # checkpoint; do not burn a GPU allocation re-training.
            print(f"TRAIN: checkpoint exists ({existing}); skipping",
                  flush=True)
        else:
            command = adapter.train_command(
                arguments.work_dir, arguments.steps, arguments.seed,
                arguments.device)
            print("TRAIN:", " ".join(command), flush=True)
            subprocess.run(command, check=True)

    checkpoint = adapter.checkpoint_path(arguments.work_dir)
    if arguments.phase in ("showcase", "all"):
        result = adapter.showcase(
            checkpoint,
            arguments.work_dir / f"{adapter.name}_showcase_{stamp}.npz",
            device, arguments.smoke)
        print(json.dumps({"phase": "showcase", **result})[:500], flush=True)
    if arguments.phase in ("ladder", "all"):
        result = adapter.ladder(
            checkpoint,
            arguments.work_dir / f"{adapter.name}_ladder_{stamp}.json",
            device, arguments.smoke)
        print(json.dumps({"phase": "ladder", **result})[:500], flush=True)


if __name__ == "__main__":
    main()
