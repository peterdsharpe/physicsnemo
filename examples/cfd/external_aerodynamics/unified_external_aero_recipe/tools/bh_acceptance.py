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

"""Barnes--Hut decode backend acceptance measurements (task #41, A + B).

Loads a trained checkpoint (same knobs as ``infer.py``: ``model=``,
``dataset=``, ``run_id=``), grabs one real validation sample, and measures:

- **Acceptance A (fidelity)**: BH-vs-dense per-field relative L2 on the
  model outputs at theta in {0.25, 0.5, 1.0}, in fp32 (autocast off) so the
  measurement sees the approximation error rather than bf16 noise.
- **Acceptance B (speed)**: median forward and forward+backward wall time,
  dense vs BH(theta=0.5), at the launched ``sampling_resolution`` -- plus
  crossover points at smaller resolutions.

The phase is selected with the ``BH_MODE`` environment variable so the four
GPUs of a minimum-size allocation each run one phase concurrently
(``fidelity`` | ``speed`` | ``crossover``; ``BH_RES`` gives crossover its
comma-separated resolution list).  Results land as JSON in
``$BH_OUT/bh_acceptance_<mode>.json``.

Example (one GPU worker):
    BH_MODE=fidelity BH_OUT=/path/out CUDA_VISIBLE_DEVICES=0 \
    python tools/bh_acceptance.py model=mesh_transformer_surface_flagship \
        dataset=drivaer_ml_surface run_id=t2_mesh_transformer_surface_flagship_seed42 \
        checkpoint_dir=runs infer_split=val sampling_resolution=50000
"""

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import build_dataloaders  # noqa: E402
from infer import resolve_checkpoint_path  # noqa: E402
from output_normalize import (  # noqa: E402
    normalize_output_to_tensordict,
    require_output_type,
)
from utils import get_autocast_context, recursive_to_device, set_seed  # noqa: E402

from physicsnemo import datapipes  # noqa: F401, E402 - registers ${dp:...}
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

THETAS = tuple(
    float(t) for t in os.environ.get("BH_THETAS", "0.25,0.5,1.0").split(",")
)
BH_LEAF_SIZE = 32
BH_PRODUCTION_THETA = float(os.environ.get("BH_SPEED_THETA", "0.5"))
#: Dense at 50k sources cannot afford 65536-query chunks (the (Q, S) member
#: tensors would not fit); both backends run with the same reduced chunk so
#: the comparison stays like-for-like.  Values are chunk-size independent by
#: the decoder's query-set-independence contract.
QUERY_CHUNK_SIZE = 8192


def _first_batch(cfg: DictConfig, device):
    """One real validation sample, collated exactly like training."""
    _train_loader, val_loader, _normalizer, dataset_info = build_dataloaders(cfg)
    dataset = val_loader.dataset
    idx = next(iter(val_loader.sampler))
    sample = dataset[idx]
    _domain, metadata = sample
    batch = recursive_to_device(val_loader.collate_fn([sample]), device)
    return batch, dataset_info, metadata


def _instantiate(cfg: DictConfig, device, *, backend: str, theta: float = 0.5):
    # Plain-dict roundtrip: cfg.model is a struct config, so merging NEW
    # keys (the BH knobs) into it directly raises ConfigKeyError.
    base = OmegaConf.to_container(cfg.model, resolve=True)
    base["query_chunk_size"] = QUERY_CHUNK_SIZE
    if backend == "barnes_hut":
        base.update(
            {
                "kernel_decode_backend": "barnes_hut",
                "kernel_bh_theta": float(theta),
                "kernel_bh_leaf_size": BH_LEAF_SIZE,
                "kernel_checkpoint_query_chunks": False,
            }
        )
    mcfg = OmegaConf.create(base)
    model = hydra.utils.instantiate(mcfg, _convert_="partial").to(device)
    model.eval()
    return model


def _fields(output, dataset_info, cfg):
    td = normalize_output_to_tensordict(
        output, dataset_info["targets"], require_output_type(cfg)
    )
    return {k: v for k, v in td.items()}


def _sample_label(metadata) -> str:
    try:
        return str(metadata.get("run_id", metadata.get("id", "?")))
    except Exception:
        return "?"


def _rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    num = (pred.double() - ref.double()).square().sum().sqrt()
    den = ref.double().square().sum().sqrt().clamp_min(1e-30)
    return float(num / den)


def _run_fidelity(cfg, device, batch, dataset_info, state, results):
    """Acceptance A: per-field BH-vs-dense relative L2 vs theta, fp32."""
    dense = _instantiate(cfg, device, backend="dense")
    dense.load_state_dict(state)
    with torch.no_grad(), nullcontext():
        out_dense = dense(**batch["forward_kwargs"])
    ref = _fields(out_dense, dataset_info, cfg)
    del dense, out_dense
    torch.cuda.empty_cache()
    curve = {}
    for theta in THETAS:
        bh = _instantiate(cfg, device, backend="barnes_hut", theta=theta)
        bh.load_state_dict(state)
        with torch.no_grad():
            out_bh = bh(**batch["forward_kwargs"])
        fields = _fields(out_bh, dataset_info, cfg)
        curve[str(theta)] = {k: _rel_l2(fields[k], ref[k]) for k in ref}
        del bh, out_bh, fields
        torch.cuda.empty_cache()
    results["fidelity_rel_l2_fp32"] = curve
    results["fidelity_pass_1e-3_at_theta"] = {
        t: all(v <= 1e-3 for v in errs.values()) for t, errs in curve.items()
    }


def _time_step(model, batch, cfg, dataset_info, *, backward, warmup=1, iters=3):
    """Median wall time of a forward (or forward+backward) pass, seconds."""
    times = []
    for i in range(warmup + iters):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if backward:
            with get_autocast_context(cfg.precision):
                output = model(**batch["forward_kwargs"])
                td = normalize_output_to_tensordict(
                    output, dataset_info["targets"], require_output_type(cfg)
                )
                loss = sum(v.float().square().sum() for v in td.values())
            loss.backward()
        else:
            with torch.no_grad(), get_autocast_context(cfg.precision):
                model(**batch["forward_kwargs"])
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def _run_speed_at(cfg, device, batch, dataset_info, state, label, results):
    dense = _instantiate(cfg, device, backend="dense")
    dense.load_state_dict(state)
    bh = _instantiate(cfg, device, backend="barnes_hut", theta=BH_PRODUCTION_THETA)
    bh.load_state_dict(state)
    entry = {}
    for name, model in (("dense", dense), ("barnes_hut", bh)):
        entry[f"{name}_forward_s"] = _time_step(
            model, batch, cfg, dataset_info, backward=False
        )
        entry[f"{name}_step_s"] = _time_step(
            model, batch, cfg, dataset_info, backward=True
        )
        torch.cuda.empty_cache()
    entry["speedup_forward"] = entry["dense_forward_s"] / entry["barnes_hut_forward_s"]
    entry["speedup_step"] = entry["dense_step_s"] / entry["barnes_hut_step_s"]
    results.setdefault("speed", {})[label] = entry
    del dense, bh
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    mode = os.environ.get("BH_MODE", "fidelity")
    out_dir = Path(os.environ.get("BH_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    DistributedManager.initialize()
    device = DistributedManager().device
    set_seed(int(cfg.training.get("seed", 42) or 42))

    ckpt_path = resolve_checkpoint_path(cfg)
    reference = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    loaded_epoch = load_checkpoint(path=ckpt_path, models=reference, device=device)
    if loaded_epoch == 0:
        raise FileNotFoundError(f"no checkpoint restored from {ckpt_path!r}")
    state = reference.state_dict()
    del reference
    torch.cuda.empty_cache()

    results = {
        "mode": mode,
        "run_id": str(cfg.run_id),
        "checkpoint_epoch": int(loaded_epoch),
        "theta": list(THETAS),
        "leaf_size": BH_LEAF_SIZE,
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "precision_speed": str(cfg.precision),
        "sampling_resolution": int(cfg.sampling_resolution),
    }

    if mode == "fidelity":
        batch, dataset_info, metadata = _first_batch(cfg, device)
        results["sample"] = _sample_label(metadata)
        _run_fidelity(cfg, device, batch, dataset_info, state, results)
    elif mode == "speed":
        batch, dataset_info, metadata = _first_batch(cfg, device)
        results["sample"] = _sample_label(metadata)
        _run_speed_at(
            cfg,
            device,
            batch,
            dataset_info,
            state,
            str(int(cfg.sampling_resolution)),
            results,
        )
    elif mode == "crossover":
        resolutions = [
            int(r) for r in os.environ.get("BH_RES", "5000,10000,20000").split(",")
        ]
        for res in resolutions:
            cfg.sampling_resolution = res
            batch, dataset_info, _metadata = _first_batch(cfg, device)
            _run_speed_at(
                cfg, device, batch, dataset_info, state, str(res), results
            )
            del batch
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"unknown BH_MODE {mode!r}")

    tag = os.environ.get("BH_TAG", "")
    out_path = out_dir / f"bh_acceptance_{mode}{('_' + tag) if tag else ''}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
