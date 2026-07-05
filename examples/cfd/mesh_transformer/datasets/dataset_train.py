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

r"""Epoch-based training on cataloged boundary-to-interior datasets.

The streaming drivers (``train.py``, ``liouville.py``, ...) train in the
infinite-data regime: every optimizer step sees a freshly generated case, so
an epoch over a small fixed geometry set cannot masquerade as geometric
generalization.  Mainstream neural-operator baselines (e.g. Transolver) are
published in the opposite regime -- a large *fixed* dataset revisited for
many epochs.  This driver trains any ``train.make_model`` arm in that fixed
dataset regime against a cataloged dataset
(:mod:`dataset_catalog`), so the two regimes can be compared under an
otherwise identical protocol.

Protocol parity with the streaming drivers (deliberate, not incidental):

- AdamW at learning rate ``3e-4`` and weight decay ``1e-6`` (every streaming
  driver in this example uses exactly these values), no learning-rate
  schedule;
- gradient-norm clipping at ``1.0`` (as in ``liouville.py`` and the other
  single-equation streaming drivers);
- per-case relative-MSE training loss
  ``sum((prediction - target)^2) / sum(target^2)`` (the streaming drivers'
  unweighted convention: cataloged cases carry solver-verified pointwise
  targets but no area-quadrature weights);
- unweighted relative-L2 evaluation, best-validation checkpoint selection
  with the best state restored before the final split evaluation.

Dataset-regime specifics (the only intentional departures):

- data comes from the catalog's ``train`` split, shuffled each epoch by a
  reproducibly seeded ``torch.Generator`` instead of being drawn fresh;
- ``--batch-cases K`` accumulates gradients over ``K`` single-case
  forward/backward passes before each optimizer step (a trailing partial
  batch is averaged over its actual size);
- validation uses a fixed subset (up to 8 cases) of the catalog's
  ``eval_id`` split, every ``max(1, epochs // 12)`` epochs and at the final
  epoch.

Cases are cached in memory after first load (a production catalog is tens of
megabytes); ``--no-cache`` re-reads from disk every access.

This is a benchmark-local research driver, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _paths  # noqa: F401
import dataset_catalog
import torch
from torch import nn
from train import CAPACITY_CONFIGS, MODEL_NAMES, make_model

from physicsnemo.mesh import DomainMesh

VALIDATION_SPLIT = "eval_id"
MAX_VALIDATION_CASES = 8


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Unweighted relative L2 error (the streaming drivers' convention)."""

    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    )


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-case relative MSE training loss (the streaming drivers' convention)."""

    return torch.sum((prediction - target).square()) / torch.sum(
        target.square()
    ).clamp_min(1.0e-30)


class CaseBank:
    """Serve cataloged cases as device-resident ``DomainMesh`` samples.

    Samples land on the requested device and dtype at load time.  With
    ``cache=True`` (default) each case is loaded from disk once and reused
    across epochs; ``cache=False`` re-reads and rebuilds on every access.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache: bool = True,
    ) -> None:
        self._directory = Path(directory)
        self._device = device
        self._dtype = dtype
        self._cache: dict[int, tuple[DomainMesh, torch.Tensor]] | None = (
            {} if cache else None
        )

    def sample(self, index: int) -> tuple[DomainMesh, torch.Tensor]:
        """Return ``(domain, target)`` for one case index."""

        if self._cache is not None and index in self._cache:
            return self._cache[index]
        case = dataset_catalog.load_case(self._directory, index)
        sample = dataset_catalog.load_domain_sample(
            case, device=self._device, dtype=self._dtype
        )
        if self._cache is not None:
            self._cache[index] = sample
        return sample


def _predict(model: nn.Module, domain: DomainMesh) -> torch.Tensor:
    """Predict interior potential with targets stripped at the benchmark boundary.

    Mirrors ``train._predict``: the loaded interior mesh carries the
    solver-verified ``potential`` target as point data, which no candidate
    model may see.
    """

    model_domain = DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}, global_data={}),
        boundaries=domain.boundaries,
        global_data=domain.global_data,
    )
    return model(model_domain).point_data["potential"]


@torch.no_grad()
def _mean_relative_l2(model: nn.Module, bank: CaseBank, indices: list[int]) -> float:
    """Mean per-case relative L2 over a fixed case-index list."""

    model.eval()
    errors = []
    for index in indices:
        domain, target = bank.sample(index)
        errors.append(_relative_l2(_predict(model, domain), target))
    return sum(errors) / len(errors)


def run_experiment(
    *,
    dataset_dir: Path | str,
    model_name: str,
    epochs: int,
    seed: int,
    device: str,
    output_dir: Path | str,
    capacity: str = "reference",
    batch_cases: int = 1,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-6,
    cache: bool = True,
) -> dict:
    """Train one arm on a cataloged dataset for a fixed number of epochs.

    Returns the report dict and writes it to
    ``<output_dir>/<model_name>_seed<seed>.json``.
    """

    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if batch_cases < 1:
        raise ValueError("batch_cases must be positive")

    dataset_dir = Path(dataset_dir)
    manifest = dataset_catalog.load_manifest(dataset_dir)
    train_indices = list(dataset_catalog.split_indices(manifest, "train"))
    eval_split_names = sorted(name for name in manifest["splits"] if name != "train")
    if VALIDATION_SPLIT not in eval_split_names:
        raise dataset_catalog.CatalogError(
            f"manifest defines no {VALIDATION_SPLIT!r} split for validation; "
            f"available: {sorted(manifest['splits'])}"
        )
    validation_indices = list(
        dataset_catalog.split_indices(manifest, VALIDATION_SPLIT)
    )[:MAX_VALIDATION_CASES]

    torch.manual_seed(seed)
    device_t = torch.device(device)
    dtype = torch.float32
    model = make_model(model_name, capacity).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]
    bank = CaseBank(dataset_dir, device=device_t, dtype=dtype, cache=cache)

    history: list[dict[str, float | int]] = []
    best_state, best_val, best_epoch = None, float("inf"), 0
    start_time = time.perf_counter()
    if parameters and epochs > 0:
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        shuffler = torch.Generator(device="cpu").manual_seed(seed)
        validate_every = max(1, epochs // 12)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_indices), generator=shuffler).tolist()
            losses: list[float] = []
            for batch_start in range(0, len(order), batch_cases):
                batch = order[batch_start : batch_start + batch_cases]
                optimizer.zero_grad(set_to_none=True)
                for position in batch:
                    domain, target = bank.sample(train_indices[position])
                    loss = _relative_mse(_predict(model, domain), target)
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach().cpu()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_relative_mse": sum(losses) / len(losses),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if epoch % validate_every == 0 or epoch == epochs:
                validation = _mean_relative_l2(model, bank, validation_indices)
                record["validation_relative_l2"] = validation
                if validation < best_val:
                    best_val = validation
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            history.append(record)
            print(json.dumps(record), flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_val = _mean_relative_l2(model, bank, validation_indices)

    splits = {
        name: _mean_relative_l2(
            model, bank, list(dataset_catalog.split_indices(manifest, name))
        )
        for name in eval_split_names
    }
    report = {
        "model": model_name,
        "capacity": capacity,
        "dataset": {
            "family": manifest["family"],
            "version": manifest["version"],
            "path": str(dataset_dir.resolve()),
            "n_cases": manifest["n_cases"],
        },
        "equation": manifest["solver_settings"].get("equation"),
        "seed": seed,
        "epochs": epochs,
        "batch_cases": batch_cases,
        "n_train_cases": len(train_indices),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": history,
        "best_validation_relative_l2": best_val,
        "best_epoch": best_epoch,
        "validation_split": VALIDATION_SPLIT,
        "validation_cases": len(validation_indices),
        "splits": splits,
        "split_sizes": {
            name: spec["stop"] - spec["start"]
            for name, spec in manifest["splits"].items()
        },
        "verification": manifest["verification"],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name}_seed{seed}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Catalog directory containing manifest.json and case_*.npz",
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--capacity", choices=tuple(CAPACITY_CONFIGS), default="reference"
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--batch-cases",
        type=int,
        default=1,
        help="Cases whose gradients accumulate before each optimizer step",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-6,
        help="AdamW weight decay; 1e-6 matches every streaming driver",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-read cases from disk every access instead of caching",
    )
    return parser.parse_args()


def main() -> None:
    """Train one declared arm on one cataloged dataset and print the report."""

    args = _parse_args()
    report = run_experiment(
        dataset_dir=args.dataset,
        model_name=args.model,
        capacity=args.capacity,
        epochs=args.epochs,
        batch_cases=args.batch_cases,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        cache=not args.no_cache,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("model", "best_validation_relative_l2", "splits")
            }
        )
    )


if __name__ == "__main__":
    main()
