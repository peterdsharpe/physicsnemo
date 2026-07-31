#!/usr/bin/env python3
"""Evaluate one adaptation checkpoint on a fixed validation point sample."""

from __future__ import annotations

import math
import os
from pathlib import Path

import hydra
from datasets import build_dataloaders
from loss import LossCalculator
from metrics import MetricCalculator, resolve_metrics
from omegaconf import DictConfig, OmegaConf
from output_normalize import require_output_type
from train import val_epoch
from utils import FieldType, make_jsonl_logger, resolve_dict, set_seed

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    """Run a validation-only pass for one stored checkpoint."""
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    if dist_manager.world_size != 1:
        raise ValueError("Fixed validation evaluation requires one process per model.")
    device = dist_manager.device
    logger = RankZeroLoggingWrapper(
        PythonLogger(name="fixed_validation"),
        dist_manager,
    )

    evaluation_epoch = int(cfg.evaluation_epoch)
    evaluation_seed = int(cfg.evaluation_seed)
    output_path = Path(str(cfg.evaluation_output))
    checkpoint_path = Path(str(cfg.checkpoint_path))
    if evaluation_epoch < 1:
        raise ValueError(
            "evaluation_epoch uses completed-epoch indexing and must be >= 1"
        )

    set_seed(evaluation_seed, rank=dist_manager.rank)
    _train_loader, val_loader, _normalizer, dataset_info = build_dataloaders(cfg)
    del _train_loader
    # This is the essential protocol distinction from the online validation
    # loop: reseed the shared dataset, then iterate validation directly without
    # consuming any training sample first.
    val_loader.set_epoch(0)

    target_config: dict[str, FieldType] = dataset_info["targets"]
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    loaded_epoch = load_checkpoint(
        path=checkpoint_path,
        models=model,
        epoch=evaluation_epoch,
        device=device,
    )
    if loaded_epoch != evaluation_epoch:
        raise FileNotFoundError(
            f"requested checkpoint {evaluation_epoch}, loaded {loaded_epoch} "
            f"from {checkpoint_path}"
        )
    model.eval()

    metric_calculator = MetricCalculator(
        target_config=target_config,
        metrics=resolve_metrics(cfg),
    )
    loss_calculator = LossCalculator(
        target_config=target_config,
        loss_type=cfg.training.get("loss_type", "huber"),
        field_weights=resolve_dict(cfg, "training.field_weights"),
    )
    output_type = require_output_type(cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary_path.unlink(missing_ok=True)
    log_jsonl = make_jsonl_logger(temporary_path)
    log_jsonl(
        {
            "phase": "fixed_eval_config",
            "run_id": str(cfg.run_id),
            "checkpoint_path": str(checkpoint_path),
            "completed_epoch": evaluation_epoch,
            "evaluation_seed": evaluation_seed,
            "validation_loader_epoch": 0,
            "validation_cases": len(val_loader.sampler),
            "sampling_resolution": int(cfg.sampling_resolution),
            "training_loader_iterated": False,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    loss, metrics = val_epoch(
        val_loader,
        model,
        loss_calculator,
        metric_calculator,
        logger,
        evaluation_epoch - 1,
        cfg,
        dist_manager,
        output_type=output_type,
        target_config=target_config,
        log_jsonl=log_jsonl,
    )
    values = {
        "loss": float(loss),
        **{name: float(value) for name, value in metrics.items()},
    }
    non_finite = {
        name: value for name, value in values.items() if not math.isfinite(value)
    }
    if non_finite:
        raise RuntimeError(
            f"fixed evaluation metric guard triggered: non-finite values={non_finite}"
        )
    log_jsonl(
        {
            "phase": "fixed_eval_complete",
            "completed_epoch": evaluation_epoch,
            "evaluation_seed": evaluation_seed,
            "metrics": {name: float(value) for name, value in metrics.items()},
        }
    )
    os.replace(temporary_path, output_path)


if __name__ == "__main__":
    main()
