#!/bin/bash
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --job-name=lap-readout-smoke
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

on_exit() {
    local rc=$?
    echo "rc=${rc}" > "${TASK_DIR}/STATUS_${SLURM_JOB_ID}"
}
trap on_exit EXIT

"${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path

import torch

task_dir = Path.cwd()
sys.path.insert(
    0,
    str(task_dir / "code/examples/cfd/mesh_transformer/studies"),
)

import laplace_readout_factorial as study
from train import TRAIN_SPLIT, make_training_case

device = torch.device("cuda")
model = study.build_arm_model("pure", 17).to(device=device, dtype=torch.float32)
sample = make_training_case(
    TRAIN_SPLIT,
    distribution="boundary_balanced_mixture",
    seed=17,
    case_index=0,
    n_boundary=16,
    n_query=16,
    device=device,
    dtype=torch.float32,
)
prediction = model(sample.domain).point_data["potential"]
loss = torch.mean((prediction - sample.target).square())
loss.backward()
assert torch.isfinite(loss)
print(
    f"SMOKE_COMPLETED device={torch.cuda.get_device_name(0)} loss={loss.item():.6g}",
    flush=True,
)
PY

touch "${TASK_DIR}/DONE_${SLURM_JOB_ID}"
