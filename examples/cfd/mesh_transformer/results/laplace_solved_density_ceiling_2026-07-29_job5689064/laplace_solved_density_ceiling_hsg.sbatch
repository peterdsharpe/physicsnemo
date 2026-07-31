#!/bin/bash
# Evaluate the parameter-free solved-density control on the factorial case bank.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=00:15:00
#SBATCH --job-name=lap-density-ceiling
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/laplace_solved_density_ceiling.py"
OUTPUT="${ARTIFACT_DIR}/solved_density_ceiling.json"

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${ARTIFACT_DIR}" "${TASK_DIR}/sbatch_logs"

sampler() {
    while true; do
        date --iso-8601=seconds
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,memory.used \
            --format=csv,noheader
        sleep 300
    done
}

sampler &
SAMPLER_PID=$!

on_exit() {
    local rc=$?
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    echo "rc=${rc}" > "${TASK_DIR}/STATUS_${SLURM_JOB_ID}"
}
trap on_exit EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${STUDY}" ]]; then
    echo "missing study source: ${STUDY}" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite existing artifact: ${OUTPUT}" >&2
    exit 2
fi

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${STUDY}" \
    --device cuda \
    --output "${OUTPUT}"

"${PYTHON_BIN}" - "${OUTPUT}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
assert report["study"] == "laplace_solved_density_ceiling_v1"
assert report["registered_parameters"] == 0
assert report["trainable_parameters"] == 0
assert set(report["evaluation"]["splits"]) == {
    "interpolation",
    "mixed_geometry_modes",
    "stronger_deformation",
    "unseen_boundary_frequencies",
    "unseen_geometry_modes",
}

def leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)

assert all(math.isfinite(value) for value in leaves(report["evaluation"]))
PY

echo "HEARTBEAT completed_units=1"
touch "${TASK_DIR}/DONE_${SLURM_JOB_ID}"
echo "DONE completed_units=1"
