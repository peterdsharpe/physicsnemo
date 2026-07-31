#!/bin/bash
# One HSG node runs the four paired readout arms concurrently, one per GPU.
# Five seed waves produce one atomic JSON artifact per arm and seed.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --job-name=lap-readout-factorial
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
RUN_LOG_DIR="${TASK_DIR}/run_logs"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/laplace_readout_factorial.py"

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${ARTIFACT_DIR}" "${RUN_LOG_DIR}" "${TASK_DIR}/sbatch_logs"

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

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

run_one() {
    local gpu=$1
    local arm=$2
    local seed=$3
    local output="${ARTIFACT_DIR}/${arm}_seed${seed}.json"
    local log="${RUN_LOG_DIR}/${arm}_seed${seed}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" \
        --arm "${arm}" \
        --seed "${seed}" \
        --device cuda \
        --output "${output}" \
        > "${log}" 2>&1
}

arms=(pure gate_only contraction_only full)
seeds=(17 29 43 59 71)
completed=0

for seed in "${seeds[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        run_one "${gpu}" "${arms[${gpu}]}" "${seed}" &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "seed wave ${seed} failed" >&2
        exit 1
    fi

    completed=$((completed + 4))
    echo "HEARTBEAT completed_units=${completed} seed=${seed}"
done

artifact_count=$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name '*_seed*.json' | wc -l)
if [[ "${artifact_count}" -ne 20 ]]; then
    echo "expected 20 run artifacts, found ${artifact_count}" >&2
    exit 1
fi

touch "${TASK_DIR}/DONE_${SLURM_JOB_ID}"
echo "DONE completed_units=${completed}"
