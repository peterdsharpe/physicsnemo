#!/bin/bash
# Confirm whether a rank-four correction can resolve nonseparable mode coupling.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=nonseparable-rank
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
OUTPUT_DIR="${ARTIFACT_DIR}/formal"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/nonseparable_rank_census.py"
REDUCER="${CODE_DIR}/examples/cfd/mesh_transformer/studies/nonseparable_rank_census_reduce.py"
SEEDS=(991081447 991081449 991081451 991081453)

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${OUTPUT_DIR}" "${TASK_DIR}/sbatch_logs"

sampler() {
    while true; do
        date --iso-8601=seconds
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,memory.used \
            --format=csv,noheader
        sleep 120
    done
}

watchdog() {
    local previous_count=-1
    local stagnant_polls=0
    local current_count
    sleep 600
    while true; do
        current_count=$(find "${OUTPUT_DIR}" -name 'seed*.json' -type f | wc -l)
        echo "PROGRESS completed_units=${current_count} expected_units=${#SEEDS[@]}"
        if [[ "${current_count}" == "${previous_count}" ]]; then
            stagnant_polls=$((stagnant_polls + 1))
        else
            stagnant_polls=0
        fi
        if ((stagnant_polls >= 6)); then
            echo "WATCHDOG completed_units_stalled=${current_count}" >&2
            scontrol cancel "${SLURM_JOB_ID}"
            return
        fi
        previous_count="${current_count}"
        sleep 300
    done
}

sampler &
SAMPLER_PID=$!
watchdog &
WATCHDOG_PID=$!

on_exit() {
    local rc=$?
    kill "${SAMPLER_PID}" "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" "${WATCHDOG_PID}" 2>/dev/null || true
    echo "rc=${rc}" > "${TASK_DIR}/STATUS_${SLURM_JOB_ID}"
}
trap on_exit EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${STUDY}" || ! -f "${REDUCER}" ]]; then
    echo "missing staged study or reducer" >&2
    exit 2
fi
if find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite nonempty output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ -e "${ARTIFACT_DIR}/summary.json" ]]; then
    echo "refusing to overwrite existing summary" >&2
    exit 2
fi

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)'

PIDS=()
for gpu in 0 1 2 3; do
    seed="${SEEDS[gpu]}"
    output="${OUTPUT_DIR}/seed${seed}.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" \
        --device cuda \
        --profiles 256 \
        --query-points 32 \
        --seed "${seed}" \
        --output "${output}" \
        > "${OUTPUT_DIR}/seed${seed}.stdout.log" &
    PIDS+=("$!")
done

WORKER_RC=0
for index in 0 1 2 3; do
    if wait "${PIDS[index]}"; then
        echo "HEARTBEAT completed_unit=seed${SEEDS[index]}"
    else
        WORKER_RC=1
    fi
done
if ((WORKER_RC != 0)); then
    echo "one or more census workers failed" >&2
    exit 1
fi

"${PYTHON_BIN}" "${REDUCER}" \
    --input-dir "${OUTPUT_DIR}" \
    --output "${ARTIFACT_DIR}/summary.json"

touch "${TASK_DIR}/DONE_${SLURM_JOB_ID}"
echo "DONE completed_units=${#SEEDS[@]}"
