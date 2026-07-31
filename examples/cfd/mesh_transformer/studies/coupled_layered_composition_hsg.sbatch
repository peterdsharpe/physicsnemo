#!/bin/bash
# Compare pooled prediction with sorted and physical coupled generators.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=coupled-composition
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
MODE="${MODE:-formal}"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/coupled_layered_composition.py"
REDUCER="${CODE_DIR}/examples/cfd/mesh_transformer/studies/coupled_layered_composition_reduce.py"

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

if [[ "${MODE}" == "pilot" ]]; then
    TASKS=(
        "analytic_path:17"
        "ordered_pool:17"
        "sorted_generator:17"
        "path_generator:17"
        "path_generator:29"
    )
    EXTRA_ARGS=(
        --train-steps 500
        --evaluation-profiles 32
        --evaluation-query-points 16
        --order-pairs 512
    )
elif [[ "${MODE}" == "formal" ]]; then
    TASKS=()
    for seed in 17 29 43 59 71; do
        for arm in ordered_pool sorted_generator path_generator; do
            TASKS+=("${arm}:${seed}")
        done
    done
    for seed in 17 29 43 59 71; do
        TASKS+=("analytic_path:${seed}")
    done
    EXTRA_ARGS=()
else
    echo "MODE must be pilot or formal, got ${MODE}" >&2
    exit 2
fi

OUTPUT_DIR="${ARTIFACT_DIR}/${MODE}"
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
    local log="${TASK_DIR}/sbatch_logs/coupled-composition.log"
    local previous_size=-1
    local stagnant_polls=0
    local current_size
    sleep 180
    while true; do
        current_size=$(stat -c %s "${log}" 2>/dev/null || echo 0)
        if [[ "${current_size}" == "${previous_size}" ]]; then
            stagnant_polls=$((stagnant_polls + 1))
        else
            stagnant_polls=0
        fi
        if ((stagnant_polls >= 3)); then
            echo "WATCHDOG stalled_log_bytes=${current_size}" >&2
            scontrol cancel "${SLURM_JOB_ID}"
            return
        fi
        previous_size="${current_size}"
        sleep 120
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
    echo "rc=${rc} mode=${MODE}" > "${TASK_DIR}/STATUS_${MODE}_${SLURM_JOB_ID}"
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
if [[ "${MODE}" == "formal" && ! -f "${REDUCER}" ]]; then
    echo "missing reducer source: ${REDUCER}" >&2
    exit 2
fi
if find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite nonempty output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ "${MODE}" == "formal" && -e "${ARTIFACT_DIR}/summary.json" ]]; then
    echo "refusing to overwrite existing summary" >&2
    exit 2
fi

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)'

worker() {
    local gpu=$1
    local index task arm seed output
    for ((index = gpu; index < ${#TASKS[@]}; index += 4)); do
        task="${TASKS[index]}"
        arm="${task%%:*}"
        seed="${task##*:}"
        output="${OUTPUT_DIR}/${arm}_seed${seed}.json"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" \
            --arm "${arm}" \
            --seed "${seed}" \
            --device cuda \
            --output "${output}" \
            "${EXTRA_ARGS[@]}"
        echo "HEARTBEAT mode=${MODE} gpu=${gpu} completed_task=${task}"
    done
}

PIDS=()
for gpu in 0 1 2 3; do
    worker "${gpu}" &
    PIDS+=("$!")
done

WORKER_RC=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        WORKER_RC=1
    fi
done
if ((WORKER_RC != 0)); then
    echo "one or more GPU workers failed" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${MODE}" "${#TASKS[@]}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
expected = int(sys.argv[3])
paths = sorted(root.glob("*.json"))
assert len(paths) == expected, (len(paths), expected)

def leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)

for path in paths:
    with path.open() as stream:
        report = json.load(stream)
    assert report["study"] == "coupled_layered_composition_v1"
    assert all(math.isfinite(value) for value in leaves(report))
print(f"validated mode={mode} reports={len(paths)}", flush=True)
PY

if [[ "${MODE}" == "formal" ]]; then
    "${PYTHON_BIN}" "${REDUCER}" \
        --input-dir "${OUTPUT_DIR}" \
        --output "${ARTIFACT_DIR}/summary.json"
fi

touch "${TASK_DIR}/DONE_${MODE}_${SLURM_JOB_ID}"
echo "DONE mode=${MODE} completed_units=${#TASKS[@]}"
