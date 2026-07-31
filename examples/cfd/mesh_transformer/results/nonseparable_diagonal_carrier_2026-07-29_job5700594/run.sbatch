#!/bin/bash
# Run the diagonal-carrier capacity and target-saturation matrix.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=00:30:00
#SBATCH --job-name=nonsep-carry
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
OUTPUT_DIR="${TASK_DIR}/artifacts/formal"
SUMMARY="${TASK_DIR}/artifacts/summary.json"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/nonseparable_diagonal_carrier.py"
EXPECTED_UNITS=15

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}/examples/cfd/mesh_transformer/studies:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${OUTPUT_DIR}" "${TASK_DIR}/sbatch_logs"
if find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite nonempty output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ -e "${SUMMARY}" ]]; then
    echo "refusing to overwrite existing summary: ${SUMMARY}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -f "${STUDY}" ]]; then
    echo "missing Python environment or staged study" >&2
    exit 2
fi

sampler() {
    while true; do
        date --iso-8601=seconds
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,memory.used \
            --format=csv,noheader
        sleep 60
    done
}

watchdog() {
    local previous_count=-1
    local stagnant_polls=0
    local current_count
    sleep 120
    while true; do
        current_count=$(find "${OUTPUT_DIR}" -name '*.json' -type f | wc -l)
        echo "PROGRESS completed_units=${current_count} expected_units=${EXPECTED_UNITS}"
        if [[ "${current_count}" == "${previous_count}" ]]; then
            stagnant_polls=$((stagnant_polls + 1))
        else
            stagnant_polls=0
        fi
        if ((stagnant_polls >= 4)); then
            echo "WATCHDOG completed_units_stalled=${current_count}" >&2
            scontrol cancel "${SLURM_JOB_ID}"
            return
        fi
        previous_count="${current_count}"
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
    echo "rc=${rc}" > "${TASK_DIR}/STATUS_${SLURM_JOB_ID}"
}
trap on_exit EXIT

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)'

TASKS=(
    "capacity:0:17" "capacity:0:29" "capacity:0:43"
    "target:1:17" "target:1:29" "target:1:43"
    "target:2:17" "target:2:29" "target:2:43"
    "target:4:17" "target:4:29" "target:4:43"
    "target:8:17" "target:8:29" "target:8:43"
)

worker() {
    local gpu=$1
    local index spec arm budget seed output
    for ((index = gpu; index < ${#TASKS[@]}; index += 4)); do
        spec="${TASKS[index]}"
        IFS=: read -r arm budget seed <<< "${spec}"
        if [[ "${arm}" == "capacity" ]]; then
            output="${OUTPUT_DIR}/capacity_seed${seed}.json"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" run \
                --arm capacity \
                --seed "${seed}" \
                --device cuda \
                --output "${output}" \
                > "${output%.json}.stdout.log"
        else
            output="${OUTPUT_DIR}/target_n${budget}_seed${seed}.json"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" run \
                --arm target \
                --budget "${budget}" \
                --seed "${seed}" \
                --device cuda \
                --output "${output}" \
                > "${output%.json}.stdout.log"
        fi
        echo "HEARTBEAT completed_unit=${arm}_n${budget}_seed${seed}"
    done
}

PIDS=()
for gpu in 0 1 2 3; do
    worker "${gpu}" &
    PIDS+=("$!")
done

WORKER_RC=0
for index in 0 1 2 3; do
    if ! wait "${PIDS[index]}"; then
        WORKER_RC=1
    fi
done
if ((WORKER_RC != 0)); then
    echo "one or more carrier workers failed" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${EXPECTED_UNITS}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
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
    report = json.loads(path.read_text())
    assert report["study"] == "nonseparable_diagonal_carrier_v1"
    assert all(math.isfinite(value) for value in leaves(report))
print(f"validated reports={len(paths)}", flush=True)
PY

"${PYTHON_BIN}" "${STUDY}" reduce \
    --input-dir "${OUTPUT_DIR}" \
    --output "${SUMMARY}"

touch "${TASK_DIR}/DONE_${SLURM_JOB_ID}"
echo "DONE completed_units=${EXPECTED_UNITS}"
