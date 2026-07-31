#!/bin/bash
# Test connection-aware moving spectral subspaces on nonseparable fields.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=nonsep-adapt
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
MODE="${MODE:-pilot}"
OUTPUT_DIR="${ARTIFACT_DIR}/${MODE}"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/nonseparable_adaptive_subspace.py"
REDUCER="${CODE_DIR}/examples/cfd/mesh_transformer/studies/nonseparable_adaptive_subspace_reduce.py"
FORMAL_PROFILE_SEEDS=(263000239 269000241 271000247 277000251)

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

if [[ "${MODE}" == "pilot" ]]; then
    PROFILE_SEEDS=(257000229)
    RUN_ARGS=(
        --evaluation-profiles 32
        --evaluation-query-points 12
    )
elif [[ "${MODE}" == "formal" ]]; then
    PROFILE_SEEDS=("${FORMAL_PROFILE_SEEDS[@]}")
    RUN_ARGS=()
else
    echo "MODE must be pilot or formal, got ${MODE}" >&2
    exit 2
fi
EXPECTED_UNITS=${#PROFILE_SEEDS[@]}

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
    sleep 300
    while true; do
        current_count=$(find "${OUTPUT_DIR}" -name 'profile_seed*.json' -type f | wc -l)
        echo "PROGRESS completed_units=${current_count} expected_units=${EXPECTED_UNITS}"
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
    echo "rc=${rc} mode=${MODE}" > "${TASK_DIR}/STATUS_${MODE}_${SLURM_JOB_ID}"
}
trap on_exit EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${STUDY}" || ("${MODE}" == "formal" && ! -f "${REDUCER}") ]]; then
    echo "missing staged study or formal reducer" >&2
    exit 2
fi
if find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite nonempty output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ "${MODE}" == "formal" && -e "${ARTIFACT_DIR}/summary.json" ]]; then
    echo "refusing to overwrite existing formal summary" >&2
    exit 2
fi

"${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)'

worker() {
    local gpu=$1
    local index seed
    for ((index = gpu; index < ${#PROFILE_SEEDS[@]}; index += 4)); do
        seed="${PROFILE_SEEDS[index]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${STUDY}" \
            --profile-seed "${seed}" \
            --device cuda \
            --output "${OUTPUT_DIR}/profile_seed${seed}.json" \
            "${RUN_ARGS[@]}" \
            > "${OUTPUT_DIR}/profile_seed${seed}.stdout.log"
        echo "HEARTBEAT completed_unit=profile_seed${seed}"
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
    echo "one or more census workers failed" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${EXPECTED_UNITS}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
paths = sorted(root.glob("profile_seed*.json"))
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
    assert report["study"] == "nonseparable_adaptive_subspace_v1"
    assert all(math.isfinite(value) for value in leaves(report))
print(f"validated reports={len(paths)}", flush=True)
PY

if [[ "${MODE}" == "formal" ]]; then
    "${PYTHON_BIN}" "${REDUCER}" \
        --input-dir "${OUTPUT_DIR}" \
        --output "${ARTIFACT_DIR}/summary.json"
fi

touch "${TASK_DIR}/DONE_${MODE}_${SLURM_JOB_ID}"
echo "DONE mode=${MODE} completed_units=${EXPECTED_UNITS}"

