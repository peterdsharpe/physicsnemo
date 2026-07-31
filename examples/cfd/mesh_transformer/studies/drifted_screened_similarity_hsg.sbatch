#!/bin/bash
# Compare raw and similarity coordinates under pointwise and hybrid losses.
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=drift-similarity
#SBATCH --output=sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${TASK_DIR}/code"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
MODE="${MODE:-formal}"
PYTHON_BIN="${PYTHON_BIN:-/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-17-paper-training-ab/.venv/bin/python}"
STUDY="${CODE_DIR}/examples/cfd/mesh_transformer/studies/drifted_screened_similarity.py"
REDUCER="${CODE_DIR}/examples/cfd/mesh_transformer/studies/drifted_screened_similarity_reduce.py"

export PATH="/cm/local/apps/slurm/current/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/cfd/mesh_transformer:${PYTHONPATH:-}"
export UV_CACHE_DIR="${HOME}/coreai_modulus_cae/users/${USER}/.uvcache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

if [[ "${MODE}" == "pilot" ]]; then
    TASKS=(
        "raw_pointwise:17"
        "similarity_pointwise:17"
        "raw_hybrid:17"
        "similarity_hybrid:17"
    )
    EXTRA_ARGS=(
        --train-steps 500
        --evaluation-cases 2
        --evaluation-boundary-points 64
        --evaluation-query-points 128
        --resolution-cases 1
        --quadrature-order 16
        --check-quadrature-order 8
        --kernel-evaluation-pairs 8192
    )
elif [[ "${MODE}" == "formal" ]]; then
    TASKS=()
    for seed in 17 29 43 59 71; do
        for arm in \
            raw_pointwise similarity_pointwise raw_hybrid similarity_hybrid
        do
            TASKS+=("${arm}:${seed}")
        done
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

sampler &
SAMPLER_PID=$!

on_exit() {
    local rc=$?
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
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
    assert report["study"] == "drifted_screened_similarity_v1"
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
