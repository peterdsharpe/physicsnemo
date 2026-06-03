#!/usr/bin/env bash
# Launch scaling.py with one process per visible GPU via torchrun.
#
# Why a wrapper?  scaling.py knows how to fan a (theta, N) sweep across
# torchrun ranks via DistributedManager, but only torchrun sets the env
# vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR/PORT) the manager
# expects.  Running the Python script directly inside an interactive
# SLURM allocation would otherwise let the manager pick up SLURM_*
# variables and try to bring up a process group with non-existent
# peers - which hangs for 10 minutes on the TCPStore connect.
#
# Usage:
#   ./run.sh                 # one rank per visible GPU
#   NPROC_PER_NODE=2 ./run.sh # override; e.g. force 2 ranks
#   ./run.sh --some-arg ...  # extra args forwarded to scaling.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

### [Detect GPU count]
# Prefer CUDA_VISIBLE_DEVICES (it's what torch.cuda.device_count() honours);
# fall back to nvidia-smi; default to 1 on CPU-only hosts.
if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    NUM_GPUS=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | grep -c .)
elif command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    NUM_GPUS=0
fi
[[ "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=1

NPROC="${NPROC_PER_NODE:-${NUM_GPUS}}"
echo "Launching scaling.py on ${NPROC} rank(s) (detected ${NUM_GPUS} visible GPU(s))"

### [CUDA Allocator]
# scaling.py also sets this internally before any torch import; we set
# it here too so torchrun's worker processes inherit it cleanly.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

### [Launch]
# --standalone uses a free local port, so this works whether or not the
# host shell is inside a SLURM allocation.
exec uv run --no-sync torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC}" \
    ./scaling.py "$@"
