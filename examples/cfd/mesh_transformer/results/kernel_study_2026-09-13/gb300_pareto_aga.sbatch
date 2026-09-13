#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla-kernel-study
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/gb300_pareto_%j.log
#SBATCH -e sbatch_logs/gb300_pareto_%j.log
#SBATCH --open-mode=append

# ISLA kernel study (2026-09-13): the step-time reconciliation table on the same
# instrument the book's cost section was written on (one GB300, bf16 autocast,
# batch 1, AdamW step; see audit_mem/matched_memory.sbatch in 2026-08-09-mt2-stage0).
# Single GPU used (cuda:0); AGA's QOSMinGRES forces a whole-node (4 GPU) request.
# Code: this task dir's staged copy of the worktree's physicsnemo package (with the
# fused geo kernel) and study scripts; venv: the recipe venv (torch 2.12.0+cu130).
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-09-13-isla-kernel-study
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="${TASK}/inductor_cache"
export TRITON_CACHE_DIR="${TASK}/triton_cache"
cd "${TASK}"
echo "=== $(date -u +%FT%TZ) on $(hostname) job ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Overridable (sbatch exports the caller's environment): KS_OPTS, KS_CONFIGS, KS_OUT.
# Job 736338 (2026-09-13): the defaults below -> results/pareto_gb300.json.
# Job 2: KS_CONFIGS=10000:2:bf16,40000:2:bf16,10000:1:fp32,40000:1:fp32 KS_OUT=results/pareto_gb300_ext.json
OPTS="${KS_OPTS:-eager_ckpt_slow_softmax,eager_ckpt,eager_nockpt,compile_model_ckpt,compile_model_nockpt,fused_geo_eager,fused_geo_compile_model,geotransolver_eager,geotransolver_compiled}"
CONFIGS="${KS_CONFIGS:-10000:1:bf16,40000:1:bf16}"
OUT="${KS_OUT:-results/pareto_gb300.json}"
PYTHONPATH="${TASK}/code" timeout 3300 "${VENV}/bin/python" -u study/bench_pareto.py "${OUT}" "${OPTS}" --configs "${CONFIGS}"
echo "=== rc=$? $(date -u +%FT%TZ)"
