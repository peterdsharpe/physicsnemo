#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla-kernel-study
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/gb300_study_%j.log
#SBATCH -e sbatch_logs/gb300_study_%j.log
#SBATCH --open-mode=append

# ISLA kernel study (2026-09-13): the complete measurement set on the target
# instrument (GB300), one node, four independent lanes on the four GPUs:
#   GPU0  Pareto matrix, bf16 configs   -> results/pareto_gb300_full_bf16.json
#   GPU1  Pareto matrix, fp32 configs   -> results/pareto_gb300_full_fp32.json
#   GPU2  roofline (peaks, analytic, achieved per stage) -> results/roofline_gb300.json
#         then the fused-kernel launch sweep            -> results/tune_fused_kernel_gb300.json
#   GPU3  torch.profiler tables (eager / compiled / fused at 10k and 40k) -> results/profile_*_gb300.json
#         then the numerics gate for the fused options   -> results/numerics_fused_10k_gb300.json
# Each lane has its own Inductor/Triton cache directory so concurrent compiles do not race.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-09-13-isla-kernel-study
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export PYTHONPATH="${TASK}/code"
cd "${TASK}"
mkdir -p results lane_logs
echo "=== $(date -u +%FT%TZ) on $(hostname) job ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
PY="${VENV}/bin/python"
S=study
OPTS=eager_ckpt,eager_ckpt_slow_softmax,eager_nockpt,compile_model_ckpt,compile_model_nockpt,compile_blocks_ckpt,compile_model_dynamic,compile_model_maxautotune,compile_model_cudagraphs,fused_geo_eager,fused_geo_compile_model,geotransolver_eager,geotransolver_compiled

lane() {  # lane <gpu> <log-name> <command...>
  local gpu="$1" name="$2"; shift 2
  ( export CUDA_VISIBLE_DEVICES="${gpu}" TORCHINDUCTOR_CACHE_DIR="${TASK}/inductor_cache_${gpu}" TRITON_CACHE_DIR="${TASK}/triton_cache_${gpu}"
    echo "=== lane ${name} START $(date -u +%T) gpu ${gpu}"
    "$@"
    echo "=== lane ${name} END rc=$? $(date -u +%T)" ) >> "lane_logs/${name}_${SLURM_JOB_ID}.log" 2>&1
}

lane 0 pareto_bf16 timeout 6600 "$PY" -u $S/bench_pareto.py results/pareto_gb300_full_bf16.json "$OPTS" --configs 10000:1:bf16,10000:2:bf16,40000:1:bf16,40000:2:bf16 &
lane 1 pareto_fp32 timeout 6600 "$PY" -u $S/bench_pareto.py results/pareto_gb300_full_fp32.json "$OPTS" --configs 10000:1:fp32,10000:2:fp32,40000:1:fp32,40000:2:fp32 &
( lane 2 roofline timeout 3000 "$PY" -u $S/roofline.py results/roofline_gb300.json
  lane 2 tune timeout 1800 "$PY" -u $S/tune_fused_kernel.py results/tune_fused_kernel_gb300.json ) &
( for spec in "eager_ckpt 10000" "compile_model_ckpt 10000" "compile_model_nockpt 10000" "fused_geo_eager 10000" "fused_geo_compile_model 10000" "eager_ckpt 40000" "compile_model_nockpt 40000" "fused_geo_compile_model 40000"; do
    set -- $spec
    lane 3 "profile_$1_$2" timeout 900 "$PY" -u $S/profile_kernels.py "results/profile_$1_$2_bf16_gb300.json" "$1" "$2" 1 bf16
  done
  lane 3 numerics_fused timeout 2400 "$PY" -u $S/numerics_check.py results/numerics_fused_10k_gb300.json 10000 1 fused_geo_eager,fused_geo_compile_model ) &
wait
echo "=== all lanes done $(date -u +%FT%TZ)"
grep -h "=== lane" lane_logs/*_${SLURM_JOB_ID}.log
echo "=== rc=0 $(date -u +%FT%TZ)"
