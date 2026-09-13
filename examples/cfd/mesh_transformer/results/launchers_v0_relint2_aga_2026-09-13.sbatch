#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J v0ri2
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%2
#SBATCH --dependency=singleton
# SUBMIT PER LANE so the initial job name equals the chain-ahead singleton name:
#   cd $TASK && for i in 0 1; do sbatch -J v0ri-$i --array=$i --dependency=singleton v0/v0_relint_aga.sbatch; done
#SBATCH -o sbatch_logs/v0ri2_%A_%a.log
#SBATCH -e sbatch_logs/v0ri2_%A_%a.log
#SBATCH --open-mode=append
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/code_relint"   # ISLA package at the 2026-09-13 repo head: relative frame + total-measure scale defaults, keyword-only API
RECIPE="${TASK}/recipe_relint"   # recipe copy with ComposeQuadratureMeasure and the drivaer_ml_volume_mt_relint dataset variant
TAG="v0ri2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SELF="${TASK}/v0/v0_relint2_aga.sbatch"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${CODE}:${RECIPE}/src"
trap "rc=\$?; printf \"rc=%s\\n\" \"\$rc\" > \"${TASK}/STATUS_${TAG}\"" EXIT
# RELINT2 (notebook #sec-nb-relint-prereg + #sec-nb-relint-void, 2026-09-13): relaunch with the
# Horvitz-Thompson-corrected boundary measure (ComposeQuadratureMeasure) on the 10,000-cell
# boundary subsample (drivaer_ml_volume_mt10k reader settings), so sum(w) = surface area.
# First attempt (relint) used raw subsample cell_areas: L 200x too small, model never learned.
# Original prereg text: the interior query-token +
# signed-distance configuration of v0_qt2_aga.sbatch (v0_isla_qtsdfval_seed42/43)
# with the reference frame instead of the similarity gauge: frame_mode=relative
# (no centroid anywhere; positions enter only as point-to-anchor differences),
# scale_mode=total_measure (length unit = sqrt of the total surface measure).
# Everything else frozen: 500 epochs, lr 1e-3, 10k boundary cells + 10k interior
# points, bf16, no augmentation, SDF-gradient query normals, SDF query scalar.
# Peter's ruling 2026-09-13: one flagship configuration; the interior must share
# the reference frame.
RES=10000
LR=1.0e-3
EPOCHS=500
DATASET=drivaer_ml_volume_mt_relint
MODEL=mt2_volume
EXTRA="forward_kwargs.measure_weights=boundaries.vehicle.cell_data.quadrature_measure +model.query_tokens=true +model.frame_mode=relative +model.scale_mode=total_measure +model.n_query_scalars=1 +forward_kwargs.query_normals=interior.point_data.sdf_normals +forward_kwargs.query_scalars=interior.point_data.sdf"
case "${SLURM_ARRAY_TASK_ID}" in
  0) SEED=42; RUN_ID="v0_isla_qtsdfval_relint2_seed42" ;;
  1) SEED=43; RUN_ID="v0_isla_qtsdfval_relint2_seed43" ;;
  2) SEED=44; RUN_ID="v0_isla_qtsdfval_relint2_seed44" ;;
  *) echo "bad array index" >&2; exit 2 ;;
esac
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
if [ -f "${TASK}/BLOCKED_${RUN_ID}" ]; then
  echo "lane ${RUN_ID} BLOCKED; no-op"; exit 0
fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then
  printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0
fi
### Chain-ahead: queue the NEXT link for this lane under a per-lane singleton
### name before training starts. Completed/blocked lanes exit above and do not chain.
cd "${TASK}" && sbatch -J "v0ri2-${SLURM_ARRAY_TASK_ID}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/v0ri2_resubmit.log" 2>&1 || true
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${TAG}.log" 2>&1 &
sampler_pid=$!
LOG_START=$(wc -l < "${RUN_DIR}/train.log" 2>/dev/null || echo 0)
set +e
srun --overlap "${VENV}/bin/torchrun" \
  --standalone --nproc-per-node 4 \
  "${RECIPE}/src/train.py" \
  model="${MODEL}" ${EXTRA} \
  dataset="${DATASET}" \
  sampling_resolution="${RES}" \
  run_id="${RUN_ID}" \
  output_dir="${TASK}/runs" \
  checkpoint_dir="${TASK}/runs" \
  training.seed="${SEED}" \
  training.num_epochs="${EPOCHS}" \
  training.optimizer.lr="${LR}" \
  training.save_interval=2 \
  +training.val_interval=10 \
  logging.log_every_n_steps=50 \
  precision=bfloat16 \
  augment=false \
  compile=false \
  dataloader.pin_memory=false \
  >> "${RUN_DIR}/train.log" 2>&1
rc=$?
set -e
kill "${sampler_pid}" 2>/dev/null || true
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then
  printf "DONE\n" > "${TASK}/RUN_${TAG}"
  exit 0
fi
if [ "${rc}" -ne 0 ]; then
  fp=$(tail -n +$((LOG_START+1)) "${RUN_DIR}/train.log" 2>/dev/null | grep -hE "step failed at epoch|Error executing job|^[A-Za-z_.]*Error" \
       | grep -v ChildFailedError | tail -1)
  prev=$(cat "${RUN_DIR}/.last_failure" 2>/dev/null || true)
  printf "%s\n" "${fp}" > "${RUN_DIR}/.last_failure"
  if [ -n "${fp}" ] && [ "${fp}" = "${prev}" ]; then
    printf "identical failure on consecutive segments: %s\n" "${fp}" > "${TASK}/BLOCKED_${RUN_ID}"
    exit "${rc}"
  fi
fi
exit "${rc}"
