#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J fp32-v0ri2-eval
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-0%1
#SBATCH -o sbatch_logs/fp32_v0ri2eval_%A_%a.log
#SBATCH -e sbatch_logs/fp32_v0ri2eval_%A_%a.log
#SBATCH --open-mode=append

# Full V0 evals (PREPARED, NOT SUBMITTED by the launch agent): infer.py on the
# DrivAerML volume val split (48 cases, manifest val) for each of the 8 full
# runs (4 arms x seeds {42, 43}), at the training resolution (10k boundary
# cells + 10k interior points), with the same model overrides used at training.
# One run per GPU, four per node -> array task 0 = seed-42 runs, task 1 =
# seed-43 runs. Idempotent: a run with <OUTDIR>/.done is skipped; a run whose
# train.log lacks "Training completed" is a no-op (resubmit later).
# Output: ${TASK}/v0_evals_fp32/<run>/  (infer.py writes <output_dir>/<run_id>/
# predictions/*.pdmsh + metrics.jsonl; log at ${TASK}/v0_evals_fp32/<run>.log).
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${TASK}/code_relint:${TASK}/recipe/src"  # RELINT lanes trained under code_relint; evaluate under the same snapshot
trap "rc=\$?; printf \"rc=%s\\n\" \"\$rc\" > \"${TASK}/STATUS_FP32V0RI2EVAL_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}\"" EXIT

ALL=(v0_isla_qtsdfval_relint2_seed42 v0_isla_qtsdfval_relint2_seed43)
MT2_FLAGS="+model.query_independent=true +model.n_decoder_layers=2 +model.interior_queries=true +model.similarity_gauge=true +model.local_readout_rho=0.1"
unit() {
  local RUN="$1" GPU="$2"
  local OUTDIR="${TASK}/v0_evals_fp32_relint2/${RUN}"
  [ -f "${OUTDIR}/.done" ] && return 0
  grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" 2>/dev/null || { echo "${RUN} not trained"; return 0; }
  case "$RUN" in
    v0_gt_vol_seed*)      MODEL=geotransolver_volume;    EXTRA="" ;;
    v0_mt1k_vol_seed*)    MODEL=mesh_transformer_volume; EXTRA="" ;;
    v0_mt2_int_seed*)     MODEL=mt2_volume;              EXTRA="${MT2_FLAGS}" ;;
    v0_mt2_int_sdf_seed*) MODEL=mt2_volume;              EXTRA="${MT2_FLAGS} +forward_kwargs.query_normals=interior.point_data.sdf_normals" ;;
    *isla_qtsdfval_relint2_seed*) MODEL=mt2_volume;      EXTRA="forward_kwargs.measure_weights=boundaries.vehicle.cell_data.quadrature_measure +model.query_tokens=true +model.frame_mode=relative +model.scale_mode=total_measure +model.n_query_scalars=1 +forward_kwargs.query_normals=interior.point_data.sdf_normals +forward_kwargs.query_scalars=interior.point_data.sdf" ;;
    *isla_qt_sdf_seed*)   MODEL=mt2_volume;              EXTRA="+model.query_tokens=true +model.similarity_gauge=true +forward_kwargs.query_normals=interior.point_data.sdf_normals" ;;
    *) echo "unknown run ${RUN}" >&2; return 1 ;;
  esac
  mkdir -p "${OUTDIR}"
  timeout 4800 env CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((32700 + GPU)) \
    "${VENV}/bin/python" "${TASK}/recipe_relint/src/infer.py" \
    model="${MODEL}" ${EXTRA} dataset=drivaer_ml_volume_mt_relint run_id="${RUN}" infer_split=val \
    sampling_resolution=10000 checkpoint_dir="${TASK}/runs" output_dir="${OUTDIR}" precision=float32 dataloader.pin_memory=false \
    force_coefficients.enabled=false \
    >> "${OUTDIR}.log" 2>&1 && touch "${OUTDIR}/.done" || echo "=== ${RUN} FAILED rc=$?" >> "${OUTDIR}.log"
}
mkdir -p "${TASK}/v0_evals"; cd "${TASK}/recipe"
pids=()
for g in 0 1 2 3; do
  idx=$(( SLURM_ARRAY_TASK_ID * 4 + g ))
  [ "$idx" -lt "${#ALL[@]}" ] || continue
  ( unit "${ALL[$idx]}" "$g" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "V0RI2EVAL-FP32-TASK-DONE ${SLURM_ARRAY_TASK_ID}"
