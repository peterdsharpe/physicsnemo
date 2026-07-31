#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J shift-fixed-eval
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%2
#SBATCH --exclude=nvl72065-T07,nvl72087-T04
#SBATCH -o sbatch_logs/fixed_eval_%A_%a.log
#SBATCH -e sbatch_logs/fixed_eval_%A_%a.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
VENV=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-21-crossfam-hsg/repo/.venv-recipe
DATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV
PILOT_ROOT="${TASK}/pilot_runs"
EVAL_ROOT="${TASK}/fixed_validation"
MANIFEST="${TASK}/fixed_validation_manifest.tsv"
TASK_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "${EVAL_ROOT}/logs" "${TASK}/cache" "${TASK}/sbatch_logs"
[ -s "${MANIFEST}" ] || {
  printf "missing fixed-validation manifest: %s\n" "${MANIFEST}" >&2
  exit 2
}
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_FIXED_EVAL_${TASK_TAG}"' EXIT

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_fixed_eval_${TASK_TAG}.log" 2>&1 &
sampler_pid=$!

(
  sleep 1200
  while true; do
    now=$(date +%s)
    stalled=0
    for lane in 0 1 2 3; do
      marker="${TASK}/RUN_FIXED_EVAL_${TASK_TAG}_${lane}"
      if [ ! -f "${marker}" ]; then
        stalled=1
        continue
      fi
      log=$(<"${marker}")
      if [ "${log}" = "DONE" ]; then
        continue
      fi
      modified=$(stat -c %Y "${log}" 2>/dev/null || printf 0)
      if [ $((now - modified)) -gt 900 ]; then
        stalled=1
      fi
    done
    if [ "${stalled}" -ne 0 ]; then
      printf "stalled fixed-evaluation logs at %s\n" "$(date -Is)" \
        > "${TASK}/BLOCKED_FIXED_EVAL_${TASK_TAG}"
      scancel "${SLURM_JOB_ID}"
      exit 0
    fi
    sleep 300
  done
) &
watchdog_pid=$!

run_unit() {
  lane=$1
  unit_id=$2
  architecture=$3
  family=$4
  sample_count=$5
  run_id=$6
  completed_epoch=$7
  evaluation_seed=$8

  output_root="${EVAL_ROOT}/${run_id}"
  stem="epoch_${completed_epoch}_seed_${evaluation_seed}"
  output_path="${output_root}/${stem}.jsonl"
  invalid_path="${output_root}/${stem}.invalid.txt"
  log_path="${EVAL_ROOT}/logs/unit_${unit_id}_${stem}.log"
  mkdir -p "${output_root}"

  if [ -f "${output_path}" ] || [ -f "${invalid_path}" ]; then
    return 0
  fi
  printf "%s\n" "${log_path}" > "${TASK}/RUN_FIXED_EVAL_${TASK_TAG}_${lane}"

  if [ "${architecture}" = mesh_transformer ]; then
    model=mesh_transformer_surface_flagship
    code_root="${TASK}/code_source0520"
    recipe_root="${TASK}/recipe_source0520"
  else
    model=geotransolver_surface
    code_root="${TASK}/code"
    recipe_root="${TASK}/recipe"
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${lane}"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((30600 + lane))
    export SHIFT_SUV_FAMILY_ROOT="${DATA_ROOT}/${family}"
    export SHIFT_SUV_PILOT_MANIFEST="${TASK}/manifests/${family}_pilot_manifest.json"
    export TORCHINDUCTOR_CACHE_DIR="${TASK}/cache/torchinductor_fixed_${TASK_TAG}_${lane}"
    export TRITON_CACHE_DIR="${TASK}/cache/triton_fixed_${TASK_TAG}_${lane}"
    export PYTHONPATH="${TASK}/tensorboard_overlay:${code_root}:${recipe_root}/src"
    cd "${recipe_root}"
    "${VENV}/bin/python" src/fixed_eval.py \
      model="${model}" \
      dataset=shift_suv_adaptation_surface \
      train_split="train_${sample_count}" \
      val_split=val \
      run_id="${run_id}" \
      +checkpoint_path="${PILOT_ROOT}/${run_id}/checkpoints" \
      +evaluation_epoch="${completed_epoch}" \
      +evaluation_seed="${evaluation_seed}" \
      +evaluation_output="${output_path}" \
      training.seed="${evaluation_seed}" \
      sampling_resolution=10000 \
      precision=bfloat16 \
      augment=false \
      compile=false \
      logging.log_every_n_steps=25 \
      > "${log_path}" 2>&1
  )
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ] && [ -f "${output_path}" ]; then
    return 0
  fi
  if grep -Eq \
    "loss guard triggered|fixed evaluation metric guard triggered" \
    "${log_path}" 2>/dev/null; then
    grep -E \
      "loss guard triggered|fixed evaluation metric guard triggered" \
      "${log_path}" | tail -n 1 > "${invalid_path}"
    return 0
  fi

  printf "unit=%s run_id=%s epoch=%s seed=%s rc=%s\n" \
    "${unit_id}" "${run_id}" "${completed_epoch}" "${evaluation_seed}" "${rc}" \
    > "${TASK}/BLOCKED_FIXED_EVAL_UNIT_${unit_id}"
  return 1
}

run_lane() {
  lane=$1
  global_lane=$((4 * SLURM_ARRAY_TASK_ID + lane))

  while IFS=$'\t' read -r \
    unit_id assigned_lane architecture _initialization family sample_count \
    _learning_rate_name run_id completed_epoch evaluation_seed; do
    if [ "${assigned_lane}" != "${global_lane}" ]; then
      continue
    fi
    run_unit \
      "${lane}" \
      "${unit_id}" \
      "${architecture}" \
      "${family}" \
      "${sample_count}" \
      "${run_id}" \
      "${completed_epoch}" \
      "${evaluation_seed}" || return $?
  done < <(tail -n +2 "${MANIFEST}")

  printf "DONE\n" > "${TASK}/RUN_FIXED_EVAL_${TASK_TAG}_${lane}"
}

pids=()
for lane in 0 1 2 3; do
  run_lane "${lane}" &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

kill "${sampler_pid}" "${watchdog_pid}" 2>/dev/null || true
wait "${sampler_pid}" "${watchdog_pid}" 2>/dev/null || true

if [ "${failed}" -ne 0 ]; then
  printf "one or more fixed-evaluation lanes failed\n" \
    > "${TASK}/BLOCKED_FIXED_EVAL_${TASK_TAG}"
  exit 1
fi

printf "four fixed-evaluation lanes completed\n" \
  > "${TASK}/DONE_FIXED_EVAL_${TASK_TAG}"
