#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J shift-convergence
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%2
#SBATCH --exclude=nvl72065-T07,nvl72087-T04
#SBATCH -o sbatch_logs/convergence_%A_%a.log
#SBATCH -e sbatch_logs/convergence_%A_%a.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-30-shift-suv-convergence-extension
PARENT_TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
VENV=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-21-crossfam-hsg/repo/.venv-recipe
DATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV
RUN_ROOT="${TASK}/convergence_runs"
MANIFEST="${TASK}/training_manifest.tsv"
TASK_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p \
  "${RUN_ROOT}" \
  "${TASK}/cache" \
  "${TASK}/sbatch_logs" \
  "${TASK}/terminal_logs"
[ -s "${MANIFEST}" ] || {
  printf "missing convergence manifest: %s\n" "${MANIFEST}" >&2
  exit 2
}
if grep -q $'\r' "${MANIFEST}"; then
  printf "convergence manifest contains carriage returns\n" >&2
  exit 2
fi
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_CONVERGENCE_${TASK_TAG}"' EXIT

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_convergence_${TASK_TAG}.log" 2>&1 &
sampler_pid=$!

(
  sleep 1800
  while true; do
    now=$(date +%s)
    stalled=0
    for lane in 0 1 2 3; do
      marker="${TASK}/RUN_CONVERGENCE_${TASK_TAG}_${lane}"
      if [ ! -f "${marker}" ]; then
        stalled=1
        continue
      fi
      log=$(<"${marker}")
      if [ "${log}" = "DONE" ]; then
        continue
      fi
      modified=$(stat -c %Y "${log}" 2>/dev/null || printf 0)
      if [ $((now - modified)) -gt 1200 ]; then
        stalled=1
      fi
    done
    if [ "${stalled}" -ne 0 ]; then
      printf "stalled convergence logs at %s\n" "$(date -Is)" \
        > "${TASK}/BLOCKED_CONVERGENCE_${TASK_TAG}"
      scancel "${SLURM_JOB_ID}"
      exit 0
    fi
    sleep 300
  done
) &
watchdog_pid=$!

run_config() {
  lane=$1
  unit_id=$2
  architecture=$3
  family=$4
  role=$5
  sample_count=$6
  parent_checkpoint=$7
  parent_checkpoint_sha256=$8
  learning_rate=$9
  continuation_epochs=${10}
  scheduler_step_size=${11}
  run_id=${12}

  if [ "${architecture}" = mesh_transformer ]; then
    model=mesh_transformer_surface_flagship
    model_file=MeshTransformer.0.0.mdlus
    code_root="${PARENT_TASK}/code_source0520"
    recipe_root="${PARENT_TASK}/recipe_source0520"
  else
    model=geotransolver_surface
    model_file=GeoTransolver.0.0.mdlus
    code_root="${PARENT_TASK}/code"
    recipe_root="${PARENT_TASK}/recipe"
  fi

  run_root="${RUN_ROOT}/${run_id}"
  checkpoint_root="${run_root}/checkpoints"
  terminal_marker="${TASK}/TERMINAL_CANDIDATE_UNIT_${unit_id}"
  final_checkpoint="${checkpoint_root}/checkpoint.0.${continuation_epochs}.pt"
  mkdir -p "${checkpoint_root}"

  if [ -f "${terminal_marker}" ]; then
    return 0
  fi
  if [ -f "${final_checkpoint}" ] &&
    grep -q "Training completed!" "${run_root}/train.log" 2>/dev/null; then
    return 0
  fi

  printf "%s/train.log\n" "${run_root}" \
    > "${TASK}/RUN_CONVERGENCE_${TASK_TAG}_${lane}"

  existing_training_checkpoint=$(
    find "${checkpoint_root}" -maxdepth 1 -type f \
      -name 'checkpoint.0.*.pt' -print -quit
  )
  if [ -z "${existing_training_checkpoint}" ]; then
    if [ ! -f "${checkpoint_root}/${model_file}" ]; then
      cp "${parent_checkpoint}" "${checkpoint_root}/${model_file}"
    fi
    actual_sha256=$(sha256sum "${parent_checkpoint}" | awk '{print $1}')
    if [ "${actual_sha256}" != "${parent_checkpoint_sha256}" ]; then
      printf "parent checkpoint hash mismatch for %s\n" "${run_id}" >&2
      return 2
    fi
    copied_sha256=$(sha256sum "${checkpoint_root}/${model_file}" | awk '{print $1}')
    if [ "${copied_sha256}" != "${parent_checkpoint_sha256}" ]; then
      printf "copied checkpoint hash mismatch for %s\n" "${run_id}" >&2
      return 2
    fi
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${lane}"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((24000 + (SLURM_JOB_ID % 10000) * 4 + lane))
    export SHIFT_SUV_FAMILY_ROOT="${DATA_ROOT}/${family}"
    export SHIFT_SUV_PILOT_MANIFEST="${PARENT_TASK}/manifests/${family}_pilot_manifest.json"
    export TORCHINDUCTOR_CACHE_DIR="${TASK}/cache/torchinductor_${TASK_TAG}_${lane}"
    export TRITON_CACHE_DIR="${TASK}/cache/triton_${TASK_TAG}_${lane}"
    export PYTHONPATH="${PARENT_TASK}/tensorboard_overlay:${code_root}:${recipe_root}/src"
    cd "${recipe_root}"
    "${VENV}/bin/python" src/train.py \
      model="${model}" \
      dataset=shift_suv_adaptation_surface \
      train_split="train_${sample_count}" \
      val_split=val \
      run_id="${run_id}" \
      output_dir="${RUN_ROOT}" \
      checkpoint_dir="${RUN_ROOT}" \
      training.seed=42 \
      training.num_epochs="${continuation_epochs}" \
      training.optimizer.lr="${learning_rate}" \
      training.scheduler.step_size="${scheduler_step_size}" \
      training.scheduler.gamma=0.1 \
      training.save_interval=5 \
      logging.log_every_n_steps=25 \
      sampling_resolution=10000 \
      precision=bfloat16 \
      augment=false \
      compile=false \
      > "${run_root}/train.log" 2>&1
  )
  rc=$?
  set -e

  if [ -f "${final_checkpoint}" ] &&
    grep -q "Training completed!" "${run_root}/train.log" 2>/dev/null; then
    return 0
  fi
  if grep -q "loss guard triggered" "${run_root}/train.log" 2>/dev/null; then
    cp "${run_root}/train.log" \
      "${TASK}/terminal_logs/unit_${unit_id}_${run_id}.log"
    printf "unit=%s run_id=%s role=%s\n" \
      "${unit_id}" "${run_id}" "${role}" > "${terminal_marker}"
    return 0
  fi

  printf "unit=%s run_id=%s rc=%s\n" "${unit_id}" "${run_id}" "${rc}" \
    > "${TASK}/BLOCKED_CONVERGENCE_UNIT_${unit_id}"
  return 1
}

run_lane() {
  lane=$1
  global_lane=$((4 * SLURM_ARRAY_TASK_ID + lane))

  while IFS=$'\t' read -r \
    unit_id assigned_lane architecture family role _initialization sample_count \
    _parent_run_id _parent_completed_epoch parent_checkpoint \
    parent_checkpoint_sha256 _learning_rate_name learning_rate \
    continuation_epochs scheduler_step_size run_id; do
    if [ "${assigned_lane}" != "${global_lane}" ]; then
      continue
    fi
    run_config \
      "${lane}" \
      "${unit_id}" \
      "${architecture}" \
      "${family}" \
      "${role}" \
      "${sample_count}" \
      "${parent_checkpoint}" \
      "${parent_checkpoint_sha256}" \
      "${learning_rate}" \
      "${continuation_epochs}" \
      "${scheduler_step_size}" \
      "${run_id}" || return $?
  done < <(tail -n +2 "${MANIFEST}")

  printf "DONE\n" > "${TASK}/RUN_CONVERGENCE_${TASK_TAG}_${lane}"
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
  printf "one or more convergence lanes failed\n" \
    > "${TASK}/BLOCKED_CONVERGENCE_${TASK_TAG}"
  exit 1
fi

printf "four convergence lanes completed\n" \
  > "${TASK}/DONE_CONVERGENCE_${TASK_TAG}"
