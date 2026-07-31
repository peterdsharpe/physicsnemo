#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J shift-adapt-pool
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%2
#SBATCH --exclude=nvl72065-T07,nvl72087-T04
#SBATCH -o sbatch_logs/pool_%A_%a.log
#SBATCH -e sbatch_logs/pool_%A_%a.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
VENV=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-21-crossfam-hsg/repo/.venv-recipe
DATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV
PILOT_ROOT="${TASK}/pilot_runs"
TASK_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p \
  "${PILOT_ROOT}" \
  "${TASK}/cache" \
  "${TASK}/sbatch_logs" \
  "${TASK}/terminal_logs"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_POOL_${TASK_TAG}"' EXIT

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_pool_${TASK_TAG}.log" 2>&1 &
sampler_pid=$!

(
  sleep 1800
  while true; do
    now=$(date +%s)
    stalled=0
    for lane in 0 1 2 3; do
      marker="${TASK}/RUN_POOL_${TASK_TAG}_${lane}"
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
      printf "stalled pool logs at %s\n" "$(date -Is)" \
        > "${TASK}/BLOCKED_POOL_${TASK_TAG}"
      scancel "${SLURM_JOB_ID}"
      exit 0
    fi
    sleep 300
  done
) &
watchdog_pid=$!

run_config() {
  lane=$1
  flat_index=$2

  learning_rates=(0.0003 0.001 0.003)
  learning_rate_names=(3e-4 1e-3 3e-3)
  sample_counts=(64 128)
  families=(estate fastback)
  initializations=(scratch pretrained)
  architectures=(mesh_transformer geotransolver)

  lr_index=$((flat_index % 3))
  reduced_index=$((flat_index / 3))
  sample_index=$((reduced_index % 2))
  reduced_index=$((reduced_index / 2))
  family_index=$((reduced_index % 2))
  reduced_index=$((reduced_index / 2))
  init_index=$((reduced_index % 2))
  reduced_index=$((reduced_index / 2))
  architecture_index=$((reduced_index % 2))

  learning_rate=${learning_rates[$lr_index]}
  learning_rate_name=${learning_rate_names[$lr_index]}
  sample_count=${sample_counts[$sample_index]}
  family=${families[$family_index]}
  initialization=${initializations[$init_index]}
  architecture=${architectures[$architecture_index]}

  if [ "${architecture}" = mesh_transformer ]; then
    model=mesh_transformer_surface_flagship
    model_file=MeshTransformer.0.0.mdlus
    source_dir=mesh_transformer
    code_root="${TASK}/code_source0520"
    recipe_root="${TASK}/recipe_source0520"
  else
    model=geotransolver_surface
    model_file=GeoTransolver.0.0.mdlus
    source_dir=geotransolver
    code_root="${TASK}/code"
    recipe_root="${TASK}/recipe"
  fi

  run_id="pilot_${architecture}_${initialization}_${family}_n${sample_count}_lr${learning_rate_name}_seed42"
  run_root="${PILOT_ROOT}/${run_id}"
  checkpoint_root="${run_root}/checkpoints"
  terminal_marker="${TASK}/TERMINAL_CANDIDATE_CONFIG_${flat_index}"
  mkdir -p "${checkpoint_root}"

  if [ -f "${terminal_marker}" ]; then
    return 0
  fi
  if grep -q "Training completed!" "${run_root}/train.log" 2>/dev/null; then
    return 0
  fi

  printf "%s/train.log\n" "${run_root}" \
    > "${TASK}/RUN_POOL_${TASK_TAG}_${lane}"

  if [ "${initialization}" = pretrained ] &&
    [ ! -f "${checkpoint_root}/${model_file}" ]; then
    cp "${TASK}/source_checkpoints/${source_dir}/${model_file}" \
      "${checkpoint_root}/"
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${lane}"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000) * 4 + lane))
    export SHIFT_SUV_FAMILY_ROOT="${DATA_ROOT}/${family}"
    export SHIFT_SUV_PILOT_MANIFEST="${TASK}/manifests/${family}_pilot_manifest.json"
    export TORCHINDUCTOR_CACHE_DIR="${TASK}/cache/torchinductor_pool_${TASK_TAG}_${lane}"
    export TRITON_CACHE_DIR="${TASK}/cache/triton_pool_${TASK_TAG}_${lane}"
    export PYTHONPATH="${TASK}/tensorboard_overlay:${code_root}:${recipe_root}/src"
    cd "${recipe_root}"
    "${VENV}/bin/python" src/train.py \
      model="${model}" \
      dataset=shift_suv_adaptation_surface \
      train_split="train_${sample_count}" \
      val_split=val \
      run_id="${run_id}" \
      output_dir="${PILOT_ROOT}" \
      checkpoint_dir="${PILOT_ROOT}" \
      training.seed=42 \
      training.num_epochs=70 \
      training.optimizer.lr="${learning_rate}" \
      training.scheduler.step_size=35 \
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

  if grep -q "Training completed!" "${run_root}/train.log" 2>/dev/null; then
    return 0
  fi
  if grep -q "loss guard triggered" "${run_root}/train.log" 2>/dev/null; then
    cp "${run_root}/train.log" \
      "${TASK}/terminal_logs/config_${flat_index}_${run_id}.log"
    printf "%s\n" "${run_id}" > "${terminal_marker}"
    return 0
  fi

  printf "config=%s run_id=%s rc=%s\n" \
    "${flat_index}" "${run_id}" "${rc}" \
    > "${TASK}/BLOCKED_POOL_CONFIG_${flat_index}"
  return 1
}

run_lane() {
  lane=$1
  global_lane=$((4 * SLURM_ARRAY_TASK_ID + lane))

  case "${global_lane}" in
    0) configs=(9 45 13 6 34 35) ;;
    1) configs=(1 19 31 18 7 26) ;;
    2) configs=(0 5 40 3 38 25) ;;
    3) configs=(16 15 27 24 42) ;;
    4) configs=(22 8 39 28 36 41) ;;
    5) configs=(21 11 46 32 44) ;;
    6) configs=(12 17 33 2 29) ;;
    7) configs=(4 43 10 47 30 37) ;;
    *)
      printf "invalid global lane %s\n" "${global_lane}" >&2
      return 2
      ;;
  esac

  for flat_index in "${configs[@]}"; do
    run_config "${lane}" "${flat_index}" || return $?
  done
  printf "DONE\n" > "${TASK}/RUN_POOL_${TASK_TAG}_${lane}"
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
  printf "one or more pool lanes failed\n" > "${TASK}/BLOCKED_POOL_${TASK_TAG}"
  exit 1
fi

printf "four pool lanes completed\n" > "${TASK}/DONE_POOL_${TASK_TAG}"
