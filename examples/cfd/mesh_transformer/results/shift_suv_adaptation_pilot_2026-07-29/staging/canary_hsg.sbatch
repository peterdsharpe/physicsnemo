#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J shift-adapt-canary
#SBATCH --time=1:00:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=nvl72087-T04
#SBATCH -o sbatch_logs/%x.log
#SBATCH -e sbatch_logs/%x.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -s 8192

TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
VENV=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-21-crossfam-hsg/repo/.venv-recipe
DATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV
DRIVAER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML
CANARY_ROOT="${TASK}/artifacts/canary_${SLURM_JOB_ID}"
CODE_ROOT="${TASK}/code"
RECIPE_ROOT="${TASK}/recipe"
if [ "${MODEL_CODE_VARIANT:-current}" = source0520 ]; then
  CODE_ROOT="${TASK}/code_source0520"
  RECIPE_ROOT="${TASK}/recipe_source0520"
fi

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export PYTHONPATH="${TASK}/tensorboard_overlay:${CODE_ROOT}:${RECIPE_ROOT}/src"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${CANARY_ROOT}" "${TASK}/cache" "${TASK}/sbatch_logs"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${SLURM_JOB_ID}"' EXIT

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${SLURM_JOB_ID}.log" 2>&1 &
sampler_pid=$!

(
  sleep 2400
  while true; do
    newest=$(find "${CANARY_ROOT}" -name train.log -printf '%T@\n' |
      sort -nr | head -n 1)
    now=$(date +%s)
    if [ -z "${newest}" ] || [ $((now - ${newest%.*})) -gt 900 ]; then
      printf "stalled canary logs at %s\n" "$(date -Is)" \
        > "${TASK}/BLOCKED_CANARY_${SLURM_JOB_ID}"
      scancel "${SLURM_JOB_ID}"
      exit 0
    fi
    sleep 180
  done
) &
watchdog_pid=$!

run_lane() {
  lane=$1
  model=$2
  init=$3
  family=$4
  precision=${5:-bfloat16}
  learning_rate=${6:-0.003}
  sampling_resolution=${7:-${SAMPLING_RESOLUTION:-10000}}
  requested_train_split=${8:-canary_train}
  profile_mode=${9:-false}
  compile_mode=${10:-${COMPILE_MODE:-true}}
  learning_rate_tag=${learning_rate//./p}
  run_id="canary_${model}_${init}_${family}_${precision}_lr${learning_rate_tag}_res${sampling_resolution}_${requested_train_split}_compile${compile_mode}"
  run_root="${CANARY_ROOT}/${run_id}"
  checkpoint_root="${run_root}/${run_id}/checkpoints"
  mkdir -p "${checkpoint_root}"

  if [ "${init}" = pretrained ]; then
    if [ "${model}" = mesh_transformer_surface_flagship ]; then
      cp "${TASK}/source_checkpoints/mesh_transformer/MeshTransformer.0.0.mdlus" \
        "${checkpoint_root}/"
    else
      cp "${TASK}/source_checkpoints/geotransolver/GeoTransolver.0.0.mdlus" \
        "${checkpoint_root}/"
    fi
  fi

  (
    export CUDA_VISIBLE_DEVICES="${lane}"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((29500 + lane))
    if [ "${family}" = drivaer ]; then
      dataset_name=drivaer_source_intrinsic_surface
      train_split="${requested_train_split}"
      val_split=canary_val
      export DRIVAER_SOURCE_ROOT="${DRIVAER_ROOT}"
      export DRIVAER_SOURCE_MANIFEST="${TASK}/manifests/drivaer_source_canary_manifest.json"
    else
      dataset_name=shift_suv_adaptation_surface
      train_split="${requested_train_split}"
      val_split=canary_val
      export SHIFT_SUV_FAMILY_ROOT="${DATA_ROOT}/${family}"
      export SHIFT_SUV_PILOT_MANIFEST="${TASK}/manifests/${family}_pilot_manifest.json"
    fi
    export TORCHINDUCTOR_CACHE_DIR="${TASK}/cache/torchinductor_${lane}"
    export TRITON_CACHE_DIR="${TASK}/cache/triton_${lane}"
    cd "${RECIPE_ROOT}"
    "${VENV}/bin/python" src/train.py \
      model="${model}" \
      dataset="${dataset_name}" \
      train_split="${train_split}" \
      val_split="${val_split}" \
      run_id="${run_id}" \
      output_dir="${run_root}" \
      checkpoint_dir="${run_root}" \
      training.num_epochs=1 \
      training.optimizer.lr="${learning_rate}" \
      training.save_interval=1 \
      logging.log_every_n_steps=1 \
      sampling_resolution="${sampling_resolution}" \
      precision="${precision}" \
      profile="${profile_mode}" \
      compile="${compile_mode}" \
      > "${run_root}/train.log" 2>&1
  )
}

pids=()
if [ "${CANARY_VARIANT:-base}" = adaptation10k ]; then
  run_lane 0 mesh_transformer_surface_flagship pretrained estate bfloat16 0.0003 10000 train_64 true &
  pids+=($!)
  run_lane 1 mesh_transformer_surface_flagship pretrained estate bfloat16 0.003 10000 train_64 true &
  pids+=($!)
  run_lane 2 mesh_transformer_surface_flagship pretrained fastback bfloat16 0.0003 10000 train_64 true &
  pids+=($!)
  run_lane 3 mesh_transformer_surface_flagship pretrained fastback bfloat16 0.003 10000 train_64 true &
  pids+=($!)
elif [ "${CANARY_VARIANT:-base}" = adaptation ]; then
  run_lane 0 mesh_transformer_surface_flagship pretrained estate bfloat16 0.0003 2000 train_64 true &
  pids+=($!)
  run_lane 1 mesh_transformer_surface_flagship pretrained estate bfloat16 0.003 2000 train_64 true &
  pids+=($!)
  run_lane 2 mesh_transformer_surface_flagship pretrained fastback bfloat16 0.0003 2000 train_64 true &
  pids+=($!)
  run_lane 3 mesh_transformer_surface_flagship pretrained fastback bfloat16 0.003 2000 train_64 true &
  pids+=($!)
elif [ "${CANARY_VARIANT:-base}" = precision ]; then
  run_lane 0 mesh_transformer_surface_flagship pretrained fastback float32 &
  pids+=($!)
  run_lane 1 mesh_transformer_surface_flagship pretrained estate bfloat16 &
  pids+=($!)
  run_lane 2 mesh_transformer_surface_flagship pretrained estate float32 &
  pids+=($!)
  run_lane 3 mesh_transformer_surface_flagship scratch fastback bfloat16 &
  pids+=($!)
elif [ "${CANARY_VARIANT:-base}" = historical ]; then
  run_lane 0 mesh_transformer_surface_flagship pretrained drivaer bfloat16 0.0 &
  pids+=($!)
  run_lane 1 mesh_transformer_surface_flagship pretrained estate bfloat16 0.0 &
  pids+=($!)
  run_lane 2 mesh_transformer_surface_flagship pretrained fastback bfloat16 0.0 &
  pids+=($!)
  run_lane 3 mesh_transformer_surface_flagship scratch fastback bfloat16 &
  pids+=($!)
else
  run_lane 0 mesh_transformer_surface_flagship scratch estate &
  pids+=($!)
  run_lane 1 mesh_transformer_surface_flagship pretrained fastback &
  pids+=($!)
  run_lane 2 geotransolver_surface scratch fastback &
  pids+=($!)
  run_lane 3 geotransolver_surface pretrained estate &
  pids+=($!)
fi

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

kill "${sampler_pid}" "${watchdog_pid}" 2>/dev/null || true
wait "${sampler_pid}" "${watchdog_pid}" 2>/dev/null || true

for log in "${CANARY_ROOT}"/*/train.log; do
  grep -q "Training completed!" "${log}" || failed=1
done

if [ "${failed}" -ne 0 ]; then
  printf "one or more canary lanes failed\n" > "${TASK}/BLOCKED_CANARY_${SLURM_JOB_ID}"
  exit 1
fi

printf "four canary lanes completed\n" > "${TASK}/DONE_CANARY_${SLURM_JOB_ID}"
