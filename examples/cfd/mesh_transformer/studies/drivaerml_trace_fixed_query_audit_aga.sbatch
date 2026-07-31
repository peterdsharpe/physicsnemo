#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-hqc-trace
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -o ./sbatch_logs/%x_%j.log
#SBATCH -e ./sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0
export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TASK_DIR="$(
  cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  pwd
)"
REPO=/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
RECIPE="$REPO/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
DATASET=/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
RUN_DIR="$RECIPE/runs/t2_mesh_transformer_surface_flagship_seed42"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
HISTORICAL_TASK=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-coverage-sweep
HISTORICAL_RUN="$HISTORICAL_TASK/artifacts/res10000/t2_mesh_transformer_surface_flagship_seed42"
PRODUCER="$TASK_DIR/drivaerml_trace_fixed_query_audit.py"
PREREG="$TASK_DIR/phase1_hqc_preregistration_v2_2026-07-27.json"
OUTPUT_DIR="$TASK_DIR/artifacts"
LANE_COUNT=4

mkdir -p "$OUTPUT_DIR" "$TASK_DIR/lane_logs" "$TASK_DIR/sbatch_logs"
cd "$TASK_DIR"

exec 9>"$TASK_DIR/.hqc-producer.lock"
if ! flock -n 9; then
  echo "ERROR: another H-QC producer job holds $TASK_DIR/.hqc-producer.lock" >&2
  exit 2
fi

completed_units=0
sampler_pid=""
# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
  fi
  printf 'rc=%s\ncompleted_units=%s/%s\n' \
    "$exit_code" "$completed_units" "$LANE_COUNT" \
    > "$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
  echo "COMPLETED_UNITS=${completed_units}/${LANE_COUNT} rc=${exit_code}"
  echo "EXIT_CODE=${exit_code}"
}
trap finalize EXIT

echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$REPO/.venv-recipe/bin/python" -c \
  'import numpy, torch, physicsnemo; print("VERSIONS", numpy.__version__, torch.__version__)'

expected_tree=fa6a7b683fa9aa02e4537ef69e8e977906df7c9fa6964cb759edfcee8d7b90cd
actual_tree="$(
  cd "$REPO"
  {
    find physicsnemo/experimental/nn/mesh_attention \
      physicsnemo/experimental/nn/symmetry \
      physicsnemo/mesh physicsnemo/datapipes \
      -type f -name '*.py' -print
    find examples/cfd/external_aerodynamics/unified_external_aero_recipe/src \
      -maxdepth 1 -type f -name '*.py' -print
  } | LC_ALL=C sort |
    while IFS= read -r file; do sha256sum "$file"; done |
    sha256sum | awk '{print $1}'
)"
if [[ "$actual_tree" != "$expected_tree" ]]; then
  echo "ERROR: execution source tree changed: $actual_tree" >&2
  exit 3
fi

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 4
  fi
}
check_sha 8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f \
  "$PRODUCER"
check_sha 88560f29c8ba970f6f17654d99f13309c0d8831e28d90139aea8ea41314f53db \
  "$PREREG"
check_sha 51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca \
  "$DATASET/manifest.json"
check_sha a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3 \
  "$RECIPE/datasets/drivaer_ml_surface.yaml"
check_sha a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1 \
  "$RUN_DIR/resolved_config.yaml"
check_sha 4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88 \
  "$CHECKPOINT_DIR/MeshTransformer.0.491.mdlus"
check_sha 3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7 \
  "$CHECKPOINT_DIR/checkpoint.0.491.pt"
check_sha 31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94 \
  "$CHECKPOINT_DIR/norm_stats.pt"
check_sha 423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a \
  "$HISTORICAL_RUN/metrics.jsonl"

for lane in 0 1 2 3; do
  json="$OUTPUT_DIR/phase1_hqc_producer_lane${lane}.json"
  npz="$OUTPUT_DIR/phase1_hqc_producer_lane${lane}.npz"
  for path in "$json" "$json.sha256" "$npz" "$npz.sha256"; do
    if [[ -e "$path" ]]; then
      echo "ERROR: immutable lane target already exists: $path" >&2
      exit 5
    fi
  done
done

(
  while true; do
    echo "GPU_HEARTBEAT $(date -Is)"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 300
  done
) &
sampler_pid=$!

declare -a lane_pids
for lane in 0 1 2 3; do
  json="$OUTPUT_DIR/phase1_hqc_producer_lane${lane}.json"
  npz="$OUTPUT_DIR/phase1_hqc_producer_lane${lane}.npz"
  echo "LAUNCH lane=$lane gpu=$lane $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$lane"
    "$REPO/.venv-recipe/bin/torchrun" --standalone --nproc_per_node=1 \
      "$PRODUCER" \
      --repo-root "$REPO" \
      --dataset-root "$DATASET" \
      --dataset-config "$RECIPE/datasets/drivaer_ml_surface.yaml" \
      --resolved-config "$RUN_DIR/resolved_config.yaml" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --historical-metrics "$HISTORICAL_RUN/metrics.jsonl" \
      --historical-predictions "$HISTORICAL_RUN/predictions" \
      --lane-ordinal "$lane" \
      --lane-count "$LANE_COUNT" \
      --output-json "$json" \
      --output-npz "$npz"
  ) > "$TASK_DIR/lane_logs/lane${lane}.log" 2>&1 &
  lane_pids[lane]=$!
done

rc=0
for lane in 0 1 2 3; do
  if wait "${lane_pids[$lane]}"; then
    (
      cd "$OUTPUT_DIR"
      sha256sum --check "phase1_hqc_producer_lane${lane}.json.sha256"
      sha256sum --check "phase1_hqc_producer_lane${lane}.npz.sha256"
    )
    completed_units=$((completed_units + 1))
    echo "LANE_DONE lane=$lane completed=$completed_units/$LANE_COUNT $(date -Is)"
  else
    lane_rc=$?
    echo "ERROR: lane=$lane rc=$lane_rc" >&2
    tail -n 80 "$TASK_DIR/lane_logs/lane${lane}.log" >&2 || true
    rc=1
  fi
done

if [[ "$completed_units" -ne "$LANE_COUNT" ]]; then
  rc=1
fi
if [[ "$rc" -eq 0 ]]; then
  touch "$TASK_DIR/DONE_${SLURM_JOB_ID:-no_job}"
  echo "HQC_ALL_LANES_DONE $(date -Is)"
fi
exit "$rc"
