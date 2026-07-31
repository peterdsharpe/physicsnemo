#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-hqc-canon-v4
#SBATCH --time=00:40:00
#SBATCH -p batch
#SBATCH -q short
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
DIAGNOSTIC="$TASK_DIR/drivaerml_hqc_canonical_geometry_diagnostic_v4.py"
PRODUCER="$TASK_DIR/drivaerml_trace_fixed_query_audit.py"
PREREG="$TASK_DIR/phase1_hqc_neutral_canonical_geometry_preregistration_v4_2026-07-28.json"
PRIOR_JSON="$TASK_DIR/hqc_center_diagnostic_four_canaries_k2500_v2.json"
PRIOR_NPZ="$TASK_DIR/hqc_center_diagnostic_four_canaries_k2500_v2.npz"
OUTPUT_DIR="$TASK_DIR/artifacts"

mkdir -p "$OUTPUT_DIR" "$TASK_DIR/sbatch_logs"
cd "$TASK_DIR"

completed_units=0
sampler_pid=""
# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
  fi
  printf 'rc=%s\ncompleted_units=%s/1\n' \
    "$exit_code" "$completed_units" > "$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
  echo "COMPLETED_UNITS=${completed_units}/1 rc=${exit_code}"
  echo "EXIT_CODE=${exit_code}"
}
trap finalize EXIT

echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$REPO/.venv-recipe/bin/python" -c \
  'import numpy, torch, physicsnemo; print("VERSIONS", numpy.__version__, torch.__version__)'

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
check_sha c890271eee04efddd43e979b8bb5cb693209a192a3901ceee149c829d9dc4100 \
  "$DIAGNOSTIC"
check_sha 8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f \
  "$PRODUCER"
check_sha a7b3bd0cddf6e97f15dd8045911068528c4201a2a2fcddd8e3f44429cd57cbfe \
  "$PREREG"
check_sha 26aed78264e9fd66f329941ce000fc438cb26f6835f9f05ff128567d29444bf5 \
  "$PRIOR_JSON"
check_sha d1e6a9fa1a39aa78a9cca26e52eb783a9e78aecbb961ce917164e25fac75a7ea \
  "$PRIOR_NPZ"

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

json="$OUTPUT_DIR/hqc_neutral_canonical_geometry_four_canaries_k2500_v4.json"
npz="$OUTPUT_DIR/hqc_neutral_canonical_geometry_four_canaries_k2500_v4.npz"
for path in "$json" "$json.sha256" "$npz" "$npz.sha256"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: immutable diagnostic target already exists: $path" >&2
    exit 5
  fi
done

(
  while true; do
    echo "GPU_HEARTBEAT $(date -Is)"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 60
  done
) &
sampler_pid=$!

export CUDA_VISIBLE_DEVICES=0
timeout --signal=TERM --kill-after=30s 32m \
  "$REPO/.venv-recipe/bin/torchrun" --standalone --nproc_per_node=1 \
  "$DIAGNOSTIC" \
  --producer "$PRODUCER" \
  --repo-root "$REPO" \
  --dataset-root "$DATASET" \
  --dataset-config "$RECIPE/datasets/drivaer_ml_surface.yaml" \
  --resolved-config "$RUN_DIR/resolved_config.yaml" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --historical-metrics "$HISTORICAL_RUN/metrics.jsonl" \
  --prior-diagnostic-json "$PRIOR_JSON" \
  --prior-diagnostic-npz "$PRIOR_NPZ" \
  --output-json "$json" \
  --output-npz "$npz"

(
  cd "$OUTPUT_DIR"
  sha256sum --check "$(basename "$json").sha256"
  sha256sum --check "$(basename "$npz").sha256"
)
completed_units=1
touch "$TASK_DIR/DONE_${SLURM_JOB_ID:-no_job}"
echo "HQC_NEUTRAL_CANONICAL_GEOMETRY_V4_DONE $(date -Is)"
