#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-geometry-freeze
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH -e /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0
export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export PYTHONNOUSERSITE=1

EXPECTED_TASK_DIR_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1
EXPECTED_TASK_DIR_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-geometry-input-freeze-v1
TASK_DIR="$EXPECTED_TASK_DIR_LOGICAL"
completed_units=0
# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  if [[ -n "$TASK_DIR" && -d "$TASK_DIR" ]]; then
    printf 'rc=%s\ncompleted_units=%s/1\n' \
      "$exit_code" "$completed_units" > "$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
  fi
  echo "COMPLETED_UNITS=${completed_units}/1 rc=${exit_code}"
  echo "EXIT_CODE=${exit_code}"
}
trap finalize EXIT

if [[ -z "${SLURM_SUBMIT_DIR:-}" || ! -d "$SLURM_SUBMIT_DIR" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is missing or not a directory" >&2
  exit 2
fi
SUBMIT_DIR_PHYSICAL="$(cd -P "$SLURM_SUBMIT_DIR" && pwd)"
if [[ "$SUBMIT_DIR_PHYSICAL" != "$EXPECTED_TASK_DIR_PHYSICAL" ]]; then
  echo "ERROR: submit from the frozen staged task directory" >&2
  echo "expected physical=$EXPECTED_TASK_DIR_PHYSICAL" >&2
  echo "observed physical=$SUBMIT_DIR_PHYSICAL" >&2
  exit 2
fi
for directory in "$TASK_DIR" "$TASK_DIR/artifacts" "$TASK_DIR/sbatch_logs"; do
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    echo "ERROR: required staged directory is missing or a symlink: $directory" >&2
    exit 2
  fi
done

REPO=/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
DATASET=/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
PRODUCER="$TASK_DIR/drivaerml_geometry_input_manifest.py"
OUTPUT="$TASK_DIR/artifacts/drivaerml_geometry_input_manifest_36cases_v1.json"

cd "$TASK_DIR"

exec 9>"$TASK_DIR/.geometry-input-manifest.lock"
if ! flock -n 9; then
  echo "ERROR: another geometry-manifest job holds the task lock" >&2
  exit 3
fi

echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$REPO/.venv-recipe/bin/python" --version

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
check_sha 1cae95d1fbc68d9396d9e1be208fc97b55656d9c21dff72cd8de89dfa9cb0d7a \
  "$PRODUCER"
check_sha 51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca \
  "$DATASET/manifest.json"

for path in "$OUTPUT" "$OUTPUT.sha256"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "ERROR: immutable geometry-manifest target already exists: $path" >&2
    exit 5
  fi
done

timeout --signal=TERM --kill-after=30s 25m \
  "$REPO/.venv-recipe/bin/python" "$PRODUCER" \
  --dataset-root "$DATASET" \
  --output-json "$OUTPUT" \
  --workers 4

(
  cd "$(dirname "$OUTPUT")"
  sha256sum --check "$(basename "$OUTPUT").sha256"
)
completed_units=1
touch "$TASK_DIR/DONE_${SLURM_JOB_ID:-no_job}"
echo "GEOMETRY_INPUT_MANIFEST_DONE $(date -Is)"
