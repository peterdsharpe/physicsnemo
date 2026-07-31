#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-target-freeze
#SBATCH --time=00:15:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=NIL
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-target-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH -e /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-target-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-target-freeze-v1
readonly TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-target-freeze-v1
readonly SCRIPT_SHA256=0f39baadf3d0e3c82b5e2e40e7008f63c41ee3b1ef4ea0d17dfdecd17276c88f
readonly SHARED_REPO=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly PYTHON="$SHARED_REPO/.venv-recipe/bin/python"
readonly DATASET=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow

TASK_DIR="$TASK_LOGICAL"
completed=0

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

finalize() {
  local exit_code=$?
  trap - EXIT
  if [[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]]; then
    local observed
    observed="$(cd -P "$TASK_DIR" 2>/dev/null && pwd)" || true
    if [[ "$observed" == "$TASK_PHYSICAL" ]]; then
      local status="$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
      if [[ ! -e "$status" && ! -L "$status" ]]; then
        printf 'rc=%s\ncompleted_units=%s/1\n' "$exit_code" "$completed" > "$status"
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed}/1 rc=${exit_code}"
  echo "EXIT_CODE=${exit_code}"
  exit "$exit_code"
}
trap finalize EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "wrapper requires an sbatch allocation"
[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || fail "SLURM_SUBMIT_DIR is absent"
submit_physical="$(cd -P "$SLURM_SUBMIT_DIR" && pwd)"
[[ "$submit_physical" == "$TASK_PHYSICAL" ]] ||
  fail "submit directory differs: $submit_physical"
[[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]] ||
  fail "logical task directory is missing or is a symlink"
task_physical="$(cd -P "$TASK_DIR" && pwd)"
[[ "$task_physical" == "$TASK_PHYSICAL" ]] ||
  fail "logical task resolves to $task_physical"
TASK_DIR="$TASK_PHYSICAL"

readonly SCRIPT="$TASK_DIR/drivaerml_historical_k10000_target_input_freeze.py"
readonly OUTPUT="$TASK_DIR/artifacts/historical_k10000_selected_target_input_manifest_v1.json"
for directory in "$TASK_DIR/artifacts" "$TASK_DIR/sbatch_logs"; do
  [[ -d "$directory" && ! -L "$directory" ]] ||
    fail "required task directory is invalid: $directory"
done
[[ -x "$PYTHON" && -f "$(readlink -f "$PYTHON")" ]] ||
  fail "Python interpreter is unavailable"
[[ -f "$SCRIPT" && ! -L "$SCRIPT" ]] || fail "producer is invalid"
actual_script="$(sha256sum "$SCRIPT" | awk '{print $1}')"
[[ "$actual_script" == "$SCRIPT_SHA256" ]] ||
  fail "producer SHA differs: $actual_script"
for path in "$OUTPUT" "$OUTPUT.sha256" "$TASK_DIR/DONE_${SLURM_JOB_ID}"; do
  [[ ! -e "$path" && ! -L "$path" ]] || fail "immutable output exists: $path"
done

exec 9>"$TASK_DIR/.target-input-freeze.lock"
flock -n 9 || fail "another target-input freeze holds the task lock"

echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
"$PYTHON" "$SCRIPT" \
  --dataset-root "$DATASET" \
  --output-json "$OUTPUT"
(
  cd "$TASK_DIR/artifacts"
  sha256sum --check "$(basename "$OUTPUT").sha256"
)
completed=1
touch "$TASK_DIR/DONE_${SLURM_JOB_ID}"
echo "TARGET_INPUT_FREEZE_DONE $(date -Is)"
