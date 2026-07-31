#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-common-40k
#SBATCH --time=02:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-1
#SBATCH -o ./sbatch_logs/%x_%A_%a.log
#SBATCH -e ./sbatch_logs/%x_%A_%a.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0
export PATH="/cm/local/apps/slurm/current/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONNOUSERSITE=1

TASK_DIR="$(
  cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  pwd
)"
DATASET=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML
PYTHON="$TASK_DIR/.venv/bin/python"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This deciding job must run as an array}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-no_job}}_${ARRAY_INDEX}"

completed_units=0
claim_active=0
claim_path=
terminate() {
  exit 143
}
finalize() {
  local exit_code=$?
  if [ "$completed_units" -ne 1 ] && [ "$exit_code" -eq 0 ]; then
    exit_code=5
  fi
  if [ "$claim_active" -eq 1 ]; then
    rmdir "$claim_path" 2>/dev/null || true
  fi
  printf 'rc=%s\n' "$exit_code" > "$TASK_DIR/STATUS_${JOB_TAG}"
  echo "COMPLETED_UNITS=${completed_units}/1 rc=$exit_code"
  echo "EXIT_CODE=$exit_code"
  trap - EXIT
  exit "$exit_code"
}
trap terminate TERM INT
trap finalize EXIT

case "$ARRAY_INDEX" in
  0)
    case_id=run_1
    cyclic_starts=(11914080 11878329 14546186 2529054)
    ;;
  1)
    case_id=run_118
    cyclic_starts=(2517171 651819 4220546 6617334)
    ;;
  *)
    echo "ERROR: unsupported array index $ARRAY_INDEX" >&2
    exit 2
    ;;
esac

cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} case=$case_id"

if ! "$PYTHON" -c \
  "import numpy, scipy; print('BACKEND numpy', numpy.__version__, 'scipy', scipy.__version__)"
then
  echo "ERROR: pinned AGA Python cannot import NumPy and SciPy" >&2
  exit 3
fi

output="$TASK_DIR/artifacts/drivaerml_common_master_${case_id}_k40000_2026-07-27.json"
if [ -e "$output" ] || [ -e "$output.sha256" ]; then
  echo "ERROR: refusing to reuse or overwrite existing deciding output $output" >&2
  echo "Submit only the missing array index after verifying completed artifacts." >&2
  exit 4
fi
claim_path="${output}.claim"
if ! mkdir "$claim_path"; then
  echo "ERROR: another job or stale claim owns $output" >&2
  exit 6
fi
claim_active=1

audit_args=(
  --case "$case_id" "$DATASET/$case_id/domain_${case_id}.pdmsh"
  --k 40000
  --lloyd-iterations 2
  --workers "${SLURM_CPUS_PER_TASK:-16}"
  --hash-inputs
  --output "$output"
)
for cyclic_start in "${cyclic_starts[@]}"; do
  audit_args+=(--cyclic-replicate "$case_id" "$cyclic_start" 20300727)
done

if srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  "$PYTHON" drivaerml_common_master_audit.py "${audit_args[@]}" </dev/null
then
  test -s "$output"
  sha256sum "$output" > "$output.sha256"
  sha256sum --check "$output.sha256"
  completed_units=1
  touch "$TASK_DIR/DONE_${JOB_TAG}"
else
  exit_code=$?
  echo "ERROR: production audit failed with rc=$exit_code" >&2
  exit "$exit_code"
fi
