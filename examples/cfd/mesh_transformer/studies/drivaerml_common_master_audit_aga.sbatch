#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-common-master
#SBATCH --time=02:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -o ./sbatch_logs/%x_%j.log
#SBATCH -e ./sbatch_logs/%x_%j.log
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

rc=0
completed_units=0
finalize() {
  local exit_code=$?
  printf 'rc=%s\n' "$exit_code" \
    > "$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
  echo "COMPLETED_UNITS=${completed_units}/2 rc=$exit_code"
  echo "EXIT_CODE=$exit_code"
}
trap finalize EXIT

cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"

if ! "$PYTHON" -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"; then
  echo "ERROR: pinned AGA Python cannot import NumPy and SciPy" >&2
  rc=2
else
  case_ids=(run_1 run_118)
  cyclic_starts=(13499736 14705232)
  for index in "${!case_ids[@]}"; do
    case_id="${case_ids[$index]}"
    cyclic_start="${cyclic_starts[$index]}"
    output="$TASK_DIR/artifacts/drivaerml_common_master_${case_id}_k10000_2026-07-27.json"
    if [ -f "$output" ]; then
      if [ -f "$output.sha256" ] && sha256sum --check "$output.sha256"; then
        echo "REUSE completed case=$case_id output=$output"
        completed_units=$((completed_units + 1))
        continue
      fi
      echo "ERROR: refusing to overwrite unverified existing output $output" >&2
      rc=3
      break
    fi
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      "$PYTHON" drivaerml_common_master_audit.py \
        --case "$case_id" "$DATASET/$case_id/domain_${case_id}.pdmsh" \
        --cyclic-replicate "$case_id" "$cyclic_start" 20270727 \
        --k 10000 \
        --lloyd-iterations 2 \
        --workers "${SLURM_CPUS_PER_TASK:-16}" \
        --hash-inputs \
        --output "$output" || {
          rc=$?
          break
        }
    sha256sum "$output" > "$output.sha256"
    completed_units=$((completed_units + 1))
  done
fi

exit "$rc"
