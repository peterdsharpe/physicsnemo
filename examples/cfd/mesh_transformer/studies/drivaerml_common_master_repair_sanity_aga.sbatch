#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-common-fix10k
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
JOB_TAG="${SLURM_JOB_ID:-no_job}"

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

cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} case=run_118"
sha256sum drivaerml_common_master_audit.py

if ! "$PYTHON" -c \
  "import numpy, scipy; print('BACKEND numpy', numpy.__version__, 'scipy', scipy.__version__)"
then
  echo "ERROR: pinned AGA Python cannot import NumPy and SciPy" >&2
  exit 2
fi

output="$TASK_DIR/artifacts/drivaerml_common_master_run_118_k10000_repairfix_2026-07-27.json"
if [ -e "$output" ] || [ -e "$output.sha256" ]; then
  echo "ERROR: refusing to reuse or overwrite repair-sanity output $output" >&2
  exit 3
fi
claim_path="${output}.claim"
if ! mkdir "$claim_path"; then
  echo "ERROR: another job or stale claim owns $output" >&2
  exit 4
fi
claim_active=1

if srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  "$PYTHON" drivaerml_common_master_audit.py \
    --case run_118 "$DATASET/run_118/domain_run_118.pdmsh" \
    --cyclic-replicate run_118 14705232 20270727 \
    --k 10000 \
    --candidate-multiplier 16 \
    --lloyd-iterations 2 \
    --geometry-chunk-cells 250000 \
    --point-chunk 1000000 \
    --repair-pool-size 2048 \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --hash-inputs \
    --output "$output" </dev/null
then
  test -s "$output"
  sha256sum "$output" > "$output.sha256"
  sha256sum --check "$output.sha256"
  completed_units=1
  touch "$TASK_DIR/DONE_${JOB_TAG}"
else
  exit_code=$?
  echo "ERROR: repair sanity failed with rc=$exit_code" >&2
  exit "$exit_code"
fi
