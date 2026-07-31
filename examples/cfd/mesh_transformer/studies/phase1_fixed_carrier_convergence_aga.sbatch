#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-fixed-carrier
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
PYTHON=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv/bin/python
OUTPUT="$TASK_DIR/artifacts/phase1_fixed_carrier_convergence_2026-07-27_level5.json"
REPRESENTATION="$TASK_DIR/phase1_representation_156faces_2026-07-27.npz"
EXPECTED_REPRESENTATION_SHA256=5df5bb3305738de575473c574369b640629b96bd5bede9de2c3e555fac88f713

record_status() {
  rc=$?
  trap - EXIT
  printf 'rc=%s\n' "$rc" > "$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
  echo "EXIT_CODE=$rc"
  exit "$rc"
}
trap record_status EXIT

cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "BASE_GIT_SHA=a3b566fa5390115a475c36c19db4157c161385bf"
echo "$EXPECTED_REPRESENTATION_SHA256  $REPRESENTATION" | sha256sum --check
sha256sum code/examples/cfd/mesh_transformer/studies/*.py \
  code/examples/cfd/mesh_transformer/provenance.py \
  > artifacts/fixed_carrier_code_manifest.sha256
"$PYTHON" -c "import pyacvd, torch; print('torch', torch.__version__, 'pyacvd', pyacvd.__version__)"

PYTHONPATH="$TASK_DIR/code" \
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  "$PYTHON" \
  code/examples/cfd/mesh_transformer/studies/phase1_fixed_carrier_convergence.py \
  --max-level 5 \
  --representation-npz "$REPRESENTATION" \
  --output "$OUTPUT"

sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "COMPLETED_UNITS=1/1"
