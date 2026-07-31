#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-curated-sampler
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -o ./sbatch_logs/%x_%j.log
#SBATCH -e ./sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0
export PATH="/cm/local/apps/slurm/current/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK_DIR="$(
  cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  pwd
)"
DATASET=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML
OUTPUT="$TASK_DIR/artifacts/drivaerml_curated_sampler_2026-07-27.json"

cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"

rc=0
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  python3 drivaerml_curated_sampler.py \
    --case run_1 "$DATASET/run_1/domain_run_1.pdmsh" \
      11914080,11878329,14546186,2529054 \
    --case run_2 "$DATASET/run_2/domain_run_2.pdmsh" \
      2471027,12664241,5152891,11042029 \
    --output "$OUTPUT" || rc=$?

if [ -f "$OUTPUT" ]; then
  sha256sum "$OUTPUT" > "$OUTPUT.sha256"
fi
printf 'rc=%s\n' "$rc" > "STATUS_${SLURM_JOB_ID}"
echo "COMPLETED_UNITS=2/2 rc=$rc"
echo "EXIT_CODE=$rc"
exit "$rc"
