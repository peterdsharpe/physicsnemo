#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-s1-wss-audit
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-27-mt-reorientation-artifact-manifest/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

TASK_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-27-mt-reorientation-artifact-manifest
REPO=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
PYTHON="$REPO/.venv-recipe/bin/python"
SCRIPT="$TASK_ROOT/reconstruct_s1_wss_metrics.py"
WRAPPER="$TASK_ROOT/s1_wss_metric_reconstruction.sbatch"

mkdir -p "$TASK_ROOT/artifacts" "$TASK_ROOT/sbatch_logs"
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "COMMAND $PYTHON $SCRIPT --repo $REPO --wrapper $WRAPPER --output-dir $TASK_ROOT/artifacts"
srun --ntasks=1 --cpus-per-task=4 \
  "$PYTHON" "$SCRIPT" \
  --repo "$REPO" \
  --wrapper "$WRAPPER" \
  --output-dir "$TASK_ROOT/artifacts"
echo "END $(date -Is)"
