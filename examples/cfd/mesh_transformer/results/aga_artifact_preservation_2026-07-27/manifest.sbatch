#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-artifact-manifest
#SBATCH --time=02:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o ./sbatch_logs/%x_%j.log
#SBATCH -e ./sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -uo pipefail
ulimit -c 0
export PATH="/cm/local/apps/slurm/current/bin:${PATH}"

TASK_DIR="$(
  cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  pwd
)"
cd "$TASK_DIR"
mkdir -p artifacts sbatch_logs

rc=0
python3 collect_manifest.py \
  --output artifacts/aga_artifact_manifest_compact_2026-07-27.json || rc=$?

if [ -f artifacts/aga_artifact_manifest_compact_2026-07-27.json ]; then
  sha256sum artifacts/aga_artifact_manifest_compact_2026-07-27.json \
    > artifacts/aga_artifact_manifest_compact_2026-07-27.json.sha256
fi

printf 'rc=%s\n' "$rc" > "STATUS_${SLURM_JOB_ID}"
echo "COMPLETED_UNITS=manifest rc=$rc"
echo "EXIT_CODE=$rc"
exit "$rc"
