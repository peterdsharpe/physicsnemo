#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J shift-morphology
#SBATCH --time=01:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o sbatch_logs/morphology_%j.log
#SBATCH -e sbatch_logs/morphology_%j.log
#SBATCH --open-mode=append

set -euo pipefail

TASK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-shift-suv-adaptation-pilot
VENV=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-21-crossfam-hsg/repo/.venv-recipe
DATA=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets
OUT="${TASK}/morphology_relation"
LOG="${TASK}/sbatch_logs/morphology_${SLURM_JOB_ID}.log"

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TASK}/code"

mkdir -p "${OUT}" "${TASK}/sbatch_logs"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_MORPHOLOGY_${SLURM_JOB_ID}"' EXIT

(
  sleep 900
  while true; do
    modified=$(stat -c %Y "${LOG}" 2>/dev/null || printf 0)
    now=$(date +%s)
    if [ $((now - modified)) -gt 600 ]; then
      printf "morphology log stalled at %s\n" "$(date -Is)" \
        > "${TASK}/BLOCKED_MORPHOLOGY_${SLURM_JOB_ID}"
      scancel "${SLURM_JOB_ID}"
      exit 0
    fi
    sleep 300
  done
) &
watchdog_pid=$!

"${VENV}/bin/python" "${TASK}/shift_suv_morphology_relation.py" \
  --preregistration "${TASK}/shift_suv_morphology_relation_preregistration.json" \
  --drivaer-root "${DATA}/PhysicsNeMo-DrivaerML" \
  --estate-root "${DATA}/PhysicsNeMo-ShiftSUV/estate" \
  --fastback-root "${DATA}/PhysicsNeMo-ShiftSUV/fastback" \
  --drivaer-manifest "${DATA}/PhysicsNeMo-DrivaerML/manifest.json" \
  --estate-manifest "${TASK}/manifests/estate_pilot_manifest.json" \
  --fastback-manifest "${TASK}/manifests/fastback_pilot_manifest.json" \
  --output "${OUT}/formal_morphology_relation.json" \
  --workers 16

kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
printf "morphology relation complete\n" \
  > "${TASK}/DONE_MORPHOLOGY_${SLURM_JOB_ID}"
