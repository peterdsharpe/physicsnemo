#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-gauge-calibration
#SBATCH --time=02:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ./sbatch_logs/%x_%j.log
#SBATCH -e ./sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -uo pipefail
ulimit -c 0

TASK_DIR="$(
  cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  pwd
)"
RECIPE_DIR="${TASK_DIR}/repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
BASE_PYTHON="/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv/bin/python"

mkdir -p "${TASK_DIR}/artifacts" "${TASK_DIR}/sbatch_logs"
cd "${RECIPE_DIR}"
export PYTHONPATH="${TASK_DIR}/repo${PYTHONPATH:+:${PYTHONPATH}}"

rc=0
"${BASE_PYTHON}" studies/calibrate_intrinsic_gauge.py || rc=$?

if [ "$rc" -eq 0 ]; then
  cp studies/intrinsic_gauge_calibration.json \
    "${TASK_DIR}/artifacts/intrinsic_gauge_calibration_train_only_2026-07-27.json"
  (
    cd "${TASK_DIR}"
    sha256sum \
      artifacts/intrinsic_gauge_calibration_train_only_2026-07-27.json \
      repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/studies/calibrate_intrinsic_gauge.py \
      repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/domain_transforms.py \
      repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/intrinsic_gauge.py \
      repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/drivaer_ml_surface.yaml \
      repo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/dataset_paths.yaml \
      repo/physicsnemo/datapipes/_indexing.py \
      repo/physicsnemo/datapipes/readers/mesh.py \
      repo/physicsnemo/datapipes/transforms/mesh/transforms.py \
      repo/physicsnemo/mesh/calculus/measure.py \
      > artifacts/SHA256SUMS
  )
fi

printf 'rc=%s\n' "$rc" > "${TASK_DIR}/STATUS_${SLURM_JOB_ID}"
echo "COMPLETED_UNITS=calibration rc=$rc"
echo "EXIT_CODE=$rc"
exit "$rc"
