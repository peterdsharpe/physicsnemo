#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-cov-sweep
#SBATCH --time=3:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-coverage-sweep/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append
# H-CC (@sec-nb-hcc-prereg): one unchanged checkpoint, five sampling
# resolutions.  Compensation => error minimised near the 10k training
# resolution; convergence => monotone improvement.
set -euo pipefail
ulimit -c 0
export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
REPO=/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
RECIPE="$REPO/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
TD=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-coverage-sweep
SHADOW=/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
JOBCONF=$(mktemp -d /tmp/mtcov.XXXXXX)
cp "$RECIPE/datasets/drivaer_ml_surface.yaml" "$JOBCONF/"
echo "drivaer_ml: $SHADOW" > "$JOBCONF/dataset_paths.yaml"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-recipe"
cd "$RECIPE"
OUT="$TD/artifacts"
run_one() {
  local RES="$1" GPU="$2"
  echo "=== COV res=$RES start $(date -Is) ==="
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$REPO/.venv-recipe/bin/torchrun" --standalone --nproc_per_node=1 \
    src/infer.py model=mesh_transformer_surface_flagship dataset="$JOBCONF/drivaer_ml_surface" \
    run_id=t2_mesh_transformer_surface_flagship_seed42 infer_split=id_reference \
    sampling_resolution="$RES" output_dir="$OUT/res$RES" \
    >> "$OUT/eval_res${RES}.log" 2>&1
  echo "$?" > "$OUT/RC_res$RES"
  echo "=== COV res=$RES EXIT=$(cat $OUT/RC_res$RES) $(date -Is) ==="
  set -e
}
run_one 2500  0 &
run_one 5000  1 &
run_one 10000 2 &
run_one 20000 3 &
wait
run_one 40000 0
echo "COV_SWEEP_ALL_DONE $(date -Is)"
touch "$TD/DONE"
