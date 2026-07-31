#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt_allbc_3way
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

# Three-way all-boundary eval (pre-registration @sec-nb-allbc-launch):
# id_reference evals for all six allbc arms (2 legacy s42 + 4 ab3), then
# spatial-band metrics over those six PLUS the three existing vehicle-only
# flagship control prediction dirs (already on disk from the
# corrected-metric wave -- controls are NOT re-evaluated).
# Job-local dataset config (no shared-file writes); the allbc arms need the
# 5-boundary dataset yaml (schema match) pointed at the OOD shadow root
# (which defines the id_reference split).
# E1 lesson applied: explicit PID-array waits (never bare `wait` with a
# heartbeat subshell alive) + per-run timeout.
set -uo pipefail
ulimit -c 0
export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
REPO=/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
RECIPE="$REPO/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
SHADOW=/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
OUT="$REPO/output/allbc_threeway"
mkdir -p "$OUT"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-recipe"

JOBCONF=$(mktemp -d /tmp/mtab3way.XXXXXX)
cp "$RECIPE/datasets/drivaer_ml_surface_allbc.yaml" "$JOBCONF/"
echo "drivaer_ml: $SHADOW" > "$JOBCONF/dataset_paths.yaml"

# ARM = "RUN_ID@MODEL_CONFIG@LABEL"
ARMS=(
  "t2_mesh_transformer_surface_flagship_allbc_bal_seed42_lr3.0e-3@mesh_transformer_surface_flagship_allbc_bal@bal_s42"
  "t2_mesh_transformer_surface_flagship_allbc_seed42_lr1.5e-3@mesh_transformer_surface_flagship_allbc@raw_s42"
  "t2_mesh_transformer_surface_flagship_allbc_bal_seed43_lr7.5e-4@mesh_transformer_surface_flagship_allbc_bal@bal_s43"
  "t2_mesh_transformer_surface_flagship_allbc_bal_seed44_lr7.5e-4@mesh_transformer_surface_flagship_allbc_bal@bal_s44"
  "t2_mesh_transformer_surface_flagship_allbc_seed43_lr7.5e-4@mesh_transformer_surface_flagship_allbc@raw_s43"
  "t2_mesh_transformer_surface_flagship_allbc_seed44_lr7.5e-4@mesh_transformer_surface_flagship_allbc@raw_s44"
)

run_one() {
  local RUN="$1" MODEL="$2" LABEL="$3" GPU="$4"
  echo "=== 3WAY $LABEL run=$RUN start $(date -Is) ==="
  timeout 2400 env CUDA_VISIBLE_DEVICES="$GPU" \
    "$REPO/.venv-recipe/bin/torchrun" --standalone --nproc_per_node=1 \
    src/infer.py model="$MODEL" dataset="$JOBCONF/drivaer_ml_surface_allbc" \
    run_id="$RUN" infer_split=id_reference sampling_resolution=10000 \
    output_dir="$OUT/$LABEL" \
    >> "$OUT/eval_${LABEL}.log" 2>&1
  echo "=== 3WAY $LABEL EXIT=$? $(date -Is) ==="
}

cd "$RECIPE"
# Wave 1: four arms, one per GPU.  Explicit PID-array wait.
PIDS=()
for i in 0 1 2 3; do
  IFS=@ read -r RUN MODEL LABEL <<< "${ARMS[$i]}"
  run_one "$RUN" "$MODEL" "$LABEL" "$i" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
# Wave 2: remaining two.
PIDS=()
for i in 4 5; do
  IFS=@ read -r RUN MODEL LABEL <<< "${ARMS[$i]}"
  run_one "$RUN" "$MODEL" "$LABEL" "$((i-4))" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

# Band metrics over all nine prediction dirs (6 fresh allbc + 3 existing
# vehicle-only controls from the corrected-metric wave).
BAND_DIRS=()
for A in "${ARMS[@]}"; do
  IFS=@ read -r RUN MODEL LABEL <<< "$A"
  BAND_DIRS+=("$OUT/$LABEL/$RUN")
done
for S in 42 43 44; do
  BAND_DIRS+=("$REPO/output/drivaer_ood/id_reference/t2_mesh_transformer_surface_flagship_seed${S}")
done
"$REPO/.venv-recipe/bin/python" "$RECIPE/tools/band_metrics.py" \
  "${BAND_DIRS[@]}" --out "$OUT/allbc_band_report.json" \
  >> "$OUT/band_metrics.log" 2>&1
echo "BAND_METRICS_EXIT=$? report=$OUT/allbc_band_report.json"
echo "ALL_DONE $(date -Is)"
