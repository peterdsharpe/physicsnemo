#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt_e1_bands
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer-ds/sbatch_logs/%x_%j.log
#SBATCH -e /home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer-ds/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

# E1 wide-Reynolds band-training arms (pre-registered; notebook
# "E1 pre-registration"). v1's EXACT fixed-dataset protocol
# (dataset_train_ns.py iteration-17 lineage: AdamW 3e-4/wd 1e-6/clip 1.0,
# 15 epochs, best-val restore, per-case-Re reporting), only the Reynolds
# band moves. Headline arm mt_singpair_nl_pseudo, full v1 seed triple
# {17,29,43}, bands {v1, v2-re1, v2-re10, v2-re100}. Lanes are ~5-15 min
# each on one GPU (v1-era pace 810 s at 1400 cases), so this is a SINGLE
# 4-GPU job: canary lane solo first (default-deny acceptance), then the
# remaining 11 lanes 4-wide. No chains needed (work << wall).
set -uo pipefail
ulimit -c 0
ulimit -s 8192
export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:${PATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

REPO=/home/psharpe/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer-ds
CAT=/home/psharpe/coreai_modulus_cae/users/psharpe/mt_datasets/ns_cavity_star/ns_cavity_star
OUT=$REPO/output/e1_bands
mkdir -p "$OUT"
cd "$REPO/examples/cfd/mesh_transformer/datasets"
PY=$REPO/.venv/bin/python

run_lane() {
  local BAND="$1" SEED="$2" GPU="$3"
  local TAG="${BAND}_s${SEED}"
  echo "=== LANE $TAG gpu=$GPU start $(date -Is) ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u dataset_train_ns.py \
      --dataset "$CAT/$BAND" --model mt_singpair_nl_pseudo \
      --epochs 15 --seed "$SEED" \
      --output-dir "$OUT/$TAG" \
      > "$OUT/${TAG}.log" 2>&1
  local RC=$?
  echo "=== LANE $TAG EXIT=$RC $(date -Is) reports=$(/bin/ls $OUT/$TAG/*.json 2>/dev/null | wc -l) ==="
  return $RC
}

# Heartbeat: completed-lane count every 10 min.
( while sleep 600; do echo "PROGRESS $(date +%H:%M) reports_done=$(/bin/ls $OUT/*/*.json 2>/dev/null | wc -l)"; done ) &
HB=$!

# CANARY: one lane solo, default-deny acceptance.
run_lane v1 17 0
CRC=$?
CERR=$(grep -cE "Traceback|[Ee]rror" "$OUT/v1_s17.log" 2>/dev/null)
if [ "$CRC" -ne 0 ] || [ "$CERR" -ne 0 ]; then
  echo "CANARY FAILED rc=$CRC errors=$CERR -- aborting fleet"
  tail -5 "$OUT/v1_s17.log"
  kill $HB 2>/dev/null
  exit 1
fi
echo "CANARY PASSED -- launching fleet 4-wide"

# FLEET: remaining 11 lanes, 4-wide across GPUs.
LANES=()
for B in v1 v2-re1 v2-re10 v2-re100; do for S in 17 29 43; do
  [ "$B/$S" = "v1/17" ] && continue
  LANES+=("$B $S")
done; done
i=0
for L in "${LANES[@]}"; do
  set -- $L
  run_lane "$1" "$2" $((i % 4)) &
  i=$((i+1))
  [ $((i % 4)) -eq 0 ] && wait
done
wait
kill $HB 2>/dev/null
echo "ALL_LANES_DONE $(date -Is) total_reports=$(/bin/ls $OUT/*/*.json 2>/dev/null | wc -l)"
