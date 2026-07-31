#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-paired-v1
#SBATCH --time=00:45:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
# AGA schedules whole four-GPU nodes; canonical A/B use 0/1 and the sentinel uses 2.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-paired-accuracy-v1/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-paired-accuracy-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly EXPECTED_TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-paired-accuracy-v1
readonly EXPECTED_TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-paired-accuracy-v1
readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly DATASET_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow

readonly EXPECTED_CANONICAL_PRODUCER_SHA256=86455a2fecd018896ecdc7989b9ea558ad8c72fd71d0d3f2f7f35d0c6f429d66
readonly EXPECTED_LEGACY_PRODUCER_SHA256=bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5
readonly EXPECTED_RUNTIME_HELPER_SHA256=dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896
readonly EXPECTED_CANONICAL_HELPER_SHA256=694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac
readonly EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256=4b4f4c893ca154b589b9d1276bf4f43cd52008664f75f3c56fcc66aa26ce8f68
readonly EXPECTED_PAIRED_REDUCER_SHA256=dcc753bb1bd1b998a395e0723acfae32f8bb2a4c4114e3fedd2d0a76d604cf88
readonly EXPECTED_PREREGISTRATION_SHA256=2d68b34b60678313d59106c89a7ad2abfe0e75a0533378cbb322611314d8e243
readonly EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256=e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4
readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256=fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e
readonly EXPECTED_DATASET_MANIFEST_SHA256=51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca
readonly EXPECTED_DATASET_CONFIG_SHA256=a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3
readonly EXPECTED_RESOLVED_CONFIG_SHA256=a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1
readonly EXPECTED_MODEL_CHECKPOINT_SHA256=4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88
readonly EXPECTED_TRAINING_STATE_SHA256=3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7
readonly EXPECTED_NORMALIZATION_STATE_SHA256=31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94
readonly EXPECTED_CURRENT_INFER_SHA256=47aec675e54d58ee4202831ae0d20039b1ff5ec40e69cc8b5087ce00bd5234ed
readonly EXPECTED_CURRENT_MODEL_SOURCE_SHA256=9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892
readonly EXPECTED_GEOMETRY_MANIFEST_SHA256=3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0
readonly EXPECTED_TARGET_MANIFEST_SHA256=d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49
readonly EXPECTED_INPUT_FREEZE_SHA256=fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10
readonly EXPECTED_JOB306114_CORRECTION_SHA256=f2e448913e34c1491586f07d9c4bdc3dac6a9ca680cce7f81c345424757e0901
readonly EXPECTED_STAGE_B_ADJUDICATION_SHA256=6a2d2274611e7be533a18ba9b68f7793ccd3a4777e96df389552d55f64839a1b
readonly EXPECTED_STAGE_B_RESULT_SHA256=89b861cc0b0ab3f0b944f96038e41814e01175c075fed75fd828e95aaebbf84d
readonly EXPECTED_STAGE_B_TREE_MANIFEST_SHA256=79e40c8117b47dafcb755faf4c16e00eb117fd3b2880c71bdbf23d51d7591810
readonly EXPECTED_STAGE_B_A_JSON_SHA256=83f0d53326a128def18d538d427ac48b11a0d789417366cfd7c1dc020b193d67
readonly EXPECTED_STAGE_B_B_JSON_SHA256=c7d1254335ac381524dace6c9b6b45c0ae1e275df99b9ca505cbe2395c957b19
readonly EXPECTED_STAGE_B_NPZ_SHA256=38353473af9eba1e606e170c70c5d6ed4d7fd736d631c6388dd2244dc1e203c4
readonly EXPECTED_STAGE_B_REMOTE_FILE_COUNT=957
readonly EXPECTED_STAGE_B_REMOTE_SIZE_BYTES=159205162
readonly EXPECTED_STAGE_B_LICENSE_FILE_COUNT=961
readonly EXPECTED_STAGE_B_LICENSE_SIZE_BYTES=159343806
readonly UNIT_COUNT=4

TASK_DIR="$EXPECTED_TASK_LOGICAL"
completed_units=0
heartbeat_pid=""
categorical_outcome="NONE"
phase="PRELAUNCH_VALIDATION"
declare -a lane_pids=()

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

write_exclusive_marker() {
  local path="$1"
  local payload="$2"
  if [[ -e "$path" || -L "$path" ]]; then
    fail "refusing to overwrite immutable marker: $path"
  fi
  (
    set -o noclobber
    printf '%s\n' "$payload" > "$path"
  ) || fail "could not publish immutable marker: $path"
}

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  local pid
  trap - EXIT
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  for pid in "${lane_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null ||
        kill -TERM "$pid" 2>/dev/null ||
        true
    fi
  done
  for pid in "${lane_pids[@]:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]]; then
    local observed_task
    observed_task="$(cd -P "$TASK_DIR" 2>/dev/null && pwd)" || true
    if [[ "$observed_task" == "$EXPECTED_TASK_PHYSICAL" ]]; then
      local status_path="$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
      if [[ -e "$status_path" || -L "$status_path" ]]; then
        echo "ERROR: refusing to overwrite status marker: $status_path" >&2
      else
        (
          set -o noclobber
          printf \
            'rc=%s\ncompleted_units=%s/%s\nphase=%s\ncategorical_outcome=%s\n' \
            "$exit_code" \
            "$completed_units" \
            "$UNIT_COUNT" \
            "$phase" \
            "$categorical_outcome" > "$status_path"
        ) || echo "ERROR: could not publish status marker: $status_path" >&2
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed_units}/${UNIT_COUNT} rc=${exit_code}"
  echo "PAIRED_ACCURACY_CATEGORICAL_OUTCOME=${categorical_outcome}"
  echo "EXIT_CODE=${exit_code}"
  exit "$exit_code"
}
trap finalize EXIT

check_absent() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    fail "immutable output already exists: $path"
  fi
}

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  if [[ ! -f "$path" || -L "$path" ]]; then
    fail "hashed input is missing, non-regular, or a symlink: $path"
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    fail "SHA mismatch path=$path expected=$expected actual=$actual"
  fi
}

verify_canonical_sidecar() {
  local path="$1"
  local sidecar="$path.sha256"
  local digest
  if [[ ! -f "$path" || -L "$path" || ! -f "$sidecar" || -L "$sidecar" ]]; then
    return 1
  fi
  digest="$(sha256sum "$path" | awk '{print $1}')" || return 1
  cmp -s \
    "$sidecar" \
    <(printf '%s  %s\n' "$digest" "$(basename "$path")")
}

heartbeat_completed_units() {
  local observed=0
  if verify_canonical_sidecar "$CANONICAL_A_JSON" &&
    verify_canonical_sidecar "$CANONICAL_A_NPZ"; then
    observed=$((observed + 1))
  fi
  if verify_canonical_sidecar "$CANONICAL_B_JSON" &&
    verify_canonical_sidecar "$CANONICAL_B_NPZ"; then
    observed=$((observed + 1))
  fi
  if verify_canonical_sidecar "$SENTINEL_JSON" &&
    verify_canonical_sidecar "$SENTINEL_NPZ"; then
    observed=$((observed + 1))
  fi
  if verify_canonical_sidecar "$ADJUDICATION"; then
    observed=$((observed + 1))
  fi
  printf '%s' "$observed"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail "this wrapper must run inside an sbatch allocation"
fi
if [[ -z "${SLURM_SUBMIT_DIR:-}" || ! -d "$SLURM_SUBMIT_DIR" ]]; then
  fail "SLURM_SUBMIT_DIR is missing or is not a directory"
fi
if [[ "${SLURM_NNODES:-1}" != 1 ]]; then
  fail "paired accuracy requires exactly one allocated node"
fi
submit_physical="$(cd -P "$SLURM_SUBMIT_DIR" && pwd)"
if [[ "$submit_physical" != "$EXPECTED_TASK_PHYSICAL" ]]; then
  fail "submit from $EXPECTED_TASK_PHYSICAL, observed $submit_physical"
fi
if [[ ! -d "$TASK_DIR" || -L "$TASK_DIR" ]]; then
  fail "logical task directory is missing or is a symlink: $TASK_DIR"
fi
task_physical="$(cd -P "$TASK_DIR" && pwd)"
if [[ "$task_physical" != "$EXPECTED_TASK_PHYSICAL" ]]; then
  fail "logical task directory resolves to $task_physical"
fi
TASK_DIR="$EXPECTED_TASK_PHYSICAL"

readonly OUTPUT_DIR="$TASK_DIR/artifacts"
readonly LANE_LOG_DIR="$TASK_DIR/lane_logs"
readonly SBATCH_LOG_DIR="$TASK_DIR/sbatch_logs"
readonly EXECUTION_SOURCE="$TASK_DIR/execution_source"
readonly STAGE_B_LICENSE_DIR="$TASK_DIR/stage_b_v2_license"
for directory in \
  "$OUTPUT_DIR" \
  "$LANE_LOG_DIR" \
  "$SBATCH_LOG_DIR" \
  "$EXECUTION_SOURCE" \
  "$STAGE_B_LICENSE_DIR"; do
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    fail "required staged directory is missing or is a symlink: $directory"
  fi
done
if [[ "$(cd -P "$EXECUTION_SOURCE" && pwd)" != "$TASK_DIR/execution_source" ]]; then
  fail "execution source is not the task-local physical snapshot"
fi
if [[ "$(cd -P "$STAGE_B_LICENSE_DIR" && pwd)" != "$TASK_DIR/stage_b_v2_license" ]]; then
  fail "Stage-B v2 license resolves outside the staged task"
fi
if [[ ! -d "$DATASET_PHYSICAL" ]]; then
  fail "dataset root is missing: $DATASET_PHYSICAL"
fi

readonly VENV="$SHARED_ROOT_PHYSICAL/.venv-recipe"
readonly PYTHON="$VENV/bin/python"
readonly RECIPE="$EXECUTION_SOURCE/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly SHARED_RECIPE="$SHARED_ROOT_PHYSICAL/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly RUN_DIR="$SHARED_RECIPE/runs/t2_mesh_transformer_surface_flagship_seed42"
readonly CHECKPOINT_DIR="$RUN_DIR/checkpoints"
readonly DATASET_MANIFEST="$DATASET_PHYSICAL/manifest.json"
readonly DATASET_CONFIG="$RECIPE/datasets/drivaer_ml_surface.yaml"
readonly RESOLVED_CONFIG="$RUN_DIR/resolved_config.yaml"
readonly MODEL_CHECKPOINT="$CHECKPOINT_DIR/MeshTransformer.0.491.mdlus"
readonly TRAINING_STATE="$CHECKPOINT_DIR/checkpoint.0.491.pt"
readonly NORMALIZATION_STATE="$CHECKPOINT_DIR/norm_stats.pt"
readonly CURRENT_INFER="$RECIPE/src/infer.py"
readonly CURRENT_MODEL_SOURCE="$EXECUTION_SOURCE/physicsnemo/experimental/nn/mesh_attention/model.py"

readonly CANONICAL_PRODUCER="$TASK_DIR/drivaerml_historical_k10000_canonical_arm.py"
readonly LEGACY_PRODUCER="$TASK_DIR/drivaerml_historical_k10000_replay.py"
readonly RUNTIME_HELPER="$TASK_DIR/drivaerml_historical_k10000_replay_runtime.py"
readonly CANONICAL_HELPER="$TASK_DIR/drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
readonly STAGE_B_REDUCER_SUPPORT="$TASK_DIR/drivaerml_historical_k10000_replay_adjudicate.py"
readonly PAIRED_REDUCER="$TASK_DIR/drivaerml_historical_k10000_paired_accuracy_adjudicate.py"
readonly PREREGISTRATION="$TASK_DIR/phase1_historical_k10000_paired_accuracy_prereg_v1_2026-07-28.json"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
readonly GEOMETRY_MANIFEST="$TASK_DIR/drivaerml_geometry_input_manifest_36cases_v1.json"
readonly TARGET_INPUT_MANIFEST="$TASK_DIR/historical_k10000_selected_target_input_manifest_v1.json"
readonly INPUT_FREEZE="$TASK_DIR/historical_k10000_input_freeze_v1.json"
readonly JOB306114_CORRECTION="$TASK_DIR/phase1_hqc_canonical_geometry_checkpoint_binding_correction_v1_2026-07-28.json"

readonly STAGE_B_A_JSON="$STAGE_B_LICENSE_DIR/artifacts/historical_k10000_replay_lane_A.json"
readonly STAGE_B_A_NPZ="$STAGE_B_LICENSE_DIR/artifacts/historical_k10000_replay_lane_A.npz"
readonly STAGE_B_B_JSON="$STAGE_B_LICENSE_DIR/artifacts/historical_k10000_replay_lane_B.json"
readonly STAGE_B_B_NPZ="$STAGE_B_LICENSE_DIR/artifacts/historical_k10000_replay_lane_B.npz"
readonly STAGE_B_ADJUDICATION="$STAGE_B_LICENSE_DIR/artifacts/historical_k10000_replay_adjudication.json"
readonly STAGE_B_TREE_MANIFEST="$STAGE_B_LICENSE_DIR/sealed_remote_task_tree_manifest.sha256"
readonly STAGE_B_RESULT="$STAGE_B_LICENSE_DIR/phase1_historical_k10000_stage_b_replay_result_v2_job306814_2026-07-28.json"

readonly CANONICAL_A_JSON="$OUTPUT_DIR/historical_k10000_canonical_lane_A.json"
readonly CANONICAL_A_NPZ="$OUTPUT_DIR/historical_k10000_canonical_lane_A.npz"
readonly CANONICAL_B_JSON="$OUTPUT_DIR/historical_k10000_canonical_lane_B.json"
readonly CANONICAL_B_NPZ="$OUTPUT_DIR/historical_k10000_canonical_lane_B.npz"
readonly SENTINEL_JSON="$OUTPUT_DIR/historical_k10000_fresh_legacy_sentinel_A.json"
readonly SENTINEL_NPZ="$OUTPUT_DIR/historical_k10000_fresh_legacy_sentinel_A.npz"
readonly CANONICAL_A_LOG="$LANE_LOG_DIR/canonical_A.log"
readonly CANONICAL_B_LOG="$LANE_LOG_DIR/canonical_B.log"
readonly SENTINEL_LOG="$LANE_LOG_DIR/fresh_legacy_sentinel_A.log"
readonly ADJUDICATION="$OUTPUT_DIR/historical_k10000_paired_accuracy_adjudication.json"

resolved_python="$(readlink -f "$PYTHON")"
if [[ ! -x "$PYTHON" || ! -f "$resolved_python" ]]; then
  fail "shared physical venv interpreter is unavailable: $PYTHON"
fi

for expected in \
  "$EXPECTED_CANONICAL_PRODUCER_SHA256" \
  "$EXPECTED_LEGACY_PRODUCER_SHA256" \
  "$EXPECTED_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_CANONICAL_HELPER_SHA256" \
  "$EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256" \
  "$EXPECTED_PAIRED_REDUCER_SHA256" \
  "$EXPECTED_PREREGISTRATION_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_DATASET_MANIFEST_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256" \
  "$EXPECTED_CURRENT_INFER_SHA256" \
  "$EXPECTED_CURRENT_MODEL_SOURCE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  "$EXPECTED_TARGET_MANIFEST_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256" \
  "$EXPECTED_JOB306114_CORRECTION_SHA256" \
  "$EXPECTED_STAGE_B_ADJUDICATION_SHA256" \
  "$EXPECTED_STAGE_B_RESULT_SHA256" \
  "$EXPECTED_STAGE_B_TREE_MANIFEST_SHA256" \
  "$EXPECTED_STAGE_B_A_JSON_SHA256" \
  "$EXPECTED_STAGE_B_B_JSON_SHA256" \
  "$EXPECTED_STAGE_B_NPZ_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

check_sha "$EXPECTED_CANONICAL_PRODUCER_SHA256" "$CANONICAL_PRODUCER"
check_sha "$EXPECTED_LEGACY_PRODUCER_SHA256" "$LEGACY_PRODUCER"
check_sha "$EXPECTED_RUNTIME_HELPER_SHA256" "$RUNTIME_HELPER"
check_sha "$EXPECTED_CANONICAL_HELPER_SHA256" "$CANONICAL_HELPER"
check_sha "$EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256" "$STAGE_B_REDUCER_SUPPORT"
check_sha "$EXPECTED_PAIRED_REDUCER_SHA256" "$PAIRED_REDUCER"
check_sha "$EXPECTED_PREREGISTRATION_SHA256" "$PREREGISTRATION"
check_sha \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXECUTION_SOURCE_MANIFEST"
check_sha "$EXPECTED_DATASET_MANIFEST_SHA256" "$DATASET_MANIFEST"
check_sha "$EXPECTED_DATASET_CONFIG_SHA256" "$DATASET_CONFIG"
check_sha "$EXPECTED_RESOLVED_CONFIG_SHA256" "$RESOLVED_CONFIG"
check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256" "$MODEL_CHECKPOINT"
check_sha "$EXPECTED_TRAINING_STATE_SHA256" "$TRAINING_STATE"
check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256" "$NORMALIZATION_STATE"
check_sha "$EXPECTED_CURRENT_INFER_SHA256" "$CURRENT_INFER"
check_sha "$EXPECTED_CURRENT_MODEL_SOURCE_SHA256" "$CURRENT_MODEL_SOURCE"
check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256" "$GEOMETRY_MANIFEST"
check_sha "$EXPECTED_TARGET_MANIFEST_SHA256" "$TARGET_INPUT_MANIFEST"
check_sha "$EXPECTED_INPUT_FREEZE_SHA256" "$INPUT_FREEZE"
check_sha "$EXPECTED_JOB306114_CORRECTION_SHA256" "$JOB306114_CORRECTION"
check_sha "$EXPECTED_STAGE_B_ADJUDICATION_SHA256" "$STAGE_B_ADJUDICATION"
check_sha "$EXPECTED_STAGE_B_RESULT_SHA256" "$STAGE_B_RESULT"
check_sha "$EXPECTED_STAGE_B_TREE_MANIFEST_SHA256" "$STAGE_B_TREE_MANIFEST"
check_sha "$EXPECTED_STAGE_B_A_JSON_SHA256" "$STAGE_B_A_JSON"
check_sha "$EXPECTED_STAGE_B_NPZ_SHA256" "$STAGE_B_A_NPZ"
check_sha "$EXPECTED_STAGE_B_B_JSON_SHA256" "$STAGE_B_B_JSON"
check_sha "$EXPECTED_STAGE_B_NPZ_SHA256" "$STAGE_B_B_NPZ"

for path in \
  "$GEOMETRY_MANIFEST" \
  "$TARGET_INPUT_MANIFEST" \
  "$INPUT_FREEZE" \
  "$JOB306114_CORRECTION" \
  "$STAGE_B_A_JSON" \
  "$STAGE_B_A_NPZ" \
  "$STAGE_B_B_JSON" \
  "$STAGE_B_B_NPZ" \
  "$STAGE_B_ADJUDICATION" \
  "$STAGE_B_TREE_MANIFEST" \
  "$STAGE_B_RESULT"; do
  if ! verify_canonical_sidecar "$path"; then
    fail "frozen input sidecar is missing, stale, or noncanonical: $path"
  fi
done

source_symlink="$(find "$EXECUTION_SOURCE" -type l -print -quit)"
if [[ -n "$source_symlink" ]]; then
  fail "execution source contains a symlink: $source_symlink"
fi
source_cache_path="$(
  find "$EXECUTION_SOURCE" \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
    -print -quit
)"
if [[ -n "$source_cache_path" ]]; then
  fail "execution source contains an unpinned Python cache path: $source_cache_path"
fi
unexpected_recipe_file="$(
  find "$RECIPE" \
    -type d -name __pycache__ -prune -o \
    -type f ! -name '*.pyc' \
    ! -path "$RECIPE/src/*" \
    ! -path "$DATASET_CONFIG" \
    -print -quit
)"
if [[ -n "$unexpected_recipe_file" ]]; then
  fail "execution snapshot contains an unregistered recipe file: $unexpected_recipe_file"
fi
unexpected_nested_recipe_source="$(
  find "$RECIPE/src" \
    -mindepth 2 \
    -type d -name __pycache__ -prune -o \
    -type f ! -name '*.pyc' -print -quit
)"
if [[ -n "$unexpected_nested_recipe_source" ]]; then
  fail "recipe src contains an unregistered nested file: $unexpected_nested_recipe_source"
fi

recomputed_execution_source_manifest="$(
  cd "$EXECUTION_SOURCE"
  {
    find physicsnemo \
      -type d -name __pycache__ -prune -o \
      -type f ! -name '*.pyc' -print
    find \
      examples/cfd/external_aerodynamics/unified_external_aero_recipe/src \
      -maxdepth 1 -type f ! -name '*.pyc' -print
    printf '%s\n' \
      examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/drivaer_ml_surface.yaml
  } |
    LC_ALL=C sort |
    while IFS= read -r file; do sha256sum "$file"; done
)"
if ! cmp -s \
  "$EXECUTION_SOURCE_MANIFEST" \
  <(printf '%s\n' "$recomputed_execution_source_manifest"); then
  fail "execution-source manifest bytes differ from the exact staged inventory"
fi
(
  cd "$EXECUTION_SOURCE"
  sha256sum --quiet --strict --check "$EXECUTION_SOURCE_MANIFEST"
) || fail "execution-source manifest content verification failed"

for root in \
  physicsnemo/experimental/nn/mesh_attention \
  physicsnemo/experimental/nn/symmetry \
  physicsnemo/mesh \
  physicsnemo/datapipes \
  examples/cfd/external_aerodynamics/unified_external_aero_recipe/src; do
  if [[ ! -d "$EXECUTION_SOURCE/$root" ]]; then
    fail "execution-source root is missing: $root"
  fi
done
actual_source_tree="$(
  cd "$EXECUTION_SOURCE"
  {
    find \
      physicsnemo/experimental/nn/mesh_attention \
      physicsnemo/experimental/nn/symmetry \
      physicsnemo/mesh \
      physicsnemo/datapipes \
      -type f -name '*.py' -print
    find \
      examples/cfd/external_aerodynamics/unified_external_aero_recipe/src \
      -maxdepth 1 -type f -name '*.py' -print
  } |
    LC_ALL=C sort |
    while IFS= read -r file; do sha256sum "$file"; done |
    sha256sum |
    awk '{print $1}'
)"
if [[ "$actual_source_tree" != "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" ]]; then
  fail "execution source tree changed: expected=$EXPECTED_EXECUTION_SOURCE_TREE_SHA256 actual=$actual_source_tree"
fi

stage_b_symlink="$(find "$STAGE_B_LICENSE_DIR" -type l -print -quit)"
if [[ -n "$stage_b_symlink" ]]; then
  fail "Stage-B license tree contains a symlink: $stage_b_symlink"
fi
stage_b_special="$(
  find "$STAGE_B_LICENSE_DIR" ! -type d ! -type f -print -quit
)"
if [[ -n "$stage_b_special" ]]; then
  fail "Stage-B license tree contains a non-regular entry: $stage_b_special"
fi
read -r stage_b_license_file_count stage_b_license_size_bytes < <(
  find "$STAGE_B_LICENSE_DIR" -type f -printf '%s\n' |
    awk '{ count += 1; bytes += $1 } END { print count + 0, bytes + 0 }'
)
if [[ "$stage_b_license_file_count" != "$EXPECTED_STAGE_B_LICENSE_FILE_COUNT" ]]; then
  fail "Stage-B license file count changed: $stage_b_license_file_count"
fi
if [[ "$stage_b_license_size_bytes" != "$EXPECTED_STAGE_B_LICENSE_SIZE_BYTES" ]]; then
  fail "Stage-B license byte count changed: $stage_b_license_size_bytes"
fi
if [[ "$(wc -l < "$STAGE_B_TREE_MANIFEST")" != "$EXPECTED_STAGE_B_REMOTE_FILE_COUNT" ]]; then
  fail "Stage-B tree-manifest line count changed"
fi
recomputed_stage_b_manifest="$(
  cd "$STAGE_B_LICENSE_DIR"
  find . -type f \
    ! -path "./sealed_remote_task_tree_manifest.sha256" \
    ! -path "./sealed_remote_task_tree_manifest.sha256.sha256" \
    ! -path "./phase1_historical_k10000_stage_b_replay_result_v2_job306814_2026-07-28.json" \
    ! -path "./phase1_historical_k10000_stage_b_replay_result_v2_job306814_2026-07-28.json.sha256" \
    -print0 |
    LC_ALL=C sort -z |
    while IFS= read -r -d '' file; do sha256sum "$file"; done
)"
if ! cmp -s \
  "$STAGE_B_TREE_MANIFEST" \
  <(printf '%s\n' "$recomputed_stage_b_manifest"); then
  fail "Stage-B tree manifest differs from the exact staged sealed inventory"
fi
(
  cd "$STAGE_B_LICENSE_DIR"
  sha256sum --quiet --strict --check "$STAGE_B_TREE_MANIFEST"
) || fail "Stage-B sealed tree all-file hash verification failed"
read -r stage_b_remote_file_count stage_b_remote_size_bytes < <(
  cd "$STAGE_B_LICENSE_DIR"
  cut -d' ' -f3- "$STAGE_B_TREE_MANIFEST" |
    while IFS= read -r file; do stat -c '%s' "$file"; done |
    awk '{ count += 1; bytes += $1 } END { print count + 0, bytes + 0 }'
)
if [[ "$stage_b_remote_file_count" != "$EXPECTED_STAGE_B_REMOTE_FILE_COUNT" ||
  "$stage_b_remote_size_bytes" != "$EXPECTED_STAGE_B_REMOTE_SIZE_BYTES" ]]; then
  fail "Stage-B sealed remote tree count or bytes changed"
fi
for binding in \
  "$EXPECTED_STAGE_B_ADJUDICATION_SHA256  ./artifacts/historical_k10000_replay_adjudication.json" \
  "$EXPECTED_STAGE_B_A_JSON_SHA256  ./artifacts/historical_k10000_replay_lane_A.json" \
  "$EXPECTED_STAGE_B_NPZ_SHA256  ./artifacts/historical_k10000_replay_lane_A.npz" \
  "$EXPECTED_STAGE_B_B_JSON_SHA256  ./artifacts/historical_k10000_replay_lane_B.json" \
  "$EXPECTED_STAGE_B_NPZ_SHA256  ./artifacts/historical_k10000_replay_lane_B.npz" \
  "$EXPECTED_LEGACY_PRODUCER_SHA256  ./drivaerml_historical_k10000_replay.py" \
  "$EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256  ./drivaerml_historical_k10000_replay_adjudicate.py"; do
  if ! grep -Fqx "$binding" "$STAGE_B_TREE_MANIFEST"; then
    fail "Stage-B seal lacks required binding: $binding"
  fi
done

"$PYTHON" - \
  "$PREREGISTRATION" \
  "$JOB306114_CORRECTION" \
  "$STAGE_B_ADJUDICATION" \
  "$STAGE_B_RESULT" \
  "$EXPECTED_TASK_PHYSICAL" \
  "$EXPECTED_CANONICAL_PRODUCER_SHA256" \
  "$EXPECTED_LEGACY_PRODUCER_SHA256" \
  "$EXPECTED_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_CANONICAL_HELPER_SHA256" \
  "$EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256" \
  "$EXPECTED_PAIRED_REDUCER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  "$EXPECTED_TARGET_MANIFEST_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256" \
  "$EXPECTED_JOB306114_CORRECTION_SHA256" \
  "$EXPECTED_STAGE_B_ADJUDICATION_SHA256" \
  "$EXPECTED_STAGE_B_RESULT_SHA256" \
  "$EXPECTED_STAGE_B_TREE_MANIFEST_SHA256" \
  "$EXPECTED_STAGE_B_A_JSON_SHA256" \
  "$EXPECTED_STAGE_B_B_JSON_SHA256" \
  "$EXPECTED_STAGE_B_NPZ_SHA256" <<'PY'
import json
import os
import stat
import sys


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON token: {value}")


def read_regular_nofollow(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ValueError(f"file changed while read: {path}")
    return b"".join(chunks)


def strict_json(path):
    return json.loads(
        read_regular_nofollow(path),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


(
    preregistration_path,
    correction_path,
    stage_b_adjudication_path,
    stage_b_result_path,
    task_physical,
    canonical_producer_sha,
    legacy_producer_sha,
    runtime_helper_sha,
    canonical_helper_sha,
    stage_b_reducer_sha,
    paired_reducer_sha,
    source_manifest_sha,
    source_tree_sha,
    geometry_manifest_sha,
    target_manifest_sha,
    input_freeze_sha,
    correction_sha,
    stage_b_adjudication_sha,
    stage_b_result_sha,
    stage_b_tree_sha,
    stage_b_a_json_sha,
    stage_b_b_json_sha,
    stage_b_npz_sha,
) = sys.argv[1:]

document = strict_json(preregistration_path)
if (
    document.get("schema_version") != 1
    or document.get("artifact_kind")
    != "phase1_historical_k10000_paired_accuracy_preregistration"
    or document.get("status") != "PREREGISTERED_PRELAUNCH"
):
    raise ValueError("paired-accuracy preregistration identity changed")

frozen = document.get("frozen_provenance", {})
expected_frozen = {
    "canonical_producer_sha256": canonical_producer_sha,
    "legacy_sentinel_producer_sha256": legacy_producer_sha,
    "runtime_helper_sha256": runtime_helper_sha,
    "canonical_geometry_helper_sha256": canonical_helper_sha,
    "stage_b_reducer_support_sha256": stage_b_reducer_sha,
    "paired_reducer_sha256": paired_reducer_sha,
    "execution_source_manifest_sha256": source_manifest_sha,
    "execution_source_tree_sha256": source_tree_sha,
    "dataset_manifest_sha256": "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca",
    "dataset_config_sha256": "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3",
    "resolved_config_sha256": "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1",
    "checkpoint_load_epoch": 491,
    "model_checkpoint_sha256": "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88",
    "training_state_sha256": "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7",
    "normalization_state_sha256": "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94",
    "current_infer_source_sha256": "47aec675e54d58ee4202831ae0d20039b1ff5ec40e69cc8b5087ce00bd5234ed",
    "current_model_source_sha256": "9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892",
    "target_free_geometry_manifest_sha256": geometry_manifest_sha,
    "selected_target_manifest_sha256": target_manifest_sha,
    "historical_input_freeze_record_sha256": input_freeze_sha,
    "job_306114_checkpoint_correction_sha256": correction_sha,
    "stage_b_adjudication_sha256": stage_b_adjudication_sha,
    "stage_b_result_record_sha256": stage_b_result_sha,
    "stage_b_sealed_task_tree_manifest_sha256": stage_b_tree_sha,
    "stage_b_lane_a_json_sha256": stage_b_a_json_sha,
    "stage_b_lane_b_json_sha256": stage_b_b_json_sha,
    "stage_b_lane_npz_sha256": stage_b_npz_sha,
}
if frozen != expected_frozen:
    raise ValueError("paired-accuracy frozen provenance changed")

canonical_isolation = document.get("canonical_producer_isolation", {})
for key in (
    "producer_reads_selected_targets",
    "producer_reads_stage_b_predictions_or_truths",
    "producer_reads_metrics_or_ceilings",
    "producer_reads_categorical_outcomes",
):
    if canonical_isolation.get(key) is not False:
        raise ValueError(f"canonical target-blind contract changed for {key}")
if "not an operating-system filesystem sandbox" not in canonical_isolation.get(
    "isolation_semantics",
    "",
):
    raise ValueError("canonical isolation semantics became overstrong")

endpoints = document.get("deciding_endpoints", {})
expected_baselines = {
    "uniform_pressure_relative_l2": 0.1671630996206787,
    "uniform_wss_frobenius_relative_l2": 0.27544940646582583,
    "archive_normalized_area_weighted_pressure_relative_l2": 0.18192540063679644,
    "archive_normalized_area_weighted_wss_frobenius_relative_l2": 0.21081553813258389,
}
expected_ceilings = {
    "uniform_pressure_relative_l2": 0.1705063616130923,
    "uniform_wss_frobenius_relative_l2": 0.2809583945951423,
    "archive_normalized_area_weighted_pressure_relative_l2": 0.18556390864953237,
    "archive_normalized_area_weighted_wss_frobenius_relative_l2": 0.21503184889523558,
}
if (
    endpoints.get("noninferiority_ratio") != 1.02
    or endpoints.get("baseline_means") != expected_baselines
    or endpoints.get("prospective_absolute_ceilings") != expected_ceilings
    or endpoints.get("all_four_required") is not True
    or endpoints.get("casewise_deciding_cutoff") is not None
):
    raise ValueError("four fixed-cohort deciding endpoints changed")

scope = document.get("scientific_scope", {})
excluded = set(scope.get("does_not_support", []))
for phrase in (
    "population-level statistical noninferiority",
    "OOD or population generalization",
    "superiority",
    "a single-tensor causal claim",
):
    if phrase not in excluded:
        raise ValueError(f"scientific scope no longer excludes {phrase}")

lanes = document.get("lane_contract", {})
if (
    lanes.get("processes")
    != "three concurrent, separate one-rank torchrun process trees under env -i and setsid"
    or lanes.get("cuda_visible_devices")
    != {
        "canonical_A": "0",
        "canonical_B": "1",
        "fresh_legacy_sentinel_A": "2",
    }
    or lanes.get("canonical_array_count_per_lane") != 792
    or lanes.get("fresh_legacy_sentinel", {}).get("array_count") != 720
    or "instrument invalidity only"
    not in lanes.get("fresh_legacy_sentinel", {}).get("failure_semantics", "")
):
    raise ValueError("three-lane execution contract changed")

expected_outcomes = [
    "CANONICAL_NONINFERIORITY_PASS",
    "VALID_CANONICAL_NONINFERIORITY_REFUTATION",
    "INVALID_CANONICAL_COMPARISON",
    "INCOMPLETE_CANONICAL_COMPARISON",
]
truth_table = document.get("adjudication", {}).get("truth_table", [])
if [row.get("outcome") for row in truth_table] != expected_outcomes:
    raise ValueError("paired-accuracy truth table changed")
if set(document.get("adjudication", {}).get("execution_exit_semantics", {})) != set(
    expected_outcomes
):
    raise ValueError("paired-accuracy exit semantics changed")

aga = document.get("aga_execution_contract", {})
expected_aga = {
    "task_physical": task_physical,
    "job_name": "mt-k10k-paired-v1",
    "nodes": 1,
    "scheduler_gpu_allocation": 4,
    "payload_gpu_count": 3,
    "payload_gpu_indices": [0, 1, 2],
    "cpus_per_task": 24,
    "memory": "256G",
    "slurm_export": "NIL",
    "wall_time": "00:45:00",
    "lane_timeout": "32m",
    "reducer_timeout": "8m",
    "immutable_outputs": True,
    "task_lock_required": True,
    "status_marker_on_shell_exit": True,
    "done_marker_only_for_pass_or_valid_refutation": True,
}
for key, expected in expected_aga.items():
    if aga.get(key) != expected:
        raise ValueError(f"AGA execution contract changed for {key}")
if any(document.get("launch_state", {}).values()):
    raise ValueError("preregistration no longer describes a prelaunch state")

correction = strict_json(correction_path)
if (
    correction.get("artifact_kind")
    != "phase1_hqc_canonical_geometry_checkpoint_binding_correction"
    or correction.get("status") != "CHECKPOINT_BINDING_LICENSE_WITHDRAWN"
    or correction.get("affected_experiment", {}).get("job_id") != "306114"
    or "The saved job-306114 predictions may serve as the canonical arm of an epoch-491 accuracy or noninferiority comparison."
    not in correction.get("withdrawn_claims", [])
    or "Adjudicate against the sealed exact Stage B v2 legacy baseline without reusing job-306114 predictions."
    not in correction.get("required_repair", [])
):
    raise ValueError("job-306114 prediction exclusion record changed")
exclusion = document.get("job_306114_exclusion", {})
if (
    exclusion.get("correction_record_sha256") != correction_sha
    or exclusion.get("forbidden_input") != "every saved job-306114 prediction"
):
    raise ValueError("job-306114 exclusion contract changed")

stage_b_adjudication = strict_json(stage_b_adjudication_path)
stage_b_result = strict_json(stage_b_result_path)
if (
    stage_b_adjudication.get("decision_outcome") != "EXACT_HISTORICAL_REPLAY_PASS"
    or stage_b_adjudication.get("status")
    != "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
    or stage_b_adjudication.get("corrected_stage2_baseline", {}).get(
        "prospective_absolute_ceilings"
    )
    != expected_ceilings
):
    raise ValueError("Stage-B exact adjudication or ceilings changed")
if (
    stage_b_result.get("job", {}).get("job_id") != 306814
    or stage_b_result.get("outcome", {}).get("categorical")
    != "EXACT_HISTORICAL_REPLAY_PASS"
    or stage_b_result.get("sealed_local_copy", {}).get("tree_manifest_sha256")
    != stage_b_tree_sha
    or stage_b_result.get("license", {}).get("status")
    != "ACTIVE_FOR_SEPARATE_PAIRED_CANONICAL_COMPARISON_ONLY"
    or stage_b_result.get("independent_postrun_artifact_audit") != "GO"
    or stage_b_result.get("independent_postrun_metrics_audit") != "GO"
):
    raise ValueError("Stage-B result record does not license this comparison")

PY

visible_gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if [[ "$visible_gpu_count" != 4 ]]; then
  fail "AGA whole-node allocation must expose four GPUs, observed $visible_gpu_count"
fi

readonly LOCK_PATH="$TASK_DIR/.paired-accuracy-v1.lock"
if [[ -L "$LOCK_PATH" ]]; then
  fail "task lock path is a symlink"
fi
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  fail "another paired-accuracy run holds the task lock"
fi

for path in \
  "$CANONICAL_A_JSON" \
  "$CANONICAL_A_JSON.sha256" \
  "$CANONICAL_A_NPZ" \
  "$CANONICAL_A_NPZ.sha256" \
  "$CANONICAL_B_JSON" \
  "$CANONICAL_B_JSON.sha256" \
  "$CANONICAL_B_NPZ" \
  "$CANONICAL_B_NPZ.sha256" \
  "$SENTINEL_JSON" \
  "$SENTINEL_JSON.sha256" \
  "$SENTINEL_NPZ" \
  "$SENTINEL_NPZ.sha256" \
  "$CANONICAL_A_LOG" \
  "$CANONICAL_B_LOG" \
  "$SENTINEL_LOG" \
  "$ADJUDICATION" \
  "$ADJUDICATION.sha256" \
  "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
  "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
  "$TASK_DIR/STATUS_${SLURM_JOB_ID}"; do
  check_absent "$path"
done

set -o noclobber
export PYTHONPATH="$TASK_DIR:$EXECUTION_SOURCE:$RECIPE/src"
phase="INFERENCE"
echo "CHECKPOINT_SELECTION epoch=491 model=$(basename "$MODEL_CHECKPOINT") training_state=$(basename "$TRAINING_STATE")"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SOURCE_TREE_SHA256=$actual_source_tree"
echo "STAGE_B_LICENSE_FILES=$stage_b_license_file_count bytes=$stage_b_license_size_bytes"
env -i \
  PATH="$PATH" \
  PYTHONPATH="$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  "$PYTHON" -c \
  'import numpy, physicsnemo, torch; print("VERSIONS", numpy.__version__, torch.__version__, physicsnemo.__file__); print("CUDA", torch.cuda.is_available(), torch.cuda.device_count())'

(
  while true; do
    echo \
      "GPU_HEARTBEAT $(date -Is) completed_units=$(heartbeat_completed_units)/${UNIT_COUNT}"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits ||
      echo "ERROR: nvidia-smi heartbeat failed" >&2
    sleep 120
  done
) &
heartbeat_pid=$!

# BEGIN CANONICAL_LANES
(
  cd "$DATASET_PHYSICAL"
  exec env -i \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$CANONICAL_PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --dataset-config "$DATASET_CONFIG" \
    --resolved-config "$RESOLVED_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --lane-label A \
    --output-json "$CANONICAL_A_JSON" \
    --output-npz "$CANONICAL_A_NPZ"
) > "$CANONICAL_A_LOG" 2>&1 &
lane_pids[0]=$!

(
  cd "$DATASET_PHYSICAL"
  exec env -i \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$CANONICAL_PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --dataset-config "$DATASET_CONFIG" \
    --resolved-config "$RESOLVED_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --lane-label B \
    --output-json "$CANONICAL_B_JSON" \
    --output-npz "$CANONICAL_B_NPZ"
) > "$CANONICAL_B_LOG" 2>&1 &
lane_pids[1]=$!
# END CANONICAL_LANES

# BEGIN LEGACY_SENTINEL_LANE
(
  cd "$DATASET_PHYSICAL"
  exec env -i \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$LEGACY_PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --resolved-config "$RESOLVED_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --target-input-manifest "$TARGET_INPUT_MANIFEST" \
    --replay-label A \
    --output-json "$SENTINEL_JSON" \
    --output-npz "$SENTINEL_NPZ"
) > "$SENTINEL_LOG" 2>&1 &
lane_pids[2]=$!
# END LEGACY_SENTINEL_LANE

declare -a lane_rcs=()
set +e
for lane_index in 0 1 2; do
  wait "${lane_pids[$lane_index]}"
  lane_rcs[lane_index]=$?
  lane_pids[lane_index]=""
done
set -e

lane_artifacts_valid=true
for lane_index in 0 1 2; do
  if [[ "${lane_rcs[$lane_index]}" -ne 0 ]]; then
    echo "LANE_EXIT lane=$lane_index rc=${lane_rcs[$lane_index]}" >&2
    lane_artifacts_valid=false
  fi
done
for path in "$CANONICAL_A_JSON" "$CANONICAL_A_NPZ"; do
  if ! verify_canonical_sidecar "$path"; then
    echo "CANONICAL_A_ARTIFACT_INVALID path=$path" >&2
    lane_artifacts_valid=false
  fi
done
for path in "$CANONICAL_B_JSON" "$CANONICAL_B_NPZ"; do
  if ! verify_canonical_sidecar "$path"; then
    echo "CANONICAL_B_ARTIFACT_INVALID path=$path" >&2
    lane_artifacts_valid=false
  fi
done
for path in "$SENTINEL_JSON" "$SENTINEL_NPZ"; do
  if ! verify_canonical_sidecar "$path"; then
    echo "SENTINEL_ARTIFACT_INVALID path=$path" >&2
    lane_artifacts_valid=false
  fi
done
if [[ "$lane_artifacts_valid" != true ]]; then
  phase="BLOCKED_BEFORE_REDUCER"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "INCOMPLETE_OR_INVALID_LANE_ARTIFACT; REDUCER_NOT_RUN"
  tail -n 80 "$CANONICAL_A_LOG" >&2 || true
  tail -n 80 "$CANONICAL_B_LOG" >&2 || true
  tail -n 80 "$SENTINEL_LOG" >&2 || true
  exit 20
fi
completed_units=3

phase="REDUCTION"
echo "REDUCER_START $(date -Is)"
set +e
env -i \
  PATH="$PATH" \
  PYTHONPATH="$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 8m \
  "$PYTHON" "$PAIRED_REDUCER" \
  --canonical-producer "$CANONICAL_PRODUCER" \
  --canonical-a-json "$CANONICAL_A_JSON" \
  --canonical-a-npz "$CANONICAL_A_NPZ" \
  --canonical-b-json "$CANONICAL_B_JSON" \
  --canonical-b-npz "$CANONICAL_B_NPZ" \
  --legacy-producer "$LEGACY_PRODUCER" \
  --fresh-legacy-sentinel-json "$SENTINEL_JSON" \
  --fresh-legacy-sentinel-npz "$SENTINEL_NPZ" \
  --sealed-stage-b-a-json "$STAGE_B_A_JSON" \
  --sealed-stage-b-a-npz "$STAGE_B_A_NPZ" \
  --sealed-stage-b-b-json "$STAGE_B_B_JSON" \
  --sealed-stage-b-b-npz "$STAGE_B_B_NPZ" \
  --sealed-stage-b-adjudication "$STAGE_B_ADJUDICATION" \
  --sealed-stage-b-result "$STAGE_B_RESULT" \
  --sealed-stage-b-tree-manifest "$STAGE_B_TREE_MANIFEST" \
  --output-json "$ADJUDICATION"
reducer_rc=$?
set -e
case "$reducer_rc" in
  0 | 2 | 3) ;;
  *)
    phase="BLOCKED_REDUCER_EXECUTION"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "REDUCER_EXECUTION_FAILED rc=$reducer_rc"
    exit 21
    ;;
esac
if ! verify_canonical_sidecar "$ADJUDICATION"; then
  phase="BLOCKED_REDUCER_ARTIFACT"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_ARTIFACT_MISSING_INVALID_OR_NONCANONICAL"
  exit 22
fi

decision="$(
  "$PYTHON" - "$ADJUDICATION" <<'PY'
import json
import math
import sys


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate adjudication key: {key}")
        result[key] = value
    return result


with open(sys.argv[1], "rb") as stream:
    document = json.load(
        stream,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {value}")
        ),
    )
if document.get("artifact_kind") != (
    "phase1_historical_k10000_paired_accuracy_adjudication"
):
    raise ValueError("adjudication artifact kind changed")
outcome = document.get("decision_outcome")
status = document.get("status")
allowed = {
    "CANONICAL_NONINFERIORITY_PASS": (
        "VALID_CANONICAL_NONINFERIORITY_ADJUDICATION",
        0,
    ),
    "VALID_CANONICAL_NONINFERIORITY_REFUTATION": (
        "VALID_CANONICAL_NONINFERIORITY_ADJUDICATION",
        0,
    ),
    "INVALID_CANONICAL_COMPARISON": (
        "INVALID_CANONICAL_NONINFERIORITY_ADJUDICATION",
        2,
    ),
    "INCOMPLETE_CANONICAL_COMPARISON": (
        "INCOMPLETE_CANONICAL_NONINFERIORITY_ADJUDICATION",
        3,
    ),
}
if outcome not in allowed or (status, allowed[outcome][1]) != allowed[outcome]:
    raise ValueError(
        f"unrecognized adjudication status/outcome: {status!r}/{outcome!r}"
    )
if outcome in {
    "CANONICAL_NONINFERIORITY_PASS",
    "VALID_CANONICAL_NONINFERIORITY_REFUTATION",
}:
    gates = document.get("decision_gates", {})
    for key in (
        "stage_b_exact_license_active",
        "sealed_stage_b_replicas_raw_byte_exact",
        "fresh_legacy_sentinel_exact_to_both_sealed_lanes",
        "canonical_replicas_all_arrays_raw_byte_exact_including_signed_zero",
        "canonical_processes_distinct_and_gpu_pinned",
        "all_precanonical_input_and_frame_controls_passed",
        "stage_b_baseline_independently_recomputed",
    ):
        if gates.get(key) is not True:
            raise ValueError(f"valid outcome lacks required gate {key}")
    if gates.get("per_case_deciding_gates_present") is not False:
        raise ValueError("casewise deciding gate appeared")
    expected_ceilings = {
        "uniform_pressure_relative_l2": 0.1705063616130923,
        "uniform_wss_frobenius_relative_l2": 0.2809583945951423,
        "archive_normalized_area_weighted_pressure_relative_l2": 0.18556390864953237,
        "archive_normalized_area_weighted_wss_frobenius_relative_l2": 0.21503184889523558,
    }
    endpoint_gates = gates.get("four_inclusive_noninferiority_endpoints", {})
    if set(endpoint_gates) != set(expected_ceilings):
        raise ValueError("four endpoint gate names changed")
    passed = []
    for name, ceiling in expected_ceilings.items():
        record = endpoint_gates[name]
        value = record.get("canonical_mean")
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or record.get("inclusive_ceiling") != ceiling
            or record.get("passed") is not (value <= ceiling)
        ):
            raise ValueError(f"endpoint gate is inconsistent for {name}")
        passed.append(record["passed"])
    if (
        gates.get("all_four_noninferiority_endpoints_passed") is not all(passed)
        or (outcome == "CANONICAL_NONINFERIORITY_PASS") is not all(passed)
    ):
        raise ValueError("categorical outcome differs from the four endpoint gates")
    scope = document.get("scientific_scope", {})
    forbidden = set(scope.get("does_not_support", []))
    for phrase in (
        "superiority",
        "other resolutions",
        "OOD or population generalization",
        "causal architecture or H-QC mechanism claims",
    ):
        if phrase not in forbidden:
            raise ValueError(f"valid artifact overclaims {phrase}")
else:
    if not document.get("failures"):
        raise ValueError("invalid or incomplete artifact lacks a failure record")
print(f"{status}\t{outcome}\t{allowed[outcome][1]}")
PY
)" || {
  phase="BLOCKED_REDUCER_ARTIFACT"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_ARTIFACT_UNREADABLE_OR_UNRECOGNIZED"
  exit 23
}
IFS=$'\t' read -r adjudication_status categorical_outcome expected_reducer_rc <<< "$decision"
if [[ "$reducer_rc" != "$expected_reducer_rc" ]]; then
  phase="BLOCKED_REDUCER_EXIT_MISMATCH"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_EXIT_MISMATCH observed=$reducer_rc expected=$expected_reducer_rc"
  exit 24
fi
completed_units=4
echo "PAIRED_ACCURACY_ADJUDICATION_STATUS=$adjudication_status"

case "$categorical_outcome" in
  CANONICAL_NONINFERIORITY_PASS)
    phase="COMPLETED_FIXED_COHORT_PASS"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; fixed 36-case epoch-491 K=10000 result only"
    ;;
  VALID_CANONICAL_NONINFERIORITY_REFUTATION)
    phase="COMPLETED_VALID_REFUTATION"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; successful scientific refutation; do not convert to infrastructure failure"
    ;;
  INVALID_CANONICAL_COMPARISON)
    phase="BLOCKED_INVALID_INSTRUMENT"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; repair and rerun paired accuracy"
    exit 30
    ;;
  INCOMPLETE_CANONICAL_COMPARISON)
    phase="BLOCKED_INCOMPLETE_COMPARISON"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; complete or rerun paired accuracy"
    exit 31
    ;;
  *)
    phase="BLOCKED_UNRECOGNIZED_OUTCOME"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "UNRECOGNIZED_REDUCER_OUTCOME=$categorical_outcome"
    exit 32
    ;;
esac

echo "PAIRED_HISTORICAL_K10000_ACCURACY_DONE $(date -Is)"
