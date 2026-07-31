#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-stageb-v2
#SBATCH --time=00:45:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
# AGA schedules whole four-GPU nodes; this payload deliberately uses only GPUs 0 and 1.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-b-v2/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-b-v2/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly EXPECTED_TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-b-v2
readonly EXPECTED_TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-b-v2
readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly DATASET_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
readonly HISTORICAL_ARCHIVE_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-coverage-sweep/artifacts/res10000/t2_mesh_transformer_surface_flagship_seed42/predictions

readonly EXPECTED_PRODUCER_SHA256=bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5
readonly EXPECTED_HELPER_SHA256=dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896
readonly EXPECTED_REDUCER_SHA256=4b4f4c893ca154b589b9d1276bf4f43cd52008664f75f3c56fcc66aa26ce8f68
readonly EXPECTED_PREREGISTRATION_SHA256=f76931326abc618f5f352f799c342913992f6e8f8a2b8968e0c9f453e73f5912
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
readonly EXPECTED_HISTORICAL_MANIFEST_SHA256=545b1f6e906002231415b84277db00eec04f3666233b8da637514e9077a585eb
readonly EXPECTED_HISTORICAL_METRICS_SHA256=423ec28e0212f0762ea814e6179da2b7a9a1feb95011b4b83c06605835b7c43a
readonly EXPECTED_INPUT_FREEZE_SHA256=fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10
readonly EXPECTED_STAGE_A_LICENSE_SHA256=e309ca89ae7f2506797e871e6bac44a808a5a6fc224346b6b92495f6d06da3e5
readonly EXPECTED_STAGE_A_TREE_MANIFEST_SHA256=deb2a02e5567474f1134ea8f646d965bb1c6811c0fc3e1cf307bf09b323c759b
readonly EXPECTED_STAGE_A_RESULT_RECORD_SHA256=a845ad02217fe0aedaa782ae1f842bba714fdfb09d731dac01f9ed7ac42b2684
readonly EXPECTED_STAGE_A_PRODUCER_SHA256=b596cb3d4a82b30255324b982f5d84d1260f53963b1697c7b2fd9d12049ed8c0
readonly EXPECTED_STAGE_A_REDUCER_SHA256=fdb232e337eeb54be4f1782f6b619f53e5ad4ed0c2aca366b504d035cab023e0
readonly EXPECTED_HISTORICAL_FILE_COUNT=1656
readonly EXPECTED_HISTORICAL_SIZE_BYTES=44790588
readonly UNIT_COUNT=3

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
  echo "STAGE_B_CATEGORICAL_OUTCOME=${categorical_outcome}"
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
  if verify_canonical_sidecar "$LANE_A_JSON" &&
    verify_canonical_sidecar "$LANE_A_NPZ"; then
    observed=$((observed + 1))
  fi
  if verify_canonical_sidecar "$LANE_B_JSON" &&
    verify_canonical_sidecar "$LANE_B_NPZ"; then
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
  fail "Stage B requires exactly one allocated node"
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
for directory in \
  "$OUTPUT_DIR" \
  "$LANE_LOG_DIR" \
  "$SBATCH_LOG_DIR" \
  "$EXECUTION_SOURCE"; do
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    fail "required staged directory is missing or is a symlink: $directory"
  fi
done
if [[ "$(cd -P "$EXECUTION_SOURCE" && pwd)" != "$TASK_DIR/execution_source" ]]; then
  fail "execution source is not the task-local physical snapshot"
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

readonly PRODUCER="$TASK_DIR/drivaerml_historical_k10000_replay.py"
readonly HELPER="$TASK_DIR/drivaerml_historical_k10000_replay_runtime.py"
readonly REDUCER="$TASK_DIR/drivaerml_historical_k10000_replay_adjudicate.py"
readonly PREREGISTRATION="$TASK_DIR/phase1_historical_k10000_stage_b_replay_prereg_v2_2026-07-28.json"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
readonly GEOMETRY_MANIFEST="$TASK_DIR/drivaerml_geometry_input_manifest_36cases_v1.json"
readonly TARGET_INPUT_MANIFEST="$TASK_DIR/historical_k10000_selected_target_input_manifest_v1.json"
readonly HISTORICAL_MANIFEST="$TASK_DIR/historical_predictions_tree_manifest.sha256"
readonly HISTORICAL_METRICS="$TASK_DIR/hcc_historical_res10000_metrics_2026-07-24.jsonl"
readonly INPUT_FREEZE="$TASK_DIR/historical_k10000_input_freeze_v1.json"
readonly STAGE_A_LICENSE_DIR="$TASK_DIR/stage_a_v2_license"
readonly STAGE_A_LICENSE="$STAGE_A_LICENSE_DIR/artifacts/stage_a_archive_canary_adjudication.json"
readonly STAGE_A_TREE_MANIFEST="$STAGE_A_LICENSE_DIR/sealed_remote_task_tree_manifest.sha256"

readonly LANE_A_JSON="$OUTPUT_DIR/historical_k10000_replay_lane_A.json"
readonly LANE_A_NPZ="$OUTPUT_DIR/historical_k10000_replay_lane_A.npz"
readonly LANE_B_JSON="$OUTPUT_DIR/historical_k10000_replay_lane_B.json"
readonly LANE_B_NPZ="$OUTPUT_DIR/historical_k10000_replay_lane_B.npz"
readonly LANE_A_LOG="$LANE_LOG_DIR/lane_A.log"
readonly LANE_B_LOG="$LANE_LOG_DIR/lane_B.log"
readonly ADJUDICATION="$OUTPUT_DIR/historical_k10000_replay_adjudication.json"

if [[ ! -d "$STAGE_A_LICENSE_DIR" || -L "$STAGE_A_LICENSE_DIR" ]]; then
  fail "Stage-A v2 license directory is missing or is a symlink"
fi
if [[ "$(cd -P "$STAGE_A_LICENSE_DIR" && pwd)" != "$TASK_DIR/stage_a_v2_license" ]]; then
  fail "Stage-A v2 license directory resolves outside the staged task"
fi

resolved_python="$(readlink -f "$PYTHON")"
if [[ ! -x "$PYTHON" || ! -f "$resolved_python" ]]; then
  fail "shared physical venv interpreter is unavailable: $PYTHON"
fi

for expected in \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_HELPER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
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
  "$EXPECTED_HISTORICAL_MANIFEST_SHA256" \
  "$EXPECTED_HISTORICAL_METRICS_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256" \
  "$EXPECTED_STAGE_A_LICENSE_SHA256" \
  "$EXPECTED_STAGE_A_TREE_MANIFEST_SHA256" \
  "$EXPECTED_STAGE_A_RESULT_RECORD_SHA256" \
  "$EXPECTED_STAGE_A_PRODUCER_SHA256" \
  "$EXPECTED_STAGE_A_REDUCER_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_HELPER_SHA256" "$HELPER"
check_sha "$EXPECTED_REDUCER_SHA256" "$REDUCER"
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
check_sha "$EXPECTED_HISTORICAL_MANIFEST_SHA256" "$HISTORICAL_MANIFEST"
check_sha "$EXPECTED_HISTORICAL_METRICS_SHA256" "$HISTORICAL_METRICS"
check_sha "$EXPECTED_INPUT_FREEZE_SHA256" "$INPUT_FREEZE"
check_sha "$EXPECTED_STAGE_A_LICENSE_SHA256" "$STAGE_A_LICENSE"
check_sha "$EXPECTED_STAGE_A_TREE_MANIFEST_SHA256" "$STAGE_A_TREE_MANIFEST"
if ! verify_canonical_sidecar "$TARGET_INPUT_MANIFEST"; then
  fail "selected-target manifest sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$GEOMETRY_MANIFEST"; then
  fail "geometry manifest sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$HISTORICAL_MANIFEST"; then
  fail "historical manifest sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$HISTORICAL_METRICS"; then
  fail "historical metrics sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$INPUT_FREEZE"; then
  fail "historical input-freeze sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$STAGE_A_LICENSE"; then
  fail "Stage-A v2 adjudication sidecar is missing, stale, or noncanonical"
fi
if ! verify_canonical_sidecar "$STAGE_A_TREE_MANIFEST"; then
  fail "Stage-A v2 tree-manifest sidecar is missing, stale, or noncanonical"
fi
if ! grep -Fqx \
  "$EXPECTED_STAGE_A_LICENSE_SHA256  ./artifacts/stage_a_archive_canary_adjudication.json" \
  "$STAGE_A_TREE_MANIFEST"; then
  fail "Stage-A v2 seal does not bind the exact adjudication artifact"
fi

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

"$PYTHON" - \
  "$PREREGISTRATION" \
  "$STAGE_A_LICENSE" \
  "$EXPECTED_TASK_PHYSICAL" \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_HELPER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  "$EXPECTED_TARGET_MANIFEST_SHA256" \
  "$EXPECTED_HISTORICAL_MANIFEST_SHA256" \
  "$EXPECTED_HISTORICAL_METRICS_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256" \
  "$EXPECTED_STAGE_A_LICENSE_SHA256" \
  "$EXPECTED_STAGE_A_TREE_MANIFEST_SHA256" \
  "$EXPECTED_STAGE_A_RESULT_RECORD_SHA256" \
  "$EXPECTED_STAGE_A_PRODUCER_SHA256" \
  "$EXPECTED_STAGE_A_REDUCER_SHA256" <<'PY'
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
    stage_a_license_path,
    task_physical,
    producer_sha,
    helper_sha,
    reducer_sha,
    source_manifest_sha,
    source_tree_sha,
    geometry_manifest_sha,
    target_manifest_sha,
    historical_manifest_sha,
    historical_metrics_sha,
    input_freeze_sha,
    stage_a_license_sha,
    stage_a_tree_manifest_sha,
    stage_a_result_record_sha,
    stage_a_producer_sha,
    stage_a_reducer_sha,
) = sys.argv[1:]

document = strict_json(preregistration_path)
if document.get("artifact_kind") != (
    "phase1_historical_k10000_stage_b_replay_preregistration"
):
    raise ValueError("preregistration artifact kind changed")
if document.get("status") != "PREREGISTERED_PRELAUNCH":
    raise ValueError("preregistration status changed")
if document.get("schema_version") != 2:
    raise ValueError("preregistration schema version changed")

frozen = document.get("frozen_provenance", {})
expected_frozen = {
    "producer_sha256": producer_sha,
    "runtime_helper_sha256": helper_sha,
    "reducer_sha256": reducer_sha,
    "execution_source_manifest_sha256": source_manifest_sha,
    "execution_source_tree_sha256": source_tree_sha,
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
    "historical_prediction_tree_manifest_sha256": historical_manifest_sha,
    "historical_metrics_sha256": historical_metrics_sha,
    "historical_input_freeze_record_sha256": input_freeze_sha,
    "stage_a_adjudication_sha256": stage_a_license_sha,
    "stage_a_sealed_task_tree_manifest_sha256": stage_a_tree_manifest_sha,
    "stage_a_result_record_sha256": stage_a_result_record_sha,
}
if frozen != expected_frozen:
    raise ValueError("preregistration frozen provenance changed")

stage_a_contract = document.get("stage_a_license", {})
if (
    stage_a_contract.get("required_outcome") != "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS"
    or stage_a_contract.get("job_id") != 306754
    or stage_a_contract.get("adjudication_sha256") != stage_a_license_sha
    or stage_a_contract.get("sealed_task_tree_manifest_sha256")
    != stage_a_tree_manifest_sha
    or stage_a_contract.get("result_record_sha256") != stage_a_result_record_sha
    or stage_a_contract.get("producer_sha256") != stage_a_producer_sha
    or stage_a_contract.get("reducer_sha256") != stage_a_reducer_sha
    or stage_a_contract.get("checkpoint_load_epoch") != 491
):
    raise ValueError("Stage-A license binding changed")
scope = document.get("scope", {})
for key in (
    "historical_predictions_passed_to_producer",
    "historical_predictions_opened_by_producer",
    "historical_metrics_passed_to_producer",
    "historical_metrics_opened_by_producer",
    "stage_a_license_passed_to_producer",
    "stage_a_license_opened_by_producer",
    "categorical_result_passed_to_producer",
    "categorical_result_opened_by_producer",
):
    if scope.get(key) is not False:
        raise ValueError(f"producer isolation contract changed for {key}")
if "not an operating-system filesystem sandbox" not in scope.get(
    "isolation_semantics",
    "",
):
    raise ValueError("producer isolation semantics became overstrong")
lane_contract = document.get("lane_contract", {})
if (
    lane_contract.get("process_isolation")
    != "two concurrent, separate one-rank torchrun processes"
    or lane_contract.get("cuda_visible_devices") != {"A": "0", "B": "1"}
    or lane_contract.get("case_count_per_lane") != 36
    or lane_contract.get("array_count_per_lane") != 720
):
    raise ValueError("two-lane replay contract changed")

expected_outcomes = [
    "EXACT_HISTORICAL_REPLAY_PASS",
    "VALID_EXACT_REPLAY_REFUTATION",
    "INVALID_REPLAY",
    "INCOMPLETE_REPLAY",
]
truth_table = document.get("adjudication", {}).get("truth_table", [])
if [row.get("outcome") for row in truth_table] != expected_outcomes:
    raise ValueError("preregistered truth-table outcomes changed")
exit_semantics = document.get("adjudication", {}).get(
    "execution_exit_semantics",
    {},
)
if set(exit_semantics) != set(expected_outcomes):
    raise ValueError("preregistered exit semantics changed")
if "must not be converted into infrastructure failure" not in exit_semantics.get(
    "VALID_EXACT_REPLAY_REFUTATION",
    "",
):
    raise ValueError("valid-refutation success semantics changed")

corrected = document.get("corrected_baseline_contract", {})
if (
    corrected.get("licensed_only_on_exact_outcome") is not True
    or corrected.get("prospective_noninferiority_ratio") != 1.02
    or "descriptive only"
    not in corrected.get("historical_pointwise_mean_wss_relative_l2", "")
    or "absolute force integration" not in corrected.get("does_not_support", [])
):
    raise ValueError("corrected baseline scope changed")

aga = document.get("aga_execution_contract", {})
expected_aga = {
    "task_physical": task_physical,
    "nodes": 1,
    "scheduler_gpu_allocation": 4,
    "payload_gpu_count": 2,
    "payload_gpu_indices": [0, 1],
    "slurm_export": "NIL",
    "wall_time": "00:45:00",
    "lane_timeout": "32m",
    "reducer_timeout": "8m",
    "immutable_outputs": True,
    "task_lock_required": True,
    "status_marker_on_shell_exit": True,
    "done_marker_only_for_exact_pass_or_valid_refutation": True,
}
for key, expected in expected_aga.items():
    if aga.get(key) != expected:
        raise ValueError(
            f"preregistration AGA contract changed for {key}: {aga.get(key)!r}"
        )
if any(document.get("launch_state", {}).values()):
    raise ValueError("preregistration no longer describes a prelaunch state")

license_document = strict_json(stage_a_license_path)
if (
    license_document.get("artifact_kind")
    != "phase1_historical_k10000_stage_a_archive_domain_adjudication"
    or license_document.get("status")
    != "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_ADJUDICATION"
    or license_document.get("outcome") != "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS"
    or license_document.get("provenance", {}).get("expected_producer_sha256")
    != stage_a_producer_sha
    or license_document.get("provenance", {}).get("reducer_sha256")
    != stage_a_reducer_sha
):
    raise ValueError("Stage-A artifact does not license Stage B")
process = license_document.get("comparisons", {}).get("process_isolation", {})
if (
    process.get("lane_a", {}).get("slurm_job_id") != "306754"
    or process.get("lane_b", {}).get("slurm_job_id") != "306754"
):
    raise ValueError("Stage-A adjudication does not attest job 306754")

PY

if [[ ! -d "$HISTORICAL_ARCHIVE_PHYSICAL" || -L "$HISTORICAL_ARCHIVE_PHYSICAL" ]]; then
  fail "historical archive root is missing or is a symlink"
fi
if [[ "$(cd -P "$HISTORICAL_ARCHIVE_PHYSICAL" && pwd)" != "$HISTORICAL_ARCHIVE_PHYSICAL" ]]; then
  fail "historical archive root is not the expected physical path"
fi
archive_symlink="$(find "$HISTORICAL_ARCHIVE_PHYSICAL" -type l -print -quit)"
if [[ -n "$archive_symlink" ]]; then
  fail "historical archive contains a symlink: $archive_symlink"
fi
archive_special="$(
  find "$HISTORICAL_ARCHIVE_PHYSICAL" \
    ! -type d ! -type f \
    -print -quit
)"
if [[ -n "$archive_special" ]]; then
  fail "historical archive contains a non-regular entry: $archive_special"
fi
read -r historical_file_count historical_size_bytes < <(
  find "$HISTORICAL_ARCHIVE_PHYSICAL" -type f -printf '%s\n' |
    awk '{ count += 1; bytes += $1 } END { print count + 0, bytes + 0 }'
)
if [[ "$historical_file_count" != "$EXPECTED_HISTORICAL_FILE_COUNT" ]]; then
  fail "historical archive file count changed: $historical_file_count"
fi
if [[ "$historical_size_bytes" != "$EXPECTED_HISTORICAL_SIZE_BYTES" ]]; then
  fail "historical archive byte count changed: $historical_size_bytes"
fi
if [[ "$(wc -l < "$HISTORICAL_MANIFEST")" != "$EXPECTED_HISTORICAL_FILE_COUNT" ]]; then
  fail "historical manifest line count changed"
fi
if ! cmp -s "$HISTORICAL_MANIFEST" <(
  cd "$HISTORICAL_ARCHIVE_PHYSICAL"
  find . -type f -print0 |
    LC_ALL=C sort -z |
    while IFS= read -r -d '' file; do sha256sum "$file"; done
); then
  fail "historical manifest differs from the exact archive inventory or content"
fi
(
  cd "$HISTORICAL_ARCHIVE_PHYSICAL"
  sha256sum --quiet --strict --check "$HISTORICAL_MANIFEST"
) || fail "historical archive all-file hash verification failed"

visible_gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if [[ "$visible_gpu_count" != 4 ]]; then
  fail "AGA whole-node allocation must expose four GPUs, observed $visible_gpu_count"
fi

readonly LOCK_PATH="$TASK_DIR/.stage-b-replay-v2.lock"
if [[ -L "$LOCK_PATH" ]]; then
  fail "task lock path is a symlink"
fi
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  fail "another Stage-B replay holds the task lock"
fi

for path in \
  "$LANE_A_JSON" \
  "$LANE_A_JSON.sha256" \
  "$LANE_A_NPZ" \
  "$LANE_A_NPZ.sha256" \
  "$LANE_B_JSON" \
  "$LANE_B_JSON.sha256" \
  "$LANE_B_NPZ" \
  "$LANE_B_NPZ.sha256" \
  "$LANE_A_LOG" \
  "$LANE_B_LOG" \
  "$ADJUDICATION" \
  "$ADJUDICATION.sha256" \
  "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
  "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
  "$TASK_DIR/STATUS_${SLURM_JOB_ID}"; do
  check_absent "$path"
done

# Preserve the preflight no-clobber decision across the subsequent log redirections.
set -o noclobber
export PYTHONPATH="$EXECUTION_SOURCE:$RECIPE/src"
phase="INFERENCE"
echo "CHECKPOINT_SELECTION epoch=491 model=$(basename "$MODEL_CHECKPOINT") training_state=$(basename "$TRAINING_STATE")"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SOURCE_TREE_SHA256=$actual_source_tree"
echo "HISTORICAL_ARCHIVE_FILES=$historical_file_count bytes=$historical_size_bytes"
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

# BEGIN PRODUCER_LANES
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
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    NUMEXPR_NUM_THREADS=12 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --resolved-config "$RESOLVED_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --target-input-manifest "$TARGET_INPUT_MANIFEST" \
    --replay-label A \
    --output-json "$LANE_A_JSON" \
    --output-npz "$LANE_A_NPZ"
) > "$LANE_A_LOG" 2>&1 &
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
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    NUMEXPR_NUM_THREADS=12 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --resolved-config "$RESOLVED_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --target-input-manifest "$TARGET_INPUT_MANIFEST" \
    --replay-label B \
    --output-json "$LANE_B_JSON" \
    --output-npz "$LANE_B_NPZ"
) > "$LANE_B_LOG" 2>&1 &
lane_pids[1]=$!
# END PRODUCER_LANES

declare -a lane_rcs=()
set +e
for lane_index in 0 1; do
  wait "${lane_pids[$lane_index]}"
  lane_rcs[lane_index]=$?
  lane_pids[lane_index]=""
done
set -e

lane_artifacts_valid=true
for lane_index in 0 1; do
  if [[ "${lane_rcs[$lane_index]}" -ne 0 ]]; then
    echo "LANE_EXIT lane=$lane_index rc=${lane_rcs[$lane_index]}" >&2
    lane_artifacts_valid=false
  fi
done
for path in "$LANE_A_JSON" "$LANE_A_NPZ"; do
  if ! verify_canonical_sidecar "$path"; then
    echo "LANE_A_ARTIFACT_INVALID path=$path" >&2
    lane_artifacts_valid=false
  fi
done
for path in "$LANE_B_JSON" "$LANE_B_NPZ"; do
  if ! verify_canonical_sidecar "$path"; then
    echo "LANE_B_ARTIFACT_INVALID path=$path" >&2
    lane_artifacts_valid=false
  fi
done
if [[ "$lane_artifacts_valid" != true ]]; then
  phase="BLOCKED_BEFORE_REDUCER"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "INCOMPLETE_OR_INVALID_LANE_ARTIFACT; REDUCER_NOT_RUN"
  tail -n 80 "$LANE_A_LOG" >&2 || true
  tail -n 80 "$LANE_B_LOG" >&2 || true
  exit 20
fi
completed_units=2

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
  "$PYTHON" "$REDUCER" \
  --producer "$PRODUCER" \
  --producer-a-json "$LANE_A_JSON" \
  --producer-a-npz "$LANE_A_NPZ" \
  --producer-b-json "$LANE_B_JSON" \
  --producer-b-npz "$LANE_B_NPZ" \
  --target-input-manifest "$TARGET_INPUT_MANIFEST" \
  --historical-predictions "$HISTORICAL_ARCHIVE_PHYSICAL" \
  --historical-manifest "$HISTORICAL_MANIFEST" \
  --historical-metrics "$HISTORICAL_METRICS" \
  --normalization-state "$NORMALIZATION_STATE" \
  --output-json "$ADJUDICATION"
reducer_rc=$?
set -e
if [[ "$reducer_rc" -ne 0 ]]; then
  phase="BLOCKED_REDUCER_EXECUTION"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_EXECUTION_FAILED rc=$reducer_rc"
  exit 21
fi
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
    "phase1_historical_k10000_replay_adjudication"
):
    raise ValueError("adjudication artifact kind changed")
outcome = document.get("decision_outcome")
status = document.get("status")
allowed = {
    "EXACT_HISTORICAL_REPLAY_PASS": (
        "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
    ),
    "VALID_EXACT_REPLAY_REFUTATION": (
        "VALID_HISTORICAL_K10000_REPLAY_ADJUDICATION"
    ),
    "INVALID_REPLAY": "INVALID_HISTORICAL_K10000_REPLAY_ADJUDICATION",
    "INCOMPLETE_REPLAY": "INCOMPLETE_HISTORICAL_K10000_REPLAY_ADJUDICATION",
}
if outcome not in allowed or status != allowed[outcome]:
    raise ValueError(
        f"unrecognized adjudication status/outcome: {status!r}/{outcome!r}"
    )
if outcome == "EXACT_HISTORICAL_REPLAY_PASS":
    baseline = document.get("corrected_stage2_baseline", {})
    gates = document.get("decision_gates", {})
    required_gates = {
        "historical_archive_exact",
        "current_replicas_exact",
        "selected_target_bytes_exact",
        "model_consumed_archive_fields_parity_passed",
        "archive_geometry_controls_passed",
        "training_physical_chain_controls_passed",
        "truth_chain_control_exact",
        "all_input_parity_passed",
        "all_historical_predictions_exact",
        "all_pressure_case_metrics_passed",
        "pressure_mean_passed",
    }
    if (
        baseline.get("licensed") is not True
        or baseline.get("metrics_available") is not True
        or baseline.get("case_count") != 36
        or baseline.get("noninferiority_ratio") != 1.02
        or any(gates.get(key) is not True for key in required_gates)
    ):
        raise ValueError("exact outcome does not satisfy its licensing gates")
print(f"{status}\t{outcome}")
PY
)" || {
  phase="BLOCKED_REDUCER_ARTIFACT"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_ARTIFACT_UNREADABLE_OR_UNRECOGNIZED"
  exit 23
}
IFS=$'\t' read -r adjudication_status categorical_outcome <<< "$decision"
completed_units=3
echo "STAGE_B_ADJUDICATION_STATUS=$adjudication_status"

case "$categorical_outcome" in
  EXACT_HISTORICAL_REPLAY_PASS)
    phase="COMPLETED_EXACT_PASS"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; licenses the corrected frozen legacy baseline for a separate canonical comparison"
    ;;
  VALID_EXACT_REPLAY_REFUTATION)
    phase="COMPLETED_VALID_REFUTATION"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; successful scientific refutation; canonical accuracy comparison remains blocked"
    ;;
  INVALID_REPLAY)
    phase="BLOCKED_INVALID_INSTRUMENT"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; repair and rerun Stage B"
    exit 30
    ;;
  INCOMPLETE_REPLAY)
    phase="BLOCKED_INCOMPLETE_REPLAY"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; complete or rerun Stage B"
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

echo "STAGE_B_HISTORICAL_REPLAY_DONE $(date -Is)"
