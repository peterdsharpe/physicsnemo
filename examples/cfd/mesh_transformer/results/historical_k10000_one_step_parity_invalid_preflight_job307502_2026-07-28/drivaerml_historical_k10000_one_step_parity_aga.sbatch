#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-step-parity-v1
#SBATCH --time=00:45:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
# AGA allocates whole four-GPU nodes; this instrument intentionally uses GPU 0 only.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-one-step-parity-v1/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-one-step-parity-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly EXPECTED_TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-one-step-parity-v1
readonly EXPECTED_TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-one-step-parity-v1
readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly DATASET_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow

# The five zero-valued hashes are replaced only after the producer, reducer,
# their focused tests, and the preregistration have completed their freeze pass.
readonly EXPECTED_PRODUCER_SHA256=dfa8a533b94620242e10f5c1bc6833cc295d97ce78ce002ecd223a3ab1c040ef
readonly EXPECTED_REDUCER_SHA256=661350255778cf83d65ca68aee7582aad815aa14dff29811462e49a30b30a378
readonly EXPECTED_PREREGISTRATION_SHA256=6fa6618d52429452d2cb372925036e81541381257295fd2f6fdc46f57bcab961
readonly EXPECTED_PRODUCER_TEST_SHA256=53bdf206582c08f08dda70079fabca7794ee94908afa0bb78596621dd096fb1c
readonly EXPECTED_REDUCER_TEST_SHA256=278bae8819e0216881d940413886a9de71404e3f92254de475a617a617a08819
readonly EXPECTED_LEGACY_HELPER_SHA256=bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5
readonly EXPECTED_RUNTIME_HELPER_SHA256=dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896
readonly EXPECTED_CANONICAL_HELPER_SHA256=694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac
readonly EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256=e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4
readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256=fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e
readonly EXPECTED_EXECUTION_SOURCE_FILE_COUNT=919
readonly EXPECTED_EXECUTION_SOURCE_SIZE_BYTES=10090497
readonly EXPECTED_DATASET_MANIFEST_SHA256=51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca
readonly EXPECTED_DATASET_CONFIG_SHA256=a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3
readonly EXPECTED_RESOLVED_CONFIG_SHA256=a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1
readonly EXPECTED_MODEL_CHECKPOINT_SHA256=4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88
readonly EXPECTED_TRAINING_STATE_SHA256=3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7
readonly EXPECTED_NORMALIZATION_STATE_SHA256=31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94
readonly EXPECTED_GEOMETRY_MANIFEST_SHA256=3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0
readonly EXPECTED_TARGET_MANIFEST_SHA256=d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49
readonly EXPECTED_CHECKPOINT_EPOCH=491
readonly UNIT_COUNT=2

TASK_DIR="$EXPECTED_TASK_LOGICAL"
phase="PRELAUNCH_VALIDATION"
run_status="BLOCKED_PREFLIGHT"
categorical_outcome="NONE"
completed_units=0
heartbeat_pid=""
payload_pid=""
PRODUCER_JSON=""
PRODUCER_NPZ=""
ADJUDICATION=""

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

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

verified_digest_or_missing() {
  local path="$1"
  if [[ -n "$path" ]] && verify_canonical_sidecar "$path"; then
    sha256sum "$path" | awk '{print $1}'
  else
    printf 'MISSING_OR_INVALID'
  fi
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

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  local producer_json_sha
  local producer_npz_sha
  local adjudication_sha
  local wrapper_sha
  trap - EXIT

  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [[ -n "$payload_pid" ]] && kill -0 "$payload_pid" 2>/dev/null; then
    kill -TERM -- "-$payload_pid" 2>/dev/null ||
      kill -TERM "$payload_pid" 2>/dev/null ||
      true
    wait "$payload_pid" 2>/dev/null || true
  fi

  producer_json_sha="$(verified_digest_or_missing "$PRODUCER_JSON")"
  producer_npz_sha="$(verified_digest_or_missing "$PRODUCER_NPZ")"
  adjudication_sha="$(verified_digest_or_missing "$ADJUDICATION")"
  wrapper_sha="$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  if [[ -z "$wrapper_sha" ]]; then
    wrapper_sha="UNAVAILABLE"
  fi

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
            'rc=%s\ncompleted_units=%s/%s\nphase=%s\nrun_status=%s\ncategorical_outcome=%s\nwrapper_sha256=%s\nproducer_json_sha256=%s\nproducer_npz_sha256=%s\nadjudication_sha256=%s\n' \
            "$exit_code" \
            "$completed_units" \
            "$UNIT_COUNT" \
            "$phase" \
            "$run_status" \
            "$categorical_outcome" \
            "$wrapper_sha" \
            "$producer_json_sha" \
            "$producer_npz_sha" \
            "$adjudication_sha" > "$status_path"
        ) || echo "ERROR: could not publish status marker: $status_path" >&2
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed_units}/${UNIT_COUNT} rc=${exit_code}"
  echo "ONE_STEP_PARITY_CATEGORICAL_OUTCOME=${categorical_outcome}"
  echo "EXIT_CODE=${exit_code}"
  exit "$exit_code"
}
trap finalize EXIT

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail "this wrapper must run inside an sbatch allocation"
fi
if [[ -z "${SLURM_SUBMIT_DIR:-}" || ! -d "$SLURM_SUBMIT_DIR" ]]; then
  fail "SLURM_SUBMIT_DIR is missing or is not a directory"
fi
if [[ "${SLURM_NNODES:-1}" != 1 ]]; then
  fail "one-step parity requires exactly one allocated node"
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
readonly PAYLOAD_LOG_DIR="$TASK_DIR/payload_logs"
readonly SBATCH_LOG_DIR="$TASK_DIR/sbatch_logs"
readonly EXECUTION_SOURCE="$TASK_DIR/execution_source"
for directory in \
  "$OUTPUT_DIR" \
  "$PAYLOAD_LOG_DIR" \
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

readonly PRODUCER="$TASK_DIR/drivaerml_historical_k10000_one_step_parity.py"
readonly REDUCER="$TASK_DIR/drivaerml_historical_k10000_one_step_parity_adjudicate.py"
readonly LEGACY_HELPER="$TASK_DIR/drivaerml_historical_k10000_replay.py"
readonly RUNTIME_HELPER="$TASK_DIR/drivaerml_historical_k10000_replay_runtime.py"
readonly CANONICAL_HELPER="$TASK_DIR/drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
readonly PREREGISTRATION="$TASK_DIR/phase1_historical_k10000_one_step_parity_prereg_v1_2026-07-28.json"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
readonly GEOMETRY_MANIFEST="$TASK_DIR/drivaerml_geometry_input_manifest_36cases_v1.json"
readonly TARGET_INPUT_MANIFEST="$TASK_DIR/historical_k10000_selected_target_input_manifest_v1.json"

PRODUCER_JSON="$OUTPUT_DIR/historical_k10000_one_step_parity_producer.json"
PRODUCER_NPZ="$OUTPUT_DIR/historical_k10000_one_step_parity_producer.npz"
ADJUDICATION="$OUTPUT_DIR/historical_k10000_one_step_parity_adjudication.json"
readonly PRODUCER_LOG="$PAYLOAD_LOG_DIR/producer.log"
readonly REDUCER_LOG="$PAYLOAD_LOG_DIR/reducer.log"
readonly LOCK_PATH="$TASK_DIR/.one-step-parity-v1.lock"

expected_root_directories="$(
  printf '%s\n' artifacts execution_source payload_logs sbatch_logs | LC_ALL=C sort
)"
actual_root_directories="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
)"
if [[ "$actual_root_directories" != "$expected_root_directories" ]]; then
  fail "task-root directory inventory differs from the minimal launch package"
fi
expected_root_files="$(
  printf '%s\n' \
    drivaerml_historical_k10000_one_step_parity.py \
    drivaerml_historical_k10000_one_step_parity_adjudicate.py \
    drivaerml_historical_k10000_one_step_parity_aga.sbatch \
    drivaerml_historical_k10000_replay.py \
    drivaerml_historical_k10000_replay_runtime.py \
    drivaerml_hqc_canonical_geometry_diagnostic_v5.py \
    phase1_historical_k10000_one_step_parity_prereg_v1_2026-07-28.json \
    execution_source_manifest.sha256 \
    drivaerml_geometry_input_manifest_36cases_v1.json \
    drivaerml_geometry_input_manifest_36cases_v1.json.sha256 \
    historical_k10000_selected_target_input_manifest_v1.json \
    historical_k10000_selected_target_input_manifest_v1.json.sha256 |
    LC_ALL=C sort
)"
actual_root_files="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort
)"
if [[ "$actual_root_files" != "$expected_root_files" ]]; then
  fail "task-root file inventory differs from the minimal launch package"
fi
unexpected_root_entry="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 ! -type d ! -type f -print -quit
)"
if [[ -n "$unexpected_root_entry" ]]; then
  fail "task root contains a symlink or special entry: $unexpected_root_entry"
fi

resolved_python="$(readlink -f "$PYTHON")"
if [[ ! -x "$PYTHON" || ! -f "$resolved_python" ]]; then
  fail "shared physical venv interpreter is unavailable: $PYTHON"
fi

for expected in \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_PREREGISTRATION_SHA256" \
  "$EXPECTED_PRODUCER_TEST_SHA256" \
  "$EXPECTED_REDUCER_TEST_SHA256" \
  "$EXPECTED_LEGACY_HELPER_SHA256" \
  "$EXPECTED_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_CANONICAL_HELPER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_DATASET_MANIFEST_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  "$EXPECTED_TARGET_MANIFEST_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_REDUCER_SHA256" "$REDUCER"
check_sha "$EXPECTED_PREREGISTRATION_SHA256" "$PREREGISTRATION"
check_sha "$EXPECTED_LEGACY_HELPER_SHA256" "$LEGACY_HELPER"
check_sha "$EXPECTED_RUNTIME_HELPER_SHA256" "$RUNTIME_HELPER"
check_sha "$EXPECTED_CANONICAL_HELPER_SHA256" "$CANONICAL_HELPER"
check_sha \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXECUTION_SOURCE_MANIFEST"
check_sha "$EXPECTED_DATASET_MANIFEST_SHA256" "$DATASET_MANIFEST"
check_sha "$EXPECTED_DATASET_CONFIG_SHA256" "$DATASET_CONFIG"
check_sha "$EXPECTED_RESOLVED_CONFIG_SHA256" "$RESOLVED_CONFIG"
check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256" "$MODEL_CHECKPOINT"
check_sha "$EXPECTED_TRAINING_STATE_SHA256" "$TRAINING_STATE"
check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256" "$NORMALIZATION_STATE"
check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256" "$GEOMETRY_MANIFEST"
check_sha "$EXPECTED_TARGET_MANIFEST_SHA256" "$TARGET_INPUT_MANIFEST"

for path in "$GEOMETRY_MANIFEST" "$TARGET_INPUT_MANIFEST"; do
  if ! verify_canonical_sidecar "$path"; then
    fail "frozen input sidecar is missing, stale, or noncanonical: $path"
  fi
done

source_symlink="$(find "$EXECUTION_SOURCE" -type l -print -quit)"
if [[ -n "$source_symlink" ]]; then
  fail "execution source contains a symlink: $source_symlink"
fi
source_special="$(
  find "$EXECUTION_SOURCE" ! -type d ! -type f -print -quit
)"
if [[ -n "$source_special" ]]; then
  fail "execution source contains a non-regular entry: $source_special"
fi
source_cache_path="$(
  find "$EXECUTION_SOURCE" \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
    -print -quit
)"
if [[ -n "$source_cache_path" ]]; then
  fail "execution source contains an unpinned Python cache path: $source_cache_path"
fi
read -r source_file_count source_size_bytes < <(
  find "$EXECUTION_SOURCE" -type f -printf '%s\n' |
    awk '{ count += 1; bytes += $1 } END { print count + 0, bytes + 0 }'
)
if [[ "$source_file_count" != "$EXPECTED_EXECUTION_SOURCE_FILE_COUNT" ||
  "$source_size_bytes" != "$EXPECTED_EXECUTION_SOURCE_SIZE_BYTES" ]]; then
  fail "execution-source count or byte size changed"
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
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_PRODUCER_TEST_SHA256" \
  "$EXPECTED_REDUCER_TEST_SHA256" \
  "$EXPECTED_LEGACY_HELPER_SHA256" \
  "$EXPECTED_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_CANONICAL_HELPER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  "$EXPECTED_TARGET_MANIFEST_SHA256" \
  "$EXPECTED_DATASET_MANIFEST_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256" <<'PY'
import json
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


(
    path,
    producer_sha,
    adjudicator_sha,
    producer_test_sha,
    adjudicator_test_sha,
    legacy_support_sha,
    runtime_helper_sha,
    canonical_helper_sha,
    execution_source_manifest_sha,
    execution_source_tree_sha,
    geometry_manifest_sha,
    target_manifest_sha,
    dataset_manifest_sha,
    dataset_config_sha,
    resolved_config_sha,
    model_checkpoint_sha,
    training_state_sha,
    normalization_state_sha,
) = sys.argv[1:]
with open(path, "rb") as stream:
    document = json.load(
        stream,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
if (
    document.get("schema_version") != 1
    or document.get("artifact_kind")
    != "phase1_historical_k10000_one_step_parity_preregistration"
    or document.get("status") != "PREREGISTERED_PRELAUNCH"
):
    raise ValueError("one-step parity preregistration identity changed")
expected_launch_bindings = {
    "producer_sha256": producer_sha,
    "adjudicator_sha256": adjudicator_sha,
    "producer_test_sha256": producer_test_sha,
    "adjudicator_test_sha256": adjudicator_test_sha,
    "legacy_replay_support_sha256": legacy_support_sha,
    "runtime_helper_sha256": runtime_helper_sha,
    "canonical_geometry_helper_v5_sha256": canonical_helper_sha,
    "execution_source_manifest_sha256": execution_source_manifest_sha,
    "execution_source_tree_sha256": execution_source_tree_sha,
    "geometry_manifest_sha256": geometry_manifest_sha,
    "target_input_manifest_sha256": target_manifest_sha,
}
if document.get("frozen_launch_bindings") != expected_launch_bindings:
    raise ValueError("frozen one-step launch bindings changed")
hash_cycle_policy = document.get("hash_cycle_policy", "")
normalized_hash_cycle_policy = hash_cycle_policy.lower()
if (
    "wrapper binds" not in normalized_hash_cycle_policy
    or "prereg" not in normalized_hash_cycle_policy
    or "excludes wrapper sha" not in normalized_hash_cycle_policy
):
    raise ValueError("one-step hash-cycle policy changed")
panel = document.get("fixed_panel", {})
if (
    panel.get("resolution") != 10000
    or [case.get("case_id") for case in panel.get("cases", [])]
    != ["run_118", "run_271", "run_429", "run_86"]
    or panel.get("regimes") != ["fresh_seed42", "checkpoint_epoch491"]
    or panel.get("precisions", {}).get("bfloat16") != "deciding"
):
    raise ValueError("fixed one-step panel changed")
expected_provenance = {
    "execution_source_tree_sha256": execution_source_tree_sha,
    "dataset_manifest_sha256": dataset_manifest_sha,
    "historical_dataset_config_sha256": dataset_config_sha,
    "resolved_config_sha256": resolved_config_sha,
    "model_checkpoint_sha256": model_checkpoint_sha,
    "training_state_sha256": training_state_sha,
    "normalization_state_sha256": normalization_state_sha,
}
if document.get("provenance_bindings") != expected_provenance:
    raise ValueError("one-step live provenance bindings changed")
path_contract = document.get("path_and_measure_contract", {})
if (
    "CombinedOptimizer" not in path_contract.get("optimizer", "")
    or "explicitly absent in both arms"
    not in path_contract.get("source_measure_weights", "").lower()
    or "Preserved" not in path_contract.get("target_measure", "")
):
    raise ValueError("optimizer or measure contract changed")
adjudication = document.get("adjudication", {})
if (
    adjudication.get("sole_categorical_publisher")
    != "drivaerml_historical_k10000_one_step_parity_adjudicate.py"
    or adjudication.get("producer_may_publish_a_categorical_outcome") is not False
    or adjudication.get("valid_pass_and_valid_fail_exit_code") != 0
    or adjudication.get("invalid_exit_code") != 2
    or adjudication.get("incomplete_exit_code") != 3
):
    raise ValueError("one-step adjudication/exit contract changed")
launch = document.get("launch_state", {})
for key in (
    "task_staged",
    "scheduler_test_run",
    "job_submitted",
    "producer_output_exists",
    "adjudication_output_exists",
):
    if launch.get(key) is not False:
        raise ValueError(f"preregistration is no longer prelaunch for {key}")
PY

visible_gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if [[ "$visible_gpu_count" != 4 ]]; then
  fail "AGA whole-node allocation must expose four GPUs, observed $visible_gpu_count"
fi

for path in \
  "$LOCK_PATH" \
  "$PRODUCER_JSON" \
  "$PRODUCER_JSON.sha256" \
  "$PRODUCER_NPZ" \
  "$PRODUCER_NPZ.sha256" \
  "$ADJUDICATION" \
  "$ADJUDICATION.sha256" \
  "$PRODUCER_LOG" \
  "$REDUCER_LOG" \
  "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
  "$TASK_DIR/INVALID_${SLURM_JOB_ID}" \
  "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
  "$TASK_DIR/STATUS_${SLURM_JOB_ID}"; do
  check_absent "$path"
done

(
  set -o noclobber
  : > "$LOCK_PATH"
) || fail "could not create immutable task lock"
exec 9<>"$LOCK_PATH"
if ! flock -n 9; then
  fail "another one-step parity run holds the task lock"
fi

set -o noclobber
export PYTHONPATH="$TASK_DIR:$EXECUTION_SOURCE:$RECIPE/src"
phase="PRODUCER"
run_status="RUNNING"
echo "CHECKPOINT_SELECTION epoch=$EXPECTED_CHECKPOINT_EPOCH model=$(basename "$MODEL_CHECKPOINT") training_state=$(basename "$TRAINING_STATE")"
echo "COMPILE_ENABLED=false TORCHDYNAMO_DISABLE=1"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SOURCE_TREE_SHA256=$actual_source_tree"

(
  while true; do
    observed=0
    if verify_canonical_sidecar "$PRODUCER_JSON" &&
      verify_canonical_sidecar "$PRODUCER_NPZ"; then
      observed=1
    fi
    if verify_canonical_sidecar "$ADJUDICATION"; then
      observed=2
    fi
    echo "GPU_HEARTBEAT $(date -Is) completed_units=${observed}/${UNIT_COUNT}"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits ||
      echo "ERROR: nvidia-smi heartbeat failed" >&2
    sleep 120
  done
) &
heartbeat_pid=$!

# BEGIN ONE_STEP_PRODUCER
set +e
(
  cd "$DATASET_PHYSICAL"
  exec env -i \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHDYNAMO_DISABLE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    setsid timeout --signal=TERM --kill-after=2m 32m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --dataset-root "$DATASET_PHYSICAL" \
    --dataset-config "$DATASET_CONFIG" \
    --resolved-config "$RESOLVED_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --geometry-manifest "$GEOMETRY_MANIFEST" \
    --target-input-manifest "$TARGET_INPUT_MANIFEST" \
    --output-json "$PRODUCER_JSON" \
    --output-npz "$PRODUCER_NPZ"
) > "$PRODUCER_LOG" 2>&1 &
payload_pid=$!
wait "$payload_pid"
producer_rc=$?
payload_pid=""
set -e
# END ONE_STEP_PRODUCER

if [[ "$producer_rc" -ne 0 ]]; then
  phase="BLOCKED_PRODUCER_EXECUTION"
  run_status="BLOCKED"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "PRODUCER_EXECUTION_FAILED rc=$producer_rc"
  tail -n 100 "$PRODUCER_LOG" >&2 || true
  exit 20
fi
if ! verify_canonical_sidecar "$PRODUCER_JSON" ||
  ! verify_canonical_sidecar "$PRODUCER_NPZ"; then
  phase="BLOCKED_PRODUCER_ARTIFACT"
  run_status="BLOCKED"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "PRODUCER_ARTIFACT_MISSING_INVALID_OR_NONCANONICAL"
  exit 21
fi
completed_units=1

# BEGIN ONE_STEP_REDUCER
phase="REDUCTION"
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
  --producer-json "$PRODUCER_JSON" \
  --producer-npz "$PRODUCER_NPZ" \
  --output-json "$ADJUDICATION" > "$REDUCER_LOG" 2>&1
reducer_rc=$?
set -e
# END ONE_STEP_REDUCER

case "$reducer_rc" in
  0 | 2 | 3) ;;
  *)
    phase="BLOCKED_REDUCER_EXECUTION"
    run_status="BLOCKED"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "REDUCER_EXECUTION_FAILED rc=$reducer_rc"
    tail -n 100 "$REDUCER_LOG" >&2 || true
    exit 22
    ;;
esac
if ! verify_canonical_sidecar "$ADJUDICATION"; then
  phase="BLOCKED_REDUCER_ARTIFACT"
  run_status="BLOCKED"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_ARTIFACT_MISSING_INVALID_OR_NONCANONICAL"
  exit 23
fi
completed_units=2

set +e
decision="$(
  "$PYTHON" - "$ADJUDICATION" "$reducer_rc" <<'PY'
import json
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


path, reducer_rc_text = sys.argv[1:]
with open(path, "rb") as stream:
    document = json.load(
        stream,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
if (
    document.get("schema_version") != 1
    or document.get("artifact_kind")
    != "drivaerml_historical_k10000_one_step_parity_adjudication"
):
    raise ValueError("adjudication identity changed")
status = document.get("status")
outcome = document.get("decision_outcome")
expected = {
    "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS": (
        "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        0,
    ),
    "MATERIAL_PARITY_DIFFERENCE_FAIL": (
        "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        0,
    ),
    "INVALID_ONE_STEP_PARITY_COMPARISON": (
        "INVALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        2,
    ),
    "INCOMPLETE_ONE_STEP_PARITY_COMPARISON": (
        "INCOMPLETE_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        3,
    ),
}
if outcome not in expected:
    raise ValueError(f"unknown categorical outcome: {outcome!r}")
expected_status, expected_rc = expected[outcome]
if status != expected_status or int(reducer_rc_text) != expected_rc:
    raise ValueError(
        f"status/outcome/exit mismatch: {status!r}/{outcome!r}/{reducer_rc_text}"
    )
print(f"{status}\t{outcome}")
PY
)"
decision_rc=$?
set -e
if [[ "$decision_rc" -ne 0 ]]; then
  phase="BLOCKED_DECISION_PARSE"
  run_status="BLOCKED"
  write_exclusive_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "ADJUDICATION_STATUS_OUTCOME_OR_EXIT_MISMATCH"
  exit 24
fi
IFS=$'\t' read -r adjudication_status categorical_outcome <<< "$decision"
adjudication_sha="$(sha256sum "$ADJUDICATION" | awk '{print $1}')"

case "$categorical_outcome" in
  NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS | MATERIAL_PARITY_DIFFERENCE_FAIL)
    phase="COMPLETED_VALID_SCIENTIFIC_OUTCOME"
    run_status="$adjudication_status"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "outcome=$categorical_outcome adjudication_sha256=$adjudication_sha"
    ;;
  INVALID_ONE_STEP_PARITY_COMPARISON)
    phase="INVALID_COMPARISON"
    run_status="$adjudication_status"
    write_exclusive_marker \
      "$TASK_DIR/INVALID_${SLURM_JOB_ID}" \
      "outcome=$categorical_outcome adjudication_sha256=$adjudication_sha"
    exit 2
    ;;
  INCOMPLETE_ONE_STEP_PARITY_COMPARISON)
    phase="INCOMPLETE_COMPARISON"
    run_status="$adjudication_status"
    write_exclusive_marker \
      "$TASK_DIR/INVALID_${SLURM_JOB_ID}" \
      "outcome=$categorical_outcome adjudication_sha256=$adjudication_sha"
    exit 3
    ;;
  *)
    phase="BLOCKED_UNKNOWN_OUTCOME"
    run_status="BLOCKED"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "UNKNOWN_CATEGORICAL_OUTCOME"
    exit 25
    ;;
esac

echo "ONE_STEP_PARITY_CATEGORICAL_OUTCOME=$categorical_outcome"
echo "ADJUDICATION_SHA256=$adjudication_sha"
echo "END $(date -Is)"
