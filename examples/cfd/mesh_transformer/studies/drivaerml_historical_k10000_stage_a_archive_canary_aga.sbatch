#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-stagea
#SBATCH --time=00:30:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
# AGA schedules whole four-GPU nodes; this payload deliberately uses only GPUs 0 and 1.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-a-v1/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-a-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly EXPECTED_TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-a-v1
readonly EXPECTED_TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-historical-k10k-stage-a-v1
readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly HISTORICAL_ARCHIVE_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-25-mt-coverage-sweep/artifacts/res10000/t2_mesh_transformer_surface_flagship_seed42/predictions

readonly EXPECTED_PRODUCER_SHA256=2aa6e518ddea77d54fba3ab5060458c86a2fd72f8dc37e8fd2cc963c4484c8e3
readonly EXPECTED_REDUCER_SHA256=98281da54489e7c54731d35ade345824949686f931ccbde3b77ebf46cce27b60
readonly EXPECTED_PREREGISTRATION_SHA256=8f35dd8b558c883bd75242d9b15062257f87638a871201321aceebeef578c1d3
readonly EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256=e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4
readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256=fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e
readonly EXPECTED_DATASET_CONFIG_SHA256=a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3
readonly EXPECTED_RESOLVED_CONFIG_SHA256=a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1
readonly EXPECTED_MODEL_CHECKPOINT_SHA256=4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88
readonly EXPECTED_TRAINING_STATE_SHA256=3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7
readonly EXPECTED_NORMALIZATION_STATE_SHA256=31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94
readonly EXPECTED_CURRENT_INFER_SHA256=47aec675e54d58ee4202831ae0d20039b1ff5ec40e69cc8b5087ce00bd5234ed
readonly EXPECTED_CURRENT_MODEL_SOURCE_SHA256=9096f61a5c54a6f92d14c586aaa8cf51a8bc22fc797f50bd0cbfdf86ef042892
readonly EXPECTED_HISTORICAL_MANIFEST_SHA256=545b1f6e906002231415b84277db00eec04f3666233b8da637514e9077a585eb
readonly EXPECTED_INPUT_FREEZE_SHA256=fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10
readonly EXPECTED_HISTORICAL_FILE_COUNT=1656
readonly EXPECTED_HISTORICAL_SIZE_BYTES=44790588
readonly EXPECTED_ARCHIVE_INPUT_FILE_COUNT=10
readonly EXPECTED_ARCHIVE_INPUT_SIZE_BYTES=719432
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
  echo "STAGE_A_CATEGORICAL_OUTCOME=${categorical_outcome}"
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
  fail "Stage A requires exactly one allocated node"
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
readonly ARCHIVE_INPUT_ROOT="$TASK_DIR/archive_input_root"
for directory in \
  "$OUTPUT_DIR" \
  "$LANE_LOG_DIR" \
  "$SBATCH_LOG_DIR" \
  "$EXECUTION_SOURCE" \
  "$ARCHIVE_INPUT_ROOT"; do
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    fail "required staged directory is missing or is a symlink: $directory"
  fi
done
if [[ "$(cd -P "$EXECUTION_SOURCE" && pwd)" != "$TASK_DIR/execution_source" ]]; then
  fail "execution source is not the task-local physical snapshot"
fi
if [[ "$(cd -P "$ARCHIVE_INPUT_ROOT" && pwd)" != "$TASK_DIR/archive_input_root" ]]; then
  fail "archive input root is not the task-local physical snapshot"
fi

readonly VENV="$SHARED_ROOT_PHYSICAL/.venv-recipe"
readonly PYTHON="$VENV/bin/python"
readonly RECIPE="$EXECUTION_SOURCE/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly SHARED_RECIPE="$SHARED_ROOT_PHYSICAL/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly RUN_DIR="$SHARED_RECIPE/runs/t2_mesh_transformer_surface_flagship_seed42"
readonly CHECKPOINT_DIR="$RUN_DIR/checkpoints"
readonly DATASET_CONFIG="$RECIPE/datasets/drivaer_ml_surface.yaml"
readonly RESOLVED_CONFIG="$RUN_DIR/resolved_config.yaml"
readonly MODEL_CHECKPOINT="$CHECKPOINT_DIR/MeshTransformer.0.491.mdlus"
readonly TRAINING_STATE="$CHECKPOINT_DIR/checkpoint.0.491.pt"
readonly NORMALIZATION_STATE="$CHECKPOINT_DIR/norm_stats.pt"
readonly CURRENT_INFER="$RECIPE/src/infer.py"
readonly CURRENT_MODEL_SOURCE="$EXECUTION_SOURCE/physicsnemo/experimental/nn/mesh_attention/model.py"

readonly PRODUCER="$TASK_DIR/drivaerml_historical_k10000_stage_a_archive_canary.py"
readonly REDUCER="$TASK_DIR/drivaerml_historical_k10000_stage_a_archive_canary_adjudicate.py"
readonly PREREGISTRATION="$TASK_DIR/phase1_historical_k10000_stage_a_archive_canary_prereg_v1_2026-07-28.json"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
readonly HISTORICAL_MANIFEST="$TASK_DIR/historical_predictions_tree_manifest.sha256"
readonly INPUT_FREEZE="$TASK_DIR/historical_k10000_input_freeze_v1.json"

readonly LANE_A_JSON="$OUTPUT_DIR/stage_a_archive_canary_lane_A.json"
readonly LANE_A_NPZ="$OUTPUT_DIR/stage_a_archive_canary_lane_A.npz"
readonly LANE_B_JSON="$OUTPUT_DIR/stage_a_archive_canary_lane_B.json"
readonly LANE_B_NPZ="$OUTPUT_DIR/stage_a_archive_canary_lane_B.npz"
readonly LANE_A_LOG="$LANE_LOG_DIR/lane_A.log"
readonly LANE_B_LOG="$LANE_LOG_DIR/lane_B.log"
readonly ADJUDICATION="$OUTPUT_DIR/stage_a_archive_canary_adjudication.json"

resolved_python="$(readlink -f "$PYTHON")"
if [[ ! -x "$PYTHON" || ! -f "$resolved_python" ]]; then
  fail "shared physical venv interpreter is unavailable: $PYTHON"
fi

for expected in \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_PREREGISTRATION_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256" \
  "$EXPECTED_CURRENT_INFER_SHA256" \
  "$EXPECTED_CURRENT_MODEL_SOURCE_SHA256" \
  "$EXPECTED_HISTORICAL_MANIFEST_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_REDUCER_SHA256" "$REDUCER"
check_sha "$EXPECTED_PREREGISTRATION_SHA256" "$PREREGISTRATION"
check_sha \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXECUTION_SOURCE_MANIFEST"
check_sha "$EXPECTED_HISTORICAL_MANIFEST_SHA256" "$HISTORICAL_MANIFEST"
check_sha "$EXPECTED_INPUT_FREEZE_SHA256" "$INPUT_FREEZE"
check_sha "$EXPECTED_DATASET_CONFIG_SHA256" "$DATASET_CONFIG"
check_sha "$EXPECTED_RESOLVED_CONFIG_SHA256" "$RESOLVED_CONFIG"
check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256" "$MODEL_CHECKPOINT"
check_sha "$EXPECTED_TRAINING_STATE_SHA256" "$TRAINING_STATE"
check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256" "$NORMALIZATION_STATE"
check_sha "$EXPECTED_CURRENT_INFER_SHA256" "$CURRENT_INFER"
check_sha "$EXPECTED_CURRENT_MODEL_SOURCE_SHA256" "$CURRENT_MODEL_SOURCE"

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
  "$ARCHIVE_INPUT_ROOT" \
  "$INPUT_FREEZE" \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256" \
  "$EXPECTED_CURRENT_INFER_SHA256" \
  "$EXPECTED_CURRENT_MODEL_SOURCE_SHA256" \
  "$EXPECTED_HISTORICAL_MANIFEST_SHA256" \
  "$EXPECTED_INPUT_FREEZE_SHA256" \
  "$EXPECTED_TASK_PHYSICAL" \
  "$EXPECTED_ARCHIVE_INPUT_FILE_COUNT" \
  "$EXPECTED_ARCHIVE_INPUT_SIZE_BYTES" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


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
    archive_input_root_text,
    input_freeze_path,
    producer_sha,
    reducer_sha,
    source_manifest_sha,
    source_tree_sha,
    dataset_config_sha,
    resolved_config_sha,
    model_checkpoint_sha,
    training_state_sha,
    normalization_state_sha,
    current_infer_sha,
    current_model_sha,
    historical_manifest_sha,
    input_freeze_sha,
    task_physical,
    expected_file_count_text,
    expected_size_bytes_text,
) = sys.argv[1:]

expected_inputs = [
    {
        "name": "archive_boundary_points_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/boundaries/vehicle/_tensordict/points.memmap",
        "sha256": "01f7e493a9ae6ce5bf514d12a608855a6ce13ba559fd3b1859aca3635c420a46",
        "shape": [29949, 3],
        "dtype": "float32",
    },
    {
        "name": "archive_boundary_cells_int64",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/boundaries/vehicle/_tensordict/cells.memmap",
        "sha256": "a66df151ecc02c04d412bd20622a0c32115c7c87f9c899380ffea5a0cfd08e0d",
        "shape": [10000, 3],
        "dtype": "int64",
    },
    {
        "name": "archive_query_points_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/interior/_tensordict/points.memmap",
        "sha256": "77bb36bb03e7f983d0763fa157ba2c61c234e4582475d3b220b0d44a572c082e",
        "shape": [10000, 3],
        "dtype": "float32",
    },
    {
        "name": "archive_global_L_ref_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/L_ref.memmap",
        "sha256": "fca31f1667a6aa1bba12fca4e4ea1becd503379d80da3213af07f6cc5702828d",
        "shape": [],
        "dtype": "float32",
    },
    {
        "name": "archive_global_U_inf_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/U_inf.memmap",
        "sha256": "527665492391c17b8dd4486fd555390eadf800b9f9b60b7b4cd79e9e5ea9e9f0",
        "shape": [3],
        "dtype": "float32",
    },
    {
        "name": "archive_global_U_inf_dir_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/U_inf_dir.memmap",
        "sha256": "480376c6bf738a0227f2bbf2b3506b7cde209152c0ba9a9077e5527169eb292e",
        "shape": [3],
        "dtype": "float32",
    },
    {
        "name": "archive_global_nu_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/nu.memmap",
        "sha256": "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c",
        "shape": [],
        "dtype": "float32",
    },
    {
        "name": "archive_global_p_inf_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/p_inf.memmap",
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
        "shape": [],
        "dtype": "float32",
    },
    {
        "name": "archive_global_reference_length_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/reference_length.memmap",
        "sha256": "03e3c2420f5066a5fa6e36735ed8cc4f6a251046263e1a6024f009deeee3b952",
        "shape": [],
        "dtype": "float32",
    },
    {
        "name": "archive_global_rho_inf_float32",
        "relative_path": "00021_run_118_domain_run_118.pdmsh/_tensordict/global_data/rho_inf.memmap",
        "sha256": "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c",
        "shape": [],
        "dtype": "float32",
    },
]

document = strict_json(preregistration_path)
if document.get("artifact_kind") != (
    "phase1_historical_k10000_stage_a_archive_domain_canary_preregistration"
):
    raise ValueError("preregistration artifact kind changed")
if document.get("status") != "PREREGISTERED_PRELAUNCH":
    raise ValueError("preregistration status changed")

expected_provenance = {
    "producer_sha256": producer_sha,
    "reducer_sha256": reducer_sha,
    "execution_source_manifest_sha256": source_manifest_sha,
    "execution_source_tree_sha256": source_tree_sha,
    "dataset_config_sha256": dataset_config_sha,
    "resolved_config_sha256": resolved_config_sha,
    "model_checkpoint_sha256": model_checkpoint_sha,
    "training_state_sha256": training_state_sha,
    "normalization_state_sha256": normalization_state_sha,
    "current_infer_source_sha256": current_infer_sha,
    "current_model_source_sha256": current_model_sha,
    "historical_prediction_tree_manifest_sha256": historical_manifest_sha,
    "historical_input_freeze_record_sha256": input_freeze_sha,
}
if document.get("frozen_provenance") != expected_provenance:
    raise ValueError("preregistration frozen provenance changed")

isolation = document.get("producer_input_isolation", {})
if isolation.get("payload_count") != len(expected_inputs):
    raise ValueError("preregistration input payload count changed")
if isolation.get("payloads") != expected_inputs:
    raise ValueError("preregistration exact input payload inventory changed")

lane_contract = document.get("lane_contract", {})
if lane_contract.get("cuda_visible_devices") != {"A": "0", "B": "1"}:
    raise ValueError("preregistration lane/GPU binding changed")
if lane_contract.get("process_isolation") != (
    "two concurrent, separate one-rank torchrun processes"
):
    raise ValueError("preregistration process-isolation contract changed")
if lane_contract.get("encoded_geometry_exact_definition") != (
    "A/B raw-byte equality of the actual EncodedBoundary source points, cells, "
    "centroids, areas, normals, center, and reference length captured inside "
    "each lane's single model(domain) forward; it is not a fresh CPU "
    "reconstruction and is not inferred from predictions"
):
    raise ValueError("encoded-geometry exactness definition changed")

expected_outcomes = [
    "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS",
    "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION",
    "INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT",
    "INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY",
]
truth_table = document.get("adjudication", {}).get("truth_table", [])
if [row.get("outcome") for row in truth_table] != expected_outcomes:
    raise ValueError("preregistered truth-table outcomes changed")
exit_semantics = document.get("adjudication", {}).get(
    "execution_exit_semantics",
    {},
)
if set(exit_semantics) != set(expected_outcomes):
    raise ValueError("preregistered exit-semantics inventory changed")
if "must not be converted into infrastructure failure" not in exit_semantics.get(
    "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION",
    "",
):
    raise ValueError("valid-refutation success semantics changed")
if document.get("decision_order", {}).get(
    "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS"
) != "licenses Stage B reconstructed historical replay only":
    raise ValueError("Stage-A pass license changed")

aga = document.get("aga_execution_contract", {})
expected_aga = {
    "task_physical": task_physical,
    "nodes": 1,
    "scheduler_gpu_allocation": 4,
    "payload_gpu_count": 2,
    "payload_gpu_indices": [0, 1],
    "slurm_export": "NIL",
    "wall_time": "00:30:00",
    "lane_timeout": "16m",
    "reducer_timeout": "5m",
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

freeze = strict_json(input_freeze_path)
if freeze.get("status") != "FROZEN_REFERENCE_BEFORE_EXACT_REPLAY":
    raise ValueError("input-freeze status changed")
historical_reference = freeze.get("historical_reference", {})
if (
    historical_reference.get("prediction_tree_file_count") != 1656
    or historical_reference.get("prediction_tree_size_bytes") != 44790588
    or historical_reference.get("prediction_tree_manifest_sha256")
    != historical_manifest_sha
):
    raise ValueError("input-freeze historical archive binding changed")

archive_input_root = Path(archive_input_root_text)
if archive_input_root.is_symlink() or not archive_input_root.is_dir():
    raise ValueError("archive input root is missing, non-directory, or a symlink")
expected_by_path = {row["relative_path"]: row for row in expected_inputs}
observed_paths = []
observed_size = 0
for walk_root, directory_names, file_names in os.walk(
    archive_input_root,
    topdown=True,
    followlinks=False,
):
    root = Path(walk_root)
    for name in directory_names:
        if (root / name).is_symlink():
            raise ValueError(f"archive input contains a symlink: {root / name}")
    for name in file_names:
        path = root / name
        relative = path.relative_to(archive_input_root).as_posix()
        payload = read_regular_nofollow(path)
        observed_paths.append(relative)
        observed_size += len(payload)
        expected = expected_by_path.get(relative)
        if expected is None:
            raise ValueError(f"archive input contains an extra file: {relative}")
        observed_sha = hashlib.sha256(payload).hexdigest()
        if observed_sha != expected["sha256"]:
            raise ValueError(
                f"archive input SHA changed: {relative}; observed={observed_sha}"
            )
if sorted(observed_paths) != sorted(expected_by_path):
    raise ValueError("archive input inventory is missing one or more exact payloads")
if len(observed_paths) != int(expected_file_count_text):
    raise ValueError("archive input file count changed")
if observed_size != int(expected_size_bytes_text):
    raise ValueError("archive input total byte count changed")
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

readonly LOCK_PATH="$TASK_DIR/.stage-a-archive-canary.lock"
if [[ -L "$LOCK_PATH" ]]; then
  fail "task lock path is a symlink"
fi
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  fail "another Stage-A archive canary holds the task lock"
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
  cd "$ARCHIVE_INPUT_ROOT"
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
    setsid timeout --signal=TERM --kill-after=2m 16m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --resolved-config "$RESOLVED_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --archive-input-root "$ARCHIVE_INPUT_ROOT" \
    --lane-label A \
    --output-json "$LANE_A_JSON" \
    --output-npz "$LANE_A_NPZ"
) > "$LANE_A_LOG" 2>&1 &
lane_pids[0]=$!

(
  cd "$ARCHIVE_INPUT_ROOT"
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
    setsid timeout --signal=TERM --kill-after=2m 16m \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$PRODUCER" \
    --repo-root "$EXECUTION_SOURCE" \
    --resolved-config "$RESOLVED_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --archive-input-root "$ARCHIVE_INPUT_ROOT" \
    --lane-label B \
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
  timeout --signal=TERM --kill-after=30s 5m \
  "$PYTHON" "$REDUCER" \
  --lane-a-json "$LANE_A_JSON" \
  --lane-a-npz "$LANE_A_NPZ" \
  --lane-b-json "$LANE_B_JSON" \
  --lane-b-npz "$LANE_B_NPZ" \
  --historical-predictions "$HISTORICAL_ARCHIVE_PHYSICAL" \
  --historical-predictions-manifest "$HISTORICAL_MANIFEST" \
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
    "phase1_historical_k10000_stage_a_archive_domain_adjudication"
):
    raise ValueError("adjudication artifact kind changed")
if document.get("status") != "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_ADJUDICATION":
    raise ValueError("adjudication status changed")
outcome = document.get("outcome")
allowed = {
    "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS",
    "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION",
    "INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT",
    "INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY",
}
if outcome not in allowed:
    raise ValueError(f"unrecognized adjudication outcome: {outcome!r}")
print(f"{document['status']}\t{outcome}")
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
echo "STAGE_A_ADJUDICATION_STATUS=$adjudication_status"

case "$categorical_outcome" in
  EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS)
    phase="COMPLETED_EXACT_PASS"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; licenses Stage B reconstructed historical replay only"
    ;;
  VALID_STAGE_A_CURRENT_SOURCE_REFUTATION)
    phase="COMPLETED_VALID_REFUTATION"
    write_exclusive_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$categorical_outcome; successful scientific refutation; Stage B remains blocked"
    ;;
  INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT)
    phase="BLOCKED_INVALID_INSTRUMENT"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; repair and rerun Stage A"
    exit 30
    ;;
  INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY)
    phase="BLOCKED_INCOMPLETE_CANARY"
    write_exclusive_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$categorical_outcome; complete or rerun Stage A"
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

echo "STAGE_A_ARCHIVE_CANARY_DONE $(date -Is)"
