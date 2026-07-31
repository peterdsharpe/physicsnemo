#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-canon-validity
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH -o /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1/sbatch_logs/%x_%j.log
#SBATCH -e /home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="${HOME}/.local/bin:/cm/local/apps/slurm/current/bin:/usr/bin:/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

readonly EXPECTED_TASK_DIR_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1
readonly EXPECTED_TASK_DIR_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-28-mt-canonical-validity-36x5-v1
readonly EXPECTED_LANE_PRODUCER_SHA256=f96e272dd5d34d0a7e7d708b312f5b2856218715f22e5a489a6acbf5192baab5
readonly EXPECTED_REDUCER_SHA256=ea5769b927f5a3ada3030ca8849706f391b4cb0aa0c61384098e698a1538d9c9
readonly EXPECTED_PREREGISTRATION_SHA256=99bc8521c98170b1569d3471739dda13fbf23677d2c55248ddedb3969ca79501
readonly EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256=e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4
readonly EXPECTED_SOURCE_TREE_SHA256=fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e
readonly EXPECTED_GEOMETRY_MANIFEST_SHA256=3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0
readonly LANE_COUNT=4
readonly CASE_COUNT=36

TASK_DIR="$EXPECTED_TASK_DIR_LOGICAL"
completed_units=0
sampler_pid=""
declare -a lane_pids=()

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  local pid
  trap - EXIT
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
  for pid in "${lane_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  if [[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]]; then
    local final_task_physical
    final_task_physical="$(cd -P "$TASK_DIR" 2>/dev/null && pwd)" || true
    if [[ "$final_task_physical" == "$EXPECTED_TASK_DIR_PHYSICAL" ]]; then
      local status_path="$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
      if [[ -e "$status_path" || -L "$status_path" ]]; then
        echo "ERROR: refusing to overwrite status marker $status_path" >&2
      else
        printf 'rc=%s\ncompleted_units=%s/%s\n' \
          "$exit_code" "$completed_units" "$LANE_COUNT" > "$status_path"
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed_units}/${LANE_COUNT} rc=${exit_code}"
  echo "EXIT_CODE=${exit_code}"
  exit "$exit_code"
}
trap finalize EXIT

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail "this wrapper must run inside an sbatch allocation"
fi
if [[ -z "${SLURM_SUBMIT_DIR:-}" || ! -d "$SLURM_SUBMIT_DIR" ]]; then
  fail "SLURM_SUBMIT_DIR is missing or is not a directory"
fi
submit_dir_physical="$(cd -P "$SLURM_SUBMIT_DIR" && pwd)"
if [[ "$submit_dir_physical" != "$EXPECTED_TASK_DIR_PHYSICAL" ]]; then
  fail "submit from $EXPECTED_TASK_DIR_PHYSICAL, observed $submit_dir_physical"
fi
if [[ ! -d "$TASK_DIR" || -L "$TASK_DIR" ]]; then
  fail "logical task directory is missing or is a symlink: $TASK_DIR"
fi
task_dir_physical="$(cd -P "$TASK_DIR" && pwd)"
if [[ "$task_dir_physical" != "$EXPECTED_TASK_DIR_PHYSICAL" ]]; then
  fail "logical task directory resolves to $task_dir_physical"
fi
TASK_DIR="$EXPECTED_TASK_DIR_PHYSICAL"

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
execution_source_physical="$(cd -P "$EXECUTION_SOURCE" && pwd)"
if [[ "$execution_source_physical" != "$EXPECTED_TASK_DIR_PHYSICAL/execution_source" ]]; then
  fail "execution source is not the task-local physical snapshot"
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

readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly VENV="$SHARED_ROOT_PHYSICAL/.venv-recipe"
readonly PYTHON="$VENV/bin/python"
readonly RECIPE="$EXECUTION_SOURCE/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly SHARED_RECIPE="$SHARED_ROOT_PHYSICAL/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
readonly DATASET=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
readonly RUN_DIR="$SHARED_RECIPE/runs/t2_mesh_transformer_surface_flagship_seed42"
readonly CHECKPOINT_DIR="$RUN_DIR/checkpoints"

readonly LANE_PRODUCER="$TASK_DIR/drivaerml_hqc_canonical_geometry_validity.py"
readonly REDUCER="$TASK_DIR/drivaerml_hqc_canonical_geometry_validity_adjudicate.py"
readonly CANONICAL_HELPER="$TASK_DIR/drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
readonly FROZEN_HQC_PRODUCER="$TASK_DIR/drivaerml_trace_fixed_query_audit.py"
readonly PREREGISTRATION="$TASK_DIR/phase1_hqc_canonical_geometry_validity_preregistration_v1_2026-07-28.json"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
readonly GEOMETRY_MANIFEST="$TASK_DIR/drivaerml_geometry_input_manifest_36cases_v1.json"
readonly ANCHOR_JSON="$TASK_DIR/hqc_neutral_canonical_geometry_four_canaries_k2500_v5.json"
readonly ANCHOR_NPZ="$TASK_DIR/hqc_neutral_canonical_geometry_four_canaries_k2500_v5.npz"
readonly ANCHOR_NPZ_SIDECAR="$ANCHOR_NPZ.sha256"
readonly ADJUDICATION="$OUTPUT_DIR/phase1_hqc_canonical_geometry_validity_adjudication_v1.json"

resolved_python="$(readlink -f "$PYTHON")"
if [[ ! -x "$PYTHON" || ! -f "$resolved_python" ]]; then
  fail "shared physical venv interpreter is unavailable: $PYTHON"
fi

for expected in \
  "$EXPECTED_LANE_PRODUCER_SHA256" \
  "$EXPECTED_REDUCER_SHA256" \
  "$EXPECTED_PREREGISTRATION_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_SOURCE_TREE_SHA256" \
  "$EXPECTED_GEOMETRY_MANIFEST_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

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

validate_preregistration() {
  "$PYTHON" - \
    "$PREREGISTRATION" \
    "$EXPECTED_PREREGISTRATION_SHA256" \
    "$EXPECTED_LANE_PRODUCER_SHA256" \
    "$EXPECTED_REDUCER_SHA256" \
    "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
    "$EXPECTED_SOURCE_TREE_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate preregistration key: {key}")
        result[key] = value
    return result


(
    path,
    expected_digest,
    expected_producer,
    expected_reducer,
    expected_manifest,
    expected_source_tree,
) = sys.argv[1:]
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("preregistration is not a regular file")
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
    raise ValueError("preregistration changed while being read")
payload = b"".join(chunks)
observed_digest = hashlib.sha256(payload).hexdigest()
if observed_digest != expected_digest:
    raise ValueError(
        "preregistration SHA mismatch: "
        f"expected={expected_digest} observed={observed_digest}"
    )
if b"__" + b"FILL_" in payload:
    raise ValueError("preregistration contains an unresolved placeholder")
document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
if document.get("status") != "PREREGISTERED_PRELAUNCH":
    raise ValueError("preregistration status differs")
provenance = document.get("frozen_provenance")
if not isinstance(provenance, dict):
    raise ValueError("preregistration frozen_provenance is missing")
expected = {
    "lane_producer_sha256": expected_producer,
    "reducer_sha256": expected_reducer,
    "execution_source_manifest_sha256": expected_manifest,
    "execution_source_tree_sha256": expected_source_tree,
}
for key, value in expected.items():
    if provenance.get(key) != value:
        raise ValueError(
            f"preregistration provenance differs for {key}: "
            f"expected={value} observed={provenance.get(key)!r}"
        )
PY
}

check_sha "$EXPECTED_LANE_PRODUCER_SHA256" "$LANE_PRODUCER"
check_sha "$EXPECTED_REDUCER_SHA256" "$REDUCER"
validate_preregistration
check_sha \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXECUTION_SOURCE_MANIFEST"
check_sha 694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac \
  "$CANONICAL_HELPER"
check_sha 8b6e8055e3563e4eec6a4ff311f567dda68f9b743092aad5805c7457ffde611f \
  "$FROZEN_HQC_PRODUCER"
check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256" "$GEOMETRY_MANIFEST"
check_sha 09e336442881f0641c14c91c17dd80ac440d474f996925cf79f2174bc4cacd88 \
  "$ANCHOR_JSON"
check_sha edee836e0cc5c66690276e6787496cbd6b81fb08decb7d54ecf1f36b333ddc9f \
  "$ANCHOR_NPZ"
check_sha ecb00237458f08432976329c2ac28582a3719be49943f1d59ced30ec1d7a7b73 \
  "$ANCHOR_NPZ_SIDECAR"
if ! cmp -s \
  "$ANCHOR_NPZ_SIDECAR" \
  <(
    printf '%s  %s\n' \
      edee836e0cc5c66690276e6787496cbd6b81fb08decb7d54ecf1f36b333ddc9f \
      hqc_neutral_canonical_geometry_four_canaries_k2500_v5.npz
  ); then
  fail "anchor NPZ sidecar is noncanonical or mismatched"
fi

check_sha 51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca \
  "$DATASET/manifest.json"
check_sha a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3 \
  "$RECIPE/datasets/drivaer_ml_surface.yaml"
check_sha a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1 \
  "$RUN_DIR/resolved_config.yaml"
check_sha 4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88 \
  "$CHECKPOINT_DIR/MeshTransformer.0.491.mdlus"
check_sha 31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94 \
  "$CHECKPOINT_DIR/norm_stats.pt"
check_sha 3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7 \
  "$CHECKPOINT_DIR/checkpoint.0.491.pt"

unexpected_recipe_file="$(
  find "$RECIPE" \
    -type d -name __pycache__ -prune -o \
    -type f ! -name '*.pyc' \
    ! -path "$RECIPE/src/*" \
    ! -path "$RECIPE/datasets/drivaer_ml_surface.yaml" \
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
if [[ "$actual_source_tree" != "$EXPECTED_SOURCE_TREE_SHA256" ]]; then
  fail "execution source tree changed: expected=$EXPECTED_SOURCE_TREE_SHA256 actual=$actual_source_tree"
fi

cd "$TASK_DIR"
readonly LOCK_PATH="$TASK_DIR/.canonical-geometry-validity.lock"
if [[ -L "$LOCK_PATH" ]]; then
  fail "task lock path is a symlink"
fi
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  fail "another canonical-geometry validity job holds the task lock"
fi

declare -a lane_jsons=()
declare -a lane_npzs=()
declare -a lane_logs=()
for lane in 0 1 2 3; do
  lane_jsons[lane]="$OUTPUT_DIR/canonical_geometry_validity_lane${lane}.json"
  lane_npzs[lane]="$OUTPUT_DIR/canonical_geometry_validity_lane${lane}.npz"
  lane_logs[lane]="$LANE_LOG_DIR/lane${lane}.log"
done

check_absent() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    fail "immutable output target already exists: $path"
  fi
}

for lane in 0 1 2 3; do
  check_absent "${lane_jsons[$lane]}"
  check_absent "${lane_jsons[$lane]}.sha256"
  check_absent "${lane_npzs[$lane]}"
  check_absent "${lane_npzs[$lane]}.sha256"
  check_absent "${lane_logs[$lane]}"
done
check_absent "$ADJUDICATION"
check_absent "$ADJUDICATION.sha256"
check_absent "$TASK_DIR/DONE_${SLURM_JOB_ID}"
check_absent "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}"

export PYTHONPATH="$EXECUTION_SOURCE:$RECIPE/src"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SOURCE_TREE_SHA256=$actual_source_tree"
"$PYTHON" -c \
  'import numpy, physicsnemo, torch; print("VERSIONS", numpy.__version__, torch.__version__, physicsnemo.__file__); print("CUDA", torch.cuda.is_available(), torch.cuda.device_count())'

completed_case_count() {
  local total=0
  local lane
  local observed
  for lane in 0 1 2 3; do
    observed="$(
      sed -n 's/.*COMPLETED_UNITS=\([0-9][0-9]*\)\/9.*/\1/p' \
        "${lane_logs[$lane]}" 2>/dev/null |
        tail -n 1
    )"
    total=$((total + ${observed:-0}))
  done
  printf '%s' "$total"
}

completed_lane_count() {
  local total=0
  local lane
  for lane in 0 1 2 3; do
    if [[
      -f "${lane_jsons[$lane]}" &&
      -f "${lane_jsons[$lane]}.sha256" &&
      -f "${lane_npzs[$lane]}" &&
      -f "${lane_npzs[$lane]}.sha256"
    ]]; then
      total=$((total + 1))
    fi
  done
  printf '%s' "$total"
}

(
  while true; do
    echo \
      "GPU_HEARTBEAT $(date -Is) completed_cases=$(completed_case_count)/${CASE_COUNT} completed_lanes=$(completed_lane_count)/${LANE_COUNT}"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits ||
      echo "ERROR: nvidia-smi heartbeat failed" >&2
    sleep 300
  done
) &
sampler_pid=$!

for lane in 0 1 2 3; do
  echo "LAUNCH lane=$lane gpu=$lane threads=8 $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$lane"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export NUMEXPR_NUM_THREADS=8
    echo "LANE_START lane=$lane gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
    exec setsid timeout --signal=TERM --kill-after=2m 95m \
      "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
      "$LANE_PRODUCER" \
      --producer "$FROZEN_HQC_PRODUCER" \
      --canonical-helper "$CANONICAL_HELPER" \
      --repo-root "$EXECUTION_SOURCE" \
      --dataset-root "$DATASET" \
      --dataset-config "$RECIPE/datasets/drivaer_ml_surface.yaml" \
      --resolved-config "$RUN_DIR/resolved_config.yaml" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --geometry-input-manifest "$GEOMETRY_MANIFEST" \
      --anchor-json "$ANCHOR_JSON" \
      --anchor-npz "$ANCHOR_NPZ" \
      --lane-ordinal "$lane" \
      --lane-count "$LANE_COUNT" \
      --output-json "${lane_jsons[$lane]}" \
      --output-npz "${lane_npzs[$lane]}"
  ) > "${lane_logs[$lane]}" 2>&1 &
  lane_pids[lane]=$!
done

declare -a lane_rcs=()
set +e
for lane in 0 1 2 3; do
  wait "${lane_pids[$lane]}"
  lane_rcs[lane]=$?
  lane_pids[lane]=""
  echo "LANE_EXIT lane=$lane rc=${lane_rcs[$lane]} $(date -Is)"
  if [[ "${lane_rcs[$lane]}" -ne 0 ]]; then
    tail -n 80 "${lane_logs[$lane]}" >&2 || true
  fi
  if (
    cd "$OUTPUT_DIR" &&
      sha256sum --check "$(basename "${lane_jsons[$lane]}").sha256" &&
      sha256sum --check "$(basename "${lane_npzs[$lane]}").sha256"
  ); then
    completed_units=$((completed_units + 1))
  else
    echo "LANE_ARTIFACT_UNAVAILABLE_OR_INVALID lane=$lane" >&2
  fi
  echo "COMPLETED_UNITS=${completed_units}/${LANE_COUNT}"
done
set -e

declare -a reducer_lane_args=()
for lane in 0 1 2 3; do
  reducer_lane_args+=(
    --lane "${lane_jsons[$lane]}" "${lane_npzs[$lane]}"
  )
done

echo "REDUCER_START $(date -Is)"
set +e
timeout --signal=TERM --kill-after=30s 12m \
  "$PYTHON" "$REDUCER" \
  "${reducer_lane_args[@]}" \
  --expected-lane-producer-sha256 "$EXPECTED_LANE_PRODUCER_SHA256" \
  --expected-source-tree-sha256 "$EXPECTED_SOURCE_TREE_SHA256" \
  --expected-geometry-input-manifest-sha256 \
    "$EXPECTED_GEOMETRY_MANIFEST_SHA256" \
  --anchor-npz "$ANCHOR_NPZ" \
  --output-json "$ADJUDICATION"
reducer_rc=$?
set -e

write_marker() {
  local path="$1"
  local payload="$2"
  if [[ -e "$path" || -L "$path" ]]; then
    fail "refusing to overwrite marker $path"
  fi
  printf '%s\n' "$payload" > "$path"
}

if [[ "$reducer_rc" -ne 0 ]]; then
  write_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_EXECUTION_FAILED rc=$reducer_rc"
  exit 20
fi
decision="$(
  "$PYTHON" - "$ADJUDICATION" <<'PY'
import hashlib
import json
import os
import stat
import sys


def read_regular_nofollow(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


path = os.path.abspath(sys.argv[1])
sidecar_path = f"{path}.sha256"
payload = read_regular_nofollow(path)
sidecar = read_regular_nofollow(sidecar_path)
digest = hashlib.sha256(payload).hexdigest()
expected_sidecar = f"{digest}  {os.path.basename(path)}\n".encode("ascii")
if sidecar != expected_sidecar:
    raise ValueError("adjudication sidecar is noncanonical or mismatched")
result = json.loads(payload)
print(f"{result['status']}\t{result['decision']['outcome']}")
PY
)" || {
  write_marker \
    "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
    "REDUCER_ARTIFACT_MISSING_INVALID_OR_UNREADABLE"
  exit 21
}
IFS=$'\t' read -r adjudication_status adjudication_outcome <<< "$decision"
echo \
  "REDUCER_DONE status=$adjudication_status outcome=$adjudication_outcome $(date -Is)"

case "$adjudication_status:$adjudication_outcome" in
  VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT:CANONICAL_FULL_VALIDITY_PASS | \
    VALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT:CANONICAL_FULL_VALIDITY_REFUTED)
    completed_units="$LANE_COUNT"
    write_marker \
      "$TASK_DIR/DONE_${SLURM_JOB_ID}" \
      "$adjudication_status outcome=$adjudication_outcome"
    echo "CANONICAL_GEOMETRY_VALIDITY_DONE $(date -Is)"
    ;;
  INCOMPLETE_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT:INCOMPLETE_COHORT)
    write_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$adjudication_status outcome=$adjudication_outcome"
    exit 30
    ;;
  INVALID_TARGET_FREE_CANONICAL_GEOMETRY_VALIDITY_COHORT:INVALID_COHORT)
    write_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "$adjudication_status outcome=$adjudication_outcome"
    exit 31
    ;;
  *)
    write_marker \
      "$TASK_DIR/BLOCKED_${SLURM_JOB_ID}" \
      "UNRECOGNIZED_REDUCER_RESULT status=$adjudication_status outcome=$adjudication_outcome"
    exit 32
    ;;
esac
