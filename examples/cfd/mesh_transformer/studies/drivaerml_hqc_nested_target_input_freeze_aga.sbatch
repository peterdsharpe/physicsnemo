#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-hqc-target-freeze-v1
#SBATCH --time=00:15:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-hqc-nested-target-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-hqc-nested-target-freeze-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-hqc-nested-target-freeze-v1
readonly TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-hqc-nested-target-freeze-v1
readonly SHARED_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer
readonly DATASET=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/mt_datasets/drivaerml_ood_shadow
readonly PYTHON="$SHARED_ROOT/.venv-recipe/bin/python"

readonly EXPECTED_PRODUCER_SHA256=8cbd315e0e91b4a08adf695eb7342d34030a1135f86982554bc3a09e7bb63440
readonly EXPECTED_PREREG_SHA256=df739d827d9b52427ff70caba44a4a7ef0db2bffc84c243cbc3b24d1f5931376
readonly EXPECTED_GEOMETRY_SHA256=3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0
readonly EXPECTED_HISTORICAL_TARGET_SHA256=d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49
readonly EXPECTED_DATASET_MANIFEST_SHA256=51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca
readonly EXPECTED_ONE_STEP_PRODUCER_SHA256=f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500
readonly EXPECTED_COHORT_SHA256=ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d
readonly EXPECTED_WRAPPER_TEST_SHA256=13c71e3a774a60b4e669a19f5b0fde8c6e98307ba99492cde3a920072f935d98
# Mechanically replace this only after the frozen v3 adjudication is sealed.
# The all-zero sentinel makes this conditional template impossible to launch.
readonly EXPECTED_ACTIVATION_SHA256=0000000000000000000000000000000000000000000000000000000000000000
readonly UNIT_COUNT=1

TASK_DIR="$TASK_LOGICAL"
phase="PRELAUNCH_VALIDATION"
run_status="BLOCKED_PREFLIGHT"
completed_units=0
OUTPUT_JSON=""
OUTPUT_NPZ=""
DONE_MARKER=""

fail() {
  phase="BLOCKED_${phase}"
  run_status="BLOCKED"
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

check_canonical_sidecar() {
  local path="$1"
  local digest
  digest="$(sha256sum "$path" | awk '{print $1}')"
  [[ -f "$path.sha256" && ! -L "$path.sha256" ]] ||
    fail "canonical sidecar is missing: $path.sha256"
  cmp -s \
    "$path.sha256" \
    <(printf '%s  %s\n' "$digest" "$(basename "$path")") ||
    fail "canonical sidecar differs: $path.sha256"
}

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
finalize() {
  local exit_code=$?
  trap - EXIT
  if [[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]]; then
    local observed_task
    observed_task="$(cd -P "$TASK_DIR" 2>/dev/null && pwd)" || true
    if [[ "$observed_task" == "$TASK_PHYSICAL" ]]; then
      local status_path="$TASK_DIR/STATUS_${SLURM_JOB_ID:-no_job}"
      if [[ -e "$status_path" || -L "$status_path" ]]; then
        echo "ERROR: refusing to overwrite status marker: $status_path" >&2
      else
        (
          set -o noclobber
          printf \
            'rc=%s\ncompleted_units=%s/%s\nphase=%s\nrun_status=%s\nscientific_outcome=NONE\n' \
            "$exit_code" \
            "$completed_units" \
            "$UNIT_COUNT" \
            "$phase" \
            "$run_status" > "$status_path"
        ) || echo "ERROR: could not publish status marker: $status_path" >&2
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed_units}/${UNIT_COUNT} rc=${exit_code}"
  echo "TARGET_FREEZE_SCIENTIFIC_OUTCOME=NONE"
  echo "EXIT_CODE=${exit_code}"
  exit "$exit_code"
}
trap finalize EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] ||
  fail "this wrapper must run inside an sbatch allocation"
[[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR" ]] ||
  fail "SLURM_SUBMIT_DIR is missing or invalid"
[[ "${SLURM_NNODES:-1}" == 1 ]] ||
  fail "target freeze requires exactly one allocated node"
submit_physical="$(cd -P "$SLURM_SUBMIT_DIR" && pwd)"
[[ "$submit_physical" == "$TASK_PHYSICAL" ]] ||
  fail "submit from $TASK_PHYSICAL, observed $submit_physical"
[[ -d "$TASK_DIR" && ! -L "$TASK_DIR" ]] ||
  fail "logical task directory is missing or a symlink"
task_physical="$(cd -P "$TASK_DIR" && pwd)"
[[ "$task_physical" == "$TASK_PHYSICAL" ]] ||
  fail "logical task resolves to $task_physical"
TASK_DIR="$TASK_PHYSICAL"

readonly PRODUCER="$TASK_DIR/drivaerml_hqc_nested_target_input_freeze.py"
readonly WRAPPER="$TASK_DIR/drivaerml_hqc_nested_target_input_freeze_aga.sbatch"
readonly PREREG="$TASK_DIR/phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json"
readonly LAUNCH_MANIFEST="$TASK_DIR/hqc_nested_target_freeze_launch_manifest_v1.json"
readonly GEOMETRY="$TASK_DIR/drivaerml_geometry_input_manifest_36cases_v1.json"
readonly HISTORICAL_TARGET="$TASK_DIR/historical_k10000_selected_target_input_manifest_v1.json"
readonly ACTIVATION="$TASK_DIR/one_step_parity_adjudication.json"
readonly ARTIFACT_DIR="$TASK_DIR/artifacts"
readonly LOG_DIR="$TASK_DIR/sbatch_logs"
OUTPUT_JSON="$ARTIFACT_DIR/hqc_nested_raw_target_bundle_v1.json"
OUTPUT_NPZ="$ARTIFACT_DIR/hqc_nested_raw_target_bundle_v1.npz"
readonly REDUCER_OUTPUT="$ARTIFACT_DIR/hqc_canonical_fixed_prefix_adjudication_v1.json"
DONE_MARKER="$TASK_DIR/DONE_${SLURM_JOB_ID}.json"
readonly STATUS_MARKER="$TASK_DIR/STATUS_${SLURM_JOB_ID}"

for directory in "$ARTIFACT_DIR" "$LOG_DIR"; do
  [[ -d "$directory" && ! -L "$directory" ]] ||
    fail "required staged directory is invalid: $directory"
done
unexpected_artifact="$(
  find "$ARTIFACT_DIR" -mindepth 1 -maxdepth 1 -print -quit
)"
[[ -z "$unexpected_artifact" ]] ||
  fail "fresh attempt artifacts directory is not empty: $unexpected_artifact"
expected_log="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
actual_logs="$(
  find "$LOG_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' |
    LC_ALL=C sort
)"
[[ "$actual_logs" == "$expected_log" ]] ||
  fail "attempt log inventory differs from the current Slurm job"
[[ -f "$LOG_DIR/$expected_log" && ! -L "$LOG_DIR/$expected_log" ]] ||
  fail "current Slurm log is missing, non-regular, or a symlink"
unexpected_log_entry="$(
  find "$LOG_DIR" -mindepth 1 -maxdepth 1 ! -type f -print -quit
)"
[[ -z "$unexpected_log_entry" ]] ||
  fail "attempt log directory contains a symlink or special entry"
expected_root_directories="$(
  printf '%s\n' artifacts sbatch_logs | LC_ALL=C sort
)"
actual_root_directories="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
)"
[[ "$actual_root_directories" == "$expected_root_directories" ]] ||
  fail "staged root directory inventory differs"
expected_root_files="$(
  printf '%s\n' \
    drivaerml_geometry_input_manifest_36cases_v1.json \
    drivaerml_geometry_input_manifest_36cases_v1.json.sha256 \
    drivaerml_hqc_nested_target_input_freeze.py \
    drivaerml_hqc_nested_target_input_freeze_aga.sbatch \
    historical_k10000_selected_target_input_manifest_v1.json \
    historical_k10000_selected_target_input_manifest_v1.json.sha256 \
    hqc_nested_target_freeze_launch_manifest_v1.json \
    hqc_nested_target_freeze_launch_manifest_v1.json.sha256 \
    one_step_parity_adjudication.json \
    one_step_parity_adjudication.json.sha256 \
    phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json \
    phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json.sha256 |
    LC_ALL=C sort
)"
actual_root_files="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort
)"
[[ "$actual_root_files" == "$expected_root_files" ]] ||
  fail "staged root file inventory differs"
unexpected_root_entry="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 ! -type d ! -type f -print -quit
)"
[[ -z "$unexpected_root_entry" ]] ||
  fail "staged root contains a symlink or special entry: $unexpected_root_entry"
[[ -x "$PYTHON" && -f "$(readlink -f "$PYTHON")" ]] ||
  fail "shared Python interpreter is unavailable"

check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_PREREG_SHA256" "$PREREG"
check_sha "$EXPECTED_GEOMETRY_SHA256" "$GEOMETRY"
check_sha "$EXPECTED_HISTORICAL_TARGET_SHA256" "$HISTORICAL_TARGET"
for path in "$PREREG" "$GEOMETRY" "$HISTORICAL_TARGET"; do
  check_canonical_sidecar "$path"
done

dataset_manifest="$DATASET/manifest.json"
check_sha "$EXPECTED_DATASET_MANIFEST_SHA256" "$dataset_manifest"
for path in \
  "$ACTIVATION" \
  "$ACTIVATION.sha256" \
  "$OUTPUT_JSON" \
  "$OUTPUT_JSON.sha256" \
  "$OUTPUT_NPZ" \
  "$OUTPUT_NPZ.sha256" \
  "$REDUCER_OUTPUT" \
  "$REDUCER_OUTPUT.sha256" \
  "$DONE_MARKER" \
  "$STATUS_MARKER"; do
  if [[ "$path" == "$ACTIVATION" || "$path" == "$ACTIVATION.sha256" ]]; then
    [[ -f "$path" && ! -L "$path" ]] ||
      fail "activation evidence is missing or invalid: $path"
  else
    check_absent "$path"
  fi
done

phase="ACTIVATION_VALIDATION"
VALIDATION_OUTPUT="$(
  "$PYTHON" - \
    "$ACTIVATION" \
    "$EXPECTED_ONE_STEP_PRODUCER_SHA256" \
    "$EXPECTED_ACTIVATION_SHA256" \
    "$LAUNCH_MANIFEST" \
    "$WRAPPER" \
    "$PREREG" \
    "$EXPECTED_PREREG_SHA256" \
    "$EXPECTED_PRODUCER_SHA256" \
    "$EXPECTED_WRAPPER_TEST_SHA256" \
    "$EXPECTED_GEOMETRY_SHA256" \
    "$EXPECTED_HISTORICAL_TARGET_SHA256" \
    "$EXPECTED_DATASET_MANIFEST_SHA256" \
    "$EXPECTED_COHORT_SHA256" \
    "$TASK_LOGICAL" \
    "$TASK_PHYSICAL" \
    "$OUTPUT_JSON" \
    "$OUTPUT_NPZ" \
    "$REDUCER_OUTPUT" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys


def fail(message):
    raise SystemExit(message)


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate activation JSON key: {key}")
        result[key] = value
    return result


def json_exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            json_exact(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            json_exact(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def stable_read(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"path is not a regular file: {path}")
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
        fail(f"path changed while being read: {path}")
    return b"".join(chunks)


path = pathlib.Path(sys.argv[1])
expected_producer = sys.argv[2]
expected_activation = sys.argv[3]
if (
    re.fullmatch(r"[0-9a-f]{64}", expected_activation) is None
    or expected_activation == "0" * 64
):
    fail("activation SHA-256 has not been frozen")
payload = stable_read(path)
digest = hashlib.sha256(payload).hexdigest()
if digest != expected_activation:
    fail("activation SHA-256 differs")
expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
if stable_read(path.with_name(path.name + ".sha256")) != expected_sidecar:
    fail("activation sidecar is not canonical")
try:
    document = json.loads(
        payload,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=lambda value: fail(f"non-finite activation token: {value}"),
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    fail(f"activation JSON is invalid: {error}")
required = {
    "schema_version": 1,
    "artifact_kind": "drivaerml_historical_k10000_one_step_parity_adjudication",
    "status": "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
    "decision_outcome": "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS",
}
if not isinstance(document, dict):
    fail("activation JSON is not an object")
if set(document) != {
    "schema_version",
    "artifact_kind",
    "status",
    "decision_outcome",
    "created_at_utc",
    "validity",
    "decision_contract",
    "results",
    "limited_claim",
    "next_step",
}:
    fail("activation top-level keys differ")
for key, value in required.items():
    if not json_exact(document.get(key), value):
        fail(f"activation {key} differs")
validity = document.get("validity")
if not isinstance(validity, dict):
    fail("activation validity is absent")
if validity.get("producer_source_sha256") != expected_producer:
    fail("activation producer identity differs")
for key in (
    "raw_array_manifest_verified",
    "shared_controls_verified",
    "parameter_partition_verified",
    "all_values_finite",
):
    if validity.get(key) is not True:
        fail(f"activation validity flag differs: {key}")
for key in ("producer_json_sha256", "producer_npz_sha256"):
    value = validity.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        fail(f"activation {key} is malformed")
expected_contract = {
    "deciding_precision": "bfloat16",
    "all_case_regime_gates_required": True,
    "gradient_cosine_inclusive_minimum": 0.999,
    "update_cosine_inclusive_minimum": 0.9999,
    "update_symmetric_relative_l2_inclusive_maximum": 0.01,
    "gradient_path_fraction_of_between_case_median_inclusive_maximum": 0.1,
    "active_module_energy_fraction_inclusive_minimum": 0.01,
    "active_module_cosine_inclusive_minimum": 0.99,
    "fp32_role": "diagnostic_only",
}
if not json_exact(document.get("decision_contract"), expected_contract):
    fail("activation decision contract differs")
results = document.get("results")
if not isinstance(results, dict) or set(results) != {"bfloat16", "float32"}:
    fail("activation precision results differ")
expected_cases = {"run_118", "run_271", "run_429", "run_86"}
for precision, deciding in (("bfloat16", True), ("float32", False)):
    precision_result = results[precision]
    if (
        not isinstance(precision_result, dict)
        or precision_result.get("deciding") is not deciding
    ):
        fail(f"activation {precision} role differs")
    regimes = precision_result.get("regimes")
    if not isinstance(regimes, dict) or set(regimes) != {
        "fresh_seed42",
        "checkpoint_epoch491",
    }:
        fail(f"activation {precision} regimes differ")
    for regime in regimes.values():
        cases = regime.get("cases") if isinstance(regime, dict) else None
        if not isinstance(cases, dict) or set(cases) != expected_cases:
            fail(f"activation {precision} case coverage differs")
        if precision == "bfloat16" and not all(
            isinstance(case, dict)
            and case.get("deciding") is True
            and case.get("passed") is True
            for case in cases.values()
        ):
            fail("activation BF16 result contains a non-passing case")

manifest_path = pathlib.Path(sys.argv[4])
wrapper_path = pathlib.Path(sys.argv[5])
prereg_path = pathlib.Path(sys.argv[6])
expected_prereg = sys.argv[7]
expected_target_producer = sys.argv[8]
expected_wrapper_test = sys.argv[9]
expected_geometry = sys.argv[10]
expected_historical = sys.argv[11]
expected_dataset = sys.argv[12]
expected_cohort = sys.argv[13]
task_logical = pathlib.Path(sys.argv[14])
task_physical = pathlib.Path(sys.argv[15])
target_json = pathlib.Path(sys.argv[16])
target_npz = pathlib.Path(sys.argv[17])
reducer_output = pathlib.Path(sys.argv[18])


def authenticated_json(json_path, label):
    json_payload = stable_read(json_path)
    json_digest = hashlib.sha256(json_payload).hexdigest()
    sidecar = json_path.with_name(json_path.name + ".sha256")
    expected = f"{json_digest}  {json_path.name}\n".encode("ascii")
    if stable_read(sidecar) != expected:
        fail(f"{label} sidecar is not canonical")
    try:
        value = json.loads(
            json_payload,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda token: fail(f"non-finite {label} token: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} JSON is invalid: {error}")
    if not isinstance(value, dict):
        fail(f"{label} JSON is not an object")
    return value, json_digest


prereg, prereg_digest = authenticated_json(prereg_path, "preregistration")
if prereg_digest != expected_prereg:
    fail("preregistration SHA-256 differs")
implementation = prereg.get("implementation_freeze")
if not isinstance(implementation, dict):
    fail("preregistration implementation freeze is absent")
manifest, manifest_digest = authenticated_json(manifest_path, "launch manifest")
attempt_id = task_physical.name
if set(manifest) != {
    "schema_version",
    "artifact_kind",
    "status",
    "attempt_id",
    "task_logical",
    "task_physical",
    "artifacts",
    "bindings",
}:
    fail("launch manifest top-level keys differ")
required_manifest = {
    "schema_version": 1,
    "artifact_kind": "drivaerml_hqc_nested_target_freeze_launch_manifest",
    "status": "FROZEN_HQC_NESTED_TARGET_FREEZE_LAUNCH",
    "attempt_id": attempt_id,
    "task_logical": str(task_logical),
    "task_physical": str(task_physical),
}
for key, value in required_manifest.items():
    if not json_exact(manifest.get(key), value):
        fail(f"launch manifest {key} differs")
if task_logical.resolve(strict=True) != task_physical.resolve(strict=True):
    fail("launch manifest logical and physical roots do not resolve together")
expected_artifacts = {
    "target_json_relative_path": str(target_json.relative_to(task_physical)),
    "target_npz_relative_path": str(target_npz.relative_to(task_physical)),
    "target_done_pattern": "DONE_<slurm_job_id>.json",
    "reducer_output_relative_path": str(reducer_output.relative_to(task_physical)),
}
if not json_exact(manifest.get("artifacts"), expected_artifacts):
    fail("launch manifest artifact namespace differs")
wrapper_digest = hashlib.sha256(stable_read(wrapper_path)).hexdigest()
sealed_predictions = prereg.get("sealed_target_free_predictions", {})
anchor = prereg.get("k10000_anchor", {})
blind_target = prereg.get("blind_target_bundle", {})
expected_bindings = {
    "preregistration_sha256": prereg_digest,
    "activation_adjudication_sha256": digest,
    "wrapper_sha256": wrapper_digest,
    "target_producer_sha256": expected_target_producer,
    "target_producer_test_sha256": implementation.get(
        "target_producer_test_sha256"
    ),
    "target_wrapper_test_sha256": expected_wrapper_test,
    "reducer_sha256": implementation.get("reducer_sha256"),
    "reducer_test_sha256": implementation.get("reducer_test_sha256"),
    "geometry_manifest_sha256": expected_geometry,
    "historical_target_manifest_sha256": expected_historical,
    "dataset_manifest_sha256": expected_dataset,
    "cohort_sha256": expected_cohort,
    "one_step_producer_sha256": expected_producer,
    "prediction_lane_json_sha256": sealed_predictions.get("lane_json_sha256"),
    "prediction_lane_npz_sha256": sealed_predictions.get("lane_npz_sha256"),
    "canonical_k10000_json_sha256": anchor.get("canonical_lane_a_json", {}).get(
        "sha256"
    ),
    "canonical_k10000_npz_sha256": anchor.get("canonical_lane_a_npz", {}).get(
        "sha256"
    ),
    "stage_b_k10000_json_sha256": anchor.get("stage_b_lane_a_json", {}).get(
        "sha256"
    ),
    "stage_b_k10000_npz_sha256": anchor.get("stage_b_lane_a_npz", {}).get(
        "sha256"
    ),
    "accepted_adjudication_sha256": anchor.get(
        "accepted_adjudication_json", {}
    ).get("sha256"),
}
if manifest.get("bindings") != expected_bindings:
    fail("launch manifest bindings differ")
if blind_target.get("producer", {}).get("sha256") != expected_target_producer:
    fail("preregistration target producer binding differs")
if blind_target.get("producer_test", {}).get("sha256") != implementation.get(
    "target_producer_test_sha256"
):
    fail("preregistration target producer test binding differs")
if (
    re.fullmatch(r"[0-9a-f]{64}", expected_wrapper_test) is None
    or expected_wrapper_test == "0" * 64
):
    fail("wrapper-test SHA-256 has not been frozen")
print(digest, manifest_digest, wrapper_digest, attempt_id)
PY
)"
read -r ACTIVATION_SHA256 LAUNCH_MANIFEST_SHA256 WRAPPER_SHA256 ATTEMPT_ID \
  <<< "$VALIDATION_OUTPUT"
readonly ACTIVATION_SHA256 LAUNCH_MANIFEST_SHA256 WRAPPER_SHA256 ATTEMPT_ID
for digest in \
  "$ACTIVATION_SHA256" \
  "$LAUNCH_MANIFEST_SHA256" \
  "$WRAPPER_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
    fail "activation/manifest verifier did not return three SHA-256 digests"
done
[[ "$ATTEMPT_ID" == "$(basename "$TASK_PHYSICAL")" ]] ||
  fail "activation/manifest verifier returned the wrong attempt ID"

exec 9<"$WRAPPER"
flock -n 9 || fail "another target freeze holds the immutable wrapper lock"

phase="TARGET_FREEZE"
run_status="RUNNING"
echo "TARGET_FREEZE_BLIND=true scientific_outcome=NONE"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=30s 12m \
  env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  "$PYTHON" "$PRODUCER" \
  --dataset-root "$DATASET" \
  --geometry-manifest "$GEOMETRY" \
  --historical-k10000-target-manifest "$HISTORICAL_TARGET" \
  --output-json "$OUTPUT_JSON" \
  --output-npz "$OUTPUT_NPZ"

# Inputs are immutable by contract. Recheck them after the potentially long
# target read so the DONE marker never commits a mixed-time snapshot.
check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_PREREG_SHA256" "$PREREG"
check_sha "$EXPECTED_GEOMETRY_SHA256" "$GEOMETRY"
check_sha "$EXPECTED_HISTORICAL_TARGET_SHA256" "$HISTORICAL_TARGET"
check_sha "$EXPECTED_DATASET_MANIFEST_SHA256" "$dataset_manifest"
check_sha "$ACTIVATION_SHA256" "$ACTIVATION"
check_sha "$LAUNCH_MANIFEST_SHA256" "$LAUNCH_MANIFEST"
check_sha "$WRAPPER_SHA256" "$WRAPPER"
for path in \
  "$PREREG" \
  "$GEOMETRY" \
  "$HISTORICAL_TARGET" \
  "$ACTIVATION" \
  "$LAUNCH_MANIFEST"; do
  check_canonical_sidecar "$path"
done

phase="BUNDLE_COMMIT"
"$PYTHON" - \
  "$OUTPUT_JSON" \
  "$OUTPUT_NPZ" \
  "$DONE_MARKER" \
  "$SLURM_JOB_ID" \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_PREREG_SHA256" \
  "$EXPECTED_COHORT_SHA256" \
  "$EXPECTED_DATASET_MANIFEST_SHA256" \
  "$EXPECTED_GEOMETRY_SHA256" \
  "$EXPECTED_HISTORICAL_TARGET_SHA256" \
  "$ACTIVATION" \
  "$ACTIVATION_SHA256" \
  "$PREREG" \
  "$LAUNCH_MANIFEST" \
  "$LAUNCH_MANIFEST_SHA256" \
  "$WRAPPER" \
  "$WRAPPER_SHA256" \
  "$ATTEMPT_ID" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import tempfile


def fail(message):
    raise SystemExit(message)


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate bundle JSON key: {key}")
        result[key] = value
    return result


def json_exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            json_exact(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            json_exact(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def stable_read(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"path is not a regular file: {path}")
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
        fail(f"path changed while being read: {path}")
    return b"".join(chunks)


def canonical_sidecar(path):
    payload = stable_read(path)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if stable_read(sidecar) != expected:
        fail(f"bundle sidecar is not canonical: {sidecar}")
    return payload, digest


json_path = pathlib.Path(sys.argv[1])
npz_path = pathlib.Path(sys.argv[2])
done_path = pathlib.Path(sys.argv[3])
job_id = sys.argv[4]
producer_sha256 = sys.argv[5]
prereg_sha256 = sys.argv[6]
cohort_sha256 = sys.argv[7]
dataset_sha256 = sys.argv[8]
geometry_sha256 = sys.argv[9]
historical_target_sha256 = sys.argv[10]
activation_path = pathlib.Path(sys.argv[11])
expected_activation_sha256 = sys.argv[12]
prereg_path = pathlib.Path(sys.argv[13])
launch_manifest_path = pathlib.Path(sys.argv[14])
expected_launch_manifest_sha256 = sys.argv[15]
wrapper_path = pathlib.Path(sys.argv[16])
expected_wrapper_sha256 = sys.argv[17]
attempt_id = sys.argv[18]

json_payload, json_sha256 = canonical_sidecar(json_path)
_, npz_sha256 = canonical_sidecar(npz_path)
activation_payload, activation_sha256 = canonical_sidecar(activation_path)
if activation_sha256 != expected_activation_sha256:
    fail("activation evidence changed after preflight")
prereg_payload, observed_prereg_sha256 = canonical_sidecar(prereg_path)
if observed_prereg_sha256 != prereg_sha256:
    fail("preregistration changed after preflight")
launch_manifest_payload, launch_manifest_sha256 = canonical_sidecar(
    launch_manifest_path
)
if launch_manifest_sha256 != expected_launch_manifest_sha256:
    fail("launch manifest changed after preflight")
wrapper_payload = stable_read(wrapper_path)
wrapper_sha256 = hashlib.sha256(wrapper_payload).hexdigest()
if wrapper_sha256 != expected_wrapper_sha256:
    fail("wrapper changed after preflight")
if (
    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id) is None
    or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None
    or done_path.name != f"DONE_{job_id}.json"
):
    fail("attempt ID, job ID, or DONE filename is malformed")
try:
    document = json.loads(
        json_payload,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=lambda value: fail(f"non-finite bundle token: {value}"),
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    fail(f"bundle JSON is invalid: {error}")
required = {
    "schema_version": 1,
    "artifact_kind": "drivaerml_hqc_nested_raw_target_bundle",
    "status": "PASSED_HQC_NESTED_RAW_TARGET_FREEZE",
    "case_count": 36,
    "resolutions": [2500, 5000, 10000, 20000, 40000],
    "max_resolution": 40000,
    "fixed_query_resolution": 2500,
    "cohort_sha256": cohort_sha256,
}
if not isinstance(document, dict):
    fail("bundle JSON is not an object")
for key, value in required.items():
    if not json_exact(document.get(key), value):
        fail(f"bundle {key} differs")
if document.get("dataset_manifest_sha256") != dataset_sha256:
    fail("bundle dataset identity differs")
if document.get("geometry_manifest", {}).get("sha256") != geometry_sha256:
    fail("bundle geometry-manifest identity differs")
historical = document.get("historical_k10000_target_manifest", {})
if historical.get("sha256") != historical_target_sha256:
    fail("bundle historical-target identity differs")
if historical.get("prefix_hashes_authenticated") != 72:
    fail("bundle does not authenticate all 72 historical prefixes")
npz = document.get("npz")
if not isinstance(npz, dict):
    fail("bundle NPZ record is absent")
if pathlib.Path(npz.get("path", "")).resolve() != npz_path.resolve():
    fail("bundle internal NPZ path differs")
if npz.get("sha256") != npz_sha256 or not json_exact(
    npz.get("array_count"), 144
):
    fail("bundle internal NPZ identity differs")
provenance = document.get("provenance")
if not isinstance(provenance, dict) or provenance.get(
    "script_sha256"
) != producer_sha256:
    fail("bundle producer identity differs")
publication = document.get("publication_contract")
expected_publication = {
    "json_manifest_linked_last": True,
    "producer_outputs_are_not_a_commit_marker": True,
    "valid_only_after_external_sidecar_checks_and_done_marker": True,
    "interrupted_partial_bundle_must_not_be_overwritten": True,
}
if not json_exact(publication, expected_publication):
    fail("bundle publication contract differs")
exclusions = document.get("read_exclusions")
expected_exclusions = {
    "model_opened": False,
    "prediction_opened": False,
    "metric_opened": False,
    "decision_threshold_opened": False,
    "other_cell_data_opened": False,
    "point_data_opened": False,
    "interior_opened": False,
}
if not json_exact(exclusions, expected_exclusions):
    fail("bundle blind-read exclusions differ")
cases = document.get("cases")
manifest = document.get("array_manifest")
if not isinstance(cases, list) or len(cases) != 36:
    fail("bundle case coverage differs")
if not isinstance(manifest, dict) or len(manifest) != 144:
    fail("bundle array manifest coverage differs")
expected_case_ids = (
    "run_118", "run_129", "run_145", "run_149", "run_17", "run_171",
    "run_18", "run_183", "run_197", "run_202", "run_225", "run_270",
    "run_271", "run_298", "run_305", "run_320", "run_367", "run_380",
    "run_382", "run_399", "run_4", "run_409", "run_419", "run_424",
    "run_429", "run_431", "run_439", "run_465", "run_468", "run_469",
    "run_478", "run_489", "run_490", "run_495", "run_71", "run_86",
)
observed_ordinals = []
expected_manifest = {}
for case in cases:
    if not isinstance(case, dict):
        fail("bundle case is not an object")
    ordinal = case.get("cohort_ordinal")
    observed_ordinals.append(ordinal)
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        fail("bundle cohort ordinal is invalid")
    if ordinal < 0 or ordinal >= len(expected_case_ids):
        fail("bundle cohort ordinal is out of range")
    case_id = expected_case_ids[ordinal]
    if case.get("case_id") != case_id:
        fail(f"bundle case {ordinal} identity differs")
    selection = case.get("selection")
    targets = case.get("targets")
    if not isinstance(selection, dict) or len(
        selection.get("selected_cell_ids_sha256_by_resolution", {})
    ) != 5:
        fail(f"bundle case {ordinal} selection coverage differs")
    if not isinstance(targets, dict) or set(targets) != {"pressure", "wss"}:
        fail(f"bundle case {ordinal} target coverage differs")
    for record in targets.values():
        if record.get("historical_k10000_prefix_authenticated") is not True:
            fail(f"bundle case {ordinal} historical prefix is unauthenticated")
        hashes = record.get("prefix_sha256_by_resolution")
        if not isinstance(hashes, dict) or set(hashes) != {
            "2500",
            "5000",
            "10000",
            "20000",
            "40000",
        }:
            fail(f"bundle case {ordinal} prefix coverage differs")
    prefix = f"case_{ordinal:02d}_{case_id}__"
    expected_manifest.update(
        {
            f"{prefix}selected_cell_ids_int64": ([40000], "<i8", 320000),
            f"{prefix}physical_globals_float32": ([7], "<f4", 28),
            f"{prefix}raw_target_pressure_float32": ([40000], "<f4", 160000),
            f"{prefix}raw_target_wss_float32": ([40000, 3], "<f4", 480000),
        }
    )
if observed_ordinals != list(range(36)):
    fail("bundle cohort ordinal order differs")
if set(manifest) != set(expected_manifest):
    fail("bundle array manifest names differ")
for key, (shape, dtype, nbytes) in expected_manifest.items():
    record = manifest[key]
    if not isinstance(record, dict) or set(record) != {
        "shape",
        "dtype",
        "nbytes",
        "sha256",
    }:
        fail("bundle array manifest entry is malformed")
    if not json_exact(record["shape"], shape) or record["dtype"] != dtype:
        fail(f"bundle array shape or dtype differs: {key}")
    if not json_exact(record["nbytes"], nbytes):
        fail(f"bundle array byte count differs: {key}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        fail(f"bundle array digest is malformed: {key}")

if done_path.exists() or done_path.is_symlink():
    fail(f"refusing to overwrite DONE marker: {done_path}")
done_document = {
    "artifact_kind": "drivaerml_hqc_nested_target_bundle_commit",
    "activation_adjudication_sha256": activation_sha256,
    "attempt_id": attempt_id,
    "job_id": job_id,
    "json_sha256": json_sha256,
    "launch_manifest_sha256": launch_manifest_sha256,
    "npz_sha256": npz_sha256,
    "preregistration_sha256": prereg_sha256,
    "producer_sha256": producer_sha256,
    "schema_version": 1,
    "reducer_schema_validation_performed": False,
    "status": "CONTENT_COMMITTED_UNVALIDATED_HQC_NESTED_TARGET_BUNDLE",
    "wrapper_sha256": wrapper_sha256,
}
done_payload = (
    json.dumps(
        done_document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    + b"\n"
)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{done_path.name}.",
    suffix=".tmp",
    dir=done_path.parent,
)
temporary = pathlib.Path(temporary_name)
linked = False
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(done_payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o644)
    _, final_json_sha256 = canonical_sidecar(json_path)
    _, final_npz_sha256 = canonical_sidecar(npz_path)
    final_activation_payload, final_activation_sha256 = canonical_sidecar(
        activation_path
    )
    final_prereg_payload, final_prereg_sha256 = canonical_sidecar(prereg_path)
    final_launch_manifest_payload, final_launch_manifest_sha256 = (
        canonical_sidecar(launch_manifest_path)
    )
    final_wrapper_payload = stable_read(wrapper_path)
    if (
        final_json_sha256 != json_sha256
        or final_npz_sha256 != npz_sha256
        or final_activation_sha256 != activation_sha256
        or final_activation_payload != activation_payload
        or final_prereg_sha256 != prereg_sha256
        or final_prereg_payload != prereg_payload
        or final_launch_manifest_sha256 != launch_manifest_sha256
        or final_launch_manifest_payload != launch_manifest_payload
        or final_wrapper_payload != wrapper_payload
    ):
        fail("a committed input changed before DONE publication")
    os.link(temporary, done_path, follow_symlinks=False)
    linked = True
    directory_descriptor = os.open(done_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    if stable_read(done_path) != done_payload:
        fail("published DONE marker verification failed")
except BaseException:
    if linked:
        try:
            destination_stat = done_path.stat(follow_symlinks=False)
            temporary_stat = temporary.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if (destination_stat.st_dev, destination_stat.st_ino) == (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            ):
                done_path.unlink()
                directory_descriptor = os.open(
                    done_path.parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
    raise
finally:
    temporary.unlink(missing_ok=True)
PY

completed_units=1
phase="COMPLETE"
run_status="COMPLETED"
echo "TARGET_BUNDLE_COMMITTED $(date -Is) marker=$(basename "$DONE_MARKER")"
