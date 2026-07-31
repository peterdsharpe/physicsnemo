#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J mt-k10k-step-smoke-v1
#SBATCH --time=00:10:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
# AGA allocates whole four-GPU nodes; this validity smoke uses GPU 0 only.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=NIL
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-historical-k10k-one-step-checkpoint-smoke-v1/sbatch_logs/%x_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-historical-k10k-one-step-checkpoint-smoke-v1/sbatch_logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail
ulimit -c 0

export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

readonly EXPECTED_TASK_LOGICAL=/home/psharpe/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-historical-k10k-one-step-checkpoint-smoke-v1
readonly EXPECTED_TASK_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-07-29-mt-historical-k10k-one-step-checkpoint-smoke-v1
readonly SHARED_ROOT_PHYSICAL=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer

readonly EXPECTED_PRODUCER_SHA256=f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500
readonly EXPECTED_LEGACY_HELPER_SHA256=bce26e1e55d9231843c2255ed7e57fe20166e6fd6098b77d9a63944e8b1dd7a5
readonly EXPECTED_RUNTIME_HELPER_SHA256=dc4d2a71a0c9c72ff62166801433b21ae6f9b672801dfe5388c7975e887f4896
readonly EXPECTED_CANONICAL_HELPER_SHA256=694c45556acd3d002fcd34ffaac2872761ce61eaa60f842c8040f71edd1af7ac
readonly EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256=e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4
readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256=fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e
readonly EXPECTED_EXECUTION_SOURCE_FILE_COUNT=919
readonly EXPECTED_EXECUTION_SOURCE_SIZE_BYTES=10090497
readonly EXPECTED_DATASET_CONFIG_SHA256=a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3
readonly EXPECTED_RESOLVED_CONFIG_SHA256=a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1
readonly EXPECTED_MODEL_CHECKPOINT_SHA256=4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88
readonly EXPECTED_TRAINING_STATE_SHA256=3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7
readonly EXPECTED_NORMALIZATION_STATE_SHA256=31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94
readonly EXPECTED_CHECKPOINT_EPOCH=491
readonly EXPECTED_PARAMETER_COUNT=1278268
readonly UNIT_COUNT=1

TASK_DIR="$EXPECTED_TASK_LOGICAL"
phase="PRELAUNCH_VALIDATION"
run_status="BLOCKED_PREFLIGHT"
categorical_scientific_outcome="NONE"
completed_units=0
PAYLOAD_LOG=""
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
  local log_sha="MISSING"
  local wrapper_sha="UNAVAILABLE"
  trap - EXIT

  if [[ -n "$PAYLOAD_LOG" && -f "$PAYLOAD_LOG" && ! -L "$PAYLOAD_LOG" ]]; then
    log_sha="$(sha256sum "$PAYLOAD_LOG" | awk '{print $1}')"
  fi
  if [[ -f "$0" && ! -L "$0" ]]; then
    wrapper_sha="$(sha256sum "$0" | awk '{print $1}')"
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
            'rc=%s\ncompleted_units=%s/%s\nphase=%s\nrun_status=%s\ncategorical_scientific_outcome=%s\nwrapper_sha256=%s\npayload_log_sha256=%s\n' \
            "$exit_code" \
            "$completed_units" \
            "$UNIT_COUNT" \
            "$phase" \
            "$run_status" \
            "$categorical_scientific_outcome" \
            "$wrapper_sha" \
            "$log_sha" > "$status_path"
        ) || echo "ERROR: could not publish status marker: $status_path" >&2
      fi
    fi
  fi
  echo "COMPLETED_UNITS=${completed_units}/${UNIT_COUNT} rc=${exit_code}"
  echo "CHECKPOINT_SMOKE_CATEGORICAL_SCIENTIFIC_OUTCOME=NONE"
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
  fail "checkpoint smoke requires exactly one allocated node"
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

readonly PAYLOAD_LOG_DIR="$TASK_DIR/payload_logs"
readonly SBATCH_LOG_DIR="$TASK_DIR/sbatch_logs"
readonly EXECUTION_SOURCE="$TASK_DIR/execution_source"
for directory in "$PAYLOAD_LOG_DIR" "$SBATCH_LOG_DIR" "$EXECUTION_SOURCE"; do
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    fail "required staged directory is missing or is a symlink: $directory"
  fi
done
if [[ "$(cd -P "$EXECUTION_SOURCE" && pwd)" != "$TASK_DIR/execution_source" ]]; then
  fail "execution source is not the task-local physical snapshot"
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

readonly PRODUCER="$TASK_DIR/drivaerml_historical_k10000_one_step_parity.py"
readonly LEGACY_HELPER="$TASK_DIR/drivaerml_historical_k10000_replay.py"
readonly RUNTIME_HELPER="$TASK_DIR/drivaerml_historical_k10000_replay_runtime.py"
readonly CANONICAL_HELPER="$TASK_DIR/drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
readonly WRAPPER="$TASK_DIR/drivaerml_historical_k10000_one_step_checkpoint_smoke_aga.sbatch"
readonly EXECUTION_SOURCE_MANIFEST="$TASK_DIR/execution_source_manifest.sha256"
PAYLOAD_LOG="$PAYLOAD_LOG_DIR/checkpoint_smoke.log"
DONE_MARKER="$TASK_DIR/DONE_${SLURM_JOB_ID}"
readonly STATUS_MARKER="$TASK_DIR/STATUS_${SLURM_JOB_ID}"

expected_root_directories="$(
  printf '%s\n' execution_source payload_logs sbatch_logs | LC_ALL=C sort
)"
actual_root_directories="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
)"
if [[ "$actual_root_directories" != "$expected_root_directories" ]]; then
  fail "task-root directory inventory differs from the minimal smoke package"
fi
expected_root_files="$(
  printf '%s\n' \
    drivaerml_historical_k10000_one_step_checkpoint_smoke_aga.sbatch \
    drivaerml_historical_k10000_one_step_parity.py \
    drivaerml_historical_k10000_replay.py \
    drivaerml_historical_k10000_replay_runtime.py \
    drivaerml_hqc_canonical_geometry_diagnostic_v5.py \
    execution_source_manifest.sha256 |
    LC_ALL=C sort
)"
actual_root_files="$(
  find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort
)"
if [[ "$actual_root_files" != "$expected_root_files" ]]; then
  fail "task-root file inventory differs from the minimal smoke package"
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
  "$EXPECTED_LEGACY_HELPER_SHA256" \
  "$EXPECTED_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_CANONICAL_HELPER_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256" \
  "$EXPECTED_DATASET_CONFIG_SHA256" \
  "$EXPECTED_RESOLVED_CONFIG_SHA256" \
  "$EXPECTED_MODEL_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_STATE_SHA256" \
  "$EXPECTED_NORMALIZATION_STATE_SHA256"; do
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    fail "an expected SHA-256 is unresolved or malformed: $expected"
  fi
done

check_sha "$EXPECTED_PRODUCER_SHA256" "$PRODUCER"
check_sha "$EXPECTED_LEGACY_HELPER_SHA256" "$LEGACY_HELPER"
check_sha "$EXPECTED_RUNTIME_HELPER_SHA256" "$RUNTIME_HELPER"
check_sha "$EXPECTED_CANONICAL_HELPER_SHA256" "$CANONICAL_HELPER"
check_sha \
  "$EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256" \
  "$EXECUTION_SOURCE_MANIFEST"
check_sha "$EXPECTED_DATASET_CONFIG_SHA256" "$DATASET_CONFIG"
check_sha "$EXPECTED_RESOLVED_CONFIG_SHA256" "$RESOLVED_CONFIG"
check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256" "$MODEL_CHECKPOINT"
check_sha "$EXPECTED_TRAINING_STATE_SHA256" "$TRAINING_STATE"
check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256" "$NORMALIZATION_STATE"

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
  fail "execution-source manifest bytes differ from the staged inventory"
fi
(
  cd "$EXECUTION_SOURCE"
  sha256sum --quiet --strict --check "$EXECUTION_SOURCE_MANIFEST"
) || fail "execution-source manifest content verification failed"

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

visible_gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if [[ "$visible_gpu_count" != 4 ]]; then
  fail "AGA whole-node allocation must expose four GPUs, observed $visible_gpu_count"
fi
for path in "$PAYLOAD_LOG" "$DONE_MARKER" "$STATUS_MARKER"; do
  check_absent "$path"
done
exec 9<"$WRAPPER"
if ! flock -n 9; then
  fail "another checkpoint smoke holds the immutable wrapper lock"
fi

set -o noclobber
export PYTHONPATH="$TASK_DIR:$EXECUTION_SOURCE:$RECIPE/src"
phase="CHECKPOINT_VALIDITY_SMOKE"
run_status="RUNNING"
echo "CHECKPOINT_SMOKE_VALIDITY_ONLY=true categorical_scientific_outcome=NONE"
echo "CHECKPOINT_SELECTION epoch=$EXPECTED_CHECKPOINT_EPOCH model=$(basename "$MODEL_CHECKPOINT") training_state=$(basename "$TRAINING_STATE")"
echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SOURCE_TREE_SHA256=$actual_source_tree"
nvidia-smi \
  --query-gpu=index,name,memory.total \
  --format=csv,noheader,nounits

set +e
env -i \
  PATH="$PATH" \
  PYTHONPATH="$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  TORCHDYNAMO_DISABLE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  RANK=0 \
  LOCAL_RANK=0 \
  WORLD_SIZE=1 \
  LOCAL_WORLD_SIZE=1 \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=29549 \
  setsid timeout --signal=TERM --kill-after=30s 7m \
  "$PYTHON" - \
  "$PRODUCER" \
  "$RESOLVED_CONFIG" \
  "$CHECKPOINT_DIR" \
  "$EXECUTION_SOURCE" \
  "$EXPECTED_PRODUCER_SHA256" \
  "$EXPECTED_CHECKPOINT_EPOCH" \
  "$EXPECTED_PARAMETER_COUNT" > "$PAYLOAD_LOG" 2>&1 <<'PY'
import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf


(
    producer_text,
    resolved_config_text,
    checkpoint_dir_text,
    execution_source_text,
    expected_producer_sha,
    expected_epoch_text,
    expected_parameter_count_text,
) = sys.argv[1:]
producer_path = Path(producer_text).resolve(strict=True)
resolved_config = Path(resolved_config_text).resolve(strict=True)
checkpoint_dir = Path(checkpoint_dir_text).resolve(strict=True)
execution_source = Path(execution_source_text).resolve(strict=True)
expected_epoch = int(expected_epoch_text)
expected_parameter_count = int(expected_parameter_count_text)

if {name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")} != {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
}:
    raise ValueError("checkpoint smoke is not running as exactly one rank")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise ValueError("checkpoint smoke must expose only physical GPU 0")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("checkpoint smoke requires exactly one visible CUDA device")
torch.cuda.set_device(0)

spec = importlib.util.spec_from_file_location("frozen_checkpoint_smoke_producer", producer_path)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load producer from {producer_path}")
producer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)
if producer._sha256_file(producer_path) != expected_producer_sha:
    raise ValueError("loaded producer differs from the shell-bound producer")
legacy, runtime_support, canonical = producer._load_support_modules(producer_path)
if (
    legacy.__name__ != "frozen_one_step_legacy_support"
    or runtime_support.__name__ != "frozen_one_step_runtime"
    or canonical.__name__ != "frozen_one_step_canonical_support"
):
    raise ImportError("producer did not load the three frozen support modules")
if (
    runtime_support.MODEL_FILENAME != "MeshTransformer.0.491.mdlus"
    or runtime_support.TRAINING_STATE_FILENAME != "checkpoint.0.491.pt"
    or runtime_support.NORM_STATS_FILENAME != "norm_stats.pt"
):
    raise ValueError("frozen runtime checkpoint filenames changed")

cfg = OmegaConf.load(resolved_config)
if (
    str(cfg.precision) != "bfloat16"
    or bool(cfg.compile) is not True
    or int(cfg.training.seed) != 42
):
    raise ValueError("resolved checkpoint configuration changed")

import physicsnemo
import utils as recipe_utils

if Path(physicsnemo.__file__).resolve().is_relative_to(execution_source) is not True:
    raise ImportError("physicsnemo did not import from the staged execution source")
expected_recipe_src = (
    execution_source
    / "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
).resolve()
if Path(recipe_utils.__file__).resolve().parent != expected_recipe_src:
    raise ImportError("flat recipe utils did not import from the staged execution source")

runtime = SimpleNamespace(cfg=cfg, device=torch.device("cuda", 0))
model, optimizer, loaded_epoch = producer._new_model_optimizer(
    runtime,
    regime="checkpoint_epoch491",
    checkpoint_dir=checkpoint_dir,
)
if loaded_epoch != expected_epoch:
    raise ValueError(f"loaded epoch {loaded_epoch}, expected {expected_epoch}")
if optimizer.__class__.__name__ != "CombinedOptimizer":
    raise TypeError(f"unexpected optimizer class: {optimizer.__class__.__name__}")
members = tuple(member.__class__.__name__ for member in optimizer.optimizers)
if members != ("Muon", "AdamW"):
    raise TypeError(f"unexpected optimizer members: {members}")
state_entries = tuple(len(member.state) for member in optimizer.optimizers)
if len(state_entries) != 2 or any(entries <= 0 for entries in state_entries):
    raise ValueError(f"checkpoint optimizer member states are incomplete: {state_entries}")

learning_rate = producer._learning_rate(optimizer)
if not math.isfinite(learning_rate) or learning_rate <= 0.0:
    raise ValueError(f"checkpoint learning rate is invalid: {learning_rate}")
optimizer_hash = producer._stable_sha256(optimizer.state_dict())
if (
    re.fullmatch(r"[0-9a-f]{64}", optimizer_hash) is None
    or producer._stable_sha256(optimizer.state_dict()) != optimizer_hash
):
    raise ValueError("checkpoint optimizer state hash is invalid or unstable")

layout, layout_arrays = producer._parameter_layout(model)
if (
    layout["parameter_count"] != expected_parameter_count
    or int(layout_arrays["parameter_slice_starts_int64"][0]) != 0
    or int(layout_arrays["parameter_slice_stops_int64"][-1]) != expected_parameter_count
    or not np.array_equal(
        layout_arrays["parameter_slice_starts_int64"][1:],
        layout_arrays["parameter_slice_stops_int64"][:-1],
    )
):
    raise ValueError("checkpoint model parameter layout changed")
print(
    "CHECKPOINT_SMOKE_PASS "
    f"epoch={loaded_epoch} "
    "optimizer=CombinedOptimizer(Muon,AdamW) "
    f"state_entries={state_entries[0]},{state_entries[1]} "
    f"learning_rate={learning_rate.hex()} "
    f"optimizer_state_sha256={optimizer_hash} "
    f"parameter_count={layout['parameter_count']} "
    "categorical_scientific_outcome=NONE",
    flush=True,
)
PY
payload_rc=$?
set -e

if [[ "$payload_rc" -ne 0 ]]; then
  phase="BLOCKED_CHECKPOINT_VALIDITY_SMOKE"
  run_status="BLOCKED"
  tail -n 100 "$PAYLOAD_LOG" >&2 || true
  exit 20
fi
if [[ ! -f "$PAYLOAD_LOG" || -L "$PAYLOAD_LOG" ]]; then
  fail "checkpoint smoke log is missing or invalid"
fi
if [[ "$(grep -c '^CHECKPOINT_SMOKE_PASS ' "$PAYLOAD_LOG")" != 1 ]]; then
  fail "checkpoint smoke did not publish exactly one validity pass line"
fi
if grep -qE 'Traceback|CUDA out of memory|NCCL|srun: error' "$PAYLOAD_LOG"; then
  fail "checkpoint smoke log contains an unexplained runtime error"
fi

completed_units=1
phase="COMPLETED_CHECKPOINT_VALIDITY_SMOKE"
run_status="PASSED_CHECKPOINT_VALIDITY_SMOKE"
log_sha="$(sha256sum "$PAYLOAD_LOG" | awk '{print $1}')"
write_exclusive_marker \
  "$DONE_MARKER" \
  "VALIDITY_ONLY categorical_scientific_outcome=NONE payload_log_sha256=$log_sha"
echo "CHECKPOINT_SMOKE_DONE validity_only=true categorical_scientific_outcome=NONE payload_log_sha256=$log_sha"
