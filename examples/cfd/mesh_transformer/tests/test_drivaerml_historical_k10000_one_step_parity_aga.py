# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the frozen one-step-parity AGA launch package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

TESTS = Path(__file__).resolve().parent
MESH_TRANSFORMER = TESTS.parent
STUDIES = MESH_TRANSFORMER / "studies"
RESULTS = MESH_TRANSFORMER / "results"

WRAPPER = STUDIES / "drivaerml_historical_k10000_one_step_parity_aga.sbatch"
PRODUCER = STUDIES / "drivaerml_historical_k10000_one_step_parity.py"
REDUCER = STUDIES / "drivaerml_historical_k10000_one_step_parity_adjudicate.py"
PRODUCER_TEST = TESTS / "test_drivaerml_historical_k10000_one_step_parity.py"
REDUCER_TEST = TESTS / "test_drivaerml_historical_k10000_one_step_parity_adjudicate.py"
PREREGISTRATION = (
    STUDIES / "phase1_historical_k10000_one_step_parity_prereg_v3_2026-07-29.json"
)
LEGACY_HELPER = STUDIES / "drivaerml_historical_k10000_replay.py"
RUNTIME_HELPER = STUDIES / "drivaerml_historical_k10000_replay_runtime.py"
CANONICAL_HELPER = STUDIES / "drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
STAGE_B_SEAL = RESULTS / "historical_k10000_stage_b_replay_v2_job306814_2026-07-28"
EXECUTION_SOURCE_MANIFEST = STAGE_B_SEAL / "execution_source_manifest.sha256"
GEOMETRY_MANIFEST = (
    RESULTS
    / "geometry_input_manifest_36cases_job305850_2026-07-28"
    / "artifacts"
    / "drivaerml_geometry_input_manifest_36cases_v1.json"
)
TARGET_MANIFEST = (
    RESULTS
    / "historical_k10000_selected_target_input_freeze_job306302_2026-07-28"
    / "artifacts"
    / "historical_k10000_selected_target_input_manifest_v1.json"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
SUCCESS_OUTCOME = "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS"
FAIL_OUTCOME = "MATERIAL_PARITY_DIFFERENCE_FAIL"
INVALID_OUTCOME = "INVALID_ONE_STEP_PARITY_COMPARISON"
INCOMPLETE_OUTCOME = "INCOMPLETE_ONE_STEP_PARITY_COMPARISON"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _block(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


def _assert_canonical_sidecar(path: Path) -> None:
    expected = f"{_digest(path)}  {path.name}\n".encode("ascii")
    assert path.with_name(f"{path.name}.sha256").read_bytes() == expected


def test_wrapper_has_valid_bash_syntax():
    subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(WRAPPER)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_embedded_python_blocks_compile():
    source = _source()
    blocks = [
        remainder.split("\nPY", 1)[0] for remainder in source.split("<<'PY'\n")[1:]
    ]
    assert len(blocks) == 2
    for ordinal, block in enumerate(blocks):
        compile(block, f"{WRAPPER.name}:heredoc-{ordinal}", "exec")


def test_wrapper_binds_every_frozen_local_input():
    source = _source()
    expected = {
        "EXPECTED_PRODUCER_SHA256": _digest(PRODUCER),
        "EXPECTED_REDUCER_SHA256": _digest(REDUCER),
        "EXPECTED_PREREGISTRATION_SHA256": _digest(PREREGISTRATION),
        "EXPECTED_PRODUCER_TEST_SHA256": _digest(PRODUCER_TEST),
        "EXPECTED_REDUCER_TEST_SHA256": _digest(REDUCER_TEST),
        "EXPECTED_LEGACY_HELPER_SHA256": _digest(LEGACY_HELPER),
        "EXPECTED_RUNTIME_HELPER_SHA256": _digest(RUNTIME_HELPER),
        "EXPECTED_CANONICAL_HELPER_SHA256": _digest(CANONICAL_HELPER),
        "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256": (
            _digest(EXECUTION_SOURCE_MANIFEST)
        ),
        "EXPECTED_GEOMETRY_MANIFEST_SHA256": _digest(GEOMETRY_MANIFEST),
        "EXPECTED_TARGET_MANIFEST_SHA256": _digest(TARGET_MANIFEST),
    }
    assert (
        expected["EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256"]
        == EXPECTED_SOURCE_MANIFEST_SHA256
    )
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source
    assert (
        f"readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256={EXPECTED_SOURCE_TREE_SHA256}"
    ) in source
    assert "readonly EXPECTED_EXECUTION_SOURCE_FILE_COUNT=919" in source
    assert "readonly EXPECTED_EXECUTION_SOURCE_SIZE_BYTES=10090497" in source

    preregistration = json.loads(PREREGISTRATION.read_bytes())
    assert preregistration["frozen_launch_bindings"] == {
        "producer_sha256": expected["EXPECTED_PRODUCER_SHA256"],
        "adjudicator_sha256": expected["EXPECTED_REDUCER_SHA256"],
        "producer_test_sha256": expected["EXPECTED_PRODUCER_TEST_SHA256"],
        "adjudicator_test_sha256": expected["EXPECTED_REDUCER_TEST_SHA256"],
        "legacy_replay_support_sha256": expected["EXPECTED_LEGACY_HELPER_SHA256"],
        "runtime_helper_sha256": expected["EXPECTED_RUNTIME_HELPER_SHA256"],
        "canonical_geometry_helper_v5_sha256": expected[
            "EXPECTED_CANONICAL_HELPER_SHA256"
        ],
        "execution_source_manifest_sha256": expected[
            "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256"
        ],
        "execution_source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "geometry_manifest_sha256": expected["EXPECTED_GEOMETRY_MANIFEST_SHA256"],
        "target_input_manifest_sha256": expected["EXPECTED_TARGET_MANIFEST_SHA256"],
    }
    hash_cycle_policy = preregistration["hash_cycle_policy"].lower()
    assert "wrapper binds" in hash_cycle_policy
    assert "prereg" in hash_cycle_policy
    assert "excludes wrapper sha" in hash_cycle_policy


def test_frozen_manifest_sidecars_are_canonical():
    _assert_canonical_sidecar(GEOMETRY_MANIFEST)
    _assert_canonical_sidecar(TARGET_MANIFEST)


def test_preregistration_fixes_panel_optimizer_and_outcome_semantics():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    assert preregistration["schema_version"] == 1
    assert preregistration["status"] == "PREREGISTERED_PRELAUNCH"
    panel = preregistration["fixed_panel"]
    assert panel["resolution"] == 10_000
    assert [case["case_id"] for case in panel["cases"]] == [
        "run_118",
        "run_271",
        "run_429",
        "run_86",
    ]
    assert panel["regimes"] == ["fresh_seed42", "checkpoint_epoch491"]
    assert panel["precisions"] == {
        "bfloat16": "deciding",
        "float32": "diagnostic only",
    }
    path_contract = preregistration["path_and_measure_contract"]
    assert "CombinedOptimizer" in path_contract["optimizer"]
    assert (
        "explicitly absent in both arms"
        in path_contract["source_measure_weights"].lower()
    )
    assert "Preserved" in path_contract["target_measure"]
    assert [row["outcome"] for row in preregistration["truth_table"]] == [
        SUCCESS_OUTCOME,
        FAIL_OUTCOME,
        INVALID_OUTCOME,
        INCOMPLETE_OUTCOME,
    ]
    adjudication = preregistration["adjudication"]
    assert adjudication["valid_pass_and_valid_fail_exit_code"] == 0
    assert adjudication["invalid_exit_code"] == 2
    assert adjudication["incomplete_exit_code"] == 3
    launch_state = preregistration["launch_state"]
    for flag in (
        "task_staged",
        "scheduler_test_run",
        "job_submitted",
        "producer_output_exists",
        "adjudication_output_exists",
    ):
        assert launch_state[flag] is False


def test_preregistration_records_both_invalid_launches_as_no_evidence():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    correction_history = preregistration["correction_history"]
    corrections_by_job = {
        entry["supersedes"]["launch_job_id"]: entry["invalid_launch"]
        for entry in correction_history
    }
    assert set(corrections_by_job) == {307502, 307511}
    for job_id in (307502, 307511):
        invalid_launch = corrections_by_job[job_id]
        assert invalid_launch["scientific_evidence"] == "NONE"
        assert invalid_launch["completed_units"] == "0/2"


def test_wrapper_requests_one_aga_node_and_uses_one_rank_on_one_gpu():
    source = _source()
    for directive in (
        "#SBATCH -J mt-k10k-step-parity-v3",
        "#SBATCH --time=00:45:00",
        "#SBATCH -N 1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=24",
        "#SBATCH --mem=256G",
        "#SBATCH --export=NIL",
    ):
        assert directive in source
    assert "2026-07-29-mt-historical-k10k-one-step-parity-v3" in source
    assert "readonly UNIT_COUNT=2" in source
    producer = _block(
        source,
        "# BEGIN ONE_STEP_PRODUCER",
        "# END ONE_STEP_PRODUCER",
    )
    assert producer.count("env -i") == 1
    assert producer.count("setsid timeout") == 1
    assert producer.count("CUDA_VISIBLE_DEVICES=0") == 1
    assert "CUDA_VISIBLE_DEVICES=1" not in source
    assert "CUDA_VISIBLE_DEVICES=2" not in source
    assert "CUDA_VISIBLE_DEVICES=3" not in source
    assert producer.count("-m torch.distributed.run") == 1
    assert producer.count("--standalone --nproc_per_node=1") == 1
    assert 'cd "$DATASET_PHYSICAL"' in producer


def test_wrapper_requires_the_minimal_staged_root_inventory():
    source = _source()
    assert (
        "task-root directory inventory differs from the minimal launch package"
        in source
    )
    assert "task-root file inventory differs from the minimal launch package" in source
    assert "task root contains a symlink or special entry" in source
    for filename in (
        WRAPPER.name,
        PRODUCER.name,
        REDUCER.name,
        PREREGISTRATION.name,
        LEGACY_HELPER.name,
        RUNTIME_HELPER.name,
        CANONICAL_HELPER.name,
        EXECUTION_SOURCE_MANIFEST.name,
        GEOMETRY_MANIFEST.name,
        f"{GEOMETRY_MANIFEST.name}.sha256",
        TARGET_MANIFEST.name,
        f"{TARGET_MANIFEST.name}.sha256",
    ):
        assert filename in source


def test_producer_receives_exact_epoch_inputs_and_compile_is_disabled():
    source = _source()
    producer = _block(
        source,
        "# BEGIN ONE_STEP_PRODUCER",
        "# END ONE_STEP_PRODUCER",
    )
    for argument in (
        "--repo-root",
        "--dataset-root",
        "--dataset-config",
        "--resolved-config",
        "--checkpoint-dir",
        "--geometry-manifest",
        "--target-input-manifest",
        "--output-json",
        "--output-npz",
    ):
        assert argument in producer
    assert "EXPECTED_CHECKPOINT_EPOCH=491" in source
    assert "MeshTransformer.0.491.mdlus" in source
    assert "checkpoint.0.491.pt" in source
    assert "TORCHDYNAMO_DISABLE=1" in producer
    assert "COMPILE_ENABLED=false" in source
    producer_source = PRODUCER.read_text(encoding="utf-8")
    assert "CHECKPOINT_EPOCH = 491" in producer_source
    assert "epoch=CHECKPOINT_EPOCH" in producer_source
    assert "compile_optimizer=False" in producer_source
    assert '"compile_enabled": False' in producer_source


def test_reducer_runs_only_after_verified_producer_artifacts():
    source = _source()
    reducer = _block(
        source,
        "# BEGIN ONE_STEP_REDUCER",
        "# END ONE_STEP_REDUCER",
    )
    for argument in (
        "--producer-json",
        "--producer-npz",
        "--output-json",
    ):
        assert argument in reducer
    assert "CUDA_VISIBLE_DEVICES=" in reducer
    assert "8m" in reducer
    producer_verification = (
        'if ! verify_canonical_sidecar "$PRODUCER_JSON" ||\n'
        '  ! verify_canonical_sidecar "$PRODUCER_NPZ"; then'
    )
    assert source.index(producer_verification) < source.index(
        "# BEGIN ONE_STEP_REDUCER"
    )


def test_preflight_rehashes_source_and_every_live_training_dependency():
    source = _source()
    for required in (
        "execution-source count or byte size changed",
        "execution-source manifest content verification failed",
        "execution source tree changed",
        'check_sha "$EXPECTED_DATASET_MANIFEST_SHA256"',
        'check_sha "$EXPECTED_DATASET_CONFIG_SHA256"',
        'check_sha "$EXPECTED_RESOLVED_CONFIG_SHA256"',
        'check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256"',
        'check_sha "$EXPECTED_TRAINING_STATE_SHA256"',
        'check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256"',
        'check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256"',
        'check_sha "$EXPECTED_TARGET_MANIFEST_SHA256"',
    ):
        assert required in source


def test_marker_and_exit_semantics_do_not_turn_valid_fail_into_job_failure():
    source = _source()
    assert "set -o noclobber" in source
    assert ".one-step-parity-v3.lock" in source
    assert "flock -n 9" in source
    assert "STATUS_${SLURM_JOB_ID:-no_job}" in source
    assert "producer_json_sha256=" in source
    assert "producer_npz_sha256=" in source
    assert "adjudication_sha256=" in source
    assert "DONE_${SLURM_JOB_ID}" in source
    assert "INVALID_${SLURM_JOB_ID}" in source
    assert "BLOCKED_${SLURM_JOB_ID}" in source

    outcome_switch = source.split('case "$categorical_outcome" in', 1)[1]
    valid_arm = outcome_switch.split(
        "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS | MATERIAL_PARITY_DIFFERENCE_FAIL)",
        1,
    )[1].split("INVALID_ONE_STEP_PARITY_COMPARISON)", 1)[0]
    invalid_arm = outcome_switch.split(
        "INVALID_ONE_STEP_PARITY_COMPARISON)",
        1,
    )[1].split("INCOMPLETE_ONE_STEP_PARITY_COMPARISON)", 1)[0]
    incomplete_arm = outcome_switch.split(
        "INCOMPLETE_ONE_STEP_PARITY_COMPARISON)",
        1,
    )[1].split("*)", 1)[0]
    assert "DONE_${SLURM_JOB_ID}" in valid_arm
    assert "exit " not in valid_arm
    assert "INVALID_${SLURM_JOB_ID}" in invalid_arm
    assert "DONE_${SLURM_JOB_ID}" not in invalid_arm
    assert "exit 2" in invalid_arm
    assert "INVALID_${SLURM_JOB_ID}" in incomplete_arm
    assert "DONE_${SLURM_JOB_ID}" not in incomplete_arm
    assert "exit 3" in incomplete_arm


def test_wrapper_has_no_resubmission_or_requeue_path():
    source = _source()
    commands = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("sbatch ") for line in commands)
    assert not any("scontrol requeue" in line for line in commands)
    assert "--dependency=" not in source
    assert "--requeue" not in source
