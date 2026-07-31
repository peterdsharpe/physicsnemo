# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the frozen Stage-B AGA launch package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import drivaerml_historical_k10000_replay as producer
import drivaerml_historical_k10000_replay_adjudicate as reducer
import drivaerml_historical_k10000_replay_runtime as runtime

STUDIES = Path(producer.__file__).resolve().parent
RESULTS = STUDIES.parent / "results"
WRAPPER = STUDIES / "drivaerml_historical_k10000_stage_b_replay_aga_v2.sbatch"
PREREGISTRATION = (
    STUDIES / "phase1_historical_k10000_stage_b_replay_prereg_v2_2026-07-28.json"
)
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
HISTORICAL_MANIFEST = (
    RESULTS
    / "phase1_historical_k10000_input_freeze_v1_2026-07-28"
    / "historical_predictions_tree_manifest.sha256"
)
HISTORICAL_METRICS = RESULTS / "hcc_historical_res10000_metrics_2026-07-24.jsonl"
INPUT_FREEZE = (
    RESULTS
    / "phase1_historical_k10000_input_freeze_v1_2026-07-28"
    / "historical_k10000_input_freeze_v1.json"
)
STAGE_A_LICENSE = (
    RESULTS
    / "historical_k10000_stage_a_archive_canary_v2_job306754_2026-07-28"
    / "artifacts"
    / "stage_a_archive_canary_adjudication.json"
)
STAGE_A_TREE_MANIFEST = (
    RESULTS
    / "historical_k10000_stage_a_archive_canary_v2_job306754_2026-07-28"
    / "sealed_remote_task_tree_manifest.sha256"
)
STAGE_A_RESULT_RECORD = (
    RESULTS
    / "phase1_historical_k10000_stage_a_archive_canary_result_v2_job306754_2026-07-28.json"
)
STAGE_A_PRODUCER_SHA256 = (
    "b596cb3d4a82b30255324b982f5d84d1260f53963b1697c7b2fd9d12049ed8c0"
)
STAGE_A_REDUCER_SHA256 = (
    "fdb232e337eeb54be4f1782f6b619f53e5ad4ed0c2aca366b504d035cab023e0"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_OUTCOMES = [
    reducer.EXACT_OUTCOME,
    reducer.VALID_REFUTATION,
    reducer.INVALID_REPLAY,
    reducer.INCOMPLETE_REPLAY,
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapper_text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _assert_canonical_sidecar(path: Path) -> None:
    expected = f"{_digest(path)}  {path.name}\n".encode("ascii")
    assert path.with_name(f"{path.name}.sha256").read_bytes() == expected


def _producer_lane_block(source: str) -> str:
    return source.split("# BEGIN PRODUCER_LANES", 1)[1].split(
        "# END PRODUCER_LANES",
        1,
    )[0]


def test_wrapper_has_valid_bash_syntax():
    subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(WRAPPER)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_wrapper_embedded_python_blocks_compile():
    source = _wrapper_text()
    blocks = [
        remainder.split("\nPY", 1)[0] for remainder in source.split("<<'PY'\n")[1:]
    ]
    assert len(blocks) == 2
    for ordinal, block in enumerate(blocks):
        compile(block, f"{WRAPPER.name}:heredoc-{ordinal}", "exec")


def test_wrapper_and_preregistration_bind_current_frozen_bytes():
    source = _wrapper_text()
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    expected = {
        "EXPECTED_PRODUCER_SHA256": _digest(Path(producer.__file__).resolve()),
        "EXPECTED_HELPER_SHA256": _digest(Path(runtime.__file__).resolve()),
        "EXPECTED_REDUCER_SHA256": _digest(Path(reducer.__file__).resolve()),
        "EXPECTED_PREREGISTRATION_SHA256": _digest(PREREGISTRATION),
        "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256": (EXPECTED_SOURCE_MANIFEST_SHA256),
        "EXPECTED_EXECUTION_SOURCE_TREE_SHA256": EXPECTED_SOURCE_TREE_SHA256,
        "EXPECTED_GEOMETRY_MANIFEST_SHA256": _digest(GEOMETRY_MANIFEST),
        "EXPECTED_TARGET_MANIFEST_SHA256": _digest(TARGET_MANIFEST),
        "EXPECTED_HISTORICAL_MANIFEST_SHA256": _digest(HISTORICAL_MANIFEST),
        "EXPECTED_HISTORICAL_METRICS_SHA256": _digest(HISTORICAL_METRICS),
        "EXPECTED_INPUT_FREEZE_SHA256": _digest(INPUT_FREEZE),
        "EXPECTED_STAGE_A_LICENSE_SHA256": _digest(STAGE_A_LICENSE),
        "EXPECTED_STAGE_A_TREE_MANIFEST_SHA256": _digest(STAGE_A_TREE_MANIFEST),
        "EXPECTED_STAGE_A_RESULT_RECORD_SHA256": _digest(STAGE_A_RESULT_RECORD),
        "EXPECTED_STAGE_A_PRODUCER_SHA256": STAGE_A_PRODUCER_SHA256,
        "EXPECTED_STAGE_A_REDUCER_SHA256": STAGE_A_REDUCER_SHA256,
    }
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source

    assert producer.EXPECTED_HELPER_SHA256 == expected["EXPECTED_HELPER_SHA256"]
    assert reducer.EXPECTED_PRODUCER_SHA256 == expected["EXPECTED_PRODUCER_SHA256"]
    frozen = preregistration["frozen_provenance"]
    assert frozen["producer_sha256"] == expected["EXPECTED_PRODUCER_SHA256"]
    assert frozen["runtime_helper_sha256"] == expected["EXPECTED_HELPER_SHA256"]
    assert frozen["reducer_sha256"] == expected["EXPECTED_REDUCER_SHA256"]
    assert frozen["execution_source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST_SHA256
    assert frozen["execution_source_tree_sha256"] == EXPECTED_SOURCE_TREE_SHA256
    assert (
        frozen["target_free_geometry_manifest_sha256"]
        == expected["EXPECTED_GEOMETRY_MANIFEST_SHA256"]
    )
    assert (
        frozen["selected_target_manifest_sha256"]
        == expected["EXPECTED_TARGET_MANIFEST_SHA256"]
    )
    assert (
        frozen["historical_prediction_tree_manifest_sha256"]
        == expected["EXPECTED_HISTORICAL_MANIFEST_SHA256"]
    )
    assert (
        frozen["historical_metrics_sha256"]
        == expected["EXPECTED_HISTORICAL_METRICS_SHA256"]
    )
    assert (
        frozen["stage_a_adjudication_sha256"]
        == expected["EXPECTED_STAGE_A_LICENSE_SHA256"]
    )
    assert (
        frozen["stage_a_sealed_task_tree_manifest_sha256"]
        == expected["EXPECTED_STAGE_A_TREE_MANIFEST_SHA256"]
    )
    assert (
        frozen["stage_a_result_record_sha256"]
        == expected["EXPECTED_STAGE_A_RESULT_RECORD_SHA256"]
    )
    _assert_canonical_sidecar(TARGET_MANIFEST)
    _assert_canonical_sidecar(GEOMETRY_MANIFEST)
    _assert_canonical_sidecar(HISTORICAL_MANIFEST)
    _assert_canonical_sidecar(HISTORICAL_METRICS)
    _assert_canonical_sidecar(INPUT_FREEZE)
    _assert_canonical_sidecar(STAGE_A_LICENSE)
    _assert_canonical_sidecar(STAGE_A_TREE_MANIFEST)
    _assert_canonical_sidecar(STAGE_A_RESULT_RECORD)


def test_stage_a_license_is_exact_and_scoped_to_this_stage():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    license_document = json.loads(STAGE_A_LICENSE.read_bytes())
    result_document = json.loads(STAGE_A_RESULT_RECORD.read_bytes())
    license_contract = preregistration["stage_a_license"]
    assert license_contract == {
        "required_outcome": "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS",
        "job_id": 306754,
        "adjudication_sha256": _digest(STAGE_A_LICENSE),
        "sealed_task_tree_manifest_sha256": _digest(STAGE_A_TREE_MANIFEST),
        "sealed_task_file_count": 954,
        "sealed_task_size_bytes": 14889440,
        "result_record_sha256": _digest(STAGE_A_RESULT_RECORD),
        "producer_sha256": STAGE_A_PRODUCER_SHA256,
        "reducer_sha256": STAGE_A_REDUCER_SHA256,
        "checkpoint_load_epoch": 491,
        "scope": "licenses only this freshly frozen Stage-B v2 reconstructed replay",
    }
    assert license_document["status"] == (
        "COMPLETED_STAGE_A_ARCHIVE_DOMAIN_ADJUDICATION"
    )
    assert license_document["outcome"] == "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS"
    assert result_document["job"]["job_id"] == 306754
    assert result_document["outcome"]["adjudication_sha256"] == _digest(STAGE_A_LICENSE)
    assert result_document["license"]["status"] == "ACTIVE_FOR_FRESH_STAGE_B_V2_ONLY"
    assert (
        f"{_digest(STAGE_A_LICENSE)}  "
        "./artifacts/stage_a_archive_canary_adjudication.json"
        in STAGE_A_TREE_MANIFEST.read_text(encoding="ascii").splitlines()
    )


def test_wrapper_preserves_whole_node_two_lane_process_isolation():
    source = _wrapper_text()
    assert "#SBATCH --export=NIL" in source
    assert "#SBATCH -N 1" in source
    assert "#SBATCH --gpus-per-node=4" in source
    assert "#SBATCH --time=00:45:00" in source
    assert "EXPECTED_TASK_PHYSICAL=/scratch/fsw/" in source
    assert "visible_gpu_count" in source

    lane_block = _producer_lane_block(source)
    assert lane_block.count("env -i") == 2
    assert lane_block.count("CUDA_VISIBLE_DEVICES=0") == 1
    assert lane_block.count("CUDA_VISIBLE_DEVICES=1") == 1
    assert lane_block.count("32m") == 2
    assert lane_block.count("-m torch.distributed.run") == 2
    assert lane_block.count("--standalone --nproc_per_node=1") == 2
    assert "--replay-label A" in lane_block
    assert "--replay-label B" in lane_block
    assert lane_block.count('cd "$DATASET_PHYSICAL"') == 2
    assert source.index("# END PRODUCER_LANES") < source.index(
        'wait "${lane_pids[$lane_index]}"'
    )


def test_producer_lanes_receive_required_inputs_but_no_historical_oracle():
    lane_block = _producer_lane_block(_wrapper_text())
    for argument in (
        "--repo-root",
        "--dataset-root",
        "--resolved-config",
        "--dataset-config",
        "--checkpoint-dir",
        "--geometry-manifest",
        "--target-input-manifest",
    ):
        assert lane_block.count(argument) == 2
    for forbidden in (
        "--historical-predictions",
        "--historical-manifest",
        "--historical-metrics",
        "--normalization-state",
        "HISTORICAL_ARCHIVE",
        "HISTORICAL_MANIFEST",
        "HISTORICAL_METRICS",
        "INPUT_FREEZE",
        "STAGE_A_LICENSE",
    ):
        assert forbidden not in lane_block
    assert "--helper" not in lane_block
    assert lane_block.count('PYTHONPATH="$PYTHONPATH"') == 2


def test_reducer_alone_receives_archive_metrics_and_normalization():
    source = _wrapper_text()
    reducer_block = source.split('phase="REDUCTION"', 1)[1]
    for argument in (
        "--producer",
        "--producer-a-json",
        "--producer-a-npz",
        "--producer-b-json",
        "--producer-b-npz",
        "--target-input-manifest",
        "--historical-predictions",
        "--historical-manifest",
        "--historical-metrics",
        "--normalization-state",
    ):
        assert argument in reducer_block
    assert "timeout --signal=TERM --kill-after=30s 8m" in reducer_block
    assert source.index('if [[ "$lane_artifacts_valid" != true ]]') < source.index(
        'phase="REDUCTION"'
    )


def test_wrapper_preflight_verifies_exact_source_archive_and_inputs():
    source = _wrapper_text()
    assert "execution-source manifest content verification failed" in source
    assert "execution source tree changed" in source
    assert "EXPECTED_HISTORICAL_FILE_COUNT=1656" in source
    assert "EXPECTED_HISTORICAL_SIZE_BYTES=44790588" in source
    assert "historical archive all-file hash verification failed" in source
    assert 'check_sha "$EXPECTED_DATASET_MANIFEST_SHA256"' in source
    assert 'check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256"' in source
    assert 'check_sha "$EXPECTED_TARGET_MANIFEST_SHA256"' in source
    assert 'check_sha "$EXPECTED_STAGE_A_LICENSE_SHA256"' in source
    assert 'check_sha "$EXPECTED_STAGE_A_TREE_MANIFEST_SHA256"' in source
    assert 'verify_canonical_sidecar "$TARGET_INPUT_MANIFEST"' in source
    assert 'verify_canonical_sidecar "$GEOMETRY_MANIFEST"' in source
    assert 'verify_canonical_sidecar "$HISTORICAL_MANIFEST"' in source
    assert 'verify_canonical_sidecar "$HISTORICAL_METRICS"' in source
    assert 'verify_canonical_sidecar "$INPUT_FREEZE"' in source
    assert 'verify_canonical_sidecar "$STAGE_A_LICENSE"' in source
    assert 'verify_canonical_sidecar "$STAGE_A_TREE_MANIFEST"' in source
    assert "Stage-A v2 seal does not bind the exact adjudication artifact" in source
    assert "Stage-A artifact does not license Stage B" in source


def test_preregistration_fixes_truth_table_controls_and_scientific_scope():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    assert preregistration["status"] == "PREREGISTERED_PRELAUNCH"
    truth_table = preregistration["adjudication"]["truth_table"]
    assert [row["outcome"] for row in truth_table] == EXPECTED_OUTCOMES
    semantics = preregistration["adjudication"]["execution_exit_semantics"]
    assert "successful completed job" in semantics[reducer.EXACT_OUTCOME]
    assert "successful completed job" in semantics[reducer.VALID_REFUTATION]
    assert "blocked nonzero wrapper exit" in semantics[reducer.INVALID_REPLAY]
    assert "blocked nonzero wrapper exit" in semantics[reducer.INCOMPLETE_REPLAY]

    controls = preregistration["deciding_controls"]
    assert controls["pipeline_normal_absolute_tolerance"] == (
        reducer.PIPELINE_NORMAL_ABS_TOLERANCE
    )
    assert controls["training_physical_absolute_tolerance"] == (
        reducer.TRAINING_PHYSICAL_ABS_TOLERANCE
    )
    assert controls["historical_pressure_case_absolute_tolerance"] == (
        reducer.PRESSURE_CASE_ABS_TOLERANCE
    )
    assert controls["historical_pressure_mean_absolute_tolerance"] == (
        reducer.PRESSURE_MEAN_ABS_TOLERANCE
    )

    baseline = preregistration["corrected_baseline_contract"]
    assert baseline["licensed_only_on_exact_outcome"] is True
    assert baseline["prospective_noninferiority_ratio"] == (
        reducer.NONINFERIORITY_RATIO
    )
    assert "descriptive only" in baseline["historical_pointwise_mean_wss_relative_l2"]
    assert "absolute native areas" in baseline["does_not_support"]
    assert "canonical-path accuracy" in baseline["does_not_support"]
    scope = preregistration["scope"]
    assert "not an operating-system filesystem sandbox" in scope["isolation_semantics"]
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
        assert scope[key] is False
    lane_contract = preregistration["lane_contract"]
    assert (
        "deterministic internal functions"
        in lane_contract["derived_geometry_semantics"]
    )
    assert "redundant compatibility control" in lane_contract["pipeline_normal_control"]
    assert not any(preregistration["launch_state"].values())


def test_v2_operational_package_excludes_withdrawn_v1_bindings():
    combined = _wrapper_text() + PREREGISTRATION.read_text(encoding="utf-8")
    for withdrawn in (
        "306623",
        "74b482c39501c556798fb7ab6c611912c33dc9aea336718b92cec210d243dbb7",
        "8cf6f8351df329cacaeb0f90030686c211ba895eebd2d5c60c788c7b967bc75d",
        "7fb49c676fd894e13b0e5710ca4185babb99b09b1d73453b91dd1988bc3956ed",
        "c0eaa523726573afc2c5cabf11064dad864986d3ad70aff92d3c279177593d95",
    ):
        assert withdrawn not in combined


def test_wrapper_treats_refutation_as_success_and_invalidity_as_blocked():
    source = _wrapper_text()
    outcome_switch = source.split('case "$categorical_outcome" in', 1)[1]
    exact_arm = outcome_switch.split("EXACT_HISTORICAL_REPLAY_PASS)", 1)[1].split(
        "VALID_EXACT_REPLAY_REFUTATION)",
        1,
    )[0]
    refutation_arm = outcome_switch.split(
        "VALID_EXACT_REPLAY_REFUTATION)",
        1,
    )[1].split("INVALID_REPLAY)", 1)[0]
    invalid_arm = outcome_switch.split("INVALID_REPLAY)", 1)[1].split(
        "INCOMPLETE_REPLAY)",
        1,
    )[0]
    incomplete_arm = outcome_switch.split("INCOMPLETE_REPLAY)", 1)[1].split(
        "*)",
        1,
    )[0]
    assert "DONE_${SLURM_JOB_ID}" in exact_arm
    assert "exit " not in exact_arm
    assert "DONE_${SLURM_JOB_ID}" in refutation_arm
    assert "exit " not in refutation_arm
    assert "BLOCKED_${SLURM_JOB_ID}" in invalid_arm
    assert "exit 30" in invalid_arm
    assert "BLOCKED_${SLURM_JOB_ID}" in incomplete_arm
    assert "exit 31" in incomplete_arm


def test_wrapper_enforces_no_clobber_lock_sidecars_and_status_marker():
    source = _wrapper_text()
    assert "set -o noclobber" in source
    assert ".stage-b-replay-v2.lock" in source
    assert "flock -n 9" in source
    assert "verify_canonical_sidecar" in source
    assert "STATUS_${SLURM_JOB_ID:-no_job}" in source
    assert "STAGE_B_CATEGORICAL_OUTCOME=" in source
    assert "DONE_${SLURM_JOB_ID}" in source
    assert "BLOCKED_${SLURM_JOB_ID}" in source
