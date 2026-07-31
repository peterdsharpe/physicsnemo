# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the epoch-pinned Stage-A v2 AGA package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import drivaerml_historical_k10000_stage_a_archive_canary as producer
import drivaerml_historical_k10000_stage_a_archive_canary_adjudicate as reducer

STUDIES = Path(producer.__file__).resolve().parent
WRAPPER = STUDIES / "drivaerml_historical_k10000_stage_a_archive_canary_aga_v2.sbatch"
PREREGISTRATION = (
    STUDIES
    / "phase1_historical_k10000_stage_a_archive_canary_prereg_v2_2026-07-28.json"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_ARCHIVE_MANIFEST_SHA256 = (
    "545b1f6e906002231415b84277db00eec04f3666233b8da637514e9077a585eb"
)
EXPECTED_INPUT_FREEZE_SHA256 = (
    "fce9444a11b0a6b71497d927573728c3d10f9da3e480a9b05dacd50505b6fe10"
)
EXPECTED_OUTCOMES = [
    "EXACT_STAGE_A_ARCHIVE_DOMAIN_PASS",
    "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION",
    "INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT",
    "INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapper_text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


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


def test_wrapper_binds_current_frozen_artifact_hashes():
    source = _wrapper_text()
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    expected = {
        "EXPECTED_PRODUCER_SHA256": _digest(Path(producer.__file__).resolve()),
        "EXPECTED_REDUCER_SHA256": _digest(Path(reducer.__file__).resolve()),
        "EXPECTED_PREREGISTRATION_SHA256": _digest(PREREGISTRATION),
        "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256": (EXPECTED_SOURCE_MANIFEST_SHA256),
        "EXPECTED_EXECUTION_SOURCE_TREE_SHA256": EXPECTED_SOURCE_TREE_SHA256,
        "EXPECTED_HISTORICAL_MANIFEST_SHA256": (EXPECTED_ARCHIVE_MANIFEST_SHA256),
        "EXPECTED_INPUT_FREEZE_SHA256": EXPECTED_INPUT_FREEZE_SHA256,
    }
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source

    frozen = preregistration["frozen_provenance"]
    assert frozen["producer_sha256"] == expected["EXPECTED_PRODUCER_SHA256"]
    assert frozen["reducer_sha256"] == expected["EXPECTED_REDUCER_SHA256"]
    assert frozen["execution_source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST_SHA256
    assert frozen["execution_source_tree_sha256"] == EXPECTED_SOURCE_TREE_SHA256
    assert (
        frozen["historical_prediction_tree_manifest_sha256"]
        == EXPECTED_ARCHIVE_MANIFEST_SHA256
    )
    assert (
        frozen["historical_input_freeze_record_sha256"] == EXPECTED_INPUT_FREEZE_SHA256
    )


def test_wrapper_preserves_aga_whole_node_and_two_lane_contract():
    source = _wrapper_text()
    assert "#SBATCH -J mt-k10k-stagea-v2" in source
    assert "#SBATCH --export=NIL" in source
    assert "#SBATCH -N 1" in source
    assert "#SBATCH --gpus-per-node=4" in source
    assert "#SBATCH --time=00:30:00" in source
    assert "EXPECTED_TASK_PHYSICAL=/scratch/fsw/" in source
    assert "HISTORICAL_ARCHIVE_PHYSICAL=/scratch/fsw/" in source
    assert "visible_gpu_count" in source
    assert "2026-07-28-mt-historical-k10k-stage-a-v2" in source
    assert "2026-07-28-mt-historical-k10k-stage-a-v1" not in source

    lane_block = _producer_lane_block(source)
    assert lane_block.count("env -i") == 2
    assert lane_block.count("CUDA_VISIBLE_DEVICES=0") == 1
    assert lane_block.count("CUDA_VISIBLE_DEVICES=1") == 1
    assert lane_block.count("16m") == 2
    assert lane_block.count("-m torch.distributed.run") == 2
    assert lane_block.count("--standalone --nproc_per_node=1") == 2
    assert "--lane-label A" in lane_block
    assert "--lane-label B" in lane_block
    assert "lane_A.log" not in lane_block
    assert '"$LANE_A_LOG"' in lane_block
    assert '"$LANE_B_LOG"' in lane_block
    assert source.index("# END PRODUCER_LANES") < source.index(
        'wait "${lane_pids[$lane_index]}"'
    )


def test_producer_lane_commands_receive_no_oracle_bearing_artifact():
    source = _wrapper_text()
    lane_block = _producer_lane_block(source)
    assert "--archive-input-root" in lane_block
    assert "--historical-predictions" not in lane_block
    assert "--historical-predictions-manifest" not in lane_block
    assert "--input-freeze" not in lane_block
    assert "HISTORICAL_ARCHIVE" not in lane_block
    assert "HISTORICAL_MANIFEST" not in lane_block
    assert "INPUT_FREEZE" not in lane_block
    assert lane_block.count('cd "$ARCHIVE_INPUT_ROOT"') == 2
    assert lane_block.count('PYTHONPATH="$PYTHONPATH"') == 2

    reducer_block = source.split('phase="REDUCTION"', 1)[1]
    assert "--historical-predictions" in reducer_block
    assert "--historical-predictions-manifest" in reducer_block
    assert "timeout --signal=TERM --kill-after=30s 5m" in reducer_block
    assert source.index('if [[ "$lane_artifacts_valid" != true ]]') < source.index(
        'phase="REDUCTION"'
    )


def test_wrapper_preflight_verifies_exact_source_archive_and_input_inventories():
    source = _wrapper_text()
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    payloads = preregistration["producer_input_isolation"]["payloads"]
    assert len(payloads) == len(producer.EXPECTED_ARCHIVE_INPUT_SHA256) == 10
    assert {row["name"]: row["sha256"] for row in payloads} == dict(
        producer.EXPECTED_ARCHIVE_INPUT_SHA256
    )
    for row in payloads:
        assert row["relative_path"] in source
        assert row["sha256"] in source

    assert "EXPECTED_ARCHIVE_INPUT_FILE_COUNT=10" in source
    assert "EXPECTED_ARCHIVE_INPUT_SIZE_BYTES=719432" in source
    assert "EXPECTED_HISTORICAL_FILE_COUNT=1656" in source
    assert "EXPECTED_HISTORICAL_SIZE_BYTES=44790588" in source
    assert "historical archive all-file hash verification failed" in source
    assert "execution-source manifest content verification failed" in source
    assert "archive input inventory is missing one or more exact payloads" in source


def test_preregistration_fixes_geometry_definition_truth_table_and_license():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    assert preregistration["status"] == "PREREGISTERED_PRELAUNCH"
    exact_definition = preregistration["lane_contract"][
        "encoded_geometry_exact_definition"
    ]
    for field in (
        "source points",
        "cells",
        "centroids",
        "areas",
        "normals",
        "center",
        "reference length",
    ):
        assert field in exact_definition
    assert "actual EncodedBoundary" in exact_definition
    assert "single model(domain) forward" in exact_definition

    truth_table = preregistration["adjudication"]["truth_table"]
    assert [row["outcome"] for row in truth_table] == EXPECTED_OUTCOMES
    semantics = preregistration["adjudication"]["execution_exit_semantics"]
    assert "successful completed job" in semantics[EXPECTED_OUTCOMES[0]]
    assert "successful completed job" in semantics[EXPECTED_OUTCOMES[1]]
    assert "blocked nonzero wrapper exit" in semantics[EXPECTED_OUTCOMES[2]]
    assert "blocked nonzero wrapper exit" in semantics[EXPECTED_OUTCOMES[3]]
    assert preregistration["decision_order"][EXPECTED_OUTCOMES[0]] == (
        "licenses Stage B reconstructed historical replay only when produced "
        "by this v2 package"
    )
    assert preregistration["decision_order"]["stage_a_pass_does_not_license"] == [
        "a canonical-versus-legacy accuracy or noninferiority claim",
        "a matched training experiment",
        "a new architecture conclusion",
    ]
    assert not any(preregistration["launch_state"].values())


def test_v2_withdraws_v1_license_and_pins_checkpoint_epoch():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    withdrawal = preregistration["v1_license_withdrawal"]
    assert withdrawal["prior_job_id"] == 306623
    assert withdrawal["prior_categorical_outcome"] == EXPECTED_OUTCOMES[0]
    assert withdrawal["license_status"] == "WITHDRAWN_PENDING_FRESH_V2_PASS"
    assert "without an epoch" in withdrawal["provenance_limitation"]
    assert "newer MeshTransformer.0.N.mdlus" in withdrawal["provenance_limitation"]
    assert "do not license Stage B" in withdrawal["retained_observation"]

    checkpoint = preregistration["checkpoint_selection"]
    assert checkpoint["epoch"] == 491
    assert checkpoint["api_contract"].endswith("epoch=491)")
    assert checkpoint["exact_named_files"] == {
        "model": "MeshTransformer.0.491.mdlus",
        "training_state": "checkpoint.0.491.pt",
        "normalization_state": "norm_stats.pt",
    }
    assert "extra sibling files are permitted" in checkpoint["sibling_policy"]
    assert preregistration["frozen_provenance"]["checkpoint_load_epoch"] == 491

    source = _wrapper_text()
    assert "readonly EXPECTED_CHECKPOINT_LOAD_EPOCH=491" in source
    assert 'check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256" "$MODEL_CHECKPOINT"' in source
    assert 'check_sha "$EXPECTED_TRAINING_STATE_SHA256" "$TRAINING_STATE"' in source
    assert 'check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256"' in source
    # Earlier paired checkpoints and even an asymmetric newer model sibling
    # are legitimate directory contents. Epoch selection, not an exhaustive
    # directory inventory, is the deciding safety mechanism.
    assert 'find "$CHECKPOINT_DIR"' not in source


def test_wrapper_treats_valid_refutation_as_success_and_other_failures_as_blocked():
    source = _wrapper_text()
    assert 'case "$categorical_outcome" in' in source
    exact_case = source.split('case "$categorical_outcome" in', 1)[1]
    refutation_arm = exact_case.split(
        "VALID_STAGE_A_CURRENT_SOURCE_REFUTATION)",
        1,
    )[1].split("INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT)", 1)[0]
    assert "DONE_${SLURM_JOB_ID}" in refutation_arm
    assert "exit " not in refutation_arm

    invalid_arm = exact_case.split(
        "INVALID_STAGE_A_ARCHIVE_DOMAIN_INSTRUMENT)",
        1,
    )[1].split("INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY)", 1)[0]
    incomplete_arm = exact_case.split(
        "INCOMPLETE_STAGE_A_ARCHIVE_DOMAIN_CANARY)",
        1,
    )[1].split("*)", 1)[0]
    assert "BLOCKED_${SLURM_JOB_ID}" in invalid_arm
    assert "exit 30" in invalid_arm
    assert "BLOCKED_${SLURM_JOB_ID}" in incomplete_arm
    assert "exit 31" in incomplete_arm

    assert "set -o noclobber" in source
    assert ".stage-a-archive-canary-v2.lock" in source
    assert "STATUS_${SLURM_JOB_ID:-no_job}" in source
    assert "STAGE_A_CATEGORICAL_OUTCOME=" in source
