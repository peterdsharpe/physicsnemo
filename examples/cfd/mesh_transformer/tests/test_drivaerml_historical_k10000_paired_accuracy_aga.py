# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the frozen paired-accuracy AGA launch package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import drivaerml_historical_k10000_canonical_arm as canonical_producer
import drivaerml_historical_k10000_paired_accuracy_adjudicate as reducer
import drivaerml_historical_k10000_replay as legacy_producer
import drivaerml_historical_k10000_replay_adjudicate as stage_b_reducer
import drivaerml_historical_k10000_replay_runtime as runtime
import drivaerml_hqc_canonical_geometry_diagnostic_v5 as canonical_helper

STUDIES = Path(canonical_producer.__file__).resolve().parent
RESULTS = STUDIES.parent / "results"
WRAPPER = STUDIES / "drivaerml_historical_k10000_paired_accuracy_aga.sbatch"
PREREGISTRATION = (
    STUDIES / "phase1_historical_k10000_paired_accuracy_prereg_v1_2026-07-28.json"
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
INPUT_FREEZE = (
    RESULTS
    / "phase1_historical_k10000_input_freeze_v1_2026-07-28"
    / "historical_k10000_input_freeze_v1.json"
)
JOB306114_CORRECTION = (
    RESULTS
    / "phase1_hqc_canonical_geometry_checkpoint_binding_correction_v1_2026-07-28.json"
)
STAGE_B_SEAL = RESULTS / "historical_k10000_stage_b_replay_v2_job306814_2026-07-28"
STAGE_B_A_JSON = STAGE_B_SEAL / "artifacts" / "historical_k10000_replay_lane_A.json"
STAGE_B_A_NPZ = STAGE_B_SEAL / "artifacts" / "historical_k10000_replay_lane_A.npz"
STAGE_B_B_JSON = STAGE_B_SEAL / "artifacts" / "historical_k10000_replay_lane_B.json"
STAGE_B_B_NPZ = STAGE_B_SEAL / "artifacts" / "historical_k10000_replay_lane_B.npz"
STAGE_B_ADJUDICATION = (
    STAGE_B_SEAL / "artifacts" / "historical_k10000_replay_adjudication.json"
)
STAGE_B_TREE_MANIFEST = STAGE_B_SEAL / "sealed_remote_task_tree_manifest.sha256"
STAGE_B_RESULT = (
    RESULTS
    / "phase1_historical_k10000_stage_b_replay_result_v2_job306814_2026-07-28.json"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)
EXPECTED_REDUCER_SHA256 = (
    "dcc753bb1bd1b998a395e0723acfae32f8bb2a4c4114e3fedd2d0a76d604cf88"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "2d68b34b60678313d59106c89a7ad2abfe0e75a0533378cbb322611314d8e243"
)
EXPECTED_OUTCOMES = [
    reducer.NONINFERIORITY_SUCCESS_OUTCOME,
    reducer.VALID_REFUTATION,
    reducer.INVALID_COMPARISON,
    reducer.INCOMPLETE_COMPARISON,
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapper_text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _assert_canonical_sidecar(path: Path) -> None:
    expected = f"{_digest(path)}  {path.name}\n".encode("ascii")
    assert path.with_name(f"{path.name}.sha256").read_bytes() == expected


def _block(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


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
        "EXPECTED_CANONICAL_PRODUCER_SHA256": _digest(
            Path(canonical_producer.__file__).resolve()
        ),
        "EXPECTED_LEGACY_PRODUCER_SHA256": _digest(
            Path(legacy_producer.__file__).resolve()
        ),
        "EXPECTED_RUNTIME_HELPER_SHA256": _digest(Path(runtime.__file__).resolve()),
        "EXPECTED_CANONICAL_HELPER_SHA256": _digest(
            Path(canonical_helper.__file__).resolve()
        ),
        "EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256": _digest(
            Path(stage_b_reducer.__file__).resolve()
        ),
        "EXPECTED_PAIRED_REDUCER_SHA256": _digest(Path(reducer.__file__).resolve()),
        "EXPECTED_PREREGISTRATION_SHA256": _digest(PREREGISTRATION),
        "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256": (EXPECTED_SOURCE_MANIFEST_SHA256),
        "EXPECTED_EXECUTION_SOURCE_TREE_SHA256": EXPECTED_SOURCE_TREE_SHA256,
        "EXPECTED_GEOMETRY_MANIFEST_SHA256": _digest(GEOMETRY_MANIFEST),
        "EXPECTED_TARGET_MANIFEST_SHA256": _digest(TARGET_MANIFEST),
        "EXPECTED_INPUT_FREEZE_SHA256": _digest(INPUT_FREEZE),
        "EXPECTED_JOB306114_CORRECTION_SHA256": _digest(JOB306114_CORRECTION),
        "EXPECTED_STAGE_B_ADJUDICATION_SHA256": _digest(STAGE_B_ADJUDICATION),
        "EXPECTED_STAGE_B_RESULT_SHA256": _digest(STAGE_B_RESULT),
        "EXPECTED_STAGE_B_TREE_MANIFEST_SHA256": _digest(STAGE_B_TREE_MANIFEST),
        "EXPECTED_STAGE_B_A_JSON_SHA256": _digest(STAGE_B_A_JSON),
        "EXPECTED_STAGE_B_B_JSON_SHA256": _digest(STAGE_B_B_JSON),
        "EXPECTED_STAGE_B_NPZ_SHA256": _digest(STAGE_B_A_NPZ),
    }
    assert _digest(STAGE_B_A_NPZ) == _digest(STAGE_B_B_NPZ)
    assert expected["EXPECTED_PAIRED_REDUCER_SHA256"] == EXPECTED_REDUCER_SHA256
    assert (
        expected["EXPECTED_PREREGISTRATION_SHA256"] == EXPECTED_PREREGISTRATION_SHA256
    )
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source

    frozen = preregistration["frozen_provenance"]
    assert (
        frozen["canonical_producer_sha256"]
        == expected["EXPECTED_CANONICAL_PRODUCER_SHA256"]
    )
    assert (
        frozen["legacy_sentinel_producer_sha256"]
        == expected["EXPECTED_LEGACY_PRODUCER_SHA256"]
    )
    assert frozen["runtime_helper_sha256"] == expected["EXPECTED_RUNTIME_HELPER_SHA256"]
    assert (
        frozen["canonical_geometry_helper_sha256"]
        == expected["EXPECTED_CANONICAL_HELPER_SHA256"]
    )
    assert (
        frozen["stage_b_reducer_support_sha256"]
        == expected["EXPECTED_STAGE_B_REDUCER_SUPPORT_SHA256"]
    )
    assert frozen["paired_reducer_sha256"] == expected["EXPECTED_PAIRED_REDUCER_SHA256"]
    assert (
        frozen["target_free_geometry_manifest_sha256"]
        == expected["EXPECTED_GEOMETRY_MANIFEST_SHA256"]
    )
    assert (
        frozen["selected_target_manifest_sha256"]
        == expected["EXPECTED_TARGET_MANIFEST_SHA256"]
    )
    assert (
        frozen["historical_input_freeze_record_sha256"]
        == expected["EXPECTED_INPUT_FREEZE_SHA256"]
    )
    assert (
        frozen["job_306114_checkpoint_correction_sha256"]
        == expected["EXPECTED_JOB306114_CORRECTION_SHA256"]
    )
    assert (
        frozen["stage_b_adjudication_sha256"]
        == expected["EXPECTED_STAGE_B_ADJUDICATION_SHA256"]
    )
    assert (
        frozen["stage_b_result_record_sha256"]
        == expected["EXPECTED_STAGE_B_RESULT_SHA256"]
    )
    assert (
        frozen["stage_b_sealed_task_tree_manifest_sha256"]
        == expected["EXPECTED_STAGE_B_TREE_MANIFEST_SHA256"]
    )
    assert "wrapper_sha256" not in preregistration
    assert (
        "intentionally does not bind the wrapper"
        in preregistration["hash_cycle_policy"]
    )


def test_all_locally_frozen_inputs_have_canonical_sidecars():
    for path in (
        GEOMETRY_MANIFEST,
        TARGET_MANIFEST,
        INPUT_FREEZE,
        JOB306114_CORRECTION,
        STAGE_B_A_JSON,
        STAGE_B_A_NPZ,
        STAGE_B_B_JSON,
        STAGE_B_B_NPZ,
        STAGE_B_ADJUDICATION,
        STAGE_B_TREE_MANIFEST,
        STAGE_B_RESULT,
    ):
        _assert_canonical_sidecar(path)


def test_preregistration_fixes_exact_four_endpoint_claim_and_scope():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    assert preregistration["status"] == "PREREGISTERED_PRELAUNCH"
    endpoints = preregistration["deciding_endpoints"]
    assert endpoints["noninferiority_ratio"] == 1.02
    assert endpoints["baseline_means"] == reducer.FROZEN_BASELINE_MEANS
    assert endpoints["prospective_absolute_ceilings"] == reducer.FROZEN_CEILINGS
    assert endpoints["inclusive_comparison"] == (
        "canonical cohort mean <= frozen prospective absolute ceiling"
    )
    assert endpoints["all_four_required"] is True
    assert endpoints["casewise_deciding_cutoff"] is None
    assert preregistration["scientific_scope"]["casewise_metric_role"] == (
        "reported descriptively only; no casewise metric cutoff is deciding"
    )
    assert (
        "coherent source geometry plus coordinate-frame intervention"
        in (preregistration["scientific_scope"]["intervention"])
    )
    excluded = preregistration["scientific_scope"]["does_not_support"]
    for limitation in (
        "population-level statistical noninferiority",
        "OOD or population generalization",
        "superiority",
        "a single-tensor causal claim",
    ):
        assert limitation in excluded
    assert not any(preregistration["launch_state"].values())


def test_preregistration_truth_table_matches_reducer_and_sentinel_semantics():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    truth_table = preregistration["adjudication"]["truth_table"]
    assert [row["outcome"] for row in truth_table] == EXPECTED_OUTCOMES
    semantics = preregistration["adjudication"]["execution_exit_semantics"]
    assert "reducer exit 0" in semantics[reducer.NONINFERIORITY_SUCCESS_OUTCOME]
    assert "successful completed wrapper job" in semantics[reducer.VALID_REFUTATION]
    assert "reducer exit 2" in semantics[reducer.INVALID_COMPARISON]
    assert "reducer exit 3" in semantics[reducer.INCOMPLETE_COMPARISON]
    sentinel = preregistration["lane_contract"]["fresh_legacy_sentinel"]
    assert sentinel["producer"] == "unchanged frozen Stage-B v2 producer"
    assert sentinel["label"] == "A"
    assert sentinel["array_count"] == 720
    assert "instrument invalidity only" in sentinel["failure_semantics"]
    assert "never changes" in sentinel["ceiling_semantics"]


def test_job306114_predictions_are_explicitly_forbidden():
    preregistration = json.loads(PREREGISTRATION.read_bytes())
    correction = json.loads(JOB306114_CORRECTION.read_bytes())
    exclusion = preregistration["job_306114_exclusion"]
    assert exclusion == {
        "correction_record_sha256": _digest(JOB306114_CORRECTION),
        "correction_status": "CHECKPOINT_BINDING_LICENSE_WITHDRAWN",
        "forbidden_input": "every saved job-306114 prediction",
        "reason": (
            "the old validity producer omitted epoch=491 and did not attest "
            "the model checkpoint it opened"
        ),
        "retained_role": "target-free canonical API runtime validity only",
        "repair": (
            "fresh target-blind canonical A/B inference with explicit epoch=491 "
            "in this experiment"
        ),
    }
    assert correction["status"] == "CHECKPOINT_BINDING_LICENSE_WITHDRAWN"
    assert correction["affected_experiment"]["job_id"] == "306114"
    source = _wrapper_text()
    assert (
        "hqc_canonical_geometry_validity_36x5_job306114_2026-07-28/artifacts"
        not in source
    )
    assert "JOB306114_CORRECTION" in source


def test_wrapper_allocates_one_whole_node_and_three_payload_gpus():
    source = _wrapper_text()
    for directive in (
        "#SBATCH -J mt-k10k-paired-v1",
        "#SBATCH --time=00:45:00",
        "#SBATCH -N 1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --cpus-per-task=24",
        "#SBATCH --mem=256G",
        "#SBATCH --export=NIL",
    ):
        assert directive in source
    assert "EXPECTED_TASK_PHYSICAL=/scratch/fsw/" in source
    assert "2026-07-28-mt-historical-k10k-paired-accuracy-v1" in source
    assert "readonly UNIT_COUNT=4" in source
    assert "visible_gpu_count" in source


def test_canonical_lanes_are_target_blind_isolated_and_concurrent():
    source = _wrapper_text()
    canonical = _block(
        source,
        "# BEGIN CANONICAL_LANES",
        "# END CANONICAL_LANES",
    )
    assert canonical.count("env -i") == 2
    assert canonical.count("setsid timeout") == 2
    assert canonical.count("CUDA_VISIBLE_DEVICES=0") == 1
    assert canonical.count("CUDA_VISIBLE_DEVICES=1") == 1
    assert canonical.count("32m") == 2
    assert canonical.count("-m torch.distributed.run") == 2
    assert canonical.count("--standalone --nproc_per_node=1") == 2
    assert canonical.count("--lane-label A") == 1
    assert canonical.count("--lane-label B") == 1
    assert canonical.count('cd "$DATASET_PHYSICAL"') == 2
    for argument in (
        "--repo-root",
        "--dataset-root",
        "--dataset-config",
        "--resolved-config",
        "--checkpoint-dir",
        "--geometry-manifest",
    ):
        assert canonical.count(argument) == 2
    for forbidden in (
        "--target-input-manifest",
        "--historical-predictions",
        "--historical-metrics",
        "--sealed-stage-b",
        "STAGE_B_",
        "TARGET_INPUT_MANIFEST",
        "JOB306114",
        "CEILING",
        "ADJUDICATION",
    ):
        assert forbidden not in canonical
    assert source.index("# END LEGACY_SENTINEL_LANE") < source.index(
        "for lane_index in 0 1 2"
    )


def test_fresh_sentinel_is_unchanged_stage_b_lane_a_on_gpu2():
    source = _wrapper_text()
    sentinel = _block(
        source,
        "# BEGIN LEGACY_SENTINEL_LANE",
        "# END LEGACY_SENTINEL_LANE",
    )
    assert sentinel.count("env -i") == 1
    assert sentinel.count("setsid timeout") == 1
    assert sentinel.count("CUDA_VISIBLE_DEVICES=2") == 1
    assert sentinel.count("32m") == 1
    assert '"$LEGACY_PRODUCER"' in sentinel
    assert "--replay-label A" in sentinel
    assert "--target-input-manifest" in sentinel
    for argument in (
        "--repo-root",
        "--dataset-root",
        "--resolved-config",
        "--dataset-config",
        "--checkpoint-dir",
        "--geometry-manifest",
    ):
        assert argument in sentinel
    for forbidden in (
        "--sealed-stage-b",
        "STAGE_B_",
        "JOB306114",
        "CEILING",
        "ADJUDICATION",
    ):
        assert forbidden not in sentinel


def test_reducer_alone_receives_three_fresh_lanes_and_sealed_license():
    source = _wrapper_text()
    reducer_block = source.split('phase="REDUCTION"', 1)[1].split(
        'decision="$(',
        1,
    )[0]
    for argument in (
        "--canonical-producer",
        "--canonical-a-json",
        "--canonical-a-npz",
        "--canonical-b-json",
        "--canonical-b-npz",
        "--legacy-producer",
        "--fresh-legacy-sentinel-json",
        "--fresh-legacy-sentinel-npz",
        "--sealed-stage-b-a-json",
        "--sealed-stage-b-a-npz",
        "--sealed-stage-b-b-json",
        "--sealed-stage-b-b-npz",
        "--sealed-stage-b-adjudication",
        "--sealed-stage-b-result",
        "--sealed-stage-b-tree-manifest",
        "--output-json",
    ):
        assert argument in reducer_block
    assert "timeout --signal=TERM --kill-after=30s 8m" in reducer_block
    assert "0 | 2 | 3" in reducer_block
    assert source.index('if [[ "$lane_artifacts_valid" != true ]]') < source.index(
        'phase="REDUCTION"'
    )


def test_wrapper_preflight_verifies_source_checkpoint_manifests_and_full_seal():
    source = _wrapper_text()
    for required in (
        "execution-source manifest content verification failed",
        "execution source tree changed",
        'check_sha "$EXPECTED_MODEL_CHECKPOINT_SHA256"',
        'check_sha "$EXPECTED_TRAINING_STATE_SHA256"',
        'check_sha "$EXPECTED_NORMALIZATION_STATE_SHA256"',
        'check_sha "$EXPECTED_GEOMETRY_MANIFEST_SHA256"',
        'check_sha "$EXPECTED_TARGET_MANIFEST_SHA256"',
        'check_sha "$EXPECTED_INPUT_FREEZE_SHA256"',
        'check_sha "$EXPECTED_JOB306114_CORRECTION_SHA256"',
        'check_sha "$EXPECTED_STAGE_B_ADJUDICATION_SHA256"',
        'check_sha "$EXPECTED_STAGE_B_RESULT_SHA256"',
        'check_sha "$EXPECTED_STAGE_B_TREE_MANIFEST_SHA256"',
        "EXPECTED_STAGE_B_REMOTE_FILE_COUNT=957",
        "EXPECTED_STAGE_B_REMOTE_SIZE_BYTES=159205162",
        "EXPECTED_STAGE_B_LICENSE_FILE_COUNT=961",
        "EXPECTED_STAGE_B_LICENSE_SIZE_BYTES=159343806",
        "Stage-B tree manifest differs from the exact staged sealed inventory",
        "Stage-B sealed tree all-file hash verification failed",
        "Stage-B result record does not license this comparison",
    ):
        assert required in source
    assert "MeshTransformer.0.491.mdlus" in source
    assert "checkpoint.0.491.pt" in source
    assert "CHECKPOINT_SELECTION epoch=491" in source


def test_wrapper_enforces_no_clobber_heartbeat_and_categorical_exit_semantics():
    source = _wrapper_text()
    assert "set -o noclobber" in source
    assert ".paired-accuracy-v1.lock" in source
    assert "flock -n 9" in source
    assert "verify_canonical_sidecar" in source
    assert "GPU_HEARTBEAT" in source
    assert "STATUS_${SLURM_JOB_ID:-no_job}" in source
    assert "PAIRED_ACCURACY_CATEGORICAL_OUTCOME=" in source
    assert "DONE_${SLURM_JOB_ID}" in source
    assert "BLOCKED_${SLURM_JOB_ID}" in source

    outcome_switch = source.split('case "$categorical_outcome" in', 1)[1]
    pass_arm = outcome_switch.split("CANONICAL_NONINFERIORITY_PASS)", 1)[1].split(
        "VALID_CANONICAL_NONINFERIORITY_REFUTATION)",
        1,
    )[0]
    refutation_arm = outcome_switch.split(
        "VALID_CANONICAL_NONINFERIORITY_REFUTATION)",
        1,
    )[1].split("INVALID_CANONICAL_COMPARISON)", 1)[0]
    invalid_arm = outcome_switch.split("INVALID_CANONICAL_COMPARISON)", 1)[1].split(
        "INCOMPLETE_CANONICAL_COMPARISON)",
        1,
    )[0]
    incomplete_arm = outcome_switch.split(
        "INCOMPLETE_CANONICAL_COMPARISON)",
        1,
    )[1].split("*)", 1)[0]
    assert "DONE_${SLURM_JOB_ID}" in pass_arm
    assert "exit " not in pass_arm
    assert "DONE_${SLURM_JOB_ID}" in refutation_arm
    assert "exit " not in refutation_arm
    assert "BLOCKED_${SLURM_JOB_ID}" in invalid_arm
    assert "exit 30" in invalid_arm
    assert "BLOCKED_${SLURM_JOB_ID}" in incomplete_arm
    assert "exit 31" in incomplete_arm
