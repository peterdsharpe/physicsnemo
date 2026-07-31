# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the validity-only epoch-491 checkpoint smoke."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

TESTS = Path(__file__).resolve().parent
MESH_TRANSFORMER = TESTS.parent
STUDIES = MESH_TRANSFORMER / "studies"
RESULTS = MESH_TRANSFORMER / "results"

WRAPPER = STUDIES / "drivaerml_historical_k10000_one_step_checkpoint_smoke_aga.sbatch"
PRODUCER = STUDIES / "drivaerml_historical_k10000_one_step_parity.py"
LEGACY_HELPER = STUDIES / "drivaerml_historical_k10000_replay.py"
RUNTIME_HELPER = STUDIES / "drivaerml_historical_k10000_replay_runtime.py"
CANONICAL_HELPER = STUDIES / "drivaerml_hqc_canonical_geometry_diagnostic_v5.py"
EXECUTION_SOURCE_MANIFEST = (
    RESULTS
    / "historical_k10000_stage_b_replay_v2_job306814_2026-07-28"
    / "execution_source_manifest.sha256"
)

NAMESPACE = "2026-07-29-mt-historical-k10k-one-step-checkpoint-smoke-v1"
EXPECTED_PRODUCER_SHA256 = (
    "f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e8edb80f34c005f85e1e87ebc27567ddf0a74fe2bc34e06ba40c35c92e54cfb4"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "fe6bbcf3c28154c7c028456b4b067aec3818effb72c73082612200e482c2c67e"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _inline_python(source: str) -> str:
    return source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def test_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(WRAPPER)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_inline_checkpoint_probe_compiles() -> None:
    source = _source()
    assert source.count("<<'PY'\n") == 1
    compile(_inline_python(source), f"{WRAPPER.name}:checkpoint-probe", "exec")


def test_wrapper_requests_one_whole_aga_node_but_exposes_only_gpu_zero() -> None:
    source = _source()
    for directive in (
        "#SBATCH -J mt-k10k-step-smoke-v1",
        "#SBATCH --time=00:10:00",
        "#SBATCH -p batch",
        "#SBATCH -q short",
        "#SBATCH -N 1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --export=NIL",
    ):
        assert directive in source
    assert source.count("CUDA_VISIBLE_DEVICES=0") == 1
    assert "CUDA_VISIBLE_DEVICES=1" not in source
    assert "WORLD_SIZE=1" in source
    assert "LOCAL_WORLD_SIZE=1" in source
    assert "setsid timeout --signal=TERM --kill-after=30s 7m" in source


def test_wrapper_uses_the_dedicated_immutable_namespace() -> None:
    source = _source()
    assert NAMESPACE in source
    assert source.count(NAMESPACE) == 4
    assert "one-step-parity-v3" not in source
    assert "one-step-parity-v2" not in source
    assert "/agents/2026-07-29-" in source


def test_wrapper_hash_binds_every_code_and_source_input() -> None:
    source = _source()
    expected = {
        "EXPECTED_PRODUCER_SHA256": _digest(PRODUCER),
        "EXPECTED_LEGACY_HELPER_SHA256": _digest(LEGACY_HELPER),
        "EXPECTED_RUNTIME_HELPER_SHA256": _digest(RUNTIME_HELPER),
        "EXPECTED_CANONICAL_HELPER_SHA256": _digest(CANONICAL_HELPER),
        "EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256": _digest(EXECUTION_SOURCE_MANIFEST),
    }
    assert expected["EXPECTED_PRODUCER_SHA256"] == EXPECTED_PRODUCER_SHA256
    assert (
        expected["EXPECTED_EXECUTION_SOURCE_MANIFEST_SHA256"]
        == EXPECTED_SOURCE_MANIFEST_SHA256
    )
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source
    assert (
        f"readonly EXPECTED_EXECUTION_SOURCE_TREE_SHA256={EXPECTED_SOURCE_TREE_SHA256}"
        in source
    )
    assert "readonly EXPECTED_EXECUTION_SOURCE_FILE_COUNT=919" in source
    assert "readonly EXPECTED_EXECUTION_SOURCE_SIZE_BYTES=10090497" in source


def test_wrapper_hash_binds_exact_config_and_epoch_491_state() -> None:
    source = _source()
    expected = {
        "EXPECTED_DATASET_CONFIG_SHA256": (
            "a86a23fb5ae87a400f6b326c597c1a1358429c020628197bd77d2465f1fabed3"
        ),
        "EXPECTED_RESOLVED_CONFIG_SHA256": (
            "a71987df4d49d38cc7f6b43c08ba0a0592fd39cf16a50aef04bf1b0d4f080fe1"
        ),
        "EXPECTED_MODEL_CHECKPOINT_SHA256": (
            "4c76b1130ffacf93d3590056734e3d8881cc7b12da4f22911f69aa4e612e7a88"
        ),
        "EXPECTED_TRAINING_STATE_SHA256": (
            "3783bda98ed561db95638d1c6fbb914b73be1bf36ed91ad79872f7f19763cea7"
        ),
        "EXPECTED_NORMALIZATION_STATE_SHA256": (
            "31a73b08f3e3f6b2d8c60ed659247deae996d2596e752f5423cabbb29f186b94"
        ),
    }
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source
        assert f'check_sha "${name}"' in source
    assert "readonly EXPECTED_CHECKPOINT_EPOCH=491" in source
    assert "MeshTransformer.0.491.mdlus" in source
    assert "checkpoint.0.491.pt" in source
    assert "norm_stats.pt" in source


def test_minimal_staged_root_inventory_is_exact() -> None:
    source = _source()
    expected_files = {
        WRAPPER.name,
        PRODUCER.name,
        LEGACY_HELPER.name,
        RUNTIME_HELPER.name,
        CANONICAL_HELPER.name,
        "execution_source_manifest.sha256",
    }
    file_block = source.split('expected_root_files="$(', 1)[1].split(
        'actual_root_files="$(', 1
    )[0]
    for filename in expected_files:
        assert filename in file_block
    assert "adjudicate" not in file_block
    assert "prereg" not in file_block
    assert "manifest_v1.json" not in file_block
    directory_block = source.split('expected_root_directories="$(', 1)[1].split(
        'actual_root_directories="$(', 1
    )[0]
    assert "execution_source payload_logs sbatch_logs" in directory_block
    assert "artifacts" not in directory_block
    assert "minimal smoke package" in source
    assert "task root contains a symlink or special entry" in source


def test_execution_source_is_recomputed_and_tree_bound() -> None:
    source = _source()
    assert "recomputed_execution_source_manifest" in source
    assert 'sha256sum --quiet --strict --check "$EXECUTION_SOURCE_MANIFEST"' in source
    assert "execution-source manifest bytes differ from the staged inventory" in source
    assert "actual_source_tree" in source
    assert (
        'if [[ "$actual_source_tree" != "$EXPECTED_EXECUTION_SOURCE_TREE_SHA256"'
        in source
    )
    assert "execution source contains an unpinned Python cache path" in source


def test_probe_loads_verified_support_and_exact_checkpoint_optimizer() -> None:
    probe = _inline_python(_source())
    assert "producer._load_support_modules(producer_path)" in probe
    assert "producer._new_model_optimizer(" in probe
    assert 'regime="checkpoint_epoch491"' in probe
    assert "loaded_epoch != expected_epoch" in probe
    assert 'members != ("Muon", "AdamW")' in probe
    assert "any(entries <= 0 for entries in state_entries)" in probe
    assert "producer._learning_rate(optimizer)" in probe
    assert "math.isfinite(learning_rate)" in probe
    assert "producer._stable_sha256(optimizer.state_dict())" in probe
    assert r're.fullmatch(r"[0-9a-f]{64}", optimizer_hash)' in probe


def test_probe_pins_parameter_layout_and_never_touches_gradients() -> None:
    probe = _inline_python(_source())
    assert "readonly EXPECTED_PARAMETER_COUNT=1278268" in _source()
    assert "producer._parameter_layout(model)" in probe
    assert 'layout["parameter_count"] != expected_parameter_count' in probe
    for forbidden in (
        ".grad",
        "loss.backward",
        "optimizer.step",
        "_flatten_gradients",
        "_flatten_parameters",
        "_run_arm",
        "_prepare_case",
    ):
        assert forbidden not in probe


def test_probe_reads_no_dataset_case_or_scientific_inputs() -> None:
    source = _source()
    probe = _inline_python(source)
    for forbidden in (
        "DATASET_PHYSICAL",
        "DATASET_MANIFEST",
        "GEOMETRY_MANIFEST",
        "TARGET_INPUT_MANIFEST",
        "_load_runtime(",
        "_load_explicit_raw_subset",
        "LossCalculator",
        "model.encode",
        "model.decode",
        "model(",
        ".json",
        ".npz",
        "case",
        "target",
        "prediction",
        "loss",
        "gradient",
        "update",
    ):
        assert forbidden not in probe
    assert "--dataset-root" not in source
    assert "--geometry-manifest" not in source
    assert "--target-input-manifest" not in source


def test_outputs_are_limited_to_log_done_and_fail_closed_status() -> None:
    source = _source()
    assert 'PAYLOAD_LOG="$PAYLOAD_LOG_DIR/checkpoint_smoke.log"' in source
    assert 'DONE_MARKER="$TASK_DIR/DONE_${SLURM_JOB_ID}"' in source
    assert 'readonly STATUS_MARKER="$TASK_DIR/STATUS_${SLURM_JOB_ID}"' in source
    assert '"$TASK_DIR/BLOCKED_' not in source
    assert '"$TASK_DIR/INVALID_' not in source
    assert "PRODUCER_JSON" not in source
    assert "PRODUCER_NPZ" not in source
    assert "ADJUDICATION" not in source
    assert "set -o noclobber" in source
    assert "check_absent" in source
    assert "trap finalize EXIT" in source


def test_smoke_is_explicitly_validity_only_and_never_scientific() -> None:
    source = _source()
    assert 'categorical_scientific_outcome="NONE"' in source
    assert "CHECKPOINT_SMOKE_VALIDITY_ONLY=true" in source
    assert "CHECKPOINT_SMOKE_CATEGORICAL_SCIENTIFIC_OUTCOME=NONE" in source
    assert "VALIDITY_ONLY categorical_scientific_outcome=NONE" in source
    assert "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS" not in source
    assert "MATERIAL_PARITY_DIFFERENCE_FAIL" not in source
    assert "decision_outcome" not in source


def test_success_requires_one_auditable_pass_line_before_done() -> None:
    source = _source()
    assert '"CHECKPOINT_SMOKE_PASS "' in _inline_python(source)
    assert "grep -c '^CHECKPOINT_SMOKE_PASS '" in source
    assert "completed_units=1" in source
    assert 'phase="COMPLETED_CHECKPOINT_VALIDITY_SMOKE"' in source
    assert 'run_status="PASSED_CHECKPOINT_VALIDITY_SMOKE"' in source
    assert "CHECKPOINT_SMOKE_DONE validity_only=true" in source
