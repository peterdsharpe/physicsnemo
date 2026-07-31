# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the blind nested-target freezer's AGA commit wrapper."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
MESH_TRANSFORMER = TESTS.parent
STUDIES = MESH_TRANSFORMER / "studies"
RESULTS = MESH_TRANSFORMER / "results"

WRAPPER = STUDIES / "drivaerml_hqc_nested_target_input_freeze_aga.sbatch"
PRODUCER = STUDIES / "drivaerml_hqc_nested_target_input_freeze.py"
PREREG = STUDIES / "phase1_hqc_canonical_fixed_prefix_prereg_v1_2026-07-29.json"
GEOMETRY = (
    RESULTS
    / "hqc_canonical_geometry_validity_36x5_job306114_2026-07-28"
    / "drivaerml_geometry_input_manifest_36cases_v1.json"
)
HISTORICAL_TARGET = (
    RESULTS
    / "historical_k10000_selected_target_input_freeze_job306302_2026-07-28"
    / "artifacts"
    / "historical_k10000_selected_target_input_manifest_v1.json"
)

NAMESPACE = "2026-07-29-mt-hqc-nested-target-freeze-v1"
CASE_IDS = (
    "run_118",
    "run_129",
    "run_145",
    "run_149",
    "run_17",
    "run_171",
    "run_18",
    "run_183",
    "run_197",
    "run_202",
    "run_225",
    "run_270",
    "run_271",
    "run_298",
    "run_305",
    "run_320",
    "run_367",
    "run_380",
    "run_382",
    "run_399",
    "run_4",
    "run_409",
    "run_419",
    "run_424",
    "run_429",
    "run_431",
    "run_439",
    "run_465",
    "run_468",
    "run_469",
    "run_478",
    "run_489",
    "run_490",
    "run_495",
    "run_71",
    "run_86",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _inline_pythons() -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY\n", _source(), flags=re.DOTALL)


def _sidecar(path: Path) -> None:
    digest = _digest(path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def _bundle_document(npz: Path, npz_sha256: str, producer_sha256: str) -> dict:
    digest = "a" * 64
    cases = []
    manifest = {}
    for ordinal, case_id in enumerate(CASE_IDS):
        prefix = f"case_{ordinal:02d}_{case_id}__"
        cases.append(
            {
                "cohort_ordinal": ordinal,
                "case_id": case_id,
                "selection": {
                    "selected_cell_ids_sha256_by_resolution": {
                        str(k): digest for k in (2500, 5000, 10000, 20000, 40000)
                    }
                },
                "targets": {
                    field: {
                        "historical_k10000_prefix_authenticated": True,
                        "prefix_sha256_by_resolution": {
                            str(k): digest for k in (2500, 5000, 10000, 20000, 40000)
                        },
                    }
                    for field in ("pressure", "wss")
                },
            }
        )
        for suffix, shape, dtype, nbytes in (
            ("selected_cell_ids_int64", [40000], "<i8", 320000),
            ("physical_globals_float32", [7], "<f4", 28),
            ("raw_target_pressure_float32", [40000], "<f4", 160000),
            ("raw_target_wss_float32", [40000, 3], "<f4", 480000),
        ):
            manifest[f"{prefix}{suffix}"] = {
                "shape": shape,
                "dtype": dtype,
                "nbytes": nbytes,
                "sha256": digest,
            }
    return {
        "schema_version": 1,
        "artifact_kind": "drivaerml_hqc_nested_raw_target_bundle",
        "status": "PASSED_HQC_NESTED_RAW_TARGET_FREEZE",
        "case_count": 36,
        "resolutions": [2500, 5000, 10000, 20000, 40000],
        "max_resolution": 40000,
        "fixed_query_resolution": 2500,
        "cohort_sha256": (
            "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d"
        ),
        "dataset_manifest_sha256": (
            "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca"
        ),
        "geometry_manifest": {
            "sha256": (
                "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0"
            )
        },
        "historical_k10000_target_manifest": {
            "sha256": (
                "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49"
            ),
            "prefix_hashes_authenticated": 72,
        },
        "npz": {
            "path": str(npz),
            "sha256": npz_sha256,
            "array_count": 144,
        },
        "provenance": {"script_sha256": producer_sha256},
        "publication_contract": {
            "json_manifest_linked_last": True,
            "producer_outputs_are_not_a_commit_marker": True,
            "valid_only_after_external_sidecar_checks_and_done_marker": True,
            "interrupted_partial_bundle_must_not_be_overwritten": True,
        },
        "read_exclusions": {
            "model_opened": False,
            "prediction_opened": False,
            "metric_opened": False,
            "decision_threshold_opened": False,
            "other_cell_data_opened": False,
            "point_data_opened": False,
            "interior_opened": False,
        },
        "cases": cases,
        "array_manifest": manifest,
    }


def _run_commit(
    tmp_path: Path,
    *,
    corrupt_sidecar: str | None = None,
    preexisting_done: bool = False,
    done_name: str = "DONE_123.json",
    mutate_prereg_before_publish: bool = False,
    boolean_schema_version: bool = False,
    float_array_count: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    activation = tmp_path / "activation.json"
    activation.write_bytes(b'{"decision":"pass"}\n')
    _sidecar(activation)
    npz = tmp_path / "bundle.npz"
    npz.write_bytes(b"synthetic-npz")
    _sidecar(npz)
    document = _bundle_document(npz, _digest(npz), _digest(PRODUCER))
    if boolean_schema_version:
        document["schema_version"] = True
    if float_array_count:
        document["npz"]["array_count"] = 144.0
    output = tmp_path / "bundle.json"
    output.write_text(
        json.dumps(document, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _sidecar(output)
    if corrupt_sidecar == "json":
        output.with_name(f"{output.name}.sha256").write_text(
            f"{'0' * 64}  {output.name}\n",
            encoding="ascii",
        )
    elif corrupt_sidecar == "npz":
        npz.with_name(f"{npz.name}.sha256").write_text(
            f"{_digest(npz)}  wrong-name.npz\n",
            encoding="ascii",
        )
    done = tmp_path / done_name
    if preexisting_done:
        done.write_text("sentinel\n", encoding="utf-8")
    prereg = tmp_path / PREREG.name
    prereg.write_bytes(PREREG.read_bytes())
    _sidecar(prereg)
    launch_manifest = tmp_path / "launch.json"
    launch_manifest.write_text('{"frozen":true}\n', encoding="ascii")
    _sidecar(launch_manifest)
    commit = _inline_pythons()[1]
    if mutate_prereg_before_publish:
        marker = "    _, final_json_sha256 = canonical_sidecar(json_path)"
        assert marker in commit
        commit = commit.replace(
            marker,
            (
                '    raced = b"raced-preregistration\\n"\n'
                "    prereg_path.write_bytes(raced)\n"
                "    prereg_path.with_name(prereg_path.name + "
                '".sha256").write_text(\n'
                "        hashlib.sha256(raced).hexdigest() + "
                'f"  {prereg_path.name}\\n",\n'
                '        encoding="ascii",\n'
                "    )\n"
                f"{marker}"
            ),
            1,
        )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            commit,
            str(output),
            str(npz),
            str(done),
            "123",
            _digest(PRODUCER),
            _digest(PREREG),
            "ec947a48495b1ddcaa9ec81e96ad299a4f34e438940d57fe5f053db47aecdf9d",
            "51c2268df5b9b365f4ef6147c6ec390f10c55f733ad967f6617bd5e52f62e7ca",
            "3d33209f775513a690d61be560e640a348268132e14dd56675d256ee380bf4b0",
            "d7502e9539b983de07ccb58a6313ab844aa5ea5ef4e3e165dd49c6bbfa1a2e49",
            str(activation),
            _digest(activation),
            str(prereg),
            str(launch_manifest),
            _digest(launch_manifest),
            str(WRAPPER),
            _digest(WRAPPER),
            NAMESPACE,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, output, npz, done


def test_wrapper_has_valid_bash_and_embedded_python_syntax() -> None:
    subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(WRAPPER)],
        check=True,
        capture_output=True,
        text=True,
    )
    blocks = _inline_pythons()
    assert len(blocks) == 2
    for index, block in enumerate(blocks):
        compile(block, f"{WRAPPER.name}:inline-{index}", "exec")


def test_wrapper_is_a_small_cpu_only_blind_job() -> None:
    source = _source()
    for directive in (
        "#SBATCH --time=00:15:00",
        "#SBATCH -p cpu",
        "#SBATCH -q cpu-short",
        "#SBATCH -N 1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=8G",
        "#SBATCH --export=NIL",
    ):
        assert directive in source
    assert "#SBATCH --gpus" not in source
    assert "nvidia-smi" not in source
    assert "timeout --signal=TERM --kill-after=30s 12m" in source
    assert "env -i" in source


def test_wrapper_binds_the_current_frozen_inputs() -> None:
    source = _source()
    expected = {
        "EXPECTED_PRODUCER_SHA256": _digest(PRODUCER),
        "EXPECTED_PREREG_SHA256": _digest(PREREG),
        "EXPECTED_WRAPPER_TEST_SHA256": _digest(Path(__file__).resolve()),
        "EXPECTED_GEOMETRY_SHA256": _digest(GEOMETRY),
        "EXPECTED_HISTORICAL_TARGET_SHA256": _digest(HISTORICAL_TARGET),
    }
    for name, digest in expected.items():
        assert f"readonly {name}={digest}" in source
    assert NAMESPACE in source
    assert source.count(NAMESPACE) == 4
    assert "cmp -s" in source


def test_wrapper_requires_exact_activation_before_reading_targets() -> None:
    source = _source()
    activation = _inline_pythons()[0]
    assert source.index('phase="ACTIVATION_VALIDATION"') < source.index(
        'phase="TARGET_FREEZE"'
    )
    for value in (
        "drivaerml_historical_k10000_one_step_parity_adjudication",
        "VALID_HISTORICAL_K10000_ONE_STEP_PARITY_ADJUDICATION",
        "NEGLIGIBLE_OPTIMIZATION_EFFECT_PASS",
        "producer_source_sha256",
        "drivaerml_hqc_nested_target_freeze_launch_manifest",
        "FROZEN_HQC_NESTED_TARGET_FREEZE_LAUNCH",
        "launch manifest bindings differ",
    ):
        assert value in activation
    assert "activation sidecar is not canonical" in activation


def test_unfinalized_wrapper_rejects_every_activation(tmp_path: Path) -> None:
    source = _source()
    assert ("readonly EXPECTED_ACTIVATION_SHA256=" + "0" * 64) in source
    activation = tmp_path / "activation.json"
    activation.write_text("{}\n", encoding="utf-8")
    _sidecar(activation)
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            _inline_pythons()[0],
            str(activation),
            "f2458d95573b188f8523602204219df98c875c6cd4b2a4e9d306a594d4542500",
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "has not been frozen" in result.stderr


def test_wrapper_stages_only_the_blind_inputs_and_outputs() -> None:
    source = _source()
    inventory = source.split('expected_root_files="$(', 1)[1].split(
        'actual_root_files="$(', 1
    )[0]
    for filename in (
        WRAPPER.name,
        PRODUCER.name,
        PREREG.name,
        "drivaerml_geometry_input_manifest_36cases_v1.json",
        "historical_k10000_selected_target_input_manifest_v1.json",
        "hqc_nested_target_freeze_launch_manifest_v1.json",
        "one_step_parity_adjudication.json",
    ):
        assert filename in inventory
    for forbidden in (
        ".npz",
        "checkpoint",
        "prediction",
        "metric",
        "adjudicate.py",
    ):
        assert forbidden not in inventory
    assert "unexpected_root_entry" in source
    assert "fresh attempt artifacts directory is not empty" in source
    assert 'expected_log="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"' in source
    assert "attempt log inventory differs from the current Slurm job" in source


def test_commit_helper_publishes_a_content_bearing_done_last(tmp_path: Path) -> None:
    result, output, npz, done = _run_commit(tmp_path)
    assert result.returncode == 0, result.stderr
    marker = json.loads(done.read_text(encoding="ascii"))
    assert marker == {
        "artifact_kind": "drivaerml_hqc_nested_target_bundle_commit",
        "activation_adjudication_sha256": _digest(tmp_path / "activation.json"),
        "attempt_id": NAMESPACE,
        "job_id": "123",
        "json_sha256": _digest(output),
        "launch_manifest_sha256": _digest(tmp_path / "launch.json"),
        "npz_sha256": _digest(npz),
        "preregistration_sha256": _digest(PREREG),
        "producer_sha256": _digest(PRODUCER),
        "reducer_schema_validation_performed": False,
        "schema_version": 1,
        "status": "CONTENT_COMMITTED_UNVALIDATED_HQC_NESTED_TARGET_BUNDLE",
        "wrapper_sha256": _digest(WRAPPER),
    }
    source = _source()
    assert source.index('"$PYTHON" - \\\n  "$OUTPUT_JSON"') < source.index(
        "completed_units=1"
    )
    assert "os.link(temporary, done_path, follow_symlinks=False)" in source
    assert "os.fsync(directory_descriptor)" in source


@pytest.mark.parametrize("corrupt_sidecar", ["json", "npz"])
def test_commit_helper_rejects_noncanonical_sidecar_without_cleanup(
    tmp_path: Path,
    corrupt_sidecar: str,
) -> None:
    result, output, npz, done = _run_commit(
        tmp_path,
        corrupt_sidecar=corrupt_sidecar,
    )
    assert result.returncode != 0
    assert not done.exists()
    assert output.exists()
    assert npz.exists()
    assert output.with_name(f"{output.name}.sha256").exists()
    assert npz.with_name(f"{npz.name}.sha256").exists()


def test_commit_helper_refuses_to_replace_done_or_artifacts(tmp_path: Path) -> None:
    result, output, npz, done = _run_commit(tmp_path, preexisting_done=True)
    assert result.returncode != 0
    assert done.read_text(encoding="utf-8") == "sentinel\n"
    assert output.exists()
    assert npz.exists()


def test_commit_helper_rejects_preregistration_race_before_done(
    tmp_path: Path,
) -> None:
    result, output, npz, done = _run_commit(
        tmp_path,
        mutate_prereg_before_publish=True,
    )
    assert result.returncode != 0
    assert "changed before DONE publication" in result.stderr
    assert not done.exists()
    assert output.exists()
    assert npz.exists()


def test_commit_helper_rejects_boolean_schema_version(tmp_path: Path) -> None:
    result, output, npz, done = _run_commit(
        tmp_path,
        boolean_schema_version=True,
    )
    assert result.returncode != 0
    assert "bundle schema_version differs" in result.stderr
    assert not done.exists()
    assert output.exists()
    assert npz.exists()


def test_commit_helper_rejects_float_array_count(tmp_path: Path) -> None:
    result, output, npz, done = _run_commit(
        tmp_path,
        float_array_count=True,
    )
    assert result.returncode != 0
    assert "bundle internal NPZ identity differs" in result.stderr
    assert not done.exists()
    assert output.exists()
    assert npz.exists()


def test_commit_helper_requires_done_filename_to_bind_numeric_job_id(
    tmp_path: Path,
) -> None:
    result, output, npz, done = _run_commit(
        tmp_path,
        done_name="DONE_wrong.json",
    )
    assert result.returncode != 0
    assert "DONE filename is malformed" in result.stderr
    assert not done.exists()
    assert output.exists()
    assert npz.exists()


def test_wrapper_never_loads_targets_or_deletes_partial_bundle() -> None:
    source = _source()
    commit = _inline_pythons()[1]
    for forbidden in (
        "np.load",
        "torch",
        "model(",
        "checkpoint_path",
        "scancel",
        "sbatch",
        "requeue",
        'rm "$OUTPUT',
        "output.unlink",
        "npz_path.unlink",
        "json_path.unlink",
    ):
        assert forbidden not in commit
    assert "temporary.unlink(missing_ok=True)" in commit
    assert (
        "OUTPUT_JSON"
        not in source.split("finalize()", 1)[1].split("trap finalize EXIT", 1)[0]
    )
