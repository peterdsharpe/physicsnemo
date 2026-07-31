#!/usr/bin/env python3
"""Build a read-only, content-addressed manifest of named AGA artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe"
)
GROUP_ROOT = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae")
AGENTS_ROOT = USER_ROOT / "agents"

TASK_DIRS = [
    AGENTS_ROOT / "2026-07-24-mt-transfer-v2-gate0",
    AGENTS_ROOT / "2026-07-25-mt-coverage-sweep",
    AGENTS_ROOT / "2026-07-25-mt-measnorm",
    AGENTS_ROOT / "2026-07-25-mt-homog",
    AGENTS_ROOT / "2026-07-25-mt-wave-harvest",
]

SOURCE_TREES = {
    "base_remote_repo_snapshot": USER_ROOT / "physicsnemo-mesh-transformer",
    "original_measnorm_isolated_tree": (
        AGENTS_ROOT / "2026-07-25-mt-measnorm" / "repo"
    ),
    "final_hboth_measnorm_isolated_tree": (
        AGENTS_ROOT / "2026-07-25-mt-homog" / "repo"
    ),
}

FINAL_TREE_LABEL = "final_hboth_measnorm_isolated_tree"
ORIGINAL_MEASNORM_TREE_LABEL = "original_measnorm_isolated_tree"

RUN_TARGETS = [
    *[
        (
            FINAL_TREE_LABEL,
            f"t2_mesh_transformer_surface_flagship_homogmn_homog_seed{seed}_lr3.0e-3",
        )
        for seed in range(42, 47)
    ],
    *[
        (
            FINAL_TREE_LABEL,
            f"t2_mesh_transformer_surface_flagship_measnorm_homog_seed{seed}_lr3.0e-3",
        )
        for seed in range(42, 47)
    ],
    (
        ORIGINAL_MEASNORM_TREE_LABEL,
        "t2_mesh_transformer_surface_flagship_measnorm_intrinsic_seed42_lr3.0e-3",
    ),
    (
        ORIGINAL_MEASNORM_TREE_LABEL,
        "t2_mesh_transformer_surface_flagship_measnorm_mn2_seed42_lr3.0e-3",
    ),
]

DATASETS = [
    {
        "label": "training_drivaerml",
        "path": GROUP_ROOT / "datasets" / "PhysicsNeMo-DrivaerML",
        "metadata_names": [
            "manifest.json",
            "units.json",
            ".bc_surgery_checkpoint.jsonl",
        ],
    },
    {
        "label": "id_reference_shadow",
        "path": USER_ROOT / "mt_datasets" / "drivaerml_ood_shadow",
        "metadata_names": ["manifest.json"],
    },
    {
        "label": "shift_suv_pilot",
        "path": USER_ROOT / "mt_datasets" / "shift_suv_pilot",
        "metadata_names": ["manifest.json"],
    },
]

TASK_ROOT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".sbatch",
    ".yaml",
    ".yml",
}
TASK_MARKER_PREFIXES = ("DONE", "RC_", "STATUS_")
TASK_MARKER_NAMES = {"SWEEP_DONE", "VALIDATED"}
TASK_EXACT_NESTED_NAMES = {"manifest.json", "metrics.jsonl", "resolved_config.yaml"}
SOURCE_RELATIVE_ROOTS = [
    Path("physicsnemo/experimental/nn/mesh_attention"),
    Path("examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf"),
    Path("examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets"),
    Path("examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"),
]
SOURCE_EXPLICIT_FILES = [
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path(
        "examples/cfd/external_aerodynamics/unified_external_aero_recipe/"
        "requirements.txt"
    ),
]
CHECKPOINT_RE = re.compile(
    r"^(?P<family>checkpoint|MeshTransformer)\.\d+\.(?P<epoch>\d+)"
    r"\.(?:pt|mdlus)$"
)

HASH_CHUNK_BYTES = 8 * 1024 * 1024
HASH_WORKERS = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    """Hash one stable file or symlink without modifying it."""
    before = path.lstat()
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": path.relative_to(relative_to).as_posix(),
        "mode": oct(stat.S_IMODE(before.st_mode)),
        "mtime_ns": before.st_mtime_ns,
    }
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(path)
        record.update(
            {
                "type": "symlink",
                "size_bytes": before.st_size,
                "link_target": target,
                "sha256": sha256_bytes(target.encode("utf-8")),
                "sha256_semantics": "UTF-8 bytes of symlink target",
            }
        )
        return record
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file or symlink: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    after = path.lstat()
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_fields = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stable_fields != after_fields:
        raise RuntimeError(f"file changed while hashing: {path}")
    record.update(
        {
            "type": "file",
            "size_bytes": before.st_size,
            "sha256": digest.hexdigest(),
            "sha256_semantics": "file content",
        }
    )
    return record


def hash_records(paths: Iterable[Path], relative_to: Path) -> list[dict[str, Any]]:
    unique_paths = sorted(set(paths), key=lambda path: path.as_posix())
    total = len(unique_paths)
    if not total:
        return []

    def collect(path: Path) -> dict[str, Any]:
        return file_record(path, relative_to)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        for completed, record in enumerate(executor.map(collect, unique_paths), 1):
            records.append(record)
            if completed % 25 == 0 or completed == total:
                print(
                    f"COMPLETED_UNITS={completed}/{total} scope={relative_to}",
                    flush=True,
                )
    return sorted(records, key=lambda record: record["relative_path"])


def tree_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["relative_path"]):
        identity = {
            "relative_path": record["relative_path"],
            "type": record["type"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": tree_digest(records),
        "tree_sha256_semantics": (
            "SHA-256 over sorted JSON lines of relative path, type, size, "
            "and content/link-target SHA-256"
        ),
    }


def relevant_task_file(path: Path, task_dir: Path) -> bool:
    relative = path.relative_to(task_dir)
    return (
        (len(relative.parts) == 1 and path.suffix.lower() in TASK_ROOT_SUFFIXES)
        or (relative.parts[0] == "sbatch_logs" and path.suffix.lower() == ".log")
        or path.name in TASK_EXACT_NESTED_NAMES
        or (path.name.startswith("eval_") and path.suffix.lower() == ".log")
        or path.name.startswith(TASK_MARKER_PREFIXES)
        or path.name in TASK_MARKER_NAMES
    )


def collect_task_dir(task_dir: Path) -> dict[str, Any]:
    if not task_dir.is_dir():
        raise FileNotFoundError(task_dir)
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(task_dir, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {"repo", "__pycache__", ".venv", ".venv-recipe"}
        ]
        parent = Path(dirpath)
        for name in filenames:
            path = parent / name
            if relevant_task_file(path, task_dir):
                paths.append(path)
    records = hash_records(paths, task_dir)
    metrics_records = [
        record
        for record in records
        if Path(record["relative_path"]).name == "metrics.jsonl"
    ]
    identity_records = [
        record
        for record in records
        if Path(record["relative_path"]).name != "metrics.jsonl"
    ]
    return {
        "path": str(task_dir),
        "selection": (
            "Named task directory, limited to top-level control files, "
            "sbatch_logs/*.log, nested eval logs, exact manifest/config names, "
            "metrics.jsonl files, and status markers; repo/ and virtual "
            "environments excluded."
        ),
        **aggregate_records(records),
        "metrics_group": {
            **aggregate_records(metrics_records),
            "individual_records_omitted_for_compactness": True,
        },
        "identity_files": identity_records,
    }


def git_identity(root: Path) -> dict[str, Any]:
    dot_git = root / ".git"
    result: dict[str, Any] = {"dot_git_exists": dot_git.exists()}
    if not dot_git.exists():
        result.update({"revision": None, "branch": None, "tracked_status": None})
        return result

    def git(*args: str) -> str:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments
            ["/usr/bin/git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return completed.stdout.strip()

    result.update(
        {
            "revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "tracked_status": git(
                "status", "--porcelain=v1", "--untracked-files=no"
            ).splitlines(),
        }
    )
    return result


def source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative_root in SOURCE_RELATIVE_ROOTS:
        scope = root / relative_root
        if not scope.is_dir():
            raise FileNotFoundError(scope)
        for dirpath, dirnames, filenames in os.walk(scope, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {"__pycache__", ".pytest_cache"}
            ]
            parent = Path(dirpath)
            for name in filenames:
                path = parent / name
                if path.is_file() or path.is_symlink():
                    paths.append(path)
    for relative_path in SOURCE_EXPLICIT_FILES:
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def collect_source_tree(label: str, root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = hash_records(source_paths(root), root)
    return {
        "label": label,
        "path": str(root),
        "selection": {
            "recursive_roots": [path.as_posix() for path in SOURCE_RELATIVE_ROOTS],
            "explicit_files": [path.as_posix() for path in SOURCE_EXPLICIT_FILES],
            "excluded": [
                ".venv*",
                "runs/",
                "output/",
                "outputs/",
                "__pycache__/",
                "all unrelated repository subtrees",
            ],
        },
        "git": git_identity(root),
        **aggregate_records(records),
        "files": records,
    }


def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: value)


def collect_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    root: Path = dataset["path"]
    if not root.is_dir():
        raise FileNotFoundError(root)

    top_level_entries: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    with os.scandir(root) as iterator:
        for entry in sorted(iterator, key=lambda item: item.name):
            entry_path = root / entry.name
            entry_stat = entry_path.lstat()
            if entry.is_symlink():
                entry_type = "symlink"
            elif entry.is_dir(follow_symlinks=False):
                entry_type = "directory"
            elif entry.is_file(follow_symlinks=False):
                entry_type = "file"
            else:
                entry_type = "other"
            type_counts[entry_type] += 1
            top_level_entries.append(
                {
                    "name": entry.name,
                    "type": entry_type,
                    "size_bytes": entry_stat.st_size,
                    "mtime_ns": entry_stat.st_mtime_ns,
                    "link_target": (
                        os.readlink(entry_path) if entry_type == "symlink" else None
                    ),
                }
            )
    entries_sha256 = sha256_bytes(
        (
            json.dumps(top_level_entries, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    )

    metadata_paths = [
        root / name for name in dataset["metadata_names"] if (root / name).exists()
    ]
    metadata_records = hash_records(metadata_paths, root)
    metadata: list[dict[str, Any]] = []
    for record in metadata_records:
        item: dict[str, Any] = {"record": record}
        if Path(record["path"]).suffix == ".json":
            item["content"] = read_json_file(Path(record["path"]))
        metadata.append(item)

    return {
        "label": dataset["label"],
        "path": str(root),
        "top_level_only": True,
        "top_level_type_counts": dict(sorted(type_counts.items())),
        "top_level_entries_sha256": entries_sha256,
        "top_level_entries_sha256_semantics": (
            "SHA-256 over sorted JSON entry names, types, sizes, mtimes, "
            "and symlink targets; no recursive raw-data scan"
        ),
        "top_level_entries": top_level_entries,
        "metadata": metadata,
    }


def last_jsonl_record(path: Path) -> dict[str, Any]:
    line_count = 0
    last_nonempty: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line_count += 1
            if line.strip():
                last_nonempty = line.rstrip("\n")
    result: dict[str, Any] = {
        "line_count": line_count,
        "last_nonempty_line": last_nonempty,
    }
    if last_nonempty is not None:
        try:
            result["last_record"] = json.loads(
                last_nonempty, parse_constant=lambda value: value
            )
        except json.JSONDecodeError as error:
            result["last_record_parse_error"] = str(error)
    return result


def summarize_training_log(path: Path) -> dict[str, Any]:
    tail: deque[str] = deque(maxlen=20)
    line_count = 0
    last_epoch_line: str | None = None
    marker_counts = Counter()
    markers = ["NaN", "DIVERGENCE-GUARD", "WATCHDOG", "Traceback", "EXIT_CODE="]
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line_count += 1
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if "Epoch [" in stripped:
                last_epoch_line = stripped
            for marker in markers:
                marker_counts[marker] += stripped.count(marker)
    return {
        "line_count": line_count,
        "last_epoch_line": last_epoch_line,
        "marker_counts": dict(marker_counts),
        "tail_lines": list(tail),
    }


def collect_run(tree_label: str, run_id: str) -> dict[str, Any]:
    tree = SOURCE_TREES[tree_label]
    recipe = tree / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
    run_dir = recipe / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    log_path = tree / "output" / "drivaer_t2" / f"{run_id.removeprefix('t2_')}.log"
    checkpoints_dir = run_dir / "checkpoints"
    required = [config_path, metrics_path, log_path, checkpoints_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    config_record = file_record(config_path, run_dir)
    metrics_record = file_record(metrics_path, run_dir)
    log_record = file_record(log_path, tree)
    checkpoint_paths = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_file() or path.is_symlink()
    ]
    checkpoint_records = hash_records(checkpoint_paths, run_dir)
    if not checkpoint_records:
        raise RuntimeError(f"no checkpoint files: {checkpoints_dir}")

    epochs_by_family: dict[str, list[int]] = {}
    for record in checkpoint_records:
        match = CHECKPOINT_RE.match(Path(record["relative_path"]).name)
        if match:
            epochs_by_family.setdefault(match.group("family"), []).append(
                int(match.group("epoch"))
            )
    latest_epoch_by_family = {
        family: max(epochs) for family, epochs in sorted(epochs_by_family.items())
    }
    terminal_records = [
        record
        for record in checkpoint_records
        if (
            (match := CHECKPOINT_RE.match(Path(record["relative_path"]).name))
            and int(match.group("epoch"))
            == latest_epoch_by_family[match.group("family")]
        )
        or Path(record["relative_path"]).name == "norm_stats.pt"
    ]

    return {
        "tree_label": tree_label,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "resolved_config": {
            "record": config_record,
            "content": config_path.read_text(encoding="utf-8"),
        },
        "metrics": {
            "record": metrics_record,
            **last_jsonl_record(metrics_path),
        },
        "training_log": {
            "record": log_record,
            **summarize_training_log(log_path),
        },
        "checkpoints": {
            **aggregate_records(checkpoint_records),
            "latest_epoch_by_family": latest_epoch_by_family,
            "terminal_files": terminal_records,
            "files": checkpoint_records,
        },
    }


def collect_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "collection": {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "python": sys.version,
            "hash_algorithm": "SHA-256",
            "read_only_targets": True,
            "originals_copied_or_deleted": False,
        },
        "scope": {
            "task_dirs": [str(path) for path in TASK_DIRS],
            "source_trees": {label: str(path) for label, path in SOURCE_TREES.items()},
            "run_targets": [
                {"tree_label": tree_label, "run_id": run_id}
                for tree_label, run_id in RUN_TARGETS
            ],
            "datasets": [
                {"label": item["label"], "path": str(item["path"])} for item in DATASETS
            ],
            "explicit_exclusions": [
                "No recursive scan outside the named task/source/run/dataset roots.",
                "No raw dataset case files are hashed.",
                "No source-tree virtual environments, output/, outputs/, or runs/ "
                "are included in source digests.",
                "TensorBoard event files are not hashed; resolved configs, "
                "metrics.jsonl, training logs, and every direct checkpoint file "
                "are hashed for named runs.",
            ],
        },
        "limitations": [
            "The isolated copied trees have no .git metadata; their identity is "
            "the scoped content digest captured at manifest time.",
            "Source and dataset snapshots establish current content identity, "
            "not proof that mutable files were unchanged since each original run.",
            "This is an identity manifest, not an archive: original artifact "
            "bytes remain only at their existing paths.",
        ],
        "task_directories": [],
        "source_trees": [],
        "datasets": [],
        "runs": [],
        "errors": [],
    }

    for task_dir in TASK_DIRS:
        print(f"BEGIN_TASK={task_dir}", flush=True)
        try:
            manifest["task_directories"].append(collect_task_dir(task_dir))
        except Exception as error:  # preserve a partial manifest for diagnosis
            manifest["errors"].append(
                {"scope": "task_directory", "path": str(task_dir), "error": repr(error)}
            )

    for label, source_tree in SOURCE_TREES.items():
        print(f"BEGIN_SOURCE_TREE={label}", flush=True)
        try:
            manifest["source_trees"].append(collect_source_tree(label, source_tree))
        except Exception as error:
            manifest["errors"].append(
                {
                    "scope": "source_tree",
                    "label": label,
                    "path": str(source_tree),
                    "error": repr(error),
                }
            )

    for dataset in DATASETS:
        print(f"BEGIN_DATASET={dataset['label']}", flush=True)
        try:
            manifest["datasets"].append(collect_dataset(dataset))
        except Exception as error:
            manifest["errors"].append(
                {
                    "scope": "dataset",
                    "label": dataset["label"],
                    "path": str(dataset["path"]),
                    "error": repr(error),
                }
            )

    for tree_label, run_id in RUN_TARGETS:
        print(f"BEGIN_RUN={run_id}", flush=True)
        try:
            manifest["runs"].append(collect_run(tree_label, run_id))
        except Exception as error:
            manifest["errors"].append(
                {
                    "scope": "run",
                    "tree_label": tree_label,
                    "run_id": run_id,
                    "error": repr(error),
                }
            )

    manifest["complete"] = not manifest["errors"]
    manifest["completed_at_utc"] = utc_now()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = collect_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, args.output)
    print(
        "MANIFEST_SUMMARY "
        f"complete={manifest['complete']} "
        f"tasks={len(manifest['task_directories'])} "
        f"sources={len(manifest['source_trees'])} "
        f"datasets={len(manifest['datasets'])} "
        f"runs={len(manifest['runs'])} "
        f"errors={len(manifest['errors'])}",
        flush=True,
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
