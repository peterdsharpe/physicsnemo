#!/usr/bin/env python3
"""Freeze nested, validation-only SHIFT-SUV adaptation pilot splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DESIGN_SALT = "shift-suv-adaptation-pilot-v1"
PILOT_BUDGETS = (64, 128)


def _read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _ordered_train_cases(family: str, cases: list[str]) -> list[str]:
    return sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f"{DESIGN_SALT}\0{family}\0{case}".encode()
        ).hexdigest(),
    )


def _split_hash(cases: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(cases)) + "\n").encode()).hexdigest()


def freeze_family(
    *,
    family: str,
    source_manifest: dict[str, list[str]],
    output_dir: Path,
) -> dict[str, Any]:
    required = {"train", "val", "test"}
    if source_manifest.keys() != required:
        raise ValueError(
            f"{family}: expected manifest keys {sorted(required)}, "
            f"got {sorted(source_manifest)}"
        )

    split_sets = {name: set(cases) for name, cases in source_manifest.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"{family}: {left}/{right} overlap: {sorted(overlap)}")

    ordered = _ordered_train_cases(family, source_manifest["train"])
    if len(ordered) < max(PILOT_BUDGETS) + 4:
        raise ValueError(f"{family}: too few training cases for the pilot")

    pilot_manifest = {
        "canary_train": sorted(ordered[:2]),
        "canary_val": sorted(ordered[2:4]),
        **{
            f"train_{budget}": sorted(ordered[:budget])
            for budget in PILOT_BUDGETS
        },
        "val": sorted(source_manifest["val"]),
    }
    if not set(pilot_manifest["train_64"]) < set(pilot_manifest["train_128"]):
        raise AssertionError(f"{family}: pilot training subsets are not nested")
    if set(pilot_manifest["train_128"]) & set(pilot_manifest["val"]):
        raise AssertionError(f"{family}: pilot train/validation overlap")

    manifest_path = output_dir / f"{family}_pilot_manifest.json"
    manifest_sha256 = _write_json(manifest_path, pilot_manifest)
    return {
        "family": family,
        "source_counts": {
            name: len(source_manifest[name]) for name in ("train", "val", "test")
        },
        "pilot_counts": {name: len(cases) for name, cases in pilot_manifest.items()},
        "pilot_manifest": manifest_path.name,
        "pilot_manifest_sha256": manifest_sha256,
        "source_split_sha256": {
            name: _split_hash(source_manifest[name])
            for name in ("train", "val", "test")
        },
        "checks": {
            "source_splits_disjoint": True,
            "pilot_train_subsets_nested": True,
            "pilot_uses_source_train_and_val_only": True,
            "test_cases_exported_to_pilot_manifest": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for family in ("estate", "fastback"):
        source = _read_json(args.inventory_dir / f"{family}_manifest.json")
        reports.append(
            freeze_family(
                family=family,
                source_manifest=source,
                output_dir=args.output_dir,
            )
        )

    touched = _read_json(args.inventory_dir / "touched_pilot_manifest.json")
    estate_source = _read_json(args.inventory_dir / "estate_manifest.json")
    touched_cases = set(touched["eval_pilot"])
    if not touched_cases <= set(estate_source["test"]):
        raise ValueError("Previously touched estate cases are not all in source test")
    if touched_cases & (set(estate_source["train"]) | set(estate_source["val"])):
        raise ValueError("Previously touched estate cases leak into train/validation")

    audit = {
        "schema_version": 1,
        "design_salt": DESIGN_SALT,
        "pilot_budgets": list(PILOT_BUDGETS),
        "families": reports,
        "previously_touched_estate_cases": len(touched_cases),
        "checks": {
            "touched_estate_cases_are_test_only": True,
            "pilot_manifests_contain_no_test_split": True,
        },
    }
    _write_json(args.output_dir / "split_audit.json", audit)


if __name__ == "__main__":
    main()
