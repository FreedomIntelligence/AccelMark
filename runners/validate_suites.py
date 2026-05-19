#!/usr/bin/env python3
"""
Validate suite folders under suites/.

Checks per folder:
  - suite.json exists and parses as JSON
  - suite.json validates against schema/suite.schema.json
  - suite.suite_id matches the folder name
  - suite.dataset resolves to datasets/<name>/requests.jsonl

Usage:
    # Validate every suite
    python runners/validate_suites.py

    # Validate a specific suite folder (name or path)
    python runners/validate_suites.py --dir suite_A
    python runners/validate_suites.py --dir suites/suite_A
    python runners/validate_suites.py --dir /abs/path/to/suite_A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not installed — schema validation skipped")

REPO_ROOT   = Path(__file__).resolve().parent.parent
SUITES_DIR  = REPO_ROOT / "suites"
SCHEMA_PATH = REPO_ROOT / "schema" / "suite.schema.json"
DATASETS_DIR = REPO_ROOT / "datasets"

# Files / folders that live flat under suites/ — not suite folders
_NON_SUITE_NAMES = {"README.md", "__pycache__", ".DS_Store"}


def _load_schema() -> dict | None:
    if not HAS_JSONSCHEMA:
        return None
    if not SCHEMA_PATH.exists():
        print(f"Error: schema not found at {SCHEMA_PATH}")
        sys.exit(1)
    return json.loads(SCHEMA_PATH.read_text())


def _iter_suite_folders() -> list[Path]:
    if not SUITES_DIR.exists():
        return []
    out = []
    for entry in sorted(SUITES_DIR.iterdir()):
        if not entry.is_dir() or entry.name in _NON_SUITE_NAMES or entry.name.startswith("."):
            continue
        out.append(entry)
    return out


def _resolve_target(target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    # Allow "suite_A" or "suites/suite_A"
    if (SUITES_DIR / target).exists():
        return SUITES_DIR / target
    return REPO_ROOT / target


def validate_suite(folder: Path, schema: dict | None) -> list[str]:
    errors: list[str] = []
    name = folder.name
    suite_json = folder / "suite.json"

    if not suite_json.exists():
        errors.append(f"missing suite.json at {suite_json}")
        return errors

    try:
        data = json.loads(suite_json.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"suite.json is not valid JSON: {exc}")
        return errors

    declared_id = data.get("suite_id")
    if declared_id != name:
        errors.append(
            f"suite_id mismatch: folder is '{name}' but suite.suite_id is "
            f"'{declared_id}'."
        )

    if schema is not None:
        validator = jsonschema.Draft7Validator(schema)
        for err in validator.iter_errors(data):
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            errors.append(f"schema: {path}: {err.message}")

    dataset = data.get("dataset")
    if dataset:
        dataset_path = DATASETS_DIR / dataset / "requests.jsonl"
        if not dataset_path.exists():
            errors.append(
                f"dataset '{dataset}' referenced by suite.json does not exist "
                f"at {dataset_path}. Add the dataset to datasets/ or fix the "
                f"'dataset' field."
            )

    return errors


def _print_result(folder: Path, errors: list[str]) -> None:
    if errors:
        print(f"FAIL  {folder.name}")
        for err in errors:
            print(f"        - {err}")
    else:
        print(f"OK    {folder.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate suite folders under suites/."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Validate a single suite folder (name, relative, or absolute path).",
    )
    args = parser.parse_args()

    schema = _load_schema()

    if args.dir:
        target = _resolve_target(args.dir)
        if not target.exists() or not target.is_dir():
            print(f"Error: '{args.dir}' is not an existing directory.")
            return 2
        folders = [target]
    else:
        folders = _iter_suite_folders()
        if not folders:
            print("No suite folders found under suites/.")
            return 0

    total_errors = 0
    for folder in folders:
        errs = validate_suite(folder, schema)
        _print_result(folder, errs)
        total_errors += len(errs)

    print()
    if total_errors:
        print(f"Found {total_errors} problem(s) across {len(folders)} suite folder(s).")
        return 1
    print(f"All {len(folders)} suite folder(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
