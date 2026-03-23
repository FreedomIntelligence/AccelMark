#!/usr/bin/env python3
"""
Validate a runner folder before submission.

Usage:
    python runners/validate_runners.py runners/nvidia_vllm_3f8a2c1d/

Checks:
  - runner.py exists
  - meta.json exists and is valid
  - meta.id matches folder name
  - folder name ends with correct SHA-256 hash of runner.py
  - no existing runner in runners/ shares the same ID
  - supersedes / deprecated_by targets exist (warnings if not found)
"""

import hashlib
import json
import sys
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

RUNNERS_DIR = Path(__file__).parent
SCHEMA_PATH = RUNNERS_DIR / "meta.schema.json"

schema = None
if HAS_JSONSCHEMA and SCHEMA_PATH.exists():
    schema = json.loads(SCHEMA_PATH.read_text())
elif not HAS_JSONSCHEMA:
    print("Warning: jsonschema not installed — schema validation skipped")
    print("         pip install jsonschema")
    print()


def compute_hash(runner_py: Path) -> str:
    return hashlib.sha256(runner_py.read_bytes()).hexdigest()[:8]


def validate(folder: Path) -> bool:
    errors   = 0
    warnings = 0

    def err(msg: str) -> None:
        nonlocal errors
        print(f"  ✗ {msg}")
        errors += 1

    def warn(msg: str) -> None:
        nonlocal warnings
        print(f"  ⚠ {msg}")
        warnings += 1

    def ok(msg: str) -> None:
        print(f"  ✓ {msg}")

    runner_py = folder / "runner.py"
    meta_path = folder / "meta.json"
    req_path  = folder / "requirements.txt"

    # ── Required files ────────────────────────────────────────────────────────
    print("Files:")
    if not runner_py.exists():
        err("runner.py is missing")
        return False  # nothing else is checkable without it
    ok("runner.py")

    if not meta_path.exists():
        err("meta.json is missing")
        return False
    ok("meta.json")

    if not req_path.exists():
        warn("requirements.txt missing (recommended — helps users install dependencies)")
    else:
        ok("requirements.txt")

    # ── Hash consistency ──────────────────────────────────────────────────────
    print("\nHash:")
    parts = folder.name.rsplit("_", 1)
    if len(parts) != 2 or len(parts[1]) != 8 or not all(
        c in "0123456789abcdef" for c in parts[1]
    ):
        err(
            f"Folder name '{folder.name}' does not end with a valid 8-char hex hash.\n"
            f"    Expected format: {{platform}}_{{customname}}_{{hash8}}\n"
            f"    Compute the correct name with:\n"
            f"    python runners/hash_runner.py {folder}/runner.py"
        )
        errors += 1
    else:
        actual_hash = compute_hash(runner_py)
        if parts[1] != actual_hash:
            err(
                f"Hash mismatch.\n"
                f"    Folder ends with : {parts[1]}\n"
                f"    runner.py hashes to: {actual_hash}\n"
                f"    Rename folder to: {parts[0]}_{actual_hash}"
            )
        else:
            ok(f"SHA-256(runner.py)[:8] = {actual_hash} ✓")

    # ── meta.json ─────────────────────────────────────────────────────────────
    print("\nmeta.json:")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        err(f"Not valid JSON: {e}")
        return False

    if schema and HAS_JSONSCHEMA:
        try:
            jsonschema.validate(meta, schema)
            ok("Valid against schema")
        except jsonschema.ValidationError as e:
            err(f"Schema error: {e.message}")
            errors += 1
    else:
        for field in ("id", "platform", "name", "framework", "submitted_by", "description"):
            if not meta.get(field):
                err(f"Missing required field: {field}")

    if meta.get("id") != folder.name:
        err(
            f"meta.id '{meta.get('id')}' does not match folder name '{folder.name}'.\n"
            f"    They must be identical."
        )
    else:
        ok(f"meta.id matches folder name")

    # ── Duplicate ID check ────────────────────────────────────────────────────
    print("\nDuplicate check:")
    existing = [
        d for d in RUNNERS_DIR.iterdir()
        if d.is_dir() and d.name == folder.name and d.resolve() != folder.resolve()
    ]
    if existing:
        err(
            f"A runner with ID '{folder.name}' already exists at:\n"
            f"    {existing[0]}\n"
            f"    If you modified runner.py, the hash will have changed — "
            f"rename your folder to the new hash."
        )
    else:
        ok("No existing runner with this ID")

    # ── supersedes / deprecated_by cross-reference ────────────────────────────
    supersedes    = meta.get("supersedes")
    deprecated_by = meta.get("deprecated_by")

    if supersedes or deprecated_by:
        print("\nLineage:")

    if supersedes:
        supersedes_path = RUNNERS_DIR / supersedes
        if not supersedes_path.exists():
            warn(
                f"meta.supersedes = '{supersedes}' but that runner folder "
                f"does not exist in runners/ yet.\n"
                f"    Make sure the old runner folder is included in your PR."
            )
        else:
            ok(f"Supersedes: {supersedes} ✓")

    if deprecated_by:
        deprecated_path = RUNNERS_DIR / deprecated_by
        if not deprecated_path.exists():
            warn(
                f"meta.deprecated_by = '{deprecated_by}' but that runner folder "
                f"does not exist in runners/ yet."
            )
        else:
            ok(f"Deprecated by: {deprecated_by} ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    if errors == 0 and warnings == 0:
        print(f"✓ PASSED — {folder.name} is ready to submit")
    elif errors == 0:
        print(f"✓ PASSED with {warnings} warning(s) — {folder.name} is ready to submit")
    else:
        print(f"✗ FAILED — {errors} error(s), {warnings} warning(s)")
        print(f"  Fix the errors above before opening a PR.")
    print("=" * 50)

    return errors == 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python runners/validate_runners.py <runner_folder>")
        print()
        print("Examples:")
        print("  python runners/validate_runners.py runners/nvidia_vllm_3f8a2c1d/")
        print("  python runners/validate_runners.py runners/amd_vllm_rocm_7b2e1d8f/")
        print()
        print("Checks hash consistency, meta.json validity, and duplicate IDs.")
        print("Run this before opening a PR to submit a new runner.")
        return 0 if len(sys.argv) == 1 else 1

    folder = Path(sys.argv[1]).resolve()

    if not folder.exists():
        print(f"Error: '{sys.argv[1]}' does not exist.")
        return 1
    if not folder.is_dir():
        print(f"Error: '{sys.argv[1]}' is not a directory.")
        print("Pass the runner folder, not the runner.py file.")
        return 1

    print(f"Validating: {folder.name}/")
    print("=" * 50)
    passed = validate(folder)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())