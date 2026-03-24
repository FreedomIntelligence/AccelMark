#!/usr/bin/env python3
"""
Validate all runner folders in runners/.

Checks per folder:
  - runner.py exists
  - meta.json exists and is valid against meta.schema.json
  - meta.id matches folder name
  - folder name ends with correct SHA-256 hash of runner.py
  - no two folders have the same ID

Usage:
    python runners/validate_runners.py
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
    print("Warning: jsonschema not installed — schema validation skipped")

RUNNERS_DIR  = Path(__file__).parent
SCHEMA_PATH  = RUNNERS_DIR / "meta.schema.json"

# Files that live flat in runners/ — not runner folders
BASE_FILES = {
    "benchmark_runner.py",
    "collect_env.py",
    "validate_submission.py",
    "validate_runners.py",
    "hash_runner.py",
    "meta.schema.json",
    "protocol.py",
    "template",
    "__pycache__",
    "__init__.py",
}

schema = None
if HAS_JSONSCHEMA and SCHEMA_PATH.exists():
    schema = json.loads(SCHEMA_PATH.read_text())


def compute_hash(runner_py: Path) -> str:
    return hashlib.sha256(runner_py.read_bytes()).hexdigest()[:8]


errors   = 0
warnings = 0
seen_ids = {}


def err(msg: str) -> None:
    global errors
    print(f"    ✗ ERROR: {msg}")
    errors += 1


def warn(msg: str) -> None:
    global warnings
    print(f"    ⚠ WARNING: {msg}")
    warnings += 1


def ok(msg: str) -> None:
    print(f"    ✓ {msg}")


runner_folders = sorted(
    d for d in RUNNERS_DIR.iterdir()
    if d.is_dir() and d.name not in BASE_FILES and not d.name.startswith(".")
)

if not runner_folders:
    print("No runner folders found.")
    sys.exit(0)

for folder in runner_folders:
    print(f"\n{folder.name}/")

    runner_py = folder / "runner.py"
    meta_path = folder / "meta.json"
    req_path  = folder / "requirements.txt"

    # ── Required files ──────────────────────────────────────────────────────
    if not runner_py.exists():
        err("runner.py is missing")
        continue  # cannot do hash check without runner.py

    ok("runner.py present")

    if not meta_path.exists():
        err("meta.json is missing")
        continue
    ok("meta.json present")

    if not req_path.exists():
        warn("requirements.txt missing (recommended)")
    else:
        ok("requirements.txt present")

    # ── Hash consistency ─────────────────────────────────────────────────────
    # Folder name must be {base}_{hash8} where hash8 = SHA-256[:8] of runner.py
    parts = folder.name.rsplit("_", 1)
    if len(parts) != 2 or len(parts[1]) != 8 or not all(
        c in "0123456789abcdef" for c in parts[1]
    ):
        err(
            f"Folder name '{folder.name}' does not end with a valid 8-char hex hash. "
            f"Expected format: {{platform}}_{{customname}}_{{hash8}}"
        )
    else:
        expected_hash = compute_hash(runner_py)
        if parts[1] != expected_hash:
            err(
                f"Hash mismatch: folder ends with '{parts[1]}' but "
                f"SHA-256(runner.py)[:8] = '{expected_hash}'. "
                f"Rename folder to: {parts[0]}_{expected_hash}"
            )
        else:
            ok(f"Hash consistent: {expected_hash}")

    # ── meta.json validation ─────────────────────────────────────────────────
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        err(f"meta.json is not valid JSON: {e}")
        continue

    if schema and HAS_JSONSCHEMA:
        try:
            jsonschema.validate(meta, schema)
            ok("meta.json valid against schema")
        except jsonschema.ValidationError as e:
            err(f"meta.json schema error: {e.message}")
            continue
    else:
        # Manual minimum checks without jsonschema
        for field in ("id", "platform", "name", "framework", "submitted_by", "description"):
            if not meta.get(field):
                err(f"meta.json missing required field: {field}")

    # ── meta.id must match folder name ───────────────────────────────────────
    if meta.get("id") != folder.name:
        err(
            f"meta.id '{meta.get('id')}' does not match folder name '{folder.name}'. "
            "They must be identical."
        )
    else:
        ok(f"meta.id matches folder name")

    # ── Duplicate ID check ───────────────────────────────────────────────────
    if folder.name in seen_ids:
        err(
            f"Duplicate runner ID '{folder.name}' — "
            f"already seen at {seen_ids[folder.name]}"
        )
    else:
        seen_ids[folder.name] = str(folder)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Checked {len(runner_folders)} runner folder(s): "
      f"{errors} error(s), {warnings} warning(s)")

sys.exit(0 if errors == 0 else 1)