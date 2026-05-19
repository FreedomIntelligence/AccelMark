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
    # Validate all runners
    python runners/validate_runners.py

    # Validate a specific runner folder (name or path)
    python runners/validate_runners.py --dir nvidia_vllm_47f5d58e
    python runners/validate_runners.py --dir runners/nvidia_vllm_47f5d58e
    python runners/validate_runners.py --dir /abs/path/to/nvidia_vllm_47f5d58e

    # Dry-run rename (shows what would change, touches nothing)
    python runners/validate_runners.py --dir nvidia_vllm_47f5d58e --rename --dry-run

    # Rename folder + update meta.json id to match correct hash
    python runners/validate_runners.py --dir nvidia_vllm_47f5d58e --rename
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not installed — schema validation skipped")

RUNNERS_DIR = Path(__file__).parent
SCHEMA_PATH = RUNNERS_DIR / "meta.schema.json"
PLATFORMS_CATALOG_PATH = RUNNERS_DIR.parent / "schema" / "platforms.json"

# Files that live flat in runners/ — not runner folders
BASE_FILES = {
    "benchmark_runner.py",
    "collect_env.py",
    "validate_submission.py",
    "validate_runners.py",
    "validate_suites.py",
    "hash_runner.py",
    "meta.schema.json",
    "protocol.py",
    "template",
    "platforms",
    "__pycache__",
    "__init__.py",
}

schema = None
if HAS_JSONSCHEMA and SCHEMA_PATH.exists():
    schema = json.loads(SCHEMA_PATH.read_text())

known_platforms: set[str] = set()
if PLATFORMS_CATALOG_PATH.exists():
    try:
        _catalog = json.loads(PLATFORMS_CATALOG_PATH.read_text())
        known_platforms = {
            p["id"]
            for p in (_catalog.get("platforms") or [])
            if isinstance(p, dict) and p.get("id")
        }
    except Exception:
        known_platforms = set()


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


def info(msg: str) -> None:
    print(f"    → {msg}")


def resolve_target_folder(dir_arg: str) -> Path:
    """
    Resolve --dir to an absolute Path.

    Accepts three forms:
      - bare folder name:   nvidia_vllm_47f5d58e
      - relative path:      runners/nvidia_vllm_47f5d58e
      - absolute path:      /home/user/AccelMark/runners/nvidia_vllm_47f5d58e
    """
    candidate = Path(dir_arg)

    # Already absolute or clearly a path (contains a separator)
    if candidate.is_absolute():
        return candidate

    # Try relative to cwd first
    if candidate.exists():
        return candidate.resolve()

    # Fall back: treat as a bare folder name inside RUNNERS_DIR
    under_runners = RUNNERS_DIR / candidate
    if under_runners.exists():
        return under_runners.resolve()

    # Return as-is; existence check will produce a clear error below
    return candidate.resolve()


def do_rename(folder: Path, correct_hash: str, dry_run: bool) -> Path:
    """
    Rename the runner folder and update meta.json so both reflect correct_hash.

    Returns the new folder path (same as input when already correct or on
    dry-run).  Exits with code 1 on any filesystem error.
    """
    parts      = folder.name.rsplit("_", 1)
    base       = parts[0]                          # e.g. "nvidia_vllm"
    old_name   = folder.name                       # e.g. "nvidia_vllm_deadbeef"
    new_name   = f"{base}_{correct_hash}"          # e.g. "nvidia_vllm_47f5d58e"
    new_folder = folder.parent / new_name

    meta_path  = folder / "meta.json"
    needs_folder_rename = old_name != new_name

    # ── Read meta so we can check / update id ────────────────────────────────
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    ✗ Cannot read meta.json for rename: {exc}")
        sys.exit(1)

    old_id         = meta.get("id", "")
    needs_meta_upd = old_id != new_name

    if not needs_folder_rename and not needs_meta_upd:
        ok("Already correct — nothing to rename.")
        return folder

    # ── Report what will happen ───────────────────────────────────────────────
    tag = "[dry-run] " if dry_run else ""

    if needs_folder_rename:
        info(f"{tag}Rename folder:  {old_name}  →  {new_name}")
    else:
        info(f"{tag}Folder name already correct: {old_name}")

    if needs_meta_upd:
        info(f"{tag}Update meta.json id:  '{old_id}'  →  '{new_name}'")
    else:
        info(f"{tag}meta.json id already correct: '{old_id}'")

    if dry_run:
        return folder   # no filesystem changes

    # ── Apply changes ─────────────────────────────────────────────────────────
    # 1. Rewrite meta.json id in-place (before rename so path is still valid)
    if needs_meta_upd:
        meta["id"] = new_name
        try:
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
            )
            ok(f"meta.json id updated → '{new_name}'")
        except OSError as exc:
            print(f"    ✗ Failed to write meta.json: {exc}")
            sys.exit(1)

    # 2. Rename the folder
    if needs_folder_rename:
        if new_folder.exists():
            print(f"    ✗ Cannot rename: target '{new_folder}' already exists.")
            sys.exit(1)
        try:
            shutil.move(str(folder), str(new_folder))
            ok(f"Folder renamed → {new_name}/")
        except OSError as exc:
            print(f"    ✗ Failed to rename folder: {exc}")
            sys.exit(1)

    return new_folder


# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Validate AccelMark runner folders.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--dir",
    metavar="RUNNER_DIR",
    default=None,
    help=(
        "Validate only this runner folder instead of all runners. "
        "Accepts a bare folder name (e.g. nvidia_vllm_47f5d58e), "
        "a path relative to cwd, or an absolute path."
    ),
)
parser.add_argument(
    "--rename",
    action="store_true",
    default=False,
    help=(
        "Fix hash mismatches by renaming the runner folder to match "
        "SHA-256(runner.py)[:8] and updating meta.json id to match. "
        "Requires --dir. Use --dry-run to preview changes first."
    ),
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    default=False,
    help="With --rename: print what would be renamed without touching anything.",
)
args = parser.parse_args()

# ── Flag guards ───────────────────────────────────────────────────────────────
if args.rename and not args.dir:
    parser.error("--rename requires --dir — specify which runner to rename.")

if args.dry_run and not args.rename:
    parser.error("--dry-run only makes sense together with --rename.")

# ── Build the list of folders to check ───────────────────────────────────────
if args.dir:
    target = resolve_target_folder(args.dir)
    if not target.exists():
        print(f"Error: --dir '{args.dir}' not found (resolved to '{target}').")
        sys.exit(1)
    if not target.is_dir():
        print(f"Error: --dir '{args.dir}' exists but is not a directory.")
        sys.exit(1)
    runner_folders = [target]
else:
    runner_folders = sorted(
        d for d in RUNNERS_DIR.iterdir()
        if d.is_dir() and d.name not in BASE_FILES and not d.name.startswith(".")
    )

if not runner_folders:
    print("No runner folders found.")
    sys.exit(0)

# ── Validate each folder ──────────────────────────────────────────────────────
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

    # ── Hash consistency ──────────────────────────────────────────────────────
    # Folder name must be {base}_{hash8} where hash8 = SHA-256[:8] of runner.py
    correct_hash = compute_hash(runner_py)
    parts        = folder.name.rsplit("_", 1)
    valid_suffix = (
        len(parts) == 2
        and len(parts[1]) == 8
        and all(c in "0123456789abcdef" for c in parts[1])
    )

    if not valid_suffix:
        err(
            f"Folder name '{folder.name}' does not end with a valid 8-char hex hash. "
            f"Expected format: {{platform}}_{{customname}}_{{hash8}}"
        )
        if args.rename:
            folder    = do_rename(folder, correct_hash, args.dry_run)
            meta_path = folder / "meta.json"
    elif parts[1] != correct_hash:
        err(
            f"Hash mismatch: folder ends with '{parts[1]}' but "
            f"SHA-256(runner.py)[:8] = '{correct_hash}'. "
            + ("" if args.rename else "Run with --rename to fix automatically.")
        )
        if args.rename:
            folder    = do_rename(folder, correct_hash, args.dry_run)
            meta_path = folder / "meta.json"
    else:
        ok(f"Hash consistent: {correct_hash}")

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

    # ── Platform catalogue check (warning only) ──────────────────────────────
    platform_id = meta.get("platform") or ""
    if known_platforms and platform_id and platform_id not in known_platforms:
        warn(
            f"Platform '{platform_id}' is not catalogued in schema/platforms.json. "
            f"The runner still validates (the schema accepts any lowercase identifier), "
            f"but please consider adding an entry so the README matrix can render a "
            f"human-readable label and stable sort order for this platform."
        )

    # ── meta.id must match folder name ────────────────────────────────────────
    if meta.get("id") != folder.name:
        err(
            f"meta.id '{meta.get('id')}' does not match folder name '{folder.name}'. "
            + ("" if args.rename else "They must be identical.")
        )
        # If --rename was given but do_rename() wasn't called yet (hash was
        # already correct but id drifted for some other reason), fix it now.
        if args.rename and not args.dry_run:
            meta["id"] = folder.name
            try:
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
                )
                ok(f"meta.json id corrected → '{folder.name}'")
            except OSError as exc:
                print(f"    ✗ Failed to write meta.json: {exc}")
                sys.exit(1)
        elif args.rename and args.dry_run:
            info(f"[dry-run] Would update meta.json id: '{meta.get('id')}' → '{folder.name}'")
    else:
        ok(f"meta.id matches folder name")

    # ── Duplicate ID check (only meaningful when scanning all runners) ────────
    if not args.dir:
        if folder.name in seen_ids:
            err(
                f"Duplicate runner ID '{folder.name}' — "
                f"already seen at {seen_ids[folder.name]}"
            )
        else:
            seen_ids[folder.name] = str(folder)

# ── Summary ──────────────────────────────────────────────────────────────────
checked = len(runner_folders)
scope   = f"'{runner_folders[0].name}'" if args.dir else f"{checked} runner folder(s)"
print(f"\n{'='*60}")
print(f"Checked {scope}: {errors} error(s), {warnings} warning(s)")

sys.exit(0 if errors == 0 else 1)