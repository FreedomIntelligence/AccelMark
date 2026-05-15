#!/usr/bin/env python3
"""
Regenerate the "Supported platforms" matrix in the top-level README.md
from runner metadata.

Each runner's ``meta.json`` declares which AccelMark suites it supports
via ``suite_support``; the table here is a simple projection of that
data and never needs to be hand-edited. ``schema/platforms.json``
provides the human-readable hardware label and the row ordering.

Modes:

    # rewrite README.md in place
    python tools/generate_platforms_matrix.py

    # fail (exit 1) if README.md is out of sync with the runner metadata —
    # intended for CI so a PR that adds a runner without updating the
    # README is rejected automatically
    python tools/generate_platforms_matrix.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNERS_DIR = _REPO_ROOT / "runners"
_PLATFORMS_CATALOG = _REPO_ROOT / "schema" / "platforms.json"
_README = _REPO_ROOT / "README.md"

START_MARKER = "<!-- platforms-matrix:start -->"
END_MARKER = "<!-- platforms-matrix:end -->"

# Files that live flat in runners/ — not runner folders.
_BASE_FILES = {
    "benchmark_runner.py",
    "collect_env.py",
    "validate_submission.py",
    "validate_runners.py",
    "hash_runner.py",
    "gen_pr_summary.py",
    "meta.schema.json",
    "protocol.py",
    "template",
    "platforms",
    "__pycache__",
    "__init__.py",
    "README.md",
}

SUITE_KEYS = ["A", "B", "C", "D", "E", "F", "G"]

STATUS_GLYPH = {
    "validated": "✓",
    "pending": "⋯",
    "unsupported": "—",
}


def _load_platforms_catalog() -> dict[str, dict]:
    if not _PLATFORMS_CATALOG.exists():
        return {}
    try:
        data = json.loads(_PLATFORMS_CATALOG.read_text())
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for entry in data.get("platforms") or []:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        if pid:
            out[pid] = entry
    return out


def _iter_runner_metas() -> Iterable[dict]:
    for folder in sorted(_RUNNERS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in _BASE_FILES or folder.name.startswith("."):
            continue
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        # Skip runners that have been superseded — they should not clutter
        # the matrix once a successor is merged.
        if meta.get("deprecated_by"):
            continue
        yield meta


def _runner_row(meta: dict, catalog: dict[str, dict]) -> tuple[tuple, list[str]]:
    """Return (sort_key, row_cells) for a runner."""
    platform_id = meta.get("platform") or "other"
    platform_entry = catalog.get(platform_id) or {}
    hardware_label = (
        meta.get("hardware_label")
        or platform_entry.get("display_name")
        or platform_id.capitalize()
    )

    runner_id = meta.get("id") or "?"
    framework = meta.get("framework") or "?"

    suite_support = meta.get("suite_support") or {}
    suite_cells = [STATUS_GLYPH.get(suite_support.get(s), "?") for s in SUITE_KEYS]

    sort_order = int(platform_entry.get("sort_order", 999))
    sort_key = (sort_order, platform_id, hardware_label, runner_id)

    row = [
        hardware_label,
        f"`{runner_id}`",
        framework,
        *suite_cells,
    ]
    return sort_key, row


def _build_table() -> str:
    catalog = _load_platforms_catalog()
    rows = []
    seen_unknown_platforms: set[str] = set()
    for meta in _iter_runner_metas():
        if (meta.get("platform") or "") not in catalog:
            seen_unknown_platforms.add(meta.get("platform") or "?")
        rows.append(_runner_row(meta, catalog))

    rows.sort(key=lambda kr: kr[0])

    header = (
        "| Hardware | Runner folder | Framework "
        + "".join(f"| {s} " for s in SUITE_KEYS)
        + "|"
    )
    sep = (
        "|---|---|---"
        + "".join("|:-:" for _ in SUITE_KEYS)
        + "|"
    )

    body_lines = []
    for _key, cells in rows:
        body_lines.append("| " + " | ".join(cells) + " |")

    legend = (
        "_Legend: ✓ validated · ⋯ author-declared "
        "(not smoke-tested in this repo yet) · — unsupported._"
    )

    parts = [header, sep, *body_lines, "", legend]

    if seen_unknown_platforms:
        unknown = ", ".join(sorted(seen_unknown_platforms))
        print(
            f"WARNING: encountered platform id(s) not catalogued in "
            f"schema/platforms.json: {unknown}. The matrix still renders "
            f"using fallbacks, but please consider opening a small PR "
            f"adding them to the catalog.",
            file=sys.stderr,
        )

    return "\n".join(parts).rstrip() + "\n"


def _splice_into_readme(table: str) -> str:
    src = _README.read_text()
    if START_MARKER not in src or END_MARKER not in src:
        raise SystemExit(
            f"README.md is missing the platforms-matrix markers. "
            f"Expected '{START_MARKER}' and '{END_MARKER}' on their own lines."
        )
    pre, _rest = src.split(START_MARKER, 1)
    _mid, post = _rest.split(END_MARKER, 1)
    return f"{pre}{START_MARKER}\n{table}{END_MARKER}{post}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if README.md is out of sync with runner metadata "
        "(does not write).",
    )
    args = parser.parse_args()

    table = _build_table()
    new_readme = _splice_into_readme(table)
    current = _README.read_text()

    if new_readme == current:
        print("README.md platforms matrix is up to date.")
        return 0

    if args.check:
        print(
            "ERROR: README.md platforms matrix is out of sync with "
            "runners/*/meta.json. Run:\n"
            "    python tools/generate_platforms_matrix.py\n"
            "and commit the result.",
            file=sys.stderr,
        )
        return 1

    _README.write_text(new_readme)
    print("README.md platforms matrix regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
