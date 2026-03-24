#!/usr/bin/env python3
"""
Compute the implementation ID for a runner folder.

The ID is:  {folder_base}_{first8ofSHA256(runner.py)}

Usage:
    # Compute ID for an existing runner
    python runners/hash_runner.py runners/nvidia_vllm_6e78e779/

    # Compute the correct ID before naming a new runner folder
    python runners/hash_runner.py path/to/new_runner_dir/
"""

import hashlib
import sys
from pathlib import Path


def compute_hash(runner_path: Path) -> str:
    """Return the first 8 hex chars of SHA-256 of runner.py content."""
    return hashlib.sha256(runner_path.read_bytes()).hexdigest()[:8]


def compute_id(folder: Path) -> str:
    """
    Compute the canonical implementation ID for a runner folder.

    The folder name must follow: {platform}_{customname}_{hash8}
    This function computes what the hash8 suffix SHOULD be, given runner.py.
    """
    runner_path = folder / "runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"runner.py not found in {folder}")

    hash8       = compute_hash(runner_path)
    folder_name = folder.name

    # Strip any existing hash8 suffix to get the base name
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and all(
        c in "0123456789abcdef" for c in parts[1]
    ):
        base = parts[0]
    else:
        base = folder_name

    return f"{base}_{hash8}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <runner_folder>", file=sys.stderr)
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"Error: {folder} does not exist", file=sys.stderr)
        sys.exit(1)

    impl_id = compute_id(folder)
    print(impl_id)
