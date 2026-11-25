"""Unit tests for AccelMark contributor onboarding CLI (run.py, advisor.py, validate_submission.py)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run as run_cli  # noqa: E402


def test_resolve_runner_aliases():
    rid, note = run_cli.resolve_runner_id("vllm")
    assert rid == "nvidia_vllm_47f5d58e"
    assert note and "alias" in note.lower()

    rid, note = run_cli.resolve_runner_id("sglang")
    assert rid == "nvidia_sglang_c43a8309"

    rid, note = run_cli.resolve_runner_id("mlx")
    assert rid == "apple_mlx_lm_9546b8b5"

    rid, note = run_cli.resolve_runner_id("ascend")
    assert rid == "ascend_vllm_ascend_d4aa9fda"

    # Exact ID passes through without alias note
    rid, note = run_cli.resolve_runner_id("nvidia_vllm_47f5d58e")
    assert rid == "nvidia_vllm_47f5d58e"
    assert note is None


def test_check_env_with_alias():
    rc = run_cli.cmd_check_env("vllm")
    assert rc == 0


def test_check_env_invalid_runner():
    rc = run_cli.cmd_check_env("nonexistent_runner_xyz")
    assert rc == 1


def test_validate_submission_accepts_positional_and_result_json():
    # Pick any existing verified submission directory
    verified_dir = REPO_ROOT / "results" / "verified"
    sample_dir = next(
        p for p in sorted(verified_dir.iterdir())
        if p.is_dir() and (p / "result.json").exists()
    )
    res_dir = subprocess.run(
        [sys.executable, str(REPO_ROOT / "runners" / "validate_submission.py"), str(sample_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res_dir.returncode == 0, res_dir.stdout + res_dir.stderr

    res_file = subprocess.run(
        [sys.executable, str(REPO_ROOT / "runners" / "validate_submission.py"), str(sample_dir / "result.json")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res_file.returncode == 0, res_file.stdout + res_file.stderr


def test_submit_zip_packaging(tmp_path: Path):
    verified_dir = REPO_ROOT / "results" / "verified"
    sample_dir = next(
        p for p in sorted(verified_dir.iterdir())
        if p.is_dir() and (p / "result.json").exists() and (p / "env_info.json").exists()
    )
    # Copy minimal files into tmp_path so we don't pollute results/verified
    sub_copy = tmp_path / sample_dir.name
    sub_copy.mkdir()
    for fname in ("result.json", "env_info.json", "accuracy.json"):
        src = sample_dir / fname
        if src.exists():
            (sub_copy / fname).write_bytes(src.read_bytes())

    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "--submit", str(sub_copy), "--zip"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    zip_path = tmp_path / f"{sample_dir.name}.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert f"{sample_dir.name}/result.json" in names
        assert f"{sample_dir.name}/env_info.json" in names


def test_advisor_runs_cleanly():
    import json

    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "advisor.py"), "--suite", "suite_A", "--limit", "3"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "AccelMark suite_A" in res.stdout

    # Budget + tier + JSON output
    res_json = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "advisor.py"),
            "--suite", "suite_A",
            "--budget", "2.0",
            "--tier", "reference",
            "--format", "json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res_json.returncode == 0, res_json.stdout + res_json.stderr
    payload = json.loads(res_json.stdout)
    assert payload["suite_id"] == "suite_A"
    assert all(r["price_usd_per_hr"] <= 2.0 for r in payload["results"])
    assert all(r["runner_tier"] == "reference" for r in payload["results"])

    # Engine comparison + Markdown output
    res_eng = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "advisor.py"),
            "--engine-compare", "A100",
            "--format", "markdown",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res_eng.returncode == 0, res_eng.stdout + res_eng.stderr
    assert "| Chip | Suite | Best Engine |" in res_eng.stdout
