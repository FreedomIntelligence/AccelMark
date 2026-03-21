"""
AccelMark OpenClaw Skill

Triggered when the user asks to benchmark their GPU or hardware.
Runs a mini benchmark and returns results via the chat interface.

This skill:
1. Detects hardware via collect_env.py
2. Auto-selects appropriate mini suite
3. Runs the benchmark (5 min max)
4. Queries AccelMark leaderboard for ranking
5. Returns human-readable report to user
6. Optionally submits result to community leaderboard
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ACCELMARK_REPO = os.environ.get("ACCELMARK_PATH", "~/accelmark")
LEADERBOARD_API = "https://juhaoliang1997.github.io/AccelMark/api"


def ensure_accelmark_installed() -> Path:
    """Check that accelmark repo is available, clone if not."""
    repo_path = Path(ACCELMARK_REPO).expanduser()

    if not repo_path.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/JuhaoLiang1997/AccelMark.git", str(repo_path)],
            check=True
        )

    return repo_path


def ensure_dependencies(repo_path: Path) -> None:
    """Install required Python packages if not present."""
    import platform
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
            check=True
        )


def collect_environment(repo_path: Path, output_dir: Path) -> dict:
    """Run collect_env.py and return parsed env_info."""
    env_path = output_dir / "env_info.json"
    subprocess.run(
        [sys.executable, "scripts/collect_env.py", "--output", str(env_path)],
        cwd=repo_path, capture_output=True, text=True, check=True
    )
    with open(env_path) as f:
        return json.load(f)


def _summarize_chips(env: dict) -> str:
    accelerators = env.get("accelerators", [])
    if not accelerators:
        cpu = env.get("cpu", {})
        return f"{cpu.get('model', 'Unknown CPU')} · {env.get('system_memory_gb', 0):.0f}GB RAM"
    chip = accelerators[0]
    mem = chip.get("memory_gb", 0)
    return f"{chip.get('name', 'Unknown')} ({mem:.0f}GB)"


def query_ranking(chip_name: str) -> dict | None:
    """
    Query the leaderboard for ranking data for this chip.
    Returns None if chip not found or network unavailable.
    """
    try:
        import urllib.request
        url = f"{LEADERBOARD_API}/index.json"
        with urllib.request.urlopen(url, timeout=5) as resp:
            index = json.loads(resp.read())
        return index.get(chip_name)
    except Exception:
        return None


def query_submission_rank(submission_name: str) -> dict | None:
    """
    Query rank for a specific submission.
    Used after submitting to show "you ranked #N".
    """
    try:
        import urllib.request
        url = f"{LEADERBOARD_API}/rank.json"
        with urllib.request.urlopen(url, timeout=5) as resp:
            ranks = json.loads(resp.read())
        return ranks.get(submission_name)
    except Exception:
        return None


def handle_submit(result: dict, openclaw_username: str) -> str:
    """
    Submit result to AccelMark leaderboard via GitHub Issue.
    The issue is processed by the process_submissions.yml CI workflow
    which validates and creates a PR automatically.
    """
    import urllib.request

    result["meta"]["submitted_by"] = openclaw_username

    issue_body = f"""## AccelMark Community Submission

**Chip**: {result['chip']['name']}
**Suite**: {result['suite_id']}
**Date**: {result['meta']['date']}
**Submitted via**: OpenClaw AccelMark Skill

```json
{json.dumps(result, indent=2)}
```
"""
    payload = json.dumps({
        "title": f"[submission] {result['chip']['name']} {result['suite_id']}",
        "body": issue_body,
        "labels": ["community-submission"],
    }).encode()

    req = urllib.request.Request(
        "https://api.github.com/repos/JuhaoLiang1997/AccelMark/issues",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        urllib.request.urlopen(req, timeout=10)
        return (
            "✅ Submitted! Your result will appear on the leaderboard "
            "after automated validation (usually within a few minutes).\n"
            f"View leaderboard: {LEADERBOARD_API.replace('/api', '/')}"
        )
    except Exception as e:
        return (
            f"❌ Submission failed: {str(e)}\n"
            "You can submit manually at: https://github.com/JuhaoLiang1997/AccelMark"
        )


def format_details(result: dict) -> str:
    """Format detailed benchmark results for 'details' follow-up."""
    rows = result["metrics"]["offline"]["results_by_batch_size"]
    model = result["model"]
    software = result["software"]
    meta = result["meta"]

    lines = [
        "📊 Full benchmark results:\n",
        f"{'Batch size':<12} {'Throughput':<16} {'Memory'}",
    ]

    best_row = None
    if rows:
        valid = [r for r in rows if not r.get("oom") and r["throughput_tokens_per_sec"]]
        best_row = max(valid, key=lambda r: r["throughput_tokens_per_sec"]) if valid else None

    for row in rows:
        bs = f"bs={row['batch_size']}"
        tput = f"{row['throughput_tokens_per_sec']:,.0f} tok/s"
        mem = f"{row['peak_memory_gb']:.1f}GB" if row.get("peak_memory_gb") else "N/A"
        marker = "  ← best" if best_row and row["batch_size"] == best_row["batch_size"] else ""
        lines.append(f"{bs:<12} {tput:<16} {mem}{marker}")

    lines.append(
        f"\nSuite: {result['suite_id']} (AccelMark)\n"
        f"Framework: {software['framework']} {software['framework_version']}\n"
        f"Model: {model['model_id'].split('/')[-1]} {model['precision']}\n"
        f"Date: {meta['date']}"
    )

    return "\n".join(lines)


def main(user_message: str, context: dict) -> str:
    """
    Called by OpenClaw when a trigger phrase is detected.
    """

    # Handle follow-up messages
    msg = user_message.lower().strip()
    if msg == "submit":
        pending = context.get("accelmark_pending_result")
        if not pending:
            return "No recent benchmark result to submit. Run 'benchmark my gpu' first."
        username = context.get("user", {}).get("username", "anonymous")
        return handle_submit(pending, username)

    if msg == "details":
        pending = context.get("accelmark_pending_result")
        if not pending:
            return "No recent benchmark result. Run 'benchmark my gpu' first."
        return format_details(pending)

    # Main benchmark flow
    try:
        repo_path = ensure_accelmark_installed()
        ensure_dependencies(repo_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Step 1: collect env
            env = collect_environment(repo_path, output_dir)
            chip_summary = _summarize_chips(env)

            # Step 2: select mode
            sys.path.insert(0, str(repo_path))
            from mini.mini_suite_selector import select_mode, select_mini_suite
            from mini.hardware_assessment import assess_hardware, format_assessment_report

            mode = select_mode(env)

            if mode == "assessment":
                result = assess_hardware(env)
                return format_assessment_report(result)

            # Step 3: select tier
            config = select_mini_suite(env)

            # Step 4: run benchmark
            from mini.run_mini import run_benchmark, build_result_json, format_benchmark_report
            benchmark_result = run_benchmark(config, env)

            # Step 5: build full result.json
            result = build_result_json(env, config, benchmark_result)

            # Step 6: query leaderboard for ranking
            chip_name = env["accelerators"][0]["name"]
            ranking = query_ranking(chip_name)

            # Step 7: store for follow-up
            context["accelmark_pending_result"] = result

            # Step 8: format and return
            return format_benchmark_report(result, config, ranking)

    except Exception as e:
        return (
            f"❌ Benchmark failed: {str(e)}\n\n"
            "Common fixes:\n"
            "• Make sure vLLM is installed: pip install vllm\n"
            "• For Apple Silicon: pip install mlx-lm\n"
            "• Check logs for details"
        )


if __name__ == "__main__":
    # For local testing outside OpenClaw
    print(main("benchmark my gpu", {}))
