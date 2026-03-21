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
        yield_message("AccelMark not found locally. Cloning repository (~30 seconds)...")
        subprocess.run(
            ["git", "clone", "https://github.com/JuhaoLiang1997/AccelMark.git", str(repo_path)],
            check=True
        )

    return repo_path


def ensure_dependencies(repo_path: Path) -> None:
    """Install required Python packages if not present."""
    try:
        import vllm
    except ImportError:
        yield_message("Installing vLLM (one-time setup, may take a few minutes)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "vllm", "--quiet"],
            check=True
        )


def run_benchmark(repo_path: Path, output_dir: Path) -> dict:
    """Run mini/run_mini.py and return result.json contents."""
    result = subprocess.run(
        [sys.executable, "mini/run_mini.py", "--output-dir", str(output_dir)],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed:\n{result.stderr}")

    result_path = output_dir / "result.json"
    with open(result_path) as f:
        return json.load(f)


def query_leaderboard(result: dict) -> dict | None:
    """Query the AccelMark leaderboard API for ranking data."""
    try:
        import urllib.request
        import urllib.parse
        chip_name = result["chip"]["name"]
        encoded = urllib.parse.quote(chip_name)
        url = f"{LEADERBOARD_API}/rank?chip={encoded}&suite=mini"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def submit_to_leaderboard(result: dict, repo_path: Path) -> bool:
    """Submit result to AccelMark community leaderboard via GitHub API."""
    # Implementation: POST to a GitHub Actions workflow dispatch endpoint
    # that creates a PR with the result.json
    # For now, guide the user to submit manually
    return False


def handle_submit(result: dict, username: str) -> str:
    """
    Called when user replies 'submit' after seeing benchmark results.
    Creates a GitHub issue with the result data for maintainer review.
    """
    result["meta"]["submitted_by"] = username
    result["meta"]["submission_type"] = "individual"

    # POST to GitHub Issues API (no auth required for public repo)
    # Maintainers periodically process these into PRs
    import urllib.request

    issue_body = f"""
## AccelMark Community Submission

**Chip**: {result['chip']['name']}
**Suite**: {result['suite_id']}
**Date**: {result['meta']['date']}

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
    urllib.request.urlopen(req)
    return "✅ Submitted! Your result will appear on the leaderboard after review."


def format_report(result: dict, ranking: dict | None) -> str:
    """Format the benchmark results for chat display."""
    chip = result["chip"]
    model = result["model"]
    rows = result["metrics"]["offline"]["results_by_batch_size"]
    valid = [r for r in rows if not r.get("oom") and r["throughput_tokens_per_sec"]]

    lines = ["✅ *AccelMark Benchmark Complete*\n"]
    lines.append(f"🖥️  **{chip['name']}** ({chip['memory_gb_per_chip']:.0f}GB VRAM)")
    lines.append(f"🤖 Model: `{model['model_id'].split('/')[-1]}`\n")

    if valid:
        best = max(valid, key=lambda r: r["throughput_tokens_per_sec"])
        lines.append(f"⚡ **{best['throughput_tokens_per_sec']:,.0f} tokens/sec**")
        if best.get("peak_memory_gb"):
            pct = best["peak_memory_gb"] / chip["memory_gb_per_chip"] * 100
            lines.append(f"💾 Memory: {best['peak_memory_gb']:.1f}GB / {chip['memory_gb_per_chip']:.0f}GB ({pct:.0f}%)")

    if ranking:
        lines.append(f"\n📊 **Leaderboard ranking:**")
        lines.append(f"   #{ranking['rank']} of {ranking['total']} · Better than {ranking['percentile']:.0f}%")

    notes = result.get("meta", {}).get("notes", "")
    if "tier:" in notes.lower():
        tier = notes.split("Tier:")[-1].strip()
        lines.append(f"\n🎯 Test tier: {tier}")

    lines.append("\nReply **'submit'** to add your result to the community leaderboard.")
    lines.append("Reply **'details'** to see full benchmark data.")

    return "\n".join(lines)


def main(user_message: str, context: dict) -> str:
    """
    Main entry point called by OpenClaw.

    Args:
        user_message: The message that triggered this skill
        context: OpenClaw context (user profile, previous messages, etc.)

    Returns:
        Response string to send back to the user
    """
    try:
        repo_path = ensure_accelmark_installed()
        ensure_dependencies(repo_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Detect hardware first, report to user
            env_result = subprocess.run(
                [sys.executable, "scripts/collect_env.py", "--output",
                 str(output_dir / "env_info.json")],
                cwd=repo_path, capture_output=True, text=True
            )

            # Read detected hardware for early feedback
            env_path = output_dir / "env_info.json"
            if env_path.exists():
                with open(env_path) as f:
                    env = json.load(f)
                accelerators = env.get("accelerators", [])
                if accelerators:
                    chip = accelerators[0]["name"]
                    mem = accelerators[0]["memory_gb"]
                    # Note: in real OpenClaw skill, you'd yield this message
                    # before blocking on the benchmark
                    print(f"Detected: {chip} ({mem:.0f}GB). Running benchmark...")

            result = run_benchmark(repo_path, output_dir)
            ranking = query_leaderboard(result)
            report = format_report(result, ranking)

            # Store result in context for follow-up "submit" message
            # OpenClaw's context/memory system would handle this
            context["accelmark_pending_result"] = result

            return report

    except Exception as e:
        return (
            f"❌ Benchmark failed: {str(e)}\n\n"
            "Common fixes:\n"
            "• Make sure you have a GPU with enough VRAM\n"
            "• Try: `pip install vllm`\n"
            "• Check AccelMark docs: https://github.com/JuhaoLiang1997/AccelMark"
        )


if __name__ == "__main__":
    # For local testing outside OpenClaw
    print(main("benchmark my gpu", {}))
