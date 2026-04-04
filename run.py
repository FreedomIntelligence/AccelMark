#!/usr/bin/env python3
"""
AccelMark unified runner entry point.

Usage:
    # List all available runners
    python run.py --list

    # Run a benchmark
    python run.py --runner nvidia_vllm_2b3890cf --suite suite_A --scenario all

    # Run a benchmark on multiple chips
    # Set tensor_parallel_size in configs/runner_configs/runner_nvidia_vllm_2b3890cf.yaml
    # or pass --tensor-parallel-size directly (supported by runners that accept it)
    python run.py --runner nvidia_vllm_2b3890cf --suite suite_B --scenario all --tensor-parallel-size 4

    # Serve — using a suite for model + generation params
    python run.py --runner nvidia_vllm_2b3890cf --suite suite_A --serve

    # Serve — specifying the model directly (no suite required)
    python run.py --runner nvidia_vllm_2b3890cf --model meta-llama/Llama-3.1-8B-Instruct --serve

    # Serve — suite as base, override model and tune params
    python run.py --runner nvidia_vllm_2b3890cf --suite suite_A --serve \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --max-tokens 4096 --port 8080 --workers 8 --api-key secret

    # All flags after --runner <id> are passed through to the runner unchanged (non-serve mode)
    python run.py --runner nvidia_vllm_2b3890cf --suite suite_A --scenario offline --output-dir ./my_result
"""

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent
RUNNERS_DIR = REPO_ROOT / "runners"
SUITES_DIR  = REPO_ROOT / "suites"

# Files that live flat in runners/ — not runner folders
_BASE_FILES = {
    "benchmark_runner.py", "collect_env.py", "validate_submission.py",
    "validate_runners.py", "hash_runner.py", "meta.schema.json",
    "protocol.py", "__pycache__", "__init__.py",
}


# ── Runner discovery ──────────────────────────────────────────────────────────

def discover_runners() -> dict[str, dict]:
    """
    Return a dict of {runner_id: meta} for all valid runner folders.
    Runners with missing or unreadable meta.json are included with partial data.
    """
    runners = {}
    for folder in sorted(RUNNERS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in _BASE_FILES or folder.name.startswith("."):
            continue
        if not (folder / "runner.py").exists():
            continue

        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {"id": folder.name, "name": folder.name, "platform": "unknown",
                        "description": "(meta.json unreadable)"}
        else:
            meta = {"id": folder.name, "name": folder.name, "platform": "unknown",
                    "description": "(no meta.json)"}

        runners[folder.name] = meta

    return runners


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args) -> int:
    show_all = getattr(args, 'all', False)
    runners  = discover_runners()

    if not runners:
        print("No runners found in runners/")
        print("See CONTRIBUTING.md to add a runner.")
        return 0

    by_platform: dict[str, list] = {}
    for rid, meta in runners.items():
        is_deprecated = bool(meta.get("deprecated_by"))
        if is_deprecated and not show_all:
            continue
        platform = meta.get("platform", "other")
        by_platform.setdefault(platform, []).append((rid, meta))

    if not by_platform:
        print("No active runners found.")
        print("Run with --all to include deprecated runners.")
        return 0

    total_shown = sum(len(v) for v in by_platform.values())
    total_all   = len(runners)
    hidden      = total_all - total_shown

    print(f"\nAvailable runners ({total_shown} active"
          + (f", {hidden} deprecated — run --list --all to show" if hidden and not show_all else "")
          + ")\n")

    for platform in sorted(by_platform):
        print(f"  {platform.upper()}")
        for rid, meta in by_platform[platform]:
            deprecated_by    = meta.get("deprecated_by")
            supersedes_chain = meta.get("supersedes_chain") or []

            status = ""
            if deprecated_by and show_all:
                status = f"  [DEPRECATED → use {deprecated_by}]"

            print(f"    {rid}{status}")
            print(f"      {meta.get('name', rid)}")
            print(f"      {meta.get('description', '')}")
            if supersedes_chain:
                print(f"      Replaces: {supersedes_chain[0]}")
            req_path = RUNNERS_DIR / rid / "requirements.txt"
            if req_path.exists():
                print(f"      Install: pip install -r runners/{rid}/requirements.txt")
            print()

    return 0


def cmd_run(runner_id: str, runner_args: list[str]) -> int:
    runner_dir = RUNNERS_DIR / runner_id
    runner_py  = runner_dir / "runner.py"

    # ── Existence check ───────────────────────────────────────────────────────
    if not runner_dir.exists():
        print(f"Error: runner '{runner_id}' not found in runners/")
        print()
        print("Available runners:")
        for rid in sorted(r.name for r in RUNNERS_DIR.iterdir()
                          if r.is_dir() and r.name not in _BASE_FILES
                          and (r / "runner.py").exists()):
            print(f"  {rid}")
        return 1

    if not runner_py.exists():
        print(f"Error: runners/{runner_id}/runner.py does not exist")
        return 1

    # ── Load and show meta ────────────────────────────────────────────────────
    meta_path = runner_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            deprecated_by = meta.get("deprecated_by")
            if deprecated_by:
                print(f"Warning: '{runner_id}' has been superseded by '{deprecated_by}'.")
                print(f"         Consider using the newer runner instead:")
                print(f"         python run.py --runner {deprecated_by} ...")
                print()
            print(f"Runner:  {meta.get('name', runner_id)}")
            print(f"ID:      {runner_id}")
            print(f"By:      {meta.get('submitted_by', '—')}")
            supersedes_chain = meta.get("supersedes_chain") or []
            if supersedes_chain:
                print(f"Replaces: {supersedes_chain[0]}")
            print()
        except Exception:
            pass

    # ── Delegate to runner.py ─────────────────────────────────────────────────
    cmd = [sys.executable, str(runner_py)] + runner_args
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


_SERVE_DEFAULT_MAX_TOKENS = 2048


def cmd_serve(
    runner_id: str,
    suite_id: str | None,
    model_id: str | None,
    model_path: str | None,
    max_tokens: int,
    max_model_len: int | None,
    port: int,
    host: str,
    workers: int,
    api_key: str | None,
) -> int:
    """
    Launch the OpenAI-compatible inference server for the given runner.

    Model and generation params can come from a suite file (--suite) or be
    specified directly (--model, --max-tokens, --max-model-len).  If both
    --suite and --model are given, --model overrides the suite's model_id.
    Explicit --max-tokens / --max-model-len always override suite values.
    """
    runner_dir = RUNNERS_DIR / runner_id
    runner_py  = runner_dir / "runner.py"

    # ── Validate runner ───────────────────────────────────────────────────────
    if not runner_dir.exists():
        print(f"Error: runner '{runner_id}' not found in runners/")
        return 1
    if not runner_py.exists():
        print(f"Error: runners/{runner_id}/runner.py does not exist")
        return 1

    # ── Build suite dict ──────────────────────────────────────────────────────
    if suite_id:
        suite_path = SUITES_DIR / suite_id / "suite.json"
        if not suite_path.exists():
            print(f"Error: suite '{suite_id}' not found at {suite_path}")
            return 1
        try:
            suite = json.loads(suite_path.read_text())
        except Exception as e:
            print(f"Error reading suite.json: {e}")
            return 1
        # --model overrides the suite's model_id if provided
        effective_model_id = model_id or suite.get("model_id")
        if not effective_model_id:
            print(f"Error: suite '{suite_id}' has no 'model_id' field")
            return 1
        # Explicit flags override suite values
        suite["output_tokens_max"] = max_tokens or suite.get("output_tokens_max",
                                                              _SERVE_DEFAULT_MAX_TOKENS)
        if max_model_len is not None:
            suite["max_model_len"] = max_model_len
    else:
        # No suite — --model is required
        if not model_id:
            print("Error: either --suite or --model is required for --serve")
            return 1
        effective_model_id = model_id
        suite = {
            "model_id":         effective_model_id,
            "output_tokens_max": max_tokens,
        }
        if max_model_len is not None:
            suite["max_model_len"] = max_model_len

    # ── Import runner class ───────────────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("runner_module", str(runner_py))
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"Error importing runners/{runner_id}/runner.py: {e}")
        return 1

    from runners.benchmark_runner import BenchmarkRunner

    runner_class = None
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if (
            cls is not BenchmarkRunner
            and issubclass(cls, BenchmarkRunner)
            and cls.__module__ == "runner_module"
        ):
            runner_class = cls
            break

    if runner_class is None:
        print(f"Error: no BenchmarkRunner subclass found in runners/{runner_id}/runner.py")
        return 1

    # ── Instantiate and configure runner ─────────────────────────────────────
    runner = runner_class()
    # Signal to load_model() that we need the async engine (streaming path)
    runner._current_scenario = "online"

    # ── Resolve model path ────────────────────────────────────────────────────
    effective_model_path = runner._resolve_model_path(effective_model_id, model_path)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Runner:  {runner_class.__name__} ({runner_id})")
    if suite_id:
        print(f"Suite:   {suite_id}")
    print(f"Model:   {effective_model_id}")
    print(f"Path:    {effective_model_path}")
    print(f"Params:  max_tokens={suite['output_tokens_max']}"
          + (f"  max_model_len={suite['max_model_len']}" if suite.get("max_model_len") else ""))
    print()
    try:
        # Serve mode bypasses parse_args(), so load runner config directly here.
        # This populates self._runner_config so load_model() can read named fields
        # and engine_kwargs exactly as it does in benchmark mode.
        _serve_cfg = runner._load_runner_config(suite_id)
        runner._runner_config = _serve_cfg
        _tp = _serve_cfg.get("tensor_parallel_size", 1)
        runner.load_model(effective_model_path, suite, {
            "tensor_parallel_size":   _tp,
            "pipeline_parallel_size": 1,
            "expert_parallel_size":   1,
            "data_parallel_size":     1,
        })
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # ── Start serve ───────────────────────────────────────────────────────────
    from serve.server import start_server
    start_server(
        runner=runner,
        model_id=effective_model_id,
        port=port,
        host=host,
        workers=workers,
        api_key=api_key,
    )
    return 0


# ── Serve argument parser ─────────────────────────────────────────────────────

def _parse_serve_args(runner_args: list[str]) -> argparse.Namespace:
    """Parse serve-specific flags from the runner args list."""
    parser = argparse.ArgumentParser(
        prog="run.py --runner <id>",
        description="AccelMark serve mode",
        add_help=False,
    )
    parser.add_argument("--suite",                default=None,
                        help="Suite ID (e.g. suite_A) — defines model and generation params. "
                             "Optional if --model is given.")
    parser.add_argument("--model",                default=None, dest="model_id",
                        help="HuggingFace model ID or name (required if --suite not given; "
                             "overrides suite model_id if both are given)")
    parser.add_argument("--model-path",           default=None, dest="model_path",
                        help="Local path to model weights (overrides HF download)")
    parser.add_argument("--max-tokens",           type=int, default=_SERVE_DEFAULT_MAX_TOKENS,
                        dest="max_tokens",
                        help=f"Max output tokens per request (default: {_SERVE_DEFAULT_MAX_TOKENS})")
    parser.add_argument("--max-model-len",        type=int, default=None, dest="max_model_len",
                        help="Max model context length — leave unset to let the framework decide")
    parser.add_argument("--port",                 type=int, default=8000,
                        help="HTTP port to listen on (default: 8000)")
    parser.add_argument("--host",                 default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--workers",              type=int, default=4,
                        help="Max concurrent in-flight requests (default: 4)")
    parser.add_argument("--api-key",              default=None, dest="api_key",
                        help="If set, all endpoints require Authorization: Bearer <key>")
    parser.add_argument("--serve",                action="store_true")  # consumed here
    args, unknown = parser.parse_known_args(runner_args)
    if unknown:
        print(f"Warning: unrecognised serve flags ignored: {' '.join(unknown)}")
    return args


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    argv = sys.argv[1:]

    if not argv:
        return cmd_list(type('args', (), {'all': False})())

    if "--list" in argv:
        show_all = "--all" in argv
        return cmd_list(type('args', (), {'all': show_all})())

    if "--runner" not in argv:
        print("Usage:")
        print("  python run.py --list [--all]")
        print("  python run.py --runner <id> --suite <suite> --scenario <scenario> [...]")
        print("  python run.py --runner <id> --serve --suite <suite>  [--model <id>] [--model-path <path>]")
        print("  python run.py --runner <id> --serve --model <model_id>")
        print("  Serve options: --port N  --host H  --workers N  --max-tokens N  --max-model-len N  --api-key K")
        return 1

    runner_idx = argv.index("--runner")
    if runner_idx + 1 >= len(argv):
        print("Error: --runner requires a runner ID")
        print("Run 'python run.py --list' to see available runners.")
        return 1

    runner_id   = argv[runner_idx + 1]
    runner_args = argv[runner_idx + 2:]

    # ── Serve mode ────────────────────────────────────────────────────────────
    if "--serve" in runner_args:
        args = _parse_serve_args(runner_args)
        return cmd_serve(
            runner_id=runner_id,
            suite_id=args.suite,
            model_id=args.model_id,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            port=args.port,
            host=args.host,
            workers=args.workers,
            api_key=args.api_key,
        )

    return cmd_run(runner_id, runner_args)


if __name__ == "__main__":
    sys.exit(main())