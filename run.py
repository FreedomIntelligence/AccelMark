#!/usr/bin/env python3
"""
AccelMark unified runner entry point.

Usage:
    # List all available runners
    python run.py --list

    # Run a benchmark
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario all

    # Run a benchmark on multiple chips
    # Set tensor_parallel_size in configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml
    # or pass --tensor-parallel-size directly (supported by runners that accept it)
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_B --scenario all --tensor-parallel-size 4

    # Serve — using a suite for model + generation params
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --serve

    # Serve — specifying the model directly (no suite required)
    python run.py --runner nvidia_vllm_47f5d58e --model meta-llama/Llama-3.1-8B-Instruct --serve

    # Serve — suite as base, override model and tune params
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --serve \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --max-tokens 4096 --port 8080 --workers 8 --api-key secret

    # All flags after --runner <id> are passed through to the runner unchanged (non-serve mode)
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario offline --output-dir ./my_result
"""

import argparse
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT   = Path(__file__).parent
RUNNERS_DIR = REPO_ROOT / "runners"
SUITES_DIR  = REPO_ROOT / "suites"

# Short human-friendly aliases -> canonical default runner folder IDs.
# Exact runner folder IDs always take precedence over aliases.
_RUNNER_ALIASES: dict[str, str] = {
    "nvidia":           "nvidia_vllm_47f5d58e",
    "vllm":             "nvidia_vllm_47f5d58e",
    "nvidia/vllm":      "nvidia_vllm_47f5d58e",
    "nvidia-vllm":      "nvidia_vllm_47f5d58e",
    "sglang":           "nvidia_sglang_c43a8309",
    "nvidia/sglang":    "nvidia_sglang_c43a8309",
    "nvidia-sglang":    "nvidia_sglang_c43a8309",
    "trtllm":           "nvidia_tensorrt_llm_21323970",
    "tensorrt":         "nvidia_tensorrt_llm_21323970",
    "tensorrt-llm":     "nvidia_tensorrt_llm_21323970",
    "nvidia/trtllm":    "nvidia_tensorrt_llm_21323970",
    "amd":              "amd_vllm_rocm_6c18cd8f",
    "rocm":             "amd_vllm_rocm_6c18cd8f",
    "amd/vllm":         "amd_vllm_rocm_6c18cd8f",
    "ascend":           "ascend_vllm_ascend_d4aa9fda",
    "huawei":           "ascend_vllm_ascend_d4aa9fda",
    "npu":              "ascend_vllm_ascend_d4aa9fda",
    "apple":            "apple_mlx_lm_9546b8b5",
    "mlx":              "apple_mlx_lm_9546b8b5",
    "mac":              "apple_mlx_lm_9546b8b5",
    "tpu":              "google_vllm_tpu_68cc9ffa",
    "google":           "google_vllm_tpu_68cc9ffa",
    "musa":             "moorethreads_vllm_musa_6607740b",
    "moorethreads":     "moorethreads_vllm_musa_6607740b",
    "mtt":              "moorethreads_vllm_musa_6607740b",
    "iluvatar":         "iluvatar_vllm_ixuca_4d3befc9",
    "ixuca":            "iluvatar_vllm_ixuca_4d3befc9",
}

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


def detect_default_runner() -> tuple[str | None, str | None]:
    """Detect local accelerator tooling and return (runner_id, hardware_label)."""
    runners = discover_runners()
    probes = [
        ("nvidia-smi",   "NVIDIA GPU",        "nvidia_vllm_47f5d58e",          "nvidia"),
        ("npu-smi",      "Huawei Ascend NPU", "ascend_vllm_ascend_d4aa9fda",   "ascend"),
        ("rocm-smi",     "AMD ROCm GPU",      "amd_vllm_rocm_6c18cd8f",        "amd"),
        ("mthreads-gmi", "Moore Threads GPU", "moorethreads_vllm_musa_f2f6f965", "moorethreads"),
    ]
    for tool, label, preferred, platform in probes:
        if shutil.which(tool):
            if preferred in runners:
                return preferred, label
            for rid, meta in runners.items():
                if meta.get("platform") == platform and not meta.get("deprecated_by"):
                    return rid, label

    if os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        if "google_vllm_tpu_68cc9ffa" in runners:
            return "google_vllm_tpu_68cc9ffa", "Google TPU"

    if sys.platform == "darwin":
        if "apple_mlx_lm_9546b8b5" in runners:
            return "apple_mlx_lm_9546b8b5", "Apple Silicon / macOS"

    return None, None


def resolve_runner_id(spec: str | None) -> tuple[str | None, str | None]:
    """Resolve a runner ID, 'auto', short alias, or unique prefix.

    Returns (resolved_runner_id, note) where note explains non-literal resolution.
    """
    if not spec:
        return None, None

    runners = discover_runners()
    if spec in runners:
        return spec, None

    key = spec.strip().lower()
    if key == "auto":
        rid, hw = detect_default_runner()
        if rid:
            return rid, f"Auto-detected {hw} -> {rid}"
        return None, "Could not auto-detect local accelerator; pass --runner <id> (see --list)."

    if key in _RUNNER_ALIASES:
        target = _RUNNER_ALIASES[key]
        if target in runners:
            return target, f"Resolved alias '{spec}' -> {target}"

    active = [rid for rid, meta in runners.items() if not meta.get("deprecated_by")]
    prefix_matches = [rid for rid in active if rid.lower().startswith(key)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], f"Resolved prefix '{spec}' -> {prefix_matches[0]}"

    return spec, None


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
            install_sh = RUNNERS_DIR / rid / "install.sh"
            req_path = RUNNERS_DIR / rid / "requirements.txt"
            if install_sh.exists():
                print(f"      Install: bash runners/{rid}/install.sh")
            elif req_path.exists():
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



def cmd_check_env(runner_id: str | None = None) -> int:
    """Fast, zero-download pre-flight check for contributors."""
    import importlib

    print("AccelMark Environment Check")
    print("=" * 40)

    py_ok = sys.version_info >= (3, 10)
    print(f"[{'OK' if py_ok else 'FAIL'}] Python {sys.version.split()[0]} (requires >= 3.10)")

    deps_ok = True
    for mod, pkg in [("jsonschema", "jsonschema"), ("numpy", "numpy"), ("yaml", "pyyaml")]:
        try:
            importlib.import_module(mod)
            print(f"[OK]   {pkg}")
        except ImportError:
            deps_ok = False
            print(f"[MISS] {pkg}  -> pip install {pkg}")

    sub_cfg = REPO_ROOT / "configs" / "submitter.yaml"
    if sub_cfg.exists():
        sub_text = sub_cfg.read_text(encoding="utf-8", errors="ignore")
        if "your_github_username" in sub_text or "your_name_or_github_username" in sub_text:
            print(f"[WARN] {sub_cfg.relative_to(REPO_ROOT)} still has placeholder username -> run 'python run.py --init'")
        else:
            print(f"[OK]   {sub_cfg.relative_to(REPO_ROOT)}")
    else:
        print(f"[WARN] configs/submitter.yaml not found -> run 'python run.py --init'")

    detected = []
    for tool, label in [
        ("nvidia-smi", "NVIDIA GPU"),
        ("npu-smi", "Huawei Ascend NPU"),
        ("rocm-smi", "AMD ROCm GPU"),
        ("mthreads-gmi", "Moore Threads GPU"),
    ]:
        if shutil.which(tool):
            detected.append(label)
    if sys.platform == "darwin":
        detected.append("Apple Silicon / macOS")
    print(f"[INFO] Detected hardware tools: {', '.join(detected) if detected else 'none (CPU / remote)'}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    if hf_token:
        print("[OK]   HF_TOKEN is set (gated Llama-3 suites A-E ready)")
    else:
        print("[INFO] HF_TOKEN not set (OK for Suite F/G/H; needed for gated Llama-3 suites A-E unless using --model-path)")
    if hf_endpoint:
        print(f"[INFO] HF_ENDPOINT={hf_endpoint}")
    else:
        print("[INFO] Tip: if HuggingFace is slow in your region, set HF_ENDPOINT=https://hf-mirror.com")

    auto_runner, _ = detect_default_runner()
    resolved_id = None
    if runner_id:
        resolved_id, note = resolve_runner_id(runner_id)
        if note:
            print(f"[INFO] {note}")

    runner_ok = True
    if resolved_id:
        rdir = RUNNERS_DIR / resolved_id
        if not rdir.is_dir() or not (rdir / "runner.py").exists():
            runner_ok = False
            print(f"[FAIL] Runner '{runner_id}' not found in runners/")
        else:
            print(f"[OK]   Runner folder: runners/{resolved_id}")
            req = rdir / "requirements.txt"
            if req.exists():
                print(f"[INFO] Install runner deps if needed: pip install -r runners/{resolved_id}/requirements.txt")
    elif runner_id:
        runner_ok = False
        print(f"[FAIL] Runner '{runner_id}' could not be resolved (run 'python run.py --list')")

    print("-" * 40)
    if py_ok and deps_ok and runner_ok:
        rec_runner = resolved_id or auto_runner or "nvidia_vllm_47f5d58e"
        print("Ready! Fastest 10-min run to get on the board (default settings welcome):")
        print(f"  python run.py --runner {rec_runner} --suite suite_F")
        return 0
    print("Please resolve the [FAIL]/[MISS] items above before running.")
    return 1


def cmd_init(argv: list[str]) -> int:
    """One-time contributor setup: create configs/submitter.yaml and run --check-env."""
    parser = argparse.ArgumentParser(prog="run.py --init", add_help=True)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--name", default=None, help="GitHub username or display name")
    parser.add_argument(
        "--type",
        dest="submission_type",
        default="individual",
        choices=["individual", "vendor", "research"],
        help="Submission type (default: individual)",
    )
    parser.add_argument("--org", default=None, help="Optional organization name")
    parser.add_argument("--runner", default=None, help="Optional runner ID or alias to check")
    args, _ = parser.parse_known_args(argv)

    sub_cfg = REPO_ROOT / "configs" / "submitter.yaml"
    needs_write = args.name is not None or not sub_cfg.exists()
    if not needs_write and sub_cfg.exists():
        text = sub_cfg.read_text(encoding="utf-8", errors="ignore")
        if "your_github_username" in text or "your_name_or_github_username" in text:
            needs_write = True

    if needs_write:
        name = args.name
        sub_type = args.submission_type
        org = args.org
        if not name and sys.stdin.isatty():
            print("AccelMark One-Time Submitter Setup")
            print("=" * 40)
            try:
                name = input("GitHub username or display name: ").strip()
                raw_type = input("Submission type [individual/vendor/research] (default: individual): ").strip().lower()
                if raw_type in ("individual", "vendor", "research"):
                    sub_type = raw_type
                raw_org = input("Organization (optional, press Enter to skip): ").strip()
                org = raw_org or None
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1
        if name:
            sub_cfg.parent.mkdir(parents=True, exist_ok=True)
            org_yaml = json.dumps(org) if org else "null"
            content = (
                "# Generated by `python run.py --init` (gitignored — never committed).\n"
                f"submitted_by: {json.dumps(name)}\n"
                f"submission_type: {sub_type}\n"
                f"organization: {org_yaml}\n"
                "contact: null\n"
            )
            sub_cfg.write_text(content, encoding="utf-8")
            print(f"[OK]   Wrote {sub_cfg.relative_to(REPO_ROOT)} (submitted_by={name})\n")
        elif not sub_cfg.exists():
            example = REPO_ROOT / "configs" / "submitter.yaml.example"
            if example.exists():
                shutil.copyfile(example, sub_cfg)
                print(f"[INFO] Copied {example.relative_to(REPO_ROOT)} -> {sub_cfg.relative_to(REPO_ROOT)}")
                print("       Pass --name <your_github_username> or edit configs/submitter.yaml.\n")
    else:
        print(f"[OK]   {sub_cfg.relative_to(REPO_ROOT)} already configured.\n")

    return cmd_check_env(args.runner or "auto")


def _find_latest_community_result() -> Path | None:
    community_dir = REPO_ROOT / "results" / "community"
    if not community_dir.is_dir():
        return None
    candidates = [
        p for p in community_dir.iterdir()
        if p.is_dir() and (p / "result.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "result.json").stat().st_mtime)


def cmd_submit(argv: list[str]) -> int:
    """Validate a result, preview its projected leaderboard rank, and prepare PR/upload."""
    parser = argparse.ArgumentParser(prog="run.py --submit", add_help=True)
    parser.add_argument("--submit", nargs="?", const="", default="",
                        help="Submission directory (or result.json path). Omit to use latest in results/community/.")
    parser.add_argument("--zip", dest="make_zip", action="store_true",
                        help="Package result.json + env_info.json into a .zip for GitHub Issue/Web UI upload")
    args, extra = parser.parse_known_args(argv)

    raw_target = args.submit or (extra[0] if extra and not extra[0].startswith("-") else "")
    if raw_target:
        target = Path(raw_target)
        result_dir = target.parent if target.is_file() or target.name == "result.json" else target
    else:
        result_dir = _find_latest_community_result()
        if result_dir is None:
            print("Error: no submission directory specified and none found in results/community/.")
            print("Usage: python run.py --submit results/community/<run_name> [--zip]")
            return 1
        print(f"[INFO] Using latest community result: {result_dir}")

    if not result_dir.is_dir() or not (result_dir / "result.json").exists():
        print(f"Error: result.json not found in {result_dir}")
        return 1

    print("\nStep 1/2: Validating submission...")
    val_cmd = [sys.executable, str(RUNNERS_DIR / "validate_submission.py"), "--dir", str(result_dir)]
    rc = subprocess.call(val_cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        print("\nSubmission validation failed. Fix the errors above and re-run `python run.py --submit`.")
        return rc

    print("\nStep 2/2: Projected leaderboard rank preview...")
    preview_script = REPO_ROOT / "tools" / "pr_rank_preview.py"
    if preview_script.exists():
        subprocess.call([sys.executable, str(preview_script), str(result_dir)], cwd=str(REPO_ROOT))

    rel_dir = result_dir
    try:
        rel_dir = result_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass

    if args.make_zip:
        zip_path = result_dir.parent / f"{result_dir.name}.zip"
        files_to_pack = []
        for name in ("result.json", "env_info.json", "accuracy.json"):
            f = result_dir / name
            if f.exists():
                files_to_pack.append(f)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_pack:
                zf.write(f, arcname=f"{result_dir.name}/{f.name}")
        print(f"\n[OK] Created upload bundle: {zip_path}")
        print("     Attach it at: https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.md")
        return 0

    branch_slug = result_dir.name[:48]
    print("\nNext steps to open your Pull Request:")
    print(f"  git checkout -b submit/{branch_slug}")
    print(f"  git add {rel_dir}/")
    print(f'  git commit -m "results: {result_dir.name}"')
    print(f"  git push -u origin submit/{branch_slug}")
    print("  gh pr create --fill   # or open the PR on GitHub")
    print("\nPrefer not to use git? Re-run with --zip to package for Issue upload:")
    print(f"  python run.py --submit {rel_dir} --zip")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def _print_usage() -> None:
    print("Usage:")
    print("  python run.py --list [--all]")
    print("  python run.py --init [--name <github_user>] [--runner <id|alias|auto>]")
    print("  python run.py --check-env [--runner <id|alias|auto>]")
    print("  python run.py --runner <id|alias|auto> --suite <suite> [--scenario <scenario> ...]")
    print("  python run.py --submit [results/community/<run_name>] [--zip]")
    print("  python run.py --runner <id|alias|auto> --serve --suite <suite> [--model <id>] [--model-path <path>]")
    print("  python run.py --runner <id|alias|auto> --serve --model <model_id>")
    print("  Serve options: --port N  --host H  --workers N  --max-tokens N  --max-model-len N  --api-key K")
    print("\nRunner aliases: auto, vllm, sglang, trtllm, amd/rocm, ascend/npu, apple/mlx, tpu, musa")


def main() -> int:
    argv = sys.argv[1:]

    if not argv:
        return cmd_list(type('args', (), {'all': False})())

    if "--list" in argv:
        show_all = "--all" in argv
        return cmd_list(type('args', (), {'all': show_all})())

    if "--init" in argv:
        return cmd_init(argv)

    if "--submit" in argv:
        return cmd_submit(argv)

    if "--check-env" in argv:
        rid = None
        if "--runner" in argv:
            idx = argv.index("--runner")
            if idx + 1 < len(argv):
                rid = argv[idx + 1]
        return cmd_check_env(rid)

    if "--runner" not in argv:
        if "--suite" in argv:
            auto_id, note = resolve_runner_id("auto")
            if auto_id:
                print(f"[INFO] {note}")
                runner_id = auto_id
                runner_args = argv
            else:
                print(f"Error: {note}")
                _print_usage()
                return 1
        else:
            _print_usage()
            return 1
    else:
        runner_idx = argv.index("--runner")
        if runner_idx + 1 >= len(argv):
            print("Error: --runner requires a runner ID or alias (e.g. auto, vllm, sglang, mlx)")
            print("Run 'python run.py --list' to see available runners.")
            return 1

        raw_runner_id = argv[runner_idx + 1]
        runner_id, note = resolve_runner_id(raw_runner_id)
        if note:
            print(f"[INFO] {note}")
        if not runner_id:
            print(f"Error: could not resolve runner '{raw_runner_id}'.")
            print("Run 'python run.py --list' to see available runners.")
            return 1
        runner_args = argv[:runner_idx] + argv[runner_idx + 2:]

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