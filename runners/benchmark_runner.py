"""
AccelMark Benchmark Runner — Shared base class for all platform scripts.

Platform scripts implement:
    load_model()              — load model weights into accelerator memory
    inference_fn_offline()    — sync batch inference for offline scenario
    release_resources()       — release accelerator memory and distributed groups

Optionally override:
    inference_fn_streaming()  — async single-prompt inference for online/interactive
    get_peak_memory_gb()      — query peak memory usage

Everything else (orchestration, result building, submission, Suite E) is
handled by this base class and shared across all platforms.

Usage:
    class MyRunner(BenchmarkRunner):
        def load_model(self, model_path, suite, tp_size): ...
        def inference_fn_offline(self, prompts): ...
        def release_resources(self): ...

    if __name__ == "__main__":
        MyRunner().main()
"""

import argparse
import asyncio
import gc
import hashlib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# Repo root — runners/benchmark_runner.py is two levels down from root
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── InferenceResult and SampleRecord ─────────────────────────────────────────

sys.path.insert(0, str(_REPO_ROOT))
from loadgen.types import InferenceResult, SampleRecord
from loadgen.loadgen import AccelMarkLoadGen


# ── Base class ────────────────────────────────────────────────────────────────

class BenchmarkRunner(ABC):
    """
    Base class for AccelMark platform scripts.

    Subclasses implement the accelerator-specific methods.
    All orchestration, result building, and submission logic lives here.
    """

    # ── Platform capability flags ─────────────────────────────────────────────
    # Override in subclass if platform has different capabilities

    SUPPORTS_STREAMING: bool = True
    """True if the platform supports streaming (token-by-token) inference.
    Required for accurate TTFT measurement in online/interactive scenarios.
    If False, online scenario is skipped and interactive uses approximated TTFT."""

    SUPPORTS_BATCHING: bool = True
    """True if the platform supports sending multiple prompts at once.
    If False (e.g. mlx-lm), offline scenario runs requests serially."""

    SUPPORTS_ONLINE: bool = True
    """True if the platform supports the online scenario.
    Set to False if the inference API does not support concurrent requests."""

    SUPPORTS_MULTI_CHIP: bool = True
    """True if tensor parallelism is supported.
    If False, --tensor-parallel-size is ignored and always treated as 1."""

    SUPPORTED_PRECISIONS: list[str] = ["bf16", "fp16", "fp32"]
    """
    List of compute precisions this runner supports, in order of preference.
    Use lowercase strings: 'fp32', 'bf16', 'fp16'.

    The first entry is the runner's preferred precision.
    BenchmarkRunner._resolve_precision() selects the best match against
    the suite's allowed_precisions and hardware capability.

    Override in subclass if the runner or hardware has restrictions.

    Examples:
      NVIDIA A100/H100 (full support):
          SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

      NVIDIA V100/T4 (no BF16):
          SUPPORTED_PRECISIONS = ["fp16", "fp32"]

      Apple Silicon M1 (limited BF16):
          SUPPORTED_PRECISIONS = ["fp16", "fp32"]

      Ascend (BF16 only for LLM):
          SUPPORTED_PRECISIONS = ["bf16"]
    """

    SUPPORTED_QUANTIZATIONS: list[str] = []
    """
    List of quantization formats this runner supports for Suite C.
    Use uppercase strings matching suite_C precision_levels:
      'FP8', 'W8A8', 'W8A16', 'W4A16'

    BF16 is always supported and does not need to be listed here.
    An empty list means the runner can run BF16 only — it will be skipped
    for all quantized formats in Suite C.

    Examples:
      NVIDIA vLLM on H100 (full support including FP8):
          SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]

      NVIDIA vLLM on A100 (no native FP8):
          SUPPORTED_QUANTIZATIONS = ["w8a8", "w8a16", "w4a16"]

      AMD ROCm vLLM (FP8 on MI300X only):
          SUPPORTED_QUANTIZATIONS = ["w8a8", "w4a16"]

      Apple MLX (no quantization support yet):
          SUPPORTED_QUANTIZATIONS = []
    """

    # ── Abstract methods (must implement) ─────────────────────────────────────

    @abstractmethod
    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        """
        Load model weights into accelerator memory.

        Args:
            model_path: Local path or HuggingFace model ID
            suite:      Parsed suite.json dict
            tp_size:    Tensor parallel size (number of chips)
        """
        raise NotImplementedError

    @abstractmethod
    def inference_fn_offline(self, prompts: list[str]) -> list[InferenceResult]:
        """
        Synchronous batch inference for offline scenario.
        Send all prompts at once and return results when all complete.

        Args:
            prompts: List of formatted prompt strings

        Returns:
            List of InferenceResult, same length as prompts
        """
        raise NotImplementedError

    @abstractmethod
    def release_resources(self) -> None:
        """
        Release accelerator memory and any distributed process groups.
        Called between scenarios and between Suite E chip counts.
        Must be safe to call multiple times.
        """
        raise NotImplementedError

    # ── Optional methods (override if needed) ─────────────────────────────────

    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        """
        Async single-prompt streaming inference for online/interactive scenarios.
        Override if SUPPORTS_STREAMING = True (default).

        Default implementation raises NotImplementedError.
        If SUPPORTS_STREAMING = False, this method is never called.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} sets SUPPORTS_STREAMING=True "
            f"but does not implement inference_fn_streaming()"
        )

    def get_peak_memory_gb(self) -> Optional[float]:
        """
        Query peak accelerator memory usage in GB.
        Override for platform-specific memory querying.
        Default: returns None (not measured).
        """
        return None

    def format_prompt(self, prompt: str) -> str:
        """
        Apply chat template or other prompt formatting.
        Override if the platform requires specific prompt formatting.
        Default: return prompt unchanged.
        """
        return prompt

    def get_supported_precisions(
        self, chip_name: str, env_info: dict
    ) -> list[str] | None:
        """
        Return the effective compute precisions supported by this runner on the
        given chip, or None to trigger automatic hardware detection.

        Default implementation returns None — the base class will run three-tier
        hardware detection (supports_bf16 field → compute_capability → chip name
        lookup) and intersect the result with SUPPORTED_PRECISIONS.

        Override this method when the runner has framework-specific knowledge
        that hardware detection cannot capture. The runner's answer is trusted
        completely — hardware detection is NOT applied as an override.

        Return values:
            list[str] — explicit precision list, e.g. ["BF16", "FP16", "FP32"]
                        Runner's answer is used directly. Hardware detection skipped.
            None      — runner has no opinion. Auto-detection runs instead.
            []        — runner explicitly supports nothing. Will cause an error
                        unless the suite allows no precisions (unusual).

        Examples:

            # H100 with vLLM FP8 — runner knows this works even though
            # hardware detection doesn't know about FP8
            def get_supported_precisions(self, chip_name, env_info):
                base = super().get_supported_precisions(chip_name, env_info)
                if "h100" in chip_name.lower():
                    return (base or ["BF16", "FP16"]) + ["FP8"]
                return None   # auto-detect for other chips

            # Framework has a BF16 bug on A100 — force FP16 even on capable HW
            def get_supported_precisions(self, chip_name, env_info):
                if "a100" in chip_name.lower():
                    return ["FP16", "FP32"]
                return None   # auto-detect elsewhere

            # Custom V100 patch that makes BF16 work via FP32 accumulation
            def get_supported_precisions(self, chip_name, env_info):
                if "v100" in chip_name.lower():
                    return ["BF16", "FP16", "FP32"]   # override hardware detection
                return None
        """
        return None

    def get_effective_dtype(self) -> Optional[str]:
        """
        Return the actual compute dtype the framework used after model loading.

        Override in subclass to report the real compute dtype, especially when
        it differs from the requested precision. Common cases:

        - FP8 weights on A100 (no native FP8 tensor cores): framework computes
          in bfloat16, storing/loading weights as fp8 → return "bfloat16"
        - FP8 weights on H100 (native FP8): framework computes in fp8
          → return "fp8"
        - W4A16: weights stored as int4, activations and compute in float16
          → return "float16"
        - W8A8: both weights and activations quantized, compute in int8
          → return "int8"

        Default: returns None (not reported). The base class will use
        self._effective_dtype if set, or fall back to None.

        Use lowercase dtype strings matching PyTorch conventions:
        "bfloat16", "float16", "float32", "float8_e4m3fn", "int8"
        """
        return getattr(self, "_effective_dtype", None)

    def get_quantization_method(self) -> Optional[str]:
        """
        Return the quantization method used for weight storage.

        This describes how weights are stored in memory, independent of
        compute dtype. Common values:

        - "fp8"    — weights stored in FP8 format
        - "awq"    — Activation-aware Weight Quantization (4-bit)
        - "gptq"   — GPTQ 4-bit quantization
        - "w8a8"   — 8-bit weights and activations (compressed sparse)
        - "w8a16"  — 8-bit weights, 16-bit activations
        - "bitsandbytes" — bitsandbytes runtime quantization
        - None     — no quantization (BF16/FP16/FP32 full precision)

        Default: returns None. The base class will use
        self._quantization_method if set, or fall back to None.
        """
        return getattr(self, "_quantization_method", None)

    # ── Implementation identity ───────────────────────────────────────────────

    def _compute_implementation_id(self) -> str | None:
        """
        Compute this runner's implementation ID from its runner.py content hash.

        The ID equals the runner's folder name: {platform}_{customname}_{hash8}
        where hash8 is the first 8 hex chars of SHA-256 of runner.py.

        Returns None if the runner cannot be located (e.g. running from an
        unexpected path or from the base class directly).
        """
        try:
            # Get the path of the concrete subclass file (not benchmark_runner.py)
            runner_file = Path(inspect.getfile(self.__class__))

            # The runner must be inside a folder named {platform}_{name}_{hash8}
            folder      = runner_file.parent
            folder_name = folder.name

            # Verify: last segment must be 8 hex chars
            parts = folder_name.rsplit("_", 1)
            if len(parts) != 2 or len(parts[1]) != 8:
                return None
            if not all(c in "0123456789abcdef" for c in parts[1]):
                return None

            # Verify: hash matches runner.py content
            actual_hash = hashlib.sha256(runner_file.read_bytes()).hexdigest()[:8]
            if actual_hash != parts[1]:
                # Hash mismatch — runner.py was modified after the folder was named
                # Return None rather than a wrong ID
                return None

            return folder_name

        except Exception:
            return None

    # ── Entry point ───────────────────────────────────────────────────────────

    def main(self) -> None:
        """Main entry point. Call this from __main__."""
        args = self.parse_args()
        suite = self._load_suite(args.suite)

        # Collect env info early — used for output dir naming and written to task dir
        env_info = self._collect_env_preview()

        # Resolve output dir
        if args.output_dir is None:
            args.output_dir = self._generate_output_dir(args, env_info)
            print(f"Output directory: {args.output_dir}")

        # Create task directory and write env_info.json at task level only.
        # If parent already has env_info.json, this is a subprocess (e.g. Suite E chip-count
        # or Suite C precision subdir) — skip to avoid polluting scenario subdirectories.
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        env_info_path = out_dir / "env_info.json"
        parent_has_env_info = (out_dir.parent / "env_info.json").exists()
        if not env_info_path.exists() and not parent_has_env_info and env_info:
            with open(env_info_path, "w") as f:
                json.dump(env_info, f, indent=2)

        # Dispatch
        # Validate scenario is available for this suite
        self._validate_scenario_for_suite(args.scenario, suite)

        if args.suite == "suite_C" and args.scenario in ("default", "all") and args.precision is None:
            self._run_suite_c(args, suite)
        elif args.suite == "suite_E" and args.scenario in (None, "default", "all"):
            self._run_suite_e(args, suite)
        elif args.scenario in ("default", "all"):
            self._run_all_scenarios(args, suite)
        else:
            self._setup_logging(args.output_dir)
            self._run_single_scenario(args, suite)

    # ── Argument parsing ──────────────────────────────────────────────────────

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments. Override to add platform-specific args."""
        parser = argparse.ArgumentParser(
            description="AccelMark benchmark runner"
        )
        parser.add_argument("--suite", required=True,
            help="Suite ID e.g. suite_A")
        parser.add_argument(
            "--scenario",
            default="default",
            choices=[
                "offline", "online", "interactive",
                "accuracy", "sustained",
                "default", "all",
            ],
            help=(
                "Scenario to run. "
                "'default' runs the suite's standard scenarios (accuracy + offline + "
                "online + interactive where applicable). "
                "'sustained' runs a 30-minute rate-controlled stability test. "
                "'all' runs default + extra scenarios defined in suite.json. "
                "'accuracy' runs the accuracy check only. "
            ),
        )
        parser.add_argument(
            "--precision",
            type=str,
            default=None,
            choices=["BF16", "FP8", "W8A8", "W8A16", "W4A16"],
            help=(
                "Quantization format for Suite C subprocesses. "
                "BF16=full precision baseline, FP8=8-bit float (H100/MI300X), "
                "W8A8=INT8 weights+activations, W8A16=INT8 weights only, "
                "W4A16=INT4 weights only (AWQ). "
                "Auto-set by _run_suite_c() — do not set manually."
            ),
        )
        parser.add_argument("--output-dir", default=None,
            help="Output directory. Auto-generated if not specified.")
        parser.add_argument(
            "--tier",
            default="community",
            choices=["community", "verified"],
            help="Results tier. 'community' (default) for self-submitted results; "
                 "'verified' for maintainer-reproduced results.",
        )
        parser.add_argument("--model-path", default=None,
            help="Override model path. If not set, uses configs/models_local.yaml.")
        parser.add_argument("--tensor-parallel-size", type=int, default=1,
            help="Number of GPUs for tensor parallelism.")
        parser.add_argument("--pipeline-parallel-size", type=int, default=1,
            help="Number of pipeline parallel stages.")
        parser.add_argument("--max-chips", type=int, default=None,
            help="Suite E only: maximum chip count to run.")
        parser.add_argument("--enforce-eager", action="store_true", default=False,
            help="Disable CUDAGraph/compilation. Use if you encounter errors.")
        parser.add_argument("--verbose", action="store_true", default=False,
            help="Show verbose framework logs.")
        parser.add_argument(
            "--skip-accuracy-gate",
            action="store_true",
            default=False,
            dest="skip_accuracy_gate",
            help=(
                "Run benchmark even if accuracy check fails. "
                "Results will be flagged as invalid on the leaderboard. "
                "Useful for debugging or hardware stress testing."
            ),
        )

        args = parser.parse_args()

        # Suite-specific scenario validation happens in main() after suite is loaded.
        # See _validate_scenario_for_suite().

        return args

    # ── Single scenario ───────────────────────────────────────────────────────

    def _run_single_scenario(self, args, suite: dict) -> dict:
        """Run one scenario. Returns the result dict."""

        # Handle accuracy scenario — needs model loaded first
        if args.scenario == "accuracy":
            output_dir = Path(args.output_dir) / "accuracy"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._setup_logging(str(output_dir))

            # Resolve and load model
            model_id = suite.get("model_id") or suite.get("base_model_id", "")
            effective_model_path = self._resolve_model_path(
                model_id, getattr(args, "model_path", None)
            )
            tp_size = getattr(args, "tensor_parallel_size", 1)

            # Load env_info for precision resolution (search up to 2 levels)
            _acc_env_info: dict = {}
            for _c in [output_dir, output_dir.parent, output_dir.parent.parent]:
                _p = _c / "env_info.json"
                if _p.exists():
                    with open(_p) as _f:
                        _acc_env_info = json.load(_f)
                    break

            if getattr(args, "precision", None):
                effective_precision = args.precision.upper()
            else:
                effective_precision = self._resolve_precision(suite, _acc_env_info)
            self._effective_precision = effective_precision

            print(f"Loading {model_id} for accuracy check...")
            t_load = time.perf_counter()
            self._current_scenario = "accuracy"
            self._advance_dist_port()
            self.load_model(effective_model_path, suite, tp_size)
            print(f"Model loaded in {round(time.perf_counter() - t_load, 1)}s")

            try:
                acc = self._run_accuracy_scenario(suite, output_dir)
            finally:
                self.release_resources()

            # Return minimal result dict
            return {"accuracy": acc}

        # ── For all other scenarios: always use a subdirectory ────────────────
        # This ensures single-scenario runs compose correctly with incremental runs.
        # Running --scenario offline then --scenario online produces:
        #   submission_dir/offline/result.json
        #   submission_dir/online/result.json
        #   submission_dir/result.json  (merged, updated each time)
        output_dir = Path(args.output_dir) / args.scenario
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging to scenario subdir
        self._setup_logging(str(output_dir))

        # Load submitter profile
        profile = self._load_submitter_profile()

        # Resolve model path
        model_id = suite.get("model_id") or suite.get("base_model_id", "")
        effective_model_path = self._resolve_model_path(
            model_id, getattr(args, "model_path", None)
        )

        # Read env_info.json from task directory.
        # For standalone runs it's in output_dir; for --scenario all it's in the parent.
        # For Suite C per-format subprocesses it's two levels up (base_dir/w8a8/offline/).
        env_info = {}
        for _candidate in [output_dir, output_dir.parent, output_dir.parent.parent]:
            _p = _candidate / "env_info.json"
            if _p.exists():
                with open(_p) as f:
                    env_info = json.load(f)
                break

        # Load model
        tp_size = getattr(args, "tensor_parallel_size", 1)
        if not self.SUPPORTS_MULTI_CHIP and tp_size > 1:
            print(f"Warning: {self.__class__.__name__} does not support multi-chip. "
                  f"Ignoring --tensor-parallel-size={tp_size}, using 1.")
            tp_size = 1

        print(f"Loading {model_id}...")
        t_load_start = time.perf_counter()
        self._current_scenario = args.scenario
        self._advance_dist_port()

        # Resolve precision — handles BF16→FP16 fallback for older hardware
        # Explicit --precision (Suite C per-format subprocess) takes priority
        if getattr(args, "precision", None):
            effective_precision = args.precision.upper()
        else:
            effective_precision = self._resolve_precision(suite, env_info)
        self._effective_precision = effective_precision

        self.load_model(effective_model_path, suite, tp_size)
        model_load_seconds = round(time.perf_counter() - t_load_start, 1)
        print(f"Model loaded in {model_load_seconds}s")

        # Load requests
        requests = []
        if args.scenario != "training":
            requests_path = _REPO_ROOT / f"suites/{args.suite}/requests.jsonl"
            with open(requests_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        requests.append(json.loads(line))

        # Build loadgen
        chip_count = tp_size * getattr(args, "pipeline_parallel_size", 1)
        loadgen = AccelMarkLoadGen(
            suite=suite,
            requests=requests,
            scenario=args.scenario,
            output_dir=str(output_dir),
            chip_count=chip_count,
        )

        # Select inference function
        if args.scenario == "offline":
            inference_fn = self.inference_fn_offline
        elif args.scenario == "sustained":
            # Rate-controlled time-based run — uses async streaming engine
            if not self.SUPPORTS_STREAMING:
                print(f"Error: sustained scenario requires SUPPORTS_STREAMING = True.")
                sys.exit(1)
            inference_fn = self.inference_fn_streaming
        elif self.SUPPORTS_STREAMING:
            inference_fn = self.inference_fn_streaming
        else:
            # Fallback for platforms without streaming
            def _sync_wrapper(prompt: str) -> InferenceResult:
                results = self.inference_fn_offline([prompt])
                return results[0]
            inference_fn = _sync_wrapper

        # Run benchmark
        benchmark_start = datetime.now(timezone.utc)
        t_bench_start = time.perf_counter()

        if args.scenario == "sustained":
            _loop = getattr(self, "_loop", None)
            if _loop is not None:
                metrics = _loop.run_until_complete(
                    loadgen.run_sustained(
                        inference_fn=inference_fn,
                        sustained_concurrency=suite.get("sustained_concurrency", 8),
                        duration_minutes=suite.get("duration_minutes", 30),
                        sample_interval_seconds=suite.get("sample_interval_seconds", 60),
                        warmup_minutes=suite.get("warmup_minutes", 2.0),
                    )
                )
            else:
                metrics = asyncio.run(
                    loadgen.run_sustained(
                        inference_fn=inference_fn,
                        sustained_concurrency=suite.get("sustained_concurrency", 8),
                        duration_minutes=suite.get("duration_minutes", 30),
                        sample_interval_seconds=suite.get("sample_interval_seconds", 60),
                        warmup_minutes=suite.get("warmup_minutes", 2.0),
                    )
                )
        else:
            metrics = loadgen.run(inference_fn)

        benchmark_end = datetime.now(timezone.utc)
        benchmark_elapsed_minutes = round(
            (time.perf_counter() - t_bench_start) / 60, 1
        )

        # Inject peak memory
        peak_memory = self.get_peak_memory_gb()
        if peak_memory and args.scenario == "offline":
            offline = metrics.get("offline", {})
            for row in (offline.get("results_by_concurrency") or offline.get("results_by_batch_size", [])):
                if not row.get("oom") and row.get("peak_memory_gb") is None:
                    row["peak_memory_gb"] = round(peak_memory, 2)

        # Build result
        driver_version = "unknown"
        accelerators = env_info.get("accelerators", [])
        if accelerators:
            driver_version = accelerators[0].get("driver_version", "unknown")

        # env_info.json always lives at task dir level (never in scenario subdirs)
        if (output_dir.parent / "env_info.json").exists():
            env_info_file = "../env_info.json"
        elif (output_dir.parent.parent / "env_info.json").exists():
            env_info_file = "../../env_info.json"
        else:
            env_info_file = "env_info.json"

        result = self._build_result_json(
            args=args,
            suite=suite,
            metrics=metrics,
            env_info=env_info,
            profile=profile,
            driver_version=driver_version,
            model_load_seconds=model_load_seconds,
            benchmark_start=benchmark_start,
            benchmark_end=benchmark_end,
            benchmark_elapsed_minutes=benchmark_elapsed_minutes,
            env_info_file=env_info_file,
        )

        # Write result.json
        out_path = output_dir / "result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult written to {out_path}")

        # ── Merge into suite-level result.json ────────────────────────────────
        # Find all completed scenario subdirectories and merge them.
        # This updates results/<submission>/result.json incrementally.
        base_dir = Path(args.output_dir)
        known_scenarios = ["offline", "online", "interactive", "sustained"]
        completed = [
            s for s in known_scenarios
            if (base_dir / s / "result.json").exists()
        ]
        if completed:
            self._merge_scenario_results(
                base_dir, suite,
                successful_scenarios=completed,
                total_elapsed_minutes=None,
            )

        return result

    # ── All scenarios ─────────────────────────────────────────────────────────

    def _run_all_scenarios(self, args, suite: dict) -> None:
        """
        Run scenarios based on --scenario flag:
          'default' → runs suite's default scenarios (standard benchmark)
          'all'     → runs default + extra scenarios

        Accuracy always runs FIRST as a gate if it is in the default list.
        """
        import copy

        base_dir = Path(args.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve which scenarios to run ────────────────────────────────────
        default_scenarios, extra_scenarios = self._parse_scenarios_config(suite)

        if args.scenario == "all":
            all_requested = default_scenarios + [
                s for s in extra_scenarios if s not in default_scenarios
            ]
        else:
            # "default"
            all_requested = default_scenarios

        run_accuracy = "accuracy" in all_requested
        skip_gate    = getattr(args, "skip_accuracy_gate", False)

        # Suite C per-format subprocess: precision is explicit, accuracy is non-blocking
        is_suite_c_format = (args.suite == "suite_C" and getattr(args, "precision", None) is not None)

        # Benchmark scenarios: exclude accuracy, training; respect capability flags
        benchmark_scenarios = [
            s for s in all_requested
            if s not in ("accuracy", "training")
            and (s != "online"      or self.SUPPORTS_ONLINE)
            and (s != "interactive" or self.SUPPORTS_STREAMING)
        ]

        results_summary = []
        total_start = time.perf_counter()
        acc_result = None

        print(f"\n{'='*60}")
        print(f"  Suite: {suite.get('suite_id', '?')}")
        print(f"  Accuracy gate: {'yes' if run_accuracy else 'no'}")
        print(f"  Benchmark scenarios: {benchmark_scenarios}")
        print(f"  Base output: {base_dir}")
        print(f"{'='*60}\n")

        # ── Step 1: Accuracy gate ─────────────────────────────────────────
        if run_accuracy:
            # Check if accuracy was already completed in this output directory
            existing_acc_path = base_dir / "accuracy" / "accuracy.json"
            if existing_acc_path.exists():
                try:
                    with open(existing_acc_path) as f:
                        acc_result = json.load(f)
                    print(
                        f"\n  Accuracy already done — loading from {existing_acc_path}\n"
                        f"  Score: {acc_result.get('subset_score')}, "
                        f"valid={acc_result.get('valid')}\n"
                    )
                    results_summary.append(("accuracy", "SKIPPED (already done)", str(base_dir / "accuracy")))
                    run_accuracy = False  # skip the gate block below
                except Exception as e:
                    print(f"  Warning: could not load existing accuracy.json ({e}) — re-running.")

        if run_accuracy:
            print(f"\n{'='*60}")
            if is_suite_c_format:
                print(f"  Step 1: Accuracy Check (data only — non-blocking for Suite C)")
            else:
                print(f"  Step 1: Accuracy Gate")
                print(f"  Must pass before benchmark runs.")
            print(f"{'='*60}\n")

            # Accuracy outputs go to base_dir/accuracy/ (consistent with other scenario subdirs)
            acc_subdir = base_dir / "accuracy"
            acc_subdir.mkdir(parents=True, exist_ok=True)

            # Load model for accuracy check
            # Model stays loaded for first benchmark scenario
            model_id = suite.get("model_id") or suite.get("base_model_id", "")
            effective_model_path = self._resolve_model_path(
                model_id, getattr(args, "model_path", None)
            )
            tp_size = getattr(args, "tensor_parallel_size", 1)

            # Load env_info for precision resolution (search up to 1 level for Suite C)
            _all_env_info: dict = {}
            for _c in [base_dir, base_dir.parent]:
                _p = _c / "env_info.json"
                if _p.exists():
                    with open(_p) as _f:
                        _all_env_info = json.load(_f)
                    break

            # Respect explicit --precision for Suite C per-format subprocesses;
            # otherwise resolve from suite requirements and hardware capability
            if getattr(args, "precision", None):
                effective_precision = args.precision.upper()
            else:
                effective_precision = self._resolve_precision(suite, _all_env_info)
            self._effective_precision = effective_precision

            if getattr(self, "llm", None) is None and getattr(self, "engine", None) is None:
                print(f"Loading model for accuracy check...")
                t_load = time.perf_counter()
                self._current_scenario = "accuracy"
                self._advance_dist_port()
                self.load_model(effective_model_path, suite, tp_size)
                print(f"Model loaded in {round(time.perf_counter() - t_load, 1)}s\n")

            try:
                acc_result = self._run_accuracy_scenario(suite, acc_subdir)

                if not acc_result.get("valid"):
                    results_summary.append(("accuracy", "FAILED: score below threshold", str(base_dir)))
                    print(
                        f"\n{'='*60}\n"
                        f"  ✗ ACCURACY GATE FAILED\n"
                        f"  Score:       {acc_result['subset_score']}\n"
                        f"  Delta:       {acc_result.get('baseline_delta', '?')} "
                        f"(score − baseline; negative means below)\n"
                        f"  Min allowed: -{suite.get('accuracy_threshold_delta', 0.03)}\n"
                        f"\n"
                        f"  Fix model weights before submitting.\n"
                        f"  To run anyway: --skip-accuracy-gate\n"
                        f"{'='*60}\n"
                    )
                    if not skip_gate and not is_suite_c_format:
                        self._release_gpu_memory()
                        return
                    elif is_suite_c_format:
                        print("  Suite C: accuracy is non-blocking — continuing benchmark.\n")
                    else:
                        print("  --skip-accuracy-gate set — continuing anyway.\n")
                else:
                    results_summary.append(("accuracy", "SUCCESS", str(base_dir)))
                    print(
                        f"\n  ✓ Accuracy gate passed: "
                        f"{acc_result['subset_score']} "
                        f"(delta={acc_result.get('baseline_delta', '?')}, "
                        f"valid=True)\n"
                    )

            except Exception as e:
                results_summary.append(("accuracy", f"FAILED: {str(e)[:80]}", ""))
                print(f"\n  ✗ Accuracy check raised an error: {e}")
                if not skip_gate:
                    print("  Aborting benchmark. Use --skip-accuracy-gate to override.")
                    self._release_gpu_memory()
                    return
                else:
                    print("  --skip-accuracy-gate set — continuing anyway.\n")

        # ── Step 2: Benchmark scenarios ───────────────────────────────────
        # Release accuracy model before loading benchmark model
        if run_accuracy:
            self._release_gpu_memory()

        for i, scenario in enumerate(benchmark_scenarios):
            scenario_dir = base_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already completed in a previous run
            existing_result = scenario_dir / "result.json"
            if existing_result.exists():
                print(f"\n  Skipping {scenario} — result.json already exists at {existing_result}")
                results_summary.append((scenario, "SKIPPED (already done)", str(scenario_dir)))
                continue

            print(f"\n{'='*60}")
            print(f"  Step {i + 2 if run_accuracy else i + 1}: {scenario}")
            print(f"{'='*60}\n")

            scenario_args = copy.copy(args)
            scenario_args.scenario = scenario
            scenario_args.output_dir = str(base_dir)

            try:
                self._run_single_scenario(scenario_args, suite)
                results_summary.append((scenario, "SUCCESS", str(scenario_dir)))
                print(f"\n✓ {scenario} completed")
            except Exception as e:
                import traceback
                results_summary.append(
                    (scenario, f"FAILED: {str(e)[:120]}", str(scenario_dir))
                )
                print(f"\n✗ {scenario} failed: {e}")
                traceback.print_exc()
            finally:
                self._release_gpu_memory()
                time.sleep(10)

        total_elapsed = round((time.perf_counter() - total_start) / 60, 1)

        # ── Print summary ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  All scenarios complete ({total_elapsed} min total)")
        print(f"{'='*60}")
        for scenario, status, _ in results_summary:
            if status == "SUCCESS":
                icon = "✓"
            elif status.startswith("SKIPPED"):
                icon = "○"
            else:
                icon = "✗"
            print(f"  [{icon}] {scenario:12s} -- {status}")
        print()

        # ── Merge results ─────────────────────────────────────────────────
        successful_benchmark = [
            s for s, status, _ in results_summary
            if status in ("SUCCESS", "SKIPPED (already done)") and s != "accuracy"
        ]
        failed = [
            s for s, status, _ in results_summary
            if status not in ("SUCCESS", "SKIPPED (already done)")
        ]

        if successful_benchmark:
            self._merge_scenario_results(
                base_dir, suite,
                successful_scenarios=successful_benchmark,
                total_elapsed_minutes=total_elapsed,
            )
            if failed:
                result_path = base_dir / "result.json"
                if result_path.exists():
                    with open(result_path) as f:
                        r = json.load(f)
                    r["meta"]["notes"] = (
                        f"Partial run: {successful_benchmark} succeeded, "
                        f"{failed} failed."
                    )
                    with open(result_path, "w") as f:
                        json.dump(r, f, indent=2)

    # ── Suite C ───────────────────────────────────────────────────────────────

    def _run_suite_c(self, args, suite: dict) -> None:
        """
        Run Suite C: per-format accuracy + offline benchmark for each
        supported quantization format.

        Each format runs as a separate subprocess for clean GPU state.
        Accuracy is NOT a gate — it runs per format and records results
        without blocking. Missing baselines (placeholders) are allowed.

        Format selection:
        - Always includes BF16 (baseline)
        - Other formats: intersection of suite["precision_levels"] and
          runner.SUPPORTED_QUANTIZATIONS
        - Formats the runner doesn't declare are skipped with a warning
        """
        base_dir     = Path(args.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        precision_model_map = suite.get("precision_model_map", {})
        all_precisions      = suite.get("precision_levels", ["BF16"])
        default_scenarios, _ = self._parse_scenarios_config(suite)
        run_accuracy        = "accuracy" in default_scenarios
        platform_script     = sys.argv[0]
        total_start         = time.time()

        # ── Resolve which formats this runner supports ────────────────────────
        runner_quants = [q.upper() for q in self.SUPPORTED_QUANTIZATIONS]
        precisions_to_run = []
        skipped = []
        for p in all_precisions:
            if p == "BF16":
                precisions_to_run.append(p)   # always run BF16
            elif p in runner_quants:
                precisions_to_run.append(p)
            else:
                skipped.append(p)

        print(f"\n{'='*60}")
        print(f"  Suite C — Quantization Efficiency Benchmark")
        print(f"  Formats to run : {precisions_to_run}")
        if skipped:
            print(f"  Skipped        : {skipped} (not in SUPPORTED_QUANTIZATIONS)")
        print(f"  Base output    : {base_dir}")
        print(f"{'='*60}\n")

        results_summary = []

        # ── Run each precision format as a subprocess ─────────────────────────
        for precision in precisions_to_run:
            fmt_info     = precision_model_map.get(precision, {})
            fmt_model_id = fmt_info.get("model_id") or suite.get("base_model_id")
            fmt_revision = fmt_info.get("model_revision")

            precision_dir = base_dir / precision.lower()
            precision_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  Precision: {precision}")
            print(f"  Model    : {fmt_model_id}")
            print(f"{'='*60}\n")

            # Resolve local path for this format's model_id
            fmt_model_path = self._resolve_model_path(
                fmt_model_id,
                getattr(args, "model_path", None) if precision == "BF16" else None,
                # Only pass CLI --model-path override for BF16 (the base model).
                # Quantized formats always use their own checkpoint from config.
            )

            # Build subprocess command — runs "default" scenarios for this format
            # (accuracy + offline per suite_C/suite.json)
            cmd = [
                sys.executable, platform_script,
                "--suite",               args.suite,
                "--scenario",            "default",
                "--output-dir",          str(precision_dir),
                "--tensor-parallel-size", str(getattr(args, "tensor_parallel_size", 1)),
                "--precision",           precision,
                "--model-path",          fmt_model_path,
                "--skip-accuracy-gate",  # Suite C never uses accuracy as a gate
            ]

            print(f"  Command: {' '.join(cmd)}\n")

            try:
                subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
                results_summary.append((precision, "SUCCESS", str(precision_dir)))
                print(f"\n  \u2713 {precision} completed")
            except subprocess.CalledProcessError as e:
                results_summary.append(
                    (precision, f"FAILED: returncode={e.returncode}", str(precision_dir))
                )
                print(f"\n  \u2717 {precision} failed (return code {e.returncode})")

            print("  Waiting 10s before next precision level...")
            time.sleep(10)

        total_elapsed = round((time.time() - total_start) / 60, 1)

        print(f"\n{'='*60}")
        print(f"  Suite C complete ({total_elapsed} min total)")
        print(f"{'='*60}")
        for precision, status, _ in results_summary:
            icon = "\u2713" if status == "SUCCESS" else "\u2717"
            print(f"  [{icon}] {precision:6s} — {status}")
        if skipped:
            for p in skipped:
                print(f"  [—] {p:6s} — skipped (not in SUPPORTED_QUANTIZATIONS)")
        print()

        successful = [p for p, status, _ in results_summary if status == "SUCCESS"]
        if successful:
            self._merge_suite_c_results(
                base_dir, suite, successful, total_elapsed,
                skipped=skipped,
            )

    def _merge_suite_c_results(
        self,
        base_dir: Path,
        suite: dict,
        successful_precisions: list[str],
        total_elapsed_minutes: float,
        skipped: list[str] = None,
    ) -> None:
        """
        Merge per-precision results into one Suite C result.
        Computes quality_efficiency = throughput × accuracy_score for each precision.
        """
        precision_results = {}
        for precision in successful_precisions:
            p = base_dir / precision.lower() / "result.json"
            if p.exists():
                with open(p) as f:
                    precision_results[precision] = json.load(f)

        if not precision_results:
            print("No precision results to merge")
            return

        # Use BF16 as base — or first available if BF16 failed
        base_result = precision_results.get("BF16") or precision_results[
            next(iter(precision_results))
        ]

        # Build quantization results
        quant_results = []
        bf16_best = None

        precision_model_map = suite.get("precision_model_map", {})
        all_precisions = suite.get("precision_levels", ["BF16", "FP8", "W8A8", "W8A16", "W4A16"])

        for precision in all_precisions:
            if precision not in precision_results:
                continue
            r           = precision_results[precision]
            fmt_model_id = precision_model_map.get(precision, {}).get("model_id") \
                           or suite.get("base_model_id")
            rows = (r.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
                    or r.get("metrics", {}).get("offline", {}).get("results_by_batch_size", []))
            valid = [row for row in rows if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            best  = max((row["throughput_tokens_per_sec"] for row in valid), default=None)

            if precision == "BF16":
                bf16_best = best

            speedup = (
                round(best / bf16_best, 3)
                if bf16_best and best and precision != "BF16"
                else (1.0 if precision == "BF16" else None)
            )

            # Per-format accuracy — read from precision subdir accuracy.json
            acc_file = base_dir / precision.lower() / "accuracy" / "accuracy.json"
            acc_score = None
            acc_delta = None
            acc_valid = None
            if acc_file.exists():
                try:
                    with open(acc_file) as f:
                        acc_data  = json.load(f)
                    acc_score = acc_data.get("subset_score")
                    acc_delta = acc_data.get("baseline_delta")
                    acc_valid = acc_data.get("valid")
                except Exception:
                    pass

            quality_eff = round(best * acc_score, 1) if best and acc_score else None

            # effective_dtype and quantization_method from per-format result.json
            fmt_model = r.get("model", {})
            fmt_effective_dtype     = fmt_model.get("effective_dtype")
            fmt_quantization_method = fmt_model.get("quantization_method")

            quant_results.append({
                "precision":                    precision,
                "model_id":                     fmt_model_id,
                "best_throughput_tokens_per_sec": round(best, 2) if best else None,
                "accuracy_score":               acc_score,
                "accuracy_baseline_delta":      acc_delta,
                "accuracy_valid":               acc_valid,
                "quality_efficiency":           quality_eff,
                "speedup_vs_bf16":              speedup,
                "results_by_concurrency":       rows,
                "result_dir":                   precision.lower(),
                "effective_dtype":              fmt_effective_dtype,
                "quantization_method":          fmt_quantization_method,
            })

        merged = {
            "schema_version":    "1.0",
            "suite_id":          "suite_C",
            "implementation_id": base_result.get("implementation_id"),
            "chip":              base_result["chip"],
            "software":          base_result["software"],
            "model": {
                **base_result["model"],
                "model_id": suite.get("base_model_id", base_result["model"]["model_id"]),
                "_note": "base_model_id. Each precision level uses its own quantized checkpoint.",
            },
            "task": {
                "scenarios_run":       ["accuracy", "offline"],
                "precision_levels_run": successful_precisions,
                "precision_levels_skipped": skipped or [],
                "parallelism":         base_result["task"]["parallelism"],
                "num_runs":            suite.get("num_runs", 3),
            },
            "metrics": {
                "quantization": {
                    "results_by_precision": quant_results,
                },
                "derived": {},
            },
            # No top-level accuracy block for Suite C — accuracy is per-format
            # inside metrics.quantization.results_by_precision[i].accuracy_score
            "accuracy": None,
            "meta": {
                **base_result["meta"],
                "benchmark_elapsed_minutes": total_elapsed_minutes,
                "benchmark_elapsed_minutes_note": "Total across all precision levels.",
                "precision_dirs":  {p: p.lower() for p in successful_precisions},
                "precision_model_map": {
                    p: suite["precision_model_map"].get(p, {})
                    for p in successful_precisions
                    if "precision_model_map" in suite
                },
            },
        }

        out_path = base_dir / "result.json"
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)

        # Primary metric: best quality_efficiency across all evaluated formats
        best_qe = max(
            (r["quality_efficiency"] for r in quant_results if r["quality_efficiency"]),
            default=None
        )
        best_thr = max(
            (r["best_throughput_tokens_per_sec"] for r in quant_results
             if r["best_throughput_tokens_per_sec"]),
            default=None
        )
        if best_qe:
            print(f"\n  Best quality efficiency : {best_qe:,.0f}")
        if best_thr:
            print(f"  Best throughput         : {best_thr:,.0f} tok/s")

        # Print summary
        print(f"\n  Quantization efficiency summary:")
        print(f"  {'Precision':>10}  {'Throughput':>14}  {'Accuracy':>10}  {'Speedup':>8}  {'Quality Eff':>12}")
        print(f"  {'-'*60}")
        for r in quant_results:
            thr = f"{r['best_throughput_tokens_per_sec']:,.0f}" if r["best_throughput_tokens_per_sec"] else "—"
            acc = f"{r['accuracy_score']:.3f}" if r["accuracy_score"] else "—"
            spd = f"{r['speedup_vs_bf16']:.2f}×" if r["speedup_vs_bf16"] else "—"
            qe = f"{r['quality_efficiency']:,.0f}" if r["quality_efficiency"] else "—"
            print(f"  {r['precision']:>10}  {thr:>14}  {acc:>10}  {spd:>8}  {qe:>12}")

        print(f"\nSuite C merged result written to {out_path}")

    # ── Suite E ───────────────────────────────────────────────────────────────

    def _run_suite_e(self, args, suite: dict) -> None:
        """
        Run Suite E: accuracy gate first, then offline at multiple chip counts.
        Uses subprocess to avoid distributed process group conflicts.
        """
        base_dir = Path(args.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        all_counts = suite.get("chip_counts_all", [1, 2, 4, 8])
        required_counts = suite.get("chip_counts_required", [1, 2])
        max_chips = getattr(args, "max_chips", None) or 4
        chip_counts = [c for c in all_counts if c <= max_chips]
        skip_gate = getattr(args, "skip_accuracy_gate", False)
        platform_script = sys.argv[0]

        if not chip_counts:
            print(f"Error: --max-chips {max_chips} too low. Min: {min(required_counts)}")
            raise SystemExit(1)

        missing_required = [c for c in required_counts if c not in chip_counts]
        if missing_required:
            print(f"Warning: required counts {missing_required} excluded "
                  f"by --max-chips={max_chips}.")

        results_summary = []
        total_start = time.perf_counter()
        acc_result = None

        print(f"\n{'='*60}")
        print(f"  Suite E — Scaling Efficiency Benchmark")
        print(f"  Chip counts: {chip_counts}")
        print(f"  Base output: {base_dir}")
        print(f"{'='*60}\n")

        # ── Step 1: Accuracy gate ─────────────────────────────────────────
        _default_scenarios_e, _ = self._parse_scenarios_config(suite)
        if "accuracy" in _default_scenarios_e:
            acc_dir = base_dir / "accuracy"
            acc_dir.mkdir(parents=True, exist_ok=True)
            acc_path = acc_dir / "accuracy.json"

            # Skip if already done
            if acc_path.exists():
                try:
                    with open(acc_path) as f:
                        acc_result = json.load(f)
                    print(
                        f"\n  Accuracy already done — loading from {acc_path}\n"
                        f"  Score: {acc_result.get('subset_score')}, "
                        f"valid={acc_result.get('valid')}\n"
                    )
                except Exception as e:
                    print(f"  Warning: could not load existing accuracy.json ({e}) — re-running.")
                    acc_result = None

            if acc_result is None:
                print(f"\n{'='*60}")
                print(f"  Step 1: Accuracy Gate (1× chip)")
                print(f"{'='*60}\n")

                cmd = [
                    sys.executable, platform_script,
                    "--suite", args.suite,
                    "--scenario", "accuracy",
                    "--output-dir", str(acc_dir),
                    "--tensor-parallel-size", "1",
                ]
                if getattr(args, "model_path", None):
                    cmd += ["--model-path", args.model_path]

                try:
                    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
                    if acc_path.exists():
                        with open(acc_path) as f:
                            acc_result = json.load(f)
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Accuracy subprocess failed (return code {e.returncode})")
                    if not skip_gate:
                        print("  Aborting Suite E. Use --skip-accuracy-gate to override.")
                        return
                    print("  --skip-accuracy-gate set — continuing anyway.\n")

            if acc_result and not acc_result.get("valid") and not skip_gate:
                delta = acc_result.get('baseline_delta', '?')
                threshold = suite.get('accuracy_threshold_delta', 0.03)
                print(
                    f"\n  ✗ ACCURACY GATE FAILED\n"
                    f"  Score: {acc_result.get('subset_score')}\n"
                    f"  Delta: {delta} (min allowed: -{threshold})\n"
                    f"  Aborting Suite E. Use --skip-accuracy-gate to override.\n"
                )
                return

            print(f"\n  ✓ Accuracy passed: {acc_result.get('subset_score') if acc_result else '?'}\n")

        # ── Step 2: Run chip counts ───────────────────────────────────────
        for count in chip_counts:
            count_dir = base_dir / f"{count}x"
            count_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  Running {count}× chips")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, platform_script,
                "--suite", args.suite,
                "--scenario", "offline",
                "--output-dir", str(count_dir),
                "--tensor-parallel-size", str(count),
                "--skip-accuracy-gate",   # accuracy already done above
            ]
            if getattr(args, "model_path", None):
                cmd += ["--model-path", args.model_path]
            if getattr(args, "enforce_eager", False):
                cmd += ["--enforce-eager"]

            print(f"  Command: {' '.join(cmd)}\n")

            try:
                subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
                results_summary.append((count, "SUCCESS", str(count_dir)))
                print(f"\n✓ {count}× completed")
            except subprocess.CalledProcessError as e:
                results_summary.append(
                    (count, f"FAILED: returncode={e.returncode}", str(count_dir))
                )
                print(f"\n✗ {count}× failed (return code {e.returncode})")

            print("  Waiting 10s before next chip count...")
            time.sleep(10)

        total_elapsed = round((time.perf_counter() - total_start) / 60, 1)

        # ── Print summary ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Suite E complete ({total_elapsed} min total)")
        print(f"{'='*60}")
        for count, status, _ in results_summary:
            icon = "✓" if status == "SUCCESS" else "✗"
            print(f"  [{icon}] {count}× — {status}")
        print()

        successful = [c for c, status, _ in results_summary if status == "SUCCESS"]
        if not successful:
            print("All chip counts failed — no result.json merged.")
            return

        self._merge_suite_e_results(
            base_dir, suite, successful, total_elapsed, accuracy=acc_result
        )

    # ── Merge results ─────────────────────────────────────────────────────────

    def _merge_scenario_results(
        self,
        base_dir: Path,
        suite: dict,
        successful_scenarios: list[str],
        total_elapsed_minutes: float = 0.0,
    ) -> dict:
        """Merge per-scenario result.json files into one suite-level result."""

        default_scenarios, extra_scenarios = self._parse_scenarios_config(suite)
        all_suite_scenarios = default_scenarios + [
            s for s in extra_scenarios if s not in default_scenarios
        ]
        scenarios_to_merge = [
            s for s in (all_suite_scenarios or ["offline"])
            if s in successful_scenarios and s not in ("training", "accuracy")
        ]

        # Find base result (first successful scenario)
        base_result = None
        for s in scenarios_to_merge:
            p = base_dir / s / "result.json"
            if p.exists():
                with open(p) as f:
                    base_result = json.load(f)
                break

        if not base_result:
            print("No scenario results found to merge")
            return {}

        # Merge metrics + compute total elapsed
        merged_metrics = {"derived": {}}
        scenario_elapsed = 0.0

        for scenario in scenarios_to_merge:
            result_path = base_dir / scenario / "result.json"
            if not result_path.exists():
                print(f"Warning: {scenario}/result.json not found, skipping")
                continue
            with open(result_path) as f:
                r = json.load(f)
            elapsed = r.get("meta", {}).get("benchmark_elapsed_minutes") or 0
            scenario_elapsed += elapsed
            for key in ["offline", "online", "interactive", "sustained", "training"]:
                if r.get("metrics", {}).get(key):
                    merged_metrics[key] = r["metrics"][key]

        total_elapsed = total_elapsed_minutes or round(scenario_elapsed, 1)

        # Load accuracy from accuracy/accuracy.json
        accuracy = None
        acc_file = base_dir / "accuracy" / "accuracy.json"
        if acc_file.exists():
            try:
                with open(acc_file) as f:
                    accuracy = json.load(f)
            except Exception:
                pass

        merged = {
            "schema_version": "1.0",
            "suite_id": base_result["suite_id"],
            "implementation_id": base_result.get("implementation_id"),
            "chip": base_result["chip"],
            "software": base_result["software"],
            "model": base_result["model"],
            "task": {
                "scenarios_run": scenarios_to_merge,
                "parallelism": base_result["task"]["parallelism"],
                "num_runs": suite.get("num_runs", 3),
            },
            "metrics": merged_metrics,
            "accuracy": accuracy or {
                "subset_score": None,
                "baseline_delta": None,
                "valid": False,
                "notes": "Run --scenario accuracy to populate.",
            },
            "meta": {
                **base_result["meta"],
                "submission_type": base_result["meta"]["submission_type"],
                "benchmark_elapsed_minutes": total_elapsed,
                "benchmark_elapsed_minutes_note": (
                    f"Total across {scenarios_to_merge} scenarios."
                ),
                "scenario_dirs": {
                    s: str(base_dir / s)
                    for s in scenarios_to_merge
                    if (base_dir / s).exists()
                },
            },
        }

        out_path = base_dir / "result.json"
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged suite result written to {out_path}")
        return merged

    def _merge_suite_e_results(
        self,
        base_dir: Path,
        suite: dict,
        successful_counts: list[int],
        total_elapsed_minutes: float,
        accuracy: dict | None = None,
    ) -> None:
        """Merge per-chip-count results into one Suite E result with scaling efficiency."""

        count_results = {}
        for count in successful_counts:
            p = base_dir / f"{count}x" / "result.json"
            if p.exists():
                with open(p) as f:
                    count_results[count] = json.load(f)

        if not count_results:
            print("No results to merge")
            return

        # Use the smallest chip count result as base — it has the most complete env info
        base_count = min(count_results.keys())
        base_result = count_results[base_count]

        # Base throughput from 1x (or smallest available)
        base_throughput = None
        rows_base = (base_result.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
                     or base_result.get("metrics", {}).get("offline", {}).get(
                         "results_by_batch_size", []))
        valid_base = [r for r in rows_base if not r.get("oom") and r.get("throughput_tokens_per_sec")]
        if valid_base:
            base_throughput = max(r["throughput_tokens_per_sec"] for r in valid_base)

        # Build scaling results
        scaling_results = []
        for count in sorted(count_results.keys()):
            r = count_results[count]
            rows = (r.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
                    or r.get("metrics", {}).get("offline", {}).get("results_by_batch_size", []))
            valid = [row for row in rows if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            best = max((row["throughput_tokens_per_sec"] for row in valid), default=None)

            efficiency = None
            if base_throughput and best and count > 0:
                efficiency = round(best / (base_throughput * count / base_count), 3)

            # Fix per-chip values in each client_concurrency row
            fixed_rows = []
            for row in rows:
                fixed_row = dict(row)
                thr = row.get("throughput_tokens_per_sec")
                if thr:
                    fixed_row["throughput_tokens_per_sec_per_chip"] = round(thr / count, 2)
                fixed_rows.append(fixed_row)

            scaling_results.append({
                "chip_count": count,
                "best_throughput_tokens_per_sec": round(best, 2) if best else None,
                "throughput_tokens_per_sec_per_chip": round(best / count, 2) if best else None,
                "scaling_efficiency": efficiency,
                "results_by_concurrency": fixed_rows,
                "result_dir": f"{count}x",
            })

        # ── Key fix: read chip/software from subprocess result, not env_preview ──
        merged = {
            "schema_version": "1.0",
            "suite_id": "suite_E",
            "implementation_id": base_result.get("implementation_id"),
            # Read chip info from 1x result — always accurate
            # Override count to show max chips used
            "chip": {
                **base_result["chip"],
                "count": max(successful_counts),
                "_count_note": (
                    "Maximum chip count used in this suite. "
                    "See task.chip_counts_run for all counts tested."
                ),
            },
            # Read software info from 1x result — always accurate
            "software": base_result["software"],
            "model": base_result["model"],
            "task": {
                "scenarios_run": ["offline"],
                "chip_counts_run": sorted(count_results.keys()),
                "parallelism_note": "Each chip_count uses tensor_parallel_size=N",
                "num_runs": suite.get("num_runs", 3),
            },
            "metrics": {
                "scaling": {
                    "base_chip_count": base_count,
                    "base_throughput_tokens_per_sec": (
                        round(base_throughput, 2) if base_throughput else None
                    ),
                    "results_by_chip_count": scaling_results,
                },
                "derived": {},
            },
            "accuracy": accuracy or base_result.get("accuracy", {
                "subset_score": None,
                "baseline_delta": None,
                "valid": False,
                "notes": "Run with --scenario accuracy to populate.",
            }),
            "meta": {
                **base_result["meta"],
                "benchmark_elapsed_minutes": total_elapsed_minutes,
                "benchmark_elapsed_minutes_note": "Total across all chip counts.",
                "chip_count_dirs": {
                    str(c): f"{c}x" for c in sorted(count_results.keys())
                },
            },
        }

        out_path = base_dir / "result.json"
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)

        # Print scaling summary
        print(f"\n  Scaling efficiency summary:")
        print(f"  {'Chips':>6}  {'Throughput':>14}  {'Per chip':>12}  {'Efficiency':>10}")
        print(f"  {'-'*48}")
        for r in scaling_results:
            eff = f"{r['scaling_efficiency']:.3f}" if r["scaling_efficiency"] else "—"
            thr = f"{r['best_throughput_tokens_per_sec']:,.0f}" if r["best_throughput_tokens_per_sec"] else "—"
            per = f"{r['throughput_tokens_per_sec_per_chip']:,.0f}" if r["throughput_tokens_per_sec_per_chip"] else "—"
            print(f"  {r['chip_count']:>6}x  {thr:>14}  {per:>12}  {eff:>10}")

        print(f"\nSuite E merged result written to {out_path}")

    # ── Integrated accuracy ───────────────────────────────────────────────────

    def _load_accuracy_questions(self) -> list[dict]:
        """Load the 100-question accuracy subset."""
        path = _REPO_ROOT / "schema" / "accuracy_subset.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"accuracy_subset.jsonl not found at {path}")
        questions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    def _load_accuracy_baseline(self, model_id: str) -> Optional[float]:
        """Load the BF16 baseline score for a model from accuracy_baselines.json."""
        path = _REPO_ROOT / "schema" / "accuracy_baselines.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                baselines = json.load(f)
            return baselines.get(model_id, {}).get("bf16_baseline_score")
        except Exception:
            return None

    def _load_accuracy_baseline_for_format(
        self, model_id: str, precision: str
    ) -> Optional[float]:
        """
        Load the accuracy baseline for a specific model_id + precision format.
        Used by Suite C where each format has its own checkpoint and baseline.

        Looks up: accuracy_baselines.json[model_id]["{precision_lower}_baseline_score"]

        Returns None if not found (placeholder) — accuracy check still runs
        but delta/valid are None, not a failure.
        """
        path = _REPO_ROOT / "schema" / "accuracy_baselines.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                baselines = json.load(f)
            key = f"{precision.lower()}_baseline_score"
            return baselines.get(model_id, {}).get(key)
        except Exception:
            return None

    def _run_accuracy_scenario(
        self,
        suite: dict,
        output_dir: Path,
    ) -> dict:
        """
        Run accuracy check as a proper scenario.
        Uses inference_fn_offline() — same model, framework, precision as the benchmark.

        Args:
            suite:      Parsed suite.json dict
            output_dir: Where to write accuracy.json

        Returns:
            Accuracy dict with subset_score, baseline_delta, valid fields.
        """
        questions = self._load_accuracy_questions()

        print(f"\n{'='*60}")
        print(f"  Accuracy Check ({len(questions)} questions)")
        print(f"  Framework: {self._get_framework_name()}")
        print(f"  Precision: {getattr(self, '_effective_precision', None) or suite.get('precision_required', 'BF16')}")
        print(f"{'='*60}\n")

        # Format prompts using the same format_prompt() as the benchmark
        # This ensures chat template, system prompt etc. are identical
        prompts = []
        for q in questions:
            raw = (
                f"Question: {q['question']}\n"
                f"A) {q['choices'][0]}\n"
                f"B) {q['choices'][1]}\n"
                f"C) {q['choices'][2]}\n"
                f"D) {q['choices'][3]}\n"
                f"Answer:"
            )
            prompts.append(self.format_prompt(raw))

        # Run through inference_fn_offline — same model, framework, precision
        t_start = time.perf_counter()
        try:
            results = self.inference_fn_offline(prompts)
        except Exception as e:
            raise RuntimeError(f"Accuracy inference failed: {e}") from e
        elapsed = round(time.perf_counter() - t_start, 1)
        print(f"Completed in {elapsed}s")

        # Score answers
        correct = 0
        wrong_examples = []
        scored_outputs = []
        for i, result in enumerate(results):
            text = (result.output_text or "").strip()
            match = re.search(r"\b([ABCD])\b", text.upper())
            predicted = match.group(1) if match else "?"
            expected = questions[i].get("answer", "")
            is_correct = (predicted == expected)
            if is_correct:
                correct += 1
            elif len(wrong_examples) < 3:
                wrong_examples.append(
                    f"  Q: {questions[i]['question'][:65]}\n"
                    f"  Expected: {expected}, Got: {predicted} "
                    f"(raw: '{text[:20]}')"
                )
            scored_outputs.append({
                "question_id": questions[i].get("question_id", i),
                "question": questions[i]["question"],
                "choices": questions[i]["choices"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "raw_output": text[:500],
            })

        score = round(correct / len(questions), 4) if questions else 0.0

        # Compare to baseline — one-sided: score must not drop more than threshold
        # below baseline. Scoring ABOVE baseline is always valid.
        model_id = suite.get("model_id", "")
        baseline_score = self._load_accuracy_baseline(model_id)
        delta = round(score - baseline_score, 4) if baseline_score is not None else None
        threshold = suite.get("accuracy_threshold_delta", 0.03)
        valid = (delta >= -threshold) if delta is not None else True

        # Print results
        print(f"Score: {correct}/{len(questions)} = {score:.4f}")
        if baseline_score is not None:
            sign = "+" if delta >= 0 else ""
            print(f"Baseline: {baseline_score:.4f}")
            print(f"Delta: {sign}{delta:.4f} (min allowed: {-threshold:.4f})")
        print(f"Valid: {valid}")
        if wrong_examples:
            print("Example wrong answers:")
            for ex in wrong_examples:
                print(ex)
        if not valid:
            print(f"WARNING: Score dropped {abs(delta):.4f} below baseline "
                  f"(threshold: {threshold}) — submission will be flagged")

        acc = {
            "subset_score": score,
            "baseline_delta": delta,
            "valid": valid,
            "framework": self._get_framework_name(),
            "precision": getattr(self, "_effective_precision", None) or suite.get("precision_required", "BF16"),
            "notes": (
                f"Integrated accuracy check — used same "
                f"{self._get_framework_name()} instance as benchmark."
            ),
        }

        # Save accuracy.json to submission directory
        acc_path = output_dir / "accuracy.json"
        with open(acc_path, "w") as f:
            json.dump(acc, f, indent=2)
        print(f"Saved to: {acc_path}")

        # Save per-question outputs (gitignored — for local debugging only)
        outputs_path = output_dir / "accuracy_outputs.jsonl"
        with open(outputs_path, "w") as f:
            for row in scored_outputs:
                f.write(json.dumps(row) + "\n")
        print(f"Per-question outputs saved to: {outputs_path}")

        return acc

    def _run_accuracy_scenario_for_format(
        self,
        suite: dict,
        output_dir: Path,
        model_id: str,
        precision: str,
    ) -> dict:
        """
        Run accuracy check for a specific precision format in Suite C.

        Unlike _run_accuracy_scenario(), this:
        - Uses model_id-specific baseline (not suite-level baseline)
        - Uses per-format threshold from suite["accuracy_thresholds"]
        - Never blocks — records valid=False as data, does not abort
        - Records the precision format in the returned dict

        Args:
            suite:      Parsed suite.json dict
            output_dir: Where to write accuracy.json
            model_id:   The specific checkpoint being tested (e.g. AWQ model_id)
            precision:  Format string e.g. "W4A16"

        Returns:
            Accuracy dict with subset_score, baseline_delta, valid, precision fields.
        """
        questions = self._load_accuracy_questions()

        print(f"\n{'='*60}")
        print(f"  Accuracy Check — {precision} ({len(questions)} questions)")
        print(f"  Model: {model_id}")
        print(f"  Framework: {self._get_framework_name()}")
        print(f"{'='*60}\n")

        # Format and run prompts
        prompts = []
        for q in questions:
            raw = (
                f"Question: {q['question']}\n"
                f"A) {q['choices'][0]}\n"
                f"B) {q['choices'][1]}\n"
                f"C) {q['choices'][2]}\n"
                f"D) {q['choices'][3]}\n"
                f"Answer:"
            )
            prompts.append(self.format_prompt(raw))

        t_start = time.perf_counter()
        try:
            results = self.inference_fn_offline(prompts)
        except Exception as e:
            raise RuntimeError(f"Accuracy inference failed: {e}") from e
        elapsed = round(time.perf_counter() - t_start, 1)
        print(f"Completed in {elapsed}s")

        # Score answers
        correct = 0
        wrong_examples = []
        scored_outputs = []
        for i, result in enumerate(results):
            text = (result.output_text or "").strip()
            match = re.search(r"\b([ABCD])\b", text.upper())
            predicted = match.group(1) if match else "?"
            expected  = questions[i].get("answer", "")
            is_correct = (predicted == expected)
            if is_correct:
                correct += 1
            elif len(wrong_examples) < 3:
                wrong_examples.append(
                    f"  Q: {questions[i]['question'][:65]}\n"
                    f"  Expected: {expected}, Got: {predicted} "
                    f"(raw: '{text[:20]}')"
                )
            scored_outputs.append({
                "question_id": questions[i].get("question_id", i),
                "question":    questions[i]["question"],
                "choices":     questions[i]["choices"],
                "expected":    expected,
                "predicted":   predicted,
                "correct":     is_correct,
                "raw_output":  text[:500],
            })

        score = round(correct / len(questions), 4) if questions else 0.0

        # Per-format baseline and threshold
        baseline_score = self._load_accuracy_baseline_for_format(model_id, precision)
        delta          = round(score - baseline_score, 4) if baseline_score is not None else None
        thresholds     = suite.get("accuracy_thresholds", {})
        threshold      = thresholds.get(precision, 0.05)
        valid          = (delta >= -threshold) if delta is not None else None
        # None = baseline not set yet (placeholder) — not a failure

        # Print results
        print(f"Score: {correct}/{len(questions)} = {score:.4f}")
        if baseline_score is not None:
            sign = "+" if delta >= 0 else ""
            print(f"Baseline ({precision}): {baseline_score:.4f}")
            print(f"Delta: {sign}{delta:.4f} (min allowed: {-threshold:.4f})")
            print(f"Valid: {valid}")
        else:
            print("Baseline: not set (placeholder) — score recorded, valid=None")
        if wrong_examples:
            print("Example wrong answers:")
            for ex in wrong_examples:
                print(ex)
        if valid is False:
            print(
                f"WARNING: Score dropped {abs(delta):.4f} below baseline "
                f"(threshold: {threshold}) — will be flagged on leaderboard"
            )

        acc = {
            "subset_score":   score,
            "baseline_delta": delta,
            "valid":          valid,
            "precision":      precision,
            "model_id":       model_id,
            "framework":      self._get_framework_name(),
            "notes":          f"Suite C per-format accuracy check. Threshold: {threshold}",
        }

        # Write accuracy.json
        acc_path = output_dir / "accuracy.json"
        with open(acc_path, "w") as f:
            json.dump(acc, f, indent=2)
        print(f"Saved to: {acc_path}")

        # Write per-question outputs (gitignored)
        outputs_path = output_dir / "accuracy_outputs.jsonl"
        with open(outputs_path, "w") as f:
            for row in scored_outputs:
                f.write(json.dumps(row) + "\n")

        return acc

    # ── GPU memory release ────────────────────────────────────────────────────

    def _release_gpu_memory(self) -> None:
        """Release GPU memory between scenarios. Calls self.release_resources()."""
        print("\nReleasing GPU memory...")

        # Platform-specific cleanup
        self.release_resources()

        # Distributed-state cleanup is handled by self.release_resources().
        # For vLLM, run_vllm.py calls vLLM's cleanup_dist_env_and_memory()
        # which clears cached group references before destroying the process
        # group, avoiding the "Invariant: value was None" error that occurs
        # when destroy_process_group() is called without prior group teardown.

        # Clear PyTorch CUDA cache
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU memory released — free: {free/1024**3:.1f} GB / "
                  f"{total/1024**3:.1f} GB\n")
        except Exception as e:
            print(f"  Warning: could not clear CUDA cache: {e}")

    # ── Distributed port management ──────────────────────────────────────────

    def _advance_dist_port(self) -> None:
        """Increment MASTER_PORT before each distributed init.

        vLLM workers communicate via a Gloo TCP store bound to MASTER_PORT.
        After a run, those sockets linger in OS TIME_WAIT (~60-120 s). If the
        next engine init reuses the same port, new workers can't connect cleanly,
        causing a collective-op desync that hangs for the full 1800 s timeout.
        Incrementing the port each time avoids the collision entirely.
        """
        port = getattr(self, "_dist_port", 29499) + 1
        self._dist_port = port
        os.environ["MASTER_PORT"] = str(port)

    # ── Result building ───────────────────────────────────────────────────────

    def _build_result_json(
        self,
        args,
        suite: dict,
        metrics: dict,
        env_info: dict,
        profile: dict,
        driver_version: str,
        model_load_seconds: float,
        benchmark_start: datetime,
        benchmark_end: datetime,
        benchmark_elapsed_minutes: float,
        env_info_file: str = "env_info.json",
    ) -> dict:
        """Build a complete result.json dict."""

        chip_info = {}
        accelerators = env_info.get("accelerators", [])
        if accelerators:
            a = accelerators[0]
            tp_size_for_chip = getattr(args, "tensor_parallel_size", 1)
            # interconnect_intra_node: N/A for single-chip runs; detected value for multi-chip
            if tp_size_for_chip <= 1:
                intra_node = "N/A"
            else:
                intra_node = env_info.get("intra_node_interconnect") or a.get("interconnect_intra_node")
            chip_info = {
                "name": a.get("name", "Unknown"),
                "vendor": a.get("vendor", suite.get("chip", {}).get("vendor", "Unknown")),
                "count": tp_size_for_chip,
                "memory_gb_per_chip": a.get("memory_gb", None),
                "interconnect_intra_node": intra_node,
                "interconnect_inter_node": a.get("interconnect_inter_node", None),
            }

        tp_size = getattr(args, "tensor_parallel_size", 1)
        pp_size = getattr(args, "pipeline_parallel_size", 1)

        return {
            "schema_version": "1.0",
            "suite_id": suite["suite_id"],
            "implementation_id": self._compute_implementation_id(),
            "chip": chip_info,
            "software": {
                "framework": self._get_framework_name(),
                "framework_version": self._get_framework_version(),
                "driver_version": driver_version,
                "runtime_version": env_info.get("runtime_version", "unknown"),
                "os": env_info.get("os", "unknown"),
                "python_version": env_info.get("python_version", "unknown"),
            },
            "model": {
                "model_id": suite.get("model_id") or suite.get("base_model_id", ""),
                "model_revision": suite.get("model_revision", "unknown"),
                "architecture": "dense",
                "parameter_count_b": self._estimate_param_count(suite.get("model_id") or suite.get("base_model_id", "")),
                "precision": getattr(self, "_effective_precision", None)
                             or getattr(args, "precision", None)
                             or suite.get("precision_required", "BF16"),
                "effective_dtype":     self.get_effective_dtype(),
                "quantization_method": self.get_quantization_method(),
                "model_format": "HuggingFace original",
            },
            "task": {
                "scenario": args.scenario,
                "num_runs": suite.get("num_runs", 3),
                "warmup_runs": suite.get("warmup_runs", 1),
                "parallelism": {
                    "tensor_parallel_size": tp_size,
                    "pipeline_parallel_size": pp_size,
                    "data_parallel_size": 1,
                    "expert_parallel_size": None,
                },
                "extra_config": None,
            },
            "metrics": metrics,
            "accuracy": {
                "subset_score": None,
                "baseline_delta": None,
                "valid": False,
        "notes": "Run --scenario accuracy to check model accuracy.",
            },
            "meta": {
                "submitted_by": profile.get("submitted_by", ""),
                "submission_type": profile.get("submission_type", "individual"),
                "date": date.today().isoformat(),
                "reproduce_script": Path(sys.argv[0]).resolve().relative_to(_REPO_ROOT).as_posix(),
                "env_info_file": env_info_file,
                "log_file": "run.log",
                "samples_file": "samples.jsonl",
                "notes": None,
                "benchmark_start_time": benchmark_start.isoformat(),
                "benchmark_end_time": benchmark_end.isoformat(),
                "benchmark_elapsed_minutes": benchmark_elapsed_minutes,
                "model_load_seconds": model_load_seconds,
            },
        }

    def _get_framework_name(self) -> str:
        """Return the framework name for result.json. Override in subclass."""
        return self.__class__.__name__.replace("Runner", "")

    def _get_framework_version(self) -> str:
        """Return the framework version. Override in subclass."""
        return "unknown"

    def _estimate_param_count(self, model_id: str) -> Optional[float]:
        """Estimate parameter count in billions from model ID."""
        m = re.search(r"(\d+(?:\.\d+)?)b", model_id.lower())
        return float(m.group(1)) if m else None

    # ── Helper utilities ──────────────────────────────────────────────────────

    def _parse_scenarios_config(self, suite: dict) -> tuple[list[str], list[str]]:
        """
        Parse the suite's scenarios config into (default_scenarios, extra_scenarios).

        Handles both legacy flat-array format and new dict format:
          Legacy: "scenarios": ["accuracy", "offline", "online", "interactive"]
          New:    "scenarios": {"default": [...], "extra": [...]}

        Returns (default_scenarios, extra_scenarios).
        """
        config = suite.get("scenarios", {})
        if isinstance(config, list):
            # Legacy format — entire list is treated as default, no extras
            return config, []
        default = config.get("default", [])
        extra   = config.get("extra", [])
        return default, extra

    def _detect_supported_precisions(self, env_info: dict) -> list[str]:
        """
        Three-tier automatic hardware detection for supported precisions.

        Tier 1 — reads acc["supports_bf16"] set by collect_env.py (all platforms)
        Tier 2 — reads acc["compute_capability"] for NVIDIA (backward compat with
                  old env_info.json files that predate supports_bf16)
        Tier 3 — chip name substring lookup for known FP16-only chips

        Result is intersected with SUPPORTED_PRECISIONS so a runner that
        declares SUPPORTED_PRECISIONS = ["fp16"] gets FP16 even on A100.

        Returns a list of uppercase precision strings, e.g. ["BF16","FP16","FP32"]
        or ["FP16","FP32"]. Never returns an empty list — falls back to
        ["BF16","FP16","FP32"] if nothing can be determined.
        """
        hw_supports_bf16 = None
        accelerators     = env_info.get("accelerators", [])

        if accelerators:
            acc = accelerators[0]

            # Tier 1: explicit supports_bf16 field (new collect_env.py format)
            if "supports_bf16" in acc:
                hw_supports_bf16 = bool(acc["supports_bf16"])

            # Tier 2: NVIDIA compute_capability fallback
            if hw_supports_bf16 is None:
                cc = acc.get("compute_capability", "")
                if cc:
                    try:
                        hw_supports_bf16 = float(str(cc)) >= 8.0
                    except (ValueError, TypeError):
                        pass

            # Tier 3: chip name substring lookup
            if hw_supports_bf16 is None:
                _FP16_ONLY = {
                    # NVIDIA
                    "v100", "t4", "p100", "p40", "p4", "k80", "k40",
                    # AMD
                    "mi100", "gfx908",
                    # Apple
                    "m1",
                }
                chip_lower = acc.get("name", "").lower()
                if any(c in chip_lower for c in _FP16_ONLY):
                    hw_supports_bf16 = False

        # Default: unknown hardware → assume BF16 capable
        if hw_supports_bf16 is None:
            hw_supports_bf16 = True

        hw_precisions = (
            ["BF16", "FP16", "FP32"] if hw_supports_bf16
            else ["FP16", "FP32"]
        )

        # Intersect with runner's declared maximum capability
        runner_max = [p.upper() for p in self.SUPPORTED_PRECISIONS]
        result     = [p for p in runner_max if p in hw_precisions]

        # Safety net: never return empty (would always error)
        return result if result else hw_precisions

    def _resolve_precision(self, suite: dict, env_info: dict) -> str:
        """
        Resolve the effective compute precision for this run.

        Priority:
          1. Runner's get_supported_precisions() — if it returns a list, use it
             directly. Hardware detection is NOT applied as an override.
          2. Auto-detection via _detect_supported_precisions() — runs when
             get_supported_precisions() returns None.

        Both results are intersected with suite.allowed_precisions.
        Raises SystemExit if no compatible precision can be found.
        Returns precision as uppercase string: "BF16", "FP16", "FP32", etc.
        """
        required  = suite.get("precision_required", "BF16").upper()
        allowed   = [p.upper() for p in suite.get("allowed_precisions", [required])]
        chip_name = (env_info.get("accelerators") or [{}])[0].get("name", "")

        # ── Step 1: ask the runner ────────────────────────────────────────────
        runner_answer = self.get_supported_precisions(chip_name, env_info)

        if runner_answer is not None:
            # Runner spoke — trust it completely, skip hardware detection
            candidate = [p.upper() for p in runner_answer]
            source    = "runner"
        else:
            # Runner silent — use automatic three-tier hardware detection
            candidate = self._detect_supported_precisions(env_info)
            source    = "auto-detection"

        # ── Step 2: intersect with suite's allowed precisions ─────────────────
        effective = [p for p in candidate if p in allowed]

        # ── Step 3: fail clearly if nothing is compatible ─────────────────────
        if not effective:
            print(
                f"\nError: No compatible precision found.\n"
                f"  Suite requires   : {required}\n"
                f"  Suite allows     : {allowed}\n"
                f"  {source:16s} : {candidate}\n"
                f"  Chip             : {chip_name or 'unknown'}\n"
                f"\nThis suite cannot be run on this hardware with this runner.\n"
                f"Check suite.allowed_precisions or runner.SUPPORTED_PRECISIONS."
            )
            sys.exit(1)

        # ── Step 4: pick best precision ───────────────────────────────────────
        # Prefer the suite's required precision if available.
        # Otherwise take the first item in effective (runner/detection preference order).
        resolved = required if required in effective else effective[0]

        if resolved != required:
            print(
                f"\nWarning: '{required}' not available "
                f"on {chip_name or 'this hardware'} "
                f"(detected via {source}).\n"
                f"  Falling back to '{resolved}'.\n"
                f"  Result will be labeled '{resolved}' on the leaderboard.\n"
            )

        return resolved

    def _validate_scenario_for_suite(self, scenario: str, suite: dict) -> None:
        """
        Validate that the requested scenario is available for this suite.
        Raises SystemExit with a clear message if not.

        Meta-scenarios ("default", "all") are not checked — they expand
        dynamically and are always valid.
        """
        meta_scenarios = {"default", "all"}
        if scenario in meta_scenarios:
            return

        default_scenarios, extra_scenarios = self._parse_scenarios_config(suite)
        available = set(default_scenarios) | set(extra_scenarios)

        if scenario not in available:
            print(
                f"Error: scenario '{scenario}' is not available for "
                f"{suite.get('suite_id', 'this suite')}.\n"
                f"  Default scenarios : {default_scenarios}\n"
                f"  Extra scenarios   : {extra_scenarios}\n"
                f"\nRun with --scenario default to run the standard benchmark,\n"
                f"or --scenario all to include extra scenarios."
            )
            sys.exit(1)

    def _load_suite(self, suite_id: str) -> dict:
        suite_path = _REPO_ROOT / f"suites/{suite_id}/suite.json"
        with open(suite_path) as f:
            return json.load(f)

    def _setup_logging(self, output_dir: str) -> None:
        """Tee stdout/stderr to run.log and capture Python logging (vLLM INFO msgs)."""
        import logging as _logging
        log_path = Path(output_dir) / "run.log"

        class TeeWriter:
            def __init__(self, *writers):
                self.writers = writers
            def write(self, text):
                for w in self.writers:
                    w.write(text)
                    w.flush()
            def flush(self):
                for w in self.writers:
                    w.flush()
            def isatty(self):
                return False
            def fileno(self):
                for w in self.writers:
                    try: return w.fileno()
                    except: continue
                raise OSError("no fileno")

        log_file = open(log_path, "w", buffering=1)
        sys.stdout = TeeWriter(sys.__stdout__, log_file)
        sys.stderr = TeeWriter(sys.__stderr__, log_file)

        # Also capture Python logging output (vLLM INFO/WARNING messages go through
        # logging module, not sys.stderr, so the tee above doesn't catch them).
        file_handler = _logging.FileHandler(log_path)
        file_handler.setLevel(_logging.DEBUG)
        file_handler.setFormatter(_logging.Formatter("%(message)s"))
        _logging.getLogger().addHandler(file_handler)

        print(f"Logging to {log_path}")


    def _load_submitter_profile(self) -> dict:
        profile_path = _REPO_ROOT / "configs" / "submitter.yaml"
        if not profile_path.exists():
            return {"submitted_by": "", "submission_type": "individual"}
        try:
            import yaml
            with open(profile_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {"submitted_by": "", "submission_type": "individual"}

    def _resolve_model_path(self, model_id: str, cli_override: Optional[str]) -> str:
        if cli_override:
            return cli_override
        for config_file in ["configs/models_local.yaml", "configs/models.yaml"]:
            config_path = _REPO_ROOT / config_file
            if not config_path.exists():
                continue
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                local_path = config.get("models", {}).get(model_id, {}).get("local_path")
                if local_path and Path(local_path).exists():
                    print(f"Using local model path: {local_path}")
                    return local_path
            except Exception:
                continue
        print(f"Using HuggingFace model: {model_id}")
        return model_id

    def _generate_output_dir(self, args, env_info: dict) -> str:
        # Chip: full name, lowercase, all non-alphanumeric stripped
        accelerators = env_info.get("accelerators", [])
        chip_full = accelerators[0].get("name", "unknown") if accelerators else "unknown"
        chip_clean = re.sub(r"[^a-z0-9_]", "", re.sub(r" +", "_", chip_full.lower())) or "gpu"
        count = getattr(args, "tensor_parallel_size", 1)
        chip_part = f"{chip_clean}x{count}"

        # Suite: keep canonical form (e.g. "suite_A")
        suite_part = args.suite

        # Runner ID: prefer content-hash ID; fall back to platform + random hex with a warning
        impl_id = self._compute_implementation_id()
        if impl_id is None:
            rand_id = hashlib.sha256(os.urandom(4)).hexdigest()[:8]
            runner_part = f"unknown_{rand_id}"
            print(
                f"Warning: runner has no valid implementation_id. "
                f"Each runner should have a content-hash ID (see runners/hash_runner.py). "
                f"Using temporary ID '{runner_part}' for this run."
            )
        else:
            runner_part = impl_id

        dir_name = f"{chip_part}_{suite_part}_{runner_part}"
        tier = getattr(args, "tier", "community")
        return str(Path("results") / tier / dir_name)

    def _collect_env_preview(self) -> dict:
        """Quickly collect env info for output dir naming. Returns empty dict on failure."""
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp = f.name
            subprocess.run(
                ["python", "runners/collect_env.py", "--output", tmp],
                check=False, capture_output=True, cwd=str(_REPO_ROOT)
            )
            with open(tmp) as f:
                return json.load(f)
        except Exception:
            return {}
