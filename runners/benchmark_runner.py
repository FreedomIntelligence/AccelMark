"""
AccelMark Benchmark Runner — Base class for all platform scripts.

Platform scripts implement:
    load_model()              — load model weights into accelerator memory
    inference_fn_offline()    — sync batch inference for offline scenario
    release_resources()       — release accelerator memory and distributed groups

Optionally override:
    inference_fn_streaming()  — async single-prompt inference for online/interactive
    get_peak_memory_gb()      — query peak memory usage

Suite-specific orchestration (Suite C quantization, Suite E scaling) lives in
each suite's own suites/{suite_id}/suite.py. BenchmarkRunner.main() loads
these dynamically when present.

Everything else (generic scenario dispatch, result building, accuracy,
precision resolution, submission) is handled by this base class.

Usage:
    class MyRunner(BenchmarkRunner):
        def load_model(self, model_path, suite, parallelism): ...
        def inference_fn_offline(self, requests): ...
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


# ── InferenceResult, InferenceRequest, SampleRecord ──────────────────────────

sys.path.insert(0, str(_REPO_ROOT))
from loadgen.types import InferenceResult, SampleRecord
from loadgen.loadgen import AccelMarkLoadGen
from dataclasses import dataclass, field as dataclass_field


@dataclass
class InferenceRequest:
    """
    A single inference request passed to inference_fn_offline / inference_fn_streaming.

    Runners read request.prompt at minimum. All other fields are optional —
    runners that don't use them simply ignore them. This allows future fields
    to be added without breaking existing runners.

    Fields:
        prompt:      Formatted prompt string (chat template already applied).
        request_id:  Integer ID matching the original requests.jsonl line.
        input_tokens: Approximate input token count (from requests.jsonl).
        max_tokens:  Per-request max output tokens. None = use suite default.
        temperature: Sampling temperature. 0.0 = greedy (default).
        extra:       Arbitrary extra fields for platform-specific use.
                     Runners may store anything here without schema impact.
    """
    prompt:       str
    request_id:   int            = 0
    input_tokens: int            = 0
    max_tokens:   Optional[int]  = None
    temperature:  float          = 0.0
    extra:        dict           = dataclass_field(default_factory=dict)


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
    Set False if no tensor parallelism — tensor_parallel_size from runner config
    and CLI is ignored; runner always uses 1 chip"""

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

    SUPPORTED_QUANTIZATION_BACKENDS: list[str] = []
    """
    List of quantization backend strings this runner's framework supports.
    Values are the framework-level quantization identifiers passed to the engine
    (e.g. vLLM's `quantization=` kwarg), NOT suite precision names.

    Suite C uses this to filter precision_model_map entries: a format is runnable
    if its engine_kwargs.quantization value appears in this list. This means adding
    a new quantized format to suite.json requires no runner code changes — only the
    suite's precision_model_map entry needs the correct engine_kwargs.quantization.

    An empty list means the runner supports no quantized formats (BF16/FP16/FP32 only).

    Examples:
      NVIDIA vLLM (full support):
          SUPPORTED_QUANTIZATION_BACKENDS = ["fp8", "compressed-tensors", "gptq_marlin"]

      AMD ROCm vLLM (no FP8 on base MI250):
          SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors", "gptq_marlin"]

      Ascend vllm-ascend:
          SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors", "gptq_marlin"]

      Apple MLX (no quantization support yet):
          SUPPORTED_QUANTIZATION_BACKENDS = []
    """

    # ── Abstract methods (must implement) ─────────────────────────────────────

    @abstractmethod
    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model weights into accelerator memory.

        Args:
            model_path:   Local path or HuggingFace model ID (already resolved
                          by _resolve_model_path()).
            parallelism:  Resolved engine configuration dict. Contains:
                              "tensor_parallel_size":   int   (default 1)
                              "pipeline_parallel_size":  int   (default 1)
                              "expert_parallel_size":   int   (default 1)
                              "data_parallel_size":     int   (default 1)
                              "max_tokens":             int   — max generation tokens
                              "max_model_len":          int|None — context window limit
                              "use_async":              bool  — True when the scenario
                                                         requires async streaming
                                                         (online/interactive/sustained);
                                                         False for offline/accuracy.
                          Read only the keys you need. New keys may be added
                          in future versions — ignore unknown keys.

        Example:
            def load_model(self, model_path, parallelism):
                tp = parallelism["tensor_parallel_size"]
                if parallelism["use_async"]:
                    self.engine = AsyncEngine(model=model_path, tp=tp)
                else:
                    self.llm = LLM(model=model_path, tp=tp)

        Example:
            def load_model(self, model_path, suite, parallelism):
                tp = parallelism["tensor_parallel_size"]
                self.llm = LLM(model=model_path, tensor_parallel_size=tp)
        """
        raise NotImplementedError

    @abstractmethod
    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """
        Synchronous batch inference for offline scenario.
        Send all requests at once and return results when all complete.

        Args:
            requests: List of InferenceRequest objects. Read request.prompt
                      at minimum. Other fields (max_tokens, temperature, etc.)
                      are optional — ignore any fields you don't need.

        Returns:
            List of InferenceResult, same length and order as requests.

        Example:
            def inference_fn_offline(self, requests):
                prompts = [r.prompt for r in requests]
                outputs = self.llm.generate(prompts)
                return [InferenceResult(...) for o in outputs]
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

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """
        Async single-request streaming inference for online/interactive scenarios.
        Override if SUPPORTS_STREAMING = True (default).

        Args:
            request: InferenceRequest object. Read request.prompt at minimum.
                     Other fields are optional — ignore any you don't need.

        Returns:
            InferenceResult with first_token_time_ms set for TTFT measurement.

        Default implementation raises NotImplementedError.
        If SUPPORTS_STREAMING = False, this method is never called.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} sets SUPPORTS_STREAMING=True "
            f"but does not implement inference_fn_streaming()"
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """
        Async generator yielding decoded text deltas for the serve layer.
        Override to enable true progressive SSE streaming in `run.py --serve`.

        This is separate from inference_fn_streaming() — they serve different
        purposes:
          inference_fn_streaming()   → benchmark use, returns complete InferenceResult
                                       with timing metrics (TTFT, total_time_ms)
          inference_fn_token_stream() → serve use, yields text deltas as they arrive
                                        for progressive HTTP/SSE delivery to clients

        If not overridden, the serve layer falls back to inference_fn_streaming()
        (single-chunk response — correct but no progressive streaming).

        Args:
            request: InferenceRequest object. Read request.prompt at minimum.
                     Use request.max_tokens for per-request length control if
                     your framework supports it.

        Yields:
            str — decoded text delta since last yield (NOT cumulative).
                  e.g. yields " hello", " world", "!" separately.
                  Do NOT yield the full accumulated string each time.

        Example (vLLM-style cumulative output):
            async def inference_fn_token_stream(self, request):
                prev_len = 0
                async for output in self.engine.generate(request.prompt, ...):
                    delta = output.outputs[0].text[prev_len:]
                    if delta:
                        yield delta
                        prev_len = len(output.outputs[0].text)
        """
        # Default: not implemented — serve layer falls back to inference_fn_streaming.
        # The `if False: yield` makes Python treat this as an async generator
        # so the serve layer's `async for` loop gets a clean StopAsyncIteration
        # rather than a TypeError when the NotImplementedError is raised.
        raise NotImplementedError
        if False:
            yield  # noqa: makes this an async generator function

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

    def get_model_format(self) -> str:
        """
        Return the weight format of the loaded model.

        Override in subclass when loading non-HuggingFace formats:

          class AscendRunner(BenchmarkRunner):
              def load_model(self, ...):
                  self._model_format = "MindIR"
                  ...

        Common values:
          "HuggingFace original" — standard .safetensors / .bin weights
          "TensorRT engine"      — compiled TensorRT .engine file
          "MindIR"               — Ascend MindIR graph format
          "MLX"                  — Apple MLX format
          "GGUF"                 — llama.cpp GGUF format
          "ONNX"                 — ONNX model

        Default: "HuggingFace original". Override or set self._model_format
        in load_model() if your runner uses a different format.
        """
        return getattr(self, "_model_format", "HuggingFace original")

    def get_extra_subprocess_args(self, args) -> list[str]:
        """
        Return extra CLI args to pass when spawning scenario subprocesses.

        Override in subclass to forward runner-specific flags to each scenario
        subprocess spawned by _run_all_scenarios(). The base class subprocess
        command includes only: --suite, --scenario, --output-dir, --tier,
        --skip-accuracy-gate, --model-path (if set), --precision (if set).

        All runner-specific flags (parallelism, enforce_eager, etc.) MUST be
        forwarded here. The base class no longer handles any of them.

        Note: self._runner_config is NOT forwarded via subprocess args — each
        subprocess reads the config yaml file independently on startup, which
        is correct. Only flags that override config values need to be forwarded
        (e.g. --tensor-parallel-size set explicitly on the CLI).

        Example:
            def get_extra_subprocess_args(self, args):
                extra = [
                    "--tensor-parallel-size",
                    str(self._parallelism.get("tensor_parallel_size", 1)),
                ]
                if self._enforce_eager:
                    extra += ["--enforce-eager"]
                return extra
        """
        return []

    def _run_subprocess(self, cmd: list, label: str) -> bool:
        """
        Run a scenario subprocess and report failures clearly.

        Uses inherited fds so tqdm progress bars render correctly on the
        terminal. On non-zero exit, prints the signal name (if the process
        was killed by a signal) so the cause is always visible.

        Args:
            cmd:   Full command list to pass to subprocess.run().
            label: Human-readable name for error messages (e.g. "offline",
                   "bf16", "2x").

        Returns:
            True on success (returncode == 0), False otherwise.
        """
        import signal as _sig
        import subprocess as _sp

        proc = _sp.run(cmd, cwd=str(_REPO_ROOT))

        if proc.returncode == 0:
            return True

        rc = proc.returncode
        if rc < 0:
            try:
                sig_name = _sig.Signals(-rc).name
            except ValueError:
                sig_name = f"signal {-rc}"
            rc_desc = f"killed by {sig_name} (return code {rc})"
        else:
            rc_desc = f"exited with code {rc}"

        print(f"\n  {label} FAILED -- subprocess {rc_desc}")
        return False


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

    def _compute_run_id(self, args, suite: dict, env_info: dict) -> str:
        """
        Compute 8-char hex hash identifying this hardware+software+suite+submitter config.
        Deterministic — same config always produces same run_id.
        Used for duplicate detection in CI and leaderboard.

        Hash key includes:
          - chip name, memory, count, interconnect  (hardware identity)
          - runner_id, framework_version            (software identity)
          - suite_id, model_id, precision           (benchmark identity)
          - submitted_by                            (submitter identity)

        Suite C: uses suite.precision_required ("BF16") — one run_id for all quantized formats.
        Suite E: chip_count is set to max chips tested by the suite plugin via self._chip_count.
        Precision: uses effective runtime precision (e.g. FP16 on V100), not suite.precision_required.
          This ensures the hash matches what check_run_id_integrity() recomputes from model.precision.
        """
        accel   = env_info.get("accelerators", [{}])[0]
        profile = self._load_submitter_profile()

        chip_count = getattr(self, "_chip_count", 1)

        # Interconnect is only meaningful for multi-chip runs
        interconnect = None
        if chip_count > 1:
            interconnect = (env_info.get("intra_node_interconnect")
                            or accel.get("interconnect_intra_node"))

        # Use effective precision if already resolved (e.g. FP16 fallback on V100),
        # otherwise fall back to suite.precision_required. This must match what
        # _build_result_json() writes to model.precision, which is what
        # check_run_id_integrity() reads back during validation.
        effective_precision = (
            getattr(self, "_effective_precision", None)
            or suite.get("precision_required", "BF16")
        )

        key = {
            # Hardware
            "chip_name":      accel.get("name", "unknown"),
            "chip_memory_gb": round(float(accel.get("memory_gb", 0))),
            "chip_count":     chip_count,
            "interconnect":   interconnect,
            # Software
            "runner_id":         self._compute_implementation_id() or "unknown",
            "framework_version": self._get_framework_version(),
            # Benchmark
            "suite_id":  suite["suite_id"],
            "model_id":  suite.get("model_id", "unknown"),
            "precision": effective_precision,
            # Submitter
            "submitted_by": profile.get("submitted_by", "unknown"),
        }
        

        raw = json.dumps(key, sort_keys=True)
        run_id = hashlib.sha256(raw.encode()).hexdigest()[:8]
        print(f"Generate new run_id from {key}\nGenerated run_id: {run_id}")
        return run_id

    def _compute_run_name(self, args, suite: dict, env_info: dict) -> str:
        """
        Build human-readable directory name for this benchmark run.
        Format: {chip_slug}x{count}_{suite_id}_{runner_id}_{run_id}
        Example: nvidia_a100_sxm4_80gbx1_suite_A_nvidia_vllm_47f5d58e_a3f2c1b8

        Stored in meta.run_name and used as the output directory name.
        Deterministic — same config always produces same run_name.
        """
        accelerators = env_info.get("accelerators", [])
        chip_full    = accelerators[0].get("name", "unknown") if accelerators else "unknown"
        chip_slug    = re.sub(r"[^a-z0-9_]", "", re.sub(r"[ /\-]+", "_", chip_full.lower())) or "gpu"

        chip_count = getattr(self, "_chip_count", 1)
        chip_part  = f"{chip_slug}x{chip_count}"

        suite_id  = suite["suite_id"]
        runner_id = self._compute_implementation_id() or "unknown"
        run_id    = self._compute_run_id(args, suite, env_info)

        return f"{chip_part}_{suite_id}_{runner_id}_{run_id}"

    # ── Entry point ───────────────────────────────────────────────────────────

    def main(self) -> None:
        """Main entry point. Call this from __main__."""
        args = self.parse_args()
        suite = self._load_suite(args.suite)

        # Collect env info early — used for output dir naming and written to task dir
        env_info = self._collect_env_preview()

        # Resolve precision before generating the output dir so that the folder
        # name and run_id use the actual runtime precision (e.g. FP16 on V100)
        # rather than suite.precision_required. This keeps the directory name,
        # run_id, and result.json model.precision all consistent.
        if getattr(args, "precision", None):
            self._effective_precision = args.precision.upper()
        else:
            self._effective_precision = self._resolve_precision(suite, env_info)

        # Resolve _chip_count before generating the output dir so the folder
        # name reflects the correct chip count for multi-chip suites.
        # Guarded by args.output_dir is None — only runs for fresh top-level
        # invocations where the folder name is auto-generated. Subprocesses
        # and resume runs always have --output-dir set explicitly, so this
        # block is safely skipped for them. Suite plugins (e.g. suite_E)
        # also set br._chip_count authoritatively inside their orchestrator
        # for run_id/run_name correctness on resume.
        #
        # Priority ladder:
        #   1. chip_counts_all (Suite E style) — max count from suite list
        #      that fits within available hardware (or --max-chips override)
        #   2. required_chips == "auto" (Suite B style) — all available GPUs
        #   3. required_chips == N (int) — exactly N
        #   4. default — 1 (getattr fallback in _compute_run_id)
        if args.output_dir is None:
            _suite_chip_counts_all = suite.get("chip_counts_all")
            _suite_required_chips  = suite.get("required_chips")
            _explicit_max_chips    = getattr(args, "max_chips", None)

            if _suite_chip_counts_all:
                _detected = len(env_info.get("accelerators", []))
                _hw_max   = _explicit_max_chips or (_detected if _detected > 0 else max(_suite_chip_counts_all))
                self._chip_count = max(c for c in _suite_chip_counts_all if c <= _hw_max)
            elif _suite_required_chips == "auto":
                self._chip_count = self._get_chip_count()
            elif isinstance(_suite_required_chips, int) and _suite_required_chips > 0:
                self._chip_count = _suite_required_chips

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

        # ── Suite plugin dispatch ──────────────────────────────────────────
        # If suites/{suite_id}/suite.py exists, delegate to it.
        # The plugin receives the full runner instance and calls base class
        # methods as needed (e.g. br._merge_scenario_results()).
        # Suites without a suite.py (A, B, D, F, stress) use the generic path.
        suite_script = _REPO_ROOT / "suites" / args.suite / "suite.py"
        if suite_script.exists():
            import importlib.util
            spec   = importlib.util.spec_from_file_location(
                         f"accelmark_suite_{args.suite}", suite_script
                     )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.run(self, args, suite, env_info)
        elif args.scenario in ("default", "all"):
            self._run_all_scenarios(args, suite)
        else:
            self._setup_logging(args.output_dir)
            self._run_single_scenario(args, suite)

    # ── Argument parsing ──────────────────────────────────────────────────────

    # ── Suite / parallelism helpers ───────────────────────────────────────────

    def _get_chip_count(self) -> int:
        """
        Return the number of available accelerator chips for this platform.

        Override in each runner to use the platform-appropriate API
        (torch.cuda.device_count() for NVIDIA/AMD, torch_npu for Ascend, etc.).
        The base implementation returns 1 as a safe default.
        """
        return 1

    def get_suite_required_chips(self, suite_id: str | None) -> int | str | None:
        """
        Return the 'required_chips' field from a suite's suite.json.

          "auto"  — suite wants all available accelerators (e.g. suite_B)
          int     — suite wants exactly N chips
          None    — field absent or suite_id unknown (caller defaults to 1)

        Runners must not call this directly — use _resolve_tensor_parallel_size()
        which applies the full priority ladder.
        """
        if not suite_id:
            return None
        try:
            suite = self._load_suite(suite_id)
            return suite.get("required_chips")
        except Exception:
            return None

    def _resolve_tensor_parallel_size(self, cli_tp: int | None) -> tuple[int, str]:
        """
        Resolve tensor parallel size using the standard priority ladder:
          CLI flag > yaml config > suite.json required_chips > default 1

        Args:
            cli_tp: value of --tensor-parallel-size from the runner's argparse,
                    or None if the flag was not provided.

        Returns:
            (tp_size, source_str) where source_str is a human-readable label
            for the print line shown at startup.
        """
        cfg = getattr(self, "_runner_config", {})
        _cfg_tp = cfg.get("tensor_parallel_size")
        _suite_chips = self.get_suite_required_chips(getattr(self, "_suite_id", None))

        if cli_tp is not None:
            return cli_tp, "cli"
        elif _cfg_tp is not None:
            return _cfg_tp, "config"
        elif _suite_chips == "auto":
            return self._get_chip_count(), "auto-detected"
        elif isinstance(_suite_chips, int) and _suite_chips > 0:
            return _suite_chips, f"suite.json required_chips={_suite_chips}"
        else:
            return 1, "default"

    def format_prompt(self, prompt: str) -> str:
        """
        Apply the tokenizer's chat template to a raw prompt string.

        Returns the formatted string if the tokenizer has a chat template,
        otherwise returns the prompt unchanged. Runners that store their
        tokenizer as self.tokenizer get this for free; runners without a
        tokenizer attribute also return the prompt unchanged.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _format_prompt(self, prompt: str) -> str:
        """Alias for format_prompt (kept for internal use)."""
        return self.format_prompt(prompt)

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
            help=(
                "Scenario to run. "
                "Standard values: offline, online, interactive, accuracy, sustained. "
                "'default' runs the suite's standard scenarios. "
                "'all' runs default + extra scenarios defined in suite.json. "
                "Valid scenarios for a suite are defined in its suite.json."
            ),
        )
        parser.add_argument(
            "--precision",
            type=str,
            default=None,
            help=(
                "Quantization format for Suite C subprocesses. "
                "BF16=full precision baseline, FP8=8-bit float (H100/MI300X), "
                "W8A8=INT8 weights+activations, W8A16=INT8 weights only, "
                "W4A16=INT4 weights only (AWQ). "
                "Auto-set by suites/suite_C/suite.py — do not set manually."
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
        parser.add_argument("--model-note", default=None, dest="model_note",
            help="Transparency note for this model.")
        parser.add_argument("--model-name", default=None, dest="model_name",
            help="Actual model name override.")
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

        args, _ = parser.parse_known_args()

        # Suite-specific scenario validation happens in main() after suite is loaded.
        # See _validate_scenario_for_suite().

        # Store suite_id so helpers (e.g. _resolve_tensor_parallel_size) can
        # access it without runners needing to pass it explicitly.
        self._suite_id = args.suite
        self._runner_config = self._load_runner_config(args.suite)

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
            # For Suite C subprocesses, --precision is set — use precision_model_map
            # to get the actual checkpoint model_id for display and metadata.
            _precision_arg = getattr(args, "precision", None)
            _precision_model_map = suite.get("precision_model_map", {})
            _fmt_entry = _precision_model_map.get((_precision_arg or "").upper(), {})
            model_id = (
                _fmt_entry.get("model_id")
                or suite.get("model_id", "unknown")
            )
            effective_model_path = self._resolve_model_path(
                model_id, getattr(args, "model_path", None)
            )
            if getattr(args, "model_note", None):
                self._model_note_override = args.model_note
            if getattr(args, "model_name", None):
                self._model_name_override = args.model_name
            _par    = getattr(self, "_parallelism", {})
            tp_size = _par.get("tensor_parallel_size", 1)
            pp_size = _par.get("pipeline_parallel_size", 1)
            ep_size = _par.get("expert_parallel_size", 1)
            dp_size = _par.get("data_parallel_size", 1)

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

            # Inject dtype_override and engine_kwargs from precision_model_map entry
            # so the runner can apply the correct quantization kernel and dtype.
            self._precision_dtype_override  = _fmt_entry.get("dtype_override")
            self._precision_engine_kwargs   = dict(_fmt_entry.get("engine_kwargs") or {})

            # If the precision_model_map entry declares a quantization engine_kwarg, the
            # runner will use dtype="auto", which lets vLLM default the compute dtype to
            # BF16 internally. On pre-Ampere hardware (V100/T4) that doesn't support BF16
            # this silently produces wrong results. If no dtype_override was already set
            # by the suite entry and the hardware doesn't support BF16, force float16.
            _entry_has_quantization = bool(
                (_fmt_entry.get("engine_kwargs") or {}).get("quantization")
            )
            if (not self._precision_dtype_override
                    and _entry_has_quantization
                    and "BF16" not in self._detect_supported_precisions(_acc_env_info)):
                self._precision_dtype_override = "float16"

            print(f"Loading {model_id} for accuracy check...")
            t_load = time.perf_counter()
            self._current_scenario = "accuracy"
            self._advance_dist_port()
            self.load_model(effective_model_path, {
                "tensor_parallel_size":   tp_size,
                "pipeline_parallel_size": pp_size,
                "expert_parallel_size":   ep_size,
                "data_parallel_size":     dp_size,
                "max_tokens":             suite.get("output_tokens_max", 512),
                "max_model_len":          suite.get("max_model_len"),
                "use_async":              False,
            })
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
        # For Suite C subprocesses, --precision is set and precision_model_map holds
        # the actual checkpoint being loaded. Use it for the display label so the log
        # doesn't show "Loading meta-llama/Llama-3.1-8B-Instruct..." when loading FP8.
        _precision_arg = getattr(args, "precision", None)
        _precision_model_map = suite.get("precision_model_map", {})
        _fmt_entry = _precision_model_map.get((_precision_arg or "").upper(), {})
        model_id = (
            _fmt_entry.get("model_id")
            or suite.get("model_id", "unknown")
        )
        effective_model_path = self._resolve_model_path(
            model_id, getattr(args, "model_path", None)
        )
        if getattr(args, "model_note", None):
            self._model_note_override = args.model_note
        if getattr(args, "model_name", None):
            self._model_name_override = args.model_name

        # Read env_info.json from task directory.
        # For standalone runs it's in output_dir; for --scenario all it's in the parent.
        # For deeply nested subprocess runs it may be two levels up — search up the tree.
        env_info = {}
        for _candidate in [output_dir, output_dir.parent, output_dir.parent.parent]:
            _p = _candidate / "env_info.json"
            if _p.exists():
                with open(_p) as f:
                    env_info = json.load(f)
                break

        # Load model
        _par    = getattr(self, "_parallelism", {})
        tp_size = _par.get("tensor_parallel_size", 1)
        pp_size = _par.get("pipeline_parallel_size", 1)
        ep_size = _par.get("expert_parallel_size", 1)
        dp_size = _par.get("data_parallel_size", 1)

        print(f"Loading {model_id}...")
        t_load_start = time.perf_counter()
        self._current_scenario = args.scenario
        self._advance_dist_port()

        # Resolve precision — handles BF16→FP16 fallback for older hardware.
        # Explicit --precision (e.g. set by a suite subprocess) takes priority.
        if getattr(args, "precision", None):
            effective_precision = args.precision.upper()
        else:
            effective_precision = self._resolve_precision(suite, env_info)
        self._effective_precision = effective_precision

        # Inject dtype_override and engine_kwargs from precision_model_map entry
        # so the runner can apply the correct quantization kernel and dtype.
        self._precision_dtype_override  = _fmt_entry.get("dtype_override")
        self._precision_engine_kwargs   = dict(_fmt_entry.get("engine_kwargs") or {})

        # If the precision_model_map entry declares a quantization engine_kwarg, the
        # runner will use dtype="auto", which lets vLLM default the compute dtype to
        # BF16 internally. On pre-Ampere hardware (V100/T4) that doesn't support BF16
        # this silently produces wrong results. If no dtype_override was already set
        # by the suite entry and the hardware doesn't support BF16, force float16.
        _entry_has_quantization = bool(
            (_fmt_entry.get("engine_kwargs") or {}).get("quantization")
        )
        if (not self._precision_dtype_override
                and _entry_has_quantization
                and "BF16" not in self._detect_supported_precisions(env_info)):
            self._precision_dtype_override = "float16"

        self.load_model(effective_model_path, {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": pp_size,
            "expert_parallel_size":   ep_size,
            "data_parallel_size":     dp_size,
            "max_tokens":             suite.get("output_tokens_max", 512),
            "max_model_len":          suite.get("max_model_len"),
            "use_async":              args.scenario not in ("offline", "accuracy"),
        })
        model_load_seconds = round(time.perf_counter() - t_load_start, 1)
        print(f"Model loaded in {model_load_seconds}s")

        # Load requests and convert to InferenceRequest objects.
        # InferenceRequest carries prompt, request_id, input_tokens and optional
        # per-request config. The prompt is raw (unformatted) here — each runner's
        # inference_fn_offline/streaming calls self.format_prompt() internally.
        inference_requests = []
        if args.scenario != "training":
            requests_path = self._resolve_requests_path(suite)
            with open(requests_path) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        inference_requests.append(InferenceRequest(
                            prompt       = r["prompt"],
                            request_id   = r.get("request_id", i),
                            input_tokens = r.get("input_tokens", 0),
                        ))

        # chip_count for throughput-per-chip calculation.
        # TP × PP = total chips. EP is within the TP group in current frameworks
        # (EP ≤ TP), so it does not multiply. DP = 1 for inference benchmarks.
        chip_count = getattr(self, "_chip_count", 1)
        loadgen = AccelMarkLoadGen(
            suite=suite,
            requests=inference_requests,
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
            def _sync_wrapper(request: InferenceRequest) -> InferenceResult:
                results = self.inference_fn_offline([request])
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

        platform_script = sys.argv[0]

        # When invoked via `python run.py --runner X`, sys.argv[0] is run.py.
        # Subprocesses need --runner to dispatch correctly through run.py.
        # Detect this case and store the runner_id for injection into cmds below.
        _runner_id_for_subprocess = None
        if Path(platform_script).name == "run.py":
            _runner_id_for_subprocess = self._compute_implementation_id()


        if run_accuracy:
            print(f"\n{'='*60}")
            if skip_gate:
                print(f"  Step 1: Accuracy Check (data only — non-blocking)")
            else:
                print(f"  Step 1: Accuracy Gate")
                print(f"  Must pass before benchmark runs.")
            print(f"{'='*60}\n")

            acc_subdir = base_dir / "accuracy"
            acc_subdir.mkdir(parents=True, exist_ok=True)

            # Run accuracy as a subprocess — identical to how benchmark scenarios
            # are run.  This is critical: a SIGKILL from the OOM killer (e.g. during
            # cudagraph capture on a large-memory GPU) terminates the child process
            # but leaves the parent alive to detect the non-zero exit code, write a
            # clear error message, and preserve accuracy/run.log.  Running in-process
            # means SIGKILL kills the whole program with no output at all.
            #
            # Output is streamed line-by-line through the parent so it appears on
            # the terminal in real time AND is captured to accuracy/run.log via the
            # parent's TeeWriter (set up by _setup_logging above).  Using
            # subprocess.run() with inherited fds would bypass the TeeWriter entirely
            # since the child writes directly to the OS fd, not through sys.stdout.
            acc_cmd = [sys.executable, platform_script]
            if _runner_id_for_subprocess:
                acc_cmd += ["--runner", _runner_id_for_subprocess]
            acc_cmd += [
                "--suite",      args.suite,
                "--scenario",   "accuracy",
                "--output-dir", str(base_dir),
                "--tier",       getattr(args, "tier", "community"),
                "--skip-accuracy-gate",
            ]
            if getattr(args, "model_path", None):
                acc_cmd += ["--model-path", args.model_path]
            if getattr(args, "precision", None):
                acc_cmd += ["--precision", args.precision]
            acc_cmd += self.get_extra_subprocess_args(args)

            print(f"  Command: {' '.join(acc_cmd)}\n")

            acc_ok = self._run_subprocess(acc_cmd, "accuracy")
            acc_json_path = acc_subdir / "accuracy.json"

            if not acc_ok or not acc_json_path.exists():
                results_summary.append(("accuracy", "FAILED: subprocess error", str(acc_subdir)))
                if not skip_gate:
                    print("  Aborting benchmark. Use --skip-accuracy-gate to override.")
                    return
                else:
                    print("  --skip-accuracy-gate set -- continuing anyway.\n")
                    acc_result = None
                    acc_result = None
            else:
                # Subprocess succeeded — read accuracy.json written by the child
                with open(acc_json_path) as f:
                    acc_result = json.load(f)

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
                    if not skip_gate:
                        return
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

# ── Step 2: Benchmark scenarios — each as a subprocess ───────────────
        # Each scenario runs in a fresh process to guarantee a clean CUDA
        # context. This avoids vLLM distributed re-initialization hangs when
        # TP > 1 (e.g. Suite B) where spawned workers inherit stale IPC state
        # from the previous scenario's process group.

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

            cmd = [sys.executable, platform_script]
            if _runner_id_for_subprocess:
                cmd += ["--runner", _runner_id_for_subprocess]
            cmd += [
                "--suite",      args.suite,
                "--scenario",   scenario,
                "--output-dir", str(base_dir),
                "--tier",       getattr(args, "tier", "community"),
                "--skip-accuracy-gate",
            ]
            if getattr(args, "model_path", None):
                cmd += ["--model-path", args.model_path]
            if getattr(args, "precision", None):
                cmd += ["--precision", args.precision]
            # All runner-specific flags (parallelism, enforce_eager, etc.) are forwarded
            # here. Runners must override get_extra_subprocess_args() to add their flags.
            # Note: self._runner_config is NOT forwarded — each subprocess reads the config
            # yaml file independently on startup, which is the correct behavior.
            cmd += self.get_extra_subprocess_args(args)

            print(f"  Command: {' '.join(cmd)}\n")

            ok = self._run_subprocess(cmd, scenario)
            if ok:
                results_summary.append((scenario, "SUCCESS", str(scenario_dir)))
                print(f"\n  {scenario} completed")
            else:
                results_summary.append((scenario, "FAILED: subprocess error", str(scenario_dir)))

            print("  Waiting 10s before next scenario...")
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

        # Always sum per-scenario elapsed times rather than using the orchestrator
        # wall-clock (total_elapsed_minutes). Summation is more accurate: it excludes
        # 10s sleep gaps between scenarios, orchestrator overhead, and model load time
        # (already captured in model_load_seconds). It also remains correct for
        # incremental runs where scenarios were added one at a time.
        total_elapsed = round(scenario_elapsed, 1) if scenario_elapsed else (total_elapsed_minutes or 0.0)

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
            "environment": base_result["environment"],
            "software": base_result["software"],
            "model": base_result["model"],
            "task": {
                "scenarios_run": scenarios_to_merge,
                "parallelism": base_result["task"]["parallelism"],
                "num_runs": suite.get("num_runs", 3),
                "extra_config": base_result["task"].get("extra_config"),
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

        # Build InferenceRequest objects with raw (unformatted) prompts.
        # format_prompt() is called by the runner's inference_fn_offline internally —
        # passing raw prompts here avoids double-formatting.
        accuracy_requests = []
        for i, q in enumerate(questions):
            raw = (
                f"Question: {q['question']}\n"
                f"A) {q['choices'][0]}\n"
                f"B) {q['choices'][1]}\n"
                f"C) {q['choices'][2]}\n"
                f"D) {q['choices'][3]}\n"
                f"Answer:"
            )
            accuracy_requests.append(InferenceRequest(
                prompt=raw,
                request_id=i,
            ))

        # Run through inference_fn_offline — same model, framework, precision
        t_start = time.perf_counter()
        try:
            results = self.inference_fn_offline(accuracy_requests)
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
        precision = getattr(self, "_effective_precision", "BF16")
        # For Suite C, the baseline lives under the quantized checkpoint's model_id
        # (e.g. RedHatAI/...-quantized.w8a8), not the suite-level model_id.
        _precision_model_map = suite.get("precision_model_map", {})
        model_id = (
            (_precision_model_map.get(precision) or {}).get("model_id")
            or getattr(self, "_resolved_model_id", None)
            or suite.get("model_id", "unknown")
        )
        baseline_score = self._load_accuracy_baseline_for_format(model_id, precision)
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

        # Build InferenceRequest objects with raw (unformatted) prompts.
        # format_prompt() is called by the runner's inference_fn_offline internally.
        accuracy_requests = []
        for i, q in enumerate(questions):
            raw = (
                f"Question: {q['question']}\n"
                f"A) {q['choices'][0]}\n"
                f"B) {q['choices'][1]}\n"
                f"C) {q['choices'][2]}\n"
                f"D) {q['choices'][3]}\n"
                f"Answer:"
            )
            accuracy_requests.append(InferenceRequest(
                prompt=raw,
                request_id=i,
            ))

        t_start = time.perf_counter()
        try:
            results = self.inference_fn_offline(accuracy_requests)
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
            # Total chips = TP × PP. Expert parallelism (EP) is within the TP
            # group in current frameworks (EP ≤ TP) so does not add chips.
            # Data parallelism = 1 for inference benchmarks.
            total_chips = getattr(self, "_chip_count", 1)
            # interconnect_intra_node: None for single-chip runs
            intra_node = None if total_chips <= 1 else (
                env_info.get("intra_node_interconnect") or a.get("interconnect_intra_node")
            )
            chip_info = {
                "name":                   a.get("name", "Unknown"),
                "vendor":                 a.get("vendor", suite.get("chip", {}).get("vendor", "Unknown")),
                "count":                  total_chips,
                "memory_gb":     a.get("memory_gb", None),
                "interconnect_intra_node": intra_node,
                "interconnect_inter_node": a.get("interconnect_inter_node", None),
            }

        _par    = getattr(self, "_parallelism", {})
        tp_size = _par.get("tensor_parallel_size", 1)
        pp_size = _par.get("pipeline_parallel_size", 1)
        ep_size = _par.get("expert_parallel_size", 1)
        dp_size = _par.get("data_parallel_size", 1)

        # For Suite C subprocesses, --precision is set and precision_model_map holds
        # the actual quantized checkpoint. Use it so each per-format result.json records
        # the real model_id/revision (e.g. RedHatAI/...-FP8), not the suite-level model_id.
        _result_precision = (
            getattr(self, "_effective_precision", None)
            or getattr(args, "precision", None)
        )
        _pm_entry = suite.get("precision_model_map", {}).get(
            (_result_precision or "").upper(), {}
        )
        _result_model_id = (
            _pm_entry.get("model_id")
            or suite.get("model_id", "unknown")
        )
        _result_model_revision = (
            _pm_entry.get("model_revision")
            or suite.get("model_revision", "unknown")
        )

        # For Suite C subprocesses, --precision is set and precision_model_map holds
        # the actual quantized checkpoint. Use it so each per-format result.json records
        # the real model_id/revision (e.g. RedHatAI/...-FP8), not the suite-level model_id.
        _result_precision = (
            getattr(self, "_effective_precision", None)
            or getattr(args, "precision", None)
        )
        _pm_entry = suite.get("precision_model_map", {}).get(
            (_result_precision or "").upper(), {}
        )
        _result_model_id = (
            _pm_entry.get("model_id")
            or suite.get("model_id", "unknown")
        )
        _result_model_revision = (
            _pm_entry.get("model_revision")
            or suite.get("model_revision", "unknown")
        )

        return {
            "schema_version": "1.0",
            "suite_id": suite["suite_id"],
            "implementation_id": self._compute_implementation_id(),
            "chip": chip_info,
            "environment": env_info,
            "software": {
                "framework": self._get_framework_name(),
                "framework_version": self._get_framework_version(),
                "driver_version": driver_version,
                "runtime_version": env_info.get("runtime_version", "unknown"),
                "os": env_info.get("os", "unknown"),
                "python_version": env_info.get("python_version", "unknown"),
            },
            "model": {
                "model_id":            _result_model_id,
                "model_revision":      _result_model_revision,
                "model_name":          getattr(self, "_model_name_override", None),
                "model_note":          getattr(self, "_model_note_override", None),
                "model_source":        getattr(self, "_model_source", "huggingface"),
                "architecture":        self._get_model_architecture(_result_model_id),
                "parameter_count_b":   self._estimate_param_count(_result_model_id),
                "precision":           getattr(self, "_effective_precision", None)
                                       or getattr(args, "precision", None)
                                       or suite.get("precision_required", "BF16"),
                "effective_dtype":     self.get_effective_dtype(),
                "quantization_method": self.get_quantization_method(),
                "model_format":        self.get_model_format(),
            },
            "task": {
                "scenario": args.scenario,
                "num_runs": suite.get("num_runs", 3),
                "warmup_runs": suite.get("warmup_runs", 1),
                "parallelism": {
                    "tensor_parallel_size":   tp_size,
                    "pipeline_parallel_size": pp_size,
                    "expert_parallel_size":   ep_size,
                    "data_parallel_size":     dp_size,
                },
                "extra_config": getattr(self, "_runner_config", None) or None,
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
                "time": datetime.now().strftime("%H:%M:%S"),
                "run_id":   self._compute_run_id(args, suite, env_info),
                "run_name": self._compute_run_name(args, suite, env_info),
                "flagged":  None,
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

    def _get_model_architecture(self, model_id: str) -> str:
        """
        Infer model architecture from model ID.

        Override in subclass for custom architectures. The default covers
        all common MoE models by name matching.

        Returns "dense" or "moe". May be extended in future versions.
        """
        _id = (model_id or "").lower()
        _MOE_KEYWORDS = [
            "mixtral", "deepseek-moe", "deepseek-v", "qwen.*moe",
            "olmoe", "jamba", "grok", "arctic", "-moe-",
        ]
        import re as _re
        if any(_re.search(kw, _id) for kw in _MOE_KEYWORDS):
            return "moe"
        return "dense"

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

        # ── Step 4: pick precision ────────────────────────────────────────────
        # Use precision_required if the hardware supports it — it is the target,
        # not a floor. Only fall back to another allowed precision when required
        # is unavailable, and always warn in that case regardless of direction
        # (downgrade BF16→FP16 or upgrade FP16→BF16 are both deviations from spec).
        if required in effective:
            resolved = required
        else:
            resolved = effective[0]
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


    def _load_runner_config(self, suite_id: str | None = None) -> dict:
        """
        Load runner-specific config from configs/runner_configs/runner_{id}.yaml.

        Called automatically at the end of base class parse_args() so that
        self._runner_config is populated before any runner subclass code runs.
        Runners must never call this method directly — they read self._runner_config
        which is already merged and ready.

        Merges global defaults with suite-specific overrides for suite_id only.
        The returned dict is flat — no 'suites' key, no other suite sections.
        Returns {} if config file does not exist (graceful degradation).

        engine_kwargs merge: if both global and suite-specific sections define
        engine_kwargs, the two dicts are merged with suite-specific keys winning.

        Merge priority (highest to lowest):
            suite-specific section  >  global defaults  >  {}
        """
        impl_id = self._compute_implementation_id()
        if not impl_id:
            return {}

        config_path = (
            _REPO_ROOT / "configs" / "runner_configs" / f"runner_{impl_id}.yaml"
        )
        if not config_path.exists():
            return {}

        try:
            import yaml
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: could not load runner config {config_path}: {e}")
            return {}

        # Global defaults — all top-level keys except 'suites'
        config = {k: v for k, v in raw.items() if k != "suites"}

        # Suite-specific overrides
        if suite_id:
            suite_overrides = (raw.get("suites") or {}).get(suite_id) or {}
            # Merge engine_kwargs specially: combine global + suite dicts,
            # suite-specific keys win on collision
            global_engine_kwargs = config.get("engine_kwargs") or {}
            suite_engine_kwargs  = suite_overrides.get("engine_kwargs") or {}
            # Apply all suite overrides (including engine_kwargs temporarily)
            config.update(suite_overrides)
            # Re-apply properly merged engine_kwargs
            if global_engine_kwargs or suite_engine_kwargs:
                config["engine_kwargs"] = {**global_engine_kwargs, **suite_engine_kwargs}

        return config

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
        """
        Resolve model path and set model transparency attributes.

        Resolution order:
          1. --model-path CLI arg → self._model_source = "local"
          2. configs/models_local.yaml → self._model_source = "local"
          3. Fall through → use model_id as HuggingFace ID directly
                          → self._model_source = "huggingface"

        Config metadata (model_name, note) is always read from
        models_local.yaml when the model_id has an entry — even when
        --model-path overrides the path. This ensures transparency
        fields are populated for all suites.

        Suite C subprocesses pass --model-note/--model-name explicitly
        (via suite.py) because the subprocess model_id is the base model
        while the note lives on the quantized checkpoint entry.
        """
        self._resolved_model_id = model_id
        self._model_source = "huggingface"

        # Always read config metadata for this model_id, even if we end
        # up using a CLI path override or HuggingFace fallthrough.
        config_path = _REPO_ROOT / "configs/models_local.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                entry = config.get("models", {}).get(model_id, {}) or {}
                model_name = entry.get("model_name")
                if model_name:
                    self._model_name_override = model_name
                note = entry.get("note")
                if note:
                    self._model_note_override = note
                # Use entry's local_path only when no CLI override
                if not cli_override:
                    local_path = entry.get("local_path")
                    if local_path and Path(local_path).exists():
                        self._model_source = "local"
                        print(f"Using local model path: {local_path}")
                        if model_name:
                            print(f"  (declared as: {model_name})")
                        return local_path
            except Exception:
                pass

        if cli_override:
            self._model_source = "local"
            print(f"Using model path from --model-path: {cli_override}")
            return cli_override

        print(f"Using HuggingFace model: {model_id}")
        return model_id

    def _resolve_requests_path(self, suite: dict) -> Path:
        """
        Resolve the requests.jsonl path for a suite.

        Resolution order:
          1. suite["dataset"] key → datasets/{dataset}/requests.jsonl
          2. Legacy: suites/{suite_id}/requests.jsonl (backward compatible)

        Datasets are shared immutable collections in the datasets/ folder.
        Suites reference them by name: "dataset": "sharegpt_standard_v1".
        If not found at either location, raises FileNotFoundError with a
        helpful message.
        """
        suite_id = suite.get("suite_id", "")
        dataset  = suite.get("dataset")

        if dataset:
            dataset_path = _REPO_ROOT / "datasets" / dataset / "requests.jsonl"
            if dataset_path.exists():
                return dataset_path
            raise FileNotFoundError(
                f"Dataset '{dataset}' not found at {dataset_path}.\n"
                f"Check 'dataset' field in suites/{suite_id}/suite.json.\n"
                f"Available datasets: "
                + ", ".join(
                    p.name for p in (_REPO_ROOT / "datasets").iterdir()
                    if p.is_dir() and (p / "requests.jsonl").exists()
                )
            )

        # Legacy path — suite has its own requests.jsonl
        legacy_path = _REPO_ROOT / "suites" / suite_id / "requests.jsonl"
        if legacy_path.exists():
            return legacy_path

        raise FileNotFoundError(
            f"No requests.jsonl found for suite '{suite_id}'.\n"
            f"Either add 'dataset' key to suite.json or create "
            f"suites/{suite_id}/requests.jsonl."
        )

    def _generate_output_dir(self, args, env_info: dict) -> str:
        """
        Generate the output directory path using run_name (deterministic hash-based name).
        Warns if runner has no valid implementation_id.
        """
        if self._compute_implementation_id() is None:
            print(
                "Warning: runner has no valid implementation_id. "
                "Each runner should have a content-hash ID (see runners/hash_runner.py). "
                "run_name will use 'unknown' as runner_id."
            )

        suite_path = _REPO_ROOT / "suites" / args.suite / "suite.json"
        try:
            with open(suite_path) as f:
                suite = json.load(f)
        except Exception:
            suite = {"suite_id": args.suite}

        run_name = self._compute_run_name(args, suite, env_info)
        tier     = getattr(args, "tier", "community")
        return str(Path("results") / tier / run_name)

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