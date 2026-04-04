"""
AccelMark — NVIDIA vLLM benchmark script.

Implements BenchmarkRunner for vLLM (sync LLM + AsyncLLMEngine).
All orchestration logic lives in runners/benchmark_runner.py.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
from vllm import LLM, AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult



# Suppress per-request vLLM logs by default
import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)


class VLLMRunner(BenchmarkRunner):
    """AccelMark benchmark runner using vLLM on NVIDIA GPUs."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    # vLLM on NVIDIA supports all precisions — hardware detection in BenchmarkRunner
    # will automatically restrict to FP16 on V100/T4
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]

    def __init__(self):
        self.llm: LLM = None
        self.engine: AsyncLLMEngine = None
        self.tokenizer: AutoTokenizer = None
        self.sampling_params: SamplingParams = None
        self._loop: asyncio.AbstractEventLoop = None

    def _get_chip_count(self) -> int:
        """Return the number of available CUDA GPUs."""
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "vLLM"

    def _get_framework_version(self) -> str:
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """Load model — sync LLM for offline/accuracy, async engine for streaming."""
        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism["pipeline_parallel_size"]
        assert pp_size <= 1, "Pipeline parallelism is not supported in VLLMRunner"

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg             = getattr(self, "_runner_config", {})
        max_num_seqs    = cfg.get("max_num_seqs", 512)
        gpu_memory_util = cfg.get("gpu_memory_utilization", 0.90)
        extra_kwargs    = dict(cfg.get("engine_kwargs") or {})

        # ── Filter engine_kwargs to only fields this vLLM version accepts ─────
        # Avoids TypeError when the runner config YAML references a field that
        # doesn't exist in the installed vLLM version (EngineArgs is a strict
        # dataclass — unknown keyword arguments raise TypeError immediately).
        try:
            import dataclasses
            from vllm.engine.arg_utils import EngineArgs as _EngineArgs
            _valid = {f.name for f in dataclasses.fields(_EngineArgs)}
            _dropped = {k: v for k, v in extra_kwargs.items() if k not in _valid}
            if _dropped:
                print(f"  Warning: engine_kwargs keys not supported by this "
                      f"vLLM version and will be ignored: {list(_dropped)}")
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in _valid}
        except Exception:
            pass  # If introspection fails, pass kwargs as-is and let vLLM report the error

        # Use precision resolved by BenchmarkRunner._resolve_precision()
        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision           = getattr(self, "_precision", None) or effective_precision
        quantization        = None
        dtype               = "bfloat16"

        # Map precision format names to vLLM kwargs.
        # For pre-quantized checkpoints, use dtype="auto" and let vLLM detect
        # the quantization method from the checkpoint's config.json.
        if precision == "BF16":
            dtype = "bfloat16"
            self._quantization_method = None
        elif precision == "FP8":
            dtype = "auto"
            self._quantization_method = "fp8"
        elif precision == "W8A8":
            dtype = "auto"
            self._quantization_method = "w8a8"
        elif precision == "W8A16":
            dtype = "auto"
            self._quantization_method = "w8a16"
        elif precision == "W4A16":
            dtype = "auto"
            self._quantization_method = "awq"
        elif precision == "FP16":
            dtype = "float16"
            self._quantization_method = None
        elif precision == "FP32":
            dtype = "float32"
            self._quantization_method = None
        else:
            dtype = "auto"
            self._quantization_method = None

        print(f"Loading model: precision={precision}, dtype={dtype}"
              + (f", quantization_method={self._quantization_method}"
                 if self._quantization_method else ""))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if not use_async:
            llm_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_util,
                **extra_kwargs,
            )
            if quantization:
                llm_kwargs["quantization"] = quantization
            if max_model_len:
                llm_kwargs["max_model_len"] = max_model_len
            self.llm = LLM(**llm_kwargs)
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            engine_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_util,
                # engine_kwargs values override named fields above if the same key appears in both.
                # This is intentional — engine_kwargs is the power-user escape hatch.
                **extra_kwargs,
            )
            if max_model_len:
                engine_kwargs["max_model_len"] = max_model_len
            engine_args = AsyncEngineArgs(**engine_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_effective_dtype(self) -> Optional[str]:
        """
        Report the actual compute dtype vLLM used after model loading.

        vLLM exposes the resolved dtype via model_config after initialization.
        This captures cases like FP8 weights on A100 computing in BF16.
        """
        try:
            if self.llm is not None:
                # Sync LLM path
                dtype = self.llm.llm_engine.model_config.dtype
                return str(dtype).replace("torch.", "")
            elif self.engine is not None:
                # Async engine path
                dtype = self.engine.engine.model_config.dtype
                return str(dtype).replace("torch.", "")
        except Exception:
            pass
        # Fall back to declared dtype if introspection fails
        return getattr(self, "_effective_dtype", None)

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Send all requests to vLLM at once. vLLM handles internal batching.

        total_time_ms in each returned InferenceResult is set to the wall-clock
        elapsed time of the entire batch — NOT an individual per-request latency.
        vLLM's sync LLM.generate() blocks until all requests finish, so there is
        no per-request completion timestamp available. All results share the same
        total_time_ms value, which is the correct denominator for throughput:
            throughput = total_tokens / (elapsed_ms / 1000)
        """
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start = time.perf_counter()
        outputs = self.llm.generate(formatted, self.sampling_params)
        elapsed = time.perf_counter() - t_start

        # Store output text for _run_accuracy_integrated()
        self._last_accuracy_outputs = [o.outputs[0].text for o in outputs]

        results = []
        for output in outputs:
            results.append(InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed * 1000,
                output_tokens=len(output.outputs[0].token_ids),
                input_tokens=len(output.prompt_token_ids),
                success=True,
                output_text=output.outputs[0].text,
            ))
        return results

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """Stream a single request, measuring TTFT."""
        from vllm.utils import random_uuid

        formatted = self._format_prompt(request.prompt)
        request_id = random_uuid()
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0
        output_text = ""

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            if (
                first_token_time_ms is None
                and len(output.outputs[0].token_ids) > 0
            ):
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens = len(output.outputs[0].token_ids)
            output_text = output.outputs[0].text

        total_time_ms = (time.perf_counter() - t_start) * 1000
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """
        Async generator yielding decoded text deltas for the serve layer.

        Each yield is the delta text since the last output — new characters
        only, not the full accumulated string.

        vLLM's engine.generate() yields cumulative outputs, so we track the
        previous text length and slice off only the new portion each step.
        """
        from vllm.utils import random_uuid

        formatted   = self._format_prompt(request.prompt)
        request_id  = random_uuid()
        prev_length = 0

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            current_text = output.outputs[0].text
            delta = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> float:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """Release vLLM engines and distributed state."""
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None

        if self.engine is not None:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(self.engine.shutdown())
            except Exception:
                pass
            try:
                del self.engine
            except Exception:
                pass
            self.engine = None

        # Destroy vLLM's distributed state so the next engine initialisation
        # creates a fresh TCPStore server.  Must call destroy_model_parallel()
        # first to clear vLLM's cached group references; only then is it safe
        # to destroy the underlying torch process group.  Skipping this step
        # leaves torch.distributed.is_initialized()==True, which causes
        # init_distributed_environment() to skip creating the new TCPStore
        # server, so spawned worker processes can never connect (→ 600 s timeout).
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except Exception:
            # Fallback for older vLLM builds that lack cleanup_dist_env_and_memory
            try:
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel, destroy_distributed_environment,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass

        # Final guard: if torch.distributed is still initialized after the cleanup
        # attempts above, destroy the default process group here.  Without this,
        # vLLM's init_distributed_environment() skips TCPStore server creation on
        # the next LLM() init, so new worker processes can never join the barrier
        # (→ 1800 s Gloo timeout) because the main driver calls barrier() on the
        # stale old group while workers wait on a fresh one that never reaches quorum.
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

    def parse_args(self):
        """Add vLLM/NVIDIA-specific CLI flags. Base class pre-loads runner config."""
        args = super().parse_args()
        cfg = self._runner_config

        # ── Runner-specific CLI flags ─────────────────────────────────────────
        # Defined here (not in benchmark_runner) — vLLM/NVIDIA-specific concepts.
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            dest="tensor_parallel_size")
        parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                            dest="pipeline_parallel_size")
        parser.add_argument("--enforce-eager", action="store_true", default=False,
                            dest="enforce_eager")
        extra, _ = parser.parse_known_args()

        # Priority: CLI flag > yaml config > required_chips > auto-detected > default 1
        # Fully resolved by base class.
        tp_size, _tp_source = self._resolve_tensor_parallel_size(
            extra.tensor_parallel_size
        )

        pp_size = (extra.pipeline_parallel_size
                   if extra.pipeline_parallel_size is not None
                   else 1)
        # enforce_eager: CLI flag OR yaml setting (either activates it)
        self._enforce_eager = extra.enforce_eager or cfg.get("enforce_eager", False)

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")

        if not self.SUPPORTS_MULTI_CHIP and tp_size * pp_size > 1:
            print(f"Warning: {self.__class__.__name__} does not support multi-chip. "
                  f"Ignoring tensor_parallel_size={tp_size}, using 1.")
            tp_size = 1
            pp_size = 1

        # Report to base class — used by _compute_run_id(), _build_result_json(), etc.
        self._parallelism = {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": pp_size,
            "expert_parallel_size":   1,
            "data_parallel_size":     1,
        }
        self._chip_count = tp_size * pp_size
        self._precision  = getattr(args, "precision", None)
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        """Forward vLLM/NVIDIA-specific flags to subprocess invocations."""
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if self._parallelism.get("pipeline_parallel_size", 1) > 1:
            extra += ["--pipeline-parallel-size",
                      str(self._parallelism["pipeline_parallel_size"])]
        if self._enforce_eager:
            extra += ["--enforce-eager"]
        return extra


if __name__ == "__main__":
    VLLMRunner().main()