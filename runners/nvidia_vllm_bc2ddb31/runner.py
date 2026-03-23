"""
AccelMark — NVIDIA vLLM benchmark script.

Implements BenchmarkRunner for vLLM (sync LLM + AsyncLLMEngine).
All orchestration logic lives in runners/benchmark_runner.py.

Supports: Suite A, B, C, D, E on NVIDIA GPUs via vLLM.
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

from runners.benchmark_runner import BenchmarkRunner
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

    def _get_framework_name(self) -> str:
        return "vLLM"

    def _get_framework_version(self) -> str:
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        """Load model — sync LLM for offline, async engine for online/interactive."""
        max_tokens = suite.get("output_tokens_max", 512)
        max_model_len = suite.get("max_model_len", None)
        enforce_eager = getattr(self, "_enforce_eager", False)
        scenario = getattr(self, "_current_scenario", None)

        # Use precision resolved by BenchmarkRunner._resolve_precision()
        # Falls back to BF16 if not set
        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision           = getattr(self, "_precision", None) or effective_precision
        quantization        = None
        dtype               = "bfloat16"

        # Map Suite C precision format names to vLLM kwargs
        # For pre-quantized checkpoints (FP8, W8A8, W8A16, W4A16), use dtype="auto"
        # and let vLLM detect quantization from the checkpoint's config.json.
        # self._quantization_method records what was detected for result.json.
        if precision == "BF16":
            dtype = "bfloat16"
            self._quantization_method = None
        elif precision == "FP8":
            dtype = "auto"    # vLLM detects fp8 from checkpoint config
            self._quantization_method = "fp8"
        elif precision == "W8A8":
            dtype = "auto"    # compressed sparse — detected from config
            self._quantization_method = "w8a8"
        elif precision == "W8A16":
            dtype = "auto"    # weight-only int8 — detected from config
            self._quantization_method = "w8a16"
        elif precision == "W4A16":
            dtype = "auto"    # AWQ int4 — detected from config
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

        # Sync LLM for offline/accuracy scenarios
        if scenario in ("offline", "accuracy"):
            llm_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
                max_num_seqs=512,
            )
            if quantization:
                llm_kwargs["quantization"] = quantization
            if max_model_len:
                llm_kwargs["max_model_len"] = max_model_len
            self.llm = LLM(**llm_kwargs)
        else:
            # Async engine for online/interactive (streaming, TTFT measurement)
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            engine_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
            )
            if quantization:
                engine_kwargs["quantization"] = quantization
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

    def format_prompt(self, prompt: str) -> str:
        """Apply chat template if tokenizer has one."""
        if self.tokenizer and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _format_prompt(self, prompt: str) -> str:
        """Alias for format_prompt (kept for internal use)."""
        return self.format_prompt(prompt)

    def inference_fn_offline(self, prompts: list[str]) -> list[InferenceResult]:
        """Send all prompts to vLLM at once. vLLM handles internal batching.

        total_time_ms in each returned InferenceResult is set to the wall-clock
        elapsed time of the entire batch — NOT an individual per-request latency.
        vLLM's sync LLM.generate() blocks until all requests finish, so there is
        no per-request completion timestamp available. All results share the same
        total_time_ms value, which is the correct denominator for throughput:
            throughput = total_tokens / (elapsed_ms / 1000)
        """
        formatted = [self._format_prompt(p) for p in prompts]
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

    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        """Stream a single prompt, measuring TTFT."""
        from vllm.utils import random_uuid

        formatted = self._format_prompt(prompt)
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
            text=output_text,
        )

    async def inference_fn_token_stream(self, prompt: str):
        """
        Async generator that yields decoded text incrementally as vLLM produces it.

        Used by the serve layer for true progressive SSE streaming.
        Each yield is the *delta* text since the last output — i.e. new characters
        only, not the full accumulated string.

        vLLM's engine.generate() yields cumulative outputs, so we track the
        previous text length and slice off only the new portion each step.
        """
        from vllm.utils import random_uuid

        formatted   = self._format_prompt(prompt)
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

    def parse_args(self):
        """Add vLLM-specific args on top of base args."""
        args = super().parse_args()
        # enforce_eager is stored on self for use in load_model
        self._enforce_eager = getattr(args, "enforce_eager", False)
        self._precision = getattr(args, "precision", None)
        return args


if __name__ == "__main__":
    VLLMRunner().main()