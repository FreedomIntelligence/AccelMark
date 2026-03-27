"""
AccelMark — NVIDIA SGLang benchmark runner.

Implements BenchmarkRunner for SGLang on NVIDIA GPUs.
All orchestration logic lives in runners/benchmark_runner.py.

SGLang differences from vLLM:
  - Offline: uses sglang.Engine (sync, batched)
  - Online/interactive: uses sglang.AsyncEngine with async generator streaming
  - Streaming output is cumulative (same as vLLM) — delta sliced by prev_length
  - Quantization loaded via engine_kwargs["quantization"] or dtype="fp8"
  - Memory query uses torch.cuda.max_memory_allocated (same as vLLM)

Supports: Suite A, B, C, D, E on NVIDIA GPUs via SGLang.
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
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult

# Suppress per-request SGLang logs
import logging
logging.getLogger("sglang").setLevel(logging.WARNING)


class SGLangRunner(BenchmarkRunner):
    """AccelMark benchmark runner using SGLang on NVIDIA GPUs."""

    SUPPORTS_STREAMING  = True
    SUPPORTS_BATCHING   = True
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True

    # SGLang on NVIDIA supports all standard precisions.
    # Hardware detection in BenchmarkRunner will automatically restrict to
    # FP16 on older chips (V100, T4) that don't support BF16.
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # SGLang supports the same quantization formats as vLLM via pre-quantized
    # checkpoints. FP8 requires H100 or newer (native FP8 tensor cores).
    # On A100, FP8 weights compute in BF16 — list it here; the leaderboard
    # will show effective_dtype to clarify.
    SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]

    def __init__(self):
        self.engine        = None   # sglang.Engine (offline/accuracy)
        self.async_engine  = None   # sglang.AsyncEngine (online/interactive)
        self.tokenizer     = None
        self._loop: asyncio.AbstractEventLoop = None
        self._sampling_params: dict = {}

    def _get_framework_name(self) -> str:
        return "SGLang"

    def _get_framework_version(self) -> str:
        try:
            import sglang
            return sglang.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, suite: dict, parallelism: dict) -> None:
        """
        Load model — sync Engine for offline/accuracy, AsyncEngine for streaming.

        SGLang Engine and AsyncEngine share the same constructor kwargs.
        The engine choice is made based on the current scenario so that
        the correct internal scheduler is used for each workload type.
        """
        tp_size       = parallelism["tensor_parallel_size"]
        max_tokens    = suite.get("output_tokens_max", 512)
        max_model_len = suite.get("max_model_len", None)
        enforce_eager = getattr(self, "_enforce_eager", False)
        scenario      = getattr(self, "_current_scenario", None)

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()

        # Map AccelMark precision names to SGLang dtype / quantization kwargs.
        # Pre-quantized checkpoints (FP8, W8A8, W8A16, W4A16) use dtype="auto"
        # so SGLang reads quantization config directly from the checkpoint.
        dtype         = "bfloat16"
        quantization  = None

        if effective_precision == "BF16":
            dtype = "bfloat16"
            self._quantization_method = None
        elif effective_precision == "FP16":
            dtype = "float16"
            self._quantization_method = None
        elif effective_precision == "FP32":
            dtype = "float32"
            self._quantization_method = None
        elif effective_precision == "FP8":
            dtype = "auto"
            self._quantization_method = "fp8"
        elif effective_precision == "W8A8":
            dtype = "auto"
            self._quantization_method = "w8a8"
        elif effective_precision == "W8A16":
            dtype = "auto"
            self._quantization_method = "w8a16"
        elif effective_precision == "W4A16":
            # AWQ int4 weight-only quantization
            dtype = "auto"
            self._quantization_method = "awq"
        else:
            dtype = "auto"
            self._quantization_method = None

        print(
            f"Loading model: precision={effective_precision}, dtype={dtype}"
            + (f", quantization_method={self._quantization_method}"
               if self._quantization_method else "")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        # Shared sampling params passed to engine.generate()
        self._sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature":    0.0,
        }

        # Common engine kwargs
        engine_kwargs = dict(
            model_path=model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=False,
            disable_cuda_graph=enforce_eager,
        )
        if quantization:
            engine_kwargs["quantization"] = quantization
        if max_model_len:
            engine_kwargs["context_length"] = max_model_len

        import sglang as sgl

        if scenario in ("offline", "accuracy"):
            # Sync engine — blocks until all requests complete
            self.engine = sgl.Engine(**engine_kwargs)
        else:
            # Async engine — required for streaming TTFT measurement
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.async_engine = sgl.AsyncEngine(**engine_kwargs)

    def get_effective_dtype(self) -> Optional[str]:
        """
        Report the actual compute dtype SGLang resolved after model loading.
        SGLang exposes the resolved dtype via engine.server_args.dtype.
        """
        try:
            eng = self.engine or self.async_engine
            if eng is not None:
                return str(eng.server_args.dtype).replace("torch.", "")
        except Exception:
            pass
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

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """
        Synchronous batch inference via sglang.Engine.generate().

        SGLang Engine accepts a list of prompt strings and sampling params,
        returns results in the same order. total_time_ms is set to the
        wall-clock elapsed time of the entire batch — the correct denominator
        for throughput = total_tokens / elapsed.
        """
        formatted  = [self.format_prompt(r.prompt) for r in requests]
        t_start    = time.perf_counter()
        outputs    = self.engine.generate(
            prompts=formatted,
            sampling_params=self._sampling_params,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results = []
        for req, out in zip(requests, outputs):
            # SGLang output format: {"text": str, "meta_info": {...}}
            text         = out.get("text", "") if isinstance(out, dict) else str(out)
            output_tokens = len(self.tokenizer.encode(text)) if text else 0
            input_tokens  = req.input_tokens or 0
            results.append(InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed_ms,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
                success=True,
                output_text=text,
            ))
        return results

    async def inference_fn_streaming(
        self, request: InferenceRequest
    ) -> InferenceResult:
        """
        Async streaming inference via sglang.AsyncEngine for TTFT measurement.

        SGLang's async generator yields cumulative text (same as vLLM).
        We track prev_length and slice off deltas to count output tokens.
        first_token_time_ms is set on the first non-empty yield.
        """
        formatted           = self.format_prompt(request.prompt)
        t_start             = time.perf_counter()
        first_token_time_ms = None
        output_text         = ""
        prev_length         = 0

        async for chunk in self.async_engine.async_generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
        ):
            # chunk is {"text": cumulative_text, "meta_info": {...}, "finished": bool}
            current_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            delta        = current_text[prev_length:]

            if delta and first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000

            output_text  = current_text
            prev_length  = len(current_text)

        total_time_ms  = (time.perf_counter() - t_start) * 1000
        output_tokens  = len(self.tokenizer.encode(output_text)) if output_text else 0

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
        Async generator yielding decoded text deltas for serve-layer SSE streaming.

        SGLang yields cumulative output — we slice off only the new delta each
        step so the serve layer receives incremental chunks, not repeated text.
        """
        formatted   = self.format_prompt(request.prompt)
        prev_length = 0

        async for chunk in self.async_engine.async_generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
        ):
            current_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            delta        = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """Shut down SGLang engines and release GPU memory."""
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            try:
                del self.engine
            except Exception:
                pass
            self.engine = None

        if self.async_engine is not None:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(self.async_engine.shutdown())
            except Exception:
                pass
            try:
                del self.async_engine
            except Exception:
                pass
            self.async_engine = None

        # SGLang uses torch.distributed internally for tensor parallelism.
        # Destroy process group so the next engine init creates a fresh one.
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def parse_args(self):
        """Add SGLang-specific CLI args on top of base args."""
        args = super().parse_args()
        self._enforce_eager = getattr(args, "enforce_eager", False)
        self._precision      = getattr(args, "precision", None)
        return args


if __name__ == "__main__":
    SGLangRunner().main()
